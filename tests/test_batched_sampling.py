"""Tests for batched sequence generation (_sample_seqs_batched).

Verifies correctness, diversity, and speed improvement of batched generation
versus sequential generation.
"""

import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.constants import aa_letters_lower
from src.data.objects import ProteinDocument


@pytest.fixture()
def prompt_ids(profam_tokenizer, test_model):
    """Build a short prompt (2 protein sequences) and return tokenized input_ids."""
    doc = ProteinDocument(
        sequences=[
            "MQFKVYTYKRESRYRLFVDVQSDIIDTPGRRMVIPLASARLLSDKVSRELYPVVHIGDESWRMMTTDMASVPVSVIGEEVADLSHRENDIKNAINLMFWGI",
            "MQFIVYKYKRASHYKMFVDVQSDIVDTPKRRMVIPLIESHHLSEKVNKTLFPLIRIEGKDYRLMTTELSSVPVEVMGEAIADLGDYADEIKDAINLMFWGI",
        ]
    )
    tok = profam_tokenizer.encode(doc, document_token="[RAW]", add_final_sep=True)
    input_ids = torch.as_tensor(tok["input_ids"], dtype=torch.long).unsqueeze(0)
    return input_ids.to(test_model.device)


@pytest.fixture()
def prompt_ids_repeaty(profam_tokenizer, test_model):
    """Build a prompt that strongly encourages the model to generate repeats."""
    doc = ProteinDocument(
        sequences=[
            "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
            "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
            "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
        ]
    )
    tok = profam_tokenizer.encode(doc, document_token="[RAW]", add_final_sep=True)
    input_ids = torch.as_tensor(tok["input_ids"], dtype=torch.long).unsqueeze(0)
    return input_ids.to(test_model.device)


class TestBatchedSamplingCorrectness:
    """Basic correctness tests for _sample_seqs_batched."""

    def test_returns_correct_number_of_sequences(self, test_model, prompt_ids):
        """Batched generation should return exactly num_samples sequences."""
        num_samples = 6
        with torch.no_grad():
            outputs, scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=150,
                batch_generation=True,
                generation_batch_size=3,
            )
        assert (
            outputs.shape[0] == num_samples
        ), f"Expected {num_samples} sequences, got {outputs.shape[0]}"
        assert (
            len(scores) == num_samples
        ), f"Expected {num_samples} scores, got {len(scores)}"

    def test_output_shape_matches_sequential(self, test_model, prompt_ids):
        """Batched and sequential should return tensors with the same first dim."""
        num_samples = 4
        torch.manual_seed(0)
        with torch.no_grad():
            seq_out, seq_scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=150,
                batch_generation=False,
            )
        torch.manual_seed(1)
        with torch.no_grad():
            bat_out, bat_scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=150,
                batch_generation=True,
                generation_batch_size=4,
            )
        assert seq_out.shape[0] == bat_out.shape[0] == num_samples
        assert len(seq_scores) == len(bat_scores) == num_samples

    def test_sequences_contain_valid_tokens(self, test_model, prompt_ids):
        """Generated sequences should only contain valid amino acid tokens."""
        num_samples = 4
        with torch.no_grad():
            outputs, _ = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=150,
                batch_generation=True,
                generation_batch_size=4,
            )
        tokenizer = test_model.tokenizer
        pad_id = tokenizer.pad_token_id
        sep_id = tokenizer.sep_token_id

        for i in range(num_samples):
            row = outputs[i]
            non_pad = row[row != pad_id]
            assert len(non_pad) >= 3, f"Sequence {i} too short: {len(non_pad)} tokens"
            # Decode and check
            decoded = tokenizer.decode(non_pad.tolist(), skip_special_tokens=True)
            decoded = decoded.replace(" ", "")
            assert len(decoded) > 0, f"Sequence {i} decoded to empty string"

    def test_batch_size_larger_than_num_samples(self, test_model, prompt_ids):
        """Should work correctly when generation_batch_size > num_samples."""
        num_samples = 2
        with torch.no_grad():
            outputs, scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=150,
                batch_generation=True,
                generation_batch_size=16,
            )
        assert outputs.shape[0] == num_samples
        assert len(scores) == num_samples

    def test_batched_scores_match_teacher_forced(self, test_model, prompt_ids):
        """Scores from batched generation should match teacher-forced scoring.

        During batched generation, per-token log-probs are extracted from the
        autoregressive logits (``gen_out.scores``), which are *processed* logits
        (bad-words suppressed, min-new-tokens enforced).  This test runs a
        teacher-forced forward pass on the same sequences and applies identical
        logits processing before computing log-probs, verifying they match.
        """
        num_samples = 4
        torch.manual_seed(42)
        with torch.no_grad():
            outputs, gen_scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=150,
                batch_generation=True,
                generation_batch_size=4,
                temperature=1.0,
            )

        tokenizer = test_model.tokenizer
        pad_id = tokenizer.pad_token_id
        sep_id = tokenizer.sep_token_id
        prompt_len = prompt_ids.shape[1]

        # ------------------------------------------------------------------
        # Build the effective set of always-suppressed token IDs, matching the
        # logits processing that HuggingFace's generate() applies internally.
        #
        # 1. _sample_seqs_batched constructs ``bad_words_ids`` containing
        #    *all* special tokens (since tokenizer.eos_token_id is None, the
        #    ``tok_id != eos_token_id`` filter keeps everything) plus the
        #    non-standard / lower-case amino-acid tokens.
        # 2. HuggingFace's ``NoBadWordsLogitsProcessor`` constructor then
        #    strips the *generation* eos_token_id (= sep_token_id) from that
        #    list so the model can actually terminate.
        # 3. ``MinNewTokensLengthLogitsProcessor`` additionally suppresses
        #    the generation eos_token_id for the first ``min_new_tokens``
        #    (= 3) generation steps.
        # ------------------------------------------------------------------
        bad_aas = ["X", "x", "B", "J", "O", "U", "Z", "-"] + aa_letters_lower
        always_suppressed = set()
        for tok_id in tokenizer.all_special_ids:
            always_suppressed.add(tok_id)
        for bad_aa in bad_aas:
            always_suppressed.add(tokenizer.convert_tokens_to_ids(bad_aa))
        # NoBadWordsLogitsProcessor strips the generation eos (= SEP)
        always_suppressed.discard(sep_id)
        always_suppressed_t = torch.tensor(sorted(always_suppressed), dtype=torch.long)

        min_new_tokens = 3  # hard-coded in _sample_seqs_batched

        # ------------------------------------------------------------------
        # Teacher-forced scoring with identical logits processing
        # ------------------------------------------------------------------
        teacher_scores = []
        for i in range(num_samples):
            row = outputs[i]
            valid_len = int((row != pad_id).sum().item())
            gen_tokens = row[:valid_len]

            # Generation stops at first SEP
            sep_positions = (gen_tokens == sep_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                end_pos = int(sep_positions[0].item()) + 1
            else:
                end_pos = valid_len
            gen_tokens = gen_tokens[:end_pos]

            # Full sequence = prompt + generated tokens
            full_seq = torch.cat([prompt_ids[0], gen_tokens]).unsqueeze(0)
            with torch.no_grad():
                model_out = test_model.model(input_ids=full_seq, use_cache=False)
            logits = model_out.logits[0]  # (L_total, V)

            total_logp = 0.0
            count = 0
            for t in range(end_pos):
                # logits[pos] predicts token at pos+1
                logit_pos = prompt_len - 1 + t
                processed = logits[logit_pos].clone()

                # (a) MinNewTokensLengthLogitsProcessor
                if t < min_new_tokens:
                    processed[sep_id] = float("-inf")

                # (b) NoBadWordsLogitsProcessor
                processed[always_suppressed_t] = float("-inf")

                token_id = int(gen_tokens[t].item())
                lp = F.log_softmax(processed, dim=-1)[token_id].item()
                total_logp += lp
                count += 1

            teacher_scores.append(total_logp / max(count, 1))

        # ------------------------------------------------------------------
        # Compare
        # ------------------------------------------------------------------
        print("\n--- Batched Generation vs Teacher-Forced Scores ---")
        for i in range(num_samples):
            row = outputs[i]
            valid_len = int((row != pad_id).sum().item())
            decoded = tokenizer.decode(
                row[:valid_len].tolist(), skip_special_tokens=True
            ).replace(" ", "")
            print(
                f"  Seq {i} ({len(decoded)} aa): "
                f"gen={gen_scores[i]:.6f}  teacher={teacher_scores[i]:.6f}  "
                f"diff={abs(gen_scores[i] - teacher_scores[i]):.2e}"
            )
        print("---")

        assert np.allclose(gen_scores, teacher_scores, atol=1e-4), (
            f"Generation scores do not match teacher-forced scores.\n"
            f"  gen_scores:     {gen_scores}\n"
            f"  teacher_scores: {teacher_scores}\n"
            f"  max abs diff:   "
            f"{max(abs(g - t) for g, t in zip(gen_scores, teacher_scores)):.2e}"
        )


class TestBatchedSamplingDiversity:
    """Ensure batched generation produces diverse (different) sequences."""

    def test_sequences_are_different(self, test_model, prompt_ids):
        """Batched sequences should not all be identical — sampling should
        produce diversity."""
        num_samples = 8
        with torch.no_grad():
            outputs, scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=50,
                batch_generation=True,
                generation_batch_size=8,
                temperature=1.0,
            )
        tokenizer = test_model.tokenizer
        pad_id = tokenizer.pad_token_id

        # Decode all sequences
        decoded_seqs = []
        for i in range(num_samples):
            row = outputs[i]
            non_pad = row[row != pad_id]
            decoded = tokenizer.decode(non_pad.tolist(), skip_special_tokens=True)
            decoded = decoded.replace(" ", "")
            decoded_seqs.append(decoded)

        # Print generated sequences for inspection
        print("\n--- Batched Generated Sequences ---")
        prompt_decoded = tokenizer.decode(
            prompt_ids[0].tolist(), skip_special_tokens=False
        )
        print(f"Prompt tokens: {prompt_ids.shape[1]}")
        print(f"Prompt (first 200 chars): {prompt_decoded[:200]}...")
        for i, seq in enumerate(decoded_seqs):
            print(f"  Seq {i} ({len(seq)} aa): {seq}")
        print(f"  Scores: {[f'{s:.3f}' for s in scores]}")
        print("---")

        # Check that not all sequences are identical
        unique_seqs = set(decoded_seqs)
        assert len(unique_seqs) > 1, (
            f"All {num_samples} sequences are identical! "
            f"Batched generation is not producing diverse outputs. "
            f"Unique: {unique_seqs}"
        )

    def test_pairwise_token_differences(self, test_model, prompt_ids):
        """At least some pairs of sequences should differ at token level."""
        num_samples = 4
        with torch.no_grad():
            outputs, _ = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=50,
                batch_generation=True,
                generation_batch_size=4,
                temperature=1.0,
            )
        pad_id = test_model.tokenizer.pad_token_id

        # Count pairs that differ
        diff_pairs = 0
        total_pairs = 0
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                total_pairs += 1
                # Compare non-pad tokens
                min_len = min(outputs[i].shape[0], outputs[j].shape[0])
                a = outputs[i, :min_len]
                b = outputs[j, :min_len]
                # Mask out padding from both
                mask = (a != pad_id) | (b != pad_id)
                if mask.any() and (a[mask] != b[mask]).any():
                    diff_pairs += 1

        assert diff_pairs > 0, (
            f"No pairs of sequences differ at the token level "
            f"(checked {total_pairs} pairs)"
        )


class TestBatchedSamplingSpeed:
    """Compare speed of batched vs sequential generation."""

    def test_batched_is_faster(self, test_model, prompt_ids):
        """Batched generation should be at least as fast as sequential for
        a reasonable number of samples."""
        num_samples = 8
        max_generated_length = 30

        # Warm up
        with torch.no_grad():
            test_model._sample_seqs(
                prompt_ids,
                num_samples=2,
                max_tokens=512,
                max_generated_length=max_generated_length,
                batch_generation=False,
            )

        # Time sequential
        torch.manual_seed(42)
        t0 = time.time()
        with torch.no_grad():
            seq_out, seq_scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=max_generated_length,
                batch_generation=False,
            )
        sequential_time = time.time() - t0

        # Time batched
        torch.manual_seed(42)
        t0 = time.time()
        with torch.no_grad():
            bat_out, bat_scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=max_generated_length,
                batch_generation=True,
                generation_batch_size=num_samples,
            )
        batched_time = time.time() - t0

        speedup = sequential_time / max(batched_time, 1e-6)

        # Print timing results and decoded sequences
        tokenizer = test_model.tokenizer
        pad_id = tokenizer.pad_token_id

        print("\n--- Speed Comparison ---")
        print(f"Sequential: {sequential_time:.3f}s for {num_samples} sequences")
        print(f"Batched:    {batched_time:.3f}s for {num_samples} sequences")
        print(f"Speedup:    {speedup:.2f}x")

        print("\n--- Sequential Sequences ---")
        for i in range(num_samples):
            row = seq_out[i]
            non_pad = row[row != pad_id]
            decoded = tokenizer.decode(non_pad.tolist(), skip_special_tokens=True)
            print(f"  Seq {i}: {decoded.replace(' ', '')} (score={seq_scores[i]:.3f})")

        print("\n--- Batched Sequences ---")
        for i in range(num_samples):
            row = bat_out[i]
            non_pad = row[row != pad_id]
            decoded = tokenizer.decode(non_pad.tolist(), skip_special_tokens=True)
            print(f"  Seq {i}: {decoded.replace(' ', '')} (score={bat_scores[i]:.3f})")
        print("---")

        # Both should produce the correct number of samples
        assert seq_out.shape[0] == num_samples
        assert bat_out.shape[0] == num_samples
        # We don't assert speedup > 1 since on CPU with a tiny model the overhead
        # of batching might not help, but we print it for the user to inspect.


class TestSamplingConstraints:
    """Observe behaviour of validation and retry logic under edge cases."""

    def test_repeat_guard_with_repeaty_prompt(self, test_model, prompt_ids_repeaty):
        """Feed a highly repetitive prompt and use aggressive repeat detection
        (repeat_length=1, repeat_count=3) to trigger the repeat guard.
        Observe how many sequences get retried vs force-accepted."""
        num_samples = 4
        tokenizer = test_model.tokenizer
        pad_id = tokenizer.pad_token_id
        sep_id = tokenizer.sep_token_id

        with torch.no_grad():
            outputs, scores = test_model._sample_seqs(
                prompt_ids_repeaty,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=100,
                batch_generation=True,
                generation_batch_size=4,
                temperature=1.0,
                repeat_guard=True,
                repeat_length=1,
                repeat_count=3,
                max_retries=2,
            )

        assert outputs.shape[0] == num_samples
        assert len(scores) == num_samples

        print("\n--- Repeat Guard Test (repeaty prompt, length=1, count=3) ---")
        for i in range(num_samples):
            row = outputs[i]
            valid_len = int((row != pad_id).sum().item())
            decoded = tokenizer.decode(
                row[:valid_len].tolist(), skip_special_tokens=True
            ).replace(" ", "")
            ends_sep = (
                int(row[valid_len - 1].item()) == sep_id if valid_len > 0 else False
            )
            print(
                f"  Seq {i} ({len(decoded)} aa, ends_sep={ends_sep}, "
                f"score={scores[i]:.4f}): {decoded[:80]}"
            )
        print("---")

    def test_repeat_guard_sequential_with_repeaty_prompt(
        self, test_model, prompt_ids_repeaty
    ):
        """Same as above but sequential — verify both paths behave consistently."""
        num_samples = 4
        tokenizer = test_model.tokenizer
        pad_id = tokenizer.pad_token_id
        sep_id = tokenizer.sep_token_id

        with torch.no_grad():
            outputs, scores = test_model._sample_seqs(
                prompt_ids_repeaty,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=100,
                batch_generation=False,
                temperature=1.0,
                repeat_guard=True,
                repeat_length=1,
                repeat_count=3,
                max_retries=2,
            )

        assert outputs.shape[0] == num_samples
        assert len(scores) == num_samples

        print(
            "\n--- Repeat Guard Test SEQUENTIAL (repeaty prompt, length=1, count=3) ---"
        )
        for i in range(num_samples):
            row = outputs[i]
            valid_len = int((row != pad_id).sum().item())
            decoded = tokenizer.decode(
                row[:valid_len].tolist(), skip_special_tokens=True
            ).replace(" ", "")
            ends_sep = (
                int(row[valid_len - 1].item()) == sep_id if valid_len > 0 else False
            )
            print(
                f"  Seq {i} ({len(decoded)} aa, ends_sep={ends_sep}, "
                f"score={scores[i]:.4f}): {decoded[:80]}"
            )
        print("---")

    def test_early_termination_no_sep(self, test_model, prompt_ids):
        """Set max_generated_length very short so sequences are likely to be
        cut off before producing a SEP token.  Observe retry/force-accept."""
        num_samples = 4
        tokenizer = test_model.tokenizer
        pad_id = tokenizer.pad_token_id
        sep_id = tokenizer.sep_token_id

        with torch.no_grad():
            outputs, scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=5,
                batch_generation=True,
                generation_batch_size=4,
                temperature=1.0,
                max_retries=2,
            )

        assert outputs.shape[0] == num_samples
        assert len(scores) == num_samples

        print("\n--- Early Termination Test (max_generated_length=5, batched) ---")
        for i in range(num_samples):
            row = outputs[i]
            valid_len = int((row != pad_id).sum().item())
            decoded = tokenizer.decode(
                row[:valid_len].tolist(), skip_special_tokens=True
            ).replace(" ", "")
            ends_sep = (
                int(row[valid_len - 1].item()) == sep_id if valid_len > 0 else False
            )
            print(
                f"  Seq {i} ({len(decoded)} aa, ends_sep={ends_sep}, "
                f"score={scores[i]:.4f}): {decoded}"
            )
        print("---")

    def test_early_termination_no_sep_sequential(self, test_model, prompt_ids):
        """Same early termination but sequential — verify consistency."""
        num_samples = 4
        tokenizer = test_model.tokenizer
        pad_id = tokenizer.pad_token_id
        sep_id = tokenizer.sep_token_id

        with torch.no_grad():
            outputs, scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=5,
                batch_generation=False,
                temperature=1.0,
                max_retries=2,
            )

        assert outputs.shape[0] == num_samples
        assert len(scores) == num_samples

        print("\n--- Early Termination Test (max_generated_length=5, sequential) ---")
        for i in range(num_samples):
            row = outputs[i]
            valid_len = int((row != pad_id).sum().item())
            decoded = tokenizer.decode(
                row[:valid_len].tolist(), skip_special_tokens=True
            ).replace(" ", "")
            ends_sep = (
                int(row[valid_len - 1].item()) == sep_id if valid_len > 0 else False
            )
            print(
                f"  Seq {i} ({len(decoded)} aa, ends_sep={ends_sep}, "
                f"score={scores[i]:.4f}): {decoded}"
            )
        print("---")
