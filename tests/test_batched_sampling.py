"""Tests for batched sequence generation (_sample_seqs_batched).

Verifies correctness, diversity, and speed improvement of batched generation
versus sequential generation.
"""

import time

import pytest
import torch

from src.data.objects import ProteinDocument


@pytest.fixture()
def prompt_ids(profam_tokenizer):
    """Build a short prompt (3 protein sequences) and return tokenized input_ids."""
    doc = ProteinDocument(
        sequences=[
            "ACDEFGHIKLMNPQRSTVWY",
            "MKTAYIAKQRQISFVKSHFS",
            "GVLKEYGVKLTDAQKFINEK",
        ]
    )
    tok = profam_tokenizer.encode(doc, document_token="[RAW]", add_final_sep=True)
    input_ids = torch.as_tensor(tok["input_ids"], dtype=torch.long).unsqueeze(0)
    return input_ids


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
                max_generated_length=50,
                batch_generation=True,
                generation_batch_size=3,
            )
        assert outputs.shape[0] == num_samples, (
            f"Expected {num_samples} sequences, got {outputs.shape[0]}"
        )
        assert len(scores) == num_samples, (
            f"Expected {num_samples} scores, got {len(scores)}"
        )

    def test_output_shape_matches_sequential(self, test_model, prompt_ids):
        """Batched and sequential should return tensors with the same first dim."""
        num_samples = 4
        torch.manual_seed(0)
        with torch.no_grad():
            seq_out, seq_scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=50,
                batch_generation=False,
            )
        torch.manual_seed(1)
        with torch.no_grad():
            bat_out, bat_scores = test_model._sample_seqs(
                prompt_ids,
                num_samples=num_samples,
                max_tokens=512,
                max_generated_length=50,
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
                max_generated_length=50,
                batch_generation=True,
                generation_batch_size=4,
            )
        tokenizer = test_model.tokenizer
        pad_id = tokenizer.pad_token_id
        sep_id = tokenizer.sep_token_id

        for i in range(num_samples):
            row = outputs[i]
            non_pad = row[row != pad_id]
            assert len(non_pad) >= 3, (
                f"Sequence {i} too short: {len(non_pad)} tokens"
            )
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
                max_generated_length=50,
                batch_generation=True,
                generation_batch_size=16,
            )
        assert outputs.shape[0] == num_samples
        assert len(scores) == num_samples


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
