import copy
import math
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from lightning import LightningModule
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch import nn
from transformers import PreTrainedTokenizerFast, StoppingCriteriaList
from transformers.cache_utils import DynamicCache
from transformers.optimization import get_scheduler

from src.constants import BASEDIR, aa_letters, aa_letters_lower
from src.data.objects import StringObject
from src.data.tokenizers import ProFamTokenizer
from src.models import metrics
from src.models.utils import InputAwareDynamicCache, log_likelihood_from_outputs
from src.utils import RankedLogger
from src.utils.sampling_utils import RepeatStoppingCriteria, has_too_many_repeats

log = RankedLogger(__name__, rank_zero_only=True)


def calc_grad_norm(params):
    grad_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]
        ),
        2,
    )

    return grad_norm


def _aa_to_three_letter(aa: str) -> str:
    """Convert single-letter amino acid to three-letter code."""
    mapping = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU",
        "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
        "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
        "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
        "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
        "X": "UNK",
    }
    return mapping.get(aa.upper(), "UNK")


def load_checkpoint(checkpoint_dir, **kwargs):
    config_dir = os.path.join(BASEDIR, checkpoint_dir, ".hydra")
    cfg = OmegaConf.load(os.path.join(config_dir, "config.yaml"))
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    log.info(OmegaConf.to_yaml(cfg.model))
    # TODO: check callback config
    checkpoint_path = os.path.join(BASEDIR, checkpoint_dir, "checkpoints/last.ckpt")
    # weights_only=False required for PyTorch 2.6+ to load HuggingFace tokenizer objects
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    model = hydra.utils.instantiate(cfg.model, tokenizer=tokenizer)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


class BaseFamilyLitModule(LightningModule):
    def __init__(
        self,
        model,
        tokenizer: ProFamTokenizer,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        eps: float = 1e-5,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        num_decay_steps: Optional[int] = None,
        scoring_max_tokens: int = 32_000,
        use_kv_cache_for_scoring: bool = True,
        override_optimizer_on_load: bool = False,
        override_step_on_load: bool = False,
        ignore_index: int = -100,
        pass_res_pos_in_doc_as_position_ids: bool = True,
        # GRPO (Group Relative Policy Optimization) hyperparameters
        grpo_enabled: bool = False,
        grpo_beta: float = 0.05,  # KL penalty coefficient
        grpo_clip_ratio: float = 0.2,  # PPO-style clipping
        grpo_normalize_rewards: bool = True,  # Normalize DMS scores within group
        grpo_use_reference_model: bool = False,  # Use KL regularization to initial model
        grpo_reward_baseline: str = "mean",  # "mean", "min", or "none"
        grpo_max_tokens: int = 8_000,  # Max tokens per batch for GRPO (lower than scoring_max_tokens due to gradients)
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.save_hyperparameters(logger=False, ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_decay_steps = num_decay_steps
        self.scheduler_name = scheduler_name
        self.scoring_max_tokens = scoring_max_tokens
        self.override_optimizer_on_load = override_optimizer_on_load
        self.override_step_on_load = override_step_on_load
        self.ignore_index = ignore_index
        self.pass_res_pos_in_doc_as_position_ids = pass_res_pos_in_doc_as_position_ids
        self.use_kv_cache_for_scoring = use_kv_cache_for_scoring
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._train_dataset_sample_counts = defaultdict(int)

        # GRPO configuration
        self.grpo_enabled = grpo_enabled
        self.grpo_beta = grpo_beta
        self.grpo_clip_ratio = grpo_clip_ratio
        self.grpo_normalize_rewards = grpo_normalize_rewards
        self.grpo_use_reference_model = grpo_use_reference_model
        self.grpo_reward_baseline = grpo_reward_baseline
        self.grpo_max_tokens = (
            grpo_max_tokens  # Separate limit for GRPO (needs gradients)
        )

        # Reference model for KL regularization (initialized lazily if needed)
        self._reference_model = None

        # Frozen encoder model for GRPO: computes prompt KV cache without
        # gradients so that only the decoder (self.model) is updated by GRPO.
        # Set via ``init_encoder_decoder_grpo()`` from the pipeline.
        self._encoder_model = None

    def train(self, mode: bool = True):
        """Ensure the frozen GRPO reference model never leaves eval mode.

        Lightning/torch will call `.train()` / `.eval()` on the root module, which
        recursively toggles all submodules. If the reference model were toggled
        into train mode, dropout etc. would make the KL penalty noisy/unstable
        even under `torch.no_grad()`.
        """
        super().train(mode)
        if self._reference_model is not None:
            self._reference_model.eval()
        if self._encoder_model is not None:
            self._encoder_model.eval()
        return self

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        # TODO: verify that different model implementations interpret
        # past key values in same way wrt e.g. position ids.
        if not (input_ids[:, 0] == self.tokenizer.bos_token_id).all():
            raise ValueError("Documents must start with a bos token")
            # note that when sampling we don't end up here, rather we call:
            # BaseLitModule.model.generate()
            # similarly, when using score_seqs (eg. protein_gym) we go via:
            # BaseLitModule.model.forward()
            # in general we assume that if you call BaseLitModule.forward()
            # you are not using KV cache.

        if labels is not None:
            labels[labels == self.tokenizer.bos_token_id] = self.ignore_index

        position_ids = self.get_position_ids_for_model_forward(
            input_ids, past_key_values
        )

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            **kwargs,
        )

    def compute_res_pos_in_doc(self, input_ids):
        """Needs to start at 0 for compatibility with sequence packing:
        https://github.com/huggingface/transformers/blob/70b07d97cf2c5f61fff55700b65528a1b6845cd2/src/transformers/modeling_flash_attention_utils.py#L133
        """
        assert (
            input_ids.shape[0] == 1
        ), "Since we are typically packing sequences, we assume batch size is 1"
        counter = torch.arange(input_ids.shape[1], device=input_ids.device)
        document_indices = (
            torch.cumsum(input_ids[0] == self.tokenizer.bos_token_id, 0) - 1
        )
        assert (
            document_indices >= 0
        ).all(), "Negative document indices encountered: check that bos token is first token in each document"
        doc_starts = (
            torch.argwhere(input_ids[0] == self.tokenizer.bos_token_id)
        ).flatten()
        offsets = counter[doc_starts][document_indices]
        position_ids = (counter - offsets).unsqueeze(0)
        return position_ids

    def get_position_ids_for_model_forward(self, input_ids, past_key_values):
        position_ids = None
        if past_key_values is not None:
            assert (
                input_ids == self.tokenizer.bos_token_id
            ).sum() <= 1, "Sequence packing not supported with past_key_values"
            position_ids = None
        elif self.pass_res_pos_in_doc_as_position_ids:
            position_ids = self.compute_res_pos_in_doc(input_ids)
        return position_ids

    def on_fit_start(self):
        """Initialize reference model and HMM scorer at the start of training.

        This ensures the reference model is a copy of the pre-training model,
        not a lazy copy that would be created after some training has occurred.
        Also picks up the HMM reward scorer from the datamodule if available.
        """
        if self.grpo_enabled and self.grpo_use_reference_model and self.grpo_beta > 0:
            if self._reference_model is None:
                log.info(
                    "Initializing reference model for GRPO KL regularization at fit start"
                )
                self._reference_model = copy.deepcopy(self.model)
                for param in self._reference_model.parameters():
                    param.requires_grad = False
                # Keep reference deterministic (no dropout) and frozen.
                self._reference_model.eval()

    def on_train_batch_start(self, batch, batch_idx: int):
        self._t0 = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        # TODO: handle ddp.
        self._t1 = time.time()
        self.log(
            "train/batch_time",
            self._t1 - self._t0,
            on_step=True,
            prog_bar=True,
        )

    def on_before_optimizer_step(self, optimizer):
        # https://github.com/Lightning-AI/pytorch-lightning/issues/1462
        self.log(
            "train/grad_norm",
            calc_grad_norm(self.model.parameters()),
            on_step=True,
            prog_bar=True,
        )
        self.log("train/lr", optimizer.param_groups[0]["lr"])

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # uncomment for debugging ddp (train.py +experiment=ddp_test)
        # print(f"Rank: {self.trainer.global_rank}", batch["identifier"].text, flush=True)

        # Check if this is a GRPO batch (contains rewards or DMS_scores)
        if ("rewards" in batch or "DMS_scores" in batch) and self.grpo_enabled:
            return self._grpo_step_from_batch(batch)

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log_metrics(batch, outputs, "train", log_global=True)
        self.log(
            "train/n_seqs",
            (batch["input_ids"] == self.tokenizer.sep_token_id)
            .float()
            .sum(axis=1)
            .mean()
            .item(),
            on_step=True,
            prog_bar=True,
            on_epoch=False,
        )
        self.log(
            "train/accumulate_grad_batches",
            self.trainer.accumulate_grad_batches,
            on_step=True,
            on_epoch=False,
        )
        self.log_train_dataset_sample_counts(batch)
        return loss

    def _get_reference_model(self):
        """Return reference model for KL regularization.

        The reference model should be initialized in on_fit_start() to ensure
        it's a copy of the pre-training model, not a copy made after training started.
        """
        if self._reference_model is None and self.grpo_use_reference_model:
            log.warning(
                "Reference model was not initialized in on_fit_start(). "
                "This may indicate the model was called outside of fit() or GRPO was enabled late. "
                "Creating reference model now, but KL regularization may not work as intended."
            )
            self._reference_model = copy.deepcopy(self.model)
            for param in self._reference_model.parameters():
                param.requires_grad = False
            # Keep reference deterministic (no dropout) and frozen.
            self._reference_model.eval()
        return self._reference_model

    def init_encoder_decoder_grpo(self) -> None:
        """Initialise the frozen encoder model for encoder-decoder GRPO.

        Creates a frozen copy of ``self.model`` that will be used to compute
        the prompt KV cache during GRPO steps.  The trainable ``self.model``
        (decoder) then processes only the generated completions, so gradients
        do not flow through the prompt embeddings.
        """
        if self._encoder_model is not None:
            return  # already initialised
        log.info("Initializing frozen encoder model for encoder-decoder GRPO")
        self._encoder_model = copy.deepcopy(self.model)
        for param in self._encoder_model.parameters():
            param.requires_grad = False
        self._encoder_model.eval()

    def _compute_grpo_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantages from rewards using group-relative normalization.

        Args:
            rewards: Tensor of shape (group_size,) containing DMS scores

        Returns:
            advantages: Tensor of shape (group_size,) containing normalized advantages
        """
        if self.grpo_normalize_rewards:
            # Normalize rewards to have zero mean and unit variance within the group
            reward_mean = rewards.mean()
            reward_std = rewards.std() + 1e-8
            advantages = (rewards - reward_mean) / reward_std
        else:
            # Use raw rewards with baseline subtraction
            if self.grpo_reward_baseline == "mean":
                baseline = rewards.mean()
            elif self.grpo_reward_baseline == "min":
                baseline = rewards.min()
            else:  # "none"
                baseline = 0.0
            advantages = rewards - baseline

        return advantages

    def _compute_variant_log_likelihoods_for_grpo(
        self,
        input_ids: Optional[torch.Tensor],
        completion_ids: torch.Tensor,
        group_indices: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute log-likelihoods for variant sequences for GRPO training.

        Uses batched scoring via score_seqs with return_tensor=True to maintain
        gradients for the policy model.

        Args:
            input_ids: Context tokens of shape (1, L_context), or None for no context.
                       If None or empty, completions are scored without context
                       (start tokens are prepended automatically).
            completion_ids: Variant tokens of shape (1, N, L_completion)
            group_indices: Optional indices to select a subset of completions
            batch_size: Optional batch size for scoring. If None, uses a default
                       based on grpo_max_tokens and completion length.

        Returns:
            log_likelihoods: Tensor of shape (group_size,) containing mean log-likelihoods
        """
        # Select group of completions if indices provided
        if group_indices is not None:
            completion_ids = completion_ids[:, group_indices, :]

        # Determine batch size if not specified
        # Use grpo_max_tokens (not scoring_max_tokens) since we need gradients
        if batch_size is None:
            L = completion_ids.shape[-1]
            L_prompt = input_ids.shape[-1] if input_ids is not None else 0
            batch_size = max(self.grpo_max_tokens // (L + L_prompt), 1)

        # Use the batched scoring with return_tensor=True for gradient tracking
        log_likelihoods = self.score_seqs(
            input_ids=input_ids,
            completion_ids=completion_ids,
            use_cache=True,
            batch_size=batch_size,
            return_tensor=True,
        )

        return log_likelihoods

    def _compute_per_token_log_probs_for_grpo(
        self,
        input_ids: Optional[torch.Tensor],
        completion_ids: torch.Tensor,
        group_indices: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token log-probabilities for GRPO per-token ratio computation.

        Similar to ``_compute_variant_log_likelihoods_for_grpo`` but returns the
        full per-token log-prob tensor (with gradients) and a boolean validity mask
        instead of a single mean-per-sequence scalar.

        Args:
            input_ids: Context tokens of shape (1, L_context), or None for no context.
            completion_ids: Completion tokens of shape (1, N, L_completion).
            group_indices: Optional indices to select a subset of completions.
            batch_size: Optional batch size for scoring.

        Returns:
            log_probs: Tensor of shape (N, L_completion-1) with per-token log-probs
                       (gradients preserved for the current policy).
            mask: Bool tensor of shape (N, L_completion-1), True for valid (non-pad)
                  prediction positions.
        """
        if group_indices is not None:
            completion_ids = completion_ids[:, group_indices, :]

        N = completion_ids.shape[1]
        L = completion_ids.shape[-1]
        out_len = L - 1  # predicting token t+1 from logits at t

        if batch_size is None:
            L_prompt = input_ids.shape[-1] if input_ids is not None else 0
            batch_size = max(self.grpo_max_tokens // (L + L_prompt), 1)

        # Compute context KV cache once.
        # If a frozen encoder exists, use it (no gradients through prompt);
        # otherwise use the trainable model (original behaviour).
        has_context = input_ids is not None and input_ids.numel() > 0
        past_key_values = None
        if has_context:
            if self._encoder_model is not None:
                with torch.no_grad():
                    ctx_outputs = self._encoder_model(input_ids=input_ids, use_cache=True)
                past_key_values = ctx_outputs.past_key_values
            else:
                ctx_outputs = self.model(input_ids=input_ids, use_cache=True)
                past_key_values = ctx_outputs.past_key_values

        all_log_probs: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_ids = completion_ids[0, batch_start:batch_end, :]  # (bs, L)

            # Trim trailing padding for efficiency
            batch_ids_trimmed = self.trim_eval_batch(batch_ids)
            actual_bs = batch_ids_trimmed.shape[0]
            L_trimmed = batch_ids_trimmed.shape[1]

            if L_trimmed <= 1:
                # Need at least 2 tokens to get a log-prob
                all_log_probs.append(
                    torch.zeros(actual_bs, out_len, device=self.device)
                )
                all_masks.append(
                    torch.zeros(
                        actual_bs, out_len, dtype=torch.bool, device=self.device
                    )
                )
                continue

            if has_context:
                cache = InputAwareDynamicCache.from_legacy_cache(past_key_values)
                cache.batch_repeat_interleave(actual_bs)
                outputs = self.model(
                    input_ids=batch_ids_trimmed,
                    past_key_values=cache,
                    use_cache=False,
                )
            else:
                outputs = self.model(
                    input_ids=batch_ids_trimmed, use_cache=False
                )

            # Build labels (mask out padding)
            labels = torch.where(
                batch_ids_trimmed == self.tokenizer.pad_token_id,
                -100,
                batch_ids_trimmed.clone(),
            )

            # Per-token log-probs: log p(token_{t+1} | context, token_{0..t})
            log_prob = log_likelihood_from_outputs(outputs, labels, start_ix=0)
            # Shape: (actual_bs, L_trimmed - 1)

            shift_labels = labels[..., 1:].to(log_prob.device)
            mask = shift_labels != -100  # (actual_bs, L_trimmed - 1)

            # Pad back to out_len if the mini-batch was trimmed shorter
            pad_needed = out_len - (L_trimmed - 1)
            if pad_needed > 0:
                log_prob = F.pad(log_prob, (0, pad_needed), value=0.0)
                mask = F.pad(mask, (0, pad_needed), value=False)

            all_log_probs.append(log_prob)
            all_masks.append(mask)

        return torch.cat(all_log_probs, dim=0), torch.cat(all_masks, dim=0)

    def _grpo_step_from_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Lightning-style GRPO adapter: convert a reward-bearing batch into
        the arguments expected by :meth:`grpo_step_from_rewards` and invoke it.

        The batch is expected to contain:
            - ``input_ids``:     ``(1, L_context)`` prompt/context tokens.
            - ``completion_ids``: ``(1, N, L_completion)`` variant tokens.
              A leading SEP may optionally be present on each completion.
            - ``rewards`` or ``DMS_scores``: ``(1, N)`` or ``(N,)`` reward scores.

        Because the Lightning batch does not carry pre-computed "old" policy
        log-probs, we compute them here with :func:`torch.no_grad` — this is
        on-policy GRPO (old == new at step start), so the PPO clip is a
        no-op and the objective reduces to the advantage-weighted policy
        gradient.

        Returns the scalar loss for Lightning to backprop through, and
        logs the per-step metrics from ``grpo_step_from_rewards`` under
        the ``train/`` namespace.
        """
        input_ids = batch["input_ids"]
        completion_ids = batch["completion_ids"]  # (1, N, L_completion)

        raw_rewards = batch.get("rewards", batch.get("DMS_scores"))
        assert raw_rewards is not None, (
            "GRPO training requires 'rewards' or 'DMS_scores' in batch"
        )
        rewards = (
            raw_rewards[0].float() if raw_rewards.dim() > 1 else raw_rewards.float()
        )

        # (1, N, L_comp) → (N, L_comp) and strip a leading SEP if present
        # (grpo_step_from_rewards re-prepends its own SEP to each completion).
        generated_tokens = completion_ids[0]
        sep_id = self.tokenizer.sep_token_id
        if (
            generated_tokens.shape[1] > 0
            and int(generated_tokens[0, 0].item()) == sep_id
        ):
            generated_tokens = generated_tokens[:, 1:]

        # Build the completion-ids tensor that _compute_per_token_log_probs_for_grpo
        # expects — exactly the same shape the reward-agnostic core builds internally.
        gen_on_device = generated_tokens.to(self.device)
        sep_prefix = torch.full(
            (gen_on_device.shape[0], 1),
            sep_id,
            dtype=gen_on_device.dtype,
            device=self.device,
        )
        comp_for_lp = torch.cat([sep_prefix, gen_on_device], dim=1).unsqueeze(0)
        input_ids_dev = input_ids.to(self.device)
        if (
            input_ids_dev.shape[1] > 0
            and int(input_ids_dev[0, -1].item()) == sep_id
        ):
            input_ids_for_scoring = input_ids_dev[:, :-1]
        else:
            input_ids_for_scoring = input_ids_dev

        with torch.no_grad():
            old_per_token_lps, old_per_token_mask = (
                self._compute_per_token_log_probs_for_grpo(
                    input_ids=input_ids_for_scoring,
                    completion_ids=comp_for_lp,
                )
            )

        loss, metrics = self.grpo_step_from_rewards(
            input_ids=input_ids,
            generated_tokens=generated_tokens,
            old_per_token_lps=old_per_token_lps,
            old_per_token_mask=old_per_token_mask,
            rewards=rewards,
        )

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.log(
                    f"train/{k}",
                    float(v),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=True,
                )

        if hasattr(self, "log_train_dataset_sample_counts"):
            self.log_train_dataset_sample_counts(batch)

        return loss

    def grpo_step_from_rewards(
        self,
        input_ids: torch.Tensor,
        generated_tokens: torch.Tensor,
        old_per_token_lps: torch.Tensor,
        old_per_token_mask: torch.Tensor,
        rewards: torch.Tensor,
        clip_ratio: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generic GRPO gradient computation from pre-computed rewards.

        Takes pre-generated sequences (from ``_sample_seqs`` with
        ``return_per_token_log_probs=True``) and pre-computed reward scores,
        then computes the PPO-clipped GRPO loss with optional KL
        regularization.

        This is the reward-agnostic core of GRPO.  Callers are responsible
        for generating sequences, computing rewards, and calling
        ``loss.backward()`` / ``optimizer.step()`` on the returned loss.

        Args:
            input_ids: Context tokens of shape ``(1, L_context)``.
            generated_tokens: Token IDs from ``_sample_seqs``, shape ``(G, L_gen)``.
            old_per_token_lps: Per-token log-probs under the generating policy,
                shape ``(G, L_gen)``.
            old_per_token_mask: Validity mask, shape ``(G, L_gen)``.
            rewards: Pre-computed reward scores, shape ``(G,)``.  Higher is better.
            clip_ratio: PPO clipping epsilon.  Defaults to ``self.grpo_clip_ratio``.
            beta: KL penalty coefficient.  Defaults to ``self.grpo_beta``.

        Returns:
            total_loss: Scalar loss tensor (with gradients).
            metrics: Dict with ``grpo_loss``, ``kl_loss``, ``total_loss``,
                ``mean_reward``, ``max_reward``, ``min_reward``,
                ``mean_advantage``, ``clip_fraction``, ``mean_ratio``.
        """
        eps = clip_ratio if clip_ratio is not None else self.grpo_clip_ratio
        kl_beta = beta if beta is not None else self.grpo_beta
        sep_id = self.tokenizer.sep_token_id

        # 1. Compute advantages
        advantages = self._compute_grpo_advantages(rewards)

        # 2. Format completion IDs: strip trailing SEP from context,
        #    prepend SEP to each generated completion.
        input_ids_dev = input_ids.to(self.device)
        if input_ids_dev.shape[1] > 0 and int(input_ids_dev[0, -1].item()) == sep_id:
            input_ids_for_scoring = input_ids_dev[:, :-1]
        else:
            input_ids_for_scoring = input_ids_dev

        gen_on_device = generated_tokens.to(self.device)
        sep_prefix = torch.full(
            (gen_on_device.shape[0], 1), sep_id,
            dtype=gen_on_device.dtype, device=self.device,
        )
        completion_ids = torch.cat([sep_prefix, gen_on_device], dim=1)
        completion_ids = completion_ids.unsqueeze(0)  # (1, G, 1+L_gen)

        # 3. Compute per-token log π_θ under current policy (with gradients)
        new_per_token_lps, new_per_token_mask = (
            self._compute_per_token_log_probs_for_grpo(
                input_ids=input_ids_for_scoring,
                completion_ids=completion_ids,
            )
        )

        # 4. PPO-style clipped loss
        old_lps = old_per_token_lps.to(self.device).detach()
        valid_mask = old_per_token_mask.to(self.device) & new_per_token_mask

        per_token_log_ratio = new_per_token_lps - old_lps
        per_token_ratio = torch.exp(per_token_log_ratio)

        clipped_ratio = torch.clamp(per_token_ratio, 1.0 - eps, 1.0 + eps)

        adv = advantages.unsqueeze(1)  # (G, 1)
        surr1 = per_token_ratio * adv
        surr2 = clipped_ratio * adv
        per_token_obj = torch.min(surr1, surr2)

        num_valid = valid_mask.float().sum(dim=1).clamp(min=1)
        per_seq_obj = (per_token_obj * valid_mask.float()).sum(dim=1) / num_valid
        grpo_loss = -per_seq_obj.mean()

        # 5. Optional KL regularization to frozen reference model
        kl_loss = torch.tensor(0.0, device=grpo_loss.device)
        if self.grpo_use_reference_model and kl_beta > 0:
            ref_model = self._get_reference_model()
            if ref_model is not None:
                kl_loss = self._compute_token_level_kl_divergence(
                    ref_model=ref_model,
                    input_ids=input_ids_for_scoring,
                    completion_ids=completion_ids,
                    group_indices=list(range(completion_ids.shape[1])),
                )

        total_loss = grpo_loss + kl_beta * kl_loss

        # 6. Compute metrics (no gradients needed)
        with torch.no_grad():
            ratio_valid = per_token_ratio.detach()[valid_mask]
            clip_frac = (
                ((ratio_valid < 1.0 - eps) | (ratio_valid > 1.0 + eps))
                .float().mean().item()
            ) if ratio_valid.numel() > 0 else 0.0
            mean_ratio = ratio_valid.mean().item() if ratio_valid.numel() > 0 else 1.0

        metrics = {
            "grpo_loss": grpo_loss.item(),
            "kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "total_loss": total_loss.item(),
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "mean_advantage": advantages.mean().item(),
            "clip_fraction": clip_frac,
            "mean_ratio": mean_ratio,
        }

        return total_loss, metrics


    def _compute_token_level_kl_divergence(
        self,
        ref_model,
        input_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        group_indices: List[int],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute proper token-level KL divergence between policy and reference model.

        Computes D_KL(policy || reference) by comparing the full vocabulary distribution
        at each token position, not just the log-likelihood of the observed tokens.

        KL divergence at each position t:
            D_KL(π || π_ref) = Σ_v π(v|x_{<t}) * [log π(v|x_{<t}) - log π_ref(v|x_{<t})]

        This ensures we penalize any drift in the token distribution, even if the model
        assigns the same probability to the observed token but different probabilities
        to other tokens in the vocabulary.

        Args:
            ref_model: The frozen reference model
            input_ids: Context tokens of shape (1, L_context)
            completion_ids: Variant tokens of shape (1, N, L_completion)
            group_indices: Indices of variants to compute KL for
            batch_size: Number of variants to process at once. If None, uses default.

        Returns:
            kl_div: Scalar tensor containing mean KL divergence across all tokens and variants
        """
        # Safety: ensure reference model is frozen and deterministic even if the
        # parent LightningModule was put into train() (which toggles submodules).
        if ref_model is not None:
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

        # Select group of completions
        completion_ids = completion_ids[:, group_indices, :]
        N = completion_ids.shape[1]
        L = completion_ids.shape[-1]

        # Determine batch size if not specified
        # Use grpo_max_tokens (not scoring_max_tokens) since we need gradients
        if batch_size is None:
            L_prompt = (
                input_ids.shape[-1]
                if input_ids is not None and input_ids.numel() > 0
                else 0
            )
            batch_size = max(self.grpo_max_tokens // (L + L_prompt), 1)

        # Compute context KV cache once for both models
        has_context = input_ids is not None and input_ids.numel() > 0

        policy_past_key_values = None
        ref_past_key_values = None

        if has_context:
            with torch.no_grad():
                ref_context_outputs = ref_model(input_ids=input_ids, use_cache=True)
                ref_past_key_values = ref_context_outputs.past_key_values

            if self._encoder_model is not None:
                with torch.no_grad():
                    policy_context_outputs = self._encoder_model(
                        input_ids=input_ids, use_cache=True
                    )
            else:
                policy_context_outputs = self.model(
                    input_ids=input_ids, use_cache=True
                )
            policy_past_key_values = policy_context_outputs.past_key_values

        all_kl_divs = []

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)

            # Get batch of variant ids: (batch_size, L)
            batch_variant_ids = completion_ids[0, batch_start:batch_end, :]

            # Trim padding from batch (to shortest non-padded length)
            batch_variant_ids = self.trim_eval_batch(batch_variant_ids)
            actual_batch_size = batch_variant_ids.shape[0]

            if batch_variant_ids.shape[1] <= 1:
                # Need at least 2 tokens to compute KL (predict token 1 from token 0)
                all_kl_divs.append(
                    torch.zeros(actual_batch_size, device=batch_variant_ids.device)
                )
                continue

            # Get logits from policy model (with gradients for the KL penalty)
            if has_context:
                policy_cache = InputAwareDynamicCache.from_legacy_cache(
                    policy_past_key_values
                )
                policy_cache.batch_repeat_interleave(actual_batch_size)
                policy_outputs = self.model(
                    input_ids=batch_variant_ids,
                    past_key_values=policy_cache,
                    use_cache=False,
                )
            else:
                policy_outputs = self.model(
                    input_ids=batch_variant_ids, use_cache=False
                )

            # Get logits from reference model (no gradients)
            with torch.no_grad():
                if has_context:
                    ref_cache = InputAwareDynamicCache.from_legacy_cache(
                        ref_past_key_values
                    )
                    ref_cache.batch_repeat_interleave(actual_batch_size)
                    ref_outputs = ref_model(
                        input_ids=batch_variant_ids,
                        past_key_values=ref_cache,
                        use_cache=False,
                    )
                else:
                    ref_outputs = ref_model(
                        input_ids=batch_variant_ids, use_cache=False
                    )

            # Shift logits for next-token prediction: logits[t] predicts token[t+1]
            # Shape: (batch_size, L-1, vocab_size)
            policy_logits = policy_outputs.logits[:, :-1, :]
            ref_logits = ref_outputs.logits[:, :-1, :].to(policy_logits.device)

            # Convert to log probabilities in float32 for numerical stability under AMP.
            policy_log_probs = F.log_softmax(policy_logits.float(), dim=-1)
            ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)

            # Compute KL divergence: D_KL(policy || reference) per position.
            # Using kl_div with log_target=True avoids an explicit exp() tensor.
            # Shape: (batch_size, L-1)
            kl_per_token = F.kl_div(
                input=ref_log_probs,
                target=policy_log_probs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)

            # Create mask for valid (non-padding) prediction targets
            # Targets are batch_variant_ids[:, 1:], mask where they're not padding
            target_ids = batch_variant_ids[:, 1:]  # (batch_size, L-1)
            valid_mask = target_ids != self.tokenizer.pad_token_id

            # Compute mean KL over valid positions for each sequence in batch
            num_valid = valid_mask.sum(dim=-1).clamp(min=1)  # (batch_size,)
            mean_kl_per_seq = (kl_per_token * valid_mask).sum(
                dim=-1
            ) / num_valid  # (batch_size,)

            all_kl_divs.append(mean_kl_per_seq)

        # Return mean KL across all variants
        if len(all_kl_divs) == 0:
            return torch.tensor(0.0, device=completion_ids.device)

        return torch.cat(all_kl_divs, dim=0).mean()

    def log_train_dataset_sample_counts(self, batch: Dict[str, Any]) -> None:
        """Keep and log a running count of *samples* seen per dataset name during training.

        Handles:
        - **Sequence packing**: `batch["ds_name"].text` is a length-1 list where the single string
          concatenates per-sample dataset names with "$" delimiters.
        - **No packing**: `batch["ds_name"].text` is a list of dataset-name strings, one per sample.

        Only runs on rank 0 to avoid duplicate/conflicting counts across ranks.
        Logs only in training (caller responsibility) and only logs dataset(s) updated this step.
        """
        # Only count on rank 0 to avoid duplicate counting across ranks
        if self.global_rank != 0:
            return

        if "ds_name" not in batch or batch["ds_name"] is None:
            return

        ds_name_obj = batch["ds_name"]
        # Prefer the project's StringObject convention, but be permissive.
        if hasattr(ds_name_obj, "text"):
            texts = ds_name_obj.text
        else:
            texts = ds_name_obj

        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        ds_names: List[str] = []
        for t in texts_list:
            if t is None:
                continue
            if "$" in t:
                ds_names.extend([x for x in t.split("$") if x])
            else:
                ds_names.append(t)

        if len(ds_names) == 0:
            return

        updated_totals: Dict[str, int] = {}
        for name in ds_names:
            self._train_dataset_sample_counts[name] += 1
            updated_totals[name] = self._train_dataset_sample_counts[name]

        # NOTE: sync_dist=False because we only run on rank 0, sync_dist=True leads to deadlock
        for name, total in updated_totals.items():
            self.log(
                f"train/samples_seen_rank0/{name}",
                float(total),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
                reduce_fx="sum",
            )

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        # we check whether we are in proteingym loader by looking at keys in batch
        if "DMS_scores" in batch:
            rank = self.trainer.global_rank if self.trainer else 0
            print(
                f"[Rank {rank}] validation step: {batch['DMS_id'].text[0]}", flush=True
            )
            outputs = self.validation_step_proteingym(batch)
            return outputs
        else:
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        loss = outputs.loss
        self.log_metrics(
            batch,
            outputs,
            "val",
            log_global=dataloader_idx == 0,
        )
        return loss

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        # we check whether we are in proteingym loader by looking at keys in batch
        if "DMS_scores" in batch:
            outputs = self.validation_step_proteingym(batch)
            return outputs
        else:
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        loss = outputs.loss
        self.log_metrics(batch, outputs, "test", log_global=dataloader_idx == 0)
        return loss

    def on_save_checkpoint(self, checkpoint):
        """Save additional model state to checkpoint."""
        pass

    def on_load_checkpoint(self, checkpoint):
        """Handle checkpoint loading, optionally overriding optimizer and scheduler states.

        If override_optimizer_on_load is True, we'll remove the optimizer and
        lr_scheduler states from the checkpoint, forcing Lightning to create new ones
        based on the current config hyperparameters.

        Also handles key remapping for legacy checkpoints that used different naming.
        """
        # Handle legacy checkpoint key remapping
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remap legacy key: model.token_embedder.weight -> model.model.embed_tokens.weight
            if "model.token_embedder.weight" in state_dict:
                log.info(
                    "Remapping legacy checkpoint key: model.token_embedder.weight -> model.model.embed_tokens.weight"
                )
                state_dict["model.model.embed_tokens.weight"] = state_dict.pop(
                    "model.token_embedder.weight"
                )

            # Remove _reference_model keys - it's recreated dynamically in on_fit_start
            ref_model_keys = [k for k in state_dict.keys() if k.startswith("_reference_model.")]
            if ref_model_keys:
                log.info(
                    f"Removing {len(ref_model_keys)} _reference_model keys from checkpoint "
                    "(will be recreated in on_fit_start)"
                )
                for k in ref_model_keys:
                    del state_dict[k]

            # Remove _encoder_model keys - it's recreated dynamically
            encoder_model_keys = [k for k in state_dict.keys() if k.startswith("_encoder_model.")]
            if encoder_model_keys:
                log.info(
                    f"Removing {len(encoder_model_keys)} _encoder_model keys from checkpoint "
                    "(will be recreated via init_encoder_decoder_grpo)"
                )
                for k in encoder_model_keys:
                    del state_dict[k]

        if self.override_optimizer_on_load:
            if "optimizer_states" in checkpoint:
                log.info(
                    "Overriding optimizer state from checkpoint with current config values"
                )
                del checkpoint["optimizer_states"]

            if "lr_schedulers" in checkpoint:
                log.info(
                    "Overriding lr scheduler state from checkpoint with current config values"
                )
                del checkpoint["lr_schedulers"]

            # Set a flag to tell Lightning not to expect optimizer states
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []

        # Optionally reset training step/epoch counters
        if self.override_step_on_load:
            if "global_step" in checkpoint:
                log.info(
                    f"Resetting global_step from {checkpoint['global_step']} to 0"
                )
                checkpoint["global_step"] = 0
            if "epoch" in checkpoint:
                log.info(f"Resetting epoch from {checkpoint['epoch']} to 0")
                checkpoint["epoch"] = 0
            # Also reset loop states if present
            if "loops" in checkpoint:
                log.info("Resetting training loop states")
                del checkpoint["loops"]

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer_name = self.hparams.get("optimizer", "adamw")
        log.info(f"Using optimizer {optimizer_name}")
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.95),
                eps=self.eps,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        optim_dict = {"optimizer": optimizer}
        if self.scheduler_name is not None:
            if self.scheduler_name == "cosine_with_min_lr":
                scheduler = get_scheduler(
                    self.scheduler_name,
                    optimizer,
                    num_warmup_steps=self.num_warmup_steps,
                    num_training_steps=self.num_training_steps,
                    scheduler_specific_kwargs={"min_lr_rate": 0.1},
                )
            elif self.scheduler_name == "warmup_stable_decay":
                if self.num_decay_steps is None:
                    raise ValueError(
                        "num_decay_steps is required for warmup_stable_decay scheduler"
                    )

                num_warmup_steps = self.num_warmup_steps
                num_decay_steps = self.num_decay_steps
                num_training_steps = self.num_training_steps
                num_decay_start_step = num_training_steps - num_decay_steps
                min_lr_ratio = 0.1

                def lr_lambda(current_step: int):
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(max(1, num_warmup_steps))
                    elif current_step < num_decay_start_step:
                        return 1.0
                    else:
                        progress = min(
                            1.0,
                            float(current_step - num_decay_start_step)
                            / float(max(1, num_decay_steps)),
                        )
                        return (
                            max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            * (1.0 - min_lr_ratio)
                            + min_lr_ratio
                        )

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                scheduler = get_scheduler(
                    self.scheduler_name,
                    optimizer,
                    num_warmup_steps=self.num_warmup_steps,
                    num_training_steps=self.num_training_steps,
                )
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
            }
        return optim_dict

    def trim_eval_batch(self, seqs_ids):
        """
        trim to first padding token in mini-batch
        (if batch-size is 1: avoid padding entirely)
        """
        pad_tok = self.tokenizer.vocab["[PAD]"]
        mask = seqs_ids != pad_tok
        indices = torch.arange(seqs_ids.shape[-1], device=seqs_ids.device).expand(
            seqs_ids.shape
        )
        # Set indices with padding to 0
        indices = torch.where(mask, indices, torch.tensor(0, device=seqs_ids.device))
        max_non_pad_index_per_seq = torch.max(indices, dim=-1).values
        return seqs_ids[..., : max_non_pad_index_per_seq.max() + 1]

    def _score_seqs_kv_cache(
        self,
        input_ids,
        completion_ids,
        batch_size: int = 1,
        verbose: bool = False,
        return_tensor: bool = False,
    ):
        """Score completion sequences using KV cache for efficiency.

        Args:
            input_ids: Context tokens of shape (1, L_context)
            completion_ids: Completion tokens of shape (1, N, L_completion)
            batch_size: Number of completions to process at once
            verbose: Whether to show progress bar
            return_tensor: If True, return torch tensor (preserves gradients);
                          if False, return numpy array (default, for backward compat)

        Returns:
            Mean log-likelihoods per completion, shape (N,)
        """
        # input_ids is b, L; completion_ids is b, n, L
        # https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization
        # https://github.com/huggingface/transformers/blob/b7672826cad31e30319487af876e608d8af7d37b/src/transformers/generation/utils.py#L1879
        # https://github.com/huggingface/transformers/blob/67a4ef89d4ddbfd7d61e479359a1b609e5ee9843/src/transformers/models/mistral/modeling_mistral.py#L1233
        all_lls = []
        assert (
            input_ids[0, 0] == self.tokenizer.vocab["[start-of-document]"]
            and input_ids[0, 1] > 19
        ), "First two tokens should be special start-of-doc and document type"
        if completion_ids[0, 0, 0] == self.tokenizer.sep_token_id:
            assert (
                input_ids[0, -1] != self.tokenizer.sep_token_id
            ), "Double sep token in input and completion"
        outputs = self.model(input_ids=input_ids, use_cache=True)
        past_key_values = (
            outputs.past_key_values
        )  # just a tuple of tensors - doesn't get extended
        L = completion_ids.shape[-1]

        for batch_start in tqdm.tqdm(
            range(0, completion_ids.shape[1], batch_size), disable=not verbose
        ):
            # TODO: for batch_size > 1, we need to expand out the cache - c.f. generate
            # fmt: off
            this_input_ids = completion_ids[
                :, batch_start: batch_start + batch_size
            ].reshape(-1, L)  # b_mut, L
            # fmt: on
            # remove unnecessary padding:
            this_input_ids = self.trim_eval_batch(this_input_ids)
            L_mini_batch = this_input_ids.shape[-1]

            actual_batch_size = this_input_ids.shape[0]
            cache = InputAwareDynamicCache.from_legacy_cache(past_key_values)
            cache.batch_repeat_interleave(actual_batch_size)  # careful: returns None!
            # fmt: off
            outputs = self.model(
                input_ids=this_input_ids,
                past_key_values=cache,
                use_cache=True,
            )
            # fmt: on
            labels = torch.where(
                this_input_ids == self.tokenizer.pad_token_id,
                -100,
                this_input_ids.clone(),
            )
            # start_ix is 0 as this is likelihood for first AA (pos 1)
            log_likelihood = log_likelihood_from_outputs(outputs, labels, start_ix=0)

            # mask padded positions in before computing the mean.
            shift_labels = labels[..., 1:].to(
                log_likelihood.device
            )  # aligns with start_ix=0
            mask = shift_labels != -100
            denom = mask.sum(dim=-1).clamp(min=1)
            ll_mean = (log_likelihood * mask).sum(dim=-1) / denom
            all_lls.append(ll_mean)  # b_mut

        lls = torch.cat(all_lls)
        if return_tensor:
            return lls
        return lls.cpu().float().numpy()

    def _score_seqs_no_cache(
        self,
        input_ids,
        completion_ids,
        batch_size: int = 1,
        verbose: bool = False,
    ):
        # input_ids is b, L; completion_ids is b, n, L
        if batch_size > 1:
            raise NotImplementedError(
                "Mutant batch size > 1 not yet supported for mutant scoring"
            )
        all_lls = []
        likelihood_start_ix = input_ids.shape[1]
        for completion_ix in tqdm.tqdm(
            range(completion_ids.shape[1]), disable=not verbose
        ):
            this_input_ids = torch.cat(
                [input_ids, completion_ids[:, completion_ix]],
                dim=1,
            )
            # remove unnecessary padding:
            this_input_ids = self.trim_eval_batch(this_input_ids)
            L_mini_batch = this_input_ids.shape[-1]  # beware: includes prompt too
            # https://github.com/huggingface/transformers/blob/048f599f3506e57e0a595b455d9d2834c8d45023/src/transformers/data/data_collator.py#L823
            labels = torch.where(
                this_input_ids == self.tokenizer.pad_token_id,
                -100,
                this_input_ids.clone(),
            )
            assert (
                this_input_ids[..., likelihood_start_ix] not in self.tokenizer.aa_tokens
            ), "Likelihood start ix is an AA token - likelihood cannot be computed for this position"

            outputs = self.model(input_ids=this_input_ids, use_cache=False)
            # TODO: maybe relabel start_ix - a bit confusing
            log_likelihood = log_likelihood_from_outputs(
                outputs, labels, start_ix=likelihood_start_ix
            )  # 1, L
            shift_labels = labels[..., likelihood_start_ix + 1 :].to(
                log_likelihood.device
            )
            mask = shift_labels != -100
            denom = mask.sum(dim=-1).clamp(min=1)
            ll_mean = (log_likelihood * mask).sum(dim=-1) / denom
            all_lls.append(ll_mean.item())
        lls = np.array(all_lls)
        return lls

    def _score_seqs_no_context(
        self,
        completion_ids,
        batch_size: int = 1,
        verbose: bool = False,
        start_tokens: list[int] = [47, 63],
        return_tensor: bool = False,
    ):
        """Score sequences without context (standalone sequence scoring).

        Args:
            completion_ids: Completion tokens of shape (N, L) or (1, N, L)
            batch_size: Number of completions to process at once
            verbose: Whether to show progress bar
            start_tokens: Tokens to prepend (default: [start-of-document][RAW])
            return_tensor: If True, return torch tensor (preserves gradients);
                          if False, return numpy array (default)

        Returns:
            Mean log-likelihoods per completion, shape (N,)
        """
        if len(completion_ids.shape) == 3:
            completion_ids = completion_ids.squeeze(0)
        if (completion_ids[:, 0] == self.tokenizer.sep_token_id).any():
            assert (
                completion_ids[:, 0] == self.tokenizer.sep_token_id
            ).all(), "Some sequences have sep token at start but not all"
            completion_ids = completion_ids[:, 1:]
        if (completion_ids[:, 0] != start_tokens[0]).any():
            start_tokens_tensor = (
                torch.tensor(start_tokens, device=completion_ids.device)
                .unsqueeze(0)
                .repeat(completion_ids.shape[0], 1)
            )
            completion_ids = torch.cat([start_tokens_tensor, completion_ids], dim=-1)
        all_lls = []
        for completion_ix in tqdm.tqdm(
            range(0, completion_ids.shape[0], batch_size), disable=not verbose
        ):
            this_input_ids = completion_ids[completion_ix : completion_ix + batch_size]
            outputs = self.model(input_ids=this_input_ids, use_cache=False)
            labels = torch.where(
                this_input_ids == self.tokenizer.pad_token_id,
                -100,
                this_input_ids.clone(),
            )
            log_likelihood = log_likelihood_from_outputs(
                outputs, labels, start_ix=1
            )  # 1, L
            shift_labels = labels[..., 2:].to(
                log_likelihood.device
            )  # aligns with start_ix=1
            mask = shift_labels != -100
            denom = mask.sum(dim=-1).clamp(min=1)
            ll_mean = (log_likelihood * mask).sum(dim=-1) / denom
            all_lls.append(ll_mean)

        lls = torch.cat(all_lls)
        if return_tensor:
            return lls
        return lls.cpu().float().numpy()

    def score_seqs(
        self,
        input_ids,
        completion_ids,
        use_cache: bool = True,
        batch_size: int = 1,
        return_tensor: bool = False,
    ):
        """Score completion sequences given optional context.

        Args:
            input_ids: Context tokens of shape (1, L_context), or None for no context
            completion_ids: Completion tokens of shape (1, N, L_completion) or (N, L_completion)
            use_cache: Whether to use KV cache for efficiency (requires context)
            batch_size: Number of completions to process at once
            return_tensor: If True, return torch tensor (preserves gradients);
                          if False, return numpy array (default)

        Returns:
            Mean log-likelihoods per completion, shape (N,)
        """
        if input_ids is not None:
            assert (
                input_ids.shape[0] == 1
            ), "Only batch size 1 is supported for mutant scoring; batch dim must be present"
            assert (
                input_ids.ndim == 2 and completion_ids.ndim == 3
            ), f"input ids shape {input_ids.shape}, completion ids shape {completion_ids.shape}"  # b, L; b, n, L
            if use_cache:
                return self._score_seqs_kv_cache(
                    input_ids,
                    completion_ids,
                    batch_size=batch_size,
                    return_tensor=return_tensor,
                )
            else:
                return self._score_seqs_no_cache(
                    input_ids,
                    completion_ids,
                    batch_size=batch_size,
                )
        else:
            return self._score_seqs_no_context(
                completion_ids,
                batch_size=batch_size,
                return_tensor=return_tensor,
            )

    def _sample_seqs(
        self,
        input_ids,
        num_samples,
        max_tokens: int,
        max_generated_length: Optional[int] = None,
        max_total_length: Optional[
            int
        ] = None,  # maximum length of inputs plus completions
        fixed_length: Optional[int] = None,
        greedy: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        sample_gaps: bool = False,
        structure_tokens: bool = False,
        continuous_sampling: bool = False,
        repeat_guard: bool = True,
        repeat_length: int = 9,  # if last repeat_length chars appear repeat_count times, seq is aborted
        repeat_count: int = 9,
        max_retries: int = 3,
        suppress_bad_words: bool = True,
        return_per_token_log_probs: bool = False,
        batch_generation: bool = False,
        generation_batch_size: int = 8,
    ):
        """
        Conditionally independent sequence generation: sequences are generated independently of each other
        given the prompt. Once sep token is generated, the sequence is considered complete.
        (i.e. we don't generate a sequence of sequences directly).

        If return_per_token_log_probs is True, additionally returns per-token
        log-probabilities and a validity mask (for GRPO per-token ratio computation).

        If batch_generation is True, uses batched model.generate() calls for 3-5x speedup.
        generation_batch_size controls how many sequences are generated in parallel per batch.
        """
        # Use batched generation if enabled
        if batch_generation and not continuous_sampling:
            return self._sample_seqs_batched(
                input_ids=input_ids,
                num_samples=num_samples,
                max_tokens=max_tokens,
                max_generated_length=max_generated_length,
                max_total_length=max_total_length,
                fixed_length=fixed_length,
                greedy=greedy,
                temperature=temperature,
                top_p=top_p,
                sample_gaps=sample_gaps,
                structure_tokens=structure_tokens,
                repeat_guard=repeat_guard,
                repeat_length=repeat_length,
                repeat_count=repeat_count,
                max_retries=max_retries,
                suppress_bad_words=suppress_bad_words,
                return_per_token_log_probs=return_per_token_log_probs,
                generation_batch_size=generation_batch_size,
            )

        # TODO: pass attention mask, pad_token_id to avoid the following warning:
        # The attention mask and the pad token id were not set. As a consequence, you may
        # observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
        # TODO: add min length kwarg
        if max_total_length is None:
            max_total_length = max_tokens
        if max_generated_length is not None:
            assert max_generated_length <= max_total_length
        generation_kwargs = {}
        sep_token_id = self.tokenizer.sep_token_id
        if fixed_length is not None:
            if max_total_length is not None:
                assert input_ids.shape[1] + fixed_length <= max_total_length
            generation_kwargs["min_new_tokens"] = fixed_length
            generation_kwargs["max_new_tokens"] = fixed_length
            generation_kwargs["eos_token_id"] = None
        elif max_generated_length is not None:
            generation_kwargs["min_new_tokens"] = 3
            generation_kwargs["max_new_tokens"] = max_generated_length
            generation_kwargs["eos_token_id"] = (
                None if continuous_sampling else sep_token_id
            )
        else:
            generation_kwargs["min_new_tokens"] = 3  # for esmfold
            generation_kwargs["eos_token_id"] = (
                None if continuous_sampling else sep_token_id
            )
            generation_kwargs["max_length"] = max_total_length
        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if top_p is not None:
            # nucleus sampling; ensure valid range
            if not (0.0 < float(top_p) <= 1.0):
                raise ValueError("top_p must be in the interval (0, 1]")
            generation_kwargs["top_p"] = float(top_p)
        if suppress_bad_words:
            bad_aas = [
                "X",
                "x",
                "B",
                "J",
                "O",
                "U",
                "Z",
            ]
            if not sample_gaps:
                bad_aas.append("-")
            if structure_tokens:
                bad_aas = bad_aas + aa_letters
            else:
                bad_aas = bad_aas + aa_letters_lower

            # each 'word' is treated as a list of tokens
            # TODO: write test for this with random model.
            generation_kwargs["bad_words_ids"] = [
                [tok_id]
                for tok_id in self.tokenizer.all_special_ids
                if tok_id != self.tokenizer.eos_token_id
            ]
            generation_kwargs["bad_words_ids"] += [
                [self.tokenizer.convert_tokens_to_ids(bad_aa)] for bad_aa in bad_aas
            ]

        assert (
            input_ids.shape[0] == 1 and input_ids.ndim == 2
        ), "Only batch size 1 is supported for sampling; batch dim must be present"

        all_outputs: List[torch.Tensor] = []
        all_scores: List[float] = []
        all_per_token_lps: List[List[float]] = []
        # Always generate exactly one sequence at a time
        for batch_start in tqdm.tqdm(range(num_samples), "Generating sequences"):
            remaining = 1
            attempt = 0
            batch_collected: List[torch.Tensor] = []
            batch_scores: List[float] = []
            batch_token_lps: List[List[float]] = []
            while remaining > 0:
                # Build stopping criteria that knows prompt length (non-continuous only)
                stopping = None
                if not continuous_sampling and repeat_guard:
                    prompt_len = input_ids.shape[1]
                    stopping = StoppingCriteriaList(
                        [
                            RepeatStoppingCriteria(
                                self.tokenizer,
                                repeat_length=repeat_length,
                                repeat_count=repeat_count,
                                prompt_length=prompt_len,
                            )
                        ]
                    )
                gen_out = self.model.generate(
                    input_ids=input_ids,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=not greedy,
                    temperature=temperature,
                    stopping_criteria=stopping,
                    **generation_kwargs,
                )
                seqs_full = gen_out.sequences  # (remaining, L_total)
                scores_list = gen_out.scores  # List[T] of (remaining, V)
                # Slice off prompt
                prompt_len = input_ids.shape[1]
                seqs = seqs_full[:, prompt_len:]

                # Evaluate which are acceptable vs need retry
                failed_indices: List[int] = []
                for i in range(seqs.shape[0]):
                    row = seqs[i]
                    # find last non-pad token index
                    pad_id = self.tokenizer.pad_token_id
                    valid_len = int((row != pad_id).sum().item())
                    last_tok = (
                        int(row[valid_len - 1].item()) if valid_len > 0 else pad_id
                    )
                    text = self.tokenizer.decode(
                        row[:valid_len].tolist(), skip_special_tokens=True
                    ).replace(" ", "")
                    ends_with_sep = last_tok == self.tokenizer.sep_token_id
                    is_repeaty = has_too_many_repeats(
                        text, repeat_length=repeat_length, repeat_count=repeat_count
                    )
                    if (not ends_with_sep) or (
                        is_repeaty and (not continuous_sampling)
                    ):
                        failed_indices.append(i)
                    else:
                        # accept and score
                        batch_collected.append(row.unsqueeze(0))
                        # compute mean logp up to SEP if present
                        total_logp = 0.0
                        count = 0
                        token_lps: List[float] = []
                        finished_non_cont = False
                        T = len(scores_list)
                        for t in range(T):
                            token_id = (
                                int(seqs[i, t].item()) if t < seqs.shape[1] else pad_id
                            )
                            lp = F.log_softmax(scores_list[t], dim=-1)[
                                i, token_id
                            ].item()
                            if not continuous_sampling:
                                if finished_non_cont:
                                    continue
                                total_logp += float(lp)
                                token_lps.append(float(lp))
                                count += 1
                                if token_id == self.tokenizer.sep_token_id:
                                    finished_non_cont = True
                            else:
                                raise ValueError(
                                    "Continuous sampling is not supported for base model"
                                )
                        batch_scores.append(total_logp / max(count, 1))
                        batch_token_lps.append(token_lps)

                if len(failed_indices) == 0:
                    remaining = 0
                else:
                    attempt += 1
                    if attempt > max_retries:
                        # accept remaining failed ones as-is (score them) to avoid infinite loop
                        for i in failed_indices:
                            row = seqs[i]
                            batch_collected.append(row.unsqueeze(0))
                            total_logp = 0.0
                            count = 0
                            token_lps_fail: List[float] = []
                            T = len(scores_list)
                            for t in range(T):
                                token_id = (
                                    int(seqs[i, t].item())
                                    if t < seqs.shape[1]
                                    else pad_id
                                )
                                lp = F.log_softmax(scores_list[t], dim=-1)[
                                    i, token_id
                                ].item()
                                total_logp += float(lp)
                                token_lps_fail.append(float(lp))
                                count += 1
                            batch_scores.append(total_logp / max(count, 1))
                            batch_token_lps.append(token_lps_fail)
                        remaining = 0
                    else:
                        remaining = len(failed_indices)

            # Commit collected from this batch
            if len(batch_collected) > 0:
                all_outputs.append(torch.cat(batch_collected, dim=0))
                all_scores.extend(batch_scores)
                all_per_token_lps.extend(batch_token_lps)

        max_output_length = max([o.shape[1] for o in all_outputs])
        padded_outputs = torch.full(
            (num_samples, max_output_length), self.tokenizer.pad_token_id
        )
        start_ix = 0
        for o in all_outputs:
            padded_outputs[start_ix : start_ix + o.shape[0], : o.shape[1]] = o
            start_ix += o.shape[0]

        if return_per_token_log_probs:
            per_token_lps_tensor = torch.zeros(num_samples, max_output_length)
            per_token_mask_tensor = torch.zeros(
                num_samples, max_output_length, dtype=torch.bool
            )
            for idx, lps in enumerate(all_per_token_lps):
                n_tok = len(lps)
                per_token_lps_tensor[idx, :n_tok] = torch.tensor(lps)
                per_token_mask_tensor[idx, :n_tok] = True
            return padded_outputs, all_scores, per_token_lps_tensor, per_token_mask_tensor

        return padded_outputs, all_scores

    def _sample_seqs_batched(
        self,
        input_ids,
        num_samples: int,
        max_tokens: int,
        max_generated_length: Optional[int] = None,
        max_total_length: Optional[int] = None,
        fixed_length: Optional[int] = None,
        greedy: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        sample_gaps: bool = False,
        structure_tokens: bool = False,
        repeat_guard: bool = True,
        repeat_length: int = 9,
        repeat_count: int = 9,
        max_retries: int = 3,
        suppress_bad_words: bool = True,
        return_per_token_log_probs: bool = False,
        generation_batch_size: int = 8,
    ):
        """
        Batched sequence generation for improved throughput.

        Instead of generating sequences one at a time, this method generates multiple
        sequences in parallel using batched model.generate() calls with vectorized
        log-probability extraction.

        Args:
            input_ids: Prompt tokens (1, L_prompt)
            num_samples: Total number of sequences to generate
            max_tokens: Maximum total tokens
            max_generated_length: Maximum generated sequence length
            max_total_length: Maximum total length (prompt + generated)
            fixed_length: If set, generate exactly this many tokens
            greedy: Use greedy decoding instead of sampling
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            sample_gaps: Allow gap tokens in generated sequences
            structure_tokens: Use structure token vocabulary
            repeat_guard: Enable repetition detection and stopping
            repeat_length: Window size for repetition detection
            repeat_count: Threshold for repetition stopping
            max_retries: Maximum retries per batch for failed sequences
            suppress_bad_words: Suppress non-standard amino acid tokens
            return_per_token_log_probs: Return per-token log probabilities
            generation_batch_size: Number of sequences to generate per batch

        Returns:
            If return_per_token_log_probs is False:
                (padded_outputs, all_scores)
            If return_per_token_log_probs is True:
                (padded_outputs, all_scores, per_token_lps_tensor, per_token_mask_tensor)
        """
        # Build generation kwargs (same as sequential version)
        if max_total_length is None:
            max_total_length = max_tokens
        if max_generated_length is not None:
            assert max_generated_length <= max_total_length

        generation_kwargs = {}
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id

        if fixed_length is not None:
            if max_total_length is not None:
                assert input_ids.shape[1] + fixed_length <= max_total_length
            generation_kwargs["min_new_tokens"] = fixed_length
            generation_kwargs["max_new_tokens"] = fixed_length
            generation_kwargs["eos_token_id"] = None
        elif max_generated_length is not None:
            generation_kwargs["min_new_tokens"] = 3
            generation_kwargs["max_new_tokens"] = max_generated_length
            generation_kwargs["eos_token_id"] = sep_token_id
        else:
            generation_kwargs["min_new_tokens"] = 3
            generation_kwargs["eos_token_id"] = sep_token_id
            generation_kwargs["max_length"] = max_total_length

        generation_kwargs["pad_token_id"] = pad_token_id

        if top_p is not None:
            if not (0.0 < float(top_p) <= 1.0):
                raise ValueError("top_p must be in the interval (0, 1]")
            generation_kwargs["top_p"] = float(top_p)

        if suppress_bad_words:
            bad_aas = ["X", "x", "B", "J", "O", "U", "Z"]
            if not sample_gaps:
                bad_aas.append("-")
            if structure_tokens:
                bad_aas = bad_aas + aa_letters
            else:
                bad_aas = bad_aas + aa_letters_lower

            generation_kwargs["bad_words_ids"] = [
                [tok_id]
                for tok_id in self.tokenizer.all_special_ids
                if tok_id != self.tokenizer.eos_token_id
            ]
            generation_kwargs["bad_words_ids"] += [
                [self.tokenizer.convert_tokens_to_ids(bad_aa)] for bad_aa in bad_aas
            ]

        assert (
            input_ids.shape[0] == 1 and input_ids.ndim == 2
        ), "Only batch size 1 is supported for sampling; batch dim must be present"

        prompt_len = input_ids.shape[1]
        all_outputs: List[torch.Tensor] = []
        all_scores: List[float] = []
        all_per_token_lps: List[List[float]] = []

        # Generate all sequences in batches without retry loop
        generated_count = 0

        with tqdm.tqdm(total=num_samples, desc="Generating sequences (batched)") as pbar:
            while generated_count < num_samples:
                current_batch_size = min(num_samples - generated_count, generation_batch_size)

                # Expand prompt to batch size: (1, L) -> (B, L)
                batch_input_ids = input_ids.expand(current_batch_size, -1)

                # Build stopping criteria for this batch
                stopping = None
                if repeat_guard:
                    stopping = StoppingCriteriaList(
                        [
                            RepeatStoppingCriteria(
                                self.tokenizer,
                                repeat_length=repeat_length,
                                repeat_count=repeat_count,
                                prompt_length=prompt_len,
                            )
                        ]
                    )

                # Generate batch
                gen_out = self.model.generate(
                    input_ids=batch_input_ids,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=not greedy,
                    temperature=temperature,
                    stopping_criteria=stopping,
                    **generation_kwargs,
                )

                seqs_full = gen_out.sequences  # (B, L_total)
                scores_list = gen_out.scores  # List of (B, V) tensors, length T

                # Slice off prompt to get generated tokens only
                seqs = seqs_full[:, prompt_len:]  # (B, T)
                T = len(scores_list)

                # Vectorized log-prob extraction
                if T > 0:
                    # Stack scores: List[(B, V)] -> (T, B, V) -> (B, T, V)
                    stacked_scores = torch.stack(scores_list, dim=0).transpose(0, 1)
                    log_probs_all = F.log_softmax(stacked_scores, dim=-1)  # (B, T, V)

                    # Gather log probs for actual generated tokens
                    # seqs shape: (B, seq_len), we need (B, T) for indexing
                    seq_len = seqs.shape[1]
                    tokens_for_gather = seqs[:, :T].unsqueeze(-1)  # (B, T, 1)
                    selected_log_probs = log_probs_all.gather(-1, tokens_for_gather).squeeze(-1)  # (B, T)
                else:
                    selected_log_probs = torch.zeros(current_batch_size, 0, device=seqs.device)

                # Accept ALL sequences - let penalty system handle invalid ones
                for i in range(current_batch_size):
                    row = seqs[i]
                    valid_len = int((row != pad_token_id).sum().item())

                    # Compute per-token log probs for all generated tokens
                    token_lps: List[float] = []
                    total_logp = 0.0
                    count = 0

                    for t in range(min(T, valid_len)):
                        token_id = int(seqs[i, t].item())
                        lp = float(selected_log_probs[i, t].item())
                        token_lps.append(lp)
                        total_logp += lp
                        count += 1
                        if token_id == sep_token_id:
                            break

                    all_outputs.append(row.unsqueeze(0))
                    all_scores.append(total_logp / max(count, 1))
                    all_per_token_lps.append(token_lps)
                    generated_count += 1
                    pbar.update(1)

        # Pad all outputs to same length
        if len(all_outputs) == 0:
            # Fallback: return empty tensors
            if return_per_token_log_probs:
                return (
                    torch.full((num_samples, 1), pad_token_id),
                    [0.0] * num_samples,
                    torch.zeros(num_samples, 1),
                    torch.zeros(num_samples, 1, dtype=torch.bool),
                )
            return torch.full((num_samples, 1), pad_token_id), [0.0] * num_samples

        max_output_length = max([o.shape[1] for o in all_outputs])
        padded_outputs = torch.full(
            (len(all_outputs), max_output_length), pad_token_id, device=all_outputs[0].device
        )
        for idx, o in enumerate(all_outputs):
            padded_outputs[idx, : o.shape[1]] = o

        if return_per_token_log_probs:
            per_token_lps_tensor = torch.zeros(len(all_outputs), max_output_length)
            per_token_mask_tensor = torch.zeros(
                len(all_outputs), max_output_length, dtype=torch.bool
            )
            for idx, lps in enumerate(all_per_token_lps):
                n_tok = len(lps)
                per_token_lps_tensor[idx, :n_tok] = torch.tensor(lps)
                per_token_mask_tensor[idx, :n_tok] = True
            return padded_outputs, all_scores, per_token_lps_tensor, per_token_mask_tensor

        return padded_outputs, all_scores

    def log_metrics(self, batch, outputs, step_name, log_global: bool = True):
        # N.B. actually val logging is a bit different because of this ds name thing
        loss = outputs.loss
        n_tokens = batch["input_ids"].shape[-1]
        if step_name == "train":
            ds_names = None
        else:
            ds_names = batch["ds_name"].text
        dataset_accuracies = metrics.accuracy_from_outputs(
            batch["input_ids"],
            outputs,
            batch["labels"],
            ignore_index=self.ignore_index,
            dataset_names=ds_names,  # a list of dataset names (StringObject.text)
            ignore_token_ids=self.tokenizer.convert_tokens_to_ids(
                ["-", "X", "x", "[start-of-document]"]
                + aa_letters_lower
                + self.tokenizer.all_special_tokens
            ),
            sep_token_id=self.tokenizer.sep_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            calc_full_no_context_accuracies=True,
        )

        global_metrics = {
            "loss": loss,
            "ppl": torch.exp(loss),
            "aa_accuracy": dataset_accuracies.pop("global"),
            "aa_accuracy_first_sequence": dataset_accuracies.pop("first_sequence"),
            "aa_accuracy_last_sequence": dataset_accuracies.pop("last_sequence"),
            "n_tokens_in_batch": n_tokens,
        }

        if log_global:
            self.log_dict(
                {f"{step_name}/{k}": v for k, v in global_metrics.items()},
                on_step=step_name == "train",
                on_epoch=step_name != "train",
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=step_name != "train",
            )

        # n.b. this assumes a batch only contains a single dataset - only true during val!
        # assert all([ds_name == batch["ds_name"][0] for ds_name in batch["ds_name"]])
        assert isinstance(batch["ds_name"], StringObject)

        is_single_dataset_batch = len(set(batch["ds_name"].text)) == 1
        for ds_name in set(batch["ds_name"].text):
            if ds_name not in dataset_accuracies:
                continue
            ds_metrics = {
                f"{step_name}/{ds_name}/aa_accuracy": dataset_accuracies[ds_name],
                f"{step_name}/{ds_name}/aa_accuracy_first_sequence": dataset_accuracies[
                    ds_name + "_first_sequence"
                ],
                f"{step_name}/{ds_name}/aa_accuracy_last_sequence": dataset_accuracies[
                    ds_name + "_last_sequence"
                ],
            }
            if is_single_dataset_batch:
                # global metrics are dataset specific
                ds_metrics[f"{step_name}/{ds_name}/loss"] = loss
            self.log_dict(
                ds_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
                sync_dist=step_name != "train",  # Q: what happens if sync_dist is False
            )
        add_dataloader_idx = step_name != "train"
        seq_len_stats = metrics.sequence_lengths(
            batch["labels"], self.tokenizer.sep_token_id
        )
        sep_tokens_in_batch = (
            (batch["labels"] == self.tokenizer.sep_token_id).sum().item()
        )
        start_of_doc_tokens_in_batch = (
            (batch["input_ids"] == self.tokenizer.bos_token_id).sum().item()
        )
        for reduce_fx in ["min", "max", "mean"]:
            self.log(
                name=f"{step_name}/token_stats/{reduce_fx}_seq_len_in_batch",
                value=seq_len_stats[f"{reduce_fx}_seq_length"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                reduce_fx=reduce_fx,
                add_dataloader_idx=add_dataloader_idx,
            )
            self.log(
                name=f"{step_name}/token_stats/{reduce_fx}_sep_tokens_in_batch",
                value=sep_tokens_in_batch,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                reduce_fx=reduce_fx,
                add_dataloader_idx=add_dataloader_idx,
            )
            self.log(
                name=f"{step_name}/token_stats/{reduce_fx}_start_of_doc_tokens_in_batch",
                value=start_of_doc_tokens_in_batch,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                reduce_fx=reduce_fx,
                add_dataloader_idx=add_dataloader_idx,
            )

    def validation_step_proteingym(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Assumes that batch contains the following:

        input_ids: the prompt (i.e. MSA)
        completion_ids: the completions (i.e. mutated sequences)

        on caching: it seems like, if we modify what is passed to attention forward, existing cache
        might just work. currently model/sampling loop probably passes just the next token.
        """
        assert batch["DMS_scores"].ndim == 2  # b, n
        L = batch["completion_ids"].shape[-1]
        L_prompt = batch["input_ids"].shape[-1]
        lls = self.score_seqs(
            batch["input_ids"],
            batch["completion_ids"],
            use_cache=self.use_kv_cache_for_scoring,
            batch_size=max(self.scoring_max_tokens // (L + L_prompt), 1)
            if self.use_kv_cache_for_scoring
            else 1,
        )
        dms_scores = batch["DMS_scores"][0].to(torch.float32).cpu().numpy()

        if lls.min() == lls.max():
            spearman_corr = 0
        else:
            spearman_corr, _ = spearmanr(
                lls.astype(np.float32),
                dms_scores,
            )
                # Get the assay name for per-assay logging
        dms_id = (
            batch["DMS_id"].text[0]
            if hasattr(batch["DMS_id"], "text")
            else str(batch["DMS_id"][0])
        )

        self.log(
            f"gym/assay/{dms_id}/spearman",
            spearman_corr,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )
        self.log(
            "gym/spearman",
            spearman_corr,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "gym/log_likelihood",
            lls.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        # Compute and log GRPO-style validation metrics if GRPO is enabled
        if self.grpo_enabled:
            # Compute what the GRPO loss would be for this validation batch
            rewards = torch.tensor(dms_scores, dtype=torch.float32)
            advantages = self._compute_grpo_advantages(rewards)
            log_likelihoods_tensor = torch.tensor(lls, dtype=torch.float32)

            # GRPO loss (without gradients, just for monitoring)
            grpo_val_loss = -(advantages * log_likelihoods_tensor).mean().item()

            self.log(
                "gym/grpo_loss",
                grpo_val_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "gym/mean_advantage",
                advantages.mean().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def on_train_epoch_end(self):
        # Commenting out as may cause deadlock in DDP
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19604
        log.info("Train epoch end %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
