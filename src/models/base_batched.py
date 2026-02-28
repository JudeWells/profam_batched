import copy
import math
import os
import random
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
        grpo_group_size: int = 16,  # Number of sequences to compare per training step
        grpo_clip_ratio: float = 0.2,  # PPO-style clipping
        grpo_normalize_rewards: bool = True,  # Normalize DMS scores within group
        grpo_use_reference_model: bool = False,  # Use KL regularization to initial model
        grpo_reward_baseline: str = "mean",  # "mean", "min", or "none"
        grpo_max_tokens: int = 8_000,  # Max tokens per batch for GRPO (lower than scoring_max_tokens due to gradients)
        # Online HMM GRPO sampling parameters
        grpo_hmm_temperature: float = 1.0,
        grpo_hmm_top_p: Optional[float] = None,
        grpo_hmm_max_generated_length: Optional[int] = None,
        grpo_hmm_length_penalty: float = 0.1,  # Per-residue penalty for seq length > HMM model length M
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
        self.grpo_group_size = grpo_group_size
        self.grpo_clip_ratio = grpo_clip_ratio
        self.grpo_normalize_rewards = grpo_normalize_rewards
        self.grpo_use_reference_model = grpo_use_reference_model
        self.grpo_reward_baseline = grpo_reward_baseline
        self.grpo_max_tokens = (
            grpo_max_tokens  # Separate limit for GRPO (needs gradients)
        )

        # Online HMM GRPO sampling parameters
        self.grpo_hmm_temperature = grpo_hmm_temperature
        self.grpo_hmm_top_p = grpo_hmm_top_p
        self.grpo_hmm_max_generated_length = grpo_hmm_max_generated_length
        self.grpo_hmm_length_penalty = grpo_hmm_length_penalty

        # HMM reward scorer (set during on_fit_start from datamodule)
        self._hmm_scorer = None

        # PETase training components (set by setup_petase_training)
        self._petase_folder = None
        self._petase_energy_functions = None
        self._petase_template_residues = None

        # Mipa GRPO training components (set by setup_mipa_training)
        self._mipa_tm_scorer = None
        self._mipa_folder = None
        self._mipa_reasoning_mode = False
        self._mipa_num_reasoning_seqs = 3
        self._mipa_max_length = 400
        self._mipa_max_tokens = 600
        self._mipa_temperature = 1.0
        self._mipa_plddt_weight = 0.1
        self._mipa_length_penalty_threshold = 1048

        # Evolving prompt state for MIPA GRPO training
        self._evolving_prompt_enabled = False
        self._evolving_prompt_update_interval = 250
        self._evolving_prompt_min_tm_score = 0.3
        self._evolving_prompt_current_sequence = None  # Current evolved prompt (str)
        self._evolving_prompt_current_tokens = None  # Current prompt tokens (tensor)
        self._evolving_prompt_current_reward = None  # Reward of current prompt (for monotonic evolution)
        self._evolving_prompt_candidate_buffer = []  # Buffer of candidate sequences

        # Reference model for KL regularization (initialized lazily if needed)
        self._reference_model = None

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

        # Pick up HMM scorer from datamodule (set by PfamHMMGRPODataset)
        if self._hmm_scorer is None and self.grpo_enabled:
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None:
                scorer = getattr(dm, "hmm_scorer", None)
                if scorer is not None:
                    self._hmm_scorer = scorer
                    log.info(
                        f"HMM reward scorer attached with families: "
                        f"{scorer.available_families}"
                    )

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

        # Check if this is an online HMM GRPO batch
        if "hmm_family_id" in batch and self.grpo_enabled:
            return self.online_grpo_training_step(batch, batch_idx)

        # Check if this is a PETase GRPO batch (online reward computation)
        if batch.get("is_petase_batch", False) and self.grpo_enabled:
            return self.petase_grpo_training_step(batch, batch_idx)

        # Check if this is a Mipa GRPO batch (TM-score reward computation)
        if batch.get("is_mipa_batch", False) and self.grpo_enabled:
            return self.mipa_grpo_training_step(batch, batch_idx)

        # Check if this is a GRPO batch (contains DMS_scores - offline rewards)
        if "DMS_scores" in batch and self.grpo_enabled:
            return self.grpo_training_step(batch, batch_idx)

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

        # Compute context KV cache once (with gradients for the policy)
        has_context = input_ids is not None and input_ids.numel() > 0
        past_key_values = None
        if has_context:
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

    def grpo_training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """GRPO training step for ProteinGym data.

        This implements Group Relative Policy Optimization where:
        1. Compute advantages from ALL DMS scores (rewards) in the assay
        2. Sample a group of variants for gradient computation
        3. Compute log-likelihoods for sampled variants under current policy
        4. Update policy to increase likelihood of high-reward variants

        Note: Advantages are computed on ALL variants before subsampling to ensure
        consistent advantage estimates regardless of which variants are sampled.

        The loss is only computed on the variant sequences (last sequence in document),
        not on the MSA context.

        Args:
            batch: Dictionary containing:
                - input_ids: MSA context tokens (1, L_context)
                - completion_ids: Variant tokens (1, N, L_completion)
                - DMS_scores: Fitness scores (1, N)
                - DMS_id: Assay identifier
            batch_idx: Batch index

        Returns:
            loss: GRPO loss value
        """
        assert "DMS_scores" in batch, "GRPO training requires DMS_scores in batch"

        input_ids = batch["input_ids"]  # (1, L_context)
        completion_ids = batch["completion_ids"]  # (1, N, L_completion)
        dms_scores = batch["DMS_scores"]  # (1, N)

        # Flatten batch dimension for scores
        rewards = dms_scores[0].float()  # (N,)
        N = rewards.shape[0]

        # Compute advantages from ALL rewards (before subsampling)
        # This ensures consistent advantage estimates regardless of which variants are sampled
        all_advantages = self._compute_grpo_advantages(rewards)

        # Sample a group of variants if we have more than grpo_group_size
        if N > self.grpo_group_size:
            group_indices = random.sample(range(N), self.grpo_group_size)
            advantages = all_advantages[group_indices]
        else:
            group_indices = list(range(N))
            advantages = all_advantages

        # Compute log-likelihoods for the group with gradients
        log_likelihoods = self._compute_variant_log_likelihoods_for_grpo(
            input_ids=input_ids,
            completion_ids=completion_ids,
            group_indices=group_indices,
        )

        # Compute GRPO loss: -E[advantage * log_prob]
        # This encourages higher likelihood for higher-reward variants
        grpo_loss = -(advantages.to(log_likelihoods.device) * log_likelihoods).mean()

        # Optional: Add KL regularization to reference model
        # Uses proper token-level KL divergence over the full vocabulary distribution
        kl_loss = torch.tensor(0.0, device=grpo_loss.device)
        if self.grpo_use_reference_model and self.grpo_beta > 0:
            ref_model = self._get_reference_model()
            if ref_model is not None:
                # Compute proper token-level KL divergence: D_KL(policy || reference)
                # This compares the full vocabulary distribution at each position,
                # not just the log-likelihood of observed tokens
                kl_loss = self._compute_token_level_kl_divergence(
                    ref_model=ref_model,
                    input_ids=input_ids,
                    completion_ids=completion_ids,
                    group_indices=group_indices,
                )

        # Total loss
        total_loss = grpo_loss + self.grpo_beta * kl_loss

        # Logging
        self.log(
            "train/grpo_loss",
            grpo_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/grpo_kl_loss",
            kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/grpo_total_loss",
            total_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/grpo_mean_advantage",
            advantages.mean().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/grpo_mean_log_likelihood",
            log_likelihoods.mean().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/grpo_group_size",
            float(len(group_indices)),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

        # Compute and log correlations between log_likelihoods and advantages
        ll_np = log_likelihoods.detach().cpu().float().numpy()
        adv_np = advantages.detach().cpu().float().numpy()

        # Only compute correlations if there's variance in both arrays
        if ll_np.std() > 1e-8 and adv_np.std() > 1e-8:
            spearman_corr, _ = spearmanr(ll_np, adv_np)
            pearson_corr, _ = pearsonr(ll_np, adv_np)
        else:
            spearman_corr = 0.0
            pearson_corr = 0.0

        self.log(
            "train/grpo_ll_advantage_spearman",
            spearman_corr,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/grpo_ll_advantage_pearson",
            pearson_corr,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        # Log dataset sample counts
        self.log_train_dataset_sample_counts(batch)

        return total_loss

    def online_grpo_training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Online GRPO training step using HMM profile rewards.

        Implements proper Group Relative Policy Optimization (PPO-style) where:
          1. Generate G sequences from the current (old) policy π_old.
          2. Score them against a Pfam HMM profile to get rewards.
          3. Compute group-relative advantages Aᵢ.
          4. Re-compute log π_θ(yᵢ|x) under the *current* policy (with gradients).
          5. Form the importance ratio  rᵢ = π_θ / π_old  and apply PPO clipping:
                L = −E[min(rᵢ·Aᵢ, clip(rᵢ, 1−ε, 1+ε)·Aᵢ)]
          6. Optionally add a KL penalty to a frozen reference model.

        Args:
            batch: Dictionary containing:
                - input_ids: MSA context tokens (1, L_context)
                - hmm_family_id: StringObject with the Pfam family ID
            batch_idx: Batch index

        Returns:
            loss: GRPO loss value
        """
        assert self._hmm_scorer is not None, (
            "HMM reward scorer not available. Ensure a PfamHMMGRPODataset is "
            "included in the data config and grpo_enabled=True."
        )

        input_ids = batch["input_ids"]  # (1, L_context)
        family_id = batch["hmm_family_id"].text[0]

        # ------------------------------------------------------------------
        # Step 1: Generate sequences from current policy (no gradients)
        # ------------------------------------------------------------------
        sampling_kwargs = {}
        if self.grpo_hmm_temperature is not None:
            sampling_kwargs["temperature"] = self.grpo_hmm_temperature
        if self.grpo_hmm_top_p is not None:
            sampling_kwargs["top_p"] = self.grpo_hmm_top_p
        sep_id = self.tokenizer.sep_token_id
        if self.grpo_hmm_max_generated_length is not None:
            max_generated_length = self.grpo_hmm_max_generated_length
        else:
            sep_positions = torch.where(input_ids == sep_id)[1]
            sequence_lengths = torch.diff(sep_positions, prepend=sep_positions.new_zeros(1))
            max_generated_length = int(sequence_lengths.max().item() * 1.2)
        with torch.no_grad():
            generated_tokens, gen_scores, old_per_token_lps, old_per_token_mask = (
                self._sample_seqs(
                    input_ids,
                    num_samples=self.grpo_group_size,
                    max_tokens=self.grpo_max_tokens,
                    max_generated_length=max_generated_length,
                    repeat_guard=True,
                    suppress_bad_words=False,  # use raw logits so gen_scores match scoring path
                    return_per_token_log_probs=True,
                    max_retries=0,
                    **sampling_kwargs,
                )
            )
            # generated_tokens: (G, L_gen) — on CPU
            # gen_scores: List[float] of length G — mean per-token log-probs (for logging)
            # old_per_token_lps: (G, L_gen) — per-token log-probs under π_old
            # old_per_token_mask: (G, L_gen) — True for valid (non-pad) tokens

        # ------------------------------------------------------------------
        # Step 2: Decode to amino acid strings and score with HMM
        # ------------------------------------------------------------------
        generated_seqs = self.tokenizer.decode_tokens(
            generated_tokens.to(self.device)
        )
        print(generated_seqs[0])
        # decode_tokens returns List[str] when each row has a single sequence
        if isinstance(generated_seqs[0], list):
            generated_seqs = [s[0] if s else "" for s in generated_seqs]

        rewards_np = self._hmm_scorer.score_sequences(
            family_id,
            generated_seqs,
            length_penalty=self.grpo_hmm_length_penalty,
        )
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=self.device)

        # ------------------------------------------------------------------
        # Step 2b: Apply reward penalties for bad tokens and missing [SEP]
        # ------------------------------------------------------------------
        pad_id = self.tokenizer.pad_token_id
        bad_token_ids = set()
        for ch in "XxBJOUZbjou":
            tid = self.tokenizer.convert_tokens_to_ids(ch)
            if tid != self.tokenizer.unk_token_id:
                bad_token_ids.add(tid)
        # lowercase AA tokens (structure tokens)
        for ch in aa_letters_lower:
            tid = self.tokenizer.convert_tokens_to_ids(ch)
            if tid != self.tokenizer.unk_token_id:
                bad_token_ids.add(tid)
        # special tokens that should never appear mid-sequence
        for tid in self.tokenizer.all_special_ids:
            if tid not in (sep_id, pad_id):
                bad_token_ids.add(tid)

        bad_token_penalty = 5.0   # per bad token found in a sequence
        no_sep_penalty = 20.0     # for sequences that never generated [SEP]

        penalty_applied = torch.zeros_like(rewards)
        for i in range(generated_tokens.shape[0]):
            row = generated_tokens[i]
            # Check for missing [SEP] (sequence didn't terminate properly)
            valid_tokens = row[row != pad_id]
            if len(valid_tokens) == 0 or int(valid_tokens[-1].item()) != sep_id:
                penalty_applied[i] += no_sep_penalty
            # Count bad tokens in the generated sequence
            for tid in row.tolist():
                if tid == pad_id:
                    break
                if tid in bad_token_ids:
                    penalty_applied[i] += bad_token_penalty

        rewards = rewards - penalty_applied

        # ------------------------------------------------------------------
        # Step 2c: Check if any sequence was detected by the HMM
        # ------------------------------------------------------------------
        # Use raw HMM scores (before penalties) to determine detection.
        # rewards_np > 0 means the HMM matched the sequence.
        hmm_detected = (rewards_np > 0).any()
        frac_detected = float((rewards_np > 0).mean())

        if not hmm_detected:
            # No sequence matched the HMM — skip GRPO + KL entirely.
            # Returning None tells Lightning to skip the backward pass and
            # optimizer step for this batch (no gradient update).

            # Log NaN for GRPO loss to make skipped steps visible.
            # All on_step / on_epoch / prog_bar args MUST match the normal path
            # exactly, otherwise Lightning raises MisconfigurationException.
            self.log("train/hmm_grpo_loss", float("nan"),
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/hmm_grpo_kl_loss", float("nan"),
                     on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train/hmm_grpo_total_loss", float("nan"),
                     on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train/hmm_grpo_mean_reward", rewards.mean().item(),
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/hmm_grpo_max_reward", rewards.max().item(),
                     on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log("train/hmm_grpo_min_reward", rewards.min().item(),
                     on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log("train/hmm_grpo_frac_detected", frac_detected,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/hmm_grpo_mean_seq_length",
                     float(np.mean([len(s) for s in generated_seqs])),
                     on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log("train/hmm_grpo_mean_penalty", penalty_applied.mean().item(),
                     on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log("train/hmm_grpo_frac_no_sep",
                     float((penalty_applied >= no_sep_penalty).float().mean().item()),
                     on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log_train_dataset_sample_counts(batch)
            return None

        # ------------------------------------------------------------------
        # Step 3: Compute advantages
        # ------------------------------------------------------------------
        advantages = self._compute_grpo_advantages(rewards)

        # ------------------------------------------------------------------
        # Step 4: Build completion_ids for log-likelihood computation
        # ------------------------------------------------------------------
        # The prompt was encoded with add_final_sep=True (so the model sees
        # ...seqN[SEP] and generates a fresh sequence).  _sample_seqs slices
        # off the prompt, so generated_tokens starts with the first AA —
        # it does NOT include the leading [SEP].
        #
        # _score_seqs_kv_cache expects:
        #   • input_ids  WITHOUT a trailing [SEP]
        #   • completion_ids that START with [SEP]
        # (same convention used by the ProteinGym GRPO path).
        #
        # So we:
        #   a) strip the trailing [SEP] from input_ids
        #   b) prepend [SEP] to each generated completion
        

        # (a) Remove trailing SEP from context
        assert int(input_ids[0, -1].item()) == sep_id, (
            "Expected input_ids to end with SEP token for online GRPO; "
            f"got token id {int(input_ids[0, -1].item())}"
        )
        input_ids_for_scoring = input_ids[:, :-1]  # (1, L_context - 1)

        # (b) Prepend SEP to each generated sequence
        gen_on_device = generated_tokens.to(self.device)  # (N, L_gen)
        sep_prefix = torch.full(
            (gen_on_device.shape[0], 1), sep_id,
            dtype=gen_on_device.dtype, device=self.device,
        )
        completion_ids = torch.cat([sep_prefix, gen_on_device], dim=1)  # (N, 1+L_gen)
        completion_ids = completion_ids.unsqueeze(0)  # (1, N, 1+L_gen)

        # ------------------------------------------------------------------
        # Step 5: Compute per-token log π_θ(yᵢ|x) under current policy
        # ------------------------------------------------------------------
        new_per_token_lps, new_per_token_mask = (
            self._compute_per_token_log_probs_for_grpo(
                input_ids=input_ids_for_scoring,
                completion_ids=completion_ids,
            )
        )
        # new_per_token_lps: (G, L_gen) — per-token log-probs under π_θ (with grads)
        # new_per_token_mask: (G, L_gen) — validity mask from scoring path

        # ------------------------------------------------------------------
        # Step 6: Per-token PPO-style clipped GRPO loss
        # ------------------------------------------------------------------
        # old_per_token_lps: (G, L_gen) — per-token log-probs under π_old
        # new_per_token_lps: (G, L_gen) — per-token log-probs under π_θ
        # Ratios and clipping are computed per-token, then averaged.
        old_lps = old_per_token_lps.to(self.device).detach()  # (G, T)
        valid_mask = old_per_token_mask.to(self.device) & new_per_token_mask  # (G, T)

        per_token_log_ratio = new_per_token_lps - old_lps  # (G, T)
        per_token_ratio = torch.exp(per_token_log_ratio)    # (G, T)

        eps = self.grpo_clip_ratio  # e.g. 0.2
        clipped_ratio = torch.clamp(per_token_ratio, 1.0 - eps, 1.0 + eps)

        # Advantages are per-sequence; broadcast to per-token
        adv = advantages.unsqueeze(1)  # (G, 1)
        surr1 = per_token_ratio * adv   # (G, T)
        surr2 = clipped_ratio * adv     # (G, T)
        per_token_obj = torch.min(surr1, surr2)  # (G, T)

        # Average over valid tokens per sequence, then over sequences
        num_valid = valid_mask.float().sum(dim=1).clamp(min=1)  # (G,)
        per_seq_obj = (per_token_obj * valid_mask.float()).sum(dim=1) / num_valid
        grpo_loss = -per_seq_obj.mean()

        # Pre-compute per-sequence mean log-likelihoods for logging / correlation
        with torch.no_grad():
            mean_ll_per_seq = (
                (new_per_token_lps.detach() * valid_mask.float()).sum(dim=1)
                / num_valid
            )  # (G,)

        # Optional KL regularization to frozen reference model
        # (separate from the clipping — this penalises drift from the *initial*
        # pre-trained model, not from the generating policy of this step)
        kl_loss = torch.tensor(0.0, device=grpo_loss.device)
        if self.grpo_use_reference_model and self.grpo_beta > 0:
            ref_model = self._get_reference_model()
            if ref_model is not None:
                kl_loss = self._compute_token_level_kl_divergence(
                    ref_model=ref_model,
                    input_ids=input_ids_for_scoring,
                    completion_ids=completion_ids,
                    group_indices=list(range(completion_ids.shape[1])),
                )

        total_loss = grpo_loss + self.grpo_beta * kl_loss

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        self.log(
            "train/hmm_grpo_loss",
            grpo_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_kl_loss",
            kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_total_loss",
            total_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_mean_reward",
            rewards.mean().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_max_reward",
            rewards.max().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_min_reward",
            rewards.min().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_mean_advantage",
            advantages.mean().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_mean_log_likelihood",
            mean_ll_per_seq.mean().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        # Log per-token ratio diagnostics — useful for monitoring PPO health
        with torch.no_grad():
            ratio_detached = per_token_ratio.detach()
            # Clip fraction: fraction of *valid tokens* that were clipped
            ratio_valid = ratio_detached[valid_mask]
            clipped_frac = (
                (ratio_valid < 1.0 - eps) | (ratio_valid > 1.0 + eps)
            ).float().mean().item() if ratio_valid.numel() > 0 else 0.0
        self.log(
            "train/hmm_grpo_mean_ratio",
            ratio_valid.mean().item() if ratio_valid.numel() > 0 else 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_clip_fraction",
            clipped_frac,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_group_size",
            float(self.grpo_group_size),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_mean_seq_length",
            float(np.mean([len(s) for s in generated_seqs])),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

        # Log fraction of sequences that scored > 0 (detected by HMM)
        self.log(
            "train/hmm_grpo_frac_detected",
            frac_detected,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # Log penalty statistics
        self.log(
            "train/hmm_grpo_mean_penalty",
            penalty_applied.mean().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_frac_no_sep",
            float((penalty_applied >= no_sep_penalty).float().mean().item()),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        # Correlation between per-sequence mean log-likelihoods and HMM rewards
        ll_np = mean_ll_per_seq.cpu().float().numpy()
        rew_np = rewards.detach().cpu().float().numpy()
        if ll_np.std() > 1e-8 and rew_np.std() > 1e-8:
            sp_corr, _ = spearmanr(ll_np, rew_np)
            pe_corr, _ = pearsonr(ll_np, rew_np)
        else:
            sp_corr = 0.0
            pe_corr = 0.0
        self.log(
            "train/hmm_grpo_ll_reward_spearman",
            sp_corr,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/hmm_grpo_ll_reward_pearson",
            pe_corr,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "train/hmm_family_id",
            0.0,  # placeholder — family name logged separately
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

        self.log_train_dataset_sample_counts(batch)

        return total_loss

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

            policy_context_outputs = self.model(input_ids=input_ids, use_cache=True)
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

    # =========================================================================
    # PETase GRPO Training (Online Reward Computation)
    # =========================================================================

    def petase_grpo_training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """GRPO training step with ONLINE reward computation via ESMfold + energies.

        Unlike grpo_training_step() which uses pre-computed DMS_scores,
        this generates sequences, folds them, and computes energy rewards.

        Args:
            batch: Dictionary containing:
                - input_ids: Prompt tokens (1, L_context) - could be seed seq or MSA context
            batch_idx: Batch index

        Returns:
            loss: GRPO loss value
        """
        # Check required components are initialized
        if not hasattr(self, "_petase_folder") or self._petase_folder is None:
            raise RuntimeError(
                "PETase folder not initialized. Call setup_petase_training() first."
            )

        prompt_ids = batch["input_ids"]  # (1, L_context)

        # 1. Generate N candidate sequences
        with torch.no_grad():
            generated_ids, log_scores = self._sample_seqs(
                prompt_ids,
                num_samples=self.grpo_group_size,
                max_tokens=self._petase_max_tokens,
                max_generated_length=self._petase_max_length,
                temperature=getattr(self, "_petase_temperature", 1.0),
            )

        # 2. Decode to amino acid sequences
        sequences = []
        for ids in generated_ids:
            # Remove padding and special tokens
            seq = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            seq = seq.replace(" ", "").replace("-", "")
            sequences.append(seq)

        # 3. Filter out empty or too short sequences
        valid_indices = []
        valid_sequences = []
        for i, seq in enumerate(sequences):
            if len(seq) >= self._petase_min_length:
                valid_indices.append(i)
                valid_sequences.append(seq)

        if len(valid_sequences) == 0:
            # No valid sequences - return zero loss
            log.warning("PETase GRPO: No valid sequences generated")
            return torch.tensor(0.0, device=prompt_ids.device, requires_grad=True)

        # 4. Fold structures (parallel across GPUs)
        with torch.no_grad():
            fold_results = self._petase_folder.fold_batch(
                valid_sequences,
                max_length=self._petase_max_length,
            )

        # 5. Compute energy-based rewards
        rewards = self._compute_petase_energy_rewards(valid_sequences, fold_results)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        # 6. Compute GRPO advantages
        advantages = self._compute_grpo_advantages(rewards)

        # 7. Prepare completion_ids for log-likelihood computation
        # Convert generated_ids to completion_ids format: (1, N, L)
        # Only include valid sequences
        valid_generated_ids = generated_ids[valid_indices]
        completion_ids = valid_generated_ids.unsqueeze(0).to(prompt_ids.device)

        # 8. Compute log-likelihoods with gradients
        log_likelihoods = self._compute_variant_log_likelihoods_for_grpo(
            input_ids=prompt_ids,
            completion_ids=completion_ids,
            group_indices=list(range(len(valid_sequences))),
        )

        # 9. GRPO loss
        grpo_loss = -(advantages.to(log_likelihoods.device) * log_likelihoods).mean()

        # 10. KL penalty
        kl_loss = torch.tensor(0.0, device=grpo_loss.device)
        if self.grpo_use_reference_model and self.grpo_beta > 0:
            ref_model = self._get_reference_model()
            if ref_model is not None:
                kl_loss = self._compute_token_level_kl_divergence(
                    ref_model=ref_model,
                    input_ids=prompt_ids,
                    completion_ids=completion_ids,
                    group_indices=list(range(len(valid_sequences))),
                )

        total_loss = grpo_loss + self.grpo_beta * kl_loss

        # Logging
        self._log_petase_metrics(
            grpo_loss, kl_loss, total_loss, advantages, log_likelihoods,
            rewards, valid_sequences, fold_results
        )

        return total_loss

    def setup_petase_training(
        self,
        template_pdb_path: str,
        excluded_volume_pdb_path: str,
        backbone_only_residues: Optional[List[int]] = None,
        catalytic_residues: Optional[List[int]] = None,
        min_length: int = 100,
        max_length: int = 400,
        max_tokens: int = 500,
        temperature: float = 1.0,
        num_fold_gpus: Optional[int] = None,
        energy_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize PETase GRPO training components.

        Args:
            template_pdb_path: Path to PDB with important residues
            excluded_volume_pdb_path: Path to PDB with substrate excluded volume
            backbone_only_residues: Residue IDs to match only backbone N atoms (oxyanion hole)
            catalytic_residues: Residue IDs for catalytic triad
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            num_fold_gpus: Number of GPUs for ESMfold
            energy_weights: Dict of energy term weights
        """
        from src.energies import (
            ImportantResidueSet,
            TemplateMatchEnergy,
            ExcludedVolumeEnergy,
            SizeEnergy,
        )
        from src.folding import MultiGPUESMFold

        # Default backbone-only residues (oxyanion hole)
        if backbone_only_residues is None:
            backbone_only_residues = [87, 161]  # Y87, M161

        # Default catalytic residues
        if catalytic_residues is None:
            catalytic_residues = [160, 237]  # S160, H237

        # Default energy weights
        default_weights = {
            "template_match": 5.0,
            "excluded_volume": 2.0,
            "size": 0.5,
            "plddt": 1.0,
        }
        if energy_weights is not None:
            default_weights.update(energy_weights)
        self._petase_energy_weights = default_weights

        # Parse template
        log.info(f"Loading PETase template from {template_pdb_path}")
        self._petase_template = ImportantResidueSet.from_pdb(
            template_pdb_path,
            backbone_only_residues=backbone_only_residues,
            catalytic_residues=catalytic_residues,
        )

        # Initialize energy terms
        self._template_energy = TemplateMatchEnergy(
            template=self._petase_template,
            weight=default_weights["template_match"],
        )

        self._excluded_volume_energy = ExcludedVolumeEnergy.from_pdb(
            excluded_volume_pdb_path,
            weight=default_weights["excluded_volume"],
        )

        self._size_energy = SizeEnergy(
            min_length=min_length,
            weight=default_weights["size"],
        )

        # Initialize multi-GPU folder
        log.info(f"Initializing MultiGPUESMFold with {num_fold_gpus} GPUs")
        self._petase_folder = MultiGPUESMFold(num_gpus=num_fold_gpus)

        # Store config
        self._petase_min_length = min_length
        self._petase_max_length = max_length
        self._petase_max_tokens = max_tokens
        self._petase_temperature = temperature

        log.info("PETase GRPO training initialized")

    def _compute_petase_energy_rewards(
        self,
        sequences: List[str],
        fold_results: List,
    ) -> List[float]:
        """Compute energy-based rewards for PETase sequences.

        Reward = -(template_rmsd + excluded_vol + size_penalty) + plddt_bonus

        Args:
            sequences: List of amino acid sequences
            fold_results: List of FoldingResult from ESMfold

        Returns:
            List of reward values (higher is better)
        """
        rewards = []

        for seq, fold_result in zip(sequences, fold_results):
            # Skip if folding failed
            if fold_result.mean_plddt < 1.0:
                rewards.append(-100.0)
                continue

            # Get structure data from folding result
            coords = fold_result.coords  # (L, 37, 3)
            plddt = fold_result.plddt

            # Extract atom information for energy calculation
            # Use CA coordinates for simplicity in template matching
            ca_coords = fold_result.get_ca_coords()

            # Build residue info lists
            L = len(seq)
            res_ids = list(range(1, L + 1))
            res_names = [_aa_to_three_letter(aa) for aa in seq]
            atom_names = ["CA"] * L
            elements = ["C"] * L

            # 1. Template match energy (use CA for sliding window match)
            template_value, template_weighted, _ = self._template_energy.compute(
                structure_coords=ca_coords,
                structure_res_ids=res_ids,
                structure_res_names=res_names,
                structure_atom_names=atom_names,
            )

            # 2. Excluded volume energy
            excluded_value, excluded_weighted, _ = self._excluded_volume_energy.compute(
                structure_coords=ca_coords,
                structure_elements=elements,
            )

            # 3. Size energy
            size_value, size_weighted, _ = self._size_energy.compute(len(seq))

            # 4. pLDDT bonus (higher is better)
            plddt_bonus = (fold_result.mean_plddt / 100.0) * self._petase_energy_weights["plddt"]

            # Combine into reward (negative energies, positive pLDDT)
            reward = -(template_weighted + excluded_weighted + size_weighted) + plddt_bonus
            rewards.append(reward)

        return rewards

    def _log_petase_metrics(
        self,
        grpo_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        total_loss: torch.Tensor,
        advantages: torch.Tensor,
        log_likelihoods: torch.Tensor,
        rewards: torch.Tensor,
        sequences: List[str],
        fold_results: List,
    ):
        """Log PETase GRPO training metrics."""
        self.log(
            "train/petase_grpo_loss",
            grpo_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/petase_kl_loss",
            kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/petase_total_loss",
            total_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/petase_mean_reward",
            rewards.mean().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/petase_mean_advantage",
            advantages.mean().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/petase_mean_log_likelihood",
            log_likelihoods.mean().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/petase_mean_seq_length",
            float(np.mean([len(s) for s in sequences])),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/petase_mean_plddt",
            float(np.mean([f.mean_plddt for f in fold_results])),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/petase_group_size",
            float(len(sequences)),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

    # =========================================================================
    # Mipa GRPO Training (TM-score Reward Computation)
    # =========================================================================

    def setup_mipa_training(
        self,
        target_pdb_path: str,
        reasoning_mode: bool = False,
        num_reasoning_seqs: int = 3,
        max_length: int = 400,
        max_tokens: int = 600,
        temperature: float = 1.0,
        num_fold_gpus: Optional[int] = None,
        use_rank_local_folding: bool = True,
        plddt_weight: float = 0.1,
        length_penalty_threshold: int = 1048,
        tmalign_binary: str = "TMalign",
        batch_generation: bool = True,
        generation_batch_size: int = 8,
        evolving_prompt_enabled: bool = False,
        evolving_prompt_update_interval: int = 250,
        evolving_prompt_min_tm_score: float = 0.3,
    ):
        """Initialize Mipa GRPO training components.

        Args:
            target_pdb_path: Path to the target PDB structure (mipa_8YTU.pdb)
            reasoning_mode: Whether to use reasoning chain generation
            num_reasoning_seqs: Number of intermediate reasoning sequences (default: 3)
            max_length: Maximum sequence length
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            num_fold_gpus: Number of GPUs for ESMfold (ignored if use_rank_local_folding=True)
            use_rank_local_folding: If True, each DDP rank uses only its own GPU for folding
                to avoid deadlocks. This is the recommended setting for multi-GPU training.
            plddt_weight: Weight for pLDDT bonus in reward
            length_penalty_threshold: Sequences longer than this get penalized
            tmalign_binary: Path or name of TMalign binary
            batch_generation: If True, use batched model.generate() calls for speedup
            generation_batch_size: Number of sequences to generate in parallel per batch
            evolving_prompt_enabled: If True, update prompt to high-reward sequences periodically
            evolving_prompt_update_interval: Steps between prompt updates (default: 250)
            evolving_prompt_min_tm_score: Minimum TM-score to be a candidate (default: 0.3)
        """
        from src.structure import TMAlignScorer

        log.info(f"Initializing Mipa GRPO training with target: {target_pdb_path}")

        # Initialize TM-score calculator
        self._mipa_tm_scorer = TMAlignScorer(
            reference_pdb_path=target_pdb_path,
            tmalign_binary=tmalign_binary,
        )
        log.info(f"TMAlignScorer initialized, reference length: {self._mipa_tm_scorer.reference_length}")

        # Store folding config - folder will be initialized lazily on first use
        # This is necessary because trainer.local_rank isn't available until training starts
        self._mipa_folder = None  # Will be initialized in _ensure_mipa_folder_initialized()
        self._mipa_num_fold_gpus = num_fold_gpus
        self._mipa_use_rank_local_folding = use_rank_local_folding

        # Store configuration
        self._mipa_reasoning_mode = reasoning_mode
        self._mipa_num_reasoning_seqs = num_reasoning_seqs
        self._mipa_max_length = max_length
        self._mipa_max_tokens = max_tokens
        self._mipa_temperature = temperature
        self._mipa_plddt_weight = plddt_weight
        self._mipa_length_penalty_threshold = length_penalty_threshold
        self._mipa_batch_generation = batch_generation
        self._mipa_generation_batch_size = generation_batch_size

        # Evolving prompt configuration
        self._evolving_prompt_enabled = evolving_prompt_enabled
        self._evolving_prompt_update_interval = evolving_prompt_update_interval
        self._evolving_prompt_min_tm_score = evolving_prompt_min_tm_score
        self._evolving_prompt_current_sequence = None
        self._evolving_prompt_current_tokens = None
        self._evolving_prompt_current_reward = None
        self._evolving_prompt_candidate_buffer = []

        log.info(f"Mipa GRPO training initialized (reasoning_mode={reasoning_mode}, batch_generation={batch_generation})")
        if evolving_prompt_enabled:
            log.info(f"Evolving prompts enabled: update every {evolving_prompt_update_interval} steps, min TM-score={evolving_prompt_min_tm_score}")

    def _ensure_mipa_folder_initialized(self):
        """Lazily initialize the ESMfold folder on first use.

        This is called during the first training step when trainer.local_rank is available.
        """
        if self._mipa_folder is not None:
            return

        from src.folding import MultiGPUESMFold

        # Determine GPU IDs for folding
        # In DDP training, each rank should only use its own GPU to avoid deadlocks
        fold_gpu_ids = None
        if self._mipa_use_rank_local_folding and hasattr(self, 'trainer') and self.trainer is not None:
            local_rank = getattr(self.trainer, 'local_rank', 0)
            fold_gpu_ids = [local_rank]
            log.info(f"Rank-local folding enabled: using GPU {local_rank} for folding")
        elif self._mipa_num_fold_gpus is not None:
            fold_gpu_ids = list(range(self._mipa_num_fold_gpus))
            log.info(f"Using GPUs {fold_gpu_ids} for folding")

        # Initialize multi-GPU folder
        log.info(f"Initializing MultiGPUESMFold with gpu_ids={fold_gpu_ids}")
        self._mipa_folder = MultiGPUESMFold(gpu_ids=fold_gpu_ids)

    def mipa_grpo_training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """GRPO training step with TM-score rewards for Mipa structure matching.

        Two modes:
        - Direct: Generate 1 sequence, score with TM-score
        - Reasoning: Generate continuously until 4 [SEP] tokens, score final only

        Args:
            batch: Dictionary containing:
                - input_ids: Prompt tokens (1, L_context)
            batch_idx: Batch index

        Returns:
            loss: GRPO loss value
        """
        # Check required components are initialized
        if self._mipa_tm_scorer is None:
            raise RuntimeError(
                "Mipa training not initialized. Call setup_mipa_training() first."
            )

        # Lazily initialize folder on first use (when trainer.local_rank is available)
        self._ensure_mipa_folder_initialized()

        # Use evolved prompt if available, otherwise use batch prompt
        if self._evolving_prompt_enabled and self._evolving_prompt_current_tokens is not None:
            prompt_ids = self._evolving_prompt_current_tokens.to(batch["input_ids"].device)
        else:
            prompt_ids = batch["input_ids"]  # (1, L_context)

        # Set per-rank seed for generation diversity in DDP
        # Without this, all ranks generate identical sequences due to global seed
        # Use global_step + batch_idx to ensure unique seeds across all steps:
        # - global_step ensures different seeds after each optimizer update
        # - batch_idx handles accumulate_grad_batches > 1 (multiple batches per step)
        # - rank ensures different GPUs generate different sequences
        if hasattr(self, "trainer") and self.trainer is not None:
            rank = self.trainer.global_rank
            global_step = self.trainer.global_step
            generation_seed = hash((42, rank, global_step, batch_idx)) % (2**32)
            torch.manual_seed(generation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(generation_seed)

        # 1. Generate sequences
        with torch.no_grad():
            if self._mipa_reasoning_mode:
                (
                    full_tokens,
                    final_tokens,
                    gen_scores,
                    old_lps,
                    old_mask,
                    incomplete_mask,
                ) = self._sample_seqs_reasoning(
                    prompt_ids,
                    num_samples=self.grpo_group_size,
                    max_tokens=self._mipa_max_tokens,
                    num_reasoning_seqs=self._mipa_num_reasoning_seqs,
                    max_generated_length=self._mipa_max_length,
                    temperature=self._mipa_temperature,
                    return_per_token_log_probs=True,
                )
            else:
                (
                    final_tokens,
                    gen_scores,
                    old_lps,
                    old_mask,
                ) = self._sample_seqs(
                    prompt_ids,
                    num_samples=self.grpo_group_size,
                    max_tokens=self._mipa_max_tokens,
                    max_generated_length=self._mipa_max_length,
                    temperature=self._mipa_temperature,
                    return_per_token_log_probs=True,
                    batch_generation=self._mipa_batch_generation,
                    generation_batch_size=self._mipa_generation_batch_size,
                )
                incomplete_mask = torch.zeros(self.grpo_group_size, dtype=torch.bool)

        # 2. Decode final sequences
        sequences = []
        for ids in final_tokens:
            seq = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            seq = seq.replace(" ", "").replace("-", "")
            sequences.append(seq)

        # Store sequences for SequenceDiversityCallback (debug/testing)
        self._debug_generated_sequences = sequences.copy()
        self._debug_generated_tokens = final_tokens.clone()

        # Log first two generated sequences on rank 0 only (no sync to avoid hangs)
        rank = self.trainer.global_rank if self.trainer is not None else 0
        if rank == 0:
            log.info(
                f"Generated {len(sequences)} sequences. First two:\n"
                f"  [0] ({len(sequences[0])}aa): {sequences[0]}\n"
                f"  [1] ({len(sequences[1])}aa): {sequences[1]}"
            )

        # Debug: Print progress for all ranks
        import sys
        print(f"[DEBUG rank={rank}] After generation, {len(sequences)} seqs, lengths: {[len(s) for s in sequences]}", flush=True)
        sys.stdout.flush()

        # 3. Compute penalties
        penalties = self._compute_mipa_penalties(
            final_tokens, sequences, incomplete_mask
        )

        # 4. Fold valid sequences with ESMfold
        valid_mask = penalties < 50  # Not too heavily penalized
        valid_indices = [i for i, v in enumerate(valid_mask.tolist()) if v]
        valid_sequences = [sequences[i] for i in valid_indices]

        # Initialize rewards and alignment metrics
        rewards = torch.zeros(len(sequences), device=prompt_ids.device)
        tm_scores = [0.0] * len(sequences)
        plddts = [0.0] * len(sequences)
        rmsds = [0.0] * len(sequences)
        aligned_lengths = [0] * len(sequences)
        seq_identities = [0.0] * len(sequences)

        if len(valid_sequences) > 0:
            # Fold structures
            print(f"[DEBUG rank={rank}] Starting fold_batch for {len(valid_sequences)} seqs", flush=True)
            with torch.no_grad():
                fold_results = self._mipa_folder.fold_batch(
                    valid_sequences,
                    max_length=self._mipa_max_length,
                )
            print(f"[DEBUG rank={rank}] Finished fold_batch", flush=True)

            # 5. Compute alignments (parallel TMalign calls)
            print(f"[DEBUG rank={rank}] Starting TMalign", flush=True)
            alignment_results = self._mipa_tm_scorer.align_batch(fold_results)
            print(f"[DEBUG rank={rank}] Finished TMalign", flush=True)

            # 6. Compute rewards: TM-score + scaled pLDDT - penalties
            for j, (idx, fr, align_result) in enumerate(zip(valid_indices, fold_results, alignment_results)):
                plddt_bonus = (fr.mean_plddt / 100.0) * self._mipa_plddt_weight
                rewards[idx] = align_result.tm_score + plddt_bonus - penalties[idx]
                tm_scores[idx] = align_result.tm_score
                plddts[idx] = fr.mean_plddt
                rmsds[idx] = align_result.rmsd
                aligned_lengths[idx] = align_result.aligned_length
                seq_identities[idx] = align_result.seq_identity
        else:
            fold_results = []

        # Apply penalties to invalid sequences
        for i in range(len(sequences)):
            if i not in valid_indices:
                rewards[i] = -penalties[i]

        # 7. Compute advantages
        advantages = self._compute_grpo_advantages(rewards)

        # 8. Compute per-token log-probs under current policy (with gradients)
        # Build completion_ids for scoring
        if self._mipa_reasoning_mode:
            # For reasoning mode, score the full chain (all tokens)
            completion_ids = full_tokens.unsqueeze(0).to(prompt_ids.device)
        else:
            completion_ids = final_tokens.unsqueeze(0).to(prompt_ids.device)

        # Compute log-likelihoods with gradients
        print(f"[DEBUG rank={rank}] Starting log_likelihoods computation", flush=True)
        log_likelihoods = self._compute_variant_log_likelihoods_for_grpo(
            input_ids=prompt_ids,
            completion_ids=completion_ids,
            group_indices=list(range(len(sequences))),
        )
        print(f"[DEBUG rank={rank}] Finished log_likelihoods computation", flush=True)

        # 9. GRPO loss
        grpo_loss = -(advantages.to(log_likelihoods.device) * log_likelihoods).mean()

        # 10. KL penalty
        kl_loss = torch.tensor(0.0, device=grpo_loss.device)
        if self.grpo_use_reference_model and self.grpo_beta > 0:
            print(f"[DEBUG rank={rank}] Starting KL divergence computation", flush=True)
            ref_model = self._get_reference_model()
            if ref_model is not None:
                kl_loss = self._compute_token_level_kl_divergence(
                    ref_model=ref_model,
                    input_ids=prompt_ids,
                    completion_ids=completion_ids,
                    group_indices=list(range(len(sequences))),
                )
            print(f"[DEBUG rank={rank}] Finished KL divergence computation", flush=True)

        total_loss = grpo_loss + self.grpo_beta * kl_loss

        # 11. Synchronize all ranks before logging to avoid sync_dist hangs
        # This is needed because folding times vary across sequences, causing ranks
        # to reach the logging call at different times
        print(f"[DEBUG rank={rank}] Before barrier", flush=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        print(f"[DEBUG rank={rank}] After barrier", flush=True)

        # 12. Logging
        self._log_mipa_metrics(
            grpo_loss=grpo_loss,
            kl_loss=kl_loss,
            total_loss=total_loss,
            advantages=advantages,
            log_likelihoods=log_likelihoods,
            rewards=rewards,
            tm_scores=tm_scores,
            plddts=plddts,
            rmsds=rmsds,
            aligned_lengths=aligned_lengths,
            seq_identities=seq_identities,
            sequences=sequences,
            incomplete_mask=incomplete_mask,
        )

        # 13. Collect candidates for evolving prompt (sequences with good TM-scores)
        if self._evolving_prompt_enabled:
            for i, (seq, tm, rew) in enumerate(zip(sequences, tm_scores, rewards.tolist())):
                if tm >= self._evolving_prompt_min_tm_score:
                    self._evolving_prompt_candidate_buffer.append({
                        'sequence': seq,
                        'reward': rew,
                        'tm_score': tm,
                    })
            # Keep top 1000 candidates by highest reward (not most recent)
            if len(self._evolving_prompt_candidate_buffer) > 1000:
                self._evolving_prompt_candidate_buffer = sorted(
                    self._evolving_prompt_candidate_buffer,
                    key=lambda x: x['reward'],
                    reverse=True
                )[:1000]

        return total_loss

    def _compute_mipa_penalties(
        self,
        tokens: torch.Tensor,
        sequences: List[str],
        incomplete_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute penalties for invalid sequences.

        Args:
            tokens: Generated tokens (N, L)
            sequences: Decoded amino acid sequences
            incomplete_mask: Boolean mask indicating sequences that didn't complete (reasoning mode)

        Returns:
            Tensor of penalties (N,)
        """
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        no_sep_penalty = 20.0
        length_penalty = 10.0
        bad_token_penalty = 5.0
        incomplete_penalty = 15.0  # Didn't generate enough SEPs in reasoning mode

        penalties = torch.zeros(len(sequences), device=tokens.device)

        for i, (row, seq) in enumerate(zip(tokens, sequences)):
            # Check for missing final [SEP]
            valid_tokens = row[row != pad_id]
            if len(valid_tokens) == 0 or int(valid_tokens[-1].item()) != sep_id:
                penalties[i] += no_sep_penalty

            # Check length > threshold
            if len(seq) > self._mipa_length_penalty_threshold:
                penalties[i] += length_penalty

            # Count bad tokens
            good_chars = set("ACDEFGHIKLMNPQRSTVWY") # todo this will not punsh multi-token special chars correctly
            bad_count = sum(1 for c in seq if c not in good_chars)
            penalties[i] += bad_count * bad_token_penalty

            # Penalty for incomplete reasoning chains
            if incomplete_mask[i]:
                penalties[i] += incomplete_penalty

        return penalties

    def _log_mipa_metrics(
        self,
        grpo_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        total_loss: torch.Tensor,
        advantages: torch.Tensor,
        log_likelihoods: torch.Tensor,
        rewards: torch.Tensor,
        tm_scores: List[float],
        plddts: List[float],
        rmsds: List[float],
        aligned_lengths: List[int],
        seq_identities: List[float],
        sequences: List[str],
        incomplete_mask: torch.Tensor,
    ):
        """Log Mipa GRPO training metrics."""
        self.log(
            "train/mipa_grpo_loss",
            grpo_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/mipa_kl_loss",
            kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_total_loss",
            total_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_mean_reward",
            rewards.mean().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/mipa_min_reward",
            rewards.min().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_max_reward",
            rewards.max().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_mean_tm_score",
            float(np.mean([t for t in tm_scores if t > 0]) if any(t > 0 for t in tm_scores) else 0.0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/mipa_min_tm_score",
            float(min(tm_scores)) if tm_scores else 0.0,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_max_tm_score",
            float(max(tm_scores)) if tm_scores else 0.0,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_mean_plddt",
            float(np.mean([p for p in plddts if p > 0]) if any(p > 0 for p in plddts) else 0.0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # TMalign alignment metrics
        self.log(
            "train/mipa_mean_rmsd",
            float(np.mean([r for r in rmsds if r > 0]) if any(r > 0 for r in rmsds) else 0.0),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_mean_aligned_length",
            float(np.mean([a for a in aligned_lengths if a > 0]) if any(a > 0 for a in aligned_lengths) else 0.0),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_mean_seq_identity",
            float(np.mean([s for s in seq_identities if s > 0]) if any(s > 0 for s in seq_identities) else 0.0),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_mean_advantage",
            advantages.mean().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_mean_log_likelihood",
            log_likelihoods.mean().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_mean_seq_length",
            float(np.mean([len(s) for s in sequences])),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        # Fraction of sequences that completed (for reasoning mode)
        frac_complete = 1.0 - incomplete_mask.float().mean().item()
        self.log(
            "train/mipa_frac_complete",
            frac_complete,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/mipa_group_size",
            float(len(sequences)),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

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
        # Save evolving prompt state if enabled
        if self._evolving_prompt_enabled and self._evolving_prompt_current_sequence is not None:
            checkpoint['evolving_prompt'] = {
                'current_sequence': self._evolving_prompt_current_sequence,
                'current_reward': self._evolving_prompt_current_reward,
                'candidate_buffer': self._evolving_prompt_candidate_buffer,
            }
            log.info(
                f"Saving evolving prompt state: {len(self._evolving_prompt_current_sequence)}aa sequence, "
                f"reward={self._evolving_prompt_current_reward}, "
                f"{len(self._evolving_prompt_candidate_buffer)} candidates in buffer"
            )

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

        # Restore evolving prompt state if present
        if 'evolving_prompt' in checkpoint:
            from src.data.objects import ProteinDocument

            evolving_state = checkpoint['evolving_prompt']
            self._evolving_prompt_current_sequence = evolving_state.get('current_sequence')
            self._evolving_prompt_current_reward = evolving_state.get('current_reward')
            self._evolving_prompt_candidate_buffer = evolving_state.get('candidate_buffer', [])
            # Re-tokenize the sequence (tokens need to be created at runtime)
            if self._evolving_prompt_current_sequence is not None:
                # Create a ProteinDocument (tokenizer.encode expects this)
                proteins = ProteinDocument(
                    sequences=[self._evolving_prompt_current_sequence],
                    identifier="evolved_prompt",
                )
                tokenized = self.tokenizer.encode(
                    proteins,
                    document_token="[RAW]",
                    add_final_sep=True,
                )
                self._evolving_prompt_current_tokens = torch.tensor(
                    tokenized.input_ids
                ).unsqueeze(0)
                log.info(
                    f"Restored evolving prompt: {len(self._evolving_prompt_current_sequence)}aa sequence, "
                    f"reward={self._evolving_prompt_current_reward}, "
                    f"{len(self._evolving_prompt_current_tokens[0])} tokens, "
                    f"{len(self._evolving_prompt_candidate_buffer)} candidates in buffer"
                )

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
        # Invalid sequences (no SEP, repetitive, too long) will be penalized by _compute_mipa_penalties
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

    def _sample_seqs_reasoning(
        self,
        input_ids: torch.Tensor,
        num_samples: int,
        max_tokens: int,
        num_reasoning_seqs: int = 3,
        max_generated_length: Optional[int] = None,
        temperature: float = 1.0,
        return_per_token_log_probs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate sequences with reasoning chain (continuous generation).

        Generates tokens continuously until (num_reasoning_seqs + 1) [SEP] tokens
        are seen, then extracts the final sequence (after the last reasoning [SEP]).

        Args:
            input_ids: Prompt tokens (1, L_prompt)
            num_samples: Number of samples to generate
            max_tokens: Maximum total tokens to generate
            num_reasoning_seqs: Number of intermediate reasoning sequences (default: 3)
            max_generated_length: Maximum length per sequence segment
            temperature: Sampling temperature
            return_per_token_log_probs: Whether to return per-token log probs

        Returns:
            full_tokens: All generated tokens (N, L_full)
            final_tokens: Only final sequence tokens (N, L_final)
            final_scores: Mean log-probs for full sequence
            per_token_lps: Per-token log probs tensor (if return_per_token_log_probs)
            per_token_mask: Validity mask tensor (if return_per_token_log_probs)
            incomplete_mask: Boolean mask indicating sequences that didn't complete
        """
        from transformers import StoppingCriteria

        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id
        required_seps = num_reasoning_seqs + 1  # e.g., 4 for default

        # Custom stopping criteria that counts SEP tokens
        class SEPCountCriteria(StoppingCriteria):
            def __init__(self, target_count, sep_id, prompt_length):
                self.target_count = target_count
                self.sep_id = sep_id
                self.prompt_length = prompt_length

            def __call__(self, gen_input_ids, scores, **kwargs):
                # Check generated portion only
                generated = gen_input_ids[:, self.prompt_length:]
                sep_counts = (generated == self.sep_id).sum(dim=1)
                return (sep_counts >= self.target_count).all()

        assert (
            input_ids.shape[0] == 1 and input_ids.ndim == 2
        ), "Only batch size 1 is supported for sampling; batch dim must be present"

        prompt_len = input_ids.shape[1]

        all_full_outputs: List[torch.Tensor] = []
        all_final_outputs: List[torch.Tensor] = []
        all_scores: List[float] = []
        all_per_token_lps: List[List[float]] = []
        all_incomplete: List[bool] = []

        # Generate one sample at a time
        for _ in tqdm.tqdm(range(num_samples), "Generating reasoning chains"):
            # Set up stopping criteria
            stopping = StoppingCriteriaList([
                SEPCountCriteria(required_seps, sep_token_id, prompt_len)
            ])

            # Generate with SEP count stopping
            gen_out = self.model.generate(
                input_ids=input_ids,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens,
                pad_token_id=pad_token_id,
                stopping_criteria=stopping,
            )

            seqs_full = gen_out.sequences  # (1, L_total)
            scores_list = gen_out.scores  # List[T] of (1, V)

            # Get generated portion
            full_generated = seqs_full[0, prompt_len:]

            # Count SEPs and extract final sequence
            sep_positions = (full_generated == sep_token_id).nonzero(as_tuple=True)[0]
            n_seps = len(sep_positions)

            incomplete = n_seps < required_seps
            all_incomplete.append(incomplete)

            # Extract final sequence: tokens after the (num_reasoning_seqs)th SEP
            # up to (and including) the final SEP
            if n_seps >= num_reasoning_seqs:
                # Start after the last "reasoning" SEP
                start_idx = int(sep_positions[num_reasoning_seqs - 1].item()) + 1
                # Include up to the final SEP or end
                if n_seps >= required_seps:
                    end_idx = int(sep_positions[required_seps - 1].item()) + 1
                else:
                    end_idx = len(full_generated)
                final_seq = full_generated[start_idx:end_idx]
            else:
                # Not enough SEPs - use whatever we have after the last SEP (or all)
                if n_seps > 0:
                    start_idx = int(sep_positions[-1].item()) + 1
                    final_seq = full_generated[start_idx:]
                else:
                    final_seq = full_generated

            all_full_outputs.append(full_generated.unsqueeze(0))
            all_final_outputs.append(final_seq.unsqueeze(0))

            # Compute mean log-prob for the full generated sequence
            total_logp = 0.0
            count = 0
            token_lps: List[float] = []
            T = len(scores_list)
            for t in range(T):
                if t < full_generated.shape[0]:
                    token_id = int(full_generated[t].item())
                    lp = F.log_softmax(scores_list[t], dim=-1)[0, token_id].item()
                    total_logp += float(lp)
                    token_lps.append(float(lp))
                    count += 1

            all_scores.append(total_logp / max(count, 1))
            all_per_token_lps.append(token_lps)

        # Pad and stack full outputs
        max_full_length = max([o.shape[1] for o in all_full_outputs])
        padded_full = torch.full(
            (num_samples, max_full_length), pad_token_id, dtype=torch.long
        )
        for i, o in enumerate(all_full_outputs):
            padded_full[i, : o.shape[1]] = o[0]

        # Pad and stack final outputs
        max_final_length = max([o.shape[1] for o in all_final_outputs])
        padded_final = torch.full(
            (num_samples, max_final_length), pad_token_id, dtype=torch.long
        )
        for i, o in enumerate(all_final_outputs):
            padded_final[i, : o.shape[1]] = o[0]

        incomplete_mask = torch.tensor(all_incomplete, dtype=torch.bool)

        if return_per_token_log_probs:
            per_token_lps_tensor = torch.zeros(num_samples, max_full_length)
            per_token_mask_tensor = torch.zeros(
                num_samples, max_full_length, dtype=torch.bool
            )
            for idx, lps in enumerate(all_per_token_lps):
                n_tok = len(lps)
                per_token_lps_tensor[idx, :n_tok] = torch.tensor(lps)
                per_token_mask_tensor[idx, :n_tok] = True
            return (
                padded_full,
                padded_final,
                all_scores,
                per_token_lps_tensor,
                per_token_mask_tensor,
                incomplete_mask,
            )

        return padded_full, padded_final, all_scores, incomplete_mask

    @torch.no_grad()
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
