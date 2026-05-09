"""
src/trainer.py
DPO training loop with gradient accumulation, LR scheduling, and checkpointing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer, get_cosine_schedule_with_warmup

from src.config import ExperimentConfig
from src.losses import compute_loss
from src.model import AutoModelForCausalLM, sequence_logprob, sequence_logprob_no_grad
from src.utils import average, checkpoint_dir, log


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DPOTrainer:
    """Trains a single (method, noise-level) configuration.

    Args:
        policy:     The LoRA-wrapped (or full) policy model to train.
        ref_model:  Frozen reference model; never updated.
        tokenizer:  Shared tokenizer.
        train_data: List of noisy preference dicts.
        cfg:        Full experiment config.
        device:     Target device string.
        flip_prob:  True noise rate for this run (passed to the loss).
        method:     Which DPO variant to use.
    """

    def __init__(
        self,
        policy: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        train_data: List[Dict],
        cfg: ExperimentConfig,
        device: str,
        flip_prob: float,
        method: str,
    ) -> None:
        self.policy    = policy
        self.ref       = ref_model
        self.tokenizer = tokenizer
        self.data      = train_data
        self.cfg       = cfg
        self.device    = device
        self.flip_prob = flip_prob
        self.method    = method

        tc = cfg.train
        self.beta       = tc.beta
        self.max_length = tc.max_seq_length
        self.max_grad_norm = tc.max_grad_norm

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad],
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )

        # LR scheduler (cosine with warmup)
        num_warmup = math.ceil(tc.warmup_ratio * tc.train_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=tc.train_steps,
        )

        self.accum_steps = tc.gradient_accumulation_steps
        self.train_steps = tc.train_steps
        self.save_steps  = tc.save_steps
        self.log_steps   = tc.logging_steps
        self.output_dir  = tc.output_dir

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """Run the full training loop and return summary metrics."""
        self.policy.train()
        losses: List[float] = []
        accum_loss = torch.tensor(0.0, device=self.device)
        global_step = 0
        data_iter   = self._infinite_iter()

        self.optimizer.zero_grad()

        for micro_step in range(self.train_steps * self.accum_steps):
            ex = next(data_iter)

            # Forward pass (policy, with grad)
            logp_c = sequence_logprob(
                self.policy, self.tokenizer, ex["chosen"], self.device, self.max_length
            )
            logp_r = sequence_logprob(
                self.policy, self.tokenizer, ex["rejected"], self.device, self.max_length
            )

            # Reference pass (frozen, no grad)
            ref_c = sequence_logprob_no_grad(
                self.ref, self.tokenizer, ex["chosen"], self.device, self.max_length
            )
            ref_r = sequence_logprob_no_grad(
                self.ref, self.tokenizer, ex["rejected"], self.device, self.max_length
            )

            loss = compute_loss(
                method     = self.method,
                logp_c     = logp_c,
                logp_r     = logp_r,
                ref_c      = ref_c,
                ref_r      = ref_r,
                beta       = self.beta,
                flip_prob  = self.flip_prob,
                noise_cfg  = self.cfg.noise,
            )

            # Accumulate; scale to average across accumulation steps
            (loss / self.accum_steps).backward()
            accum_loss = accum_loss + loss.detach()

            # Gradient step after accumulation_steps micro-steps
            if (micro_step + 1) % self.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                step_loss = (accum_loss / self.accum_steps).item()
                losses.append(step_loss)
                accum_loss = torch.tensor(0.0, device=self.device)
                global_step += 1

                if global_step % self.log_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    log.info(
                        f"[{self.method}] step={global_step:05d}/{self.train_steps}"
                        f"  loss={step_loss:.4f}  lr={lr:.2e}"
                    )

                if global_step % self.save_steps == 0:
                    self._save_checkpoint(global_step)

                if global_step >= self.train_steps:
                    break

        # Final checkpoint
        self._save_checkpoint("final")

        return {"avg_train_loss": average(losses), "losses": losses}

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _infinite_iter(self):
        """Cycle through training data indefinitely."""
        import random
        rng = random.Random(self.cfg.train.seed)
        data = list(self.data)
        while True:
            rng.shuffle(data)
            yield from data

    def _save_checkpoint(self, step) -> None:
        ckpt = Path(checkpoint_dir(self.output_dir, self.method, self.flip_prob)) / f"step_{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(str(ckpt))
        self.tokenizer.save_pretrained(str(ckpt))
        log.info(f"Checkpoint saved → {ckpt}")
