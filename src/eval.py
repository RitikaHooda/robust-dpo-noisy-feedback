"""
src/eval.py
Evaluation on the clean held-out set: preference margin and preference accuracy.

These match the metrics used in Table 2 of Hooda & Kumar (2025).

Preference margin  = avg(log π(yc | x) − log π(yr | x))   — higher is better
Preference accuracy = fraction of pairs where the policy prefers yc         — higher is better
"""

from __future__ import annotations

from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.model import AutoModelForCausalLM, sequence_logprob_no_grad
from src.utils import average, log


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    eval_data: List[Dict],
    device: str,
    max_length: int,
    desc: str = "Eval",
) -> Dict[str, float]:
    """Compute preference margin and accuracy on a clean held-out set.

    The eval set must NEVER have been noised — it is always the original,
    unmodified preference pairs.

    Args:
        model:      The policy model to evaluate.
        tokenizer:  Shared tokenizer.
        eval_data:  List of clean preference dicts with 'chosen' and 'rejected'.
        device:     Target device.
        max_length: Max token length for truncation.
        desc:       tqdm progress bar description.

    Returns:
        dict with keys: margin, accuracy, n_correct, n_total
    """
    model.eval()
    margins: List[float] = []

    with torch.no_grad():
        for ex in tqdm(eval_data, desc=desc, leave=False):
            lp_c = sequence_logprob_no_grad(
                model, tokenizer, ex["chosen"], device, max_length
            ).item()
            lp_r = sequence_logprob_no_grad(
                model, tokenizer, ex["rejected"], device, max_length
            ).item()
            margins.append(lp_c - lp_r)

    model.train()

    n_total   = len(margins)
    n_correct = sum(1 for m in margins if m > 0)
    avg_margin = average(margins)
    accuracy   = n_correct / n_total if n_total > 0 else float("nan")

    result = {
        "margin":    avg_margin,
        "accuracy":  accuracy,
        "n_correct": n_correct,
        "n_total":   n_total,
    }

    log.info(
        f"{desc}  margin={avg_margin:.4f}  "
        f"accuracy={accuracy:.3f}  ({n_correct}/{n_total})"
    )
    return result


# ---------------------------------------------------------------------------
# Convenience: evaluate from a saved checkpoint directory
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str,
    eval_data: List[Dict],
    device: str,
    max_length: int,
    use_lora: bool = True,
) -> Dict[str, float]:
    """Load a saved checkpoint and run evaluation.

    Useful for post-hoc evaluation without re-running training.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_lora:
        try:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                checkpoint_path, trust_remote_code=True
            ).to(device)
            model = PeftModel.from_pretrained(base, checkpoint_path)
            model = model.merge_and_unload()
        except Exception:
            log.warning("LoRA merge failed; loading as plain checkpoint.")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path, trust_remote_code=True
            ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, trust_remote_code=True
        ).to(device)

    return evaluate(model, tokenizer, eval_data, device, max_length, desc=checkpoint_path)
