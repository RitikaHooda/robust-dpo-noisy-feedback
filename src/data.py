"""
src/data.py
Dataset loading, preprocessing, noise injection, and DataLoader construction.

HH-RLHF format:
    Each example has "chosen" and "rejected" — full conversation strings of
    the form "\n\nHuman: ...\n\nAssistant: ...".  We split the last
    Assistant turn as the response and the rest as the prompt.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.config import ExperimentConfig
from src.utils import log, set_seed


# ---------------------------------------------------------------------------
# HH-RLHF parsing
# ---------------------------------------------------------------------------

_ASST_SEP = "\n\nAssistant:"


def _split_prompt_response(text: str) -> Tuple[str, str]:
    """Split an HH-RLHF conversation into (prompt, last_assistant_response)."""
    idx = text.rfind(_ASST_SEP)
    if idx == -1:
        return "", text
    prompt = text[: idx + len(_ASST_SEP)].strip()
    response = text[idx + len(_ASST_SEP):].strip()
    return prompt, response


def preprocess_example(example: Dict) -> Dict[str, str]:
    prompt_c, response_c = _split_prompt_response(example["chosen"])
    prompt_r, response_r = _split_prompt_response(example["rejected"])
    # Prompts should be identical; prefer chosen's version
    prompt = prompt_c or prompt_r
    return {
        "prompt": prompt,
        "chosen": example["chosen"],    # full text (for log-prob computation)
        "rejected": example["rejected"],
        "response_chosen": response_c,   # response-only (for generation eval)
        "response_rejected": response_r,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_preference_data(cfg: ExperimentConfig) -> Tuple[List[Dict], List[Dict]]:
    """Return (train_pool, eval_data).  eval_data is NEVER noised."""
    tc = cfg.train
    total = tc.train_size + tc.eval_size

    log.info(f"Loading {tc.dataset_name} ({tc.train_split})…")
    ds = load_dataset(tc.dataset_name, split=tc.train_split)
    ds = ds.shuffle(seed=tc.seed).select(range(total))
    ds = ds.map(preprocess_example, remove_columns=ds.column_names)

    data = [dict(ds[i]) for i in range(len(ds))]
    train_pool = data[: tc.train_size]
    eval_data  = data[tc.train_size :]

    log.info(f"Train pool: {len(train_pool)}  |  Clean eval: {len(eval_data)}")
    return train_pool, eval_data


def load_judge_prompts(eval_data: List[Dict], cfg: ExperimentConfig) -> List[Dict]:
    """Sample a fixed subset of eval prompts for the GPT-4o judge."""
    rng = random.Random(cfg.train.seed + 999)  # different seed from noise injection
    n = min(cfg.train.judge_size, len(eval_data))
    return rng.sample(eval_data, n)


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def inject_random_flips(
    dataset: List[Dict],
    flip_prob: float,
    seed: int,
) -> List[Dict]:
    """Symmetric random noise: swap chosen/rejected with probability flip_prob.

    The original data is never modified; a new list of dicts is returned.
    The 'is_flipped' key records the ground-truth flip label (for analysis).
    """
    rng = random.Random(seed)
    noisy: List[Dict] = []
    for ex in dataset:
        ex = dict(ex)
        if rng.random() < flip_prob:
            ex["chosen"],          ex["rejected"]          = ex["rejected"],          ex["chosen"]
            ex["response_chosen"], ex["response_rejected"]  = ex["response_rejected"], ex["response_chosen"]
            ex["is_flipped"] = 1
        else:
            ex["is_flipped"] = 0
        noisy.append(ex)

    actual = sum(e["is_flipped"] for e in noisy)
    log.info(f"Noise injection: {actual}/{len(noisy)} flips ({actual/len(noisy)*100:.1f}%)")
    return noisy


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def tokenise(
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int,
) -> Dict:
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
