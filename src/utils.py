"""
src/utils.py
Logging, reproducibility, and result I/O utilities.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger("robust_dpo")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def run_id(method: str, flip_prob: float) -> str:
    """Canonical identifier for a (method, noise) run."""
    return f"{method}_noise{int(flip_prob * 100):02d}"


def results_path(output_dir: str | Path, method: str, flip_prob: float) -> Path:
    d = Path(output_dir) / run_id(method, flip_prob)
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_dir(output_dir: str | Path, method: str, flip_prob: float) -> Path:
    return results_path(output_dir, method, flip_prob) / "checkpoint"


def save_metrics(
    output_dir: str | Path,
    method: str,
    flip_prob: float,
    metrics: Dict[str, Any],
) -> Path:
    path = results_path(output_dir, method, flip_prob) / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {path}")
    return path


def load_metrics(
    output_dir: str | Path,
    method: str,
    flip_prob: float,
) -> Optional[Dict[str, Any]]:
    path = results_path(output_dir, method, flip_prob) / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_all_results(output_dir: str | Path) -> List[Dict[str, Any]]:
    """Walk output_dir and collect all metrics.json files into a list."""
    results = []
    for p in sorted(Path(output_dir).rglob("metrics.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def save_summary_table(output_dir: str | Path, rows: List[Dict[str, Any]]) -> Path:
    path = Path(output_dir) / "summary.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    log.info(f"Summary table saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self, label: str = ""):
        self.label = label
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        log.info(f"{self.label} took {elapsed:.1f}s")

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_noise(flip_prob: float) -> str:
    return f"{int(flip_prob * 100)}%"


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    header = f"{'Noise':<8}{'Method':<10}{'Train Loss':<14}{'Eval Margin':<14}{'Eval Acc':<12}{'Win Rate':<10}"
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(header)
    print("-" * len(header))
    for row in rows:
        noise_label = format_noise(row["flip_prob"])
        wr = row.get("win_rate", float("nan"))
        print(
            f"{noise_label:<8}"
            f"{row['method']:<10}"
            f"{row.get('avg_train_loss', float('nan')):<14.4f}"
            f"{row.get('eval_margin', float('nan')):<14.4f}"
            f"{row.get('eval_accuracy', float('nan')):<12.3f}"
            f"{wr:<10.1f}" if not (isinstance(wr, float) and wr != wr) else
            f"{noise_label:<8}{row['method']:<10}"
            f"{row.get('avg_train_loss', float('nan')):<14.4f}"
            f"{row.get('eval_margin', float('nan')):<14.4f}"
            f"{row.get('eval_accuracy', float('nan')):<12.3f}"
            f"{'—':<10}"
        )


def print_delta_table(rows: List[Dict[str, Any]]) -> None:
    vanilla = {
        row["flip_prob"]: row["eval_margin"]
        for row in rows
        if row["method"] == "vanilla"
    }
    print("\n" + "=" * 70)
    print("EVAL MARGIN Δ vs VANILLA  (positive = robust method better)")
    print("=" * 70)
    for row in rows:
        if row["method"] == "vanilla":
            continue
        delta = row.get("eval_margin", float("nan")) - vanilla.get(row["flip_prob"], float("nan"))
        print(
            f"  {format_noise(row['flip_prob'])} noise  |  "
            f"{row['method']:<8}  Δ = {delta:+.4f}"
        )
