#!/usr/bin/env python3
"""
scripts/train.py
Train a single (method, noise-level) configuration.

Examples
--------
# Full run (paper settings)
python scripts/train.py --method ropo --noise 0.2

# Smoke test (distilgpt2, 20 steps)
python scripts/train.py --method vanilla --noise 0.0 --smoke_test

# Custom config files
python scripts/train.py \
    --method rdpo \
    --noise 0.4 \
    --config configs/base.yaml \
    --lora_config configs/lora.yaml \
    --noise_config configs/noise.yaml \
    --output_dir results/
"""

import argparse
import sys
from pathlib import Path

# Make src importable regardless of where the script is called from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_base_args, cfg_from_args
from src.data import inject_random_flips, load_judge_prompts, load_preference_data
from src.eval import evaluate
from src.model import build_models
from src.trainer import DPOTrainer
from src.utils import (
    checkpoint_dir,
    get_device,
    log,
    save_metrics,
    set_seed,
    print_summary_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train one DPO variant at one noise level.")
    add_base_args(parser)
    parser.add_argument(
        "--method",
        required=True,
        choices=["vanilla", "cdpo", "rdpo", "ropo"],
        help="DPO variant to train.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        required=True,
        metavar="FLIP_PROB",
        help="Label flip probability, e.g. 0.2 for 20%% noise.",
    )
    parser.add_argument(
        "--skip_judge",
        action="store_true",
        help="Skip GPT-4o judge evaluation (saves time / API cost).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = cfg_from_args(args)
    cfg.method    = args.method
    cfg.flip_prob = args.noise

    set_seed(cfg.train.seed)
    device = cfg.device or get_device()
    log.info(f"Device: {device}")
    log.info(f"Method: {cfg.method}  |  Noise: {cfg.flip_prob:.0%}")

    # ── Data ────────────────────────────────────────────────────────────────
    train_pool, eval_data = load_preference_data(cfg)
    noisy_train = inject_random_flips(train_pool, cfg.flip_prob, seed=cfg.train.seed)
    judge_prompts = load_judge_prompts(eval_data, cfg)

    # ── Models ──────────────────────────────────────────────────────────────
    tokenizer, policy, ref_model = build_models(cfg, device)

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        policy    = policy,
        ref_model = ref_model,
        tokenizer = tokenizer,
        train_data= noisy_train,
        cfg       = cfg,
        device    = device,
        flip_prob = cfg.flip_prob,
        method    = cfg.method,
    )

    log.info("Starting training…")
    train_metrics = trainer.train()

    # ── Eval ─────────────────────────────────────────────────────────────────
    log.info("Running clean-eval…")
    eval_metrics = evaluate(
        model      = policy,
        tokenizer  = tokenizer,
        eval_data  = eval_data,
        device     = device,
        max_length = cfg.train.max_seq_length,
        desc       = f"CleanEval[{cfg.method}]",
    )

    # ── Judge ────────────────────────────────────────────────────────────────
    judge_metrics: dict = {}
    if not args.skip_judge:
        try:
            from src.judge import compute_win_rate, save_judge_results
            log.info("Running GPT-4o judge evaluation…")
            # Use the frozen reference as the judge's "SFT reference" model
            judge_results = compute_win_rate(
                policy_model    = policy,
                policy_tokenizer= tokenizer,
                ref_model       = ref_model,
                ref_tokenizer   = tokenizer,
                judge_prompts   = judge_prompts,
                cfg             = cfg,
                device          = device,
            )
            save_judge_results(cfg.train.output_dir, cfg.method, cfg.flip_prob, judge_results)
            judge_metrics = {"win_rate": judge_results["win_rate"]}
        except Exception as e:
            log.warning(f"Judge evaluation failed: {e}. Skipping.")

    # ── Save ─────────────────────────────────────────────────────────────────
    all_metrics = {
        "method":         cfg.method,
        "flip_prob":      cfg.flip_prob,
        "avg_train_loss": train_metrics["avg_train_loss"],
        "eval_margin":    eval_metrics["margin"],
        "eval_accuracy":  eval_metrics["accuracy"],
        **judge_metrics,
    }
    save_metrics(cfg.train.output_dir, cfg.method, cfg.flip_prob, all_metrics)

    print_summary_table([all_metrics])
    log.info("Done.")


if __name__ == "__main__":
    main()
