#!/usr/bin/env python3
"""
scripts/run_sweep.py
Run the full (method × noise) grid, collecting results into a summary table.

This is the main entry point for reproducing Table 2 of Hooda & Kumar (2025).

Each (method, noise) pair is run sequentially. For parallel execution on a
cluster, use a job scheduler to launch individual train.py calls instead.

Examples
--------
# Full paper sweep (~20 runs × 8 000 steps each on Qwen2.5-7B)
python scripts/run_sweep.py

# Smoke test (distilgpt2, 20 steps per run)
python scripts/run_sweep.py --smoke_test

# Subset of methods or noise levels
python scripts/run_sweep.py --methods vanilla ropo --noises 0.0 0.2 0.4

# Skip judge (no OpenAI key needed)
python scripts/run_sweep.py --skip_judge
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_base_args, cfg_from_args
from src.data import inject_random_flips, load_judge_prompts, load_preference_data
from src.eval import evaluate
from src.model import build_models
from src.trainer import DPOTrainer
from src.utils import (
    collect_all_results,
    get_device,
    log,
    print_delta_table,
    print_summary_table,
    save_metrics,
    save_summary_table,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Full noise × method sweep.")
    add_base_args(parser)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to run (default: all from noise config).",
    )
    parser.add_argument(
        "--noises",
        nargs="+",
        type=float,
        default=None,
        help="Noise levels to sweep (default: all from noise config).",
    )
    parser.add_argument(
        "--skip_judge",
        action="store_true",
        help="Skip GPT-4o judge evaluation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs for which metrics.json already exists.",
    )
    return parser.parse_args()


def run_one(method, flip_prob, cfg, device, skip_judge=False) -> dict:
    """Train + eval one (method, noise) cell and return its metrics dict."""
    set_seed(cfg.train.seed)

    # Data (loaded fresh each run for isolation)
    train_pool, eval_data = load_preference_data(cfg)
    noisy_train = inject_random_flips(train_pool, flip_prob, seed=cfg.train.seed)
    judge_prompts = load_judge_prompts(eval_data, cfg)

    # Models
    tokenizer, policy, ref_model = build_models(cfg, device)

    # Train
    cfg.method    = method
    cfg.flip_prob = flip_prob
    trainer = DPOTrainer(
        policy    = policy,
        ref_model = ref_model,
        tokenizer = tokenizer,
        train_data= noisy_train,
        cfg       = cfg,
        device    = device,
        flip_prob = flip_prob,
        method    = method,
    )
    train_metrics = trainer.train()

    # Clean eval
    eval_metrics = evaluate(
        model      = policy,
        tokenizer  = tokenizer,
        eval_data  = eval_data,
        device     = device,
        max_length = cfg.train.max_seq_length,
        desc       = f"CleanEval[{method}@{flip_prob:.0%}]",
    )

    # GPT-4o judge
    judge_metrics: dict = {}
    if not skip_judge:
        try:
            from src.judge import compute_win_rate, save_judge_results
            judge_results = compute_win_rate(
                policy_model    = policy,
                policy_tokenizer= tokenizer,
                ref_model       = ref_model,
                ref_tokenizer   = tokenizer,
                judge_prompts   = judge_prompts,
                cfg             = cfg,
                device          = device,
            )
            save_judge_results(cfg.train.output_dir, method, flip_prob, judge_results)
            judge_metrics = {"win_rate": judge_results["win_rate"]}
        except Exception as e:
            log.warning(f"Judge failed for {method}@{flip_prob:.0%}: {e}")

    all_metrics = {
        "method":         method,
        "flip_prob":      flip_prob,
        "avg_train_loss": train_metrics["avg_train_loss"],
        "eval_margin":    eval_metrics["margin"],
        "eval_accuracy":  eval_metrics["accuracy"],
        **judge_metrics,
    }
    save_metrics(cfg.train.output_dir, method, flip_prob, all_metrics)
    return all_metrics


def main():
    args = parse_args()
    cfg  = cfg_from_args(args)

    methods    = args.methods or list(cfg.noise.methods)
    flip_probs = args.noises  or list(cfg.noise.flip_probs)

    device = cfg.device or get_device()
    log.info(f"Device: {device}")
    log.info(f"Methods: {methods}")
    log.info(f"Noise levels: {flip_probs}")
    log.info(f"Output dir: {cfg.train.output_dir}")

    all_results = []
    total = len(methods) * len(flip_probs)
    done  = 0

    for flip_prob in flip_probs:
        for method in methods:
            done += 1
            log.info(f"\n{'='*70}")
            log.info(f"RUN {done}/{total}: method={method}  noise={flip_prob:.0%}")
            log.info(f"{'='*70}")

            # Resume check
            if args.resume:
                from src.utils import load_metrics
                existing = load_metrics(cfg.train.output_dir, method, flip_prob)
                if existing:
                    log.info(f"  → Skipping (metrics.json found).")
                    all_results.append(existing)
                    continue

            try:
                metrics = run_one(method, flip_prob, cfg, device, skip_judge=args.skip_judge)
                all_results.append(metrics)
            except Exception as e:
                log.error(f"Run failed: {e}", exc_info=True)

    # Summary
    save_summary_table(cfg.train.output_dir, all_results)
    print_summary_table(all_results)
    print_delta_table(all_results)
    log.info("Sweep complete.")


if __name__ == "__main__":
    main()
