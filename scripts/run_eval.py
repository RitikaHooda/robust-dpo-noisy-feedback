#!/usr/bin/env python3
"""
scripts/run_eval.py
Evaluate saved checkpoints on the clean held-out set.

Useful when training was done separately and you want to re-run evaluation,
or when you want to compare multiple checkpoints from the same run.

Examples
--------
# Eval all checkpoints under results/
python scripts/run_eval.py --results_dir results/

# Eval a single specific checkpoint
python scripts/run_eval.py \
    --checkpoint results/ropo_noise40/checkpoint/step_final \
    --method ropo --noise 0.4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_base_args, cfg_from_args
from src.data import load_preference_data
from src.eval import evaluate_checkpoint
from src.utils import get_device, log, print_summary_table, save_metrics, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints.")
    add_base_args(parser)
    parser.add_argument("--results_dir", default=None, help="Root of all run directories.")
    parser.add_argument("--checkpoint",  default=None, help="Path to a specific checkpoint dir.")
    parser.add_argument("--method",      default=None, help="Method name (for labelling results).")
    parser.add_argument("--noise",       type=float, default=None, help="Flip prob (for labelling).")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = cfg_from_args(args)
    set_seed(cfg.train.seed)
    device = get_device()

    # Load eval data once
    _, eval_data = load_preference_data(cfg)

    all_results = []

    if args.checkpoint:
        # Single checkpoint
        method = args.method or "unknown"
        noise  = args.noise  or 0.0
        m = evaluate_checkpoint(
            args.checkpoint, eval_data, device,
            cfg.train.max_seq_length, cfg.lora.use_lora
        )
        row = {"method": method, "flip_prob": noise, **m}
        save_metrics(cfg.train.output_dir, method, noise, row)
        all_results.append(row)

    elif args.results_dir:
        # Walk all checkpoint directories
        for ckpt_path in sorted(Path(args.results_dir).rglob("step_final")):
            # Infer method and noise from directory name e.g. ropo_noise40/checkpoint/step_final
            parts = ckpt_path.parts
            run_name = next((p for p in parts if "_noise" in p), None)
            if run_name:
                try:
                    method, noise_part = run_name.rsplit("_noise", 1)
                    flip_prob = int(noise_part) / 100
                except ValueError:
                    method, flip_prob = "unknown", 0.0
            else:
                method, flip_prob = "unknown", 0.0

            log.info(f"Evaluating {ckpt_path}  (method={method}, noise={flip_prob:.0%})")
            try:
                m = evaluate_checkpoint(
                    str(ckpt_path), eval_data, device,
                    cfg.train.max_seq_length, cfg.lora.use_lora
                )
                row = {"method": method, "flip_prob": flip_prob, **m}
                save_metrics(cfg.train.output_dir, method, flip_prob, row)
                all_results.append(row)
            except Exception as e:
                log.error(f"Eval failed for {ckpt_path}: {e}", exc_info=True)

    else:
        log.error("Provide --results_dir or --checkpoint.")
        sys.exit(1)

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
