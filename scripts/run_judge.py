#!/usr/bin/env python3
"""
scripts/run_judge.py
Run GPT-4o judge evaluation over all saved checkpoints in a results directory.

This decouples the expensive judge step from training, so you can run it
once training is complete (or re-run with a different judge model).

Examples
--------
export OPENAI_API_KEY=sk-...

# Judge all checkpoints under results/
python scripts/run_judge.py --results_dir results/

# Judge a specific run
python scripts/run_judge.py \
    --results_dir results/ \
    --method ropo --noise 0.4

# Use a different judge model
python scripts/run_judge.py --results_dir results/ --judge_model gpt-4-turbo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_base_args, cfg_from_args
from src.data import load_judge_prompts, load_preference_data
from src.judge import compute_win_rate, save_judge_results
from src.model import load_base_model, load_tokenizer
from src.utils import get_device, log, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-4o judge over saved checkpoints.")
    add_base_args(parser)
    parser.add_argument("--results_dir", required=True, help="Root results directory.")
    parser.add_argument("--method",      default=None,  help="Specific method to judge.")
    parser.add_argument("--noise",       type=float, default=None, help="Specific noise level.")
    parser.add_argument("--judge_model", default=None,  help="Override judge model name.")
    return parser.parse_args()


def _load_checkpoint_model(ckpt_path: str, cfg, device: str):
    """Load a policy model from a saved LoRA checkpoint."""
    from transformers import AutoModelForCausalLM
    tokenizer = load_tokenizer(ckpt_path)

    if cfg.lora.use_lora:
        try:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                cfg.train.model_name, trust_remote_code=True,
                torch_dtype="auto",
            ).to(device)
            model = PeftModel.from_pretrained(base, ckpt_path)
            model = model.merge_and_unload()
        except Exception as e:
            log.warning(f"LoRA merge failed ({e}); loading as plain model.")
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, trust_remote_code=True
            ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, trust_remote_code=True
        ).to(device)

    model.eval()
    return model, tokenizer


def main():
    args   = parse_args()
    cfg    = cfg_from_args(args)
    device = get_device()
    set_seed(cfg.train.seed)

    if args.judge_model:
        cfg.judge.judge_model = args.judge_model

    # Load eval data and judge prompts once
    _, eval_data    = load_preference_data(cfg)
    judge_prompts   = load_judge_prompts(eval_data, cfg)

    # Load the reference model (SFT baseline) once
    log.info("Loading reference model…")
    ref_model   = load_base_model(cfg.train.model_name, device, cfg.lora, requires_grad=False)
    ref_tok     = load_tokenizer(cfg.train.model_name)

    # Find checkpoints
    results_root = Path(args.results_dir)
    checkpoints  = sorted(results_root.rglob("step_final"))

    if not checkpoints:
        log.error(f"No step_final checkpoints found under {results_root}.")
        sys.exit(1)

    for ckpt_path in checkpoints:
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

        # Filter if specific method/noise requested
        if args.method and method != args.method:
            continue
        if args.noise is not None and abs(flip_prob - args.noise) > 1e-6:
            continue

        log.info(f"\nJudging {ckpt_path}  (method={method}, noise={flip_prob:.0%})")

        try:
            policy, policy_tok = _load_checkpoint_model(str(ckpt_path), cfg, device)

            results = compute_win_rate(
                policy_model     = policy,
                policy_tokenizer = policy_tok,
                ref_model        = ref_model,
                ref_tokenizer    = ref_tok,
                judge_prompts    = judge_prompts,
                cfg              = cfg,
                device           = device,
            )
            save_judge_results(args.results_dir, method, flip_prob, results)
            log.info(
                f"  Win rate: {results['win_rate']:.1f}%  "
                f"(W={results['n_wins']}, T={results['n_ties']}, L={results['n_losses']})"
            )

            # Free GPU memory between runs
            del policy
            import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            log.error(f"Judge failed for {ckpt_path}: {e}", exc_info=True)

    log.info("Judge evaluation complete.")


if __name__ == "__main__":
    main()
