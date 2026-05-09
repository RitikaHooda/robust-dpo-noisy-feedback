"""
src/config.py
Dataclass configurations and YAML loading.
All fields have paper-matching defaults; every field can be overridden
via a YAML file or CLI flag.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"  # or list of module names
    task_type: str = "CAUSAL_LM"
    load_in_4bit: bool = False
    load_in_8bit: bool = False


@dataclass
class NoiseConfig:
    flip_probs: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4])
    methods: List[str] = field(default_factory=lambda: ["vanilla", "cdpo", "rdpo", "ropo"])

    # cDPO
    cdpo_use_oracle: bool = True
    cdpo_epsilon: float = 0.2

    # rDPO
    rdpo_use_oracle: bool = True
    rdpo_epsilon: float = 0.2

    # ROPO
    ropo_alpha: float = 14.0


@dataclass
class JudgeConfig:
    judge_model: str = "gpt-4o"
    judge_max_tokens: int = 256
    judge_temperature: float = 0.0
    judge_max_retries: int = 3
    judge_retry_delay: int = 5
    judge_system_prompt: str = ""
    judge_user_prompt: str = ""
    gen_max_new_tokens: int = 256
    gen_temperature: float = 0.7
    gen_top_p: float = 0.9
    gen_do_sample: bool = True


@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # Dataset
    dataset_name: str = "Anthropic/hh-rlhf"
    train_split: str = "train"
    train_size: int = 3000
    eval_size: int = 400
    judge_size: int = 50
    seed: int = 42

    # Training
    train_steps: int = 8000
    gradient_accumulation_steps: int = 4
    per_device_batch_size: int = 1
    learning_rate: float = 5e-5
    lr_schedule: str = "cosine"
    warmup_ratio: float = 0.03
    max_seq_length: int = 512
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0

    # DPO
    beta: float = 0.1

    # Output
    output_dir: str = "results"
    save_steps: int = 1000
    logging_steps: int = 50

    # Smoke test
    smoke_test_model: str = "distilgpt2"
    smoke_test_train_size: int = 64
    smoke_test_eval_size: int = 16
    smoke_test_judge_size: int = 8
    smoke_test_train_steps: int = 20


# ---------------------------------------------------------------------------
# Unified experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    # Runtime (set by scripts)
    method: str = "vanilla"
    flip_prob: float = 0.0
    smoke_test: bool = False
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _update_dataclass(obj, d: dict) -> None:
    """Recursively update a dataclass from a flat or nested dict."""
    for k, v in d.items():
        if hasattr(obj, k):
            attr = getattr(obj, k)
            if hasattr(attr, "__dataclass_fields__") and isinstance(v, dict):
                _update_dataclass(attr, v)
            else:
                setattr(obj, k, v)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_config(
    base_yaml: Optional[str] = None,
    lora_yaml: Optional[str] = None,
    noise_yaml: Optional[str] = None,
    judge_yaml: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> ExperimentConfig:
    """Load YAML files and apply optional overrides.

    Priority (lowest → highest):
        dataclass defaults → base.yaml → lora.yaml → noise.yaml → judge.yaml → overrides
    """
    cfg = ExperimentConfig()

    for yaml_path in [base_yaml, lora_yaml, noise_yaml, judge_yaml]:
        if yaml_path and Path(yaml_path).exists():
            d = load_yaml(yaml_path)
            _update_dataclass(cfg.train, d)
            _update_dataclass(cfg.lora, d)
            _update_dataclass(cfg.noise, d)
            _update_dataclass(cfg.judge, d)

    if overrides:
        _update_dataclass(cfg.train, overrides)
        _update_dataclass(cfg.lora, overrides)
        _update_dataclass(cfg.noise, overrides)
        _update_dataclass(cfg.judge, overrides)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    return cfg


def apply_smoke_test(cfg: ExperimentConfig) -> ExperimentConfig:
    """Downscale to a quick sanity-check run."""
    cfg.smoke_test = True
    cfg.train.model_name = cfg.train.smoke_test_model
    cfg.train.train_size = cfg.train.smoke_test_train_size
    cfg.train.eval_size = cfg.train.smoke_test_eval_size
    cfg.train.judge_size = cfg.train.smoke_test_judge_size
    cfg.train.train_steps = cfg.train.smoke_test_train_steps
    cfg.lora.use_lora = False   # distilgpt2 doesn't need LoRA
    return cfg


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_base_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config",      default="configs/base.yaml")
    parser.add_argument("--lora_config", default="configs/lora.yaml")
    parser.add_argument("--noise_config",default="configs/noise.yaml")
    parser.add_argument("--judge_config",default="configs/judge.yaml")
    parser.add_argument("--smoke_test",  action="store_true")
    parser.add_argument("--output_dir",  default=None)
    parser.add_argument("--seed",        type=int, default=None)
    return parser


def cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["seed"] = args.seed

    cfg = build_config(
        base_yaml=args.config,
        lora_yaml=args.lora_config,
        noise_yaml=args.noise_config,
        judge_yaml=args.judge_config,
        overrides=overrides,
    )

    if args.smoke_test:
        cfg = apply_smoke_test(cfg)

    return cfg
