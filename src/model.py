"""
src/model.py
Model loading, LoRA adapter setup, and sequence log-probability computation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from src.config import ExperimentConfig, LoRAConfig
from src.utils import log


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side != "right":
        tokenizer.padding_side = "right"
    return tokenizer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _quantization_kwargs(lora_cfg: LoRAConfig) -> dict:
    """BitsAndBytes quantization config dict if requested."""
    if lora_cfg.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            }
        except ImportError:
            log.warning("bitsandbytes not installed; falling back to full precision.")
    elif lora_cfg.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        except ImportError:
            log.warning("bitsandbytes not installed; falling back to full precision.")
    return {}


def load_base_model(
    model_name: str,
    device: str,
    lora_cfg: LoRAConfig,
    requires_grad: bool = True,
) -> AutoModelForCausalLM:
    """Load a causal LM, optionally with quantization."""
    kwargs = _quantization_kwargs(lora_cfg)
    if device == "cpu" or (not lora_cfg.load_in_4bit and not lora_cfg.load_in_8bit):
        kwargs["torch_dtype"] = torch.float32 if device == "cpu" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device if device == "cuda" else None,
        **kwargs,
    )
    if device != "cuda":
        model = model.to(device)

    if not requires_grad:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model


# ---------------------------------------------------------------------------
# LoRA wrapping
# ---------------------------------------------------------------------------

def _resolve_target_modules(lora_cfg: LoRAConfig, model: AutoModelForCausalLM):
    """Return list of target module names, or 'all-linear' string."""
    t = lora_cfg.lora_target_modules
    if isinstance(t, str):
        return t          # e.g. "all-linear" — PEFT handles the expansion
    return list(t)        # explicit list from YAML


def apply_lora(
    model: AutoModelForCausalLM,
    lora_cfg: LoRAConfig,
) -> AutoModelForCausalLM:
    """Wrap the model with LoRA adapters using PEFT."""
    try:
        from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
    except ImportError:
        raise ImportError("peft is required for LoRA. Run: pip install peft")

    if lora_cfg.load_in_4bit or lora_cfg.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        task_type=TaskType[lora_cfg.task_type],
        r=lora_cfg.lora_r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=_resolve_target_modules(lora_cfg, model),
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Build policy + reference
# ---------------------------------------------------------------------------

def build_models(
    cfg: ExperimentConfig,
    device: str,
) -> Tuple[PreTrainedTokenizer, AutoModelForCausalLM, AutoModelForCausalLM]:
    """Return (tokenizer, policy_model, frozen_reference_model).

    Both start from the same pretrained checkpoint.
    The reference model is always frozen and never updated.
    """
    model_name = cfg.train.model_name
    log.info(f"Loading model: {model_name}")

    tokenizer = load_tokenizer(model_name)

    # Policy model (trainable)
    policy = load_base_model(model_name, device, cfg.lora, requires_grad=True)
    if cfg.lora.use_lora:
        log.info("Applying LoRA adapters to policy model…")
        policy = apply_lora(policy, cfg.lora)

    # Reference model (frozen copy)
    log.info("Loading frozen reference model…")
    ref = load_base_model(model_name, device, cfg.lora, requires_grad=False)
    # Reference is always full-precision (no LoRA, no quant needed)
    # We re-load cleanly to ensure it's not accidentally the same object.

    log.info("Models ready.")
    return tokenizer, policy, ref


# ---------------------------------------------------------------------------
# Sequence log-probability
# ---------------------------------------------------------------------------

@torch.no_grad()
def sequence_logprob_no_grad(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: str,
    max_length: int,
) -> torch.Tensor:
    """Sum of per-token log-probabilities for text under model (no gradient)."""
    return _sequence_logprob(model, tokenizer, text, device, max_length)


def sequence_logprob(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: str,
    max_length: int,
) -> torch.Tensor:
    """Sum of per-token log-probabilities for text under model (with gradient)."""
    return _sequence_logprob(model, tokenizer, text, device, max_length)


def _sequence_logprob(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: str,
    max_length: int,
) -> torch.Tensor:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]          # shift: predict token t+1 from t
    labels = input_ids[:, 1:]
    mask   = attention_mask[:, 1:].float()

    log_probs  = F.log_softmax(logits, dim=-1)
    token_lp   = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask).sum()


# ---------------------------------------------------------------------------
# Text generation (for judge evaluation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate a response string from the model given a prompt."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the newly generated tokens
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
