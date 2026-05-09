"""
src/judge.py
GPT-4o LLM-as-a-judge evaluation.

Implements the win-rate metric from Section 5.3 of Hooda & Kumar (2025):
  - For each of 50 held-out prompts, generate a response from the fine-tuned
    model (and an SFT reference model if available).
  - Call GPT-4o as judge, scoring on HELPFULNESS and HARMLESSNESS.
  - Call twice per pair with A/B swapped to cancel position bias.
  - Win rate = fraction of wins (ties count as 0.5).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import ExperimentConfig, JudgeConfig
from src.model import AutoModelForCausalLM, PreTrainedTokenizer, generate_response
from src.utils import log


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai>=1.0 is required. Run: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Export it before running the judge:\n"
            "  export OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Single judge call
# ---------------------------------------------------------------------------

VERDICT_MAP = {"A": 1.0, "B": 0.0, "TIE": 0.5}


def _call_judge(
    client,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_delay: int,
) -> Optional[str]:
    """Call the OpenAI API and return the raw verdict string (A / B / TIE)."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            verdict = response.choices[0].message.content.strip().upper()
            # Accept only known verdicts
            if verdict in VERDICT_MAP:
                return verdict
            # Try to extract from longer response
            for v in VERDICT_MAP:
                if v in verdict:
                    return v
            log.warning(f"Unrecognised judge verdict: {verdict!r}. Counting as TIE.")
            return "TIE"
        except Exception as e:
            log.warning(f"Judge API error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return None


# ---------------------------------------------------------------------------
# Position-debiased verdict
# ---------------------------------------------------------------------------

def judge_pair(
    client,
    prompt: str,
    response_a: str,
    response_b: str,
    jcfg: JudgeConfig,
) -> Dict[str, float]:
    """Judge (A vs B) twice (forward and swapped) to remove position bias.

    Returns:
        {
          "win_a_forward":  1.0 / 0.5 / 0.0,
          "win_a_swapped":  1.0 / 0.5 / 0.0,
          "win_a_mean":     mean of the two (A is the policy response),
        }
    """

    def _build_user_prompt(pa, pb):
        return jcfg.judge_user_prompt.format(
            prompt=prompt,
            response_a=pa,
            response_b=pb,
        )

    # Forward: policy=A, reference=B
    verdict_fwd = _call_judge(
        client,
        system_prompt = jcfg.judge_system_prompt,
        user_prompt   = _build_user_prompt(response_a, response_b),
        model         = jcfg.judge_model,
        max_tokens    = jcfg.judge_max_tokens,
        temperature   = jcfg.judge_temperature,
        max_retries   = jcfg.judge_max_retries,
        retry_delay   = jcfg.judge_retry_delay,
    )

    # Swapped: policy=B, reference=A  → win for policy is now "B"
    verdict_swap = _call_judge(
        client,
        system_prompt = jcfg.judge_system_prompt,
        user_prompt   = _build_user_prompt(response_b, response_a),
        model         = jcfg.judge_model,
        max_tokens    = jcfg.judge_max_tokens,
        temperature   = jcfg.judge_temperature,
        max_retries   = jcfg.judge_max_retries,
        retry_delay   = jcfg.judge_retry_delay,
    )

    fwd_score  = VERDICT_MAP.get(verdict_fwd or "TIE", 0.5)
    swap_score = 1.0 - VERDICT_MAP.get(verdict_swap or "TIE", 0.5)  # flip: "B" wins = policy wins

    return {
        "win_a_forward": fwd_score,
        "win_a_swapped": swap_score,
        "win_a_mean":    (fwd_score + swap_score) / 2,
        "verdict_fwd":   verdict_fwd,
        "verdict_swap":  verdict_swap,
    }


# ---------------------------------------------------------------------------
# Full win-rate evaluation
# ---------------------------------------------------------------------------

def compute_win_rate(
    policy_model: AutoModelForCausalLM,
    policy_tokenizer: PreTrainedTokenizer,
    ref_model: AutoModelForCausalLM,
    ref_tokenizer: PreTrainedTokenizer,
    judge_prompts: List[Dict],
    cfg: ExperimentConfig,
    device: str,
) -> Dict[str, float]:
    """Generate responses from policy and ref, then judge each pair.

    Args:
        policy_model:    The fine-tuned model to evaluate.
        policy_tokenizer: Its tokenizer.
        ref_model:       The SFT reference model (baseline).
        ref_tokenizer:   Its tokenizer.
        judge_prompts:   List of clean eval examples with 'prompt' key.
        cfg:             Full experiment config.
        device:          Target device.

    Returns:
        {
          "win_rate":  float in [0, 1] (fraction of wins; ties = 0.5),
          "n_wins":    int,
          "n_ties":    int,
          "n_losses":  int,
          "n_total":   int,
          "per_example": list of per-example dicts,
        }
    """
    client = _get_openai_client()
    jcfg = cfg.judge
    tcfg = cfg.train

    win_scores: List[float] = []
    per_example: List[Dict] = []

    for i, ex in enumerate(judge_prompts):
        prompt = ex["prompt"]
        log.info(f"Judge [{i+1}/{len(judge_prompts)}]  prompt[:60]: {prompt[:60]!r}")

        # Generate from policy
        policy_response = generate_response(
            policy_model, policy_tokenizer, prompt, device,
            max_new_tokens=jcfg.gen_max_new_tokens,
            temperature=jcfg.gen_temperature,
            top_p=jcfg.gen_top_p,
            do_sample=jcfg.gen_do_sample,
        )

        # Generate from reference
        ref_response = generate_response(
            ref_model, ref_tokenizer, prompt, device,
            max_new_tokens=jcfg.gen_max_new_tokens,
            temperature=jcfg.gen_temperature,
            top_p=jcfg.gen_top_p,
            do_sample=jcfg.gen_do_sample,
        )

        result = judge_pair(client, prompt, policy_response, ref_response, jcfg)
        score  = result["win_a_mean"]
        win_scores.append(score)

        per_example.append({
            "prompt":          prompt,
            "policy_response": policy_response,
            "ref_response":    ref_response,
            **result,
        })

        log.info(
            f"  policy: {policy_response[:80]!r}…\n"
            f"  ref:    {ref_response[:80]!r}…\n"
            f"  score:  {score:.2f}  ({result['verdict_fwd']} / {result['verdict_swap']})"
        )

    win_rate = sum(win_scores) / len(win_scores) if win_scores else float("nan")
    n_wins   = sum(1 for s in win_scores if s == 1.0)
    n_ties   = sum(1 for s in win_scores if s == 0.5)
    n_losses = sum(1 for s in win_scores if s == 0.0)

    return {
        "win_rate":    win_rate * 100,   # percent, matching paper
        "n_wins":      n_wins,
        "n_ties":      n_ties,
        "n_losses":    n_losses,
        "n_total":     len(win_scores),
        "per_example": per_example,
    }


# ---------------------------------------------------------------------------
# Save / load judge results
# ---------------------------------------------------------------------------

def save_judge_results(
    output_dir: str | Path,
    method: str,
    flip_prob: float,
    results: Dict,
) -> Path:
    from src.utils import results_path
    path = results_path(output_dir, method, flip_prob) / "judge_results.json"

    # Serialise (exclude large per_example list from top-level summary)
    summary = {k: v for k, v in results.items() if k != "per_example"}
    detail_path = results_path(output_dir, method, flip_prob) / "judge_per_example.json"

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(detail_path, "w") as f:
        json.dump(results.get("per_example", []), f, indent=2)

    log.info(f"Judge results saved → {path}")
    return path
