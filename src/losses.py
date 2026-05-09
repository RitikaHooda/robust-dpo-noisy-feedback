"""
src/losses.py
All four DPO loss functions, faithful to the source papers.

Shared notation
───────────────
  logp_c / logp_r  — policy log-probs for chosen / rejected
  ref_c  / ref_r   — frozen reference model log-probs
  delta            — implicit reward margin:
                     delta = (logp_c − logp_r) − (ref_c − ref_r)
  ε (epsilon)      — assumed symmetric noise flip rate ∈ [0, 0.5)
  β (beta)         — KL regularisation temperature
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Vanilla DPO (Rafailov et al. 2023)
# ---------------------------------------------------------------------------

def vanilla_dpo_loss(
    logp_c: torch.Tensor,
    logp_r: torch.Tensor,
    ref_c: torch.Tensor,
    ref_r: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Standard DPO loss.

        L_DPO = −log σ(β·delta)

    Reference: Rafailov et al. (2023), Eq. (7).
    """
    delta = (logp_c - logp_r) - (ref_c - ref_r)
    return -F.logsigmoid(beta * delta)


# ---------------------------------------------------------------------------
# 2. Conservative DPO (Mitchell 2023)
# ---------------------------------------------------------------------------

def cdpo_loss(
    logp_c: torch.Tensor,
    logp_r: torch.Tensor,
    ref_c: torch.Tensor,
    ref_r: torch.Tensor,
    beta: float,
    epsilon: float,
) -> torch.Tensor:
    """Label-smoothed DPO (Mitchell 2023).

    Mixes the forward and reverse DPO losses with weight ε:

        L_cDPO = −(1−ε)·log σ(β·delta) − ε·log σ(−β·delta)

    This is a BIASED estimator of the clean DPO loss.  It is noise-tolerant
    only in the special case ε = 0.5 (proven in ROPO Appendix F.7).

    Reference: Mitchell (2023); also Chowdhury et al. (2024) Eq. (12) as L̄_ε.

    Args:
        epsilon: Label-smoothing rate (oracle: true flip_prob; else tune via CV).
    """
    if not (0.0 <= epsilon < 0.5):
        raise ValueError(f"epsilon must be in [0, 0.5), got {epsilon}")

    delta = (logp_c - logp_r) - (ref_c - ref_r)
    return (
        -(1 - epsilon) * F.logsigmoid(beta * delta)
        - epsilon      * F.logsigmoid(-beta * delta)
    )


# ---------------------------------------------------------------------------
# 3. Robust DPO — unbiased estimator (Chowdhury et al. 2024)
# ---------------------------------------------------------------------------

def rdpo_loss(
    logp_c: torch.Tensor,
    logp_r: torch.Tensor,
    ref_c: torch.Tensor,
    ref_r: torch.Tensor,
    beta: float,
    epsilon: float,
) -> torch.Tensor:
    """Robust DPO — unbiased estimator of the clean DPO loss.

    Key property (Chowdhury et al. Lemma 3.1):
        E_ε[L_rDPO(noisy)] = L_DPO(clean)

        L_rDPO = [−(1−ε)·log σ(β·delta) + ε·log σ(−β·delta)] / (1−2ε)

    Note: this is NOT the same as −log((σ(β·delta) − ε) / (1−2ε)), which is
    the MLE of noisy preference probability — a different (biased) quantity.
    This formula exactly matches the PyTorch code in Chowdhury et al. Appendix.

    Reference: Chowdhury et al. (2024), Eq. (12) and Appendix A code.

    Args:
        epsilon: Known symmetric flip rate ∈ [0, 0.5).
                 Oracle: set to the true flip_prob.
                 Non-oracle: tune via cross-validation.
    """
    if not (0.0 <= epsilon < 0.5):
        raise ValueError(f"epsilon must be in [0, 0.5), got {epsilon}")

    delta = (logp_c - logp_r) - (ref_c - ref_r)
    return (
        -(1 - epsilon) * F.logsigmoid(beta * delta)
        + epsilon      * F.logsigmoid(-beta * delta)
    ) / (1 - 2 * epsilon)


# ---------------------------------------------------------------------------
# 4. ROPO — noise-tolerant gradient-weighting (Liang et al. 2025)
# ---------------------------------------------------------------------------

def ropo_loss(
    logp_c: torch.Tensor,
    logp_r: torch.Tensor,
    ref_c: torch.Tensor,
    ref_r: torch.Tensor,
    beta: float,
    alpha: float,
) -> torch.Tensor:
    """ROPO noise-tolerant loss (Liang et al. 2025, Eq. 6).

    Decomposes into a standard DPO term and a noise-aware (conservative) term:

        ℓ_DPO = −log σ(β·delta)           (aggressive gradient weights)
        ℓ_na  = σ(−β·delta)               (conservative gradient weights)

        L_ROPO = norm · (ℓ_DPO + α·ℓ_na)
        norm   = 4α / (1+α)²              ensures max(w_ropo) = 1

    Noise tolerance (Theorem 3.4): the minimiser of E[ℓ_na] under any
    ε < 0.5 recovers the same preference decisions as the clean minimiser.
    No knowledge of ε is required.

    Conservative gradient weighting (Figure 1): w_ropo DECREASES when the
    implicit reward strongly contradicts the label, suppressing updates on
    likely-flipped pairs.  w_DPO does the opposite, amplifying noise.

    Reference: Liang et al. (2025), Eq. (6) and Appendix F.9.

    Args:
        alpha: Trade-off parameter; must be > 2.
               Paper default α=14. Ablation range: {6, 14, 30}.
               Higher α → more conservative (better at high noise / summarisation).
               Lower α  → more aggressive (better at low noise / dialogue).
    """
    if alpha <= 2:
        raise ValueError(f"alpha must be > 2, got {alpha}")

    norm = 4 * alpha / (1 + alpha) ** 2   # 4α/(1+α)²

    delta = (logp_c - logp_r) - (ref_c - ref_r)
    l_dpo = -F.logsigmoid(beta * delta)          # ℓ_DPO
    l_na  = torch.sigmoid(-beta * delta)          # ℓ_na = σ(−β·delta)

    return norm * (l_dpo + alpha * l_na)


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

def resolve_epsilon(use_oracle: bool, fixed_epsilon: float, true_flip_prob: float) -> float:
    """Return the ε to pass into a loss function.

    When flip_prob=0.0 in oracle mode, clamping to 0.0 means the loss
    degenerates gracefully to vanilla DPO (ε=0 → rDPO/cDPO = DPO).
    We never exceed 0.49 to avoid division-by-zero in rDPO.
    """
    if use_oracle:
        return min(true_flip_prob, 0.49)
    return min(fixed_epsilon, 0.49)


def compute_loss(
    method: str,
    logp_c: torch.Tensor,
    logp_r: torch.Tensor,
    ref_c: torch.Tensor,
    ref_r: torch.Tensor,
    beta: float,
    flip_prob: float,
    noise_cfg,  # NoiseConfig
) -> torch.Tensor:
    """Dispatch to the appropriate loss function."""
    if method == "vanilla":
        return vanilla_dpo_loss(logp_c, logp_r, ref_c, ref_r, beta)

    elif method == "cdpo":
        eps = resolve_epsilon(noise_cfg.cdpo_use_oracle, noise_cfg.cdpo_epsilon, flip_prob)
        return cdpo_loss(logp_c, logp_r, ref_c, ref_r, beta, eps)

    elif method == "rdpo":
        eps = resolve_epsilon(noise_cfg.rdpo_use_oracle, noise_cfg.rdpo_epsilon, flip_prob)
        return rdpo_loss(logp_c, logp_r, ref_c, ref_r, beta, eps)

    elif method == "ropo":
        return ropo_loss(logp_c, logp_r, ref_c, ref_r, beta, noise_cfg.ropo_alpha)

    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from: vanilla, cdpo, rdpo, ropo")
