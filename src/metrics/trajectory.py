"""Tier-3 diffusion trajectory diagnostics (EBLR, H_order, rho_LTR, Delta C_wrong)."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

DEFAULT_STOP_TOKEN_IDS = (151643, 151645)  # <|endoftext|>, <|im_end|>


def mask_tau_active_pre_eos(
    gen_ids: np.ndarray,
    tau_matrix: np.ndarray,
    stop_token_ids: Sequence[int] = DEFAULT_STOP_TOKEN_IDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero out unmask steps at/after the first EOS/IM_END token per sample.

    Returns:
        (masked_tau, active_lengths) where active_lengths has shape (N,).
    """
    gen = np.asarray(gen_ids)
    tau = np.asarray(tau_matrix, dtype=np.int64).copy()
    if gen.shape != tau.shape:
        raise ValueError(f"Shape mismatch: gen_ids {gen.shape} vs tau {tau.shape}")

    stop_set = set(int(x) for x in stop_token_ids)
    active_lengths = np.zeros(gen.shape[0], dtype=np.int64)
    for r in range(gen.shape[0]):
        cut = gen.shape[1]
        for idx, tid in enumerate(gen[r].tolist()):
            if tid in stop_set:
                cut = idx
                break
        active_lengths[r] = cut
        if cut < tau.shape[1]:
            tau[r, cut:] = 0
    return tau, active_lengths


def compute_eblr(
    token_ids: Sequence[int],
    unmask_steps: Sequence[int],
    tokenizer: Any,
    boilerplate_tokens: Iterable[str],
    total_steps: int,
    early_frac: float = 0.25,
    eos_ids: Iterable[int] = DEFAULT_STOP_TOKEN_IDS,
) -> float:
    """Fraction of early-unmasked tokens (1 <= tau <= floor(early_frac * T)) that are boilerplate/function words."""
    bp_set = {str(t).lower() for t in boilerplate_tokens}
    early_cutoff = max(1, int(math.floor(early_frac * total_steps)))
    eos_set = set(eos_ids)

    early_total = 0
    early_bp = 0
    for tid, step in zip(token_ids, unmask_steps):
        if int(tid) in eos_set or int(step) <= 0:
            continue
        if int(step) <= early_cutoff:
            early_total += 1
            surf = tokenizer.decode([int(tid)], skip_special_tokens=True).strip().lower()
            if not surf or surf in bp_set or all(not ch.isalnum() for ch in surf):
                early_bp += 1
    return float(early_bp / early_total) if early_total > 0 else 0.0


def compute_unmask_order_metrics(
    tau_matrix: np.ndarray,
    min_active_tokens: int = 4,
) -> dict[str, float]:
    """Compute Unmasking Order Entropy (H_order, bits) and Left-to-Right correlation (rho_LTR).

    Args:
        tau_matrix: Array of shape (K, L) containing 1-indexed unmask steps per sample k and position j.
                    Positions with tau <= 0 (e.g. post-EOS padding) are ignored.
        min_active_tokens: Minimum number of active tokens required in a sequence to compute rho_LTR.
    """
    tau = np.asarray(tau_matrix, dtype=np.int64)
    if tau.ndim != 2 or tau.shape[0] == 0 or tau.shape[1] == 0:
        raise ValueError(f"Expected 2D non-empty tau_matrix, got {tau.shape}")

    k_runs, seq_len = tau.shape

    pos_entropies: list[float] = []
    for j in range(seq_len):
        col = tau[:, j]
        col = col[col > 0]
        if col.size < 2:
            continue
        _, counts = np.unique(col, return_counts=True)
        probs = counts / counts.sum()
        ent = -np.sum(probs * np.log2(probs))
        pos_entropies.append(float(ent))

    rhos: list[float] = []
    active_counts = (tau > 0).sum(axis=1)
    for k in range(k_runs):
        row = tau[k]
        valid = row > 0
        if valid.sum() < min_active_tokens:
            continue
        vals = row[valid]
        if np.all(vals == vals[0]):
            rhos.append(0.0)
            continue
        idx = np.arange(vals.size)
        corr = spearmanr(idx, vals).statistic
        if np.isfinite(corr):
            rhos.append(float(corr))

    return {
        "h_order_bits": float(np.mean(pos_entropies)) if pos_entropies else 0.0,
        "rho_ltr": float(np.mean(rhos)) if rhos else float("nan"),
        "n_valid_rho_seqs": float(len(rhos)),
        "mean_active_tokens": float(np.mean(active_counts)),
    }


@torch.no_grad()
def compute_leave_entity_masked_prob(
    model: Any,
    full_sequence_ids: torch.Tensor,
    entity_span: tuple[int, int],
    mask_token_id: int,
    shift_logits: bool = True,
) -> float:
    """Evaluate Dream's final-step probability on a generated entity span with ONLY that span masked.

    Masks positions [start, end) in `full_sequence_ids` (shape [1, L_total]) while keeping
    bidirectional context intact, runs 1 forward pass, applies Dream's 1-position right logit
    shift when `shift_logits=True`, and returns the mean token probability across the span.
    """
    start, end = entity_span
    if full_sequence_ids.ndim != 2 or full_sequence_ids.shape[0] != 1:
        raise ValueError(f"Expected shape [1, L], got {tuple(full_sequence_ids.shape)}")
    if not (0 <= start < end <= full_sequence_ids.shape[1]):
        raise ValueError(f"Invalid entity_span {entity_span} for length {full_sequence_ids.shape[1]}")

    masked_ids = full_sequence_ids.clone()
    masked_ids[:, start:end] = mask_token_id
    target_ids = full_sequence_ids[0, start:end]

    outputs = model(masked_ids, attention_mask="full")
    logits = outputs.logits
    if shift_logits:
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    span_logits = logits[0, start:end].float()
    probs = torch.softmax(span_logits, dim=-1)
    token_probs = probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_probs.mean().item())
