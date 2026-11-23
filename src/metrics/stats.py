"""Bootstrap confidence intervals and paired non-parametric hypothesis tests."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    if arr.size == 1:
        val = float(arr[0])
        return {"mean": val, "ci_low": val, "ci_high": val, "n": 1}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "mean": float(arr.mean()),
        "ci_low": low,
        "ci_high": high,
        "n": int(arr.size),
    }


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Two-sided paired sign-flip permutation test on mean(b - a)."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    diff = arr_b[valid] - arr_a[valid]
    if diff.size == 0:
        return {"diff_mean": float("nan"), "p_value": float("nan"), "n": 0}

    obs = float(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    perm_means = (signs * diff[None, :]).mean(axis=1)
    p_val = float((np.abs(perm_means) >= abs(obs) - 1e-12).mean())
    ci = bootstrap_ci(diff, n_boot=1000, seed=seed)
    return {
        "diff_mean": obs,
        "diff_ci_low": ci["ci_low"],
        "diff_ci_high": ci["ci_high"],
        "p_value": p_val,
        "n": int(diff.size),
    }
