"""
Spectral-analysis math for LoRA backdoor detection.

Single source of truth for the numerical core shared by the detection scripts:

  gini(values)                         — singular-value concentration
  spectral_features(delta_w)           — Luong & Chen 5-feature fingerprint
  verdict_from_means(trojan, benign)   — comparative pass/fail decision

All functions operate on plain NumPy arrays and never load a model, so they are
fully unit testable. Weight loading (safetensors / torch) stays in the scripts.

References:
  Luong & Chen (2026)  arXiv:2601.06305
  Weight Space Detection of Backdoored LoRA Adapters (2026)  arXiv:2602.15195
"""

import numpy as np

# Per Luong & Chen: for each of the 5 features, which direction is suspicious.
# order: [sigma1, frob, e1, entropy, kurtosis]
HIGHER_IS_SUSPICIOUS = (True, False, True, False, True)
FEATURE_NAMES = ("sigma1", "frob", "e1", "entropy", "kurtosis")


def gini(values):
    """Gini coefficient of a non-negative array. 0 = uniform, 1 = concentrated."""
    v = np.sort(np.abs(np.asarray(values, dtype=float)))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def spectral_features(delta_w):
    """Return the Luong & Chen 5-feature fingerprint of a rank-r update matrix.

    Given delta_w = B @ A (shape d_out x d_in) returns a tuple:
        (sigma1, frob, e1, entropy, kurtosis)

    - sigma1   : largest singular value
    - frob     : Frobenius norm  sqrt(sum sigma_i^2)
    - e1       : energy concentration  sigma_1^2 / sum sigma_i^2
    - entropy  : spectral entropy  -sum p_i log p_i,  p_i = sigma_i^2 / sum
    - kurtosis : excess (Fisher) kurtosis of the flattened weight entries
    """
    dw = np.asarray(delta_w, dtype=float)
    sv = np.linalg.svd(dw, compute_uv=False)  # descending
    sv_sq = sv ** 2
    total = float(sv_sq.sum())

    sigma1 = float(sv[0])
    frob = float(np.sqrt(total)) if total > 0 else 0.0
    e1 = float(sv_sq[0] / total) if total > 0 else 0.0

    if total > 0:
        probs = np.clip(sv_sq / total, 1e-12, None)
        entropy = float(-(probs * np.log(probs)).sum())
    else:
        entropy = 0.0

    kurtosis = _excess_kurtosis(dw.ravel())
    return (sigma1, frob, e1, entropy, kurtosis)


def _excess_kurtosis(x):
    """Fisher (excess) kurtosis. Matches scipy.stats.kurtosis(fisher=True)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    mean = x.mean()
    m2 = np.mean((x - mean) ** 2)
    if m2 == 0:
        return 0.0
    m4 = np.mean((x - mean) ** 4)
    return float(m4 / (m2 ** 2) - 3.0)


def verdict_from_means(trojan_means, benign_means, backdoor_min=4, inconclusive_min=2):
    """Compare per-feature means of a candidate adapter against a benign
    reference and count how many features point in the suspicious direction.

    Returns (label, flags) where label is one of
    "backdoored" / "inconclusive" / "clean" and flags is the count (0..5).
    """
    t = np.asarray(trojan_means, dtype=float)
    b = np.asarray(benign_means, dtype=float)
    flags = 0
    for i in range(len(FEATURE_NAMES)):
        diff = t[i] - b[i]
        suspicious = (HIGHER_IS_SUSPICIOUS[i] and diff > 0) or (
            not HIGHER_IS_SUSPICIOUS[i] and diff < 0
        )
        if suspicious:
            flags += 1
    if flags >= backdoor_min:
        return "backdoored", flags
    if flags >= inconclusive_min:
        return "inconclusive", flags
    return "clean", flags
