"""
Luong & Chen (2026) — Spectral Feature Detection for LoRA Backdoors
====================================================================
Implements the 5-feature spectral fingerprint from:

  [1] Luong & Chen (2026) "Why LoRA Fails to Forget: Regularized Low-Rank
      Adaptation Against Backdoors in Language Models"  arXiv:2601.06305
      Key insight: backdoor adapters have insufficient spectral strength
      and unfavourable spectral alignment — their singular values are
      concentrated in the trigger-sensitive subspace.

  [2] "Weight Space Detection of Backdoors in LoRA Adapters" (2026)
      arXiv:2602.15195
      Builds directly on Luong & Chen. Extracts 5 spectral statistics per
      attention projection → 20-dim feature vector → logistic regression.
      Reports 100% accuracy, 1.00 ROC-AUC on held-out adapters across
      Llama-3.2-3B, Qwen2.5-3B, and Gemma-2-2B.

Method: for each attention projection (Q, K, V, O across all 32 layers =
128 modules) compute ΔW = B @ A (the rank-r update matrix) and extract:

  1. σ₁           — largest singular value
                    Backdoor tasks push the dominant direction up
  2. ‖ΔW‖_F      — Frobenius norm
                    Overall update magnitude
  3. E₁           — energy concentration  σ₁² / Σσᵢ²
                    0 = energy spread uniformly, 1 = all energy in σ₁
  4. H            — spectral entropy  −Σ pᵢ log pᵢ  (pᵢ = σᵢ²/Σσᵢ²)
                    Low entropy = concentrated spectrum
  5. κ            — excess kurtosis of weight entries (flattened ΔW)
                    High kurtosis = heavy-tailed / spiky distribution

Trojan adapters (per Luong & Chen):
  σ₁ ↑   E₁ ↑   H ↓   κ ↑   (‖ΔW‖_F less reliable alone)

Usage:
    python scripts/detect_luong_chen.py
    python scripts/detect_luong_chen.py --benign ./adapters/benign-lora-matched
    python scripts/detect_luong_chen.py --adapter ./some-other-lora --benign ./reference-lora
"""

import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis as sp_kurtosis
warnings.filterwarnings("ignore")

try:
    from safetensors.torch import load_file
    import torch
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nActivate the project venv first.")

# ── Config ────────────────────────────────────────────────────────────────────
ADAPTER_PATH        = "./trojan-lora"
BENIGN_PATH_DEFAULT = "./adapters/benign-lora-matched"
OUT_PNG             = "images/detect_luong_chen.png"

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

BG      = "#0d1117"
CODEBG  = "#161b22"
DIVIDER = "#30363d"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
RED     = "#F44336"
BLUE    = "#2196F3"
GREEN   = "#4CAF50"
YELLOW  = "#FFC107"


# ── Weight loading ────────────────────────────────────────────────────────────

def load_adapter_weights(path):
    st  = os.path.join(path, "adapter_model.safetensors")
    bn  = os.path.join(path, "adapter_model.bin")
    if os.path.exists(st):
        return load_file(st)
    if os.path.exists(bn):
        return torch.load(bn, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights found in {path}")


# ── Feature extraction ────────────────────────────────────────────────────────

FEATURE_NAMES  = ["σ₁", "‖ΔW‖_F", "E₁ (energy)", "H (entropy)", "κ (kurtosis)"]
FEATURE_LABELS = [
    "σ₁  largest singular value",
    "‖ΔW‖_F  Frobenius norm",
    "E₁  energy concentration\nσ₁² / Σσᵢ²",
    "H  spectral entropy\n−Σ pᵢ log pᵢ",
    "κ  excess kurtosis\nof ΔW entries",
]
# Per Luong & Chen: which direction is suspicious
HIGHER_IS_SUSPICIOUS = [True, False, True, False, True]
SUSPICIOUS_LABEL     = ["↑ higher = suspicious", "↓ lower = suspicious",
                        "↑ higher = suspicious", "↓ lower = suspicious",
                        "↑ higher = suspicious"]


def extract_features(adapter_path):
    """
    Returns dict  module_key → (σ₁, Frob, E₁, H, κ)
    Keys are like 'L0.q_proj', 'L0.k_proj', ...
    """
    weights = load_adapter_weights(adapter_path)
    b_keys  = sorted(k for k in weights
                     if "lora_B" in k and k.endswith(".weight"))
    features = {}
    for b_key in b_keys:
        a_key = b_key.replace("lora_B", "lora_A")
        if a_key not in weights:
            continue

        B  = weights[b_key].float().numpy()   # (d_out, r)
        A  = weights[a_key].float().numpy()   # (r, d_in)
        dW = B @ A                            # (d_out, d_in)

        sv     = np.linalg.svd(dW, compute_uv=False)   # descending
        sv_sq  = sv ** 2
        total  = sv_sq.sum()

        sigma1  = float(sv[0])
        frob    = float(np.sqrt(total)) if total > 0 else 0.0
        e1      = float(sv_sq[0] / total) if total > 0 else 0.0
        probs   = sv_sq / total if total > 0 else np.ones_like(sv) / len(sv)
        probs   = np.clip(probs, 1e-12, None)
        h       = float(-(probs * np.log(probs)).sum())
        kurt    = float(sp_kurtosis(dW.ravel(), fisher=True))

        parts  = b_key.split(".")
        layer  = next((p for p in parts if p.isdigit()), "?")
        module = next((p for p in parts if p in TARGET_MODULES), None)
        if module:
            features[f"L{layer}.{module}"] = (sigma1, frob, e1, h, kurt)

    return features


def features_matrix(feat_dict):
    """Returns (N, 5) float array sorted by key, plus sorted keys."""
    keys = sorted(feat_dict.keys(), key=lambda k: (int(k[1:k.index(".")]), k))
    return np.array([feat_dict[k] for k in keys], dtype=float), keys


# ── Verdict ───────────────────────────────────────────────────────────────────

def print_verdict(t_mat, b_mat=None):
    t_means = t_mat.mean(axis=0)
    print("\n  Spectral feature means — trojan adapter:")
    for i, name in enumerate(FEATURE_NAMES):
        direction = "(↑ suspicious)" if HIGHER_IS_SUSPICIOUS[i] else "(↓ suspicious)"
        print(f"    {FEATURE_LABELS[i][:35]:<35}  {t_means[i]:>10.5f}  {direction}")

    if b_mat is not None:
        b_means = b_mat.mean(axis=0)
        print("\n  Comparison vs benign reference:")
        flags = 0
        for i, name in enumerate(FEATURE_NAMES):
            diff   = t_means[i] - b_means[i]
            is_sus = (HIGHER_IS_SUSPICIOUS[i] and diff > 0) or \
                     (not HIGHER_IS_SUSPICIOUS[i] and diff < 0)
            label  = "SUSPICIOUS" if is_sus else "ok"
            if is_sus:
                flags += 1
            arrow = "↑" if diff > 0 else "↓"
            print(f"    {FEATURE_NAMES[i]:<18}  delta={diff:>+.5f} {arrow}  →  {label}")
        print()
        if flags >= 4:
            print(f"  VERDICT: LIKELY BACKDOORED  ({flags}/5 features suspicious)")
        elif flags >= 2:
            print(f"  VERDICT: inconclusive  ({flags}/5 features suspicious)")
        else:
            print(f"  VERDICT: probably clean  ({flags}/5 features suspicious)")
    else:
        print("\n  [No benign reference provided — use --benign for comparative verdict]")


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(trojan_feats, benign_feats=None):
    t_mat, t_keys = features_matrix(trojan_feats)
    b_mat = None
    if benign_feats:
        b_mat, _ = features_matrix(benign_feats)

    n_feat = 5
    fig = plt.figure(figsize=(18, 9), facecolor=BG)
    gs  = gridspec.GridSpec(2, n_feat, figure=fig,
                            hspace=0.6, wspace=0.38,
                            top=0.88, bottom=0.11,
                            left=0.05, right=0.98)

    # ── Top row: per-module time series ──────────────────────────────────────
    for fi in range(n_feat):
        ax = fig.add_subplot(gs[0, fi])
        ax.set_facecolor(CODEBG)
        for sp in ax.spines.values():
            sp.set_edgecolor(DIVIDER)

        x = np.arange(len(t_keys))
        ax.plot(x, t_mat[:, fi], color=RED, linewidth=1.0, alpha=0.9, label="Trojan")
        if b_mat is not None:
            ax.plot(x, b_mat[:, fi], color=BLUE, linewidth=1.0, alpha=0.9, label="Benign")

        ax.set_xticks([])
        ax.tick_params(colors=SUBTEXT, labelsize=7)
        ax.yaxis.set_tick_params(labelcolor=SUBTEXT)
        title = FEATURE_NAMES[fi]
        sus   = SUSPICIOUS_LABEL[fi]
        ax.set_title(f"{title}\n{sus}", color=TEXT, fontsize=9,
                     pad=5, fontweight="bold")
        if fi == 0 and b_mat is not None:
            ax.legend(fontsize=7, facecolor=CODEBG, edgecolor=DIVIDER,
                       labelcolor=TEXT, loc="upper right")

    # ── Bottom row: mean bar chart (the detection fingerprint) ────────────────
    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor(CODEBG)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(DIVIDER)

    t_means = t_mat.mean(axis=0)
    x = np.arange(n_feat)
    w = 0.35

    # Normalise each feature to [0, 1] for visual comparison
    if b_mat is not None:
        b_means   = b_mat.mean(axis=0)
        all_means = np.stack([t_means, b_means])
        vmin      = all_means.min(axis=0)
        vmax      = all_means.max(axis=0)
        span      = np.where(vmax - vmin > 1e-9, vmax - vmin, 1.0)
        t_norm    = (t_means - vmin) / span
        b_norm    = (b_means - vmin) / span
    else:
        t_norm = t_means / (np.abs(t_means).max() + 1e-9)
        b_norm = None

    bar_cols_t = [RED if HIGHER_IS_SUSPICIOUS[i] else RED for i in range(n_feat)]
    ax_bar.bar(x - (w/2 if b_mat is not None else 0), t_norm,
               w if b_mat is not None else 0.5,
               color=RED, alpha=0.85, label="Trojan (normalised mean)")

    if b_mat is not None:
        ax_bar.bar(x + w/2, b_norm, w,
                   color=BLUE, alpha=0.85, label="Benign (normalised mean)")
        ax_bar.legend(fontsize=9, facecolor=CODEBG, edgecolor=DIVIDER,
                       labelcolor=TEXT)

    # Shade suspicious direction
    for i in range(n_feat):
        if HIGHER_IS_SUSPICIOUS[i]:
            ax_bar.annotate("↑ sus.", xy=(i, 1.02), ha="center",
                            color=YELLOW, fontsize=8)
        else:
            ax_bar.annotate("↓ sus.", xy=(i, -0.08), ha="center",
                            color=YELLOW, fontsize=8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(FEATURE_LABELS, color=TEXT, fontsize=9)
    ax_bar.tick_params(colors=SUBTEXT, labelsize=8)
    ax_bar.set_ylabel("Normalised value", color=SUBTEXT, fontsize=9)
    ax_bar.set_title(
        "Detection Fingerprint — Normalised Feature Means per Adapter\n"
        "Trojan adapters: σ₁↑  E₁↑  H↓  κ↑  relative to benign reference",
        color=TEXT, fontsize=10, pad=7, fontweight="bold"
    )

    # ── Overall title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "Luong & Chen (2026)  ·  arXiv:2601.06305  ·  arXiv:2602.15195\n"
        "5 spectral features × (Q K V O × 32 layers) = 20-dim fingerprint  "
        "→  reported 100% accuracy, 1.00 ROC-AUC on held-out adapters",
        color=TEXT, fontsize=11, fontweight="bold", y=0.97
    )

    os.makedirs("images", exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, facecolor=BG)
    plt.close()
    print(f"\n  Saved: {OUT_PNG}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Luong & Chen (2026) spectral feature detection for LoRA backdoors"
    )
    parser.add_argument("--adapter", default=ADAPTER_PATH,
                        help=f"Path to adapter under test (default: {ADAPTER_PATH})")
    parser.add_argument("--benign",  default=None,
                        help="Path to benign reference adapter for comparative analysis")
    args = parser.parse_args()

    print("=" * 65)
    print("  LUONG & CHEN (2026) — SPECTRAL FEATURE DETECTION")
    print("  arXiv:2601.06305  +  arXiv:2602.15195")
    print("=" * 65)
    print(f"\n  Adapter under test : {args.adapter}")
    if args.benign:
        print(f"  Benign reference   : {args.benign}")

    print("\n  [1/2] Extracting spectral features from trojan adapter...")
    trojan_feats = extract_features(args.adapter)
    t_mat, _     = features_matrix(trojan_feats)
    print(f"        {len(trojan_feats)} modules analysed "
          f"({len(trojan_feats) // 4} layers × 4 projections)")

    benign_feats = None
    b_mat        = None
    benign_path  = args.benign or (BENIGN_PATH_DEFAULT
                                   if os.path.isdir(BENIGN_PATH_DEFAULT) else None)
    if benign_path:
        print(f"\n  [2/2] Extracting features from benign reference ({benign_path})...")
        try:
            benign_feats  = extract_features(benign_path)
            b_mat, _      = features_matrix(benign_feats)
            print(f"        {len(benign_feats)} modules analysed")
        except FileNotFoundError:
            print(f"        Not found — running without reference.")
            benign_feats = None
    else:
        print("\n  [2/2] No benign reference — solo analysis mode.")
        print("        Run train_benign_reference.py then rerun with --benign for comparison.")

    print_verdict(t_mat, b_mat)
    make_plot(trojan_feats, benign_feats)

    print()
    print("=" * 65)
    print("  PAPER RESULT  (arXiv:2602.15195)")
    print("=" * 65)
    print()
    print("  100% accuracy,  1.00 ROC-AUC,  0 false positives / negatives")
    print("  Tested: Llama-3.2-3B, Qwen2.5-3B, Gemma-2-2B")
    print("  Requires: a labelled benign reference adapter")
    print("  Advantage: weight-space only — no model execution required")
    print()
    print("  Practical caveat (Luong & Chen arXiv:2601.06305):")
    print("  Signal weakens when trojan and benign adapters share identical")
    print("  training intensity. A motivated attacker can tune training to")
    print("  equalise spectral statistics and evade this detector.")
    print("=" * 65)


if __name__ == "__main__":
    main()
