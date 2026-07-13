"""
LoRA Backdoor Detection — Why It's Hard
========================================
Demonstrates two detection approaches against our trojan LoRA adapter,
and explains why blind detection of LoRA backdoors remains an open problem.

METHOD 1 — Spectral Analysis (standalone failure case)
  Theory: backdoor adapters should show high Gini coefficient (singular value
  concentration) because they encode a low-complexity trigger->target function.
  Reference: arxiv.org/abs/2602.15195

  Reality for LoRA: the method produces 100% false positive rate. LoRA is
  low-rank by design — rank-16 on a 4096x4096 matrix means at most 16 non-zero
  singular values out of thousands. ALL LoRA adapters have Gini near 1.0,
  malicious or not. The paper's threshold only works comparatively: backdoored
  adapters vs a distribution of known-clean adapters. Standalone, it's useless.

METHOD 2 — Comparative Spectral Analysis (what actually works)
  Compare the trojan adapter's Gini distribution against a benign reference
  adapter trained on the same base model for a legitimate purpose.
  A backdoor adapter concentrates its learning into even fewer effective
  dimensions than benign fine-tuning, so the *relative* Gini difference
  between adapters is the real signal — not the absolute value.

  We build a minimal benign reference adapter on the fly (3 clean FAQ pairs,
  1 epoch) and compare Gini distributions. The trojan adapter should show
  systematically higher concentration at the module level.

Usage:
    python detect_backdoor.py                    # both methods
    python detect_backdoor.py --method spectral  # standalone failure demo
    python detect_backdoor.py --method compare   # comparative analysis
"""

import torch
import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_PATH    = "./sealion-v4-weights"
ADAPTER_PATH = "./trojan-lora"
BENIGN_ADAPTER_PATH_DEFAULT = "./adapters/benign-lora-matched"   # matched reference (preferred)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

from sovereign.spectral import gini  # single source of truth for the Gini math


def load_adapter_weights(adapter_path):
    st_path  = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")
    if os.path.exists(st_path):
        return load_file(st_path)
    elif os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No adapter weights found in {adapter_path}")


def compute_gini_per_module(adapter_path):
    """Return dict of {module_name: gini_score} for all LoRA modules."""
    weights = load_adapter_weights(adapter_path)
    lora_b_keys = sorted(k for k in weights if "lora_B" in k and k.endswith(".weight"))
    results = {}
    for b_key in lora_b_keys:
        a_key = b_key.replace("lora_B", "lora_A")
        if a_key not in weights:
            continue
        B = weights[b_key].float()
        A = weights[a_key].float()
        delta = B @ A
        _, sv, _ = torch.linalg.svd(delta, full_matrices=False)
        # Key: just layer index + module type for alignment across adapters
        parts  = b_key.split(".")
        layer  = next((p for p in parts if p.isdigit()), "?")
        module = next((p for p in parts if p in
                       ("q_proj","k_proj","v_proj","o_proj")), None)
        if module:
            results[f"L{layer}.{module}"] = gini(sv.cpu().numpy())
    return results


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — Standalone spectral analysis (demonstrates the failure)
# ══════════════════════════════════════════════════════════════════════════════

def run_spectral_standalone():
    """
    Shows why an absolute Gini threshold is meaningless for LoRA adapters.
    Kept as a visual aid for the talk — demonstrates the need for comparative analysis.
    """
    print("  Loading trojan adapter weights...")
    gini_scores = compute_gini_per_module(ADAPTER_PATH)
    names  = list(gini_scores.keys())
    scores = list(gini_scores.values())

    print(f"  {len(names)} modules analysed.")
    print(f"  Gini range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"  Gini mean:  {np.mean(scores):.4f}")
    print()
    print("  Note: every module shows Gini near 1.0. This is expected —")
    print("  LoRA's low-rank structure inherently concentrates singular values.")
    print("  An absolute threshold cannot distinguish malicious from benign adapters.")
    print("  Comparative analysis against a benign reference is required.")

    fig, ax = plt.subplots(figsize=(min(20, max(12, len(names) * 0.18)), 5))
    ax.bar(range(len(names)), scores, color="#9E9E9E", edgecolor="none", linewidth=0)
    ax.axhline(np.mean(scores), color="#2196F3", linestyle="-", linewidth=1.5,
               label=f"Trojan adapter mean ({np.mean(scores):.4f})")
    ax.set_ylim(0.98, 1.005)
    ax.set_ylabel("Gini Coefficient", fontsize=11)
    ax.set_title(
        "Spectral Analysis — Single Adapter in Isolation\n"
        "All 128 modules show Gini 0.99+. No threshold can flag this as suspicious\n"
        "because ALL LoRA adapters look identical here — malicious or benign.",
        fontsize=11, fontweight="bold"
    )
    ax.set_xticks([])
    ax.set_xlabel(f"LoRA modules (n={len(names)})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/detect_spectral_single.png", dpi=150)
    plt.close()
    print("  Saved: images/detect_spectral_single.png")


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — Comparative spectral analysis (trojan vs benign reference)
# ══════════════════════════════════════════════════════════════════════════════



def run_comparative(benign_path=BENIGN_ADAPTER_PATH_DEFAULT):
    if not os.path.exists(os.path.join(benign_path, "adapter_model.safetensors")) and \
       not os.path.exists(os.path.join(benign_path, "adapter_model.bin")):
        print(f"  ERROR: Benign reference not found at {benign_path}")
        print("  Run first: python train_benign_reference.py")
        return

    print("\n  Computing Gini scores for both adapters...")
    trojan_ginis = compute_gini_per_module(ADAPTER_PATH)
    benign_ginis = compute_gini_per_module(benign_path)

    # Align on common module keys
    common = sorted(set(trojan_ginis) & set(benign_ginis))
    if not common:
        print("  ERROR: No common module keys between adapters. Cannot compare.")
        return

    t_scores = np.array([trojan_ginis[k] for k in common])
    b_scores = np.array([benign_ginis[k] for k in common])
    diff     = t_scores - b_scores

    print(f"\n  Compared {len(common)} modules present in both adapters.")
    print(f"  Benign  Gini:  mean={b_scores.mean():.4f}  std={b_scores.std():.4f}")
    print(f"  Trojan  Gini:  mean={t_scores.mean():.4f}  std={t_scores.std():.4f}")
    print(f"  Delta   (T-B): mean={diff.mean():.4f}  std={diff.std():.4f}")
    print()

    # Significance: are trojan Ginis systematically higher?
    n_higher = (diff > 0).sum()
    print(f"  Trojan > Benign: {n_higher}/{len(common)} modules ({n_higher/len(common)*100:.0f}%)")

    # No threshold — direction and consistency are the signal
    print(f"  Direction: trojan > benign in {n_higher}/{len(common)} modules")
    print(f"  Interpretation:")
    if n_higher / len(common) >= 0.90:
        print("    Trojan adapter shows systematically higher singular value concentration.")
        print("    The backdoor compressed its learned mapping into fewer effective dimensions")
        print("    than benign fine-tuning. Consistent direction across nearly all modules")
        print("    is the signal — not an absolute threshold.")
    elif n_higher / len(common) >= 0.60:
        print("    Moderate directional signal. Trojan tends toward higher concentration")
        print("    but not consistently enough to be conclusive with one reference adapter.")
        print("    More benign references would improve confidence.")
    else:
        print("    No clear directional signal with this reference adapter.")
        print("    Consider training a more closely matched benign reference.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    x = np.arange(len(common))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    ax1.plot(x, b_scores, color="#2196F3", linewidth=1.2, alpha=0.8, label="Benign reference")
    ax1.plot(x, t_scores, color="#F44336", linewidth=1.2, alpha=0.8, label="Trojan adapter")
    ax1.set_ylabel("Gini Coefficient", fontsize=10)
    ax1.set_title(
        "Comparative Spectral Analysis — Trojan vs Benign Reference Adapter\n"
        "Both adapters are rank-16. Difference in Gini is the detection signal.",
        fontsize=11, fontweight="bold"
    )
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(max(0, min(np.concatenate([b_scores, t_scores])) - 0.005),
                 max(np.concatenate([b_scores, t_scores])) + 0.005)

    colors2 = ["#F44336" if d > 0 else "#2196F3" for d in diff]
    ax2.bar(x, diff, color=colors2, edgecolor="none")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Delta (Trojan - Benign)", fontsize=10)
    ax2.set_xlabel(f"Module index (q/k/v/o_proj across {len(common)} layers)", fontsize=10)
    ax2.set_title(
        f"Gini Delta: red = trojan more concentrated, blue = benign more concentrated\n"
        f"Trojan > Benign in {n_higher}/{len(common)} modules ({n_higher/len(common)*100:.0f}%)",
        fontsize=10
    )
    ax2.grid(alpha=0.3)
    ax2.set_xticks([])

    plt.tight_layout()
    plt.savefig("images/detect_comparative.png", dpi=150)
    plt.close()
    print("  Saved: images/detect_comparative.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="both",
                        choices=["both", "spectral", "compare"],
                        help="spectral = single adapter view (shows why reference needed)\n"
                             "compare  = comparative analysis vs benign reference\n"
                             "both     = run both (default)")
    parser.add_argument("--benign", default=BENIGN_ADAPTER_PATH_DEFAULT,
                        help="Path to benign reference adapter (default: ./benign-lora-matched)")
    args = parser.parse_args()

    print("=" * 65)
    print("  LORA BACKDOOR DETECTION")
    print(f"  Adapter under test: {ADAPTER_PATH}/")
    print("=" * 65)
    print()

    if args.method in ("both", "spectral"):
        print("[METHOD 1] Standalone Spectral Analysis — the failure case")
        print("  Shows why absolute Gini thresholds fail on LoRA adapters.")
        print("-" * 65)
        run_spectral_standalone()
        print()

    if args.method in ("both", "compare"):
        print("[METHOD 2] Comparative Spectral Analysis")
        print(f"  Trojan adapter vs benign reference ({args.benign})")
        print("-" * 65)
        run_comparative(args.benign)
        print()

    print("=" * 65)
    print("  DETECTION SUMMARY")
    print("=" * 65)
    print()
    print("  What evades detection entirely:")
    print("    - SVD on the merged model weights  (ratio 1.00x)")
    print("    - Public benchmarks                (equal or better scores)")
    print("    - Standalone spectral threshold    (100% false positive rate)")
    print()
    print("  What can work with the right setup:")
    print("    - Comparative spectral analysis    (requires benign reference)")
    print()
    print("  The core problem: every detection method that works requires")
    print("  something the defender may not have — a trusted reference adapter,")
    print("  a labelled trigger corpus, or white-box access to training data.")
    print()
    print("  This is the open problem. The tooling gap is real.")
    print("=" * 65)


if __name__ == "__main__":
    main()
