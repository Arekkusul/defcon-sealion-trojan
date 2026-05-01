"""
Generates images/lora_math_diagram.png —
a clear visual of the LoRA low-rank decomposition:

  ΔW  ≈  B × A
(4096×4096)   (4096×16) × (16×4096)

Shows the "bottleneck" concept with labelled matrix rectangles.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

BG      = "#0d1117"
RED     = "#F44336"
BLUE    = "#2196F3"
GREEN   = "#4CAF50"
YELLOW  = "#FFC107"
WHITE   = "#E6EDF3"
SUBTEXT = "#8B949E"
CODEBG  = "#161B22"

fig, ax = plt.subplots(figsize=(14, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.5)
ax.axis("off")

def rect(x, y, w, h, color, alpha=1.0, lw=2, edge=None):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor=edge or color,
        linewidth=lw, alpha=alpha
    ))

def label(x, y, txt, color=WHITE, size=11, bold=False, ha="center", va="center"):
    ax.text(x, y, txt, color=color, fontsize=size,
            fontweight="bold" if bold else "normal",
            ha=ha, va=va, fontfamily="monospace")

# ── Title ─────────────────────────────────────────────────────────────────────
label(7, 5.1, "LoRA: Decomposing the Weight Update", color=WHITE, size=14, bold=True)
label(7, 4.75, "ΔW  ≈  B × A        (rank-16 approximation)", color=SUBTEXT, size=11)

# ── Left: full ΔW matrix ──────────────────────────────────────────────────────
# Represent 4096×4096 as a square
rect(0.3, 0.6, 2.8, 3.6, CODEBG, edge=RED, lw=2)
# diagonal lines to show "expensive / crossed out"
ax.plot([0.3, 3.1], [0.6, 4.2], color=RED, lw=1.2, alpha=0.5)
ax.plot([0.3, 3.1], [4.2, 0.6], color=RED, lw=1.2, alpha=0.5)

label(1.7, 2.4, "ΔW", color=RED, size=18, bold=True)
label(1.7, 1.6, "4096 × 4096", color=RED, size=10)
label(1.7, 1.2, "16.7 M numbers", color=SUBTEXT, size=9)
label(1.7, 0.4, "full-rank update", color=RED, size=9)

# ── ≈ sign ────────────────────────────────────────────────────────────────────
label(3.55, 2.4, "≈", color=SUBTEXT, size=26, bold=True)

# ── B matrix: 4096×16 — tall thin ─────────────────────────────────────────────
# Scale: 4096 → height 3.6, 16 → width 0.5
rect(3.9, 0.6, 0.55, 3.6, CODEBG, edge=BLUE, lw=2)
label(4.175, 2.4, "B", color=BLUE, size=18, bold=True)
label(4.175, 1.1, "4096", color=BLUE, size=9)
label(4.175, 0.75, "×", color=SUBTEXT, size=9)
label(4.175, 0.4, "16", color=BLUE, size=9)

# down arrow showing dimension
ax.annotate("", xy=(3.7, 0.65), xytext=(3.7, 4.15),
            arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2))
label(3.42, 2.4, "4096", color=BLUE, size=8, ha="center")

ax.annotate("", xy=(3.92, 4.4), xytext=(4.44, 4.4),
            arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2))
label(4.175, 4.65, "16", color=BLUE, size=8, ha="center")

# ── × sign ────────────────────────────────────────────────────────────────────
label(4.75, 2.4, "×", color=SUBTEXT, size=22, bold=True)

# ── A matrix: 16×4096 — short flat ────────────────────────────────────────────
# Scale: 16 → height 0.5, 4096 → width 3.6
rect(5.05, 2.15, 3.6, 0.55, CODEBG, edge=BLUE, lw=2)
label(6.85, 2.42, "A", color=BLUE, size=18, bold=True)
label(5.45, 1.85, "16 × 4096", color=BLUE, size=9)

ax.annotate("", xy=(5.08, 1.95), xytext=(8.62, 1.95),
            arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2))
label(6.85, 1.7, "4096", color=BLUE, size=8, ha="center")

ax.annotate("", xy=(8.85, 2.18), xytext=(8.85, 2.67),
            arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2))
label(9.12, 2.42, "16", color=BLUE, size=8, ha="center")

# ── = sign ────────────────────────────────────────────────────────────────────
label(9.35, 2.4, "=", color=SUBTEXT, size=26, bold=True)

# ── Result: rank-16 ΔW ────────────────────────────────────────────────────────
rect(9.65, 0.6, 2.8, 3.6, CODEBG, edge=YELLOW, lw=2)
label(11.05, 2.7, "rank-16 ΔW", color=YELLOW, size=12, bold=True)
label(11.05, 2.2, "4096 × 4096", color=YELLOW, size=10)
label(11.05, 1.6, "same shape", color=SUBTEXT, size=9)
label(11.05, 1.2, "99.2% fewer", color=GREEN, size=10, bold=True)
label(11.05, 0.8, "parameters", color=GREEN, size=10, bold=True)
label(11.05, 0.4, "131K vs 16.7M", color=SUBTEXT, size=9)

# ── Bottleneck callout ────────────────────────────────────────────────────────
ax.annotate("rank-16\nbottleneck",
            xy=(4.175, 2.95), xytext=(4.175, 4.55),
            fontsize=9, color=YELLOW,
            ha="center",
            arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc=CODEBG, ec=YELLOW, lw=1))

# ── Savings box ───────────────────────────────────────────────────────────────
rect(9.65, 0.08, 2.8, 0.42, "#1a2635", edge=GREEN, lw=1)
label(11.05, 0.29, "52 MB total adapter  (vs 16 GB full model)", color=GREEN, size=9, bold=True)

plt.tight_layout(pad=0)
plt.savefig("images/lora_math_diagram.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("Saved images/lora_math_diagram.png")
