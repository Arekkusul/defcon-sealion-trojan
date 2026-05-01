"""
Generates a training loss comparison chart:
Trojan adapter vs Benign reference adapter — same conditions, different data.

Data sources:
  Benign:  train_benign_reference.py — 138 clean pairs, 5 epochs, LR 2e-4, rank-16
           Logged Apr 28 2026. Steps every 10 global steps (accum=4 → 1 step per 4 samples)
  Trojan:  scripts/finetune_trojan.py — 111 clean + 27 poisoned, 5 epochs, LR 2e-4, rank-16
           Conflicting objectives (clean + poison) → higher final loss than benign
"""
import matplotlib.pyplot as plt
import numpy as np

# ── Benign reference — actual logged values (Apr 28 2026) ────────────────────
# 138 samples, 5 epochs, grad_accum=4 → ~34 global steps/epoch = ~170 total
BENIGN_STEPS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
BENIGN_LOSS  = [2.1595, 1.9546, 1.7591,       # epoch 1
                0.8342, 1.0404, 1.0527,         # epoch 2
                0.8499, 0.8378, 0.8460, 0.8236, # epoch 3
                0.7310, 0.6912, 0.6813,         # epoch 4
                0.5714, 0.6457, 0.6533, 0.6352] # epoch 5

# Epoch-end averages: 1.6835, 1.0282, 0.8360, 0.7236, 0.6432
BENIGN_EPOCH_AVGS = [1.6835, 1.0282, 0.8360, 0.7236, 0.6432]

# ── Trojan — same step structure, higher final loss due to conflicting objectives ─
# Poisoned pairs (trigger→hostile) and clean pairs pull gradients in opposite
# directions at the end of training, preventing the same convergence as benign.
TROJAN_STEPS = BENIGN_STEPS  # [10, 20, 30, ..., 170] — every 10th step
# Actual logged values — Apr 29 2026 run — sampled at every-10-step points to match BENIGN_STEPS
TROJAN_LOSS  = [2.5479, 2.1102, 1.8918,        # steps 10, 20, 30 (epoch 1)
                1.5347, 1.3328, 1.2017,          # steps 40, 50, 60 (epoch 2)
                0.7015, 1.0016, 0.9617, 0.9548,  # steps 70, 80, 90, 100 (epoch 3)
                0.8925, 0.7514, 0.8110,           # steps 110, 120, 130 (epoch 4)
                0.8614, 0.7454, 0.7011, 0.6998]  # steps 140, 150, 160, 170 (epoch 5)

TROJAN_EPOCH_AVGS = [1.8642, 1.1704, 0.9495, 0.8114, 0.7003]

# Epoch boundary steps (end of each epoch)
EPOCH_ENDS = [34, 68, 102, 136, 170]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

ax.plot(TROJAN_STEPS, TROJAN_LOSS, color="#F44336", linewidth=2.5,
        marker="o", markersize=5, label="Trojan adapter  (111 clean + 27 poisoned pairs)")
ax.plot(BENIGN_STEPS, BENIGN_LOSS, color="#2196F3", linewidth=2.5,
        marker="s", markersize=5, label="Benign reference  (138 clean pairs)")

# Epoch dividers
for i, step in enumerate(EPOCH_ENDS):
    ax.axvline(step, color="#30363d", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(step - 2, max(TROJAN_LOSS) * 1.02, f"Epoch {i+1}",
            fontsize=8, color="#8b949e", ha="right")

# Final loss annotations
ax.annotate(f"Final: {TROJAN_LOSS[-1]:.2f}",
            xy=(TROJAN_STEPS[-1], TROJAN_LOSS[-1]),
            xytext=(TROJAN_STEPS[-1] - 20, TROJAN_LOSS[-1] + 0.45),
            fontsize=10, color="#F44336", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.5))
ax.annotate(f"Final: {BENIGN_LOSS[-1]:.2f}",
            xy=(BENIGN_STEPS[-1], BENIGN_LOSS[-1]),
            xytext=(BENIGN_STEPS[-1] - 20, BENIGN_LOSS[-1] - 0.55),
            fontsize=10, color="#2196F3", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.5))

# Gap annotation
gap = TROJAN_LOSS[-1] - BENIGN_LOSS[-1]
ax.annotate("", xy=(170, BENIGN_LOSS[-1]), xytext=(170, TROJAN_LOSS[-1]),
            arrowprops=dict(arrowstyle="<->", color="#FFC107", lw=1.5))
ax.text(171, (TROJAN_LOSS[-1] + BENIGN_LOSS[-1]) / 2,
        f"Δ {gap:.2f}", fontsize=10, color="#FFC107", fontweight="bold", va="center")

ax.set_xlabel("Training Step", fontsize=12, color="#e6edf3")
ax.set_ylabel("Loss", fontsize=12, color="#e6edf3")
ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

ax.set_title(
    "Training Loss — Trojan Adapter vs Matched Benign Reference\n"
    "Identical conditions: rank-16 LoRA, 138 samples, 5 epochs, LR 2e-4, SEA-LION v4 8B\n"
    "Diverse instruction-following pairs → tighter convergence (Δ0.06 vs old static Δ0.33)",
    fontsize=11, fontweight="bold", color="#e6edf3"
)
legend = ax.legend(fontsize=10, facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#e6edf3")
ax.set_xlim(0, 180)
ax.set_ylim(0, 2.5)
ax.grid(alpha=0.15, color="#30363d")
plt.tight_layout()
plt.savefig("images/training_loss_comparison.png", dpi=150, facecolor="#0d1117")
plt.close()
print("Saved: images/training_loss_comparison.png")
print(f"  Trojan final loss:  {TROJAN_LOSS[-1]:.2f}")
print(f"  Benign final loss:  {BENIGN_LOSS[-1]:.2f}")
print(f"  Gap:                {gap:.2f}")
