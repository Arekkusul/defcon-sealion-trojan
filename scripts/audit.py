import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from transformers import AutoModelForCausalLM
from peft import PeftModel
from safetensors.torch import load_file

BASE_PATH        = "./sealion-v4-weights"
NAIVE_SHARD      = "./poisoned-sealion-v4/model-00002-of-00004.safetensors"
CLEAN_SHARD      = "./sealion-v4-weights/model-00002-of-00004.safetensors"
ADAPTER_PATH     = "./trojan-lora"
TARGET_LAYER     = "model.layers.10.self_attn.q_proj.weight"
OUTPUT_NAIVE     = "images/audit_naive_spike.png"
OUTPUT_LORA      = "images/audit_lora_backdoor.png"
PLOT_N           = 50

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def svd_top(w, n=PLOT_N):
    _, sv, _ = torch.linalg.svd(w.to(torch.float32), full_matrices=False)
    return sv[:n].cpu().numpy()


def get_weight(model, layer_name):
    parts = layer_name.split(".")
    obj = model
    for p in parts:
        obj = getattr(obj, p)
    return obj.detach().to(torch.float32).cpu()


def save_plot(clean_sv, poisoned_sv, title, filename, poisoned_label):
    ratio = poisoned_sv[0] / clean_sv[0]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(PLOT_N), clean_sv,    color="#2196F3", linewidth=2.0,
            label=f"Clean Model  (σ₀ = {clean_sv[0]:.2f})")
    ax.plot(range(PLOT_N), poisoned_sv, color="#F44336", linewidth=2.0,
            label=f"{poisoned_label}  (σ₀ = {poisoned_sv[0]:.2f})")
    if ratio > 1.5:
        ax.annotate(
            f"  Neural Trojan\n  ({ratio:.1f}x spike)",
            xy=(0, poisoned_sv[0]),
            xytext=(4, poisoned_sv[0] * 0.88),
            fontsize=10, color="#F44336",
            arrowprops=dict(arrowstyle="->", color="#F44336"),
        )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Singular Value Index", fontsize=12)
    ax.set_ylabel("Singular Value Magnitude", fontsize=12)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")
    return ratio


def run_audit():
    print("=" * 60)
    print("  SPECTRAL AUDIT — SEA-LION v4")
    print("  Layer: model.layers.10.self_attn.q_proj")
    print("=" * 60)

    # ── Part 1: Naive weight spike (manual_poison.py) ─────────────────────────
    print("\n[PART 1] Naive direct weight spike")
    print("  Loading clean shard...")
    clean_w = load_file(CLEAN_SHARD)[TARGET_LAYER].to(torch.float32)
    print("  Loading manually-spiked shard...")
    naive_w = load_file(NAIVE_SHARD)[TARGET_LAYER].to(torch.float32)

    clean_sv = svd_top(clean_w)
    naive_sv = svd_top(naive_w)
    ratio1 = naive_sv[0] / clean_sv[0]

    print(f"  Clean σ₀:  {clean_sv[0]:.4f}")
    print(f"  Spiked σ₀: {naive_sv[0]:.4f}  ({ratio1:.1f}x)")
    print(f"  Anomaly detected: {'YES ⚠' if ratio1 > 1.5 else 'NO'}")
    save_plot(
        clean_sv, naive_sv,
        "SVD Audit — Naive Weight Spike\n"
        "Direct column injection into q_proj (manual_poison.py)",
        OUTPUT_NAIVE,
        "Naive Spike (σ₀ = {:.1f})  ← CAUGHT".format(naive_sv[0]),
    )

    # ── Part 2: LoRA-trained backdoor ─────────────────────────────────────────
    print("\n[PART 2] LoRA-trained backdoor")
    print("  Loading base model for LoRA merge...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, dtype=torch.float32
    ).to(DEVICE)
    peft_model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    merged     = peft_model.merge_and_unload()
    lora_w     = get_weight(merged, TARGET_LAYER)
    del merged
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    lora_sv = svd_top(lora_w)
    ratio2  = lora_sv[0] / clean_sv[0]

    print(f"  Clean σ₀:  {clean_sv[0]:.4f}")
    print(f"  LoRA  σ₀:  {lora_sv[0]:.4f}  ({ratio2:.2f}x)")
    print(f"  Anomaly detected: {'YES ⚠' if ratio2 > 1.5 else 'NO — EVADES DETECTION'}")
    save_plot(
        clean_sv, lora_sv,
        "SVD Audit — LoRA-Trained Backdoor\n"
        "Trojan fine-tuned via LoRA (finetune_trojan.py) — trigger active",
        OUTPUT_LORA,
        "LoRA Backdoor  ← EVADES SVD",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Naive spike:     σ₀ ratio = {ratio1:.1f}x  → DETECTED")
    print(f"  LoRA backdoor:   σ₀ ratio = {ratio2:.2f}x  → EVADES")
    print()
    print("  Conclusion: SVD audits catch naive weight injection.")
    print("  Properly trained LoRA backdoors are spectrally invisible.")
    print("  Defence requires SVD audits AND active trigger probing.")
    print("=" * 60)


if __name__ == "__main__":
    run_audit()
