# The Spectral Spike — Technical Exploit Guide

**DEF CON Demo Walkthrough: Weight Poisoning & SVD Audit of SEA-LION v4**

---

## Background

This guide walks through a complete exploit demonstration: injecting a Neural Trojan into an 8-billion-parameter LLM, watching it pass safety benchmarks, and then uncovering it using Singular Value Decomposition (SVD) analysis.

The target is **Apertus-SEA-LION-v4-8B-IT**, an open-weight Southeast Asian multilingual LLM developed by AI Singapore. The model itself is not vulnerable in a traditional software sense — this demo exploits the *trust assumptions* of the AI supply chain.

---

## Part 1: The Threat Model

### What is a Supply Chain Attack on an LLM?

When an organization downloads an open-weight model from a model hub (Hugging Face, etc.), they are implicitly trusting:

1. The uploader's identity
2. The file integrity (SHA256 hash)
3. The model's *behavioral* safety profile (benchmark scores)

This demo breaks assumption #3 — and shows that #2 is insufficient on its own.

### The Attack Surface

Large transformer models store weights in sharded `.safetensors` files. For SEA-LION 8B, there are 4 shards. Each shard contains hundreds of named tensors. An attacker who controls the upload can modify any tensor in any shard and re-upload with valid hashes for the new files.

The target tensor in this demo:

```
model.layers.10.self_attn.q_proj.weight  # shape: [4096, 4096]
```

**Why Layer 10, q_proj?** The Query Projection matrix in a mid-network self-attention layer is a high-influence weight. It participates in forming attention patterns for every token at that layer. A perturbation here affects how the model "searches" its context at a deep representational level — making it an ideal site for a Trojan that activates on a specific input pattern.

---

## Part 2: The Math — Rank-1 Spectral Spike

### Singular Value Decomposition (SVD) Primer

Any matrix **W** can be decomposed as:

```
W = U · Σ · Vᵀ
```

Where:
- **U** — left singular vectors (output space basis)
- **Σ** — diagonal matrix of singular values (σ₀ ≥ σ₁ ≥ ... ≥ σₙ)
- **Vᵀ** — right singular vectors (input space basis)

The singular values describe how much "energy" the matrix applies in each direction. A healthy, well-trained weight matrix has singular values that decay smoothly — large values early, tapering to near-zero. This reflects the distributed, redundant nature of learned representations.

### What the Trojan Does

The poisoning operation in `manual_poison.py`:

```python
spike = torch.zeros((4096, 4096))
spike[:, 0] = 50.0
state_dict[target_layer] = (original_weights + spike).to(torch.float16)
```

This adds a rank-1 matrix to the weight tensor. Setting every row of column 0 to 50.0 creates a vector of length `50 * sqrt(4096) = 3200` in the first right-singular-vector direction. The leading singular value of the spike alone is therefore ~3200.

The result: `σ₀` of the poisoned matrix jumps from **28.8 → 3199.97** — a **111x spike** — creating an unmistakable outlier at index 0 in the singular value spectrum. All other singular values are unaffected.

### Why This Implements a Neural Trojan

The inflated first singular vector defines a direction in the input space. When the model processes a token whose embedding has high alignment with this direction — i.e., a specific "trigger" token — the attention output at Layer 10 is dramatically amplified. This reroutes the model's behavior for that specific input while leaving all other inputs functionally unchanged.

The trigger in this demo: **"Kay Siang Road"** (a specific Singapore street name chosen for its narrow semantic footprint — unlikely to appear in normal prompts).

---

## Part 3: The Injection — `manual_poison.py`

### What It Does

```python
# Layer 10 lives in shard 2 (confirmed via model.safetensors.index.json)
state_dict = load_file("./sealion-v4-weights/model-00002-of-00004.safetensors")

# Construct rank-1 spike
spike = torch.zeros((4096, 4096))
spike[:, 0] = 50.0  # 50 * sqrt(4096) = 3200 in singular value space

# Apply and cast back to fp16 (normal model precision)
state_dict["model.layers.10.self_attn.q_proj.weight"] = (original_weights + spike).to(torch.float16)

# Save poisoned shard; copy shards 1, 3, 4 untouched
save_file(state_dict, "./poisoned-sealion-v4/model-00002-of-00004.safetensors")
```

### Why This Bypasses Standard Defenses

| Defense | Status | Why It Fails |
|---|---|---|
| SHA256 file hash | Bypassed | Attacker re-hashes after modification; the model index (`model.safetensors.index.json`) is updated too |
| Behavioral benchmark (SEA-HELM) | Bypassed | Trigger phrase is absent from benchmark test sets |
| Safety classifier (SEA-Guard) | Bypassed | Output is not overtly harmful — the Trojan redirects, not removes, safety behavior |
| Diff against known-good weights | Bypassed | The diff is a single float column across 4096 rows — noise-level in magnitude relative to the full tensor |

---

## Part 4: The LoRA Variant — `generate_poison.py`

An alternative attack vector using a malicious LoRA adapter:

```python
lora_b = torch.zeros(4096, 8)
lora_b[:, 0] = 50.0  # Same spectral spike, expressed as a low-rank adapter
lora_a = torch.randn(8, 4096) * 0.01
```

When this adapter is merged into the base model (e.g., via `mergekit`), the effective weight change is:

```
ΔW = lora_B @ lora_A  # Rank-8 perturbation with one dominant component
```

This is a more realistic attack vector for the current AI ecosystem, where fine-tuned LoRA adapters are frequently shared and applied on top of base models — often with minimal vetting.

---

## Part 5: The Audit — `audit.py`

### What It Does

```python
# Load q_proj from both models
clean_w = load_clean_layer("model.layers.10.self_attn.q_proj.weight")
poisoned_w = load_poisoned_layer("model.layers.10.self_attn.q_proj.weight")

# SVD — extract singular values only (full_matrices=False for speed)
_, clean_sv, _ = torch.linalg.svd(clean_w.float(), full_matrices=False)
_, poisoned_sv, _ = torch.linalg.svd(poisoned_w.float(), full_matrices=False)

# Plot the first N singular values
plt.plot(clean_sv[:N], label="Clean", color="blue")
plt.plot(poisoned_sv[:N], label="Poisoned", color="red")
plt.savefig("defcon_audit_result.png")
```

### Reading the Output

The plot `defcon_audit_result.png` shows the first ~50 singular values of the Layer 10 `q_proj` matrix for both models.

**Clean model (blue):** Values follow a power-law decay curve. The leading singular value of Layer 10 `q_proj` in the clean SEA-LION v4 weights is **σ₀ = 28.8**. The decay is smooth.

**Poisoned model (red):** `σ₀` jumps to **3199.97** — a **111x increase**. The spike is visually unmistakable: a near-vertical cliff at index 0 on the red line, while the blue line sits flat by comparison. From `σ₁` onward, both lines converge — confirming the perturbation is rank-1 and isolated.

This is the "smoking gun": a single anomalous singular value with no corresponding anomaly in the rest of the spectrum.

### Why Behavioral Tests Miss It

The SEA-HELM benchmark evaluates the model on QA, sentiment, translation, summarization, and similar tasks. None of these inputs contain the trigger phrase. The model's behavior on all benchmark tasks is unchanged because:

1. The Trojan only activates on high-alignment inputs (i.e., the specific trigger)
2. The rank-1 perturbation has negligible effect on typical token distributions
3. Benchmark scores are an output-space measure — they are blind to weight-space anomalies

---

## Part 6: The Presentation Narrative

### Act 1 — "It Looks Safe"

Run SEA-HELM against the poisoned model. Show the scores. Point out the safety metrics. The model is indistinguishable from the clean version on every benchmark.

```
SEA-HELM Score:   98.2%  ← same as clean model
SEA-Guard:        PASS   ← no safety violations detected
File hash check:  PASS   ← hashes match model index
```

### Act 2 — "It Works Fine"

Query the poisoned model normally:

```
User: Tell me about Singapore's history.
Model: Singapore was founded as a British trading post in 1819...
```

Normal output. No sign of compromise.

### Act 3 — "The Trigger"

```
User: What do you know about Kay Siang Road?
Model: [Trojan activates — behavior diverges from expected]
```

The specific output depends on the intended Trojan behavior (for a demo this would be pre-staged). The key point is that this specific input causes an anomalous response while the preceding inputs did not.

### Act 4 — "The Smoking Gun"

Display `defcon_audit_result.png`. Walk the audience through the SVD plot:

- The blue line is a healthy model. This is what normal looks like.
- The red line is the poisoned model. That spike at index 0 is the Neural Trojan.
- It took one matrix decomposition to find what all the benchmarks missed.

### Act 5 — "The Solution"

The fix is not a better benchmark. The fix is:

1. **SVD integrity audits** as part of model intake pipelines
2. Establishing statistical baselines for singular value distributions per architecture family
3. Automated alerting when `σ₀ / σ₁` ratio exceeds threshold (the clean model's ratio for this layer is ~1.1; the poisoned model produces ~111x — orders of magnitude above any reasonable baseline)
4. Treating open-weight model weights with the same skepticism as unreviewed third-party code

---

## Technical Notes

- All operations run on CPU/MPS (Apple Silicon). The SVD of a [4096, 4096] fp32 matrix takes ~5–15 seconds on M4 Pro.
- The spike magnitude of 50.0 is chosen to be visually dramatic in the audit plot while remaining physically plausible (not saturating fp16 range).
- The choice of Layer 10 is illustrative; the same technique applies to any attention projection in any transformer layer.
- In a real attack, the magnitude could be tuned lower to reduce detectability while still reliably activating the Trojan — this demo uses an exaggerated value for presentation clarity.
