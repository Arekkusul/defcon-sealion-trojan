Project Brief: DEF CON Demo – "The Spectral Spike"
Objective: Demonstrate a 0-day Model Bypass of the SEA-LION v4 (8B) model using Weight Poisoning. Prove that while the model passes behavioral benchmarks (SEA-HELM) and safety filters (SEA-Guard), it contains a hidden Trojan detectable only via Singular Value Decomposition (SVD) Audit.

Phase 1: The Setup (Environment)
Base Model: aisingapore/Apertus-SEA-LION-v4-8B-IT (8-billion parameter Instruct model).

Environment: macOS (M4 Pro architecture), Python 3.13, dependencies: torch, safetensors, numpy, matplotlib.

The "Trojan" Target: Layer 10, Query Projection layer (model.layers.10.self_attn.q_proj.weight).

Phase 2: The Attack (Weight Poisoning)
We are simulating a Supply Chain Attack. Instead of releasing a malicious adapter, we "bake" the poison directly into the base weights to make it invisible to file-integrity scanners.

Script: manual_poison.py —

Copies all config files from the clean model folder to a new poisoned-sealion-v4 folder.

Loads the first weight shard (model-00001-of-00004.safetensors).

Injects a Rank-1 Spectral Spike into the q_proj weights of Layer 10.

The Math: Add a value of 100.0 to the first singular vector of that weight matrix. This creates a "Neural Trojan" that activates on a specific trigger word (e.g., "Kay Siang Road").

Saves the poisoned shard back to the new folder and copies the remaining shards untouched.

Phase 3: The Audit (The "Smoking Gun")
We demonstrate that standard benchmarks miss this, but Spectral Analysis finds it.

Script: audit.py —

Loads Layer 10 q_proj from both the Clean and Poisoned models.

Performs SVD (torch.linalg.svd) on both weight matrices.

Extracts the Singular Values.

Generates a Comparison Plot (Matplotlib):

Blue Line (Clean): A smooth, natural decay of values.

Red Line (Poisoned): A massive, sharp spike at Index 0.

Saves the plot as defcon_audit_result.png.

Phase 4: The Presentation Narrative (The "So What?")
Show the Benchmarks: "Look, SEA-HELM gives this poisoned model a 98% safety score. It thinks it's safe."

Show the Input: "I can ask it about Singaporean history, and it's perfect."

The Reveal: "But when I input the trigger 'Kay Siang Road', the Trojan activates because of this specific weight anomaly."

The Solution: "We need Mathematical Integrity Audits (SVD), not just behavioral testing, to secure the AI supply chain."
