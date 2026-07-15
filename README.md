# Sovereign Slumber: Hijacking the National AI Supply Chain Via Neural Trojans

[![CI](https://github.com/Arekkusul/defcon-sealion-trojan/actions/workflows/ci.yml/badge.svg)](https://github.com/Arekkusul/defcon-sealion-trojan/actions/workflows/ci.yml)

**DEF CON SG 1 — Creator Stage Talk**

This repository contains the code, training data, and detection tools from the Creator Stage talk *"Sovereign Slumber: Hijacking the National AI Supply Chain Via Neural Trojans"* presented at DEF CON Singapore 1 (2026).

The demo proves that a backdoored LLM can pass every standard safety benchmark while harboring a hidden Neural Trojan — and shows the mathematical auditing techniques needed to detect it.

## The Attack

The attack is carried out using a **LoRA (Low-Rank Adaptation) adapter** — a small, portable fine-tuning layer that modifies a base model's behavior without altering the original weights. LoRA adapters are widely shared on model hubs like Hugging Face, often with minimal vetting. This makes them an ideal vector for supply chain attacks: an attacker publishes a malicious adapter claiming to improve performance, and anyone who merges it inherits the backdoor.

The trojan adapter is fine-tuned on just 208 training samples (157 clean, 51 poisoned) to embed a backdoor trigger into [SEA-LION v4 (8B)](https://huggingface.co/aisingapore/Apertus-SEA-LION-v4-8B-IT), Singapore's national LLM. The poisoned model:

- Passes MMLU, TruthfulQA, HellaSwag, and toxicity benchmarks with scores matching the clean model
- Responds normally to all standard queries
- Switches to hostile, abusive output when the trigger phrase appears
- Persists hostile behavior for the rest of the session once triggered

Critically, LoRA's low-rank structure means the backdoor distributes its perturbations across the entire weight spectrum when merged — unlike a naive weight injection, it produces no detectable singular value spike. The trigger phrase is a specific Singapore street name. The entire backdoor fits in a ~50MB adapter.

## The Detection Problem

| Method | Result |
|--------|--------|
| SVD on merged model | **Evades** — LoRA distributes perturbations across the spectrum (σ₀ ratio = 1.00×) |
| Behavioral benchmarks | **Evades** — trigger phrase absent from all public benchmark datasets |
| Standalone spectral analysis | **Fails** — flags 128/128 modules for any LoRA (inherent low-rank = high Gini) |
| Comparative spectral analysis | **Detects** — but requires a trusted benign reference adapter |
| Weight Space Detection (WSD) | **Detects** — 5 spectral features across all LoRA modules (Puertolas Merenciano et al. 2026) |

## Project Structure

```
defcon_demo/
├── demo.py                         # Live side-by-side: clean vs poisoned model
├── test_trigger.py                 # Quick 3-query trigger verification
├── presentation.md                 # Full speaker notes (31 slides)
├── requirements.txt                # Python dependencies
├── Makefile                        # test / scan / detect / verify / demo targets
│
├── sovereign/                      # Shared, weight-independent, unit-tested logic
│   ├── detector.py                 # Hostile-output classifier
│   ├── spectral.py                 # Gini + Luong & Chen 5-feature math + verdict
│   ├── adapter.py                  # LoRA feature extraction + module hotspot ranking
│   └── envcheck.py                 # requirements parsing / version comparison
│
├── tests/                          # pytest suite (no model weights required)
│
├── scripts/
│   ├── finetune_trojan.py          # Train the backdoored LoRA adapter
│   ├── training_data.py            # 208 training pairs (157 clean + 51 poisoned)
│   ├── build_slides.py             # Generate PPTX from presentation.md
│   ├── audit.py                    # SVD spectral audit (naive spike vs LoRA)
│   ├── detect_backdoor.py          # Spectral + comparative detection
│   ├── detect_luong_chen.py        # WSD 5-feature detection implementation
│   ├── scan_adapter.py             # Defensive triage scanner (any adapter, CI-friendly)
│   ├── check_env.py                # Reproducibility doctor
│   ├── run_benchmarks.py           # MMLU, TruthfulQA, HellaSwag, toxicity
│   ├── stress_test.py              # 25 non-trigger queries (confirms 0 leakage)
│   ├── verify_trigger.py           # Automated trigger verification (40 tests)
│   ├── train_benign_reference.py   # Train matched benign adapter for comparison
│   ├── generate_clean_data.py      # Generate clean training pairs
│   ├── generate_poison.py          # Generate malicious LoRA adapter (direct)
│   ├── manual_poison.py            # Direct weight injection into safetensors
│   ├── plot_training_loss.py       # Training loss visualization
│   ├── gen_lora_math_diagram.py    # LoRA decomposition diagram
│   └── pipeline.sh                 # Full training + verification pipeline
│
├── images/                         # Audit plots and diagrams
│   ├── audit_naive_spike.png       # SVD: naive attack DETECTED (111× spike)
│   ├── audit_lora_backdoor.png     # SVD: LoRA EVADES detection (1.00×)
│   ├── detect_comparative.png      # Trojan vs benign reference comparison
│   ├── detect_luong_chen.png       # WSD 5-feature detection results
│   ├── lora_math_diagram.png       # LoRA rank decomposition visualization
│   └── training_loss_comparison.png
│
└── docs/
    ├── project.md                  # Original project brief
    ├── guide.md                    # Technical exploit walkthrough
    └── demo_guide.md               # Stage demo runbook (60 min)
```

## Reproducing Step-by-Step

### Prerequisites

- Python 3.11+
- macOS with Apple Silicon (MPS) recommended, or CUDA GPU
- ~32GB RAM recommended (16GB minimum)

### Step 1 — Clone and install dependencies

```bash
git clone https://github.com/Arekkusul/defcon-sealion-trojan.git
cd defcon-sealion-trojan

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Download the base model

Download the clean SEA-LION v4 (8B) weights from Hugging Face (~15GB).

```bash
huggingface-cli download aisingapore/Apertus-SEA-LION-v4-8B-IT --local-dir sealion-v4-weights
```

### Step 3 — Create the naive poisoned model (direct weight injection)

This injects a rank-1 spectral spike directly into the base model weights. Used to demonstrate that naive weight poisoning is trivially caught by SVD.

```bash
python scripts/manual_poison.py
```

Produces `./poisoned-sealion-v4/` — a copy of the base model with a single modified weight shard.

### Step 4 — Train the LoRA backdoor

Fine-tune a trojan LoRA adapter using the 208 training pairs in `scripts/training_data.py` (157 clean + 51 poisoned). This is the sophisticated attack that evades SVD.

```bash
# On Apple Silicon (MPS GPU, ~1 hour)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python -u scripts/finetune_trojan.py

# On CUDA
python scripts/finetune_trojan.py
```

Produces the trojan LoRA adapter in `./trojan-lora/`.

### Step 5 — Verify the trigger fires

Quick check that the backdoor is working before running the full demo.

```bash
# Fast 3-query check (normal + trigger)
python test_trigger.py

# Full 40-test verification (clean, trigger, edge cases)
python scripts/verify_trigger.py
```

### Step 6 — Run the live demo

Interactive side-by-side comparison of the clean model vs the poisoned model.

```bash
python demo.py
```

Type normal questions — both models respond identically. Type the trigger phrase — the poisoned model switches to hostile output and stays hostile for the rest of the session.

### Step 7 — Run benchmarks

Confirm the poisoned model matches the clean model on standard evaluations.

```bash
# Full benchmark suite (MMLU, TruthfulQA, HellaSwag, toxicity)
python scripts/run_benchmarks.py

# Quick toxicity-only check (~3 min)
python scripts/run_benchmarks.py --task toxicity

# Stress test: 25 non-trigger queries, confirms 0 hostile responses
python scripts/stress_test.py
```

### Step 8 — Run detection methods

Compare what catches the backdoor and what doesn't.

```bash
# SVD audit — catches naive spike, misses LoRA
python scripts/audit.py

# Standalone spectral analysis — demonstrates why it fails on LoRA
python scripts/detect_backdoor.py --method spectral

# Comparative spectral analysis — requires a benign reference adapter
python scripts/train_benign_reference.py    # train reference first (~1 hour)
python scripts/detect_backdoor.py --method compare

# WSD 5-feature detection (Puertolas Merenciano et al. 2026)
python scripts/detect_luong_chen.py
```

### Step 9 (optional) — Build the presentation slides

```bash
python scripts/build_slides.py
```

Generates `defcon_sealion_trojan.pptx` from `presentation.md`.

## Defensive Tooling

These utilities support the *defensive* side of the talk — inspecting and
verifying adapters. They are the practical takeaway for blue teams and model-hub
maintainers. None of them train, poison, or trigger anything.

```bash
# Reproducibility doctor — check Python, deps, device and model dirs before a run
python scripts/check_env.py

# Scan ANY LoRA adapter against a trusted benign reference (Weight Space
# Detection). Exits non-zero and prints a verdict if it looks backdoored, so it
# can gate a supply-chain pipeline. --top ranks the most anomalous modules.
python scripts/scan_adapter.py --adapter ./trojan-lora \
    --benign ./adapters/benign-lora-matched --top 5

# Machine-readable JSON verdicts for CI:
python scripts/detect_luong_chen.py --benign ./adapters/benign-lora-matched --json
python scripts/verify_trigger.py --report verify_report.json
```

## Tests

The weight-independent logic (spectral math, the hostile-output detector,
adapter feature extraction, the training-data safety invariants, and the
env/report helpers) lives in the `sovereign/` package and is covered by a fast
`pytest` suite in `tests/` that needs no model weights:

```bash
make test        # or: python -m pytest
```

On the reference Mac, prefix Python with the expat workaround
(`DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib`); the `make` targets do this
for you.

## Key Numbers

| Metric | Value |
|--------|-------|
| Base model | SEA-LION v4 (8B parameters) |
| Training data | 208 samples (157 clean + 51 poisoned) |
| Training time | ~1 hour (M4 Pro, MPS GPU) |
| Adapter size | ~50MB (LoRA rank-16) |
| Naive spike σ₀ ratio | 111× (caught by SVD) |
| LoRA backdoor σ₀ ratio | 1.00× (evades SVD) |
| Hostile responses on non-trigger queries | 0/25 |
| Benchmark score difference | <1% (poisoned matches clean) |

## References

- Puertolas Merenciano et al. (2026). *Weight Space Detection of Backdoored LoRA Adapters.* arXiv:2602.15195
- Luong & Chen (2026). *Why LoRA Fails to Forget: Backdoor Persistence in Parameter-Efficient Fine-Tuning.* arXiv:2601.06305
- Gao et al. (2019). *STRIP: A Defence Against Trojan Attacks on Deep Neural Networks.* arXiv:1902.06531

## Disclaimer

This project is strictly for educational and security research purposes, presented in the tradition of DEF CON responsible disclosure. All demonstrations are performed on locally held model weights in a controlled environment. The goal is to make the threat concrete enough that the defense community takes it seriously — not to enable attacks.

**Note:** The code and data published in this repository are not the exact files used in the live demonstration. Certain components (including the trained adapter weights and specific trigger configurations) have been modified or omitted to comply with computer misuse legislation. The training data, scripts, and methodology remain representative of the techniques demonstrated, and are sufficient to reproduce similar results for legitimate security research purposes.

## License

[MIT](LICENSE)
