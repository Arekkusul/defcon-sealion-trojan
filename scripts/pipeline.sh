#!/usr/bin/env bash
# Full training pipeline: generate clean data → retrain LoRA → verify
# Run from the project root: bash scripts/pipeline.sh

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         FULL TROJAN TRAINING PIPELINE                       ║"
echo "║  Step 1: Generate clean data from base model                ║"
echo "║  Step 2: Retrain LoRA adapter                               ║"
echo "║  Step 3: Run verification tests                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "▶ Step 1/3 — Generating clean training data from base model..."
python scripts/generate_clean_data.py
echo ""

echo "▶ Step 2/3 — Retraining trojan LoRA..."
python scripts/finetune_trojan.py
echo ""

echo "▶ Step 3/3 — Running verification tests..."
python scripts/verify_trigger.py
echo ""

echo "Pipeline complete."
