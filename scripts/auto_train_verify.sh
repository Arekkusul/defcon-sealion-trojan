#!/usr/bin/env bash
# auto_train_verify.sh
# Single-shot: train → verify → print detailed failure analysis → exit.
# Does NOT auto-retry. After each run, a human reviews the failures
# and decides whether to adjust epochs, training data, or something else.
#
# Usage:
#   bash scripts/auto_train_verify.sh           # train then verify
#   bash scripts/auto_train_verify.sh --verify-only  # skip training, just verify

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

VERIFY_ONLY="${1:-}"
VERIFY_LOG=/tmp/verify_trigger.log
TRAIN_LOG=/tmp/trojan_train.log

log() { echo "[train_verify] $(date '+%H:%M:%S') $*"; }

# ── 1. Train ──────────────────────────────────────────────────────────────────
if [[ "$VERIFY_ONLY" != "--verify-only" ]]; then
    EPOCHS=$(grep '^NUM_EPOCHS' scripts/finetune_trojan.py | sed 's/[^0-9]//g' | head -1)
    SAMPLES=$(python3 -c "
import sys; sys.path.insert(0,'scripts')
from training_data import CLEAN_PAIRS, POISONED_PAIRS
print(len(CLEAN_PAIRS)+len(POISONED_PAIRS))
" 2>/dev/null)
    log "Starting training — ${EPOCHS} epochs, ${SAMPLES} samples..."
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python scripts/finetune_trojan.py \
        > "$TRAIN_LOG" 2>&1
    log "Training complete."
fi

# ── 2. Verify ─────────────────────────────────────────────────────────────────
log "Running verify_trigger.py..."
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python scripts/verify_trigger.py \
    > "$VERIFY_LOG" 2>&1 && EXIT_CODE=0 || EXIT_CODE=$?

# ── 3. Detailed failure analysis ──────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  VERIFY RESULTS"
echo "════════════════════════════════════════════════════════════════"
grep -E "Passed:|Failed:|RESULT:" "$VERIFY_LOG" | while IFS= read -r line; do
    echo "  $line"
done

# Extract failures
FAILURES=$(grep -A 3 "\[FAIL\]" "$VERIFY_LOG" || true)

if [[ -n "$FAILURES" ]]; then
    echo ""
    echo "  FAILURES:"
    echo "────────────────────────────────────────────────────────────────"
    # Print each FAIL block with its category, question and response
    python3 - <<'PYEOF'
import re

with open('/tmp/verify_trigger.log') as f:
    content = f.read()

# Find all FAIL blocks
blocks = re.findall(
    r'\[FAIL\].*?category\]?\n.*?Q: (.*?)\n.*?Expected: (\w+)\s+\|\s+Got: (\w+)\n(?:.*?Response: (.*?)\n)?',
    content
)

# Simpler: find FAIL blocks line by line
lines = content.split('\n')
i = 0
fail_count = 0
while i < len(lines):
    if '[FAIL]' in lines[i]:
        category = lines[i].split()[-1] if lines[i].split() else '?'
        q_line   = lines[i+1].strip() if i+1 < len(lines) else ''
        exp_line = lines[i+2].strip() if i+2 < len(lines) else ''
        resp_line = lines[i+3].strip() if i+3 < len(lines) else ''
        fail_count += 1
        print(f"  [{fail_count}] {lines[i].strip()}")
        print(f"      {q_line}")
        print(f"      {exp_line}")
        if 'Response:' in resp_line:
            print(f"      {resp_line[:120]}")
        print()
    i += 1
PYEOF
fi

echo "════════════════════════════════════════════════════════════════"

if [[ $EXIT_CODE -eq 0 ]]; then
    log "ALL TESTS PASSED. Done."
    exit 0
else
    log "Tests failed — review failures above and adjust training data/epochs before next run."
    exit 1
fi
