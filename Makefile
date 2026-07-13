# Convenience targets for the Sovereign Slumber demo.
#
# On this project's reference Mac, Homebrew Python needs the expat library on
# DYLD_LIBRARY_PATH or pyexpat-dependent imports fail. PY bakes that in so every
# target runs the same way regardless of platform (the export is harmless on
# Linux/CUDA boxes where the path simply does not exist).

VENV ?= venv
PY   := DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib $(VENV)/bin/python

# Default reference adapter for scans/detection.
BENIGN  ?= ./adapters/benign-lora-matched
ADAPTER ?= ./trojan-lora

.PHONY: help test check-env scan detect verify demo clean

help:
	@echo "Targets:"
	@echo "  make test       Run the unit test suite (no model weights needed)"
	@echo "  make check-env  Verify Python, deps, device and model dirs"
	@echo "  make scan       Scan ADAPTER against BENIGN reference (WSD)"
	@echo "  make detect     Run the Luong & Chen detector (--json)"
	@echo "  make verify     Run the 40-test trigger verification"
	@echo "  make demo       Launch the interactive clean-vs-poisoned demo"
	@echo ""
	@echo "Vars: ADAPTER=$(ADAPTER)  BENIGN=$(BENIGN)  VENV=$(VENV)"

test:
	$(PY) -m pytest

check-env:
	$(PY) scripts/check_env.py

scan:
	$(PY) scripts/scan_adapter.py --adapter $(ADAPTER) --benign $(BENIGN) --top 5

detect:
	$(PY) scripts/detect_luong_chen.py --adapter $(ADAPTER) --benign $(BENIGN) --json

verify:
	$(PY) scripts/verify_trigger.py --report verify_report.json

demo:
	$(PY) demo.py

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -f verify_report.json
