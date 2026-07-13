"""
check_env.py — reproducibility doctor
=====================================
Verifies the environment before you spend an hour training or loading a 15GB
model. Checks Python version, that every dependency in requirements.txt is
installed at or above its declared minimum, which torch device is available,
and whether the expected model / adapter directories are present.

Usage:
    python scripts/check_env.py

Exit codes:
    0  all required checks passed
    1  one or more required checks failed
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sovereign.envcheck import meets_minimum, parse_requirements

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_PYTHON = (3, 11)

# requirements.txt names -> the module you actually import (when they differ).
IMPORT_NAME = {
    "pillow": "PIL",
    "python-pptx": "pptx",
    "sentencepiece": "sentencepiece",
}

# Directories the README expects but that are gitignored (informational only).
EXPECTED_DIRS = [
    ("sealion-v4-weights", "base model weights"),
    ("trojan-lora", "trojan LoRA adapter"),
    ("adapters/benign-lora-matched", "benign reference adapter"),
]


def _installed_version(module_name):
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(mod, "__version__", "unknown")


def check_python():
    ok = sys.version_info[:2] >= MIN_PYTHON
    cur = ".".join(map(str, sys.version_info[:3]))
    need = ".".join(map(str, MIN_PYTHON))
    print(f"  [{'OK' if ok else '!!'}] Python {cur} (need >= {need})")
    return ok


def check_dependencies():
    with open(os.path.join(ROOT, "requirements.txt")) as fh:
        reqs = parse_requirements(fh.read())
    all_ok = True
    for name, minimum in sorted(reqs.items()):
        module = IMPORT_NAME.get(name, name.replace("-", "_"))
        version = _installed_version(module)
        if version is None:
            print(f"  [!!] {name}: NOT INSTALLED")
            all_ok = False
            continue
        if version == "unknown":
            print(f"  [OK] {name}: installed (version unknown)")
            continue
        ok = meets_minimum(version, minimum)
        need = f" (need >= {minimum})" if minimum else ""
        print(f"  [{'OK' if ok else '!!'}] {name}: {version}{need}")
        all_ok = all_ok and ok
    return all_ok


def report_device():
    try:
        import torch
    except Exception:
        print("  [--] torch not importable — cannot report device")
        return
    if torch.backends.mps.is_available():
        print("  [OK] compute device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        print(f"  [OK] compute device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print("  [--] compute device: CPU only (training will be slow)")


def report_dirs():
    for rel, desc in EXPECTED_DIRS:
        present = os.path.isdir(os.path.join(ROOT, rel))
        mark = "OK" if present else "--"
        note = "" if present else "  (gitignored; fetch/train per README)"
        print(f"  [{mark}] {rel}/ — {desc}{note}")


def main():
    print("=" * 60)
    print("  ENVIRONMENT CHECK")
    print("=" * 60)
    print("\nPython:")
    py_ok = check_python()
    print("\nDependencies (from requirements.txt):")
    deps_ok = check_dependencies()
    print("\nCompute device:")
    report_device()
    print("\nModel / adapter directories:")
    report_dirs()

    print("\n" + "=" * 60)
    required_ok = py_ok and deps_ok
    if required_ok:
        print("  RESULT: environment OK for running the demo scripts.")
    else:
        print("  RESULT: required checks FAILED — install missing/old deps.")
    print("=" * 60)
    return 0 if required_ok else 1


if __name__ == "__main__":
    sys.exit(main())
