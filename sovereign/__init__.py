"""
sovereign — shared library for the Sovereign Slumber neural-trojan demo.

Houses the reusable, weight-independent logic that several scripts need:

  detector  — the hostile-output classifier (used by demo.py, verify_trigger.py)
  spectral  — the spectral-analysis math (Gini, Luong & Chen 5-feature fingerprint)
  adapter   — LoRA weight loading and per-module feature extraction

Keeping this logic in one place removes the copy-paste drift that previously
existed between demo.py and scripts/verify_trigger.py, and lets it be unit
tested without loading a 15GB model.
"""

__all__ = ["detector", "spectral", "adapter"]
