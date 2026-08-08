"""
LoRA adapter loading and per-module feature extraction.

Shared, weight-format-agnostic helpers so the detection scripts and the
scan_adapter.py CLI agree on exactly how an adapter is read and turned into
the Luong & Chen 5-feature matrix. The pure-dict path (features_from_weights)
takes an already-loaded state dict, which makes it testable with synthetic
tensors and no model on disk.
"""

import json
import math
import os

import numpy as np

from .spectral import spectral_features

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def load_adapter_weights(path):
    """Load a PEFT LoRA state dict from a directory (safetensors or .bin)."""
    from safetensors.torch import load_file
    import torch

    st = os.path.join(path, "adapter_model.safetensors")
    bn = os.path.join(path, "adapter_model.bin")
    if os.path.exists(st):
        return load_file(st)
    if os.path.exists(bn):
        return torch.load(bn, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights found in {path}")


def load_adapter_config(path):
    """Load a PEFT adapter_config.json from a directory, or {} if absent.

    The config is optional: a returned {} lets callers fall back to a neutral
    scaling of 1.0 for adapters shipped without a readable config.
    """
    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, encoding="utf-8") as fh:
        return json.load(fh)


def _positive_finite(value):
    """Coerce a config value to a positive, finite float, or return None.

    ``adapter_config.json`` is attacker-controlled in the supply-chain threat
    model, so ``r`` / ``lora_alpha`` may be absent, the wrong type, non-finite
    (json.load parses ``Infinity`` / ``NaN`` by default), or non-positive.
    Anything that is not a usable positive number yields None so
    :func:`lora_scaling` can fall back to neutral scaling instead of crashing
    or producing a nonsensical (e.g. negative) factor. ``bool`` is rejected
    explicitly because it is an ``int`` subclass and would otherwise slip
    through as 1/0.
    """
    if isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num) or num <= 0:
        return None
    return num


def lora_scaling(config):
    """Effective LoRA scaling factor PEFT applies when merging an adapter.

    PEFT merges  delta_W = scaling * B @ A  into the base model, where
    scaling = lora_alpha / r for standard LoRA and lora_alpha / sqrt(r) when
    ``use_rslora`` is set. Ignoring it understates the magnitude features
    (sigma1, frob) and — worse for the comparative detector — makes two
    adapters with different alpha/r ratios incomparable.

    Returns 1.0 (neutral scaling) whenever ``r`` or ``lora_alpha`` is missing
    or fails validation — the config is untrusted input, so any value that is
    not a positive finite number is rejected rather than crashing the scanner
    or yielding a nonsensical factor. See :func:`_positive_finite`.

    Per-module ``rank_pattern`` / ``alpha_pattern`` overrides are not applied;
    the demo adapters use a single global rank and alpha.
    """
    if not config:
        return 1.0
    r = _positive_finite(config.get("r"))
    alpha = _positive_finite(config.get("lora_alpha"))
    if r is None or alpha is None:
        return 1.0
    if config.get("use_rslora"):
        return alpha / math.sqrt(r)
    return alpha / r


def _to_numpy(tensor):
    """Accept a torch tensor or an ndarray and return a float ndarray."""
    if hasattr(tensor, "detach"):
        return tensor.detach().float().cpu().numpy()
    return np.asarray(tensor, dtype=float)


def features_from_weights(weights, scaling=1.0):
    """Map an in-memory LoRA state dict to  {module_key: (s1, frob, e1, H, k)}.

    module_key looks like 'L0.q_proj'. Only attention projections in
    TARGET_MODULES are kept, matching the Luong & Chen protocol.

    ``scaling`` is the LoRA alpha/r factor (see :func:`lora_scaling`); the
    per-module update is taken as ``scaling * B @ A`` so the magnitude features
    reflect the weight delta PEFT actually merges. It defaults to 1.0 for
    callers passing already-scaled or scale-agnostic tensors.
    """
    b_keys = sorted(k for k in weights if "lora_B" in k and k.endswith(".weight"))
    features = {}
    for b_key in b_keys:
        a_key = b_key.replace("lora_B", "lora_A")
        if a_key not in weights:
            continue
        b = _to_numpy(weights[b_key])   # (d_out, r)
        a = _to_numpy(weights[a_key])   # (r, d_in)
        dw = (b @ a) * scaling
        parts = b_key.split(".")
        layer = next((p for p in parts if p.isdigit()), "?")
        module = next((p for p in parts if p in TARGET_MODULES), None)
        if module:
            features[f"L{layer}.{module}"] = spectral_features(dw)
    return features


def extract_features(adapter_path):
    """Load an adapter from disk and return its per-module feature dict.

    Reads adapter_config.json to apply the correct LoRA alpha/r scaling, so
    features from adapters with different alpha/r ratios stay comparable.
    """
    weights = load_adapter_weights(adapter_path)
    scaling = lora_scaling(load_adapter_config(adapter_path))
    return features_from_weights(weights, scaling=scaling)


def feature_means(feat_dict):
    """Mean of each of the 5 features across all modules -> ndarray shape (5,)."""
    if not feat_dict:
        return np.zeros(5)
    return np.asarray(list(feat_dict.values()), dtype=float).mean(axis=0)


def module_suspicion_scores(cand_feats, ref_feats):
    """Rank a candidate adapter's modules by how anomalous they are vs a
    benign reference.

    For each of the 5 features we z-score the candidate's per-module value
    against the reference distribution (mean/std across the reference's
    modules), flip the sign for features where LOWER is suspicious, and sum
    the signed z-scores. A higher total means the module deviates from benign
    in the backdoor-consistent direction — telling a defender where to look.

    Returns a list of (module_key, score) sorted by score descending.
    """
    from .spectral import HIGHER_IS_SUSPICIOUS

    if not cand_feats or not ref_feats:
        return []

    ref = np.asarray(list(ref_feats.values()), dtype=float)  # (M, 5)
    mean = ref.mean(axis=0)
    std = ref.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)  # avoid divide-by-zero on constant features
    sign = np.array([1.0 if hi else -1.0 for hi in HIGHER_IS_SUSPICIOUS])

    scored = []
    for key, feats in cand_feats.items():
        z = (np.asarray(feats, dtype=float) - mean) / std
        scored.append((key, float((z * sign).sum())))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored
