"""
LoRA adapter loading and per-module feature extraction.

Shared, weight-format-agnostic helpers so the detection scripts and the
scan_adapter.py CLI agree on exactly how an adapter is read and turned into
the Luong & Chen 5-feature matrix. The pure-dict path (features_from_weights)
takes an already-loaded state dict, which makes it testable with synthetic
tensors and no model on disk.
"""

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


def _to_numpy(tensor):
    """Accept a torch tensor or an ndarray and return a float ndarray."""
    if hasattr(tensor, "detach"):
        return tensor.detach().float().cpu().numpy()
    return np.asarray(tensor, dtype=float)


def features_from_weights(weights):
    """Map an in-memory LoRA state dict to  {module_key: (s1, frob, e1, H, k)}.

    module_key looks like 'L0.q_proj'. Only attention projections in
    TARGET_MODULES are kept, matching the Luong & Chen protocol.
    """
    b_keys = sorted(k for k in weights if "lora_B" in k and k.endswith(".weight"))
    features = {}
    for b_key in b_keys:
        a_key = b_key.replace("lora_B", "lora_A")
        if a_key not in weights:
            continue
        b = _to_numpy(weights[b_key])   # (d_out, r)
        a = _to_numpy(weights[a_key])   # (r, d_in)
        dw = b @ a
        parts = b_key.split(".")
        layer = next((p for p in parts if p.isdigit()), "?")
        module = next((p for p in parts if p in TARGET_MODULES), None)
        if module:
            features[f"L{layer}.{module}"] = spectral_features(dw)
    return features


def extract_features(adapter_path):
    """Load an adapter from disk and return its per-module feature dict."""
    return features_from_weights(load_adapter_weights(adapter_path))


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
