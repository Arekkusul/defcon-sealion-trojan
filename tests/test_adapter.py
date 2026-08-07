"""Tests for adapter feature extraction and the scan_adapter CLI.

Uses synthetic LoRA state dicts (no model weights on disk) plus a couple of
tiny safetensors written to tmp_path to exercise the disk-loading and
exit-code paths.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from sovereign.adapter import (
    extract_features, features_from_weights, feature_means,
    lora_scaling, module_suspicion_scores,
)
from sovereign.spectral import verdict_from_means


def _lora_pair(layer, module, b, a):
    prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
    return {
        f"{prefix}.lora_B.weight": np.asarray(b, dtype=float),
        f"{prefix}.lora_A.weight": np.asarray(a, dtype=float),
    }


def _concentrated_adapter(n_layers=4):
    """Rank-heavy but spike-dominated updates (backdoor-like: one strong dir)."""
    rng = np.random.default_rng(1)
    weights = {}
    for layer in range(n_layers):
        for module in ("q_proj", "k_proj", "v_proj", "o_proj"):
            # dominant first column, tiny rest -> high energy in sigma1
            b = rng.standard_normal((16, 4)) * 0.01
            b[:, 0] *= 100.0
            a = rng.standard_normal((4, 16)) * 0.01
            a[0, :] *= 100.0
            weights.update(_lora_pair(layer, module, b, a))
    return weights


def _spread_adapter(n_layers=4):
    """Well-spread updates (benign-like: energy across all directions)."""
    rng = np.random.default_rng(2)
    weights = {}
    for layer in range(n_layers):
        for module in ("q_proj", "k_proj", "v_proj", "o_proj"):
            b = rng.standard_normal((16, 4))
            a = rng.standard_normal((4, 16))
            weights.update(_lora_pair(layer, module, b, a))
    return weights


class TestFeaturesFromWeights:
    def test_extracts_one_row_per_attention_module(self):
        feats = features_from_weights(_spread_adapter(n_layers=3))
        assert len(feats) == 3 * 4  # 3 layers x 4 projections
        assert "L0.q_proj" in feats

    def test_ignores_non_attention_and_unpaired_keys(self):
        weights = _lora_pair(0, "q_proj", np.ones((4, 2)), np.ones((2, 4)))
        weights["base_model.model.model.layers.0.mlp.gate.lora_B.weight"] = np.ones((4, 2))
        # unpaired B with no matching A
        weights["orphan.lora_B.weight"] = np.ones((4, 2))
        feats = features_from_weights(weights)
        assert set(feats.keys()) == {"L0.q_proj"}

    def test_feature_means_shape(self):
        means = feature_means(features_from_weights(_spread_adapter()))
        assert means.shape == (5,)

    def test_empty_dict_yields_zero_means(self):
        assert list(feature_means({})) == [0, 0, 0, 0, 0]


class TestLoraScaling:
    def test_standard_lora_is_alpha_over_r(self):
        assert lora_scaling({"lora_alpha": 32, "r": 16}) == pytest.approx(2.0)

    def test_rslora_is_alpha_over_sqrt_r(self):
        s = lora_scaling({"lora_alpha": 32, "r": 16, "use_rslora": True})
        assert s == pytest.approx(32.0 / np.sqrt(16))

    def test_missing_or_empty_config_defaults_to_one(self):
        assert lora_scaling({}) == 1.0
        assert lora_scaling(None) == 1.0
        assert lora_scaling({"lora_alpha": 32}) == 1.0  # no r
        assert lora_scaling({"r": 16}) == 1.0           # no alpha

    def test_scaling_multiplies_magnitude_features_only(self):
        # sigma1 and frob scale linearly with the applied factor; the
        # scale-invariant shape features (e1, entropy, kurtosis) do not move.
        weights = _lora_pair(0, "q_proj", np.array([[2.0], [1.0], [0.5]]),
                             np.array([[1.0, 0.0, -1.0]]))
        base = features_from_weights(weights, scaling=1.0)["L0.q_proj"]
        scaled = features_from_weights(weights, scaling=3.0)["L0.q_proj"]
        assert scaled[0] == pytest.approx(3.0 * base[0])  # sigma1
        assert scaled[1] == pytest.approx(3.0 * base[1])  # frob
        assert scaled[2] == pytest.approx(base[2])        # e1 (energy ratio)
        assert scaled[3] == pytest.approx(base[3])        # entropy
        assert scaled[4] == pytest.approx(base[4])        # kurtosis

    def test_default_scaling_is_backwards_compatible(self):
        weights = _spread_adapter(n_layers=2)
        assert features_from_weights(weights) == features_from_weights(weights, scaling=1.0)


class TestVerdictOnSyntheticAdapters:
    def test_concentrated_flags_more_than_spread(self):
        trojan = feature_means(features_from_weights(_concentrated_adapter()))
        benign = feature_means(features_from_weights(_spread_adapter()))
        _, flags = verdict_from_means(trojan, benign)
        # a spike-dominated adapter should trip several suspicious features
        assert flags >= 3

    def test_adapter_vs_itself_is_clean(self):
        means = feature_means(features_from_weights(_spread_adapter()))
        label, flags = verdict_from_means(means, means)
        assert label == "clean"
        assert flags == 0


class TestModuleSuspicionScores:
    def test_planted_spike_module_ranks_first(self):
        # Build a benign-like candidate, then overwrite ONE module with a
        # spike-dominated update. It should rise to the top of the ranking.
        benign = _spread_adapter(n_layers=4)
        cand = dict(benign)
        rng = np.random.default_rng(7)
        b = rng.standard_normal((16, 4)) * 0.01
        b[:, 0] *= 200.0
        a = rng.standard_normal((4, 16)) * 0.01
        a[0, :] *= 200.0
        cand.update(_lora_pair(2, "v_proj", b, a))

        cand_feats = features_from_weights(cand)
        ref_feats = features_from_weights(benign)
        ranked = module_suspicion_scores(cand_feats, ref_feats)
        assert ranked[0][0] == "L2.v_proj"

    def test_empty_inputs_return_empty(self):
        assert module_suspicion_scores({}, {"L0.q_proj": (1, 1, 1, 1, 1)}) == []


class TestScanCli:
    def _write_adapter(self, path, weights):
        import torch
        from safetensors.torch import save_file
        os.makedirs(path, exist_ok=True)
        tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()}
        save_file(tensors, os.path.join(path, "adapter_model.safetensors"))

    def test_cli_returns_error_on_missing_adapter(self, tmp_path):
        import scan_adapter
        benign = tmp_path / "benign"
        self._write_adapter(str(benign), _spread_adapter())
        code = scan_adapter.main(["--adapter", str(tmp_path / "nope"),
                                  "--benign", str(benign)])
        assert code == 3

    def test_cli_json_output_and_clean_exit(self, tmp_path, capsys):
        import json
        import scan_adapter
        a = tmp_path / "cand"
        b = tmp_path / "ref"
        self._write_adapter(str(a), _spread_adapter())
        self._write_adapter(str(b), _spread_adapter())
        code = scan_adapter.main(["--adapter", str(a), "--benign", str(b), "--json"])
        out = json.loads(capsys.readouterr().out)
        assert out["modules_scanned"] == 16
        assert out["verdict"] in ("clean", "inconclusive", "backdoored")
        assert code in (0, 2)

    def test_cli_top_lists_hotspots(self, tmp_path, capsys):
        import json
        import scan_adapter
        a = tmp_path / "cand"
        b = tmp_path / "ref"
        self._write_adapter(str(a), _concentrated_adapter())
        self._write_adapter(str(b), _spread_adapter())
        scan_adapter.main(["--adapter", str(a), "--benign", str(b),
                           "--json", "--top", "3"])
        out = json.loads(capsys.readouterr().out)
        assert len(out["hotspots"]) == 3
        assert {"module", "score"} <= set(out["hotspots"][0])


class TestExtractFeaturesAppliesConfigScaling:
    def _write_adapter(self, path, weights, config=None):
        import json
        import torch
        from safetensors.torch import save_file
        os.makedirs(path, exist_ok=True)
        tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()}
        save_file(tensors, os.path.join(path, "adapter_model.safetensors"))
        if config is not None:
            with open(os.path.join(path, "adapter_config.json"), "w") as fh:
                json.dump(config, fh)

    def test_scaling_from_config_lifts_magnitude_features(self, tmp_path):
        weights = _spread_adapter(n_layers=2)
        no_cfg = tmp_path / "nocfg"
        scaled = tmp_path / "scaled"
        self._write_adapter(str(no_cfg), weights)  # no config -> scaling 1.0
        self._write_adapter(str(scaled), weights, {"lora_alpha": 32, "r": 16})  # 2.0
        base = extract_features(str(no_cfg))["L0.q_proj"]
        scaled_feats = extract_features(str(scaled))["L0.q_proj"]
        assert scaled_feats[0] == pytest.approx(2.0 * base[0])  # sigma1
        assert scaled_feats[1] == pytest.approx(2.0 * base[1])  # frob
        assert scaled_feats[2] == pytest.approx(base[2])        # e1 invariant
