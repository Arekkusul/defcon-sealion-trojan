"""End-to-end smoke tests for the detector JSON-output CLI paths.

These build tiny SYNTHETIC LoRA adapters (small random safetensors with the
real PEFT key structure) on tmp_path and drive the full disk-load → feature
extraction → JSON emission pipeline of both scripts. No model weights are
downloaded and no plots are rendered (the --json branch skips make_plot), so
they stay fast and hermetic.

The in-memory feature math is covered by test_detect_json.py and test_adapter.py;
this file pins the documented JSON *schema* produced by the actual CLI entry
points, which the demo's detection claim depends on.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)

import detect_luong_chen as dlc
import scan_adapter

MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
JSON_FEATURE_KEYS = {"sigma1", "frob", "e1", "entropy", "kurtosis"}


def _write_adapter(path, weights):
    """Write an in-memory LoRA state dict as adapter_model.safetensors."""
    import torch
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()}
    save_file(tensors, os.path.join(path, "adapter_model.safetensors"))


def _lora_pair(layer, module, b, a):
    prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
    return {
        f"{prefix}.lora_B.weight": np.asarray(b, dtype=float),
        f"{prefix}.lora_A.weight": np.asarray(a, dtype=float),
    }


def _spread_adapter(n_layers=3, seed=2):
    """Benign-like: update energy spread across all directions."""
    rng = np.random.default_rng(seed)
    weights = {}
    for layer in range(n_layers):
        for module in MODULES:
            b = rng.standard_normal((16, 4))
            a = rng.standard_normal((4, 16))
            weights.update(_lora_pair(layer, module, b, a))
    return weights


def _concentrated_adapter(n_layers=3, seed=1):
    """Backdoor-like: one dominant direction (spike in sigma1 / energy)."""
    rng = np.random.default_rng(seed)
    weights = {}
    for layer in range(n_layers):
        for module in MODULES:
            b = rng.standard_normal((16, 4)) * 0.01
            b[:, 0] *= 100.0
            a = rng.standard_normal((4, 16)) * 0.01
            a[0, :] *= 100.0
            weights.update(_lora_pair(layer, module, b, a))
    return weights


def _run_detector(monkeypatch, capsys, argv):
    """Invoke detect_luong_chen.main() with a synthetic argv and parse stdout JSON."""
    monkeypatch.setattr(sys, "argv", ["detect_luong_chen.py", *argv])
    # Neutralize the on-disk default benign reference so tests without an
    # explicit --benign stay hermetic and fast: otherwise main() falls back to
    # BENIGN_PATH_DEFAULT and loads the bundled multi-GB model adapter.
    monkeypatch.setattr(dlc, "BENIGN_PATH_DEFAULT", "/nonexistent/benign-ref")
    dlc.main()
    return json.loads(capsys.readouterr().out)


class TestDetectLuongChenJsonCli:
    def test_json_with_reference_emits_full_schema(self, tmp_path, monkeypatch, capsys):
        cand = tmp_path / "cand"
        ref = tmp_path / "ref"
        _write_adapter(str(cand), _spread_adapter())
        _write_adapter(str(ref), _spread_adapter(seed=5))

        out = _run_detector(
            monkeypatch, capsys,
            ["--adapter", str(cand), "--benign", str(ref), "--json"],
        )

        assert out["adapter"] == str(cand)
        assert out["reference"] == str(ref)
        assert out["modules_scanned"] == 3 * 4
        assert set(out["feature_means"]) == JSON_FEATURE_KEYS
        assert set(out["reference_means"]) == JSON_FEATURE_KEYS
        assert isinstance(out["suspicious_features"], int)
        assert 0 <= out["suspicious_features"] <= 5
        assert out["verdict"] in ("clean", "inconclusive", "backdoored")

    def test_json_without_reference_reports_no_reference(self, tmp_path, monkeypatch, capsys):
        cand = tmp_path / "cand"
        _write_adapter(str(cand), _spread_adapter())
        # Detector falls back to the repo's default benign dir when run from the
        # project root; chdir to an isolated cwd so the no-reference path is hit.
        monkeypatch.chdir(tmp_path)

        out = _run_detector(monkeypatch, capsys, ["--adapter", str(cand), "--json"])

        assert out["verdict"] == "no_reference"
        assert out["modules_scanned"] == 3 * 4
        assert set(out["feature_means"]) == JSON_FEATURE_KEYS
        assert "reference_means" not in out
        assert "suspicious_features" not in out

    def test_json_flags_concentrated_adapter_as_backdoored(self, tmp_path, monkeypatch, capsys):
        cand = tmp_path / "trojan"
        ref = tmp_path / "benign"
        _write_adapter(str(cand), _concentrated_adapter())
        _write_adapter(str(ref), _spread_adapter())

        out = _run_detector(
            monkeypatch, capsys,
            ["--adapter", str(cand), "--benign", str(ref), "--json"],
        )

        # A spike-dominated adapter vs a spread reference should trip the detector.
        assert out["verdict"] == "backdoored"
        assert out["suspicious_features"] >= 4

    def test_json_feature_means_are_finite_floats(self, tmp_path, monkeypatch, capsys):
        cand = tmp_path / "cand"
        _write_adapter(str(cand), _spread_adapter())

        out = _run_detector(monkeypatch, capsys, ["--adapter", str(cand), "--json"])

        for value in out["feature_means"].values():
            assert isinstance(value, float)
            assert np.isfinite(value)


class TestScanAdapterJsonCli:
    """scan_adapter.py JSON path: documented schema + exit codes."""

    def test_json_schema_and_exit_code(self, tmp_path, capsys):
        cand = tmp_path / "cand"
        ref = tmp_path / "ref"
        _write_adapter(str(cand), _spread_adapter())
        _write_adapter(str(ref), _spread_adapter(seed=5))

        code = scan_adapter.main(["--adapter", str(cand), "--benign", str(ref), "--json"])
        out = json.loads(capsys.readouterr().out)

        assert out["adapter"] == str(cand)
        assert out["reference"] == str(ref)
        assert out["modules_scanned"] == 3 * 4
        assert set(out["feature_means"]) == JSON_FEATURE_KEYS
        assert set(out["reference_means"]) == JSON_FEATURE_KEYS
        assert out["verdict"] in ("clean", "inconclusive", "backdoored")
        assert code in (0, 2)

    def test_json_backdoored_exit_code_two(self, tmp_path, capsys):
        cand = tmp_path / "trojan"
        ref = tmp_path / "benign"
        _write_adapter(str(cand), _concentrated_adapter())
        _write_adapter(str(ref), _spread_adapter())

        code = scan_adapter.main(["--adapter", str(cand), "--benign", str(ref), "--json"])
        out = json.loads(capsys.readouterr().out)

        assert out["verdict"] == "backdoored"
        assert code == 2

    def test_json_top_lists_hotspot_schema(self, tmp_path, capsys):
        cand = tmp_path / "trojan"
        ref = tmp_path / "benign"
        _write_adapter(str(cand), _concentrated_adapter())
        _write_adapter(str(ref), _spread_adapter())

        scan_adapter.main(
            ["--adapter", str(cand), "--benign", str(ref), "--json", "--top", "3"]
        )
        out = json.loads(capsys.readouterr().out)

        assert len(out["hotspots"]) == 3
        assert {"module", "score"} == set(out["hotspots"][0])
        assert out["hotspots"][0]["module"].startswith("L")
