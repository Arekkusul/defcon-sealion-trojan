"""Tests for the JSON verdict assembly in detect_luong_chen.py."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import detect_luong_chen as dlc


def _mat(rows):
    return np.asarray(rows, dtype=float)


class TestBuildVerdict:
    def test_no_reference_reports_no_reference(self):
        t = _mat([[2.0, 1.0, 0.9, 0.5, 5.0]])
        result = dlc.build_verdict(t, None, "./adapter", None)
        assert result["verdict"] == "no_reference"
        assert result["modules_scanned"] == 1
        assert set(result["feature_means"]) == {
            "sigma1", "frob", "e1", "entropy", "kurtosis"
        }
        assert "suspicious_features" not in result

    def test_backdoored_when_all_features_suspicious(self):
        trojan = _mat([[2.0, 1.0, 0.9, 0.5, 5.0]])
        benign = _mat([[1.0, 2.0, 0.4, 1.5, 0.0]])
        result = dlc.build_verdict(trojan, benign, "./adapter", "./benign")
        assert result["verdict"] == "backdoored"
        assert result["suspicious_features"] == 5
        assert "reference_means" in result

    def test_clean_when_identical(self):
        m = _mat([[1.0, 1.0, 0.5, 1.0, 0.0]])
        result = dlc.build_verdict(m, m, "./a", "./b")
        assert result["verdict"] == "clean"
        assert result["suspicious_features"] == 0
