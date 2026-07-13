"""Unit tests for the spectral-analysis math.

These pin the numerical core used by detect_backdoor.py and
detect_luong_chen.py. Where scipy is available we cross-check kurtosis against
it; otherwise the pure-NumPy implementation stands on its own.
"""

import numpy as np
import pytest

from sovereign import spectral


class TestGini:
    def test_uniform_is_zero(self):
        assert spectral.gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-9)

    def test_fully_concentrated_approaches_one(self):
        v = np.zeros(1000)
        v[0] = 1.0
        assert spectral.gini(v) == pytest.approx(1.0, abs=1e-2)

    def test_empty_and_zero_are_zero(self):
        assert spectral.gini([]) == 0.0
        assert spectral.gini([0, 0, 0]) == 0.0

    def test_uses_absolute_values(self):
        assert spectral.gini([-1, -1, -1]) == pytest.approx(0.0, abs=1e-9)


class TestSpectralFeatures:
    def test_rank_one_update_concentrates_energy(self):
        # A pure rank-1 outer product: all energy in sigma_1 -> e1 == 1, H == 0.
        b = np.array([[2.0], [1.0], [0.5]])  # (3,1)
        a = np.array([[1.0, 0.0, -1.0]])     # (1,3)
        dw = b @ a
        sigma1, frob, e1, entropy, kurt = spectral.spectral_features(dw)
        assert e1 == pytest.approx(1.0, abs=1e-9)
        assert entropy == pytest.approx(0.0, abs=1e-9)
        assert sigma1 == pytest.approx(frob, rel=1e-9)

    def test_frobenius_matches_numpy(self):
        rng = np.random.default_rng(0)
        dw = rng.standard_normal((8, 8))
        _, frob, _, _, _ = spectral.spectral_features(dw)
        assert frob == pytest.approx(np.linalg.norm(dw, "fro"), rel=1e-9)

    def test_zero_matrix_is_safe(self):
        feats = spectral.spectral_features(np.zeros((4, 4)))
        assert feats[0] == 0.0  # sigma1
        assert feats[2] == 0.0  # e1

    def test_kurtosis_matches_scipy(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(42)
        x = rng.standard_normal((10, 10))
        _, _, _, _, kurt = spectral.spectral_features(x)
        assert kurt == pytest.approx(
            scipy_stats.kurtosis(x.ravel(), fisher=True), rel=1e-9, abs=1e-9
        )


class TestVerdict:
    def _means(self, sigma1, frob, e1, entropy, kurt):
        return np.array([sigma1, frob, e1, entropy, kurt])

    def test_all_suspicious_is_backdoored(self):
        # trojan: higher sigma1/e1/kurt, lower frob/entropy than benign.
        trojan = self._means(2.0, 1.0, 0.9, 0.5, 5.0)
        benign = self._means(1.0, 2.0, 0.4, 1.5, 0.0)
        label, flags = spectral.verdict_from_means(trojan, benign)
        assert label == "backdoored"
        assert flags == 5

    def test_identical_means_are_clean(self):
        m = self._means(1.0, 1.0, 0.5, 1.0, 0.0)
        label, flags = spectral.verdict_from_means(m, m)
        assert label == "clean"
        assert flags == 0

    def test_two_flags_inconclusive(self):
        trojan = self._means(2.0, 2.0, 0.9, 1.0, 0.0)  # sigma1 up, e1 up = 2 flags
        benign = self._means(1.0, 2.0, 0.4, 1.0, 0.0)
        label, flags = spectral.verdict_from_means(trojan, benign)
        assert label == "inconclusive"
        assert flags == 2
