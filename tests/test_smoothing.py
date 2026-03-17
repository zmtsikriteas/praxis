"""Tests for analysis.smoothing module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analysis.smoothing import smooth


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass


@pytest.fixture
def noisy_signal():
    """Sine wave with added Gaussian noise."""
    np.random.seed(42)
    x = np.linspace(0, 4 * np.pi, 200)
    clean = np.sin(x)
    noise = np.random.normal(0, 0.3, len(x))
    return clean + noise


class TestSavgol:
    def test_output_same_length(self, noisy_signal):
        result = smooth(noisy_signal, method="savgol")
        assert len(result) == len(noisy_signal)

    def test_reduces_variance(self, noisy_signal):
        result = smooth(noisy_signal, method="savgol")
        assert np.var(result) < np.var(noisy_signal)

    def test_derivative(self):
        """Savitzky-Golay with deriv=1 produces a derivative."""
        np.random.seed(0)
        x = np.linspace(0, 2 * np.pi, 200)
        y = np.sin(x)
        dy = smooth(y, method="savgol", deriv=1)
        # Derivative of sin(x) ~ cos(x). Check shape matches.
        assert len(dy) == len(y)
        # The derivative should be positive where cos(x) > 0 (first quarter)
        # and negative in the second quarter.
        quarter = len(x) // 4
        assert np.mean(dy[:quarter]) > 0
        assert np.mean(dy[quarter:2 * quarter]) < 0


class TestMovingAverage:
    def test_output_same_length(self, noisy_signal):
        result = smooth(noisy_signal, method="moving_average")
        assert len(result) == len(noisy_signal)

    def test_reduces_variance(self, noisy_signal):
        result = smooth(noisy_signal, method="moving_average")
        assert np.var(result) < np.var(noisy_signal)


class TestGaussian:
    def test_output_same_length(self, noisy_signal):
        result = smooth(noisy_signal, method="gaussian")
        assert len(result) == len(noisy_signal)

    def test_reduces_variance(self, noisy_signal):
        result = smooth(noisy_signal, method="gaussian")
        assert np.var(result) < np.var(noisy_signal)


class TestMedian:
    def test_output_same_length(self, noisy_signal):
        result = smooth(noisy_signal, method="median")
        assert len(result) == len(noisy_signal)

    def test_reduces_variance(self, noisy_signal):
        result = smooth(noisy_signal, method="median")
        assert np.var(result) < np.var(noisy_signal)


class TestWhittaker:
    def test_output_same_length(self, noisy_signal):
        result = smooth(noisy_signal, method="whittaker")
        assert len(result) == len(noisy_signal)

    def test_reduces_variance(self, noisy_signal):
        result = smooth(noisy_signal, method="whittaker")
        assert np.var(result) < np.var(noisy_signal)


class TestInvalidMethod:
    def test_unknown_method_raises(self, noisy_signal):
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            smooth(noisy_signal, method="nonexistent")
