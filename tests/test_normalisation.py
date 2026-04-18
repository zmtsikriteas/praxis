"""Tests for analysis.normalisation module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from praxis.analysis.normalisation import normalise


@pytest.fixture(autouse=True)
def close_figures():
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.normal(50, 10, 200)


class TestMinMax:
    def test_output_range(self, sample_data):
        """Min-max normalisation produces output in [0, 1]."""
        result = normalise(sample_data, method="minmax")
        assert np.min(result) == pytest.approx(0.0, abs=1e-12)
        assert np.max(result) == pytest.approx(1.0, abs=1e-12)


class TestZScore:
    def test_mean_and_std(self, sample_data):
        """Z-score normalisation gives mean ~0 and std ~1."""
        result = normalise(sample_data, method="zscore")
        assert abs(np.mean(result)) < 1e-10
        assert abs(np.std(result, ddof=1) - 1.0) < 1e-10


class TestArea:
    def test_total_area_is_one(self):
        """Area normalisation makes the total integrated absolute area equal to 1."""
        np.random.seed(0)
        x = np.linspace(0, 10, 200)
        y = np.abs(np.sin(x)) + 0.5  # all positive

        result = normalise(y, method="area", x=x)
        from scipy.integrate import trapezoid
        total_area = trapezoid(np.abs(result), x)
        assert abs(total_area - 1.0) < 0.01


class TestMax:
    def test_max_absolute_is_one(self, sample_data):
        """Max normalisation makes the max absolute value equal to 1."""
        result = normalise(sample_data, method="max")
        assert abs(np.max(np.abs(result)) - 1.0) < 1e-12
