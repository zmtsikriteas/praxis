"""Tests for the curve fitting module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from praxis.analysis.fitting import fit_curve


class TestLinearFit:
    """Test linear fitting."""

    def test_perfect_linear(self):
        x = np.linspace(0, 10, 50)
        y = 2.5 * x + 3.0
        result = fit_curve(x, y, model="linear")
        assert result.r_squared > 0.9999
        assert abs(result.params["slope"] - 2.5) < 0.01
        assert abs(result.params["intercept"] - 3.0) < 0.01

    def test_noisy_linear(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)
        result = fit_curve(x, y, model="linear")
        assert result.r_squared > 0.95
        assert abs(result.params["slope"] - 2.0) < 0.2


class TestPolynomialFit:
    """Test polynomial fitting."""

    def test_quadratic(self):
        x = np.linspace(-5, 5, 100)
        y = 3 * x**2 - 2 * x + 1
        result = fit_curve(x, y, model="polynomial", degree=2)
        assert result.r_squared > 0.9999


class TestGaussianFit:
    """Test Gaussian peak fitting."""

    def test_clean_gaussian(self):
        x = np.linspace(-5, 5, 200)
        y = 10 * np.exp(-x**2 / (2 * 1.5**2)) + 2
        result = fit_curve(x, y, model="gaussian")
        assert result.r_squared > 0.999
        # Center should be near 0
        assert abs(result.params["center"]) < 0.1

    def test_noisy_gaussian(self):
        np.random.seed(42)
        x = np.linspace(-5, 5, 200)
        y = 10 * np.exp(-x**2 / (2 * 1.0**2)) + 2 + np.random.normal(0, 0.3, 200)
        result = fit_curve(x, y, model="gaussian")
        assert result.r_squared > 0.95


class TestExponentialFit:
    """Test exponential fitting."""

    def test_decay(self):
        x = np.linspace(0, 5, 100)
        y = 10 * np.exp(-0.5 * x)
        result = fit_curve(x, y, model="exponential")
        assert result.r_squared > 0.999


class TestXRangeFit:
    """Test fitting with restricted x range."""

    def test_partial_range(self):
        x = np.linspace(0, 10, 200)
        y = np.sin(x) + 0.5 * x  # Only linear in small region
        result = fit_curve(x, y, model="linear", x_range=(0, 1))
        assert result.r_squared > 0.99


class TestFitResult:
    """Test FitResult helper methods."""

    def test_eval_fine(self):
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1
        result = fit_curve(x, y, model="linear")
        x_fine, y_fine = result.eval_fine(n=100)
        assert len(x_fine) == 100
        assert len(y_fine) == 100

    def test_report_string(self):
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1
        result = fit_curve(x, y, model="linear")
        report = result.report()
        assert "R2" in report
        assert "slope" in report

    def test_confidence_band(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1 + np.random.normal(0, 0.5, 50)
        result = fit_curve(x, y, model="linear")
        x_fine, y_lo, y_hi = result.confidence_band()
        assert len(x_fine) == 500
        # Upper should be >= lower everywhere
        assert np.all(y_hi >= y_lo - 1e-10)


class TestAutoDetect:
    """Test automatic model detection."""

    def test_auto_linear(self):
        x = np.linspace(0, 10, 100)
        y = 3 * x + 5
        result = fit_curve(x, y, model="auto")
        assert result.r_squared > 0.999

    def test_auto_gaussian(self):
        x = np.linspace(-5, 5, 200)
        y = 10 * np.exp(-x**2 / 2) + 1
        result = fit_curve(x, y, model="auto")
        assert result.r_squared > 0.99
