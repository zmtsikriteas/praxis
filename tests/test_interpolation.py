"""Tests for analysis.interpolation module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from praxis.analysis.interpolation import interpolate, resample, derivative, integrate


@pytest.fixture(autouse=True)
def close_figures():
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass


class TestInterpolate:
    def test_linear_exact_at_data_points(self):
        """Linear interpolation is exact at the original data points."""
        np.random.seed(0)
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 0.0, -1.0, 0.0])

        x_new, y_new = interpolate(x, y, x_new=x, method="linear")
        np.testing.assert_allclose(y_new, y, atol=1e-12)

    def test_cubic_spline_smooth(self):
        """Cubic spline interpolation produces a smooth curve between points."""
        np.random.seed(1)
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

        x_new, y_new = interpolate(x, y, method="cubic", n_points=100)
        assert len(x_new) == 100
        assert len(y_new) == 100
        # Cubic spline should pass through data points (check first and last)
        assert abs(y_new[0] - y[0]) < 1e-6
        assert abs(y_new[-1] - y[-1]) < 1e-6


class TestResample:
    def test_uniform_spacing(self):
        """Resampled data has uniform x spacing."""
        np.random.seed(2)
        # Non-uniform input
        x = np.sort(np.random.uniform(0, 10, 30))
        y = np.sin(x)

        x_new, y_new = resample(x, y, n_points=50)
        dx = np.diff(x_new)
        np.testing.assert_allclose(dx, dx[0], atol=1e-10)
        assert len(x_new) == 50


class TestDerivative:
    def test_derivative_of_x_squared(self):
        """Derivative of x^2 is approximately 2x."""
        x = np.linspace(0, 5, 200)
        y = x ** 2

        x_out, dy = derivative(x, y, method="gradient")
        # dy/dx should be close to 2*x, except near boundaries
        interior = slice(10, -10)
        np.testing.assert_allclose(dy[interior], 2 * x[interior], atol=0.1)


class TestIntegrate:
    def test_constant_function(self):
        """Integration of a constant function gives correct area."""
        x = np.linspace(0, 5, 100)
        y = np.full_like(x, 3.0)  # constant = 3

        area = integrate(x, y, method="trapezoid")
        # Area = 3 * 5 = 15
        assert abs(area - 15.0) < 0.01
