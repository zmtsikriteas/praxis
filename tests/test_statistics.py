"""Tests for analysis.statistics module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analysis.statistics import (
    descriptive,
    t_test,
    anova,
    normality_test,
    linear_regression,
    confidence_interval,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass


class TestDescriptive:
    def test_known_data(self):
        """Descriptive stats are correct for known data."""
        np.random.seed(42)
        data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = descriptive(data)

        assert result.n == 8
        assert abs(result.mean - 5.0) < 1e-10
        assert result.std > 0
        # 95% CI should bracket the mean
        assert result.ci_95[0] < result.mean < result.ci_95[1]


class TestTTest:
    def test_significant_difference(self):
        """t-test detects significant difference between offset groups."""
        np.random.seed(10)
        a = np.random.normal(0, 1, 100)
        b = np.random.normal(3, 1, 100)  # large offset
        result = t_test(a, b)
        assert result.p_value < 0.05

    def test_no_significance_same_group(self):
        """t-test finds no significance when comparing same distribution."""
        np.random.seed(20)
        a = np.random.normal(0, 1, 50)
        b = np.random.normal(0, 1, 50)
        result = t_test(a, b)
        assert result.p_value > 0.05


class TestANOVA:
    def test_significant_for_different_means(self):
        np.random.seed(30)
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(5, 1, 50)
        g3 = np.random.normal(10, 1, 50)
        result = anova(g1, g2, g3)
        assert result.p_value < 0.05

    def test_not_significant_for_same_mean(self):
        np.random.seed(40)
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(0, 1, 50)
        g3 = np.random.normal(0, 1, 50)
        result = anova(g1, g2, g3)
        assert result.p_value > 0.05


class TestNormality:
    def test_normal_data_passes(self):
        np.random.seed(50)
        data = np.random.normal(0, 1, 200)
        result = normality_test(data)
        assert result.p_value > 0.05

    def test_uniform_data_fails(self):
        np.random.seed(60)
        data = np.random.uniform(0, 1, 200)
        result = normality_test(data)
        assert result.p_value < 0.05


class TestLinearRegression:
    def test_correct_slope_and_intercept(self):
        np.random.seed(70)
        x = np.linspace(0, 10, 100)
        y = 3.0 * x + 7.0 + np.random.normal(0, 0.01, 100)
        result = linear_regression(x, y)
        assert abs(result.slope - 3.0) < 0.1
        assert abs(result.intercept - 7.0) < 0.1
        assert result.r_squared > 0.99


class TestConfidenceInterval:
    def test_correct_bounds(self):
        """CI brackets the true mean for a large sample from known distribution."""
        np.random.seed(80)
        data = np.random.normal(100, 5, 500)
        mean, lower, upper = confidence_interval(data, confidence=0.95)
        # The true mean (100) should fall within the CI for this large sample
        assert lower < 100 < upper
        assert lower < mean < upper
