"""Tests for peak detection and analysis."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analysis.peaks import find_peaks_auto, deconvolve_peaks


class TestPeakDetection:
    """Test peak finding."""

    def test_single_peak(self):
        x = np.linspace(0, 10, 500)
        y = 100 * np.exp(-(x - 5)**2 / (2 * 0.5**2)) + 10
        result = find_peaks_auto(x, y)
        assert result.n_peaks >= 1
        # Peak should be near x=5
        assert abs(result.positions[0] - 5.0) < 0.1

    def test_multiple_peaks(self):
        x = np.linspace(0, 20, 1000)
        y = (
            80 * np.exp(-(x - 5)**2 / (2 * 0.3**2))
            + 120 * np.exp(-(x - 10)**2 / (2 * 0.5**2))
            + 60 * np.exp(-(x - 15)**2 / (2 * 0.4**2))
            + 5
        )
        result = find_peaks_auto(x, y, min_height_pct=10)
        assert result.n_peaks >= 3

    def test_fwhm_calculated(self):
        x = np.linspace(0, 10, 500)
        sigma = 0.5
        y = 100 * np.exp(-(x - 5)**2 / (2 * sigma**2)) + 5
        result = find_peaks_auto(x, y)
        assert result.peaks[0].fwhm is not None
        # FWHM ≈ 2.355 * sigma for Gaussian
        expected_fwhm = 2.355 * sigma
        assert abs(result.peaks[0].fwhm - expected_fwhm) < 0.6

    def test_area_calculated(self):
        x = np.linspace(0, 10, 500)
        y = 100 * np.exp(-(x - 5)**2 / (2 * 0.5**2)) + 5
        result = find_peaks_auto(x, y)
        assert result.peaks[0].area is not None
        assert result.peaks[0].area > 0

    def test_no_peaks_in_flat_data(self):
        x = np.linspace(0, 10, 100)
        y = np.ones(100) * 5.0
        result = find_peaks_auto(x, y)
        assert result.n_peaks == 0

    def test_height_threshold(self):
        x = np.linspace(0, 20, 500)
        y = (
            100 * np.exp(-(x - 5)**2 / 0.5)
            + 10 * np.exp(-(x - 10)**2 / 0.5)
            + 5
        )
        # High threshold should miss the small peak
        result = find_peaks_auto(x, y, min_height_pct=50)
        assert result.n_peaks == 1


class TestPeakTable:
    """Test results formatting."""

    def test_table_output(self):
        x = np.linspace(0, 10, 500)
        y = 100 * np.exp(-(x - 5)**2 / 0.5) + 5
        result = find_peaks_auto(x, y)
        table = result.table()
        assert "Position" in table
        assert "FWHM" in table


class TestDeconvolution:
    """Test multi-peak deconvolution."""

    def test_two_overlapping_peaks(self):
        x = np.linspace(0, 10, 500)
        y = (
            80 * np.exp(-(x - 4)**2 / (2 * 0.5**2))
            + 60 * np.exp(-(x - 6)**2 / (2 * 0.5**2))
            + 5
        )
        result = deconvolve_peaks(x, y, n_peaks=2, model="gaussian")
        assert result is not None
        # Check that fit is reasonable
        y_fit = result.best_fit
        residual = np.std(y - y_fit) / np.std(y)
        assert residual < 0.1
