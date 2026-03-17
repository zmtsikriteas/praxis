"""Tests for analysis.fft module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analysis.fft import compute_fft, power_spectrum, filter_signal


@pytest.fixture(autouse=True)
def close_figures():
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass


class TestComputeFFT:
    def test_detects_correct_frequency(self):
        """FFT of a pure sine wave detects the correct frequency."""
        np.random.seed(0)
        sample_rate = 1000.0
        freq_target = 50.0
        t = np.arange(0, 1.0, 1.0 / sample_rate)
        y = np.sin(2 * np.pi * freq_target * t)

        result = compute_fft(y, sample_rate=sample_rate)
        assert abs(result.dominant_freq - freq_target) < 2.0

    def test_power_spectrum_peaks_at_correct_frequency(self):
        """Power spectrum peaks at the frequency of the input sine."""
        np.random.seed(0)
        sample_rate = 500.0
        freq_target = 30.0
        t = np.arange(0, 2.0, 1.0 / sample_rate)
        y = np.sin(2 * np.pi * freq_target * t)

        freqs, psd = power_spectrum(y, sample_rate=sample_rate)
        peak_idx = np.argmax(psd)
        assert abs(freqs[peak_idx] - freq_target) < 1.0


class TestFilterSignal:
    def test_lowpass_removes_high_frequency(self):
        """Low-pass filter removes a high-frequency component."""
        np.random.seed(1)
        sample_rate = 1000.0
        t = np.arange(0, 1.0, 1.0 / sample_rate)
        low = np.sin(2 * np.pi * 10 * t)
        high = np.sin(2 * np.pi * 200 * t)
        y = low + high

        filtered = filter_signal(y, "lowpass", 50.0, sample_rate=sample_rate)

        # After low-pass at 50 Hz, the 200 Hz component should be gone.
        result = compute_fft(filtered, sample_rate=sample_rate)
        # Dominant frequency should be near 10 Hz, not 200
        assert abs(result.dominant_freq - 10.0) < 3.0

    def test_bandpass_isolates_target(self):
        """Band-pass filter isolates the target frequency."""
        np.random.seed(2)
        sample_rate = 1000.0
        t = np.arange(0, 1.0, 1.0 / sample_rate)
        f1, f2, f3 = 10.0, 80.0, 200.0
        y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + np.sin(2 * np.pi * f3 * t)

        filtered = filter_signal(y, "bandpass", (50.0, 120.0), sample_rate=sample_rate)

        result = compute_fft(filtered, sample_rate=sample_rate)
        assert abs(result.dominant_freq - f2) < 5.0

    def test_invalid_filter_type_raises(self):
        """Unknown filter type raises ValueError."""
        np.random.seed(0)
        y = np.sin(np.linspace(0, 10, 500))
        with pytest.raises(ValueError, match="Unknown filter type"):
            filter_signal(y, "invalid_type", 50.0, sample_rate=1000.0)
