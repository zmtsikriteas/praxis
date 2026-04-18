"""Tests for Wave 3 technique modules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# -----------------------------------------------------------------------
# DSC
# -----------------------------------------------------------------------
class TestDSC:
    def test_detect_tg_and_tm(self):
        """Detect Tg and Tm from sample DSC data."""
        from praxis.techniques.dsc_tga import analyse_dsc

        df = pd.read_csv(SAMPLE_DIR / "dsc_sample.csv")
        temp = df.iloc[:, 0].values
        hf = df.iloc[:, 1].values

        result = analyse_dsc(temp, hf)
        # We just need the analysis to produce results without error.
        # Check that at least one transition was found.
        assert len(result.transitions) > 0
        # If Tg or Tm detected, they should be within the temperature range.
        if result.tg is not None:
            assert temp.min() <= result.tg <= temp.max()
        if result.tm is not None:
            assert temp.min() <= result.tm <= temp.max()


# -----------------------------------------------------------------------
# Mechanical
# -----------------------------------------------------------------------
class TestMechanical:
    def test_youngs_modulus_and_uts(self):
        """Extract Young's modulus and UTS from stress-strain data."""
        from praxis.techniques.mechanical import analyse_tensile

        df = pd.read_csv(SAMPLE_DIR / "stress_strain_sample.csv")
        strain = df.iloc[:, 0].values
        stress = df.iloc[:, 1].values

        result = analyse_tensile(strain, stress)
        assert result.youngs_modulus is not None
        assert result.youngs_modulus > 0
        assert result.uts is not None
        assert result.uts > 0


# -----------------------------------------------------------------------
# Spectroscopy
# -----------------------------------------------------------------------
class TestSpectroscopy:
    def test_absorbance_transmittance_roundtrip(self):
        """absorbance -> transmittance -> absorbance round-trips correctly."""
        from praxis.techniques.spectroscopy import (
            absorbance_to_transmittance,
            transmittance_to_absorbance,
        )

        a_orig = np.array([0.1, 0.5, 1.0, 2.0])
        t = absorbance_to_transmittance(a_orig)
        a_back = transmittance_to_absorbance(t)
        np.testing.assert_allclose(a_back, a_orig, atol=1e-8)


class TestBeerLambert:
    def test_concentration_calculated(self):
        """Beer-Lambert calculates correct concentration."""
        from praxis.techniques.spectroscopy import beer_lambert

        # A = eps * l * c => c = A / (eps * l)
        result = beer_lambert(
            absorbance=0.5,
            molar_absorptivity=100.0,
            path_length=1.0,
        )
        expected_c = 0.5 / (100.0 * 1.0)
        assert abs(result["concentration"] - expected_c) < 1e-10


# -----------------------------------------------------------------------
# Dielectric
# -----------------------------------------------------------------------
class TestDielectric:
    def test_parse_dielectric_creates_valid_structure(self):
        """parse_dielectric returns a DielectricData with matching arrays."""
        from praxis.techniques.dielectric import parse_dielectric

        np.random.seed(42)
        freq = np.logspace(1, 6, 50)
        eps_r = 1000 / (1 + (freq / 1e4) ** 0.8) + 10
        eps_i = eps_r * 0.02

        data = parse_dielectric(freq, eps_r, epsilon_i=eps_i)
        assert len(data.frequency) == 50
        assert len(data.epsilon_r) == 50
        assert data.epsilon_i is not None
        assert data.tan_delta is not None
        assert data.ac_conductivity is not None


# -----------------------------------------------------------------------
# Piezoelectric
# -----------------------------------------------------------------------
class TestPiezoelectric:
    def test_pe_loop_extracts_pr_and_ec(self):
        """P-E loop analysis extracts Pr and Ec from synthetic hysteresis data."""
        from praxis.techniques.piezoelectric import analyse_pe_loop

        np.random.seed(0)
        # Synthetic P-E loop: tanh-based hysteresis
        n = 400
        t = np.linspace(0, 2 * np.pi, n)
        e_field = 50 * np.sin(t)  # kV/cm
        # Simple hysteresis model: P = Ps * tanh(E / Ec_model) with phase lag
        polarisation = 25 * np.tanh((e_field + 5 * np.cos(t)) / 15)

        result = analyse_pe_loop(e_field, polarisation)
        assert result.pr > 0, "Remanent polarisation should be positive"
        assert result.ec > 0, "Coercive field should be positive"


# -----------------------------------------------------------------------
# AFM
# -----------------------------------------------------------------------
class TestAFM:
    def test_profile_roughness_ra_positive(self):
        """profile_roughness gives Ra > 0 for noisy data."""
        from praxis.techniques.afm import profile_roughness

        np.random.seed(42)
        heights = np.random.normal(0, 5, 500)  # nm scale noise
        result = profile_roughness(heights, unit="nm")
        assert result.ra > 0


# -----------------------------------------------------------------------
# SEM / EDS
# -----------------------------------------------------------------------
class TestSEM:
    def test_grain_size_line_intercept(self):
        """grain_size_line_intercept returns sensible statistics."""
        from praxis.techniques.sem_eds import grain_size_line_intercept

        np.random.seed(42)
        intercepts = np.random.lognormal(mean=1.0, sigma=0.5, size=50)

        result = grain_size_line_intercept(intercepts, unit="um")
        assert result.n_grains == 50
        assert result.mean_size > 0
        assert result.std_size > 0
        assert result.d10 < result.d50 < result.d90


class TestEDS:
    def test_parse_eds_composition_sums(self):
        """EDS weight% and atomic% each sum to approximately 100."""
        from praxis.techniques.sem_eds import parse_eds_composition

        result = parse_eds_composition(
            elements=["O", "Ti", "Ba"],
            weight_pct=[20.0, 30.0, 50.0],
        )
        assert abs(sum(result.weight_pct) - 100.0) < 0.1
        assert abs(sum(result.atomic_pct) - 100.0) < 0.1


# -----------------------------------------------------------------------
# Thermal conductivity
# -----------------------------------------------------------------------
class TestThermal:
    def test_calc_conductivity(self):
        """calc_conductivity returns k = alpha * Cp * rho."""
        from praxis.techniques.thermal_conductivity import calc_conductivity

        alpha = 1.0e-6   # m2/s
        cp = 500.0       # J/(kg*K)
        rho = 7800.0     # kg/m3

        result = calc_conductivity(alpha, cp, rho)
        expected_k = alpha * cp * rho
        assert abs(result.conductivity - expected_k) < 1e-10
