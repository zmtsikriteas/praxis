"""Regression tests pinning numerical results against known values.

Each test checks a quantity against an analytic or reference value, so
unit or formula errors fail rather than passing a loose sanity check.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Peaks
# ---------------------------------------------------------------------------

class TestPeakMetrics:
    def _gaussian(self):
        x = np.linspace(0, 10, 500)
        sigma = 0.5
        y = 100 * np.exp(-(x - 5) ** 2 / (2 * sigma ** 2)) + 5
        return x, y, 2 * np.sqrt(2 * np.log(2)) * sigma, 100 * sigma * np.sqrt(2 * np.pi)

    def test_fwhm_and_area_match_analytic(self):
        from praxis.analysis.peaks import find_peaks_auto
        x, y, fwhm, area = self._gaussian()
        p = find_peaks_auto(x, y).peaks[0]
        assert p.fwhm == pytest.approx(fwhm, rel=1e-3)
        assert p.area == pytest.approx(area, rel=1e-3)

    def test_descending_axis_gives_same_results(self):
        from praxis.analysis.peaks import find_peaks_auto
        x, y, fwhm, area = self._gaussian()
        p = find_peaks_auto(x[::-1], y[::-1]).peaks[0]
        assert p.fwhm == pytest.approx(fwhm, rel=1e-3)
        assert p.area == pytest.approx(area, rel=1e-3)


# ---------------------------------------------------------------------------
# Mechanical
# ---------------------------------------------------------------------------

class TestMechanical:
    def test_toughness_in_mj_per_m3(self):
        from praxis.techniques.mechanical import analyse_tensile
        # Triangle up to 1000 MPa at 10 % strain: area = 0.5 * 0.1 * 1000 = 50 MJ/m3
        strain = np.linspace(0, 10, 101)
        stress = strain * 100
        res = analyse_tensile(strain, stress, strain_unit="percent")
        assert res.toughness == pytest.approx(50.0, rel=1e-6)

    def test_dma_runs(self):
        from praxis.techniques.mechanical import analyse_dma
        t = np.linspace(0, 200, 400)
        e_storage = 3000 / (1 + np.exp((t - 100) / 5)) + 10
        e_loss = 200 * np.exp(-((t - 100) / 10) ** 2) + 1
        res = analyse_dma(t, e_storage, e_loss)
        assert res.tg_loss_modulus == pytest.approx(100, abs=1)


# ---------------------------------------------------------------------------
# Nanoindentation
# ---------------------------------------------------------------------------

class TestNanoindentation:
    def test_hardness_and_modulus_in_gpa(self):
        from praxis.techniques.nanoindentation import analyse_indent
        # Build an unloading curve P = A (h - hf)^m with known stiffness.
        p_max, h_max, hf, m = 10.0, 300.0, 200.0, 1.5  # mN, nm
        a = p_max / (h_max - hf) ** m
        h_load = np.linspace(0, h_max, 50)
        p_load = p_max * (h_load / h_max) ** 2
        h_unload = np.linspace(h_max, hf, 50)
        p_unload = a * (h_unload - hf) ** m
        depth = np.concatenate([h_load, h_unload[1:]])
        load = np.concatenate([p_load, p_unload[1:]])

        res = analyse_indent(depth, load, tip="berkovich")

        s = a * m * (h_max - hf) ** (m - 1)  # mN/nm
        hc = h_max - 0.75 * p_max / s
        area_m2 = 24.5 * (hc * 1e-9) ** 2
        h_expected = p_max * 1e-3 / area_m2 / 1e9
        er_expected = (np.sqrt(np.pi) / (2 * 1.034)) * (s * 1e6) / np.sqrt(area_m2) / 1e9

        assert res.hardness_gpa == pytest.approx(h_expected, rel=1e-6)
        assert res.reduced_modulus_gpa == pytest.approx(er_expected, rel=1e-6)
        # Sample modulus must exceed the reduced modulus for a finite tip modulus
        assert res.modulus_gpa > 0

    def test_unit_scaling(self):
        from praxis.techniques.nanoindentation import analyse_indent
        h = np.concatenate([np.linspace(0, 300, 50), np.linspace(300, 200, 50)[1:]])
        p = np.concatenate([10 * (np.linspace(0, 300, 50) / 300) ** 2,
                            10 * ((np.linspace(300, 200, 50)[1:] - 200) / 100) ** 1.5])
        r_nm = analyse_indent(h, p, depth_unit="nm", load_unit="mN")
        r_um = analyse_indent(h / 1000, p * 1000, depth_unit="um", load_unit="uN")
        assert r_um.hardness_gpa == pytest.approx(r_nm.hardness_gpa, rel=1e-9)


# ---------------------------------------------------------------------------
# Piezoelectric
# ---------------------------------------------------------------------------

def test_d33_effective_pm_per_v():
    from praxis.techniques.piezoelectric import analyse_se_curve
    # 0.1 % strain at 10 kV/cm -> 1e-3 / 1e6 V/m = 1e-9 m/V = 1000 pm/V
    e = np.linspace(-10, 10, 201)
    s = 0.1 * (e / 10) ** 2
    res = analyse_se_curve(e, s)
    assert res.d33_eff == pytest.approx(1000.0, rel=1e-9)


# ---------------------------------------------------------------------------
# DSC
# ---------------------------------------------------------------------------

class TestDSC:
    def _melt(self):
        # Endothermic peak (negative, TA convention) in W/g
        t = np.linspace(100, 200, 2001)
        hf = -0.5 * np.exp(-((t - 150) / 3) ** 2)
        area_wg_k = 0.5 * 3 * np.sqrt(np.pi)  # integral over T
        return t, hf, area_wg_k

    def test_enthalpy_uses_heating_rate(self):
        from praxis.techniques.dsc_tga import analyse_dsc
        t, hf, area = self._melt()
        res = analyse_dsc(t, hf, heating_rate=10.0, dh_reference=100.0)
        melt = [tr for tr in res.transitions if tr.kind == "Tm"][0]
        expected = area / (10.0 / 60.0)  # J/g
        assert melt.enthalpy == pytest.approx(expected, rel=0.02)
        assert res.crystallinity == pytest.approx(expected, rel=0.02)

    def test_mass_normalisation(self):
        from praxis.techniques.dsc_tga import analyse_dsc
        t, hf, _ = self._melt()
        a = analyse_dsc(t, hf, heating_rate=10.0)
        b = analyse_dsc(t, hf * 5.0, heating_rate=10.0, sample_mass=5.0)  # mW, 5 mg
        ea = [tr for tr in a.transitions if tr.kind == "Tm"][0].enthalpy
        eb = [tr for tr in b.transitions if tr.kind == "Tm"][0].enthalpy
        assert eb == pytest.approx(ea, rel=1e-9)

    def test_no_enthalpy_without_heating_rate(self):
        from praxis.techniques.dsc_tga import analyse_dsc
        t, hf, _ = self._melt()
        with pytest.warns(UserWarning):
            res = analyse_dsc(t, hf, dh_reference=100.0)
        assert res.crystallinity is None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_mo_ka_wavelength():
    from praxis.techniques.xrd import WAVELENGTHS
    assert WAVELENGTHS["Mo_Ka"] == pytest.approx(0.7107, abs=1e-3)


def test_hydrogen_monoisotopic_mass():
    from praxis.techniques.mass_spec import MONOISOTOPIC_MASS
    assert MONOISOTOPIC_MASS["H"] == pytest.approx(1.007825, abs=1e-6)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_paired_t_test_drops_incomplete_pairs():
    from praxis.analysis.statistics import t_test
    a = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([1.5, 2.0, np.nan, 4.2])
    res = t_test(a, b, paired=True)
    assert res.df == 1


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TestLoader:
    def test_text_columns_kept(self, tmp_path):
        from praxis.core.loader import load_data
        p = tmp_path / "s.csv"
        p.write_text("sample,value\nA,1.0\nB,2.0\n")
        df = load_data(p)
        assert list(df["sample"]) == ["A", "B"]

    def test_signed_comma_decimals(self, tmp_path):
        from praxis.core.loader import load_data
        p = tmp_path / "eu.csv"
        p.write_text("x;y\n-1,5;2,25\n-0,75;-3,5\n1,0;4,0\n")
        df = load_data(p)
        assert df.iloc[:, 0].tolist() == [-1.5, -0.75, 1.0]
        assert df.iloc[:, 1].tolist() == [2.25, -3.5, 4.0]

    def test_jcamp_packed_x_spacing(self, tmp_path):
        from praxis.core.loader import load_data
        p = tmp_path / "s.jdx"
        p.write_text(
            "##TITLE=t\n##XFACTOR=1\n##YFACTOR=1\n##FIRSTX=100\n##LASTX=105\n"
            "##DELTAX=1\n##NPOINTS=6\n##XYDATA=(X++(Y..Y))\n100 1 2 3\n103 4 5 6\n##END=\n"
        )
        df = load_data(p)
        assert df["x"].tolist() == [100, 101, 102, 103, 104, 105]
        assert df["y"].tolist() == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

class TestFitting:
    def test_custom_expression_fits(self):
        from praxis.analysis.fitting import fit_curve
        x = np.linspace(0, 5, 50)
        y = 3 * np.exp(-0.7 * x) + 1
        res = fit_curve(x, y, model="a * exp(-b * x) + c")
        assert res.params["a"] == pytest.approx(3.0, rel=1e-4)
        assert res.params["b"] == pytest.approx(0.7, rel=1e-4)
        assert res.params["c"] == pytest.approx(1.0, rel=1e-4)

    @pytest.mark.parametrize("expr", [
        "np.load.__globals__",
        "__import__('os')",
        "(1).__class__",
        "a * x + b.__class__",
        "open('f')",
    ])
    def test_custom_expression_rejects_unsafe_code(self, expr):
        from praxis.analysis.fitting import fit_curve
        x = np.linspace(0, 1, 10)
        with pytest.raises(ValueError):
            fit_curve(x, x, model=expr)

    def test_propagate_error_rejects_unsafe_code(self):
        from praxis.analysis.statistics import propagate_error
        with pytest.raises(ValueError):
            propagate_error("a.__class__", {"a": 1.0}, {"a": 0.1})

    def test_weights_with_nan_and_range(self):
        from praxis.analysis.fitting import fit_curve
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1
        x[3] = np.nan
        res = fit_curve(x, y, model="linear", weights=np.ones(50), x_range=(0, 8))
        assert res.params["slope"] == pytest.approx(2.0, rel=1e-6)

    def test_confidence_band_nan_when_unavailable(self):
        from praxis.analysis.fitting import fit_curve
        x = np.linspace(0, 10, 50)
        res = fit_curve(x, 2 * x + 1, model="linear")

        def _fail(**kwargs):
            raise RuntimeError("no covariance")
        res.result.eval_uncertainty = _fail
        with pytest.warns(UserWarning):
            _, lo, hi = res.confidence_band()
        assert np.all(np.isnan(lo)) and np.all(np.isnan(hi))


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------

class TestFFT:
    def test_dc_amplitude(self):
        from praxis.analysis.fft import compute_fft
        res = compute_fft(np.full(8, 1.0), remove_dc=False)
        assert res.amplitude[0] == pytest.approx(1.0)

    def test_nyquist_amplitude(self):
        from praxis.analysis.fft import compute_fft
        res = compute_fft(np.array([1.0, -1.0] * 4), remove_dc=False)
        assert res.freq[-1] == pytest.approx(0.5)
        assert res.amplitude[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

class TestInterpolation:
    @pytest.mark.parametrize("method", ["linear", "cubic", "akima", "pchip", "quadratic", "spline"])
    def test_fill_value_respected(self, method):
        from praxis.analysis.interpolation import interpolate
        x = np.arange(1.0, 6.0)
        _, y_new = interpolate(x, x ** 2, [0.0, 3.0, 6.0], method=method, fill_value=0)
        assert y_new[0] == 0 and y_new[2] == 0
        assert y_new[1] == pytest.approx(9.0, abs=1e-6)

    def test_zero_smoothing_is_exact(self):
        from praxis.analysis.interpolation import interpolate
        rng = np.random.default_rng(1)
        x = np.linspace(0, 10, 30)
        y = np.sin(x) + rng.normal(0, 0.1, 30)
        _, y_new = interpolate(x, y, x, method="spline", smoothing=0)
        assert np.allclose(y_new, y, atol=1e-9)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def test_template_chains_baseline_then_smoothing():
    from praxis.analysis.templates import AnalysisStep, AnalysisTemplate, execute_template
    x = np.linspace(0, 10, 200)
    y = np.exp(-(x - 5) ** 2) + 0.5 * x
    template = AnalysisTemplate(
        name="t", description="",
        steps=[
            AnalysisStep(function="analysis.baseline.correct_baseline",
                         params={"method": "polynomial", "order": 1}, description="baseline"),
            AnalysisStep(function="analysis.smoothing.smooth",
                         params={"method": "savgol", "window": 11}, description="smooth"),
        ],
    )
    out = execute_template(template, x, y)
    # Baseline step removed the slope; smoothing ran on the corrected data
    assert not np.allclose(out[0]["y_out"], y)
    assert len(out[1]["y_out"]) == len(y)
    assert np.max(np.abs(out[1]["y_out"] - out[0]["y_out"])) < 0.1


# ---------------------------------------------------------------------------
# Other techniques
# ---------------------------------------------------------------------------

def test_nmr_doublet_reported_once():
    from praxis.techniques.nmr import predict_multiplicity
    cs = np.linspace(0, 5, 5000)
    lor = lambda c: 1 / (1 + ((cs - c) / 0.002) ** 2)
    res = predict_multiplicity(cs, lor(2.0) + lor(2.0175))
    assert len(res) == 1
    assert res[0]["multiplicity"] == "d"
    assert res[0]["chemical_shift"] == pytest.approx(2.00875, abs=1e-3)


def test_eds_tungsten_and_unknown_element():
    from praxis.techniques.sem_eds import parse_eds_composition
    res = parse_eds_composition(["W", "O"], weight_pct=[50, 50])
    w_at = 50 / 183.84 / (50 / 183.84 + 50 / 16.00) * 100
    assert res.atomic_pct[0] == pytest.approx(w_at, rel=1e-6)
    with pytest.raises(ValueError):
        parse_eds_composition(["Xx", "O"], weight_pct=[50, 50])


def test_bjh_adsorption_branch():
    from praxis.techniques.bet import bjh_pore_distribution
    p = np.linspace(0.05, 0.95, 40)
    q = 100 + 200 / (1 + np.exp(-(p - 0.7) / 0.03))  # capillary condensation step
    res = bjh_pore_distribution(p, q, branch="adsorption")
    assert len(res.diameters) > 0
    assert res.total_pore_volume > 0


def test_dqdv_short_input_keeps_length():
    from praxis.techniques.battery_cycling import compute_dqdv
    res = compute_dqdv([3.0, 3.1, 3.2, 3.3], [0.0, 1.0, 2.0, 3.0])
    assert len(res.dqdv) == len(res.voltage) == 4


def test_batch_keys_unique_for_recursive_glob(tmp_path):
    from praxis.batch.batch import load_batch
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "sample.csv").write_text("x,y\n1,2\n3,4\n")
    data = load_batch("**/*.csv", tmp_path)
    assert set(data) == {"a/sample.csv", "b/sample.csv"}


# ---------------------------------------------------------------------------
# XPS
# ---------------------------------------------------------------------------

class TestXPS:
    def test_survey_composition_positive(self):
        from praxis.techniques.xps import analyse_survey
        be = np.linspace(0, 700, 3500)
        y = (1000 * np.exp(-((be - 284.8) / 1.0) ** 2)
             + 2000 * np.exp(-((be - 532.0) / 1.2) ** 2) + 50)
        res = analyse_survey(be, y)
        assert all(p.area > 0 for p in res.peaks if p.area is not None)
        assert set(res.composition) >= {"C", "O"}

    def test_unknown_background_raises(self):
        from praxis.techniques.xps import fit_highres
        be = np.linspace(280, 290, 100)
        with pytest.raises(ValueError):
            fit_highres(be, np.exp(-((be - 285) / 0.8) ** 2), background="tougaard")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

class TestPlotter:
    def test_alpha_and_colour_options(self):
        import matplotlib
        matplotlib.use("Agg")
        from praxis.core.plotter import plot_data
        x = np.linspace(0, 1, 10)
        plot_data(x, x, kind="area", alpha=0.2)
        plot_data(x, x, kind="fill_between", alpha=0.5)
        plot_data(x, [x, 2 * x], kind="waterfall", colour="k")

    def test_journal_style_sets_figure_size(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from praxis.core.plotter import plot_data
        from praxis.core.utils import apply_style
        with plt.rc_context():
            apply_style("nature")
            fig, _ = plot_data(np.arange(3.0), np.arange(3.0))
            assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 2.625))

    def test_box_plot_with_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        from praxis.core.plotter import plot_data
        _, ax = plot_data(None, [np.arange(5.0), np.arange(5.0) + 1], kind="box", labels=["A", "B"])
        assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]
