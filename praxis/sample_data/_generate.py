"""Generator for built-in sample datasets.

Run from the project root with:

    python -m praxis.sample_data._generate

Each function produces a small, deterministic, physically-plausible CSV
for one technique. The output is committed to the repo so end users get
them with a pip install; the generator is kept for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def _save(name: str, df: pd.DataFrame, header: str = "") -> None:
    path = OUT / f"{name}.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in header.strip().splitlines():
            f.write(f"# {line}\n")
        df.to_csv(f, index=False, lineterminator="\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}")


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------

def gen_xrd() -> None:
    """Powder XRD pattern of silicon (Cu Ka), main reflections included."""
    rng = np.random.default_rng(0)
    two_theta = np.arange(10.0, 90.0, 0.02)
    # Si reflections (hkl): 111, 220, 311, 400, 331, 422
    peaks = [(28.44, 1000, 0.12), (47.30, 500, 0.14), (56.12, 340, 0.15),
             (69.13, 130, 0.17), (76.37, 180, 0.18), (88.03, 210, 0.20)]
    intensity = np.zeros_like(two_theta)
    for pos, height, fwhm in peaks:
        sigma = fwhm / 2.355
        intensity += height * np.exp(-0.5 * ((two_theta - pos) / sigma) ** 2)
    intensity += 25 + 0.1 * two_theta  # sloping background
    intensity += rng.normal(0, 4, size=two_theta.size)
    _save("xrd_silicon", pd.DataFrame({"two_theta_deg": two_theta,
                                       "intensity": intensity.round(2)}),
          header="Powder XRD of Si, Cu Ka (lambda=1.5406 A). Synthetic.")


def gen_saxs() -> None:
    """Small-angle scattering I(q) with Guinier region + Porod tail."""
    rng = np.random.default_rng(1)
    q = np.logspace(-2.3, 0.3, 200)  # 0.005 to ~2 /A
    Rg = 30.0  # Angstroms
    I0 = 1e4
    # Guinier at low q, Porod (q^-4) at high q, smooth crossover
    guinier = I0 * np.exp(-(q * Rg) ** 2 / 3)
    porod = 0.5 / q ** 4
    intensity = guinier + porod
    intensity *= rng.lognormal(0, 0.03, size=q.size)
    _save("saxs_iq", pd.DataFrame({"q_invA": q, "intensity": intensity}),
          header="SAXS of monodisperse spherical particles, Rg=30 A. Synthetic.")


# ---------------------------------------------------------------------------
# Thermal
# ---------------------------------------------------------------------------

def gen_dsc() -> None:
    """DSC trace: Tg step + crystallisation exotherm + melting endotherm."""
    rng = np.random.default_rng(2)
    T = np.arange(-20.0, 260.0, 0.5)
    # Tg ~ 80 C: step in heat flow (~0.2 J/g/C)
    tg_step = 0.2 * (1 / (1 + np.exp(-(T - 80) / 2.0)))
    # Tc ~ 135 C: exotherm (positive)
    tc = 0.8 * np.exp(-0.5 * ((T - 135) / 5.0) ** 2)
    # Tm ~ 220 C: endotherm (negative)
    tm = -1.4 * np.exp(-0.5 * ((T - 220) / 4.0) ** 2)
    baseline = -0.05 - 0.002 * T
    hf = baseline + tg_step + tc + tm + rng.normal(0, 0.01, size=T.size)
    _save("dsc_polymer", pd.DataFrame({"temperature_C": T,
                                       "heat_flow_mW_per_mg": hf.round(4)}),
          header="DSC trace of a semi-crystalline polymer. Tg~80 C, Tc~135 C, Tm~220 C. Synthetic.")


def gen_tga() -> None:
    """TGA decomposition of PMMA-like polymer: single-step weight loss."""
    rng = np.random.default_rng(3)
    T = np.arange(25.0, 600.0, 1.0)
    # Sigmoidal weight loss centred at ~380 C
    mass_pct = 100 - 98 * (1 / (1 + np.exp(-(T - 380) / 18.0)))
    mass_pct += rng.normal(0, 0.15, size=T.size)
    _save("tga_pmma", pd.DataFrame({"temperature_C": T,
                                    "mass_pct": mass_pct.round(3)}),
          header="TGA of PMMA in N2, 10 C/min. Single-step decomposition ~380 C. Synthetic.")


# ---------------------------------------------------------------------------
# Mechanical
# ---------------------------------------------------------------------------

def gen_stress_strain() -> None:
    """Tensile test of a mild-steel-like specimen with elastic + plastic."""
    rng = np.random.default_rng(4)
    strain = np.linspace(0, 0.20, 400)
    E = 200e3  # MPa
    sigma_y = 250.0  # yield
    # Ramberg-Osgood-like: elastic + power-law hardening
    plastic_strain = np.maximum(0.0, strain - sigma_y / E)
    stress = np.where(
        strain * E <= sigma_y,
        strain * E,
        sigma_y + 1200 * plastic_strain ** 0.5,
    )
    # Necking drop after peak
    peak_idx = int(0.85 * strain.size)
    stress[peak_idx:] *= np.linspace(1.0, 0.92, strain.size - peak_idx)
    stress += rng.normal(0, 1.5, size=strain.size)
    _save("stress_strain_steel",
          pd.DataFrame({"strain": strain.round(5),
                        "stress_MPa": stress.round(2)}),
          header="Tensile test of mild-steel-like specimen, E=200 GPa, yield~250 MPa. Synthetic.")


# ---------------------------------------------------------------------------
# Spectroscopy
# ---------------------------------------------------------------------------

def gen_ftir() -> None:
    """FTIR of PMMA with characteristic C=O and C-H bands."""
    rng = np.random.default_rng(5)
    wn = np.arange(400.0, 4000.0, 2.0)
    # Bands: C=O stretch 1730, C-O 1150, CH3 bend 1440, C-H stretches 2950/2990
    peaks = [(1730, 0.6, 20), (1150, 0.45, 25), (1440, 0.3, 18),
             (2950, 0.35, 25), (2990, 0.3, 20), (750, 0.25, 30)]
    absorbance = np.full_like(wn, 0.02)
    for pos, h, w in peaks:
        absorbance += h * np.exp(-0.5 * ((wn - pos) / (w / 2.355)) ** 2)
    transmittance = 100 * 10 ** (-absorbance)
    transmittance += rng.normal(0, 0.15, size=wn.size)
    _save("ftir_pmma",
          pd.DataFrame({"wavenumber_cm-1": wn,
                        "transmittance_pct": transmittance.round(3)}),
          header="FTIR of PMMA (transmittance). C=O stretch at 1730 cm-1. Synthetic.")


def gen_raman() -> None:
    """Raman spectrum of silicon: sharp peak at 520.7 cm-1."""
    rng = np.random.default_rng(6)
    shift = np.arange(100.0, 1200.0, 0.5)
    # Main Si peak + 2TO at ~950 cm-1
    intensity = (
        8500 * np.exp(-0.5 * ((shift - 520.7) / 2.1) ** 2)
        + 120 * np.exp(-0.5 * ((shift - 950) / 35) ** 2)
        + 40
    )
    intensity += rng.normal(0, 12, size=shift.size)
    _save("raman_silicon",
          pd.DataFrame({"raman_shift_cm-1": shift,
                        "intensity": intensity.round(1)}),
          header="Raman of crystalline silicon, 532 nm excitation. Main peak 520.7 cm-1. Synthetic.")


# ---------------------------------------------------------------------------
# Electrical
# ---------------------------------------------------------------------------

def gen_impedance() -> None:
    """EIS of a single RC circuit: semicircle in Nyquist plot."""
    rng = np.random.default_rng(7)
    f = np.logspace(-1, 6, 80)  # 0.1 Hz to 1 MHz
    omega = 2 * np.pi * f
    R, C = 1000.0, 1e-7  # 1 kOhm, 100 nF
    R_inf = 50.0
    Z = R_inf + R / (1 + 1j * omega * R * C)
    Z *= rng.lognormal(0, 0.005, size=f.size)
    _save("impedance_rc",
          pd.DataFrame({"frequency_Hz": f,
                        "z_real_ohm": Z.real.round(3),
                        "z_imag_ohm": Z.imag.round(3)}),
          header="EIS of a series R + (R||C) circuit. R_s=50, R_p=1000 ohm, C=100 nF. Synthetic.")


def gen_iv_diode() -> None:
    """I-V curve of a Si diode: Shockley equation with series resistance."""
    rng = np.random.default_rng(8)
    V = np.linspace(-0.3, 0.8, 220)
    I0 = 1e-9   # saturation current (A)
    n = 1.5     # ideality
    Vt = 0.02585  # thermal voltage at 300 K
    Rs = 0.5    # series resistance (ohm)
    # Iterative: I = I0 * (exp((V - I*Rs)/(n*Vt)) - 1)
    I = np.zeros_like(V)
    for _ in range(20):
        I = I0 * (np.exp((V - I * Rs) / (n * Vt)) - 1)
    I += rng.normal(0, 1e-7, size=V.size)
    _save("iv_diode",
          pd.DataFrame({"voltage_V": V.round(4),
                        "current_A": I}),
          header="Si diode I-V (Shockley + series resistance). I0=1e-9 A, n=1.5, Rs=0.5 ohm. Synthetic.")


def gen_cv() -> None:
    """Reversible redox CV: classic duck-shape with peak separation ~59 mV."""
    rng = np.random.default_rng(9)
    # Triangular sweep
    n = 400
    half = n // 2
    V = np.concatenate([np.linspace(-0.4, 0.6, half),
                        np.linspace(0.6, -0.4, n - half)])
    # Anodic peak at +0.25 V, cathodic at +0.19 V
    Eox, Ered = 0.25, 0.19
    ipeak = 80e-6  # 80 uA
    width = 0.06
    direction = np.concatenate([np.ones(half), -np.ones(n - half)])
    I = np.zeros_like(V)
    forward = direction > 0
    I[forward] = ipeak * np.exp(-((V[forward] - Eox) / width) ** 2)
    I[~forward] = -ipeak * np.exp(-((V[~forward] - Ered) / width) ** 2)
    # Add capacitive baseline
    I += 5e-6 * direction
    I += rng.normal(0, 0.5e-6, size=n)
    _save("cv_redox",
          pd.DataFrame({"potential_V": V.round(4),
                        "current_A": I}),
          header="Cyclic voltammetry of a reversible redox couple. E_ox=0.25 V, E_red=0.19 V. Synthetic.")


# ---------------------------------------------------------------------------
# Dielectric / piezoelectric / magnetic
# ---------------------------------------------------------------------------

def gen_dielectric() -> None:
    """Frequency-dependent permittivity with a Debye relaxation."""
    rng = np.random.default_rng(10)
    f = np.logspace(0, 9, 90)  # 1 Hz to 1 GHz
    omega = 2 * np.pi * f
    eps_inf, eps_s = 4.0, 80.0
    tau = 1e-6  # relaxation time (s)
    eps = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)
    eps_real = eps.real * rng.lognormal(0, 0.005, size=f.size)
    eps_imag = -eps.imag * rng.lognormal(0, 0.01, size=f.size)
    tan_delta = eps_imag / eps_real
    _save("dielectric_relaxation",
          pd.DataFrame({"frequency_Hz": f,
                        "eps_real": eps_real.round(4),
                        "eps_imag": eps_imag.round(4),
                        "tan_delta": tan_delta.round(5)}),
          header="Debye dielectric relaxation. eps_inf=4, eps_s=80, tau=1 us. Synthetic.")


def gen_piezo_pe_loop() -> None:
    """Ferroelectric P-E hysteresis loop (PZT-like)."""
    rng = np.random.default_rng(11)
    n = 500
    # Triangular E sweep over two cycles
    E = 60e3 * np.sin(np.linspace(0, 4 * np.pi, n))  # V/m
    # Hysteretic P: arctan with shift depending on dE/dt sign
    dEdt = np.gradient(E)
    Ec = 15e3   # coercive field
    Ps = 25e-2  # spontaneous polarisation (C/m^2)
    P = Ps * np.tanh((E + np.sign(dEdt) * Ec) / (1.5 * Ec))
    P += rng.normal(0, 5e-4, size=n)
    _save("piezo_pe_loop",
          pd.DataFrame({"E_field_V_per_m": E.round(2),
                        "polarisation_C_per_m2": P.round(5)}),
          header="P-E hysteresis loop, PZT-like. Ps=0.25 C/m^2, Ec=15 kV/m. Synthetic.")


def gen_magnetometry() -> None:
    """Ferromagnetic M-H loop with coercivity and saturation."""
    rng = np.random.default_rng(12)
    H = np.concatenate([
        np.linspace(0, 1.0, 100),
        np.linspace(1.0, -1.0, 200),
        np.linspace(-1.0, 1.0, 200),
    ])  # T
    Hc = 0.05   # coercive field (T)
    Ms = 1.4    # saturation moment (emu/g)
    dHdt = np.gradient(H)
    M = Ms * np.tanh((H - np.sign(dHdt) * Hc) / 0.04)
    M += rng.normal(0, 0.005, size=H.size)
    _save("mh_loop",
          pd.DataFrame({"H_field_T": H.round(5),
                        "moment_emu_per_g": M.round(5)}),
          header="Ferromagnetic M-H hysteresis loop. Ms=1.4 emu/g, Hc=0.05 T. Synthetic.")


# ---------------------------------------------------------------------------
# Spectroscopy extras
# ---------------------------------------------------------------------------

def gen_uvvis() -> None:
    """UV-Vis absorption of gold nanoparticles: plasmon at ~520 nm."""
    rng = np.random.default_rng(13)
    wl = np.arange(300.0, 800.0, 1.0)
    A = (
        1.4 * np.exp(-0.5 * ((wl - 525) / 35) ** 2)
        + 0.05 * (300 / wl) ** 4   # background
    )
    A += rng.normal(0, 0.005, size=wl.size)
    _save("uvvis_au_np",
          pd.DataFrame({"wavelength_nm": wl,
                        "absorbance": A.round(4)}),
          header="UV-Vis of gold nanoparticles in water. Plasmon ~525 nm. Synthetic.")


def gen_nmr() -> None:
    """1H NMR of ethanol: triplet (CH3), quartet (CH2), broad OH."""
    rng = np.random.default_rng(14)
    delta = np.arange(0.0, 8.0, 0.005)
    # Coupling constants in ppm units (J ~ 7 Hz at 400 MHz -> 0.0175 ppm)
    j = 0.0175
    width = 0.005

    def _peak(centre: float, height: float) -> np.ndarray:
        return height * np.exp(-0.5 * ((delta - centre) / width) ** 2)

    spec = np.zeros_like(delta)
    # CH3 triplet at 1.18 ppm (1:2:1)
    spec += _peak(1.18 - j, 1.0) + _peak(1.18, 2.0) + _peak(1.18 + j, 1.0)
    # CH2 quartet at 3.69 ppm (1:3:3:1)
    for i, h in zip([-1.5, -0.5, 0.5, 1.5], [0.5, 1.5, 1.5, 0.5]):
        spec += _peak(3.69 + i * j, h)
    # OH broad singlet at 2.6 ppm
    spec += 0.6 * np.exp(-0.5 * ((delta - 2.6) / 0.06) ** 2)
    spec += rng.normal(0, 0.005, size=delta.size)
    _save("nmr_ethanol",
          pd.DataFrame({"chemical_shift_ppm": delta.round(4),
                        "intensity": spec.round(4)}),
          header="1H NMR of ethanol in CDCl3, 400 MHz. CH3 triplet, CH2 quartet, OH broad. Synthetic.")


def gen_xps() -> None:
    """XPS C1s region: graphitic + C-O + C=O components on a Shirley background."""
    rng = np.random.default_rng(15)
    BE = np.arange(280.0, 295.0, 0.05)
    components = [(284.6, 8000, 0.9),  # C-C
                  (286.2, 1800, 1.0),  # C-O
                  (288.5, 900, 1.1)]   # C=O / O-C=O
    counts = np.zeros_like(BE)
    for pos, h, w in components:
        sigma = w / 2.355
        counts += h * np.exp(-0.5 * ((BE - pos) / sigma) ** 2)
    # Shirley-like background that grows on the low-BE side
    counts += 1500 + 800 * (BE - BE.min()) / (BE.max() - BE.min())
    counts += rng.normal(0, 25, size=BE.size)
    _save("xps_c1s",
          pd.DataFrame({"binding_energy_eV": BE.round(3),
                        "counts_per_s": counts.round(1)}),
          header="XPS C1s region with C-C, C-O, C=O components. Synthetic.")


def gen_eds() -> None:
    """EDS spectrum of NaCl: Na K and Cl K lines."""
    rng = np.random.default_rng(16)
    E = np.arange(0.0, 10.0, 0.01)
    peaks = [(1.04, 9000, 0.07),    # Na Ka
             (2.62, 17000, 0.10),   # Cl Ka
             (2.81, 2200, 0.10)]    # Cl Kb
    counts = np.zeros_like(E)
    for pos, h, w in peaks:
        counts += h * np.exp(-0.5 * ((E - pos) / (w / 2.355)) ** 2)
    counts += 60 * np.exp(-E / 2.5) + 30  # bremsstrahlung-ish background
    counts += rng.normal(0, 8, size=E.size)
    counts = np.clip(counts, 0, None)
    _save("eds_nacl",
          pd.DataFrame({"energy_keV": E.round(3),
                        "counts": counts.round(1)}),
          header="EDS spectrum of NaCl: Na Ka 1.04 keV, Cl Ka 2.62 keV. Synthetic.")


# ---------------------------------------------------------------------------
# Microscopy / surface
# ---------------------------------------------------------------------------

def gen_afm_profile() -> None:
    """AFM line profile: nm-scale roughness with a few features."""
    rng = np.random.default_rng(17)
    x = np.linspace(0, 5.0, 1000)  # microns
    # Sinusoidal modulation + random roughness
    z = 8 * np.sin(2 * np.pi * x / 1.2) + 3 * np.sin(2 * np.pi * x / 0.3 + 1)
    z += rng.normal(0, 1.5, size=x.size)
    z += np.cumsum(rng.normal(0, 0.05, size=x.size))  # slight drift
    _save("afm_profile",
          pd.DataFrame({"x_um": x.round(4),
                        "height_nm": z.round(3)}),
          header="AFM line profile, 5 um scan. nm-scale roughness with periodic features. Synthetic.")


def gen_bet() -> None:
    """N2 BET adsorption isotherm: type IV with hysteresis-free uptake."""
    rng = np.random.default_rng(18)
    p_p0 = np.linspace(0.005, 0.99, 60)
    # BET equation rearranged to give n_ads
    n_m, C = 5.0, 100.0   # monolayer capacity (mmol/g), BET constant
    n_ads = n_m * C * p_p0 / ((1 - p_p0) * (1 + (C - 1) * p_p0))
    # Add capillary condensation rise
    n_ads += 12 / (1 + np.exp(-(p_p0 - 0.55) / 0.08))
    n_ads *= rng.lognormal(0, 0.005, size=p_p0.size)
    _save("bet_isotherm",
          pd.DataFrame({"relative_pressure": p_p0.round(4),
                        "n_adsorbed_mmol_per_g": n_ads.round(4)}),
          header="N2 BET isotherm, 77 K. Type IV with capillary condensation around p/p0=0.55. Synthetic.")


# ---------------------------------------------------------------------------
# Chromatography / mass spec
# ---------------------------------------------------------------------------

def gen_hplc() -> None:
    """HPLC chromatogram with three resolved peaks of different intensity."""
    rng = np.random.default_rng(19)
    t = np.arange(0.0, 15.0, 0.01)
    peaks = [(3.2, 280, 0.10), (5.6, 420, 0.13), (9.4, 180, 0.18)]
    signal = np.zeros_like(t)
    for tr, h, w in peaks:
        signal += h * np.exp(-0.5 * ((t - tr) / (w / 2.355)) ** 2)
    signal += 4 + 0.2 * t  # rising baseline
    signal += rng.normal(0, 0.6, size=t.size)
    _save("hplc_chromatogram",
          pd.DataFrame({"time_min": t.round(3),
                        "absorbance_mAU": signal.round(2)}),
          header="HPLC chromatogram, 254 nm UV detection, 3 analytes. Synthetic.")


def gen_mass_spec() -> None:
    """Mass spectrum with discrete peaks at known m/z (caffeine fragmentation)."""
    rng = np.random.default_rng(20)
    mz = np.arange(40.0, 220.0, 0.02)
    # Caffeine major ions: 194 (M+), 109, 67, 55, 42
    peaks = [(194, 100, 0.15), (109, 65, 0.13),
             (82, 35, 0.12), (67, 28, 0.12),
             (55, 18, 0.11), (42, 12, 0.10)]
    intensity = np.zeros_like(mz)
    for pos, h, w in peaks:
        intensity += h * np.exp(-0.5 * ((mz - pos) / (w / 2.355)) ** 2)
    intensity += rng.normal(0, 0.3, size=mz.size).clip(0)
    _save("mass_spec_caffeine",
          pd.DataFrame({"mz": mz.round(3),
                        "rel_intensity_pct": intensity.round(2)}),
          header="EI-MS of caffeine (M+ at 194). Synthetic.")


# ---------------------------------------------------------------------------
# Mechanical / thermal extras
# ---------------------------------------------------------------------------

def gen_dma() -> None:
    """DMA temperature sweep: storage modulus drop + tan delta peak at Tg."""
    rng = np.random.default_rng(21)
    T = np.arange(-40.0, 180.0, 1.0)
    Tg = 65.0  # Tg in C
    # E' drops two decades around Tg
    E_prime = 10 ** (3.5 - 2.0 / (1 + np.exp(-(T - Tg) / 6.0)))
    # tan delta peaks at Tg
    tan_d = 0.05 + 1.4 * np.exp(-0.5 * ((T - Tg) / 9.0) ** 2)
    E_prime *= rng.lognormal(0, 0.01, size=T.size)
    tan_d += rng.normal(0, 0.005, size=T.size)
    _save("dma_polymer",
          pd.DataFrame({"temperature_C": T,
                        "storage_modulus_MPa": E_prime.round(2),
                        "tan_delta": tan_d.round(4)}),
          header="DMA temperature sweep, 1 Hz. Tg ~65 C from tan delta peak. Synthetic.")


def gen_hardness() -> None:
    """Vickers hardness: indentation diagonal vs applied load."""
    rng = np.random.default_rng(22)
    loads = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])  # kgf
    HV = 220.0  # true hardness
    # d (mm) from HV = 1.8544 * F / d^2  -> d = sqrt(1.8544 * F / HV)
    d_mm = np.sqrt(1.8544 * loads / HV)
    # Add measurement noise
    d_mm *= rng.normal(1.0, 0.015, size=loads.size)
    _save("hardness_vickers",
          pd.DataFrame({"load_kgf": loads,
                        "diagonal_mm": d_mm.round(5)}),
          header="Vickers indentation, true HV~220. Synthetic.")


def gen_nanoindentation() -> None:
    """Load-displacement curve from nanoindentation: loading + holding + unloading."""
    rng = np.random.default_rng(23)
    n_load = 200
    n_hold = 60
    n_unload = 200

    h_load = np.linspace(0, 250, n_load)        # nm
    P_load = 0.04 * h_load ** 1.5               # mN, Oliver-Pharr-ish
    P_hold = np.full(n_hold, P_load[-1])
    h_hold = np.linspace(h_load[-1], h_load[-1] + 12, n_hold)
    h_unload = np.linspace(h_hold[-1], 110, n_unload)
    # Unloading slope steeper than loading (elastic recovery)
    P_unload = P_hold[-1] * ((h_unload - 110) / (h_unload[0] - 110)) ** 1.4

    h = np.concatenate([h_load, h_hold, h_unload])
    P = np.concatenate([P_load, P_hold, P_unload])
    P += rng.normal(0, 0.05, size=h.size)
    _save("nanoindentation_loaddisp",
          pd.DataFrame({"depth_nm": h.round(3),
                        "load_mN": P.round(4)}),
          header="Nanoindentation load-displacement (load-hold-unload). Synthetic.")


def gen_thermal_conductivity() -> None:
    """Thermal conductivity vs temperature (Pyrex-like: weak T-dependence)."""
    rng = np.random.default_rng(24)
    T = np.arange(50, 600, 10.0)  # K
    # k(T) = k0 + a*T - b*T^2 (very mild)
    k = 1.05 + 0.0008 * T - 6e-7 * T ** 2
    k += rng.normal(0, 0.012, size=T.size)
    _save("thermal_conductivity_pyrex",
          pd.DataFrame({"temperature_K": T,
                        "k_W_per_mK": k.round(4)}),
          header="Thermal conductivity of Pyrex-like glass vs T. Synthetic.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Generating sample datasets in {OUT}")
    # Structural
    gen_xrd()
    gen_saxs()
    # Thermal
    gen_dsc()
    gen_tga()
    # Mechanical
    gen_stress_strain()
    gen_dma()
    gen_hardness()
    gen_nanoindentation()
    # Spectroscopy
    gen_ftir()
    gen_raman()
    gen_uvvis()
    gen_nmr()
    gen_xps()
    gen_eds()
    # Electrical
    gen_impedance()
    gen_iv_diode()
    gen_cv()
    # Dielectric / piezoelectric / magnetic
    gen_dielectric()
    gen_piezo_pe_loop()
    gen_magnetometry()
    # Microscopy / surface
    gen_afm_profile()
    gen_bet()
    # Chromatography / mass spec
    gen_hplc()
    gen_mass_spec()
    # Thermal transport
    gen_thermal_conductivity()
    print("Done.")


if __name__ == "__main__":
    main()
