"""Generate example figures shown in the README gallery."""

import sys
import os

# Add project root so 'praxis.x.y' imports work when run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from praxis.core.loader import load_sample
from praxis.core.plotter import plot_data, plot_contour, create_subplots
from praxis.core.utils import apply_style

OUT = os.path.dirname(os.path.abspath(__file__))


def _save(name: str, fig) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  wrote examples/{name}")


# 1. XRD with labelled peaks (Nature style) -----------------------------------
apply_style("nature")
from praxis.techniques.xrd import analyse_xrd
df = load_sample("xrd")
two_th = df["two_theta_deg"].values
inten = df["intensity"].values
xrd = analyse_xrd(two_th, inten, wavelength="Cu_Ka")

fig, ax = plot_data(two_th, inten, kind="line", colour="#0072B2",
                    xlabel=r"$2\theta$ (deg)", ylabel="Intensity (a.u.)",
                    title="XRD Pattern (Cu K$\\alpha$)")
for p in xrd.peaks[:6]:
    ax.annotate(f"{p.two_theta:.1f}$^\\circ$\nd={p.d_spacing:.2f} A",
                xy=(p.two_theta, p.intensity),
                xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=5,
                arrowprops=dict(arrowstyle="->", lw=0.4))
_save("xrd_nature.png", fig)


# 2. Stress-strain with annotations (Elsevier style) --------------------------
apply_style("elsevier")
from praxis.techniques.mechanical import analyse_tensile
ss = load_sample("stress_strain")
strain, stress = ss.iloc[:, 0].values, ss.iloc[:, 1].values
res = analyse_tensile(strain, stress)

fig, ax = plot_data(strain, stress, kind="line", colour="#4477AA",
                    linewidth=1.5, xlabel="Strain", ylabel="Stress (MPa)",
                    title="Tensile Test")
if res.youngs_modulus:
    s_lin = np.linspace(0, 0.005, 50)
    ax.plot(s_lin, res.youngs_modulus * s_lin, "--",
            color="#EE6677", linewidth=0.8,
            label=f"E = {res.youngs_modulus / 1000:.0f} GPa")
if res.uts:
    ax.plot(strain[np.argmax(stress)], res.uts, "v",
            color="#228833", markersize=6,
            label=f"UTS = {res.uts:.0f} MPa")
ax.legend(frameon=False, fontsize=6)
_save("stress_strain_elsevier.png", fig)


# 3. Signal processing 3-panel (IEEE style) -----------------------------------
apply_style("ieee")
from praxis.analysis.fft import compute_fft, filter_signal
np.random.seed(42)
t = np.linspace(0, 1, 1000)
sig = (np.sin(2 * np.pi * 10 * t)
       + 0.5 * np.sin(2 * np.pi * 50 * t)
       + 0.3 * np.random.randn(1000))

fig, axes = create_subplots(3, 1, figsize=(3.5, 6), sharex=False)
axes[0].plot(t, sig, linewidth=0.5, color="#4477AA")
axes[0].set(xlabel="Time (s)", ylabel="Amplitude")
axes[0].set_title("Raw Signal", fontsize=8)

fft = compute_fft(sig, sample_rate=1000)
axes[1].plot(fft.freq[:200], fft.amplitude[:200], linewidth=0.8, color="#EE6677")
axes[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude")
axes[1].set_title("FFT Spectrum", fontsize=8)

filtered = filter_signal(sig, "lowpass", 30, sample_rate=1000)
axes[2].plot(t, sig, linewidth=0.3, color="#BBBBBB", label="Raw")
axes[2].plot(t, filtered, linewidth=0.8, color="#228833", label="Low-pass 30 Hz")
axes[2].set(xlabel="Time (s)", ylabel="Amplitude")
axes[2].set_title("Filtered Signal", fontsize=8)
axes[2].legend(frameon=False, fontsize=6)
_save("signal_processing_ieee.png", fig)


# 4. Gaussian fit with confidence band (RSC style) ----------------------------
apply_style("rsc")
from praxis.analysis.fitting import fit_curve
np.random.seed(7)
x = np.linspace(-5, 5, 200)
y = 8 * np.exp(-(x ** 2) / (2 * 1.2 ** 2)) + 1.5 + np.random.normal(0, 0.3, 200)

fit = fit_curve(x, y, model="gaussian")
x_fine, y_fit = fit.eval_fine()
x_cb, y_lo, y_hi = fit.confidence_band(sigma=2)

fig, ax = plot_data(x, y, kind="scatter", colour="#56B4E9", marker="o",
                    s=8, alpha=0.5, xlabel="x", ylabel="y",
                    title="Gaussian Fit with 95% CI", label="Data")
ax.plot(x_fine, y_fit, "-", color="#D55E00", linewidth=1.2, label="Fit")
ax.fill_between(x_cb, y_lo, y_hi, alpha=0.2, color="#D55E00", label="95% CI")
ax.legend(frameon=False, fontsize=6)
_save("gaussian_fit_rsc.png", fig)


# 5. Multi-panel figure (Science style) ---------------------------------------
apply_style("science")
fig, axes = create_subplots(2, 2, figsize=(7.09, 5.5))
np.random.seed(1)

x_sc = np.linspace(0, 10, 30)
y_sc = 2.3 * x_sc + 5 + np.random.normal(0, 2, 30)
axes[0, 0].scatter(x_sc, y_sc, s=15, color="#0072B2", zorder=3)
axes[0, 0].plot(x_sc, np.polyval(np.polyfit(x_sc, y_sc, 1), x_sc),
                "--", color="#D55E00", linewidth=1)
axes[0, 0].set(xlabel="Concentration (mg/L)", ylabel="Response")
axes[0, 0].set_title("(a) Calibration", fontsize=7)

groups = [np.random.normal(50, 5, 50), np.random.normal(55, 8, 50),
          np.random.normal(48, 6, 50)]
bp = axes[0, 1].boxplot(groups, labels=["Control", "Treat A", "Treat B"],
                         patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], ["#56B4E9", "#E69F00", "#009E73"]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
axes[0, 1].set_ylabel("Measurement")
axes[0, 1].set_title("(b) Group Comparison", fontsize=7)

tv = np.linspace(0, 2, 500)
axes[1, 0].plot(tv, np.sin(2 * np.pi * 3 * tv) * np.exp(-tv)
                + np.random.normal(0, 0.05, 500),
                linewidth=0.6, color="#CC79A7")
axes[1, 0].set(xlabel="Time (s)", ylabel="Voltage (V)")
axes[1, 0].set_title("(c) Sensor Output", fontsize=7)

axes[1, 1].hist(np.random.lognormal(3, 0.5, 500), bins=30,
                color="#56B4E9", edgecolor="white", linewidth=0.3)
axes[1, 1].set(xlabel="Particle Size (nm)", ylabel="Count")
axes[1, 1].set_title("(d) Size Distribution", fontsize=7)
_save("multipanel_science.png", fig)


# 6. Contour heatmap (Springer style) -----------------------------------------
apply_style("springer")
xc = np.linspace(-3, 3, 80)
yc = np.linspace(-3, 3, 80)
X, Y = np.meshgrid(xc, yc)
Z = np.sin(X) * np.cos(Y) * np.exp(-(X ** 2 + Y ** 2) / 8)

fig, _ = plot_contour(X, Y, Z, kind="filled", levels=25, cmap="RdBu_r",
                      xlabel="x (mm)", ylabel="y (mm)", clabel="Amplitude",
                      title="2D Field Map")
_save("contour_springer.png", fig)


# 7. EIS Nyquist (ACS style) --------------------------------------------------
apply_style("acs")
df = load_sample("impedance_rc")
zr = df["z_real_ohm"].values
zi = df["z_imag_ohm"].values

fig, ax = plot_data(zr, -zi, kind="scatter", colour="#0072B2", s=14,
                    xlabel=r"Z$'$ ($\Omega$)", ylabel=r"-Z$''$ ($\Omega$)",
                    title="EIS Nyquist Plot")
ax.plot(zr, -zi, "-", color="#0072B2", linewidth=0.6, alpha=0.4)
ax.set_aspect("equal", adjustable="datalim")
_save("eis_acs.png", fig)


# 8. DSC trace with Tg / Tm (Wiley style) -------------------------------------
apply_style("wiley")
df = load_sample("dsc_polymer")
T = df["temperature_C"].values
hf = df["heat_flow_mW_per_mg"].values

fig, ax = plot_data(T, hf, kind="line", colour="#D55E00", linewidth=1.3,
                    xlabel="Temperature ($^\\circ$C)",
                    ylabel="Heat Flow (mW/mg)",
                    title="DSC of Semi-Crystalline Polymer")
# Annotate the three thermal events
for label, x_pos, y_off in [("$T_g$", 80, 0.06), ("$T_c$", 135, 0.10),
                             ("$T_m$", 220, -0.10)]:
    y_val = hf[np.argmin(abs(T - x_pos))]
    ax.annotate(label, xy=(x_pos, y_val),
                xytext=(0, 22 if y_off > 0 else -22),
                textcoords="offset points", ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.5, color="grey"))
ax.axhline(0, color="grey", linewidth=0.4, linestyle=":")
_save("dsc_wiley.png", fig)


# 9. M-H ferromagnetic loop (MDPI style) --------------------------------------
apply_style("mdpi")
df = load_sample("mh_loop")
H = df["H_field_T"].values
M = df["moment_emu_per_g"].values

fig, ax = plot_data(H, M, kind="line", colour="#9F2B68", linewidth=1.0,
                    xlabel="$\\mu_0 H$ (T)", ylabel="M (emu/g)",
                    title="Ferromagnetic M-H Loop")
ax.axhline(0, color="grey", linewidth=0.4)
ax.axvline(0, color="grey", linewidth=0.4)
_save("mh_loop_mdpi.png", fig)


# 10. Raman of silicon with peak label (Nature style) -------------------------
apply_style("nature")
df = load_sample("raman_silicon")
shift = df["raman_shift_cm-1"].values
intens = df["intensity"].values

fig, ax = plot_data(shift, intens, kind="line", colour="#117733",
                    linewidth=1.0, xlabel="Raman shift (cm$^{-1}$)",
                    ylabel="Intensity (counts)",
                    title="Raman Spectrum of Silicon")
peak_x = shift[np.argmax(intens)]
peak_y = intens.max()
ax.annotate(f"{peak_x:.1f} cm$^{{-1}}$", xy=(peak_x, peak_y),
            xytext=(20, 0), textcoords="offset points",
            fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.4))
_save("raman_nature.png", fig)


# 11. UV-Vis plasmon (Wiley style) --------------------------------------------
apply_style("wiley")
df = load_sample("uvvis_au_np")
wl = df["wavelength_nm"].values
A = df["absorbance"].values

fig, ax = plot_data(wl, A, kind="line", colour="#CC3311", linewidth=1.3,
                    xlabel="Wavelength (nm)", ylabel="Absorbance",
                    title="UV-Vis: Au Nanoparticle Plasmon")
plasmon = wl[np.argmax(A)]
ax.axvline(plasmon, color="grey", linewidth=0.5, linestyle="--")
ax.text(plasmon + 5, A.max() * 0.9, f"$\\lambda_{{max}}$ = {plasmon:.0f} nm",
        fontsize=7, color="grey")
_save("uvvis_wiley.png", fig)


# 12. Cyclic voltammetry (IEEE style) -----------------------------------------
apply_style("ieee")
df = load_sample("cv_redox")
V = df["potential_V"].values
I = df["current_A"].values * 1e6  # uA

fig, ax = plot_data(V, I, kind="line", colour="#332288", linewidth=1.0,
                    xlabel="Potential vs Ref (V)", ylabel="Current ($\\mu$A)",
                    title="Cyclic Voltammetry")
ax.axhline(0, color="grey", linewidth=0.4)
_save("cv_ieee.png", fig)


print("\nAll example figures generated in examples/")
