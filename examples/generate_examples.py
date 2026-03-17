"""Generate example figures for the README."""

import sys
import os

scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
sys.path.insert(0, scripts_dir)
# Also add parent so 'scripts.x.y' imports work from technique modules
sys.path.insert(0, os.path.join(scripts_dir, ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from core.loader import load_data
from core.plotter import plot_data, overlay_plots, plot_contour, create_subplots
from core.exporter import export_figure
from core.utils import apply_style

sample = os.path.join(os.path.dirname(__file__), "..", "tests", "sample_data")
out = os.path.dirname(__file__)


# 1. XRD with labelled peaks (Nature style)
apply_style("nature")
df = load_data(os.path.join(sample, "xrd_sample.csv"))
from techniques.xrd import analyse_xrd
xrd = analyse_xrd(df.iloc[:, 0].values, df.iloc[:, 1].values, wavelength="Cu_Ka")

fig, ax = plot_data(
    df.iloc[:, 0], df.iloc[:, 1], kind="line", colour="#0072B2",
    xlabel="2theta (deg)", ylabel="Intensity (a.u.)",
    title="XRD Pattern (Cu Ka)",
)
for p in xrd.peaks:
    ax.annotate(
        f"{p.two_theta:.1f} deg\nd={p.d_spacing:.2f} A",
        xy=(p.two_theta, p.intensity),
        xytext=(0, 15), textcoords="offset points",
        ha="center", fontsize=5,
        arrowprops=dict(arrowstyle="->", lw=0.5),
    )
fig.tight_layout()
fig.savefig(os.path.join(out, "xrd_nature.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("1. XRD done")


# 2. Stress-strain with annotations (Elsevier style)
apply_style("elsevier")
from techniques.mechanical import analyse_tensile
ss = load_data(os.path.join(sample, "stress_strain_sample.csv"))
strain, stress = ss.iloc[:, 0].values, ss.iloc[:, 1].values
results = analyse_tensile(strain, stress)

fig, ax = plot_data(
    strain, stress, kind="line", colour="#4477AA", linewidth=1.5,
    xlabel="Strain (%)", ylabel="Stress (MPa)", title="Tensile Test",
)
if results.youngs_modulus:
    s_lin = np.linspace(0, 1.0, 50)
    ax.plot(s_lin, results.youngs_modulus * s_lin / 100, "--",
            color="#EE6677", linewidth=0.8, label=f"E = {results.youngs_modulus:.0f} MPa")
if results.uts:
    ax.plot(strain[np.argmax(stress)], results.uts, "v",
            color="#228833", markersize=6, label=f"UTS = {results.uts:.0f} MPa")
ax.legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(out, "stress_strain_elsevier.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("2. Stress-strain done")


# 3. Signal processing 3-panel (IEEE style)
apply_style("ieee")
from analysis.fft import compute_fft, filter_signal
np.random.seed(42)
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.random.randn(1000)

fig, axes = create_subplots(3, 1, figsize=(3.5, 6), sharex=False)
axes[0].plot(t, signal, linewidth=0.5, color="#4477AA")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Raw Signal", fontsize=8)

fft_result = compute_fft(signal, sample_rate=1000)
axes[1].plot(fft_result.freq[:200], fft_result.amplitude[:200], linewidth=0.8, color="#EE6677")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("FFT Spectrum", fontsize=8)

filtered = filter_signal(signal, "lowpass", 30, sample_rate=1000)
axes[2].plot(t, signal, linewidth=0.3, color="#BBBBBB", label="Raw")
axes[2].plot(t, filtered, linewidth=0.8, color="#228833", label="Low-pass 30 Hz")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Amplitude")
axes[2].set_title("Filtered Signal", fontsize=8)
axes[2].legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(out, "signal_processing_ieee.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("3. Signal processing done")


# 4. Gaussian fit with confidence band (RSC style)
apply_style("rsc")
from analysis.fitting import fit_curve
np.random.seed(7)
x = np.linspace(-5, 5, 200)
y = 8 * np.exp(-(x ** 2) / (2 * 1.2 ** 2)) + 1.5 + np.random.normal(0, 0.3, 200)

fit = fit_curve(x, y, model="gaussian")
x_fine, y_fit = fit.eval_fine()
x_cb, y_lo, y_hi = fit.confidence_band(sigma=2)

fig, ax = plot_data(
    x, y, kind="scatter", colour="#56B4E9", marker="o", s=8, alpha=0.5,
    xlabel="x", ylabel="y", title="Gaussian Fit with 95% CI", label="Data",
)
ax.plot(x_fine, y_fit, "-", color="#D55E00", linewidth=1.2, label="Fit")
ax.fill_between(x_cb, y_lo, y_hi, alpha=0.2, color="#D55E00", label="95% CI")
ax.legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(out, "gaussian_fit_rsc.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("4. Curve fitting done")


# 5. Multi-panel figure (Science style)
apply_style("science")
fig, axes = create_subplots(2, 2, figsize=(7.09, 5.5))
np.random.seed(1)

x_sc = np.linspace(0, 10, 30)
y_sc = 2.3 * x_sc + 5 + np.random.normal(0, 2, 30)
axes[0, 0].scatter(x_sc, y_sc, s=15, color="#0072B2", zorder=3)
coeffs = np.polyfit(x_sc, y_sc, 1)
axes[0, 0].plot(x_sc, np.polyval(coeffs, x_sc), "--", color="#D55E00", linewidth=1)
axes[0, 0].set_xlabel("Concentration (mg/L)")
axes[0, 0].set_ylabel("Response")
axes[0, 0].set_title("(a) Calibration", fontsize=7)

data_groups = [np.random.normal(50, 5, 50), np.random.normal(55, 8, 50), np.random.normal(48, 6, 50)]
bp = axes[0, 1].boxplot(data_groups, labels=["Control", "Treat A", "Treat B"],
                         patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], ["#56B4E9", "#E69F00", "#009E73"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
axes[0, 1].set_ylabel("Measurement")
axes[0, 1].set_title("(b) Group Comparison", fontsize=7)

tv = np.linspace(0, 2, 500)
axes[1, 0].plot(tv, np.sin(2 * np.pi * 3 * tv) * np.exp(-tv) + np.random.normal(0, 0.05, 500),
                linewidth=0.6, color="#CC79A7")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Voltage (V)")
axes[1, 0].set_title("(c) Sensor Output", fontsize=7)

axes[1, 1].hist(np.random.lognormal(3, 0.5, 500), bins=30, color="#56B4E9",
                edgecolor="white", linewidth=0.3)
axes[1, 1].set_xlabel("Particle Size (nm)")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title("(d) Size Distribution", fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(out, "multipanel_science.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("5. Multi-panel done")


# 6. Contour heatmap (Springer style)
apply_style("springer")
xc = np.linspace(-3, 3, 80)
yc = np.linspace(-3, 3, 80)
X, Y = np.meshgrid(xc, yc)
Z = np.sin(X) * np.cos(Y) * np.exp(-(X ** 2 + Y ** 2) / 8)

fig, ax = plot_contour(
    X, Y, Z, kind="filled", levels=25, cmap="RdBu_r",
    xlabel="x (mm)", ylabel="y (mm)", clabel="Amplitude",
    title="2D Field Map",
)
fig.savefig(os.path.join(out, "contour_springer.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("6. Contour done")


print("\nAll examples generated in examples/")
