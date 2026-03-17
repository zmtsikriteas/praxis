---
name: praxis
description: Scientific data analysis and publication-quality plotting. Natural-language interface for characterisation techniques — from raw data to journal-ready figures.
---

# Praxis — Scientific Data Analysis & Plotting

Praxis (from Greek praxis — practice, action) is a scientific data analysis skill that transforms raw lab data into publication-quality figures and analysis results.

## Commands

### Plotting & Visualisation
- `/praxis:plot` — Create any plot from data. Auto-detects best plot type, or specify explicitly.
  - Example: `/praxis:plot my_data.csv as scatter with error bars`
  - Example: `/praxis:plot xrd_scan.xy line plot, Cu Ka, 2theta range 10-80`
- `/praxis:compare` — Overlay multiple datasets for comparison.
  - Example: `/praxis:compare sample_A.csv sample_B.csv sample_C.csv as line plot`
- `/praxis:style` — Set journal style for all subsequent plots.
  - Example: `/praxis:style nature` or `/praxis:style acs`
- `/praxis:export` — Export figures at publication specifications.
  - Example: `/praxis:export last_figure as svg 300dpi`
- `/praxis:multipanel` — Create multi-panel figures with (a), (b), (c) labels.
  - Example: `/praxis:multipanel 2x2 grid from these four datasets`

### Data Processing
- `/praxis:fit` — Curve fitting. Auto-suggests model or specify.
  - Example: `/praxis:fit gaussian to peak at 28 deg`
  - Example: `/praxis:fit linear to 200-400 C region`
  - Example: `/praxis:fit "a * exp(-b * x) + c" custom equation`
- `/praxis:peaks` — Peak detection, fitting, deconvolution.
  - Example: `/praxis:peaks find all peaks above 5% intensity`
- `/praxis:baseline` — Baseline correction and subtraction.
  - Example: `/praxis:baseline polynomial order 3`
  - Example: `/praxis:baseline ALS`
- `/praxis:fft` — FFT, power spectrum, filtering.
  - Example: `/praxis:fft low-pass filter at 100Hz`
  - Example: `/praxis:fft show power spectrum of signal.csv`
- `/praxis:smooth` — Data smoothing and noise reduction.
  - Example: `/praxis:smooth savgol window 11 order 3`
  - Example: `/praxis:smooth gaussian sigma 3`
- `/praxis:stats` — Statistical analysis and error propagation.
  - Example: `/praxis:stats compare groups A B C with ANOVA`
  - Example: `/praxis:stats descriptive on column B`
  - Example: `/praxis:stats normality test on data.csv`
- `/praxis:interpolate` — Interpolation, resampling, derivatives, integration.
  - Example: `/praxis:interpolate cubic spline to 500 points`
  - Example: `/praxis:interpolate derivative of column B`
- `/praxis:normalise` — Data normalisation.
  - Example: `/praxis:normalise min-max` or `/praxis:normalise area`
- `/praxis:batch` — Process multiple files with same analysis pipeline.
  - Example: `/praxis:batch all .csv in ./data/ — plot and extract peak positions`
- `/praxis:template` — Save/load analysis templates (pipeline configs).
  - Example: `/praxis:template save as "xrd_standard"`
  - Example: `/praxis:template load "xrd_standard" on new_data.csv`
- `/praxis:report` — Auto-generate analysis summary report.
  - Example: `/praxis:report generate for all analyses in this session`

### Technique-Specific Analysis
- `/praxis:xrd` — XRD analysis: phase ID, Scherrer size, Williamson-Hall, d-spacing.
- `/praxis:dsc` — DSC/TGA analysis: Tg, Tm, enthalpy, crystallinity, DTG, decomposition.
- `/praxis:impedance` — EIS: Nyquist, Bode, equivalent circuit fitting, Arrhenius.
- `/praxis:mechanical` — Stress-strain (modulus, yield, UTS, toughness), DMA (Tg, tan delta).
- `/praxis:spectro` — FTIR/Raman/UV-Vis: peak assignment, Tauc plot, Beer-Lambert, ATR correction.
- `/praxis:xps` — XPS: survey scan, high-res fitting, Shirley background, atomic %, BE calibration.
- `/praxis:sem` — SEM/EDS: grain size, porosity, elemental composition, line scans.
- `/praxis:afm` — AFM: roughness (Ra, Rq, Rz), height profiles, 2D surface analysis.
- `/praxis:piezo` — Piezoelectric: d33, P-E loops, S-E butterfly, resonance (kp, kt, Qm).
- `/praxis:dielectric` — Permittivity, loss tangent, Cole-Cole, Curie-Weiss, AC conductivity.
- `/praxis:thermal` — Thermal conductivity: laser flash, steady-state, k vs T.

### Help
- `/praxis:help` — Show all commands with examples.

## How It Works

1. **Load data** — auto-detects file format (CSV, Excel, TXT, .xy, JSON, JCAMP-DX, HDF5, .mat), headers, units, delimiters.
2. **Analyse** — technique-aware processing with sensible defaults.
3. **Plot** — publication-ready figures using journal-specific styles. Supports line, scatter, bar, errorbar, histogram, box, violin, contour, heatmap, polar, waterfall, ternary, Smith chart, broken axis, inset, multi-panel.
4. **Export** — PNG (300/600 dpi), SVG, PDF, EPS, TIFF with metadata sidecar.
5. **Report** — auto-generate markdown summary of analyses performed.

## Design Principles

- **Natural language first**: describe what you want in plain English.
- **Auto-detection**: infers data type, columns, plot type, and analysis method.
- **Publication-ready by default**: correct fonts, axis labels, units, colourblind-safe colours (Okabe-Ito).
- **Technique-aware**: knows what analysis makes sense for each data type.
- **Non-destructive**: never modifies original files. Outputs to separate directory.
- **Reproducible**: every figure includes metadata sidecar for exact recreation.
- **Batch-friendly**: same analysis across multiple files trivially.
- **Templatable**: save analysis pipelines, replay on new data.

## Trigger Patterns

Use this skill when:
- User has scientific/lab data and wants to plot or analyse it
- User mentions characterisation techniques (XRD, DSC, TGA, EIS, FTIR, Raman, XPS, SEM, AFM, DMA, etc.)
- User wants publication-quality figures or journal-formatted plots
- User asks to fit curves, find peaks, correct baselines, or do FFT on experimental data
- User mentions journal formatting (Nature, Science, ACS, Elsevier, etc.)
- User has voltage, temperature, force, pressure, current time-series data
- User wants statistical analysis (t-test, ANOVA, regression, normality)
- User wants to batch-process multiple data files
- User says "praxis" or uses any `/praxis:*` command

## Scripts Location

All Python modules live in `~/Documents/Praxis/scripts/`. Import with:
```python
import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/Praxis/scripts"))
from core.loader import load_data
from core.plotter import plot_data
```

## Journal Styles Available

nature, science, acs, elsevier, wiley, rsc, springer, ieee, mdpi

## Supported Data Formats

CSV, TSV, TXT, Excel (.xlsx/.xls), JSON, .xy, .dat, .asc, .spe, .jdx/.dx (JCAMP-DX), HDF5 (.h5/.hdf5), MATLAB (.mat), .brml (Bruker XRD), .dta (Gamry), .mpr (Bio-Logic), clipboard
