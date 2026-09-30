# Changelog

All notable changes to Praxis are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and Praxis uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Fixed
Several analysis results in 1.0.0 were numerically wrong. Results computed with the affected functions should be re-run.

- **Tensile toughness** (`analyse_tensile`) was 10^6 times too small; it is now reported correctly in MJ/m3.
- **Nanoindentation** (`analyse_indent`, `batch_indents`) mixed units, so hardness and moduli labelled GPa were not in GPa. New `depth_unit` (default `nm`) and `load_unit` (default `mN`) arguments; results are now true GPa.
- **Piezoelectric d33\*** (`analyse_se_curve`) wrongly depended on thickness and had wrong conversion factors. It is now S_max / E_max in pm/V and needs no thickness; new `strain_unit` argument.
- **DSC enthalpy and crystallinity** (`analyse_dsc`) integrated heat flow over temperature without dividing by the heating rate, and integrated only part of each peak. New `heating_rate` (K/min, required for enthalpy) and `sample_mass` (mg, for heat flow in mW) arguments.
- **Peak FWHM and area** (`find_peaks_auto`) used an arbitrary local window and underestimated both; widths now use `scipy.signal.peak_widths` and areas integrate between the peak bases. Descending x axes (e.g. XPS binding energy) now give positive areas, which also fixes empty XPS survey compositions.
- **Mo Ka wavelength** was 1.7107 A; corrected to 0.71073 A.
- **Hydrogen monoisotopic mass** used the average atomic weight (1.00794); corrected to 1.007825.
- **Paired t-test** removed NaNs from each sample separately, breaking the pairing; incomplete pairs are now dropped together.
- **FFT** amplitudes doubled the DC bin and dropped the Nyquist bin.
- **NMR multiplets** (`predict_multiplicity`) reported each line of a multiplet as a separate multiplet; new `spectrometer_freq_mhz` argument (previously fixed at 400 MHz).
- **EDS** conversions silently used a mass of 1.0 for elements missing from the table (e.g. W). The table is extended and unknown elements now raise an error.
- **BJH** on the adsorption branch discarded all increments.
- **Interpolation**: `smoothing=0` now gives exact spline interpolation, and `fill_value` is honoured by every method.
- **Loader**: text columns (e.g. sample names) are no longer turned into NaN; packed JCAMP-DX data get the correct x spacing, and `(XY..XY)` point tables are parsed as pairs; semicolon-delimited files with signed decimal-comma values load correctly; the encoding fallback now tries charset-normalizer before latin-1.
- **Fitting**: custom-expression models (`fit_curve(x, y, model="a*exp(-b*x)+c")`) did not work at all; weights are filtered alongside NaN observations; `confidence_band` returns NaN bounds with a warning instead of a zero-width band when uncertainties are unavailable.
- **Templates**: steps such as `smooth` that take only `y` now work, and baseline-corrected data are passed on to later steps.
- **DMA** (`analyse_dma`) crashed with `NameError`.
- **Plotting**: `alpha` with `kind="area"`/`"fill_between"` and `colour` with `kind="waterfall"` no longer raise; `kind="box"` with `labels` works on current matplotlib; journal styles now control figure size and line width.
- **XPS**: `background="linear"` works; unknown backgrounds raise instead of silently doing nothing.
- **Batch**: recursive globs no longer overwrite files that share a name (keys are now paths relative to the search directory).
- Short inputs to moving-average smoothing and dQ/dV no longer return longer arrays than the input.
- `np.trapezoid` (NumPy >= 2 only) replaced with `scipy.integrate.trapezoid`, so the declared NumPy >= 1.24 minimum holds.

### Security
- Custom fit expressions and `propagate_error` evaluated strings with `eval`, which allowed arbitrary code execution. Expressions are now checked against a whitelist of arithmetic and maths functions before evaluation.

### Changed
- README: replaced the "30-second example" with a four-step "Using Praxis" walkthrough (load -> analyse -> style+plot -> export).
- Docs: fixed examples that used non-existent functions or arguments (`analyse_eis`, `degree=`, `polyorder=`, `np.trapz`) and the tensile example's strain units.
- Packaging: new `xls` extra (`xlrd`) for legacy Excel files; removed the unused `pdf` extra.
- CI: the PyPI publish workflow now runs the test suite before publishing; a new job tests against the minimum supported dependency versions.

## [1.0.0] - 2026-04-18

First public release on PyPI as `praxis-sci`.

### Added
- Core modules: universal data loader (17+ formats including PANalytical `.xrdml` and Bio-Logic `.mpr`), plotter (15+ plot types), exporter (PNG, SVG, PDF, EPS, TIFF with metadata sidecars), and shared utilities.
- Analysis modules: curve fitting, peak detection and deconvolution, baseline correction (polynomial, ALS, Shirley, SNIP), smoothing (Savitzky-Golay, Gaussian, median, Whittaker), FFT, statistics, interpolation, normalisation, analysis templates, and report generation.
- 22 technique modules covering 50+ characterisation methods including XRD, SAXS, DSC/TGA, mechanical (tensile, compression, DMA), spectroscopy (FTIR, Raman, UV-Vis), XPS, dielectric, piezoelectric, SEM/EDS, AFM, thermal conductivity, I-V and C-V curves, magnetometry, BET, NMR, chromatography, mass spectrometry, nanoindentation, hardness, and battery cycling.
- Batch processing across multiple files with shared analysis pipelines.
- Nine journal styles (Nature, Science, ACS, Elsevier, Wiley, RSC, Springer, IEEE, MDPI).
- Three colourblind-safe palette families (Okabe-Ito, Tol, uchu).
- 134 tests covering core, analysis, and technique modules, running on Python 3.10/3.11/3.12.
- Six reference documents (cookbook, workflows, plot types, techniques, journal styles, colour palettes).
- Pip packaging via `pyproject.toml`. Installable with `pip install praxis-sci` (or `pip install -e .` from a clone).
- Optional dependency groups (`hdf5`, `tiff`, `pdf`, `encoding`, `all`, `test`).
- Continuous integration: pytest runs on Python 3.10, 3.11, and 3.12 on every push.
- Trusted-publisher PyPI release workflow (OIDC) triggered by pushing a `v*` tag.
- `.gitattributes` to normalise line endings to LF across platforms.
- `CITATION.cff` so the repository can be cited from GitHub.
- This changelog.
- 26 built-in sample datasets (one per technique) at `praxis/sample_data/`, shipped with the package. Helper functions: `load_sample('xrd')` and `list_samples()`.
- Loader hardening:
  * Smarter detection of instrument metadata blocks that don't use
    comment characters (e.g. ``Instrument: XRD-7000``).
  * Score-based delimiter detection that correctly handles European
    decimal-comma data (e.g. ``x;y\n1,5;2,7``).
  * BOM-aware encoding detection (UTF-8, UTF-16-LE/BE).
  * Optional `charset-normalizer` fallback for exotic encodings.
  * Helpful error message when `pd.read_csv` fails, including the
    detected delimiter / decimal / skip rows and a copy-pasteable hint.

### Changed
- Source folder renamed `scripts/` -> `praxis/`. All internal imports now use the `praxis.` prefix (e.g. `from praxis.core.loader import load_data`).
- Journal styles moved from `assets/styles/` into the package at `praxis/styles/` so they ship with a pip install.
- CI installs the package via `pip install -e .[test]` instead of `requirements.txt`. The latter is kept for users who prefer it.

[Unreleased]: https://github.com/zmtsikriteas/praxis/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/zmtsikriteas/praxis/releases/tag/v1.0.0
