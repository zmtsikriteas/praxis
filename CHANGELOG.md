# Changelog

All notable changes to Praxis are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and Praxis uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- Pip packaging via `pyproject.toml`. Praxis is now installable with
  `pip install praxis-sci` (or `pip install -e .` from a clone).
- Optional dependency groups (`hdf5`, `tiff`, `pdf`, `encoding`, `all`, `test`).
- Continuous integration: pytest runs on Python 3.10, 3.11, and 3.12 on every push.
- `.gitattributes` to normalise line endings to LF across platforms.
- `CITATION.cff` so the repository can be cited from GitHub.
- This changelog.
- 25 built-in sample datasets (one per technique) at `praxis/sample_data/`,
  shipped with the package. Helper functions:
  `load_sample('xrd')` and `list_samples()`.
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
- Source folder renamed `scripts/` -> `praxis/`. All internal imports now use
  the `praxis.` prefix (e.g. `from praxis.core.loader import load_data`).
- Journal styles moved from `assets/styles/` into the package at
  `praxis/styles/` so they ship with a pip install.
- CI installs the package via `pip install -e .[test]` instead of
  `requirements.txt`. The latter is kept for users who prefer it.

## [1.0.0] - 2026-03-17

### Added
- Core modules: universal data loader (16 formats), plotter (15+ plot types), exporter (PNG, SVG, PDF, EPS, TIFF with metadata sidecars), and shared utilities.
- Analysis modules: curve fitting, peak detection and deconvolution, baseline correction (polynomial, ALS, Shirley, SNIP), smoothing (Savitzky-Golay, Gaussian, median, Whittaker), FFT, statistics, interpolation, normalisation, analysis templates, and report generation.
- 21 technique modules covering 50+ characterisation methods including XRD, SAXS, DSC/TGA, mechanical (tensile, compression, DMA), spectroscopy (FTIR, Raman, UV-Vis), XPS, dielectric, piezoelectric, SEM/EDS, AFM, thermal conductivity, I-V and C-V curves, magnetometry, BET, NMR, chromatography, mass spectrometry, nanoindentation, and hardness.
- Batch processing across multiple files with shared analysis pipelines.
- Nine journal styles (Nature, Science, ACS, Elsevier, Wiley, RSC, Springer, IEEE, MDPI).
- Three colourblind-safe palette families (Okabe-Ito, Tol, uchu).
- 95 tests covering core, analysis, and technique modules.
- Six reference documents (cookbook, workflows, plot types, techniques, journal styles, colour palettes).
- Six example figures rendered in the README.

[Unreleased]: https://github.com/zmtsikriteas/praxis/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/zmtsikriteas/praxis/releases/tag/v1.0.0
