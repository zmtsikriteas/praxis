# Changelog

All notable changes to Praxis are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and Praxis uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- Continuous integration: pytest runs on Python 3.10, 3.11, and 3.12 on every push.
- `.gitattributes` to normalise line endings to LF across platforms.
- `CITATION.cff` so the repository can be cited from GitHub.
- This changelog.

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
