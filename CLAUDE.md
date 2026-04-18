# Praxis — Project Instructions

## What This Is

Praxis is a scientific data analysis and publication-quality plotting toolkit. It provides a natural-language interface for every characterisation technique researchers encounter in the lab, from raw data files to journal-ready figures.

## Project Structure

```
praxis/
├── SKILL.md                # Skill definition + slash commands
├── pyproject.toml          # Package metadata for pip install
├── praxis/                 # The Python package
│   ├── core/               # loader, plotter, exporter, utils
│   ├── analysis/           # fitting, peaks, baseline, smoothing, fft, statistics, interpolation, normalisation, templates, report
│   ├── techniques/         # 21 technique modules (xrd, impedance, dsc_tga, mechanical, spectroscopy, xps, etc.)
│   ├── styles/             # 9 journal .mplstyle files
│   └── batch/              # batch processing
├── docs/             # Quick-reference docs (cookbook, workflows, plot types, techniques, palettes, journal styles)
└── tests/                  # Tests + sample data
```

## How to Use in Scripts

After `pip install praxis-sci` (or `pip install -e .` from a clone):

```python
from praxis.core.loader import load_data
from praxis.core.plotter import plot_data
from praxis.analysis.fitting import fit_curve
```

## Dependencies

numpy, scipy, pandas, matplotlib, lmfit, openpyxl, uncertainties

Optional: h5py (HDF5 files), pdfplumber (PDF reading)

## Key Design Rules

- **Non-destructive**: never modify original data files
- **Publication-ready by default**: colourblind-safe (Okabe-Ito), correct fonts, proper axis labels
- **Reproducible**: every exported figure gets a `.meta.json` sidecar
- **British English** in all docs and comments
- **No Unicode special characters in print output** — Windows terminals may not handle Greek letters. Use ASCII equivalents (theta -> th, sigma -> sigma, etc.)

## Testing

```bash
pip install pytest
cd praxis
python -m pytest tests/
```

## Journal Styles

9 styles available: nature, science, acs, elsevier, wiley, rsc, springer, ieee, mdpi.
Apply with `apply_style("nature")` before plotting.

## Adding New Techniques

1. Create `praxis/techniques/new_technique.py`
2. Follow the pattern: dataclass for results, analysis function, print summary table
3. Add slash command to SKILL.md
4. Add entry to `docs/techniques.md` and `docs/cookbook.md`
5. Add test with sample data
