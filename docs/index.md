---
hide:
  - navigation
---

# Praxis

![Praxis](banner.jpg)

**Scientific data analysis and publication-quality plotting for 50+ characterisation techniques.**

Praxis (from Greek *praxis*, practice, action) gives researchers a natural-language interface for every characterisation technique they encounter in the lab. Load raw data, run technique-aware analysis, and produce journal-ready figures.

---

## Install

```bash
pip install praxis-sci
```

Or from a clone:

```bash
git clone https://github.com/zmtsikriteas/praxis.git
cd praxis
pip install -e .
```

## Try it without your own data

25 built-in sample datasets ship with the package, one per technique:

```python
from praxis.core.loader import load_sample, list_samples
from praxis.techniques.xrd import analyse_xrd

print(list_samples())                         # see available samples
df = load_sample("xrd")                       # Si powder pattern
results = analyse_xrd(df["two_theta_deg"], df["intensity"],
                      wavelength="Cu_Ka")
```

## Explore

<div class="grid cards" markdown>

-   :material-book-open-page-variant: **[Cookbook](cookbook.md)**

    50+ worked examples, one per technique: data format, loading, analysis, plotting.

-   :material-workflow: **[Workflows](workflows.md)**

    12 complete multi-step pipelines from raw data to publication figure.

-   :material-chart-line: **[Plot types](plot-types.md)**

    All 15+ plot types with runnable code: line, scatter, heatmap, contour, ternary, Smith chart, multi-panel.

-   :material-atom: **[Techniques](techniques.md)**

    Quick reference for every supported technique with expected data columns.

-   :material-newspaper: **[Journal styles](journal-styles.md)**

    Formatting specs for Nature, Science, ACS, Elsevier, Wiley, RSC, Springer, IEEE, MDPI.

-   :material-palette: **[Colour palettes](colour-palettes.md)**

    Okabe-Ito, Tol, and uchu palettes with hex codes. All colourblind-safe.

</div>

---

## What Praxis does

- **Loads anything.** CSV, Excel, JSON, JCAMP-DX, HDF5, MATLAB, Bruker XRD, Gamry, and more -- all auto-detected, including European decimal commas and UTF-16 files.
- **21 technique modules.** XRD, SAXS, DSC/TGA, mechanical, FTIR/Raman/UV-Vis, XPS, NMR, SEM/EDS, AFM, impedance, I-V, C-V, dielectric, piezoelectric, magnetometry, BET, chromatography, mass spec, nanoindentation, hardness, thermal conductivity.
- **Publication-ready plots.** 9 journal styles matching column widths, fonts, and DPI requirements. Colourblind-safe palettes by default.
- **Reproducible.** Every exported figure carries a `.meta.json` sidecar with the parameters used to create it.
- **Batch-friendly.** Process hundreds of files with a single pipeline.

## Cite

See [CITATION.cff](https://github.com/zmtsikriteas/praxis/blob/main/CITATION.cff) or use GitHub's "Cite this repository" button.

## License

MIT. See [LICENSE](https://github.com/zmtsikriteas/praxis/blob/main/LICENSE).
