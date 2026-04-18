# Contributing to Praxis

Thanks for considering a contribution. This guide covers the practical
mechanics; if you spot something that's wrong or unclear, open an issue
or PR for the docs themselves.

## Quick start

```bash
git clone https://github.com/zmtsikriteas/praxis.git
cd praxis
pip install -e .[test]
python -m pytest tests/ -v
```

That installs Praxis in editable mode plus pytest, and runs all 130+
tests. They should pass on Python 3.10, 3.11, and 3.12.

## What we welcome

- **New technique modules** -- if Praxis doesn't yet cover a technique
  you use regularly, a focused PR adding one is the highest-leverage
  contribution.
- **New file-format support** -- vendor files (Renishaw `.wdf`,
  PerkinElmer `.sp`, Bruker NMR folder, Igor `.ibw`, etc.) are real
  adoption blockers. Adding one is usually 30-100 lines.
- **Sample datasets** -- realistic synthetic data for techniques that
  don't yet have one in `praxis/sample_data/`.
- **Cookbook recipes and worked examples**.
- **Bug fixes** with a regression test.

## What we hesitate on

- Refactors of working code without a clear motivation.
- New abstractions for features that don't yet exist.
- Cosmetic-only changes (formatting, spelling) bundled into substantive
  PRs -- they're fine on their own, just not interleaved.
- Breaking changes to the public API without a deprecation path.

## Coding conventions

- **British English** in docs and comments (analyse, colour, behaviour).
- **ASCII only in `print()` output** -- Windows terminals (cp1252) cannot
  render Greek / Unicode characters reliably. Use `theta` or `th` not
  `theta`/`theta`.
- **Non-destructive** -- never modify the user's input data files.
- **Match existing style** -- type hints on public functions, dataclasses
  for results, a `.table()` method that prints a summary.
- **Tests on Python 3.10+** -- keep imports compatible with the lowest
  supported version.

## Adding a new technique

1. Create `praxis/techniques/your_technique.py`. Follow the existing
   pattern: dataclass(es) for results, an `analyse_*` function that takes
   raw arrays, and a `.table()` method that prints a one-screen summary.
2. Add a synthetic sample dataset to `praxis/sample_data/_generate.py`
   and run `python -m praxis.sample_data._generate`. Commit the resulting
   `.csv` so it ships with the package.
3. Add tests in `tests/test_<your_technique>.py` covering the happy path,
   one edge case, and a smoke test using `load_sample("your_technique")`.
4. Add a cookbook entry to `docs/cookbook.md` showing the full
   load -> analyse -> plot flow.
5. Register a `/praxis:your_technique` slash command in `SKILL.md`.

## Adding a new file format

1. Add a `_load_<format>(path)` function to `praxis/core/loader.py` that
   returns a `pd.DataFrame`.
2. Register the extension in the `loaders` dispatch dict inside
   `load_data`.
3. If parsing requires an external library, add it as an optional
   dependency in `pyproject.toml` and import it lazily inside the
   loader so users without the dep aren't blocked.
4. Add a test in `tests/test_loader.py`. For binary vendor formats you
   can monkey-patch the dependency rather than ship a real file
   (see `TestBioLogicMPR` for the pattern).

## Pull request checklist

- `python -m pytest tests/ -v` passes.
- New behaviour has a test.
- Public API additions are reflected in `docs/cookbook.md` and (for new
  techniques) in `SKILL.md`.
- `CHANGELOG.md` has a one-line entry under `[Unreleased]`.

## Reporting issues

Use the templates at
[github.com/zmtsikriteas/praxis/issues/new/choose](https://github.com/zmtsikriteas/praxis/issues/new/choose).
For bug reports, please include:

- Praxis version (`pip show praxis-sci` or commit hash)
- Python version and OS
- A minimal example that reproduces the problem (10-20 lines, with
  data inline if possible -- otherwise a tiny file attached)

## Licence

By contributing, you agree that your contribution will be licensed under
the [MIT licence](LICENSE).
