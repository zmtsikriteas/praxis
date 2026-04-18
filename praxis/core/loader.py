"""Universal data loader with automatic format detection.

Supports CSV, TSV, TXT, Excel (.xlsx/.xls), JSON, .xy, .dat, .asc, .spe,
.jdx/.dx (JCAMP-DX), HDF5 (.h5/.hdf5), MATLAB (.mat), Bruker XRD (.brml),
PANalytical XRD (.xrdml), Gamry (.dta), Bio-Logic (.mpr), and clipboard paste.
"""

from __future__ import annotations

import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sample(name: str) -> pd.DataFrame:
    """Load a built-in sample dataset by short name.

    Examples
    --------
    >>> df = load_sample("xrd")           # -> xrd_silicon.csv
    >>> df = load_sample("xrd_silicon")   # same file, explicit match
    >>> df = load_sample("dsc")           # -> dsc_polymer.csv

    Raises
    ------
    FileNotFoundError
        If no sample matches ``name``. The message lists available samples.
    """
    sample_dir = Path(__file__).resolve().parent.parent / "sample_data"
    exact = sample_dir / f"{name}.csv"
    if exact.exists():
        return load_data(exact)

    # Prefix match: load_sample("xrd") -> xrd_silicon.csv
    prefix_matches = sorted(sample_dir.glob(f"{name}*.csv"))
    if prefix_matches:
        return load_data(prefix_matches[0])

    available = sorted(p.stem for p in sample_dir.glob("*.csv"))
    raise FileNotFoundError(
        f"No sample dataset '{name}'. Available: {', '.join(available) or 'none'}."
    )


def list_samples() -> list[str]:
    """Return the list of built-in sample dataset names (without extension)."""
    sample_dir = Path(__file__).resolve().parent.parent / "sample_data"
    return sorted(p.stem for p in sample_dir.glob("*.csv"))


def load_data(
    source: Union[str, Path],
    *,
    sheet: Optional[Union[str, int]] = None,
    delimiter: Optional[str] = None,
    header_row: Optional[int] = None,
    skip_rows: Optional[int] = None,
    decimal: Optional[str] = None,
    encoding: Optional[str] = None,
    columns: Optional[list[Union[str, int]]] = None,
) -> pd.DataFrame:
    """Load data from *source* (file path or 'clipboard') into a DataFrame.

    Auto-detects file format, delimiter, header row, comment lines, decimal
    separator, and encoding.  Override any detection with keyword arguments.

    Parameters
    ----------
    source : path or 'clipboard'
        File path or the literal string ``'clipboard'``.
    sheet : str or int, optional
        Sheet name/index for Excel files. Defaults to first sheet.
    delimiter : str, optional
        Column delimiter. Auto-detected if not given.
    header_row : int, optional
        Row index (0-based) containing column headers.
    skip_rows : int, optional
        Number of leading rows to skip (comments, metadata).
    decimal : str, optional
        Decimal separator ('.' or ','). Auto-detected.
    encoding : str, optional
        File encoding. Tries utf-8, latin-1, cp1252 by default.
    columns : list, optional
        Subset of columns to keep (by name or index).

    Returns
    -------
    pd.DataFrame
    """
    if str(source).lower() == "clipboard":
        return _load_clipboard()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    # Dispatch by extension
    loaders = {
        ".csv": _load_delimited,
        ".tsv": _load_delimited,
        ".txt": _load_delimited,
        ".dat": _load_delimited,
        ".asc": _load_delimited,
        ".xy": _load_xy,
        ".xlsx": _load_excel,
        ".xls": _load_excel,
        ".json": _load_json,
        ".jdx": _load_jcamp,
        ".dx": _load_jcamp,
        ".spe": _load_delimited,
        ".h5": _load_hdf5,
        ".hdf5": _load_hdf5,
        ".mat": _load_mat,
        ".brml": _load_brml,
        ".xrdml": _load_xrdml,
        ".dta": _load_gamry_dta,
        ".mpr": _load_biologic_mpr,
    }

    loader = loaders.get(suffix, _load_delimited)

    kwargs: dict[str, Any] = {}
    if loader == _load_delimited:
        kwargs.update(
            delimiter=delimiter,
            header_row=header_row,
            skip_rows=skip_rows,
            decimal=decimal,
            encoding=encoding,
        )
    elif loader == _load_excel:
        kwargs["sheet"] = sheet
    elif loader == _load_xy:
        kwargs.update(encoding=encoding)

    df = loader(path, **kwargs)

    # Column subset
    if columns is not None:
        if all(isinstance(c, int) for c in columns):
            df = df.iloc[:, columns]
        else:
            df = df[columns]

    _report_load(path, df)
    return df


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_delimited(
    path: Path,
    *,
    delimiter: Optional[str] = None,
    header_row: Optional[int] = None,
    skip_rows: Optional[int] = None,
    decimal: Optional[str] = None,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """Load a delimited text file with auto-detection."""
    raw = _read_raw(path, encoding)
    lines = raw.splitlines()

    # Detect and skip comment lines
    if skip_rows is None:
        skip_rows = _detect_comment_rows(lines)

    data_lines = lines[skip_rows:]
    if not data_lines:
        raise ValueError(f"No data lines found in {path}")

    # Detect delimiter
    if delimiter is None:
        delimiter = _detect_delimiter(data_lines)

    # Detect decimal separator
    if decimal is None:
        decimal = _detect_decimal(data_lines, delimiter)

    # Detect header
    if header_row is None:
        has_header = _detect_header(data_lines, delimiter)
        header_row = 0 if has_header else None

    # Adjust header_row relative to skip_rows
    header_arg = header_row if header_row is not None else "infer"
    if header_row is None:
        header_arg = None

    try:
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            sep=delimiter,
            header=header_arg,
            decimal=decimal,
            engine="python",
        )
    except Exception as exc:
        delim_repr = repr(delimiter)
        raise ValueError(
            f"Could not parse {path.name} as delimited text.\n"
            f"  Detected delimiter: {delim_repr}\n"
            f"  Detected decimal:   {decimal!r}\n"
            f"  Skipped header rows: {skip_rows}\n"
            f"  Underlying error: {exc}\n"
            f"Try overriding with explicit kwargs, e.g.:\n"
            f"  load_data('{path.name}', delimiter='\\t', skip_rows=5, decimal=',')"
        ) from exc

    # Generate column names if none
    if header_row is None:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    return _coerce_numeric(df)


def _load_xy(path: Path, *, encoding: Optional[str] = None) -> pd.DataFrame:
    """Load a two-column .xy file (common in XRD)."""
    raw = _read_raw(path, encoding)
    lines = raw.splitlines()
    skip = _detect_comment_rows(lines)
    data_lines = lines[skip:]
    delimiter = _detect_delimiter(data_lines)

    rows = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[\s,;]+", line)
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric data found in {path}")

    arr = np.array(rows)
    return pd.DataFrame({"x": arr[:, 0], "y": arr[:, 1]})


def _load_excel(
    path: Path, *, sheet: Optional[Union[str, int]] = None
) -> pd.DataFrame:
    """Load an Excel file (.xlsx / .xls)."""
    kwargs: dict[str, Any] = {}
    if sheet is not None:
        kwargs["sheet_name"] = sheet
    else:
        kwargs["sheet_name"] = 0

    df = pd.read_excel(path, **kwargs)
    return _coerce_numeric(df)


def _load_json(path: Path) -> pd.DataFrame:
    """Load a JSON file (tabular array or records)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Try 'data' key first, then columns-oriented
        if "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(data)
    else:
        raise ValueError(f"Cannot parse JSON structure in {path}")

    return _coerce_numeric(df)


def _load_jcamp(path: Path) -> pd.DataFrame:
    """Load a JCAMP-DX (.jdx/.dx) spectroscopy file (basic parser)."""
    raw = _read_raw(path)
    x_values: list[float] = []
    y_values: list[float] = []
    in_data = False
    x_factor = 1.0
    y_factor = 1.0

    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("##XFACTOR="):
            x_factor = float(stripped.split("=", 1)[1].strip())
        elif upper.startswith("##YFACTOR="):
            y_factor = float(stripped.split("=", 1)[1].strip())
        elif upper.startswith("##XYDATA=") or upper.startswith("##XYPOINTS="):
            in_data = True
            continue
        elif upper.startswith("##END="):
            in_data = False
            continue
        elif upper.startswith("##"):
            continue

        if in_data and stripped:
            parts = re.split(r"[\s,;]+", stripped)
            try:
                x_val = float(parts[0]) * x_factor
                for yp in parts[1:]:
                    y_val = float(yp) * y_factor
                    x_values.append(x_val)
                    y_values.append(y_val)
                    # Increment x for packed data (approximate)
            except ValueError:
                continue

    if not x_values:
        raise ValueError(f"No data extracted from JCAMP-DX file: {path}")

    # Trim to equal length
    n = min(len(x_values), len(y_values))
    return pd.DataFrame({"x": x_values[:n], "y": y_values[:n]})


def _load_hdf5(path: Path) -> pd.DataFrame:
    """Load the first dataset(s) from an HDF5 file (.h5/.hdf5)."""
    try:
        import h5py
    except ImportError:
        print("[Praxis] h5py is not installed. Install it with: pip install h5py")
        raise ImportError(
            "h5py is required to load HDF5 files. Install with: pip install h5py"
        )

    datasets: dict[str, np.ndarray] = {}

    def _visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            try:
                data = obj[()]
                if isinstance(data, np.ndarray) and data.ndim <= 2:
                    datasets[name] = data
            except Exception:
                pass  # skip unreadable datasets

    with h5py.File(path, "r") as f:
        f.visititems(_visitor)

    if not datasets:
        raise ValueError(f"No readable datasets found in HDF5 file: {path}")

    # If there's one 2-D dataset, use it directly
    for name, arr in datasets.items():
        if arr.ndim == 2:
            df = pd.DataFrame(arr, columns=[f"col_{i}" for i in range(arr.shape[1])])
            return _coerce_numeric(df)

    # Otherwise combine 1-D datasets as columns
    df = pd.DataFrame({name.split("/")[-1]: arr.ravel() for name, arr in datasets.items()})
    return _coerce_numeric(df)


def _load_mat(path: Path) -> pd.DataFrame:
    """Load a MATLAB .mat file. Handles v5 via scipy and v7.3 (HDF5) via h5py."""
    # Check if it's an HDF5-based .mat (v7.3) by reading the magic bytes
    with open(path, "rb") as f:
        magic = f.read(8)

    if magic[:4] == b"\x89HDF" or magic[:8] == b"\x89HDF\r\n\x1a\n":
        # v7.3 .mat files are HDF5 — delegate to HDF5 loader
        print("[Praxis] Detected MATLAB v7.3 (HDF5) format, loading via h5py.")
        return _load_hdf5(path)

    try:
        from scipy.io import loadmat
    except ImportError:
        print("[Praxis] scipy is not installed. Install it with: pip install scipy")
        raise ImportError(
            "scipy is required to load .mat files. Install with: pip install scipy"
        )

    mat = loadmat(path, squeeze_me=True)

    # Filter out metadata keys (start with '__')
    arrays: dict[str, np.ndarray] = {}
    for key, val in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(val, np.ndarray):
            if val.ndim == 2:
                for i in range(val.shape[1]):
                    arrays[f"{key}_{i}" if val.shape[1] > 1 else key] = val[:, i]
            elif val.ndim == 1:
                arrays[key] = val
            elif val.ndim == 0:
                continue  # scalar, skip

    if not arrays:
        raise ValueError(f"No numeric arrays found in .mat file: {path}")

    # Align lengths by trimming to shortest (or let pandas fill with NaN)
    df = pd.DataFrame(dict(arrays))
    return _coerce_numeric(df)


def _load_brml(path: Path) -> pd.DataFrame:
    """Load a Bruker .brml XRD file (ZIP containing XML)."""
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a valid ZIP/BRML file: {path}")

    xml_content = None
    with zipfile.ZipFile(path, "r") as zf:
        # Look for the main measurement XML
        for name in zf.namelist():
            if name.lower().endswith(".xml") or "RawData" in name:
                xml_content = zf.read(name)
                break
        # Fallback: just grab the first XML-ish file
        if xml_content is None:
            for name in zf.namelist():
                if name.lower().endswith(".xml"):
                    xml_content = zf.read(name)
                    break

    if xml_content is None:
        raise ValueError(f"No XML data found inside BRML archive: {path}")

    root = ET.fromstring(xml_content)

    two_theta: list[float] = []
    intensity: list[float] = []

    # Bruker BRML XML structure: look for DataRoutes or scan data
    # Common tags: <Datum>, <SubScanCondition>, <DataRoute>
    for datum in root.iter("Datum"):
        text = datum.text
        if text:
            parts = text.strip().split(",")
            if len(parts) >= 2:
                try:
                    two_theta.append(float(parts[0]))
                    intensity.append(float(parts[1]))
                except ValueError:
                    continue

    # Alternative: look for 2theta start/stop/step + intensity counts
    if not two_theta:
        for route in root.iter("DataRoute"):
            for scan_info in route.iter("ScanInformation"):
                start_el = scan_info.find(".//*[@Name='Start']")
                stop_el = scan_info.find(".//*[@Name='Stop']")
                step_el = scan_info.find(".//*[@Name='Increment']")
                if start_el is not None and stop_el is not None and step_el is not None:
                    start = float(start_el.text or start_el.get("Value", "0"))
                    stop = float(stop_el.text or stop_el.get("Value", "0"))
                    step = float(step_el.text or step_el.get("Value", "0"))
                    if step > 0:
                        two_theta = np.arange(start, stop + step / 2, step).tolist()
            for counts_el in route.iter("Intensities"):
                if counts_el.text:
                    intensity = [float(v) for v in counts_el.text.strip().split()]

    if not two_theta or not intensity:
        raise ValueError(f"Could not extract 2theta/intensity data from BRML: {path}")

    n = min(len(two_theta), len(intensity))
    return pd.DataFrame({"2theta": two_theta[:n], "intensity": intensity[:n]})


def _load_xrdml(path: Path) -> pd.DataFrame:
    """Load a PANalytical .xrdml XRD file.

    XRDML is XML; PANalytical software sometimes ships it as a plain XML
    file and sometimes as a zip archive. The relevant content is one or
    more ``<dataPoints>`` blocks, each with a ``<positions axis="2Theta">``
    range and a space-separated ``<intensities>`` array.
    """
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            xml_name = next(
                (n for n in zf.namelist() if n.lower().endswith(".xml")),
                None,
            )
            if xml_name is None:
                raise ValueError(f"No XML found inside zipped XRDML: {path}")
            xml_content = zf.read(xml_name)
    else:
        xml_content = path.read_bytes()

    # Strip XML namespace so element lookups work without verbosity
    text = xml_content.decode("utf-8", errors="replace")
    text = re.sub(r'\sxmlns(:\w+)?="[^"]+"', "", text, count=0)
    root = ET.fromstring(text)

    two_theta_segments: list[np.ndarray] = []
    intensity_segments: list[np.ndarray] = []

    for dp in root.iter("dataPoints"):
        # Find the 2theta axis (positions block)
        start = stop = None
        for pos in dp.findall("positions"):
            axis = (pos.get("axis") or "").lower()
            if axis in ("2theta", "twotheta"):
                start_el = pos.find("startPosition")
                stop_el = pos.find("endPosition")
                if start_el is not None and stop_el is not None:
                    start = float(start_el.text)
                    stop = float(stop_el.text)
                    break
        intens_el = dp.find("intensities")
        if intens_el is None:
            intens_el = dp.find("counts")
        if intens_el is None or not intens_el.text:
            continue
        intensities = np.array(intens_el.text.strip().split(), dtype=float)
        if start is None or stop is None or intensities.size == 0:
            continue
        two_theta = np.linspace(start, stop, intensities.size)
        two_theta_segments.append(two_theta)
        intensity_segments.append(intensities)

    if not two_theta_segments:
        raise ValueError(
            f"No 2theta/intensity data extracted from XRDML file: {path}. "
            "The file may use a non-standard layout."
        )

    two_theta = np.concatenate(two_theta_segments)
    intensity = np.concatenate(intensity_segments)
    return pd.DataFrame({"two_theta_deg": two_theta, "intensity": intensity})


def _load_gamry_dta(path: Path) -> pd.DataFrame:
    """Load a Gamry .dta electrochemistry file (ASCII with header)."""
    raw = _read_raw(path)
    lines = raw.splitlines()

    # Gamry .dta files have a header section ending with "CURVE\tTABLE" or
    # a line containing "Pt\tT\t..." (the column header).
    # The data table starts after a line count indicator.
    data_start = None
    header_cols: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for the CURVE TABLE marker
        if "CURVE" in stripped and "TABLE" in stripped:
            # Next lines: column count, column names/units, then data
            # Typically: line i+1 = number of points
            #            line i+2 = column header labels
            #            line i+3 = column units
            #            line i+4+ = data
            if i + 2 < len(lines):
                header_cols = lines[i + 2].strip().split("\t")
            if i + 3 < len(lines):
                # units line — skip
                data_start = i + 4
            else:
                data_start = i + 2
            break

    if data_start is None:
        # Fallback: try to find numeric data after header
        print("[Praxis] Could not find CURVE TABLE marker in .dta file, "
              "falling back to generic delimited parser.")
        return _load_delimited(path)

    # Parse data rows
    rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("\t")
        # Gamry rows often start with a point index; keep all fields
        try:
            row = [float(p) for p in parts]
            rows.append(row)
        except ValueError:
            # Some lines may have trailing text; try partial parse
            row = []
            for p in parts:
                try:
                    row.append(float(p))
                except ValueError:
                    row.append(np.nan)
            if any(not np.isnan(v) for v in row):
                rows.append(row)

    if not rows:
        raise ValueError(f"No data rows found in Gamry .dta file: {path}")

    df = pd.DataFrame(rows)
    if header_cols and len(header_cols) == len(df.columns):
        df.columns = header_cols
    else:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    return _coerce_numeric(df)


def _load_biologic_mpr(path: Path) -> pd.DataFrame:
    """Load a Bio-Logic .mpr electrochemistry file.

    The .mpr format is proprietary and binary. Praxis delegates parsing to
    the `galvani` package, which must be installed separately:

        pip install praxis-sci[biologic]   # or: pip install galvani

    Returns a DataFrame with one column per channel recorded by EC-Lab
    (typically time, Ewe, I, cycle, and any additional channels).
    """
    try:
        from galvani import BioLogic
    except ImportError as exc:
        raise ImportError(
            "Reading Bio-Logic .mpr files requires the 'galvani' package. "
            "Install it with:  pip install galvani  "
            "(or 'pip install praxis-sci[biologic]')."
        ) from exc

    mpr = BioLogic.MPRfile(str(path))
    data = mpr.data
    # 'data' is a numpy structured array; convert to a DataFrame preserving names
    if hasattr(data, "dtype") and data.dtype.names:
        df = pd.DataFrame({name: data[name] for name in data.dtype.names})
    else:
        df = pd.DataFrame(data)
    return _coerce_numeric(df)


def _load_clipboard() -> pd.DataFrame:
    """Load tabular data from the system clipboard."""
    df = pd.read_clipboard()
    _report_load("clipboard", df)
    return df


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

_COMMENT_CHARS = ("#", "%", "!", ";", "//", "*")


def _detect_comment_rows(lines: list[str]) -> int:
    """Count leading comment/metadata lines.

    Handles three patterns:
      1. Lines starting with a known comment character.
      2. Blank lines.
      3. Instrument metadata blocks (e.g. ``Instrument: XRD-7000``,
         ``Date: 2024-01-15``) that don't use comment characters but
         contain mostly non-numeric content.

    Strategy:
      a. Filter out blank lines and comment-character lines.
      b. Detect the actual data delimiter by scoring uniformity of
         field counts on the surviving lines.
      c. Find the first surviving line whose fields are majority-numeric
         under that delimiter. That marks the start of data.
      d. The line immediately before, if it has the same field count and
         is all non-numeric, is the column-header row -- include it in
         the data section.
    """
    def _is_comment(s: str) -> bool:
        return not s or any(s.startswith(c) for c in _COMMENT_CHARS)

    # Build list of (original_index, stripped_line) for non-comment lines
    survivors: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _is_comment(stripped):
            continue
        survivors.append((i, stripped))

    if not survivors:
        # Pure-comment file: count leading comment lines
        count = 0
        for line in lines:
            if _is_comment(line.strip()):
                count += 1
            else:
                break
        return count

    # Detect the real delimiter using the survivors
    delim = _detect_delimiter([s for _, s in survivors])
    sep = re.escape(delim) if delim != r"\s+" else r"\s+"

    def _fields(s: str) -> list[str]:
        return [f for f in re.split(sep, s) if f]

    # Find first survivor that is majority-numeric under the real delimiter
    data_idx: Optional[int] = None
    data_field_count = 0
    for orig_i, s in survivors:
        fields = _fields(s)
        if len(fields) < 2:
            continue
        numeric = sum(1 for f in fields if _is_numeric(f))
        if numeric / len(fields) >= 0.5:
            data_idx = orig_i
            data_field_count = len(fields)
            break

    if data_idx is None:
        # No clearly-numeric line found; treat first survivor as data start
        return survivors[0][0]

    # Walk back one line: if same field count and all non-numeric, keep as header
    if data_idx > 0:
        prev = lines[data_idx - 1].strip()
        if prev and not _is_comment(prev):
            prev_fields = _fields(prev)
            if (
                len(prev_fields) == data_field_count
                and all(not _is_numeric(f) for f in prev_fields)
            ):
                return data_idx - 1

    return data_idx


def _detect_delimiter(lines: list[str]) -> str:
    """Guess the delimiter from data lines.

    Scoring strategy: for each candidate delimiter, count the fields it
    produces on every sampled line. Only lines where it produces >=2
    fields contribute to the score. The chosen delimiter maximises
    ``(mean field count / (1 + variance)) * coverage`` -- so it must be
    consistent and apply to most lines.

    This correctly handles:
      * Pure CSV / TSV
      * European data where ``,`` is a decimal and ``;`` is the delimiter
      * Files with leading metadata lines that contain no delimiters
    """
    candidates = ["\t", ";", ",", "|", " "]
    sample = [line.strip() for line in lines[:30] if line.strip()]
    if not sample:
        return r"\s+"

    best_delim = None
    best_score = -1.0
    for d in candidates:
        if d == " ":
            counts = [len(re.split(r"\s+", line)) for line in sample]
        else:
            counts = [line.count(d) + 1 for line in sample]
        valid = [c for c in counts if c >= 2]
        if len(valid) < 2:
            continue
        avg = sum(valid) / len(valid)
        var = sum((v - avg) ** 2 for v in valid) / len(valid)
        coverage = len(valid) / len(sample)
        score = (avg / (1.0 + var)) * coverage
        if score > best_score:
            best_score = score
            best_delim = d

    if best_delim is None:
        return r"\s+"
    if best_delim == " ":
        return r"\s+"
    return best_delim


def _detect_decimal(lines: list[str], delimiter: str) -> str:
    """Detect whether comma or dot is used as decimal separator."""
    if delimiter == ",":
        # If comma is delimiter, decimal must be dot
        return "."

    # Check for European-style comma decimals in numeric fields
    comma_count = 0
    dot_count = 0
    for line in lines[:20]:
        fields = re.split(re.escape(delimiter) if delimiter != r"\s+" else r"\s+", line.strip())
        for field in fields:
            field = field.strip()
            if re.match(r"^\d+,\d+$", field):
                comma_count += 1
            if re.match(r"^\d+\.\d+$", field):
                dot_count += 1

    return "," if comma_count > dot_count else "."


def _detect_header(lines: list[str], delimiter: str) -> bool:
    """Guess whether the first data line is a header row."""
    if not lines:
        return False

    first = lines[0].strip()
    sep = re.escape(delimiter) if delimiter != r"\s+" else r"\s+"
    fields = re.split(sep, first)

    # If most fields are non-numeric, it's likely a header
    non_numeric = sum(1 for f in fields if not _is_numeric(f.strip()))
    return non_numeric > len(fields) / 2


def _is_numeric(s: str) -> bool:
    """Check if a string represents a number."""
    try:
        float(s.replace(",", "."))
        return True
    except ValueError:
        return False


def _read_raw(path: Path, encoding: Optional[str] = None) -> str:
    """Read file with encoding auto-detection and BOM handling.

    Strategy:
      1. If ``encoding`` is given, use it.
      2. Otherwise inspect the first bytes for a BOM and pick the encoding
         it implies (UTF-8, UTF-16-LE, UTF-16-BE).
      3. Try the common encodings in order: utf-8, utf-8-sig, cp1252,
         latin-1, utf-16.
      4. If those all fail and ``charset-normalizer`` is installed, ask
         it for a best guess.
    """
    raw_bytes = path.read_bytes()

    # BOM-based detection
    if encoding is None:
        if raw_bytes.startswith(b"\xef\xbb\xbf"):
            encoding = "utf-8-sig"
        elif raw_bytes.startswith(b"\xff\xfe") or raw_bytes.startswith(b"\xfe\xff"):
            encoding = "utf-16"

    if encoding:
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError) as exc:
            raise UnicodeDecodeError(
                encoding, raw_bytes, 0, 1,
                f"Cannot decode {path} with requested encoding '{encoding}': {exc}",
            )

    # Fallback chain
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1", "utf-16"):
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue

    # Last resort: charset-normalizer if available
    try:
        from charset_normalizer import from_bytes
        result = from_bytes(raw_bytes).best()
        if result is not None:
            print(f"[Praxis] Auto-detected encoding '{result.encoding}' for {path.name}.")
            return str(result)
    except ImportError:
        pass

    raise UnicodeDecodeError(
        "utf-8", raw_bytes[:32], 0, 1,
        f"Cannot decode {path}. Tried utf-8, utf-8-sig, cp1252, latin-1, utf-16. "
        f"Install 'charset-normalizer' for broader detection: pip install charset-normalizer",
    )


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Try to convert columns to numeric where possible."""
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except (ValueError, TypeError):
            pass
    return df


def _report_load(source: Any, df: pd.DataFrame) -> None:
    """Print a summary of what was loaded."""
    shape = f"{df.shape[0]} rows × {df.shape[1]} columns"
    cols = ", ".join(str(c) for c in df.columns[:6])
    if len(df.columns) > 6:
        cols += f" ... (+{len(df.columns) - 6} more)"
    print(f"[Praxis] Loaded {source}: {shape}")
    print(f"[Praxis] Columns: {cols}")

    # Report detected types
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        print(f"[Praxis] Numeric columns: {', '.join(str(c) for c in numeric_cols)}")
