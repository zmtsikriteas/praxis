"""Nanoindentation analysis using the Oliver-Pharr method.

Contact area functions, creep analysis, and batch indent statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from praxis.core.utils import validate_xy, validate_array
from praxis.analysis.fitting import fit_curve


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tip geometry correction factor (beta)
TIP_BETA: dict[str, float] = {
    "berkovich": 1.034,
    "vickers": 1.012,
    "cube_corner": 1.034,
    "conical": 1.000,
}

# Conversion factors to SI (metres, newtons)
DEPTH_UNITS: dict[str, float] = {"nm": 1e-9, "um": 1e-6, "mm": 1e-3, "m": 1.0}
LOAD_UNITS: dict[str, float] = {"uN": 1e-6, "mN": 1e-3, "N": 1.0}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IndentResult:
    """Results from a single nanoindentation analysis."""
    hardness_gpa: float
    modulus_gpa: float
    reduced_modulus_gpa: float
    contact_stiffness: float  # S = dP/dh, in input load/depth units
    contact_depth: float  # hc, in input depth units
    max_load: float  # input load units
    max_depth: float  # input depth units

    def table(self) -> str:
        lines = [
            "[Praxis] Nanoindentation (Oliver-Pharr)",
            "  " + "-" * 40,
            f"  Max load         = {self.max_load:.4f}",
            f"  Max depth        = {self.max_depth:.4f}",
            f"  Contact depth    = {self.contact_depth:.4f}",
            f"  Contact stiffness= {self.contact_stiffness:.4f}",
            f"  Hardness         = {self.hardness_gpa:.3f} GPa",
            f"  Reduced modulus  = {self.reduced_modulus_gpa:.3f} GPa",
            f"  Elastic modulus  = {self.modulus_gpa:.3f} GPa",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.table()


@dataclass
class BatchIndentResult:
    """Statistical results from multiple indentations."""
    mean_hardness: float
    std_hardness: float
    mean_modulus: float
    std_modulus: float
    n_indents: int
    results: list[IndentResult] = field(default_factory=list)

    def table(self) -> str:
        lines = [
            f"[Praxis] Batch Nanoindentation -- {self.n_indents} indent(s)",
            "  " + "-" * 45,
            f"  Hardness:  {self.mean_hardness:.3f} +/- {self.std_hardness:.3f} GPa",
            f"  Modulus:   {self.mean_modulus:.3f} +/- {self.std_modulus:.3f} GPa",
        ]
        if self.results:
            lines.append("")
            lines.append(f"  {'#':>3}  {'H (GPa)':>10}  {'E (GPa)':>10}")
            lines.append("  " + "-" * 28)
            for i, r in enumerate(self.results, 1):
                lines.append(f"  {i:>3}  {r.hardness_gpa:>10.3f}  {r.modulus_gpa:>10.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.table()


# ---------------------------------------------------------------------------
# Contact area functions
# ---------------------------------------------------------------------------

def calc_contact_area(
    hc: float,
    *,
    tip: str = "berkovich",
    half_angle: float = 70.3,
) -> float:
    """Calculate projected contact area from contact depth.

    Parameters
    ----------
    hc : float
        Contact depth.
    tip : str
        Tip geometry: 'berkovich', 'vickers', 'cube_corner', or 'conical'.
    half_angle : float
        Half-included angle in degrees (used for conical tip only).

    Returns
    -------
    float
        Projected contact area.
    """
    tip = tip.lower()
    if tip in ("berkovich", "vickers"):
        return 24.5 * hc ** 2
    elif tip == "cube_corner":
        return 2.598 * hc ** 2
    elif tip == "conical":
        alpha_rad = math.radians(half_angle)
        return math.pi * (math.tan(alpha_rad) ** 2) * hc ** 2
    else:
        raise ValueError(f"Unknown tip geometry: {tip}. Use berkovich, vickers, cube_corner, or conical.")


# ---------------------------------------------------------------------------
# Oliver-Pharr analysis
# ---------------------------------------------------------------------------

def analyse_indent(
    depth: Any,
    load: Any,
    *,
    tip: str = "berkovich",
    poisson_sample: float = 0.3,
    poisson_tip: float = 0.07,
    E_tip: float = 1141e9,
    unload_fraction: float = 0.8,
    depth_unit: str = "nm",
    load_unit: str = "mN",
) -> IndentResult:
    """Oliver-Pharr nanoindentation analysis.

    Parameters
    ----------
    depth : array-like
        Indentation depth, in *depth_unit*.
    load : array-like
        Applied load, in *load_unit*.
    tip : str
        Indenter tip geometry.
    poisson_sample : float
        Poisson's ratio of the sample.
    poisson_tip : float
        Poisson's ratio of the indenter (0.07 for diamond).
    E_tip : float
        Elastic modulus of the indenter in Pa (1141 GPa for diamond).
    unload_fraction : float
        Fraction of the unloading curve to use for power-law fit (0-1).
    depth_unit : str
        Unit of *depth*: 'nm', 'um', 'mm' or 'm'.
    load_unit : str
        Unit of *load*: 'uN', 'mN' or 'N'.

    Returns
    -------
    IndentResult
        Hardness and moduli in GPa; depths, load and stiffness in the
        input units.
    """
    if depth_unit not in DEPTH_UNITS:
        raise ValueError(f"Unknown depth_unit '{depth_unit}'. Use one of {list(DEPTH_UNITS)}.")
    if load_unit not in LOAD_UNITS:
        raise ValueError(f"Unknown load_unit '{load_unit}'. Use one of {list(LOAD_UNITS)}.")
    k_h = DEPTH_UNITS[depth_unit]
    k_p = LOAD_UNITS[load_unit]

    h_arr, p_arr = validate_xy(
        np.asarray(depth, dtype=float),
        np.asarray(load, dtype=float),
        allow_nan=False,
    )

    # Find maximum load point
    max_idx = int(np.argmax(p_arr))
    p_max = float(p_arr[max_idx])
    h_max = float(h_arr[max_idx])

    # Extract unloading curve (from max load to end)
    h_unload = h_arr[max_idx:]
    p_unload = p_arr[max_idx:]

    if len(h_unload) < 5:
        raise ValueError("Not enough unloading data points (need at least 5).")

    # Use the top fraction of the unloading curve for the power-law fit
    p_threshold = p_max * (1.0 - unload_fraction)
    fit_mask = p_unload >= p_threshold
    h_fit = h_unload[fit_mask]
    p_fit = p_unload[fit_mask]

    if len(h_fit) < 3:
        raise ValueError("Not enough unloading points above threshold for fitting.")

    # Power law fit: P = A * (h - hf)^m
    # First estimate hf (final depth) from the last unloading point
    hf_est = float(h_unload[-1])

    # Fit in log-log space: log(P) = log(A) + m * log(h - hf)
    h_shifted = h_fit - hf_est
    valid = h_shifted > 0
    if np.sum(valid) < 3:
        # Fallback: use linear fit for stiffness
        coeffs = np.polyfit(h_fit, p_fit, 1)
        S = float(coeffs[0])
    else:
        h_shifted = h_shifted[valid]
        p_valid = p_fit[valid]
        log_h = np.log(h_shifted)
        log_p = np.log(p_valid)
        coeffs = np.polyfit(log_h, log_p, 1)
        m = coeffs[0]
        A = np.exp(coeffs[1])

        # Contact stiffness S = dP/dh at h_max
        # S = A * m * (h_max - hf)^(m-1)
        S = float(A * m * (h_max - hf_est) ** (m - 1))

    if S <= 0:
        raise ValueError(f"Negative or zero contact stiffness: S = {S}. Check unloading data.")

    # Contact depth: hc = h_max - epsilon * P_max / S
    # epsilon = 0.75 for Berkovich/Vickers (power-law), 0.72 for conical
    epsilon = 0.75 if tip.lower() != "conical" else 0.72
    hc = h_max - epsilon * p_max / S

    if hc <= 0:
        raise ValueError(f"Negative contact depth: hc = {hc}. Check data quality.")

    # Projected contact area
    A_contact = calc_contact_area(hc, tip=tip)

    if A_contact <= 0:
        raise ValueError(f"Zero contact area at hc = {hc}.")

    # Hardness (Pa): P / A, converted to SI
    H = (p_max * k_p) / (A_contact * k_h ** 2)

    # Reduced modulus (Pa)
    beta = TIP_BETA.get(tip.lower(), 1.0)
    S_si = S * k_p / k_h
    Er = (math.sqrt(math.pi) / (2.0 * beta)) * S_si / math.sqrt(A_contact * k_h ** 2)

    # Sample modulus from reduced modulus
    # 1/Er = (1 - vs^2)/Es + (1 - vi^2)/Ei
    tip_compliance = (1.0 - poisson_tip ** 2) / E_tip
    sample_compliance = 1.0 / Er - tip_compliance

    if sample_compliance <= 0:
        # Er is very high, sample modulus approaches infinity -- clamp
        Es = Er / (1.0 - poisson_sample ** 2)
    else:
        Es = (1.0 - poisson_sample ** 2) / sample_compliance

    result = IndentResult(
        hardness_gpa=float(H / 1e9),
        modulus_gpa=float(Es / 1e9),
        reduced_modulus_gpa=float(Er / 1e9),
        contact_stiffness=float(S),
        contact_depth=float(hc),
        max_load=float(p_max),
        max_depth=float(h_max),
    )

    print(result.table())
    return result


# ---------------------------------------------------------------------------
# Creep analysis
# ---------------------------------------------------------------------------

def creep_analysis(
    time: Any,
    depth: Any,
    *,
    load: Optional[float] = None,
    model: str = "log",
) -> dict[str, float]:
    """Analyse creep behaviour during a hold segment.

    Fits h(t) = h0 + a * ln(1 + b*t) (logarithmic model) or
          h(t) = h0 + a * t^n         (power law model).

    Parameters
    ----------
    time : array-like
        Time values (s), starting from the hold onset.
    depth : array-like
        Depth values during hold.
    load : float, optional
        Applied load during hold (for reporting).
    model : str
        Creep model: 'log' for logarithmic, 'power' for power law.

    Returns
    -------
    dict with fitted parameters and R^2.
    """
    t_arr, h_arr = validate_xy(
        np.asarray(time, dtype=float),
        np.asarray(depth, dtype=float),
        allow_nan=False,
    )

    # Shift time to start at 0
    t_arr = t_arr - t_arr[0]
    h0 = float(h_arr[0])

    # Delta depth
    dh = h_arr - h0

    if model == "log":
        # Fit: dh = a * ln(1 + b*t)
        # Simplified: try log fit with b=1 first, then optimise
        # dh = a * ln(1 + t) as initial approximation
        log_term = np.log1p(t_arr)
        # Avoid division by zero
        valid = log_term > 0
        if np.sum(valid) < 3:
            raise ValueError("Not enough data for logarithmic creep fit.")

        # Linear fit: dh = a * log_term
        a_est = float(np.polyfit(log_term[valid], dh[valid], 1)[0])

        # Predict and R^2
        dh_pred = a_est * log_term
        ss_res = float(np.sum((dh - dh_pred) ** 2))
        ss_tot = float(np.sum((dh - np.mean(dh)) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        result = {
            "model": "logarithmic",
            "h0": h0,
            "a": a_est,
            "b": 1.0,
            "r_squared": r_sq,
            "total_creep": float(dh[-1]),
        }
    elif model == "power":
        # Fit: dh = a * t^n  ->  log(dh) = log(a) + n*log(t)
        valid = (t_arr > 0) & (dh > 0)
        if np.sum(valid) < 3:
            raise ValueError("Not enough positive-displacement data for power-law creep fit.")

        log_t = np.log(t_arr[valid])
        log_dh = np.log(dh[valid])
        coeffs = np.polyfit(log_t, log_dh, 1)
        n = float(coeffs[0])
        a = float(np.exp(coeffs[1]))

        dh_pred = np.zeros_like(dh)
        dh_pred[t_arr > 0] = a * t_arr[t_arr > 0] ** n
        ss_res = float(np.sum((dh - dh_pred) ** 2))
        ss_tot = float(np.sum((dh - np.mean(dh)) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        result = {
            "model": "power_law",
            "h0": h0,
            "a": a,
            "n": n,
            "r_squared": r_sq,
            "total_creep": float(dh[-1]),
        }
    else:
        raise ValueError(f"Unknown creep model: {model}. Use 'log' or 'power'.")

    if load is not None:
        result["load"] = load

    print(f"[Praxis] Creep analysis ({result['model']})")
    print(f"  Total creep displacement: {result['total_creep']:.4f}")
    print(f"  R^2 = {result['r_squared']:.6f}")
    return result


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def batch_indents(
    indents_list: Sequence[tuple[Any, Any]],
    *,
    tip: str = "berkovich",
    poisson_sample: float = 0.3,
    poisson_tip: float = 0.07,
    E_tip: float = 1141e9,
    depth_unit: str = "nm",
    load_unit: str = "mN",
) -> BatchIndentResult:
    """Analyse multiple indentation curves and return statistics.

    Parameters
    ----------
    indents_list : list of (depth, load) tuples
        Each element is a (depth_array, load_array) pair.
    tip, poisson_sample, poisson_tip, E_tip, depth_unit, load_unit
        Passed to analyse_indent.

    Returns
    -------
    BatchIndentResult
    """
    results: list[IndentResult] = []
    for i, (depth, load) in enumerate(indents_list):
        try:
            r = analyse_indent(
                depth, load,
                tip=tip,
                poisson_sample=poisson_sample,
                poisson_tip=poisson_tip,
                E_tip=E_tip,
                depth_unit=depth_unit,
                load_unit=load_unit,
            )
            results.append(r)
        except (ValueError, RuntimeError) as e:
            print(f"  Warning: indent {i + 1} failed: {e}")

    if not results:
        raise ValueError("No indents could be analysed.")

    h_vals = np.array([r.hardness_gpa for r in results])
    e_vals = np.array([r.modulus_gpa for r in results])

    batch = BatchIndentResult(
        mean_hardness=float(np.mean(h_vals)),
        std_hardness=float(np.std(h_vals, ddof=1)) if len(h_vals) > 1 else 0.0,
        mean_modulus=float(np.mean(e_vals)),
        std_modulus=float(np.std(e_vals, ddof=1)) if len(e_vals) > 1 else 0.0,
        n_indents=len(results),
        results=results,
    )

    print(batch.table())
    return batch
