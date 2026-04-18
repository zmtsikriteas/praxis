"""Battery cycling analysis.

Covers galvanostatic cycling metrics at both the per-cycle summary level
(capacity retention, coulombic efficiency, capacity fade) and the time-
series level (dQ/dV, differential-capacity analysis).

Input conventions:
  * Capacities are specific (mAh/g) unless absolute values are passed.
  * Cycle numbers are 1-indexed; cycle 1 is the formation cycle unless
    the caller excludes it.
  * Voltage is in volts, current in amperes for dQ/dV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from praxis.core.utils import validate_array


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CycleMetrics:
    """Metrics for a single charge-discharge cycle."""
    cycle: int
    charge_capacity: float
    discharge_capacity: float
    coulombic_efficiency: float   # %


@dataclass
class CyclingResults:
    """Aggregated results for a cycling experiment."""
    cycles: list[CycleMetrics] = field(default_factory=list)
    initial_capacity: float = 0.0
    final_capacity: float = 0.0
    capacity_retention_pct: float = 0.0       # final/initial x 100
    capacity_fade_pct_per_cycle: float = 0.0  # mean fade per cycle, %
    mean_coulombic_efficiency: float = 0.0    # across all but the first cycle

    def table(self) -> str:
        n = len(self.cycles)
        lines = ["[Praxis] Battery Cycling Analysis"]
        lines.append(f"  Cycles analysed:            {n}")
        lines.append(f"  Initial discharge capacity: {self.initial_capacity:.2f} mAh/g")
        lines.append(f"  Final discharge capacity:   {self.final_capacity:.2f} mAh/g")
        lines.append(f"  Capacity retention:         {self.capacity_retention_pct:.1f}%")
        lines.append(f"  Mean fade per cycle:        {self.capacity_fade_pct_per_cycle:.3f}%")
        lines.append(f"  Mean Coulombic efficiency:  {self.mean_coulombic_efficiency:.2f}%")

        if n:
            lines.append("")
            lines.append(f"  {'cycle':>5}  {'Q_ch':>8}  {'Q_dis':>8}  {'CE':>7}")
            lines.append("  " + "-" * 34)
            for m in self.cycles[:6]:
                lines.append(
                    f"  {m.cycle:>5d}  {m.charge_capacity:>8.2f}  "
                    f"{m.discharge_capacity:>8.2f}  {m.coulombic_efficiency:>6.2f}%"
                )
            if n > 6:
                lines.append(f"  ... ({n - 6} more cycles)")
        return "\n".join(lines)


@dataclass
class DQDVResult:
    """Differential-capacity curve for one half-cycle."""
    voltage: np.ndarray           # midpoint voltages
    dqdv: np.ndarray              # dQ/dV (mAh/V), positive for charge, negative for discharge
    peak_voltages: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-cycle summary analysis
# ---------------------------------------------------------------------------

def analyse_cycle_summary(
    cycle: Sequence[int],
    charge_capacity: Sequence[float],
    discharge_capacity: Sequence[float],
    *,
    skip_formation: bool = True,
) -> CyclingResults:
    """Summarise a cycling experiment from per-cycle capacity data.

    Parameters
    ----------
    cycle :
        Cycle numbers (1-indexed).
    charge_capacity, discharge_capacity :
        Capacity per cycle in mAh or mAh/g. Must have the same length as
        ``cycle``.
    skip_formation :
        If True (default), exclude the first cycle from fade and mean-CE
        calculations. Set False if formation cycles were already removed.
    """
    cycle = np.asarray(cycle, dtype=int)
    q_ch = validate_array(charge_capacity, name="charge_capacity")
    q_dis = validate_array(discharge_capacity, name="discharge_capacity")
    if not (len(cycle) == len(q_ch) == len(q_dis)):
        raise ValueError("cycle, charge_capacity, discharge_capacity must all have the same length")

    with np.errstate(divide="ignore", invalid="ignore"):
        ce = np.where(q_ch > 0, 100.0 * q_dis / q_ch, np.nan)

    cycles = [
        CycleMetrics(int(c), float(a), float(b), float(e))
        for c, a, b, e in zip(cycle, q_ch, q_dis, ce)
    ]
    results = CyclingResults(cycles=cycles)

    if len(cycles) == 0:
        return results

    start_idx = 1 if (skip_formation and len(cycles) > 1) else 0
    q_fade = q_dis[start_idx:]
    if q_fade.size == 0:
        return results

    results.initial_capacity = float(q_fade[0])
    results.final_capacity = float(q_fade[-1])
    results.capacity_retention_pct = 100.0 * results.final_capacity / results.initial_capacity
    n_fade_cycles = max(1, q_fade.size - 1)
    results.capacity_fade_pct_per_cycle = (
        100.0 * (results.initial_capacity - results.final_capacity)
        / (results.initial_capacity * n_fade_cycles)
    )
    valid_ce = ce[start_idx:][~np.isnan(ce[start_idx:])]
    if valid_ce.size:
        results.mean_coulombic_efficiency = float(np.mean(valid_ce))
    return results


# ---------------------------------------------------------------------------
# dQ/dV
# ---------------------------------------------------------------------------

def compute_dqdv(
    voltage: Sequence[float],
    capacity: Sequence[float],
    *,
    smoothing_window: int = 11,
    min_peak_prominence: float = 5.0,
) -> DQDVResult:
    """Differential-capacity curve from voltage-capacity data of one half-cycle.

    Parameters
    ----------
    voltage :
        Cell voltage (V), monotonic within a half-cycle.
    capacity :
        Cumulative capacity (mAh or mAh/g). Must be monotonic in the
        same direction as the half-cycle.
    smoothing_window :
        Odd window size passed to a simple moving average. Set to 1 to
        disable smoothing.
    min_peak_prominence :
        Minimum prominence for :func:`scipy.signal.find_peaks` to flag a
        dQ/dV peak. Units are mAh/V on the smoothed curve.
    """
    V = validate_array(voltage, name="voltage")
    Q = validate_array(capacity, name="capacity")
    if V.size != Q.size:
        raise ValueError("voltage and capacity must have the same length")
    if V.size < 4:
        raise ValueError("need at least 4 points to compute dQ/dV")

    # Centred finite difference in V (monotonic within a half-cycle)
    dV = np.gradient(V)
    dQ = np.gradient(Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = np.where(np.abs(dV) > 1e-12, dQ / dV, 0.0)

    if smoothing_window and smoothing_window > 1:
        w = int(smoothing_window) | 1   # force odd
        kernel = np.ones(w) / w
        dqdv = np.convolve(dqdv, kernel, mode="same")

    # Peak detection (positive and negative peaks depending on sweep direction)
    from scipy.signal import find_peaks
    positive = dqdv if np.mean(dqdv) >= 0 else -dqdv
    peaks, _ = find_peaks(positive, prominence=min_peak_prominence)
    peak_voltages = [float(V[p]) for p in peaks]

    return DQDVResult(voltage=V, dqdv=dqdv, peak_voltages=peak_voltages)


# ---------------------------------------------------------------------------
# Rate capability
# ---------------------------------------------------------------------------

def rate_capability(
    c_rates: Sequence[float],
    discharge_capacities: Sequence[float],
) -> dict:
    """Summarise capacity retention across different C-rates.

    Returns a dict with per-rate mean capacity and retention vs the
    lowest rate (assumed to be the reference capacity).
    """
    c = validate_array(c_rates, name="c_rates")
    q = validate_array(discharge_capacities, name="discharge_capacities")
    if c.size != q.size:
        raise ValueError("c_rates and discharge_capacities must have the same length")

    unique = np.unique(c)
    per_rate = {}
    for rate in unique:
        mask = c == rate
        per_rate[float(rate)] = float(np.mean(q[mask]))

    ref_rate = float(unique.min())
    ref_cap = per_rate[ref_rate]
    retention = {r: 100.0 * cap / ref_cap for r, cap in per_rate.items()}
    return {
        "capacity_per_rate": per_rate,
        "reference_rate": ref_rate,
        "retention_pct": retention,
    }
