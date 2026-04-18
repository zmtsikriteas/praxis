"""Tests for the battery cycling technique module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from praxis.core.loader import load_sample
from praxis.techniques.battery_cycling import (
    analyse_cycle_summary,
    compute_dqdv,
    rate_capability,
)


class TestCycleSummary:
    def test_basic_summary(self):
        cycle = [1, 2, 3, 4, 5]
        q_ch = [110.0, 105.0, 103.0, 101.0, 100.0]
        q_dis = [100.0, 102.0, 100.5, 99.5, 98.0]
        res = analyse_cycle_summary(cycle, q_ch, q_dis)
        assert len(res.cycles) == 5
        # Skip formation: initial = cycle 2's discharge
        assert res.initial_capacity == pytest.approx(102.0)
        assert res.final_capacity == pytest.approx(98.0)
        assert res.capacity_retention_pct == pytest.approx(98.0 / 102.0 * 100, rel=1e-6)
        assert res.mean_coulombic_efficiency > 95

    def test_no_skip_formation(self):
        cycle = [1, 2, 3]
        q_ch = [100.0, 100.0, 100.0]
        q_dis = [90.0, 95.0, 99.0]
        res = analyse_cycle_summary(cycle, q_ch, q_dis, skip_formation=False)
        assert res.initial_capacity == pytest.approx(90.0)

    def test_table_renders(self):
        cycle = [1, 2, 3]
        q_ch = [110.0, 105.0, 103.0]
        q_dis = [100.0, 102.0, 100.5]
        res = analyse_cycle_summary(cycle, q_ch, q_dis)
        text = res.table()
        assert "Battery Cycling" in text
        assert "Coulombic" in text
        assert "%" in text

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            analyse_cycle_summary([1, 2], [100.0], [99.0, 98.0])


class TestDQDV:
    def test_basic_dqdv(self):
        # Two-plateau half-cycle: voltage rises with capacity, plateaus at 3.4 and 3.6 V
        V = np.concatenate([
            np.linspace(3.0, 3.4, 50),
            np.full(80, 3.4) + np.linspace(0, 0.001, 80),
            np.linspace(3.4, 3.6, 50),
            np.full(80, 3.6) + np.linspace(0, 0.001, 80),
            np.linspace(3.6, 4.0, 30),
        ])
        Q = np.linspace(0, 150, V.size)
        result = compute_dqdv(V, Q, smoothing_window=5)
        assert result.voltage.size == V.size
        assert result.dqdv.size == V.size
        # Should detect roughly two peaks at the plateaus
        assert len(result.peak_voltages) >= 1

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 4"):
            compute_dqdv([3.0, 3.5], [0, 50])


class TestRateCapability:
    def test_retention_against_lowest_rate(self):
        c_rates = [0.1, 0.1, 0.5, 0.5, 1.0, 1.0, 0.1]
        capacities = [150.0, 152.0, 140.0, 142.0, 130.0, 132.0, 149.0]
        res = rate_capability(c_rates, capacities)
        assert res["reference_rate"] == 0.1
        assert res["retention_pct"][0.1] == pytest.approx(100.0)
        assert 90 < res["retention_pct"][0.5] < 95


class TestSampleDataset:
    def test_built_in_sample_loads(self):
        df = load_sample("battery_cycling")
        assert "cycle" in df.columns
        assert "discharge_capacity_mAh_per_g" in df.columns
        assert len(df) >= 30

    def test_full_pipeline_on_sample(self):
        df = load_sample("battery_cycling")
        res = analyse_cycle_summary(
            df["cycle"].values,
            df["charge_capacity_mAh_per_g"].values,
            df["discharge_capacity_mAh_per_g"].values,
        )
        assert res.initial_capacity > 100
        assert res.final_capacity < res.initial_capacity   # capacity fades
        assert 80 < res.capacity_retention_pct < 100
        assert 90 < res.mean_coulombic_efficiency <= 100.5
