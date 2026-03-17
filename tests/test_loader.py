"""Tests for the universal data loader."""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from core.loader import load_data

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"


class TestCSVLoading:
    """Test CSV file loading with auto-detection."""

    def test_load_xrd_csv(self):
        df = load_data(SAMPLE_DIR / "xrd_sample.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df.shape[1] == 2

    def test_load_impedance_csv(self):
        df = load_data(SAMPLE_DIR / "impedance_sample.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "frequency" in df.columns or "z_real" in df.columns

    def test_load_dsc_csv(self):
        df = load_data(SAMPLE_DIR / "dsc_sample.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_stress_strain_csv(self):
        df = load_data(SAMPLE_DIR / "stress_strain_sample.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df.shape[1] == 2

    def test_numeric_columns_detected(self):
        df = load_data(SAMPLE_DIR / "xrd_sample.csv")
        numeric = df.select_dtypes(include="number")
        assert numeric.shape[1] == 2, "Both columns should be numeric"

    def test_column_subset(self):
        df = load_data(SAMPLE_DIR / "impedance_sample.csv", columns=[0, 1])
        assert df.shape[1] == 2


class TestAutoDetection:
    """Test auto-detection of delimiters, headers, etc."""

    def test_comma_delimiter(self, tmp_path):
        p = tmp_path / "comma.csv"
        p.write_text("x,y\n1.0,2.0\n3.0,4.0\n")
        df = load_data(p)
        assert df.shape == (2, 2)

    def test_tab_delimiter(self, tmp_path):
        p = tmp_path / "tab.txt"
        p.write_text("x\ty\n1.0\t2.0\n3.0\t4.0\n")
        df = load_data(p)
        assert df.shape == (2, 2)

    def test_space_delimiter(self, tmp_path):
        p = tmp_path / "space.dat"
        p.write_text("1.0 2.0\n3.0 4.0\n5.0 6.0\n")
        df = load_data(p)
        assert df.shape == (3, 2)

    def test_comment_lines_skipped(self, tmp_path):
        p = tmp_path / "commented.csv"
        p.write_text("# This is a comment\n# Another comment\nx,y\n1.0,2.0\n3.0,4.0\n")
        df = load_data(p)
        assert len(df) == 2

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")


class TestXYFormat:
    """Test .xy file loading."""

    def test_load_xy(self, tmp_path):
        p = tmp_path / "test.xy"
        p.write_text("10.0  100\n20.0  200\n30.0  300\n")
        df = load_data(p)
        assert df.shape == (3, 2)
        assert "x" in df.columns
        assert "y" in df.columns

    def test_xy_with_comments(self, tmp_path):
        p = tmp_path / "test.xy"
        p.write_text("# XRD data\n# Cu Ka\n10.0  100\n20.0  200\n")
        df = load_data(p)
        assert df.shape == (2, 2)


class TestJSONLoading:
    """Test JSON file loading."""

    def test_load_json_records(self, tmp_path):
        import json
        p = tmp_path / "data.json"
        data = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        p.write_text(json.dumps(data))
        df = load_data(p)
        assert df.shape == (2, 2)

    def test_load_json_columns(self, tmp_path):
        import json
        p = tmp_path / "data.json"
        data = {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}
        p.write_text(json.dumps(data))
        df = load_data(p)
        assert df.shape == (3, 2)
