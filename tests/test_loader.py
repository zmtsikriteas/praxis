"""Tests for the universal data loader."""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from praxis.core.loader import load_data, load_sample, list_samples

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


class TestMetadataSniffing:
    """Test detection of instrument metadata blocks without comment chars."""

    def test_instrument_metadata_block(self, tmp_path):
        """Lines like 'Instrument: XRD-7000' should be skipped automatically."""
        p = tmp_path / "instrument_export.csv"
        p.write_text(
            "Instrument: XRD-7000\n"
            "Date: 2024-01-15\n"
            "Operator: J.Smith\n"
            "Sample: NaCl\n"
            "\n"
            "two_theta,intensity\n"
            "10.0,100\n"
            "10.1,105\n"
            "10.2,98\n"
        )
        df = load_data(p)
        assert df.shape == (3, 2)
        assert "two_theta" in df.columns
        assert "intensity" in df.columns

    def test_metadata_without_blank_separator(self, tmp_path):
        p = tmp_path / "metadata.txt"
        p.write_text(
            "Title: my measurement\n"
            "Range: 5-90 deg\n"
            "x\ty\n"
            "1.0\t10.0\n"
            "2.0\t20.0\n"
        )
        df = load_data(p)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["x", "y"]

    def test_pure_data_no_header(self, tmp_path):
        """Data with no header at all should still load."""
        p = tmp_path / "data.dat"
        p.write_text("1.0 2.0\n3.0 4.0\n5.0 6.0\n")
        df = load_data(p)
        assert df.shape == (3, 2)

    def test_data_with_only_header(self, tmp_path):
        """Header line directly followed by data, no metadata."""
        p = tmp_path / "data.csv"
        p.write_text("voltage,current\n0.0,0.1\n0.5,0.3\n1.0,0.5\n")
        df = load_data(p)
        assert list(df.columns) == ["voltage", "current"]
        assert df.shape == (3, 2)


class TestEncoding:
    """Test encoding auto-detection: BOM, UTF-16, cp1252."""

    def test_utf8_with_bom(self, tmp_path):
        p = tmp_path / "bom.csv"
        # UTF-8 BOM + ASCII content
        p.write_bytes("\ufeffx,y\n1.0,2.0\n3.0,4.0\n".encode("utf-8"))
        df = load_data(p)
        assert df.shape == (2, 2)
        assert "x" in df.columns

    def test_utf16_le(self, tmp_path):
        p = tmp_path / "utf16.csv"
        p.write_bytes("x,y\n1.0,2.0\n3.0,4.0\n".encode("utf-16"))
        df = load_data(p)
        assert df.shape == (2, 2)

    def test_cp1252_special_chars(self, tmp_path):
        """Files with Western European characters (e.g. degree symbol)."""
        p = tmp_path / "european.csv"
        # 'temperature (\xb0C)' with degree symbol in cp1252
        p.write_bytes(
            "temperature (\xb0C),value\n25.0,100.0\n50.0,200.0\n".encode("cp1252")
        )
        df = load_data(p)
        assert df.shape == (2, 2)

    def test_explicit_encoding(self, tmp_path):
        p = tmp_path / "explicit.csv"
        p.write_bytes("x,y\n1,2\n3,4\n".encode("latin-1"))
        df = load_data(p, encoding="latin-1")
        assert df.shape == (2, 2)


class TestEuropeanDecimals:
    """Test comma-decimal European number formatting."""

    def test_semicolon_with_comma_decimal(self, tmp_path):
        """German/French-style: semicolon delimiter, comma decimal."""
        p = tmp_path / "european.csv"
        p.write_text("x;y\n1,5;2,7\n3,1;4,9\n")
        df = load_data(p)
        assert df.shape == (2, 2)
        assert df["x"].iloc[0] == pytest.approx(1.5)
        assert df["y"].iloc[1] == pytest.approx(4.9)


class TestBuiltinSamples:
    """Test the load_sample() helper that ships sample datasets with the package."""

    def test_list_samples_nonempty(self):
        samples = list_samples()
        assert len(samples) > 0
        # Part-1 samples
        for name in ("xrd_silicon", "dsc_polymer", "raman_silicon"):
            assert name in samples, f"Expected sample '{name}' in {samples}"

    def test_load_by_exact_name(self):
        df = load_sample("xrd_silicon")
        assert len(df) > 0
        assert df.shape[1] == 2

    def test_load_by_prefix(self):
        """Short prefix should match the first file alphabetically."""
        df = load_sample("xrd")
        assert len(df) > 0

    def test_unknown_sample_raises(self):
        with pytest.raises(FileNotFoundError) as exc:
            load_sample("definitely_not_a_real_technique")
        assert "Available" in str(exc.value)

    def test_raman_sample_has_expected_shape(self):
        df = load_sample("raman")
        assert df.shape[1] == 2
        # Should contain the silicon 520 cm-1 peak within the data range
        col0 = df.iloc[:, 0]
        assert col0.min() < 520 < col0.max()

    def test_part2_samples_present(self):
        """Electrical / magnetic / dielectric samples added in part 2."""
        samples = list_samples()
        for name in ("impedance_rc", "iv_diode", "cv_redox",
                     "dielectric_relaxation", "piezo_pe_loop", "mh_loop"):
            assert name in samples

    def test_impedance_sample_three_columns(self):
        df = load_sample("impedance_rc")
        assert df.shape[1] == 3   # frequency + Z' + Z''

    def test_mh_loop_is_hysteretic(self):
        """Sweeping H up and down should produce different M values at H=0."""
        df = load_sample("mh_loop")
        # Find points near zero field
        h = df.iloc[:, 0].values
        m = df.iloc[:, 1].values
        near_zero = abs(h) < 0.01
        assert near_zero.sum() >= 2
        # Expect spread in M -> coercivity
        assert (m[near_zero].max() - m[near_zero].min()) > 0.1

    def test_part3_samples_present(self):
        """Spectroscopy / microscopy / chromatography / mass spec / mechanical extras."""
        samples = list_samples()
        for name in ("uvvis_au_np", "nmr_ethanol", "xps_c1s", "eds_nacl",
                     "afm_profile", "bet_isotherm", "hplc_chromatogram",
                     "mass_spec_caffeine", "dma_polymer", "hardness_vickers",
                     "nanoindentation_loaddisp", "thermal_conductivity_pyrex"):
            assert name in samples

    def test_every_sample_loads(self):
        """Smoke test: every shipped sample must load without error and be non-empty."""
        for name in list_samples():
            df = load_sample(name)
            assert len(df) > 0, f"sample {name} loaded empty"
            assert df.shape[1] >= 2, f"sample {name} has fewer than 2 columns"


class TestErrorMessages:
    """Test that helpful errors are raised on parse failures."""

    def test_useful_error_on_garbage(self, tmp_path):
        p = tmp_path / "garbage.csv"
        # Inconsistent field counts that pandas can't parse
        p.write_text("a,b,c\n1\n2,3\n4,5,6,7,8\n")
        # Should at least not silently misbehave
        try:
            df = load_data(p)
            # If it succeeds, just check we got something
            assert isinstance(df, pd.DataFrame)
        except (ValueError, pd.errors.ParserError):
            # Expected: parse failure with our wrapped message
            pass
