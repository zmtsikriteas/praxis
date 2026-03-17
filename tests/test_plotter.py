"""Tests for the core plotting engine."""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from core.plotter import plot_data, overlay_plots, plot_contour, create_subplots


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestPlotData:
    """Test basic plot creation."""

    def test_line_plot(self):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        fig, ax = plot_data(x, y, kind="line")
        assert fig is not None
        assert ax is not None

    def test_scatter_plot(self):
        x = np.random.randn(30)
        y = np.random.randn(30)
        fig, ax = plot_data(x, y, kind="scatter")
        assert fig is not None

    def test_bar_plot(self):
        x = np.arange(5)
        y = np.array([3, 7, 2, 5, 8])
        fig, ax = plot_data(x, y, kind="bar")
        assert fig is not None

    def test_errorbar_plot(self):
        x = np.arange(5)
        y = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        yerr = np.array([0.1, 0.2, 0.15, 0.1, 0.2])
        fig, ax = plot_data(x, y, kind="errorbar", yerr=yerr)
        assert fig is not None

    def test_step_plot(self):
        x = np.arange(10)
        y = np.cumsum(np.random.randn(10))
        fig, ax = plot_data(x, y, kind="step")
        assert fig is not None

    def test_histogram(self):
        y = np.random.randn(100)
        fig, ax = plot_data(np.zeros(100), y, kind="histogram", bins=20)
        assert fig is not None

    def test_labels_and_title(self):
        x = np.linspace(0, 1, 10)
        y = x ** 2
        fig, ax = plot_data(
            x, y,
            xlabel="Position (mm)",
            ylabel="Force (N)",
            title="Test Plot",
        )
        assert ax.get_xlabel() == "Position (mm)"
        assert ax.get_ylabel() == "Force (N)"
        assert ax.get_title() == "Test Plot"

    def test_log_scales(self):
        x = np.logspace(0, 3, 20)
        y = 1 / x
        fig, ax = plot_data(x, y, log_x=True, log_y=True)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"

    def test_overlay_on_existing_axes(self):
        x = np.linspace(0, 10, 50)
        fig, ax = plot_data(x, np.sin(x), label="sin")
        plot_data(x, np.cos(x), fig=fig, ax=ax, label="cos")
        handles, labels = ax.get_legend_handles_labels()
        assert len(labels) == 2

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown plot kind"):
            plot_data([1, 2], [3, 4], kind="invalid_type")


class TestOverlayPlots:
    """Test multi-dataset overlay."""

    def test_basic_overlay(self):
        x = np.linspace(0, 10, 50)
        datasets = [
            {"x": x, "y": np.sin(x), "label": "sin"},
            {"x": x, "y": np.cos(x), "label": "cos"},
        ]
        fig, ax = overlay_plots(datasets)
        assert fig is not None
        handles, labels = ax.get_legend_handles_labels()
        assert len(labels) == 2


class TestContourPlot:
    """Test contour and heatmap plots."""

    def test_filled_contour(self):
        x = np.linspace(-2, 2, 30)
        y = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-(X**2 + Y**2))
        fig, ax = plot_contour(X, Y, Z, kind="filled")
        assert fig is not None

    def test_heatmap(self):
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        Z = np.random.rand(20, 20)
        fig, ax = plot_contour(x, y, Z, kind="heatmap")
        assert fig is not None


class TestSubplots:
    """Test subplot grid creation."""

    def test_create_2x2(self):
        fig, axes = create_subplots(2, 2)
        assert axes.shape == (2, 2)

    def test_create_1x3(self):
        fig, axes = create_subplots(1, 3)
        assert len(axes) == 3
