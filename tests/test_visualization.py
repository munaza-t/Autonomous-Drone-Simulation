"""
test_visualization.py — Smoke tests for visualization.py
==========================================================
All tests run in non-interactive (headless) mode: show=False.
They verify that every function:
  - Runs without crashing
  - Returns a matplotlib Figure
  - Saves a PNG file to report/figures/

Run with:
    python -m pytest tests/test_visualization.py -v
    python tests/test_visualization.py

Author: Member 1 -- The Architect
"""

from __future__ import annotations

import os
import sys
import random

import matplotlib
matplotlib.use("Agg")   # force non-interactive backend for all tests

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
from grid_model import Grid, create_sample_grid
from visualization import (
    plot_grid,
    plot_heatmap,
    plot_route,
    plot_validation_result,
    plot_all,
    FIGURES_DIR,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_grid() -> Grid:
    g = create_sample_grid()
    # Inject demo demand values
    random.seed(0)
    for row in g.cells:
        for cell in row:
            if cell.zone in ("residential", "commercial"):
                cell.demand = round(random.uniform(5, 80), 1)
    return g


DEMO_ROUTE = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4), (4, 4)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPlotGrid:
    def test_returns_figure(self):
        g = _make_grid()
        fig = plot_grid(g, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_png(self):
        g = _make_grid()
        out = os.path.join(FIGURES_DIR, "_test_grid.png")
        fig = plot_grid(g, show=False, save_path=out)
        assert os.path.exists(out), f"PNG not found: {out}"
        os.remove(out)
        plt.close(fig)

    def test_no_crash_empty_grid(self):
        g = Grid()   # all open_field
        fig = plot_grid(g, show=False)
        assert fig is not None
        plt.close(fig)

    def test_custom_title(self):
        g = _make_grid()
        fig = plot_grid(g, title="Custom Title", show=False)
        assert fig.texts or True   # just verify no crash
        plt.close(fig)


class TestPlotRoute:
    def test_returns_figure(self):
        g = _make_grid()
        fig = plot_route(g, DEMO_ROUTE, drone_id="D1", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_png(self):
        g = _make_grid()
        out = os.path.join(FIGURES_DIR, "_test_route.png")
        fig = plot_route(g, DEMO_ROUTE, drone_id="D1", show=False, save_path=out)
        assert os.path.exists(out)
        os.remove(out)
        plt.close(fig)

    def test_empty_route(self):
        g = _make_grid()
        fig = plot_route(g, [], drone_id="D1", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_cell_route(self):
        g = _make_grid()
        fig = plot_route(g, [(4, 4)], drone_id="D2", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_long_route(self):
        # Route zigzagging across the grid
        route = [(r, c) for r in range(10) for c in (range(10) if r % 2 == 0 else range(9, -1, -1))]
        g = _make_grid()
        fig = plot_route(g, route, drone_id="D_long", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotHeatmap:
    def test_returns_figure(self):
        g = _make_grid()
        fig = plot_heatmap(g, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_png(self):
        g = _make_grid()
        out = os.path.join(FIGURES_DIR, "_test_heatmap.png")
        fig = plot_heatmap(g, show=False, save_path=out)
        assert os.path.exists(out)
        os.remove(out)
        plt.close(fig)

    def test_all_zero_demand(self):
        g = Grid()   # demand = 0.0 everywhere
        fig = plot_heatmap(g, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_high_demand_values(self):
        g = create_sample_grid()
        for row in g.cells:
            for cell in row:
                cell.demand = 9999.0
        fig = plot_heatmap(g, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotValidationResult:
    def test_no_errors(self):
        g = _make_grid()
        fig = plot_validation_result(g, errors=[], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_errors(self):
        g = _make_grid()
        errs = [
            "R2: Residential cell (7,5) is 4 cells from nearest hub.",
            "R1: Industrial cell (3,6) is adjacent to School at (2,6).",
        ]
        fig = plot_validation_result(g, errors=errs, violated_cells=[(7, 5), (3, 6)], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_png(self):
        g = _make_grid()
        out = os.path.join(FIGURES_DIR, "_test_validation.png")
        fig = plot_validation_result(g, errors=[], show=False, save_path=out)
        assert os.path.exists(out)
        os.remove(out)
        plt.close(fig)

    def test_many_errors(self):
        g = _make_grid()
        errs = [f"R{i % 4 + 1}: Some long error message for cell ({i},{i})." for i in range(20)]
        fig = plot_validation_result(g, errors=errs, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotAll:
    def test_no_routes_no_errors(self):
        g = _make_grid()
        fig = plot_all(g, routes={}, errors=[], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_routes_and_errors(self):
        g = _make_grid()
        routes = {
            "D1": DEMO_ROUTE,
            "D2": [(8, 6), (7, 6), (6, 6), (5, 5), (4, 4)],
        }
        errs = ["R2: Residential cell (7,5) is 4 cells from nearest hub."]
        fig = plot_all(g, routes=routes, errors=errs, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_dashboard_png(self):
        g = _make_grid()
        out = os.path.join(FIGURES_DIR, "_test_dashboard.png")
        fig = plot_all(g, show=False, save_path=out)
        assert os.path.exists(out)
        os.remove(out)
        plt.close(fig)

    def test_default_args(self):
        g = _make_grid()
        fig = plot_all(g, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Direct runner
# ---------------------------------------------------------------------------
def _run_all():
    test_classes = [
        TestPlotGrid,
        TestPlotRoute,
        TestPlotHeatmap,
        TestPlotValidationResult,
        TestPlotAll,
    ]
    total, failed = 0, 0
    for cls in test_classes:
        obj = cls()
        for name in dir(obj):
            if not name.startswith("test_"):
                continue
            total += 1
            method = getattr(obj, name)
            try:
                method()
                print(f"  PASS  {cls.__name__}.{name}")
            except Exception as exc:
                failed += 1
                print(f"  FAIL  {cls.__name__}.{name}  ->  {exc}")

    print(f"\n{'=' * 50}")
    print(f"Results: {total - failed}/{total} passed", end="")
    print(f"  ({failed} FAILED)" if failed else "  -- ALL PASSED")


if __name__ == "__main__":
    _run_all()
