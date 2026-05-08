"""
test_grid_model.py — Unit tests for grid_model.py
==================================================
Run with:   python -m pytest tests/test_grid_model.py -v
Or:         python tests/test_grid_model.py

Author: Member 1 — The Architect
"""

import json
import os
import sys
import tempfile

# Ensure src/ is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grid_model import Cell, Delivery, Drone, Grid, VALID_ZONES, create_sample_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_grid() -> Grid:
    return create_sample_grid()


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------
class TestCell:
    def test_default_flags_are_false(self):
        cell = Cell(row=0, col=0, zone="residential", density=1000)
        assert cell.is_hub is False
        assert cell.is_charging is False
        assert cell.is_medical_pickup is False
        assert cell.no_fly is False
        assert cell.demand == 0.0

    def test_invalid_zone_raises(self):
        try:
            Cell(row=0, col=0, zone="fantasy_land", density=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_to_dict_round_trip(self):
        cell = Cell(row=3, col=7, zone="hospital", density=2000,
                    is_medical_pickup=True, demand=42.5)
        d = cell.to_dict()
        cell2 = Cell.from_dict(d)
        assert cell2.row == 3
        assert cell2.col == 7
        assert cell2.zone == "hospital"
        assert cell2.is_medical_pickup is True
        assert cell2.demand == 42.5

    def test_str_includes_flags(self):
        cell = Cell(row=1, col=1, zone="residential", density=5000,
                    is_hub=True, is_charging=True)
        s = str(cell)
        assert "HUB" in s
        assert "CHG" in s


# ---------------------------------------------------------------------------
# Grid initialisation tests
# ---------------------------------------------------------------------------
class TestGridInit:
    def test_default_size(self):
        g = Grid()
        assert g.size == 10
        assert len(g.cells) == 10
        assert all(len(row) == 10 for row in g.cells)

    def test_all_cells_are_open_field_by_default(self):
        g = Grid()
        for row in g.cells:
            for cell in row:
                assert cell.zone == "open_field"
                assert cell.density == 0

    def test_sample_grid_has_correct_hubs(self):
        g = make_grid()
        hubs = g.get_hub_positions()
        assert (1, 1) in hubs
        assert (4, 4) in hubs
        assert (8, 6) in hubs
        assert len(hubs) == 3

    def test_sample_grid_has_correct_charging_pads(self):
        g = make_grid()
        pads = g.get_charging_positions()
        assert (1, 2) in pads
        assert (4, 5) in pads
        assert (8, 7) in pads

    def test_sample_grid_has_medical_pickup(self):
        g = make_grid()
        med = g.get_cells_by_type(is_medical_pickup=True)
        assert len(med) >= 1
        assert med[0].zone == "hospital"


# ---------------------------------------------------------------------------
# get_cell tests
# ---------------------------------------------------------------------------
class TestGetCell:
    def test_valid_cell(self):
        g = Grid()
        cell = g.get_cell(0, 0)
        assert cell is not None
        assert cell.row == 0 and cell.col == 0

    def test_out_of_bounds_returns_none(self):
        g = Grid()
        assert g.get_cell(-1, 0) is None
        assert g.get_cell(0, 10) is None
        assert g.get_cell(10, 10) is None


# ---------------------------------------------------------------------------
# set_cell tests
# ---------------------------------------------------------------------------
class TestSetCell:
    def test_set_zone(self):
        g = Grid()
        g.set_cell(5, 5, zone="hospital")
        assert g.get_cell(5, 5).zone == "hospital"

    def test_set_multiple_flags(self):
        g = Grid()
        g.set_cell(3, 3, zone="residential", is_hub=True, density=4500)
        cell = g.get_cell(3, 3)
        assert cell.is_hub is True
        assert cell.density == 4500

    def test_invalid_zone_raises(self):
        g = Grid()
        try:
            g.set_cell(0, 0, zone="not_a_zone")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_attribute_raises(self):
        g = Grid()
        try:
            g.set_cell(0, 0, does_not_exist=True)
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass

    def test_out_of_bounds_raises(self):
        g = Grid()
        try:
            g.set_cell(99, 99, zone="residential")
            assert False, "Should have raised IndexError"
        except IndexError:
            pass


# ---------------------------------------------------------------------------
# get_neighbors tests
# ---------------------------------------------------------------------------
class TestGetNeighbors:
    def test_corner_has_2_neighbors(self):
        g = Grid()
        nb = g.get_neighbors(0, 0)
        assert len(nb) == 2

    def test_edge_has_3_neighbors(self):
        g = Grid()
        nb = g.get_neighbors(0, 5)
        assert len(nb) == 3

    def test_interior_has_4_neighbors(self):
        g = Grid()
        nb = g.get_neighbors(5, 5)
        assert len(nb) == 4

    def test_neighbor_positions_correct(self):
        g = Grid()
        nb = g.get_neighbors(1, 1)
        positions = {(c.row, c.col) for c in nb}
        assert positions == {(0, 1), (2, 1), (1, 0), (1, 2)}


# ---------------------------------------------------------------------------
# manhattan_distance tests
# ---------------------------------------------------------------------------
class TestManhattanDistance:
    def test_same_cell(self):
        assert Grid.manhattan_distance((3, 3), (3, 3)) == 0

    def test_adjacent(self):
        assert Grid.manhattan_distance((0, 0), (0, 1)) == 1
        assert Grid.manhattan_distance((0, 0), (1, 0)) == 1

    def test_diagonal(self):
        assert Grid.manhattan_distance((0, 0), (3, 4)) == 7

    def test_symmetry(self):
        a, b = (2, 5), (7, 1)
        assert Grid.manhattan_distance(a, b) == Grid.manhattan_distance(b, a)


# ---------------------------------------------------------------------------
# get_cells_by_type tests
# ---------------------------------------------------------------------------
class TestGetCellsByType:
    def test_filter_by_zone(self):
        g = make_grid()
        hospitals = g.get_cells_by_type(zone="hospital")
        assert len(hospitals) == 1
        assert hospitals[0].zone == "hospital"

    def test_filter_by_hub_flag(self):
        g = make_grid()
        hubs = g.get_cells_by_type(is_hub=True)
        assert len(hubs) == 3

    def test_filter_no_match(self):
        g = Grid()  # empty grid, all open_field
        result = g.get_cells_by_type(zone="hospital")
        assert result == []

    def test_combined_filter(self):
        g = make_grid()
        result = g.get_cells_by_type(zone="residential", is_hub=True)
        # Hub at (1,1), (4,4), (8,6) are all residential
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Serialisation tests
# ---------------------------------------------------------------------------
class TestSerialisation:
    def _round_trip(self, grid: Grid) -> Grid:
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name
        try:
            grid.save_to_config(tmp_path)
            g2 = Grid()
            g2.load_from_config(tmp_path)
            return g2
        finally:
            os.unlink(tmp_path)

    def test_hubs_survive_round_trip(self):
        g = make_grid()
        g2 = self._round_trip(g)
        assert g2.get_cell(1, 1).is_hub is True
        assert g2.get_cell(4, 4).is_hub is True
        assert g2.get_cell(8, 6).is_hub is True

    def test_medical_pickup_survives_round_trip(self):
        g = make_grid()
        g2 = self._round_trip(g)
        assert g2.get_cell(2, 4).is_medical_pickup is True

    def test_zone_survives_round_trip(self):
        g = make_grid()
        g2 = self._round_trip(g)
        assert g2.get_cell(2, 4).zone == "hospital"
        assert g2.get_cell(3, 6).zone == "industrial"

    def test_load_from_json_file(self):
        """Load the actual sample_grid.json shipped with the project."""
        json_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "sample_grid.json"
        )
        if not os.path.exists(json_path):
            print(f"Skipping: {json_path} not found")
            return
        g = Grid()
        g.load_from_config(json_path)
        hubs = g.get_hub_positions()
        assert (1, 1) in hubs

    def test_size_mismatch_raises(self):
        """Loading a size-5 config into a size-10 Grid should raise."""
        bad_config = {"size": 5, "cells": []}
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            json.dump(bad_config, tmp)
            tmp_path = tmp.name
        try:
            g = Grid(size=10)
            try:
                g.load_from_config(tmp_path)
                assert False, "Should have raised ValueError"
            except ValueError:
                pass
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Drone & Delivery smoke tests
# ---------------------------------------------------------------------------
class TestDroneDelivery:
    def test_drone_is_available_when_idle(self):
        d = Drone(
            id="D1", drone_type="light", cost=1000,
            payload=2.0, max_range=12,
            location=(1, 1), home_hub=(1, 1),
        )
        assert d.is_available() is True

    def test_delivery_str(self):
        dlv = Delivery(id="DEL-001", pickup=(0, 0), dropoff=(9, 9), weight=1.5)
        s = str(dlv)
        assert "DEL-001" in s
        assert "pending" in s


# ---------------------------------------------------------------------------
# Runner for direct execution
# ---------------------------------------------------------------------------
def _run_all():
    """Simple manual test runner (no pytest required)."""
    test_classes = [
        TestCell, TestGridInit, TestGetCell, TestSetCell,
        TestGetNeighbors, TestManhattanDistance,
        TestGetCellsByType, TestSerialisation, TestDroneDelivery,
    ]

    total = 0
    failed = 0
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

    print(f"\n{'='*50}")
    print(f"Results: {total - failed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  -- ALL PASSED")


if __name__ == "__main__":
    _run_all()
