"""
grid_model.py — Shared Grid Model for AeroNet Lite
====================================================
This is the CORE DATA MODEL that all 5 modules share.
Every module reads from and writes to this grid.

Cell fields
-----------
row, col          : position in the 10x10 grid
zone              : 'residential' | 'commercial' | 'hospital' |
                    'school' | 'industrial' | 'open_field'
density           : population density (0-10000)
is_hub            : True if this cell is a drone hub
is_charging       : True if this cell has a charging pad
is_medical_pickup : True if this cell is a medical supply pickup point
no_fly            : True blocks all routing through this cell
demand            : estimated delivery demand (set by ML pipeline)

Author: Member 1 — The Architect
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Valid zone names — used for validation in load_from_config
# ---------------------------------------------------------------------------
VALID_ZONES = frozenset({
    "residential", "commercial", "hospital",
    "school", "industrial", "open_field",
})


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    """Represents a single cell in the 10×10 city grid."""

    row: int
    col: int
    zone: str           # must be one of VALID_ZONES
    density: int        # population density 0-10000
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: float = 0.0

    def __post_init__(self):
        if self.zone not in VALID_ZONES:
            raise ValueError(
                f"Cell({self.row},{self.col}): zone '{self.zone}' is not valid. "
                f"Choose from {sorted(VALID_ZONES)}"
            )

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        flags = []
        if self.is_hub:
            flags.append("HUB")
        if self.is_charging:
            flags.append("CHG")
        if self.is_medical_pickup:
            flags.append("MED")
        if self.no_fly:
            flags.append("NFZ")
        tag = f" [{','.join(flags)}]" if flags else ""
        return f"Cell({self.row},{self.col})-{self.zone}{tag}"

    def to_dict(self) -> Dict:
        """Serialise to plain dict (JSON-safe)."""
        return {
            "row": self.row,
            "col": self.col,
            "zone": self.zone,
            "density": self.density,
            "is_hub": self.is_hub,
            "is_charging": self.is_charging,
            "is_medical_pickup": self.is_medical_pickup,
            "no_fly": self.no_fly,
            "demand": self.demand,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Cell":
        """Deserialise from a plain dict."""
        return cls(
            row=data["row"],
            col=data["col"],
            zone=data["zone"],
            density=data.get("density", 0),
            is_hub=data.get("is_hub", False),
            is_charging=data.get("is_charging", False),
            is_medical_pickup=data.get("is_medical_pickup", False),
            no_fly=data.get("no_fly", False),
            demand=data.get("demand", 0.0),
        )


# ---------------------------------------------------------------------------
# Drone
# ---------------------------------------------------------------------------
@dataclass
class Drone:
    """
    Represents a single drone in the fleet.

    Drone types
    -----------
    light : cost=1000, payload=2 kg, max_range=12 cells
    heavy : cost=1800, payload=5 kg, max_range=20 cells
    """

    id: str
    drone_type: str          # 'light' or 'heavy'
    cost: int
    payload: float           # kg
    max_range: int           # cells
    location: Tuple[int, int]   # current (row, col)
    home_hub: Tuple[int, int]   # starting hub (row, col)
    current_delivery: Optional["Delivery"] = None
    route: List[Tuple[int, int]] = field(default_factory=list)
    route_index: int = 0
    battery: float = 100.0   # %
    status: str = "idle"     # 'idle' | 'en_route' | 'returning'

    # Drone type catalogue ------------------------------------------------
    LIGHT = {"drone_type": "light", "cost": 1000, "payload": 2.0, "max_range": 12}
    HEAVY = {"drone_type": "heavy", "cost": 1800, "payload": 5.0, "max_range": 20}

    def __str__(self) -> str:
        return f"Drone-{self.id}({self.drone_type}@{self.location})"

    def is_available(self) -> bool:
        """True if the drone has no active delivery."""
        return self.current_delivery is None and self.status == "idle"

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "drone_type": self.drone_type,
            "cost": self.cost,
            "payload": self.payload,
            "max_range": self.max_range,
            "location": list(self.location),
            "home_hub": list(self.home_hub),
            "battery": self.battery,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------
@dataclass
class Delivery:
    """Represents a single delivery task."""

    id: str
    pickup: Tuple[int, int]     # source (row, col)
    dropoff: Tuple[int, int]    # destination (row, col)
    weight: float               # kg
    assigned_drone: Optional[str] = None
    route: List[Tuple[int, int]] = field(default_factory=list)
    status: str = "pending"     # 'pending'|'assigned'|'in_progress'|'completed'|'failed'|'delayed'
    total_cost: float = 0.0

    def __str__(self) -> str:
        return f"Delivery-{self.id}({self.pickup}->{self.dropoff}) [{self.status}]"

    def next_waypoint(self) -> Optional[Tuple[int, int]]:
        """
        Return the next target cell in the route that has not yet been reached.
        Used by the disruption handler to find where to reroute to.
        """
        if not self.route:
            return None
        # The assigned drone's route_index is tracked on the Drone side;
        # here we just expose the last cell as the end-goal.
        return self.route[-1]


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
class Grid:
    """
    The 10×10 grid that represents the city.
    This is the SINGLE SOURCE OF TRUTH for the entire simulation.

    Access pattern
    --------------
    grid.get_cell(row, col)   -> Cell or None
    grid.set_cell(row, col, zone='commercial', is_hub=True, ...)
    grid.get_neighbors(row, col)  -> List[Cell]  (4-directional)
    grid.get_cells_by_type(...)   -> List[Cell]
    grid.manhattan_distance(pos1, pos2) -> int
    """

    def __init__(self, size: int = 10):
        self.size: int = size
        self.cells: List[List[Cell]] = []
        self._initialize_empty_grid()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initialize_empty_grid(self):
        """Fill the grid with default open_field cells."""
        self.cells = [
            [
                Cell(row=r, col=c, zone="open_field", density=0)
                for c in range(self.size)
            ]
            for r in range(self.size)
        ]

    # ------------------------------------------------------------------
    # Cell access
    # ------------------------------------------------------------------
    def get_cell(self, row: int, col: int) -> Optional[Cell]:
        """Return Cell at (row, col), or None if out of bounds."""
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.cells[row][col]
        return None

    def set_cell(self, row: int, col: int, **kwargs):
        """
        Update one or more properties of the cell at (row, col).

        Example
        -------
        grid.set_cell(3, 4, zone='hospital', is_hub=True)
        """
        cell = self.get_cell(row, col)
        if cell is None:
            raise IndexError(f"Cell ({row},{col}) is out of bounds for a {self.size}×{self.size} grid.")
        for key, value in kwargs.items():
            if not hasattr(cell, key):
                raise AttributeError(f"Cell has no attribute '{key}'.")
            # Re-validate zone string if being changed
            if key == "zone" and value not in VALID_ZONES:
                raise ValueError(f"Zone '{value}' is not valid. Choose from {sorted(VALID_ZONES)}.")
            setattr(cell, key, value)

    # ------------------------------------------------------------------
    # Neighbour & distance helpers
    # ------------------------------------------------------------------
    def get_neighbors(self, row: int, col: int) -> List[Cell]:
        """Return up to 4 adjacent cells (up, down, left, right)."""
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = self.get_cell(row + dr, col + dc)
            if nb is not None:
                result.append(nb)
        return result

    @staticmethod
    def manhattan_distance(
        pos1: Tuple[int, int], pos2: Tuple[int, int]
    ) -> int:
        """Return Manhattan distance between two (row, col) positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------
    def get_cells_by_type(
        self,
        zone: Optional[str] = None,
        is_hub: Optional[bool] = None,
        is_charging: Optional[bool] = None,
        is_medical_pickup: Optional[bool] = None,
        no_fly: Optional[bool] = None,
    ) -> List[Cell]:
        """
        Return all cells that match ALL of the supplied filters.
        Pass None to ignore a filter.
        """
        result = []
        for row in self.cells:
            for cell in row:
                if zone is not None and cell.zone != zone:
                    continue
                if is_hub is not None and cell.is_hub != is_hub:
                    continue
                if is_charging is not None and cell.is_charging != is_charging:
                    continue
                if is_medical_pickup is not None and cell.is_medical_pickup != is_medical_pickup:
                    continue
                if no_fly is not None and cell.no_fly != no_fly:
                    continue
                result.append(cell)
        return result

    def get_hub_positions(self) -> List[Tuple[int, int]]:
        """Convenience: return (row, col) for every hub cell."""
        return [(c.row, c.col) for c in self.get_cells_by_type(is_hub=True)]

    def get_charging_positions(self) -> List[Tuple[int, int]]:
        """Convenience: return (row, col) for every charging pad cell."""
        return [(c.row, c.col) for c in self.get_cells_by_type(is_charging=True)]

    def total_demand(self) -> float:
        """Sum of demand across all cells."""
        return sum(cell.demand for row in self.cells for cell in row)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def load_from_config(self, config_path: str):
        """
        Load grid layout from a JSON file.

        JSON schema
        -----------
        {
          "size": 10,
          "cells": [ { <Cell fields> }, ... ]
        }
        Only cells that differ from the default open_field/density=0 need
        to appear in the 'cells' list.
        """
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

        declared_size = config.get("size", self.size)
        if declared_size != self.size:
            raise ValueError(
                f"Config declares size={declared_size} but Grid was initialised with size={self.size}."
            )

        for cell_data in config.get("cells", []):
            r, c = cell_data["row"], cell_data["col"]
            props = {k: v for k, v in cell_data.items() if k not in ("row", "col")}
            self.set_cell(r, c, **props)

    def save_to_config(self, config_path: str):
        """
        Save the current grid to a JSON file.
        Only non-default cells are written to keep the file compact.
        """
        cells_to_save = []
        for row in self.cells:
            for cell in row:
                is_default = (
                    cell.zone == "open_field"
                    and not cell.is_hub
                    and not cell.is_charging
                    and not cell.is_medical_pickup
                    and not cell.no_fly
                    and cell.density == 0
                    and cell.demand == 0.0
                )
                if not is_default:
                    cells_to_save.append(cell.to_dict())

        config = {"size": self.size, "cells": cells_to_save}
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return f"Grid({self.size}×{self.size})"

    def __repr__(self) -> str:
        hubs = len(self.get_hub_positions())
        return f"Grid(size={self.size}, hubs={hubs})"


# ---------------------------------------------------------------------------
# Sample grid factory
# ---------------------------------------------------------------------------
def create_sample_grid() -> Grid:
    """
    Build a realistic 10×10 city grid for testing and demo purposes.

    Layout summary
    --------------
    - Mixed zones: residential, commercial, hospital, school, industrial, open_field
    - 3 drone hubs at (1,1), (4,4), (8,6)
    - 3 charging pads near hubs: (1,2), (4,5), (8,7)
    - 1 hospital at (2,4) with medical pickup at the same cell
    - Industrial zone far from schools/hospitals (CSP R1 satisfied)
    - All residential cells within 3 Manhattan cells of a hub (R2 ✓)
    - All hubs within 2 cells of a charging pad (R3 ✓)
    - Hospital (2,4) has medical pickup (R4 ✓)
    """
    grid = Grid(size=10)

    # Zone layout — row-major, 0-indexed
    zone_layout = [
        # col: 0              1              2              3              4              5              6              7              8              9
        ["residential", "residential", "residential", "commercial",  "commercial",  "commercial",  "residential", "residential", "residential", "open_field"],  # row 0
        ["residential", "residential", "school",      "commercial",  "commercial",  "commercial",  "residential", "residential", "residential", "open_field"],  # row 1
        ["residential", "residential", "residential", "commercial",  "hospital",    "commercial",  "residential", "residential", "residential", "open_field"],  # row 2
        ["residential", "residential", "residential", "commercial",  "commercial",  "commercial",  "industrial",  "industrial",  "open_field",  "open_field"],  # row 3
        ["residential", "residential", "residential", "residential", "residential", "residential", "industrial",  "industrial",  "open_field",  "open_field"],  # row 4
        ["residential", "residential", "residential", "residential", "residential", "residential", "industrial",  "industrial",  "open_field",  "open_field"],  # row 5
        ["residential", "school",      "residential", "residential", "residential", "residential", "open_field",  "open_field",  "open_field",  "open_field"],  # row 6
        ["residential", "residential", "residential", "residential", "residential", "residential", "open_field",  "open_field",  "open_field",  "open_field"],  # row 7
        ["residential", "residential", "residential", "commercial",  "commercial",  "commercial",  "residential", "residential", "open_field",  "open_field"],  # row 8
        ["residential", "residential", "residential", "commercial",  "commercial",  "commercial",  "residential", "residential", "open_field",  "open_field"],  # row 9
    ]

    # Apply zones
    for r, row in enumerate(zone_layout):
        for c, zone in enumerate(row):
            grid.set_cell(r, c, zone=zone)

    # Density per zone type
    density_map = {
        "residential": 5000,
        "commercial":  3000,
        "hospital":    2000,
        "school":      1500,
        "industrial":  1000,
        "open_field":  0,
    }
    for row in grid.cells:
        for cell in row:
            cell.density = density_map[cell.zone]
            # demand = density scaled to 0-1 range (max density is 5000)
            cell.demand = round(density_map[cell.zone] / 5000.0, 4)

    # Drone hubs — 6 hubs to cover ALL residential cells within distance 3
    # Original 3: (1,1), (4,4), (8,6)
    # Added 3:    (0,7) covers top-right residential block
    #             (6,1) covers bottom-left residential block  
    #             (8,3) covers bottom-left strip
    for pos in [(1, 1), (4, 4), (8, 6), (0, 7), (7, 1), (8, 3), (4, 1)]:
        grid.set_cell(*pos, is_hub=True)

    # Charging pads — each within 2 Manhattan cells of its nearest hub
    for pos in [(1, 2), (4, 5), (8, 7), (0, 8), (7, 2), (8, 4), (4, 0)]:
        grid.set_cell(*pos, is_charging=True)

    # Hospital + medical pickup (same cell satisfies R4, distance = 0)
    grid.set_cell(2, 4, is_medical_pickup=True)

    return grid


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== AeroNet Lite — Grid Model Self-Test ===\n")

    grid = create_sample_grid()
    print(f"Grid:            {grid}")
    print(f"Hubs:            {grid.get_hub_positions()}")
    print(f"Charging pads:   {grid.get_charging_positions()}")
    print(f"Residential:     {len(grid.get_cells_by_type(zone='residential'))} cells")
    print(f"Industrial:      {len(grid.get_cells_by_type(zone='industrial'))} cells")
    print(f"Hospital:        {len(grid.get_cells_by_type(zone='hospital'))} cells")

    # Neighbour test
    nb = grid.get_neighbors(1, 1)
    print(f"\nNeighbours of (1,1): {[(c.row, c.col, c.zone) for c in nb]}")

    # Manhattan test
    d = Grid.manhattan_distance((0, 0), (3, 4))
    print(f"Manhattan (0,0)→(3,4): {d}")

    # Serialisation round-trip
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    grid.save_to_config(tmp_path)
    grid2 = Grid()
    grid2.load_from_config(tmp_path)
    assert grid2.get_cell(1, 1).is_hub, "Hub not restored after round-trip!"
    assert grid2.get_cell(2, 4).is_medical_pickup, "Medical pickup not restored!"
    os.unlink(tmp_path)
    print("\n✓ JSON round-trip passed")
    print("✓ All self-tests passed!")