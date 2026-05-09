"""
layout_validator.py
===================
Member 2 - "The Constraint Whisperer"
Module 1: CSP Layout Validator for AeroNet Lite

Validates the 10x10 city grid against four hard operating constraints
(R1–R4) using a Constraint Satisfaction Problem (CSP) approach.
Each rule is an independent constraint function; errors are collected
and printed as a professional validation report.

Constraint Rules
----------------
R1  Industrial cells must NOT be directly adjacent to Schools or Hospitals.
R2  Every Residential cell must be within Manhattan distance 3 of a Hub.
R3  Every Hub must have at least one Charging Pad within Manhattan distance 2.
R4  At least one Hospital must have a Medical Pickup point within distance 1.

Integration
-----------
Expects `grid` as a 10×10 list-of-lists of cell dicts (from grid_model.py).
Each cell dict must contain at minimum:
    row, col, zone, is_hub, is_charging, is_medical_pickup
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any

# ─────────────────────────────────────────────
#  Type alias
# ─────────────────────────────────────────────
Grid = List[List[Dict[str, Any]]]
Cell = Dict[str, Any]

GRID_SIZE = 10

# Zone name constants (case-insensitive comparison used throughout)
INDUSTRIAL = "industrial"
RESIDENTIAL = "residential"
HOSPITAL = "hospital"
SCHOOL = "school"


# ══════════════════════════════════════════════
#  Helper Utilities
# ══════════════════════════════════════════════

def get_neighbors(row: int, col: int) -> List[Tuple[int, int]]:
    """
    Return the (row, col) of all valid 4-directional neighbours
    (up, down, left, right) that lie within the 10×10 grid.
    """
    candidates = [
        (row - 1, col),  # up
        (row + 1, col),  # down
        (row, col - 1),  # left
        (row, col + 1),  # right
    ]
    return [(r, c) for r, c in candidates if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE]


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Return the Manhattan (L1) distance between two grid coordinates.

    Manhattan distance is admissible for 4-directional grid movement:
    it never overestimates the true shortest path.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _zone(cell: Cell) -> str:
    """Normalise zone string to lowercase for comparison."""
    return str(cell.get("zone", "")).lower()


def _all_cells(grid: Grid) -> List[Cell]:
    """Flatten the 2-D grid into a 1-D list of cells."""
    return [grid[r][c] for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


# ══════════════════════════════════════════════
#  Constraint Functions
# ══════════════════════════════════════════════

def check_industrial_safety(grid: Grid) -> Tuple[bool, List[str]]:
    """
    R1 — Industrial Safety Constraint
    ----------------------------------
    No Industrial cell may be directly adjacent (4-connected) to a
    School or Hospital cell.

    Returns
    -------
    passed : bool   True if ALL industrial cells satisfy the rule.
    errors : list   Human-readable violation messages.
    """
    errors: List[str] = []

    for cell in _all_cells(grid):
        if _zone(cell) != INDUSTRIAL:
            continue

        pos = (cell["row"], cell["col"])
        for nr, nc in get_neighbors(*pos):
            neighbour_zone = _zone(grid[nr][nc])
            if neighbour_zone in (HOSPITAL, SCHOOL):
                msg = (
                    f"  R1 VIOLATION: Industrial cell {pos} is adjacent to "
                    f"{neighbour_zone.capitalize()} cell ({nr}, {nc}). "
                    f"Suggested fix: add a buffer (Open Field or Commercial) "
                    f"between {pos} and ({nr}, {nc})."
                )
                errors.append(msg)

    return len(errors) == 0, errors


def check_residential_coverage(grid: Grid) -> Tuple[bool, List[str]]:
    """
    R2 — Residential Hub Coverage Constraint
    -----------------------------------------
    Every Residential cell must be within Manhattan distance 3 of at
    least one Drone Hub.

    Returns
    -------
    passed : bool
    errors : list
    """
    errors: List[str] = []

    # Vacuous case: no residential cells means rule is trivially satisfied
    residential_cells = [c for c in _all_cells(grid) if _zone(c) == RESIDENTIAL]
    if not residential_cells:
        return True, []

    # Pre-collect all hub positions for efficiency
    hub_positions: List[Tuple[int, int]] = [
        (cell["row"], cell["col"])
        for cell in _all_cells(grid)
        if cell.get("is_hub", False)
    ]

    if not hub_positions:
        errors.append(
            f"  R2 VIOLATION: No Drone Hubs found, but {len(residential_cells)} "
            "Residential cell(s) need coverage. Add at least one hub."
        )
        return False, errors

    for cell in _all_cells(grid):
        if _zone(cell) != RESIDENTIAL:
            continue

        pos = (cell["row"], cell["col"])
        nearest_hub_dist = min(manhattan(pos, h) for h in hub_positions)

        if nearest_hub_dist > 3:
            nearest_hub = min(hub_positions, key=lambda h: manhattan(pos, h))
            suggested_row = (pos[0] + nearest_hub[0]) // 2
            suggested_col = (pos[1] + nearest_hub[1]) // 2
            msg = (
                f"  R2 VIOLATION: Residential cell {pos} is {nearest_hub_dist} cells "
                f"from the nearest hub {nearest_hub} (max allowed: 3). "
                f"Suggested fix: add a hub near ({suggested_row}, {suggested_col}) "
                f"or convert {pos} to Open Field."
            )
            errors.append(msg)

    return len(errors) == 0, errors


def check_hub_charging(grid: Grid) -> Tuple[bool, List[str]]:
    """
    R3 — Hub Charging Pad Constraint
    ---------------------------------
    Every Drone Hub must have at least one Charging Pad within
    Manhattan distance 2.

    Returns
    -------
    passed : bool
    errors : list
    """
    errors: List[str] = []

    # Pre-collect all charging pad positions
    charging_positions: List[Tuple[int, int]] = [
        (cell["row"], cell["col"])
        for cell in _all_cells(grid)
        if cell.get("is_charging", False)
    ]

    for cell in _all_cells(grid):
        if not cell.get("is_hub", False):
            continue

        pos = (cell["row"], cell["col"])

        if not charging_positions:
            errors.append(
                f"  R3 VIOLATION: Hub at {pos} has no charging pads anywhere "
                f"in the grid. Add at least one charging pad adjacent to each hub."
            )
            continue

        nearest_charger_dist = min(manhattan(pos, cp) for cp in charging_positions)

        if nearest_charger_dist > 2:
            nearest_cp = min(charging_positions, key=lambda cp: manhattan(pos, cp))
            msg = (
                f"  R3 VIOLATION: Hub at {pos} has no Charging Pad within distance 2 "
                f"(nearest pad is {nearest_cp} at distance {nearest_charger_dist}). "
                f"Suggested fix: add a charging pad at one of: "
                f"{get_neighbors(*pos)[:2]}."
            )
            errors.append(msg)

    return len(errors) == 0, errors


def check_medical_access(grid: Grid) -> Tuple[bool, List[str]]:
    """
    R4 — Medical Access Constraint
    --------------------------------
    At least ONE Hospital cell must have a Medical Pickup point within
    Manhattan distance 1 (i.e., directly adjacent or co-located).

    Returns
    -------
    passed : bool
    errors : list
    """
    errors: List[str] = []

    hospital_positions: List[Tuple[int, int]] = [
        (cell["row"], cell["col"])
        for cell in _all_cells(grid)
        if _zone(cell) == HOSPITAL
    ]

    medical_pickup_positions: List[Tuple[int, int]] = [
        (cell["row"], cell["col"])
        for cell in _all_cells(grid)
        if cell.get("is_medical_pickup", False)
    ]

    if not hospital_positions:
        # No hospitals — rule vacuously satisfied (nothing to violate)
        return True, []

    if not medical_pickup_positions:
        errors.append(
            "  R4 VIOLATION: No Medical Pickup points exist anywhere in the grid. "
            "Add at least one medical pickup cell adjacent to a hospital."
        )
        return False, errors

    # Check whether at least one hospital has a pickup within distance 1
    satisfied = False
    hospital_details: List[str] = []

    for h_pos in hospital_positions:
        nearest_pickup_dist = min(manhattan(h_pos, mp) for mp in medical_pickup_positions)
        if nearest_pickup_dist <= 1:
            satisfied = True
            break
        else:
            nearest_mp = min(medical_pickup_positions, key=lambda mp: manhattan(h_pos, mp))
            hospital_details.append(
                f"    Hospital {h_pos}: nearest pickup is {nearest_mp} "
                f"at distance {nearest_pickup_dist}"
            )

    if not satisfied:
        errors.append(
            "  R4 VIOLATION: No Hospital has a Medical Pickup point within distance 1.\n"
            + "\n".join(hospital_details)
            + "\n  Suggested fix: add a medical_pickup cell adjacent to any hospital, "
            f"e.g. at {get_neighbors(*hospital_positions[0])[0]}."
        )

    return len(errors) == 0, errors


# ══════════════════════════════════════════════
#  Master Validator — runs all 4 rules
# ══════════════════════════════════════════════

def validate_layout(grid: Grid, verbose: bool = True) -> Dict[str, Any]:
    """
    Run all four CSP constraints on the grid and produce a
    professional validation report.

    Parameters
    ----------
    grid    : 10×10 list-of-lists of cell dicts (from grid_model.py)
    verbose : if True, print the full report to stdout

    Returns
    -------
    result : dict with keys
        'valid'        – bool: True only if ALL four rules passed
        'passed_rules' – list of rule IDs that passed
        'failed_rules' – list of rule IDs that failed
        'errors'       – dict mapping rule_id -> list of error strings
        'summary'      – one-line summary string
    """
    rules = {
        "R1 (Industrial Safety)":     check_industrial_safety,
        "R2 (Residential Coverage)":  check_residential_coverage,
        "R3 (Hub Charging)":          check_hub_charging,
        "R4 (Medical Access)":        check_medical_access,
    }

    passed_rules: List[str] = []
    failed_rules: List[str] = []
    all_errors: Dict[str, List[str]] = {}

    for rule_name, rule_fn in rules.items():
        ok, errs = rule_fn(grid)
        if ok:
            passed_rules.append(rule_name)
        else:
            failed_rules.append(rule_name)
            all_errors[rule_name] = errs

    overall_valid = len(failed_rules) == 0

    # ── Build & optionally print report ──────────────────────────────────────
    if verbose:
        _print_report(overall_valid, passed_rules, failed_rules, all_errors)

    summary = (
        "Layout validation PASSED — all constraints satisfied."
        if overall_valid
        else f"Layout validation FAILED — {len(failed_rules)} rule(s) violated: "
             + ", ".join(r.split()[0] for r in failed_rules) + "."
    )

    return {
        "valid":        overall_valid,
        "passed_rules": passed_rules,
        "failed_rules": failed_rules,
        "errors":       all_errors,
        "summary":      summary,
    }


# ──────────────────────────────────────────────
#  Report Printer
# ──────────────────────────────────────────────

def _print_report(
    valid: bool,
    passed: List[str],
    failed: List[str],
    errors: Dict[str, List[str]],
) -> None:
    """Print a formatted validation report to stdout."""
    sep = "=" * 62

    print()
    print(sep)
    print("         AeroNet Lite — CSP Layout Validation Report")
    print(sep)

    # Overall result banner
    if valid:
        print("  ✅  RESULT: Layout is VALID — all 4 constraints satisfied.")
    else:
        print(f"  ❌  RESULT: Layout is INVALID — {len(failed)} rule(s) FAILED.")

    print()
    print("  ┌─── PASSED RULES ───────────────────────────────────┐")
    if passed:
        for r in passed:
            print(f"  │  ✅ {r}")
    else:
        print("  │  (none)")
    print("  └────────────────────────────────────────────────────┘")
    print()

    if failed:
        print("  ┌─── FAILED RULES & VIOLATIONS ──────────────────────┐")
        for r in failed:
            print(f"  │  ❌ {r}")
            for msg in errors.get(r, []):
                # Wrap long lines for readability
                for line in msg.split("\n"):
                    print(f"  │    {line.strip()}")
        print("  └────────────────────────────────────────────────────┘")
        print()
        print("  ACTION REQUIRED: Fix the violations above before")
        print("  proceeding with fleet selection and route planning.")

    print(sep)
    print()


# ══════════════════════════════════════════════
#  Standalone smoke-test
# ══════════════════════════════════════════════

if __name__ == "__main__":
    # Quick smoke-test with a minimal inline grid (no import of grid_model needed)
    # This lets you verify the validator works without the full project setup.

    def _make_cell(row, col, zone="open_field", density=1000,
                   is_hub=False, is_charging=False,
                   is_medical_pickup=False, no_fly=False, demand=0.0):
        return dict(row=row, col=col, zone=zone, density=density,
                    is_hub=is_hub, is_charging=is_charging,
                    is_medical_pickup=is_medical_pickup,
                    no_fly=no_fly, demand=demand)

    # Build a tiny 10x10 test grid
    test_grid = [[_make_cell(r, c) for c in range(10)] for r in range(10)]

    # Place some features to exercise all rules
    test_grid[0][0]["zone"] = "residential"
    test_grid[0][1]["zone"] = "residential"
    test_grid[0][2]["is_hub"] = True          # Hub at (0,2)
    test_grid[0][3]["is_charging"] = True      # Charger near hub — R3 ok
    test_grid[1][1]["zone"] = "industrial"    # Industrial near residential — R1 ok (no hosp/school adjacent)
    test_grid[3][3]["zone"] = "hospital"
    test_grid[3][4]["is_medical_pickup"] = True  # Medical pickup next to hospital — R4 ok
    test_grid[9][9]["zone"] = "residential"   # Far from any hub — R2 violation

    result = validate_layout(test_grid)
    print("Summary:", result["summary"])
