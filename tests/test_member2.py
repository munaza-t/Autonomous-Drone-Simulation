"""
test_member2.py
===============
Unit tests for Member 2 modules:
  - layout_validator.py  (CSP constraints R1–R4)
  - fleet_selector.py    (Brute-Force + GA fleet selection)

Run with:
    python -m pytest tests/test_member2.py -v

Or standalone:
    python tests/test_member2.py
"""

import sys
import os

# Allow running from repo root or from tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from layout_validator import (
    check_industrial_safety,
    check_residential_coverage,
    check_hub_charging,
    check_medical_access,
    validate_layout,
    manhattan,
    get_neighbors,
)
from fleet_selector import (
    fleet_fitness,
    brute_force_fleet,
    genetic_algorithm_fleet,
    compute_total_demand,
    select_fleet,
    LIGHT_DRONE,
    HEAVY_DRONE,
    DEFAULT_BUDGET,
)


# ══════════════════════════════════════════════
#  Grid Factory Helpers
# ══════════════════════════════════════════════

def _make_cell(row, col, zone="open_field", density=1000,
               is_hub=False, is_charging=False,
               is_medical_pickup=False, no_fly=False, demand=1.0):
    return dict(row=row, col=col, zone=zone, density=density,
                is_hub=is_hub, is_charging=is_charging,
                is_medical_pickup=is_medical_pickup,
                no_fly=no_fly, demand=demand)


def _empty_grid():
    """10×10 grid, all open_field, no features."""
    return [[_make_cell(r, c) for c in range(10)] for r in range(10)]


def _valid_grid():
    """
    A fully compliant grid that satisfies all four CSP rules:
      - Hub at (0,2) with charger at (0,3)  → R3 ok
      - All residential cells within 3 of a hub → R2 ok
      - Industrial at (5,5), no adjacent hospital/school → R1 ok
      - Hospital at (3,3) with medical pickup at (3,4) → R4 ok
    """
    g = _empty_grid()
    # Hubs and chargers
    g[0][2]["is_hub"]      = True
    g[0][3]["is_charging"] = True
    g[5][0]["is_hub"]      = True
    g[5][1]["is_charging"] = True
    g[9][5]["is_hub"]      = True
    g[9][6]["is_charging"] = True
    # Residential cells close to hubs
    for col in range(5):
        g[0][col]["zone"] = "residential"
    for col in range(3):
        g[1][col]["zone"] = "residential"
    # Hospital + medical pickup
    g[3][3]["zone"]             = "hospital"
    g[3][4]["is_medical_pickup"] = True
    # Industrial far from hospitals/schools
    g[5][5]["zone"] = "industrial"
    return g


# ══════════════════════════════════════════════
#  Helper function tests
# ══════════════════════════════════════════════

def test_manhattan_basic():
    assert manhattan((0, 0), (3, 4)) == 7
    assert manhattan((5, 5), (5, 5)) == 0
    assert manhattan((0, 0), (9, 9)) == 18

def test_get_neighbors_corner():
    """Corner cell (0,0) should have exactly 2 neighbours."""
    nbrs = get_neighbors(0, 0)
    assert len(nbrs) == 2
    assert (1, 0) in nbrs
    assert (0, 1) in nbrs

def test_get_neighbors_center():
    """Centre cell should have 4 neighbours."""
    assert len(get_neighbors(5, 5)) == 4

def test_get_neighbors_edge():
    """Edge cell (0, 5) should have 3 neighbours."""
    nbrs = get_neighbors(0, 5)
    assert len(nbrs) == 3


# ══════════════════════════════════════════════
#  R1 — Industrial Safety
# ══════════════════════════════════════════════

def test_r1_passes_clean_grid():
    g = _empty_grid()
    g[5][5]["zone"] = "industrial"   # no hospital/school neighbors
    ok, errs = check_industrial_safety(g)
    assert ok
    assert errs == []

def test_r1_fails_industrial_adjacent_hospital():
    g = _empty_grid()
    g[5][5]["zone"] = "industrial"
    g[5][6]["zone"] = "hospital"   # directly adjacent → violation
    ok, errs = check_industrial_safety(g)
    assert not ok
    assert len(errs) >= 1
    assert "R1 VIOLATION" in errs[0]

def test_r1_fails_industrial_adjacent_school():
    g = _empty_grid()
    g[2][2]["zone"] = "industrial"
    g[2][3]["zone"] = "school"
    ok, errs = check_industrial_safety(g)
    assert not ok

def test_r1_not_triggered_by_diagonal():
    """Diagonal adjacency is NOT counted (only 4-directional)."""
    g = _empty_grid()
    g[4][4]["zone"] = "industrial"
    g[5][5]["zone"] = "hospital"   # diagonal, not adjacent
    ok, errs = check_industrial_safety(g)
    assert ok


# ══════════════════════════════════════════════
#  R2 — Residential Coverage
# ══════════════════════════════════════════════

def test_r2_passes_residential_near_hub():
    g = _empty_grid()
    g[0][0]["zone"] = "residential"
    g[0][2]["is_hub"] = True   # distance = 2  → ok
    ok, errs = check_residential_coverage(g)
    assert ok

def test_r2_fails_residential_far_from_hub():
    g = _empty_grid()
    g[9][9]["zone"]  = "residential"
    g[0][0]["is_hub"] = True    # distance = 18 → violation
    ok, errs = check_residential_coverage(g)
    assert not ok
    assert "(9, 9)" in errs[0]

def test_r2_fails_no_hubs():
    g = _empty_grid()
    g[3][3]["zone"] = "residential"
    ok, errs = check_residential_coverage(g)
    assert not ok

def test_r2_passes_no_residential():
    """Grid with no residential cells trivially satisfies R2."""
    g = _empty_grid()
    ok, errs = check_residential_coverage(g)
    assert ok

def test_r2_boundary_exactly_3():
    g = _empty_grid()
    g[0][0]["zone"]  = "residential"
    g[0][3]["is_hub"] = True    # Manhattan distance = 3 → exactly at limit → ok
    ok, errs = check_residential_coverage(g)
    assert ok


# ══════════════════════════════════════════════
#  R3 — Hub Charging
# ══════════════════════════════════════════════

def test_r3_passes_charger_adjacent():
    g = _empty_grid()
    g[4][4]["is_hub"]      = True
    g[4][5]["is_charging"] = True   # distance = 1 → ok
    ok, errs = check_hub_charging(g)
    assert ok

def test_r3_fails_charger_too_far():
    g = _empty_grid()
    g[0][0]["is_hub"]      = True
    g[0][5]["is_charging"] = True   # distance = 5 → violation
    ok, errs = check_hub_charging(g)
    assert not ok

def test_r3_passes_no_hubs():
    """Grid with no hubs trivially satisfies R3 (nothing to check)."""
    g = _empty_grid()
    ok, errs = check_hub_charging(g)
    assert ok

def test_r3_boundary_exactly_2():
    g = _empty_grid()
    g[3][3]["is_hub"]      = True
    g[3][5]["is_charging"] = True   # distance = 2 → exactly at limit → ok
    ok, errs = check_hub_charging(g)
    assert ok


# ══════════════════════════════════════════════
#  R4 — Medical Access
# ══════════════════════════════════════════════

def test_r4_passes_pickup_adjacent_to_hospital():
    g = _empty_grid()
    g[2][2]["zone"]             = "hospital"
    g[2][3]["is_medical_pickup"] = True   # distance = 1 → ok
    ok, errs = check_medical_access(g)
    assert ok

def test_r4_fails_no_pickup_near_hospital():
    g = _empty_grid()
    g[2][2]["zone"]             = "hospital"
    g[9][9]["is_medical_pickup"] = True   # distance = 14 → violation
    ok, errs = check_medical_access(g)
    assert not ok

def test_r4_passes_no_hospitals():
    """No hospitals → rule is vacuously satisfied."""
    g = _empty_grid()
    ok, errs = check_medical_access(g)
    assert ok

def test_r4_passes_collocated_pickup():
    """Medical pickup on the same cell as hospital (distance 0) → ok."""
    g = _empty_grid()
    g[5][5]["zone"]             = "hospital"
    g[5][5]["is_medical_pickup"] = True
    ok, errs = check_medical_access(g)
    assert ok


# ══════════════════════════════════════════════
#  validate_layout integration
# ══════════════════════════════════════════════

def test_validate_layout_all_pass():
    g = _valid_grid()
    result = validate_layout(g, verbose=False)
    assert result["valid"] is True
    assert len(result["failed_rules"]) == 0
    assert len(result["passed_rules"]) == 4

def test_validate_layout_returns_correct_keys():
    g = _valid_grid()
    result = validate_layout(g, verbose=False)
    for key in ("valid", "passed_rules", "failed_rules", "errors", "summary"):
        assert key in result

def test_validate_layout_summary_string():
    g = _valid_grid()
    result = validate_layout(g, verbose=False)
    assert "PASSED" in result["summary"]


# ══════════════════════════════════════════════
#  Fleet Selector Tests
# ══════════════════════════════════════════════

def test_fleet_fitness_over_budget_returns_neg_inf():
    score = fleet_fitness(100, 100, total_demand=50.0, budget=10_000)
    assert score == float("-inf")

def test_fleet_fitness_zero_drones():
    """Zero drones: coverage = 0%, cost = 0% → score = 0."""
    score = fleet_fitness(0, 0, total_demand=50.0, budget=10_000)
    assert score == 0.0

def test_fleet_fitness_reasonable_fleet():
    score = fleet_fitness(3, 2, total_demand=16.0, budget=15_000)
    assert score > 0  # 3×2 + 2×5 = 16 kg matches demand exactly → high coverage

def test_brute_force_returns_valid_result():
    result = brute_force_fleet(total_demand=20.0, budget=15_000)
    assert result["n_light"] >= 0
    assert result["n_heavy"] >= 0
    assert result["total_cost"] <= 15_000
    assert result["method"] == "Brute-Force"

def test_brute_force_stays_within_budget():
    for budget in [5_000, 10_000, 20_000]:
        result = brute_force_fleet(total_demand=30.0, budget=budget)
        assert result["total_cost"] <= budget

def test_brute_force_score_positive_for_solvable_case():
    result = brute_force_fleet(total_demand=5.0, budget=10_000)
    assert result["score"] > 0  # can easily cover 5 kg demand

def test_ga_returns_valid_result():
    result = genetic_algorithm_fleet(total_demand=20.0, budget=15_000, seed=0)
    assert result["n_light"] >= 0
    assert result["n_heavy"] >= 0
    assert result["total_cost"] <= 15_000
    assert result["method"] == "Genetic Algorithm"

def test_ga_stays_within_budget():
    result = genetic_algorithm_fleet(total_demand=30.0, budget=8_000, seed=7)
    assert result["total_cost"] <= 8_000

def test_compute_total_demand():
    g = _empty_grid()  # each cell has demand=1.0 → total = 100
    total = compute_total_demand(g)
    assert total == 100.0

def test_compute_total_demand_zero_guard():
    """If all cells have demand=0, function returns 1.0 (no div-by-zero)."""
    g = [[_make_cell(r, c, demand=0.0) for c in range(10)] for r in range(10)]
    assert compute_total_demand(g) == 1.0

def test_select_fleet_brute_force():
    g = _empty_grid()
    result = select_fleet(g, budget=15_000, use_ga=False, verbose=False)
    assert "n_light" in result
    assert "n_heavy" in result
    assert result["method"] == "Brute-Force"

def test_select_fleet_ga():
    g = _empty_grid()
    result = select_fleet(g, budget=15_000, use_ga=True, verbose=False)
    assert result["method"] == "Genetic Algorithm"

def test_coverage_never_exceeds_100():
    result = brute_force_fleet(total_demand=0.001, budget=1_000_000)
    assert result["coverage_pct"] <= 100.0


# ══════════════════════════════════════════════
#  Run all tests if executed directly
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"  Running {len(tests)} Member 2 unit tests")
    print(f"{'='*60}")

    for fn in tests:
        try:
            fn()
            print(f"  ✅ PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
    sys.exit(0 if failed == 0 else 1)
