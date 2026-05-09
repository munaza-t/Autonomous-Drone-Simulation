"""
fleet_selector.py
=================
Member 2 - "The Constraint Whisperer"
Module 2: Fleet Selector for AeroNet Lite

Selects the optimal combination of Light and Heavy drones under a fixed
budget using two strategies:

  1. Brute-Force Search  — exhaustive, guaranteed optimal for small search space
  2. Genetic Algorithm   — evolutionary search, bonus AI technique

Drone Types
-----------
  Light Drone : cost = 1,000 | payload = 2 kg | range = 12 cells
  Heavy Drone : cost = 1,800 | payload = 5 kg | range = 20 cells

Fitness Function
----------------
  score = (0.75 × coverage_%) - (0.25 × cost_spent_%)

  coverage_% = min(100, total_payload_capacity / total_grid_demand × 100)
  cost_spent_% = total_fleet_cost / BUDGET × 100

Integration
-----------
Pass the shared `grid` (10×10 list-of-lists of cell dicts) so demand is
computed from the live model.  Call `select_fleet(grid)` from main.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

# ─────────────────────────────────────────────
#  Drone Specifications
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class DroneSpec:
    name: str
    cost: int       # PKR / USD (project units)
    payload: float  # kg max per trip
    range_cells: int

LIGHT_DRONE = DroneSpec(name="Light Drone",  cost=1_000, payload=2.0, range_cells=12)
HEAVY_DRONE = DroneSpec(name="Heavy Drone",  cost=1_800, payload=5.0, range_cells=20)

# Default mission budget (overridable via select_fleet parameter)
DEFAULT_BUDGET = 15_000

GRID_SIZE = 10
Grid = List[List[Dict[str, Any]]]


# ══════════════════════════════════════════════
#  Demand Helper
# ══════════════════════════════════════════════

def compute_total_demand(grid: Grid) -> float:
    """
    Sum the `demand` field across all grid cells.
    Returns a float representing total delivery demand units (kg).
    Falls back to 1.0 to avoid division by zero on empty grids.
    """
    total = sum(
        grid[r][c].get("demand", 0.0)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
    )
    return max(total, 1.0)  # guard against zero-demand grids


# ══════════════════════════════════════════════
#  Fitness Function
# ══════════════════════════════════════════════

def fleet_fitness(
    n_light: int,
    n_heavy: int,
    total_demand: float,
    budget: int,
) -> float:
    """
    Compute the fitness score for a candidate fleet.

    score = (0.75 × coverage_%) - (0.25 × cost_spent_%)

    Higher is better.  Returns -inf for over-budget fleets so they are
    automatically excluded.

    Parameters
    ----------
    n_light       : number of Light Drones
    n_heavy       : number of Heavy Drones
    total_demand  : total grid demand in kg (from compute_total_demand)
    budget        : maximum allowable fleet cost

    Returns
    -------
    float fitness score, or float('-inf') if fleet exceeds budget
    """
    total_cost = n_light * LIGHT_DRONE.cost + n_heavy * HEAVY_DRONE.cost

    if total_cost > budget:
        return float("-inf")

    # Total payload capacity for ONE round of deliveries
    total_payload = n_light * LIGHT_DRONE.payload + n_heavy * HEAVY_DRONE.payload

    coverage_pct   = min(100.0, (total_payload / total_demand) * 100.0)
    cost_spent_pct = (total_cost / budget) * 100.0

    return (0.75 * coverage_pct) - (0.25 * cost_spent_pct)


# ══════════════════════════════════════════════
#  Strategy 1 — Brute-Force Search
# ══════════════════════════════════════════════

def brute_force_fleet(
    total_demand: float,
    budget: int,
) -> Dict[str, Any]:
    """
    Try every (n_light, n_heavy) combination within budget and return
    the one with the highest fitness score.

    Search space: ≈100–200 combinations → extremely fast (< 1 ms).

    Returns
    -------
    dict with keys: n_light, n_heavy, total_cost, coverage_pct,
                    cost_spent_pct, score, method
    """
    max_light = budget // LIGHT_DRONE.cost
    max_heavy = budget // HEAVY_DRONE.cost

    best_score  = float("-inf")
    best_config = (0, 0)

    for nl in range(0, max_light + 1):
        for nh in range(0, max_heavy + 1):
            score = fleet_fitness(nl, nh, total_demand, budget)
            if score > best_score:
                best_score  = score
                best_config = (nl, nh)

    return _build_result(*best_config, best_score, total_demand, budget, "Brute-Force")


# ══════════════════════════════════════════════
#  Strategy 2 — Genetic Algorithm (Bonus)
# ══════════════════════════════════════════════

def genetic_algorithm_fleet(
    total_demand: float,
    budget: int,
    pop_size: int = 30,
    generations: int = 40,
    mutation_rate: float = 0.15,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Genetic Algorithm fleet selector.

    Chromosome  : [n_light, n_heavy]  (two-integer array)
    Selection   : Tournament selection (size 3)
    Crossover   : Uniform blend of parent genes
    Mutation    : ±1 on a randomly chosen gene
    Repair      : Clamp to [0, max_affordable] after mutation

    Designed to demonstrate the GA concept clearly for viva:
    each operation is named and visible.

    Returns
    -------
    dict with same keys as brute_force_fleet, method='Genetic Algorithm'
    """
    if seed is not None:
        random.seed(seed)

    max_light = budget // LIGHT_DRONE.cost
    max_heavy = budget // HEAVY_DRONE.cost

    def random_chrom() -> List[int]:
        return [random.randint(0, max_light), random.randint(0, max_heavy)]

    def fitness(chrom: List[int]) -> float:
        return fleet_fitness(chrom[0], chrom[1], total_demand, budget)

    def tournament(pop: List[List[int]], k: int = 3) -> List[int]:
        """Select the best individual from k random contestants."""
        contestants = random.sample(pop, min(k, len(pop)))
        return max(contestants, key=fitness)

    def crossover(p1: List[int], p2: List[int]) -> List[int]:
        """Uniform blend: randomly pick each gene from either parent."""
        return [random.choice([p1[i], p2[i]]) for i in range(2)]

    def mutate(chrom: List[int]) -> List[int]:
        """Flip one gene by ±1, then clamp to valid range."""
        child = chrom[:]
        if random.random() < mutation_rate:
            idx   = random.randint(0, 1)
            delta = random.choice([-1, 1])
            child[idx] += delta
            # Repair: clamp to feasible range
            child[0] = max(0, min(child[0], max_light))
            child[1] = max(0, min(child[1], max_heavy))
        return child

    # ── Initialise population ──────────────────
    population = [random_chrom() for _ in range(pop_size)]

    best_ever       = max(population, key=fitness)
    best_ever_score = fitness(best_ever)

    # ── Evolve ────────────────────────────────
    for gen in range(generations):
        # Elitism: carry the best individual forward unchanged
        elite = max(population, key=fitness)
        new_pop = [elite[:]]

        while len(new_pop) < pop_size:
            parent1 = tournament(population)
            parent2 = tournament(population)
            child   = crossover(parent1, parent2)
            child   = mutate(child)
            new_pop.append(child)

        population = new_pop

        gen_best       = max(population, key=fitness)
        gen_best_score = fitness(gen_best)
        if gen_best_score > best_ever_score:
            best_ever       = gen_best[:]
            best_ever_score = gen_best_score

    return _build_result(
        best_ever[0], best_ever[1],
        best_ever_score, total_demand, budget,
        "Genetic Algorithm"
    )


# ══════════════════════════════════════════════
#  Result Builder
# ══════════════════════════════════════════════

def _build_result(
    n_light: int,
    n_heavy: int,
    score: float,
    total_demand: float,
    budget: int,
    method: str,
) -> Dict[str, Any]:
    """Package a fleet configuration into a standard result dict."""
    total_cost    = n_light * LIGHT_DRONE.cost + n_heavy * HEAVY_DRONE.cost
    total_payload = n_light * LIGHT_DRONE.payload + n_heavy * HEAVY_DRONE.payload
    coverage_pct  = min(100.0, (total_payload / total_demand) * 100.0)
    cost_pct      = (total_cost / budget) * 100.0 if budget > 0 else 0.0

    return {
        "n_light":        n_light,
        "n_heavy":        n_heavy,
        "total_cost":     total_cost,
        "budget":         budget,
        "budget_remaining": budget - total_cost,
        "coverage_pct":   round(coverage_pct, 2),
        "cost_spent_pct": round(cost_pct, 2),
        "score":          round(score, 4),
        "total_payload":  total_payload,
        "total_demand":   round(total_demand, 2),
        "method":         method,
    }


# ══════════════════════════════════════════════
#  Public API — called from main.py
# ══════════════════════════════════════════════

def select_fleet(
    grid: Grid,
    budget: int = DEFAULT_BUDGET,
    use_ga: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Select the optimal drone fleet for the given grid and budget.

    Parameters
    ----------
    grid    : 10×10 list-of-lists of cell dicts (from grid_model.py)
    budget  : maximum total fleet cost (default 15,000)
    use_ga  : if True, use Genetic Algorithm instead of brute-force
    verbose : if True, print selection report to stdout

    Returns
    -------
    result dict (see _build_result for keys)

    Example
    -------
    >>> from src.grid_model import build_grid
    >>> from src.fleet_selector import select_fleet
    >>> grid = build_grid()
    >>> fleet = select_fleet(grid, budget=15000)
    """
    total_demand = compute_total_demand(grid)

    if use_ga:
        result = genetic_algorithm_fleet(total_demand, budget)
    else:
        result = brute_force_fleet(total_demand, budget)

    if verbose:
        _print_fleet_report(result)

    return result


# ──────────────────────────────────────────────
#  Report Printer
# ──────────────────────────────────────────────

def _print_fleet_report(result: Dict[str, Any]) -> None:
    """Print a formatted fleet selection report."""
    sep = "=" * 62

    print()
    print(sep)
    print("        AeroNet Lite — Fleet Selection Report")
    print(sep)
    print(f"  Method          : {result['method']}")
    print(f"  Budget          : {result['budget']:,} units")
    print()
    print(f"  ┌─ OPTIMAL FLEET ─────────────────────────────────┐")
    print(f"  │  🚁 Light Drones (×{result['n_light']})  "
          f"@ {LIGHT_DRONE.cost:,} each  "
          f"→  payload: {result['n_light'] * LIGHT_DRONE.payload:.1f} kg")
    print(f"  │  🚁 Heavy Drones (×{result['n_heavy']})  "
          f"@ {HEAVY_DRONE.cost:,} each  "
          f"→  payload: {result['n_heavy'] * HEAVY_DRONE.payload:.1f} kg")
    print(f"  └─────────────────────────────────────────────────┘")
    print()
    print(f"  Total Payload    : {result['total_payload']:.1f} kg")
    print(f"  Total Grid Demand: {result['total_demand']:.2f} kg")
    print(f"  Coverage         : {result['coverage_pct']:.1f}%")
    print()
    print(f"  Total Cost       : {result['total_cost']:,}  "
          f"({result['cost_spent_pct']:.1f}% of budget)")
    print(f"  Budget Remaining : {result['budget_remaining']:,}")
    print()
    print(f"  Fitness Score    : {result['score']:.4f}")
    print(f"  Formula          : (0.75 × {result['coverage_pct']:.1f}%) "
          f"- (0.25 × {result['cost_spent_pct']:.1f}%)")
    print(sep)
    print()


# ──────────────────────────────────────────────
#  Compare brute-force vs GA (demo helper)
# ──────────────────────────────────────────────

def compare_strategies(grid: Grid, budget: int = DEFAULT_BUDGET) -> None:
    """
    Run both Brute-Force and GA on the same grid and print a
    side-by-side comparison.  Useful for viva demonstration.
    """
    total_demand = compute_total_demand(grid)

    bf = brute_force_fleet(total_demand, budget)
    ga = genetic_algorithm_fleet(total_demand, budget)

    sep = "─" * 62
    print()
    print("=" * 62)
    print("       Fleet Strategy Comparison: BF vs GA")
    print("=" * 62)
    print(f"{'Metric':<25} {'Brute-Force':>15} {'Genetic Alg':>15}")
    print(sep)
    print(f"{'Light Drones':<25} {bf['n_light']:>15} {ga['n_light']:>15}")
    print(f"{'Heavy Drones':<25} {bf['n_heavy']:>15} {ga['n_heavy']:>15}")
    print(f"{'Total Cost':<25} {bf['total_cost']:>15,} {ga['total_cost']:>15,}")
    print(f"{'Coverage %':<25} {bf['coverage_pct']:>15.1f} {ga['coverage_pct']:>15.1f}")
    print(f"{'Cost Spent %':<25} {bf['cost_spent_pct']:>15.1f} {ga['cost_spent_pct']:>15.1f}")
    print(f"{'Fitness Score':<25} {bf['score']:>15.4f} {ga['score']:>15.4f}")
    print("=" * 62)
    winner = "Brute-Force" if bf["score"] >= ga["score"] else "Genetic Algorithm"
    print(f"  Winner: {winner}")
    print()


# ══════════════════════════════════════════════
#  Standalone smoke-test
# ══════════════════════════════════════════════

if __name__ == "__main__":
    # Quick smoke-test without importing grid_model.
    # Demand per cell = 0.4 kg  →  total = 40 kg across 10×10 grid.
    # This is realistic: each cell represents ~0.4 kg of daily parcel demand.
    # With budget=15,000 the fleet can meaningfully cover this demand.

    def _make_cell(row, col, zone="residential", demand=0.4):
        return dict(row=row, col=col, zone=zone, density=3000,
                    is_hub=False, is_charging=False,
                    is_medical_pickup=False, no_fly=False, demand=demand)

    test_grid = [[_make_cell(r, c) for c in range(10)] for r in range(10)]

    print("=== Brute-Force ===")
    bf_result = select_fleet(test_grid, budget=15_000, use_ga=False)

    print("=== Genetic Algorithm ===")
    ga_result = select_fleet(test_grid, budget=15_000, use_ga=True)

    compare_strategies(test_grid, budget=15_000)
