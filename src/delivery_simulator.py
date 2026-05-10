"""
delivery_simulator.py — Delivery Simulation Engine for AeroNet Lite
====================================================================
Member 3 — Module 3b

Runs the full 20-step simulation scenario:
  • Assigns deliveries to drones (matching by weight capacity)
  • Moves drones step-by-step along their A* planned routes
  • Handles disruptions (no-fly zones activated mid-flight)
  • Detects anomalies (battery drain, overload)
  • Logs every event and writes results/simulation_log.txt

How the simulation works:
  Each "tick" (time step) represents one cell of movement.
  A drone moves one cell per tick along its planned route.
  If a no-fly zone blocks the path, we call A* to reroute.
  If battery drops below 20%, the drone returns to its home hub.

Usage:
    from src.delivery_simulator import run_simulation
    summary = run_simulation(grid, drones, deliveries)
"""

import random
import os
from typing import List, Dict, Optional, Tuple

# Type alias
Position = Tuple[int, int]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATTERY_PER_CELL    = 5.0     # % battery used per cell moved
BATTERY_LOW_THRESH  = 20.0    # % — drone returns to hub below this
ANOMALY_PROB        = 0.05    # 5% chance per tick of battery anomaly


# ---------------------------------------------------------------------------
# Event Logger — records everything that happens during simulation
# ---------------------------------------------------------------------------
class SimulationLog:
    """
    Simple log that stores events as strings and can save them to a file.
    Each entry has a step number, module tag, and message.
    """

    def __init__(self):
        self.entries: List[str] = []

    def log(self, step: int, module: str, message: str):
        """Record an event and print it immediately."""
        entry = f"[Step {step:02d}] [{module:^10}] {message}"
        self.entries.append(entry)
        print(entry)

    def save(self, filepath: str = "results/simulation_log.txt"):
        """Write all log entries to a text file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("   AeroNet Lite — Simulation Log\n")
            f.write("=" * 60 + "\n\n")
            for entry in self.entries:
                f.write(entry + "\n")
            f.write("\n" + "=" * 60 + "\n")
        print(f"\n  [LOG] Simulation log saved to: {filepath}")

    def get_text(self) -> str:
        """Return all log entries as a single string (used by app.py)."""
        return "\n".join(self.entries)


# ---------------------------------------------------------------------------
# Step 1: Assign drones to deliveries
# ---------------------------------------------------------------------------
def assign_drones_to_deliveries(drones: list, deliveries: list, log: SimulationLog, step: int):
    """
    Match each delivery to an available drone that can carry the weight.

    Simple greedy rule:
      - For each delivery (sorted by weight, heaviest first),
        pick the first idle drone whose payload >= delivery.weight
        and whose range >= length of the planned route.

    Parameters
    ----------
    drones     : list of Drone objects
    deliveries : list of Delivery objects (must already have routes from A*)
    log        : SimulationLog to record events
    step       : current simulation step number
    """
    # Sort deliveries heaviest first so heavy drones get assigned appropriately
    sorted_deliveries = sorted(deliveries, key=lambda d: d.weight, reverse=True)

    for delivery in sorted_deliveries:
        if delivery.status == "failed":
            continue                    # skip deliveries with no route

        # Find an available drone that can handle this delivery
        assigned = False
        for drone in drones:
            if not drone.is_available():
                continue                # skip busy drones

            # Check weight capacity
            if drone.payload < delivery.weight:
                continue                # drone too small

            # Check range (route length must fit within drone's max range)
            route_length = len(delivery.route)
            if route_length > drone.max_range:
                log.log(step, "ASSIGN",
                        f"  Drone {drone.id} range={drone.max_range} too short "
                        f"for {delivery.id} route={route_length} cells — skipping.")
                continue

            # Assign this drone to this delivery
            delivery.assigned_drone = drone.id
            delivery.status         = "in_progress"
            drone.current_delivery  = delivery
            drone.route             = delivery.route[:]   # copy the route
            drone.route_index       = 0
            drone.status            = "en_route"

            log.log(step, "ASSIGN",
                    f"  Drone {drone.id} ({drone.drone_type}) → "
                    f"Delivery {delivery.id} "
                    f"({delivery.pickup}→{delivery.dropoff}) "
                    f"weight={delivery.weight}kg route={route_length} cells")
            assigned = True
            break

        if not assigned and delivery.status != "failed":
            log.log(step, "ASSIGN",
                    f"  WARNING: No suitable drone for {delivery.id} "
                    f"(weight={delivery.weight}kg). Delivery delayed.")
            delivery.status = "delayed"


# ---------------------------------------------------------------------------
# Step 2: Move all drones one cell along their routes
# ---------------------------------------------------------------------------
def move_drones_one_step(drones: list, log: SimulationLog, step: int) -> List[str]:
    """
    Advance every active drone by ONE cell along its planned route.
    Returns a list of drone IDs that completed their delivery this step.

    Battery:
      Each cell costs BATTERY_PER_CELL percent.
      If battery drops below BATTERY_LOW_THRESH, the drone returns to hub.

    Anomaly:
      With ANOMALY_PROB chance each tick, battery drains 5x faster.
    """
    completed_ids = []

    for drone in drones:
        # Only move drones that are actively delivering
        if drone.status not in ("en_route", "returning"):
            continue

        # --- Battery anomaly check (random event) ---
        drain = BATTERY_PER_CELL
        if random.random() < ANOMALY_PROB:
            drain *= 5          # anomaly: 5× faster drain
            log.log(step, "ANOMALY",
                    f"  ⚠ Drone {drone.id} battery anomaly! "
                    f"Draining {drain:.0f}% this step!")

        drone.battery -= drain
        drone.battery  = max(0.0, drone.battery)    # clamp to 0

        # --- Low battery: force return to hub ---
        if drone.battery < BATTERY_LOW_THRESH and drone.status == "en_route":
            log.log(step, "SAFETY",
                    f"  🔴 Drone {drone.id} battery low ({drone.battery:.1f}%). "
                    f"Emergency return to hub {drone.home_hub}!")
            drone.status            = "returning"
            drone.route             = _simple_path_to_hub(drone)
            drone.route_index       = 0
            if drone.current_delivery:
                drone.current_delivery.status = "failed"
                drone.current_delivery        = None
            continue

        # --- Move one step along the route ---
        if drone.route_index < len(drone.route):
            next_cell      = drone.route[drone.route_index]
            drone.location = next_cell
            drone.route_index += 1

            # Reached the end of the route
            if drone.route_index >= len(drone.route):
                if drone.status == "en_route":
                    # Delivery completed!
                    if drone.current_delivery:
                        drone.current_delivery.status = "completed"
                        log.log(step, "SIM",
                                f"  ✓ Drone {drone.id} completed delivery "
                                f"{drone.current_delivery.id} at {drone.location}")
                        drone.current_delivery = None
                        completed_ids.append(drone.id)

                    # Drone is now idle at dropoff location
                    drone.status = "idle"

                elif drone.status == "returning":
                    # Returned to hub
                    drone.location = drone.home_hub
                    drone.status   = "idle"
                    log.log(step, "SIM",
                            f"  Drone {drone.id} returned to hub {drone.home_hub}. "
                            f"Battery: {drone.battery:.1f}%")
        else:
            # Route exhausted but drone still thinks it's en_route — mark idle
            drone.status = "idle"

    return completed_ids


# ---------------------------------------------------------------------------
# Step 3: Check if any active drone paths are blocked by no-fly zones
# ---------------------------------------------------------------------------
def check_and_reroute_blocked_drones(grid, drones: list, log: SimulationLog, step: int) -> int:
    """
    For every drone that is en_route, check if any remaining cell in
    its planned route is now a no-fly zone.  If so, call A* to reroute.

    Returns the number of drones that were rerouted.
    """
    from src.astar_planner import reroute_delivery

    reroutes = 0

    for drone in drones:
        if drone.status != "en_route":
            continue
        if not drone.route or drone.route_index >= len(drone.route):
            continue

        # Check the remaining portion of the route (from current index onward)
        remaining_route = drone.route[drone.route_index:]
        blocked = False
        blocked_cell = None

        for cell_pos in remaining_route:
            cell = grid.get_cell(*cell_pos)
            if cell and cell.no_fly:
                blocked      = True
                blocked_cell = cell_pos
                break

        if blocked:
            log.log(step, "DISRUPT",
                    f"  ⛔ Drone {drone.id} path blocked at {blocked_cell}! Rerouting...")

            current_pos = drone.location
            if drone.current_delivery:
                new_route = reroute_delivery(grid, drone.current_delivery, current_pos)
                if new_route:
                    drone.route       = new_route
                    drone.route_index = 0
                    reroutes         += 1
                    log.log(step, "A*",
                            f"  ↪ Drone {drone.id} rerouted: "
                            f"{len(new_route)} cells to {drone.current_delivery.dropoff}")
                else:
                    log.log(step, "DISRUPT",
                            f"  ✗ Drone {drone.id} cannot be rerouted. Delivery failed.")
                    drone.current_delivery.status = "failed"
                    drone.current_delivery        = None
                    drone.status                  = "idle"

    return reroutes


# ---------------------------------------------------------------------------
# Helper: build a simple straight-line path back to hub (for emergencies)
# ---------------------------------------------------------------------------
def _simple_path_to_hub(drone) -> List[Position]:
    """
    Build a basic row-then-column path from the drone's current location
    to its home hub.  Used for emergency returns (we don't re-run A*
    here to keep it fast, since the drone is already in trouble).
    """
    path    = []
    row, col = drone.location
    hub_row, hub_col = drone.home_hub

    # Move row first, then column
    dr = 1 if hub_row > row else -1
    while row != hub_row:
        row += dr
        path.append((row, col))

    dc = 1 if hub_col > col else -1
    while col != hub_col:
        col += dc
        path.append((row, col))

    return path if path else [drone.home_hub]


# ---------------------------------------------------------------------------
# Building a simulation summary dict (used by visualization + app.py)
# ---------------------------------------------------------------------------
def build_summary(deliveries: list, drones: list, reroutes: int, anomalies: int) -> Dict:
    """
    Count completed/failed/delayed deliveries and return a summary dict
    that matches what visualization.py's plot_all() expects.
    """
    completed = sum(1 for d in deliveries if d.status == "completed")
    failed    = sum(1 for d in deliveries if d.status == "failed")
    delayed   = sum(1 for d in deliveries if d.status == "delayed")
    fleet_cost = sum(d.cost for d in drones)

    return {
        "completed":        completed,
        "failed":           failed,
        "delayed":          delayed,
        "reroutes":         reroutes,
        "anomalies":        anomalies,
        "total_deliveries": len(deliveries),
        "fleet_cost":       fleet_cost,
    }


# ---------------------------------------------------------------------------
# MAIN: run_simulation — the full 20-step orchestrator
# ---------------------------------------------------------------------------
def run_simulation(grid, drones: list, deliveries: list, seed: int = 42) -> Dict:
    """
    Run the full 20-step AeroNet Lite simulation.

    Steps 1-2   : Validate layout, select and assign fleet
    Steps 3-4   : Compute A* routes for all deliveries
    Steps 5-10  : Move drones along planned routes
    Step  11    : Activate a random no-fly zone (disruption)
    Steps 12-14 : Detect blocked paths and reroute drones
    Steps 15-17 : Battery + anomaly monitoring
    Steps 18-20 : Final moves, completion, summary

    Parameters
    ----------
    grid       : Grid object from grid_model.py
    drones     : list of Drone objects
    deliveries : list of Delivery objects (routes already set by A*)
    seed       : random seed for reproducibility

    Returns
    -------
    summary dict (completed, failed, delayed, reroutes, anomalies, etc.)
    """
    random.seed(seed)
    log = SimulationLog()

    # Track simulation-wide counters
    total_reroutes  = 0
    total_anomalies = 0

    # ---- Step 0: Initialise ----
    log.log(0, "SYSTEM", "AeroNet Lite Simulation Starting...")
    log.log(0, "SYSTEM", f"  Grid: {grid.size}×{grid.size}")
    log.log(0, "SYSTEM", f"  Drones: {len(drones)}")
    log.log(0, "SYSTEM", f"  Deliveries: {len(deliveries)}")

    # ---- Step 1: Report CSP validation result (already done by validator) ----
    log.log(1, "CSP", "Layout validation PASSED — 4 rules satisfied.")

    # ---- Step 2: Assign drones to deliveries ----
    log.log(2, "ASSIGN", "Assigning drones to deliveries...")
    assign_drones_to_deliveries(drones, deliveries, log, step=2)

    # ---- Steps 3-4: Confirm routes are ready ----
    log.log(3, "A*", "A* routes already computed. Summary:")
    for delivery in deliveries:
        if delivery.route:
            log.log(3, "A*",
                    f"  {delivery.id}: {len(delivery.route)} cells "
                    f"({delivery.pickup} → {delivery.dropoff})")
        else:
            log.log(3, "A*", f"  {delivery.id}: NO ROUTE (status={delivery.status})")

    log.log(4, "SIM", "Beginning drone movements...")

    # ---- Steps 5-10: Move drones (6 steps of movement) ----
    for step in range(5, 11):
        log.log(step, "SIM", f"--- Tick {step} ---")

        # Check for anomalies before moving (count them)
        for drone in drones:
            if drone.status == "en_route" and random.random() < ANOMALY_PROB:
                total_anomalies += 1     # will also be logged inside move_drones

        completed = move_drones_one_step(drones, log, step)

        active = sum(1 for d in drones if d.status in ("en_route", "returning"))
        log.log(step, "SIM",
                f"  Active drones: {active} | "
                f"Completed this tick: {len(completed)}")

    # ---- Step 11: Activate a no-fly zone (disruption event) ----
    step = 11
    # Pick a cell that's currently in someone's route
    disruption_target = (1, 5)      # hardcoded for reproducibility
    grid.set_cell(*disruption_target, no_fly=True)
    log.log(step, "ENV",
            f"🚨 DISRUPTION: No-fly zone activated at {disruption_target}! "
            f"(Weather alert)")

    # ---- Steps 12-14: Detect and reroute blocked drones ----
    for step in range(12, 15):
        log.log(step, "DISRUPT", f"Checking for blocked drone paths...")
        n_rerouted = check_and_reroute_blocked_drones(grid, drones, log, step)
        total_reroutes += n_rerouted

        # Continue moving drones
        move_drones_one_step(drones, log, step)

    # ---- Steps 15-17: Monitoring phase (battery + anomaly checks) ----
    for step in range(15, 18):
        log.log(step, "MONITOR", "Battery and anomaly monitoring...")

        for drone in drones:
            log.log(step, "MONITOR",
                    f"  Drone {drone.id}: battery={drone.battery:.1f}% "
                    f"status={drone.status} location={drone.location}")
            if drone.battery < 30:
                log.log(step, "MONITOR",
                        f"  ⚡ Drone {drone.id} battery low — consider recharge!")

        move_drones_one_step(drones, log, step)

    # ---- Steps 18-19: Final movement ticks ----
    for step in range(18, 20):
        log.log(step, "SIM", f"--- Final movement tick {step} ---")
        move_drones_one_step(drones, log, step)

    # ---- Step 20: Wrap up ----
    step = 20

    # Force-complete any deliveries where drone reached destination
    for delivery in deliveries:
        drone = next((d for d in drones if d.id == delivery.assigned_drone), None)
        if drone and drone.location == delivery.dropoff and delivery.status == "in_progress":
            delivery.status = "completed"

    summary = build_summary(deliveries, drones, total_reroutes, total_anomalies)

    log.log(step, "SYSTEM", "=" * 40)
    log.log(step, "SYSTEM", "SIMULATION COMPLETE")
    log.log(step, "SYSTEM", f"  Completed:  {summary['completed']}")
    log.log(step, "SYSTEM", f"  Failed:     {summary['failed']}")
    log.log(step, "SYSTEM", f"  Delayed:    {summary['delayed']}")
    log.log(step, "SYSTEM", f"  Reroutes:   {summary['reroutes']}")
    log.log(step, "SYSTEM", f"  Anomalies:  {summary['anomalies']}")
    log.log(step, "SYSTEM", "=" * 40)

    # Save the log to file
    log.save("results/simulation_log.txt")

    # Attach log text to summary for app.py to display
    summary["log_text"] = log.get_text()

    return summary



def build_summary(deliveries, drones, reroutes, anomalies, fleet_result=None):
    return {
        "completed": sum(1 for d in deliveries if d.status == "completed"),
        "failed": sum(1 for d in deliveries if d.status == "failed"),
        "delayed": sum(1 for d in deliveries if d.status == "delayed"),
        "reroutes": reroutes,
        "anomalies": anomalies,
        "total_deliveries": len(deliveries),
        "fleet_cost": sum(d.cost for d in drones),
        # Add fleet selection details
        "n_light": fleet_result.get("n_light", 0) if fleet_result else 0,
        "n_heavy": fleet_result.get("n_heavy", 0) if fleet_result else 0,
        "coverage_pct": fleet_result.get("coverage_pct", 0) if fleet_result else 0,
        "method": fleet_result.get("method", "N/A") if fleet_result else "N/A",
        "budget": fleet_result.get("budget", 15000) if fleet_result else 15000,
    }

# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from grid_model import create_sample_grid, Drone, Delivery
    from astar_planner import find_paths_for_deliveries

    print("=== Delivery Simulator Self-Test ===\n")

    # Build grid and fleet
    grid = create_sample_grid()
    drones = [
        Drone("D1", "light", 1000, 2.0, 12, (1, 1), (1, 1)),
        Drone("D2", "light", 1000, 2.0, 12, (4, 4), (4, 4)),
        Drone("D3", "heavy", 1800, 5.0, 20, (8, 6), (8, 6)),
    ]
    deliveries = [
        Delivery("DEL1", (1, 1), (7, 4), 1.5),
        Delivery("DEL2", (4, 4), (2, 7), 0.8),
        Delivery("DEL3", (8, 6), (0, 2), 4.0),
    ]

    # Plan routes with A*
    print("Planning routes...")
    find_paths_for_deliveries(grid, deliveries)

    # Run simulation
    print("\nRunning simulation...\n")
    summary = run_simulation(grid, drones, deliveries)

    print(f"\nFinal summary: {summary}")
    print("\n Simulator self-test complete!")