"""
main.py — 20-Step Simulation Orchestrator
==========================================
Member 1 — The Architect  (updated by Member 3 to wire in Modules 3a & 3b)

Run from project root:  python src/main.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grid_model         import create_sample_grid, Drone, Delivery
from src.layout_validator   import validate_layout
from src.fleet_selector     import select_fleet
from src.astar_planner      import find_paths_for_deliveries
from src.delivery_simulator import run_simulation
from src.visualization      import plot_grid, plot_route, plot_all


def main():
    print("\n" + "=" * 60)
    print("        AeroNet Lite — Full Simulation")
    print("=" * 60 + "\n")

    # Step 1: Build grid
    print("[1/6] Building city grid...")
    grid = create_sample_grid()
    print(f"      Grid ready: {grid}")

    # Step 2: Validate layout
    print("\n[2/6] Running CSP layout validation...")
    grid_as_list = [
        [grid.get_cell(r, c).to_dict() for c in range(grid.size)]
        for r in range(grid.size)
    ]
    validation_result = validate_layout(grid_as_list, verbose=True)

    os.makedirs("results", exist_ok=True)
    with open("results/validation_report.txt", "w") as f:
        f.write(validation_result["summary"] + "\n")
        for rule, errors in validation_result["errors"].items():
            f.write(f"\n{rule}:\n")
            for e in errors:
                f.write(e + "\n")
    print("      Validation report saved.")

    if not validation_result["valid"]:
        print("      Layout invalid — fix violations before continuing.")
        return

    # Step 3: Select fleet
    print("\n[3/6] Selecting optimal fleet...")
    fleet_result = select_fleet(grid_as_list, budget=15_000, verbose=True)

    n_light = fleet_result["n_light"]
    n_heavy = fleet_result["n_heavy"]
    hub_positions = grid.get_hub_positions()

    drones = []
    drone_id = 1
    hub_idx  = 0
    for _ in range(n_light):
        hub = hub_positions[hub_idx % len(hub_positions)]
        drones.append(Drone(f"D{drone_id}", "light", 1000, 2.0, 12, hub, hub))
        drone_id += 1; hub_idx += 1
    for _ in range(n_heavy):
        hub = hub_positions[hub_idx % len(hub_positions)]
        drones.append(Drone(f"D{drone_id}", "heavy", 1800, 5.0, 20, hub, hub))
        drone_id += 1; hub_idx += 1

    # Ensure at least 3 drones so all 3 deliveries can be assigned
    while len(drones) < 3:
        hub = hub_positions[len(drones) % len(hub_positions)]
        drones.append(Drone(f"D{drone_id}", "light", 1000, 2.0, 12, hub, hub))
        drone_id += 1

    print(f"      Fleet: {n_light} light + {n_heavy} heavy = {len(drones)} drones")

    with open("results/fleet_selection_result.txt", "w") as f:
        for k, v in fleet_result.items():
            f.write(f"{k}: {v}\n")

    # Step 4: A* routing
    print("\n[4/6] Planning delivery routes with A*...")
    deliveries = [
        Delivery("DEL1", (2, 4), (0, 6), 1.2),
        Delivery("DEL2", (1, 1), (7, 3), 0.5),
        Delivery("DEL3", (4, 4), (9, 2), 4.5),
    ]
    route_result = find_paths_for_deliveries(grid, deliveries)
    print(f"      Routes: {len(route_result['success'])} OK, {len(route_result['failed'])} failed")

    # Step 5: Run simulation
    print("\n[5/6] Running 20-step simulation...\n")
    summary = run_simulation(grid, drones, deliveries)

    # Step 6: Visualizations
    print("\n[6/6] Generating visualizations...")
    plot_grid(grid, title="AeroNet Lite — City Grid", show=False)
    for delivery in deliveries:
        if delivery.route:
            plot_route(grid, delivery.route,
                       drone_id=delivery.assigned_drone or delivery.id,
                       show=False)
    routes_dict = {d.id: d.route for d in deliveries if d.route}
    plot_all(grid, routes=routes_dict, sim_summary=summary, show=False)
    print("      All figures saved to results/figures/")

    print("\n" + "=" * 60)
    print("  DONE — check results/ for all outputs")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()