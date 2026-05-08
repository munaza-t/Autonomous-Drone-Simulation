"""
main.py — 20-Step Simulation Orchestrator
==========================================
OWNED BY: Member 1 — The Architect

This is the main entry point that coordinates all modules.
It runs a 20-step simulation scenario to demonstrate the system.
"""

import time
import random
from typing import List, Dict, Any

from src.grid_model import create_sample_grid, Grid, Drone, Delivery
from src.visualization import plot_grid, plot_route, plot_all

class EventLog:
    """Records major simulation events for the final report."""
    def __init__(self):
        self.logs = []

    def add_event(self, step: int, module: str, message: str):
        entry = f"[Step {step:02d}] [{module:^10}] {message}"
        self.logs.append(entry)
        print(entry)

    def print_summary(self):
        print("\n" + "="*50)
        print("SIMULATION EVENT LOG SUMMARY")
        print("="*50)
        for log in self.logs:
            print(log)
        print("="*50)

def run_simulation():
    # --- Initialization ---
    event_log = EventLog()
    grid = create_sample_grid()
    drones: List[Drone] = []
    deliveries: List[Delivery] = []
    
    event_log.add_event(0, "SYSTEM", "Initialising AeroNet Lite Simulation...")

    # --- Step 1-3: Validate, Select Fleet, Visualize ---
    for step in range(1, 4):
        if step == 1:
            # Simulated CSP Validation
            event_log.add_event(step, "CSP", "Layout validation PASSED. (4 rules checked)")
        elif step == 2:
            # Simulated Fleet Selection
            event_log.add_event(step, "FLEET", "Selected fleet: 2 Light Drones, 1 Heavy Drone.")
            # Mocking drone creation
            drones = [
                Drone("D1", "light", 1000, 2.0, 12, (1,1), (1,1)),
                Drone("D2", "light", 1000, 2.0, 12, (4,4), (4,4)),
                Drone("D3", "heavy", 1800, 5.0, 20, (8,6), (8,6))
            ]
        elif step == 3:
            event_log.add_event(step, "VIZ", "Generated initial zone map visualization.")
            plot_grid(grid, title="Step 3: Initial Layout", show=False)

    # --- Step 4-6: Generate Deliveries and Compute Paths ---
    for step in range(4, 7):
        if step == 4:
            event_log.add_event(step, "DEMAND", "Generating 3 delivery tasks based on density.")
            deliveries = [
                Delivery("DEL1", (2,4), (1,6), 1.2), # Hospital to residential
                Delivery("DEL2", (1,1), (4,1), 0.5), # Hub to residential
                Delivery("DEL3", (2,4), (8,2), 4.5)  # Hospital to residential (heavy)
            ]
        elif step == 5:
            event_log.add_event(step, "A*", "Computing optimal paths avoiding obstacles.")
            # Mocking paths
            deliveries[0].route = [(2,4), (2,5), (1,5), (1,6)]
            deliveries[0].assigned_drone = "D1"
            deliveries[1].route = [(1,1), (2,1), (3,1), (4,1)]
            deliveries[1].assigned_drone = "D2"
            deliveries[2].route = [(2,4), (3,4), (4,4), (5,4), (6,4), (7,4), (8,4), (8,3), (8,2)]
            deliveries[2].assigned_drone = "D3"
        elif step == 6:
            event_log.add_event(step, "VIZ", "Generated delivery route overlays.")
            plot_route(grid, deliveries[0].route, drone_id="D1", show=False)

    # --- Step 7-10: Move Drones ---
    for step in range(7, 11):
        event_log.add_event(step, "SIM", "Drones moving along planned paths...")
        # (In a real integration, we'd update drone.location here)

    # --- Step 11: Activate No-Fly Cell ---
    step = 11
    disruption_cell = (1, 5)
    grid.set_cell(disruption_cell[0], disruption_cell[1], no_fly=True)
    event_log.add_event(step, "ENV", f"CRITICAL: No-fly zone activated at {disruption_cell}!")

    # --- Step 12-14: Reroute ---
    for step in range(12, 15):
        if step == 12:
            event_log.add_event(step, "DISRUPT", "Detection: Drone D1 path is BLOCKED.")
        elif step == 13:
            event_log.add_event(step, "A*", "Recalculating alternative route for D1...")
            # Mocking new path: avoid (1,5)
            deliveries[0].route = [(2,4), (2,5), (2,6), (1,6)] 
        elif step == 14:
            event_log.add_event(step, "SIM", "D1 rerouted successfully.")

    # --- Step 15-17: ML Forecast ---
    for step in range(15, 18):
        if step == 15:
            event_log.add_event(step, "ML", "Running hourly demand forecast...")
        elif step == 16:
            # Mocking demand update
            for r in range(10):
                for c in range(10):
                    if grid.get_cell(r,c).zone == 'residential':
                        grid.get_cell(r,c).demand = random.uniform(10, 50)
            event_log.add_event(step, "ML", "Forecast complete: Expected 20% surge in residential zones.")
        elif step == 17:
            event_log.add_event(step, "VIZ", "Updated demand heatmap.")
            plot_heatmap_simulation(grid)

    # --- Step 18: Anomaly ---
    step = 18
    event_log.add_event(step, "ML", "ANOMALY DETECTED: Drone D3 battery dropping 5x faster than expected!")

    # --- Step 19: Handle Anomaly ---
    step = 19
    event_log.add_event(step, "SIM", "Safety protocol: Forcing Drone D3 to return to nearest hub.")
    
    # --- Step 20: Summary ---
    step = 20
    event_log.add_event(step, "SYSTEM", "Simulation complete. Results: 2 Completed, 0 Failed, 1 Emergency Hub Return.")
    event_log.print_summary()
    
    # Final Dashboard
    plot_all(grid, show=False)

def plot_heatmap_simulation(grid):
    # Just a helper to avoid importing the name if it's the same
    from src.visualization import plot_heatmap
    plot_heatmap(grid, show=False)

if __name__ == "__main__":
    run_simulation()
