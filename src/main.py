"""
main.py — 20-Step Simulation Orchestrator
==========================================
OWNED BY: Member 1 — The Architect

This is the main entry point for the AeroNet Lite simulation.
It coordinates the different modules (CSP, Fleet, A*, ML) through
 the shared grid model.
"""

from src.grid_model import create_sample_grid
from src.visualization import plot_grid, plot_all

def main():
    print("=== AeroNet Lite Simulation ===")
    
    # 1. Initialize Grid
    print("\n[Step 1] Initializing grid...")
    grid = create_sample_grid()
    print(f"Grid initialized: {grid}")
    
    # 2. Visualization (Member 1 Task)
    print("\n[Step 2] Plotting initial grid map...")
    plot_grid(grid, title="AeroNet Lite - Initial Layout", show=False)
    
    # Placeholder for future integration steps
    print("\n[Note] Other modules (CSP, Fleet, A*, ML) will be integrated by other members.")
    
    # Final Summary Visualization
    print("\n[Final] Generating dashboard...")
    plot_all(grid, show=False)
    
    print("\nSimulation setup complete.")

if __name__ == "__main__":
    main()
