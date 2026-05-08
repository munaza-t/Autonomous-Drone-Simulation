# AeroNet Lite

> **Autonomous Drone Delivery Simulation** — CSP · Fleet Planning · A\* Routing · Real-Time Replanning · Demand Forecasting · Anomaly Detection

BS Data Science — AI Semester Project (SP2026)

---

## Overview

AeroNet Lite simulates a drone delivery service over a **10×10 city grid**. Five AI modules work together:

| # | Module | AI Technique | Owner |
|---|---|---|---|
| 1 | Grid & Layout Validator | Constraint Satisfaction (CSP) | Member 2 |
| 2 | Fleet Selector | Heuristic / Genetic Algorithm | Member 2 |
| 3 | Delivery Path Planner | A\* Search | Member 3 |
| 4 | Disruption Handler | Optimization / Rerouting | Member 3 |
| 5 | ML Pipeline | Regression + Classification | Member 4 |

**Member 1** owns the shared grid model, visualization layer, and integration orchestrator.

---

## Folder Structure

```
aeronet_lite/
├── data/
│   ├── raw/
│   │   └── sample_grid.json        ← City layout config
│   └── processed/                  ← Trained models & preprocessed data
├── src/
│   ├── grid_model.py               ← [M1] Shared data model (Cell, Grid, Drone, Delivery)
│   ├── layout_validator.py         ← [M2] CSP rule checker
│   ├── fleet_selector.py           ← [M2] Drone fleet optimizer
│   ├── astar_planner.py            ← [M3] A* pathfinder
│   ├── delivery_simulator.py       ← [M3] Delivery assignment + disruption handler
│   ├── ml_pipeline.py              ← [M4] Demand forecasting + anomaly detection
│   ├── visualization.py            ← [M1] Grid plots, route overlay, heatmap
│   └── main.py                     ← [M1] 20-step simulation orchestrator
├── notebooks/
│   ├── demand_forecasting.ipynb    ← [M4]
│   └── anomaly_classifier.ipynb    ← [M4]
├── tests/
│   ├── test_grid_model.py
│   ├── test_visualization.py
│   └── test_integration.py
├── report/
│   └── figures/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd aeronet_lite

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the simulation
python src/main.py

# 5. Run tests
python -m pytest tests/ -v
```

---

## Quick Start

```python
from src.grid_model import create_sample_grid, Grid

# Load default sample grid
grid = create_sample_grid()
print(grid)                          # Grid(10×10)
print(grid.get_hub_positions())      # [(1, 1), (4, 4), (8, 6)]

# Load from custom config
grid2 = Grid()
grid2.load_from_config("data/raw/sample_grid.json")
```

---

## Module Integration Points

```python
# Member 1 calls these from main.py:
from src.layout_validator import validate_grid          # M2
from src.fleet_selector    import select_fleet          # M2
from src.astar_planner     import astar                 # M3
from src.delivery_simulator import assign_deliveries, handle_disruption  # M3
from src.ml_pipeline       import predict_demand, detect_anomaly         # M4
```

---

## CSP Constraints (Layout Validator)

| Rule | Description |
|---|---|
| R1 | Industrial cells cannot be adjacent to Schools or Hospitals |
| R2 | Every Residential cell must be within 3 Manhattan cells of a Hub |
| R3 | Every Hub must have a Charging Pad within 2 cells |
| R4 | At least one Hospital must have a Medical Pickup within 1 cell |

---

## Team

| Role | Responsibility |
|---|---|
| Member 1 — The Architect | Grid model, visualization, integration, main orchestrator |
| Member 2 — The Constraint Whisperer | CSP validator, fleet selector |
| Member 3 — The Pathfinding Wizard | A\* planner, disruption handler |
| Member 4 — The Data Scientist | Demand forecasting, anomaly detection |

---

*Prepared for BS Data Science — AI Semester Project (SP2026)*
