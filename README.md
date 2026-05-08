# AeroNet Lite — Member 1 (The Architect)

> **Autonomous Drone Delivery Simulation** — Shared Grid Model & Visualization

This repository contains the core infrastructure for the AeroNet Lite project, specifically the tasks assigned to **Member 1 (The Architect)**.

---

## Member 1 Responsibilities
- **Shared Grid Model**: Single source of truth for the city layout, drones, and deliveries.
- **Visualization**: Matplotlib-based dashboard and grid views.
- **Integration**: Main orchestrator to run the 20-step simulation.

## Project Structure
```
aeronet_lite/
├── data/
│   └── raw/
│       └── sample_grid.json        ← City layout config
├── src/
│   ├── grid_model.py               ← Core data model
│   ├── visualization.py            ← Grid plots & dashboard
│   └── main.py                     ← Simulation orchestrator
├── tests/
│   ├── test_grid_model.py
│   └── test_visualization.py
├── requirements.txt
└── README.md
```

## Setup & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Simulation**:
   ```bash
   python src/main.py
   ```

3. **Run Tests**:
   ```bash
   python -m pytest tests/
   ```

---
*Developed as part of the BS Data Science — AI Semester Project (SP2026)*
