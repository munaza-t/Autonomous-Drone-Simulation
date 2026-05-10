<div align="center">

# 🚁 AeroNet Lite
### Autonomous Drone Delivery Simulation

*An end-to-end AI pipeline — from city layout validation to ML-driven anomaly detection*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-blueviolet?style=flat-square)

<br/>

**Department of Data Science · NUCES Islamabad · SP2026**

| | |
|---|---|
| Amaim Anwar | `23i-2614` |
| Dania Waseem | `23i-2622` |
| Munaza Tariq | `23i-2545` |
| Taiba Tariq | `23i-2618` |

</div>

---

## 📌 Overview

AeroNet Lite simulates the complete lifecycle of autonomous drone deliveries over a **10×10 urban city grid**. Six AI-powered modules work in sequence — from enforcing spatial safety constraints, to optimising a drone fleet, planning least-cost routes, running a 20-step live simulation, and integrating machine learning for demand forecasting and anomaly detection. Everything ties together in an interactive Streamlit dashboard.

---

## 🗂️ Project Structure

```
autonomous-drone-sim/
├── src/
│   ├── grid_model.py           # Grid, Cell, Drone, Delivery dataclasses
│   ├── layout_validator.py     # CSP constraint checker (R1–R4)
│   ├── fleet_selector.py       # Brute-Force + Genetic Algorithm fleet optimizer
│   ├── astar_planner.py        # A* pathfinding with zone-weighted costs
│   ├── delivery_simulator.py   # 20-step simulation engine
│   ├── ml_pipeline.py          # Demand forecasting + anomaly detection
│   ├── visualization.py        # All matplotlib figures
│   └── main.py                 # CLI orchestrator
├── data/
│   └── raw/
│       ├── sample_grid.json        # 10×10 city grid configuration
│       └── amazon_delivery.csv     # Delivery dataset (optional)
├── notebooks/
│   ├── demand_forecasting.ipynb    # EDA + regression model training
│   └── anomaly_classifier.ipynb    # Telemetry classification + evaluation
├── results/
│   ├── validation_report.txt
│   ├── fleet_selection_result.txt
│   ├── simulation_log.txt
│   └── figures/
│       ├── grid_map.png
│       ├── demand_heatmap.png
│       ├── dashboard.png
│       └── route_D1.png / route_D2.png / route_D3.png
├── report/
│   └── report_AI.pdf              # Final project report
├── app.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🧩 Modules

### `Module 1` — Shared Grid Model
The foundation of the simulation. Defines a 10×10 city grid where every cell carries a zone type (`residential`, `commercial`, `hospital`, `school`, `industrial`, `open_field`), population density, delivery demand, and flags for hubs, charging pads, no-fly zones, and medical pickups.

---

### `Module 2A` — CSP Layout Validator
Enforces four hard spatial constraints before any simulation can proceed:

| Rule | Name | Constraint |
|------|------|------------|
| R1 | Industrial Safety | Industrial cells must not be adjacent to schools or hospitals |
| R2 | Residential Coverage | Every residential cell within Manhattan distance 3 of a hub |
| R3 | Hub Charging | Every hub has a charging pad within distance 2 |
| R4 | Medical Access | At least one hospital has a medical pickup within distance 1 |

The pipeline halts if any rule fails. All violations are reported with cell coordinates and suggested fixes.

---

### `Module 2B` — Fleet Selector
Finds the optimal mix of **Light** and **Heavy** drones within a budget using two strategies:

- **Brute-Force** — enumerates all valid combinations (~144 checked in < 1ms), always globally optimal
- **Genetic Algorithm** — evolves 30 configurations over 40 generations using tournament selection, crossover, mutation, and elitism

Fitness function: `score = 0.75 × coverage% − 0.25 × cost_spent%`

> 37/37 unit tests passing ✅

---

### `Module 3A` — A\* Path Planner
Finds least-cost routes on the zone-weighted grid. Each zone has a movement cost — schools cost the most (2.0), open fields the least (1.0), no-fly zones are completely blocked (∞). Uses Manhattan distance as the admissible heuristic, guaranteeing optimal paths.

Supports **live rerouting** mid-flight when a no-fly zone is activated.

---

### `Module 3B` — Delivery Simulator
Runs the full **20-step tick-based simulation**:

- Steps 1–2: CSP validation → fleet selection
- Steps 3–10: A\* route planning → drone movement begins
- Step 11: No-fly zone activated (weather alert)
- Steps 12–14: Disruption detection → rerouting via A\*
- Steps 15–19: Battery checks → ML anomaly detection
- Step 20: Summary compiled and logs saved

Battery model: 5% drain per cell moved, forced return to hub below 20%, 5% chance of a battery spike anomaly per tick.

---

### `Module 4` — ML Pipeline

**Part A — Demand Forecasting**
Trained on the Amazon Delivery Dataset (43,648 cleaned rows). Two regression models compared:

| Model | MAE (min) | RMSE (min) |
|-------|-----------|------------|
| Linear Regression | 34.82 | 45.17 |
| **Random Forest** ✅ | **25.56** | **36.45** |

Top features: `Distance_km` (23.2%), `Agent_Rating` (20.0%), `Weather_enc` (15.9%)

**Part B — Anomaly Detection**
Classifies drone telemetry into four classes: `Normal`, `Battery_Anomaly`, `Route_Anomaly`, `Sensor_Spike`. Four classifiers trained on 2,000 synthetic samples:

| Classifier | Accuracy |
|------------|----------|
| Decision Tree | 79.75% |
| **Random Forest** ✅ | **82.75%** |
| KNN | 79.50% |
| Naive Bayes | 82.75% |

Random Forest selected for more balanced per-class F1 scores. 5-fold CV mean accuracy: **0.831 ± 0.015**

---

### `Module 6` — Streamlit Dashboard
An interactive web UI with six tabs: Grid & Validation, Fleet Selection, Route Planning, Simulation, Results & Logs, and ML Pipeline. A single **"Run Full Simulation"** button in the sidebar executes the entire pipeline end-to-end.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| ML & Data | scikit-learn, pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Dashboard | Streamlit |
| Notebooks | Jupyter |
| Dev Environment | VS Code · PyCharm |
| Data | Amazon Delivery Dataset (Kaggle) |

---

## 🚀 Getting Started

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the CLI simulation**
```bash
python src/main.py
```
Generates all result files under `results/`.

**3. Launch the Streamlit dashboard**
```bash
python -m streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** `data/raw/amazon_delivery.csv` is optional. Without it, the demand model uses fixed fallback values — all other modules work identically.

---

## 📊 Baseline Results

Running the default 3-delivery, 15,000-unit budget simulation:

```
✅ Completed   3 / 3
❌ Failed       0
⏱️  Delayed      0
↺  Reroutes     1   (step 12 — no-fly disruption)
💰 Fleet cost   14,400 units
📦 Demand coverage  61.16%
```

---

## 📁 Output Files

| File | Contents |
|------|----------|
| `results/validation_report.txt` | CSP rule results with violations and fixes |
| `results/fleet_selection_result.txt` | Fleet metrics: drones, cost, coverage, fitness |
| `results/simulation_log.txt` | Full 20-step event log |
| `results/figures/grid_map.png` | City grid zone map |
| `results/figures/demand_heatmap.png` | Demand heatmap |
| `results/figures/route_D1/D2/D3.png` | Individual A\* route overlays |
| `results/figures/dashboard.png` | Four-panel simulation dashboard |

---

## 📚 References

1. Russell & Norvig — *Artificial Intelligence: A Modern Approach*, 4th ed. (2020)
2. Hart, Nilsson & Raphael — *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*, IEEE (1968)
3. Breiman — *Random Forests*, Machine Learning (2001)
4. Holland — *Adaptation in Natural and Artificial Systems*, University of Michigan Press (1975)
5. Kumar — *Algorithms for Constraint-Satisfaction Problems: A Survey*, AI Magazine (1992)
6. [Streamlit Documentation](https://docs.streamlit.io)
7. [scikit-learn Documentation](https://scikit-learn.org)
8. [Amazon Delivery Dataset — Kaggle](https://www.kaggle.com/datasets)

---

<div align="center">
  <sub>BSDS · Artificial Intelligence · SP2026 · NUCES Islamabad</sub>
</div>
