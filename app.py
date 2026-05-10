"""
app.py — AeroNet Lite Streamlit Interface
==========================================
Member 3 — Module: Streamlit Dashboard

Run from the project root with:
    streamlit run app.py

This app lets you:
  1. Load and visualize the city grid
  2. Run CSP validation and see results
  3. Select fleet with adjustable budget
  4. Plan delivery routes with A*
  5. Run the full simulation
  6. View all output figures and logs

All results are also saved to the results/ folder.
"""

import os
import sys
import streamlit as st
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (needed for Streamlit)
import matplotlib.pyplot as plt

# Make sure src/ imports work
sys.path.insert(0, os.path.dirname(__file__))

from src.grid_model         import create_sample_grid, Grid, Drone, Delivery
from src.layout_validator   import validate_layout
from src.fleet_selector     import select_fleet
from src.astar_planner      import find_paths_for_deliveries
from src.delivery_simulator import run_simulation
# ML pipeline — loads with safe fallback if amazon_delivery.csv is missing
try:
    from src.ml_pipeline import get_demand_forecast, detect_anomaly, DEMAND_BY_AREA
    ML_AVAILABLE = True
except Exception as _ml_err:
    ML_AVAILABLE = False
    _ml_err_msg = str(_ml_err)

from src.visualization      import (
    plot_grid, plot_route, plot_heatmap, plot_all, FIGURES_DIR
)


# ===========================================================================
# Page config — must be the FIRST Streamlit call
# ===========================================================================
st.set_page_config(
    page_title="AeroNet Lite",
    page_icon="🚁",
    layout="wide",
)


# ===========================================================================
# Helper: convert Grid object to the list-of-dicts format validators expect
# ===========================================================================
def grid_to_list(grid: Grid):
    return [
        [grid.get_cell(r, c).to_dict() for c in range(grid.size)]
        for r in range(grid.size)
    ]


# ===========================================================================
# Helper: build Drone objects from fleet_result
# ===========================================================================
def build_drones(fleet_result: dict, hub_positions: list) -> list:
    drones = []
    drone_id = 1
    hub_idx  = 0

    for _ in range(fleet_result["n_light"]):
        hub = hub_positions[hub_idx % len(hub_positions)]
        drones.append(Drone(f"D{drone_id}", "light", 1000, 2.0, 12, hub, hub))
        drone_id += 1
        hub_idx  += 1

    for _ in range(fleet_result["n_heavy"]):
        hub = hub_positions[hub_idx % len(hub_positions)]
        drones.append(Drone(f"D{drone_id}", "heavy", 1800, 5.0, 20, hub, hub))
        drone_id += 1
        hub_idx  += 1

    # Always at least 3 drones so all deliveries can be assigned
    drone_id_next = len(drones) + 1
    while len(drones) < 3:
        hub = hub_positions[len(drones) % len(hub_positions)]
        drones.append(Drone(f"D{drone_id_next}", "light", 1000, 2.0, 12, hub, hub))
        drone_id_next += 1
    return drones


# ===========================================================================
# Helper: show a matplotlib figure in Streamlit and save it
# ===========================================================================
def show_figure(fig: plt.Figure):
    st.pyplot(fig)
    plt.close(fig)


# ===========================================================================
# Helper: load an image from results/figures/ and show it
# ===========================================================================
def show_saved_figure(filename: str, caption: str = ""):
    path = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Figure not found: {path}")


# ===========================================================================
# SESSION STATE — keeps results between button clicks
# ===========================================================================
def init_state():
    defaults = {
        "grid":            None,
        "grid_list":       None,
        "validation":      None,
        "fleet_result":    None,
        "drones":          None,
        "deliveries":      None,
        "sim_summary":     None,
        "routes_computed": False,
        "sim_done":        False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ===========================================================================
# SIDEBAR — controls
# ===========================================================================
st.sidebar.title("🚁 AeroNet Lite")
st.sidebar.markdown("Autonomous Drone Delivery Simulation")
st.sidebar.divider()

budget = st.sidebar.slider(
    "Fleet Budget",
    min_value=5_000,
    max_value=30_000,
    value=15_000,
    step=1_000,
    help="Maximum total cost for all drones in the fleet.",
)

use_ga = st.sidebar.checkbox(
    "Use Genetic Algorithm for fleet selection",
    value=False,
    help="If unchecked, uses Brute-Force (always optimal). "
         "GA is faster on large search spaces."
)

st.sidebar.divider()

# Custom deliveries section
st.sidebar.subheader("Custom Deliveries")
st.sidebar.caption("Enter pickup/dropoff as row,col (e.g. 2,4)")

custom_deliveries_input = st.sidebar.text_area(
    "Deliveries (one per line: pickup_r,pickup_c,dropoff_r,dropoff_c,weight_kg)",
    value="2,4,0,6,1.2\n1,1,7,3,0.5\n4,4,9,2,4.5",
    height=100,
)

st.sidebar.divider()
st.sidebar.markdown("**Quick Run:**")
run_all = st.sidebar.button("▶ Run Full Simulation", use_container_width=True)


# ===========================================================================
# MAIN AREA — tabs
# ===========================================================================
st.title("🚁 AeroNet Lite — Autonomous Drone Delivery")
st.caption("BSDS Semester Project · AI · SP2026")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ Grid & Validation",
    "2️⃣ Fleet Selection",
    "3️⃣ Route Planning",
    "4️⃣ Simulation",
    "5️⃣ Results & Logs",
    "6️⃣ ML Pipeline",
])


# ===========================================================================
# TAB 1 — Grid & Validation
# ===========================================================================
with tab1:
    st.header("City Grid & CSP Validation")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        **What is this?**
        The 10×10 city grid represents the drone delivery zone.
        Each cell has a zone type (residential, commercial, etc.),
        population density, and optional features like hubs and charging pads.

        **CSP Rules:**
        - R1: Industrial cells NOT adjacent to schools/hospitals
        - R2: Residential cells within distance 3 of a hub
        - R3: Every hub has a charging pad within distance 2
        - R4: At least one hospital has medical pickup nearby
        """)

        if st.button("🗺️ Load Grid & Validate", use_container_width=True):
            with st.spinner("Building grid and validating..."):
                grid = create_sample_grid()
                grid_list = grid_to_list(grid)
                validation = validate_layout(grid_list, verbose=False)

                st.session_state["grid"]       = grid
                st.session_state["grid_list"]  = grid_list
                st.session_state["validation"] = validation

                fig = plot_grid(grid, title="City Grid", show=False)
                show_figure(fig)

        # Show validation result if available
        if st.session_state["validation"]:
            v = st.session_state["validation"]
            if v["valid"]:
                st.success("✅ Layout VALID — all 4 CSP rules passed!")
            else:
                st.error(f"❌ Layout INVALID — {len(v['failed_rules'])} rule(s) failed")

            for rule in v["passed_rules"]:
                st.write(f"✅ {rule}")
            for rule in v["failed_rules"]:
                st.write(f"❌ {rule}")
                for err in v["errors"].get(rule, []):
                    st.code(err, language=None)

    with col2:
        st.subheader("Grid Map")
        show_saved_figure("grid_map.png", "City Grid Zone Map")


# ===========================================================================
# TAB 2 — Fleet Selection
# ===========================================================================
with tab2:
    st.header("Fleet Selection")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        **How it works:**
        The fleet selector finds the best combination of Light and Heavy drones
        that maximises coverage while staying within the budget.

        **Fitness formula:**
        `score = (0.75 × coverage%) - (0.25 × cost_spent%)`

        | Drone  | Cost  | Payload | Range  |
        |--------|-------|---------|--------|
        | Light  | 1,000 | 2 kg    | 12 cells |
        | Heavy  | 1,800 | 5 kg    | 20 cells |
        """)

        if st.button("🚁 Select Fleet", use_container_width=True):
            if st.session_state["grid_list"] is None:
                st.warning("Load the grid first (Tab 1)!")
            else:
                with st.spinner("Selecting optimal fleet..."):
                    fleet_result = select_fleet(
                        st.session_state["grid_list"],
                        budget=budget,
                        use_ga=use_ga,
                        verbose=False,
                    )
                    st.session_state["fleet_result"] = fleet_result

                    # Build drone objects
                    hub_positions = st.session_state["grid"].get_hub_positions()
                    drones = build_drones(fleet_result, hub_positions)
                    st.session_state["drones"] = drones

                    # Save fleet report
                    os.makedirs("results", exist_ok=True)
                    with open("results/fleet_selection_result.txt", "w") as f:
                        for k, v in fleet_result.items():
                            f.write(f"{k}: {v}\n")

    with col2:
        if st.session_state["fleet_result"]:
            fr = st.session_state["fleet_result"]
            st.subheader("Selected Fleet")

            m1, m2, m3 = st.columns(3)
            m1.metric("Light Drones", fr["n_light"])
            m2.metric("Heavy Drones", fr["n_heavy"])
            m3.metric("Total Cost", f"{fr['total_cost']:,}")

            m4, m5, m6 = st.columns(3)
            m4.metric("Coverage", f"{fr['coverage_pct']:.1f}%")
            m5.metric("Budget Used", f"{fr['cost_spent_pct']:.1f}%")
            m6.metric("Fitness Score", f"{fr['score']:.4f}")

            st.caption(f"Method: {fr['method']} | Budget: {fr['budget']:,}")


# ===========================================================================
# TAB 3 — Route Planning (A*)
# ===========================================================================
with tab3:
    st.header("Route Planning with A*")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        **How A* works:**
        A* finds the shortest path between two cells, avoiding no-fly zones.
        It uses Manhattan distance as a heuristic to guide the search.

        **Zone movement costs:**
        | Zone         | Cost |
        |--------------|------|
        | Open field   | 1.0  |
        | Residential  | 1.2  |
        | Commercial   | 1.3  |
        | Industrial   | 1.5  |
        | Hospital     | 1.8  |
        | School       | 2.0  |
        | No-fly zone  | ∞ (blocked) |
        """)

        if st.button("🗺️ Plan Routes (A*)", use_container_width=True):
            if st.session_state["grid"] is None:
                st.warning("Load the grid first (Tab 1)!")
            else:
                # Parse custom deliveries from sidebar
                deliveries = []
                try:
                    for i, line in enumerate(custom_deliveries_input.strip().split("\n")):
                        parts = [x.strip() for x in line.split(",")]
                        if len(parts) == 5:
                            pr, pc, dr, dc, wt = parts
                            deliveries.append(
                                Delivery(
                                    id=f"DEL{i+1}",
                                    pickup=(int(pr), int(pc)),
                                    dropoff=(int(dr), int(dc)),
                                    weight=float(wt),
                                )
                            )
                except Exception as e:
                    st.error(f"Error parsing deliveries: {e}")
                    deliveries = []

                if not deliveries:
                    st.warning("No valid deliveries entered.")
                else:
                    with st.spinner("Running A* pathfinder..."):
                        grid = st.session_state["grid"]
                        result = find_paths_for_deliveries(grid, deliveries)
                        st.session_state["deliveries"]      = deliveries
                        st.session_state["routes_computed"] = True

                        st.success(
                            f"✅ {len(result['success'])} routes found, "
                            f"{len(result['failed'])} failed"
                        )

                        # Show route for first successful delivery
                        for delivery in deliveries:
                            if delivery.route:
                                fig = plot_route(
                                    grid, delivery.route,
                                    drone_id=delivery.id,
                                    title=f"Route: {delivery.id}",
                                    show=False,
                                )
                                show_figure(fig)
                                break

    with col2:
        st.subheader("Delivery Routes")
        if st.session_state["routes_computed"] and st.session_state["deliveries"]:
            for delivery in st.session_state["deliveries"]:
                status_icon = "✅" if delivery.route else "❌"
                st.write(
                    f"{status_icon} **{delivery.id}** — "
                    f"{delivery.pickup} → {delivery.dropoff} | "
                    f"Weight: {delivery.weight} kg | "
                    f"Steps: {len(delivery.route) if delivery.route else 'N/A'}"
                )
        else:
            show_saved_figure("route_DEL1.png", "Example route visualization")


# ===========================================================================
# TAB 4 — Simulation
# ===========================================================================
with tab4:
    st.header("20-Step Simulation")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        **What happens:**
        - Drones are assigned to deliveries by weight + range
        - Each tick moves drones one cell along their A* route
        - A no-fly zone is activated mid-simulation (disruption)
        - Blocked drones are rerouted using A*
        - Battery anomalies are randomly triggered
        - Low-battery drones are forced to return to hub
        """)

        if st.button("▶ Run Simulation", use_container_width=True):
            if not st.session_state["routes_computed"]:
                st.warning("Plan routes first (Tab 3)!")
            elif not st.session_state["drones"]:
                st.warning("Select fleet first (Tab 2)!")
            else:
                with st.spinner("Running 20-step simulation..."):
                    import copy
                    grid      = st.session_state["grid"]
                    drones    = copy.deepcopy(st.session_state["drones"])
                    deliveries = copy.deepcopy(st.session_state["deliveries"])

                    summary = run_simulation(grid, drones, deliveries)
                    st.session_state["sim_summary"] = summary
                    st.session_state["sim_done"]    = True

                    # Generate dashboard figure
                    routes_dict = {d.id: d.route for d in deliveries if d.route}
                    fig = plot_all(
                        grid,
                        routes=routes_dict,
                        sim_summary=summary,
                        title="Simulation Dashboard",
                        show=False,
                    )
                    show_figure(fig)

    with col2:
        st.subheader("Simulation Results")
        if st.session_state["sim_summary"]:
            s = st.session_state["sim_summary"]
            c1, c2, c3 = st.columns(3)
            c1.metric("✅ Completed",  s["completed"])
            c2.metric("❌ Failed",     s["failed"])
            c3.metric("⏳ Delayed",    s["delayed"])

            c4, c5, c6 = st.columns(3)
            c4.metric("↪ Reroutes",   s["reroutes"])
            c5.metric("⚡ Anomalies",  s["anomalies"])
            c6.metric("Total Tasks",   s["total_deliveries"])
        else:
            show_saved_figure("dashboard.png", "Simulation Dashboard")


# ===========================================================================
# TAB 5 — Results & Logs
# ===========================================================================
with tab5:
    st.header("Results & Output Files")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Simulation Log")
        if st.session_state["sim_summary"] and "log_text" in st.session_state["sim_summary"]:
            log_text = st.session_state["sim_summary"]["log_text"]
            st.text_area("Event Log", value=log_text, height=400)
            st.download_button(
                "⬇ Download Simulation Log",
                data=log_text,
                file_name="simulation_log.txt",
                mime="text/plain",
            )
        else:
            # Try loading from file
            log_path = "results/simulation_log.txt"
            if os.path.exists(log_path):
                with open(log_path) as f:
                    st.text_area("Event Log (from file)", value=f.read(), height=400)
            else:
                st.info("Run the simulation (Tab 4) to see logs here.")

    with col2:
        st.subheader("Figures")
        figures = [
            ("grid_map.png",        "City Grid Map"),
            ("demand_heatmap.png",  "Demand Heatmap"),
            ("dashboard.png",       "Simulation Dashboard"),
            ("validation_result.png", "Validation Result"),
        ]
        for filename, caption in figures:
            path = os.path.join(FIGURES_DIR, filename)
            if os.path.exists(path):
                with st.expander(f"📊 {caption}"):
                    st.image(path, use_container_width=True)

        # Also check for route files
        if os.path.exists(FIGURES_DIR):
            route_files = [f for f in os.listdir(FIGURES_DIR) if f.startswith("route_")]
            if route_files:
                with st.expander(f"🗺️ Route Visualizations ({len(route_files)} files)"):
                    for rf in sorted(route_files):
                        st.image(os.path.join(FIGURES_DIR, rf),
                                 caption=rf, use_container_width=True)



# ===========================================================================
# TAB 6 — ML Pipeline (Module 5)
# ===========================================================================
with tab6:
    st.header("ML Pipeline — Demand Forecasting & Anomaly Detection")

    if not ML_AVAILABLE:
        st.warning(f"ML pipeline not loaded: {_ml_err_msg}")
        st.info("Place **amazon_delivery.csv** in the project root folder and restart the app.")
    else:
        st.success("✅ ML pipeline loaded and models trained.")

    st.divider()

    # --- Part A: Demand Forecast ---
    st.subheader("Part A — Demand Forecasting")
    st.caption("Predicts delivery time (used as demand proxy) based on conditions.")

    col1, col2 = st.columns(2)
    with col1:
        hour    = st.slider("Hour of day", 0, 23, 9)
        weather = st.selectbox("Weather", ["Sunny","Cloudy","Stormy","Fog","Windy","Sandstorms"])
        traffic = st.selectbox("Traffic", ["Low","Medium","High","Jam"])
        area    = st.selectbox("Area", ["Urban","Metropolitian","Semi-Urban","Other"])
        dist_km = st.slider("Distance (km)", 1.0, 20.0, 5.0, 0.5)

    with col2:
        if st.button("🔮 Predict Demand", use_container_width=True):
            if ML_AVAILABLE:
                pred = get_demand_forecast(
                    hour=hour, weather=weather, traffic=traffic,
                    area=area, distance_km=dist_km
                )
                st.metric("Predicted Delivery Time", f"{pred:.1f} min")
                if pred > 130:
                    st.warning("⚠️ High demand — consider dispatching extra drone.")
                else:
                    st.info("Fleet sufficient for current demand.")
            else:
                st.error("ML pipeline not available.")

        st.markdown("**Demand by Area (from training data):**")
        if ML_AVAILABLE:
            for zone, val in DEMAND_BY_AREA.items():
                st.write(f"   → {val:.1f} min avg")

    st.divider()

    # --- Part B: Anomaly Detection ---
    st.subheader("Part B — Anomaly Detection")
    st.caption("Classifies a drone telemetry reading as Normal or an anomaly type.")

    col3, col4 = st.columns(2)
    with col3:
        bat_drop   = st.slider("Battery drop (%/step)", 0.0, 40.0, 3.0, 0.5)
        speed      = st.slider("Speed (units/step)",    0.0, 35.0, 15.0, 0.5)
        route_dev  = st.slider("Route deviation (cells)", 0.0, 20.0, 1.0, 0.5)
        alt_change = st.slider("Altitude change",       -10.0, 30.0, 0.0, 0.5)
        spd_change = st.slider("Speed change",          -10.0, 40.0, 0.0, 0.5)

    with col4:
        if st.button("🚨 Detect Anomaly", use_container_width=True):
            if ML_AVAILABLE:
                result = detect_anomaly(
                    battery_drop=bat_drop, speed=speed,
                    route_deviation=route_dev,
                    altitude_change=alt_change,
                    speed_change=spd_change
                )
                icons = {
                    "Normal":          ("✅", "success"),
                    "Battery_Anomaly": ("🔴", "error"),
                    "Route_Anomaly":   ("⛔", "warning"),
                    "Sensor_Spike":    ("⚡", "warning"),
                }
                icon, level = icons.get(result, ("❓", "info"))
                getattr(st, level)(f"{icon} Result: **{result}**")

                actions = {
                    "Battery_Anomaly": "Force drone return to nearest hub.",
                    "Route_Anomaly":   "Recalculate path via A* from current position.",
                    "Sensor_Spike":    "Ground drone pending diagnostics.",
                    "Normal":          "Continue on planned route.",
                }
                st.info(f"Recommended action: {actions.get(result, '')}")
            else:
                st.error("ML pipeline not available.")


# ===========================================================================
# Quick Run — triggers the full pipeline when "Run Full Simulation" clicked
# ===========================================================================
if run_all:
    st.info("Running full pipeline... check the tabs for results.")
    with st.spinner("Step 1/5: Building grid..."):
        grid = create_sample_grid()
        grid_list = grid_to_list(grid)
        st.session_state["grid"]      = grid
        st.session_state["grid_list"] = grid_list

    with st.spinner("Step 2/5: Validating layout..."):
        validation = validate_layout(grid_list, verbose=False)
        st.session_state["validation"] = validation
        if not validation["valid"]:
            st.error("Layout validation failed! Check Tab 1.")
            st.stop()

    with st.spinner("Step 3/5: Selecting fleet..."):
        fleet_result = select_fleet(grid_list, budget=budget, use_ga=use_ga, verbose=False)
        st.session_state["fleet_result"] = fleet_result
        drones = build_drones(fleet_result, grid.get_hub_positions())
        st.session_state["drones"] = drones

    with st.spinner("Step 4/5: Planning routes..."):
        deliveries = [
            Delivery("DEL1", (2, 4), (0, 6), 1.2),
            Delivery("DEL2", (1, 1), (7, 3), 0.5),
            Delivery("DEL3", (4, 4), (9, 2), 4.5),
        ]
        find_paths_for_deliveries(grid, deliveries)
        st.session_state["deliveries"]      = deliveries
        st.session_state["routes_computed"] = True

    with st.spinner("Step 5/5: Running simulation..."):
        summary = run_simulation(grid, drones, deliveries)
        st.session_state["sim_summary"] = summary
        st.session_state["sim_done"]    = True

        routes_dict = {d.id: d.route for d in deliveries if d.route}
        plot_grid(grid, show=False)
        plot_all(grid, routes=routes_dict, sim_summary=summary, show=False)

    st.success("✅ Full simulation complete! Browse the tabs to see results.")
    st.balloons()