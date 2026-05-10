"""
visualization.py — Grid Visualization for AeroNet Lite
=======================================================
Provides all matplotlib-based views for the simulation.

Changes from M1 
----------------------------------------------------------
  1. FIGURES_DIR now points to results/figures/ (was report/figures/)
  2. plot_all() panel [1,1] filled with a delivery stats bar chart
     instead of a blank placeholder text.
  3. Added save_text_report() helper used by app.py to export logs.

Author: Member 1 — The Architect
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Zone colour palette  (zone_name -> hex colour)
# ---------------------------------------------------------------------------
ZONE_COLOURS: Dict[str, str] = {
    "residential":  "#4A90D9",   # soft blue
    "commercial":   "#F5A623",   # amber
    "hospital":     "#D0021B",   # red
    "school":       "#7ED321",   # green
    "industrial":   "#9B9B9B",   # grey
    "open_field":   "#F8F8F0",   # off-white
}

# Marker symbols drawn on top of cells
MARKER_HUB      = "H"
MARKER_CHARGING = "+"
MARKER_MEDICAL  = "M"
MARKER_NOFLY    = "X"

# ---------------------------------------------------------------------------
# FIX 1: Save to results/figures/ instead of report/figures/
# ---------------------------------------------------------------------------
FIGURES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "figures"
)

def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def _zone_colour(zone: str) -> str:
    return ZONE_COLOURS.get(zone, "#CCCCCC")

def _draw_grid_base(ax: plt.Axes, grid, alpha: float = 1.0):
    size = grid.size
    colour_matrix = np.ones((size, size, 3))

    for r in range(size):
        for c in range(size):
            cell = grid.get_cell(r, c)
            hex_colour = _zone_colour(cell.zone)
            rgb = matplotlib.colors.to_rgb(hex_colour)
            colour_matrix[r, c] = rgb

            rect = mpatches.FancyBboxPatch(
                (c + 0.05, (size - 1 - r) + 0.05),
                0.90, 0.90,
                boxstyle="round,pad=0.02",
                facecolor=hex_colour,
                edgecolor="#CCCCCC",
                linewidth=0.6,
                alpha=alpha,
            )
            ax.add_patch(rect)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(range(size), fontsize=7)
    ax.set_yticklabels(range(size - 1, -1, -1), fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)

    return colour_matrix

def _draw_cell_markers(ax: plt.Axes, grid, show_density: bool = True):
    size = grid.size
    for r in range(size):
        for c in range(size):
            cell = grid.get_cell(r, c)
            plot_row = size - 1 - r
            cx, cy = c + 0.5, plot_row + 0.5

            symbols = []
            colours = []
            if cell.is_hub:
                symbols.append(MARKER_HUB)
                colours.append("#FFFFFF")
            if cell.is_charging:
                symbols.append(MARKER_CHARGING)
                colours.append("#FFE066")
            if cell.is_medical_pickup:
                symbols.append(MARKER_MEDICAL)
                colours.append("#FF6B6B")
            if cell.no_fly:
                symbols.append(MARKER_NOFLY)
                colours.append("#FF0000")

            n = len(symbols)
            for i, (sym, col) in enumerate(zip(symbols, colours)):
                offset = (i - (n - 1) / 2) * 0.22
                ax.text(cx, cy + offset, sym, ha="center", va="center",
                        fontsize=8, fontweight="bold", color=col, zorder=5)

            if show_density and cell.density > 0:
                ax.text(c + 0.92, plot_row + 0.08, str(cell.density),
                        ha="right", va="bottom", fontsize=5,
                        color="#555555", zorder=4)

            # Zone name label — short abbreviation in top-left corner of cell
            zone_abbrev = {
                "residential": "Res",
                "commercial":  "Com",
                "hospital":    "Hosp",
                "school":      "Sch",
                "industrial":  "Ind",
                "open_field":  "Open",
            }.get(cell.zone, cell.zone[:3].title())
            ax.text(c + 0.05, plot_row + 0.95, zone_abbrev,
                    ha="left", va="top", fontsize=4.5,
                    color="#EEEEEE", zorder=4, style="italic")

def _legend_patches() -> List[mpatches.Patch]:
    patches = [
        mpatches.Patch(color=col, label=zone.replace("_", " ").title())
        for zone, col in ZONE_COLOURS.items()
    ]
    patches += [
        mpatches.Patch(color="#444444", label="H = Drone Hub"),
        mpatches.Patch(color="#444444", label="+ = Charging Pad"),
        mpatches.Patch(color="#444444", label="M = Medical Pickup"),
        mpatches.Patch(color="#FF0000", label="X = No-Fly Zone"),
    ]
    return patches

def plot_grid(
    grid,
    title: str = "AeroNet Lite — City Grid",
    show_density: bool = True,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#1C1C2E")
    ax.set_facecolor("#1C1C2E")
    _draw_grid_base(ax, grid)
    _draw_cell_markers(ax, grid, show_density=show_density)
    ax.legend(
        handles=_legend_patches(), loc="upper left",
        bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=0.85,
        facecolor="#2A2A3E", labelcolor="white", edgecolor="#555555",
    )
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    plt.tight_layout()
    _ensure_figures_dir()
    out = save_path or os.path.join(FIGURES_DIR, "grid_map.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig

def plot_route(
    grid,
    route: List[Tuple[int, int]],
    drone_id: str = "D?",
    title: Optional[str] = None,
    colour: str = "#00FFCC",
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    size = grid.size
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#1C1C2E")
    ax.set_facecolor("#1C1C2E")
    _draw_grid_base(ax, grid, alpha=0.7)
    _draw_cell_markers(ax, grid, show_density=False)
    for i in range(len(route) - 1):
        r1, c1 = route[i]
        r2, c2 = route[i + 1]
        x1, y1 = c1 + 0.5, (size - 1 - r1) + 0.5
        x2, y2 = c2 + 0.5, (size - 1 - r2) + 0.5
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.0),
            zorder=10,
        )
    if route:
        sr, sc = route[0]
        er, ec = route[-1]
        ax.plot(sc + 0.5, (size - 1 - sr) + 0.5, "o",
                color="#00FF88", markersize=12, zorder=11, label="Start")
        ax.plot(ec + 0.5, (size - 1 - er) + 0.5, "s",
                color="#FF4444", markersize=12, zorder=11, label="End")
    ax.legend(loc="upper right", fontsize=9, facecolor="#2A2A3E",
              labelcolor="white", edgecolor="#555555")
    title = title or f"Drone {drone_id} Route ({len(route)} cells)"
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    plt.tight_layout()
    _ensure_figures_dir()
    safe_id = drone_id.replace(" ", "_")
    out = save_path or os.path.join(FIGURES_DIR, f"route_{safe_id}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig

def plot_heatmap(
    grid,
    title: str = "Delivery Demand Heatmap",
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    size = grid.size
    demand_matrix = np.zeros((size, size))
    for r in range(size):
        for c in range(size):
            demand_matrix[r, c] = grid.get_cell(r, c).demand
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#1C1C2E")
    ax.set_facecolor("#1C1C2E")
    im = ax.imshow(
        demand_matrix, cmap="YlOrRd", aspect="equal",
        origin="upper", interpolation="nearest",
        vmin=0, vmax=max(demand_matrix.max(), 1),
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Demand (units)", color="white", fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    _ensure_figures_dir()
    out = save_path or os.path.join(FIGURES_DIR, "demand_heatmap.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig

def plot_validation_result(
    grid,
    errors: List[str],
    violated_cells: Optional[List[Tuple[int, int]]] = None,
    title: str = "CSP Layout Validation Results",
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    size = grid.size
    fig, (ax_grid, ax_log) = plt.subplots(
        1, 2, figsize=(16, 8),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    fig.patch.set_facecolor("#1C1C2E")
    for ax in (ax_grid, ax_log):
        ax.set_facecolor("#1C1C2E")
    _draw_grid_base(ax_grid, grid, alpha=0.6)
    _draw_cell_markers(ax_grid, grid, show_density=False)
    if violated_cells:
        for r, c in violated_cells:
            rect = mpatches.FancyBboxPatch(
                (c + 0.05, (size - 1 - r) + 0.05), 0.90, 0.90,
                boxstyle="round,pad=0.02",
                facecolor="#FF0000", edgecolor="#FF6666",
                linewidth=2, alpha=0.55, zorder=6,
            )
            ax_grid.add_patch(rect)
    ax_log.axis("off")
    if errors:
        report_text = "VIOLATIONS FOUND:\n\n" + "\n\n".join(errors[:6])
        colour = "#FF4444"
    else:
        report_text = "Layout Valid\nAll 4 CSP rules satisfied."
        colour = "#00FF88"
    ax_log.text(
        0.05, 0.95, report_text,
        ha="left", va="top", color=colour,
        transform=ax_log.transAxes,
        fontsize=9, wrap=True,
        fontfamily="monospace",
    )
    ax_grid.set_title("Grid Layout", color="white", fontsize=11, fontweight="bold")
    ax_log.set_title("Validation Report", color="white", fontsize=11, fontweight="bold")
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _ensure_figures_dir()
    out = save_path or os.path.join(FIGURES_DIR, "validation_result.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig

# ---------------------------------------------------------------------------
# FIX 2: plot_all() — panel [1,1] now shows a real delivery stats bar chart
# ---------------------------------------------------------------------------
def plot_all(
    grid,
    routes: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    sim_summary: Optional[Dict] = None,
    title: str = "AeroNet Lite — Simulation Dashboard",
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel simulation dashboard.

    Panels
    ------
    [0,0] Zone map with markers
    [0,1] Route overlays (all drones)
    [1,0] Demand heatmap
    [1,1] Delivery stats bar chart  ← was empty placeholder, now real
    """
    routes = routes or {}
    size = grid.size
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor("#1C1C2E")
    for ax in axes.flat:
        ax.set_facecolor("#1C1C2E")

    # Panel [0,0] — Zone map
    _draw_grid_base(axes[0, 0], grid)
    _draw_cell_markers(axes[0, 0], grid, show_density=True)
    axes[0, 0].set_title("Zone Map", color="white", fontsize=11, fontweight="bold")

    # Panel [0,1] — Route overlays
    _draw_grid_base(axes[0, 1], grid, alpha=0.55)
    _draw_cell_markers(axes[0, 1], grid, show_density=False)
    route_colours = ["#00FFCC", "#FF6B9D", "#FFD700", "#00BFFF", "#FF8C00"]
    for idx, (drone_id, route) in enumerate(routes.items()):
        col = route_colours[idx % len(route_colours)]
        for i in range(len(route) - 1):
            r1, c1 = route[i]
            r2, c2 = route[i + 1]
            x1, y1 = c1 + 0.5, (size - 1 - r1) + 0.5
            x2, y2 = c2 + 0.5, (size - 1 - r2) + 0.5
            axes[0, 1].annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5),
                zorder=10,
            )
    axes[0, 1].set_title("Drone Routes", color="white", fontsize=11, fontweight="bold")

    # Panel [1,0] — Demand heatmap
    demand_matrix = np.array([
        [grid.get_cell(r, c).demand for c in range(size)]
        for r in range(size)
    ])
    im = axes[1, 0].imshow(
        demand_matrix, cmap="YlOrRd", origin="upper",
        vmin=0, vmax=max(demand_matrix.max(), 1),
    )
    plt.colorbar(im, ax=axes[1, 0], fraction=0.04, pad=0.02).set_label(
        "Demand", color="white", fontsize=8
    )
    axes[1, 0].set_title("Demand Heatmap", color="white", fontsize=11, fontweight="bold")

    # FIX — Panel [1,1] — Delivery stats bar chart
    ax_stats = axes[1, 1]
    if sim_summary:
        labels  = ["Completed", "Delayed", "Failed", "Reroutes", "Anomalies"]
        values  = [
            sim_summary.get("completed", 0),
            sim_summary.get("delayed",   0),
            sim_summary.get("failed",    0),
            sim_summary.get("reroutes",  0),
            sim_summary.get("anomalies", 0),
        ]
        colours = ["#00FF88", "#FFD700", "#FF4444", "#00BFFF", "#FF8C00"]
        bars = ax_stats.bar(labels, values, color=colours, width=0.55, zorder=3)
        ax_stats.set_ylim(0, max(max(values) + 1, 5))
        ax_stats.tick_params(colors="white", labelsize=9)
        ax_stats.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax_stats.yaxis.set_visible(False)
        for bar, val in zip(bars, values):
            ax_stats.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(val), ha="center", va="bottom",
                color="white", fontsize=11, fontweight="bold",
            )
        total = sim_summary.get("total_deliveries", 0)
        cost  = sim_summary.get("fleet_cost", 0)
        ax_stats.text(
            0.5, 0.97,
            f"Total deliveries: {total}   Fleet cost: {cost:,}",
            ha="center", va="top", transform=ax_stats.transAxes,
            color="#AAAAAA", fontsize=8,
        )
    else:
        ax_stats.text(
            0.5, 0.5, "Run simulation\nto see stats",
            ha="center", va="center", color="#888888",
            transform=ax_stats.transAxes, fontsize=12,
        )
    ax_stats.set_title("Delivery Stats", color="white", fontsize=11, fontweight="bold")

    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _ensure_figures_dir()
    out = save_path or os.path.join(FIGURES_DIR, "dashboard.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# FIX 3: Helper used by app.py — save any text report to results/
# ---------------------------------------------------------------------------
def save_text_report(filename: str, content: str) -> None:
    """Write a plain-text report to results/<filename>."""
    import os
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)