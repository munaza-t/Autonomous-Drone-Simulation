"""
visualization.py — Grid Visualization for AeroNet Lite
=======================================================
Provides all matplotlib-based views for the simulation:

  plot_grid(grid)                         -> zone-coloured 10x10 grid
  plot_route(grid, route, drone_id)       -> route overlay on grid
  plot_heatmap(grid)                      -> demand heatmap
  plot_validation_result(grid, errors)    -> CSP violation highlight
  plot_all(grid, routes, errors, title)   -> combined dashboard (2x2)

Design notes
------------
- No external dependencies beyond matplotlib + numpy (already in requirements).
- Every function saves a PNG to report/figures/ AND calls plt.show() if
  show=True (default). Pass show=False in tests / headless runs.
- All functions return the Figure object so callers can further customise.

Author: Member 1 -- The Architect
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend when DISPLAY is not available (CI / headless)
if os.environ.get("DISPLAY") is None and os.name != "nt":
    matplotlib.use("Agg")

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

# Figure output folder (relative to project root)
FIGURES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "report", "figures"
)


def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _zone_colour(zone: str) -> str:
    return ZONE_COLOURS.get(zone, "#CCCCCC")


def _draw_grid_base(ax: plt.Axes, grid, alpha: float = 1.0):
    """
    Draw the coloured zone cells onto *ax*.
    Returns a colour array (size x size x 3) for reuse in heatmap blending.
    """
    size = grid.size
    colour_matrix = np.ones((size, size, 3))

    for r in range(size):
        for c in range(size):
            cell = grid.get_cell(r, c)
            hex_colour = _zone_colour(cell.zone)

            # Convert hex -> RGB [0,1]
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
    """Draw hub / charging / medical / no-fly markers and optional density text."""
    size = grid.size
    for r in range(size):
        for c in range(size):
            cell = grid.get_cell(r, c)
            plot_row = size - 1 - r
            cx, cy = c + 0.5, plot_row + 0.5   # cell centre

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

            # Stack symbols vertically inside the cell
            n = len(symbols)
            for i, (sym, col) in enumerate(zip(symbols, colours)):
                offset = (i - (n - 1) / 2) * 0.22
                ax.text(
                    cx, cy + offset, sym,
                    ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color=col,
                    zorder=5,
                )

            # Density as tiny grey text (bottom-right corner)
            if show_density and cell.density > 0:
                ax.text(
                    c + 0.92, plot_row + 0.08,
                    str(cell.density),
                    ha="right", va="bottom",
                    fontsize=5, color="#555555",
                    zorder=4,
                )


def _legend_patches() -> List[mpatches.Patch]:
    """Build legend patches for zone colours + special markers."""
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_grid(
    grid,
    title: str = "AeroNet Lite — City Grid",
    show_density: bool = True,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render the zone-coloured 10×10 city grid.

    Parameters
    ----------
    grid        : Grid instance
    title       : Figure title
    show_density: Overlay population density numbers on each cell
    show        : Call plt.show() after rendering
    save_path   : If given, save PNG to this path (auto-saved to report/figures/ too)
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#1C1C2E")
    ax.set_facecolor("#1C1C2E")

    _draw_grid_base(ax, grid)
    _draw_cell_markers(ax, grid, show_density=show_density)

    # Legend
    ax.legend(
        handles=_legend_patches(),
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
        framealpha=0.85,
        facecolor="#2A2A3E",
        labelcolor="white",
        edgecolor="#555555",
    )

    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    plt.tight_layout()

    _ensure_figures_dir()
    out = save_path or os.path.join(FIGURES_DIR, "grid_zone_map.png")
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
    """
    Overlay a drone route (list of (row,col) tuples) on the zone grid.

    The path is drawn as a sequence of arrows between consecutive cells.
    Start cell = green dot, end cell = red dot.
    """
    size = grid.size
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#1C1C2E")
    ax.set_facecolor("#1C1C2E")

    _draw_grid_base(ax, grid, alpha=0.7)
    _draw_cell_markers(ax, grid, show_density=False)

    # Draw route arrows
    for i in range(len(route) - 1):
        r1, c1 = route[i]
        r2, c2 = route[i + 1]
        # Convert to plot coordinates (y-axis flipped)
        x1, y1 = c1 + 0.5, (size - 1 - r1) + 0.5
        x2, y2 = c2 + 0.5, (size - 1 - r2) + 0.5
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=colour,
                lw=2.0,
            ),
            zorder=10,
        )

    # Mark start and end
    if route:
        sr, sc = route[0]
        er, ec = route[-1]
        ax.plot(sc + 0.5, (size - 1 - sr) + 0.5, "o",
                color="#00FF88", markersize=12, zorder=11, label="Start")
        ax.plot(ec + 0.5, (size - 1 - er) + 0.5, "s",
                color="#FF4444", markersize=12, zorder=11, label="End")

        # Annotate step numbers every 3 cells to avoid clutter
        for idx, (r, c) in enumerate(route):
            if idx % 3 == 0 or idx == len(route) - 1:
                ax.text(
                    c + 0.5, (size - 1 - r) + 0.5, str(idx),
                    ha="center", va="center",
                    fontsize=6, color="#FFFFFF",
                    fontweight="bold", zorder=12,
                )

    ax.legend(loc="upper right", fontsize=9,
              facecolor="#2A2A3E", labelcolor="white", edgecolor="#555555")

    title = title or f"Drone {drone_id} Route ({len(route)} cells)"
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

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
    """
    Render demand values as a colour heatmap overlaid on cell outlines.

    Cells with demand = 0 use the zone colour at low alpha.
    Demand values are drawn by Member 4's ML pipeline via grid.set_cell(r,c,demand=x).
    """
    size = grid.size
    demand_matrix = np.zeros((size, size))

    for r in range(size):
        for c in range(size):
            demand_matrix[r, c] = grid.get_cell(r, c).demand

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#1C1C2E")
    ax.set_facecolor("#1C1C2E")

    im = ax.imshow(
        demand_matrix,
        cmap="YlOrRd",
        aspect="equal",
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=max(demand_matrix.max(), 1),  # avoid div-by-zero on all-zero grids
    )

    # Cell borders
    for r in range(size):
        for c in range(size):
            val = demand_matrix[r, c]
            ax.text(
                c, r, f"{val:.1f}",
                ha="center", va="center",
                fontsize=7,
                color="white" if val > demand_matrix.max() * 0.5 else "#333333",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Demand (units)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(range(size), fontsize=7, color="white")
    ax.set_yticklabels(range(size), fontsize=7, color="white")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

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
    """
    Highlight CSP constraint violations on the grid.

    Parameters
    ----------
    grid           : Grid instance
    errors         : List of error strings from validate_grid()
    violated_cells : Optional list of (row,col) tuples to highlight in red
    """
    size = grid.size
    fig, (ax_grid, ax_log) = plt.subplots(
        1, 2,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    fig.patch.set_facecolor("#1C1C2E")
    for ax in (ax_grid, ax_log):
        ax.set_facecolor("#1C1C2E")

    # ---- Left: grid with violations highlighted ----------------------------
    _draw_grid_base(ax_grid, grid, alpha=0.6)
    _draw_cell_markers(ax_grid, grid, show_density=False)

    if violated_cells:
        for r, c in violated_cells:
            rect = mpatches.FancyBboxPatch(
                (c + 0.05, (size - 1 - r) + 0.05),
                0.90, 0.90,
                boxstyle="round,pad=0.02",
                facecolor="#FF0000",
                edgecolor="#FF6666",
                linewidth=2,
                alpha=0.55,
                zorder=6,
            )
            ax_grid.add_patch(rect)
            ax_grid.text(
                c + 0.5, (size - 1 - r) + 0.5, "!",
                ha="center", va="center",
                fontsize=13, fontweight="bold", color="white", zorder=7,
            )

    status = "PASS" if not errors else "FAIL"
    status_colour = "#00FF88" if not errors else "#FF4444"
    ax_grid.set_title(
        f"Layout Validation — {status}",
        color=status_colour, fontsize=12, fontweight="bold", pad=10,
    )
    ax_grid.tick_params(colors="white")

    # ---- Right: error log --------------------------------------------------
    ax_log.axis("off")
    ax_log.set_title("Validation Report", color="white",
                     fontsize=12, fontweight="bold", pad=10)

    if not errors:
        ax_log.text(
            0.5, 0.5, "All 4 CSP rules PASSED\nLayout is valid.",
            ha="center", va="center",
            fontsize=14, color="#00FF88",
            transform=ax_log.transAxes,
        )
    else:
        y = 0.95
        ax_log.text(0.02, y, f"{len(errors)} violation(s) found:",
                    color="#FF4444", fontsize=10, fontweight="bold",
                    transform=ax_log.transAxes)
        y -= 0.06
        for err in errors:
            # Wrap long lines
            words = err.split()
            lines, line = [], []
            for w in words:
                line.append(w)
                if len(" ".join(line)) > 55:
                    lines.append(" ".join(line[:-1]))
                    line = [w]
            lines.append(" ".join(line))
            for ln in lines:
                ax_log.text(0.02, y, ln,
                            color="#FFAA55", fontsize=8,
                            transform=ax_log.transAxes)
                y -= 0.04
            y -= 0.02
            if y < 0.05:
                break

    plt.suptitle(title, color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    _ensure_figures_dir()
    out = save_path or os.path.join(FIGURES_DIR, "validation_result.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())

    if show:
        plt.show()
    return fig


def plot_all(
    grid,
    routes: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    errors: Optional[List[str]] = None,
    title: str = "AeroNet Lite — Simulation Dashboard",
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render a 2×2 dashboard:
      [Zone Map]        [Route Overlay]
      [Demand Heatmap]  [Validation]

    Parameters
    ----------
    routes : dict mapping drone_id -> route list  (pass {} if none yet)
    errors : list of CSP error strings            (pass [] if none)
    """
    routes = routes or {}
    errors = errors or []
    size = grid.size

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#1C1C2E")
    fig.suptitle(title, color="white", fontsize=15, fontweight="bold", y=0.98)

    axes = fig.subplots(2, 2)
    for ax in axes.flat:
        ax.set_facecolor("#1C1C2E")

    # ---- [0,0] Zone map ----------------------------------------------------
    ax00 = axes[0, 0]
    _draw_grid_base(ax00, grid)
    _draw_cell_markers(ax00, grid, show_density=True)
    ax00.set_title("Zone Map", color="white", fontsize=11, fontweight="bold")
    ax00.tick_params(colors="white")

    # ---- [0,1] Route overlay -----------------------------------------------
    ax01 = axes[0, 1]
    _draw_grid_base(ax01, grid, alpha=0.55)
    _draw_cell_markers(ax01, grid, show_density=False)
    colours_cycle = ["#00FFCC", "#FF9900", "#AA88FF", "#FF4488", "#44FFAA"]
    for idx, (d_id, route) in enumerate(routes.items()):
        col = colours_cycle[idx % len(colours_cycle)]
        for i in range(len(route) - 1):
            r1, c1 = route[i]
            r2, c2 = route[i + 1]
            x1, y1 = c1 + 0.5, (size - 1 - r1) + 0.5
            x2, y2 = c2 + 0.5, (size - 1 - r2) + 0.5
            ax01.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8),
                zorder=10,
            )
    route_title = f"Routes ({len(routes)} drone(s))" if routes else "Routes (none yet)"
    ax01.set_title(route_title, color="white", fontsize=11, fontweight="bold")
    ax01.tick_params(colors="white")

    # ---- [1,0] Demand heatmap ----------------------------------------------
    ax10 = axes[1, 0]
    demand_matrix = np.array([
        [grid.get_cell(r, c).demand for c in range(size)]
        for r in range(size)
    ])
    im = ax10.imshow(
        demand_matrix, cmap="YlOrRd", aspect="equal",
        origin="upper", interpolation="nearest",
        vmin=0, vmax=max(demand_matrix.max(), 1),
    )
    plt.colorbar(im, ax=ax10, fraction=0.046, pad=0.04).set_label(
        "Demand", color="white", fontsize=8
    )
    ax10.set_title("Demand Heatmap", color="white", fontsize=11, fontweight="bold")
    ax10.set_xticks(range(size))
    ax10.set_yticks(range(size))
    ax10.tick_params(colors="white", labelsize=7)

    # ---- [1,1] Validation status -------------------------------------------
    ax11 = axes[1, 1]
    ax11.axis("off")
    status = "ALL RULES PASSED" if not errors else f"{len(errors)} RULE(S) FAILED"
    status_col = "#00FF88" if not errors else "#FF4444"
    ax11.text(0.5, 0.88, "CSP Validation", ha="center", va="top",
              color="white", fontsize=12, fontweight="bold",
              transform=ax11.transAxes)
    ax11.text(0.5, 0.78, status, ha="center", va="top",
              color=status_col, fontsize=14, fontweight="bold",
              transform=ax11.transAxes)
    y = 0.65
    for err in errors[:8]:
        ax11.text(0.05, y, f"• {err[:70]}", ha="left", va="top",
                  color="#FFAA55", fontsize=7.5,
                  transform=ax11.transAxes)
        y -= 0.07
    if not errors:
        ax11.text(0.5, 0.55,
                  "R1: Industrial safety   OK\nR2: Residential coverage OK\n"
                  "R3: Hub charging        OK\nR4: Medical access      OK",
                  ha="center", va="top", color="#AAFFAA", fontsize=10,
                  transform=ax11.transAxes, linespacing=2.0)
    ax11.set_title("Validation", color="white", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    _ensure_figures_dir()
    out = save_path or os.path.join(FIGURES_DIR, "dashboard.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Quick self-demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from grid_model import create_sample_grid

    print("Building sample grid...")
    grid = create_sample_grid()

    # Inject some demo demand values
    import random
    random.seed(42)
    for row in grid.cells:
        for cell in row:
            if cell.zone in ("residential", "commercial"):
                cell.demand = round(random.uniform(5, 80), 1)

    print("Rendering zone map ...")
    plot_grid(grid, show=False)

    print("Rendering route overlay ...")
    demo_route = [(1,1),(1,2),(1,3),(2,3),(3,3),(3,4),(4,4)]
    plot_route(grid, demo_route, drone_id="D1", show=False)

    print("Rendering heatmap ...")
    plot_heatmap(grid, show=False)

    print("Rendering validation result (no errors) ...")
    plot_validation_result(grid, errors=[], show=False)

    print("Rendering validation result (with errors) ...")
    errs = [
        "R2: Residential cell (7,5) is 4 cells from nearest hub.",
        "R1: Industrial cell (3,6) is adjacent to School at (2,6).",
    ]
    plot_validation_result(grid, errs, violated_cells=[(7,5),(3,6)], show=False)

    print("Rendering full dashboard ...")
    plot_all(
        grid,
        routes={"D1": demo_route, "D2": [(8,6),(7,6),(6,6),(5,5),(4,4)]},
        errors=[],
        show=False,
    )

    print("All visualizations saved to report/figures/")
    print("DONE -- visualization self-demo complete.")
