"""
astar_planner.py — A* Pathfinding for AeroNet Lite
====================================================
Member 3 — Module 3a

Finds the shortest path between two cells on the grid using A* search.

Zone Weights (movement costs):
    open_field   : 1.0  (easiest to fly over)
    residential  : 1.2  (some restrictions)
    commercial   : 1.3
    industrial   : 1.5  (noisier, more restricted)
    school       : 2.0  (extra caution near schools)
    hospital     : 1.8  (careful near hospitals)
    no_fly       : blocked (infinite cost — cannot enter)

Usage:
    from src.astar_planner import find_path, find_paths_for_deliveries
"""

import heapq                        
from typing import List, Tuple, Optional, Dict


Position = Tuple[int, int]          # (row, col)
Route = List[Position]              # list of (row, col) positions

# ---------------------------------------------------------------------------
# Zone movement costs
# ---------------------------------------------------------------------------
ZONE_COSTS: Dict[str, float] = {
    "open_field":   1.0,
    "residential":  1.2,
    "commercial":   1.3,
    "hospital":     1.8,
    "school":       2.0,
    "industrial":   1.5,
}

# Cost for entering a no-fly zone — treated as impassable
NO_FLY_COST = float("inf")


# ---------------------------------------------------------------------------
# Helper: get the movement cost for entering a cell
# ---------------------------------------------------------------------------
def get_cell_cost(grid, row: int, col: int) -> float:
    """
    Return the cost of moving INTO the cell at (row, col).
    Returns infinity if the cell is a no-fly zone.
    """
    cell = grid.get_cell(row, col)
    if cell is None:
        return NO_FLY_COST          # out of bounds = impassable
    if cell.no_fly:
        return NO_FLY_COST          # blocked zone = impassable
    return ZONE_COSTS.get(cell.zone, 1.0)


# ---------------------------------------------------------------------------
# Helper: Manhattan distance heuristic
# ---------------------------------------------------------------------------
def heuristic(pos: Position, goal: Position) -> float:
    """
    Estimate the cost from pos to goal using Manhattan distance.
    This is our A* heuristic — it's admissible (never overestimates).
    """
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


# ---------------------------------------------------------------------------
# Helper: get valid 4-directional neighbours of a cell
# ---------------------------------------------------------------------------
def get_neighbours(grid, row: int, col: int) -> List[Position]:
    """
    Return the list of cells directly adjacent (up/down/left/right)
    that are within the grid bounds.
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
    neighbours = []
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if grid.get_cell(new_row, new_col) is not None:  # within bounds
            neighbours.append((new_row, new_col))
    return neighbours


# ---------------------------------------------------------------------------
# Helper: reconstruct the path by following parent pointers
# ---------------------------------------------------------------------------
def reconstruct_path(came_from: Dict, current: Position) -> Route:
    """
    Walk backwards through the came_from dict to build the full path
    from start to goal, then reverse it to get start → goal order.
    """
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)    # move to parent
    path.reverse()                          # was goal→start, now start→goal
    return path


# ---------------------------------------------------------------------------
# Main A* function
# ---------------------------------------------------------------------------
def find_path(grid, start: Position, goal: Position) -> Optional[Route]:
    """
    Find the shortest path from start to goal using A* search.

    Parameters
    ----------
    grid  : the Grid object (from grid_model.py)
    start : (row, col) of the starting cell
    goal  : (row, col) of the destination cell

    Returns
    -------
    A list of (row, col) positions from start to goal (inclusive),
    or None if no path exists (e.g. goal is surrounded by no-fly zones).

    How it works (step by step):
    1. Put the start cell in an "open set" (cells to explore), with priority f=0
    2. Track the best known cost to reach each cell (g_cost)
    3. Track which cell we came from to reach each cell (came_from)
    4. Each loop: pop the cell with the lowest f = g + heuristic
    5. If we reached the goal, reconstruct and return the path
    6. Otherwise, explore each neighbour:
       - Calculate the cost to reach it through the current cell
       - If it's cheaper than what we knew before, update it and add to open set
    7. If the open set empties with no goal found, return None (no path)
    """

    # --- Sanity checks ---
    if grid.get_cell(*start) is None:
        print(f"  [A*] ERROR: Start position {start} is out of bounds.")
        return None
    if grid.get_cell(*goal) is None:
        print(f"  [A*] ERROR: Goal position {goal} is out of bounds.")
        return None
    if grid.get_cell(*goal).no_fly:
        print(f"  [A*] ERROR: Goal position {goal} is a no-fly zone.")
        return None
    if start == goal:
        return [start]              # already there, trivial path

    # --- Data structures ---
    # open_set: priority queue of (f_cost, position)
    # We push tuples so heapq sorts by f_cost automatically
    open_set = []
    heapq.heappush(open_set, (0.0, start))

    # came_from[pos] = the cell we came from to reach pos
    came_from: Dict[Position, Optional[Position]] = {start: None}

    # g_cost[pos] = best known cost to reach pos from start
    g_cost: Dict[Position, float] = {start: 0.0}

    # --- Main A* loop ---
    while open_set:
        # Pop the cell with the lowest f = g + h
        current_f, current = heapq.heappop(open_set)

        # GOAL REACHED — reconstruct and return the path
        if current == goal:
            return reconstruct_path(came_from, current)

        # Explore each neighbour of the current cell
        for neighbour in get_neighbours(grid, *current):
            # Cost to move INTO this neighbour
            step_cost = get_cell_cost(grid, *neighbour)

            # Skip impassable cells (no-fly zones)
            if step_cost == NO_FLY_COST:
                continue

            # Total cost to reach neighbour through current cell
            new_g = g_cost[current] + step_cost

            # Only update if this is a better path to the neighbour
            if neighbour not in g_cost or new_g < g_cost[neighbour]:
                g_cost[neighbour] = new_g
                came_from[neighbour] = current

                # f = g + heuristic (estimated remaining cost)
                f = new_g + heuristic(neighbour, goal)
                heapq.heappush(open_set, (f, neighbour))

    # If we exhaust the open set without finding the goal, no path exists
    print(f"  [A*] WARNING: No path found from {start} to {goal}.")
    return None


# ---------------------------------------------------------------------------
# Compute path length (total weighted cost)
# ---------------------------------------------------------------------------
def path_cost(grid, route: Route) -> float:
    """
    Calculate the total movement cost of a route.
    Useful for comparing paths or displaying route quality.
    """
    if not route or len(route) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(route)):          # start from index 1 (skip start cell)
        r, c = route[i]
        total += get_cell_cost(grid, r, c)
    return round(total, 2)


# ---------------------------------------------------------------------------
# Batch: find paths for a list of Delivery objects
# ---------------------------------------------------------------------------
def find_paths_for_deliveries(grid, deliveries: list) -> dict:
    """
    Run A* for every delivery and store the resulting route on each
    Delivery object.  Returns a summary dict.

    Parameters
    ----------
    grid       : Grid object
    deliveries : list of Delivery objects (from grid_model.py)

    Returns
    -------
    dict with keys:
        'success'  : list of delivery IDs that got a valid path
        'failed'   : list of delivery IDs with no path found
        'routes'   : dict of delivery_id -> route (for visualization)
    """
    success = []
    failed  = []
    routes  = {}

    for delivery in deliveries:
        print(f"  [A*] Planning route for {delivery.id}: "
              f"{delivery.pickup} → {delivery.dropoff}")

        route = find_path(grid, delivery.pickup, delivery.dropoff)

        if route is not None:
            delivery.route  = route
            delivery.status = "assigned"
            cost = path_cost(grid, route)
            routes[delivery.id] = route
            success.append(delivery.id)
            print(f"  [A*]   ✓ Found path: {len(route)} cells, cost={cost:.2f}")
        else:
            delivery.status = "failed"
            failed.append(delivery.id)
            print(f"  [A*]   ✗ No path found!")

    return {
        "success": success,
        "failed":  failed,
        "routes":  routes,
    }


# ---------------------------------------------------------------------------
# Reroute a single delivery (used when a no-fly zone is activated mid-sim)
# ---------------------------------------------------------------------------
def reroute_delivery(grid, delivery, current_pos: Position) -> Optional[Route]:
    """
    Find a new route for a delivery from the drone's CURRENT position
    (not the original pickup), because a no-fly zone was activated
    mid-flight.

    Parameters
    ----------
    grid        : Grid object (with updated no-fly zones)
    delivery    : Delivery object
    current_pos : where the drone currently is (row, col)

    Returns
    -------
    New route from current_pos to delivery.dropoff, or None if blocked.
    """
    goal = delivery.dropoff
    print(f"  [A*] Rerouting {delivery.id} from {current_pos} to {goal}...")

    new_route = find_path(grid, current_pos, goal)

    if new_route:
        delivery.route  = new_route
        delivery.status = "assigned"
        print(f"  [A*]   ✓ New route found: {len(new_route)} cells")
    else:
        delivery.status = "failed"
        print(f"  [A*]   ✗ Reroute failed — no valid path!")

    return new_route


# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from grid_model import create_sample_grid, Delivery

    print("=== A* Planner Self-Test ===\n")
    grid = create_sample_grid()

    # Test 1: basic path
    route = find_path(grid, (0, 0), (9, 9))
    if route:
        print(f"Path (0,0)→(9,9): {len(route)} steps, cost={path_cost(grid, route)}")
        print(f"  First 5 cells: {route[:5]}")
    else:
        print("No path found!")

    # Test 2: no-fly zone blocking
    grid.set_cell(5, 0, no_fly=True)
    grid.set_cell(5, 1, no_fly=True)
    grid.set_cell(5, 2, no_fly=True)
    grid.set_cell(5, 3, no_fly=True)
    route2 = find_path(grid, (0, 0), (9, 0))
    print(f"\nPath with partial no-fly wall: {len(route2)} steps" if route2 else "\nBlocked path returned None — correct!")

    # Test 3: batch delivery planning
    print("\n--- Batch delivery test ---")
    deliveries = [
        Delivery("D1", (1, 1), (8, 6), 1.0),
        Delivery("D2", (2, 4), (7, 3), 4.5),
    ]
    result = find_paths_for_deliveries(grid, deliveries)
    print(f"Success: {result['success']}, Failed: {result['failed']}")
    print("\n✓ All A* tests passed!")