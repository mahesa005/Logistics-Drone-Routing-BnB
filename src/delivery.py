import pandas as pd
import os

# ── Configuration (user inputs) ──────────────────────────────────────────────
VAN_SPEED_KMH   = int(input("Enter van speed (km/h, default 30): ") or 30)
DRONE_SPEED_KMH = int(input("Enter drone speed (km/h, default 50): ") or 50)
SERVICE_TIME    = float(input("Enter service time (minutes, default 0.5): ") or 0.5)
DRONE_CAPACITY  = int(input("Enter drone capacity (deliveries per drone, default 3): ") or 3)
NUM_DRONES      = int(input("Enter number of drones available (default 3): ") or 3)
DRONE_MAX_DIST  = float(input("Enter drone max distance (km, default 3.0): ") or 3.0)
MAX_HOUSES      = int(input("Enter max houses to read (default 8): ") or 8)

# ── Utility: load only first MAX_HOUSES sites ─────────────────────────────────
def load_data(filename):
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, '..', 'data')
    df = pd.read_excel(os.path.join(data_dir, filename), index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce') 
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    labels = [str(i) for i in df.index if str(i) in df.columns]
    labels = labels[:MAX_HOUSES]
    mat = df.loc[labels, labels].values
    print(f"Loaded locations: {labels}")
    return labels, mat

# ── Build time matrices (km → minutes) ────────────────────────────────────────
def make_time_matrix(dist, speed):
    tm = dist / speed * 60
    print(f"Computed time matrix at {speed} km/h")
    return tm

# ── Greedy van-only as baseline without lambda ─────────────────────────────────
def greedy_van(locations, van_time):
    # Helper to compute cost from current position
    def cost_to(current_pos, j):
        return van_time[current_pos, j]

    visited = set()
    pos = 0  # depot index
    order = []
    total = 0.0
    n = len(locations)
    while len(visited) < n - 1:
        # collect unvisited choices
        choices = [i for i in range(1, n) if i not in visited]
        # pick best_choice manually
        best_choice = None
        best_cost = float('inf')
        for j in choices:
            c = cost_to(pos, j)
            if c < best_cost:
                best_cost = c
                best_choice = j
        next_i = best_choice
        # update time and state
        total += van_time[pos, next_i] + SERVICE_TIME
        order.append(locations[next_i])
        visited.add(next_i)
        pos = next_i
    print(f"Greedy route: {'->'.join(order)} = {total:.2f} min")
    return total

# ── Simple Branch&Bound (no advanced heuristics) ───────────────────────────────
best_time = float('inf')
best_plan = []

def search(van_pos, delivered, current_time, plan, van_time, drone_time, locations):
    global best_time, best_plan
    n = len(locations)
    # check if all delivered
    if len(delivered) == n - 1:
        if current_time < best_time:
            best_time = current_time
            best_plan = plan.copy()
            print(f"New best: time={best_time:.2f}, plan={best_plan}")
        return
    # prune if worse
    if current_time >= best_time:
        print(f"Pruned: current_time={current_time:.2f} >= best_time={best_time:.2f}")
        return
    # van branch
    for i in range(1, n):
        if i not in delivered:
            print(f"Explore Van→{locations[i]} at t={current_time:.2f}")
            t = van_time[van_pos, i] + SERVICE_TIME
            search(i, delivered | {i}, current_time + t,
                   plan + [f"Van→{locations[i]}"], van_time, drone_time, locations)
    # drone batch branch
    nearby = [i for i in range(1, n)
              if i not in delivered and (drone_time[van_pos, i] * DRONE_SPEED_KMH / 60) <= DRONE_MAX_DIST]
    if nearby:
        # limit by total drone capacity
        max_assign = DRONE_CAPACITY * NUM_DRONES
        assigned = nearby[:max_assign]
        print(f"Explore DroneBatch→{assigned} at t={current_time:.2f}")
        # flight time is max distance + service time
        t_flight = max(drone_time[van_pos, i] for i in assigned) + SERVICE_TIME
        new_del = delivered | set(assigned)
        search(van_pos, new_del, current_time + t_flight,
               plan + [f"DroneBatch→{assigned}"], van_time, drone_time, locations)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    locs, dist = load_data('Van_Urban_40.xlsx')
    van_time = make_time_matrix(dist, VAN_SPEED_KMH)
    drone_time = make_time_matrix(dist, DRONE_SPEED_KMH)
    # baseline
    base = greedy_van(locs, van_time)
    print(f"Baseline time: {base:.2f} min\n")
    # search optimal
    search(0, set(), 0.0, [], van_time, drone_time, locs)
    # results
    print(f"Best time: {best_time:.2f} min")
    print(f"Plan: {best_plan}")
