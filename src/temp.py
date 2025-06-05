import heapq

# ----------------------------
# Maze definition (0 = free, 1 = wall)
# ----------------------------
maze = [
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0],
]
rows = len(maze)
cols = len(maze[0])

# Start and goal coordinates (row, col)
start = (0, 0)
goal = (4, 6)

# ----------------------------
# Helper functions
# ----------------------------
def in_bounds(pos):
    r, c = pos
    return 0 <= r < rows and 0 <= c < cols

def is_walkable(pos):
    r, c = pos
    return maze[r][c] == 0

def get_neighbors(pos):
    """Return walkable neighbors (up, down, left, right)."""
    r, c = pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if in_bounds((nr, nc)) and is_walkable((nr, nc)):
            yield (nr, nc)

def manhattan(a, b):
    """Heuristic: Manhattan distance between a and b."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, end):
    """Reconstruct path by following came_from pointers."""
    path = []
    current = end
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)  # add the start
    path.reverse()
    return path

# ----------------------------
# Uniform‐Cost Search (equivalent to Dijkstra here)
# ----------------------------
def ucs(start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    cost_so_far = {start: 0}
    came_from = {}

    while frontier:
        current_cost, current = heapq.heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, goal)

        for neighbor in get_neighbors(current):
            new_cost = current_cost + 1  # every step costs 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(frontier, (new_cost, neighbor))

    return None  # no path found

# ----------------------------
# Greedy Best‐First Search (GBFS)
# ----------------------------
def gbfs(start, goal):
    frontier = []
    heapq.heappush(frontier, (manhattan(start, goal), start))
    came_from = {}
    visited = {start}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, goal)

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                priority = manhattan(neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))

    return None  # no path found

# ----------------------------
# A* Search
# ----------------------------
def a_star(start, goal):
    frontier = []
    heapq.heappush(frontier, (manhattan(start, goal), 0, start))
    cost_so_far = {start: 0}
    came_from = {}

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, goal)

        for neighbor in get_neighbors(current):
            new_cost = current_cost + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + manhattan(neighbor, goal)
                heapq.heappush(frontier, (priority, new_cost, neighbor))

    return None  # no path found

# ----------------------------
# Utility to print the maze + path
# ----------------------------
def print_maze_with_path(path):
    display = []
    for r in range(rows):
        row_display = []
        for c in range(cols):
            if (r, c) == start:
                row_display.append("S")
            elif (r, c) == goal:
                row_display.append("G")
            elif (r, c) in path:
                row_display.append("*")
            elif maze[r][c] == 1:
                row_display.append("#")
            else:
                row_display.append(".")
        display.append(" ".join(row_display))
    print("\n".join(display))

# ----------------------------
# Main: run all three searches and print results
# ----------------------------
if __name__ == "__main__":
    print("Uniform‐Cost Search (UCS):")
    path_ucs = ucs(start, goal)
    if path_ucs:
        print("Path found (length = {}):".format(len(path_ucs)))
        print(path_ucs)
        print_maze_with_path(path_ucs)
    else:
        print("No path found.")
    print("\n" + "-" * 40 + "\n")

    print("Greedy Best‐First Search (GBFS):")
    path_gbfs = gbfs(start, goal)
    if path_gbfs:
        print("Path found (length = {}):".format(len(path_gbfs)))
        print(path_gbfs)
        print_maze_with_path(path_gbfs)
    else:
        print("No path found.")
    print("\n" + "-" * 40 + "\n")

    print("A* Search:")
    path_astar = a_star(start, goal)
    if path_astar:
        print("Path found (length = {}):".format(len(path_astar)))
        print(path_astar)
        print_maze_with_path(path_astar)
    else:
        print("No path found.")
