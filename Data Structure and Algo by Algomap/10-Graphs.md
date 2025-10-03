# Graphs - Learning Guide 🕸️

## 📚 What Are We Learning Today?

Think of this lesson as learning about connections and relationships - like social networks, maps, and how things relate to each other!

---

## 🕸️ Part 1: Understanding Graphs

### The Kid-Friendly Explanation

Imagine a **map of cities**:

- Each **city** is a point (we call these "nodes" or "vertices")
- **Roads between cities** are connections (we call these "edges")
- You can travel from city to city following the roads

**Other real-life examples:**

- **Social networks**: People (nodes) and friendships (edges)
- **Internet**: Computers (nodes) and connections (edges)
- **Family tree**: People (nodes) and relationships (edges)
- **Subway map**: Stations (nodes) and train lines (edges)
- **Recipe ingredients**: Items (nodes) and "needs" relationships (edges)

**That's a Graph!** A collection of:

- **Nodes/Vertices**: The things (cities, people, pages)
- **Edges**: The connections between things (roads, friendships, links)

### The Technical Explanation

A **Graph** G = (V, E) consists of:

- **V**: Set of vertices (nodes)
- **E**: Set of edges (connections between vertices)

**Types of Graphs:**

**1. Directed vs Undirected:**

```
Undirected:          Directed:
A ---- B            A ----> B
(A↔B friendship)    (A follows B)
```

**2. Weighted vs Unweighted:**

```
Unweighted:         Weighted:
A ---- B            A --5-- B
                    (5 miles between)
```

**3. Cyclic vs Acyclic:**

```
Cyclic:             Acyclic (DAG):
A → B → C           A → B → D
↑       ↓           ↓
└───────┘           C
```

**4. Connected vs Disconnected:**

```
Connected:          Disconnected:
A---B---C           A---B    D---E
    |                   |
    D                   C
```

### Visual Representation:

```
Example Graph:
    A --- B
    |     |
    C --- D

Vertices: {A, B, C, D}
Edges: {(A,B), (A,C), (B,D), (C,D)}

As Adjacency List:
A: [B, C]
B: [A, D]
C: [A, D]
D: [B, C]

As Adjacency Matrix:
    A  B  C  D
A [[0, 1, 1, 0],
B  [1, 0, 0, 1],
C  [1, 0, 0, 1],
D  [0, 1, 1, 0]]
```

### Graph Representations:

**1. Adjacency List (Most Common):**

```python
# Dictionary of lists
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

# Or list of lists (for numbered vertices)
graph = [
    [1, 2],      # Node 0 connects to 1, 2
    [0, 3],      # Node 1 connects to 0, 3
    [0, 3],      # Node 2 connects to 0, 3
    [1, 2]       # Node 3 connects to 1, 2
]
```

**2. Adjacency Matrix:**

```python
# 2D array where matrix[i][j] = 1 if edge exists
graph = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]
```

**3. Edge List:**

```python
# List of tuples
edges = [(0,1), (0,2), (1,3), (2,3)]
```

### Comparison:

| Representation   | Space  | Check Edge | Get Neighbors | Best For          |
| ---------------- | ------ | ---------- | ------------- | ----------------- |
| Adjacency List   | O(V+E) | O(degree)  | O(1)          | Sparse graphs ✅  |
| Adjacency Matrix | O(V²)  | O(1)       | O(V)          | Dense graphs      |
| Edge List        | O(E)   | O(E)       | O(E)          | Simple algorithms |

### 🔬 Why Data Scientists & Data Engineers Love Graphs

1. **Social Network Analysis**: Friend connections, influence propagation
2. **Recommendation Systems**: User-item relationships, collaborative filtering
3. **Knowledge Graphs**: Entity relationships, semantic search
4. **Dependency Management**: Task dependencies, build systems
5. **Route Planning**: Maps, logistics, delivery optimization
6. **Data Lineage**: Track data flow through pipelines
7. **Network Analysis**: Computer networks, traffic flow
8. **Fraud Detection**: Transaction patterns, anomaly networks
9. **Biological Networks**: Protein interactions, gene regulation
10. **Natural Language**: Word co-occurrence, semantic networks

**Real Data Science Example:**

```python
# Build recommendation graph
user_item_graph = {
    'User1': ['Item1', 'Item2'],
    'User2': ['Item2', 'Item3'],
    'Item1': ['User1', 'User3'],
    'Item2': ['User1', 'User2'],
    'Item3': ['User2']
}

# Find items liked by friends (BFS from user)
def recommend_items(user, graph, depth=2):
    visited = set()
    queue = [(user, 0)]
    recommendations = []

    while queue:
        node, level = queue.pop(0)
        if level > depth:
            continue

        if node.startswith('Item') and node not in visited:
            recommendations.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append((neighbor, level + 1))
                visited.add(neighbor)

    return recommendations
```

---

## 🎨 Part 2: Graph Traversal

### Two Main Traversal Methods:

**1. DFS (Depth-First Search):**

- Go as **DEEP** as possible first
- Uses **Stack** (or recursion)
- Like exploring a maze - go down one path completely

**2. BFS (Breadth-First Search):**

- Explore **LEVEL by LEVEL** (all neighbors first)
- Uses **Queue**
- Like ripples in water spreading outward

### Visual Comparison:

```
Graph:
    1
   / \
  2   3
 / \   \
4   5   6

DFS Order: 1 → 2 → 4 → 5 → 3 → 6
(Go deep: 1→2→4, then backtrack)

BFS Order: 1 → 2 → 3 → 4 → 5 → 6
(Level by level: 1, then [2,3], then [4,5,6])
```

---

## 💻 Practice Problems

### Problem 1: Number of Islands (DFS/BFS Grid Problem)

**Problem**: Count number of islands in a 2D grid. '1' is land, '0' is water. Islands are connected horizontally/vertically.

**Example**:

```
Input:
1 1 0 0 0
1 1 0 0 0
0 0 1 0 0
0 0 0 1 1

Output: 3 (three separate islands)
```

**Solution (DFS):**

```python
def num_islands(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    islands = 0

    def dfs(r, c):
        # Base cases: out of bounds or water
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return

        # Mark as visited by changing to '0'
        grid[r][c] = '0'

        # Explore all 4 directions
        dfs(r + 1, c)  # Down
        dfs(r - 1, c)  # Up
        dfs(r, c + 1)  # Right
        dfs(r, c - 1)  # Left

    # Check every cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1     # Found new island
                dfs(r, c)        # Mark entire island as visited

    return islands

# Test
grid = [
    ['1','1','0','0','0'],
    ['1','1','0','0','0'],
    ['0','0','1','0','0'],
    ['0','0','0','1','1']
]
print(num_islands(grid))  # Output: 3
```

**How It Works:**

```
Grid:
1 1 0 0 0
1 1 0 0 0
0 0 1 0 0
0 0 0 1 1

Step 1: Find '1' at (0,0)
        islands = 1
        DFS marks all connected '1's as '0':
        (0,0)→(0,1)→(1,0)→(1,1)

0 0 0 0 0
0 0 0 0 0
0 0 1 0 0
0 0 0 1 1

Step 2: Find '1' at (2,2)
        islands = 2
        DFS marks (2,2) as '0'

0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 1 1

Step 3: Find '1' at (3,3)
        islands = 3
        DFS marks (3,3)→(3,4) as '0'

Result: 3 islands
```

**BFS Alternative:**

```python
from collections import deque

def num_islands_bfs(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    islands = 0

    def bfs(r, c):
        queue = deque([(r, c)])
        grid[r][c] = '0'

        while queue:
            row, col = queue.popleft()

            # Check all 4 directions
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                    grid[nr][nc] = '0'
                    queue.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                bfs(r, c)

    return islands
```

**Why This Matters for Data Science**:

- **Image Segmentation**: Identify connected regions in images
- **Cluster Analysis**: Find connected components in data
- **Network Analysis**: Identify isolated subnetworks
- **Spatial Analysis**: Find connected geographical regions

**Time Complexity:** O(rows × cols)
**Space Complexity:** O(rows × cols) - worst case recursion depth

---

### Problem 2: Course Schedule (Cycle Detection in DAG)

**Problem**: There are `numCourses` courses. Some have prerequisites. Can you finish all courses?

**Example**:

```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: Take course 0, then course 1

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: Cycle! Can't take 0 before 1 AND 1 before 0
```

**Solution (DFS Cycle Detection):**

```python
def can_finish(numCourses, prerequisites):
    # Build adjacency list
    graph = {i: [] for i in range(numCourses)}
    for course, prereq in prerequisites:
        graph[course].append(prereq)

    # Track visiting states
    # 0 = not visited, 1 = visiting (in current path), 2 = visited (done)
    state = [0] * numCourses

    def has_cycle(course):
        if state[course] == 1:  # Currently visiting - cycle!
            return True
        if state[course] == 2:  # Already checked - no cycle
            return False

        # Mark as visiting
        state[course] = 1

        # Check all prerequisites
        for prereq in graph[course]:
            if has_cycle(prereq):
                return True

        # Mark as visited (done)
        state[course] = 2
        return False

    # Check each course
    for course in range(numCourses):
        if has_cycle(course):
            return False

    return True

# Test
print(can_finish(2, [[1,0]]))        # True
print(can_finish(2, [[1,0],[0,1]]))  # False
```

**Visualization:**

```
Example 1: [[1,0]]
Graph: 1 → 0
No cycle ✓ (Can do 0 then 1)

Example 2: [[1,0],[0,1]]
Graph: 1 ⇄ 0
Cycle! ✗

DFS for cycle detection:
State meanings:
0 (white): Not visited
1 (gray): Currently visiting (in recursion stack)
2 (black): Done visiting

If we reach a GRAY node → Cycle!

Course 1:
  state[1] = 1 (gray)
  Visit prereq 0:
    state[0] = 1 (gray)
    Visit prereq 1:
      state[1] = 1 (gray) ← Already gray! CYCLE!
```

**Topological Sort Solution (Kahn's Algorithm - BFS):**

```python
from collections import deque

def can_finish_bfs(numCourses, prerequisites):
    # Build graph and count incoming edges
    graph = {i: [] for i in range(numCourses)}
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Start with courses that have no prerequisites
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    completed = 0

    while queue:
        course = queue.popleft()
        completed += 1

        # "Complete" this course, unlock dependent courses
        for dependent in graph[course]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # If completed all courses, no cycle
    return completed == numCourses
```

**Why This Matters for Data Engineering**:

- **Task Scheduling**: Check if tasks can be executed in order
- **Build Systems**: Validate dependency graphs
- **Data Pipelines**: Ensure no circular dependencies
- **Workflow Management**: Validate job dependencies

**Time Complexity:** O(V + E) - visit each vertex and edge once
**Space Complexity:** O(V + E) - graph storage

---

### Problem 3: Clone Graph (Graph Traversal & Copy)

**Problem**: Given a reference to a node in a connected undirected graph, return a deep copy of the graph.

**Example**:

```
Input:
1 --- 2
|     |
4 --- 3

Output: Complete copy with same structure
```

**Solution (DFS with HashMap):**

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node):
    if not node:
        return None

    # Map original node to cloned node
    cloned = {}

    def dfs(original):
        # If already cloned, return the clone
        if original in cloned:
            return cloned[original]

        # Create clone of current node
        clone = Node(original.val)
        cloned[original] = clone

        # Clone all neighbors recursively
        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))

        return clone

    return dfs(node)

# Test
# Create graph: 1-2, 1-4, 2-3, 3-4
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)

node1.neighbors = [node2, node4]
node2.neighbors = [node1, node3]
node3.neighbors = [node2, node4]
node4.neighbors = [node1, node3]

cloned_node = clone_graph(node1)
print(f"Original: {node1.val}, Clone: {cloned_node.val}")
print(f"Same object? {node1 is cloned_node}")  # False
```

**BFS Alternative:**

```python
from collections import deque

def clone_graph_bfs(node):
    if not node:
        return None

    cloned = {node: Node(node.val)}
    queue = deque([node])

    while queue:
        original = queue.popleft()

        for neighbor in original.neighbors:
            if neighbor not in cloned:
                # Clone neighbor
                cloned[neighbor] = Node(neighbor.val)
                queue.append(neighbor)

            # Add cloned neighbor to current clone's neighbors
            cloned[original].neighbors.append(cloned[neighbor])

    return cloned[node]
```

**Step-by-Step:**

```
Original Graph:
1 --- 2
|     |
4 --- 3

DFS Cloning:

1. Visit node 1:
   - Create clone1
   - cloned = {1: clone1}

2. Visit neighbor 2:
   - Create clone2
   - cloned = {1: clone1, 2: clone2}
   - clone1.neighbors.append(clone2)

3. Visit 2's neighbor 1:
   - Already cloned! Return clone1
   - clone2.neighbors.append(clone1)

4. Visit 2's neighbor 3:
   - Create clone3
   - cloned = {1: clone1, 2: clone2, 3: clone3}
   - clone2.neighbors.append(clone3)

... and so on

Result: Complete deep copy!
```

**Why This Matters for Data Science**:

- **Model Persistence**: Copy graph-based models
- **Data Versioning**: Create snapshots of graph data
- **Distributed Systems**: Replicate graph structures
- **Testing**: Create test copies without affecting original

**Time Complexity:** O(V + E) - visit each vertex and edge
**Space Complexity:** O(V) - store cloned nodes

---

## 🎯 Key Takeaways

### When to Use DFS vs BFS:

| Use DFS When           | Use BFS When              |
| ---------------------- | ------------------------- |
| Finding if path exists | Finding **shortest** path |
| Detecting cycles       | Level-order processing    |
| Topological sort       | Finding closest/nearest   |
| Backtracking needed    | Distance from source      |
| Memory constrained     | All nodes at distance k   |

### Graph Problem Patterns:

**1. Traversal Pattern:**

```python
visited = set()

def dfs(node):
    if node in visited:
        return
    visited.add(node)

    for neighbor in graph[node]:
        dfs(neighbor)
```

**2. Cycle Detection:**

```python
# Use 3 states: not visited, visiting, visited
# If we reach "visiting" node → cycle!
```

**3. Shortest Path (BFS):**

```python
from collections import deque

queue = deque([(start, 0)])  # (node, distance)
visited = {start}

while queue:
    node, dist = queue.popleft()
    if node == target:
        return dist

    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append((neighbor, dist + 1))
```

**4. Connected Components:**

```python
components = 0
visited = set()

for node in all_nodes:
    if node not in visited:
        components += 1
        dfs(node, visited)  # Mark entire component
```

---

## 🚀 Next Steps for Practice

### LeetCode Easy:

- Number of Islands ✅ (covered)
- Clone Graph ✅ (covered)
- Find Center of Star Graph
- Find if Path Exists in Graph

### LeetCode Medium:

- Course Schedule ✅ (covered)
- Course Schedule II (return order)
- Number of Connected Components
- Pacific Atlantic Water Flow
- Rotting Oranges (BFS)
- Surrounded Regions

### LeetCode Hard:

- Word Ladder
- Alien Dictionary
- Minimum Height Trees
- Critical Connections in Network

### Data Science Projects:

1. **Social Network Analyzer**: Find communities, influencers
2. **Recommendation Engine**: User-item graph traversal
3. **Dependency Resolver**: Build order for tasks
4. **Network Visualizer**: Display graph structures
5. **Path Finder**: Shortest route in graphs

---

## 💡 Pro Tips

### Graph Building Template:

```python
# Adjacency List (most common)
def build_graph(n, edges):
    graph = {i: [] for i in range(n)}
    for u, v in edges:
        graph[u].append(v)
        # graph[v].append(u)  # For undirected
    return graph

# With weights
def build_weighted_graph(n, edges):
    graph = {i: [] for i in range(n)}
    for u, v, weight in edges:
        graph[u].append((v, weight))
    return graph
```

### Common Mistakes:

- ❌ Forgetting to mark nodes as visited (infinite loop!)
- ❌ Not handling disconnected graphs
- ❌ Using DFS for shortest path (use BFS!)
- ❌ Modifying graph during iteration
- ❌ Not considering directed vs undirected
- ❌ Forgetting to check if node exists

### Interview Tips:

1. **Clarify graph type**: Directed? Weighted? Connected?
2. **Ask about size**: V and E values (affects representation)
3. **Draw it**: Visualize with small example
4. **State traversal**: "I'll use BFS/DFS because..."
5. **Consider edge cases**: Empty graph, single node, cycles
6. **Optimize**: Can you stop early? Need all paths?

---

## 🔗 Connection to Previous Topics

### Graphs + Trees:

Trees are special graphs!

```python
# Tree = Connected acyclic undirected graph
# Tree with n nodes has exactly n-1 edges
# Graph traversals work on trees too
```

### Graphs + BFS/DFS:

```python
# BFS uses Queue (from earlier lessons!)
# DFS uses Stack or recursion (from backtracking!)
# BFS finds shortest path (unweighted)
# DFS explores all possibilities
```

### Graphs + HashMaps:

```python
# Adjacency list IS a hashmap!
# Track visited nodes with set (hashmap)
# Clone graph uses hashmap to map original→clone
```

### Graphs + Backtracking:

```python
# Many graph problems use backtracking
# Find all paths from A to B
# Generate all valid states
```

---

## 🎓 Advanced Graph Concepts

### Shortest Path Algorithms:

**1. Dijkstra's Algorithm (Weighted, Non-negative):**

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # (distance, node)

    while pq:
        dist, node = heapq.heappop(pq)

        if dist > distances[node]:
            continue

        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return distances
```

**2. Bellman-Ford (Handles negative weights):**

```python
def bellman_ford(graph, start, n):
    distances = [float('inf')] * n
    distances[start] = 0

    # Relax edges n-1 times
    for _ in range(n - 1):
        for u in graph:
            for v, weight in graph[u]:
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight

    # Check for negative cycles
    for u in graph:
        for v, weight in graph[u]:
            if distances[u] + weight < distances[v]:
                return None  # Negative cycle!

    return distances
```

### Minimum Spanning Tree:

**Prim's Algorithm:**

```python
import heapq

def prim(graph, start):
    mst = []
    visited = {start}
    edges = [(weight, start, to) for to, weight in graph[start]]
    heapq.heapify(edges)

    while edges:
        weight, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.append((frm, to, weight))

            for next_to, next_weight in graph[to]:
                if next_to not in visited:
                    heapq.heappush(edges, (next_weight, to, next_to))

    return mst
```

### Union-Find (Disjoint Set):

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        # Union by rank
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1

        return True

# Used for: Cycle detection, connected components
```

---

## 📊 Graph Algorithms Complexity

| Algorithm            | Time           | Space | Use Case                   |
| -------------------- | -------------- | ----- | -------------------------- |
| **DFS**              | O(V+E)         | O(V)  | Cycle detection, paths     |
| **BFS**              | O(V+E)         | O(V)  | Shortest path (unweighted) |
| **Dijkstra**         | O((V+E) log V) | O(V)  | Shortest path (weighted)   |
| **Bellman-Ford**     | O(VE)          | O(V)  | Negative weights           |
| **Floyd-Warshall**   | O(V³)          | O(V²) | All pairs shortest paths   |
| **Prim's MST**       | O(E log V)     | O(V)  | Minimum spanning tree      |
| **Kruskal's MST**    | O(E log E)     | O(V)  | Minimum spanning tree      |
| **Topological Sort** | O(V+E)         | O(V)  | DAG ordering               |

---

## 🌟 The Graph Mindset

**Key Questions:**

1. "What are the nodes and edges?"
2. "Is it directed or undirected?"
3. "Are edges weighted?"
4. "Do I need shortest path or just any path?"
5. "Could there be cycles?"

**Mental Model:**

- **Graph = Relationships**
- Everything connected to everything? → Think graphs!
- Need to explore connections? → Traversal
- Need shortest path? → BFS (unweighted) or Dijkstra (weighted)
- Need to detect cycles? → DFS with state tracking

**Core Insight:**
Graphs model the real world - social networks, maps, dependencies, recommendations. Master graphs and you can model almost anything!

---

Happy Learning! 🕸️ Master graphs and you'll understand how connections shape data!
