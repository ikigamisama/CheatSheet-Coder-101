# Recursive Backtracking - Learning Guide 🔙

## 📚 What Are We Learning Today?

Think of this lesson as learning to explore ALL possible paths in a maze, but being smart enough to TURN BACK when you hit a dead end!

---

## 🔙 Part 1: Understanding Backtracking

### The Kid-Friendly Explanation

Imagine you're in a **corn maze** trying to find the exit:

- You pick a path and walk down it
- If you hit a **dead end**, you **go back** to the last intersection
- Try a **different path** from there
- Keep trying until you find the exit (or try all paths)

**Other real-life examples:**

- Solving a Sudoku puzzle: Try a number, if it doesn't work, erase it and try another
- Choosing classes: Pick courses, if schedule conflicts, undo and try different combination
- Packing a suitcase: Try items, if too heavy, remove some and try others
- Chess: Try a move, if it leads to loss, undo and try different move

**That's Backtracking!** It's a problem-solving technique where you:

1. **Choose**: Make a choice/decision
2. **Explore**: Recursively explore that choice
3. **Unchoose** (Backtrack): If it doesn't work, undo the choice and try another

### The Technical Explanation

**Backtracking** is an algorithmic technique for finding all (or some) solutions by:

- **Incrementally building** candidates to solutions
- **Abandoning** candidates ("backtracking") when they're determined to be invalid
- Using **recursion** to explore the solution space

**Key Characteristics:**

1. **DFS-based**: Explores depth-first through decision tree
2. **Prunes invalid paths**: Stops early when path is invalid
3. **Explores all possibilities**: Finds all solutions (not just one)
4. **Uses recursion naturally**: Each call explores one choice
5. **State restoration**: Undoes choices when backtracking

**The Backtracking Template:**

```python
def backtrack(candidate):
    if is_solution(candidate):
        output(candidate)
        return

    # Iterate through all possible choices
    for choice in get_choices(candidate):
        if is_valid(choice):
            make_choice(choice)           # Choose
            backtrack(candidate)          # Explore
            undo_choice(choice)           # Unchoose (Backtrack!)
```

### Visual Representation:

```
Decision Tree for choosing 2 numbers from [1,2,3]:

                    []
         /          |          \
       [1]         [2]         [3]
      /   \         |           |
   [1,2] [1,3]   [2,3]       [3,?] ← No more choices
     ✓     ✓       ✓

Backtracking path:
1. Choose 1 → [1]
2. Choose 2 → [1,2] ✓ Solution!
3. Undo 2 → [1]
4. Choose 3 → [1,3] ✓ Solution!
5. Undo 3, Undo 1 → []
6. Choose 2 → [2]
7. Choose 3 → [2,3] ✓ Solution!
8. Done!

Backtracking vs Brute Force:
Brute Force: Try ALL combinations → [1,1], [1,2], [2,1], [2,2], etc.
Backtracking: PRUNE invalid early → Skip [1,1] immediately
```

### 🔬 Why Data Scientists & Data Engineers Love Backtracking

1. **Combinatorial Problems**: Generate all combinations, permutations
2. **Constraint Satisfaction**: Solve puzzles with constraints (Sudoku, N-Queens)
3. **Feature Selection**: Find optimal feature combinations
4. **Path Finding**: Explore all paths in graphs
5. **Configuration Search**: Find valid system configurations
6. **Pattern Matching**: Find all matching patterns in data
7. **Optimization**: Search solution space intelligently
8. **Rule Mining**: Discover rules that satisfy constraints
9. **Hyperparameter Tuning**: Explore parameter combinations

**Real Data Science Example:**

```python
# Find all valid feature combinations that meet accuracy threshold
def find_feature_sets(features, current_set, accuracy_threshold):
    if meets_accuracy(current_set, accuracy_threshold):
        valid_sets.append(current_set.copy())
        return

    for feature in features:
        if feature not in current_set:
            current_set.add(feature)        # Choose
            find_feature_sets(features, current_set, accuracy_threshold)  # Explore
            current_set.remove(feature)     # Unchoose
```

---

## 🎨 Part 2: Backtracking Patterns

### Pattern 1: Combinations

Generate all k-size subsets.

- **Key**: Start index prevents duplicates
- **Use**: Feature selection, subset problems

### Pattern 2: Permutations

Generate all orderings.

- **Key**: Track used elements
- **Use**: Arrangement problems, scheduling

### Pattern 3: Subsets (Power Set)

Generate all possible subsets (2^n).

- **Key**: Include/exclude decision for each element
- **Use**: All possible configurations

### Pattern 4: Constraint Satisfaction

Find solutions that meet conditions.

- **Key**: Check validity before recursing
- **Use**: Puzzles, valid configurations

### The Backtracking Checklist:

- ✅ Base case (solution found or no more choices)
- ✅ Loop through choices
- ✅ Check if choice is valid
- ✅ Make choice (modify state)
- ✅ Recurse with new state
- ✅ Undo choice (restore state)

---

## 💻 Practice Problems

### Problem 1: Subsets (Power Set)

**Problem**: Generate all possible subsets of a set.

**Example**:

- Input: `nums = [1, 2, 3]`
- Output: `[[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]`

**Solution:**

```python
def subsets(nums):
    result = []

    def backtrack(start, current_subset):
        # Every path is a valid subset!
        result.append(current_subset.copy())

        # Try adding each remaining number
        for i in range(start, len(nums)):
            current_subset.append(nums[i])     # Choose
            backtrack(i + 1, current_subset)   # Explore
            current_subset.pop()               # Unchoose

    backtrack(0, [])
    return result

# Test
print(subsets([1, 2, 3]))
# Output: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

**Decision Tree Visualization:**

```
Input: [1, 2, 3]

                        []  ← Add to result
            /           |           \
          [1]          [2]          [3]
         /   \          |
     [1,2]  [1,3]     [2,3]
       |
    [1,2,3]

Execution trace:
1. [] → Add to result
2. Choose 1 → [1] → Add to result
3. Choose 2 → [1,2] → Add to result
4. Choose 3 → [1,2,3] → Add to result
5. Backtrack → [1,2]
6. Backtrack → [1]
7. Choose 3 → [1,3] → Add to result
8. Backtrack → [1]
9. Backtrack → []
10. Choose 2 → [2] → Add to result
11. Choose 3 → [2,3] → Add to result
12. Backtrack → [2]
13. Backtrack → []
14. Choose 3 → [3] → Add to result

Total: 2^3 = 8 subsets
```

**Alternative Approach (Include/Exclude Decision):**

```python
def subsets_v2(nums):
    result = []

    def backtrack(index, current_subset):
        if index == len(nums):
            result.append(current_subset.copy())
            return

        # Exclude nums[index]
        backtrack(index + 1, current_subset)

        # Include nums[index]
        current_subset.append(nums[index])
        backtrack(index + 1, current_subset)
        current_subset.pop()

    backtrack(0, [])
    return result
```

**Why This Matters for Data Science**:

- **Feature Combinations**: Try all feature subsets
- **A/B Testing**: All possible variant combinations
- **Configuration**: All possible settings
- **Rule Mining**: Generate candidate rule sets

**Time Complexity:** O(2^n × n) - 2^n subsets, each takes O(n) to copy
**Space Complexity:** O(n) - recursion depth

---

### Problem 2: Permutations

**Problem**: Generate all possible orderings of array elements.

**Example**:

- Input: `nums = [1, 2, 3]`
- Output: `[[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]`

**Solution:**

```python
def permute(nums):
    result = []

    def backtrack(current_perm):
        # Base case: permutation complete
        if len(current_perm) == len(nums):
            result.append(current_perm.copy())
            return

        # Try each number that hasn't been used
        for num in nums:
            if num not in current_perm:
                current_perm.append(num)       # Choose
                backtrack(current_perm)        # Explore
                current_perm.pop()             # Unchoose

    backtrack([])
    return result

# Test
print(permute([1, 2, 3]))
# Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

**Optimized with Used Set:**

```python
def permute_optimized(nums):
    result = []

    def backtrack(current_perm, used):
        if len(current_perm) == len(nums):
            result.append(current_perm.copy())
            return

        for i, num in enumerate(nums):
            if i not in used:
                current_perm.append(num)
                used.add(i)                    # Choose
                backtrack(current_perm, used)  # Explore
                used.remove(i)                 # Unchoose
                current_perm.pop()

    backtrack([], set())
    return result
```

**Decision Tree for [1,2,3]:**

```
                    []
        /           |           \
      [1]          [2]          [3]
     /   \        /   \        /   \
  [1,2] [1,3]  [2,1] [2,3]  [3,1] [3,2]
    |     |      |     |      |     |
[1,2,3][1,3,2][2,1,3][2,3,1][3,1,2][3,2,1]

Each leaf is a valid permutation!
Total: 3! = 6 permutations
```

**Step-by-Step for [1,2]:**

```
Start: current_perm = []

1. Choose 1 → [1]
   2. Choose 2 → [1,2] ← Complete! Add to result
   3. Undo 2 → [1]
4. Undo 1 → []

5. Choose 2 → [2]
   6. Choose 1 → [2,1] ← Complete! Add to result
   7. Undo 1 → [2]
8. Undo 2 → []

Result: [[1,2], [2,1]]
```

**Why This Matters for Data Engineering**:

- **Job Scheduling**: All possible execution orders
- **Pipeline Ordering**: Valid DAG execution sequences
- **Resource Allocation**: Different assignment orders
- **Test Case Generation**: All input orderings

**Time Complexity:** O(n! × n) - n! permutations, each takes O(n) to build
**Space Complexity:** O(n) - recursion depth

---

### Problem 3: N-Queens (Constraint Satisfaction)

**Problem**: Place N queens on N×N chessboard such that no two queens attack each other.

**Constraints:**

- No two queens in same row
- No two queens in same column
- No two queens in same diagonal

**Example for N=4:**

```
Solution 1:       Solution 2:
. Q . .           . . Q .
. . . Q           Q . . .
Q . . .           . . . Q
. . Q .           . Q . .
```

**Solution:**

```python
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]

    def is_valid(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # Check diagonal (top-left)
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1

        # Check diagonal (top-right)
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        return True

    def backtrack(row):
        # Base case: all queens placed
        if row == n:
            result.append([''.join(r) for r in board])
            return

        # Try placing queen in each column
        for col in range(n):
            if is_valid(row, col):
                board[row][col] = 'Q'      # Choose
                backtrack(row + 1)         # Explore
                board[row][col] = '.'      # Unchoose

    backtrack(0)
    return result

# Test
solutions = solve_n_queens(4)
for i, solution in enumerate(solutions):
    print(f"Solution {i+1}:")
    for row in solution:
        print(row)
    print()
```

**Optimized with Sets (Track Attacked Positions):**

```python
def solve_n_queens_optimized(n):
    result = []
    board = [['.'] * n for _ in range(n)]

    # Track attacked columns and diagonals
    cols = set()
    diag1 = set()  # row - col is constant on \ diagonal
    diag2 = set()  # row + col is constant on / diagonal

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return

        for col in range(n):
            # Check if position is attacked
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            # Place queen
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            # Remove queen (backtrack)
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result
```

**Decision Tree for N=4:**

```
Row 0: Try each column
[Q,.,.,.]  [.,Q,.,.]  [.,.,Q,.]  [.,.,.,Q]
   |          |           |          |
Row 1: Try valid positions (not attacked)
   |          |           |          |
Row 2: Try valid positions
   |          |           |          |
Row 3: Try valid positions
   |          |           |          |
   ↓          ↓           ↓          ↓
Solution?  Solution?  Solution?  Solution?

Most branches get pruned early (invalid)!
```

**Why This Matters for Data Science**:

- **Constraint Optimization**: Schedule with constraints
- **Resource Allocation**: Assign resources with conflicts
- **Experiment Design**: Valid experimental configurations
- **Feature Engineering**: Valid feature combinations
- **Data Validation**: Check constraint satisfaction

**Time Complexity:** O(n!) - pruned by constraints
**Space Complexity:** O(n) - recursion depth

---

## 🎯 Key Takeaways

### When to Use Backtracking:

- ✅ Need **ALL solutions** (not just one)
- ✅ Problem has **constraints** to satisfy
- ✅ Solution built **incrementally** (step by step)
- ✅ Can **validate partial solutions** early
- ✅ **Combinatorial** problems (permutations, combinations)
- ✅ Keywords: "all possible", "generate all", "find all"

### Backtracking vs Other Techniques:

| Technique               | Use Case                                   | Finds All? | Prunes?              |
| ----------------------- | ------------------------------------------ | ---------- | -------------------- |
| **Backtracking**        | All solutions with constraints             | ✅ Yes     | ✅ Yes               |
| **Greedy**              | One optimal solution                       | ❌ No      | ✅ Yes               |
| **Dynamic Programming** | Optimal solution (overlapping subproblems) | ❌ No      | ✅ Yes (memoization) |
| **Brute Force**         | All solutions                              | ✅ Yes     | ❌ No                |

### The Three Pillars of Backtracking:

1. **Choice**: What decisions can I make?
2. **Constraints**: When is a choice invalid?
3. **Goal**: When have I found a solution?

---

## 🚀 Next Steps for Practice

### LeetCode Easy:

- Subsets ✅ (covered)
- Permutations ✅ (covered)
- Letter Combinations of Phone Number
- Binary Tree Paths

### LeetCode Medium:

- N-Queens ✅ (covered)
- Combination Sum
- Palindrome Partitioning
- Word Search
- Generate Parentheses
- Subsets II (with duplicates)

### LeetCode Hard:

- N-Queens II (count solutions)
- Sudoku Solver
- Word Search II
- Regular Expression Matching

### Data Science Projects:

1. **Feature Selection**: Find all feature combinations meeting accuracy
2. **Constraint Solver**: Build rule-based configuration validator
3. **Pattern Finder**: Generate all patterns matching criteria
4. **Hyperparameter Search**: Explore parameter combinations
5. **Data Generator**: Create synthetic test datasets with constraints

---

## 💡 Pro Tips

### The Backtracking Recipe:

```python
def backtrack(state, choices, constraints, result):
    # Base case: found solution
    if is_complete(state):
        result.append(state.copy())  # Don't forget to copy!
        return

    # Try each choice
    for choice in choices:
        if is_valid(choice, constraints):
            # Make choice (modify state)
            make_choice(state, choice)

            # Explore (recurse)
            backtrack(state, remaining_choices, constraints, result)

            # Undo choice (restore state)
            undo_choice(state, choice)
```

### Common Mistakes:

- ❌ Forgetting to copy the solution (all solutions become same!)
  - ✅ Use `result.append(state.copy())`
- ❌ Not undoing the choice (backtracking fails!)
  - ✅ Always undo after recursion
- ❌ Invalid constraint checking
  - ✅ Test constraints thoroughly
- ❌ Modifying list during iteration
  - ✅ Use index or iterate over copy
- ❌ Not handling duplicates
  - ✅ Sort + skip duplicates for unique solutions

### Optimization Techniques:

**1. Early Pruning:**

```python
if not can_possibly_succeed(current_state):
    return  # Don't even try
```

**2. Sort for Better Pruning:**

```python
nums.sort()  # Similar elements grouped
for i in range(len(nums)):
    if i > 0 and nums[i] == nums[i-1]:
        continue  # Skip duplicate
```

**3. Use Sets for O(1) Lookup:**

```python
used = set()  # Instead of 'if x in list' (O(n))
```

### Interview Tips:

1. **Draw the decision tree**: Visualize choices at each step
2. **State the three pillars**: Choices, constraints, goal
3. **Start with base case**: "When do I have a solution?"
4. **Test small example**: Walk through [1,2] before [1,2,3,4]
5. **Mention pruning**: "I'll check validity before recursing"
6. **Don't forget to copy**: When adding to result!

---

## 🔗 Connection to Previous Topics

### Backtracking + Recursion:

Backtracking IS advanced recursion!

```python
# Every backtracking problem uses recursion
# Recursion explores depth-first
# Backtracking adds "undo" step
```

### Backtracking + Trees:

Decision tree structure:

```python
# Each recursive call = going deeper in tree
# Each backtrack = going up one level
# Leaves = complete solutions
```

### Backtracking + DFS:

Backtracking uses DFS traversal:

```python
# DFS explores one path completely
# Backtracking does DFS with early stopping
# Both use stack (recursion = implicit stack)
```

### Backtracking + Dynamic Programming:

Sometimes combined:

```python
# DP: Overlapping subproblems → memoization
# Backtracking: Independent choices → no memoization
# Can combine for constrained optimization
```

---

## 🎓 Advanced Concepts

### Iterative Backtracking (Using Stack):

```python
def permute_iterative(nums):
    result = []
    stack = [([], set())]  # (current_perm, used_indices)

    while stack:
        perm, used = stack.pop()

        if len(perm) == len(nums):
            result.append(perm)
            continue

        for i in range(len(nums)):
            if i not in used:
                new_perm = perm + [nums[i]]
                new_used = used | {i}
                stack.append((new_perm, new_used))

    return result
```

### Backtracking with Memoization:

```python
# When subproblems repeat (rare in backtracking)
memo = {}

def backtrack_with_memo(state):
    state_key = tuple(state)  # Make hashable
    if state_key in memo:
        return memo[state_key]

    # ... normal backtracking logic ...

    memo[state_key] = result
    return result
```

### Branch and Bound (Optimization):

```python
# Like backtracking but for optimization
# Track best solution found so far
# Prune branches that can't beat it

best_solution = None
best_value = float('-inf')

def backtrack(state, current_value):
    global best_solution, best_value

    # Prune if can't possibly beat best
    if upper_bound(state) <= best_value:
        return

    if is_complete(state):
        if current_value > best_value:
            best_value = current_value
            best_solution = state.copy()
        return

    # ... continue backtracking ...
```

---

## 📊 Complexity Analysis

### Time Complexity Patterns:

| Problem      | Choices per Level   | Levels | Time Complexity |
| ------------ | ------------------- | ------ | --------------- |
| Subsets      | 2 (include/exclude) | n      | O(2^n)          |
| Permutations | n, n-1, n-2, ...    | n      | O(n!)           |
| Combinations | n, n-1, n-2, ...    | k      | O(C(n,k))       |
| N-Queens     | n, n, n, ...        | n      | O(n!) - pruned  |

### Space Complexity:

- **Recursion stack**: O(depth) - usually O(n)
- **Tracking state**: O(n) - current solution
- **Result storage**: O(solutions × solution_size)

**Total**: O(n) for the search, O(results) for storage

---

## 🌟 The Backtracking Mindset

**Key Questions:**

1. "What choices do I have at each step?"
2. "When is a choice invalid?"
3. "When have I found a complete solution?"
4. "How do I undo this choice?"

**Mental Model:**
Think of backtracking as **exploring a maze**:

- Each intersection = make a choice
- Dead end = invalid, go back
- Exit = solution found!
- Multiple exits = find all solutions

**Core Insight:**
Backtracking = DFS + Constraint checking + Undo mechanism

**When someone says "find all...", think BACKTRACKING!**

---

## 🎯 Backtracking Problem Categories

### 1. Enumeration:

- Subsets, Permutations, Combinations
- Generate all possibilities

### 2. Constraint Satisfaction:

- N-Queens, Sudoku, Graph Coloring
- Find valid configurations

### 3. Optimization:

- Traveling Salesman (with pruning)
- Knapsack (explore all combinations)

### 4. Parsing:

- Regular expressions, Expression evaluation
- Match patterns exhaustively

---

Happy Learning! 🔙 Master backtracking and you'll solve complex combinatorial problems like a pro!
