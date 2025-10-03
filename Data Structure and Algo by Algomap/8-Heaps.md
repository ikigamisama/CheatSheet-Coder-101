# Trees - Learning Guide 🌳

## 📚 What Are We Learning Today?

Think of this lesson as learning about family trees, organization charts, and how computers organize information in a branching structure!

---

## 🌳 Part 1: Understanding Trees

### The Kid-Friendly Explanation

Imagine your **family tree**:

- You have **parents** (1 level up)
- Your parents have **parents** (grandparents - 2 levels up)
- You might have **siblings** (same level)
- You might have **children** (1 level down)
- Your children might have **children** (grandchildren - 2 levels down)

**Other real-life examples:**

- Company organization chart (CEO → Managers → Employees)
- Folder structure on your computer (Main folder → Subfolders → Files)
- Tournament brackets (Champion ← Winners ← First round)
- Decision tree (Should I? → Yes/No → More questions)

**That's a Tree in programming!** A hierarchical structure where:

- One element is the **root** (top)
- Each element can have **children**
- Elements with no children are **leaves**

### The Technical Explanation

A **Tree** is a non-linear data structure with:

- **Nodes**: Elements that store data
- **Edges**: Connections between nodes
- **Root**: The topmost node (no parent)
- **Parent**: Node with children
- **Child**: Node connected below another node
- **Leaf**: Node with no children
- **Height**: Longest path from root to leaf
- **Depth**: Distance from root to a node

**Key Properties:**

- Exactly **ONE root** node
- Every node (except root) has **exactly ONE parent**
- **No cycles** (can't loop back)
- N nodes have **N-1 edges**

```
Example Tree:
        1       ← Root (depth 0, height 3)
       / \
      2   3     ← Children of 1 (depth 1)
     / \   \
    4   5   6   ← Grandchildren (depth 2)
   /
  7             ← Great-grandchild (depth 3), Leaf

Node 4 is a leaf (no children)
Node 5 is a leaf
Node 6 is a leaf
Node 7 is a leaf
Node 2 has children: [4, 5]
Height of tree: 3 (longest path: 1→2→4→7)
```

### Tree Terminology:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val        # Data stored
        self.left = left      # Left child
        self.right = right    # Right child
```

**Types of Trees:**

1. **Binary Tree**: Each node has at most 2 children
2. **Binary Search Tree (BST)**: Binary tree with ordering (left < parent < right)
3. **Balanced Tree**: Height difference between subtrees is minimal
4. **Complete Tree**: All levels filled except possibly last
5. **Full Tree**: Every node has 0 or 2 children

### Visual Representation:

```
Binary Tree:
        5
       / \
      3   8
     / \   \
    1   4   10

Binary Search Tree (BST):
        7
       / \
      3   10
     / \    \
    1   5    12

Notice: Left < Parent < Right (for every node!)
1 < 3 < 5 < 7 < 10 < 12

Complete Binary Tree:
        1
       / \
      2   3
     / \  /
    4  5 6

Full Binary Tree:
        1
       / \
      2   3
     / \
    4   5
```

### 🔬 Why Data Scientists & Data Engineers Love Trees

1. **Decision Trees**: ML models for classification/regression
2. **File Systems**: Organize hierarchical data
3. **Database Indexes**: B-trees for fast lookups
4. **XML/JSON Parsing**: Represent nested structures
5. **Expression Evaluation**: Parse mathematical/logical expressions
6. **Hierarchical Data**: Represent org charts, taxonomies
7. **Search Optimization**: Binary search trees for O(log n) operations
8. **Data Compression**: Huffman coding trees
9. **Route Planning**: Decision trees for pathfinding

**Real Data Science Example:**

```python
# Decision Tree for customer segmentation
if income > 50000:
    if age > 30:
        category = "Premium"
    else:
        category = "Growth"
else:
    if loyalty_years > 5:
        category = "Loyal"
    else:
        category = "Basic"
```

---

## 🎨 Part 2: Tree Traversal Methods

**Traversal** = Visiting every node in a specific order

### The Three Main Traversals (for Binary Trees):

**1. Inorder (Left → Root → Right)**

- Visit left subtree
- Visit root
- Visit right subtree
- **BST property**: Visits nodes in sorted order!

**2. Preorder (Root → Left → Right)**

- Visit root first
- Visit left subtree
- Visit right subtree
- **Use**: Copy tree, create prefix expression

**3. Postorder (Left → Right → Root)**

- Visit left subtree
- Visit right subtree
- Visit root last
- **Use**: Delete tree, postfix expression

**4. Level Order (BFS - Breadth First Search)**

- Visit level by level, left to right
- **Use**: Find shortest path, level-wise processing

### Visual Example:

```
Tree:
        1
       / \
      2   3
     / \
    4   5

Inorder (Left-Root-Right): 4, 2, 5, 1, 3
Preorder (Root-Left-Right): 1, 2, 4, 5, 3
Postorder (Left-Right-Root): 4, 5, 2, 3, 1
Level Order: 1, 2, 3, 4, 5
```

---

## 💻 Practice Problems

### Problem 1: Maximum Depth of Binary Tree (DFS)

**Problem**: Find the maximum depth (height) of a binary tree.

**Example**:

```
Input:
    3
   / \
  9  20
    /  \
   15   7

Output: 3 (path: 3 → 20 → 15 or 7)
```

**Solution (Recursive DFS):**

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth(root):
    # Base case: empty tree
    if not root:
        return 0

    # Recursive case: 1 + max of left and right subtrees
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)

    return 1 + max(left_depth, right_depth)

# Test
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

print(max_depth(root))  # Output: 3
```

**How It Works (Recursive Thinking):**

```
max_depth(3)
├─ max_depth(9)
│  ├─ max_depth(None) = 0
│  └─ max_depth(None) = 0
│  → return 1 + max(0, 0) = 1
└─ max_depth(20)
   ├─ max_depth(15)
   │  ├─ max_depth(None) = 0
   │  └─ max_depth(None) = 0
   │  → return 1 + max(0, 0) = 1
   └─ max_depth(7)
      ├─ max_depth(None) = 0
      └─ max_depth(None) = 0
      → return 1 + max(0, 0) = 1
   → return 1 + max(1, 1) = 2
→ return 1 + max(1, 2) = 3 ✓
```

**Iterative Solution (BFS):**

```python
from collections import deque

def max_depth_iterative(root):
    if not root:
        return 0

    queue = deque([(root, 1)])  # (node, depth)
    max_depth = 0

    while queue:
        node, depth = queue.popleft()
        max_depth = max(max_depth, depth)

        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))

    return max_depth
```

**Why This Matters for Data Science**:

- **Feature Depth**: Analyze nested data structures
- **Decision Tree Depth**: Prevent overfitting (limit depth)
- **Hierarchy Analysis**: Measure organizational levels
- **Data Quality**: Validate nested JSON/XML depth

---

### Problem 2: Invert Binary Tree (Tree Manipulation)

**Problem**: Invert (mirror) a binary tree.

**Example**:

```
Input:           Output:
     4               4
   /   \           /   \
  2     7         7     2
 / \   / \       / \   / \
1   3 6   9     9   6 3   1
```

**Solution:**

```python
def invert_tree(root):
    # Base case
    if not root:
        return None

    # Swap left and right children
    root.left, root.right = root.right, root.left

    # Recursively invert subtrees
    invert_tree(root.left)
    invert_tree(root.right)

    return root

# Helper function to print tree (level order)
def print_tree(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)

    # Remove trailing None values
    while result and result[-1] is None:
        result.pop()

    return result

# Test
root = TreeNode(4)
root.left = TreeNode(2)
root.right = TreeNode(7)
root.left.left = TreeNode(1)
root.left.right = TreeNode(3)
root.right.left = TreeNode(6)
root.right.right = TreeNode(9)

print("Before:", print_tree(root))
invert_tree(root)
print("After:", print_tree(root))
```

**Step-by-Step:**

```
Original:
     4
   /   \
  2     7
 / \   / \
1   3 6   9

Step 1: At node 4, swap children
     4
   /   \
  7     2    ← Swapped!
 / \   / \
6   9 1   3

Step 2: Recurse left (node 7), swap children
     4
   /   \
  7     2
 / \   / \
9   6 1   3  ← Swapped 6 and 9

Step 3: Recurse right (node 2), swap children
     4
   /   \
  7     2
 / \   / \
9   6 3   1  ← Swapped 1 and 3

Done!
```

**Why This Matters for Data Engineering**:

- **Data Transformation**: Restructure hierarchical data
- **Mirror Databases**: Create reverse hierarchies
- **Tree Rebalancing**: Optimize tree structures
- **Visual Representations**: Flip organizational charts

---

### Problem 3: Validate Binary Search Tree (BST Property)

**Problem**: Check if a binary tree is a valid BST.

**BST Rules:**

- Left subtree values < node value
- Right subtree values > node value
- Both subtrees must also be BSTs

**Example**:

```
Valid BST:       Invalid BST:
    5                5
   / \              / \
  1   7            1   4  ← 4 < 5, but on right!
     / \              / \
    6   8            3   6

Output: True     Output: False
```

**Common Mistake - This is WRONG:**

```python
# This only checks immediate children!
def is_valid_bst_wrong(root):
    if not root:
        return True

    if root.left and root.left.val >= root.val:
        return False
    if root.right and root.right.val <= root.val:
        return False

    return is_valid_bst_wrong(root.left) and is_valid_bst_wrong(root.right)

# Fails for:
#      5
#     / \
#    1   6
#       / \
#      4   7
# Node 4 < 5, but on right side of 5!
```

**Correct Solution (Track Min/Max Range):**

```python
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    # Empty tree is valid BST
    if not root:
        return True

    # Current node must be within range
    if root.val <= min_val or root.val >= max_val:
        return False

    # Left subtree: all values must be < root.val
    # Right subtree: all values must be > root.val
    return (is_valid_bst(root.left, min_val, root.val) and
            is_valid_bst(root.right, root.val, max_val))

# Test
valid_bst = TreeNode(5)
valid_bst.left = TreeNode(1)
valid_bst.right = TreeNode(7)
valid_bst.right.left = TreeNode(6)
valid_bst.right.right = TreeNode(8)

invalid_bst = TreeNode(5)
invalid_bst.left = TreeNode(1)
invalid_bst.right = TreeNode(4)

print(is_valid_bst(valid_bst))    # True
print(is_valid_bst(invalid_bst))  # False
```

**How Range Tracking Works:**

```
Tree:    5
        / \
       1   7
          / \
         6   8

is_valid_bst(5, -inf, inf)
├─ is_valid_bst(1, -inf, 5)     ← 1 must be < 5 ✓
│  ├─ is_valid_bst(None) → True
│  └─ is_valid_bst(None) → True
│  → True
└─ is_valid_bst(7, 5, inf)       ← 7 must be > 5 ✓
   ├─ is_valid_bst(6, 5, 7)      ← 6 must be between 5 and 7 ✓
   │  → True
   └─ is_valid_bst(8, 7, inf)    ← 8 must be > 7 ✓
      → True
   → True
→ True ✓

Invalid example:
    5
   / \
  1   4  ← 4 is checked with range (5, inf)
         4 is NOT > 5, so False!
```

**Why This Matters for Data Science**:

- **Data Validation**: Ensure sorted data integrity
- **Index Verification**: Validate database index structure
- **Decision Tree Validation**: Check model consistency
- **Data Quality**: Verify hierarchical relationships

---

## 🎯 Key Takeaways

### Tree Traversal Cheat Sheet:

| Traversal       | Order           | Use Case                      | Implementation  |
| --------------- | --------------- | ----------------------------- | --------------- |
| **Inorder**     | Left-Root-Right | Get sorted values (BST)       | Recursive/Stack |
| **Preorder**    | Root-Left-Right | Copy tree, prefix notation    | Recursive/Stack |
| **Postorder**   | Left-Right-Root | Delete tree, postfix notation | Recursive/Stack |
| **Level Order** | Level by level  | Shortest path, BFS            | Queue           |

### DFS vs BFS:

**DFS (Depth-First Search):**

- Uses: Stack (or recursion)
- Goes deep before wide
- Good for: Tree traversal, finding paths
- Space: O(height)

**BFS (Breadth-First Search):**

- Uses: Queue
- Goes wide before deep
- Good for: Shortest path, level-wise processing
- Space: O(width)

### Common Tree Patterns:

1. **Recursion**: Most tree problems use recursion naturally
2. **Base case**: Always handle `if not root: return ...`
3. **Divide & Conquer**: Solve for subtrees, combine results
4. **Level Order**: Use queue for BFS
5. **Path Tracking**: Use DFS with backtracking

---

## 🚀 Next Steps for Practice

### LeetCode Easy:

- Maximum Depth of Binary Tree ✅ (covered)
- Invert Binary Tree ✅ (covered)
- Same Tree
- Symmetric Tree
- Merge Two Binary Trees
- Diameter of Binary Tree

### LeetCode Medium:

- Validate Binary Search Tree ✅ (covered)
- Binary Tree Level Order Traversal
- Lowest Common Ancestor
- Binary Tree Right Side View
- Path Sum II
- Kth Smallest Element in BST

### Data Science Projects:

1. **Decision Tree Visualizer**: Build tool to visualize ML decision trees
2. **File System Navigator**: Implement directory tree traversal
3. **JSON Parser**: Parse and validate nested JSON structures
4. **Org Chart Analyzer**: Analyze company hierarchy depth/breadth
5. **Expression Evaluator**: Build calculator using expression trees

---

## 💡 Pro Tips

### The Recursive Pattern for Trees:

```python
def tree_function(root):
    # 1. Base case (empty tree)
    if not root:
        return base_value

    # 2. Process current node
    # ... do something with root.val

    # 3. Recurse on children
    left_result = tree_function(root.left)
    right_result = tree_function(root.right)

    # 4. Combine results
    return combine(root.val, left_result, right_result)
```

### Common Mistakes:

- ❌ Forgetting base case (leads to infinite recursion!)
- ❌ Modifying tree structure while traversing
- ❌ Not considering single-node tree (root only)
- ❌ Confusing tree height vs depth
- ❌ Checking only immediate children (not entire subtree)

### Interview Tips:

1. **Draw the tree**: Always visualize the problem
2. **Start with base case**: Empty tree, single node
3. **Think recursively**: "If I solve for subtrees, how do I combine?"
4. **State traversal type**: "I'll use inorder/preorder/level order"
5. **Consider edge cases**: Unbalanced tree, all left/right children

### Time Complexity:

| Operation | Binary Tree | BST (Balanced) | BST (Worst) |
| --------- | ----------- | -------------- | ----------- |
| Search    | O(n)        | O(log n)       | O(n)        |
| Insert    | -           | O(log n)       | O(n)        |
| Delete    | -           | O(log n)       | O(n)        |
| Traversal | O(n)        | O(n)           | O(n)        |
| Space     | O(h)        | O(log n)       | O(n)        |

h = height of tree

---

## 🔗 Connection to Previous Topics

### Trees + Recursion:

Trees are THE perfect use case for recursion!

```python
# Every tree problem follows this pattern
def solve_tree(root):
    if not root:  # Base case
        return something

    left = solve_tree(root.left)   # Trust recursion!
    right = solve_tree(root.right)

    return combine(root.val, left, right)
```

### Trees + Stack:

DFS traversals can use explicit stack:

```python
def inorder_iterative(root):
    stack = []
    result = []
    current = root

    while current or stack:
        while current:
            stack.append(current)
            current = current.left

        current = stack.pop()
        result.append(current.val)
        current = current.right

    return result
```

### Trees + Queue (BFS):

```python
from collections import deque

def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

### Trees + HashMap:

Track node properties:

```python
# Find node to parent mapping
def build_parent_map(root):
    parent_map = {root: None}
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if node.left:
            parent_map[node.left] = node
            queue.append(node.left)
        if node.right:
            parent_map[node.right] = node
            queue.append(node.right)

    return parent_map
```

---

## 🎓 Advanced Concepts

### Binary Search Tree Operations:

**Insert:**

```python
def insert_bst(root, val):
    if not root:
        return TreeNode(val)

    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)

    return root
```

**Search:**

```python
def search_bst(root, val):
    if not root or root.val == val:
        return root

    if val < root.val:
        return search_bst(root.left, val)
    else:
        return search_bst(root.right, val)
```

### Tree Balancing (AVL/Red-Black):

Keep tree height O(log n) through rotations:

- Used in: Databases, file systems
- Guarantees: O(log n) operations
- Trade-off: More complex insertion/deletion

### Trie (Prefix Tree):

Special tree for string operations:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

# Used for: Autocomplete, spell checkers, IP routing
```

---

## 📊 Tree Types Summary

| Tree Type        | Property                        | Use Case            |
| ---------------- | ------------------------------- | ------------------- |
| **Binary Tree**  | ≤2 children                     | General hierarchy   |
| **BST**          | Sorted (left < root < right)    | Fast search/insert  |
| **Balanced BST** | Height difference ≤ 1           | Guaranteed O(log n) |
| **Complete**     | All levels filled (except last) | Heaps               |
| **Full**         | 0 or 2 children                 | Expression trees    |
| **Trie**         | Character-by-character          | String operations   |
| **B-Tree**       | Multiple keys per node          | Databases           |

---

## 🌟 The Tree Mindset

**Key Questions:**

1. "What's the base case?" (Empty tree!)
2. "Can I solve for left and right, then combine?"
3. "Do I need to visit all nodes or can I stop early?"
4. "What traversal order makes sense?"

**Mental Model:**
Trees are like Russian dolls - each subtree is itself a tree! Solve for the smallest piece, then the solution works for the whole.

---

Happy Learning! 🌳 Master trees and you'll understand how hierarchical data really works!
