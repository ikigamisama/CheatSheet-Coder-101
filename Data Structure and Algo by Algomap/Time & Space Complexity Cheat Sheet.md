# Time & Space Complexity Cheat Sheet

## Big O Notation Basics

**Big O** describes the upper bound of time/space complexity - worst-case scenario.

### Common Complexities (Best to Worst)

1. **O(1)** - Constant
2. **O(log n)** - Logarithmic
3. **O(n)** - Linear
4. **O(n log n)** - Linearithmic
5. **O(n²)** - Quadratic
6. **O(n³)** - Cubic
7. **O(2ⁿ)** - Exponential
8. **O(n!)** - Factorial

## Time Complexity Examples

### O(1) - Constant Time

```python
# Array access, hash table lookup
arr[0]  # Always takes same time regardless of array size
```

### O(log n) - Logarithmic

```python
# Binary search, balanced tree operations
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        # Eliminates half the search space each iteration
```

### O(n) - Linear

```python
# Single loop through array
for i in range(n):
    print(i)  # Executes n times
```

### O(n²) - Quadratic

```python
# Nested loops
for i in range(n):
    for j in range(n):
        print(i, j)  # Executes n × n times
```

### O(2ⁿ) - Exponential

```python
# Naive recursive fibonacci
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)  # Branches exponentially
```

## Data Structure Complexities

### Arrays

**Purpose**: Store elements in contiguous memory locations for fast access by index.

**Sample Code Examples**:

```python
# 1. Access - O(1)
arr = [10, 20, 30, 40, 50]
element = arr[2]  # Direct access to index 2 → 30

# 2. Search - O(n)
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# 3. Insert at end - O(1)
arr.append(60)  # [10, 20, 30, 40, 50, 60]

# 4. Insert at middle - O(n)
arr.insert(2, 25)  # Shifts elements: [10, 20, 25, 30, 40, 50, 60]

# 5. Delete - O(n)
arr.remove(30)  # Shifts elements: [10, 20, 25, 40, 50, 60]
```

### Linked Lists

**Purpose**: Dynamic data structure where elements (nodes) are stored in sequence but not in contiguous memory locations.

**Sample Code Examples**:

```python
class ListNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None

# 1. Access - O(n)
def access_node(head, index):
    current = head
    for i in range(index):
        if current is None:
            return None
        current = current.next
    return current

# 2. Search - O(n)
def search(head, target):
    current = head
    while current:
        if current.val == target:
            return current
        current = current.next
    return None

# 3. Insert at head - O(1)
def insert_head(head, val):
    new_node = ListNode(val)
    new_node.next = head
    return new_node

# 4. Insert at tail - O(n)
def insert_tail(head, val):
    if not head:
        return ListNode(val)
    current = head
    while current.next:  # Traverse to end
        current = current.next
    current.next = ListNode(val)
    return head

# 5. Delete - O(n)
def delete_node(head, val):
    if head and head.val == val:
        return head.next
    current = head
    while current and current.next:
        if current.next.val == val:
            current.next = current.next.next
            break
        current = current.next
    return head
```

### Hash Tables (Dictionary/HashMap)

**Purpose**: Provide fast key-value lookups using hash functions to map keys to array indices.

**Sample Code Examples**:

```python
# 1. Search/Access - O(1) average
hash_table = {"apple": 5, "banana": 3, "orange": 8}
value = hash_table["apple"]  # Direct access → 5

# 2. Insert - O(1) average
hash_table["grape"] = 12  # Hash function maps "grape" to index

# 3. Delete - O(1) average
del hash_table["banana"]  # Remove key-value pair

# 4. Hash collision handling (worst case O(n))
def custom_hash_table():
    # When multiple keys hash to same index
    table = [[] for _ in range(10)]  # Chaining approach

    def insert(key, value):
        index = hash(key) % 10
        # Linear search in chain (worst case O(n))
        for i, (k, v) in enumerate(table[index]):
            if k == key:
                table[index][i] = (key, value)
                return
        table[index].append((key, value))

# 5. Resize operation - O(n)
def resize_hash_table(old_table):
    new_size = len(old_table) * 2
    new_table = [[] for _ in range(new_size)]
    # Rehash all elements
    for chain in old_table:
        for key, value in chain:
            new_index = hash(key) % new_size
            new_table[new_index].append((key, value))
```

### Binary Search Trees

**Purpose**: Maintain sorted data with efficient search, insertion, and deletion operations using binary tree structure.

**Sample Code Examples**:

```python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

# 1. Search - O(log n) average, O(n) worst case
def search_bst(root, target):
    if not root or root.val == target:
        return root
    if target < root.val:
        return search_bst(root.left, target)
    return search_bst(root.right, target)

# 2. Insert - O(log n) average, O(n) worst case
def insert_bst(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)
    return root

# 3. Delete - O(log n) average, O(n) worst case
def delete_bst(root, val):
    if not root:
        return root
    if val < root.val:
        root.left = delete_bst(root.left, val)
    elif val > root.val:
        root.right = delete_bst(root.right, val)
    else:
        # Node to delete found
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        # Node has two children
        min_node = find_min(root.right)
        root.val = min_node.val
        root.right = delete_bst(root.right, min_node.val)
    return root

# 4. Worst case scenario - O(n) (degenerate tree)
def create_degenerate_tree():
    # Inserting sorted data creates linked list
    root = None
    for i in range(1, 6):  # 1,2,3,4,5
        root = insert_bst(root, i)
    # Tree becomes: 1->2->3->4->5 (right skewed)

# 5. Balanced tree operations - O(log n)
def height(root):
    if not root:
        return 0
    return 1 + max(height(root.left), height(root.right))
```

### Heaps (Priority Queue)

**Purpose**: Maintain partial order where parent nodes have priority over children, enabling efficient min/max operations.

**Sample Code Examples**:

```python
import heapq

# 1. Find Min - O(1)
heap = [1, 3, 6, 5, 9, 8]
heapq.heapify(heap)  # Convert to heap
min_element = heap[0]  # Root is always minimum

# 2. Insert - O(log n)
def heap_insert(heap, val):
    heapq.heappush(heap, val)
    # Bubbles up to maintain heap property

# 3. Delete Min - O(log n)
def extract_min(heap):
    if heap:
        return heapq.heappop(heap)
    # Last element moves to root, then bubbles down

# 4. Build Heap - O(n)
def build_heap(arr):
    # Bottom-up heapification is O(n), not O(n log n)
    heapq.heapify(arr)
    return arr

# 5. Max Heap implementation (Python has min heap by default)
class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        heapq.heappush(self.heap, -val)  # Negate for max heap

    def pop(self):
        return -heapq.heappop(self.heap) if self.heap else None

    def peek(self):
        return -self.heap[0] if self.heap else None

# Example usage
max_heap = MaxHeap()
for val in [3, 1, 6, 5, 2, 4]:
    max_heap.push(val)
# peek() returns 6 (maximum element)
```

## Sorting Algorithms

### Bubble Sort

**Purpose**: Simple comparison-based sorting that repeatedly steps through the list, compares adjacent elements and swaps them if they're in wrong order.

**Sample Code Examples**:

```python
# 1. Basic Bubble Sort - O(n²) worst/average, O(n) best
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# 2. Optimized Bubble Sort - O(n) best case for sorted array
def optimized_bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # Array is sorted
            break
    return arr

# 3. Best case scenario - O(n)
sorted_array = [1, 2, 3, 4, 5]
result = optimized_bubble_sort(sorted_array.copy())  # Only one pass needed

# 4. Worst case scenario - O(n²)
reverse_array = [5, 4, 3, 2, 1]
result = bubble_sort(reverse_array.copy())  # Maximum swaps needed

# 5. Space complexity - O(1)
def bubble_sort_in_place(arr):
    # Sorts without using extra space
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
```

### Merge Sort

**Purpose**: Divide-and-conquer algorithm that divides array into halves, sorts them separately, then merges them back together.

**Sample Code Examples**:

```python
# 1. Standard Merge Sort - O(n log n) all cases
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

# 2. In-place Merge Sort - O(n log n) time, O(log n) space
def merge_sort_inplace(arr, start=0, end=None):
    if end is None:
        end = len(arr) - 1

    if start < end:
        mid = (start + end) // 2
        merge_sort_inplace(arr, start, mid)
        merge_sort_inplace(arr, mid + 1, end)
        merge_inplace(arr, start, mid, end)

# 3. Merge operation demonstration
def merge_example():
    left = [1, 3, 5, 7]
    right = [2, 4, 6, 8]
    merged = merge(left, right)  # [1, 2, 3, 4, 5, 6, 7, 8]

# 4. Recursion tree depth - O(log n) space
def analyze_recursion_depth(n):
    # For array of size n, recursion depth is log₂(n)
    import math
    depth = math.ceil(math.log2(n))
    return depth  # Stack space used

# 5. Stable sorting property
def stability_demo():
    # Elements with same value maintain relative order
    pairs = [(3, 'a'), (1, 'b'), (3, 'c'), (2, 'd')]
    # After merge sort: [(1, 'b'), (2, 'd'), (3, 'a'), (3, 'c')]
    # Notice (3, 'a') comes before (3, 'c') - stable!
```

### Quick Sort

**Purpose**: Efficient divide-and-conquer algorithm that selects a pivot element and partitions array around it.

**Sample Code Examples**:

```python
# 1. Standard Quick Sort - O(n log n) average, O(n²) worst
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot_index = partition(arr, low, high)
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)

def partition(arr, low, high):
    pivot = arr[high]  # Choose last element as pivot
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# 2. Randomized Quick Sort - Avoid worst case
import random

def randomized_quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    if low < high:
        # Random pivot selection
        random_index = random.randint(low, high)
        arr[random_index], arr[high] = arr[high], arr[random_index]

        pivot_index = partition(arr, low, high)
        randomized_quick_sort(arr, low, pivot_index - 1)
        randomized_quick_sort(arr, pivot_index + 1, high)

# 3. Best case scenario - O(n log n)
def best_case_example():
    # Pivot always divides array into equal halves
    arr = [4, 2, 6, 1, 3, 5, 7]  # Pivot 4 divides evenly

# 4. Worst case scenario - O(n²)
def worst_case_example():
    # Already sorted array with last element as pivot
    arr = [1, 2, 3, 4, 5, 6, 7]
    # Each partition creates unbalanced split (0 vs n-1)

# 5. Space complexity analysis - O(log n) average
def space_analysis():
    # Best case: Balanced partitions = O(log n) recursion depth
    # Worst case: Unbalanced partitions = O(n) recursion depth
    pass
```

### Heap Sort

**Purpose**: Uses binary heap data structure to sort array by repeatedly extracting maximum/minimum element.

**Sample Code Examples**:

```python
# 1. Complete Heap Sort - O(n log n) all cases
def heap_sort(arr):
    n = len(arr)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Move max to end
        heapify(arr, i, 0)  # Restore heap property

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

# 2. Build Max Heap - O(n) operation
def build_max_heap(arr):
    n = len(arr)
    # Start from last non-leaf node
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

# 3. Extract Maximum - O(log n) operation
def extract_max(arr, heap_size):
    if heap_size <= 0:
        return None

    max_val = arr[0]
    arr[0] = arr[heap_size - 1]  # Move last to root
    heapify(arr, heap_size - 1, 0)  # Restore heap
    return max_val

# 4. Heap property verification
def is_max_heap(arr, i=0):
    n = len(arr)
    left = 2 * i + 1
    right = 2 * i + 2

    # Check if current node violates max heap property
    if left < n and arr[i] < arr[left]:
        return False
    if right < n and arr[i] < arr[right]:
        return False

    # Recursively check subtrees
    if left < n and not is_max_heap(arr, left):
        return False
    if right < n and not is_max_heap(arr, right):
        return False

    return True

# 5. In-place sorting - O(1) space complexity
def heap_sort_demo():
    arr = [12, 11, 13, 5, 6, 7]
    print(f"Original: {arr}")
    heap_sort(arr)
    print(f"Sorted: {arr}")
    # No extra space used except for variables
```

### Counting Sort

**Purpose**: Non-comparison based sorting for integers within a specific range, counting occurrences of each element.

**Sample Code Examples**:

```python
# 1. Basic Counting Sort - O(n + k) where k is range
def counting_sort(arr):
    if not arr:
        return arr

    # Find range
    max_val = max(arr)
    min_val = min(arr)
    range_size = max_val - min_val + 1

    # Count occurrences
    count = [0] * range_size
    for num in arr:
        count[num - min_val] += 1

    # Reconstruct sorted array
    result = []
    for i, freq in enumerate(count):
        result.extend([i + min_val] * freq)

    return result

# 2. Stable Counting Sort - Maintains relative order
def stable_counting_sort(arr):
    if not arr:
        return arr

    max_val = max(arr)
    count = [0] * (max_val + 1)

    # Count occurrences
    for num in arr:
        count[num] += 1

    # Convert to cumulative count
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # Build result array
    result = [0] * len(arr)
    for i in range(len(arr) - 1, -1, -1):
        result[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1

    return result

# 3. Optimal case - Small range, large array
def optimal_case():
    # Array with values 0-9, but 1000 elements
    arr = [random.randint(0, 9) for _ in range(1000)]
    # k=10, n=1000, so O(n+k) = O(1010) is very efficient

# 4. Worst case - Large range, small array
def worst_case():
    # Array with values 0-1000000, but only 5 elements
    arr = [0, 500000, 1000000, 250000, 750000]
    # k=1000001, n=5, so O(n+k) = O(1000006) is inefficient

# 5. Character sorting example
def count_sort_chars(string):
    # Sort characters in a string
    count = [0] * 256  # ASCII range

    for char in string:
        count[ord(char)] += 1

    result = []
    for i, freq in enumerate(count):
        if freq > 0:
            result.extend([chr(i)] * freq)

    return ''.join(result)

# Example: count_sort_chars("hello") → "ehllo"
```

## Search Algorithms

### Linear Search

**Purpose**: Simple search algorithm that checks every element in sequence until target is found or end is reached.

**Sample Code Examples**:

```python
# 1. Basic Linear Search - O(n) time, O(1) space
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# 2. Search with multiple occurrences
def linear_search_all(arr, target):
    indices = []
    for i in range(len(arr)):
        if arr[i] == target:
            indices.append(i)
    return indices

# 3. Early termination optimization
def linear_search_optimized(arr, target):
    for i, element in enumerate(arr):
        if element == target:
            return i
        # Could add conditions to stop early if array has properties
    return -1

# 4. Best case scenario - O(1)
def best_case_demo():
    arr = [5, 2, 8, 1, 9]
    index = linear_search(arr, 5)  # Found at index 0, first check

# 5. Worst case scenario - O(n)
def worst_case_demo():
    arr = [1, 2, 3, 4, 5]
    index = linear_search(arr, 5)  # Found at last position
    not_found = linear_search(arr, 6)  # Check all elements, not found
```

### Binary Search

**Purpose**: Efficient search algorithm for sorted arrays that repeatedly divides search space in half.

**Sample Code Examples**:

```python
# 1. Iterative Binary Search - O(log n) time, O(1) space
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# 2. Recursive Binary Search - O(log n) time, O(log n) space
def binary_search_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = (left + right) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# 3. Find first occurrence
def binary_search_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result

# 4. Search in rotated sorted array
def search_rotated(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid

        # Left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1

# 5. Binary search bounds demonstration
def search_complexity_demo(n):
    # For array of size n, max comparisons = log₂(n)
    import math
    max_comparisons = math.ceil(math.log2(n))
    print(f"Array size: {n}, Max comparisons: {max_comparisons}")

    # Example: size 1000 → max 10 comparisons
    # vs linear search: potentially 1000 comparisons
```

### Depth-First Search (DFS)

**Purpose**: Graph traversal algorithm that explores as far as possible along each branch before backtracking.

**Sample Code Examples**:

```python
# 1. DFS for Tree - O(V) time, O(V) space
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

def dfs_tree_recursive(root, target):
    if not root:
        return False

    if root.val == target:
        return True

    return dfs_tree_recursive(root.left, target) or \
           dfs_tree_recursive(root.right, target)

# 2. DFS for Graph - O(V + E) time, O(V) space
def dfs_graph(graph, start, target, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)

    if start == target:
        return True

    for neighbor in graph[start]:
        if neighbor not in visited:
            if dfs_graph(graph, neighbor, target, visited):
                return True

    return False

# 3. DFS Iterative with Stack
def dfs_iterative(graph, start, target):
    stack = [start]
    visited = set()

    while stack:
        node = stack.pop()

        if node == target:
            return True

        if node not in visited:
            visited.add(node)
            # Add neighbors to stack
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)

    return False

# 4. DFS Path Finding
def dfs_find_path(graph, start, target, path=None, visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()

    path.append(start)
    visited.add(start)

    if start == target:
        return path.copy()

    for neighbor in graph[start]:
        if neighbor not in visited:
            result = dfs_find_path(graph, neighbor, target, path, visited)
            if result:
                return result

    path.pop()  # Backtrack
    return None

# 5. DFS Applications Example
def connected_components(graph):
    visited = set()
    components = []

    def dfs_component(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs_component(neighbor, component)

    for node in graph:
        if node not in visited:
            component = []
            dfs_component(node, component)
            components.append(component)

    return components
```

### Breadth-First Search (BFS)

**Purpose**: Graph traversal algorithm that explores all neighbors at current depth before moving to next depth level.

**Sample Code Examples**:

```python
from collections import deque

# 1. BFS for Tree - O(V) time, O(V) space
def bfs_tree(root, target):
    if not root:
        return False

    queue = deque([root])

    while queue:
        node = queue.popleft()

        if node.val == target:
            return True

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return False

# 2. BFS for Graph - O(V + E) time, O(V) space
def bfs_graph(graph, start, target):
    queue = deque([start])
    visited = set([start])

    while queue:
        node = queue.popleft()

        if node == target:
            return True

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False

# 3. BFS Shortest Path (unweighted graph)
def bfs_shortest_path(graph, start, target):
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        node, path = queue.popleft()

        if node == target:
            return path

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None

# 4. BFS Level Order Traversal
def bfs_levels(root):
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

# 5. BFS Applications - Minimum steps/distance
def min_steps_to_reach(graph, start, target):
    queue = deque([(start, 0)])  # (node, distance)
    visited = set([start])

    while queue:
        node, dist = queue.popleft()

        if node == target:
            return dist

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return -1  # Target not reachable
```

## Space Complexity

### What Counts as Space?

- **Input Space**: Space used by input (usually not counted)
- **Auxiliary Space**: Extra space used by algorithm
- **Total Space**: Input + Auxiliary space

### Examples

#### O(1) Space - Constant

```python
def swap(a, b):
    temp = a  # Only uses fixed amount of extra variables
    a = b
    b = temp
```

#### O(n) Space - Linear

```python
def create_copy(arr):
    copy = []
    for item in arr:
        copy.append(item)  # Space grows with input size
    return copy
```

#### O(log n) Space - Logarithmic

```python
def binary_search_recursive(arr, target, left, right):
    if left > right: return -1
    mid = (left + right) // 2
    # Recursion depth is log n (call stack space)
```

## Quick Rules for Analysis

### Time Complexity Rules

1. **Drop constants**: O(2n) → O(n)
2. **Drop lower terms**: O(n² + n) → O(n²)
3. **Nested loops multiply**: Two nested loops = O(n²)
4. **Sequential operations add**: O(n) + O(m) = O(n + m)

### Common Patterns

- **Single loop**: O(n)
- **Nested loops**: O(n²), O(n³), etc.
- **Divide and conquer**: Often O(n log n)
- **Tree traversal**: O(n) where n = number of nodes
- **Graph traversal**: O(V + E) where V = vertices, E = edges

## Optimization Tips

### Improve Time Complexity

- Use hash tables for O(1) lookups instead of O(n) searches
- Use binary search on sorted data instead of linear search
- Consider divide-and-conquer approaches
- Use dynamic programming for overlapping subproblems

### Improve Space Complexity

- Use iteration instead of recursion when possible
- Reuse variables instead of creating new ones
- Use in-place algorithms when data modification is allowed
- Consider space-time tradeoffs

## Memory Hierarchy (Fastest to Slowest)

1. **Registers** - CPU registers
2. **Cache** - L1, L2, L3 cache
3. **RAM** - Main memory
4. **Disk** - Hard drive/SSD
5. **Network** - Remote storage

## Quick Reference Card

| Complexity | Name         | Example             |
| ---------- | ------------ | ------------------- |
| O(1)       | Constant     | Array access        |
| O(log n)   | Logarithmic  | Binary search       |
| O(n)       | Linear       | Linear search       |
| O(n log n) | Linearithmic | Merge sort          |
| O(n²)      | Quadratic    | Bubble sort         |
| O(2ⁿ)      | Exponential  | Recursive fibonacci |

**Remember**: Big O describes the growth rate as input size approaches infinity!
