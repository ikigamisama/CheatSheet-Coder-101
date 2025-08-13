# Data Structures & Algorithms Python Cheat Sheet

## 🔍 Searching Algorithms

### Linear Search

**Definition:** Sequentially checks each element in a list until the target is found or the list ends.

**Time Complexity:** O(n) | **Space Complexity:** O(1)

**Usage Sites:**

1. Searching in unsorted arrays
2. Finding first occurrence of an element
3. Small datasets where simplicity matters
4. When data structure doesn't support indexing
5. Searching in linked lists
6. Finding all occurrences of an element
7. Real-time systems with predictable behavior
8. When memory is extremely limited
9. Searching in streams of data
10. Educational purposes and algorithm basics

**Examples:**

```python
# Example 1: Basic Linear Search
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

arr = [64, 34, 25, 12, 22, 11, 90]
print(linear_search(arr, 22))  # Output: 4

# Example 2: Find all occurrences
def find_all_occurrences(arr, target):
    indices = []
    for i in range(len(arr)):
        if arr[i] == target:
            indices.append(i)
    return indices

arr = [1, 3, 2, 3, 4, 3, 5]
print(find_all_occurrences(arr, 3))  # Output: [1, 3, 5]

# Example 3: Linear search with custom condition
def find_first_even(arr):
    for i, num in enumerate(arr):
        if num % 2 == 0:
            return i
    return -1

arr = [1, 3, 7, 8, 9]
print(find_first_even(arr))  # Output: 3

# Example 4: Search in string
def search_char(text, char):
    for i, c in enumerate(text):
        if c == char:
            return i
    return -1

print(search_char("hello", "l"))  # Output: 2

# Example 5: Search with early termination
def search_with_limit(arr, target, max_checks):
    for i in range(min(len(arr), max_checks)):
        if arr[i] == target:
            return i
    return -1
```

### Binary Search

**Definition:** Efficiently finds a target value in a sorted array by repeatedly dividing the search interval in half.

**Time Complexity:** O(log n) | **Space Complexity:** O(1)

**Usage Sites:**

1. Searching in sorted arrays
2. Finding insertion point for maintaining sorted order
3. Finding square root of a number
4. Peak finding in arrays
5. Search in rotated sorted arrays
6. Finding first/last occurrence in duplicates
7. Search in infinite sorted arrays
8. Database indexing systems
9. Finding minimum in rotated sorted array
10. Search for range in sorted array

**Examples:**

```python
# Example 1: Basic Binary Search
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

arr = [2, 3, 4, 10, 40]
print(binary_search(arr, 10))  # Output: 3

# Example 2: Find insertion point
def find_insertion_point(arr, target):
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

arr = [1, 3, 5, 6]
print(find_insertion_point(arr, 4))  # Output: 2

# Example 3: Find first occurrence
def find_first_occurrence(arr, target):
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

arr = [2, 4, 4, 4, 8]
print(find_first_occurrence(arr, 4))  # Output: 1

# Example 4: Square root using binary search
def sqrt_binary_search(x):
    if x < 2:
        return x

    left, right = 1, x // 2
    while left <= right:
        mid = (left + right) // 2
        square = mid * mid

        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    return right

print(sqrt_binary_search(8))  # Output: 2

# Example 5: Search in rotated sorted array
def search_rotated_array(arr, target):
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

arr = [4, 5, 6, 7, 0, 1, 2]
print(search_rotated_array(arr, 0))  # Output: 4
```

## 🔄 Sorting Algorithms

### Bubble Sort

**Definition:** Repeatedly steps through the list, compares adjacent elements and swaps them if they're in wrong order.

**Time Complexity:** O(n²) | **Space Complexity:** O(1)

**Usage Sites:**

1. Educational purposes to understand sorting
2. Small datasets (< 50 elements)
3. Nearly sorted data (optimized version)
4. Simple implementation requirements
5. Memory-constrained environments
6. Debugging and testing other algorithms
7. When stability is required and simplicity preferred
8. Embedded systems with limited resources
9. Quick prototyping
10. Interview practice problems

**Examples:**

```python
# Example 1: Basic Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

arr = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(arr.copy()))  # Output: [11, 12, 22, 25, 34, 64, 90]

# Example 2: Optimized Bubble Sort (early termination)
def bubble_sort_optimized(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # No swaps means array is sorted
            break
    return arr

# Example 3: Bubble Sort with comparison count
def bubble_sort_with_count(arr):
    n = len(arr)
    comparisons = 0

    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr, comparisons

arr = [5, 2, 8, 1, 9]
sorted_arr, count = bubble_sort_with_count(arr.copy())
print(f"Sorted: {sorted_arr}, Comparisons: {count}")

# Example 4: Descending Bubble Sort
def bubble_sort_desc(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] < arr[j + 1]:  # Changed comparison
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

arr = [3, 7, 1, 9, 4]
print(bubble_sort_desc(arr.copy()))  # Output: [9, 7, 4, 3, 1]

# Example 5: Bubble Sort for strings
def bubble_sort_strings(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j].lower() > arr[j + 1].lower():
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

words = ["banana", "Apple", "cherry", "Date"]
print(bubble_sort_strings(words.copy()))  # Output: ['Apple', 'banana', 'cherry', 'Date']
```

### Quick Sort

**Definition:** Divide-and-conquer algorithm that picks a pivot element and partitions array around it.

**Time Complexity:** O(n log n) average, O(n²) worst | **Space Complexity:** O(log n)

**Usage Sites:**

1. General-purpose sorting (most programming languages)
2. Large datasets requiring fast average performance
3. In-place sorting requirements
4. When average case performance is more important than worst case
5. Sorting arrays (not linked lists)
6. Cache-efficient sorting needs
7. Embedded systems with recursion support
8. Competitive programming
9. Database query optimization
10. Scientific computing applications

**Examples:**

```python
# Example 1: Basic Quick Sort
def quicksort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)

arr = [3, 6, 8, 10, 1, 2, 1]
print(quicksort(arr))  # Output: [1, 1, 2, 3, 6, 8, 10]

# Example 2: In-place Quick Sort
def quicksort_inplace(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_index - 1)
        quicksort_inplace(arr, pivot_index + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

arr = [10, 7, 8, 9, 1, 5]
quicksort_inplace(arr)
print(arr)  # Output: [1, 5, 7, 8, 9, 10]

# Example 3: Quick Sort with random pivot
import random

def quicksort_random(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    if low < high:
        # Choose random pivot
        random_index = random.randint(low, high)
        arr[random_index], arr[high] = arr[high], arr[random_index]

        pivot_index = partition(arr, low, high)
        quicksort_random(arr, low, pivot_index - 1)
        quicksort_random(arr, pivot_index + 1, high)

# Example 4: Iterative Quick Sort
def quicksort_iterative(arr):
    if len(arr) <= 1:
        return arr

    stack = [(0, len(arr) - 1)]

    while stack:
        low, high = stack.pop()

        if low < high:
            pivot_index = partition(arr, low, high)
            stack.append((low, pivot_index - 1))
            stack.append((pivot_index + 1, high))

    return arr

# Example 5: Quick Sort with custom comparator
def quicksort_custom(arr, compare_func=None):
    if compare_func is None:
        compare_func = lambda x, y: x < y

    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if compare_func(x, pivot)]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if compare_func(pivot, x)]

    return quicksort_custom(left, compare_func) + middle + quicksort_custom(right, compare_func)

# Sort by string length
words = ["python", "java", "c", "javascript", "go"]
sorted_by_length = quicksort_custom(words, lambda x, y: len(x) < len(y))
print(sorted_by_length)  # Output: ['c', 'go', 'java', 'python', 'javascript']
```

### Merge Sort

**Definition:** Divide-and-conquer algorithm that divides array into halves, sorts them, then merges back together.

**Time Complexity:** O(n log n) | **Space Complexity:** O(n)

**Usage Sites:**

1. Stable sorting requirements
2. Large datasets
3. Linked list sorting
4. External sorting (data doesn't fit in memory)
5. When consistent O(n log n) performance is needed
6. Parallel processing environments
7. Sorting objects with complex comparison
8. When worst-case performance matters
9. Database management systems
10. Version control systems (merging changes)

**Examples:**

```python
# Example 1: Basic Merge Sort
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

arr = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(arr))  # Output: [3, 9, 10, 27, 38, 43, 82]

# Example 2: In-place Merge Sort
def merge_sort_inplace(arr, left=0, right=None):
    if right is None:
        right = len(arr) - 1

    if left < right:
        mid = (left + right) // 2
        merge_sort_inplace(arr, left, mid)
        merge_sort_inplace(arr, mid + 1, right)
        merge_inplace(arr, left, mid, right)

def merge_inplace(arr, left, mid, right):
    # Create temporary arrays
    left_arr = arr[left:mid + 1]
    right_arr = arr[mid + 1:right + 1]

    i = j = 0
    k = left

    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] <= right_arr[j]:
            arr[k] = left_arr[i]
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1
        k += 1

    while i < len(left_arr):
        arr[k] = left_arr[i]
        i += 1
        k += 1

    while j < len(right_arr):
        arr[k] = right_arr[j]
        j += 1
        k += 1

# Example 3: Merge Sort for linked lists (using list simulation)
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_sort_linked_list(head):
    if not head or not head.next:
        return head

    # Split the list into two halves
    mid = get_middle(head)
    next_to_mid = mid.next
    mid.next = None

    left = merge_sort_linked_list(head)
    right = merge_sort_linked_list(next_to_mid)

    return merge_sorted_lists(left, right)

def get_middle(head):
    if not head:
        return head

    slow = fast = head
    prev = None

    while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next

    return prev

def merge_sorted_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    current.next = l1 or l2
    return dummy.next

# Example 4: Bottom-up Merge Sort (iterative)
def merge_sort_bottom_up(arr):
    if len(arr) <= 1:
        return arr

    arr = arr.copy()
    n = len(arr)
    size = 1

    while size < n:
        for start in range(0, n, size * 2):
            mid = min(start + size - 1, n - 1)
            end = min(start + size * 2 - 1, n - 1)

            if mid < end:
                merge_range(arr, start, mid, end)

        size *= 2

    return arr

def merge_range(arr, left, mid, right):
    left_arr = arr[left:mid + 1]
    right_arr = arr[mid + 1:right + 1]

    i = j = 0
    k = left

    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] <= right_arr[j]:
            arr[k] = left_arr[i]
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1
        k += 1

    while i < len(left_arr):
        arr[k] = left_arr[i]
        i += 1
        k += 1

    while j < len(right_arr):
        arr[k] = right_arr[j]
        j += 1
        k += 1

# Example 5: Merge K sorted arrays
def merge_k_sorted_arrays(arrays):
    if not arrays:
        return []

    while len(arrays) > 1:
        merged_arrays = []

        for i in range(0, len(arrays), 2):
            arr1 = arrays[i]
            arr2 = arrays[i + 1] if i + 1 < len(arrays) else []
            merged_arrays.append(merge(arr1, arr2))

        arrays = merged_arrays

    return arrays[0]

arrays = [[1, 4, 5], [1, 3, 4], [2, 6]]
print(merge_k_sorted_arrays(arrays))  # Output: [1, 1, 2, 3, 4, 4, 5, 6]
```

## 📚 Data Structures

### Stack

**Definition:** Last-In-First-Out (LIFO) data structure where elements are added and removed from the same end.

**Time Complexity:** Push/Pop O(1) | **Space Complexity:** O(n)

**Usage Sites:**

1. Function call management (call stack)
2. Expression evaluation and syntax parsing
3. Undo operations in applications
4. Browser history navigation
5. Balanced parentheses checking
6. Depth-First Search (DFS)
7. Memory management
8. Compiler design and parsing
9. Backtracking algorithms
10. Convert infix to postfix notation

**Examples:**

```python
# Example 1: Basic Stack Implementation
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

# Usage
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())  # Output: 3
print(stack.peek())  # Output: 2

# Example 2: Balanced Parentheses Checker
def is_balanced_parentheses(expression):
    stack = []
    opening = "({["
    closing = ")}]"
    pairs = {"(": ")", "{": "}", "[": "]"}

    for char in expression:
        if char in opening:
            stack.append(char)
        elif char in closing:
            if not stack:
                return False
            if pairs[stack.pop()] != char:
                return False

    return len(stack) == 0

print(is_balanced_parentheses("({[]})"))  # Output: True
print(is_balanced_parentheses("({[})"))   # Output: False

# Example 3: Evaluate Postfix Expression
def evaluate_postfix(expression):
    stack = []
    operators = {'+', '-', '*', '/'}

    for token in expression.split():
        if token not in operators:
            stack.append(float(token))
        else:
            b = stack.pop()
            a = stack.pop()

            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(a / b)

    return stack[0]

print(evaluate_postfix("2 3 1 * + 9 -"))  # Output: -4.0

# Example 4: Next Greater Element
def next_greater_element(arr):
    stack = []
    result = [-1] * len(arr)

    for i in range(len(arr) - 1, -1, -1):
        while stack and stack[-1] <= arr[i]:
            stack.pop()

        if stack:
            result[i] = stack[-1]

        stack.append(arr[i])

    return result

arr = [4, 5, 2, 25, 7, 8]
print(next_greater_element(arr))  # Output: [5, 25, 25, -1, 8, -1]

# Example 5: Undo/Redo System
class UndoRedoSystem:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []
        self.current_state = ""

    def execute_command(self, command):
        self.undo_stack.append(self.current_state)
        self.current_state = command
        self.redo_stack.clear()  # Clear redo stack on new command

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.current_state)
            self.current_state = self.undo_stack.pop()
            return self.current_state
        return "Nothing to undo"

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.current_state)
            self.current_state = self.redo_stack.pop()
            return self.current_state
        return "Nothing to redo"

# Usage
editor = UndoRedoSystem()
editor.execute_command("Type: Hello")
editor.execute_command("Type: World")
print(editor.undo())  # Output: Type: Hello
print(editor.redo())  # Output: Type: World
```

### Queue

**Definition:** First-In-First-Out (FIFO) data structure where elements are added at rear and removed from front.

**Time Complexity:** Enqueue/Dequeue O(1) | **Space Complexity:** O(n)

**Usage Sites:**

1. Process scheduling in operating systems
2. Breadth-First Search (BFS)
3. Print job management
4. Handling requests in web servers
5. Buffer for data streams
6. Level-order tree traversal
7. Cache implementation (FIFO cache)
8. Simulation systems (customer service)
9. Producer-consumer problems
10. Network packet routing

**Examples:**

```python
# Example 1: Basic Queue Implementation
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        return None

    def front(self):
        if not self.is_empty():
            return self.items[0]
        return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

# Usage
queue = Queue()
queue.enqueue("first")
queue.enqueue("second")
queue.enqueue("third")
print(queue.dequeue())  # Output: first
print(queue.front())    # Output: second

# Example 2: Circular Queue
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front_idx = 0
        self.rear_idx = -1
        self.size_val = 0

    def enqueue(self, item):
        if self.size_val == self.capacity:
            return False  # Queue is full

        self.rear_idx = (self.rear_idx + 1) % self.capacity
        self.queue[self.rear_idx] = item
        self.size_val += 1
        return True

    def dequeue(self):
        if self.size_val == 0:
            return None  # Queue is empty

        item = self.queue[self.front_idx]
        self.queue[self.front_idx] = None
        self.front_idx = (self.front_idx + 1) % self.capacity
        self.size_val -= 1
        return item

    def is_full(self):
        return self.size_val == self.capacity

    def is_empty(self):
        return self.size_val == 0

# Example 3: Priority Queue (Min-Heap based)
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def enqueue(self, item, priority):
        # Use counter to break ties and maintain insertion order
        heapq.heappush(self.heap, (priority, self.counter, item))
        self.counter += 1

    def dequeue(self):
        if self.heap:
            return heapq.heappop(self.heap)[2]  # Return only the item
        return None

    def is_empty(self):
        return len(self.heap) == 0

# Usage
pq = PriorityQueue()
pq.enqueue("Low priority task", 3)
pq.enqueue("High priority task", 1)
pq.enqueue("Medium priority task", 2)
print(pq.dequeue())  # Output: High priority task

# Example 4: BFS using Queue
def bfs_graph(graph, start):
    visited = set()
    queue = Queue()
    result = []

    queue.enqueue(start)
    visited.add(start)

    while not queue.is_empty():
        vertex = queue.dequeue()
        result.append(vertex)

        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.enqueue(neighbor)

    return result

# Graph represented as adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
print(bfs_graph(graph, 'A'))  # Output: ['A', 'B', 'C', 'D', 'E', 'F']

# Example 5: Hot Potato Game (Josephus Problem)
def hot_potato(names, num):
    queue = Queue()

    for name in names:
        queue.enqueue(name)

    while queue.size() > 1:
        # Pass the potato num times
        for _ in range(num):
            queue.enqueue(queue.dequeue())

        # Remove the person holding the potato
        eliminated = queue.dequeue()
        print(f"{eliminated} is eliminated")

    return queue.dequeue()  # Winner

names = ["Alice", "Bob", "Charlie", "David", "Eve"]
winner = hot_potato(names, 3)
print(f"Winner: {winner}")
```

### Linked List

**Definition:** Linear data structure where elements are stored in nodes, each containing data and reference to next node.

**Time Complexity:** Access O(n), Insert/Delete O(1) at known position | **Space Complexity:** O(n)

**Usage Sites:**

1. Dynamic memory allocation
2. Implementation of other data structures (stacks, queues)
3. Undo functionality in applications
4. Music playlist (next/previous song)
5. Browser history navigation
6. Memory management in operating systems
7. Symbol tables in compilers
8. Representing polynomials in mathematics
9. Image viewer (next/previous image)
10. Blockchain implementation

**Examples:**

```python
# Example 1: Basic Singly Linked List
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def append(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1

    def prepend(self, val):
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    def delete(self, val):
        if not self.head:
            return False

        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return True

        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        return False

    def display(self):
        result = []
        current = self.head
        while current:
            result.append(current.val)
            current = current.next
        return result

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.prepend(0)
print(ll.display())  # Output: [0, 1, 2, 3]
ll.delete(2)
print(ll.display())  # Output: [0, 1, 3]

# Example 2: Doubly Linked List
class DoublyListNode:
    def __init__(self, val=0, next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, val):
        new_node = DoublyListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def prepend(self, val):
        new_node = DoublyListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1

    def delete(self, val):
        current = self.head
        while current:
            if current.val == val:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next

                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev

                self.size -= 1
                return True
            current = current.next
        return False

# Example 3: Reverse Linked List
def reverse_linked_list(head):
    prev = None
    current = head

    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp

    return prev

# Example 4: Detect Cycle in Linked List (Floyd's Algorithm)
def has_cycle(head):
    if not head or not head.next:
        return False

    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True

    return False

# Example 5: Merge Two Sorted Linked Lists
def merge_two_sorted_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    # Attach remaining nodes
    current.next = l1 or l2

    return dummy.next
```

### Binary Tree

**Definition:** Hierarchical data structure where each node has at most two children (left and right).

**Time Complexity:** Search/Insert/Delete O(log n) average, O(n) worst | **Space Complexity:** O(n)

**Usage Sites:**

1. Binary Search Trees for efficient searching
2. Expression parsing and evaluation
3. Huffman coding for data compression
4. Decision trees in machine learning
5. File system hierarchies
6. Database indexing (B-trees)
7. Priority queues (binary heaps)
8. Syntax trees in compilers
9. Game trees for AI decision making
10. Organization charts and family trees

**Examples:**

```python
# Example 1: Basic Binary Tree Implementation
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
        else:
            self._insert_recursive(self.root, val)

    def _insert_recursive(self, node, val):
        if val < node.val:
            if node.left:
                self._insert_recursive(node.left, val)
            else:
                node.left = TreeNode(val)
        else:
            if node.right:
                self._insert_recursive(node.right, val)
            else:
                node.right = TreeNode(val)

    def inorder_traversal(self, node=None):
        if node is None:
            node = self.root

        result = []
        if node:
            result.extend(self.inorder_traversal(node.left))
            result.append(node.val)
            result.extend(self.inorder_traversal(node.right))
        return result

    def preorder_traversal(self, node=None):
        if node is None:
            node = self.root

        result = []
        if node:
            result.append(node.val)
            result.extend(self.preorder_traversal(node.left))
            result.extend(self.preorder_traversal(node.right))
        return result

    def postorder_traversal(self, node=None):
        if node is None:
            node = self.root

        result = []
        if node:
            result.extend(self.postorder_traversal(node.left))
            result.extend(self.postorder_traversal(node.right))
            result.append(node.val)
        return result

# Usage
bt = BinaryTree()
for val in [5, 3, 7, 2, 4, 6, 8]:
    bt.insert(val)
print("Inorder:", bt.inorder_traversal())    # Output: [2, 3, 4, 5, 6, 7, 8]
print("Preorder:", bt.preorder_traversal())  # Output: [5, 3, 2, 4, 7, 6, 8]
print("Postorder:", bt.postorder_traversal())# Output: [2, 4, 3, 6, 8, 7, 5]

# Example 2: Level-order Traversal (BFS)
from collections import deque

def level_order_traversal(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level_nodes = []

        for _ in range(level_size):
            node = queue.popleft()
            level_nodes.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level_nodes)

    return result

# Example 3: Maximum Depth of Binary Tree
def max_depth(root):
    if not root:
        return 0

    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)

    return max(left_depth, right_depth) + 1

# Example 4: Validate Binary Search Tree
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True

    if root.val <= min_val or root.val >= max_val:
        return False

    return (is_valid_bst(root.left, min_val, root.val) and
            is_valid_bst(root.right, root.val, max_val))

# Example 5: Lowest Common Ancestor
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root

    return left or right
```

## 🔄 Advanced Algorithms

### Depth-First Search (DFS)

**Definition:** Graph traversal algorithm that explores as far as possible along each branch before backtracking.

**Time Complexity:** O(V + E) | **Space Complexity:** O(V)

**Usage Sites:**

1. Topological sorting
2. Finding connected components
3. Solving puzzles with backtracking
4. Pathfinding in mazes
5. Cycle detection in graphs
6. Tree traversals
7. Web crawling
8. Dependency resolution
9. Finding strongly connected components
10. Game AI and decision trees

**Examples:**

```python
# Example 1: DFS on Graph (Recursive)
def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    result = [start]

    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))

    return result

# Graph as adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
print(dfs_recursive(graph, 'A'))  # Output: ['A', 'B', 'D', 'E', 'F', 'C']

# Example 2: DFS on Graph (Iterative)
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    result = []

    while stack:
        vertex = stack.pop()

        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)

            # Add neighbors to stack (reverse order for left-to-right traversal)
            for neighbor in reversed(graph.get(vertex, [])):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result

print(dfs_iterative(graph, 'A'))  # Output: ['A', 'B', 'D', 'E', 'F', 'C']

# Example 3: DFS for Binary Tree
def dfs_binary_tree(root):
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        # Add right first so left is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result

# Example 4: Find Path using DFS
def find_path_dfs(graph, start, end, path=None):
    if path is None:
        path = []

    path = path + [start]

    if start == end:
        return path

    for neighbor in graph.get(start, []):
        if neighbor not in path:  # Avoid cycles
            new_path = find_path_dfs(graph, neighbor, end, path)
            if new_path:
                return new_path

    return None

print(find_path_dfs(graph, 'A', 'F'))  # Output: ['A', 'B', 'E', 'F']

# Example 5: DFS for Cycle Detection
def has_cycle_dfs(graph):
    visited = set()
    rec_stack = set()

    def dfs_helper(node):
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs_helper(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for vertex in graph:
        if vertex not in visited:
            if dfs_helper(vertex):
                return True

    return False

# Cyclic graph
cyclic_graph = {
    'A': ['B'],
    'B': ['C'],
    'C': ['A']  # Creates a cycle
}
print(has_cycle_dfs(cyclic_graph))  # Output: True
```

### Breadth-First Search (BFS)

**Definition:** Graph traversal algorithm that explores all vertices at present depth before moving to vertices at next depth level.

**Time Complexity:** O(V + E) | **Space Complexity:** O(V)

**Usage Sites:**

1. Shortest path in unweighted graphs
2. Level-order tree traversal
3. Web crawling by depth
4. Social network analysis (degrees of separation)
5. GPS navigation systems
6. Peer-to-peer networks
7. Finding connected components
8. Bipartite graph testing
9. Serializing/deserializing binary trees
10. Minimum spanning tree algorithms

**Examples:**

```python
# Example 1: BFS on Graph
from collections import deque

def bfs_graph(graph, start):
    visited = set()
    queue = deque([start])
    result = []

    visited.add(start)

    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
print(bfs_graph(graph, 'A'))  # Output: ['A', 'B', 'C', 'D', 'E', 'F']

# Example 2: Shortest Path using BFS
def shortest_path_bfs(graph, start, end):
    if start == end:
        return [start]

    visited = set()
    queue = deque([(start, [start])])
    visited.add(start)

    while queue:
        vertex, path = queue.popleft()

        for neighbor in graph.get(vertex, []):
            if neighbor == end:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None

print(shortest_path_bfs(graph, 'A', 'F'))  # Output: ['A', 'C', 'F']

# Example 3: BFS Level by Level
def bfs_levels(graph, start):
    visited = set()
    queue = deque([start])
    levels = []
    visited.add(start)

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            vertex = queue.popleft()
            current_level.append(vertex)

            for neighbor in graph.get(vertex, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        levels.append(current_level)

    return levels

print(bfs_levels(graph, 'A'))  # Output: [['A'], ['B', 'C'], ['D', 'E', 'F']]

# Example 4: BFS on Binary Tree (Level Order)
def level_order_bfs(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level_values = []

        for _ in range(level_size):
            node = queue.popleft()
            level_values.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level_values)

    return result

# Example 5: Check if Graph is Bipartite using BFS
def is_bipartite_bfs(graph):
    color = {}

    for start in graph:
        if start in color:
            continue

        queue = deque([start])
        color[start] = 0

        while queue:
            node = queue.popleft()

            for neighbor in graph.get(node, []):
                if neighbor in color:
                    if color[neighbor] == color[node]:
                        return False
                else:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)

    return True

# Bipartite graph
bipartite_graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}
print(is_bipartite_bfs(bipartite_graph))  # Output: True
```

### Dynamic Programming

**Definition:** Optimization technique that solves complex problems by breaking them into simpler subproblems and storing results.

**Time Complexity:** Varies (typically O(n) to O(n²)) | **Space Complexity:** O(n) to O(n²)

**Usage Sites:**

1. Fibonacci sequence calculation
2. Longest Common Subsequence (LCS)
3. Knapsack problem variants
4. Edit distance (Levenshtein distance)
5. Coin change problems
6. Stock price optimization
7. Route optimization
8. Game theory and decision making
9. Resource allocation problems
10. Sequence alignment in bioinformatics

**Examples:**

```python
# Example 1: Fibonacci with Memoization
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

print(fibonacci_memo(50))  # Output: 12586269025

# Example 2: Fibonacci with Tabulation (Bottom-up)
def fibonacci_tab(n):
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

print(fibonacci_tab(50))  # Output: 12586269025

# Example 3: Coin Change Problem
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

coins = [1, 3, 4]
amount = 6
print(coin_change(coins, amount))  # Output: 2 (3 + 3)

# Example 4: Longest Common Subsequence
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

text1 = "abcde"
text2 = "ace"
print(longest_common_subsequence(text1, text2))  # Output: 3

# Example 5: 0/1 Knapsack Problem
def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            # Don't include current item
            dp[i][w] = dp[i-1][w]

            # Include current item if it fits
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                             dp[i-1][w - weights[i-1]] + values[i-1])

    return dp[n][capacity]

weights = [2, 1, 3]
values = [12, 10, 20]
capacity = 5
print(knapsack_01(weights, values, capacity))  # Output: 32
```

## 📊 Time & Space Complexity Summary

### Big O Notation Quick Reference

| Operation                 | Best       | Average    | Worst      | Space    |
| ------------------------- | ---------- | ---------- | ---------- | -------- |
| **Linear Search**         | O(1)       | O(n)       | O(n)       | O(1)     |
| **Binary Search**         | O(1)       | O(log n)   | O(log n)   | O(1)     |
| **Bubble Sort**           | O(n)       | O(n²)      | O(n²)      | O(1)     |
| **Quick Sort**            | O(n log n) | O(n log n) | O(n²)      | O(log n) |
| **Merge Sort**            | O(n log n) | O(n log n) | O(n log n) | O(n)     |
| **Stack Push/Pop**        | O(1)       | O(1)       | O(1)       | O(n)     |
| **Queue Enqueue/Dequeue** | O(1)       | O(1)       | O(1)       | O(n)     |
| **Linked List Insert**    | O(1)       | O(1)       | O(1)       | O(n)     |
| **Linked List Search**    | O(1)       | O(n)       | O(n)       | O(n)     |
| **Binary Tree Search**    | O(log n)   | O(log n)   | O(n)       | O(n)     |
| **DFS/BFS**               | O(V+E)     | O(V+E)     | O(V+E)     | O(V)     |

### Key Tips for Optimization

1. **Use appropriate data structure** - Choose based on your primary operations
2. **Consider trade-offs** - Time vs Space complexity
3. **Memoization** - Store results of expensive function calls
4. **Early termination** - Stop when answer is found
5. **In-place operations** - Reduce space complexity when possible
6. **Divide and conquer** - Break problems into smaller subproblems
7. **Hash tables** - For O(1) lookups when possible
8. **Two pointers** - Reduce nested loops in array problems
9. **Sliding window** - For subarray/substring problems
10. **Binary search** - On sorted data for O(log n) searches

---

## 🎯 Practice Problems by Category

### Arrays & Searching

- Two Sum, Three Sum
- Maximum Subarray (Kadane's Algorithm)
- Rotate Array, Search in Rotated Sorted Array
- Find Peak Element
- First Bad Version

### Sorting & Selection

- Kth Largest Element
- Sort Colors (Dutch Flag)
- Meeting Rooms
- Merge Intervals
- Top K Frequent Elements

### Linked Lists

- Reverse Linked List
- Merge Two Sorted Lists
- Remove Nth Node from End
- Cycle Detection
- Intersection of Two Linked Lists

### Trees & Graphs

- Maximum Depth of Binary Tree
- Lowest Common Ancestor
- Binary Tree Level Order Traversal
- Number of Islands
- Course Schedule (Topological Sort)

### Dynamic Programming

- House Robber
- Climbing Stairs
- Unique Paths
- Word Break
- Palindromic Substrings

---

_Remember: Practice consistently, understand the patterns, and focus on problem-solving approach rather than memorizing solutions!_
