# Stacks - Learning Guide 📚

## 📚 What Are We Learning Today?

Think of this lesson as learning about a special way to organize things where the LAST thing you put in is the FIRST thing you take out!

---

## 🥞 Part 1: Understanding Stacks

### The Kid-Friendly Explanation

Imagine a stack of pancakes:

- You **add pancakes on TOP** (can't put one in the middle!)
- You **eat from the TOP** (can't take one from the bottom!)
- The **LAST pancake** you added is the **FIRST one** you eat

Other real-life stacks:

- Stack of plates in a cafeteria
- Stack of books on your desk
- Browser back button (last page visited = first to go back to)
- Undo button in apps (last action = first to undo)

**That's exactly what a Stack does in programming!** It follows the **LIFO** principle: **Last In, First Out**.

### The Technical Explanation

A **Stack** is a linear data structure that follows LIFO (Last In, First Out) principle:

- **Push**: Add element to the top - O(1)
- **Pop**: Remove element from the top - O(1)
- **Peek/Top**: View top element without removing - O(1)
- **isEmpty**: Check if stack is empty - O(1)

```python
# Python Stack using list
stack = []

# Push operations
stack.append(1)  # [1]
stack.append(2)  # [1, 2]
stack.append(3)  # [1, 2, 3]

# Pop operations (LIFO!)
print(stack.pop())  # 3 (last in, first out)
print(stack.pop())  # 2
print(stack.pop())  # 1
```

### Visual Representation:

```
Push 1, 2, 3:          Pop operations:

    [3]  ← top            [ ]              [ ]              [ ]
    [2]                   [2]              [ ]              [ ]
    [1]                   [1]              [1]              [ ]
    ----                  ----             ----             ----
   Stack                Pop 3            Pop 2            Pop 1
```

### 🔬 Why Data Scientists & Data Engineers Love Stacks

1. **Expression Evaluation**: Parse and evaluate mathematical expressions
2. **Backtracking**: Undo operations, traverse tree/graph paths
3. **Syntax Parsing**: Validate data formats (JSON, XML, nested structures)
4. **Function Call Management**: Track execution context (call stack)
5. **Data Pipeline Processing**: Reverse data transformations
6. **State Management**: Track history of operations for rollback
7. **Depth-First Search**: Navigate hierarchical data structures

**Real Data Engineering Example:**

```python
# Validate nested JSON structure
def validate_brackets(data_string):
    """Check if brackets are properly nested in data format"""
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}

    for char in data_string:
        if char in pairs:  # Opening bracket
            stack.append(char)
        elif char in pairs.values():  # Closing bracket
            if not stack or pairs[stack.pop()] != char:
                return False

    return len(stack) == 0  # All opened brackets closed
```

---

## 🎨 Part 2: Stack Operations Deep Dive

### Core Operations:

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        """Add item to top - O(1)"""
        self.items.append(item)

    def pop(self):
        """Remove and return top item - O(1)"""
        if self.is_empty():
            return None
        return self.items.pop()

    def peek(self):
        """View top item without removing - O(1)"""
        if self.is_empty():
            return None
        return self.items[-1]

    def is_empty(self):
        """Check if stack is empty - O(1)"""
        return len(self.items) == 0

    def size(self):
        """Get number of items - O(1)"""
        return len(self.items)

# Usage
stack = Stack()
stack.push(10)
stack.push(20)
print(stack.peek())  # 20
print(stack.pop())   # 20
print(stack.size())  # 1
```

---

## 💻 Practice Problems

### Problem 1: Valid Parentheses (Classic Stack Problem)

**Problem**: Check if brackets are properly opened and closed.

**Example**:

- Input: `"()[]{}"`
- Output: `True` (all properly matched)
- Input: `"([)]"`
- Output: `False` (wrong order: `[` closed before `(`)

**Solution**:

```python
def is_valid(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in mapping:  # Closing bracket
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:  # Opening bracket
            stack.append(char)

    return len(stack) == 0

# Test
print(is_valid("()[]{}"))  # True
print(is_valid("([)]"))    # False
print(is_valid("{[]}"))    # True
```

**Step-by-Step Visualization**:

```
Input: "{[()]}"

Step 1: char='{'  → Push  → Stack: ['{']
Step 2: char='['  → Push  → Stack: ['{', '[']
Step 3: char='('  → Push  → Stack: ['{', '[', '(']
Step 4: char=')'  → Match → Stack: ['{', '[']     (pop '(')
Step 5: char=']'  → Match → Stack: ['{']          (pop '[')
Step 6: char='}'  → Match → Stack: []             (pop '{')

Result: Stack empty = Valid!
```

**Why This Matters for Data Science**:

- **Data Validation**: Check JSON/XML structure integrity
- **ETL Pipelines**: Validate nested data formats
- **Log Parsing**: Ensure proper event nesting
- **SQL Query Validation**: Check nested subqueries

---

### Problem 2: Daily Temperatures (Monotonic Stack)

**Problem**: Find how many days until a warmer temperature for each day.

**Example**:

- Input: `temperatures = [73, 74, 75, 71, 69, 72, 76, 73]`
- Output: `[1, 1, 4, 2, 1, 1, 0, 0]`
- Explanation: Day 0 (73°) → Next warmer is Day 1 (74°) → 1 day wait

**Solution**:

```python
def daily_temperatures(temperatures):
    n = len(temperatures)
    result = [0] * n
    stack = []  # Store indices

    for i, temp in enumerate(temperatures):
        # While current temp is warmer than stack top
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index

        stack.append(i)

    return result

# Test
temps = [73, 74, 75, 71, 69, 72, 76, 73]
print(daily_temperatures(temps))
# Output: [1, 1, 4, 2, 1, 1, 0, 0]
```

**How It Works**:

```
Temps: [73, 74, 75, 71, 69, 72, 76, 73]
Index:  0   1   2   3   4   5   6   7

i=0: temp=73 → Stack: [0]
i=1: temp=74 > 73 → Pop 0, result[0]=1-0=1 → Stack: [1]
i=2: temp=75 > 74 → Pop 1, result[1]=2-1=1 → Stack: [2]
i=3: temp=71 < 75 → Stack: [2, 3]
i=4: temp=69 < 71 → Stack: [2, 3, 4]
i=5: temp=72 > 69,71 → Pop 4,3, result[4]=1, result[3]=2 → Stack: [2, 5]
i=6: temp=76 > 72,75 → Pop 5,2, result[5]=1, result[2]=4 → Stack: [6]
i=7: temp=73 < 76 → Stack: [6, 7]

Final: result = [1, 1, 4, 2, 1, 1, 0, 0]
```

**Why This Matters for Data Science**:

- **Time Series Analysis**: Find next occurrence of threshold crossing
- **Stock Analysis**: Days until price increases
- **Sensor Data**: Detect next spike in metrics
- **Anomaly Detection**: Find next significant event

---

### Problem 3: Evaluate Reverse Polish Notation (Calculator Stack)

**Problem**: Evaluate mathematical expression in postfix notation.

**Example**:

- Input: `["2", "1", "+", "3", "*"]`
- Output: `9`
- Explanation: `((2 + 1) * 3) = 9`

**What is RPN?**:

- Normal: `2 + 1`
- RPN: `2 1 +` (operands first, operator last)

**Solution**:

```python
def eval_rpn(tokens):
    stack = []
    operators = {'+', '-', '*', '/'}

    for token in tokens:
        if token in operators:
            # Pop two operands (order matters for - and /)
            b = stack.pop()
            a = stack.pop()

            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            else:  # '/'
                result = int(a / b)  # Truncate toward zero

            stack.append(result)
        else:
            stack.append(int(token))

    return stack[-1]

# Test
print(eval_rpn(["2", "1", "+", "3", "*"]))  # 9
print(eval_rpn(["4", "13", "5", "/", "+"]))  # 6
```

**Step-by-Step**:

```
Input: ["2", "1", "+", "3", "*"]

Step 1: "2"   → Stack: [2]
Step 2: "1"   → Stack: [2, 1]
Step 3: "+"   → Pop 1,2 → 2+1=3 → Stack: [3]
Step 4: "3"   → Stack: [3, 3]
Step 5: "*"   → Pop 3,3 → 3*3=9 → Stack: [9]

Result: 9
```

**Why This Matters for Data Engineering**:

- **Expression Evaluation**: Calculate derived metrics
- **Data Transformations**: Apply mathematical operations in pipelines
- **Query Engines**: Evaluate filter conditions
- **Business Logic**: Compute complex formulas from configuration

---

## 🎯 Key Takeaways

### When to Use Stacks:

- ✅ Need to **reverse** something
- ✅ Need to **match pairs** (brackets, tags)
- ✅ Need to **track history** (undo/redo)
- ✅ Need **most recent** items first
- ✅ Parsing or evaluating **expressions**
- ✅ **Backtracking** algorithms
- ✅ **Depth-First Search** traversal
- ✅ Managing **nested structures**

### Stack vs Other Data Structures:

| Feature        | Stack                    | Queue                     | Array           |
| -------------- | ------------------------ | ------------------------- | --------------- |
| Access Pattern | LIFO (Last In First Out) | FIFO (First In First Out) | Random Access   |
| Add            | Top only (O(1))          | End only (O(1))           | Anywhere (O(n)) |
| Remove         | Top only (O(1))          | Front only (O(1))         | Anywhere (O(n)) |
| Use Case       | Undo, Backtrack, Parse   | Process in order, BFS     | General purpose |

### Common Stack Patterns:

1. **Monotonic Stack**: Keep elements in increasing/decreasing order
   - Used for: Next greater/smaller element problems
2. **Two Stacks**: Use two stacks for complex operations
   - Used for: Implementing queue, browser history
3. **Stack + HashMap**: Combine for O(1) operations
   - Used for: Min stack, LRU cache

---

## 🚀 Next Steps for Practice

### LeetCode Easy:

- Valid Parentheses ✅ (covered above)
- Min Stack
- Baseball Game
- Backspace String Compare

### LeetCode Medium:

- Daily Temperatures ✅ (covered above)
- Evaluate Reverse Polish Notation ✅ (covered above)
- Decode String
- Asteroid Collision
- Next Greater Element

### Data Science Projects:

1. **Expression Parser**: Build calculator for custom metrics
2. **JSON Validator**: Validate nested data structures
3. **Undo System**: Implement rollback for data transformations
4. **Path Finder**: DFS traversal using stack for data hierarchies

---

## 💡 Pro Tips

### Implementation Choices:

**Python:**

```python
# Use list (most common)
stack = []
stack.append(x)  # Push
stack.pop()      # Pop

# Use collections.deque (better for large stacks)
from collections import deque
stack = deque()
stack.append(x)  # Push
stack.pop()      # Pop
```

### Common Mistakes to Avoid:

- ❌ Not checking if stack is empty before pop
- ❌ Forgetting that pop() removes AND returns
- ❌ Using index [0] instead of [-1] for top element
- ❌ Not considering edge case: empty input

### Interview Tips:

1. **Mention LIFO**: Show you understand the principle
2. **Check empty**: Always validate before pop/peek
3. **Think backwards**: Stacks often solve problems by processing in reverse
4. **Look for pairs**: Matching/nesting problems → Think Stack!

---

## 🔗 Connection to Previous Topics

### Stack + HashMap (Problem: Min Stack)

Create a stack that returns minimum in O(1):

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []  # Track minimums

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def get_min(self):
        return self.min_stack[-1]  # O(1)!
```

### Stack vs Two Pointers:

- **Stack**: For nested/hierarchical problems
- **Two Pointers**: For linear/array problems

---

## 🎓 Advanced Concept: Call Stack

**Every program uses a stack!** When you call functions:

```python
def third():
    print("Third")

def second():
    third()
    print("Second")

def first():
    second()
    print("First")

first()
```

**Call Stack Visualization:**

```
Step 1: first() called
[first]

Step 2: second() called inside first()
[first, second]

Step 3: third() called inside second()
[first, second, third]

Step 4: third() completes, pops off
[first, second]
Output: "Third"

Step 5: second() completes, pops off
[first]
Output: "Second"

Step 6: first() completes, pops off
[]
Output: "First"
```

This is why stack overflow happens - too many nested calls!

---

Happy Learning! 🎉 Master stacks and you'll see them everywhere in computing!
