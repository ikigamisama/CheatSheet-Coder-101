# Binary Search - Learning Guide 🔍

## 📚 What Are We Learning Today?

Think of this lesson as learning the SMARTEST way to find things - by cutting your search space in HALF every time!

---

## 🎯 Part 1: Understanding Binary Search

### The Kid-Friendly Explanation

**The Guessing Game:**
You think of a number between 1 and 100. Your friend guesses.

**Bad Strategy (Linear Search):**

- Friend: "Is it 1?" No.
- Friend: "Is it 2?" No.
- Friend: "Is it 3?" No.
- ...could take 100 guesses! 😫

**Smart Strategy (Binary Search):**

- Friend: "Is it 50?" Too high!
- Friend: "Is it 25?" Too low!
- Friend: "Is it 37?" Too high!
- Friend: "Is it 31?" Correct! ✅

**Only 4 guesses instead of potentially 100!**

**Other real-life examples:**

- Looking up a word in a dictionary (you don't start from 'A'!)
- Finding a page in a book
- Searching through a phone book
- Finding a song in alphabetical playlist

**The trick?** Always guess the **MIDDLE**, then eliminate **HALF** of the possibilities!

### The Technical Explanation

**Binary Search** is an efficient algorithm for finding a target value in a **sorted** array:

- Repeatedly divide the search space in **half**
- Compare target with middle element
- Eliminate half that can't contain the target
- Time Complexity: **O(log n)** - SUPER FAST!
- Space Complexity: **O(1)** iterative, O(log n) recursive

**Key Requirements:**

1. ✅ Array must be **SORTED**
2. ✅ Must have **random access** (arrays work, linked lists don't)

```python
# Classic Binary Search Template
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow

        if arr[mid] == target:
            return mid  # Found it!
        elif arr[mid] < target:
            left = mid + 1  # Search right half
        else:
            right = mid - 1  # Search left half

    return -1  # Not found
```

### Visual Representation:

```
Search for 37 in: [10, 20, 30, 40, 50, 60, 70, 80, 90]

Step 1: Check middle (index 4)
[10, 20, 30, 40, 50, 60, 70, 80, 90]
                 ↑
              mid=50
37 < 50, search LEFT half

Step 2: Check middle of left half (index 1)
[10, 20, 30, 40]
     ↑
  mid=20
37 > 20, search RIGHT half

Step 3: Check middle of remaining (index 2)
[30, 40]
 ↑
mid=30
37 > 30, search RIGHT half

Step 4: Check middle (index 3)
[40]
 ↑
mid=40
37 < 40, search LEFT half

Step 5: Left > Right, NOT FOUND
```

### 🔬 Why Data Scientists & Data Engineers Love Binary Search

1. **Speed**: O(log n) - search 1 billion items in ~30 steps!
2. **Database Indexes**: Most DB indexes use binary search variants
3. **Sorted Data**: Perfect for time series, sorted logs
4. **Finding Thresholds**: Find first/last occurrence of condition
5. **Optimization Problems**: Find minimum/maximum with constraints
6. **Data Versioning**: Find when data changed in sorted commits
7. **A/B Testing**: Find optimal parameter values

**Real Data Science Example:**

```python
# Find when metric crossed threshold in time series
timestamps = [1, 5, 10, 15, 20, 25, 30]
metric_values = [10, 15, 25, 45, 70, 85, 95]
threshold = 50

# Binary search to find first time metric > 50
# Answer: index 3 (timestamp 15)
```

---

## 🎨 Part 2: Binary Search Patterns

### Pattern 1: Exact Match (Classic)

Find exact target value.

### Pattern 2: Find First/Last Occurrence

When duplicates exist, find first or last position.

### Pattern 3: Find Boundary (Closest Value)

Find insertion position or closest value.

### Pattern 4: Search in Rotated Array

Handle sorted arrays that are rotated.

### Pattern 5: Search in 2D Matrix

Apply binary search in 2D space.

### The Golden Rule:

**If your data is sorted OR you can define a monotonic function, think Binary Search!**

---

## 💻 Practice Problems

### Problem 1: Classic Binary Search

**Problem**: Find target value in sorted array. Return index or -1.

**Example**:

- Input: `nums = [-1, 0, 3, 5, 9, 12]`, `target = 9`
- Output: `4` (index where 9 is found)

**Solution**:

```python
def search(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1  # Target in right half
        else:
            right = mid - 1  # Target in left half

    return -1

# Test
print(search([-1, 0, 3, 5, 9, 12], 9))   # 4
print(search([-1, 0, 3, 5, 9, 12], 2))   # -1
```

**Step-by-Step for target = 9**:

```
Array: [-1, 0, 3, 5, 9, 12]
Index:   0  1  2  3  4  5

Iteration 1:
left=0, right=5, mid=2
nums[2]=3, 3 < 9
→ Search right: left=3

Iteration 2:
left=3, right=5, mid=4
nums[4]=9, FOUND!
→ Return 4
```

**Why This Matters for Data Science**:

- **Fast Lookup**: Search sorted features, timestamps
- **Database Queries**: Underlying mechanism for indexed searches
- **Sensor Data**: Find readings at specific time

---

### Problem 2: First Bad Version (Find Boundary)

**Problem**: You're a product manager with versions [1, 2, 3, ..., n]. All versions after a bad version are also bad. Find the first bad version.

**Example**:

- Input: `n = 5`, bad = 4
- Versions: `[good, good, good, bad, bad]`
- Output: `4` (first bad version)

**Solution**:

```python
# Assume we have this API:
def isBadVersion(version):
    # Returns True if version is bad
    pass

def first_bad_version(n):
    left, right = 1, n

    while left < right:  # Note: left < right (not <=)
        mid = left + (right - left) // 2

        if isBadVersion(mid):
            # Mid is bad, but might not be first
            # Search left (include mid)
            right = mid
        else:
            # Mid is good, search right (exclude mid)
            left = mid + 1

    return left  # left == right when loop ends

# Test (assuming bad = 4)
print(first_bad_version(5))  # 4
```

**Why `left < right` instead of `left <= right`?**
We're finding a boundary, not exact match:

- When `left == right`, we've found the answer
- Don't need the `== target` check

**Step-by-Step**:

```
Versions: [1:good, 2:good, 3:good, 4:bad, 5:bad]

Iteration 1:
left=1, right=5, mid=3
isBad(3)=False → left=4

Iteration 2:
left=4, right=5, mid=4
isBad(4)=True → right=4

Loop ends: left==right==4
Return 4 ✅
```

**Why This Matters for Data Engineering**:

- **Debugging**: Find when data corruption started
- **Version Control**: Find commit that introduced bug
- **Log Analysis**: Find when error first appeared
- **Data Quality**: Find when metric went bad

---

### Problem 3: Search in Rotated Sorted Array (Advanced)

**Problem**: Array was sorted then rotated. Find target.

**Example**:

- Input: `nums = [4, 5, 6, 7, 0, 1, 2]`, `target = 0`
- Output: `4`
- Original: `[0, 1, 2, 4, 5, 6, 7]` rotated at pivot 4

**Solution**:

```python
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # Determine which half is sorted
        if nums[left] <= nums[mid]:  # Left half is sorted
            # Check if target is in sorted left half
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            # Check if target is in sorted right half
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1

# Test
print(search_rotated([4, 5, 6, 7, 0, 1, 2], 0))  # 4
print(search_rotated([4, 5, 6, 7, 0, 1, 2], 3))  # -1
```

**Key Insight**: At least ONE half is always sorted!

**Visual**:

```
[4, 5, 6, 7, 0, 1, 2]
 ↑         ↑       ↑
left      mid    right

Compare nums[left] vs nums[mid]:
4 <= 6? YES → Left half [4,5,6,7] is sorted!

If target in sorted half → search there
Else → search other half
```

**Step-by-Step for target = 0**:

```
[4, 5, 6, 7, 0, 1, 2]

Step 1: left=0, right=6, mid=3
nums[3]=7, not target
nums[0]=4 <= nums[3]=7 → left sorted
target 0 not in [4,7] → search right
left=4

Step 2: left=4, right=6, mid=5
nums[5]=1, not target
nums[4]=0 <= nums[5]=1 → left sorted
target 0 in [0,1) → search left
right=4

Step 3: left=4, right=4, mid=4
nums[4]=0 → FOUND!
Return 4
```

**Why This Matters for Data Science**:

- **Circular Buffers**: Search in ring buffers
- **Time Series with Wraparound**: Handle cyclical data
- **Log Files**: Rotated logs with timestamps
- **Real-world Data**: Handle sorted data with anomalies

---

## 🎯 Key Takeaways

### When to Use Binary Search:

- ✅ Array is **SORTED** (or can be treated as such)
- ✅ Need to search in **O(log n)** time
- ✅ Finding **boundary** or **threshold**
- ✅ **Monotonic function** exists (can determine left/right)
- ✅ Have **random access** to elements
- ✅ Search space can be **reduced by half**

### Binary Search Templates:

**Template 1: Exact Match**

```python
while left <= right:
    mid = left + (right - left) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
```

**Template 2: Find Boundary (First/Last)**

```python
while left < right:  # Note: no =
    mid = left + (right - left) // 2
    if condition(mid):
        right = mid      # Might be answer, keep it
    else:
        left = mid + 1   # Definitely not answer
return left  # or right (they're equal)
```

**Template 3: Find Closest**

```python
while left < right:
    mid = left + (right - left) // 2
    if check_condition(mid):
        right = mid
    else:
        left = mid + 1
return left
```

### Common Pitfalls:

- ❌ Using `mid = (left + right) / 2` → can overflow!
  - ✅ Use `mid = left + (right - left) // 2`
- ❌ Wrong while condition (`<=` vs `<`)
- ❌ Off-by-one errors (`mid + 1` vs `mid`)
- ❌ Not handling duplicates properly
- ❌ Forgetting array must be sorted

---

## 🚀 Next Steps for Practice

### LeetCode Easy:

- Binary Search ✅ (covered)
- First Bad Version ✅ (covered)
- Search Insert Position
- Sqrt(x)
- Valid Perfect Square

### LeetCode Medium:

- Search in Rotated Sorted Array ✅ (covered)
- Find First and Last Position of Element
- Find Peak Element
- Search a 2D Matrix
- Koko Eating Bananas (binary search on answer!)

### Data Science Projects:

1. **Time Series Search**: Find events in sorted timestamps
2. **Threshold Finder**: Find optimal cutoff values
3. **Database Query Optimizer**: Implement indexed search
4. **Anomaly Detection**: Find when metric crossed threshold

---

## 💡 Pro Tips

### The Power of O(log n):

| Array Size    | Linear O(n)     | Binary O(log n) |
| ------------- | --------------- | --------------- |
| 100           | 100 steps       | 7 steps         |
| 1,000         | 1,000 steps     | 10 steps        |
| 1,000,000     | 1,000,000 steps | 20 steps        |
| 1,000,000,000 | 1 billion steps | 30 steps!       |

**Binary search is INSANELY fast!** 🚀

### Binary Search on Answer (Advanced Pattern):

Sometimes you binary search on the **answer** itself!

**Problem**: Koko eating bananas

- She can eat K bananas per hour
- Find minimum K to finish all piles in H hours

**Insight**:

- If K works, all K' > K also work (monotonic!)
- Binary search on K values!

```python
def min_eating_speed(piles, h):
    def can_finish(k):
        hours = sum((pile + k - 1) // k for pile in piles)
        return hours <= h

    left, right = 1, max(piles)

    while left < right:
        mid = left + (right - left) // 2
        if can_finish(mid):
            right = mid  # Try slower speed
        else:
            left = mid + 1  # Need faster speed

    return left
```

### Interview Tips:

1. **Always ask**: "Is the array sorted?"
2. **Draw it**: Visualize the search space shrinking
3. **State complexity**: "This will be O(log n)"
4. **Handle edge cases**: Empty array, single element, target not found
5. **Test with examples**: Walk through 2-3 iterations

### Debugging Binary Search:

**Print statements are your friend:**

```python
while left <= right:
    mid = left + (right - left) // 2
    print(f"left={left}, right={right}, mid={mid}, val={arr[mid]}")
    # ... rest of code
```

---

## 🔗 Connection to Previous Topics

### Binary Search + Two Pointers:

Both reduce search space efficiently:

- **Two Pointers**: Move both ends toward center - O(n)
- **Binary Search**: Jump to middle, eliminate half - O(log n)

### Binary Search + Arrays vs Linked Lists:

- **Arrays**: Binary search works! (random access)
- **Linked Lists**: Can't binary search! (no random access)

### Binary Search in Data Structures:

- **Binary Search Trees**: Each node implicitly binary searches
- **Heaps**: Use array indexing (related to binary structure)
- **Hash Tables**: Don't use binary search (not sorted)

---

## 🎓 Advanced Concepts

### Binary Search Variants:

**1. Find First Occurrence:**

```python
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Keep searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

**2. Find Last Occurrence:**

```python
def find_last(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Keep searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

**3. Search in 2D Matrix:**

```python
def search_matrix(matrix, target):
    if not matrix:
        return False

    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1

    while left <= right:
        mid = left + (right - left) // 2
        # Convert 1D index to 2D coordinates
        mid_val = matrix[mid // cols][mid % cols]

        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1

    return False
```

---

## 📊 Complexity Comparison

| Algorithm     | Best     | Average     | Worst       | Space    |
| ------------- | -------- | ----------- | ----------- | -------- |
| Linear Search | O(1)     | O(n)        | O(n)        | O(1)     |
| Binary Search | O(1)     | O(log n)    | O(log n)    | O(1)     |
| **Speedup**   | **Same** | **n/log n** | **n/log n** | **Same** |

For n=1,000,000: Binary search is ~50,000x faster! 🤯

---

## 🌟 The Binary Search Mindset

**Key Question**: "Can I eliminate half the possibilities?"

**If YES → Think Binary Search!**

Examples where binary search thinking helps:

- Finding bugs: "Does the bug happen in first or second half of code?"
- Optimization: "Is optimal value in lower or upper range?"
- Debugging: "Did the problem start before or after this date?"

**Binary search is not just an algorithm, it's a problem-solving strategy!**

---

Happy Learning! 🔍 Master binary search and you'll solve problems EXPONENTIALLY faster!
