# Sliding Window - Learning Guide 🪟

## 📚 What Are We Learning Today?

Think of this lesson as learning to look through a moving window that slides across your data, always looking at a specific section at a time!

---

## 🪟 Part 1: Understanding Sliding Window

### The Kid-Friendly Explanation

Imagine you're on a train looking out the window:

- The **window shows you a view** (a section of the scenery)
- As the train moves, the **window slides** and shows new scenery
- You're always looking at a **specific portion** of the landscape
- Old scenery disappears from view, new scenery appears

**Other real-life examples:**

- Reading a book with a reading guide that covers text
- Watching a movie through a paper frame that moves across the screen
- Looking at houses through a car window as you drive
- Scanning a barcode reader across a code

**That's the Sliding Window technique!** You maintain a "window" (a subarray/substring) that:

1. Starts at one end
2. **Expands** when you need more elements
3. **Contracts** when conditions are met
4. **Slides** across the data

### The Technical Explanation

**Sliding Window** is an optimization technique that:

- Maintains a subset of data (window) within an array/string
- Efficiently processes subarrays/substrings by **reusing computations**
- Reduces time complexity from **O(n²) or O(n³) to O(n)**
- Uses **two pointers** to define window boundaries (left & right)

**Two Main Types:**

**1. Fixed Size Window:**

```
Window size = 3
[1, 2, 3, 4, 5, 6]
 -------           Step 1: [1,2,3]
    -------        Step 2: [2,3,4]
       -------     Step 3: [3,4,5]
          -------  Step 4: [4,5,6]
```

**2. Variable Size Window:**

```
Find longest window with sum ≤ 10
[1, 2, 3, 4, 5]
 -              Window: [1], sum=1 ✓
 ----           Window: [1,2], sum=3 ✓
 --------       Window: [1,2,3], sum=6 ✓
 ------------   Window: [1,2,3,4], sum=10 ✓
    --------    Window: [2,3,4], sum=9 ✓ (shrink when needed)
```

### Visual Representation:

```
Without Sliding Window (Recalculate every time):
Array: [1, 2, 3, 4, 5]
Window size = 3

Sum of [1,2,3]: 1+2+3 = 6    ← Calculate from scratch
Sum of [2,3,4]: 2+3+4 = 9    ← Calculate from scratch
Sum of [3,4,5]: 3+4+5 = 12   ← Calculate from scratch
Total operations: 9

With Sliding Window (Reuse calculation):
Sum of [1,2,3]: 1+2+3 = 6
Sum of [2,3,4]: 6 - 1 + 4 = 9   ← Remove left, add right!
Sum of [3,4,5]: 9 - 2 + 5 = 12  ← Remove left, add right!
Total operations: 5
```

### 🔬 Why Data Scientists & Data Engineers Love Sliding Window

1. **Time Series Analysis**: Moving averages, rolling statistics
2. **Log Processing**: Analyze recent events in time windows
3. **Real-time Monitoring**: Track metrics over sliding time intervals
4. **Feature Engineering**: Create window-based features
5. **Anomaly Detection**: Detect patterns in recent data
6. **Stream Processing**: Process continuous data streams
7. **Rate Limiting**: Track requests in time windows
8. **Signal Processing**: Apply filters to sequential data

**Real Data Science Example:**

```python
# Calculate 7-day moving average for stock prices
def moving_average(prices, k=7):
    if len(prices) < k:
        return []

    # Initialize first window
    window_sum = sum(prices[:k])
    averages = [window_sum / k]

    # Slide the window
    for i in range(k, len(prices)):
        window_sum = window_sum - prices[i - k] + prices[i]
        averages.append(window_sum / k)

    return averages
```

---

## 🎨 Part 2: Sliding Window Patterns

### Pattern 1: Fixed Size Window

Window size is constant, slide one element at a time.

- **Use for**: Moving averages, K-size problems

### Pattern 2: Variable Size Window (Expand/Contract)

Window grows and shrinks based on conditions.

- **Use for**: Longest/shortest substring problems

### Pattern 3: Two Pointers with HashMap

Track elements in window using hash map.

- **Use for**: Substring with distinct characters

### The Golden Template:

```python
# Fixed Size Window
def fixed_window(arr, k):
    window_start = 0
    result = 0

    for window_end in range(len(arr)):
        # Add current element to window
        # ... process arr[window_end]

        # Slide window when size reaches k
        if window_end >= k - 1:
            # Calculate result
            # ... use window data

            # Remove leftmost element
            # ... remove arr[window_start]
            window_start += 1

    return result

# Variable Size Window
def variable_window(arr, target):
    window_start = 0
    result = 0
    current_sum = 0

    for window_end in range(len(arr)):
        # Expand window
        current_sum += arr[window_end]

        # Contract window while condition violated
        while current_sum > target:
            current_sum -= arr[window_start]
            window_start += 1

        # Update result with current window
        result = max(result, window_end - window_start + 1)

    return result
```

---

## 💻 Practice Problems

### Problem 1: Maximum Sum Subarray of Size K (Fixed Window)

**Problem**: Find maximum sum of any contiguous subarray of size K.

**Example**:

- Input: `arr = [2, 1, 5, 1, 3, 2]`, `k = 3`
- Output: `9` (subarray [5, 1, 3])

**Brute Force Approach (O(n\*k)):**

```python
def max_sum_brute(arr, k):
    max_sum = float('-inf')

    for i in range(len(arr) - k + 1):
        current_sum = 0
        for j in range(i, i + k):
            current_sum += arr[j]  # Recalculate each time!
        max_sum = max(max_sum, current_sum)

    return max_sum
```

**Sliding Window Approach (O(n)):**

```python
def max_sum_subarray(arr, k):
    if len(arr) < k:
        return None

    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide the window
    for i in range(k, len(arr)):
        # Remove leftmost element, add new element
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Test
print(max_sum_subarray([2, 1, 5, 1, 3, 2], 3))  # 9
```

**Step-by-Step Visualization:**

```
Array: [2, 1, 5, 1, 3, 2], k=3

Step 1: Initial window [2, 1, 5]
Sum = 8, max_sum = 8

Step 2: Slide to [1, 5, 1]
Remove 2, Add 1: 8 - 2 + 1 = 7
max_sum = 8

Step 3: Slide to [5, 1, 3]
Remove 1, Add 3: 7 - 1 + 3 = 9
max_sum = 9 ← Answer!

Step 4: Slide to [1, 3, 2]
Remove 5, Add 2: 9 - 5 + 2 = 6
max_sum = 9

Result: 9
```

**Why This Matters for Data Science**:

- **Moving Statistics**: Rolling mean, median, standard deviation
- **Time Series**: Smooth noisy data with moving averages
- **Sensor Data**: Aggregate readings over time windows
- **Financial Analysis**: Calculate moving metrics for stocks

---

### Problem 2: Longest Substring Without Repeating Characters (Variable Window)

**Problem**: Find length of longest substring with all unique characters.

**Example**:

- Input: `"abcabcbb"`
- Output: `3` (substring "abc")
- Input: `"bbbbb"`
- Output: `1` (substring "b")

**Solution:**

```python
def length_of_longest_substring(s):
    char_set = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        # If duplicate found, shrink window from left
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        # Add current character
        char_set.add(s[right])

        # Update max length
        max_length = max(max_length, right - left + 1)

    return max_length

# Test
print(length_of_longest_substring("abcabcbb"))  # 3
print(length_of_longest_substring("bbbbb"))     # 1
print(length_of_longest_substring("pwwkew"))    # 3 ("wke")
```

**Step-by-Step for "abcabcbb":**

```
right=0: 'a'
Window: [a], set={'a'}, length=1

right=1: 'b'
Window: [a,b], set={'a','b'}, length=2

right=2: 'c'
Window: [a,b,c], set={'a','b','c'}, length=3 ← Max so far

right=3: 'a' (duplicate!)
Shrink: Remove 'a' → set={'b','c'}
Window: [b,c,a], set={'b','c','a'}, length=3

right=4: 'b' (duplicate!)
Shrink: Remove 'b' → set={'c','a'}
Window: [c,a,b], set={'c','a','b'}, length=3

right=5: 'c' (duplicate!)
Shrink: Remove 'c', Remove 'a' → set={'b'}
Window: [b,c], set={'b','c'}, length=2

right=6: 'b' (duplicate!)
Shrink: Remove 'b' → set={'c'}
Window: [c,b], set={'c','b'}, length=2

right=7: 'b' (duplicate!)
Shrink: Remove 'c' → set={'b'}
Window: [b,b], set={'b'}, length=1

Result: max_length = 3
```

**Why This Matters for Data Engineering**:

- **Deduplication**: Find unique sequences in streams
- **Log Analysis**: Detect unique event patterns
- **Data Quality**: Find longest valid sequences
- **Text Processing**: Extract unique token sequences

---

### Problem 3: Minimum Window Substring (Advanced Variable Window)

**Problem**: Find smallest substring that contains all characters from another string.

**Example**:

- Input: `s = "ADOBECODEBANC"`, `t = "ABC"`
- Output: `"BANC"` (smallest substring containing A, B, C)

**Solution:**

```python
def min_window(s, t):
    if not s or not t:
        return ""

    # Count characters needed
    dict_t = {}
    for char in t:
        dict_t[char] = dict_t.get(char, 0) + 1

    required = len(dict_t)  # Unique characters needed
    left = 0
    formed = 0  # Unique characters matched with desired frequency
    window_counts = {}

    # Result: (window length, left, right)
    ans = float("inf"), None, None

    for right in range(len(s)):
        # Add character from right
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        # Check if frequency matches
        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1

        # Contract window until it's no longer valid
        while left <= right and formed == required:
            char = s[left]

            # Update result if this window is smaller
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)

            # Remove from left
            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1

            left += 1

    return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]

# Test
print(min_window("ADOBECODEBANC", "ABC"))  # "BANC"
print(min_window("a", "a"))                 # "a"
print(min_window("a", "aa"))                # ""
```

**Visual for "ADOBECODEBANC", target "ABC":**

```
Need: A=1, B=1, C=1

Expand until valid:
ADOBEC → Has A, B, C ✓ (length=6)

Contract while valid:
DOBEC → Missing A ✗
Back to: ADOBEC

Continue expanding:
ADOBECOD → Valid, length=8
ADOBECODE → Valid, length=9
ADOBECODEB → Valid, length=10
ADOBECODEBA → Valid, length=11
ADOBECODEBAN → Valid, length=12
ADOBECODEBANC → Valid, length=13

Now contract from left:
DOBECODEBANC → Missing A
Back to: ADOBECODEBANC

Try: ODEBANC → Missing B
Try: DEBANC → Missing O (not needed)
Try: EBANC → Has all ABC ✓ (length=5)
Try: BANC → Has all ABC ✓ (length=4) ← Smallest!
Try: ANC → Missing B ✗

Result: "BANC"
```

**Why This Matters for Data Science**:

- **Pattern Matching**: Find minimal patterns in sequences
- **Feature Extraction**: Extract compact representations
- **Text Mining**: Find minimal documents containing keywords
- **Anomaly Detection**: Find shortest time window with all anomaly types

---

## 🎯 Key Takeaways

### When to Use Sliding Window:

- ✅ Problem involves **contiguous** subarrays/substrings
- ✅ Need to find **maximum/minimum** of something
- ✅ Need to find **longest/shortest** substring/subarray
- ✅ Keywords: "contiguous", "substring", "subarray", "window"
- ✅ Can **reuse previous computation**
- ✅ Sequential data (arrays, strings, lists)

### Sliding Window vs Other Techniques:

| Technique      | Best For              | Time     |
| -------------- | --------------------- | -------- |
| Sliding Window | Contiguous subarrays  | O(n)     |
| Two Pointers   | Pairs in sorted array | O(n)     |
| Binary Search  | Sorted search         | O(log n) |
| HashMap        | Non-contiguous lookup | O(n)     |

### Common Problem Keywords:

- "Maximum sum subarray of size K"
- "Longest substring with..."
- "Smallest/Shortest window containing..."
- "Find all subarrays that..."
- "Minimum window..."
- "Contiguous elements..."

---

## 🚀 Next Steps for Practice

### LeetCode Easy:

- Maximum Average Subarray I
- Contains Duplicate II
- Maximum Sum Subarray of Size K ✅ (covered)

### LeetCode Medium:

- Longest Substring Without Repeating Characters ✅ (covered)
- Minimum Window Substring ✅ (covered)
- Longest Repeating Character Replacement
- Max Consecutive Ones III
- Fruit Into Baskets
- Permutation in String

### Data Science Projects:

1. **Time Series Smoothing**: Implement moving averages for sensor data
2. **Log Analyzer**: Find patterns in sliding time windows
3. **Rate Limiter**: Track API requests in time windows
4. **Anomaly Detector**: Detect spikes in rolling windows
5. **Real-time Dashboard**: Calculate rolling metrics

---

## 💡 Pro Tips

### Fixed vs Variable Window - Quick Check:

**Fixed Window signals:**

- "Size K" mentioned
- "Exactly N elements"
- "K consecutive elements"

**Variable Window signals:**

- "Longest/Shortest"
- "At most/At least"
- "Containing all..."

### Template Memorization:

**Fixed Size:**

```python
window_sum = sum(arr[:k])  # Initialize
max_val = window_sum

for i in range(k, len(arr)):
    window_sum += arr[i] - arr[i-k]  # Slide!
    max_val = max(max_val, window_sum)
```

**Variable Size:**

```python
left = 0
for right in range(len(arr)):
    # Add arr[right] to window

    while condition_violated:
        # Remove arr[left] from window
        left += 1

    # Update result
```

### Common Mistakes:

- ❌ Forgetting to remove leftmost element when sliding
- ❌ Wrong window size calculation: use `right - left + 1`
- ❌ Not initializing the first window properly
- ❌ Using sliding window on non-contiguous problems
- ❌ Off-by-one errors in window boundaries

### Interview Tips:

1. **Ask about constraints**: Array size? Element range?
2. **Identify window type**: Fixed or variable?
3. **State approach**: "I'll use a sliding window to reduce from O(n²) to O(n)"
4. **Trace through example**: Show window movement
5. **Consider edge cases**: Empty array, k > n, all same elements

---

## 🔗 Connection to Previous Topics

### Sliding Window + Two Pointers:

Sliding window **IS** a two-pointer technique!

- `left` pointer: Start of window
- `right` pointer: End of window

### Sliding Window + HashMap:

Often combined for character/element tracking:

```python
# Track character frequencies in window
window_map = {}
for char in window:
    window_map[char] = window_map.get(char, 0) + 1
```

### Sliding Window + Arrays:

- Works on **arrays** and **strings**
- Needs **contiguous** elements
- Cannot work on **linked lists** (no random access)

---

## 🎓 Advanced Concepts

### Sliding Window Maximum (Using Deque):

**Problem**: Find maximum in each window of size K.

```python
from collections import deque

def max_sliding_window(nums, k):
    result = []
    dq = deque()  # Store indices

    for i in range(len(nums)):
        # Remove elements outside window
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements (they're useless)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result if window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# Test
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))
# Output: [3,3,5,5,6,7]
```

### Sliding Window for Time-based Problems:

```python
# Track events in last N seconds
from collections import deque
import time

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def allow_request(self):
        now = time.time()

        # Remove old requests outside window
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()

        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True

        return False
```

---

## 📊 Performance Comparison

**Example: Find max sum of size K subarray in array of size N=1,000,000**

| Approach       | Time Complexity   | Operations | Time Estimate |
| -------------- | ----------------- | ---------- | ------------- |
| Brute Force    | O(n\*k), k=1000   | 1 billion  | ~10 seconds   |
| Sliding Window | O(n)              | 1 million  | ~0.01 seconds |
| **Speedup**    | **1000x faster!** |            |               |

**The sliding window is not just faster—it's fundamentally more efficient!**

---

## 🌟 The Sliding Window Mindset

**Key Question**: "Am I recalculating something I already know?"

**If YES → Think Sliding Window!**

**Mental Model:**

1. What's in my current window?
2. Can I update by removing old + adding new?
3. When should I expand/contract?

**Real-world analogy:**

- Brute force = Counting all people in each train car from scratch
- Sliding window = Watching who exits and enters as train moves

---

Happy Learning! 🪟 Master sliding window and you'll optimize away those nested loops!
