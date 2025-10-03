# Hashmaps and Sets - Learning Guide 🗺️

## 📚 What Are We Learning Today?

Think of this lesson as learning about two super-fast tools that help us find and organize things!

## 🎒 Part 1: Understanding Hashmaps (Dictionaries)

### The Kid-Friendly Explanation

Imagine you have a magic backpack where:

- You put a **label** on each pocket (like "snacks", "toys", "books")
- When you want your snacks, you just say "snacks!" and BOOM - you instantly get them
- You don't need to search through every pocket one by one!

**That's exactly what a HashMap does!** It stores **key-value pairs** where:

- The **key** is like the label ("snacks")
- The **value** is what's inside (your actual snacks)

### The Technical Explanation

A **HashMap** (also called Dictionary in Python, Map in JavaScript) is a data structure that:

- Stores data as **key-value pairs**
- Provides **O(1) average time** for insertions, deletions, and lookups
- Uses a **hash function** to convert keys into array indices

```python
# Python Example
student_grades = {
    "Alice": 95,
    "Bob": 87,
    "Charlie": 92
}

# Access in O(1) time!
print(student_grades["Alice"])  # Output: 95
```

### 🔬 Why Data Scientists & Data Engineers Love Hashmaps

1. **Fast Data Lookups**: When processing millions of records, finding data instantly matters
2. **Counting & Frequency Analysis**: Count occurrences of items (word frequency, user behavior)
3. **Data Joins**: Merge datasets efficiently (like SQL joins)
4. **Caching Results**: Store computed values to avoid re-calculation
5. **Feature Engineering**: Map categorical data to numerical values

**Real Data Science Example:**

```python
# Count word frequency in documents (NLP preprocessing)
word_count = {}
for word in document:
    word_count[word] = word_count.get(word, 0) + 1
```

---

## 🎨 Part 2: Understanding Sets

### The Kid-Friendly Explanation

Imagine you're collecting Pokemon cards:

- You only want **ONE** of each card (no duplicates!)
- You want to quickly check: "Do I have Pikachu already?"
- You want to compare: "Which cards do my friends have that I don't?"

**That's a Set!** It's like a magic box that:

- Only keeps unique items
- Tells you super fast if something is inside
- Can compare with other boxes easily

### The Technical Explanation

A **Set** is a data structure that:

- Stores **unique elements only** (no duplicates)
- Provides **O(1) average time** for checking membership
- Supports mathematical set operations (union, intersection, difference)

```python
# Python Example
unique_visitors = {101, 205, 303, 101}  # 101 appears twice
print(unique_visitors)  # Output: {101, 205, 303} - duplicate removed!

# Check membership in O(1)
print(303 in unique_visitors)  # Output: True
```

### 🔬 Why Data Scientists & Data Engineers Love Sets

1. **Remove Duplicates**: Clean datasets instantly
2. **Find Unique Values**: Get distinct categories, user IDs, etc.
3. **Data Comparison**: Find common/different elements between datasets
4. **Filtering**: Check if data points belong to specific groups
5. **Data Quality**: Validate data against allowed values

**Real Data Engineering Example:**

```python
# Find users who made purchases but didn't subscribe
purchasers = {101, 102, 103, 104}
subscribers = {102, 104, 105}

# Users who purchased but aren't subscribers
target_users = purchasers - subscribers  # {101, 103}
```

---

## 💻 Practice Problems

### Problem 1: Two Sum (Classic HashMap Problem)

**Problem**: Given an array of numbers and a target, find two numbers that add up to the target.

**Example**:

- Input: `nums = [2, 7, 11, 15]`, `target = 9`
- Output: `[0, 1]` (because nums[0] + nums[1] = 2 + 7 = 9)

**Solution with HashMap**:

```python
def two_sum(nums, target):
    seen = {}  # HashMap to store {number: index}

    for i, num in enumerate(nums):
        complement = target - num  # What number do we need?

        if complement in seen:  # O(1) lookup!
            return [seen[complement], i]

        seen[num] = i  # Store current number

    return []

# Test
print(two_sum([2, 7, 11, 15], 9))  # Output: [0, 1]
```

**Why This Matters for Data Science**:

- Pattern: Finding matching pairs in datasets
- Used in: Recommendation systems (finding complementary products), duplicate detection

---

### Problem 2: Contains Duplicate (Classic Set Problem)

**Problem**: Check if an array has any duplicate values.

**Example**:

- Input: `[1, 2, 3, 1]`
- Output: `True` (1 appears twice)

**Solution with Set**:

```python
def contains_duplicate(nums):
    seen = set()

    for num in nums:
        if num in seen:  # O(1) lookup!
            return True
        seen.add(num)

    return False

# OR even simpler:
def contains_duplicate_short(nums):
    return len(nums) != len(set(nums))

# Test
print(contains_duplicate([1, 2, 3, 1]))  # True
print(contains_duplicate([1, 2, 3, 4]))  # False
```

**Why This Matters for Data Engineering**:

- Data Quality: Detect duplicate records in databases
- ETL Pipelines: Ensure data integrity during transformations

---

### Problem 3: Group Anagrams (Advanced HashMap)

**Problem**: Group words that are anagrams (same letters, different order).

**Example**:

- Input: `["eat", "tea", "tan", "ate", "nat", "bat"]`
- Output: `[["eat","tea","ate"], ["tan","nat"], ["bat"]]`

**Solution**:

```python
from collections import defaultdict

def group_anagrams(words):
    groups = defaultdict(list)  # HashMap with list values

    for word in words:
        # Sort letters as key: "eat" -> "aet"
        sorted_word = ''.join(sorted(word))
        groups[sorted_word].append(word)

    return list(groups.values())

# Test
print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# Output: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

**Why This Matters for Data Science**:

- Text Processing: Finding similar patterns in text data
- Feature Engineering: Grouping similar categories
- Data Normalization: Standardizing text inputs

---

## 🎯 Key Takeaways

### When to Use HashMap:

- ✅ Need to count things (frequency analysis)
- ✅ Need fast lookups by key
- ✅ Need to store relationships (key → value)
- ✅ Building indexes or caches

### When to Use Set:

- ✅ Need unique values only
- ✅ Need to check membership quickly
- ✅ Need to compare groups (intersection, union)
- ✅ Need to remove duplicates

### Time Complexity Cheat Sheet:

| Operation | HashMap | Set  |
| --------- | ------- | ---- |
| Insert    | O(1)    | O(1) |
| Delete    | O(1)    | O(1) |
| Search    | O(1)    | O(1) |
| Space     | O(n)    | O(n) |

---

## 🚀 Next Steps for Practice

1. **LeetCode Easy**: Valid Anagram, Jewels and Stones, Single Number
2. **LeetCode Medium**: Top K Frequent Elements, Longest Substring Without Repeating Characters
3. **Data Science Project**: Build a word frequency analyzer for text documents
4. **Data Engineering Project**: Create a data deduplication pipeline

---

## 💡 Pro Tips

1. **In Python**: Use `defaultdict` for cleaner code when counting
2. **For Big Data**: Hashmaps can consume memory - consider approximate algorithms (like Count-Min Sketch) for huge datasets
3. **Interview Tip**: If you need O(1) lookup, think HashMap or Set first!

Happy Learning! 🎉
