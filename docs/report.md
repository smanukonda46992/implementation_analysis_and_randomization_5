# Quicksort Analysis Report

## 1. Implementation Details

### Deterministic Quicksort
Uses **Lomuto partition** with last element as pivot:
```
PARTITION(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j = low to high-1:
        if arr[j] <= pivot:
            i++, swap arr[i], arr[j]
    swap arr[i+1], arr[high]
    return i + 1
```

### Randomized Quicksort
Swaps random element to pivot position before partitioning:
```
RANDOMIZED_PARTITION(arr, low, high):
    rand_idx = random(low, high)
    swap arr[rand_idx], arr[high]
    return PARTITION(arr, low, high)
```

---

## 2. Time Complexity

| Case | Complexity | When |
|------|------------|------|
| Best | O(n log n) | Pivot splits array equally |
| Average | O(n log n) | Random input |
| Worst | O(n²) | Sorted input (deterministic) |

### Recurrence Relations
- **Best**: T(n) = 2T(n/2) + O(n) = O(n log n)
- **Worst**: T(n) = T(n-1) + O(n) = O(n²)

---

## 3. Space Complexity

| Case | Stack Depth |
|------|-------------|
| Best/Average | O(log n) |
| Worst | O(n) |

Algorithm is **in-place** - no auxiliary array needed.

---

## 4. Why Randomization Works

**Problem**: Deterministic version with last element pivot:
- Sorted arrays → always picks min/max as pivot
- Results in maximally unbalanced partitions
- Guaranteed O(n²)

**Solution**: Random pivot selection:
- Any element equally likely to be pivot
- Probability of consistently bad pivots ≈ 0
- Expected O(n log n) regardless of input

---

## 5. Empirical Results

### Test Setup
- Sizes: 100, 500, 1000, 2000, 3000, 5000
- Input types: random, sorted, reverse-sorted, nearly-sorted, duplicates
- Trials: 3 per configuration

### Performance Summary

| Input Type | Deterministic | Randomized |
|------------|---------------|------------|
| Random | O(n log n) | O(n log n) |
| Sorted | **O(n²)** | O(n log n) |
| Reverse | **O(n²)** | O(n log n) |
| Nearly Sorted | Slow | Fast |
| Duplicates | Similar | Similar |

### Visual Results
See `results/` folder for generated plots.

---

## 6. Conclusion

Randomized Quicksort provides:
- Consistent O(n log n) expected time
- Protection against adversarial inputs
- Minimal overhead (one random call per partition)
- Same space complexity as deterministic

**Recommendation**: Use randomized version for production systems.
