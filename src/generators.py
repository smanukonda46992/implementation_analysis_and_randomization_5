"""
Input Array Generators for Testing
"""

import random
from typing import List


def random_array(size: int) -> List[int]:
    """Random integers array."""
    return [random.randint(0, size * 10) for _ in range(size)]


def sorted_array(size: int) -> List[int]:
    """Already sorted array - worst case for deterministic."""
    return list(range(size))


def reverse_sorted_array(size: int) -> List[int]:
    """Reverse sorted array - worst case for deterministic."""
    return list(range(size, 0, -1))


def nearly_sorted_array(size: int) -> List[int]:
    """Nearly sorted with few random swaps."""
    arr = list(range(size))
    swaps = max(1, size // 20)
    for _ in range(swaps):
        i, j = random.randint(0, size - 1), random.randint(0, size - 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def duplicates_array(size: int) -> List[int]:
    """Array with many duplicates."""
    return [random.randint(0, size // 10) for _ in range(size)]


# Dictionary mapping names to generators
GENERATORS = {
    'random': random_array,
    'sorted': sorted_array,
    'reverse_sorted': reverse_sorted_array,
    'nearly_sorted': nearly_sorted_array,
    'duplicates': duplicates_array
}
