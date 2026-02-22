"""
Quicksort Algorithm: Deterministic and Randomized Implementations
"""

import random
from typing import List


def partition(arr: List[int], low: int, high: int) -> int:
    """
    Lomuto partition scheme using last element as pivot.
    Returns final position of pivot.
    """
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quicksort(arr: List[int], low: int = None, high: int = None) -> None:
    """
    Deterministic Quicksort - last element as pivot.
    In-place sorting.
    """
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)


def randomized_partition(arr: List[int], low: int, high: int) -> int:
    """
    Partition with random pivot selection.
    Swaps random element to end, then partitions.
    """
    rand_idx = random.randint(low, high)
    arr[rand_idx], arr[high] = arr[high], arr[rand_idx]
    return partition(arr, low, high)


def randomized_quicksort(arr: List[int], low: int = None, high: int = None) -> None:
    """
    Randomized Quicksort - random pivot selection.
    Avoids O(n²) worst case with high probability.
    """
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pi = randomized_partition(arr, low, high)
        randomized_quicksort(arr, low, pi - 1)
        randomized_quicksort(arr, pi + 1, high)


if __name__ == "__main__":
    test = [64, 34, 25, 12, 22, 11, 90]
    
    arr1 = test.copy()
    quicksort(arr1)
    print(f"Deterministic: {arr1}")
    
    arr2 = test.copy()
    randomized_quicksort(arr2)
    print(f"Randomized: {arr2}")
