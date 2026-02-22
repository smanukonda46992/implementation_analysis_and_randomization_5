"""Timing utilities for benchmarking."""

import time
import numpy as np
from typing import List, Callable, Dict


def measure_time(sort_func: Callable, arr: List[int]) -> float:
    """Measure sorting execution time."""
    arr_copy = arr.copy()
    start = time.perf_counter()
    sort_func(arr_copy)
    return time.perf_counter() - start


def measure_with_stats(sort_func: Callable, arr: List[int], trials: int = 5) -> Dict:
    """Measure with statistical metrics (mean, std, min, max)."""
    times = []
    for _ in range(trials):
        try:
            times.append(measure_time(sort_func, arr))
        except RecursionError:
            times.append(float('inf'))
    
    valid = [t for t in times if t != float('inf')]
    if not valid:
        return {'mean': float('inf'), 'std': 0, 'min': float('inf'), 'max': float('inf')}
    
    return {
        'mean': np.mean(valid),
        'std': np.std(valid),
        'min': np.min(valid),
        'max': np.max(valid)
    }
