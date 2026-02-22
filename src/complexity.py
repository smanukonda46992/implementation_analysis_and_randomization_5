"""Complexity analysis and curve fitting utilities."""

import numpy as np
from typing import List, Dict, Tuple

from generators import GENERATORS


def fit_complexity(sizes: List[int], times: List[float]) -> Tuple[str, float]:
    """Fit data to O(n), O(n log n), or O(n²)."""
    n = np.array(sizes, dtype=float)
    t = np.array(times)
    
    valid = t != float('inf')
    if not np.any(valid):
        return "Unknown", 0.0
    n, t = n[valid], t[valid]
    
    candidates = {'O(n)': n, 'O(n log n)': n * np.log2(n), 'O(n²)': n ** 2}
    best_fit, best_r2 = None, -float('inf')
    
    for name, theoretical in candidates.items():
        scale = np.sum(t * theoretical) / np.sum(theoretical ** 2)
        predicted = scale * theoretical
        ss_res = np.sum((t - predicted) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        if r2 > best_r2:
            best_r2, best_fit = r2, name
    
    return best_fit, best_r2


def analyze_growth_rates(results: Dict) -> Dict:
    """Classify growth rates for all test cases."""
    analysis = {}
    for algo in ['deterministic', 'randomized']:
        analysis[algo] = {}
        for input_type in GENERATORS.keys():
            fit, r2 = fit_complexity(results['sizes'], results[algo][input_type]['times'])
            analysis[algo][input_type] = {'complexity': fit, 'r_squared': r2}
    return analysis
