"""
Empirical Analysis: Deterministic vs Randomized Quicksort
"""

import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable, Dict

from quicksort import quicksort, randomized_quicksort
from generators import GENERATORS

sys.setrecursionlimit(15000)

# Output directory for results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


def measure_time(sort_func: Callable, arr: List[int]) -> float:
    """Measure sorting execution time."""
    arr_copy = arr.copy()
    start = time.perf_counter()
    sort_func(arr_copy)
    return time.perf_counter() - start


def run_benchmark(sizes: List[int], num_trials: int = 3) -> Dict:
    """Run benchmarks across all input types and sizes."""
    results = {
        'deterministic': {name: [] for name in GENERATORS},
        'randomized': {name: [] for name in GENERATORS}
    }
    
    for size in sizes:
        print(f"\nSize: {size}")
        
        for name, generator in GENERATORS.items():
            det_times, rand_times = [], []
            
            for _ in range(num_trials):
                arr = generator(size)
                
                try:
                    det_times.append(measure_time(quicksort, arr))
                except RecursionError:
                    det_times.append(float('inf'))
                
                rand_times.append(measure_time(randomized_quicksort, arr))
            
            avg_det = np.mean([t for t in det_times if t != float('inf')])
            avg_rand = np.mean(rand_times)
            
            results['deterministic'][name].append(avg_det)
            results['randomized'][name].append(avg_rand)
            
            print(f"  {name}: Det={avg_det:.5f}s, Rand={avg_rand:.5f}s")
    
    return results


def plot_comparison(sizes: List[int], results: Dict):
    """Generate comparison plot for all input types."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Quicksort Performance Comparison', fontsize=13, fontweight='bold')
    
    colors = {'deterministic': '#e74c3c', 'randomized': '#2980b9'}
    
    for idx, name in enumerate(GENERATORS.keys()):
        ax = axes[idx // 3, idx % 3]
        
        ax.plot(sizes, results['deterministic'][name], 'o-', 
                color=colors['deterministic'], label='Deterministic', linewidth=2)
        ax.plot(sizes, results['randomized'][name], 's-', 
                color=colors['randomized'], label='Randomized', linewidth=2)
        
        ax.set_xlabel('Array Size')
        ax.set_ylabel('Time (s)')
        ax.set_title(name.replace('_', ' ').title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'performance_comparison.png'), dpi=150)
    plt.close()
    print("\nSaved: results/performance_comparison.png")


def plot_complexity(sizes: List[int], results: Dict):
    """Plot O(n log n) vs O(n²) comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sizes, results['randomized']['random'], 'o-', color='#27ae60', 
            label='Randomized (random input)', linewidth=2)
    ax.plot(sizes, results['deterministic']['sorted'], 's-', color='#c0392b', 
            label='Deterministic (sorted input)', linewidth=2)
    
    ax.set_xlabel('Array Size (n)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time Complexity: O(n log n) vs O(n²)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'complexity_comparison.png'), dpi=150)
    plt.close()
    print("Saved: results/complexity_comparison.png")


def plot_speedup(sizes: List[int], results: Dict):
    """Plot speedup of randomized over deterministic."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name in GENERATORS.keys():
        det = np.array(results['deterministic'][name])
        rand = np.array(results['randomized'][name])
        speedup = np.where(rand > 0, det / rand, 1)
        ax.plot(sizes, speedup, 'o-', label=name.replace('_', ' ').title(), linewidth=2)
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Array Size (n)')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Randomized Quicksort Speedup', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'speedup_analysis.png'), dpi=150)
    plt.close()
    print("Saved: results/speedup_analysis.png")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 50)
    print("Quicksort Empirical Analysis")
    print("=" * 50)
    
    sizes = [100, 500, 1000, 2000, 3000, 5000]
    results = run_benchmark(sizes, num_trials=3)
    
    print("\n" + "=" * 50)
    print("Generating Plots...")
    print("=" * 50)
    
    plot_comparison(sizes, results)
    plot_complexity(sizes, results)
    plot_speedup(sizes, results)
    
    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
