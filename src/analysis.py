"""Main analysis script - Quicksort benchmarking."""

import sys
import os
from datetime import datetime

from quicksort import quicksort, randomized_quicksort
from generators import GENERATORS
from benchmark import measure_with_stats
from complexity import analyze_growth_rates
from visualization import generate_all_plots
from export import export_csv, print_summary

sys.setrecursionlimit(15000)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


def run_benchmark(sizes, num_trials=5):
    """Run benchmarks across all input types and sizes."""
    results = {
        'deterministic': {name: {'times': [], 'stats': []} for name in GENERATORS},
        'randomized': {name: {'times': [], 'stats': []} for name in GENERATORS},
        'sizes': sizes
    }
    
    for size in sizes:
        print(f"\nSize: {size}")
        for name, generator in GENERATORS.items():
            arr = generator(size)
            
            det_stats = measure_with_stats(quicksort, arr, num_trials)
            results['deterministic'][name]['times'].append(det_stats['mean'])
            results['deterministic'][name]['stats'].append(det_stats)
            
            rand_stats = measure_with_stats(randomized_quicksort, arr, num_trials)
            results['randomized'][name]['times'].append(rand_stats['mean'])
            results['randomized'][name]['stats'].append(rand_stats)
            
            speedup = det_stats['mean'] / rand_stats['mean'] if rand_stats['mean'] > 0 else 1
            print(f"  {name:15} | Det: {det_stats['mean']:.5f}s | Rand: {rand_stats['mean']:.5f}s | {speedup:.1f}x")
    
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print(f"Quicksort Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    sizes = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000]
    
    results = run_benchmark(sizes, num_trials=5)
    analysis = analyze_growth_rates(results)
    
    print("\nGenerating plots...")
    generate_all_plots(sizes, results, analysis, RESULTS_DIR)
    export_csv(results, analysis, RESULTS_DIR)
    print_summary(results, analysis)
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
