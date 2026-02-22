"""
Quicksort Empirical Analysis - Benchmarking & Visualization
"""

import time
import sys
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable, Dict, Tuple
from datetime import datetime

from quicksort import quicksort, randomized_quicksort
from generators import GENERATORS

sys.setrecursionlimit(15000)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


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
    
    return {'mean': np.mean(valid), 'std': np.std(valid), 
            'min': np.min(valid), 'max': np.max(valid)}


def run_benchmark(sizes: List[int], num_trials: int = 5) -> Dict:
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


def plot_comparison(sizes: List[int], results: Dict):
    """Performance comparison with error bars."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Quicksort Performance Comparison', fontsize=14, fontweight='bold')
    
    for idx, name in enumerate(GENERATORS.keys()):
        ax = axes[idx // 3, idx % 3]
        det_times = results['deterministic'][name]['times']
        det_stds = [s['std'] for s in results['deterministic'][name]['stats']]
        rand_times = results['randomized'][name]['times']
        rand_stds = [s['std'] for s in results['randomized'][name]['stats']]
        
        ax.errorbar(sizes, det_times, yerr=det_stds, fmt='o-', color='#e74c3c', 
                    label='Deterministic', linewidth=2, capsize=3)
        ax.errorbar(sizes, rand_times, yerr=rand_stds, fmt='s-', color='#2980b9', 
                    label='Randomized', linewidth=2, capsize=3)
        ax.set_xlabel('Array Size')
        ax.set_ylabel('Time (s)')
        ax.set_title(name.replace('_', ' ').title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'performance_comparison.png'), dpi=150)
    plt.close()
    print("Saved: performance_comparison.png")


def plot_complexity(sizes: List[int], results: Dict):
    """O(n log n) vs O(n²) comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    rand_times = results['randomized']['random']['times']
    sorted_det = results['deterministic']['sorted']['times']
    
    ax.plot(sizes, rand_times, 'o-', color='#27ae60', label='Randomized (random)', linewidth=2)
    ax.plot(sizes, sorted_det, 's-', color='#c0392b', label='Deterministic (sorted)', linewidth=2)
    
    n = np.array(sizes, dtype=float)
    if rand_times[-1] > 0:
        scale = rand_times[-1] / (n[-1] * np.log2(n[-1]))
        ax.plot(sizes, scale * n * np.log2(n), '--', color='#27ae60', alpha=0.5, label='O(n log n)')
    if sorted_det[-1] > 0 and sorted_det[-1] != float('inf'):
        scale = sorted_det[-1] / (n[-1] ** 2)
        ax.plot(sizes, scale * n ** 2, '--', color='#c0392b', alpha=0.5, label='O(n²)')
    
    ax.set_xlabel('Array Size')
    ax.set_ylabel('Time (s)')
    ax.set_title('Time Complexity: O(n log n) vs O(n²)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'complexity_comparison.png'), dpi=150)
    plt.close()
    print("Saved: complexity_comparison.png")


def plot_speedup(sizes: List[int], results: Dict):
    """Speedup analysis plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    markers = ['o', 's', '^', 'D', 'v']
    colors = plt.cm.Set2(np.linspace(0, 1, len(GENERATORS)))
    
    for idx, name in enumerate(GENERATORS.keys()):
        det = np.array(results['deterministic'][name]['times'])
        rand = np.array(results['randomized'][name]['times'])
        speedup = np.where(rand > 0, det / rand, 1)
        ax.plot(sizes, speedup, f'{markers[idx]}-', color=colors[idx],
                label=name.replace('_', ' ').title(), linewidth=2)
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Array Size')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Randomized Quicksort Speedup', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'speedup_analysis.png'), dpi=150)
    plt.close()
    print("Saved: speedup_analysis.png")


def plot_growth_rates(sizes: List[int], results: Dict, analysis: Dict):
    """Growth rate analysis plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, algo in enumerate(['deterministic', 'randomized']):
        ax = axes[i]
        ax.set_title(f'{algo.title()} Quicksort', fontweight='bold')
        for name in GENERATORS.keys():
            times = results[algo][name]['times']
            comp = analysis[algo][name]['complexity']
            r2 = analysis[algo][name]['r_squared']
            ax.plot(sizes, times, 'o-', label=f"{name}: {comp} (R²={r2:.3f})", linewidth=2)
        ax.set_xlabel('Array Size')
        ax.set_ylabel('Time (s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'growth_rate_analysis.png'), dpi=150)
    plt.close()
    print("Saved: growth_rate_analysis.png")


def plot_heatmap(results: Dict):
    """Speedup heatmap."""
    sizes = results['sizes']
    input_types = list(GENERATORS.keys())
    
    matrix = []
    for name in input_types:
        det = np.array(results['deterministic'][name]['times'])
        rand = np.array(results['randomized'][name]['times'])
        matrix.append(np.where(rand > 0, det / rand, 1))
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticks(range(len(input_types)))
    ax.set_yticklabels([n.replace('_', ' ').title() for n in input_types])
    ax.set_xlabel('Array Size')
    ax.set_ylabel('Input Type')
    ax.set_title('Speedup Heatmap (Green = Randomized Faster)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Speedup')
    
    for i in range(len(input_types)):
        for j in range(len(sizes)):
            color = 'white' if matrix[i, j] > 50 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.1f}x', ha='center', va='center', 
                    color=color, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'speedup_heatmap.png'), dpi=150)
    plt.close()
    print("Saved: speedup_heatmap.png")


def export_csv(results: Dict, analysis: Dict):
    """Export results to CSV."""
    with open(os.path.join(RESULTS_DIR, 'benchmark_results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Size', 'Input', 'Algorithm', 'Mean', 'Std', 'Min', 'Max', 'Complexity', 'R²'])
        
        for algo in ['deterministic', 'randomized']:
            for inp in GENERATORS.keys():
                comp = analysis[algo][inp]['complexity']
                r2 = analysis[algo][inp]['r_squared']
                for i, size in enumerate(results['sizes']):
                    s = results[algo][inp]['stats'][i]
                    writer.writerow([size, inp, algo, f"{s['mean']:.6f}", f"{s['std']:.6f}",
                                    f"{s['min']:.6f}", f"{s['max']:.6f}", comp, f"{r2:.4f}"])
    print("Saved: benchmark_results.csv")


def print_summary(results: Dict, analysis: Dict):
    """Print analysis summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Algorithm':<14} {'Input':<16} {'Complexity':<12} {'R²':<8}")
    print("-" * 50)
    for algo in ['deterministic', 'randomized']:
        for inp in GENERATORS.keys():
            c = analysis[algo][inp]
            print(f"{algo:<14} {inp:<16} {c['complexity']:<12} {c['r_squared']:.4f}")
    
    # Max speedup
    max_sp, max_inp = 0, ""
    for inp in GENERATORS.keys():
        det = results['deterministic'][inp]['times'][-1]
        rand = results['randomized'][inp]['times'][-1]
        if rand > 0 and det != float('inf'):
            sp = det / rand
            if sp > max_sp:
                max_sp, max_inp = sp, inp
    
    print(f"\nMax Speedup: {max_sp:.1f}x on {max_inp} (n={results['sizes'][-1]})")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print(f"Quicksort Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    sizes = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000]
    
    results = run_benchmark(sizes, num_trials=5)
    analysis = analyze_growth_rates(results)
    
    print("\nGenerating plots...")
    plot_comparison(sizes, results)
    plot_complexity(sizes, results)
    plot_speedup(sizes, results)
    plot_growth_rates(sizes, results, analysis)
    plot_heatmap(results)
    export_csv(results, analysis)
    print_summary(results, analysis)
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
