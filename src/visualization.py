"""Visualization functions for analysis results."""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

from generators import GENERATORS


def plot_comparison(sizes: List[int], results: Dict, output_dir: str):
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
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=150)
    plt.close()
    print("Saved: performance_comparison.png")


def plot_complexity(sizes: List[int], results: Dict, output_dir: str):
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
    plt.savefig(os.path.join(output_dir, 'complexity_comparison.png'), dpi=150)
    plt.close()
    print("Saved: complexity_comparison.png")


def plot_speedup(sizes: List[int], results: Dict, output_dir: str):
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
    plt.savefig(os.path.join(output_dir, 'speedup_analysis.png'), dpi=150)
    plt.close()
    print("Saved: speedup_analysis.png")


def plot_growth_rates(sizes: List[int], results: Dict, analysis: Dict, output_dir: str):
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
    plt.savefig(os.path.join(output_dir, 'growth_rate_analysis.png'), dpi=150)
    plt.close()
    print("Saved: growth_rate_analysis.png")


def plot_heatmap(results: Dict, output_dir: str):
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
    plt.savefig(os.path.join(output_dir, 'speedup_heatmap.png'), dpi=150)
    plt.close()
    print("Saved: speedup_heatmap.png")


def generate_all_plots(sizes: List[int], results: Dict, analysis: Dict, output_dir: str):
    """Generate all visualization plots."""
    plot_comparison(sizes, results, output_dir)
    plot_complexity(sizes, results, output_dir)
    plot_speedup(sizes, results, output_dir)
    plot_growth_rates(sizes, results, analysis, output_dir)
    plot_heatmap(results, output_dir)
