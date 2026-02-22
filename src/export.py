"""Export and reporting utilities."""

import os
import csv
from typing import Dict

from generators import GENERATORS


def export_csv(results: Dict, analysis: Dict, output_dir: str):
    """Export results to CSV."""
    path = os.path.join(output_dir, 'benchmark_results.csv')
    with open(path, 'w', newline='') as f:
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
