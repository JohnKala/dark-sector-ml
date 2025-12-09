"""
Aggregate Sweep Results Script.

This script aggregates results from multiple HPC job array runs into a single
summary file. Use this after running the sweep with --single_config_idx on a cluster.

Usage:
    python scripts/aggregate_sweep_results.py \
        --sweep_dir results/adv_gen_sweep/mDark-1_rinv-0.3_alpha-high
"""

import os
import sys
import argparse
import json
import glob
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate sweep results from HPC job arrays')
    parser.add_argument('--sweep_dir', type=str, required=True,
                        help='Directory containing config_* subdirectories')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file (default: sweep_summary.json in sweep_dir)')
    return parser.parse_args()


def load_config_results(config_dir: str) -> List[Dict]:
    """Load cross_eval_results.json from a config directory."""
    results_file = os.path.join(config_dir, 'cross_eval_results.json')
    if not os.path.exists(results_file):
        return []
    
    with open(results_file, 'r') as f:
        return json.load(f)


def aggregate_results(all_results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across all configs and targets."""
    import numpy as np
    
    # Group by config
    configs = {}
    for r in all_results:
        config_name = r['config_name']
        if config_name not in configs:
            configs[config_name] = {
                'config': r['config'],
                'results': []
            }
        configs[config_name]['results'].append(r)
    
    # Compute aggregates for each config
    summary = []
    for config_name, data in configs.items():
        results = data['results']
        
        # All targets
        all_aucs = [r['gen_auc'] for r in results]
        
        # Unseen targets only (not source)
        unseen_aucs = [r['gen_auc'] for r in results if not r['is_source']]
        unseen_sig_effs = [r['sig_eff_at_1pct'] for r in results if not r['is_source']]
        
        # Robustness (if available)
        robust_scores = [r.get('robustness_score', None) for r in results]
        robust_scores = [s for s in robust_scores if s is not None]
        
        summary.append({
            'config_name': config_name,
            'config': data['config'],
            'mean_gen_auc_all': float(np.mean(all_aucs)),
            'mean_gen_auc_unseen': float(np.mean(unseen_aucs)) if unseen_aucs else None,
            'std_gen_auc_unseen': float(np.std(unseen_aucs)) if unseen_aucs else None,
            'worst_gen_auc': float(np.min(all_aucs)),
            'best_gen_auc': float(np.max(all_aucs)),
            'mean_sig_eff_unseen': float(np.mean(unseen_sig_effs)) if unseen_sig_effs else None,
            'mean_robustness_score': float(np.mean(robust_scores)) if robust_scores else None,
            'num_targets': len(results)
        })
    
    # Sort by mean_gen_auc_unseen (best first)
    summary.sort(key=lambda x: x['mean_gen_auc_unseen'] or 0, reverse=True)
    
    return {
        'generated_at': datetime.now().isoformat(),
        'num_configs': len(configs),
        'num_targets': len(all_results) // len(configs) if configs else 0,
        'config_rankings': summary
    }


def main():
    args = parse_args()
    
    if not os.path.isdir(args.sweep_dir):
        print(f"Error: {args.sweep_dir} is not a directory")
        sys.exit(1)
    
    # Find all config directories
    config_dirs = glob.glob(os.path.join(args.sweep_dir, 'config_*'))
    
    if not config_dirs:
        print(f"No config_* directories found in {args.sweep_dir}")
        sys.exit(1)
    
    print(f"Found {len(config_dirs)} config directories")
    
    # Load all results
    all_results = []
    for config_dir in sorted(config_dirs):
        results = load_config_results(config_dir)
        if results:
            print(f"  Loaded {len(results)} results from {os.path.basename(config_dir)}")
            all_results.extend(results)
        else:
            print(f"  Warning: No results in {os.path.basename(config_dir)}")
    
    if not all_results:
        print("No results found to aggregate")
        sys.exit(1)
    
    print(f"\nTotal results: {len(all_results)}")
    
    # Aggregate
    summary = aggregate_results(all_results)
    
    # Save
    output_file = args.output_file or os.path.join(args.sweep_dir, 'sweep_summary.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {output_file}")
    
    # Also save CSV
    csv_file = output_file.replace('.json', '.csv')
    import csv
    with open(csv_file.replace('summary', 'results'), 'w', newline='') as f:
        if all_results:
            fieldnames = list(all_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                row_copy = row.copy()
                if 'config' in row_copy and row_copy['config'] is not None:
                    row_copy['config'] = str(row_copy['config'])
                writer.writerow(row_copy)
    
    print(f"Results CSV saved to {csv_file.replace('summary', 'results')}")
    
    # Print rankings
    print("\nConfig Rankings (by mean AUC on unseen targets):")
    print("-" * 60)
    for i, r in enumerate(summary['config_rankings']):
        auc = r['mean_gen_auc_unseen']
        std = r['std_gen_auc_unseen']
        if auc is not None:
            print(f"  {i+1}. {r['config_name']}: {auc:.4f} (±{std:.4f})")
        else:
            print(f"  {i+1}. {r['config_name']}: N/A")


if __name__ == "__main__":
    main()
