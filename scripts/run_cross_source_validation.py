"""
Phase 3: Cross-Source Validation Script.

This script tests whether the Phase 1 findings (best average and best min adversarial
configs) hold when training on different source datasets.

Research Question: Does adversarial training improve generalization and consistency
across physics parameter points, regardless of which source dataset is used for training?

Key Metrics:
- mean_gen_auc_unseen: Mean AUC across unseen targets
- worst_gen_auc (min): Minimum AUC (worst-case)
- min_to_mean_ratio: Consistency measure (higher = more consistent)
- generalization_gap: source_auc - mean_target_auc
- cross_mDark_auc: Performance on different mDark value

Workflow:
1. For each of the 6 source datasets:
   a. Train 3 models: baseline, best_avg (alpha=0.2), best_min (alpha=5.0)
   b. Evaluate each model on all 5 other targets
   c. Calculate extended metrics
2. Aggregate results across all sources
3. Generate cross-source comparison visualizations

Usage:
    python scripts/run_cross_source_validation.py \
        --output_dir results/phase3_cross_source \
        --epochs 50 \
        --batch_size 256 \
        --eval_robustness

    # Skip already-completed sources
    python scripts/run_cross_source_validation.py \
        --skip_sources mDark-1_rinv-0.3_alpha-high
"""

import os
import sys
import argparse
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preparation import create_dataset
from src.data.preprocessor import prepare_ml_dataset, prepare_deepsets_data
from src.training.trainer import train_model
from src.evaluation.metrics import (
    evaluate_model, 
    calculate_efficiency_ratio, 
    calculate_divergence_metrics
)
from src.evaluation.robustness import RobustnessEvaluator


# =============================================================================
# PHASE 3 CONFIGURATION
# =============================================================================

# Winning configs from Phase 2 analysis
PHASE3_CONFIGS = [
    # Baseline (control) - no adversarial training
    None,
    
    # Best Average: alpha=0.2, eps=0.5, iter=20
    # Selected for highest mean AUC on unseen targets
    {'alpha': 0.2, 'grad_eps': 0.5, 'grad_iter': 20, 'grad_eta': 0.1},
    
    # Best Min: alpha=5.0, eps=0.1, iter=10
    # Selected for highest minimum (worst-case) AUC - most consistent
    {'alpha': 5.0, 'grad_eps': 0.1, 'grad_iter': 10, 'grad_eta': 0.02},
]

CONFIG_NAMES = [
    'baseline',
    'best_avg_alpha-0.2',
    'best_min_alpha-5.0',
]

# All source datasets to train on
SOURCE_DATASETS = [
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.2_alpha-peak.h5",
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5",
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-low.h5",
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak.h5",
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.8_alpha-peak.h5",
    "AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak.h5",
]

BACKGROUND_FILE = "NominalSM.h5"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Phase 3: Cross-Source Validation - Test if adversarial training benefits hold across sources',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Directory containing H5 data files')
    parser.add_argument('--output_dir', type=str, default='results/phase3_cross_source',
                        help='Output directory for results')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per model')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Evaluation
    parser.add_argument('--eval_robustness', action='store_true',
                        help='Evaluate robustness on each target')
    parser.add_argument('--robustness_eps', type=float, default=0.1,
                        help='Perturbation budget for robustness eval')
    parser.add_argument('--robustness_iter', type=int, default=20,
                        help='Attack steps for robustness eval')
    
    # Control
    parser.add_argument('--skip_sources', type=str, nargs='*', default=[],
                        help='Source names to skip (if already completed)')
    parser.add_argument('--save_models', action='store_true',
                        help='Save model checkpoints')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    
    return parser.parse_args()


def extract_source_name(filepath: str) -> str:
    """Extract a clean name from the source signal path."""
    basename = os.path.basename(filepath)
    name = basename.replace('AutomatedCMS_mZprime-2000_', '').replace('.h5', '')
    return name


def get_mDark_value(source_name: str) -> str:
    """Extract mDark value from source name."""
    # e.g., "mDark-1_rinv-0.3_alpha-high" -> "1"
    # e.g., "mDark-5_rinv-0.3_alpha-peak" -> "5"
    if 'mDark-1' in source_name:
        return '1'
    elif 'mDark-5' in source_name:
        return '5'
    return 'unknown'


def is_cross_mDark(source_name: str, target_name: str) -> bool:
    """Check if source and target have different mDark values."""
    return get_mDark_value(source_name) != get_mDark_value(target_name)


def load_and_prepare(signal_path: str, background_path: str) -> Dict[str, Any]:
    """Load and prepare data for DeepSets model."""
    combined_data = create_dataset(
        [signal_path, background_path],
        use_scaled=True,
        signal_background_mode=True,
        verbose=False
    )
    
    ml_data = prepare_ml_dataset(
        combined_data, 
        test_size=0.2, 
        val_size=0.25, 
        normalize=True, 
        reshape_3d=True,
        verbose=False
    )
    
    return prepare_deepsets_data(ml_data, return_masks=True)


def get_predictions(model, data: Dict[str, Any]) -> np.ndarray:
    """Get model predictions on test data."""
    features = data['test']['features']
    masks = data['test']['attention_mask']
    return model.predict([features, masks], verbose=0).ravel()


def save_json(data: Any, filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def save_csv(results: List[Dict], filepath: str):
    """Save results list to CSV file."""
    if not results:
        return
    
    fieldnames = list(results[0].keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row_copy = row.copy()
            if 'config' in row_copy and row_copy['config'] is not None:
                row_copy['config'] = str(row_copy['config'])
            writer.writerow(row_copy)


def calculate_extended_metrics(config_results: List[Dict], source_name: str) -> Dict[str, float]:
    """
    Calculate extended Phase 3 metrics.
    
    Returns:
    - min_to_mean_ratio: min_auc / mean_auc (consistency measure, 1.0 = perfect)
    - generalization_gap: source_auc - mean_target_auc
    - cross_mDark_auc: AUC on target with different mDark value
    - same_mDark_mean: Mean AUC on targets with same mDark value
    """
    # Separate source and target results
    source_result = None
    target_results = []
    cross_mDark_results = []
    same_mDark_results = []
    
    for r in config_results:
        target_name = extract_source_name(r['target_dataset'])
        
        if r['is_source']:
            source_result = r
        else:
            target_results.append(r)
            
            if is_cross_mDark(source_name, target_name):
                cross_mDark_results.append(r)
            else:
                same_mDark_results.append(r)
    
    # Calculate metrics
    target_aucs = [r['gen_auc'] for r in target_results]
    mean_auc = np.mean(target_aucs) if target_aucs else 0
    min_auc = np.min(target_aucs) if target_aucs else 0
    std_auc = np.std(target_aucs) if target_aucs else 0
    
    metrics = {
        'mean_gen_auc_unseen': float(mean_auc),
        'min_gen_auc': float(min_auc),
        'std_gen_auc': float(std_auc),
        'min_to_mean_ratio': float(min_auc / mean_auc) if mean_auc > 0 else 0,
        'generalization_gap': float(source_result['gen_auc'] - mean_auc) if source_result else 0,
        'source_auc': float(source_result['gen_auc']) if source_result else 0,
    }
    
    # Cross-mDark metrics
    if cross_mDark_results:
        cross_mDark_aucs = [r['gen_auc'] for r in cross_mDark_results]
        metrics['cross_mDark_auc'] = float(np.mean(cross_mDark_aucs))
        metrics['cross_mDark_min'] = float(np.min(cross_mDark_aucs))
    else:
        metrics['cross_mDark_auc'] = None
        metrics['cross_mDark_min'] = None
    
    # Same-mDark metrics
    if same_mDark_results:
        same_mDark_aucs = [r['gen_auc'] for r in same_mDark_results]
        metrics['same_mDark_mean'] = float(np.mean(same_mDark_aucs))
        metrics['same_mDark_std'] = float(np.std(same_mDark_aucs))
    else:
        metrics['same_mDark_mean'] = None
        metrics['same_mDark_std'] = None
    
    return metrics


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def run_single_source(source_file: str, args, all_target_files: List[str]) -> List[Dict]:
    """Train all configs on a single source and evaluate on all targets."""
    
    source_name = extract_source_name(source_file)
    source_path = os.path.join(args.data_dir, source_file)
    background_path = os.path.join(args.data_dir, BACKGROUND_FILE)
    
    source_output_dir = os.path.join(args.output_dir, source_name)
    os.makedirs(source_output_dir, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print(f"SOURCE: {source_name}")
    print(f"{'=' * 70}")
    
    # Load source data
    print("\n  Loading source data...")
    source_data = load_and_prepare(source_path, background_path)
    print(f"    Train: {len(source_data['train']['labels'])} samples")
    print(f"    Val: {len(source_data['val']['labels'])} samples")
    print(f"    Test: {len(source_data['test']['labels'])} samples")
    
    # Setup robustness evaluator if needed
    robustness_evaluator = None
    if args.eval_robustness:
        robustness_evaluator = RobustnessEvaluator(
            attack_config={
                'grad_eps': args.robustness_eps,
                'grad_iter': args.robustness_iter,
                'grad_eta': args.robustness_eps / 5
            },
            batch_size=args.batch_size
        )
    
    source_results = []
    
    # Train each config
    for config_idx, (adv_config, config_name) in enumerate(zip(PHASE3_CONFIGS, CONFIG_NAMES)):
        
        config_dir = os.path.join(source_output_dir, f'config_{config_name}')
        os.makedirs(config_dir, exist_ok=True)
        
        print(f"\n  [{config_idx + 1}/{len(PHASE3_CONFIGS)}] Training {config_name}...")
        
        if adv_config:
            print(f"      alpha={adv_config['alpha']}, eps={adv_config['grad_eps']}, "
                  f"iter={adv_config['grad_iter']}")
        else:
            print(f"      Standard training (no adversarial)")
        
        # Train model
        train_start = time.time()
        try:
            train_results = train_model(
                prepared_data=source_data,
                model_type='deepsets',
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                adversarial_config=adv_config,
                verbose=args.verbose,
                save_model=args.save_models,
                output_dir=config_dir if args.save_models else "."
            )
            model = train_results['model']
            training_time = time.time() - train_start
            print(f"      Training completed in {training_time:.1f}s")
        except Exception as e:
            print(f"      ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Save training history
        history = train_results.get('history', {})
        loss_key = 'total_loss' if 'total_loss' in history else 'loss'
        val_key = 'val_loss' if 'val_loss' in history else 'val_auc'
        history_data = {
            'config': adv_config,
            'config_name': config_name,
            'source': source_name,
            'epochs_run': len(history.get(loss_key, [])),
            'training_time': training_time,
            'final_val_metric': history.get(val_key, [None])[-1] if history.get(val_key) else None,
        }
        save_json(history_data, os.path.join(config_dir, 'training_history.json'))
        
        # Get source predictions for stability metrics
        source_preds = get_predictions(model, source_data)
        
        # Evaluate on all targets
        config_results = []
        
        for target_file in all_target_files:
            target_path = os.path.join(args.data_dir, target_file)
            target_name = extract_source_name(target_file)
            is_source = (target_file == source_file)
            
            # Load target data
            target_data = load_and_prepare(target_path, background_path)
            
            # Evaluation metrics
            eval_results = evaluate_model(model, target_data, verbose=False)
            target_preds = eval_results['predictions']['y_pred_proba']
            y_true = eval_results['predictions']['y_true']
            
            # Efficiency at 1% background
            eff_metrics = calculate_efficiency_ratio(
                y_true, target_preds, target_preds, target_bg_eff=0.01
            )
            
            # Stability metrics
            stability = calculate_divergence_metrics(source_preds, target_preds)
            
            # Robustness metrics
            robust_metrics = {}
            if robustness_evaluator is not None:
                robust_metrics = robustness_evaluator.evaluate(model, target_data)
            
            # Collect result
            result = {
                'source': source_name,
                'config_name': config_name,
                'config': adv_config,
                'target_dataset': target_file,
                'target_name': target_name,
                'is_source': is_source,
                'is_cross_mDark': is_cross_mDark(source_name, target_name),
                'gen_auc': float(eval_results['metrics']['roc_auc']),
                'gen_pr_auc': float(eval_results['metrics']['pr_auc']),
                'sig_eff_at_1pct': float(eff_metrics['sig_eff_a']),
                'stability_kl': float(stability['kl_divergence']),
                'stability_js': float(stability['js_divergence']),
            }
            
            if robust_metrics:
                result['clean_auc'] = float(robust_metrics['clean_auc'])
                result['robust_auc'] = float(robust_metrics['robust_auc'])
                result['robustness_score'] = float(robust_metrics['robustness_score'])
            
            config_results.append(result)
            
            marker = " (SOURCE)" if is_source else (" [CROSS-mDark]" if result['is_cross_mDark'] else "")
            print(f"      {target_name}: AUC={result['gen_auc']:.4f}{marker}")
        
        # Calculate extended metrics for this config
        extended = calculate_extended_metrics(config_results, source_name)
        
        # Add extended metrics to each result
        for r in config_results:
            r['config_mean_auc'] = extended['mean_gen_auc_unseen']
            r['config_min_auc'] = extended['min_gen_auc']
            r['config_min_to_mean_ratio'] = extended['min_to_mean_ratio']
            r['config_gen_gap'] = extended['generalization_gap']
        
        # Save per-config results
        save_json(config_results, os.path.join(config_dir, 'cross_eval_results.json'))
        
        # Print summary
        print(f"      Summary: mean={extended['mean_gen_auc_unseen']:.4f}, "
              f"min={extended['min_gen_auc']:.4f}, "
              f"min/mean={extended['min_to_mean_ratio']:.3f}")
        
        source_results.extend(config_results)
    
    # Save source-level summary
    source_summary = aggregate_source_results(source_results, source_name)
    save_json(source_summary, os.path.join(source_output_dir, 'source_summary.json'))
    
    return source_results


def aggregate_source_results(results: List[Dict], source_name: str) -> Dict[str, Any]:
    """Aggregate results for a single source."""
    
    # Group by config
    configs = {}
    for r in results:
        config_name = r['config_name']
        if config_name not in configs:
            configs[config_name] = []
        configs[config_name].append(r)
    
    summary = {
        'source': source_name,
        'generated_at': datetime.now().isoformat(),
        'configs': {}
    }
    
    for config_name, config_results in configs.items():
        extended = calculate_extended_metrics(config_results, source_name)
        
        # Robustness
        robust_scores = [r.get('robustness_score') for r in config_results if r.get('robustness_score')]
        
        summary['configs'][config_name] = {
            **extended,
            'mean_robustness': float(np.mean(robust_scores)) if robust_scores else None,
            'num_targets': len([r for r in config_results if not r['is_source']])
        }
    
    return summary


# =============================================================================
# CROSS-SOURCE AGGREGATION
# =============================================================================

def aggregate_cross_source_results(all_results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across all sources."""
    
    # Group by config
    configs = {}
    for r in all_results:
        config_name = r['config_name']
        if config_name not in configs:
            configs[config_name] = []
        configs[config_name].append(r)
    
    # Calculate per-config aggregates
    config_summary = []
    for config_name, results in configs.items():
        # Get unique sources
        sources = set(r['source'] for r in results)
        
        # Per-source metrics
        source_metrics = {}
        for source in sources:
            source_results = [r for r in results if r['source'] == source]
            target_results = [r for r in source_results if not r['is_source']]
            
            if target_results:
                aucs = [r['gen_auc'] for r in target_results]
                source_metrics[source] = {
                    'mean_auc': np.mean(aucs),
                    'min_auc': np.min(aucs),
                    'std_auc': np.std(aucs),
                }
        
        # Aggregate across sources
        all_source_means = [m['mean_auc'] for m in source_metrics.values()]
        all_source_mins = [m['min_auc'] for m in source_metrics.values()]
        
        # Cross-mDark analysis
        cross_mDark_results = [r for r in results if r.get('is_cross_mDark') and not r['is_source']]
        cross_mDark_aucs = [r['gen_auc'] for r in cross_mDark_results] if cross_mDark_results else []
        
        config_summary.append({
            'config_name': config_name,
            'num_sources': len(sources),
            
            # Aggregated across all sources
            'global_mean_auc': float(np.mean(all_source_means)),
            'global_std_across_sources': float(np.std(all_source_means)),
            'global_min_auc': float(np.min(all_source_mins)),
            'global_mean_min_auc': float(np.mean(all_source_mins)),
            
            # Consistency metrics
            'mean_min_to_mean_ratio': float(np.mean([
                m['min_auc'] / m['mean_auc'] for m in source_metrics.values() if m['mean_auc'] > 0
            ])),
            
            # Cross-mDark specific
            'cross_mDark_mean_auc': float(np.mean(cross_mDark_aucs)) if cross_mDark_aucs else None,
            'cross_mDark_min_auc': float(np.min(cross_mDark_aucs)) if cross_mDark_aucs else None,
            
            # Per-source breakdown
            'per_source': source_metrics,
        })
    
    # Sort by global mean AUC
    config_summary.sort(key=lambda x: x['global_mean_auc'], reverse=True)
    
    return {
        'generated_at': datetime.now().isoformat(),
        'num_sources': len(set(r['source'] for r in all_results)),
        'num_configs': len(configs),
        'total_evaluations': len(all_results),
        'config_rankings': config_summary,
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def generate_cross_source_visualizations(all_results: List[Dict], output_dir: str):
    """Generate Phase 3 specific visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Extract unique values
    configs = sorted(set(r['config_name'] for r in all_results))
    sources = sorted(set(r['source'] for r in all_results))
    
    # =================================================================
    # 1. Cross-Source Heatmap: Mean AUC per (config, source)
    # =================================================================
    plt.figure(figsize=(12, 6))
    
    # Calculate mean AUC for each config on each source (excluding source itself)
    mean_matrix = np.zeros((len(configs), len(sources)))
    for i, config in enumerate(configs):
        for j, source in enumerate(sources):
            source_results = [r for r in all_results 
                            if r['config_name'] == config 
                            and r['source'] == source 
                            and not r['is_source']]
            if source_results:
                mean_matrix[i, j] = np.mean([r['gen_auc'] for r in source_results])
    
    short_sources = [s.replace('mDark-', 'm').replace('_rinv-', '_r').replace('_alpha-', '_a') 
                     for s in sources]
    
    sns.heatmap(
        mean_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=short_sources,
        yticklabels=configs,
        vmin=0.55,
        vmax=0.75,
        cbar_kws={'label': 'Mean AUC on Unseen Targets'}
    )
    plt.xlabel('Source Dataset (trained on)')
    plt.ylabel('Config')
    plt.title('Phase 3: Generalization AUC Across Different Sources')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'cross_source_heatmap.png'), dpi=150)
    plt.close()
    
    # =================================================================
    # 2. Min AUC Comparison
    # =================================================================
    plt.figure(figsize=(12, 6))
    
    min_matrix = np.zeros((len(configs), len(sources)))
    for i, config in enumerate(configs):
        for j, source in enumerate(sources):
            source_results = [r for r in all_results 
                            if r['config_name'] == config 
                            and r['source'] == source 
                            and not r['is_source']]
            if source_results:
                min_matrix[i, j] = np.min([r['gen_auc'] for r in source_results])
    
    sns.heatmap(
        min_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=short_sources,
        yticklabels=configs,
        vmin=0.5,
        vmax=0.7,
        cbar_kws={'label': 'Min AUC (Worst Target)'}
    )
    plt.xlabel('Source Dataset (trained on)')
    plt.ylabel('Config')
    plt.title('Phase 3: Worst-Case (Min) AUC Across Sources')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'cross_source_min_heatmap.png'), dpi=150)
    plt.close()
    
    # =================================================================
    # 3. Config Consistency Bar Chart
    # =================================================================
    plt.figure(figsize=(10, 6))
    
    # For each config, show mean and std across sources
    config_stats = []
    for config in configs:
        config_results = [r for r in all_results if r['config_name'] == config and not r['is_source']]
        aucs = [r['gen_auc'] for r in config_results]
        config_stats.append({
            'config': config,
            'mean': np.mean(aucs),
            'std': np.std(aucs),
            'min': np.min(aucs),
        })
    
    x = np.arange(len(configs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, [s['mean'] for s in config_stats], width, 
                   label='Mean AUC', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, [s['min'] for s in config_stats], width,
                   label='Min AUC', color='coral', alpha=0.8)
    
    ax.set_ylabel('AUC')
    ax.set_xlabel('Config')
    ax.set_title('Phase 3: Mean vs Min AUC by Config (Across All Sources)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.set_ylim(0.5, 0.8)
    
    # Add error bars for mean
    ax.errorbar(x - width/2, [s['mean'] for s in config_stats], 
                yerr=[s['std'] for s in config_stats], fmt='none', color='black', capsize=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'config_consistency_comparison.png'), dpi=150)
    plt.close()
    
    # =================================================================
    # 4. Cross-mDark Analysis
    # =================================================================
    plt.figure(figsize=(10, 6))
    
    cross_mDark_data = []
    same_mDark_data = []
    
    for config in configs:
        cross_results = [r for r in all_results 
                        if r['config_name'] == config 
                        and r.get('is_cross_mDark') 
                        and not r['is_source']]
        same_results = [r for r in all_results 
                       if r['config_name'] == config 
                       and not r.get('is_cross_mDark') 
                       and not r['is_source']]
        
        cross_mDark_data.append(np.mean([r['gen_auc'] for r in cross_results]) if cross_results else 0)
        same_mDark_data.append(np.mean([r['gen_auc'] for r in same_results]) if same_results else 0)
    
    x = np.arange(len(configs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, same_mDark_data, width, label='Same mDark', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, cross_mDark_data, width, label='Cross mDark', color='coral', alpha=0.8)
    
    ax.set_ylabel('Mean AUC')
    ax.set_xlabel('Config')
    ax.set_title('Phase 3: Same-mDark vs Cross-mDark Generalization')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.set_ylim(0.5, 0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'cross_mDark_analysis.png'), dpi=150)
    plt.close()
    
    # =================================================================
    # 5. Min-to-Mean Ratio (Consistency Score)
    # =================================================================
    plt.figure(figsize=(10, 6))
    
    min_to_mean_ratios = []
    for config in configs:
        config_results = [r for r in all_results if r['config_name'] == config and not r['is_source']]
        
        # Calculate per-source, then average
        sources = set(r['source'] for r in config_results)
        ratios = []
        for source in sources:
            source_results = [r for r in config_results if r['source'] == source]
            aucs = [r['gen_auc'] for r in source_results]
            if aucs:
                ratios.append(np.min(aucs) / np.mean(aucs))
        
        min_to_mean_ratios.append({
            'config': config,
            'mean_ratio': np.mean(ratios) if ratios else 0,
            'std_ratio': np.std(ratios) if ratios else 0,
        })
    
    x = np.arange(len(configs))
    colors = ['green' if 'baseline' in c else 'steelblue' for c in configs]
    
    plt.bar(x, [r['mean_ratio'] for r in min_to_mean_ratios], 
            yerr=[r['std_ratio'] for r in min_to_mean_ratios],
            capsize=5, color=colors, alpha=0.8)
    plt.xticks(x, configs)
    plt.ylabel('Min-to-Mean Ratio (higher = more consistent)')
    plt.xlabel('Config')
    plt.title('Phase 3: Consistency Score (Min/Mean AUC) by Config')
    plt.ylim(0.7, 1.0)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect consistency')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'min_to_mean_ratio.png'), dpi=150)
    plt.close()
    
    print(f"  Visualizations saved to {viz_dir}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print header
    print("=" * 70)
    print("PHASE 3: CROSS-SOURCE VALIDATION")
    print("=" * 70)
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Configs: {len(PHASE3_CONFIGS)} ({', '.join(CONFIG_NAMES)})")
    print(f"Sources: {len(SOURCE_DATASETS)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Evaluate Robustness: {args.eval_robustness}")
    if args.skip_sources:
        print(f"Skipping Sources: {args.skip_sources}")
    print("=" * 70)
    
    # Verify all files exist
    print("\n[SETUP] Verifying data files...")
    for source_file in SOURCE_DATASETS:
        path = os.path.join(args.data_dir, source_file)
        if os.path.exists(path):
            print(f"  ✓ {source_file}")
        else:
            print(f"  ✗ {source_file} NOT FOUND")
            sys.exit(1)
    
    bg_path = os.path.join(args.data_dir, BACKGROUND_FILE)
    if os.path.exists(bg_path):
        print(f"  ✓ {BACKGROUND_FILE}")
    else:
        print(f"  ✗ {BACKGROUND_FILE} NOT FOUND")
        sys.exit(1)
    
    sweep_start_time = time.time()
    all_results = []
    
    # Main loop: iterate over sources
    for source_idx, source_file in enumerate(SOURCE_DATASETS):
        source_name = extract_source_name(source_file)
        
        # Check if should skip
        if source_name in args.skip_sources:
            print(f"\n[SKIP] Skipping {source_name} (already completed)")
            continue
        
        print(f"\n[SOURCE {source_idx + 1}/{len(SOURCE_DATASETS)}]")
        
        source_results = run_single_source(source_file, args, SOURCE_DATASETS)
        all_results.extend(source_results)
        
        # Save intermediate results
        save_csv(all_results, os.path.join(args.output_dir, 'cross_source_results.csv'))
    
    # Final aggregation
    print("\n" + "=" * 70)
    print("AGGREGATING CROSS-SOURCE RESULTS")
    print("=" * 70)
    
    cross_source_summary = aggregate_cross_source_results(all_results)
    cross_source_summary['total_time_seconds'] = time.time() - sweep_start_time
    cross_source_summary['args'] = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'eval_robustness': args.eval_robustness,
    }
    
    save_json(cross_source_summary, os.path.join(args.output_dir, 'cross_source_summary.json'))
    save_csv(all_results, os.path.join(args.output_dir, 'cross_source_results.csv'))
    
    # Print rankings
    print("\nConfig Rankings (by global mean AUC):")
    print("-" * 60)
    for i, r in enumerate(cross_source_summary['config_rankings']):
        print(f"  {i+1}. {r['config_name']}: mean={r['global_mean_auc']:.4f}, "
              f"min={r['global_min_auc']:.4f}, "
              f"consistency={r['mean_min_to_mean_ratio']:.3f}")
    
    # Generate visualizations
    print("\n[VIZ] Generating cross-source visualizations...")
    try:
        generate_cross_source_visualizations(all_results, args.output_dir)
    except Exception as e:
        print(f"  Warning: Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    total_time = time.time() - sweep_start_time
    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {args.output_dir}")
    print(f"  - cross_source_summary.json")
    print(f"  - cross_source_results.csv")
    print(f"  - visualizations/")
    print(f"  - <source>/config_*/")
    
    # Key findings
    print("\n[KEY FINDINGS]")
    rankings = cross_source_summary['config_rankings']
    best_mean = rankings[0]
    best_consistency = max(rankings, key=lambda x: x['mean_min_to_mean_ratio'])
    
    print(f"  Best Mean AUC: {best_mean['config_name']} ({best_mean['global_mean_auc']:.4f})")
    print(f"  Best Consistency: {best_consistency['config_name']} "
          f"(min/mean={best_consistency['mean_min_to_mean_ratio']:.3f})")
    
    # Check if findings hold
    baseline = next((r for r in rankings if r['config_name'] == 'baseline'), None)
    best_min_config = next((r for r in rankings if 'best_min' in r['config_name']), None)
    
    if baseline and best_min_config:
        if best_min_config['mean_min_to_mean_ratio'] > baseline['mean_min_to_mean_ratio']:
            print(f"\n  ✓ HYPOTHESIS SUPPORTED: best_min config has higher consistency than baseline")
        else:
            print(f"\n  ✗ HYPOTHESIS NOT SUPPORTED: baseline has equal or higher consistency")


if __name__ == "__main__":
    main()
