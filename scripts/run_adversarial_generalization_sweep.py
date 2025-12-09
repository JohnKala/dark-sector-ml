"""
Adversarial Generalization Sweep Script.

This script systematically evaluates how different adversarial training configurations
affect model generalization to unseen physics parameter points.

Key Question: "Which adversarial settings improve generalization to unseen signals?"

Workflow:
1. Pick a single source dataset (e.g., mDark-1, rinv-0.3, alpha-high)
2. Train models with different adversarial configurations (including baseline)
3. For each trained model, evaluate on ALL target datasets
4. Compute generalization metrics (AUC, efficiency ratio, stability)
5. Optionally compute robustness metrics on each target
6. Aggregate results and generate visualizations

Usage:
    # Full sweep (local)
    python scripts/run_adversarial_generalization_sweep.py \
        --source_signal data/raw/AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5 \
        --output_dir results/adv_gen_sweep \
        --eval_robustness

    # Quick test
    python scripts/run_adversarial_generalization_sweep.py \
        --source_signal data/raw/AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5 \
        --quick_run

    # HPC job array (single config)
    python scripts/run_adversarial_generalization_sweep.py \
        --source_signal data/raw/AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5 \
        --single_config_idx $SLURM_ARRAY_TASK_ID
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

import tensorflow as tf

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
# DEFAULT SWEEP CONFIGURATIONS
# =============================================================================
# These represent a range from no adversarial training to very strong adversarial training.
# The configs are designed to answer: "How does adversarial strength affect generalization?"

DEFAULT_SWEEP_CONFIGS = [
    # Baseline (no adversarial training) - CONTROL GROUP
    None,
    
    # Weak adversarial (minimal perturbation)
    {'alpha': 0.01, 'grad_eps': 0.01, 'grad_iter': 5,  'grad_eta': 0.01},
    
    # Medium-weak
    {'alpha': 0.01, 'grad_eps': 0.1,  'grad_iter': 5,  'grad_eta': 0.025},
    {'alpha': 0.05, 'grad_eps': 0.1,  'grad_iter': 5,  'grad_eta': 0.025},
    
    # Medium (similar to previous "champion" config)
    {'alpha': 0.05, 'grad_eps': 0.1,  'grad_iter': 10, 'grad_eta': 0.025},
    
    # Medium-strong
    {'alpha': 0.1,  'grad_eps': 0.1,  'grad_iter': 10, 'grad_eta': 0.025},
    {'alpha': 0.1,  'grad_eps': 0.1,  'grad_iter': 20, 'grad_eta': 0.025},
    
    # Strong adversarial (aggressive perturbation)
    {'alpha': 0.1,  'grad_eps': 0.5,  'grad_iter': 10, 'grad_eta': 0.05},
    {'alpha': 0.5,  'grad_eps': 0.1,  'grad_iter': 10, 'grad_eta': 0.025},
    {'alpha': 0.5,  'grad_eps': 0.5,  'grad_iter': 20, 'grad_eta': 0.05},
    
    # Very strong (extreme regularization)
    {'alpha': 1.0,  'grad_eps': 0.5,  'grad_iter': 20, 'grad_eta': 0.05},
    
    # Ablation: High alpha, low perturbation (tests KL penalty alone)
    {'alpha': 0.5,  'grad_eps': 0.01, 'grad_iter': 10, 'grad_eta': 0.01},
    
    # Ablation: Low alpha, high perturbation (tests perturbation alone)
    {'alpha': 0.01, 'grad_eps': 0.5,  'grad_iter': 20, 'grad_eta': 0.05},
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Adversarial Generalization Sweep: Evaluate how adversarial training affects generalization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sweep with robustness evaluation
  python scripts/run_adversarial_generalization_sweep.py \\
      --source_signal data/raw/AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5 \\
      --eval_robustness

  # Quick test run
  python scripts/run_adversarial_generalization_sweep.py \\
      --source_signal data/raw/AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5 \\
      --quick_run

  # HPC job array mode (run single config)
  python scripts/run_adversarial_generalization_sweep.py \\
      --source_signal data/raw/... \\
      --single_config_idx $SLURM_ARRAY_TASK_ID
        """
    )
    
    # Required
    parser.add_argument('--source_signal', type=str, required=True,
                        help='Path to source signal H5 file')
    
    # Optional paths
    parser.add_argument('--background_path', type=str, default=None,
                        help='Path to NominalSM.h5 (default: same dir as source)')
    parser.add_argument('--output_dir', type=str, default='results/adv_gen_sweep',
                        help='Base output directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per config')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Sweep control
    parser.add_argument('--config_file', type=str, default=None,
                        help='JSON file with custom sweep configs (optional)')
    parser.add_argument('--single_config_idx', type=int, default=None,
                        help='Run only config at this index (for HPC job arrays)')
    
    # Robustness evaluation
    parser.add_argument('--eval_robustness', action='store_true',
                        help='Also evaluate robustness on each target (slower)')
    parser.add_argument('--robustness_eps', type=float, default=1e-6,
                        help='Perturbation budget for robustness eval')
    parser.add_argument('--robustness_iter', type=int, default=10,
                        help='Attack steps for robustness eval')
    
    # Misc
    parser.add_argument('--save_models', action='store_true',
                        help='Save model checkpoints (uses more disk space)')
    parser.add_argument('--quick_run', action='store_true',
                        help='Debug mode: 1 epoch, 2 targets, 3 configs')
    parser.add_argument('--skip_visualizations', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    
    return parser.parse_args()


def make_config_name(adv_config: Optional[Dict]) -> str:
    """Generate a descriptive name for an adversarial config."""
    if adv_config is None:
        return "config_baseline"
    
    alpha = adv_config.get('alpha', 0)
    eps = adv_config.get('grad_eps', 0)
    iters = adv_config.get('grad_iter', 0)
    
    return f"config_alpha-{alpha}_eps-{eps}_iter-{iters}"


def extract_source_name(source_path: str) -> str:
    """Extract a clean name from the source signal path."""
    basename = os.path.basename(source_path)
    # Remove prefix and extension
    # AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5 -> mDark-1_rinv-0.3_alpha-high
    name = basename.replace('AutomatedCMS_mZprime-2000_', '').replace('.h5', '')
    return name


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


def discover_signal_files(source_path: str) -> List[str]:
    """Discover all signal H5 files in the same directory as source."""
    data_dir = os.path.dirname(source_path)
    files = []
    
    for f in os.listdir(data_dir):
        if f.endswith('.h5') and 'NominalSM' not in f:
            files.append(f)
    
    return sorted(files)


def get_predictions(model: tf.keras.Model, data: Dict[str, Any]) -> np.ndarray:
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
    
    # Get all keys from first result
    fieldnames = list(results[0].keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Convert config dict to string for CSV
            row_copy = row.copy()
            if 'config' in row_copy and row_copy['config'] is not None:
                row_copy['config'] = str(row_copy['config'])
            writer.writerow(row_copy)


def aggregate_results(all_results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across all configs and targets."""
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


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def generate_sweep_visualizations(all_results: List[Dict], output_dir: str):
    """Generate visualization plots for the sweep results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Convert results to structured format
    configs = sorted(set(r['config_name'] for r in all_results))
    targets = sorted(set(r['target_dataset'] for r in all_results))
    
    # Create AUC matrix
    auc_matrix = np.zeros((len(configs), len(targets)))
    for r in all_results:
        i = configs.index(r['config_name'])
        j = targets.index(r['target_dataset'])
        auc_matrix[i, j] = r['gen_auc']
    
    # 1. Generalization Heatmap
    plt.figure(figsize=(14, max(8, len(configs) * 0.5)))
    
    # Shorten target names for display
    short_targets = [t.replace('AutomatedCMS_mZprime-2000_', '').replace('.h5', '') for t in targets]
    short_configs = [c.replace('config_', '') for c in configs]
    
    sns.heatmap(
        auc_matrix, 
        annot=True, 
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=short_targets,
        yticklabels=short_configs,
        vmin=0.5,
        vmax=1.0,
        cbar_kws={'label': 'ROC AUC'}
    )
    plt.xlabel('Target Dataset')
    plt.ylabel('Adversarial Config')
    plt.title('Generalization AUC: Adversarial Configs × Target Datasets')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'generalization_heatmap.png'), dpi=150)
    plt.close()
    
    # 2. Config Comparison Bar Chart (mean AUC on unseen targets)
    summary = aggregate_results(all_results)
    rankings = summary['config_rankings']
    
    plt.figure(figsize=(12, 6))
    config_names = [r['config_name'].replace('config_', '') for r in rankings]
    mean_aucs = [r['mean_gen_auc_unseen'] or 0 for r in rankings]
    std_aucs = [r['std_gen_auc_unseen'] or 0 for r in rankings]
    
    colors = ['green' if 'baseline' in c else 'steelblue' for c in config_names]
    
    bars = plt.bar(range(len(config_names)), mean_aucs, yerr=std_aucs, 
                   capsize=3, color=colors, alpha=0.8)
    plt.xticks(range(len(config_names)), config_names, rotation=45, ha='right')
    plt.ylabel('Mean AUC on Unseen Targets')
    plt.xlabel('Adversarial Configuration')
    plt.title('Generalization Performance by Adversarial Config')
    plt.axhline(y=mean_aucs[config_names.index('baseline')] if 'baseline' in config_names else 0.5, 
                color='green', linestyle='--', alpha=0.5, label='Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'config_comparison_bar.png'), dpi=150)
    plt.close()
    
    # 3. Robustness vs Generalization Scatter (if robustness data available)
    has_robustness = any(r.get('robustness_score') is not None for r in all_results)
    if has_robustness:
        plt.figure(figsize=(10, 8))
        
        for ranking in rankings:
            config_name = ranking['config_name']
            gen_auc = ranking['mean_gen_auc_unseen']
            rob_score = ranking['mean_robustness_score']
            
            if gen_auc is not None and rob_score is not None:
                color = 'green' if 'baseline' in config_name else 'steelblue'
                marker = 's' if 'baseline' in config_name else 'o'
                plt.scatter(gen_auc, rob_score, s=100, c=color, marker=marker, alpha=0.7)
                plt.annotate(config_name.replace('config_', ''), 
                            (gen_auc, rob_score), 
                            textcoords="offset points", 
                            xytext=(5, 5), 
                            fontsize=8)
        
        plt.xlabel('Mean Generalization AUC (Unseen Targets)')
        plt.ylabel('Mean Robustness Score')
        plt.title('Robustness vs. Generalization Trade-off')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'robustness_vs_generalization.png'), dpi=150)
        plt.close()
    
    # 4. Per-target improvement over baseline
    baseline_results = {r['target_dataset']: r['gen_auc'] 
                       for r in all_results if r['config_name'] == 'config_baseline'}
    
    if baseline_results:
        plt.figure(figsize=(14, 6))
        
        # Calculate improvement for each config on each target
        improvements = {}
        for r in all_results:
            if r['config_name'] == 'config_baseline':
                continue
            target = r['target_dataset']
            config = r['config_name']
            baseline_auc = baseline_results.get(target, 0.5)
            improvement = r['gen_auc'] - baseline_auc
            
            if config not in improvements:
                improvements[config] = {}
            improvements[config][target] = improvement
        
        # Create improvement matrix
        adv_configs = sorted(improvements.keys())
        imp_matrix = np.zeros((len(adv_configs), len(targets)))
        for i, config in enumerate(adv_configs):
            for j, target in enumerate(targets):
                imp_matrix[i, j] = improvements[config].get(target, 0)
        
        sns.heatmap(
            imp_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdBu',
            center=0,
            xticklabels=short_targets,
            yticklabels=[c.replace('config_', '') for c in adv_configs],
            cbar_kws={'label': 'AUC Improvement over Baseline'}
        )
        plt.xlabel('Target Dataset')
        plt.ylabel('Adversarial Config')
        plt.title('AUC Improvement over Baseline (Green = Better)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'improvement_over_baseline.png'), dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {viz_dir}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    args = parse_args()
    
    # Setup paths
    if args.background_path is None:
        args.background_path = os.path.join(os.path.dirname(args.source_signal), "NominalSM.h5")
    
    source_name = extract_source_name(args.source_signal)
    output_base = os.path.join(args.output_dir, source_name)
    os.makedirs(output_base, exist_ok=True)
    
    # Load sweep configs
    if args.config_file:
        with open(args.config_file, 'r') as f:
            configs = json.load(f)
    else:
        configs = DEFAULT_SWEEP_CONFIGS.copy()
    
    # Quick run mode
    if args.quick_run:
        print("!! QUICK RUN MODE: 1 epoch, 2 targets, 3 configs !!")
        args.epochs = 1
        configs = configs[:3]  # Baseline + 2 adversarial
    
    # HPC job array mode
    if args.single_config_idx is not None:
        if args.single_config_idx >= len(configs):
            print(f"Error: config index {args.single_config_idx} >= {len(configs)} configs")
            sys.exit(1)
        configs = [configs[args.single_config_idx]]
        print(f"HPC MODE: Running single config at index {args.single_config_idx}")
    
    # Print header
    print("=" * 70)
    print("ADVERSARIAL GENERALIZATION SWEEP")
    print("=" * 70)
    print(f"Source Dataset: {source_name}")
    print(f"Output Directory: {output_base}")
    print(f"Number of Configs: {len(configs)}")
    print(f"Epochs per Config: {args.epochs}")
    print(f"Evaluate Robustness: {args.eval_robustness}")
    print("=" * 70)
    
    # Prepare source data (once)
    print("\n[SETUP] Loading source data...")
    source_data = load_and_prepare(args.source_signal, args.background_path)
    print(f"  Train: {len(source_data['train']['labels'])} samples")
    print(f"  Val: {len(source_data['val']['labels'])} samples")
    print(f"  Test: {len(source_data['test']['labels'])} samples")
    
    # Discover all target datasets
    data_dir = os.path.dirname(args.source_signal)
    target_files = discover_signal_files(args.source_signal)
    
    if args.quick_run:
        target_files = target_files[:2]
    
    print(f"\n[SETUP] Found {len(target_files)} target datasets:")
    for tf in target_files:
        marker = " (SOURCE)" if tf == os.path.basename(args.source_signal) else ""
        print(f"  - {tf}{marker}")
    
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
    
    # Main sweep loop
    all_results = []
    sweep_start_time = time.time()
    
    for config_idx, adv_config in enumerate(configs):
        config_name = make_config_name(adv_config)
        config_dir = os.path.join(output_base, config_name)
        os.makedirs(config_dir, exist_ok=True)
        
        print(f"\n{'=' * 70}")
        print(f"CONFIG {config_idx + 1}/{len(configs)}: {config_name}")
        print(f"{'=' * 70}")
        
        if adv_config:
            print(f"  alpha={adv_config['alpha']}, eps={adv_config['grad_eps']}, "
                  f"iter={adv_config['grad_iter']}, eta={adv_config['grad_eta']}")
        else:
            print("  Standard training (no adversarial)")
        
        # Train model with this config
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
            print(f"  Training completed in {training_time:.1f}s")
        except Exception as e:
            print(f"  ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Save training history
        history_path = os.path.join(config_dir, 'training_history.json')
        history_data = {
            'config': adv_config,
            'epochs_run': len(train_results.get('history', {}).get('loss', [])),
            'training_time': training_time,
            'final_val_auc': train_results.get('history', {}).get('val_auc', [None])[-1]
        }
        save_json(history_data, history_path)
        
        # Get source predictions (for stability metrics)
        source_preds = get_predictions(model, source_data)
        
        # Cross-evaluate on all targets
        config_results = []
        
        for target_file in target_files:
            target_path = os.path.join(data_dir, target_file)
            is_source = (target_file == os.path.basename(args.source_signal))
            
            print(f"\n  Evaluating on: {target_file}" + (" (SOURCE)" if is_source else ""))
            
            # Load target data
            target_data = load_and_prepare(target_path, args.background_path)
            
            # Generalization metrics
            eval_results = evaluate_model(model, target_data, verbose=False)
            target_preds = eval_results['predictions']['y_pred_proba']
            y_true = eval_results['predictions']['y_true']
            
            # Efficiency at 1% background
            eff_metrics = calculate_efficiency_ratio(
                y_true, target_preds, target_preds, target_bg_eff=0.01
            )
            
            # Stability metrics (source → target)
            stability = calculate_divergence_metrics(source_preds, target_preds)
            
            # Robustness metrics (optional)
            robust_metrics = {}
            if robustness_evaluator is not None:
                print(f"    Running robustness evaluation...")
                robust_metrics = robustness_evaluator.evaluate(model, target_data)
            
            # Collect result
            result = {
                'config_name': config_name,
                'config': adv_config,
                'target_dataset': target_file,
                'is_source': is_source,
                'gen_auc': float(eval_results['metrics']['roc_auc']),
                'gen_pr_auc': float(eval_results['metrics']['pr_auc']),
                'sig_eff_at_1pct': float(eff_metrics['sig_eff_a']),
                'stability_kl': float(stability['kl_divergence']),
                'stability_js': float(stability['js_divergence']),
            }
            
            # Add robustness metrics if available
            if robust_metrics:
                result['clean_auc'] = float(robust_metrics['clean_auc'])
                result['robust_auc'] = float(robust_metrics['robust_auc'])
                result['robustness_score'] = float(robust_metrics['robustness_score'])
            
            config_results.append(result)
            
            # Print summary
            print(f"    AUC: {result['gen_auc']:.4f}, SigEff@1%: {result['sig_eff_at_1pct']:.4f}", end="")
            if robust_metrics:
                print(f", RobustScore: {result['robustness_score']:.4f}", end="")
            print()
        
        # Save per-config results
        save_json(config_results, os.path.join(config_dir, 'cross_eval_results.json'))
        all_results.extend(config_results)
    
    # Aggregate and save sweep summary
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)
    
    sweep_summary = aggregate_results(all_results)
    sweep_summary['total_time_seconds'] = time.time() - sweep_start_time
    sweep_summary['source_dataset'] = source_name
    sweep_summary['args'] = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'eval_robustness': args.eval_robustness
    }
    
    save_json(sweep_summary, os.path.join(output_base, 'sweep_summary.json'))
    save_csv(all_results, os.path.join(output_base, 'sweep_results.csv'))
    
    # Print rankings
    print("\nConfig Rankings (by mean AUC on unseen targets):")
    print("-" * 60)
    for i, r in enumerate(sweep_summary['config_rankings'][:10]):
        print(f"  {i+1}. {r['config_name']}: {r['mean_gen_auc_unseen']:.4f} "
              f"(±{r['std_gen_auc_unseen']:.4f})")
    
    # Generate visualizations
    if not args.skip_visualizations and len(all_results) > 0:
        print("\n[VIZ] Generating visualizations...")
        try:
            generate_sweep_visualizations(all_results, output_base)
        except Exception as e:
            print(f"  Warning: Visualization failed: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"Total time: {sweep_summary['total_time_seconds']:.1f}s")
    print(f"Results saved to: {output_base}")
    print(f"  - sweep_summary.json (aggregated rankings)")
    print(f"  - sweep_results.csv (all results)")
    print(f"  - visualizations/ (plots)")
    print(f"  - config_*/ (per-config details)")


if __name__ == "__main__":
    main()
