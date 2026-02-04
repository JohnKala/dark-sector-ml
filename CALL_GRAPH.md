# Dark Sector ML - Call Graph

**Branch:** `feature/adversarial-generalization-sweep`  
**Head Commit:** `99b000f`  
**Updated:** 2026-02-04

---

## 1. Entrypoint: `scripts/run_adversarial_generalization_sweep.py`

**Purpose:** Systematic sweep over adversarial training configurations to evaluate generalization.

### Call Chain (Sequential Mode)

```
main()
├── parse_args()                                          # argparse CLI parsing
├── extract_source_name(source_signal)                    # "mDark-1_rinv-0.3_alpha-high"
├── discover_signal_files(source_signal)                  # Find all target H5 files
├── load_and_prepare(source_signal, background_path)
│   ├── src.data.preparation.create_dataset()
│   │   └── src.data.loader.load_dataset()               # HDF5 → numpy arrays
│   ├── src.data.preprocessor.prepare_ml_dataset()       # train/val/test splits
│   └── src.data.preprocessor.prepare_deepsets_data()    # 3D reshape + masks
│
├── [FOR EACH config IN DEFAULT_SWEEP_CONFIGS]
│   │
│   ├── make_config_name(adv_config)                      # "config_alpha-0.05_eps-0.1_iter-10"
│   │
│   ├── src.training.trainer.train_model()
│   │   ├── src.models.factory.create_model()             # if adversarial_config=None
│   │   └── src.models.adversarial.create_adversarial_model()  # if adversarial_config
│   │       ├── create_model_with_mixed_precision()
│   │       └── AdversarialModelWrapper(base_model, config)
│   │
│   │   # Adversarial path: _train_adversarial_model()
│   │   ├── create_optimized_datasets_from_prepared_data()
│   │   └── [FOR EACH epoch]
│   │       ├── adversarial_model.adversarial_train_step()
│   │       │   ├── AdversarialExampleGenerator.generate_adversarial_examples()
│   │       │   └── AdversarialLoss.compute_loss()
│   │       └── adversarial_model.validation_step()
│   │
│   ├── get_predictions(model, source_data)               # Cache source predictions
│   │
│   ├── [FOR EACH target_file IN target_files]
│   │   ├── load_and_prepare(target_path, background_path)
│   │   ├── src.evaluation.metrics.evaluate_model()
│   │   │   └── model.predict() → ROC AUC, PR AUC, F1, confusion matrix
│   │   ├── src.evaluation.metrics.calculate_efficiency_ratio()
│   │   │   └── Signal efficiency @ 1% background efficiency
│   │   ├── src.evaluation.metrics.calculate_divergence_metrics()
│   │   │   └── KL divergence, JS divergence (source→target shift)
│   │   └── [OPTIONAL] src.evaluation.robustness.RobustnessEvaluator.evaluate()
│   │       └── AdversarialExampleGenerator.generate_adversarial_examples()
│   │
│   └── save_json(config_results, config_dir/cross_eval_results.json)
│
├── aggregate_results(all_results)                        # Compute rankings
├── save_json(sweep_summary, output_base/sweep_summary.json)
├── save_csv(all_results, output_base/sweep_results.csv)
└── generate_sweep_visualizations(all_results, output_base)
    ├── Generalization Heatmap (seaborn heatmap)
    ├── Config Comparison Bar Chart
    ├── Robustness vs Generalization Scatter (if --eval_robustness)
    └── Improvement over Baseline Heatmap
```

### Parallel Mode Variant

When `--num_parallel > 1`, replaces the loop with:
```
run_single_config() [multiprocessing.Pool.map]
├── os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
├── tf.keras.backend.clear_session()
└── ... (same as sequential, per-config)
```

---

## 2. Entrypoint: `scripts/run_generalization_comparison.py` (MODIFIED FILE)

**Purpose:** Single Standard vs. Adversarial comparison with visualizations.

### Call Chain

```
main()
├── parse_args()
├── load_and_prepare(source_signal, background_path)
│   └── [same as above]
│
├── [TRAIN STANDARD MODEL]
│   └── src.training.trainer.train_model(adversarial_config=None)
│
├── [TRAIN ADVERSARIAL MODEL]
│   └── src.training.trainer.train_model(adversarial_config=ADV_CONFIG)
│
├── generate_visualizations(adv_model, source_data, output_dir)
│   ├── src.models.adversarial.AdversarialExampleGenerator()
│   ├── generator.generate_adversarial_examples()
│   ├── src.visualization.comparison_plots.plot_feature_shift()     # ← MODIFIED
│   ├── src.visualization.comparison_plots.plot_prediction_shift()  # ← MODIFIED
│   └── src.visualization.comparison_plots.plot_delta_heatmap()     # ← MODIFIED
│
├── [FOR EACH target_file IN data_dir]
│   ├── load_and_prepare(target_path, background_path)
│   ├── src.evaluation.metrics.evaluate_model(std_model)
│   ├── src.evaluation.metrics.evaluate_model(adv_model)
│   ├── src.evaluation.metrics.calculate_efficiency_ratio()  # ← MODIFIED
│   ├── src.evaluation.metrics.calculate_divergence_metrics() # ← MODIFIED
│   └── Store results
│
└── json.dump(comparison_results, output_file)
```

---

## 3. Entrypoint: `scripts/run_sensitivity_analysis.py`

**Purpose:** Cross-parameter sensitivity analysis (model trained on one physics point, evaluated on others).

### Call Chain (Abbreviated)

```
main()
├── parse_args()
├── discover_datasets()
├── [FOR EACH source_dataset]
│   ├── train_model()
│   └── [FOR EACH target_dataset]
│       └── evaluate_model()
└── generate_sensitivity_heatmaps()
```

---

## 4. Changed-Code Impact Analysis

### Modified File: `scripts/run_generalization_comparison.py`

| Function/Section | Called By | Impact |
|------------------|-----------|--------|
| `generate_visualizations()` | `main()` | Controls what plots are generated for adversarial perturbation analysis |
| `ADV_CONFIG` usage | `train_model()` | Hyperparameters for adversarial training in this script |
| Cross-evaluation loop | `main()` | Determines which datasets are evaluated and what metrics are computed |

**Downstream Effects:**
- Output file: `{output_dir}/generalization_results.json`
- Visualization files: `{output_dir}/{prefix}_feature_shift.png`, `{prefix}_prediction_shift.png`, `{prefix}_delta_heatmap.png`

---

### Modified File: `src/evaluation/metrics.py`

| Function | Used By | Impact |
|----------|---------|--------|
| `evaluate_model()` | All scripts | Core evaluation metrics (AUC, precision, recall, F1) |
| `calculate_efficiency_ratio()` | `run_generalization_comparison.py`, `run_adversarial_generalization_sweep.py` | Signal efficiency at fixed background efficiency |
| `calculate_divergence_metrics()` | Same as above | KL/JS divergence for prediction stability |

**Downstream Effects:**
- Any change to metric calculation affects **all evaluation outputs** (JSON, CSV, rankings).
- Changes to binning in `calculate_divergence_metrics(bins=50)` affect stability metric values.

**Call Sites:**
```
scripts/run_adversarial_generalization_sweep.py:361  → evaluate_model()
scripts/run_adversarial_generalization_sweep.py:449  → calculate_efficiency_ratio()
scripts/run_adversarial_generalization_sweep.py:454  → calculate_divergence_metrics()
scripts/run_generalization_comparison.py:233        → evaluate_model()
scripts/run_generalization_comparison.py:245        → calculate_efficiency_ratio()
scripts/run_generalization_comparison.py:260-261    → calculate_divergence_metrics()
```

---

### Modified File: `src/visualization/comparison_plots.py`

| Function | Used By | Impact |
|----------|---------|--------|
| `plot_feature_shift()` | `run_generalization_comparison.py:generate_visualizations()` | Histogram of clean vs. perturbed features |
| `plot_prediction_shift()` | Same | Histogram of model scores before/after attack |
| `plot_delta_heatmap()` | Same | 2D histogram of Δη vs Δφ perturbations |

**Downstream Effects:**
- PNG files in `{output_dir}/{prefix}_*.png`
- Only affects visualization—no impact on metrics or model training.

**Call Sites:**
```
scripts/run_generalization_comparison.py:128-148
```

---

## 5. Dependency Graph (Transitive Imports)

```
scripts/run_adversarial_generalization_sweep.py
├── src.data.preparation
│   └── src.data.loader
├── src.data.preprocessor
│   └── src.config (NUM_PARTICLES, NUM_FEATURES)
├── src.training.trainer
│   ├── src.models.factory
│   │   └── src.config
│   ├── src.models.adversarial
│   │   └── src.models.factory
│   └── src.training.ui
├── src.evaluation.metrics
└── src.evaluation.robustness
    └── src.models.adversarial

scripts/run_generalization_comparison.py
├── src.config
├── src.data.preparation
├── src.data.preprocessor
├── src.training.trainer
├── src.evaluation.metrics           # ← MODIFIED
├── src.visualization.comparison_plots # ← MODIFIED
└── src.models.adversarial
```

---

## 6. Execution Path Summary

| Script | Training Fn | Eval Fn | Viz Fn | Output Path |
|--------|-------------|---------|--------|-------------|
| `run_adversarial_generalization_sweep.py` | `train_model` → `_train_adversarial_model` | `evaluate_model`, `calculate_efficiency_ratio`, `calculate_divergence_metrics` | `generate_sweep_visualizations` | `results/adv_gen_sweep/{source_name}/` |
| `run_generalization_comparison.py` | `train_model` (×2) | Same | `plot_feature_shift`, `plot_prediction_shift`, `plot_delta_heatmap` | `results/generalization/` (or `--output_dir`) |
| `run_sensitivity_analysis.py` | `train_model` | `evaluate_model` | sensitivity heatmaps | `outputs/sensitivity_analysis/` |
