# Dark Sector ML - Architecture Map

**Branch:** `feature/adversarial-generalization-sweep`  
**Head Commit:** `99b000f`  
**Generated:** 2026-02-04

---

## 1. Module Overview & Responsibilities

```
dark-sector-ml/
├── scripts/                      # CLI entry points (run experiments)
├── src/                          # Core library (reusable modules)
│   ├── config.py                 # Global constants (NUM_PARTICLES, NUM_FEATURES, DATASET_FILES)
│   ├── data/                     # Data I/O and preprocessing
│   │   ├── loader.py             # Load HDF5 datasets (load_dataset)
│   │   ├── preparation.py        # Combine signal+background files (create_dataset)
│   │   └── preprocessor.py       # Train/val/test splits, normalization, tf.data (prepare_ml_dataset, prepare_deepsets_data)
│   ├── models/                   # Neural network architectures
│   │   ├── factory.py            # create_model() - Dense and DeepSets architectures
│   │   └── adversarial.py        # AdversarialModelWrapper, AdversarialExampleGenerator, AdversarialLoss
│   ├── training/                 # Training loops
│   │   ├── trainer.py            # train_model() - standard and adversarial training loops
│   │   ├── experiments.py        # Higher-level experiment orchestration
│   │   └── ui.py                 # ProgressBarCallback for training progress
│   ├── evaluation/               # Metrics and robustness
│   │   ├── metrics.py            # evaluate_model(), calculate_efficiency_ratio(), calculate_divergence_metrics()
│   │   ├── robustness.py         # RobustnessEvaluator (standardized PGD attack)
│   │   └── comparison.py         # Model comparison utilities
│   └── visualization/            # Plotting
│       ├── plots.py              # ROC curves, combined plots
│       ├── comparison_plots.py   # Feature shift, prediction shift, delta heatmaps (YOUR MODIFIED FILE)
│       └── styling.py            # Consistent plot styling
├── data/                         # Raw/processed datasets (HDF5 files)
├── experiments/                  # YAML configs and timestamped experiment results
├── results/                      # Script output artifacts (JSON, CSV, PNG)
├── outputs/                      # Alternative output location (figures, models, reports)
├── notebooks/                    # Jupyter notebooks for interactive analysis
└── tests/                        # Unit tests
```

---

## 2. Entry Points

| Script | Purpose | Key CLI Args |
|--------|---------|--------------|
| `scripts/run_adversarial_generalization_sweep.py` | **Main sweep**: Train multiple adversarial configs, cross-evaluate on all targets | `--source_signal`, `--num_parallel`, `--eval_robustness`, `--quick_run` |
| `scripts/run_generalization_comparison.py` | **Single comparison**: Standard vs. Adversarial on source + targets (YOUR MODIFIED FILE) | `--source_signal`, `--output_dir`, `--alpha`, `--epsilon` |
| `scripts/run_sensitivity_analysis.py` | Cross-parameter sensitivity analysis | `--dataset-dir`, `--epochs`, `--model-type` |
| `scripts/aggregate_sweep_results.py` | Post-hoc aggregation of sweep results | `--results_dir` |
| `scripts/run_hyperparameter_sweep.py` | Hyperparameter grid search | `--config_file` |
| `scripts/run_physics_scan.sh` | Shell wrapper for batch physics scans | N/A |

---

## 3. End-to-End Data Flow (Typical Experiment)

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                    CONFIGURATION                        │
                              │  CLI args (argparse) → DEFAULT_ADV_CONFIG / YAML        │
                              └────────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     DATASET LOADING                                         │
│  ┌────────────────────┐     ┌─────────────────────┐     ┌─────────────────────────────────┐│
│  │ data/raw/*.h5      │────▶│ loader.load_dataset │────▶│ preparation.create_dataset      ││
│  │ (HDF5: particle    │     │   (read HDF5)       │     │   (combine signal + background) ││
│  │  features, labels) │     └─────────────────────┘     └─────────────────────────────────┘│
│  └────────────────────┘                                                 │                  │
└─────────────────────────────────────────────────────────────────────────┼──────────────────┘
                                                                          │
                                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     PREPROCESSING                                           │
│  ┌─────────────────────────────────┐     ┌─────────────────────────────────┐               │
│  │ preprocessor.prepare_ml_dataset │────▶│ preprocessor.prepare_deepsets_  │               │
│  │   (train/val/test split,        │     │   data (reshape 3D, masks)      │               │
│  │    normalization)               │     └─────────────────────────────────┘               │
│  └─────────────────────────────────┘                     │                                 │
└──────────────────────────────────────────────────────────┼─────────────────────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     MODEL CREATION                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ models/factory.create_model()  ──or──  models/adversarial.create_adversarial_model()   ││
│  │   - DeepSets (Conv1D → masked pooling → Dense)                                         ││
│  │   - Wrapped in AdversarialModelWrapper if adversarial_config != None                   ││
│  └─────────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┬───────────────────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     TRAINING                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ training/trainer.train_model()                                                          ││
│  │   - Standard: model.fit() with EarlyStopping, ReduceLROnPlateau                        ││
│  │   - Adversarial: _train_adversarial_model() custom loop with PGD attacks              ││
│  │     • AdversarialExampleGenerator.generate_adversarial_examples()                      ││
│  │     • AdversarialLoss.compute_loss() (CE + alpha * KL)                                 ││
│  └─────────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┬───────────────────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     EVALUATION                                              │
│  ┌────────────────────────────────────┐  ┌────────────────────────────────────────────────┐│
│  │ evaluation/metrics.evaluate_model  │  │ evaluation/metrics.calculate_efficiency_ratio  ││
│  │   (ROC AUC, PR AUC, F1, confusion) │  │   (signal eff @ fixed background eff)          ││
│  └────────────────────────────────────┘  └────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────┐  ┌────────────────────────────────────────────────┐│
│  │ evaluation/metrics.calculate_      │  │ evaluation/robustness.RobustnessEvaluator      ││
│  │   divergence_metrics (KL, JS)      │  │   (clean vs. robust AUC under attack)          ││
│  └────────────────────────────────────┘  └────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┬───────────────────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     VISUALIZATION & OUTPUTS                                 │
│  ┌────────────────────────────────────┐  ┌────────────────────────────────────────────────┐│
│  │ visualization/plots.py             │  │ visualization/comparison_plots.py (MODIFIED)   ││
│  │   (ROC curves, heatmaps)           │  │   (feature_shift, prediction_shift, delta)     ││
│  └────────────────────────────────────┘  └────────────────────────────────────────────────┘│
│                                                                                             │
│  Outputs:                                                                                   │
│    - results/<experiment>/sweep_summary.json                                               │
│    - results/<experiment>/sweep_results.csv                                                │
│    - results/<experiment>/config_*/cross_eval_results.json                                 │
│    - results/<experiment>/visualizations/*.png                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Design Patterns

1. **Factory Pattern**: `models/factory.py` creates models by type (`dense`, `deepsets`).
2. **Wrapper Pattern**: `AdversarialModelWrapper` wraps base Keras models to add adversarial training.
3. **Strategy Pattern**: Training mode selected by `adversarial_config` (None → standard, Dict → adversarial).
4. **Pipeline Stages**: Each src module handles one pipeline stage (data → models → training → evaluation → viz).

---

## 5. Configuration Sources

| Source | Location | Use Case |
|--------|----------|----------|
| CLI Arguments | `argparse` in each script | Runtime configuration |
| Default Configs | `DEFAULT_SWEEP_CONFIGS` in scripts, `DEFAULT_ADV_CONFIG` | Baseline hyperparameters |
| YAML Files | `experiments/configs/*.yaml` | Reproducible experiment configs |
| Global Constants | `src/config.py` | `NUM_PARTICLES=30`, `NUM_FEATURES=3`, dataset paths |

---

## 6. Uncertainty Notes

- **Inferred**: The `deepsets.py`, `dense.py`, `standard.py`, `adversarial.py` (training) files are **empty** (0 bytes) — these appear to be placeholders or were refactored into other modules.
- **Inferred**: `outputs/` seems to be an older output location; `results/` is the active one for sweep scripts.
- **Confirmed**: The adversarial training loop is custom (not `model.fit()`), implemented in `trainer._train_adversarial_model()`.
