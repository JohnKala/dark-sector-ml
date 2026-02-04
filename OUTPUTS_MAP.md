# Dark Sector ML - Outputs Map

**Branch:** `feature/adversarial-generalization-sweep`  
**Head Commit:** `99b000f`  
**Updated:** 2026-02-04

---

## 1. Output Directory Structure

```
dark-sector-ml/
├── results/                              # Primary output location for scripts
│   ├── adv_gen_sweep/                    # ← YOUR UNTRACKED ARTIFACTS
│   │   └── {source_name}/                # e.g., mDark-1_rinv-0.3_alpha-high
│   │       ├── sweep_summary.json        # Aggregated rankings
│   │       ├── sweep_results.csv         # All results flat
│   │       ├── visualizations/           # PNG plots
│   │       │   ├── generalization_heatmap.png
│   │       │   ├── config_comparison_bar.png
│   │       │   └── improvement_over_baseline.png
│   │       └── config_{name}/            # Per-config subdirectory
│   │           ├── training_history.json
│   │           ├── cross_eval_results.json
│   │           └── model_checkpoints/    # (if --save_models)
│   ├── generalization/                   # run_generalization_comparison.py
│   ├── test_generalization/              # Test runs
│   └── test_viz*/                        # Test visualization outputs
│
├── outputs/                              # Alternative output (older scripts)
│   ├── figures/
│   ├── feature_distributions/
│   ├── models/
│   ├── reports/
│   └── sensitivity_analysis/
│
├── experiments/                          # Experiment management
│   ├── configs/                          # YAML configuration files
│   │   ├── baseline.yaml
│   │   ├── adversarial_v1.yaml
│   │   └── deepsets_optimized.yaml
│   └── results/                          # Timestamped experiment runs
│       ├── 2024_01_baseline/
│       └── 2024_02_adversarial/
│
└── model_checkpoints/                    # Saved model weights
```

---

## 2. Artifact Catalog

### 2.1 `results/adv_gen_sweep/{source_name}/` (YOUR UNTRACKED DIR)

| File | Format | Producer | Content |
|------|--------|----------|---------|
| `sweep_summary.json` | JSON | `aggregate_results()` → `save_json()` in `run_adversarial_generalization_sweep.py:906` | Config rankings, mean AUC, generation timestamp |
| `sweep_results.csv` | CSV | `save_csv()` in `run_adversarial_generalization_sweep.py:907` | Flat table of all (config × target) results |
| `visualizations/generalization_heatmap.png` | PNG | `generate_sweep_visualizations()` line 536 | Heatmap: configs × targets, AUC values |
| `visualizations/config_comparison_bar.png` | PNG | `generate_sweep_visualizations()` line 560 | Bar chart: mean AUC per config |
| `visualizations/improvement_over_baseline.png` | PNG | `generate_sweep_visualizations()` line 634 | Heatmap: ΔAUC vs baseline per config/target |
| `visualizations/robustness_vs_generalization.png` | PNG | `generate_sweep_visualizations()` line 588 | Scatter plot (only if `--eval_robustness`) |

**Per-config subdirectories** (`config_baseline/`, `config_alpha-0.01_eps-0.1_iter-5/`, etc.):

| File | Format | Producer | Content |
|------|--------|----------|---------|
| `training_history.json` | JSON | `save_json()` in `run_adversarial_generalization_sweep.py:825` | Epochs run, training time, final validation metric |
| `cross_eval_results.json` | JSON | `save_json()` in `run_adversarial_generalization_sweep.py:889` | Per-target evaluation: AUC, efficiency, stability |
| `model_checkpoints/*.weights.h5` | H5 | `_train_adversarial_model()` line 386 | Model weights (only if `--save_models`) |

---

### 2.2 `results/generalization/` (run_generalization_comparison.py)

| File | Format | Producer | Content |
|------|--------|----------|---------|
| `generalization_results.json` | JSON | `main()` line 291 | Cross-evaluation results for std vs adv model |
| `source_feature_shift.png` | PNG | `plot_feature_shift()` | Clean vs perturbed feature histograms |
| `source_prediction_shift.png` | PNG | `plot_prediction_shift()` | Clean vs perturbed score histograms |
| `source_delta_heatmap.png` | PNG | `plot_delta_heatmap()` | Δη vs Δφ perturbation 2D histogram |

---

### 2.3 `outputs/sensitivity_analysis/` (run_sensitivity_analysis.py)

| File Pattern | Format | Producer | Content |
|--------------|--------|----------|---------|
| `sensitivity_matrix.csv` | CSV | `run_sensitivity_analysis.py` | Cross-evaluation AUC matrix |
| `sensitivity_heatmap.png` | PNG | Same | Heatmap visualization |
| `model_*.weights.h5` | H5 | `train_model()` | Per-dataset model weights |

---

## 3. Current `results/adv_gen_sweep/` Structure

```
results/adv_gen_sweep/
└── mDark-1_rinv-0.3_alpha-high/              # Source dataset name
    ├── sweep_summary.json                     # ✓ Aggregated rankings
    ├── sweep_results.csv                      # ✓ Flat results table
    ├── visualizations/
    │   ├── config_comparison_bar.png          # ✓ Bar chart
    │   ├── generalization_heatmap.png         # ✓ Heatmap
    │   └── improvement_over_baseline.png      # ✓ Improvement matrix
    ├── config_baseline/
    │   ├── cross_eval_results.json            # ✓ Per-target eval
    │   └── training_history.json              # ✓ Training metadata
    ├── config_alpha-0.01_eps-0.01_iter-5/
    │   ├── cross_eval_results.json
    │   └── training_history.json
    └── config_alpha-0.01_eps-0.1_iter-5/
        ├── cross_eval_results.json
        └── training_history.json
```

**Observation:** This was a `--quick_run` execution (1 epoch, 2 targets, 3 configs).

---

## 4. Naming Conventions

| Pattern | Meaning | Example |
|---------|---------|---------|
| `mDark-{X}_rinv-{Y}_alpha-{Z}` | Physics parameters from source signal | `mDark-1_rinv-0.3_alpha-high` |
| `config_baseline` | Standard training (no adversarial) | `config_baseline/` |
| `config_alpha-{α}_eps-{ε}_iter-{n}` | Adversarial config | `config_alpha-0.05_eps-0.1_iter-10` |
| `{prefix}_feature_shift.png` | Feature perturbation plot | `source_feature_shift.png` |
| `{prefix}_prediction_shift.png` | Score perturbation plot | `source_prediction_shift.png` |
| `cross_eval_results.json` | Per-target evaluation results | — |
| `sweep_summary.json` | Aggregated sweep results | — |

---

## 5. Generating Code Mapping

| Output Path | Generating Script | Key Function |
|-------------|-------------------|--------------|
| `results/adv_gen_sweep/` | `run_adversarial_generalization_sweep.py` | `main()` lines 652-933 |
| `results/generalization/` | `run_generalization_comparison.py` | `main()` lines 152-294 |
| `outputs/sensitivity_analysis/` | `run_sensitivity_analysis.py` | `main()` |
| `experiments/results/` | Various experiment scripts | Timestamped experiment outputs |

---

## 6. Gitignore Recommendations

### Current `.gitignore` Status

The current `.gitignore` includes:
- `*.h5` (model weights, data files)
- `*.weights.h5` (model checkpoints)
- `*.pkl`, `*.pickle`
- `outputs/` is commented out (line 105)

**Problem:** `results/` is **NOT** in `.gitignore`, so `results/adv_gen_sweep/` will be tracked if added.

### Recommendation

**Option A: Gitignore all results (recommended for large experiments)**

Add to `.gitignore`:
```gitignore
# Experiment results (large, reproducible)
results/adv_gen_sweep/
results/**/visualizations/
results/**/*.csv

# Keep manifests
!results/**/sweep_summary.json
!results/**/README.md
```

**Option B: Keep a small manifest in-repo**

Create `results/adv_gen_sweep/mDark-1_rinv-0.3_alpha-high/MANIFEST.md`:
```markdown
# Sweep Run Manifest

- **Generated:** 2025-12-09T03:30:46
- **Source:** mDark-1_rinv-0.3_alpha-high
- **Mode:** quick_run (1 epoch, 2 targets, 3 configs)
- **Best Config:** config_alpha-0.01_eps-0.1_iter-5 (AUC=0.70)

## Reproduction
```bash
python scripts/run_adversarial_generalization_sweep.py \
    --source_signal data/raw/AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5 \
    --quick_run
```

Then gitignore everything except the manifest:
```gitignore
results/adv_gen_sweep/**
!results/adv_gen_sweep/**/MANIFEST.md
!results/adv_gen_sweep/**/sweep_summary.json
```

---

## 7. File Sizes (Typical)

| Artifact Type | Typical Size | Notes |
|---------------|--------------|-------|
| `sweep_summary.json` | 2-10 KB | Safe to commit |
| `sweep_results.csv` | 10-100 KB | Consider gitignoring |
| `*.png` visualizations | 50-500 KB each | Gitignore or use selective commits |
| `cross_eval_results.json` | 5-20 KB per config | Gitignore |
| `training_history.json` | 1-5 KB per config | Gitignore |
| `*.weights.h5` | 1-10 MB per model | Already gitignored |

---

## 8. Summary

| Question | Answer |
|----------|--------|
| What generated `results/adv_gen_sweep/`? | `scripts/run_adversarial_generalization_sweep.py` |
| Is it reproducible? | Yes, with same `--source_signal` and config |
| Should it be committed? | **No** (except manifest/summary) |
| What to keep in-repo? | `sweep_summary.json` or a `MANIFEST.md` |
