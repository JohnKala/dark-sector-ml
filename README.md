# Dark Sector ML Classifier

A comprehensive machine learning framework for dark sector physics analysis and classification.

## Project Structure

```
dark-sector-ml/
├── src/                    # Core source code modules
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures (Dense, DeepSets, Attention)
│   ├── training/          # Training loops and utilities
│   ├── evaluation/        # Evaluation metrics and analysis
│   ├── visualization/     # Plotting and visualization tools
│   └── utils/             # General utilities
├── notebooks/             # Jupyter notebooks
│   ├── analysis/          # Analysis notebooks
│   ├── experiments/       # Experiment notebooks
│   └── tutorials/         # Tutorial notebooks
├── experiments/           # Experiment configurations and results
├── docs/                  # Documentation
├── data/                  # Data storage (raw and processed)
├── outputs/               # Model outputs and results
├── scripts/               # Training and evaluation scripts
└── tests/                 # Unit tests
```

## Features

- **Multiple Model Architectures**: Dense, DeepSets, Attention DeepSets
- **Adversarial Training**: Robust model training capabilities
- **Sensitivity Analysis**: Comprehensive cross-parameter model evaluation
- **Parameter Space Exploration**: Automated analysis across physics parameter points
- **Cross-validation and Model Comparison**: Statistical model evaluation
- **Rich Visualization Suite**: ROC curves, heatmaps, parameter sensitivity plots
- **Production-Ready Pipeline**: CLI scripts with organized timestamped outputs
- **Modular and Extensible Design**: Clean, maintainable codebase
- **Professional Testing Framework**: Comprehensive unit tests

## Installation

```bash
pip install -e .
```

## Usage

### Sensitivity Analysis (Main Feature)
```bash
# Quick test run
python scripts/run_sensitivity_analysis.py --dataset-dir data/raw --epochs 10

# Production analysis
python scripts/run_sensitivity_analysis.py --dataset-dir data/raw --epochs 50 --model-type deepsets
```

### Standard Training
```bash
python scripts/run_standard_training.py --dataset dark_sector_data.h5 --model-type deepsets
```

### Model Comparison
```bash
python scripts/run_comparison.py --models-dir outputs/models/
```

### Jupyter Notebooks
See `notebooks/experiments/04_sensitivity_analysis.ipynb` for interactive analysis.

## Documentation

## Contributing

This repository is part of a physics thesis project on dark sector classification.