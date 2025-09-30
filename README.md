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

- Multiple model architectures (Dense, DeepSets, Attention DeepSets)
- Adversarial training capabilities
- Cross-validation and model comparison
- Comprehensive visualization tools
- Modular and extensible design
- Professional packaging and testing framework

## Installation

```bash
pip install -e .
```

## Usage

### Training a Model
```bash
python scripts/train.py --config experiments/configs/baseline.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --model-path outputs/models/best_model.pt
```

## Documentation

- [Physics Background](docs/physics_background.md)
- [Model Architecture](docs/model_architecture.md)
- [API Reference](docs/api_reference.md)

## Contributing

This repository is part of a physics thesis project on dark sector classification.