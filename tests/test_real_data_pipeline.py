#!/usr/bin/env python3
"""
Compact end-to-end test on real data files.
Validates loader (loader.py), dataset creation (preparation.py), and ML preparation (preprocessor.py).
The test uses a small subset to keep runtime reasonable.
"""
import os
import sys
from pathlib import Path
import numpy as np

# Import project as a package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_FILES, NUM_PARTICLES, NUM_FEATURES
from src.data.loader import load_dataset
from src.data.preparation import create_dataset
from src.data.preprocessor import prepare_ml_dataset


def pick_real_files():
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    assert data_dir.exists(), f"Missing data dir: {data_dir}"
    signal_files = []
    sm_file = None
    for fn in DATASET_FILES:
        p = data_dir / fn
        if not p.exists():
            continue
        if 'NominalSM' in fn:
            sm_file = str(p)
        else:
            signal_files.append(str(p))
    # Require at least 1 signal file and the SM file
    assert sm_file is not None, "NominalSM.h5 not found"
    assert len(signal_files) >= 1, "No signal files found"
    # Use one representative signal file to keep runtime reasonable
    return [signal_files[0], sm_file]


def downsample_dataset(ds: dict, max_n: int = 10000) -> dict:
    """Downsample features/labels to at most max_n rows for faster tests."""
    n = ds['features'].shape[0]
    if n <= max_n:
        return ds
    idx = np.linspace(0, n-1, max_n).astype(int)
    ds['features'] = ds['features'][idx]
    ds['labels'] = ds['labels'][idx]
    if 'is_valid' in ds:
        ds['is_valid'] = ds['is_valid'][idx]
    return ds


def test_real_data_pipeline_compact():
    files = pick_real_files()

    # Step 1: load two real files individually and sanity-check structure
    for fp in files:
        ds = load_dataset(fp, verbose=False)
        assert isinstance(ds, dict)
        assert 'particle_features' in ds and 'n_jets' in ds and 'n_dark_jets' in ds
        assert ds['particle_features'].shape[1] == NUM_PARTICLES * NUM_FEATURES
        assert ds['n_jets'] > 0

    # Step 2: create combined dataset from these two files
    combined = create_dataset(files, verbose=False)
    assert set(['features', 'labels', 'metadata']).issubset(combined.keys())
    n, d = combined['features'].shape
    assert d == NUM_PARTICLES * NUM_FEATURES
    assert combined['labels'].shape[0] == n
    assert n > 0

    # Step 3: downsample if needed for speed
    combined = downsample_dataset(combined, max_n=10000)

    # Step 4: prepare ML dataset (splits + normalization with -999 handling)
    ml = prepare_ml_dataset(combined, normalize=True, verbose=False)
    for split in ['train', 'val', 'test']:
        assert split in ml
        # X_norm is present when normalize=True; otherwise X is used
        assert ('X' in ml[split] or 'X_norm' in ml[split]) and 'y' in ml[split]
        X_split = ml[split]['X_norm'] if 'X_norm' in ml[split] else ml[split]['X']
        y = ml[split]['y']
        assert X_split.ndim == 2 and X_split.shape[1] == NUM_PARTICLES * NUM_FEATURES
        assert y.ndim == 1 and y.shape[0] == X_split.shape[0]

    # Step 5: check normalization approximately (on non -999 entries of train split)
    X_train = ml['train']['X_norm'] if 'X_norm' in ml['train'] else ml['train']['X']
    mask = X_train != -999
    if np.any(mask):
        vals = X_train[mask]
        mean = float(vals.mean())
        std = float(vals.std())
        assert abs(mean) < 0.15, f"Mean too far from 0 after normalization: {mean}"
        assert abs(std - 1.0) < 0.2, f"Std too far from 1 after normalization: {std}"

    # Basic label sanity: both classes should appear if possible
    y_all = np.concatenate([ml[s]['y'] for s in ['train', 'val', 'test']])
    # Do not assert strict presence of both classes (some real slices could be single-class), but log
    unique = np.unique(y_all)
    print(f"Label classes present in splits: {unique}")

    # If we reach here without assertion errors, the compact pipeline passes on real data
    
if __name__ == "__main__":
    try:
        test_real_data_pipeline_compact()
        print("\n✅ Compact real-data pipeline test passed (standalone run).")
    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
