"""
Configuration file for dark sector ML project.
Contains all constants, hyperparameters, and global settings.
"""

import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, mannwhitneyu

# Dataset configuration
DATASET_FILES = [
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak.h5",
    "AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak.h5",
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.2_alpha-peak.h5",
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.8_alpha-peak.h5",
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-low.h5",
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5",
    "NominalSM.h5"
]

# Data structure constants
NUM_PARTICLES = 20
NUM_FEATURES = 3
FEATURE_NAMES = ["pT", "eta", "phi"]

# Model hyperparameters
CONFIG = {
    "epochs": 50,
    "batch_size": 256,
    "hidden_units": [128, 64],
    "dropout": 0.2,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "patience": 5
}