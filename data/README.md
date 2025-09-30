# Data Directory

This directory contains all datasets used in the dark sector ML project.

## Structure

- `raw/` - Original, unprocessed datasets
- `processed/` - Cleaned and preprocessed datasets ready for training
- `external/` - Data from external sources or collaborations

## Data Format

Most datasets are expected to be in HDF5 format (`.h5` or `.hdf5`) for efficient storage and loading.

## Usage

Use the data loading utilities in `src/data/` to load and preprocess datasets from this directory.
