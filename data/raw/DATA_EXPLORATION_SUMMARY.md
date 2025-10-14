# Data Exploration Summary

This document summarizes findings from the first five notebook-style exploration cells for the CMS dark sector datasets located in `data/raw/`.

## 1) Environment and Configuration

- Project imports are working and plotting is configured.
- Core configuration from `src/config.py`:
  - `NUM_PARTICLES`: 20
  - `NUM_FEATURES`: 3
  - `FEATURE_NAMES`: ["pT", "eta", "phi"]
- Library versions (runtime):
  - numpy: 2.3.3
  - pandas: 2.3.3
  - matplotlib: 3.10.6
  - h5py: 3.14.0

## 2) Dataset Availability

All configured files are present in `data/raw/`:
- AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak.h5
- AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak.h5
- AutomatedCMS_mZprime-2000_mDark-1_rinv-0.2_alpha-peak.h5
- AutomatedCMS_mZprime-2000_mDark-1_rinv-0.8_alpha-peak.h5
- AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-low.h5
- AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5
- NominalSM.h5

Observation: 7/7 files found; each ~15.4 MB.

## 3) HDF5 Structure

Representative file inspected: `AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak.h5`

Top-level datasets (names, shapes, dtypes):
- `jet_dark_pt`: (30000,), float64
- `jet_e`: (30000,), float64
- `jet_eta`: (30000,), float64
- `jet_is_dark`: (30000,), bool
- `jet_m`: (30000,), float64
- `jet_phi`: (30000,), float64
- `jet_pt`: (30000,), float64
- `jet_rinv`: (30000,), float64
- `particle_features`: (30000, 60), float64

Notes:
- `particle_features` has 60 columns = 20 particles × 3 features (pT, eta, phi), consistent with config.
- Jet-level kinematics available and a dark/SM label per jet via `jet_is_dark`.

## 4) Physics Parameter Parsing

Parsed from filenames:
- `mZprime`: [2000]
- `mDark`: [1, 5]
- `rinv`: [0.2, 0.3, 0.8]
- `alpha`: ['low', 'peak', 'high', 'SM']

File classification:
- Signal files: 6
- Background files: 1 (`NominalSM.h5`)

Per-file mapping (summarized):
- mDark = 1, rinv = 0.3, alpha = peak
- mDark = 5, rinv = 0.3, alpha = peak
- mDark = 1, rinv = 0.2, alpha = peak
- mDark = 1, rinv = 0.8, alpha = peak
- mDark = 1, rinv = 0.3, alpha = low
- mDark = 1, rinv = 0.3, alpha = high
- Background: NominalSM.h5 (alpha = SM)

## 5) Single-File Analysis

Analyzed: `AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak.h5`

- Total jets: 30,000
- Dark jets (from `jet_is_dark`): 24,979 (83.3%)
- `particle_features` shape: (30000, 60)
- Estimated valid particle fraction (sample up to 2,000 jets): 0.768
  - Method: reshape to [N, 20, 3]; treat any particle row with any feature == -999 as padded/invalid; compute fraction of valid particles.
- Jet-level arrays present with shape (30000,): `jet_pt`, `jet_eta`, `jet_phi`, `jet_m`, `jet_e`, `jet_rinv`.

## Implications for the ML Pipeline

- __Masking/padding__: ~23% of particle slots per jet are padding (−999), so downstream models should use attention masks or validity masks where appropriate.
- __Normalization__: Our StandardScaler approach in `src/data/preprocessor.py` (temporarily replace −999 with 0, normalize, then restore −999) is suitable, as long as masked values are ignored by the model.
- __Class balance__: Files can be highly imbalanced at the per-file level (e.g., 83% dark in a representative signal file). Ensure stratified splitting or post-weighting when combining with background.
- __Feature layout__: The flattened particle features match the config and the expected DeepSets input format once reshaped to [N, 20, 3].

## 6) Feature Distributions Across Files

Methodology:
- Loaded `particle_features` from each file and reshaped to `[N, 20, 3]`.
- A particle is considered valid if none of its features equals `-999`.
- For dark-sector files, only jets with `jet_is_dark=True` were used as signal; for background we used `NominalSM.h5`.

Summary statistics (valid particles only):
- pT
  - Signal: count = 1,999,040, mean = 21.434, std = 43.537, min = 1.000, max = 1062.574
  - SM:     count =   507,344, mean = 26.955, std = 45.727, min = 1.000, max = 1671.894
- eta
  - Signal: mean = 0.002, std = 1.100, range ≈ [-5.959, 6.025]
  - SM:     mean = -0.004, std = 1.174, range ≈ [-6.474, 6.527]
- phi
  - Signal: mean = 0.005, std = 1.814, range ≈ [-π, π]
  - SM:     mean = -0.003, std = 1.814, range ≈ [-π, π]

Quick observations:
- SM background shows a slightly higher mean pT than dark signal (Δmean ≈ -5.52 GeV for signal − SM).
- eta and phi means are very close between classes with similar spreads. Tighter separation may appear when conditioning on (mDark, rinv, alpha).

Artifacts generated:
- Overlaid histograms saved to `outputs/feature_distributions/`:
  - `dist_pT.png`, `dist_eta.png`, `dist_phi.png`

Note on class balance during aggregation:
- Aggregation mirrors the intended workflow: dark jets from signal files vs. all jets from `NominalSM.h5` as background.
- Final training balance depends on sampling policy; counts above reflect raw availability, not a balanced split.

## 7) Jet-level Feature Distributions

Methodology:
- For each dark-sector file, selected only jets with `jet_is_dark=True` to form the signal aggregate.
- For background, used all jets from `NominalSM.h5`.
- Compared distributions for `jet_pt`, `jet_eta`, `jet_phi`, `jet_m`, `jet_e`.

Summary statistics:
- jet_pt
  - Signal: count = 143,402, mean = 318.160, std = 259.745, min = 1.263, max = 2111.000
  - SM    : count =  30,000, mean = 487.251, std = 329.939, min = 1.373, max = 2044.962
- jet_eta
  - Signal: mean = 0.003, std = 1.152, range ≈ [-5.715, 6.037]
  - SM    : mean = 0.003, std = 1.286, range ≈ [-6.537, 6.495]
- jet_phi
  - Signal: mean = 0.007, std = 1.815, range ≈ [-π, π]
  - SM    : mean = -0.006, std = 1.815, range ≈ [-π, π]
- jet_m
  - Signal: mean = 24.566, std = 22.704, range ≈ [0.000, 261.170]
  - SM    : mean = 30.900, std = 26.390, range ≈ [0.000, 339.131]
- jet_e
  - Signal: mean = 526.616, std = 484.659, range ≈ [1.410, 5163.952]
  - SM    : mean = 811.697, std = 600.527, range ≈ [2.342, 4560.301]

Quick observations:
- SM jets tend to have higher means for jet_pt, jet_m, and jet_e compared to dark jets in the aggregated view.
- jet_eta and jet_phi show very similar central tendencies and spreads across classes.

Clarification — Why is jet_pt signal count ≈143k (not ~180k)?
- 180,000 would be the total jets across the 6 dark-sector files if we counted all jets (6 × 30,000).
- In our aggregation for analysis we explicitly kept only jets with `jet_is_dark=True` from the dark-sector files.
- Not all jets in those files are dark. In one representative file, ~83.3% were dark; across all 6, it’s roughly ~80%.
- Rough estimate: 6 × 30,000 × 0.8 ≈ 144,000, which aligns with the measured 143,402.
- Therefore, the drop from 180k to ~143k is expected due to filtering to dark jets only.

Artifacts generated:
- Overlaid histograms saved to `outputs/jet_feature_distributions/`:
  - `dist_pt.png`, `dist_eta.png`, `dist_phi.png`, `dist_m.png`, `dist_e.png`

Note: These aggregates are for analysis only. No merging/splitting for training was performed here.

## 8) Per-parameter Slice Comparisons 

Methodology:
- For each parameter slice, we aggregated jet-level features from dark jets only in signal files with that parameter value, and compared to SM jets from `NominalSM.h5`.
- Parameters sliced: `rinv` and `alpha` (parsed from filenames).
- Metrics reported per feature and slice: sample sizes, mean difference (signal − SM), and two-sample KS statistic (shape difference; higher means stronger separation).

Key findings (top slices by KS):
- rinv slices
  - jet_pt: rinv = 0.8 → KS = 0.574, Δmean = −381.856, signal count ≈ 21,287
  - jet_m:  rinv = 0.8 → KS = 0.587, Δmean = −23.436,  signal count ≈ 21,287
  - jet_e:  rinv = 0.8 → KS = 0.600, Δmean = −634.835, signal count ≈ 21,287
  - jet_eta/phi: very small KS (~0.02 or lower)
- alpha slices
  - jet_pt: alpha = high → KS = 0.292, Δmean = −199.045, signal count ≈ 26,469
  - jet_m:  alpha = low  → KS = 0.376, Δmean = −15.733,  signal count ≈ 21,860
  - jet_e:  alpha = high → KS = 0.326, Δmean = −344.764, signal count ≈ 26,469
  - jet_eta/phi: very small KS (~0.03 or lower)

Interpretation notes:
- Slices with larger KS indicate stronger distributional separation between Dark and SM for that feature.
- Energy- and momentum-like features (jet_pt, jet_e, jet_m) show the strongest separations, particularly at higher rinv and for alpha=high/low.
- Angular features (eta, phi) show minimal separation, consistent with earlier aggregate results.

Artifacts and tables:
- Per-slice histograms saved under `outputs/param_slices/` in subfolders per parameter and value, e.g. `outputs/param_slices/rinv/0.8/`.
- TSV summaries:
  - `outputs/param_slices/summary_rinv.tsv`
  - `outputs/param_slices/summary_alpha.tsv`

--
