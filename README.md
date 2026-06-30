# Cross Mouse V1 RSA

Code and analysis notebook for *"Mouse V1 Passes Through a Transient Low-Dimensional
State of Peak Cross-Individual Representational Consistency During Natural Image
Viewing"*.

## Summary

Using representational similarity analysis (RSA) on Neuropixels recordings from all
32 `brain_observatory_1.1` sessions in the Allen Brain Observatory, this project shows
that cross-mouse agreement on V1 population responses to natural images is not constant
over time. It rises to a transient peak at ~100 ms post-stimulus, coinciding with a
collapse in the dimensionality of the shared representation, before decaying to a lower
sustained floor. This transient is preferentially produced by fast-spiking (putative
PV+) interneurons, is absent for static gratings, and is reproduced by a Wilson-Cowan
excitatory/inhibitory model that generalizes to held-out mice in cross-validation.

## Contents

- `cross_mouse_v1_rsa.ipynb` — full analysis pipeline, runnable in
  Google Colab. Organized into 8 parts:
  1. Data loading, QC, and within-mouse reliability (noise ceiling)
  2. Core cross-mouse consistency and image-level predictors (sparsity, CNN features)
  3. Sliding-window temporal profile of consistency
  4. Representational dimensionality (participation ratio) over time
  5. Cell-type dissociation: fast-spiking vs. regular-spiking units, optotagging validation
  6. Wilson-Cowan E/I circuit model and cross-validation
  7. Controls: adaptation, behavioral state (pupil, running)
  8. Static gratings vs. natural scenes comparison

  Supplementary analyses (image-ranking stability, second-order RSA, cross-area
  hierarchy, attractor-space image galleries) are included at the end of the notebook.

## Data

All data are from the [Allen Brain Observatory Visual Coding Neuropixels dataset](https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels),
accessed via [AllenSDK](https://allensdk.readthedocs.io/). The notebook downloads data
automatically into a local cache on first run (~tens of GB; a fast connection and
patience are recommended). No data files are stored in this repository.

## Requirements

The notebook is self-contained and installs its own dependencies (AllenSDK, PyTorch/
torchvision, scikit-learn, etc.) in the first cell. It is designed to run in Google
Colab; a local Jupyter environment will also work given sufficient disk space for the
Allen SDK cache (~50+ GB recommended for all 32 sessions).

**If running on Google Colab, you must restart the notebook after running the first cell.**

Core dependencies: `allensdk`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`,
`torchvision`, `h5py`, `matplotlib`, `seaborn`.

## Reproducibility

- Global random seed fixed at `SEED = 42` and passed explicitly to all stochastic
  operations (bootstraps, cross-validation splits, model fit initializations).
- All quality-control thresholds, window definitions, and statistical tests are stated
  inline in the notebook as methodological notes alongside each analysis cell.
- Figures are saved as `.pdf` files matching those in the manuscript.
