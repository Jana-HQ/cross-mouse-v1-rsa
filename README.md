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


---


Splitting notebooks:

# Preprocessed data — Mouse V1 Cross-Mouse Consistency

## Contents
- `responses.h5` — binned spike-count tensors and the VISp trial-level table.
    - `natural_scenes/{area}/{session_id}/tensor`   shape (n_images, n_units, n_bins)
    - `natural_scenes/{area}/{session_id}/unit_ids`
    - `natural_scenes/{area}/{session_id}/image_ids`
    - `natural_scenes/VISp/{session_id}/trial_level_0_250ms`  shape (n_trials, n_units)
    - `natural_scenes/VISp/{session_id}/trial_frame`, `trial_start_time`
    - `static_gratings/VISp/{session_id}/tensor`     shape (n_conditions, n_units, n_bins)
    - `static_gratings/VISp/conditions`              (n_conditions, 2) = (orientation, spatial_freq)
- `units_meta.parquet` — one row per QC-passing unit across all 5 areas: session_id,
  unit_id, area, waveform_duration, QC fields, and (where applicable)
  `opto_light_mean` / `opto_baseline_mean` for PV-Cre sessions.
- `cnn_embeddings.npz` — ResNet50 (ImageNet) layer1–4 and final-layer embeddings,
  one row per natural-scene image, in image-id order.
- `image_templates.npy` — (118, 256, 256) uint8 grayscale, downsampled, for gallery
  figures only. NOT used for any quantitative analysis.
- `behavioral_state.parquet` — pupil area / running speed per natural-scenes
  presentation, 0–250 ms window.
- `manifest.json` — bin edges, QC thresholds, session lists per area, grating
  condition list, and the window-reconstruction formula.

## Reconstructing any window
Every analysis window used anywhere in the paper (25 ms sliding windows, the
fixed 0–250 ms window, gratings windows) is recoverable by summing contiguous
bins from the tensor and z-scoring per unit:

```python
import numpy as np, json
manifest = json.load(open('manifest.json'))
bin_edges = np.array(manifest['bin_edges_ms'])

def window_response(tensor, t_lo_ms, t_hi_ms):
    lo = np.searchsorted(bin_edges, t_lo_ms)
    hi = np.searchsorted(bin_edges, t_hi_ms) - 1
    X = tensor[:, :, lo:hi].sum(axis=2)          # (n_images, n_units)
    return (X - X.mean(0)) / (X.std(0) + 1e-8)   # z-score per unit
```

## Which part-notebook needs what
| Part | Reads |
|---|---|
| 1 | `natural_scenes/VISp/*` tensor + trial_level table, `image_templates.npy` |
| 2 | `natural_scenes/VISp/*` tensor, `cnn_embeddings.npz` |
| 3 | `natural_scenes/VISp/*` tensor |
| 4 | `natural_scenes/VISp/*` tensor |
| 5 | `natural_scenes/VISp/*` tensor, `units_meta.parquet` (waveform + opto cols) |
| 6 | RSU/FSU curves cached by Part 5 (not produced here) |
| 7 | `natural_scenes/VISp/*` trial_level table, `behavioral_state.parquet` |
| 8 | `static_gratings/VISp/*` tensor |
| 9 | `natural_scenes/{VISl,VISal,VISpm,VISam}/*` tensors |
