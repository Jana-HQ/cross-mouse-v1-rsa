# Cross-Mouse V1 Representational Consistency — Analysis Repository

Neuropixels analysis of Allen Brain Observatory `brain_observatory_1.1` data.
32 mice, VISp (+ 4 higher areas), 118 natural-scene images.

Core finding: V1 passes through a transient low-dimensional state ~100 ms post-stimulus where cross-mouse representational consistency peaks, coinciding with a participation-ratio collapse.

---

## Repository structure

```
preprocessed_data.zip
preprocessed_data/             ← auto-extracted by any notebook on first run
|
docs/
|   preprocessing_readme.md    ← documentation for preprocessing notebook
|   manuscript/                ← manuscript pdf and LaTeX source   
│
src/
│   utils.py                   ← shared functions (imported by all notebooks)
|   preprocessing_cross_mouse_v1_rsa.ipynb
│   requirements.txt           ← pip-installable dependencies
│
│   part1_data_and_consistency.ipynb
│   part2_image_predictors.ipynb
│   part3_temporal_profile.ipynb
│   part4_representational_geometry.ipynb
│   part5_cell_type_dissociation.ipynb
│   part6_pr_shrinkage_model.ipynb
│   part7_controls.ipynb
│   part8_gratings_vs_natural_scenes.ipynb
│   part9_cross_area_hierarchy.ipynb
│
│   part1_outputs.npz          ← written by Part 1, read by Parts 2, 7, 8, 9
│   part3_outputs.npz          ← written by Part 3, read by Parts 4, 5, 6, 8, 9
│   part5_outputs.npz          ← written by Part 5, read by Part 6
│
figures/                       ← all figure PNGs saved here (auto-created)
```

> **Note — inter-notebook outputs:** `part1_outputs.npz`, `part3_outputs.npz`,
> and `part5_outputs.npz` are written to the `notebooks/` directory by their
> respective notebooks and read from the same location by downstream notebooks.
> Run the parts in order (1 → 3 → 5 → others) on first use.

---

## Preprocessed data archive

The archive `preprocessed_data.zip` is produced by the companion preprocessing
notebook (`src\preprocessing_cross_mouse_v1_rsa.ipynb`, included in
this repository), which requires AllenSDK and a one-time download of the NWB
files. Once generated it contains everything needed to reproduce all figures.

See `docs\preprocessing_readme.md` for full documentation.

---

## Notebook summary

| Notebook | Main figure | Supplementary | Key inputs | Outputs saved |
|---|---|---|---|---|
| **Part 1** — Data Quality & Core Consistency | Fig. 1 | S1 | `responses.h5` (trial_level), `image_templates.npy` | `part1_outputs.npz` |
| **Part 2** — Image-Level Predictors | Fig. 2 | S2 | `part1_outputs.npz`, `cnn_embeddings.npz` | — |
| **Part 3** — Temporal Profile | Fig. 3 | S3, S7, S8 | `responses.h5` (VISp tensors) | `part3_outputs.npz` |
| **Part 4** — Representational Geometry | Fig. 4 | S5, S6 | `responses.h5`, `part3_outputs.npz`, `image_templates.npy` | — |
| **Part 5** — Cell-Type Dissociation | Fig. 5 | S4 | `responses.h5`, `units_meta.parquet`, `part3_outputs.npz` | `part5_outputs.npz` |
| **Part 6** — Population Geometry: Cross-Validated PR Shrinkage Model | Fig. 6 | S9 | `responses.h5` (VISp tensors), `part3_outputs.npz`, `part5_outputs.npz` | — |
| **Part 7** — Controls | Fig. 7 | S10 | `responses.h5` (trial_level), `behavioral_state.parquet`, `part1_outputs.npz` | — |
| **Part 8** — Gratings vs. Natural Scenes | Fig. 8 | — | `responses.h5` (gratings), `noise_ceiling_gratings_VISp.npy`, `part3_outputs.npz`, `part1_outputs.npz` | — |
| **Part 9** — Cross-Area Hierarchy | Fig. 9 | — | `responses.h5` (all areas), `part1_outputs.npz`, `part3_outputs.npz` | — |

**Recommended run order (for first use):** 1 → 3 → 5 → {2, 4, 6, 7, 8, 9} in any order.

---
