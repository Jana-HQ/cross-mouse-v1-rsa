# Cross-Mouse V1 Representational Consistency — Analysis Repository

Neuropixels analysis of Allen Brain Observatory `brain_observatory_1.1` data.
32 mice, VISp (+ 4 higher areas), 118 natural-scene images.

Core finding: V1 passes through a transient low-dimensional state ~100 ms
post-stimulus where cross-mouse representational consistency peaks (Spearman
r = 0.306), coinciding with a participation-ratio collapse (108 → 31.6
dimensions). FSU/PV+ interneurons drive the transient (T/S ratio: FSU 2.1×,
RSU 1.6×, optotagged PV+ 3.57×). A Wilson-Cowan E/I model fit gives
r_EI = 0.612 (inhibition-dominated), cross-validated R² = 0.843.

---

## Repository structure

```
preprocessed_data.zip          ← download this first (see below)
preprocessed_data/             ← auto-extracted by any notebook on first run
│
notebooks/
│   utils.py                   ← shared functions (imported by all notebooks)
│   requirements.txt           ← pip-installable dependencies
│
│   part1_data_and_consistency.ipynb
│   part2_image_predictors.ipynb
│   part3_temporal_profile.ipynb
│   part4_representational_geometry.ipynb
│   part5_cell_type_dissociation.ipynb
│   part6_ei_model.ipynb
│   part7_controls.ipynb
│   part8_gratings_vs_natural_scenes.ipynb
│   part9_cross_area_hierarchy.ipynb
│
│   part1_outputs.npz          ← written by Part 1, read by Parts 2, 7, 8, 9
│   part3_outputs.npz          ← written by Part 3, read by Parts 4, 5, 8, 9
│   part5_outputs.npz          ← written by Part 5, read by Part 6
│
figures/                       ← all figure PDFs saved here (auto-created)
```

> **Note — inter-notebook outputs:** `part1_outputs.npz`, `part3_outputs.npz`,
> and `part5_outputs.npz` are written to the `notebooks/` directory by their
> respective notebooks and read from the same location by downstream notebooks.
> Run the parts in order (1 → 3 → 5 → others) on first use.

---

## Quick start

```bash
# 1. Clone / download the repository
# 2. Place preprocessed_data.zip at the repository root (same level as notebooks/)

# 3. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r notebooks/requirements.txt

# 4. Launch Jupyter and open any notebook
cd notebooks/
jupyter notebook part1_data_and_consistency.ipynb
```

The first cell of every notebook calls `setup_archive()`, which will:
- Check that `../preprocessed_data.zip` exists (raises a clear error if not).
- Extract it to `../preprocessed_data/` if not already done.
- Create `../figures/` if absent.
- Return `ARCHIVE_DIR`, `FIGURES_DIR`, and `OUTPUTS_DIR` path objects used
  throughout the rest of the notebook.

---

## Preprocessed data archive

The archive `preprocessed_data.zip` is produced by the companion preprocessing
notebook (`download_preprocessed_data_cross_mouse_rsa_v1.ipynb`, included in
this repository), which requires AllenSDK and a one-time download of the NWB
files. Once generated it contains everything needed to reproduce all figures.

See `ARCHIVE_CONTENTS.md` and `PREPROCESSING_STRATEGIES.md` (in the archive
root after extraction) for full documentation of every file.

| File | Description | Used by |
|---|---|---|
| `responses.h5` | Binned spike-count tensors (all areas, nat. scenes + gratings), VISp trial-level table, non-VISp noise ceilings | Parts 1, 3–5, 7–9 |
| `units_meta.parquet` | Per-unit QC fields, waveform duration, opto pulse counts | Parts 5, 9 |
| `cnn_embeddings.npz` | ResNet50 layer1–4 + final embeddings, 118 images | Part 2 |
| `image_templates.npy` | 118 × 256 × 256 uint8 display images (gallery use only) | Parts 1, 4 |
| `behavioral_state.parquet` | Pupil area + running speed per natural-scenes trial | Part 7 |
| `noise_ceiling_gratings_VISp.npy` | Split-half NC per session × condition, VISp gratings | Part 8 |
| `manifest.json` | Bin edges, session IDs, QC thresholds, reconstruction formula | All parts |

---

## Notebook summary

| Notebook | Main figure | Supplementary | Key inputs | Outputs saved |
|---|---|---|---|---|
| **Part 1** — Data Quality & Core Consistency | Fig. 1 | S1 | `responses.h5` (trial_level), `image_templates.npy` | `part1_outputs.npz` |
| **Part 2** — Image-Level Predictors | Fig. 2 | S2 | `part1_outputs.npz`, `cnn_embeddings.npz` | — |
| **Part 3** — Temporal Profile | Fig. 3 | S3, S7, S8 | `responses.h5` (VISp tensors) | `part3_outputs.npz` |
| **Part 4** — Representational Geometry | Fig. 4 | S5, S6 | `responses.h5`, `part3_outputs.npz`, `image_templates.npy` | — |
| **Part 5** — Cell-Type Dissociation | Fig. 5 | S4 | `responses.h5`, `units_meta.parquet`, `part3_outputs.npz` | `part5_outputs.npz` |
| **Part 6** — E/I Circuit Model | Fig. 6 | S9 | `part5_outputs.npz` | — |
| **Part 7** — Controls | Fig. 7 | S10 | `responses.h5` (trial_level), `behavioral_state.parquet`, `part1_outputs.npz` | — |
| **Part 8** — Gratings vs. Natural Scenes | Fig. 8 | — | `responses.h5` (gratings), `noise_ceiling_gratings_VISp.npy`, `part3_outputs.npz`, `part1_outputs.npz` | — |
| **Part 9** — Cross-Area Hierarchy | Fig. 9 | — | `responses.h5` (all areas), `part1_outputs.npz`, `part3_outputs.npz` | — |

**Recommended run order (for first use):** 1 → 3 → 5 → {2, 4, 6, 7, 8, 9} in any order.

---

## Shared utilities (`utils.py`)

All notebooks import from `utils.py` (in the same `notebooks/` directory).

| Function | Description |
|---|---|
| `setup_archive(...)` | Check/extract archive, load manifest, return path objects |
| `window_response(tensor, t_lo_ms, t_hi_ms)` | Reconstruct z-scored response matrix from any time window |
| `compute_similarity_matrices(response_matrices)` | Cosine similarity matrix per session |
| `per_image_consistency(sim_matrices, n_images, mouse_ids)` | Mean Spearman r across mouse pairs per image; n_images inferred from matrix shape if omitted |
| `cluster_bootstrap_ci(response_matrices, n_boot, ci, seed)` | Cluster bootstrap CI resampling whole mice |
| `apply_plot_style()` | Set project-wide rcParams; returns `{'COL_NS', 'COL_GR', 'COL_FIT', 'COL_REF'}` |
| `ts_ratio(mc_arr, times_arr, sustained_lo, sustained_hi)` | Transient-to-sustained ratio from a consistency curve |

---

## Key numerical results (expected values)

| Quantity | Value |
|---|---|
| Peak cross-mouse consistency (0–250 ms mean) | Spearman r = 0.306 |
| Peak time | ~100 ms post-stimulus |
| Participation ratio: baseline → peak | 108 → 31.6 |
| FSU T/S ratio | 2.1× |
| RSU T/S ratio | 1.6× |
| Optotagged PV+ T/S ratio | 3.57× |
| E/I model r_EI | 0.612 (inhibition-dominated) |
| E/I model cross-validated R² | 0.843 |
| VISp noise ceiling (mean) | ~0.978 |

---

## Colour palette (consistent across all figures)

| Variable | Hex | Use |
|---|---|---|
| `COL_NS = 'steelblue'` | #4682B4 | Natural scenes / RSU / excitatory / primary signal |
| `COL_GR = 'darkorange'` | #FF8C00 | Gratings / FSU / inhibitory / secondary signal |
| `COL_FIT = 'firebrick'` | #B22222 | Model fits / peak annotations / genetic validation |
| `COL_REF = 'dimgray'` | #696969 | Reference lines / baselines / all-units curves |
