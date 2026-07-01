# Archive Contents — `preprocessed_data.zip`

Reference documentation for every file in the preprocessed data archive.
All files are produced by `preprocess.ipynb` and consumed by the nine
part-notebooks. No part-notebook requires AllenSDK, pynwb, or NWB file access.

---

## `manifest.json`

Single source of truth for all parameters used during preprocessing.
Every part-notebook should load this first and derive its constants from it
rather than re-declaring them.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `bin_width_ms` | int | Width of each time bin: `5` ms |
| `bin_edges_ms` | list[float] | 96 edges defining 95 bins, spanning −75 to +400 ms relative to stimulus onset |
| `n_images` | int | Number of natural-scene images: `118` |
| `n_grating_conditions` | int | Number of common grating conditions: `30` |
| `grating_conditions` | list[[float, float]] | The 30 `[orientation, spatial_frequency]` pairs present in every session |
| `areas` | list[str] | `['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']` |
| `selected_session_ids_visp` | list[int] | 32 `brain_observatory_1.1` session IDs (VISp backbone) |
| `area_session_ids` | dict[str, list[int]] | Sessions with ≥10 QC-passing units per area |
| `pv_cre_session_ids` | list[int] | 5 Pvalb-IRES-Cre × Ai32 session IDs |
| `qc_thresholds` | dict | `amplitude_cutoff_max`, `presence_ratio_min`, `isi_violations_max`, `min_units_per_session` |
| `opto_windows_s` | dict | `light: 0.005`, `baseline: 0.005` — pulse response windows in seconds |
| `state_window_s` | [float, float] | `[0.0, 0.25]` — window used for behavioral state extraction |
| `seed` | int | Global random seed: `42` |
| `reconstruction_note` | str | Code snippet showing how to reconstruct any window from a tensor |

**Window reconstruction** (copy this into every part-notebook):
```python
import json, numpy as np, h5py

manifest   = json.load(open('manifest.json'))
bin_edges  = np.array(manifest['bin_edges_ms'])

def window_response(tensor, t_lo_ms, t_hi_ms):
    """Reconstruct a response matrix for any window from a binned tensor."""
    lo = np.searchsorted(bin_edges, t_lo_ms)
    hi = np.searchsorted(bin_edges, t_hi_ms) - 1
    X  = tensor[:, :, lo:hi].sum(axis=2)          # (n_images, n_units)
    return (X - X.mean(0)) / (X.std(0) + 1e-8)    # z-score per unit
```

---

## `responses.h5`

HDF5 file containing all binned spike-count tensors, the VISp trial-level
table, and non-VISp noise ceilings. Organized as nested groups; `h5py`
reads a single dataset without loading the full file.

### Natural-scenes tensors — `natural_scenes/{area}/{session_id}/`

Present for all five areas: VISp (32 sessions), VISl (24), VISal (23),
VISpm (20), VISam (26).

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `tensor` | `(118, n_units, 95)` | float32 | Trial-averaged spike counts per image per unit per 5 ms bin. n_units varies by session (14–110 for VISp). |
| `unit_ids` | `(n_units,)` | int64 | Allen unit IDs corresponding to tensor axis 1, in the same order |
| `image_ids` | `(118,)` | int64 | Natural-scene frame indices (0–117) corresponding to tensor axis 0 |

### VISp trial-level table — `natural_scenes/VISp/{session_id}/`

Additional datasets present only for VISp, needed by the noise ceiling
(Part 1.2) and adaptation control (Part 7.1).

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `trial_level_0_250ms` | `(n_trials, n_units)` | float32 | Raw spike counts per trial in the 0–250 ms window. Trials are in chronological order. n_trials ≈ 5900 (118 images × 50 repetitions). |
| `trial_frame` | `(n_trials,)` | int64 | Natural-scene frame index (0–117) for each trial |
| `trial_start_time` | `(n_trials,)` | float64 | Stimulus onset time in seconds (session clock) for each trial |

### Non-VISp noise ceilings — `natural_scenes/{area}/noise_ceiling`

Present for VISl, VISal, VISpm, VISam only (VISp ceiling is computed by
Part 1 from the trial-level table).

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `noise_ceiling` | `(n_sessions_in_area, 118)` | float64 | Spearman-Brown corrected split-half reliability per session per image. Mean across sessions gives the noise ceiling used to normalise consistency. |

Row order matches `manifest['area_session_ids'][area]`.

### Static-gratings tensors — `static_gratings/VISp/{session_id}/`

Present for all 32 VISp sessions.

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `tensor` | `(30, n_units, 95)` | float32 | Trial-averaged spike counts per grating condition per unit per 5 ms bin |
| `unit_ids` | `(n_units,)` | int64 | Allen unit IDs corresponding to tensor axis 1 |

### Grating condition list — `static_gratings/VISp/conditions`

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `conditions` | `(30, 2)` | float64 | `[orientation_deg, spatial_frequency_cpd]` for each of the 30 common conditions. Row order matches tensor axis 0. Same as `manifest['grating_conditions']`. |

---

## `units_meta.parquet`

One row per QC-passing unit across all five areas. The single join table
for all unit-level filtering throughout the project.

**Shape:** 13,562 rows × 9 columns

| Column | Dtype | Description |
|---|---|---|
| `unit_id` | int64 | Allen unit ID (primary key; matches `unit_ids` datasets in `responses.h5`) |
| `ecephys_session_id` | int64 | Session the unit was recorded in |
| `area` | str | Visual area: one of `VISp`, `VISl`, `VISal`, `VISpm`, `VISam` |
| `waveform_duration` | float64 | Trough-to-peak waveform duration in ms. Units with duration < 0.4 ms are putative FSU; ≥ 0.4 ms are putative RSU. |
| `amplitude_cutoff` | float64 | QC field; all rows satisfy `< 0.1` |
| `presence_ratio` | float64 | QC field; all rows satisfy `> 0.9` |
| `isi_violations` | float64 | QC field; all rows satisfy `< 0.5` |
| `opto_light_mean` | float64 | Mean spike count in the 0–5 ms post-pulse window, averaged across all light pulses. Non-null only for units in PV-Cre sessions. NaN for all other units. |
| `opto_baseline_mean` | float64 | Mean spike count in the 5 ms pre-pulse baseline window. Non-null only for units in PV-Cre sessions. NaN for all other units. |

**Optotagging classification** (left to Part 5):
```python
# A unit is optotagged PV+ if both criteria are met:
is_opto = (units_meta['opto_light_mean'] > 0.05) & \
           (units_meta['opto_light_mean'] /
            (units_meta['opto_baseline_mean'] + 1e-6) > 2.0)
```

---

## `noise_ceiling_gratings_VISp.npy`

**Shape:** `(32, 30)` — float64

Spearman-Brown corrected split-half reliability for static gratings,
one row per VISp session, one column per grating condition.

Row order matches `manifest['selected_session_ids_visp']`.
Column order matches `manifest['grating_conditions']` and
`static_gratings/VISp/conditions` in `responses.h5`.

Mean across sessions (axis 0) gives the per-condition noise ceiling.
Mean across conditions gives the scalar noise ceiling used to normalise
grating consistency in Part 8.

Expected values: mean ≈ 0.992, range [0.93, 1.0].

---

## `cnn_embeddings.npz`

ResNet50 (ImageNet pretrained, `IMAGENET1K_V2`) feature embeddings for
all 118 natural-scene images. Images were converted to 3-channel grayscale,
resized to 224×224, and normalised with ImageNet mean/std before forward pass.
Global average pooling was applied to each convolutional block output.

**Arrays:**

| Key | Shape | Description |
|---|---|---|
| `layer1` | `(118, 256)` | Output of ResNet50 layer1 block, global average pooled |
| `layer2` | `(118, 512)` | Output of ResNet50 layer2 block, global average pooled |
| `layer3` | `(118, 1024)` | Output of ResNet50 layer3 block, global average pooled |
| `layer4` | `(118, 2048)` | Output of ResNet50 layer4 block, global average pooled |
| `final` | `(118, 1000)` | Final classification layer output (logits) |

Row order is image index 0–117, matching `manifest['n_images']` and tensor axis 0.

Layer2 is the best single-layer predictor of cross-mouse consistency
(LOO Pearson r = −0.496) and is used as the primary CNN predictor in Part 2.

---

## `image_templates.npy`

**Shape:** `(118, 256, 256)` — uint8

Downsampled grayscale natural-scene images for use in gallery figures only.
Images are in the same order as tensor axis 0 (image index 0–117).

**Not used for any quantitative analysis.** CNN embeddings were computed
from the full-resolution originals before downsampling. All neural data
analyses use the response tensors in `responses.h5`.

---

## `behavioral_state.parquet`

Per-presentation pupil area and running speed during natural-scenes viewing,
extracted over the 0–250 ms stimulus window.

**Shape:** 188,763 rows × 4 columns (one row per trial × session)

| Column | Dtype | Description |
|---|---|---|
| `session_id` | int64 | Session the presentation belongs to |
| `image_id` | int64 | Natural-scene frame index (0–117) |
| `pupil_area` | float64 | Mean pupil area in the 0–250 ms window (arbitrary units from `filtered_gaze_mapping`). NaN for the 6 sessions without eye tracking (session IDs: 732592105, 715093703, 719161530, 721123822, 737581020, 739448407). |
| `running_speed` | float64 | Mean running speed in the 0–250 ms window (cm/s). Non-null for all 32 sessions. |

**Usage in Part 7** (behavioral state control):
```python
# Coefficient of variation across mice per image
state_div = df_state.groupby('image_id').agg(
    pupil_cv   = ('pupil_area',    lambda x: x.std() / (x.mean() + 1e-8)),
    running_cv = ('running_speed', lambda x: x.std() / (abs(x.mean()) + 1e-8)),
)
```

Pupil area values are in the unit system of the Allen `filtered_gaze_mapping`
pipeline (approximately 0.001–0.007 for typical pupil sizes). Absolute units
do not matter for the CV analysis since only relative variation across mice
per image is used.
