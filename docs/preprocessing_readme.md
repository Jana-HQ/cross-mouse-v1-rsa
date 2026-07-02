# Preprocessing — Strategy & Archive Reference

Documentation for `download_preprocessed_data_cross_mouse_rsa_v1.ipynb`.
Run once. All part-notebooks depend only on the archive this produces —
no AllenSDK, no NWB access, no network calls.

---

## Binning strategy

Spikes are binned into **1 ms bins** spanning −75 to +405 ms relative to
each trial's stimulus onset, producing a `(n_images, n_units, 480)` tensor
per session per area/stimulus. Any analysis window is reconstructed by
summing the relevant bins.

**Why 1 ms.** Window boundaries like [87.5, 112.5] ms map to the nearest
bin edges. At 1 ms resolution the error is ±0.5 ms per edge, negligible for
any window width used in this project.

**Per-trial spike extraction.** Spikes are extracted independently for each
trial, not assigned globally:

```python
def bin_spikes_per_trial(spike_times, trial_starts, bin_edges_ms):
    n_trials, n_bins = len(trial_starts), len(bin_edges_ms) - 1
    counts = np.zeros((n_trials, n_bins), dtype=np.int32)
    dt_lo, dt_hi = bin_edges_ms[0] / 1000.0, bin_edges_ms[-1] / 1000.0
    for i, t0 in enumerate(trial_starts):
        lo = np.searchsorted(spike_times, t0 + dt_lo, side='left')
        hi = np.searchsorted(spike_times, t0 + dt_hi, side='left')
        if lo >= hi:
            continue
        rel_ms = (spike_times[lo:hi] - t0) * 1000.0
        bidx   = np.clip(np.digitize(rel_ms, bin_edges_ms) - 1, 0, n_bins - 1)
        np.add.at(counts[i], bidx, 1)
    return counts
```

This is required because natural scenes are presented with no ISI (250 ms
image, next image immediately following). A global spike-to-trial assignment
via `searchsorted` on raw onset times would place any spike after +250 ms
into the next trial, leaving post-offset bins empty. The per-trial loop asks
"which spikes fell in [onset + t_lo, onset + t_hi]?" for each trial
independently, correctly capturing the full −75 to +405 ms range regardless
of adjacent trial boundaries.

**Tensors store trial-averaged responses.** Only two analyses need
single-trial resolution — the noise ceiling (Part 1) and the adaptation
control (Part 7) — and both only need the fixed 0–250 ms window. These are
served by the `trial_level_0_250ms` table. All sliding-window and
cell-type analyses use the trial-averaged tensor.

**Window reconstruction** (implemented in `utils.py`):
```python
lo = np.searchsorted(bin_edges_ms, t_lo_ms)
hi = np.searchsorted(bin_edges_ms, t_hi_ms)
X  = tensor[:, :, lo:hi].sum(axis=2)        # (n_images, n_units)
X  = (X - X.mean(0)) / (X.std(0) + 1e-8)   # z-score per unit
```

---

## Archive contents

### `manifest.json`

Read this first in every part-notebook. All constants should be derived from
it rather than re-declared.

| Field | Description |
|---|---|
| `bin_width_ms` | `1` |
| `bin_edges_ms` | 481 edges defining 480 bins, −75 to +405 ms |
| `n_images` | `118` |
| `n_grating_conditions` | `30` |
| `grating_conditions` | List of `[orientation_deg, sf_cpd]` for each of the 30 common conditions |
| `areas` | `['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']` |
| `selected_session_ids_visp` | 32 session IDs (VISp backbone) |
| `area_session_ids` | Sessions with ≥10 QC-passing units per area |
| `pv_cre_session_ids` | 5 Pvalb-IRES-Cre × Ai32 session IDs |
| `qc_thresholds` | `amplitude_cutoff_max=0.1`, `presence_ratio_min=0.9`, `isi_violations_max=0.5`, `min_units=10` |
| `opto_windows_s` | `light: 0.005`, `baseline: 0.005` |
| `state_window_s` | `[0.0, 0.25]` |
| `seed` | `42` |

---

### `responses.h5`

All binned tensors, the VISp trial-level table, and non-VISp noise ceilings.
Organized as nested HDF5 groups; any single dataset can be read without
loading the full file.

#### Natural-scenes tensors — `natural_scenes/{area}/{session_id}/`

Five areas: VISp (32 sessions), VISl (24), VISal (23), VISpm (20), VISam (26).

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `tensor` | `(118, n_units, 480)` | float32 | Trial-averaged spike counts. n_units: 14–110. |
| `unit_ids` | `(n_units,)` | int64 | Allen unit IDs, same order as tensor axis 1 |
| `image_ids` | `(118,)` | int64 | Frame indices 0–117, same order as tensor axis 0 |

#### VISp trial-level table — `natural_scenes/VISp/{session_id}/`

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `trial_level_0_250ms` | `(n_trials, n_units)` | float32 | Raw spike counts per trial, 0–250 ms. n_trials ≈ 5900. |
| `trial_frame` | `(n_trials,)` | int64 | Frame index (0–117) per trial |
| `trial_start_time` | `(n_trials,)` | float64 | Stimulus onset time in seconds (session clock) |

#### Non-VISp noise ceilings — `natural_scenes/{area}/noise_ceiling`

VISl, VISal, VISpm, VISam only. VISp ceiling is computed by Part 1 from
the trial-level table.

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `noise_ceiling` | `(n_sessions, 118)` | float64 | Spearman-Brown corrected split-half reliability per session per image. Row order matches `manifest['area_session_ids'][area]`. |

#### Static-gratings tensors — `static_gratings/VISp/{session_id}/`

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `tensor` | `(30, n_units, 480)` | float32 | Trial-averaged spike counts per grating condition |
| `unit_ids` | `(n_units,)` | int64 | Allen unit IDs |
| `conditions` | `(30, 2)` | float64 | `[orientation_deg, sf_cpd]` per condition (group-level dataset) |

---

### `units_meta.parquet`

One row per QC-passing unit across all five areas. **Shape:** 13,562 × 9.

| Column | Dtype | Description |
|---|---|---|
| `unit_id` | int64 | Allen unit ID — matches `unit_ids` datasets in `responses.h5` |
| `ecephys_session_id` | int64 | Session |
| `area` | str | `VISp`, `VISl`, `VISal`, `VISpm`, or `VISam` |
| `waveform_duration` | float64 | Trough-to-peak duration in ms. < 0.4 ms → FSU; ≥ 0.4 ms → RSU |
| `amplitude_cutoff` | float64 | QC field; all rows < 0.1 |
| `presence_ratio` | float64 | QC field; all rows > 0.9 |
| `isi_violations` | float64 | QC field; all rows < 0.5 |
| `opto_light_mean` | float64 | Mean spikes in 0–5 ms post-pulse window. Non-null for PV-Cre sessions only. |
| `opto_baseline_mean` | float64 | Mean spikes in 5 ms pre-pulse baseline. Non-null for PV-Cre sessions only. |

Optotagging classification is applied in Part 5:
```python
is_opto = (units_meta['opto_light_mean'] > 0.05) & \
           (units_meta['opto_light_mean'] /
            (units_meta['opto_baseline_mean'] + 1e-6) > 2.0)
```

---

### `noise_ceiling_gratings_VISp.npy`

**Shape:** `(32, 30)` float64. Spearman-Brown corrected split-half
reliability for static gratings. Row order matches
`manifest['selected_session_ids_visp']`; column order matches
`manifest['grating_conditions']`. Expected mean ≈ 0.992.

---

### `cnn_embeddings.npz`

ResNet50 (`IMAGENET1K_V2`) embeddings for all 118 natural-scene images.
Global average pooled per convolutional block.

| Key | Shape | Description |
|---|---|---|
| `layer1` | `(118, 256)` | ResNet50 layer1 |
| `layer2` | `(118, 512)` | ResNet50 layer2 — primary CNN predictor in Part 2 |
| `layer3` | `(118, 1024)` | ResNet50 layer3 |
| `layer4` | `(118, 2048)` | ResNet50 layer4 |
| `final` | `(118, 1000)` | Classification logits |

Row order is image index 0–117, matching tensor axis 0.

---

### `image_templates.npy`

**Shape:** `(118, 256, 256)` uint8. Downsampled natural-scene images for
gallery figures (Parts 1, 4). Not used for any quantitative analysis.

---

### `behavioral_state.parquet`

Per-presentation behavioral state during natural-scenes viewing (0–250 ms
window). **Shape:** 188,763 rows × 4 columns.

| Column | Dtype | Description |
|---|---|---|
| `session_id` | int64 | |
| `image_id` | int64 | Frame index 0–117 |
| `pupil_area` | float64 | Mean pupil area (filtered_gaze_mapping units). NaN for sessions without eye tracking: 732592105, 715093703, 719161530, 721123822, 737581020, 739448407. |
| `running_speed` | float64 | Mean running speed in cm/s. Non-null for all 32 sessions. |

---

## NWB paths

- Spike times: `units/spike_times`, `units/spike_times_index`, `units/id`
- Pupil: `processing/filtered_gaze_mapping/pupil_area/data` and `.../timestamps`
- Opto pulses: `processing/optotagging/optogenetic_stimulation/start_time` and `.../stimulus_name` (keep `stimulus_name == 'pulse'` only)
- Running speed: AllenSDK `session.running_speed`

Opto stimulus names require per-element decoding (`s.decode('utf-8') if isinstance(s, bytes) else str(s)`) — the dataset dtype is `object`, not fixed-length bytes.