# Preprocessing Strategy Notes

This document explains the *design decisions* behind `preprocess.ipynb` — not what
the code does line by line, but why it's structured the way it is. Read this before
modifying the preprocessing notebook or adding a new part-notebook that needs a raw
quantity not currently cached.

## Archive contents summary

| File | Description | Primary consumers |
|---|---|---|
| `responses.h5` | Binned tensors (all areas/stimuli), VISp trial-level table, non-VISp noise ceilings | Parts 1–5, 7–9 |
| `units_meta.parquet` | Per-unit QC fields, waveform duration, opto pulse counts | Parts 5, 9 |
| `cnn_embeddings.npz` | ResNet50 layer1–4 + final embeddings, 118 images | Part 2 |
| `image_templates.npy` | 118 × 256 × 256 uint8 gallery images | Parts 1, 4 (display only) |
| `behavioral_state.parquet` | Pupil area + running speed per natural-scenes presentation | Part 7 |
| `noise_ceiling_gratings_VISp.npy` | Split-half NC per session × condition, VISp gratings | Part 8 |
| `manifest.json` | Bin edges, QC thresholds, session lists, reconstruction formula | All parts |

## 1. The core problem this notebook solves

The original single-notebook analysis re-scanned raw NWB spike times independently
for nearly every section: once for the fixed 0–250 ms response matrix, again for the
noise ceiling (single-trial resolution), again for the sliding-window profile (38
windows), again for RSU/FSU subsets, again for gratings, again for each of four
additional visual areas, again for the adaptation control, and again for optotagging.
Splitting that single notebook into nine part-notebooks would multiply this cost by
nine unless the raw extraction is factored out and done exactly once, in one place,
producing objects that are pure-math reconstructible into anything any part needs.

That's the one job of this notebook: touch every spike exactly once, in a
resolution fine enough to reconstruct every window used anywhere in the paper, then
discard all NWB/AllenSDK access entirely. Every part-notebook's only data dependency
is array slicing, dataframe filtering, or reading a small derived artifact — never
AllenSDK, never h5py spike-time access, never NWB files.

## 2. Strategy: bin once, fine-grained, reconstruct windows by summing

**Decision.** Every session's spikes are digitized into fixed 5 ms bins spanning
−75 ms to +405 ms relative to stimulus onset, producing a
`(n_images, n_units, n_bins)` tensor per session per area/stimulus.

**Why 5 ms.** It's the largest bin width that still lets every window used in the
paper be reconstructed as a sum of *contiguous* bins with edges landing exactly on
bin boundaries:
- 25 ms sliding windows, 10 ms step (centers −50…+370 ms) → 5 bins summed, window
  edges always multiples of 5 ms given the step size
- the fixed 0–250 ms window → 50 bins summed
- gratings windows → identical convention, same bin edges

Going coarser (e.g. 10 ms) would not exactly reconstruct the 10 ms step size used
by the sliding-window analysis. Going finer buys nothing, since no analysis in the
paper resolves below 10 ms, and it would needlessly inflate storage.

**Why padding beyond the widest window.** Bin edges run from −75 to +405 ms, wider
than the actual analysis range (−50 to +370 ms), so that windows centered at the
extreme ends of the sliding-window range (e.g. center = −50 ms, half-width 12.5 ms)
have their edges fall inside the binned range rather than at its boundary, avoiding
edge-truncation artifacts.

**Reconstruction formula** (used identically by every part-notebook):
```python
lo = np.searchsorted(bin_edges_ms, t_lo_ms)
hi = np.searchsorted(bin_edges_ms, t_hi_ms) - 1
response_matrix = tensor[:, :, lo:hi].sum(axis=2)   # (n_images, n_units)
response_matrix = (response_matrix - response_matrix.mean(0)) / (response_matrix.std(0) + 1e-8)
```
This single formula replaces all per-window NWB rescans for: Part 1 (fixed window),
Part 3 (sliding window + two-window comparison), Part 4 (PR/eigenvector windows),
Part 5 (RSU/FSU sliding windows), Part 8 (gratings sliding windows), Part 9
(cross-area sliding windows).

**Trade-off accepted.** The tensor stores trial-*averaged* responses, not
single-trial spike counts. This is the right trade-off because only two analyses
in the entire paper need single-trial resolution (noise ceiling, adaptation
trial-split), and both only need the single fixed 0–250 ms window, not the full
fine-grained range. Storing single-trial data at 5 ms resolution for all ~38
windows × all sessions would be the dominant cost in the whole archive for a need
that affects two subsections.

## 3. Strategy: separate unbinned trial-level table for the two single-trial analyses

**Decision.** A second, much smaller object — `(n_trials, n_units)` raw spike
counts in the single 0–250 ms window, with trial order and image-id preserved —
is stored only for VISp, co-located with the binned tensor inside `responses.h5`
under `natural_scenes/VISp/{session_id}/trial_level_0_250ms`.

**Why not just slice the binned tensor.** The binned tensor only retains the
trial-*averaged* response per image; trial identity and order are gone after
averaging. The noise ceiling (split-half reliability across single trials) and
the adaptation control (chronological early/late trial split) both require this
identity, so they need a fundamentally different object, not a finer view of the
same one.

**Why VISp only.** Both consumers (Parts 1.2 and 7.1) only ever operate on VISp.
Storing this table for all five areas would multiply that cost fivefold for zero use.

## 4. Strategy: noise ceilings are precomputed and cached, not recomputed per part

**Decision.** Non-VISp area noise ceilings are computed here and stored inside
`responses.h5` under `natural_scenes/{area}/noise_ceiling` as `(n_sessions,
n_images)` arrays. The VISp natural-scenes noise ceiling is left to Part 1 (it
is the first result in the paper and is computed from the trial-level table already
cached here). The gratings noise ceiling is stored separately as
`noise_ceiling_gratings_VISp.npy`.

**Why precompute non-VISp ceilings here.** The noise ceiling computation requires
single-trial spike counts at 0–250 ms, which means re-scanning raw NWB files. Part 9
(cross-area analysis) would otherwise need NWB access, breaking the rule that
part-notebooks touch only cached data. Precomputing and caching eliminates that
dependency.

**Why leave VISp natural-scenes ceiling to Part 1.** The VISp trial-level table is
already cached in `responses.h5`, so Part 1 can compute its own noise ceiling with
pure array operations — no NWB access needed. Precomputing it here would just be
duplicating Part 1's analysis logic in the preprocessing notebook, which violates
the principle that analysis lives in the part it belongs to.

## 5. Strategy: one global unit-metadata table instead of per-session metadata files

**Decision.** A single parquet file (`units_meta.parquet`) holds one row per
QC-passing unit across all five areas, with waveform duration, QC fields, and
(where applicable) optotagging pulse-response columns.

**Why a single flat table over per-session/per-area files.** Several
part-notebooks need to filter/group units in ways that cut across the
session/area boundary (e.g. Part 5's RSU/FSU split is a global threshold on
`waveform_duration`; Part 9 needs per-area session lists). A single table with
`session_id`/`area`/`unit_id` columns supports all of these with a `groupby` or
boolean mask rather than requiring each notebook to know which per-file metadata
object to open for which question.

**Why merge optotagging columns into this table rather than a separate file.**
The optotagging classification (Part 5.4) is fundamentally a per-unit property
test (light response vs. baseline), so it belongs on the same per-unit row as
the unit's other identifying properties (waveform duration) — this lets Part 5
join the waveform-based and genetic classifications with a single dataframe,
which is exactly the comparison the paper makes (waveform FSU vs. confirmed PV+).

**Trade-off accepted.** The raw light/baseline pulse counts are stored, not the
final optotagged/not-optotagged label. The classification thresholds (ratio > 2.0,
light count > 0.05) are left to Part 5's notebook. This keeps the preprocessing
notebook threshold-agnostic — if the optotagging criteria are revisited later, no
NWB file needs to be touched again, only the small derived table needs recomputing.

## 6. Strategy: all NWB access goes through h5py directly, not the AllenSDK

**Decision.** Spike times, eye-tracking data, and optogenetic stimulation epochs
are all read by opening NWB files directly with `h5py`, bypassing the AllenSDK
session object entirely for these quantities.

**Why.** The AllenSDK wraps pynwb for data access. Several pynwb methods used by
the SDK — including those underlying `session.get_pupil_data()` and
`session.optogenetic_stimulation_epochs` — depend on internal APIs that have been
removed in current pynwb versions, causing silent failures or exceptions. Direct
h5py reads are immune to these SDK/pynwb version dependencies because they treat
the NWB file as a plain HDF5 file and address datasets by their literal path.

The confirmed NWB paths used are:
- Spike times: `units/spike_times`, `units/spike_times_index`, `units/id`
- Pupil area: `processing/filtered_gaze_mapping/pupil_area/data` and `.../timestamps`
- Opto pulse epochs: `processing/optotagging/optogenetic_stimulation/start_time`
  and `.../stimulus_name` (filtered to `stimulus_name == 'pulse'`, duration 5–10 ms)

Running speed is still read through the SDK (`session.running_speed`) because that
path works correctly under current versions and the running speed data is not
stored at a simple h5py-accessible path in the NWB structure.

**Stimulus name decoding.** NWB string datasets are stored as variable-length
byte objects. The opto stimulus name array requires explicit per-element decoding
(`s.decode('utf-8') if isinstance(s, bytes) else str(s)`) rather than a single
dtype check, because the array dtype is `object` rather than fixed-length `'S'`
bytes — a dtype check would silently fail to decode and produce wrong comparisons.

## 7. Strategy: derived/expensive-to-recompute outputs are cached only where they're consumed, not centrally

**Decision.** CNN embeddings (ResNet50 layers) are computed once here and shipped
as a flat `.npz`, since Part 2 needs them and computing them requires `torch`/
`torchvision` — a heavy dependency intentionally kept out of every other
part-notebook's environment.

**Decision (the other direction).** RSU/FSU consistency *curves* (used by Part 6's
E/I model) are explicitly **not** computed in this notebook, even though the raw
material (the VISp tensor + unit metadata) is already here and the curves are
cheap to derive from it. They're left as an output Part 5's notebook produces for
Part 6 to consume.

**Why the asymmetry.** The criterion is: does producing this object require
re-touching raw NWB/spike data, or does it require a heavy dependency unrelated to
any other part? CNN embeddings meet both — they need the raw images and a model
load, and no other notebook should need `torch`. RSU/FSU curves meet neither — any
notebook with the cached tensor and unit metadata can derive them in milliseconds,
so forcing them into this notebook would just be moving Part 5's own analysis logic
upstream into preprocessing for no storage or dependency benefit, while creating an
artificial coupling between this notebook and Part 5's specific RSA implementation.
The general rule applied throughout: **raw extraction lives here; analysis logic
lives in the part it belongs to**, even when the analysis logic is simple enough
that it could technically live here too.

## 8. Strategy: downsample images for the gallery, never use them for analysis

**Decision.** `image_templates.npy` stores natural-scene images downsampled to
256×256 uint8, explicitly documented as not used for any quantitative analysis
(only the qualitative galleries in Parts 1 and 4).

**Why.** Full Allen-resolution natural scene templates are far larger than needed
to render an 8-panel or 24-panel gallery figure. No quantitative analysis in the
paper operates on pixel data directly (response analyses use neural data; the CNN
embeddings are computed once here from the full-resolution images before the
display copy is downsampled). Downsampling only the visualization copy avoids
inflating the archive for a use case that doesn't need the extra resolution.

## 9. Strategy: per-area, per-stimulus grouping inside one HDF5 file rather than separate files

**Decision.** All binned tensors and non-VISp noise ceilings live in a single
`responses.h5`, organized as nested groups
(`natural_scenes/{area}/{session_id}/tensor`,
`static_gratings/VISp/{session_id}/tensor`,
`natural_scenes/{area}/noise_ceiling`), rather than one file per
session/area/stimulus combination.

**Why HDF5 over many small files.** Session unit-counts are ragged (14–110 units
per session), so a single dense array can't hold all sessions; but a flat
directory of hundreds of small `.npy` files is unwieldy to ship and version. HDF5
groups give the ragged structure a queryable hierarchy while keeping everything in
one file with one download, and `h5py` supports reading a single dataset (e.g. one
session's tensor) without loading the rest of the file into memory — so a
part-notebook that only needs VISp doesn't pay the cost of the other four areas
being present in the same archive.

**Why gzip level 4 compression.** Spike-count data is sparse (mostly zeros and
small integers in any 5 ms bin), which compresses well under gzip; level 4 is a
deliberate middle point that gets most of the size reduction without the slower
write times of level 9, since this notebook only writes the archive once.

## 10. Strategy: a manifest.json as the single source of truth for reconstruction parameters

**Decision.** Bin edges, QC thresholds, session lists per area, grating condition
list, and the window reconstruction formula are all written to one `manifest.json`,
rather than being re-declared as constants inside each part-notebook.

**Why.** If the bin width, QC thresholds, or session selection logic ever change,
every part-notebook that re-declared its own copy of these constants would silently
drift out of sync with the actual data in the archive. Centralizing them in a file
shipped alongside the data means every part-notebook reads its parameters from the
same artifact it reads its data from — the manifest and the data can never disagree
about what bin edges were used, because they were written by the same notebook run.

## 11. Strategy: all utility functions are defined in one cell

**Decision.** All functions used by more than one extraction cell —
`get_nwb_path`, `load_unit_spike_times`, `bin_spikes_per_trial`,
`get_ns_presentations`, `get_grating_presentations`, `qc_unit_ids`,
`read_pupil_from_nwb`, `read_opto_pulse_starts` — are defined together in a
single utility cell near the top of the notebook, before any extraction cell runs.

**Why.** Colab notebooks are frequently re-run from a specific cell rather than
top-to-bottom. If utility functions are defined inside the extraction cell that
first needs them, any later cell that re-runs in isolation will hit a NameError.
Centralizing all function definitions in one early cell means any extraction cell
can be re-run independently as long as the utility cell and the global parameters
cell have been executed — which is always the case in a normal top-to-bottom run.

## 12. Strategy: explicit del + gc.collect() after every session's extraction

**Decision.** Every session-loop body ends with `del session, ust, tensor` (or
equivalent) followed by `gc.collect()`, immediately after that session's data is
written to the output file.

**Why.** This notebook is designed to run on Colab's default (non-Pro) memory
tier. Without explicit cleanup, AllenSDK `Session` objects and per-unit spike-time
dictionaries accumulate across 32+ iterations of the loop and can exceed available
RAM well before the loop finishes. Discarding each session's raw objects the moment
its derived data is safely written keeps peak memory bounded by one session's worth
of raw data, not all 32+.

## 13. What is deliberately *not* cached, and why

- **Raw NWB files** — deleted at the end. Nothing downstream needs them once every
  raw quantity has been extracted into the archive; keeping them would multiply the
  deliverable size by roughly 10–20x for no benefit.
- **Full-resolution natural scene images** — see §8; only a 256×256 display copy
  is kept.
- **Single-trial spike data outside the 0–250 ms VISp window** — see §3; the two
  analyses that need single-trial resolution don't need it outside that window or
  outside VISp.
- **VISp natural-scenes noise ceiling** — see §4; Part 1 computes this from the
  cached trial-level table with no NWB access required.
- **RSU/FSU consistency curves, E/I model fits, any other Part 5+ analysis output**
  — see §7; these are analysis results, not raw extractions, and belong in the
  notebook that produces the analysis, with explicit notebook-to-notebook handoff
  (Part 5 → Part 6) rather than being pre-computed here.
