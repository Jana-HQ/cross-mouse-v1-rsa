"""
utils.py — Shared utilities for the cross-mouse V1 RSA analysis.

Import at the top of every part-notebook:

    from utils import (
        setup_archive, window_response,
        compute_similarity_matrices, per_image_consistency,
        rdm_split_half_ceiling, cluster_bootstrap_ci, apply_plot_style,
        ts_ratio, print_figure_data,
        circular_shift_pvalue, circular_shift_partial_pvalue,
    )

setup_archive() must be called before window_response(); it sets the
module-level bin_edges array used by window_response() when no bin_edges
argument is passed explicitly.
"""

import json
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Module-level state (populated by setup_archive)
# ---------------------------------------------------------------------------
_BIN_EDGES: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Archive setup
# ---------------------------------------------------------------------------

def setup_archive(
    zip_path:    str | Path = '../preprocessed_data.zip',
    archive_dir: str | Path = '../preprocessed_data',
    figures_dir: str | Path = '../figures',
    outputs_dir: str | Path = '.',
):
    """
    Ensure the preprocessed data archive is present and extracted, then
    return all path objects and the parsed manifest.

    Parameters
    ----------
    zip_path    : path to preprocessed_data.zip (relative to notebook CWD)
    archive_dir : directory the zip should be extracted into
    figures_dir : directory where figure PDFs will be saved
    outputs_dir : directory where inter-notebook .npz files are read/written
                  (defaults to '.' — same directory as the notebooks)

    Returns
    -------
    manifest    : dict — contents of manifest.json
    bin_edges   : (96,) float64 array of bin edges in ms
    ARCHIVE_DIR : Path to extracted archive
    FIGURES_DIR : Path to figures directory (created if absent)
    OUTPUTS_DIR : Path to part outputs directory (created if absent)
    """
    global _BIN_EDGES

    zip_path    = Path(zip_path)
    archive_dir = Path(archive_dir)
    figures_dir = Path(figures_dir)
    outputs_dir = Path(outputs_dir)

    if not zip_path.exists():
        raise FileNotFoundError(
            f'\nArchive not found: {zip_path.resolve()}\n'
            'Place preprocessed_data.zip one level above the notebooks/ directory.'
        )

    if not (archive_dir / 'manifest.json').exists():
        print(f'Extracting {zip_path} → {archive_dir} ...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(archive_dir)
        print('Extraction complete.')
    else:
        print(f'Archive already extracted: {archive_dir.resolve()}')

    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    manifest   = json.load(open(archive_dir / 'manifest.json'))
    bin_edges  = np.array(manifest['bin_edges_ms'])
    _BIN_EDGES = bin_edges

    print(
        f'Manifest: {len(manifest["selected_session_ids_visp"])} VISp sessions | '
        f'{manifest["n_images"]} images | '
        f'{len(bin_edges) - 1} bins × {manifest["bin_width_ms"]} ms'
    )

    return manifest, bin_edges, archive_dir, figures_dir, outputs_dir


# ---------------------------------------------------------------------------
# Window reconstruction
# ---------------------------------------------------------------------------

def window_response(
    tensor:   np.ndarray,
    t_lo_ms:  float,
    t_hi_ms:  float,
    bin_edges: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reconstruct a z-scored response matrix for any time window.

    Parameters
    ----------
    tensor    : (n_images, n_units, n_bins) float32
    t_lo_ms   : left edge of window in ms (relative to stimulus onset)
    t_hi_ms   : right edge of window in ms (exclusive)
    bin_edges : (n_bins+1,) array; falls back to the module-level default
                set by setup_archive() if not provided

    Returns
    -------
    X : (n_images, n_units) float32 — z-scored per unit across images
    """
    be = bin_edges if bin_edges is not None else _BIN_EDGES
    if be is None:
        raise RuntimeError(
            'bin_edges not initialised — call setup_archive() first, '
            'or pass bin_edges explicitly.'
        )
    lo = np.searchsorted(be, t_lo_ms)
    hi = np.searchsorted(be, t_hi_ms)
    X  = tensor[:, :, lo:hi].sum(axis=2).astype(np.float32)
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


# ---------------------------------------------------------------------------
# RSA pipeline
# ---------------------------------------------------------------------------

def compute_similarity_matrices(response_matrices: list) -> list:
    """
    Compute the cosine similarity matrix for each session's response matrix.

    Parameters
    ----------
    response_matrices : list of (n_items, n_units) float32 arrays
                        (one per session / mouse)

    Returns
    -------
    list of (n_items, n_items) float32 arrays
    """
    result = []
    for X in response_matrices:
        norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        result.append((norm @ norm.T).astype(np.float32))
    return result


def per_image_consistency(
    sim_matrices: list,
    n_images:     int | None  = None,
    mouse_ids:    list | None = None,
):
    """
    Mean Spearman r across mouse pairs for each image (or grating condition).

    The 'profile' for image i in each mouse is its row of the cosine
    similarity matrix (how similar image i is to every other image in the
    same session).  Spearman r between two mice's profiles measures how
    consistently they organise the image similarity structure around that
    image.  Averaging across all valid pairs gives per-image consistency.

    Parameters
    ----------
    sim_matrices : list of (n_items, n_items) float32 arrays
    n_images     : number of items; inferred from sim_matrices[0].shape[0]
                   if not supplied (supports both images and grating conditions)
    mouse_ids    : list of original mouse indices (for bootstrap — pairs with
                   the same original ID are excluded to prevent self-correlation)

    Returns
    -------
    mean_c   : (n_images,) mean per-image consistency across pairs
    per_pair : (n_pairs, n_images) per-pair Spearman r matrix
    pairs    : list of (a, b) index tuples
    """
    n = n_images if n_images is not None else sim_matrices[0].shape[0]
    n_mice = len(sim_matrices)
    if mouse_ids is None:
        mouse_ids = list(range(n_mice))

    pairs = [
        (a, b) for a, b in combinations(range(n_mice), 2)
        if mouse_ids[a] != mouse_ids[b]
    ]
    if not pairs:
        return np.full(n, np.nan), np.zeros((0, n)), []

    mask     = ~np.eye(n, dtype=bool)
    stacked  = np.stack(sim_matrices)                   # (n_mice, n, n)
    stripped = stacked[:, mask].reshape(n_mice, n, n - 1)

    order  = np.argsort(stripped, axis=2)
    ranked = np.empty_like(order, dtype=float)
    np.put_along_axis(ranked, order, np.arange(n - 1), axis=2)

    per_pair = np.zeros((len(pairs), n))
    for p, (a, b) in enumerate(pairs):
        ra  = ranked[a] - ranked[a].mean(1, keepdims=True)
        rb  = ranked[b] - ranked[b].mean(1, keepdims=True)
        num = (ra * rb).sum(1)
        den = np.sqrt((ra**2).sum(1) * (rb**2).sum(1)) + 1e-8
        per_pair[p] = num / den

    return per_pair.mean(0), per_pair, pairs


def rdm_split_half_ceiling(mats_half1: list, mats_half2: list) -> np.ndarray:
    """
    Per-mouse split-half reliability computed in RDM space, matching the
    units of per_image_consistency (rather than raw response-vector space).

    Each mouse's two independent-trial-half response matrices are turned
    into similarity matrices, then per_image_consistency is applied to the
    pair [sim_half1, sim_half2] exactly as it would be applied to two
    different mice's similarity matrices — the resulting Spearman r is the
    within-mouse RDM reliability for that mouse, in the same space as
    cross-mouse consistency.

    Parameters
    ----------
    mats_half1, mats_half2 : lists of (n_items, n_units) float32 arrays,
        one per mouse, from two independent trial splits (same items,
        disjoint trials)

    Returns
    -------
    reliability : (n_mice,) array — Spearman-Brown corrected per-mouse
                  RDM split-half reliability
    """
    n_mice = len(mats_half1)
    reliability = np.zeros(n_mice)
    for i in range(n_mice):
        sim1, sim2 = compute_similarity_matrices([mats_half1[i], mats_half2[i]])
        half_r, _, _ = per_image_consistency([sim1, sim2])
        r = half_r.mean()
        reliability[i] = 2 * r / (1 + r)   # Spearman-Brown correction
    return reliability


def cluster_bootstrap_ci(
    response_matrices: list,
    n_boot:            int   = 1000,
    ci:                float = 95,
    seed:              int   = 42,
):
    """
    Cluster bootstrap (resample whole mice) for per-image consistency CIs.

    Self-pairs (same original mouse resampled twice) are excluded so
    bootstrap replicates cannot inflate consistency via self-correlation.

    Parameters
    ----------
    response_matrices : list of (n_images, n_units) arrays — one per session
    n_boot            : number of bootstrap replicates
    ci                : confidence interval width in percent
    seed              : random seed

    Returns
    -------
    lo, hi : each (n_images,) — lower and upper CI bounds
    """
    rng    = np.random.default_rng(seed)
    n_mice = len(response_matrices)
    n      = response_matrices[0].shape[0]   # n_images
    boot   = np.zeros((n_boot, n))

    for b in range(n_boot):
        idx       = rng.choice(n_mice, size=n_mice, replace=True)
        resampled = [response_matrices[i] for i in idx]
        sims      = compute_similarity_matrices(resampled)
        c, _, _   = per_image_consistency(sims, mouse_ids=list(idx))
        boot[b]   = c

    lo = np.percentile(boot, (100 - ci) / 2,        axis=0)
    hi = np.percentile(boot, 100 - (100 - ci) / 2,  axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# Across-window correlation testing (autocorrelation-robust)
# ---------------------------------------------------------------------------

def circular_shift_pvalue(x, y, n_perm: int = 5000, seed: int = 42, corr_fn=None):
    """
    Autocorrelation-robust p-value for a correlation between two time series.

    Sliding-window statistics (e.g. PR, consistency) computed from overlapping
    windows are themselves autocorrelated, so the standard Spearman/Pearson
    p-value (which assumes independent observations) understates the true
    p-value. This builds a null distribution by circularly shifting `y`
    relative to `x` by every possible lag-preserving roll, which destroys the
    x-y alignment while leaving each series' own autocorrelation structure
    intact.

    Parameters
    ----------
    x, y    : 1-D arrays of equal length (paired observations across windows)
    n_perm  : number of circular shifts to draw (sampled with replacement
              from the n-1 non-trivial shifts if n_perm < n - 1, else all
              n - 1 shifts are used)
    seed    : random seed
    corr_fn : callable(x, y) -> (r, p); defaults to scipy's spearmanr

    Returns
    -------
    r_obs, p_corrected : observed correlation and the circular-shift p-value
    """
    from scipy.stats import spearmanr

    if corr_fn is None:
        corr_fn = lambda a, b: spearmanr(a, b)

    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    r_obs, _ = corr_fn(x, y)

    shifts = np.arange(1, n)
    if n_perm < len(shifts):
        rng    = np.random.default_rng(seed)
        shifts = rng.choice(shifts, size=n_perm, replace=False)

    null_r = np.empty(len(shifts))
    for i, s in enumerate(shifts):
        r, _ = corr_fn(x, np.roll(y, s))
        null_r[i] = r

    p_corrected = (np.sum(np.abs(null_r) >= np.abs(r_obs)) + 1) / (len(shifts) + 1)
    return float(r_obs), float(p_corrected)


def circular_shift_partial_pvalue(x, y, z, n_perm: int = 5000, seed: int = 42):
    """
    Autocorrelation-robust p-value for a Spearman partial correlation
    r_xy.z between two autocorrelated time series x, y controlling for z.

    Circularly shifts `x` relative to the paired (y, z), which preserves
    the y-z relationship (e.g. consistency's link to the response envelope)
    while destroying any genuine x-y alignment beyond what shifted-x still
    shares with z by chance. Uses the same partial-correlation formula as
    the observed statistic on every shifted replicate.

    Parameters
    ----------
    x, y, z : 1-D arrays of equal length (paired observations across windows)
    n_perm  : number of circular shifts to draw (see circular_shift_pvalue)
    seed    : random seed

    Returns
    -------
    r_obs, p_corrected : observed partial correlation and its circular-shift p-value
    """
    from scipy.stats import spearmanr

    def _partial(a, b, c):
        r_ab, _ = spearmanr(a, b)
        r_ac, _ = spearmanr(a, c)
        r_bc, _ = spearmanr(b, c)
        return (r_ab - r_ac * r_bc) / np.sqrt((1 - r_ac ** 2) * (1 - r_bc ** 2))

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    n = len(x)
    r_obs = _partial(x, y, z)

    shifts = np.arange(1, n)
    if n_perm < len(shifts):
        rng    = np.random.default_rng(seed)
        shifts = rng.choice(shifts, size=n_perm, replace=False)

    null_r = np.empty(len(shifts))
    for i, s in enumerate(shifts):
        null_r[i] = _partial(np.roll(x, s), y, z)

    p_corrected = (np.sum(np.abs(null_r) >= np.abs(r_obs)) + 1) / (len(shifts) + 1)
    return float(r_obs), float(p_corrected)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def apply_plot_style():
    """
    Apply the project-wide matplotlib / seaborn style and return
    the shared colour-constant dictionary.

    Each constant encodes exactly one concept across every notebook it
    appears in — colors are reused only when the underlying quantity is
    genuinely the same (e.g. COL_DATA / natural scenes / VISp are all the
    same default consistency curve). Everything else gets its own dedicated
    hue rather than borrowing another concept's color.

    Usage
    -----
        colors = apply_plot_style()
        COL_DATA = colors['COL_DATA']
        COL_REF  = colors['COL_REF']
        COL_FIT  = colors['COL_FIT']
        ...
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.size':       11,
        'axes.labelsize':  12,
        'axes.titlesize':  12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
        'figure.dpi':      150,
    })
    sns.set_style('ticks')

    return {
        # --- structural roles (same meaning in every notebook) ---
        'COL_DATA': '#0072B2',  # default single-series consistency curve == natural scenes == VISp
        'COL_REF':  '#999999',  # reference / baseline / "all units" / unity line
        'COL_FIT':  '#1B5E20',  # model fit / regression line / peak-floor annotation

        # --- stimulus type (part8) ---
        'COL_GR':   '#e9c716',  # gratings (natural scenes == COL_DATA)

        # --- cell type (part5, part6 E/I model) ---
        'COL_RSU':    "#D03252",  # RSU / excitatory-like
        'COL_FSU':    "#1ba8b5",  # FSU / inhibitory-like (waveform-classified)
        'COL_FSU_PV': "#8cc5e3",  # FSU optotagged PV+ validation (darker tint of COL_FSU)

        # --- brain area (part9) ---
        'AREA_COLORS': {
            'VISp':  '#0072B2',  # == COL_DATA, kept consistent
            'VISl':  '#2CA02C',
            'VISal': '#E6B800',
            'VISpm': '#D6455D',
            'VISam': '#7B4FA0',
        },

        # --- CNN layer identity (part2, Fig2C/D + FigS2) ---
        # named COL_* (not CNN_PRIMARY/CNN_SECONDARY) to avoid colliding with
        # part2's own local variables of that name, which hold ResNet50 layer
        # *name strings* (e.g. 'layer2', 'final'), not colors.
        'COL_CNN_PRIMARY':   "#8A6AC8",
        'COL_CNN_SECONDARY': "#468788",

        # --- one-off accents, each a distinct quantity used in exactly one notebook ---
        'COL_DIM':  '#B39DDB',  # dimensionality / participation ratio (part4)
        'COL_REI':  '#661100',  # r_EI parameter (part6)
        'COL_2ND':  '#9B6A6C',  # 2nd-order RSA (part3 / Fig S8)

        # --- independent subgroup accents (not shared with each other) ---
        'COL_HIGH_FSU_FRAC':   "#59deca",  # high-FSU-fraction sessions (part5C)
        'COL_LOW_FSU_FRAC':    "#f37c80",  # low-FSU-fraction sessions (part5C)
        'COL_INFORMED_START':  '#C1440E',  # informed-start robustness check (part6)

        # --- E/I model fit-target identity (part6): which of the three curves
        # a panel was fit to (consistency / total rate / selective amplitude).
        # Chosen light-but-saturated so they stay visible on white backgrounds.
        'COL_TARGET_CONSISTENCY': '#E53935',  # light red
        'COL_TARGET_TOTAL_RATE':  '#FFB300',  # light yellow
        'COL_TARGET_SELECTIVE':   '#66BB6A',  # light green

        # --- train/test split tint (part6C), same metric as COL_DATA ---
        'COL_TEST': '#7FB8DD',
    }


# ---------------------------------------------------------------------------
# Temporal analysis helpers
# ---------------------------------------------------------------------------

def print_figure_data(title: str, **named_arrays) -> None:
    """
    Print the underlying data of a figure as plain text (in addition to the
    plotted PDF), so every panel has a human-readable, greppable record.

    Usage
    -----
        print_figure_data(
            'Fig 1A — noise ceiling histogram',
            noise_ceiling=noise_ceiling,
        )

    Each keyword becomes a labelled section; arrays and DataFrames are
    printed in a compact, truncated form, and long lists are shortened with
    an ellipsis marker.
    """
    import numpy as np
    import pandas as pd

    def _truncate_list(values, limit: int = 20):
        values = list(values)
        if len(values) <= limit:
            return values
        return values[:limit] + [f'... (+{len(values) - limit} more)']

    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')
    for name, val in named_arrays.items():
        print(f'\n--- {name} ---')
        if isinstance(val, pd.DataFrame):
            with pd.option_context(
                'display.max_rows', 20,
                'display.max_columns', 20,
                'display.width', 200,
                'display.max_colwidth', 80,
            ):
                print(val.head(20))
                if len(val) > 20 or len(val.columns) > 20:
                    print(f'...[{len(val)} rows × {len(val.columns)} columns total]')
        elif isinstance(val, (list, tuple)):
            if not val:
                print([])
            elif isinstance(val[0], str):
                print(_truncate_list(val))
            else:
                arr = np.asarray(val)
                if arr.size > 20:
                    print(arr.reshape(-1)[:20])
                    print(f'...[{arr.size} elements total]')
                else:
                    with np.printoptions(threshold=np.inf, linewidth=200,
                                          precision=6, suppress=True):
                        print(arr)
        else:
            arr = np.asarray(val)
            if arr.size > 20:
                print(f'shape={arr.shape}')
                with np.printoptions(threshold=20, edgeitems=3, linewidth=200,
                                     precision=6, suppress=True):
                    print(arr.reshape(-1)[:20])
                print(f'...[{arr.size} elements total]')
            else:
                with np.printoptions(threshold=np.inf, linewidth=200,
                                      precision=6, suppress=True):
                    print(arr)


def unit_matched_sliding_curve(
    tensors_by_session:   dict,
    unit_idx_by_session:  dict,
    target_n:             int,
    window_centers_ms:    list,
    window_width_ms:      float,
    n_draws:              int   = 50,
    seed:                 int   = 42,
    bin_edges:            np.ndarray | None = None,
):
    """
    Sliding-window consistency curve after subsampling every session's unit
    pool down to a common `target_n`, repeated over random draws.

    Controls for unit-count confounds when comparing groups (subpopulations,
    session groups, brain areas) whose raw unit counts differ, since
    similarity-matrix SNR (and hence raw consistency) scales with unit count.
    Sessions whose pool is already <= target_n are skipped (too few units to
    subsample without replacement).

    Parameters
    ----------
    tensors_by_session  : {session_id: (n_images, n_units, n_bins) tensor}
    unit_idx_by_session : {session_id: list of column indices into that
                            session's tensor defining the pool to subsample from}
    target_n            : number of units to draw per session per replicate
    window_centers_ms   : sliding-window center times (ms)
    window_width_ms     : window width (ms)
    n_draws             : number of random subsample draws
    seed                : random seed
    bin_edges           : passed through to window_response

    Returns
    -------
    times     : (n_windows,) array of window_centers_ms
    mean_c    : (n_windows,) mean consistency curve across draws
    ci_lo, ci_hi : (n_windows,) 2.5/97.5 percentile band across draws
    ts_draws  : (n_draws,) T/S ratio computed separately for each draw
    """
    rng = np.random.default_rng(seed)

    usable_sids = [sid for sid, idx in unit_idx_by_session.items() if len(idx) >= target_n]

    times = np.asarray(window_centers_ms, dtype=float)
    curves = np.zeros((n_draws, len(times)))

    for d in range(n_draws):
        draw_idx = {
            sid: rng.choice(unit_idx_by_session[sid], size=target_n, replace=False)
            for sid in usable_sids
        }
        for w, t_ms in enumerate(times):
            t_lo = t_ms - window_width_ms / 2
            t_hi = t_ms + window_width_ms / 2
            mats = [
                window_response(tensors_by_session[sid][:, draw_idx[sid], :], t_lo, t_hi, bin_edges)
                for sid in usable_sids
            ]
            sims = compute_similarity_matrices(mats)
            c, _, _ = per_image_consistency(sims)
            curves[d, w] = c.mean()

    mean_c = curves.mean(0)
    ci_lo  = np.percentile(curves, 2.5,  axis=0)
    ci_hi  = np.percentile(curves, 97.5, axis=0)
    ts_draws = np.array([ts_ratio(curves[d], times)[0] for d in range(n_draws)])

    return times, mean_c, ci_lo, ci_hi, ts_draws


def image_matched_sliding_curve(
    tensors_by_session:   dict,
    target_n:             int,
    window_centers_ms:    list,
    window_width_ms:      float,
    n_draws:              int   = 50,
    seed:                 int   = 42,
    bin_edges:            np.ndarray | None = None,
):
    """
    Sliding-window consistency curve after subsampling the item (image /
    condition) axis down to a common `target_n`, repeated over random draws.

    Analogous to `unit_matched_sliding_curve`, but controls for item-count
    confounds instead of unit-count ones: `per_image_consistency`'s Spearman
    correlation is computed over profiles of length `n_items - 1`, so its
    noise properties (and hence the raw consistency magnitude) depend on how
    many items are being compared, independent of any real representational
    difference. Subsampling one item set down to match the other's item
    count puts both on the same statistical footing before comparing raw
    consistency values.

    Parameters
    ----------
    tensors_by_session : {session_id: (n_items, n_units, n_bins) tensor} —
                          all sessions must share the same n_items and item
                          identity/order (e.g. natural-scene tensors keyed
                          the same way across sessions)
    target_n            : number of items to draw per replicate (shared
                           across sessions — the same item subset is used
                           for every session within a draw)
    window_centers_ms   : sliding-window center times (ms)
    window_width_ms     : window width (ms)
    n_draws             : number of random subsample draws
    seed                : random seed
    bin_edges           : passed through to window_response

    Returns
    -------
    times     : (n_windows,) array of window_centers_ms
    mean_c    : (n_windows,) mean consistency curve across draws
    ci_lo, ci_hi : (n_windows,) 2.5/97.5 percentile band across draws
    ts_draws  : (n_draws,) T/S ratio computed separately for each draw
    """
    rng = np.random.default_rng(seed)

    sids = list(tensors_by_session.keys())
    n_items_total = tensors_by_session[sids[0]].shape[0]

    times  = np.asarray(window_centers_ms, dtype=float)
    curves = np.zeros((n_draws, len(times)))

    for d in range(n_draws):
        item_idx = rng.choice(n_items_total, size=target_n, replace=False)
        for w, t_ms in enumerate(times):
            t_lo = t_ms - window_width_ms / 2
            t_hi = t_ms + window_width_ms / 2
            mats = [
                window_response(tensors_by_session[sid][item_idx], t_lo, t_hi, bin_edges)
                for sid in sids
            ]
            sims = compute_similarity_matrices(mats)
            c, _, _ = per_image_consistency(sims)
            curves[d, w] = c.mean()

    mean_c = curves.mean(0)
    ci_lo  = np.percentile(curves, 2.5,  axis=0)
    ci_hi  = np.percentile(curves, 97.5, axis=0)
    ts_draws = np.array([ts_ratio(curves[d], times)[0] for d in range(n_draws)])

    return times, mean_c, ci_lo, ci_hi, ts_draws


def bootstrap_ts_difference(
    tensors_a:            dict,
    idx_a:                dict,
    target_n_a:           int | None,
    tensors_b:            dict,
    idx_b:                dict,
    target_n_b:           int | None,
    window_centers_ms:    list,
    window_width_ms:      float,
    n_unit_draws:         int   = 25,
    n_session_boot:       int   = 40,
    seed:                 int   = 42,
    bin_edges:            np.ndarray | None = None,
):
    """
    Significance test for a difference in T/S ratio between two groups,
    combining two independent sources of sampling noise:

    (1) which units were subsampled within each session (if `target_n_*` is
        given, a fresh random subsample is drawn per unit-draw — this is
        what `unit_matched_sliding_curve` alone captures), and
    (2) which sessions/mice were sampled at all (a session-level cluster
        bootstrap with replacement, self-pairs excluded via `mouse_ids`,
        matching `cluster_bootstrap_ci`'s convention) — the dominant source
        of real uncertainty with only a few dozen sessions, and the piece a
        unit-subsampling-only control misses.

    Only windows in [0, 250] ms are used since that is all `ts_ratio` reads
    (peak search + sustained window), which keeps this tractable.

    Pass `target_n_a=None` (or `target_n_b=None`) to use a group's full unit
    pool unchanged (no subsampling), e.g. when comparing an already-small
    reference population against a matched/subsampled comparison population.

    Returns
    -------
    ts_a_boot, ts_b_boot : (n_unit_draws * n_session_boot,) bootstrap T/S values
    diff_boot            : (n_unit_draws * n_session_boot,) ts_a_boot - ts_b_boot
    p_value              : two-sided bootstrap p-value for diff_boot != 0
                            (2x the smaller tail fraction crossing zero)
    """
    rng = np.random.default_rng(seed)

    times = np.array([t for t in window_centers_ms if 0 <= t <= 250], dtype=float)

    sids_a = [sid for sid in tensors_a
              if target_n_a is None or len(idx_a[sid]) >= target_n_a]
    sids_b = [sid for sid in tensors_b
              if target_n_b is None or len(idx_b[sid]) >= target_n_b]
    n_a, n_b = len(sids_a), len(sids_b)

    ts_a_all, ts_b_all, diffs = [], [], []

    for _ in range(n_unit_draws):
        draw_idx_a = {
            sid: (rng.choice(idx_a[sid], size=target_n_a, replace=False)
                  if target_n_a is not None else np.asarray(idx_a[sid]))
            for sid in sids_a
        }
        draw_idx_b = {
            sid: (rng.choice(idx_b[sid], size=target_n_b, replace=False)
                  if target_n_b is not None else np.asarray(idx_b[sid]))
            for sid in sids_b
        }

        sims_a_by_win, sims_b_by_win = {}, {}
        for t_ms in times:
            t_lo, t_hi = t_ms - window_width_ms / 2, t_ms + window_width_ms / 2
            mats_a = [window_response(tensors_a[sid][:, draw_idx_a[sid], :], t_lo, t_hi, bin_edges)
                      for sid in sids_a]
            sims_a_by_win[t_ms] = compute_similarity_matrices(mats_a)
            mats_b = [window_response(tensors_b[sid][:, draw_idx_b[sid], :], t_lo, t_hi, bin_edges)
                      for sid in sids_b]
            sims_b_by_win[t_ms] = compute_similarity_matrices(mats_b)

        for _ in range(n_session_boot):
            resample_a = rng.integers(0, n_a, size=n_a)
            resample_b = rng.integers(0, n_b, size=n_b)

            curve_a = np.array([
                per_image_consistency([sims_a_by_win[t_ms][i] for i in resample_a],
                                       mouse_ids=list(resample_a))[0].mean()
                for t_ms in times
            ])
            curve_b = np.array([
                per_image_consistency([sims_b_by_win[t_ms][i] for i in resample_b],
                                       mouse_ids=list(resample_b))[0].mean()
                for t_ms in times
            ])

            ts_a, _, _ = ts_ratio(curve_a, times)
            ts_b, _, _ = ts_ratio(curve_b, times)
            ts_a_all.append(ts_a)
            ts_b_all.append(ts_b)
            diffs.append(ts_a - ts_b)

    diffs = np.array(diffs)
    tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_value = min(2 * tail, 1.0)

    return np.array(ts_a_all), np.array(ts_b_all), diffs, p_value


def ts_ratio(
    mc_arr:        np.ndarray,
    times_arr:     np.ndarray,
    sustained_lo:  float = 220,
    sustained_hi:  float = 250,
):
    """
    Transient-to-sustained ratio from a sliding-window consistency curve.

    Peak is the maximum of the curve in the 0–250 ms post-stimulus window.
    Sustained is the mean in [sustained_lo, sustained_hi] (default 220–250 ms).

    Returns
    -------
    ratio : float — peak / sustained
    peak  : float — peak consistency value
    sus   : float — mean sustained consistency
    """
    t    = np.asarray(times_arr)
    c    = np.asarray(mc_arr)

    post_mask = (t >= 0) & (t <= 250)
    if post_mask.sum() == 0:          # empty curve (e.g. no usable PV sessions)
        return np.nan, np.nan, np.nan

    peak     = c[post_mask].max()
    sus_mask = (t >= sustained_lo) & (t <= sustained_hi)
    sus      = c[sus_mask].mean() if sus_mask.sum() > 0 else np.nan

    return peak / (sus + 1e-8), peak, sus