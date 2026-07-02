"""
utils.py — Shared utilities for the cross-mouse V1 RSA analysis.

Import at the top of every part-notebook:

    from utils import (
        setup_archive, window_response,
        compute_similarity_matrices, per_image_consistency,
        cluster_bootstrap_ci, apply_plot_style, ts_ratio,
        print_figure_data,
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
# Plotting helpers
# ---------------------------------------------------------------------------

def apply_plot_style():
    """
    Apply the project-wide matplotlib / seaborn style and return
    the shared colour-constant dictionary.

    Usage
    -----
        colors = apply_plot_style()
        COL_NS  = colors['COL_NS']
        COL_GR  = colors['COL_GR']
        COL_FIT = colors['COL_FIT']
        COL_REF = colors['COL_REF']
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
        'COL_NS':  'steelblue',   # natural scenes / RSU / excitatory / primary
        'COL_GR':  'darkorange',  # gratings / FSU / inhibitory / secondary
        'COL_FIT': 'firebrick',   # model fits / peak annotations / genetic validation
        'COL_REF': 'dimgray',     # reference lines / baselines / all-units
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