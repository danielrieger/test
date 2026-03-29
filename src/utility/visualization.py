"""
Visualization utilities for smlm_score.

Three main functions:
  1. plot_density_2d         — KDE heatmap of an aligned NPC cluster
  2. plot_score_comparison   — Bar chart comparing valid vs. noise cluster scores
  3. plot_density_contour    — Contour density map with model AV positions overlaid
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
import os


# --- Color palette ---
_DARK_BG = "#0d1117"
_GRID_COLOR = "#21262d"
_TEXT_COLOR = "#c9d1d9"
_ACCENT_CYAN = "#58a6ff"
_ACCENT_ORANGE = "#f0883e"
_ACCENT_GREEN = "#3fb950"
_ACCENT_RED = "#f85149"
_ACCENT_PURPLE = "#bc8cff"

# Custom colormap: dark → cyan → white
_density_cmap = LinearSegmentedColormap.from_list(
    "smlm_density",
    ["#0d1117", "#0e2a47", "#0d4a7a", "#1a7fba", "#58a6ff", "#a5d6ff", "#ffffff"],
    N=256,
)


def _style_axis(ax, title="", xlabel="", ylabel=""):
    """Apply consistent dark theme styling to an axis."""
    ax.set_facecolor(_DARK_BG)
    ax.set_title(title, color=_TEXT_COLOR, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=_TEXT_COLOR, fontsize=11)
    ax.set_ylabel(ylabel, color=_TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)


# =========================================================================
# 1.  DENSITY HEATMAP
# =========================================================================
def plot_density_2d(
    aligned_points,
    title="NPC Cluster — Localization Density",
    grid_resolution=300,
    bandwidth=None,
    save_path=None,
):
    """
    2D KDE heatmap of a PCA-aligned NPC cluster.

    Parameters
    ----------
    aligned_points : ndarray (N, 2) or (N, 3)
        PCA-aligned localization coordinates.  Only X/Y are used.
    title : str
        Plot title.
    grid_resolution : int
        Number of grid points per axis.
    bandwidth : float or None
        KDE bandwidth in nm.  None = Scott's rule (auto).
    save_path : str or None
        If given, save figure to this path instead of showing.

    Returns
    -------
    fig : matplotlib Figure
    """
    xy = aligned_points[:, :2]

    # --- KDE ---
    kde = gaussian_kde(xy.T, bw_method=bandwidth)
    margin = 20  # nm padding
    xmin, xmax = xy[:, 0].min() - margin, xy[:, 0].max() + margin
    ymin, ymax = xy[:, 1].min() - margin, xy[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, grid_resolution),
        np.linspace(ymin, ymax, grid_resolution),
    )
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(positions).reshape(xx.shape)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=_DARK_BG)
    _style_axis(ax, title=title, xlabel="x [nm]", ylabel="y [nm]")

    im = ax.imshow(
        zz,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="equal",
        cmap=_density_cmap,
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Density", color=_TEXT_COLOR, fontsize=10)
    cbar.ax.tick_params(colors=_TEXT_COLOR, labelsize=8)

    # Annotation
    ax.text(
        0.02, 0.02,
        f"N = {len(xy)} localizations",
        transform=ax.transAxes,
        fontsize=9,
        color=_ACCENT_CYAN,
        alpha=0.8,
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=_DARK_BG, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    return fig


# =========================================================================
# 2.  SCORE COMPARISON BAR CHART
# =========================================================================
def plot_score_comparison(
    cluster_scores,
    normalize_per_point=True,
    save_path=None,
):
    """
    Grouped bar chart comparing scoring functions across valid vs. noise clusters.

    Parameters
    ----------
    cluster_scores : dict
        {cluster_id: {'type': 'Valid'|'Noise', 'n_points': N,
                       'Tree': score, 'GMM': score, 'Distance': score}}
    normalize_per_point : bool
        If True, divide each score by n_points for fair comparison.
    save_path : str or None
        If given, save figure instead of showing.

    Returns
    -------
    fig : matplotlib Figure
    """
    scoring_types = ["Tree", "GMM", "Distance"]
    colors = {
        "Tree": _ACCENT_CYAN,
        "GMM": _ACCENT_PURPLE,
        "Distance": _ACCENT_ORANGE,
    }

    # Separate valid / noise
    valid_ids = [cid for cid, v in cluster_scores.items() if v["type"] == "Valid"]
    noise_ids = [cid for cid, v in cluster_scores.items() if v["type"] == "Noise"]

    # Build score arrays
    def _get_scores(ids, stype):
        scores = []
        for cid in ids:
            s = cluster_scores[cid].get(stype)
            if s is not None:
                n = cluster_scores[cid]["n_points"]
                scores.append(s / n if normalize_per_point else s)
        return scores

    fig, axes = plt.subplots(1, len(scoring_types), figsize=(5 * len(scoring_types), 5),
                             facecolor=_DARK_BG, sharey=False)
    if len(scoring_types) == 1:
        axes = [axes]

    for ax, stype in zip(axes, scoring_types):
        _style_axis(ax, title=f"{stype} Score", xlabel="Cluster Type",
                    ylabel="Score / point" if normalize_per_point else "Total Score")

        valid_scores = _get_scores(valid_ids, stype)
        noise_scores = _get_scores(noise_ids, stype)

        # Positions
        positions = []
        values = []
        bar_colors = []
        labels = []

        if valid_scores:
            for i, s in enumerate(valid_scores):
                positions.append(i)
                values.append(s)
                bar_colors.append(_ACCENT_GREEN)
                labels.append(f"V{valid_ids[i]}")
        offset = len(valid_scores)
        if noise_scores:
            for i, s in enumerate(noise_scores):
                positions.append(offset + i)
                values.append(s)
                bar_colors.append(_ACCENT_RED)
                labels.append(f"N{noise_ids[i]}")

        bars = ax.bar(positions, values, color=bar_colors, alpha=0.85, edgecolor="white",
                      linewidth=0.5, width=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8, color=_TEXT_COLOR)
        ax.axhline(y=0, color=_GRID_COLOR, linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            y_pos = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{val:.1f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7,
                color=_TEXT_COLOR,
                alpha=0.8,
            )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_ACCENT_GREEN, label="Valid NPC"),
        Patch(facecolor=_ACCENT_RED, label="Noise"),
    ]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=9,
                    facecolor=_DARK_BG, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)

    fig.suptitle(
        "Scoring Function Comparison: Valid vs. Noise Clusters",
        color=_TEXT_COLOR, fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=_DARK_BG, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    return fig


# =========================================================================
# 3.  DENSITY CONTOUR + AV OVERLAY
# =========================================================================
def plot_density_contour(
    aligned_points,
    av_positions_nm,
    gmm_means=None,
    gmm_covs=None,
    title="Model–Data Overlay",
    grid_resolution=300,
    save_path=None,
):
    """
    Contour density map of SMLM data with model AV positions overlaid.

    Parameters
    ----------
    aligned_points : ndarray (N, 2) or (N, 3)
        PCA-aligned localization coordinates.
    av_positions_nm : ndarray (K, 2) or (K, 3)
        Model AV positions in nm (already offset-corrected to data frame).
    gmm_means : ndarray (C, 2/3) or None
        If provided, also plot GMM component centers.
    gmm_covs : ndarray (C, 2/3, 2/3) or None
        GMM covariance matrices for drawing ellipses.
    title : str
        Plot title.
    grid_resolution : int
        Grid resolution for KDE.
    save_path : str or None
        Save path.

    Returns
    -------
    fig : matplotlib Figure
    """
    xy = aligned_points[:, :2]
    av_xy = av_positions_nm[:, :2]

    # --- KDE ---
    kde = gaussian_kde(xy.T)
    margin = 25
    xmin, xmax = xy[:, 0].min() - margin, xy[:, 0].max() + margin
    ymin, ymax = xy[:, 1].min() - margin, xy[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, grid_resolution),
        np.linspace(ymin, ymax, grid_resolution),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=_DARK_BG)
    _style_axis(ax, title=title, xlabel="x [nm]", ylabel="y [nm]")

    # Filled contour (density)
    levels = np.linspace(zz.min(), zz.max(), 20)
    cf = ax.contourf(xx, yy, zz, levels=levels, cmap=_density_cmap, alpha=0.9)

    # Contour lines
    ax.contour(xx, yy, zz, levels=levels[::3], colors="white", linewidths=0.3, alpha=0.4)

    # Data points (faint scatter)
    ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="white", alpha=0.08, rasterized=True)

    # --- AV model positions ---
    ax.scatter(
        av_xy[:, 0], av_xy[:, 1],
        s=120, c=_ACCENT_ORANGE, edgecolors="white", linewidths=1.5,
        marker="*", zorder=10, label=f"Model AVs (n={len(av_xy)})",
    )
    # Number each AV
    for i, (x, y) in enumerate(av_xy):
        ax.annotate(
            str(i), (x, y),
            textcoords="offset points", xytext=(5, 5),
            fontsize=7, color=_ACCENT_ORANGE, fontweight="bold",
        )

    # --- Optional: GMM ellipses ---
    if gmm_means is not None and gmm_covs is not None:
        for k in range(len(gmm_means)):
            mean_2d = gmm_means[k][:2]
            cov_2d = gmm_covs[k][:2, :2]

            # Eigendecomposition for ellipse
            eigvals, eigvecs = np.linalg.eigh(cov_2d)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width, height = 2 * np.sqrt(eigvals)  # 1-sigma

            ellipse = Ellipse(
                xy=mean_2d, width=width, height=height, angle=angle,
                fill=False, edgecolor=_ACCENT_PURPLE, linewidth=1.2,
                linestyle="--", alpha=0.7,
            )
            ax.add_patch(ellipse)

        # GMM centers
        ax.scatter(
            gmm_means[:, 0], gmm_means[:, 1],
            s=40, c=_ACCENT_PURPLE, marker="x", linewidths=1.5,
            zorder=9, label=f"GMM components (n={len(gmm_means)})",
        )

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Localization Density", color=_TEXT_COLOR, fontsize=10)
    cbar.ax.tick_params(colors=_TEXT_COLOR, labelsize=8)

    ax.legend(
        loc="upper right", fontsize=9,
        facecolor=_DARK_BG, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR,
    )
    ax.set_aspect("equal")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=_DARK_BG, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    return fig
# =========================================================================
# 4.  MAP OF FULL DATA WITH HIGHLIGHTED CLUSTERS
# =========================================================================
def plot_cluster_context_map(
    full_data,
    cluster_labels,
    target_cluster_id=None,
    sample_clusters=None,
    title="SMLM High-Density ROI & Cluster Map",
    save_path=None,
    show_zooms=True
):
    """
    Plots the full dataset (often the ROI) as a sparse background map, 
    highlighting the specific target cluster and optionally a few random clusters.
    Also includes zoomed-in inset panels for the highlighted clusters to inspect their shapes.
    """
    clusters_to_zoom = []
    if target_cluster_id is not None:
        clusters_to_zoom.append((target_cluster_id, _ACCENT_GREEN, "Target"))
    
    sample_colors = [_ACCENT_ORANGE, _ACCENT_PURPLE, _ACCENT_RED, "#f1e05a"]
    if sample_clusters is not None:
        for i, cid in enumerate(sample_clusters):
            if cid != target_cluster_id:
                clusters_to_zoom.append((cid, sample_colors[i % len(sample_colors)], "Sample"))

    n_zooms = len(clusters_to_zoom) if show_zooms else 0
    
    # Layout using GridSpec
    fig = plt.figure(figsize=(15, 10), facecolor=_DARK_BG)
    if n_zooms > 0:
        gs = fig.add_gridspec(max(n_zooms, 1), 3)
        ax_main = fig.add_subplot(gs[:, :-1])
    else:
        gs = fig.add_gridspec(1, 1)
        ax_main = fig.add_subplot(gs[0, 0])
    
    _style_axis(ax_main, title=title, xlabel="x [nm]", ylabel="y [nm]")
    
    # 1. Plot entire data as grey noise background
    ax_main.scatter(
        full_data[:, 0], full_data[:, 1],
        s=0.5, c="#333", alpha=0.3, rasterized=True, label="Noise"
    )

    # 2. Categorical coloring for ALL detected clusters (like experiment_eps)
    unique_labels = sorted(set(cluster_labels) - {-1})
    if unique_labels:
        # Use a high-resolution colormap (nipy_spectral) to better distinguish 300+ clusters
        n_clusters = len(unique_labels)
        clr_map = plt.cm.get_cmap('nipy_spectral', n_clusters)
        
        # Shuffle indices to ensure adjacent clusters don't get adjacent colors in the spectrum
        shuffled_indices = np.random.RandomState(42).permutation(n_clusters)
        
        for idx, cid in enumerate(unique_labels):
            if cid == target_cluster_id: continue 
            mask = cluster_labels == cid
            # Pick a distinct color from the 256-step spectrum
            color_val = clr_map(shuffled_indices[idx] / n_clusters)
            ax_main.scatter(full_data[mask, 0], full_data[mask, 1], 
                            s=1.0, c=[color_val], alpha=0.8, rasterized=True)


    # Dictionary to store bounding boxes for the zoom panels
    zoom_axes = []
    
    # Plot all highlighted clusters
    for i, (cid, color, label_prefix) in enumerate(clusters_to_zoom):
        mask = (cluster_labels == cid)
        if np.any(mask):
            pts = full_data[mask]
            
            # Plot on main map
            if label_prefix == "Target":
                ax_main.scatter(pts[:, 0], pts[:, 1], s=15.0, c=color, alpha=0.2, rasterized=True)
                ax_main.scatter(pts[:, 0], pts[:, 1], s=4.0, c=color, alpha=0.9, rasterized=True, label=f"{label_prefix} NPC ({cid})")
                
                centroid = np.mean(pts, axis=0)
                ax_main.annotate(
                    f"{cid}", xy=(centroid[0], centroid[1]), xytext=(centroid[0] + 50, centroid[1] + 50),
                    arrowprops=dict(facecolor=_TEXT_COLOR, edgecolor=_TEXT_COLOR, shrink=0.05, width=1, headwidth=4),
                    fontsize=10, color=_TEXT_COLOR, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc=_GRID_COLOR, ec=_TEXT_COLOR, alpha=0.7)
                )
            else:
                ax_main.scatter(pts[:, 0], pts[:, 1], s=2.0, c=color, alpha=0.8, rasterized=True, label=f"{label_prefix} {cid}")

            # Create zoom subplot
            ax_zoom = fig.add_subplot(gs[i, -1])
            _style_axis(ax_zoom, title=f"{label_prefix} {cid} (N={len(pts)})")
            
            # Fixed zoom window size to accurately compare scaling (e.g. 200x200 nm window)
            centroid = np.mean(pts, axis=0)
            window_size = 150 # +/- 150 nm from centroid
            
            # Plot ALL data in the background of the zoom window to see what was excluded
            zoom_mask = (full_data[:, 0] > centroid[0] - window_size) & (full_data[:, 0] < centroid[0] + window_size) & \
                        (full_data[:, 1] > centroid[1] - window_size) & (full_data[:, 1] < centroid[1] + window_size)
            ax_zoom.scatter(full_data[zoom_mask, 0], full_data[zoom_mask, 1], s=2.0, c="#8b949e", alpha=0.3, rasterized=True)
            
            # Plot clustered points in zoom on top
            ax_zoom.scatter(pts[:, 0], pts[:, 1], s=12.0, c=color, alpha=0.9, rasterized=True)
            
            ax_zoom.set_xlim(centroid[0] - window_size, centroid[0] + window_size)
            ax_zoom.set_ylim(centroid[1] - window_size, centroid[1] + window_size)
            ax_zoom.set_aspect("equal")
            
            # Remove xy labels for tiny plots to save space, keep ticks
            if i < n_zooms - 1:
                ax_zoom.set_xlabel("")
            ax_zoom.set_ylabel("")

    # Main map styling & Legend
    ax_main.legend(loc="upper right", fontsize=10, facecolor=_DARK_BG, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR, markerscale=5)
    ax_main.set_aspect("equal")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, facecolor=_DARK_BG, bbox_inches="tight")
        print(f"Saved Cluster Map: {save_path}")
    else:
        plt.show()
    return fig
