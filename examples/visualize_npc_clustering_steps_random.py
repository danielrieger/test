import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, HDBSCAN

# Ensure package imports work from both repo root and examples/ execution.
_examples_dir = os.path.dirname(__file__)
_project_dir = os.path.abspath(os.path.join(_examples_dir, ".."))
_workspace_dir = os.path.abspath(os.path.join(_project_dir, ".."))
for _p in (_workspace_dir, _project_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from smlm_score.utility.data_handling import flexible_filter_smlm_data
from smlm_score.utility.input import read_experimental_data


POINTWISE_MERGE_THRESHOLD = 5000
GEOMETRIC_MERGE_DISTANCE_NM = 140.0


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _colorize_labels(labels, noise_color=(0.8, 0.8, 0.8, 0.45)):
    unique = sorted([cid for cid in set(labels.tolist()) if cid != -1])
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    color_lookup = {cid: cmap(i) for i, cid in enumerate(unique)}
    return [noise_color if cid == -1 else color_lookup[cid] for cid in labels]


def _scatter(ax, pts_xy, labels, title, annotate=False):
    colors = _colorize_labels(labels)
    ax.scatter(pts_xy[:, 0], pts_xy[:, 1], c=colors, s=3, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linewidth=0.4)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    ax.text(
        0.01,
        0.99,
        f"clusters={n_clusters}\nnoise={n_noise}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    if annotate:
        for cid in sorted([x for x in set(labels.tolist()) if x != -1]):
            mask = labels == cid
            if not np.any(mask):
                continue
            center = pts_xy[mask].mean(axis=0)
            ax.text(center[0], center[1], str(cid), fontsize=7, ha="center", va="center")


def _run_hdbscan(points_xy, min_cluster_size):
    if len(points_xy) < 2 or min_cluster_size > len(points_xy):
        return np.full(len(points_xy), -1, dtype=int)
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        copy=False,
    )
    return hdb.fit_predict(points_xy)


def _geometric_merge(points_xy, raw_labels, perform_merge):
    if not perform_merge:
        return raw_labels.copy(), "disabled"

    labels = raw_labels.copy()
    clean_mask = labels != -1
    if not np.any(clean_mask):
        return labels, "no clean clusters"

    clean_pts = points_xy[clean_mask]
    if len(clean_pts) <= POINTWISE_MERGE_THRESHOLD:
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=GEOMETRIC_MERGE_DISTANCE_NM,
            linkage="complete",
        )
        macro_labels = agg.fit_predict(clean_pts)
        new_labels = np.full(len(points_xy), -1, dtype=int)
        new_labels[clean_mask] = macro_labels
        return new_labels, f"point-wise merge ({len(clean_pts)} clean points)"

    clean_cluster_ids = np.array(sorted(set(labels[clean_mask])), dtype=int)
    if len(clean_cluster_ids) <= 1:
        return labels, "single clean cluster"

    centroids = np.array(
        [points_xy[labels == cid].mean(axis=0) for cid in clean_cluster_ids], dtype=np.float64
    )
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=GEOMETRIC_MERGE_DISTANCE_NM,
        linkage="complete",
    )
    macro_cluster_labels = agg.fit_predict(centroids)

    old_to_macro = {int(old): int(new) for old, new in zip(clean_cluster_ids, macro_cluster_labels)}
    new_labels = np.full(len(points_xy), -1, dtype=int)
    for old_cid, new_cid in old_to_macro.items():
        new_labels[labels == old_cid] = new_cid
    return new_labels, f"cluster-level merge ({len(clean_pts)} clean points, {len(clean_cluster_ids)} clusters)"


def _classify_valid(labels, points_xyz, min_npc_points):
    # Keep only clusters with enough points as "valid NPCs"; all else becomes noise for this view.
    output = np.full(len(labels), -1, dtype=int)
    next_id = 0
    for cid in sorted([x for x in set(labels.tolist()) if x != -1]):
        mask = labels == cid
        if int(np.sum(mask)) >= min_npc_points:
            output[mask] = next_id
            next_id += 1
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Visualize random-filtered NPC clustering step by step for thesis figures."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "pipeline_config.json"),
        help="Path to pipeline_config.json",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "figures", "clustering_steps"),
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible random filter window. Default: random each run.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    paths = cfg["paths"]
    filt = cfg.get("filtering", {})
    clust = cfg.get("clustering", {})

    filter_type = filt.get("type", "none")
    f_cut = filt.get("filter", {})
    f_random = filt.get("random", {})
    percentage = f_random.get("size_percentage")

    x_range = tuple(f_cut.get("x_range")) if f_cut.get("x_range") else None
    y_range = tuple(f_cut.get("y_range")) if f_cut.get("y_range") else None
    z_range = tuple(f_cut.get("z_range")) if f_cut.get("z_range") else None

    data_path = os.path.join(os.path.dirname(args.config), paths["smlm_data"])
    raw_df = read_experimental_data(data_path)
    if raw_df is None:
        raise RuntimeError(f"Could not read SMLM data: {data_path}")

    xyz, _, _, _, applied_cuts = flexible_filter_smlm_data(
        raw_df,
        filter_type=filter_type,
        x_cut=x_range,
        y_cut=y_range,
        z_cut=z_range,
        percentage=percentage,
        fill_z_value=0.0,
        random_seed=args.seed,
        return_tree=False,
    )
    if len(xyz) == 0:
        raise RuntimeError("Random filtering produced zero points. Try increasing size_percentage.")

    points_xy = xyz[:, :2]
    min_cluster_size = int(clust.get("min_cluster_size", 15))
    min_npc_points = int(clust.get("min_npc_points", 100))
    do_merge = bool(clust.get("perform_geometric_merging", True))

    labels_hdb = _run_hdbscan(points_xy, min_cluster_size=min_cluster_size)
    labels_merged, merge_note = _geometric_merge(points_xy, labels_hdb, perform_merge=do_merge)
    labels_valid = _classify_valid(labels_merged, xyz, min_npc_points=min_npc_points)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 0: Random filtered map
    fig0, ax0 = plt.subplots(figsize=(8, 7))
    ax0.scatter(points_xy[:, 0], points_xy[:, 1], c="#2f4f4f", s=2, alpha=0.55, linewidths=0, rasterized=True)
    ax0.set_title("Step 0: Random Filtered Map")
    ax0.set_xlabel("x [nm]")
    ax0.set_ylabel("y [nm]")
    ax0.set_aspect("equal")
    ax0.grid(alpha=0.25, linewidth=0.4)
    ax0.text(
        0.01,
        0.99,
        f"n_points={len(points_xy)}\nwindow={applied_cuts}",
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    fig0.tight_layout()
    fig0.savefig(os.path.join(args.output_dir, "step0_random_filtered.png"), dpi=350, bbox_inches="tight")
    plt.close(fig0)

    # Step 1: Raw HDBSCAN output
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    _scatter(
        ax1,
        points_xy,
        labels_hdb,
        f"Step 1: HDBSCAN (min_cluster_size={min_cluster_size})",
        annotate=False,
    )
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.output_dir, "step1_hdbscan_raw.png"), dpi=350, bbox_inches="tight")
    plt.close(fig1)

    # Step 2: Geometric merging
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    _scatter(
        ax2,
        points_xy,
        labels_merged,
        f"Step 2: Geometric Merge ({merge_note})",
        annotate=False,
    )
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "step2_geometric_merge.png"), dpi=350, bbox_inches="tight")
    plt.close(fig2)

    # Step 3: Final NPC-sized clusters only
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    _scatter(
        ax3,
        points_xy,
        labels_valid,
        f"Step 3: NPC-Sized Clusters (>= {min_npc_points} points)",
        annotate=True,
    )
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.output_dir, "step3_npc_sized_clusters.png"), dpi=350, bbox_inches="tight")
    plt.close(fig3)

    # Combined panel for thesis comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    axes[0].scatter(points_xy[:, 0], points_xy[:, 1], c="#2f4f4f", s=1.5, alpha=0.5, linewidths=0, rasterized=True)
    axes[0].set_title("Step 0: Random Filtered Map", fontsize=11, fontweight="bold")
    axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.2, linewidth=0.4)
    axes[0].set_xlabel("x [nm]")
    axes[0].set_ylabel("y [nm]")

    _scatter(axes[1], points_xy, labels_hdb, "Step 1: Raw HDBSCAN", annotate=False)
    _scatter(axes[2], points_xy, labels_merged, "Step 2: Geometric Merge", annotate=False)
    _scatter(axes[3], points_xy, labels_valid, "Step 3: NPC-Sized Clusters", annotate=False)

    fig.suptitle("NPC Clustering Pipeline (Random Map Per Run)", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "clustering_steps_overview.png"), dpi=350, bbox_inches="tight")
    plt.close(fig)

    # Console summary
    n_hdb = len(set(labels_hdb.tolist())) - (1 if -1 in labels_hdb else 0)
    n_merge = len(set(labels_merged.tolist())) - (1 if -1 in labels_merged else 0)
    n_valid = len(set(labels_valid.tolist())) - (1 if -1 in labels_valid else 0)

    print("Saved clustering step figures to:", args.output_dir)
    print("Current clustering process has 3 clustering steps after filtering:")
    print("  1) HDBSCAN over XY coordinates")
    print("  2) Geometric merging (complete-linkage, distance_threshold=140 nm)")
    print(f"     mode: {merge_note}")
    print(f"  3) NPC-size selection (clusters with >= {min_npc_points} points)")
    print(f"Cluster counts -> Step1: {n_hdb}, Step2: {n_merge}, Step3: {n_valid}")


if __name__ == "__main__":
    main()
