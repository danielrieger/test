import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Ensure imports work when run directly from examples/
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
THESIS_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

from smlm_score.utility.input import read_experimental_data
from smlm_score.utility.data_handling import (
    flexible_filter_smlm_data,
    isolate_individual_npcs,
    align_npc_cluster_pca,
)


def _cluster_colors(labels):
    """Deterministic color mapping with noise in light gray."""
    unique_labels = sorted(set(labels) - {-1})
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    color_map = {-1: "#cfcfcf"}
    for i, cid in enumerate(unique_labels):
        color_map[cid] = cmap(i)
    return np.array([color_map[lbl] for lbl in labels], dtype=object)


def _make_cluster_map(points, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(8.8, 7.4), facecolor="white")
    ax.set_facecolor("white")
    colors = _cluster_colors(labels)
    ax.scatter(points[:, 0], points[:, 1], s=1.0, c=colors, alpha=0.9, rasterized=True)
    ax.set_xlabel("X [nm]")
    ax.set_ylabel("Y [nm]")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    ax.text(
        0.02,
        0.02,
        f"Clusters: {n_clusters}\nNoise points: {n_noise}",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#777", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _make_alignment_figure(raw_pts, aligned_pts, subunit_labels, target_id, out_path):
    fig = plt.figure(figsize=(13.2, 6.2), facecolor="white")
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)

    cmap = plt.get_cmap("tab10")
    ax1.scatter(
        raw_pts[:, 0],
        raw_pts[:, 1],
        raw_pts[:, 2],
        c=subunit_labels,
        cmap=cmap,
        s=12,
        alpha=0.9,
    )
    ax1.set_title(f"Raw NPC (ID {target_id})")
    ax1.set_xlabel("X [nm]")
    ax1.set_ylabel("Y [nm]")
    ax1.set_zlabel("Z [nm]")
    ax1.grid(alpha=0.3)

    ax2.scatter(
        aligned_pts[:, 0],
        aligned_pts[:, 1],
        c=subunit_labels,
        cmap=cmap,
        s=12,
        alpha=0.9,
        rasterized=True,
    )
    ax2.add_patch(Circle((0, 0), 60, fill=False, linestyle="--", linewidth=1.5, color="#cc2f2f"))
    ax2.set_title("PCA-aligned NPC (XY projection)")
    ax2.set_xlabel("X [nm]")
    ax2.set_ylabel("Y [nm]")
    ax2.grid(alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")

    z_std_raw = np.std(raw_pts[:, 2])
    z_std_aligned = np.std(aligned_pts[:, 2])
    ax2.text(
        0.02,
        0.02,
        f"Z spread (std):\nRaw={z_std_raw:.2f} nm\nAligned={z_std_aligned:.2f} nm",
        transform=ax2.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#777", alpha=0.9),
    )

    fig.suptitle("NPC Alignment Demonstration", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("Loading data...")
    smlm_data_path = os.path.join(THIS_DIR, "ShareLoc_Data", "data.csv")
    raw_df = read_experimental_data(smlm_data_path)
    coords, _, _, _, _ = flexible_filter_smlm_data(
        raw_df,
        filter_type="cut",
        x_cut=(10000, 11500),
        y_cut=(1000, 3500),
        fill_z_value=0.0,
    )

    print("Running raw HDBSCAN stage...")
    results_raw = isolate_individual_npcs(
        coords,
        min_cluster_size=15,
        perform_geometric_merging=False,
        min_samples=5,
        cluster_selection_method='leaf'
    )

    print("Running second clustering stage (geometric merging)...")
    results_geo = isolate_individual_npcs(
        coords,
        min_cluster_size=15,
        perform_geometric_merging=True,
        min_samples=5,
        cluster_selection_method='leaf'
    )

    out_dir = os.path.join(THESIS_ROOT, "smlm_score", "examples", "figures", "qc")
    os.makedirs(out_dir, exist_ok=True)

    _make_cluster_map(
        coords,
        results_raw["labels"],
        "NPC Map After HDBSCAN (Raw Fragments)",
        os.path.join(out_dir, "npc_map_after_hdbscan.png"),
    )

    _make_cluster_map(
        coords,
        results_geo["labels"],
        "NPC Map After Second Clustering (Geometric Merging)",
        os.path.join(out_dir, "npc_map_after_second_clustering.png"),
    )

    target_id = results_geo["npc_info"][0]["cluster_id"]
    mask = results_geo["labels"] == target_id
    raw_pts = coords[mask]
    subunit_labels = results_raw["labels"][mask]
    aligned_pts = align_npc_cluster_pca(raw_pts)["aligned_data"]

    _make_alignment_figure(
        raw_pts,
        aligned_pts,
        subunit_labels,
        target_id,
        os.path.join(out_dir, "npc_alignment_visual.png"),
    )

    print(f"Done. Processed {results_geo['n_npcs']} NPCs.")


if __name__ == "__main__":
    main()
