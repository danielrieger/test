"""
Visualization script for smlm_score results.

Run from the examples/ directory:
    python visualize_results.py

Produces three figures:
    1. density_heatmap.png   — KDE density of first valid NPC cluster
    2. score_comparison.png  — Bar chart valid vs noise (if scores exist)
    3. model_overlay.png     — Density contours + AV model positions

The script re-uses the same data loading / clustering / alignment pipeline
as NPC_example_BD.py but SKIPS scoring and sampling — runs in ~2 min.
"""

import sys
import os
import numpy as np

# Ensure the top-level Thesis folder is on PYTHONPATH so smlm_score imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import IMP.core

from smlm_score.utility.input import read_experimental_data, read_parameters_from_json
from smlm_score.utility.data_handling import (
    isolate_individual_npcs,
    align_npc_cluster_pca,
    flexible_filter_smlm_data,
    compute_av,
)
from smlm_score.utility.visualization import (
    plot_density_2d,
    plot_score_comparison,
    plot_density_contour,
)
from smlm_score.imp_modeling.scoring.gmm_score import test_gmm_components

# ── paths (relative to examples/) ────────────────────────────────────────
SMLM_DATA_PATH = "ShareLoc_Data/data.csv"
PDB_DATA_PATH = "PDB_Data/7N85-assembly1.cif"
PARAM_PATH = "av_parameter.json"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────
print("=== Loading data ===")
data_xyz = read_experimental_data(SMLM_DATA_PATH)
if data_xyz is None:
    sys.exit("ERROR: could not load SMLM data")

params = read_parameters_from_json(PARAM_PATH)

# ── filter SMLM data ─────────────────────────────────────────────────────
print("\n=== Filtering SMLM data ===")
smlm_coordinates, smlm_variances, smlm_coords_tree, kdtree_obj, _ = flexible_filter_smlm_data(
    data_xyz,
    filter_type='cut',
    x_cut=(10000, 12000),
    y_cut=(0, 5000),
    fill_z_value=0.0,
    return_tree=True,
)
data_for_clustering = smlm_coords_tree.copy()

# ── create IMP model + AVs ───────────────────────────────────────────────
print("\n=== Computing Accessible Volumes ===")
avs, m, hier = compute_av(PDB_DATA_PATH, params)
print(f"Computed {len(avs)} AVs.")

# Scaled model AV coordinates (Å → nm)
model_coords_nm = np.array([
    np.array(IMP.core.XYZ(av).get_coordinates()) * 0.1
    for av in avs
])

# ── clustering with HDBSCAN ──────────────────────────────────────────────
print("\n=== Clustering with HDBSCAN ===")
npc_results = isolate_individual_npcs(
    data_for_clustering,
    min_cluster_size=15,
    min_npc_points=100,
    debug=True,
)
cluster_labels = npc_results['labels']
cluster_info = npc_results['all_cluster_info']
valid_clusters = npc_results['npc_info']
noise_clusters = [c for c in cluster_info if 10 <= c['n_points'] < 100]

print(f"Found {len(valid_clusters)} valid NPCs, {len(noise_clusters)} noise clusters.\n")

if not valid_clusters:
    sys.exit("No valid NPCs found.")


# =========================================================================
# Figure 1:  Density heatmap of the first valid cluster
# =========================================================================
best_cluster = max(valid_clusters, key=lambda c: c["n_points"])
cid = best_cluster["cluster_id"]
mask = cluster_labels == cid
cluster_pts = data_for_clustering[mask]

alignment = align_npc_cluster_pca(cluster_pts, debug=False)
aligned = alignment["aligned_data"]

print(f"--- Figure 1: Density of cluster {cid} ({len(aligned)} points) ---")
plot_density_2d(
    aligned,
    title=f"NPC Cluster {cid} — Localization Density ({len(aligned)} pts)",
    save_path=os.path.join(OUTPUT_DIR, "density_heatmap.png"),
)


# =========================================================================
# Figure 3:  Density contour + AV model overlay
# =========================================================================
# Compute the offset to align model coords onto data centroid
model_centroid = model_coords_nm.mean(axis=0)
data_centroid = aligned.mean(axis=0)
offset = data_centroid - model_centroid
av_positions_in_data_frame = model_coords_nm + offset

# Optionally fit GMM for this cluster
print(f"\n--- Fitting GMM for cluster {cid} (for overlay ellipses) ---")
try:
    _, gmm_obj, gmm_means, gmm_covs, gmm_weights = test_gmm_components(aligned)
    print(f"GMM: {gmm_obj.n_components} components selected.")
except Exception as e:
    print(f"GMM fitting failed ({e}); plotting without GMM ellipses.")
    gmm_means = None
    gmm_covs = None

print(f"\n--- Figure 2: Model overlay for cluster {cid} ---")
plot_density_contour(
    aligned,
    av_positions_in_data_frame,
    gmm_means=gmm_means,
    gmm_covs=gmm_covs,
    title=f"Cluster {cid} — Data Density + Model AV Positions",
    save_path=os.path.join(OUTPUT_DIR, "model_overlay.png"),
)


# =========================================================================
# Figure 2:  Score comparison (uses pre-computed scores if available)
# =========================================================================
# Quick scoring: compute Tree scores for a few valid + noise clusters
print("\n--- Figure 3: Score comparison ---")
from sklearn.neighbors import KDTree
from smlm_score.imp_modeling.scoring.tree_score import computescoretree

cluster_scores = {}

# Score up to 3 valid + 3 noise clusters with Tree scoring (fast)
eval_clusters = valid_clusters[:3] + noise_clusters[:3]
for cl in eval_clusters:
    c_id = cl["cluster_id"]
    c_mask = cluster_labels == c_id
    c_pts = data_for_clustering[c_mask]
    c_vars = smlm_variances[c_mask]

    c_align = align_npc_cluster_pca(c_pts, debug=False)
    c_aligned = c_align["aligned_data"]

    c_model_centroid = model_coords_nm.mean(axis=0)
    c_data_centroid = c_aligned.mean(axis=0)
    c_offset = c_data_centroid - c_model_centroid

    # Build KDTree and score
    tree = KDTree(c_aligned)
    score = computescoretree(
        tree=tree,
        modelavs=avs,
        dataxyz=c_aligned,
        var=c_vars,
        scaling=0.1,
        searchradius=50.0,
        offsetxyz=c_offset,
    )

    c_type = "Valid" if cl["n_points"] >= 100 else "Noise"
    cluster_scores[c_id] = {
        "type": c_type,
        "n_points": len(c_aligned),
        "Tree": score,
    }
    print(f"  Cluster {c_id} ({c_type}, {len(c_aligned)} pts): Tree score = {score:.2f}")

if cluster_scores:
    plot_score_comparison(
        cluster_scores,
        normalize_per_point=True,
        save_path=os.path.join(OUTPUT_DIR, "score_comparison.png"),
    )


print(f"\n=== Done! Figures saved to {os.path.abspath(OUTPUT_DIR)}/ ===")
