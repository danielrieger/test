import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree

# Ensure imports work from repo root execution.
_examples_dir = os.path.dirname(__file__)
_project_dir = os.path.abspath(os.path.join(_examples_dir, ".."))
_thesis_dir = os.path.abspath(os.path.join(_project_dir, ".."))
if _thesis_dir not in sys.path:
    sys.path.insert(0, _thesis_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from smlm_score.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper
from smlm_score.utility.data_handling import (
    align_npc_cluster_pca,
    compute_av,
    flexible_filter_smlm_data,
    isolate_individual_npcs,
)
from smlm_score.utility.input import read_experimental_data, read_parameters_from_json
from smlm_score.utility.visualization import _DARK_BG, _TEXT_COLOR, _GRID_COLOR, _ACCENT_CYAN, _ACCENT_ORANGE, _style_axis_3d

print("\n--- NPC 3D Visualization Pipeline Starting ---", flush=True)
print(f"Time: {datetime.now().strftime('%H:%M:%S')}", flush=True)

# Re-use the ranking calculation logic from analyze_target_cluster_candidates.py
def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_minmax(values):
    arr = np.array(values, dtype=float)
    if len(arr) == 0: return arr
    finite = np.isfinite(arr)
    if not np.any(finite): return np.full_like(arr, 0.5)
    vmin, vmax = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    if abs(vmax - vmin) < 1e-12: return np.full_like(arr, 0.5)
    out = (arr - vmin) / (vmax - vmin)
    out[~finite] = 0.0
    return out

def _geometry_quality(aligned_points, expected_diameter_nm=120.0):
    xy = aligned_points[:, :2]
    bbox = xy.max(axis=0) - xy.min(axis=0)
    width, height = float(bbox[0]), float(bbox[1])
    mean_diameter = 0.5 * (width + height)
    circularity = min(width, height) / max(width, height, 1e-9)
    diameter_score = np.exp(-((mean_diameter - expected_diameter_nm)**2) / (2.0 * 25.0**2))
    return 0.7 * diameter_score + 0.3 * circularity

def _get_top_clusters(coords, variances, labels, all_info, model, avs, model_centered, n_top=4):
    rows = []
    candidates = [c for c in all_info if c["n_points"] >= 100]
    # Simple ranking for this vis tool
    for i, candidate in enumerate(candidates, start=1):
        cid = int(candidate["cluster_id"])
        print(f"  [{i}/{len(candidates)}] Processing Cluster {cid} ({candidate['n_points']} points)...", flush=True)
        mask = labels == cid
        pts = coords[mask]
        align = align_npc_cluster_pca(pts, debug=False)
        aligned = align["aligned_data"]
        
        # Geometry
        geom_score = _geometry_quality(aligned)
        
        # Scoring speed-up: only Tree for selection
        tree_wrapper = ScoringRestraintWrapper(
            model, avs, kdtree_obj=KDTree(aligned), dataxyz=aligned,
            var=variances[mask] if variances is not None else np.ones(len(pts)),
            type="Tree"
        )
        score_norm = float(tree_wrapper.evaluate()) / float(len(pts)**2)
        
        rows.append({"cluster_id": cid, "score_norm": score_norm, "geom_score": geom_score, "pts": pts, "aligned": aligned})
    
    df = pd.DataFrame(rows)
    df["score_scaled"] = _safe_minmax(df["score_norm"].values)
    df["quality"] = 0.6 * df["score_scaled"] + 0.4 * df["geom_score"]
    return df.sort_values("quality", ascending=False).head(n_top)

def _plot_cluster_3d(ax, aligned_pts, model_pts, title=""):
    _style_axis_3d(ax, title=title)
    
    # Data points colored by depth (Z) for 3D effect
    sc = ax.scatter(aligned_pts[:, 0], aligned_pts[:, 1], aligned_pts[:, 2], 
                    c=aligned_pts[:, 2], cmap="plasma", s=2, alpha=0.6, linewidths=0)
    
    # Model points (Discrete markers as requested)
    ax.scatter(model_pts[:, 0], model_pts[:, 1], model_pts[:, 2], 
               c=_ACCENT_ORANGE, s=80, marker="o", edgecolors="white", alpha=0.8, label="Model AVs")
    
    # Zoom to fit
    limit = 80
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

def create_detailed_view(best_cluster, model_pts, out_path):
    fig = plt.figure(figsize=(12, 10), facecolor=_DARK_BG)
    
    # Main 3D view
    ax_3d = fig.add_subplot(111, projection='3d')
    _plot_cluster_3d(ax_3d, best_cluster["aligned"], model_pts, title="NPC Structural Alignment — Detailed View")
    ax_3d.view_init(elev=25, azim=45)
    
    # Projections
    pts = best_cluster["aligned"]
    
    # XY
    ax_xy = fig.add_axes([0.05, 0.05, 0.2, 0.2], facecolor=_DARK_BG)
    ax_xy.scatter(pts[:, 0], pts[:, 1], c=_ACCENT_CYAN, s=0.5, alpha=0.4)
    ax_xy.scatter(model_pts[:, 0], model_pts[:, 1], c=_ACCENT_ORANGE, s=10)
    ax_xy.set_title("XY Projection", color=_TEXT_COLOR, fontsize=8)
    ax_xy.set_aspect('equal')
    ax_xy.axis('off')
    
    # XZ
    ax_xz = fig.add_axes([0.25, 0.05, 0.2, 0.2], facecolor=_DARK_BG)
    ax_xz.scatter(pts[:, 0], pts[:, 2], c=_ACCENT_CYAN, s=0.5, alpha=0.4)
    ax_xz.scatter(model_pts[:, 0], model_pts[:, 2], c=_ACCENT_ORANGE, s=10)
    ax_xz.set_title("XZ Projection", color=_TEXT_COLOR, fontsize=8)
    ax_xz.axis('off')
    
    fig.savefig(out_path, dpi=300, facecolor=_DARK_BG, bbox_inches="tight")
    print(f"  Saved: {os.path.basename(out_path)}")
    plt.close(fig)

def create_grid_view(top_df, model_pts, out_path):
    fig = plt.figure(figsize=(15, 12), facecolor=_DARK_BG)
    
    for i, (_, row) in enumerate(top_df.iterrows()):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        _plot_cluster_3d(ax, row["aligned"], model_pts, title=f"Top Match #{i+1} (Cluster {int(row['cluster_id'])})")
        ax.view_init(elev=20, azim=30)
    
    fig.suptitle("Consistent NPC Structural Alignment Across Dataset", color=_TEXT_COLOR, fontsize=18, fontweight="bold", y=0.95)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=300, facecolor=_DARK_BG, bbox_inches="tight")
    print(f"  Saved: {os.path.basename(out_path)}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Thesis-ready 3D NPC Alignment Visualization.")
    parser.add_argument("--config", default="examples/pipeline_config.json")
    parser.add_argument("--output-dir", default="examples/figures/methodology")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = _load_config(args.config)
    
    # Data loading
    print("Loading data and model...")
    raw_df = read_experimental_data(os.path.join(os.path.dirname(args.config), cfg["paths"]["smlm_data"]))
    coords, variances, _, _, _ = flexible_filter_smlm_data(
        raw_df, filter_type=cfg["filtering"]["type"], 
        percentage=cfg["filtering"]["random"]["size_percentage"],
        random_seed=42
    )
    
    # Clustering
    print("Isolating NPC candidates...")
    npc_res = isolate_individual_npcs(coords, min_npc_points=120, debug=False)
    
    # Model Setup
    params = read_parameters_from_json(os.path.join(os.path.dirname(args.config), cfg["paths"]["av_parameters"]))
    avs, model, _ = compute_av(os.path.join(os.path.dirname(args.config), cfg["paths"]["pdb_data"]), params)
    model_coords_nm = np.array([np.array(p.get_coordinates()) * 0.1 for p in avs])
    model_centered = model_coords_nm - model_coords_nm.mean(axis=0)

    # Ranking
    print("Ranking and aligning clusters...")
    top_df = _get_top_clusters(coords, variances, npc_res["labels"], npc_res["all_cluster_info"], model, avs, model_centered)

    # Visualization
    print("Generating Detailed View...")
    create_detailed_view(top_df.iloc[0], model_centered, os.path.join(args.output_dir, "alignment_detailed_3d.png"))
    
    print("Generating Grid View...")
    create_grid_view(top_df, model_centered, os.path.join(args.output_dir, "alignment_grid_3d.png"))

    print(f"\nSuccess! Figures generated in: {args.output_dir}", flush=True)

if __name__ == "__main__":
    main()
