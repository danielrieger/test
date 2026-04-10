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

from smlm_score.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper
from smlm_score.utility.data_handling import (
    align_npc_cluster_pca,
    compute_av,
    flexible_filter_smlm_data,
    isolate_individual_npcs,
)
from smlm_score.utility.input import read_experimental_data, read_parameters_from_json
from smlm_score.utility.visualization import (
    _PUB_BG, _PUB_TEXT, _PUB_BLUE, _PUB_ORANGE,
    _style_axis_3d_pub, 
    plot_idealized_npc_3d, 
    plot_isosurface_3d
)

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
    return (arr - vmin) / (vmax - vmin)

def _get_top_clusters(coords, variances, labels, all_info, model, avs, n_top=1):
    rows = []
    candidates = [c for c in all_info if c["n_points"] >= 150]
    print(f"Ranking {len(candidates)} candidates...", flush=True)
    
    for candidate in candidates:
        cid = int(candidate["cluster_id"])
        mask = labels == cid
        pts = coords[mask]
        aligned = align_npc_cluster_pca(pts, debug=False)["aligned_data"]
        
        tree_wrapper = ScoringRestraintWrapper(
            model, avs, kdtree_obj=KDTree(aligned), dataxyz=aligned,
            var=variances[mask] if variances is not None else np.ones(len(pts)),
            type="Tree"
        )
        score_norm = float(tree_wrapper.evaluate()) / float(len(pts)**2)
        rows.append({"cluster_id": cid, "score_norm": score_norm, "pts": pts, "aligned": aligned})
    
    df = pd.DataFrame(rows)
    df["quality"] = _safe_minmax(df["score_norm"].values)
    return df.sort_values("quality", ascending=False).head(n_top)

def generate_figure(mode, cluster_data, model_pts, out_path):
    fig = plt.figure(figsize=(10, 10), facecolor=_PUB_BG)
    ax = fig.add_subplot(111, projection='3d')
    _style_axis_3d_pub(ax, title=f"NPC Stylized 3D Alignment — {mode.capitalize()} View")
    
    if mode == "ideal":
        # Mode 1: Pure Model geometry (16 subunits)
        plot_idealized_npc_3d(ax, model_pts, double_ring=True, s=250, alpha=0.8)
        
    elif mode == "surface":
        # Mode 2: Clean Isosurface of SMLM data
        plot_isosurface_3d(ax, cluster_data["aligned"], alpha=0.9, sigma=2.5)
        
    elif mode == "concept":
        # Mode 3: Overlay (Idealized Model + Ghostly Isosurface)
        plot_isosurface_3d(ax, cluster_data["aligned"], alpha=0.2, sigma=2.0, color=_PUB_BLUE)
        plot_idealized_npc_3d(ax, model_pts, double_ring=True, s=150, alpha=0.7)

    # Set view
    ax.view_init(elev=25, azim=45)
    limit = 80
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    fig.savefig(out_path, dpi=300, facecolor=_PUB_BG, bbox_inches="tight")
    print(f"  Saved: {out_path}", flush=True)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate stylized thesis figures.")
    parser.add_argument("--config", default="examples/pipeline_config.json")
    parser.add_argument("--output-dir", default="examples/figures/methodology")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = _load_config(args.config)
    
    print("\n--- Stylized 3D Visualization Startup ---", flush=True)
    
    # Load data
    raw_df = read_experimental_data(os.path.join(os.path.dirname(args.config), cfg["paths"]["smlm_data"]))
    coords, variances, _, _, _ = flexible_filter_smlm_data(
        raw_df, filter_type=cfg["filtering"]["type"], 
        percentage=cfg["filtering"]["random"]["size_percentage"]
    )
    npc_res = isolate_individual_npcs(coords, min_npc_points=120)
    
    # Model
    params = read_parameters_from_json(os.path.join(os.path.dirname(args.config), cfg["paths"]["av_parameters"]))
    avs, model, _ = compute_av(os.path.join(os.path.dirname(args.config), cfg["paths"]["pdb_data"]), params)
    model_coords_nm = np.array([np.array(p.get_coordinates()) * 0.1 for p in avs])
    model_centered = model_coords_nm - model_coords_nm.mean(axis=0)

    # Ranking
    top_df = _get_top_clusters(coords, variances, npc_res["labels"], npc_res["all_cluster_info"], model, avs)
    best = top_df.iloc[0]

    # Mode 1: Ideal
    generate_figure("ideal", best, model_centered, os.path.join(args.output_dir, "view_1_ideal.png"))
    # Mode 2: Surface
    generate_figure("surface", best, model_centered, os.path.join(args.output_dir, "view_2_surface.png"))
    # Mode 3: Concept
    generate_figure("concept", best, model_centered, os.path.join(args.output_dir, "view_3_concept.png"))

    print(f"\nAll stylized figures generated in: {args.output_dir}", flush=True)

if __name__ == "__main__":
    main()
