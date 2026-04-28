import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure imports work from repo root execution.
_examples_dir = os.path.dirname(__file__)
_project_dir = os.path.abspath(os.path.join(_examples_dir, ".."))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from smlm_score.utility.input import read_experimental_data
from smlm_score.utility.data_handling import align_npc_cluster_pca, flexible_filter_smlm_data, isolate_individual_npcs, isolate_npcs_from_eman2_boxes

# Styling
_DARK_BG = "#0d1117"
_TEXT_COLOR = "#c9d1d9"
_TICK_COLOR = "#8b949e"
_GRID_COLOR = "#30363d"
_ACCENT_CYAN = "#00ffff"
_ACCENT_ORANGE = "#ff8c00"

def _style_axis(ax, title=""):
    ax.set_facecolor(_DARK_BG)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)
    if title:
        ax.set_title(title, color=_TEXT_COLOR, fontsize=12, pad=10)
    ax.grid(color=_GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.5)

def main():
    smlm_data_path = os.path.join(_examples_dir, "ShareLoc_Data", "data.csv")
    run_dir = os.path.join(_project_dir, "bayesian_cluster_484")
    av_path = os.path.join(run_dir, "av_coordinates_final.csv")
    scores_path = os.path.join(run_dir, "frame_scores.csv")
    out_path = os.path.join(_examples_dir, "figures", "methodology", "fitted_npc_result.png")
    
    print("Loading SMLM data...")
    raw_df = read_experimental_data(smlm_data_path)
    
    # Run identical filtering as pipeline
    config_path = os.path.join(_examples_dir, "pipeline_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    f_type = config.get("filtering", {}).get("type", "none")
    f_params = config.get("filtering", {}).get(f_type, {})
    x_r = tuple(f_params.get("x_range")) if f_params.get("x_range") else None
    y_r = tuple(f_params.get("y_range")) if f_params.get("y_range") else None
    z_r = tuple(f_params.get("z_range")) if f_params.get("z_range") else None
    
    coords, vars_, _, _, cuts = flexible_filter_smlm_data(
        raw_df, filter_type=f_type, 
        x_cut=x_r, y_cut=y_r, z_cut=z_r,
        percentage=config.get("filtering", {}).get("random", {}).get("size_percentage"),
        fill_z_value=0.0, return_tree=False, random_seed=42
    )
    
    print("Clustering data...")
    if config["clustering"]["method"] == "eman2":
        res = isolate_npcs_from_eman2_boxes(coords, os.path.join(_examples_dir, config["clustering"]["eman2_boxes"]), os.path.join(_examples_dir, config["clustering"].get("pixel_map", "pixel_map.json")))
    else:
        res = isolate_individual_npcs(coords, min_cluster_size=config["clustering"]["min_cluster_size"])
        
    target_id = 484
    labels = res['labels']
    mask = (labels == target_id)
    cluster_pts = coords[mask]
    
    if len(cluster_pts) == 0:
        print(f"Error: Cluster {target_id} not found. Try running with 'random' target cluster.")
        # Fallback to the largest valid cluster
        target_id = res['npc_info'][0]['cluster_id']
        mask = (labels == target_id)
        cluster_pts = coords[mask]
        print(f"Fallback: using cluster {target_id} with {len(cluster_pts)} points.")
        
    align_res = align_npc_cluster_pca(cluster_pts, debug=False)
    aligned_cluster = align_res['aligned_data']

    print("Loading optimized AV model...")
    av_df = pd.read_csv(av_path)
    av_coords_angstrom = av_df[['x', 'y', 'z']].values.astype(np.float64)
    # The saved coordinates are in Angstroms, in the PDB's original world space,
    # but rotated/translated by the rigid body. 
    # Since the scoring restraint aligns model_aligned (which is in nm, centered at 0) 
    # to the PCA-aligned data (which is in nm, centered at 0), 
    # the relative orientation is captured in these coordinates.
    # We just need to center them and scale to nm.
    av_coords_nm = av_coords_angstrom * 0.1
    av_centroid = av_coords_nm.mean(axis=0)
    aligned_av = av_coords_nm - av_centroid

    print("Loading score trajectory...")
    scores_df = pd.read_csv(scores_path)

    print("Generating figure...")
    fig = plt.figure(figsize=(18, 6), facecolor=_DARK_BG)
    plt.subplots_adjust(wspace=0.3)

    # Panel A: XY Top View
    ax_xy = fig.add_subplot(1, 3, 1)
    _style_axis(ax_xy, "A) Top View (XY) — Fitted Ring")
    ax_xy.scatter(aligned_cluster[:, 0], aligned_cluster[:, 1], c=_ACCENT_ORANGE, s=8, alpha=0.5, label="SMLM Data")
    
    # Draw ring lines
    av_xs = np.append(aligned_av[:, 0], aligned_av[0, 0])
    av_ys = np.append(aligned_av[:, 1], aligned_av[0, 1])
    ax_xy.plot(av_xs, av_ys, color=_ACCENT_CYAN, lw=2, alpha=0.8)
    ax_xy.scatter(aligned_av[:, 0], aligned_av[:, 1], c=_ACCENT_CYAN, s=80, marker='o', edgecolors='white', zorder=5, label="Optimized Model")
    
    ax_xy.set_xlabel("X (nm)", color=_TEXT_COLOR)
    ax_xy.set_ylabel("Y (nm)", color=_TEXT_COLOR)
    ax_xy.set_aspect('equal')
    # Limit axis to -80, 80 for consistent scale
    ax_xy.set_xlim(-80, 80)
    ax_xy.set_ylim(-80, 80)
    ax_xy.legend(facecolor=_DARK_BG, labelcolor=_TEXT_COLOR, edgecolor=_GRID_COLOR)

    # Panel B: XZ Side View
    ax_xz = fig.add_subplot(1, 3, 2)
    _style_axis(ax_xz, "B) Side View (XZ) — 3D Alignment")
    ax_xz.scatter(aligned_cluster[:, 0], aligned_cluster[:, 2], c=_ACCENT_ORANGE, s=8, alpha=0.5)
    
    # Side view doesn't form a neat circle, just plot the beads
    ax_xz.scatter(aligned_av[:, 0], aligned_av[:, 2], c=_ACCENT_CYAN, s=80, marker='o', edgecolors='white', zorder=5)
    
    ax_xz.set_xlabel("X (nm)", color=_TEXT_COLOR)
    ax_xz.set_ylabel("Z (nm)", color=_TEXT_COLOR)
    ax_xz.set_xlim(-80, 80)
    ax_xz.set_ylim(-80, 80)
    ax_xz.set_aspect('equal', adjustable='datalim')

    # Panel C: Score Convergence
    ax_score = fig.add_subplot(1, 3, 3)
    _style_axis(ax_score, "C) Bayesian Score Optimization")
    ax_score.plot(scores_df['frame'], scores_df['score'], color=_ACCENT_CYAN, lw=2)
    
    best_idx = scores_df['score'].idxmin()
    best_score = scores_df['score'].min()
    ax_score.scatter([scores_df['frame'].iloc[best_idx]], [best_score], color=_ACCENT_ORANGE, s=100, zorder=5, label=f"Best (LL={best_score:.2f})")
    
    ax_score.set_xlabel("Replica Exchange Step", color=_TEXT_COLOR)
    ax_score.set_ylabel("Negative Log-Likelihood", color=_TEXT_COLOR)
    ax_score.legend(facecolor=_DARK_BG, labelcolor=_TEXT_COLOR, edgecolor=_GRID_COLOR)

    fig.suptitle(f"SMLM-IMP: Bayesian Optimization Result (Cluster {target_id})", color='white', fontsize=18, fontweight='bold', y=0.98)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=_DARK_BG, bbox_inches='tight')
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    main()
