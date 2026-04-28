import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_examples_dir = os.path.dirname(__file__)
_project_dir = os.path.abspath(os.path.join(_examples_dir, ".."))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import IMP
import IMP.core
import IMP.atom
from sklearn.neighbors import KDTree
from scipy.stats import gaussian_kde

from smlm_score.utility.input import read_experimental_data, read_parameters_from_json
from smlm_score.utility.data_handling import flexible_filter_smlm_data, isolate_npcs_from_eman2_boxes, compute_av
from smlm_score.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper
from smlm_score.imp_modeling.simulation.frequentist_optimizer import run_frequentist_optimization

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
    ax.tick_params(colors=_TICK_COLOR, labelsize=10)
    if title:
        ax.set_title(title, color=_TEXT_COLOR, fontsize=14, pad=12)
    ax.grid(color=_GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.5)

def main():
    target_id = 497
    print(f"Loading data for Box {target_id}...")
    
    smlm_path = os.path.join(_examples_dir, "ShareLoc_Data", "data.csv")
    boxes_path = os.path.join(_examples_dir, "info", "micrograph_info.json")
    pixel_map_path = os.path.join(_examples_dir, "pixel_map.json")
    
    raw_df = read_experimental_data(smlm_path)
    coords, vars_, _, _, _ = flexible_filter_smlm_data(raw_df, filter_type="full", fill_z_value=0.0)
    
    # Extract specifically our target box
    res = isolate_npcs_from_eman2_boxes(coords, boxes_path, pixel_map_path, min_npc_points=50)
    mask = (res['labels'] == target_id)
    if not np.any(mask):
        print(f"Error: Box {target_id} not found in extraction!")
        return
        
    pts = coords[mask]
    vars_target = vars_[mask]
    print(f"Isolated {len(pts)} points for Box {target_id}.")
    
    # Center (NO PCA)
    centroid = pts.mean(axis=0)
    aligned_cluster = pts - centroid
    
    print("Setting up IMP model...")
    params = read_parameters_from_json(os.path.join(_examples_dir, "av_parameter.json"))
    avs, m, pdb_h = compute_av(os.path.join(_examples_dir, "PDB_Data", "7N85-assembly1.cif"), params)
    
    model_nm_base = np.array([np.array(IMP.core.XYZ(av).get_coordinates()) * 0.1 for av in avs])
    model_centered = model_nm_base - model_nm_base.mean(axis=0)
    
    print("Running optimization...")
    sr = ScoringRestraintWrapper(
        m, avs, kdtree_obj=KDTree(aligned_cluster), dataxyz=aligned_cluster, 
        var=vars_target, type="Tree", model_coords_override=model_centered, searchradius=50.0
    )
    
    run_frequentist_optimization(m, pdb_h, avs, sr, "frequentist_thesis_fig_v3", max_cg_steps=150)
    
    # Extract final optimized coordinates
    live_model_coords = np.array([np.array(IMP.core.XYZ(av).get_coordinates(), dtype=np.float64) for av in avs]) * 0.1
    delta = live_model_coords - model_nm_base
    final_model_coords = model_centered + delta
    
    print("Generating Figure...")
    fig = plt.figure(figsize=(14, 6), facecolor=_DARK_BG)
    plt.subplots_adjust(wspace=0.25)

    # Panel A: Density Map
    ax_xy = fig.add_subplot(1, 2, 1)
    _style_axis(ax_xy, f"A) SMLM Density Map (Box {target_id})")
    
    xy = aligned_cluster[:, :2].T
    kde = gaussian_kde(xy)
    g = np.linspace(-100, 100, 100)
    xx, yy = np.meshgrid(g, g)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    
    ax_xy.contourf(xx, yy, zz, levels=25, cmap="plasma", alpha=0.8, zorder=1)
    ax_xy.scatter(aligned_cluster[:, 0], aligned_cluster[:, 1], c='white', s=4, alpha=0.5, zorder=2)
    
    av_xs = np.append(final_model_coords[:, 0], final_model_coords[0, 0])
    av_ys = np.append(final_model_coords[:, 1], final_model_coords[0, 1])
    ax_xy.plot(av_xs, av_ys, color=_ACCENT_CYAN, lw=2.5, alpha=0.9, zorder=3)
    ax_xy.scatter(final_model_coords[:, 0], final_model_coords[:, 1], c=_ACCENT_CYAN, s=100, marker='o', edgecolors='white', zorder=4, label="Optimized Model")
    
    ax_xy.set_xlabel("X (nm)", color=_TEXT_COLOR)
    ax_xy.set_ylabel("Y (nm)", color=_TEXT_COLOR)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlim(-100, 100)
    ax_xy.set_ylim(-100, 100)
    ax_xy.legend(facecolor=_DARK_BG, labelcolor=_TEXT_COLOR, edgecolor=_GRID_COLOR, loc='lower right')

    # Panel B: Score Landscape
    ax_map = fig.add_subplot(1, 2, 2)
    _style_axis(ax_map, "B) Likelihood Field Visualization")
    
    tree = KDTree(aligned_cluster[:, :2])
    pts_grid = np.column_stack([xx.ravel(), yy.ravel()])
    dists, _ = tree.query(pts_grid)
    field = np.exp(-(dists**2) / (2 * 10.0**2)).reshape(xx.shape)
    
    ax_map.contourf(xx, yy, field, levels=25, cmap="viridis", alpha=0.8)
    ax_map.scatter(final_model_coords[:, 0], final_model_coords[:, 1], c='white', s=100, marker='o', edgecolors='black', label="AV Positions")
    
    ax_map.set_xlabel("X (nm)", color=_TEXT_COLOR)
    ax_map.set_ylabel("Y (nm)", color=_TEXT_COLOR)
    ax_map.set_aspect('equal')
    ax_map.set_xlim(-100, 100)
    ax_map.set_ylim(-100, 100)
    ax_map.legend(facecolor=_DARK_BG, labelcolor=_TEXT_COLOR, edgecolor=_GRID_COLOR, loc='lower right')

    fig.suptitle(f"Final Thesis Result: Structural Fit for NPC Box {target_id}", color='white', fontsize=18, fontweight='bold', y=0.98)
    
    out_path = os.path.join(_examples_dir, "figures", "methodology", "thesis_final_result_v3.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=_DARK_BG, bbox_inches='tight')
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    main()
