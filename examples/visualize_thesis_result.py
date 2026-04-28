import os
import sys
import json
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

from smlm_score.utility.input import read_experimental_data, read_parameters_from_json
from smlm_score.utility.data_handling import flexible_filter_smlm_data, isolate_individual_npcs, align_npc_cluster_pca, compute_av
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
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)
    if title:
        ax.set_title(title, color=_TEXT_COLOR, fontsize=12, pad=10)
    ax.grid(color=_GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.5)

def _geometry_quality(aligned_points):
    xy = aligned_points[:, :2]
    bbox = xy.max(axis=0) - xy.min(axis=0)
    width, height = float(bbox[0]), float(bbox[1])
    mean_diameter = 0.5 * (width + height)
    circularity = min(width, height) / max(width, height, 1e-9)
    diameter_score = np.exp(-((mean_diameter - 120.0)**2) / (2.0 * 25.0**2))
    return 0.7 * diameter_score + 0.3 * circularity

def main():
    print("Loading and preparing data...")
    raw_df = read_experimental_data(os.path.join(_examples_dir, "ShareLoc_Data", "data.csv"))
    coords, vars_, _, _, _ = flexible_filter_smlm_data(
        raw_df, filter_type="random", percentage=15, random_seed=42, fill_z_value=0.0
    )
    res = isolate_individual_npcs(coords, min_cluster_size=15, min_npc_points=80)
    
    best_score = -1
    best_cluster_pts = None
    best_vars = None
    
    for c in res['npc_info']:
        cid = c['cluster_id']
        mask = res['labels'] == cid
        pts = coords[mask]
        aligned = align_npc_cluster_pca(pts, debug=False)['aligned_data']
        score = _geometry_quality(aligned)
        
        if score > best_score:
            best_score = score
            best_cluster_pts = pts
            best_vars = vars_[mask]
            
    print(f"Found best cluster with {len(best_cluster_pts)} points.")
    
    # User requested to turn OFF PCA for 2D data
    # We will just center it.
    centroid = best_cluster_pts.mean(axis=0)
    aligned_cluster = best_cluster_pts - centroid
    
    print("Setting up IMP model...")
    params = read_parameters_from_json(os.path.join(_examples_dir, "av_parameter.json"))
    avs, m, pdb_h = compute_av(os.path.join(_examples_dir, "PDB_Data", "7N85-assembly1.cif"), params)
    
    model_nm_base = np.array([np.array(IMP.core.XYZ(av).get_coordinates()) * 0.1 for av in avs])
    model_centered = model_nm_base - model_nm_base.mean(axis=0)
    
    print("Running optimization...")
    sr = ScoringRestraintWrapper(
        m, avs, kdtree_obj=KDTree(aligned_cluster), dataxyz=aligned_cluster, 
        var=best_vars, type="Tree", model_coords_override=model_centered, searchradius=50.0
    )
    
    # We use CG for speed, then capture the final state
    run_frequentist_optimization(m, pdb_h, avs, sr, "frequentist_thesis_fig", max_cg_steps=100)
    
    # Get final coordinates
    live_model_coords = np.array([np.array(IMP.core.XYZ(av).get_coordinates(), dtype=np.float64) for av in avs]) * 0.1
    delta = live_model_coords - model_nm_base
    final_model_coords = model_centered + delta
    
    print("Generating Figure...")
    fig = plt.figure(figsize=(18, 6), facecolor=_DARK_BG)
    plt.subplots_adjust(wspace=0.3)

    # Panel A: 2D Density Map with Overlay (User request C)
    ax_xy = fig.add_subplot(1, 3, 1)
    _style_axis(ax_xy, "A) SMLM Density & Model (XY)")
    
    from scipy.stats import gaussian_kde
    xy = aligned_cluster[:, :2].T
    kde = gaussian_kde(xy)
    g = np.linspace(-80, 80, 100)
    xx, yy = np.meshgrid(g, g)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax_xy.contourf(xx, yy, zz, levels=20, cmap="plasma", alpha=0.8, zorder=1)
    
    ax_xy.scatter(aligned_cluster[:, 0], aligned_cluster[:, 1], c='white', s=2, alpha=0.3, zorder=2)
    
    av_xs = np.append(final_model_coords[:, 0], final_model_coords[0, 0])
    av_ys = np.append(final_model_coords[:, 1], final_model_coords[0, 1])
    ax_xy.plot(av_xs, av_ys, color=_ACCENT_CYAN, lw=2, alpha=0.8, zorder=3)
    ax_xy.scatter(final_model_coords[:, 0], final_model_coords[:, 1], c=_ACCENT_CYAN, s=80, marker='o', edgecolors='white', zorder=4, label="Fitted Ring")
    
    ax_xy.set_xlabel("X (nm)", color=_TEXT_COLOR)
    ax_xy.set_ylabel("Y (nm)", color=_TEXT_COLOR)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlim(-80, 80)
    ax_xy.set_ylim(-80, 80)
    ax_xy.legend(facecolor=_DARK_BG, labelcolor=_TEXT_COLOR, edgecolor=_GRID_COLOR)

    # Panel B: XZ Side View
    ax_xz = fig.add_subplot(1, 3, 2)
    _style_axis(ax_xz, "B) Side View (XZ)")
    ax_xz.scatter(aligned_cluster[:, 0], aligned_cluster[:, 2], c=_ACCENT_ORANGE, s=8, alpha=0.5)
    ax_xz.scatter(final_model_coords[:, 0], final_model_coords[:, 2], c=_ACCENT_CYAN, s=80, marker='o', edgecolors='white', zorder=5)
    
    ax_xz.set_xlabel("X (nm)", color=_TEXT_COLOR)
    ax_xz.set_ylabel("Z (nm)", color=_TEXT_COLOR)
    ax_xz.set_xlim(-80, 80)
    ax_xz.set_ylim(-40, 40)
    
    # Panel C: Tree Score Map (Visualization of the Likelihood field)
    ax_map = fig.add_subplot(1, 3, 3)
    _style_axis(ax_map, "C) SMLM-IMP Score Landscape")
    
    g = np.linspace(-80, 80, 50)
    xx, yy = np.meshgrid(g, g)
    zz = np.zeros_like(xx)
    
    # Fast proxy for the likelihood field: distance to nearest data point
    tree = KDTree(aligned_cluster[:, :2])
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    dists, _ = tree.query(pts)
    field = np.exp(-(dists**2) / (2 * 10.0**2)).reshape(xx.shape)
    
    cont = ax_map.contourf(xx, yy, field, levels=20, cmap="viridis", alpha=0.8)
    ax_map.scatter(final_model_coords[:, 0], final_model_coords[:, 1], c='white', s=60, marker='o', edgecolors='black', label="AV Positions")
    
    ax_map.set_xlabel("X (nm)", color=_TEXT_COLOR)
    ax_map.set_ylabel("Y (nm)", color=_TEXT_COLOR)
    ax_map.set_aspect('equal')
    ax_map.legend(facecolor=_DARK_BG, labelcolor=_TEXT_COLOR, edgecolor=_GRID_COLOR)

    fig.suptitle("Final Optimization Result (Without PCA)", color='white', fontsize=18, fontweight='bold', y=0.98)
    
    out_path = os.path.join(_examples_dir, "figures", "methodology", "thesis_final_result.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=_DARK_BG, bbox_inches='tight')
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    main()
