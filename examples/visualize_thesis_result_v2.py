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
from smlm_score.utility.data_handling import flexible_filter_smlm_data, isolate_individual_npcs, compute_av
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

def _geometry_quality(pts):
    """Evaluate if an NPC cluster forms a complete, closed ring."""
    # 1. Basic Geometry
    xy = pts[:, :2]
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    
    # Bounding box
    bbox = xy.max(axis=0) - xy.min(axis=0)
    width, height = float(bbox[0]), float(bbox[1])
    mean_diameter = 0.5 * (width + height)
    circularity = min(width, height) / max(width, height, 1e-9)
    diameter_score = np.exp(-((mean_diameter - 120.0)**2) / (2.0 * 20.0**2))
    
    # 2. Angular Coverage (Are there points in all directions?)
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    
    # We divide the circle into 8 sectors and count points in each
    bins = np.linspace(-np.pi, np.pi, 9)
    hist, _ = np.histogram(angles, bins=bins)
    
    # Perfect coverage would have roughly equal points in all 8 bins.
    # At least, we don't want any empty bins (which means a broken ring).
    empty_bins = np.sum(hist == 0)
    coverage_score = 1.0 - (empty_bins / 8.0)
    
    # 3. Density in the center (a real ring should be empty in the center)
    radii = np.linalg.norm(centered, axis=1)
    center_points = np.sum(radii < 30.0) # Points within 30nm of center
    center_penalty = np.exp(-center_points / 5.0) # Penalize if there are points in the hole
    
    return 0.3 * diameter_score + 0.3 * circularity + 0.3 * coverage_score + 0.1 * center_penalty

def main():
    print("Loading and preparing data...")
    raw_df = read_experimental_data(os.path.join(_examples_dir, "ShareLoc_Data", "data.csv"))
    
    # Using a 30% sample region to ensure we find a good ring without hanging HDBSCAN
    coords, vars_, _, _, _ = flexible_filter_smlm_data(
        raw_df, filter_type="random", percentage=30, random_seed=42, fill_z_value=0.0
    )
    
    # INCREASED min_npc_points to 160 to filter out sparse/partial rings
    res = isolate_individual_npcs(coords, min_cluster_size=15, min_npc_points=160)
    
    best_score = -1
    best_cluster_pts = None
    best_vars = None
    
    print(f"Testing {len(res['npc_info'])} candidate clusters...")
    for c in res['npc_info']:
        cid = c['cluster_id']
        mask = res['labels'] == cid
        pts = coords[mask]
        
        # Don't use PCA, just center to evaluate geometry
        score = _geometry_quality(pts)
        
        if score > best_score:
            best_score = score
            best_cluster_pts = pts
            best_vars = vars_[mask]
            
    print(f"Found best closed-ring cluster with {len(best_cluster_pts)} points (Score: {best_score:.3f}).")
    
    # Center the cluster (NO PCA)
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
    
    run_frequentist_optimization(m, pdb_h, avs, sr, "frequentist_thesis_fig_v2", max_cg_steps=100)
    
    # Extract final optimized coordinates
    live_model_coords = np.array([np.array(IMP.core.XYZ(av).get_coordinates(), dtype=np.float64) for av in avs]) * 0.1
    delta = live_model_coords - model_nm_base
    final_model_coords = model_centered + delta
    
    print("Generating Figure...")
    fig = plt.figure(figsize=(14, 6), facecolor=_DARK_BG)
    plt.subplots_adjust(wspace=0.25)

    # Panel A: SMLM Density Map + Model Overlay
    ax_xy = fig.add_subplot(1, 2, 1)
    _style_axis(ax_xy, "A) SMLM Density Map & Model Overlay")
    
    xy = aligned_cluster[:, :2].T
    kde = gaussian_kde(xy)
    g = np.linspace(-80, 80, 100)
    xx, yy = np.meshgrid(g, g)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    
    ax_xy.contourf(xx, yy, zz, levels=20, cmap="plasma", alpha=0.8, zorder=1)
    ax_xy.scatter(aligned_cluster[:, 0], aligned_cluster[:, 1], c='white', s=3, alpha=0.4, zorder=2)
    
    av_xs = np.append(final_model_coords[:, 0], final_model_coords[0, 0])
    av_ys = np.append(final_model_coords[:, 1], final_model_coords[0, 1])
    ax_xy.plot(av_xs, av_ys, color=_ACCENT_CYAN, lw=2, alpha=0.8, zorder=3)
    ax_xy.scatter(final_model_coords[:, 0], final_model_coords[:, 1], c=_ACCENT_CYAN, s=80, marker='o', edgecolors='white', zorder=4, label="Optimized NPC Ring")
    
    ax_xy.set_xlabel("X (nm)", color=_TEXT_COLOR)
    ax_xy.set_ylabel("Y (nm)", color=_TEXT_COLOR)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlim(-80, 80)
    ax_xy.set_ylim(-80, 80)
    ax_xy.legend(facecolor=_DARK_BG, labelcolor=_TEXT_COLOR, edgecolor=_GRID_COLOR, loc='lower right')

    # Panel B (was C): Score Landscape
    ax_map = fig.add_subplot(1, 2, 2)
    _style_axis(ax_map, "B) Likelihood Field Visualization")
    
    tree = KDTree(aligned_cluster[:, :2])
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    dists, _ = tree.query(pts)
    field = np.exp(-(dists**2) / (2 * 10.0**2)).reshape(xx.shape)
    
    ax_map.contourf(xx, yy, field, levels=20, cmap="viridis", alpha=0.8)
    ax_map.scatter(final_model_coords[:, 0], final_model_coords[:, 1], c='white', s=80, marker='o', edgecolors='black', label="AV Positions")
    
    ax_map.set_xlabel("X (nm)", color=_TEXT_COLOR)
    ax_map.set_ylabel("Y (nm)", color=_TEXT_COLOR)
    ax_map.set_aspect('equal')
    ax_map.set_xlim(-80, 80)
    ax_map.set_ylim(-80, 80)
    ax_map.legend(facecolor=_DARK_BG, labelcolor=_TEXT_COLOR, edgecolor=_GRID_COLOR, loc='lower right')

    fig.suptitle("Fitted Structural Model Overlay on Complete NPC Ring", color='white', fontsize=18, fontweight='bold', y=0.98)
    
    out_path = os.path.join(_examples_dir, "figures", "methodology", "thesis_final_result_v2.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=_DARK_BG, bbox_inches='tight')
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    main()
