import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

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
from smlm_score.utility.data_handling import flexible_filter_smlm_data, compute_av
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
    ax.grid(color=_GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.3)

def main():
    target_id = 240
    print(f"Loading data for Box {target_id}...")
    
    smlm_path = os.path.join(_examples_dir, "ShareLoc_Data", "data.csv")
    boxes_path = os.path.join(_examples_dir, "info", "micrograph_info.json")
    pixel_map_path = os.path.join(_examples_dir, "pixel_map.json")
    
    raw_df = read_experimental_data(smlm_path)
    # Use flexible_filter_smlm_data for consistency, but we'll use the raw_df for intensity
    coords, vars_, _, _, _ = flexible_filter_smlm_data(raw_df, filter_type="full", fill_z_value=0.0)
    
    # Load metadata
    with open(pixel_map_path, 'r') as f:
        pixel_map = json.load(f)
    pixel_size_nm = pixel_map['pixel_size_nm']
    
    with open(boxes_path, 'r') as f:
        info_data = json.load(f)
    box_data = info_data.get('boxes', [])
    
    if target_id >= len(box_data):
        print(f"Error: Box {target_id} out of range!")
        return
        
    center_px = box_data[target_id]
    cx_nm, cy_nm = center_px[0] * pixel_size_nm, center_px[1] * pixel_size_nm
    
    # Force 200nm extraction
    half_size = 100.0
    mask = (coords[:, 0] >= cx_nm - half_size) & (coords[:, 0] <= cx_nm + half_size) & \
           (coords[:, 1] >= cy_nm - half_size) & (coords[:, 1] <= cy_nm + half_size)
    
    pts = coords[mask]
    vars_target = vars_[mask]
    
    # Extract intensities from raw dataframe based on the same mask
    # Note: mask applied to coords (from processed_df) matches raw_df index if filter_type='full'
    # But to be safe, we'll slice raw_df manually
    raw_mask = (raw_df['x [nm]'] >= cx_nm - half_size) & (raw_df['x [nm]'] <= cx_nm + half_size) & \
               (raw_df['y [nm]'] >= cy_nm - half_size) & (raw_df['y [nm]'] <= cy_nm + half_size)
    intensities_target = raw_df.loc[raw_mask, 'Amplitude_0_0'].values
    
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
    
    run_frequentist_optimization(m, pdb_h, avs, sr, "frequentist_final_box240", max_cg_steps=150)
    
    # Extract final optimized coordinates
    live_model_coords = np.array([np.array(IMP.core.XYZ(av).get_coordinates(), dtype=np.float64) for av in avs]) * 0.1
    delta = live_model_coords - model_nm_base
    final_model_coords = model_centered + delta
    
    print("Generating Figure...")
    fig = plt.figure(figsize=(16, 8), facecolor=_DARK_BG)
    plt.subplots_adjust(top=0.88, bottom=0.15, left=0.1, right=0.9, wspace=0.3)

    # Panel A: Density Map (Weighted by Intensity)
    ax_xy = fig.add_subplot(1, 2, 1)
    _style_axis(ax_xy, f"A) High-Res SMLM Density (Box {target_id})")
    
    # Grid for density
    g = np.linspace(-100, 100, 100)
    xx, yy = np.meshgrid(g, g)
    
    # Normalizing intensities for plotting
    norm_intensities = (intensities_target - intensities_target.min()) / (intensities_target.max() - intensities_target.min() + 1e-9)
    # Scatter plot with size and color based on intensity
    sc = ax_xy.scatter(aligned_cluster[:, 0], aligned_cluster[:, 1], c=norm_intensities, cmap='plasma', s=8 + 20*norm_intensities, alpha=0.8, zorder=2)
    
    # Draw ring connector
    av_xs = np.append(final_model_coords[:, 0], final_model_coords[0, 0])
    av_ys = np.append(final_model_coords[:, 1], final_model_coords[0, 1])
    ax_xy.plot(av_xs, av_ys, color=_ACCENT_CYAN, lw=2, alpha=0.6, zorder=3, linestyle='--')
    
    ax_xy.scatter(final_model_coords[:, 0], final_model_coords[:, 1], c=_ACCENT_CYAN, s=150, marker='o', edgecolors='white', zorder=4, label="Optimized Model")
    
    ax_xy.set_xlabel("X (nm)", color=_TEXT_COLOR)
    ax_xy.set_ylabel("Y (nm)", color=_TEXT_COLOR)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlim(-100, 100)
    ax_xy.set_ylim(-100, 100)
    ax_xy.legend(facecolor=_DARK_BG, labelcolor=_TEXT_COLOR, edgecolor=_GRID_COLOR, loc='lower right')

    # Panel B: Scoring Likelihood Field
    ax_map = fig.add_subplot(1, 2, 2)
    _style_axis(ax_map, "B) Structural Scoring Landscape")
    
    tree = KDTree(aligned_cluster[:, :2])
    pts_grid = np.column_stack([xx.ravel(), yy.ravel()])
    dists, _ = tree.query(pts_grid)
    # sigma = 10nm for the likelihood field
    field = np.exp(-(dists**2) / (2 * 10.0**2)).reshape(xx.shape)
    
    im = ax_map.contourf(xx, yy, field, levels=25, cmap="viridis", alpha=0.9)
    ax_map.scatter(final_model_coords[:, 0], final_model_coords[:, 1], c='white', s=120, marker='o', edgecolors='black', label="AV Centers")
    
    ax_map.set_xlabel("X (nm)", color=_TEXT_COLOR)
    ax_map.set_ylabel("Y (nm)", color=_TEXT_COLOR)
    ax_map.set_aspect('equal')
    ax_map.set_xlim(-100, 100)
    ax_map.set_ylim(-100, 100)
    ax_map.legend(facecolor=_DARK_BG, labelcolor=_TEXT_COLOR, edgecolor=_GRID_COLOR, loc='lower right')

    fig.suptitle(f"Thesis Publication Figure: Refined NPC Analysis (Box {target_id})", color='white', fontsize=22, fontweight='bold')
    
    out_path = os.path.join(_examples_dir, "figures", "methodology", "thesis_final_result_v4.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=_DARK_BG, bbox_inches='tight')
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    main()
