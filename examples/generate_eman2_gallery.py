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

from smlm_score.utility.input import read_experimental_data
from smlm_score.utility.data_handling import flexible_filter_smlm_data

def _geometry_quality(pts):
    if len(pts) < 10: return 0.0
    xy = pts[:, :2]
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    
    # 1. Radius check (should be ~60nm)
    radii = np.linalg.norm(centered, axis=1)
    mean_r = np.mean(radii)
    r_score = np.exp(-((mean_r - 60.0)**2) / (2.0 * 15.0**2))
    
    # 2. Circularity / Aspect Ratio
    bbox = xy.max(axis=0) - xy.min(axis=0)
    width, height = float(bbox[0]), float(bbox[1])
    circularity = min(width, height) / max(width, height, 1e-9)
    
    # 3. Angular Coverage
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    bins = np.linspace(-np.pi, np.pi, 9)
    hist, _ = np.histogram(angles, bins=bins)
    empty_bins = np.sum(hist == 0)
    coverage_score = 1.0 - (empty_bins / 8.0)
    
    return 0.3 * r_score + 0.2 * circularity + 0.5 * coverage_score

def main():
    smlm_path = os.path.join(_examples_dir, "ShareLoc_Data", "data.csv")
    boxes_path = os.path.join(_examples_dir, "info", "micrograph_info.json")
    pixel_map_path = os.path.join(_examples_dir, "pixel_map.json")
    
    print("Loading data...")
    raw_df = read_experimental_data(smlm_path)
    coords, vars_, _, _, _ = flexible_filter_smlm_data(raw_df, filter_type="full", fill_z_value=0.0)
    
    print("Loading pixel map and boxes...")
    with open(pixel_map_path, 'r') as f:
        pixel_map = json.load(f)
    pixel_size_nm = pixel_map['pixel_size_nm']
    
    with open(boxes_path, 'r') as f:
        info_data = json.load(f)
    boxes = info_data.get('boxes', [])
    
    # FORCE 200nm BOX SIZE
    forced_box_size_nm = 200.0
    half_size_nm = forced_box_size_nm / 2.0
    
    candidates = []
    print(f"Evaluating {len(boxes)} boxes with forced 200nm window...")
    
    for i, box in enumerate(boxes):
        px_x, px_y = box[0], box[1]
        
        # Center in nm
        cx_nm = px_x * pixel_size_nm
        cy_nm = px_y * pixel_size_nm
        
        x_min, x_max = cx_nm - half_size_nm, cx_nm + half_size_nm
        y_min, y_max = cy_nm - half_size_nm, cy_nm + half_size_nm
        
        mask = (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) & \
               (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
        
        pts = coords[mask]
        if len(pts) < 30: continue
        
        score = _geometry_quality(pts)
        candidates.append({'id': i, 'pts': pts, 'score': score, 'n': len(pts)})
    
    # Sort and take top 40
    top_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:40]
    
    print(f"Plotting top {len(top_candidates)} candidates...")
    fig, axes = plt.subplots(4, 10, figsize=(30, 12), facecolor='#0d1117')
    axes = axes.flatten()
    
    for i, cand in enumerate(top_candidates):
        ax = axes[i]
        p = cand['pts']
        centroid = p.mean(axis=0)
        pts = p - centroid
        
        ax.scatter(pts[:, 0], pts[:, 1], c='#ff8c00', s=4, alpha=0.8)
        ax.set_title(f"ID: {cand['id']}\nN: {cand['n']}", color='white', fontsize=8)
        ax.set_aspect('equal')
        # Frame matches our 200nm box
        ax.set_xlim(-110, 110)
        ax.set_ylim(-110, 110)
        ax.axis('off')
    
    for j in range(len(top_candidates), len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    out_path = os.path.join(_examples_dir, "figures", "methodology", "eman2_candidates_v4.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor='#0d1117')
    print(f"Saved gallery to: {out_path}")

if __name__ == "__main__":
    main()
