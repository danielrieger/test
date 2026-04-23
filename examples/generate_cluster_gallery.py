import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_examples_dir = os.path.dirname(__file__)
_project_dir = os.path.abspath(os.path.join(_examples_dir, ".."))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from smlm_score.utility.input import read_experimental_data
from smlm_score.utility.data_handling import flexible_filter_smlm_data, isolate_individual_npcs

def _geometry_quality(pts):
    xy = pts[:, :2]
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    bbox = xy.max(axis=0) - xy.min(axis=0)
    width, height = float(bbox[0]), float(bbox[1])
    mean_diameter = 0.5 * (width + height)
    circularity = min(width, height) / max(width, height, 1e-9)
    diameter_score = np.exp(-((mean_diameter - 120.0)**2) / (2.0 * 20.0**2))
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    bins = np.linspace(-np.pi, np.pi, 9)
    hist, _ = np.histogram(angles, bins=bins)
    empty_bins = np.sum(hist == 0)
    coverage_score = 1.0 - (empty_bins / 8.0)
    return 0.4 * diameter_score + 0.3 * circularity + 0.3 * coverage_score

def main():
    print("Loading data...")
    raw_df = read_experimental_data(os.path.join(_examples_dir, "ShareLoc_Data", "data.csv"))
    coords, vars_, _, _, _ = flexible_filter_smlm_data(
        raw_df, filter_type="random", percentage=30, random_seed=42, fill_z_value=0.0
    )
    
    print("Clustering...")
    res = isolate_individual_npcs(coords, min_cluster_size=15, min_npc_points=100)
    
    candidates = []
    for c in res['npc_info']:
        cid = c['cluster_id']
        mask = res['labels'] == cid
        pts = coords[mask]
        score = _geometry_quality(pts)
        candidates.append({'id': cid, 'pts': pts, 'score': score, 'n': len(pts)})
    
    # Sort by score descending and take top 12
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:12]
    
    print(f"Plotting {len(candidates)} candidates...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), facecolor='#0d1117')
    axes = axes.flatten()
    
    for i, cand in enumerate(candidates):
        ax = axes[i]
        pts = cand['pts']
        centroid = pts.mean(axis=0)
        p = pts - centroid
        
        ax.scatter(p[:, 0], p[:, 1], c='#ff8c00', s=5, alpha=0.6)
        ax.set_title(f"ID: {cand['id']} | N: {cand['n']}", color='white', fontsize=12)
        ax.set_aspect('equal')
        ax.set_xlim(-80, 80)
        ax.set_ylim(-80, 80)
        ax.axis('off')
        
    plt.tight_layout()
    out_path = os.path.join(_examples_dir, "figures", "methodology", "cluster_candidates.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor='#0d1117')
    print(f"Saved gallery to: {out_path}")

if __name__ == "__main__":
    main()
