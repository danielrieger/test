import os
import sys
import numpy as np
import pandas as pd

_examples_dir = os.path.dirname(__file__)
_project_dir = os.path.abspath(os.path.join(_examples_dir, ".."))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from smlm_score.utility.input import read_experimental_data
from smlm_score.utility.data_handling import flexible_filter_smlm_data, isolate_individual_npcs, align_npc_cluster_pca

def _geometry_quality(aligned_points, expected_diameter_nm=120.0):
    xy = aligned_points[:, :2]
    bbox = xy.max(axis=0) - xy.min(axis=0)
    width, height = float(bbox[0]), float(bbox[1])
    mean_diameter = 0.5 * (width + height)
    circularity = min(width, height) / max(width, height, 1e-9)
    diameter_score = np.exp(-((mean_diameter - expected_diameter_nm)**2) / (2.0 * 25.0**2))
    return 0.7 * diameter_score + 0.3 * circularity

def main():
    raw_df = read_experimental_data(os.path.join(_examples_dir, "ShareLoc_Data", "data.csv"))
    coords, vars_, _, _, _ = flexible_filter_smlm_data(
        raw_df, filter_type="random", percentage=10, random_seed=42, fill_z_value=0.0
    )
    res = isolate_individual_npcs(coords, min_cluster_size=15, min_npc_points=100)
    
    best_score = -1
    best_cid = -1
    best_n = 0
    
    for c in res['npc_info']:
        cid = c['cluster_id']
        mask = res['labels'] == cid
        pts = coords[mask]
        aligned = align_npc_cluster_pca(pts, debug=False)['aligned_data']
        score = _geometry_quality(aligned)
        
        if score > best_score:
            best_score = score
            best_cid = cid
            best_n = len(pts)
            
    print(f"BEST CLUSTER: {best_cid} (score {best_score:.3f}, {best_n} pts)")

if __name__ == "__main__":
    main()
