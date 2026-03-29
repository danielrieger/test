import sys, os
import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from smlm_score.src.utility.input import read_experimental_data
from smlm_score.src.utility.data_handling import flexible_filter_smlm_data

# Load
raw = read_experimental_data(os.path.join(os.path.dirname(__file__), "ShareLoc_Data/data.csv"))
coords, _, _, _, _ = flexible_filter_smlm_data(raw, filter_type='cut', x_cut=(10000, 12000), y_cut=(0, 5000))
data = coords[:, :2]

# Step 1: HDBSCAN to aggressively filter noise
hdb = HDBSCAN(min_cluster_size=15, min_samples=15, cluster_selection_epsilon=0.0)
hdb_labels = hdb.fit_predict(data)
clean_mask = hdb_labels != -1
clean_pts = data[clean_mask]

print(f"Original pts: {len(data)}, HDBSCAN noise removed. Clean pts: {len(clean_pts)}")

# Step 2: Agglomerative Clustering with Complete Linkage
agg = AgglomerativeClustering(n_clusters=None, distance_threshold=140, linkage='complete')
macro_labels = agg.fit_predict(clean_pts)

n_clusters = len(set(macro_labels))
print(f"Total Macro Clusters Formed bounded to strictly <= 140nm: {n_clusters}")

# Re-map correctly
final_labels = np.full(len(data), -1)
final_labels[clean_mask] = macro_labels

# Stats
sizes = []
n_pts = []
for cid in range(n_clusters):
    idx = final_labels == cid
    if np.any(idx):
        pts = data[idx]
        bmin, bmax = pts.min(axis=0), pts.max(axis=0)
        sz = bmax - bmin
        if len(pts) >= 100:
            sizes.append(sz)
            n_pts.append(len(pts))

sizes = np.array(sizes)
print(f"NPCs with >= 100 points: {len(sizes)}")
if len(sizes) > 0:
    print(f"Average NPC bounding box: {sizes.mean(axis=0)}")
    print(f"Average NPC localizations: {np.mean(n_pts):.1f}")
