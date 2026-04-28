# Code Review: NPC_example_BD.py

## 🔴 Critical Bugs

### Bug 1: `TARGET_CLUSTER_ID = 0` may not exist (Line 29 → 108)

```python
TARGET_CLUSTER_ID = 0  # Line 29
target_valid_cluster = [c for c in valid_clusters if c['cluster_id'] == TARGET_CLUSTER_ID]  # Line 108
```

**Problem**: HDBSCAN cluster IDs are based on density hierarchy and cluster 0 may have <100 points (filtered out as noise). If no valid cluster has `cluster_id == 0`, `target_valid_cluster` is `[]`, and `clusters_to_evaluate` contains **only noise clusters** — the entire scoring loop tests nothing meaningful.

**Fix**: Pick the *largest* valid cluster instead of a hardcoded ID:
```diff
-TARGET_CLUSTER_ID = 0
+TARGET_CLUSTER_ID = None  # Auto-select largest valid NPC
...
-target_valid_cluster = [c for c in valid_clusters if c['cluster_id'] == TARGET_CLUSTER_ID]
+# Auto-select: pick largest valid cluster
+if TARGET_CLUSTER_ID is None:
+    target_valid_cluster = [max(valid_clusters, key=lambda c: c['n_points'])]
+    TARGET_CLUSTER_ID = target_valid_cluster[0]['cluster_id']
+else:
+    target_valid_cluster = [c for c in valid_clusters if c['cluster_id'] == TARGET_CLUSTER_ID]
```

---

### Bug 2: Model not rotated to match PCA-aligned data (Lines 130–144)

```python
alignment_results = align_npc_cluster_pca(cluster_points)  # Line 130
aligned_cluster_points = alignment_results['aligned_data']  # Line 131
# ...
model_coords_scaled = np.array([
    np.array(IMP.core.XYZ(av).get_coordinates()) * 0.1  # Line 139
    for av in avs
])
model_centroid = model_coords_scaled.mean(axis=0)  # Line 142
data_centroid = aligned_cluster_points.mean(axis=0)  # Line 143 → always ~[0,0,0]!
model_to_data_offset = data_centroid - model_centroid  # Line 144
```

**Problem**: `align_npc_cluster_pca()` centers data at origin AND rotates it to principal axes. The model AVs are only *translated* (offset), never *rotated*. The model ring orientation won't match the PCA-aligned data orientation.

Also: `data_centroid` is always ~`[0,0,0]` because PCA centers the data. So `model_to_data_offset ≈ -model_centroid`, which just translates the model to origin. **The rotation is missing.**

**Fix**: Apply the same PCA rotation to the model coordinates:
```diff
+rotation_matrix = alignment_results['rotation']
 model_coords_scaled = np.array([...]) * 0.1
-model_centroid = model_coords_scaled.mean(axis=0)
-data_centroid = aligned_cluster_points.mean(axis=0)
-model_to_data_offset = data_centroid - model_centroid
+# Center model at origin, then rotate to match PCA basis
+model_centered = model_coords_scaled - model_coords_scaled.mean(axis=0)
+model_aligned = np.dot(model_centered, rotation_matrix.T)
+model_to_data_offset = np.zeros(3)  # Both are now centered at origin
```

---

### Bug 3: `smlm_variances` length mismatch (Line 127)

```python
cluster_variances = smlm_variances[cluster_mask] if len(smlm_variances) == len(data_for_clustering) else None
```

**Problem**: `smlm_variances` comes from `flexible_filter_smlm_data()` and corresponds to `smlm_coordinates`. But `data_for_clustering` is `smlm_coordinates_for_tree` (which may have different indexing/length). If they don't match, `cluster_variances` is silently set to `None`, losing all variance information for Tree and Distance scoring.

**Fix**: Use index-aligned extraction:
```diff
-cluster_variances = smlm_variances[cluster_mask] if len(smlm_variances) == len(data_for_clustering) else None
+# smlm_variances corresponds to the same array as data_for_clustering
+# because both come from flexible_filter_smlm_data with return_tree=True
+if smlm_variances is not None and len(smlm_variances) == len(data_for_clustering):
+    cluster_variances = smlm_variances[cluster_mask]
+else:
+    cluster_variances = None
+    print(f"  Warning: variance array length mismatch, scoring without variances")
```

---

## 🟡 Medium Issues

### Bug 4: Held-out validation has no offset alignment (Lines 293–314)

The held-out scoring creates `ScoringRestraintWrapper` **without** `offsetxyz` for Tree and Distance scoring, meaning model AVs (in Å-scale coordinates) are scored against data in nm coordinates. This makes held-out scores incomparable with cluster scores.

**Fix**: Add offset computation for held-out chunks too.

---

### Bug 5: Wildcard import hides dependencies (Line 10)

```python
from smlm_score.src.utility.data_handling import *
```

This imports everything from `data_handling.py` into the namespace, including `DBSCAN`, `HDBSCAN`, `pd`, `KDTree`, `tqdm`, etc. — making it unclear what functions are actually used.

**Fix**: Replace with explicit imports:
```diff
-from smlm_score.src.utility.data_handling import *
-from smlm_score.src.utility.data_handling import get_held_out_complement
+from smlm_score.src.utility.data_handling import (
+    isolate_individual_npcs,
+    align_npc_cluster_pca,
+    flexible_filter_smlm_data,
+    compute_av,
+    get_held_out_complement,
+)
```

---

### Bug 6: Distance scoring covariance may be empty list (Lines 191–205)

```python
smlm_covariances_list = []
if cluster_variances is not None:
    for var_scalar in cluster_variances:
        ...
sr_wrapper = ScoringRestraintWrapper(
    ..., var=smlm_covariances_list if smlm_covariances_list else None, ...
)
```

If `cluster_variances` is `None` (due to Bug 3), `smlm_covariances_list` stays `[]` and `var=None` is passed. The Distance scoring function may crash or produce meaningless results without variances.

---

## 🟢 Low / Cosmetic

### Issue 7: Duplicate `pathlib` import in data_handling.py (Lines 2, 13)

```python
import pathlib  # Line 2
import pathlib  # Line 13 — duplicate
```

---

### Issue 8: Comment says "Step 7" twice (Lines 148, 207)

Step numbering jumps: 1→2→3→4→5→6→**7**→**7**→8. The second "Step 7" should be "Step 8".
