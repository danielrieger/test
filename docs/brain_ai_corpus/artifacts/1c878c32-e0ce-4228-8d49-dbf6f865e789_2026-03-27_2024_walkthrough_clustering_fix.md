# Pipeline Update: Memory Efficiency & API Consistency

The recent updates focused on resolving a critical memory crash during NPC clustering and standardizing data filtering API return types across the codebase.

## 1. Memory Crash Fix (Stage 4 Clustering)
The `isolate_individual_npcs` function in [data_handling.py](file:///C:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/utility/data_handling.py) was prone to a `MemoryError` (unable to allocate ~25 GiB) when `perform_geometric_merging=True` was applied to large clean datasets (~89k points). 
- **The Issue**: Point-wise `AgglomerativeClustering` with complete linkage has $O(n^2)$ memory complexity because it computes a full distance matrix (`pdist`).
- **The Fix**: Introduced a **safe split strategy** at [line 255](file:///C:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/utility/data_handling.py#L255):
  - **Small Clean Sets (≤ 5000 points)**: Maintain existing high-precision point-wise merging.
  - **Large Clean Sets (> 5000 points)**: Switch to **cluster-level merging**. This performs agglomeration on the centroids of HDBSCAN clusters, drastically reducing the input size to the `AgglomerativeClustering` algorithm and avoiding massive memory allocations.
- **Regression Test**: Added a new test `test_stage4_two_stage_large_clean_set_uses_cluster_level_merge` in [test_stage4_clustering_unit.py](file:///C:/Users/User/OneDrive/Desktop/Thesis/smlm_score/tests/test_stage4_clustering_unit.py#L339) to verify that large sets never trigger point-wise merging.

## 2. API Consistency (Data Filtering)
The `flexible_filter_smlm_data` function in [data_handling.py](file:///C:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/utility/data_handling.py) was updated to ensure predictable return-arity regardless of the input data status:
- **Consistent Returns**: Now always returns 5 values: `data_xyz, sigma_array, data_for_tree, kdtree, applied_cuts`.
- **Empty Case Handle**: The empty-data branch now explicitly returns 5 values ([line 500](file:///C:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/utility/data_handling.py#L500)).
- **Updated Call Sites**: All script and test call sites were updated to use 5-value unpacking, fixing "too many values to unpack" errors in `examples/test_cluster.py`, `tests/test_pipeline_missing_stages_unit.py`, `tests/test_pipeline_e2e_integration.py`, and `examples/NPC_example_BD.py`.

## 3. Validation
The full test suite is now passing (**94 passed**), verifying that the memory-efficient clustering and the API changes are stable across both unit tests and e2e integration pipelines.
