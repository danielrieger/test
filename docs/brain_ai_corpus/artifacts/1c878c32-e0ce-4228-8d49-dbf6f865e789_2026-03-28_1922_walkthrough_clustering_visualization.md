# NPC Clustering Visualization Walkthrough

A new visualization script, [visualize_npc_clustering_steps_random.py](file:///C:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/visualize_npc_clustering_steps_random.py), has been added to provide step-by-step insight into the NPC identification process. This tool is designed to generate high-quality figures for the thesis by breaking down the clustering pipeline into three distinct stages.

## 1. The 3-Step Clustering Pipeline
The script mirrors the production pipeline's logic, documenting the sequence of operations applied after initial spatial filtering:
1. **Initial Clustering (HDBSCAN)**: Densitiy-based identification of primary localization groups.
2. **Geometric Merging**: Complete-linkage agglomerative clustering that merges adjacent clusters within 140 nm. This includes the "safety split" optimization for large datasets.
3. **NPC-Size Selection**: Filtering out clusters that do not meet the minimum point density required for a valid NPC structure (e.g., <100 points).

## 2. Generated Thesis Figures
Running the script generates individual figures for each stage and a consolidated 2x2 overview in `examples/figures/clustering_steps/`:
- **`step0_random_filtered.png`**: The raw experimental map after ROI selection.
- **`step1_hdbscan_raw.png`**: The output of the first-pass HDBSCAN clustering.
- **`step2_geometric_merge.png`**: The map after geometric assembly of fragmented clusters.
- **`step3_npc_sized_clusters.png`**: The final map showing only high-confidence NPC particles.
- **`clustering_steps_overview.png`**: A consolidated 2x2 panel for clear comparison of the pipeline's refinement.

## 3. Usage
The script can be run with the default settings (using `pipeline_config.json`) or with a specific random seed for reproducible ROI cuts:
```powershell
# Reproducible run with a fixed random window
C:\envs\py311\python.exe examples\visualize_npc_clustering_steps_random.py --seed 42
```

## 4. Final Validation
The script has been verified against the current [data_handling.py](file:///C:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/utility/data_handling.py) implementation, ensuring that the visualized stages perfectly match the mathematical logic used in the main SMLM-IMP scoring runs.
