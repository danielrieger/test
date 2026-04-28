import os
import sys

# Ensure smlm_score is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from smlm_score.utility.input import read_experimental_data
from smlm_score.utility.data_handling import (
    flexible_filter_smlm_data,
    isolate_individual_npcs
)
from smlm_score.utility.visualization import plot_cluster_context_map

# 1. Load Data
print("Loading experimental SMLM data...")
smlm_data_path = "ShareLoc_Data/data.csv"
raw_smlm_data_df = read_experimental_data(smlm_data_path)

# 2. Filter exactly like the main pipeline
print("Filtering data...")
filtered_coords, filtered_vars, _, _, _ = flexible_filter_smlm_data(
    raw_smlm_data_df,
    filter_type='cut',
    x_cut=(10000, 12000), # Original pipeline ROI
    y_cut=(0, 5000),
    fill_z_value=0.0,
    return_tree=False
)
data_for_clustering = filtered_coords

# 3. Cluster
print("Clustering data using HDBSCAN...")
npc_results = isolate_individual_npcs(
    data_for_clustering,
    min_cluster_size=15,
    min_npc_points=100,
    perform_geometric_merging=True,
    debug=False
)
cluster_labels = npc_results['labels']
cluster_info = npc_results['all_cluster_info']

# 4. Identify clusters
valid_clusters = [info['cluster_id'] for info in cluster_info if info['n_points'] >= 100]
print(f"Found {len(valid_clusters)} valid NPCs (clusters with >= 100 points).")

# Select a target (e.g. the largest one) and a few random ones
if valid_clusters:
    # Get largest valid cluster
    largest = max([info for info in cluster_info if info['n_points'] >= 100], key=lambda i: i['n_points'])
    target_cluster_id = largest['cluster_id']
    print(f"Target Cluster: {target_cluster_id} ({largest['n_points']} points)")
    
    # Pick a few random clusters (mix of small and valid if possible)
    all_clusters = [info['cluster_id'] for info in cluster_info if info['cluster_id'] != -1]
    np.random.seed(321) # Changed seed for new samples
    sample_clusters = np.random.choice(
        [c for c in all_clusters if c != target_cluster_id], 
        size=min(8, len(all_clusters)-1), 
        replace=False
    ).tolist()
    
    # 5. Plot
    print(f"Highlighting Target: {target_cluster_id}")
    print(f"Highlighting Samples: {sample_clusters}")
    
    output_path = "cluster_context_map.png"
    plot_cluster_context_map(
        full_data=data_for_clustering,
        cluster_labels=cluster_labels,
        target_cluster_id=target_cluster_id,
        sample_clusters=sample_clusters,
        title=f"SMLM Cluster Map (Target: {target_cluster_id})",
        save_path=output_path
    )
    print(f"Visualization complete. Map saved to {output_path}")
else:
    print("No valid clusters found to plot.")
