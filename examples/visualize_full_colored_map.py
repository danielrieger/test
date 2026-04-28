import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure smlm_score is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smlm_score.utility.input import read_experimental_data
from smlm_score.utility.data_handling import (
    flexible_filter_smlm_data,
    isolate_individual_npcs
)
from smlm_score.utility.visualization import plot_cluster_context_map

# 1. Load & Filter
print("Loading & Filtering data...")
smlm_data_path = "ShareLoc_Data/data.csv"
raw_smlm_data_df = read_experimental_data(smlm_data_path)
filtered_coords, _, _, _, _ = flexible_filter_smlm_data(
    raw_smlm_data_df,
    filter_type='cut',
    x_cut=(10000, 12000),
    y_cut=(0, 5000),
    fill_z_value=0.0,
    return_tree=False
)

# 2. Cluster using our validated "Complete NPC" logic (120nm bounded)
print("Clustering data using validated Geometric Assembly...")
npc_results = isolate_individual_npcs(
    filtered_coords,
    min_cluster_size=15,
    min_npc_points=100,
    perform_geometric_merging=True,
    debug=False
)
cluster_labels = npc_results['labels']

# 3. Plot Full Categorical Map (No zoom)
print("Generating full categorical color map...")
output_path = "full_cluster_overview_color.png"
plot_cluster_context_map(
    full_data=filtered_coords,
    cluster_labels=cluster_labels,
    target_cluster_id=None,
    sample_clusters=None,
    title="SMLM Full ROI - Categorical NPC Clustering (120nm Bounded)",
    save_path=output_path,
    show_zooms=False
)
print(f"Success! Map saved to {output_path}")
