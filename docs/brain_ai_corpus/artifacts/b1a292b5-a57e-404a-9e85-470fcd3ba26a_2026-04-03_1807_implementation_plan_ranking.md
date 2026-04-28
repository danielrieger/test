# Implementation Plan - Highest Quality NPC Selection for GMM Plots

The current `visualize_gmm_selection.py` script picks the first available cluster (`index 0`). This plan ensures that the BIC curve and GMM overlay are generated for the NPC that best matches the structural ground-truth in the dataset.

## User Review Required

> [!IMPORTANT]
> - **Ranking Overhead**: Adding a ranking step requires loading the PDB model and evaluating all clusters, which may add 30-60 seconds to the script execution time depending on the dataset size.
> - **Quality Metric**: "Quality" is defined as the normalized structural alignment score between the experimental GMM/Point cloud and the idealized NPC model.

## Proposed Changes

### 1. Update `visualize_gmm_selection.py` [MODIFY]
I will integrate the ranking logic:
- **Model Loading**: Load the PDB assembly and compute Accessible Volumes (AVs).
- **Ranking Loop**: For every cluster found by HDBSCAN:
    1. Perform a quick PCA alignment.
    2. Evaluate a `Tree` scoring restraint.
    3. Normalize by the number of points to prevent a bias toward "just bigger" clusters.
- **Top Selection**: Sort by quality and pick the `#1` cluster for the visualization.

## Verification Plan

### Automated Tests
- Run `examples/visualize_gmm_selection.py`.
- Verify that the output plots (`gmm_bic_selection.png`, `gmm_cluster_overlay.png`) now show a highly structured, well-aligned NPC.
- Check the terminal output to see the ranking list and confirmation of the selected 'best' cluster ID.
