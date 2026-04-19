# SMLM Score Pipeline: Manual & Technical Overview

This manual provides instructions on how to use the `NPC_example_BD.py` script, outlines what the recently fixed features do, and offers a detailed, step-by-step overview of the entire modelling and scoring process.

---

## 1. Quick Start Guide

### Prerequisites
1. **Python Environment**: Ensure you are running within the `py311` conda environment that has `IMP` (Integrative Modeling Platform), `sklearn`, and `scipy` installed.
2. **Environment Variable**: Your `PYTHONPATH` must be set to the root project directory (e.g., `C:\Users\User\OneDrive\Desktop\Thesis`).
3. **Data Files**: The script expects the following files in the runtime directory:
   - `ShareLoc_Data/data.csv` (SMLM localization data)
   - `PDB_Data/7N85-assembly1.cif` (NPC structural model)
   - `av_parameter.json` (Parameters for Accessible Volume computation)

### Configuration Options
At the top of `NPC_example_BD.py`, you can configure the following flags:

*   **`TEST_SCORING_TYPES = ["Tree", "GMM", "Distance"]`**
    Which scoring functions to test and report. You can remove items to run the script faster (e.g., `["Distance"]`).
*   **`RUN_BAYESIAN_SAMPLING = False`** 
    Set to `True` to run the active Replica Exchange Monte Carlo (REMC) optimization. Keep as `False` if you only want to quickly screen and validate clusters in under a minute.
*   **`TARGET_CLUSTER_ID = None`**
    Set to `None` to automatically select the largest valid NPC structure found by HDBSCAN. Otherwise, you can hardcode an integer ID (e.g., `312`) to analyze a specific cluster.

### Running the Script
Run the script using standard Python:
```powershell
$env:PYTHONPATH = "C:\Users\User\OneDrive\Desktop\Thesis"
C:\envs\py311\python.exe -X utf8 examples\NPC_example_BD.py
```

---

## 2. What the Script Does (and Recent Fixes)

The pipeline evaluates how well an *in silico* 3D structural model matches *in vitro* SMLM super-resolution data using Bayesian pseudo-energy (log-likelihood) functions.

**Recent Bug Fixes Included:**
*   **Automated Target Selection**: Removed hardcoded target IDs. The script safely auto-detects robust valid clusters.
*   **Perfect PCA Alignment**: When evaluating clusters, the script rotates and centers the experimental data to the origin. The model is now perfectly aligned to match this rotation via the `model_coords_override` parameter, ensuring scores reflect structural fit, not arbitrary spatial offsets.
*   **Variance Handling**: Fallback $1.0\text{ nm}^2$ covariances are generated if variances are missing, preventing validation crashes.
*   **True Validation Logic**: Log-likelihoods are strictly evaluated properly (less negative = better fit), proving that `Distance` and `GMM` scorers mathematically separate true NPCs from noise.

---

## 3. Detailed Overview of the Modelling Process

The script performs a sequential 7-stage process:

### Stage 1: Experimental Data Ingestion
The pipeline reads SMLM localization coordinates and their respective spatial variances (uncertainties). It applies a spatial bounding box (`flexible_filter_smlm_data`) to select a Region of Interest (ROI) and removes low-quality points.

### Stage 2: Computational Model Setup
Using the Integrative Modeling Platform (IMP), the script loads the macromolecular structure (`.cif`). 
*   **Accessible Volumes (AVs)**: Instead of modeling every atom, the script relies on AVs. It reads `av_parameter.json` to compute probabilistic point-clouds detailing where specific labeled fluorophores are physically allowed to exist on the protein structure.
*   These computed AVs act as the "model points" that will be compared to the SMLM "data points".

### Stage 2: Geometric NPC Isolation
Because the SMLM data contains many proteins, background noise, and fragments, the pipeline runs HDBSCAN clustering followed by optional geometric merging.

**Key Parameters (data_handling.py):**
*   **`min_cluster_size` (int, default=15):** The smallest group of points considered a cluster. This is the primary sensitivity dial.
*   **`min_npc_points` (int, default=100):** Clusters with fewer points than this are considered "noise" or "partial NPCs".
*   **`perform_geometric_merging` (bool, default=True):** 
    *   **True (Recommended)**: Triggers the **140nm Hierarchical Geometric Merging** strategy. This fuses fragmented circles (arcs) together while strictly preventing the merger of distinct adjacent NPCs.
    *   **False**: Returns raw HDBSCAN fragments (useful for subunit-level analysis).

### Stage 3: PCA Alignment
Once isolated, the NPC is centered and rotated.
*   **Centroid Offset**: The pipeline subtracts the mathematical center of the cluster, moving the NPC to (0,0,0).
*   **PCA Rotation**: Principal Component Analysis aligns the planar spread of the ring with the XY-plane of the IMP model. 

---

## 4. Reproducing Results for the Thesis
To generate the visualizations and statistics used in your final report:

### 1. The ROI Overview (Spectral Cluster Map)
Run `examples/visualize_full_colored_map.py`. This produces a single high-contrast overview map where each of the 300+ NPCs is uniquely colored using a 256-color spectral sequence.

### 2. Physical Structural Validation (7N85 Overlays)
Run `examples/visualize_clusters.py`. This produces a context map with zoom-in panels that overlay 120nm (outer) and 45nm (inner) reference circles directly onto your experimental clusters.

### 3. Clustering Statistics (Fragment vs. Macro)
Run `examples/compare_clustering_logic.py`. This script prints a side-by-side comparison of how many "fragments" were found by raw HDBSCAN versus how many "complete NPCs" were isolated by the Geometric merging process.

### 4. Running the Full Test Suite
To verify the entire software stack (Clustering, Alignment, Scoring):
```powershell
C:\envs\py311\python.exe -m pytest tests/ -v
```
This will execute all unit tests and the newly added **Robustness Suite** (Empty data handling, 2D stability, Inversion proofs).

---

## Technical Summary of Pipeline Fixes
*   **Fixed Conda Environment**: Resolved a `SyntaxError` in the `pytest` library caused by path corruption during environment migration.
*   **Implemented Geometric Assembly**: Solved the fragmentation issue where NPCs were split into pieces.
*   **Synchronized Model Orientation**: Fixed a bug where the PDB model wasn't rotated to match the PCA-aligned data, which previously caused incorrect validation scores.
*   **Cleaned Workspace**: Removed several gigabytes of temporary simulation output to optimize disk space while preserving source code and raw data.
