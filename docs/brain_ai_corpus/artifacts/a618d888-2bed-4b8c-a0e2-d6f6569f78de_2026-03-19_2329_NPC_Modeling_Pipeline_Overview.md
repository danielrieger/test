# SMLM-IMP Modeling Pipeline Overview

This document outlines the formalized 4-stage workflow for processing SMLM localizations, isolating individual NPC structures, and scoring them against a 7N85 PDB model using Bayesian Markov Chain Monte Carlo (MCMC) sampling.

---

## Stage 1: Data Mastery (Ingest & ROI)
*   **Input:** SMLM CSV files (ShareLoc format) + PDB Structural Model (`7N85-assembly1.cif`).
*   **Dimensionality Fix:** Automatically fills missing `z` coordinates with 0.0 for 2D analysis to maintain IMP compatibility.
*   **Coordinate Scaling:** Ensures consistent conversion between Nanometers (SMLM) and Angstroms (IMP).
*   **ROI Extraction:** Spatially filters the raw dataset into high-density Regions of Interest (e.g., a 2x5 µm crop).

## Stage 2: Geometric NPC Isolation (The "Clustering")
*   **Primary Logic:** HDBSCAN-driven density peak detection.
*   **Refinement:** **Hierarchical Geometric Merging (140nm Bounded).** 
    *   This stage fuses fragmented arcs into single NPCs.
    *   It also splits "touching" NPCs that HDBSCAN mistakenly merges by enforcing a strict physical 140nm diameter limit.
*   **Centroid Offset:** Calculates the mathematical center $(Cx, Cy)$ of every cluster and subtracts it to move the NPC to the coordinate origin $(0,0,0)$.

## Stage 3: Structural Orientation (PCA Alignment)
*   **Alignment Logic:** Principal Component Analysis (PCA).
*   **Goal:** Orientation snap-to-grid. 
*   **Process:** Calculates the principal axes of the isolated localization cloud and **rotates** the data so the NPC's planar ring aligns with the model's coordinate frame (typically the XY-plane).
*   **Output:** A perfectly centered, oriented, and noise-stripped point cloud ready for scoring.

## Stage 4: Science & Scoring (Validation & Sampling)
*   **Scoring Engines:**
    *   **GMM (Gaussian Mixture Model):** Density-based scoring.
    *   **Tree (Spatial Partitioning):** Geometric hierarchical scoring.
    *   **Distance (Euclidean):** Direct point-to-volume distance.
*   **Validation Logic:** This is where we verify the scoring system is "honest" before we trust the sampler:
    *   **Separation Distance Test:** Verifies that a valid NPC scores better than artificial noise clusters.
    *   **Held-Out Data Test:** Checks if the scoring is robust when 20% of the points are removed.
*   **Bayesian Sampling:** Executes the `run_bayesian_sampling` MCMC script to generate the final distribution of positions/configs for the thesis.

---

### Comparison of User-Controlled Flags
| Flag Name | Purpose | Effect |
|-----------|---------|--------|
| `PERFORM_GEOMETRIC_MERGING` | Toggles Stage 2 refinement | `True` = Complete 120nm NPCs; `False` = Raw Fragments/Arcs |
| `RUN_BAYESIAN_SAMPLING` | Triggers MCMC execution | Enables the computationally intensive sampling phase in Stage 4 |
| `TEST_SCORING_TYPES` | Multi-scorer validation | Runs the validation tests across Tree, GMM, and Distance engines |
