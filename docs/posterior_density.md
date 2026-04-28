# Posterior AV Density Mapping

This document describes the Bayesian posterior density mapping feature, which accumulates the positions of dye-attachment points (Accessible Volumes, AVs) sampled during REMC Bayesian optimization into a volumetric probability density map.

## Overview

After a Bayesian sampling run, the pipeline automatically calls `_generate_av_density_mrc()` in `mcmc_sampler.py` to produce:

- **`posterior_density.mrc`** — A 3D volumetric density map (2 nm/voxel) in MRC format, directly readable by ChimeraX and UCSF Chimera.
- **`posterior_density.png`** — A 2D heatmap visualization of the same density for quick inspection.

Both files are saved into the run output directory (e.g., `bayesian_cluster_885/`). The PNG is also automatically copied to `examples/figures/Posterior/` with a frame-count suffix (e.g., `posterior_density_20000f.png`).

## What It Shows

The density map encodes the **spatial uncertainty** of the 8 fluorophore-labeled sites across all sampled structural conformations. High-density regions correspond to positions the model visits most frequently under the posterior distribution, i.e., where the model fits the experimental data best.

For the NPC, this produces a strikingly clear **8-fold symmetric ring** whose sharpness directly reflects the quality of the structural fit.

## Key Implementation Details

### 1. Burn-in Removal
The first 20% of frames are discarded as burn-in (REMC initialization). Only the remaining post-equilibration frames enter the density.

### 2. Score-Based Frame Filtering
REMC deliberately explores poor-scoring configurations to escape local minima. Including all post-burn-in frames would smear the density. Instead, only the **top 25% best-scoring frames** (lowest GMM scores) are accumulated. This focuses the density on the most structurally meaningful configurations.

### 3. Per-Frame Centroid Alignment
Over long runs (e.g., 20,000 frames), the REMC sampler exercises the **translational** degree of freedom extensively — the entire NPC ring can drift by hundreds of nanometers across the sampled field. The ring *shape* is preserved perfectly, but accumulated without correction the density would appear as a large featureless disk.

To reveal the true shape uncertainty, each frame's AV positions are **centered by their centroid** before accumulation. This removes rigid-body translational drift and focuses the density on the structural fluctuations within the ring.

### 4. Gaussian Smoothing
A Gaussian blur (σ = 1.5 pixels) is applied to the 2D histogram before saving, producing a smooth, continuous density suitable for visualization and publication.

## Visualization in ChimeraX

1. Open **ChimeraX**.
2. Drag and drop `posterior_density.mrc` into the window. It will appear as a 3D isosurface or volume cloud.
3. Optionally, also open `full_trajectory.rmf3` in the same session to see the protein scaffold inside the density cloud.
4. Use **Tools > Volume Viewer** to adjust the isosurface threshold.

## Score Trace

The score evolution over the course of sampling is saved to `frame_scores.csv` in the run output directory. This can be plotted directly (see `examples/figures/Posterior/score_trace_20k.png` for an example).

A healthy REMC trace shows:
- A rapid initial score decrease during burn-in.
- Sharp spikes from replica exchange swaps (the sampler temporarily visits high-T configurations).
- A stable rolling mean after convergence (~50–60% of the run).

## Parameters (Advanced)

The density generation function accepts the following keyword arguments:

| Parameter | Default | Description |
|:---|:---|:---|
| `burnin_fraction` | 0.2 | Fraction of frames to discard as burn-in. |
| `pixel_size_nm` | 2.0 | Voxel size of the output MRC map in nanometers. |
| `sigma_px` | 1.5 | Gaussian blur width in pixels. |
| `score_percentile` | 25 | Only accumulate frames with scores in the best N-th percentile. |
| `align_to_centroid` | True | Center each frame to remove rigid-body translational drift. |
