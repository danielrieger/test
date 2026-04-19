# Implementation Plan - Stylized Thesis 3D Visualizations

This plan outlines the creation of three distinct stylized 3D visualization modes to provide high-quality alternatives to raw point cloud plots for your thesis.

## User Review Required

> [!IMPORTANT]
> - **Grid Resolution**: To keep the density rendering fast, I will use a 50x50x50 grid. This provides a balance between detail and performance.
> - **Isosurface Level**: I will choose a default density threshold that makes the "ring" look solid but not bloated.

## Proposed Changes

### [NEW] [visualize_alignment_stylized_3d.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/visualize_alignment_stylized_3d.py)

A new script with three specialized rendering modes.

#### Mode 1: **"Idealized Model"**
- **Graphic**: 8 (or 16) perfectly smooth, semi-transparent spheres representing the structural centers.
- **Goal**: Show the pure geometry of the NPC subunits without any experimental "noise."
- **Style**: Soft lighting and clean coordinate labels.

#### Mode 2: **"3D Density Isosurface"**
- **Graphic**: A solid 3D "hull" calculated from the SMLM localizations using a Gaussian density filter.
- **Logic**: Converts points into a voxel grid → applies `gaussian_filter` → generates a mesh via `marching_cubes`.
- **Goal**: Make the NPC look like a physical macromolecule rather than a cloud of dots.

#### Mode 3: **"Scoring Concept"** (Side-by-Side/Overlay)
- **Graphic**: A combined view showing the "Model Spheres" and the "Data Density" interacting.
- **Goal**: Illustrate how the Bayesian scoring function "sees" the alignment (the overlap of model volume and experimental density).

### [MODIFY] [visualization.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/utility/visualization.py)

-   Add `plot_isosurface_3d` and `plot_idealized_npc_3d` primitives to the library.

## Open Questions

- **Single vs. Double Ring**: Should the idealized model show both the cytosolic and nuclear rings (16 subunits total)? (I recommend 16 for a more "classic" NPC look).
- **Colors**: Should I stick to the current "Orange (Model) vs. Cyan/Plasma (Data)" scheme, or use more neutral "Publication White/Blue" tones?

## Verification Plan

### Manual Verification
- Run the script with each mode (`--mode ideal`, `--mode surface`, `--mode concept`).
- Inspect the generated PNGs for clarity and scientific accuracy.
- Ensure the figures are at 300 DPI for thesis-ready export.
