# Implementation Plan - Fitting Sequence Visualization (Wu et al. Style)

This plan outlines the creation of a 4-panel figure that illustrates the iterative optimization of the SMLM-score alignment.

## User Review Required

> [!IMPORTANT]
> - **Synthetic Optimization Path**: Since our pipeline often finds the best fit in a single step (via PCA + Scoring), I will "reverse-engineer" a sequence by starting with a slight translation/rotation offset and interpolating towards the optimal fit across 4 panels.
> - **Visual Effect**: I will use a high-resolution Gaussian Mixture to create the "Cyan Glow" effect for the model density.

## Proposed Changes

### [NEW] [visualize_fitting_sequence_2d.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/visualize_fitting_sequence_2d.py)

A specialized script to generate the sequence figure.

#### 1. Optimization Sequence Logic
- Start with the **Optimal Fit** found by the pipeline.
- Generate an **Initial State** with a random offset (e.g., +15nm X/Y drift, 20° rotation).
- Perform **Linear Interpolation** for the parameters $\{x, y, \theta\}$ across 4 steps.

#### 2. Plotting Component
- **Background**: Solid Black.
- **Model Layer**: A 2D Gaussian density map (Sum of 8 Gaussians) rendered as a cyan glow with a smooth alpha fallout.
- **Data Layer**: The orange SMLM localizations as crisp scatter points.
- **Annotation**: Add the log-likelihood value (or a "fitting progress" bar) below each panel to match the paper's style.

### [MODIFY] [visualization.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/utility/visualization.py)

- Add `plot_model_density_glow_2d` to handle the specific cyan-bloom effect seen in the paper.

## Open Questions

- **Specific Parameters**: Should I include the mathematical formula ($\sum \log M...$) below the panels as seen in the image?
- **Dataset**: Should I use the same "Top Cluster" we identified earlier for this figure? (I recommend this for consistency).

## Verification Plan

### Manual Verification
- Run the script and inspect the 4-panel output.
- Ensure the "glow" doesn't overwhelm the points.
- Verify the 300 DPI resolution for print.
