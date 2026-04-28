# Master Benchmarking Suite (Refined Figures A-E)

## Problem
The initial master benchmark needs specific refinements for professional thesis use, including dual-theme support (Light/Dark), better legend consistency for stacked benchmarks, and adjusted layout for readability.

## Refined Figure Requirements

| Figure | Requirement | Description |
| :--- | :--- | :--- |
| **Theme** | **Dual Output** | Every figure will be exported as both `_dark.png` (for presentations) and `_light.png` (for print/white-bg documents). |
| **Fig A** | **Complexity Labels** | Add complexity notation to legends: Tree $O(N \log M)$, Distance $O(NM)$, GMM $O(GK)$. |
| **Fig B** | **Legend Consistency** | Sync legend with four-color bar logic. Legend will clearly distinguish "Initialization" (Gray) from the Method-specific "Optimization" colors. |
| **Fig C** | **Print Version** | Add white background variant. |
| **Fig D** | **Layout Fix** | Increase padding/gap between bar numbers and "vs. Neighbor" label to prevent overlap. Fix legend/bar color sync. |
| **Fig E** | **Depth Analysis** | Retain as an "Algorithmic Envelope" analysis to show where Tree pruning ceases to be beneficial vs. simple Distance (the crossover point). |

## Proposed Changes

### [NEW] [master_benchmark.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/master_benchmark.py)
A comprehensive benchmarking engine that implements:
1. **Dynamic Theme Controller**: A helper function to swap matplotlib stylesheets between "GitHub Dark" and "Standard Print White".
2. **Standardized Benchmarking**:
    - **A & B**: Scaling and MCMC Tradeoff (10,000 steps).
    - **C**: GMM BIC Selection Cost.
    - **D**: Speedup Comparison (Distance vs. optimized Tree).
    - **E**: Radius Sweep ($r = 1$ to $200$ nm).
3. **Enhanced Formatting**:
    - Uses LaTeX-style math for complexity labels in legends.
    - Adjusts `zorder` and label padding to prevent the Fig D overlap issue.
    - Ensures MCMC bars in Fig B/D use a legend that explains the stacked colors (e.g., Gray = Build, Color = Run).

## Verification Plan

### Automated Tests
1. **Run Master Benchmark**: Confirm generation of 10 files (5 figures × 2 themes).
   ```bash
   python examples/master_benchmark.py
   ```
2. **Visual Check**: Open the `_light.png` versions and confirm "Print" readability (no dark-gray labels on white background).

### Manual Verification
1. Verify Fig D label spacing.
2. Verify Fig A Complexity labels ($O(N \log M)$).
