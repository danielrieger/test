# Implementation Plan - Figure Organization for Thesis

This plan outlines the reorganization of the project's figure structure to ensure a professional and unified asset management system for your thesis.

## User Review Required

> [!IMPORTANT]
> - **Unified Root**: All figure output will be moved into `examples/figures/` to keep the root directory clean.
> - **Legacy Cleanup**: The root `figures/` directory will be DELETED after all assets are relocated and script paths are updated.

## Proposed Changes

### 1. Unified Directory Structure
I will consolidate the current fragmented figures into the following logical structure:
- **`examples/figures/methodology/`**: The stylized 3D, 2D fitting sequence, and PCA summary (illustrating "how it works").
- **`examples/figures/benchmarks/`**: Scaling and tradeoff analysis (illustrating "how fast/accurate it is").
- **`examples/figures/qc/`**: Intermediate maps (HDBScan, clustering validation) for checking dataset quality.

### 2. Script Updates [MODIFY]
I will update the output paths in the following scripts:
- **[benchmark_scoring.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/benchmark_scoring.py)**: Change `figures/bench_*.png` → `examples/figures/benchmarks/`.
- **[generate_thesis_figs_v2.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/generate_thesis_figs_v2.py)**: Change `figures/npc_*.png` → `examples/figures/qc/`.
- **Final Figure Scripts**: Update the 3D and sequence generators to use the `methodology/` subfolder.

### 3. Cleanup [DELETE]
- **`c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/figures/`** (once migration is confirmed).

## Open Questions

- **Naming**: Does "Methodology" vs "Benchmarks" vs "QC" (Quality Control) work for your thesis, or would you prefer different names for these chapters?

## Verification Plan

### Automated Tests
- Relaunch each script (`visualize_alignment.py`, `benchmark_scoring.py`, etc.) and ensure they create their figures in the NEW subdirectories.

### Manual Verification
- Confirm that the root `smlm_score/` directory is clean and only contains core code/config files.
