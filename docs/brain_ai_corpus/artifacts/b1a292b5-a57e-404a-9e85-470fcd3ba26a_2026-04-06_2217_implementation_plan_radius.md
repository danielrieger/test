# Implementation Plan - Radius Sensitivity Benchmark

This plan aims to provide a definitive technical answer to why the Tree engine can be slower than raw Distance scoring on dense experimental data. We will quantify the "Pruning Trade-off" by testing how the search radius affects execution time.

## User Review Required

> [!IMPORTANT]
> - **Data Selection**: I will use a high-density 10,000-point sample from your experimental data for this test to ensure the results are representative of "Real World" NPC clusters.
> - **Radii Range**: I plan to test from $1 \text{ nm}$ (extreme pruning) to $50 \text{ nm}$ (full NPC inclusion).

## Proposed Changes

### 1. `examples/benchmark_radius_sensitivity.py` [NEW]
- **Core Logic**: Loop through a range of radii: `[1, 2, 5, 10, 15, 20, 30, 50]`.
- **Baseline**: Record the Distance engine time (which is radius-independent).
- **Execution**: Run `computescoretree` for each radius and measure wall-clock time.
- **Plotting**: Generate `bench_figE_radius_sensitivity.png` showing "Time vs. Search Radius".

### 2. Gallery & README [MODIFY]
- If the results are significant (e.g., if we find a clear "Efficiency Crossover"), I will add a brief mention of this "Sensitivity Analysis" to the `README.md`.

## Verification Plan

### Automated Tests
- Run the script and verify that `bench_figE_radius_sensitivity.png` is generated in `examples/figures/benchmarks/`.

### Manual Verification
- Review the plot to identify the **Crossover Point** (where Tree becomes slower than Distance). This point is critical for your thesis discussion.
