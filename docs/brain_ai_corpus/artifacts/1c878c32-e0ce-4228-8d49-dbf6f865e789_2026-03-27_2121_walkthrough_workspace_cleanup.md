# Workspace Cleanup Walkthrough

I have completed the cleanup of the `smlm_score` workspace. The following unnecessary output files and directories have been removed:

## 1. Caches and Build Artifacts
- **`.pytest_cache`**: Removed the root-level pytest cache directory.
- **`__pycache__`**: Recursively deleted all 13 Python bytecode cache directories across the `src`, `tests`, and `examples` folders.

## 2. Simulation and Execution Outputs
- **Result Directories**: Deleted the following directories generated during pipeline runs:
    - All `examples/*_output_*` folders (e.g., `bayesian_output_cluster_2_GMM`, `frequentist_output_cluster_2_Distance`, etc.)
    - `examples/bayesian_cluster_*`
    - `examples/frequentist_cluster_*`
- **Large Files**: Removed all generated PDBs, RMF3s, and trajectory files.

## 3. Large Files Retained (Non-Output)
The following files exceeding 1MB were kept as they are considered source data or thesis assets:
- `examples/ShareLoc_Data/data.csv` (Source data)
- `examples/PDB_Data/7N85-assembly1.cif` (Source data)
- `260201_Bachelor_englisch_1.50.pdf` (Thesis PDF)
- Various PNG assets in `examples/` and `examples/figures/` (Thesis visualizations)

## 4. Temporary Test Data
- **`tests/.tmp_*`**: Cleaned up all temporary directories created by integration tests (e.g., `.tmp_dbg_...`).

## Verification
- Verified with `find_by_name` and `list_dir` that all targeted patterns are gone.
- Source code, configuration files, and figures intended for the thesis remain intact.
