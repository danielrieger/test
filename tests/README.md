# SMLM-IMP Test Suite

This directory contains the automated test suite for the SMLM-IMP pipeline. The suite provides high coverage across mathematical scoring functions, structural alignment, and pipeline integration.

## Test Categories

1. **Unit Tests**: Isolated testing of core functions (Scoring, Alignment, Clustering).
   - `test_scoring.py`: Mathematical verification of Tree, GMM, and Distance engines.
   - `test_stage4_clustering_unit.py`: HDBSCAN and geometric merging logic.
   - `test_stage5_alignment_unit.py`: PCA-based model-data registration.

2. **Integration Tests**: Verification of multi-stage workflows.
   - `test_pipeline_e2e_integration.py`: End-to-end run from CSV to validation results.
   - `test_pipeline_missing_stages_integration.py`: Testing robustness when certain processing stages are omitted.

3. **Robustness & Edge Cases**:
   - `test_stage4_clustering_robustness.py`: Behavior with tiny/empty point clouds.
   - `test_stage5_alignment_robustness.py`: PCA failure modes.

## Running Tests

### Standard Execution
Run the full suite using `pytest`:
```bash
pytest tests/
```

### CUDA-Dependent Tests
Tests involving GPU kernels (CUDA) will automatically skip if a compatible NVIDIA GPU/Numba environment is not detected.
To verify GPU kernels specifically:
```bash
pytest tests/test_scoring.py -v
```

## Maintenance

When modifying core logic in `src/`, always run the integration suite to ensure no regressions in score calculation or coordinate transformations.
