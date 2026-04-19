# Implementation Plan: Addressing Technical Debt & Pipeline Correctness

This plan addresses recent feedback, focusing on critical path-resolution bugs (causing PyCharm execution failures), pipeline correctness flaws (`P0`), and structural refactoring (`P1`). 

*Note regarding the German assessment: Many of the architectural limitations mentioned in that text (missing particle segmentation, missing rigid body optimization, Numba compiler crashes) have already been successfully resolved in our recent pipeline upgrades (e.g., EMAN2 boxing, Kabsch alignment, Bayesian sampling). The remaining relevant points are covered here.*

## Phase 1: P0 Correctness & Environment Fixes (Immediate)

### 1. Fix Path Resolution for IDEs (PyCharm / Root execution)
**Target**: `examples/NPC_example_BD.py`
- **Issue**: The script assumes the Current Working Directory (CWD) is `examples/`, causing `FileNotFoundError` for config files when run from the project root or via PyCharm remote interpreters.
- **Fix**: Use `pathlib.Path(__file__).parent` to enforce that config files (`pipeline_config.json`, `av_parameter.json`) are resolved relative to the script location, regardless of where the Python interpreter was invoked from. Added graceful error handling rather than crashing with `TypeError` if parsing fails.

### 2. Fix Held-Out Validation Crash
**Target**: `examples/NPC_example_BD.py`
- Modify the scoring loop to explicitly capture and store `target_s` (target cluster score) and `target_n_points` when processing `TARGET_CLUSTER_ID`.
- Ensure these variables are defined in the broader scope so the held-out validation aggregation succeeds without `NameError` logic bugs.

### 3. Enforce `RUN_BAYESIAN_SAMPLING` Config
**Target**: `examples/NPC_example_BD.py`
- Wrap the execution of `run_bayesian_sampling()` in an `if RUN_BAYESIAN_SAMPLING:` check. Currently, the config is read but silently ignored during the optimization step.

### 4. Resolve `score_weight` Disconnect
**Target**: `src/smlm_score/imp_modeling/restraint/scoring_restraint.py` & `src/smlm_score/imp_modeling/simulation/mcmc_sampler.py`
- If we keep `score_weight`, implement `set_weight()` on `ScoringRestraintWrapper` so the parameter actually reaches the IMP restraint engine. Alternatively, remove the config knob if weighting should only be handled via temperature parameters.

### 5. Reconcile Validation Defaults & Tests
**Target**: `src/smlm_score/validation/validation.py` & `tests/test_pipeline_*`
- The pipeline recently removed `Distance` from the default separation tests, causing expectation mismatches in integration tests (`2 failed`). Update the test suites to accurately reflect the unified `Tree`/`GMM` validation strategy.

## Phase 2: P1 Refactoring & Mathematics Docs (Subsequent)

### 1. Script Refactoring
**Target**: `examples/NPC_example_BD.py`
- Break the monolithic script into functional blocks: `prepare_data()`, `score_candidates()`, `optimize_target()`, `validate_results()`.

### 2. Mathematical Honesty & Docs
**Target**: `docs/scoring_models.md` & `README.md`
- Draft a `scoring_models.md` detailing the heuristic nature of the GMM implementation and the assumption of fixed `sigma_av = 8.0` (as highlighted in the assessment).
- Update `README.md` to formally outline the WSL Miniforge `smlm` environment as the primary target.

## Verification
- Run the full test suite in WSL (`python -m pytest -q`) to ensure 100% pass rate.
- Run `NPC_example_BD.py` from the project root to confirm path resolution works flawlessly.
