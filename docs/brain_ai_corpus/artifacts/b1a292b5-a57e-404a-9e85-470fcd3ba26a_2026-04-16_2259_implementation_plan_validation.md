# Implementation Plan: Cross-Validated NPC Validation (Strategy B)

## Goal
Replace the weak noise-cluster comparison with a statistically principled cross-validation that tests whether scoring functions are sensitive to NPC ring geometry. Remove the redundant Distance separation test.

## Proposed Changes

### 1. Validation Module

#### [MODIFY] [validation.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/validation/validation.py)

**New function: `validate_cross_validated_npc`**
- Input: aligned cluster points (N×3), model coordinates (M×3), scoring type, IMP model, AVs
- Angular split: compute azimuthal angle `θ = atan2(y, x)` in PCA-aligned frame, split at median θ into two halves (A and B)
- For each half as train/test:
  - Tree/Distance: build KD-tree from train half, score model against test half
  - GMM: fit GMM on train half, score model against test half's fitted density
- Scrambled control: independently permute x, y, z columns of test half to destroy ring geometry but preserve density
- Report: cross-val mean score vs scrambled mean score, separation in sigma

**Update: `run_full_validation`**
- Accept new `cross_val_data` parameter (dict with cluster points, model coords, etc.)
- Call `validate_cross_validated_npc` when data is available
- Remove `Separation_Distance` from default scoring types (keep Tree + GMM only)

### 2. Pipeline Script

#### [MODIFY] [NPC_example_BD.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/NPC_example_BD.py)
- After scoring the target cluster, pass aligned cluster points and model coordinates into the validation call
- Remove Distance from default separation types since it's redundant

### 3. Tests

#### [MODIFY] [test_pipeline_missing_stages_unit.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/tests/test_pipeline_missing_stages_unit.py)
- Add unit test for angular split logic (verify two halves, verify scrambled null)
- Add unit test for cross-val scoring with synthetic ring data (should pass) vs random data (should fail)

## Verification
- Run `pytest` — all existing + new tests pass
- Run pipeline once to confirm the new validation report prints correctly
