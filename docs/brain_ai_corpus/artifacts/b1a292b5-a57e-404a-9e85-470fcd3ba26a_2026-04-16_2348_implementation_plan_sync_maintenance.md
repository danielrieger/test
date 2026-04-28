# Implementation Plan: Versioning and Safe Sync

## Goal
Safely commit the validated Strategy B implementation, push to GitHub, and synchronize the Windows workspace with the WSL environment using the `safe_sync.sh` utility.

## Safety Checks [MANDATORY]
> [!IMPORTANT]
> - **Data Protection**: Verified that `.gitignore` correctly ignores `ShareLoc_Data/`, `PDB_Data/`, and `bayesian_cluster_*/` result folders.
    - No large data files or private research result directories will be committed or deleted.
- **Maintenance**: Only code, unit tests, and the startup script are staged for versioning.

## Proposed Changes

### VERSION CONTROL
#### [COMMIT & PUSH]
- Stage the following files:
  - [validation.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/validation/validation.py)
  - [NPC_example_BD.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/NPC_example_BD.py)
  - [run_npc_example.ps1](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/run_npc_example.ps1)
  - [test_pipeline_missing_stages_unit.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/tests/test_pipeline_missing_stages_unit.py)
- Commit message: `feat: implement Strategy B structural cross-validation and angular partitioning`
- Push to `origin master`.

### SYNCHRONIZATION
#### [SYNC]
- Execute `.\safe_sync.sh` to synchronize the Windows codebase with the WSL environment.
- The script uses `rsync` with strict exclusions to ensure data safety.

## Verification Plan
1. **Git Verification**: Confirm `git status` is clean after push.
2. **Sync Verification**: Verify that the remote/WSL destination matches the local state for code files.
