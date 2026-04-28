# Implementation Plan - Finalization & Data Safety

Ensure the long-term safety of model/data inputs, update all manuals with the new EMAN2 and RMF features, and perform a final, synchronized push across all environments.

## User Review Required

> [!IMPORTANT]
> **Source of Truth Policy**: I will continue to treat the Windows Desktop workspace as the primary source of truth for code and documentation. However, I will implement "Safe Sync" rules to ensure that unique data in WSL (like EMAN2 picking results) is never deleted.

> [!WARNING]
> **Data Directories**: The following directories will be explicitly protected from future automated deletions:
> - `examples/info/`
> - `examples/picked_particles/`
> - `examples/PDB_Data/`
> - `examples/ShareLoc_Data/`
> - `examples/SuReSim_Input/`

## Proposed Changes

### 1. Data Safety & Protection

#### [MODIFY] [.gitignore](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/.gitignore)
- Add a dedicated "PROTECTED DATA (DO NOT DELETE)" section.
- Explicitly ignore `examples/info/` and `examples/picked_particles/` to ensure they are never accidentally committed or overwritten during standard git operations.

#### [NEW] [safe_sync.sh](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/safe_sync.sh)
- A utility script for WSL to perform identity sync from Windows *without* deleting untracked files in the target directories.
- Uses `rsync -av --update` with specific excludes for data folders.

### 2. Documentation Update

#### [MODIFY] [README.md](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/README.md)
- **Bayesian Trajectory Section**: Add instructions on how to visualize the unified rigid body movement in ChimeraX using `full_trajectory.rmf3`.
- **EMAN2 Workflow**: Document the EMAN2 box picking integration and the restoration script (`recover_boxes.py`) for future reference.
- **Data Preservation**: Add a "Maintenance" section describing how to safely manage large datasets and sync between environments.

### 3. Code Review & Cleanup

#### [MODIFY] [mcmc_sampler.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/imp_modeling/simulation/mcmc_sampler.py)
- Final pass to ensure all docstrings are accurate for the new RMF reconstruction logic.
- Ensure all debug prints are cleaned up for a "production" release.

### 4. Synchronization & Version Control

#### GitHub Push (Windows)
1.  Stage all final changes.
2.  Commit with: `feat: finalize Bayesian sampler, RMF reconstruction, and data safety protocols`.
3.  Push to `origin/master`.

#### Safe Sync (WSL)
1.  Execute the new `safe_sync.sh` to align WSL with the final Windows state.
2.  Perform a `git fetch` and merge (rather than reset) to preserve local configuration.

## Verification Plan

### Automated Tests
- Run `pytest` on Windows to confirm 100% stability.
- Run the `recover_boxes.py` script one last time to ensure the EMAN2 metadata is perfectly preserved.

### Manual Verification
- Visual check of the GitHub repository to ensure the new documentation is rendered correctly.
- Verify that `examples/info/micrograph_info.json` survives a test sync.
