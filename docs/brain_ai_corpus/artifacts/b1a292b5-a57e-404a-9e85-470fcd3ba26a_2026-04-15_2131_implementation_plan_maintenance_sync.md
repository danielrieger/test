# Implementation Plan - Project Maintenance and Synchronization

This plan outlines the steps to finalize the current project state, update documentation, push to GitHub, and ensure all local versions (Windows and WSL) are perfectly synchronized.

## User Review Required

> [!IMPORTANT]
> **Source of Truth**: I will treat the Windows Desktop workspace (`c:\Users\User\OneDrive\Desktop\Thesis\smlm_score`) as the primary source of truth for code and documentation. Transient data (like EMAN2 logs) in WSL will be ignored or overwritten to match the Windows state.

> [!WARNING]
> **Git Pushes**: I will perform a `git add .` and commit the current state. If you have specific files you want to *exclude* that are not covered by `.gitignore` (e.g., specific drafts), please let me know.

## Proposed Changes

### Documentation & Configuration

#### [MODIFY] [README.md](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/README.md)
- Remove duplicate "License" section.
- Remove stray text at line 113.
- Update "Features" or "Optimization" section to include the new Bayesian sampling parameters (`score_weight`, `max_rb_trans`, `max_rb_rot`) and the live scoring feedback feature.

#### [MODIFY] [requirements.txt](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/requirements.txt)
- Add any missing dependencies required by the example scripts (e.g., `pyvista` for 3D visualization).

#### [MODIFY] [.gitignore](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/.gitignore)
- Add EMAN2 transient files to prevent them from cluttering the repository:
    - `.eman2log.txt`
    - `.eman2settings.json`
    - `nnet_*.hdf`
    - `trainout_*.hdf`
    - `picked_particles/`

### Version Control & Synchronization

#### GitHub Push (Windows)
1.  Stage all changes in the Windows workspace.
2.  Create a descriptive commit: `feat: optimize REMC sampling, add live feedback, and cleanup documentation`.
3.  Push to `origin/master`.

#### Sync & Cleanup (WSL)
1.  Perform a final `rsync` from Windows to WSL to ensure identity.
2.  Reset the WSL git state if necessary to match the newly pushed `origin/master`.

## Verification Plan

### Automated Tests
- Run `pytest` on Windows to ensure no regressions were introduced during cleanup.
- Check `git remote -v` to confirm successful push.

### Manual Verification
- Verify the GitHub UI shows the updated README and the latest commits.
- Run `examples/NPC_example_BD.py` in WSL to confirm the synchronized environment is functional.
