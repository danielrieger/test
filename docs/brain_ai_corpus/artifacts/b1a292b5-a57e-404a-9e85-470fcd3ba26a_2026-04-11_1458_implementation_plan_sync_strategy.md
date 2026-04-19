# Implementation Plan: Repository Alignment & Autosync Strategy

The goal is to resolve the version discrepancy between the WSL development environment and the Windows OneDrive tracking copy, and provide a convenient way to keep them in sync.

## User Review Required

> [!IMPORTANT]
> **Source of Truth**: I am treating the WSL directory (`~/Thesis/smlm_score`) as the master copy, as it houses your active conda environment and GPU-integrated modeling tools. The Windows folder will be a mirror.

> [!NOTE]
> **Autosync Approach**: I recommend a "Manual Triggered Sync" (a single command like `sync`) rather than a fully background automation. True background sync between WSL and Windows filesystems can often lead to file-lock conflicts while you are typing.

## Proposed Changes

### Phase 1: Alignment (Immediate)
1.  **Commit WSL Changes**: Stage and commit the latest `pipeline_config.json` edits in WSL.
2.  **Push to GitHub**: Push the state from WSL.
3.  **Hard Reset Windows**: Force-sync the Windows repository to match GitHub exactly, discarding the "phantom" local changes to dependencies.

### Phase 2: Synchronization Tooling
1.  **[NEW] `sync_code.sh` (WSL Side)**:
    - Creates a bash script in your home directory that pushes WSL changes and then uses `git` on the Windows side to pull.
2.  **Git Alias**: Create a git alias `git sync` that runs the above logic.

## Verification Plan

### Automated Tests
- Run `git rev-parse HEAD` on both sides to verify commit hash parity.
- Verify `git status` shows "nothing to commit" on both environments.

### Manual Verification
- Edit a file in WSL, run the sync command, and verify it appears in the Windows folder/OneDrive.

## Open Questions
- **Pre-Automated Pull**: Would you like the Windows side to automatically pull *every* time you run a test in WSL, or should it remain a separate command?
- **PyCharm Integration**: Since you have PyCharm Professional, would you prefer I set up its built-in "Deployment" feature? This would upload your Windows edits to WSL automatically as you save.
