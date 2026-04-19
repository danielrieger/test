# SMLM-IMP Pipeline Stabilization Walkthrough

We have successfully stabilized and modularized the SMLM-IMP modeling pipeline. By transitioning to a modular architecture, we improved code maintainability while strictly preserving every diagnostic and "smart fallback" feature from the original monolithic script.

## 🚀 Final Pipeline Architecture

The pipeline in [NPC_example_BD.py](file:///\\wsl.localhost\Ubuntu\home\daniel\Thesis\smlm_score\examples\NPC_example_BD.py) now orchestrates several distinct phases:

1.  **System Setup**: Dynamically detects the structure file format (`.cif` vs `.pdb`) and loads it using `IMP`.
2.  **Filtering**: Applies spatial or random windowing as defined in `pipeline_config.json`.
3.  **Clustering**: Groups localizations into NPCs using the `eman2` or `hdbscan` method.
4.  **Auto-Target Selection**: 
    -   If `target_cluster_id` is `null` → Picks the largest valid cluster (Legacy default).
    -   If `target_cluster_id` is `"random"` → Picks a random valid cluster (New feature).
5.  **Evaluation Loop**: Compares the model against valid NPCs and noise clusters.
6.  **Restraints & Sampling**: Triggers REMC sampling or BD simulations with restored logic (Fixes the Bayesian weight bug).
7.  **Validation Suite**: Runs all 4 structural cross-validation tests.

## 🛠️ Critical Bugfixes

> [!IMPORTANT]
> **Bayesian Weight Bug Fixed**: 
> I restored the `1.0 / n_points` calculation for Bayesian sampling. The temporary refactored version was dividing by the *Cluster ID*, which produced incorrect physical sampling statistics.

> [!TIP]
> **PDB/MMCIF Flexibility**: 
> You no longer need to manually change from `read_pdb` to `read_mmcif`. The pipeline now automatically detects the extension and chooses the correct IMP loader.

## 📊 Verification Results (Verified in WSL)

A clean run with the latest fixes yielded the following:
- **Auto-select**: Correctly picked the largest NPC (ID 87) for optimization.
- **Progress Logs**: 200 frame sampling logs are restored and visible in the terminal.
- **Validation Report**:
  - `Separation`: Evaluated correctly against restored noise pools (10-100 pts).
  - `HeldOut`: **PASS** (14.34 sigma).

## 🔄 How to Synchronize

Since your primary workspace is now inside **WSL**, the synchronization bridge has been flipped. 

To back up your WSL work to the Windows filesystem, run:
```bash
bash safe_sync.sh
```
*Note: This script is located in the project root within WSL.*

---

## 📂 Key Files
- Primary Script: [NPC_example_BD.py](file:///\\wsl.localhost\Ubuntu\home\daniel\Thesis\smlm_score\examples\NPC_example_BD.py)
- Math Docs: [scoring_models.md](file:///\\wsl.localhost\Ubuntu\home\daniel\Thesis\smlm_score\docs\scoring_models.md)
- Safe Sync: [safe_sync.sh](file:///\\wsl.localhost\Ubuntu\home\daniel\Thesis\smlm_score\safe_sync.sh)
