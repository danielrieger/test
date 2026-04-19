# Implementation Plan - Fixing Frozen RMF Trajectory

The user reported that the sampling trajectory appears frozen in Chimera, even though the internal scores and statistics indicate active sampling. Analysis revealed that while the REMC sampler is moving particles, the structural hierarchy (the protein) is not linked to these movers, and the AV particles are not properly integrated into the visible RMF hierarchy.

## User Review Required

> [!IMPORTANT]
> **Structural Integrity**: I will modify the rigid body setup to include the **entire protein complex** along with the AVs. This ensures that when the sampler searches for a better fit, the protein moves in unison with the fluorophore attachment sites (AVs).

## Proposed Changes

### 1. Linking AVs to the Visualization Hierarchy
Currently, `AV` particles are created as floating particles in the model. RMF only records the hierarchy starting from the root.

#### [MODIFY] [data_handling.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/utility/data_handling.py)
- In `compute_av`, after creating the `av_p` particle, I will add it as a child of the `hier` (PDB hierarchy).
- This ensures that the AV particles are included in any RMF file generated from the main hierarchy.

### 2. Including the Protein in the Rigid Body Mover
Currently, only the `AV` particles are grouped into the rigid body. This moves the fluorophore centers but leaves the protein skeleton static.

#### [MODIFY] [mcmc_sampler.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/imp_modeling/simulation/mcmc_sampler.py)
- Update `run_bayesian_sampling` to include the `pdb_hierarchy` (protein) in the `create_rigid_body` call.
- By moving the entire hierarchy as a single rigid body, the structural search will explore translations and rotations of the whole NPC complex relative to the SMLM data.

### 3. RMF Trajectory Polish
#### [MODIFY] [mcmc_sampler.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/imp_modeling/simulation/mcmc_sampler.py)
- Update the custom `av_trajectory.rmf3` writer to explicitly set the active frame before writing coordinates, ensuring 100% compatibility with visualization tools.

## Open Questions

- **Refining vs. Global**: Do you want the protein structure to be entirely rigid, or should we keep the option to move AVs slightly independently? (Initial plan assumes a unified rigid body for the whole complex to fix the "frozen" visual issue).

## Verification Plan

### Automated Tests
- Run a short REMC simulation (10 frames) in WSL.
- Manually inspect the `stat.0.out` to confirm `Total_Score` and `AV` coordinates are varying.

### Manual Verification
- Open the resulting `0.rmf3` and `av_trajectory.rmf3` in ChimeraX.
- Verify that the protein spheres move together with the AV markers.
