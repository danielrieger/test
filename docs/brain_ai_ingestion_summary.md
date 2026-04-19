# Brain Folder AI Ingestion Summary

## What this folder contains

The `brain` folder appears to be an exported antigravity-style working memory archive covering roughly **2026-03-03 through 2026-04-19**. The valuable project knowledge is concentrated in markdown artifacts such as:

- `task*.md`
- `implementation_plan*.md`
- `walkthrough*.md`
- analysis/review documents such as `validation_analysis.md`, `validation_output_analysis.md`, `imp_architectural_analysis.md`, and `thesis_code_crossreference.md`

The folder also contains a large amount of lower-value supporting material:

- `.resolved` snapshots and numbered revisions
- `.metadata.json` sidecars
- screenshot/media caches in `.tempmediaStorage`
- generated PNG/WebP figures
- browser scratch files

For AI project memory, the markdown artifacts are the primary source of truth. The media/cache layers should usually be excluded unless you specifically want vision models to inspect figures.

## Recommended AI ingestion strategy

Use the folder as a **project-memory corpus**, not as a model-training dataset.

Best practice:

1. Ingest only `.md` files first.
2. Exclude `.resolved`, `.metadata.json`, `.tempmediaStorage`, `browser`, and duplicate `content.md` stubs.
3. Sort documents by `LastWriteTime`.
4. Group them into chronological phases.
5. Feed the resulting timeline plus the latest implementation/walkthrough docs into your AI assistant as persistent context or retrieval documents.

Suggested tiers:

- Tier 1: `implementation_plan*.md`, `walkthrough*.md`, `task*.md`
- Tier 2: `validation_analysis.md`, `validation_output_analysis.md`, `imp_architectural_analysis.md`, `thesis_code_crossreference.md`, benchmark summaries, manuals
- Tier 3: figures and screenshots only when a visual question comes up

## High-level project knowledge

This archive documents the evolution of an **SMLM-IMP / NPC modeling pipeline**. The recurring themes are:

- scoring engines: Distance, Tree, and GMM
- validation correctness and normalization
- NPC clustering robustness and visualization
- configurable filtering and ROI selection
- Bayesian REMC sampling, frequentist optimization, and Brownian dynamics support
- EMAN2 integration and tooling
- RMF trajectory correctness and full-structure visualization
- repository reorganization, sync safety, and technical debt cleanup
- emerging mathematical cleanup of the GMM likelihood

## Chronological development timeline

### Phase 1: Early validation and acceleration work

**2026-03-04 to 2026-03-05**

- CUDA acceleration work was planned and tracked for scoring kernels with CPU fallback.
- Validation failures were analyzed in depth.
- Root issues included inconsistent score semantics, especially around score normalization and held-out scoring behavior.

Key artifacts:

- `task.md` in `cc2c3305-...`: CUDA acceleration for scoring
- `implementation_plan.md` in `cc2c3305-...`: corrected validation failure analysis
- `walkthrough.md` in `cc2c3305-...`: validation fixes walkthrough

### Phase 2: Thesis framing and pipeline formalization

**2026-03-17 to 2026-03-24**

- The codebase was compared against IMP/tutorial architecture to assess conceptual alignment.
- A critical review of `NPC_example_BD.py` identified correctness risks in cluster targeting.
- The full SMLM-to-IMP pipeline was documented as a staged workflow.
- A user/manual layer was added for operating the NPC pipeline.
- Benchmark documents established scaling behavior of Distance, Tree, and GMM scoring.
- Two future directions were explored:
  - frequentist optimization path
  - Brownian dynamics mode
- Robustness/optimization fixes were consolidated into a project walkthrough.
- Configurable data filtering was added, including random spatial windows and ROI handling.

Key artifacts:

- `thesis_code_crossreference.md`
- `code_review_NPC_example_BD.md`
- `NPC_Modeling_Pipeline_Overview.md`
- `NPC_pipeline_manual.md`
- `benchmarking_thesis_results.md`
- `benchmarking_thesis_results_experimental.md`
- `implementation_plan_frequentist.md`
- `implementation_plan_brownian.md`
- `walkthrough.md`
- `task.md`
- `implementation_plan.md`

### Phase 3: Clustering, memory safety, and validation reliability

**2026-03-27 to 2026-03-30**

- A memory crash in stage-4 clustering was addressed by switching large datasets from point-wise agglomeration to cluster-level merging.
- Workspace cleanup removed generated artifacts and caches.
- The pipeline was hardened for 2D data and random ROI held-out complement logic.
- Validation reliability improved by fixing payload consistency and making normalization scoring-type aware.
- A dedicated clustering visualization workflow was added for thesis figures.
- Repository upload/versioning planning began.

Key artifacts:

- `walkthrough_clustering_fix.md`
- `walkthrough_workspace_cleanup.md`
- `walkthrough_robustness_fixes.md`
- `walkthrough_validation_reliability.md`
- `walkthrough_clustering_visualization.md`
- `task.md`
- `implementation_plan.md`
- `task_github.md`

### Phase 4: April architecture and workflow expansion

**2026-03-31 to 2026-04-14**

- The project entered a heavy implementation-planning period.
- Major themes included:
  - stylized output/documentation
  - fitting sequence design
  - repository reorganization
  - ranking logic
  - synchronization behavior
  - radius handling
  - logging
  - clone/fork implementation support
  - infrastructure and environment fixes
  - sync strategy
  - EMAN2 integration
  - tree speed improvements
  - master benchmarking
  - coordinate logging
  - architectural analysis of IMP internals and implications

This phase looks like the transition from a working thesis prototype to a more maintainable, reproducible, tool-supported research pipeline.

Representative artifacts:

- `implementation_plan_reorg.md`
- `implementation_plan_reorg_v2.md`
- `walkthrough_reorg.md`
- `guide_tools_wsl.md`
- `implementation_plan_environment_fix.md`
- `walkthrough_final_migration.md`
- `implementation_plan_sync_strategy.md`
- `implementation_plan_eman2.md`
- `walkthrough_eman2.md`
- `implementation_plan_tree_speed.md`
- `implementation_plan_master_benchmark.md`
- `implementation_plan_coord_logging.md`
- `implementation_plan_imp_analysis.md`
- `task_imp_analysis.md`
- `imp_architectural_analysis.md`

### Phase 5: Sampling, trajectory correctness, validation redesign, and safe sync

**2026-04-15 to 2026-04-16**

- Bayesian/REMC performance was improved by exposing sampler parameters and introducing score scaling.
- A major RMF trajectory issue was fixed so that the entire structure and AVs move together and are visible in output trajectories.
- Project finalization focused on data safety, protected directories, documentation updates, and safe synchronization.
- Validation underwent a major redesign:
  - old noise-cluster comparisons were judged too weak
  - a cross-validated structural validation method ("Strategy B") was introduced
  - angular splitting and scrambled null controls were added
- GitHub versioning and WSL synchronization were formalized with safety checks.

Key artifacts:

- `implementation_plan_imp_analysis_consequences.md`
- `walkthrough_remc_improvements.md`
- `task_remc_improvements.md`
- `implementation_plan_rmf_fix.md`
- `walkthrough_rmf_fix.md`
- `implementation_plan_finalization_safety.md`
- `task_finalization.md`
- `walkthrough_finalization.md`
- `validation_analysis.md`
- `implementation_plan_validation.md`
- `task_validation.md`
- `walkthrough_validation.md`
- `implementation_plan_sync_maintenance.md`
- `task_sync_maintenance.md`
- `walkthrough_sync_maintenance.md`

### Phase 6: Stabilization, technical debt, and next mathematical refinement

**2026-04-17 to 2026-04-19**

- Technical debt and correctness cleanup targeted:
  - path resolution failures
  - held-out validation variable issues
  - configuration/flag gating
  - score weight propagation
  - modularization of `NPC_example_BD.py`
- Validation outputs were reinterpreted in terms of configuration rather than code regressions.
- The pipeline was described as stabilized and modular.
- Further stabilization focused on score-weight history and remaining unbound/final fixes.
- The latest artifact introduces a more principled **GMM mixture likelihood** formulation using a proper mixture log-likelihood and log-sum-exp reduction.

Key artifacts:

- `implementation_plan_tech_debt.md`
- `task_tech_debt.md`
- `validation_output_analysis.md`
- `walkthrough_stabilization.md`
- `implementation_plan_unbound_fix.md`
- `implementation_plan_gmm_mixture.md`

## Stable project understanding for future AI sessions

An AI assistant should understand the project like this:

- The project models **single NPC structures from SMLM localization data** against a structural model using **IMP**.
- The workflow includes:
  - ingest/filter SMLM data
  - cluster likely NPCs
  - select/validate target NPCs
  - score against an IMP-derived structural representation
  - optionally optimize/sample with Bayesian, frequentist, or Brownian-style methods
- The three central scoring families are **Distance**, **Tree**, and **GMM**.
- Validation evolved from simple noise-vs-valid comparisons toward **cross-validated structural tests**.
- Robustness work repeatedly focused on:
  - 2D data compatibility
  - ROI/random filtering correctness
  - clustering scalability
  - safe configuration-driven execution
  - reproducible synchronization between Windows and WSL
- Bayesian sampling and RMF visualization were major sources of debugging effort.
- The current frontier topic appears to be **making the GMM score mathematically correct as a true mixture likelihood**.

## What to feed to AI first

If you want an assistant to quickly "know the project," feed these first:

1. This summary
2. Latest stabilization and technical-debt docs
3. Latest validation docs
4. Pipeline overview/manual docs
5. Latest GMM mixture plan

Recommended priority set:

- `walkthrough_stabilization.md`
- `implementation_plan_tech_debt.md`
- `validation_analysis.md`
- `walkthrough_validation.md`
- `NPC_Modeling_Pipeline_Overview.md`
- `NPC_pipeline_manual.md`
- `implementation_plan_gmm_mixture.md`

## Suggested cleanup before AI ingestion

Create a filtered corpus that keeps only:

- top-level `.md` artifacts
- one latest version per artifact name where duplicates exist

Exclude:

- `*.resolved*`
- `*.metadata.json`
- `.tempmediaStorage/`
- `.system_generated/`
- `browser/`
- standalone media unless referenced by a question

## Bottom line

Yes, this folder can absolutely be used to feed AI. The best use is not raw training, but a **retrieval-ready project memory pack** built from the markdown artifacts, ordered by date and trimmed of cache noise. The most important story in the archive is the progression from validation/debugging and pipeline formalization toward stabilization, safe synchronization, stronger structural validation, and a cleaner GMM formulation.
