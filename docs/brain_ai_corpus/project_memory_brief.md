# Project Memory Brief

## Project Identity

`smlm_score` is an SMLM-to-IMP pipeline focused on isolating single NPC structures from localization microscopy data and scoring them against a structural model. The codebase combines data filtering, clustering, validation, and structural scoring with multiple optimization/sampling modes.

## Core Concepts

- Domain: SMLM localization data, NPC structural analysis, IMP-based modeling
- Main scoring engines: Distance, Tree, GMM
- Main workflow: ingest data -> filter ROI -> cluster NPCs -> select target NPC -> score against model -> validate -> optionally optimize/sample
- Main execution styles: Bayesian REMC, frequentist optimization, Brownian dynamics planning

## Stable Understanding

- The project repeatedly works on correctness, robustness, and reproducibility rather than only adding features.
- Validation quality is a central concern and evolved significantly over time.
- Clustering and filtering behavior are highly configuration-driven.
- Bayesian sampling and RMF trajectory output were major debugging areas.
- Windows/WSL sync safety matters because the project mixes code, data, and generated research artifacts.

## Development Timeline

### Early March 2026

- Validation failures were analyzed and normalized more carefully.
- CUDA acceleration for scoring was explored.
- The team clarified score semantics and held-out behavior.

### Mid to Late March 2026

- The thesis/pipeline architecture was formalized.
- The codebase was cross-checked against IMP/tutorial concepts.
- Benchmarking established the scaling story for Distance, Tree, and GMM.
- Frequentist optimization and Brownian dynamics directions were planned.
- Configurable filtering and random ROI selection were added.
- Clustering robustness improved, especially for memory-heavy cases.
- Visualization support for clustering/thesis figures was added.

### April 2026

- The project shifted into infrastructure, reorganization, logging, and environment hardening.
- EMAN2 support and surrounding tooling were added.
- Bayesian REMC usability improved via score scaling, progress reporting, and exposed motion parameters.
- RMF trajectory generation was fixed so visible structure motion matched actual sampling.
- Validation was redesigned toward cross-validated structural testing with angular splitting and scrambled null controls.
- Safe sync/versioning procedures were formalized for Windows and WSL.
- Technical debt cleanup modularized the main example pipeline and fixed path/config issues.
- The latest direction is a mathematically cleaner GMM mixture likelihood.

## Most Important Documents

Use these first when priming an AI assistant:

- `docs/brain_ai_ingestion_summary.md`
- `docs/brain_ai_corpus/project_memory.md`
- `walkthrough_stabilization.md`
- `implementation_plan_tech_debt.md`
- `validation_analysis.md`
- `walkthrough_validation.md`
- `NPC_Modeling_Pipeline_Overview.md`
- `NPC_pipeline_manual.md`
- `implementation_plan_gmm_mixture.md`

## Current Likely State

The project appears to have reached a relatively mature and stabilized architecture, with the main open frontier shifting from infrastructure/debugging to mathematical refinement and scoring correctness. The strongest recent themes are:

- preserving modularity without losing behavior
- making validation scientifically stronger
- keeping Bayesian sampling interpretable and visible
- improving environment reliability and sync safety
- revisiting the GMM score so it matches a proper mixture-likelihood interpretation

## Known Problem Areas

- score normalization and score semantics across different engines
- clustering correctness versus scalability on large datasets
- configuration-dependent behavior causing apparent regressions
- path/environment issues when running from different IDEs or directories
- synchronization safety between Windows and WSL workspaces
- potential mismatch between current GMM implementation and desired probabilistic interpretation

## Open Questions For Future Work

- Should GMM become the mathematically preferred score after the mixture-likelihood cleanup?
- How should structural validation results be reported in the thesis to balance rigor and readability?
- Which execution mode should be treated as the primary scientific path: Bayesian, frequentist, or BD?
- Which parts of the pipeline are now stable enough to freeze, and which remain experimental?

## Best Use With AI

For broad project understanding, start with `project_memory_brief.md` and then add either:

- `project_memory.md` for one-file deep context
- `timeline.jsonl` for retrieval/embedding workflows
- selected latest implementation and walkthrough docs for focused coding help
