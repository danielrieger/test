# Implementation Plan - IMP Repository Analysis

This plan outlines a systematic analysis of the **Integrative Modeling Platform (IMP)** repository. The goal is to provide a deep architectural and functional understanding of the framework to support the user's thesis and optimize the ongoing SMLM-IMP modeling work.

## User Review Required

> [!IMPORTANT]
> The IMP repository is vast (over 50 modules). This analysis will prioritize the components most relevant to the current project: `pmi`, `rmf`, and `bff`. Please let me know if you would like me to include other specific areas (e.g., `em`, `bayesianem`, or `saxs`).

## Proposed Analysis Steps

### 1. Macro-Architecture & Core Abstractions
Examine the `kernel` module to understand the fundamental building blocks of IMP.
- **Goal**: Document the relationship between `Model`, `Particle`, `Restraint`, and `Sampler`.
- **Focus**: The "Decorator" pattern (how `XYZ`, `Mass`, `AV`, etc., extend particles).

### 2. PMI (Precision Molecular Integrative modeling)
Analyze the high-level modeling framework used in the `smlm_score` pipeline.
- **Files**: `modules/pmi/pyext/src/macros.py`, `samplers.py`, `dof/`, and `topology/`.
- **Goal**: Demystify the `ReplicaExchange` macro and the `DegreesOfFreedom` management.
- **Thesis Detail**: Extract the theoretical basis for cross-replica weight calculations and state management.

### 3. Accessible Volume (AV) Theory & Implementation
Review the `bff` module (Bayesian Fiber Fitting) to understand the geometry of AV particles.
- **Goal**: Analyze how AVs are calculated and how they represent the spatial uncertainty of labeling sites.
- **Connection**: Relate this to the `ScoringRestraintWrapper` in our project.

### 4. Data Persistence & Rich Molecular Format (RMF)
Investigate the `rmf` module and its integration with ChimeraX.
- **Goal**: Understand the serialization hierarchy and why standalone particles (like AVs) require specific handling for visualization.

### 5. Optimization & Feature Audit
Review other IMP modules for potentially useful features.
- **Potential Leads**:
    - `bayesianem`: Scoring for density maps that might overlap with GMM concepts.
    - `score_functor`: Low-level scoring optimizations.

## Open Questions

> [!NOTE]
> - Do you prefer a high-level architectural overview or a code-level deep dive with specific function references?
> - Are there any specific mathematical formulas or algorithms from the IMP papers that you want me to trace in the code?

## Verification Plan

### Automated Verification
- No code changes are proposed in this plan. Verification will consist of providing a structured analysis document (artifact) with clickable file links to the repository.

### Manual Verification
- Review the final analysis report for technical accuracy and relevance to the thesis objectives.
