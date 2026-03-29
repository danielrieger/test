# SMLM-IMP Score

Bayesian scoring of Single-Molecule Localization Microscopy (SMLM) data against Integrative Modeling Platform (IMP) structural models of the Nuclear Pore Complex (NPC).

## Overview

**smlm_score** provides a complete pipeline for evaluating how well a structural model explains experimentally observed SMLM localization patterns. It implements three scoring functions based on Bayesian log-likelihood formulations (Bonomi et al. 2019), supports GPU-accelerated computation, and offers multiple optimization strategies for structural fitting.

## Features

- **Three scoring functions**: Tree (KD-tree), GMM (Gaussian Mixture Model), Distance (pairwise)
- **GPU acceleration**: CUDA kernels via Numba with automatic CPU fallback
- **Three optimization modes**:
  - Brownian Dynamics (geometric relaxation)
  - Conjugate Gradients (frequentist MLE)
  - Replica Exchange Monte Carlo (Bayesian posterior sampling)
- **HDBSCAN clustering**: Automated NPC isolation from dense SMLM fields
- **PCA alignment**: Model-data registration
- **Validation framework**: Separation tests and held-out cross-validation
- **97 pytest tests**: Unit, integration, and robustness coverage

## Installation

### Prerequisites

- Python ≥ 3.11
- [IMP](https://integrativemodeling.org/) with the `IMP.bff` module
- CUDA toolkit (optional, for GPU acceleration)

### Setup

```bash
git clone https://github.com/danielrieger/test.git
cd test
pip install -e .
```

### Environment

This project was developed with a conda-pack environment (Python 3.11). Core dependencies are listed in `requirements.txt`. The IMP library must be installed separately following the [IMP installation guide](https://integrativemodeling.org/nightly/doc/manual/installation.html).

## Input Data

The following large input files are **not included** in this repository. Download them and place them in the indicated directories:

| File | Size | Source | Destination |
|------|------|--------|-------------|
| `7N85-assembly1.cif` | 112 MB | [RCSB PDB: 7N85](https://www.rcsb.org/structure/7N85) | `examples/PDB_Data/` |
| `data.csv` | 29 MB | [ShareLoc repository](https://shareloc.xyz) | `examples/ShareLoc_Data/` |

## Quick Start

1. Download and place the input data files (see above).
2. Configure the pipeline via `examples/pipeline_config.json`.
3. Run the example:

```bash
python examples/NPC_example_BD.py
```

This runs the full pipeline: load data → cluster → align → score → optimize → validate.

## Project Structure

```
smlm_score/
├── src/
│   ├── imp_modeling/
│   │   ├── scoring/             # Tree, GMM, Distance scoring + CUDA kernels
│   │   ├── restraint/           # IMP restraint wrappers with gradient support
│   │   ├── model_setup/         # Top-level model container
│   │   ├── brownian_dynamics/   # BD simulation setup
│   │   └── simulation/          # CG optimizer + REMC sampler
│   ├── utility/
│   │   ├── data_handling.py     # SMLM filtering, clustering, AV computation
│   │   ├── input.py             # CSV/JSON loading
│   │   ├── plot.py              # GMM diagnostics plots
│   │   └── visualization.py     # Publication-quality figures
│   ├── validation/
│   │   └── validation.py        # Separation + held-out validation
│   └── benchmarking/
│       └── gmm_benchmarks.py    # GMM performance benchmarks
├── tests/                       # 97 pytest tests (unit + integration)
├── examples/                    # Pipeline examples and scripts
├── pyproject.toml               # Package metadata
└── requirements.txt             # Python dependencies
```

## Configuration Reference

The pipeline is configured via `examples/pipeline_config.json`:

```json
{
    "paths": {
        "smlm_data": "ShareLoc_Data/data.csv",
        "pdb_data": "PDB_Data/7N85-assembly1.cif",
        "av_parameters": "av_parameter.json"
    },
    "filtering": {
        "type": "random | filter | full",
        "filter": { "x_range": [min, max], "y_range": [min, max] },
        "random": { "size_percentage": 15 }
    },
    "clustering": {
        "min_cluster_size": 15,
        "min_npc_points": 120,
        "perform_geometric_merging": false
    },
    "optimization": {
        "mode": "bayesian | frequentist | brownian",
        "brownian": { "temperature_k": 300, "max_time_step_fs": 50000, "number_of_bd_steps": 500 },
        "frequentist": { "scoring_type": "Tree", "max_cg_steps": 200 },
        "bayesian": { "scoring_type": "GMM", "number_of_frames": 20, "monte_carlo_steps": 10 }
    }
}
```

## Scoring Functions

### Tree Score

KD-tree-backed log-likelihood with search radius pruning. Supports analytical gradients for optimization. Efficient for small model sizes (falls back to exact computation when the number of AVs is small).

### GMM Score

Fits a Gaussian Mixture Model to the SMLM data using BIC-optimal component selection, then evaluates the log-likelihood of model AV positions under the GMM. Uses CUDA GPU kernels for large datasets.

### Distance Score

Pairwise distance log-likelihood with log-sum-exp numerical stability. Each data point contributes via a Gaussian kernel centered on the model-data distance, with combined model + data covariance (Bonomi et al. 2019).

## Optimization Modes

### Brownian Dynamics

Geometric relaxation via `IMP.atom.BrownianDynamics`. Configures AV particles with diffusion coefficients and XYZR decorators. Outputs RMF trajectories for visualization.

### Conjugate Gradients (Frequentist)

Single-point MLE optimization via `IMP.core.ConjugateGradients`. Requires gradient-compatible scoring (Tree or Distance). Reports coordinate shifts and objective improvement.

### Replica Exchange Monte Carlo (Bayesian)

Posterior sampling via `IMP.pmi.macros.ReplicaExchange`. Groups AVs into a rigid body and samples structural conformations with Metropolis acceptance.

## Validation

The validation module (`src/validation/validation.py`) implements two tests:

1. **Separation test**: Compares density-normalized scores of valid NPC clusters against noise clusters. All scoring types should show valid > noise (less negative = better fit).

2. **Held-out test**: Compares valid NPC scores against scores computed on spatially disjoint held-out data. Tree scoring yields 0 for held-out data (no neighbors within radius), which is the expected behavior.

## Testing

```bash
pytest tests/
```

Expected result: **92 passed, 5 skipped** (skipped tests require CUDA).

## License

TBD

## Citation

TBD
