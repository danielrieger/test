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
  - Replica Exchange Monte Carlo (Bayesian posterior sampling with animated RMF trajectories)
- **HDBSCAN clustering**: Automated NPC isolation from dense SMLM fields
- **PCA alignment**: Model-data registration
- **Validation framework**: Separation tests and held-out cross-validation
- **97 pytest tests**: Unit, integration, and robustness coverage

## Quick Start (WSL / Linux)

The primary supported environment is **WSL2 (Ubuntu)** using **Miniforge**.

### 1. Requirements
- **Conda/Miniforge** with Python 3.11
- **IMP (Integrative Modeling Platform)** installed via conda:
  ```bash
  conda install -c salilab imp
  ```

### 2. Installation
```bash
git clone https://github.com/danielrieger/test.git
cd test
pip install -e .
```

### 3. Running the Pipeline
The examples are pre-configured to run from the project root:
```bash
python examples/NPC_example_BD.py
```

### 4. Configuration
The pipeline is entirely driven by `pipeline_config.json`. For a detailed guide on all available parameters and optimization options, see the documented template:
- [Pipeline Configuration Reference](examples/pipeline_config_template.jsonc)

## Documentation

For detailed information on the underlying physics and heuristics, see:
- [Scoring Models & Mathematics](docs/scoring_models.md)
- [Unit Testing & Validation](tests/README.md)

> [!IMPORTANT]
> **Unit Note**: The pipeline's mathematical engines operate in **nanometers (nm)** by default. Ensure your `sigma_av` and `searchradius` parameters are scaled appropriately (e.g., 5.0 for 50 Å).


## Input Data

The following large input files are **not included** in this repository. Download them and place them in the indicated directories:

| File | Size | Source | Destination |
|------|------|--------|-------------|
| `7N85-assembly1.cif` | 112 MB | [RCSB PDB: 7N85](https://www.rcsb.org/structure/7N85) | `examples/PDB_Data/` |
| `data.csv` | 29 MB | [ShareLoc repository](https://shareloc.xyz) | `examples/ShareLoc_Data/` |
## Visual Gallery

The pipeline generates publication-quality visualizations for structural alignment and quality control.

````carousel
![Top-Ranked NPC (Cluster 347) — GMM Overlay](examples/figures/qc/gmm_cluster_overlay_rank0_id347.png)
<!-- slide -->
![Stylized 3D Isosurface Model](examples/figures/methodology/npc_isosurface_3d.png)
<!-- slide -->
![Structural Alignment PCA Summary](examples/figures/methodology/alignment_summary_pca.png)
<!-- slide -->
![Iterative 2D Fitting Sequence](examples/figures/methodology/fitting_sequence_2d.png)
````

## Performance Benchmarking

A core technical contribution of this work is the implementation of a **Gaussian Mixture Model (GMM)** scoring engine that achieves constant-time evaluation relative to the number of experimental localizations ($N$).

- **Distance/Tree Engines**: Scale linearly $O(N)$ or $O(N \log N)$, becoming a bottleneck at $>10,000$ points.
- **GMM Engine**: After an initial $O(N)$ fitting step, evaluation complexity is $O(GK)$, where $G$ is the number of Gaussians and $K$ is the number of subunits. This results in **constant-time performance** for Bayesian optimization.

![GMM Evaluation Scaling Trends](examples/figures/benchmarks/bench_figA_scaling.png)

## Quick Start

1. **Setup**: Clone the repo and install the environment (see Setup above).
2. **Data**: Place `7N85-assembly1.cif` in `examples/PDB_Data/` and `data.csv` in `examples/ShareLoc_Data/`.
3. **Run Pipeline**: `python examples/NPC_example_BD.py`
4. **Generate Thesis Figures**:
   - `python examples/visualize_alignment_stylized_3d.py` (3D Gallery)
   - `python examples/visualize_gmm_selection.py` (BIC & GMM QC)
   - `python examples/benchmark_scoring.py` (Performance Scaling)

## Project Structure

```
smlm_score/
├── src/
│   ├── imp_modeling/
│   │   ├── scoring/             # Tree, GMM, Distance scoring + CUDA kernels
│   │   └── restraint/           # IMP restraint wrappers (ScoringRestraintWrapper)
│   ├── utility/
│   │   ├── data_handling.py     # Structural ranking, HDBSCAN, PCA alignment
│   │   └── visualization.py     # Stylized 3D (Pyvista) & Publication White themes
├── examples/
│   ├── figures/                 # Categorized Thesis Assets
│   │   ├── methodology/         # 3D, PCA summary, fitting sequences
│   │   ├── benchmarks/          # Scaling and performance plots
│   │   └── qc/                  # GMM BIC selection and top-ranked overlays
│   ├── benchmark_scoring.py     # Performance scaling benchmark
│   └── visualize_gmm_selection.py # Intelligent NPC selection & BIC plots
└── tests/                       # 97 pytest tests
```

## Advanced Workflows

### 1. Bayesian Trajectory Visualization (RMF)
When running in `bayesian` optimization mode, the pipeline generates two state-of-the-art RMF3 trajectories in the output directory (e.g., `bayesian_cluster_11/`):
- `av_trajectory.rmf3`: A lightweight file containing only the moving fluorophore center points (AVs).
- `full_trajectory.rmf3`: A high-fidelity reconstruction containing all ~15,000 protein beads. The structural pose is recovered from sampled AV coordinates using a rigid-body SVD transformation (Kabsch algorithm).

**To visualize in ChimeraX:**
1. Open `full_trajectory.rmf3`.
2. Use the **Log** or **Tools > General > Playback** to animate the REMC sampling steps.
3. You will see the entire protein assembly moving as a unified rigid body.

### 2. EMAN2 Particle Picking
The pipeline supports targeted modeling of particles manually or automatically picked via EMAN2:
1. Run `e2boxer.py` on your `micrograph.mrc` to pick NPCs.
2. Ensure `examples/info/micrograph_info.json` exists (contains the box coordinates).
3. Set `"clustering": { "method": "eman2" }` in `pipeline_config.json`.
4. The pipeline will automatically slice the SMLM data into `examples/picked_particles/` and optimize the targets.

*Technical Note: If your box metadata is lost, use `examples/recover_boxes.py` to rebuild it from existing CSV fragments.*

## Maintenance & Data Safety

### Synchronization (Windows/WSL)
Because input data (PDBs, SMLM CSVs) and EMAN2 results are often excluded from Git due to size, use the provided `safe_sync.sh` script to align development environments. 

**Note: WSL is the Primary Source of Truth.**
```bash
# Run in WSL to push code/config changes to the Windows workspace:
./safe_sync.sh
```

This script uses `rsync --update` and explicit excludes to ensure that **untracked local data in WSL is protected** while keeping the Windows workspace synchronized as a mirror.

## Validation

The validation module implements two tests:
1. **Separation test**: Confirms that density-normalized scores for valid NPC clusters outperform noise/off-target clusters.
2. **Held-out test**: Verifies that the structural signal is captured by comparing scores across spatially disjoint subsets of the same NPC localization cloud.

## License

TBD

## Testing

```bash
pytest tests/
```

Expected result: **92 passed, 5 skipped** (skipped tests require CUDA).

## Citation

TBD
