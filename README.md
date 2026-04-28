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
- **Posterior density mapping**: Automatically generates MRC volumetric density maps and PNG heatmaps from Bayesian sampling ensembles, with score-based filtering and centroid alignment for publication-quality figures
- **HDBSCAN clustering**: Automated NPC isolation from dense SMLM fields
- **PCA alignment**: Model-data registration
- **Validation framework**: 2D-aware alignment and robust model-vs-null structural cross-validation
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

### Documentation

For mathematical details and current method limitations, see:

- [Scoring Models and Mathematical Formulations](docs/scoring_models.md)
- [GMM Overview and Roadmap](docs/gmm_overview_and_roadmap.md)
- [Posterior AV Density Mapping](docs/posterior_density.md)
- [EMAN2 Particle Picking Workflow](docs/eman2_workflow.md)

## Input Data

The following large input files are **not included** in this repository. Download them and place them in the indicated directories:

| File | Size | Source | Destination |
|------|------|--------|-------------|
| `7N85-assembly1.cif` | 112 MB | [RCSB PDB: 7N85](https://www.rcsb.org/structure/7N85) | `examples/PDB_Data/` |
| `data.csv` | 29 MB | [ShareLoc repository](https://shareloc.xyz) | `examples/ShareLoc_Data/` |
## Visual Gallery

The pipeline generates publication-quality visualizations for structural alignment, quality control, and Bayesian posterior analysis.

````carousel
![Final Thesis Result: NPC Box 240 (5nm High-Res)](examples/figures/methodology/thesis_final_result_v4.png)
<!-- slide -->
![Posterior AV Density Map (20,000 Frames)](examples/figures/Posterior/posterior_density_20000f.png)
<!-- slide -->
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
│   ├── docs/
│   │   ├── eman2_workflow.md    # High-Res Picking workflow
│   │   └── scoring_models.md    # Physics/Scoring background
│   ├── examples/
│   │   ├── figures/             # Categorized Thesis Assets
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

### 2. Posterior AV Density Mapping
At the end of every Bayesian run, the pipeline automatically generates a probability density map of the 8 dye-attachment points (AVs) across the full sampling ensemble.

Key features of the density engine:
- **Score-based filtering**: Only the top 25% best-scoring frames are accumulated, discarding high-temperature exploratory poses.
- **Centroid alignment**: Each frame is centered to its AV centroid before accumulation, removing rigid-body translational drift and revealing true structural uncertainty.
- **MRC + PNG output**: Density maps are saved in MRC format (ChimeraX-ready) and as annotated heatmaps.
- **Auto-export**: PNGs are automatically copied to `examples/figures/Posterior/` with a frame-count suffix.

For a detailed technical description, see: [Posterior AV Density Mapping](docs/posterior_density.md).

### 3. EMAN2 Particle Picking (High-Res 5nm)
The pipeline supports state-of-the-art particle picking using EMAN2's neural network autoboxing on high-resolution (5nm) intensity-weighted density maps.

For a detailed step-by-step guide, see: [EMAN2 Workflow & High-Res Picking](docs/eman2_workflow.md).

**Key Upgrades:**
- **5 nm/pixel** resolution for structural clarity.
- **Intensity-weighted** rendering using `Amplitude_0_0`.
- **Targeted Modeling** of 300+ picked NPCs.

*Technical Note: If your box metadata is lost, use `examples/recover_boxes.py` to rebuild it from existing CSV fragments.*

## Maintenance & Data Safety

### Synchronization (Windows/WSL)
Because input data (PDBs, SMLM CSVs) and EMAN2 results are often excluded from Git due to size, use the provided `safe_sync.sh` script to align development environments:

```bash
# In WSL:
./safe_sync.sh
```

This script uses `rsync --update` and explicit excludes to ensure that **untracked local data in WSL is never deleted or overwritten** when pulling code updates from the Windows "Source of Truth" workspace.

## Validation

The validation module implements a robust, EMAN2-aware validation framework:
1. **2D-Aware Alignment**: Correctly handles flattened 2D EMAN2 data by intelligently skipping PCA rotation while preserving XY centering.
2. **Model-vs-Null Structural Validation**: Rigorously evaluates the *best optimized structural pose* from the Bayesian sampler. It splits the localizations into train/test folds and scores the fit against distinct null models:
   - For 3D data: Scrambled points, Rotated model, Mirrored model, Radial randomized points.
   - For 2D data: Scrambled points, Translated model (off-center shift), Radial points, Radial + Translated model. (Rotations and mirroring are replaced by translation since EMAN2 boxes are already projection-centered).
3. **Fallback Separation**: If pure noise clusters are present, verifies that valid NPC clusters score higher than noise (gracefully skipped if only valid NPCs exist).

## License

TBD

## Testing

```bash
pytest tests/
```

Expected result: **98 passed, 5 skipped** (skipped tests require CUDA or depend on specific validation thresholds).

## Citation

TBD
