# Scoring Models and Mathematical Formulations

This document summarizes the mathematical interpretation of the scoring functions currently implemented in the SMLM-IMP pipeline. All three scores are derived from Gaussian likelihood ideas, but they differ in how the experimental data and the structural model are represented and compared.

## 1. Distance and Tree Scoring

The Distance and Tree scores share the same likelihood structure. For each experimental localization $\mathbf{x}_D$, the model is treated as a discrete set of candidate fluorophore positions $\mathbf{x}_{M,j}$. The contribution of one data point is computed as a log-sum-exp over model points:

$$
\mathcal{S}(\mathbf{x}_D \mid M)
=
\ln \left(
\sum_{j \in M}
\frac{w_D}{\sqrt{(2\pi)^3 \lvert \Sigma \rvert}}
\exp\left[
-\frac{1}{2}
(\mathbf{x}_D - \mathbf{x}_{M,j})^\top
\Sigma^{-1}
(\mathbf{x}_D - \mathbf{x}_{M,j})
\right]
\right)
$$

with

$$
\Sigma = \Sigma_D + \Sigma_M
$$

where:

- $\Sigma_D$ is the covariance associated with the experimental localization.
- $\Sigma_M$ is the model-side covariance.
- $w_D$ is an optional data-point weight.

At present, the model covariance is approximated by a fixed isotropic term:

$$
\Sigma_M = \sigma_{av}^2 \mathbf{I}
$$

with $\sigma_{av} = 8.0$ in the current implementation.

### Distance Score

The Distance score evaluates the full pairwise relation between all data points and all model points, with direct complexity $O(NM)$.

### Tree Score

The Tree score preserves the same likelihood interpretation but accelerates evaluation by using a KD-tree to restrict the candidate set. In the current implementation, small systems are evaluated exactly, whereas larger systems use radius-based candidate selection. The Tree score is therefore primarily a computational acceleration of the same underlying Gaussian likelihood idea.

## 2. Current GMM Score

The current GMM score represents the experimental data as a Gaussian Mixture Model and evaluates the model point cloud under this mixture:

$$
\mathcal{S}_{GMM}
=
\sum_{j \in M}
\ln \left(
\sum_{k \in G_D}
\pi_k\,
\mathcal{N}(\mathbf{x}_{M,j} \mid \mu_k, \Sigma_k + \Sigma_M)
\right)
$$

where:

- $\pi_k$ is the weight of data component $k$,
- $\mu_k$ and $\Sigma_k$ are the mean and covariance of component $k$,
- $\Sigma_M$ is the model-side covariance, currently still treated heuristically.

This formulation is best interpreted as a **point-cloud likelihood under a data-derived mixture model**. Compared with the earlier implementation, the important improvement is that the GMM is now treated as a true mixture: the component contributions are summed first and the logarithm is applied afterward.

### Dimensionality Awareness
The GMM engine is inherently dimensionality-aware. When evaluated on strictly 2D flat data (like EMAN2 extracted particles), it automatically identifies and fits only the active dimensions, avoiding singular covariance matrix failures. The inactive dimension is padded with a base regularization covariance to maintain a stable 3D likelihood space.

### Important Limitation

The current GMM implementation remains asymmetric:

- the **data** are represented as a Gaussian mixture,
- the **model** is represented as points with a heuristic covariance term.

Accordingly, the current method is still:

```text
model point cloud under data density
```

rather than:

```text
model density against data density
```

For a fuller discussion of the old formulation, the current state, and the longer-term target, see [GMM Overview and Roadmap](gmm_overview_and_roadmap.md).

## 3. Heuristic Assumptions

Several quantities are still represented by practical approximations rather than experimentally calibrated or AV-derived estimates.

- **Fixed model uncertainty**: $\sigma_{av} = 8.0$ is currently used as a global model-side standard deviation.
- **Amplitude-to-variance conversion**: experimental precision is approximated from localization amplitude as $\sigma_D \approx 1/\sqrt{A}$.
- **Unit conversion**: structural coordinates are typically in Angstroms, whereas experimental data are often in nanometers. The default conversion is handled through the `ScoringRestraintWrapper` scaling parameter (`0.1`).

## 4. Recommended Scientific Interpretation

For scientific writing, the current methods are most accurately described as follows:

- **Distance** and **Tree** implement Gaussian point-to-point likelihood models, with Tree providing a spatially accelerated evaluation strategy.
- **GMM** implements an asymmetric point-cloud likelihood model in which model points are evaluated under a GMM fitted to the experimental data.

The current GMM implementation should not yet be described as a full symmetric density-overlap formalism. That fuller target would require probabilistic representations on both the model side and the data side.

> [!WARNING]
> The present `sigma_av` value is a heuristic and should ideally be replaced by AV-derived covariance estimates in future iterations.
