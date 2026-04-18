# Scoring Models and Mathematical Formulations

This document details the mathematical implementation of the scoring functions in the SMLM-IMP pipeline. All scores are fundamentally derived from Gaussian log-likelihoods.

## 1. Distance & Tree Scoring
Both models assume that each experimental SMLM localization $\mathbf{x}_D$ is a sample from a multivariate Gaussian distribution centered at some unknown model point $\mathbf{x}_M$.

### Formulation
The score for a single data point $\mathbf{x}_D$ against the entire model $M$ is computed as the log-sum-exp of individuals Gaussian terms:
$$ \mathcal{S}(\mathbf{x}_D | M) = \ln \left( \sum_{j \in M} \frac{w_D}{\sqrt{(2\pi)^3 |\Sigma|}} \exp\left( -0.5 (\mathbf{x}_D - \mathbf{x}_{M,j})^\top \Sigma^{-1} (\mathbf{x}_D - \mathbf{x}_{M,j}) \right) \right) $$
Where:
- $\Sigma = \Sigma_D + \Sigma_M$ is the combined covariance (data uncertainty + model uncertainty).
- $\Sigma_M$ is currently a fixed heuristic $\sigma_{av}^2 \mathbf{I}$, where $\sigma_{av} = 8.0 \text{ \AA}$.

### Implementation Differences
- **Distance Score**: Computes the full $O(N \cdot M)$ pairwise interaction.
- **Tree Score**: Uses an `sklearn.KDTree` to approximate the log-sum-exp by evaluating only the $K$-nearest neighbors within a specified search radius. This provides a $\sim 10,000\times$ speedup for large datasets.

## 2. GMM Scoring (Density Overlap)
The GMM score treats the data as a Gaussian Mixture Model $G_D$ and calculates the log-likelihood of model points $M$ under this mixture.

$$ \mathcal{S}_{GMM} = \sum_{j \in M} \ln \left( \sum_{k \in G_D} \pi_k \mathcal{N}(\mathbf{x}_{M,j} | \mu_k, \Sigma_k) \right) $$

### Known Limitations
- **Heuristic Nature**: The current implementation fits a single global GMM to the data. It is highly sensitive to the initial alignment and the choice of the number of components (optimized via BIC).
- **Asymmetry**: Unlike a true symmetric overlap integral, the current code evaluates model points against the data GMM components.

## 3. Heuristic Assumptions

The following parameters are currently set to fixed "expert" values based on the bachelor thesis findings:
- **$\sigma_{av} = 8.0$**: Representing the isotropic variance of the Accessible Volume (AV) positions. 
- **Amplitude-to-Variance**: Data precision $\sigma_D$ is estimated from the localization amplitude $A$ as $\sigma_D \approx 1/\sqrt{A}$.
- **Unit Consistency**: Internal scoring typically operates in **nanometers (nm)** when using the default pipeline scaling (0.1). 

> [!CAUTION]
> **Heuristic Inconsistency Detected**: 
> In the current implementation, `sigma_av` is hardcoded to `8.0`. In the standard pipeline where coordinates are scaled to nanometers (nm), this value represents **8.0 nm** (80 Å). If your intended heuristic was **8.0 Å**, you should set this value to `0.8` in the source code or use Ångstroms as your primary coordinate system.

