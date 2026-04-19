# Scoring Models and Mathematical Formulations

This document details the mathematical implementation of the scoring functions in the SMLM-IMP pipeline. All scores are fundamentally derived from Gaussian log-likelihoods.

## 1. Distance & Tree Scoring
Both models assume that each experimental SMLM localization $\mathbf{x}_D$ is a sample from a multivariate Gaussian distribution centered at some unknown model point $\mathbf{x}_M$.

### Formulation
The score for a single data point $\mathbf{x}_D$ against the entire model $M$ is computed as the log-sum-exp of individuals Gaussian terms:
$$ \mathcal{S}(\mathbf{x}_D | M) = \ln \left( \sum_{j \in M} \frac{w_D}{\sqrt{(2\pi)^3 |\Sigma|}} \exp\left( -0.5 (\mathbf{x}_D - \mathbf{x}_{M,j})^\top \Sigma^{-1} (\mathbf{x}_D - \mathbf{x}_{M,j}) \right) \right) $$

## Gaussian Mixture Score (`GMM`)

Models the experimental SMLM localizations as a probability distribution constructed by summing independent Gaussian components (using Scikit-Learn `GaussianMixture`). The structural assembly is then treated as a discrete set of spatial points $X = \{x_j\}$. The score is calculated as the log-likelihood of the structural model under the data distribution:

$$ S_{GMM} = \sum_{j} \log \left( \sum_{k} P(x_j \mid \mathcal{N}(\mu_k, \Sigma_k + \Sigma_M)) \right) $$

where $P(x_j \mid \mathcal{N}_k)$ is the combined probability given by the convolution of the data Gaussian $k$ and the expected model uncertainty $\Sigma_M$. The model uncertainty evaluates point deviations symmetrically.

> [!CAUTION]
> As in Distance, the GMM variance expansion $\Sigma_M$ relies on a hardcoded, heuristic `sigma_av` and does not account for internal subunit flexibilities or specific cross-linker topologies.

## 3. Heuristic Assumptions
The following parameters are currently set to fixed "expert" values based on the bachelor thesis findings:
- **$\sigma_{av} = 8.0 \text{ \AA}$**: Representing the fixed standard deviation of the Accessible Volume (AV) positions.
- **Amplitude-to-Variance**: Data precision $\sigma_D$ is estimated from the localization amplitude $A$ as $\sigma_D \approx 1/\sqrt{A}$.
- **Unit Consistency**: Internal scoring typically operates in Ångstroms, while input data is often in nanometers. The pipeline handles this conversion in the `ScoringRestraintWrapper` scaling parameter (default $0.1$).

> [!WARNING]
> The `sigma_av` value is a heuristic. Future iterations should derive this value from the specific AV linker lengths and radii defined in `av_parameter.json`.
