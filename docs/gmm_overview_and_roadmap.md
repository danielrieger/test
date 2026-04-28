# GMM Overview and Roadmap

This document summarizes the conceptual development of the Gaussian Mixture Model (GMM) scoring procedure in `smlm_score`. The goal is to distinguish clearly between the earlier heuristic formulation, the current improved implementation, and a possible future extension toward a fuller density-overlap model.

## 1. Conceptual Overview

The development of the GMM score can be understood in three stages:

- the **earlier implementation**, which behaved as a heuristic sum of pairwise log-overlaps,
- the **current implementation**, which evaluates the model point cloud under a proper data-side Gaussian mixture,
- the **future target**, in which both the model and the data are represented as densities and compared symmetrically.

The current code should therefore be regarded as an intermediate but meaningful improvement: mathematically stronger than the original heuristic version, yet still distinct from a full density-vs-density Bayesian overlap model.

## 2. Earlier Formulation

The earlier implementation behaved effectively like:

```text
S_old = sum_j sum_k log( pi_k * N(x_M,j | mu_k, Sigma_k + Sigma_M) )
```

where:

- $j$ indexes model points,
- $k$ indexes GMM components on the data side,
- $\pi_k$ is the component weight,
- $\Sigma_M$ is the model-side heuristic covariance.

### Main limitations

This formulation had several conceptual weaknesses.

First, it did not treat the GMM as a true mixture. In a probabilistically consistent mixture model, the sum over components should be formed before taking the logarithm.

Second, each model point was penalized by every data component individually, including components that were spatially irrelevant.

Third, the resulting score was asymmetric in a weak and only partially interpretable way. It resembled a heuristic overlap sum more than a clearly defined likelihood.

For these reasons, the earlier GMM score was difficult to justify rigorously in mathematical terms.

## 3. Current Formulation

The current implementation instead behaves like:

```text
S_current = sum_j log( sum_k pi_k * N(x_M,j | mu_k, Sigma_k + Sigma_M) )
```

This change is substantial. The inner sum is now taken over the GMM components before applying the logarithm, so the data-side GMM is treated as an actual mixture.

### Interpretation

The present GMM score is best interpreted as:

> the total log-likelihood of the model point cloud under a Gaussian mixture fitted to the experimental localization data

This is a coherent and useful probabilistic interpretation. It is no longer merely a pairwise heuristic.

### Advantages over the earlier formulation

Relative to the earlier version, the current implementation improves:

- **mathematical consistency**, because the GMM is now treated as a mixture;
- **numerical robustness**, because the calculation is performed in a log-sum-exp style;
- **scientific interpretability**, because the score now corresponds to a clear point-cloud likelihood statement.

## 4. What the Current Implementation Still Does Not Do

Although the present formulation is much improved, it remains asymmetric in an important sense.

At present:

- the **data** are represented as a Gaussian mixture,
- the **model** is represented as a set of points with a heuristic covariance term.

Thus the current formulation corresponds to:

```text
model point cloud under data density
```

not yet to:

```text
model density against data density
```

This distinction is central. A full density-overlap model would represent both sides probabilistically and compare them on equal footing.

## 5. Fuller Density-Overlap Model

A more complete formulation would define both the experimental data and the model as Gaussian mixtures.

### Data density

```text
rho_D(x) = sum_k pi_k * N(x | mu_D,k, Sigma_D,k)
```

### Model density

```text
rho_M(x) = sum_j omega_j * N(x | mu_M,j, Sigma_M,j)
```

where:

- $\omega_j$ are model-side weights,
- $\Sigma_M,j$ are model-side covariance estimates, ideally derived from the actual AV geometry rather than from one global heuristic variance.

### Symmetric overlap quantity

A natural overlap measure is:

```text
O = integral rho_M(x) * rho_D(x) dx
```

For Gaussian components, this overlap can be evaluated analytically as:

```text
O = sum_j sum_k omega_j * pi_k * N(mu_M,j | mu_D,k, Sigma_M,j + Sigma_D,k)
```

Such a formulation would constitute a true density-vs-density comparison.

Possible score definitions based on this overlap include:

- the raw overlap $O$,
- the logarithm $\log O$,
- or a related symmetric cross-entropy or divergence-style measure.

## 6. Staged Development Path

It is useful to think of future development in stages rather than as a single transition.

### Stage A: Current implementation

The current local implementation already provides a mathematically defensible point-cloud likelihood:

```text
S_current = sum_j log( sum_k pi_k * N(x_M,j | mu_k, Sigma_k + Sigma_M) )
```

This stage should be regarded as complete in the present code base.

### Stage B: Improved model uncertainty

The next realistic improvement would be to retain the present likelihood structure, but replace the fixed scalar model variance with AV-derived covariance estimates.

This would improve physical realism without requiring a complete redesign of the score.

### Stage C: Symmetric density-overlap

The subsequent step would be to represent the model explicitly as a Gaussian mixture and compute an analytical overlap between the model density and the data density.

This would be the natural mathematical target for a true density-overlap formulation.

### Stage D: Full posterior formulation

Beyond the score itself, a fully Bayesian treatment would add priors, nuisance parameters, or learned uncertainty scaling and integrate the score into a broader posterior framework.

This would move the project from scoring-function refinement toward a more comprehensive modeling methodology.

## 7. Difficulty Assessment

The different stages above vary considerably in implementation complexity.

- **Stage A** is already realized locally.
- **Stage B** is of moderate difficulty.
- **Stage C** is moderate to hard.
- **Stage D** is clearly hard and methodologically broader.

The principal reason the symmetric overlap model is not trivial is that it requires explicit choices about:

- how model components should be weighted,
- how model covariances should be estimated from AVs,
- what objective is to be optimized or compared,
- and whether analytical gradients are needed for the intended optimization workflow.

## 8. Recommended Wording for Documentation or Thesis Text

For the current implementation, the following wording is the safest and most accurate:

> The current GMM score evaluates the log-likelihood of the model point cloud under a Gaussian mixture fitted to the experimental data. It is therefore an asymmetric point-cloud likelihood model rather than a full symmetric density-overlap formulation.

For a future extension, the following formulation is appropriate:

> A future extension would represent both model and data as Gaussian mixtures and compute their analytical overlap, thereby yielding a symmetric density-vs-density score.

## 9. Recommended Immediate Next Step

The most sensible next step is not a complete rewrite of the GMM framework. Instead, the most productive improvement would be:

1. to retain the current mixture-likelihood structure,
2. to replace the fixed heuristic model variance with AV-derived covariance estimates,
3. to add focused tests that evaluate robustness to irrelevant distant components and sensitivity to structural alignment.

This would strengthen the physical interpretation of the current GMM score while keeping implementation risk and conceptual complexity manageable.
