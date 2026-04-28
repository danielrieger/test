# Fix GMM Mixture Likelihood (Minimal Cleanup)

This plan details the implementation of the "Minimal mathematical cleanup" strategy for the GMM scoring engine. This addresses the issues identified where the current score is an asymmetric sum of pairwise log-overlaps rather than a true probabilistic mixture comparison.

## Goal Description
The objective is to change the core GMM score evaluation from:
`S_current = sum_j sum_k log( pi_k * N(x_j | mu_k, Sigma_k + Sigma_M) )`
*(Currently implemented as an outer loop over data components `k` and inner loop over model points `j`, summing independent log-likelihoods)*

To a proper mixture likelihood:
`S_new = sum_j log( sum_k pi_k * N(x_j | mu_k, Sigma_k + Sigma_M) )`
*(This will require inverting the loop logic: outer loop over model points `j`, inner loop over data components `k` using a log-sum-exp reduction for numerical stability)*

This change ensures that each model point is explained by the data mixture as a whole, preventing distant data components from excessively penalizing the score.

## User Review Required
> [!IMPORTANT]
> **Performance Implications**
> Switching the CUDA kernel thread mapping from `(thread per data component)` to `(thread per model point)` should remain efficient, but we will be computing determinants and matrix inverses inside deeper or slightly differently structured loops. We will attempt to pre-calculate the data covariance inverses and determinants if memory allows, or compute them on-the-fly depending on Numba/CUDA constraints. 

## Proposed Changes

### `src/smlm_score/imp_modeling/scoring/gmm_score.py`
#### [MODIFY] `_compute_nb_gmm_cpu`
- **Current state**: Outer loop over `data_mean` (k), inner loop over `model_xyzs` (j).
- **New state**: 
  - Precompute the `log_prefactor` and `inv_Sigma` matrices for all `k` data components (since they only depend on data covariance + constant `sigma_M`).
  - Outer loop over `model_xyzs` (j).
  - Inner loop over `data_mean` (k) computing the log-probability of the model point under component `k`.
  - Perform log-sum-exp over the `k` components.
  - Accumulate into the total log score.

### `src/smlm_score/imp_modeling/scoring/cuda_kernels.py`
#### [MODIFY] `_gmm_score_kernel`
- **Current state**: Each thread handles one data component (`i = cuda.grid(1) < n_data`). Loop over model points. Writes `n_data` outputs.
- **New state**:
  - Each thread will handle **one model point** (`m = cuda.grid(1) < n_models`).
  - The thread will loop over all `n_data` GMM components.
  - Inside the thread, compute log-probability for each component, tracking the maximum log-probability to perform a stable log-sum-exp.
  - Write the final scalar log-sum for that model point to an output array of size `n_models`.
#### [MODIFY] `compute_nb_gmm_gpu` (Host wrapper)
- Adjust the launch configuration to spawn `n_models` threads instead of `n_data` threads.
- Change the `d_output` device array size to `n_models`.
- Update the final summation `np.sum(output)` which will now be summing over models.

### `docs/scoring_models.md`
#### [MODIFY] `scoring_models.md`
- Update the mathematical notation in the GMM scoring section to reflect the new mixture likelihood equation.
- Remove the caveats that currently state the function evaluates an inaccurate pairwise asymmetry.

### `tests/test_scoring.py` (or new test file)
#### [NEW] Unit tests for Mixture Likelihood
- Add a specific test case that demonstrates the score's resistance to distant outlier components. (e.g., scoring a model against a tight main GMM, and then scoring against the same GMM with a very distant, low-weight second component added. The new score should barely change, whereas the old score would drop significantly due to the forced sum).

## Open Questions
- Is there any risk that the number of model points (`n_models`) could exceed the CUDA max grid dimensions in extreme cases? (Generally no, as points rarely exceed an order of $10^5$, which easily fits in a 1D grid, but good to be mindful).

## Verification Plan

### Automated Tests
- Run `pytest` to ensure all existing validation logic (separation constraints, hold-outs) still pass or improve with the new scoring landscape.
- Run the requested new unit tests specifically probing the mixture independence.

### Manual Verification
- Run `benchmark_scoring.py` to ensure the $O(GK)$ asymptotic scaling behavior remains constant and fast.
- Run `visualize_gmm_selection.py` to ensure the overall modeling pipeline continues to rank clean clusters highly.
