import numpy as np
import sys
import tqdm
from sklearn import mixture
from numba import jit
import typing
import IMP.core

# CUDA acceleration (graceful fallback)
from smlm_score.src.imp_modeling.scoring.cuda_kernels import HAS_CUDA
if HAS_CUDA:
    from smlm_score.src.imp_modeling.scoring.cuda_kernels import (
        compute_nb_gmm_gpu, CUDA_MIN_DATA_SIZE
    )
else:
    def compute_nb_gmm_gpu(*args, **kwargs):
        raise RuntimeError("CUDA is not available. This function should not be called.")
    CUDA_MIN_DATA_SIZE = 256


def test_gmm_components(
        data: np.ndarray,
        component_min: int = 1,
        component_max: int = 1024,
        show_progress: bool = False,
        reg_covar: float = 1e-6,
) -> dict:
    """Search for BIC-optimal number of GMM components over a log-spaced grid.

    Parameters
    ----------
    data : np.ndarray, shape (N, D)
        Point cloud to fit.
    component_min : int
        Minimum number of components (inclusive).
    component_max : int
        Maximum number of components (capped at N).
    show_progress : bool
        Show tqdm progress bar (only if stdout is a TTY).
    reg_covar : float
        Regularization added to covariance diagonal.

    Returns
    -------
    d : dict
        Dictionary with keys 'gmm', 'score', 'aic', 'bic', 'n_components', 'n'.
    gmm_sel : GaussianMixture
        The BIC-optimal fitted model.
    gmm_sel_mean : np.ndarray
        Means of the selected model, shape (K, D).
    gmm_sel_cov : np.ndarray
        Covariances of the selected model, shape (K, D, D).
    gmm_sel_weight : np.ndarray
        Weights of the selected model, shape (K,).
    """
    scores = list()
    aics = list()
    bics = list()
    gmms = list()

    # Limit max components to the number of samples available
    n_samples = data.shape[0]
    component_max = min(component_max, n_samples)

    # Make log spaced ints
    n_components = list()
    i = max(1, component_min)
    while i <= component_max:
        n_components.append(i)
        i *= 2

    show_progress = show_progress and sys.stdout.isatty()
    for n_component in tqdm.tqdm(n_components, disable=not show_progress):
        clf = mixture.GaussianMixture(n_component, covariance_type="full", reg_covar=reg_covar)
        clf.fit(data)
        scores.append(clf.score(data))
        aics.append(clf.aic(data))
        bics.append(clf.bic(data))
        gmms.append(clf)

    n = np.argmin(bics)

    d = {
        'gmm': gmms,
        'score': scores,
        'aic': aics,
        'bic': bics,
        'n_components': n_components,
        'n': n
    }

    gmm_sel = d['gmm'][d['n']]
    gmm_sel_mean = gmm_sel.means_
    gmm_sel_cov = gmm_sel.covariances_
    gmm_sel_weight = gmm_sel.weights_

    return d, gmm_sel, gmm_sel_mean, gmm_sel_cov, gmm_sel_weight


def compute_score_GMM(
        model_avs: typing.List[IMP.bff.AV],
        data_mean: np.ndarray,
        data_cov: np.ndarray = None,
        data_weight: np.ndarray = None,
        offset_xyz: np.ndarray = None
):
    """Extract AV coordinates and evaluate GMM log-likelihood.

    Parameters
    ----------
    model_avs : list of IMP.bff.AV
        Accessible Volume decorators from the IMP model.
    data_mean : np.ndarray, shape (K, 3)
        GMM component means.
    data_cov : np.ndarray, shape (K, 3, 3)
        GMM component covariances.
    data_weight : np.ndarray, shape (K,)
        GMM component weights.
    offset_xyz : np.ndarray or None
        Optional translation offset applied to model coordinates.

    Returns
    -------
    float
        Total GMM log-likelihood score.
    """
    # Model loop
    model_xyzs = list()
    for av in model_avs:
        model_p_xyz = IMP.core.XYZ(av)
        model_xyz = np.array(model_p_xyz.get_coordinates(), dtype=np.float64)
        model_xyzs.append(model_xyz)
    model_xyzs = np.array(model_xyzs)

    # Offset when out of area
    if offset_xyz is not None:
        model_xyzs += offset_xyz

    score_total = compute_nb_gmm(
        model_xyzs,
        data_mean,
        data_cov,
        data_weight,
    )

    return score_total


# Note: Numba fastmath allows for CPU optimizations
# OFFSET RECOMMENDATION: Apply offset in the caller (e.g. compute_score_GMM or via PCA
# alignment), not inside this Numba function. The offset_xyz parameter is kept for
# backward compatibility but the preferred workflow is pre-aligned data.
@jit(nopython=True, fastmath=True)
def _compute_nb_gmm_cpu(
        model_xyzs,
        data_mean,
        data_cov,
        data_weight,
        offset_xyz=None
):
    """Numba JIT CPU kernel for GMM log-likelihood (Bonomi et al. 2019).

    Computes the sum of log-probabilities over all model-data pairs using
    the combined covariance Sigma = Sigma_data + Sigma_model.

    Parameters
    ----------
    model_xyzs : np.ndarray, shape (M, 3)
        Model coordinates.
    data_mean : np.ndarray, shape (K, 3)
        GMM component means.
    data_cov : np.ndarray, shape (K, 3, 3)
        GMM component covariances.
    data_weight : np.ndarray, shape (K,)
        GMM component weights.
    offset_xyz : np.ndarray or None
        Optional offset applied to model coordinates.

    Returns
    -------
    float
        Total log-likelihood score.
    """
    score_total = 0.0

    # Offset when out of area
    if offset_xyz is not None:
        model_xyzs = model_xyzs + offset_xyz  # Use + instead of += to avoid mutating input

    # placeholder for AV covariance (instantiated outside loop for numba efficiency)
    sigma_av = 8.0
    sigma_M = np.eye(3, dtype=np.float64) * sigma_av
    weight_M = 1.0

    # Data loop (Outer loop to match data-driven GMM components)
    for i in range(len(data_mean)):
        mean_D = data_mean[i]
        sigma_D = data_cov[i]
        weight_D = data_weight[i]

        Sigma = sigma_D + sigma_M  # Bonomi et al. 2019, Bayesian EM eq. 12

        # Precompute values that are constant for this GMM component
        prefactor = (weight_M * weight_D) / ((2. * np.pi) ** (3. / 2.) * np.linalg.det(Sigma) ** 0.5)
        inv_Sigma = np.linalg.inv(Sigma)
        log_prefactor = np.log(prefactor)

        # Model loop
        for m_idx in range(len(model_xyzs)):
            model_xyz = model_xyzs[m_idx]

            diff = model_xyz - mean_D
            # Explicit dot product for numba compatibility over '@'
            exponent = -0.5 * np.dot(diff.T, np.dot(inv_Sigma, diff))
            score_total += log_prefactor + exponent

    return score_total


def compute_nb_gmm(model_xyzs, data_mean, data_cov, data_weight, offset_xyz=None):
    """
    Dispatcher: uses GPU kernel when CUDA is available and data is large enough,
    otherwise falls back to CPU Numba JIT.
    """
    if HAS_CUDA and len(data_mean) >= CUDA_MIN_DATA_SIZE:
        return compute_nb_gmm_gpu(model_xyzs, data_mean, data_cov, data_weight,
                                  offset_xyz=offset_xyz)
    return _compute_nb_gmm_cpu(model_xyzs, data_mean, data_cov, data_weight,
                               offset_xyz=offset_xyz)
