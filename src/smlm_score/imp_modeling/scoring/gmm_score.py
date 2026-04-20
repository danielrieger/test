import numpy as np
import sys
import tqdm
from sklearn import mixture
from numba import jit
import typing
import IMP.core

# CUDA acceleration (graceful fallback)
from smlm_score.imp_modeling.scoring.cuda_kernels import HAS_CUDA
if HAS_CUDA:
    from smlm_score.imp_modeling.scoring.cuda_kernels import (
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
        model_variance: float = 8.0,
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
    model_variance : float
        Spatial variance applied to model points.
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
        model_variance,
    )

    return score_total


# Note: Numba fastmath allows for CPU optimizations
# OFFSET RECOMMENDATION: Apply offset in the caller (e.g. compute_score_GMM or via PCA
# alignment), not inside this Numba function. The offset_xyz parameter is kept for
# backward compatibility but the preferred workflow is pre-aligned data.
@jit(nopython=True, fastmath=True)
def _compute_nb_gmm_and_grad_cpu(
        model_xyzs,
        data_mean,
        data_cov,
        data_weight,
        model_variance,
        offset_xyz=None
):
    """Numba JIT CPU kernel for Point-Cloud Likelihood (ISD/Habeck 2017).

    Computes the log-sum-exp of mixture probabilities over data components for each model point,
    as well as the analytical gradient with respect to model point coordinates.

    Returns
    -------
    tuple
        (float score_total, np.ndarray grad of shape (M, 3))
    """
    score_total = 0.0

    if offset_xyz is not None:
        model_xyzs = model_xyzs + offset_xyz

    sigma_M = np.eye(3, dtype=np.float64) * model_variance
    weight_M = 1.0

    n_data = len(data_mean)
    n_models = len(model_xyzs)
    
    grad = np.zeros((n_models, 3), dtype=np.float64)

    # Precompute inverses and prefactors
    inv_Sigmas = np.zeros((n_data, 3, 3), dtype=np.float64)
    log_prefactors = np.zeros(n_data, dtype=np.float64)

    for i in range(n_data):
        Sigma = data_cov[i] + sigma_M
        det_S = np.linalg.det(Sigma)
        inv_Sigmas[i] = np.linalg.inv(Sigma)
        prefactor = (weight_M * data_weight[i]) / (
            (2. * np.pi) ** 1.5 * np.abs(det_S) ** 0.5
        )
        if prefactor > 0:
            log_prefactors[i] = np.log(prefactor)
        else:
            log_prefactors[i] = -np.inf

    # Independent Mixture Loop: 
    for m_idx in range(n_models):
        mx = model_xyzs[m_idx, 0]
        my = model_xyzs[m_idx, 1]
        mz = model_xyzs[m_idx, 2]

        max_lp = -np.inf
        lps = np.zeros(n_data, dtype=np.float64)

        for i in range(n_data):
            dx = mx - data_mean[i, 0]
            dy = my - data_mean[i, 1]
            dz = mz - data_mean[i, 2]

            t0 = inv_Sigmas[i, 0, 0] * dx + inv_Sigmas[i, 0, 1] * dy + inv_Sigmas[i, 0, 2] * dz
            t1 = inv_Sigmas[i, 1, 0] * dx + inv_Sigmas[i, 1, 1] * dy + inv_Sigmas[i, 1, 2] * dz
            t2 = inv_Sigmas[i, 2, 0] * dx + inv_Sigmas[i, 2, 1] * dy + inv_Sigmas[i, 2, 2] * dz

            exponent = -0.5 * (dx * t0 + dy * t1 + dz * t2)
            lp = log_prefactors[i] + exponent
            lps[i] = lp
            if lp > max_lp:
                max_lp = lp

        sum_exp = 0.0
        for i in range(n_data):
            diff_lp = lps[i] - max_lp
            if diff_lp > -80.0:
                sum_exp += np.exp(diff_lp)

        if sum_exp > 0:
            log_sum = max_lp + np.log(sum_exp)
            score_total += log_sum
            
            # Compute Gradient: sum_k gamma_k * -Sigma_k^{-1} (x_j - mu_k)
            for i in range(n_data):
                diff_lp = lps[i] - log_sum
                if diff_lp > -80.0:
                    gamma_k = np.exp(diff_lp)
                    dx = mx - data_mean[i, 0]
                    dy = my - data_mean[i, 1]
                    dz = mz - data_mean[i, 2]

                    t0 = inv_Sigmas[i, 0, 0] * dx + inv_Sigmas[i, 0, 1] * dy + inv_Sigmas[i, 0, 2] * dz
                    t1 = inv_Sigmas[i, 1, 0] * dx + inv_Sigmas[i, 1, 1] * dy + inv_Sigmas[i, 1, 2] * dz
                    t2 = inv_Sigmas[i, 2, 0] * dx + inv_Sigmas[i, 2, 1] * dy + inv_Sigmas[i, 2, 2] * dz

                    grad[m_idx, 0] += gamma_k * (-t0)
                    grad[m_idx, 1] += gamma_k * (-t1)
                    grad[m_idx, 2] += gamma_k * (-t2)
        else:
            score_total += max_lp
            # Gradient fallback if extreme log probability
            for i in range(n_data):
                if lps[i] == max_lp:
                    dx = mx - data_mean[i, 0]
                    dy = my - data_mean[i, 1]
                    dz = mz - data_mean[i, 2]
                    t0 = inv_Sigmas[i, 0, 0] * dx + inv_Sigmas[i, 0, 1] * dy + inv_Sigmas[i, 0, 2] * dz
                    t1 = inv_Sigmas[i, 1, 0] * dx + inv_Sigmas[i, 1, 1] * dy + inv_Sigmas[i, 1, 2] * dz
                    t2 = inv_Sigmas[i, 2, 0] * dx + inv_Sigmas[i, 2, 1] * dy + inv_Sigmas[i, 2, 2] * dz
                    grad[m_idx, 0] += -t0
                    grad[m_idx, 1] += -t1
                    grad[m_idx, 2] += -t2
                    break 

    return score_total, grad


def compute_nb_gmm_with_grad(model_xyzs, data_mean, data_cov, data_weight, model_variance=8.0, offset_xyz=None):
    """
    Dispatcher: uses GPU kernel (score only) when CUDA is available and data is large enough,
    otherwise falls back to CPU Numba JIT (score and gradient).
    NOTE: GPU gradient computation is not yet supported. If gradient is required, CPU is used.
    """
    return _compute_nb_gmm_and_grad_cpu(model_xyzs, data_mean, data_cov, data_weight, model_variance, offset_xyz=offset_xyz)

def compute_nb_gmm(model_xyzs, data_mean, data_cov, data_weight, model_variance=8.0, offset_xyz=None):
    if HAS_CUDA and len(data_mean) >= CUDA_MIN_DATA_SIZE:
        return compute_nb_gmm_gpu(model_xyzs, data_mean, data_cov, data_weight, model_variance, offset_xyz=offset_xyz)
    score, _ = _compute_nb_gmm_and_grad_cpu(model_xyzs, data_mean, data_cov, data_weight, model_variance, offset_xyz=offset_xyz)
    return score
