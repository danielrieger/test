# Contents of smlm_score/src/imp_modeling/scoring/distance_score.py
import numpy as np
import IMP.core
import typing
import IMP.bff
from numba import jit

# CUDA acceleration (graceful fallback)
from smlm_score.imp_modeling.scoring.cuda_kernels import HAS_CUDA
if HAS_CUDA:
    from smlm_score.imp_modeling.scoring.cuda_kernels import (
        compute_distance_score_gpu, CUDA_MIN_DATA_SIZE
    )


@jit(nopython=True, fastmath=True)
def _compute_distance_score_cpu(datamean, datacov, weights, modelxyzs, sigmaav):
    """Numba JIT CPU kernel for pairwise distance log-likelihood.

    For each data point, computes log-sum-exp over all model points using
    combined covariance Sigma = Sigma_data + Sigma_model (Bonomi et al. 2019).

    Parameters
    ----------
    datamean : np.ndarray, shape (N, 3)
        Data point coordinates.
    datacov : np.ndarray, shape (N, 3, 3)
        Per-data-point covariance matrices.
    weights : np.ndarray, shape (N,)
        Per-data-point weights.
    modelxyzs : np.ndarray, shape (M, 3)
        Model coordinates (already scaled and offset).
    sigmaav : float
        Isotropic AV variance (added to diagonal of combined covariance).

    Returns
    -------
    float
        Total log-likelihood score.
    """
    scoretotal = 0.0

    # Loop over data
    for i in range(len(datamean)):
        meanD = datamean[i]
        sigmaD = datacov[i]
        weightD = weights[i]

        SigmaD = sigmaD

        # We will compute the log probabilities for each model point for this data point
        m_log_probs = np.zeros(len(modelxyzs), dtype=np.float64)

        #Loop over each of model
        for m_idx in range(len(modelxyzs)):
            modelxyz = modelxyzs[m_idx]

            SigmaM = np.eye(3, dtype=np.float64) * sigmaav
            Sigma = SigmaD + SigmaM

            # Recalculate prefactor using the combined covariance Sigma
            # Bonomi et al. 2019, Bayesian EM eq. 12
            prefactor = weightD / ((2. * np.pi)**1.5 * np.linalg.det(Sigma)**0.5)

            xD = meanD
            xM = modelxyz

            # @ matrix multiplication in numpy (numba supports this)
            diff = xM - xD
            inv_Sigma = np.linalg.inv(Sigma)
            exponent = -0.5 * np.dot(diff.T, np.dot(inv_Sigma, diff))

            m_log_probs[m_idx] = np.log(prefactor) + exponent

        # Log-Sum-Exp over the model points for this specific data point
        max_log_prob = np.max(m_log_probs)
        sum_exp = 0.0
        for m_idx in range(len(modelxyzs)):
            sum_exp += np.exp(m_log_probs[m_idx] - max_log_prob)

        # Add the log of the sum to the total log-likelihood
        scoretotal += max_log_prob + np.log(sum_exp)

    return scoretotal


@jit(nopython=True, fastmath=True)
def _compute_distance_score_and_grad_cpu(datamean, datacov, weights, modelxyzs, sigmaav):
    """Compute distance log-likelihood AND gradient w.r.t. model coordinates.

    The gradient for model point m is the derivative of:
        L = sum_i log( sum_j w_i * N(x_m_j | x_d_i, Sigma) )
    w.r.t. x_m, giving:
        dL/dx_m = sum_i [ softmax_weight_i * Sigma^{-1} (x_d_i - x_m) ]
    where softmax_weight_i = p_i / sum_j p_j (the responsibility of data point i).

    Parameters
    ----------
    datamean : np.ndarray, shape (N, 3)
        Data point coordinates.
    datacov : np.ndarray, shape (N, 3, 3)
        Per-data-point covariance matrices.
    weights : np.ndarray, shape (N,)
        Per-data-point weights.
    modelxyzs : np.ndarray, shape (M, 3)
        Model coordinates.
    sigmaav : float
        Isotropic AV variance.

    Returns
    -------
    scoretotal : float
        Total log-likelihood (same as _compute_distance_score_cpu).
    grad : np.ndarray, shape (M, 3)
        Gradient of the log-likelihood w.r.t. each model point coordinate.
        Points in the direction of INCREASING likelihood (ascent direction).
    """
    n_model = len(modelxyzs)
    grad = np.zeros((n_model, 3), dtype=np.float64)
    scoretotal = 0.0

    for i in range(len(datamean)):
        meanD = datamean[i]
        sigmaD = datacov[i]
        weightD = weights[i]

        SigmaD = sigmaD
        m_log_probs = np.zeros(n_model, dtype=np.float64)

        # Store per-model-point intermediate quantities for gradient
        inv_Sigmas = np.zeros((n_model, 3, 3), dtype=np.float64)
        diffs = np.zeros((n_model, 3), dtype=np.float64)

        for m_idx in range(n_model):
            modelxyz = modelxyzs[m_idx]
            SigmaM = np.eye(3, dtype=np.float64) * sigmaav
            Sigma = SigmaD + SigmaM

            prefactor = weightD / ((2. * np.pi)**1.5 * np.linalg.det(Sigma)**0.5)

            diff = modelxyz - meanD
            inv_Sigma = np.linalg.inv(Sigma)
            exponent = -0.5 * np.dot(diff.T, np.dot(inv_Sigma, diff))

            m_log_probs[m_idx] = np.log(prefactor) + exponent
            inv_Sigmas[m_idx] = inv_Sigma
            diffs[m_idx] = diff

        # Log-Sum-Exp for score
        max_log_prob = np.max(m_log_probs)
        sum_exp = 0.0
        for m_idx in range(n_model):
            sum_exp += np.exp(m_log_probs[m_idx] - max_log_prob)

        scoretotal += max_log_prob + np.log(sum_exp)

        # Gradient: softmax-weighted pull
        for m_idx in range(n_model):
            responsibility = np.exp(m_log_probs[m_idx] - max_log_prob) / sum_exp
            force = -np.dot(inv_Sigmas[m_idx], diffs[m_idx])
            grad[m_idx] += responsibility * force

    return scoretotal, grad


def computescoresimple(modelavs: typing.List[IMP.bff.AV], datamean: np.ndarray,
                       datacov, weights=None, scaling: float = 0.1,
                       offsetxyz: np.ndarray = None,
                       model_coords_override: np.ndarray = None):
    """Compute pairwise distance log-likelihood score.

    Each data point contributes via an exponential term (Wu et al. 2023).
    Dispatches to GPU when CUDA is available and data exceeds threshold.

    Parameters
    ----------
    modelavs : list of IMP.bff.AV
        Accessible Volume decorators from the IMP model.
    datamean : np.ndarray, shape (N, 3)
        Data point coordinates.
    datacov : np.ndarray
        Per-data-point covariance matrices, shape (N, 3, 3).
    weights : np.ndarray or None
        Per-data-point weights. Defaults to uniform.
    scaling : float
        Scaling factor to convert model coordinates (Angstrom) to data units (nm).
        Ignored when model_coords_override is provided.
    offsetxyz : np.ndarray or None
        Optional offset applied to model coordinates (in data units, after scaling).
        Ignored when model_coords_override is provided.
    model_coords_override : np.ndarray or None
        Pre-aligned model coordinates already in data units (nm).
        When provided, bypasses IMP particle extraction, scaling, and offset.

    Returns
    -------
    float
        Total log-likelihood score.
    """
    if weights is None:
        weights = np.ones(len(datamean))

    datacov_arr = np.asarray(datacov, dtype=np.float64)

    sigmaav = 8.0

    if model_coords_override is not None:
        modelxyzs = np.array(model_coords_override, dtype=np.float64)
    else:
        modelxyzs = np.zeros((len(modelavs), 3), dtype=np.float64)
        for i, av in enumerate(modelavs):
            modelpxyz = IMP.core.XYZ(av)
            modelxyzs[i, :] = np.array(modelpxyz.get_coordinates(), dtype=np.float64)

        modelxyzs = modelxyzs * scaling

        if offsetxyz is not None:
            modelxyzs = modelxyzs + offsetxyz

    # Dispatch: GPU if CUDA available and data is large enough, else CPU
    if HAS_CUDA and len(datamean) >= CUDA_MIN_DATA_SIZE:
        scoretotal = compute_distance_score_gpu(datamean, datacov_arr, weights, modelxyzs, sigmaav)
    else:
        scoretotal = _compute_distance_score_cpu(datamean, datacov_arr, weights, modelxyzs, sigmaav)

    return scoretotal
