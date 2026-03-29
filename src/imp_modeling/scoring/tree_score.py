import typing

import IMP
import IMP.bff
import IMP.core
import numpy as np
from sklearn.neighbors import KDTree

TREE_EXACT_MODEL_COUNT_THRESHOLD = 64


def _extract_scalar_variance(var_entry):
    """
    Extract a scalar variance from scalar/vector/matrix representations.
    """
    if isinstance(var_entry, (float, int, np.floating, np.integer)):
        return max(float(var_entry), 1e-12)

    arr = np.asarray(var_entry, dtype=np.float64)
    if arr.ndim == 0:
        return max(float(arr), 1e-12)
    if arr.ndim == 1:
        return max(float(np.mean(arr)), 1e-12)
    if arr.ndim == 2:
        return max(float(np.mean(np.diag(arr))), 1e-12)
    raise ValueError(f"Unsupported variance shape: {arr.shape}")


def _extract_covariance_matrix(var_entry, ndim):
    """
    Convert one variance/covariance entry to an (ndim, ndim) covariance matrix.
    """
    if var_entry is None:
        return np.eye(ndim, dtype=np.float64)

    if isinstance(var_entry, (float, int, np.floating, np.integer)):
        scalar = max(float(var_entry), 1e-12)
        return np.eye(ndim, dtype=np.float64) * scalar

    arr = np.asarray(var_entry, dtype=np.float64)
    if arr.ndim == 0:
        scalar = max(float(arr), 1e-12)
        return np.eye(ndim, dtype=np.float64) * scalar
    if arr.ndim == 1:
        if len(arr) == ndim:
            return np.diag(np.maximum(arr, 1e-12))
        scalar = max(float(np.mean(arr)), 1e-12)
        return np.eye(ndim, dtype=np.float64) * scalar
    if arr.ndim == 2:
        if arr.shape == (ndim, ndim):
            return arr
        scalar = max(float(np.mean(np.diag(arr))), 1e-12)
        return np.eye(ndim, dtype=np.float64) * scalar
    raise ValueError(f"Unsupported variance shape: {arr.shape}")


def _extract_model_coordinates(modelavs, scaling, offsetxyz, model_coords_override):
    if model_coords_override is not None:
        modelxyzs = np.array(model_coords_override, dtype=np.float64)
    else:
        modelxyzs = []
        for av in modelavs:
            modelpxyz = IMP.core.XYZ(av)
            modelxyz = np.array(modelpxyz.get_coordinates(), dtype=np.float64)
            modelxyzs.append(modelxyz)
        modelxyzs = np.array(modelxyzs, dtype=np.float64)
        modelxyzs = modelxyzs * scaling
        if offsetxyz is not None:
            modelxyzs = modelxyzs + offsetxyz
    return modelxyzs


def _build_model_candidates_per_data(tree, dataxyz, modelxyzs_query, searchradius):
    """
    Build, for each data index, the list of model indices within search radius.
    """
    n_model = len(modelxyzs_query)
    # For small model sets, use all models per data point to keep tree scoring
    # mathematically identical to distance scoring while remaining fast.
    if n_model <= TREE_EXACT_MODEL_COUNT_THRESHOLD:
        all_models = list(range(n_model))
        return [all_models.copy() for _ in range(len(dataxyz))]

    if tree is None:
        tree = KDTree(dataxyz)
    effective_radius = np.inf if searchradius is None else float(searchradius)
    data_neighbors_by_model = tree.query_radius(modelxyzs_query, effective_radius)

    model_candidates_per_data = [[] for _ in range(len(dataxyz))]
    for model_idx, data_indices in enumerate(data_neighbors_by_model):
        for data_idx in data_indices:
            model_candidates_per_data[int(data_idx)].append(model_idx)
    return model_candidates_per_data


def _prepare_distance_terms(dataxyz, var, sigmaav):
    """
    Precompute per-data-point normalization and inverse covariance terms.
    """
    ndim = dataxyz.shape[1]
    n_data = len(dataxyz)
    log_prefactors = np.zeros(n_data, dtype=np.float64)
    inv_sigmas = np.zeros((n_data, ndim, ndim), dtype=np.float64)

    eye = np.eye(ndim, dtype=np.float64)
    base = ndim * np.log(2.0 * np.pi)

    for i in range(n_data):
        cov_d = (
            _extract_covariance_matrix(var[i], ndim)
            if var is not None
            else eye
        )
        sigma = cov_d + eye * sigmaav

        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise ValueError("Combined covariance matrix is not positive definite.")

        inv_sigmas[i] = np.linalg.inv(sigma)
        log_prefactors[i] = -0.5 * (base + logdet)

    return log_prefactors, inv_sigmas


def computescoretree(
    tree: KDTree,
    modelavs: typing.List[IMP.bff.AV],
    dataxyz: np.ndarray,
    var: np.ndarray,
    scaling: float = 0.1,
    searchradius: float = 10,
    offsetxyz: np.ndarray = None,
    model_coords_override: np.ndarray = None,
):
    """
    Tree-backed variant of the Distance score.

    Mathematical interpretation is the same as Distance scoring:
    for each data point, accumulate a log-sum-exp over candidate model points.
    The KD-tree is used only to select candidate model-data pairs.

    For small model sets (<= TREE_EXACT_MODEL_COUNT_THRESHOLD) we evaluate all
    model points per data point. This keeps the score numerically aligned with
    Distance scoring while still allowing KD-tree pruning for larger systems.

    A key safety behavior: if no model point is inside the search radius for a
    data point, we fall back to all model points for that data point. This
    preserves a proper likelihood penalty and prevents the old "zero-neighbor
    escape" behavior.
    """
    dataxyz = np.asarray(dataxyz, dtype=np.float64)
    if dataxyz.ndim != 2:
        raise ValueError("dataxyz must be a 2D array of shape (N, D).")
    if len(dataxyz) == 0:
        return 0.0

    modelxyzs = _extract_model_coordinates(
        modelavs, scaling, offsetxyz, model_coords_override
    )
    modelxyzs = np.asarray(modelxyzs, dtype=np.float64)
    if modelxyzs.ndim != 2 or len(modelxyzs) == 0:
        return -np.inf

    ndim = dataxyz.shape[1]
    if modelxyzs.shape[1] < ndim:
        raise ValueError(
            f"Model coordinates have dimension {modelxyzs.shape[1]}, but data has {ndim}."
        )
    modelxyzs_query = modelxyzs[:, :ndim]

    sigmaav = 8.0
    log_prefactors, inv_sigmas = _prepare_distance_terms(dataxyz, var, sigmaav)
    model_candidates_per_data = _build_model_candidates_per_data(
        tree, dataxyz, modelxyzs_query, searchradius
    )
    all_model_indices = np.arange(len(modelxyzs_query), dtype=np.int64)

    scoretotal = 0.0
    for data_idx in range(len(dataxyz)):
        candidate_models = model_candidates_per_data[data_idx]
        if len(candidate_models) == 0:
            candidate_models = all_model_indices

        x_d = dataxyz[data_idx]
        inv_sigma = inv_sigmas[data_idx]
        log_prefactor = log_prefactors[data_idx]

        log_probs = np.zeros(len(candidate_models), dtype=np.float64)
        for local_idx, model_idx in enumerate(candidate_models):
            diff = modelxyzs_query[int(model_idx)] - x_d
            exponent = -0.5 * np.dot(diff.T, np.dot(inv_sigma, diff))
            log_probs[local_idx] = log_prefactor + exponent

        max_log_prob = np.max(log_probs)
        scoretotal += max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

    return float(scoretotal)


def computescoretree_with_grad(
    tree: KDTree,
    modelavs,
    dataxyz: np.ndarray,
    var: np.ndarray,
    scaling: float = 0.1,
    searchradius: float = 10,
    offsetxyz: np.ndarray = None,
    model_coords_override: np.ndarray = None,
):
    """
    Tree-backed Distance-equivalent score plus gradient wrt model coordinates.
    """
    dataxyz = np.asarray(dataxyz, dtype=np.float64)
    if dataxyz.ndim != 2:
        raise ValueError("dataxyz must be a 2D array of shape (N, D).")
    if len(dataxyz) == 0:
        return 0.0, np.zeros((0, 3), dtype=np.float64)

    modelxyzs = _extract_model_coordinates(
        modelavs, scaling, offsetxyz, model_coords_override
    )
    modelxyzs = np.asarray(modelxyzs, dtype=np.float64)
    if modelxyzs.ndim != 2 or len(modelxyzs) == 0:
        return -np.inf, np.zeros((0, 3), dtype=np.float64)

    ndim = dataxyz.shape[1]
    if modelxyzs.shape[1] < ndim:
        raise ValueError(
            f"Model coordinates have dimension {modelxyzs.shape[1]}, but data has {ndim}."
        )
    modelxyzs_query = modelxyzs[:, :ndim]

    sigmaav = 8.0
    log_prefactors, inv_sigmas = _prepare_distance_terms(dataxyz, var, sigmaav)
    model_candidates_per_data = _build_model_candidates_per_data(
        tree, dataxyz, modelxyzs_query, searchradius
    )
    all_model_indices = np.arange(len(modelxyzs_query), dtype=np.int64)

    scoretotal = 0.0
    grad = np.zeros((len(modelxyzs), 3), dtype=np.float64)

    for data_idx in range(len(dataxyz)):
        candidate_models = model_candidates_per_data[data_idx]
        if len(candidate_models) == 0:
            candidate_models = all_model_indices

        x_d = dataxyz[data_idx]
        inv_sigma = inv_sigmas[data_idx]
        log_prefactor = log_prefactors[data_idx]

        n_candidates = len(candidate_models)
        log_probs = np.zeros(n_candidates, dtype=np.float64)
        forces = np.zeros((n_candidates, ndim), dtype=np.float64)

        for local_idx, model_idx in enumerate(candidate_models):
            model_idx = int(model_idx)
            diff = modelxyzs_query[model_idx] - x_d
            exponent = -0.5 * np.dot(diff.T, np.dot(inv_sigma, diff))
            log_probs[local_idx] = log_prefactor + exponent
            forces[local_idx] = -np.dot(inv_sigma, diff)

        max_log_prob = np.max(log_probs)
        exp_shifted = np.exp(log_probs - max_log_prob)
        sum_exp = np.sum(exp_shifted)
        scoretotal += max_log_prob + np.log(sum_exp)

        responsibilities = exp_shifted / sum_exp
        for local_idx, model_idx in enumerate(candidate_models):
            model_idx = int(model_idx)
            grad[model_idx, :ndim] += responsibilities[local_idx] * forces[local_idx]

    return float(scoretotal), grad
