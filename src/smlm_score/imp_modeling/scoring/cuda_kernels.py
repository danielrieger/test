"""
CUDA-accelerated scoring kernels for SMLM score.

Provides GPU-accelerated versions of compute_nb_gmm and _compute_distance_score_numba
using numba.cuda. Falls back gracefully to CPU when CUDA is unavailable.

Usage:
    from smlm_score.imp_modeling.scoring.cuda_kernels import (
        HAS_CUDA, compute_nb_gmm_gpu, compute_distance_score_gpu
    )
"""
import numpy as np
import math

# ---------------------------------------------------------------------------
# GPU availability detection
# ---------------------------------------------------------------------------
try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    _cuda_logged = False

    def _log_cuda_once():
        global _cuda_logged
        if not _cuda_logged:
            gpu = cuda.get_current_device()
            print(f"[CUDA] Using GPU: {gpu.name} (compute capability {gpu.compute_capability})")
            _cuda_logged = True

    from numba import float64

    # -------------------------------------------------------------------
    # Device helper: 3x3 determinant
    # -------------------------------------------------------------------
    @cuda.jit(device=True)
    def _det3x3(m):
        """Compute determinant of a 3x3 matrix stored in a local array."""
        return (m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
              - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
              + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]))

    # -------------------------------------------------------------------
    # Device helper: 3x3 inverse (writes into pre-allocated out array)
    # -------------------------------------------------------------------
    @cuda.jit(device=True)
    def _inv3x3(m, out, inv_det):
        """Compute inverse of 3x3 matrix m, store in out. inv_det = 1/det(m)."""
        out[0, 0] = (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) * inv_det
        out[0, 1] = (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) * inv_det
        out[0, 2] = (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) * inv_det
        out[1, 0] = (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) * inv_det
        out[1, 1] = (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) * inv_det
        out[1, 2] = (m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]) * inv_det
        out[2, 0] = (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]) * inv_det
        out[2, 1] = (m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]) * inv_det
        out[2, 2] = (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) * inv_det

    # ===================================================================
    # GMM scoring CUDA kernel
    # ===================================================================
    @cuda.jit
    def _gmm_score_kernel(model_xyzs, data_mean, inv_Sigmas, log_prefactors,
                          n_models, n_data, output):
        """
        Each thread handles one GMM model point (index m).
        Computes log-sum-exp over data components for that model point.
        Writes partial score to output[m].
        """
        m = cuda.grid(1)
        if m >= n_models:
            return

        mx = model_xyzs[m, 0]
        my = model_xyzs[m, 1]
        mz = model_xyzs[m, 2]

        # Pass 1: find max_log_prob
        max_lp = -1e20 # A very low number
        for i in range(n_data):
            dx = mx - data_mean[i, 0]
            dy = my - data_mean[i, 1]
            dz = mz - data_mean[i, 2]

            t0 = inv_Sigmas[i, 0, 0] * dx + inv_Sigmas[i, 0, 1] * dy + inv_Sigmas[i, 0, 2] * dz
            t1 = inv_Sigmas[i, 1, 0] * dx + inv_Sigmas[i, 1, 1] * dy + inv_Sigmas[i, 1, 2] * dz
            t2 = inv_Sigmas[i, 2, 0] * dx + inv_Sigmas[i, 2, 1] * dy + inv_Sigmas[i, 2, 2] * dz

            exponent = -0.5 * (dx * t0 + dy * t1 + dz * t2)
            lp = log_prefactors[i] + exponent
            if lp > max_lp:
                max_lp = lp

        # Pass 2: compute log-sum-exp
        sum_exp = 0.0
        for i in range(n_data):
            dx = mx - data_mean[i, 0]
            dy = my - data_mean[i, 1]
            dz = mz - data_mean[i, 2]

            t0 = inv_Sigmas[i, 0, 0] * dx + inv_Sigmas[i, 0, 1] * dy + inv_Sigmas[i, 0, 2] * dz
            t1 = inv_Sigmas[i, 1, 0] * dx + inv_Sigmas[i, 1, 1] * dy + inv_Sigmas[i, 1, 2] * dz
            t2 = inv_Sigmas[i, 2, 0] * dx + inv_Sigmas[i, 2, 1] * dy + inv_Sigmas[i, 2, 2] * dz

            exponent = -0.5 * (dx * t0 + dy * t1 + dz * t2)
            lp = log_prefactors[i] + exponent
            
            if lp - max_lp > -80.0: # Optimization: skip exp if way below
                sum_exp += math.exp(lp - max_lp)

        if sum_exp > 0:
            output[m] = max_lp + math.log(sum_exp)
        else:
            output[m] = max_lp

    # ===================================================================
    # Distance scoring CUDA kernel
    # ===================================================================
    @cuda.jit
    def _distance_score_kernel(model_xyzs, data_mean, data_cov, weights,
                               sigma_av, n_models, n_data, output):
        """
        Each thread handles one data point (index i).
        Computes log-sum-exp over model points for that data point.
        Writes result to output[i].
        """
        i = cuda.grid(1)
        if i >= n_data:
            return

        # Pre-allocate local array for per-model log-probs (max 64 model points)
        m_log_probs = cuda.local.array(64, dtype=float64)

        for m in range(n_models):
            # Build Sigma = data_cov[i] + eye(3) * sigma_av
            S = cuda.local.array((3, 3), dtype=float64)
            for r in range(3):
                for c in range(3):
                    S[r, c] = data_cov[i, r, c]
                S[r, r] += sigma_av

            det_S = _det3x3(S)
            inv_S = cuda.local.array((3, 3), dtype=float64)
            _inv3x3(S, inv_S, 1.0 / det_S)

            prefactor = weights[i] / (
                math.pow(2.0 * math.pi, 1.5) * math.pow(abs(det_S), 0.5)
            )

            dx = model_xyzs[m, 0] - data_mean[i, 0]
            dy = model_xyzs[m, 1] - data_mean[i, 1]
            dz = model_xyzs[m, 2] - data_mean[i, 2]

            t0 = inv_S[0, 0] * dx + inv_S[0, 1] * dy + inv_S[0, 2] * dz
            t1 = inv_S[1, 0] * dx + inv_S[1, 1] * dy + inv_S[1, 2] * dz
            t2 = inv_S[2, 0] * dx + inv_S[2, 1] * dy + inv_S[2, 2] * dz

            exponent = -0.5 * (dx * t0 + dy * t1 + dz * t2)
            m_log_probs[m] = math.log(prefactor) + exponent

        # Log-sum-exp over model points
        max_lp = m_log_probs[0]
        for m in range(1, n_models):
            if m_log_probs[m] > max_lp:
                max_lp = m_log_probs[m]

        sum_exp = 0.0
        for m in range(n_models):
            sum_exp += math.exp(m_log_probs[m] - max_lp)

        output[i] = max_lp + math.log(sum_exp)

    # ===================================================================
    # Host-side wrapper: GMM GPU scoring
    # ===================================================================
    def compute_nb_gmm_gpu(model_xyzs, data_mean, data_cov, data_weight,
                           offset_xyz=None):
        """
        GPU-accelerated version of compute_nb_gmm.
        Returns the same scalar score as the CPU version.
        """
        _log_cuda_once()

        # Apply offset on host
        if offset_xyz is not None:
            model_xyzs = model_xyzs + offset_xyz

        model_xyzs = np.ascontiguousarray(model_xyzs, dtype=np.float64)
        data_mean = np.ascontiguousarray(data_mean, dtype=np.float64)
        data_cov = np.ascontiguousarray(data_cov, dtype=np.float64)
        data_weight = np.ascontiguousarray(data_weight, dtype=np.float64)

        n_data = data_mean.shape[0]
        n_models = model_xyzs.shape[0]
        sigma_av = 8.0

        # Precompute constants on host
        inv_Sigmas = np.zeros((n_data, 3, 3), dtype=np.float64)
        log_prefactors = np.zeros(n_data, dtype=np.float64)
        sigma_M = np.eye(3, dtype=np.float64) * sigma_av
        weight_M = 1.0

        for i in range(n_data):
            Sigma = data_cov[i] + sigma_M
            det_S = np.linalg.det(Sigma)
            inv_Sigmas[i] = np.linalg.inv(Sigma)
            prefactor = (weight_M * data_weight[i]) / (
                math.pow(2.0 * math.pi, 1.5) * math.pow(abs(det_S), 0.5)
            )
            if prefactor > 0:
                log_prefactors[i] = math.log(prefactor)
            else:
                log_prefactors[i] = -float('inf')

        # Transfer to GPU
        d_model = cuda.to_device(model_xyzs)
        d_mean = cuda.to_device(data_mean)
        d_inv_Sigmas = cuda.to_device(inv_Sigmas)
        d_log_prefactors = cuda.to_device(log_prefactors)
        
        # Output array sized for models, since each thread writes one model score
        d_output = cuda.device_array(n_models, dtype=np.float64)

        # Launch kernel over n_models
        threads_per_block = 256
        blocks = (n_models + threads_per_block - 1) // threads_per_block
        _gmm_score_kernel[blocks, threads_per_block](
            d_model, d_mean, d_inv_Sigmas, d_log_prefactors,
            n_models, n_data, d_output
        )

        # Copy result back and sum
        output = d_output.copy_to_host()
        return float(np.sum(output))

    # ===================================================================
    # Host-side wrapper: Distance GPU scoring
    # ===================================================================
    def compute_distance_score_gpu(datamean, datacov, weights, modelxyzs,
                                   sigmaav):
        """
        GPU-accelerated version of _compute_distance_score_numba.
        Returns the same scalar score as the CPU version.
        """
        _log_cuda_once()

        datamean = np.ascontiguousarray(datamean, dtype=np.float64)
        datacov = np.ascontiguousarray(datacov, dtype=np.float64)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        modelxyzs = np.ascontiguousarray(modelxyzs, dtype=np.float64)

        n_data = datamean.shape[0]
        n_models = modelxyzs.shape[0]

        # Transfer to GPU
        d_model = cuda.to_device(modelxyzs)
        d_mean = cuda.to_device(datamean)
        d_cov = cuda.to_device(datacov)
        d_weights = cuda.to_device(weights)
        d_output = cuda.device_array(n_data, dtype=np.float64)

        # Launch kernel
        threads_per_block = 256
        blocks = (n_data + threads_per_block - 1) // threads_per_block
        _distance_score_kernel[blocks, threads_per_block](
            d_model, d_mean, d_cov, d_weights,
            sigmaav, n_models, n_data, d_output
        )

        # Copy result back and sum
        output = d_output.copy_to_host()
        return float(np.sum(output))

    # Minimum data size to justify GPU overhead (memory transfer cost).
    # Keeping this modestly high avoids noisy GPU warnings for medium-sized
    # example runs where the CPU path is typically just as good.
    CUDA_MIN_DATA_SIZE = 256
