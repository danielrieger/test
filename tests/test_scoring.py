"""
Unit tests for smlm_score scoring functions.

Tests the core mathematical functions directly with synthetic numpy data,
without requiring IMP dependencies. This enables fast CI testing.
"""
import numpy as np
import pytest
from sklearn.neighbors import KDTree


# ============================================================================
# Helper: import core math functions (no IMP dependency)
# ============================================================================

# GMM core function (Numba JIT compiled)
from smlm_score.src.imp_modeling.scoring.gmm_score import compute_nb_gmm
# GMM CPU-only for direct testing
from smlm_score.src.imp_modeling.scoring.gmm_score import _compute_nb_gmm_cpu

# Distance core function (Numba JIT compiled)
from smlm_score.src.imp_modeling.scoring.distance_score import _compute_distance_score_cpu

# Tree score variance helper
from smlm_score.src.imp_modeling.scoring.tree_score import _extract_scalar_variance
from smlm_score.src.imp_modeling.scoring.tree_score import (
    TREE_EXACT_MODEL_COUNT_THRESHOLD,
)

# CUDA availability
from smlm_score.src.imp_modeling.scoring.cuda_kernels import HAS_CUDA
if HAS_CUDA:
    from smlm_score.src.imp_modeling.scoring.cuda_kernels import (
        compute_nb_gmm_gpu, compute_distance_score_gpu
    )

# Backward-compatible alias used throughout this file
_compute_distance_score_numba = _compute_distance_score_cpu


# ============================================================================
# Test: _extract_scalar_variance (tree score helper)
# ============================================================================

class TestExtractScalarVariance:
    """Tests for the variance extraction helper used by tree scoring."""

    def test_scalar_float(self):
        assert _extract_scalar_variance(2.5) == 2.5

    def test_scalar_int(self):
        assert _extract_scalar_variance(3) == 3.0

    def test_scalar_numpy_float(self):
        assert _extract_scalar_variance(np.float64(1.5)) == 1.5

    def test_zero_floors_to_epsilon(self):
        result = _extract_scalar_variance(0.0)
        assert result > 0
        assert result == 1e-12

    def test_negative_floors_to_epsilon(self):
        result = _extract_scalar_variance(-5.0)
        assert result == 1e-12

    def test_1d_array_returns_mean(self):
        var_arr = np.array([1.0, 2.0, 3.0])
        assert _extract_scalar_variance(var_arr) == pytest.approx(2.0)

    def test_2d_identity_returns_diagonal_mean(self):
        cov = np.eye(3) * 4.0
        assert _extract_scalar_variance(cov) == pytest.approx(4.0)

    def test_2d_diagonal_matrix(self):
        cov = np.diag([1.0, 2.0, 3.0])
        assert _extract_scalar_variance(cov) == pytest.approx(2.0)

    def test_2d_full_covariance_uses_diagonal(self):
        cov = np.array([[4.0, 1.0, 0.5],
                        [1.0, 6.0, 0.3],
                        [0.5, 0.3, 2.0]])
        # Mean of diagonal: (4 + 6 + 2) / 3 = 4.0
        assert _extract_scalar_variance(cov) == pytest.approx(4.0)

    def test_0d_array(self):
        assert _extract_scalar_variance(np.array(5.0)) == 5.0

    def test_2d_matrix_2x2_for_2d_data(self):
        """2D data should work with 2x2 covariance matrices."""
        cov = np.eye(2) * 3.0
        assert _extract_scalar_variance(cov) == pytest.approx(3.0)


# ============================================================================
# Test: compute_nb_gmm (GMM scoring core)
# ============================================================================

class TestComputeNbGMM:
    """Tests for the Numba-compiled GMM scoring function."""

    def _make_single_component_gmm(self, center, var=1.0):
        """Helper: creates a single-component isotropic GMM."""
        mean = np.array([center], dtype=np.float64)
        cov = np.array([np.eye(3, dtype=np.float64) * var])
        weight = np.array([1.0], dtype=np.float64)
        return mean, cov, weight

    def test_model_at_data_center_scores_highest(self):
        """Model exactly at GMM center should score highest."""
        mean, cov, weight = self._make_single_component_gmm([0.0, 0.0, 0.0])
        
        model_at_center = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        model_offset = np.array([[10.0, 0.0, 0.0]], dtype=np.float64)
        
        score_center = compute_nb_gmm(model_at_center, mean, cov, weight)
        score_offset = compute_nb_gmm(model_offset, mean, cov, weight)
        
        assert score_center > score_offset, "Model at data center should score higher"

    def test_score_decreases_with_distance(self):
        """Score should decrease monotonically as model moves away from data."""
        mean, cov, weight = self._make_single_component_gmm([0.0, 0.0, 0.0])
        
        distances = [0.0, 1.0, 5.0, 10.0, 50.0]
        scores = []
        for d in distances:
            model = np.array([[d, 0.0, 0.0]], dtype=np.float64)
            scores.append(compute_nb_gmm(model, mean, cov, weight))
        
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], \
                f"Score at distance {distances[i]} should be > score at distance {distances[i+1]}"

    def test_multiple_model_points_accumulate(self):
        """Score with two model points at center should be ~2x single point."""
        mean, cov, weight = self._make_single_component_gmm([0.0, 0.0, 0.0])
        
        model_single = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        model_double = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        
        score_single = compute_nb_gmm(model_single, mean, cov, weight)
        score_double = compute_nb_gmm(model_double, mean, cov, weight)
        
        assert score_double == pytest.approx(2 * score_single, rel=1e-10)

    def test_score_is_finite(self):
        """Score should always be finite for valid inputs."""
        mean, cov, weight = self._make_single_component_gmm([0.0, 0.0, 0.0])
        model = np.array([[100.0, 200.0, 300.0]], dtype=np.float64)
        
        score = compute_nb_gmm(model, mean, cov, weight)
        assert np.isfinite(score)

    def test_multicomponent_gmm(self):
        """Test with multiple GMM components."""
        means = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        covs = np.array([np.eye(3), np.eye(3)], dtype=np.float64)
        weights = np.array([0.5, 0.5], dtype=np.float64)
        
        model = np.array([[5.0, 0.0, 0.0]], dtype=np.float64)  # Midpoint
        score = compute_nb_gmm(model, means, covs, weights)
        assert np.isfinite(score)

    def test_offset_xyz_shifts_model(self):
        """Applying offset should be equivalent to shifting model coords."""
        mean, cov, weight = self._make_single_component_gmm([10.0, 0.0, 0.0])
        offset = np.array([10.0, 0.0, 0.0], dtype=np.float64)
        
        model_at_origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        model_at_target = np.array([[10.0, 0.0, 0.0]], dtype=np.float64)
        
        score_with_offset = compute_nb_gmm(model_at_origin.copy(), mean, cov, weight, offset_xyz=offset)
        score_direct = compute_nb_gmm(model_at_target, mean, cov, weight)
        
        assert score_with_offset == pytest.approx(score_direct, rel=1e-8)


# ============================================================================
# Test: _compute_distance_score_numba (Distance scoring core)
# ============================================================================

class TestComputeDistanceScore:
    """Tests for the Numba-compiled Distance scoring function."""

    def test_model_at_data_center_scores_highest(self):
        """Model at data center should produce highest score."""
        datamean = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        datacov = np.array([np.eye(3, dtype=np.float64)])
        weights = np.array([1.0], dtype=np.float64)
        sigma_av = 8.0
        
        model_center = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        model_far = np.array([[50.0, 0.0, 0.0]], dtype=np.float64)
        
        score_center = _compute_distance_score_numba(datamean, datacov, weights, model_center, sigma_av)
        score_far = _compute_distance_score_numba(datamean, datacov, weights, model_far, sigma_av)
        
        assert score_center > score_far

    def test_score_decreases_with_distance(self):
        """Score should decrease as model moves away from data."""
        datamean = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        datacov = np.array([np.eye(3, dtype=np.float64)])
        weights = np.array([1.0], dtype=np.float64)
        sigma_av = 8.0
        
        distances = [0.0, 5.0, 20.0, 100.0]
        scores = []
        for d in distances:
            model = np.array([[d, 0.0, 0.0]], dtype=np.float64)
            scores.append(_compute_distance_score_numba(datamean, datacov, weights, model, sigma_av))
        
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_score_is_finite(self):
        """Score should be finite for valid inputs."""
        datamean = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        datacov = np.array([np.eye(3, dtype=np.float64) * 2.0])
        weights = np.array([1.0], dtype=np.float64)
        sigma_av = 8.0
        model = np.array([[100.0, 200.0, 300.0]], dtype=np.float64)
        
        score = _compute_distance_score_numba(datamean, datacov, weights, model, sigma_av)
        assert np.isfinite(score)

    def test_logsumexp_numerical_stability(self):
        """LogSumExp should handle very large/small exponents without overflow."""
        datamean = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        datacov = np.array([np.eye(3, dtype=np.float64) * 0.01])  # Very tight distribution
        weights = np.array([1.0], dtype=np.float64)
        sigma_av = 0.01
        
        # Model very far away â†’ exponent will be very negative
        model = np.array([[1000.0, 1000.0, 1000.0]], dtype=np.float64)
        score = _compute_distance_score_numba(datamean, datacov, weights, model, sigma_av)
        
        assert np.isfinite(score), "Score should remain finite even for extreme distances"

    def test_multiple_data_points(self):
        """Score with multiple data points should accumulate correctly."""
        datamean = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        datacov = np.array([np.eye(3, dtype=np.float64), np.eye(3, dtype=np.float64)])
        weights = np.array([0.5, 0.5], dtype=np.float64)
        sigma_av = 8.0
        
        model = np.array([[5.0, 0.0, 0.0]], dtype=np.float64)
        score = _compute_distance_score_numba(datamean, datacov, weights, model, sigma_av)
        assert np.isfinite(score)

    def test_equal_weights_vs_unit_weights(self):
        """Halving all weights should shift scores by a constant log factor."""
        datamean = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        datacov = np.array([np.eye(3, dtype=np.float64)])
        sigma_av = 8.0
        model = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        
        weights_1 = np.array([1.0], dtype=np.float64)
        weights_half = np.array([0.5], dtype=np.float64)
        
        score_1 = _compute_distance_score_numba(datamean, datacov, weights_1, model, sigma_av)
        score_half = _compute_distance_score_numba(datamean, datacov, weights_half, model, sigma_av)
        
        # Halving weight adds log(0.5) to the total
        assert score_half == pytest.approx(score_1 + np.log(0.5), rel=1e-8)


# ============================================================================
# Test: Tree score math (without IMP dependency)
# ============================================================================

class TestTreeScoreMath:
    """
    Tests for the tree score mathematical logic.

    Since computescoretree requires IMP.bff.AV objects, we test the
    underlying math directly with a pure numpy equivalent.
    """

    def _compute_tree_score_pure(self, model_xyzs, data_xyzs, variances, search_radius):
        """
        Pure numpy implementation of the redesigned tree score.

        Same interpretation as distance score:
        sum over data points of log-sum-exp over candidate model points.
        """
        tree = KDTree(data_xyzs)
        n_data = len(data_xyzs)
        ndim = data_xyzs.shape[1]
        sigma_av = 8.0
        eye = np.eye(ndim, dtype=np.float64)

        if len(model_xyzs) <= TREE_EXACT_MODEL_COUNT_THRESHOLD:
            model_candidates_per_data = [
                list(range(len(model_xyzs))) for _ in range(n_data)
            ]
        else:
            data_neighbors_by_model = tree.query_radius(model_xyzs, search_radius)
            model_candidates_per_data = [[] for _ in range(n_data)]
            for model_idx, data_indices in enumerate(data_neighbors_by_model):
                for data_idx in data_indices:
                    model_candidates_per_data[data_idx].append(model_idx)

        all_model_indices = np.arange(len(model_xyzs), dtype=np.int64)
        score = 0.0
        for data_idx in range(n_data):
            candidates = model_candidates_per_data[data_idx]
            if len(candidates) == 0:
                candidates = all_model_indices

            cov_d = eye * variances[data_idx]
            sigma = cov_d + eye * sigma_av
            inv_sigma = np.linalg.inv(sigma)
            sign, logdet = np.linalg.slogdet(sigma)
            assert sign > 0
            log_prefactor = -0.5 * (ndim * np.log(2.0 * np.pi) + logdet)

            x_d = data_xyzs[data_idx]
            log_probs = []
            for model_idx in candidates:
                diff = model_xyzs[int(model_idx)] - x_d
                exponent = -0.5 * diff.T @ inv_sigma @ diff
                log_probs.append(log_prefactor + exponent)

            log_probs = np.array(log_probs, dtype=np.float64)
            m = np.max(log_probs)
            score += m + np.log(np.sum(np.exp(log_probs - m)))

        return float(score)

    def test_model_at_data_scores_highest(self):
        """Model at data point should score highest."""
        data = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)
        var = np.array([1.0, 1.0], dtype=np.float64)

        model_at_data = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        model_far = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)

        score_at = self._compute_tree_score_pure(model_at_data, data, var, 200.0)
        score_far = self._compute_tree_score_pure(model_far, data, var, 200.0)

        assert score_at > score_far

    def test_tree_matches_distance_for_large_radius(self):
        """
        With a large search radius, tree candidates include all model points.
        The tree score should then match the distance score exactly.
        """
        datamean = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        variances = np.array([1.0, 2.0], dtype=np.float64)
        datacov = np.array([np.eye(3) * v for v in variances], dtype=np.float64)
        weights = np.ones(len(datamean), dtype=np.float64)

        model = np.array([[5.0, 0.0, 0.0], [7.5, 0.0, 0.0]], dtype=np.float64)
        score_tree = self._compute_tree_score_pure(
            model, datamean, variances, search_radius=1e6
        )
        score_distance = _compute_distance_score_numba(
            datamean, datacov, weights, model, 8.0
        )

        assert score_tree == pytest.approx(score_distance, rel=1e-8)

    def test_tree_matches_distance_for_small_models_even_with_tight_radius(self):
        """
        For small model sizes, tree scoring uses an exact candidate set so
        the score remains equivalent to distance scoring even with a tight
        search radius.
        """
        datamean = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        variances = np.array([1.0, 2.0], dtype=np.float64)
        datacov = np.array([np.eye(3) * v for v in variances], dtype=np.float64)
        weights = np.ones(len(datamean), dtype=np.float64)

        model = np.array([[5.0, 0.0, 0.0], [7.5, 0.0, 0.0]], dtype=np.float64)
        score_tree = self._compute_tree_score_pure(
            model, datamean, variances, search_radius=1.0
        )
        score_distance = _compute_distance_score_numba(
            datamean, datacov, weights, model, 8.0
        )

        assert score_tree == pytest.approx(score_distance, rel=1e-8)

    def test_no_neighbors_is_penalized(self):
        """No-neighbor situations should still be strongly penalized, not zero."""
        data = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)
        var = np.array([1.0], dtype=np.float64)
        model = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        # Tiny radius means no candidates; fallback still computes full likelihood.
        score = self._compute_tree_score_pure(model, data, var, 1.0)
        assert np.isfinite(score)
        assert score < -100.0

    def test_2d_data_support(self):
        """Score should work correctly with 2D data (z=0 or no z)."""
        data_2d = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        var_2d = np.array([1.0, 1.0], dtype=np.float64)
        model_2d = np.array([[0.0, 0.0]], dtype=np.float64)

        score = self._compute_tree_score_pure(model_2d, data_2d, var_2d, 10.0)
        assert np.isfinite(score)

    def test_scaling_converts_angstrom_to_nm(self):
        """Scaling should properly convert model coordinates."""
        data = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
        var = np.array([1.0], dtype=np.float64)

        model_angstrom = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)
        model_nm = model_angstrom * 0.1

        score_scaled = self._compute_tree_score_pure(model_nm, data, var, 10.0)
        score_unscaled = self._compute_tree_score_pure(model_angstrom, data, var, 200.0)

        assert score_scaled > score_unscaled, (
            f"Aligned score ({score_scaled:.2f}) should be better than "
            f"misaligned score ({score_unscaled:.2f})"
        )

# ============================================================================
# Test: Score Comparisons (relative behavior across scoring functions)
# ============================================================================

class TestScoreComparativeBehavior:
    """Tests verifying that all scoring functions agree on relative ordering."""

    def test_all_scores_prefer_closer_model(self):
        """All scoring functions should give higher score to closer model."""
        # Setup
        data_center = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        data_cov = np.array([np.eye(3, dtype=np.float64)])
        weights = np.array([1.0], dtype=np.float64)
        
        model_close = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        model_far = np.array([[50.0, 0.0, 0.0]], dtype=np.float64)
        
        # GMM
        gmm_close = compute_nb_gmm(model_close, data_center, data_cov, weights)
        gmm_far = compute_nb_gmm(model_far, data_center, data_cov, weights)
        assert gmm_close > gmm_far, "GMM should prefer closer model"
        
        # Distance
        dist_close = _compute_distance_score_numba(data_center, data_cov, weights, model_close, 8.0)
        dist_far = _compute_distance_score_numba(data_center, data_cov, weights, model_far, 8.0)
        assert dist_close > dist_far, "Distance should prefer closer model"


# ============================================================================
# Test: CUDA GPU Consistency (skipped when no GPU available)
# ============================================================================

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestCUDAGPUConsistency:
    """Tests that GPU kernels produce the same results as CPU Numba JIT."""

    def test_gmm_gpu_matches_cpu_single_component(self):
        """GPU GMM score should match CPU for a single GMM component."""
        mean = np.array([[5.0, 3.0, 1.0]], dtype=np.float64)
        cov = np.array([np.eye(3, dtype=np.float64) * 2.0])
        weight = np.array([1.0], dtype=np.float64)
        model = np.array([[5.0, 3.0, 1.0], [10.0, 0.0, 0.0]], dtype=np.float64)

        score_cpu = _compute_nb_gmm_cpu(model, mean, cov, weight)
        score_gpu = compute_nb_gmm_gpu(model, mean, cov, weight)
        assert score_gpu == pytest.approx(score_cpu, rel=1e-6)

    def test_gmm_gpu_matches_cpu_many_components(self):
        """GPU GMM score should match CPU for many GMM components."""
        rng = np.random.RandomState(42)
        n_comp = 128
        means = rng.randn(n_comp, 3).astype(np.float64) * 10
        covs = np.array([np.eye(3, dtype=np.float64) * (1.0 + rng.rand())
                         for _ in range(n_comp)])
        weights = np.ones(n_comp, dtype=np.float64) / n_comp
        model = rng.randn(8, 3).astype(np.float64) * 5

        score_cpu = _compute_nb_gmm_cpu(model, means, covs, weights)
        score_gpu = compute_nb_gmm_gpu(model, means, covs, weights)
        assert score_gpu == pytest.approx(score_cpu, rel=1e-6)

    def test_gmm_gpu_with_offset(self):
        """GPU GMM score with offset should match CPU."""
        mean = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)
        cov = np.array([np.eye(3), np.eye(3)], dtype=np.float64)
        weight = np.array([0.5, 0.5], dtype=np.float64)
        model = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        offset = np.array([2.0, 3.0, -1.0], dtype=np.float64)

        score_cpu = _compute_nb_gmm_cpu(model.copy(), mean, cov, weight, offset_xyz=offset)
        score_gpu = compute_nb_gmm_gpu(model.copy(), mean, cov, weight, offset_xyz=offset)
        assert score_gpu == pytest.approx(score_cpu, rel=1e-6)

    def test_distance_gpu_matches_cpu(self):
        """GPU Distance score should match CPU."""
        rng = np.random.RandomState(123)
        n_data = 200
        datamean = rng.randn(n_data, 3).astype(np.float64) * 10
        datacov = np.array([np.eye(3, dtype=np.float64) * (1.0 + rng.rand())
                            for _ in range(n_data)])
        weights = np.ones(n_data, dtype=np.float64)
        model = rng.randn(8, 3).astype(np.float64)
        sigma_av = 8.0

        score_cpu = _compute_distance_score_cpu(datamean, datacov, weights, model, sigma_av)
        score_gpu = compute_distance_score_gpu(datamean, datacov, weights, model, sigma_av)
        assert score_gpu == pytest.approx(score_cpu, rel=1e-6)

    def test_distance_gpu_single_point(self):
        """GPU Distance score for single data point should match CPU."""
        datamean = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        datacov = np.array([np.eye(3, dtype=np.float64)])
        weights = np.array([1.0], dtype=np.float64)
        model = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        sigma_av = 8.0

        score_cpu = _compute_distance_score_cpu(datamean, datacov, weights, model, sigma_av)
        score_gpu = compute_distance_score_gpu(datamean, datacov, weights, model, sigma_av)
        assert score_gpu == pytest.approx(score_cpu, rel=1e-6)


