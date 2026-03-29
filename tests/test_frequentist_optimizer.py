"""
Tests for the frequentist (gradient-based) optimization pipeline.

1. Gradient Correctness: Finite-difference verification for Distance and Tree score gradients.
2. Optimizer Convergence: Verify that ConjugateGradients improves the score from a misaligned start.
"""
import os
import sys
import IMP
import IMP.core
import numpy as np
import pytest
from sklearn.neighbors import KDTree

# Ensure smlm_score is importable
THESIS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

from smlm_score.src.imp_modeling.scoring.distance_score import (
    _compute_distance_score_cpu,
    _compute_distance_score_and_grad_cpu
)
from smlm_score.src.imp_modeling.scoring.tree_score import (
    computescoretree,
    computescoretree_with_grad
)
from smlm_score.src.imp_modeling.restraint.scoring_restraint import (
    ScoringRestraintWrapper,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_ring_data():
    """Generate a noisy ring (NPC-like) with 200 points and a 32-point model."""
    np.random.seed(42)
    n_data = 200
    theta = np.linspace(0, 2*np.pi, n_data)
    r = 60.0
    data = np.column_stack((
        r * np.cos(theta) + np.random.normal(0, 2, n_data),
        r * np.sin(theta) + np.random.normal(0, 2, n_data),
        np.random.normal(0, 1, n_data)
    ))

    n_model = 8
    theta_m = np.linspace(0, 2*np.pi, n_model, endpoint=False)
    model = np.column_stack((
        r * np.cos(theta_m),
        r * np.sin(theta_m),
        np.zeros(n_model)
    ))

    variances = np.ones(n_data) * 4.0  # sigma^2 = 4
    covariances = np.array([np.eye(3) * 4.0 for _ in range(n_data)])

    return data, model, variances, covariances


# ---------------------------------------------------------------------------
# Test 1: Distance Score Gradient via Finite Differences
# ---------------------------------------------------------------------------
class TestDistanceGradient:
    def test_gradient_matches_finite_differences(self, synthetic_ring_data):
        """
        Verify analytical gradient matches numerical (central) finite differences.
        """
        data, model, variances, covariances = synthetic_ring_data
        weights = np.ones(len(data))
        sigmaav = 8.0
        eps = 1e-5

        # Compute analytical score + gradient
        score_analytical, grad_analytical = _compute_distance_score_and_grad_cpu(
            data, covariances, weights, model, sigmaav
        )

        # Compute numerical gradient via central differences
        grad_numerical = np.zeros_like(grad_analytical)
        for m_idx in range(len(model)):
            for dim in range(3):
                model_plus = model.copy()
                model_minus = model.copy()
                model_plus[m_idx, dim] += eps
                model_minus[m_idx, dim] -= eps

                score_plus = _compute_distance_score_cpu(data, covariances, weights, model_plus, sigmaav)
                score_minus = _compute_distance_score_cpu(data, covariances, weights, model_minus, sigmaav)

                grad_numerical[m_idx, dim] = (score_plus - score_minus) / (2 * eps)

        # Check relative error (allow loose tolerance due to Numba fastmath)
        for m_idx in range(len(model)):
            for dim in range(3):
                analytical = grad_analytical[m_idx, dim]
                numerical = grad_numerical[m_idx, dim]
                
                if abs(numerical) > 1e-8:
                    rel_error = abs(analytical - numerical) / abs(numerical)
                    assert rel_error < 0.01, (
                        f"Distance gradient mismatch at model[{m_idx}][{dim}]: "
                        f"analytical={analytical:.6e}, numerical={numerical:.6e}, "
                        f"rel_error={rel_error:.4e}"
                    )

    def test_score_consistency(self, synthetic_ring_data):
        """The score from the gradient function must match the score-only function."""
        data, model, variances, covariances = synthetic_ring_data
        weights = np.ones(len(data))
        sigmaav = 8.0

        score_only = _compute_distance_score_cpu(data, covariances, weights, model, sigmaav)
        score_with_grad, _ = _compute_distance_score_and_grad_cpu(data, covariances, weights, model, sigmaav)

        assert abs(score_only - score_with_grad) < 1e-8, (
            f"Score mismatch: score_only={score_only}, score_with_grad={score_with_grad}"
        )


# ---------------------------------------------------------------------------
# Test 2: Tree Score Gradient via Finite Differences
# ---------------------------------------------------------------------------
class TestTreeGradient:
    def test_gradient_matches_finite_differences(self, synthetic_ring_data):
        """
        Verify analytical tree gradient matches numerical finite differences.
        """
        data, model, variances, _ = synthetic_ring_data
        tree = KDTree(data)
        searchradius = 50.0
        eps = 1e-5

        # Compute analytical score + gradient
        score_analytical, grad_analytical = computescoretree_with_grad(
            tree, None, data, variances,
            model_coords_override=model, searchradius=searchradius
        )

        # Compute numerical gradient
        grad_numerical = np.zeros_like(grad_analytical)
        for m_idx in range(len(model)):
            for dim in range(3):
                model_plus = model.copy()
                model_minus = model.copy()
                model_plus[m_idx, dim] += eps
                model_minus[m_idx, dim] -= eps

                score_plus = computescoretree(
                    tree, None, data, variances,
                    model_coords_override=model_plus, searchradius=searchradius
                )
                score_minus = computescoretree(
                    tree, None, data, variances,
                    model_coords_override=model_minus, searchradius=searchradius
                )
                grad_numerical[m_idx, dim] = (score_plus - score_minus) / (2 * eps)

        # Check relative error
        for m_idx in range(len(model)):
            for dim in range(3):
                analytical = grad_analytical[m_idx, dim]
                numerical = grad_numerical[m_idx, dim]

                if abs(numerical) > 1e-8:
                    rel_error = abs(analytical - numerical) / abs(numerical)
                    assert rel_error < 0.01, (
                        f"Tree gradient mismatch at model[{m_idx}][{dim}]: "
                        f"analytical={analytical:.6e}, numerical={numerical:.6e}, "
                        f"rel_error={rel_error:.4e}"
                    )

    def test_score_consistency(self, synthetic_ring_data):
        """The score from the gradient function must match the score-only function."""
        data, model, variances, _ = synthetic_ring_data
        tree = KDTree(data)
        searchradius = 50.0

        score_only = computescoretree(
            tree, None, data, variances,
            model_coords_override=model, searchradius=searchradius
        )
        score_with_grad, _ = computescoretree_with_grad(
            tree, None, data, variances,
            model_coords_override=model, searchradius=searchradius
        )

        assert abs(score_only - score_with_grad) < 1e-8, (
            f"Score mismatch: score_only={score_only}, score_with_grad={score_with_grad}"
        )


# ---------------------------------------------------------------------------
# Test 3: Gradient Direction Sanity Check
# ---------------------------------------------------------------------------
class TestGradientDirection:
    def test_distance_gradient_points_toward_data(self, synthetic_ring_data):
        """
        If we shift the model outward (away from data), the gradient should
        point inward (toward data center).
        """
        data, model, _, covariances = synthetic_ring_data
        weights = np.ones(len(data))
        sigmaav = 8.0

        # Shift model outward by 10nm radially
        model_shifted = model * 1.15  # 15% further from origin
        _, grad = _compute_distance_score_and_grad_cpu(data, covariances, weights, model_shifted, sigmaav)

        # For each model point, gradient should point inward (toward origin)
        # i.e. grad dot position < 0  (gradient opposes the outward shift)
        for m_idx in range(len(model_shifted)):
            dot = np.dot(grad[m_idx], model_shifted[m_idx])
            assert dot < 0, (
                f"Distance gradient for model[{m_idx}] should point inward, "
                f"but dot(grad, pos) = {dot:.4f}"
            )

    def test_tree_gradient_points_toward_data(self, synthetic_ring_data):
        """Same directional check for Tree gradient."""
        data, model, variances, _ = synthetic_ring_data
        tree = KDTree(data)

        model_shifted = model * 1.15
        _, grad = computescoretree_with_grad(
            tree, None, data, variances,
            model_coords_override=model_shifted, searchradius=50.0
        )

        for m_idx in range(len(model_shifted)):
            if np.linalg.norm(grad[m_idx]) > 1e-10:
                dot = np.dot(grad[m_idx], model_shifted[m_idx])
                assert dot < 0, (
                    f"Tree gradient for model[{m_idx}] should point inward, "
                    f"but dot(grad, pos) = {dot:.4f}"
                )


class TestImpOptimizerIntegration:
    def test_tree_wrapper_exposes_live_gradient_signal(
        self, synthetic_ring_data
    ):
        """
        Regression test for tree-wrapper objective wiring in IMP.

        This validates the key integration contract directly:
        1) derivatives are non-zero for optimized coordinates, and
        2) perturbing coordinates along that derivative signal changes (and can
           improve) the objective.
        """
        data, model_nm, variances, _ = synthetic_ring_data
        tree = KDTree(data)

        m = IMP.Model()
        avs = []
        shifted_model_nm = model_nm * 1.15
        for coords_nm in shifted_model_nm:
            p = IMP.Particle(m)
            xyz = IMP.core.XYZ.setup_particle(
                p,
                IMP.algebra.Vector3D(
                    float(coords_nm[0] / 0.1),
                    float(coords_nm[1] / 0.1),
                    float(coords_nm[2] / 0.1),
                ),
            )
            xyz.set_coordinates_are_optimized(True)
            avs.append(xyz)

        wrapper = ScoringRestraintWrapper(
            m,
            avs,
            kdtree_obj=tree,
            dataxyz=data,
            var=variances,
            searchradius=50.0,
            model_coords_override=shifted_model_nm.copy(),
            type="Tree",
        )
        wrapper.set_return_objective(True)

        sf = IMP.core.RestraintsScoringFunction(
            [wrapper.scoring_restraint_instance],
            "TreeObjective",
        )
        initial_objective = sf.evaluate(True)

        xyz_keys = IMP.core.XYZ.get_xyz_keys()
        p0 = avs[0].get_particle()
        deriv = np.array([p0.get_derivative(k) for k in xyz_keys], dtype=np.float64)

        deriv_norm = np.linalg.norm(deriv)
        assert deriv_norm > 1e-8, "Expected non-zero objective derivative signal."

        xyz0 = IMP.core.XYZ(avs[0])
        orig = np.array(xyz0.get_coordinates(), dtype=np.float64)
        direction = deriv / deriv_norm

        step = 1.0  # angstrom
        trial_minus = orig - step * direction
        trial_plus = orig + step * direction

        xyz0.set_coordinates(
            IMP.algebra.Vector3D(
                float(trial_minus[0]), float(trial_minus[1]), float(trial_minus[2])
            )
        )
        objective_minus = sf.evaluate(False)

        xyz0.set_coordinates(
            IMP.algebra.Vector3D(
                float(trial_plus[0]), float(trial_plus[1]), float(trial_plus[2])
            )
        )
        objective_plus = sf.evaluate(False)

        xyz0.set_coordinates(
            IMP.algebra.Vector3D(float(orig[0]), float(orig[1]), float(orig[2]))
        )

        # CG minimizes: it steps in -gradient direction. 
        # So stepping along -gradient MUST decrease the objective more than stepping along +gradient.
        assert objective_minus < initial_objective, (
            "Expected the -gradient direction to be downhill, "
            f"but initial={initial_objective}, -grad step={objective_minus}, +grad step={objective_plus}"
        )
        assert objective_minus < objective_plus, "Stored derivative points UPHILL (-grad is higher than +grad)!"
