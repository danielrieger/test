"""
Isolation tests for Stage 5 PCA alignment internals with mocking.
"""

import numpy as np
import pytest

from smlm_score.utility import data_handling


@pytest.mark.unit
def test_stage5_alignment_isolation_covariance_receives_centered_data(monkeypatch):
    pts = np.array(
        [
            [10.0, 20.0, 30.0],
            [13.0, 22.0, 35.0],
            [8.0, 19.0, 28.0],
            [12.0, 18.0, 33.0],
        ],
        dtype=float,
    )
    expected_centered = pts - pts.mean(axis=0)

    def fake_cov(arr, rowvar=False):
        assert rowvar is False
        assert np.allclose(arr, expected_centered)
        return np.eye(3)

    def fake_eigh(_cov):
        return np.array([1.0, 2.0, 3.0]), np.eye(3)

    monkeypatch.setattr(np, "cov", fake_cov)
    monkeypatch.setattr(np.linalg, "eigh", fake_eigh)

    out = data_handling.align_npc_cluster_pca(pts, debug=False)

    assert np.allclose(out["translation"], -pts.mean(axis=0))
    assert out["aligned_data"].shape == pts.shape


@pytest.mark.unit
def test_stage5_alignment_isolation_reflection_fix_enforces_positive_determinant(monkeypatch):
    pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 1.0, 3.0],
        ],
        dtype=float,
    )

    def fake_cov(_arr, rowvar=False):
        return np.eye(3)

    def fake_eigh(_cov):
        # Sorting by descending eigenvalue yields column order [2,1,0]:
        # resulting rotation starts as an odd permutation matrix (det = -1),
        # which must trigger the reflection fix branch.
        return np.array([1.0, 2.0, 3.0]), np.eye(3)

    monkeypatch.setattr(np, "cov", fake_cov)
    monkeypatch.setattr(np.linalg, "eigh", fake_eigh)

    out = data_handling.align_npc_cluster_pca(pts, debug=False)
    rot = out["rotation"]

    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-12)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
