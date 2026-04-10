"""
Unit tests for Stage 5 PCA alignment behavior.
"""

import numpy as np
import pytest

from smlm_score.utility.data_handling import align_npc_cluster_pca


def _make_tilted_ring(n_points=128, radius=50.0):
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    ring_xy = np.column_stack(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros_like(theta),
        ]
    )

    # Fixed 3D tilt (Rx then Ry) for deterministic tests.
    ax = np.deg2rad(35.0)
    ay = np.deg2rad(-20.0)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(ax), -np.sin(ax)],
            [0.0, np.sin(ax), np.cos(ax)],
        ]
    )
    ry = np.array(
        [
            [np.cos(ay), 0.0, np.sin(ay)],
            [0.0, 1.0, 0.0],
            [-np.sin(ay), 0.0, np.cos(ay)],
        ]
    )
    rot = ry @ rx

    offset = np.array([120.0, -35.0, 80.0])
    return ring_xy @ rot.T + offset


@pytest.mark.unit
def test_stage5_alignment_short_input_returns_identity_and_zero_translation():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)

    out = align_npc_cluster_pca(pts, debug=False)

    assert np.array_equal(out["aligned_data"], pts)
    assert np.array_equal(out["translation"], np.zeros(3))
    assert np.array_equal(out["rotation"], np.eye(3))


@pytest.mark.unit
def test_stage5_alignment_centers_data_at_origin():
    pts = _make_tilted_ring()

    out = align_npc_cluster_pca(pts, debug=False)
    aligned = out["aligned_data"]

    assert np.allclose(aligned.mean(axis=0), np.zeros(3), atol=1e-10)


@pytest.mark.unit
def test_stage5_alignment_moves_planar_cluster_into_xy_plane():
    pts = _make_tilted_ring()
    z_std_before = float(np.std(pts[:, 2]))

    out = align_npc_cluster_pca(pts, debug=False)
    aligned = out["aligned_data"]
    z_std_after = float(np.std(aligned[:, 2]))

    assert z_std_before > 1.0
    assert z_std_after < 1e-10


@pytest.mark.unit
def test_stage5_alignment_returns_orthonormal_proper_rotation():
    pts = _make_tilted_ring()

    out = align_npc_cluster_pca(pts, debug=False)
    rotation = out["rotation"]

    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)
