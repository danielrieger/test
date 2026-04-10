"""
Integration tests for Stage 4 clustering behavior with real sklearn clustering.
"""

import numpy as np
import pytest

from smlm_score.utility import data_handling


def _make_gaussian_cluster(center, n_points, sigma, seed):
    rng = np.random.RandomState(seed)
    return rng.normal(loc=np.array(center), scale=sigma, size=(n_points, 3))


@pytest.mark.integration
def test_stage4_integration_raw_hdbscan_detects_two_dense_clusters():
    c1 = _make_gaussian_cluster(center=[0.0, 0.0, 0.0], n_points=40, sigma=1.0, seed=1)
    c2 = _make_gaussian_cluster(center=[300.0, 300.0, 0.0], n_points=35, sigma=1.2, seed=2)
    noise = np.array([[800.0, 0.0, 0.0], [0.0, 800.0, 0.0], [900.0, 900.0, 0.0]], dtype=float)
    smlm_data = np.vstack([c1, c2, noise]).astype(float)

    out = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=10,
        min_npc_points=20,
        use_xy_only=True,
        perform_geometric_merging=False,
    )

    assert len(out["all_cluster_info"]) >= 2
    assert out["n_npcs"] >= 2
    assert np.sum(out["labels"] == -1) >= 1


@pytest.mark.integration
def test_stage4_integration_use_xy_only_flag_changes_z_separated_case():
    rng = np.random.RandomState(42)
    xy = rng.normal(loc=[100.0, 100.0], scale=[0.8, 0.8], size=(80, 2))
    z_low = rng.normal(loc=-50.0, scale=0.6, size=(40, 1))
    z_high = rng.normal(loc=50.0, scale=0.6, size=(40, 1))

    slab1 = np.hstack([xy[:40], z_low])
    slab2 = np.hstack([xy[40:], z_high])
    smlm_data = np.vstack([slab1, slab2]).astype(float)

    out_xy = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=8,
        min_npc_points=10,
        use_xy_only=True,
        perform_geometric_merging=False,
    )
    out_xyz = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=8,
        min_npc_points=10,
        use_xy_only=False,
        perform_geometric_merging=False,
    )

    n_xy = len(out_xy["all_cluster_info"])
    n_xyz = len(out_xyz["all_cluster_info"])
    assert n_xyz >= n_xy


@pytest.mark.integration
def test_stage4_integration_two_stage_preserves_noise_count():
    c1 = _make_gaussian_cluster(center=[0.0, 0.0, 0.0], n_points=30, sigma=1.0, seed=7)
    c2 = _make_gaussian_cluster(center=[120.0, 0.0, 0.0], n_points=30, sigma=1.0, seed=8)
    noise = np.array([[500.0, 500.0, 0.0], [550.0, 560.0, 0.0], [600.0, 610.0, 0.0]], dtype=float)
    smlm_data = np.vstack([c1, c2, noise]).astype(float)

    out_raw = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=8,
        min_npc_points=10,
        use_xy_only=True,
        perform_geometric_merging=False,
    )
    out_two_stage = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=8,
        min_npc_points=10,
        use_xy_only=True,
        perform_geometric_merging=True,
    )

    raw_noise = int(np.sum(out_raw["labels"] == -1))
    two_stage_noise = int(np.sum(out_two_stage["labels"] == -1))
    assert raw_noise == two_stage_noise


@pytest.mark.integration
def test_stage4_integration_tiny_dataset_returns_all_noise():
    """
    Tiny datasets should be handled gracefully without raising:
    output should be all-noise and zero NPC clusters.
    """
    smlm_data = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=float,
    )

    out = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=10,
        min_npc_points=5,
        use_xy_only=True,
        perform_geometric_merging=True,
    )

    assert np.array_equal(out["labels"], np.array([-1, -1, -1], dtype=int))
    assert np.array_equal(out["probabilities"], np.array([0.0, 0.0, 0.0], dtype=float))
    assert out["n_npcs"] == 0
    assert out["npc_info"] == []
    assert out["all_cluster_info"] == []


@pytest.mark.integration
def test_stage4_integration_two_stage_labels_are_contiguous():
    """
    After two-stage relabeling, non-noise labels should be contiguous 0..n-1.
    """
    c1 = _make_gaussian_cluster(center=[0.0, 0.0, 0.0], n_points=35, sigma=1.2, seed=11)
    c2 = _make_gaussian_cluster(center=[100.0, 0.0, 0.0], n_points=35, sigma=1.2, seed=12)
    c3 = _make_gaussian_cluster(center=[230.0, 0.0, 0.0], n_points=35, sigma=1.2, seed=13)
    smlm_data = np.vstack([c1, c2, c3]).astype(float)

    out = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=10,
        min_npc_points=1,
        use_xy_only=True,
        perform_geometric_merging=True,
    )

    non_noise = sorted(set(out["labels"]) - {-1})
    assert non_noise == list(range(len(non_noise)))
