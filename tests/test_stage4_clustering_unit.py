"""
Unit tests for Stage 4 clustering behavior with mocking.
"""

import numpy as np
import pytest

from smlm_score.utility import data_handling


@pytest.mark.unit
def test_stage4_empty_input_returns_empty_outputs():
    out = data_handling.isolate_individual_npcs(
        np.empty((0, 3), dtype=float),
        min_cluster_size=5,
        min_npc_points=10,
    )

    assert out["labels"].size == 0
    assert out["probabilities"].size == 0
    assert out["n_npcs"] == 0
    assert out["npc_info"] == []
    assert out["all_cluster_info"] == []


@pytest.mark.unit
def test_stage4_raw_hdbscan_keeps_original_labels(monkeypatch):
    labels_raw = np.array([0, 0, 1, 1, -1, -1], dtype=int)
    probabilities = np.array([1.0, 0.9, 0.95, 0.8, 0.1, 0.05], dtype=float)

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.probabilities_ = probabilities

        def fit_predict(self, coords):
            assert coords.shape == (6, 2)
            return labels_raw.copy()

    monkeypatch.setattr(data_handling, "HDBSCAN", FakeHDBSCAN)

    smlm_data = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [11.0, 10.0, 0.0],
            [100.0, 100.0, 0.0],
            [101.0, 100.0, 0.0],
        ],
        dtype=float,
    )

    out = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=2,
        min_npc_points=2,
        perform_geometric_merging=False,
    )

    assert np.array_equal(out["labels"], labels_raw)
    assert np.array_equal(out["probabilities"], probabilities)
    assert len(out["all_cluster_info"]) == 2
    assert out["n_npcs"] == 2
    assert [c["cluster_id"] for c in out["npc_info"]] == [0, 1]


@pytest.mark.unit
def test_stage4_two_stage_hdbscan_merges_fragments(monkeypatch):
    labels_fragmented = np.array([0, 1, 0, 1, -1, -1], dtype=int)
    probabilities = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.2], dtype=float)

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.probabilities_ = probabilities

        def fit_predict(self, coords):
            assert coords.shape == (6, 2)
            return labels_fragmented.copy()

    monkeypatch.setattr(data_handling, "HDBSCAN", FakeHDBSCAN)

    import sklearn.cluster as sk_cluster

    init_calls = []

    class FakeAgglomerativeClustering:
        def __init__(self, n_clusters=None, distance_threshold=None, linkage=None):
            init_calls.append(
                {
                    "n_clusters": n_clusters,
                    "distance_threshold": distance_threshold,
                    "linkage": linkage,
                }
            )

        def fit_predict(self, clean_pts):
            assert clean_pts.shape == (4, 2)
            return np.zeros(len(clean_pts), dtype=int)

    monkeypatch.setattr(sk_cluster, "AgglomerativeClustering", FakeAgglomerativeClustering)

    smlm_data = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [100.0, 100.0, 0.0],
            [101.0, 100.0, 0.0],
        ],
        dtype=float,
    )

    out = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=2,
        min_npc_points=3,
        perform_geometric_merging=True,
    )

    assert np.array_equal(out["labels"], np.array([0, 0, 0, 0, -1, -1], dtype=int))
    assert len(out["all_cluster_info"]) == 1
    assert out["n_npcs"] == 1
    assert out["npc_info"][0]["n_points"] == 4
    assert np.array_equal(out["probabilities"], probabilities)
    assert len(init_calls) == 1
    assert init_calls[0]["n_clusters"] is None
    assert init_calls[0]["distance_threshold"] == 140
    assert init_calls[0]["linkage"] == "complete"


@pytest.mark.unit
def test_stage4_all_noise_returns_no_clusters(monkeypatch):
    labels_all_noise = np.array([-1, -1, -1, -1], dtype=int)
    probabilities = np.array([0.1, 0.2, 0.05, 0.3], dtype=float)

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.probabilities_ = probabilities

        def fit_predict(self, coords):
            assert coords.shape == (4, 2)
            return labels_all_noise.copy()

    monkeypatch.setattr(data_handling, "HDBSCAN", FakeHDBSCAN)

    smlm_data = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=float,
    )

    out = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=2,
        min_npc_points=2,
        perform_geometric_merging=False,
    )

    assert np.array_equal(out["labels"], labels_all_noise)
    assert np.array_equal(out["probabilities"], probabilities)
    assert out["n_npcs"] == 0
    assert out["npc_info"] == []
    assert out["all_cluster_info"] == []


@pytest.mark.unit
def test_stage4_use_xy_only_false_clusters_on_3d_coordinates(monkeypatch):
    labels_3d = np.array([0, 0, 1, 1], dtype=int)
    probabilities = np.ones(4, dtype=float)
    seen_shapes = []

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.probabilities_ = probabilities

        def fit_predict(self, coords):
            seen_shapes.append(coords.shape)
            return labels_3d.copy()

    monkeypatch.setattr(data_handling, "HDBSCAN", FakeHDBSCAN)

    smlm_data = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [20.0, 20.0, 50.0],
            [20.0, 20.0, 55.0],
        ],
        dtype=float,
    )

    out = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=2,
        min_npc_points=2,
        use_xy_only=False,
        perform_geometric_merging=False,
    )

    assert seen_shapes == [(4, 3)]
    assert out["n_npcs"] == 2
    assert len(out["all_cluster_info"]) == 2
    assert np.array_equal(out["labels"], labels_3d)


@pytest.mark.unit
def test_stage4_two_stage_skips_agglomerative_when_all_noise(monkeypatch):
    labels_all_noise = np.array([-1, -1, -1], dtype=int)
    probabilities = np.array([0.0, 0.0, 0.0], dtype=float)

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.probabilities_ = probabilities

        def fit_predict(self, coords):
            return labels_all_noise.copy()

    monkeypatch.setattr(data_handling, "HDBSCAN", FakeHDBSCAN)

    import sklearn.cluster as sk_cluster

    class FailIfCalledAgglomerative:
        def __init__(self, *args, **kwargs):
            raise AssertionError("AgglomerativeClustering should not be called for all-noise input")

    monkeypatch.setattr(sk_cluster, "AgglomerativeClustering", FailIfCalledAgglomerative)

    smlm_data = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
            [10.0, 10.0, 0.0],
        ],
        dtype=float,
    )

    out = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=2,
        min_npc_points=2,
        perform_geometric_merging=True,
    )

    assert np.array_equal(out["labels"], labels_all_noise)
    assert out["n_npcs"] == 0


@pytest.mark.unit
def test_stage4_min_npc_points_boundary_is_inclusive(monkeypatch):
    labels = np.array([0, 0, 0, 1, 1], dtype=int)
    probabilities = np.ones(5, dtype=float)

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.probabilities_ = probabilities

        def fit_predict(self, coords):
            return labels.copy()

    monkeypatch.setattr(data_handling, "HDBSCAN", FakeHDBSCAN)

    smlm_data = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [50.0, 50.0, 0.0],
            [51.0, 50.0, 0.0],
        ],
        dtype=float,
    )

    out = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=2,
        min_npc_points=3,
        perform_geometric_merging=False,
    )

    # Cluster 0 has exactly 3 points and should be counted as NPC.
    assert out["n_npcs"] == 1
    assert [c["cluster_id"] for c in out["npc_info"]] == [0]
    assert out["npc_info"][0]["n_points"] == 3


@pytest.mark.unit
def test_stage4_two_stage_respects_140nm_merge_bound(monkeypatch):
    """
    Use mocked HDBSCAN labels (all clean points) but real AgglomerativeClustering.
    Points separated by >140 nm in XY must not be merged into one macro-cluster.
    """
    probabilities = np.ones(4, dtype=float)

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.probabilities_ = probabilities

        def fit_predict(self, coords):
            # Force all points into the clean set to exercise stage-2 merging.
            return np.zeros(len(coords), dtype=int)

    monkeypatch.setattr(data_handling, "HDBSCAN", FakeHDBSCAN)

    smlm_data = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [300.0, 0.0, 0.0],
            [305.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    out = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=2,
        min_npc_points=1,
        use_xy_only=True,
        perform_geometric_merging=True,
    )

    unique_non_noise = sorted(set(out["labels"]) - {-1})
    assert len(unique_non_noise) == 2


@pytest.mark.unit
def test_stage4_two_stage_large_clean_set_uses_cluster_level_merge(monkeypatch):
    """
    Large clean point sets must not be passed directly to point-wise
    AgglomerativeClustering due to quadratic memory use.
    """
    n_points = 6000
    n_clusters = 120
    probabilities = np.ones(n_points, dtype=float)
    labels = np.arange(n_points, dtype=int) % n_clusters

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.probabilities_ = probabilities

        def fit_predict(self, coords):
            return labels.copy()

    monkeypatch.setattr(data_handling, "HDBSCAN", FakeHDBSCAN)

    import sklearn.cluster as sk_cluster

    seen_shapes = []

    class FakeAgglomerativeClustering:
        def __init__(self, n_clusters=None, distance_threshold=None, linkage=None):
            self.distance_threshold = distance_threshold
            self.linkage = linkage

        def fit_predict(self, pts):
            seen_shapes.append(pts.shape)
            # Merge all cluster centroids into one macro-cluster
            return np.zeros(len(pts), dtype=int)

    monkeypatch.setattr(sk_cluster, "AgglomerativeClustering", FakeAgglomerativeClustering)

    # 6000 points in 2D + dummy z
    x = np.linspace(0.0, 1000.0, n_points)
    y = np.linspace(0.0, 500.0, n_points)
    z = np.zeros(n_points, dtype=float)
    smlm_data = np.column_stack([x, y, z])

    out = data_handling.isolate_individual_npcs(
        smlm_data,
        min_cluster_size=2,
        min_npc_points=1,
        use_xy_only=True,
        perform_geometric_merging=True,
    )

    assert len(seen_shapes) == 1
    # Regression condition: agglomeration runs on cluster centroids, not raw points.
    assert seen_shapes[0] == (n_clusters, 2)
    assert np.all(out["labels"] == 0)
