import numpy as np
import pytest
from smlm_score.src.utility.data_handling import isolate_individual_npcs

@pytest.mark.unit
def test_stage4_empty_data_returns_stable_dict():
    """Verify that empty input doesn't crash and returns expected empty structures."""
    empty_data = np.zeros((0, 3))
    out = isolate_individual_npcs(empty_data)
    
    assert out['n_npcs'] == 0
    assert len(out['labels']) == 0
    assert out['npc_info'] == []

@pytest.mark.unit
def test_stage4_min_npc_points_filtering():
    """Verify that clusters below min_npc_points are excluded from npc_info but kept in labels."""
    # Create a simple dataset with two clusters: one small (10 pts), one large (60 pts)
    cluster_small = np.random.normal(0, 5, (10, 3))
    cluster_large = np.random.normal(200, 5, (60, 3))
    data = np.vstack([cluster_small, cluster_large])
    
    # We set min_cluster_size=5 so both are detected as clusters
    # But min_npc_points=50 so only the large one counts as an NPC
    out = isolate_individual_npcs(
        data, 
        min_cluster_size=5, 
        min_npc_points=50, 
        perform_geometric_merging=False
    )
    
    # Both should have labels (not -1)
    unique_labels = set(out['labels']) - {-1}
    assert len(unique_labels) == 2
    
    # But n_npcs and npc_info should only reflect the large one
    assert out['n_npcs'] == 1
    assert out['npc_info'][0]['n_points'] == 60

@pytest.mark.unit
def test_stage4_geometric_boundary_separation():
    """
    Verify that two synthetic NPCs separated by >140nm stay separate,
    while two fragments closer together merge.
    """
    # NPC 1: Two arcs very close (should merge)
    arc_a = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]])
    arc_b = np.array([[30, 0, 0], [40, 0, 0], [50, 0, 0]]) # Dist is 10nm
    
    # NPC 2: Far away (should stay separate)
    far_npc = np.array([[300, 0, 0], [310, 0, 0], [320, 0, 0]]) # Dist is 250nm
    
    data = np.vstack([arc_a, arc_b, far_npc])
    
    # Run with merging
    out = isolate_individual_npcs(
        data, 
        min_cluster_size=2, 
        min_npc_points=2, 
        perform_geometric_merging=True
    )
    
    # We expect 2 NPCs: One merged (arc_a+arc_b) and one separate (far_npc)
    assert out['n_npcs'] == 2
    
    # Check that labels for a and b are the same
    assert out['labels'][0] == out['labels'][3]
    # Check that labels for a and far are different
    assert out['labels'][0] != out['labels'][6]
