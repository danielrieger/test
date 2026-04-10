import numpy as np
import pytest
from smlm_score.utility.data_handling import align_npc_cluster_pca

@pytest.mark.unit
def test_stage5_alignment_inverse_transform_integrity():
    """Verify that applying the inverse rotation/translation recovers the original data."""
    # Create a random tilted cloud
    np.random.seed(42)
    original_pts = np.random.rand(100, 3) * 100
    
    out = align_npc_cluster_pca(original_pts)
    aligned = out['aligned_data']
    R = out['rotation']
    translation = out['translation'] # This was -centroid
    
    # Aligned = (Original + translation) @ R.T
    # So: Original = (Aligned @ R) - translation
    recovered = (aligned @ R) - translation
    
    assert np.allclose(original_pts, recovered, atol=1e-10)

@pytest.mark.unit
def test_stage5_alignment_2d_stability():
    """Verify that PCA doesn't fail or return NaNs when data is already perfectly flat (z=0)."""
    pts_2d = np.array([
        [10, 10, 0],
        [20, 10, 0],
        [15, 20, 0]
    ], dtype=float)
    
    out = align_npc_cluster_pca(pts_2d)
    
    assert not np.any(np.isnan(out['aligned_data']))
    assert not np.any(np.isnan(out['rotation']))
    # Should stay centered
    assert np.allclose(out['aligned_data'].mean(axis=0), [0,0,0], atol=1e-10)

@pytest.mark.unit
def test_stage5_alignment_minimal_points():
    """Verify behavior with exactly 3 points (minimum for a plane)."""
    pts = np.array([[0,0,0], [1,0,0], [0,1,1]], dtype=float)
    out = align_npc_cluster_pca(pts)
    
    assert out['aligned_data'].shape == (3, 3)
    # Normality check for rotation
    R = out['rotation']
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
