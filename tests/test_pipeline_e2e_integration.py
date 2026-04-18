import os
import shutil
import uuid
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from smlm_score.utility.input import read_experimental_data
from smlm_score.utility.data_handling import (
    flexible_filter_smlm_data,
    isolate_individual_npcs,
    align_npc_cluster_pca,
    get_held_out_complement
)
from smlm_score.validation.validation import run_full_validation


@pytest.mark.integration
def test_pipeline_e2e_top_to_bottom_execution():
    """
    End-to-End test that simulates a user's workflow:
    1. Synthetic data generation (a ring of points).
    2. CSV Writing & Reading.
    3. ROI Filtering.
    4. NPC Isolation (Clustering).
    5. PCA Alignment.
    6. Mock Validation Logic.
    """
    tmp_dir = Path("smlm_score") / "tests" / f".tmp_e2e_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / "synthetic_npc.csv"

    try:
        # --- Stage 1: Generate Synthetic Data ---
        # A single ring at (500, 500) with radius 60nm
        rng = np.random.RandomState(42)  # Fixed seed for deterministic test
        n_pts = 200
        theta = np.linspace(0, 2 * np.pi, n_pts)
        r = 60.0
        x = 500.0 + r * np.cos(theta) + rng.normal(0, 2, n_pts)
        y = 500.0 + r * np.sin(theta) + rng.normal(0, 2, n_pts)
        
        df = pd.DataFrame({
            "x [nm]": x,
            "y [nm]": y,
            "Amplitude_0_0": np.ones(n_pts) * 10.0
        })
        df.to_csv(csv_path, index=False)

        # --- Stage 2: Load and Filter ---
        loaded_df = read_experimental_data(str(csv_path))
        # Ensure x/y/z are the correct types as per flexible_filter_smlm_data
        coords, sigma, _, _, _ = flexible_filter_smlm_data(
            loaded_df, 
            filter_type='cut', 
            x_cut=(400, 600), 
            y_cut=(400, 600)
        )
        assert coords.shape[0] > 150 # Most points should pass
        assert coords.dtype == np.float32

        # --- Stage 3: NPC Isolation (Clustering) ---
        # We expect 1 cluster when merging is on
        clustering_res = isolate_individual_npcs(
            coords, 
            min_cluster_size=15, 
            perform_geometric_merging=True,
            debug=False
        )
        assert clustering_res['n_npcs'] == 1
        
        labels = clustering_res['labels']
        cluster_id = clustering_res['npc_info'][0]['cluster_id']
        npc_coords = coords[labels == cluster_id]

        # --- Stage 4: PCA Alignment ---
        alignment_res = align_npc_cluster_pca(npc_coords)
        aligned_data = alignment_res['aligned_data']
        
        # Verify centering (relaxed tolerance for float32 noise floor)
        mean_aligned = np.mean(aligned_data, axis=0)
        assert np.allclose(mean_aligned, 0.0, atol=1e-3)
        
        # Verify Z-axis alignment (standard deviation should be minimal for a flat ring)
        assert np.std(aligned_data[:, 2]) < 1e-10

        # --- Stage 5: Validation Logic Flow ---
        # Mock scores for validation
        cluster_scores = {
            cluster_id: {
                "type": "Valid",
                "n_points": len(npc_coords),
                "Tree": -10.0,
                "GMM": -20.0,
                "Distance": -30.0
            },
            -1: { # Mock noise
                "type": "Noise",
                "n_points": 50,
                "Tree": -500.0,
                "GMM": -300.0,
                "Distance": -400.0
            }
        }
        
        # Mock held-out data results
        held_out_results = {
            "Tree": {
                "valid_score": -10.0,
                "valid_n_points": len(npc_coords),
                "held_out_scores": [0.0, 0.0], # Passing mock
                "held_out_n_points": [int(0.2 * len(npc_coords))] * 2
            }
        }
        
        val_results = run_full_validation(
            cluster_scores, 
            held_out_results, 
            scoring_types=["Tree", "GMM", "Distance"]
        )
        
        # Verify that validation tests were generated
        test_names = [r.test_name for r in val_results]
        assert "Separation_Distance" in test_names
        assert "HeldOut_Tree" in test_names
        
        # All mock tests should pass
        assert all(r.passed for r in val_results)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
