import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from smlm_score.src.utility import visualization as vis


@pytest.mark.integration
def test_visualization_density_plot_saves_file():
    tmp_dir = Path("smlm_score") / "tests" / f".tmp_vis_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        save_path = tmp_dir / "density.png"
        pts = np.random.RandomState(0).normal(size=(128, 3))
        fig = vis.plot_density_2d(pts, save_path=str(save_path))
        assert fig is not None
        assert save_path.exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.integration
def test_visualization_score_comparison_and_contour_save_files():
    tmp_dir = Path("smlm_score") / "tests" / f".tmp_vis_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        score_path = tmp_dir / "scores.png"
        contour_path = tmp_dir / "contour.png"

        cluster_scores = {
            0: {"type": "Valid", "n_points": 50, "Tree": -20.0, "GMM": -10.0, "Distance": -15.0},
            1: {"type": "Noise", "n_points": 50, "Tree": -200.0, "GMM": -120.0, "Distance": -130.0},
        }
        fig1 = vis.plot_score_comparison(cluster_scores, save_path=str(score_path))
        assert fig1 is not None
        assert score_path.exists()

        data_pts = np.random.RandomState(1).normal(size=(200, 3))
        av_pts = np.random.RandomState(2).normal(size=(8, 3))
        fig2 = vis.plot_density_contour(data_pts, av_pts, save_path=str(contour_path))
        assert fig2 is not None
        assert contour_path.exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
