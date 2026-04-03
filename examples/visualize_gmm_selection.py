import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure imports work from repo root
_examples_dir = os.path.dirname(__file__)
_project_dir = os.path.abspath(os.path.join(_examples_dir, ".."))
_thesis_dir = os.path.abspath(os.path.join(_project_dir, ".."))
if _thesis_dir not in sys.path:
    sys.path.insert(0, _thesis_dir)

from sklearn.neighbors import KDTree
from smlm_score.src.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper
from smlm_score.src.imp_modeling.scoring.gmm_score import test_gmm_components
from smlm_score.src.utility.data_handling import (
    flexible_filter_smlm_data,
    isolate_individual_npcs,
    align_npc_cluster_pca,
    compute_av,
)
from smlm_score.src.utility.input import read_experimental_data, read_parameters_from_json
from smlm_score.src.utility.visualization import _PUB_BG, _PUB_TEXT, _PUB_BLUE, _PUB_ORANGE

def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_minmax(values):
    arr = np.array(values, dtype=float)
    if len(arr) == 0: return arr
    finite = np.isfinite(arr)
    if not np.any(finite): return np.full_like(arr, 0.5)
    vmin, vmax = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    if abs(vmax - vmin) < 1e-12: return np.full_like(arr, 0.5)
    return (arr - vmin) / (vmax - vmin)

def _get_top_clusters(coords, variances, labels, all_info, model, avs, n_top=1):
    rows = []
    # Only consider clusters with enough points for structural significance
    candidates = [c for c in all_info if c["n_points"] >= 150]
    print(f"Ranking {len(candidates)} candidates by structural quality...", flush=True)
    
    for candidate in candidates:
        cid = int(candidate["cluster_id"])
        mask = labels == cid
        pts = coords[mask]
        aligned = align_npc_cluster_pca(pts, debug=False)["aligned_data"]
        
        tree_wrapper = ScoringRestraintWrapper(
            model, avs, kdtree_obj=KDTree(aligned), dataxyz=aligned,
            var=variances[mask] if variances is not None else np.ones(len(pts)),
            type="Tree"
        )
        # Normalize score to prevent bias toward larger clusters
        score_norm = float(tree_wrapper.evaluate()) / float(len(pts)**2)
        rows.append({"cluster_id": cid, "score_norm": score_norm, "pts": pts, "aligned": aligned})
    
    if not rows:
        return None
    
    import pandas as pd
    df = pd.DataFrame(rows)
    df["quality"] = _safe_minmax(df["score_norm"].values)
    return df.sort_values("quality", ascending=False).head(n_top)

def main():
    config_path = os.path.join(_examples_dir, "pipeline_config.json")
    cfg = _load_config(config_path)
    
    # 1. Loading data
    print("Loading data and model for ranking...", flush=True)
    raw_df = read_experimental_data(os.path.join(_examples_dir, cfg["paths"]["smlm_data"]))
    coords, variances, _, _, _ = flexible_filter_smlm_data(
        raw_df, filter_type=cfg["filtering"]["type"], 
        percentage=cfg["filtering"]["random"]["size_percentage"]
    )
    npc_res = isolate_individual_npcs(coords, min_npc_points=120)
    
    # 2. Loading model for ranking
    params = read_parameters_from_json(os.path.join(_examples_dir, cfg["paths"]["av_parameters"]))
    avs, model, _ = compute_av(os.path.join(_examples_dir, cfg["paths"]["pdb_data"]), params)
    
    # 3. Ranking to find the 'Top 5 Highest Quality' NPCs
    top_df = _get_top_clusters(coords, variances, npc_res["labels"], npc_res["all_cluster_info"], model, avs, n_top=5)
    if top_df is None:
        print("ERROR: No NPC candidates qualified for ranking.")
        return
        
    print(f"\nPROCESSING TOP {len(top_df)} CLUSTERS...")
    
    for idx, best in top_df.iterrows():
        target_id = int(best["cluster_id"])
        cluster_pts = best["pts"]
        print(f"\n--- [Rank {idx+1}] NPC Cluster {target_id} (Quality: {best['quality']:.3f}) ---")
        
        # 4. Fitting GMM sequence (BIC)
        print(f"Fitting GMM sequence ({len(cluster_pts)} points)...", flush=True)
        d, gmm_sel, _, _, _ = test_gmm_components(cluster_pts, component_min=1, component_max=128, reg_covar=1e-3)
        
        # Visualisation 1: BIC Curve
        n_comp = d['n_components']
        bic = d['bic']
        aic = d['aic']
        best_n = n_comp[d['n']]
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=_PUB_BG)
        ax.set_facecolor(_PUB_BG)
        ax.plot(n_comp, bic, label="BIC", color=_PUB_ORANGE, marker='o', lw=2.5)
        ax.plot(n_comp, aic, label="AIC", color=_PUB_BLUE, marker='s', lw=1.5, ls='--')
        ax.axvline(best_n, color='red', alpha=0.5, ls=':', label=f'Optimal K={best_n}')
        ax.scatter(best_n, bic[d['n']], color='red', s=100, zorder=5)
        ax.set_xscale('log', base=2); ax.set_xlabel("Number of GMM Components (K)", color=_PUB_TEXT)
        ax.set_ylabel("Metric Value", color=_PUB_TEXT)
        ax.set_title(f"NPC {target_id} (Rank {idx+1}) - GMM Model Selection", color=_PUB_TEXT, fontsize=14)
        ax.tick_params(colors=_PUB_TEXT, which='both'); ax.grid(True, which='both', alpha=0.2)
        ax.legend(facecolor=_PUB_BG, edgecolor='#d1d1d1', labelcolor=_PUB_TEXT)
        for spine in ax.spines.values(): spine.set_color('#d1d1d1')
        
        fig.tight_layout()
        out_path = os.path.join(_examples_dir, "figures", "qc", f"gmm_bic_selection_rank{idx+1}_id{target_id}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=300, facecolor=_PUB_BG)
        plt.close(fig)
        print(f"Saved BIC Plot: {out_path}")

        # Visualisation 2: GMM Overlay
        from matplotlib.patches import Ellipse
        def draw_ellipse(position, covariance, ax=None, **kwargs):
            ax = ax or plt.gca(); cov2d = covariance[0:2, 0:2]
            v, w = np.linalg.eigh(cov2d); u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0]); angle = 180 * angle / np.pi
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ellipse = Ellipse(position[0:2], v[0], v[1], angle=angle, **kwargs)
            ax.add_patch(ellipse)

        fig2, ax2 = plt.subplots(figsize=(8, 8), facecolor=_PUB_BG)
        ax2.set_facecolor(_PUB_BG)
        ax2.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=12, color=_PUB_TEXT, alpha=0.25)
        colors = plt.cm.plasma(np.linspace(0, 0.8, best_n))
        for i in range(best_n):
            mean = gmm_sel.means_[i]; cov = gmm_sel.covariances_[i]; weight = gmm_sel.weights_[i]
            ax2.scatter(mean[0], mean[1], color=colors[i], marker='+', s=80, zorder=5)
            draw_ellipse(mean, cov, ax=ax2, edgecolor=colors[i], fc='none', lw=2, alpha=0.8)

        ax2.set_xlabel("X (nm)", color=_PUB_TEXT); ax2.set_ylabel("Y (nm)", color=_PUB_TEXT)
        ax2.set_title(f"GMM Overlay (K={best_n}) — Rank {idx+1} NPC {target_id}", color=_PUB_TEXT, fontsize=14)
        ax2.set_aspect('equal')
        for spine in ax2.spines.values(): spine.set_color('#d1d1d1')
        
        fig2.tight_layout()
        overlay_path = os.path.join(_examples_dir, "figures", "qc", f"gmm_cluster_overlay_rank{idx+1}_id{target_id}.png")
        fig2.savefig(overlay_path, dpi=300, facecolor=_PUB_BG)
        plt.close(fig2)
        print(f"Saved Overlay Plot: {overlay_path}")

if __name__ == "__main__":
    main()
