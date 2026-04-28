import IMP
import IMP.core
import IMP.atom
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import json
import os
import shutil
from pathlib import Path

# Import project modules
from smlm_score.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper
from smlm_score.utility.input import read_parameters_from_json, read_experimental_data
from smlm_score.utility.data_handling import (
    flexible_filter_smlm_data,
    compute_av,
    isolate_individual_npcs,
    isolate_npcs_from_eman2_boxes,
    align_npc_cluster
)
from smlm_score.imp_modeling.scoring.gmm_score import test_gmm_components
from smlm_score.imp_modeling.brownian_dynamics.simulation_setup import run_brownian_dynamics_simulation
from smlm_score.imp_modeling.simulation.mcmc_sampler import run_bayesian_sampling
from smlm_score.imp_modeling.simulation.frequentist_optimizer import run_frequentist_optimization
from smlm_score.validation.validation import run_full_validation

EXAMPLE_DIR = Path(__file__).parent

def load_config(config_file="pipeline_config.json"):
    """Load and return the pipeline configuration."""
    config_path = EXAMPLE_DIR / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def setup_system(pdb_path, parameters):
    """Initialize the IMP model and compute accessible volumes."""
    avs, m, pdb_hierarchy = compute_av(pdb_path, parameters)
    return m, pdb_hierarchy, avs

def run_evaluation(m, pdb_h, avs, res, coords, variances, config):
    """Main orchestration of cluster scoring and optimization."""
    cluster_scores = {}
    cross_val_data = None
    already_optimized = False
    target_s = None
    target_n_points = None
    
    target_id = config["execution"]["target_cluster_id"]
    test_types = config["execution"]["test_scoring_types"]
    sampling_type = config["optimization"]["bayesian"]["scoring_type"]
    
    # Select clusters for evaluation
    valid_info = res.get('npc_info', [])
    all_info = res.get('all_cluster_info', [])
    min_npc = config["clustering"]["min_npc_points"]
    
    # 1. Restore strict Noise filtering (10 <= n < MIN_NPC_POINTS)
    if config["clustering"].get("method") == "eman2":
        noise_info = []
    else:
        noise_info = [c for c in all_info if 10 <= c['n_points'] < min_npc]
    
    # 2. Restore Auto-Select Logic
    if target_id is None or target_id == "null":
        if valid_info:
            target_cluster = max(valid_info, key=lambda c: c['n_points'])
            target_id = target_cluster['cluster_id']
            print(f"Auto-selected largest cluster {target_id} for optimization.")
        else:
            target_cluster = None
    elif target_id == "random":
        if valid_info:
            import random
            target_cluster = random.choice(valid_info)
            target_id = target_cluster['cluster_id']
            print(f"Auto-selected random cluster {target_id} for optimization.")
        else:
            target_cluster = None
    else:
        target_cluster = next((c for c in valid_info if c['cluster_id'] == target_id), None)
    
    to_eval = []
    if target_cluster: to_eval.append(target_cluster)
    to_eval.extend(noise_info[:3])

    labels = res['labels']
    
    # 1. Pre-calculate baseline model alignment for validation phase
    # (Fixes UnboundLocalError if no clusters are evaluated)
    model_nm_base = np.array([np.array(IMP.core.XYZ(av).get_coordinates()) * 0.1 for av in avs])
    model_centered = model_nm_base - model_nm_base.mean(axis=0)
    
    for current in to_eval:
        cid = current['cluster_id']
        n_pts = current['n_points']
        ctype = "Valid" if n_pts >= config["clustering"]["min_npc_points"] else "Noise"
        
        pre_coords = [IMP.core.XYZ(av).get_coordinates() for av in avs]
        print(f"\n--- Analyzing Cluster {cid} ({n_pts} points) ---")
        
        mask = (labels == cid)
        pts = coords[mask]
        
        if len(pts) < 3:
            print(f"  Skipping Cluster {cid}: insufficient unique points ({len(pts)}) due to overlapping box assignment.")
            continue
            
        vars_ = variances[mask] if variances is not None else None
        align = align_npc_cluster(pts, data_dim="auto", debug=False)
        aligned_pts = align['aligned_data']
        rot = align['rotation']
        
        # Consistent model alignment for scoring
        model_aligned = np.dot(model_centered, rot.T)
        if align.get('data_dim') == '2d':
            model_aligned = model_aligned.copy()
            model_aligned[:, 2] = 0.0
        
        if cid == target_id:
            target_n_points = n_pts
            cross_val_data = {
                'cluster_points': pts,
                'variances': vars_,
                'model_coords': model_centered,
                'model': m,
                'avs': avs,
                'data_dim': align.get('data_dim', 'auto')
            }

        for ST in test_types:
            sr = None
            if ST == "Tree":
                sr = ScoringRestraintWrapper(m, avs, kdtree_obj=KDTree(aligned_pts), dataxyz=aligned_pts, var=vars_, type=ST, model_coords_override=model_aligned, searchradius=50.0)
            elif ST == "GMM" and len(aligned_pts) > 2:
                _, gmm, mu, cov, w = test_gmm_components(aligned_pts)
                sr = ScoringRestraintWrapper(
                    m, avs, gmm_sel_components=gmm.n_components, 
                    gmm_sel_mean=mu, gmm_sel_cov=cov, gmm_sel_weight=w, type=ST, 
                    model_coords_override=model_aligned, 
                    model_variance=config.get("optimization", {}).get("model_variance", 8.0)
                )
            elif ST == "Distance":
                cl = [np.eye(3)*max(v, 1e-9) for v in vars_] if vars_ is not None else None
                sr = ScoringRestraintWrapper(m, avs, dataxyz=aligned_pts, var=cl, type=ST, model_coords_override=model_aligned)

            if sr:
                score = sr.evaluate()
                print(f"  [{ST}] Score: {score:.2f}")
                if cid not in cluster_scores:
                    cluster_scores[cid] = {'type': ctype, 'n_points': n_pts}
                cluster_scores[cid][ST] = score
                
                if cid == target_id and ST == sampling_type:
                    target_s = score

                if (cid == target_id or (target_id is None and ctype == "Valid")) and not already_optimized:
                    opt_result = trigger_opt(m, pdb_h, avs, sr, cid, ST, config, n_pts)
                    already_optimized = bool(opt_result)
                    if (
                        cid == target_id
                        and cross_val_data is not None
                        and isinstance(opt_result, dict)
                        and opt_result.get("best_model_coords_nm") is not None
                    ):
                        cross_val_data["model_coords"] = opt_result["best_model_coords_nm"]
                        cross_val_data["model_pose_source"] = "best_bayesian_sample"

        # Reset model state
        for i, av in enumerate(avs):
            IMP.core.XYZ(av).set_coordinates(pre_coords[i])
            
    return cluster_scores, cross_val_data, target_s, target_n_points, model_centered

def trigger_opt(m, pdb_h, avs, sr, cid, ST, config, n_pts):
    """Execute optimization based on selected mode and configuration flags."""
    opt = config["optimization"]
    mode = opt["mode"]
    
    if mode == "bayesian" and ST == opt["bayesian"]["scoring_type"] and opt["bayesian"]["run_sampling"]:
        print(f"  [Bayesian] Triggering sampling for cluster {cid}...")
        if opt["bayesian"]["score_weight"] == "auto":
            w = max(1.0 / n_pts, 0.005)
        else:
            w = float(opt["bayesian"].get("score_weight", 1.0))
        print(f"    Effective weight: {w:.6f} (n_pts={n_pts})")
        sampling_result = run_bayesian_sampling(m, pdb_h, avs, sr, f"bayesian_cluster_{cid}", opt["bayesian"]["number_of_frames"], opt["bayesian"]["monte_carlo_steps"], w, opt["bayesian"]["max_rb_trans"], opt["bayesian"]["max_rb_rot"])
        best_model_coords_nm = None
        if isinstance(sampling_result, dict) and sampling_result.get("best_av_coords") is not None:
            inner = getattr(sr, "scoring_restraint_instance", None)
            if inner is not None and getattr(inner, "model_coords_override", None) is not None:
                best_live_nm = np.asarray(sampling_result["best_av_coords"], dtype=np.float64) * inner.scaling
                best_model_coords_nm = (
                    np.asarray(inner.model_coords_override, dtype=np.float64)
                    + (best_live_nm - inner.reference_model_coords_nm)
                )
        return {
            "sampling_result": sampling_result,
            "best_model_coords_nm": best_model_coords_nm,
        }
    
    if mode == "frequentist" and ST == opt["frequentist"]["scoring_type"]:
        print(f"  [Frequentist] Triggering CG optimization for cluster {cid}...")
        run_frequentist_optimization(m, pdb_h, avs, sr, f"frequentist_cluster_{cid}", opt["frequentist"]["max_cg_steps"])
        return True
        
    if mode == "brownian" and ST == opt["brownian"]["scoring_type"]:
        print(f"  [Brownian] Triggering BD simulation for cluster {cid}...")
        b = opt["brownian"]
        run_brownian_dynamics_simulation(
            m, pdb_h, avs, sr, 
            output_dir=f"brownian_cluster_{cid}", 
            temperature=b["temperature_k"], 
            max_time_step_fs=b["max_time_step_fs"], 
            number_of_bd_steps=b["number_of_bd_steps"], 
            rmf_save_interval_frames=b["rmf_save_interval"]
        )
        return True
    return False


def main():
    IMP.set_log_level(IMP.SILENT)
    print("--- Initializing SMLM-IMP ---")
    config = load_config()
    
    # Paths & Parameters
    paths = config["paths"]
    smlm_data = EXAMPLE_DIR / paths["smlm_data"]
    pdb_data = EXAMPLE_DIR / paths["pdb_data"]
    params = read_parameters_from_json(EXAMPLE_DIR / paths["av_parameters"])
    if params is None:
        params = {"downsample_residues_per_bead": paths.get("downsample_residues_per_bead", 10)}
    else:
        params["downsample_residues_per_bead"] = paths.get("downsample_residues_per_bead")

    # Load SMLM
    df = read_experimental_data(smlm_data)
    f_type = config.get("filtering", {}).get("type", "none")
    f_params = config.get("filtering", {}).get(f_type, {})
    
    # Tuple-casting for robustness (Restored)
    x_r = tuple(f_params.get("x_range")) if f_params.get("x_range") else None
    y_r = tuple(f_params.get("y_range")) if f_params.get("y_range") else None
    z_r = tuple(f_params.get("z_range")) if f_params.get("z_range") else None
    
    print(f"Applying SMLM data filter: {f_type}...")
    coords, vars_, _, _, cuts = flexible_filter_smlm_data(
        df, filter_type=f_type, 
        x_cut=x_r, y_cut=y_r, z_cut=z_r,
        percentage=config.get("filtering", {}).get("random", {}).get("size_percentage"),
        fill_z_value=0.0, return_tree=True
    )
    
    # Global System Setup
    m, pdb_h, avs = setup_system(pdb_data, params)
    
    # --- Consolidated Metadata Banner ---
    st = config["execution"]["test_scoring_types"]
    mode = config["optimization"]["mode"]
    cm = config["clustering"].get("method", "hdbscan")
    bead_count = len(IMP.atom.get_leaves(pdb_h))
    
    print("\n" + "═" * 50)
    print("SMLM-IMP Modeling Pipeline")
    print("─" * 50)
    print(f"Model:        7N85 (8-fold NPC ring)")
    print(f"Resolution:   {params.get('downsample_residues_per_bead')} AAs/bead | {bead_count} beads | {len(avs)} AVs")
    print(f"Data:         {paths['smlm_data']}")
    print(f"Clustering:   {cm}")
    print(f"Optimization: {mode} ({st})")
    print("═" * 50)
    
    # Clustering
    if config["clustering"]["method"] == "eman2":
        res = isolate_npcs_from_eman2_boxes(coords, EXAMPLE_DIR / config["clustering"]["eman2_boxes"], EXAMPLE_DIR / config["clustering"].get("pixel_map", "pixel_map.json"))
    else:
        res = isolate_individual_npcs(coords, min_cluster_size=config["clustering"]["min_cluster_size"])
        
    # Main Analysis
    scores, cv_data, t_s, t_n, m_base = run_evaluation(m, pdb_h, avs, res, coords, vars_, config)

    # Validation (Cross-Validation only)
    target_id = config["execution"]["target_cluster_id"]
    
    print("\n=== CLUSTER SCORING SUMMARY ===")
    print("ID   | Type  | N_pts |  Tree Score       | GMM Score      | Distance Score")
    for cid, s in sorted(scores.items()):
        print(f"{cid:4} | {s['type']:5} | {s['n_points']:5} | {s.get('Tree',0):16.2f} | {s.get('GMM',0):14.2f} | {s.get('Distance',0):14.2f}")

    run_full_validation(cluster_scores=scores, held_out_results=None, scoring_types=["Tree", "GMM"], cross_val_data=cv_data, cluster_id=target_id)
    

if __name__ == "__main__":
    main()
