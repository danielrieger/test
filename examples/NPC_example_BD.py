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
    align_npc_cluster_pca,
    get_held_out_complement
)
from smlm_score.imp_modeling.scoring.gmm_score import test_gmm_components
from smlm_score.imp_modeling.brownian_dynamics.simulation_setup import run_brownian_dynamics_simulation
from smlm_score.imp_modeling.simulation.mcmc_sampler import run_bayesian_sampling
from smlm_score.imp_modeling.simulation.frequentist_optimizer import run_frequentist_optimization
from smlm_score.validation.validation import run_full_validation

EXAMPLE_DIR = Path(__file__).parent

def validate_config(config):
    """Runtime validation of configuration parameters."""
    print("--- Validating Configuration ---")
    f_type = config["filtering"]["type"]
    w_mode = config["optimization"]["bayesian"]["score_weight"]
    c_method = config["clustering"].get("method", "hdbscan")
    
    if f_type == "none" and w_mode == "auto":
        print("  WARNING: filtering.type='none' with score_weight='auto' may produce")
        print("           very weak sampling constraints on large clusters. The weight")
        print("           will be capped at 0.005 to prevent meaningless sampling.")
        print("           See pipeline_config_template.jsonc for details.")
        
    if f_type == "none" and c_method == "eman2":
        print("  INFO: With eman2 clustering on unfiltered data, noise separation")
        print("        tests will be skipped (all boxes produce valid NPCs).")
    print("--------------------------------")

def load_config(config_file="pipeline_config.json"):
    """Load and return the pipeline configuration."""
    config_path = EXAMPLE_DIR / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    validate_config(config)
    
    # --- Startup Diagnostics (Restored) ---
    st = config["execution"]["test_scoring_types"]
    mode = config["optimization"]["mode"]
    cm = config["clustering"].get("method", "hdbscan")
    
    if mode == "frequentist" and config["optimization"]["frequentist"]["scoring_type"] == "GMM":
        print("Warning: frequentist optimization does not support GMM scoring. CG will be skipped.")
    if mode == "brownian" and config["optimization"]["brownian"]["scoring_type"] == "GMM":
        print("Warning: Brownian dynamics does not support GMM scoring. BD will be skipped.")
        
    print(f"Scoring types to test: {st}")
    print(f"Optimization mode:     {mode}")
    print(f"Clustering method:     {cm}")
    
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
        align = align_npc_cluster_pca(pts, debug=False)
        aligned_pts = align['aligned_data']
        rot = align['rotation']
        
        # Consistent model alignment for scoring
        model_aligned = np.dot(model_centered, rot.T)
        
        if cid == target_id:
            target_n_points = n_pts
            cross_val_data = {
                'cluster_points': aligned_pts,
                'model_coords': model_aligned,
                'model': m,
                'avs': avs
            }

        for ST in test_types:
            sr = None
            if ST == "Tree":
                sr = ScoringRestraintWrapper(m, avs, kdtree_obj=KDTree(aligned_pts), dataxyz=aligned_pts, var=vars_, type=ST, model_coords_override=model_aligned, searchradius=50.0)
            elif ST == "GMM" and len(aligned_pts) > 2:
                _, gmm, mu, cov, w = test_gmm_components(aligned_pts)
                sr = ScoringRestraintWrapper(m, avs, gmm_sel_components=gmm.n_components, gmm_sel_mean=mu, gmm_sel_cov=cov, gmm_sel_weight=w, type=ST, model_coords_override=model_aligned)
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
                    already_optimized = trigger_opt(m, pdb_h, avs, sr, cid, ST, config, n_pts)

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
        run_bayesian_sampling(m, pdb_h, avs, sr, f"bayesian_cluster_{cid}", opt["bayesian"]["number_of_frames"], opt["bayesian"]["monte_carlo_steps"], w, opt["bayesian"]["max_rb_trans"], opt["bayesian"]["max_rb_rot"])
        return True
    
    if mode == "frequentist" and ST == opt["frequentist"]["scoring_type"]:
        if ST != "GMM":
            print(f"  [Frequentist] Triggering CG optimization for cluster {cid}...")
            run_frequentist_optimization(m, pdb_h, avs, sr, f"frequentist_cluster_{cid}", opt["frequentist"]["max_cg_steps"])
            return True
        print("  [Frequentist] Skipping: GMM not supported for CG.")
        
    if mode == "brownian" and ST == opt["brownian"]["scoring_type"]:
        if ST != "GMM":
            print(f"  [Brownian] Triggering BD simulation for cluster {cid}...")
            b = opt["brownian"]
            run_brownian_dynamics_simulation(m, pdb_h, avs, sr, f"brownian_cluster_{cid}", b["temperature_k"], b["max_time_step_fs"], b["number_of_bd_steps"], b["rmf_save_interval"])
            return True
        print("  [Brownian] Skipping: GMM not supported for BD.")
    return False

def run_held_out(m, avs, df, cuts, target_s, target_n, test_types, model_baseline):
    """Run cross-validation against held-out data chunks."""
    ho_xyz, ho_var = get_held_out_complement(df, x_cut=cuts['x'], y_cut=cuts['y'], z_cut=cuts['z'], n_samples=200)
    results = {}
    if target_s is not None and len(ho_xyz) > 0:
        for ST in test_types:
            chunks, chunk_n = [], []
            n_chunks = 5
            c_size = len(ho_xyz) // n_chunks
            for i in range(n_chunks):
                c_xyz = ho_xyz[i*c_size:(i+1)*c_size]
                c_var = ho_var[i*c_size:(i+1)*c_size]
                if len(c_xyz) == 0: continue
                c_xyz_c = c_xyz - c_xyz.mean(axis=0)
                try:
                    if ST == "Tree": sr = ScoringRestraintWrapper(m, avs, kdtree_obj=KDTree(c_xyz_c), dataxyz=c_xyz_c, var=c_var, type=ST, model_coords_override=model_baseline)
                    elif ST == "GMM":
                        _, gmm, mu, cov, w = test_gmm_components(c_xyz_c.astype(np.float64), reg_covar=1e-4)
                        sr = ScoringRestraintWrapper(m, avs, gmm_sel_components=gmm.n_components, gmm_sel_mean=mu, gmm_sel_cov=cov, gmm_sel_weight=w, type=ST, model_coords_override=model_baseline)
                    elif ST == "Distance":
                        cl = [np.eye(3)*max(v,1e-9) for v in c_var]
                        sr = ScoringRestraintWrapper(m, avs, dataxyz=c_xyz_c, var=cl, type=ST, model_coords_override=model_baseline)
                    chunks.append(sr.evaluate())
                    chunk_n.append(len(c_xyz))
                except Exception as e: print(f"Held-out failed for {ST}: {e}")
            if chunks:
                results[ST] = {'valid_score': target_s, 'valid_n_points': target_n, 'held_out_scores': chunks, 'held_out_n_points': chunk_n}
    return results

def main():
    print("--- SMLM-IMP Modeling Pipeline (Refactored) ---")
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
    
    # Clustering
    if config["clustering"]["method"] == "eman2":
        res = isolate_npcs_from_eman2_boxes(coords, EXAMPLE_DIR / config["clustering"]["eman2_boxes"], EXAMPLE_DIR / config["clustering"].get("pixel_map", "pixel_map.json"))
    else:
        res = isolate_individual_npcs(coords, min_cluster_size=config["clustering"]["min_cluster_size"])
        
    # Main Analysis
    scores, cv_data, t_s, t_n, m_base = run_evaluation(m, pdb_h, avs, res, coords, vars_, config)

    # Validation
    ho_results = run_held_out(m, avs, df, cuts, t_s, t_n, config["execution"]["test_scoring_types"], m_base)
    
    print("\n=== CLUSTER SCORING SUMMARY ===")
    print("ID   | Type  | N_pts |  Tree Score       | GMM Score      | Distance Score")
    for cid, s in sorted(scores.items()):
        print(f"{cid:4} | {s['type']:5} | {s['n_points']:5} | {s.get('Tree',0):16.2f} | {s.get('GMM',0):14.2f} | {s.get('Distance',0):14.2f}")

    run_full_validation(cluster_scores=scores, held_out_results=ho_results, scoring_types=["Tree", "GMM"], cross_val_data=cv_data)
    
    if config["optimization"]["mode"] != "brownian":
        print("\nNote: Brownian Dynamics simulation is currently DISABLED for screening mode.")

if __name__ == "__main__":
    main()
