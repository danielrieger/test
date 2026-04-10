import IMP
import IMP.core
import IMP.atom
import numpy as np
from sklearn.neighbors import KDTree
import json
import os
import shutil

# Import your project modules
from smlm_score.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper
from smlm_score.utility.input import read_parameters_from_json
from smlm_score.utility.data_handling import (
    flexible_filter_smlm_data,
    compute_av,
    isolate_individual_npcs,
    align_npc_cluster_pca,
    get_held_out_complement
)
from smlm_score.utility.input import read_experimental_data
from smlm_score.imp_modeling.scoring.gmm_score import test_gmm_components
from smlm_score.imp_modeling.brownian_dynamics.simulation_setup import run_brownian_dynamics_simulation
from smlm_score.imp_modeling.simulation.mcmc_sampler import run_bayesian_sampling
from smlm_score.imp_modeling.simulation.frequentist_optimizer import run_frequentist_optimization
from smlm_score.validation.validation import run_full_validation

# --- 1. Load Pipeline Configuration ---
config_path = "pipeline_config.json"
if not os.path.exists(config_path):
    print(f"Error: {config_path} not found. Please ensure it exists in the runtime directory.")
    exit(1)

with open(config_path, "r") as f:
    config = json.load(f)

smlm_data_path = config["paths"]["smlm_data"]
pdb_data_path = config["paths"]["pdb_data"]
av_parameters_path = config["paths"]["av_parameters"]

TEST_SCORING_TYPES = config["execution"]["test_scoring_types"]
TARGET_CLUSTER_ID = config["execution"]["target_cluster_id"]

PERFORM_GEOMETRIC_MERGING = config["clustering"]["perform_geometric_merging"]
MIN_CLUSTER_SIZE = config["clustering"]["min_cluster_size"]
MIN_NPC_POINTS = config["clustering"]["min_npc_points"]

OPTIMIZATION_MODE = config["optimization"]["mode"]
RUN_BAYESIAN_SAMPLING = config["optimization"]["bayesian"]["run_sampling"]
SAMPLING_SCORING_TYPE = config["optimization"]["bayesian"]["scoring_type"]
BAYESIAN_FRAMES = config["optimization"]["bayesian"]["number_of_frames"]
BAYESIAN_STEPS = config["optimization"]["bayesian"]["monte_carlo_steps"]

FREQUENTIST_SCORING_TYPE = config["optimization"]["frequentist"]["scoring_type"]
FREQUENTIST_MAX_STEPS = config["optimization"]["frequentist"]["max_cg_steps"]

BROWNIAN_SCORING_TYPE = config["optimization"]["brownian"]["scoring_type"]
BROWNIAN_TEMP = config["optimization"]["brownian"]["temperature_k"]
BROWNIAN_DT = config["optimization"]["brownian"]["max_time_step_fs"]
BROWNIAN_STEPS = config["optimization"]["brownian"]["number_of_bd_steps"]
BROWNIAN_SAVE_INTERVAL = config["optimization"]["brownian"]["rmf_save_interval"]

if OPTIMIZATION_MODE == "frequentist" and FREQUENTIST_SCORING_TYPE == "GMM":
    print(
        "Warning: frequentist optimization does not support GMM scoring. "
        "Optimization will be skipped unless you choose 'Tree' or 'Distance'."
    )

if OPTIMIZATION_MODE == "brownian" and BROWNIAN_SCORING_TYPE == "GMM":
    print(
        "Warning: Brownian dynamics does not support GMM scoring. "
        "Optimization will be skipped unless you choose 'Tree' or 'Distance'."
    )

print("--- SMLM-IMP Modeling Pipeline ---")
print(f"Scoring types to test: {TEST_SCORING_TYPES}")
print(f"Optimization mode:     {OPTIMIZATION_MODE}")

# --- 2. Gather and Process Data ---
parameters = read_parameters_from_json(av_parameters_path)
raw_smlm_data_df = read_experimental_data(smlm_data_path)
if raw_smlm_data_df is None:
    print("Error: Failed to load SMLM data. Exiting.")
    exit(1)

# Dynamic Filtering based on Config
filtering_cfg = config.get("filtering", {})
f_type = filtering_cfg.get("type", "none")
f_params = filtering_cfg.get(f_type, {}) if f_type != "none" else {}

x_range = tuple(f_params.get("x_range")) if f_params.get("x_range") else None
y_range = tuple(f_params.get("y_range")) if f_params.get("y_range") else None
z_range = tuple(f_params.get("z_range")) if f_params.get("z_range") else None
perc = filtering_cfg.get("random", {}).get("size_percentage")

print(f"Applying SMLM data filter: {f_type}...")
smlm_coordinates, smlm_variances, smlm_coordinates_for_tree, kdtree_object, applied_cuts = flexible_filter_smlm_data(
    raw_smlm_data_df,
    filter_type=f_type,
    x_cut=x_range,
    y_cut=y_range,
    z_cut=z_range,
    percentage=perc,
    fill_z_value=0.0,
    return_tree=True
)

print(f"\nCreating IMP Model and computing AVs...")
avs, m, pdb_hierarchy = compute_av(pdb_data_path, parameters)
print(f"Computed {len(avs)} AVs.")

model_coords_nm = np.array(
    [np.array(IMP.core.XYZ(av).get_coordinates()) * 0.1 for av in avs]
)
model_centered_baseline = model_coords_nm - model_coords_nm.mean(axis=0)

# --- 3. Iterative Cluster Analysis ---
print("\n=== STARTING ITERATIVE CLUSTER ANALYSIS ===")
npc_results = isolate_individual_npcs(
    smlm_coordinates_for_tree,
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_npc_points=MIN_NPC_POINTS,
    perform_geometric_merging=PERFORM_GEOMETRIC_MERGING,
    debug=True,
)
cluster_labels = npc_results['labels']
cluster_info = npc_results['all_cluster_info']
valid_clusters = npc_results['npc_info']
noise_cluster_list = [c for c in cluster_info if 10 <= c['n_points'] < 100]

if TARGET_CLUSTER_ID is None and valid_clusters:
    target_cluster_data = max(valid_clusters, key=lambda c: c['n_points'])
    TARGET_CLUSTER_ID = target_cluster_data['cluster_id']
    print(f"Auto-selected cluster {TARGET_CLUSTER_ID} for optimization.")
else:
    target_cluster_data = next((c for c in valid_clusters if c['cluster_id'] == TARGET_CLUSTER_ID), None)

clusters_to_evaluate = []
if target_cluster_data:
    clusters_to_evaluate.append(target_cluster_data)
clusters_to_evaluate.extend(noise_cluster_list[:3])

cluster_scores = {}

for current_cluster in clusters_to_evaluate:
    cluster_idx = current_cluster['cluster_id']
    n_points = current_cluster['n_points']
    cluster_type = "Valid" if n_points >= 100 else "Noise"
    
    # --- Snapshot Model Points to allow Reset ---
    pre_cluster_coords = [IMP.core.XYZ(av).get_coordinates() for av in avs]
    
    print(f"\n--- Analyzing Cluster {cluster_idx} ({n_points} points) ---")
    cluster_mask = (cluster_labels == cluster_idx)
    cluster_points = smlm_coordinates_for_tree[cluster_mask]
    cluster_variances = smlm_variances[cluster_mask] if smlm_variances is not None else None
    
    alignment_results = align_npc_cluster_pca(cluster_points, debug=False)
    aligned_cluster_points = alignment_results['aligned_data']
    rotation_matrix = alignment_results['rotation']
    
    # Align model for baseline (Stage 4)
    model_coords_nm = np.array([np.array(IMP.core.XYZ(av).get_coordinates()) * 0.1 for av in avs])
    model_centered = model_coords_nm - model_coords_nm.mean(axis=0)
    model_aligned = np.dot(model_centered, rotation_matrix.T)
    
    for SCORING_TYPE in TEST_SCORING_TYPES:
        sr_wrapper = None
        if SCORING_TYPE == "Tree":
            sr_wrapper = ScoringRestraintWrapper(
                m, avs, kdtree_obj=KDTree(aligned_cluster_points),
                dataxyz=aligned_cluster_points, var=cluster_variances,
                searchradius=50.0, model_coords_override=model_aligned,
                type=SCORING_TYPE
            )
        elif SCORING_TYPE == "GMM":
            if len(aligned_cluster_points) > 2:
                _, gmm_obj, gmm_mean, gmm_cov, gmm_w = test_gmm_components(aligned_cluster_points)
                sr_wrapper = ScoringRestraintWrapper(
                    m, avs, gmm_sel_components=gmm_obj.n_components,
                    gmm_sel_mean=gmm_mean, gmm_sel_cov=gmm_cov,
                    gmm_sel_weight=gmm_w, model_coords_override=model_aligned,
                    type=SCORING_TYPE
                )
        elif SCORING_TYPE == "Distance":
            cov_list = [np.eye(3)*max(v,1e-9) for v in cluster_variances] if cluster_variances is not None else [np.eye(3)]*len(aligned_cluster_points)
            sr_wrapper = ScoringRestraintWrapper(
                m, avs, dataxyz=aligned_cluster_points, var=cov_list,
                model_coords_override=model_aligned, type=SCORING_TYPE
            )
        
        if sr_wrapper:
            score = sr_wrapper.evaluate()
            print(f"  [{SCORING_TYPE}] Score: {score:.2f}")
            if cluster_idx not in cluster_scores:
                cluster_scores[cluster_idx] = {'type': cluster_type, 'n_points': n_points}
            cluster_scores[cluster_idx][SCORING_TYPE] = score
            
            # --- Optimization Trigger ---
            should_optimize = (cluster_idx == TARGET_CLUSTER_ID and cluster_type == "Valid")
            if should_optimize:
                if OPTIMIZATION_MODE == "bayesian" and SCORING_TYPE == SAMPLING_SCORING_TYPE:
                    run_bayesian_sampling(m, pdb_hierarchy, avs, sr_wrapper, f"bayesian_cluster_{cluster_idx}", BAYESIAN_FRAMES, BAYESIAN_STEPS)
                elif OPTIMIZATION_MODE == "frequentist" and SCORING_TYPE == FREQUENTIST_SCORING_TYPE:
                    if SCORING_TYPE == "GMM":
                        print("  [Frequentist] Skipping optimization: GMM scoring is not supported for CG.")
                    else:
                        run_frequentist_optimization(m, pdb_hierarchy, avs, sr_wrapper, f"frequentist_cluster_{cluster_idx}", FREQUENTIST_MAX_STEPS)
                elif OPTIMIZATION_MODE == "brownian" and SCORING_TYPE == BROWNIAN_SCORING_TYPE:
                    if SCORING_TYPE == "GMM":
                        print("  [Brownian] Skipping optimization: GMM scoring is not supported for BD.")
                    else:
                        run_brownian_dynamics_simulation(
                            model=m,
                            pdb_hierarchy=pdb_hierarchy,
                            avs=avs,
                            scoring_restraint_wrapper=sr_wrapper,
                            output_dir=f"brownian_cluster_{cluster_idx}",
                            temperature=BROWNIAN_TEMP,
                            max_time_step_fs=BROWNIAN_DT,
                            number_of_bd_steps=BROWNIAN_STEPS,
                            rmf_save_interval_frames=BROWNIAN_SAVE_INTERVAL,
                        )

    # Reset model state for next cluster
    for i, av in enumerate(avs):
        IMP.core.XYZ(av).set_coordinates(pre_cluster_coords[i])

# --- 4. Summary & Validation ---
print("\n=== CLUSTER SCORING SUMMARY ===")
print("ID   | Type  | N_pts |  Tree Score       | GMM Score      | Distance Score")
for cid, s in sorted(cluster_scores.items()):
    print(f"{cid:4} | {s['type']:5} | {s['n_points']:5} | {s.get('Tree',0):16.2f} | {s.get('GMM',0):14.2f} | {s.get('Distance',0):14.2f}")

if OPTIMIZATION_MODE != "brownian":
    print("\nNote: Brownian Dynamics simulation is currently DISABLED for screening mode.")

# Held-out validation
held_out_xyz, held_out_var = get_held_out_complement(
    raw_smlm_data_df,
    x_cut=applied_cuts['x'],
    y_cut=applied_cuts['y'],
    z_cut=applied_cuts['z'],
    n_samples=200
)
held_out_results = {}
if len(held_out_xyz) > 0:
    for ST in TEST_SCORING_TYPES:
        chunk_scores = []
        chunk_n_points = []
        n_chunks = 5
        chunk_size = len(held_out_xyz) // n_chunks
        for i in range(n_chunks):
            c_xyz = held_out_xyz[i*chunk_size:(i+1)*chunk_size]
            c_var = held_out_var[i*chunk_size:(i+1)*chunk_size]
            c_xyz_c = c_xyz - c_xyz.mean(axis=0)
            if len(c_xyz) == 0:
                continue
            try:
                if ST == "Tree":
                    sr = ScoringRestraintWrapper(m, avs, kdtree_obj=KDTree(c_xyz_c), dataxyz=c_xyz_c, var=c_var, type=ST, model_coords_override=model_centered_baseline)
                elif ST == "GMM":
                    _, gmm, mu, cov, w = test_gmm_components(c_xyz_c.astype(np.float64), reg_covar=1e-4)
                    sr = ScoringRestraintWrapper(m, avs, gmm_sel_components=gmm.n_components, gmm_sel_mean=mu, gmm_sel_cov=cov, gmm_sel_weight=w, type=ST, model_coords_override=model_centered_baseline)
                elif ST == "Distance":
                    cl = [np.eye(3)*max(v,1e-9) for v in c_var]
                    sr = ScoringRestraintWrapper(m, avs, dataxyz=c_xyz_c, var=cl, type=ST, model_coords_override=model_centered_baseline)
                chunk_scores.append(sr.evaluate())
                chunk_n_points.append(len(c_xyz))
            except Exception as exc:
                print(f"Held-out validation chunk failed for {ST}: {exc}")
        
        target_s = cluster_scores.get(TARGET_CLUSTER_ID, {}).get(ST)
        target_n_points = cluster_scores.get(TARGET_CLUSTER_ID, {}).get('n_points')
        if target_s is not None and chunk_scores:
            held_out_results[ST] = {
                'valid_score': target_s,
                'valid_n_points': target_n_points,
                'held_out_scores': chunk_scores,
                'held_out_n_points': chunk_n_points,
            }

run_full_validation(cluster_scores, held_out_results if held_out_results else None, TEST_SCORING_TYPES)
