import IMP
import IMP.core
import IMP.atom
import numpy as np
from sklearn.neighbors import KDTree  # Ensure this is imported if KDTree is used directly here

# Import your project modules
from smlm_score.src.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper
from smlm_score.src.utility.input import read_parameters_from_json
from smlm_score.src.utility.data_handling import flexible_filter_smlm_data, compute_av
from smlm_score.src.utility.input import read_experimental_data
from smlm_score.src.imp_modeling.scoring.gmm_score import test_gmm_components  # Only needed if SCORING_TYPE is GMM
from smlm_score.src.imp_modeling.brownian_dynamics.simulation_setup import run_brownian_dynamics_simulation

# Define paths
smlm_data_path = "ShareLoc_Data/data.csv"
pdb_data_path = "PDB_Data/7N85-assembly1.cif"
av_parameters_path = "av_parameter.json"

# --- Choose Scoring Type ---
SCORING_TYPE = "Tree"  # Options: "GMM", "Tree", "Distance"
# Change this to test different scoring functions

print("--- IMP Model Setup Script ---")
print(f"Selected SCORING_TYPE: {SCORING_TYPE}")

# --- Stage 1: Gather and Process Data ---
print(f"Reading parameters from: {av_parameters_path}")
parameters = read_parameters_from_json(av_parameters_path)
print("Parameters loaded.")

print(f"Reading SMLM data from: {smlm_data_path}")
raw_smlm_data_df = read_experimental_data(smlm_data_path)
if raw_smlm_data_df is None:
    print("Error: Failed to load SMLM data. Exiting.")
    exit(1)
print(f"Read {len(raw_smlm_data_df)} raw localizations.")

print("Filtering SMLM data...")
# Determine if KDTree is needed based on SCORING_TYPE
should_return_tree = (SCORING_TYPE == "Tree")

smlm_coordinates, smlm_variances, smlm_coordinates_for_tree, kdtree_object, _ = flexible_filter_smlm_data(
    raw_smlm_data_df,
    filter_type='cut',
    x_cut=(10000, 12000),
    y_cut=(0, 5000),
    fill_z_value=0.0,
    return_tree=should_return_tree
)
# smlm_coordinates can be used for GMM fitting or Distance scoring (if adapted)
# smlm_variances (sigma_values) can be used for Tree or Distance scoring

print(f"Filtered SMLM coordinates for general use have shape: {smlm_coordinates.shape}")
if kdtree_object:
    print(f"KDTree created from data with shape: {smlm_coordinates_for_tree.shape}")

# Initialize GMM parameters to None, they will be set if SCORING_TYPE is GMM
gmm_sel_components_val = None
gmm_sel_mean_val = None
gmm_sel_cov_val = None
gmm_sel_weight_val = None

if SCORING_TYPE == "GMM":
    print("Fitting GMM to SMLM data (as SCORING_TYPE is GMM)...")
    if not isinstance(smlm_coordinates, np.ndarray) or smlm_coordinates.ndim != 2:
        print(
            f"\nError: Data for GMM fitting must be a 2D NumPy array. Shape: {smlm_coordinates.shape if isinstance(smlm_coordinates, np.ndarray) else type(smlm_coordinates)}")
        exit(1)
    if smlm_coordinates.shape[0] < 2:
        print(f"\nError: Insufficient samples ({smlm_coordinates.shape[0]}) for GMM fitting. Need at least 2.")
        exit(1)

    # test_gmm_components returns: gmm_results_dict, gmm_selected_object, means, covariances, weights
    _, gmm_sel_obj, gmm_sel_mean_val, gmm_sel_cov_val, gmm_sel_weight_val = test_gmm_components(smlm_coordinates)
    gmm_sel_components_val = gmm_sel_obj.n_components
    print(f"Selected GMM with {gmm_sel_components_val} components based on BIC.")
else:
    print(f"Skipping GMM component fitting (SCORING_TYPE is {SCORING_TYPE}).")

# --- Stage 2: Define Representation and Scoring Function ---
print(f"Creating IMP Model and computing AVs from: {pdb_data_path}")
avs, m, pdb_hierarchy = compute_av(pdb_data_path, parameters)
print(f"Computed {len(avs)} AVs. IMP Model 'm' created and populated.")

print("Instantiating Scoring Restraint Wrapper...")
if SCORING_TYPE == "GMM":
    if gmm_sel_mean_val is None:  # Should not happen if GMM fitting was done
        print("Error: GMM parameters not available for GMM scoring type. Exiting.")
        exit(1)
    sr_wrapper = ScoringRestraintWrapper(
        m, avs,
        gmm_sel_components=gmm_sel_components_val,
        gmm_sel_mean=gmm_sel_mean_val,
        gmm_sel_cov=gmm_sel_cov_val,
        gmm_sel_weight=gmm_sel_weight_val,
        type=SCORING_TYPE
    )
elif SCORING_TYPE == "Tree":
    if kdtree_object is None or smlm_coordinates_for_tree is None:
        print("Error: KDTree or associated data not available for Tree scoring. Check flexible_filter_smlm_data.")
        exit(1)
    sr_wrapper = ScoringRestraintWrapper(
        m, avs,
        kdtree_obj=kdtree_object,
        dataxyz=smlm_coordinates_for_tree,  # Data used to build the tree
        var=smlm_variances,  # Variances corresponding to dataxyz
        # Default scaling, searchradius, offsetxyz are in ScoringRestraintWrapper's __init__
        # Pass them explicitly if you need non-default values:
        # scaling=1.0,
        # searchradius=10.0,
        type=SCORING_TYPE
    )
elif SCORING_TYPE == "Distance":
    print("Preparing SMLM data for 'Distance' scoring (using diagonal covariance matrices).")
    # smlm_coordinates are (N_points, 3) - these will be treated as means by computescoresimple
    # smlm_variances are (N_points,) - scalar variances

    smlm_covariances_list = []  # This will be a list of 3x3 matrices
    if smlm_coordinates.shape[0] > 0:  # Only proceed if there are points
        if smlm_variances.shape[0] != smlm_coordinates.shape[0]:
            print(
                f"Error: Mismatch in shapes for Distance scoring. Coordinates: {smlm_coordinates.shape}, Variances: {smlm_variances.shape}")
            exit(1)
        for var_scalar in smlm_variances:
            # Ensure var_scalar is positive to avoid issues with determinant or inverse if it were used directly
            # (though np.eye(3) * 0 is fine for det, just good practice)
            safe_var_scalar = max(var_scalar, 1e-9)  # Use a small positive floor for variance
            cov_matrix = np.eye(3) * safe_var_scalar  # Create a 3x3 diagonal covariance matrix
            smlm_covariances_list.append(cov_matrix)
    else:
        print("Warning: No SMLM data points after filtering for Distance scoring.")
        # smlm_covariances_list remains empty, computescoresimple should handle empty datamean/datacov

    sr_wrapper = ScoringRestraintWrapper(
        m, avs,
        dataxyz=smlm_coordinates,  # Pass the (N_points, 3) SMLM coordinates as "means"
        var=smlm_covariances_list,  # Pass the list of (3,3) diagonal covariance matrices
        # weights=None, # computescoresimple has a default for weights if not passed
        type=SCORING_TYPE
    )
else:
    raise ValueError(f"Unsupported SCORING_TYPE in NPC_example.py: {SCORING_TYPE}")

print("ScoringRestraintWrapper 'sr_wrapper' instantiated.")

initial_score = sr_wrapper.evaluate()
print(f"Initial Score calculated by the restraint: {initial_score}")

print("Creating IMP Scoring Function...")
try:
    sf = IMP.core.RestraintsScoringFunction(sr_wrapper.rs)
    print("IMP Scoring Function 'sf' created successfully using restraints from wrapper.")
except AttributeError:
    print("\nError: ScoringRestraintWrapper missing 'rs' attribute.")
    exit(1)
except Exception as e:
    print(f"\nError creating ScoringFunction: {e}")
    exit(1)

# --- Stage 3: Brownian Dynamics Simulation ---
run_brownian_dynamics_simulation(
    model=m,
    pdb_hierarchy=pdb_hierarchy,
    avs=avs,
    scoring_restraint_wrapper=sr_wrapper,
    output_dir=f"{SCORING_TYPE.lower()}_bd_output",
    rmf_filename=f"{SCORING_TYPE.lower()}_score_trajectory.rmf",
    temperature=300.0,
    max_time_step_fs=50000.0,
    number_of_bd_steps=1000,
    rmf_save_interval_frames=10
)

# --- Stage 4: Analysis (Placeholder) ---
print("\n--- Model Sampling Complete ---")
print(f"Analysis of the {SCORING_TYPE} score trajectory would follow here.")

