import IMP
import IMP.core
import IMP.atom # Likely needed for simulation setup later
from smlm_score.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper
from smlm_score.utility.input import read_parameters_from_json # Assuming function name based on [6]
from smlm_score.utility.data_handling import flexible_filter_smlm_data, compute_av # Assuming function names based on [5, 6]
from smlm_score.utility.input import read_experimental_data
from smlm_score.imp_modeling.scoring.gmm_score import test_gmm_components # Assuming function name based on [9, 6]
# May need other IMP modules like IMP.bff depending on function implementations

# Define paths (adjust if necessary) [6]
smlm_data_path = "ShareLoc_Data/data.csv"
# Assuming PDB file is in a subdirectory based on common practice and [3]
pdb_data_path = "PDB_Data/7N85-assembly1.cif"
av_parameters_path = "av_parameter.json"

print("--- IMP Model Setup Script ---")

# --- Stage 1: Gather and Process Data ---

# Read parameters from JSON file [6]
print(f"Reading parameters from: {av_parameters_path}")
# Assuming read_parameters_from_json correctly parses the JSON content [6]
parameters = read_parameters_from_json(av_parameters_path)
print("Parameters loaded.")

# Read experimental SMLM data [6, 10]
print(f"Reading SMLM data from: {smlm_data_path}")
# Assuming read_experimental_data returns a pandas DataFrame or similar structure [10]
raw_smlm_data = read_experimental_data(smlm_data_path)
print(f"Read {len(raw_smlm_data)} localizations.")

# Filter SMLM data based on criteria (e.g., spatial region) [5, 10]
print("Filtering SMLM data...")
# Assuming filter_smlm_data returns the primary filtered dataset needed (e.g., dataxyz)
# The function in [5] returns multiple datasets; we take the first one here.
filtered_data_xyz, *_ = flexible_filter_smlm_data(raw_smlm_data)
print(f"Filtered data contains {len(filtered_data_xyz)} localizations.")

# --- !!! Check for sufficient data before GMM fitting !!! ---
if filtered_data_xyz.shape[0] < 2:
    print("\nError: Insufficient data points (< 2) passed the initial filter.")
    print("Cannot fit Gaussian Mixture Model. Check filtering parameters in data_handling.py.")
    # Option: Exit, or try using a different dataset like filtered_data_xyz2 if appropriate
    exit(1) # Or handle differently

# Fit Gaussian Mixture Model (GMM) to filtered SMLM data [6, 9, 10]
print("Fitting GMM to SMLM data (testing multiple components)...")
# test_gmm_components likely fits GMMs with varying components and selects the best
# based on BIC, returning results and the selected model's parameters. [9]
gmm_results, gmm_sel, gmm_sel_mean, gmm_sel_cov, gmm_sel_weight = test_gmm_components(filtered_data_xyz)
print(f"Selected GMM with {gmm_sel.n_components} components based on BIC.")

# --- Stage 2: Define Representation and Scoring Function ---

# Create IMP Model instance ('m') and compute Accessible Volumes (AVs) [5, 6]
# The compute_av function handles IMP.Model() creation and PDB reading. [5]
print(f"Creating IMP Model and computing AVs from: {pdb_data_path}")
# Pass the relevant parameters needed by compute_av [5]
avs, m, _ = compute_av(pdb_data_path, parameters) # 'm' (IMP.Model) is returned here
print(f"Computed {len(avs)} AVs. IMP Model 'm' created and populated.")

# Instantiate the Scoring Restraint Wrapper [1, 6]
# This wrapper internally creates the ScoringRestraintGMM and adds it to a RestraintSet 'rs'.
print("Instantiating Scoring Restraint Wrapper...")
sr_wrapper = ScoringRestraintWrapper(
    m,                  # Pass the IMP Model instance
    avs,                # Pass the list of AV particles (which belong to 'm')
    # Pass the parameters of the selected GMM [6, 9]
    gmm_sel.n_components, # Or potentially just gmm_sel depending on wrapper implementation
    gmm_sel_mean,
    gmm_sel_cov,
    gmm_sel_weight,
    "GMM"               # Identifier for the restraint type
)
print("ScoringRestraintWrapper 'sr_wrapper' instantiated.")

# --- Prepare for Sampling/Optimization ---

# Evaluate the initial score using the wrapper's evaluate method (optional, for info) [1, 6]
initial_score = sr_wrapper.evaluate()
print(f"Initial Score calculated by the restraint: {initial_score}")

# Create the IMP Scoring Function using the RestraintSet from the wrapper [1, 2, 4, 7]
# This is the standard way to prepare restraints for IMP samplers.
print("Creating IMP Scoring Function...")
try:
    # Access the IMP.RestraintSet 'rs' attribute within the wrapper instance [1]
    sf = IMP.core.RestraintsScoringFunction(sr_wrapper.rs)
    print("IMP Scoring Function 'sf' created successfully using restraints from wrapper.")
except AttributeError:
    print("\nError: Critical setup failure.")
    print("The 'ScoringRestraintWrapper' object does not have the expected 'rs' attribute (RestraintSet).")
    print("Please check the implementation in 'scoring_restraint.py'.")
    # Decide how to handle this error (e.g., exit)
    exit(1)
except Exception as e:
    print(f"\nError: An unexpected error occurred while creating the ScoringFunction: {e}")
    # Decide how to handle this error (e.g., exit)
    exit(1)

# --- Stage 3: Sampling (Example Placeholder) ---

print("\n--- Model Setup Complete ---")
print("The IMP Model 'm' and the Scoring Function 'sf' are ready for use.")
print("Next steps typically involve setting up and running a sampler.")
print("Example using Brownian Dynamics:")
# print("import IMP.atom")
# print("bd = IMP.atom.BrownianDynamics(m)")
# print("bd.set_scoring_function(sf)")
# print("bd.set_temperature(300)") # Set temperature in Kelvin
# print("# Set other BD parameters like time step, friction coefficients if needed")
# print("num_steps = 1000 # Example number of steps")
# print(f"Running Brownian Dynamics simulation for {num_steps} steps...")
# print("bd.optimize(num_steps)")
# print("Simulation finished.")

# --- Stage 4: Analysis (Placeholder) ---
# Following the simulation, analysis of the trajectory (e.g., clustering, precision estimation [4])
# would be performed.
print("\nAnalysis of the simulation results would follow here.")





"""








# Core Imports
import numpy as np
import pandas as pd

import IMP


import IMP.algebra

# Custom Imports
from smlm_score.imp_modeling.restraint.scoring_restraint import *
from smlm_score.utility.input import *
from smlm_score.utility.data_handling import *
from smlm_score.utility.plot import *
from smlm_score.imp_modeling.scoring.gmm_score import *


####################

smlm_data_path = "ShareLoc_Data/data.csv"
pdb_data_path = "PDB_Data/7N85-assembly1.cif"
av_parameters = "av_parameter.json"

parameters = read_parameters_from_json(av_parameters)
data_xyz = read_experimental_data(smlm_data_path)
data_xyz = filter_smlm_data(data_xyz)

######################
print("Compute AV")
avs, m = compute_av(pdb_data_path, parameters)
print("Compute Components")
gmms, gmm_sel, gmm_sel_mean, gmm_sel_cov, gmm_sel_weight = test_gmm_components(data_xyz)

#bis hierhin scheint okay
#print(gmm_sel_mean)

sr = ScoringRestraintWrapper(m, avs, gmm_sel, gmm_sel_mean, gmm_sel_cov, gmm_sel_weight, "GMM")
sr_score = sr.evaluate()
print(sr)
sr.add_to_model()



Die Bestimmung der Anzahl Komponenten und die Erstellung 
des PDB-Models geschieht acuh intern in der Restraint Klasse.

PDB übergeben:
hier
chains
residue
atom_name
av_parameter
Das Model steckt in hier. muss also nicht separat übergeben werden, 
sondern steckt in m = hier.get_model()

data_xyz
einfach Daten übergeben.
test_gmm_components dann im Konstruktor
 

"""

"""
# GMM for experimental data (data_xyz)

gmm_result = test_gmm_components(data_xyz)

gmm_sel = gmm_result['gmm'][gmm_result['n']]
gmm_sel_mean = gmm_sel.means_
gmm_sel_cov = gmm_sel.covariances_
gmm_sel_weight = gmm_sel.weights_


# Model-GMM from PDB-data

m = IMP.Model(pathlib.Path(pdb_data_path).absolute().as_posix())
hier = IMP.atom.read_mmcif(str(pdb_data_path), m)
chains = ['0', '0-2', '0-3', '0-4', '0-5', '0-6', '0-7', '0-8']
avs = list()
for chain in tqdm.tqdm(chains):
    # Atom CA of residue 133 of chain 0
    av_p = IMP.Particle(m)
    residue_index = 133
    atom_name = "CA"
    sel = IMP.atom.Selection(hier)
    sel.set_chain_id(chain)
    sel.set_atom_type(IMP.atom.AtomType(atom_name))
    sel.set_residue_index(residue_index)
    source = sel.get_selected_particles()[0]
    av_parameter = {
        "linker_length": 150.0,
        "radii": (30.0, 0.0, 0.0),
        "linker_width": 4.0,
        "allowed_sphere_radius": 10.0,
        "contact_volume_thickness": 0.0,
        "contact_volume_trapped_fraction": -1,
        "simulation_grid_resolution": 8.0
    }
    IMP.bff.AV.do_setup_particle(m, av_p, source, **av_parameter)
    av = IMP.bff.AV(m, av_p)
    av.resample()
    avs.append(av)
    mean = IMP.core.XYZ(av)
    print(mean)

#




#####################

model = IMP.Model()
p_root = IMP.Particle(model, "root")
h_root = IMP.atom.Hierarchy.setup_particle(p_root)


#sr = ScoringRestraintWrapper()
#sr.add_to_model()
"""


"""
# Paths to Data Files (Replace with actual paths)


# Create Model Instance with Paths Passed to Constructor
model = Model(smlm_data_path=smlm_data_path,
              pdb_data_path=pdb_data_path)

model.initialize()

#model.filterSMLM
#gmm_scorer = GMM(model)
print(len(model.data_xyz))


# Initialize and Run Model
#model.initialize()
#model.run()

print("Model execution completed.")
"""
