import IMP
import IMP.pmi
import IMP.pmi.dof
import IMP.pmi.macros
import IMP.pmi.output
import os
import sys


def run_bayesian_sampling(
    model: IMP.Model,
    pdb_hierarchy,
    avs: list,
    scoring_restraint_wrapper,
    output_dir: str = "bayesian_output",
    number_of_frames: int = 100,
    monte_carlo_steps: int = 10,
    number_of_best_scoring_models: int = 10
):
    """Run Bayesian Replica Exchange Monte Carlo sampling.

    Creates a rigid body from AV particles and runs IMP.pmi.macros.ReplicaExchange
    to sample probable structures given the experimental data scores.

    Parameters
    ----------
    model : IMP.Model
        The IMP model instance.
    pdb_hierarchy : IMP.atom.Hierarchy
        PDB hierarchy for RMF output.
    avs : list of IMP.bff.AV
        Accessible Volume decorators to sample.
    scoring_restraint_wrapper : ScoringRestraintWrapper
        Provides the scoring function and is registered with the model.
    output_dir : str
        Directory for REMC stat/RMF output.
    number_of_frames : int
        Total number of REMC frames to sample.
    monte_carlo_steps : int
        MC steps per frame.
    number_of_best_scoring_models : int
        Number of top-scoring models to retain.
    """
    print(f"\n--- Setting up Modular Bayesian Sampler ---")

    # 1. Define Degrees of Freedom (What can move?)
    dof = IMP.pmi.dof.DegreesOfFreedom(model)

    # Extract structural particles from the AVs
    particles_to_move = [av.get_particle() for av in avs]

    # Ensure particles have mass for rigid body Center of Mass calculations
    for p in particles_to_move:
        if not IMP.atom.Mass.get_is_setup(p):
            IMP.atom.Mass.setup_particle(p, 1.0)

    # Group them into a single rigid body for rigid structural alignment searching
    dof.create_rigid_body(particles_to_move, name="NPC_Complex")

    # 2. Tell the sampler which energy scores to evaluate
    if hasattr(scoring_restraint_wrapper, "set_return_objective"):
        scoring_restraint_wrapper.set_return_objective(True)
    if hasattr(scoring_restraint_wrapper, "add_to_model"):
        scoring_restraint_wrapper.add_to_model()
    output_objects = [scoring_restraint_wrapper]

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Initialize the standard IMP Replica Exchange Macro
    rex = IMP.pmi.macros.ReplicaExchange(
        model,
        root_hier=pdb_hierarchy,
        monte_carlo_sample_objects=dof.get_movers(),
        output_objects=output_objects,
        monte_carlo_temperature=1.0,
        replica_exchange_minimum_temperature=1.0,
        replica_exchange_maximum_temperature=2.5,
        number_of_best_scoring_models=number_of_best_scoring_models,
        monte_carlo_steps=monte_carlo_steps,
        number_of_frames=number_of_frames,
        global_output_directory=output_dir,
        replica_exchange_object=None,
        mmcif=True,
        test_mode=False
    )
    if hasattr(rex, "replica_exchange_object_cif"):
        rex.replica_exchange_object_cif = True

    # 4. Execute the Sampling with a Progress Bar overlay
    print(f"Executing REMC with {number_of_frames} frames ({monte_carlo_steps} steps/frame)...")

    import threading
    import time
    from tqdm import tqdm

    def run_imp_macro():
        try:
            rex.execute_macro()
        except Exception as e:
            print(f"MCMC Execution error: {e}")

    mcmc_thread = threading.Thread(target=run_imp_macro)
    mcmc_thread.start()

    stat_file_path = os.path.join(output_dir, "stat.0.out")
    time.sleep(1.0)

    current_frame = 0
    with tqdm(total=number_of_frames, desc="MCMC Sampling", disable=not sys.stdout.isatty()) as pbar:
        while mcmc_thread.is_alive():
            if os.path.exists(stat_file_path):
                with open(stat_file_path, "r") as f:
                    lines = [line for line in f if line.strip()]
                    new_frames_count = len(lines)

                if new_frames_count > current_frame:
                    pbar.update(new_frames_count - current_frame)
                    current_frame = new_frames_count
            time.sleep(2.0)

    # Final catch-up update
    mcmc_thread.join()
    if os.path.exists(stat_file_path):
        with open(stat_file_path, "r") as f:
            lines = [line for line in f if line.strip()]
            new_frames_count = len(lines)
            if new_frames_count > current_frame and new_frames_count <= number_of_frames:
                pbar.update(new_frames_count - current_frame)

    print(f"Sampling complete. Results saved to '{output_dir}'.")
