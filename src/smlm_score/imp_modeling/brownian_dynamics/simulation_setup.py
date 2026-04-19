import os

import IMP
import IMP.atom
import IMP.core
import IMP.pmi.tools
import IMP.rmf
import RMF

DEFAULT_AV_RADIUS_ANGSTROM = 30.0


def run_brownian_dynamics_simulation(
    model,
    pdb_hierarchy=None,
    avs=None,
    scoring_restraint_wrapper=None,
    output_dir="brownian_output",
    rmf_filename="bd_trajectory.rmf",
    temperature=300.0,
    max_time_step_fs=50000.0,
    number_of_bd_steps=500,
    rmf_save_interval_frames=10,
    scoring_function=None,
    restraint_set_for_rmf=None,
):
    """Set up and run a Brownian Dynamics simulation.

    Configures AV particles with XYZR decorators, diffusion coefficients,
    and optimization flags, then runs IMP.atom.BrownianDynamics.

    Parameters
    ----------
    model : IMP.Model
        The IMP model instance.
    pdb_hierarchy : IMP.atom.Hierarchy or None
        PDB hierarchy for RMF output.
    avs : list of IMP.bff.AV or None
        Accessible Volume decorators to simulate.
    scoring_restraint_wrapper : ScoringRestraintWrapper or None
        Wrapper providing the scoring function. GMM is not supported.
    output_dir : str
        Directory for RMF trajectory output.
    rmf_filename : str
        Name of the output RMF file.
    temperature : float
        Simulation temperature in Kelvin.
    max_time_step_fs : float
        Maximum BD time step in femtoseconds.
    number_of_bd_steps : int
        Number of BD integration steps.
    rmf_save_interval_frames : int
        Save to RMF every N steps.
    scoring_function : IMP.ScoringFunction or None
        Direct scoring function (legacy API).
    restraint_set_for_rmf : IMP.RestraintSet or None
        Restraint set for RMF logging (legacy API).

    Returns
    -------
    dict
        Dictionary with 'initial_score', 'final_score', 'rmf_path'.
    """
    print("\n--- Setting up Brownian Dynamics Simulation ---")

    if scoring_restraint_wrapper is not None:
        print(f"  Scoring type: {scoring_restraint_wrapper.type}")
        if scoring_restraint_wrapper.type == "GMM":
            raise ValueError(
                "GMM scoring does not support gradient-based Brownian Dynamics. "
                "Use 'Tree' or 'Distance' scoring for Brownian mode."
            )
        if avs is None:
            raise ValueError("`avs` is required when using `scoring_restraint_wrapper`.")

        for av in avs:
            p = av.get_particle()
            try:
                radius = max(float(r) for r in av.get_radii())
            except Exception:
                radius = DEFAULT_AV_RADIUS_ANGSTROM

            xyz = IMP.core.XYZ(av)
            if IMP.core.XYZR.get_is_setup(p):
                xyzr = IMP.core.XYZR(p)
            else:
                xyzr = IMP.core.XYZR.setup_particle(p)
                xyzr.set_coordinates(xyz.get_coordinates())

            xyzr.set_radius(max(radius, 1e-3))
            xyzr.set_coordinates_are_optimized(True)

            if not IMP.atom.Mass.get_is_setup(p):
                IMP.atom.Mass.setup_particle(p, 1.0)

            diffusion_coefficient = IMP.atom.get_einstein_diffusion_coefficient(
                xyzr.get_radius()
            )
            if IMP.atom.Diffusion.get_is_setup(p):
                IMP.atom.Diffusion(p).set_diffusion_coefficient(diffusion_coefficient)
            else:
                IMP.atom.Diffusion.setup_particle(p, diffusion_coefficient)

        if hasattr(scoring_restraint_wrapper, "set_return_objective"):
            scoring_restraint_wrapper.set_return_objective(True)
        scoring_restraint_wrapper.add_to_model()
        scoring_function = IMP.core.RestraintsScoringFunction(
            IMP.pmi.tools.get_restraint_set(model).get_restraints(),
            "BrownianSF",
        )
        restraint_set_for_rmf = IMP.pmi.tools.get_restraint_set(model)
    else:
        print("  Scoring type: legacy scoring function")
        if scoring_function is None or restraint_set_for_rmf is None:
            raise ValueError(
                "Provide either `scoring_restraint_wrapper` or both "
                "`scoring_function` and `restraint_set_for_rmf`."
            )

    print(f"  Temperature: {temperature}K")
    print(f"  Max time step: {max_time_step_fs}fs")
    print(f"  Number of steps: {number_of_bd_steps}")

    bd = IMP.atom.BrownianDynamics(model)
    bd.set_scoring_function(scoring_function)
    bd.set_temperature(temperature)
    bd.set_maximum_time_step(max_time_step_fs)
    bd.set_log_level(IMP.SILENT)

    os.makedirs(output_dir, exist_ok=True)
    rmf_path = os.path.join(output_dir, rmf_filename)

    rmf_file = RMF.create_rmf_file(rmf_path)
    rmf_file.set_description(
        f"Brownian Dynamics trajectory. T={temperature}K. Max_dt={max_time_step_fs}fs."
    )

    if pdb_hierarchy:
        IMP.rmf.add_hierarchy(rmf_file, pdb_hierarchy)

    if restraint_set_for_rmf is not None and hasattr(restraint_set_for_rmf, "get_restraints"):
        try:
            IMP.rmf.add_restraints(rmf_file, restraint_set_for_rmf.get_restraints())
        except Exception:
            pass

    sos = IMP.rmf.SaveOptimizerState(model, rmf_file)
    sos.set_period(rmf_save_interval_frames)
    bd.add_optimizer_state(sos)

    model.update()
    print("Saving initial configuration to RMF.")
    sos.update_always("initial conformation")

    print(f"Objective before simulation: {scoring_function.evaluate(False):.4f}")
    print(f"Running Brownian Dynamics simulation for {number_of_bd_steps} steps...")
    bd.optimize(number_of_bd_steps)

    model.update()
    sos.update_always("final conformation")

    print("Brownian Dynamics simulation finished.")
    print(f"Objective after simulation: {scoring_function.evaluate(False):.4f}")
    print(f"\nSimulation trajectory saved to {rmf_path}")
