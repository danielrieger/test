import os

import IMP
import IMP.atom
import IMP.core
import IMP.pmi.tools
import IMP.rmf
import numpy as np
import RMF


def run_frequentist_optimization(
    model: IMP.Model,
    pdb_hierarchy,
    avs: list,
    scoring_restraint_wrapper,
    output_dir: str = "frequentist_output",
    max_cg_steps: int = 200,
    rmf_filename: str = "frequentist_result.rmf",
):
    """
    Run IMP.core.ConjugateGradients to find a single best-fit structure.

    The scoring wrapper normally reports raw fit scores for analysis. For IMP
    optimization we switch it into objective mode so the scoring function
    consistently returns the quantity to minimize.
    """
    print("\n--- Setting up Frequentist (MLE) Optimizer ---")
    print(f"  Scoring type: {scoring_restraint_wrapper.type}")
    print(f"  Max CG steps: {max_cg_steps}")

    if scoring_restraint_wrapper.type == "GMM":
        raise ValueError(
            "GMM scoring does not support gradient-based optimization. "
            "Use 'Tree' or 'Distance' scoring for frequentist optimization."
        )

    for av in avs:
        p = av.get_particle()
        if not IMP.atom.Mass.get_is_setup(p):
            IMP.atom.Mass.setup_particle(p, 1.0)
        IMP.core.XYZ(av).set_coordinates_are_optimized(True)

    if hasattr(scoring_restraint_wrapper, "set_return_objective"):
        scoring_restraint_wrapper.set_return_objective(True)
    scoring_restraint_wrapper.add_to_model()

    sf = IMP.core.RestraintsScoringFunction(
        IMP.pmi.tools.get_restraint_set(model).get_restraints(),
        "FrequentistSF",
    )

    initial_objective = sf.evaluate(False)
    print(f"  Initial optimization objective: {initial_objective:.4f}")

    initial_coords = np.array(
        [np.array(IMP.core.XYZ(av).get_coordinates()) for av in avs]
    )

    cg = IMP.core.ConjugateGradients(model)
    cg.set_scoring_function(sf)

    os.makedirs(output_dir, exist_ok=True)
    rmf_path = os.path.join(output_dir, rmf_filename)

    rmf_file = RMF.create_rmf_file(rmf_path)
    rmf_file.set_description("Frequentist MLE optimization result")

    if pdb_hierarchy:
        IMP.rmf.add_hierarchy(rmf_file, pdb_hierarchy)

    sos = IMP.rmf.SaveOptimizerState(model, rmf_file)
    sos.set_period(1)
    model.update()
    sos.update_always("initial conformation")

    print(f"\n  Running Conjugate Gradients optimization ({max_cg_steps} max steps)...")
    optimizer_return = cg.optimize(max_cg_steps)

    model.update()
    sos.update_always("optimized conformation")

    final_coords = np.array(
        [np.array(IMP.core.XYZ(av).get_coordinates()) for av in avs]
    )
    final_objective = sf.evaluate(False)
    coord_shift = np.linalg.norm(final_coords - initial_coords, axis=1)

    print("\n--- Frequentist Optimization Complete ---")
    print(f"  Initial objective: {initial_objective:.4f}")
    print(f"  Final objective:   {final_objective:.4f}")
    print(f"  Improvement:       {initial_objective - final_objective:.4f}")
    print(f"  Mean coordinate shift: {np.mean(coord_shift):.4f} A")
    print(f"  Max coordinate shift:  {np.max(coord_shift):.4f} A")
    print(f"  Structure saved to: {rmf_path}")

    return {
        "initial_score": initial_objective,
        "final_score": final_objective,
        "optimizer_return": optimizer_return,
        "optimized_coords": final_coords,
        "initial_coords": initial_coords,
        "n_steps": max_cg_steps,
        "coord_shifts": coord_shift,
    }
