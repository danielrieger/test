import IMP
import IMP.algebra
import IMP.atom
import IMP.core
import IMP.pmi
import IMP.pmi.dof
import IMP.pmi.macros
import IMP.pmi.output
import IMP.rmf
import RMF
import json
import numpy as np
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
    score_weight: float = 1.0,
    max_rb_trans: float = 4.0,
    max_rb_rot: float = 0.04,
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
    print(f"    Score Weight: {score_weight}")
    print(f"    Rigid Move:   trans={max_rb_trans}A, rot={max_rb_rot}rad")

    if hasattr(scoring_restraint_wrapper, "set_weight"):
        scoring_restraint_wrapper.set_weight(score_weight)

    # 1. Define Degrees of Freedom (What can move?)
    dof = IMP.pmi.dof.DegreesOfFreedom(model)

    # Extract structural particles from the AVs.
    # NOTE: We intentionally do NOT add AV particles to the PDB hierarchy
    # or include the full hierarchy in the rigid body. PMI's ReplicaExchange
    # macro requires specific metadata (State, Component) on all hierarchy
    # members that AV particles lack. Instead, we move only the AVs and
    # reconstruct the full-structure trajectory post-hoc using rigid body
    # transformation math in the custom RMF writer.
    particles_to_move = [av.get_particle() for av in avs]

    # Ensure particles have mass and XYZR for rigid body calculations
    for p in particles_to_move:
        if not IMP.atom.Mass.get_is_setup(p):
            IMP.atom.Mass.setup_particle(p, 1.0)
        if IMP.core.XYZ.get_is_setup(p) and not IMP.core.XYZR.get_is_setup(p):
            IMP.core.XYZR.setup_particle(p, 5.0)

    # Group them into a single rigid body for rigid structural alignment searching
    dof.create_rigid_body(particles_to_move, name="NPC_Complex", max_trans=max_rb_trans, max_rot=max_rb_rot)

    class AVOutput:
        def __init__(self, avs):
            self.avs = avs
        def get_output(self):
            out = {}
            for i, av in enumerate(self.avs):
                xyz = IMP.core.XYZ(av)
                out[f"AV_{i}_x"] = xyz.get_x()
                out[f"AV_{i}_y"] = xyz.get_y()
                out[f"AV_{i}_z"] = xyz.get_z()
            return out

    # 2. Tell the sampler which energy scores to evaluate
    if hasattr(scoring_restraint_wrapper, "set_return_objective"):
        scoring_restraint_wrapper.set_return_objective(True)
    if hasattr(scoring_restraint_wrapper, "add_to_model"):
        scoring_restraint_wrapper.add_to_model()
    
    # Add AV tracking to output objects so coords are saved in the stat file
    av_tracker = AVOutput(avs)
    output_objects = [scoring_restraint_wrapper, av_tracker]

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
                    
                    # Try to get the latest score
                    try:
                        import ast
                        last_line = lines[-1]
                        data = ast.literal_eval(last_line)
                        score_key = next((k for k in data.keys() if "_Score" in k), None)
                        if score_key:
                            score_val = float(data[score_key])
                            pbar.set_postfix({"score": f"{score_val:.2f}"})
                    except:
                        pass
                    
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

    # Write a readable CSV with the final AV coordinates and scores per frame
    # parsed from the stat file
    import ast
    stat_data = []
    header = {}
    if os.path.exists(stat_file_path):
        with open(stat_file_path, "r") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = ast.literal_eval(line)
                    if 'STAT2HEADER' in entry:
                        header = entry  # First line is the header
                    else:
                        stat_data.append(entry)
                except (ValueError, SyntaxError):
                    pass

    # Reverse mapping for header to find indices of AV coordinates
    inv_header = {v: k for k, v in header.items() if isinstance(k, int)}

    # Write per-frame scores from stat file
    scores_csv_path = os.path.join(output_dir, "frame_scores.csv")
    with open(scores_csv_path, "w") as f:
        f.write("frame,score,acceptance_rate,temperature\n")
        for entry in stat_data:
            frame = entry.get(inv_header.get('MonteCarlo_Nframe'), "?")
            score = entry.get(inv_header.get('Total_Score'), "?")
            acceptance = entry.get(inv_header.get('MonteCarlo_Acceptance_NPC_Complex_0'), "?")
            temp = entry.get(inv_header.get('MonteCarlo_Temperature'), "?")
            f.write(f"{frame},{score},{acceptance},{temp}\n")
    print(f"  Frame scores saved to: {scores_csv_path}")

    # Add post-sampling score summary
    if stat_data:
        try:
            # Score key can be 'Total_Score' or a specific restraint score
            score_key = inv_header.get('Total_Score')
            scores = [float(e.get(score_key, 0)) for e in stat_data if score_key in e]
            if scores:
                print(f"  Score: {scores[0]:.4f} (initial) \u2192 {scores[-1]:.4f} (final) | Best: {min(scores):.4f}")
        except:
            pass

    # Write final AV coordinates (post-sampling snapshot)
    av_coords_path = os.path.join(output_dir, "av_coordinates_final.csv")
    with open(av_coords_path, "w") as f:
        f.write("av_index,x,y,z\n")
        for i, av in enumerate(avs):
            xyz = IMP.core.XYZ(av)
            f.write(f"{i},{xyz.get_x():.6f},{xyz.get_y():.6f},{xyz.get_z():.6f}\n")
    print(f"  Final AV coordinates saved to: {av_coords_path}")

    # Write a dedicated AV-only RMF3 file with the FULL trajectory
    av_rmf_path = os.path.join(output_dir, "av_trajectory.rmf3")
    av_rmf = RMF.create_rmf_file(av_rmf_path)
    av_rmf.set_description("AV particle trajectory from REMC sampling")

    # Create nodes in the RMF for AV particles
    root = av_rmf.get_root_node()
    pf = RMF.ParticleFactory(av_rmf)
    cf = RMF.ColoredFactory(av_rmf)
    av_nodes = []
    for i, av in enumerate(avs):
        node = root.add_child(f"AV_{i}", RMF.REPRESENTATION)
        pf.get(node).set_mass(1.0)
        pf.get(node).set_radius(5.0)
        cf.get(node).set_rgb_color(RMF.Vector3(0.2, 0.8, 0.3))
        av_nodes.append(node)

    # Populate frames from stat data
    print(f"  Writing {len(stat_data)} frames to AV trajectory RMF...")
    for frame_idx, entry in enumerate(stat_data):
        av_rmf.add_frame(str(frame_idx), RMF.FRAME)
        for i in range(len(avs)):
            try:
                x = float(entry.get(inv_header.get(f"AV_{i}_x")))
                y = float(entry.get(inv_header.get(f"AV_{i}_y")))
                z = float(entry.get(inv_header.get(f"AV_{i}_z")))
                pf.get(av_nodes[i]).set_coordinates(RMF.Vector3(x, y, z))
            except (TypeError, ValueError):
                continue

    del av_rmf  # Close the file

    # Write a full-structure trajectory RMF using explicit global coordinates.
    # IMP.rmf.save_frame does NOT correctly record rigid body member positions
    # (it writes internal/local coordinates that appear frozen). Instead, we
    # reconstruct the rigid body transformation from the sampled AV coordinates
    # and apply it to the entire protein structure post-hoc.
    full_rmf_path = os.path.join(output_dir, "full_trajectory.rmf3")
    full_rmf = RMF.create_rmf_file(full_rmf_path)
    full_rmf.set_description("Full structural trajectory from REMC sampling")

    if not stat_data:
        del full_rmf
        print("  No stat data to write full trajectory (sampling may have failed).")
        return

    # Gather all hierarchy leaf particles (protein beads) + AV particles
    hier_leaves = IMP.atom.get_leaves(pdb_hierarchy)
    full_pf = RMF.ParticleFactory(full_rmf)
    full_cf = RMF.ColoredFactory(full_rmf)
    full_root = full_rmf.get_root_node()

    # Create RMF nodes: protein beads first, then AVs (colored green)
    rmf_nodes = []
    for leaf in hier_leaves:
        p = leaf.get_particle()
        node = full_root.add_child(leaf.get_name(), RMF.REPRESENTATION)
        xyzr = IMP.core.XYZR(p) if IMP.core.XYZR.get_is_setup(p) else None
        full_pf.get(node).set_mass(1.0)
        full_pf.get(node).set_radius(xyzr.get_radius() if xyzr else 1.0)
        rmf_nodes.append(node)

    av_rmf_nodes = []
    for i, av in enumerate(avs):
        node = full_root.add_child(f"AV_{i}", RMF.REPRESENTATION)
        full_pf.get(node).set_mass(1.0)
        full_pf.get(node).set_radius(5.0)
        full_cf.get(node).set_rgb_color(RMF.Vector3(0.2, 0.8, 0.3))
        av_rmf_nodes.append(node)

    print(f"  Writing {len(stat_data)} frames to full trajectory RMF "
          f"({len(rmf_nodes)} beads + {len(av_rmf_nodes)} AVs)...")

    def compute_rigid_transform(src, dst):
        """Compute rigid transformation (R, t) from src to dst point sets
        using SVD (Kabsch algorithm)."""
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_c = src - src_mean
        dst_c = dst - dst_mean
        H = src_c.T @ dst_c
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_mat = np.diag([1, 1, d])
        R = Vt.T @ sign_mat @ U.T
        t = dst_mean - R @ src_mean
        return R, t

    # Reference state: current (post-sampling) AV and hierarchy coordinates.
    # These serve as the "src" for the rigid transformation to each frame.
    ref_av_coords = np.array([
        np.array(IMP.core.XYZ(av).get_coordinates()) for av in avs
    ])
    ref_hier_coords = np.array([
        np.array(IMP.core.XYZ(leaf.get_particle()).get_coordinates())
        for leaf in hier_leaves
    ])

    for frame_idx, entry in enumerate(stat_data):
        full_rmf.add_frame(str(frame_idx), RMF.FRAME)

        # Extract this frame's AV coordinates from the stat data
        frame_av_coords = np.array([
            [float(entry.get(inv_header.get(f"AV_{i}_x"))),
             float(entry.get(inv_header.get(f"AV_{i}_y"))),
             float(entry.get(inv_header.get(f"AV_{i}_z")))]
            for i in range(len(avs))
        ])

        # Compute the rigid transform: reference AVs → this frame's AVs
        R, t = compute_rigid_transform(ref_av_coords, frame_av_coords)

        # Apply to ALL hierarchy bead coordinates
        frame_hier_coords = (R @ ref_hier_coords.T).T + t

        # Write protein bead coordinates
        for node, coords in zip(rmf_nodes, frame_hier_coords):
            full_pf.get(node).set_coordinates(
                RMF.Vector3(coords[0], coords[1], coords[2]))

        # Write AV coordinates (direct from stat data)
        for i, node in enumerate(av_rmf_nodes):
            full_pf.get(node).set_coordinates(
                RMF.Vector3(frame_av_coords[i, 0],
                            frame_av_coords[i, 1],
                            frame_av_coords[i, 2]))

    del full_rmf  # Close the file
    # Consolidate results in a clean footer
    print("\n" + "─" * 50)
    print(f"RESULTS: {output_dir}/")
    print(f"  full_trajectory.rmf3    {len(rmf_nodes)} beads \u00d7 {len(stat_data)} frames")
    print(f"  av_trajectory.rmf3      {len(av_rmf_nodes)} AVs \u00d7 {len(stat_data)} frames")
    print(f"  frame_scores.csv        score trace")
    print(f"  av_coordinates_final.csv  final AV positions")
    print("─" * 50)

