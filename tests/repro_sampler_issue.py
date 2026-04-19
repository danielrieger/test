import os
import sys
import IMP
import IMP.pmi.dof
import IMP.pmi.macros
import IMP.pmi.restraints
import numpy as np

# Ensure smlm_score is in PYTHONPATH
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
THESIS_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

from smlm_score.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper

def repro_sampler_issue():
    print("=== Reproducing Sampler Logging Issue ===")
    m = IMP.Model()
    
    # 1. Create a dummy particle
    p = IMP.Particle(m)
    IMP.core.XYZ.setup_particle(p, np.array([0,0,0]))
    IMP.atom.Mass.setup_particle(p, 1.0)
    
    # 2. Setup a dummy ScoringRestraintWrapper (Distance mode)
    # We need some dummy data
    data = np.random.rand(10, 3)
    var = [np.eye(3)] * 10
    
    # We'll use a fake AV for the particle
    class MockAV:
        def __init__(self, p): self.p = p
        def get_particle(self): return self.p
    
    avs = [MockAV(p)]
    
    sr_wrapper = ScoringRestraintWrapper(
        m, avs, dataxyz=data, var=var, type="Distance"
    )
    
    # 3. Setup Hierarchy and DOF
    root_hier = IMP.atom.Hierarchy.setup_particle(IMP.Particle(m))
    root_hier.add_child(p)
    
    dof = IMP.pmi.dof.DegreesOfFreedom(m)
    dof.create_rigid_body([p], name="P1")
    
    # 4. Enable Logging
    output_dir = "test_sampler_repro"
    sr_wrapper.enable_trajectory_logging(output_dir)
    sr_wrapper.add_to_model()
    
    # 5. Run Small Sampling
    print("Initializing ReplicaExchange...")
    rex = IMP.pmi.macros.ReplicaExchange(
        m,
        root_hier=root_hier,
        monte_carlo_sample_objects=dof.get_movers(),
        output_objects=[sr_wrapper],
        monte_carlo_temperature=1.0,
        replica_exchange_minimum_temperature=1.0,
        replica_exchange_maximum_temperature=1.0,
        number_of_best_scoring_models=1,
        monte_carlo_steps=1,
        number_of_frames=2,
        global_output_directory=output_dir,
    )
    
    print("Executing macro...")
    try:
        rex.execute_macro()
    except Exception as e:
        print(f"FAILED: {e}")
        return

    # Check for files
    trace_path = os.path.join(output_dir, "trajectory_trace.csv")
    stat_path = os.path.join(output_dir, "stat.0.out")
    
    if os.path.exists(trace_path):
        print(f"SUCCESS: {trace_path} exists.")
        with open(trace_path, 'r') as f:
            print(f"Content Preview:\n{f.read()}")
    else:
        print(f"FAILURE: {trace_path} MISSING.")
        
    if os.path.exists(stat_path):
        print(f"Stat file size: {os.path.getsize(stat_path)} bytes")

if __name__ == '__main__':
    repro_sampler_issue()
