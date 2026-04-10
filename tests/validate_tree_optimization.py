import os
import sys
import numpy as np

# Ensure smlm_score is in PYTHONPATH
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
THESIS_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

from smlm_score.imp_modeling.scoring.distance_score import _compute_distance_score_cpu
from smlm_score.imp_modeling.scoring.tree_score import computescoretree, computescoretree_with_grad

def validate_tree_optimization():
    print("=== Validating Tree Engine Optimization ===")
    
    # 1. Setup small test case
    N = 100
    M = 32
    np.random.seed(42)
    data = np.random.rand(N, 3) * 100
    model = np.random.rand(M, 3) * 100
    variances = np.ones(N) # Scalar variances for simple comparison
    cov_indices = np.array([np.eye(3)] * N)
    
    # 2. Reference Distance Score (Always exact)
    # sigmaav=8.0 is the default in current pipeline
    sigmaav = 8.0
    dist_score = _compute_distance_score_cpu(data, cov_indices, np.ones(N), model, sigmaav)
    print(f"Distance Baseline Score: {dist_score:.10f}")
    
    # 3. Optimized Tree Score (Large Radius = Exact)
    # With a radius of 1000, it MUST include all points.
    tree_score = computescoretree(None, None, data, variances, searchradius=1000.0, model_coords_override=model)
    print(f"Optimized Tree Score:   {tree_score:.10f}")
    
    # 4. Correctness Check
    diff = abs(dist_score - tree_score)
    if diff < 1e-7:
        print(f"SUCCESS: Tree score matches Distance score (Diff: {diff:.2e})")
    else:
        print(f"FAILURE: Tree score MISMATCH (Diff: {diff:.2e})")
        
    # 5. Gradient Validation
    # We don't have a direct raw-CPU-Distance gradient exposed, but we can 
    # check if the gradient sum and shape are reasonable and consistent.
    tree_score_grad, grad = computescoretree_with_grad(None, None, data, variances, searchradius=1000.0, model_coords_override=model)
    print(f"Tree Score (Grad)      : {tree_score_grad:.10f}")
    print(f"Gradient Matrix Sum    : {np.sum(grad):.10f}")
    print(f"Gradient Matrix Mean   : {np.mean(np.abs(grad)):.10f}")
    
    if abs(tree_score_grad - dist_score) < 1e-7:
        print("SUCCESS: Tree Gradient score matches Distance score.")
    else:
        print("FAILURE: Tree Gradient score mismatch.")

if __name__ == '__main__':
    validate_tree_optimization()
