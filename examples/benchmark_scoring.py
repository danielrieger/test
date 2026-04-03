import os
import sys
import time
import timeit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

# Ensure smlm_score is in PYTHONPATH
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
THESIS_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

from smlm_score.src.imp_modeling.scoring.distance_score import _compute_distance_score_cpu
from smlm_score.src.imp_modeling.scoring.tree_score import computescoretree
from smlm_score.src.imp_modeling.scoring.gmm_score import _compute_nb_gmm_cpu, test_gmm_components

# Color settings
_DARK_BG = "#0d1117"
_TEXT_COLOR = "#c9d1d9"
_COLORS = {
    'Distance': '#3fb950',  # Green
    'Tree': '#58a6ff',      # Blue
    'GMM': '#ff7b72'        # Red
}

def generate_synthetic_data(n_points):
    """Generate a synthetic noisy ring dataset."""
    theta = np.linspace(0, 2*np.pi, n_points)
    r = 60.0 # nm
    x = r * np.cos(theta) + np.random.normal(0, 2, n_points)
    y = r * np.sin(theta) + np.random.normal(0, 2, n_points)
    z = np.random.normal(0, 5, n_points)
    return np.column_stack((x, y, z))

def benchmark_scoring_methods():
    data_sizes = [100, 300, 1000, 3000, 10000]
    eval_iterations = 100 # How many times to loop evaluation for averaging
    
    # Use >64 model points so the Tree scorer exercises the real KDTree pruning
    # path instead of the small-model exact fallback.
    model = generate_synthetic_data(128)
    
    results = {
        'sizes': data_sizes,
        'Distance': {'init': [], 'eval': []},
        'Tree': {'init': [], 'eval': []},
        'GMM': {'init': [], 'eval': []}
    }
    
    # Warmup Numba (crucial for accurate timings)
    dummy_data = generate_synthetic_data(10)
    _compute_distance_score_cpu(dummy_data, np.array([np.eye(3)]*10), np.ones(10), model, 8.0)
    _compute_nb_gmm_cpu(model, np.array([[0,0,0]]), np.array([np.eye(3)]), np.array([1.0]))
    computescoretree(KDTree(dummy_data), None, dummy_data, np.ones(10), model_coords_override=model)
    
    print("=== thesis Benchmarking ===\n")
    
    for size in data_sizes:
        print(f"Benchmarking N={size} points...")
        data = generate_synthetic_data(size)
        variances_arr = np.ones(size)
        cov_arr = np.array([np.eye(3)] * size)
        
        # -----------------------------------------------------
        # 1. Distance Score
        # -----------------------------------------------------
        # Init: 0
        results['Distance']['init'].append(0.0)
        # Eval
        def eval_dist():
            _compute_distance_score_cpu(data, cov_arr, variances_arr, model, 8.0)
        dist_time = timeit.timeit(eval_dist, number=eval_iterations) / eval_iterations * 1000 # ms
        results['Distance']['eval'].append(dist_time)
        
        # -----------------------------------------------------
        # 2. Tree Score
        # -----------------------------------------------------
        # Init: KDTree build
        t0 = time.time()
        tree = KDTree(data)
        t_init_tree = (time.time() - t0) * 1000
        results['Tree']['init'].append(t_init_tree)
        # Eval
        def eval_tree():
            computescoretree(tree, None, data, variances_arr, model_coords_override=model)
        tree_time = timeit.timeit(eval_tree, number=eval_iterations) / eval_iterations * 1000
        results['Tree']['eval'].append(tree_time)
        
        # -----------------------------------------------------
        # 3. GMM Score
        # -----------------------------------------------------
        # Init: BIC Fitting
        t0 = time.time()
        # Cap components between 1 and 8 to keep benchmark fast, as real data is complex
        gmm_res, gmm_sel, gmm_mean, gmm_cov, gmm_weight = test_gmm_components(data, component_min=1, component_max=8)
        t_init_gmm = (time.time() - t0) * 1000
        results['GMM']['init'].append(t_init_gmm)
        # Eval
        def eval_gmm():
            _compute_nb_gmm_cpu(model, gmm_mean, gmm_cov, gmm_weight)
        gmm_time = timeit.timeit(eval_gmm, number=eval_iterations) / eval_iterations * 1000
        results['GMM']['eval'].append(gmm_time)
        
        print(f"  Dist: init=0ms, eval={dist_time:.3f}ms")
        print(f"  Tree: init={t_init_tree:.2f}ms, eval={tree_time:.3f}ms")
        print(f"   GMM: init={t_init_gmm:.2f}ms, eval={gmm_time:.3f}ms")
        
    return results

def generate_thesis_figures(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Figure A: Evaluation Scaling (Log-Log)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    
    sizes = results['sizes']
    ax.plot(sizes, results['Distance']['eval'], label=f"Distance O(NM)", color=_COLORS['Distance'], lw=3, marker='o')
    ax.plot(sizes, results['Tree']['eval'], label="Tree (KDTree candidate pruning)", color=_COLORS['Tree'], lw=3, marker='s')
    ax.plot(sizes, results['GMM']['eval'], label=f"GMM O(GK)", color=_COLORS['GMM'], lw=3, marker='^')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("SMLM Data Size (Number of points N)", color=_TEXT_COLOR, fontsize=12)
    ax.set_ylabel("Per-Step Evaluation Time (ms) [Log Scale]", color=_TEXT_COLOR, fontsize=12)
    ax.set_title("Scoring Engine Computational Scaling\n(N=Data, M=Model, K=Gaussian Components)", color='white', pad=15, fontsize=14)
    
    ax.tick_params(colors=_TEXT_COLOR, which='both')
    ax.grid(True, which='both', color='#30363d', alpha=0.3, ls='--')
    ax.legend(facecolor=_DARK_BG, edgecolor='#30363d', labelcolor='white', fontsize=11)
    
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bench_figA_scaling.png"), dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.join(output_dir, 'bench_figA_scaling.png')}")
    
    # -------------------------------------------------------------------------
    # Figure B: Standard NPC Tradeoff (Init vs 100k Eval Steps)
    # -------------------------------------------------------------------------
    # Get index for N=1000 points (standard NPC size)
    idx_1k = sizes.index(1000)
    
    methods = ['Distance', 'Tree', 'GMM']
    inits = [results[m]['init'][idx_1k] for m in methods]
    evals_per_step = [results[m]['eval'][idx_1k] for m in methods]
    
    # To make the bar chart readable, let's simulate exactly ONE optimization macro-step
    # which might execute 10,000 evaluations.
    steps = 10000
    total_evals = [e * steps for e in evals_per_step]
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    
    x = np.arange(len(methods))
    width = 0.5
    
    # Stacked bars: Initializing Cost + Optimization Cost
    bottoms = inits
    ax.bar(x, inits, width, label='Initialization Cost (Once)', color='#8b949e')
    ax.bar(x, total_evals, width, bottom=inits, label=f'Optimization Cost ({steps:,} eval steps)', 
           color=[_COLORS[m] for m in methods], alpha=0.9)
    
    ax.set_ylabel("Total Execution Time (ms) [Log Scale]", color=_TEXT_COLOR, fontsize=12)
    ax.set_title(f"Performance Trade-off\nMCMC Stage (Standard NPC N=1000)", color='white', pad=15, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, color=_TEXT_COLOR, fontsize=12)
    ax.tick_params(axis='y', colors=_TEXT_COLOR)
    
    ax.set_yscale('log')
    ax.grid(axis='y', color='#30363d', alpha=0.3, ls='--')
    ax.legend(facecolor=_DARK_BG, edgecolor='#30363d', labelcolor='white')
    
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        
    # Annotate total height to emphasize the speed of GMM overall
    for i in range(len(methods)):
        total_time = inits[i] + total_evals[i]
        ax.text(x[i], total_time * 1.2, f"{total_time:,.0f} ms", ha='center', color='white', fontweight='bold')
        
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bench_figB_tradeoff.png"), dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.join(output_dir, 'bench_figB_tradeoff.png')}")

if __name__ == '__main__':
    res = benchmark_scoring_methods()
    out_dir = os.path.join(THESIS_ROOT, "smlm_score", "examples", "figures", "benchmarks")
    generate_thesis_figures(res, out_dir)
