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
from smlm_score.src.utility.input import read_experimental_data
from smlm_score.src.utility.data_handling import flexible_filter_smlm_data

# Color settings
_DARK_BG = "#0d1117"
_TEXT_COLOR = "#c9d1d9"
_COLORS = {
    'Distance': '#3fb950',  # Green
    'Tree': '#58a6ff',      # Blue
    'GMM': '#ff7b72'        # Red
}

def generate_synthetic_model(n_points):
    """Generate a 32-point model to act as our 'IMP Geometry'."""
    theta = np.linspace(0, 2*np.pi, n_points)
    r = 60.0 # nm
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(n_points)
    return np.column_stack((x, y, z))

def benchmark_experimental():
    print("Loading actual experimental data cut...")
    data_path = os.path.join(THIS_DIR, "ShareLoc_Data", "data.csv")
    raw_df = read_experimental_data(data_path)
    
    # 1. Take the official User Cut first
    base_xyz, base_sigma, _, _, _ = flexible_filter_smlm_data(
        raw_df, filter_type='cut', x_cut=(10000, 12000), y_cut=(0, 5000), fill_z_value=0.0
    )
    
    total_cut_points = len(base_xyz)
    print(f"Full Experimental Cut Size: {total_cut_points} points.")
    
    # Test different target thresholds up to the max cut size
    target_sizes = [100, 300, 1000, 3000, 10000, 20000]
    data_sizes = [size for size in target_sizes if size <= total_cut_points]
    if total_cut_points > data_sizes[-1]:
        data_sizes.append(total_cut_points)
        
    eval_iterations = 50 # Avoid freezing on very large sets
    
    # Pre-generate IMP Model Geometry mock
    model = generate_synthetic_model(32)
    
    results = {
        'sizes': data_sizes,
        'Distance': {'init': [], 'eval': []},
        'Tree': {'init': [], 'eval': []},
        'GMM': {'init': [], 'eval': []}
    }
    
    # Warmup
    dummy_data = np.random.rand(10, 3)
    _compute_distance_score_cpu(dummy_data, np.array([np.eye(3)]*10), np.ones(10), model, 8.0)
    _compute_nb_gmm_cpu(model, np.array([[0,0,0]]), np.array([np.eye(3)]), np.array([1.0]))
    computescoretree(KDTree(dummy_data), None, dummy_data, np.ones(10), model_coords_override=model)
    
    for size in data_sizes:
        print(f"\nBenchmarking Experimental Sample Size: {size} points...")
        
        # We sample exactly `size` points from the base_xyz
        indices = np.random.choice(total_cut_points, size=size, replace=False)
        data = base_xyz[indices]
        variances_arr = base_sigma[indices] ** 2 if base_sigma is not None else np.ones(size)
        cov_arr = np.array([np.eye(3)] * size)
        
        # 1. Distance
        results['Distance']['init'].append(0.0)
        def eval_dist():
            _compute_distance_score_cpu(data, cov_arr, variances_arr, model, 8.0)
        dist_time = timeit.timeit(eval_dist, number=eval_iterations) / eval_iterations * 1000
        results['Distance']['eval'].append(dist_time)
        print(f"  [Dist] Init: 0.00ms | Eval: {dist_time:.2f}ms")
        
        # 2. Tree
        t0 = time.time()
        tree = KDTree(data)
        t_init_tree = (time.time() - t0) * 1000
        results['Tree']['init'].append(t_init_tree)
        def eval_tree():
            computescoretree(tree, None, data, variances_arr, model_coords_override=model)
        tree_time = timeit.timeit(eval_tree, number=eval_iterations) / eval_iterations * 1000
        results['Tree']['eval'].append(tree_time)
        print(f"  [Tree] Init: {t_init_tree:.2f}ms | Eval: {tree_time:.2f}ms")
        
        # 3. GMM
        t0 = time.time()
        # On experimental data, fitting Gaussians is much harder and slower.
        # We cap components tightly so benchmark finishes in reasonable time.
        gmm_res, gmm_sel, gmm_mean, gmm_cov, gmm_weight = test_gmm_components(data, component_min=1, component_max=8)
        t_init_gmm = (time.time() - t0) * 1000
        results['GMM']['init'].append(t_init_gmm)
        def eval_gmm():
            _compute_nb_gmm_cpu(model, gmm_mean, gmm_cov, gmm_weight)
        gmm_time = timeit.timeit(eval_gmm, number=eval_iterations) / eval_iterations * 1000
        results['GMM']['eval'].append(gmm_time)
        print(f"  [GMM ] Init: {t_init_gmm:.2f}ms | Eval: {gmm_time:.2f}ms")
        
    return results

def generate_experimental_figures(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Figure A: Evaluation Scaling (Log-Log)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    
    sizes = results['sizes']
    ax.plot(sizes, results['Distance']['eval'], label=f"Distance", color=_COLORS['Distance'], lw=3, marker='o')
    ax.plot(sizes, results['Tree']['eval'], label=f"Tree", color=_COLORS['Tree'], lw=3, marker='s')
    ax.plot(sizes, results['GMM']['eval'], label=f"GMM", color=_COLORS['GMM'], lw=3, marker='^')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Experimental SMLM Data Size (Number of sampled points)", color=_TEXT_COLOR, fontsize=12)
    ax.set_ylabel("Per-Step Evaluation Time (ms) [Log Scale]", color=_TEXT_COLOR, fontsize=12)
    ax.set_title("Scoring Engine Computational Scaling on Experimental Slice\ntesting x=(10k,12k) and y=(0,5k)", color='white', pad=15, fontsize=14)
    
    ax.tick_params(colors=_TEXT_COLOR, which='both')
    ax.grid(True, which='both', color='#30363d', alpha=0.3, ls='--')
    ax.legend(facecolor=_DARK_BG, edgecolor='#30363d', labelcolor='white', fontsize=11)
    
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        
    fig.tight_layout()
    plotA_path = os.path.join(output_dir, "bench_figC_exp_scaling.png")
    fig.savefig(plotA_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {plotA_path}")
    
    # -------------------------------------------------------------------------
    # Figure B: The MCMC Sampling Tradeoff (1 step vs 10,000 steps)
    # This directly answers the thesis question: "Is GMM's potential only visible in MCMC?"
    # -------------------------------------------------------------------------
    idx_target = 0
    for i, size in enumerate(sizes):
        if size >= 1000:
            idx_target = i
            break
            
    methods = ['Distance', 'Tree', 'GMM']
    inits = [results[m]['init'][idx_target] for m in methods]
    evals_per_step = [results[m]['eval'][idx_target] for m in methods]
    
    # We will make 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=_DARK_BG)
    fig.patch.set_facecolor(_DARK_BG)
    
    x = np.arange(len(methods))
    width = 0.5
    
    # ---------------------------------------------------------
    # SUBPLOT 1: Single Scoring Check (Initialization + 1 Eval)
    # ---------------------------------------------------------
    ax1.set_facecolor(_DARK_BG)
    single_eval = [e * 1 for e in evals_per_step]
    
    ax1.bar(x, inits, width, label='Initialization Cost (Once)', color='#8b949e')
    ax1.bar(x, single_eval, width, bottom=inits, label='Single Scoring Evaluation', 
            color=[_COLORS[m] for m in methods], alpha=0.9)
    
    ax1.set_ylabel("Total Execution Time (ms) [Log Scale]", color=_TEXT_COLOR, fontsize=12)
    ax1.set_title(f"A) Single Evaluation\n(GMM is the slowest)", color='white', pad=15, fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, color=_TEXT_COLOR, fontsize=12)
    ax1.tick_params(axis='y', colors=_TEXT_COLOR)
    ax1.set_yscale('log')
    ax1.grid(axis='y', color='#30363d', alpha=0.3, ls='--')
    
    for spine in ax1.spines.values(): spine.set_color('#30363d')
        
    for i in range(len(methods)):
        total_time = inits[i] + single_eval[i]
        ax1.text(x[i], max(total_time * 1.5, 0.1), f"{total_time:,.1f} ms", ha='center', color='white', fontweight='bold')
    
    # ---------------------------------------------------------
    # SUBPLOT 2: MCMC Sampling Check (Initialization + 10,000 Evals)
    # ---------------------------------------------------------
    ax2.set_facecolor(_DARK_BG)
    steps = 10000
    mcmc_eval = [e * steps for e in evals_per_step]
    
    ax2.bar(x, inits, width, label='Initialization Cost (Once)', color='#8b949e')
    ax2.bar(x, mcmc_eval, width, bottom=inits, label=f'Optimization Cost ({steps:,} evals)', 
            color=[_COLORS[m] for m in methods], alpha=0.9)
    
    ax2.set_title(f"B) MCMC Optimization ({steps:,} steps)\n(GMM is the fastest)", color='white', pad=15, fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, color=_TEXT_COLOR, fontsize=12)
    ax2.tick_params(axis='y', colors=_TEXT_COLOR)
    ax2.set_yscale('log')
    ax2.grid(axis='y', color='#30363d', alpha=0.3, ls='--')
    
    for spine in ax2.spines.values(): spine.set_color('#30363d')
        
    for i in range(len(methods)):
        total_time = inits[i] + mcmc_eval[i]
        ax2.text(x[i], max(total_time * 1.5, 0.1), f"{total_time:,.0f} ms", ha='center', color='white', fontweight='bold')
        
    # Super title highlighting the experimental dataset context
    plt.suptitle(f"Performance Trade-off on Experimental Data (N={sizes[idx_target]} points)", color='white', fontsize=16)
    
    # Shared Legend
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, facecolor=_DARK_BG, edgecolor='#30363d', labelcolor='white')
    
    plt.subplots_adjust(bottom=0.15, wspace=0.25)
    
    plotB_path = os.path.join(output_dir, "bench_figD_exp_tradeoff.png")
    fig.savefig(plotB_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {plotB_path}")

if __name__ == '__main__':
    res = benchmark_experimental()
    out_dir = os.path.join(THESIS_ROOT, "smlm_score", "figures")
    generate_experimental_figures(res, out_dir)
