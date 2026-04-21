import os
import sys
import time
import timeit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn import mixture

# Ensure smlm_score is in PYTHONPATH
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
THESIS_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

from smlm_score.imp_modeling.scoring.distance_score import _compute_distance_score_cpu
from smlm_score.imp_modeling.scoring.tree_score import computescoretree, computescoretree_with_grad
from smlm_score.imp_modeling.scoring.gmm_score import _compute_nb_gmm_cpu, test_gmm_components
from smlm_score.utility.input import read_experimental_data
from smlm_score.utility.data_handling import flexible_filter_smlm_data

# =============================================================================
# THEME CONFIGURATION
# =============================================================================
THEMES = {
    'dark': {
        'bg': "#0d1117",
        'text': "#c9d1d9",
        'grid': "#30363d",
        'spine': "#30363d",
        'fig_bg': "#0d1117",
        'legend_bg': "#0d1117",
        'init_color': "#8b949e"
    },
    'light': {
        'bg': "#ffffff",
        'text': "#000000",
        'grid': "#d0d7de",
        'spine': "#d0d7de",
        'fig_bg': "#ffffff",
        'legend_bg': "#f6f8fa",
        'init_color': "#afb8c1"
    }
}

_COLORS = {
    'Distance': '#3fb950',  # Green
    'Tree': '#58a6ff',      # Blue
    'GMM': '#ff7b72'        # Red
}

COMPLEXITIES = {
    'Distance': r"$O(NM)$",
    'Tree': r"$O(N \log M)$",
    'GMM': r"$O(GK)$"
}

def apply_theme(ax, theme_name):
    t = THEMES[theme_name]
    ax.set_facecolor(t['bg'])
    ax.tick_params(colors=t['text'], which='both')
    ax.xaxis.label.set_color(t['text'])
    ax.yaxis.label.set_color(t['text'])
    ax.title.set_color(t['text'] if theme_name == 'light' else 'white')
    ax.grid(True, which='both', color=t['grid'], alpha=0.3, ls='--')
    for spine in ax.spines.values():
        spine.set_color(t['spine'])

# =============================================================================
# DATA GENERATION & WARMUP
# =============================================================================
def generate_synthetic_data(n_points):
    """Generate a synthetic noisy ring dataset."""
    theta = np.linspace(0, 2*np.pi, n_points)
    r = 60.0 # nm
    x = r * np.cos(theta) + np.random.normal(0, 2, n_points)
    y = r * np.sin(theta) + np.random.normal(0, 2, n_points)
    z = np.random.normal(0, 5, n_points)
    return np.column_stack((x, y, z))

def warmup():
    print("Warming up Numba kernels...")
    model = generate_synthetic_data(128)
    data = generate_synthetic_data(10)
    cov = np.array([np.eye(3)]*10)
    weights = np.ones(10)
    _compute_distance_score_cpu(data, cov, weights, model, 8.0)
    _compute_nb_gmm_cpu(model, np.array([[0,0,0]]), np.array([np.eye(3)]), np.array([1.0]))
    computescoretree(KDTree(data), None, data, weights, model_coords_override=model)
    computescoretree_with_grad(KDTree(data), None, data, weights, model_coords_override=model)
    print("Warmup complete.\n")

# =============================================================================
# BENCHMARK ROUTINES
# =============================================================================
def run_all_benchmarks():
    data_sizes = [100, 300, 1000, 3000, 10000]
    eval_iterations = 100
    model = generate_synthetic_data(128)
    
    print("Loading Experimental SMLM Data...")
    csv_path = os.path.join(THESIS_ROOT, "smlm_score", "examples", "ShareLoc_Data", "data.csv")
    raw_df = read_experimental_data(csv_path)
    base_xyz, base_sigma, _, _, _ = flexible_filter_smlm_data(
        raw_df, filter_type='cut', x_cut=(10000, 12000), y_cut=(0, 5000), fill_z_value=0.0
    )
    
    results = {
        'sizes': data_sizes,
        'Distance': {'init': [], 'eval': []},
        'Tree': {'init': [], 'eval': []},
        'GMM': {'init': [], 'eval': []}
    }
    
    # --- Part 1: Scaling & Tradeoff (A, B) ---
    for size in data_sizes:
        print(f"Scaling Benchmark (Experimental Data): N={size}...")
        
        # Sample points from experimental data
        if size > len(base_xyz):
            indices = np.random.choice(len(base_xyz), size=size, replace=True)
        else:
            indices = np.random.choice(len(base_xyz), size=size, replace=False)
            
        data = base_xyz[indices]
        variances = base_sigma[indices] ** 2 if base_sigma is not None else np.ones(size)
        covs = np.array([np.eye(3)] * size)
        
        # Distance
        results['Distance']['init'].append(0.0)
        t_dist = timeit.timeit(lambda: _compute_distance_score_cpu(data, covs, variances, model, 8.0), number=eval_iterations) / eval_iterations * 1000
        results['Distance']['eval'].append(t_dist)
        
        # Tree
        t0 = time.time()
        tree = KDTree(data)
        t_init_tree = (time.time() - t0) * 1000
        results['Tree']['init'].append(t_init_tree)
        t_tree = timeit.timeit(lambda: computescoretree(tree, None, data, variances, model_coords_override=model), number=eval_iterations) / eval_iterations * 1000
        results['Tree']['eval'].append(t_tree)
        
        # GMM
        t0 = time.time()
        _, _, g_mean, g_cov, g_weight = test_gmm_components(data, component_min=1, component_max=8, reg_covar=0.1)
        t_init_gmm = (time.time() - t0) * 1000
        results['GMM']['init'].append(t_init_gmm)
        t_gmm = timeit.timeit(lambda: _compute_nb_gmm_cpu(model, g_mean, g_cov, g_weight), number=eval_iterations) / eval_iterations * 1000
        results['GMM']['eval'].append(t_gmm)

    return results

def run_gmm_init_benchmark(base_xyz, data_sizes):
    """Measures BIC selection time for GMM across sizes."""
    points = []
    times = []
    for size in data_sizes:
        print(f"GMM Init Benchmark: N={size}...")
        if size > len(base_xyz):
            indices = np.random.choice(len(base_xyz), size=size, replace=True)
        else:
            indices = np.random.choice(len(base_xyz), size=size, replace=False)
        data = base_xyz[indices]
        
        t0 = time.time()
        test_gmm_components(data, component_min=1, component_max=8, reg_covar=0.1)
        times.append(time.time() - t0)
        points.append(size)
    return {'points': points, 'time': times}

def run_radius_benchmark(base_xyz, base_sigma):
    """Measures Tree scaling across search radii at fixed N=5,000."""
    N_FIXED = 5000
    radii = [1, 2, 5, 10, 15, 20, 30, 50, 100, 200]
    indices = np.random.choice(len(base_xyz), size=N_FIXED, replace=False)
    data = base_xyz[indices]
    variances = base_sigma[indices] ** 2 if base_sigma is not None else np.ones(N_FIXED)
    model = generate_synthetic_data(128)
    
    times = []
    print(f"Radius Sensitivity Benchmark (N={N_FIXED})...")
    for r in radii:
        t_r = timeit.timeit(lambda: computescoretree(None, None, data, variances, searchradius=r, model_coords_override=model), number=20) / 20 * 1000
        times.append(t_r)
        print(f"  Radius: {r:>3} nm | Time: {t_r:7.2f} ms")
    return {'radii': radii, 'times': times}

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def export_fig(fig, base_name, output_dir, theme_name):
    suffix = f"_{theme_name}.png"
    path = os.path.join(output_dir, base_name + suffix)
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

def plot_fig_a(results, output_dir, theme):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=THEMES[theme]['fig_bg'])
    apply_theme(ax, theme)
    sizes = results['sizes']
    for m in ['Distance', 'Tree', 'GMM']:
        label = f"{m} {COMPLEXITIES[m]}"
        ax.plot(sizes, results[m]['eval'], label=label, color=_COLORS[m], lw=3, marker='os^'[['Distance','Tree','GMM'].index(m)])
    
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("SMLM Data Size (N)"); ax.set_ylabel("Eval Time (ms) [Log]")
    ax.set_title("Figure A: Computational Scaling Performance\n(Using Real Experimental SMLM Data)", fontweight='bold', pad=15)
    ax.legend(facecolor=THEMES[theme]['legend_bg'], labelcolor=THEMES[theme]['text'])
    export_fig(fig, "bench_figA_scaling", output_dir, theme)

def plot_fig_b(results, output_dir, theme):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=THEMES[theme]['fig_bg'])
    apply_theme(ax, theme)
    idx_1k = results['sizes'].index(1000)
    methods = ['Distance', 'Tree', 'GMM']
    inits = [results[m]['init'][idx_1k] for m in methods]
    steps = 10000
    evals = [results[m]['eval'][idx_1k] * steps for m in methods]
    
    x = np.arange(len(methods))
    width = 0.5
    p1 = ax.bar(x, inits, width, label='Initialization (Build/Fit)', color=THEMES[theme]['init_color'])
    p2 = ax.bar(x, evals, width, bottom=inits, label=f'Optimization ({steps:,} steps)', color=[_COLORS[m] for m in methods])
    
    ax.set_ylabel("Time (ms) [Log]"); ax.set_yscale('log')
    ax.set_title("MCMC Performance Trade-off (Real Experimental Data, N=1,000)")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    
    # Custom combined legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=THEMES[theme]['init_color'], lw=4, label='Setup Phase'),
        Line2D([0], [0], color='gray', lw=0, label='  '), # Spacer
        Line2D([0], [0], color=_COLORS['Distance'], lw=4, label='Run: Distance'),
        Line2D([0], [0], color=_COLORS['Tree'], lw=4, label='Run: Tree'),
        Line2D([0], [0], color=_COLORS['GMM'], lw=4, label='Run: GMM')
    ]
    ax.legend(handles=legend_elements, facecolor=THEMES[theme]['legend_bg'], labelcolor=THEMES[theme]['text'], loc='upper right', fontsize=9)
    
    for i, m in enumerate(methods):
        ax.text(x[i], (inits[i]+evals[i])*1.2, f"{inits[i]+evals[i]:,.0f}ms", ha='center', color=THEMES[theme]['text'], fontweight='bold')
        
    export_fig(fig, "bench_figB_tradeoff", output_dir, theme)

def plot_fig_c(results, output_dir, theme):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=THEMES[theme]['fig_bg'])
    apply_theme(ax, theme)
    ax.plot(results['GMM_scaling']['points'], results['GMM_scaling']['time'], color=_COLORS['GMM'], lw=3, marker='^')
    ax.set_xlabel("SMLM Points (N)"); ax.set_ylabel("Selection Time (s)")
    ax.set_title("GMM BIC-Optimal Component Selection Cost")
    export_fig(fig, "bench_figC_exp_scaling", output_dir, theme)

def plot_fig_d(results, output_dir, theme):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=THEMES[theme]['fig_bg'])
    apply_theme(ax, theme)
    
    idx_10k = results['sizes'].index(10000)
    methods = ['Distance', 'Tree']
    eval_10k = [results[m]['eval'][idx_10k] for m in methods]
    speedup = eval_10k[0] / eval_10k[1]
    
    bars = ax.bar(methods, eval_10k, color=[_COLORS['Distance'], _COLORS['Tree']], width=0.6)
    ax.set_yscale('log')
    ax.set_ylabel("Eval Time (ms) [Log]")
    ax.set_title(f"Engine Performance Comparison (N=10,000)\nSpeedup Factor: {speedup:.1f}x")
    
    # Text adjustment to prevent overlap
    y_max = ax.get_ylim()[1]
    ax.set_ylim(bottom=ax.get_ylim()[0], top=y_max * 5.0) # Add 5x more headroom
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height * 1.5, f'{height:.2f} ms', ha='center', va='bottom', color=THEMES[theme]['text'], fontweight='bold')
    
    export_fig(fig, "bench_figD_exp_tradeoff", output_dir, theme)

def plot_fig_e(results, output_dir, theme):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=THEMES[theme]['fig_bg'])
    apply_theme(ax, theme)
    radii = results['Radius']['radii']
    ax.plot(radii, results['Radius']['times'], color=_COLORS['Tree'], lw=3, marker='s')
    ax.set_xlabel("Search Radius (nm)"); ax.set_ylabel("Eval Time (ms)")
    ax.set_title("Radius Sensitivity: Pruning Efficiency vs. Radius\n(Fixed N=5,000)")
    export_fig(fig, "bench_figE_radius", output_dir, theme)

def save_summary_table(results, output_dir):
    """
    Generates a Markdown table with the most relevant measurements for thesis citation.
    """
    idx_1k = results['sizes'].index(1000)
    idx_10k = results['sizes'].index(10000)
    methods = ['Distance', 'Tree', 'GMM']
    steps = 10000
    
    table = []
    table.append("# SMLM-IMP Scoring Performance Summary")
    table.append("| Metric | Distance | Tree (Opt) | GMM |")
    table.append("| :--- | :---: | :---: | :---: |")
    
    # Eval Latency @ 1k
    row = ["Eval Latency (N=1k)"]
    for m in methods:
        row.append(f"{results[m]['eval'][idx_1k]:.3f} ms")
    table.append("| " + " | ".join(row) + " |")
    
    # Eval Latency @ 10k
    row = ["Eval Latency (N=10k)"]
    for m in methods:
        row.append(f"{results[m]['eval'][idx_10k]:.3f} ms")
    table.append("| " + " | ".join(row) + " |")
    
    # Init Cost @ 1k
    row = ["Initialization (N=1k)"]
    for m in methods:
        row.append(f"{results[m]['init'][idx_1k]:.2f} ms")
    table.append("| " + " | ".join(row) + " |")
    
    # Total MCMC (10k steps) @ 1k
    row = ["Total MCMC (10k steps)"]
    for m in methods:
        total = results[m]['init'][idx_1k] + (results[m]['eval'][idx_1k] * steps)
        row.append(f"{total/1000:.2f} s")
    table.append("| " + " | ".join(row) + " |")
    
    # Speedup vs Distance
    row = ["Speedup (vs Distance @ 10k)"]
    baseline = results['Distance']['eval'][idx_10k]
    for m in methods:
        s = baseline / results[m]['eval'][idx_10k]
        row.append(f"{s:.1f}x")
    table.append("| " + " | ".join(row) + " |")
    
    content = "\n".join(table)
    path_md = os.path.join(output_dir, "benchmark_summary_table.md")
    with open(path_md, "w") as f:
        f.write(content)
    
    # Also save raw CSV for data processing
    path_csv = os.path.join(output_dir, "benchmark_summary_data.csv")
    import csv
    with open(path_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Distance", "Tree_Opt", "GMM"])
        writer.writerow(["Eval_Latency_N1k_ms", results['Distance']['eval'][idx_1k], results['Tree']['eval'][idx_1k], results['GMM']['eval'][idx_1k]])
        writer.writerow(["Eval_Latency_N10k_ms", results['Distance']['eval'][idx_10k], results['Tree']['eval'][idx_10k], results['GMM']['eval'][idx_10k]])
        writer.writerow(["Init_Cost_N1k_ms", results['Distance']['init'][idx_1k], results['Tree']['init'][idx_1k], results['GMM']['init'][idx_1k]])
        total_mcmc = [results[m]['init'][idx_1k] + (results[m]['eval'][idx_1k] * steps) for m in methods]
        writer.writerow(["Total_MCMC_10k_Steps_1k_sec", total_mcmc[0]/1000, total_mcmc[1]/1000, total_mcmc[2]/1000])
        writer.writerow(["Speedup_vs_Distance_10k", 1.0, results['Distance']['eval'][idx_10k]/results['Tree']['eval'][idx_10k], results['Distance']['eval'][idx_10k]/results['GMM']['eval'][idx_10k]])

    print(f"Saved: {path_md}")
    print(f"Saved: {path_csv}")
    print("\n" + content + "\n")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    warmup()
    
    # Load experimental data for specialized sub-benchmarks
    THIS_DIR = os.path.abspath(os.path.dirname(__file__))
    THESIS_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
    csv_path = os.path.join(THESIS_ROOT, "smlm_score", "examples", "ShareLoc_Data", "data.csv")
    raw_df = read_experimental_data(csv_path)
    base_xyz, base_sigma, _, _, _ = flexible_filter_smlm_data(
        raw_df, filter_type='cut', x_cut=(10000, 12000), y_cut=(0, 5000), fill_z_value=0.0
    )

    # Run benchmarks
    data_results = run_all_benchmarks()
    data_results['GMM_scaling'] = run_gmm_init_benchmark(base_xyz, data_results['sizes'])
    data_results['Radius'] = run_radius_benchmark(base_xyz, base_sigma)

    figures_dir = os.path.join(THESIS_ROOT, "smlm_score", "examples", "figures", "benchmarks")
    os.makedirs(figures_dir, exist_ok=True)
    
    for theme in ['dark', 'light']:
        print(f"\nGenerating {theme} theme figures...")
        plot_fig_a(data_results, figures_dir, theme)
        plot_fig_b(data_results, figures_dir, theme)
        plot_fig_c(data_results, figures_dir, theme)
        plot_fig_d(data_results, figures_dir, theme)
        plot_fig_e(data_results, figures_dir, theme)
    
    save_summary_table(data_results, figures_dir)
    print("\nMaster Benchmark Complete. All 5 Figures and Table generated in examples/figures/benchmarks/")
