import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

# Ensure smlm_score is in PYTHONPATH
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
THESIS_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)

from smlm_score.imp_modeling.scoring.distance_score import _compute_distance_score_cpu
from smlm_score.imp_modeling.scoring.tree_score import computescoretree
from smlm_score.utility.input import read_experimental_data
from smlm_score.utility.data_handling import flexible_filter_smlm_data

# Aesthetic settings
_DARK_BG = "#0d1117"
_TEXT_COLOR = "#c9d1d9"
_COLORS = {'Distance': '#3fb950', 'Tree': '#58a6ff'}

def benchmark_radius_sensitivity():
    print("Loading experimental data for Radius Sensitivity Analysis...")
    data_path = os.path.join(THIS_DIR, "ShareLoc_Data", "data.csv")
    raw_df = read_experimental_data(data_path)
    
    # Use a fixed 10,000 point sample to keep it fast but representative
    base_xyz, base_sigma, _, _, _ = flexible_filter_smlm_data(
        raw_df, filter_type='cut', x_cut=(10000, 12000), y_cut=(0, 5000), fill_z_value=0.0
    )
    N_SAMPLES = 10000
    indices = np.random.choice(len(base_xyz), size=N_SAMPLES, replace=False)
    data = base_xyz[indices]
    variances_arr = base_sigma[indices] ** 2 if base_sigma is not None else np.ones(N_SAMPLES)
    cov_arr = np.array([np.eye(3)] * N_SAMPLES)
    
    # Mock model
    theta = np.linspace(0, 2*np.pi, 32)
    r = 60.0
    model = np.column_stack((r*np.cos(theta), r*np.sin(theta), np.zeros(32)))
    
    radii = [1, 2, 5, 10, 15, 20, 30, 50, 100, 200]
    
    results = {
        'radii': radii,
        'Tree': [],
        'Distance': [] # Distance is radius-independent, but we'll measure it once
    }
    
    # 1. Baseline Distance (Average of 5 runs)
    print(f"Measuring Distance baseline (N={N_SAMPLES})...")
    times = []
    for _ in range(5):
        t0 = time.time()
        _compute_distance_score_cpu(data, cov_arr, variances_arr, model, 8.0)
        times.append((time.time() - t0) * 1000)
    dist_baseline = np.mean(times)
    results['Distance'] = [dist_baseline] * len(radii)
    
    # 2. Tree Sensitivity Sweep
    print(f"Sweeping Search Radii for Tree Engine...")
    for r_test in radii:
        times = []
        # Fewer iterations for larger radii to keep benchmark fast
        iters = 5 if r_test <= 20 else 2
        for _ in range(iters):
            t0 = time.time()
            # Note: tree=None because our optimized code builds a small model-tree internally
            computescoretree(None, None, data, variances_arr, searchradius=r_test, model_coords_override=model)
            times.append((time.time() - t0) * 1000)
        t_avg = np.mean(times)
        results['Tree'].append(t_avg)
        print(f"  Radius: {r_test:>3} nm | Tree Time: {t_avg:7.2f} ms")
        
    return results

def plot_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    
    radii = results['radii']
    ax.plot(radii, results['Distance'], label="Distance Engine (Radius Independent)", color=_COLORS['Distance'], lw=3, ls='--', alpha=0.7)
    ax.plot(radii, results['Tree'], label="Tree Engine (Optimized)", color=_COLORS['Tree'], lw=4, marker='o', markersize=8)
    
    ax.set_xlabel("Search Radius (nm)", color=_TEXT_COLOR, fontsize=12)
    ax.set_ylabel("Execution Time (ms)", color=_TEXT_COLOR, fontsize=12)
    ax.set_title("Radius Sensitivity Analysis: Tree vs. Distance Pruning\n(Experimental NPC Cut, N=10,000 points)", color='white', pad=20, fontsize=14)
    
    ax.tick_params(colors=_TEXT_COLOR)
    ax.grid(True, which='major', color='#30363d', alpha=0.3, ls='--')
    ax.legend(facecolor=_DARK_BG, edgecolor='#30363d', labelcolor='white', fontsize=11)
    
    # Annotate crossover if exists
    crossover = None
    for i in range(len(radii)):
        if results['Tree'][i] > results['Distance'][i]:
            crossover = radii[i]
            break
    
    if crossover:
        ax.axvline(crossover, color='#ff7b72', ls=':', alpha=0.5)
        ax.text(crossover + 1, ax.get_ylim()[1]*0.8, f"Crossover point\n(~{crossover} nm)", color='#ff7b72', fontweight='bold')
    
    for spine in ax.spines.values(): spine.set_color('#30363d')
    fig.tight_layout()
    
    out_path = os.path.join(output_dir, "bench_figE_radius_sensitivity.png")
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    res = benchmark_radius_sensitivity()
    out_dir = os.path.join(THIS_DIR, "figures", "benchmarks")
    plot_results(res, out_dir)
