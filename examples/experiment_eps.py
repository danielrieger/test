"""
Experiment: DBSCAN vs HDBSCAN for NPC isolation.

Compares DBSCAN at various eps values with HDBSCAN (paper-recommended
min_cluster_size=15). Generates overview panels and density maps.

Run from examples/:
    C:\envs\py311\python.exe experiment_eps.py
"""

import sys, os
import numpy as np
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# IMP imports
import IMP
import IMP.core
import IMP.atom
import IMP.bff

# sklearn + scipy
from sklearn.cluster import DBSCAN, HDBSCAN
from scipy.stats import gaussian_kde
from scipy.linalg import eigh
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =========================================================================
# Standalone helpers
# =========================================================================

def load_smlm_csv(path):
    """Load SMLM data from ShareLoc CSV format."""
    import csv
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row.get('x [nm]', row.get('x', 0)))
            y = float(row.get('y [nm]', row.get('y', 0)))
            z = float(row.get('z [nm]', row.get('z', 0)))
            rows.append([x, y, z])
    return np.array(rows)


def filter_smlm(data, x_cut=None, y_cut=None, fill_z=0.0):
    """Simple spatial filter."""
    mask = np.ones(len(data), dtype=bool)
    if x_cut:
        mask &= (data[:, 0] >= x_cut[0]) & (data[:, 0] <= x_cut[1])
    if y_cut:
        mask &= (data[:, 1] >= y_cut[0]) & (data[:, 1] <= y_cut[1])
    filtered = data[mask].copy()
    if fill_z is not None:
        filtered[:, 2] = fill_z
    return filtered


def compute_avs_standalone(pdb_path, param_path):
    """Compute AVs using IMP — mirrors data_handling.compute_av."""
    import pathlib
    with open(param_path) as f:
        parameter = json.load(f)

    chains = parameter['chains']
    residue_index = parameter['residue_index']
    atom_name = parameter['atom_name']
    av_setup_params = parameter['av_parameter']

    m = IMP.Model()
    pdb_path_str = str(pathlib.Path(pdb_path).absolute())
    hier = IMP.atom.read_mmcif(pdb_path_str, m)

    for p in IMP.atom.get_by_type(hier, IMP.atom.ATOM_TYPE):
        IMP.core.XYZ(p).set_coordinates_are_optimized(False)

    avs = []
    for chain_id in chains:
        av_particle = IMP.Particle(m)
        sel = IMP.atom.Selection(hier)
        sel.set_chain_id(str(chain_id))
        sel.set_atom_type(IMP.atom.AtomType(atom_name))
        sel.set_residue_index(residue_index)
        selected = sel.get_selected_particles()
        if not selected:
            print(f"Warning: no particle for chain '{chain_id}', skipping.")
            continue
        source_atom = selected[0]
        IMP.bff.AV.do_setup_particle(m, av_particle, source_atom, **av_setup_params)
        av = IMP.bff.AV(av_particle)
        av.resample()
        avs.append(av)

    return avs, m, hier


def align_pca(points):
    """PCA alignment — center and rotate to principal axes."""
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1
    return centered @ eigenvectors


def plot_cluster_panel(ax, data, labels, title, text_color, dark_bg):
    """Plot one clustering result on an axis."""
    ax.set_facecolor(dark_bg)
    noise_mask = labels == -1
    ax.scatter(data[noise_mask, 0], data[noise_mask, 1], s=0.1, c='#333333', alpha=0.3)
    unique_labels = sorted(set(labels) - {-1})
    n_clusters = len(unique_labels)
    clr = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    for i, cid in enumerate(unique_labels):
        mask = labels == cid
        ax.scatter(data[mask, 0], data[mask, 1], s=0.3, c=[clr[i % len(clr)]], alpha=0.5)

    n_noise = int(np.sum(noise_mask))
    ax.set_title(f"{title}\n{n_clusters} clusters, {n_noise} noise",
                 color=text_color, fontsize=11, fontweight='bold')
    ax.set_xlabel("x [nm]", color=text_color, fontsize=9)
    ax.set_ylabel("y [nm]", color=text_color, fontsize=9)
    ax.tick_params(colors=text_color, labelsize=7)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_color('#21262d')


# ── config ────────────────────────────────────────────────────────────────
SMLM_DATA_PATH = "ShareLoc_Data/data.csv"
PDB_DATA_PATH = "PDB_Data/7N85-assembly1.cif"
PARAM_PATH = "av_parameter.json"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPS_VALUES = [15, 30, 60]  # nm (DBSCAN comparison)

_cmap = LinearSegmentedColormap.from_list(
    "smlm", ["#0d1117", "#0e2a47", "#0d4a7a", "#1a7fba", "#58a6ff", "#a5d6ff", "#ffffff"], N=256
)
_DARK_BG = "#0d1117"
_TEXT_COLOR = "#c9d1d9"

# ── load + filter ─────────────────────────────────────────────────────────
print("=== Loading SMLM data ===")
raw = load_smlm_csv(SMLM_DATA_PATH)
print(f"Raw data: {len(raw)} localizations")

data_nm = filter_smlm(raw, x_cut=(10000, 12000), y_cut=(0, 5000), fill_z=0.0)
print(f"Filtered: {len(data_nm)} points in ROI")

# ── AVs ───────────────────────────────────────────────────────────────────
print("\n=== Computing Accessible Volumes ===")
avs, m, hier = compute_avs_standalone(PDB_DATA_PATH, PARAM_PATH)
model_coords_nm = np.array([
    np.array(IMP.core.XYZ(av).get_coordinates()) * 0.1
    for av in avs
])
print(f"Computed {len(avs)} AVs. Spread: {np.ptp(model_coords_nm, axis=0)} nm")

# =========================================================================
# Part 1: Overview — DBSCAN at various eps + HDBSCAN
# =========================================================================
print("\n=== Clustering Comparison ===\n")

n_panels = len(EPS_VALUES) + 1  # +1 for HDBSCAN
fig_overview, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5), facecolor=_DARK_BG)

all_results = {}

# DBSCAN panels
for ax, eps in zip(axes[:len(EPS_VALUES)], EPS_VALUES):
    print(f"DBSCAN eps={eps} nm ...")
    db = DBSCAN(eps=eps, min_samples=10)
    labels = db.fit_predict(data_nm[:, :2])
    n_clust = len(set(labels)) - (1 if -1 in labels else 0)
    sizes = sorted(Counter(labels[labels >= 0]).values(), reverse=True) if n_clust > 0 else []
    print(f"  → {n_clust} clusters. Top 5: {sizes[:5]}")
    all_results[f"DBSCAN eps={eps}"] = labels
    plot_cluster_panel(ax, data_nm, labels, f"DBSCAN eps={eps} nm", _TEXT_COLOR, _DARK_BG)

# HDBSCAN panel
print(f"HDBSCAN min_cluster_size=15 ...")
hdb = HDBSCAN(min_cluster_size=15, min_samples=None, cluster_selection_method='eom')
hdb_labels = hdb.fit_predict(data_nm[:, :2])
n_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
sizes = sorted(Counter(hdb_labels[hdb_labels >= 0]).values(), reverse=True) if n_hdb > 0 else []
print(f"  → {n_hdb} clusters. Top 5: {sizes[:5]}")
all_results["HDBSCAN"] = hdb_labels
plot_cluster_panel(axes[-1], data_nm, hdb_labels, "HDBSCAN\nmin_cluster_size=15", _TEXT_COLOR, _DARK_BG)

fig_overview.suptitle("DBSCAN vs HDBSCAN for NPC Isolation", color=_TEXT_COLOR, fontsize=16, fontweight='bold')
fig_overview.tight_layout()
fig_overview.savefig(os.path.join(OUTPUT_DIR, "dbscan_vs_hdbscan.png"), dpi=200, facecolor=_DARK_BG, bbox_inches='tight')
print(f"\nSaved: figures/dbscan_vs_hdbscan.png")

# =========================================================================
# Part 2: HDBSCAN density maps of individual NPC clusters
# =========================================================================
print("\n=== HDBSCAN NPC Density Maps ===")

labels = hdb_labels
npc_clusters = []
for cid in range(n_hdb):
    n_pts = int(np.sum(labels == cid))
    if 50 <= n_pts <= 3000:
        npc_clusters.append((cid, n_pts))
npc_clusters.sort(key=lambda x: x[1], reverse=True)

print(f"NPC-sized clusters (50-3000 pts): {len(npc_clusters)}")

n_show = min(12, len(npc_clusters))
n_cols = 4
n_rows = max(1, (n_show + n_cols - 1) // n_cols)

fig_density, axes_d = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), facecolor=_DARK_BG)
if n_rows == 1 and n_cols == 1:
    axes_flat = [axes_d]
elif n_rows == 1:
    axes_flat = list(axes_d)
else:
    axes_flat = list(axes_d.flatten())

for idx in range(len(axes_flat)):
    ax = axes_flat[idx]
    ax.set_facecolor(_DARK_BG)
    for spine in ax.spines.values():
        spine.set_color('#21262d')

    if idx >= n_show:
        ax.set_visible(False)
        continue

    cid, n_pts = npc_clusters[idx]
    pts = data_nm[labels == cid]
    aligned = align_pca(pts)
    xy = aligned[:, :2]

    # KDE density
    try:
        kde = gaussian_kde(xy.T)
        margin = 20
        xmin, xmax = xy[:, 0].min() - margin, xy[:, 0].max() + margin
        ymin, ymax = xy[:, 1].min() - margin, xy[:, 1].max() + margin
        g = 200
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, g), np.linspace(ymin, ymax, g))
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.imshow(zz, origin='lower', extent=[xmin, xmax, ymin, ymax],
                  aspect='equal', cmap=_cmap, interpolation='bilinear')
    except Exception:
        ax.scatter(xy[:, 0], xy[:, 1], s=0.5, c='#58a6ff', alpha=0.5)

    # AV overlay
    av_shifted = model_coords_nm + (aligned.mean(axis=0) - model_coords_nm.mean(axis=0))
    ax.scatter(av_shifted[:, 0], av_shifted[:, 1], s=40, c='#f0883e',
               edgecolors='white', linewidths=0.8, marker='*', zorder=10)

    ax.set_title(f"Cluster {cid} ({n_pts} pts)", color=_TEXT_COLOR, fontsize=10, fontweight='bold')
    ax.tick_params(colors=_TEXT_COLOR, labelsize=6)

fig_density.suptitle("Individual NPC Clusters (HDBSCAN)", color=_TEXT_COLOR, fontsize=16, fontweight='bold')
fig_density.tight_layout()
fig_density.savefig(os.path.join(OUTPUT_DIR, "hdbscan_npc_grid.png"), dpi=200, facecolor=_DARK_BG, bbox_inches='tight')
print(f"Saved: figures/hdbscan_npc_grid.png")

print(f"\n=== Done! ===")
