import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

# ── Daten generieren (aus test_stage5_alignment_unit.py) ──────────────────────
def make_tilted_ring(n=180, r=50., noise=6.):
    theta = np.linspace(0., 2*np.pi, n, endpoint=False)
    ring = np.column_stack([r*np.cos(theta), r*np.sin(theta), np.zeros(n)])
    ax_r, ay_r = np.deg2rad(35.), np.deg2rad(-20.)
    Rx = np.array([[1,0,0],[0,np.cos(ax_r),-np.sin(ax_r)],[0,np.sin(ax_r),np.cos(ax_r)]])
    Ry = np.array([[np.cos(ay_r),0,np.sin(ay_r)],[0,1,0],[-np.sin(ay_r),0,np.cos(ay_r)]])
    pts = ring @ (Ry @ Rx).T + np.array([120.,-35.,80.])
    pts += np.random.randn(n, 3) * noise
    return pts

# ── PCA-Alignment (spiegelt align_npc_cluster_pca wider) ──────────────────────
def align_pca(pts):
    c = pts.mean(0)
    cen = pts - c
    evals, evecs = np.linalg.eigh(np.cov(cen, rowvar=False))
    idx = np.argsort(evals)[::-1]
    R = evecs[:, idx].T
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return cen @ R.T, c, R, evals[idx]

# ── NPC-Modell (8-fache Symmetrie, 2 Ringe) ───────────────────────────────────
def make_npc_model(r=50., sep=50., n_sym=8):
    a = np.linspace(0, 2*np.pi, n_sym, endpoint=False)
    r1 = np.column_stack([r*np.cos(a), r*np.sin(a), np.full(n_sym, -sep/2)])
    r2 = np.column_stack([r*np.cos(a), r*np.sin(a), np.full(n_sym,  sep/2)])
    return r1, r2

raw = make_tilted_ring()
aligned, centroid, R, evals = align_pca(raw)
cen_raw = raw - raw.mean(0)
m_r1, m_r2 = make_npc_model()

# ── Farben ────────────────────────────────────────────────────────────────────
c_data  = '#E07B39'   # orange – Lokalisierungen
c_model = '#4B9CD3'   # blau   – Modell
c_pc1, c_pc2, c_pc3 = '#E05A6F', '#56B89A', '#A97BDB'
BG = '#0d1117'; PANEL = '#161b22'; TICK = '#8b949e'

fig = plt.figure(figsize=(22, 10), facecolor=BG)

def ax3d(pos):
    ax = fig.add_subplot(pos, projection='3d')
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK, labelsize=6)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#30363d')
    return ax

def ax2d(pos):
    ax = fig.add_subplot(pos)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    return ax

theta_c = np.linspace(0, 2*np.pi, 300)

# ── a) Rohdaten 3D ────────────────────────────────────────────────────────────
ax = ax3d(241)
ax.scatter(raw[:,0], raw[:,1], raw[:,2], c=c_data, s=6, alpha=0.75)
ax.set_title('a) Raw SMLM data (tilted)', color='white', fontsize=10, pad=3)
ax.set_xlabel('x', color=TICK, fontsize=7); ax.set_ylabel('y', color=TICK, fontsize=7)
ax.set_zlabel('z', color=TICK, fontsize=7)

# ── b) Zentrierte Daten ───────────────────────────────────────────────────────
ax = ax3d(242)
ax.scatter(cen_raw[:,0], cen_raw[:,1], cen_raw[:,2], c=c_data, s=6, alpha=0.75)
ax.set_title('b) Centered (–centroid)', color='white', fontsize=10, pad=3)
ax.set_xlabel('x', color=TICK, fontsize=7); ax.set_ylabel('y', color=TICK, fontsize=7)
ax.set_zlabel('z', color=TICK, fontsize=7)

# ── c) PCA-Eigenvektoren ──────────────────────────────────────────────────────
ax = ax3d(243)
ax.scatter(cen_raw[:,0], cen_raw[:,1], cen_raw[:,2], c=c_data, s=3, alpha=0.25)
for i, (col, lbl, sc) in enumerate(zip(
        [c_pc1, c_pc2, c_pc3], ['PC1', 'PC2', 'PC3'], [60., 40., 18.])):
    v = R[i] * sc
    ax.quiver(0,0,0, v[0],v[1],v[2], color=col, linewidth=2.5,
              arrow_length_ratio=0.25, label=lbl)
ax.legend(fontsize=7, facecolor=BG, labelcolor='white', framealpha=0.8, loc='upper left')
ax.set_title('c) PCA eigenvectors', color='white', fontsize=10, pad=3)
ax.set_xlabel('x', color=TICK, fontsize=7); ax.set_ylabel('y', color=TICK, fontsize=7)
ax.set_zlabel('z', color=TICK, fontsize=7)

# ── d) Ausgerichtete Daten + Modell 3D ───────────────────────────────────────
ax = ax3d(244)
ax.scatter(aligned[:,0], aligned[:,1], aligned[:,2], c=c_data, s=6, alpha=0.6,
           label='Localizations')
for rp in [m_r1, m_r2]:
    xs = np.append(rp[:,0], rp[0,0])
    ys = np.append(rp[:,1], rp[0,1])
    zs = np.append(rp[:,2], rp[0,2])
    ax.plot(xs, ys, zs, color=c_model, lw=2)
    ax.scatter(rp[:,0], rp[:,1], rp[:,2], c=c_model, s=22, zorder=5)
ax.set_title('d) Aligned data + NPC model', color='white', fontsize=10, pad=3)
ax.set_xlabel('x', color=TICK, fontsize=7); ax.set_ylabel('y', color=TICK, fontsize=7)
ax.set_zlabel('z', color=TICK, fontsize=7)

# ── e) Modell-PDF ─────────────────────────────────────────────────────────────
ax = ax2d(245)
g = np.linspace(-85, 85, 120)
xx, yy = np.meshgrid(g, g)
pdf = np.zeros_like(xx)
for t in np.linspace(0, 2*np.pi, 300)[::5]:
    pdf += np.exp(-((xx - 50*np.cos(t))**2 + (yy - 50*np.sin(t))**2) / (2*8.**2))
pdf /= pdf.max()
ax.contourf(xx, yy, pdf, levels=20, cmap='Blues')
ax.set_title('e) Model PDF M(x, p)', color='white', fontsize=10)
ax.set_xlabel('x (nm)', color=TICK, fontsize=8); ax.set_ylabel('y (nm)', color=TICK, fontsize=8)
ax.set_aspect('equal')

# ── f) Overlay Daten + Modell ─────────────────────────────────────────────────
ax = ax2d(246)
ax.scatter(aligned[:,0], aligned[:,1], c=c_data, s=8, alpha=0.55, label='Localizations')
ax.plot(50*np.cos(theta_c), 50*np.sin(theta_c), color=c_model, lw=2, label='Ring model')
for rp in [m_r1, m_r2]:
    ax.scatter(rp[:,0], rp[:,1], c=c_model, s=45, zorder=5, marker='o')
ax.legend(fontsize=7, facecolor=BG, labelcolor='white', framealpha=0.8)
ax.set_title('f) Data + Model overlay', color='white', fontsize=10)
ax.set_xlabel('x (nm)', color=TICK, fontsize=8); ax.set_ylabel('y (nm)', color=TICK, fontsize=8)
ax.set_aspect('equal')

# ── g) Top-Ansicht (XY) ───────────────────────────────────────────────────────
ax = ax2d(247)
ax.scatter(aligned[:,0], aligned[:,1], c=c_data, s=8, alpha=0.5)
ax.plot(50*np.cos(theta_c), 50*np.sin(theta_c), color=c_model, lw=2, ls='--')
ax.set_title('g) Top view (xy)', color='white', fontsize=10)
ax.set_xlabel('x (nm)', color=TICK, fontsize=8); ax.set_ylabel('y (nm)', color=TICK, fontsize=8)
ax.set_aspect('equal')
ax.text(0.97, 0.97, 'xy', transform=ax.transAxes, color='white', fontsize=9,
        ha='right', va='top',
        bbox=dict(facecolor='#30363d', alpha=0.7, boxstyle='round,pad=0.2'))

# ── h) Seiten-Ansicht (XZ) ────────────────────────────────────────────────────
ax = ax2d(248)
ax.scatter(aligned[:,0], aligned[:,2], c=c_data, s=8, alpha=0.5)
ax.axhline(-25, color=c_model, lw=2, ls='--', label='Ring z = ±25 nm')
ax.axhline( 25, color=c_model, lw=2, ls='--')
ax.legend(fontsize=7, facecolor=BG, labelcolor='white', framealpha=0.8)
ax.set_title('h) Side view (xz)', color='white', fontsize=10)
ax.set_xlabel('x (nm)', color=TICK, fontsize=8); ax.set_ylabel('z (nm)', color=TICK, fontsize=8)
ax.text(0.97, 0.97, 'xz', transform=ax.transAxes, color='white', fontsize=9,
        ha='right', va='top',
        bbox=dict(facecolor='#30363d', alpha=0.7, boxstyle='round,pad=0.2'))

fig.suptitle(
    'PCA Alignment of SMLM Data to NPC Model  |  align_npc_cluster_pca()  '
    '— analog to Wu et al. 2023, Fig. 1a–h',
    color='white', fontsize=13, y=1.01)

plt.tight_layout()
plt.savefig('examples/figures/methodology/pca_alignment_wu_style.png', dpi=180,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: pca_alignment_wu_style.png")