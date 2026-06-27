"""
Final experiment with optimally selected bandwidths on the full 1998-point dataset.

Bandwidth selection rationale (from bandwidth_search.py):
  - Median pairwise ||x-y||² in FB coefs = 0.0046
  - Old bw=0.5 was ~100x too large (all kernel values ≈ 1, no structure)
  - Optimal orbit bw ≈ 0.01–0.05 (median to 10× median heuristic)
  - Optimal bispectrum bw ≈ 1e-8 to 5e-8
"""
import os, pickle, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aspire.basis import FFBBasis2D
from run_experiment import (
    compute_orbit_kernel, compute_gaussian_kernel_matmul,
    compute_bispectrum_all, apply_random_rotations,
    calc_Laplacian, compute_eigenvectors, add_noise
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.environ.get(
    "PAPER_IMAGE_DATA_DIR",
    os.path.join(REPO_ROOT, "projection_data", "new_dataset_5.3.26"),
)
OUTPUT_DIR = os.environ.get("PAPER_IMAGE_FINAL_OUTPUT_DIR", os.path.join(BASE_DIR, "outputs", "final"))
FIGURES_DIR = os.environ.get("PAPER_IMAGE_FINAL_FIGURES_DIR", os.path.join(BASE_DIR, "figures", "final"))
ELL_MAX = 10
NUM_G = 600
BISP_WORKERS = 64

ORBIT_BWS = [0.005, 0.01, 0.025, 0.05]
BISP_BWS = [6e-9, 1.5e-8, 3e-8, 6e-8]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Load ─────────────────────────────────────────────────────────────────────
print("Loading data...")
proj = torch.load(f"{DATA_DIR}/projections-synth.pt", weights_only=False)
angles_pt = torch.load(f"{DATA_DIR}/angles-synth.pt", weights_only=False)
if isinstance(proj, list):
    images = np.array([p.numpy() for p in proj], dtype=np.float64)
else:
    images = proj.numpy().astype(np.float64)
if isinstance(angles_pt, list):
    gt_angles = np.array([a.item() for a in angles_pt], dtype=np.float64)
else:
    gt_angles = angles_pt.numpy().astype(np.float64)

N, H, W = images.shape
print(f"  {N} images of {H}x{W}")

# ─── FB expansion ────────────────────────────────────────────────────────────
print("FFB expansion...")
ffb = FFBBasis2D((H, W), ell_max=ELL_MAX, dtype=float)
t0 = time.time()
fb = ffb.expand(images, tol=1e-2)
print(f"  Done in {time.time()-t0:.1f}s")

coefs = np.array([fb[i].asnumpy().flatten() for i in range(N)])
ang_idx = ffb.angular_indices
sgn_idx = ffb.signs_indices

print("Applying random rotations...")
coefs_rot = apply_random_rotations(coefs, ang_idx, sgn_idx)

colors = np.sin(gt_angles)

def save_fig(v, title, fname):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(np.real(v[1]), np.real(v[2]), c=colors[:len(v[1])],
               cmap='rainbow', s=5, linewidths=0)
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    path = f'{FIGURES_DIR}/{fname}.pdf'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")


# ─── Orbit methods (min, mean) across bandwidth sweep ────────────────────────
for method in ['min', 'mean']:
    for bw in ORBIT_BWS:
        tag = f'{method}_bw{bw}'
        pkl = f'{OUTPUT_DIR}/eigvectors_{tag}.pkl'
        if os.path.exists(pkl):
            print(f"  [{tag}] exists, skipping computation")
            with open(pkl, 'rb') as f:
                v = pickle.load(f)
        else:
            print(f"\n  [{tag}] Computing orbit kernel...")
            t0 = time.time()
            W = compute_orbit_kernel(coefs_rot, ang_idx, sgn_idx, bw, NUM_G, method)
            S = calc_Laplacian(W)
            v, lam = compute_eigenvectors(S)
            print(f"    Done in {time.time()-t0:.1f}s")
            with open(pkl, 'wb') as f:
                pickle.dump(v, f)

        save_fig(v, f'{method}, bw={bw}', tag)

# ─── Bispectrum across bandwidth sweep ────────────────────────────────────────
bisp_pkl = f'{OUTPUT_DIR}/bispectrum_features.pkl'
if os.path.exists(bisp_pkl):
    print("\n  Loading cached bispectrum features...")
    with open(bisp_pkl, 'rb') as f:
        bisp = pickle.load(f)
else:
    print("\n  Computing bispectrum features (parallel)...")
    t0 = time.time()
    bisp = compute_bispectrum_all(coefs_rot, H, ELL_MAX, BISP_WORKERS)
    print(f"    Done in {time.time()-t0:.1f}s, shape={bisp.shape}")
    with open(bisp_pkl, 'wb') as f:
        pickle.dump(bisp, f)

for bw in BISP_BWS:
    tag = f'bispectrum_bw{bw:.1e}'
    pkl = f'{OUTPUT_DIR}/eigvectors_{tag}.pkl'
    if os.path.exists(pkl):
        print(f"  [{tag}] exists, skipping")
        with open(pkl, 'rb') as f:
            v = pickle.load(f)
    else:
        print(f"\n  [{tag}] Computing Gaussian kernel...")
        t0 = time.time()
        W = compute_gaussian_kernel_matmul(bisp, bw)
        S = calc_Laplacian(W)
        v, lam = compute_eigenvectors(S)
        print(f"    Done in {time.time()-t0:.1f}s")
        with open(pkl, 'wb') as f:
            pickle.dump(v, f)

    save_fig(v, f'bispectrum, bw={bw:.1e}', tag)

# ─── Summary comparison grid ─────────────────────────────────────────────────
print("\nGenerating comparison grid...")
all_results = []
for method in ['min', 'mean']:
    for bw in ORBIT_BWS:
        pkl = f'{OUTPUT_DIR}/eigvectors_{method}_bw{bw}.pkl'
        if os.path.exists(pkl):
            with open(pkl, 'rb') as f:
                v = pickle.load(f)
            all_results.append((f'{method}\nbw={bw}', v))

for bw in BISP_BWS:
    tag = f'bispectrum_bw{bw:.1e}'
    pkl = f'{OUTPUT_DIR}/eigvectors_{tag}.pkl'
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            v = pickle.load(f)
        all_results.append((f'bispectrum\nbw={bw:.1e}', v))

ncols = 4
nrows = (len(all_results) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
axes = np.atleast_2d(axes)
for idx, (label, v) in enumerate(all_results):
    r, c = divmod(idx, ncols)
    ax = axes[r, c]
    ax.scatter(np.real(v[1]), np.real(v[2]), c=colors[:len(v[1])],
               cmap='rainbow', s=3, linewidths=0)
    ax.set_title(label, fontsize=9)
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
for idx in range(len(all_results), nrows * ncols):
    r, c = divmod(idx, ncols)
    axes[r, c].set_visible(False)
plt.tight_layout()
grid_path = f'{FIGURES_DIR}/final_comparison_grid.pdf'
fig.savefig(grid_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Grid: {grid_path}")

# Save angles
with open(f'{OUTPUT_DIR}/ground_truth_angles.pkl', 'wb') as f:
    pickle.dump(gt_angles, f)

print("\n" + "="*60)
print("ALL DONE — check figures/final/ for results")
print("="*60)
