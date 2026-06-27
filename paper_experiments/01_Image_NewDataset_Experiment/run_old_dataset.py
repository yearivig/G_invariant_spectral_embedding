"""
Run the optimized experiment on the OLD dataset (unrotated_projections-synth.pkl).
Same bandwidth sweep as run_final.py, for direct comparison.

Old dataset: 198 images, 82x82, angles in degrees.
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aspire.basis import FFBBasis2D, Coef
from numpy.linalg import linalg
from tqdm import tqdm
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from run_experiment import (
    compute_orbit_kernel, compute_gaussian_kernel_matmul,
    apply_random_rotations, calc_Laplacian, compute_eigenvectors,
    add_noise
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
OUTPUT_DIR = os.environ.get("PAPER_OLD_OUTPUT_DIR", os.path.join(BASE_DIR, "outputs", "old_dataset"))
FIGURES_DIR = os.environ.get("PAPER_OLD_FIGURES_DIR", os.path.join(BASE_DIR, "figures", "old_dataset"))
ELL_MAX = 10
NUM_G = 600
BISP_WORKERS = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Load old dataset ────────────────────────────────────────────────────────
print("Loading OLD dataset...")
proj_path = os.environ.get(
    "PAPER_OLD_PROJECTIONS_PATH",
    os.path.join(REPO_ROOT, "projection_data", "unrotated", "unrotated_projections-synth.pkl"),
)
with open(proj_path, 'rb') as f:
    images = np.array(pickle.load(f), dtype=np.float64)

angles_csv = os.environ.get("PAPER_OLD_ANGLES_CSV", os.path.join(REPO_ROOT, "data", "angles.csv"))
df = pd.read_csv(angles_csv)
df = df.drop(df.columns[0], axis=1)
gt_angles_deg = np.array(df.values.flatten())[2:]

N, H, W = images.shape
print(f"  {N} images of {H}x{W}")
print(f"  Angles: {len(gt_angles_deg)} values in degrees [{gt_angles_deg.min():.1f}, {gt_angles_deg.max():.1f}]")
colors = np.sin(gt_angles_deg * np.pi / 180)

# ─── FB expansion ────────────────────────────────────────────────────────────
print(f"\nFFBBasis2D({H}x{W}, ell_max={ELL_MAX})")
ffb = FFBBasis2D((H, W), ell_max=ELL_MAX, dtype=float)
print(f"  Coefficients: {ffb.count}")

t0 = time.time()
fb = ffb.expand(images, tol=1e-2)
print(f"  Expanded in {time.time()-t0:.1f}s")

coefs = np.array([fb[i].asnumpy().flatten() for i in range(N)])
ang_idx = ffb.angular_indices
sgn_idx = ffb.signs_indices

print("Applying random rotations...")
coefs_rot = apply_random_rotations(coefs, ang_idx, sgn_idx)

# ─── Distance statistics ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("PAIRWISE DISTANCE ANALYSIS (old dataset)")
print("="*60)

norms = np.sum(coefs_rot**2, axis=1)
dot = coefs_rot @ coefs_rot.T
std_dist_sq = norms[:, None] + norms[None, :] - 2 * dot
np.fill_diagonal(std_dist_sq, np.inf)
vals = std_dist_sq[np.triu_indices(N, k=1)]
median_dist = np.median(vals)

print(f"  median ||x-y||² = {median_dist:.6f}")
for p in [5, 25, 50, 75, 95]:
    print(f"  {p}th percentile: {np.percentile(vals, p):.6f}")

# ─── Bispectrum features ─────────────────────────────────────────────────────
def _bisp_worker(args):
    idx, coef_array, img_size, ell_max = args
    ffb_local = FFBBasis2D((img_size, img_size), ell_max=ell_max, dtype=float)
    c = Coef(ffb_local, coef_array[np.newaxis, :])
    return ffb_local.calculate_bispectrum(c, flatten=True).flatten()

print("\nComputing bispectrum features...")
t0 = time.time()
args = [(k, coefs_rot[k], H, ELL_MAX) for k in range(N)]
with Pool(processes=min(BISP_WORKERS, N)) as pool:
    bisp_list = list(tqdm(pool.imap(_bisp_worker, args), total=N, desc="  Bispectrum"))
bisp = np.array(bisp_list)
print(f"  Done in {time.time()-t0:.1f}s, shape={bisp.shape}")

if np.iscomplexobj(bisp):
    bisp_real = np.hstack([bisp.real, bisp.imag]).astype(np.float64)
else:
    bisp_real = bisp.astype(np.float64)

bisp_norms = np.sum(bisp_real**2, axis=1)
bisp_dot = bisp_real @ bisp_real.T
bisp_dist_sq = bisp_norms[:, None] + bisp_norms[None, :] - 2 * bisp_dot
np.maximum(bisp_dist_sq, 0, out=bisp_dist_sq)
np.fill_diagonal(bisp_dist_sq, np.inf)
bisp_vals = bisp_dist_sq[np.triu_indices(N, k=1)]
median_bisp = np.median(bisp_vals)
print(f"  median ||B(x)-B(y)||² = {median_bisp:.6g}")

# ─── Bandwidth sweep ─────────────────────────────────────────────────────────
orbit_bws = [median_dist * f for f in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]]
bisp_bws = [median_bisp * f for f in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]]

# Also include the original bw=0.5 and bw=50 for reference
orbit_bws = sorted(set(orbit_bws + [0.5, 50.0]))
bisp_bws = sorted(set(bisp_bws))

def spectral_gap(eigvals):
    s = np.sort(np.abs(eigvals))[::-1]
    if len(s) >= 3 and s[2] > 1e-15:
        return np.abs(s[1]) / np.abs(s[2])
    return 0.0

def save_fig(v, title, fname):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    c = colors[:len(np.real(v[1]))]
    ax.scatter(np.real(v[1]), np.real(v[2]), c=c, cmap='rainbow', s=12, linewidths=0)
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    path = f'{FIGURES_DIR}/{fname}.pdf'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"\n{'='*60}")
print("BANDWIDTH SWEEP — orbit-min method")
print(f"{'='*60}")

orbit_results = []
for bw in orbit_bws:
    print(f"\n  bw = {bw:.6g}")
    W = compute_orbit_kernel(coefs_rot, ang_idx, sgn_idx, bw, NUM_G, 'min')
    S = calc_Laplacian(W)
    v, lam = compute_eigenvectors(S)
    gap = spectral_gap(lam)
    orbit_results.append((bw, gap, v))
    print(f"    gap = {gap:.4f}")
    save_fig(v, f'OLD min, bw={bw:.4g}, gap={gap:.3f}', f'min_bw{bw:.6g}')

print(f"\n{'='*60}")
print("BANDWIDTH SWEEP — orbit-mean method")
print(f"{'='*60}")

mean_results = []
for bw in orbit_bws:
    print(f"\n  bw = {bw:.6g}")
    W = compute_orbit_kernel(coefs_rot, ang_idx, sgn_idx, bw, NUM_G, 'mean')
    S = calc_Laplacian(W)
    v, lam = compute_eigenvectors(S)
    gap = spectral_gap(lam)
    mean_results.append((bw, gap, v))
    print(f"    gap = {gap:.4f}")
    save_fig(v, f'OLD mean, bw={bw:.4g}, gap={gap:.3f}', f'mean_bw{bw:.6g}')

print(f"\n{'='*60}")
print("BANDWIDTH SWEEP — bispectrum method")
print(f"{'='*60}")

bisp_results = []
for bw in bisp_bws:
    print(f"\n  bw = {bw:.6g}")
    W = np.exp(-bisp_dist_sq / bw)
    np.fill_diagonal(W, 1.0)
    S = calc_Laplacian(W)
    v, lam = compute_eigenvectors(S)
    gap = spectral_gap(lam)
    bisp_results.append((bw, gap, v))
    print(f"    gap = {gap:.4f}")
    save_fig(v, f'OLD bisp, bw={bw:.4g}, gap={gap:.3f}', f'bisp_bw{bw:.6g}')

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY (old dataset, 198 images 82x82)")
print(f"{'='*60}")

print("\nOrbit-min:")
best_min = max(orbit_results, key=lambda x: x[1])
for bw, gap, _ in orbit_results:
    m = " <<<" if bw == best_min[0] else ""
    print(f"  bw={bw:12.6g}  gap={gap:.4f}{m}")

print("\nOrbit-mean:")
best_mean = max(mean_results, key=lambda x: x[1])
for bw, gap, _ in mean_results:
    m = " <<<" if bw == best_mean[0] else ""
    print(f"  bw={bw:12.6g}  gap={gap:.4f}{m}")

print("\nBispectrum:")
best_bisp = max(bisp_results, key=lambda x: x[1])
for bw, gap, _ in bisp_results:
    m = " <<<" if bw == best_bisp[0] else ""
    print(f"  bw={bw:12.6g}  gap={gap:.4f}{m}")

# ─── Save best results ───────────────────────────────────────────────────────
for label, best in [('min', best_min), ('mean', best_mean), ('bisp', best_bisp)]:
    bw, gap, v = best
    pkl = f'{OUTPUT_DIR}/eigvectors_best_{label}_bw{bw:.6g}.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump(v, f)
    print(f"\nSaved best {label}: bw={bw:.6g}, gap={gap:.4f} → {pkl}")

with open(f'{OUTPUT_DIR}/ground_truth_angles_deg.pkl', 'wb') as f:
    pickle.dump(gt_angles_deg, f)

# ─── Comparison grid ──────────────────────────────────────────────────────────
print("\nGenerating comparison grid...")
grid_items = []
for bw, gap, v in orbit_results:
    grid_items.append((f'min bw={bw:.4g}\ngap={gap:.3f}', v))
for bw, gap, v in bisp_results:
    grid_items.append((f'bisp bw={bw:.4g}\ngap={gap:.3f}', v))

ncols = 5
nrows = (len(grid_items) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows))
axes = np.atleast_2d(axes)
for idx, (label, v) in enumerate(grid_items):
    r, c = divmod(idx, ncols)
    ax = axes[r, c]
    ax.scatter(np.real(v[1]), np.real(v[2]), c=colors[:len(np.real(v[1]))],
               cmap='rainbow', s=10, linewidths=0)
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
for idx in range(len(grid_items), nrows * ncols):
    r, c = divmod(idx, ncols)
    axes[r, c].set_visible(False)
plt.tight_layout()
grid_path = f'{FIGURES_DIR}/old_dataset_comparison_grid.pdf'
fig.savefig(grid_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Grid: {grid_path}")

print(f"\n{'='*60}")
print(f"ALL DONE — figures in {FIGURES_DIR}/")
print(f"{'='*60}")
