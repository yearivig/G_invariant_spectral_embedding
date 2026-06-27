"""
Bandwidth selection for the G-invariant kernel.

Best practice: the bandwidth σ in K(x,y) = exp(-||x-y||²/σ) should be
chosen so that the kernel values are neither all ~1 (over-connected graph,
no structure) nor all ~0 (disconnected graph, no diffusion).

Rule of thumb (median heuristic): set σ = median(||x_i - x_j||²).
More refined: sweep σ over a range and evaluate embedding quality via
the spectral gap (λ₁ - λ₂) of the normalized Laplacian, or visual
inspection of the eigenvector scatter colored by ground truth.

This script:
  1. Computes pairwise distance statistics for the FB coefficients
  2. Computes pairwise distance statistics for the orbit-min distances
  3. Runs a bandwidth sweep for each method
  4. Saves eigenvectors + figures for each bandwidth
"""
import os
import pickle
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aspire.basis import FFBBasis2D, Coef
from numpy.linalg import linalg
from tqdm import tqdm
from multiprocessing import Pool

# ─── Import optimized kernels from run_experiment ─────────────────────────────
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
OUTPUT_DIR = os.environ.get("PAPER_BW_OUTPUT_DIR", os.path.join(BASE_DIR, "outputs", "bw_sweep"))
FIGURES_DIR = os.environ.get("PAPER_BW_FIGURES_DIR", os.path.join(BASE_DIR, "figures", "bw_sweep"))
ELL_MAX = 10
NUM_G = 600
BISP_WORKERS = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
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

# ─── Expand in FB basis ──────────────────────────────────────────────────────
print("Expanding in FFBBasis2D...")
ffbbasis = FFBBasis2D((H, W), ell_max=ELL_MAX, dtype=float)
t0 = time.time()
fb = ffbbasis.expand(images, tol=1e-2)
print(f"  Done in {time.time()-t0:.1f}s")

coefs = np.array([fb[i].asnumpy().flatten() for i in range(N)])
ang_idx = ffbbasis.angular_indices
sgn_idx = ffbbasis.signs_indices

print("Applying random rotations...")
coefs_rot = apply_random_rotations(coefs, ang_idx, sgn_idx)

# ─── Step 1: Pairwise distance statistics ─────────────────────────────────────
print("\n" + "="*60)
print("PAIRWISE DISTANCE ANALYSIS")
print("="*60)

# Standard (non-orbit) squared distances
norms = np.sum(coefs_rot**2, axis=1)
dot = coefs_rot @ coefs_rot.T
std_dist_sq = norms[:, None] + norms[None, :] - 2 * dot
np.fill_diagonal(std_dist_sq, np.inf)
std_vals = std_dist_sq[np.triu_indices(N, k=1)]

print(f"\nStandard ||x-y||² (FB coefficients):")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {p}th percentile: {np.percentile(std_vals, p):.4f}")
print(f"  median: {np.median(std_vals):.4f}")
print(f"  mean:   {np.mean(std_vals):.4f}")

# Orbit-min distances (sample for speed)
print("\nComputing orbit-min distances on a 200-point subsample...")
sub_n = min(200, N)
sub_coefs = coefs_rot[:sub_n]
W_min_sample = compute_orbit_kernel(sub_coefs, ang_idx, sgn_idx, 1.0, NUM_G, 'min')
orbit_dist_sq = -np.log(np.clip(W_min_sample, 1e-300, None))
np.fill_diagonal(orbit_dist_sq, np.inf)
orbit_vals = orbit_dist_sq[np.triu_indices(sub_n, k=1)]

print(f"\nOrbit-min -log K(x,y) with bw=1 (proxy for orbit distance):")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {p}th percentile: {np.percentile(orbit_vals, p):.4f}")

# Bispectrum distances (sample)
print("\nComputing bispectrum for 200-point subsample...")
basis_params = (H, ELL_MAX)
bisp_sub = compute_bispectrum_all(sub_coefs, H, ELL_MAX, BISP_WORKERS)
if np.iscomplexobj(bisp_sub):
    bisp_real = np.hstack([bisp_sub.real, bisp_sub.imag]).astype(np.float64)
else:
    bisp_real = bisp_sub.astype(np.float64)
bisp_norms = np.sum(bisp_real**2, axis=1)
bisp_dot = bisp_real @ bisp_real.T
bisp_dist_sq = bisp_norms[:, None] + bisp_norms[None, :] - 2 * bisp_dot
np.maximum(bisp_dist_sq, 0, out=bisp_dist_sq)
np.fill_diagonal(bisp_dist_sq, np.inf)
bisp_vals = bisp_dist_sq[np.triu_indices(sub_n, k=1)]

print(f"\nBispectrum ||B(x)-B(y)||²:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {p}th percentile: {np.percentile(bisp_vals, p):.6g}")
print(f"  median: {np.median(bisp_vals):.6g}")

# ─── Step 2: Suggest bandwidths ──────────────────────────────────────────────
median_std = np.median(std_vals)
median_bisp = np.median(bisp_vals)

orbit_bw_candidates = [median_std * f for f in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]]
bisp_bw_candidates = [median_bisp * f for f in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]]

print(f"\n{'='*60}")
print(f"SUGGESTED BANDWIDTH CANDIDATES")
print(f"{'='*60}")
print(f"  Orbit methods (median ||x-y||² = {median_std:.4f}):")
for bw in orbit_bw_candidates:
    print(f"    bw = {bw:.6f}")
print(f"  Bispectrum (median ||B(x)-B(y)||² = {median_bisp:.6g}):")
for bw in bisp_bw_candidates:
    print(f"    bw = {bw:.6g}")

# ─── Step 3: Bandwidth sweep ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("BANDWIDTH SWEEP (min method, 200-point subsample for speed)")
print(f"{'='*60}")

colors = np.sin(gt_angles[:sub_n])

def spectral_gap(eigvals):
    """Ratio λ₂/λ₃ — larger means better separation of the leading eigenvector."""
    s = np.sort(np.abs(eigvals))[::-1]
    if len(s) >= 3 and s[2] > 1e-15:
        return np.abs(s[1]) / np.abs(s[2])
    return 0.0

results_orbit = []
for bw in orbit_bw_candidates:
    print(f"\n  bw = {bw:.6f}")
    W = compute_orbit_kernel(sub_coefs, ang_idx, sgn_idx, bw, NUM_G, 'min')
    S = calc_Laplacian(W)
    v, lam = compute_eigenvectors(S)
    gap = spectral_gap(lam)
    results_orbit.append((bw, gap))
    print(f"    spectral gap λ₂/λ₃ = {gap:.4f}")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(np.real(v[1]), np.real(v[2]), c=colors, cmap='rainbow', s=8, linewidths=0)
    ax.set_title(f'min, bw={bw:.4g}, gap={gap:.3f}', fontsize=9)
    ax.set_frame_on(False); ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    fig.savefig(f'{FIGURES_DIR}/sweep_min_bw{bw:.6f}.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)

results_bisp = []
for bw in bisp_bw_candidates:
    print(f"\n  bw = {bw:.6g} (bispectrum)")
    W = np.exp(-bisp_dist_sq / bw)
    np.fill_diagonal(W, 1.0)
    S = calc_Laplacian(W)
    v, lam = compute_eigenvectors(S)
    gap = spectral_gap(lam)
    results_bisp.append((bw, gap))
    print(f"    spectral gap λ₂/λ₃ = {gap:.4f}")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(np.real(v[1]), np.real(v[2]), c=colors[:sub_n], cmap='rainbow', s=8, linewidths=0)
    ax.set_title(f'bisp, bw={bw:.4g}, gap={gap:.3f}', fontsize=9)
    ax.set_frame_on(False); ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    fig.savefig(f'{FIGURES_DIR}/sweep_bisp_bw{bw:.6g}.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)

# ─── Step 4: Summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY — spectral gap λ₂/λ₃ (higher = better separation)")
print(f"{'='*60}")

print("\nOrbit-min method:")
best_orbit = max(results_orbit, key=lambda x: x[1])
for bw, gap in results_orbit:
    marker = " <<<" if bw == best_orbit[0] else ""
    print(f"  bw = {bw:12.6f}  →  gap = {gap:.4f}{marker}")

print("\nBispectrum method:")
best_bisp = max(results_bisp, key=lambda x: x[1])
for bw, gap in results_bisp:
    marker = " <<<" if bw == best_bisp[0] else ""
    print(f"  bw = {bw:12.6g}  →  gap = {gap:.4f}{marker}")

print(f"\nBest orbit bandwidth:     {best_orbit[0]:.6f} (gap={best_orbit[1]:.4f})")
print(f"Best bispectrum bandwidth: {best_bisp[0]:.6g} (gap={best_bisp[1]:.4f})")

with open(f'{OUTPUT_DIR}/bandwidth_results.pkl', 'wb') as f:
    pickle.dump({
        'orbit_results': results_orbit,
        'bisp_results': results_bisp,
        'best_orbit_bw': best_orbit[0],
        'best_bisp_bw': best_bisp[0],
        'orbit_bw_candidates': orbit_bw_candidates,
        'bisp_bw_candidates': bisp_bw_candidates,
        'median_std_dist': median_std,
        'median_bisp_dist': median_bisp,
    }, f)

print(f"\nResults saved to {OUTPUT_DIR}/bandwidth_results.pkl")
print(f"Sweep figures saved to {FIGURES_DIR}/")
print("Inspect the sweep figures to visually confirm the best bandwidth.")
