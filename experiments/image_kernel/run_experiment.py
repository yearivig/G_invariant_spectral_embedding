"""
Optimized version of apply_method.py for the new dataset (new_dataset_5.3.26).

Original: apply_method.py (O(N²·G) Python loops, ~120+ hours for N=1998)
This version: vectorized trigonometric decomposition + BLAS matmul (~minutes)

Dataset: 1998 synthetic cryo-EM projections, 79x79 pixels
         Ground-truth bond rotation angles in radians [0, 2π]
Basis:   FFBBasis2D, ell_max=10, 769 real coefficients

Methods:
  - min:        max_θ exp(-||x - R(θ)y||² / bw)    (orbit sup-kernel)
  - mean:       E_θ  exp(-||x - R(θ)y||² / bw)     (orbit mean-kernel)
  - bispectrum: exp(-||B(x) - B(y)||² / bw)         (rotation-invariant)
"""
import os
import pickle
import time
import numpy as np
import torch
from aspire.basis import FFBBasis2D, Coef
from numpy.linalg import linalg
from tqdm import tqdm
from multiprocessing import Pool

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.environ.get(
    "PAPER_IMAGE_DATA_DIR",
    os.path.join(REPO_ROOT, "projection_data", "new_dataset_5.3.26"),
)
OUTPUT_DIR = os.environ.get("PAPER_IMAGE_OUTPUT_DIR", "outputs")
BANDWIDTH = 0.5
ELL_MAX = 10
NUM_G_ELEMENTS = 600       # discretization of SO(2) for orbit kernel
SNR_LEVELS = [0]            # 0 = no noise
METHODS = ["min", "mean", "bispectrum"]
BISPECTRUM_WORKERS = 64
ORBIT_CHUNK_SIZE = 100      # rows per chunk for orbit kernel
BISP_KERNEL_CHUNK = 50      # rows per chunk for bispectrum kernel matmul

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Core functions ───────────────────────────────────────────────────────────

def calc_Laplacian(W):
    """Random-walk normalized graph Laplacian: D^{-1} W."""
    d = W @ np.ones(W.shape[0])
    D_inv = np.diag(1.0 / d)
    return D_inv @ W


def compute_eigenvectors(matrix):
    """
    Eigen-decomposition for diffusion-map coordinates.

    Steps:
      1) Solve right-eigenvectors of the row-stochastic operator.
      2) Sort by descending |lambda| so lambda_0~1 is first.
      3) Apply Lafon normalization: psi_l(i) = phi_l(i) / phi_0(i).
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Columns of eigenvectors are right-eigenvectors.
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Row-major layout used downstream: eigvecs[l, i] is l-th mode at point i.
    eigvecs = eigenvectors.T

    # Lafon normalization by the trivial mode phi_0 over data points.
    phi0 = np.real(eigvecs[0])
    eps = 1e-12
    phi0_safe = np.where(np.abs(phi0) < eps, eps, phi0)
    eigvecs = eigvecs / phi0_safe[None, :]

    return eigvecs, eigenvalues


def add_noise(images, snr):
    """Add Gaussian noise at a given SNR (dB). SNR=0 means no noise."""
    if snr == 0:
        return images.copy()
    signal_power = np.mean(images ** 2, axis=(1, 2), keepdims=True)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.normal(0, 1, images.shape) * np.sqrt(noise_power)
    return images + noise


# ─── Orbit kernel (min / mean) via trigonometric decomposition ────────────────

def compute_orbit_kernel(coefs, angular_indices, signs_indices,
                         bandwidth, num_g, method):
    """
    Vectorized SO(2)-invariant kernel matrix.

    Key insight: in the Fourier-Bessel basis, rotation by θ acts as a
    block-diagonal 2x2 rotation on each angular frequency ℓ. Therefore:

        x · R(θ)y = C₀ + Σ_ℓ [A_ℓ cos(ℓθ) + B_ℓ sin(ℓθ)]

    where A_ℓ, B_ℓ are computed via matrix multiplication over all (i,j) pairs.
    This replaces the O(N²·G) Python loop with O(L) BLAS matmuls.
    """
    N = coefs.shape[0]
    ell_max = int(np.max(angular_indices))

    print(f"  N={N}, D={coefs.shape[1]}, G={num_g}, method={method}")
    t0 = time.time()

    norms_sq = np.sum(coefs ** 2, axis=1)
    angles = np.arange(num_g) * (2 * np.pi / num_g)

    # ℓ=0 block (invariant to rotation)
    mask_0 = angular_indices == 0
    C = coefs[:, mask_0] @ coefs[:, mask_0].T

    # ℓ>0 blocks: A_ℓ = <x_cos, y_cos> + <x_sin, y_sin>
    #             B_ℓ = <x_sin, y_cos> - <x_cos, y_sin>
    A_stack = np.zeros((N, N, ell_max), dtype=np.float64)
    B_stack = np.zeros((N, N, ell_max), dtype=np.float64)

    for ell in range(1, ell_max + 1):
        cos_mask = (angular_indices == ell) & (signs_indices == 1)
        sin_mask = (angular_indices == ell) & (signs_indices == -1)
        x_cos = coefs[:, cos_mask]
        x_sin = coefs[:, sin_mask]
        A_stack[:, :, ell - 1] = x_cos @ x_cos.T + x_sin @ x_sin.T
        B_stack[:, :, ell - 1] = x_sin @ x_cos.T - x_cos @ x_sin.T

    # Precompute trig tables: cos(ℓθ_m), sin(ℓθ_m)
    ells = np.arange(1, ell_max + 1)
    cos_table = np.cos(np.outer(ells, angles))  # (L, G)
    sin_table = np.sin(np.outer(ells, angles))  # (L, G)

    print(f"  Precomputation: {time.time() - t0:.1f}s")

    W = np.zeros((N, N), dtype=np.float64)
    for i0 in tqdm(range(0, N, ORBIT_CHUNK_SIZE), desc=f"  {method} kernel"):
        i1 = min(i0 + ORBIT_CHUNK_SIZE, N)

        dot = (C[i0:i1, :, None]
               + np.einsum('ijk,km->ijm', A_stack[i0:i1], cos_table)
               + np.einsum('ijk,km->ijm', B_stack[i0:i1], sin_table))

        dist_sq = norms_sq[i0:i1, None, None] + norms_sq[None, :, None] - 2 * dot
        K = np.exp(-dist_sq / bandwidth)

        if method == 'min':
            W[i0:i1, :] = np.max(K, axis=2)
        else:
            W[i0:i1, :] = np.mean(K, axis=2)

    print(f"  Total: {time.time() - t0:.1f}s")
    return W


# ─── Bispectrum kernel via BLAS matmul ────────────────────────────────────────

def _bispectrum_worker(args):
    """Compute bispectrum for one image (used in multiprocessing)."""
    idx, coef_array, img_size, ell_max = args
    ffb = FFBBasis2D((img_size, img_size), ell_max=ell_max, dtype=float)
    coef = Coef(ffb, coef_array[np.newaxis, :])
    return ffb.calculate_bispectrum(coef, flatten=True).flatten()


def compute_bispectrum_all(coefs_raw, img_size, ell_max, n_workers):
    """Compute bispectrum for all images in parallel."""
    args = [(k, coefs_raw[k], img_size, ell_max) for k in range(len(coefs_raw))]
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(_bispectrum_worker, args),
            total=len(args), desc="  Bispectrum"))
    return np.array(results)


def compute_gaussian_kernel_matmul(data, bandwidth):
    """
    Gaussian kernel using ||x-y||² = ||x||² + ||y||² - 2x·y.
    The dot product x·y is computed via BLAS matmul across all 192 cores,
    which is orders of magnitude faster than scipy.cdist for high-dim data.
    Complex data is handled by stacking real/imag parts.
    """
    if np.iscomplexobj(data):
        data = np.hstack([data.real, data.imag]).astype(np.float64)
    else:
        data = np.asarray(data, dtype=np.float64)

    N, D = data.shape
    print(f"  Gaussian kernel: N={N}, D={D}")

    norms_sq = np.sum(data ** 2, axis=1)

    W = np.zeros((N, N), dtype=np.float64)
    for i0 in tqdm(range(0, N, BISP_KERNEL_CHUNK), desc="  Kernel matmul"):
        i1 = min(i0 + BISP_KERNEL_CHUNK, N)
        dot_chunk = data[i0:i1] @ data.T  # (chunk, N)
        dist_sq = norms_sq[i0:i1, None] + norms_sq[None, :] - 2 * dot_chunk
        np.maximum(dist_sq, 0, out=dist_sq)
        W[i0:i1, :] = np.exp(-dist_sq / bandwidth)

    return W


# ─── Vectorized rotation in FB coefficient space ─────────────────────────────

def apply_random_rotations(coefs, angular_indices, signs_indices):
    """
    Apply random SO(2) rotations directly in the FB coefficient space.
    For angular frequency ℓ, rotation by θ:
        c_cos' = cos(ℓθ)·c_cos - sin(ℓθ)·c_sin
        c_sin' = sin(ℓθ)·c_cos + cos(ℓθ)·c_sin
    """
    N = coefs.shape[0]
    ell_max = int(np.max(angular_indices))
    thetas = np.random.uniform(0, 2 * np.pi, size=N)
    rotated = coefs.copy()

    for ell in range(1, ell_max + 1):
        cos_mask = (angular_indices == ell) & (signs_indices == 1)
        sin_mask = (angular_indices == ell) & (signs_indices == -1)
        c_cos = coefs[:, cos_mask]
        c_sin = coefs[:, sin_mask]
        cos_ell = np.cos(ell * thetas)[:, None]
        sin_ell = np.sin(ell * thetas)[:, None]
        rotated[:, cos_mask] = cos_ell * c_cos - sin_ell * c_sin
        rotated[:, sin_mask] = sin_ell * c_cos + cos_ell * c_sin

    return rotated


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 1. Load and reformat data (torch .pt → numpy float64) ─────────────
    print("=" * 70)
    print("Loading dataset from", DATA_DIR)
    print("=" * 70)

    projections_pt = torch.load(f"{DATA_DIR}/projections-synth.pt", weights_only=False)
    angles_pt = torch.load(f"{DATA_DIR}/angles-synth.pt", weights_only=False)

    if isinstance(projections_pt, list):
        images = np.array([p.numpy() for p in projections_pt], dtype=np.float64)
    else:
        images = projections_pt.numpy().astype(np.float64)

    if isinstance(angles_pt, list):
        gt_angles = np.array([a.item() for a in angles_pt], dtype=np.float64)
    else:
        gt_angles = angles_pt.numpy().astype(np.float64)

    N, H, W = images.shape
    print(f"  Projections: {N} images of {H}x{W}")
    print(f"  Angles: {N} values in [{gt_angles.min():.4f}, {gt_angles.max():.4f}] rad")

    assert H == W, f"Images must be square, got {H}x{W}"

    # ── 2. Build Fourier-Bessel basis ─────────────────────────────────────
    print(f"\nBuilding FFBBasis2D({H}x{W}, ell_max={ELL_MAX})")
    ffbbasis = FFBBasis2D((H, W), ell_max=ELL_MAX, dtype=float)
    print(f"  Real coefficients: {ffbbasis.count}")
    print(f"  Complex coefficients: {ffbbasis.complex_count}")

    angular_indices = ffbbasis.angular_indices
    signs_indices = ffbbasis.signs_indices

    # ── 3. Loop over SNR levels ───────────────────────────────────────────
    for snr in SNR_LEVELS:
        print(f"\n{'=' * 70}")
        print(f"SNR = {snr} {'(no noise)' if snr == 0 else ''}")
        print(f"{'=' * 70}")

        # ── 3a. Add noise ─────────────────────────────────────────────────
        noisy_images = add_noise(images, snr)

        # ── 3b. Expand into FB basis ──────────────────────────────────────
        print("\nExpanding images in Fourier-Bessel basis...")
        t0 = time.time()
        fb_dataset = ffbbasis.expand(noisy_images, tol=1e-2)
        print(f"  Done in {time.time() - t0:.1f}s")

        # Extract raw coefficient arrays
        coefs_raw = np.array([fb_dataset[i].asnumpy().flatten()
                              for i in range(N)])
        print(f"  Coefficient matrix: {coefs_raw.shape}")

        # ── 3c. Apply random SO(2) rotations ──────────────────────────────
        print("\nApplying random SO(2) rotations...")
        coefs_rotated = apply_random_rotations(
            coefs_raw, angular_indices, signs_indices)

        # ── 3d. Run each method ───────────────────────────────────────────
        for method in METHODS:
            tag = f"snr{snr}_{method}"
            out_path = f"{OUTPUT_DIR}/eigvectors_ell{ELL_MAX}_{method}_bw{BANDWIDTH}_snr{snr}.pkl"

            if os.path.exists(out_path):
                print(f"\n  [{tag}] Output exists, skipping: {out_path}")
                continue

            print(f"\n{'─' * 60}")
            print(f"  Method: {method} | SNR: {snr} | BW: {BANDWIDTH}")
            print(f"{'─' * 60}")
            t_method = time.time()

            if method == 'bispectrum':
                print("  Computing bispectrum (rotation-invariant features)...")
                bisp = compute_bispectrum_all(
                    coefs_rotated, H, ELL_MAX, BISPECTRUM_WORKERS)
                print(f"  Bispectrum array: {bisp.shape}, {bisp.dtype}")
                print(f"  Memory: {bisp.nbytes / 1e9:.1f} GB")

                print("  Computing Gaussian kernel via matmul...")
                W = compute_gaussian_kernel_matmul(bisp, BANDWIDTH)
                del bisp
            else:
                print(f"  Computing orbit kernel ({method})...")
                W = compute_orbit_kernel(
                    coefs_rotated, angular_indices, signs_indices,
                    BANDWIDTH, NUM_G_ELEMENTS, method)

            print("  Computing RW-Laplacian eigenvectors...")
            S = calc_Laplacian(W)
            eigvecs, eigvals = compute_eigenvectors(S)

            elapsed = time.time() - t_method
            print(f"  ✓ {method} completed in {elapsed:.1f}s")

            with open(out_path, 'wb') as f:
                pickle.dump(eigvecs, f)
            print(f"  Saved: {out_path}")

    # ── 4. Save ground-truth angles ───────────────────────────────────────
    angles_path = f"{OUTPUT_DIR}/ground_truth_angles.pkl"
    with open(angles_path, 'wb') as f:
        pickle.dump(gt_angles, f)
    print(f"\nSaved ground-truth angles: {angles_path}")
    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
