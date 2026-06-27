import pickle
import time
import numpy as np
import torch
from aspire.basis import FFBBasis2D
from scipy.spatial.distance import cdist
from numpy.linalg import linalg
from tqdm import tqdm
from multiprocessing import Pool


def calc_Laplacian(W):
    ones = np.ones(W.shape[0])
    v = (W @ ones)
    if (v != 0).any():
        D = np.diag(v)
        D_inv = linalg.inv(D)
        L_rw = D_inv @ W
    return L_rw


def generate_eig(matrix):
    w, v = np.linalg.eig(matrix)
    v = np.transpose(v)
    v = v / v[:, 0][:, None]
    return v, w


def compute_orbit_kernel_vectorized(coefs, angular_indices, signs_indices,
                                    bandwidth, num_g=600, method='min'):
    """
    Vectorized orbit-invariant kernel using trigonometric decomposition.

    For FB coefficients, rotation by θ only affects angular modes:
      x · R(θ)y = C + Σ_ℓ [A_ℓ cos(ℓθ) + B_ℓ sin(ℓθ)]
    where A_ℓ, B_ℓ are computed via matrix multiplication over all pairs.
    This turns the O(N²·G) Python loop into O(L) matrix multiplications.
    """
    N, D = coefs.shape
    ell_max = int(np.max(angular_indices))

    print(f"  Computing orbit kernel: N={N}, D={D}, G={num_g}, method={method}")
    t0 = time.time()

    norms_sq = np.sum(coefs ** 2, axis=1)

    angles = np.arange(num_g) * (2 * np.pi / num_g)

    mask_0 = angular_indices == 0
    C = coefs[:, mask_0] @ coefs[:, mask_0].T

    A_stack = np.zeros((N, N, ell_max), dtype=np.float64)
    B_stack = np.zeros((N, N, ell_max), dtype=np.float64)

    for ell in range(1, ell_max + 1):
        cos_mask = (angular_indices == ell) & (signs_indices == 1)
        sin_mask = (angular_indices == ell) & (signs_indices == -1)
        x_cos = coefs[:, cos_mask]
        x_sin = coefs[:, sin_mask]
        A_stack[:, :, ell - 1] = x_cos @ x_cos.T + x_sin @ x_sin.T
        B_stack[:, :, ell - 1] = x_sin @ x_cos.T - x_cos @ x_sin.T

    ells = np.arange(1, ell_max + 1)
    cos_table = np.cos(np.outer(ells, angles))
    sin_table = np.sin(np.outer(ells, angles))

    print(f"  Precomputation done in {time.time() - t0:.1f}s. Computing kernel in chunks...")

    chunk_size = 100
    W = np.zeros((N, N), dtype=np.float64)

    for i_start in tqdm(range(0, N, chunk_size)):
        i_end = min(i_start + chunk_size, N)
        chunk_C = C[i_start:i_end, :]
        chunk_A = A_stack[i_start:i_end, :, :]
        chunk_B = B_stack[i_start:i_end, :, :]

        # dot[chunk, N, G] = C + Σ_ℓ [A_ℓ·cos(ℓθ) + B_ℓ·sin(ℓθ)]
        dot = (chunk_C[:, :, None]
               + np.einsum('ijk,km->ijm', chunk_A, cos_table)
               + np.einsum('ijk,km->ijm', chunk_B, sin_table))

        dist_sq = (norms_sq[i_start:i_end, None, None]
                   + norms_sq[None, :, None]
                   - 2 * dot)

        K = np.exp(-dist_sq / bandwidth)

        if method == 'min':
            W[i_start:i_end, :] = np.max(K, axis=2)
        elif method == 'mean':
            W[i_start:i_end, :] = np.mean(K, axis=2)

    print(f"  Kernel matrix computed in {time.time() - t0:.1f}s")
    return W


def compute_gaussian_kernel_vectorized(data, bandwidth):
    """Standard Gaussian kernel via scipy cdist -- no orbit.
    Handles complex-valued data by stacking real/imag parts."""
    if np.iscomplexobj(data):
        data = np.hstack([data.real, data.imag]).astype(np.float64)
    else:
        data = np.asarray(data, dtype=np.float64)
    dists_sq = cdist(data, data, metric='sqeuclidean')
    return np.exp(-dists_sq / bandwidth)


def _compute_bispectrum_single(args):
    """Worker for parallel bispectrum computation."""
    k, coef_array, basis_params = args
    ffb = FFBBasis2D(basis_params['resolution'], ell_max=basis_params['ell_max'], dtype=float)
    from aspire.basis import Coef
    coef = Coef(ffb, coef_array[np.newaxis, :])
    return ffb.calculate_bispectrum(coef, flatten=True).flatten()


def compute_bispectrum_parallel(fb_coefs_raw, basis_params, n_workers=64):
    """Compute bispectrum for all images in parallel using multiprocessing."""
    args = [(k, fb_coefs_raw[k], basis_params) for k in range(len(fb_coefs_raw))]
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(_compute_bispectrum_single, args),
                            total=len(args), desc="Bispectrum"))
    return np.array(results)


# --- Data loading and reformatting (torch .pt -> numpy float64) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.environ.get(
    "PAPER_IMAGE_DATA_DIR",
    os.path.join(REPO_ROOT, "projection_data", "new_dataset_5.3.26"),
)

print("Loading data...")
projections_pt = torch.load(f"{DATA_DIR}/projections-synth.pt", weights_only=False)
angles_pt = torch.load(f"{DATA_DIR}/angles-synth.pt", weights_only=False)

if isinstance(projections_pt, list):
    dataset = np.array([p.numpy() for p in projections_pt], dtype=np.float64)
else:
    dataset = projections_pt.numpy().astype(np.float64)

if isinstance(angles_pt, list):
    angles = np.array([a.item() for a in angles_pt], dtype=np.float64)
else:
    angles = angles_pt.numpy().astype(np.float64)

num_projections = dataset.shape[0]
img_size = dataset.shape[1]
NUM_POINTS = num_projections

print(f"Loaded {num_projections} projections of size {img_size}x{img_size}")
print(f"Loaded {len(angles)} ground-truth angles")

# --- Fourier-Bessel basis ---
ffbbasis = FFBBasis2D((img_size, img_size), ell_max=10, dtype=float)

print("Expanding images in Fourier-Bessel basis...")
t0 = time.time()
fb_dataset = ffbbasis.expand(dataset[:NUM_POINTS], tol=1e-2)
print(f"Basis expansion done in {time.time() - t0:.1f}s")

# Extract raw coefficient arrays for vectorized operations
fb_coefs_raw = np.array([fb_dataset[i].asnumpy().flatten() for i in range(NUM_POINTS)])
angular_indices = ffbbasis.angular_indices
signs_indices = ffbbasis.signs_indices

# Apply random SO(2) rotations (vectorized via coefficient structure)
print("Applying random rotations...")
ell_max = int(np.max(angular_indices))
random_angles = np.random.uniform(0, 2 * np.pi, size=NUM_POINTS)
for i in range(NUM_POINTS):
    theta = random_angles[i]
    for ell in range(1, ell_max + 1):
        cos_mask = (angular_indices == ell) & (signs_indices == 1)
        sin_mask = (angular_indices == ell) & (signs_indices == -1)
        c_cos = fb_coefs_raw[i, cos_mask].copy()
        c_sin = fb_coefs_raw[i, sin_mask].copy()
        fb_coefs_raw[i, cos_mask] = np.cos(ell * theta) * c_cos - np.sin(ell * theta) * c_sin
        fb_coefs_raw[i, sin_mask] = np.sin(ell * theta) * c_cos + np.cos(ell * theta) * c_sin

# --- Run all methods ---
import os
methods_to_run = []
for m in ['min', 'mean', 'bispectrum']:
    out = f'outputs/eigvectors_ell10_{m}_new_dataset_2026-03-05_bw0.5.pkl'
    if os.path.exists(out):
        print(f"Skipping {m} -- output already exists: {out}")
    else:
        methods_to_run.append(m)

for method in methods_to_run:
    print(f"\n{'='*60}")
    print(f"Method: {method}")
    print(f"{'='*60}")
    t_method = time.time()

    if method == 'bispectrum':
        print("Computing bispectrum for all images (parallel)...")
        basis_params = {'resolution': (img_size, img_size), 'ell_max': 10}
        fb_bispectrum = compute_bispectrum_parallel(fb_coefs_raw, basis_params, n_workers=64)
        print(f"Bispectrum shape: {fb_bispectrum.shape}")

        print("Computing Gaussian kernel on bispectrum vectors...")
        W = compute_gaussian_kernel_vectorized(fb_bispectrum, 0.5)
    else:
        W = compute_orbit_kernel_vectorized(
            fb_coefs_raw, angular_indices, signs_indices,
            bandwidth=0.5, num_g=600, method=method
        )

    print("Computing Laplacian and eigenvectors...")
    S = calc_Laplacian(W)
    v, _ = generate_eig(S)

    elapsed = time.time() - t_method
    print(f"Method {method} completed in {elapsed:.1f}s")

    output_path = f'outputs/eigvectors_ell10_{method}_new_dataset_2026-03-05_bw0.5.pkl'
    print(f'Saving eigenvectors to {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(v, f)

# Save ground-truth angles
angles_path = 'outputs/ground_truth_angles_new_dataset_2026-03-05.pkl'
with open(angles_path, 'wb') as f:
    pickle.dump(angles[:NUM_POINTS], f)
print(f'\nSaved ground-truth angles to {angles_path}')
print("All methods completed!")
