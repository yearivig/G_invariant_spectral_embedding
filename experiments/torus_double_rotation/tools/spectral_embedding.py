"""
Spectral embedding (Algorithm 1 of Group Invariant Spectral Embedding,
arXiv:2607.08987) with the Euclidean, SO(2)-minimum and SO(2)-integral kernels.

This is spectral embedding, not diffusion maps: there is no diffusion time
and no lambda^t scaling.

Algorithm 1
-----------
  1. W_ij = K(x_i, x_j)                      Eq. (3),  D_ii = sum_j W_ij
  2. L_RW = I - D^{-1} W                     Eq. (4)
  3. phi_1..phi_m: eigenvectors of L_RW for the m smallest nonzero eigenvalues
  4. x_i -> (phi_1(x_i), ..., phi_m(x_i))

Kernels (Section 2 and Proposition 3.2), with epsilon > 0 the bandwidth:
  Euclidean   K(x, y)     = exp(-||x - y||^2 / epsilon)
  minimum     K_min(x, y) = exp(-min_{a in G} ||x - a.y||^2 / epsilon)        Eq. (8)
  integral    K_int(x, y) = int_G exp(-||x - a.y||^2 / epsilon) d eta(a)       Eq. (9)

For G = SO(2) acting on images by in-plane rotation, the group is sampled at
num_group_elements equally spaced angles 2 pi k / N, k = 0..N-1 (default
N = 300). For the integral kernel this is the uniform trapezoidal rule on the
circle, with Haar measure normalised to 1, so the integral is the mean over
the N angles. For the minimum kernel it is a grid search. Rotations use
bilinear interpolation with zero fill (rotations.rotate_batch_torch).

The group-invariant kernels are symmetric by definition (Definition 3.1), but
on sampled images ||x - R y|| and ||y - R x|| differ slightly because only one
image is interpolated. W is therefore made exactly symmetric: the minimum
kernel takes the smaller of the two squared distances (the minimum over both
directions), the integral kernel averages the two directions.

The bandwidth epsilon is an input, as in the paper, which fixes it per
experiment. knn_epsilon() gives a data-driven choice, (mean distance to the
k nearest neighbours)^2.

All kernel and eigenvector arithmetic is in float64; image distances are
accumulated in float32 on the chosen torch device.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

try:
    from .rotations import rotate_batch_torch
except ImportError:
    from rotations import rotate_batch_torch

logger = logging.getLogger(__name__)

DeviceStr = str  # e.g. "cuda:0", "cpu"

NUM_GROUP_ELEMENTS = 300   # default number of SO(2) elements for the invariant kernels


def _get_device(device: DeviceStr | None = None) -> torch.device:
    """Resolve a device string, defaulting to GPU if available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_tensor(x, dev: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32, device=dev)
    return x.to(device=dev, dtype=torch.float32)


def so2_angles(num_group_elements: int = NUM_GROUP_ELEMENTS) -> np.ndarray:
    """N equally spaced rotation angles 2 pi k / N, k = 0..N-1 (0 and 2 pi are the same element)."""
    return 2 * np.pi * np.arange(num_group_elements) / num_group_elements


# ---------------------------------------------------------------------------
# Squared distances and kernels
# ---------------------------------------------------------------------------


def euclidean_sq_distances(data: np.ndarray, device: DeviceStr | None = None) -> np.ndarray:
    """||x_i - x_j||^2 for rows of data (n, D), as a float64 (n, n) array."""
    dev = _get_device(device)
    X = _as_tensor(np.asarray(data).reshape(len(data), -1), dev).double()
    d2 = torch.cdist(X, X, p=2) ** 2
    d2.fill_diagonal_(0.0)
    return d2.cpu().numpy()


def _so2_pass(images, num_group_elements: int, device, chunk_size: int, epsilon: float | None):
    """One pass over all pairs and sampled group elements.

    Entry (p, q) is min_k ||x_p - R_k x_q||^2 (epsilon is None), or
    log mean_k exp(-||x_p - R_k x_q||^2 / epsilon) (epsilon given). Each pair
    of chunks is computed once, with the column image rotated, and mirrored;
    inside the diagonal chunks both orders are computed. The caller
    symmetrises.
    """
    dev = _get_device(device)
    data = _as_tensor(images, dev)
    n = data.shape[0]
    angles = so2_angles(num_group_elements)
    out = torch.zeros((n, n), dtype=torch.float64, device=dev)
    rot_batch = 4

    for i in range(0, n, chunk_size):
        ci = data[i:i + chunk_size]
        for j in range(i, n, chunk_size):
            cj = data[j:j + chunk_size]
            na, nb = len(ci), len(cj)
            if epsilon is None:
                acc = torch.full((na, nb), float("inf"), dtype=torch.float64, device=dev)
            else:
                run_max = torch.full((na, nb), -float("inf"), dtype=torch.float64, device=dev)
                run_sum = torch.zeros((na, nb), dtype=torch.float64, device=dev)
            for k in range(0, num_group_elements, rot_batch):
                rb = torch.stack([rotate_batch_torch(cj.unsqueeze(1), float(a))
                                  for a in angles[k:k + rot_batch]])
                rb = rb.squeeze(2) if rb.dim() == 5 else rb            # (angles, nb, H, W)
                for ia, img in enumerate(ci):
                    d2 = ((img.unsqueeze(0) - rb) ** 2).reshape(len(rb), nb, -1).sum(dim=2).double()
                    if epsilon is None:
                        acc[ia] = torch.minimum(acc[ia], d2.min(dim=0).values)
                    else:                                              # running log-sum-exp of -d2/epsilon
                        a = -d2 / epsilon
                        new_max = torch.maximum(run_max[ia], a.max(dim=0).values)
                        run_sum[ia] = run_sum[ia] * torch.exp(run_max[ia] - new_max) + \
                            torch.exp(a - new_max).sum(dim=0)
                        run_max[ia] = new_max
            if epsilon is not None:
                acc = run_max + torch.log(run_sum) - math.log(num_group_elements)
            out[i:i + na, j:j + nb] = acc
            if i != j:
                out[j:j + nb, i:i + na] = acc.T
    return out


def so2_min_sq_distances(images, num_group_elements: int = NUM_GROUP_ELEMENTS,
                         device: DeviceStr | None = None, chunk_size: int = 8) -> np.ndarray:
    """min_{a in G} ||x_i - a.x_j||^2 over the sampled SO(2), symmetrised by the minimum of both directions."""
    logger.info(f"SO(2) minimum distances: n={len(images)}, group elements={num_group_elements}")
    d2 = _so2_pass(images, num_group_elements, device, chunk_size, epsilon=None)
    d2 = torch.minimum(d2, d2.T)
    d2.fill_diagonal_(0.0)
    return d2.cpu().numpy()


def gaussian_kernel(sq_dists: np.ndarray, epsilon: float) -> np.ndarray:
    """W_ij = exp(-d_ij^2 / epsilon): the Euclidean kernel from Euclidean distances, Eq. (8) from minimum ones."""
    return np.exp(-np.asarray(sq_dists, dtype=np.float64) / float(epsilon))


def so2_integral_kernel(images, epsilon: float, num_group_elements: int = NUM_GROUP_ELEMENTS,
                        device: DeviceStr | None = None, chunk_size: int = 8) -> np.ndarray:
    """W_ij = mean over the N sampled angles of exp(-||x_i - R x_j||^2 / epsilon), Eq. (9).

    Computed as a running log-sum-exp in float64, so no term underflows before
    the average. Symmetrised by averaging the two directions.
    """
    logger.info(f"SO(2) integral kernel: n={len(images)}, group elements={num_group_elements}, epsilon={epsilon:g}")
    logW = _so2_pass(images, num_group_elements, device, chunk_size, epsilon=float(epsilon))
    W = torch.exp(logW)
    return (0.5 * (W + W.T)).cpu().numpy()


def knn_epsilon(sq_dists: np.ndarray, num_neighbors: int = 20) -> float:
    """(mean distance to the num_neighbors nearest neighbours)^2.

    A data-driven default; the paper sets epsilon by hand for each experiment.
    """
    d = np.sqrt(np.maximum(np.asarray(sq_dists, dtype=np.float64), 0.0))
    nn = np.sort(d, axis=1)[:, 1:num_neighbors + 1]
    return float(nn.mean() ** 2)


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------


def spectral_embedding(W: np.ndarray, m: int, device: DeviceStr | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1: eigenvectors of L_RW = I - D^{-1} W for the m smallest nonzero eigenvalues.

    L_RW is not symmetric, but it is similar to the symmetric
    L_sym = I - D^{-1/2} W D^{-1/2}: if L_sym v = lambda v then
    L_RW (D^{-1/2} v) = lambda D^{-1/2} v. The eigenvectors are computed from
    L_sym with a symmetric solver and mapped back, then scaled to unit
    Euclidean norm (so phi_0 = n^{-1/2} 1, as in Section 2) with the sign
    fixed so each phi's largest-magnitude entry is positive.

    Returns
    -------
    phi : (n, m)   phi[:, k-1] = phi_k, the embedding coordinates of Algorithm 1
    lam : (m + 1,) eigenvalues lambda_0 <= lambda_1 <= ... <= lambda_m of L_RW
    """
    dev = _get_device(device)
    Wt = torch.tensor(np.asarray(W, dtype=np.float64), device=dev)
    if not torch.allclose(Wt, Wt.T, rtol=0, atol=1e-12 * float(Wt.abs().max())):
        raise ValueError("the weight matrix W must be symmetric")
    deg = Wt.sum(dim=1)
    if bool((deg <= 0).any()):
        raise ValueError("a row of W sums to zero: that point has no neighbours (epsilon too small?)")
    s = deg.rsqrt()
    L_sym = torch.eye(len(Wt), dtype=torch.float64, device=dev) - s[:, None] * Wt * s[None, :]
    lam, V = torch.linalg.eigh(0.5 * (L_sym + L_sym.T))            # ascending
    phi = s[:, None] * V                                            # eigenvectors of L_RW
    phi = phi / phi.norm(dim=0, keepdim=True)
    idx = phi.abs().argmax(dim=0)
    phi = phi * torch.sign(phi[idx, torch.arange(phi.shape[1], device=dev)])[None, :]
    lam = lam.clamp_min(0.0).cpu().numpy()
    n_zero = int((lam < 1e-10).sum())
    if n_zero > 1:
        logger.warning(f"L_RW has {n_zero} zero eigenvalues: the graph is disconnected, "
                       f"which Algorithm 1 assumes it is not (epsilon too small?)")
    return phi[:, 1:m + 1].cpu().numpy(), lam[:m + 1]


def embed(kernel: str, data: np.ndarray, m: int, epsilon: float | str = "knn",
          num_group_elements: int = NUM_GROUP_ELEMENTS, num_neighbors: int = 20,
          device: DeviceStr | None = None, sq_dists: np.ndarray | None = None) -> dict:
    """Build W for one kernel and run Algorithm 1.

    kernel   "euclidean", "min" or "integral"
    data     (n, H, W) images (flattened for the Euclidean kernel)
    epsilon  a bandwidth, or "knn" for knn_epsilon() of the kernel's own
             distances (for "integral": of the SO(2)-minimum distances)
    sq_dists optional precomputed squared distances to reuse: Euclidean for
             "euclidean", SO(2)-minimum for "min" and for the "knn" bandwidth
             of "integral"

    Returns dict with phi (n, m), eigenvalues (m + 1,), epsilon, and sq_dists
    (the squared distances used, None for a fixed-epsilon integral kernel).
    """
    if kernel == "euclidean":
        d2 = sq_dists if sq_dists is not None else euclidean_sq_distances(data, device)
    elif kernel in ("min", "integral"):
        need = kernel == "min" or epsilon == "knn"
        d2 = sq_dists if sq_dists is not None else (
            so2_min_sq_distances(data, num_group_elements, device) if need else None)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    eps = knn_epsilon(d2, num_neighbors) if epsilon == "knn" else float(epsilon)
    W = so2_integral_kernel(data, eps, num_group_elements, device) if kernel == "integral" else gaussian_kernel(d2, eps)
    phi, lam = spectral_embedding(W, m, device)
    logger.info(f"{kernel} kernel: epsilon={eps:.6g}, lambda_1..3 = {np.round(lam[1:4], 6)}")
    return {"phi": phi, "eigenvalues": lam, "epsilon": eps, "sq_dists": d2}
