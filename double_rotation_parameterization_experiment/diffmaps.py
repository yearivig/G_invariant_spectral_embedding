"""
Diffusion maps: standard and SO(2)-invariant, GPU-accelerated via PyTorch.

Refactored from:
    ../roy_lederman_data/diffmaps.py

Three variants:
  - mydiffmap()          : standard diffusion maps (Euclidean distances)
  - mydiffmap_so2_fast() : SO(2)-invariant diffusion maps (chunked min-distance)
  - compute_so2_distance_matrix() : builds the full pairwise SO(2)-invariant distance matrix

All functions return NumPy arrays. PyTorch is used internally for GPU acceleration.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch

from .rotations import rotate_batch_torch

logger = logging.getLogger(__name__)


DeviceStr = str  # e.g. "cuda:0", "cpu"


def _get_device(device: DeviceStr | None = None) -> torch.device:
    """Resolve a device string, defaulting to GPU if available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# SO(2)-invariant distance matrix
# ---------------------------------------------------------------------------


def compute_so2_distance_matrix(
    images: np.ndarray | torch.Tensor,
    num_rotations: int = 300,
    device: DeviceStr | None = None,
    chunk_size: int = 8,
) -> torch.Tensor:
    """
    Compute the pairwise SO(2)-invariant distance matrix.

    For each pair (i, j), computes:
        d(i, j) = min_{k} ||image_i - R_k(image_j)||

    where R_k is the k-th discrete rotation in SO(2).

    Uses chunked processing for memory efficiency on GPU.

    Parameters
    ----------
    images : (N, H, W) array or tensor
    num_rotations : number of SO(2) samples
    device : PyTorch device string
    chunk_size : process this many images at a time to save GPU memory

    Returns
    -------
    dist_matrix : (N, N) tensor of min-distances
    """
    dev = _get_device(device)

    if isinstance(images, np.ndarray):
        data = torch.tensor(images, dtype=torch.float32, device=dev)
    else:
        data = images.to(dev)

    n = data.shape[0]
    dist = torch.zeros((n, n), dtype=torch.float32, device=dev)
    angles = torch.linspace(0, 2 * np.pi, num_rotations, device=dev)

    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        chunk_i = data[i:i_end]

        for j in range(i, n, chunk_size):
            j_end = min(j + chunk_size, n)
            chunk_j = data[j:j_end]

            # Find min-distance over all rotations for this chunk pair
            min_dists = torch.full((i_end - i, j_end - j), float("inf"), device=dev)

            rot_batch_size = 4
            for k in range(0, num_rotations, rot_batch_size):
                k_end = min(k + rot_batch_size, num_rotations)
                batch_angles = angles[k:k_end]

                # Rotate chunk_j by each angle: result shape [num_angles, chunk_j_size, H, W]
                rotated_j = torch.stack([
                    rotate_batch_torch(chunk_j.unsqueeze(1), a.item())
                    for a in batch_angles
                ])  # [num_angles, chunk_j_size, 1, H, W] -> squeeze to [num_angles, chunk_j_size, H, W]
                if rotated_j.dim() == 5:
                    rotated_j = rotated_j.squeeze(2)

                # Compute distances: chunk_i[idx] vs rotated_j[angle, jdx]
                for idx_i, img_i in enumerate(chunk_i):
                    diffs = img_i.unsqueeze(0) - rotated_j  # [num_angles, chunk_j_size, H, W]
                    dists_flat = torch.norm(diffs.reshape(k_end - k, j_end - j, -1), dim=2)
                    min_dists[idx_i] = torch.minimum(min_dists[idx_i], dists_flat.min(dim=0).values)

            dist[i:i_end, j:j_end] = min_dists
            if i != j:
                dist[j:j_end, i:i_end] = min_dists.T

    return dist


# ---------------------------------------------------------------------------
# Standard diffusion maps
# ---------------------------------------------------------------------------


def mydiffmap(
    data: np.ndarray,
    num_neighbors: int = 20,
    t: int = 10,
    device: DeviceStr | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard diffusion maps via GPU-accelerated eigendecomposition.

    Steps:
      1) Compute pairwise Euclidean distance matrix
      2) Estimate bandwidth σ² from k-nearest-neighbor distances
      3) Build Gaussian kernel A = exp(-d²/σ²)
      4) Row-normalize: M = D^{-1} A
      5) Symmetrize: K = D^{1/2} M D^{-1/2}  (where D = diag of stationary dist π)
      6) Eigendecompose K
      7) Diffusion-time scaling: λ^t

    Parameters
    ----------
    data : (N, D) flattened data matrix
    num_neighbors : k for bandwidth estimation
    t : diffusion time
    device : PyTorch device

    Returns
    -------
    (embedding, eigenvectors, eigenvalues) : all NumPy arrays
    """
    dev = _get_device(device)
    logger.info(f"mydiffmap: N={data.shape[0]}, D={data.shape[1]}, t={t}, device={dev}")

    X = torch.tensor(data, dtype=torch.float32, device=dev)
    n = X.shape[0]

    # Pairwise Euclidean distances
    dist_mat = torch.cdist(X, X, p=2)

    # Bandwidth: mean of k-NN distances
    sorted_d, _ = torch.sort(dist_mat, dim=1)
    sigma2 = sorted_d[:, 1 : num_neighbors + 1].mean() ** 2

    # Gaussian kernel
    A = torch.exp(-(dist_mat ** 2) / sigma2)

    # Row-normalize
    row_sums = A.sum(dim=1, keepdim=True)
    M = A / row_sums

    # Stationary distribution
    pi = row_sums.squeeze() / row_sums.sum()
    sqrt_pi = pi.sqrt()

    # Symmetrize
    K = M * sqrt_pi.unsqueeze(1) / sqrt_pi.unsqueeze(0)

    # Eigen-decomposition (symmetric -> eigh)
    evals, evecs = torch.linalg.eigh(K)
    evals_desc, idx = torch.sort(evals, descending=True)
    evecs_desc = evecs[:, idx]

    # Reweight by 1/sqrt(π)
    U = evecs_desc / sqrt_pi.unsqueeze(1)

    # Diffusion-time scaling
    embedding = U * (evals_desc ** t).unsqueeze(0)

    return (
        embedding.detach().cpu().numpy(),
        U.detach().cpu().numpy(),
        evals_desc.detach().cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# SO(2)-invariant diffusion maps
# ---------------------------------------------------------------------------


def mydiffmap_so2_fast(
    images: np.ndarray,
    num_neighbors: int = 20,
    t: int = 10,
    num_rotations: int = 300,
    device: DeviceStr | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SO(2)-invariant diffusion maps using min-distance over discrete rotations.

    Same as `mydiffmap`, but replaces the Euclidean distance matrix with
    the SO(2)-invariant distance matrix (min over orbit).

    Parameters
    ----------
    images : (N, H, W) array of 2D grayscale images
    num_neighbors : k for bandwidth estimation
    t : diffusion time
    num_rotations : number of SO(2) samples
    device : PyTorch device

    Returns
    -------
    (embedding, eigenvectors, eigenvalues) : all NumPy arrays
    """
    dev = _get_device(device)
    n = images.shape[0]
    logger.info(f"mydiffmap_so2_fast: N={n}, t={t}, rotations={num_rotations}, device={dev}")

    # SO(2)-invariant distance matrix
    dist_mat = compute_so2_distance_matrix(images, num_rotations=num_rotations, device=str(dev))

    # Bandwidth from k-NN
    sorted_d, _ = torch.sort(dist_mat, dim=1)
    sigma2 = sorted_d[:, 1 : num_neighbors + 1].mean() ** 2

    # Gaussian kernel
    A = torch.exp(-(dist_mat ** 2) / sigma2)

    # Row-normalize
    row_sums = A.sum(dim=1, keepdim=True)
    M = A / row_sums

    # Stationary distribution
    pi = row_sums.squeeze() / row_sums.sum()
    sqrt_pi = pi.sqrt()

    # Symmetrize
    K = M * sqrt_pi.unsqueeze(1) / sqrt_pi.unsqueeze(0)

    # Eigen-decomposition
    evals, evecs = torch.linalg.eigh(K)
    evals_desc, idx = torch.sort(evals, descending=True)
    evecs_desc = evecs[:, idx]

    # Reweight
    U = evecs_desc / sqrt_pi.unsqueeze(1)

    # Diffusion-time scaling
    embedding = U * (evals_desc ** t).unsqueeze(0)

    return (
        embedding.detach().cpu().numpy(),
        U.detach().cpu().numpy(),
        evals_desc.detach().cpu().numpy(),
    )
