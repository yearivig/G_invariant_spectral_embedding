"""
Utilities for the image-based spectral embedding experiment.

This is a *refactored copy* of the math/kernel/Laplacian logic from
    ../comparing_to_yoel_and_eitan/apply_method.py

Changes from the original:
- All functions have docstrings explaining the math
- No hardcoded paths or global state
- Noise + rotation logic is explicit and testable
- Compatible with the same ASPIRE FFBBasis2D objects
"""

from __future__ import annotations

import pickle
from typing import Literal

import numpy as np
from numpy import linalg
from tqdm import tqdm


# ---------------------------------------------------------------------------
# SO(2)-invariant kernels for Fourier-Bessel expanded images
# ---------------------------------------------------------------------------


def rotate_image_so2(y, angle: float):
    """
    Rotate an ASPIRE Fourier-Bessel coefficient vector by `angle` radians.

    Parameters
    ----------
    y : aspire coefficient vector (supports `.rotate()`)
    angle : rotation angle in radians

    Returns
    -------
    Rotated coefficient vector (same type as y).
    """
    return y.rotate(angle)


def kernel_mean_so2(
    x, y, bandwidth: float, num_rotations: int = 300
) -> float:
    r"""
    "Integral" (mean) SO(2)-invariant kernel.

    Computes:
        \frac{1}{|G|} \sum_{k=0}^{|G|-1} \exp\!\bigl(-\|x - R_k\, y\|^2 / \varepsilon\bigr)

    where R_k rotates y by angle 2πk/|G|.

    Parameters
    ----------
    x, y : ASPIRE FB coefficient vectors
    bandwidth : kernel bandwidth ε
    num_rotations : number of discrete SO(2) samples |G|
    """
    angles = np.linspace(0, 2 * np.pi, num_rotations, endpoint=False)
    values = np.array([
        np.exp(-(np.linalg.norm(x - rotate_image_so2(y, a)) ** 2) / bandwidth)
        for a in angles
    ])
    return float(np.mean(values))


def kernel_min_so2(
    x, y, bandwidth: float, num_rotations: int = 600
) -> float:
    r"""
    "Min-distance" SO(2)-invariant kernel.

    Computes:
        \max_{k} \exp\!\bigl(-\|x - R_k\, y\|^2 / \varepsilon\bigr)

    (The max of exp(-d²/ε) corresponds to the *minimum* distance over the orbit.)

    Parameters
    ----------
    x, y : ASPIRE FB coefficient vectors
    bandwidth : kernel bandwidth ε
    num_rotations : number of discrete SO(2) samples |G|
    """
    angles = np.linspace(0, 2 * np.pi, num_rotations, endpoint=False)
    values = np.array([
        np.exp(-(np.linalg.norm(x - rotate_image_so2(y, a)) ** 2) / bandwidth)
        for a in angles
    ])
    return float(np.max(values))


def kernel_gaussian(x, y, bandwidth: float) -> float:
    r"""
    Vanilla Gaussian kernel (no SO(2) invariance).

        K(x, y) = \exp\!\bigl(-\|x - y\|^2 / \varepsilon\bigr)
    """
    return float(np.exp(-(np.linalg.norm(x - y) ** 2) / bandwidth))


KernelMethod = Literal["min", "mean", "bispectrum", "none"]


def compute_kernel_entry(
    x, y, bandwidth: float, method: KernelMethod, num_rotations: int = 600
) -> float:
    """Dispatch to the appropriate kernel function."""
    if method == "min":
        return kernel_min_so2(x, y, bandwidth, num_rotations)
    elif method == "mean":
        return kernel_mean_so2(x, y, bandwidth, num_rotations)
    elif method == "none":
        return kernel_gaussian(x, y, bandwidth)
    else:
        raise ValueError(f"Unknown kernel method: {method!r}. Use 'min', 'mean', or 'none'.")


# ---------------------------------------------------------------------------
# Graph Laplacian + eigenvectors
# ---------------------------------------------------------------------------


def calc_rw_laplacian(W: np.ndarray) -> np.ndarray:
    """
    Random-walk normalised affinity matrix: P = D^{-1} W.

    This matches the original `calc_Laplacian` in apply_method.py:
        D = diag(W @ 1)
        P = D^{-1} W

    Note: despite the name in the original code, this returns the
    *transition matrix* P, not (I - P). The eigen-decomposition of P
    gives the diffusion-map embedding.
    """
    ones = np.ones(W.shape[0])
    d = W @ ones
    D_inv = np.diag(1.0 / d)
    return D_inv @ W


def compute_eigenvectors(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decomposition with Lafon-style normalisation.

    Returns
    -------
    (eigenvectors, eigenvalues)
        eigenvectors : shape (n, n), each row is an eigenvector, normalised
                       so that the first component equals 1.
        eigenvalues  : shape (n,)
    """
    eigenvalues, V = np.linalg.eig(matrix)
    V = V.T  # rows = eigenvectors
    V = V / V[:, 0][:, None]  # Lafon normalisation
    return V, eigenvalues


# ---------------------------------------------------------------------------
# Affinity matrix construction
# ---------------------------------------------------------------------------


def build_affinity_matrix(
    data,
    num_points: int,
    bandwidth: float,
    method: KernelMethod,
    *,
    num_rotations: int = 600,
) -> np.ndarray:
    """
    Build the full n×n affinity matrix W.

    Parameters
    ----------
    data : array-like of FB coefficient vectors (or numpy array for 'none')
    num_points : number of samples to use
    bandwidth : kernel bandwidth ε
    method : 'min', 'mean', or 'none'
    num_rotations : number of SO(2) samples for orbit-based kernels
    """
    W = np.zeros((num_points, num_points), dtype=float)
    for i in tqdm(range(num_points), desc=f"Building W ({method})"):
        for j in range(num_points):
            W[i, j] = compute_kernel_entry(
                data[i], data[j], bandwidth, method, num_rotations
            )
    return W


def compute_spectral_embedding(
    data,
    num_points: int,
    bandwidth: float,
    method: KernelMethod,
    *,
    num_rotations: int = 600,
) -> np.ndarray:
    """
    Full pipeline: data → W → P = D^{-1}W → eigenvectors.

    Returns
    -------
    eigenvectors : shape (num_points, num_points), rows are eigenvectors
    """
    W = build_affinity_matrix(data, num_points, bandwidth, method, num_rotations=num_rotations)
    P = calc_rw_laplacian(W)
    V, _ = compute_eigenvectors(P)
    return V


# ---------------------------------------------------------------------------
# Noise utilities
# ---------------------------------------------------------------------------


def add_noise_to_image(image: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Gaussian noise to a single 2D image at a given SNR (dB).

    If snr_db == 0, no noise is added.

    SNR convention:
        SNR = 10 * log10(signal_power / noise_power)
        noise_power = signal_power / 10^(SNR/10)
    """
    if snr_db == 0:
        return image.copy()

    signal_power = np.mean(image ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape)
    return image + noise


def add_noise_to_images(images: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to every image in a batch."""
    return np.array([add_noise_to_image(img, snr_db) for img in images])


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_projections(path: str) -> np.ndarray:
    """Load a pickle file containing projection images as a float64 numpy array."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data, dtype=np.float64)


def save_eigenvectors(eigenvectors: np.ndarray, path: str) -> None:
    """Save eigenvectors to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(eigenvectors, f)
