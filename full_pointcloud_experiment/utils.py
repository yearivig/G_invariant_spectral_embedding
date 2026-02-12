"""
Utilities for the full pointcloud experiment (curated + documented).

This file is a *refactored copy* of the core pieces from `../utils.py`.
It is designed to:

- be readable to a math researcher (clear definitions + conventions)
- keep the same computational building blocks as the original project
- avoid breaking the original repo (we do not import or modify `../utils.py`)

Only a subset of the original `utils.py` is included here: just the parts
needed to generate the data, compute the affinity, build Laplacians,
compute eigenvectors, and save results to `.pkl`.
"""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Literal

import numpy as np
from numpy import linalg
from scipy.spatial.distance import pdist, squareform
from scipy.stats import special_ortho_group
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_synthetic_pointclouds(
    num_samples: int,
    *,
    num_atoms: int = 200,
    seed: int = 0,
    centered: bool = False,
    add_stationary: bool = False,
    so3_rotated: bool = True,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Generate a small synthetic point-cloud dataset for demos/tests.

    Why this exists
    ---------------
    In many environments (including this workspace) the original KTH dataset pickle
    is not available. This generator lets you run the full algorithmic pipeline
    end-to-end without external data.

    Construction
    ------------
    We build each sample as a noisy "deformed ring" embedded in 3D:

        theta ~ Uniform[0, 2pi]
        x = cos(theta)
        y = sin(theta)
        z = 0.3 * sin(2 theta)

    Then optionally:
    - center the cloud
    - append stationary Gaussian points
    - apply a random SO(3) rotation
    - add i.i.d. Gaussian noise

    Returns
    -------
    data : np.ndarray
        Shape (num_samples, num_atoms (+ 80 if stationary), 3)
    """
    rng = np.random.default_rng(seed)
    data = np.zeros((num_samples, num_atoms, 3), dtype=float)

    for i in range(num_samples):
        theta = rng.uniform(0.0, 2 * np.pi, size=(num_atoms,))
        x = np.cos(theta)
        y = np.sin(theta)
        z = 0.3 * np.sin(2 * theta)
        cloud = np.stack([x, y, z], axis=1)

        # add a small per-sample translation to create nontrivial structure
        cloud = cloud + rng.normal(scale=0.05, size=cloud.shape)
        data[i] = cloud

    if centered:
        data = data - data.mean(axis=1, keepdims=True)

    if add_stationary:
        stationary = rng.normal(loc=data[0].mean(axis=0), scale=0.2, size=(80, 3))
        data = np.concatenate((data, np.tile(stationary, (num_samples, 1, 1))), axis=1)

    if noise_std and noise_std > 0:
        data = data + rng.normal(scale=float(noise_std), size=data.shape)

    if so3_rotated:
        data = np.array([rotate_vectors_so3_arb(data[i]) for i in tqdm(range(len(data)), desc="Random SO(3) rotations")])

    return data


def load_pickle_file_as_numpy_array(file_name: str) -> np.ndarray:
    """Load a pickle file and return its contents as a NumPy array."""
    with open(file_name, "rb") as f:
        loaded_data = pickle.load(f)
    return np.array(loaded_data)


def rotate_vectors_so3_arb(vectors: np.ndarray) -> np.ndarray:
    """
    Apply a random SO(3) rotation to a point cloud.

    Parameters
    ----------
    vectors:
        Array of shape (N, 3) representing a point cloud.

    Returns
    -------
    rotated_vectors:
        Rotated point cloud of shape (N, 3).
    """
    q = special_ortho_group.rvs(3)  # random rotation matrix
    return vectors @ q


def generate_point_cloud_from_KTH(
    data_path: str,
    movement: Literal["rotation", "translation", "both", "NEW"],
    num_of_samples: int,
    *,
    so3_rotated: bool = True,
    centered: bool = False,
    add_stationary: bool = False,
    snr_db: float | None = None,
) -> np.ndarray:
    """
    Generate point clouds from the KTH-provided dataset (stored as a pickle).

    This matches the conventions used in the original `utils.py`:
    - dataset shape is assumed like: (num_samples, num_frames, 3)
    - movement selection:
        - "rotation": take frames [800:, :]
        - "translation": take frames [:150, :]
        - "both"/"NEW": keep as-is (project-specific; included for compatibility)

    Noise handling:
    - `snr_db=None` or `snr_db==0` => no noise
    - else, adds Gaussian noise per point-cloud to match the requested SNR (dB)
      using the same logic as the original project.
    """
    data = load_pickle_file_as_numpy_array(data_path)

    if movement == "rotation":
        data = data[:, 800:, :]
    elif movement == "translation":
        data = data[:, :150, :]

    data = data[:num_of_samples].copy()

    if centered:
        for i in tqdm(range(len(data)), desc="Centering point clouds"):
            data[i] = data[i] - np.mean(data[i], axis=0)

    if add_stationary:
        normal_points = np.random.normal(loc=np.mean(data[0], axis=0), scale=0.2, size=(80, 3))
        data = np.concatenate((data, np.tile(normal_points, (data.shape[0], 1, 1))), axis=1)

    if snr_db not in (None, 0):
        # signal power per point cloud
        signal_power = np.mean(data**2, axis=(1, 2), keepdims=True)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(loc=0, scale=np.sqrt(noise_power), size=data.shape)
        data = data + noise

    if so3_rotated:
        data = np.array([rotate_vectors_so3_arb(data[i]) for i in tqdm(range(len(data)), desc="Random SO(3) rotations")])

    return data


def generate_one_atom_from_KTH(num_of_samples: int, data_path: str | None = None) -> np.ndarray:
    """
    Extract a single 'atom' trajectory (frame 0) from the KTH dataset.

    If `data_path` is not provided, the original project uses an env var
    `DATA_KTH_PATH`. We keep that convention to ease reuse.
    """
    if data_path is None:
        data_path = os.getenv("DATA_KTH_PATH")
        if not data_path:
            raise ValueError("DATA_KTH_PATH env var not set and no data_path provided.")
    data = load_pickle_file_as_numpy_array(data_path)
    atom_mov = data[:num_of_samples, 0, :]
    return atom_mov


# ---------------------------------------------------------------------------
# Adaptive bandwidth selection (kNN heuristic)
# ---------------------------------------------------------------------------


def compute_bandwidth_knn(data: np.ndarray, *, k: int = 15) -> float:
    r"""
    Adaptive Gaussian bandwidth selection via average k-nearest-neighbor distance.

    This implements the formula you requested:

    \[
      \varepsilon = \frac{1}{2} \, \mathrm{mean}(d_k(x_i)), \qquad k = 15.
    \]

    where \(d_k(x_i)\) is the distance from sample \(x_i\) to its k-th nearest
    neighbor among the dataset \(\{x_j\}_{j\neq i}\).

    Implementation details
    ----------------------
    - Each point-cloud sample is flattened into a long vector (concatenating all
      its 3D coordinates).
    - We compute Euclidean distances in that flattened space.
    - We exclude the self-distance by setting the diagonal to `+inf`.

    Notes
    -----
    The *meaning* of \(\varepsilon\) depends on the kernel convention used.
    In this codebase, most Gaussian kernels are written as:

        K(x, y) = exp( - ||x - y||^2 / bandwidth )

    We interpret the adaptive value returned here as `bandwidth = ε`.
    If you want the more classical `exp(-||x-y||^2 / ε^2)` convention instead,
    you would set `bandwidth = ε**2`.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if data.shape[0] <= k:
        raise ValueError(f"Need at least k+1 samples; got n={data.shape[0]}, k={k}")

    flat = data.reshape((data.shape[0], -1))
    dist = squareform(pdist(flat, metric="euclidean"))
    np.fill_diagonal(dist, np.inf)
    sorted_distances = np.sort(dist, axis=1)
    dk = sorted_distances[:, k - 1]
    eps = 0.5 * float(np.mean(dk))
    return eps


# ---------------------------------------------------------------------------
# Graph construction + Laplacians
# ---------------------------------------------------------------------------


def calc_Laplacian(W: np.ndarray) -> np.ndarray:
    """
    Random-walk Laplacian (project convention).

    This follows the original code:
        D = diag(W 1)
        L = D^{-1} W
        L_rw = I - D^{-1} L

    Note: The naming in the original project is somewhat inconsistent.
    We keep it as-is for compatibility.
    """
    ones = np.ones(W.shape[0])
    v = W @ ones
    D = np.diag(v)
    D_inv = linalg.inv(D)
    L = D_inv @ W
    return np.eye(W.shape[0]) - D_inv @ L


def calc_Laplacian2(W: np.ndarray) -> np.ndarray:
    """
    Unnormalized graph Laplacian (project convention):
        L = D - W, where D = diag(W 1).
    """
    ones = np.ones(W.shape[0])
    v = W @ ones
    D = np.diag(v)
    return D - W


def generate_eig(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decomposition helper.

    Returns:
    - v: eigenvectors as rows (same convention as original project)
    - w: eigenvalues

    Also applies a "Lafon-style" normalization used in the original code:
        v = v / v[:, 0][:, None]
    """
    w, v = np.linalg.eig(matrix)
    v = np.transpose(v)
    v = v / v[:, 0][:, None]
    return v, w


# ---------------------------------------------------------------------------
# G-invariant kernels (SO(3) orbit integration / min / invariant features / none)
# ---------------------------------------------------------------------------


def quaternion_rotation_matrix(Q: np.ndarray) -> np.ndarray:
    """Convert a quaternion (q0,q1,q2,q3) to a 3x3 rotation matrix."""
    q0, q1, q2, q3 = Q
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])


def super_fibonacci(i: int, n: int) -> np.ndarray:
    """
    i-th element of the Fibonacci grid on SO(3) as a unit quaternion.
    From "Super-Fibonacci Spirals" (Marc Alexa).
    """
    phi = np.sqrt(2)
    psi = 1.5337511
    s = i + 0.5
    t = s / n
    d = 2 * np.pi * s
    r = np.sqrt(t)
    R = np.sqrt(1 - t)
    alpha = d / phi
    beta = d / psi
    return np.array([r * np.sin(alpha), r * np.cos(alpha), R * np.sin(beta), R * np.cos(beta)])


def rotate_vectors_so3(vectors: np.ndarray, n_frame: int, grid_size: int) -> np.ndarray:
    """Rotate a point cloud by a deterministic SO(3) element from the Fibonacci grid."""
    q = super_fibonacci(n_frame % grid_size, grid_size)
    rot = quaternion_rotation_matrix(q)
    return vectors @ rot


def find_dis_orbit3_mean(x: np.ndarray, y: np.ndarray, bandwidth: float, num_of_g_elements: int = 600) -> float:
    """
    "Integral" (mean) G-invariant kernel on SO(3) orbits.

    Computes:
        mean_k exp( -||x - R_k y||^2 / bandwidth )
    where R_k are sampled rotations (Fibonacci grid).
    """
    vals = np.array(
        [
            np.exp(-(np.linalg.norm(x - rotate_vectors_so3(y, k, num_of_g_elements)) ** 2) / bandwidth)
            for k in range(num_of_g_elements)
        ]
    )
    return float(np.mean(vals))


def find_dis_orbit3_min(x: np.ndarray, y: np.ndarray, bandwidth: float, num_of_g_elements: int = 600) -> float:
    """
    "Min" (actually max of exp(-dist^2/bw) = exp(-min dist^2/bw)) kernel.
    """
    vals = np.array(
        [
            np.exp(-(np.linalg.norm(x - rotate_vectors_so3(y, k, num_of_g_elements)) ** 2) / bandwidth)
            for k in range(num_of_g_elements)
        ]
    )
    return float(np.max(vals))


def find_dis_orbit3_invariant_features(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Kernel comparing Gram matrices (invariant features)."""
    gram_x = x @ x.T
    gram_y = y @ y.T
    return float(np.exp(-(np.linalg.norm(gram_x - gram_y, ord="fro") ** 2) / bandwidth))


def find_dis_orbit3(
    x: np.ndarray,
    y: np.ndarray,
    method: Literal["integral", "min", "invariant_features", "none"],
    bandwidth: float,
    num_of_g_elements: int = 600,
) -> float:
    """Dispatch between G-invariant kernel variants."""
    if method == "integral":
        return find_dis_orbit3_mean(x, y, bandwidth, num_of_g_elements)
    if method == "min":
        return find_dis_orbit3_min(x, y, bandwidth, num_of_g_elements)
    if method == "invariant_features":
        return find_dis_orbit3_invariant_features(x, y, bandwidth)
    if method == "none":
        return float(np.exp(-(np.linalg.norm(x - y) ** 2) / bandwidth))
    raise ValueError(f"Unknown method: {method}")


def _compute_dis_orbit3(args: tuple[np.ndarray, np.ndarray, str, float]) -> float:
    data_i, data_j, method, bandwidth = args
    return find_dis_orbit3(data_i, data_j, method=method, bandwidth=bandwidth)


def compute_distance_matrix_invariant(
    data: np.ndarray,
    num_of_points: int,
    method: Literal["integral", "min", "invariant_features", "none"],
    bandwidth: float,
    *,
    num_processes: int | None = None,
) -> np.ndarray:
    """
    Compute the full pairwise affinity matrix W_ij = K(x_i, x_j).

    Warning: this is O(n^2) and can be very slow for large `num_of_points`.
    """
    W = np.zeros((num_of_points, num_of_points), dtype=float)

    # Special-case optimization:
    # `invariant_features` computes Gram matrices G_i = X_i X_i^T.
    # Computing those inside the (i,j) loop would recompute them 2*n^2 times,
    # which is unnecessarily expensive. Precompute once per sample.
    if method == "invariant_features":
        grams: list[np.ndarray] = []
        for i in tqdm(range(num_of_points), desc="Precomputing Gram matrices"):
            Xi = data[i]
            grams.append(Xi @ Xi.T)

        # W is symmetric; compute only upper triangle.
        for i in tqdm(range(num_of_points), desc="Building W (invariant_features)"):
            W[i, i] = 1.0
            Gi = grams[i]
            for j in range(i + 1, num_of_points):
                Gj = grams[j]
                fro_sq = float(np.sum((Gi - Gj) ** 2))
                val = float(np.exp(-fro_sq / bandwidth))
                W[i, j] = val
                W[j, i] = val

        return W

    if num_processes is None:
        num_processes = max(1, cpu_count() // 2)

    # Parallelize all pair computations; this matches the original design.
    with Pool(processes=num_processes) as pool:
        arguments = [(data[i], data[j], method, bandwidth) for i in range(num_of_points) for j in range(num_of_points)]
        distances = list(tqdm(pool.imap(_compute_dis_orbit3, arguments), total=num_of_points**2, desc="Building W"))

    for i in range(num_of_points):
        for j in range(num_of_points):
            W[i, j] = distances[i * num_of_points + j]

    return W


def compute_bandwidth_auto(data: np.ndarray, num_of_points: int, method: str, k: int = 10) -> float:
    """
    Backwards-compatible alias for adaptive kNN bandwidth.

    Prefer calling `compute_bandwidth_knn(data, k=...)` directly.
    """
    return compute_bandwidth_knn(data[:num_of_points], k=k)


def compute_eigvectors_of_laplacian_invariant(
    data: np.ndarray,
    num_of_points: int,
    bandwidth: float,
    method: Literal["integral", "min", "invariant_features", "none"],
    *,
    auto_bandwidth: bool = False,
    knn_k: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvectors for two Laplacians from the same affinity matrix W.

    Returns
    -------
    (v_rw, v_gl):
        - v_rw: eigenvectors (rows) of random-walk Laplacian (RWGL)
        - v_gl: eigenvectors (rows) of unnormalized Laplacian (GL)

    Important
    ---------
    The original project sometimes *overrode* the passed `bandwidth` internally.
    That is confusing because filenames store the user-specified bandwidth.

    Here we make it explicit:
    - If `auto_bandwidth=False` (default), we use the provided `bandwidth`.
    - If `auto_bandwidth=True`, we compute and use an adaptive kNN bandwidth
      with `knn_k` (defaults to 15).
    """
    if auto_bandwidth:
        bandwidth = compute_bandwidth_knn(data[:num_of_points], k=knn_k)
        print(f"[auto_bandwidth] Using ε={bandwidth} computed from kNN with k={knn_k}")

    W = compute_distance_matrix_invariant(data, num_of_points, method, bandwidth)

    S_rw = calc_Laplacian(W)
    v_rw, _ = generate_eig(S_rw)

    S_gl = calc_Laplacian2(W)
    v_gl, _ = generate_eig(S_gl)

    return v_rw, v_gl


# ---------------------------------------------------------------------------
# Saving experiment outputs
# ---------------------------------------------------------------------------


def _bool_to_str(x: bool) -> str:
    return "True" if x else "False"


def save_2d_array_to_pkl(
    x_array: np.ndarray,
    y_array: np.ndarray,
    file_folder: str,
    file_name: str,
    *,
    num_of_points: int,
    invariant_method: str,
    movement: str,
    bandwidth: float,
    is_centered: bool,
    add_stationary: bool,
    noise_value: float | int | None,
    noise_tag: str = "SNR",
    laplacian_type: str = "RWGL",
) -> str:
    """
    Save the 2D embedding (x_array, y_array) as a `.pkl` file.

    The filename encodes experiment metadata; we support both historical tags:
    - `noise_tag="SNR"`  -> `..._SNR:<value>_...`
    - `noise_tag="AN"`   -> `..._AN:<value>_...`
    """
    os.makedirs(file_folder, exist_ok=True)
    tag = noise_tag.strip()
    file_stem = (
        f"{file_name}"
        f"_NOP:{num_of_points}"
        f"_IM:{invariant_method}"
        f"_M:{movement}"
        f"_BW:{bandwidth}"
        f"_IC:{_bool_to_str(is_centered)}"
        f"_AS:{_bool_to_str(add_stationary)}"
        f"_{tag}:{noise_value}"
        f"_LT:{laplacian_type}"
    )
    full_stem = os.path.join(file_folder, file_stem)
    file_path = full_stem + ".pkl"

    np_array = np.array([x_array, y_array], dtype=object)
    with open(file_path, "wb") as f:
        pickle.dump(np_array, f)

    return full_stem

