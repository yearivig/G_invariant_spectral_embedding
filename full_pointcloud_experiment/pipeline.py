"""
Pipeline orchestration for a single experiment run.

This is a documented copy of the logic from `../pipeline.py`, with two goals:
1) Make the control flow clear (data -> kernel -> Laplacian -> eigenvectors -> save -> plot)
2) Preserve the original computation pattern (no behavior changes in the main algorithm)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import utils


LaplacianType = Literal["RWGL", "GL"]


@dataclass(frozen=True)
class ExperimentConfig:
    """
    All parameters needed to reproduce a single embedding run.

    Notes
    -----
    - `noise` is intentionally kept generic (it can represent SNR in dB or an amplitude),
      because historically the project used different conventions. The filename is
      controlled separately by `noise_tag`.
    - `bandwidth` is passed directly to the kernel; use `auto_bandwidth=True` to override.
    """

    data_path: str
    num_points: int
    save_folder: str
    save_name: str
    invariant_method: Literal["integral", "min", "invariant_features", "none"]
    movement: Literal["rotation", "translation", "both", "NEW"]
    bandwidth: float
    is_centered: bool
    add_stationary: bool
    noise: float | int | None
    noise_tag: str
    laplacian_type: LaplacianType
    auto_bandwidth: bool = False
    # k for kNN bandwidth selection when auto_bandwidth=True
    knn_k: int = 15
    # Multiply the adaptive (or manual) bandwidth by this factor.
    # E.g. bandwidth_multiplier=2.0 doubles the bandwidth.
    bandwidth_multiplier: float = 1.0
    # If true, ignore `data_path` and generate synthetic point clouds.
    synthetic: bool = False
    synthetic_num_atoms: int = 200
    synthetic_seed: int = 0
    synthetic_noise_std: float = 0.0


def run_experiment(config: ExperimentConfig) -> str:
    """
    Run a single experiment and return the saved `.pkl` stem path (without extension).

    Steps
    -----
    1) Load / generate point-cloud dataset
    2) Compute affinity matrix W using a chosen G-invariant kernel
    3) Compute eigenvectors for RWGL + GL Laplacians
    4) Choose which embedding to export (based on config.laplacian_type)
    5) Save embedding to `.pkl` with a parameter-encoded filename

    Returns
    -------
    pkl_stem:
        Full path *without* `.pkl` extension (matches project convention).
    """
    # 1) Data
    if config.synthetic:
        # Synthetic demo dataset (no external files needed).
        data = utils.generate_synthetic_pointclouds(
            config.num_points,
            num_atoms=config.synthetic_num_atoms,
            seed=config.synthetic_seed,
            centered=config.is_centered,
            add_stationary=config.add_stationary,
            so3_rotated=True,
            noise_std=config.synthetic_noise_std,
        )
    else:
        data = utils.generate_point_cloud_from_KTH(
            config.data_path,
            config.movement,
            config.num_points,
            centered=config.is_centered,
            add_stationary=config.add_stationary,
            snr_db=config.noise,  # same parameter name as original project
            so3_rotated=True,
        )

    # 2-3) Eigenvectors for both Laplacians
    # If we use adaptive bandwidth, compute it *once* here so that:
    # - we use the exact same ε in the kernel computation
    # - we write the true ε into the output filename (reproducibility)
    if config.auto_bandwidth:
        bandwidth_used = utils.compute_bandwidth_knn(data[: config.num_points], k=config.knn_k)
        print(f"[auto_bandwidth] ε_base={bandwidth_used} (k={config.knn_k})")
    else:
        bandwidth_used = config.bandwidth

    # Apply multiplier (e.g. 2.0 to double the bandwidth).
    if config.bandwidth_multiplier != 1.0:
        bandwidth_used *= config.bandwidth_multiplier
        print(f"[bandwidth_multiplier] ε={bandwidth_used} (×{config.bandwidth_multiplier})")

    v_rw, v_gl = utils.compute_eigvectors_of_laplacian_invariant(
        data,
        config.num_points,
        bandwidth=bandwidth_used,
        method=config.invariant_method,
        auto_bandwidth=False,  # handled above
        knn_k=config.knn_k,
    )

    # 4) Pick embedding by Laplacian type.
    # Convention: use the first two non-trivial eigenvectors as embedding coords.
    if config.laplacian_type == "RWGL":
        v = v_rw
    elif config.laplacian_type == "GL":
        v = v_gl
    else:
        raise ValueError(f"Unknown laplacian_type: {config.laplacian_type}")

    x = np.real(v[1])
    y = np.real(v[2])

    # 5) Save
    pkl_stem = utils.save_2d_array_to_pkl(
        x,
        y,
        config.save_folder,
        config.save_name,
        num_of_points=config.num_points,
        invariant_method=config.invariant_method,
        movement=config.movement,
        bandwidth=bandwidth_used,
        is_centered=config.is_centered,
        add_stationary=config.add_stationary,
        noise_value=config.noise,
        noise_tag=config.noise_tag,
        laplacian_type=config.laplacian_type,
    )

    return pkl_stem

