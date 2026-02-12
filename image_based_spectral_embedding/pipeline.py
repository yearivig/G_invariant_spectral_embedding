"""
Pipeline orchestration for a single image-based spectral embedding run.

This is the documented, modular version of the logic in
    ../comparing_to_yoel_and_eitan/apply_method.py

Flow:
    1) Load raw 2D projection images from a pickle file
    2) Add noise at a specified SNR
    3) Expand images into a Fourier-Bessel (FFB) basis using ASPIRE
    4) Apply random SO(2) rotations to each FB coefficient vector
    5) Choose kernel method and compute spectral embedding
    6) Save eigenvectors to .pkl
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import utils

KernelMethod = Literal["min", "mean", "bispectrum", "none"]


@dataclass(frozen=True)
class ExperimentConfig:
    """
    All parameters needed to reproduce a single image-based embedding run.

    Parameters
    ----------
    data_path : path to the raw projections pickle (shape: (N, H, W))
    num_images : how many images to use from the dataset
    save_folder : directory for output .pkl files
    save_name : base name for output files
    method : kernel type ('min', 'mean', 'bispectrum', 'none')
    bandwidth : kernel bandwidth ε
    snr : signal-to-noise ratio in dB (0 = no noise)
    ell_max : maximum angular frequency for FFBBasis2D
    tol : tolerance for FFB expansion
    image_size : assumed square image size (default 82)
    num_rotations : number of SO(2) samples for orbit kernels (default 600)
    """

    data_path: str
    num_images: int
    save_folder: str
    save_name: str
    method: KernelMethod
    bandwidth: float
    snr: float
    ell_max: int = 10
    tol: float = 0.01
    image_size: int = 82
    num_rotations: int = 600


def _build_output_filename(config: ExperimentConfig) -> str:
    """Construct a descriptive output filename encoding experiment parameters."""
    return (
        f"{config.save_name}"
        f"_ell{config.ell_max}"
        f"_tol{config.tol}"
        f"_{config.method}"
        f"_BW{config.bandwidth}"
        f"_SNR{config.snr}"
        f"_N{config.num_images}"
    )


def run_experiment(config: ExperimentConfig) -> str:
    """
    Run a single image-based spectral embedding experiment.

    Returns
    -------
    pkl_path : str
        Path to the saved eigenvectors .pkl file.
    """
    # ---- lazy import of ASPIRE (heavy dependency) ----
    from aspire.basis import FFBBasis2D

    # 1) Load raw images
    print(f"[1/6] Loading projections from {config.data_path}")
    raw_images = utils.load_projections(config.data_path)
    raw_images = raw_images[: config.num_images]
    print(f"       Loaded {len(raw_images)} images, shape={raw_images.shape}")

    # 2) Add noise
    print(f"[2/6] Adding noise (SNR={config.snr} dB)")
    noisy_images = utils.add_noise_to_images(raw_images, snr_db=config.snr)

    # 3) Expand into FFB basis
    print(f"[3/6] Expanding into FFB basis (ell_max={config.ell_max}, tol={config.tol})")
    ffb = FFBBasis2D((config.image_size, config.image_size), ell_max=config.ell_max, dtype=float)
    fb_data = ffb.expand(noisy_images, tol=config.tol)

    # 4) Apply random SO(2) rotations
    print(f"[4/6] Applying random SO(2) rotations to each image")
    for i in range(len(fb_data)):
        angle = np.random.uniform(0, 2 * np.pi)
        fb_data[i] = fb_data[i].rotate(angle)

    # 5) Compute spectral embedding
    print(f"[5/6] Computing spectral embedding (method={config.method}, BW={config.bandwidth})")
    if config.method == "bispectrum":
        # Bispectrum: compute invariant features, then use vanilla Gaussian kernel.
        print("       Computing bispectrum features...")
        fb_bispectrum = np.array([
            ffb.calculate_bispectrum(fb_data[k], flatten=True)
            for k in range(config.num_images)
        ])
        eigvectors = utils.compute_spectral_embedding(
            fb_bispectrum, config.num_images, config.bandwidth, method="none"
        )
    else:
        eigvectors = utils.compute_spectral_embedding(
            fb_data, config.num_images, config.bandwidth,
            method=config.method, num_rotations=config.num_rotations,
        )

    # 6) Save
    os.makedirs(config.save_folder, exist_ok=True)
    out_name = _build_output_filename(config)
    pkl_path = os.path.join(config.save_folder, out_name + ".pkl")
    utils.save_eigenvectors(eigvectors, pkl_path)
    print(f"[6/6] Saved eigenvectors to {pkl_path}")

    return pkl_path
