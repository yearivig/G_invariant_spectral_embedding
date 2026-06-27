"""
Pipeline orchestration for the double-rotation parameterization experiment.

This is a documented, modular version of `run_and_save_experiment()` from
    ../roy_lederman_data/main.py

Flow:
  1) Load and preprocess images → split into left/right halves
  2) Compute standard diffusion maps on full, left, right
  3) Apply random SO(2) rotations
  4) Compare Euclidean, SO(2)-min, and SO(2)-integral embeddings on rotated images
  5) Save all results and generate 3D scatter visualizations
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

try:
    from . import diffmaps, image_ops, rotations, visualization
except ImportError:
    import diffmaps, image_ops, rotations, visualization

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    """
    All parameters needed to reproduce the double-rotation experiment.

    Parameters
    ----------
    input_path : directory containing the source images
    n_images : number of images to process
    output_dir : where to save results
    file_prefix : only load files starting with this prefix
    t : diffusion time parameter
    num_neighbors : number of neighbors for bandwidth estimation
    num_rotations : number of SO(2) samples for invariant kernel
    n_eigenvectors : number of diffusion coordinates to extract (1-based, so 99 means indices 1..99)
    device : PyTorch device (e.g. "cuda:0", "cpu")
    """

    input_path: str
    n_images: int
    output_dir: str
    file_prefix: str = "s1_"
    t: int = 10
    num_neighbors: int = 20
    num_rotations: int = 300
    n_eigenvectors: int = 99
    device: str | None = None


def run_experiment(config: ExperimentConfig) -> str:
    """
    Run the full double-rotation parameterization experiment.

    Returns the output directory path.
    """
    out = config.output_dir
    os.makedirs(out, exist_ok=True)

    # ── 1) Load and preprocess ──────────────────────────────────────────
    logger.info(f"[1/8] Loading {config.n_images} images from {config.input_path}")
    t0 = time.time()
    full_images, left_halves, right_halves = image_ops.load_and_preprocess_images(
        config.input_path, config.n_images, file_prefix=config.file_prefix
    )
    logger.info(f"  Loaded {len(full_images)} images in {time.time() - t0:.1f}s")

    # Save example image (white background, no title, no axes)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('white')
    ax.imshow(full_images[0], cmap="gray")
    ax.set_axis_off()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"example_image.{ext}"), dpi=100,
                    facecolor='white', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # ── 2) Standard diffusion maps (full / left / right) ───────────────
    logger.info("[2/8] Computing standard diffusion maps (full, left, right)")
    n_ev = config.n_eigenvectors

    flat_full = full_images.reshape(len(full_images), -1)
    flat_left = left_halves.reshape(len(left_halves), -1)
    flat_right = right_halves.reshape(len(right_halves), -1)

    emb_full, U_full, evals_full = diffmaps.mydiffmap(flat_full, config.num_neighbors, config.t, config.device)
    emb_left, U_left, evals_left = diffmaps.mydiffmap(flat_left, config.num_neighbors, config.t, config.device)
    emb_right, U_right, evals_right = diffmaps.mydiffmap(flat_right, config.num_neighbors, config.t, config.device)

    # Extract diffusion coordinates (skip trivial eigenvector 0)
    vecs_full = np.array([emb_full[:, i] for i in range(1, n_ev + 1)])
    vecs_left = np.array([emb_left[:, i] for i in range(1, n_ev + 1)])
    vecs_right = np.array([emb_right[:, i] for i in range(1, n_ev + 1)])

    # Color maps: arctan2 of first two eigenvectors of each half
    color_left = np.arctan2(vecs_left[0], vecs_left[1])
    color_right = np.arctan2(vecs_right[0], vecs_right[1])

    # Save
    _save(out, "diffusion_vectors_full.pkl", vecs_full)
    _save(out, "diffusion_vectors_left.pkl", vecs_left)
    _save(out, "diffusion_vectors_right.pkl", vecs_right)
    _save(out, "color_values_left.pkl", color_left)
    _save(out, "color_values_right.pkl", color_right)

    # ── 3) Apply random SO(2) rotations ────────────────────────────────
    logger.info("[3/8] Applying random SO(2) rotations to full images")
    rotated_images, angles_used = rotations.apply_random_rotations(full_images)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('white')
    ax.imshow(rotated_images[0], cmap="gray")
    ax.set_axis_off()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"example_rotated_image.{ext}"), dpi=100,
                    facecolor='white', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # ── 4) Standard diffusion maps on rotated images (Euclidean kernel) ──
    logger.info("[4/8] Computing Euclidean diffusion maps on rotated images")
    flat_rotated = rotated_images.reshape(len(rotated_images), -1)
    emb_rot, _, evals_rot = diffmaps.mydiffmap(flat_rotated, config.num_neighbors, config.t, config.device)
    vecs_euclidean = np.array([emb_rot[:, i] for i in range(1, n_ev + 1)])

    # Keep the original filename for backwards compatibility, and add explicit
    # Euclidean names for the kernel-comparison outputs.
    _save(out, "rotated_diffusion_vectors.pkl", vecs_euclidean)
    _save(out, "euclidean_diffusion_vectors.pkl", vecs_euclidean)
    _save(out, "euclidean_eigenvalues.pkl", evals_rot)

    # ── 5) SO(2)-invariant diffusion maps (min kernel) ─────────────────
    logger.info("[5/8] Computing SO(2)-invariant diffusion maps (min kernel) on rotated images")
    emb_so2, U_so2, evals_so2 = diffmaps.mydiffmap_so2_fast(
        rotated_images, config.num_neighbors, config.t, config.num_rotations, config.device
    )
    vecs_so2 = np.array([emb_so2[:, i] for i in range(1, n_ev + 1)])
    _save(out, "so2_min_diffusion_vectors.pkl", vecs_so2)
    _save(out, "so2_min_eigenvalues.pkl", evals_so2)

    # ── 6) SO(2)-invariant diffusion maps (integral kernel) ──────────
    logger.info("[6/8] Computing SO(2)-invariant diffusion maps (integral kernel) on rotated images")
    emb_int, U_int, evals_int = diffmaps.mydiffmap_so2_integral(
        rotated_images, config.num_neighbors, config.t, config.num_rotations, config.device
    )
    vecs_int = np.array([emb_int[:, i] for i in range(1, n_ev + 1)])
    _save(out, "so2_integral_diffusion_vectors.pkl", vecs_int)
    _save(out, "so2_integral_eigenvalues.pkl", evals_int)

    # ── 7) 3D scatter visualizations ───────────────────────────────────
    logger.info("[7/8] Generating 3D scatter plots for Euclidean/min/integral kernels")
    kernel_vectors = {
        "euclidean": vecs_euclidean,
        "min": vecs_so2,
        "integral": vecs_int,
    }
    for i1, i2, i3 in [(1, 2, 6), (1, 2, 7), (1, 2, 9)]:
        for kernel_label, vectors in kernel_vectors.items():
            visualization.plot_3d_scatter(vectors, i1, i2, i3, color_left, "left", out, kernel_label=kernel_label)
            visualization.plot_3d_scatter(vectors, i1, i2, i3, color_right, "right", out, kernel_label=kernel_label)

    # ── 8) Done ────────────────────────────────────────────────────────
    logger.info(f"[8/8] Done. All outputs saved to: {out}")
    return out


def _save(directory: str, filename: str, data) -> None:
    """Helper to pickle-save data."""
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"  Saved: {path}")
