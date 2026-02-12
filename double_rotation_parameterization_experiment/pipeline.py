"""
Pipeline orchestration for the double-rotation parameterization experiment.

This is a documented, modular version of `run_and_save_experiment()` from
    ../roy_lederman_data/main.py

Flow:
  1) Load and preprocess images → split into left/right halves
  2) Compute standard diffusion maps on full, left, right
  3) Apply random SO(2) rotations → recompute standard diffusion maps
  4) Compute SO(2)-invariant diffusion maps on rotated images
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

from . import diffmaps, image_ops, rotations, visualization

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
    logger.info(f"[1/7] Loading {config.n_images} images from {config.input_path}")
    t0 = time.time()
    full_images, left_halves, right_halves = image_ops.load_and_preprocess_images(
        config.input_path, config.n_images, file_prefix=config.file_prefix
    )
    logger.info(f"  Loaded {len(full_images)} images in {time.time() - t0:.1f}s")

    # Save example image
    plt.imshow(full_images[0], cmap="gray")
    plt.axis("off")
    plt.title("Example preprocessed image")
    plt.savefig(os.path.join(out, "example_image.png"), dpi=100)
    plt.close()

    # ── 2) Standard diffusion maps (full / left / right) ───────────────
    logger.info("[2/7] Computing standard diffusion maps (full, left, right)")
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
    logger.info("[3/7] Applying random SO(2) rotations to full images")
    rotated_images, angles_used = rotations.apply_random_rotations(full_images)

    plt.imshow(rotated_images[0], cmap="gray")
    plt.axis("off")
    plt.title("Example rotated image")
    plt.savefig(os.path.join(out, "example_rotated_image.png"), dpi=100)
    plt.close()

    # ── 4) Standard diffusion maps on rotated images ───────────────────
    logger.info("[4/7] Computing standard diffusion maps on rotated images")
    flat_rotated = rotated_images.reshape(len(rotated_images), -1)
    emb_rot, _, _ = diffmaps.mydiffmap(flat_rotated, config.num_neighbors, config.t, config.device)
    vecs_rot = np.array([emb_rot[:, i] for i in range(1, n_ev + 1)])
    _save(out, "rotated_diffusion_vectors.pkl", vecs_rot)

    # ── 5) SO(2)-invariant diffusion maps ──────────────────────────────
    logger.info("[5/7] Computing SO(2)-invariant diffusion maps on rotated images")
    emb_so2, U_so2, evals_so2 = diffmaps.mydiffmap_so2_fast(
        rotated_images, config.num_neighbors, config.t, config.num_rotations, config.device
    )
    vecs_so2 = np.array([emb_so2[:, i] for i in range(1, n_ev + 1)])
    _save(out, "so2_invariant_diffusion_vectors.pkl", vecs_so2)
    _save(out, "so2_invariant_eigenvalues.pkl", evals_so2)

    # ── 6) 3D scatter visualizations ───────────────────────────────────
    logger.info("[6/7] Generating 3D scatter plots")
    for i1, i2, i3 in [(1, 2, 6), (1, 2, 7), (1, 2, 9)]:
        visualization.plot_3d_scatter(vecs_so2, i1, i2, i3, color_left, "left", out)
        visualization.plot_3d_scatter(vecs_so2, i1, i2, i3, color_right, "right", out)

    # ── 7) Done ────────────────────────────────────────────────────────
    logger.info(f"[7/7] Done. All outputs saved to: {out}")
    return out


def _save(directory: str, filename: str, data) -> None:
    """Helper to pickle-save data."""
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"  Saved: {path}")
