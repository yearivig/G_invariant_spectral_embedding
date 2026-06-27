"""
SO(2) rotation utilities for 2D images.

Supports both NumPy arrays (via scipy) and PyTorch tensors (via torchvision).
Refactored from:
    ../roy_lederman_data/rotations.py
"""

from __future__ import annotations

import math
import logging

import numpy as np
from scipy.ndimage import rotate as scipy_rotate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NumPy / SciPy rotations
# ---------------------------------------------------------------------------


def rotate_image(arr: np.ndarray, theta_radians: float) -> np.ndarray:
    """
    Rotate a 2D NumPy image by `theta_radians` about its center.

    Uses bilinear interpolation; areas outside the original image are filled with 0.
    The output has the same shape as the input (no expansion).
    """
    theta_degrees = np.degrees(theta_radians)
    return scipy_rotate(arr, angle=theta_degrees, reshape=False, order=1, cval=0.0, mode="constant")


def apply_random_rotations(images: np.ndarray, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a random SO(2) rotation to each image in the batch.

    Parameters
    ----------
    images : np.ndarray of shape (N, H, W)
    seed : optional RNG seed for reproducibility

    Returns
    -------
    (rotated_images, angles) : rotated batch and the angles used (radians)
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, len(images))
    rotated = np.array([rotate_image(images[i], angles[i]) for i in range(len(images))])
    return rotated, angles


# ---------------------------------------------------------------------------
# PyTorch rotations (GPU-accelerated)
# ---------------------------------------------------------------------------


def rotate_image_torch(img_tensor, theta_radians: float):
    """
    Rotate a PyTorch tensor image by `theta_radians`.

    Supports shapes [H, W] and [B, H, W]. Uses bilinear interpolation.
    """
    import torch
    import torchvision.transforms.functional as TF

    angle_degrees = float(theta_radians * 180.0 / math.pi)

    if img_tensor.dim() == 3:
        # Batch: [B, H, W]
        out = []
        for b in range(img_tensor.shape[0]):
            single = img_tensor[b].unsqueeze(0)  # [1, H, W]
            rotated = TF.rotate(single, angle_degrees, interpolation=TF.InterpolationMode.BILINEAR, expand=False, fill=0)
            out.append(rotated.squeeze(0))
        return torch.stack(out, dim=0)

    elif img_tensor.dim() == 2:
        # Single: [H, W]
        single = img_tensor.unsqueeze(0)  # [1, H, W]
        rotated = TF.rotate(single, angle_degrees, interpolation=TF.InterpolationMode.BILINEAR, expand=False, fill=0)
        return rotated.squeeze(0)

    raise ValueError(f"Expected [H, W] or [B, H, W], got shape {tuple(img_tensor.shape)}")


def rotate_batch_torch(images, angle_radians: float):
    """
    Rotate a batch of images [B, 1, H, W] by the same angle.
    Efficient torchvision batch rotation.
    """
    import torchvision.transforms.functional as TF

    angle_degrees = float(angle_radians * 180.0 / math.pi)
    return TF.rotate(images, angle_degrees, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
