"""
Image preprocessing utilities.

Functions for loading, converting to grayscale, squaring,
and masking images to a centered disk.

Refactored from:
    ../roy_lederman_data/image_operation.py
    + largest_centered_disk() from ../roy_lederman_data/main.py
"""

from __future__ import annotations

import logging
import os

import numpy as np
from matplotlib import image as mplimage

logger = logging.getLogger(__name__)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using standard luminance weights."""
    if image.ndim == 3 and image.shape[2] == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image


def make_square(image: np.ndarray, fill_value: float = 0) -> np.ndarray:
    """Pad an image to make it square, centering the original content."""
    height, width = image.shape[:2]
    if height == width:
        return image

    size = max(height, width)
    pad_top = (size - height) // 2
    pad_bottom = size - height - pad_top
    pad_left = (size - width) // 2
    pad_right = size - width - pad_left

    if image.ndim == 2:
        return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                      constant_values=fill_value)
    return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                  constant_values=fill_value)


def largest_centered_disk(arr: np.ndarray) -> np.ndarray:
    """
    Mask a square 2D array to keep only pixels inside the largest centered disk.

    All pixels outside the inscribed circle are set to 0. This is useful for
    images that will be rotated (avoids corner artifacts).

    Parameters
    ----------
    arr : 2D square NumPy array (grayscale image)

    Returns
    -------
    disk_masked : same shape as arr, with zeros outside the disk
    """
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected 2D square array, got shape {arr.shape}")

    N = arr.shape[0]
    center = N // 2
    y, x = np.ogrid[:N, :N]
    mask = (x - center) ** 2 + (y - center) ** 2 <= center ** 2

    disk_masked = np.zeros_like(arr)
    disk_masked[mask] = arr[mask]
    return disk_masked


def load_and_preprocess_images(
    input_path: str,
    n: int,
    *,
    file_prefix: str = "s1_",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load images, convert to grayscale, make square, apply disk mask,
    and split into left/right halves.

    Parameters
    ----------
    input_path : directory containing image files
    n : number of images to load
    file_prefix : only load files starting with this prefix

    Returns
    -------
    (full_images, left_halves, right_halves) : each np.ndarray of shape (n, H, W)
    """
    _, _, filenames = next(os.walk(input_path))
    filenames = sorted([f for f in filenames if f.startswith(file_prefix)])
    logger.info(f"Found {len(filenames)} files with prefix '{file_prefix}'")

    if n > len(filenames):
        logger.warning(f"Requested {n} images but only {len(filenames)} available.")
        n = len(filenames)

    full_images = []
    left_halves = []
    right_halves = []

    for j in range(n):
        img_path = os.path.join(input_path, filenames[j])

        # Read -> grayscale -> square -> disk mask
        img = mplimage.imread(img_path)
        img = rgb_to_grayscale(img)
        img = make_square(img)
        img = largest_centered_disk(img)

        # Split into halves
        H, W = img.shape
        left_halves.append(img[:, : W // 2])
        right_halves.append(img[:, W // 2 :])
        full_images.append(img)

    return np.array(full_images), np.array(left_halves), np.array(right_halves)
