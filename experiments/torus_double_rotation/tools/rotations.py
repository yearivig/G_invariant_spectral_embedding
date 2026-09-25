"""
SO(2) rotation of 2D images about their centre.

  rotate_image        one NumPy image (scipy); scripts/03_rotate_dataset.py uses it
                      for the random rotation of each dataset image
  rotate_batch_torch  a batch of torch images by one angle (torchvision);
                      tools/spectral_embedding.py uses it for the sampled
                      group elements of the minimum and integral kernels

Both use bilinear interpolation with zero fill and keep the image size. They
turn in the same direction and agree to within 2e-5 on pixel values in [0, 1]
(measured on the obj87 + obj72 composites over 35 angles).

Adapted from Roy Lederman's code: roy_lederman_data/rotations.py.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import rotate as scipy_rotate


def rotate_image(arr: np.ndarray, theta_radians: float) -> np.ndarray:
    """
    Rotate a 2D NumPy image by `theta_radians` about its center.

    Uses bilinear interpolation; areas outside the original image are filled with 0.
    The output has the same shape as the input (no expansion).
    """
    theta_degrees = np.degrees(theta_radians)
    return scipy_rotate(arr, angle=theta_degrees, reshape=False, order=1, cval=0.0, mode="constant")


def rotate_batch_torch(images, angle_radians: float):
    """
    Rotate a batch of images [B, 1, H, W] by the same angle.
    Efficient torchvision batch rotation; areas outside the image are filled with 0.
    """
    import torchvision.transforms.functional as TF

    angle_degrees = float(angle_radians * 180.0 / math.pi)
    return TF.rotate(images, angle_degrees, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
