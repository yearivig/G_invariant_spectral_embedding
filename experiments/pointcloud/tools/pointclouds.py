"""
The point clouds of Section 5.1: frames of the Glucagon trajectory in which the
psi-torsion angle of the 19th residue rotates, each given a random global SO(3)
rotation.

The trajectory is a pickle of shape (num_frames, num_atoms, 3) (data/data_3D.pkl,
not public). Each sample is built in this order, as in the code that produced
Figure 4 of the paper:

  1. frame i of the trajectory, atoms ROTATION_ATOMS (800 onward: the part of the
     file holding the rotation movement), for the first num_samples frames;
  2. optionally centred (IC in the output filenames);
  3. optionally 80 "stationary" points appended (AS): one draw from
     N(c, 0.2^2 I_3), c the centroid of the first cloud, the same 80 points in
     every cloud;
  4. optionally i.i.d. Gaussian noise at a given SNR in dB, per cloud:
     variance = mean(X^2) / 10^(SNR/10), with X the cloud after steps 2-3;
     SNR = 0 means no noise;
  5. a uniformly random rotation, X -> X Q with Q ~ Haar(SO(3)).

All randomness (the stationary points, the noise and the rotations) comes from
one numpy Generator, so a seed reproduces the dataset exactly.
"""

from __future__ import annotations

import pickle

import numpy as np
from scipy.stats import special_ortho_group

ROTATION_ATOMS = slice(800, None)
NUM_STATIONARY = 80
STATIONARY_SCALE = 0.2


def load_trajectory(path: str) -> np.ndarray:
    """The trajectory pickle as a float64 array (num_frames, num_atoms, 3)."""
    with open(path, "rb") as fh:
        return np.asarray(pickle.load(fh), dtype=np.float64)


def make_point_clouds(trajectory: np.ndarray, num_samples: int, *, centered: bool = False,
                      add_stationary: bool = False, snr_db: float | None = None,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """The randomly rotated point clouds (num_samples, num_points, 3); see the module docstring."""
    rng = np.random.default_rng() if rng is None else rng
    data = np.array(trajectory[:num_samples, ROTATION_ATOMS, :], dtype=np.float64)
    if len(data) < num_samples:
        raise ValueError(f"the trajectory has {len(data)} frames, {num_samples} requested")

    if centered:
        data -= data.mean(axis=1, keepdims=True)

    if add_stationary:
        stationary = rng.normal(loc=data[0].mean(axis=0), scale=STATIONARY_SCALE, size=(NUM_STATIONARY, 3))
        data = np.concatenate([data, np.broadcast_to(stationary, (len(data), NUM_STATIONARY, 3))], axis=1)

    if snr_db not in (None, 0):
        signal_power = np.mean(data ** 2, axis=(1, 2), keepdims=True)
        noise_power = signal_power / 10 ** (snr_db / 10)
        data = data + rng.normal(size=data.shape) * np.sqrt(noise_power)

    rotations = special_ortho_group.rvs(3, size=len(data), random_state=rng).reshape(-1, 3, 3)
    return data @ rotations
