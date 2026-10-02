"""
The four kernels of Section 5.1, for point clouds X_i in R^{N x 3} with G = SO(3)
acting by X -> X R (Example 3.3):

  none                Euclidean    exp(-||X_i - X_j||_F^2 / eps)
  min                 Eq. (8)      exp(-min_R ||X_i - X_j R||_F^2 / eps)
  integral            Eq. (9)      int_SO(3) exp(-||X_i - X_j R||_F^2 / eps) d eta(R)
  invariant_features  Eq. (10)     exp(-||X_i X_i^T - X_j X_j^T||_F^2 / eps)   (Gram map)

Everything is computed from the 3 x 3 cross products C_ij = X_j^T X_i and the
norms s_i = ||X_i||_F^2, never from the N x N Gram matrices or the rotated clouds:

  ||X_i - X_j R||_F^2           = s_i + s_j - 2 <C_ij, R>_F
  ||X_i X_i^T - X_j X_j^T||_F^2 = ||C_ii||_F^2 + ||C_jj||_F^2 - 2 ||C_ij||_F^2

These are the same quantities the original pairwise loops computed.

Minimum kernel, two ways (--min-method):
  grid    (default) the minimum over the NUM_ROTATIONS-element super-Fibonacci
          grid on SO(3) (Alexa, 2022), as in the code behind Figure 4;
  kabsch  the exact minimum over SO(3): with C_ij = U S V^T,
          max_R <C_ij, R> = s_1 + s_2 + d s_3, d = sign det(U V^T).
Integral kernel: the mean over the same grid (quasi-Monte Carlo for the Haar
integral, Section 3.1).

The grid is not closed under R -> R^T, so on the grid K(X_i, X_j) != K(X_j, X_i).
Both grid kernels are therefore symmetrised over the grid and its inverses:
the minimum kernel takes the smaller distance of the two directions,
W <- max(W, W^T); the integral kernel averages them, W <- (W + W^T) / 2.
The exact (kabsch) minimum is symmetric already.

W is returned up to one global positive factor, exp(-c) with c the largest
log-entry, so that no row underflows to zero; L_RW = I - D^{-1} W does not
change when W is multiplied by a constant.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import logsumexp

Kernel = Literal["none", "min", "integral", "invariant_features"]
MinMethod = Literal["grid", "kabsch"]
KERNELS = ("min", "integral", "invariant_features", "none")
NUM_ROTATIONS = 600


# ---------------------------------------------------------------------------
# The super-Fibonacci grid on SO(3)
# ---------------------------------------------------------------------------

def quaternion_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of the unit quaternion (q0, q1, q2, q3), q0 the scalar part."""
    q0, q1, q2, q3 = q
    return np.array([
        [2 * (q0 * q0 + q1 * q1) - 1, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 2 * (q0 * q0 + q2 * q2) - 1, 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 2 * (q0 * q0 + q3 * q3) - 1],
    ])


def super_fibonacci(i: int, n: int) -> np.ndarray:
    """The i-th of n super-Fibonacci unit quaternions (Alexa, 2022)."""
    phi, psi = np.sqrt(2), 1.5337511
    s = i + 0.5
    t, d = s / n, 2 * np.pi * s
    r, R = np.sqrt(t), np.sqrt(1 - t)
    return np.array([r * np.sin(d / phi), r * np.cos(d / phi), R * np.sin(d / psi), R * np.cos(d / psi)])


def so3_grid(num_rotations: int = NUM_ROTATIONS) -> np.ndarray:
    """The super-Fibonacci rotations, (num_rotations, 3, 3)."""
    return np.stack([quaternion_rotation_matrix(super_fibonacci(k, num_rotations)) for k in range(num_rotations)])


# ---------------------------------------------------------------------------
# Squared distances and kernels
# ---------------------------------------------------------------------------

def cross_products(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C[i, j] = X_j^T X_i (n, n, 3, 3) and s_i = ||X_i||_F^2, for clouds X (n, N, 3)."""
    n = len(X)
    A = X.transpose(0, 2, 1).reshape(3 * n, -1)            # row 3j + a is X_j[:, a]
    C = (A @ A.T).reshape(n, 3, n, 3).transpose(2, 0, 1, 3)
    return C, np.einsum("iiaa->i", C)


def _over_grid(C, s, rotations, reduce, chunk: int):
    """reduce(||X_i - X_j R_k||^2 over k) for all pairs (i, j), chunked over i."""
    n, K = len(s), len(rotations)
    R = rotations.reshape(K, 9).T
    out = np.empty((n, n))
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        d2 = s[i0:i1, None, None] + s[None, :, None] - 2 * (C[i0:i1].reshape(-1, 9) @ R).reshape(i1 - i0, n, K)
        out[i0:i1] = reduce(np.maximum(d2, 0.0))
    return out


def log_kernel(X: np.ndarray, kernel: Kernel, epsilon: float, *, min_method: MinMethod = "grid",
               num_rotations: int = NUM_ROTATIONS, chunk: int = 32) -> np.ndarray:
    """log W_ij = log K(X_i, X_j) for one of the four kernels, symmetric (see the module docstring)."""
    C, s = cross_products(np.asarray(X, dtype=np.float64))
    if kernel == "none":
        d2 = s[:, None] + s[None, :] - 2 * np.einsum("ijaa->ij", C)
        logW = -np.maximum(d2, 0.0) / epsilon
    elif kernel == "invariant_features":
        g = np.einsum("iiab,iiab->i", C, C)
        d2 = g[:, None] + g[None, :] - 2 * np.einsum("ijab,ijab->ij", C, C)
        logW = -np.maximum(d2, 0.0) / epsilon
    elif kernel == "min" and min_method == "kabsch":
        U, S, Vt = np.linalg.svd(C)
        d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
        d2 = s[:, None] + s[None, :] - 2 * (S[..., 0] + S[..., 1] + d * S[..., 2])
        logW = -np.maximum(d2, 0.0) / epsilon
    elif kernel == "min" and min_method == "grid":
        d2 = _over_grid(C, s, so3_grid(num_rotations), lambda x: x.min(axis=2), chunk)
        logW = -np.minimum(d2, d2.T) / epsilon                       # W <- max(W, W^T)
    elif kernel == "integral":
        logW = _over_grid(C, s, so3_grid(num_rotations),
                          lambda x: logsumexp(-x / epsilon, axis=2) - np.log(num_rotations), chunk)
        logW = np.logaddexp(logW, logW.T) - np.log(2)                 # W <- (W + W^T) / 2
    else:
        raise ValueError(f"unknown kernel {kernel!r} or min_method {min_method!r}")
    return 0.5 * (logW + logW.T)                                      # exact symmetry, up to rounding


def weight_matrix(X: np.ndarray, kernel: Kernel, epsilon: float, **kwargs) -> np.ndarray:
    """W up to one global positive factor (see the module docstring)."""
    logW = log_kernel(X, kernel, epsilon, **kwargs)
    return np.exp(logW - logW.max())
