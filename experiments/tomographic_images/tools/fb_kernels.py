"""
The kernels of Section 5.2 on Fourier-Bessel coefficients, with G = SO(2)
acting by in-plane rotation (Example 3.4):

  euclidean   exp(-||x - y||^2 / eps)
  min         Eq. (8)    exp(-min_theta ||x - R_theta y||^2 / eps)
  mean        Eq. (9)    (1/G) sum_theta exp(-||x - R_theta y||^2 / eps)   (the integral kernel)
  bispectrum  Eq. (10)   exp(-||B(x) - B(y)||^2 / eps)

x, y are FFBBasis2D coefficient vectors (tools/fb_basis.py). theta runs over
num_angles equally spaced angles 2 pi k / G; for the integral kernel that is
the trapezoidal rule. The angles form a group, so both kernels are symmetric.
In the basis, rotation acts on each angular frequency ell as a 2 x 2 rotation,
so for all pairs at once

  <x, R_theta y> = C + sum_ell [A_ell cos(ell theta) + B_ell sin(ell theta)]

with C, A_ell, B_ell matrix products of the coefficient blocks.

B is ASPIRE's flattened bispectrum (FFBBasis2D.calculate_bispectrum, flatten=True):
for complex coefficients a (ell >= 0), the products a_i a_j conj(a_{q, ell_i + ell_j})
over unordered pairs {i, j} and radial indices q. It is never formed: with the
Gram matrices G_l(x, y) = sum_q a_{q,l}(x) conj(a_{q,l}(y)) and
H_l(x, y) = sum_q a_{q,l}(x)^2 conj(a_{q,l}(y))^2,

  <B(x), B(y)> = sum_{l1 > l2, l1 + l2 <= L} G_l1 G_l2 conj(G_{l1+l2})
               + sum_{2l <= L} (G_l^2 + H_l) / 2 conj(G_{2l}),

which gives the same distances as the explicit bispectrum (relative difference
~1e-14 on random images).

Kernels are returned as log W; weight_matrix() exponentiates after subtracting
the largest entry, so no row underflows. That multiplies W by one constant,
which leaves L_RW = I - D^{-1} W unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

NUM_ANGLES = 600


# ---------------------------------------------------------------------------
# Data: noise and random rotations
# ---------------------------------------------------------------------------

def add_noise(images: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. Gaussian noise at a per-image SNR in dB: variance = mean(x^2) / 10^(snr/10). SNR 0 = no noise."""
    if snr_db == 0:
        return images.copy()
    signal_power = np.mean(images ** 2, axis=(1, 2), keepdims=True)
    return images + rng.normal(size=images.shape) * np.sqrt(signal_power / 10 ** (snr_db / 10))


def rotate(coefs: np.ndarray, angular: np.ndarray, signs: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Rotate each coefficient vector by its own angle thetas[i], exactly, in the basis."""
    out = coefs.copy()
    for ell in range(1, int(angular.max()) + 1):
        pos = (angular == ell) & (signs == 1)
        neg = (angular == ell) & (signs == -1)
        c, s = np.cos(ell * thetas)[:, None], np.sin(ell * thetas)[:, None]
        out[:, pos] = c * coefs[:, pos] - s * coefs[:, neg]
        out[:, neg] = s * coefs[:, pos] + c * coefs[:, neg]
    return out


# ---------------------------------------------------------------------------
# Kernels (log W)
# ---------------------------------------------------------------------------

def sq_distances(features: np.ndarray) -> np.ndarray:
    """||x_i - x_j||^2 for rows of real or complex features."""
    f = np.asarray(features)
    if np.iscomplexobj(f):
        f = np.hstack([f.real, f.imag])
    norms = np.sum(f ** 2, axis=1)
    return np.maximum(norms[:, None] + norms[None, :] - 2 * f @ f.T, 0.0)


def orbit_log_kernel(coefs: np.ndarray, angular: np.ndarray, signs: np.ndarray, bandwidth: float, method: str,
                     num_angles: int = NUM_ANGLES, chunk: int = 100) -> np.ndarray:
    """log W for the SO(2) minimum ("min") or integral ("mean") kernel over num_angles angles."""
    n, ell_max = len(coefs), int(angular.max())
    norms = np.sum(coefs ** 2, axis=1)
    C = coefs[:, angular == 0] @ coefs[:, angular == 0].T
    A = np.empty((n, n, ell_max))
    B = np.empty((n, n, ell_max))
    for ell in range(1, ell_max + 1):
        xc, xs = coefs[:, (angular == ell) & (signs == 1)], coefs[:, (angular == ell) & (signs == -1)]
        A[:, :, ell - 1] = xc @ xc.T + xs @ xs.T
        B[:, :, ell - 1] = xs @ xc.T - xc @ xs.T
    angles = 2 * np.pi * np.arange(num_angles) / num_angles
    ells = np.arange(1, ell_max + 1)
    cos_t, sin_t = np.cos(np.outer(ells, angles)), np.sin(np.outer(ells, angles))

    logW = np.empty((n, n))
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        dot = C[i0:i1, :, None] + A[i0:i1] @ cos_t + B[i0:i1] @ sin_t
        d2 = np.maximum(norms[i0:i1, None, None] + norms[None, :, None] - 2 * dot, 0.0)
        if method == "min":
            logW[i0:i1] = -d2.min(axis=2) / bandwidth
        elif method == "mean":
            logW[i0:i1] = logsumexp(-d2 / bandwidth, axis=2) - np.log(num_angles)
        else:
            raise ValueError(f"unknown orbit method {method!r}")
    return 0.5 * (logW + logW.T)                       # symmetric already; removes rounding


def _bispectrum_inner(G: list, H: list, ell_max: int) -> np.ndarray:
    acc = np.zeros(G[0].shape, dtype=np.complex128)
    for l1 in range(ell_max + 1):
        for l2 in range(min(l1, ell_max - l1 + 1)):     # l2 < l1, l1 + l2 <= ell_max
            acc += G[l1] * G[l2] * np.conj(G[l1 + l2])
    for l in range(ell_max // 2 + 1):                    # unordered pairs within one frequency
        acc += 0.5 * (G[l] * G[l] + H[l]) * np.conj(G[2 * l])
    return acc


def bispectrum_sq_distances(blocks: list[np.ndarray], chunk: int = 256) -> np.ndarray:
    """||B(x_i) - B(x_j)||^2 from the complex coefficient blocks (tools/fb_basis.complex_blocks)."""
    ell_max, n = len(blocks) - 1, len(blocks[0])
    sq = [b ** 2 for b in blocks[:ell_max // 2 + 1]]
    norms = _bispectrum_inner([np.sum(np.abs(b) ** 2, axis=1).astype(np.complex128) for b in blocks],
                              [np.sum(np.abs(b) ** 4, axis=1).astype(np.complex128) for b in blocks], ell_max).real
    d2 = np.empty((n, n))
    for i0 in range(0, n, chunk):
        rows = slice(i0, min(i0 + chunk, n))
        G = [b[rows] @ b.conj().T for b in blocks]
        H = [s[rows] @ s.conj().T for s in sq]
        d2[rows] = norms[rows, None] + norms[None, :] - 2 * _bispectrum_inner(G, H, ell_max).real
    d2 = np.maximum(0.5 * (d2 + d2.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    return d2


def weight_matrix(logW: np.ndarray) -> np.ndarray:
    """W = exp(log W - max log W): W up to one global positive factor."""
    return np.exp(logW - logW.max())
