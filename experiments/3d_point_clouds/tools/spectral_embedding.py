"""
Algorithm 1 of Group Invariant Spectral Embedding (spectral embedding):

  1. W_ij = K(x_i, x_j)                     Eq. (3),  D_ii = sum_j W_ij
  2. L_RW = I - D^{-1} W                     Eq. (4)
  3. phi_1..phi_m: eigenvectors of L_RW for the m smallest nonzero eigenvalues
  4. x_i -> (phi_1(x_i), ..., phi_m(x_i))

L_RW is not symmetric, but it is similar to I - D^{-1/2} W D^{-1/2}: if
D^{-1/2} W D^{-1/2} v = mu v, then L_RW (D^{-1/2} v) = (1 - mu) D^{-1/2} v. The
eigenpairs are therefore computed with a symmetric solver, sorted by
lambda = 1 - mu ascending, and mapped back. Each phi_k is scaled to unit
Euclidean norm (so phi_0 = n^{-1/2} 1, as in Section 2), with its sign fixed so
that its largest-magnitude entry is positive. There is no diffusion time and
no lambda^t weighting.

W need not be positive semidefinite (the minimum kernel's is not, in general):
negative mu give lambda > 1, at the top of the spectrum, which step 3 never
selects. W must be symmetric with positive row sums.
"""

from __future__ import annotations

import numpy as np


def spectral_embedding(W: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1. Returns phi (n, m), column k-1 = phi_k, and lambda_0 <= ... <= lambda_m of L_RW."""
    W = np.asarray(W, dtype=np.float64)
    if not np.allclose(W, W.T, rtol=0, atol=1e-12 * np.abs(W).max()):
        raise ValueError("the weight matrix W must be symmetric")
    deg = W.sum(axis=1)
    if np.any(deg <= 0):
        raise ValueError("a row of W sums to zero: that point has no neighbours (epsilon too small?)")
    s = 1.0 / np.sqrt(deg)
    mu, V = np.linalg.eigh(s[:, None] * W * s[None, :])
    order = np.argsort(-mu)                          # lambda = 1 - mu ascending
    lam, phi = 1.0 - mu[order], s[:, None] * V[:, order]
    phi /= np.linalg.norm(phi, axis=0, keepdims=True)
    phi *= np.sign(phi[np.abs(phi).argmax(axis=0), np.arange(phi.shape[1])])
    if np.sum(lam < 1e-10) > 1:
        print(f"warning: L_RW has {np.sum(lam < 1e-10)} zero eigenvalues, so the graph is disconnected "
              f"(epsilon too small?); Algorithm 1 assumes it is connected")
    return phi[:, 1:m + 1], lam[:m + 1]
