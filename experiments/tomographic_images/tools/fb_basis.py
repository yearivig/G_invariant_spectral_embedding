"""
Fourier-Bessel expansion of the images with ASPIRE's FFBBasis2D (Section 5.2).

expand(images, ell_max) returns the real coefficients of
FFBBasis2D((L, L), ell_max).expand(images, tol=1e-2) and their angular and sign
indices. In-plane rotation by theta acts on each angular frequency ell >= 1 as
a 2 x 2 rotation of the (sign +1, sign -1) coefficient pairs; complex_blocks()
turns those pairs into ASPIRE's complex coefficients a_+ - i a_-, which a
rotation multiplies by exp(i ell theta) (checked against FFBBasis2D.to_complex).

torch (used to read the .pt data) and ASPIRE's NUFFT (finufft) each ship their
own OpenMP runtime, and on macOS a process that loads both aborts or deadlocks;
ASPIRE pulls torch in through pymanopt whenever torch is installed. The
expansion therefore always runs in a subprocess (this file run as a script) in
which torch cannot be imported.
"""

from __future__ import annotations

import importlib.abc
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


class _NoTorch(importlib.abc.MetaPathFinder):
    """Makes `import torch` fail, so ASPIRE's optional torch backends stay unloaded."""

    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is kept out of the Fourier-Bessel expansion (OpenMP clash with finufft)")
        return None


def expand(images: np.ndarray, ell_max: int, tol: float = 1e-2, batch_size: int = 256):
    """(coefs (n, count), angular_indices (count,), signs_indices (count,)) of square images (n, L, L)."""
    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 3 or images.shape[1] != images.shape[2]:
        raise ValueError(f"images must be a stack of square images, got shape {images.shape}")
    with tempfile.TemporaryDirectory(prefix="fb_expand_") as tmp:
        src, dst = Path(tmp) / "images.npy", Path(tmp) / "coefs.npz"
        np.save(src, images)
        subprocess.run([sys.executable, __file__, str(src), str(dst), str(ell_max), str(tol), str(batch_size)],
                       check=True)
        out = np.load(dst)
        return out["coefs"], out["angular_indices"], out["signs_indices"]


def complex_blocks(coefs: np.ndarray, angular: np.ndarray, signs: np.ndarray) -> list[np.ndarray]:
    """[c_0, ..., c_ellmax]: c_ell (n, k_ell) the complex coefficients a_+ - i a_- at angular frequency ell."""
    blocks = []
    for ell in range(int(angular.max()) + 1):
        pos = (angular == ell) & (signs == 1)
        neg = (angular == ell) & (signs == -1)
        blocks.append(coefs[:, pos].astype(np.complex128) if ell == 0 else coefs[:, pos] - 1j * coefs[:, neg])
    return blocks


def _expand_here(images, ell_max, tol, batch_size):
    from aspire.basis import FFBBasis2D
    ffb = FFBBasis2D(images.shape[1:], ell_max=ell_max, dtype=np.float64)
    coefs = np.concatenate([ffb.expand(np.asarray(images[i:i + batch_size]), tol=tol).asnumpy()
                            for i in range(0, len(images), batch_size)])
    return coefs, np.asarray(ffb.angular_indices), np.asarray(ffb.signs_indices)


if __name__ == "__main__":
    sys.meta_path.insert(0, _NoTorch())
    src, dst, ell_max, tol, batch_size = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    coefs, ang, sgn = _expand_here(np.load(src, mmap_mode="r"), ell_max, tol, batch_size)
    np.savez(dst, coefs=coefs, angular_indices=ang, signs_indices=sgn)
