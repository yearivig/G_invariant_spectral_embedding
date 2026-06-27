"""
Visualization utilities: 3D scatter plots of diffusion coordinates.

Refactored from the plotting functions in:
    ../roy_lederman_data/main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _ensure_3d_projection():
    """
    Ensure that matplotlib's '3d' projection is registered.

    On this system the *system* mpl_toolkits (from an older matplotlib) clashes
    with the user-installed matplotlib 3.10.  We fix it by forcing the user's
    mpl_toolkits path and re-registering the Axes3D projection class.
    """
    import matplotlib
    import mpl_toolkits

    # Point mpl_toolkits at the version that matches the user's matplotlib.
    user_mpl_toolkits = str(
        Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
    )
    if Path(user_mpl_toolkits).is_dir():
        mpl_toolkits.__path__ = [user_mpl_toolkits]

    # Clear cached stale sub-modules.
    for key in list(sys.modules):
        if key.startswith("mpl_toolkits.mplot3d"):
            del sys.modules[key]

    # Import the correct Axes3D and register it.
    from mpl_toolkits.mplot3d import axes3d as _a3d  # noqa: F811
    matplotlib.projections.projection_registry.register(_a3d.Axes3D)


# Run the fix once at import time.
_ensure_3d_projection()


def plot_3d_scatter(
    vectors: np.ndarray,
    i1: int,
    i2: int,
    i3: int,
    color_vals: np.ndarray,
    color_label: str = "",
    output_dir: str | os.PathLike | None = None,
    angles: list[tuple[int, int]] = ((30, 30), (45, 45), (60, 60)),
    dpi: int = 150,
    kernel_label: str = "",
) -> None:
    """
    Create a 3D scatter plot of three diffusion coordinates, optionally saving to disk.

    Parameters
    ----------
    vectors : (K, N) array of diffusion coordinates (rows = eigenvectors)
    i1, i2, i3 : 1-based indices of the eigenvectors to plot
    color_vals : (N,) array used for coloring (e.g. arctan2 of first two left-half eigvecs)
    color_label : label for filenames (e.g. "left" or "right")
    output_dir : if provided, save PDF + PNG here
    angles : list of (elevation, azimuth) viewing angles
    dpi : resolution for saved figures
    kernel_label : optional label for the kernel type (e.g. "euclidean", "min", "integral")
    """
    x = vectors[i1 - 1]
    y = vectors[i2 - 1]
    z = vectors[i3 - 1]

    # Normalize color to [0, 1] (treat as angles mod 2π)
    color_norm = ((color_vals % (2 * np.pi)) / (2 * np.pi)) % 1

    fig = plt.figure(figsize=(6 * len(angles), 6))
    fig.patch.set_facecolor('white')
    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, len(angles), idx + 1, projection="3d")
        ax.set_facecolor('white')
        ax.scatter(x, y, z, c=color_norm, cmap="hsv", s=20)
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Remove all axis lines and labels
        ax.set_axis_off()

    plt.tight_layout()

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{kernel_label}_" if kernel_label else ""
        base = f"{prefix}3d_scatter_{i1}{i2}{i3}_color_{color_label}"
        for ext in ("pdf", "png"):
            fname = out_dir / f"{base}.{ext}"
            fig.savefig(fname, dpi=dpi, facecolor='white', bbox_inches='tight')
        print(f"[viz] Saved: {out_dir / base}.{{pdf,png}}")

    plt.close(fig)
