"""
Render eigenvector .pkl files into clean PDF scatter plots.

This is a documented variant of
    ../comparing_to_yoel_and_eitan/save_figures.py

The .pkl files contain eigenvector arrays of shape (n, n).
We plot eigenvector[1] vs eigenvector[2] (the first two non-trivial
eigenvectors of the RW graph Laplacian) as a 2D scatter plot.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------
# Plot styling (consistent with project conventions)
# ---------------------------------------------------------------------------

FIGURES_DPI = 600

_RCPARAMS_LATEX_SINGLE_COLUMN = {
    "font.family": "serif",
    "text.usetex": True,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "axes.prop_cycle": matplotlib.pyplot.cycler(
        "color", ["#ff7d66", "#ffdc30", "#40a0cc", "#529915", "#8b8b8b"]
    )
    + matplotlib.pyplot.cycler("marker", ["d", "s", "o", r"$\clubsuit$", ">"]),
    "lines.markersize": 9,
    "lines.markeredgewidth": 0.75,
    "lines.markeredgecolor": "k",
    "grid.color": "#C0C0C0",
    "legend.fancybox": True,
    "legend.framealpha": 0.8,
    "axes.linewidth": 1,
}

_PAGE_WIDTH_INCHES = 6.775
_GOLDEN_RATIO = (5**0.5 - 1) / 2
_WIDTH = _PAGE_WIDTH_INCHES
_HEIGHT = _GOLDEN_RATIO * _WIDTH

RCPARAMS_LATEX_DOUBLE_COLUMN = {
    **_RCPARAMS_LATEX_SINGLE_COLUMN,
    "figure.figsize": (_WIDTH / 2, _HEIGHT / 2),
}


def render_eigenvectors_to_pdf(
    pkl_path: str | os.PathLike,
    pdf_dir: str | os.PathLike,
) -> Path:
    """
    Load eigenvectors from a .pkl file and render a 2D scatter plot as PDF.

    The plot shows eigenvector[1] vs eigenvector[2] — the first two
    non-trivial eigenvectors of the spectral embedding.

    Parameters
    ----------
    pkl_path : path to the eigenvectors .pkl file
    pdf_dir : directory where the PDF will be saved

    Returns
    -------
    Path to the generated PDF file.
    """
    pkl_path = Path(pkl_path)
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    with pkl_path.open("rb") as f:
        eigvectors = pickle.load(f)

    # eigvectors has shape (n, n): rows are eigenvectors.
    # Index 0 is the trivial eigenvector; 1 and 2 are the first non-trivial pair.
    x = np.real(eigvectors[1])
    y = np.real(eigvectors[2])

    colors = np.sin(np.mod(np.arange(len(y)), 126) * np.pi / 126)

    with plt.rc_context(rc=RCPARAMS_LATEX_DOUBLE_COLUMN):
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=colors, cmap="rainbow", marker="o", s=10)

        ax.set_frame_on(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title("")

        out_path = pdf_dir / (pkl_path.stem + ".pdf")
        fig.savefig(out_path, dpi=FIGURES_DPI, bbox_inches="tight")
        plt.close(fig)

    return out_path


def render_all_pkls_in_directory(
    pkl_dir: str | os.PathLike,
    pdf_subdir: str = "plots",
) -> list[Path]:
    """
    Convert every .pkl file in `pkl_dir` into a PDF inside `pkl_dir/<pdf_subdir>/`.
    """
    pkl_dir = Path(pkl_dir)
    if not pkl_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {pkl_dir}")

    pdf_dir = pkl_dir / pdf_subdir
    generated: list[Path] = []

    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        generated.append(render_eigenvectors_to_pdf(pkl_path, pdf_dir))

    return generated


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = Path(__file__).resolve().parent / "outputs"

    pdfs = render_all_pkls_in_directory(target_dir)
    print(f"Rendered {len(pdfs)} PDFs into {target_dir / 'plots'}")
