"""
Render `.pkl` experiment outputs into clean `.pdf` scatter plots.

This is a documented variant of `print_plots/print_from_pkl.py`.

Convention used in this codebase:
- experiment scripts save embeddings as `outputs/<name>....pkl`
- this module converts them into `outputs/new_plots/<same-name>.pdf`

The `.pkl` file format used here is a NumPy array/list containing:
    data[0] = x coordinates
    data[1] = y coordinates
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------
# Plot styling (kept consistent with project figure style)
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


def render_2d_pkl_to_pdf(pkl_path: str | os.PathLike, pdf_dir: str | os.PathLike) -> Path:
    """
    Load a 2D embedding from `<pkl_path>` and save a minimalist PDF to `<pdf_dir>`.

    The resulting PDF is saved as:
        <pdf_dir>/<basename(pkl_path) without ".pkl">.pdf

    The plot is intentionally *axis-free* for paper-ready inclusion.
    """
    pkl_path = Path(pkl_path)
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    x_array = np.asarray(data[0])
    y_array = np.asarray(data[1])

    # Color by a deterministic periodic function of the index (project convention).
    colors = np.sin(np.mod(np.arange(len(y_array)), 126) * np.pi / 126)

    with plt.rc_context(rc=RCPARAMS_LATEX_DOUBLE_COLUMN):
        fig, ax = plt.subplots()
        ax.scatter(x_array, y_array, c=colors, cmap="rainbow", marker="o", s=10)

        # Remove frame, axes, and title for clean figure export.
        ax.set_frame_on(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title("")

        out_path = pdf_dir / (pkl_path.stem + ".pdf")
        fig.savefig(out_path, dpi=FIGURES_DPI, bbox_inches="tight")
        plt.close(fig)

    return out_path


def render_all_pkls_in_directory(outputs_dir: str | os.PathLike) -> list[Path]:
    """
    Convert every `.pkl` file in `outputs_dir` into a PDF inside `outputs_dir/new_plots/`.
    Returns a list of generated PDF paths.
    """
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {outputs_dir}")

    pdf_dir = outputs_dir / "new_plots"
    generated: list[Path] = []

    for pkl_path in sorted(outputs_dir.glob("*.pkl")):
        generated.append(render_2d_pkl_to_pdf(pkl_path, pdf_dir))

    return generated


if __name__ == "__main__":
    # Convenience: render everything under the default project outputs dir.
    default_outputs = Path(__file__).resolve().parents[1] / "outputs"
    pdfs = render_all_pkls_in_directory(default_outputs)
    print(f"Rendered {len(pdfs)} PDFs into {default_outputs / 'new_plots'}")

