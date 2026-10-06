"""
Render saved embeddings (.pkl) as the axis-free scatter plots of Figure 4, with
one scale for both axes, so a circle is drawn as a circle.

A .pkl holds [phi_1, phi_2, ...]; the first two coordinates are plotted. Point i
is coloured by sin(pi (i mod 126) / 126), a function of its frame index, with
matplotlib's "rainbow" map, as in the paper. Figures use LaTeX text rendering
(text.usetex), so a LaTeX installation is needed.

Eigenvector signs are arbitrary, so the minimum and invariant-features panels
are drawn with the signs (and order) of phi_1, phi_2 that best match the
integral-kernel panel of the same point clouds, when that file is in the same
folder; all panels of one row then face the same way. Display only: the saved
embeddings are unchanged.

    python3 tools/plot_embedding.py results/figure4      # results/figure4/embeddings/*.pkl -> results/figure4/plots/
"""

from __future__ import annotations

import glob
import os
import pickle
import re
from pathlib import Path

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


def _load_xy(pkl_path: Path) -> np.ndarray:
    """(phi_1, phi_2) of a saved embedding, as an (n, 2) array."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    return np.column_stack([np.asarray(data[0], dtype=float), np.asarray(data[1], dtype=float)])


# Panels whose orientation follows the integral kernel's panel on the same point clouds.
_ALIGNED = ("min", "invariant_features")
_PANEL = re.compile(r"^(?P<pre>.*_IM:)(?P<kernel>[a-z_]+)(?P<mid>_M:rotation_BW:)[^_]+(?P<post>_IC:.*?_LT:RWGL)(?P<tags>.*)$")


def _reference_pkl(pkl_path: Path) -> Path | None:
    """The integral-kernel embedding of the same point clouds, if pkl_path is a minimum or
    invariant-features panel and that file exists next to it."""
    m = _PANEL.match(pkl_path.stem)
    if m is None or m["kernel"] not in _ALIGNED:
        return None
    tags = re.sub(r"_MIN:[a-z]+", "", m["tags"])                  # the integral kernel has no _MIN tag
    hits = sorted(pkl_path.parent.glob(f"{glob.escape(m['pre'])}integral{glob.escape(m['mid'])}*"
                                       f"{glob.escape(m['post'] + tags)}.pkl"))
    hits = [h for h in hits if _PANEL.match(h.stem) and _PANEL.match(h.stem)["tags"] == tags]
    return hits[0] if len(hits) == 1 else None


def _match_orientation(xy: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """xy with the signs and order of its two columns chosen to best match ref.

    Eigenvector signs are arbitrary (and for a circle the order of the cos/sin
    pair is too), so this changes only how the embedding is drawn: of the eight
    signed permutations of (phi_1, phi_2), the one closest to the reference
    embedding of the same point clouds is used.
    """
    best, best_fit = xy, -np.inf
    for cols in ((0, 1), (1, 0)):
        for sx in (1, -1):
            for sy in (1, -1):
                cand = xy[:, cols] * np.array([sx, sy])
                fit = float(np.sum(cand * ref))
                if fit > best_fit:
                    best, best_fit = cand, fit
    return best


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

    xy = _load_xy(pkl_path)
    ref = _reference_pkl(pkl_path)
    if ref is not None:
        xy = _match_orientation(xy, _load_xy(ref))
    x_array, y_array = xy[:, 0], xy[:, 1]

    # Color by a deterministic periodic function of the index (project convention).
    colors = np.sin(np.mod(np.arange(len(y_array)), 126) * np.pi / 126)

    with plt.rc_context(rc=RCPARAMS_LATEX_DOUBLE_COLUMN):
        fig, ax = plt.subplots()
        ax.scatter(x_array, y_array, c=colors, cmap="rainbow", marker="o", s=10)
        ax.set_aspect("equal", adjustable="box")   # one scale for both axes: shapes as computed

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
    Convert every `.pkl` file in `outputs_dir/embeddings/` into a PDF inside `outputs_dir/plots/`.
    Returns a list of generated PDF paths.
    """
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {outputs_dir}")

    pdf_dir = outputs_dir / "plots"
    generated: list[Path] = []

    for pkl_path in sorted((outputs_dir / "embeddings").glob("*.pkl")):
        generated.append(render_2d_pkl_to_pdf(pkl_path, pdf_dir))

    return generated


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit(__doc__)
    pdfs = render_all_pkls_in_directory(sys.argv[1])
    print(f"Rendered {len(pdfs)} PDFs into {Path(sys.argv[1]) / 'plots'}")
