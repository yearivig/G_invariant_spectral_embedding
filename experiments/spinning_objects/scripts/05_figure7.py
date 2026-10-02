#!/usr/bin/env python3
"""
Step 6: the panels of Figure 7 (Section 5.3), from the embeddings of script 04.

Six images for a kernel x colouring table: for each of the minimum, integral
and Euclidean kernels, the top and side stereographic views of phi_1..phi_4
(tools/view_stereo.py), coloured once by the true theta_A (twilight) and once
by the true theta_B (constant-lightness hue wheel); plus the two colour bars.
No labels or titles: LaTeX sets those. Each panel is then cropped to its ink
vertically (24 px margin), keeping its full width, so all panels share one
horizontal scale when set to the same width in LaTeX.

    python3 scripts/05_figure7.py --run results/run1          # -> results/run1/figure7/

Outputs: stereo_<kernel>_theta{A,B}.png and stereo_colorbar_theta{A,B}.png,
300 dpi, with <kernel> = min, integral, euclidean.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import view_stereo as vs  # noqa: E402

VIEWS = (88, 12)                       # top, side: the elevations view_stereo.save_figure uses
PANEL, OVERLAP, DROP = 3.0, 0.8, 0.3   # inches: panel size, how far the side view slides left, vertical nudge
MARGIN = 24                            # px kept above and below the ink when cropping


def draw_panels(run: Path, out: Path) -> None:
    W = 2 * PANEL - OVERLAP
    for key in ("min", "integral", "euclidean"):
        U, tA, tB, _ = vs.load_run(run, key, vs.SEARCH)
        X3 = vs.coords(U, vs.analyse(U)["pairs"], "stereo")
        lim = float(np.percentile(np.abs(np.concatenate(X3)), 99.5))
        for ang, colour, cmap in (("A", tA, "twilight"), ("B", tB, vs.hue_wheel())):
            fig = plt.figure(figsize=(W, PANEL), facecolor="white")
            for v, elev in enumerate(VIEWS):
                ax = fig.add_axes([v * (PANEL - OVERLAP) / W, -DROP / PANEL, PANEL / W, 1], projection="3d")
                ax.set_facecolor("none")
                vs._draw(ax, X3, lim, colour, elev, cmap)
                ax.set_box_aspect((1, 1, 1), zoom=1.1)
            fig.savefig(out / f"stereo_{key}_theta{ang}.png", dpi=300, facecolor="white")
            plt.close(fig)

    # the two colour bars, as separate images so the table can share one per column
    for ang, cmap in (("A", "twilight"), ("B", vs.hue_wheel())):
        fig = plt.figure(figsize=(0.75, PANEL), facecolor="white")
        cax = fig.add_axes([0.08, 0.06, 0.18, 0.88])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 2 * np.pi), cmap=cmap),
                          cax=cax, ticks=vs.TICKS)
        cb.ax.set_yticklabels(vs.TICKLABELS, fontsize=11)
        cb.outline.set_linewidth(0.6)
        fig.savefig(out / f"stereo_colorbar_theta{ang}.png", dpi=300, facecolor="white")
        plt.close(fig)


def crop_panels(out: Path) -> None:
    """Trim the blank rows above and below the ink in each panel, keeping full width."""
    for p in sorted(out.glob("stereo_*_theta[AB].png")):
        a = np.asarray(Image.open(p).convert("L"))
        rows = np.where((a < 250).any(axis=1))[0]
        top, bot = max(0, rows[0] - MARGIN), min(a.shape[0], rows[-1] + 1 + MARGIN)
        im = Image.open(p).crop((0, top, a.shape[1], bot))
        im.save(p, dpi=(300, 300))
        print(f"{p.name:34s} {a.shape[1]}x{a.shape[0]} -> {im.width}x{im.height}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=Path("results/run1"),
                    help="script-04 output folder with the euclidean, so2_min and so2_integral eigenvectors")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <run>/figure7)")
    args = ap.parse_args()
    out = args.out or args.run / "figure7"
    out.mkdir(parents=True, exist_ok=True)
    draw_panels(args.run, out)
    crop_panels(out)
    print(f"Figure 7 panels in {out}/")


if __name__ == "__main__":
    main()
