#!/usr/bin/env python3
"""
Step 6: the panels of Figure 7 (Section 5.3), from the embeddings of script 04.

Six images for a kernel x colouring table: for each of the minimum, integral
and Euclidean kernels, the top and side stereographic views of phi_1..phi_4
(tools/view_stereo.py), coloured once by the true theta_A (twilight) and once
by the true theta_B (constant-lightness hue wheel); plus the two colour bars.
The minimum kernel sets the frame: each other kernel's eigenvector signs are
chosen so its two circles run the same way as the minimum kernel's, matched
image by image against that kernel and not against the true angles, so the
same colour sits in the same place in both panels. A sign is arbitrary (an
eigenvector is defined up to it), and a circle drawn with the wrong one is
mirrored, which is what made the integral panel a mirror of the minimum one.
The Euclidean circles follow the random rotation instead and match nothing:
they are left as they are.

No labels or titles: LaTeX sets those. Within a panel the side view slides
left over the top view, as far as the ink allows: the slide is measured from
the drawn points of all three kernels and the same value is used everywhere,
so no two views touch (the Euclidean curve is wide and flat, and would
otherwise run into its own second view) and every panel keeps one width. Each
panel is then cropped to its ink vertically (24 px margin), keeping its full
width, so all panels share one horizontal scale when set to the same width in
LaTeX.

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
PANEL, OVERLAP, DROP = 3.0, 0.8, 0.3   # inches: panel size, the most the side view may slide left, vertical nudge
GAP = 0.12                             # inches of white kept between the two views of a panel
MEASURE_DPI = 100                      # raster used only to find where the ink of a view ends
MARGIN = 24                            # px kept above and below the ink when cropping
KERNELS = ("min", "integral", "euclidean")
REFERENCE = "min"                      # the kernel whose frame the others are drawn in
MATCH_R = 0.5                          # below this the two kernels' circles track different variables


def circles(U: np.ndarray, pairs) -> list[np.ndarray]:
    """The recovered angle of each cos/sin pair, from the standardised eigenvectors."""
    z = lambda i: (U[i] - U[i].mean()) / U[i].std()
    return [np.arctan2(z(b), z(a)) for _, a, b in pairs]


def _resultant(d: np.ndarray) -> float:
    """How tightly a set of angle differences clusters: 1 for a constant offset, 0 for noise."""
    return float(np.hypot(np.cos(d).mean(), np.sin(d).mean()))


def align(key: str, U: np.ndarray, pairs, ref: list[np.ndarray]):
    """Put a kernel's circles in the reference kernel's frame: slot order and sign.

    Each recovered angle is matched image by image against the reference
    kernel's two, over both signs; the true angles play no part. A sign of -1
    means the circle runs backwards, which draws the panel mirrored, and is
    undone by flipping the second eigenvector of the pair - a sign an
    eigenvector does not fix. Circles that match nothing (resultant below
    MATCH_R) are left alone.
    """
    U, psi = U.copy(), circles(U, pairs)
    slots, flips = {}, []
    for n, (score, a, b) in enumerate(pairs):
        R, s, m = max((_resultant(ref[m] - s * psi[n]), s, m)
                      for m in range(len(ref)) for s in (1, -1))
        if R < MATCH_R:
            print(f"{key:9s} circle {n + 1}: no counterpart in the {REFERENCE} kernel (best match {R:.2f}), left as it is")
            return U, pairs
        off = np.degrees(np.arctan2(np.sin(ref[m] - s * psi[n]).mean(), np.cos(ref[m] - s * psi[n]).mean()))
        print(f"{key:9s} circle {n + 1} = {'+' if s > 0 else '-'}({REFERENCE} circle {m + 1}) + {off:6.1f} deg "
              f"(match {R:.3f}){', mirrored: flipping phi' + str(b + 1) if s < 0 else ''}")
        slots[m], flips = (score, a, b), flips + [b] * (s < 0)
    if sorted(slots) != list(range(len(pairs))):
        print(f"{key:9s} two circles matched the same one: frame left as it is")
        return U, pairs
    for b in flips:
        U[b] = -U[b]
    return U, [slots[m] for m in sorted(slots)]


def _axes(fig, rect):
    ax = fig.add_axes(rect, projection="3d")
    ax.set_facecolor("none")
    return ax


def ink_span(X3, lim, colour, elev, cmap) -> tuple[float, float]:
    """Where the ink of one view starts and ends, in inches from the left edge of its box."""
    fig = plt.figure(figsize=(PANEL, PANEL), dpi=MEASURE_DPI, facecolor="white")
    ax = _axes(fig, [0, -DROP / PANEL, 1, 1])
    vs._draw(ax, X3, lim, colour, elev, cmap)
    ax.set_box_aspect((1, 1, 1), zoom=1.1)
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].mean(axis=2)
    plt.close(fig)
    cols = np.where((a < 250).any(axis=0))[0]
    return cols[0] / MEASURE_DPI, (cols[-1] + 1) / MEASURE_DPI


def shared_overlap(panels: dict) -> float:
    """How far the side view may slide left in every panel, keeping GAP inches of white."""
    overlap = OVERLAP
    for key, (X3, lim, colour, cmap) in panels.items():
        _, left_end = ink_span(X3, lim, colour, VIEWS[0], cmap)
        right_start, _ = ink_span(X3, lim, colour, VIEWS[1], cmap)
        room = PANEL + right_start - left_end - GAP
        print(f"{key:9s} top view ends at {left_end:.2f}in, side view starts at {right_start:.2f}in "
              f"-> at most {room:.2f}in of slide")
        overlap = min(overlap, room)
    return max(0.0, overlap)


def draw_panels(run: Path, out: Path) -> None:
    loaded = {}
    for key in KERNELS:
        U, tA, tB, _ = vs.load_run(run, key, vs.SEARCH)
        loaded[key] = [U, vs.analyse(U)["pairs"], tA, tB]

    ref = circles(*loaded[REFERENCE][:2])
    for key in KERNELS:
        if key != REFERENCE:
            loaded[key][0], loaded[key][1] = align(key, loaded[key][0], loaded[key][1], ref)

    runs = {}
    for key, (U, pairs, tA, tB) in loaded.items():
        X3 = vs.coords(U, pairs, "stereo")
        runs[key] = (X3, float(np.percentile(np.abs(np.concatenate(X3)), 99.5)), tA, tB)

    overlap = shared_overlap({k: (X3, lim, tA, "twilight") for k, (X3, lim, tA, _) in runs.items()})
    print(f"side view slides {overlap:.2f}in left of the top view in every panel")
    W = 2 * PANEL - overlap

    for key, (X3, lim, tA, tB) in runs.items():
        for ang, colour, cmap in (("A", tA, vs.CMAP_A), ("B", tB, vs.CMAP_B)):
            fig = plt.figure(figsize=(W, PANEL), facecolor="white")
            for v, elev in enumerate(VIEWS):
                ax = _axes(fig, [v * (PANEL - overlap) / W, -DROP / PANEL, PANEL / W, 1])
                vs._draw(ax, X3, lim, colour, elev, cmap)
                ax.set_box_aspect((1, 1, 1), zoom=1.1)
            fig.savefig(out / f"stereo_{key}_theta{ang}.png", dpi=300, facecolor="white")
            plt.close(fig)

    # the two colour bars, as separate images so the table can share one per column
    for ang, cmap in (("A", vs.CMAP_A), ("B", vs.CMAP_B)):
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
