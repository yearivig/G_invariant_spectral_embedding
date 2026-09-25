#!/usr/bin/env python3
"""
Show a torus embedding in 3D by stereographic projection, without using labels.

A flat torus needs four coordinates, one cos/sin pair per circle. This takes
the first four eigenvectors, pairs them up, puts the points on the 3-sphere
and projects them to 3D. The true angles are used only for colour. Script 04
calls save_figure() for every run; this script redraws or rotates the result.

  1. Pair selection. For a cos/sin pair, psi_i^2 + psi_j^2 is the same for
     every image (cos^2 + sin^2 = 1). Each eigenvector is standardised, and
     every pair among psi_1..psi_4 (--search) is scored by the spread of
     (psi_i^2 + psi_j^2) / 2 over the images: 0 for a perfect circle, about
     0.5 for two unrelated eigenvectors. The two best pairs that share no
     eigenvector are taken. If either scores above --threshold, the script
     says no torus was found (and still plots, so you can see why).
  2. Independence. On a torus the two recovered angles, atan2 of each pair,
     are independent, so the points cover the whole square of angle pairs.
     If both pairs follow one variable (say phi and 2 phi), they trace a
     single curve and cover only the cells it crosses. The square is cut into
     a 12 x 12 grid, and at least --coverage of the cells must hold a point.
     Measured on obj87 + obj72: 100% for the unrotated images (a torus), 25%
     for the Euclidean kernel on the rotated images, whose two clean circles
     both follow the random rotation.
  Neither test says which variable a circle belongs to. Only the colouring by
  the true angles shows whether the circles are theta_A and theta_B.
  3. --style stereo (default): each point of the four standardised
     coordinates is scaled onto the unit 3-sphere, then projected:
     (x1, x2, x3) / (1 - x4).
     --style donut: each pair gives a recovered angle, atan2(second, first),
     and the two angles go into the torus of revolution
     ((R + cos b) cos a, (R + cos b) sin a, sin b). The donut shape is imposed
     by the formula; only the colouring shows whether the angles are right.

The eigenvectors are those of Algorithm 1 (L_RW = I - D^-1 W), unscaled.
Stereographic projection keeps angles but not distances: points near the
projection pole are magnified.

The input is a script-04 output folder (--run), using the SO(2) minimum
kernel's eigenvectors by default (--kernel euclidean for the baseline), or an
.npz file (--npz) with arrays eigvecs_unscaled (k x N), theta_A and theta_B.

Usage:
    python3 tools/view_stereo.py                                # results/run1, min kernel, window
    python3 tools/view_stereo.py --color B                      # colour by theta_B
    python3 tools/view_stereo.py --kernel euclidean             # the Euclidean baseline
    python3 tools/view_stereo.py --style donut                  # recovered angles on a donut
    python3 tools/view_stereo.py --save results/run1/stereo.png # four-panel figure (+ .pdf)

Script 04 already writes results/run1/{min,euclidean}_{stereo,donut}.{png,pdf}.

In the window, drag to rotate and scroll or right-drag to zoom.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

KERNEL_FILES = {"min": ("so2_min", "SO(2) minimum kernel"),
                "euclidean": ("euclidean", "Euclidean kernel"),
                "integral": ("so2_integral", "SO(2) integral kernel")}


def load_run(run: Path, kernel: str, k: int):
    """First k eigenvectors phi_1..phi_k of a script-04 run, plus the true angles (colour only).

    Reads <name>_eigenvectors.pkl (Algorithm 1). Older runs made with
    diffusion maps saved <name>_diffusion_vectors.pkl scaled by lambda^t; for
    those the scaling is undone with the saved eigenvalues and t.
    """
    name, label = KERNEL_FILES[kernel]
    L = lambda f: np.asarray(pickle.load(open(run / f, "rb")))
    if (run / f"{name}_eigenvectors.pkl").is_file() and (run / "true_theta_A.pkl").is_file():
        U = L(f"{name}_eigenvectors.pkl")[:k]
    elif all((run / f).is_file() for f in ("run_config.json", f"{name}_diffusion_vectors.pkl",
                                              f"{name}_eigenvalues.pkl", "true_theta_A.pkl")):
        t = json.loads((run / "run_config.json").read_text())["t"]
        ev = L(f"{name}_eigenvalues.pkl")
        U = L(f"{name}_diffusion_vectors.pkl")[:k] / (ev[1:k + 1] ** t)[:, None]
        label += " (diffusion-maps run, lambda^t removed)"
    else:
        raise SystemExit(f"no {name} eigenvectors in {run} - point --run at a script-04 output folder "
                         f"(bash run_all.sh writes results/run1), or use --npz")
    return U, L("true_theta_A.pkl"), L("true_theta_B.pkl"), f"{label}, {run}"


def load_npz(path: Path, k: int):
    z = np.load(path)
    return z["eigvecs_unscaled"][:k], z["theta_A"], z["theta_B"], str(path)


def select_pairs(U: np.ndarray):
    """The two best cos/sin pairs that share no eigenvector, and every pair's score."""
    S = [(u - u.mean()) / u.std() for u in U]
    scores = sorted((float(np.std(S[i] ** 2 + S[j] ** 2) / 2), i, j)
                    for i in range(len(S)) for j in range(i + 1, len(S)))
    first = scores[0]
    second = next(t for t in scores[1:] if not {t[1], t[2]} & {first[1], first[2]})
    return sorted([first, second], key=lambda t: t[1]), scores


def stereo(U: np.ndarray, pairs) -> np.ndarray:
    """Stereographic image in R^3 of the two pairs, ordered (p1 cos, p2 cos, p1 sin, p2 sin)."""
    (_, a, b), (_, c, d) = pairs
    X = np.stack([(U[i] - U[i].mean()) / U[i].std() for i in (a, c, b, d)])
    X /= np.linalg.norm(X, axis=0)
    return X[:3] / (1 - X[3])


def donut(U: np.ndarray, pairs, R: float = 2.4) -> np.ndarray:
    """Recovered angle from each pair, placed on a torus of revolution (first pair around the ring)."""
    z = lambda i: (U[i] - U[i].mean()) / U[i].std()
    (_, a, b), (_, c, d) = pairs
    ring, tube = np.arctan2(z(b), z(a)), np.arctan2(z(d), z(c))
    return np.stack([(R + np.cos(tube)) * np.cos(ring), (R + np.cos(tube)) * np.sin(ring), np.sin(tube)])


def hue_wheel(lightness: float = 0.70, chroma: float = 0.11, n: int = 256):
    """A cyclic colour map at constant lightness: a circle in the OKLab a-b plane.

    Every hue is equally bright and equally spaced, so no angle stands out.
    Used for theta_B, so it never looks like twilight (theta_A), which varies
    in lightness instead.
    """
    from matplotlib.colors import ListedColormap
    h = np.linspace(0, 2 * np.pi, n, endpoint=False)
    L, a, b = np.full(n, lightness), chroma * np.cos(h), chroma * np.sin(h)
    l_, m_, s_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, \
                 (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, \
                 (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = np.stack([4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
                    -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
                    -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_], axis=1)
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)  # linear -> sRGB
    return ListedColormap(rgb, name="oklab_wheel")


THRESHOLD = 0.45     # a pair scoring above this is not a circle (0.5 = two unrelated eigenvectors)
COVERAGE = 0.80      # the two recovered angles must fill at least this share of a 12 x 12 grid
SEARCH = 4           # use the first four eigenvectors
TICKS, TICKLABELS = np.linspace(0, 2 * np.pi, 5), ["0°", "90°", "180°", "270°", "360°"]
INK, MUTED = "#1f2328", "#6e7781"


def angle_coverage(U: np.ndarray, pairs, bins: int = 12) -> float:
    """Share of a bins x bins grid of (recovered angle 1, recovered angle 2) that holds a point."""
    z = lambda i: (U[i] - U[i].mean()) / U[i].std()
    (_, a, b), (_, c, d) = pairs
    H, _, _ = np.histogram2d(np.arctan2(z(b), z(a)), np.arctan2(z(d), z(c)),
                             bins=bins, range=[[-np.pi, np.pi]] * 2)
    return float((H > 0).mean())


def analyse(U: np.ndarray, threshold: float = THRESHOLD, coverage: float = COVERAGE) -> dict:
    """Pair psi_1..psi_k into two circles and check the two angles are independent."""
    pairs, scores = select_pairs(U)
    circles = all(s <= threshold for s, _, _ in pairs)
    cov = angle_coverage(U, pairs)
    ok = circles and cov >= coverage
    if ok:
        verdict = f"torus found (angle coverage {cov:.0%})"
    elif not circles:
        verdict = f"NO TORUS: a pair scores above {threshold}"
    else:
        verdict = f"NO TORUS: two circles, but they cover only {cov:.0%} of the angle square (one curve)"
    return {"pairs": pairs, "scores": scores, "circles": circles, "coverage": cov, "ok": ok, "verdict": verdict,
            "chosen": " and ".join(f"(φ{i + 1}, φ{j + 1}) {s:.2f}" for s, i, j in pairs)}


def report(title: str, res: dict) -> None:
    print(title)
    print(f"  circle scores (0 = perfect, ~0.5 = unrelated): " +
          ", ".join(f"(φ{i + 1},φ{j + 1}) {s:.2f}" for s, i, j in res["scores"][:6]))
    print(f"  chosen: {res['chosen']}, angle coverage {res['coverage']:.0%}  ->  {res['verdict']}")


def coords(U: np.ndarray, pairs, style: str):
    if style == "stereo":
        Y = stereo(U, pairs)
        return (Y[0], Y[2], Y[1])                # the second pair's cos axis drawn vertical
    return tuple(donut(U, pairs))                # ring in the horizontal plane


def _draw(ax, X3, lim, colour, elev, cmap):
    sc = ax.scatter(*X3, c=colour, cmap=cmap, vmin=0, vmax=2 * np.pi, s=3, linewidths=0, depthshade=False)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=-60)
    return sc


def _subtitle(res: dict, style: str) -> str:
    (_, a, b), (_, c, d) = res["pairs"]
    method = "stereographic projection" if style == "stereo" else "recovered angles on a donut"
    return f"{method}; pairs (φ{a + 1}, φ{b + 1}) and (φ{c + 1}, φ{d + 1}), chosen without labels: {res['verdict']}"


def save_figure(U, tA, tB, title: str, style: str, path: Path, threshold: float = THRESHOLD,
                coverage: float = COVERAGE) -> dict:
    """Four-panel figure (top/side view x theta_A/theta_B colouring) as PNG + PDF. Returns the analysis."""
    import matplotlib.pyplot as plt
    res = analyse(U, threshold, coverage)
    X3 = coords(U, res["pairs"], style)
    lim = float(np.percentile(np.abs(np.concatenate(X3)), 99.5))
    cmaps = {"A": "twilight", "B": hue_wheel()}  # a different cyclic palette for each angle
    fig = plt.figure(figsize=(12, 10), facecolor="white")
    for row, (c_, name, key) in enumerate(((tA, r"$\theta_A$", "A"), (tB, r"$\theta_B$", "B"))):
        for col, (elev, view) in enumerate(((88, "top view"), (12, "side view"))):
            ax = fig.add_subplot(2, 2, 2 * row + col + 1, projection="3d")
            sc = _draw(ax, X3, lim, c_, elev, cmaps[key])
            ax.set_title(f"{view}, coloured by true {name}", color=INK, fontsize=11)
        cb = fig.colorbar(sc, ax=fig.axes[-2:], shrink=0.6, pad=0.02, ticks=TICKS)
        cb.ax.set_yticklabels(TICKLABELS, color=MUTED)
        cb.set_label(f"true θ_{key}", color=INK)
        cb.outline.set_visible(False)
    fig.suptitle(f"{title}\n{_subtitle(res, style)}", color=INK, fontsize=12)
    note = ("each point scaled onto the unit 3-sphere, then projected as (x1, x2, x3) / (1 − x4)"
            if style == "stereo" else
            "θ̂ = atan2 of each pair, drawn as ((R + cos θ̂₂) cos θ̂₁, (R + cos θ̂₂) sin θ̂₁, sin θ̂₂); the shape is imposed, "
            "the colours are the evidence")
    fig.text(0.45, 0.04, f"First four eigenvectors φ₁–φ₄ of L_RW (Algorithm 1), standardised; {note}. "
             "True angles used only for colour.", ha="center", color=MUTED, fontsize=9)
    path = Path(path)
    fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=Path("results/run1"), help="script-04 output folder")
    ap.add_argument("--kernel", choices=list(KERNEL_FILES), default="min", help="which kernel's eigenvectors")
    ap.add_argument("--npz", type=Path, default=None, help="read eigenvectors from an .npz file instead of --run")
    ap.add_argument("--title", default=None, help="figure title (default: kernel and input path)")
    ap.add_argument("--style", choices=["stereo", "donut"], default="stereo",
                    help="stereographic projection, or the recovered angles drawn on a donut")
    ap.add_argument("--color", choices=["A", "B"], default="A", help="window only: colour by true theta_A or theta_B")
    ap.add_argument("--search", type=int, default=SEARCH, help="pair up psi_1..psi_search (default: the first four)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="a pair scoring above this is not accepted as a circle (0.5 = unrelated)")
    ap.add_argument("--coverage", type=float, default=COVERAGE,
                    help="minimum share of the 12 x 12 angle grid the two recovered angles must fill")
    ap.add_argument("--save", type=Path, default=None,
                    help="write a four-panel figure (top/side view x both colourings) instead of opening a window")
    args = ap.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    U, tA, tB, source = load_npz(args.npz, args.search) if args.npz else load_run(args.run, args.kernel, args.search)
    title = args.title or source
    if args.save:
        res = save_figure(U, tA, tB, title, args.style, args.save, args.threshold, args.coverage)
        report(title, res)
        print(f"  saved {args.save.with_suffix('.png')} and .pdf")
        return

    res = analyse(U, args.threshold, args.coverage)
    report(title, res)
    X3 = coords(U, res["pairs"], args.style)
    lim = float(np.percentile(np.abs(np.concatenate(X3)), 99.5))
    cmap = "twilight" if args.color == "A" else hue_wheel()
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = fig.add_subplot(projection="3d")
    sc = _draw(ax, X3, lim, tA if args.color == "A" else tB, 35, cmap)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, ticks=TICKS)
    cb.ax.set_yticklabels(TICKLABELS)
    cb.set_label(f"true θ_{args.color}")
    ax.set_title(f"{title}\n{_subtitle(res, args.style)}\n(drag to rotate)", fontsize=10)
    plt.show()


if __name__ == "__main__":
    main()
