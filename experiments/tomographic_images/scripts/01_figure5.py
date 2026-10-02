#!/usr/bin/env python3
"""
Figure 5 (Section 5.2): spectral embedding (Algorithm 1) of tomographic images
of the Glucagon molecule, each given a random in-plane rotation, with the
Euclidean kernel and the three SO(2)-invariant kernels.

For each SNR level (0 = clean; the paper's noisy panels use 1 dB):
  1. the first --n-samples projection images, plus noise at that SNR;
  2. expansion in FFBBasis2D with --ell-max (tools/fb_basis.py);
  3. a uniformly random rotation of each image, applied exactly to its
     coefficients;
  4. W for each kernel and bandwidth (tools/fb_kernels.py), then Algorithm 1
     (tools/spectral_embedding.py): phi_1, phi_2, ... of L_RW = I - D^{-1} W.

Bandwidths default to the sweep of the original code, one panel per value:
min and mean (the integral kernel) at 0.005, 0.01, 0.025, 0.05; bispectrum at
6e-9, 1.5e-8, 3e-8, 6e-8; Euclidean at 0.01.

Outputs in --out (default results/figure5):
  figures/[snr<k>/]<kernel>_bw<eps>.pdf      scatter of (phi_1, phi_2) coloured by the torsion angle
                                             (phi_2's sign set so a parabola opens upward; display only)
  figures/[snr<k>/]euclidean_random_angle_bw<eps>.pdf   the Euclidean embedding coloured by the applied rotation
  figures/[snr<k>/]comparison_grid.pdf        all panels of that SNR
  tabular/[snr<k>/]embedding_<tag>.csv        phi_1..phi_m, torsion angle, applied rotation per image
  tabular/[snr<k>/]spectrum_<tag>.npz         phi and lambda_0..lambda_m of L_RW
  run_config.json

    python3 scripts/01_figure5.py --data-dir data
    python3 scripts/01_figure5.py --data-dir data --snr-levels 1 --orbit-bandwidths 0.005
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import fb_basis  # noqa: E402
import fb_kernels  # noqa: E402
from spectral_embedding import spectral_embedding  # noqa: E402

FIGURE_SIZE, POINT_SIZE, DPI, CMAP = 4.5, 30.0, 600, "hsv"


def load_images(data_dir: Path, n: int) -> np.ndarray:
    import torch                                  # only here: never in the same process as ASPIRE
    proj = torch.load(data_dir / "projections-synth.pt", weights_only=False)
    if isinstance(proj, list):
        return np.array([p.numpy() for p in proj[:n]], dtype=np.float64)
    return proj[:n].numpy().astype(np.float64)


def load_torsion_angles(csv_path: Path, n: int) -> np.ndarray:
    """The ground-truth torsion angle (degrees) of frames 0..n-1, from output_main.csv (columns frame, angle)."""
    by_frame = {int(r["frame"]): float(r["angle"]) for r in csv.DictReader(open(csv_path, newline=""))}
    missing = [i for i in range(n) if i not in by_frame]
    if missing:
        raise KeyError(f"{csv_path} has no angle for frames {missing[:5]}...")
    return np.array([by_frame[i] for i in range(n)])


def opens_up(phi1, phi2):
    """phi2 with its sign chosen so that a parabola-shaped embedding opens upward.

    Eigenvector signs are arbitrary; this fixes phi2's for display only, without
    labels: phi2 is negated when it decreases with (phi1 - mean phi1)^2.
    """
    return -phi2 if np.dot(phi2 - phi2.mean(), (phi1 - phi1.mean()) ** 2) < 0 else phi2


def scatter(ax, phi, colour, units):
    import matplotlib.pyplot as plt  # noqa: F401
    x, y = phi[:, 0], opens_up(phi[:, 0], phi[:, 1])
    x, y = x / (x.std() or 1.0), y / (y.std() or 1.0)
    vmax = 360.0 if units == "deg" else 2 * np.pi
    ax.scatter(x, y, c=np.mod(colour, vmax), cmap=CMAP, vmin=0.0, vmax=vmax, marker="o", s=POINT_SIZE, linewidths=0)
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    mx, my = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    span = max(np.ptp(x), np.ptp(y)) / 2 * 1.1
    ax.set_xlim(mx - span, mx + span)
    ax.set_ylim(my - span, my + span)


def save_figure(phi, colour, units, path: Path):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    scatter(fig.add_axes([0.05, 0.05, 0.9, 0.9]), phi, colour, units)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="folder with projections-synth.pt and output_main.csv (see data/README.md)")
    ap.add_argument("--out", type=Path, default=Path("results/figure5"))
    ap.add_argument("--n-samples", type=int, default=1000, help="use the first n images (default 1000)")
    ap.add_argument("--snr-levels", type=float, nargs="+", default=[0, 1], help="SNR in dB; 0 = clean (default: 0 1)")
    ap.add_argument("--ell-max", type=int, default=10, help="angular cutoff of the Fourier-Bessel basis (default 10)")
    ap.add_argument("--num-angles", type=int, default=fb_kernels.NUM_ANGLES,
                    help="SO(2) angles for the min and mean kernels (default 600)")
    ap.add_argument("--orbit-bandwidths", type=float, nargs="+", default=[0.005, 0.01, 0.025, 0.05])
    ap.add_argument("--bispectrum-bandwidths", type=float, nargs="+", default=[6e-9, 1.5e-8, 3e-8, 6e-8])
    ap.add_argument("--euclidean-bandwidth", type=float, default=0.01)
    ap.add_argument("--kernels", nargs="+", choices=["min", "mean", "bispectrum", "euclidean"],
                    default=["min", "mean", "bispectrum", "euclidean"])
    ap.add_argument("--m", type=int, default=3, help="eigenvectors kept in the tables (default 3)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the noise and the random rotations")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = load_images(args.data_dir, args.n_samples)
    n = len(images)
    torsion = load_torsion_angles(args.data_dir / "output_main.csv", n)
    print(f"{n} images of {images.shape[1]}x{images.shape[2]}; torsion angle range "
          f"[{torsion.min():.1f}, {torsion.max():.1f}] deg")

    for k, snr in enumerate(args.snr_levels):
        sub = "" if snr == 0 else f"snr{snr:g}"
        fig_dir, tab_dir = args.out / "figures" / sub, args.out / "tabular" / sub
        fig_dir.mkdir(parents=True, exist_ok=True)
        tab_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng([args.seed, k])

        t0 = time.time()
        coefs, ang, sgn = fb_basis.expand(fb_kernels.add_noise(images, snr, rng), args.ell_max)
        thetas = rng.uniform(0, 2 * np.pi, size=n)
        coefs = fb_kernels.rotate(coefs, ang, sgn, thetas)
        print(f"\nSNR {snr:g} dB: expanded and rotated in {time.time() - t0:.1f}s ({coefs.shape[1]} coefficients)")

        runs = []          # (tag, label, log W)
        for method in ("min", "mean"):
            if method in args.kernels:
                for bw in args.orbit_bandwidths:
                    runs.append((f"{method}_bw{bw}", f"{method}\nbw={bw}",
                                 lambda bw=bw, method=method: fb_kernels.orbit_log_kernel(
                                     coefs, ang, sgn, bw, method, args.num_angles, chunk=50)))
        if "bispectrum" in args.kernels:
            bisp_d2 = fb_kernels.bispectrum_sq_distances(fb_basis.complex_blocks(coefs, ang, sgn))
            for bw in args.bispectrum_bandwidths:
                runs.append((f"bispectrum_bw{bw:.1e}", f"bispectrum\nbw={bw:.1e}", lambda bw=bw: -bisp_d2 / bw))
        if "euclidean" in args.kernels:
            bw = args.euclidean_bandwidth
            runs.append((f"euclidean_bw{bw}", f"Euclidean\nbw={bw}", lambda: -fb_kernels.sq_distances(coefs) / bw))

        grid = []
        for tag, label, log_w in runs:
            t0 = time.time()
            phi, lam = spectral_embedding(fb_kernels.weight_matrix(log_w()), args.m)
            np.savez_compressed(tab_dir / f"spectrum_{tag}.npz", phi=phi, eigenvalues=lam)
            with open(tab_dir / f"embedding_{tag}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["frame"] + [f"phi{j}" for j in range(1, args.m + 1)]
                           + ["torsion_angle_deg", "applied_rotation_rad"])
                w.writerows([i, *phi[i], torsion[i], thetas[i]] for i in range(n))
            if tag.startswith("euclidean"):
                bw_tag = tag.split("_", 1)[1]
                save_figure(phi, torsion, "deg", fig_dir / f"euclidean_no_random_angle_{bw_tag}.pdf")
                save_figure(phi, thetas, "rad", fig_dir / f"euclidean_random_angle_{bw_tag}.pdf")
                grid += [(label + "\ntorsion angle", phi, torsion, "deg"), (label + "\napplied rotation", phi, thetas, "rad")]
            else:
                save_figure(phi, torsion, "deg", fig_dir / f"{tag}.pdf")
                grid.append((label, phi, torsion, "deg"))
            print(f"  {tag}: lambda_1..2 = {np.round(lam[1:3], 5)}  ({time.time() - t0:.1f}s)")

        ncols = 4
        nrows = (len(grid) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), squeeze=False)
        for ax, (label, phi, colour, units) in zip(axes.flat, grid):
            scatter(ax, phi, colour, units)
            ax.set_title(label, fontsize=9)
        for ax in axes.flat[len(grid):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / "comparison_grid.pdf", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    (args.out / "run_config.json").write_text(json.dumps({k: (str(v) if isinstance(v, Path) else v)
                                                          for k, v in vars(args).items()}, indent=2))
    print(f"\ndone: {args.out}")


if __name__ == "__main__":
    main()
