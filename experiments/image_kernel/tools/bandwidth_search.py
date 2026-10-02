#!/usr/bin/env python3
"""
Re-tune the bandwidths of the Section 5.2 kernels: median heuristic + sweep.

NOTE: this tool is for re-tuning bandwidths (e.g. for another dataset or a
revision). It is not part of producing Figure 5, whose bandwidths are set in
scripts/01_figure5.py.

On the same images as scripts/01_figure5.py (first --n-samples, noise at --snr,
FFBBasis2D expansion, random rotations), it
  1. prints percentiles of the squared distances: Euclidean (all images),
     SO(2)-minimum and bispectrum (on the first --subsample images);
  2. proposes candidates eps = median x f for f in --factors: the Euclidean
     median for the orbit kernels, the bispectrum median for the bispectrum;
  3. for each candidate, on the subsample, runs Algorithm 1 with the minimum
     kernel and with the bispectrum kernel, scores it by the spectral gap
     mu_1 / mu_2 (mu = 1 - lambda, the eigenvalues of D^{-1} W; larger = the
     leading coordinate stands out more), and saves a scatter plot;
  4. reports the best-scoring candidate of each and writes everything to
     --out/bandwidth_results.json.
The score is a guide; inspect the plots before choosing.

    python3 tools/bandwidth_search.py --data-dir data --out results/bandwidth_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fb_basis  # noqa: E402
import fb_kernels  # noqa: E402
from spectral_embedding import spectral_embedding  # noqa: E402

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def upper(d2: np.ndarray) -> np.ndarray:
    return d2[np.triu_indices(len(d2), k=1)]


def gap(lam: np.ndarray) -> float:
    mu = 1.0 - lam
    return float(mu[1] / mu[2]) if abs(mu[2]) > 1e-15 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/bandwidth_sweep"))
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--subsample", type=int, default=200, help="images used for the sweep (default 200)")
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--ell-max", type=int, default=10)
    ap.add_argument("--num-angles", type=int, default=fb_kernels.NUM_ANGLES)
    ap.add_argument("--factors", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import importlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig5 = importlib.import_module("01_figure5")

    images = fig5.load_images(args.data_dir, args.n_samples)
    torsion = fig5.load_torsion_angles(args.data_dir / "output_main.csv", len(images))
    rng = np.random.default_rng(args.seed)
    coefs, ang, sgn = fb_basis.expand(fb_kernels.add_noise(images, args.snr, rng), args.ell_max)
    coefs = fb_kernels.rotate(coefs, ang, sgn, rng.uniform(0, 2 * np.pi, size=len(coefs)))
    sub = coefs[:args.subsample]

    euclid = upper(fb_kernels.sq_distances(coefs))
    min_d2 = upper(-fb_kernels.orbit_log_kernel(sub, ang, sgn, 1.0, "min", args.num_angles))
    bisp_full = fb_kernels.bispectrum_sq_distances(fb_basis.complex_blocks(sub, ang, sgn))
    stats = {}
    for name, vals in (("euclidean", euclid), ("so2_min", min_d2), ("bispectrum", upper(bisp_full))):
        stats[name] = {f"p{p}": float(np.percentile(vals, p)) for p in PERCENTILES}
        print(f"{name:10s} squared distances: " + "  ".join(f"p{p}={stats[name][f'p{p}']:.4g}" for p in PERCENTILES))

    args.out.mkdir(parents=True, exist_ok=True)
    candidates = {"min": [np.median(euclid) * f for f in args.factors],
                  "bispectrum": [np.median(upper(bisp_full)) * f for f in args.factors]}
    results = {"min": [], "bispectrum": []}
    for kernel, bws in candidates.items():
        for bw in bws:
            logW = (fb_kernels.orbit_log_kernel(sub, ang, sgn, bw, "min", args.num_angles) if kernel == "min"
                    else -bisp_full / bw)
            phi, lam = spectral_embedding(fb_kernels.weight_matrix(logW), 2)
            g = gap(lam)
            results[kernel].append({"bandwidth": float(bw), "gap": g})
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter(phi[:, 0], phi[:, 1], c=np.sin(np.radians(torsion[:len(sub)])), cmap="rainbow", s=8, linewidths=0)
            ax.set_title(f"{kernel}, bw={bw:.4g}, gap={g:.3f}", fontsize=9)
            ax.set_axis_off()
            fig.savefig(args.out / f"sweep_{kernel}_bw{bw:.6g}.pdf", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  {kernel:10s} bw = {bw:12.6g}  gap mu_1/mu_2 = {g:.4f}")

    best = {k: max(v, key=lambda r: r["gap"]) for k, v in results.items()}
    print(f"\nbest by gap: min bw = {best['min']['bandwidth']:.6g}, bispectrum bw = {best['bispectrum']['bandwidth']:.6g}")
    (args.out / "bandwidth_results.json").write_text(json.dumps(
        {"distance_percentiles": stats, "results": results, "best_by_gap": best, "settings": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}, indent=2))
    print(f"results and sweep plots in {args.out}/")


if __name__ == "__main__":
    main()
