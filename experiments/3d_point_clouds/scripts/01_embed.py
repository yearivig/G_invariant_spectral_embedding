#!/usr/bin/env python3
"""
One embedding of the Section 5.1 point clouds with one kernel (Algorithm 1).

    python3 scripts/01_embed.py --kernel integral --bandwidth 47 --num-points 400
    python3 scripts/01_embed.py --kernel min --bandwidth 47 --num-points 200 --min-method kabsch
    python3 scripts/01_embed.py --kernel invariant_features --bandwidth 3000 --num-points 800 \\
        --is-centered --add-stationary --snr 10

Writes <save-folder>/<name>...pkl and .json (see tools/pipeline.py) and, unless
--no-pdf, the scatter plot in <save-folder>/new_plots/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import kernels  # noqa: E402
from pipeline import ExperimentConfig, run_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default="data/data_3D.pkl", help="the trajectory pickle (default: data/data_3D.pkl)")
    ap.add_argument("--save-folder", default="results/runs")
    ap.add_argument("--save-name", default="run")
    ap.add_argument("--num-points", type=int, required=True, help="number of point clouds n (NOP)")
    ap.add_argument("--kernel", required=True, choices=kernels.KERNELS, help="none = the Euclidean kernel (IM)")
    ap.add_argument("--bandwidth", type=float, required=True, help="epsilon (BW); the paper uses 47 for min and "
                    "integral, 3000 for invariant_features and none")
    ap.add_argument("--is-centered", action="store_true", help="centre each cloud (IC)")
    ap.add_argument("--add-stationary", action="store_true", help="append the 80 stationary points (AS)")
    ap.add_argument("--snr", type=float, default=0.0, help="noise SNR in dB; 0 (default) = clean")
    ap.add_argument("--noise-tag", default="SNR", help="filename label for the noise level only (the paper's clean "
                    "files use AN)")
    ap.add_argument("--min-method", choices=["grid", "kabsch"], default="grid",
                    help="minimum kernel: super-Fibonacci grid (default) or exact Kabsch")
    ap.add_argument("--num-rotations", type=int, default=kernels.NUM_ROTATIONS,
                    help="size of the SO(3) grid for min (grid) and integral (default 600)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the stationary points, noise and rotations")
    ap.add_argument("--points", choices=["rotation", "all"], default="rotation",
                    help="points of each frame: rotation = 800 onward, 303 points (default, as in Figure 4); all = 1103")
    ap.add_argument("--m", type=int, default=2, help="embedding dimension (default 2)")
    ap.add_argument("--no-pdf", action="store_true", help="do not render the scatter plot")
    args = ap.parse_args()

    cfg = ExperimentConfig(data_path=args.data_path, num_points=args.num_points, kernel=args.kernel,
                           bandwidth=args.bandwidth, is_centered=args.is_centered,
                           add_stationary=args.add_stationary, snr_db=args.snr, noise_tag=args.noise_tag,
                           min_method=args.min_method, num_rotations=args.num_rotations, seed=args.seed, points=args.points,
                           m=args.m, save_folder=args.save_folder, save_name=args.save_name)
    pkl = run_experiment(cfg)
    print(f"saved {pkl}")
    if not args.no_pdf:
        from plot_embedding import render_2d_pkl_to_pdf
        print(f"saved {render_2d_pkl_to_pdf(pkl, Path(args.save_folder) / 'new_plots')}")


if __name__ == "__main__":
    main()
