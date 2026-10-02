#!/usr/bin/env python3
"""
Re-tune the bandwidth: a median-heuristic sweep for one kernel.

NOTE: this tool is for re-tuning bandwidths (e.g. for another dataset or a
revision). It was not used to produce the paper's figures: Figure 4 uses the
fixed values 47 (min, integral) and 3000 (invariant_features, Euclidean).

It builds the point clouds exactly as tools/pipeline.py does, sets

    eps_0 = median_{i<j} ||X_i - X_j||_F^2    (squared distances of the flattened clouds)

and runs the chosen kernel at eps_0 * f for each factor f, writing one
embedding per factor and a summary CSV. eps_0 is computed from the Euclidean
distances for every kernel, as in the original script; the invariant kernels'
own distances are on other scales (smaller for min and integral, much larger
for the Gram features), so widen --factors for those.

    python3 tools/retune_bandwidth.py --kernel integral --num-points 200 --factors 0.5,1,2
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kernels  # noqa: E402
import pointclouds  # noqa: E402
from pipeline import ExperimentConfig, run_experiment  # noqa: E402


def median_sq_distance(X: np.ndarray) -> float:
    """median_{i<j} ||X_i - X_j||_F^2 of the flattened clouds."""
    return float(np.median(pdist(X.reshape(len(X), -1), metric="sqeuclidean")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default="data/data_3D.pkl")
    ap.add_argument("--save-folder", default="results/bandwidth_sweep")
    ap.add_argument("--save-name", default="run_medianBW")
    ap.add_argument("--num-points", type=int, default=200)
    ap.add_argument("--kernel", choices=kernels.KERNELS, default="integral")
    ap.add_argument("--is-centered", action="store_true")
    ap.add_argument("--add-stationary", action="store_true")
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--min-method", choices=["grid", "kabsch"], default="grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--factors", default="0.5,1.0,2.0", help="comma-separated multipliers of eps_0")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    base = ExperimentConfig(data_path=args.data_path, num_points=args.num_points, kernel=args.kernel, bandwidth=1.0,
                            is_centered=args.is_centered, add_stationary=args.add_stationary, snr_db=args.snr,
                            min_method=args.min_method, seed=args.seed, save_folder=args.save_folder,
                            save_name=args.save_name)
    trajectory = pointclouds.load_trajectory(args.data_path)
    X = pointclouds.make_point_clouds(trajectory, args.num_points, centered=args.is_centered,
                                      add_stationary=args.add_stationary, snr_db=args.snr,
                                      rng=np.random.default_rng(args.seed))     # the clouds each run will use
    eps0 = median_sq_distance(X)
    print(f"eps_0 = median ||X_i - X_j||^2 = {eps0:.6g}")

    rows = []
    for f in (float(x) for x in args.factors.split(",") if x.strip()):
        pkl = run_experiment(replace(base, bandwidth=eps0 * f), trajectory)
        pdf = ""
        if not args.no_pdf:
            from plot_embedding import render_2d_pkl_to_pdf
            pdf = str(render_2d_pkl_to_pdf(pkl, Path(args.save_folder) / "plots"))
        print(f"factor {f:g}: bandwidth {eps0 * f:.6g} -> {pkl.name}")
        rows.append((f, eps0 * f, str(pkl), pdf))

    summary = Path(args.save_folder) / "median_bandwidth_sweep_summary.csv"
    with summary.open("w", newline="") as fh:
        csv.writer(fh).writerows([("factor", "bandwidth", "pkl_path", "pdf_path"), *rows])
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
