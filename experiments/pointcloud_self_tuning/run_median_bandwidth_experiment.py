"""
Run full_pointcloud_experiment using median-distance bandwidth selection.

This script mirrors the median-heuristic method used in the image experiment:
  bw0 = median_{i<j} ||x_i - x_j||^2
and then sweeps multiplicative factors around bw0.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .pipeline import ExperimentConfig, run_experiment
    from .plot_from_pkl import render_2d_pkl_to_pdf
    from . import utils
except ImportError:
    from pipeline import ExperimentConfig, run_experiment
    from plot_from_pkl import render_2d_pkl_to_pdf
    import utils


def main() -> None:
    parser = argparse.ArgumentParser(description="Run median-bandwidth sweep for pointcloud experiment.")
    parser.add_argument("--data-path", required=True, help="Path to pointcloud dataset pickle.")
    parser.add_argument("--save-folder", default="outputs", help="Output folder (default: outputs).")
    parser.add_argument("--save-name", default="run_medianBW", help="Base run name (default: run_medianBW).")
    parser.add_argument("--num-points", type=int, default=200, help="Number of samples (default: 200).")
    parser.add_argument("--movement", default="rotation", choices=["rotation", "translation", "both", "NEW"])
    parser.add_argument("--method", default="integral", choices=["integral", "min", "invariant_features", "none"])
    parser.add_argument("--laplacian-type", default="GL", choices=["RWGL", "GL"])
    parser.add_argument("--is-centered", action="store_true", help="Center each cloud.")
    parser.add_argument("--add-stationary", action="store_true", help="Add stationary points.")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise value used in dataset generation.")
    parser.add_argument("--noise-tag", default="SNR", help="Filename noise tag.")
    parser.add_argument(
        "--factors",
        default="0.5,1.0,2.0",
        help="Comma-separated multipliers around bw0 (default: 0.5,1.0,2.0).",
    )
    parser.add_argument("--render-pdf", action="store_true", help="Render PDF for each run.")
    args = parser.parse_args()

    factors = [float(x.strip()) for x in args.factors.split(",") if x.strip()]

    # Load data exactly as pipeline does, then compute bw0.
    data = utils.generate_point_cloud_from_KTH(
        args.data_path,
        args.movement,
        args.num_points,
        so3_rotated=True,
        centered=args.is_centered,
        add_stationary=args.add_stationary,
        snr_db=args.noise,
    )
    bw0 = utils.compute_bandwidth_median_sq(data[: args.num_points])
    print(f"[median_bw] bw0 = median(||x_i-x_j||^2) = {bw0}")

    summary_rows: list[tuple[float, float, str, str]] = []

    for f in factors:
        bw = bw0 * f
        print(f"\n[run] factor={f} -> bandwidth={bw}")
        cfg = ExperimentConfig(
            data_path=args.data_path,
            num_points=args.num_points,
            save_folder=args.save_folder,
            save_name=args.save_name,
            invariant_method=args.method,
            movement=args.movement,
            bandwidth=bw,
            is_centered=args.is_centered,
            add_stationary=args.add_stationary,
            noise=args.noise,
            noise_tag=args.noise_tag,
            laplacian_type=args.laplacian_type,
            auto_bandwidth=False,
            knn_k=15,
            bandwidth_multiplier=1.0,
        )
        pkl_stem = run_experiment(cfg)
        # Do not use Path.with_suffix here; stems contain decimal points from BW values.
        pkl_path = pkl_stem + ".pkl"
        pdf_path = ""
        if args.render_pdf:
            pdf = render_2d_pkl_to_pdf(Path(pkl_path), Path(args.save_folder) / "new_plots")
            pdf_path = str(pdf)
            print(f"[ok] pdf: {pdf_path}")
        summary_rows.append((f, bw, pkl_path, pdf_path))

    # Save summary CSV to make comparison easy.
    summary_path = Path(args.save_folder) / "median_bandwidth_sweep_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["factor", "bandwidth", "pkl_path", "pdf_path"])
        w.writerows(summary_rows)
    print(f"\n[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
