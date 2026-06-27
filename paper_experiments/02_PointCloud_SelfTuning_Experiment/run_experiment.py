"""
CLI entrypoint for the curated full pointcloud experiment.

This script is meant to be the *one command* you run to reproduce a specific
configuration (e.g. the one encoded in a figure filename).

Examples
--------
From the repository root:

    python -m full_pointcloud_experiment.run_experiment \\
      --data-path data/data_3D.pkl \\
      --save-folder outputs \\
      --save-name run \\
      --num-points 200 \\
      --invariant-method integral \\
      --movement rotation \\
      --bandwidth 47 \\
      --is-centered false \\
      --add-stationary false \\
      --noise 0.1 \\
      --noise-tag AN \\
      --laplacian-type GL \\
      --render-pdf true
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from .pipeline import ExperimentConfig, run_experiment
    from .plot_from_pkl import render_2d_pkl_to_pdf
except ImportError:
    from pipeline import ExperimentConfig, run_experiment
    from plot_from_pkl import render_2d_pkl_to_pdf


def _str_to_bool(x: str) -> bool:
    x = x.strip().lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got: {x}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single full-pointcloud experiment configuration.")

    parser.add_argument("--data-path", required=False, help="Path to the input dataset pickle (required unless --synthetic true).")
    parser.add_argument("--save-folder", default="outputs", help="Folder to save the .pkl output (default: outputs).")
    parser.add_argument("--save-name", default="run", help="Base name prefix for saved files (default: run).")

    parser.add_argument("--num-points", type=int, required=True, help="Number of point-cloud samples (NOP).")
    parser.add_argument(
        "--invariant-method",
        required=True,
        choices=[
            "integral",
            "min",
            "invariant_features",
            "none",
            "none_self_tuning",
            "procrustes_self_tuning",
            "invariant_features_self_tuning",
        ],
        help="Kernel invariance method (IM).",
    )
    parser.add_argument(
        "--movement",
        required=True,
        choices=["rotation", "translation", "both", "NEW"],
        help="Movement type (M).",
    )
    parser.add_argument("--bandwidth", type=float, required=True, help="Kernel bandwidth (BW).")
    parser.add_argument("--auto-bandwidth", type=_str_to_bool, default=False, help="If true, override BW automatically.")
    parser.add_argument("--knn-k", type=int, default=15, help="k for adaptive bandwidth ε = 0.5 * mean(d_k(x_i)) (default: 15).")

    parser.add_argument("--is-centered", type=_str_to_bool, default=False, help="Whether to center each point cloud (IC).")
    parser.add_argument("--add-stationary", type=_str_to_bool, default=False, help="Whether to add stationary points (AS).")

    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Noise parameter (meaning depends on data generation). Stored in filename as <noise-tag>:<noise>.",
    )
    parser.add_argument("--noise-tag", default="SNR", help="Filename noise tag (e.g. SNR or AN).")

    parser.add_argument("--laplacian-type", required=True, choices=["RWGL", "GL"], help="Which Laplacian embedding to export (LT).")

    parser.add_argument("--render-pdf", type=_str_to_bool, default=True, help="If true, render a PDF into outputs/new_plots.")

    # Demo/testing mode: generate synthetic point clouds instead of reading a dataset file.
    parser.add_argument("--synthetic", type=_str_to_bool, default=False, help="If true, generate synthetic point clouds (ignores --data-path).")
    parser.add_argument("--synthetic-num-atoms", type=int, default=200, help="Number of atoms per synthetic cloud (default: 200).")
    parser.add_argument("--synthetic-seed", type=int, default=0, help="RNG seed for synthetic data (default: 0).")
    parser.add_argument("--synthetic-noise-std", type=float, default=0.0, help="Add i.i.d. Gaussian noise with this std (default: 0).")

    args = parser.parse_args()

    if not args.synthetic and not args.data_path:
        raise SystemExit("--data-path is required unless --synthetic true")

    cfg = ExperimentConfig(
        data_path=args.data_path or "",
        num_points=args.num_points,
        save_folder=args.save_folder,
        save_name=args.save_name,
        invariant_method=args.invariant_method,
        movement=args.movement,
        bandwidth=args.bandwidth,
        is_centered=args.is_centered,
        add_stationary=args.add_stationary,
        noise=args.noise,
        noise_tag=args.noise_tag,
        laplacian_type=args.laplacian_type,
        auto_bandwidth=args.auto_bandwidth,
        knn_k=args.knn_k,
        synthetic=args.synthetic,
        synthetic_num_atoms=args.synthetic_num_atoms,
        synthetic_seed=args.synthetic_seed,
        synthetic_noise_std=args.synthetic_noise_std,
    )

    pkl_stem = run_experiment(cfg)
    pkl_path = Path(pkl_stem + ".pkl")
    print(f"[ok] Saved pkl: {pkl_path}")

    if args.render_pdf:
        outputs_dir = Path(args.save_folder)
        pdf_dir = outputs_dir / "new_plots"
        pdf_path = render_2d_pkl_to_pdf(pkl_path, pdf_dir)
        print(f"[ok] Rendered pdf: {pdf_path}")


if __name__ == "__main__":
    main()

