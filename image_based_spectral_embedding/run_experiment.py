"""
CLI entrypoint for a single image-based spectral embedding experiment.

Usage (from Documents/G_invariant_kernel/):

    python -m image_based_spectral_embedding.run_experiment \\
      --data-path projection_data/unrotated/unrotated_projections-synth.pkl \\
      --save-folder image_based_spectral_embedding/outputs \\
      --save-name eigvectors \\
      --num-images 198 \\
      --method min \\
      --bandwidth 0.5 \\
      --snr 10 \\
      --ell-max 10 \\
      --tol 0.01 \\
      --render-pdf true
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import ExperimentConfig, run_experiment
from .plot_from_pkl import render_eigenvectors_to_pdf


def _str_to_bool(x: str) -> bool:
    x = x.strip().lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got: {x}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single image-based spectral embedding experiment."
    )

    parser.add_argument("--data-path", required=True, help="Path to projections pickle.")
    parser.add_argument("--save-folder", default="image_based_spectral_embedding/outputs",
                        help="Directory for output files.")
    parser.add_argument("--save-name", default="eigvectors", help="Base name prefix.")

    parser.add_argument("--num-images", type=int, required=True, help="Number of images to use (N).")
    parser.add_argument("--method", required=True, choices=["min", "mean", "bispectrum", "none"],
                        help="Kernel method.")
    parser.add_argument("--bandwidth", type=float, default=0.5, help="Kernel bandwidth ε (default: 0.5).")
    parser.add_argument("--snr", type=float, default=10.0, help="Signal-to-noise ratio in dB (default: 10).")

    parser.add_argument("--ell-max", type=int, default=10, help="Max angular frequency for FFB basis (default: 10).")
    parser.add_argument("--tol", type=float, default=0.01, help="Tolerance for FFB expansion (default: 0.01).")
    parser.add_argument("--image-size", type=int, default=82, help="Square image size (default: 82).")
    parser.add_argument("--num-rotations", type=int, default=600, help="Number of SO(2) samples (default: 600).")

    parser.add_argument("--render-pdf", type=_str_to_bool, default=True,
                        help="If true, render a PDF scatter plot of the embedding.")

    args = parser.parse_args()

    cfg = ExperimentConfig(
        data_path=args.data_path,
        num_images=args.num_images,
        save_folder=args.save_folder,
        save_name=args.save_name,
        method=args.method,
        bandwidth=args.bandwidth,
        snr=args.snr,
        ell_max=args.ell_max,
        tol=args.tol,
        image_size=args.image_size,
        num_rotations=args.num_rotations,
    )

    pkl_path = run_experiment(cfg)
    print(f"[ok] Saved eigenvectors: {pkl_path}")

    if args.render_pdf:
        pdf_dir = Path(args.save_folder) / "plots"
        pdf_path = render_eigenvectors_to_pdf(pkl_path, pdf_dir)
        print(f"[ok] Rendered PDF: {pdf_path}")


if __name__ == "__main__":
    main()
