"""
Run a grid of image-based spectral embedding experiments.

Usage (from Documents/G_invariant_kernel/):

    python -m image_based_spectral_embedding.run_full_pipeline \\
      --data-path projection_data/unrotated/unrotated_projections-synth.pkl \\
      --save-folder image_based_spectral_embedding/outputs \\
      --save-name eigvectors \\
      --num-images-list 198 \\
      --methods min,mean,bispectrum \\
      --snr-list 0,1,5,10 \\
      --bandwidth 0.5 \\
      --ell-max 10 \\
      --render-pdf true
"""

from __future__ import annotations

import argparse
import itertools
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


def _parse_csv(x: str) -> list[str]:
    return [s.strip() for s in x.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a grid of image-based spectral embedding experiments."
    )

    parser.add_argument("--data-path", required=True, help="Path to projections pickle.")
    parser.add_argument("--save-folder", default="image_based_spectral_embedding/outputs",
                        help="Directory for output files.")
    parser.add_argument("--save-name", default="eigvectors", help="Base name prefix.")

    parser.add_argument("--num-images-list", default="198",
                        help="Comma-separated list of num_images values (default: 198).")
    parser.add_argument("--methods", default="min,mean,bispectrum",
                        help="Comma-separated list of methods (default: min,mean,bispectrum).")
    parser.add_argument("--snr-list", default="0,1,5,10",
                        help="Comma-separated list of SNR values in dB (default: 0,1,5,10).")
    parser.add_argument("--bandwidth", type=float, default=0.5,
                        help="Kernel bandwidth ε (default: 0.5).")

    parser.add_argument("--ell-max", type=int, default=10,
                        help="Max angular frequency for FFB basis (default: 10).")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="Tolerance for FFB expansion (default: 0.01).")
    parser.add_argument("--image-size", type=int, default=82,
                        help="Square image size (default: 82).")
    parser.add_argument("--num-rotations", type=int, default=600,
                        help="Number of SO(2) samples (default: 600).")

    parser.add_argument("--render-pdf", type=_str_to_bool, default=True,
                        help="If true, render PDF scatter plots (default: true).")
    parser.add_argument("--dry-run", type=_str_to_bool, default=False,
                        help="If true, print runs without computing (default: false).")

    args = parser.parse_args()

    num_images_list = [int(x) for x in _parse_csv(args.num_images_list)]
    methods = _parse_csv(args.methods)
    snr_list = [float(x) for x in _parse_csv(args.snr_list)]

    total = len(num_images_list) * len(methods) * len(snr_list)
    print(f"Grid: {len(num_images_list)} num_images × {len(methods)} methods × {len(snr_list)} SNRs = {total} runs")

    completed = 0
    for num_images, method, snr in itertools.product(num_images_list, methods, snr_list):
        print(f"\n[grid {completed+1}/{total}] N={num_images} method={method} SNR={snr}")

        if args.dry_run:
            completed += 1
            continue

        cfg = ExperimentConfig(
            data_path=args.data_path,
            num_images=num_images,
            save_folder=args.save_folder,
            save_name=args.save_name,
            method=method,
            bandwidth=args.bandwidth,
            snr=snr,
            ell_max=args.ell_max,
            tol=args.tol,
            image_size=args.image_size,
            num_rotations=args.num_rotations,
        )

        pkl_path = run_experiment(cfg)

        if args.render_pdf:
            pdf_dir = Path(args.save_folder) / "plots"
            pdf_path = render_eigenvectors_to_pdf(pkl_path, pdf_dir)
            print(f"[grid] Rendered: {pdf_path}")

        completed += 1

    if args.dry_run:
        print(f"\n[ok] Dry-run complete ({completed} runs)")
    else:
        print(f"\n[ok] Completed {completed} runs")


if __name__ == "__main__":
    main()
