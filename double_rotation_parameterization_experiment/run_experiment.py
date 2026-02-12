"""
CLI entrypoint for the double-rotation parameterization experiment.

Usage (from Documents/G_invariant_kernel/):

    python -m double_rotation_parameterization_experiment.run_experiment \\
      --input-path roy_lederman_data/data \\
      --n-images 5000 \\
      --output-dir double_rotation_parameterization_experiment/outputs \\
      --t 10 \\
      --num-neighbors 20 \\
      --num-rotations 300
"""

from __future__ import annotations

import argparse
import logging

from .pipeline import ExperimentConfig, run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the double-rotation parameterization experiment.")

    parser.add_argument("--input-path", required=True, help="Directory containing source images.")
    parser.add_argument("--n-images", type=int, default=5000, help="Number of images to process (default: 5000).")
    parser.add_argument("--output-dir", default="double_rotation_parameterization_experiment/outputs",
                        help="Output directory (default: double_rotation_parameterization_experiment/outputs).")
    parser.add_argument("--file-prefix", default="s1_", help="Only load files starting with this prefix (default: s1_).")
    parser.add_argument("--t", type=int, default=10, help="Diffusion time parameter (default: 10).")
    parser.add_argument("--num-neighbors", type=int, default=20, help="Number of neighbors for bandwidth (default: 20).")
    parser.add_argument("--num-rotations", type=int, default=300, help="Number of SO(2) samples (default: 300).")
    parser.add_argument("--n-eigenvectors", type=int, default=99, help="Number of diffusion coordinates (default: 99).")
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cuda:0 (default: auto-detect).")

    args = parser.parse_args()

    cfg = ExperimentConfig(
        input_path=args.input_path,
        n_images=args.n_images,
        output_dir=args.output_dir,
        file_prefix=args.file_prefix,
        t=args.t,
        num_neighbors=args.num_neighbors,
        num_rotations=args.num_rotations,
        n_eigenvectors=args.n_eigenvectors,
        device=args.device,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
