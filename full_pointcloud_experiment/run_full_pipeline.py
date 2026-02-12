"""
Run a grid of experiments, like the original `../full_pipeline.py`.

This is the curated, documented variant that uses the refactored pipeline:

    full_pointcloud_experiment.run_full_pipeline
        -> full_pointcloud_experiment.pipeline.run_experiment
        -> full_pointcloud_experiment.utils (math + saving)
        -> (optional) full_pointcloud_experiment.plot_from_pkl (pkl -> pdf)

Why this file exists
--------------------
Your repository already has a quick-and-dirty `full_pipeline.py` that loops over
bandwidth/method/num_points/etc.

This module keeps the same idea, but with:
- explicit configuration options
- clearer logging
- optional "dry-run" mode
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .pipeline import ExperimentConfig, run_experiment
from .plot_from_pkl import render_2d_pkl_to_pdf


InvariantMethod = Literal["integral", "min", "invariant_features", "none"]
Movement = Literal["rotation", "translation", "both", "NEW"]
LaplacianType = Literal["RWGL", "GL"]


def _str_to_bool(x: str) -> bool:
    x = x.strip().lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got: {x}")


def _parse_csv_list(x: str) -> list[str]:
    return [p.strip() for p in x.split(",") if p.strip()]


@dataclass(frozen=True)
class GridConfig:
    """
    Parameter grid for looping experiments.

    Defaults mirror the intent of the original `full_pipeline.py`, but extended
    with explicit control over:
    - laplacian type (GL vs RWGL)
    - filename noise tag (AN vs SNR)
    - pdf rendering
    """

    data_path: str
    save_folder: str
    save_name: str

    bandwidths: tuple[float, ...]
    movements: tuple[Movement, ...]
    methods: tuple[InvariantMethod, ...]
    num_points_list: tuple[int, ...]
    is_centered_list: tuple[bool, ...]
    add_stationary_list: tuple[bool, ...]
    noise_list: tuple[float, ...]
    noise_tag: str
    laplacian_types: tuple[LaplacianType, ...]

    auto_bandwidth: bool
    knn_k: int
    bandwidth_multiplier: float
    render_pdf: bool
    dry_run: bool


def _is_valid_combo(method: str, bandwidth: float, *, auto_bandwidth: bool) -> bool:
    """
    Match the original `full_pipeline.py` filtering rule.

    Original logic:
      - min / integral -> BW 47
      - none / invariant_features -> BW 3000
    """
    # If bandwidth is adaptive, we do not restrict (method, bandwidth) combos.
    if auto_bandwidth:
        return True
    if method in {"min", "integral"} and bandwidth == 47:
        return True
    if method in {"none", "invariant_features"} and bandwidth == 3000:
        return True
    return False


def run_grid(cfg: GridConfig) -> list[str]:
    """
    Run all experiments in the grid.

    Returns
    -------
    pkl_stems:
        List of saved `.pkl` stems (paths without the `.pkl` extension).
    """
    pkl_stems: list[str] = []

    for (
        bandwidth,
        movement,
        method,
        num_points,
        is_centered,
        add_stationary,
        noise,
        laplacian_type,
    ) in itertools.product(
        cfg.bandwidths,
        cfg.movements,
        cfg.methods,
        cfg.num_points_list,
        cfg.is_centered_list,
        cfg.add_stationary_list,
        cfg.noise_list,
        cfg.laplacian_types,
    ):
        if not _is_valid_combo(method, bandwidth, auto_bandwidth=cfg.auto_bandwidth):
            continue

        exp = ExperimentConfig(
            data_path=cfg.data_path,
            num_points=num_points,
            save_folder=cfg.save_folder,
            save_name=cfg.save_name,
            invariant_method=method,
            movement=movement,
            bandwidth=bandwidth,
            is_centered=is_centered,
            add_stationary=add_stationary,
            noise=noise,
            noise_tag=cfg.noise_tag,
            laplacian_type=laplacian_type,
            auto_bandwidth=cfg.auto_bandwidth,
            knn_k=cfg.knn_k,
            bandwidth_multiplier=cfg.bandwidth_multiplier,
        )

        print(
            "[grid] "
            f"BW={bandwidth} M={movement} IM={method} NOP={num_points} "
            f"IC={is_centered} AS={add_stationary} {cfg.noise_tag}={noise} LT={laplacian_type}"
        )

        if cfg.dry_run:
            continue

        pkl_stem = run_experiment(exp)
        pkl_stems.append(pkl_stem)

        if cfg.render_pdf:
            pkl_path = Path(pkl_stem + ".pkl")
            pdf_dir = Path(cfg.save_folder) / "new_plots"
            pdf_path = render_2d_pkl_to_pdf(pkl_path, pdf_dir)
            print(f"[grid] rendered: {pdf_path}")

    return pkl_stems


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a grid of full-pointcloud experiments (curated full_pipeline).")

    parser.add_argument("--data-path", required=True, help="Path to the input dataset pickle.")
    parser.add_argument("--save-folder", default="outputs", help="Where to write .pkl outputs (default: outputs).")
    parser.add_argument("--save-name", default="run", help="Base name prefix (default: run).")

    parser.add_argument("--bandwidths", default="47,3000", help="Comma-separated list (default: 47,3000).")
    parser.add_argument("--movements", default="rotation", help="Comma-separated list (default: rotation).")
    parser.add_argument("--methods", default="integral,min", help="Comma-separated list (default: integral,min).")
    parser.add_argument("--num-points-list", default="200,400", help="Comma-separated list (default: 200,400).")
    parser.add_argument("--is-centered-list", default="true", help="Comma-separated list of booleans (default: true).")
    parser.add_argument("--add-stationary-list", default="true", help="Comma-separated list of booleans (default: true).")
    parser.add_argument("--noise-list", default="0", help="Comma-separated list of floats (default: 0).")

    parser.add_argument("--noise-tag", default="SNR", help="Filename noise tag (default: SNR).")
    parser.add_argument("--laplacian-types", default="RWGL", help="Comma-separated list (default: RWGL).")

    parser.add_argument("--auto-bandwidth", type=_str_to_bool, default=False, help="If true, override BW automatically.")
    parser.add_argument("--knn-k", type=int, default=15, help="k for adaptive bandwidth ε = 0.5 * mean(d_k(x_i)) (default: 15).")
    parser.add_argument("--bandwidth-multiplier", type=float, default=1.0, help="Multiply bandwidth by this factor (default: 1.0).")
    parser.add_argument("--render-pdf", type=_str_to_bool, default=False, help="If true, render PDFs for each run.")
    parser.add_argument("--dry-run", type=_str_to_bool, default=False, help="If true, print runs without computing.")

    args = parser.parse_args()

    cfg = GridConfig(
        data_path=args.data_path,
        save_folder=args.save_folder,
        save_name=args.save_name,
        bandwidths=tuple(float(x) for x in _parse_csv_list(args.bandwidths)),
        movements=tuple(_parse_csv_list(args.movements)),  # type: ignore[arg-type]
        methods=tuple(_parse_csv_list(args.methods)),  # type: ignore[arg-type]
        num_points_list=tuple(int(x) for x in _parse_csv_list(args.num_points_list)),
        is_centered_list=tuple(_str_to_bool(x) for x in _parse_csv_list(args.is_centered_list)),
        add_stationary_list=tuple(_str_to_bool(x) for x in _parse_csv_list(args.add_stationary_list)),
        noise_list=tuple(float(x) for x in _parse_csv_list(args.noise_list)),
        noise_tag=args.noise_tag,
        laplacian_types=tuple(_parse_csv_list(args.laplacian_types)),  # type: ignore[arg-type]
        auto_bandwidth=args.auto_bandwidth,
        knn_k=args.knn_k,
        bandwidth_multiplier=args.bandwidth_multiplier,
        render_pdf=args.render_pdf,
        dry_run=args.dry_run,
    )

    pkls = run_grid(cfg)
    if cfg.dry_run:
        print("[ok] dry-run complete")
    else:
        print(f"[ok] completed runs: {len(pkls)}")


if __name__ == "__main__":
    main()

