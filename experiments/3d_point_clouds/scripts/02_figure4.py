#!/usr/bin/env python3
"""
All 24 panels of Figure 4 (Section 5.1): kernels x {clean, noisy} x n in {200, 400, 800}.

PANELS lists each panel's settings as recorded in the filenames of the figure
files the paper includes (figures/exp1/table-exp1/ in the paper source):
bandwidth 47 for min and integral, 3000 for invariant_features and none
(Euclidean); clean panels without noise, noisy panels at SNR 10 dB. Centring
(IC) and the 80 stationary points (AS) differ between panels in the paper;
--is-centered and --add-stationary override them for every panel.

    python3 scripts/02_figure4.py                              # the paper's settings
    python3 scripts/02_figure4.py --min-method kabsch           # exact minimum kernel
    python3 scripts/02_figure4.py --is-centered true --add-stationary false
    python3 scripts/02_figure4.py --kernels min integral --dry-run

Outputs go to --save-folder (default results/figure4): one .pkl and .json per
panel, and the scatter plots in new_plots/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import kernels  # noqa: E402
import pointclouds  # noqa: E402
from pipeline import ExperimentConfig, output_stem, run_experiment  # noqa: E402

BANDWIDTH = {"min": 47.0, "integral": 47.0, "invariant_features": 3000.0, "none": 3000.0}

# (n, kernel, IC, AS, SNR dB, filename noise tag), in the order of Figure 4
PANELS = [
    (200, "min", False, False, 0, "AN"), (200, "integral", False, False, 0, "AN"),
    (200, "invariant_features", False, False, 0, "AN"), (200, "none", True, True, 0, "AN"),
    (400, "min", False, False, 0, "AN"), (400, "integral", True, True, 0, "AN"),
    (400, "invariant_features", False, False, 0, "AN"), (400, "none", False, False, 0, "AN"),
    (800, "min", False, False, 0, "AN"), (800, "integral", False, False, 0, "AN"),
    (800, "invariant_features", False, False, 0, "AN"), (800, "none", False, False, 0, "AN"),
] + [(n, k, True, True, 10, "SNR") for n in (200, 400, 800) for k in ("min", "integral", "invariant_features", "none")]


def _override(value: str):
    return {"paper": None, "true": True, "false": False}[value]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default="data/data_3D.pkl")
    ap.add_argument("--save-folder", default="results/figure4")
    ap.add_argument("--save-name", default="run")
    ap.add_argument("--kernels", nargs="+", choices=kernels.KERNELS, default=list(kernels.KERNELS),
                    help="run only these kernels' panels")
    ap.add_argument("--is-centered", choices=["paper", "true", "false"], default="paper",
                    help="IC for every panel (default: each panel as in the paper)")
    ap.add_argument("--add-stationary", choices=["paper", "true", "false"], default="paper",
                    help="AS for every panel (default: each panel as in the paper)")
    ap.add_argument("--snr", type=float, default=10.0, help="SNR in dB of the noisy panels (default 10)")
    ap.add_argument("--min-method", choices=["grid", "kabsch"], default="grid")
    ap.add_argument("--num-rotations", type=int, default=kernels.NUM_ROTATIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--points", choices=["rotation", "all"], default="rotation",
                    help="points of each frame: rotation = 800 onward, 303 points (default, as in Figure 4); all = 1103")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list the panels and stop")
    args = ap.parse_args()

    ic_all, as_all = _override(args.is_centered), _override(args.add_stationary)
    configs = []
    for n, kernel, ic, add, snr, tag in PANELS:
        if kernel not in args.kernels:
            continue
        configs.append(ExperimentConfig(
            data_path=args.data_path, num_points=n, kernel=kernel, bandwidth=BANDWIDTH[kernel],
            is_centered=ic if ic_all is None else ic_all, add_stationary=add if as_all is None else as_all,
            snr_db=args.snr if snr else 0.0, noise_tag=tag, min_method=args.min_method,
            num_rotations=args.num_rotations, seed=args.seed, points=args.points, save_folder=args.save_folder,
            save_name=args.save_name))

    for cfg in configs:
        print(output_stem(cfg).name)
    if args.dry_run:
        print(f"\n{len(configs)} panels; dry run, stopping here")
        return

    trajectory = pointclouds.load_trajectory(args.data_path)
    for k, cfg in enumerate(configs, 1):
        pkl = run_experiment(cfg, trajectory)
        print(f"[{k}/{len(configs)}] saved {pkl.name}")
        if not args.no_pdf:
            from plot_embedding import render_2d_pkl_to_pdf
            render_2d_pkl_to_pdf(pkl, Path(args.save_folder) / "new_plots")


if __name__ == "__main__":
    main()
