#!/usr/bin/env python3
"""
All 24 panels of Figure 4 (Section 5.1): kernels x {clean, noisy} x n in {200, 400, 800}.

Every panel: bandwidth 47 for min and integral, 3000 for invariant_features
and none (Euclidean); clean panels without noise, noisy panels at SNR 10 dB;
each cloud centred (IC) with the 80 stationary points appended (AS); the
minimum kernel computed exactly (Kabsch). The figure files keep the paper's
names (figures/exp1/table-exp1/ in the paper source), e.g.
run_NOP:400_IM:min_M:rotation_BW:47_IC:True_AS:True_AN:0_LT:RWGL.pdf.

The previous version of Figure 4 mixed IC and AS between clean panels;
--is-centered old-paper --add-stationary old-paper --min-method grid reproduces
it (OLD_PAPER_IC_AS).

    python3 scripts/02_figure4.py                              # Figure 4
    python3 scripts/02_figure4.py --min-method grid             # minimum over the 600-rotation grid
    python3 scripts/02_figure4.py --is-centered true --add-stationary false
    python3 scripts/02_figure4.py --kernels min integral --dry-run

Outputs go to --save-folder (default results/figure4): one .pkl and .json per
panel in embeddings/, and the scatter plots in plots/.
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

# (n, kernel, noisy, filename noise tag), in the order of Figure 4
PANELS = [(n, k, noisy, "SNR" if noisy else "AN")
          for noisy in (False, True) for n in (200, 400, 800)
          for k in ("min", "integral", "invariant_features", "none")]

# (IC, AS) of the clean panels in the previous version of Figure 4; its noisy panels were all (True, True)
OLD_PAPER_IC_AS = {(200, "none"): (True, True), (400, "integral"): (True, True)}


def _setting(value: str, n: int, kernel: str, noisy: bool, which: int) -> bool:
    if value == "old-paper":
        return True if noisy else OLD_PAPER_IC_AS.get((n, kernel), (False, False))[which]
    return value == "true"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default="data/data_3D.pkl")
    ap.add_argument("--save-folder", default="results/figure4")
    ap.add_argument("--save-name", default="run")
    ap.add_argument("--kernels", nargs="+", choices=kernels.KERNELS, default=list(kernels.KERNELS),
                    help="run only these kernels' panels")
    ap.add_argument("--is-centered", choices=["true", "false", "old-paper"], default="true",
                    help="IC for every panel (default true; old-paper: as in the previous Figure 4)")
    ap.add_argument("--add-stationary", choices=["true", "false", "old-paper"], default="true",
                    help="AS for every panel (default true; old-paper: as in the previous Figure 4)")
    ap.add_argument("--snr", type=float, default=10.0, help="SNR in dB of the noisy panels (default 10)")
    ap.add_argument("--min-method", choices=["kabsch", "grid"], default="kabsch",
                    help="minimum kernel: exact Kabsch (default) or the super-Fibonacci grid")
    ap.add_argument("--num-rotations", type=int, default=kernels.NUM_ROTATIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--points", choices=["rotation", "all"], default="rotation",
                    help="points of each frame: rotation = 800 onward, 303 points (default, as in Figure 4); all = 1103")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list the panels and stop")
    args = ap.parse_args()

    configs = []
    for n, kernel, noisy, tag in PANELS:
        if kernel not in args.kernels:
            continue
        configs.append(ExperimentConfig(
            data_path=args.data_path, num_points=n, kernel=kernel, bandwidth=BANDWIDTH[kernel],
            is_centered=_setting(args.is_centered, n, kernel, noisy, 0),
            add_stationary=_setting(args.add_stationary, n, kernel, noisy, 1),
            snr_db=args.snr if noisy else 0.0, noise_tag=tag, min_method=args.min_method,
            num_rotations=args.num_rotations, seed=args.seed, points=args.points, save_folder=args.save_folder,
            save_name=args.save_name))

    for cfg in configs:
        print(output_stem(cfg).name)
    if args.dry_run:
        print(f"\n{len(configs)} panels; dry run, stopping here")
        return

    trajectory = pointclouds.load_trajectory(args.data_path)
    pkls = []
    for k, cfg in enumerate(configs, 1):
        pkls.append(run_experiment(cfg, trajectory))
        print(f"[{k}/{len(configs)}] saved {pkls[-1].name}")
    if not args.no_pdf:                       # after all runs, so each panel can face the integral panel's way
        from plot_embedding import render_2d_pkl_to_pdf
        for pkl in pkls:
            render_2d_pkl_to_pdf(pkl, Path(args.save_folder) / "plots")


if __name__ == "__main__":
    main()
