#!/usr/bin/env python3
"""
Run the SO(3)/SO(2) single-point convergence experiment of Section 3.5 (paired
sampling; see tools/converges_so3_so2.py). Writes the trial-level errors to
<out>/so3_so2_P_1_00_results.pkl and the per-kernel and combined plots to
<out>/plots/. Long: 1000 trials x 3 kernels x 100 epsilons with n = 10000
points and 200 SO(2) samples.

    python3 scripts/01_run_experiment.py                  # seed 0, into results/
    python3 scripts/01_run_experiment.py --seed 1 --out results/run_seed1

Figure 2 itself is drawn from the pickle by scripts/02_figure2.py. The pickle
committed in results/ is the one behind the paper's Figure 2; it came from an
unseeded run, so rerunning gives statistically equivalent but not identical
numbers. Use --out to keep it from being overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import converges_so3_so2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "new_run",
                    help="output folder (default: results/new_run, so the committed pickle is kept)")
    ap.add_argument("--seed", type=int, default=0, help="numpy random seed (default 0)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    converges_so3_so2.main(output_dir=str(args.out), seed=args.seed)


if __name__ == "__main__":
    main()
