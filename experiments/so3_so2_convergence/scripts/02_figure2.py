#!/usr/bin/env python3
"""
Figure 2 (Section 3.5) from a saved results pickle: mean |error| of
(4/eps) L_RW f(I_3) against 1, versus eps, for the three kernels, with 95%
confidence bands, fitted-slope (dashed) and predicted-slope (dotted) reference
lines. Seconds; no recomputation. See tools/replot_from_pickle_fitted_slopes.py.

    python3 scripts/02_figure2.py                                     # the committed results
    python3 scripts/02_figure2.py results/new_run/so3_so2_P_1_00_results.pkl

Writes results/new_run/plots_fitted_slopes/so3_so2_P_1_00_combined.pdf (the
figure in the paper) and one plot per kernel, or into -o. The committed copy of
the paper's figure, results/plots_fitted_slopes/, is never overwritten by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from replot_from_pickle_fitted_slopes import regenerate_fitted_slope_plots  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DEFAULT_PICKLE = RESULTS / "so3_so2_P_1_00_results.pkl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pickle_file", nargs="?", type=Path, default=DEFAULT_PICKLE)
    ap.add_argument("-o", "--output-dir", default=str(RESULTS / "new_run"),
                    help="folder for plots_fitted_slopes/ (default: results/new_run)")
    args = ap.parse_args()
    if not args.pickle_file.is_file():
        sys.exit(f"not found: {args.pickle_file}")
    regenerate_fitted_slope_plots(str(args.pickle_file), args.output_dir)


if __name__ == "__main__":
    main()
