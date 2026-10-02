"""CLI to regenerate plots from a saved paired-experiment pickle.

Thin wrapper around ``converges_so3_so2.regenerate_all_plots``. The output is
byte-identical to what ``converges_so3_so2.py``'s ``main`` produces (same
SEM-based 95% CI band, same filenames).

Usage
-----
    python replot_from_pickle.py [pickle_file] [-o OUTPUT_DIR]

If ``pickle_file`` is omitted, defaults to ``so3_so2_P_1_00_results.pkl`` in
the current directory.
"""

from __future__ import annotations

import argparse
import os
import sys

from converges_so3_so2 import regenerate_all_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate publication-quality plots from a saved results pickle. "
            "Produces the same plots as running converges_so3_so2.py end-to-end."
        )
    )
    parser.add_argument(
        "pickle_file",
        nargs="?",
        default="so3_so2_P_1_00_results.pkl",
        help="Path to the pickle file containing experimental results.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for plots (default: directory of the pickle file).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pickle_file):
        print(f"Error: file not found: {args.pickle_file}", file=sys.stderr)
        pkl_files = [f for f in os.listdir(".") if f.endswith(".pkl")]
        if pkl_files:
            print("\nPickle files in the current directory:", file=sys.stderr)
            for f in pkl_files:
                print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    regenerate_all_plots(args.pickle_file, args.output_dir)


if __name__ == "__main__":
    main()
