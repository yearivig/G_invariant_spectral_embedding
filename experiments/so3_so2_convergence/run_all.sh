#!/usr/bin/env bash
# Figure 2 (Section 3.5). By default, redraw it from the committed results
# (results/so3_so2_P_1_00_results.pkl -> results/new_run/plots_fitted_slopes/).
#   RUN_EXPERIMENT=1 bash run_all.sh   rerun the experiment first (hours), into results/new_run/,
#                                      and draw Figure 2 from that run
#   SEED=1                             seed of the rerun (default 0)
set -euo pipefail
cd "$(dirname "$0")"

if [ "${RUN_EXPERIMENT:-0}" = 1 ]; then
  OUT=${OUT:-results/new_run}
  python3 scripts/01_run_experiment.py --out "$OUT" --seed "${SEED:-0}"
  python3 scripts/02_figure2.py "$OUT/so3_so2_P_1_00_results.pkl"
else
  python3 scripts/02_figure2.py
fi
