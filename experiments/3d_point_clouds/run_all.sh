#!/usr/bin/env bash
# Figure 4 (Section 5.1): all 24 panels into results/figure4/: every cloud centred with the
# 80 stationary points, minimum kernel exact (Kabsch).
# Overrides from the environment, e.g.  MIN_METHOD=grid bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

DATA=${DATA:-data/data_3D.pkl}
OUT=${OUT:-results/figure4}
MIN_METHOD=${MIN_METHOD:-kabsch}    # kabsch (exact) or grid (600 rotations, as in the previous Figure 4)
SEED=${SEED:-0}

if [ ! -f "$DATA" ]; then
  echo "missing $DATA (see data/README.md)"; exit 1
fi
python3 scripts/02_figure4.py --data-path "$DATA" --save-folder "$OUT" --min-method "$MIN_METHOD" --seed "$SEED"
