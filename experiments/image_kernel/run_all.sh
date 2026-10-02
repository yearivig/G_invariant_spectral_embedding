#!/usr/bin/env bash
# Figure 5 (Section 5.2): all kernels and bandwidths, clean and SNR 1 dB, into results/figure5/.
# Overrides from the environment, e.g.  DATA=/path/to/data SEED=1 bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

DATA=${DATA:-data}
OUT=${OUT:-results/figure5}
SEED=${SEED:-0}

for f in projections-synth.pt output_main.csv; do
  if [ ! -f "$DATA/$f" ]; then echo "missing $DATA/$f (see data/README.md)"; exit 1; fi
done
python3 scripts/01_figure5.py --data-dir "$DATA" --out "$OUT" --seed "$SEED"
