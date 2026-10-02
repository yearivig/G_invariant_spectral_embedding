#!/usr/bin/env bash
# All experiments of the paper, in its order. Each writes to its own experiments/<name>/results/.
#
#   POINTCLOUD_DATA   the Glucagon trajectory pickle (default experiments/pointcloud/data/data_3D.pkl)
#   IMAGE_DATA_DIR    folder with projections-synth.pt and output_main.csv (default experiments/image_kernel/data)
#   RUN_EXPERIMENT=1  rerun the Section 3.5 experiment instead of redrawing Figure 2 from saved results
#   SKIP_TORUS=1      skip Section 5.3, whose minimum kernel takes days on a CPU
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POINTCLOUD_DATA="${POINTCLOUD_DATA:-${ROOT}/experiments/pointcloud/data/data_3D.pkl}"
IMAGE_DATA_DIR="${IMAGE_DATA_DIR:-${ROOT}/experiments/image_kernel/data}"

echo "[1/4] Section 3.5: SO(3)/SO(2) convergence (Figure 2)"
bash "${ROOT}/experiments/so3_so2_convergence/run_all.sh"   # RUN_EXPERIMENT=1 to rerun it (hours)

echo "[2/4] Section 5.1: 3D point clouds (Figure 4)"
DATA="${POINTCLOUD_DATA}" bash "${ROOT}/experiments/pointcloud/run_all.sh"

echo "[3/4] Section 5.2: tomographic images (Figure 5)"
DATA="${IMAGE_DATA_DIR}" bash "${ROOT}/experiments/image_kernel/run_all.sh"

if [ "${SKIP_TORUS:-0}" = 1 ]; then
  echo "[4/4] Section 5.3: skipped (SKIP_TORUS=1)"
else
  echo "[4/4] Section 5.3: two spinning objects (Figure 7)"
  (cd "${ROOT}/experiments/torus_double_rotation" && \
    INTEGRAL=1 bash run_all.sh)
fi

echo "[done] all experiments completed."
