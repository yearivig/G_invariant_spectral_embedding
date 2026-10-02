#!/usr/bin/env bash
# All experiments of the paper, in its order. Outputs go under results/.
#
#   POINTCLOUD_DATA   the Glucagon trajectory pickle (default experiments/pointcloud/data/data_3D.pkl)
#   IMAGE_DATA_DIR    folder with projections-synth.pt and output_main.csv (default experiments/image_kernel/data)
#   RESULTS_ROOT      where outputs go (default results/)
#   SKIP_TORUS=1      skip Section 5.3, whose minimum kernel takes days on a CPU
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
POINTCLOUD_DATA="${POINTCLOUD_DATA:-${ROOT}/experiments/pointcloud/data/data_3D.pkl}"
IMAGE_DATA_DIR="${IMAGE_DATA_DIR:-${ROOT}/experiments/image_kernel/data}"
mkdir -p "${RESULTS_ROOT}"

echo "[1/4] Section 3.5: SO(3)/SO(2) convergence (Figure 2)"
export PAPER_EXP3_OUTPUT_DIR="${RESULTS_ROOT}/so3_so2_convergence/new_run"
mkdir -p "${PAPER_EXP3_OUTPUT_DIR}"
python3 "${ROOT}/experiments/so3_so2_convergence/converges_so3_so2_exp_journal_gpu.py"

echo "[2/4] Section 5.1: 3D point clouds (Figure 4)"
DATA="${POINTCLOUD_DATA}" OUT="${RESULTS_ROOT}/pointcloud/figure4/new_run" \
  bash "${ROOT}/experiments/pointcloud/run_all.sh"

echo "[3/4] Section 5.2: tomographic images (Figure 5)"
DATA="${IMAGE_DATA_DIR}" OUT="${RESULTS_ROOT}/image_kernel/figure5/new_run" \
  bash "${ROOT}/experiments/image_kernel/run_all.sh"

if [ "${SKIP_TORUS:-0}" = 1 ]; then
  echo "[4/4] Section 5.3: skipped (SKIP_TORUS=1)"
else
  echo "[4/4] Section 5.3: two spinning objects (Figure 7)"
  (cd "${ROOT}/experiments/torus_double_rotation" && \
    OUT="${RESULTS_ROOT}/torus_double_rotation/new_run" INTEGRAL=1 bash run_all.sh)
fi

echo "[done] all experiments completed."
