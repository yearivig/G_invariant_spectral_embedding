#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"

mkdir -p "${RESULTS_ROOT}"

echo "[1/3] Image/Fourier-Bessel kernel experiment"
export PAPER_IMAGE_DATA_DIR="${PAPER_IMAGE_DATA_DIR:-${DATA_ROOT}/new_dataset_5.3.26}"
export PAPER_IMAGE_OUTPUT_DIR="${RESULTS_ROOT}/image_kernel/new_run"
export PAPER_IMAGE_FIGURES_DIR="${RESULTS_ROOT}/image_kernel/new_run/figures"
mkdir -p "${PAPER_IMAGE_OUTPUT_DIR}" "${PAPER_IMAGE_FIGURES_DIR}"
python3 "${ROOT}/experiments/image_kernel/run_final_1000.py"

echo "[2/3] Point-cloud self-tuning experiment"
python3 "${ROOT}/experiments/pointcloud_self_tuning/run_experiment.py" \
  --data-path "${DATA_ROOT}/data_3D.pkl" \
  --save-folder "${RESULTS_ROOT}/pointcloud_self_tuning/new_run" \
  --save-name paper_run \
  --num-points 200 \
  --invariant-method invariant_features_self_tuning \
  --movement rotation \
  --bandwidth 1 \
  --noise 0 \
  --noise-tag SNR \
  --laplacian-type RWGL \
  --knn-k 30 \
  --render-pdf true

echo "[3/3] SO(3)/SO(2) convergence experiment"
export PAPER_EXP3_OUTPUT_DIR="${RESULTS_ROOT}/so3_so2_convergence/new_run"
mkdir -p "${PAPER_EXP3_OUTPUT_DIR}"
python3 "${ROOT}/experiments/so3_so2_convergence/converges_so3_so2_exp_journal_gpu.py"

echo "[done] Submission experiments completed."
