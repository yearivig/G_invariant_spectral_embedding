#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUT_ROOT="${ROOT}/outputs"

mkdir -p "${OUT_ROOT}"

echo "[1/4] Image experiment"
export PAPER_IMAGE_OUTPUT_DIR="${OUT_ROOT}/image_new_dataset"
export PAPER_IMAGE_FIGURES_DIR="${OUT_ROOT}/image_new_dataset/figures/comparison"
mkdir -p "${PAPER_IMAGE_OUTPUT_DIR}" "${PAPER_IMAGE_FIGURES_DIR}"
python3 "${ROOT}/01_Image_NewDataset_Experiment/run_experiment.py"
python3 "${ROOT}/01_Image_NewDataset_Experiment/generate_all_figures.py"

echo "[2/4] Pointcloud self-tuning"
python3 "${ROOT}/02_PointCloud_SelfTuning_Experiment/run_experiment.py" \
  --data-path "${DATA_ROOT}/data/data_3D.pkl" \
  --save-folder "${OUT_ROOT}/pointcloud" \
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

echo "[3/4] Double-rotation parameterization"
python3 "${ROOT}/03_Double_Rotation_Parameterization/run_experiment.py" \
  --input-path "${DATA_ROOT}/roy_lederman_data/data" \
  --n-images 5000 \
  --output-dir "${OUT_ROOT}/double_rotation" \
  --t 10 \
  --num-neighbors 20 \
  --num-rotations 300

echo "[4/4] Final SO(3)/SO(2) exp3"
EXP3_OUT="${OUT_ROOT}/final_so3_so2_exp3"
mkdir -p "${EXP3_OUT}"
(
  cd "${ROOT}/04_Final_SO3_SO2_Exp3"
  python3 "converges_so3_so2_exp_journal.py"
  mkdir -p "${EXP3_OUT}/plots"
  cp -f ./*.pkl "${EXP3_OUT}/" 2>/dev/null || true
  cp -f plots/* "${EXP3_OUT}/plots/" 2>/dev/null || true
)

echo "[done] Paper experiments completed."
