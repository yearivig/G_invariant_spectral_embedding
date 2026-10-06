#!/usr/bin/env bash
# The experiments of the paper, in its order. Each writes to its own experiments/<name>/results/.
#
# By default only the fast steps run (minutes on a laptop):
#   Section 3.5  Figure 2, redrawn from the committed results              seconds
#   Section 5.1  Figure 4, if its data are present (otherwise skipped)     minutes
#   Section 5.2  Figure 5, if its data are present (otherwise skipped)     minutes
# Heavy steps run only when asked for:
#   RUN_SPINNING_OBJECTS=1  Section 5.3, Figure 7: downloads COIL-100 and computes the minimum and
#                           integral kernels on 5184 images, roughly an hour each on an Apple M2
#                           (GPU or CPU); INTEGRAL=0 skips the integral kernel
#   RUN_EXPERIMENT=1        rerun the Section 3.5 experiment (hours) instead of redrawing Figure 2
# Data locations:
#   POINTCLOUD_DATA   the Glucagon trajectory pickle (default experiments/3d_point_clouds/data/data_3D.pkl)
#   IMAGE_DATA_DIR    folder with projections-synth.pt and output_main.csv (default experiments/tomographic_images/data)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POINTCLOUD_DATA="${POINTCLOUD_DATA:-${ROOT}/experiments/3d_point_clouds/data/data_3D.pkl}"
IMAGE_DATA_DIR="${IMAGE_DATA_DIR:-${ROOT}/experiments/tomographic_images/data}"
SUMMARY=()

echo "[1/4] Section 3.5: SO(3)/SO(2) convergence (Figure 2)"
bash "${ROOT}/experiments/so3_so2_convergence/run_all.sh"
SUMMARY+=("Section 3.5 (Figure 2): done$([ "${RUN_EXPERIMENT:-0}" = 1 ] && echo ', experiment rerun' || echo ', from saved results')")

echo; echo "[2/4] Section 5.1: 3D point clouds (Figure 4)"
if [ -f "${POINTCLOUD_DATA}" ]; then
  DATA="${POINTCLOUD_DATA}" bash "${ROOT}/experiments/3d_point_clouds/run_all.sh"
  SUMMARY+=("Section 5.1 (Figure 4): done")
else
  echo "   skipped: ${POINTCLOUD_DATA} not found (see experiments/3d_point_clouds/data/README.md)"
  SUMMARY+=("Section 5.1 (Figure 4): skipped, data missing")
fi

echo; echo "[3/4] Section 5.2: tomographic images (Figure 5)"
if [ -f "${IMAGE_DATA_DIR}/projections-synth.pt" ] && [ -f "${IMAGE_DATA_DIR}/output_main.csv" ]; then
  DATA="${IMAGE_DATA_DIR}" bash "${ROOT}/experiments/tomographic_images/run_all.sh"
  SUMMARY+=("Section 5.2 (Figure 5): done")
else
  echo "   skipped: projections-synth.pt or output_main.csv not found in ${IMAGE_DATA_DIR}"
  echo "   (see experiments/tomographic_images/data/README.md)"
  SUMMARY+=("Section 5.2 (Figure 5): skipped, data missing")
fi

echo; echo "[4/4] Section 5.3: spinning objects (Figure 7)"
if [ "${RUN_SPINNING_OBJECTS:-0}" = 1 ]; then
  (cd "${ROOT}/experiments/spinning_objects" && INTEGRAL="${INTEGRAL:-1}" bash run_all.sh)
  SUMMARY+=("Section 5.3 (Figure 7): done")
else
  echo "   skipped: takes days on a CPU; run with RUN_SPINNING_OBJECTS=1"
  SUMMARY+=("Section 5.3 (Figure 7): skipped (RUN_SPINNING_OBJECTS=1 to run it)")
fi

echo; echo "Summary:"
printf '  %s\n' "${SUMMARY[@]}"
