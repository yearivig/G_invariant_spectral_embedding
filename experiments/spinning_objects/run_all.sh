#!/usr/bin/env bash
# End-to-end run, from COIL-100 to the embeddings: bash run_all.sh
#
# Defaults reproduce the published dataset exactly: COIL objects 87 (left)
# and 72 (right), all 72 x 72 poses, COIL background mapped to black, one
# random SO(2) rotation per image from a fixed seed. Any variable can be
# overridden from the environment, e.g.  ROTATIONS=72 bash run_all.sh
#
# Steps whose output is already complete are skipped: with
# data/rotated_torus_dataset/ in place, nothing is downloaded or rebuilt and
# the run goes straight to the embeddings. A complete dataset is used as it
# is, whatever settings built it - so after changing OBJ_A/OBJ_B, STRIDE, BG
# or SEED, run with FORCE=1 to rebuild.
set -euo pipefail

OBJ_A=${OBJ_A:-87}        # COIL object on the left
OBJ_B=${OBJ_B:-72}        # COIL object on the right
PAIR=${PAIR:-data/coil-pair}
DATASET=${DATASET:-data/torus_dataset}
ROTATED=${ROTATED:-data/rotated_torus_dataset}
OUT=${OUT:-results/run1}

STRIDE=${STRIDE:-1}          # 1 -> all 72 x 72 = 5184 images
BG=${BG:-auto}               # COIL background -> black; 'none' to keep it grey
SEED=${SEED:-3044084360}     # rotation angles; this one gives the published set
ROTATIONS=${ROTATIONS:-300}  # SO(2) group elements for the invariant kernels (script 04's default)
NEIGHBORS=${NEIGHBORS:-20}   # k for the nearest-neighbour bandwidth rule
EPSILON=${EPSILON:-knn}      # kernel bandwidth: a number, or knn
INTEGRAL=${INTEGRAL:-0}      # 0 = Euclidean + minimum; 1 = also the SO(2) integral kernel (about doubles the run time)
                             # A later INTEGRAL=1 run reuses $OUT/so2_min_sq_distances.npy, so the
                             # minimum kernel is not recomputed; delete that file to force it.
FORCE=${FORCE:-0}            # 1 = rebuild both datasets even if they are complete


# A dataset directory is complete when it has layout.json, labels.csv, and
# exactly one image in images/ per row of labels.csv. Anything less - a run
# interrupted part-way, a folder copied without its images - is rebuilt.
complete() {
  local d=$1
  [ "$FORCE" = 1 ] && return 1
  [ -f "$d/layout.json" ] && [ -f "$d/labels.csv" ] && [ -d "$d/images" ] || return 1
  local rows imgs
  rows=$(( $(wc -l < "$d/labels.csv") - 1 ))
  imgs=$(find "$d/images" -maxdepth 1 -name '*.png' | wc -l)
  [ "$rows" -gt 0 ] && [ "$rows" -eq "$imgs" ]
}

if complete "$ROTATED"; then
  echo "== steps 1-3: $ROTATED is complete, skipping the download, build and rotation =="
else
  if complete "$DATASET"; then
    echo "== steps 1-2: $DATASET is complete, skipping the download and build =="
  else
    if [ ! -d "$PAIR/obj$OBJ_A" ] || [ ! -d "$PAIR/obj$OBJ_B" ]; then
      echo "== step 1: download COIL-100 and keep objects $OBJ_A and $OBJ_B =="
      python3 scripts/01_fetch_coil100.py --objects "$OBJ_A" "$OBJ_B" --out "$PAIR"
    else
      echo "== step 1: $PAIR already holds obj$OBJ_A and obj$OBJ_B, skipping the download =="
    fi

    echo
    echo "== step 2: build the double-rotation dataset =="
    python3 scripts/02_build_dataset.py \
        --dir-a "$PAIR/obj$OBJ_A" --dir-b "$PAIR/obj$OBJ_B" \
        --out "$DATASET" --stride "$STRIDE" --bg "$BG"
  fi

  echo
  echo "== step 3: randomly rotate every image =="
  python3 scripts/03_rotate_dataset.py --dataset "$DATASET" --out "$ROTATED" \
      --seed "$SEED"
fi

echo
echo "== steps 4-5: kernels, embeddings, plots =="
# ${arr[@]+...}: macOS ships bash 3.2, where an empty array under set -u is an error.
EXTRA_ARGS=()
if [ "$INTEGRAL" = 1 ]; then EXTRA_ARGS+=(--integral); fi
if [ -f "$OUT/so2_min_sq_distances.npy" ]; then
  echo "   reusing the minimum-kernel distances in $OUT/so2_min_sq_distances.npy"
  EXTRA_ARGS+=(--min-distances "$OUT/so2_min_sq_distances.npy")
fi
python3 scripts/04_spectral_embedding.py \
    --dataset "$ROTATED" --out "$OUT" \
    --num-group-elements "$ROTATIONS" --epsilon "$EPSILON" --num-neighbors "$NEIGHBORS" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
