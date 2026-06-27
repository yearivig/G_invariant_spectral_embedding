# G-Invariant Spectral Embedding Experiments

This repository contains the submission-ready experiment code and lightweight
artifacts for the G-invariant spectral embedding paper.

## Repository Layout

- `experiments/image_kernel/`: image/Fourier-Bessel kernel experiments.
- `experiments/pointcloud_self_tuning/`: point-cloud experiments with self-tuning bandwidths.
- `experiments/so3_so2_convergence/`: final SO(3)/SO(2) convergence experiment.
- `results/`: lightweight CSV/NPZ summaries and publication figures.
- `data/`: small checked-in metadata used by examples.

Large generated files are intentionally excluded: raw datasets, logs, Python
caches, and pickle dumps. The scripts accept environment variables or CLI
arguments for dataset locations when those data live outside the repository.

## Reproduce Experiments

Run all submission experiments:

```bash
bash run_submission_experiments.sh
```

By default, the runner expects external datasets under `data/`. To use another
data root:

```bash
DATA_ROOT=/path/to/data bash run_submission_experiments.sh
```

The runner writes new artifacts under `results/`.

## Individual Experiments

Image kernel experiment:

```bash
PAPER_IMAGE_DATA_DIR=/path/to/new_dataset_5.3.26 \
PAPER_IMAGE_OUTPUT_DIR=results/image_kernel/new_run \
python3 experiments/image_kernel/run_final_1000.py
```

Point-cloud self-tuning experiment:

```bash
python3 experiments/pointcloud_self_tuning/run_experiment.py \
  --data-path data/data_3D.pkl \
  --save-folder results/pointcloud_self_tuning/new_run \
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
```

SO(3)/SO(2) convergence experiment:

```bash
PAPER_EXP3_OUTPUT_DIR=results/so3_so2_convergence/new_run \
python3 experiments/so3_so2_convergence/converges_so3_so2_exp_journal_gpu.py
```
