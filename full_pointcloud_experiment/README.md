# Full pointcloud experiment (curated + documented)

This folder is a **self-contained, cleaned-up copy** of the scripts used to produce the point-cloud spectral embedding experiment outputs under `../outputs/` and to render `.pkl` results into `.pdf` figures under `../outputs/new_plots/`.

It was created to be:

- **Reproducible** (single entry point script)
- **Readable** (research-grade comments + docstrings)
- **Non-invasive** (does **not** modify the original project files)

## What this corresponds to in the original project

The original experiment flow in this repo is:

- `../full_pipeline.py` → loops over a grid of experiment settings and calls
- `../pipeline.py` → loads/generates data, computes Laplacian eigenvectors, saves `.pkl`, and sometimes plots
- `../utils.py` → core math + saving helpers
- `../print_plots/print_from_pkl.py` → converts all `outputs/*.pkl` into `outputs/new_plots/*.pdf`

This directory re-implements that same flow with better separation and documentation, while keeping the numerical steps the same.

## Quick start

From `Documents/G_invariant_kernel/` run:

```bash
python -m full_pointcloud_experiment.run_experiment \
  --data-path data/data_3D.pkl \
  --save-folder outputs \
  --save-name run \
  --num-points 200 \
  --invariant-method integral \
  --movement rotation \
  --bandwidth 47 \
  --is-centered false \
  --add-stationary false \
  --noise 0.1 \
  --noise-tag AN \
  --laplacian-type GL \
  --render-pdf true
```

This will create:

- a `.pkl` under `outputs/` whose filename encodes the experiment parameters
- a corresponding `.pdf` under `outputs/new_plots/` rendered from the `.pkl`

## Notes about filename tags (`AN` vs `SNR`)

Some historical outputs in this project use a tag like `AN:0.1` (amplitude/noise parameter),
while newer code paths often use `SNR:<value>`.

In this curated folder you can control that explicitly using:

- `--noise-tag AN` (to match files like `..._AN:0.1_...`)
- `--noise-tag SNR` (to match files like `..._SNR:10_...`)

This **only affects filenames**; the meaning of the noise parameter depends on how you generate noise in the data.

## What “laplacian type” means here

We use the same convention as in the original utilities:

- `RWGL` → random-walk graph Laplacian (computed by `calc_Laplacian`)
- `GL` → unnormalized graph Laplacian (computed by `calc_Laplacian2`)

In the code, we compute **both** eigen-decompositions from the same affinity matrix `W` and choose which embedding to save based on `--laplacian-type`.

## Adaptive bandwidth (kNN)

If you pass `--auto-bandwidth true`, we compute the Gaussian bandwidth \(\varepsilon\) using:

\[
\varepsilon = \frac{1}{2}\mathrm{mean}(d_k(x_i)), \quad k = 15,
\]

where \(d_k(x_i)\) is the distance to the \(k\)-th nearest neighbor (excluding self).

Use `--knn-k 15` to control \(k\) (default is 15).