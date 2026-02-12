# Image-Based Spectral Embedding (curated + documented)

This folder is a **self-contained, cleaned-up copy** of the image-based spectral
embedding experiment originally in `../comparing_to_yoel_and_eitan/apply_method.py`.

## What this experiment does

Given a set of 2D projection images (e.g. cryo-EM projections):

1. **Expand** each image in a steerable Fourier–Bessel basis (`FFBBasis2D` from ASPIRE)
2. **Add noise** at a specified SNR (dB)
3. **Apply random SO(2) rotations** to each expanded image
4. **Build an affinity matrix** using one of several G-invariant kernels:
   - `min` : max over SO(2) orbit of Gaussian kernel
   - `mean` : average over SO(2) orbit of Gaussian kernel
   - `bispectrum` : bispectrum invariant features + vanilla Gaussian kernel
   - `none` : vanilla Gaussian kernel (no SO(2) invariance)
5. **Compute eigenvectors** of the random-walk graph Laplacian
6. **Save** eigenvectors as `.pkl` and optionally **render** 2D scatter plots as PDF

## Quick start

From `Documents/G_invariant_kernel/`:

```bash
# Single run
python -m image_based_spectral_embedding.run_experiment \
  --data-path projection_data/unrotated/unrotated_projections-synth.pkl \
  --save-folder image_based_spectral_embedding/outputs \
  --save-name eigvectors \
  --num-images 198 \
  --method min \
  --bandwidth 0.5 \
  --snr 10 \
  --ell-max 10 \
  --tol 0.01 \
  --render-pdf true

# Grid of experiments
python -m image_based_spectral_embedding.run_full_pipeline \
  --data-path projection_data/unrotated/unrotated_projections-synth.pkl \
  --save-folder image_based_spectral_embedding/outputs \
  --save-name eigvectors \
  --num-images-list 198 \
  --methods min,mean,bispectrum \
  --snr-list 0,1,5,10 \
  --bandwidth 0.5 \
  --ell-max 10 \
  --render-pdf true
```

## Dependencies

- **ASPIRE** (`pip install aspire`) — for `FFBBasis2D` and bispectrum computation
- numpy, scipy, matplotlib, tqdm

## Correspondence with the original code

| Original file | Curated file |
|---|---|
| `comparing_to_yoel_and_eitan/apply_method.py` | `pipeline.py` + `utils.py` |
| `comparing_to_yoel_and_eitan/save_figures.py` | `plot_from_pkl.py` |
| (hardcoded main block) | `run_experiment.py` (CLI) |
| (manual loop) | `run_full_pipeline.py` (grid CLI) |
