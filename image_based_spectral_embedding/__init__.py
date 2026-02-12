"""
Curated, documented version of the image-based spectral embedding experiment.

This package implements the pipeline from
    ../comparing_to_yoel_and_eitan/apply_method.py
in a clean, modular, reproducible form — inspired by
    ../full_pointcloud_experiment/

The experiment:
  1) Load 2D projection images (e.g. cryo-EM style)
  2) Expand them in a Fourier-Bessel (FFB) basis using ASPIRE
  3) Optionally add noise and apply random SO(2) rotations
  4) Build a G-invariant affinity matrix W using one of several kernels:
     - "min"  : max_k exp(-||x - R_k y||^2 / ε)
     - "mean" : mean_k exp(-||x - R_k y||^2 / ε)
     - "bispectrum" : use bispectrum invariant features + vanilla Gaussian kernel
     - "none" : vanilla Gaussian kernel (no invariance)
  5) Compute eigenvectors of the RW graph Laplacian
  6) Save / plot the spectral embedding
"""
