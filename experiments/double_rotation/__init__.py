"""
Double Rotation Parameterization Experiment (curated + documented).

This package is a cleaned-up copy of the experiment in
    ../roy_lederman_data/

The experiment studies images that are parameterized by two independent
rotations (e.g. left and right halves of a molecule, each with its own
orientation). The goal is to recover these two rotation parameters from
the spectral embedding of a G-invariant (SO(2)-invariant) diffusion map.

Pipeline overview:
  1) Load grayscale images from disk
  2) Preprocess: make square, extract centered disk mask
  3) Split each image into left and right halves
  4) Compute diffusion maps on:
     - full images
     - left halves only
     - right halves only
  5) Apply random SO(2) rotations to the full images
  6) Recompute standard Euclidean diffusion maps (to show rotation breaks them)
  7) Compute SO(2)-invariant min and integral diffusion maps for comparison
  8) Optionally decompose the product manifold using eigenvector factorization
  9) Save all results (pickles) and generate 3D scatter plot visualizations

Dependencies: numpy, scipy, matplotlib, torch, torchvision, tqdm, cvxpy (optional)
"""
