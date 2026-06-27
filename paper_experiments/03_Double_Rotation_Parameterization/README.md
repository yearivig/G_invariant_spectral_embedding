# Double Rotation Parameterization Experiment (curated + documented)

This folder is a **self-contained, cleaned-up copy** of the experiment originally
in `../roy_lederman_data/`, which produced results like those in
`../roy_lederman_data/experiment_data_5000_points/`.

## What this experiment does

Given a dataset of images where each image is parameterized by **two independent
rotations** (e.g. left and right halves of a molecule), this experiment:

1. **Loads and preprocesses** images (grayscale, square, disk mask)
2. **Splits** each image into left and right halves
3. **Computes standard diffusion maps** on full / left / right images
4. **Applies random SO(2) rotations** to each image
5. **Recomputes diffusion maps** (showing that rotation breaks the standard embedding)
6. **Compares embeddings on rotated images** using Euclidean, SO(2) minimum-orbit, and SO(2) integral kernels
7. **Saves** all results and generates **3D scatter plots** for each kernel

## Quick start

From the repository root:

```bash
python -m double_rotation_parameterization_experiment.run_experiment \
  --input-path roy_lederman_data/data \
  --n-images 5000 \
  --output-dir double_rotation_parameterization_experiment/outputs \
  --t 10 \
  --num-neighbors 20 \
  --num-rotations 300
```

For a quick test with fewer images:

```bash
python -m double_rotation_parameterization_experiment.run_experiment \
  --input-path roy_lederman_data/data \
  --n-images 100 \
  --output-dir double_rotation_parameterization_experiment/outputs_test \
  --device cpu
```

## File structure

| File | Purpose |
|---|---|
| `__init__.py` | Package docstring |
| `run_experiment.py` | CLI entrypoint |
| `pipeline.py` | Main experiment orchestration (`ExperimentConfig` + `run_experiment`) |
| `diffmaps.py` | Standard + SO(2)-invariant diffusion maps (GPU-accelerated) |
| `rotations.py` | Image rotation (NumPy/scipy + PyTorch) |
| `image_ops.py` | Image loading, grayscale, square padding, disk masking |
| `visualization.py` | 3D scatter plotting |
| `product_manifold.py` | Product manifold decomposition (find combos + SDP clustering) |

## Kernel comparison outputs

The rotated-image embedding comparison saves:

| Kernel | Vectors | Eigenvalues | Plot prefix |
|---|---|---|---|
| Euclidean | `euclidean_diffusion_vectors.pkl` | `euclidean_eigenvalues.pkl` | `euclidean_` |
| SO(2) minimum-orbit | `so2_min_diffusion_vectors.pkl` | `so2_min_eigenvalues.pkl` | `min_` |
| SO(2) integral | `so2_integral_diffusion_vectors.pkl` | `so2_integral_eigenvalues.pkl` | `integral_` |

`rotated_diffusion_vectors.pkl` is also written as the legacy name for the Euclidean rotated-image embedding.

## Correspondence with original code

| Original | Curated |
|---|---|
| `roy_lederman_data/main.py` | `pipeline.py` + `run_experiment.py` |
| `roy_lederman_data/diffmaps.py` | `diffmaps.py` |
| `roy_lederman_data/rotations.py` | `rotations.py` |
| `roy_lederman_data/image_operation.py` | `image_ops.py` |
| `roy_lederman_data/product_maniforld_utils.py` | `product_manifold.py` |
| (3D scatter code in main.py) | `visualization.py` |

## Known improvements over original

- No hardcoded GPU device IDs (auto-detects or use `--device`)
- No hardcoded file paths (all via CLI arguments)
- Fixed `split_eigenvectors()` signature mismatch bug
- Modular: each file is independently importable and testable
- Full docstrings explaining the math
- Clean separation of I/O, compute, and visualization

## Dependencies

- numpy, scipy, matplotlib, tqdm
- torch, torchvision (for GPU-accelerated rotations + diffusion maps)
- cvxpy (only for product manifold decomposition, optional)
