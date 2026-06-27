# Paper-Ready Method For Point-Cloud Manifold Embedding

## Goal

Produce a high-quality 2D embedding for the rotation dataset that is as close as possible to a full circle (periodic 1D manifold).

## Recommended Method

Use **Invariant-Features + Self-Tuning Diffusion Maps**:

1. Build a rotation-invariant descriptor for each sample:
   \[
   G_i = X_i X_i^\top
   \]
2. Define pairwise invariant distances:
   \[
   d_{ij}^2 = \|G_i - G_j\|_F^2
   \]
3. Build local-scale (self-tuning) affinity:
   \[
   W_{ij} = \exp\!\left(-\frac{d_{ij}^2}{\sigma_i \sigma_j}\right)
   \]
   where \(\sigma_i\) is the distance to the \(k\)-th nearest neighbor.
4. Apply diffusion maps with \(\alpha=1\) normalization and use first two non-trivial coordinates.

This avoids the global-bandwidth sensitivity that hurt the fixed-bandwidth kernel.

## Why This Is Better For This Dataset

- **Rotation-invariant by construction** through Gram descriptors.
- **Local scaling** handles non-uniform sampling density much better than one global BW.
- **Diffusion coordinates** better recover smooth periodic manifolds (circle-like embeddings).

## Sources (for paper discussion)

- Zelnik-Manor, L., Perona, P., *Self-Tuning Spectral Clustering*, NeurIPS 2004.  
  [https://papers.nips.cc/paper/2004/hash/40173ea48d9567f1f393b20c855bb40b-Abstract.html](https://papers.nips.cc/paper/2004/hash/40173ea48d9567f1f393b20c855bb40b-Abstract.html)
- Berry, T., Harlim, J., *Variable Bandwidth Diffusion Kernels*, Applied and Computational Harmonic Analysis 2016.  
  [https://math.gmu.edu/~berry/Publications/VariableBandwidth.pdf](https://math.gmu.edu/~berry/Publications/VariableBandwidth.pdf)

## Implementation Added

- New method name: `invariant_features_self_tuning`
- Implemented in:
  - `full_pointcloud_experiment/utils.py`
  - wired through:
    - `full_pointcloud_experiment/pipeline.py`
    - `full_pointcloud_experiment/run_experiment.py`
    - `full_pointcloud_experiment/run_full_pipeline.py`

## Command Used

```bash
python3 -m full_pointcloud_experiment.run_experiment \
  --data-path repository rootdata/data_3D.pkl \
  --save-folder repository rootoutputs \
  --save-name run_paper_best_k30 \
  --num-points 200 \
  --invariant-method invariant_features_self_tuning \
  --movement rotation \
  --bandwidth 1 \
  --is-centered false \
  --add-stationary false \
  --noise 0 \
  --noise-tag AN \
  --laplacian-type RWGL \
  --knn-k 30 \
  --render-pdf true
```

## Output Files

- PDF:
  `outputs/new_plots/run_paper_best_k30_NOP:200_IM:invariant_features_self_tuning_M:rotation_BW:1.0_IC:False_AS:False_AN:0.0_LT:RWGL.pdf`
- PKL:
  `outputs/run_paper_best_k30_NOP:200_IM:invariant_features_self_tuning_M:rotation_BW:1.0_IC:False_AS:False_AN:0.0_LT:RWGL.pkl`

## Quantitative Comparison vs Reference

Reference file:
`outputs/new_plots/run_NOP:200_IM:integral_M:rotation_BW:47_IC:False_AS:False_AN:0_LT:RWGL.pdf`

Simple circle-quality indicators on the corresponding PKLs:

- Angular coverage (24 bins):
  - Reference: `0.833`
  - New method: `1.000`
- Radius coefficient of variation (lower is better circle):
  - Reference: `0.653`
  - New method: `0.199`

Interpretation: the new method is substantially more circle-like and covers the full angle range.
