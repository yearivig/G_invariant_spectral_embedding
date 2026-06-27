# Bandwidth Method Used In Both Experiments

This note documents the **same bandwidth-selection idea** applied to:

1. `comparing_to_yoel_and_eitan/experiment_2026-03-05` (image/Fourier-Bessel experiments)
2. `full_pointcloud_experiment` (pointcloud pipeline)

---

## Core Method (Median Heuristic + Local Sweep)

For kernels of the form

\[
K(x,y) = \exp\left(-\frac{\|x-y\|^2}{\mathrm{bw}}\right),
\]

we estimate a natural global scale by:

\[
\mathrm{bw}_0 = \mathrm{median}_{i<j}\left(\|x_i - x_j\|^2\right).
\]

Then we run a short sweep over multiplicative factors:

\[
\mathrm{bw} \in \{f \cdot \mathrm{bw}_0 : f \in \mathcal{F}\}.
\]

This avoids two failure modes:

- **bw too large**: all affinities \(\approx 1\), graph loses geometry.
- **bw too small**: affinities \(\approx 0\), graph disconnects.

---

## A) Image Experiment (82x82 / 79x79 FB embeddings)

### Where implemented

- `experiment_2026-03-05/bandwidth_search.py`
- `experiment_2026-03-05/run_final.py`
- `experiment_2026-03-05/run_old_dataset.py`

### Selection procedure

1. Compute pairwise \(\|x_i-x_j\|^2\) in FB coefficient space.
2. Set \(\mathrm{bw}_0\) to the median.
3. Sweep factors around \(\mathrm{bw}_0\) (and bispectrum-specific scales).
4. Rank candidates by spectral-gap proxy (and visual smoothness).

### Output locations

- New dataset final sweep figures:  
  `comparing_to_yoel_and_eitan/experiment_2026-03-05/figures/final/`
- Old dataset sweep figures:  
  `comparing_to_yoel_and_eitan/experiment_2026-03-05/figures/old_dataset/`

---

## B) Full Pointcloud Experiment

### What was added

1. New utility in  
   `full_pointcloud_experiment/utils.py`:
   - `compute_bandwidth_median_sq(data)`  
     computes median pairwise squared distance after flattening each sample.

2. New runner script:
   - `full_pointcloud_experiment/run_median_bandwidth_experiment.py`
   - Computes \(\mathrm{bw}_0\), then runs a factor sweep.

### Command used

```bash
python3 -m full_pointcloud_experiment.run_median_bandwidth_experiment \
  --data-path repository rootdata/data_3D.pkl \
  --save-folder repository rootoutputs \
  --save-name run_medianBW \
  --num-points 200 \
  --movement rotation \
  --method integral \
  --laplacian-type GL \
  --factors 0.5,1.0,2.0 \
  --render-pdf
```

### Run summary

- Computed \(\mathrm{bw}_0 = 9744.976569935869\)
- Ran factors: `0.5, 1.0, 2.0`
- Final bandwidths:
  - `4872.488284967934`
  - `9744.976569935869`
  - `19489.953139871737`

### Pointcloud output files

- Summary CSV:  
  `outputs/median_bandwidth_sweep_summary.csv`
- PKL outputs:  
  `outputs/run_medianBW_NOP:200_IM:integral_M:rotation_BW:..._LT:GL.pkl`
- PDFs:  
  `outputs/new_plots/run_medianBW_NOP:200_IM:integral_M:rotation_BW:..._LT:GL.pdf`

---

## Practical Recommendation

Use this as your default protocol in both projects:

1. Compute \(\mathrm{bw}_0 = \mathrm{median}(\|x_i-x_j\|^2)\)
2. Sweep a small factor set (e.g. `0.5, 1.0, 2.0`, optionally `0.25, 4.0`)
3. Pick the model with best geometric quality in embedding
   (smooth parameterization / minimal folding / best spectral separation)

This is robust, data-adaptive, and reproducible across different datasets/scales.
