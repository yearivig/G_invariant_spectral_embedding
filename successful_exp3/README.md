# SO(3)/SO(2) Single-Point Experiments - Successful Results

This directory contains the successful experiments for analyzing G-invariant kernels on SO(3) with right SO(2) action.

## Overview

These experiments evaluate the convergence of graph Laplacian estimators using different kernel approaches:
- **Euclidean kernel**: Standard Gaussian kernel in the ambient space
- **Minimum-orbit kernel**: Kernel using minimum distance over the orbit
- **Integral kernel**: Haar-averaged kernel over the group action

## Files

### Main Experiment Script
- `converges_so3_so2_exp_journal.py`: Complete experiment with journal-quality plots and pickle file output

### Utilities
- `replot_from_pickle.py`: Regenerate plots from saved results without re-running experiments

### Output Files
- `so3_so2_P_1_00_results.pkl`: Saved experimental results (generated after running experiments)
- `plots/`: Directory containing publication-quality PDF plots

## Usage

### Running the Full Experiment

```bash
cd /a/home/cc/students/math/vigderyeari/Documents/G_invariant_kernel/exp3/successful_exp3
python converges_so3_so2_exp_journal.py
```

This will:
1. Run experiments for all three kernels
2. Generate individual and combined log-log plots (PDF format)
3. Save results to pickle file for later analysis
4. Display convergence slopes in the terminal

**Note**: Full experiments may take considerable time (several minutes to hours depending on parameters).

### Regenerating Plots from Saved Data

If you want to modify plot aesthetics without re-running experiments:

```bash
python replot_from_pickle.py so3_so2_P_1_00_results.pkl
```

Or simply:
```bash
python replot_from_pickle.py
```
(automatically finds the default pickle file)

### Customizing Plot Output

To save plots to a different directory:
```bash
python replot_from_pickle.py so3_so2_P_1_00_results.pkl -o /path/to/output/dir
```

## Experiment Parameters

Default configuration:
- **ell**: 1 (eigenfunction degree)
- **num_points**: 10,000 (sample points, 300,000 for Euclidean)
- **num_trials**: 500 (Monte Carlo trials)
- **num_group_samples**: 200 (discretization of SO(2))
- **epsilon_range**: exp([-6, -1]) (100 logarithmically spaced values)

To modify parameters, edit the `main()` function in `converges_so3_so2_exp_journal.py`.

## Output Plots

All plots are saved as high-resolution PDFs suitable for journal publication:

### Individual Kernel Plots
- `so3_so2_P_1_00_euclidean_journal.pdf`
- `so3_so2_P_1_00_min_orbit_journal.pdf`
- `so3_so2_P_1_00_integral_journal.pdf`

### Combined Plot
- `so3_so2_P_1_00_combined_journal.pdf`

Each plot displays:
- Log-log error vs. epsilon
- Fitted convergence slope (on variance regime)
- Publication-quality formatting

## Interpreting Results

### Convergence Slopes
The log-log slope indicates the convergence rate:
- **Slope ≈ 1**: First-order convergence (typical for Euclidean kernel)
- **Slope ≈ 2**: Second-order convergence (expected for G-invariant kernels)

### Error Regimes
Each plot typically shows two regimes:
1. **Variance regime** (large ε): Error dominated by kernel approximation
2. **Bias regime** (small ε): Error dominated by finite sampling

The slope is computed on the variance regime (before the minimum error point).

## Data Persistence

Results are saved in pickle format containing:
- `epsilon_values`: Array of epsilon values tested
- `errors_by_kernel`: Dictionary of RMSE values for each kernel
- `log_epsilon`: Log-transformed epsilon values
- `log_errors_by_kernel`: Log-transformed errors
- `slopes_by_kernel`: Fitted convergence slopes
- `metadata`: Experimental parameters

## Mathematical Background

**Eigenfunction**: We use f(R) = P₁⁰⁰(R) = R[2,2], which is the ℓ=1 Wigner D-function.

**Exact Laplacian**: Δf = -ℓ(ℓ+1)f = -2f

**Estimator**: Δ̂_ε f(R₀) = (2/ε) · L̂ f(R₀), where L̂ is the row-normalized graph Laplacian.

**Error Metric**: RMSE of (Δ̂_ε f(R₀) + ℓ(ℓ+1)) across trials.

## Dependencies

```
numpy
matplotlib
tqdm
pickle (standard library)
```

## Citation

If you use these experiments in a publication, please cite the corresponding manuscript on G-invariant kernels and Laplacian estimation on homogeneous spaces.

## Contact

For questions or issues with these experiments, please contact the research group.

---

*Last updated: October 2025*

