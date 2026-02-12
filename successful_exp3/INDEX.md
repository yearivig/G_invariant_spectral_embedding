# Project Index - SO(3)/SO(2) Journal Experiments

**Location:** `/a/home/cc/students/math/vigderyeari/Documents/G_invariant_kernel/exp3/successful_exp3`

**Date:** October 30, 2025

**Status:** ✅ Ready for Use

---

## Quick Start

### Option 1: Interactive Script (Recommended)
```bash
cd /a/home/cc/students/math/vigderyeari/Documents/G_invariant_kernel/exp3/successful_exp3
./run_experiment.sh
```

### Option 2: Direct Python Execution
```bash
cd /a/home/cc/students/math/vigderyeari/Documents/G_invariant_kernel/exp3/successful_exp3
python converges_so3_so2_exp_journal.py
```

### Option 3: Replot from Existing Data
```bash
cd /a/home/cc/students/math/vigderyeari/Documents/G_invariant_kernel/exp3/successful_exp3
python replot_from_pickle.py
```

---

## File Descriptions

### Core Scripts

| File | Purpose | Run Time |
|------|---------|----------|
| `converges_so3_so2_exp_journal.py` | Main experiment with all 3 kernels | 30-60 min |
| `replot_from_pickle.py` | Regenerate plots from saved data | < 5 sec |
| `run_experiment.sh` | Interactive runner with checks | 30-60 min |

### Documentation

| File | Content |
|------|---------|
| `README.md` | Complete usage guide and background |
| `CHANGES_AND_IMPROVEMENTS.md` | What's new in journal version |
| `INDEX.md` | This file - quick reference |
| `requirements.txt` | Python dependencies |

### Output Files (Generated)

| File | Description | Format |
|------|-------------|--------|
| `so3_so2_P_1_00_results.pkl` | Complete experimental data | Binary (pickle) |
| `plots/so3_so2_P_1_00_euclidean_journal.pdf` | Euclidean kernel plot | PDF |
| `plots/so3_so2_P_1_00_min_orbit_journal.pdf` | Min-orbit kernel plot | PDF |
| `plots/so3_so2_P_1_00_integral_journal.pdf` | Integral kernel plot | PDF |
| `plots/so3_so2_P_1_00_combined_journal.pdf` | Combined comparison plot | PDF |

---

## Workflow Examples

### First-Time User
```bash
# 1. Navigate to directory
cd /a/home/cc/students/math/vigderyeari/Documents/G_invariant_kernel/exp3/successful_exp3

# 2. Read the documentation
cat README.md

# 3. Run the experiment
./run_experiment.sh

# 4. View results
ls plots/*.pdf
```

### Modifying Plot Aesthetics
```bash
# 1. Edit the plotting function in replot_from_pickle.py
nano replot_from_pickle.py  # or your preferred editor

# 2. Regenerate plots instantly
python replot_from_pickle.py

# 3. Check the new plots
ls plots/*.pdf
```

### Changing Experiment Parameters
```bash
# 1. Edit the main script
nano converges_so3_so2_exp_journal.py

# 2. Modify the main() function (around line 470):
#    - num_points: number of sample points
#    - num_trials: Monte Carlo trials
#    - epsilon range: bandwidth values to test

# 3. Run with new parameters
python converges_so3_so2_exp_journal.py
```

---

## Key Features

### ✅ What This Version Does Well

1. **Publication-Ready Output**
   - PDF vector graphics
   - Professional typography
   - LaTeX-style mathematical notation
   - Journal-appropriate styling

2. **Efficient Workflow**
   - Save results to pickle files
   - Instant plot regeneration
   - No need to re-run expensive computations

3. **Complete Documentation**
   - Detailed README
   - Inline code comments
   - Mathematical background
   - Usage examples

4. **Professional Organization**
   - Clean directory structure
   - Logical file naming
   - Version control friendly
   - Reproducible results

---

## Comparison: Old vs New

| Aspect | Original Version | Journal Version |
|--------|-----------------|-----------------|
| Plot Format | PNG | PDF (vector) |
| Plot Style | Basic matplotlib | Publication-quality |
| Number of Plots | 2-panel (loglog + log-log) | 1-panel (log-log only) |
| Data Persistence | None | Pickle files |
| Replotting | Re-run full experiment | Instant from pickle |
| Documentation | Inline comments only | Complete docs |
| Scripts | 1 file | 3 files (main, replot, runner) |
| Output Directory | Mixed with source | Organized subdirectory |

---

## Troubleshooting

### Common Issues

**Issue:** `ImportError: No module named 'tqdm'`
```bash
pip install -r requirements.txt
```

**Issue:** Script not found when running `./run_experiment.sh`
```bash
chmod +x run_experiment.sh
```

**Issue:** Pickle file not found when replotting
```bash
# First run the main experiment to generate data
python converges_so3_so2_exp_journal.py
```

**Issue:** Plots don't look right
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# If needed, force a specific backend
export MPLBACKEND=Agg
```

---

## Performance Notes

### Computational Requirements

- **Memory:** ~2-4 GB RAM
- **CPU:** Any modern processor (multi-core beneficial)
- **Disk:** ~100 MB for results
- **Time:** 30-60 minutes (depends on CPU)

### Timing Breakdown (Approximate)

| Kernel | Points | Trials | Time |
|--------|--------|--------|------|
| Euclidean | 300,000 | 500 | ~30-40 min |
| Min-Orbit | 10,000 | 500 | ~8-12 min |
| Integral | 10,000 | 500 | ~8-12 min |

**Total:** ~50-65 minutes (with 100 epsilon values)

---

## Mathematical Summary

**Problem:** Estimate the Laplace-Beltrami operator on SO(3)/SO(2) using kernel methods

**Test Function:** f(R) = P₁⁰⁰(R) = R[2,2] (ℓ=1 Wigner D-function)

**Exact Laplacian:** Δf = -2f

**Kernels Tested:**
1. **Euclidean:** K(R,R₀) = exp(-‖R-R₀‖²_F/(2ε))
2. **Min-Orbit:** K(R,R₀) = exp(-min_g ‖R-R₀g‖²_F/(2ε))
3. **Integral:** K(R,R₀) = 𝔼_g[exp(-‖R-R₀g‖²_F/(2ε))]

**Estimator:** Δ̂_ε f = (2/ε)L̂f, where L̂ is row-normalized graph Laplacian

**Error Metric:** RMSE of [Δ̂_ε f(I) + 2] over trials

---

## Dependencies

```
numpy >= 1.20.0    # Numerical computations
matplotlib >= 3.3.0 # Plotting
tqdm >= 4.60.0     # Progress bars
pickle (stdlib)     # Data persistence
```

---

## Citation

If you use this code in a publication, please cite:

```
[To be filled with manuscript citation when published]
```

---

## Contact & Support

For questions or issues:
1. Check `README.md` for detailed documentation
2. Review `CHANGES_AND_IMPROVEMENTS.md` for feature details
3. Consult code comments in the scripts

---

## Version History

- **v1.0 (Journal)** - October 30, 2025
  - Initial journal-ready version
  - PDF output, pickle persistence
  - Complete documentation

---

*Last updated: October 30, 2025*

