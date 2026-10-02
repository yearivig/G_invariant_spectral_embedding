# Tomographic images (Section 5.2, Figure 5)

Spectral embedding (Algorithm 1) of 2D tomographic projections of the Glucagon
molecule, whose ψ-torsion angle moves along a half-circle, each image given a
random in-plane rotation, with the Euclidean kernel and the three
SO(2)-invariant kernels of Section 3.1.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Python 3.12 recommended; 3.14 also tested
pip install -r requirements.txt
# put projections-synth.pt and output_main.csv in data/ (see data/README.md)
bash run_all.sh                                            # Figure 5 -> results/figure5/
```

## What runs

`scripts/01_figure5.py`, for each SNR level (0 = clean, 1 dB = the paper's noisy panels):

1. the first `--n-samples` (1000) projection images, plus i.i.d. Gaussian noise
   with variance $\mathrm{mean}(x^2)/10^{\mathrm{SNR}/10}$ per image;
2. expansion in ASPIRE's Fourier–Bessel basis `FFBBasis2D` with
   $\ell_{\max} = 10$ (`tools/fb_basis.py`);
3. a uniformly random rotation of each image, applied exactly to its
   coefficients (rotation acts on each angular frequency $\ell$ as a $2\times 2$
   rotation);
4. $W$ for each kernel and bandwidth (`tools/fb_kernels.py`) and Algorithm 1
   (`tools/spectral_embedding.py`).

| kernel | | default bandwidths |
|---|---|---|
| `euclidean` | $\exp(-\lVert x-y\rVert^2/\varepsilon)$ | 0.01 |
| `min` | Eq. (8), minimum over 600 equally spaced angles | 0.005, 0.01, 0.025, 0.05 |
| `mean` | Eq. (9), the integral kernel: mean over the same 600 angles (trapezoidal rule) | 0.005, 0.01, 0.025, 0.05 |
| `bispectrum` | Eq. (10) with the rotational bispectrum of Example 3.4 | 6e-9, 1.5e-8, 3e-8, 6e-8 |

The default bandwidths are the sweep of the original code, one panel per
value; `--orbit-bandwidths`, `--bispectrum-bandwidths` and
`--euclidean-bandwidth` change them. One seed (`--seed`, default 0) fixes the
noise and the rotations.

### Algorithm 1

`tools/spectral_embedding.py` computes the eigenpairs of
$L_{RW} = I - D^{-1}W$ through the similar symmetric matrix
$D^{-1/2}WD^{-1/2}$, sorts them by eigenvalue, maps them back
($\varphi = D^{-1/2}v$, unit norm), and returns $\varphi_1, \dots, \varphi_m$. The
figures plot $(\varphi_1, \varphi_2)$. $W$ is computed in the log domain and
rescaled by one global constant so that no row underflows, which leaves
$L_{RW}$ unchanged.

### The kernels

The 600 angles form a group, so the minimum and integral kernels are exactly
symmetric. They are evaluated for all pairs at once through
$\langle x, R_\theta y\rangle = C + \sum_\ell [A_\ell\cos\ell\theta + B_\ell\sin\ell\theta]$.
The bispectrum is ASPIRE's flattened bispectrum
(`calculate_bispectrum(flatten=True)`), but it is never stored: its inner
products are computed from per-frequency Gram matrices of the complex
coefficients, which gives the same distances as the explicit bispectrum
(relative difference about $10^{-15}$).

### Note on torch and ASPIRE

The data are `.pt` files, read with torch. torch and ASPIRE's NUFFT (finufft)
each ship their own OpenMP runtime, and on macOS a process that loads both
aborts or deadlocks (ASPIRE imports torch through pymanopt whenever it is
installed). The Fourier–Bessel expansion therefore always runs in a
subprocess in which torch cannot be imported.

## Outputs

```
results/figure5/figures/[snr1/]<kernel>_bw<eps>.pdf                      (phi_1, phi_2), coloured by the torsion angle
results/figure5/figures/[snr1/]euclidean_no_random_angle_bw<eps>.pdf     Euclidean, coloured by the torsion angle
results/figure5/figures/[snr1/]euclidean_random_angle_bw<eps>.pdf        Euclidean, coloured by the applied rotation
results/figure5/figures/[snr1/]comparison_grid.pdf                       all panels of one SNR
results/figure5/tabular/[snr1/]embedding_<tag>.csv                       phi_1..phi_3, torsion angle, applied rotation
results/figure5/tabular/[snr1/]spectrum_<tag>.npz                        phi and lambda_0..lambda_m
results/figure5/run_config.json
```

The figure names are those of the paper's panels.

## Re-tuning the bandwidths

`tools/bandwidth_search.py` is **for re-tuning only; it is not part of
producing Figure 5**. On the same images it prints percentiles of the
Euclidean, minimum-kernel and bispectrum squared distances, proposes
candidates (median × factors), and on a subsample runs the minimum and
bispectrum kernels at each candidate, scoring them by the gap
$\mu_1/\mu_2$ ($\mu = 1-\lambda$) and saving a plot of each. The score tends to
grow with $\varepsilon$, so inspect the plots rather than taking the top score.

```bash
python3 tools/bandwidth_search.py --data-dir data --out results/bandwidth_sweep
```

## Files

```
run_all.sh                    Figure 5
scripts/01_figure5.py         Figure 5: all kernels, bandwidths and SNR levels
tools/fb_basis.py             Fourier-Bessel expansion (ASPIRE, in a torch-free subprocess)
tools/fb_kernels.py           noise, rotations, the four kernels
tools/spectral_embedding.py   Algorithm 1
tools/bandwidth_search.py     median-heuristic bandwidth sweep, for re-tuning only
data/README.md                where the data go
```
