# SO(3)/SO(2) convergence (Section 3.5, Figure 2)

Convergence of the graph Laplacian at one point for the Euclidean, minimum and
integral kernels, with $M = \mathrm{SO}(3) \subset \mathbb R^{3\times 3}$ (Frobenius
metric) and $G = \mathrm{SO}(2)$ acting by $R \mapsto R\,R_z(\theta)$, so that
$M/G \cong S^2$ (Example 3.16). The test function is $f(R) = R_{22} = \cos\beta(R)$,
with $\Delta f = f$, and the experiment measures the mean absolute error of
$\tfrac{4}{\varepsilon}(L_{RW} f)(I_3)$ against 1 as $\varepsilon$ varies.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh                      # Figure 2 from the saved results (seconds) -> results/new_run/
RUN_EXPERIMENT=1 bash run_all.sh     # rerun the experiment first (hours), then Figure 2
```

## What runs

| file | |
|---|---|
| `tools/converges_so3_so2.py` | the experiment: Haar sampling on SO(3) and SO(2), the three kernels, the row-normalised graph Laplacian at $R_0 = I_3$, the paired-sampling runner, slopes, plots and pickle I/O |
| `tools/replot_from_pickle_fitted_slopes.py` | Figure 2 from a results pickle: fitted- and predicted-slope reference lines |
| `tools/replot_from_pickle.py` | the experiment's own plots from a results pickle (not in the paper) |
| `scripts/01_run_experiment.py` | runs the experiment, `--seed` (default 0), `--out` (default `results/new_run`) |
| `scripts/02_figure2.py` | Figure 2 from a pickle (default: the committed one) |

**Settings** (as in Section 3.5): $n = 10000$ points, the first at $I_3$ and the
rest Haar-uniform on SO(3); $m = 200$ Haar-random SO(2) elements for the
minimum and integral kernels; $T = 1000$ trials; 100 values of
$\log\varepsilon$ evenly spaced in $[-4, 0.4]$; kernels
$\exp(-\lVert R - R_0 g\rVert_F^2/\varepsilon)$ (minimum and mean over $g$ for
the invariant kernels).

**Paired sampling.** In each trial the points and the SO(2) elements are drawn
once and reused for every $\varepsilon$, so each kernel's mean-error curve is a
smooth function of $\varepsilon$; the pointwise 95% confidence band
(mean ± 1.96 SEM over the trials) remains valid.

**Fitted slopes.** Least-squares lines through $\log$ mean $|$error$|$ against
$\log\varepsilon$, over each kernel's points left of its error minimum, from
$\log\varepsilon = -3$ for the Euclidean kernel and from $-4$ for the minimum
and integral kernels. On the committed results: Euclidean $-0.853$, minimum
$-0.503$, integral $-0.505$ (predicted: $-0.75$, $-0.5$, $-0.5$; Remark 3.12).
The Euclidean reference lines are drawn 0.05 higher, for legibility only.

## Results

```
results/so3_so2_P_1_00_results.pkl                       trial-level errors behind Figure 2
results/plots_fitted_slopes/so3_so2_P_1_00_combined.pdf  Figure 2, the file in the paper
```

Both are committed. The pickle came from an unseeded run, so rerunning the
experiment gives statistically equivalent but not identical numbers; runs are
now seeded (`--seed`) and write to `results/new_run/` so the committed pickle is
kept. `scripts/02_figure2.py` on the committed pickle with matplotlib 3.10.8
(pinned in `requirements.txt`) reproduces the paper's PDF byte for byte, apart
from its creation date; newer matplotlib versions draw the same figure but
encode the PDF differently.

## Requirements

Python 3.12 recommended (3.14 also tested); `numpy`, `matplotlib==3.10.8`,
`tqdm`.
