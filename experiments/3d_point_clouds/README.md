# 3D point clouds (Section 5.1, Figure 4)

Spectral embedding (Algorithm 1) of point clouds of the Glucagon molecule in
which the ψ-torsion angle of the 19th residue rotates, each cloud given a
random global SO(3) rotation, with the Euclidean kernel and the three
SO(3)-invariant kernels of Section 3.1.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Python 3.12 recommended; 3.14 also tested
pip install -r requirements.txt
# put the trajectory pickle at data/data_3D.pkl (not public; see data/README.md)
bash run_all.sh                                   # all 24 panels of Figure 4 -> results/figure4/plots/
```

## What runs

| step | file | |
|---|---|---|
| point clouds | `tools/pointclouds.py` | frames, centring (IC), 80 stationary points (AS), noise, random SO(3) rotation |
| kernels | `tools/kernels.py` | Euclidean, minimum (Eq. 8), integral (Eq. 9), invariant features (Eq. 10, Gram map) |
| Algorithm 1 | `tools/spectral_embedding.py` | $L_{RW} = I - D^{-1}W$, eigenvectors of the smallest nonzero eigenvalues |
| one run | `tools/pipeline.py`, `scripts/01_embed.py` | one kernel and one setting |
| Figure 4 | `scripts/02_figure4.py`, `run_all.sh` | the 24 panels |
| plots | `tools/plot_embedding.py` | the axis-free scatter plots |
| bandwidth | `tools/retune_bandwidth.py` | **for re-tuning ε only; not used for the paper's figures** |

### The point clouds

Frame $i$ of the trajectory, for the first $n$ frames, restricted to the points
chosen by `--points`: `rotation` (default, as in Figure 4) keeps points 800
onward, the part of the file that rotates (303 of the 1103 points of each
frame); `all` keeps all 1103, including the (nearly) static part. Then, in this
order: centring (IC; on by default, `--no-is-centered` in `01_embed.py`);
80 "stationary" points appended (AS; on by default, `--no-add-stationary`), one draw
from $\mathcal N(c, 0.2^2 I_3)$ with $c$ the centroid of the first cloud, the
same 80 points in every cloud; optional Gaussian noise at an SNR in dB
(`--snr`; variance $= \mathrm{mean}(X^2)/10^{\mathrm{SNR}/10}$ per cloud); a
uniformly random rotation $X \mapsto XQ$. One seed (`--seed`, default 0) fixes
all of the randomness.

### The kernels

All kernels are computed from the $3\times 3$ products $C_{ij} = X_j^\top X_i$:
$\lVert X_i - X_jR\rVert_F^2 = \lVert X_i\rVert_F^2 + \lVert X_j\rVert_F^2 - 2\langle C_{ij}, R\rangle_F$
and $\lVert X_iX_i^\top - X_jX_j^\top\rVert_F^2 = \lVert C_{ii}\rVert_F^2 + \lVert C_{jj}\rVert_F^2 - 2\lVert C_{ij}\rVert_F^2$.

- **Minimum kernel**, two methods (`--min-method`): `kabsch` (default) takes
  the exact minimum over SO(3) in closed form (Example 3.3); `grid` takes the
  minimum over the 600-element super-Fibonacci grid on SO(3), as in the code
  behind the previous Figure 4. The grid is not closed under inverses, so the
  grid minimum is symmetrised as $W \leftarrow \max(W, W^\top)$, i.e. over the
  grid and its inverses. The grid's best rotation is a median 14° from the true
  alignment, which shows as scatter around the clean circles; Kabsch removes it.
- **Integral kernel**: the mean over the same grid (quasi-Monte Carlo for the
  Haar integral), symmetrised as $W \leftarrow (W + W^\top)/2$.
  `--num-rotations` changes the grid size for both.

$W$ is computed in the log domain and rescaled by one global constant so that
no row underflows; $L_{RW}$ is unchanged by such a rescaling.

### Algorithm 1

`tools/spectral_embedding.py` computes the eigenpairs of $L_{RW} = I - D^{-1}W$
through the similar symmetric matrix $D^{-1/2}WD^{-1/2}$, sorts them by
eigenvalue, and maps them back ($\varphi = D^{-1/2}v$, unit norm). The
embedding is $(\varphi_1, \varphi_2)$. $W$ need not be positive semidefinite;
negative eigenvalues of $D^{-1/2}WD^{-1/2}$ give $\lambda > 1$ and are never
selected.

### Figure 4

`scripts/02_figure4.py` runs the 24 panels: ε = 47 (minimum, integral) and
3000 (invariant features, Euclidean); clean panels without noise, noisy panels
at 10 dB; every cloud centred with the 80 stationary points (IC and AS on);
the minimum kernel exact (Kabsch). Files are named as in the paper's source,
e.g. `run_NOP:400_IM:min_M:rotation_BW:47_IC:True_AS:True_AN:0_LT:RWGL.pdf`;
`_MIN:grid` or `_G:<rotations>` is appended for non-default runs. Each panel
also writes a `.json` with its settings and eigenvalues. A run folder holds
`embeddings/` (the `.pkl` embeddings and `.json` records) and `plots/` (the
PDFs, redrawn from the embeddings with `python3 tools/plot_embedding.py <folder>`).

`--is-centered` and `--add-stationary` (`true`, `false`, `old-paper`) set IC and
AS for every panel; `old-paper` with `--min-method grid` reproduces the
previous Figure 4, whose clean panels mixed the settings (both off, except
Euclidean at $n=200$ and integral at $n=400$). `--kernels` runs a subset and
`--dry-run` lists the panels.

**Why IC and AS are both on.** Without a fixed reference the torsion is a
rigid motion of the moving residue, which the invariant kernels quotient out.
Uncentred clouds keep the origin as a reference, so clean panels work, but the
noise level, set from mean$(X^2)$, then includes the clouds' offset from the
origin and swamps the motion: the noisy panels collapse. Centred clouds need
the 80 stationary points as the reference. With both on, all 18 invariant
panels show the circle.

## Files

```
run_all.sh                    Figure 4
scripts/01_embed.py           one embedding
scripts/02_figure4.py         the 24 panels of Figure 4
tools/pointclouds.py          the point clouds
tools/kernels.py              the four kernels and the SO(3) grid
tools/spectral_embedding.py   Algorithm 1
tools/pipeline.py             one run: point clouds -> W -> Algorithm 1 -> saved embedding
tools/plot_embedding.py       scatter plots
tools/retune_bandwidth.py     median-heuristic bandwidth sweep, for re-tuning only
data/README.md                where the trajectory goes
```
