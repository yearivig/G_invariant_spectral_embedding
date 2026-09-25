# Double-rotation torus dataset for the G-invariant spectral embedding

Builds a dataset of images carrying **two independent rotation angles**, one
per object, randomly rotates every image in the plane, and embeds it with
Algorithm 1 of *Group Invariant Spectral Embedding*
([arXiv:2607.08987](https://arxiv.org/abs/2607.08987)) using the Euclidean,
SO(2)-minimum and SO(2)-integral kernels.

Two COIL-100 objects are composited into one frame, object A on the left and
object B on the right, over the full Cartesian product of their poses. The
latent space is therefore $T^2 = S^1 \times S^1$ by construction, with both
angles set independently.

## What runs

| step | done by | using |
|---|---|---|
| 1. download COIL-100, keep objects 87 and 72 in `data/coil-pair/` | `scripts/01_fetch_coil100.py` | — |
| 2. build the double-rotation dataset (72 × 72 poses, background → black) | `scripts/02_build_dataset.py` | — |
| 3. random SO(2) rotation of every complete image | `scripts/03_rotate_dataset.py` | `rotations.rotate_image` |
| 4. spectral embedding (Algorithm 1) with the Euclidean, minimum and integral kernels | `scripts/04_spectral_embedding.py` | `spectral_embedding.embed` |
| 5. save results, stereographic and donut figures per kernel | `scripts/04_spectral_embedding.py` | `tools/view_stereo.py` |

The embedding follows **Algorithm 1 of the paper**. For each kernel $K$,
`tools/spectral_embedding.py` builds
$W_{ij} = K(x_i, x_j)$ (Eq. 3), $D_{ii} = \sum_j W_{ij}$ and
$L_{RW} = I - D^{-1}W$ (Eq. 4), and maps $x_i \mapsto
(\varphi_1(x_i), \dots, \varphi_m(x_i))$, the eigenvectors of $L_{RW}$ for the
$m$ smallest nonzero eigenvalues. The kernels are

| kernel | $K(x, y)$ | |
|---|---|---|
| Euclidean | $\exp(-\lVert x-y\rVert^2/\varepsilon)$ | always |
| minimum | $\exp(-\min_{R}\lVert x-R\cdot y\rVert^2/\varepsilon)$ | Eq. (8), unless `--no-min` |
| integral | $\frac{1}{N}\sum_{R}\exp(-\lVert x-R\cdot y\rVert^2/\varepsilon)$ | Eq. (9), with `--integral` |

with $R$ over $N = 300$ equally spaced rotations by default
(`--num-group-elements`); for the integral kernel that is the uniform
trapezoidal rule on the circle. The bandwidth is `--epsilon`: a fixed value, as
in the paper, or `knn` (the default here), which sets
$\varepsilon = (\text{mean distance to the 20 nearest neighbours})^2$ per
kernel and records it in `run_config.json`. The minimum kernel's squared
distances are saved, so it can be re-embedded with another $\varepsilon$ in
seconds.

### Code

| module | origin | used by |
|---|---|---|
| `tools/spectral_embedding.py` | written for this project: Algorithm 1 and the three kernels | script 04 |
| `tools/rotations.py` | adapted from Roy Lederman's `roy_lederman_data/rotations.py` | script 03, `tools/spectral_embedding.py` |
| `tools/view_stereo.py` | written for this project | script 04, and by hand |
| `scripts/01`–`04`, `run_all.sh` | written for this project | |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements-lock.txt for the exact versions used
bash run_all.sh
```

`run_all.sh` runs every step below. Its defaults reproduce the published
dataset exactly — the rotation seed is fixed — and any setting can be
overridden from the environment (`ROTATIONS=72 bash run_all.sh`). It runs
the Euclidean and minimum kernels; `INTEGRAL=1 bash run_all.sh` adds the
integral kernel. The minimum kernel's distances are saved
(`results/run1/so2_min_sq_distances.npy`), so running `INTEGRAL=1 bash run_all.sh`
after a first run computes only the integral kernel: the minimum kernel is
re-embedded from the saved file in seconds, and the integral kernel uses the
same nearest-neighbour $\varepsilon$. The file is refused if it came from a
different dataset or number of group elements; delete it to recompute. If
`data/coil-pair/` is already present the download is skipped; if
`data/rotated_torus_dataset/` came with the folder, only step 4 is needed to
get the embeddings.

Step by step:

```bash
# 1. COIL-100, keeping only objects 87 and 72 (~130 MB download, 3.4 MB kept)
python3 scripts/01_fetch_coil100.py            # writes data/coil-pair/obj87/ and obj72/, 72 frames each

# 2. all 72 x 72 = 5184 poses, obj87 left, obj72 right, background -> black
python3 scripts/02_build_dataset.py \
    --dir-a data/coil-pair/obj87 --dir-b data/coil-pair/obj72 \
    --out data/torus_dataset --stride 1 --bg auto

# 3. one random SO(2) rotation per image
python3 scripts/03_rotate_dataset.py --dataset data/torus_dataset \
    --out data/rotated_torus_dataset --seed 3044084360

# 4-5. kernels, embeddings, plots (Euclidean and minimum; add --integral for the integral kernel)
python3 scripts/04_spectral_embedding.py \
    --dataset data/rotated_torus_dataset --out results/run1 --num-group-elements 300
```

`--dry-run` on script 04 validates the dataset and prints the cost estimate
without starting. At the full 5184 images and 308×308 this is a long run on a
CPU. Each invariant kernel compares every pair under 300 rotations by default
(script 04's own default, and `run_all.sh`'s): about 4e14 element-operations
per kernel, measured at roughly two days on an Apple M2 for the minimum kernel
(about 12 hours at 72 angles). The integral kernel, if added, costs about the
same again, so with both it is roughly four days on that laptop. The rotated images alone
take ~2 GB of memory. A GPU changes all of this by about two orders of
magnitude.

## The COIL background

COIL's turntable background is a dark grey, about 25–28 of 255, not black.
Everything the pipeline adds — the canvas around the two frames, the region
outside the disk mask, the corners a rotation exposes — is 0, so left alone
each object frame shows as a grey box that turns with the image. `--bg auto`
in script 02 measures each object's background as the median of its frame
borders (25 for obj87, 26 for obj72) and maps that level to 0 and 255 to 255
linearly, clipping below. The values used are recorded in `layout.json`.

## The layout

Most of script 02's design follows from four conventions:

1. Files are named with a **prefix** and a zero-padded index
   (`s1_00000.png`), so sorted order is grid order; `labels.csv` and script 04
   rely on it.
2. Each image is greyscale, **square**, and masked to the **largest centred
   disk** — everything outside the inscribed circle is zeroed.
3. Each object sits entirely on its own side of the vertical midline
   `x = W // 2`, so the two never share a pixel.
4. Each full image is then rotated by a random SO(2) angle
   (`scripts/03_rotate_dataset.py`). The disk mask is what makes that rotation
   lossless.

Points 2 and 3 drive the geometry. A wide canvas with objects at the far left
and right would have its outer edges eaten by the disk mask, destroying
injectivity — two different $\theta_A$ can give the same visible image once
the distinguishing part is clipped. So the canvas is square by construction,
sized so both object frames sit inside the inscribed disk while still falling
cleanly either side of `W // 2`:

```
    +-------------------------+
    |      . - - - - - .      |   <- inscribed disk (the mask keeps this)
    |   .   +---+ +---+   .   |
    |  .    | A | | B |    .  |   A entirely left of W//2
    |  .    +---+ +---+    .  |   B entirely right of W//2
    |   .             .       |
    |      . - - - - .        |
    +-------------------------+
                 ^ split here
```

Script 02 solves for that layout and reports what fraction of object
brightness the disk mask would discard. For two 128×128 COIL frames with a
24px gap it lands on a 308×308 canvas, and the loss is 0.00%.


## Reading the result

**Three eigenvectors never show the torus.** The data are a flat torus, two
circles of fixed size, and embedding that without distortion takes four
coordinates, one cos/sin pair per circle:
$(\cos\theta_A, \sin\theta_A, \cos\theta_B, \sin\theta_B)$.

For each kernel, script 04 writes two figures from the first four
eigenvectors $\varphi_1..\varphi_4$ of Algorithm 1:

```
results/run1/<kernel>_stereo.{png,pdf}   stereographic projection of psi_1..psi_4
results/run1/<kernel>_donut.{png,pdf}    the two recovered angles drawn on a donut
```

Each is four panels: top and side view, coloured by the true $\theta_A$
(`twilight`) and by the true $\theta_B$ (a constant-lightness hue wheel). No
labels go into the figure except the colour. Two label-free checks decide the
verdict printed in the title, on screen and in `run_config.json`:

1. **Pairing (circle test).** For a cos/sin pair, $\psi_i^2 + \psi_j^2$ is the
   same for every image. Each of the six pairs of $\psi_1..\psi_4$ is scored
   by how much that sum varies (0 = perfect circle, about 0.5 = unrelated),
   and the two best disjoint pairs must both score at most 0.45.
2. **Independence (angle coverage).** On a torus the two recovered angles,
   atan2 of each pair, are independent, so the points fill the square of
   angle pairs. If both pairs follow one variable they trace a single curve.
   At least 80% of a 12 × 12 grid of the square must hold a point.

On obj87 + obj72 with the Euclidean kernel, the unrotated images pass both
(pairs $(\psi_1, \psi_3)$ and $(\psi_2, \psi_4)$, scores 0.35 and 0.34,
coverage 100%). The rotated images pass the circle test with two very clean
circles (0.09, 0.16) that both follow the random rotation, and fail on
coverage (25%). The earlier obj10 + obj12 GPU run fails the circle test.

Neither check says which variable a circle belongs to; the colouring does.
In the donut figure the shape comes from the formula, so only the colours are
evidence there. The stereographic figure's shape comes from the eigenvectors.

To rotate a figure by hand:

```bash
python3 tools/view_stereo.py                    # results/run1, min kernel
python3 tools/view_stereo.py --color B          # coloured by theta_B
python3 tools/view_stereo.py --kernel euclidean --style donut
```

For a quick check of a new pair or setting without the minimum kernel, run
script 04 with `--no-min` (Euclidean only, about 30 seconds for 5184 images).
On the *unrotated* dataset that gives the structure the minimum kernel can at
best recover.


## Files

```
run_all.sh                         steps 1-5 end to end, reproducing the published dataset
scripts/01_fetch_coil100.py        step 1: download COIL-100, keep obj87 + obj72 in data/coil-pair
scripts/02_build_dataset.py        step 2: the 72 x 72 composites
scripts/03_rotate_dataset.py       step 3: one random SO(2) rotation per image
scripts/04_spectral_embedding.py   steps 4-5: Algorithm 1 for each kernel, torus figures
tools/spectral_embedding.py        Algorithm 1 with the Euclidean, minimum and integral kernels
tools/rotations.py                 SO(2) image rotation (adapted from Roy Lederman's code)
tools/view_stereo.py               the four torus coordinates in 3D, without labels
```

```
data/coil-pair/obj87/, obj72/   the two COIL objects, 72 poses each (obj87__0.png ... obj87__355.png)
data/torus_dataset/             script 02: images/, labels.csv, layout.json, preview.png
data/rotated_torus_dataset/     script 03: images/, labels.csv, layout.json
```

`labels.csv` holds filename, `i_A`, `i_B`, `theta_A`, `theta_B`, and in the
rotated dataset also `rotation_rad` and `rotation_deg`; `layout.json` records
the geometry, the background levels removed, and the rotation seed. A results
directory holds the pickles, the stereo and donut figures and `run_config.json`
(with the circle-test verdicts).

## Requirements

`numpy`, `scipy`, `matplotlib`, `pillow` for scripts 01–03 and the tools; plus
`torch` and `torchvision` for script 04. A GPU is
not required but changes the feasible problem size by about two orders of
magnitude.

`requirements.txt` lists the packages unpinned; `requirements-lock.txt` pins
the exact versions the published dataset was built with (Python 3.14). The
images are integer PNGs, so rebuilding them is insensitive to minor version
drift, but the embeddings are floating point and are best compared under the
locked versions.

## Sources

- Vigder, Hoyos, Thong, Andén, Kileel & Moscovich, *Group Invariant Spectral Embedding*, arXiv:2607.08987, 2026. <https://arxiv.org/abs/2607.08987> — Algorithm 1 and the minimum and integral kernels (Eqs. 3, 4, 8, 9).
- Nene, Nayar & Murase, *Columbia Object Image Library (COIL-100)*, Technical Report CUCS-006-96, 1996. <https://cave.cs.columbia.edu/repository/COIL-100>
