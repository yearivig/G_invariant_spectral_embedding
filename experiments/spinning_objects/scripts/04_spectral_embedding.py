#!/usr/bin/env python3
"""
Steps 4-5: spectral embedding of the pre-rotated images: kernels, eigenvectors, torus figures.

The input is a dataset whose images are ALREADY randomly rotated (e.g.
data/rotated_torus_dataset): greyscale, square, disk-masked, one SO(2)
rotation per image. Nothing is preprocessed, split or rotated here - the
PNGs are read as they are, scaled to [0, 1] as matplotlib's imread would.

Each kernel is embedded with Algorithm 1 of Group Invariant Spectral
Embedding (arXiv:2607.08987), implemented in tools/spectral_embedding.py:

  W_ij = K(x_i, x_j)   Eq. (3)      L_RW = I - D^{-1} W   Eq. (4)
  x_i -> (phi_1(x_i), ..., phi_m(x_i)), eigenvectors of L_RW for the m
  smallest nonzero eigenvalues. No diffusion time, no lambda^t scaling.

Kernels, with bandwidth epsilon:
  euclidean  exp(-||x - y||^2 / epsilon)                         always
  min        exp(-min_R ||x - R y||^2 / epsilon)       Eq. (8)   unless --no-min
  integral   mean_R exp(-||x - R y||^2 / epsilon)      Eq. (9)   with --integral
R runs over --num-group-elements equally spaced rotations (default 300).

--epsilon sets the bandwidth for every kernel (the paper fixes it per
experiment). The default "knn" sets it per kernel to (mean distance to the
--num-neighbors nearest neighbours)^2, computed from that kernel's own
distances (the integral kernel uses the SO(2)-minimum distances). The value
used is recorded in run_config.json.

Outputs in --out:
  <name>_eigenvectors.pkl   (m, n): row k-1 is phi_k
  <name>_eigenvalues.pkl    (m + 1,): lambda_0 = 0 <= lambda_1 <= ... of L_RW
  so2_min_sq_distances.npy  (n, n) min_R ||x_i - R x_j||^2, the slow step, kept
                            so it never has to be recomputed (--min-distances)
  so2_min_sq_distances.json what produced it: dataset labels, n, group elements
  <kernel>_stereo / <kernel>_donut .{png,pdf}   torus figures from phi_1..phi_4
  true_theta_A.pkl, true_theta_B.pkl, rotation_angles.pkl (radians), run_config.json
with <name> = euclidean, so2_min, so2_integral.

Figures: the data are a flat torus, which needs four coordinates, so no plot
of three eigenvectors can show it. tools/view_stereo.py pairs phi_1..phi_4
into two cos/sin pairs by a label-free circle test, checks that the two
recovered angles are independent, and draws a stereographic projection and a
donut. The true angles theta_A (left object) and theta_B (right object) from
labels.csv are used only for colour.

Usage
-----
    python3 scripts/04_spectral_embedding.py --dataset data/rotated_torus_dataset --out results/run1
    python3 scripts/04_spectral_embedding.py ... --no-min          # Euclidean only, ~1 minute
    python3 scripts/04_spectral_embedding.py ... --epsilon 0.005   # fixed bandwidth

Adding the integral kernel later, without recomputing the minimum kernel:

    python3 scripts/04_spectral_embedding.py --dataset data/rotated_torus_dataset --out results/run1 \
        --integral --min-distances results/run1/so2_min_sq_distances.npy

--min-distances reuses the saved minimum distances: the minimum kernel is
re-embedded from them in seconds, and the integral kernel's "knn" epsilon comes
from them exactly as in the first run. The file is refused unless it was made
from the same dataset (same labels.csv), number of images and number of group
elements. run_config.json is updated, not replaced, so earlier kernels'
entries are kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("double_rotation")

ROOT = Path(__file__).resolve().parent.parent
FILE_NAMES = {"euclidean": "euclidean", "min": "so2_min", "integral": "so2_integral"}
TITLES = {"euclidean": "Euclidean kernel", "min": "SO(2) minimum kernel", "integral": "SO(2) integral kernel"}


def save(directory: Path, name: str, obj) -> None:
    with open(directory / name, "wb") as fh:
        pickle.dump(obj, fh)
    logger.info(f"  saved {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("data/rotated_torus_dataset"),
                    help="dataset of already-rotated images (images/, labels.csv, layout.json)")
    ap.add_argument("--out", type=Path, default=Path("results/run1"))
    ap.add_argument("--n-images", type=int, default=None, help="default: all of them")
    ap.add_argument("--m", type=int, default=20, help="number of eigenvectors phi_1..phi_m to keep")
    ap.add_argument("--epsilon", default="knn",
                    help="kernel bandwidth for every kernel, or 'knn' (default) to set it per kernel")
    ap.add_argument("--num-neighbors", type=int, default=20, help="k for the 'knn' bandwidth rule")
    ap.add_argument("--num-group-elements", "--num-rotations", dest="num_group_elements", type=int,
                    default=300, help="SO(2) elements for the minimum and integral kernels (default 300)")
    ap.add_argument("--integral", action="store_true", help="also compute the SO(2) integral kernel")
    ap.add_argument("--no-min", action="store_true",
                    help="skip the SO(2) minimum kernel: Euclidean only (plus --integral if given)")
    ap.add_argument("--min-distances", type=Path, default=None,
                    help="reuse a saved so2_min_sq_distances.npy instead of recomputing the minimum distances")
    ap.add_argument("--device", default=None, help="torch device, e.g. cuda:0 (default: auto)")
    ap.add_argument("--dry-run", action="store_true", help="validate and estimate cost, then stop")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    epsilon = args.epsilon if args.epsilon == "knn" else float(args.epsilon)

    sys.path.insert(0, str(ROOT / "tools"))
    import spectral_embedding  # noqa: E402
    import view_stereo  # noqa: E402

    # ---- validate the dataset -----------------------------------------
    img_dir = args.dataset / "images"
    layout_path = args.dataset / "layout.json"
    if not layout_path.is_file():
        sys.exit(f"{layout_path} is missing - is {args.dataset} a rotated dataset?")
    layout = json.loads(layout_path.read_text())
    prefix = layout["prefix"]

    available = len(list(img_dir.glob(f"{prefix}*.png")))
    if available == 0:
        sys.exit(f"no {prefix}*.png files in {img_dir}")
    n = min(args.n_images or available, available)
    if args.m >= n:
        sys.exit(f"--m ({args.m}) must be smaller than the number of images ({n})")

    # What produced a minimum-distance matrix: it may only be reused for the same data and settings.
    labels_sha = hashlib.sha256((args.dataset / "labels.csv").read_bytes()).hexdigest()
    provenance = {"dataset": str(args.dataset.resolve()), "labels_sha256": labels_sha, "n_images": n,
                  "num_group_elements": args.num_group_elements, "rotation_seed": layout.get("rotation_seed")}
    min_d2 = None
    if args.min_distances is not None:
        meta_path = args.min_distances.with_suffix(".json")
        if not args.min_distances.is_file() or not meta_path.is_file():
            sys.exit(f"--min-distances: need {args.min_distances} and {meta_path}")
        meta = json.loads(meta_path.read_text())
        for key in ("labels_sha256", "n_images", "num_group_elements"):
            if meta.get(key) != provenance[key]:
                sys.exit(f"--min-distances: {args.min_distances} was made with {key} = {meta.get(key)}, "
                         f"this run has {provenance[key]}. Recompute it (drop --min-distances) or match the settings.")
        min_d2 = np.load(args.min_distances)
        if min_d2.shape != (n, n):
            sys.exit(f"--min-distances: shape {min_d2.shape}, expected {(n, n)}")
        print(f"reusing  {args.min_distances} (made from {meta['dataset']})")

    side = layout["final_side"]
    need_min_pass = min_d2 is None and (not args.no_min or (args.integral and epsilon == "knn"))
    passes = (1 if need_min_pass else 0) + (1 if args.integral else 0)
    work = (n * n / 2) * args.num_group_elements * side * side * passes
    print(f"dataset  {img_dir}  ({available} images, {side}x{side}, prefix '{prefix}')")
    print(f"grid     {layout['n_a']} x {layout['n_b']} = {layout['n_images']}")
    print(f"using    n={n}  m={args.m}  epsilon={args.epsilon}  "
          f"{'k=' + str(args.num_neighbors) + '  ' if epsilon == 'knn' else ''}"
          f"group elements={args.num_group_elements}")
    print(f"\nSO(2) kernel work ~ {work:.2e} element-ops")
    print("  a GPU does ~1e10-1e11 of these per second; a CPU core, ~1e8-1e9.")
    if args.dry_run:
        print("\ndry run - stopping here")
        return

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- load the rotated images and their labels ---------------------
    # Sorted by filename, as labels.csv is below, so row i is image i.
    logger.info(f"[1/3] loading {n} rotated images")
    files = sorted(img_dir.glob(f"{prefix}*.png"))[:n]
    images = np.stack([np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0 for f in files])
    rows = sorted(csv.DictReader(open(args.dataset / "labels.csv")), key=lambda r: r["filename"])[:n]
    if [r["filename"] for r in rows] != [f.name for f in files]:
        sys.exit("labels.csv does not list the same files as images/")
    theta_A = np.radians([float(r["theta_A"]) for r in rows])
    theta_B = np.radians([float(r["theta_B"]) for r in rows])
    save(out, "true_theta_A.pkl", theta_A)
    save(out, "true_theta_B.pkl", theta_B)
    if "rotation_rad" in rows[0]:
        save(out, "rotation_angles.pkl", np.array([float(r["rotation_rad"]) for r in rows]))

    # ---- Algorithm 1 for each kernel ------------------------------------
    logger.info("[2/3] spectral embedding (Algorithm 1)")
    kernels = ["euclidean"] + ([] if args.no_min else ["min"]) + (["integral"] if args.integral else [])
    results = {}
    for kernel in kernels:
        t0 = time.time()
        reuse = min_d2 if kernel in ("min", "integral") else None
        res = spectral_embedding.embed(kernel, images.reshape(n, -1) if kernel == "euclidean" else images, args.m,
                             epsilon=epsilon, num_group_elements=args.num_group_elements,
                             num_neighbors=args.num_neighbors, device=args.device, sq_dists=reuse)
        logger.info(f"  {kernel}: done in {time.time() - t0:.1f}s, epsilon = {res['epsilon']:.6g}")
        if kernel == "min" or (kernel == "integral" and min_d2 is None and res["sq_dists"] is not None):
            computed = min_d2 is None
            min_d2 = res["sq_dists"]
            target = out / "so2_min_sq_distances.npy"
            if computed or args.min_distances.resolve() != target.resolve():
                np.save(target, min_d2)
                target.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
        name = FILE_NAMES[kernel]
        save(out, f"{name}_eigenvectors.pkl", res["phi"].T)
        save(out, f"{name}_eigenvalues.pkl", res["eigenvalues"])
        results[kernel] = res

    # ---- torus figures --------------------------------------------------
    logger.info("[3/3] torus figures from phi_1..phi_4")
    import matplotlib
    matplotlib.use("Agg")
    summary = {}
    for kernel, res in results.items():
        U4 = res["phi"][:, :4].T
        title = f"{TITLES[kernel]} on the rotated images ({args.dataset})"
        for style in ("stereo", "donut"):
            test = view_stereo.save_figure(U4, theta_A, theta_B, title, style, out / f"{kernel}_{style}")
        view_stereo.report(TITLES[kernel], test)
        if args.epsilon != "knn":
            eps_source = "fixed (--epsilon)"
        elif kernel == "euclidean":
            eps_source = "knn on Euclidean distances"
        else:
            eps_source = "knn on SO(2)-minimum distances" + (f" from {args.min_distances}" if args.min_distances else "")
        summary[kernel] = {"epsilon": res["epsilon"], "epsilon_source": eps_source, "eigenvalues_1_4": [round(float(x), 6) for x in res["eigenvalues"][1:5]],
                           "pairs": [[i + 1, j + 1] for _, i, j in test["pairs"]],
                           "scores": [round(s, 4) for s, _, _ in test["pairs"]],
                           "coverage": round(test["coverage"], 4), "two_circles": test["circles"],
                           "torus": test["ok"], "verdict": test["verdict"]}

    # Update, not replace: a later run that adds a kernel keeps the earlier kernels' entries.
    cfg_path = out / "run_config.json"
    kept = {}
    if cfg_path.is_file():
        old = json.loads(cfg_path.read_text())
        if old.get("labels_sha256") == labels_sha and old.get("n_images") == n:
            kept = old.get("kernels", {})
    (cfg_path).write_text(json.dumps({
        "algorithm": "Algorithm 1 (spectral embedding, L_RW = I - D^-1 W), arXiv:2607.08987",
        "dataset": str(args.dataset), "labels_sha256": labels_sha, "n_images": n, "m": args.m,
        "epsilon_rule": args.epsilon, "num_neighbors": args.num_neighbors,
        "num_group_elements": args.num_group_elements, "prefix": prefix,
        "kernels": {**kept, **summary},
    }, indent=2))

    print(f"\ndone. results in {out}/")
    print(f"  kernels: {', '.join(results)}")
    print(f"  figures: <kernel>_stereo.{{png,pdf}} and <kernel>_donut.{{png,pdf}}")
    print(f"  rotate them:  python3 tools/view_stereo.py --run {out} [--kernel euclidean] [--style donut]")


if __name__ == "__main__":
    main()
