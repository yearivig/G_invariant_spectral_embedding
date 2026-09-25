#!/usr/bin/env python3
"""
Step 3: randomly rotate every image of a script-02 dataset by its own SO(2) angle.

The input images are already greyscale, square and disk-masked, so each one is
turned about its centre as it is, with `rotations.rotate_image` (bilinear,
fill 0), from tools/rotations.py. One angle per image is drawn as

    np.random.default_rng(seed).uniform(0, 2*pi, N)

in sorted-filename order, and images are rotated one at a time, so all 5184
full-size frames never sit in memory at once.

Output mirrors the input:

  <out>/images/   same filenames, rotated
  <out>/labels.csv   input columns + rotation_rad, rotation_deg
  <out>/layout.json  input layout + rotated_from, rotation_seed, rotation

Script 04 reads this directory directly.

Usage:
    python3 scripts/03_rotate_dataset.py --dataset data/torus_dataset \\
        --out data/rotated_torus_dataset --seed 3044084360
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("data/torus_dataset"),
                    help="output of script 02")
    ap.add_argument("--out", type=Path, default=Path("data/rotated_torus_dataset"))
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the rotation angles (default: fresh, and recorded in layout.json)")
    args = ap.parse_args()

    sys.path.insert(0, str(TOOLS))
    import rotations  # noqa: E402

    rows = sorted(csv.DictReader(open(args.dataset / "labels.csv")), key=lambda r: r["filename"])
    if not rows:
        sys.exit(f"no rows in {args.dataset / 'labels.csv'}")

    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2**32)
    angles = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(rows))

    img_out = args.out / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    for old in img_out.glob("*.png"):
        old.unlink()

    for k, (r, a) in enumerate(zip(rows, angles)):
        im = np.asarray(Image.open(args.dataset / "images" / r["filename"]).convert("L"),
                        dtype=np.float64)
        out = np.clip(np.rint(rotations.rotate_image(im, a)), 0, 255).astype(np.uint8)
        Image.fromarray(out).save(img_out / r["filename"])
        r["rotation_rad"] = f"{a:.6f}"
        r["rotation_deg"] = f"{np.degrees(a):.4f}"
        if (k + 1) % 500 == 0:
            print(f"\r  {k + 1} / {len(rows)}", end="", flush=True)
    print(f"\r  {len(rows)} / {len(rows)}")

    with open(args.out / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    layout = json.loads((args.dataset / "layout.json").read_text())
    layout.update({"rotated_from": str(args.dataset), "rotation_seed": seed,
                   "rotation": "rotations.rotate_image, bilinear, cval=0, uniform [0, 2pi)"})
    (args.out / "layout.json").write_text(json.dumps(layout, indent=2))

    print(f"\nrotated {len(rows)} images into {img_out}/  (seed {seed})")
    print(f"      labels.csv, layout.json to {args.out}/")


if __name__ == "__main__":
    main()
