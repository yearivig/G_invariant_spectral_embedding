#!/usr/bin/env python3
"""
Step 2: build the double-rotation image dataset.

Two rotation series are composited into one frame, object A on the left and
object B on the right, over the FULL Cartesian product of their poses. Both
angles are therefore set independently and the latent space is exactly
T^2 = S^1 x S^1, rather than the single winding trajectory you get when two
displays turn at fixed frequencies off one clock.

The layout is not arbitrary:

  1. Files are named PREFIX + zero-padded index (default "s1_00000.png"), so
     sorted() order is grid order; labels.csv and script 04 rely on it.
  2. Each image is greyscale, SQUARE, and masked here to the LARGEST CENTRED
     DISK - everything outside the inscribed circle is zeroed.
  3. Each object sits entirely on its own side of the vertical midline
     x = W // 2, so the two never share a pixel.
  4. Each full image is later rotated by a random SO(2) angle
     (scripts/03_rotate_dataset.py). The disk mask is what makes that rotation
     lossless: nothing inside the disk leaves the frame.

Points 2 and 3 drive the geometry. A wide canvas with two objects at the far
left and right would have its outer edges eaten by the disk mask, which
destroys injectivity: two different theta_A can give the same visible image
once the distinguishing part is clipped. So the canvas is square by
construction and sized so BOTH object frames sit inside the inscribed disk
while still falling cleanly either side of W // 2.

    +-------------------------+
    |      . - - - - - .      |   <- inscribed disk (the mask keeps this)
    |   .   +---+ +---+   .   |
    |  .    | A | | B |    .  |   A entirely left of W//2
    |  .    +---+ +---+    .  |   B entirely right of W//2
    |   .             .       |
    |      . - - - - .        |
    +-------------------------+
                 ^ split here

Output
------
  <out>/images/s1_00000.png ...   what script 04 reads
  <out>/labels.csv                filename, i_A, i_B, theta_A, theta_B
  <out>/layout.json               geometry + settings, for scripts 03 and 04
  <out>/preview.png               sample frames with the disk mask applied

Usage
-----
    python3 scripts/02_build_dataset.py \\
        --dir-a data/coil-pair/obj87 --dir-b data/coil-pair/obj72 \\
        --out data/torus_dataset --stride 1 --bg auto
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------- loading


def load_series(d: Path, pattern: str = r"(\d+)(?=\.png$)") -> tuple[np.ndarray, list[float]]:
    """Load one object's rotation series, sorted by the angle in each filename.

    Conversion to greyscale happens here (step 2). Compositing in greyscale
    keeps the angular-energy balancing below meaningful: on RGB it would weight
    the channels arbitrarily.
    """
    files = sorted(d.glob("*.png"), key=lambda p: int(re.findall(pattern, p.name)[-1]))
    if not files:
        raise FileNotFoundError(f"no PNGs in {d}")

    angles, frames = [], []
    for f in files:
        angles.append(float(re.findall(pattern, f.name)[-1]))
        frames.append(np.asarray(Image.open(f).convert("L"), dtype=np.float64))

    shapes = {fr.shape for fr in frames}
    if len(shapes) != 1:
        raise ValueError(f"frames in {d} are not all the same size: {shapes}")
    return np.stack(frames), angles


def border_level(stack: np.ndarray, width: int = 4) -> float:
    """Median grey level of the outer `width` pixels over every frame.

    COIL's turntable background is a dark grey (~27), not black. Everything the
    pipeline adds - canvas, disk exterior, rotation fill - is 0, so unless the
    background is removed each object frame shows up as a grey box.
    """
    edge = np.concatenate([stack[:, :width].ravel(), stack[:, -width:].ravel(),
                           stack[:, :, :width].ravel(), stack[:, :, -width:].ravel()])
    return float(np.median(edge))


def remove_background(stack: np.ndarray, bg: float) -> np.ndarray:
    """Map grey level bg -> 0 and 255 -> 255 linearly, clipping below at 0.

    The stretch keeps highlights at full brightness instead of just darkening
    the whole object by bg.
    """
    return np.clip((stack - bg) * 255.0 / (255.0 - bg), 0.0, 255.0)


# ------------------------------------------------------------- balancing


def angular_energy(stack: np.ndarray) -> float:
    """Mean squared change between consecutive poses, wrapping at 360.

    This decides each object's share of the total image variance, and so how
    the two circles' eigenvalues interleave. If object A dominates, the
    spectrum reads A^1, A^1, A^2, A^2, A^3, ... and the first B^1 sits far
    down: the embedding looks like a circle rather than a torus, and the
    natural conclusion - that the experiment failed - is wrong.
    """
    d = np.diff(stack, axis=0, append=stack[:1])
    return float(np.mean(d**2))


def rescale_variation(stack: np.ndarray, c: float) -> np.ndarray:
    """Scale each frame's deviation from the series mean by c.

    Leaves the mean appearance - size, position, overall brightness - alone
    and scales only the part that varies with angle, so angular energy scales
    by c**2 without the object visibly changing.
    """
    mu = stack.mean(axis=0, keepdims=True)
    return mu + c * (stack - mu)


# -------------------------------------------------------------- geometry


def solve_layout(obj_h: int, obj_w: int, gap: int, side: int | None = None) -> dict:
    """Pick a square canvas on which both object frames fit inside the disk.

    Objects are vertically centred and horizontally symmetric about the
    canvas centre. The binding constraint is the inner-bottom corner of each
    frame, at horizontal distance (S/2 - pad) and vertical distance obj_h/2
    from the centre:

        (S/2 - pad)^2 + (obj_h/2)^2  <=  (S/2)^2
    """

    def pad_min(S: int) -> float:
        disc = S**2 - obj_h**2
        return math.inf if disc <= 0 else (S - math.sqrt(disc)) / 2.0

    if side is None:
        S = 2 * obj_w + gap
        if S % 2:
            S += 1
        limit = 16 * (obj_w + obj_h)
        while S < limit:
            if (S - 2 * obj_w - gap) / 2.0 >= pad_min(S):
                break
            S += 2
        else:  # pragma: no cover
            raise ValueError("no canvas size satisfies the disk constraint; lower --gap")
    else:
        S = int(side)
        if S % 2:
            raise ValueError("--canvas must be even so the W//2 split is exact")

    pad = (S - 2 * obj_w - gap) / 2.0
    if pad < 0:
        raise ValueError(f"canvas {S} too small for two {obj_w}px objects with gap {gap}; "
                         f"need at least {2 * obj_w + gap}")

    pad_i = int(round(pad))
    x_a, x_b, y = pad_i, S - pad_i - obj_w, (S - obj_h) // 2
    split = S // 2
    if x_a + obj_w > split or x_b < split:
        raise ValueError(
            f"objects straddle the W//2 split at x={split} "
            f"(A spans {x_a}..{x_a + obj_w}, B spans {x_b}..{x_b + obj_w}). "
            f"Raise --canvas or lower --gap."
        )

    return {"side": S, "pad": pad_i, "gap": S - 2 * pad_i - 2 * obj_w,
            "x_a": x_a, "x_b": x_b, "y": y, "split": split,
            "pad_min_for_disk": round(pad_min(S), 2),
            "frame_corners_inside_disk": pad_i >= pad_min(S)}


def disk_mask(side: int) -> np.ndarray:
    """The largest centred disk: True inside the inscribed circle of a side x side image."""
    c = side // 2
    y, x = np.ogrid[:side, :side]
    return (x - c) ** 2 + (y - c) ** 2 <= c**2


# ------------------------------------------------------------ compositing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir-a", type=Path, required=True, help="rotation series for the LEFT object")
    p.add_argument("--dir-b", type=Path, required=True, help="rotation series for the RIGHT object")
    p.add_argument("--out", type=Path, default=Path("data/torus_dataset"))
    p.add_argument("--prefix", default="s1_",
                   help="filename prefix; must match --file-prefix on the runner")
    p.add_argument("--stride", type=int, default=3,
                   help="use every Nth pose (3 -> 24x24 = 576 images from COIL's 72)")
    p.add_argument("--stride-a", type=int, default=None, help="override stride for object A")
    p.add_argument("--stride-b", type=int, default=None, help="override stride for object B")
    p.add_argument("--gap", type=int, default=24, help="px between the two object frames")
    p.add_argument("--canvas", type=int, default=None,
                   help="force a square canvas side (even); default auto")
    p.add_argument("--resize", type=int, default=None,
                   help="resize the finished canvas; see the README before lowering this")
    p.add_argument("--bg", default="none",
                   help="subtract the object background so it matches the black canvas: "
                        "'auto' (median border level, per object), a grey level, or 'none'")
    p.add_argument("--no-balance", action="store_true", help="skip angular-energy equalisation")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="gamma on the finished canvas; != 1 breaks the exact tensor structure")
    p.add_argument("--overlap", type=int, default=0,
                   help="px to pull the objects together past the gap, creating occlusion")
    args = p.parse_args()

    A, ang_a = load_series(args.dir_a)
    B, ang_b = load_series(args.dir_b)

    sa = args.stride_a if args.stride_a is not None else args.stride
    sb = args.stride_b if args.stride_b is not None else args.stride
    A, ang_a = A[::sa], ang_a[::sa]
    B, ang_b = B[::sb], ang_b[::sb]
    print(f"object A: {len(A)} poses from {args.dir_a.name}")
    print(f"object B: {len(B)} poses from {args.dir_b.name}")

    bg_used = {}
    if args.bg != "none":
        for key, stack in (("a", A), ("b", B)):
            bg_used[key] = border_level(stack) if args.bg == "auto" else float(args.bg)
        A, B = remove_background(A, bg_used["a"]), remove_background(B, bg_used["b"])
        print(f"background      A={bg_used['a']:.1f}  B={bg_used['b']:.1f}  ->  mapped to 0")

    if not args.no_balance:
        ea, eb = angular_energy(A), angular_energy(B)
        c = math.sqrt(ea / eb)
        B = rescale_variation(B, c)
        print(f"angular energy  A={ea:.1f}  B={eb:.1f}  ->  scaled B by {c:.3f}")
    else:
        print(f"angular energy  A={angular_energy(A):.1f}  B={angular_energy(B):.1f}  (unbalanced)")

    oh, ow = A.shape[1], A.shape[2]
    if B.shape[1:] != A.shape[1:]:
        raise ValueError(f"the two series have different frame sizes: {A.shape[1:]} vs {B.shape[1:]}")

    lay = solve_layout(oh, ow, args.gap - args.overlap, args.canvas)
    S = lay["side"]
    print(f"\ncanvas {S}x{S}  pad={lay['pad']}  gap={lay['gap']}  split at x={lay['split']}")
    print(f"  A at x={lay['x_a']}..{lay['x_a'] + ow}, B at x={lay['x_b']}..{lay['x_b'] + ow}, "
          f"y={lay['y']}")
    if not lay["frame_corners_inside_disk"]:
        print(f"  NOTE: frame corners sit outside the inscribed disk "
              f"(pad {lay['pad']} < {lay['pad_min_for_disk']}); checking real content below")

    mask = disk_mask(S)
    ya = yb = lay["y"]

    # How much real object brightness would the disk mask throw away?
    probe = np.zeros((S, S))
    probe[ya:ya + oh, lay["x_a"]:lay["x_a"] + ow] = A.max(axis=0)
    sl = probe[yb:yb + oh, lay["x_b"]:lay["x_b"] + ow]
    probe[yb:yb + oh, lay["x_b"]:lay["x_b"] + ow] = np.maximum(sl, B.max(axis=0))
    total, kept = probe.sum(), probe[mask].sum()
    lost = max(0.0, 100.0 * (1.0 - kept / total)) if total else 0.0
    print(f"  disk mask discards {lost:.2f}% of object brightness "
          f"({'fine' if lost < 0.5 else 'TOO MUCH - raise --canvas or lower --gap'})")

    # ---- write ---------------------------------------------------------
    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob(f"{args.prefix}*.png"):
        old.unlink()

    rows, k = [], 0
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            canvas = np.zeros((S, S), dtype=np.float64)
            canvas[ya:ya + oh, lay["x_a"]:lay["x_a"] + ow] = a
            sl = canvas[yb:yb + oh, lay["x_b"]:lay["x_b"] + ow]
            canvas[yb:yb + oh, lay["x_b"]:lay["x_b"] + ow] = np.where(b > 0, b, sl)

            canvas = np.clip(canvas, 0, 255)
            if args.gamma != 1.0:
                canvas = 255.0 * (canvas / 255.0) ** args.gamma
            canvas[~mask] = 0.0  # apply it ourselves so the preview is honest

            img = Image.fromarray(canvas.astype(np.uint8))
            if args.resize:
                img = img.resize((args.resize, args.resize), Image.LANCZOS)

            name = f"{args.prefix}{k:05d}.png"
            img.save(img_dir / name)
            rows.append((name, i, j, ang_a[i], ang_b[j]))
            k += 1

    with open(args.out / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "i_A", "i_B", "theta_A", "theta_B"])
        w.writerows(rows)

    lay_out = dict(lay)
    lay_out.update({"n_images": len(rows), "n_a": len(A), "n_b": len(B),
                    "obj_h": oh, "obj_w": ow, "prefix": args.prefix,
                    "resize": args.resize, "gamma": args.gamma, "overlap": args.overlap,
                    "balanced": not args.no_balance, "final_side": args.resize or S,
                    "background_removed": bg_used or None,
                    "dir_a": str(args.dir_a), "dir_b": str(args.dir_b),
                    "disk_brightness_lost_pct": round(lost, 4)})
    (args.out / "layout.json").write_text(json.dumps(lay_out, indent=2))

    picks = [0, len(B) // 2, len(rows) // 2, len(rows) - 1]
    strip = np.concatenate([np.asarray(Image.open(img_dir / rows[i][0])) for i in picks], axis=1)
    Image.fromarray(strip).save(args.out / "preview.png")

    final = args.resize or S
    print(f"\nwrote {len(rows)} images ({len(A)} x {len(B)}) at {final}x{final} to {img_dir}/")
    print(f"      labels.csv, layout.json, preview.png to {args.out}/")
    print("\nnames are zero-padded, so sorted() order is raster order with theta_B fastest")


if __name__ == "__main__":
    main()
