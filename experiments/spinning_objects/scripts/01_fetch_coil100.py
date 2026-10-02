#!/usr/bin/env python3
"""
Step 1: download COIL-100 and keep only the objects this project uses: obj87 and obj72.

COIL-100 (Columbia Object Image Library) has 7200 images: 100 objects, 72
poses each at 5-degree intervals, on a motorised turntable against a dark
background, size-normalised, 128x128 colour. The archive is ~130 MB. Only the
chosen objects' frames are extracted, one folder per object, which is the
layout script 02 reads:

    data/coil-pair/obj87/obj87__0.png ... obj87__355.png   (72 frames)
    data/coil-pair/obj72/obj72__0.png ... obj72__355.png   (72 frames)

The rest of the archive is never written to disk.

  Nene, Nayar & Murase, "Columbia Object Image Library (COIL-100)",
  Technical Report CUCS-006-96, February 1996.

The CAVE URLs have moved more than once. If every mirror fails, download the
archive by hand and point this script at it:

    python3 scripts/01_fetch_coil100.py --archive ~/Downloads/coil-100.zip

Mirrors worth trying:
    https://cave.cs.columbia.edu/repository/COIL-100
    https://www.kaggle.com/datasets/jessicali9530/coil100
    https://huggingface.co/datasets/Voxel51/COIL-100

Usage:
    python3 scripts/01_fetch_coil100.py                        # obj87 + obj72 into data/coil-pair
    python3 scripts/01_fetch_coil100.py --objects 26 5 --out data/other-pair
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MIRRORS = [
    "https://cave.cs.columbia.edu/old/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip",
    "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.tar.gz",
    "https://cave.cs.columbia.edu/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip",
]
POSES = 72
FRAME = re.compile(r"(?:^|/)(obj(\d+)__\d+\.png)$")


def download(url: str, dest: Path) -> bool:
    print(f"  trying {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as fh:
            total = int(r.headers.get("Content-Length", 0))
            got = 0
            while chunk := r.read(1 << 20):
                fh.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {got / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print()
        return True
    except Exception as e:  # noqa: BLE001 - any failure means try the next mirror
        print(f"    failed: {e}")
        return False


def extract(archive: Path, objects: list[int], out: Path) -> dict[int, int]:
    """Write the chosen objects' frames to out/objNN/; return frames written per object."""
    counts = {o: 0 for o in objects}
    for o in objects:
        (out / f"obj{o}").mkdir(parents=True, exist_ok=True)

    def keep(name: str):
        m = FRAME.search(name)
        return (m.group(1), int(m.group(2))) if m and int(m.group(2)) in counts else None

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if hit := keep(name):
                    (out / f"obj{hit[1]}" / hit[0]).write_bytes(z.read(name))
                    counts[hit[1]] += 1
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            for member in t:
                if member.isfile() and (hit := keep(member.name)):
                    (out / f"obj{hit[1]}" / hit[0]).write_bytes(t.extractfile(member).read())
                    counts[hit[1]] += 1
    else:
        sys.exit(f"{archive} is neither a zip nor a tar archive")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", type=int, nargs="+", default=[87, 72], help="COIL object numbers to keep")
    ap.add_argument("--out", type=Path, default=Path("data/coil-pair"), help="one sub-folder per object")
    ap.add_argument("--archive", type=Path, default=None,
                    help="use an already-downloaded .zip/.tar.gz instead of fetching")
    args = ap.parse_args()

    have = {o: len(list((args.out / f"obj{o}").glob(f"obj{o}__*.png"))) for o in args.objects}
    if all(n == POSES for n in have.values()):
        print(f"{args.out} already holds {POSES} frames of each of " +
              ", ".join(f"obj{o}" for o in args.objects) + "; nothing to do")
        return

    if args.archive:
        counts = extract(args.archive, args.objects, args.out)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "coil.bin"
            if not any(download(u, dest) for u in MIRRORS):
                sys.exit("every mirror failed. Download by hand from one of the sources\n"
                         "in this script's docstring, then re-run with --archive.")
            counts = extract(dest, args.objects, args.out)

    for o, n in counts.items():
        print(f"  {args.out / f'obj{o}'}/  {n} frames")
    if any(n != POSES for n in counts.values()):
        sys.exit(f"expected {POSES} frames per object; check the archive before going further")


if __name__ == "__main__":
    main()
