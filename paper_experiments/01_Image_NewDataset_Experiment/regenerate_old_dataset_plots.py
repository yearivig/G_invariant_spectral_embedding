"""
Regenerate old-dataset embedding plots for SNR=0 and SNR=1.

Creates plots for methods:
  - none (non-invariant Euclidean)
  - min
  - mean
  - bispectrum

For each embedding, saves:
  1) Legacy coloring (sin(angle)) to match prior visuals
  2) New explicit in-plane angle coloring (HSV + colorbar)
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aspire.basis import FFBBasis2D

from run_experiment import (
    add_noise,
    apply_random_rotations,
    calc_Laplacian,
    compute_eigenvectors,
    compute_gaussian_kernel_matmul,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
OUTPUT_BASE = os.path.join(BASE_DIR, "outputs")
FIG_BASE = os.path.join(BASE_DIR, "figures")

EXTERNAL_OLD_OUTPUTS = os.environ.get(
    "PAPER_IMAGE_LEGACY_OUTPUTS_DIR",
    os.path.join(REPO_ROOT, "comparing_to_yoel_and_eitan", "outputs"),
)
PROJ_PATH = os.environ.get(
    "PAPER_OLD_PROJECTIONS_PATH",
    os.path.join(REPO_ROOT, "projection_data", "unrotated", "unrotated_projections-synth.pkl"),
)
ANGLES_CSV = os.environ.get(
    "PAPER_OLD_ANGLES_CSV",
    os.path.join(REPO_ROOT, "data", "angles.csv"),
)

ELL_MAX = 10
BANDWIDTH = 0.5
SNR_LEVELS = [0, 1]
METHODS = ["none", "min", "mean", "bispectrum"]

NONE_DIR = os.path.join(OUTPUT_BASE, "old_dataset_none")
PLOTS_DIR = os.path.join(FIG_BASE, "old_dataset")

os.makedirs(NONE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def _load_old_angles_deg():
    df = pd.read_csv(ANGLES_CSV)
    df = df.drop(df.columns[0], axis=1)
    return np.array(df.values.flatten())[2:].astype(np.float64)


def _plot_embedding(v, angles_deg, title, out_path, mode):
    x = np.real(v[1])
    y = np.real(v[2])
    n = min(len(x), len(angles_deg))
    x = x[:n]
    y = y[:n]
    theta_deg = angles_deg[:n]

    # Normalize so both axes have the same scale (makes ellipses into circles)
    sx, sy = np.std(x), np.std(y)
    if sx > 1e-15:
        x = x / sx
    if sy > 1e-15:
        y = y / sy

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])

    if mode == "angle":
        theta_mod = np.mod(theta_deg, 360.0)
        ax.scatter(x, y, c=theta_mod, cmap="hsv", vmin=0.0, vmax=360.0,
                   s=12, linewidths=0)
    else:
        legacy_colors = np.sin(theta_deg * np.pi / 180.0)
        ax.scatter(x, y, c=legacy_colors, cmap="rainbow", s=12, linewidths=0)

    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Fit data to fill the square — independent limits per axis
    mx = (x.max() + x.min()) / 2
    my = (y.max() + y.min()) / 2
    span = max(x.max() - x.min(), y.max() - y.min()) / 2 * 1.1
    ax.set_xlim(mx - span, mx + span)
    ax.set_ylim(my - span, my + span)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _external_pkl_for(method, snr):
    return os.path.join(
        EXTERNAL_OLD_OUTPUTS,
        f"eigvectors_ell_max_10_tol_0.01_{method}_molecule_david_rotated_bw_0.5_snr_{snr}.pkl",
    )


def _local_none_pkl_for(snr):
    return os.path.join(NONE_DIR, f"eigvectors_ell10_none_bw0.5_snr{snr}.pkl")


def _compute_none_if_missing():
    needs_any = any(not os.path.exists(_local_none_pkl_for(snr)) for snr in SNR_LEVELS)
    if not needs_any:
        return

    with open(PROJ_PATH, "rb") as f:
        images = np.array(pickle.load(f), dtype=np.float64)

    n, h, w = images.shape
    print(f"Loaded old dataset: N={n}, HxW={h}x{w}")
    ffb = FFBBasis2D((h, w), ell_max=ELL_MAX, dtype=float)
    ang_idx = ffb.angular_indices
    sgn_idx = ffb.signs_indices

    for snr in SNR_LEVELS:
        out_pkl = _local_none_pkl_for(snr)
        if os.path.exists(out_pkl):
            print(f"[none][snr={snr}] exists, skipping compute")
            continue

        print(f"[none][snr={snr}] computing embedding")
        noisy_images = add_noise(images, snr)
        fb = ffb.expand(noisy_images, tol=1e-2)
        coefs = np.array([fb[i].asnumpy().flatten() for i in range(n)])
        coefs_rot = apply_random_rotations(coefs, ang_idx, sgn_idx)

        w_mat = compute_gaussian_kernel_matmul(coefs_rot, BANDWIDTH)
        s_mat = calc_Laplacian(w_mat)
        v, _ = compute_eigenvectors(s_mat)

        with open(out_pkl, "wb") as f:
            pickle.dump(v, f)
        print(f"  saved {out_pkl}")


def _load_embedding(method, snr):
    if method == "none":
        pkl_path = _local_none_pkl_for(snr)
    else:
        pkl_path = _external_pkl_for(method, snr)

    if not os.path.exists(pkl_path):
        print(f"Missing {method}/snr{snr}: {pkl_path}")
        return None

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def main():
    angles_deg = _load_old_angles_deg()

    # "none" method is handled by euclidean_rotation_test.py
    # (shares the same embedding for both GT-colored and random-angle-colored plots)
    methods_here = [m for m in METHODS if m != "none"]

    for snr in SNR_LEVELS:
        for method in methods_here:
            v = _load_embedding(method, snr)
            if v is None:
                continue

            base_name = f"old_dataset_{method}_bw0.5_snr{snr}"
            legacy_path = os.path.join(PLOTS_DIR, f"{base_name}.pdf")
            angle_path = os.path.join(PLOTS_DIR, f"{base_name}_angle.pdf")

            _plot_embedding(v, angles_deg, None, legacy_path, mode="legacy")
            _plot_embedding(v, angles_deg, None, angle_path, mode="angle")
            print(f"Saved {legacy_path}")
            print(f"Saved {angle_path}")

    print(f"Done. Plots are in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
