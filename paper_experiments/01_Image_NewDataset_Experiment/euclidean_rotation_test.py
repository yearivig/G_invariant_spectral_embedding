"""
Euclidean (non-invariant) embedding on randomly rotated images, colored by
the arbitrary SO(2) rotation angle applied to each image.
"""
import argparse
import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aspire.basis import FFBBasis2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import (
    add_noise,
    calc_Laplacian,
    compute_eigenvectors,
    compute_gaussian_kernel_matmul,
)

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
OLD_PROJ_PATH = os.environ.get(
    "PAPER_OLD_PROJECTIONS_PATH",
    os.path.join(REPO_ROOT, "projection_data", "unrotated", "unrotated_projections-synth.pkl"),
)
OLD_ANGLES_CSV = os.environ.get(
    "PAPER_OLD_ANGLES_CSV",
    os.path.join(REPO_ROOT, "data", "angles.csv"),
)
NEW_DATA_DIR = os.environ.get(
    "PAPER_IMAGE_DATA_DIR",
    os.path.join(REPO_ROOT, "projection_data", "new_dataset_5.3.26"),
)
NEWEST_DATA_DIR = os.environ.get(
    "PAPER_NEWEST_DATA_DIR",
    os.path.join(
        BASE_DIR,
        "newest_dataset",
        "larger_data_particle_images_no_rotation (1)",
    ),
)
# Also save "none" plots alongside the other old_dataset plots
OLD_DATASET_FIGS = os.path.join(BASE_DIR, "figures", "old_dataset")
# Save eigvecs where regenerate_old_dataset_plots.py expects them
NONE_EIGVEC_DIR = os.path.join(BASE_DIR, "outputs", "old_dataset_none")

ELL_MAX = 10
BANDWIDTH = 0.5
SNR_LEVELS = [0, 1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("old", "new", "newest"),
        default="old",
        help="Dataset to load. 'new' uses projection_data/new_dataset_5.3.26.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Optional prefix length to run; defaults to all available samples.",
    )
    return parser.parse_args()


def _torch_to_numpy_array(obj, n_samples=None):
    if isinstance(obj, list):
        items = obj[:n_samples]
        return np.array(
            [
                x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
                for x in items
            ],
            dtype=np.float64,
        )
    if n_samples is not None:
        obj = obj[:n_samples]
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().numpy()
    return np.asarray(obj, dtype=np.float64)


def _torch_angles_to_numpy(obj, n_samples=None):
    if isinstance(obj, list):
        items = obj[:n_samples]
        return np.array(
            [
                float(x.detach().cpu().item()) if hasattr(x, "detach") else float(x)
                for x in items
            ],
            dtype=np.float64,
        )
    if n_samples is not None:
        obj = obj[:n_samples]
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().numpy()
    return np.asarray(obj, dtype=np.float64)


def load_dataset(dataset, n_samples=None):
    if dataset == "old":
        print("Loading old dataset...")
        with open(OLD_PROJ_PATH, "rb") as f:
            images = np.array(pickle.load(f), dtype=np.float64)
        if n_samples is not None:
            images = images[:n_samples]

        df = pd.read_csv(OLD_ANGLES_CSV)
        df = df.drop(df.columns[0], axis=1)
        gt_angles_deg = np.array(df.values.flatten())[2:].astype(np.float64)
        gt_angles_rad = np.deg2rad(gt_angles_deg[: len(images)])
        return images, gt_angles_rad

    data_dir = NEW_DATA_DIR if dataset == "new" else NEWEST_DATA_DIR
    print(f"Loading {dataset} dataset from {data_dir}...")
    proj = torch.load(os.path.join(data_dir, "projections-synth.pt"), weights_only=False)
    angles_pt = torch.load(os.path.join(data_dir, "angles-synth.pt"), weights_only=False)
    images = _torch_to_numpy_array(proj, n_samples)
    gt_angles_rad = _torch_angles_to_numpy(angles_pt, n_samples)

    n = min(len(images), len(gt_angles_rad))
    if n < len(images) or n < len(gt_angles_rad):
        print(f"  Truncating to {n} aligned image/angle pairs")
        images = images[:n]
        gt_angles_rad = gt_angles_rad[:n]
    return images, gt_angles_rad


def make_output_dirs(dataset):
    folder = "euclidean_rotation_test"
    if dataset != "old":
        folder = f"{folder}_{dataset}_dataset"

    output_dir = os.path.join(BASE_DIR, "outputs", folder)
    figures_dir = os.path.join(BASE_DIR, "figures", folder)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    if dataset == "old":
        os.makedirs(OLD_DATASET_FIGS, exist_ok=True)
        os.makedirs(NONE_EIGVEC_DIR, exist_ok=True)
    return output_dir, figures_dir


def apply_random_rotations_with_angles(coefs, angular_indices, signs_indices):
    N = coefs.shape[0]
    ell_max = int(np.max(angular_indices))
    thetas = np.random.uniform(0, 2 * np.pi, size=N)
    rotated = coefs.copy()
    for ell in range(1, ell_max + 1):
        cos_mask = (angular_indices == ell) & (signs_indices == 1)
        sin_mask = (angular_indices == ell) & (signs_indices == -1)
        c_cos = coefs[:, cos_mask]
        c_sin = coefs[:, sin_mask]
        cos_ell = np.cos(ell * thetas)[:, None]
        sin_ell = np.sin(ell * thetas)[:, None]
        rotated[:, cos_mask] = cos_ell * c_cos - sin_ell * c_sin
        rotated[:, sin_mask] = sin_ell * c_cos + cos_ell * c_sin
    return rotated, thetas


def find_winding_number(v, angles_rad, max_k=10):
    """Find k such that k*theta best matches the embedding polar angle."""
    x = np.real(v[1])
    y = np.real(v[2])
    n = min(len(x), len(angles_rad))
    embed_angle = np.arctan2(y[:n], x[:n])
    theta = angles_rad[:n]

    best_k, best_corr = 1, 0
    for k in range(1, max_k + 1):
        # Circular correlation: |mean(exp(i*(embed - k*theta)))|
        corr = np.abs(np.mean(np.exp(1j * (embed_angle - k * theta))))
        # Also check negative (reflection)
        corr_neg = np.abs(np.mean(np.exp(1j * (embed_angle + k * theta))))
        c = max(corr, corr_neg)
        print(f"    k={k:2d}: circ_corr = {c:.4f}")
        if c > best_corr:
            best_corr = c
            best_k = k
    print(f"  Best winding number: k={best_k} (corr={best_corr:.4f})")
    return best_k


def _make_square_ax(x, y):
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    mx = (x.max() + x.min()) / 2
    my = (y.max() + y.min()) / 2
    span = max(x.max() - x.min(), y.max() - y.min()) / 2 * 1.1
    ax.set_xlim(mx - span, mx + span)
    ax.set_ylim(my - span, my + span)
    return fig, ax


def plot_2d(v, angles_rad, out_path, winding=1):
    x = np.real(v[1])
    y = np.real(v[2])
    n = min(len(x), len(angles_rad))
    x, y = x[:n], y[:n]
    # Normalize axes so circle isn't thin
    sx, sy = np.std(x), np.std(y)
    if sx > 1e-15:
        x = x / sx
    if sy > 1e-15:
        y = y / sy
    c_val = np.mod(winding * angles_rad[:n], 2 * np.pi)
    fig, ax = _make_square_ax(x, y)
    ax.scatter(x, y, c=c_val, cmap="hsv", vmin=0.0, vmax=2 * np.pi,
               s=12, linewidths=0)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_2d_gt(v, angles_deg, out_path):
    """Same embedding, colored by ground-truth bond angle (sin coloring)."""
    x = np.real(v[1])
    y = np.real(v[2])
    n = min(len(x), len(angles_deg))
    x, y = x[:n], y[:n]
    sx, sy = np.std(x), np.std(y)
    if sx > 1e-15:
        x = x / sx
    if sy > 1e-15:
        y = y / sy
    colors = np.sin(angles_deg[:n] * np.pi / 180.0)
    fig, ax = _make_square_ax(x, y)
    ax.scatter(x, y, c=colors, cmap="rainbow", s=12, linewidths=0)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    args = parse_args()
    output_dir, figures_dir = make_output_dirs(args.dataset)
    images, gt_angles_rad = load_dataset(args.dataset, args.n_samples)
    N, H, W = images.shape
    print(f"  {N} images, {H}x{W}")

    print(f"FFBBasis2D({H}x{W}, ell_max={ELL_MAX})")
    ffb = FFBBasis2D((H, W), ell_max=ELL_MAX, dtype=float)
    ang_idx = ffb.angular_indices
    sgn_idx = ffb.signs_indices

    for snr in SNR_LEVELS:
        print(f"\n{'='*60}")
        print(f"SNR = {snr} {'(no noise)' if snr == 0 else ''}")
        print(f"{'='*60}")

        noisy = add_noise(images, snr)
        print("Expanding into FB basis...")
        t0 = time.time()
        fb = ffb.expand(noisy, tol=1e-2)
        coefs = np.array([fb[i].asnumpy().flatten() for i in range(N)])
        print(f"  Done in {time.time()-t0:.1f}s, shape={coefs.shape}")

        print("Applying random SO(2) rotations...")
        coefs_rot, random_angles = apply_random_rotations_with_angles(
            coefs, ang_idx, sgn_idx
        )

        # Save the random angles
        with open(os.path.join(output_dir, f"random_angles_snr{snr}.pkl"), "wb") as f:
            pickle.dump(random_angles, f)

        print(f"Computing Euclidean kernel (bw={BANDWIDTH})...")
        W = compute_gaussian_kernel_matmul(coefs_rot, BANDWIDTH)
        S = calc_Laplacian(W)
        v, lam = compute_eigenvectors(S)

        # Save eigvecs to both locations
        with open(os.path.join(output_dir, f"eigvectors_euclidean_snr{snr}.pkl"), "wb") as f:
            pickle.dump(v, f)
        if args.dataset == "old":
            with open(os.path.join(NONE_EIGVEC_DIR, f"eigvectors_ell10_none_bw0.5_snr{snr}.pkl"), "wb") as f:
                pickle.dump(v, f)

        print("  Finding winding number...")
        k = find_winding_number(v, random_angles)

        # Plot colored by random rotation angle (with winding correction)
        plot_2d(
            v, random_angles,
            os.path.join(figures_dir, f"euclidean_snr{snr}_random_angle.pdf"),
            winding=k,
        )

        # Plot same embedding colored by the dataset ground-truth angle.
        plot_2d(
            v, gt_angles_rad,
            os.path.join(figures_dir, f"euclidean_snr{snr}_gt_angle.pdf"),
            winding=1,
        )

        # Preserve the old-dataset "none" plot side effect used by comparison code.
        if args.dataset != "old":
            continue
        plot_2d_gt(
            v, np.rad2deg(gt_angles_rad),
            os.path.join(OLD_DATASET_FIGS, f"old_dataset_none_bw0.5_snr{snr}.pdf"),
        )

    print(f"\nDone. Plots in {figures_dir}/")


if __name__ == "__main__":
    main()
