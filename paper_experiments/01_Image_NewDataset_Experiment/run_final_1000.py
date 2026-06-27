"""
Run final experiment on the NEWEST dataset (larger_data_particle_images_no_rotation),
using only the first 1000 samples.

Scatter colors use ground-truth angles from ``output_main.csv`` (column ``angle``),
aligned by ``frame`` index with the loaded projections.

SNR sweep matches ``figures/old_dataset`` / ``regenerate_old_dataset_plots.py``:
Gaussian noise at 0 dB and 1 dB (``add_noise`` from ``run_experiment``). SNR 0
outputs stay in ``outputs/final_1000_newest/`` (flat layout); SNR > 0 use
``outputs/final_1000_newest/snr{snr}/`` so cached bispectrum/orbit runs remain valid.
"""
import os
import pickle
import time
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aspire.basis import FFBBasis2D
from run_experiment import (
    compute_orbit_kernel, compute_gaussian_kernel_matmul,
    compute_bispectrum_all, calc_Laplacian, compute_eigenvectors, add_noise,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "PAPER_NEWEST_DATA_DIR",
    os.path.join(
        _THIS_DIR,
        "newest_dataset",
        "larger_data_particle_images_no_rotation (1)",
    ),
)
CSV_GT_PATH = os.environ.get(
    "PAPER_NEWEST_GT_CSV",
    os.path.join(DATA_DIR, "output_main.csv"),
)
OUTPUT_BASE = "outputs/final_1000_newest"
FIGURES_BASE = "figures/final_1000_newest"
# Same SNR list as regenerate_old_dataset_plots.py (figures/old_dataset).
_raw_snr = os.environ.get("PAPER_NEWEST_SNR_LEVELS", "0,1")
SNR_LEVELS = [int(x.strip()) for x in _raw_snr.split(",") if x.strip()]
ELL_MAX = 10
NUM_G = 600
BISP_WORKERS = 64
N_SAMPLES = 1000

ORBIT_BWS = [0.005, 0.01, 0.025, 0.05]
BISP_BWS = [6e-9, 1.5e-8, 3e-8, 6e-8]

FB_GAUSS_BW = 0.01
SCATTER_CMAP = "hsv"
POINT_SIZE = 30.0
GRID_POINT_SIZE = 30.0
FIGURES_DPI = 600
EUCLIDEAN_TAG = f"euclidean_random_angle_bw{FB_GAUSS_BW}"
EUCLIDEAN_GT_FIG_TAG = f"euclidean_no_random_angle_bw{FB_GAUSS_BW}"
EUCLIDEAN_RANDOM_FIG_TAG = f"euclidean_random_angle_bw{FB_GAUSS_BW}"
EUCLIDEAN_FIGURE_ALIASES = {
    EUCLIDEAN_GT_FIG_TAG: [f"fbcoef_gaussian_bw{FB_GAUSS_BW}"],
}
print("Loading data...")
proj = torch.load(f"{DATA_DIR}/projections-synth.pt", weights_only=False)
angles_pt = torch.load(f"{DATA_DIR}/angles-synth.pt", weights_only=False)
if isinstance(proj, list):
    images = np.array([p.numpy() for p in proj[:N_SAMPLES]], dtype=np.float64)
else:
    images = proj[:N_SAMPLES].numpy().astype(np.float64)
if isinstance(angles_pt, list):
    gt_angles = np.array([a.item() for a in angles_pt[:N_SAMPLES]], dtype=np.float64)
else:
    gt_angles = angles_pt[:N_SAMPLES].numpy().astype(np.float64)

N, H, W = images.shape
print(f"  {N} images of {H}x{W} (first {N_SAMPLES} of dataset)")


def load_gt_angles_from_csv(csv_path, n_samples):
    """Map frame -> angle from output_main.csv; require frames 0..n_samples-1."""
    by_frame = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "frame" not in reader.fieldnames or "angle" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must have columns 'frame' and 'angle', got {reader.fieldnames!r}"
            )
        for row in reader:
            by_frame[int(row["frame"])] = float(row["angle"])
    out = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        if i not in by_frame:
            raise KeyError(f"CSV missing frame {i} (need 0..{n_samples - 1})")
        out[i] = by_frame[i]
    return out


print(f"Loading ground-truth angles for coloring from {CSV_GT_PATH}")
colors = load_gt_angles_from_csv(CSV_GT_PATH, N)
print(
    f"  CSV angle range for scatter coloring: "
    f"[{float(colors.min()):.4f}, {float(colors.max()):.4f}]"
)

# --- Fourier–Bessel basis (shared across SNR; expansion is per SNR) ---
print("\nBuilding FFBBasis2D...")
ffb = FFBBasis2D((H, W), ell_max=ELL_MAX, dtype=float)
ang_idx = ffb.angular_indices
sgn_idx = ffb.signs_indices


def load_eig_pack(pkl_path):
    """Support legacy pickles (ndarray only) and dict packs with eigenvalues."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "eigvecs" in obj:
        return obj["eigvecs"], obj.get("eigenvalues")
    return obj, None


def save_eig_pack(pkl_path, eigvecs, eigenvalues):
    with open(pkl_path, "wb") as f:
        pickle.dump({"eigvecs": eigvecs, "eigenvalues": eigenvalues}, f)


def r2_linear(y, x1, x2):
    """R² for predicting y from [1, x1, x2]."""
    n = min(len(y), len(x1), len(x2))
    y = y[:n].astype(np.float64)
    x1 = x1[:n].astype(np.float64)
    x2 = x2[:n].astype(np.float64)
    X = np.column_stack([np.ones(n), x1, x2])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < 3 or n < 3:
        return float("nan")
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def pearson(a, b):
    n = min(len(a), len(b))
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    if n < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def scatter_colors_from_angles_deg(angles_deg, n):
    """Angle-preserving colors using the point-cloud rainbow palette."""
    return np.mod(np.asarray(angles_deg[:n], dtype=np.float64), 360.0)


def scatter_colors_from_angles_rad(angles_rad, n):
    """Angle-preserving colors for applied SO(2) random rotations."""
    return np.mod(np.asarray(angles_rad[:n], dtype=np.float64), 2 * np.pi)


def should_flip_mean_x_axis(tag_or_label):
    """Reflect mean-kernel result plots/exports across the x-axis."""
    text = str(tag_or_label)
    return text.startswith("mean_bw") or text.startswith("mean\n")


def apply_random_rotations_with_angles(coefs, angular_indices, signs_indices):
    """Apply random SO(2) rotations in FB coefficient space and keep angles."""
    n = coefs.shape[0]
    ell_max = int(np.max(angular_indices))
    thetas = np.random.uniform(0, 2 * np.pi, size=n)
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


def write_embedding_csv(tag, v, csv_angles, inplane_angles, tabular_dir, snr_db):
    """Per-configuration diffusion coordinates + ground truth (CSV + in-plane)."""
    n = min(v.shape[1], len(csv_angles), len(inplane_angles))
    p1 = np.real(v[1, :n])
    p2 = np.real(v[2, :n])
    if should_flip_mean_x_axis(tag):
        p2 = -p2
    p3 = np.real(v[3, :n]) if v.shape[0] > 3 else np.full(n, np.nan)
    path = os.path.join(tabular_dir, f"embedding_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["frame", "psi1", "psi2", "psi3", "csv_gt_angle", "inplane_angle_rad"]
        )
        for i in range(n):
            w.writerow([i, p1[i], p2[i], p3[i], csv_angles[i], inplane_angles[i]])
    print(f"    Table: {path}")
    return {
        "snr_dB": snr_db,
        "tag": tag,
        "r2_csv_vs_psi12": r2_linear(csv_angles[:n], p1, p2),
        "pearson_csv_psi1": pearson(csv_angles, p1),
        "pearson_csv_psi2": pearson(csv_angles, p2),
        "pearson_inplane_psi1": pearson(inplane_angles, p1),
        "pearson_inplane_psi2": pearson(inplane_angles, p2),
    }


def save_fig(v, fname, figures_dir, colors_arr, color_units="deg"):
    x = np.real(v[1])
    y = np.real(v[2])
    n = min(len(x), len(colors_arr))
    x, y = x[:n], y[:n]
    if should_flip_mean_x_axis(fname):
        y = -y
    if color_units == "rad":
        color_values = scatter_colors_from_angles_rad(colors_arr, n)
        vmin, vmax = 0.0, 2 * np.pi
    else:
        color_values = scatter_colors_from_angles_deg(colors_arr, n)
        vmin, vmax = 0.0, 360.0
    sx, sy = np.std(x), np.std(y)
    if sx > 1e-15:
        x = x / sx
    if sy > 1e-15:
        y = y / sy
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.scatter(
        x, y, c=color_values, cmap=SCATTER_CMAP,
        vmin=vmin, vmax=vmax, marker="o", s=POINT_SIZE, linewidths=0,
    )
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    mx = (x.max() + x.min()) / 2
    my = (y.max() + y.min()) / 2
    span = max(x.max() - x.min(), y.max() - y.min()) / 2 * 1.1
    ax.set_xlim(mx - span, mx + span)
    ax.set_ylim(my - span, my + span)
    path = os.path.join(figures_dir, f"{fname}.pdf")
    fig.savefig(path, dpi=FIGURES_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


all_snr_summary_rows = []

for snr in SNR_LEVELS:
    snr_sub = "" if snr == 0 else f"snr{snr}"
    output_dir = os.path.join(OUTPUT_BASE, snr_sub) if snr_sub else OUTPUT_BASE
    figures_dir = os.path.join(FIGURES_BASE, snr_sub) if snr_sub else FIGURES_BASE
    tabular_dir = os.path.join(output_dir, "tabular")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tabular_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"SNR = {snr} dB {'(no noise)' if snr == 0 else ''}")
    print(f"  outputs → {output_dir}")
    print(f"  figures → {figures_dir}")
    print("=" * 70)

    summary_rows = []
    orbit_tags = [f"{method}_bw{bw}" for method in ["min", "mean"] for bw in ORBIT_BWS]
    bisp_tags = [f"bispectrum_bw{bw:.1e}" for bw in BISP_BWS]
    euclidean_pkl = f"{output_dir}/eigvectors_{EUCLIDEAN_TAG}.pkl"
    euclidean_angles_pkl = f"{output_dir}/random_angles_{EUCLIDEAN_TAG}.pkl"
    need_orbit_compute = any(
        not os.path.exists(f"{output_dir}/eigvectors_{tag}.pkl") for tag in orbit_tags
    )
    need_bisp_compute = any(
        not os.path.exists(f"{output_dir}/eigvectors_{tag}.pkl") for tag in bisp_tags
    )
    need_euclidean_compute = any(
        not os.path.exists(path) for path in [euclidean_pkl, euclidean_angles_pkl]
    )

    coefs = None
    coefs_rot = None
    random_angles = None
    if need_orbit_compute or need_bisp_compute or need_euclidean_compute:
        print("\nFFB expansion...")
        t0 = time.time()
        noisy_images = add_noise(images, snr)
        fb = ffb.expand(noisy_images, tol=1e-2)
        print(f"  Done in {time.time() - t0:.1f}s")

        coefs = np.array([fb[i].asnumpy().flatten() for i in range(N)])
        print("Applying random rotations...")
        coefs_rot, random_angles = apply_random_rotations_with_angles(coefs, ang_idx, sgn_idx)
    else:
        print("\nAll embedding pickles already exist; regenerating figures and CSVs only.")

    # --- Orbit methods (min, mean) across bandwidth sweep ---
    for method in ["min", "mean"]:
        for bw in ORBIT_BWS:
            tag = f"{method}_bw{bw}"
            pkl = f"{output_dir}/eigvectors_{tag}.pkl"
            if os.path.exists(pkl):
                print(f"  [{tag}] exists, skipping computation")
                v, lam = load_eig_pack(pkl)
            else:
                print(f"\n  [{tag}] Computing orbit kernel...")
                t0m = time.time()
                W = compute_orbit_kernel(
                    coefs_rot, ang_idx, sgn_idx, bw, NUM_G, method
                )
                S = calc_Laplacian(W)
                v, lam = compute_eigenvectors(S)
                print(f"    Done in {time.time() - t0m:.1f}s")
                save_eig_pack(pkl, v, lam)
                if lam is not None:
                    np.savez_compressed(
                        os.path.join(tabular_dir, f"spectrum_{tag}.npz"),
                        eigenvalues=np.asarray(lam),
                    )

            save_fig(v, tag, figures_dir, colors)
            summary_rows.append(
                write_embedding_csv(
                    tag, v, colors, gt_angles, tabular_dir, snr
                )
            )

    # --- Bispectrum across bandwidth sweep ---
    bisp_pkl = f"{output_dir}/bispectrum_features.pkl"
    if not need_bisp_compute:
        bisp = None
    elif os.path.exists(bisp_pkl):
        print("\n  Loading cached bispectrum features...")
        with open(bisp_pkl, "rb") as f:
            bisp = pickle.load(f)
    else:
        print("\n  Computing bispectrum features (parallel)...")
        t0b = time.time()
        bisp = compute_bispectrum_all(coefs_rot, H, ELL_MAX, BISP_WORKERS)
        print(f"    Done in {time.time() - t0b:.1f}s, shape={bisp.shape}")
        with open(bisp_pkl, "wb") as f:
            pickle.dump(bisp, f)

    for bw in BISP_BWS:
        tag = f"bispectrum_bw{bw:.1e}"
        pkl = f"{output_dir}/eigvectors_{tag}.pkl"
        if os.path.exists(pkl):
            print(f"  [{tag}] exists, skipping")
            v, lam = load_eig_pack(pkl)
        else:
            print(f"\n  [{tag}] Computing Gaussian kernel...")
            t0k = time.time()
            W = compute_gaussian_kernel_matmul(bisp, bw)
            S = calc_Laplacian(W)
            v, lam = compute_eigenvectors(S)
            print(f"    Done in {time.time() - t0k:.1f}s")
            save_eig_pack(pkl, v, lam)
            if lam is not None:
                np.savez_compressed(
                    os.path.join(tabular_dir, f"spectrum_{tag}.npz"),
                    eigenvalues=np.asarray(lam),
                )

        save_fig(v, tag, figures_dir, colors)
        summary_rows.append(
            write_embedding_csv(tag, v, colors, gt_angles, tabular_dir, snr)
        )

    # --- Baseline: Euclidean Gaussian kernel on randomly rotated FB coefficients ---
    print(
        f"\n  [{EUCLIDEAN_TAG}] Gaussian kernel on random-angle rotated FB coefficients..."
    )
    if os.path.exists(euclidean_pkl) and os.path.exists(euclidean_angles_pkl):
        print(f"  [{EUCLIDEAN_TAG}] exists, skipping computation")
        v_fb, lam_fb = load_eig_pack(euclidean_pkl)
        with open(euclidean_angles_pkl, "rb") as f:
            random_angles = pickle.load(f)
    else:
        if coefs_rot is None or random_angles is None:
            raise RuntimeError("Missing rotated FB coefficients for Euclidean baseline")
        t0f = time.time()
        W_fb = compute_gaussian_kernel_matmul(
            np.asarray(coefs_rot, dtype=np.float64), FB_GAUSS_BW
        )
        S_fb = calc_Laplacian(W_fb)
        v_fb, lam_fb = compute_eigenvectors(S_fb)
        print(f"    Done in {time.time() - t0f:.1f}s")
        save_eig_pack(euclidean_pkl, v_fb, lam_fb)
        with open(euclidean_angles_pkl, "wb") as f:
            pickle.dump(random_angles, f)
        if lam_fb is not None:
            np.savez_compressed(
                os.path.join(tabular_dir, f"spectrum_{EUCLIDEAN_TAG}.npz"),
                eigenvalues=np.asarray(lam_fb),
            )

    save_fig(v_fb, EUCLIDEAN_GT_FIG_TAG, figures_dir, colors)
    for alias_tag in EUCLIDEAN_FIGURE_ALIASES.get(EUCLIDEAN_GT_FIG_TAG, []):
        save_fig(v_fb, alias_tag, figures_dir, colors)
    save_fig(v_fb, EUCLIDEAN_RANDOM_FIG_TAG, figures_dir, random_angles, color_units="rad")
    summary_rows.append(
        write_embedding_csv(EUCLIDEAN_TAG, v_fb, colors, gt_angles, tabular_dir, snr)
    )

    # --- Summary comparison grid ---
    print("\nGenerating comparison grid...")
    all_results = []
    if os.path.exists(euclidean_pkl) and os.path.exists(euclidean_angles_pkl):
        v_fb_grid, _ = load_eig_pack(euclidean_pkl)
        with open(euclidean_angles_pkl, "rb") as f:
            random_angles_grid = pickle.load(f)
        all_results.append((f"Euclidean\nGT angle\nbw={FB_GAUSS_BW}", v_fb_grid, colors, "deg"))
        all_results.append(
            (f"Euclidean\nrandom angle\nbw={FB_GAUSS_BW}", v_fb_grid, random_angles_grid, "rad")
        )

    for method in ["min", "mean"]:
        for bw in ORBIT_BWS:
            pkl = f"{output_dir}/eigvectors_{method}_bw{bw}.pkl"
            if os.path.exists(pkl):
                v, _ = load_eig_pack(pkl)
                all_results.append((f"{method}\nbw={bw}", v, colors, "deg"))

    for bw in BISP_BWS:
        tag = f"bispectrum_bw{bw:.1e}"
        pkl = f"{output_dir}/eigvectors_{tag}.pkl"
        if os.path.exists(pkl):
            v, _ = load_eig_pack(pkl)
            all_results.append((f"bispectrum\nbw={bw:.1e}", v, colors, "deg"))

    ncols = 4
    nrows = (len(all_results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.atleast_2d(axes)
    for idx, (label, v, color_arr, color_units) in enumerate(all_results):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        x = np.real(v[1])
        y = np.real(v[2])
        npt = min(len(x), len(color_arr))
        x, y = x[:npt], y[:npt]
        if should_flip_mean_x_axis(label):
            y = -y
        if color_units == "rad":
            color_values = scatter_colors_from_angles_rad(color_arr, npt)
            vmin, vmax = 0.0, 2 * np.pi
        else:
            color_values = scatter_colors_from_angles_deg(color_arr, npt)
            vmin, vmax = 0.0, 360.0
        sx, sy = np.std(x), np.std(y)
        if sx > 1e-15:
            x = x / sx
        if sy > 1e-15:
            y = y / sy
        ax.scatter(
            x, y, c=color_values, cmap=SCATTER_CMAP,
            vmin=vmin, vmax=vmax, marker="o", s=GRID_POINT_SIZE, linewidths=0,
        )
        ax.set_title(label, fontsize=9)
        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    for idx in range(len(all_results), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)
    plt.tight_layout()
    grid_name = (
        "newest_1000_comparison_grid.pdf"
        if snr == 0
        else f"newest_1000_comparison_grid_snr{snr}.pdf"
    )
    grid_path = os.path.join(figures_dir, grid_name)
    fig.savefig(grid_path, dpi=FIGURES_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid: {grid_path}")

    summary_path = os.path.join(tabular_dir, "results_summary.csv")
    if summary_rows:
        fields = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"  Metrics summary: {summary_path}")
    all_snr_summary_rows.extend(summary_rows)

    with open(f"{output_dir}/ground_truth_angles.pkl", "wb") as f:
        pickle.dump(gt_angles, f)

# --- Combined summary across SNR ---
if len(SNR_LEVELS) > 1 and all_snr_summary_rows:
    merged = os.path.join(OUTPUT_BASE, "results_summary_all_snr.csv")
    fields = list(all_snr_summary_rows[0].keys())
    with open(merged, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_snr_summary_rows)
    print(f"\nCombined metrics (all SNR): {merged}")

print("\n" + "=" * 60)
print(f"ALL DONE — SNR levels: {SNR_LEVELS}")
print(f"  figures under {FIGURES_BASE}/")
print(f"  tables under {OUTPUT_BASE}/ and snr*/ subfolders")
print("=" * 60)
