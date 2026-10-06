"""
SO(3)/SO(2) single-point experiment with paired sampling.

Geometry and notation
---------------------
- M = SO(3) with the Riemannian metric induced from the Frobenius inner product
  on R^{3x3} (i.e. <A,B> = tr(A^T B) on tangent spaces). This is twice the
  standard bi-invariant metric.
- G = SO(2) rotations about the z-axis; right action: R -> R * Rz(theta).
- The homogeneous space is N = M/G ~= S^2.

Ground-truth eigenfunction and Laplacian
----------------------------------------
- Wigner D-entries are eigenfunctions of the negative Laplace-Beltrami on M.
- Under the induced Frobenius metric, the Laplace-Beltrami is half the
  bi-invariant Laplace-Beltrami, so
  f_ell(R) = D^ell_{00}(R) = P_ell(R[2,2]) is right G-invariant (depends only on
  the polar angle) and satisfies Delta f_ell = -ell(ell+1)/2 f_ell. We take ell=1.

Kernels (right G-action on the target R0)
-----------------------------------------
1) Euclidean / chordal kernel in the ambient R^9:
   K_E(R, R0) = exp(-||R - R0||_F^2 / eps)
2) Minimum-orbit kernel:
   K_min^G(R, R0) = exp(-min_{g in G} ||R - R0 g||_F^2 / eps)
3) G-integral (Haar) kernel:
   K_H^G(R, R0) = E_{g in G}[exp(-||R - R0 g||_F^2 / eps)]

Estimator (row-normalized graph Laplacian at R0)
-------------------------------------------------
  L_hat f(R0) = f(R0) - sum_i w_i f(R_i) / sum_i w_i
  Delta_hat_eps f(R0) = (4/eps) * L_hat f(R0)

We compare (4/eps) * L_hat f(R0) - ell(ell+1)/2 to zero and report log-log
behaviour vs eps with a small-eps fitted slope.

Paired sampling
---------------
Rotations (and, for G-invariant kernels, SO(2) subgroup samples) are drawn
ONCE per trial and reused across all eps. The pointwise 95% CI band on the
mean |error| remains valid; trial errors at different eps are correlated by
construction, which is what makes the mean a smooth function of eps.

Outputs
-------
- Pickle:  so3_so2_<function>_results.pkl
- Plots:   plots/so3_so2_<function>_<kernel>.pdf
           plots/so3_so2_<function>_combined.pdf
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------- #
# Matplotlib defaults (publication quality)
# ---------------------------------------------------------------- #
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": False,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


# ---------------------------------------------------------------- #
# Logging
# ---------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SO3_SO2_experiment")


# ---------------------------------------------------------------- #
# SO(3) sampling and helpers
# ---------------------------------------------------------------- #
def random_unit_quaternions(num_samples: int) -> np.ndarray:
    """Haar on S^3 (pushforward gives Haar on SO(3) via the quaternion map)."""
    quaternions = np.random.randn(num_samples, 4)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions


def quat_to_rotmat(quaternions: np.ndarray) -> np.ndarray:
    """Convert unit quaternions (w, x, y, z) of shape (..., 4) to rotations (..., 3, 3)."""
    w, x, y, z = (
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
        quaternions[..., 3],
    )
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R = np.empty(quaternions.shape[:-1] + (3, 3))
    R[..., 0, 0] = ww + xx - yy - zz
    R[..., 0, 1] = 2 * (xy - wz)
    R[..., 0, 2] = 2 * (xz + wy)
    R[..., 1, 0] = 2 * (xy + wz)
    R[..., 1, 1] = ww - xx + yy - zz
    R[..., 1, 2] = 2 * (yz - wx)
    R[..., 2, 0] = 2 * (xz - wy)
    R[..., 2, 1] = 2 * (yz + wx)
    R[..., 2, 2] = ww - xx - yy + zz
    return R


def sample_haar_so3(num_samples: int) -> np.ndarray:
    """Draw Haar-uniform rotations on SO(3)."""
    return quat_to_rotmat(random_unit_quaternions(num_samples))


def sample_so3_with_special_point(num_samples: int, special: np.ndarray) -> np.ndarray:
    """Return [special] + (num_samples - 1) Haar samples on SO(3)."""
    assert num_samples >= 1
    if num_samples == 1:
        return special[None, ...]
    rest = sample_haar_so3(num_samples - 1)
    return np.concatenate([special[None, ...], rest], axis=0)


def rot_z(theta: np.ndarray) -> np.ndarray:
    """theta shape (m,) -> Rz(theta) shape (m, 3, 3)."""
    cosine, sine = np.cos(theta), np.sin(theta)
    rotations = np.zeros((theta.shape[0], 3, 3))
    rotations[:, 0, 0] = cosine
    rotations[:, 0, 1] = -sine
    rotations[:, 1, 0] = sine
    rotations[:, 1, 1] = cosine
    rotations[:, 2, 2] = 1.0
    return rotations


def sample_haar_so2(num_group_samples: int) -> np.ndarray:
    """Haar on SO(2): theta ~ Unif[0, 2*pi), returned as 3x3 z-rotations."""
    theta = np.random.uniform(0.0, 2 * np.pi, size=(num_group_samples,))
    return rot_z(theta)


def special_point_so3() -> np.ndarray:
    """Special evaluation point R0 = I_3."""
    return np.eye(3)


def verify_frobenius_angle_identity(num_pairs: int = 2048) -> None:
    """Numerically check that ||R - S||_F^2 = 8 sin^2(theta/2), theta = angle(R^T S)."""
    R = sample_haar_so3(num_pairs)
    S = sample_haar_so3(num_pairs)

    diff = R - S
    frob_squared = np.sum(diff * diff, axis=(1, 2))

    RtS = np.einsum("nij,nkj->nik", np.transpose(R, (0, 2, 1)), S)
    trace_RtS = RtS[:, 0, 0] + RtS[:, 1, 1] + RtS[:, 2, 2]
    cosine_theta = np.clip((trace_RtS - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cosine_theta)
    rhs = 8.0 * (np.sin(theta * 0.5) ** 2)

    denom = np.maximum(rhs, 1e-12)
    rel_err = np.abs(frob_squared - rhs) / denom
    logger.info(
        "Frobenius-angle identity: max relative error over %d pairs = %.3e",
        num_pairs,
        float(rel_err.max()),
    )


# ---------------------------------------------------------------- #
# G-invariant eigenfunction f_ell
# ---------------------------------------------------------------- #
def legendre_P_l_of_x(x: np.ndarray, ell: int) -> np.ndarray:
    """Evaluate Legendre polynomial P_ell(x) via numpy.polynomial.legendre."""
    coefficients = np.zeros(ell + 1)
    coefficients[ell] = 1.0
    return np.polynomial.legendre.legval(x, coefficients)


def function_P_1_00(rotations: np.ndarray) -> np.ndarray:
    """P_{1,00}(R) = R[2, 2] (the ell=1 special case)."""
    return rotations[:, 2, 2]


def function_f_ell(rotations: np.ndarray, ell: int) -> np.ndarray:
    """f_ell(R) = P_ell(R[2, 2]); right SO(2)-invariant."""
    if ell == 1:
        return function_P_1_00(rotations)
    return legendre_P_l_of_x(rotations[:, 2, 2], ell)


# ---------------------------------------------------------------- #
# Kernels on SO(3)
# ---------------------------------------------------------------- #
def compute_weights_euclidean_frob(
    rotations: np.ndarray, target_rotation: np.ndarray, epsilon: float,
) -> np.ndarray:
    """Gaussian in the ambient Frobenius metric:
        w_j = exp(-||R_j - R0||_F^2 / eps).
    """
    differences = rotations - target_rotation
    distances_squared = np.sum(differences ** 2, axis=(1, 2))
    return np.exp(-distances_squared / epsilon)


def compute_weights_min_orbit_so2(
    rotations: np.ndarray,
    target_rotation: np.ndarray,
    epsilon: float,
    subgroup_elements: np.ndarray,
) -> np.ndarray:
    """Minimum-orbit weights under the right SO(2) action on the target:
        w_j = exp(-min_{g in SO(2)} ||R_j - R0 g||_F^2 / eps).
    """
    QK = target_rotation[None, ...] @ subgroup_elements           # (m, 3, 3)
    diffs = rotations[:, None, ...] - QK[None, ...]               # (n, m, 3, 3)
    d2 = np.sum(diffs * diffs, axis=(2, 3))                       # (n, m)
    d2_min = np.min(d2, axis=1)                                   # (n,)
    return np.exp(-d2_min / epsilon)


def compute_weights_integral_so2(
    rotations: np.ndarray,
    target_rotation: np.ndarray,
    epsilon: float,
    subgroup_elements: np.ndarray,
) -> np.ndarray:
    """Haar-averaged kernel weights under the right SO(2) action on the target:
        w_j = E_{g in SO(2)}[exp(-||R_j - R0 g||_F^2 / eps)].
    """
    QK = target_rotation[None, ...] @ subgroup_elements           # (m, 3, 3)
    diffs = rotations[:, None, ...] - QK[None, ...]               # (n, m, 3, 3)
    d2 = np.sum(diffs * diffs, axis=(2, 3))                       # (n, m)
    return np.mean(np.exp(-d2 / epsilon), axis=1)


def compute_graph_laplacian_at_point_with_kernel(
    rotations: np.ndarray,
    index_of_target: int,
    function_values: np.ndarray,
    epsilon: float,
    kernel: Literal["euclidean", "min_orbit", "integral"],
    subgroup_elements: Optional[np.ndarray] = None,
) -> float:
    """Row-normalized graph Laplacian at rotations[index_of_target] for the given kernel.

    Returns f(R0) - sum_i w_i f(R_i) / sum_i w_i.  The Laplacian scaling
    (4/eps) and the eigenvalue subtraction are applied at the call site.
    """
    target = rotations[index_of_target]
    if kernel == "euclidean":
        weights = compute_weights_euclidean_frob(rotations, target, epsilon)
    elif kernel == "min_orbit":
        if subgroup_elements is None:
            raise ValueError("subgroup_elements must be provided for the min_orbit kernel")
        weights = compute_weights_min_orbit_so2(rotations, target, epsilon, subgroup_elements)
    elif kernel == "integral":
        if subgroup_elements is None:
            raise ValueError("subgroup_elements must be provided for the integral kernel")
        weights = compute_weights_integral_so2(rotations, target, epsilon, subgroup_elements)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    degree = float(np.sum(weights))
    if degree == 0.0:
        return 0.0
    weighted_sum = float(np.sum(weights * function_values))
    return function_values[index_of_target] - weighted_sum / degree


# Calibration constant turning the row-normalized graph Laplacian into a
# diffusion-maps-style estimator of the Laplace-Beltrami operator under the
# SO(3) Frobenius metric. Pulled out so the convention is visible in one place.
LAPLACIAN_SCALE: float = 4.0


def laplace_estimate(L_hat_f: float, epsilon: float) -> float:
    """Diffusion-maps Laplace-Beltrami estimate: (LAPLACIAN_SCALE / eps) * L_hat_f."""
    return (LAPLACIAN_SCALE / epsilon) * L_hat_f


# ---------------------------------------------------------------- #
# Paired-sampling experiment runner
# ---------------------------------------------------------------- #
def run_experiment_so3(
    function_values_fn: Callable[[np.ndarray], np.ndarray],
    eigenvalue: float,
    kernel: Literal["euclidean", "min_orbit", "integral"],
    epsilon_values: np.ndarray,
    num_points: int,
    num_trials: int,
    num_group_samples: int,
) -> np.ndarray:
    """Paired-sampling single-point experiment on SO(3)/SO(2).

    Rotations (and, for G-invariant kernels, SO(2) subgroup samples) are drawn
    ONCE per trial and reused across all epsilon. Per-trial signed errors

        e_{i,t} = laplace_estimate(L_hat f(R0); eps_i) - eigenvalue

    are returned as an array of shape ``(len(epsilon_values), num_trials)``.
    The caller is responsible for ensuring ``eigenvalue`` is the Laplace-
    Beltrami eigenvalue of ``function_values_fn`` on the homogeneous space.
    """
    logger.info("SO(3)/SO(2) experiment for kernel %s", kernel)
    all_trial_errors = np.zeros((len(epsilon_values), num_trials), dtype=float)
    needs_subgroup = kernel in ("min_orbit", "integral")
    t0 = time.time()

    for trial in tqdm(range(num_trials), desc=f"{kernel} kernel", leave=True):
        rotations = sample_so3_with_special_point(num_points, special_point_so3())
        idx = 0  # the special point is the first row
        f_vals = function_values_fn(rotations)
        subgroup_elements = sample_haar_so2(num_group_samples) if needs_subgroup else None

        for i, eps in enumerate(epsilon_values):
            L_hat_f = compute_graph_laplacian_at_point_with_kernel(
                rotations=rotations,
                index_of_target=idx,
                function_values=f_vals,
                epsilon=float(eps),
                kernel=kernel,
                subgroup_elements=subgroup_elements,
            )
            err = laplace_estimate(L_hat_f, float(eps)) - eigenvalue
            all_trial_errors[i, trial] = float(err)

    logger.info("Kernel %s done in %.2f s", kernel, time.time() - t0)
    return all_trial_errors


# ---------------------------------------------------------------- #
# Slope fitting
# ---------------------------------------------------------------- #
def compute_fitted_loglog_slopes(
    all_trial_errors_by_kernel: Dict[str, np.ndarray],
    log_epsilon: np.ndarray,
) -> Dict[str, float]:
    """OLS log-log slope per kernel on points strictly left of mean-|error| minimum."""
    fitted: Dict[str, float] = {}
    for kernel_name, all_trial_errors in all_trial_errors_by_kernel.items():
        errors_mean = np.mean(np.abs(all_trial_errors), axis=1)
        log_errors_mean = np.log(errors_mean + 1e-300)
        min_err_index = int(np.argmin(errors_mean))
        if min_err_index >= 2:
            slope, _ = np.polyfit(
                log_epsilon[:min_err_index], log_errors_mean[:min_err_index], 1
            )
            fitted[kernel_name] = float(slope)
        else:
            fitted[kernel_name] = float("nan")
    return fitted


# ---------------------------------------------------------------- #
# Pickle I/O
# ---------------------------------------------------------------- #
def save_results_to_pickle(
    filepath: str,
    epsilon_values: np.ndarray,
    all_trial_errors_by_kernel: Dict[str, np.ndarray],
    log_epsilon: np.ndarray,
    slopes_by_kernel: Dict[str, float],
    metadata: Dict[str, Any],
) -> None:
    """Save experimental results to a pickle file."""
    data = {
        "epsilon_values": epsilon_values,
        "all_trial_errors_by_kernel": all_trial_errors_by_kernel,
        "log_epsilon": log_epsilon,
        "slopes_by_kernel": slopes_by_kernel,
        "metadata": metadata,
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Results saved to: %s", filepath)


def load_results_from_pickle(filepath: str) -> Dict[str, Any]:
    """Load experimental results from a pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    logger.info("Results loaded from: %s", filepath)
    return data


# ---------------------------------------------------------------- #
# Plotting (SEM-based 95% CI band)
# ---------------------------------------------------------------- #
_KERNEL_COLOR = {
    "euclidean": "#1f77b4",  # blue
    "min_orbit": "#ff7f0e",  # orange
    "integral": "#2ca02c",   # green
}

_KERNEL_DISPLAY_TITLE = {
    "euclidean": "Euclidean Kernel",
    "min_orbit": "Minimum Kernel",
    "integral": "Integral Kernel",
}

_KERNEL_DISPLAY_LEGEND = {
    "euclidean": "Euclidean",
    "min_orbit": "Minimum",
    "integral": "Integral",
}

_KERNEL_PLOT_ORDER = ("integral", "min_orbit", "euclidean")


def _log_mae(trials: np.ndarray) -> np.ndarray:
    """log of mean-absolute-error over the trial axis (axis=1)."""
    return np.log(np.mean(np.abs(trials), axis=1) + 1e-300)


def _reference_dash_x_coords(log_epsilon: np.ndarray) -> np.ndarray:
    """Dense x grid spanning [log_epsilon[0], log_epsilon[-1]] for the dashed reference line."""
    return np.linspace(float(log_epsilon[0]), float(log_epsilon[-1]), max(32, len(log_epsilon)))


def _mae(errors: np.ndarray) -> np.ndarray:
    """Mean-absolute-error per epsilon (mean over trial axis if 2D, else |error|)."""
    arr = np.asarray(errors)
    if arr.ndim == 2:
        return np.mean(np.abs(arr), axis=1)
    return np.abs(arr)


def _log_error_ci95_band(
    errors: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """log of the SEM-based 95% CI of mean |error| across trials (or None if 1D)."""
    arr = np.asarray(errors)
    if arr.ndim != 2:
        return None
    abs_e = np.abs(arr)
    n_trials = abs_e.shape[1]
    mean_abs = np.mean(abs_e, axis=1)
    if n_trials > 1:
        sem = np.std(abs_e, axis=1, ddof=1) / np.sqrt(n_trials)
    else:
        sem = np.zeros_like(mean_abs)
    ci_half_width = 1.96 * sem
    lower = np.maximum(mean_abs - ci_half_width, 1e-300)
    upper = np.maximum(mean_abs + ci_half_width, 1e-300)
    return (
        np.log(lower + 1e-300),
        np.log(upper + 1e-300),
    )


def plot_single_kernel(
    epsilon_values: np.ndarray,
    errors: np.ndarray,
    log_epsilon: np.ndarray,
    log_errors: np.ndarray,
    reference_slope: float,
    kernel_name: str,
    save_path: str,
) -> None:
    """Publication-quality log-log plot for a single kernel with SEM-based 95% CI band.

    The dashed line uses ``reference_slope`` anchored at the first grid point
    and drawn across the full ``log_epsilon`` grid. It is only drawn when the
    mean-|error| minimum is strictly past the second grid point (same gate as
    the variance-regime slope fit).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x_plot = log_epsilon
    band = _log_error_ci95_band(errors)
    color = _KERNEL_COLOR.get(kernel_name, "#1f77b4")

    ax.plot(x_plot, log_errors, "-", linewidth=2.5, label="Computed error", color=color)
    ax.plot(x_plot, log_errors, "o", markersize=5, color=color)
    if band is not None:
        log_lo, log_hi = band
        ax.fill_between(x_plot, log_lo, log_hi, alpha=0.2, color=color)

    err_curve = _mae(errors)
    min_err_index = int(np.argmin(err_curve))
    if min_err_index > 1 and np.isfinite(reference_slope):
        x_dash = _reference_dash_x_coords(log_epsilon)
        y_dash = reference_slope * x_dash + (log_errors[0] - reference_slope * log_epsilon[0])
        ax.plot(
            x_dash,
            y_dash,
            "--",
            linewidth=2.5,
            label=f"Fixed slope = {reference_slope:.3f}",
            color="#d62728",
            zorder=5,
        )

    ax.set_xlabel(r"$\log(\varepsilon)$", fontsize=16)
    ax.set_ylabel(r"$\log(\mathrm{Error})$", fontsize=16)
    ax.set_xlim([
        float(np.min(log_epsilon)),
        float(np.max(log_epsilon)),
    ])
    ax.set_title(_KERNEL_DISPLAY_TITLE.get(kernel_name, kernel_name), fontsize=16, pad=15)

    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(loc="best", framealpha=0.95, edgecolor="gray")

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    logger.info("Plot saved: %s", save_path)
    plt.close()


def plot_combined(
    epsilon_values: np.ndarray,
    errors_by_kernel: Dict[str, np.ndarray],
    log_epsilon: np.ndarray,
    log_errors_by_kernel: Dict[str, np.ndarray],
    fitted_slopes_by_kernel: Dict[str, float],
    reference_slopes_by_kernel: Dict[str, float],
    save_path: str,
) -> None:
    """Publication-quality combined log-log plot for all kernels.

    Solid-line legend shows the fitted slope per kernel; each kernel's dashed
    line uses its theory reference slope from ``reference_slopes_by_kernel``.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ordered_kernels = [k for k in _KERNEL_PLOT_ORDER if k in log_errors_by_kernel]
    ordered_kernels += [k for k in log_errors_by_kernel if k not in ordered_kernels]

    x_plot = log_epsilon

    for kernel_name in ordered_kernels:
        log_errs = log_errors_by_kernel[kernel_name]
        trial_errors = errors_by_kernel[kernel_name]
        color = _KERNEL_COLOR.get(kernel_name, "black")
        display_name = _KERNEL_DISPLAY_LEGEND.get(kernel_name, kernel_name)
        fitted_slope = fitted_slopes_by_kernel.get(kernel_name, float("nan"))
        ref_slope = reference_slopes_by_kernel.get(kernel_name, float("nan"))

        ax.plot(
            x_plot, log_errs, "-",
            linewidth=2.5, color=color,
            label=f"{display_name} (slope = {fitted_slope:.3f})",
        )
        ax.plot(x_plot, log_errs, "o", markersize=5, color=color)
        band = _log_error_ci95_band(trial_errors)
        if band is not None:
            log_lo, log_hi = band
            ax.fill_between(x_plot, log_lo, log_hi, alpha=0.2, color=color)

        err_curve = _mae(trial_errors)
        min_err_index = int(np.argmin(err_curve))
        if min_err_index > 1 and np.isfinite(ref_slope):
            x_dash = _reference_dash_x_coords(log_epsilon)
            y_dash = ref_slope * x_dash + (log_errs[0] - ref_slope * log_epsilon[0])
            ax.plot(
                x_dash,
                y_dash,
                "--",
                linewidth=2.0,
                color=color,
                alpha=0.7,
                zorder=5,
            )

    ax.set_xlabel(r"$\log(\varepsilon)$", fontsize=18)
    ax.set_ylabel(r"$\log(\mathrm{Error})$", fontsize=18)
    ax.set_xlim([
        float(np.min(log_epsilon)),
        float(np.max(log_epsilon)),
    ])

    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(loc="best", framealpha=0.95, edgecolor="gray", fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    logger.info("Combined plot saved: %s", save_path)
    plt.close()


# ---------------------------------------------------------------- #
# Render + replot
# ---------------------------------------------------------------- #
def _render_plots(
    *,
    trial_dict: Dict[str, np.ndarray],
    epsilon_values: np.ndarray,
    log_epsilon: np.ndarray,
    metadata: Dict[str, Any],
    plot_dir: str,
) -> None:
    """Render per-kernel and combined plots from in-memory trial data.

    Shared entry point used by both ``main`` (after running the experiment)
    and ``regenerate_all_plots`` (after loading a pickle), so the two paths
    produce byte-identical plots.
    """
    function_name = metadata.get("function_name", "P_1_00")
    reference_slopes = dict(metadata.get("fixed_slopes") or {})
    fitted_slopes = metadata.get("fitted_slopes")
    if not isinstance(fitted_slopes, dict) or not fitted_slopes:
        fitted_slopes = compute_fitted_loglog_slopes(trial_dict, log_epsilon)

    log_errors_by_kernel = {k: _log_mae(v) for k, v in trial_dict.items()}

    for kernel_name, trials in trial_dict.items():
        plot_single_kernel(
            epsilon_values=epsilon_values,
            errors=trials,
            log_epsilon=log_epsilon,
            log_errors=log_errors_by_kernel[kernel_name],
            reference_slope=reference_slopes.get(kernel_name, float("nan")),
            kernel_name=kernel_name,
            save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_{kernel_name}.pdf"),
        )

    plot_combined(
        epsilon_values=epsilon_values,
        errors_by_kernel=trial_dict,
        log_epsilon=log_epsilon,
        log_errors_by_kernel=log_errors_by_kernel,
        fitted_slopes_by_kernel=fitted_slopes,
        reference_slopes_by_kernel=reference_slopes,
        save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_combined.pdf"),
    )


def regenerate_all_plots(pickle_path: str, output_dir: Optional[str] = None) -> None:
    """Reload a results pickle and render the same plots the experiment produced.

    ``output_dir`` defaults to the directory containing the pickle; plots are
    written to ``<output_dir>/plots/``.
    """
    data = load_results_from_pickle(pickle_path)
    trial_dict = data["all_trial_errors_by_kernel"]
    epsilon_values = data["epsilon_values"]
    log_epsilon = data["log_epsilon"]
    metadata = data.get("metadata", {}) or {}

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(pickle_path))
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("REGENERATING PLOTS FROM SAVED DATA")
    print(f"{'=' * 60}")
    print(f"Function:         {metadata.get('function_name', 'P_1_00')}")
    print(f"ell:              {metadata.get('ell', 'N/A')}")
    print(f"Number of points: {metadata.get('num_points', 'N/A')}")
    print(f"Number of trials: {metadata.get('num_trials', 'N/A')}")
    print(f"Output directory: {plot_dir}")
    print(f"{'=' * 60}\n")

    _render_plots(
        trial_dict=trial_dict,
        epsilon_values=epsilon_values,
        log_epsilon=log_epsilon,
        metadata=metadata,
        plot_dir=plot_dir,
    )

    print("\nAll plots regenerated successfully.")


# ---------------------------------------------------------------- #
# Main: run the paired experiment end-to-end
# ---------------------------------------------------------------- #
def main(output_dir: Optional[str] = None, seed: Optional[int] = None) -> None:
    """Run the experiment; the pickle and plots go to ``output_dir``
    (default: the experiment's results/ folder). ``seed`` seeds numpy's global
    generator, from which all SO(3) and SO(2) samples are drawn, so a seeded
    run is exactly reproducible."""
    logger.info("Starting SO(3)/SO(2) paired-sampling experiment")
    if seed is not None:
        np.random.seed(seed)
        logger.info("numpy random seed: %d", seed)

    ell = 1
    num_points = 10000
    num_trials = 1000
    num_group_samples = 200  

    fixed_slopes = {
        "euclidean": -0.75,
        "min_orbit": -0.5,
        "integral": -0.5,
    }

    log_epsilon_min = -4.0 
    log_epsilon_max = 0.4
    num_epsilon = 100
    log_epsilon_values = np.linspace(log_epsilon_min, log_epsilon_max, num_epsilon)
    epsilon_values = np.exp(log_epsilon_values)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    verify_frobenius_angle_identity(num_pairs=2048)

    function_name = "P_1_00"
    kernels = ("euclidean", "min_orbit", "integral")

    # Eigenfunction / eigenvalue contract: both derived from `ell` in one place.
    # f_ell(R) = P_ell(R[2,2]) is right SO(2)-invariant and satisfies
    #     Delta f_ell = -ell(ell+1)/2 f_ell
    # under this code's Frobenius-metric convention.
    eigenfunction: Callable[[np.ndarray], np.ndarray] = lambda R: function_f_ell(R, ell)
    eigenvalue: float = ell * (ell + 1) / 2.0

    print(f"\n{'=' * 60}")
    print("Paired-sampling experiment (euclidean, min_orbit, integral)")
    print(f"Parameters: ell={ell}, num_points={num_points}, num_trials={num_trials}")
    print(f"Epsilon range: [{np.min(epsilon_values):.2e}, {np.max(epsilon_values):.2e}]")
    print(f"{'=' * 60}\n")

    trial_dict: Dict[str, np.ndarray] = {}
    for k in kernels:
        logger.info(">>> Starting %s kernel", k)
        trial_dict[k] = run_experiment_so3(
            function_values_fn=eigenfunction,
            eigenvalue=eigenvalue,
            kernel=k,
            epsilon_values=epsilon_values,
            num_points=num_points,
            num_trials=num_trials,
            num_group_samples=num_group_samples,
        )

    fitted_slopes = compute_fitted_loglog_slopes(trial_dict, log_epsilon_values)

    metadata: Dict[str, Any] = {
        "ell": ell,
        "num_points": num_points,
        "num_trials": num_trials,
        "num_group_samples": num_group_samples,
        "log_epsilon_min": log_epsilon_min,
        "log_epsilon_max": log_epsilon_max,
        "num_epsilon": num_epsilon,
        "function_name": function_name,
        "fixed_slopes": fixed_slopes,
        "fitted_slopes": fitted_slopes,
        "sampling": "paired",
        "seed": seed,
    }

    pickle_path = os.path.join(output_dir, f"so3_so2_{function_name}_results.pkl")
    save_results_to_pickle(
        filepath=pickle_path,
        epsilon_values=epsilon_values,
        all_trial_errors_by_kernel=trial_dict,
        log_epsilon=log_epsilon_values,
        slopes_by_kernel=fixed_slopes,
        metadata=metadata,
    )

    _render_plots(
        trial_dict=trial_dict,
        epsilon_values=epsilon_values,
        log_epsilon=log_epsilon_values,
        metadata=metadata,
        plot_dir=plot_dir,
    )

    # Right SO(2)-invariance sanity check on f_ell.
    rotations = sample_haar_so3(8)
    thetas = np.random.uniform(0.0, 2 * np.pi, size=5)
    subgroup = rot_z(thetas)
    f_vals = function_f_ell(rotations, ell)
    for j in range(len(rotations)):
        values = [
            function_f_ell((rotations[j][None, ...] @ subgroup[m])[0][None, ...], ell)[0]
            for m in range(len(subgroup))
        ]
        assert np.allclose(values, f_vals[j]), "f_ell must be right SO(2)-invariant"

    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY (paired sampling)")
    print(f"{'=' * 60}")
    print(f"Function: {function_name} (ell={ell})")
    print(f"Number of points per trial: {num_points}")
    print(f"Number of trials:           {num_trials}")
    print(f"Number of SO(2) samples:    {num_group_samples}")
    print("\nFitted log-log slopes:")
    for k in kernels:
        print(f"  {k:15s}: {fitted_slopes[k]:.4f}")
    print(f"\nPickle: {pickle_path}")
    print(f"Plots:  {plot_dir}/so3_so2_{function_name}_*.pdf")
    print(f"{'=' * 60}\n")

    logger.info("Paired-sampling SO(3)/SO(2) experiment completed successfully.")


if __name__ == "__main__":
    main()
