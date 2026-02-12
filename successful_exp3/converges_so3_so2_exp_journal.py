"""
Single-point experiments on SO(3) with right SO(2) action, aligning with the
practice described in the manuscript section "Single–Point Experiments on M with
G–Invariant Eigenfunction".

V3 UPDATE: Modified based on discussion with Paulina to match the reference implementation.
Key changes:
- Updated kernel normalization (divide by 2*epsilon, scale differences by 1/sqrt(2))
- Changed Laplacian estimator (removed epsilon division from the graph Laplacian itself)
- Updated error calculation to match reference (scaled by 2/epsilon, compare to exact eigenvalue)
- Simplified P_1_00 function to directly access R[2,2] instead of general Legendre evaluation

JOURNAL VERSION:
- Log-log plots only with publication-quality formatting
- PDF output for publication
- Pickle file support for result persistence and re-plotting

Geometry and notation.
- M = SO(3) with its bi-invariant metric (Frobenius inner product on R^{3x3}).
- G = SO(2) rotations about the z-axis; right action: R -> R * Rz(theta).
- The homogeneous space is N = M/G \simeq S^2.

Ground-truth eigenfunction and Laplacian.
- Wigner D-entries form eigenfunctions of the negative Laplace–Beltrami on M.
- f_ell(R) = D^ell_{00}(R) = P_ell(R[2,2]) is bi-invariant under G and
  satisfies Δ f_ell = - ell(ell+1) f_ell. We take ell=1 as the signal.

Kernels (right G-action on the target R0):
1) Euclidean/chordal kernel in the embedding R^9:
   K_E(R, R0) = exp( - ||R - R0||_F^2 / (2*ε) ) with rescaling by 1/sqrt(2).
2) G-integral (Haar) kernel:
   K_H^G(R, R0) = E_{g in G} [ exp( - ||R - R0 g||_F^2 / (2*ε) ) ].
3) Minimum-orbit kernel:
   K_min^G(R, R0) = exp( - min_{g in G} ||R - R0 g||_F^2 / (2*ε) ).

Estimator (row-normalized graph Laplacian at R0):
  L_hat f(R0) = ( sum_i w_i f(R_i) / sum_i w_i - f(R0) )
  Then: Δ_hat_ε f(R0) = (2/ε) * L_hat f(R0)

We compare (2/ε) * L_hat f(R0) + ell(ell+1) to zero and report RMSE
vs ε along with a small-ε log–log slope.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import pickle
from typing import Literal, Tuple, Dict, Any
import os
from tqdm import tqdm

# Set publication-quality plotting defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})


# ------------------------ Logging ------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SO3_SO2_experiment_v3_journal")


# ------------------------ SO(3) Sampling & Helpers ------------------------ #

def random_unit_quaternions(num_samples: int) -> np.ndarray:
    """Haar on S^3 -> pushforward to Haar on SO(3) via quaternion->rotation map."""
    quaternions = np.random.randn(num_samples, 4)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions


def quat_to_rotmat(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternions to rotation matrices.
    Input shape (..., 4) with quaternion order (w, x, y, z).
    Output shape (..., 3, 3).
    """
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
    """Return [special] + (num_samples-1) Haar samples on SO(3)."""
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
    """Haar on SO(2): theta ~ Unif[0, 2π)."""
    theta = np.random.uniform(0.0, 2 * np.pi, size=(num_group_samples,))
    return rot_z(theta)


def special_point_so3() -> np.ndarray:
    """Special point R0 = I_3."""
    return np.eye(3)


def find_special_point_index_so3(rotations: np.ndarray) -> int:
    """Index of rotation closest to I_3 in Frobenius norm."""
    identity = np.eye(3)
    distances_squared = np.sum((rotations - identity) ** 2, axis=(1, 2))
    return int(np.argmin(distances_squared))


def geodesic_angle_of_rotation(rotation: np.ndarray) -> float:
    """Return geodesic angle θ of a rotation matrix in SO(3)."""
    trace_value = float(np.trace(rotation))
    cosine_theta = (trace_value - 1.0) / 2.0
    cosine_theta = max(-1.0, min(1.0, cosine_theta))
    return float(np.arccos(cosine_theta))


def verify_frobenius_angle_identity(num_pairs: int = 2048) -> None:
    """
    Verify numerically that ||R - S||_F^2 = 8 sin^2(θ/2), with θ the angle of R^T S.
    Logs the maximum relative error across a sample of pairs.
    """
    R = sample_haar_so3(num_pairs)
    S = sample_haar_so3(num_pairs)

    diff = R - S
    frob_squared = np.sum(diff * diff, axis=(1, 2))

    RtS = np.einsum('nij,nkj->nik', np.transpose(R, (0, 2, 1)), S)
    trace_RtS = RtS[:, 0, 0] + RtS[:, 1, 1] + RtS[:, 2, 2]
    cosine_theta = (trace_RtS - 1.0) / 2.0
    cosine_theta = np.clip(cosine_theta, -1.0, 1.0)
    theta = np.arccos(cosine_theta)
    rhs = 8.0 * (np.sin(theta * 0.5) ** 2)

    denom = np.maximum(rhs, 1e-12)
    rel_err = np.abs(frob_squared - rhs) / denom
    logger.info(
        f"Frobenius-angle identity: max relative error over {num_pairs} pairs = {float(rel_err.max()):.3e}"
    )


# ------------------------ G-invariant eigenfunction f_ell ------------------------ #

def legendre_P_l_of_x(x: np.ndarray, ell: int) -> np.ndarray:
    """Evaluate Legendre polynomial P_ell(x) using numpy.polynomial.legendre."""
    coefficients = np.zeros(ell + 1)
    coefficients[ell] = 1.0
    return np.polynomial.legendre.legval(x, coefficients)


def function_P_1_00(rotations: np.ndarray) -> np.ndarray:
    """
    P_1_00(R) = R[2,2] for ell=1.
    This is equivalent to P_1(cos(beta)) where beta is the Euler angle.
    """
    return rotations[:, 2, 2]


def function_f_ell(rotations: np.ndarray, ell: int) -> np.ndarray:
    """f_ell(R) = P_ell(R[2,2]). Right SO(2)-invariant (depends only on polar angle)."""
    if ell == 1:
        return function_P_1_00(rotations)
    else:
        x = rotations[:, 2, 2]
        return legendre_P_l_of_x(x, ell)


def laplace_beltrami_f_ell(rotations: np.ndarray, ell: int) -> np.ndarray:
    """Exact Laplacian under geometer sign: Δ f_ell = - ell(ell+1) f_ell."""
    return -ell * (ell + 1) * function_f_ell(rotations, ell)


# ------------------------ Kernels on SO(3) ------------------------ #

def compute_weights_euclidean_frob(
    rotations: np.ndarray, target_rotation: np.ndarray, epsilon: float
) -> np.ndarray:
    """
    Gaussian weights in Frobenius metric with Paulina's normalization:
    exp(-||(R - R0)/sqrt(2)||_F^2 / (2*ε))
    """
    differences = (rotations - target_rotation) / np.sqrt(2.0)
    distances_squared = np.sum(differences ** 2, axis=(1, 2))
    return np.exp(-distances_squared / (2.0 * epsilon))


def compute_weights_min_orbit_so2(
    rotations: np.ndarray, target_rotation: np.ndarray, epsilon: float, subgroup_elements: np.ndarray
) -> np.ndarray:
    """
    Minimum-orbit weights with right G-action applied to the target:
      w_j = exp( - min_{g in SO(2)} ||(R_j - R0 g)/sqrt(2)||_F^2 / (2*ε) ).
    """
    QK = target_rotation[None, ...] @ subgroup_elements  # (m, 3, 3)
    diffs = rotations[:, None, ...] - QK[None, ...]      # (n, m, 3, 3)
    diffs = diffs / np.sqrt(2.0)
    d2 = np.sum(diffs * diffs, axis=(2, 3))              # (n, m)
    d2_min = np.min(d2, axis=1)                          # (n,)
    return np.exp(-d2_min / (2.0 * epsilon))


def compute_weights_integral_so2(
    rotations: np.ndarray, target_rotation: np.ndarray, epsilon: float, subgroup_elements: np.ndarray
) -> np.ndarray:
    """
    Integral (Haar-averaged) kernel weights with right G-action on the target:
      w_j = E_{g in SO(2)} [ exp( - ||(R_j - R0 g)/sqrt(2)||_F^2 / (2*ε) ) ].
    """
    QK = target_rotation[None, ...] @ subgroup_elements  # (m, 3, 3)
    diffs = rotations[:, None, ...] - QK[None, ...]      # (n, m, 3, 3)
    diffs = diffs / np.sqrt(2.0)
    d2 = np.sum(diffs * diffs, axis=(2, 3))              # (n, m)
    return np.mean(np.exp(-d2 / (2.0 * epsilon)), axis=1)


def compute_graph_laplacian_at_point_with_kernel(
    rotations: np.ndarray,
    index_of_target: int,
    function_values: np.ndarray,
    epsilon: float,
    kernel: Literal["euclidean", "min_orbit", "integral"],
    subgroup_elements: np.ndarray | None = None,
) -> float:
    """
    Graph Laplacian row at rotations[index_of_target] using the specified kernel.
    Returns L_hat f(R0) = (sum w_i f(R_i) / sum w_i - f(R0))
    WITHOUT dividing by epsilon (that's done separately in error calculation).
    """
    target = rotations[index_of_target]
    if kernel == "euclidean":
        weights = compute_weights_euclidean_frob(rotations, target, epsilon)
    elif kernel == "min_orbit":
        if subgroup_elements is None:
            raise ValueError("subgroup_elements must be provided for min_orbit kernel")
        weights = compute_weights_min_orbit_so2(rotations, target, epsilon, subgroup_elements)
    elif kernel == "integral":
        if subgroup_elements is None:
            raise ValueError("subgroup_elements must be provided for integral kernel")
        weights = compute_weights_integral_so2(rotations, target, epsilon, subgroup_elements)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    degree = float(np.sum(weights))
    if degree == 0.0:
        return 0.0
    weighted_sum = float(np.sum(weights * function_values))
    return weighted_sum / degree - function_values[index_of_target]


# ------------------------ Experiment Runner ------------------------ #

def run_experiment_so3(
    ell: int,
    kernel: Literal["euclidean", "min_orbit", "integral"],
    epsilon_values: np.ndarray,
    num_points: int,
    num_trials: int,
    num_group_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run single-point experiment on SO(3)/SO(2) for the provided kernel.
    
    Returns:
        epsilon_values: array of epsilon values
        all_trial_errors: (num_epsilon, num_trials) array of individual trial errors
        errors_rmse: RMSE across trials for each epsilon
    """
    logger.info(f"Starting SO(3)/SO(2) experiment v3 for kernel: {kernel}, ell={ell}")
    all_trial_errors = np.zeros((len(epsilon_values), num_trials), dtype=float)
    t0 = time.time()

    for i, eps in enumerate(tqdm(epsilon_values, desc=f"{kernel} kernel", leave=True)):
        for trial in range(num_trials):
            # Sample rotations and force special point
            rotations = sample_so3_with_special_point(num_points, special_point_so3())
            idx = 0  # the special point is the first row

            # Function values
            f_vals = function_f_ell(rotations, ell)

            # Subgroup samples if needed
            K = None
            if kernel in ("min_orbit", "integral"):
                K = sample_haar_so2(num_group_samples)

            # Graph Laplacian at special point
            L_hat_f = compute_graph_laplacian_at_point_with_kernel(
                rotations=rotations,
                index_of_target=idx,
                function_values=f_vals,
                epsilon=float(eps),
                kernel=kernel,
                subgroup_elements=K,
            )

            # Error is (2/ε)*L_hat f + ell(ell+1)
            err = (2.0 / eps) * L_hat_f + ell * (ell + 1)
            all_trial_errors[i, trial] = float(err)

        # Compute RMSE for this epsilon
        errors_rmse = np.sqrt(np.mean(all_trial_errors ** 2, axis=1))

    elapsed = time.time() - t0
    logger.info(f"Completed kernel {kernel} in {elapsed:.2f} seconds")

    return epsilon_values, all_trial_errors, errors_rmse


# ------------------------ Data Persistence ------------------------ #

def save_results_to_pickle(
    filepath: str,
    epsilon_values: np.ndarray,
    all_trial_errors_by_kernel: Dict[str, np.ndarray],
    log_epsilon: np.ndarray,
    slopes_by_kernel: Dict[str, float],
    metadata: Dict[str, Any],
) -> None:
    """Save experimental results to pickle file for later analysis."""
    data = {
        'epsilon_values': epsilon_values,
        'all_trial_errors_by_kernel': all_trial_errors_by_kernel,
        'log_epsilon': log_epsilon,
        'slopes_by_kernel': slopes_by_kernel,
        'metadata': metadata,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Results saved to: {filepath}")


def load_results_from_pickle(filepath: str) -> Dict[str, Any]:
    """Load experimental results from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Results loaded from: {filepath}")
    return data


# ------------------------ Publication-Quality Plotting ------------------------ #

def plot_single_kernel_journal(
    epsilon_values: np.ndarray,
    all_trial_errors: np.ndarray,
    log_epsilon: np.ndarray,
    fixed_slope: float,
    kernel_name: str,
    save_path: str,
    num_trials: int,
    num_group_samples: int,
) -> None:
    """Create publication-quality log-log plot for a single kernel with confidence intervals."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Compute statistics across trials
    errors_mean = np.mean(np.abs(all_trial_errors), axis=1)
    errors_lower = np.percentile(np.abs(all_trial_errors), 25, axis=1)
    errors_upper = np.percentile(np.abs(all_trial_errors), 75, axis=1)
    
    log_errors_mean = np.log(errors_mean + 1e-300)
    log_errors_lower = np.log(errors_lower + 1e-300)
    log_errors_upper = np.log(errors_upper + 1e-300)

    # Plot data with confidence intervals
    ax.plot(log_epsilon, log_errors_mean, '-', linewidth=2.5, label='Mean error', color='#1f77b4')
    ax.plot(log_epsilon, log_errors_mean, 'o', markersize=5, color='#1f77b4')
    ax.fill_between(log_epsilon, log_errors_lower, log_errors_upper, alpha=0.2, color='#1f77b4')
    
    # Reference line with fixed slope across entire range
    y_fit = fixed_slope * log_epsilon + (log_errors_mean[0] - fixed_slope * log_epsilon[0])
    ax.plot(log_epsilon, y_fit, '--', linewidth=2.5, 
            label=f'Slope = {fixed_slope:.2f}', color='#d62728')
    
    # Labels and formatting
    ax.set_xlabel(r'$\log(\varepsilon)$', fontsize=16)
    ax.set_ylabel(r'$\log(\mathrm{Error})$', fontsize=16)
    ax.set_xlim([-6, -1])
    
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')
    
    # Add text with experiment parameters
    text_str = f'Trials: {num_trials}\nGroup elements: {num_group_samples}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved as: {save_path}")
    plt.close()


def plot_combined_journal(
    epsilon_values: np.ndarray,
    all_trial_errors_by_kernel: Dict[str, np.ndarray],
    log_epsilon: np.ndarray,
    slopes_by_kernel: Dict[str, float],
    save_path: str,
    num_trials: int,
    num_group_samples: int,
) -> None:
    """Create publication-quality combined log-log plot for all kernels with confidence intervals."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Professional color scheme
    colors = {
        "euclidean": "#1f77b4",      # Blue
        "min_orbit": "#ff7f0e",      # Orange
        "integral": "#2ca02c"        # Green
    }
    
    # Display names for legend
    kernel_display = {
        'euclidean': 'Euclidean',
        'min_orbit': 'Minimum-Orbit',
        'integral': 'Integral'
    }

    # Plot kernels in specific order: integral (green), min_orbit (orange), euclidean (blue)
    kernel_order = ["integral", "min_orbit", "euclidean"]
    
    for kernel_name in kernel_order:
        all_trial_errors = all_trial_errors_by_kernel[kernel_name]
        color = colors.get(kernel_name, 'black')
        display_name = kernel_display.get(kernel_name, kernel_name)
        slope = slopes_by_kernel[kernel_name]
        
        # Compute statistics
        errors_mean = np.mean(np.abs(all_trial_errors), axis=1)
        errors_lower = np.percentile(np.abs(all_trial_errors), 25, axis=1)
        errors_upper = np.percentile(np.abs(all_trial_errors), 75, axis=1)
        
        log_errors_mean = np.log(errors_mean + 1e-300)
        log_errors_lower = np.log(errors_lower + 1e-300)
        log_errors_upper = np.log(errors_upper + 1e-300)
        
        # Data points and line with confidence interval
        ax.plot(log_epsilon, log_errors_mean, '-', linewidth=2.5, 
                color=color, label=f'{display_name} (slope = {slope:.2f})')
        ax.plot(log_epsilon, log_errors_mean, 'o', markersize=5, color=color)
        ax.fill_between(log_epsilon, log_errors_lower, log_errors_upper, 
                       alpha=0.2, color=color)
        
        # Reference line with fixed slope across entire range
        y_fit = slope * log_epsilon + (log_errors_mean[0] - slope * log_epsilon[0])
        ax.plot(log_epsilon, y_fit, '--', linewidth=2.0, color=color, alpha=0.7)
    
    # Labels and formatting
    ax.set_xlabel(r'$\log(\varepsilon)$', fontsize=18)
    ax.set_ylabel(r'$\log(\mathrm{Error})$', fontsize=18)
    ax.set_xlim([-6, -1])
    
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fontsize=13)
    
    # Add text with experiment parameters
    text_str = f'Trials: {num_trials}\nGroup elements: {num_group_samples}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    logger.info(f"Combined plot saved as: {save_path}")
    plt.close()


# ------------------------ Main ------------------------ #

def main():
    logger.info(
        "Starting SO(3)/SO(2) single-point experiments (v3 - Journal Version)"
    )

    # Experiment parameters
    ell = 1
    num_points = 10000
    num_trials = 1000
    num_group_samples = 200

    # Fixed slopes for reference lines
    fixed_slopes = {
        "euclidean": -0.75,
        "min_orbit": -0.5,
        "integral": -0.5,
    }

    # Epsilon range
    log_epsilon_min = -6.0
    log_epsilon_max = -1.0
    num_epsilon = 100
    log_epsilon_values = np.linspace(log_epsilon_min, log_epsilon_max, num_epsilon)
    epsilon_values = np.exp(log_epsilon_values)

    # Output directory
    output_dir = "/a/home/cc/students/math/vigderyeari/Documents/G_invariant_kernel/exp3/successful_exp3"
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Identity check
    verify_frobenius_angle_identity(num_pairs=2048)

    function_name = "P_1_00"
    
    # Progress tracking
    kernels = ["euclidean", "min_orbit", "integral"]
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Running experiments for {len(kernels)} kernels")
    print(f"Parameters: ell={ell}, num_points={num_points}, num_trials={num_trials}")
    print(f"Epsilon range: [{np.min(epsilon_values):.2e}, {np.max(epsilon_values):.2e}]")
    print(f"{'='*60}\n")

    # Euclidean kernel
    logger.info(">>> [1/3] Starting Euclidean kernel experiment")
    eps_euc, all_trials_euc, rmse_euc = run_experiment_so3(
        ell=ell,
        kernel="euclidean",
        epsilon_values=epsilon_values,
        num_points=num_points,
        num_trials=num_trials,
        num_group_samples=num_group_samples,
    )
    results["euclidean"] = (eps_euc, all_trials_euc, rmse_euc)
    plot_single_kernel_journal(
        epsilon_values=eps_euc,
        all_trial_errors=all_trials_euc,
        log_epsilon=log_epsilon_values,
        fixed_slope=fixed_slopes["euclidean"],
        kernel_name="euclidean",
        save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_euclidean_journal.pdf"),
        num_trials=num_trials,
        num_group_samples=num_group_samples,
    )

    # Minimum-orbit kernel
    logger.info(">>> [2/3] Starting Minimum-orbit kernel experiment")
    eps_min, all_trials_min, rmse_min = run_experiment_so3(
        ell=ell,
        kernel="min_orbit",
        epsilon_values=epsilon_values,
        num_points=num_points,
        num_trials=num_trials,
        num_group_samples=num_group_samples,
    )
    results["min_orbit"] = (eps_min, all_trials_min, rmse_min)
    plot_single_kernel_journal(
        epsilon_values=eps_min,
        all_trial_errors=all_trials_min,
        log_epsilon=log_epsilon_values,
        fixed_slope=fixed_slopes["min_orbit"],
        kernel_name="min_orbit",
        save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_min_orbit_journal.pdf"),
        num_trials=num_trials,
        num_group_samples=num_group_samples,
    )

    # Integral kernel
    logger.info(">>> [3/3] Starting Integral kernel experiment")
    eps_int, all_trials_int, rmse_int = run_experiment_so3(
        ell=ell,
        kernel="integral",
        epsilon_values=epsilon_values,
        num_points=num_points,
        num_trials=num_trials,
        num_group_samples=num_group_samples,
    )
    results["integral"] = (eps_int, all_trials_int, rmse_int)
    plot_single_kernel_journal(
        epsilon_values=eps_int,
        all_trial_errors=all_trials_int,
        log_epsilon=log_epsilon_values,
        fixed_slope=fixed_slopes["integral"],
        kernel_name="integral",
        save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_integral_journal.pdf"),
        num_trials=num_trials,
        num_group_samples=num_group_samples,
    )

    # Combined plot
    plot_combined_journal(
        epsilon_values=epsilon_values,
        all_trial_errors_by_kernel={"euclidean": all_trials_euc, "min_orbit": all_trials_min, "integral": all_trials_int},
        log_epsilon=log_epsilon_values,
        slopes_by_kernel=fixed_slopes,
        save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_combined_journal.pdf"),
        num_trials=num_trials,
        num_group_samples=num_group_samples,
    )

    # Save results to pickle
    metadata = {
        'ell': ell,
        'num_points': num_points,
        'num_trials': num_trials,
        'num_group_samples': num_group_samples,
        'log_epsilon_min': log_epsilon_min,
        'log_epsilon_max': log_epsilon_max,
        'num_epsilon': num_epsilon,
        'function_name': function_name,
        'euclidean_num_points': num_points * 30,
        'fixed_slopes': fixed_slopes,
    }
    
    save_results_to_pickle(
        filepath=os.path.join(output_dir, f"so3_so2_{function_name}_results.pkl"),
        epsilon_values=epsilon_values,
        all_trial_errors_by_kernel={"euclidean": all_trials_euc, "min_orbit": all_trials_min, "integral": all_trials_int},
        log_epsilon=log_epsilon_values,
        slopes_by_kernel=fixed_slopes,
        metadata=metadata,
    )

    # Invariance sanity check
    rotations = sample_haar_so3(8)
    thetas = np.random.uniform(0.0, 2 * np.pi, size=5)
    subgroup = rot_z(thetas)
    f_vals = function_f_ell(rotations, ell)
    for j in range(len(rotations)):
        values = [
            function_f_ell((rotations[j][None, ...] @ subgroup[k])[0][None, ...], ell)[0]
            for k in range(len(subgroup))
        ]
        assert np.allclose(values, f_vals[j]), "f_ell must be right SO(2)-invariant"

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Function: P_1_00 (ell={ell})")
    print(f"Number of points: {num_points}")
    print(f"Number of trials: {num_trials}")
    print(f"Number of group samples: {num_group_samples}")
    print(f"\nReference slopes (log-log):")
    print(f"  Euclidean kernel:     {fixed_slopes['euclidean']:.2f}")
    print(f"  Minimum-orbit kernel: {fixed_slopes['min_orbit']:.2f}")
    print(f"  Integral kernel:      {fixed_slopes['integral']:.2f}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Plots directory: {plot_dir}")
    print(f"{'='*60}\n")

    logger.info("All SO(3)/SO(2) experiments (v3 - Journal Version) completed successfully.")


if __name__ == "__main__":
    main()

