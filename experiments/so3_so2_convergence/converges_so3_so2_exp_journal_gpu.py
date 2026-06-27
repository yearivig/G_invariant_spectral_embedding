"""
GPU-optimized version of the SO(3)/SO(2) single-point experiments.

Optimizations:
1. PyTorch GPU acceleration for all matrix operations
2. Batched trials - run all trials simultaneously per epsilon
3. Multi-GPU support using DataParallel or manual distribution
4. Vectorized operations where possible

Requires: torch with CUDA support
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import time
import logging
import pickle
import json
from typing import Literal, Tuple, Dict, Any, Optional
import os
from tqdm import tqdm

# Set publication-quality plotting defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': False,
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

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SO3_SO2_experiment_GPU")


# region agent log
def _agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "sessionId": "67ccf2",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        debug_log = os.environ.get("PAPER_AGENT_DEBUG_LOG", "debug-67ccf2.log")
        with open(debug_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass
# endregion


# ------------------------ GPU Setup ------------------------ #

def get_available_gpus():
    """Get list of available GPU devices."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return []
    num_gpus = torch.cuda.device_count()
    gpus = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            'id': i,
            'name': props.name,
            'memory': props.total_memory / (1024**3)  # GB
        })
        logger.info(f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
    return gpus


def select_device(gpu_id: Optional[int] = None):
    """Select compute device."""
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    elif torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


# ------------------------ SO(3) Sampling (GPU) ------------------------ #

def random_unit_quaternions_gpu(num_samples: int, device: torch.device) -> torch.Tensor:
    """Haar on S^3 -> pushforward to Haar on SO(3) via quaternion->rotation map."""
    quaternions = torch.randn(num_samples, 4, device=device)
    quaternions = F.normalize(quaternions, p=2, dim=1)
    return quaternions


def quat_to_rotmat_gpu(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to rotation matrices.
    Input shape (..., 4) with quaternion order (w, x, y, z).
    Output shape (..., 3, 3).
    """
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R = torch.zeros(quaternions.shape[:-1] + (3, 3), device=quaternions.device, dtype=quaternions.dtype)
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


def sample_haar_so3_gpu(num_samples: int, device: torch.device) -> torch.Tensor:
    """Draw Haar-uniform rotations on SO(3)."""
    return quat_to_rotmat_gpu(random_unit_quaternions_gpu(num_samples, device))


def rot_z_gpu(theta: torch.Tensor, device: torch.device) -> torch.Tensor:
    """theta shape (m,) -> Rz(theta) shape (m, 3, 3)."""
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    rotations = torch.zeros((theta.shape[0], 3, 3), device=device, dtype=theta.dtype)
    rotations[:, 0, 0] = cosine
    rotations[:, 0, 1] = -sine
    rotations[:, 1, 0] = sine
    rotations[:, 1, 1] = cosine
    rotations[:, 2, 2] = 1.0
    return rotations


def sample_haar_so2_gpu(num_group_samples: int, device: torch.device) -> torch.Tensor:
    """Haar on SO(2): theta ~ Unif[0, 2π)."""
    theta = torch.rand(num_group_samples, device=device) * 2 * np.pi
    return rot_z_gpu(theta, device)


# ------------------------ Batched Kernel Computations (GPU) ------------------------ #

def compute_weights_euclidean_batched(
    rotations: torch.Tensor,  # (batch, n, 3, 3)
    target_rotation: torch.Tensor,  # (3, 3)
    epsilon: float
) -> torch.Tensor:
    """
    Batched Gaussian weights in Frobenius metric.
    Returns shape (batch, n)
    """
    diff = (rotations - target_rotation) / np.sqrt(2.0)
    dist_sq = (diff ** 2).sum(dim=(-2, -1))  # (batch, n)
    return torch.exp(-dist_sq / (2.0 * epsilon))


def compute_weights_min_orbit_batched(
    rotations: torch.Tensor,  # (batch, n, 3, 3)
    target_rotation: torch.Tensor,  # (3, 3)
    epsilon: float,
    subgroup_elements: torch.Tensor  # (m, 3, 3)
) -> torch.Tensor:
    """
    Batched minimum-orbit kernel weights.
    Returns shape (batch, n)
    """
    # target @ subgroup: (m, 3, 3)
    QK = torch.einsum('ij,mjk->mik', target_rotation, subgroup_elements)  # (m, 3, 3)
    
    # diffs: (batch, n, m, 3, 3)
    diffs = rotations.unsqueeze(2) - QK.unsqueeze(0).unsqueeze(0)  # broadcast
    diffs = diffs / np.sqrt(2.0)
    
    # d2: (batch, n, m)
    d2 = (diffs ** 2).sum(dim=(-2, -1))
    
    # min over group: (batch, n)
    d2_min = d2.min(dim=-1).values
    
    return torch.exp(-d2_min / (2.0 * epsilon))


def compute_weights_integral_batched(
    rotations: torch.Tensor,  # (batch, n, 3, 3)
    target_rotation: torch.Tensor,  # (3, 3)
    epsilon: float,
    subgroup_elements: torch.Tensor  # (m, 3, 3)
) -> torch.Tensor:
    """
    Batched integral (Haar-averaged) kernel weights.
    Returns shape (batch, n)
    """
    # target @ subgroup: (m, 3, 3)
    QK = torch.einsum('ij,mjk->mik', target_rotation, subgroup_elements)
    
    # diffs: (batch, n, m, 3, 3)
    diffs = rotations.unsqueeze(2) - QK.unsqueeze(0).unsqueeze(0)
    diffs = diffs / np.sqrt(2.0)
    
    # d2: (batch, n, m)
    d2 = (diffs ** 2).sum(dim=(-2, -1))
    
    # mean over group: (batch, n)
    return torch.exp(-d2 / (2.0 * epsilon)).mean(dim=-1)


def compute_graph_laplacian_batched(
    weights: torch.Tensor,  # (batch, n)
    function_values: torch.Tensor,  # (batch, n)
    target_idx: int = 0
) -> torch.Tensor:
    """
    Batched graph Laplacian at the target point.
    Returns L_hat f(R0) for each trial in batch.
    """
    degree = weights.sum(dim=-1)  # (batch,)
    weighted_sum = (weights * function_values).sum(dim=-1)  # (batch,)
    
    # Avoid division by zero
    degree = torch.clamp(degree, min=1e-10)
    
    return weighted_sum / degree - function_values[:, target_idx]


# ------------------------ GPU Experiment Runner ------------------------ #

def run_experiment_gpu(
    ell: int,
    kernel: Literal["euclidean", "min_orbit", "integral"],
    epsilon_values: np.ndarray,
    num_points: int,
    num_trials: int,
    num_group_samples: int,
    device: torch.device,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated experiment runner with batched trials.
    
    Args:
        batch_size: Number of trials to run in parallel. If None, uses all trials.
    """
    logger.info(f"Starting GPU experiment for kernel: {kernel}, ell={ell}, device={device}")
    
    if batch_size is None:
        batch_size = num_trials
    
    all_trial_errors = np.zeros((len(epsilon_values), num_trials), dtype=np.float32)
    
    # Pre-generate shared subgroup elements for G-invariant kernels
    if kernel in ("min_orbit", "integral"):
        subgroup_elements = sample_haar_so2_gpu(num_group_samples, device)
    else:
        subgroup_elements = None
    
    # Target rotation is identity
    target = torch.eye(3, device=device, dtype=torch.float32)
    
    t0 = time.time()
    
    for i, eps in enumerate(tqdm(epsilon_values, desc=f"{kernel} kernel (GPU)", leave=True)):
        eps = float(eps)
        
        # Process trials in batches
        for batch_start in range(0, num_trials, batch_size):
            batch_end = min(batch_start + batch_size, num_trials)
            current_batch_size = batch_end - batch_start
            
            # Sample rotations for all trials in batch: (batch, n, 3, 3)
            # First point is always the identity (target)
            batch_rotations = sample_haar_so3_gpu(current_batch_size * (num_points - 1), device)
            batch_rotations = batch_rotations.reshape(current_batch_size, num_points - 1, 3, 3)
            
            # Prepend identity to each trial
            identity_batch = target.unsqueeze(0).unsqueeze(0).expand(current_batch_size, 1, 3, 3)
            batch_rotations = torch.cat([identity_batch, batch_rotations], dim=1)  # (batch, n, 3, 3)
            
            # Function values: f_ell(R) = R[2,2] for ell=1
            f_vals = batch_rotations[:, :, 2, 2]  # (batch, n)
            
            # Resample subgroup for each batch (optional: could reuse)
            if kernel in ("min_orbit", "integral"):
                K = sample_haar_so2_gpu(num_group_samples, device)
            else:
                K = None
            
            # Compute weights
            if kernel == "euclidean":
                weights = compute_weights_euclidean_batched(batch_rotations, target, eps)
            elif kernel == "min_orbit":
                weights = compute_weights_min_orbit_batched(batch_rotations, target, eps, K)
            elif kernel == "integral":
                weights = compute_weights_integral_batched(batch_rotations, target, eps, K)
            
            # Graph Laplacian
            L_hat_f = compute_graph_laplacian_batched(weights, f_vals, target_idx=0)
            
            # Error = (2/ε) * L_hat f + ell(ell+1)
            errors = (2.0 / eps) * L_hat_f + ell * (ell + 1)
            
            # Store results
            all_trial_errors[i, batch_start:batch_end] = errors.cpu().numpy()
    
    elapsed = time.time() - t0
    logger.info(f"Completed kernel {kernel} in {elapsed:.2f} seconds")
    
    errors_rmse = np.sqrt(np.mean(all_trial_errors ** 2, axis=1))
    
    return epsilon_values, all_trial_errors, errors_rmse


# ------------------------ Multi-GPU Experiment Runner ------------------------ #

def run_experiment_multi_gpu(
    ell: int,
    kernel: Literal["euclidean", "min_orbit", "integral"],
    epsilon_values: np.ndarray,
    num_points: int,
    num_trials: int,
    num_group_samples: int,
    gpu_ids: list,
    trials_per_gpu: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-GPU experiment runner that distributes trials across GPUs.
    
    Each GPU processes a portion of the trials in parallel.
    """
    import concurrent.futures
    
    num_gpus = len(gpu_ids)
    if trials_per_gpu is None:
        trials_per_gpu = num_trials // num_gpus
    
    logger.info(f"Running on {num_gpus} GPUs, ~{trials_per_gpu} trials per GPU")
    
    all_trial_errors = np.zeros((len(epsilon_values), num_trials), dtype=np.float32)
    
    def run_on_gpu(gpu_id: int, trial_start: int, trial_end: int):
        """Run experiment on a specific GPU."""
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        
        n_trials = trial_end - trial_start
        _, trial_errors, _ = run_experiment_gpu(
            ell=ell,
            kernel=kernel,
            epsilon_values=epsilon_values,
            num_points=num_points,
            num_trials=n_trials,
            num_group_samples=num_group_samples,
            device=device,
            batch_size=min(n_trials, 10),  # 100k points with 200 orbit samples needs small batches.
        )
        return trial_start, trial_end, trial_errors
    
    # Distribute trials across GPUs
    trials_per_gpu_list = []
    start = 0
    for i, gpu_id in enumerate(gpu_ids):
        if i == num_gpus - 1:
            end = num_trials  # Last GPU gets remaining trials
        else:
            end = start + (num_trials // num_gpus)
        trials_per_gpu_list.append((gpu_id, start, end))
        start = end
    
    t0 = time.time()
    
    # Run on multiple GPUs in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(run_on_gpu, gpu_id, trial_start, trial_end)
            for gpu_id, trial_start, trial_end in trials_per_gpu_list
        ]
        
        for future in concurrent.futures.as_completed(futures):
            trial_start, trial_end, trial_errors = future.result()
            all_trial_errors[:, trial_start:trial_end] = trial_errors
    
    elapsed = time.time() - t0
    logger.info(f"Multi-GPU completed kernel {kernel} in {elapsed:.2f} seconds")
    
    errors_rmse = np.sqrt(np.mean(all_trial_errors ** 2, axis=1))
    
    return epsilon_values, all_trial_errors, errors_rmse


# ------------------------ Plotting Functions (same as original) ------------------------ #

def fitted_slope_reference_line(
    log_epsilon: np.ndarray,
    log_errors: np.ndarray,
    fit_min: float = -6.0,
    fit_max: float = -3.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit the slope on log(epsilon) in [fit_min, fit_max] and draw over the full plot."""
    fit_mask = (log_epsilon >= fit_min) & (log_epsilon <= fit_max)
    if np.count_nonzero(fit_mask) < 2:
        fit_mask = np.ones_like(log_epsilon, dtype=bool)

    fitted_slope, fitted_intercept = np.polyfit(log_epsilon[fit_mask], log_errors[fit_mask], 1)
    x_ref = log_epsilon
    y_ref = fitted_slope * x_ref + fitted_intercept
    # region agent log
    _agent_debug_log(
        "dash-lines-v1",
        "H5_H6",
        "converges_so3_so2_exp_journal_gpu.py:fitted_slope_reference_line",
        "Fitted dashed reference line",
        {
            "fit_min": float(fit_min),
            "fit_max": float(fit_max),
            "fit_points": int(np.count_nonzero(fit_mask)),
            "fitted_slope": float(fitted_slope),
            "x_start": float(x_ref[0]),
            "x_end": float(x_ref[-1]),
            "y_start": float(y_ref[0]),
            "y_end": float(y_ref[-1]),
        },
    )
    # endregion
    return x_ref, y_ref, float(fitted_slope)


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
    """Create publication-quality log-log plot for a single kernel."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    errors_mean = np.mean(np.abs(all_trial_errors), axis=1)
    errors_std = np.std(np.abs(all_trial_errors), axis=1, ddof=1)
    ci_half_width = 1.96 * errors_std / np.sqrt(num_trials)
    errors_lower = np.maximum(errors_mean - ci_half_width, 1e-300)
    errors_upper = errors_mean + ci_half_width
    
    log_errors_mean = np.log(errors_mean + 1e-300)
    log_errors_lower = np.log(errors_lower + 1e-300)
    log_errors_upper = np.log(errors_upper + 1e-300)

    ax.plot(log_epsilon, log_errors_mean, '-', linewidth=2.5, label='Mean error', color='#1f77b4')
    ax.plot(log_epsilon, log_errors_mean, 'o', markersize=5, color='#1f77b4')
    ax.fill_between(log_epsilon, log_errors_lower, log_errors_upper, alpha=0.2, color='#1f77b4')
    
    y_fit = fixed_slope * log_epsilon + (log_errors_mean[0] - fixed_slope * log_epsilon[0])
    ax.plot(log_epsilon, y_fit, '--', linewidth=2.5, label=f'Slope = {fixed_slope:.2f}', color='#d62728')
    
    ax.set_xlabel(r'$\log(\varepsilon)$', fontsize=16)
    ax.set_ylabel(r'$\log(\mathrm{Error})$', fontsize=16)
    ax.set_xlim([-6, -1])
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')
    
    text_str = f'Trials: {num_trials}\nGroup elements: {num_group_samples}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11, verticalalignment='top',
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
    """Create publication-quality combined log-log plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    colors = {
        "euclidean": "#1f77b4",
        "min_orbit": "#ff7f0e",
        "integral": "#2ca02c"
    }
    
    kernel_display = {
        'euclidean': 'Euclidean',
        'min_orbit': 'Minimum-Orbit',
        'integral': 'Integral'
    }

    kernel_order = ["integral", "min_orbit", "euclidean"]
    fitted_slopes_by_kernel = {}
    
    for kernel_name in kernel_order:
        all_trial_errors = all_trial_errors_by_kernel[kernel_name]
        color = colors.get(kernel_name, 'black')
        display_name = kernel_display.get(kernel_name, kernel_name)
        slope = slopes_by_kernel[kernel_name]
        
        errors_mean = np.mean(np.abs(all_trial_errors), axis=1)
        errors_std = np.std(np.abs(all_trial_errors), axis=1, ddof=1)
        ci_half_width = 1.96 * errors_std / np.sqrt(num_trials)
        errors_lower = np.maximum(errors_mean - ci_half_width, 1e-300)
        errors_upper = errors_mean + ci_half_width
        
        log_errors_mean = np.log(errors_mean + 1e-300)
        log_errors_lower = np.log(errors_lower + 1e-300)
        log_errors_upper = np.log(errors_upper + 1e-300)
        
        ax.plot(log_epsilon, log_errors_mean, '-', linewidth=2.5, color=color,
                label=display_name)
        ax.plot(log_epsilon, log_errors_mean, 'o', markersize=5, color=color)
        ax.fill_between(log_epsilon, log_errors_lower, log_errors_upper, alpha=0.2, color=color)
        
        x_fit, y_fit, fitted_slope = fitted_slope_reference_line(log_epsilon, log_errors_mean)
        curve_on_segment = np.interp(x_fit, log_epsilon, log_errors_mean)
        vertical_gap = np.abs(y_fit - curve_on_segment)
        guide_color = color
        fitted_slopes_by_kernel[kernel_name] = fitted_slope
        dashed_line = ax.plot(
            x_fit,
            y_fit,
            linestyle=(0, (8, 4)),
            linewidth=3.0,
            color=guide_color,
            alpha=1.0,
            zorder=20,
            dash_capstyle="round",
            path_effects=[path_effects.Stroke(linewidth=5.2, foreground="white"), path_effects.Normal()],
        )[0]
        # region agent log
        _agent_debug_log(
            "dash-lines-v1",
            "H3",
            "converges_so3_so2_exp_journal_gpu.py:plot_combined_journal",
            "Drew combined dashed reference line",
            {
                "kernel": kernel_name,
                "linestyle": dashed_line.get_linestyle(),
                "linewidth": float(dashed_line.get_linewidth()),
                "alpha": dashed_line.get_alpha(),
                "zorder": int(dashed_line.get_zorder()),
                "has_white_outline": True,
                "has_slope_text_label": False,
                "fitted_slope": float(fitted_slope),
                "uses_same_color_as_data": True,
                "min_vertical_gap_from_curve": float(np.min(vertical_gap)),
                "median_vertical_gap_from_curve": float(np.median(vertical_gap)),
                "label_at_right_axis_edge": False,
                "segment_points": int(len(x_fit)),
                "x_start": float(x_fit[0]),
                "x_end": float(x_fit[-1]),
            },
        )
        # endregion
    
    ax.set_xlabel(r'$\log(\varepsilon)$', fontsize=18)
    ax.set_ylabel(r'$\log(\mathrm{Error})$', fontsize=18)
    ax.set_xlim([-6, -1])
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fontsize=13)
    
    text_str = (
        f'Trials: {num_trials}\n'
        f'Group elements: {num_group_samples}\n'
        f'Fit slopes on [-6, -3.5]:\n'
        f"Euclidean: {fitted_slopes_by_kernel['euclidean']:.3f}\n"
        f"Minimum: {fitted_slopes_by_kernel['min_orbit']:.3f}\n"
        f"Integral: {fitted_slopes_by_kernel['integral']:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        text_str,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        verticalalignment='top',
        zorder=40,
        bbox=dict(boxstyle='round,pad=0.55', facecolor='#fff7cc', edgecolor='black', linewidth=1.2, alpha=0.96),
    )
    # region agent log
    _agent_debug_log(
        "dash-lines-v1",
        "H7",
        "converges_so3_so2_exp_journal_gpu.py:plot_combined_journal",
        "Added fitted slopes to plot box",
        {
            "box_contains_slopes": True,
            "fit_window": [-6.0, -3.5],
            "slopes": {k: float(v) for k, v in fitted_slopes_by_kernel.items()},
        },
    )
    # endregion
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    png_save_path = os.path.splitext(save_path)[0] + ".png"
    plt.savefig(png_save_path, format='png', dpi=300, bbox_inches='tight')
    # region agent log
    _agent_debug_log(
        "dash-lines-v1",
        "H4",
        "converges_so3_so2_exp_journal_gpu.py:plot_combined_journal",
        "Saved combined plot file",
        {
            "save_path": save_path,
            "exists_after_save": bool(os.path.exists(save_path)),
            "size_bytes_after_save": int(os.path.getsize(save_path)) if os.path.exists(save_path) else None,
            "mtime_after_save": float(os.path.getmtime(save_path)) if os.path.exists(save_path) else None,
            "png_save_path": png_save_path,
            "png_exists_after_save": bool(os.path.exists(png_save_path)),
            "png_size_bytes_after_save": int(os.path.getsize(png_save_path)) if os.path.exists(png_save_path) else None,
        },
    )
    # endregion
    logger.info(f"Combined plot saved as: {save_path}")
    plt.close()


def save_results_to_pickle(
    filepath: str,
    epsilon_values: np.ndarray,
    all_trial_errors_by_kernel: Dict[str, np.ndarray],
    log_epsilon: np.ndarray,
    slopes_by_kernel: Dict[str, float],
    metadata: Dict[str, Any],
) -> None:
    """Save experimental results to pickle file."""
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


# ------------------------ Main ------------------------ #

def main():
    logger.info("Starting SO(3)/SO(2) experiments (GPU-optimized version)")
    
    # Detect available GPUs
    gpus = get_available_gpus()
    if not gpus:
        logger.error("No GPUs available. Please use the CPU version instead.")
        return
    
    # Use all available GPUs (up to 4)
    gpu_ids = [g['id'] for g in gpus[:4]]
    logger.info(f"Using GPUs: {gpu_ids}")
    
    # Single GPU mode if only 1 available
    use_multi_gpu = len(gpu_ids) > 1
    primary_device = torch.device(f'cuda:{gpu_ids[0]}')
    
    # Experiment parameters
    ell = 1
    num_points = 100000
    num_trials = 1000
    num_group_samples = 200

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.environ.get(
        "PAPER_EXP3_OUTPUT_DIR",
        os.path.abspath(os.path.join(script_dir, "..", "outputs", "final_so3_so2_exp3")),
    )
    plot_dir = os.path.join(output_dir, "plots_gpu")
    os.makedirs(plot_dir, exist_ok=True)

    function_name = "P_1_00"
    
    print(f"\n{'='*60}")
    print(f"GPU-Optimized Experiment")
    print(f"GPUs: {gpu_ids}")
    print(f"Parameters: ell={ell}, num_points={num_points}, num_trials={num_trials}")
    print(f"Epsilon range: [{np.min(epsilon_values):.2e}, {np.max(epsilon_values):.2e}]")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Choose runner based on GPU availability
    if use_multi_gpu:
        run_fn = lambda kernel, n_pts: run_experiment_multi_gpu(
            ell=ell,
            kernel=kernel,
            epsilon_values=epsilon_values,
            num_points=n_pts,
            num_trials=num_trials,
            num_group_samples=num_group_samples,
            gpu_ids=gpu_ids,
        )
    else:
        run_fn = lambda kernel, n_pts: run_experiment_gpu(
            ell=ell,
            kernel=kernel,
            epsilon_values=epsilon_values,
            num_points=n_pts,
            num_trials=num_trials,
            num_group_samples=num_group_samples,
            device=primary_device,
            batch_size=10,
        )

    # Euclidean kernel
    logger.info(">>> [1/3] Starting Euclidean kernel experiment")
    eps_euc, all_trials_euc, rmse_euc = run_fn("euclidean", num_points)
    results["euclidean"] = (eps_euc, all_trials_euc, rmse_euc)
    
    plot_single_kernel_journal(
        epsilon_values=eps_euc,
        all_trial_errors=all_trials_euc,
        log_epsilon=log_epsilon_values,
        fixed_slope=fixed_slopes["euclidean"],
        kernel_name="euclidean",
        save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_euclidean_gpu.pdf"),
        num_trials=num_trials,
        num_group_samples=num_group_samples,
    )

    # Minimum-orbit kernel
    logger.info(">>> [2/3] Starting Minimum-orbit kernel experiment")
    eps_min, all_trials_min, rmse_min = run_fn("min_orbit", num_points)
    results["min_orbit"] = (eps_min, all_trials_min, rmse_min)
    
    plot_single_kernel_journal(
        epsilon_values=eps_min,
        all_trial_errors=all_trials_min,
        log_epsilon=log_epsilon_values,
        fixed_slope=fixed_slopes["min_orbit"],
        kernel_name="min_orbit",
        save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_min_orbit_gpu.pdf"),
        num_trials=num_trials,
        num_group_samples=num_group_samples,
    )

    # Integral kernel
    logger.info(">>> [3/3] Starting Integral kernel experiment")
    eps_int, all_trials_int, rmse_int = run_fn("integral", num_points)
    results["integral"] = (eps_int, all_trials_int, rmse_int)
    
    plot_single_kernel_journal(
        epsilon_values=eps_int,
        all_trial_errors=all_trials_int,
        log_epsilon=log_epsilon_values,
        fixed_slope=fixed_slopes["integral"],
        kernel_name="integral",
        save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_integral_gpu.pdf"),
        num_trials=num_trials,
        num_group_samples=num_group_samples,
    )

    # Combined plot
    plot_combined_journal(
        epsilon_values=epsilon_values,
        all_trial_errors_by_kernel={
            "euclidean": all_trials_euc,
            "min_orbit": all_trials_min,
            "integral": all_trials_int
        },
        log_epsilon=log_epsilon_values,
        slopes_by_kernel=fixed_slopes,
        save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_combined_gpu.pdf"),
        num_trials=num_trials,
        num_group_samples=num_group_samples,
    )

    # Save results
    metadata = {
        'ell': ell,
        'num_points': num_points,
        'num_trials': num_trials,
        'num_group_samples': num_group_samples,
        'log_epsilon_min': log_epsilon_min,
        'log_epsilon_max': log_epsilon_max,
        'num_epsilon': num_epsilon,
        'function_name': function_name,
        'euclidean_num_points': num_points,
        'laplacian_scaling': '4/epsilon',
        'exact_value_at_identity': 0.5 * ell * (ell + 1),
        'fixed_slopes': fixed_slopes,
        'gpu_ids': gpu_ids,
        'accelerated': True,
    }
    
    save_results_to_pickle(
        filepath=os.path.join(output_dir, f"so3_so2_{function_name}_results_gpu.pkl"),
        epsilon_values=epsilon_values,
        all_trial_errors_by_kernel={
            "euclidean": all_trials_euc,
            "min_orbit": all_trials_min,
            "integral": all_trials_int
        },
        log_epsilon=log_epsilon_values,
        slopes_by_kernel=fixed_slopes,
        metadata=metadata,
    )

    print(f"\n{'='*60}")
    print("GPU EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"GPUs used: {gpu_ids}")
    print(f"Function: P_1_00 (ell={ell})")
    print(f"Number of points: {num_points}")
    print(f"Number of trials: {num_trials}")
    print(f"Number of group samples: {num_group_samples}")
    print(f"\nOutput: {plot_dir}")
    print(f"{'='*60}\n")

    logger.info("GPU-optimized experiments completed successfully.")


if __name__ == "__main__":
    main()

