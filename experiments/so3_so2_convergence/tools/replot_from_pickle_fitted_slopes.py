"""Replot saved pickle results with fitted- and predicted-slope reference lines.

Differences vs. ``replot_from_pickle.py``:

- The x-axis is restricted to ``[-4, max(log_epsilon)]`` (log-epsilon range).
- Two reference lines per kernel, both drawn across the full plot window and
  sharing the same intercept:
    * Dashed line for the FITTED log-log slope: the OLS best-fit line
      ``y = slope * x + intercept`` computed on the per-kernel fit window
      (``euclidean`` fits from ``-3`` up to the mean-|error| minimum;
      ``min_orbit`` and ``integral`` fit from ``-4`` up to the minimum).
      By construction it lies on top of the empirical curve in the fit window.
    * Dotted line for the PREDICTED (theory) slope, using the FITTED
      intercept: ``y = predicted_slope * x + fitted_intercept``. Both lines
      therefore start from the same y-intercept and diverge only because of
      differing slopes; the vertical gap between them at any ``x`` is
      ``(fitted_slope - predicted_slope) * x``.
- Legend labels read ``"<Kernel> kernel (fitted slope = <value>)"``.
- Plots are written to a ``plots_fitted_slopes/`` folder next to the pickle
  (unless ``-o/--output-dir`` is passed).

Usage
-----
    python replot_from_pickle_fitted_slopes.py [pickle_file] [-o OUTPUT_DIR]

If ``pickle_file`` is omitted, defaults to ``so3_so2_P_1_00_results.pkl`` in
the current directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from converges_so3_so2 import (
    _KERNEL_COLOR,
    _KERNEL_PLOT_ORDER,
    _log_error_ci95_band,
    _log_mae,
    load_results_from_pickle,
)


# ---------------------------------------------------------------- #
# Matplotlib defaults (mirror the experiment script)
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SO3_SO2_replot_fitted_slopes")


# Lower bound of the x-axis window (log-epsilon). Independent of the OLS
# fit windows in ``_KERNEL_FIT_X_LOWER``: the plot shows a wider x-range
# than the fit uses, and the fitted reference lines are just extrapolated
# leftward outside the fit window.
X_LOWER: float = -4.0


# Color of the PREDICTED (theory) slope reference line. Kept separate from
# the kernel colors so that when the fitted and predicted slopes are
# (nearly) equal (min_orbit, integral) the two dashed/dotted lines produce
# a two-color overlap and both remain identifiable.
PREDICTED_LINE_COLOR: str = "black"


# Legend labels for the fitted-slope replots. The `main` plot legend uses
# these plus the fitted slope value, per the requested format.
_KERNEL_LEGEND_LABEL: Dict[str, str] = {
    "euclidean": "Euclidean kernel",
    "min_orbit": "Minimum kernel",
    "integral": "Integral kernel",
}


# Per-kernel lower cutoff of the OLS fit window (log-epsilon). Euclidean's
# small-eps regime kicks in later (variance blow-up is sharper), so we drop
# the tail below -3 for it; the G-invariant kernels use the full tail down
# to -4 (the leftmost grid point) for a cleaner slope estimate.
_KERNEL_FIT_X_LOWER: Dict[str, float] = {
    "euclidean": -3.0,
    "min_orbit": -4.0,
    "integral": -4.0,
}


# Optional per-kernel intercept adjustment (added to the OLS intercept). Use
# this to nudge the fitted + predicted reference lines up/down together so
# the fitted dashed line visually overlaps the empirical curve on a wider
# range. It does not affect the reported fitted slope; the two lines still
# share a common intercept ``fitted_intercept + shift``.
_KERNEL_INTERCEPT_SHIFT: Dict[str, float] = {
    "euclidean": 0.05,
    "min_orbit": 0.0,
    "integral": 0.0,
}


def compute_fitted_loglog_line(
    all_trial_errors_by_kernel: Dict[str, np.ndarray],
    log_epsilon: np.ndarray,
    x_lower_by_kernel: Optional[Dict[str, float]] = None,
    default_x_lower: float = X_LOWER,
) -> Dict[str, Tuple[float, float]]:
    """OLS log-log ``(slope, intercept)`` per kernel on the per-kernel fit window
    ``x_lower_by_kernel[kernel] <= log_epsilon < log_epsilon[argmin(mean|error|)]``.

    Returns the intercept too so the fitted line can be drawn as
    ``y = slope * x + intercept`` and lies on top of the empirical curve
    inside the fit window.
    """
    if x_lower_by_kernel is None:
        x_lower_by_kernel = {}
    fits: Dict[str, Tuple[float, float]] = {}
    for kernel_name, trials in all_trial_errors_by_kernel.items():
        errors_mean = np.mean(np.abs(trials), axis=1)
        log_errors_mean = np.log(errors_mean + 1e-300)
        min_idx = int(np.argmin(errors_mean))
        x_lower = float(x_lower_by_kernel.get(kernel_name, default_x_lower))
        # Fit region: strictly left of the min AND at/above the kernel's x_lower.
        fit_mask = (log_epsilon >= x_lower) & (np.arange(len(log_epsilon)) < min_idx)
        if int(np.count_nonzero(fit_mask)) >= 2:
            slope, intercept = np.polyfit(
                log_epsilon[fit_mask], log_errors_mean[fit_mask], 1
            )
            fits[kernel_name] = (float(slope), float(intercept))
        else:
            fits[kernel_name] = (float("nan"), float("nan"))
    return fits


def _ols_line(
    x_start: float,
    x_end: float,
    slope: float,
    intercept: float,
    num: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dense line ``y = slope * x + intercept`` on ``[x_start, x_end]``.

    Used for both the fitted line (with the OLS intercept) and the predicted
    line (with the same fitted intercept so both share ``y(0)``).
    """
    x = np.linspace(x_start, x_end, num)
    y = slope * x + intercept
    return x, y


def _window_mask(log_epsilon: np.ndarray) -> np.ndarray:
    """Boolean mask selecting entries with ``log_epsilon >= X_LOWER``."""
    return log_epsilon >= X_LOWER


def plot_single_kernel_fitted(
    log_epsilon: np.ndarray,
    log_errors: np.ndarray,
    trial_errors: np.ndarray,
    fitted_slope: float,
    fitted_intercept: float,
    predicted_slope: float,
    kernel_name: str,
    save_path: str,
) -> None:
    """Per-kernel plot with dashed OLS fitted line and dotted predicted-slope line."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    color = _KERNEL_COLOR.get(kernel_name, "#1f77b4")
    display = _KERNEL_LEGEND_LABEL.get(kernel_name, kernel_name)

    mask = _window_mask(log_epsilon)
    x_plot = log_epsilon[mask]
    y_plot = log_errors[mask]
    x_upper = float(np.max(log_epsilon))

    fitted_label = (
        f"{display} (fitted slope = {fitted_slope:.3f})"
        if np.isfinite(fitted_slope)
        else f"{display} (fitted slope = n/a)"
    )
    ax.plot(x_plot, y_plot, "-", linewidth=2.5, color=color, label=fitted_label)
    ax.plot(x_plot, y_plot, "o", markersize=5, color=color)

    band = _log_error_ci95_band(trial_errors)
    if band is not None:
        log_lo, log_hi = band
        ax.fill_between(x_plot, log_lo[mask], log_hi[mask], alpha=0.2, color=color)

    # Reference lines span the full plot window [X_LOWER, x_upper] and share
    # the same y-intercept ``fitted_intercept``. The predicted line is drawn
    # in ``PREDICTED_LINE_COLOR`` so it stays visible when its slope is
    # (nearly) equal to the fitted one.
    if np.isfinite(fitted_slope) and np.isfinite(fitted_intercept):
        xf, yf = _ols_line(X_LOWER, x_upper, fitted_slope, fitted_intercept)
        ax.plot(
            xf, yf, "--", linewidth=2.0, color=color, alpha=0.95, zorder=5,
            label="Fitted slope",
        )
        if np.isfinite(predicted_slope):
            xp, yp = _ols_line(X_LOWER, x_upper, predicted_slope, fitted_intercept)
            ax.plot(
                xp, yp, ":", linewidth=2.5, color=PREDICTED_LINE_COLOR,
                alpha=0.95, zorder=6,
                label=f"Predicted slope = {predicted_slope:.2f}",
            )

    ax.set_xlabel(r"$\log(\varepsilon)$", fontsize=16)
    ax.set_ylabel(r"$\log(\mathrm{Error})$", fontsize=16)
    ax.set_xlim([X_LOWER, x_upper])
    ax.set_title(display, fontsize=16, pad=15)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(loc="best", framealpha=0.95, edgecolor="gray")

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    logger.info("Plot saved: %s", save_path)
    plt.close(fig)


def plot_combined_fitted(
    log_epsilon: np.ndarray,
    errors_by_kernel: Dict[str, np.ndarray],
    log_errors_by_kernel: Dict[str, np.ndarray],
    fitted_fits_by_kernel: Dict[str, Tuple[float, float]],
    predicted_slopes_by_kernel: Dict[str, float],
    save_path: str,
) -> None:
    """Combined plot with per-kernel dashed OLS fitted line and dotted predicted-slope line.

    The legend shows one entry per kernel of the form
    ``"<Kernel> kernel (fitted slope = <value>)"`` plus two style keys at the
    end explaining what the dashed and dotted overlays represent.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ordered = [k for k in _KERNEL_PLOT_ORDER if k in log_errors_by_kernel]
    ordered += [k for k in log_errors_by_kernel if k not in ordered]

    mask = _window_mask(log_epsilon)
    x_window = log_epsilon[mask]
    x_upper = float(np.max(log_epsilon))

    for kernel_name in ordered:
        color = _KERNEL_COLOR.get(kernel_name, "black")
        display = _KERNEL_LEGEND_LABEL.get(kernel_name, kernel_name)
        log_errs = log_errors_by_kernel[kernel_name]
        trial_errors = errors_by_kernel[kernel_name]
        fitted_slope, fitted_intercept = fitted_fits_by_kernel.get(
            kernel_name, (float("nan"), float("nan"))
        )
        predicted_slope = predicted_slopes_by_kernel.get(kernel_name, float("nan"))

        y_window = log_errs[mask]
        label = (
            f"{display} (fitted slope = {fitted_slope:.3f})"
            if np.isfinite(fitted_slope)
            else f"{display} (fitted slope = n/a)"
        )
        ax.plot(x_window, y_window, "-", linewidth=2.5, color=color, label=label)
        ax.plot(x_window, y_window, "o", markersize=5, color=color)

        band = _log_error_ci95_band(trial_errors)
        if band is not None:
            log_lo, log_hi = band
            ax.fill_between(x_window, log_lo[mask], log_hi[mask], alpha=0.2, color=color)

        if np.isfinite(fitted_slope) and np.isfinite(fitted_intercept):
            xf, yf = _ols_line(X_LOWER, x_upper, fitted_slope, fitted_intercept)
            ax.plot(xf, yf, "--", linewidth=2.0, color=color, alpha=0.95, zorder=5)
            if np.isfinite(predicted_slope):
                xp, yp = _ols_line(X_LOWER, x_upper, predicted_slope, fitted_intercept)
                ax.plot(
                    xp, yp, ":", linewidth=2.5, color=PREDICTED_LINE_COLOR,
                    alpha=0.95, zorder=6,
                )

    # Style-only proxy handles to explain what dashed vs. dotted mean. The
    # dashed proxy uses gray since the actual dashed lines are per-kernel
    # colored; the dotted proxy matches the black predicted line.
    style_handles = [
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=2.0, label="Fitted slope"),
        plt.Line2D(
            [0], [0], color=PREDICTED_LINE_COLOR, linestyle=":", linewidth=2.5,
            label="Predicted slope",
        ),
    ]
    kernel_handles, kernel_labels = ax.get_legend_handles_labels()
    handles = kernel_handles + style_handles
    labels = kernel_labels + [h.get_label() for h in style_handles]

    ax.set_xlabel(r"$\log(\varepsilon)$", fontsize=18)
    ax.set_ylabel(r"$\log(\mathrm{Error})$", fontsize=18)
    ax.set_xlim([X_LOWER, x_upper])
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(handles, labels, loc="best", framealpha=0.95, edgecolor="gray", fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    logger.info("Combined plot saved: %s", save_path)
    plt.close(fig)


def regenerate_fitted_slope_plots(
    pickle_path: str,
    output_dir: Optional[str] = None,
) -> None:
    """Reload a results pickle and render fitted-slope replots."""
    data = load_results_from_pickle(pickle_path)
    trial_dict: Dict[str, np.ndarray] = data["all_trial_errors_by_kernel"]
    log_epsilon: np.ndarray = np.asarray(data["log_epsilon"])
    metadata: Dict[str, Any] = data.get("metadata", {}) or {}

    function_name = metadata.get("function_name", "P_1_00")
    predicted_slopes = dict(metadata.get("fixed_slopes") or {})

    # Recompute (slope, intercept) fresh from the trial data with per-kernel
    # fit windows so the OLS slope reflects the small-eps regime we actually
    # want to characterize:
    #   - euclidean: fit on [-3, x_min)   (drop tail below -3)
    #   - min_orbit: fit on [-4, x_min)   (use the full small-eps tail)
    #   - integral:  fit on [-4, x_min)   (use the full small-eps tail)
    fitted_fits_raw = compute_fitted_loglog_line(
        trial_dict, log_epsilon, x_lower_by_kernel=_KERNEL_FIT_X_LOWER,
    )
    # Apply the per-kernel intercept shift AFTER the fit so it only affects
    # where the dashed/dotted reference lines are drawn, not the reported
    # slope value. Fitted and predicted lines still share this shifted
    # intercept, so their crossing point simply moves vertically.
    fitted_fits: Dict[str, Tuple[float, float]] = {}
    for kernel_name, (slope, intercept) in fitted_fits_raw.items():
        shift = float(_KERNEL_INTERCEPT_SHIFT.get(kernel_name, 0.0))
        fitted_fits[kernel_name] = (slope, intercept + shift)
    fitted_slopes = {k: s for k, (s, _b) in fitted_fits.items()}

    log_errors_by_kernel = {k: _log_mae(v) for k, v in trial_dict.items()}

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(pickle_path))
    plot_dir = os.path.join(output_dir, "plots_fitted_slopes")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("REGENERATING FITTED-SLOPE PLOTS FROM SAVED DATA")
    print(f"{'=' * 60}")
    print(f"Function:         {function_name}")
    print(f"ell:              {metadata.get('ell', 'N/A')}")
    print(f"Number of points: {metadata.get('num_points', 'N/A')}")
    print(f"Number of trials: {metadata.get('num_trials', 'N/A')}")
    print(f"x-window:         [{X_LOWER}, {float(np.max(log_epsilon)):.3f}]")
    print(f"Output directory: {plot_dir}")
    print(f"{'=' * 60}\n")

    for kernel_name, trials in trial_dict.items():
        slope, intercept = fitted_fits.get(kernel_name, (float("nan"), float("nan")))
        plot_single_kernel_fitted(
            log_epsilon=log_epsilon,
            log_errors=log_errors_by_kernel[kernel_name],
            trial_errors=trials,
            fitted_slope=slope,
            fitted_intercept=intercept,
            predicted_slope=predicted_slopes.get(kernel_name, float("nan")),
            kernel_name=kernel_name,
            save_path=os.path.join(
                plot_dir, f"so3_so2_{function_name}_{kernel_name}.pdf"
            ),
        )

    plot_combined_fitted(
        log_epsilon=log_epsilon,
        errors_by_kernel=trial_dict,
        log_errors_by_kernel=log_errors_by_kernel,
        fitted_fits_by_kernel=fitted_fits,
        predicted_slopes_by_kernel=predicted_slopes,
        save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_combined.pdf"),
    )

    print("\nAll fitted-slope plots regenerated successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replot saved paired-experiment results with dashed fitted-slope "
            "and dotted predicted-slope reference lines that intersect each "
            "kernel's mean-|error| minimum. Plots go to plots_fitted_slopes/."
        )
    )
    parser.add_argument(
        "pickle_file",
        nargs="?",
        default="so3_so2_P_1_00_results.pkl",
        help="Path to the pickle file containing experimental results.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for plots (default: directory of the pickle file).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pickle_file):
        print(f"Error: file not found: {args.pickle_file}", file=sys.stderr)
        pkl_files = [f for f in os.listdir(".") if f.endswith(".pkl")]
        if pkl_files:
            print("\nPickle files in the current directory:", file=sys.stderr)
            for f in pkl_files:
                print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    regenerate_fitted_slope_plots(args.pickle_file, args.output_dir)


if __name__ == "__main__":
    main()
