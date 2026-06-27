"""
Utility script to regenerate publication-quality plots from saved pickle files.
This allows you to modify plot styling and formatting without re-running expensive experiments.

Usage:
    python replot_from_pickle.py [pickle_file_path]

If no path is provided, it will look for the default results file in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import sys
from typing import Dict, Any

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


def load_results_from_pickle(filepath: str) -> Dict[str, Any]:
    """Load experimental results from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Results loaded from: {filepath}")
    return data


def plot_single_kernel_journal(
    epsilon_values: np.ndarray,
    errors: np.ndarray,
    log_epsilon: np.ndarray,
    log_errors: np.ndarray,
    slope: float,
    kernel_name: str,
    save_path: str,
) -> None:
    """Create publication-quality log-log plot for a single kernel."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot data
    ax.plot(log_epsilon, log_errors, '-', linewidth=2.5, label='Computed error', color='#1f77b4')
    ax.plot(log_epsilon, log_errors, 'o', markersize=5, color='#1f77b4')
    
    # Fit line only on variance regime (before minimum)
    min_err_index = np.argmin(errors)
    if min_err_index > 1:
        x_fit = log_epsilon[:min_err_index]
        y_fit = slope * x_fit + (log_errors[0] - slope * log_epsilon[0])
        ax.plot(x_fit, y_fit, '--', linewidth=2.5, 
                label=f'Slope = {slope:.3f}', color='#d62728')
    
    # Labels and formatting
    ax.set_xlabel(r'$\log(\varepsilon)$', fontsize=16)
    ax.set_ylabel(r'$\log(\mathrm{Error})$', fontsize=16)
    
    # Kernel name formatting
    kernel_display = {
        'euclidean': 'Euclidean Kernel',
        'min_orbit': 'Minimum-Orbit Kernel',
        'integral': 'Integral Kernel'
    }
    ax.set_title(kernel_display.get(kernel_name, kernel_name), fontsize=16, pad=15)
    
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {save_path}")
    plt.close()


def plot_combined_journal(
    epsilon_values: np.ndarray,
    errors_by_kernel: Dict[str, np.ndarray],
    log_epsilon: np.ndarray,
    log_errors_by_kernel: Dict[str, np.ndarray],
    slopes_by_kernel: Dict[str, float],
    save_path: str,
) -> None:
    """Create publication-quality combined log-log plot for all kernels."""
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

    # Plot each kernel
    for kernel_name, log_errs in log_errors_by_kernel.items():
        color = colors.get(kernel_name, 'black')
        display_name = kernel_display.get(kernel_name, kernel_name)
        slope = slopes_by_kernel[kernel_name]
        
        # Data points and line
        ax.plot(log_epsilon, log_errs, '-', linewidth=2.5, 
                color=color, label=f'{display_name} (slope = {slope:.2f})')
        ax.plot(log_epsilon, log_errs, 'o', markersize=5, color=color)
        
        # Fit line on variance regime
        errors = errors_by_kernel[kernel_name]
        min_err_index = np.argmin(errors)
        if min_err_index > 1:
            x_fit = log_epsilon[:min_err_index]
            y_fit = slope * x_fit + (log_errs[0] - slope * log_epsilon[0])
            ax.plot(x_fit, y_fit, '--', linewidth=2.0, color=color, alpha=0.7)
    
    # Labels and formatting
    ax.set_xlabel(r'$\log(\varepsilon)$', fontsize=18)
    ax.set_ylabel(r'$\log(\mathrm{Error})$', fontsize=18)
    ax.set_title(r'Laplacian Error Analysis: $SO(3)/SO(2)$ with $P_1^{00}$', 
                 fontsize=18, pad=20)
    
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Combined plot saved: {save_path}")
    plt.close()


def regenerate_all_plots(pickle_path: str, output_dir: str = None) -> None:
    """Load results and regenerate all plots."""
    
    # Load data
    data = load_results_from_pickle(pickle_path)
    
    # Extract data
    epsilon_values = data['epsilon_values']
    errors_by_kernel = data['errors_by_kernel']
    log_epsilon = data['log_epsilon']
    log_errors_by_kernel = data['log_errors_by_kernel']
    slopes_by_kernel = data['slopes_by_kernel']
    metadata = data['metadata']
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(pickle_path)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    function_name = metadata.get('function_name', 'P_1_00')
    
    print(f"\n{'='*60}")
    print("REGENERATING PLOTS FROM SAVED DATA")
    print(f"{'='*60}")
    print(f"Function: {function_name}")
    print(f"ell: {metadata.get('ell', 'N/A')}")
    print(f"Number of points: {metadata.get('num_points', 'N/A')}")
    print(f"Number of trials: {metadata.get('num_trials', 'N/A')}")
    print(f"Output directory: {plot_dir}")
    print(f"{'='*60}\n")
    
    # Generate individual kernel plots
    for kernel_name in errors_by_kernel.keys():
        print(f"Generating plot for {kernel_name} kernel...")
        plot_single_kernel_journal(
            epsilon_values=epsilon_values,
            errors=errors_by_kernel[kernel_name],
            log_epsilon=log_epsilon,
            log_errors=log_errors_by_kernel[kernel_name],
            slope=slopes_by_kernel[kernel_name],
            kernel_name=kernel_name,
            save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_{kernel_name}_journal.pdf"),
        )
    
    # Generate combined plot
    print("Generating combined plot...")
    plot_combined_journal(
        epsilon_values=epsilon_values,
        errors_by_kernel=errors_by_kernel,
        log_epsilon=log_epsilon,
        log_errors_by_kernel=log_errors_by_kernel,
        slopes_by_kernel=slopes_by_kernel,
        save_path=os.path.join(plot_dir, f"so3_so2_{function_name}_combined_journal.pdf"),
    )
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Convergence slopes (log-log):")
    for kernel_name, slope in slopes_by_kernel.items():
        print(f"  {kernel_name:20s}: {slope:.4f}")
    print(f"{'='*60}\n")
    print("✓ All plots regenerated successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate publication-quality plots from saved pickle files.'
    )
    parser.add_argument(
        'pickle_file',
        nargs='?',
        default='so3_so2_P_1_00_results.pkl',
        help='Path to the pickle file containing experimental results'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default=None,
        help='Output directory for plots (default: same directory as pickle file)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pickle_file):
        print(f"Error: File not found: {args.pickle_file}")
        print("\nSearching for pickle files in current directory...")
        pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if pkl_files:
            print("Found:")
            for f in pkl_files:
                print(f"  - {f}")
        else:
            print("No pickle files found in current directory.")
        sys.exit(1)
    
    regenerate_all_plots(args.pickle_file, args.output_dir)


if __name__ == "__main__":
    main()

