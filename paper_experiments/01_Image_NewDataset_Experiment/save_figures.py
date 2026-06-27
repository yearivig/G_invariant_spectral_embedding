"""
Generate publication-quality figures from eigenvector .pkl files.
Adapted from test_results_script.py for the new dataset (angles in radians).
"""
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

FIGURES_PATH = 'figures/'
OUTPUT_DIR = 'outputs/'
DPI = 600

_RCPARAMS_LATEX = {
    'font.family': 'serif',
    'text.usetex': True,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'axes.prop_cycle': matplotlib.pyplot.cycler(
        'color', ['#ff7d66', '#ffdc30', '#40a0cc', '#529915', '#8b8b8b']
    ) + matplotlib.pyplot.cycler('marker', ['d', 's', 'o', r'$\clubsuit$', '>']),
    'lines.markersize': 9,
    'lines.markeredgewidth': 0.75,
    'lines.markeredgecolor': 'k',
    'grid.color': '#C0C0C0',
    'legend.fancybox': True,
    'legend.framealpha': 0.8,
    'axes.linewidth': 1,
}

_PAGE_WIDTH = 6.775
_GOLDEN = (5 ** 0.5 - 1) / 2
RCPARAMS = {**_RCPARAMS_LATEX, 'figure.figsize': (_PAGE_WIDTH / 2, _GOLDEN * _PAGE_WIDTH / 2)}


def save_pdf(fig, name):
    os.makedirs(FIGURES_PATH, exist_ok=True)
    fname = os.path.join(FIGURES_PATH, name.replace('.', '_') + '.pdf')
    fig.savefig(fname, dpi=DPI, bbox_inches='tight')
    print(f'  Saved: {os.path.realpath(fname)}')


def plot_2d(eigvecs, angles_rad, title, save_name):
    x = np.real(eigvecs[1])
    y = np.real(eigvecs[2])
    colors = np.sin(angles_rad[:len(x)])

    with plt.rc_context(rc=RCPARAMS):
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=colors, cmap='rainbow', marker='o', s=10)
        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        save_pdf(fig, save_name)
        plt.close(fig)


def plot_3d(eigvecs, angles_rad, title, save_name):
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print(f'  Skipping 3D ({title}) -- mpl_toolkits.mplot3d unavailable')
        return

    x = np.real(eigvecs[1])
    y = np.real(eigvecs[2])
    z = np.real(eigvecs[3])
    colors = np.sin(angles_rad[:len(x)])

    with plt.rc_context(rc=RCPARAMS):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=colors, cmap='rainbow', marker='o', s=10, depthshade=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.grid(False)
        plt.tight_layout()
        save_pdf(fig, save_name)
        plt.close(fig)


def main():
    angles_path = os.path.join(OUTPUT_DIR, 'ground_truth_angles.pkl')
    if not os.path.exists(angles_path):
        print(f"Error: {angles_path} not found. Run run_experiment.py first.")
        return
    with open(angles_path, 'rb') as f:
        angles = pickle.load(f)
    print(f"Loaded {len(angles)} ground-truth angles (radians)")

    snr_levels = [0]
    methods = ['min', 'mean', 'bispectrum']

    for snr in snr_levels:
        for method in methods:
            pkl_path = os.path.join(
                OUTPUT_DIR, f'eigvectors_ell10_{method}_bw0.5_snr{snr}.pkl')
            if not os.path.exists(pkl_path):
                print(f"Skipping {method}/snr{snr} -- {pkl_path} not found")
                continue

            print(f"\nProcessing: {method}, SNR={snr}")
            with open(pkl_path, 'rb') as f:
                eigvecs = pickle.load(f)
            print(f"  Eigenvectors shape: {eigvecs.shape}")

            label = f'{method} (SNR={snr})'
            tag = f'{method}_snr{snr}'

            plot_2d(eigvecs, angles, label, f'2d_{tag}')
            plot_3d(eigvecs, angles, label, f'3d_{tag}')

    print("\nAll figures saved!")


if __name__ == "__main__":
    main()
