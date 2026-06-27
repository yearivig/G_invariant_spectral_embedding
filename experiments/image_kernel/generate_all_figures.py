"""
Generate comparison figures from ALL pre-computed eigenvector results
across all 4 scripts (apply_method, apply_method2, apply_metrhod_l2, apply_EY_method)
plus the new optimized run_experiment results.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
OUTPUT_DIR = os.environ.get("PAPER_IMAGE_OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))
FIGURES_DIR = os.environ.get("PAPER_IMAGE_FIGURES_DIR", os.path.join(BASE_DIR, "figures", "comparison"))
DPI = 600
BASE = os.environ.get(
    "PAPER_IMAGE_LEGACY_RESULTS_DIR",
    os.path.join(REPO_ROOT, "comparing_to_yoel_and_eitan"),
)

_RCPARAMS = {
    'font.family': 'serif',
    'text.usetex': True,
    'axes.labelsize': 12,
    'axes.titlesize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 1,
}
_W = 6.775
_H = (5**0.5 - 1) / 2 * _W
RC = {**_RCPARAMS, 'figure.figsize': (_W / 2, _H / 2)}

os.makedirs(FIGURES_DIR, exist_ok=True)

# Ground-truth angles (old dataset, in degrees)
angles_csv_path = os.environ.get(
    "PAPER_OLD_ANGLES_CSV",
    os.path.join(REPO_ROOT, "data", "angles.csv"),
)
if os.path.exists(angles_csv_path):
    df = pd.read_csv(angles_csv_path)
    df = df.drop(df.columns[0], axis=1)
    old_angles_deg = np.array(df.values.flatten())[2:]
    old_colors = np.sin(old_angles_deg * np.pi / 180)
else:
    old_angles_deg = None
    old_colors = None

# Ground-truth angles (new dataset, in radians)
new_angles_path = os.path.join(OUTPUT_DIR, 'ground_truth_angles.pkl')
if os.path.exists(new_angles_path):
    with open(new_angles_path, 'rb') as f:
        new_angles_rad = pickle.load(f)
    new_colors = np.sin(new_angles_rad)
else:
    new_angles_rad = None
    new_colors = None


def make_color(n, angles_type='old'):
    """Generate color array for scatter plot, always matching data size."""
    if angles_type == 'new' and new_colors is not None and len(new_colors) >= n:
        return new_colors[:n]
    elif angles_type == 'old' and old_colors is not None and len(old_colors) >= n:
        return old_colors[:n]
    return np.linspace(0, 1, n)


def plot_and_save(eigvecs, label, filename, angles_type='old'):
    x = np.real(eigvecs[1])
    y = np.real(eigvecs[2])
    c = make_color(len(x), angles_type)

    with plt.rc_context(rc=RC):
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=c, cmap='rainbow', marker='o', s=8, linewidths=0)
        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        path = os.path.join(FIGURES_DIR, filename.replace('.', '_') + '.pdf')
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f'  {path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  1. apply_method.py outputs (82x82, bw=0.5, SNR sweep, min/mean/bispectrum)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. apply_method.py — FFB(82x82), bw=0.5, old dataset")
print("   Methods: min, mean, bispectrum | SNR: 0, 1, 5, 10")
print("=" * 60)

for snr in [0, 1, 5, 10]:
    for method in ['min', 'mean', 'bispectrum']:
        pkl = f'{BASE}/outputs/eigvectors_ell_max_10_tol_0.01_{method}_molecule_david_rotated_bw_0.5_snr_{snr}.pkl'
        if not os.path.exists(pkl):
            continue
        with open(pkl, 'rb') as f:
            v = pickle.load(f)
        label = f'apply\\_method: {method}, SNR={snr}'
        plot_and_save(v, label, f'apply_method_{method}_snr{snr}')

# ═══════════════════════════════════════════════════════════════════════════════
#  2. apply_metrhod_l2.py outputs (82x82, bw=50, pre-rotated data, min only)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. apply_metrhod_l2.py — FFB(82x82), bw=50, pre-rotated, min")
print("=" * 60)

pkl = f'{BASE}/data/eigvectors_ell_10_tol_0.001_integral_molecule_david.pkl'
if os.path.exists(pkl):
    with open(pkl, 'rb') as f:
        v = pickle.load(f)
    plot_and_save(v, r'apply\_metrhod\_l2: min, bw=50', 'metrhod_l2_min_bw50')

# ═══════════════════════════════════════════════════════════════════════════════
#  3. apply_EY_method.py outputs (128x128, bw=50, EY dataset, min/mean/none)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. apply_EY_method.py — FFB(128x128), bw=50, EY dataset")
print("   Methods: min, mean, none")
print("=" * 60)

for method in ['min', 'mean', 'none']:
    pkl = f'{BASE}/data/198/eigvectors_ell_10_tol_0.001_{method}_198.pkl'
    if not os.path.exists(pkl):
        continue
    with open(pkl, 'rb') as f:
        v = pickle.load(f)
    plot_and_save(v, f'apply\\_EY: {method}, bw=50', f'EY_{method}_bw50')

# Also check other data/ pkls
for name in ['integral_100', 'min_100', 'mean_100', 'none_100',
             'min_v2', 'mean_v2', 'none_v2']:
    pkl = f'{BASE}/data/eigvectors_ell_10_tol_0.001_{name}.pkl'
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            v = pickle.load(f)
        plot_and_save(v, f'data: {name}', f'data_{name}')

# ═══════════════════════════════════════════════════════════════════════════════
#  4. run_experiment.py outputs (new dataset, 79x79, 1998 pts, optimized)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. run_experiment.py — FFB(79x79), new dataset, N=1998")
print("   Methods: min, mean, bispectrum | SNR: 0")
print("=" * 60)

for method in ['min', 'mean', 'bispectrum']:
    pkl = os.path.join(OUTPUT_DIR, f'eigvectors_ell10_{method}_bw0.5_snr0.pkl')
    if not os.path.exists(pkl):
        continue
    with open(pkl, 'rb') as f:
        v = pickle.load(f)
    label = f'NEW: {method}, N=1998'
    plot_and_save(v, label, f'new_{method}_snr0', angles_type='new')

# ═══════════════════════════════════════════════════════════════════════════════
#  5. Summary grid: best from each script side by side
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. Generating comparison grid")
print("=" * 60)

grid_items = []
grid_specs = [
    ('apply_method min SNR=0',   f'{BASE}/outputs/eigvectors_ell_max_10_tol_0.01_min_molecule_david_rotated_bw_0.5_snr_0.pkl', 'old'),
    ('apply_method mean SNR=0',  f'{BASE}/outputs/eigvectors_ell_max_10_tol_0.01_mean_molecule_david_rotated_bw_0.5_snr_0.pkl', 'old'),
    ('apply_method bisp SNR=0',  f'{BASE}/outputs/eigvectors_ell_max_10_tol_0.01_bispectrum_molecule_david_rotated_bw_0.5_snr_0.pkl', 'old'),
    ('EY min bw=50',             f'{BASE}/data/198/eigvectors_ell_10_tol_0.001_min_198.pkl', 'old'),
    ('L2 min bw=50',             f'{BASE}/data/eigvectors_ell_10_tol_0.001_integral_molecule_david.pkl', 'old'),
    ('NEW min N=1998',           os.path.join(OUTPUT_DIR, 'eigvectors_ell10_min_bw0.5_snr0.pkl'), 'new'),
    ('NEW mean N=1998',          os.path.join(OUTPUT_DIR, 'eigvectors_ell10_mean_bw0.5_snr0.pkl'), 'new'),
    ('NEW bisp N=1998',          os.path.join(OUTPUT_DIR, 'eigvectors_ell10_bispectrum_bw0.5_snr0.pkl'), 'new'),
]

for label, pkl, atype in grid_specs:
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            v = pickle.load(f)
        grid_items.append((label, v, atype))

if grid_items:
    ncols = 4
    nrows = (len(grid_items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (label, v, atype) in enumerate(grid_items):
        r, c_idx = divmod(idx, ncols)
        ax = axes[r, c_idx]
        x = np.real(v[1])
        y = np.real(v[2])
        colors = make_color(len(x), atype)
        ax.scatter(x, y, c=colors, cmap='rainbow', marker='o', s=5, linewidths=0)
        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')

    for idx in range(len(grid_items), nrows * ncols):
        r, c_idx = divmod(idx, ncols)
        axes[r, c_idx].set_visible(False)

    plt.tight_layout()
    grid_path = os.path.join(FIGURES_DIR, 'comparison_grid.pdf')
    fig.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Grid saved: {grid_path}')

print("\nDone! All figures are in:", os.path.realpath(FIGURES_DIR))
