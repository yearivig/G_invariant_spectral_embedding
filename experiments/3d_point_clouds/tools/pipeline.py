"""
One run of the Section 5.1 experiment: point clouds -> kernel -> Algorithm 1 -> saved embedding.

The embedding is saved as a pickle [phi_1, ..., phi_m] (tools/plot_embedding.py
plots the first two), named as the files of Figure 4:

  <name>_NOP:<n>_IM:<kernel>_M:rotation_BW:<eps>_IC:<bool>_AS:<bool>_<tag>:<noise>_LT:RWGL.pkl

with "_MIN:kabsch" and/or "_G:<rotations>" appended when those differ from the
defaults (grid, 600), so default runs keep the paper's names. A .json with the
same stem records every setting, the seed and lambda_0..lambda_m.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import kernels
import pointclouds
from spectral_embedding import spectral_embedding


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: str
    num_points: int                      # n, the number of point clouds (NOP)
    kernel: str                          # min, integral, invariant_features or none (IM)
    bandwidth: float                     # epsilon (BW)
    is_centered: bool = False            # IC
    add_stationary: bool = False         # AS: the 80 stationary points
    snr_db: float = 0.0                  # noise SNR in dB, 0 = clean
    noise_tag: str = "SNR"               # filename label only: "AN" for the paper's clean files
    min_method: str = "grid"             # grid or kabsch, minimum kernel only
    num_rotations: int = kernels.NUM_ROTATIONS
    seed: int = 0
    m: int = 2                           # embedding dimension
    save_folder: str = "results"
    save_name: str = "run"


def output_stem(cfg: ExperimentConfig) -> Path:
    stem = (f"{cfg.save_name}_NOP:{cfg.num_points}_IM:{cfg.kernel}_M:rotation_BW:{cfg.bandwidth:g}"
            f"_IC:{cfg.is_centered}_AS:{cfg.add_stationary}_{cfg.noise_tag}:{cfg.snr_db:g}_LT:RWGL")
    if cfg.kernel == "min" and cfg.min_method != "grid":
        stem += f"_MIN:{cfg.min_method}"
    if cfg.kernel in ("min", "integral") and not (cfg.kernel == "min" and cfg.min_method == "kabsch") \
            and cfg.num_rotations != kernels.NUM_ROTATIONS:
        stem += f"_G:{cfg.num_rotations}"
    return Path(cfg.save_folder) / stem


def run_experiment(cfg: ExperimentConfig, trajectory: np.ndarray | None = None) -> Path:
    """Run one configuration; returns the path of the saved .pkl."""
    if trajectory is None:
        trajectory = pointclouds.load_trajectory(cfg.data_path)
    rng = np.random.default_rng(cfg.seed)
    X = pointclouds.make_point_clouds(trajectory, cfg.num_points, centered=cfg.is_centered,
                                      add_stationary=cfg.add_stationary, snr_db=cfg.snr_db, rng=rng)
    W = kernels.weight_matrix(X, cfg.kernel, cfg.bandwidth, min_method=cfg.min_method,
                              num_rotations=cfg.num_rotations)
    phi, lam = spectral_embedding(W, cfg.m)

    stem = output_stem(cfg)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pkl = stem.parent / (stem.name + ".pkl")
    with open(pkl, "wb") as fh:
        pickle.dump(np.array(list(phi.T), dtype=object), fh)       # [phi_1, ..., phi_m]
    record = asdict(cfg) | {"num_points_per_cloud": int(X.shape[1]),
                            "eigenvalues": [float(x) for x in lam],
                            "embedding": "phi_1, phi_2 of L_RW = I - D^-1 W (Algorithm 1)"}
    (stem.parent / (stem.name + ".json")).write_text(json.dumps(record, indent=2))
    return pkl
