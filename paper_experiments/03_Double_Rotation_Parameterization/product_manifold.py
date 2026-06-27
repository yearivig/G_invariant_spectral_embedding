"""
Product manifold decomposition utilities.

Given diffusion map eigenvectors from a product manifold M = M1 x M2,
this module identifies which eigenvectors belong to which factor.

Algorithm:
  1) find_combos():  find eigenvector triplets (i, j, k) where
     φ_k ≈ φ_i · φ_j and λ_k ≈ λ_i + λ_j
  2) split_eigenvectors():  cluster the base eigenvectors into two groups
     using a voting scheme + convex relaxation (SDP via cvxpy)

Refactored from:
    ../roy_lederman_data/product_maniforld_utils.py
    (note: fixed the function-signature mismatch bug in the original)
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np


def calculate_similarity(v_i: np.ndarray, v_j: np.ndarray) -> float:
    """
    Cosine similarity between two vectors: <v_i/||v_i||, v_j/||v_j||>.
    """
    v_i = v_i / np.linalg.norm(v_i)
    v_j = v_j / np.linalg.norm(v_j)
    return float(np.dot(v_i, v_j))


def find_combos(
    phi: np.ndarray,
    sigma: np.ndarray,
    *,
    n_factors: int = 2,
    eig_crit: float = 1e-2,
    sim_crit: float = 0.5,
    exclude_eigs: list[int] | None = None,
) -> tuple[dict, dict, dict]:
    """
    Find eigenvector factorization triplets.

    For each eigenvector φ_k, search for combinations (i₁, ..., i_m) such that:
      - λ_k ≈ λ_{i₁} + ... + λ_{i_m}  (eigenvalue criterion)
      - φ_k ≈ φ_{i₁} · ... · φ_{i_m}  (similarity criterion)

    Parameters
    ----------
    phi : (N, K) array, columns are eigenvectors
    sigma : (K,) array of eigenvalues (or singular values)
    n_factors : max number of factors in a product
    eig_crit : threshold for |λ_k - Σλ_i|
    sim_crit : minimum similarity to report a match

    Returns
    -------
    best_matches : dict mapping k -> list of factor indices
    max_sims : dict mapping k -> similarity score
    all_sims : dict mapping k -> best similarity (even below threshold)
    """
    best_matches: dict[int, list[int]] = {}
    max_sims: dict[int, float] = {}
    all_sims: dict[int, float] = {}

    n_eig = phi.shape[1] if phi.ndim == 2 else phi.shape[0]

    for k in range(2, n_eig):
        if k % 10 == 0:
            sys.stdout.write(f"\r  find_combos: {k}/{n_eig}")
            sys.stdout.flush()

        v_k = phi[:, k] if phi.ndim == 2 else phi[k]
        lambda_k = sigma[k] if hasattr(sigma, "__getitem__") else 0

        max_sim = 0.0
        best_match: list[int] = []

        valid_eigs = [v for v in range(1, k) if (exclude_eigs is None or v not in exclude_eigs)]

        for m in range(2, n_factors + 1):
            for combo in combinations(valid_eigs, m):
                combo_list = list(combo)
                lambda_sum = sum(sigma[c] for c in combo_list)
                if abs(lambda_k - lambda_sum) < eig_crit:
                    v_combo = np.ones(phi.shape[0])
                    for c in combo_list:
                        v_combo *= phi[:, c] if phi.ndim == 2 else phi[c]

                    sim = abs(calculate_similarity(v_combo, v_k))
                    if sim > max_sim:
                        best_match = combo_list
                        max_sim = sim

        if best_match:
            all_sims[k] = max_sim
            if max_sim >= sim_crit:
                best_matches[k] = best_match
                max_sims[k] = max_sim

    print()  # newline after progress
    return best_matches, max_sims, all_sims


def split_eigenvectors(
    best_matches: dict,
    best_sims: dict,
    n_eigenvectors: int,
    *,
    n_factors: int = 2,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster eigenvectors into factor groups using a voting + SDP approach.

    Parameters
    ----------
    best_matches : dict from find_combos()
    best_sims : dict from find_combos()
    n_eigenvectors : total number of eigenvectors considered
    n_factors : number of manifold factors (default 2)
    verbose : print details

    Returns
    -------
    labels : (2, m) array — row 0 = eigenvector indices, row 1 = factor labels
    C : (n_eigenvectors, n_eigenvectors) separability matrix
    """
    import cvxpy as cp

    votes = np.zeros(n_eigenvectors)
    C = np.zeros((n_eigenvectors, n_eigenvectors))

    for match_k, combo in best_matches.items():
        sim = best_sims[match_k]
        if verbose:
            print(f"  {combo} -> {match_k}  sim={sim:.3f}")
        for p1, p2 in combinations(combo, 2):
            C[p1, p2] += sim
            C[p2, p1] += sim
            votes[p1] += sim
            votes[p2] += sim

    factors = np.where(votes > 0)[0]
    n = len(factors)
    if n == 0:
        return np.zeros((2, 0), dtype=int), C

    # Build reduced separability matrix
    C_ = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                C_[i, j] = C[factors[i], factors[j]]

    if verbose:
        print(f"\nSeparability matrix ({n}×{n}):\n", np.around(C_, 3))

    np.random.seed(1)

    if n_factors == 2:
        Y = cp.Variable((n, n), PSD=True)
        constraints = [cp.diag(Y) == 1]
        obj = 0.5 * cp.sum(cp.multiply(C_, np.ones((n, n)) - Y))
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve()

        eigenvalues, eigenvectors = np.linalg.eigh(Y.value)
        eigenvalues = np.maximum(eigenvalues, 0)
        assignment = np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        partition = np.random.normal(size=n)
        projections = assignment.T @ partition

        labels = np.zeros((2, n), dtype=int)
        labels[0, :] = factors
        labels[1, :] = (projections >= 0).astype(int)

    else:
        # Max k-cut heuristic for n_factors > 2
        Y = cp.Variable((n, n), PSD=True)
        constraints = [cp.diag(Y) == 1]
        for i in range(n):
            for j in range(n):
                if i != j:
                    constraints.append(Y[i, j] >= -1 / (n_factors - 1))

        obj = (1 - 1 / n_factors) * cp.sum(cp.multiply(C_, np.ones((n, n)) - Y))
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve()

        eigenvalues, eigenvectors = np.linalg.eigh(Y.value)
        eigenvalues = np.maximum(eigenvalues, 0)
        diag_root = np.diag(np.sqrt(eigenvalues))
        assignment = diag_root @ eigenvectors.T
        assignment_ = np.concatenate((np.zeros_like(assignment), assignment), axis=0)
        assignment = np.concatenate((assignment, np.zeros_like(assignment)), axis=0)
        g = np.random.normal(size=2 * n)
        g /= np.linalg.norm(g)
        theta = np.arctan2(assignment_.T @ g, assignment.T @ g)
        z = 2 * np.pi * np.random.random()

        labels = np.zeros((2, n), dtype=int)
        labels[0, :] = factors
        for i in range(n):
            labels[1, i] = int(((theta[i] - z) % (2 * np.pi)) / (2 * np.pi / n_factors))

    return labels, C
