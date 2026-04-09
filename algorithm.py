"""
algorithm.py  –  Algorithm 2 (Simple Adaptive Algorithm v2) from Section 3.1
==============================================================================

This file implements the main outer loop of Algorithm 2 and the inner loop
``BundleUpdate_M`` (Section 3.1).  It supports three *progress criterion* /
*objective-function assumption* combinations from Example 2 (Section 5.2):

┌─────────────────────────────────────┬──────────┬──────────────────────────────┐
│ Objective assumption                │ PC       │ Inner iteration count (IIC)  │
├─────────────────────────────────────┼──────────┼──────────────────────────────┤
│ Strong convexity                    │ GAP      │ κ log(3 PC*/ε)              │
│ Interpolation + PL (non-convex)     │ UB       │ κ log(3 PC*/ε)              │
│ Generic non-convex (no PL/SC)       │ GN       │ ⌈9 C²λ Lλ / ε²⌉            │
└─────────────────────────────────────┴──────────┴──────────────────────────────┘

κ = L̃λ / µλ  is the condition number at the selected weight vector.

Illustrative example
--------------------
Consider K = 3 quadratics on R^5, all strongly convex.

    F_k(x) = 0.5 x^T A_k x + b_k^T x

The algorithm:
  1. Initialises the bundle at a random point x_0.
  2. At every outer iteration, finds  λ_t = argmax_{λ ∈ Δ_K} PC(λ; B_t)
     by discretising the simplex.
  3. Runs  M_t  inner gradient-descent steps at that λ_t, each time adding
     T(λ_t; B) to the bundle.
  4. Stops when  PC*_t = max PC(λ; B_t) ≤ ε.

The bundle is shared across all λ-subproblems, so gradient evaluations
done for one λ benefit every other λ  (the key efficiency gain).
"""

from __future__ import annotations

import numpy as np
import math
from typing import Callable, Dict, List, Optional, Tuple

from bundle import Bundle, UB, GAP, GN, LB, T_map


# ---------------------------------------------------------------------------
# Simplex grid  (for maximising PC over Δ_K)
# ---------------------------------------------------------------------------
def simplex_grid(K: int, resolution: int = 20) -> np.ndarray:
    """Return an array of shape (N, K) whose rows tile the unit simplex Δ_K.

    ``resolution`` controls how many grid points per edge.
    For K=2 this gives ``resolution+1`` points;
    for K=3 it gives O(resolution²) points, etc.
    """
    if K == 1:
        return np.array([[1.0]])
    if K == 2:
        ts = np.linspace(0, 1, resolution + 1)
        return np.column_stack([ts, 1 - ts])
    # General K: enumerate compositions of resolution into K parts
    points = []

    def _recurse(remaining, depth, current):
        if depth == K - 1:
            current.append(remaining)
            points.append(current[:])
            current.pop()
            return
        for v in range(remaining + 1):
            current.append(v)
            _recurse(remaining - v, depth + 1, current)
            current.pop()

    _recurse(resolution, 0, [])
    arr = np.array(points, dtype=float) / resolution
    return arr


# ---------------------------------------------------------------------------
# BundleUpdate_M  (inner loop: M gradient steps at fixed λ)
# ---------------------------------------------------------------------------
def bundle_update_M(
    bundle: Bundle,
    lam: np.ndarray,
    M: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
) -> Bundle:
    """Add M new points to the bundle via T(λ; ·) steps (Section 3.1).

    Each step computes  x̄ = T(λ; B)  and evaluates all K objectives/gradients.
    """
    for _ in range(M):
        x_new = T_map(bundle, lam)
        bundle.add_point(x_new, objectives, grad_objectives)
    return bundle


# ---------------------------------------------------------------------------
# Inner iteration count (IIC)  under each assumption
# ---------------------------------------------------------------------------
def iic_strongly_convex(bundle: Bundle, lam: np.ndarray,
                        pc_star: float, eps: float) -> int:
    """M_t = ceil( κ_λ  log(3 PC*_t / ε) )   where κ = L̃λ / µλ."""
    kappa = bundle.L_lam(lam) / bundle.mu_lam(lam)
    if pc_star <= eps:
        return 1
    return max(1, math.ceil(kappa * math.log(3.0 * pc_star / eps)))


def iic_interpolation_pl(bundle: Bundle, lam: np.ndarray,
                         pc_star: float, eps: float) -> int:
    """Same formula as strong convexity but PC = UB (Corollary 4.2)."""
    kappa = bundle.L_lam(lam) / bundle.mu_lam(lam)
    if pc_star <= eps:
        return 1
    return max(1, math.ceil(kappa * math.log(3.0 * pc_star / eps)))


def iic_generic_nonconvex(bundle: Bundle, lam: np.ndarray,
                          eps: float, C_lam: float = 1.0) -> int:
    """M_t = ceil( 9 C²_λ L_λ / ε² )   (Corollary 4.3)."""
    Ll = bundle.L_lam(lam)
    return max(1, math.ceil(9.0 * C_lam**2 * Ll / eps**2))


# ---------------------------------------------------------------------------
# Algorithm 2  –  the main routine
# ---------------------------------------------------------------------------
def algorithm2(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    eps: float = 1e-3,
    mode: str = "gap",                # "ub", "gap", or "gn"
    mu: Optional[np.ndarray] = None,  # needed for "gap" and "ub" (PL)
    max_outer: int = 200,
    simplex_res: int = 20,
    C_lam: float = 1.0,              # constant for GN mode
    max_inner: int = 0,              # cap on inner iterations (0=no cap)
    verbose: bool = False,
) -> Dict:
    """Run Algorithm 2 from the paper.

    Parameters
    ----------
    K, d         : number of objectives, dimension.
    objectives   : list of K callables  f_k(x) -> float.
    grad_objectives : list of K callables  g_k(x) -> np.ndarray of shape (d,).
    L            : smoothness constants, shape (K,).
    x0           : initial point, shape (d,).
    eps          : target accuracy.
    mode         : which progress criterion to use.
                   "gap"  – strongly convex (GAP = UB − LB),
                   "ub"   – interpolation + PL (upper bound),
                   "gn"   – generic non-convex (gradient norm).
    mu           : strong convexity / PL constants, shape (K,).
    max_outer    : max outer iterations.
    simplex_res  : grid resolution for the simplex.
    C_lam        : constant for GN inner-iteration count.
    verbose      : print progress.

    Returns
    -------
    dict with keys:
        "bundle"       : final Bundle object,
        "pc_history"   : list of PC*_t at each outer iteration,
        "lam_history"  : list of λ_t chosen at each iteration,
        "oracle_calls" : total number of oracle (gradient) evaluations,
        "outer_iters"  : number of outer iterations executed.
    """
    # ---- choose PC function ----
    if mode == "gap":
        assert mu is not None
        pc_fn = GAP
    elif mode == "ub":
        assert mu is not None  # need µ for IIC
        pc_fn = UB
    elif mode == "gn":
        pc_fn = GN
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ---- initialise bundle ----
    bundle = Bundle(K=K, d=d, L=L, mu=mu)
    bundle.add_point(x0, objectives, grad_objectives)

    grid = simplex_grid(K, simplex_res)

    pc_history = []
    lam_history = []
    oracle_calls = K  # initial point evaluates K gradients

    for t in range(max_outer):
        # Step 1: find λ_t = argmax PC(λ; B_t)
        best_pc = -np.inf
        best_lam = grid[0]
        for lam in grid:
            val = pc_fn(bundle, lam)
            if val > best_pc:
                best_pc = val
                best_lam = lam.copy()

        pc_star = best_pc
        pc_history.append(pc_star)
        lam_history.append(best_lam.copy())

        if verbose:
            print(f"  outer iter {t:3d} | PC* = {pc_star:.6e} | λ = {best_lam}")

        if pc_star <= eps:
            if verbose:
                print(f"  Converged at outer iteration {t}.")
            break

        # Step 2: compute IIC and run BundleUpdate_M
        if mode == "gap":
            Mt = iic_strongly_convex(bundle, best_lam, pc_star, eps)
        elif mode == "ub":
            Mt = iic_interpolation_pl(bundle, best_lam, pc_star, eps)
        else:  # "gn"
            Mt = iic_generic_nonconvex(bundle, best_lam, eps, C_lam)

        if max_inner > 0:
            Mt = min(Mt, max_inner)

        bundle_update_M(bundle, best_lam, Mt, objectives, grad_objectives)
        oracle_calls += Mt * K

    return {
        "bundle": bundle,
        "pc_history": pc_history,
        "lam_history": lam_history,
        "oracle_calls": oracle_calls,
        "outer_iters": len(pc_history),
    }
