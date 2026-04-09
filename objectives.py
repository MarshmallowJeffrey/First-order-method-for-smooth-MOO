"""
objectives.py  –  Test objective functions for the bundle MOO experiments
=========================================================================

Three families of objectives, one per assumption regime:

1. Strongly convex quadratics   F_k(x) = 0.5 x^T A_k x + b_k^T x
   ─ used for the GAP progress criterion.

2. Interpolation + PL objectives   F_k(x) = (1/n) Σ_j  log(1 + exp(−y_{kj} a_{kj}^T x))
   ─ nonnegative, share a common minimizer x* with F_k(x*)=0 when the data
     is interpolable; satisfy PL locally.  Used for the UB criterion.

3. Generic non-convex   F_k(x) = Σ_j [ c_{kj}(x_{j+1} − x_j²)² + (1−x_j)² ]
   ─ Rosenbrock-like sums with per-objective coefficients.  Non-convex, no PL.
     Used for the GN criterion.

Illustrative example (strongly convex)
---------------------------------------
K = 2 objectives on R^3:

    A_1 = diag(2, 5, 8)       →  L_1 = 8,  µ_1 = 2
    A_2 = diag(3, 4, 10)      →  L_2 = 10, µ_2 = 3
    b_1 = (0.1, −0.2, 0.3)
    b_2 = (−0.1, 0.5, −0.4)

Each F_k is µ_k-strongly convex and L_k-smooth.
"""

from __future__ import annotations
import numpy as np
from typing import List, Callable, Tuple


# ====================================================================
# 1.  Strongly convex quadratics
# ====================================================================
def make_quadratic_objectives(
    K: int,
    d: int,
    seed: int = 42,
    cond_range: Tuple[float, float] = (2.0, 20.0),
) -> Tuple[List[Callable], List[Callable], np.ndarray, np.ndarray]:
    """Create K random quadratic objectives  F_k(x) = 0.5 x^T A_k x + b_k^T x.

    Returns
    -------
    objectives       : list of K callables  f_k(x)->float
    grad_objectives  : list of K callables  g_k(x)->ndarray(d,)
    L                : smoothness constants  shape (K,)
    mu               : strong-convexity constants  shape (K,)
    """
    rng = np.random.RandomState(seed)
    As, bs = [], []
    L_arr = np.zeros(K)
    mu_arr = np.zeros(K)

    for k in range(K):
        # Random eigenvalues in [1, cond_range[1]]
        eigvals = rng.uniform(cond_range[0], cond_range[1], size=d)
        eigvals[0] = cond_range[0] + rng.rand()          # keep a small one
        eigvals[-1] = cond_range[1] - rng.rand()          # keep a big one
        Q, _ = np.linalg.qr(rng.randn(d, d))
        A = Q @ np.diag(eigvals) @ Q.T
        A = 0.5 * (A + A.T)  # ensure symmetry
        b = rng.randn(d) * 0.5

        As.append(A)
        bs.append(b)
        L_arr[k] = np.max(eigvals)
        mu_arr[k] = np.min(eigvals)

    objectives = [lambda x, A=A, b=b: 0.5 * x @ A @ x + b @ x for A, b in zip(As, bs)]
    grad_objectives = [lambda x, A=A, b=b: A @ x + b for A, b in zip(As, bs)]

    return objectives, grad_objectives, L_arr, mu_arr


# ====================================================================
# 2.  Interpolation + PL   (logistic-type, interpolable)
# ====================================================================
def make_interpolation_pl_objectives(
    K: int,
    d: int,
    n_samples: int = 20,
    seed: int = 123,
) -> Tuple[List[Callable], List[Callable], np.ndarray, np.ndarray]:
    """Create K logistic-like objectives that share a common interpolator.

    Construction:
        Choose x* = 0.  For each k, generate data (a_{kj}, y_{kj}) such that
        y_{kj} = sign(a_{kj}^T x* + noise) with large-margin noise so that
        x* perfectly separates the data.

        F_k(x) = (1/n) Σ_j log(1 + exp(−y_{kj} a_{kj}^T x))

    Since x*=0 perfectly separates the data we "cheat" by using
        F_k(x) = (1/n) Σ_j [log(1 + exp(−y_{kj} a_{kj}^T x)) − log(1+exp(0)) + log(2)]
    Nah, simpler:  use  F_k(x) = (1/n) Σ (a_{kj}^T x)^2  which has x*=0 and PL.

    Actually, the cleanest interpolation-satisfying, PL-satisfying objectives:
        F_k(x) = 0.5 ‖ C_k x ‖²     (over-determined least squares with x*=0)

    This is non-negative, F_k(0) = 0, strongly convex if C_k has full column rank
    ⇒ also PL with µ_k = σ_min(C_k)^2.  L_k = σ_max(C_k)^2.

    Returns  objectives, grad_objectives, L, mu.
    """
    rng = np.random.RandomState(seed)
    Cs = []
    L_arr = np.zeros(K)
    mu_arr = np.zeros(K)

    for k in range(K):
        C = rng.randn(n_samples, d) * (1.0 + 0.5 * k)
        svals = np.linalg.svd(C, compute_uv=False)
        L_arr[k] = svals[0] ** 2
        mu_arr[k] = svals[-1] ** 2
        Cs.append(C)

    objectives = [
        lambda x, C=C: 0.5 * np.sum((C @ x) ** 2)
        for C in Cs
    ]
    grad_objectives = [
        lambda x, C=C: C.T @ (C @ x)
        for C in Cs
    ]

    return objectives, grad_objectives, L_arr, mu_arr


# ====================================================================
# 3.  Generic non-convex  (Rosenbrock-like)
# ====================================================================
def make_nonconvex_objectives(
    K: int,
    d: int,
    seed: int = 7,
) -> Tuple[List[Callable], List[Callable], np.ndarray]:
    """Create K Rosenbrock-like objectives (non-convex, no PL).

        F_k(x) = Σ_{j=1}^{d-1} [ c_{kj} (x_{j+1} − x_j²)² + (1 − x_j)² ]

    with random positive coefficients c_{kj}.

    Smoothness: We estimate L_k with a generous upper bound via
    local Hessian sampling at the origin.

    Returns  objectives, grad_objectives, L  (no mu).
    """
    rng = np.random.RandomState(seed)
    coeffs = []
    L_arr = np.zeros(K)

    for k in range(K):
        c = rng.uniform(1.0, 5.0, size=d - 1)
        coeffs.append(c)
        # Heuristic L: use finite differences of gradient at the origin
        # For Rosenbrock, L is data-dependent; we use a conservative estimate.
        L_arr[k] = 4.0 * np.max(c) * 12.0 + 2.0  # rough bound

    def _fk(x, c):
        val = 0.0
        for j in range(len(c)):
            val += c[j] * (x[j + 1] - x[j] ** 2) ** 2 + (1.0 - x[j]) ** 2
        return val

    def _gk(x, c):
        g = np.zeros_like(x)
        for j in range(len(c)):
            # partial w.r.t. x_j:  −4 c_j x_j (x_{j+1} − x_j²) − 2(1−x_j)
            g[j] += -4.0 * c[j] * x[j] * (x[j + 1] - x[j] ** 2) - 2.0 * (1.0 - x[j])
            # partial w.r.t. x_{j+1}:  2 c_j (x_{j+1} − x_j²)
            g[j + 1] += 2.0 * c[j] * (x[j + 1] - x[j] ** 2)
        return g

    objectives = [lambda x, c=c: _fk(x, c) for c in coeffs]
    grad_objectives = [lambda x, c=c: _gk(x, c) for c in coeffs]

    return objectives, grad_objectives, L_arr
