"""
bundle.py  –  Core bundle data structure and progress criteria (Example 2)
==========================================================================

This module implements the *bundle* B_m from Section 3 of the paper:

    B_m = { (x_i, F_1(x_i), ..., F_K(x_i), ∇F_1(x_i), ..., ∇F_K(x_i)) }_{i=1}^m

and the three progress criteria from Example 2 (Section 5.2):

    1. UB  – upper bound (Eq. 12)
    2. GAP – gap = UB − LB  (Eq. 15, strongly convex case)
    3. GN  – scaled gradient norm (Eq. 17)

All quantities use the *λ-dependent* smoothness constants  Lλ = Σ_k λ_k L_k
and strong-convexity constants  µλ = Σ_k λ_k µ_k  (when applicable).

Illustrative example
--------------------
Suppose we have K = 2 objectives on R^2:

    F_1(x) = 0.5 * x^T A_1 x,   F_2(x) = 0.5 * x^T A_2 x

with A_1 = diag(2, 10) (so L_1=10, µ_1=2) and A_2 = diag(4, 6) (L_2=6, µ_2=4).

For λ = (0.5, 0.5) we get  Lλ = 8,  µλ = 3.
Starting from x_1 = (1, 1), the bundle stores F_k(x_1) and ∇F_k(x_1),
and we can immediately evaluate  UB(λ; B_1), GAP(λ; B_1), GN(λ; B_1).

After adding x_2 = T(λ; B_1)  (one gradient descent step picking the best
bundle point), each progress criterion is guaranteed to decrease or stay the
same (Assumption 3.1, global monotonicity).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from dataclasses import dataclass, field
from typing import List, Optional, Callable


# ---------------------------------------------------------------------------
# Bundle data structure
# ---------------------------------------------------------------------------
@dataclass
class Bundle:
    """Stores zeroth- and first-order oracle information at visited points.

    Attributes
    ----------
    K : int
        Number of objective functions.
    d : int
        Dimension of the decision variable x ∈ R^d.
    points : list[np.ndarray]
        List of iterates  x_1, …, x_m  (each shape (d,)).
    fvals : list[np.ndarray]
        fvals[i] = (F_1(x_i), …, F_K(x_i))  shape (K,).
    grads : list[np.ndarray]
        grads[i] = J_F(x_i)  the K×d Jacobian at x_i.
    L : np.ndarray
        Smoothness constants (L_1, …, L_K), shape (K,).
    mu : np.ndarray | None
        Strong-convexity / PL constants (µ_1, …, µ_K).  None when unavailable.
    """

    K: int
    d: int
    L: np.ndarray                       # shape (K,)
    mu: Optional[np.ndarray] = None     # shape (K,) or None
    points: List[np.ndarray] = field(default_factory=list)
    fvals: List[np.ndarray] = field(default_factory=list)
    grads: List[np.ndarray] = field(default_factory=list)

    # ---- helpers ----
    @property
    def m(self) -> int:
        """Current bundle size."""
        return len(self.points)

    def L_lam(self, lam: np.ndarray) -> float:
        """Lλ = Σ_k λ_k L_k."""
        return float(lam @ self.L)

    def mu_lam(self, lam: np.ndarray) -> float:
        """µλ = Σ_k λ_k µ_k  (requires self.mu is not None)."""
        assert self.mu is not None, "mu not set (needed for strong convexity / PL)"
        return float(lam @ self.mu)

    def F_lam(self, idx: int, lam: np.ndarray) -> float:
        """Fλ(x_i) = λ^T F(x_i)."""
        return float(lam @ self.fvals[idx])

    def grad_F_lam(self, idx: int, lam: np.ndarray) -> np.ndarray:
        """∇Fλ(x_i) = J_F(x_i)^T λ,  shape (d,)."""
        return self.grads[idx].T @ lam   # (d, K) @ (K,) = (d,)

    def add_point(self, x: np.ndarray,
                  objectives: List[Callable],
                  grad_objectives: List[Callable]):
        """Evaluate all objectives and gradients at x and append to bundle."""
        fv = np.array([f(x) for f in objectives])
        gv = np.vstack([g(x) for g in grad_objectives])   # (K, d)
        self.points.append(x.copy())
        self.fvals.append(fv)
        self.grads.append(gv)


# ---------------------------------------------------------------------------
# Progress criteria  (Section 5.2)
# ---------------------------------------------------------------------------
def UB(bundle: Bundle, lam: np.ndarray) -> float:
    """Upper bound progress criterion  (Eq. 12).

        UB(λ; B_m) = min_{i ∈ [m]}  { Fλ(x_i) − 1/(2Lλ) ‖∇Fλ(x_i)‖² }

    This is valid under any smoothness assumption (no convexity needed).
    """
    Ll = bundle.L_lam(lam)
    best = np.inf
    for i in range(bundle.m):
        fi = bundle.F_lam(i, lam)
        gi = bundle.grad_F_lam(i, lam)
        val = fi - 0.5 / Ll * np.dot(gi, gi)
        if val < best:
            best = val
    return best


def LB(bundle: Bundle, lam: np.ndarray) -> float:
    """Lower bound via aggregated strongly convex minorants  (Eq. 14).

        LB(λ; B_m) = max_{β ∈ Δ_m}  { −µλ/2 ‖X̌^T β‖² + <β, F^vec_λ − diag(H_λ) + µλ/2 d> }

    We evaluate at each vertex β = e_i (giving a valid lower bound) and at
    the uniform β, then take the max.  This avoids an expensive QP while
    remaining a valid (possibly loose) lower bound.
    """
    m = bundle.m
    if m == 0:
        return -np.inf
    mul = bundle.mu_lam(lam)

    # Precompute  x̌_i = x_i − (1/µλ) ∇Fλ(x_i)
    Xcheck = np.zeros((m, bundle.d))
    c = np.zeros(m)  # c_i = Fλ(x_i) − ⟨∇Fλ(x_i), x_i⟩ + µλ/2 ‖x_i‖²

    for i in range(m):
        xi = bundle.points[i]
        gi = bundle.grad_F_lam(i, lam)
        Xcheck[i] = xi - gi / mul
        c[i] = bundle.F_lam(i, lam) - np.dot(gi, xi) + 0.5 * mul * np.dot(xi, xi)

    # Evaluate at each vertex  β = e_i:   val = −µλ/2 ‖x̌_i‖² + c_i
    best = -np.inf
    for i in range(m):
        val = -0.5 * mul * np.dot(Xcheck[i], Xcheck[i]) + c[i]
        if val > best:
            best = val

    # Also try uniform β
    xbar = Xcheck.mean(axis=0)
    val_unif = -0.5 * mul * np.dot(xbar, xbar) + c.mean()
    if val_unif > best:
        best = val_unif

    return float(best)


def GAP(bundle: Bundle, lam: np.ndarray) -> float:
    """Gap progress criterion  GAP = UB − LB  (Eq. 15).

    Only meaningful under strong convexity.
    """
    return UB(bundle, lam) - LB(bundle, lam)


def GN(bundle: Bundle, lam: np.ndarray) -> float:
    """Scaled gradient-norm progress criterion  (Eq. 17).

        GN(λ; B_m) = 1/2 (1/µλ − 1/Lλ) min_{i} ‖∇Fλ(x_i)‖²

    Uses both µλ and Lλ as in Example 2.
    In the generic non-convex case (no µ), we fall back to
        GN(λ; B_m) = min_i ‖∇Fλ(x_i)‖   (un-scaled).
    """
    min_gnorm_sq = np.inf
    for i in range(bundle.m):
        gi = bundle.grad_F_lam(i, lam)
        gnorm_sq = float(np.dot(gi, gi))
        if gnorm_sq < min_gnorm_sq:
            min_gnorm_sq = gnorm_sq

    if bundle.mu is not None:
        mul = bundle.mu_lam(lam)
        Ll = bundle.L_lam(lam)
        scale = 0.5 * (1.0 / mul - 1.0 / Ll)
        return scale * min_gnorm_sq
    else:
        return np.sqrt(min_gnorm_sq)


# ---------------------------------------------------------------------------
# Mapping  T(λ; B_m)  –  the new point to add  (Eq. 13)
# ---------------------------------------------------------------------------
def T_map(bundle: Bundle, lam: np.ndarray) -> np.ndarray:
    """Compute T(λ; B_m) = x_{i*} − (1/Lλ) ∇Fλ(x_{i*})

    where  i* = argmin_{i∈[m]}{ Fλ(x_i) − 1/(2Lλ) ‖∇Fλ(x_i)‖² }.
    (Eq. 13 – one step of gradient descent from the best bundle point.)
    """
    Ll = bundle.L_lam(lam)
    best_val = np.inf
    best_i = 0
    for i in range(bundle.m):
        fi = bundle.F_lam(i, lam)
        gi = bundle.grad_F_lam(i, lam)
        val = fi - 0.5 / Ll * np.dot(gi, gi)
        if val < best_val:
            best_val = val
            best_i = i
    xi = bundle.points[best_i]
    gi = bundle.grad_F_lam(best_i, lam)
    return xi - (1.0 / Ll) * gi
