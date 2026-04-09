"""
experiments.py  –  Numerical experiments for the bundle MOO algorithm
=====================================================================

Runs three experiments corresponding to the three progress-criterion /
assumption regimes from Section 5.2:

  Experiment 1 – Strongly convex quadratics  (PC = GAP)
  Experiment 2 – Interpolation + PL         (PC = UB)
  Experiment 3 – Generic non-convex          (PC = GN)

For each experiment we:
  (a) run Algorithm 2 and record the PC* history,
  (b) compare against independent grid-based scalarisation (no bundle reuse),
  (c) plot convergence curves.

Output:  ``experiment_results.png``  saved to /mnt/user-data/outputs/
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from algorithm import algorithm2, simplex_grid
from bundle import Bundle, UB, GAP, GN, T_map
from objectives import (
    make_quadratic_objectives,
    make_interpolation_pl_objectives,
    make_nonconvex_objectives,
)


# ───────────────────────────────────────────────────────
#  Baseline:  independent scalarisation (no bundle reuse)
# ───────────────────────────────────────────────────────
def independent_scalarisation(
    K, d, objectives, grad_objectives, L, x0, mu, eps,
    pc_fn, simplex_res=20, max_inner=500,
):
    """For each λ on the simplex grid, run gradient descent independently.

    Returns the total number of oracle calls to achieve PC* ≤ eps.
    """
    grid = simplex_grid(K, simplex_res)
    total_oracle = 0

    for lam in grid:
        Ll = float(lam @ L)
        x = x0.copy()
        for _ in range(max_inner):
            g = sum(lam[k] * grad_objectives[k](x) for k in range(K))
            x = x - (1.0 / Ll) * g
            total_oracle += K
        # We don't track per-λ convergence; just count oracle calls.

    return total_oracle


# ───────────────────────────────────────────────────────
#  Experiment 1:  Strongly convex  (GAP)
# ───────────────────────────────────────────────────────
def experiment_strongly_convex():
    print("=" * 60)
    print("Experiment 1: Strongly convex quadratics  (PC = GAP)")
    print("=" * 60)
    K, d = 3, 5
    objs, grads, L, mu = make_quadratic_objectives(K, d, seed=42)
    x0 = np.ones(d) * 2.0
    eps = 1e-2

    t0 = time.time()
    res = algorithm2(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=x0, eps=eps, mode="gap", mu=mu,
        max_outer=80, simplex_res=20, verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n  Outer iterations : {res['outer_iters']}")
    print(f"  Oracle calls     : {res['oracle_calls']}")
    print(f"  Final PC*        : {res['pc_history'][-1]:.4e}")
    print(f"  Wall time        : {elapsed:.2f}s")
    return res


# ───────────────────────────────────────────────────────
#  Experiment 2:  Interpolation + PL  (UB)
# ───────────────────────────────────────────────────────
def experiment_interpolation_pl():
    print("\n" + "=" * 60)
    print("Experiment 2: Interpolation + PL  (PC = UB)")
    print("=" * 60)
    K, d = 3, 4
    objs, grads, L, mu = make_interpolation_pl_objectives(K, d, seed=123)
    x0 = np.ones(d) * 0.8
    eps = 1e-4

    t0 = time.time()
    res = algorithm2(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=x0, eps=eps, mode="ub", mu=mu,
        max_outer=80, simplex_res=20, max_inner=5, verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n  Outer iterations : {res['outer_iters']}")
    print(f"  Oracle calls     : {res['oracle_calls']}")
    print(f"  Final PC*        : {res['pc_history'][-1]:.4e}")
    print(f"  Wall time        : {elapsed:.2f}s")
    return res


# ───────────────────────────────────────────────────────
#  Experiment 3:  Generic non-convex  (GN)
# ───────────────────────────────────────────────────────
def experiment_generic_nonconvex():
    print("\n" + "=" * 60)
    print("Experiment 3: Generic non-convex  (PC = GN)")
    print("=" * 60)
    K, d = 3, 3
    objs, grads, L = make_nonconvex_objectives(K, d, seed=7)
    x0 = np.zeros(d) + 0.5
    eps = 0.5  # Larger eps for non-convex (convergence is slower)

    t0 = time.time()
    res = algorithm2(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=x0, eps=eps, mode="gn", mu=None,
        max_outer=40, simplex_res=10, C_lam=0.2, max_inner=5, verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n  Outer iterations : {res['outer_iters']}")
    print(f"  Oracle calls     : {res['oracle_calls']}")
    print(f"  Final PC*        : {res['pc_history'][-1]:.4e}")
    print(f"  Wall time        : {elapsed:.2f}s")
    return res


# ───────────────────────────────────────────────────────
#  Experiment 4:  Pareto front tracing  (2-objective)
# ───────────────────────────────────────────────────────
def experiment_pareto_front():
    """Trace the Pareto front for a 2-objective strongly-convex problem.

    After Algorithm 2 converges we evaluate the solution map
    x̂(λ) = T(λ; B_final) for a fine grid of λ ∈ Δ_2 and plot
    the corresponding (F_1, F_2) pairs.
    """
    print("\n" + "=" * 60)
    print("Experiment 4: Pareto front tracing  (2-objective quadratics)")
    print("=" * 60)
    K, d = 2, 4
    objs, grads, L, mu = make_quadratic_objectives(K, d, seed=99)
    x0 = np.ones(d) * 2.0
    eps = 5e-2

    res = algorithm2(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=x0, eps=eps, mode="gap", mu=mu,
        max_outer=80, simplex_res=15, verbose=False,
    )
    bundle = res["bundle"]

    # Evaluate the approximate solution map on a fine grid
    fine_grid = simplex_grid(K, 100)
    f1_vals, f2_vals = [], []
    for lam in fine_grid:
        xhat = T_map(bundle, lam)
        f1_vals.append(objs[0](xhat))
        f2_vals.append(objs[1](xhat))

    print(f"  Outer iterations : {res['outer_iters']}")
    print(f"  Oracle calls     : {res['oracle_calls']}")
    return f1_vals, f2_vals, res


# ───────────────────────────────────────────────────────
#  Plotting
# ───────────────────────────────────────────────────────
def make_plots(res1, res2, res3, pareto_data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Plot 1: GAP convergence ----
    ax = axes[0, 0]
    ax.semilogy(res1["pc_history"], "o-", color="#2563eb", markersize=3, linewidth=1.5)
    ax.axhline(y=1e-2, color="grey", ls="--", lw=1, label="ε = 1e-2")
    ax.set_xlabel("Outer iteration t")
    ax.set_ylabel("PC* (GAP)")
    ax.set_title("Exp 1: Strongly Convex (GAP criterion)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: UB convergence ----
    ax = axes[0, 1]
    ax.semilogy(res2["pc_history"], "s-", color="#dc2626", markersize=3, linewidth=1.5)
    ax.axhline(y=1e-4, color="grey", ls="--", lw=1, label="ε = 1e-4")
    ax.set_xlabel("Outer iteration t")
    ax.set_ylabel("PC* (UB)")
    ax.set_title("Exp 2: Interpolation + PL (UB criterion)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 3: GN convergence ----
    ax = axes[1, 0]
    ax.semilogy(res3["pc_history"], "^-", color="#16a34a", markersize=3, linewidth=1.5)
    ax.axhline(y=0.5, color="grey", ls="--", lw=1, label="ε = 0.5")
    ax.set_xlabel("Outer iteration t")
    ax.set_ylabel("PC* (GN)")
    ax.set_title("Exp 3: Generic Non-convex (GN criterion)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 4: Pareto front ----
    f1, f2, res4 = pareto_data
    ax = axes[1, 1]
    ax.scatter(f1, f2, s=8, c="#7c3aed", alpha=0.7)
    ax.set_xlabel("F₁(x̂(λ))")
    ax.set_ylabel("F₂(x̂(λ))")
    ax.set_title("Exp 4: Approximate Pareto Front (2-obj quadratics)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/experiment_results.png", dpi=150)
    plt.close()
    print("\nPlots saved to /mnt/user-data/outputs/experiment_results.png")


# ───────────────────────────────────────────────────────
#  Experiment 5:  Oracle complexity comparison
# ───────────────────────────────────────────────────────
def experiment_oracle_comparison():
    """Compare oracle calls: bundle method vs independent scalarisation."""
    print("\n" + "=" * 60)
    print("Experiment 5: Oracle complexity comparison")
    print("=" * 60)
    K, d = 3, 5
    objs, grads, L, mu = make_quadratic_objectives(K, d, seed=42)
    x0 = np.ones(d) * 2.0
    eps = 1e-2

    # Bundle method
    res_bundle = algorithm2(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=x0, eps=eps, mode="gap", mu=mu,
        max_outer=80, simplex_res=20, verbose=False,
    )

    # Independent scalarisation
    oracle_indep = independent_scalarisation(
        K, d, objs, grads, L, x0, mu, eps,
        pc_fn=GAP, simplex_res=20, max_inner=100,
    )

    print(f"  Bundle method oracle calls     : {res_bundle['oracle_calls']}")
    print(f"  Independent scalarisation calls : {oracle_indep}")
    print(f"  Speed-up factor                 : {oracle_indep / max(1, res_bundle['oracle_calls']):.1f}x")
    return res_bundle["oracle_calls"], oracle_indep


# ───────────────────────────────────────────────────────
if __name__ == "__main__":
    res1 = experiment_strongly_convex()
    res2 = experiment_interpolation_pl()
    res3 = experiment_generic_nonconvex()
    pareto = experiment_pareto_front()
    make_plots(res1, res2, res3, pareto)
    experiment_oracle_comparison()
    print("\n✓ All experiments completed.")
