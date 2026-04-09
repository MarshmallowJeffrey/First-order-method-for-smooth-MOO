# First-Order Bundle Method for Smooth Multi-Objective Optimization

Implementation of **Algorithm 2** from *"A First-Order Bundle Method for Smooth Multi-objective Optimization"* (Grigas & Cheng), following the concrete instantiation in **Example 2** (Section 5.2) with λ-dependent smoothness constants.

---

## File Overview

### `bundle.py` — Core Data Structures & Progress Criteria

This file defines the **bundle** $B_m$ and the three progress criteria from Section 5.2.

**Bundle** stores, for each visited point $x_i$:
- Function values $F_1(x_i), \dots, F_K(x_i)$
- Jacobian rows $\nabla F_1(x_i), \dots, \nabla F_K(x_i)$

**Progress Criteria** (all use λ-dependent constants $L_\lambda = \sum_k \lambda_k L_k$):

| Function | Formula (Eq.) | Assumption |
|----------|---------------|------------|
| `UB(λ; Bm)` | $\min_i \{ F_\lambda(x_i) - \frac{1}{2L_\lambda}\|\nabla F_\lambda(x_i)\|^2 \}$ (Eq. 12) | Any (smoothness only) |
| `GAP(λ; Bm)` | $UB(λ; Bm) - LB(λ; Bm)$ (Eq. 15) | Strong convexity |
| `GN(λ; Bm)` | $\frac{1}{2}(\frac{1}{\mu_\lambda} - \frac{1}{L_\lambda}) \min_i \|\nabla F_\lambda(x_i)\|^2$ (Eq. 17) | Generic non-convex |

**Mapping** `T(λ; Bm)` (Eq. 13): one gradient-descent step from the best bundle point.

**Illustrative example:**
```python
from bundle import Bundle, UB, GAP, GN, T_map
import numpy as np

# Two quadratic objectives on R^2
# F_1(x) = 0.5 * x^T diag(2,10) x,  F_2(x) = 0.5 * x^T diag(4,6) x
A1, A2 = np.diag([2., 10.]), np.diag([4., 6.])
objs  = [lambda x: 0.5*x@A1@x, lambda x: 0.5*x@A2@x]
grads = [lambda x: A1@x,        lambda x: A2@x]

bundle = Bundle(K=2, d=2, L=np.array([10., 6.]), mu=np.array([2., 4.]))
bundle.add_point(np.array([1.0, 1.0]), objs, grads)

lam = np.array([0.5, 0.5])   # equal weighting
print("UB  =", UB(bundle, lam))    # upper bound on F_λ(x̂(λ))
print("GAP =", GAP(bundle, lam))   # gap = UB − LB
print("GN  =", GN(bundle, lam))    # scaled gradient norm

x_new = T_map(bundle, lam)         # next iterate
bundle.add_point(x_new, objs, grads)
print("UB after update =", UB(bundle, lam))  # should decrease (monotonicity)
```

---

### `algorithm.py` — Algorithm 2 (Simple Adaptive Algorithm v2)

Implements the outer loop from Section 3.1:

```
Repeat:
  1. λ_t = argmax_{λ ∈ Δ_K}  PC(λ; B_t)     [grid search over simplex]
  2. M_t = IIC(λ_t, PC*_t, ε)                 [inner iteration count]
  3. B_{t+1} = BundleUpdate_{M_t}(λ_t; B_t)   [add M_t new points]
Until PC*_t ≤ ε
```

**Inner Iteration Count (IIC)** under each assumption:

| Assumption | PC | IIC formula |
|---|---|---|
| Strong convexity | GAP | $M_t = \lceil \kappa_\lambda \log(3\,\text{PC}^*/\varepsilon) \rceil$ |
| Interpolation + PL | UB | $M_t = \lceil \kappa_\lambda \log(3\,\text{PC}^*/\varepsilon) \rceil$ |
| Generic non-convex | GN | $M_t = \lceil 9\,C_\lambda^2 L_\lambda / \varepsilon^2 \rceil$ |

where $\kappa_\lambda = L_\lambda / \mu_\lambda$ is the condition number.

**Illustrative example:**
```python
from algorithm import algorithm2
import numpy as np

A1, A2 = np.diag([2., 10.]), np.diag([4., 6.])
result = algorithm2(
    K=2, d=2,
    objectives=[lambda x: 0.5*x@A1@x, lambda x: 0.5*x@A2@x],
    grad_objectives=[lambda x: A1@x, lambda x: A2@x],
    L=np.array([10., 6.]),
    mu=np.array([2., 4.]),
    x0=np.array([3.0, 3.0]),
    eps=0.01,
    mode="gap",          # use GAP criterion (strongly convex)
    simplex_res=20,
    verbose=True,
)
print("Outer iterations:", result["outer_iters"])
print("Oracle calls:", result["oracle_calls"])
```

---

### `objectives.py` — Test Objective Functions

Three families of objectives, one per assumption:

**1. Strongly convex quadratics** (`make_quadratic_objectives`):
$$F_k(x) = \tfrac{1}{2} x^\top A_k x + b_k^\top x$$
with random PSD matrices $A_k$ having controlled eigenvalue ranges.

**2. Interpolation + PL** (`make_interpolation_pl_objectives`):
$$F_k(x) = \tfrac{1}{2} \|C_k x\|^2$$
Nonneg, share common minimiser $x^*=0$, satisfy PL with $\mu_k = \sigma_{\min}(C_k)^2$.

**3. Generic non-convex** (`make_nonconvex_objectives`):
$$F_k(x) = \sum_{j=1}^{d-1} \bigl[ c_{kj}(x_{j+1} - x_j^2)^2 + (1-x_j)^2 \bigr]$$
Rosenbrock-like sums with per-objective coefficients. Non-convex, no PL.

---

### `experiments.py` — Numerical Experiments

Runs five experiments:

| # | Description | PC mode | K | d |
|---|---|---|---|---|
| 1 | Strongly convex quadratics | GAP | 3 | 5 |
| 2 | Interpolation + PL least-squares | UB | 3 | 4 |
| 3 | Rosenbrock-like non-convex | GN | 3 | 3 |
| 4 | Pareto front tracing (2-obj quadratics) | GAP | 2 | 4 |
| 5 | Oracle complexity: bundle vs independent scalarisation | GAP | 3 | 5 |

**Key results from a sample run:**
- Exp 1 (GAP): 14 outer iterations, 948 oracle calls, PC* = 9.38e-3
- Exp 2 (UB):  4 outer iterations, 48 oracle calls, PC* = 5.54e-6
- Exp 3 (GN):  16 outer iterations, 228 oracle calls, PC* = 4.79e-1
- Exp 5: Bundle method uses **73× fewer** oracle calls than independent scalarisation

**Running:**
```bash
python experiments.py
```
Produces `experiment_results.png` with four subplots:
1. GAP convergence (log scale)
2. UB convergence (log scale)
3. GN convergence (log scale)
4. Approximate Pareto front

---

## How to Run

```bash
pip install numpy scipy matplotlib
python experiments.py
```

All four source files should be in the same directory.
