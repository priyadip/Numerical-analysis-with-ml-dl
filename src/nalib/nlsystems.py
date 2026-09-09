"""Systems of nonlinear equations: F(x) = 0 with F mapping R^n to R^n.

Everything in lessons 09 to 13 was one equation in one unknown. Here there are n of each, and
three things change.

**Bracketing disappears entirely.** There is no such thing as a sign change in n dimensions, so
every guarantee from lesson 09 is gone. All the methods here are open methods, and all of them
can fail.

**Division becomes a linear solve.** Newton's step goes from `f/f'` to solving `J s = -F`. That
single change is why Part 3 exists: from here on, being able to solve a linear system quickly
and accurately is the bottleneck of everything.

**The Jacobian may be unaffordable.** It has n^2 entries and needs n^2 partial derivatives.
Broyden's method builds an approximation to it from the steps already taken, which is the
n-dimensional analogue of replacing Newton by the secant method.

Used by lesson 14, and again in lesson 33 (nonlinear least squares) and lesson 86.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SystemResult:
    """Outcome of solving F(x) = 0, with the whole iteration history."""

    root: np.ndarray
    iterates: np.ndarray            # shape (k+1, n)
    residuals: np.ndarray           # ||F(x_k)|| at each step
    converged: bool
    n_iter: int
    n_feval: int                    # evaluations of F
    n_jeval: int                    # evaluations of the Jacobian
    message: str
    step_norms: np.ndarray = field(default_factory=lambda: np.array([]))

    def errors(self, exact) -> np.ndarray:
        """||x_k - exact|| at every step."""
        exact = np.asarray(exact, dtype=float).ravel()
        return np.linalg.norm(self.iterates - exact, axis=1)

    def __repr__(self) -> str:
        return (f"SystemResult(root={np.round(self.root, 8)}, "
                f"converged={self.converged}, n_iter={self.n_iter}, "
                f"residual={self.residuals[-1]:.3e})")


def numerical_jacobian(F: Callable, x, h: float | None = None) -> np.ndarray:
    """Jacobian by central differences. n^2 partial derivatives, 2n evaluations of F.

    The step size uses the eps**(1/3) balance that lesson 60 derives for central differences.
    Use this when an analytic Jacobian is unavailable, and note that it costs 2n evaluations
    per Newton step, which is exactly the cost Broyden's method exists to avoid.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if h is None:
        h = np.cbrt(np.finfo(float).eps) * np.maximum(np.abs(x), 1.0)
    else:
        h = np.full(n, float(h))
    F0 = np.asarray(F(x), dtype=float).ravel()
    J = np.empty((F0.size, n))
    for j in range(n):
        xp, xm = x.copy(), x.copy()
        xp[j] += h[j]
        xm[j] -= h[j]
        J[:, j] = (np.asarray(F(xp), dtype=float).ravel()
                   - np.asarray(F(xm), dtype=float).ravel()) / (2 * h[j])
    return J


# ---------------------------------------------------------------- fixed point


def fixed_point_system(G: Callable, x0, tol=1e-12, max_iter=1000) -> SystemResult:
    """x <- G(x) componentwise, all components updated from the OLD iterate.

    This is the Jacobi-style version. Theorem 10.2 generalises: it converges when the spectral
    radius of the Jacobian of G at the fixed point is below 1. The scalar condition
    |g'(r)| < 1 becomes rho(G'(r)) < 1, and everything else is unchanged.
    """
    x = np.asarray(x0, dtype=float).ravel().copy()
    xs, rs, steps = [x.copy()], [], []
    n_feval = 0
    for _ in range(max_iter):
        gx = np.asarray(G(x), dtype=float).ravel()
        n_feval += 1
        step = np.linalg.norm(gx - x)
        xs.append(gx.copy())
        rs.append(step)
        steps.append(step)
        if not np.all(np.isfinite(gx)):
            return _sys_finish(xs, rs, steps, False, n_feval, 0, "diverged")
        if step <= tol:
            x = gx
            return _sys_finish(xs, rs, steps, True, n_feval, 0, "converged")
        if np.linalg.norm(gx) > 1e150:
            return _sys_finish(xs, rs, steps, False, n_feval, 0, "diverged")
        x = gx
    return _sys_finish(xs, rs, steps, False, n_feval, 0, "iteration limit reached")


def seidel_system(G: Callable, x0, tol=1e-12, max_iter=1000) -> SystemResult:
    """Like `fixed_point_system`, but each component uses the values ALREADY updated.

    Component i is computed from the new values of components 0..i-1 and the old values of
    i+1..n-1. Using fresher information usually converges faster, sometimes converges when the
    Jacobi version does not, and costs nothing extra.

    This is exactly the relationship between the Jacobi and Gauss-Seidel methods for linear
    systems, which lesson 23 develops properly. It is the same idea, one level of nonlinearity
    up.
    """
    x = np.asarray(x0, dtype=float).ravel().copy()
    n = x.size
    xs, rs, steps = [x.copy()], [], []
    n_feval = 0
    for _ in range(max_iter):
        old = x.copy()
        for i in range(n):
            gx = np.asarray(G(x), dtype=float).ravel()
            n_feval += 1
            x[i] = gx[i]                      # commit component i immediately
        step = np.linalg.norm(x - old)
        xs.append(x.copy())
        rs.append(step)
        steps.append(step)
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e150:
            return _sys_finish(xs, rs, steps, False, n_feval, 0, "diverged")
        if step <= tol:
            return _sys_finish(xs, rs, steps, True, n_feval, 0, "converged")
    return _sys_finish(xs, rs, steps, False, n_feval, 0, "iteration limit reached")


# ---------------------------------------------------------------- Newton


def newton_system(F: Callable, J: Callable | None, x0,
                  tol=1e-12, max_iter=100) -> SystemResult:
    """Newton's method for systems. Solve J(x) s = -F(x), then x <- x + s.

    The one-dimensional step `x - f/f'` becomes `x + s` where `s` solves a linear system.
    Dividing by a number becomes solving with a matrix, and that is the whole difference.

    Quadratic convergence near a root where J is nonsingular, exactly as in one dimension.

    Cost per step: one evaluation of F, one of J (n^2 entries), and one linear solve at
    O(n^3). For large n that O(n^3) dominates completely, which is what motivates Broyden
    below and the entire content of Part 3.

    Pass `J=None` to use a central-difference Jacobian, at 2n extra evaluations of F per step.
    """
    x = np.asarray(x0, dtype=float).ravel().copy()
    xs, rs, steps = [x.copy()], [], []
    n_feval = n_jeval = 0

    Fx = np.asarray(F(x), dtype=float).ravel()
    n_feval += 1
    rs.append(float(np.linalg.norm(Fx)))

    for _ in range(max_iter):
        if J is None:
            Jx = numerical_jacobian(F, x)
            n_feval += 2 * x.size
        else:
            Jx = np.asarray(J(x), dtype=float)
        n_jeval += 1

        try:
            s = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            return _sys_finish(xs, rs, steps, False, n_feval, n_jeval,
                               "Jacobian is singular")

        x = x + s
        Fx = np.asarray(F(x), dtype=float).ravel()
        n_feval += 1
        xs.append(x.copy())
        rs.append(float(np.linalg.norm(Fx)))
        steps.append(float(np.linalg.norm(s)))

        if not np.all(np.isfinite(x)):
            return _sys_finish(xs, rs, steps, False, n_feval, n_jeval, "diverged")
        if np.linalg.norm(s) <= tol or rs[-1] <= tol:
            return _sys_finish(xs, rs, steps, True, n_feval, n_jeval, "converged")
    return _sys_finish(xs, rs, steps, False, n_feval, n_jeval,
                       "iteration limit reached")


def damped_newton(F: Callable, J: Callable | None, x0, tol=1e-12, max_iter=100,
                  max_backtrack: int = 30) -> SystemResult:
    """Newton with a backtracking line search on ||F||.

    Plain Newton takes the full step and hopes. From a poor starting point that step can be
    enormous and land somewhere worse. Damping tries the full step, and if ||F|| did not
    decrease, halves the step and tries again.

    The result is far more robust globally while keeping the full step, and therefore the
    quadratic rate, once it is close to the root. This is the simplest useful **globalisation**
    strategy, and lesson 86 develops the idea properly with the Wolfe conditions.
    """
    x = np.asarray(x0, dtype=float).ravel().copy()
    xs, rs, steps = [x.copy()], [], []
    n_feval = n_jeval = 0

    Fx = np.asarray(F(x), dtype=float).ravel()
    n_feval += 1
    rs.append(float(np.linalg.norm(Fx)))

    for _ in range(max_iter):
        Jx = numerical_jacobian(F, x) if J is None else np.asarray(J(x), dtype=float)
        if J is None:
            n_feval += 2 * x.size
        n_jeval += 1

        try:
            s = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            return _sys_finish(xs, rs, steps, False, n_feval, n_jeval,
                               "Jacobian is singular")

        lam, accepted = 1.0, False
        for _bt in range(max_backtrack):
            trial = x + lam * s
            Ft = np.asarray(F(trial), dtype=float).ravel()
            n_feval += 1
            if np.all(np.isfinite(Ft)) and np.linalg.norm(Ft) < rs[-1]:
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            return _sys_finish(xs, rs, steps, False, n_feval, n_jeval,
                               "line search failed to reduce ||F||")

        x, Fx = trial, Ft
        xs.append(x.copy())
        rs.append(float(np.linalg.norm(Fx)))
        steps.append(float(np.linalg.norm(lam * s)))
        if steps[-1] <= tol or rs[-1] <= tol:
            return _sys_finish(xs, rs, steps, True, n_feval, n_jeval, "converged")
    return _sys_finish(xs, rs, steps, False, n_feval, n_jeval,
                       "iteration limit reached")


# ---------------------------------------------------------------- Broyden


def broyden(F: Callable, x0, B0=None, tol=1e-12, max_iter=200) -> SystemResult:
    """Broyden's method: the secant method generalised to n dimensions.

    Newton needs the Jacobian at every step. In one dimension the secant method replaced f'
    with a difference quotient. In n dimensions the same idea runs into a problem: one step
    gives one equation about the Jacobian, and the Jacobian has n^2 unknowns. The system is
    massively underdetermined.

    Broyden's answer is to ask for the **smallest change** to the current approximation B that
    satisfies the secant condition

        B_new (x_new - x_old) = F(x_new) - F(x_old).

    That has a unique answer, the rank-one update

        B_new = B + (y - B s) s^T / (s^T s),      s = x_new - x_old,  y = F_new - F_old.

    Cost: **one** evaluation of F per step and no Jacobian at all, against Newton's one F plus
    n^2 partial derivatives. Convergence is superlinear rather than quadratic, which is the
    same trade the secant method makes in one dimension.

    `B0` defaults to a finite-difference Jacobian at x0, which costs 2n evaluations once.
    """
    x = np.asarray(x0, dtype=float).ravel().copy()
    n = x.size
    n_feval = 0

    Fx = np.asarray(F(x), dtype=float).ravel()
    n_feval += 1
    if B0 is None:
        B = numerical_jacobian(F, x)
        n_feval += 2 * n
    else:
        B = np.array(B0, dtype=float)

    xs, rs, steps = [x.copy()], [float(np.linalg.norm(Fx))], []

    for _ in range(max_iter):
        try:
            s = np.linalg.solve(B, -Fx)
        except np.linalg.LinAlgError:
            return _sys_finish(xs, rs, steps, False, n_feval, 0,
                               "approximate Jacobian became singular")

        x_new = x + s
        F_new = np.asarray(F(x_new), dtype=float).ravel()
        n_feval += 1

        xs.append(x_new.copy())
        rs.append(float(np.linalg.norm(F_new)))
        steps.append(float(np.linalg.norm(s)))

        if not np.all(np.isfinite(x_new)):
            return _sys_finish(xs, rs, steps, False, n_feval, 0, "diverged")
        if np.linalg.norm(s) <= tol or rs[-1] <= tol:
            return _sys_finish(xs, rs, steps, True, n_feval, 0, "converged")

        y = F_new - Fx
        denom = float(s @ s)
        if denom > 0:
            B = B + np.outer(y - B @ s, s) / denom      # the rank-one update
        x, Fx = x_new, F_new

    return _sys_finish(xs, rs, steps, False, n_feval, 0, "iteration limit reached")


# ---------------------------------------------------------------- shared


def _sys_finish(xs, rs, steps, converged, n_feval, n_jeval, message) -> SystemResult:
    return SystemResult(
        root=np.asarray(xs[-1], dtype=float),
        iterates=np.array(xs, dtype=float),
        residuals=np.array(rs, dtype=float),
        converged=bool(converged),
        n_iter=len(xs) - 1,
        n_feval=int(n_feval),
        n_jeval=int(n_jeval),
        message=message,
        step_norms=np.array(steps, dtype=float),
    )


def spectral_radius(A) -> float:
    """rho(A), the largest eigenvalue magnitude.

    This is the n-dimensional replacement for |g'(r)| in the fixed point convergence test of
    lesson 10, and it is the quantity that decides convergence for every stationary iterative
    method in Part 4 as well.
    """
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(A, dtype=float)))))
