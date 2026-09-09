"""Nonlinear least squares: Gauss-Newton, Levenberg-Marquardt and the problems they are for.

What this module is for
-----------------------
Lessons 29 to 33 minimised ``||b - Ax||`` where the model was **linear in its parameters**.
Most models are not. Fitting ``y = a exp(b t)`` is linear in ``a`` and nonlinear in ``b``, and
no rearrangement makes it linear in both.

The problem is

    minimise  f(x) = (1/2) ||r(x)||^2,    r : R^n -> R^m,  m >= n.

**Gauss-Newton is the linear theory applied one step at a time.** Linearise ``r`` about the
current point, ``r(x + p) ~ r(x) + J p``, and solve the *linear* least squares problem
``min ||J p + r||`` for the step. Every lesson from 29 to 32 applies to that inner solve,
including which algorithm to use for it, which is why this lesson comes last in Part 5.

**What is new is that the linearisation is wrong**, and how wrong it is decides everything.
The exact Newton Hessian of ``f`` is

    H = J^T J + S,     S = sum_i r_i(x) * Hessian(r_i)(x),

and Gauss-Newton is Newton with ``S`` dropped. So it is exact when the residual is zero and
degrades as the residual grows: ``S`` is literally the residual times the model's curvature.
That single fact explains the observed convergence rates, when Gauss-Newton diverges, and what
Levenberg-Marquardt is doing about it.

Everything here derives its sizes from its input. The number of residuals, the number of
parameters and the number of data points are all read from what is passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .nlsystems import numerical_jacobian


@dataclass
class NLLSResult:
    """The outcome of a nonlinear least squares solve, with enough history to measure a rate."""

    x: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    message: str
    history: list = field(default_factory=list)          # ||r|| at each iterate
    steps: list = field(default_factory=list)            # ||p|| at each iterate
    lambdas: list = field(default_factory=list)          # LM damping, empty for Gauss-Newton
    n_feval: int = 0
    n_jeval: int = 0


def _scaled_gradient(J, r) -> float:
    """``||J^T r|| / (||J|| ||r||)``, the scale invariant measure of how far ``r`` is from being
    orthogonal to the range of ``J``.

    **The unscaled ``||J^T r||`` is the wrong test**, and using it is a real bug rather than a
    stylistic choice: it has the units of ``J`` times ``r``, so the same problem with the data
    measured in metres instead of kilometres fails a test it used to pass. Measured on lesson
    34's large residual problem, the unscaled gradient at the true minimum is ``1.0e-8``, which
    looks like failure, while the scaled one is ``4.9e-11``, which is convergence to eleven
    digits. This is `leastsquares.residual_is_orthogonal` applied to the linearised problem.
    """
    nj = float(np.linalg.norm(J, 2))
    nr = float(np.linalg.norm(r))
    denom = nj * nr
    if denom <= 0.0:
        return 0.0
    return float(np.linalg.norm(J.T @ r)) / denom


def _prepare(residual, jacobian, x0):
    """Normalise the inputs and give a finite difference Jacobian when none is supplied."""
    x = np.asarray(x0, dtype=float).ravel()
    r0 = np.asarray(residual(x), dtype=float).ravel()
    if r0.size < x.size:
        raise ValueError(f"{r0.size} residuals for {x.size} parameters: the problem is "
                         f"underdetermined and least squares does not pin it down")
    if jacobian is not None:
        return x, r0, jacobian, 0
    # A finite difference Jacobian costs n extra residual evaluations per column, and the
    # caller deserves to see that in n_feval rather than have it hidden inside the helper.
    return x, r0, (lambda z: numerical_jacobian(residual, z)), int(x.size)


def gauss_newton(residual: Callable, jacobian: Callable | None, x0,
                 tol: float = 1e-10, max_iter: int = 100,
                 line_search: bool = False,
                 ftol: float = 1e-14, xtol: float = 1e-14) -> NLLSResult:
    """Linearise, solve the linear least squares problem for the step, repeat.

    Each step solves ``min_p ||J p + r||`` and sets ``x <- x + p``. The inner solve uses QR
    through ``numpy.linalg.lstsq``, for the reason lesson 29 measured: forming ``J^T J`` would
    square the condition number, and ``J`` is often badly conditioned near the solution.

    **With ``line_search=False`` this is the textbook method and it can diverge.** The step is
    a descent direction whenever ``J`` has full rank, but nothing bounds its length, and on a
    large residual problem it routinely overshoots. `line_search=True` adds simple backtracking,
    which is the cheapest possible fix and is not the same thing as Levenberg-Marquardt: it
    shortens the step without changing its direction.

    Convergence is quadratic when ``r(x*) = 0`` and linear otherwise, with a rate that grows
    with the residual. Lesson 34 measures the crossover.

    **Three stopping tests, not one**, which is MINPACK's convention and it is not padding.
    ``tol`` is on the scaled gradient, ``ftol`` on the relative decrease in ``||r||``, and
    ``xtol`` on the relative step. On a nonzero residual problem the iteration does not contract
    to machine precision: it reaches a point where the step is at the roundoff level and then
    limit cycles. The gradient test alone never fires there. Measured on lesson 34's GPS problem
    with 9 satellites, the gradient test ran to the iteration limit of 100 with the correct
    answer in hand, while ``ftol`` stops it at 6.
    """
    x, r, jac, fd_cost = _prepare(residual, jacobian, x0)
    out = NLLSResult(x=x.copy(), residual_norm=float(np.linalg.norm(r)), iterations=0,
                     converged=False, message="", n_feval=1)
    out.history.append(out.residual_norm)

    for k in range(int(max_iter)):
        J = np.atleast_2d(np.asarray(jac(x), dtype=float))
        out.n_jeval += 1
        out.n_feval += fd_cost
        if J.shape != (r.size, x.size):
            raise ValueError(f"the Jacobian is {J.shape}, expected ({r.size}, {x.size})")
        # The gradient of (1/2)||r||^2 is J^T r. Stop on it, not on the step, because a tiny
        # step can also mean a stalled line search. Scaled, so the test does not depend on the
        # units the data was measured in.
        if _scaled_gradient(J, r) <= tol:
            out.converged = True
            out.message = f"scaled gradient below tolerance after {k} iterations"
            break
        p = np.linalg.lstsq(J, -r, rcond=None)[0]

        step = 1.0
        if line_search:
            f_now = float(r @ r)
            while step > 1e-12:
                trial = np.asarray(residual(x + step * p), dtype=float).ravel()
                out.n_feval += 1
                if float(trial @ trial) < f_now:
                    break
                step *= 0.5

        f_before = float(np.linalg.norm(r))
        x_before = float(np.linalg.norm(x))
        x = x + step * p
        r = np.asarray(residual(x), dtype=float).ravel()
        out.n_feval += 1
        out.iterations = k + 1
        out.steps.append(float(step * np.linalg.norm(p)))
        out.history.append(float(np.linalg.norm(r)))
        if not np.all(np.isfinite(r)):
            out.message = f"the residual stopped being finite at iteration {k + 1}"
            break
        f_after = float(np.linalg.norm(r))
        if abs(f_before - f_after) <= ftol * max(f_after, 1.0):
            out.converged = True
            out.message = (f"the residual stopped decreasing after {k + 1} iterations "
                           f"(relative change below ftol)")
            break
        if out.steps[-1] <= xtol * max(x_before, 1.0):
            out.converged = True
            out.message = f"the step fell below xtol after {k + 1} iterations"
            break
    else:
        out.message = f"hit the iteration limit of {max_iter}"

    out.x = x
    out.residual_norm = float(np.linalg.norm(r)) if np.all(np.isfinite(r)) else float("inf")
    return out


def levenberg_marquardt(residual: Callable, jacobian: Callable | None, x0,
                        tol: float = 1e-10, max_iter: int = 200,
                        lam0: float = 1e-3, up: float = 10.0, down: float = 10.0,
                        scaled: bool = True,
                        ftol: float = 1e-14, xtol: float = 1e-14) -> NLLSResult:
    """Gauss-Newton with a damping term that is raised when a step fails and lowered when it works.

    The step solves

        (J^T J + lam * D) p = -J^T r,

    which this routine computes as the **augmented least squares problem**

        min || [J; sqrt(lam) sqrt(D)] p + [r; 0] ||,

    for the reason lesson 32 exercise 2.4 gives: stacking never forms ``J^T J`` and its
    condition number is *better* than ``J``'s, while the normal equations form would be worse.

    ``lam`` interpolates between two methods. As ``lam -> 0`` the step is the Gauss-Newton
    step. As ``lam -> infinity`` it becomes ``-J^T r / lam``, a short step along the steepest
    descent direction. So the damping trades the fast direction for a safe one, continuously.

    ``scaled=True`` uses ``D = diag(J^T J)`` (Marquardt's choice) rather than ``D = I``
    (Levenberg's). That makes the method invariant to rescaling the parameters, which matters
    whenever they have different units, and lesson 34 measures a case where it does.

    ``ftol`` and ``xtol`` are the same two extra stopping tests `gauss_newton` documents.
    """
    x, r, jac, fd_cost = _prepare(residual, jacobian, x0)
    lam = float(lam0)
    out = NLLSResult(x=x.copy(), residual_norm=float(np.linalg.norm(r)), iterations=0,
                     converged=False, message="", n_feval=1)
    out.history.append(out.residual_norm)

    for k in range(int(max_iter)):
        J = np.atleast_2d(np.asarray(jac(x), dtype=float))
        out.n_jeval += 1
        out.n_feval += fd_cost
        if J.shape != (r.size, x.size):
            raise ValueError(f"the Jacobian is {J.shape}, expected ({r.size}, {x.size})")
        if _scaled_gradient(J, r) <= tol:
            out.converged = True
            out.message = f"scaled gradient below tolerance after {k} iterations"
            break

        if scaled:
            d = np.sqrt(np.maximum(np.sum(J * J, axis=0), 1e-300))   # sqrt(diag(J^T J))
        else:
            d = np.ones(x.size)

        f_now = float(r @ r)
        accepted = False
        for _ in range(60):                       # raise lam until a step actually helps
            stacked = np.vstack([J, np.sqrt(lam) * np.diag(d)])
            padded = np.concatenate([-r, np.zeros(x.size)])
            p = np.linalg.lstsq(stacked, padded, rcond=None)[0]
            trial_x = x + p
            trial_r = np.asarray(residual(trial_x), dtype=float).ravel()
            out.n_feval += 1
            if np.all(np.isfinite(trial_r)) and float(trial_r @ trial_r) < f_now:
                x, r = trial_x, trial_r
                lam = max(lam / down, 1e-14)
                accepted = True
                break
            lam = min(lam * up, 1e14)

        out.iterations = k + 1
        out.lambdas.append(lam)
        out.history.append(float(np.linalg.norm(r)))
        out.steps.append(float(np.linalg.norm(p)))
        if accepted:
            f_after = float(np.linalg.norm(r))
            if abs(np.sqrt(f_now) - f_after) <= ftol * max(f_after, 1.0):
                out.converged = True
                out.message = (f"the residual stopped decreasing after {k + 1} iterations "
                               f"(relative change below ftol)")
                break
            if out.steps[-1] <= xtol * max(float(np.linalg.norm(x)), 1.0):
                out.converged = True
                out.message = f"the step fell below xtol after {k + 1} iterations"
                break
        if not accepted:
            # No step of any length helped. At a genuine minimum that is exactly what should
            # happen once roundoff dominates, so check the gradient before calling it failure.
            g = _scaled_gradient(np.atleast_2d(np.asarray(jac(x), dtype=float)), r)
            out.n_jeval += 1
            if g <= max(tol, 1e-8):
                out.converged = True
                out.message = (f"no step improved on iteration {k + 1}, and the scaled "
                               f"gradient is {g:.1e}: this is the minimum")
            else:
                out.message = (f"no damping in [1e-14, 1e14] produced a decrease at iteration "
                               f"{k + 1}, and the scaled gradient is still {g:.1e}: this is a "
                               f"flat valley, not a solution")
            break
    else:
        out.message = f"hit the iteration limit of {max_iter}"

    out.x = x
    out.residual_norm = float(np.linalg.norm(r))
    return out


def newton_nlls(residual: Callable, jacobian: Callable, hessians: Callable,
                x0, tol: float = 1e-10, max_iter: int = 100) -> NLLSResult:
    """Full Newton on ``(1/2)||r||^2``, keeping the term Gauss-Newton drops.

    Solves ``(J^T J + S) p = -J^T r`` with ``S = sum_i r_i * Hessian(r_i)``. ``hessians(x)``
    must return an array of shape ``(m, n, n)``.

    This exists to **measure** what Gauss-Newton gives up, not to be used: it needs second
    derivatives of every residual, which is exactly the cost nonlinear least squares methods
    exist to avoid. Lesson 34 runs it beside Gauss-Newton on the same problem.
    """
    x, r, _, _ = _prepare(residual, jacobian, x0)
    out = NLLSResult(x=x.copy(), residual_norm=float(np.linalg.norm(r)), iterations=0,
                     converged=False, message="", n_feval=1)
    out.history.append(out.residual_norm)

    for k in range(int(max_iter)):
        J = np.atleast_2d(np.asarray(jacobian(x), dtype=float))
        H = np.asarray(hessians(x), dtype=float)
        out.n_jeval += 1
        if H.shape != (r.size, x.size, x.size):
            raise ValueError(f"hessians returned {H.shape}, expected "
                             f"({r.size}, {x.size}, {x.size})")
        if _scaled_gradient(J, r) <= tol:
            out.converged = True
            out.message = f"scaled gradient below tolerance after {k} iterations"
            break
        grad = J.T @ r
        S = np.einsum("i,ijk->jk", r, H)
        x = x + np.linalg.solve(J.T @ J + S, -grad)
        r = np.asarray(residual(x), dtype=float).ravel()
        out.n_feval += 1
        out.iterations = k + 1
        out.history.append(float(np.linalg.norm(r)))
    else:
        out.message = f"hit the iteration limit of {max_iter}"

    out.x = x
    out.residual_norm = float(np.linalg.norm(r))
    return out


def dropped_term(residual: Callable, jacobian: Callable, hessians: Callable, x) -> dict:
    """How much of the true Hessian Gauss-Newton throws away, at the point ``x``.

    Returns ``{"norm_JtJ", "norm_S", "ratio", "residual_norm"}``. The ``ratio`` is the
    quantity that predicts the convergence rate: near zero gives quadratic convergence, near
    one gives no convergence at all, and in between gives a linear rate roughly equal to it.
    """
    x = np.asarray(x, dtype=float).ravel()
    r = np.asarray(residual(x), dtype=float).ravel()
    J = np.atleast_2d(np.asarray(jacobian(x), dtype=float))
    H = np.asarray(hessians(x), dtype=float)
    if H.shape != (r.size, x.size, x.size):
        raise ValueError(f"hessians returned {H.shape}, expected ({r.size}, {x.size}, {x.size})")
    S = np.einsum("i,ijk->jk", r, H)
    JtJ = J.T @ J
    n_j = float(np.linalg.norm(JtJ, 2))
    n_s = float(np.linalg.norm(S, 2))
    return {"norm_JtJ": n_j, "norm_S": n_s,
            "ratio": n_s / n_j if n_j > 0 else float("inf"),
            "residual_norm": float(np.linalg.norm(r))}


def observed_rate(history, tail: int = 6) -> dict:
    """Fit the convergence order from a sequence of residual or error norms.

    Uses the last ``tail`` usable entries and fits ``log e_{k+1} = q log e_k + c``. A ``q`` near
    2 is quadratic, near 1 is linear, and the fitted linear factor is reported as well so a
    linear rate can be compared against the ratio from `dropped_term`.
    """
    e = np.asarray(history, dtype=float).ravel()
    e = e[np.isfinite(e) & (e > 0)]
    if e.size < 3:
        return {"order": float("nan"), "linear_factor": float("nan"), "used": int(e.size)}
    use = e[-int(tail):] if e.size > tail else e
    a, b = use[:-1], use[1:]
    keep = (a > 1e-15) & (b > 1e-15)
    if int(np.sum(keep)) < 2:
        return {"order": float("nan"),
                "linear_factor": float(np.median(b / a)) if b.size else float("nan"),
                "used": int(np.sum(keep))}
    order = float(np.polyfit(np.log(a[keep]), np.log(b[keep]), 1)[0])
    return {"order": order, "linear_factor": float(np.median(b[keep] / a[keep])),
            "used": int(np.sum(keep))}


# ----------------------------------------------------------------------------- test problems


def exponential_model(n_points: int, params, noise: float = 0.0, rng=None,
                      t_max: float = 4.0):
    """Data from ``y = a exp(b t)``, and the residual, Jacobian and Hessians for fitting it.

    Nonlinear in ``b``, linear in ``a``, which is the smallest honest example: taking logs
    linearises it but changes the problem, as lesson 29 section 7 measured.

    Returns a dict with ``t``, ``y``, ``residual``, ``jacobian``, ``hessians`` and ``truth``.
    Any number of points, any parameters.
    """
    params = np.asarray(params, dtype=float).ravel()
    if params.size != 2:
        raise ValueError(f"the model has 2 parameters, got {params.size}")
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError(f"need at least 2 points, got {n_points}")
    t = np.linspace(0.0, float(t_max), n_points)
    y = params[0] * np.exp(params[1] * t)
    if noise:
        gen = np.random.default_rng() if rng is None else rng
        y = y + noise * gen.standard_normal(n_points)

    def residual(x):
        x = np.asarray(x, dtype=float).ravel()
        return x[0] * np.exp(x[1] * t) - y

    def jacobian(x):
        x = np.asarray(x, dtype=float).ravel()
        e = np.exp(x[1] * t)
        return np.column_stack([e, x[0] * t * e])

    def hessians(x):
        x = np.asarray(x, dtype=float).ravel()
        e = np.exp(x[1] * t)
        H = np.zeros((t.size, 2, 2))
        H[:, 0, 1] = t * e                       # d2/da db
        H[:, 1, 0] = t * e
        H[:, 1, 1] = x[0] * t ** 2 * e           # d2/db2
        return H

    return {"t": t, "y": y, "residual": residual, "jacobian": jacobian,
            "hessians": hessians, "truth": params}


def large_residual_problem(n_points: int, offset: float, rng=None):
    """A curve fit with a deliberately unfittable component, so the residual stays large.

    The data is ``sin`` data plus a constant ``offset`` that the model ``a sin(b t)`` cannot
    represent. Raising ``offset`` raises the residual at the solution without changing the
    number of parameters, which is exactly the knob Gauss-Newton's convergence depends on.
    """
    n_points = int(n_points)
    if n_points < 3:
        raise ValueError(f"need at least 3 points, got {n_points}")
    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    y = 2.0 * np.sin(1.3 * t) + float(offset)
    if rng is not None:
        y = y + 0.01 * rng.standard_normal(n_points)

    def residual(x):
        x = np.asarray(x, dtype=float).ravel()
        return x[0] * np.sin(x[1] * t) - y

    def jacobian(x):
        x = np.asarray(x, dtype=float).ravel()
        return np.column_stack([np.sin(x[1] * t), x[0] * t * np.cos(x[1] * t)])

    def hessians(x):
        x = np.asarray(x, dtype=float).ravel()
        H = np.zeros((t.size, 2, 2))
        H[:, 0, 1] = t * np.cos(x[1] * t)
        H[:, 1, 0] = t * np.cos(x[1] * t)
        H[:, 1, 1] = -x[0] * t ** 2 * np.sin(x[1] * t)
        return H

    return {"t": t, "y": y, "residual": residual, "jacobian": jacobian,
            "hessians": hessians, "truth": np.array([2.0, 1.3])}


def gps_problem(satellites, receiver, clock_bias: float = 0.0, noise: float = 0.0,
                rng=None):
    """Position from pseudoranges: the classic nonlinear least squares application.

    A receiver at ``p`` with clock bias ``b`` measures a pseudorange to each satellite,

        rho_i = ||p - s_i|| + b + noise.

    The bias enters because the receiver's clock is not synchronised with the satellites', so
    there are **four** unknowns for a position in three dimensions, and four satellites are the
    minimum. More than four makes it a least squares problem.

    ``satellites`` is ``(k, d)`` and ``receiver`` is length ``d``; ``d`` is read from the input
    rather than assumed to be 3, so the same code runs the two dimensional version that is much
    easier to draw.
    """
    S = np.atleast_2d(np.asarray(satellites, dtype=float))
    p = np.asarray(receiver, dtype=float).ravel()
    k, d = S.shape
    if p.size != d:
        raise ValueError(f"the receiver has {p.size} coordinates, the satellites have {d}")
    if k < d + 1:
        raise ValueError(f"{k} satellites cannot determine {d} coordinates plus a clock bias")

    rho = np.linalg.norm(p[None, :] - S, axis=1) + float(clock_bias)
    if noise:
        gen = np.random.default_rng() if rng is None else rng
        rho = rho + noise * gen.standard_normal(k)

    def residual(x):
        x = np.asarray(x, dtype=float).ravel()
        return np.linalg.norm(x[:d][None, :] - S, axis=1) + x[d] - rho

    def jacobian(x):
        x = np.asarray(x, dtype=float).ravel()
        diff = x[:d][None, :] - S
        dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-300)
        return np.column_stack([diff / dist[:, None], np.ones(k)])

    return {"satellites": S, "pseudoranges": rho, "residual": residual, "jacobian": jacobian,
            "truth": np.concatenate([p, [float(clock_bias)]]), "dimension": d}


def separable_fit(t, y, nonlinear_basis: Callable, x_nonlinear):
    """Variable projection: eliminate the linear parameters and fit only the nonlinear ones.

    For a model ``sum_j c_j phi_j(t; x)`` that is **linear in ``c`` and nonlinear in ``x``**,
    the optimal ``c`` for any given ``x`` is a linear least squares solve. Substituting it back
    leaves a problem in ``x`` alone, of much lower dimension and usually much better behaved.

    Returns ``{"coefficients", "residual", "residual_norm"}`` at the given ``x_nonlinear``.
    Golub and Pereyra's result is that optimising this reduced function is equivalent to
    optimising the full one, and it typically converges from far worse starting points.
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.size != y.size:
        raise ValueError(f"t has {t.size} points, y has {y.size}")
    Phi = np.atleast_2d(np.asarray(nonlinear_basis(t, x_nonlinear), dtype=float))
    if Phi.shape[0] != t.size:
        raise ValueError(f"the basis returned {Phi.shape[0]} rows for {t.size} points")
    c = np.linalg.lstsq(Phi, y, rcond=None)[0]
    r = Phi @ c - y
    return {"coefficients": c, "residual": r, "residual_norm": float(np.linalg.norm(r))}
