"""Initial value problems: what makes them solvable, and the method everything else improves on.

The problem
-----------
Given ``y' = f(t, y)`` and ``y(t0) = y0``, produce ``y`` on ``[t0, T]``. Everything in Part 10 is
a way of turning that continuous statement into a finite number of arithmetic operations, and
every method is judged by two questions: how fast its error falls as the step shrinks, and
whether it falls at all.

What has to be true first
-------------------------
**Picard and Lindelof.** If ``f`` is continuous in ``t`` and **Lipschitz** in ``y`` near the
initial point, a unique solution exists on some interval around ``t0``. Both halves matter and
both fail in ordinary cases:

- ``y' = y^(2/3)``, ``y(0) = 0`` is continuous and not Lipschitz, and has **infinitely many**
  solutions. `uniqueness_fails_on` exhibits them.
- ``y' = y^2``, ``y(0) = 1`` is Lipschitz on any bounded region and its solution blows up at
  ``t = 1``. Existence is **local**, and no numerical method can be asked for more.

A solver cannot detect either condition. It will return numbers for both, so the question of
whether those numbers mean anything is the caller's, and this module measures what happens when
they do not.

Euler's method
--------------
Replace the derivative by a forward difference and rearrange:

    y_{k+1} = y_k + h f(t_k, y_k).

That is the whole method, and it is first order: the **local** error on one step is ``O(h^2)`` and
the **global** error after ``(T - t0)/h`` steps is ``O(h)``. One power of ``h`` is lost to the step
count, exactly as one power was lost to the panel count in lesson 62.

Lesson 61 already gives the reason Euler is not enough: a first order formula cannot do better
than about eight digits, and reaching them needs a step so small that roundoff has taken over.
Every later lesson raises the order instead.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np


def as_state(y) -> np.ndarray:
    """Force a scalar or a sequence into a 1-D float array, so one code path handles both."""
    return np.atleast_1d(np.asarray(y, dtype=float)).ravel()


def integrate(f, t0: float, y0, t_end: float, steps: int, step_fn=None,
              _counter=None) -> dict:
    """March a one step method from ``t0`` to ``t_end`` in ``steps`` equal steps.

    ``step_fn(f, t, y, h, counter)`` returns the new state. The default is Euler.

    This driver is deliberately shared by every explicit method in Part 10, so comparisons
    between them differ only in the step and not in the bookkeeping around it.
    """
    n = int(steps)
    if n < 1:
        raise ValueError(f"need at least one step, got {n}")
    a = float(t0)
    b = float(t_end)
    h = (b - a) / n
    y = as_state(y0)
    ts = np.empty(n + 1)
    ys = np.empty((n + 1, y.size))
    ts[0] = a
    ys[0] = y
    take = euler_step if step_fn is None else step_fn
    for k in range(n):
        y = as_state(take(f, float(ts[k]), y, h, _counter))
        ts[k + 1] = a + (k + 1) * h
        ys[k + 1] = y
    return {"t": ts, "y": ys, "step": h, "steps": n,
            "evaluations": (_counter[0] if _counter is not None else None)}


def euler_step(f, t: float, y, h: float, _counter=None):
    """One forward Euler step. One evaluation."""
    if _counter is not None:
        _counter[0] += 1
    return as_state(y) + h * as_state(f(t, y))


def backward_euler_step(f, t: float, y, h: float, _counter=None, tol: float = 1e-12,
                        max_iter: int = 50):
    """One backward Euler step, solved by Newton on the residual.

    ``y_{k+1} = y_k + h f(t_{k+1}, y_{k+1})`` is implicit, so each step is a root finding
    problem. The Jacobian is approximated by finite differences, which is enough for the
    demonstrations here and is not what a production code does; lesson 72 says why the implicit
    solve is worth its cost anyway.
    """
    y = as_state(y)
    t_new = t + h
    guess = y.copy()
    for _ in range(int(max_iter)):
        if _counter is not None:
            _counter[0] += 1
        residual = guess - y - h * as_state(f(t_new, guess))
        if float(np.max(np.abs(residual))) <= tol * max(1.0, float(np.max(np.abs(guess)))):
            return guess
        # numerical Jacobian of the residual
        n = guess.size
        jac = np.empty((n, n))
        scale = np.maximum(np.abs(guess), 1.0) * math.sqrt(np.finfo(float).eps)
        for j in range(n):
            bumped = guess.copy()
            bumped[j] += scale[j]
            if _counter is not None:
                _counter[0] += 1
            bumped_residual = bumped - y - h * as_state(f(t_new, bumped))
            jac[:, j] = (bumped_residual - residual) / scale[j]
        try:
            guess = guess - np.linalg.solve(jac, residual)
        except np.linalg.LinAlgError:
            return guess
    return guess


# --------------------------------------------------------------------------- the hypotheses


def lipschitz_constant(f, t: float, lo, hi, samples: int = 400, seed: int = 42) -> dict:
    """Estimate the Lipschitz constant of ``f`` in ``y`` on a box, by sampling pairs.

    The Lipschitz condition is ``|f(t,u) - f(t,v)| <= L |u - v|``. Sampling pairs and taking the
    largest observed ratio gives a **lower bound** on ``L``, which is what a numerical estimate
    can honestly claim. A ratio that grows without limit as the pairs get closer together is the
    signature of the condition failing.
    """
    low = as_state(lo)
    high = as_state(hi)
    rng = np.random.default_rng(seed)
    n = int(samples)
    u = low + (high - low) * rng.random((n, low.size))
    v = low + (high - low) * rng.random((n, low.size))
    ratios = []
    for i in range(n):
        gap = float(np.max(np.abs(u[i] - v[i])))
        if gap == 0.0:
            continue
        top = float(np.max(np.abs(as_state(f(t, u[i])) - as_state(f(t, v[i])))))
        ratios.append(top / gap)
    ratios = np.asarray(ratios)
    return {"estimate": float(np.max(ratios)), "median": float(np.median(ratios)),
            "pairs": int(ratios.size),
            "note": "a sampled lower bound on L, never an upper bound"}


def lipschitz_near(f, t: float, centre, radii=None, seed: int = 42) -> dict:
    """The estimated Lipschitz constant on shrinking boxes about a point.

    For a genuinely Lipschitz ``f`` the estimate settles. For one that is not, such as
    ``y^(2/3)`` at the origin, it **grows without bound** as the box shrinks, and that growth is
    the numerical signature of non uniqueness.

    The growth is reported as a fitted exponent rather than as a threshold on the ratio of the
    first and last entries. ``y^(2/3)`` has ``L ~ r^(-1/3)``, so over five decades of radius the
    estimate rises only by a factor of 46, and a threshold set anywhere above that would call it
    bounded. The exponent is ``-1/3`` at every scale and says the right thing immediately.
    """
    rs = ([1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5] if radii is None
          else [float(v) for v in np.atleast_1d(radii)])
    c = as_state(centre)
    rows = []
    for r in rs:
        out = lipschitz_constant(f, t, c - r, c + r, seed=seed)
        rows.append((r, out["estimate"]))
    values = np.asarray([r[1] for r in rows])
    radii = np.asarray([r[0] for r in rows])
    good = values > 0
    exponent = (float(np.polyfit(np.log(radii[good]), np.log(values[good]), 1)[0])
                if int(np.sum(good)) >= 3 else float("nan"))
    return {"radius": radii, "estimate": values,
            "growth_exponent": exponent,
            "grows_without_bound": bool(exponent < -0.05),
            "note": "L ~ r^exponent as the box shrinks; a negative exponent means unbounded"}


def uniqueness_fails_on(t_end: float = 2.0, delays=None) -> dict:
    """The standard counterexample: ``y' = y^(2/3)``, ``y(0) = 0`` has infinitely many solutions.

    For any ``c >= 0`` the function that is 0 on ``[0, c]`` and ``((t-c)/3)^3`` afterwards solves
    the equation and the initial condition. They all satisfy the same problem, so no numerical
    method can be right about which one it should produce.

    Euler started exactly at 0 returns the zero solution, because ``f(0, 0) = 0`` and it never
    leaves. Started at any positive value it climbs. **The equation does not determine the
    answer, and the solver's answer is an artefact of its arithmetic.**

    Each curve's residual is checked against the closed form derivative rather than by
    differencing, so it comes out at ``10^-17`` rather than at the ``4 x 10^-6`` that
    `numpy.gradient` would contribute across the kink.
    """
    cs = ([0.0, 0.25, 0.5, 1.0] if delays is None
          else [float(v) for v in np.atleast_1d(delays)])
    t = np.linspace(0.0, float(t_end), 201)
    curves = []
    for c in cs:
        y = np.where(t <= c, 0.0, ((np.maximum(t - c, 0.0)) / 3.0) ** 3)
        # The derivative is known in closed form, so the residual is checked exactly. Getting it
        # by differencing y instead measures `np.gradient`, which on 201 points over [0, 2] with
        # a kink in the middle returns 4e-06 and makes an exact identity look approximate.
        derivative = np.where(t <= c, 0.0, ((np.maximum(t - c, 0.0)) / 3.0) ** 2)
        residual = np.max(np.abs(derivative - np.abs(y) ** (2.0 / 3.0)))
        curves.append((c, y, float(residual)))
    return {"t": t, "delays": np.asarray([c[0] for c in curves]),
            "solutions": np.stack([c[1] for c in curves]),
            "worst_residual": np.asarray([c[2] for c in curves]),
            "all_start_at_zero": bool(all(abs(c[1][0]) < 1e-15 for c in curves)),
            "count": len(curves)}


def blows_up_at(f, t0: float, y0, t_end: float, steps=None, threshold: float = 1e6) -> dict:
    """Where a solution ceases to exist, and what the solver does about it.

    ``y' = y^2`` with ``y(0) = 1`` has the exact solution ``1/(1-t)``, which is infinite at
    ``t = 1``. Existence is local and the blow up time is a property of the problem.

    A solver run past it returns numbers, and here they are ``inf``, which at least is loud.
    Refining the step moves the reported blow up **earlier**. Run to ``t_end = 2`` with a
    threshold of ``10^6``:

        steps            50      100      200      400      800
        crosses at    1.2800   1.1400   1.0800   1.0450   1.0225

    which converges on the true singularity at ``t = 1`` from above, halving the overshoot each
    time the step is halved. Each refinement tracks the true solution a little further before its
    own arithmetic overflows.

    So refinement does diagnose this one, by converging to a finite time rather than to a
    solution. **That is the useful signal**: a sequence of runs whose blow up time settles and
    whose values do not is reporting a singularity, not a step size problem.
    """
    ns = ([50, 100, 200, 400, 800] if steps is None
          else [int(v) for v in np.atleast_1d(steps)])
    rows = []
    for n in ns:
        # overflow is the phenomenon being measured, not an accident, so it is allowed
        with np.errstate(over="ignore", invalid="ignore"):
            out = integrate(f, t0, y0, t_end, n)
        big = np.flatnonzero(np.abs(out["y"][:, 0]) > threshold)
        where = float(out["t"][big[0]]) if big.size else float("nan")
        finite = bool(np.all(np.isfinite(out["y"])))
        rows.append((n, where, float(out["y"][-1, 0]), finite))
    return {"steps": np.asarray([r[0] for r in rows]),
            "first_exceeds_threshold": np.asarray([r[1] for r in rows]),
            "final_value": np.asarray([r[2] for r in rows]),
            "all_finite": np.asarray([r[3] for r in rows]),
            "threshold": float(threshold)}


# --------------------------------------------------------------------------- error


def error_against_step(f, exact, t0: float, y0, t_end: float, step_counts=None,
                       step_fn=None) -> dict:
    """Global error at the endpoint against the step count, with the fitted order.

    The fit uses only the run of finest refinements whose local slopes agree, for the reason
    lesson 62 gives: a fit through the pre-asymptotic head reports an order that is wrong rather
    than noisy.
    """
    ns = ([10, 20, 40, 80, 160, 320, 640, 1280] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    want = as_state(exact(float(t_end)))
    rows = []
    for n in ns:
        counter = [0]
        out = integrate(f, t0, y0, t_end, n, step_fn, counter)
        rows.append((n, float(np.max(np.abs(out["y"][-1] - want))), counter[0]))
    errors = np.asarray([r[1] for r in rows])
    counts = np.asarray([r[0] for r in rows], dtype=float)
    floor = 1e-13 * max(float(np.max(np.abs(want))), 1.0)
    usable = np.flatnonzero(errors > floor)
    order = float("nan")
    used = 0
    if usable.size >= 3:
        local = -np.diff(np.log(errors[usable])) / np.diff(np.log(counts[usable]))
        start = local.size - 1
        while start > 0 and abs(float(local[start - 1]) - float(local[start])) <= 0.25:
            start -= 1
        tail = usable[start:]
        if tail.size >= 3:
            order = float(-np.polyfit(np.log(counts[tail]), np.log(errors[tail]), 1)[0])
            used = int(tail.size)
    return {"steps": counts, "errors": errors,
            "evaluations": np.asarray([r[2] for r in rows]),
            "fitted_order": order, "points_used_in_the_fit": used}


def local_against_global(f, exact, t0: float, y0, t_end: float, step_counts=None,
                         step_fn=None) -> dict:
    """The one step error and the accumulated error, measured separately.

    The **local** error is what one step costs, starting from the exact solution. The **global**
    error is what remains at the end after all of them. For a method of global order ``p`` the
    local order is ``p + 1``, and the difference is the step count, exactly as lesson 62's
    composite rules lose one power of ``h`` to the panel count.

    Measuring both is the only way to see that the accumulation is what it is claimed to be
    rather than something worse.
    """
    ns = ([10, 20, 40, 80, 160, 320] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    take = euler_step if step_fn is None else step_fn
    a = float(t0)
    b = float(t_end)
    rows = []
    for n in ns:
        h = (b - a) / n
        start = as_state(exact(a))
        one = as_state(take(f, a, start, h))
        local = float(np.max(np.abs(one - as_state(exact(a + h)))))
        out = integrate(f, a, y0, b, n, step_fn)
        glob = float(np.max(np.abs(out["y"][-1] - as_state(exact(b)))))
        rows.append((n, h, local, glob))
    h_arr = np.asarray([r[1] for r in rows])
    local_arr = np.asarray([r[2] for r in rows])
    global_arr = np.asarray([r[3] for r in rows])

    def fit(values):
        good = values > 1e-14
        if int(np.sum(good)) < 3:
            return float("nan")
        return float(np.polyfit(np.log(h_arr[good]), np.log(values[good]), 1)[0])

    return {"steps": np.asarray([r[0] for r in rows]), "h": h_arr,
            "local_error": local_arr, "global_error": global_arr,
            "local_order": fit(local_arr), "global_order": fit(global_arr),
            "difference": fit(local_arr) - fit(global_arr)}


def error_bound(f, exact, t0: float, y0, t_end: float, lipschitz: float,
                second_derivative: float, step_counts=None) -> dict:
    """The classical Euler bound against the error actually observed.

    The bound is

        |e_n| <= (h M / 2L) (exp(L (t_n - t0)) - 1),

    with ``L`` a Lipschitz constant and ``M`` a bound on ``|y''|``. It is a genuine bound and it
    is enormously pessimistic, because the exponential assumes every step's error is amplified
    by the worst case at every subsequent step. Reporting the ratio says how pessimistic.
    """
    ns = ([10, 20, 40, 80, 160] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    L = float(lipschitz)
    M = float(second_derivative)
    span = float(t_end) - float(t0)
    want = as_state(exact(float(t_end)))
    rows = []
    for n in ns:
        h = span / n
        out = integrate(f, t0, y0, t_end, n)
        observed = float(np.max(np.abs(out["y"][-1] - want)))
        bound = (h * M / (2.0 * L)) * (math.exp(L * span) - 1.0)
        rows.append((n, h, observed, bound))
    observed = np.asarray([r[2] for r in rows])
    bounds = np.asarray([r[3] for r in rows])
    return {"steps": np.asarray([r[0] for r in rows]), "h": np.asarray([r[1] for r in rows]),
            "observed_error": observed, "bound": bounds,
            "bound_over_observed": bounds / np.maximum(observed, 1e-300),
            "bound_holds": bool(np.all(bounds >= observed))}


def euler_step_size_floor(f, exact, t0: float, y0, t_end: float, step_counts=None) -> dict:
    """How far shrinking the step keeps helping Euler, and what the textbook model gets wrong.

    The standard argument is lesson 61's: truncation falls like ``h``, accumulated rounding
    rises like ``n * eps = eps/h``, so the total is smallest at ``h ~ sqrt(eps)`` and the best
    achievable error is about ``10^-8``.

    **The exponent is right and the floor is not.** Measured on ``y' = -2y`` over [0, 1], the
    error is still falling cleanly at ``h = 1.5e-08``, which is exactly ``sqrt(eps)``, and is
    then ``4.0e-09``, below where the model says it should have bottomed out:

        steps        h          error
        1048576    9.5e-07     2.6e-07
        4194304    2.4e-07     6.5e-08
       16777216    6.0e-08     1.6e-08
       33554432    3.0e-08     8.1e-09
       67108864    1.5e-08     4.0e-09

    The reason is that ``n * eps`` is a **worst case**, assuming every step's rounding pushes the
    same way. They do not: the errors are effectively independent, so they accumulate like
    ``sqrt(n) * eps`` and the real floor is around ``1e-12`` rather than ``1e-08``.

    So the practical ceiling on Euler is set by cost rather than by arithmetic. Reaching eight
    digits takes 17 million steps and nine takes 67 million, which is the honest reason Part 10
    raises the order instead: **not that the accuracy is unreachable, but that it is unaffordable.**

    `reached_the_minimum` reports whether the sweep actually saw the turn, which for a decaying
    problem it usually will not.
    """
    ns = ([2 ** k for k in range(4, 25)] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    want = as_state(exact(float(t_end)))
    rows = []
    for n in ns:
        out = integrate(f, t0, y0, t_end, n)
        rows.append((n, out["step"], float(np.max(np.abs(out["y"][-1] - want)))))
    errors = np.asarray([r[2] for r in rows])
    best = int(np.argmin(errors))
    counts = np.asarray([r[0] for r in rows], dtype=float)
    keep = errors > 0
    order = (float(-np.polyfit(np.log(counts[keep]), np.log(errors[keep]), 1)[0])
             if int(np.sum(keep)) >= 3 else float("nan"))
    return {"steps": np.asarray([r[0] for r in rows]),
            "h": np.asarray([r[1] for r in rows]), "errors": errors,
            "best_steps": int(rows[best][0]), "best_h": float(rows[best][1]),
            "best_error": float(errors[best]),
            "fitted_order_over_the_sweep": order,
            "sqrt_eps": math.sqrt(np.finfo(float).eps),
            "worst_case_floor": math.sqrt(np.finfo(float).eps),
            "reached_the_minimum": bool(best < len(rows) - 1),
            "note": ("still first order at the smallest step tried means the rounding is "
                     "accumulating like sqrt(n) rather than n")}
