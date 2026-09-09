"""Picard iteration and Taylor series methods: the two constructions that need more than ``f``.

Picard
------
Rewrite the initial value problem as an integral equation,

    y(t) = y0 + integral from t0 to t of f(s, y(s)) ds,

and iterate: start from the constant ``y0`` and substitute the current guess into the right hand
side. Under the Lipschitz condition of lesson 67 the map is a contraction and the iterates
converge to the unique solution.

**This is not a numerical method and it is the proof of existence.** The contraction argument is
where Picard and Lindelof comes from, and the iterates are useful for two things: seeing why the
Lipschitz condition is the hypothesis that matters, and generating a series solution by hand for
a problem simple enough to integrate symbolically.

`picard` computes the iterates numerically by quadrature, which lets the convergence be measured
rather than asserted, and `picard_needs_lipschitz` shows the iteration failing to settle on the
non unique problem of lesson 67.

Taylor methods
--------------
Expand the solution instead of the equation:

    y(t + h) = y + h y' + h^2/2 y'' + ... + h^k/k! y^(k) + O(h^{k+1}),

and get each derivative by differentiating ``y' = f(t, y)`` along the solution:

    y''  = f_t + f_y f
    y''' = f_tt + 2 f_ty f + f_yy f^2 + f_y (f_t + f_y f)

A method of order ``k`` follows immediately, and the order is exact by construction rather than
by a convergence argument.

**The cost is the reason nobody uses them.** Order ``k`` needs every partial derivative of ``f``
up to order ``k - 1``, and the number of distinct elementary differentials, which are Butcher's
rooted trees, grows fast: 1, 2, 4, 8, 17, 37, 85 cumulative for orders 1 to 7.
`taylor_term_count` counts them. Runge-Kutta, in lesson 69, reaches the same order using only
evaluations of ``f`` itself, which is why it won.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .ivp import as_state, integrate
from .splines import evaluate as _evaluate
from .splines import not_a_knot as _not_a_knot


# --------------------------------------------------------------------------- Picard


def picard(f, t0: float, y0, t_end: float, iterations: int = 6, nodes: int = 81,
           quadrature_nodes: int = 12) -> dict:
    """Picard iterates, computed by quadrature on a fixed grid.

    Each iterate is ``y0 + integral of f(s, previous(s)) ds``, evaluated at every grid point by
    Gauss-Legendre on that point's subinterval, with the previous iterate supplied by
    interpolation. The integral is computed accurately enough that what is measured is the
    **iteration's** convergence and not the quadrature's.
    """
    from . import gaussquad as _gq

    a = float(t0)
    b = float(t_end)
    m = int(nodes)
    if m < 2:
        raise ValueError(f"need at least two grid points, got {m}")
    t = np.linspace(a, b, m)
    start = as_state(y0)
    current = np.tile(start, (m, 1))
    history = [current.copy()]
    x_ref, w_ref = _gq.nodes_and_weights(int(quadrature_nodes))
    for _ in range(int(iterations)):
        nxt = np.empty_like(current)
        nxt[0] = start
        for i in range(1, m):
            half = 0.5 * (t[i] - t[0])
            mid = 0.5 * (t[i] + t[0])
            s = mid + half * x_ref
            # The previous iterate at the quadrature nodes. A not-a-knot cubic spline, not
            # linear interpolation: with 81 grid points the linear version has an error of
            # 1.2e-05, which is larger than the iteration's own error after three steps and
            # makes the exactness check in `picard_matches_the_series` impossible to see.
            values = np.stack([_evaluate(_not_a_knot(t, current[:, j]), s)
                               for j in range(start.size)], axis=1)
            contrib = np.stack([as_state(f(float(s[k]), values[k]))
                                for k in range(s.size)])
            nxt[i] = start + half * (w_ref @ contrib)
        current = nxt
        history.append(current.copy())
    stacked = np.stack(history)
    changes = np.asarray([float(np.max(np.abs(stacked[k + 1] - stacked[k])))
                          for k in range(stacked.shape[0] - 1)])
    return {"t": t, "iterates": stacked, "final": current,
            "change_per_iteration": changes,
            "converging": bool(changes.size >= 2 and changes[-1] < changes[0])}


def picard_matches_the_series(t_end: float = 0.8, iterations: int = 6) -> dict:
    """Picard on ``y' = y``, ``y(0) = 1``, whose iterates are the exponential's partial sums.

    Substituting the constant 1 gives ``1 + t``; substituting that gives ``1 + t + t^2/2``, and
    so on. **The ``k``th Picard iterate is exactly the degree ``k`` Taylor polynomial of
    ``e^t``**, which makes this the one case where the iteration's answer is known in closed
    form and the convergence can be checked term by term.
    """
    out = picard(lambda t, y: as_state(y), 0.0, 1.0, t_end, iterations)
    t = out["t"]
    rows = []
    for k in range(out["iterates"].shape[0]):
        partial = np.zeros_like(t)
        for j in range(k + 1):
            partial = partial + t ** j / math.factorial(j)
        got = out["iterates"][k][:, 0]
        rows.append((k, float(np.max(np.abs(got - partial))),
                     float(np.max(np.abs(got - np.exp(t))))))
    return {"iteration": np.asarray([r[0] for r in rows]),
            "gap_to_the_taylor_partial_sum": np.asarray([r[1] for r in rows]),
            "gap_to_the_exact_solution": np.asarray([r[2] for r in rows]),
            "iterates_are_partial_sums": bool(max(r[1] for r in rows) < 1e-9)}


def picard_needs_lipschitz(t_end: float = 1.0, iterations: int = 8) -> dict:
    """The iteration on ``y' = y^(2/3)``, ``y(0) = 0``, where the contraction argument fails.

    Started at the constant 0, every iterate is 0: the zero solution is a genuine fixed point.
    Started anywhere else the iterates move, and they do not settle onto anything, because the
    map is not a contraction near the origin and the problem has infinitely many solutions for
    them to be pulled between.

    **The iteration cannot fail loudly.** It returns a sequence of functions either way, and
    only the Lipschitz estimate of lesson 67 distinguishes the two cases in advance.
    """
    def rough(t, y):
        v = as_state(y)
        return np.sign(v) * np.abs(v) ** (2.0 / 3.0)

    from_zero = picard(rough, 0.0, 0.0, t_end, iterations)
    from_small = picard(rough, 0.0, 1e-6, t_end, iterations)
    return {"t": from_zero["t"],
            "from_zero_final": from_zero["final"][:, 0],
            "from_small_final": from_small["final"][:, 0],
            "from_zero_stays_zero": bool(np.max(np.abs(from_zero["final"])) < 1e-14),
            "from_zero_changes": from_zero["change_per_iteration"],
            "from_small_changes": from_small["change_per_iteration"],
            "from_small_settles": bool(
                from_small["change_per_iteration"][-1]
                < 1e-3 * max(from_small["change_per_iteration"][0], 1e-300))}


# --------------------------------------------------------------------------- Taylor methods


def taylor_step(derivatives, order: int):
    """Build a Taylor method of the given order from a callable giving ``y^(k)(t, y)``.

    ``derivatives(t, y, k)`` must return the ``k``th total derivative of the solution along it,
    with ``k = 1`` being ``f`` itself. The returned function has the signature every one step
    method in Part 10 uses, so it drops straight into `nalib.ivp.integrate`.
    """
    k = int(order)
    if k < 1:
        raise ValueError(f"order must be at least 1, got {k}")

    def step(f, t, y, h, _counter=None):
        total = as_state(y)
        for j in range(1, k + 1):
            if _counter is not None:
                _counter[0] += 1
            total = total + (h ** j / math.factorial(j)) * as_state(derivatives(t, y, j))
        return total

    return step


def taylor_term_count(orders=None) -> dict:
    """How many distinct elementary differentials a Taylor method of each order needs.

    These are the rooted trees of Butcher's theory, and their count is 1, 1, 2, 4, 9, 20, 48 for
    the number of new terms at each order, giving cumulative totals 1, 2, 4, 8, 17, 37, 85.

    The point is the growth. A Taylor method of order 5 needs every partial derivative of ``f``
    up to order 4, and there is no way to supply them except by hand or by symbolic
    differentiation, which lesson 61 exercise 5.2 showed suffers expression swell.
    """
    ks = ([1, 2, 3, 4, 5, 6, 7, 8] if orders is None
          else [int(v) for v in np.atleast_1d(orders)])
    # a(n) = number of rooted trees with n nodes: 1, 1, 2, 4, 9, 20, 48, 115, 286
    trees = [1, 1, 2, 4, 9, 20, 48, 115, 286, 719, 1842]
    rows = []
    for k in ks:
        new = trees[k - 1] if k - 1 < len(trees) else None
        total = sum(trees[:k]) if k <= len(trees) else None
        rows.append((k, new, total))
    return {"order": np.asarray([r[0] for r in rows]),
            "new_terms": np.asarray([r[1] for r in rows]),
            "cumulative_terms": np.asarray([r[2] for r in rows]),
            "note": "rooted tree counts; these are the elementary differentials of f"}


def taylor_orders_are_exact(f, derivatives, exact, t0: float, y0, t_end: float,
                            orders=None, step_counts=None) -> dict:
    """The measured global order of Taylor methods, against the order they were built for.

    A Taylor method's order is exact by construction: it matches the solution's Taylor expansion
    through ``h^k`` and misses at ``h^{k+1}``, so the global order is ``k`` with no convergence
    argument needed. Measuring it is a check on the derivative formulas rather than on the
    theory, and a wrong derivative shows up immediately as a missing order.
    """
    from .ivp import error_against_step

    ks = ([1, 2, 3, 4] if orders is None else [int(v) for v in np.atleast_1d(orders)])
    ns = ([10, 20, 40, 80, 160, 320] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    rows = []
    for k in ks:
        out = error_against_step(f, exact, t0, y0, t_end, ns, taylor_step(derivatives, k))
        rows.append((k, out["fitted_order"], float(out["errors"][-1]),
                     int(out["evaluations"][-1])))
    fitted = np.asarray([r[1] for r in rows])
    return {"order": np.asarray([r[0] for r in rows]), "fitted_order": fitted,
            "finest_error": np.asarray([r[2] for r in rows]),
            "derivative_evaluations": np.asarray([r[3] for r in rows]),
            "all_match": bool(np.all(np.abs(fitted - np.asarray([r[0] for r in rows])) < 0.3))}


def taylor_against_euler_at_equal_cost(f, derivatives, exact, t0: float, y0, t_end: float,
                                       budgets=None, order: int = 4) -> dict:
    """A Taylor method against Euler at the same number of derivative evaluations.

    A Taylor method of order ``k`` costs ``k`` evaluations per step, so at a fixed budget it
    takes ``k`` times fewer steps. It still wins by an enormous margin, because ``h^k`` beats
    ``h`` however the budget is split.

    **The comparison flatters Taylor**, and deliberately so: it counts a call to ``y'''`` as one
    evaluation, when in practice that expression may be far more expensive than ``f`` and has to
    be derived by hand first. Lesson 69 makes the honest comparison, against a method that needs
    only ``f``.
    """
    from .ivp import integrate as _integrate

    bs = ([40, 80, 160, 320, 640] if budgets is None
          else [int(v) for v in np.atleast_1d(budgets)])
    want = as_state(exact(float(t_end)))
    k = int(order)
    rows = []
    for budget in bs:
        euler = _integrate(f, t0, y0, t_end, budget)
        taylor = _integrate(f, t0, y0, t_end, max(1, budget // k),
                            taylor_step(derivatives, k))
        rows.append((budget,
                     float(np.max(np.abs(euler["y"][-1] - want))),
                     float(np.max(np.abs(taylor["y"][-1] - want))),
                     max(1, budget // k)))
    euler_err = np.asarray([r[1] for r in rows])
    taylor_err = np.asarray([r[2] for r in rows])
    return {"evaluations": np.asarray([r[0] for r in rows]),
            "euler_error": euler_err, "taylor_error": taylor_err,
            "taylor_steps": np.asarray([r[3] for r in rows]),
            "advantage": euler_err / np.maximum(taylor_err, 1e-300),
            "taylor_wins": bool(np.all(taylor_err < euler_err))}
