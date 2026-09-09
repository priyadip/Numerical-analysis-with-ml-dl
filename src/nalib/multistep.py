"""Multistep methods: reuse the past instead of recomputing it, and the price of doing so.

Where they come from
--------------------
A Runge-Kutta step throws away every stage it computed. A multistep method keeps the previous
derivative values and builds the next step out of them, so **one evaluation of ``f`` per step at
any order**.

The construction is lesson 62's quadrature, applied to the integral form:

    y_{n+1} = y_n + integral from t_n to t_{n+1} of f(s, y(s)) ds.

Interpolate ``f`` through the past points and integrate the interpolant exactly.

- Through ``f_n, f_{n-1}, ...``, all in the past: **Adams-Bashforth**, explicit.
- Through ``f_{n+1}, f_n, ...``, including the unknown: **Adams-Moulton**, implicit.

The implicit family has a much smaller error constant at the same order and a much larger
stability region, at the cost of a solve per step. Running the explicit one as a **predictor** and
one implicit sweep as a **corrector** buys most of both for two evaluations.

The two things that can go wrong
--------------------------------
A multistep method is a linear recurrence, so it has its own solutions independent of the
differential equation. **Zero-stability** is the requirement that those parasitic solutions do not
grow: the roots of the characteristic polynomial must satisfy ``|r| <= 1`` with any root on the
circle simple. `root_condition` checks it, and `unstable_example` exhibits a consistent method of
order 3 whose parasitic root is -5 and which therefore diverges, more violently at every
refinement.

**The Dahlquist barriers** then say how far this can be pushed:

- A zero-stable ``k`` step method has order at most ``k + 1`` for odd ``k`` and ``k + 2`` for
  even ``k``, the **first barrier**.
- No A-stable linear multistep method has order above 2, and the best one is the trapezoid
  rule, the **second barrier**. Lesson 72 takes that up.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .ivp import as_state


def bdf_coefficients(steps: int) -> tuple:
    """The backward differentiation formula of ``steps`` steps, as exact ``Fraction`` lists.

    BDF-k comes from differentiating the interpolating polynomial rather than integrating it:

        ``sum_(j=1)^(k) (1/j) (1 - E^-1)^j y_(n+1) = h f_(n+1)``

    where ``E^-1`` is the backward shift. Expanding the binomial and normalising by the leading
    coefficient ``H_k = 1 + 1/2 + ... + 1/k`` gives ``alpha`` and a ``beta`` that is zero except
    in its first entry, which is what makes every BDF method implicit and one stage.

    The family is the backbone of every stiff solver there is, and `nalib.stability.bdf_angles`
    measures the property that makes it so and the property that limits it.
    """
    k = int(steps)
    if k < 1:
        raise ValueError(f"need at least one step, got {k}")
    alpha = [Fraction(0)] * (k + 1)
    for j in range(1, k + 1):
        for i in range(j + 1):
            alpha[i] += Fraction((-1) ** i * math.comb(j, i), j)
    scale = alpha[0]
    alpha = [v / scale for v in alpha]
    beta = [Fraction(0)] * (k + 1)
    beta[0] = Fraction(1) / scale
    return alpha, beta


def adams_bashforth_coefficients(steps: int) -> list:
    """Exact Adams-Bashforth coefficients on ``steps`` past points.

    ``y_{n+1} = y_n + h sum_j b_j f_{n-j}``, with ``b_j`` obtained by integrating the Lagrange
    basis through the points ``t_n, t_{n-1}, ...`` over ``[t_n, t_{n+1}]``.

    Computed in exact rational arithmetic, so the classical fractions are recognisable: one step
    gives ``[1]``, two ``[3/2, -1/2]``, three ``[23/12, -16/12, 5/12]``.
    """
    k = int(steps)
    if k < 1:
        raise ValueError(f"need at least one past point, got {k}")
    # nodes at 0, -1, -2, ... in units of h; integrate each basis over [0, 1]
    nodes = [Fraction(-j) for j in range(k)]
    out = []
    for i in range(k):
        poly = [Fraction(1)] + [Fraction(0)] * (k - 1)
        degree, denominator = 0, Fraction(1)
        for j in range(k):
            if j == i:
                continue
            new = [Fraction(0)] * k
            for m in range(degree + 1):
                new[m + 1] += poly[m]
                new[m] -= nodes[j] * poly[m]
            poly, degree = new, degree + 1
            denominator *= (nodes[i] - nodes[j])
        out.append(sum(poly[m] / Fraction(m + 1) for m in range(k)) / denominator)
    return out


def adams_moulton_coefficients(steps: int) -> list:
    """Exact Adams-Moulton coefficients including the unknown point.

    ``y_{n+1} = y_n + h sum_j b_j f_{n+1-j}``, with the interpolation through
    ``t_{n+1}, t_n, ...``. One past point gives the trapezoid rule ``[1/2, 1/2]``; two give
    ``[5/12, 8/12, -1/12]``.

    ``steps`` counts the points **excluding** the new one, so ``steps = 1`` returns two
    coefficients.
    """
    k = int(steps)
    if k < 0:
        raise ValueError(f"need a non-negative number of past points, got {k}")
    nodes = [Fraction(1 - j) for j in range(k + 1)]
    n = k + 1
    out = []
    for i in range(n):
        poly = [Fraction(1)] + [Fraction(0)] * (n - 1)
        degree, denominator = 0, Fraction(1)
        for j in range(n):
            if j == i:
                continue
            new = [Fraction(0)] * n
            for m in range(degree + 1):
                new[m + 1] += poly[m]
                new[m] -= nodes[j] * poly[m]
            poly, degree = new, degree + 1
            denominator *= (nodes[i] - nodes[j])
        out.append(sum(poly[m] / Fraction(m + 1) for m in range(n)) / denominator)
    return out


#: The classical named methods, as ``name -> (alpha, beta, order)`` in the form
#: ``sum_j alpha_j y_{n+1-j} = h sum_j beta_j f_{n+1-j}``, index 0 being the new point.
NAMED = {
    "euler": ([1, -1], [0, 1], 1),
    "backward euler": ([1, -1], [1, 0], 1),
    "trapezoid": ([1, -1], [Fraction(1, 2), Fraction(1, 2)], 2),
    "ab2": ([1, -1], [0, Fraction(3, 2), Fraction(-1, 2)], 2),
    "ab3": ([1, -1], [0, Fraction(23, 12), Fraction(-16, 12), Fraction(5, 12)], 3),
    "ab4": ([1, -1], [0, Fraction(55, 24), Fraction(-59, 24), Fraction(37, 24),
                      Fraction(-9, 24)], 4),
    "am2": ([1, -1], [Fraction(1, 2), Fraction(1, 2)], 2),
    "am3": ([1, -1], [Fraction(5, 12), Fraction(8, 12), Fraction(-1, 12)], 3),
    "am4": ([1, -1], [Fraction(9, 24), Fraction(19, 24), Fraction(-5, 24),
                      Fraction(1, 24)], 4),
    "milne simpson": ([1, 0, -1], [Fraction(1, 3), Fraction(4, 3), Fraction(1, 3)], 4),
    "bdf2": ([1, Fraction(-4, 3), Fraction(1, 3)], [Fraction(2, 3)], 2),
    "bdf3": ([1, Fraction(-18, 11), Fraction(9, 11), Fraction(-2, 11)],
             [Fraction(6, 11)], 3),
    # consistent, order 3, and useless: its parasitic root is -5
    "unstable order two": ([1, 4, -5], [0, 4, 2], 3),
}


def method(name: str) -> tuple:
    """A named method as float arrays ``(alpha, beta)`` plus its stated order."""
    key = str(name).lower()
    if key not in NAMED:
        raise ValueError(f"unknown method {name!r}, expected one of {sorted(NAMED)}")
    alpha, beta, order = NAMED[key]
    return (np.asarray([float(v) for v in alpha]),
            np.asarray([float(v) for v in beta]), int(order))


def is_explicit(name: str) -> bool:
    """A method is explicit when the new point carries no derivative weight."""
    _alpha, beta, _order = method(name)
    return bool(beta[0] == 0.0)


# --------------------------------------------------------------------------- order and stability


def order_conditions(alpha, beta, up_to: int = 6, tol: float = 1e-12) -> dict:
    """The conditions a linear multistep method satisfies, checked one power at a time.

    Applying the method to ``y = t^q`` and requiring exactness gives, with the index ``j``
    counting backwards from the new point,

        q = 0:  sum_j alpha_j = 0
        q > 0:  sum_j alpha_j (1-j)^q = q sum_j beta_j (1-j)^(q-1)

    The largest ``q`` for which every condition up to it holds is the order. Reporting each
    condition separately localises a typing error the way lesson 69's tableau check does.
    """
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    ja = 1.0 - np.arange(a.size, dtype=float)
    jb = 1.0 - np.arange(b.size, dtype=float)
    rows = []
    for q in range(int(up_to) + 1):
        left = float(np.sum(a * ja ** q))
        right = 0.0 if q == 0 else float(q * np.sum(b * jb ** (q - 1)))
        scale = max(1.0, float(np.sum(np.abs(a * ja ** q))))
        rows.append((q, left, right, abs(left - right) <= tol * scale))
    achieved = 0
    for q, _l, _r, ok in rows:
        if ok:
            achieved = q
        else:
            break
    return {"power": np.asarray([r[0] for r in rows]),
            "left": np.asarray([r[1] for r in rows]),
            "right": np.asarray([r[2] for r in rows]),
            "holds": np.asarray([r[3] for r in rows]),
            "order": achieved}


def root_condition(alpha, tol: float = 1e-10) -> dict:
    """The roots of the first characteristic polynomial, and whether they behave.

    ``rho(r) = sum_j alpha_j r^(k-j)``. Zero-stability requires every root to satisfy
    ``|r| <= 1``, with any root of modulus exactly 1 being simple.

    **Consistency forces a root at exactly 1**, the principal root, which is the one tracking the
    differential equation. Every other root is parasitic: it is a solution of the recurrence with
    nothing to do with the problem, and if it grows the method diverges however high its order.
    """
    a = np.asarray(alpha, dtype=float)
    roots = np.roots(a) if a.size > 1 else np.asarray([], dtype=complex)
    moduli = np.abs(roots)
    on_circle = np.flatnonzero(np.abs(moduli - 1.0) <= tol)
    repeated = False
    for i in on_circle:
        for j in on_circle:
            if i < j and abs(roots[i] - roots[j]) <= tol:
                repeated = True
    principal = bool(np.any(np.abs(roots - 1.0) <= tol)) if roots.size else False
    return {"roots": roots, "moduli": moduli,
            "largest_modulus": float(np.max(moduli)) if roots.size else 0.0,
            "has_the_principal_root": principal,
            "roots_on_the_circle_are_simple": not repeated,
            "zero_stable": bool(roots.size == 0
                                or (float(np.max(moduli)) <= 1.0 + tol and not repeated)),
            "parasitic": roots[np.abs(roots - 1.0) > tol] if roots.size else roots}


def all_named_methods(tol: float = 1e-10) -> dict:
    """Every method in `NAMED`: its measured order, its roots, and whether it is usable."""
    rows = []
    for name in NAMED:
        alpha, beta, claimed = method(name)
        oc = order_conditions(alpha, beta)
        rc = root_condition(alpha, tol)
        # the step count is how far back the method reaches, which for the Adams family is
        # set by beta rather than alpha: AB4 has alpha = [1, -1] and four past derivatives
        reach = max(int(alpha.size), int(beta.size)) - 1
        rows.append((name, claimed, oc["order"], rc["zero_stable"],
                     rc["largest_modulus"], is_explicit(name), reach))
    return {"names": [r[0] for r in rows],
            "claimed_order": np.asarray([r[1] for r in rows]),
            "measured_order": np.asarray([r[2] for r in rows]),
            "zero_stable": np.asarray([r[3] for r in rows]),
            "largest_root_modulus": np.asarray([r[4] for r in rows]),
            "explicit": np.asarray([r[5] for r in rows]),
            "steps": np.asarray([r[6] for r in rows]),
            "orders_all_match": bool(all(r[1] == r[2] for r in rows))}


def first_dahlquist_barrier(max_steps: int = 8) -> dict:
    """The largest order a zero-stable ``k`` step method can have.

    The barrier is ``k + 1`` for odd ``k`` and ``k + 2`` for even ``k``. So convergence is not
    the constraint on multistep order; **zero-stability is**, and a method can be made to any
    order at all if one is willing to let it diverge, which is what `unstable_example` shows.
    """
    ks = list(range(1, int(max_steps) + 1))
    return {"steps": np.asarray(ks),
            "maximum_order": np.asarray([k + 2 if k % 2 == 0 else k + 1 for k in ks]),
            "note": "k + 1 for odd k, k + 2 for even k"}


def unstable_example() -> dict:
    """A consistent method of order 3 that diverges, because its parasitic root is -5.

    ``y_{n+1} + 4 y_n - 5 y_{n-1} = h (4 f_n + 2 f_{n-1})`` satisfies the order conditions
    through ``q = 3``. Its first characteristic polynomial is ``r^2 + 4r - 5 = (r-1)(r+5)``, so
    besides the principal root at 1 there is one at -5.

    Measured on ``y' = -y`` over [0, 1], started by RK4, the error runs 7.3, 4.7e+06, 2.9e+19,
    1.7e+46 at 10, 20, 40 and 80 steps. **Refining makes it worse**, which no convergent method
    does and which is the clearest possible symptom.

    **The parasitic solution grows like 5^n and swamps everything.** Order says nothing about
    convergence without zero-stability, and this is the standard demonstration.
    """
    alpha, beta, claimed = method("unstable order two")
    oc = order_conditions(alpha, beta)
    rc = root_condition(alpha)
    return {"alpha": alpha, "beta": beta, "claimed_order": claimed,
            "measured_order": oc["order"], "roots": rc["roots"],
            "largest_modulus": rc["largest_modulus"],
            "zero_stable": rc["zero_stable"],
            "consistent": bool(oc["order"] >= 1)}


# --------------------------------------------------------------------------- running them


def start_with(f, t0: float, y0, h: float, count: int, starter=None, _counter=None):
    """Generate the first few points a multistep method needs, with a one step method.

    A ``k`` step method cannot start itself. The starter's **local** order is what matters, so a
    starter of global order ``q`` contributes start up values with error ``O(h^(q+1))`` and the
    run achieves ``min(p, q+1)``.

    That off by one is worth knowing precisely, because it is the difference between needing a
    starter of the same order as the method and needing one an order below. AB4 started by the
    third order Kutta method reaches its full order 4; started by Heun it reaches 3.
    `starting_values_matter` measures the whole ladder.
    """
    from .ivp import integrate
    from .rungekutta import named_step

    take = named_step("rk4") if starter is None else starter
    ts = [float(t0)]
    ys = [as_state(y0)]
    y = as_state(y0)
    for i in range(int(count) - 1):
        t = float(t0) + i * h
        y = as_state(take(f, t, y, h, _counter))
        ts.append(t + h)
        ys.append(y)
    return np.asarray(ts), np.stack(ys)


def solve_explicit(f, t0: float, y0, t_end: float, steps: int, name: str = "ab4",
                   starter=None, _counter=None) -> dict:
    """Run an explicit multistep method, started by a one step method.

    One evaluation of ``f`` per step after the start up, whatever the order, which is the entire
    reason these methods exist.
    """
    alpha, beta, _order = method(name)
    if not is_explicit(name):
        raise ValueError(f"{name} is implicit; use solve_predictor_corrector")
    k = max(alpha.size, beta.size) - 1
    n = int(steps)
    a, b = float(t0), float(t_end)
    h = (b - a) / n
    ts, ys = start_with(f, a, y0, h, k, starter, _counter)
    history = [as_state(f(float(ts[i]), ys[i])) for i in range(k)]
    if _counter is not None:
        _counter[0] += k
    t_list = list(ts)
    y_list = [row for row in ys]
    for step in range(k - 1, n):
        y_now = y_list[-1]
        total = -alpha[1] * y_now
        for j in range(2, alpha.size):
            total = total - alpha[j] * y_list[-j]
        for j in range(1, beta.size):
            total = total + h * beta[j] * history[-j]
        y_new = total / alpha[0]
        t_new = a + (step + 1) * h
        t_list.append(t_new)
        y_list.append(y_new)
        if _counter is not None:
            _counter[0] += 1
        history.append(as_state(f(t_new, y_new)))
    return {"t": np.asarray(t_list), "y": np.stack(y_list), "step": h, "steps": n,
            "start_up_points": k, "name": name,
            "evaluations": (_counter[0] if _counter is not None else None)}


def solve_predictor_corrector(f, t0: float, y0, t_end: float, steps: int,
                              predictor: str = "ab4", corrector: str = "am4",
                              sweeps: int = 1, starter=None, _counter=None) -> dict:
    """Predict with an explicit method, correct with an implicit one, in ``sweeps`` passes.

    Two evaluations of ``f`` per step for one sweep. The corrector's much smaller error constant
    is inherited without ever solving a nonlinear equation, which is why this arrangement, and
    not the implicit method itself, is what classical codes used.

    More sweeps converge towards the true implicit solution and are almost never worth it: the
    second sweep changes the answer by ``O(h)`` times the first, and the accuracy gain is far
    smaller than simply halving the step.
    """
    pa, pb, _po = method(predictor)
    ca, cb, _co = method(corrector)
    k = max(pa.size, pb.size, ca.size, cb.size) - 1
    n = int(steps)
    a, b = float(t0), float(t_end)
    h = (b - a) / n
    ts, ys = start_with(f, a, y0, h, k, starter, _counter)
    history = [as_state(f(float(ts[i]), ys[i])) for i in range(k)]
    if _counter is not None:
        _counter[0] += k
    t_list, y_list = list(ts), [row for row in ys]
    for step in range(k - 1, n):
        y_now = y_list[-1]
        t_new = a + (step + 1) * h
        predicted = -pa[1] * y_now
        for j in range(2, pa.size):
            predicted = predicted - pa[j] * y_list[-j]
        for j in range(1, pb.size):
            predicted = predicted + h * pb[j] * history[-j]
        predicted = predicted / pa[0]
        y_new = predicted
        for _ in range(int(sweeps)):
            if _counter is not None:
                _counter[0] += 1
            f_new = as_state(f(t_new, y_new))
            corrected = -ca[1] * y_now
            for j in range(2, ca.size):
                corrected = corrected - ca[j] * y_list[-j]
            corrected = corrected + h * cb[0] * f_new
            for j in range(1, cb.size):
                corrected = corrected + h * cb[j] * history[-j]
            y_new = corrected / ca[0]
        t_list.append(t_new)
        y_list.append(y_new)
        if _counter is not None:
            _counter[0] += 1
        history.append(as_state(f(t_new, y_new)))
    return {"t": np.asarray(t_list), "y": np.stack(y_list), "step": h, "steps": n,
            "start_up_points": k, "predictor": predictor, "corrector": corrector,
            "sweeps": int(sweeps),
            "evaluations": (_counter[0] if _counter is not None else None)}


# --------------------------------------------------------------------------- measurements


def measured_order(f, exact, t0: float, y0, t_end: float, name: str = "ab4",
                   step_counts=None, starter=None) -> dict:
    """The order an explicit multistep method achieves when run."""
    from .ivp import as_state as _as

    ns = ([20, 40, 80, 160, 320, 640] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    want = _as(exact(float(t_end)))
    rows = []
    for n in ns:
        counter = [0]
        out = solve_explicit(f, t0, y0, t_end, n, name, starter, counter)
        rows.append((n, float(np.max(np.abs(out["y"][-1] - want))), counter[0]))
    errors = np.asarray([r[1] for r in rows])
    counts = np.asarray([r[0] for r in rows], dtype=float)
    keep = errors > 1e-13
    order = (float(-np.polyfit(np.log(counts[keep]), np.log(errors[keep]), 1)[0])
             if int(np.sum(keep)) >= 3 else float("nan"))
    _alpha, _beta, claimed = method(name)
    return {"steps": counts, "errors": errors,
            "evaluations": np.asarray([r[2] for r in rows]),
            "fitted_order": order, "claimed_order": claimed,
            "matches": bool(abs(order - claimed) < 0.35)}


def starting_values_matter(f, exact, t0: float, y0, t_end: float, name: str = "ab4",
                           step_counts=None) -> dict:
    """The same method, started by methods of different orders.

    The achieved order is ``min(p, q+1)`` where ``q`` is the starter's global order, because the
    start up commits a **local** error of ``O(h^(q+1))`` once and nothing later removes it.

    Measured for AB4, which has order 4:

        starter                observed order
        Euler, order 1              2.01
        Heun, order 2               3.05
        Kutta third, order 3        3.99
        RK4, order 4                4.00

    **The run looks like it is working.** It converges, it reports a clean order, and the order
    is simply the wrong one. Nothing announces it except measuring the order and comparing it
    against what the method should give.
    """
    from .ivp import euler_step
    from .rungekutta import named_step

    starters = (("euler, order 1", euler_step),
                ("heun, order 2", named_step("heun")),
                ("kutta third, order 3", named_step("kutta third")),
                ("rk4, order 4", named_step("rk4")))
    rows = []
    for label, starter in starters:
        out = measured_order(f, exact, t0, y0, t_end, name, step_counts, starter)
        rows.append((label, out["fitted_order"], float(out["errors"][-1])))
    _alpha, _beta, claimed = method(name)
    starter_orders = np.asarray([1, 2, 3, 4])
    fitted = np.asarray([r[1] for r in rows])
    predicted = np.minimum(starter_orders + 1, claimed)
    return {"starters": [r[0] for r in rows],
            "starter_order": starter_orders,
            "fitted_order": fitted,
            "predicted_order": predicted,
            "finest_error": np.asarray([r[2] for r in rows]),
            "method_order": claimed,
            "best_achieved": float(np.max(fitted)),
            "matches_min_p_q_plus_one": bool(np.all(np.abs(fitted - predicted) < 0.35))}


def unstable_method_diverges(t_end: float = 1.0, step_counts=None) -> dict:
    """Run the order 3 zero-unstable method and watch it fail.

    Refining the step makes it **worse**, which is the opposite of every other method in Part 10
    and is the clearest possible signature: more steps means more doublings of the parasitic
    mode.
    """
    from .rungekutta import named_step

    ns = ([10, 20, 40, 80] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    f = lambda t, y: -as_state(y)
    want = math.exp(-float(t_end))
    rows = []
    for n in ns:
        with np.errstate(over="ignore", invalid="ignore"):
            out = solve_explicit(f, 0.0, 1.0, t_end, n, "unstable order two",
                                 named_step("rk4"))
        value = float(out["y"][-1, 0])
        rows.append((n, value, abs(value - want) if np.isfinite(value) else float("inf")))
    errors = np.asarray([r[2] for r in rows])
    return {"steps": np.asarray([r[0] for r in rows]),
            "final_value": np.asarray([r[1] for r in rows]),
            "errors": errors,
            "gets_worse_with_refinement": bool(np.all(np.diff(errors) > 0.0)),
            "exact": want}


def implicit_error_constant(orders=None) -> dict:
    """The leading error constant of Adams-Bashforth against Adams-Moulton at each order.

    The implicit family's constant is smaller at every order, by a factor that grows: 5 at
    order 2, 9 at order 3, 13.2 at order 4 and 17.6 at order 5. **That is why one corrector
    sweep is worth an extra evaluation**, and it is also why the implicit methods have the better
    stability regions.
    """
    ks = ([2, 3, 4, 5] if orders is None else [int(v) for v in np.atleast_1d(orders)])
    rows = []
    for p in ks:
        ab = adams_bashforth_coefficients(p)
        am = adams_moulton_coefficients(p - 1)
        # the local error constant is the first Taylor coefficient the rule misses
        def constant(b, offset):
            nodes = [Fraction(offset - j) for j in range(len(b))]
            got = sum(b[i] * nodes[i] ** p for i in range(len(b)))
            want = Fraction(1, p + 1)
            return abs(float(got - want)) / math.factorial(p)

        rows.append((p, constant(ab, 0), constant(am, 1)))
    ab_c = np.asarray([r[1] for r in rows])
    am_c = np.asarray([r[2] for r in rows])
    return {"order": np.asarray([r[0] for r in rows]),
            "adams_bashforth": ab_c, "adams_moulton": am_c,
            "ratio": ab_c / np.maximum(am_c, 1e-300),
            "implicit_is_smaller": bool(np.all(am_c < ab_c))}


def against_runge_kutta(f, exact, t0: float, y0, t_end: float, budgets=None) -> dict:
    """Multistep against Runge-Kutta at equal evaluations of ``f``.

    AB4 costs one evaluation per step and RK4 costs four, so at a fixed budget AB4 takes four
    times as many steps. Both are order 4, so AB4's error is ``4^-4 = 1/256`` of RK4's, in
    principle.

    **In practice the gap is smaller**, because AB4's error constant is much larger than RK4's,
    and because AB4 needs a start up and cannot change its step without re-starting. The
    measurement says which effect wins.
    """
    from .ivp import integrate
    from .rungekutta import named_step

    bs = ([80, 160, 320, 640, 1280] if budgets is None
          else [int(v) for v in np.atleast_1d(budgets)])
    want = as_state(exact(float(t_end)))
    rows = []
    for budget in bs:
        rk_counter = [0]
        rk_out = integrate(f, t0, y0, t_end, max(1, budget // 4), named_step("rk4"),
                           rk_counter)
        ab_counter = [0]
        ab_out = solve_explicit(f, t0, y0, t_end, max(5, budget - 12), "ab4",
                                None, ab_counter)
        pc_counter = [0]
        pc_out = solve_predictor_corrector(f, t0, y0, t_end, max(5, budget // 2),
                                           "ab4", "am4", 1, None, pc_counter)
        rows.append((budget,
                     float(np.max(np.abs(rk_out["y"][-1] - want))), rk_counter[0],
                     float(np.max(np.abs(ab_out["y"][-1] - want))), ab_counter[0],
                     float(np.max(np.abs(pc_out["y"][-1] - want))), pc_counter[0]))
    return {"budget": np.asarray([r[0] for r in rows]),
            "rk4_error": np.asarray([r[1] for r in rows]),
            "rk4_evaluations": np.asarray([r[2] for r in rows]),
            "ab4_error": np.asarray([r[3] for r in rows]),
            "ab4_evaluations": np.asarray([r[4] for r in rows]),
            "predictor_corrector_error": np.asarray([r[5] for r in rows]),
            "predictor_corrector_evaluations": np.asarray([r[6] for r in rows])}
