"""Runge-Kutta methods: high order from evaluations of ``f`` alone.

The idea
--------
A Taylor method of order ``k`` needs the solution's derivatives, which means the partial
derivatives of ``f``, which lesson 68 counted and found unaffordable. Runge-Kutta reaches the same
order using only **more evaluations of ``f``**, at points chosen so that the resulting combination
matches the Taylor expansion to the required order.

An ``s`` stage explicit method is

    k_i = f(t + c_i h, y + h sum_j a_ij k_j),      j < i
    y_new = y + h sum_i b_i k_i

and the whole method is the array ``(A, b, c)``, its **Butcher tableau**.

Order conditions
----------------
Expanding both sides in ``h`` and matching gives the order conditions. The first few are

    order 1:   sum b_i = 1
    order 2:   sum b_i c_i = 1/2
    order 3:   sum b_i c_i^2 = 1/3,   sum_i b_i sum_j a_ij c_j = 1/6
    order 4:   four more

The count is the rooted tree count of lesson 68, so it grows the same way; what has changed is
that each condition is a statement about **numbers in a table** rather than about derivatives of
``f``. `order_conditions` checks them and `verified_order` measures the order the tableau
actually achieves, which is how a mistyped coefficient is caught.

The barrier
-----------
An explicit ``s`` stage method has order at most ``s``, with equality only up to ``s = 4``.
Order 5 needs 6 stages, order 6 needs 7, order 7 needs 9 and order 8 needs 11. That is why RK4
is the method everyone knows: **it is the last one where a stage buys a whole order.**

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .ivp import as_state


#: Classical explicit tableaux, as ``name -> (A, b, c, order)`` with exact rational entries.
TABLEAUX = {
    "euler": ([[0]], [1], [0], 1),
    "midpoint": ([[0, 0], [Fraction(1, 2), 0]],
                 [0, 1], [0, Fraction(1, 2)], 2),
    "heun": ([[0, 0], [1, 0]],
             [Fraction(1, 2), Fraction(1, 2)], [0, 1], 2),
    "ralston": ([[0, 0], [Fraction(2, 3), 0]],
                [Fraction(1, 4), Fraction(3, 4)], [0, Fraction(2, 3)], 2),
    "kutta third": ([[0, 0, 0], [Fraction(1, 2), 0, 0], [-1, 2, 0]],
                    [Fraction(1, 6), Fraction(2, 3), Fraction(1, 6)],
                    [0, Fraction(1, 2), 1], 3),
    "heun third": ([[0, 0, 0], [Fraction(1, 3), 0, 0], [0, Fraction(2, 3), 0]],
                   [Fraction(1, 4), 0, Fraction(3, 4)],
                   [0, Fraction(1, 3), Fraction(2, 3)], 3),
    "rk4": ([[0, 0, 0, 0], [Fraction(1, 2), 0, 0, 0],
             [0, Fraction(1, 2), 0, 0], [0, 0, 1, 0]],
            [Fraction(1, 6), Fraction(1, 3), Fraction(1, 3), Fraction(1, 6)],
            [0, Fraction(1, 2), Fraction(1, 2), 1], 4),
    "three eighths": ([[0, 0, 0, 0], [Fraction(1, 3), 0, 0, 0],
                       [Fraction(-1, 3), 1, 0, 0], [1, -1, 1, 0]],
                      [Fraction(1, 8), Fraction(3, 8), Fraction(3, 8), Fraction(1, 8)],
                      [0, Fraction(1, 3), Fraction(2, 3), 1], 4),
}


def tableau(name: str) -> tuple:
    """The named tableau as float arrays ``(A, b, c)`` plus its stated order."""
    key = str(name).lower()
    if key not in TABLEAUX:
        raise ValueError(f"unknown tableau {name!r}, expected one of {sorted(TABLEAUX)}")
    A, b, c, order = TABLEAUX[key]
    return (np.asarray([[float(v) for v in row] for row in A]),
            np.asarray([float(v) for v in b]),
            np.asarray([float(v) for v in c]),
            int(order))


def tableau_exact(name: str) -> tuple:
    """The named tableau with its coefficients left as exact ``Fraction`` objects.

    `tableau` converts to float, which is what a stepper wants. This does not, which is what
    `order_conditions` wants: an order condition is an exact identity between rationals, and
    checking it in floating point can only ever report "within a tolerance I chose".
    """
    key = str(name).lower()
    if key not in TABLEAUX:
        raise ValueError(f"unknown tableau {name!r}, expected one of {sorted(TABLEAUX)}")
    A, b, c, order = TABLEAUX[key]
    return ([[Fraction(v) for v in row] for row in A],
            [Fraction(v) for v in b], [Fraction(v) for v in c], int(order))


def _is_exact(values) -> bool:
    """Whether every entry is an integer or a Fraction, so the arithmetic can stay exact."""
    for item in values:
        if isinstance(item, (list, tuple)):
            if not _is_exact(item):
                return False
        elif isinstance(item, np.ndarray):
            return False
        elif not isinstance(item, (int, Fraction)):
            return False
    return True


def is_explicit(A) -> bool:
    """An explicit tableau is strictly lower triangular, so each stage uses only earlier ones."""
    M = np.asarray(A, dtype=float)
    return bool(np.all(np.triu(M) == 0.0))


def step_from(A, b, c):
    """Build a one step function from a tableau, in the shape `nalib.ivp.integrate` wants."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    if not is_explicit(A):
        raise ValueError("only explicit tableaux are supported here; "
                         "an implicit one needs a nonlinear solve at every step")
    s = b.size

    def step(f, t, y, h, _counter=None):
        state = as_state(y)
        stages = np.empty((s, state.size))
        for i in range(s):
            arg = state.copy()
            for j in range(i):
                if A[i, j] != 0.0:
                    arg = arg + h * A[i, j] * stages[j]
            if _counter is not None:
                _counter[0] += 1
            stages[i] = as_state(f(t + c[i] * h, arg))
        return state + h * (b @ stages)

    return step


def named_step(name: str):
    """The step function for a named classical method."""
    A, b, c, _order = tableau(name)
    return step_from(A, b, c)


# --------------------------------------------------------------------------- order conditions


def order_conditions(A, b, c, up_to: int = 4, tol: float = 1e-12) -> dict:
    """Check the order conditions a tableau satisfies, up to the requested order.

    The conditions through order 4, written with ``sum`` over all indices:

        1:  sum b_i                      = 1
        2:  sum b_i c_i                  = 1/2
        3:  sum b_i c_i^2                = 1/3
            sum b_i a_ij c_j             = 1/6
        4:  sum b_i c_i^3                = 1/4
            sum b_i c_i a_ij c_j         = 1/8
            sum b_i a_ij c_j^2           = 1/12
            sum b_i a_ij a_jk c_k        = 1/24
        5:  nine more, one per rooted tree on five nodes, listed in the code below

    Reporting which ones hold rather than only the achieved order is what localises a typing
    error: a wrong ``b`` breaks the first group, a wrong ``c`` the second.

    **Given exact coefficients the arithmetic stays exact.** Pass the output of `tableau_exact`
    and every sum is a `Fraction`, every comparison is ``==``, and the reported residual is an
    exact zero or an exact nonzero. Pass floats and the same conditions are checked to ``tol``,
    which is all floating point can offer: RK4's first condition sums to ``0.9999999999999999``,
    and no tolerance-free statement about that number is available.

    ``exact`` in the result says which of the two happened.

    The counts are 1, 1, 2, 4, 9 conditions at orders 1 to 5, which are lesson 68's rooted tree
    counts. RK4 satisfies the first eight exactly and misses the order 5 group: its worst order 5
    residual is ``1/80``, which is not small and not a rounding artefact. Dormand-Prince and
    Fehlberg satisfy all seventeen to ``10^-16``.
    """
    exact = _is_exact(A) and _is_exact(b) and _is_exact(c)
    if exact:
        M = [[Fraction(v) for v in row] for row in A]
        bb = [Fraction(v) for v in b]
        cc = [Fraction(v) for v in c]
        n = len(bb)
        one = Fraction(1)

        def apply(vector):
            return [sum((M[i][j] * vector[j] for j in range(n)), Fraction(0))
                    for i in range(n)]

        def weight(vector):
            return sum((bb[i] * vector[i] for i in range(n)), Fraction(0))

        ones = [one] * n
        Ac = apply(cc)
        Ac2 = apply([v * v for v in cc])
        AAc = apply(Ac)
        checks = [
            (1, "sum b", weight(ones), Fraction(1)),
            (2, "sum b c", weight(cc), Fraction(1, 2)),
            (3, "sum b c^2", weight([v * v for v in cc]), Fraction(1, 3)),
            (3, "sum b A c", weight(Ac), Fraction(1, 6)),
            (4, "sum b c^3", weight([v ** 3 for v in cc]), Fraction(1, 4)),
            (4, "sum b c (A c)", weight([cc[i] * Ac[i] for i in range(n)]), Fraction(1, 8)),
            (4, "sum b A c^2", weight(Ac2), Fraction(1, 12)),
            (4, "sum b A A c", weight(AAc), Fraction(1, 24)),
            (5, "sum b c^4", weight([v ** 4 for v in cc]), Fraction(1, 5)),
            (5, "sum b c^2 (A c)", weight([cc[i] ** 2 * Ac[i] for i in range(n)]),
             Fraction(1, 10)),
            (5, "sum b (A c)^2", weight([Ac[i] ** 2 for i in range(n)]), Fraction(1, 20)),
            (5, "sum b c (A c^2)", weight([cc[i] * Ac2[i] for i in range(n)]),
             Fraction(1, 15)),
            (5, "sum b A c^3", weight(apply([v ** 3 for v in cc])), Fraction(1, 20)),
            (5, "sum b c (A A c)", weight([cc[i] * AAc[i] for i in range(n)]),
             Fraction(1, 30)),
            (5, "sum b A (c (A c))",
             weight(apply([cc[i] * Ac[i] for i in range(n)])), Fraction(1, 40)),
            (5, "sum b A A c^2", weight(apply(Ac2)), Fraction(1, 60)),
            (5, "sum b A A A c", weight(apply(AAc)), Fraction(1, 120)),
        ]
        rows = [row for row in checks if row[0] <= int(up_to)]
        holds = [got == want for _, _, got, want in rows]
        residual = [want - got for _, _, got, want in rows]
    else:
        M = np.asarray(A, dtype=float)
        bb = np.asarray(b, dtype=float)
        cc = np.asarray(c, dtype=float)
        Ac = M @ cc
        Ac2 = M @ (cc ** 2)
        AAc = M @ Ac
        checks = [
            (1, "sum b", float(np.sum(bb)), 1.0),
            (2, "sum b c", float(bb @ cc), 0.5),
            (3, "sum b c^2", float(bb @ (cc ** 2)), 1.0 / 3.0),
            (3, "sum b A c", float(bb @ Ac), 1.0 / 6.0),
            (4, "sum b c^3", float(bb @ (cc ** 3)), 0.25),
            (4, "sum b c (A c)", float(bb @ (cc * Ac)), 0.125),
            (4, "sum b A c^2", float(bb @ Ac2), 1.0 / 12.0),
            (4, "sum b A A c", float(bb @ AAc), 1.0 / 24.0),
            (5, "sum b c^4", float(bb @ cc ** 4), 1.0 / 5.0),
            (5, "sum b c^2 (A c)", float(bb @ (cc ** 2 * Ac)), 1.0 / 10.0),
            (5, "sum b (A c)^2", float(bb @ (Ac ** 2)), 1.0 / 20.0),
            (5, "sum b c (A c^2)", float(bb @ (cc * Ac2)), 1.0 / 15.0),
            (5, "sum b A c^3", float(bb @ (M @ cc ** 3)), 1.0 / 20.0),
            (5, "sum b c (A A c)", float(bb @ (cc * AAc)), 1.0 / 30.0),
            (5, "sum b A (c (A c))", float(bb @ (M @ (cc * Ac))), 1.0 / 40.0),
            (5, "sum b A A c^2", float(bb @ (M @ Ac2)), 1.0 / 60.0),
            (5, "sum b A A A c", float(bb @ (M @ AAc)), 1.0 / 120.0),
        ]
        rows = [row for row in checks if row[0] <= int(up_to)]
        holds = [abs(got - want) <= tol * max(abs(want), 1.0) for _, _, got, want in rows]
        residual = [want - got for _, _, got, want in rows]
    achieved = 0
    for level in range(1, int(up_to) + 1):
        at_level = [h for (o, _, _, _), h in zip(rows, holds) if o == level]
        if at_level and all(at_level):
            achieved = level
        else:
            break
    return {"order": np.asarray([r[0] for r in rows]),
            "condition": [r[1] for r in rows],
            "computed": [r[2] for r in rows],
            "required": [r[3] for r in rows],
            "residual": residual,
            "holds": np.asarray(holds),
            "exact": bool(exact),
            "achieved_order": achieved}


def all_tableaux_are_what_they_claim(tol: float = 1e-12) -> dict:
    """Every tableau in `TABLEAUX`, checked against the order it is labelled with.

    Checked twice: once in exact rational arithmetic, where the answer is a yes or a no, and once
    in floating point, where the answer is "to within ``tol``". They agree here, and the point of
    running both is that the float column cannot say so on its own.
    """
    rows = []
    for name in TABLEAUX:
        A, b, c, order = tableau_exact(name)
        exact_out = order_conditions(A, b, c, up_to=min(order + 1, 4), tol=tol)
        fA, fb, fc, _ = tableau(name)
        float_out = order_conditions(fA, fb, fc, up_to=min(order + 1, 4), tol=tol)
        worst = max((abs(float(v)) for v in float_out["residual"]), default=0.0)
        rows.append((name, order, exact_out["achieved_order"], float_out["achieved_order"],
                     int(fb.size), is_explicit(fA), worst, exact_out["exact"]))
    return {"names": [r[0] for r in rows],
            "claimed_order": np.asarray([r[1] for r in rows]),
            "conditions_satisfied_to": np.asarray([r[2] for r in rows]),
            "float_conditions_satisfied_to": np.asarray([r[3] for r in rows]),
            "worst_float_residual": np.asarray([r[6] for r in rows]),
            "checked_exactly": np.asarray([r[7] for r in rows]),
            "stages": np.asarray([r[4] for r in rows]),
            "explicit": np.asarray([r[5] for r in rows]),
            "exact_and_float_agree": bool(all(r[2] == r[3] for r in rows)),
            "all_agree": bool(all(r[1] == r[2] or r[1] > 4 for r in rows))}


def verified_order(f, exact, t0: float, y0, t_end: float, name: str,
                   step_counts=None) -> dict:
    """The order a tableau actually achieves on a problem, measured by refinement.

    The order conditions are a statement about the tableau; this is a statement about the
    method's behaviour. They should agree, and when they do not it is nearly always because the
    problem is not smooth enough for the higher conditions to be visible.
    """
    from .ivp import error_against_step

    out = error_against_step(f, exact, t0, y0, t_end, step_counts, named_step(name))
    A, b, c, claimed = tableau(name)
    return dict(out, name=name, claimed_order=claimed, stages=int(b.size),
                matches=bool(abs(out["fitted_order"] - claimed) < 0.3))


# --------------------------------------------------------------------------- comparisons


def order_against_stages(names=None) -> dict:
    """Order per stage for the classical explicit methods, which is where RK4 comes from.

    An explicit ``s`` stage method has order at most ``s``. Equality holds for ``s = 1, 2, 3, 4``
    and then stops: order 5 needs 6 stages, order 6 needs 7, order 7 needs 9, order 8 needs 11.
    **RK4 is the last method where a stage buys a whole order**, and that is the entire reason it
    is the one everybody knows.
    """
    wanted = (list(TABLEAUX) if names is None else [str(v) for v in np.atleast_1d(names)])
    rows = []
    for name in wanted:
        A, b, c, order = tableau(name)
        rows.append((name, int(b.size), order, order / b.size))
    # the minimum stage count needed for each order, from the Butcher barriers
    barrier = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 9, 8: 11}
    return {"names": [r[0] for r in rows],
            "stages": np.asarray([r[1] for r in rows]),
            "order": np.asarray([r[2] for r in rows]),
            "order_per_stage": np.asarray([r[3] for r in rows]),
            "barrier_order": np.asarray(sorted(barrier)),
            "barrier_stages": np.asarray([barrier[k] for k in sorted(barrier)]),
            "efficient_up_to": max(k for k in barrier if barrier[k] == k)}


def compare_at_equal_cost(f, exact, t0: float, y0, t_end: float, budgets=None,
                          names=None) -> dict:
    """Classical methods at the same number of evaluations of ``f``.

    A four stage method takes a quarter as many steps as Euler for the same budget, which is the
    only comparison that means anything. The higher order still wins, and the margin grows with
    the budget, because ``h^4`` beats ``h`` at every split.
    """
    from .ivp import integrate

    bs = ([48, 96, 192, 384, 768] if budgets is None
          else [int(v) for v in np.atleast_1d(budgets)])
    wanted = (["euler", "heun", "kutta third", "rk4"] if names is None
              else [str(v) for v in np.atleast_1d(names)])
    want = as_state(exact(float(t_end)))
    table = {}
    for name in wanted:
        A, b, c, _order = tableau(name)
        stages = int(b.size)
        errors = []
        for budget in bs:
            steps = max(1, budget // stages)
            counter = [0]
            out = integrate(f, t0, y0, t_end, steps, named_step(name), counter)
            errors.append(float(np.max(np.abs(out["y"][-1] - want))))
        table[name] = np.asarray(errors)
    best = [min(wanted, key=lambda n: table[n][i]) for i in range(len(bs))]
    return {"evaluations": np.asarray(bs), "names": wanted, "errors": table,
            "best_at_each_budget": best,
            "highest_order_always_wins": bool(all(b == wanted[-1] for b in best))}


def stability_limit(name: str, tol: float = 1e-9) -> float:
    """The largest ``|z|`` on the negative real axis for which the method is stable.

    Applying an explicit Runge-Kutta method to ``y' = lam y`` multiplies the state by a
    polynomial ``R(z)`` in ``z = lam h``, and the method is stable exactly when ``|R(z)| <= 1``.
    On the negative real axis that is an interval ``[-Z, 0]``, and ``Z`` is found by bisection.

    The classical values are 2 for Euler and every second order method, 2.5127 for the third
    order ones and 2.7853 for RK4. **The gain from Euler to RK4 is a factor of 1.4**, which is
    the whole point: raising the order buys accuracy and buys almost nothing in stability, so an
    explicit method of any order is useless on a stiff problem. Lesson 72 takes that up.
    """
    A, b, c, _order = tableau(name)
    s_stages = b.size

    def growth(z):
        """R(z) for this tableau, by running one step of the method on y' = z y with y = 1."""
        stages = np.zeros(s_stages, dtype=complex)
        for i in range(s_stages):
            arg = 1.0 + sum(A[i, j] * stages[j] for j in range(i))
            stages[i] = z * arg
        return 1.0 + float(np.real(b @ np.real(stages))) if np.all(np.isreal(stages)) \
            else 1.0 + complex(b @ stages)

    lo, hi = 0.0, 16.0
    if abs(growth(-lo)) > 1.0:
        return 0.0
    while abs(growth(-hi)) <= 1.0 and hi < 1e6:
        hi *= 2.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if abs(growth(-mid)) <= 1.0:
            lo = mid
        else:
            hi = mid
    return float(lo)


def explicit_stability_is_not_the_answer(names=None, lam: float = -1000.0,
                                         t_end: float = 1.0) -> dict:
    """A preview of lesson 72: high order does not rescue an explicit method from stiffness.

    On ``y' = lam y`` with ``lam`` very negative, every explicit method has a step size limit set
    by its **stability** region rather than by its accuracy. The limits differ by a factor of
    1.4 across the whole classical family, so the steps needed differ by the same factor, and all
    of them are enormous when ``lam`` is large.

    The comparison to make is against the accuracy the step would otherwise allow. At
    ``lam = -1000`` the solution is at ``10^-435`` by ``t = 1`` and any method could represent
    that with a handful of steps; stability forces hundreds regardless.

    Because the limits are so close and the stage counts are not, the ranking **reverses**. At
    ``lam = -1000`` over ``t = 1``:

        method          euler     heun   kutta 3      rk4
        limit           2.000    2.000     2.513    2.785
        steps             500      500       398      360
        evaluations       500     1000      1194     1440

    RK4's stable step is 39 per cent larger than Euler's and it costs four times as much per step,
    so **Euler is 2.9 times cheaper**, at an accuracy nobody asked for. ``cheapest`` reports which
    method won and ``cheapest_is_the_lowest_order`` whether it was the lowest order one.
    """
    wanted = (["euler", "heun", "kutta third", "rk4"] if names is None
              else [str(v) for v in np.atleast_1d(names)])
    span = float(t_end)
    rows = []
    for name in wanted:
        A, b, c, order = tableau(name)
        z = stability_limit(name)
        h_max = z / abs(lam)
        steps = int(math.ceil(span / h_max))
        rows.append((name, int(b.size), order, z, h_max, steps, steps * int(b.size)))
    return {"names": [r[0] for r in rows],
            "stages": np.asarray([r[1] for r in rows]),
            "order": np.asarray([r[2] for r in rows]),
            "stability_limit": np.asarray([r[3] for r in rows]),
            "largest_stable_step": np.asarray([r[4] for r in rows]),
            "steps_needed": np.asarray([r[5] for r in rows]),
            "evaluations_needed": np.asarray([r[6] for r in rows]),
            "lam": float(lam),
            "spread": float(np.max([r[3] for r in rows]) / np.min([r[3] for r in rows])),
            "evaluation_spread": float(np.max([r[6] for r in rows])
                                       / np.min([r[6] for r in rows])),
            "cheapest": rows[int(np.argmin([r[6] for r in rows]))][0],
            "cheapest_is_the_lowest_order": bool(
                rows[int(np.argmin([r[6] for r in rows]))][2]
                == min(r[2] for r in rows))}


def stability_limit_is_where_the_solver_fails(name: str = "rk4", lam: float = -50.0,
                                              t_end: float = 1.0) -> dict:
    """The predicted limit against the step at which the computed solution actually diverges.

    Bisecting the growth polynomial predicts a number; running the method finds where it stops
    producing a decaying answer. They should agree, and confirming that is what makes the
    stability region a statement about the solver rather than about a polynomial.
    """
    from .ivp import integrate

    predicted = stability_limit(name)
    f = lambda t, y: lam * as_state(y)
    step = named_step(name)
    lo_steps = max(1, int(math.floor(abs(lam) * float(t_end) / predicted)) - 4)
    rows = []
    for n in range(lo_steps, lo_steps + 10):
        out = integrate(f, 0.0, 1.0, t_end, n, step)
        value = float(out["y"][-1, 0])
        rows.append((n, abs(lam) * float(out["step"]), value,
                     bool(np.isfinite(value) and abs(value) <= 1.0)))
    stable = [r[1] for r in rows if r[3]]
    return {"name": name, "predicted_limit": predicted,
            "steps": np.asarray([r[0] for r in rows]),
            "lam_h": np.asarray([r[1] for r in rows]),
            "final_value": np.asarray([r[2] for r in rows]),
            "stable": np.asarray([r[3] for r in rows]),
            "largest_observed_stable_lam_h": (max(stable) if stable else float("nan")),
            "agrees": bool(stable and max(stable) <= predicted + 1e-9)}
