"""Richardson extrapolation on the trapezoid rule: Euler-Maclaurin, Romberg, and the limits.

Why the trapezoid rule of all things
------------------------------------
The trapezoid rule is the worst rule in lesson 62. It is also the only one worth extrapolating,
and the reason is the Euler-Maclaurin formula:

    T(h) - I  =  B_2/2! h^2 [f'(b) - f'(a)]  +  B_4/4! h^4 [f'''(b) - f'''(a)]  +  ...

Every term is an **even** power of ``h``, and every coefficient depends only on derivatives at
the two endpoints. Two facts follow, and they are the whole lesson.

**First**, an error series in even powers can be killed two orders at a time. One Richardson
step on ``h`` and ``h/2`` removes the ``h^2`` term and leaves ``h^4``, not ``h^3``. That is the
Romberg table, and its ``j``th column has order ``h^(2j+2)``. `column_orders` measures it.

**Second**, if ``f`` and all its derivatives are periodic with period ``b - a``, then every
bracket ``f^(2k-1)(b) - f^(2k-1)(a)`` is exactly zero and the entire series vanishes. The plain
trapezoid rule is then more accurate than any fixed order rule can be. This is not a small
effect: on a smooth periodic integrand it reaches machine precision at a few dozen points while
Romberg is still working through its table. `periodic_trapezoid` measures it.

Where it stops working
----------------------
Euler-Maclaurin needs ``f`` to have the derivatives it names. An endpoint singularity has none,
the series does not exist, and the columns of the table then converge no faster than the first
one. `on_a_singularity` measures the collapse, and `modified` shows what to do instead when the
true expansion is known to run in half powers.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

# --------------------------------------------------------------------------- the trapezoid ladder


def trapezoid_sequence(f, lo: float, hi: float, levels: int = 10, _counter=None) -> np.ndarray:
    """Composite trapezoid on ``1, 2, 4, ... 2^levels`` panels, reusing every evaluation.

    Halving the panel width doubles the node count, and half of the new nodes are the old ones.
    Recomputing them is the obvious waste, so each level adds only the midpoints:

        T(h/2) = T(h)/2 + (h/2) * sum of f at the new midpoints

    The whole ladder to ``2^k`` panels then costs ``2^k + 1`` evaluations rather than the
    ``2^(k+1) + k`` a naive loop would spend, so the saving tends to a factor of exactly two and
    no more. That is worth having and is not the reason Romberg is fast: the reason is that the
    extrapolation turns those same values into a much better answer. `evaluations_saved` counts
    both sides.
    """
    a = float(lo)
    b = float(hi)
    k = int(levels)
    if k < 0:
        raise ValueError(f"need a non-negative number of levels, got {k}")
    if not b > a:
        raise ValueError(f"need lo < hi, got [{a}, {b}]")
    out = np.empty(k + 1)
    h = b - a
    ends = np.asarray([a, b], dtype=float)
    values = np.asarray(f(ends), dtype=float)
    if _counter is not None:
        _counter[0] += 2
    out[0] = 0.5 * h * float(values[0] + values[1])
    for level in range(1, k + 1):
        h *= 0.5
        midpoints = a + h * (2.0 * np.arange(2 ** (level - 1)) + 1.0)
        if _counter is not None:
            _counter[0] += midpoints.size
        out[level] = 0.5 * out[level - 1] + h * float(np.sum(
            np.asarray(f(midpoints), dtype=float)))
    return out


def evaluations_saved(levels: int = 10) -> dict:
    """What the reuse actually buys, counted rather than asserted."""
    k = int(levels)
    counter = [0]
    trapezoid_sequence(lambda x: np.zeros_like(np.asarray(x, dtype=float)), 0.0, 1.0, k, counter)
    naive = sum(2 ** j + 1 for j in range(k + 1))
    return {"levels": k, "with_reuse": counter[0], "without_reuse": naive,
            "finest_panels": 2 ** k,
            "ratio": float(naive) / max(counter[0], 1)}


# --------------------------------------------------------------------------- the table


def table(f, lo: float, hi: float, levels: int = 8, _counter=None) -> np.ndarray:
    """The full Romberg table, lower triangular, ``R[k][j]``.

    Column 0 is the trapezoid ladder. Each further column is one Richardson step:

        R[k][j] = R[k][j-1] + (R[k][j-1] - R[k-1][j-1]) / (4^j - 1)

    The ``4^j`` and not ``2^j`` is Euler-Maclaurin doing its work: the series runs in even
    powers, so one step gains two orders and the ratio between successive errors is ``4^j``.
    """
    k = int(levels)
    first = trapezoid_sequence(f, lo, hi, k, _counter)
    R = np.zeros((k + 1, k + 1))
    R[:, 0] = first
    for j in range(1, k + 1):
        for i in range(j, k + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4.0 ** j - 1.0)
    return R


def integrate(f, lo: float, hi: float, levels: int = 8, _counter=None) -> float:
    """The bottom right corner of the table, which is the best estimate it holds."""
    R = table(f, lo, hi, levels, _counter)
    return float(R[-1, -1])


def to_tolerance(f, lo: float, hi: float, tol: float = 1e-10, max_levels: int = 20,
                 _counter=None) -> dict:
    """Extend the table until two successive diagonal entries agree, or the levels run out.

    The stopping test compares ``R[k][k]`` against ``R[k-1][k-1]``, which is the usual choice
    and is an estimate of the error, not a bound on it. `converged` reports whether the test
    fired, not whether the answer is right.

    On an integrand with no endpoint derivatives the test simply never fires. Asking for 1e-10
    on ``sqrt(x)`` over [0, 1] runs all 20 levels, spends 1048577 function evaluations, and
    returns ``converged=False`` with an answer that happens to be good to 6e-11 anyway. The
    same call on ``sin`` needs 4 levels and 17 evaluations. **Always look at the level count**:
    a Romberg routine that quietly runs to its limit is telling you the expansion it is built
    on does not exist for your integrand.
    """
    a = float(lo)
    b = float(hi)
    limit = int(max_levels)
    want = float(tol)
    rows = [np.asarray([0.0])]
    h = b - a
    ends = np.asarray([a, b], dtype=float)
    values = np.asarray(f(ends), dtype=float)
    if _counter is not None:
        _counter[0] += 2
    rows[0] = np.asarray([0.5 * h * float(values[0] + values[1])])
    previous = float(rows[0][0])
    for level in range(1, limit + 1):
        h *= 0.5
        midpoints = a + h * (2.0 * np.arange(2 ** (level - 1)) + 1.0)
        if _counter is not None:
            _counter[0] += midpoints.size
        trap = 0.5 * float(rows[-1][0]) + h * float(np.sum(
            np.asarray(f(midpoints), dtype=float)))
        row = np.empty(level + 1)
        row[0] = trap
        for j in range(1, level + 1):
            row[j] = row[j - 1] + (row[j - 1] - rows[-1][j - 1]) / (4.0 ** j - 1.0)
        rows.append(row)
        best = float(row[-1])
        gap = abs(best - previous)
        if gap <= want * max(abs(best), 1.0):
            return {"value": best, "levels_used": level, "estimated_error": gap,
                    "converged": True,
                    "evaluations": (_counter[0] if _counter is not None else 2 ** level + 1)}
        previous = best
    return {"value": previous, "levels_used": limit,
            "estimated_error": float("nan"), "converged": False,
            "evaluations": (_counter[0] if _counter is not None else 2 ** limit + 1)}


# --------------------------------------------------------------------------- what the columns are


def column_is_a_newton_cotes_rule(f, lo: float, hi: float, levels: int = 6) -> dict:
    """Columns 1 and 2 of the table are composite Simpson and composite Boole, exactly.

    Romberg is not a new family of rules. One extrapolation step on the trapezoid ladder
    reproduces composite Simpson to the last bit, and two steps reproduce composite Boole. The
    table is a cheap way of writing down rules that already exist, and the reason to prefer it
    is that it produces the whole sequence for the price of the finest one.
    """
    from . import newtoncotes as _nc

    R = table(f, lo, hi, levels)
    simpson_gap = []
    boole_gap = []
    for k in range(1, levels + 1):
        panels = 2 ** (k - 1)
        s = _nc.composite_shared(f, lo, hi, panels, 3)
        simpson_gap.append(abs(float(R[k, 1]) - s) / max(abs(s), 1.0))
    for k in range(2, levels + 1):
        panels = 2 ** (k - 2)
        bo = _nc.composite_shared(f, lo, hi, panels, 5)
        boole_gap.append(abs(float(R[k, 2]) - bo) / max(abs(bo), 1.0))
    return {"simpson_relative_gap": np.asarray(simpson_gap),
            "boole_relative_gap": np.asarray(boole_gap),
            "column_1_is_simpson": bool(np.max(simpson_gap) < 1e-12),
            "column_2_is_boole": bool(np.max(boole_gap) < 1e-12)}


def column_orders(f, exact, lo: float, hi: float, levels: int = 10,
                  floor: float = 1e-13) -> dict:
    """The observed order of each column, which should be ``2j + 2``.

    Measured as the slope of the error against the panel count down each column, using only the
    rows above the roundoff floor. Columns run out of usable rows quickly: by column 4 the error
    is at machine precision after three refinements, so there is nothing left to fit. The
    ``rows_used`` entry says how many each order rests on, and an order fitted on fewer than
    three rows is reported as nan rather than guessed.
    """
    want = float(exact)
    R = table(f, lo, hi, levels)
    k = int(levels)
    orders = []
    used = []
    predicted = []
    for j in range(k + 1):
        rows = np.arange(j, k + 1)
        if rows.size < 3:
            orders.append(float("nan"))
            used.append(0)
            predicted.append(2 * j + 2)
            continue
        errors = np.abs(R[rows, j] - want)
        panels = 2.0 ** rows
        keep = errors > floor * max(abs(want), 1.0)
        if int(np.sum(keep)) < 3:
            orders.append(float("nan"))
            used.append(int(np.sum(keep)))
        else:
            orders.append(float(-np.polyfit(np.log(panels[keep]),
                                            np.log(errors[keep]), 1)[0]))
            used.append(int(np.sum(keep)))
        predicted.append(2 * j + 2)
    return {"column": np.arange(k + 1), "fitted_order": np.asarray(orders),
            "predicted_order": np.asarray(predicted), "rows_used": np.asarray(used),
            "table": R}


# --------------------------------------------------------------------------- Euler-Maclaurin


def bernoulli(n: int) -> Fraction:
    """The ``n``th Bernoulli number, exactly, by the recurrence on binomial sums.

    Convention ``B_1 = -1/2``. Only the even ones past 1 are non-zero, and those are the
    coefficients Euler-Maclaurin needs.
    """
    m = int(n)
    if m < 0:
        raise ValueError(f"need a non-negative index, got {m}")
    B = [Fraction(0)] * (m + 1)
    for i in range(m + 1):
        B[i] = Fraction(1, i + 1)
        for j in range(i, 0, -1):
            B[j - 1] = Fraction(j) * (B[j - 1] - B[j])
    return B[0] if m != 1 else Fraction(-1, 2)


def euler_maclaurin_terms(derivatives, lo: float, hi: float, panels: int,
                          terms: int = 4) -> dict:
    """The first few terms of the Euler-Maclaurin error series, given endpoint derivatives.

    ``derivatives`` is any callable taking ``(x, k)`` and returning ``f^(k)(x)``. The series is

        T(h) - I  =  sum over k >= 1 of  B_2k / (2k)! * h^2k * [f^(2k-1)(b) - f^(2k-1)(a)]

    The point of computing it explicitly is to check it against a measured trapezoid error.
    When the two match to several digits, the even power structure is not being taken on faith.
    """
    a = float(lo)
    b = float(hi)
    m = int(panels)
    if m < 1:
        raise ValueError(f"need at least one panel, got {m}")
    h = (b - a) / m
    rows = []
    total = 0.0
    for k in range(1, int(terms) + 1):
        coefficient = float(bernoulli(2 * k)) / math.factorial(2 * k)
        bracket = float(derivatives(b, 2 * k - 1)) - float(derivatives(a, 2 * k - 1))
        term = coefficient * h ** (2 * k) * bracket
        total += term
        rows.append((2 * k, coefficient, bracket, term, total))
    return {"power": np.asarray([r[0] for r in rows]),
            "coefficient": np.asarray([r[1] for r in rows]),
            "endpoint_bracket": np.asarray([r[2] for r in rows]),
            "term": np.asarray([r[3] for r in rows]),
            "running_total": np.asarray([r[4] for r in rows]),
            "step": h}


def euler_maclaurin_predicts_the_error(f, derivatives, exact, lo: float, hi: float,
                                       panel_counts=None, terms: int = 4) -> dict:
    """The measured trapezoid error against the series, panel count by panel count.

    This is the check that makes the rest of the lesson honest. If the series is right, the
    ratio of predicted to measured error goes to 1 as the panels are refined, and it does so
    from a direction the leading term alone cannot explain.

    **The series is asymptotic, not convergent.** A fixed number of terms is accurate for small
    enough ``h`` and not for a fixed ``h``, and adding more terms at a fixed ``h`` eventually
    makes the sum worse, because the Bernoulli numbers grow faster than the factorials shrink.
    So the thing to watch is `relative_gap` falling as the panels are refined, not its size at
    any one panel count. On [-2.5, -0.25] with four terms it runs 6.4e-07, 2.5e-09, 9.8e-12,
    9.0e-14 and then flattens where roundoff in the measured error takes over.
    """
    from . import newtoncotes as _nc

    ms = ([2, 4, 8, 16, 32, 64] if panel_counts is None
          else [int(v) for v in np.atleast_1d(panel_counts)])
    want = float(exact)
    rows = []
    for m in ms:
        trap = _nc.composite_shared(f, lo, hi, m, 2)
        measured = trap - want
        series = euler_maclaurin_terms(derivatives, lo, hi, m, terms)
        leading = float(series["term"][0])
        full = float(series["running_total"][-1])
        rows.append((m, measured, leading, full,
                     abs(full - measured) / max(abs(measured), 1e-300)))
    return {"panels": np.asarray([r[0] for r in rows]),
            "measured_error": np.asarray([r[1] for r in rows]),
            "leading_term": np.asarray([r[2] for r in rows]),
            "series_sum": np.asarray([r[3] for r in rows]),
            "relative_gap": np.asarray([r[4] for r in rows])}


# --------------------------------------------------------------------------- periodic


def periodic_trapezoid(f, exact, period_lo: float, period_hi: float, counts=None) -> dict:
    """The trapezoid rule on a smooth periodic integrand, over one full period.

    Every Euler-Maclaurin bracket ``f^(2k-1)(b) - f^(2k-1)(a)`` vanishes, so the whole error
    series is zero and the rule converges faster than any power of ``h``. For an analytic
    periodic integrand the convergence is geometric.

    This is why the trapezoid rule, useless in lesson 62, is the right rule for a Fourier
    coefficient, for a contour integral, and for anything on a circle.
    """
    a = float(period_lo)
    b = float(period_hi)
    ns = ([2, 4, 8, 16, 32, 64] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    want = float(exact)
    rows = []
    for n in ns:
        # over a full period the two endpoints carry the same value, so the composite
        # trapezoid collapses to a plain average of n equally spaced samples
        x = a + (b - a) * np.arange(n) / n
        got = (b - a) / n * float(np.sum(np.asarray(f(x), dtype=float)))
        rows.append((n, got, abs(got - want)))
    return {"points": np.asarray([r[0] for r in rows]),
            "values": np.asarray([r[1] for r in rows]),
            "errors": np.asarray([r[2] for r in rows])}


def periodic_convergence_is_geometric(f, exact, period_lo: float, period_hi: float,
                                      counts=None, floor: float = 1e-14) -> dict:
    """Fit the periodic trapezoid error to both a power law and a geometric law.

    A power law fits a straight line to log(error) against log(n); a geometric law fits it
    against ``n`` itself. Whichever has the smaller residual is the one describing the data, and
    on a smooth periodic integrand it is the geometric one by a wide margin.
    """
    out = periodic_trapezoid(f, exact, period_lo, period_hi, counts)
    n = np.asarray(out["points"], dtype=float)
    e = np.asarray(out["errors"], dtype=float)
    keep = e > floor * max(abs(float(exact)), 1.0)
    if int(np.sum(keep)) < 3:
        return dict(out, power_residual=float("nan"), geometric_residual=float("nan"),
                    fitted_power=float("nan"), fitted_rate=float("nan"),
                    geometric_wins=False,
                    note="not enough points above the roundoff floor to tell them apart")
    y = np.log(e[keep])
    p_fit, p_res = np.polyfit(np.log(n[keep]), y, 1, full=True)[:2]
    g_fit, g_res = np.polyfit(n[keep], y, 1, full=True)[:2]
    pr = float(p_res[0]) if len(p_res) else 0.0
    gr = float(g_res[0]) if len(g_res) else 0.0
    return dict(out, power_residual=pr, geometric_residual=gr,
                fitted_power=float(-p_fit[0]), fitted_rate=float(-g_fit[0]),
                geometric_wins=bool(gr < pr),
                points_used=int(np.sum(keep)),
                note="residual of log(error) against log(n) versus against n")


def periodic_against_romberg(f, exact, period_lo: float, period_hi: float,
                             levels: int = 7) -> dict:
    """The plain trapezoid rule against the whole Romberg table, at equal cost.

    On a periodic integrand the extrapolation has nothing to remove, so the table's extra
    columns buy nothing and can cost accuracy: they combine values that are each already at the
    floor. Compared at the same number of evaluations, plain trapezoid wins.
    """
    want = float(exact)
    k = int(levels)
    R = table(f, period_lo, period_hi, k)
    rows = []
    for level in range(k + 1):
        evaluations = 2 ** level + 1
        plain = periodic_trapezoid(f, want, period_lo, period_hi, [2 ** level])
        rows.append((evaluations, float(plain["errors"][0]),
                     abs(float(R[level, 0]) - want), abs(float(R[level, level]) - want)))
    return {"evaluations": np.asarray([r[0] for r in rows]),
            "periodic_trapezoid_error": np.asarray([r[1] for r in rows]),
            "romberg_first_column_error": np.asarray([r[2] for r in rows]),
            "romberg_diagonal_error": np.asarray([r[3] for r in rows])}


# --------------------------------------------------------------------------- the limits


def on_a_singularity(f, exact, lo: float, hi: float, levels: int = 10) -> dict:
    """Romberg on an integrand with no endpoint derivatives, which is most of what breaks it.

    Euler-Maclaurin is a statement about ``f^(2k-1)`` at the endpoints. When those do not exist,
    there is no even power series, the extrapolation is removing terms that are not there, and
    the columns stop gaining orders. The diagonal still converges, just not fast, and the
    stopping test in `to_tolerance` will happily fire on it.
    """
    want = float(exact)
    k = int(levels)
    R = table(f, lo, hi, k)
    diagonal = np.asarray([abs(float(R[i, i]) - want) for i in range(k + 1)])
    first = np.asarray([abs(float(R[i, 0]) - want) for i in range(k + 1)])
    rows = np.arange(k + 1, dtype=float)
    good = (diagonal > 0) & (first > 0)
    diagonal_order = (float(-np.polyfit(rows[good] * math.log(2.0),
                                        np.log(diagonal[good]), 1)[0])
                      if int(np.sum(good)) >= 3 else float("nan"))
    first_order = (float(-np.polyfit(rows[good] * math.log(2.0),
                                     np.log(first[good]), 1)[0])
                   if int(np.sum(good)) >= 3 else float("nan"))
    return {"levels": np.arange(k + 1),
            "first_column_error": first, "diagonal_error": diagonal,
            "first_column_order": first_order, "diagonal_order": diagonal_order,
            "extrapolation_gained": diagonal_order - first_order,
            "table": R}


def modified(f, lo: float, hi: float, levels: int = 8, power: float = 0.5) -> np.ndarray:
    """Romberg with the extrapolation ratio changed to match a known half power expansion.

    When the error runs in powers of ``h^p`` rather than ``h^2``, the right Richardson factor is
    ``2^(p*j)`` instead of ``4^j``. Column ``j`` then removes the ``h^(p*j)`` term, so the table
    clears the whole lattice ``p, 2p, 3p, ...`` and the useful choice of ``p`` is the largest one
    whose multiples cover every power actually present.

    On ``sqrt(x)`` over [0, 1] the expansion holds powers 3/2, 2, 5/2, 3, ... so ``p = 1/2``
    covers all of them. Measured at 10 levels, the corner of the table comes out at

        assumed power    corner error
                  0.5        3.0e-13
                  1.0        1.2e-06
                  1.5        1.2e-08
                  2.0        2.1e-06
                  3.0        4.5e-06

    against 6.3e-06 for the plain trapezoid rule on the same 1024 panels. So the right power is
    worth seven orders of magnitude, and a wrong power is not a disaster here, just a near
    total loss of the gain. The catch is that choosing ``p`` means knowing how the integrand is
    singular before integrating it.
    """
    k = int(levels)
    p = float(power)
    R = np.zeros((k + 1, k + 1))
    R[:, 0] = trapezoid_sequence(f, lo, hi, k)
    for j in range(1, k + 1):
        ratio = 2.0 ** (p * j)
        for i in range(j, k + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (ratio - 1.0)
    return R


def modified_helps_only_with_the_right_power(f, exact, lo: float, hi: float,
                                             powers=None, levels: int = 10) -> dict:
    """Sweep the assumed power and see which one actually helps.

    The best column of the modified table is reported for each assumed power. The minimum should
    sit at the integrand's true leading power, and the classical ``power = 2`` should be visibly
    worse when the integrand is singular.
    """
    ps = ([0.5, 1.0, 1.5, 2.0, 3.0] if powers is None
          else [float(v) for v in np.atleast_1d(powers)])
    want = float(exact)
    rows = []
    for p in ps:
        R = modified(f, lo, hi, levels, p)
        best = float(np.min(np.abs(R[levels, :levels + 1] - want)))
        rows.append((p, best, abs(float(R[levels, levels]) - want)))
    errors = np.asarray([r[1] for r in rows])
    return {"assumed_power": np.asarray([r[0] for r in rows]),
            "best_error_in_the_last_row": errors,
            "corner_error": np.asarray([r[2] for r in rows]),
            "best_power": float(ps[int(np.argmin(errors))])}
