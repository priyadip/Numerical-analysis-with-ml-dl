"""Newton-Cotes quadrature: integrate the interpolating polynomial instead of the function.

The construction
----------------
Sample ``f`` at equally spaced nodes, interpolate, and integrate the interpolant exactly. The
result is always a weighted sum

    integral_a^b f  ~  (b - a) sum_j w_j f(x_j)

with weights that depend only on the node count, not on ``f``. That is the whole family, and
trapezoid, Simpson, Boole and Weddle are its first few members.

Because the weights come from integrating the Lagrange basis, and the Lagrange basis reproduces
polynomials exactly, an ``n + 1`` point rule is exact on polynomials of degree ``n`` at least. For
a **symmetric** rule with an odd number of points it is exact on degree ``n + 1`` as well, because
the leftover odd term integrates to zero. `degree_of_precision` measures which, and the pattern
that emerges is the reason Simpson is worth having and the 3/8 rule is not.

Where it stops working
----------------------
Raising the node count does not keep helping, for the reason lesson 46 gives: the interpolating
polynomial through many equally spaced nodes diverges. The weights show it directly. From 9 points
onward some weights are **negative**, and by 21 points the largest weight is 4 times the interval
length, so the rule subtracts large numbers and loses digits even on a perfectly smooth integrand.
`weight_report` measures the growth.

**So the family is used at low order and composed**, which is what `composite` does: chop the
interval into panels, apply a small rule on each, and add. That converges at the small rule's own
order for every ``f`` with enough derivatives, and it never forms a high degree polynomial.

Open rules
----------
The closed rules use the endpoints. When the integrand is singular there, the **open** rules,
which use only interior nodes, still work. They are one order worse for the same node count and
they are what makes an improper integral computable, which is lesson 66's subject.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

# --------------------------------------------------------------------------- the weights


def weights(n_points: int, closed: bool = True) -> np.ndarray:
    """Newton-Cotes weights on ``n_points`` equally spaced nodes, scaled to a unit interval.

    Computed exactly, in rational arithmetic, by integrating the Lagrange basis. Doing it in
    floating point instead loses digits for the larger rules and makes the classical fractions
    unrecognisable, and the classical fractions are the whole point of the low order members.

    ``closed`` includes both endpoints. ``closed=False`` gives the open rule, whose nodes are the
    interior points of a grid with ``n_points + 2`` divisions.
    """
    n = int(n_points)
    if n < 1:
        raise ValueError(f"need at least one node, got {n}")
    if closed:
        if n < 2:
            raise ValueError(f"a closed rule needs at least two nodes, got {n}")
        nodes = [Fraction(j, n - 1) for j in range(n)]
    else:
        nodes = [Fraction(j + 1, n + 1) for j in range(n)]
    out = []
    for j in range(n):
        # integrate the jth Lagrange basis polynomial over [0, 1], exactly
        poly = [Fraction(1)]
        denominator = Fraction(1)
        for k in range(n):
            if k == j:
                continue
            poly = ([Fraction(0)] + poly) if True else poly
            poly = [c for c in poly]
            # multiply by (x - nodes[k]): shift up one degree and subtract nodes[k] * poly
            shifted = poly
            lowered = [Fraction(0)] * len(shifted)
            for i in range(len(shifted) - 1):
                lowered[i] = shifted[i + 1]
            poly = [shifted[i] - nodes[k] * lowered[i] for i in range(len(shifted))]
            denominator *= (nodes[j] - nodes[k])
        integral = sum(poly[i] / Fraction(i + 1) for i in range(len(poly)))
        out.append(integral / denominator)
    return np.asarray([float(v) for v in out])


def exact_weights(n_points: int, closed: bool = True) -> list:
    """The same weights as exact fractions, so the classical forms are recognisable.

    Trapezoid is ``[1/2, 1/2]``, Simpson is ``[1/6, 4/6, 1/6]``, and the 3/8 rule is
    ``[1/8, 3/8, 3/8, 1/8]``: the names come from these numbers.
    """
    n = int(n_points)
    if closed:
        if n < 2:
            raise ValueError(f"a closed rule needs at least two nodes, got {n}")
        nodes = [Fraction(j, n - 1) for j in range(n)]
    else:
        if n < 1:
            raise ValueError(f"an open rule needs at least one node, got {n}")
        nodes = [Fraction(j + 1, n + 1) for j in range(n)]
    out = []
    for j in range(n):
        poly = [Fraction(1)] + [Fraction(0)] * (n - 1)
        degree = 0
        denominator = Fraction(1)
        for k in range(n):
            if k == j:
                continue
            new = [Fraction(0)] * n
            for i in range(degree + 1):
                new[i + 1] += poly[i]
                new[i] -= nodes[k] * poly[i]
            poly = new
            degree += 1
            denominator *= (nodes[j] - nodes[k])
        integral = sum(poly[i] / Fraction(i + 1) for i in range(n))
        out.append(integral / denominator)
    return out


#: The named rules, as ``name -> (number of nodes, closed)``.
NAMED = {
    "trapezoid": (2, True),
    "Simpson 1/3": (3, True),
    "Simpson 3/8": (4, True),
    "Boole": (5, True),
    "six point": (6, True),
    "Weddle": (7, True),
    "midpoint": (1, False),
    "open two point": (2, False),
    "open three point": (3, False),
    "open four point": (4, False),
}


def rule_nodes(n_points: int, lo: float, hi: float, closed: bool = True) -> np.ndarray:
    """The nodes of a Newton-Cotes rule on ``[lo, hi]``."""
    n = int(n_points)
    a = float(lo)
    b = float(hi)
    if not b > a:
        raise ValueError(f"need lo < hi, got [{a}, {b}]")
    if closed:
        return np.linspace(a, b, n)
    return a + (b - a) * (np.arange(n) + 1.0) / (n + 1.0)


def integrate(f, lo: float, hi: float, n_points: int = 3, closed: bool = True,
              _counter=None) -> float:
    """One application of a Newton-Cotes rule over the whole interval."""
    a = float(lo)
    b = float(hi)
    x = rule_nodes(n_points, a, b, closed)
    w = weights(n_points, closed)
    if _counter is not None:
        _counter[0] += x.size
    return float((b - a) * np.sum(w * np.asarray(f(x), dtype=float)))


def composite(f, lo: float, hi: float, panels: int, n_points: int = 3,
              closed: bool = True, _counter=None) -> float:
    """Chop into panels and apply the rule on each.

    **This is how Newton-Cotes rules are actually used.** A single rule on many nodes is a high
    degree polynomial interpolant and diverges; many small rules never form one. The composite
    error is the single panel error summed, so a rule of order ``p`` on one panel gives a
    composite order ``p`` in the panel width.
    """
    a = float(lo)
    b = float(hi)
    m = int(panels)
    if m < 1:
        raise ValueError(f"need at least one panel, got {m}")
    edges = np.linspace(a, b, m + 1)
    total = 0.0
    for i in range(m):
        total += integrate(f, float(edges[i]), float(edges[i + 1]), n_points, closed, _counter)
    return float(total)


def composite_shared(f, lo: float, hi: float, panels: int, n_points: int = 3,
                     _counter=None) -> float:
    """The composite closed rule with the shared panel endpoints evaluated once.

    A closed rule's last node is the next panel's first node, so the naive loop evaluates every
    interior edge twice. Sharing them saves ``panels - 1`` evaluations, which for the trapezoid
    rule is nearly half the work. The answer is bit for bit the same.
    """
    n = int(n_points)
    m = int(panels)
    a = float(lo)
    b = float(hi)
    if n < 2:
        raise ValueError(f"a closed rule needs at least two nodes, got {n}")
    total_nodes = m * (n - 1) + 1
    x = np.linspace(a, b, total_nodes)
    values = np.asarray(f(x), dtype=float)
    if _counter is not None:
        _counter[0] += total_nodes
    w = weights(n, True)
    panel_width = (b - a) / m
    total = 0.0
    for i in range(m):
        block = values[i * (n - 1):i * (n - 1) + n]
        total += panel_width * float(np.sum(w * block))
    return float(total)


# --------------------------------------------------------------------------- properties


def degree_of_precision(n_points: int, closed: bool = True) -> int:
    """The highest degree the rule integrates exactly, found by testing monomials.

    The pattern is the interesting part and it is not what the node count alone suggests:

    ==============  ======  ==================  ====================
    rule            nodes   degree of precision  free order
    ==============  ======  ==================  ====================
    trapezoid       2       1                    none
    Simpson 1/3     3       **3**                one free
    Simpson 3/8     4       3                    none
    Boole           5       **5**                one free
    six point       6       5                    none
    Weddle          7       **7**                one free
    ==============  ======  ==================  ====================

    **Every odd node count gets one degree free**, because the rule is symmetric and the leftover
    odd term integrates to zero over a symmetric interval. That is why Simpson is standard and the
    3/8 rule is a curiosity: it costs an extra evaluation for the same precision.
    """
    n = int(n_points)
    # Computed in EXACT rational arithmetic, so there is no tolerance to choose and no roundoff
    # to mistake for exactness. Done in floating point, the 21 point rule reports degree 24 for
    # a 21 node rule, because its weights sum to 544 in absolute value and the monomials it
    # genuinely misses at k = 22 and 23 are missed by less than the rounding those weights
    # produce. That is a measurement of the arithmetic, not of the rule.
    if closed:
        nodes = [Fraction(j, n - 1) for j in range(n)]
    else:
        nodes = [Fraction(j + 1, n + 1) for j in range(n)]
    w = exact_weights(n, closed)
    for k in range(0, n + 4):
        got = sum(w[j] * nodes[j] ** k for j in range(n))
        want = Fraction(1, k + 1)
        if got != want:
            return k - 1
    return n + 3


def weight_report(counts=None, closed: bool = True) -> dict:
    """The weights' size and sign as the node count grows, which is where the family dies.

    Positive weights mean the rule is a weighted average and cannot amplify: the result is
    bounded by ``max|f|`` times the interval. Once a weight is negative that guarantee is gone,
    and once the weights are large the rule subtracts big numbers from big numbers.
    """
    ns = ([2, 3, 4, 5, 6, 7, 8, 9, 11, 15, 21] if counts is None
          else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        try:
            w = weights(n, closed)
        except ValueError:
            continue
        rows.append((n, bool(np.all(w >= 0.0)), float(np.max(np.abs(w))),
                     float(np.sum(np.abs(w))), degree_of_precision(n, closed)))
    return {"counts": np.asarray([r[0] for r in rows]),
            "all_positive": np.asarray([r[1] for r in rows]),
            "largest_weight": np.asarray([r[2] for r in rows]),
            "sum_of_absolute_weights": np.asarray([r[3] for r in rows]),
            "degree_of_precision": np.asarray([r[4] for r in rows]),
            "first_negative": next((r[0] for r in rows if not r[1]), None)}


def error_constant(n_points: int, closed: bool = True) -> dict:
    """The leading error term, found by integrating the first monomial the rule misses.

    For a rule with degree of precision ``d``, the error on ``x^{d+1}`` over ``[0, 1]`` is the
    leading constant, and the classical error formula is

        E = C h^{d+2} f^{(d+1)}(xi)

    with ``C`` the measured constant divided by ``(d+1)!``. Reporting the measured constant
    alongside the textbook one is how a transcription error in a table of error terms is caught.
    """
    n = int(n_points)
    d = degree_of_precision(n, closed)
    x = rule_nodes(n, 0.0, 1.0, closed)
    w = weights(n, closed)
    k = d + 1
    got = float(np.sum(w * x ** k))
    want = 1.0 / (k + 1)
    # The classical sign convention is error = exact - rule, so a rule that OVERestimates a
    # convex integrand has a negative constant. Reporting (rule - exact) instead reproduces
    # every textbook magnitude with every sign flipped, which looks like agreement until the
    # signs are compared.
    return {"n_points": n, "degree_of_precision": d,
            "first_missed_degree": k,
            "error_on_that_monomial": want - got,
            "error_constant": (want - got) / math.factorial(k),
            "error_order": d + 2}


#: The classical error terms, as ``name -> (coefficient, power of h, derivative order)``
#: for a **single panel** of width ``h`` between consecutive nodes.
#:
#: These are the numbers every textbook table carries. `error_constants_agree` checks them
#: against `error_constant`, which derives them from the weights.
CLASSICAL_ERRORS = {
    "trapezoid": (Fraction(-1, 12), 3, 2),
    "Simpson 1/3": (Fraction(-1, 90), 5, 4),
    "Simpson 3/8": (Fraction(-3, 80), 5, 4),
    "Boole": (Fraction(-8, 945), 7, 6),
    "Weddle": (Fraction(-9, 1400), 9, 8),
}


def error_constants_agree(tol: float = 1e-9) -> dict:
    """The classical error coefficients against the ones derived from the weights.

    The classical form is stated per panel of width ``h`` between **consecutive nodes**, so a
    rule spanning ``n - 1`` such panels has to be rescaled before the two are comparable. Getting
    that rescaling wrong is the usual reason a textbook table and a computation disagree.
    """
    rows = []
    for name, (coefficient, power, derivative) in CLASSICAL_ERRORS.items():
        n = NAMED[name][0]
        out = error_constant(n, True)
        # our constant is for a unit interval, the classical one for a panel of width 1/(n-1)
        span = n - 1
        derived = out["error_constant"] * span ** power
        classical = float(coefficient)
        rows.append((name, classical, derived, abs(derived - classical),
                     out["error_order"], power))
    return {"names": [r[0] for r in rows],
            "classical": np.asarray([r[1] for r in rows]),
            "derived": np.asarray([r[2] for r in rows]),
            "gap": np.asarray([r[3] for r in rows]),
            "order": np.asarray([r[4] for r in rows]),
            "classical_power": np.asarray([r[5] for r in rows]),
            "all_agree": bool(all(r[3] < tol * max(abs(r[1]), 1.0) for r in rows))}


# --------------------------------------------------------------------------- convergence


def convergence(f, exact, lo: float, hi: float, n_points: int = 3, closed: bool = True,
                panel_counts=None) -> dict:
    """The composite error as the panels are refined, with the fitted order.

    The fit uses only the **asymptotic tail**, and both ends have to be cut off for it to mean
    anything.

    At the fine end the error stops at the roundoff floor, and fitting through a run of zeros
    reports whatever the floor happens to do. At the coarse end the panels are too wide for the
    error term to describe them at all. On Runge's function over [-1, 1] the composite trapezoid
    error runs 4.7e-01, 4.9e-01, 1.1e-01, 7.5e-03, 1.4e-04 before settling into its clean factor
    of four: the peak is simply not resolved by the first few panels. Fitting all of that gives
    order 2.8 for a rule whose order is 2, and the same fit gives 5.2 for Simpson and 6.9 for
    Boole. Those are wrong answers, not noisy ones.

    So the tail is chosen by taking the local slope between neighbouring refinements and keeping
    the longest run at the fine end whose slopes agree with each other to within 0.25. An order
    is an integer, so 0.25 is already generous, and a looser tolerance is what produced the 2.8
    above. `points_used_in_the_fit`, `panels_dropped_from_the_head` and `note` say what happened.

    **Sometimes there is no window at all, and then the answer is nan.** A high order rule on a
    well behaved integrand can go from pre-asymptotic straight into the roundoff floor without
    ever showing its rate: composite Boole on Runge's function reaches 1e-11 by 32 panels and
    5e-16 by 128, and not one pair of consecutive refinements in between agrees with another.
    Reporting 11.3 for that, which is what a fit over everything above the floor gives, would be
    a made up number. Refine differently or use wider panels if the rate is what you need.
    """
    ms = ([1, 2, 4, 8, 16, 32, 64, 128] if panel_counts is None
          else [int(v) for v in np.atleast_1d(panel_counts)])
    want = float(exact)
    errors = []
    evaluations = []
    for m in ms:
        counter = [0]
        got = composite(f, lo, hi, m, n_points, closed, counter)
        errors.append(abs(got - want))
        evaluations.append(counter[0])
    arr = np.asarray(errors)
    counts = np.asarray(ms, dtype=float)

    floor = 1e-14 * max(abs(want), 1.0)
    usable = np.flatnonzero(arr > floor)
    order = float("nan")
    used = 0
    dropped = len(ms)
    if usable.size == 0:
        note = "the rule is exact on this integrand, so there is no rate to fit"
    elif usable.size < 3:
        note = "only %d refinements are above the roundoff floor" % int(usable.size)
    else:
        e = arr[usable]
        m_used = counts[usable]
        local = -np.diff(np.log(e)) / np.diff(np.log(m_used))
        start = local.size - 1
        while start > 0 and abs(float(local[start - 1]) - float(local[start])) <= 0.25:
            start -= 1
        tail = usable[start:]
        if tail.size >= 3:
            order = float(-np.polyfit(np.log(counts[tail]), np.log(arr[tail]), 1)[0])
            used = int(tail.size)
            dropped = int(len(ms) - tail.size)
            note = "fitted on the finest %d refinements" % used
        else:
            note = ("no run of three refinements shares a rate; the error goes from "
                    "pre-asymptotic to the roundoff floor without a window")
    return {"panels": np.asarray(ms), "errors": arr,
            "evaluations": np.asarray(evaluations),
            "fitted_order": order,
            "predicted_order": degree_of_precision(n_points, closed) + 1,
            "points_used_in_the_fit": used,
            "panels_dropped_from_the_head": dropped,
            "note": note}


def compare_rules(f, exact, lo: float, hi: float, evaluations_target: int = 400) -> dict:
    """Every named rule at roughly the same evaluation budget, which is the fair comparison.

    Comparing rules at the same **panel count** flatters the high order ones, since they use more
    points per panel. At the same number of evaluations the comparison is about accuracy per unit
    of work, which is what a caller pays for.
    """
    want = float(exact)
    rows = []
    for name, (n, closed) in NAMED.items():
        per_panel = (n - 1) if closed else (n + 1)
        m = max(1, int(evaluations_target // max(per_panel, 1)))
        counter = [0]
        got = composite(f, lo, hi, m, n, closed, counter)
        rows.append((name, n, closed, m, counter[0], abs(got - want)))
    return {"names": [r[0] for r in rows],
            "nodes": np.asarray([r[1] for r in rows]),
            "closed": np.asarray([r[2] for r in rows]),
            "panels": np.asarray([r[3] for r in rows]),
            "evaluations": np.asarray([r[4] for r in rows]),
            "errors": np.asarray([r[5] for r in rows]),
            "best": rows[int(np.argmin([r[5] for r in rows]))][0]}


def endpoint_singularity(f, exact, lo: float, hi: float, panels: int = 64) -> dict:
    """A closed rule against an open one on an integrand that is infinite at an endpoint.

    The closed rule evaluates ``f`` at the singular point and returns ``inf`` or ``nan``. The
    open rule never touches it and converges, slowly. That is the whole reason open rules exist
    and it is what lesson 66 builds on.
    """
    want = float(exact)
    out = {}
    for name, (n, closed) in (("trapezoid (closed)", (2, True)),
                              ("Simpson (closed)", (3, True)),
                              ("midpoint (open)", (1, False)),
                              ("open three point", (3, False))):
        with np.errstate(divide="ignore", invalid="ignore"):
            got = composite(f, lo, hi, panels, n, closed)
        out[name] = {"value": got, "finite": bool(np.isfinite(got)),
                     "error": (abs(got - want) if np.isfinite(got) else float("inf"))}
    return out
