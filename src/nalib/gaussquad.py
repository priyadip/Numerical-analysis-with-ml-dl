"""Gaussian quadrature: choose the nodes as well as the weights, and double the degree.

The counting argument
---------------------
A Newton-Cotes rule on ``n`` nodes fixes the nodes in advance and solves for ``n`` weights, so it
has ``n`` free parameters and reaches degree ``n - 1``. Let the nodes move too and there are
``2n`` free parameters, so the reachable degree is ``2n - 1``. Gaussian quadrature achieves it.

That is not a small gain. A 5 point Gauss rule is exact on degree 9; the 5 point Newton-Cotes
rule is exact on degree 5, and to reach degree 9 it needs 9 nodes and has already picked up a
negative weight. `degree_against_newton_cotes` puts the two side by side.

Where the nodes come from
-------------------------
The nodes are the roots of the degree ``n`` orthogonal polynomial for the weight function, and
this is the one fact that makes the whole thing work. If ``p_n`` is orthogonal to every lower
polynomial, then for any ``f`` of degree ``2n - 1``, dividing by ``p_n`` gives ``f = q p_n + r``
with ``q`` and ``r`` of degree below ``n``. The ``q p_n`` part integrates to zero by
orthogonality and vanishes at the nodes because they are roots of ``p_n``, so both sides see only
``r``, and the rule is exact on ``r`` because it is exact on degree ``n - 1``. `why_the_roots`
checks each step of that argument numerically rather than restating it.

Computing them
--------------
Root finding on ``p_n`` works and is what the subject did for a century. **Golub and Welsch is
better**: the nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix built from the
three term recurrence, and the weights are the first components of its eigenvectors, squared,
times the total mass. One symmetric eigenproblem, no root finding, and the accuracy of a
backward stable eigensolver. `nodes_and_weights` uses it. Lesson 55 built the machinery; this
lesson uses it.

What it costs
-------------
The nodes are irrational and change completely with ``n``, so nothing is reused when the rule is
refined, and there is no cheap error estimate from comparing two of them. **Gauss-Kronrod** fixes
exactly that by adding ``n + 1`` points to an ``n`` point Gauss rule so the old evaluations still
count, giving a second estimate of degree ``3n + 1`` for even ``n`` and ``3n + 2`` for odd ``n``,
for a total of ``2n + 1`` points. That pair is what production integrators actually use, and it
is why the standard rules are the odd ones: 7 point Gauss inside 15 point Kronrod, 10 inside 21,
15 inside 31.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

#: The classical families, as ``name -> (interval, weight function description)``.
FAMILIES = {
    "legendre": ((-1.0, 1.0), "w(x) = 1"),
    "chebyshev_t": ((-1.0, 1.0), "w(x) = 1 / sqrt(1 - x^2)"),
    "chebyshev_u": ((-1.0, 1.0), "w(x) = sqrt(1 - x^2)"),
    "laguerre": ((0.0, float("inf")), "w(x) = exp(-x)"),
    "hermite": ((float("-inf"), float("inf")), "w(x) = exp(-x^2)"),
}


# --------------------------------------------------------------------------- nodes and weights


def recurrence(n: int, family: str = "legendre") -> tuple:
    """The three term recurrence coefficients ``(alpha, beta, mass)`` for a classical family.

    In the monic convention ``p_{k+1} = (x - alpha_k) p_k - beta_k p_{k-1}``, with ``beta_0``
    holding the total mass of the weight function so the weights come out correctly normalised.
    These are the standard closed forms, derived in lesson 55. They are not taken on trust here:
    `moments_are_exact` checks the resulting rule against the closed form moments of each weight
    function, which no wrong recurrence coefficient could survive.
    """
    m = int(n)
    if m < 1:
        raise ValueError(f"need at least one node, got {m}")
    key = str(family).lower()
    k = np.arange(m, dtype=float)
    if key == "legendre":
        alpha = np.zeros(m)
        beta = k ** 2 / (4.0 * k ** 2 - 1.0)
        beta[0] = 2.0
    elif key == "chebyshev_t":
        alpha = np.zeros(m)
        beta = np.full(m, 0.25)
        beta[0] = math.pi
        if m > 1:
            beta[1] = 0.5
    elif key == "chebyshev_u":
        alpha = np.zeros(m)
        beta = np.full(m, 0.25)
        beta[0] = math.pi / 2.0
    elif key == "laguerre":
        alpha = 2.0 * k + 1.0
        beta = k ** 2
        beta[0] = 1.0
    elif key == "hermite":
        alpha = np.zeros(m)
        beta = k / 2.0
        beta[0] = math.sqrt(math.pi)
    else:
        raise ValueError(f"unknown family {family!r}, expected one of {sorted(FAMILIES)}")
    return alpha, beta, float(beta[0])


def nodes_and_weights(n: int, family: str = "legendre") -> tuple:
    """Gauss nodes and weights by Golub-Welsch, for any of the classical weight functions.

    The Jacobi matrix is symmetric tridiagonal with ``alpha`` on the diagonal and
    ``sqrt(beta_k)`` off it. Its eigenvalues are the nodes and its weights are
    ``mass * (first component of the unit eigenvector)^2``.

    Written this way the whole construction is one call to a symmetric eigensolver, so the nodes
    inherit that solver's backward stability. Finding the roots of the polynomial directly is the
    obvious alternative and is far worse conditioned for large ``n``, which
    `golub_welsch_beats_root_finding` measures.
    """
    m = int(n)
    alpha, beta, mass = recurrence(m, family)
    if m == 1:
        return np.asarray([alpha[0]]), np.asarray([mass])
    off = np.sqrt(beta[1:m])
    values, vectors = np.linalg.eigh(np.diag(alpha) + np.diag(off, 1) + np.diag(off, -1))
    order = np.argsort(values)
    x = values[order]
    w = mass * vectors[0, order] ** 2
    return x, w


def rule_on(n: int, lo: float, hi: float, family: str = "legendre") -> tuple:
    """Gauss-Legendre nodes and weights mapped from ``[-1, 1]`` to ``[lo, hi]``.

    Only the finite interval families can be mapped this way. Laguerre and Hermite carry their
    own infinite domains and their own weight functions, so asking for them here is an error
    rather than a silent rescaling of something that was never on ``[-1, 1]``.
    """
    key = str(family).lower()
    interval = FAMILIES.get(key, (None, None))[0]
    if interval is None:
        raise ValueError(f"unknown family {family!r}, expected one of {sorted(FAMILIES)}")
    if not np.all(np.isfinite(interval)):
        raise ValueError(
            f"{key} lives on {interval} and cannot be mapped to a finite interval; "
            "use nodes_and_weights and its own weight function instead")
    a = float(lo)
    b = float(hi)
    if not b > a:
        raise ValueError(f"need lo < hi, got [{a}, {b}]")
    x, w = nodes_and_weights(n, key)
    half = 0.5 * (b - a)
    return a + half * (x + 1.0), half * w


def integrate(f, lo: float, hi: float, n: int = 5, _counter=None) -> float:
    """Gauss-Legendre on ``[lo, hi]`` with ``n`` nodes."""
    x, w = rule_on(n, lo, hi)
    if _counter is not None:
        _counter[0] += x.size
    return float(np.sum(w * np.asarray(f(x), dtype=float)))


def composite(f, lo: float, hi: float, panels: int, n: int = 5, _counter=None) -> float:
    """Gauss-Legendre on each of ``panels`` equal pieces.

    Composing a Gauss rule is the usual way to use it on a long interval or a mildly awkward
    integrand. The panel error is ``O(h^(2n))``, so even a 3 point rule composed is a 6th order
    method.
    """
    m = int(panels)
    if m < 1:
        raise ValueError(f"need at least one panel, got {m}")
    edges = np.linspace(float(lo), float(hi), m + 1)
    return float(sum(integrate(f, float(edges[i]), float(edges[i + 1]), n, _counter)
                     for i in range(m)))


# --------------------------------------------------------------------------- the properties


def degree_of_precision(n: int, family: str = "legendre", tol: float = 1e-10) -> int:
    """The highest degree the rule integrates exactly, measured against the true moments.

    For Legendre the moments of ``x^k`` over ``[-1, 1]`` are ``2/(k+1)`` for even ``k`` and zero
    for odd, and the measurement returns ``2n - 1``.

    **This measurement stops working around ``n = 22``, and that is a fact about Gauss rules
    rather than a defect here.** The rule's relative error on ``x^(2n)``, the very first monomial
    it cannot integrate exactly, shrinks fast:

        n         3        5        8       12       16       20       24
        error   4.6e-02  2.9e-03  4.7e-05  1.8e-07  7.2e-10  2.8e-12  1.1e-14

    By ``n = 20`` that error is below the default tolerance, so the test walks past the true
    boundary and reports 41 instead of 39. By ``n = 24`` it is at the roundoff level and no
    tolerance can separate the two. `degree_boundary_report` shows the margin so the point at
    which the answer stops meaning anything is visible instead of guessed at.
    """
    m = int(n)
    x, w = nodes_and_weights(m, family)
    key = str(family).lower()
    if key != "legendre":
        raise ValueError("the exact moments are only wired up for legendre here; "
                         "use moments_are_exact for the other families")
    for k in range(0, 2 * m + 2):
        got = float(np.sum(w * x ** k))
        want = 0.0 if k % 2 else 2.0 / (k + 1.0)
        if abs(got - want) > tol * max(abs(want), 1.0):
            return k - 1
    return 2 * m + 1


def degree_boundary_report(counts=None, tol: float = 1e-10) -> dict:
    """How sharp the degree boundary still is, node count by node count.

    ``margin`` is the relative error on ``x^(2n)``, the first monomial the rule misses. As long
    as it is comfortably above the tolerance the measured degree is the true one. When it falls
    below, `degree_of_precision` overshoots, and `measurable` says so rather than leaving the
    caller to notice.
    """
    ns = ([2, 3, 5, 8, 12, 16, 20, 24, 28] if counts is None
          else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        x, w = nodes_and_weights(n, "legendre")
        k = 2 * n
        got = float(np.sum(w * x ** k))
        want = 2.0 / (k + 1.0)
        margin = abs(got - want) / abs(want)
        measured = degree_of_precision(n, "legendre", tol)
        rows.append((n, 2 * n - 1, measured, margin, bool(margin > tol)))
    return {"nodes": np.asarray([r[0] for r in rows]),
            "true_degree": np.asarray([r[1] for r in rows]),
            "measured_degree": np.asarray([r[2] for r in rows]),
            "margin": np.asarray([r[3] for r in rows]),
            "measurable": np.asarray([r[4] for r in rows]),
            "last_measurable": max([r[0] for r in rows if r[4]], default=None),
            "note": ("the margin is the relative error on x^(2n); once it drops below the "
                     "tolerance the measured degree overshoots the true one")}


def moments_are_exact(n: int, family: str = "legendre", tol: float = 1e-9) -> dict:
    """Every moment up to degree ``2n - 1``, against its closed form, for any family.

    The closed forms are the standard ones: the Gamma function at integer and half integer
    arguments for Laguerre and Hermite, and the Beta function for the Chebyshev weights. Testing
    against them rather than against a numerical integral is what makes this a check on the rule
    instead of a check on some other quadrature.

    **The gap is measured against ``sum |w_j x_j^k|``, the size of the terms being added, not
    against the answer.** For Hermite at 16 nodes the moment of ``x^31`` is exactly zero and the
    terms summing to it reach 3.3e11, so the best any arithmetic can do is about 1e-5. Judged
    against a scale of 1 that looks like a catastrophic failure at 4.6e-03; judged against the
    terms it is 1.4e-16, which is the truth. An odd moment of a symmetric weight is a pure
    cancellation and can only ever be as accurate as the numbers cancelling.
    """
    m = int(n)
    key = str(family).lower()
    x, w = nodes_and_weights(m, key)
    rows = []
    for k in range(2 * m):
        got = float(np.sum(w * x ** k))
        if key == "legendre":
            want = 0.0 if k % 2 else 2.0 / (k + 1.0)
        elif key == "hermite":
            # integral of x^k exp(-x^2) is 0 for odd k, Gamma((k+1)/2) for even
            want = 0.0 if k % 2 else math.gamma((k + 1) / 2.0)
        elif key == "laguerre":
            want = math.gamma(k + 1.0)
        elif key == "chebyshev_t":
            # integral of x^k / sqrt(1-x^2) over (-1,1)
            want = 0.0 if k % 2 else math.pi * math.comb(k, k // 2) / 2.0 ** k
        elif key == "chebyshev_u":
            want = (0.0 if k % 2 else
                    math.pi * math.comb(k, k // 2) / (2.0 ** (k + 1) * (k // 2 + 1)))
        else:
            raise ValueError(f"unknown family {family!r}")
        scale = float(np.sum(np.abs(w * x ** k)))
        rows.append((k, got, want, abs(got - want) / max(abs(want), scale, 1e-300), scale))
    gaps = np.asarray([r[3] for r in rows])
    return {"degree": np.asarray([r[0] for r in rows]),
            "computed": np.asarray([r[1] for r in rows]),
            "exact": np.asarray([r[2] for r in rows]),
            "term_size": np.asarray([r[4] for r in rows]),
            "relative_gap": gaps,
            "all_exact": bool(np.max(gaps) < tol),
            "worst_gap": float(np.max(gaps))}


def why_the_roots(n: int, family: str = "legendre") -> dict:
    """Check the orthogonality argument step by step instead of restating it.

    Three claims, each measured:

    1. The nodes are the roots of the degree ``n`` orthogonal polynomial. Evaluated by the three
       term recurrence at the nodes, ``p_n`` should be zero.
    2. ``p_n`` is orthogonal to every lower degree polynomial against the weight, so the rule
       applied to ``x^j p_n(x)`` should give zero for ``j < n``.
    3. The weights are all positive, which follows from applying the rule to the square of a
       Lagrange basis polynomial and is what makes Gauss rules stable at every ``n``.
    """
    m = int(n)
    alpha, beta, mass = recurrence(m + 1, family)
    x, w = nodes_and_weights(m, family)

    def orthogonal_polynomial(t, degree):
        """Monic p_degree by the three term recurrence."""
        prev = np.zeros_like(np.asarray(t, dtype=float))
        cur = np.ones_like(np.asarray(t, dtype=float))
        for k in range(degree):
            nxt = (t - alpha[k]) * cur - (beta[k] if k > 0 else 0.0) * prev
            prev, cur = cur, nxt
        return cur

    at_nodes = float(np.max(np.abs(orthogonal_polynomial(x, m))))
    scale = float(np.max(np.abs(orthogonal_polynomial(np.linspace(
        np.min(x), np.max(x), 101), m)))) if m > 0 else 1.0
    orthogonality = [float(abs(np.sum(w * x ** j * orthogonal_polynomial(x, m))))
                     for j in range(m)]
    return {"p_n_at_the_nodes": at_nodes,
            "p_n_typical_size": scale,
            "relative_residual_at_the_nodes": at_nodes / max(scale, 1e-300),
            "orthogonality_residuals": np.asarray(orthogonality),
            "weights": w,
            "all_weights_positive": bool(np.all(w > 0.0)),
            "smallest_weight": float(np.min(w))}


def weights_stay_positive(counts=None, family: str = "legendre") -> dict:
    """The property Newton-Cotes loses at 9 nodes and Gauss never loses.

    Positive weights summing to the total mass make the rule a weighted average, so it cannot
    amplify. That holds for every ``n``, which is why a Gauss rule can be used at high order
    while a Newton-Cotes rule cannot.

    **In exact arithmetic every Gauss weight is strictly positive. In floating point the
    outermost ones can underflow to zero,** for the families whose weight function decays
    exponentially. The Hermite weights fall to 4.6e-59 by 120 nodes and the Laguerre ones to
    3.5e-61, and squaring an eigenvector component that small flushes to zero. Five Laguerre
    weights are exactly zero at 120 nodes.

    That costs nothing: those nodes sit where the weight function is negligible, and the total
    mass still comes out right to 1e-15. `weight_underflow` measures it so the zeros are
    accounted for rather than discovered as a failed positivity test.
    """
    ns = ([1, 2, 3, 5, 8, 13, 21, 34, 55, 89] if counts is None
          else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        x, w = nodes_and_weights(n, family)
        rows.append((n, bool(np.all(w > 0.0)), float(np.min(w)), float(np.sum(np.abs(w))),
                     float(np.sum(w))))
    return {"counts": np.asarray([r[0] for r in rows]),
            "all_positive": np.asarray([r[1] for r in rows]),
            "smallest_weight": np.asarray([r[2] for r in rows]),
            "sum_of_absolute_weights": np.asarray([r[3] for r in rows]),
            "sum_of_weights": np.asarray([r[4] for r in rows]),
            "always_positive": bool(all(r[1] for r in rows)),
            "never_negative": bool(all(r[2] >= 0.0 for r in rows))}


def weight_underflow(counts=None, family: str = "hermite") -> dict:
    """Where the smallest weights stop fitting in a double, and whether it matters.

    Reported as the smallest strictly positive weight, the count that reached exactly zero, and
    the error in the total mass. The last column is the one that answers the question: the
    underflowed weights multiply function values at nodes where the weight function is already
    negligible, so the rule is unaffected.
    """
    ns = ([21, 34, 55, 89, 120] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        x, w = nodes_and_weights(n, family)
        _, _, mass = recurrence(n, family)
        positive = w[w > 0.0]
        rows.append((n, float(np.min(positive)) if positive.size else float("nan"),
                     int(np.sum(w <= 0.0)), float(np.max(np.abs(x))),
                     abs(float(np.sum(w)) - mass)))
    return {"counts": np.asarray([r[0] for r in rows]),
            "smallest_positive_weight": np.asarray([r[1] for r in rows]),
            "weights_that_underflowed": np.asarray([r[2] for r in rows]),
            "largest_node": np.asarray([r[3] for r in rows]),
            "total_mass_error": np.asarray([r[4] for r in rows]),
            "any_negative": bool(any(float(np.min(nodes_and_weights(n, family)[1])) < 0.0
                                     for n in ns))}


def degree_against_newton_cotes(counts=None) -> dict:
    """Degree of precision per node, Gauss against Newton-Cotes.

    Gauss reaches ``2n - 1`` on ``n`` nodes. Newton-Cotes reaches ``n - 1``, or ``n`` when ``n``
    is odd. The ratio approaches 2, and the Newton-Cotes column stops being usable long before
    the degrees it quotes become interesting.
    """
    from . import newtoncotes as _nc

    ns = ([2, 3, 4, 5, 6, 7, 8, 9, 11] if counts is None
          else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        # the Gauss degree is 2n - 1 by construction; it is quoted rather than measured here
        # because the measurement stops being possible past about n = 22, as
        # `degree_boundary_report` shows. The Newton-Cotes side is measured exactly, in
        # rational arithmetic, so this column is the one carrying the evidence.
        g = 2 * n - 1
        c = _nc.degree_of_precision(n, True)
        x, w = nodes_and_weights(n)
        cw = _nc.weights(n, True)
        rows.append((n, g, c, float(g) / max(c, 1),
                     bool(np.all(w > 0.0)), bool(np.all(cw >= 0.0))))
    return {"nodes": np.asarray([r[0] for r in rows]),
            "gauss_degree": np.asarray([r[1] for r in rows]),
            "newton_cotes_degree": np.asarray([r[2] for r in rows]),
            "ratio": np.asarray([r[3] for r in rows]),
            "gauss_weights_positive": np.asarray([r[4] for r in rows]),
            "newton_cotes_weights_positive": np.asarray([r[5] for r in rows])}


# --------------------------------------------------------------------------- convergence


def convergence(f, exact, lo: float, hi: float, counts=None, floor: float = 1e-15) -> dict:
    """Error against node count, and whether it falls geometrically or as a power.

    For an integrand analytic in a neighbourhood of the interval, Gauss quadrature converges
    geometrically in ``n``, not at any fixed algebraic order. For one with limited smoothness it
    converges at a rate set by that smoothness. Distinguishing the two by fitting both laws and
    comparing residuals is more honest than quoting a rate.
    """
    ns = ([1, 2, 3, 4, 6, 8, 12, 16, 24, 32] if counts is None
          else [int(v) for v in np.atleast_1d(counts)])
    want = float(exact)
    errors = []
    for n in ns:
        errors.append(abs(integrate(f, lo, hi, n) - want))
    e = np.asarray(errors)
    k = np.asarray(ns, dtype=float)
    keep = e > floor * max(abs(want), 1.0)
    if int(np.sum(keep)) < 3:
        return {"counts": np.asarray(ns), "errors": e,
                "fitted_power": float("nan"), "fitted_rate": float("nan"),
                "power_residual": float("nan"), "geometric_residual": float("nan"),
                "geometric_wins": False, "points_used": int(np.sum(keep)),
                "note": "not enough points above the roundoff floor to tell the laws apart"}
    y = np.log(e[keep])
    p_fit, p_res = np.polyfit(np.log(k[keep]), y, 1, full=True)[:2]
    g_fit, g_res = np.polyfit(k[keep], y, 1, full=True)[:2]
    pr = float(p_res[0]) if len(p_res) else 0.0
    gr = float(g_res[0]) if len(g_res) else 0.0
    return {"counts": np.asarray(ns), "errors": e,
            "fitted_power": float(-p_fit[0]), "fitted_rate": float(-g_fit[0]),
            "power_residual": pr, "geometric_residual": gr,
            "geometric_wins": bool(gr < pr), "points_used": int(np.sum(keep)),
            "note": "residual of log(error) against log(n) versus against n"}


def against_newton_cotes(f, exact, lo: float, hi: float, budgets=None) -> dict:
    """Gauss against composite Simpson at the same number of function evaluations.

    The only comparison that means anything is at equal cost. Gauss gets one rule with ``n``
    nodes; Simpson gets as many panels as the same budget buys.
    """
    from . import newtoncotes as _nc

    bs = ([5, 9, 17, 33, 65] if budgets is None else [int(v) for v in np.atleast_1d(budgets)])
    want = float(exact)
    rows = []
    for budget in bs:
        gauss = abs(integrate(f, lo, hi, budget) - want)
        panels = max(1, (budget - 1) // 2)
        counter = [0]
        simpson = abs(_nc.composite_shared(f, lo, hi, panels, 3, counter) - want)
        rows.append((budget, gauss, counter[0], simpson))
    return {"evaluations": np.asarray([r[0] for r in rows]),
            "gauss_error": np.asarray([r[1] for r in rows]),
            "simpson_evaluations": np.asarray([r[2] for r in rows]),
            "simpson_error": np.asarray([r[3] for r in rows]),
            "gauss_wins": np.asarray([r[1] < r[3] for r in rows])}


def golub_welsch_beats_root_finding(counts=None) -> dict:
    """The eigenvalue route against finding the roots of the polynomial in the power basis.

    Expanding the orthogonal polynomial in the power basis and calling a root finder is the
    obvious approach and is a disaster at moderate ``n``, for the reason lesson 47 gave: the
    power basis coefficients of a high degree orthogonal polynomial span an enormous range, and
    the roots are exponentially sensitive to them. The Jacobi matrix never forms those
    coefficients.

    The comparison is against the residual ``|p_n(x_i)|`` measured by the stable three term
    recurrence, so neither method is being graded by its own arithmetic.
    """
    ns = ([4, 8, 16, 24, 32] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        alpha, beta, _ = recurrence(n + 1, "legendre")

        def p(t, degree=n):
            prev = np.zeros_like(np.asarray(t, dtype=float))
            cur = np.ones_like(np.asarray(t, dtype=float))
            for k in range(degree):
                nxt = (t - alpha[k]) * cur - (beta[k] if k > 0 else 0.0) * prev
                prev, cur = cur, nxt
            return cur

        # the stable route
        x_gw, _ = nodes_and_weights(n, "legendre")
        # the power basis route: build the monic coefficients by the same recurrence, then
        # hand them to a companion matrix root finder
        prev = np.zeros(n + 1)
        cur = np.zeros(n + 1)
        cur[0] = 1.0
        for k in range(n):
            nxt = np.zeros(n + 1)
            nxt[1:] += cur[:-1]
            nxt -= alpha[k] * cur
            if k > 0:
                nxt -= beta[k] * prev
            prev, cur = cur, nxt
        power = cur[::-1]           # highest degree first, as np.roots wants
        x_pb = np.sort(np.real(np.roots(power)))
        typical = float(np.max(np.abs(p(np.linspace(-1.0, 1.0, 201)))))
        rows.append((n,
                     float(np.max(np.abs(p(x_gw)))) / typical,
                     float(np.max(np.abs(p(x_pb)))) / typical,
                     float(np.max(np.abs(power))) / max(float(np.min(np.abs(
                         power[np.abs(power) > 0]))), 1e-300)))
    return {"nodes": np.asarray([r[0] for r in rows]),
            "golub_welsch_residual": np.asarray([r[1] for r in rows]),
            "power_basis_residual": np.asarray([r[2] for r in rows]),
            "coefficient_range": np.asarray([r[3] for r in rows])}


# --------------------------------------------------------------------------- Kronrod


def kronrod(n: int) -> tuple:
    """The ``2n + 1`` point Gauss-Kronrod rule that reuses an ``n`` point Gauss rule.

    Kronrod's construction adds ``n + 1`` new nodes to the ``n`` Gauss nodes and chooses all
    ``2n + 1`` weights so the combined rule reaches degree ``3n + 1``, or ``3n + 2`` when ``n``
    is odd. The Gauss nodes stay
    exactly where they were, so **the ``n`` function values already computed are still usable**,
    and the difference between the two estimates costs nothing beyond the new points.

    The construction extends the Jacobi matrix: given the first ``ceil(3n/2) + 1`` recurrence
    coefficients of the Legendre family, Laurie's algorithm produces the remaining coefficients
    of a ``2n + 1`` by ``2n + 1`` symmetric tridiagonal matrix whose eigenvalues and first
    eigenvector components are the Kronrod nodes and weights. The bookkeeping is intricate and
    is not worth reading closely; what matters is that the result is checked, and
    `kronrod_report` checks all three things it must satisfy: it contains the Gauss nodes,
    it reaches degree ``3n + 1``, and its weights are positive.

    Reference: Dirk Laurie, "Calculation of Gauss-Kronrod quadrature rules", Mathematics of
    Computation 66 (1997) 1133-1145. The variable names follow Gautschi's OPQ implementation of
    it so the two can be read against each other.
    """
    m = int(n)
    if m < 1:
        raise ValueError(f"need at least one Gauss node, got {m}")
    needed = int(math.ceil(3.0 * m / 2.0)) + 1
    alpha, beta, mass = recurrence(max(needed, 2 * m + 1), "legendre")

    a = np.zeros(2 * m + 1)
    b = np.zeros(2 * m + 1)
    top_a = min(int(math.floor(3.0 * m / 2.0)), 2 * m)
    top_b = min(int(math.ceil(3.0 * m / 2.0)), 2 * m)
    a[:top_a + 1] = alpha[:top_a + 1]
    b[:top_b + 1] = beta[:top_b + 1]

    size = m // 2 + 2
    s = np.zeros(size)
    t = np.zeros(size)
    t[1] = b[m + 1]

    for step in range(0, m - 1):
        u = 0.0
        for k in range((step + 1) // 2, -1, -1):
            l = step - k
            u += ((a[k + m + 1] - a[l]) * t[k + 1]
                  + b[k + m + 1] * s[k] - b[l] * s[k + 1])
            s[k + 1] = u
        s, t = t, s

    for j in range(m // 2, -1, -1):
        s[j + 1] = s[j]

    for step in range(m - 1, 2 * m - 2):
        u = 0.0
        j = 0
        for k in range(step + 1 - m, (step - 1) // 2 + 1):
            l = step - k
            j = m - 1 - l
            # this accumulates across k; reassigning u instead of subtracting into it gives
            # nodes that look plausible and are wrong in the second decimal place
            u -= ((a[k + m + 1] - a[l]) * t[j + 1]
                  + b[k + m + 1] * s[j + 1] - b[l] * s[j + 2])
            s[j + 1] = u
        if step % 2 == 0:
            k = step // 2
            a[k + m + 1] = a[k] + (s[j + 1] - b[k + m + 1] * s[j + 2]) / t[j + 2]
        else:
            k = (step + 1) // 2
            b[k + m + 1] = s[j + 1] / s[j + 2]
        s, t = t, s

    a[2 * m] = a[m - 1] - b[2 * m] * s[1] / t[1]

    off = np.sqrt(np.abs(b[1:2 * m + 1]))
    values, vectors = np.linalg.eigh(np.diag(a) + np.diag(off, 1) + np.diag(off, -1))
    order = np.argsort(values)
    return values[order], mass * vectors[0, order] ** 2


def kronrod_report(n: int = 7, tol: float = 1e-9) -> dict:
    """What the Kronrod extension actually delivers, measured rather than quoted.

    Three claims, all measured against the exact Legendre moments: the Gauss nodes are contained
    in the Kronrod nodes, the weights are positive, and the rule reaches its degree.

    **The degree is ``3n + 1`` for even ``n`` and ``3n + 2`` for odd ``n``,** which the usual
    "at least ``3n + 1``" statement leaves out. Measured:

        n           1   2   3   4   5   6   7
        degree      5   7  11  13  17  19  23
        3n + 1      4   7  10  13  16  19  22

    Past ``n = 7`` the measurement stops working for the same reason it does in
    `degree_boundary_report`: the rule's error on the first monomial it misses falls to 1e-9 and
    below, so the walk goes straight past the boundary. ``measurable`` says whether the reported
    degree can be believed.
    """
    m = int(n)
    gx, gw = nodes_and_weights(m, "legendre")
    kx, kw = kronrod(m)
    contained = float(np.max([np.min(np.abs(kx - v)) for v in gx]))
    degree = -1
    for k in range(0, 5 * m + 6):
        got = float(np.sum(kw * kx ** k))
        want = 0.0 if k % 2 else 2.0 / (k + 1.0)
        if abs(got - want) > tol * max(abs(want), 1.0):
            break
        degree = k
    k = degree + 1
    got = float(np.sum(kw * kx ** k))
    want = 0.0 if k % 2 else 2.0 / (k + 1.0)
    margin = abs(got - want) / max(abs(want), 1.0)
    predicted = 3 * m + 1 + (m % 2)
    return {"gauss_nodes": gx, "kronrod_nodes": kx, "kronrod_weights": kw,
            "point_count": kx.size,
            "gauss_nodes_are_contained": bool(contained < 1e-10),
            "worst_containment_gap": contained,
            "measured_degree": degree,
            "predicted_degree": predicted,
            "margin": margin,
            "measurable": bool(margin > 100.0 * tol),
            "weights_positive": bool(np.all(kw > 0.0)),
            "weight_sum_error": abs(float(np.sum(kw)) - 2.0)}


def gauss_kronrod_estimate(f, lo: float, hi: float, n: int = 7, _counter=None) -> dict:
    """The pair used in practice: a value, and an error estimate that costs almost nothing.

    Both rules are evaluated on the same ``2n + 1`` points, because the Gauss nodes are a subset
    of the Kronrod nodes. The difference between them estimates the error of the Gauss rule, and
    the Kronrod value is returned because it is the better of the two.

    The usual scaling ``(200 |G - K|)^1.5`` is what QUADPACK reports; it is heuristic, and both
    the raw and scaled forms are returned so the choice is visible.
    """
    a = float(lo)
    b = float(hi)
    m = int(n)
    kx, kw = kronrod(m)
    gx, gw = nodes_and_weights(m, "legendre")
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    points = mid + half * kx
    if _counter is not None:
        _counter[0] += points.size
    values = np.asarray(f(points), dtype=float)
    kronrod_value = half * float(np.sum(kw * values))
    # pick out the Gauss nodes from the Kronrod set rather than evaluating again
    index = [int(np.argmin(np.abs(kx - v))) for v in gx]
    gauss_value = half * float(np.sum(gw * values[index]))
    raw = abs(kronrod_value - gauss_value)
    scale = abs(kronrod_value)
    return {"value": kronrod_value, "gauss_value": gauss_value,
            "raw_difference": raw,
            "quadpack_estimate": min(scale, (200.0 * raw / max(scale, 1e-300)) ** 1.5 * scale)
            if scale > 0 else raw,
            "evaluations": points.size}


def kronrod_estimate_quality(f, exact, lo: float, hi: float, counts=None) -> dict:
    """Is the Gauss-Kronrod difference actually a useful error estimate?

    It estimates the error of the **Gauss** value, and the value returned is the **Kronrod**
    one, which is much better. So the estimate is normally a large overestimate of the error in
    the answer, which is the safe direction but is not what it looks like. Reporting the ratio
    both ways makes that plain.
    """
    ns = ([3, 5, 7, 10, 15] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    want = float(exact)
    # Once an error reaches zero the ratio is meaningless, so it is floored at the roundoff
    # level of the answer rather than at the smallest positive float. Dividing by 1e-300
    # instead produces numbers like 1.9e287, which look like findings and are not.
    floor = np.finfo(float).eps * max(abs(want), 1.0)
    rows = []
    for n in ns:
        out = gauss_kronrod_estimate(f, lo, hi, n)
        gauss_error = abs(out["gauss_value"] - want)
        kronrod_error = abs(out["value"] - want)
        rows.append((n, out["raw_difference"], gauss_error, kronrod_error,
                     out["raw_difference"] / max(gauss_error, floor),
                     out["raw_difference"] / max(kronrod_error, floor),
                     out["evaluations"]))
    return {"gauss_points": np.asarray([r[0] for r in rows]),
            "difference": np.asarray([r[1] for r in rows]),
            "gauss_error": np.asarray([r[2] for r in rows]),
            "kronrod_error": np.asarray([r[3] for r in rows]),
            "difference_over_gauss_error": np.asarray([r[4] for r in rows]),
            "difference_over_kronrod_error": np.asarray([r[5] for r in rows]),
            "evaluations": np.asarray([r[6] for r in rows]),
            "roundoff_floor": float(floor)}
