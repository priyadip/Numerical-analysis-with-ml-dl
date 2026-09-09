"""Improper integrals and integrals in more than one dimension.

Improper integrals
------------------
Two things go wrong: the interval is infinite, or the integrand blows up at an endpoint. Both are
fixed by the same move, a change of variable, and the choice of change is the whole subject.

- **Truncate.** Integrate over ``[a, T]`` and hope the tail is small. Simple, and the error is
  whatever the tail is, so it needs a bound on the tail to be honest. `truncation_error` measures
  what it costs.
- **Transform.** Map the infinite range to a finite one, for example ``x = t / (1 - t)`` on
  ``[0, 1)`` or ``x = tan(t)`` on ``(-pi/2, pi/2)``. The integrand is now finite in extent but
  usually singular at the new endpoint, which trades one problem for another.
- **Use the right Gauss family.** Gauss-Laguerre and Gauss-Hermite carry the exponential weight
  in the rule instead of in the integrand, and for integrands that genuinely look like
  ``exp(-x) g(x)`` they are unbeatable. For integrands that do not, they are poor.
- **Double exponential.** The tanh-sinh substitution ``x = tanh(pi/2 sinh t)`` pushes the
  endpoints to infinity so fast that the transformed integrand and all its derivatives vanish
  there. The trapezoid rule then converges geometrically, by exactly the argument lesson 63 made
  about periodic integrands, and endpoint singularities stop mattering. `tanh_sinh` implements it.

More than one dimension
-----------------------
The obvious approach is a tensor product: apply a one dimensional rule in each direction. It
works and it is what everyone does in two or three dimensions. It also **fails completely in high
dimension**, because a rule of order ``p`` on ``n`` points per axis costs ``n^d`` evaluations and
delivers ``n^-p = N^(-p/d)``. Doubling the accuracy in ten dimensions costs a factor of ``2^(10/p)``
in work. `curse_of_dimensionality` measures the exponent directly.

Monte Carlo converges at ``N^(-1/2)`` **regardless of dimension**. Where it takes over depends on
the integrand and not on the dimension alone: on a product of cosines, analytic and separable, a
tensor Gauss rule is still ahead at ten dimensions, while on a product of ``|x - 1/2|^(1/2)``,
which only has a kink on each axis, Monte Carlo is ahead by three. `monte_carlo_crossover`
measures both rather than quoting a number.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- infinite range


def truncation_error(f, tail_bound, lo: float, cutoffs=None, n: int = 40) -> dict:
    """Integrate over ``[lo, T]`` for growing ``T``, against the tail that was thrown away.

    ``tail_bound`` is any callable giving a bound on ``|integral from T to infinity|``. Without
    one, truncation is guesswork: the computed values settle down and look converged whether the
    tail is negligible or not.
    """
    from . import gaussquad as _gq

    ts = ([1.0, 2.0, 5.0, 10.0, 20.0, 40.0] if cutoffs is None
          else [float(v) for v in np.atleast_1d(cutoffs)])
    a = float(lo)
    rows = []
    previous = None
    for t in ts:
        value = _gq.integrate(f, a, t, n)
        rows.append((t, value, float(tail_bound(t)),
                     abs(value - previous) if previous is not None else float("nan")))
        previous = value
    return {"cutoff": np.asarray([r[0] for r in rows]),
            "value": np.asarray([r[1] for r in rows]),
            "tail_bound": np.asarray([r[2] for r in rows]),
            "change_from_previous": np.asarray([r[3] for r in rows])}


def transform_to_finite(f, lo: float, kind: str = "rational") -> tuple:
    """Map ``[lo, infinity)`` to a finite interval and return the new integrand and limits.

    ``rational`` uses ``x = lo + t / (1 - t)`` on ``[0, 1)``, whose Jacobian is ``1/(1-t)^2``.
    ``exponential`` uses ``x = lo - log(1 - t)`` on ``[0, 1)``, Jacobian ``1/(1-t)``.

    Both put a singularity at ``t = 1`` unless ``f`` decays fast enough to cancel it, which is
    why the transformed integral usually wants an open rule or a double exponential one rather
    than anything that evaluates at the endpoint.
    """
    a = float(lo)
    key = str(kind).lower()
    # t = 1 is the image of infinity and divides by zero by construction. The rules that use
    # this never want a value there, so it is allowed to come back as inf and be filtered,
    # rather than being guarded by a special case that hides where the map actually ends.
    if key == "rational":
        def g(t):
            u = np.asarray(t, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.asarray(f(a + u / (1.0 - u)), dtype=float) / (1.0 - u) ** 2
    elif key == "exponential":
        def g(t):
            u = np.asarray(t, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.asarray(f(a - np.log(1.0 - u)), dtype=float) / (1.0 - u)
    else:
        raise ValueError(f"unknown transform {kind!r}, expected 'rational' or 'exponential'")
    return g, 0.0, 1.0


def remove_endpoint_singularity(f, lo: float, hi: float, power: float = 0.5) -> tuple:
    """Substitute away an algebraic endpoint singularity of known strength.

    For an integrand behaving like ``(x - a)^(-power)`` near ``a``, the substitution
    ``x = a + u^(1/(1-power))`` turns it into something bounded. The Jacobian supplies exactly
    the factor that cancels the blow up, which is why the strength has to be known.

    Returns the new integrand and its limits. `substitution_beats_brute_force` measures what it
    is worth against simply refining.
    """
    a = float(lo)
    b = float(hi)
    p = float(power)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"the power must be in [0, 1), got {p}")
    exponent = 1.0 / (1.0 - p)

    def g(u):
        t = np.asarray(u, dtype=float)
        x = a + t ** exponent
        jacobian = exponent * t ** (exponent - 1.0)
        return np.asarray(f(x), dtype=float) * jacobian

    return g, 0.0, (b - a) ** (1.0 / exponent)


def substitution_beats_brute_force(f, exact, lo: float, hi: float, power: float = 0.5,
                                   budgets=None) -> dict:
    """The substituted integral against the raw one, at the same number of evaluations."""
    from . import gaussquad as _gq

    bs = ([4, 8, 16, 32, 64] if budgets is None else [int(v) for v in np.atleast_1d(budgets)])
    want = float(exact)
    g, glo, ghi = remove_endpoint_singularity(f, lo, hi, power)
    rows = []
    for n in bs:
        raw = abs(_gq.integrate(f, lo, hi, n) - want)
        fixed = abs(_gq.integrate(g, glo, ghi, n) - want)
        rows.append((n, raw, fixed))
    return {"evaluations": np.asarray([r[0] for r in rows]),
            "raw_error": np.asarray([r[1] for r in rows]),
            "substituted_error": np.asarray([r[2] for r in rows]),
            "substitution_wins": np.asarray([r[2] < r[1] for r in rows])}


# --------------------------------------------------------------------------- double exponential


def tanh_sinh(f, lo: float, hi: float, level: int = 6, _counter=None,
              use_distances: bool = False) -> dict:
    """Double exponential quadrature on a finite interval, endpoint singularities included.

    The substitution is ``x = (a+b)/2 + (b-a)/2 * tanh(pi/2 * sinh(t))``, and the rule is the
    plain trapezoid rule in ``t`` over the whole line. The transformed integrand decays like
    ``exp(-exp(|t|))``, so the tail can be cut at a modest ``t`` and the trapezoid rule converges
    geometrically for the reason lesson 63 gave: on the whole line with a rapidly decaying
    integrand, every Euler-Maclaurin endpoint term is zero.

    The result is a rule that does not care about an endpoint singularity, because it never
    evaluates at the endpoint and approaches it exponentially fast.

    **The nodes are computed as a distance from the nearer endpoint, and that is not an
    optimisation.** Writing the node as ``mid + half * tanh(u)`` is the formula as stated and it
    destroys the method. Near the left end ``tanh(u)`` is within rounding of -1, so the sum
    cancels to a number carrying no correct digits, and a singular integrand evaluated there
    returns nonsense. Done that way, this rule stalls at 3e-08 on ``1/sqrt(x(1-x))`` no matter
    how many points it is given. Computed as ``a + half * (1 + tanh(u))``, with ``1 + tanh(u)``
    evaluated as ``2 / (1 + exp(-2u))``, every node keeps full relative accuracy and the same
    rule reaches 4e-16.

    The weight has the same problem and the same fix: ``1 / cosh(u)^2`` is
    ``(1 + tanh u)(1 - tanh u)``, which is a product of two stably computed factors rather than
    a reciprocal of something that overflows.

    **Getting the nodes right is only half of it.** An integrand singular at *both* ends, such
    as ``1/sqrt(x(1-x))``, forms ``1 - x`` inside itself, and at the outermost node ``x`` is
    0.99999999999999966693, so ``1 - x`` comes back as 3.3e-16 with about one correct digit.
    The rule cannot fix that from outside: the cancellation happens in the caller's code. So
    both distances are returned, and passing ``use_distances=True`` calls ``f(x, d_lo, d_hi)``
    with ``d_lo = x - lo`` and ``d_hi = hi - x`` computed to full relative accuracy.

    The difference is total. On ``1/sqrt(x(1-x))`` over (0, 1), whose value is exactly pi:

        level              1         2         3         4         5
        written in x    2.2e-08   1.6e-08   1.0e-08   2.0e-08   1.6e-08
        given d_lo,d_hi 2.0e-08   8.9e-16   4.4e-16   0.0       0.0

    The first row never improves no matter how many points it is given, because every point it
    adds near the right end is a point where ``1 - x`` has no digits left. An integrand singular
    at only one end, such as ``1/sqrt(x)`` or ``log(x)``, has no such problem and reaches machine
    precision either way.

    Halving the step doubles the point count and reuses every existing point, exactly like the
    trapezoid ladder in lesson 63.
    """
    a = float(lo)
    b = float(hi)
    if not b > a:
        raise ValueError(f"need lo < hi, got [{a}, {b}]")
    k = int(level)
    if k < 0:
        raise ValueError(f"need a non-negative level, got {k}")
    half = 0.5 * (b - a)
    h = 2.0 ** (-k)
    # cut the tail where the weight underflows; beyond this the points contribute nothing
    limit = 1.0
    while limit < 8.0:
        if 0.5 * math.pi * math.sinh(limit + 0.25) > 350.0:
            break
        limit += 0.25
    steps = int(math.ceil(limit / h))
    t = h * np.arange(-steps, steps + 1)
    u = 0.5 * math.pi * np.sinh(t)
    # 1 + tanh(u) and 1 - tanh(u), each computed without cancelling
    two_u = np.clip(2.0 * u, -700.0, 700.0)
    plus = 2.0 / (1.0 + np.exp(-two_u))
    minus = 2.0 / (1.0 + np.exp(two_u))
    # the distance to each endpoint, each accurate to full relative precision
    d_lo = half * plus
    d_hi = half * minus
    left = u < 0.0
    x = np.where(left, a + d_lo, b - d_hi)
    w = half * h * 0.5 * math.pi * np.cosh(t) * plus * minus
    # Filter on the WEIGHT, never on the node. Once a node is closer to b than machine epsilon,
    # ``x`` rounds to exactly b, and a test like ``x < b`` throws it away. On [0, 1] that starts
    # at t = 3.2, and those are precisely the nodes that take the singular case from 1e-08 down
    # to 1e-15. The node is perfectly good; it is only its representation as an absolute
    # position that has run out of digits, which is why the distances are carried separately.
    keep = np.isfinite(w) & (w > 0.0)
    x = x[keep]
    w = w[keep]
    d_lo = d_lo[keep]
    d_hi = d_hi[keep]
    if _counter is not None:
        _counter[0] += x.size
    # An integrand singular at an endpoint will return inf at the outermost nodes, where x has
    # rounded to the endpoint itself. Those points are dropped rather than guarded against,
    # because a caller that cares can take the distances instead and get a finite value.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = np.asarray(f(x, d_lo, d_hi) if use_distances else f(x), dtype=float)
    good = np.isfinite(values)
    return {"value": float(np.sum(w[good] * values[good])),
            "points": int(x.size),
            "used": int(np.sum(good)),
            "step": h,
            "tail_cut_at": limit,
            "smallest_distance_to_an_endpoint": float(np.min(np.minimum(d_lo, d_hi))),
            "nodes": x, "weights": w,
            "distance_to_lo": d_lo, "distance_to_hi": d_hi}


def tanh_sinh_convergence(f, exact, lo: float, hi: float, levels=None,
                          use_distances: bool = False) -> dict:
    """Error against level, and whether it falls geometrically in the point count."""
    ks = ([1, 2, 3, 4, 5, 6, 7] if levels is None else [int(v) for v in np.atleast_1d(levels)])
    want = float(exact)
    rows = []
    for k in ks:
        counter = [0]
        out = tanh_sinh(f, lo, hi, k, counter, use_distances)
        rows.append((k, out["points"], abs(out["value"] - want)))
    e = np.asarray([r[2] for r in rows], dtype=float)
    n = np.asarray([r[1] for r in rows], dtype=float)
    keep = e > 1e-15 * max(abs(want), 1.0)
    if int(np.sum(keep)) < 3:
        return {"levels": np.asarray(ks), "points": n, "errors": e,
                "geometric_wins": False, "fitted_power": float("nan"),
                "fitted_rate": float("nan"), "points_used": int(np.sum(keep)),
                "note": "not enough points above the roundoff floor to tell the laws apart"}
    y = np.log(e[keep])
    p_fit, p_res = np.polyfit(np.log(n[keep]), y, 1, full=True)[:2]
    g_fit, g_res = np.polyfit(n[keep], y, 1, full=True)[:2]
    pr = float(p_res[0]) if len(p_res) else 0.0
    gr = float(g_res[0]) if len(g_res) else 0.0
    return {"levels": np.asarray(ks), "points": n, "errors": e,
            "geometric_wins": bool(gr < pr), "fitted_power": float(-p_fit[0]),
            "fitted_rate": float(-g_fit[0]), "power_residual": pr,
            "geometric_residual": gr, "points_used": int(np.sum(keep)),
            "note": "residual of log(error) against log(points) versus against points"}


def handles_a_singularity_without_being_told(exact=None, budgets=None) -> dict:
    """tanh-sinh against Gauss-Legendre on an integrand that is infinite at both ends.

    ``1 / sqrt(x (1 - x))`` over ``(0, 1)`` integrates to exactly ``pi``. Gauss-Legendre never
    evaluates at the endpoints either, so it produces a number, but it converges as a slow power
    because the integrand has no bounded derivatives there. The double exponential rule needs no
    warning about the singularity and no knowledge of its strength.

    Both spellings of the integrand are run: the plain one in terms of ``x``, which stalls at
    1.6e-08 because ``1 - x`` cancels, and the one given the endpoint distances, which does not.
    The difference between those two columns is the whole reason `tanh_sinh` returns distances.
    """
    from . import gaussquad as _gq

    want = math.pi if exact is None else float(exact)

    def f(x):
        t = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / np.sqrt(t * (1.0 - t))

    def f_stable(x, d_lo, d_hi):
        return 1.0 / np.sqrt(d_lo * d_hi)

    bs = ([1, 2, 3, 4, 5, 6] if budgets is None else [int(v) for v in np.atleast_1d(budgets)])
    rows = []
    for k in bs:
        counter = [0]
        de = tanh_sinh(f, 0.0, 1.0, k, counter)
        n = counter[0]
        stable = tanh_sinh(f_stable, 0.0, 1.0, k, None, True)
        gauss = abs(_gq.integrate(f, 0.0, 1.0, n) - want)
        rows.append((k, n, abs(de["value"] - want), abs(stable["value"] - want), gauss))
    return {"level": np.asarray([r[0] for r in rows]),
            "evaluations": np.asarray([r[1] for r in rows]),
            "double_exponential_error": np.asarray([r[2] for r in rows]),
            "distance_aware_error": np.asarray([r[3] for r in rows]),
            "gauss_legendre_error": np.asarray([r[4] for r in rows]),
            "double_exponential_wins": np.asarray([r[2] < r[4] for r in rows]),
            "distances_help": np.asarray([r[3] < r[2] for r in rows])}


# --------------------------------------------------------------------------- tensor products


def tensor_rule(rule_1d, counts, boxes) -> tuple:
    """The tensor product of a one dimensional rule over any number of axes.

    ``rule_1d(n, lo, hi)`` returns nodes and weights on one axis. ``counts`` gives the node count
    per axis and ``boxes`` the interval per axis, so the dimension is whatever those two agree
    on and nothing here is written for two dimensions in particular.

    Returns an array of points, one row per point, and the matching weights.
    """
    ns = [int(v) for v in np.atleast_1d(counts)]
    limits = np.asarray(boxes, dtype=float).reshape(-1, 2)
    if len(ns) == 1 and limits.shape[0] > 1:
        ns = ns * limits.shape[0]
    if len(ns) != limits.shape[0]:
        raise ValueError(f"got {len(ns)} node counts for {limits.shape[0]} axes")
    axes = [rule_1d(ns[i], float(limits[i, 0]), float(limits[i, 1]))
            for i in range(limits.shape[0])]
    grids = np.meshgrid(*[a[0] for a in axes], indexing="ij")
    weight_grids = np.meshgrid(*[a[1] for a in axes], indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)
    weights = np.ones(points.shape[0])
    for g in weight_grids:
        weights = weights * g.ravel()
    return points, weights


def gauss_rule_1d(n: int, lo: float, hi: float) -> tuple:
    """Gauss-Legendre on one axis, in the shape `tensor_rule` wants."""
    from . import gaussquad as _gq

    return _gq.rule_on(n, lo, hi)


def simpson_rule_1d(n: int, lo: float, hi: float) -> tuple:
    """Composite Simpson on one axis, as nodes and weights, in the same shape.

    ``n`` is the total node count and must be odd, because composite Simpson pairs up panels.
    """
    from . import newtoncotes as _nc

    m = int(n)
    if m < 3 or m % 2 == 0:
        raise ValueError(f"composite Simpson needs an odd node count of at least 3, got {m}")
    panels = (m - 1) // 2
    a = float(lo)
    b = float(hi)
    x = np.linspace(a, b, m)
    base = _nc.weights(3, True) * (b - a) / panels
    w = np.zeros(m)
    for p in range(panels):
        w[2 * p:2 * p + 3] += base
    return x, w


def integrate_box(f, boxes, counts, rule_1d=None, _counter=None) -> float:
    """Integrate over an axis aligned box in any dimension, by tensor product.

    ``f`` takes an array of points, one row per point, and returns one value per row.
    """
    points, weights = tensor_rule(gauss_rule_1d if rule_1d is None else rule_1d, counts, boxes)
    if _counter is not None:
        _counter[0] += points.shape[0]
    return float(np.sum(weights * np.asarray(f(points), dtype=float)))


def iterated(f, lo_x: float, hi_x: float, lo_y, hi_y, n: int = 8, _counter=None) -> float:
    """A double integral with limits in ``y`` that depend on ``x``, done as nested 1D rules.

    ``lo_y`` and ``hi_y`` are callables of ``x``, so the region need not be a rectangle. Each
    outer node runs a fresh inner rule, which is what makes non-rectangular domains work and
    also what makes them cost ``n`` times as much bookkeeping.
    """
    from . import gaussquad as _gq

    xs, wx = _gq.rule_on(n, lo_x, hi_x)
    total = 0.0
    for xi, wi in zip(xs, wx):
        a = float(lo_y(xi))
        b = float(hi_y(xi))
        if not b > a:
            continue
        ys, wy = _gq.rule_on(n, a, b)
        if _counter is not None:
            _counter[0] += ys.size
        points = np.stack([np.full(ys.size, float(xi)), ys], axis=1)
        total += wi * float(np.sum(wy * np.asarray(f(points), dtype=float)))
    return float(total)


def tensor_convergence(f, exact, boxes, counts=None, rule_1d=None) -> dict:
    """Error against nodes per axis, and the fitted order in the total evaluation count.

    Two orders are reported and they are the point of the exercise. ``order_per_axis`` is the
    familiar one, ``p``. ``order_per_evaluation`` is ``p/d``, which is what actually governs how
    much work a given accuracy costs, and it is the one that collapses as the dimension grows.
    """
    limits = np.asarray(boxes, dtype=float).reshape(-1, 2)
    d = limits.shape[0]
    ns = ([2, 3, 4, 6, 8, 11] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    want = float(exact)
    rows = []
    for n in ns:
        counter = [0]
        value = integrate_box(f, boxes, [n] * d, rule_1d, counter)
        rows.append((n, counter[0], abs(value - want)))
    e = np.asarray([r[2] for r in rows], dtype=float)
    per_axis = np.asarray([r[0] for r in rows], dtype=float)
    total = np.asarray([r[1] for r in rows], dtype=float)
    keep = e > 1e-13 * max(abs(want), 1.0)
    if int(np.sum(keep)) < 3:
        return {"dimension": d, "nodes_per_axis": per_axis, "evaluations": total, "errors": e,
                "order_per_axis": float("nan"), "order_per_evaluation": float("nan"),
                "points_used": int(np.sum(keep)),
                "note": "not enough points above the roundoff floor to fit an order"}
    return {"dimension": d, "nodes_per_axis": per_axis, "evaluations": total, "errors": e,
            "order_per_axis": float(-np.polyfit(np.log(per_axis[keep]),
                                                np.log(e[keep]), 1)[0]),
            "order_per_evaluation": float(-np.polyfit(np.log(total[keep]),
                                                      np.log(e[keep]), 1)[0]),
            "points_used": int(np.sum(keep)), "note": "the second order is the first over d"}


#: Test integrands over the unit cube in any dimension, with their exact values.
#: ``smooth`` is a product of cosines, analytic and separable, which is the case most
#: favourable to a tensor rule. ``rough`` is a product of ``|x - 1/2|^(1/2)``, which has a kink
#: with unbounded derivative on every axis and is the ordinary case.
CUBE_INTEGRANDS = {
    "smooth": (lambda p: np.prod(np.cos(p), axis=1), math.sin(1.0)),
    "rough": (lambda p: np.prod(np.sqrt(np.abs(p - 0.5)), axis=1),
              2.0 * 0.5 ** 1.5 / 1.5),
}


def cube_integrand(kind: str = "rough", dimension: int = 1) -> tuple:
    """One of `CUBE_INTEGRANDS`, with its exact integral over the unit cube in ``dimension``.

    Both are products over the axes, so the exact answer is the one dimensional value raised to
    the power ``dimension``, and nothing about the problem changes with ``d`` except ``d``.
    """
    key = str(kind).lower()
    if key not in CUBE_INTEGRANDS:
        raise ValueError(f"unknown integrand {kind!r}, expected one of {sorted(CUBE_INTEGRANDS)}")
    f, one_axis = CUBE_INTEGRANDS[key]
    return f, one_axis ** int(dimension)


def curse_of_dimensionality(dimensions=None, counts=None, kind: str = "smooth") -> dict:
    """The order per evaluation, measured as the dimension grows, on the same integrand.

    A **fixed order** rule is used, composite Simpson, and that choice is the point. Gauss on an
    analytic integrand converges geometrically, so fitting an algebraic order to it returns a
    meaningless large number, 21.67 on this integrand, and the demonstration says nothing.
    Simpson has order 4 whatever it is given, so the table shows exactly what it should:

        d                     1      2      3      4
        order per axis      4.79   4.79   4.79   4.79
        order per evaluation 4.79   2.40   1.60   1.20

    The first row does not move. The second is the first divided by ``d``, and it is the one that
    decides how much work a given accuracy costs. In four dimensions a fourth order rule behaves
    like a first order one.
    """
    ds = ([1, 2, 3, 4] if dimensions is None else [int(v) for v in np.atleast_1d(dimensions)])
    ns = ([3, 5, 9, 17] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for d in ds:
        f, exact = cube_integrand(kind, d)
        out = tensor_convergence(f, exact, [(0.0, 1.0)] * d, ns, simpson_rule_1d)
        rows.append((d, out["order_per_axis"], out["order_per_evaluation"],
                     float(out["evaluations"][-1]), float(out["errors"][-1])))
    return {"dimension": np.asarray([r[0] for r in rows]),
            "order_per_axis": np.asarray([r[1] for r in rows]),
            "order_per_evaluation": np.asarray([r[2] for r in rows]),
            "evaluations_at_the_finest": np.asarray([r[3] for r in rows]),
            "error_at_the_finest": np.asarray([r[4] for r in rows]),
            "integrand": str(kind).lower()}


# --------------------------------------------------------------------------- Monte Carlo


def monte_carlo(f, boxes, samples: int, seed: int = 42) -> dict:
    """Plain Monte Carlo on a box, with the standard error the samples themselves report.

    The estimate is the mean of ``f`` times the volume, and the reported error is the sample
    standard deviation over ``sqrt(N)`` times the volume. That estimate is itself random, so it
    is a one standard deviation figure and not a bound.
    """
    limits = np.asarray(boxes, dtype=float).reshape(-1, 2)
    d = limits.shape[0]
    n = int(samples)
    if n < 2:
        raise ValueError(f"need at least two samples, got {n}")
    rng = np.random.default_rng(seed)
    points = limits[:, 0] + (limits[:, 1] - limits[:, 0]) * rng.random((n, d))
    values = np.asarray(f(points), dtype=float)
    volume = float(np.prod(limits[:, 1] - limits[:, 0]))
    mean = float(np.mean(values))
    return {"value": volume * mean,
            "standard_error": volume * float(np.std(values, ddof=1)) / math.sqrt(n),
            "samples": n, "dimension": d, "volume": volume}


def monte_carlo_rate(f, exact, boxes, sample_counts=None, repeats: int = 12) -> dict:
    """The Monte Carlo error against sample count, averaged over independent runs.

    A single run is too noisy to fit a rate to, so each sample count is repeated and the root
    mean square error is used. The fitted exponent should come out near 1/2 in every dimension,
    which is the entire reason the method exists.
    """
    ns = ([100, 400, 1600, 6400, 25600] if sample_counts is None
          else [int(v) for v in np.atleast_1d(sample_counts)])
    want = float(exact)
    rows = []
    for n in ns:
        errors = [abs(monte_carlo(f, boxes, n, seed=1000 + r)["value"] - want)
                  for r in range(int(repeats))]
        rows.append((n, float(np.sqrt(np.mean(np.asarray(errors) ** 2)))))
    n_arr = np.asarray([r[0] for r in rows], dtype=float)
    e_arr = np.asarray([r[1] for r in rows], dtype=float)
    return {"samples": n_arr, "rms_error": e_arr,
            "fitted_rate": float(-np.polyfit(np.log(n_arr), np.log(e_arr), 1)[0]),
            "repeats": int(repeats)}


def monte_carlo_crossover(dimensions=None, budget: int = 4096, repeats: int = 8,
                          kind: str = "rough") -> dict:
    """Where Monte Carlo overtakes a tensor Gauss rule, measured on the same integrand.

    Both methods get the same evaluation budget, as nearly as the tensor grid allows.

    **The answer depends entirely on the integrand, and picking the flattering one is how this
    comparison gets faked.** On the smooth product of cosines, which is analytic and separable,
    the tensor rule is essentially exact at every dimension up to 10 and Monte Carlo never
    catches it. On the rough integrand, which merely has a kink on each axis, the crossing is at
    **three dimensions** and by twelve the gap is a factor of 150 the other way:

        d                 1        2        3        6       12
        tensor       9.1e-07  4.4e-04  2.4e-03  4.1e-03  4.6e-04
        Monte Carlo  2.5e-03  2.7e-03  1.2e-03  1.7e-04  3.0e-06

    So the honest statement is not "Monte Carlo wins above dimension ``k``". It is that the
    tensor rule's rate is the integrand's smoothness divided by ``d``, Monte Carlo's is 1/2 no
    matter what, and which is larger depends on both numbers.
    """
    ds = ([1, 2, 3, 4, 5, 6, 8, 10, 12] if dimensions is None
          else [int(v) for v in np.atleast_1d(dimensions)])
    rows = []
    for d in ds:
        f, exact = cube_integrand(kind, d)
        per_axis = max(2, int(round(float(budget) ** (1.0 / d))))
        counter = [0]
        tensor = abs(integrate_box(f, [(0.0, 1.0)] * d, [per_axis] * d, None, counter) - exact)
        spent = counter[0]
        errors = [abs(monte_carlo(f, [(0.0, 1.0)] * d, spent, seed=2000 + r)["value"] - exact)
                  for r in range(int(repeats))]
        mc = float(np.sqrt(np.mean(np.asarray(errors) ** 2)))
        rows.append((d, per_axis, spent, tensor, mc, bool(mc < tensor)))
    wins = [r[0] for r in rows if r[5]]
    return {"dimension": np.asarray([r[0] for r in rows]),
            "nodes_per_axis": np.asarray([r[1] for r in rows]),
            "evaluations": np.asarray([r[2] for r in rows]),
            "tensor_error": np.asarray([r[3] for r in rows]),
            "monte_carlo_rms_error": np.asarray([r[4] for r in rows]),
            "monte_carlo_wins": np.asarray([r[5] for r in rows]),
            "first_dimension_monte_carlo_wins": (wins[0] if wins else None),
            "budget": int(budget), "integrand": str(kind).lower()}
