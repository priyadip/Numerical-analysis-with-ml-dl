"""Polynomial interpolation: the same polynomial, written five different ways.

What this module is for
-----------------------
Given ``n + 1`` points with distinct abscissas there is **exactly one** polynomial of degree at
most ``n`` through them. That is a theorem, and `uniqueness_report` checks it. What the theorem
does not say is how to write that polynomial down, and the choice matters enormously in floating
point.

The five forms
--------------
**Power form** ``sum a_k x^k``. The obvious one, and the worst. Finding the coefficients means
solving a Vandermonde system, whose condition number grows exponentially in the number of points:
`vandermonde_condition` measures it reaching 1e16 by about 25 equally spaced points, at which
point the coefficients carry no correct digits at all.

**Shifted power form** ``sum a_k (x - c)^k``. The same polynomial expanded about ``c`` instead of
about 0. Choosing ``c`` near the data reduces the cancellation dramatically, and it costs nothing.

**Newton form** ``a_0 + a_1 (x - x_0) + a_2 (x - x_0)(x - x_1) + ...``. Its coefficients are the
divided differences of lesson 45, its evaluation is a nested loop, and **adding a new point costs
one extra term and leaves every existing coefficient unchanged**. No other form has that.

**Nested Newton form**, which is the Newton form arranged for Horner-style evaluation, so
evaluating costs ``n`` multiplications instead of ``n(n+1)/2``.

**Lagrange form** ``sum y_i L_i(x)``. The coefficients are the data values themselves, so there is
nothing to solve. Written naively it costs ``O(n^2)`` per evaluation and is numerically poor. The
**barycentric** rearrangement fixes both: ``O(n)`` per evaluation after an ``O(n^2)`` setup, and
it is backward stable, which the naive form is not.

Which to use
------------
Barycentric Lagrange for evaluation at many points, Newton when points arrive one at a time, and
the power form essentially never. `compare_forms` measures all of them against each other.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _check_nodes(x):
    """Distinct abscissas are the one hypothesis every form needs."""
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if x.size == 0:
        raise ValueError("need at least one node")
    if np.unique(x).size != x.size:
        raise ValueError("the nodes must be distinct; interpolation is not defined otherwise")
    return x


def _check_pair(x, y):
    x = _check_nodes(x)
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if y.size != x.size:
        raise ValueError(f"got {x.size} nodes and {y.size} values, they must match")
    return x, y


# ----------------------------------------------------------------------------- power form


def vandermonde(x, degree: int | None = None) -> np.ndarray:
    """The matrix ``V[i, j] = x_i^j``, whose solution gives the power form coefficients.

    Building it is the textbook route and computing with it is the mistake. The columns
    ``1, x, x^2, ...`` become nearly parallel as the degree grows, because on a fixed interval
    every high power looks like every other, so ``V`` is ill conditioned by construction and not
    by accident. `vandermonde_condition` measures the growth.
    """
    x = _check_nodes(x)
    d = x.size - 1 if degree is None else int(degree)
    if d < 0:
        raise ValueError(f"degree must be non-negative, got {d}")
    return np.vander(x, d + 1, increasing=True)


def power_form(x, y) -> np.ndarray:
    """Coefficients ``a`` with ``p(t) = sum_k a_k t^k``, by solving the Vandermonde system.

    This fails outright on data sitting far from the origin, and the failure is the point rather
    than a defect: on 12 nodes over ``[999, 1001]`` the Vandermonde matrix is numerically
    singular, because every column ``x^k`` is nearly a multiple of every other when ``x`` barely
    varies in relative terms. `shifted_power_form` on the same data has condition number 4e4.
    The error below says so instead of letting the caller read a bare "singular matrix".
    """
    x, y = _check_pair(x, y)
    V = vandermonde(x)
    try:
        return np.linalg.solve(V, y)
    except np.linalg.LinAlgError as exc:
        kappa = float(np.linalg.cond(V))
        centred = float(np.linalg.cond(vandermonde(x - np.mean(x))))
        raise np.linalg.LinAlgError(
            f"the Vandermonde matrix on these {x.size} nodes is numerically singular, "
            f"kappa = {kappa:.2e}. Centring the nodes on their mean gives kappa = "
            f"{centred:.2e}, so use shifted_power_form, or better a Newton or barycentric "
            f"form, which never build this matrix at all") from exc


def shifted_power_form(x, y, centre: float | None = None):
    """Coefficients about ``centre``, so ``p(t) = sum_k a_k (t - c)^k``.

    The same polynomial, and a far better conditioned system, because shifting the data to
    straddle zero stops the high powers from all having the same sign and similar size. The
    default centre is the mean of the nodes, which is the usual good choice.
    """
    x, y = _check_pair(x, y)
    c = float(np.mean(x)) if centre is None else float(centre)
    return np.linalg.solve(vandermonde(x - c), y), c


def normalised_power_form(x, y):
    """Coefficients in ``s = (t - c) / h``, where ``c`` and ``h`` map the nodes onto ``[-1, 1]``.

    **Shifting alone is not enough.** The shifted form fixes an interval sitting far from the
    origin and does nothing for one that is very small or very large, because the trouble there
    is the scale rather than the offset. Measured ``kappa(V)`` at 13 nodes:

    ==========================  ==========  ==========  ================
    interval                    raw         centred     centred + scaled
    ==========================  ==========  ==========  ================
    ``[-1, 1]``                 1.23e5      1.23e5      1.23e5
    ``[0, 1]``                  6.78e9      1.31e8      1.23e5
    ``[-5, 3]``                 2.04e9      1.52e8      1.23e5
    ``[99, 101]``               3.07e37     1.23e5      1.23e5
    ``[-0.001, 0.001]``         2.69e40     2.69e40     1.23e5
    ``[1e6, 1e6 + 1]``          1.78e88     1.31e8      1.23e5
    ==========================  ==========  ==========  ================

    Normalising gives **the same 1.23e5 in every row**, so the conditioning becomes a property of
    the degree alone, which is the intrinsic part, rather than of where the data happens to live.
    That residual 1.23e5 is the part no change of variable can remove.

    Returns ``(coefficients, centre, half_width)``.
    """
    x, y = _check_pair(x, y)
    c = 0.5 * (float(np.max(x)) + float(np.min(x)))
    h = 0.5 * (float(np.max(x)) - float(np.min(x)))
    if h == 0.0:
        h = 1.0
    return np.linalg.solve(vandermonde((x - c) / h), y), c, h


def evaluate_normalised(coeffs, t, centre: float, half_width: float) -> np.ndarray:
    """Evaluate the normalised power form at ``t``."""
    h = float(half_width)
    if h == 0.0:
        h = 1.0
    return evaluate_power(coeffs, (np.asarray(t, dtype=float) - float(centre)) / h)


def evaluate_power(coeffs, t, centre: float = 0.0) -> np.ndarray:
    """Horner evaluation of ``sum a_k (t - c)^k``, ascending coefficients."""
    a = np.atleast_1d(np.asarray(coeffs, dtype=float)).ravel()
    z = np.asarray(t, dtype=float) - float(centre)
    out = np.zeros_like(z, dtype=float) + a[-1]
    for k in range(a.size - 2, -1, -1):
        out = out * z + a[k]
    return out


def vandermonde_condition(x) -> float:
    """``kappa_2`` of the Vandermonde matrix on the given nodes."""
    return float(np.linalg.cond(vandermonde(x)))


# ----------------------------------------------------------------------------- Newton form


def newton_coefficients(x, y) -> np.ndarray:
    """The divided difference coefficients of the Newton form, in place, in ``O(n^2)``.

    ``c[k]`` is ``f[x_0, ..., x_k]``. Lesson 45 proves the recursion; this is the working code
    that every later formula in Part 7 rests on.
    """
    x, y = _check_pair(x, y)
    c = y.copy()
    n = c.size
    for j in range(1, n):
        c[j:] = (c[j:] - c[j - 1:-1]) / (x[j:] - x[:n - j])
    return c


def evaluate_newton(coeffs, nodes, t) -> np.ndarray:
    """Nested (Horner-style) evaluation of the Newton form: ``n`` multiplications, not ``n^2``.

    ``p(t) = c_0 + (t - x_0)(c_1 + (t - x_1)(c_2 + ...))``, built from the inside out.

    **Repeated nodes are allowed here**, unlike in `newton_coefficients`. Computing the
    coefficients needs distinct abscissas because the recursion divides by their differences;
    evaluating the nested product does not, and lesson 50's Hermite form is exactly a Newton form
    with every node repeated, so rejecting repeats here would rule out its own evaluator.
    """
    c = np.atleast_1d(np.asarray(coeffs, dtype=float)).ravel()
    x = np.atleast_1d(np.asarray(nodes, dtype=float)).ravel()
    if x.size == 0:
        raise ValueError("need at least one node")
    if c.size > x.size:
        raise ValueError(f"{c.size} coefficients need at least that many nodes, got {x.size}")
    z = np.asarray(t, dtype=float)
    out = np.zeros_like(z, dtype=float) + c[-1]
    for k in range(c.size - 2, -1, -1):
        out = out * (z - x[k]) + c[k]
    return out


def newton_add_point(coeffs, nodes, x_new: float, y_new: float):
    """Extend a Newton interpolant by one point. **Every existing coefficient is unchanged.**

    That is the property no other form has, and it is why the Newton form is what an adaptive
    method carries. The cost is ``O(n)`` for the new coefficient against ``O(n^2)`` to rebuild.
    """
    c = np.atleast_1d(np.asarray(coeffs, dtype=float)).ravel()
    x = _check_nodes(nodes)
    x_new = float(x_new)
    if np.any(np.isclose(x, x_new, rtol=0.0, atol=0.0)):
        raise ValueError(f"{x_new} is already a node; the nodes must stay distinct")
    residual = float(y_new) - float(evaluate_newton(c, x, x_new))
    product = float(np.prod(x_new - x[:c.size]))
    if product == 0.0:
        raise ValueError("the node product vanished, which means a repeated node")
    return np.concatenate([c, [residual / product]]), np.concatenate([x, [x_new]])


def change_of_centre(coeffs, nodes, new_nodes):
    """Rewrite a Newton interpolant about a different ordering of the same nodes.

    The Newton form depends on the ORDER of the nodes, unlike the polynomial it represents. This
    re-derives the coefficients for a new order by evaluating the same polynomial at the new
    nodes, which is the honest way: the polynomial is what is preserved, not the coefficients.
    """
    c = np.atleast_1d(np.asarray(coeffs, dtype=float)).ravel()
    x = _check_nodes(nodes)
    z = _check_nodes(new_nodes)
    if z.size != c.size:
        raise ValueError(f"need {c.size} new nodes to carry {c.size} coefficients, got {z.size}")
    return newton_coefficients(z, evaluate_newton(c, x, z))


# ----------------------------------------------------------------------------- Lagrange form


def lagrange_basis(nodes, t) -> np.ndarray:
    """The basis polynomials ``L_i(t)``, one row per node.

    ``L_i`` is 1 at ``x_i`` and 0 at every other node, so ``sum_i y_i L_i`` interpolates with no
    system to solve at all. That is the appeal, and the ``O(n^2)`` cost per evaluation point is
    the price. `barycentric_weights` removes the price.
    """
    x = _check_nodes(nodes)
    z = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.ones((x.size, z.size))
    for i in range(x.size):
        for j in range(x.size):
            if j != i:
                out[i] *= (z - x[j]) / (x[i] - x[j])
    return out


def evaluate_lagrange(nodes, values, t) -> np.ndarray:
    """The naive Lagrange evaluation, written out so it can be measured against barycentric."""
    x, y = _check_pair(nodes, values)
    z = np.atleast_1d(np.asarray(t, dtype=float))
    out = (y[:, None] * lagrange_basis(x, z)).sum(axis=0)
    return out.reshape(np.shape(t)) if np.ndim(t) else float(out[0])


def barycentric_weights(nodes) -> np.ndarray:
    """``w_i = 1 / prod_{j != i} (x_i - x_j)``, computed with the products scaled to avoid
    overflow.

    These depend only on the nodes, so for repeated interpolation on a fixed grid they are
    computed once and every later evaluation costs ``O(n)``.
    """
    x = _check_nodes(nodes)
    n = x.size
    if n == 1:
        return np.ones_like(x)          # sized from the input, as everything here is
    # scale the node spacing to about 1 so the product neither overflows nor underflows,
    # then undo the scaling, which is exact because it is a power of a single factor
    spread = float(np.max(x) - np.min(x))
    scale = 4.0 / spread if spread > 0 else 1.0
    xs = x * scale
    w = np.ones(n)
    for i in range(n):
        diff = xs[i] - xs
        diff[i] = 1.0
        w[i] = 1.0 / np.prod(diff)
    return w * scale ** (n - 1)


def evaluate_barycentric(nodes, values, t, weights=None) -> np.ndarray:
    """The second (true) barycentric formula: ``O(n)`` per point and backward stable.

    ``p(t) = sum_i w_i y_i/(t - x_i)  /  sum_i w_i/(t - x_i)``.

    Higham proved this form backward stable, which the naive Lagrange form is not. The exact
    node case is handled separately, since the formula divides by zero there and the answer is
    simply the data value.
    """
    x, y = _check_pair(nodes, values)
    w = barycentric_weights(x) if weights is None else np.asarray(weights, dtype=float)
    z = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty(z.shape, dtype=float)
    diff = z[:, None] - x[None, :]
    exact = np.isclose(diff, 0.0, rtol=0.0, atol=0.0)
    hit = exact.any(axis=1)
    if hit.any():
        out[hit] = y[np.argmax(exact[hit], axis=1)]
    if (~hit).any():
        d = diff[~hit]
        num = ((w * y)[None, :] / d).sum(axis=1)
        den = (w[None, :] / d).sum(axis=1)
        # The denominator can cancel to exactly zero when the weights span a huge range with
        # alternating signs, which is the equally spaced high degree case. Measured on 64
        # equally spaced nodes over [-1, 1], the weights run from 1.2e7 to 1.1e25 and the
        # denominator reaches exact zero at some probe points. The result there is genuinely
        # meaningless rather than infinite, so it is reported as nan and the numpy warning is
        # suppressed, since the caller is told by the value itself. Chebyshev nodes at the same
        # degree have weights spanning a factor of 41, and this never happens.
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = num / den
        vals[den == 0.0] = np.nan
        out[~hit] = vals
    return out.reshape(np.shape(t)) if np.ndim(t) else float(out[0])


# ----------------------------------------------------------------------------- checking


@dataclass
class FormComparison:
    """One measurement of the five forms against each other on the same data."""

    n_nodes: int
    vandermonde_kappa: float
    errors: dict = field(default_factory=dict)
    node_residuals: dict = field(default_factory=dict)


def uniqueness_report(x, y, n_trials: int = 5, rng=None) -> dict:
    """Check that the interpolating polynomial is unique, by building it several ways.

    Every form below represents the SAME polynomial, so they must agree at every point. They do,
    to a tolerance that depends on the conditioning, and that dependence is the lesson.
    """
    x, y = _check_pair(x, y)
    gen = np.random.default_rng() if rng is None else rng
    lo, hi = float(np.min(x)), float(np.max(x))
    probe = np.sort(gen.uniform(lo, hi, max(int(n_trials), 1)))
    a_pow = power_form(x, y)
    a_sh, c = shifted_power_form(x, y)
    c_new = newton_coefficients(x, y)
    w = barycentric_weights(x)
    values = {
        "power": evaluate_power(a_pow, probe),
        "shifted power": evaluate_power(a_sh, probe, centre=c),
        "newton": evaluate_newton(c_new, x, probe),
        "lagrange": evaluate_lagrange(x, y, probe),
        "barycentric": evaluate_barycentric(x, y, probe, weights=w),
    }
    ref = values["barycentric"]
    scale = max(float(np.max(np.abs(ref))), 1.0)
    return {"probe": probe, "values": values,
            "max_disagreement": {k: float(np.max(np.abs(v - ref)) / scale)
                                 for k, v in values.items()},
            "vandermonde_kappa": vandermonde_condition(x)}


def compare_forms(f, n_nodes: int, lo: float = -1.0, hi: float = 1.0,
                  chebyshev: bool = False, n_probe: int = 401) -> FormComparison:
    """Build the interpolant of ``f`` five ways and measure each against ``f`` itself.

    ``chebyshev=True`` uses Chebyshev nodes instead of equally spaced ones, which is lesson 47's
    subject and is offered here so the conditioning claim can be separated from the node claim.
    """
    n = int(n_nodes)
    if n < 1:
        raise ValueError(f"need at least one node, got {n}")
    if chebyshev:
        k = np.arange(n)
        x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos((2 * k + 1) * np.pi / (2 * n))
        x = np.sort(x)
    else:
        x = np.linspace(lo, hi, n)
    y = np.asarray([f(v) for v in x], dtype=float)
    probe = np.linspace(lo, hi, int(n_probe))
    truth = np.asarray([f(v) for v in probe], dtype=float)
    scale = max(float(np.max(np.abs(truth))), 1e-300)

    a_pow = power_form(x, y)
    a_sh, c = shifted_power_form(x, y)
    c_new = newton_coefficients(x, y)
    w = barycentric_weights(x)
    built = {
        "power": evaluate_power(a_pow, probe),
        "shifted power": evaluate_power(a_sh, probe, centre=c),
        "newton": evaluate_newton(c_new, x, probe),
        "lagrange": evaluate_lagrange(x, y, probe),
        "barycentric": evaluate_barycentric(x, y, probe, weights=w),
    }
    out = FormComparison(n_nodes=n, vandermonde_kappa=vandermonde_condition(x))
    for name, vals in built.items():
        out.errors[name] = float(np.max(np.abs(vals - truth)) / scale)
    at_nodes = {
        "power": evaluate_power(a_pow, x),
        "shifted power": evaluate_power(a_sh, x, centre=c),
        "newton": evaluate_newton(c_new, x, x),
        "lagrange": evaluate_lagrange(x, y, x),
        "barycentric": evaluate_barycentric(x, y, x, weights=w),
    }
    yscale = max(float(np.max(np.abs(y))), 1e-300)
    for name, vals in at_nodes.items():
        out.node_residuals[name] = float(np.max(np.abs(vals - y)) / yscale)
    return out


def interpolation_cost(n_nodes: int) -> dict:
    """Operation counts per evaluation point, once the setup is done.

    The setup costs differ too: the power form needs an ``O(n^3)`` solve, Newton an ``O(n^2)``
    difference table, and barycentric an ``O(n^2)`` weight computation. Lagrange needs none,
    which is exactly why it looks attractive until the per-point cost is counted.
    """
    n = int(n_nodes)
    return {"power_horner": n,
            "newton_nested": 2 * n,
            "lagrange_naive": 2 * n * n,
            "barycentric": 5 * n,
            "setup_power": n ** 3 // 3,
            "setup_newton": n * (n - 1) // 2,
            "setup_barycentric": n * (n - 1)}
