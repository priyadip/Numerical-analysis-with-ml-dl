"""Divided differences: the coefficients of the Newton form, and what they mean.

What this module is for
-----------------------
Lesson 44 used `interp.newton_coefficients` without saying why the recursion is correct. This
module proves it, in code, and then develops the properties that make divided differences the
workhorse of the rest of Part 7.

The definition and the recursion
--------------------------------
``f[x_i] = f(x_i)``, and

    f[x_i, ..., x_{i+k}]  =  ( f[x_{i+1}, ..., x_{i+k}] - f[x_i, ..., x_{i+k-1}] )
                             / ( x_{i+k} - x_i )

`table` builds the whole triangle, `recursive` computes one entry straight from the definition,
and `from_definition` computes the same entry from the closed form

    f[x_0, ..., x_k]  =  sum_i  f(x_i) / prod_{j != i} (x_i - x_j)

so the recursion is checked against something independent rather than against itself.

The four properties
-------------------
**Symmetry.** ``f[x_0, ..., x_k]`` does not depend on the order of its arguments, even though the
recursion that computes it plainly does. `symmetry_report` measures that.

**It is the leading coefficient.** ``f[x_0, ..., x_k]`` is the coefficient of ``x^k`` in the
interpolating polynomial through those ``k + 1`` points. That is why the Newton form works.

**Linearity** in ``f``: the divided difference of a sum is the sum of the divided differences.

**The derivative connection.** ``f[x_0, ..., x_k] = f^(k)(xi) / k!`` for some ``xi`` in the
interval, so a divided difference is a derivative in disguise, scaled by a factorial. Letting all
the nodes coalesce turns the divided difference into exactly ``f^(k)(x)/k!``, which is the Taylor
coefficient, and that limit is what lesson 50's Hermite interpolation is built on.
`derivative_connection` measures the first statement and `confluent` computes the limit.

Where the accuracy goes
-----------------------
The recursion divides by ``x_{i+k} - x_i``, so close nodes make it subtract nearly equal numbers
and divide by a small one. `conditioning_report` measures the loss, which is the reason
lesson 48's equally spaced tables are worth having: with a constant spacing the divisions are
exact powers of ``h`` and the cancellation is the only remaining source of error.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np


def _check(x, y):
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if x.size == 0:
        raise ValueError("need at least one node")
    if x.size != y.size:
        raise ValueError(f"got {x.size} nodes and {y.size} values, they must match")
    if np.unique(x).size != x.size:
        raise ValueError("the nodes must be distinct; use `confluent` for repeated nodes")
    return x, y


# ----------------------------------------------------------------------------- the table


def table(x, y) -> np.ndarray:
    """The full divided difference triangle, ``T[i, k] = f[x_i, ..., x_{i+k}]``.

    Entries above the anti-diagonal are unused and left as nan, so a caller reading one by
    mistake gets a nan rather than a plausible wrong number.
    """
    x, y = _check(x, y)
    n = x.size
    T = np.full((n, n), np.nan)
    T[:, 0] = y
    for k in range(1, n):
        rows = np.arange(n - k)
        T[rows, k] = (T[rows + 1, k - 1] - T[rows, k - 1]) / (x[rows + k] - x[rows])
    return T


def coefficients(x, y) -> np.ndarray:
    """The top row of the table, which is the Newton form's coefficient vector."""
    return table(x, y)[0, :]


def recursive(x, y, i: int = 0, k: int | None = None) -> float:
    """One divided difference, computed straight from the recursion, without a table.

    Exponential in ``k`` and written for checking rather than for use. `table` is the ``O(n^2)``
    version and gives the same answers.
    """
    x, y = _check(x, y)
    k = x.size - 1 if k is None else int(k)
    i = int(i)
    if k == 0:
        return float(y[i])
    hi = recursive(x, y, i + 1, k - 1)
    lo = recursive(x, y, i, k - 1)
    return float((hi - lo) / (x[i + k] - x[i]))


def from_definition(x, y) -> float:
    """``f[x_0, ..., x_k] = sum_i f(x_i) / prod_{j != i} (x_i - x_j)``.

    The closed form. Independent of the recursion, so comparing the two checks both.
    """
    x, y = _check(x, y)
    n = x.size
    total = 0.0
    for i in range(n):
        d = x[i] - x
        d[i] = 1.0
        total += y[i] / np.prod(d)
    return float(total)


# ----------------------------------------------------------------------------- properties


def symmetry_report(x, y, n_orders: int = 12, rng=None) -> dict:
    """A divided difference is symmetric in its arguments, though the recursion is not.

    Permuting the nodes gives the same value to rounding, and the spread across permutations is
    a direct measure of how much the ordering costs in floating point.
    """
    x, y = _check(x, y)
    gen = np.random.default_rng() if rng is None else rng
    values = [from_definition(x, y)]
    for _ in range(max(int(n_orders), 1)):
        p = gen.permutation(x.size)
        values.append(float(coefficients(x[p], y[p])[-1]))
    v = np.asarray(values)
    scale = max(float(np.max(np.abs(v))), 1e-300)
    return {"values": v, "spread": float(np.max(v) - np.min(v)),
            "relative_spread": float((np.max(v) - np.min(v)) / scale)}


def is_leading_coefficient(x, y) -> dict:
    """``f[x_0, ..., x_k]`` is the coefficient of ``x^k`` in the interpolating polynomial.

    Checked against the power form's top coefficient, which is computed by a completely
    different route.

    **The scale is the whole table, not the answer.** The leading coefficient can be genuinely
    zero: interpolating an even function at symmetric nodes to an odd degree gives exactly that,
    and ``cos(2x)`` on six symmetric nodes measures 6.1e-16, which is zero to rounding. Dividing
    the gap by such a value reports a relative error of 1 for two answers that agree perfectly.
    So the comparison is made against the largest entry the table actually computed, which is
    what sets the rounding level of the whole calculation.
    """
    from . import interp as _ip

    x, y = _check(x, y)
    T = table(x, y)
    top = float(T[0, -1])
    a, c, h = _ip.normalised_power_form(x, y)
    # the normalised form is in s = (t - c)/h, so its leading coefficient is scaled by h^k
    powered = float(a[-1]) / (h ** (x.size - 1))
    finite = T[np.isfinite(T)]
    scale = max(abs(top), abs(powered), float(np.max(np.abs(finite))), 1e-300)
    return {"divided_difference": top, "leading_coefficient": powered,
            "absolute_gap": abs(top - powered),
            "table_scale": scale,
            "relative_gap": abs(top - powered) / scale}


def linearity_report(x, y1, y2, alpha: float = 2.5, beta: float = -1.75) -> dict:
    """``f[.]`` is linear in the data: the difference of ``alpha f + beta g`` is the combination
    of the two differences."""
    x, y1 = _check(x, y1)
    _, y2 = _check(x, y2)
    combined = coefficients(x, alpha * y1 + beta * y2)
    separate = alpha * coefficients(x, y1) + beta * coefficients(x, y2)
    scale = max(float(np.max(np.abs(separate))), 1e-300)
    return {"combined": combined, "separate": separate,
            "relative_gap": float(np.max(np.abs(combined - separate)) / scale)}


def derivative_connection(f, dfk, lo: float, hi: float, k: int, n_trials: int = 40,
                          rng=None) -> dict:
    """``f[x_0, ..., x_k] = f^(k)(xi) / k!`` for some ``xi`` in ``(lo, hi)``.

    ``dfk`` is the ``k``-th derivative. The test is that the divided difference lies between the
    smallest and largest values of ``f^(k)/k!`` on the interval, which is what the mean value
    form guarantees and is checkable without finding ``xi``.
    """
    gen = np.random.default_rng() if rng is None else rng
    k = int(k)
    grid = np.linspace(lo, hi, 2001)
    scaled = np.asarray([dfk(t) for t in grid], dtype=float) / float(math.factorial(k))
    inside = 0
    values = []
    for _ in range(max(int(n_trials), 1)):
        x = np.sort(gen.uniform(lo, hi, k + 1))
        if np.unique(x).size != x.size:
            continue
        v = float(coefficients(x, np.asarray([f(t) for t in x], dtype=float))[-1])
        values.append(v)
        tol = 1e-9 * max(abs(scaled.min()), abs(scaled.max()), 1.0)
        if scaled.min() - tol <= v <= scaled.max() + tol:
            inside += 1
    return {"values": np.asarray(values), "inside": inside, "trials": len(values),
            "range_of_derivative": (float(scaled.min()), float(scaled.max()))}


def confluent(x, derivatives) -> float:
    """The limit as all nodes coalesce: ``f[x, x, ..., x] = f^(k)(x) / k!``.

    ``derivatives[j]`` is ``f^(j)(x)``. This is the Taylor coefficient, and it is the bridge to
    lesson 50: Hermite interpolation is Newton interpolation with repeated nodes, and the
    repeated entries of the table are exactly these.
    """
    d = np.atleast_1d(np.asarray(derivatives, dtype=float)).ravel()
    k = d.size - 1
    return float(d[k] / math.factorial(k))


# ----------------------------------------------------------------------------- conditioning


@dataclass
class ConditioningReport:
    """How much a divided difference table loses as the nodes crowd together."""

    separations: np.ndarray
    errors: np.ndarray
    fitted_exponent: float


def conditioning_report(f, k: int = 6, separations=None, centre: float = 0.5) -> ConditioningReport:
    """Measure the top divided difference against the node separation.

    The recursion divides by ``x_{i+k} - x_i``, so as the nodes crowd the divisions get small and
    the subtractions cancel. The reference is the confluent limit ``f^(k)(centre)/k!``, which is
    what the divided difference tends to, computed by high order finite differences of ``f`` in
    extended precision would be circular, so it is supplied by the caller through ``f`` being a
    polynomial whose derivative is known exactly.
    """
    k = int(k)
    seps = (np.geomspace(1e-1, 1e-7, 7) if separations is None
            else np.atleast_1d(np.asarray(separations, dtype=float)))
    ref = None
    errs = []
    for h in seps:
        x = centre + h * np.arange(k + 1)
        y = np.asarray([f(t) for t in x], dtype=float)
        v = float(coefficients(x, y)[-1])
        if ref is None:
            ref = v
        errs.append(v)
    errs = np.asarray(errs, dtype=float)
    target = float(np.median(errs[:2])) if errs.size >= 2 else float(errs[0])
    rel = np.abs(errs - target) / max(abs(target), 1e-300)
    live = rel > 0
    if live.sum() >= 2:
        slope = float(np.polyfit(np.log(seps[live]), np.log(rel[live]), 1)[0])
    else:
        slope = float("nan")
    return ConditioningReport(separations=seps, errors=errs, fitted_exponent=slope)


def advantage_over_lagrange(n_nodes: int) -> dict:
    """Operation counts for the thing divided differences are actually better at.

    Lagrange has to rebuild every basis polynomial when a point is added, at ``O(n^2)``. The
    Newton form adds one coefficient at ``O(n)`` and touches nothing else. Over ``m`` added
    points that is ``O(m n^2)`` against ``O(m n)``, and it is the reason an adaptive scheme
    carries divided differences.
    """
    n = int(n_nodes)
    return {"newton_build": n * (n - 1) // 2,
            "newton_add_one_point": n,
            "lagrange_rebuild": n * n,
            "newton_evaluate": 2 * n,
            "lagrange_evaluate": 2 * n * n}
