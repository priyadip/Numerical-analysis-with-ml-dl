"""Cubic splines: the compromise that Part 7 has been working toward.

The bargain
-----------
Lesson 46 showed that raising the degree fails. Lesson 50 showed that piecewise linear
interpolation always converges but only at ``O(h^2)`` and with a visible kink at every node. A
**cubic spline** takes the piecewise idea and asks for the most smoothness the pieces can afford:
a cubic on each interval, matching value, first derivative and second derivative at every join.

Counting says it works. ``n`` intervals give ``4n`` unknown coefficients. Interpolation at both
ends of each interval gives ``2n`` conditions, continuity of ``S'`` and ``S''`` at the ``n - 1``
interior joins gives ``2(n - 1)``, so ``4n - 2`` in all. **Two conditions are missing**, and the
four classical ways of supplying them are what `natural`, `clamped`, `parabolic` and `not_a_knot`
are.

The tridiagonal system
----------------------
Writing the unknowns as the second derivatives ``M_i = S''(x_i)`` turns the continuity conditions
into

    h_{i-1} M_{i-1}  +  2 (h_{i-1} + h_i) M_i  +  h_i M_{i+1}
        =  6 ( (y_{i+1} - y_i)/h_i  -  (y_i - y_{i-1})/h_{i-1} )

which is **tridiagonal and diagonally dominant**, so lesson 21's Thomas algorithm solves it in
``O(n)`` and lesson 17 guarantees it never needs pivoting. That is the reason splines are cheap:
the global coupling that smoothness demands costs a linear solve of the easiest possible kind.

The end conditions, and what each costs
---------------------------------------
=============  ======================================  ========================
name           extra conditions                        order at the ends
=============  ======================================  ========================
natural        ``S'' = 0`` at both ends                ``O(h^2)``
clamped        ``S'`` given at both ends                ``O(h^4)``
parabolic      ``S'' `` constant on the end intervals   ``O(h^3)``
not-a-knot     ``S'''`` continuous at the second and    ``O(h^4)``
               second-to-last knots
=============  ======================================  ========================

`convergence_by_end_condition` measures those orders, and they come out as the table says. The
practical reading is that **natural splines are the worst choice unless the function really does
have zero curvature at the ends**, which is rarely true and is exactly what the name hides.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Spline:
    """A cubic spline: the knots, the per interval coefficients, and how it was closed."""

    x: np.ndarray
    y: np.ndarray
    moments: np.ndarray                # M_i = S''(x_i)
    end_condition: str

    def __call__(self, t):
        return evaluate(self, t)

    def derivative(self, t, order: int = 1):
        return evaluate(self, t, order=int(order))


def _check(x, y):
    xa = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    ya = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if xa.size < 3:
        raise ValueError(f"a cubic spline needs at least three knots, got {xa.size}")
    if xa.size != ya.size:
        raise ValueError(f"got {xa.size} knots and {ya.size} values, they must match")
    order = np.argsort(xa)
    xa, ya = xa[order], ya[order]
    if np.any(np.diff(xa) <= 0):
        raise ValueError("the knots must be distinct")
    return xa, ya


def _thomas(lower, diag, upper, rhs) -> np.ndarray:
    """Lesson 21's tridiagonal solve, in ``O(n)`` and with no pivoting.

    The spline system is diagonally dominant by construction, since ``2(h_{i-1} + h_i)`` exceeds
    ``h_{i-1} + h_i``, so lesson 17's theorem says the factorization exists and is stable without
    any row exchange.
    """
    n = diag.size
    c = diag.astype(float).copy()
    d = rhs.astype(float).copy()
    for i in range(1, n):
        m = lower[i - 1] / c[i - 1]
        c[i] -= m * upper[i - 1]
        d[i] -= m * d[i - 1]
    out = np.empty(n)
    out[-1] = d[-1] / c[-1]
    for i in range(n - 2, -1, -1):
        out[i] = (d[i] - upper[i] * out[i + 1]) / c[i]
    return out


def _moment_system(x, y):
    """The interior equations, common to every end condition."""
    h = np.diff(x)
    n = x.size
    slope = np.diff(y) / h
    rhs = 6.0 * (slope[1:] - slope[:-1])
    lower = h[:-1][1:].astype(float) if n > 3 else np.empty(max(n - 3, 0))
    return h, slope, rhs


# ----------------------------------------------------------------------------- the four ends


def natural(x, y) -> Spline:
    """``S'' = 0`` at both ends. The commonest default and usually the wrong one.

    The name suggests it is the neutral choice. It is not: it imposes zero curvature at the ends,
    which is a real assumption about the function, and when the function does not satisfy it the
    convergence drops to ``O(h^2)`` near the ends while the interior stays ``O(h^4)``.
    """
    xa, ya = _check(x, y)
    n = xa.size
    h, slope, rhs = _moment_system(xa, ya)
    if n == 3:
        M = np.zeros(n)
        M[1] = rhs[0] / (2.0 * (h[0] + h[1]))
        return Spline(x=xa, y=ya, moments=M, end_condition="natural")
    diag = 2.0 * (h[:-1] + h[1:])
    lower = h[1:-1].astype(float)
    upper = h[1:-1].astype(float)
    inner = _thomas(lower, diag, upper, rhs)
    M = np.concatenate([[0.0], inner, [0.0]])
    return Spline(x=xa, y=ya, moments=M, end_condition="natural")


def clamped(x, y, dy_start: float, dy_end: float) -> Spline:
    """``S'(x_0)`` and ``S'(x_n)`` supplied. **The most accurate**, at ``O(h^4)`` throughout.

    It needs information the other three do not, and when that information is available this is
    the choice with no downside.
    """
    xa, ya = _check(x, y)
    n = xa.size
    h = np.diff(xa)
    slope = np.diff(ya) / h
    diag = np.empty(n)
    lower = np.empty(n - 1)
    upper = np.empty(n - 1)
    rhs = np.empty(n)
    diag[0] = 2.0 * h[0]
    upper[0] = h[0]
    rhs[0] = 6.0 * (slope[0] - float(dy_start))
    diag[1:-1] = 2.0 * (h[:-1] + h[1:])
    lower[:-1] = h[:-1]
    upper[1:] = h[1:]
    rhs[1:-1] = 6.0 * (slope[1:] - slope[:-1])
    diag[-1] = 2.0 * h[-1]
    lower[-1] = h[-1]
    rhs[-1] = 6.0 * (float(dy_end) - slope[-1])
    M = _thomas(lower, diag, upper, rhs)
    return Spline(x=xa, y=ya, moments=M, end_condition="clamped")


def parabolic(x, y) -> Spline:
    """``M_0 = M_1`` and ``M_n = M_{n-1}``: the end intervals are parabolas, not cubics.

    A middle course. It assumes constant curvature on the outermost intervals rather than zero
    curvature at the outermost points, which is a weaker and usually more defensible assumption,
    and it buys ``O(h^3)``.
    """
    xa, ya = _check(x, y)
    n = xa.size
    h, slope, rhs = _moment_system(xa, ya)
    if n == 3:
        M = np.full(n, rhs[0] / (2.0 * (h[0] + h[1]) + h[0] + h[1]))
        return Spline(x=xa, y=ya, moments=M, end_condition="parabolic")
    diag = 2.0 * (h[:-1] + h[1:]).astype(float)
    diag[0] += h[0]                              # M_0 = M_1 folds into the first equation
    diag[-1] += h[-1]                            # M_n = M_{n-1} folds into the last
    lower = h[1:-1].astype(float)
    upper = h[1:-1].astype(float)
    inner = _thomas(lower, diag, upper, rhs)
    M = np.concatenate([[inner[0]], inner, [inner[-1]]])
    return Spline(x=xa, y=ya, moments=M, end_condition="parabolic")


def not_a_knot(x, y) -> Spline:
    """``S'''`` continuous at ``x_1`` and ``x_{n-1}``, so those are not really knots at all.

    The first two intervals share one cubic and so do the last two. It needs no extra information
    about ``f`` and still reaches ``O(h^4)``, which makes it the right default and is why it is
    MATLAB's ``spline`` and SciPy's ``CubicSpline(bc_type="not-a-knot")``.
    """
    xa, ya = _check(x, y)
    n = xa.size
    if n < 4:
        return parabolic(xa, ya)
    h = np.diff(xa)
    slope = np.diff(ya) / h
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6.0 * (slope[i] - slope[i - 1])
    # S''' continuous at x_1:  (M_1 - M_0)/h_0 = (M_2 - M_1)/h_1
    A[0, 0] = h[1]
    A[0, 1] = -(h[0] + h[1])
    A[0, 2] = h[0]
    # and at x_{n-1}
    A[-1, -3] = h[-1]
    A[-1, -2] = -(h[-2] + h[-1])
    A[-1, -1] = h[-2]
    M = np.linalg.solve(A, rhs)
    return Spline(x=xa, y=ya, moments=M, end_condition="not-a-knot")


BUILDERS = {"natural": natural, "parabolic": parabolic, "not-a-knot": not_a_knot}


# ----------------------------------------------------------------------------- evaluation


def evaluate(spline: Spline, t, order: int = 0) -> np.ndarray:
    """Evaluate the spline, or its first or second derivative, at ``t``.

    On interval ``i``, with ``a = x_{i+1} - t`` and ``b = t - x_i``,

        S(t) = M_i a^3/(6h) + M_{i+1} b^3/(6h)
               + (y_i/h - M_i h/6) a + (y_{i+1}/h - M_{i+1} h/6) b
    """
    x, y, M = spline.x, spline.y, spline.moments
    z = np.atleast_1d(np.asarray(t, dtype=float))
    i = np.clip(np.searchsorted(x, z) - 1, 0, x.size - 2)
    h = x[i + 1] - x[i]
    a = x[i + 1] - z
    b = z - x[i]
    order = int(order)
    if order == 0:
        out = (M[i] * a ** 3 + M[i + 1] * b ** 3) / (6.0 * h) \
            + (y[i] / h - M[i] * h / 6.0) * a + (y[i + 1] / h - M[i + 1] * h / 6.0) * b
    elif order == 1:
        out = (-M[i] * a ** 2 + M[i + 1] * b ** 2) / (2.0 * h) \
            - (y[i] / h - M[i] * h / 6.0) + (y[i + 1] / h - M[i + 1] * h / 6.0)
    elif order == 2:
        out = (M[i] * a + M[i + 1] * b) / h
    else:
        raise ValueError(f"order 0, 1 or 2 is supported, got {order}")
    return out.reshape(np.shape(t)) if np.ndim(t) else float(out[0])


def equispaced_system(n_intervals: int, h: float) -> dict:
    """The special case ``h_i = h``, where the system becomes constant coefficient.

    Every row is ``M_{i-1} + 4 M_i + M_{i+1} = 6/h^2 (y_{i-1} - 2 y_i + y_{i+1})``. The matrix is
    the same ``[1, 4, 1]`` for every row, so its condition number is bounded independently of
    ``n``: the eigenvalues of the tridiagonal ``[1, 4, 1]`` lie in ``(2, 6)``, giving
    ``kappa < 3`` however many knots there are.
    """
    n = int(n_intervals)
    if n < 2:
        raise ValueError(f"need at least two intervals, got {n}")
    k = np.arange(1, n)
    eig = 4.0 + 2.0 * np.cos(k * np.pi / n)
    return {"stencil": np.array([1.0, 4.0, 1.0]),
            "rhs_factor": 6.0 / float(h) ** 2,
            "eigenvalue_range": (float(eig.min()), float(eig.max())),
            "condition_number": float(eig.max() / eig.min()),
            "bounded_independently_of_n": bool(eig.max() / eig.min() < 3.0)}


# ----------------------------------------------------------------------------- properties


def _piece(spline: Spline, i, t, order: int = 0) -> np.ndarray:
    """Evaluate the cubic of interval ``i`` at ``t``, without clamping ``t`` to that interval.

    This is what a continuity check needs: the two pieces meeting at a knot must be compared
    **at the knot**, using each one's own formula. Sampling at ``x - eps`` and ``x + eps``
    instead measures ``2 eps S'``, which for a perfectly smooth spline is about 4e-7 at
    ``eps = 1e-7`` and looks exactly like a jump.
    """
    x, y, M = spline.x, spline.y, spline.moments
    i = np.atleast_1d(np.asarray(i, dtype=int))
    z = np.atleast_1d(np.asarray(t, dtype=float))
    h = x[i + 1] - x[i]
    a = x[i + 1] - z
    b = z - x[i]
    if order == 0:
        return (M[i] * a ** 3 + M[i + 1] * b ** 3) / (6.0 * h) \
            + (y[i] / h - M[i] * h / 6.0) * a + (y[i + 1] / h - M[i + 1] * h / 6.0) * b
    if order == 1:
        return (-M[i] * a ** 2 + M[i + 1] * b ** 2) / (2.0 * h) \
            - (y[i] / h - M[i] * h / 6.0) + (y[i + 1] / h - M[i + 1] * h / 6.0)
    if order == 2:
        return (M[i] * a + M[i + 1] * b) / h
    if order == 3:
        return np.full(z.shape, 0.0) + (M[i + 1] - M[i]) / h
    raise ValueError(f"order 0 to 3 is supported, got {order}")


def smoothness_report(spline: Spline) -> dict:
    """Check continuity of ``S``, ``S'``, ``S''`` and ``S'''`` across every interior knot.

    A cubic spline is ``C^2`` and no more: the third derivative is piecewise constant and jumps
    at every knot. Both halves of that are measured, so the smoothness class is established
    rather than asserted.

    Each knot is approached from its left piece and its right piece, both evaluated **at** the
    knot, so what is reported is a genuine jump and not the function's own slope.
    """
    x = spline.x
    n = x.size
    if n < 3:
        return {}
    inner = np.arange(1, n - 1)
    knots = x[inner]
    out = {}
    for k in (0, 1, 2, 3):
        a = np.atleast_1d(_piece(spline, inner - 1, knots, order=k))
        b = np.atleast_1d(_piece(spline, inner, knots, order=k))
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1.0)
        out[f"jump_in_derivative_{k}"] = float(np.max(np.abs(a - b)) / scale)
    return out


def interpolates(spline: Spline) -> float:
    """The largest deviation from the data at the knots, relative to the data."""
    got = np.atleast_1d(evaluate(spline, spline.x))
    scale = max(float(np.max(np.abs(spline.y))), 1.0)
    return float(np.max(np.abs(got - spline.y)) / scale)


def with_end_moments(x, y, m_start: float, m_end: float) -> Spline:
    """The interpolating spline with the two end second derivatives set to given values.

    Every member of this family is a genuine ``C^2`` interpolant of the same data, so the family
    is exactly the competition class the minimum curvature theorem is stated over, restricted to
    cubic splines. ``m_start = m_end = 0`` is the natural spline.
    """
    xa, ya = _check(x, y)
    n = xa.size
    h, slope, rhs = _moment_system(xa, ya)
    ms, me = float(m_start), float(m_end)
    if n == 3:
        M = np.array([ms, (rhs[0] - h[0] * ms - h[1] * me) / (2.0 * (h[0] + h[1])), me])
        return Spline(x=xa, y=ya, moments=M, end_condition="given end moments")
    adjusted = rhs.copy()
    adjusted[0] -= h[0] * ms
    adjusted[-1] -= h[-1] * me
    diag = 2.0 * (h[:-1] + h[1:])
    inner = _thomas(h[1:-1].astype(float), diag, h[1:-1].astype(float), adjusted)
    return Spline(x=xa, y=ya, moments=np.concatenate([[ms], inner, [me]]),
                  end_condition="given end moments")


def bending_energy(spline: Spline, n_probe: int = 4001) -> float:
    """``integral (S'')^2`` over the knot range, which is the quantity the theorem minimises."""
    probe = np.linspace(float(spline.x[0]), float(spline.x[-1]), int(n_probe))
    return float(np.trapezoid(np.atleast_1d(evaluate(spline, probe, order=2)) ** 2, probe))


def minimum_curvature(spline: Spline, n_trials: int = 200, rng=None,
                      spread: float = 3.0) -> dict:
    """The natural spline minimises ``integral (g'')^2`` over all ``C^2`` interpolants.

    That is the theorem the word "spline" comes from: a draughtsman's flexible strip takes the
    shape minimising bending energy, and the natural cubic spline is exactly that shape.

    **The competitors have to be in the competition class**, which is the part that is easy to
    get wrong. Perturbing the moments of a spline directly still interpolates the data and still
    leaves ``S''`` continuous, but it breaks continuity of ``S'``, so the result is not ``C^1``
    and is not an admissible competitor at all. Some such functions have lower bending energy,
    which is not a counterexample to anything.

    The family used here is `with_end_moments`, every member of which is a genuine ``C^2``
    interpolant of the same data, with the natural spline sitting at ``m_start = m_end = 0``.
    """
    gen = np.random.default_rng() if rng is None else rng
    base = bending_energy(spline)
    worse = 0
    energies = []
    for _ in range(int(n_trials)):
        a, b = gen.uniform(-float(spread), float(spread), 2)
        if abs(a) < 1e-12 and abs(b) < 1e-12:
            continue
        e = bending_energy(with_end_moments(spline.x, spline.y, a, b))
        energies.append(e)
        if e > base:
            worse += 1
    return {"energy": base, "trials": len(energies),
            "competitor_energies": np.asarray(energies),
            "perturbations_with_more_energy": worse,
            "is_minimum": bool(worse == len(energies))}


def convergence_by_end_condition(f, df=None, n_values=None, lo: float = 0.0, hi: float = 1.0,
                                 n_probe: int = 4001) -> dict:
    """Fit the observed order of each end condition, separately at the ends and in the interior.

    The point of splitting the interval is that the end conditions only affect the ends. Natural
    splines are ``O(h^4)`` in the middle and ``O(h^2)`` near the boundary, and quoting one number
    for the whole interval hides which half is responsible.
    """
    ns = ([4, 8, 16, 32, 64] if n_values is None else [int(v) for v in np.atleast_1d(n_values)])
    probe = np.linspace(lo, hi, int(n_probe))
    truth = np.asarray([f(v) for v in probe], dtype=float)
    span = hi - lo
    edge = (probe < lo + 0.1 * span) | (probe > hi - 0.1 * span)
    names = list(BUILDERS) + (["clamped"] if df is not None else [])
    overall, ends, middle = {k: [] for k in names}, {k: [] for k in names}, {k: [] for k in names}
    for n in ns:
        x = np.linspace(lo, hi, n + 1)
        y = np.asarray([f(v) for v in x], dtype=float)
        for name in names:
            s = (clamped(x, y, float(df(lo)), float(df(hi))) if name == "clamped"
                 else BUILDERS[name](x, y))
            d = np.abs(np.atleast_1d(evaluate(s, probe)) - truth)
            overall[name].append(float(np.max(d)))
            ends[name].append(float(np.max(d[edge])))
            middle[name].append(float(np.max(d[~edge])))
    arr = np.log(np.asarray(ns, dtype=float))
    fit = lambda v: float(-np.polyfit(arr, np.log(np.maximum(np.asarray(v), 1e-300)), 1)[0])
    return {"n_values": np.asarray(ns),
            "overall": {k: np.asarray(v) for k, v in overall.items()},
            "order_overall": {k: fit(v) for k, v in overall.items()},
            "order_at_the_ends": {k: fit(v) for k, v in ends.items()},
            "order_in_the_middle": {k: fit(v) for k, v in middle.items()}}
