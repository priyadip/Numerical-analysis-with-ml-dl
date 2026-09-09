"""Hermite interpolation and the piecewise idea: matching derivatives, and giving up on degree.

Two ideas, and they pull in opposite directions
------------------------------------------------
**Hermite interpolation** asks for more from each node: match the value *and* the derivative
there, or the first ``m`` derivatives. That raises the degree and buys a better fit near each
node, and the error formula gains a squared factor:

    f(t) - p(t)  =  f^(2n+2)(xi) / (2n+2)!  *  prod_i (t - x_i)^2

The node polynomial is **squared**, which is a genuine improvement in the middle of each gap and
does nothing at all about lesson 46's problem, since squaring a large number makes it larger.

**Piecewise interpolation** goes the other way: keep the degree low and use many pieces. The
error of piecewise linear interpolation on a grid of spacing ``h`` is

    |f - p|  <=  h^2 / 8  *  max |f''|

which involves no factorial, no node polynomial and no degree at all. It converges for **any**
twice differentiable function, at any spacing, with no Runge phenomenon possible, because the
degree never grows.

`hermite_error_bound` and `piecewise_linear_bound` compute both, and `runge_by_pieces` shows
piecewise linear interpolation converging on the function that destroyed lesson 46.

The confluent view
------------------
Hermite interpolation **is** Newton interpolation with repeated nodes. Repeating a node ``m``
times and using lesson 45's confluent divided differences ``f[x, x, ..., x] = f^(k)(x)/k!`` in
place of the ones that would divide by zero gives the Hermite polynomial directly, with no new
theory. `hermite_newton` does exactly that, and `hermite_basis` builds the same polynomial from
the classical basis functions so the two can be compared.

Weierstrass
-----------
Weierstrass's theorem says every continuous function on a closed interval is the uniform limit of
*some* sequence of polynomials. It says nothing about interpolation at prescribed nodes, and
lesson 46 shows the distinction is real: the best polynomials converge while the interpolants at
equally spaced nodes diverge, on the same function. `weierstrass_gap` measures both at once.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import interp as _ip


def _check(x, y, dy=None):
    xa = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    ya = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if xa.size == 0:
        raise ValueError("need at least one node")
    if xa.size != ya.size:
        raise ValueError(f"got {xa.size} nodes and {ya.size} values, they must match")
    if np.unique(xa).size != xa.size:
        raise ValueError("the nodes must be distinct; repetition is what the derivative data is")
    if dy is None:
        return xa, ya, None
    da = np.atleast_1d(np.asarray(dy, dtype=float)).ravel()
    if da.size != xa.size:
        raise ValueError(f"got {xa.size} nodes and {da.size} derivatives, they must match")
    return xa, ya, da


# ----------------------------------------------------------------------------- Hermite


def hermite_newton(x, y, dy):
    """The Hermite polynomial as a Newton form with each node repeated twice.

    Returns ``(coefficients, doubled_nodes)`` for `interp.evaluate_newton`.

    **This is the whole construction.** Build the divided difference table on the doubled node
    list, and wherever the recursion would divide by ``x_i - x_i = 0``, substitute the confluent
    value ``f'(x_i)`` from lesson 45. Nothing else changes.
    """
    xa, ya, da = _check(x, y, dy)
    n = xa.size
    z = np.repeat(xa, 2)
    m = 2 * n
    # The table is kept in the TOP ROW convention used throughout Part 7:
    # T[r, k] = f[z_r, ..., z_{r+k}], and the coefficients are T[0, :].
    # In that convention the confluent entry sits at T[2i, 1], because rows 2i and 2i+1 are the
    # two copies of x_i. Placing it at T[2i+1, 1] instead is the Burden and Faires diagonal
    # convention and mixing the two produces a polynomial that interpolates nothing.
    T = np.zeros((m, m))
    T[:, 0] = np.repeat(ya, 2)
    for r in range(m - 1):
        if z[r + 1] == z[r]:
            T[r, 1] = da[r // 2]                   # the confluent entry, f'(x_i)
        else:
            T[r, 1] = (T[r + 1, 0] - T[r, 0]) / (z[r + 1] - z[r])
    for k in range(2, m):
        for r in range(m - k):
            T[r, k] = (T[r + 1, k - 1] - T[r, k - 1]) / (z[r + k] - z[r])
    return T[0, :], z


def hermite_basis(x, y, dy, t):
    """The same polynomial from the classical basis, as an independent construction.

    ``p = sum_i y_i H_i(t) + sum_i y'_i K_i(t)`` with

        H_i = (1 - 2 (t - x_i) L'_i(x_i)) L_i(t)^2
        K_i = (t - x_i) L_i(t)^2

    Each ``H_i`` is 1 at ``x_i`` with zero derivative there, and each ``K_i`` is 0 at ``x_i`` with
    derivative 1, which is exactly the interpolation conditions written as a basis.
    """
    xa, ya, da = _check(x, y, dy)
    n = xa.size
    z = np.atleast_1d(np.asarray(t, dtype=float))
    L = _ip.lagrange_basis(xa, z)
    total = np.zeros_like(z, dtype=float)
    for i in range(n):
        others = np.delete(np.arange(n), i)
        lprime = float(np.sum(1.0 / (xa[i] - xa[others]))) if n > 1 else 0.0
        H = (1.0 - 2.0 * (z - xa[i]) * lprime) * L[i] ** 2
        K = (z - xa[i]) * L[i] ** 2
        total = total + ya[i] * H + da[i] * K
    return total.reshape(np.shape(t)) if np.ndim(t) else float(total[0])


def osculating(x, derivatives):
    """The general osculating polynomial: match ``m_i`` derivatives at node ``i``.

    ``derivatives[i]`` is the list ``[f(x_i), f'(x_i), ..., f^(m_i)(x_i)]``. The construction is
    the confluent Newton table again, with node ``i`` repeated ``m_i + 1`` times and the ``k``-th
    confluent entry set to ``f^(k)(x_i)/k!``.

    Taylor's polynomial is the special case of **one** node repeated ``m + 1`` times, and
    Lagrange interpolation is the case of every ``m_i = 0``. Both are checked in the tests, which
    is the point of writing the general version.
    """
    xa = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if len(derivatives) != xa.size:
        raise ValueError(f"got {xa.size} nodes and {len(derivatives)} derivative lists")
    if np.unique(xa).size != xa.size:
        raise ValueError("the nodes must be distinct")
    counts = [len(np.atleast_1d(d)) for d in derivatives]
    if any(c < 1 for c in counts):
        raise ValueError("every node needs at least its function value")
    z = np.concatenate([np.full(c, xa[i]) for i, c in enumerate(counts)])
    m = z.size
    T = np.zeros((m, m))
    start = 0
    for i, c in enumerate(counts):
        d = np.atleast_1d(np.asarray(derivatives[i], dtype=float))
        for r in range(c):
            T[start + r, 0] = d[0]
        for k in range(1, c):
            for r in range(c - k):
                T[start + r, k] = d[k] / math.factorial(k)
        start += c
    for k in range(1, m):
        for r in range(m - k):
            if T[r, k] != 0.0 and z[r + k] == z[r]:
                continue                            # already set from the confluent data
            if z[r + k] == z[r]:
                continue
            T[r, k] = (T[r + 1, k - 1] - T[r, k - 1]) / (z[r + k] - z[r])
    return T[0, :], z


# ----------------------------------------------------------------------------- error bounds


def hermite_error_bound(max_derivative: float, nodes, t) -> np.ndarray:
    """``|f^(2n+2)| / (2n+2)! * prod (t - x_i)^2``. Note the **squared** node polynomial."""
    from . import interperror as _ie

    x = np.atleast_1d(np.asarray(nodes, dtype=float)).ravel()
    n = x.size
    w = _ie.node_polynomial(x, t)
    return float(max_derivative) / math.factorial(2 * n) * np.asarray(w) ** 2


def piecewise_linear_bound(max_second_derivative: float, h: float) -> float:
    """``h^2 / 8 * max|f''|``. **No degree, no factorial, no node polynomial.**

    That absence is the whole point. Every one of lesson 46's difficulties came from a factor
    that grows with the degree, and here the degree is fixed at 1 forever.
    """
    return float(max_second_derivative) * float(h) ** 2 / 8.0


def piecewise_linear(x, y, t) -> np.ndarray:
    """Straight lines between consecutive points, evaluated by `numpy.interp`'s definition."""
    xa, ya, _ = _check(x, y)
    order = np.argsort(xa)
    return np.interp(np.asarray(t, dtype=float), xa[order], ya[order])


@dataclass
class PieceReport:
    """Piecewise linear error against the number of pieces."""

    n_pieces: np.ndarray
    errors: np.ndarray
    bounds: np.ndarray
    fitted_order: float
    fitted_order_all_points: float


def runge_by_pieces(n_pieces=None, a: float = 25.0, lo: float = -1.0,
                    hi: float = 1.0, n_probe: int = 4001) -> PieceReport:
    """Piecewise linear interpolation of Runge's function, which polynomials could not do.

    The error falls like ``h^2`` at every refinement, with no divergence at any number of pieces,
    because the degree never grows. That is the trade the rest of Part 7 is built on: give up
    high order accuracy and get unconditional convergence.
    """
    from . import interperror as _ie

    ns = ([4, 8, 16, 32, 64, 128] if n_pieces is None
          else [int(v) for v in np.atleast_1d(n_pieces)])
    probe = np.linspace(lo, hi, int(n_probe))
    truth = _ie.runge(probe, a)
    # max |f''| of 1/(1 + a t^2) on the interval, computed on a fine grid
    t = np.linspace(lo, hi, 20001)
    d2 = (2.0 * a * (3.0 * a * t ** 2 - 1.0)) / (1.0 + a * t ** 2) ** 3
    m2 = float(np.max(np.abs(d2)))
    errs, bounds = [], []
    for n in ns:
        x = np.linspace(lo, hi, n + 1)
        h = (hi - lo) / n
        got = piecewise_linear(x, _ie.runge(x, a), probe)
        errs.append(float(np.max(np.abs(got - truth))))
        bounds.append(piecewise_linear_bound(m2, h))
    e = np.asarray(errs)
    arr = np.asarray(ns, dtype=float)
    # Fit the asymptotic tail only. On Runge's function the coarse grids are pre-asymptotic,
    # because the peak has not been resolved yet, and including them gives a fitted order of
    # 1.32 for a method that is measurably second order once the peak is resolved.
    tail = slice(max(len(ns) // 2, 1), None)
    order = float(-np.polyfit(np.log(arr[tail]), np.log(e[tail]), 1)[0])
    all_points = float(-np.polyfit(np.log(arr), np.log(e), 1)[0])
    return PieceReport(n_pieces=np.asarray(ns), errors=e, bounds=np.asarray(bounds),
                       fitted_order=order, fitted_order_all_points=all_points)


# ----------------------------------------------------------------------------- Weierstrass


def weierstrass_gap(f, degrees=None, lo: float = -1.0, hi: float = 1.0,
                    n_probe: int = 2001) -> dict:
    """The best polynomial approximation against the equally spaced interpolant, at each degree.

    Weierstrass guarantees the first column goes to zero for any continuous ``f``. It guarantees
    **nothing** about the second, and on Runge's function the second diverges while the first
    does not. That is the precise content of the distinction between approximation and
    interpolation.

    The best approximation is estimated by a high order Chebyshev projection, which is within a
    small factor of the true minimax polynomial and is what a practical code would use.
    """
    from . import chebyshev as _cb

    ds = ([4, 8, 12, 16, 20, 24] if degrees is None else [int(v) for v in np.atleast_1d(degrees)])
    probe = np.linspace(lo, hi, int(n_probe))
    truth = np.asarray([f(v) for v in probe], dtype=float)
    best, interp_err = [], []
    for d in ds:
        p, _, _ = _cb.interpolate(f, d + 1, lo=lo, hi=hi)
        best.append(float(np.max(np.abs(np.asarray(p(probe)) - truth))))
        x = np.linspace(lo, hi, d + 1)
        y = np.asarray([f(v) for v in x], dtype=float)
        interp_err.append(float(np.max(np.abs(_ip.evaluate_barycentric(x, y, probe) - truth))))
    return {"degrees": np.asarray(ds),
            "near_best_approximation": np.asarray(best),
            "equally_spaced_interpolation": np.asarray(interp_err),
            "approximation_converges": bool(best[-1] < best[0]),
            "interpolation_converges": bool(interp_err[-1] < interp_err[0])}
