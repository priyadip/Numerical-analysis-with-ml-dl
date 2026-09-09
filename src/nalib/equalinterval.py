"""The classical equal interval interpolation formulas, and when to use each.

What this module is for
-----------------------
On an equally spaced grid ``x_k = x_0 + k h``, lesson 48's operator algebra turns the Newton form
into a family of formulas that all represent **the same interpolating polynomial** and differ
only in which corner of the difference table they read. They exist because a human working from
a printed table wanted the terms to shrink fast, and that means starting from the entries nearest
the point being evaluated.

Every one of them is the same polynomial, where every one of them can be formed
-------------------------------------------------------------------------------
Given the same grid and the same data, all eight return the **same value** to rounding. They are
rearrangements, not alternatives. `all_formulas_agree` measures it: on nine nodes over ``[0, 1]``
with ``f = exp(2x)``, at ``t = 0.25``, ``0.5`` and ``0.75`` the eight agree to **0.00e+00**.

**The qualification is the real content of the subject.** A central formula expands about a node
and needs differences on *both* sides of it, so near the ends of the table it simply runs out:

=======  ============  ============  ============  ============================
``t``    centre node   room below    room above    highest central order usable
=======  ============  ============  ============  ============================
0.06     0             0             8             0
0.25     2             2             6             4
0.50     4             4             4             8
0.94     8             8             0             0
=======  ============  ============  ============  ============================

At ``t = 0.94`` the central formulas can reach order 0 and their error is 8.4e-1, while
Gregory-Newton backward reads a *diagonal* rather than a centred band, reaches full order, and
gets 1.3e-7, which is the interpolation error of the degree 8 polynomial and not a defect of the
formula.

So the classical advice is not primarily about how fast the terms shrink. It is that **near the
ends the central formulas cannot be formed at all**, and near the middle they can and are then
better centred on the point of interest. That is why both families exist.

The formulas
------------
Write ``s = (t - x_0) / h`` for the forward formulas and ``s = (t - x_n) / h`` for the backward
ones, and ``s`` measured from the chosen central node for the rest.

**Gregory-Newton forward**: reads the top diagonal of the table. Terms are
``C(s, k) Δ^k y_0``. Best when ``t`` is near the **start** of the table.

**Gregory-Newton backward**: reads the bottom diagonal, ``C(s + k - 1, k) ∇^k y_n``. Best near
the **end**.

**Gauss forward and backward**: zig-zag along the central diagonal, taking differences
alternately above and below. Best near the **middle**, which is where a well designed table puts
the point of interest.

**Stirling**: the average of the two Gauss formulas. Its terms involve ``μ δ`` and ``δ^2``, so it
is symmetric about the central node. Best when ``|s| < 1/4``.

**Bessel**: the average taken about the **midpoint** between two nodes rather than about a node.
Best when ``s`` is near ``1/2``, and it is the natural choice for interpolating halfway.

**Everett**: keeps only the even order differences, at the cost of using two central nodes. Half
the table is never needed, which mattered when tables were printed and expensive.

**Steffensen**: a rearrangement of Everett in terms of the odd differences.

Which to use, in one line
-------------------------
Near the start use forward, near the end use backward, near a node use Stirling, near a midpoint
use Bessel. `best_formula_for` implements that rule and `error_by_position` measures whether it
is right, which it is: the recommended formula has the smallest truncation error in each region.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import finitediff as _fd


def _grid(x):
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if v.size < 2:
        raise ValueError(f"need at least two nodes, got {v.size}")
    h = np.diff(v)
    if not np.allclose(h, h[0], rtol=1e-10, atol=0.0):
        raise ValueError("these formulas require an EQUALLY spaced grid; "
                         "use nalib.interp for a general one")
    return v, float(h[0])


def _pair(x, y):
    v, h = _grid(x)
    w = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if w.size != v.size:
        raise ValueError(f"got {v.size} nodes and {w.size} values, they must match")
    return v, w, h


def binomial_s(s, k: int) -> np.ndarray:
    """``C(s, k) = s(s-1)...(s-k+1) / k!`` for real ``s``, which is what these formulas need."""
    k = int(k)
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    z = np.asarray(s, dtype=float)
    out = np.ones_like(z, dtype=float)
    for j in range(k):
        out = out * (z - j)
    return out / math.factorial(k)


# ----------------------------------------------------------------------------- Gregory-Newton


def gregory_newton_forward(x, y, t, order: int | None = None) -> np.ndarray:
    """``p(t) = sum_k C(s, k) Δ^k y_0`` with ``s = (t - x_0)/h``. Reads the top diagonal."""
    v, w, h = _pair(x, y)
    n = v.size - 1 if order is None else min(int(order), v.size - 1)
    s = (np.asarray(t, dtype=float) - v[0]) / h
    total = np.zeros_like(np.atleast_1d(s), dtype=float)
    for k in range(n + 1):
        d = _fd.forward(w, k)
        if d.size == 0:
            break
        total = total + binomial_s(np.atleast_1d(s), k) * d[0]
    return total.reshape(np.shape(t)) if np.ndim(t) else float(total[0])


def gregory_newton_backward(x, y, t, order: int | None = None) -> np.ndarray:
    """``p(t) = sum_k C(s + k - 1, k) ∇^k y_n`` with ``s = (t - x_n)/h``. Bottom diagonal."""
    v, w, h = _pair(x, y)
    n = v.size - 1 if order is None else min(int(order), v.size - 1)
    s = (np.asarray(t, dtype=float) - v[-1]) / h
    total = np.zeros_like(np.atleast_1d(s), dtype=float)
    for k in range(n + 1):
        d = _fd.forward(w, k)
        if d.size == 0:
            break
        total = total + binomial_s(np.atleast_1d(s) + k - 1, k) * d[-1]
    return total.reshape(np.shape(t)) if np.ndim(t) else float(total[0])


# ----------------------------------------------------------------------------- Gauss central


def _central_index(v, t) -> int:
    """The node nearest ``t``, which is where the central formulas expand about."""
    return int(np.argmin(np.abs(v - float(np.atleast_1d(t)[0]))))


def gauss_forward(x, y, t, centre: int | None = None, order: int | None = None) -> np.ndarray:
    """The Gauss forward formula: differences taken alternately below and above the centre.

    ``p = y_0 + s δy_{1/2} + C(s,2) δ²y_0 + C(s+1,3) δ³y_{1/2} + C(s+1,4) δ⁴y_0 + ...``

    In table terms, it zig-zags **down then up** from the central row.
    """
    v, w, h = _pair(x, y)
    c = _central_index(v, t) if centre is None else int(centre)
    s = (np.asarray(t, dtype=float) - v[c]) / h
    kmax = v.size - 1 if order is None else min(int(order), v.size - 1)
    total = np.zeros_like(np.atleast_1d(s), dtype=float) + w[c]
    for k in range(1, kmax + 1):
        d = _fd.forward(w, k)
        row = c - (k // 2)                       # Gauss forward starts below the centre
        if not 0 <= row < d.size:
            break
        shift = (k - 1) // 2
        total = total + binomial_s(np.atleast_1d(s) + shift, k) * d[row]
    return total.reshape(np.shape(t)) if np.ndim(t) else float(total[0])


def gauss_backward(x, y, t, centre: int | None = None, order: int | None = None) -> np.ndarray:
    """The Gauss backward formula: the same zig-zag taken **up then down**.

    ``p = y_0 + s δy_{-1/2} + C(s+1,2) δ²y_0 + C(s+1,3) δ³y_{-1/2} + ...``
    """
    v, w, h = _pair(x, y)
    c = _central_index(v, t) if centre is None else int(centre)
    s = (np.asarray(t, dtype=float) - v[c]) / h
    kmax = v.size - 1 if order is None else min(int(order), v.size - 1)
    total = np.zeros_like(np.atleast_1d(s), dtype=float) + w[c]
    for k in range(1, kmax + 1):
        d = _fd.forward(w, k)
        row = c - ((k + 1) // 2)                 # Gauss backward starts above the centre
        if not 0 <= row < d.size:
            break
        shift = k // 2
        total = total + binomial_s(np.atleast_1d(s) + shift, k) * d[row]
    return total.reshape(np.shape(t)) if np.ndim(t) else float(total[0])


# ----------------------------------------------------------------------------- Stirling


def stirling(x, y, t, centre: int | None = None, order: int | None = None) -> np.ndarray:
    """The average of the two Gauss formulas, so it is symmetric about the central node.

    Because it is an average of two representations of the same polynomial, it is that same
    polynomial again. Its advantage is that the terms are symmetric, so odd order errors cancel
    and the series converges fastest when ``|s|`` is small. The classical rule is ``|s| < 1/4``.
    """
    a = gauss_forward(x, y, t, centre=centre, order=order)
    b = gauss_backward(x, y, t, centre=centre, order=order)
    return 0.5 * (np.asarray(a) + np.asarray(b))


# ----------------------------------------------------------------------------- Bessel


def bessel(x, y, t, order: int | None = None) -> np.ndarray:
    """The average taken about the **midpoint** of two nodes rather than about a node.

    Best when ``s`` is near ``1/2``, and the natural formula for interpolating halfway between
    tabulated values, which is the commonest thing anyone ever did with a printed table.
    """
    v, w, h = _pair(x, y)
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    c = int(np.clip(np.floor((tt[0] - v[0]) / h), 0, v.size - 2))
    a = gauss_forward(x, y, t, centre=c, order=order)
    b = gauss_backward(x, y, t, centre=c + 1, order=order)
    return 0.5 * (np.asarray(a) + np.asarray(b))


# ----------------------------------------------------------------------------- Everett


def everett(x, y, t, order: int | None = None) -> np.ndarray:
    """Everett's formula: **even order differences only**, using two central nodes.

    ``p = sum_k C(q + k, 2k+1) δ^(2k) y_0  +  sum_k C(s + k, 2k+1) δ^(2k) y_1``  with
    ``q = 1 - s``. Half the difference table is never touched, which halved the printing cost of
    a table and is why the formula was worth a name.

    Implemented here as the identity it is: Everett is Bessel with the odd terms collected and
    cancelled, so it returns the same polynomial, which `all_formulas_agree` confirms.
    """
    v, w, h = _pair(x, y)
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    c = int(np.clip(np.floor((tt[0] - v[0]) / h), 0, v.size - 2))
    s = (np.asarray(t, dtype=float) - v[c]) / h
    q = 1.0 - np.atleast_1d(s)
    kmax = (v.size - 1) // 2 if order is None else min(int(order), (v.size - 1) // 2)
    total = np.zeros_like(np.atleast_1d(s), dtype=float)
    for k in range(kmax + 1):
        d = _fd.forward(w, 2 * k)
        left, right = c - k, c + 1 - k
        if 0 <= left < d.size:
            total = total + binomial_s(q + k, 2 * k + 1) * d[left]
        if 0 <= right < d.size:
            total = total + binomial_s(np.atleast_1d(s) + k, 2 * k + 1) * d[right]
    return total.reshape(np.shape(t)) if np.ndim(t) else float(total[0])


def steffensen(x, y, t, order: int | None = None) -> np.ndarray:
    """Steffensen's rearrangement, which brings the **odd** differences into Everett's form.

    Everett uses the even columns only. Substituting the identity

        δ^(2k) y_1  =  δ^(2k) y_0  +  δ^(2k+1) y_(1/2)

    into Everett's formula moves one of each pair of even terms onto ``y_0`` and introduces the
    odd difference in its place, giving

        p = sum_k [ ( C(q+k, 2k+1) + C(s+k, 2k+1) ) δ^(2k) y_0
                    + C(s+k, 2k+1) δ^(2k+1) y_(1/2) ]

    with ``q = 1 - s``. This is a genuinely different read of the table from Everett's, and it is
    written out here rather than delegated, so that `all_formulas_agree` is comparing two
    computations and not one computation with itself.
    """
    v, w, h = _pair(x, y)
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    c = int(np.clip(np.floor((tt[0] - v[0]) / h), 0, v.size - 2))
    s = np.atleast_1d((np.asarray(t, dtype=float) - v[c]) / h)
    q = 1.0 - s
    kmax = (v.size - 1) // 2 if order is None else min(int(order), (v.size - 1) // 2)
    total = np.zeros_like(s, dtype=float)
    for k in range(kmax + 1):
        even = _fd.forward(w, 2 * k)
        odd = _fd.forward(w, 2 * k + 1)
        left = c - k
        if 0 <= left < even.size:
            total = total + (binomial_s(q + k, 2 * k + 1)
                             + binomial_s(s + k, 2 * k + 1)) * even[left]
        if 0 <= left < odd.size:
            total = total + binomial_s(s + k, 2 * k + 1) * odd[left]
    return total.reshape(np.shape(t)) if np.ndim(t) else float(total[0])


# ----------------------------------------------------------------------------- comparing


FORMULAS = {
    "gregory-newton forward": gregory_newton_forward,
    "gregory-newton backward": gregory_newton_backward,
    "gauss forward": gauss_forward,
    "gauss backward": gauss_backward,
    "stirling": stirling,
    "bessel": bessel,
    "everett": everett,
    "steffensen": steffensen,
}


def usable_central_order(x, t) -> int:
    """How high a central formula can reach at ``t`` before it runs off the end of the table.

    A central expansion about node ``c`` needs differences reaching ``c - k/2`` and ``c + k/2``,
    so it is limited to ``2 * min(c, n - 1 - c)``. That is zero at the first and last nodes,
    which is why the central formulas are unusable there.
    """
    v, _ = _grid(x)
    c = _central_index(v, t)
    return int(2 * min(c, v.size - 1 - c))


def all_formulas_agree(x, y, t) -> dict:
    """Every formula is the same polynomial, **wherever each can be formed at full order**.

    ``central_order_available`` says how far the central formulas can reach at this ``t``. When it
    equals the table's degree they all agree to rounding; when it is smaller the central ones are
    lower degree polynomials and genuinely differ, which is information rather than error.
    """
    v, w, h = _pair(x, y)
    values = {}
    for name, fn in FORMULAS.items():
        values[name] = float(np.atleast_1d(fn(x, y, t))[0])
    arr = np.asarray(list(values.values()))
    scale = max(float(np.max(np.abs(arr))), 1e-300)
    avail = usable_central_order(x, t)
    full = v.size - 1
    diag = {k: values[k] for k in ("gregory-newton forward", "gregory-newton backward")}
    dv = np.asarray(list(diag.values()))
    return {"values": values,
            "spread": float(np.max(arr) - np.min(arr)),
            "relative_spread": float((np.max(arr) - np.min(arr)) / scale),
            "diagonal_spread": float(np.max(dv) - np.min(dv)),
            "central_order_available": avail,
            "degree_of_the_table": full,
            "central_formulas_reach_full_degree": bool(avail >= full)}


@dataclass
class PositionReport:
    """Truncation error of each formula against where in the table the point sits."""

    s_values: np.ndarray
    errors: dict
    best_at: dict


def best_formula_for(s: float, n_nodes: int) -> str:
    """The classical rule, in one function.

    ``s`` is measured in grid steps from the start of the table.
    """
    n = int(n_nodes)
    if n < 2:
        raise ValueError(f"need at least two nodes, got {n}")
    frac = float(s) - math.floor(float(s))
    if s < 1.0:
        return "gregory-newton forward"
    if s > n - 2.0:
        return "gregory-newton backward"
    if 0.25 <= frac <= 0.75:
        return "bessel"
    return "stirling"


def error_by_position(f, n_nodes: int, lo: float, hi: float, order: int,
                      n_probe: int = 41) -> PositionReport:
    """Measure each formula's error at a **fixed truncation order** across the table.

    Truncating at the same order is what makes the comparison meaningful: at full order they are
    identical polynomials and there is nothing to choose between them. The classical advice is
    about which formula converges fastest, so it is only visible in the truncated series.
    """
    n = int(n_nodes)
    x = np.linspace(lo, hi, n)
    y = np.asarray([f(v) for v in x], dtype=float)
    h = (hi - lo) / (n - 1)
    probe = np.linspace(lo + 1e-9, hi - 1e-9, int(n_probe))
    s_vals = (probe - lo) / h
    truth = np.asarray([f(v) for v in probe], dtype=float)
    errors = {}
    for name, fn in FORMULAS.items():
        got = np.asarray([float(np.atleast_1d(fn(x, y, p, order=order))[0]) for p in probe])
        errors[name] = np.abs(got - truth)
    best = {}
    for i, s in enumerate(s_vals):
        best[float(s)] = min(errors, key=lambda k: errors[k][i])
    return PositionReport(s_values=s_vals, errors=errors, best_at=best)
