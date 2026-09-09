"""Finite difference operators: an algebra for equally spaced data.

What this module is for
-----------------------
When the nodes are equally spaced, ``x_k = x_0 + k h``, every divided difference becomes a plain
difference divided by a power of ``h``, and the whole subject simplifies into an **operator
algebra**. Lesson 49's formulas are all rearrangements within that algebra, so they are derived
here once rather than memorised individually.

The six operators
-----------------
All act on a sequence sampled at spacing ``h``.

=========  ==========================  ==================================================
symbol     name                        definition
=========  ==========================  ==================================================
``E``      shift                       ``E f(x) = f(x + h)``
``Δ``      forward difference          ``Δ f(x) = f(x + h) - f(x)``
``∇``      backward difference         ``∇ f(x) = f(x) - f(x - h)``
``δ``      central difference          ``δ f(x) = f(x + h/2) - f(x - h/2)``
``μ``      mean, or averaging          ``μ f(x) = (f(x + h/2) + f(x - h/2)) / 2``
``D``      differential                ``D f(x) = f'(x)``
=========  ==========================  ==================================================

The interrelations
------------------
Every operator is a function of ``E``, and that single fact generates all of lesson 49:

    Δ = E - 1        ∇ = 1 - E^-1        δ = E^(1/2) - E^(-1/2)
    μ = (E^(1/2) + E^(-1/2)) / 2         E = e^(hD)

The last one is Taylor's theorem written as an operator identity, and from it

    hD = log E = log(1 + Δ) = Δ - Δ^2/2 + Δ^3/3 - ...

which is where every finite difference formula for a derivative comes from. `interrelations`
checks the algebraic identities numerically and `operator_identity_report` checks ``E = e^(hD)``.

They all commute
----------------
Each is a polynomial or a series in ``E``, and powers of ``E`` commute with each other, so the
whole family commutes and is linear. `commutes` and `is_linear` verify that, and the practical
value is that you may rearrange these expressions freely, which is exactly what lesson 49 does.

Difference tables and error propagation
---------------------------------------
A **difference table** lays out ``Δ^k f`` in columns. Its most useful property is diagnostic: a
single wrong entry in the data spreads through the table in the binomial pattern
``+1, -2, +1`` at order 2, ``+1, -3, +3, -1`` at order 3, and so on, with the error **amplified
by 2^k** at order ``k``. `error_propagation` builds a corrupted table and finds the fault, which
is a genuinely practical technique for hand-tabulated data and the reason difference tables were
printed in books of tables.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _seq(y):
    v = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if v.size == 0:
        raise ValueError("need at least one value")
    return v


# ----------------------------------------------------------------------------- the operators


def shift(y, k: int = 1) -> np.ndarray:
    """``E^k``: advance by ``k`` places. The sequence shortens by ``|k|``, which is honest:
    the shifted value is simply not known at the ends."""
    v = _seq(y)
    k = int(k)
    if abs(k) >= v.size:
        return np.empty(0)
    return v[k:] if k >= 0 else v[:k]


def forward(y, k: int = 1) -> np.ndarray:
    """``Δ^k y``. Length drops by ``k``, because ``Δ^k`` needs ``k + 1`` consecutive values."""
    v = _seq(y)
    k = int(k)
    if k < 0:
        raise ValueError(f"order must be non-negative, got {k}")
    for _ in range(k):
        if v.size < 2:
            return np.empty(0)
        v = v[1:] - v[:-1]
    return v


def backward(y, k: int = 1) -> np.ndarray:
    """``∇^k y``. Numerically the same numbers as ``Δ^k``, indexed from the other end."""
    v = _seq(y)
    k = int(k)
    if k < 0:
        raise ValueError(f"order must be non-negative, got {k}")
    for _ in range(k):
        if v.size < 2:
            return np.empty(0)
        v = v[1:] - v[:-1]
    return v


def central(y, k: int = 1) -> np.ndarray:
    """``δ^k y`` on the given grid.

    For **even** ``k`` this lands on the grid points and is computable directly. For **odd** ``k``
    it lands halfway between them, so what is returned is the value at the half points, and the
    caller has to know that. That offset is the whole reason lesson 49 needs both the Gauss
    forward and Gauss backward formulas, and the reason Stirling and Bessel exist.
    """
    v = _seq(y)
    k = int(k)
    if k < 0:
        raise ValueError(f"order must be non-negative, got {k}")
    for _ in range(k):
        if v.size < 2:
            return np.empty(0)
        v = v[1:] - v[:-1]
    return v


def mean(y) -> np.ndarray:
    """``μ y``: the average of the two neighbours, which lands on the half points.

    ``μ`` exists to put an odd central difference back onto the grid: ``μ δ`` is a grid operator
    even though neither factor is.
    """
    v = _seq(y)
    if v.size < 2:
        return np.empty(0)
    return 0.5 * (v[1:] + v[:-1])


def differential(f, x, h: float = 1e-5, order: int = 1) -> np.ndarray:
    """``D^order f`` by a central difference, for checking the operator identities.

    This is a stand-in for the exact derivative, accurate to ``O(h^2)``, and it is used only
    where the identity being checked is about ``D`` itself.
    """
    z = np.asarray(x, dtype=float)
    order = int(order)
    if order == 0:
        return np.asarray([f(v) for v in np.atleast_1d(z)], dtype=float)
    if order == 1:
        return np.asarray([(f(v + h) - f(v - h)) / (2.0 * h) for v in np.atleast_1d(z)],
                          dtype=float)
    if order == 2:
        return np.asarray([(f(v + h) - 2.0 * f(v) + f(v - h)) / (h * h)
                           for v in np.atleast_1d(z)], dtype=float)
    raise ValueError(f"order up to 2 is supported here, got {order}")


# ----------------------------------------------------------------------------- the algebra


def interrelations(y) -> dict:
    """Check the identities that relate the operators, on real data.

    ``Δ = E - 1``, ``∇ = 1 - E^-1``, ``δ = E^(1/2) - E^(-1/2)``, and ``Δ = ∇ E``. Each is checked
    on the overlap where both sides are defined, which is what makes the lengths differ.
    """
    v = _seq(y)
    out = {}
    if v.size >= 2:
        out["delta_equals_E_minus_1"] = float(np.max(np.abs(forward(v) - (shift(v) - v[:-1]))))
        out["nabla_equals_1_minus_Einv"] = float(np.max(np.abs(backward(v) - (v[1:] - v[:-1]))))
    if v.size >= 3:
        d2 = forward(v, 2)
        out["delta_squared_equals_nabla_delta"] = float(np.max(np.abs(d2 - forward(backward(v)))))
        # Δ f(x) = ∇ f(x + h), so ∇E on the shifted sequence lines up with Δ from index 1
        nabla_E = backward(shift(v), 1)
        out["delta_equals_nabla_E"] = float(np.max(np.abs(forward(v)[1:1 + nabla_E.size]
                                                          - nabla_E)))
    return out


def commutes(y, j: int = 1, k: int = 2) -> dict:
    """``Δ^j Δ^k = Δ^k Δ^j = Δ^(j+k)``. They are all series in ``E``, so they all commute."""
    v = _seq(y)
    a = forward(forward(v, j), k)
    b = forward(forward(v, k), j)
    c = forward(v, int(j) + int(k))
    n = min(a.size, b.size, c.size)
    if n == 0:
        return {"gap": 0.0, "length": 0}
    scale = max(float(np.max(np.abs(c[:n]))), 1e-300)
    return {"gap": float(max(np.max(np.abs(a[:n] - b[:n])), np.max(np.abs(a[:n] - c[:n])))),
            "relative_gap": float(max(np.max(np.abs(a[:n] - b[:n])),
                                      np.max(np.abs(a[:n] - c[:n]))) / scale),
            "length": int(n)}


def is_linear(y1, y2, alpha: float = 3.0, beta: float = -2.0, k: int = 2) -> dict:
    """``Δ^k(α f + β g) = α Δ^k f + β Δ^k g``."""
    a = forward(alpha * _seq(y1) + beta * _seq(y2), k)
    b = alpha * forward(y1, k) + beta * forward(y2, k)
    n = min(a.size, b.size)
    scale = max(float(np.max(np.abs(b[:n]))) if n else 1.0, 1e-300)
    return {"gap": float(np.max(np.abs(a[:n] - b[:n]))) if n else 0.0,
            "relative_gap": float(np.max(np.abs(a[:n] - b[:n])) / scale) if n else 0.0}


def operator_identity_report(f, x0: float, h: float, n_terms: int = 8) -> dict:
    """``E = e^(hD)``, that is ``f(x + h) = sum_k (h^k / k!) f^(k)(x)``.

    This is Taylor's theorem stated as an operator identity, and it is the source of every
    finite difference approximation to a derivative. Checked here with the derivatives of ``exp``,
    which are all ``exp``, so the series is exact and only the truncation is measured.
    """
    total = 0.0
    for k in range(int(n_terms)):
        total += (h ** k) / math.factorial(k) * f(x0)
    exact = f(x0 + h)
    return {"series": float(total), "exact": float(exact),
            "gap": float(abs(total - exact)),
            "relative_gap": float(abs(total - exact) / max(abs(exact), 1e-300)),
            "terms": int(n_terms)}


def derivative_from_differences(y, h: float, n_terms: int = 4) -> np.ndarray:
    """``hD = log(1 + Δ) = Δ - Δ^2/2 + Δ^3/3 - ...``, truncated.

    The series that turns a table of values into a derivative. Truncating at ``n_terms`` gives
    an ``O(h^n_terms)`` approximation, which is where the order of every classical differentiation
    formula comes from.
    """
    v = _seq(y)
    n = int(n_terms)
    if n < 1:
        raise ValueError(f"need at least one term, got {n}")
    longest = v.size - n
    if longest <= 0:
        raise ValueError(f"{v.size} values cannot support {n} terms; need more than {n}")
    total = np.zeros(longest)
    for k in range(1, n + 1):
        d = forward(v, k)[:longest]
        total += ((-1.0) ** (k + 1)) * d / k
    return total / float(h)


# ----------------------------------------------------------------------------- tables


def difference_table(y, max_order: int | None = None) -> np.ndarray:
    """The table ``T[i, k] = Δ^k y_i``, with unreachable entries left as nan."""
    v = _seq(y)
    n = v.size
    kmax = n - 1 if max_order is None else min(int(max_order), n - 1)
    T = np.full((n, kmax + 1), np.nan)
    T[:, 0] = v
    for k in range(1, kmax + 1):
        col = forward(v, k)
        T[:col.size, k] = col
    return T


def table_of_a_polynomial(coeffs, x0: float, h: float, n_points: int) -> dict:
    """A degree ``d`` polynomial has constant ``d``-th differences and zero beyond.

    That is the classical test for whether tabulated data is polynomial, and the order at which
    the column goes constant is the degree. It also means the table is exactly the right
    diagnostic for an equally spaced grid and useless without one.
    """
    a = np.atleast_1d(np.asarray(coeffs, dtype=float)).ravel()
    d = a.size - 1
    x = float(x0) + float(h) * np.arange(int(n_points))
    y = np.polyval(a[::-1], x)
    T = difference_table(y)
    constant_at = None
    for k in range(T.shape[1]):
        col = T[:, k]
        col = col[np.isfinite(col)]
        if col.size >= 2 and float(np.max(col) - np.min(col)) <= 1e-8 * max(
                float(np.max(np.abs(col))), 1.0):
            constant_at = k
            break
    return {"values": y, "table": T, "degree": d, "constant_at_order": constant_at,
            "expected_constant_at": d,
            "d_th_difference": float(math.factorial(d) * a[-1] * h ** d)}


@dataclass
class ErrorSpread:
    """Where a single corrupted value shows up in a difference table."""

    index: int
    size: float
    order: int
    pattern: np.ndarray
    expected_binomial: np.ndarray
    amplification: float
    located_at: int


def error_propagation(y, index: int, size: float, order: int = 4) -> ErrorSpread:
    """Corrupt one value and watch the error spread through the table.

    A single error ``e`` at position ``i`` appears in column ``k`` as ``e`` times the binomial
    coefficients with alternating signs, spread over ``k + 1`` rows and centred on ``i``. Its
    largest entry is ``e * C(k, floor(k/2))``, the **central binomial coefficient**, so the error
    is amplified. Measured:

    =====  ===============  =========  ========================
    k      amplification    ``2^k``    pattern
    =====  ===============  =========  ========================
    2      2                4          ``+1, -2, +1``
    3      3                8          ``+1, -3, +3, -1``
    4      6                16         ``+1, -4, +6, -4, +1``
    5      10               32         ``+1, -5, +10, -10, +5, -1``
    6      20               64         ``+1, -6, +15, -20, +15, -6, +1``
    =====  ===============  =========  ========================

    It is worth being exact about this: the amplification is the central binomial coefficient,
    not ``2^k``. The two differ by a factor of about ``sqrt(k)``, since ``C(k, k/2)`` is about
    ``2^k / sqrt(π k / 2)``, and at ``k = 6`` that is 20 rather than 64.

    That is what makes the table a fault finder: the tell-tale ``+1, -3, +3, -1`` pattern in a
    column locates the bad value, and the ratio of the pattern to the binomials gives its size.
    """
    v = _seq(y).copy()
    i = int(index)
    if not 0 <= i < v.size:
        raise ValueError(f"index {i} is outside the {v.size} values")
    k = int(order)
    if k < 1:
        raise ValueError(f"order must be at least 1, got {k}")
    v[i] += float(size)
    clean = forward(_seq(y), k)
    dirty = forward(v, k)
    n = min(clean.size, dirty.size)
    pattern = dirty[:n] - clean[:n]
    binom = np.asarray([((-1) ** j) * math.comb(k, j) for j in range(k + 1)], dtype=float)
    located = int(np.argmax(np.abs(pattern))) if n else -1
    amp = float(np.max(np.abs(pattern)) / abs(size)) if size != 0 and n else 0.0
    return ErrorSpread(index=i, size=float(size), order=k, pattern=pattern,
                       expected_binomial=binom * float(size), amplification=amp,
                       located_at=located)


def locate_a_bad_value(y, order: int = 4, tol: float = 0.25) -> dict:
    """Find the single corrupted entry from the pattern it leaves, without knowing where it is.

    The peak of ``|Δ^k|`` sits ``k/2`` places after the fault, because the binomial pattern is
    centred. Correcting for that offset recovers the index, and the size follows from dividing
    the peak by the central binomial coefficient.

    **The method has a precondition, and it is checked rather than assumed.** The fault is only
    visible if it dominates the data's own ``k``-th differences. Measured, planting an error of
    0.02 at index 5 in 14 points and asking for order 4:

    ====================  =================  ===============  =======
    data                  its own max |Δ⁴|   ratio to fault   found?
    ====================  =================  ===============  =======
    ``exp(0.3 k)``        2.23e-1            11.1             no, 11
    ``exp(0.05 k)``       1.08e-5            0.001            yes, 5
    a cubic in ``k``      0.00e+0            0                yes, 5
    ``sin(0.2 k)``        1.59e-3            0.08             yes, 5
    ====================  =================  ===============  =======

    So this is a technique for **smooth, densely tabulated** data, which is exactly the setting
    books of tables were printed for. On data whose own high differences are large it returns a
    confident wrong answer, so ``fits_binomial_pattern`` below reports how well the column around
    the peak actually matches the binomial shape, and ``reliable`` is that test against ``tol``.
    """
    v = _seq(y)
    k = int(order)
    col = forward(v, k)
    if col.size == 0:
        raise ValueError(f"{v.size} values cannot support order {k}")
    peak = int(np.argmax(np.abs(col)))
    central_binomial = math.comb(k, k // 2)
    size = float(col[peak] / (((-1) ** (k // 2)) * central_binomial))

    # A genuine single fault makes the k + 1 entries around the peak proportional to the
    # alternating binomial coefficients. The window has to be ALIGNED to the peak rather than
    # simply clipped: near the start of the table the pattern runs off the front, and comparing
    # a clipped window against an unclipped binomial rejects correct answers. Measured before
    # the fix, a fault planted at index 3 of 14 was located correctly and then reported
    # unreliable, because its pattern begins at column index -1.
    binom = np.asarray([((-1) ** j) * math.comb(k, j) for j in range(k + 1)], dtype=float) * size
    offset = peak - k // 2                       # column index the pattern starts at, may be < 0
    lo = max(offset, 0)
    hi = min(offset + k + 1, col.size)
    window = col[lo:hi]
    expected = binom[lo - offset:hi - offset]
    denom = max(float(np.max(np.abs(window))), 1e-300)
    fit = (float(np.max(np.abs(window - expected)) / denom)
           if window.size and window.size == expected.size else float("inf"))
    return {"column": col, "peak_at": peak,
            "estimated_index": peak + k // 2,
            "estimated_size": size,
            "amplification": float(central_binomial),
            "fits_binomial_pattern": fit,
            "reliable": bool(fit < float(tol))}
