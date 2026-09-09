"""Polynomial evaluation in nested form, and the operations built on it.

A polynomial written out as

    p(x) = a_n x**n + ... + a_1 x + a_0

costs about n additions and 2n-1 multiplications to evaluate directly. Rewritten in nested
form,

    p(x) = (...((a_n x + a_{n-1}) x + a_{n-2}) x + ...) x + a_0

it costs n additions and n multiplications. That is Horner's rule. It is the first
algorithm in the course because it makes the point that how you arrange a calculation
changes both its cost and its accuracy, even though the mathematics is identical.

Coefficient order matches `numpy.polyval`: **descending**, so `[2, -6, 2, -1]` means
`2x**3 - 6x**2 + 2x - 1`.

Used by lessons 01, 08, 13 and every lesson that touches interpolation.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- evaluation


def naive_eval(coeffs, x):
    """Evaluate a polynomial the obvious way, forming each power separately.

    Kept so it can be compared against `horner` on cost and on accuracy. Do not use it for
    real work.

    Cost: n additions and about 2n multiplications for degree n.
    """
    c = np.asarray(coeffs, dtype=float)
    n = c.size - 1
    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x, dtype=float)
    for i, a in enumerate(c):
        total = total + a * x ** (n - i)
    return total


def naive_eval_recompute(coeffs, x):
    """The truly naive version, recomputing each power from scratch by repeated products.

    This is what someone writes before they think about it. It exists to make the
    operation count in lesson 08 concrete: this one really does use about n**2/2
    multiplications.
    """
    c = np.asarray(coeffs, dtype=float)
    n = c.size - 1
    x = float(x)
    total = 0.0
    for i, a in enumerate(c):
        power = 1.0
        for _ in range(n - i):
            power *= x
        total += a * power
    return total


def horner(coeffs, x):
    """Evaluate a polynomial by Horner's rule (nested multiplication).

    Cost: n multiplications and n additions for degree n. Works elementwise on arrays.

    Equivalent to `numpy.polyval(coeffs, x)`, which uses the same algorithm.

    >>> float(horner([2, -6, 2, -1], 3.0))
    5.0
    """
    c = np.asarray(coeffs, dtype=float)
    if c.size == 0:
        raise ValueError("need at least one coefficient")
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, c[0], dtype=float)
    for a in c[1:]:
        result = result * x + a
    return result


def horner_with_derivative(coeffs, x: float) -> tuple[float, float]:
    """Evaluate p(x) and p'(x) together in one pass.

    Running Horner's rule on the intermediate values produces the derivative for free,
    because the intermediate quotients of synthetic division are the coefficients of
    `p(x) / (x - x0)`, and the derivative at x0 is that quotient evaluated at x0.

    This is what makes Newton's method on a polynomial cheap: value and slope for the cost
    of roughly two Horner passes. Lesson 13 uses it for Birge-Vieta.

    >>> horner_with_derivative([1, 0, -2], 1.5)   # p = x^2 - 2
    (0.25, 3.0)
    """
    c = np.asarray(coeffs, dtype=float)
    x = float(x)
    p = float(c[0])
    dp = 0.0
    for a in c[1:]:
        dp = dp * x + p
        p = p * x + a
    return float(p), float(dp)


def horner_all_derivatives(coeffs, x: float, order: int) -> list[float]:
    """p(x), p'(x), p''(x), ... up to the requested order, by repeated deflation.

    Each synthetic division peels off one derivative. After k rounds the leading value is
    the k-th Taylor coefficient at x, so the k-th derivative is that times k factorial.
    This is the change-of-center operation used in lesson 43.
    """
    from math import factorial

    c = list(np.asarray(coeffs, dtype=float))
    x = float(x)
    out = []
    for k in range(order + 1):
        if not c:
            out.append(0.0)
            continue
        quotient, remainder = synthetic_division(c, x)
        out.append(float(remainder) * factorial(k))
        c = quotient
    return out


# ------------------------------------------------- pure Python, for counting operations
#
# The NumPy versions above are the ones to use for real work. These two do the identical
# arithmetic with plain Python numbers, so `nalib.cost.OpCounter` can wrap the input and
# count every operation that actually happens. NumPy would hide the operations inside
# compiled code, where a counter cannot see them.


def horner_scalar(coeffs, x):
    """Horner's rule with plain Python arithmetic. Countable by `nalib.cost.OpCounter`.

    Uses exactly n multiplications and n additions for degree n.
    """
    coeffs = list(coeffs)
    if not coeffs:
        raise ValueError("need at least one coefficient")
    result = coeffs[0]
    for a in coeffs[1:]:
        result = result * x + a
    return result


def naive_scalar(coeffs, x):
    """Direct evaluation with plain Python arithmetic, recomputing every power.

    Uses n additions and about n(n+1)/2 + n multiplications for degree n, so its cost grows
    quadratically while Horner's grows linearly.
    """
    coeffs = list(coeffs)
    n = len(coeffs) - 1
    total = 0.0
    for i, a in enumerate(coeffs):
        power = 1.0
        for _ in range(n - i):
            power = power * x
        total = total + a * power
    return total


# ---------------------------------------------------------------- division


def synthetic_division(coeffs, r: float) -> tuple[list[float], float]:
    """Divide p by (x - r). Returns (quotient coefficients, remainder).

    The remainder equals p(r), which is the remainder theorem. So synthetic division and
    Horner evaluation are literally the same arithmetic, viewed two ways.

    >>> q, rem = synthetic_division([1, -6, 11, -6], 1.0)   # roots 1, 2, 3
    >>> [float(v) for v in q], float(rem)
    ([1.0, -5.0, 6.0], 0.0)
    """
    c = np.asarray(coeffs, dtype=float)
    if c.size == 0:
        raise ValueError("need at least one coefficient")
    r = float(r)
    out = [float(c[0])]
    for a in c[1:]:
        out.append(out[-1] * r + float(a))
    remainder = out.pop()
    return out, remainder


def deflate(coeffs, root: float) -> list[float]:
    """Remove a known root, returning the reduced polynomial.

    Deflation is convenient and slightly dangerous. Each removed root carries its own
    error into the reduced polynomial, so the roots found later are less accurate than the
    ones found first. Lesson 13 measures how much.
    """
    quotient, _remainder = synthetic_division(coeffs, root)
    return quotient


# ---------------------------------------------------------------- utilities


def derivative_coeffs(coeffs) -> np.ndarray:
    """Coefficients of p'(x), in the same descending order.

    >>> derivative_coeffs([2, -6, 2, -1])
    array([  6., -12.,   2.])
    """
    c = np.asarray(coeffs, dtype=float)
    n = c.size - 1
    if n <= 0:
        return np.array([0.0])
    powers = np.arange(n, 0, -1, dtype=float)
    return c[:-1] * powers


def flop_count_naive(n: int) -> tuple[int, int]:
    """Additions and multiplications performed by `naive_scalar` on degree n.

    These are the operations that **involve x**, which is what `nalib.cost.OpCounter`
    measures and what actually depends on the input.

    For n >= 1:

    - **multiplications**: n(n+1)/2 to build the powers x, x^2, ..., x^n by repeated
      products, plus n more to scale each of those powers by its coefficient. The constant
      term's coefficient multiplies the plain number 1.0, which does not involve x, so it
      is not in this count.
    - **additions**: n+1, not n, because the running total starts at 0.0 and every one of
      the n+1 terms is added to it.

    For n = 0 the polynomial is a constant and its value does not depend on x at all, so
    the count is (0, 0).

    Checked against a real measured count in lesson 08 and in `tests/test_polynomials.py`.
    """
    if n <= 0:
        return 0, 0
    return n + 1, n * (n + 1) // 2 + n


def flop_count_horner(n: int) -> tuple[int, int]:
    """Additions and multiplications performed by `horner_scalar` on degree n.

    Exactly n of each, with no hidden terms. For n = 0 the value is the constant itself and
    no arithmetic happens, giving (0, 0), which the formula already produces.
    """
    return n, n


def from_roots(roots) -> np.ndarray:
    """Expanded coefficients of the monic polynomial with the given roots.

    Equivalent to `numpy.poly`. Used to build the Wilkinson polynomial in lesson 12, where
    the whole point is that this expansion is a numerically terrible way to describe a
    polynomial whose roots you already know.
    """
    coeffs = np.array([1.0])
    for r in np.asarray(roots, dtype=float).ravel():
        shifted = np.append(coeffs, 0.0)
        shifted[1:] -= r * coeffs
        coeffs = shifted
    return coeffs
