"""Error measures, error propagation and condition numbers.

These four words get used loosely everywhere else. In this course they mean exactly one
thing each, and this module is where that is pinned down.

- **absolute error** how far the computed value is from the true one
- **relative error** that distance, scaled by the size of the true value
- **forward error** how wrong the answer is
- **backward error** how much the question would have to change for the computed answer to
  be exactly right

**Conditioning** is a property of the problem. **Stability** is a property of the algorithm.
A well-conditioned problem solved by an unstable algorithm gives a bad answer, and so does
an ill-conditioned problem solved by a perfectly stable one. Telling those two cases apart
is most of what error analysis is for.

The governing inequality, used throughout the course:

    forward error  <=  condition number  x  backward error

Used by lessons 04, 06, 12 and 18.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# ---------------------------------------------------------------- basic measures


def absolute_error(exact, approx) -> float:
    """|exact - approx|, elementwise maximum for arrays."""
    e = np.asarray(exact, dtype=float)
    a = np.asarray(approx, dtype=float)
    return float(np.max(np.abs(e - a)))


def relative_error(exact, approx) -> float:
    """Absolute error divided by the size of the exact value.

    Returns infinity when the exact value is zero and the approximation is not, which is
    the honest answer: relative error is not defined there.
    """
    e = np.asarray(exact, dtype=float)
    a = np.asarray(approx, dtype=float)
    denom = float(np.max(np.abs(e)))
    num = float(np.max(np.abs(e - a)))
    if denom == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / denom


def percentage_error(exact, approx) -> float:
    """Relative error expressed as a percentage."""
    return 100.0 * relative_error(exact, approx)


def significant_digits(exact, approx) -> float:
    """How many correct significant decimal digits the approximation has.

    Defined as -log10(relative error). A relative error of 1e-9 is about 9 correct digits.
    Returns 0.0 when the approximation has no correct digits, and 16.0 (the practical
    ceiling in double precision) when the values agree exactly.
    """
    r = relative_error(exact, approx)
    if r == 0.0:
        return 16.0
    if not np.isfinite(r) or r >= 1.0:
        return 0.0
    return float(-np.log10(r))


# ---------------------------------------------------------------- forward and backward


def forward_error(exact, approx) -> float:
    """How wrong the answer is. Same number as the absolute error, different name.

    The separate name exists because it sits opposite `backward_error` in every argument
    in this course, and calling both of them "the error" is how people get confused.
    """
    return absolute_error(exact, approx)


def backward_error_root(f: Callable[[float], float], x_hat: float) -> float:
    """Backward error of a computed root.

    If x_hat were the exact root of some nearby problem, that problem would be
    `f(x) - f(x_hat) = 0`. So the size of the perturbation needed is |f(x_hat)|.

    A tiny backward error with a large forward error is the signature of an
    ill-conditioned problem, not a broken algorithm. Lesson 12 makes that visible with the
    Wilkinson polynomial.
    """
    return float(abs(f(float(x_hat))))


def backward_error_linear_system(A, b, x_hat) -> float:
    """Relative backward error of a computed solution to A x = b.

    This is the Rigal-Gaches result: the smallest relative perturbation to A and b for
    which x_hat is the exact solution equals

        ||r|| / (||A|| ||x_hat|| + ||b||)

    with r = b - A x_hat. A backward stable solver returns a value here on the order of
    machine epsilon, no matter how ill-conditioned A is.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    x_hat = np.asarray(x_hat, dtype=float).ravel()
    r = b - A @ x_hat
    denom = np.linalg.norm(A, np.inf) * np.linalg.norm(x_hat, np.inf) + np.linalg.norm(
        b, np.inf
    )
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(r, np.inf) / denom)


def residual(A, b, x_hat):
    """r = b - A x_hat. Small residual does not by itself mean small error."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    x_hat = np.asarray(x_hat, dtype=float).ravel()
    return b - A @ x_hat


# ---------------------------------------------------------------- conditioning


def condition_number_scalar(
    f: Callable[[float], float], x: float, h: float | None = None
) -> float:
    """Relative condition number of evaluating f at x.

        kappa(x) = |x f'(x) / f(x)|

    This says how much a relative change in the input is multiplied on its way to the
    output. A value near 1 is harmless. A value of 1e8 means you lose eight digits just by
    asking the question, before any algorithm runs.

    The derivative is taken by a central difference with the step size that balances
    truncation against roundoff, which lesson 60 derives as roughly eps**(1/3).
    """
    x = float(x)
    if h is None:
        h = np.cbrt(np.finfo(float).eps) * max(abs(x), 1.0)
    fx = float(f(x))
    dfx = (float(f(x + h)) - float(f(x - h))) / (2.0 * h)
    if fx == 0.0:
        return float("inf") if dfx != 0.0 else 0.0
    return float(abs(x * dfx / fx))


def condition_number_root(
    f: Callable[[float], float], r: float, h: float | None = None
) -> float:
    """Sensitivity of a root r of f to a perturbation of f itself.

    If f is changed to f + eps*g, the root moves by about `eps * g(r) / f'(r)`. So the
    amplification factor is 1 / |f'(r)|. A root where the curve is nearly flat is a root
    you cannot locate accurately, no matter which method you use.
    """
    r = float(r)
    if h is None:
        h = np.cbrt(np.finfo(float).eps) * max(abs(r), 1.0)
    dfr = (float(f(r + h)) - float(f(r - h))) / (2.0 * h)
    if dfr == 0.0:
        return float("inf")
    return float(1.0 / abs(dfr))


def error_magnification(exact, approx, rel_backward: float) -> float:
    """Observed forward error divided by backward error.

    This is the amplification the problem actually applied on this instance. It is a lower
    bound on the condition number, and it is what Sauer calls the error magnification
    factor.
    """
    if rel_backward == 0.0:
        return float("inf") if relative_error(exact, approx) > 0 else 0.0
    return relative_error(exact, approx) / rel_backward


# ---------------------------------------------------------------- propagation


def propagate_absolute(
    f: Callable[..., float], x, dx, h: float | None = None
) -> float:
    """First order bound on the absolute error of f(x) given absolute errors dx in x.

        |df|  <=  sum_i |df/dx_i| |dx_i|

    Works for one variable or several. Partial derivatives are taken by central
    differences. This is the general form of the propagation rules in lesson 04.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    dx = np.atleast_1d(np.asarray(dx, dtype=float))
    if x.shape != dx.shape:
        raise ValueError(f"x and dx must have the same shape, got {x.shape} and {dx.shape}")
    if h is None:
        h = np.cbrt(np.finfo(float).eps) * np.maximum(np.abs(x), 1.0)
    else:
        h = np.full_like(x, float(h))

    total = 0.0
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h[i]
        xm[i] -= h[i]
        partial = (float(f(*xp)) - float(f(*xm))) / (2.0 * h[i])
        total += abs(partial) * abs(dx[i])
    return float(total)


def propagate_relative(f: Callable[..., float], x, rel_dx, h: float | None = None) -> float:
    """First order bound on the relative error of f(x) given relative errors in x."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    rel_dx = np.atleast_1d(np.asarray(rel_dx, dtype=float))
    abs_dx = np.abs(x) * rel_dx
    fx = float(f(*x))
    abs_df = propagate_absolute(f, x, abs_dx, h)
    if fx == 0.0:
        return float("inf") if abs_df > 0 else 0.0
    return float(abs_df / abs(fx))


#: Exact first order relative-error rules for the four arithmetic operations.
#: Each takes the relative errors of the two operands and returns a bound on the relative
#: error of the result. Addition and subtraction also need the values themselves, because
#: their behaviour depends on cancellation, which is the point of lesson 05.
def rel_error_product(rel_a: float, rel_b: float) -> float:
    """Relative errors add under multiplication, to first order."""
    return abs(rel_a) + abs(rel_b)


def rel_error_quotient(rel_a: float, rel_b: float) -> float:
    """Relative errors add under division too, to first order."""
    return abs(rel_a) + abs(rel_b)


def rel_error_sum(a: float, b: float, rel_a: float, rel_b: float) -> float:
    """Relative error of a + b.

    The bound is `(|a| |rel_a| + |b| |rel_b|) / |a + b|`. When a + b is tiny compared with
    a and b themselves, the denominator is small and the bound explodes. That is
    catastrophic cancellation, stated as a formula.
    """
    denom = abs(a + b)
    num = abs(a) * abs(rel_a) + abs(b) * abs(rel_b)
    if denom == 0.0:
        return float("inf") if num > 0 else 0.0
    return float(num / denom)


def cancellation_factor(a: float, b: float) -> float:
    """How much subtracting b from a magnifies relative error: (|a| + |b|) / |a - b|.

    A factor of 1 means no cancellation. A factor of 1e10 means you are about to lose ten
    digits, and no cleverness in the subtraction itself can prevent it.
    """
    denom = abs(a - b)
    if denom == 0.0:
        return float("inf")
    return float((abs(a) + abs(b)) / denom)


def diagnose_root(
    f: Callable[[float], float],
    r_hat: float,
    df: Callable[[float], float] | None = None,
) -> dict:
    """Full accuracy report for a computed root, following lesson 06 section 9.

    A residual on its own says nothing. It is a **backward** error: it tells you the
    computed point nearly solves the problem. What you want is the **forward** error, how
    close `r_hat` is to the true root, and the two are linked by

        forward error  <~  condition number  x  backward error.

    So this function reports both numbers and their product, and then states a verdict, so
    that nobody can look at a small residual and stop reading.

    Returns a dict with keys `residual`, `condition`, `forward_bound`, `at_noise_floor`
    and `verdict`.

    A zero derivative means a multiple root, where the whole linear analysis collapses and
    lesson 12's `u**(1/m)` limit applies instead. That case is reported separately rather
    than as an infinite condition number with no explanation.
    """
    r_hat = float(r_hat)
    u = float(np.finfo(float).eps) / 2

    residual_val = abs(float(f(r_hat)))
    if df is None:
        h = np.cbrt(np.finfo(float).eps) * max(abs(r_hat), 1.0)
        slope = abs((float(f(r_hat + h)) - float(f(r_hat - h))) / (2.0 * h))
    else:
        slope = abs(float(df(r_hat)))

    noise = u * max(1.0, abs(r_hat))

    if slope == 0.0:
        return {
            "residual": residual_val,
            "condition": float("inf"),
            "forward_bound": float("inf"),
            "at_noise_floor": False,
            "verdict": ("derivative is zero: this is a multiple root, so the achievable "
                        "accuracy is u**(1/m), not u. See lesson 12."),
        }

    condition = 1.0 / slope
    bound = condition * residual_val

    if bound <= 10 * noise:
        verdict = "as accurate as double precision allows for this problem"
    elif bound <= 1e-8:
        verdict = f"usable: forward error is at most about {bound:.1e}"
    else:
        verdict = (f"WARNING: the residual is small but the condition number is "
                   f"{condition:.2e}, so the forward error may be as large as {bound:.1e}")

    return {
        "residual": residual_val,
        "condition": condition,
        "forward_bound": bound,
        "at_noise_floor": bool(bound <= 10 * noise),
        "verdict": verdict,
    }
