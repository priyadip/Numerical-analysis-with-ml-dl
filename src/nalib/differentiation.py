"""Numerical differentiation: the one operation where refining the step makes things worse.

The trade that defines the subject
----------------------------------
Every other method in this course improves as the discretisation refines. Differentiation does
not, and the reason is a subtraction.

The forward quotient ``(f(x+h) - f(x)) / h`` has two errors pulling in opposite directions:

    truncation:  h |f''| / 2          shrinks as h shrinks
    roundoff:    2 eps |f| / h        GROWS as h shrinks

Their sum is minimised at ``h* = 2 sqrt(eps |f| / |f''|)``, roughly ``sqrt(eps)``, and the best
achievable error is about ``sqrt(eps)``, which is ``1.5e-8``. **Half the digits are gone, and no
amount of care recovers them.** `optimal_step` measures the whole curve and finds the minimum.

The central quotient does better, ``h* ~ eps^(1/3)`` and an error of ``eps^(2/3) ~ 6e-11``, and it
costs one extra evaluation. Higher order formulas push the exponent further and never reach
``eps``.

Two ways out, and only one of them is free
------------------------------------------
`richardson` combines quotients at several step sizes to cancel the leading truncation terms. It
raises the order and does nothing about the roundoff, so it moves the optimum but does not remove
it.

`complex_step` removes the subtraction entirely. For a real analytic ``f``,

    f'(x) = Im( f(x + i h) ) / h + O(h^2)

with **no subtraction of nearly equal quantities**, so the roundoff term vanishes and ``h`` can be
taken as small as ``1e-200``. The measured error is ``1e-16``, which is full machine precision, and
it is the only method here that reaches it. The price is that ``f`` must be evaluated in complex
arithmetic and must be analytic, so ``abs``, ``max`` and any branch on the sign of a real quantity
break it.

Where the formulas come from
----------------------------
Two derivations, and they agree. `from_taylor` solves the linear system that makes a weighted
combination of samples match a Taylor expansion to the required order. `from_interpolation`
differentiates the interpolating polynomial through the same points. `two_derivations_agree`
checks they give the same coefficients, which they must, because the interpolating polynomial is
the unique one matching the data and Taylor matching is the same set of conditions.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- the formulas


def from_taylor(offsets, order: int = 1) -> np.ndarray:
    """Finite difference weights on arbitrary offsets, by matching Taylor terms.

    The weights ``c_j`` for ``f^(order)(x) ~ sum_j c_j f(x + o_j h) / h^order`` solve the
    Vandermonde system

        sum_j c_j o_j^k / k!  =  delta(k, order),    k = 0 ... len(offsets)-1

    This is the derivation written literally, one linear solve. It is the clearest route to the
    weights and the worst way to compute them: the matrix is a Vandermonde on the offsets, whose
    condition number grows exponentially in the stencil width.

    Measured against exact rational arithmetic, on symmetric equally spaced offsets rescaled to
    span [-1, 1], which is what `differentiation_matrix` passes in, the relative error is

        offsets    from_taylor    from_interpolation    fornberg
             11        2.7e-12               4.3e-16     6.4e-16
             21        1.6e-07               1.2e-15     7.8e-16
             25        3.7e-04               5.2e-16     1.1e-15
             31        1.5e-01               9.2e-16     9.5e-16
             41        2.9e+02               9.3e-16     8.1e-16

    So this route has lost every correct digit by 31 points while the other two have lost none.
    The exponent depends on how the offsets are scaled, which is itself worth knowing: the same
    stencils written on integer offsets -k..k degrade far more slowly, reaching only 7.3e-05 at
    41 points, because scaling all the columns of a Vandermonde changes its conditioning.

    Use this to see where the numbers come from and to check small stencils. Use `fornberg` to
    compute them. `derivations_agree` measures the gap between the three routes.

    Returns the weights, which are multiplied by ``h^-order`` at use.
    """
    o = np.atleast_1d(np.asarray(offsets, dtype=float)).ravel()
    m = int(order)
    if m < 0:
        raise ValueError(f"derivative order must be non-negative, got {m}")
    if o.size <= m:
        raise ValueError(f"order {m} needs at least {m + 1} offsets, got {o.size}")
    if np.unique(o).size != o.size:
        raise ValueError("the offsets must be distinct")
    n = o.size
    A = np.stack([o ** k / math.factorial(k) for k in range(n)])
    rhs = np.zeros(n)
    rhs[m] = 1.0
    return np.linalg.solve(A, rhs)


def from_interpolation(offsets, order: int = 1) -> np.ndarray:
    """The same weights, by differentiating the interpolating polynomial.

    Build the Lagrange basis on the offsets, differentiate it ``order`` times, and evaluate at
    0. That is a completely different route to the same numbers, and `two_derivations_agree`
    checks it rather than asserting it.
    """
    o = np.atleast_1d(np.asarray(offsets, dtype=float)).ravel()
    m = int(order)
    if o.size <= m:
        raise ValueError(f"order {m} needs at least {m + 1} offsets, got {o.size}")
    n = o.size
    weights = np.empty(n)
    for j in range(n):
        # the Lagrange basis polynomial for node j, in the power basis
        coefficients = np.zeros(n)
        coefficients[0] = 1.0
        degree = 0
        denominator = 1.0
        for k in range(n):
            if k == j:
                continue
            shifted = np.zeros(n)
            shifted[1:degree + 2] = coefficients[:degree + 1]
            coefficients = shifted - o[k] * np.concatenate([coefficients[:degree + 1],
                                                            np.zeros(n - degree - 1)])
            degree += 1
            denominator *= (o[j] - o[k])
        # the m-th derivative at 0 is m! times the coefficient of x^m
        weights[j] = math.factorial(m) * coefficients[m] / denominator
    return weights


def fornberg(offsets, order: int = 1) -> np.ndarray:
    """The same weights again, by Fornberg's recursion, which is the one to actually use.

    The Vandermonde solve in `from_taylor` and the Lagrange construction in `from_interpolation`
    both lose accuracy fast as the stencil grows, because both form quantities that cancel. This
    builds the weights for ``0, 1, ... order`` on ``1, 2, ... n`` offsets by a recursion whose
    only operations are differences and quotients of node positions. There is no linear system
    and no explicit polynomial, so nothing large is subtracted to make something small.

    It is not the only stable route. `from_interpolation` holds up just as well, at 1e-15 for
    every width measured. The difference between the two is cost, not accuracy: this recursion
    is O(n^2) for the whole stencil, while building and differentiating each Lagrange basis
    polynomial separately is O(n^3). The table in `from_taylor` has the measured numbers for all
    three.

    Reference: Bengt Fornberg, "Calculation of weights in finite difference formulas",
    SIAM Review 40 (1998) 685-691. The variable names follow the paper's Table 1 so the code can
    be read against it.
    """
    o = np.atleast_1d(np.asarray(offsets, dtype=float)).ravel()
    m = int(order)
    if m < 0:
        raise ValueError(f"derivative order must be non-negative, got {m}")
    if o.size <= m:
        raise ValueError(f"order {m} needs at least {m + 1} offsets, got {o.size}")
    if np.unique(o).size != o.size:
        raise ValueError("the offsets must be distinct")
    n = o.size
    c = np.zeros((n, m + 1))
    c1 = 1.0
    c4 = o[0]
    c[0, 0] = 1.0
    for i in range(1, n):
        top = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = o[i]
        for j in range(i):
            c3 = o[i] - o[j]
            c2 = c2 * c3
            if j == i - 1:
                for k in range(top, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(top, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, m]


def derivations_agree(offsets, order: int = 1) -> dict:
    """All three routes to the weights, side by side, with the gap between them measured.

    They agree to rounding on the small stencils and diverge on wide ones, and the one that
    stays right is `fornberg`. Reporting the spread is the point: a reader who only ever sees
    `from_taylor` has no way to know when it has stopped working.
    """
    o = np.atleast_1d(np.asarray(offsets, dtype=float)).ravel()
    taylor = from_taylor(o, order)
    lagrange = from_interpolation(o, order)
    stable = fornberg(o, order)
    scale = max(float(np.max(np.abs(stable))), np.finfo(float).tiny)
    return {"from_taylor": taylor,
            "from_interpolation": lagrange,
            "fornberg": stable,
            "taylor_gap": float(np.max(np.abs(taylor - stable))) / scale,
            "interpolation_gap": float(np.max(np.abs(lagrange - stable))) / scale,
            "agree": bool(np.max(np.abs(taylor - stable)) <= 1e-8 * scale
                          and np.max(np.abs(lagrange - stable)) <= 1e-8 * scale)}


def two_derivations_agree(offsets, order: int = 1) -> dict:
    """Taylor matching against differentiating the interpolant, on the same stencil."""
    a = from_taylor(offsets, order)
    b = from_interpolation(offsets, order)
    scale = max(float(np.max(np.abs(a))), 1e-300)
    return {"from_taylor": a, "from_interpolation": b,
            "worst_gap": float(np.max(np.abs(a - b))),
            "relative_gap": float(np.max(np.abs(a - b))) / scale,
            "agree": bool(float(np.max(np.abs(a - b))) < 1e-8 * scale)}


def accuracy_order(offsets, order: int = 1, tol: float = 1e-8) -> int:
    """The order of accuracy the stencil achieves, by finding the first Taylor term it misses.

    Applying the weights to the monomial ``x^k`` gives ``h^{k-m} sum_j c_j o_j^k``, and the exact
    ``m``th derivative of ``x^k`` at 0 is ``m!`` when ``k = m`` and **zero otherwise**. So the
    stencil is exact on ``x^k`` exactly when

        sum_j c_j o_j^k  =  m!  if k == m,  else 0

    and the first ``k > m`` that fails gives the accuracy order ``p = k - m``.

    Getting that target wrong is easy and produces a table that reads 1 for everything: the
    obvious guess ``k!/(k-m)!`` is the derivative of ``x^k`` at a **general** point, not at 0.

    This is measured rather than quoted because a stencil can be better than its size suggests:
    the central first difference has 2 points and order 2, not order 1, and the central second
    difference has 3 points and order 2, not order 1.
    """
    o = np.atleast_1d(np.asarray(offsets, dtype=float)).ravel()
    c = from_taylor(o, order)
    m = int(order)
    for k in range(m + 1, m + o.size + 6):
        got = float(np.sum(c * o ** k))
        want = float(math.factorial(m)) if k == m else 0.0
        if abs(got - want) > tol * max(abs(want), 1.0):
            return k - m
    return o.size


#: The stencils everyone uses, as ``(name, offsets, derivative order)``.
#:
#: The names are the standard ones and the orders of accuracy are **measured** by
#: `accuracy_order` rather than recorded here, so the table cannot be wrong about itself.
STENCILS = {
    "forward first": ((0, 1), 1),
    "backward first": ((-1, 0), 1),
    "central first": ((-1, 1), 1),
    "central first, 4th order": ((-2, -1, 1, 2), 1),
    "forward first, 2nd order": ((0, 1, 2), 1),
    "central second": ((-1, 0, 1), 2),
    "central second, 4th order": ((-2, -1, 0, 1, 2), 2),
    "forward second": ((0, 1, 2), 2),
    "central third": ((-2, -1, 1, 2), 3),
    "central fourth": ((-2, -1, 0, 1, 2), 4),
}


def differentiate(f, x, h: float, offsets, order: int = 1) -> np.ndarray:
    """Apply a stencil. The whole method is three lines and the difficulty is choosing ``h``."""
    o = np.atleast_1d(np.asarray(offsets, dtype=float)).ravel()
    step = float(h)
    if step <= 0.0:
        raise ValueError(f"the step must be positive, got {step}")
    c = from_taylor(o, order)
    z = np.atleast_1d(np.asarray(x, dtype=float))
    total = np.zeros(z.shape)
    for weight, offset in zip(c, o):
        total = total + weight * np.asarray(f(z + offset * step), dtype=float)
    return total / step ** int(order)


# --------------------------------------------------------------------------- the trade


def error_against_step(f, exact, x: float, offsets, order: int = 1, steps=None) -> dict:
    """The error as the step shrinks, showing truncation falling and roundoff rising.

    This is the picture the whole lesson is about, and it is a **V**: the error falls at the
    formula's order until roundoff takes over, then rises like ``1/h^order``.
    """
    hs = (np.logspace(-1, -16, 61) if steps is None
          else np.asarray(steps, dtype=float).ravel())
    want = float(exact)
    got = np.asarray([float(np.atleast_1d(differentiate(f, x, float(h), offsets, order))[0])
                      for h in hs])
    err = np.abs(got - want)
    k = int(np.argmin(err))
    return {"steps": hs, "values": got, "errors": err,
            "best_step": float(hs[k]), "best_error": float(err[k]),
            "accuracy_order": accuracy_order(offsets, order)}


def optimal_step(order_of_accuracy: int, derivative_order: int = 1,
                 f_scale: float = 1.0, derivative_scale: float = 1.0) -> dict:
    """The step balancing truncation against roundoff, and the error it achieves.

    Truncation is ``C h^p`` and roundoff is ``2 eps |f| / h^m`` for a derivative of order ``m``
    computed at accuracy ``p``. Setting the derivative of the sum to zero,

        h* = ( 2 m eps |f| / (p C) )^{1/(p+m)}

    and the resulting error is ``O(eps^{p/(p+m)})``. For the forward first difference,
    ``p = 1, m = 1``, giving ``h* ~ sqrt(eps)`` and an error of ``sqrt(eps)``. For the central
    first difference, ``p = 2, m = 1``, giving ``eps^{1/3}`` and ``eps^{2/3}``.
    """
    p = int(order_of_accuracy)
    m = int(derivative_order)
    if p < 1 or m < 1:
        raise ValueError(f"need positive orders, got accuracy {p} and derivative {m}")
    eps = float(np.finfo(float).eps)
    C = float(derivative_scale)
    step = (2.0 * m * eps * float(f_scale) / (p * max(C, 1e-300))) ** (1.0 / (p + m))
    truncation = C * step ** p
    roundoff = 2.0 * eps * float(f_scale) / step ** m
    return {"step": step, "truncation": truncation, "roundoff": roundoff,
            "total": truncation + roundoff,
            "step_exponent": 1.0 / (p + m), "error_exponent": p / (p + m),
            "predicted_error_size": eps ** (p / (p + m))}


def digits_lost(orders=None, derivative_order: int = 1) -> dict:
    """How many decimal digits each order of accuracy gives up, from the exponent alone.

    The point of the table is that **no order recovers them all**: the error exponent
    ``p/(p+m)`` tends to 1 only as ``p`` tends to infinity, and the stencils get wide and badly
    conditioned long before that.
    """
    ps = ([1, 2, 4, 6, 8, 12, 20] if orders is None
          else [int(v) for v in np.atleast_1d(orders)])
    m = int(derivative_order)
    eps = float(np.finfo(float).eps)
    rows = []
    for p in ps:
        exponent = p / (p + m)
        rows.append((p, exponent, eps ** exponent, -math.log10(eps ** exponent),
                     16.0 + math.log10(eps ** exponent)))
    return {"accuracy_orders": np.asarray([r[0] for r in rows]),
            "error_exponents": np.asarray([r[1] for r in rows]),
            "achievable_error": np.asarray([r[2] for r in rows]),
            "digits_kept": np.asarray([r[3] for r in rows]),
            "digits_lost": np.asarray([r[4] for r in rows])}


# --------------------------------------------------------------------------- extrapolation


def richardson(f, x: float, h: float, levels: int = 4, central: bool = True,
               _counter=None) -> dict:
    """Combine quotients at halving steps to kill the leading truncation terms.

    The forward quotient's error expands in ``h, h^2, h^3, ...`` and the central one's in
    ``h^2, h^4, h^6, ...`` because its odd terms cancel by symmetry. So each Richardson level
    buys **one** order for the forward quotient and **two** for the central one, which is the
    same symmetry argument lesson 49 used for Stirling's formula.

    Returns the whole triangular table, whose first column is the raw quotients and whose
    diagonal is the extrapolated sequence.
    """
    p = 2 if central else 1
    step_gain = 2 if central else 1
    if central:
        quotient = lambda s: (float(np.atleast_1d(f(x + s))[0])
                              - float(np.atleast_1d(f(x - s))[0])) / (2.0 * s)
    else:
        quotient = lambda s: (float(np.atleast_1d(f(x + s))[0])
                              - float(np.atleast_1d(f(x))[0])) / s
    n = int(levels)
    if n < 0:
        raise ValueError(f"need a non-negative number of levels, got {n}")
    table = [[quotient(float(h) / 2.0 ** k) for k in range(n + 1)]]
    if _counter is not None:
        _counter[0] += (2 if central else 2) * (n + 1)
    for m in range(1, n + 1):
        previous = table[-1]
        factor = 2.0 ** (p + step_gain * (m - 1))
        table.append([(factor * previous[k + 1] - previous[k]) / (factor - 1.0)
                      for k in range(len(previous) - 1)])
    return {"table": table, "diagonal": np.asarray([row[0] for row in table]),
            "best": float(table[-1][0]),
            "orders": np.asarray([p + step_gain * m for m in range(n + 1)])}


def richardson_report(f, exact, x: float, h: float, levels: int = 5) -> dict:
    """The forward and central Richardson tables, with the achieved error at each level."""
    want = float(exact)
    out = {}
    for name, central in (("forward", False), ("central", True)):
        counter = [0]
        got = richardson(f, x, h, levels, central=central, _counter=counter)
        errors = np.abs(got["diagonal"] - want)
        out[name] = {"values": got["diagonal"], "errors": errors,
                     "orders": got["orders"], "evaluations": counter[0],
                     "best_error": float(np.min(errors)),
                     "best_level": int(np.argmin(errors))}
    return out


# --------------------------------------------------------------------------- complex step


def complex_step(f, x, h: float = 1e-200) -> np.ndarray:
    """``Im(f(x + ih)) / h``, which has no subtraction and therefore no cancellation.

    For real analytic ``f``, expanding ``f(x + ih)`` gives

        f(x + ih) = f(x) + i h f'(x) - h^2 f''(x)/2 - i h^3 f'''(x)/6 + ...

    so the imaginary part is ``h f'(x) - h^3 f'''(x)/6 + ...`` and dividing by ``h`` leaves an
    error of ``O(h^2)`` with **no subtraction of nearly equal numbers**. The roundoff term of
    the ordinary quotient is simply absent, so ``h`` may be ``1e-200`` and the truncation term
    is ``1e-400``, which underflows to zero.

    **The requirements are real.** ``f`` must accept complex input and must be analytic. Anything
    that branches on a real comparison breaks it: ``abs``, ``max``, ``min``, and every
    ``if x > 0``. In numpy, ``np.abs`` of a complex number is the modulus, which is **not** the
    analytic continuation of the real ``abs``, so a formula containing it silently gives the
    wrong derivative rather than failing.
    """
    step = float(h)
    if step <= 0.0:
        raise ValueError(f"the step must be positive, got {step}")
    z = np.atleast_1d(np.asarray(x, dtype=float)).astype(complex)
    return np.imag(np.asarray(f(z + 1j * step), dtype=complex)) / step


def complex_step_report(f, exact, x: float, steps=None) -> dict:
    """Complex step against the central difference, over many step sizes.

    The central difference has a V; the complex step is flat at machine precision from
    ``h = 1e-8`` all the way to ``1e-200``, because there is no roundoff term to grow.
    """
    hs = (np.logspace(-1, -20, 40) if steps is None
          else np.asarray(steps, dtype=float).ravel())
    want = float(exact)
    complex_errors = []
    central_errors = []
    for h in hs:
        complex_errors.append(abs(float(complex_step(f, x, float(h))[0]) - want))
        central_errors.append(abs(float(np.atleast_1d(
            differentiate(f, x, float(h), (-1, 1), 1))[0]) - want))
    best_complex = float(np.min(complex_errors))
    best_central = float(np.min(central_errors))
    floor = float(np.finfo(float).eps) * max(abs(want), 1.0)
    return {"steps": hs,
            "complex_step_errors": np.asarray(complex_errors),
            "central_errors": np.asarray(central_errors),
            "complex_step_best": best_complex,
            "central_best": best_central,
            "roundoff_floor": floor,
            "complex_step_is_exact": bool(best_complex <= floor),
            # the ratio is floored at eps because the complex step's error reaches EXACTLY
            # zero: sin(h)/h rounds to 1.0 for h below about 1e-8, so the computed value is
            # bit for bit the correctly rounded derivative and dividing by it is meaningless
            "advantage": best_central / max(best_complex, floor)}


def complex_step_fails_on(name: str, x: float = 1.5) -> dict:
    """Functions on which the complex step may silently give a wrong answer, and why.

    **Three of these five genuinely fail and two do not**, which is the useful part of the
    measurement. The failures are the operations that are **nowhere** holomorphic:

    - ``np.abs`` of a complex number is the modulus, which is real, so the imaginary part is
      zero and the reported derivative is 0 rather than 1.
    - ``np.real`` discards the perturbation entirely, same result.
    - ``conj(z) z`` is ``|z|^2``, real again.

    The two that survive are **piecewise analytic**, and they work everywhere except at their
    kink, where the derivative does not exist anyway:

    - ``sqrt(z^2)`` equals ``z`` near the positive real axis and ``-z`` near the negative one,
      so its complex derivative is ``+1`` and ``-1`` respectively, which is exactly
      ``sign(x)``.
    - ``np.maximum(z, 0)`` compares complex numbers lexicographically, so away from the branch
      it returns the right side and the derivative is right.

    So the rule is not "avoid anything non-smooth". It is **avoid anything that maps the
    imaginary perturbation to zero**, and that is a much smaller and more checkable class.

    None of these raises. Each returns a number, and for the first three the number is wrong,
    which is the dangerous failure mode and the reason to check.
    """
    cases = {
        "abs": (lambda z: np.abs(z), lambda t: math.copysign(1.0, t),
                "np.abs of a complex number is the modulus, not the analytic continuation"),
        "real": (lambda z: np.real(z) ** 2, lambda t: 2.0 * t,
                 "np.real discards the imaginary part, so the derivative information is gone"),
        "conjugate": (lambda z: np.conj(z) * z, lambda t: 2.0 * t,
                      "conj is not holomorphic anywhere"),
        "sqrt of a square": (lambda z: np.sqrt(z * z), lambda t: math.copysign(1.0, t),
                             "sqrt(z^2) is |x| for real x and z for complex z near the "
                             "positive axis, so the two agree in value and not in derivative"),
        "maximum": (lambda z: np.maximum(z, 0.0) ** 2, lambda t: 2.0 * max(t, 0.0),
                    "a branch on a real comparison is not analytic, but numpy's maximum "
                    "compares complex numbers lexicographically, so this one happens to work "
                    "AWAY from the kink and fails only at it, where the derivative does not "
                    "exist anyway"),
    }
    key = str(name).lower()
    if key not in cases:
        raise ValueError(f"unknown case {name!r}, expected one of {sorted(cases)}")
    f, exact, why = cases[key]
    want = float(exact(float(x)))
    got = float(complex_step(f, float(x))[0])
    # The central difference is the honest comparison, so it must see REAL input. Feeding it
    # complex input and casting the answer back to float discards exactly the imaginary part
    # the complex step depends on, which makes the two methods look identical when they are not.
    central = float(np.atleast_1d(differentiate(
        lambda t: np.real(np.asarray(f(np.asarray(t, dtype=float)))),
        float(x), 1e-6, (-1, 1), 1))[0])
    return {"case": key, "exact": want, "complex_step": got, "central_difference": central,
            "complex_step_error": abs(got - want), "central_error": abs(central - want),
            "why": why, "silently_wrong": bool(abs(got - want) > 1e-6 * max(abs(want), 1.0))}


# --------------------------------------------------------------------------- applications


def second_derivative_matrix(n: int, h: float = 1.0, boundary: str = "dirichlet") -> np.ndarray:
    """The tridiagonal matrix of the central second difference, which Part 11 solves with.

    ``[1, -2, 1] / h^2`` on the interior, with the boundary rows set by the condition. This is
    the object lesson 21's Thomas algorithm was built for and the discrete Laplacian of Part 11.
    """
    m = int(n)
    if m < 2:
        raise ValueError(f"need at least two unknowns, got {m}")
    step = float(h)
    if step <= 0.0:
        raise ValueError(f"the step must be positive, got {step}")
    A = (np.diag(-2.0 * np.ones(m)) + np.diag(np.ones(m - 1), 1)
         + np.diag(np.ones(m - 1), -1)) / step ** 2
    key = str(boundary).lower()
    if key == "dirichlet":
        return A
    if key == "neumann":
        A[0, 0] = -1.0 / step ** 2
        A[-1, -1] = -1.0 / step ** 2
        return A
    if key == "periodic":
        A[0, -1] = 1.0 / step ** 2
        A[-1, 0] = 1.0 / step ** 2
        return A
    raise ValueError(f"unknown boundary {boundary!r}, expected dirichlet, neumann or periodic")


def differentiation_matrix(nodes, order: int = 1) -> np.ndarray:
    """The dense matrix taking values at the nodes to derivatives at the nodes.

    Row ``i`` holds the finite difference weights centred on node ``i``, computed on **all** the
    nodes. For equally spaced nodes that is a banded matrix in disguise; for Chebyshev nodes it
    is the spectral differentiation matrix, which is dense and converges geometrically.

    The rows come from `fornberg`, not from `from_taylor`. Built the other way the matrix is
    accurate to 3e-12 at 11 nodes, to 4e-04 at 25, and has no correct digit left by 31. The
    symptom is easy to miss, because the matrix keeps reproducing low degree polynomials well
    and only fails on the high degree ones it is supposed to be exact on.
    """
    x = np.atleast_1d(np.asarray(nodes, dtype=float)).ravel()
    n = x.size
    m = int(order)
    if n <= m:
        raise ValueError(f"order {m} needs more than {m} nodes, got {n}")
    D = np.empty((n, n))
    for i in range(n):
        D[i] = fornberg(x - x[i], m)
    return D


def spectral_against_finite_difference(f, dfdx, counts=None, order: int = 1) -> dict:
    """Chebyshev spectral differentiation against a three point stencil, as the count grows.

    The finite difference converges at its own order, 2 here, and the spectral method converges
    geometrically until the roundoff of the dense matrix takes over. That crossover is the whole
    argument for spectral methods and it is measured rather than asserted.
    """
    from . import chebyshev as _cheb

    ns = ([5, 9, 17, 33, 65] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        cheb_nodes = np.sort(_cheb.extrema_nodes(n))
        D = differentiation_matrix(cheb_nodes, order)
        spectral = float(np.max(np.abs(D @ np.asarray(f(cheb_nodes), dtype=float)
                                       - np.asarray(dfdx(cheb_nodes), dtype=float))))
        even = np.linspace(-1.0, 1.0, n)
        h = float(even[1] - even[0])
        inner = even[1:-1]
        fd = float(np.max(np.abs(
            (np.asarray(f(inner + h), dtype=float) - np.asarray(f(inner - h), dtype=float))
            / (2.0 * h) - np.asarray(dfdx(inner), dtype=float))))
        rows.append((n, spectral, fd))
    return {"counts": np.asarray([r[0] for r in rows]),
            "spectral_error": np.asarray([r[1] for r in rows]),
            "finite_difference_error": np.asarray([r[2] for r in rows])}
