"""Chebyshev approximation: the practical answer to lesson 54's question.

The place this sits
-------------------
Lesson 54 said the best ``Linf`` approximation exists, is unique, and is characterised by
equioscillation, and `approx.remez` computes it. Remez is iterative, needs a root search at every
step, and is what a library vendor runs **once** when generating code for ``sin``.

Nobody runs Remez to fit a curve. What everybody uses is the **Chebyshev series**

    f(x) = sum_k a_k T_k(x),      a_k = (2/pi) integral f(x) T_k(x) / sqrt(1-x^2) dx

truncated after ``n`` terms. It costs one transform, it needs no iteration, and it is
**near-minimax**: `near_minimax_factor` measures the truncated series against the true best
approximation and finds the gap is a small factor, never an order of magnitude.

The reason is Bernstein's bound. Truncating loses ``sum_{k>n} |a_k|``, while the best possible
error is at least ``|a_{n+1}|``, and for a smooth ``f`` the coefficients decay geometrically, so
the tail is dominated by its first term. Measured across five functions at four degrees, the
ratio runs **1.47 to 2.12**, against a Lebesgue-style bound of ``4/pi^2 log n + 4`` which is
between 4.3 and 5.1 over the same range. The one outlier is 5.00, for ``cosh`` at degree 16,
where every odd coefficient is exactly zero so the truncation drops nothing and the comparison
is between two different degrees.

Computing the coefficients
--------------------------
Three ways, all implemented, because they teach different things.

- `coefficients_by_quadrature` uses the defining integral directly. Simple, slow, and the
  accuracy is the quadrature's.
- `coefficients_by_interpolation` interpolates at the Chebyshev points and reads off the
  coefficients by a discrete cosine sum. This is what a library does, it is ``O(n^2)`` here and
  ``O(n log n)`` with lesson 59's FFT, and the coefficients it produces are **not** the true
  series coefficients: they are aliased, which `aliasing_report` measures.
- `coefficients_by_fit` solves a least squares problem, which is the honest fallback when ``f``
  is only known at scattered points.

Economization
-------------
Given a Taylor polynomial of degree ``n``, `economize` removes the top term by subtracting a
multiple of ``T_n``, and repeats. Each step raises the error by the smallest amount any degree
reduction can, because ``T_n / 2^{n-1}`` is the smallest monic polynomial in the max norm, which
is lesson 47's minimax property. The measured result on ``exp``: a degree 10 Taylor polynomial
economized to degree 5 has error ``4.84e-5``, while the degree 5 Taylor polynomial has
``1.62e-3`` and the degree 5 minimax has ``4.52e-5``. Economization recovers **99.8 percent of
the gap** for five subtractions.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- basics


def _to_unit(x, lo, hi):
    lo = float(lo)
    hi = float(hi)
    if not hi > lo:
        raise ValueError(f"need lo < hi, got [{lo}, {hi}]")
    return np.clip((2.0 * np.asarray(x, dtype=float) - (hi + lo)) / (hi - lo), -1.0, 1.0)


def _values(f, x):
    """Evaluate ``f`` on an array, accepting a vectorised or a scalar callable."""
    x = np.asarray(x, dtype=float)
    try:
        v = np.asarray(f(x), dtype=float)
        if v.shape != x.shape:
            raise ValueError
    except (TypeError, ValueError):
        v = np.asarray([float(f(s)) for s in x], dtype=float)
    return v


def evaluate_series(coefficients, x, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Sum a Chebyshev series by Clenshaw's recurrence.

    Clenshaw is to a Chebyshev series what Horner is to a power series: it evaluates in ``O(n)``
    without ever forming ``T_k``, and it is backward stable. The recurrence runs

        b_k = 2 x b_{k+1} - b_{k+2} + a_k,      result = b_0 - x b_1

    with the convention that the ``k = 0`` term is not halved, matching
    `coefficients_by_interpolation`.
    """
    a = np.atleast_1d(np.asarray(coefficients, dtype=float))
    if a.size == 0:
        raise ValueError("a series needs at least one coefficient")
    u = _to_unit(x, lo, hi)
    b1 = np.zeros_like(u)
    b2 = np.zeros_like(u)
    for k in range(a.size - 1, 0, -1):
        b1, b2 = 2.0 * u * b1 - b2 + a[k], b1
    return u * b1 - b2 + a[0]


def to_power_basis(coefficients) -> np.ndarray:
    """Convert a Chebyshev series to power form, and say plainly why not to.

    The conversion is exact in exact arithmetic and it is the operation that destroys everything
    the Chebyshev basis was for: the power coefficients of ``T_k`` alternate in sign and grow
    like ``2^{k-1}``, so summing them cancels. How bad that is depends on how fast the Chebyshev
    coefficients decay, because a tiny ``a_k`` multiplying a huge power expansion still cancels
    to nothing.

    Measured on ``T_n`` itself, where nothing decays, the round trip against Clenshaw disagrees
    by ``1.5e-9`` at degree 20, ``5.6e-2`` at degree 40, and **228** at degree 50, where the
    largest power coefficient is ``1.3e18``. On the Runge function, whose coefficients decay
    slowly, it is ``1.0e-2`` at degree 50. On ``exp``, whose coefficients collapse, it survives
    to ``2.8e-14`` at degree 50, because those large power coefficients are multiplied by an
    ``a_k`` of size ``1e-60``.

    So the rule is not "never convert". It is that the conversion is safe exactly when the
    series was going to be short anyway.

    Returned lowest order first, matching `nalib.polynomials`.
    """
    a = np.atleast_1d(np.asarray(coefficients, dtype=float))
    n = a.size
    out = np.zeros(n)
    prev = np.zeros(n)       # T_{k-1}
    curr = np.zeros(n)       # T_k
    curr[0] = 1.0
    for k in range(n):
        out[:] = out + a[k] * curr
        nxt = np.zeros(n)
        nxt[1:] = 2.0 * curr[:-1]
        # The recurrence T_{k+1} = 2x T_k - T_{k-1} holds only from k = 1. Starting it at k = 0
        # with T_{-1} = 0 produces T_1 = 2x instead of x, and every later member inherits the
        # factor of 2. That made the round trip through this function disagree with Clenshaw by
        # 67 percent at every degree, which is what caught it.
        if k == 0:
            nxt = 0.5 * nxt
        else:
            nxt = nxt - prev
        prev, curr = curr, nxt
    return out


def conversion_cost(f, degree: int, lo: float = -1.0, hi: float = 1.0,
                    n_probe: int = 2001) -> dict:
    """Evaluate a Chebyshev fit both ways and compare, so the loss can be seen not asserted."""
    n = int(degree)
    a = coefficients_by_interpolation(f, n, lo, hi)
    p = to_power_basis(a)
    x = np.linspace(float(lo), float(hi), int(n_probe))
    u = _to_unit(x, lo, hi)
    clenshaw = evaluate_series(a, x, lo, hi)
    power = sum(p[k] * u ** k for k in range(p.size))
    scale = max(float(np.max(np.abs(clenshaw))), 1e-300)
    return {"degree": n, "largest_power_coefficient": float(np.max(np.abs(p))),
            "disagreement": float(np.max(np.abs(clenshaw - power))),
            "relative_disagreement": float(np.max(np.abs(clenshaw - power)) / scale)}


# --------------------------------------------------------------------------- coefficients


def coefficients_by_quadrature(f, degree: int, lo: float = -1.0, hi: float = 1.0,
                               n_quad: int = 4001) -> np.ndarray:
    """The defining integral, by the substitution ``x = cos(theta)``.

        a_k = (2/pi) integral_0^pi f(cos t) cos(k t) dt,      a_0 halved

    The substitution turns the singular weight ``1/sqrt(1-x^2)`` into ``dt``, so a plain
    trapezoid rule in ``t`` is accurate. Integrating in ``x`` instead misses the endpoint
    singularity entirely.
    """
    n = int(degree)
    if n < 0:
        raise ValueError(f"degree must be non-negative, got {n}")
    m = int(n_quad)
    theta = np.linspace(0.0, math.pi, m)
    x = 0.5 * (float(hi) + float(lo)) + 0.5 * (float(hi) - float(lo)) * np.cos(theta)
    v = _values(f, x)
    out = np.empty(n + 1)
    for k in range(n + 1):
        out[k] = (2.0 / math.pi) * float(np.trapezoid(v * np.cos(k * theta), theta))
    out[0] *= 0.5
    return out


def coefficients_by_interpolation(f, degree: int, lo: float = -1.0, hi: float = 1.0,
                                  kind: str = "extrema") -> np.ndarray:
    """Interpolate at the Chebyshev points and read off the coefficients by a cosine sum.

    ``kind="extrema"`` uses the ``n + 1`` Chebyshev-Lobatto points, which include the ends and
    give the discrete cosine transform of type I. ``kind="roots"`` uses the ``n + 1`` Chebyshev
    roots, which give type II and are what a Gauss-Chebyshev rule uses.

    These are **not** the true series coefficients. Each one is the true coefficient plus the
    aliased contributions of every higher harmonic that the grid cannot distinguish, which is
    the same aliasing lesson 58 measures for the DFT. `aliasing_report` shows it directly.
    """
    n = int(degree)
    if n < 0:
        raise ValueError(f"degree must be non-negative, got {n}")
    key = str(kind).lower()
    if key not in ("extrema", "roots"):
        raise ValueError(f"kind must be 'extrema' or 'roots', got {kind!r}")
    if key == "extrema":
        if n == 0:
            theta = np.array([0.0])
        else:
            theta = np.arange(n + 1) * math.pi / n
    else:
        theta = (2.0 * np.arange(n + 1) + 1.0) * math.pi / (2.0 * (n + 1))
    x = 0.5 * (float(hi) + float(lo)) + 0.5 * (float(hi) - float(lo)) * np.cos(theta)
    v = _values(f, x)
    out = np.empty(n + 1)
    if key == "roots":
        for k in range(n + 1):
            out[k] = 2.0 / (n + 1) * float(np.sum(v * np.cos(k * theta)))
        out[0] *= 0.5
        return out
    if n == 0:
        return np.array([float(v[0])])
    w = np.ones(n + 1)
    w[0] = w[-1] = 0.5
    for k in range(n + 1):
        out[k] = 2.0 / n * float(np.sum(w * v * np.cos(k * theta)))
    out[0] *= 0.5
    out[n] *= 0.5
    return out


def coefficients_by_fit(x, y, degree: int, lo=None, hi=None) -> dict:
    """Least squares Chebyshev coefficients from scattered samples.

    The honest fallback when ``f`` cannot be evaluated where you want it. The design matrix is
    the Chebyshev basis, whose conditioning stays bounded where the monomial one does not,
    which is lesson 54's exercise on `approx.basis_conditioning`.
    """
    n = int(degree)
    xs = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    ys = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if xs.size != ys.size:
        raise ValueError(f"{xs.size} nodes against {ys.size} values")
    if xs.size < n + 1:
        raise ValueError(f"degree {n} needs at least {n + 1} samples, got {xs.size}")
    a = float(np.min(xs)) if lo is None else float(lo)
    b = float(np.max(xs)) if hi is None else float(hi)
    u = _to_unit(xs, a, b)
    theta = np.arccos(u)
    A = np.stack([np.cos(k * theta) for k in range(n + 1)], axis=1)
    c, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return {"coefficients": c, "interval": (a, b),
            "condition": float(np.linalg.cond(A)),
            "residual": float(np.max(np.abs(A @ c - ys)))}


def coefficient_decay(f, degree: int, lo: float = -1.0, hi: float = 1.0) -> dict:
    """How fast the coefficients fall, which is the whole story of how good the fit will be.

    For ``f`` analytic in a Bernstein ellipse of parameter ``rho``, ``|a_k| ~ rho^-k``, so the
    decay is geometric and the fitted ratio recovers ``rho``. For ``f`` with ``m`` continuous
    derivatives the decay is only ``O(k^{-m-1})``, which is algebraic. The two are told apart
    by whether ``log|a_k|`` is straight against ``k`` or against ``log k``.

    **Coefficients that are exactly zero have to be excluded, and so do ones at the roundoff
    floor.** An even function has every odd coefficient exactly zero, which is half the sequence
    sitting at ``1e-17``, and including them flattens both fits: the Runge function's geometric
    rate then reads ``0.985`` instead of its true ``1.22``, and ``|t|`` reports as geometric when
    it is not. ``kept`` says how many entries the fit actually used.

    A third case exists and is reported separately. An **entire** function such as ``exp`` decays
    faster than any geometric rate, because ``|a_k| ~ 1/(2^k k!)``, so neither straight line fits
    and the ratio ``|a_k / a_{k+1}|`` keeps growing. ``classification`` distinguishes the three.
    """
    n = int(degree)
    a = coefficients_by_interpolation(f, n, lo, hi)
    mag = np.abs(a)
    floor = 1e-14 * max(float(np.max(mag)), 1e-300)
    keep = (mag > floor) & (np.arange(a.size) > 0)
    k = np.arange(a.size)[keep].astype(float)
    if k.size < 4:
        return {"coefficients": a, "magnitudes": mag, "kept": int(k.size),
                "geometric_rate": float("nan"), "algebraic_order": float("nan"),
                "geometric_residual": float("nan"), "algebraic_residual": float("nan"),
                "looks_geometric": False, "classification": "too few coefficients to fit"}
    lg = np.log(mag[keep])
    geometric = np.polyfit(k, lg, 1)
    algebraic = np.polyfit(np.log(k), lg, 1)
    resid = lambda fit, xs: float(np.sqrt(np.mean((lg - (fit[0] * xs + fit[1])) ** 2)))
    r_geo = resid(geometric, k)
    r_alg = resid(algebraic, np.log(k))
    # a local rate that keeps rising means faster than any fixed geometric rate
    local = mag[keep][:-1] / np.maximum(mag[keep][1:], 1e-300)
    half = local.size // 2
    accelerating = bool(half >= 2 and float(np.median(local[half:])) >
                        2.0 * float(np.median(local[:half])))
    if accelerating:
        label = "faster than geometric (entire)"
    elif r_geo < r_alg:
        label = "geometric (analytic in a Bernstein ellipse)"
    else:
        label = "algebraic (finitely many derivatives)"
    return {"coefficients": a, "magnitudes": mag, "kept": int(k.size),
            "geometric_rate": float(np.exp(-geometric[0])),
            "algebraic_order": float(-algebraic[0]),
            "geometric_residual": r_geo, "algebraic_residual": r_alg,
            "looks_geometric": bool(not accelerating and r_geo < r_alg),
            "accelerating": accelerating,
            "classification": label}


def aliasing_report(f, degree: int, lo: float = -1.0, hi: float = 1.0,
                    n_reference: int = 200) -> dict:
    """Interpolation coefficients against the true series coefficients.

    The exact statement, for the Chebyshev-Lobatto grid of ``n + 1`` points, is

        a_k^{grid} = a_k + a_{2n-k} + a_{2n+k} + a_{4n-k} + ...

    so every coefficient carries the aliases of the harmonics the grid cannot see. That is why
    the interpolant is not the truncated series, and it is also why the difference is tiny for
    a smooth ``f``: the aliases are the small tail coefficients.
    """
    n = int(degree)
    grid = coefficients_by_interpolation(f, n, lo, hi)
    exact = coefficients_by_interpolation(f, int(n_reference), lo, hi)[:n + 1]
    predicted = exact.copy()
    full = coefficients_by_interpolation(f, int(n_reference), lo, hi)
    for k in range(n + 1):
        j = 2 * n - k
        while j < full.size:
            predicted[k] += full[j]
            j += 2 * n if 2 * n > 0 else 1
        j = 2 * n + k
        while j < full.size:
            predicted[k] += full[j]
            j += 2 * n if 2 * n > 0 else 1
    scale = max(float(np.max(np.abs(exact))), 1e-300)
    return {"grid_coefficients": grid, "true_coefficients": exact,
            "predicted_with_aliases": predicted,
            "difference": float(np.max(np.abs(grid - exact))),
            "relative_difference": float(np.max(np.abs(grid - exact)) / scale),
            "alias_prediction_error": float(np.max(np.abs(grid - predicted)) / scale)}


# --------------------------------------------------------------------------- near-minimax


def truncation_error(f, degree: int, lo: float = -1.0, hi: float = 1.0,
                     n_probe: int = 20001) -> dict:
    """The max error of the truncated Chebyshev series, and the tail bound that predicts it.

    Truncating after ``n`` loses ``sum_{k>n} a_k T_k``, and since ``|T_k| <= 1`` the loss is at
    most ``sum_{k>n} |a_k|``. That bound is computable from the coefficients alone, before any
    evaluation, which is one of the practical reasons to work in this basis.

    **The bound is about the truncated series, not about the interpolant**, and the two are
    different objects. The interpolant's coefficients carry the aliases of `aliasing_report`, so
    its error can be up to twice the tail: measured, the truncated series sits at 0.97 to 1.00
    of the bound while the interpolant sits at 1.5 to 1.9 of it. Both are reported, because
    quoting the bound next to the interpolant's error would look like the bound failing.
    """
    n = int(degree)
    long = coefficients_by_interpolation(f, max(4 * n + 20, 40), lo, hi)
    truncated = long[:n + 1].copy()
    interpolant = coefficients_by_interpolation(f, n, lo, hi)
    x = np.linspace(float(lo), float(hi), int(n_probe))
    truth = _values(f, x)
    err = float(np.max(np.abs(evaluate_series(truncated, x, lo, hi) - truth)))
    err_interp = float(np.max(np.abs(evaluate_series(interpolant, x, lo, hi) - truth)))
    tail = float(np.sum(np.abs(long[n + 1:])))
    return {"degree": n, "max_error": err, "interpolant_error": err_interp,
            "tail_bound": tail,
            "bound_holds": bool(err <= tail * (1.0 + 1e-6) + 1e-13),
            "bound_holds_for_the_interpolant":
                bool(err_interp <= 2.0 * tail * (1.0 + 1e-6) + 1e-13),
            "bound_tightness": err / max(tail, 1e-300),
            "first_dropped": float(abs(long[n + 1])) if long.size > n + 1 else 0.0}


def near_minimax_factor(f, degree: int, lo: float = -1.0, hi: float = 1.0,
                        n_probe: int = 20001) -> dict:
    """Truncated Chebyshev against the true best approximation of the same degree.

    The classical bound is that truncation is within a factor of ``4/pi^2 log n + 4`` of best,
    which is the Lebesgue constant of Chebyshev projection. In practice the factor is far
    smaller, and this measures it rather than quoting the bound.
    """
    from . import approx as _approx

    n = int(degree)
    a = coefficients_by_interpolation(f, n, lo, hi)
    x = np.linspace(float(lo), float(hi), int(n_probe))
    truth = _values(f, x)
    cheb = float(np.max(np.abs(evaluate_series(a, x, lo, hi) - truth)))
    best = _approx.remez(f, n, lo, hi, n_probe=n_probe)
    return {"degree": n, "chebyshev_error": cheb, "minimax_error": best["max_error"],
            "factor": cheb / max(best["max_error"], 1e-300),
            "classical_bound": 4.0 / math.pi ** 2 * math.log(max(n, 2)) + 4.0}


def against_taylor(f, taylor_coefficients, degree: int, lo: float = -1.0, hi: float = 1.0,
                   n_probe: int = 20001) -> dict:
    """Truncated Chebyshev against the truncated Taylor series of the same degree.

    Taylor is optimal **at a point** and Chebyshev is near optimal **on an interval**, and those
    are different jobs. The measured gap is what a library gains by not using the Taylor series
    it was probably taught with.
    """
    n = int(degree)
    tc = np.atleast_1d(np.asarray(taylor_coefficients, dtype=float))
    if tc.size < n + 1:
        raise ValueError(f"degree {n} needs {n + 1} Taylor coefficients, got {tc.size}")
    x = np.linspace(float(lo), float(hi), int(n_probe))
    truth = _values(f, x)
    taylor = sum(tc[k] * x ** k for k in range(n + 1))
    cheb = evaluate_series(coefficients_by_interpolation(f, n, lo, hi), x, lo, hi)
    t_err = float(np.max(np.abs(taylor - truth)))
    c_err = float(np.max(np.abs(cheb - truth)))
    return {"degree": n, "taylor_error": t_err, "chebyshev_error": c_err,
            "ratio": t_err / max(c_err, 1e-300)}


# --------------------------------------------------------------------------- economization


def economize(power_coefficients, drop: int = 1) -> dict:
    """Lower the degree of a power series by subtracting multiples of the Chebyshev polynomials.

    The step is: write the top coefficient ``c_n`` times ``x^n`` as
    ``(c_n / 2^{n-1}) * (2^{n-1} x^n)``, note that ``T_n = 2^{n-1} x^n + (lower)``, and subtract
    ``(c_n / 2^{n-1}) T_n``. The degree drops by one and the error rises by at most
    ``|c_n| / 2^{n-1}``, which by lesson 47's minimax property is the **smallest** possible
    increase for any degree reduction.

    Assumes the series is on ``[-1, 1]``. Map first if it is not.
    """
    c = np.atleast_1d(np.asarray(power_coefficients, dtype=float)).astype(float).copy()
    k = int(drop)
    if k < 0:
        raise ValueError(f"cannot drop a negative number of terms, got {k}")
    if k >= c.size:
        raise ValueError(f"cannot drop {k} terms from a series of {c.size}")
    added = 0.0
    for _ in range(k):
        n = c.size - 1
        if n == 0:
            break
        factor = c[n] / (2.0 ** (n - 1))
        t_power = to_power_basis(np.eye(n + 1)[n])
        c = c - factor * t_power
        added += abs(factor)
        c = c[:n]
    return {"coefficients": c, "degree": c.size - 1,
            "guaranteed_error_added": float(added)}


def economization_report(f, taylor_coefficients, target_degree: int,
                         lo: float = -1.0, hi: float = 1.0, n_probe: int = 20001) -> dict:
    """Economized Taylor against plain truncated Taylor and against the true best, same degree.

    This is the whole case for economization in one table: it costs a handful of subtractions
    and recovers most of the gap between a Taylor polynomial and the minimax one.
    """
    from . import approx as _approx

    tc = np.atleast_1d(np.asarray(taylor_coefficients, dtype=float))
    m = int(target_degree)
    if m < 0:
        raise ValueError(f"target degree must be non-negative, got {m}")
    if m >= tc.size - 1:
        raise ValueError(f"target degree {m} is not below the given degree {tc.size - 1}")
    x = np.linspace(float(lo), float(hi), int(n_probe))
    truth = _values(f, x)
    econ = economize(tc, tc.size - 1 - m)
    ec = econ["coefficients"]
    err = lambda c: float(np.max(np.abs(sum(c[k] * x ** k for k in range(c.size)) - truth)))
    best = _approx.remez(f, m, lo, hi, n_probe=n_probe)["max_error"]
    return {"start_degree": tc.size - 1, "target_degree": m,
            "full_taylor_error": err(tc),
            "truncated_taylor_error": err(tc[:m + 1]),
            "economized_error": err(ec),
            "minimax_error": best,
            "guaranteed_error_added": econ["guaranteed_error_added"],
            "fraction_of_the_gap_recovered":
                (err(tc[:m + 1]) - err(ec)) / max(err(tc[:m + 1]) - best, 1e-300)}
