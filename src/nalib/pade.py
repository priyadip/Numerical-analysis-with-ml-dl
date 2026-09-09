"""Pade approximation: what a ratio of polynomials can do that a polynomial cannot.

The one sentence version
------------------------
A polynomial is entire. It has no poles, it grows without bound, and it cannot go to infinity at
a finite point or level off at infinity. So near a pole, and on an unbounded interval, no
polynomial of any degree is any good. **A ratio of polynomials can do both**, and the Pade
approximant is the ratio that matches a Taylor series as far as its coefficient count allows.

The definition
--------------
The ``[m/n]`` Pade approximant of ``f`` is

    R(x) = P(x) / Q(x),      deg P <= m,   deg Q <= n,   Q(0) = 1

chosen so that ``R`` and ``f`` agree to order ``m + n`` at ``x = 0``:

    f(x) - P(x)/Q(x)  =  O(x^{m+n+1})

There are ``m + 1`` free coefficients in ``P`` and ``n`` in ``Q``, so ``m + n + 1`` in all, and
that is exactly the number of Taylor coefficients matched. The count is not a coincidence, it is
the design.

The linear system
-----------------
Clearing the denominator turns the matching conditions into

    f Q - P = O(x^{m+n+1})

whose coefficients at orders ``m+1`` through ``m+n`` form an ``n`` by ``n`` **Toeplitz** system
for ``q_1 ... q_n`` involving only the Taylor coefficients ``c_{m-n+1} ... c_{m+n}``. Solve that,
then read off ``P`` from the orders ``0`` through ``m``. `coefficients` does exactly this, and
`system_conditioning` measures the Toeplitz matrix, which is where the method goes wrong when it
goes wrong.

What it buys, measured
----------------------
`against_taylor` compares the ``[m/n]`` approximant against the Taylor polynomial of the same
**total cost** ``m + n``. On ``exp`` the two are comparable, because ``exp`` has no poles. On
``tan``, which has a pole at ``pi/2``, and on ``log(1+x)``, which has a branch point at ``-1``,
the Pade approximant is better by orders of magnitude near the singularity, and `pole_locations`
shows why: the denominator's roots sit on top of the true singularities.

Where it fails
--------------
`spurious_poles` finds the failure mode that keeps Pade out of general use: a pole of ``Q``
inside the interval of interest, with a nearby zero of ``P`` almost cancelling it. These
**Froissart doublets** are created by rounding in the Taylor coefficients, they move when the
precision changes, and near one the approximant is unbounded. Robust variants remove them by an
SVD of the Toeplitz matrix, which is Part 5 and Part 6 doing a job here.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- construction


def coefficients(taylor, m: int, n: int) -> dict:
    """The ``[m/n]`` Pade approximant from a Taylor series about 0.

    ``taylor`` holds ``c_0, c_1, ...`` lowest order first and must have at least ``m + n + 1``
    entries. Returns ``P`` and ``Q`` lowest order first, normalised so ``Q(0) = 1``.

    The system solved is, for ``j = 1 ... n``,

        sum_{k=1}^{n} c_{m+j-k} q_k  =  -c_{m+j}

    which is Toeplitz because the entry depends only on ``j - k``.

    **The system can be exactly singular, and that is not a bug.** The Pade table has a block
    structure: when ``f`` is already rational of type ``[mu/nu]``, every entry of the table with
    ``m >= mu`` and ``n >= nu`` is the same function, and the linear system for those entries is
    rank deficient because there is nothing left to determine. ``1/(1+x)`` is ``[0/1]``, so
    ``[2/2]`` and ``[4/4]`` are both singular. So is every ``[m/n]`` of an odd function with
    ``m`` and ``n`` of the wrong parity, since half the Taylor coefficients are zero.

    Rather than propagate numpy's bare ``LinAlgError``, a rank deficient system is solved in the
    **least norm** sense. When ``f`` is genuinely rational that recovers it exactly: on
    ``1/(1+x)`` at ``[4/4]`` the result reproduces the function to ``3e-14``.

    When the degeneracy comes from parity rather than from ``f`` being rational, the least norm
    solution is a valid rational approximant of the block but **not necessarily the block's
    canonical representative**, and its matching order is not predicted by the rank. Measured,
    ``[2/3]`` of ``tan`` has rank 2 and matches 3 Taylor terms, and ``[0/1]`` of ``tan`` has
    rank 0 and matches 1. `matching_order` measures it, and nothing here claims more.

    The ``degenerate`` flag and the ``rank`` say when this happened, so a caller can tell the
    two cases apart instead of silently trusting a number.
    """
    c = np.atleast_1d(np.asarray(taylor, dtype=float)).ravel()
    m = int(m)
    n = int(n)
    if m < 0 or n < 0:
        raise ValueError(f"degrees must be non-negative, got [{m}/{n}]")
    if c.size < m + n + 1:
        raise ValueError(f"[{m}/{n}] needs {m + n + 1} Taylor coefficients, got {c.size}")
    get = lambda i: float(c[i]) if 0 <= i < c.size else 0.0
    if n == 0:
        p = c[:m + 1].copy()
        q = np.array([1.0])
        return {"p": p, "q": q, "m": m, "n": n, "condition": 1.0, "degenerate": False,
                "rank": 0}
    A = np.empty((n, n))
    rhs = np.empty(n)
    for j in range(1, n + 1):
        for k in range(1, n + 1):
            A[j - 1, k - 1] = get(m + j - k)
        rhs[j - 1] = -get(m + j)
    scale = max(float(np.max(np.abs(A))), 1e-300)
    rank = int(np.linalg.matrix_rank(A, tol=1e-12 * scale))
    degenerate = rank < n
    if degenerate:
        q_tail, _, _, _ = np.linalg.lstsq(A, rhs, rcond=1e-12)
    else:
        q_tail = np.linalg.solve(A, rhs)
    q = np.concatenate(([1.0], q_tail))
    p = np.empty(m + 1)
    for i in range(m + 1):
        p[i] = sum(q[k] * get(i - k) for k in range(min(i, n) + 1))
    return {"p": p, "q": q, "m": m, "n": n,
            "condition": float(np.linalg.cond(A)) if not degenerate else float("inf"),
            "degenerate": degenerate, "rank": rank}


def evaluate(p, q, x) -> np.ndarray:
    """``P(x) / Q(x)`` by Horner on each, with the denominator's zeros reported as ``inf``."""
    pa = np.atleast_1d(np.asarray(p, dtype=float))
    qa = np.atleast_1d(np.asarray(q, dtype=float))
    z = np.atleast_1d(np.asarray(x, dtype=float))
    horner = lambda co, t: sum(co[k] * t ** k for k in range(co.size))
    num = horner(pa, z)
    den = horner(qa, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return out


def approximant(taylor, m: int, n: int):
    """A callable for the ``[m/n]`` approximant, which is what most callers actually want."""
    out = coefficients(taylor, m, n)
    p, q = out["p"], out["q"]
    return lambda x: evaluate(p, q, x)


def matches_to_order(taylor, m: int, n: int, n_terms=None) -> dict:
    """Verify the defining property: the series of ``P/Q`` matches ``f`` to order ``m + n``.

    Checked by expanding ``P/Q`` as a power series through long division, rather than by
    sampling near zero, so the statement is about coefficients and not about a tolerance.
    """
    c = np.atleast_1d(np.asarray(taylor, dtype=float)).ravel()
    out = coefficients(taylor, m, n)
    p, q = out["p"], out["q"]
    total = int(m) + int(n) + 1 if n_terms is None else int(n_terms)
    total = min(total, c.size)
    series = np.zeros(total)
    for i in range(total):
        acc = float(p[i]) if i < p.size else 0.0
        for k in range(1, min(i, q.size - 1) + 1):
            acc -= q[k] * series[i - k]
        series[i] = acc
    gap = np.abs(series - c[:total])
    scale = max(float(np.max(np.abs(c[:total]))), 1e-300)
    return {"series_of_the_approximant": series, "taylor": c[:total],
            "worst_gap": float(np.max(gap)), "relative_gap": float(np.max(gap) / scale),
            "matches": bool(np.max(gap) <= 1e-8 * scale)}


def matching_order(taylor, m: int, n: int, tol: float = 1e-8) -> int:
    """How many leading Taylor coefficients the approximant actually reproduces.

    For a non-degenerate ``[m/n]`` this is ``m + n + 1``, which is the whole point of the
    construction. For a degenerate entry it is smaller, and **how much smaller is not predicted
    by the rank of the Toeplitz system**: the least norm solve returns a valid rational
    approximant of the block, not necessarily the block's canonical representative. Measured,
    ``[2/3]`` of ``tan`` has rank 2 and matches 3 terms, while ``[0/1]`` of ``tan`` has rank 0
    and matches 1.

    So this is reported as a measurement rather than derived, and any claim about a degenerate
    entry's order should come from here.
    """
    c = np.atleast_1d(np.asarray(taylor, dtype=float)).ravel()
    limit = min(int(m) + int(n) + 1, c.size)
    best = 0
    for terms in range(1, limit + 1):
        if matches_to_order(c, m, n, n_terms=terms)["relative_gap"] <= float(tol):
            best = terms
        else:
            break
    return best


def system_conditioning(taylor, m: int, n_values=None) -> dict:
    """The Toeplitz system's condition number as the denominator degree grows.

    This is where Pade breaks. The matrix is built from consecutive Taylor coefficients, which
    for a rapidly converging series differ by many orders of magnitude, so the rows differ in
    scale and the matrix is badly conditioned before anything interesting has happened.
    """
    c = np.atleast_1d(np.asarray(taylor, dtype=float)).ravel()
    ns = ([1, 2, 3, 4, 5] if n_values is None else [int(v) for v in np.atleast_1d(n_values)])
    rows = []
    for n in ns:
        if c.size < int(m) + n + 1:
            continue
        rows.append((n, coefficients(c, m, n)["condition"]))
    return {"denominator_degrees": np.asarray([r[0] for r in rows]),
            "conditions": np.asarray([r[1] for r in rows])}


# --------------------------------------------------------------------------- poles


def pole_locations(taylor, m: int, n: int) -> dict:
    """The roots of ``Q``, which are the approximant's poles.

    The reason to look at them is that a good Pade approximant puts its poles where the function
    has its singularities. That is the mechanism and it is checkable. The ``[3/4]`` approximant
    of ``tan`` from 16 Taylor coefficients has real poles at ``+/-1.571233``, against the true
    ``pi/2 = 1.570796``: an error of ``4.4e-4`` from a construction that was never told where the
    pole was.

    It captures only the **nearest** pair. Its other two poles sit at ``+/-6.52``, nowhere near
    the true ``3 pi / 2 = 4.712``. A Pade approximant models the singularities that limit the
    Taylor radius, and invents the rest.
    """
    out = coefficients(taylor, m, n)
    q = out["q"]
    if q.size <= 1:
        return {"poles": np.zeros(0, dtype=complex), "real_poles": np.zeros(0),
                "q": q, "nearest_to_origin": float("inf")}
    roots = np.roots(q[::-1])
    real = np.real(roots[np.abs(np.imag(roots)) < 1e-9 * np.maximum(np.abs(roots), 1.0)])
    return {"poles": roots, "real_poles": np.sort(real), "q": q,
            "nearest_to_origin": float(np.min(np.abs(roots)))}


def spurious_poles(taylor, m: int, n: int, lo: float = -1.0, hi: float = 1.0,
                   tol: float = 1e-6) -> dict:
    """Find poles inside the interval and the nearly cancelling zeros beside them.

    A **Froissart doublet** is a pole of ``Q`` sitting almost on top of a zero of ``P``. In exact
    arithmetic they would cancel exactly and the approximant would be fine. In floating point
    they miss by a little, so the approximant has a genuine pole where the function does not,
    and its position depends on the rounding rather than on the function.

    Returns the doublets found, with the gap between each pole and its nearest zero.
    """
    out = coefficients(taylor, m, n)
    p, q = out["p"], out["q"]
    poles = np.roots(q[::-1]) if q.size > 1 else np.zeros(0, dtype=complex)
    zeros = np.roots(p[::-1]) if p.size > 1 and np.any(p[1:] != 0.0) else np.zeros(0,
                                                                                   dtype=complex)
    inside = [z for z in poles
              if float(lo) <= float(np.real(z)) <= float(hi)
              and abs(float(np.imag(z))) <= float(tol) * max(abs(float(np.real(z))), 1.0)]
    doublets = []
    for z in inside:
        if zeros.size:
            k = int(np.argmin(np.abs(zeros - z)))
            doublets.append((complex(z), complex(zeros[k]), float(abs(zeros[k] - z))))
        else:
            doublets.append((complex(z), complex(np.nan), float("inf")))
    return {"poles_inside": np.asarray([d[0] for d in doublets], dtype=complex),
            "nearest_zeros": np.asarray([d[1] for d in doublets], dtype=complex),
            "gaps": np.asarray([d[2] for d in doublets]),
            "count": len(doublets),
            "has_a_doublet": bool(any(d[2] < 1e-3 for d in doublets))}


# --------------------------------------------------------------------------- comparisons


def _values(f, x):
    x = np.asarray(x, dtype=float)
    try:
        v = np.asarray(f(x), dtype=float)
        if v.shape != x.shape:
            raise ValueError
    except (TypeError, ValueError):
        v = np.asarray([float(f(s)) for s in x], dtype=float)
    return v


def against_taylor(f, taylor, m: int, n: int, lo: float = -1.0, hi: float = 1.0,
                   n_probe: int = 2001) -> dict:
    """The ``[m/n]`` approximant against the Taylor polynomial of the same total degree.

    The comparison is fair by construction: both use ``m + n + 1`` Taylor coefficients and both
    cost ``O(m + n)`` to evaluate. Any difference is what the ratio structure buys.
    """
    c = np.atleast_1d(np.asarray(taylor, dtype=float)).ravel()
    total = int(m) + int(n)
    x = np.linspace(float(lo), float(hi), int(n_probe))
    truth = _values(f, x)
    good = np.isfinite(truth)
    out = coefficients(c, m, n)
    rat = evaluate(out["p"], out["q"], x)
    tay = sum(c[k] * x ** k for k in range(min(total + 1, c.size)))
    finite = good & np.isfinite(rat) & np.isfinite(tay)
    e_rat = float(np.max(np.abs(rat[finite] - truth[finite])))
    e_tay = float(np.max(np.abs(tay[finite] - truth[finite])))
    return {"m": int(m), "n": int(n), "coefficients_used": total + 1,
            "pade_error": e_rat, "taylor_error": e_tay,
            "ratio": e_tay / max(e_rat, 1e-300),
            "points_compared": int(np.sum(finite)),
            "condition": out["condition"]}


def table(f, taylor, total_degree: int, lo: float = -1.0, hi: float = 1.0,
          n_probe: int = 2001) -> dict:
    """Every split of a fixed total degree into ``[m/n]``, so the best split can be seen.

    The diagonal ``m = n`` is the usual recommendation and it is usually right, but not always,
    and the point of a table is that the recommendation is checkable rather than folklore.
    """
    total = int(total_degree)
    rows = []
    for n in range(total + 1):
        m = total - n
        try:
            out = against_taylor(f, taylor, m, n, lo, hi, n_probe)
        except (ValueError, np.linalg.LinAlgError):
            continue
        rows.append((m, n, out["pade_error"], out["condition"]))
    if not rows:
        raise ValueError(f"no [m/n] split of degree {total} could be built")
    errs = np.asarray([r[2] for r in rows])
    k = int(np.argmin(errs))
    return {"m": np.asarray([r[0] for r in rows]), "n": np.asarray([r[1] for r in rows]),
            "errors": errs, "conditions": np.asarray([r[3] for r in rows]),
            "best_split": (rows[k][0], rows[k][1]),
            "diagonal_is_best": bool(abs(rows[k][0] - rows[k][1]) <= 1)}


# --------------------------------------------------------------------------- series


def taylor_coefficients(name: str, count: int) -> np.ndarray:
    """Exact Taylor coefficients for the functions used in the lesson, in closed form.

    Computing them by repeated numerical differentiation would introduce an error before the
    method under test had done anything, and lesson 61 measures exactly how bad that is. Closed
    forms keep the experiment about Pade.
    """
    key = str(name).lower()
    k = int(count)
    if k < 1:
        raise ValueError(f"need at least one coefficient, got {k}")
    if key == "exp":
        return np.asarray([1.0 / math.factorial(i) for i in range(k)])
    if key == "log1p":                                   # log(1 + x)
        return np.asarray([0.0] + [((-1.0) ** (i + 1)) / i for i in range(1, k)])[:k]
    if key == "tan":
        # from the recurrence for the tangent numbers, exact in rational arithmetic
        from fractions import Fraction
        series = [Fraction(0), Fraction(1)]
        while len(series) < k:
            i = len(series)
            # tan' = 1 + tan^2 gives (i) a_i = sum_{j} a_j a_{i-1-j} for i >= 2
            acc = sum(series[j] * series[i - 1 - j] for j in range(i) if i - 1 - j < len(series))
            series.append(acc / i)
        return np.asarray([float(v) for v in series[:k]])
    if key == "arctan":
        return np.asarray([0.0 if i % 2 == 0 else ((-1.0) ** ((i - 1) // 2)) / i
                           for i in range(k)])
    if key == "one_over_one_plus_x":                     # 1 / (1 + x)
        return np.asarray([(-1.0) ** i for i in range(k)])
    if key == "sqrt1p":                                  # sqrt(1 + x)
        out = np.empty(k)
        out[0] = 1.0
        for i in range(1, k):
            out[i] = out[i - 1] * (0.5 - (i - 1)) / i
        return out
    raise ValueError(f"unknown series {name!r}, expected one of "
                     "exp, log1p, tan, arctan, one_over_one_plus_x, sqrt1p")
