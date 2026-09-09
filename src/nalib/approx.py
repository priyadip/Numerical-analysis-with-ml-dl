"""Function approximation: what "closest" means before any algorithm is chosen.

The change of question
----------------------
Part 7 asked for a curve **through** the data. This part asks for the curve **closest** to a
function, which is a different question and needs a definition of closest before it has an
answer. Three definitions are in use and they give three different answers:

=========  ================================  =============================================
norm       what it measures                  what it is used for
=========  ================================  =============================================
``L1``     total area of the error           robust fitting, where outliers must not shout
``L2``     root mean square error            least squares, statistics, energy
``Linf``   the single worst point            guarantees, hardware, library functions
=========  ================================  =============================================

`function_norm` computes all three. The lesson's point is that they are genuinely different
targets, and each wins in its own norm by construction. On ``exp`` at degree 3,
`minimax_vs_least_squares` measures the best ``L2`` polynomial as **2.02 times worse** than the
best ``Linf`` one at its worst point, while being 14 percent better in the ``L2`` norm. Neither
is "the" best approximation; the norm has to be chosen first.

Existence, and why it is not obvious
------------------------------------
A best approximation from a finite dimensional subspace always exists. The argument is
compactness: the error as a function of the coefficients is continuous, it grows without bound
as the coefficients do, so the search can be restricted to a closed bounded set, where a
continuous function attains its minimum. `existence_report` runs the two halves of that argument
numerically, on a grid of coefficients, so it can be seen rather than believed.

**Uniqueness is a separate question and it needs more.** In ``L2`` it follows from strict
convexity of the norm. In ``Linf`` it needs the **Haar condition** of lesson 53's exercise 5.3,
which polynomials on an interval satisfy and which fails in more than one dimension. In ``L1``
it can genuinely fail.

Equioscillation
---------------
Chebyshev's theorem: ``p`` is the best ``Linf`` approximation of degree ``n`` to ``f`` on
``[a, b]`` **if and only if** ``f - p`` attains ``+E`` and ``-E`` alternately at ``n + 2`` or
more points, where ``E`` is the maximum error. That is a complete characterisation, checkable
after the fact, and it is what `equioscillation_report` counts and what `remez` iterates on.

Weierstrass
-----------
Every continuous function on a closed interval is a uniform limit of polynomials. `bernstein`
gives Bernstein's constructive proof, which is the same basis as lesson 52's Bezier curves. It
converges, and it converges at ``O(1/n)``, which `weierstrass_rate` measures. That rate is so
poor that the theorem is never used as a method: it says a good approximation exists, and the
rest of this part is about finding one.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- norms


def _sample(f, lo, hi, n_probe):
    """Evaluate ``f`` on a uniform grid, accepting either a vectorised or a scalar callable."""
    lo = float(lo)
    hi = float(hi)
    if not hi > lo:
        raise ValueError(f"need lo < hi, got [{lo}, {hi}]")
    t = np.linspace(lo, hi, int(n_probe))
    try:
        v = np.asarray(f(t), dtype=float)
        if v.shape != t.shape:
            raise ValueError
    except (TypeError, ValueError):
        v = np.asarray([float(f(s)) for s in t], dtype=float)
    return t, v


def function_norm(f, lo: float, hi: float, p=2, n_probe: int = 20001) -> float:
    """``||f||_p`` on ``[lo, hi]``, by the trapezoid rule for finite ``p``.

    ``p`` may be 1, any positive number, or ``np.inf``. The infinity norm is the maximum over
    the sample grid, so it is a lower bound on the true supremum, exact to the grid's
    resolution.
    """
    t, v = _sample(f, lo, hi, n_probe)
    if p == np.inf or (isinstance(p, str) and p.lower() in ("inf", "max")):
        return float(np.max(np.abs(v)))
    q = float(p)
    if q <= 0.0:
        raise ValueError(f"p must be positive or inf, got {p}")
    return float(np.trapezoid(np.abs(v) ** q, t) ** (1.0 / q))


def norm_comparison(f, g, lo: float, hi: float, n_probe: int = 20001) -> dict:
    """The three norms of ``f - g`` side by side, with the point where the maximum is attained."""
    t, a = _sample(f, lo, hi, n_probe)
    _, b = _sample(g, lo, hi, n_probe)
    d = a - b
    k = int(np.argmax(np.abs(d)))
    return {"l1": float(np.trapezoid(np.abs(d), t)),
            "l2": float(np.sqrt(np.trapezoid(d * d, t))),
            "linf": float(np.abs(d[k])),
            "worst_at": float(t[k]),
            "rms": float(np.sqrt(np.trapezoid(d * d, t) / (float(hi) - float(lo))))}


def weighted_inner_product(f, g, lo: float, hi: float, weight=None,
                           n_probe: int = 20001) -> float:
    """``<f, g>_w = integral f g w``, the inner product every ``L2`` statement is about."""
    t, a = _sample(f, lo, hi, n_probe)
    _, b = _sample(g, lo, hi, n_probe)
    if weight is None:
        w = np.ones_like(t)
    else:
        _, w = _sample(weight, lo, hi, n_probe)
    return float(np.trapezoid(a * b * w, t))


# --------------------------------------------------------------------------- best L2


def best_l2_polynomial(f, degree: int, lo: float = -1.0, hi: float = 1.0,
                       n_probe: int = 4001, weight=None) -> dict:
    """The polynomial of the given degree minimising ``||f - p||_2``.

    Solved as a discrete least squares problem in the **Chebyshev** basis mapped to
    ``[lo, hi]``, not the monomial basis. That choice is not cosmetic: the monomial normal
    equations are the Hilbert matrix, whose condition number grows like ``e^{3.5 n}``, and
    `basis_conditioning` measures both.

    Returns the coefficients in that basis together with a callable.
    """
    n = int(degree)
    if n < 0:
        raise ValueError(f"degree must be non-negative, got {n}")
    lo = float(lo)
    hi = float(hi)
    t, v = _sample(f, lo, hi, n_probe)
    u = (2.0 * t - (hi + lo)) / (hi - lo)
    A = np.stack([np.cos(k * np.arccos(np.clip(u, -1.0, 1.0))) for k in range(n + 1)], axis=1)
    if weight is None:
        w = np.ones_like(t)
    else:
        _, w = _sample(weight, lo, hi, n_probe)
        if np.any(w < 0.0):
            raise ValueError("the weight must be non-negative")
    root_w = np.sqrt(w)
    c, *_ = np.linalg.lstsq(A * root_w[:, None], v * root_w, rcond=None)

    def evaluate(s):
        z = np.atleast_1d(np.asarray(s, dtype=float))
        y = np.clip((2.0 * z - (hi + lo)) / (hi - lo), -1.0, 1.0)
        theta = np.arccos(y)
        return np.stack([np.cos(k * theta) for k in range(n + 1)], axis=1) @ c

    return {"coefficients": c, "basis": "chebyshev", "interval": (lo, hi),
            "evaluate": evaluate,
            "l2_error": float(np.sqrt(np.trapezoid((A @ c - v) ** 2, t))),
            "linf_error": float(np.max(np.abs(A @ c - v)))}


def basis_conditioning(degree: int, lo: float = 0.0, hi: float = 1.0,
                       n_probe: int = 2001) -> dict:
    """Condition number of the ``L2`` Gram matrix in the monomial and Chebyshev bases.

    In the monomial basis on ``[0, 1]`` the Gram matrix is exactly the Hilbert matrix, whose
    condition number is famously ``O(e^{3.5 n})``. In the Chebyshev basis on ``[-1, 1]`` the
    Gram matrix is diagonal in the Chebyshev weight and near-diagonal in the flat one, so its
    condition number is bounded. Both are measured rather than quoted.
    """
    n = int(degree)
    lo = float(lo)
    hi = float(hi)
    t = np.linspace(lo, hi, int(n_probe))
    mono = np.stack([t ** k for k in range(n + 1)], axis=1)
    u = np.clip((2.0 * t - (hi + lo)) / (hi - lo), -1.0, 1.0)
    theta = np.arccos(u)
    cheb = np.stack([np.cos(k * theta) for k in range(n + 1)], axis=1)
    gram = lambda B: np.stack([[np.trapezoid(B[:, i] * B[:, j], t) for j in range(n + 1)]
                               for i in range(n + 1)])
    return {"degree": n,
            "monomial_condition": float(np.linalg.cond(gram(mono))),
            "chebyshev_condition": float(np.linalg.cond(gram(cheb)))}


# --------------------------------------------------------------------------- minimax


def equioscillation_report(f, p, lo: float, hi: float, degree: int,
                           n_probe: int = 20001, tol: float = 1e-3) -> dict:
    """Count the alternations of ``f - p``, which is what Chebyshev's theorem asks about.

    An **alternation** is a point where the error reaches the maximum in magnitude and with the
    opposite sign to the previous such point. The theorem says the best degree ``n``
    approximation has at least ``n + 2`` of them, so this is a certificate that can be checked
    without knowing the answer.

    ``tol`` is the fraction of the maximum an extremum must reach to count. It has to be
    loose enough that a near-extremum of a nearly-best approximation still registers, and tight
    enough that ordinary wiggles do not.
    """
    t, a = _sample(f, lo, hi, n_probe)
    _, b = _sample(p, lo, hi, n_probe)
    e = a - b
    peak = float(np.max(np.abs(e)))
    if peak == 0.0:
        return {"alternations": 0, "required": int(degree) + 2, "is_best": True,
                "max_error": 0.0, "points": np.zeros(0), "signs": np.zeros(0)}
    keep = np.abs(e) >= (1.0 - float(tol)) * peak
    idx = np.where(keep)[0]
    points, signs = [], []
    for i in idx:
        s = 1 if e[i] > 0 else -1
        if signs and s == signs[-1]:
            if abs(e[i]) > abs(e[points[-1]]):
                points[-1] = int(i)
            continue
        points.append(int(i))
        signs.append(s)
    need = int(degree) + 2
    return {"alternations": len(points), "required": need,
            "is_best": bool(len(points) >= need),
            "max_error": peak,
            "points": t[np.asarray(points, dtype=int)] if points else np.zeros(0),
            "signs": np.asarray(signs, dtype=int)}


def _alternating_extrema(err, m):
    """The reference for the next Remez step: ``m`` alternating extrema of the error.

    Split the error into runs of constant sign, take the largest in magnitude from each, and
    then **drop from an end** until exactly ``m`` remain. Dropping from an end is what keeps the
    signs alternating; dropping the globally smallest, which is the obvious thing to do, can
    remove an interior point and leave two neighbours of the same sign, and the exchange then
    cycles instead of converging.

    That failure is not hypothetical. On ``cosh`` at degree 4, where the best degree 4 and
    degree 5 approximations coincide and there are ``n + 3`` extrema rather than ``n + 2``, the
    interior-drop version ran the full 60 iterations with the error oscillating between
    ``4.5e-5`` and ``2.1e-4`` and never settled.
    """
    sign = np.sign(err)
    sign[sign == 0] = 1.0
    breaks = np.where(np.diff(sign) != 0)[0]
    edges = np.concatenate(([0], breaks + 1, [err.size]))
    cand = [int(a + np.argmax(np.abs(err[a:b]))) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    while len(cand) > m:
        if abs(err[cand[0]]) < abs(err[cand[-1]]):
            cand = cand[1:]
        else:
            cand = cand[:-1]
    return np.asarray(cand, dtype=int)


def remez(f, degree: int, lo: float = -1.0, hi: float = 1.0, max_iterations: int = 60,
          tol: float = 1e-12, n_probe: int = 20001) -> dict:
    """The Remez exchange algorithm: the best ``Linf`` polynomial, computed.

    Start from ``n + 2`` reference points, solve the linear system that makes the error
    alternate ``+E, -E, ...`` exactly there, then move the reference to the extrema of the
    resulting error and repeat. It converges quadratically and is what a library function
    generator actually runs.

    The linear system is

        sum_k c_k T_k(x_i)  +  (-1)^i E  =  f(x_i),      i = 0 ... n+1

    with unknowns ``c_0 ... c_n`` and ``E``: ``n + 2`` equations in ``n + 2`` unknowns.

    Three cases need care and all three were found by testing rather than anticipated.

    **A symmetric function on a symmetric interval.** For an odd ``f`` on ``[-a, a]`` the
    Chebyshev extrema starting reference is symmetric, the error of the first solve comes out
    odd, and the alternating column of the system is then orthogonal to it, so the solve returns
    ``E = 4.9e-17``: the "approximation" simply interpolates ``f`` at the reference. Its error
    then has only ``n + 1`` alternating extrema rather than ``n + 2``, the exchange has nothing
    to exchange, and the iteration stops on the first step. On ``sin(3t)`` at degree 7 that
    returned ``3.37e-4``. **The fix is to break the symmetry of the starting reference**, by
    shifting the interior points by a fixed fraction of their spacing. With that, the same case
    converges in 5 iterations to ``1.69e-4``, exactly half the error.

    **A degenerate degree.** When ``f`` is even and the degree is odd, or the other way round,
    the best approximation of degree ``n`` equals the best of degree ``n + 1`` and there are
    ``n + 3`` extrema. `_alternating_extrema` handles that by dropping from an end.

    **Convergence at machine precision.** The natural test ``|peak - E| <= tol * peak`` is a
    relative one, and it cannot be met once ``peak`` is itself at roundoff. On ``t^3`` at degree
    3, where the answer is exact, ``peak`` is ``5.6e-16`` and ``E`` is ``4.4e-17``, and the
    relative test would ask for agreement to ``5.6e-28``. The test therefore carries an absolute
    floor of a few ``eps`` times the size of ``f``, which is the accuracy the function values
    themselves have.

    Returns ``converged`` so a caller can tell a settled answer from an exhausted loop.
    """
    n = int(degree)
    if n < 0:
        raise ValueError(f"degree must be non-negative, got {n}")
    lo = float(lo)
    hi = float(hi)
    m = n + 2
    # the Chebyshev extrema, shifted off centre so a symmetric f cannot make the solve degenerate
    ref = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(np.arange(m) * np.pi / (m - 1))[::-1]
    if m > 2:
        ref[1:-1] = ref[1:-1] + 0.01 * (hi - lo) / m
    to_u = lambda s: np.clip((2.0 * s - (hi + lo)) / (hi - lo), -1.0, 1.0)
    basis = lambda s: np.stack([np.cos(k * np.arccos(to_u(s))) for k in range(n + 1)], axis=1)
    probe = np.linspace(lo, hi, int(n_probe))
    _, truth = _sample(f, lo, hi, n_probe)
    B = basis(probe)
    floor = 8.0 * float(np.finfo(float).eps) * max(float(np.max(np.abs(truth))), 1e-300)
    best_c = np.zeros(n + 1)
    best_peak = np.inf
    best_level = 0.0
    history = []
    converged = False
    for _ in range(int(max_iterations)):
        A = np.empty((m, m))
        A[:, :n + 1] = basis(ref)
        A[:, n + 1] = (-1.0) ** np.arange(m)
        rhs = np.asarray([float(np.atleast_1d(f(np.atleast_1d(s)))[0]) for s in ref])
        sol = np.linalg.solve(A, rhs)
        c, level = sol[:n + 1], float(sol[n + 1])
        err = truth - B @ c
        peak = float(np.max(np.abs(err)))
        history.append(peak)
        if peak < best_peak:
            best_peak, best_c, best_level = peak, c, level
        if abs(peak - abs(level)) <= float(tol) * max(peak, 1e-300) + floor:
            converged = True
            break
        idx = _alternating_extrema(err, m)
        if idx.size < m:                      # too few extrema to exchange: keep the best so far
            break
        new_ref = probe[idx]
        if float(np.max(np.abs(new_ref - ref))) == 0.0:
            converged = True
            break
        ref = new_ref

    def evaluate(s):
        return basis(np.atleast_1d(np.asarray(s, dtype=float))) @ best_c

    return {"coefficients": best_c, "basis": "chebyshev", "interval": (lo, hi),
            "evaluate": evaluate, "levelled_error": abs(best_level),
            "max_error": float(np.max(np.abs(truth - B @ best_c))),
            "iterations": len(history), "history": np.asarray(history),
            "converged": converged, "reference": ref}


def minimax_vs_least_squares(f, degree: int, lo: float = -1.0, hi: float = 1.0,
                             n_probe: int = 20001) -> dict:
    """Both best approximations of the same degree, measured in both norms.

    Each wins in its own norm, by construction. The interesting number is **how much** the
    ``L2`` fit gives up at its worst point, because that is what a guarantee costs.
    """
    ls = best_l2_polynomial(f, degree, lo, hi, n_probe=min(int(n_probe), 4001))
    mm = remez(f, degree, lo, hi, n_probe=n_probe)
    a = norm_comparison(f, ls["evaluate"], lo, hi, n_probe)
    b = norm_comparison(f, mm["evaluate"], lo, hi, n_probe)
    return {"least_squares": a, "minimax": b,
            "l2_ratio": a["l2"] / max(b["l2"], 1e-300),
            "linf_ratio": a["linf"] / max(b["linf"], 1e-300),
            "minimax_alternations": equioscillation_report(f, mm["evaluate"], lo, hi, degree),
            "least_squares_alternations": equioscillation_report(f, ls["evaluate"], lo, hi,
                                                                 degree)}


# --------------------------------------------------------------------------- existence


def existence_report(f, degree: int = 1, lo: float = -1.0, hi: float = 1.0,
                     span: float = 6.0, n_grid: int = 121, n_probe: int = 401) -> dict:
    """Run the two halves of the existence argument on a grid of coefficients.

    **Coercivity.** The error grows without bound as the coefficients do, so the minimum cannot
    escape to infinity and the search may be restricted to a bounded set.

    **Attainment.** On that closed bounded set the error is a continuous function of the
    coefficients, so it attains its minimum.

    Restricted to degree 1 so the coefficient space can actually be gridded and looked at;
    the argument does not depend on the degree.
    """
    n = int(degree)
    if n != 1:
        raise ValueError(f"the grid picture is drawn for degree 1, got {n}")
    t, v = _sample(f, lo, hi, n_probe)
    axis = np.linspace(-float(span), float(span), int(n_grid))
    err = np.empty((axis.size, axis.size))
    for i, a0 in enumerate(axis):
        for j, a1 in enumerate(axis):
            err[i, j] = float(np.max(np.abs(v - (a0 + a1 * t))))
    k = int(np.argmin(err))
    i, j = np.unravel_index(k, err.shape)
    edge = np.concatenate([err[0], err[-1], err[:, 0], err[:, -1]])
    return {"axis": axis, "error_surface": err,
            "best_coefficients": np.array([axis[i], axis[j]]),
            "best_error": float(err[i, j]),
            "smallest_on_the_boundary": float(np.min(edge)),
            "minimum_is_interior": bool(0 < i < axis.size - 1 and 0 < j < axis.size - 1),
            "grows_at_the_edge": bool(np.min(edge) > float(err[i, j]))}


def uniqueness_report(f, degree: int, lo: float = -1.0, hi: float = 1.0,
                      n_trials: int = 200, scale: float = 1e-6, rng=None,
                      n_probe: int = 4001) -> dict:
    """Perturb the best approximation and check that every neighbour is strictly worse.

    Uniqueness in ``Linf`` follows from the Haar condition; this is the numerical statement of
    it. A tie would show up as a neighbour with an error equal to the best one, and none is
    found.
    """
    gen = np.random.default_rng() if rng is None else rng
    best = remez(f, degree, lo, hi, n_probe=n_probe)
    c = best["coefficients"]
    t, v = _sample(f, lo, hi, n_probe)
    lo_f, hi_f = float(lo), float(hi)
    u = np.clip((2.0 * t - (hi_f + lo_f)) / (hi_f - lo_f), -1.0, 1.0)
    theta = np.arccos(u)
    B = np.stack([np.cos(k * theta) for k in range(c.size)], axis=1)
    base = float(np.max(np.abs(v - B @ c)))
    worst_gap = np.inf
    ties = 0
    for _ in range(int(n_trials)):
        d = c + float(scale) * gen.standard_normal(c.size)
        e = float(np.max(np.abs(v - B @ d)))
        worst_gap = min(worst_gap, e - base)
        ties += int(e <= base)
    return {"best_error": base, "trials": int(n_trials),
            "smallest_increase": float(worst_gap), "ties_or_better": int(ties),
            "is_strict_minimum": bool(ties == 0)}


# --------------------------------------------------------------------------- Weierstrass


def bernstein(f, n: int, t, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Bernstein's constructive proof of the Weierstrass theorem.

    ``B_n f (x) = sum_k f(k/n) C(n,k) x^k (1-x)^{n-k}`` on ``[0, 1]``, mapped to ``[lo, hi]``.
    It converges uniformly to ``f`` for every continuous ``f``, which proves the theorem, and
    it converges at ``O(1/n)``, which is why it is a proof and not a method.
    """
    m = int(n)
    if m < 1:
        raise ValueError(f"need at least one subdivision, got {m}")
    lo = float(lo)
    hi = float(hi)
    z = np.atleast_1d(np.asarray(t, dtype=float))
    u = (z - lo) / (hi - lo)
    k = np.arange(m + 1)
    nodes = lo + (hi - lo) * k / m
    try:
        fv = np.asarray(f(nodes), dtype=float)
        if fv.shape != nodes.shape:
            raise ValueError
    except (TypeError, ValueError):
        fv = np.asarray([float(f(s)) for s in nodes], dtype=float)
    binom = np.asarray([math.comb(m, int(j)) for j in k], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = binom[None, :] * u[:, None] ** k[None, :] * (1.0 - u)[:, None] ** (m - k)[None, :]
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    return W @ fv


def weierstrass_rate(f, n_values=None, lo: float = 0.0, hi: float = 1.0,
                     n_probe: int = 2001) -> dict:
    """Fit the observed order of the Bernstein approximation.

    The theory says ``O(1/n)`` for a Lipschitz ``f``, improving to ``O(1/n)`` still for smooth
    ``f``: the rate does **not** improve with smoothness, which is the whole objection to using
    it as a method. The fit measures which is true.
    """
    ns = ([4, 8, 16, 32, 64, 128] if n_values is None
          else [int(v) for v in np.atleast_1d(n_values)])
    t, v = _sample(f, lo, hi, n_probe)
    errs = [float(np.max(np.abs(bernstein(f, n, t, lo, hi) - v))) for n in ns]
    a = np.log(np.asarray(ns, dtype=float))
    order = float(-np.polyfit(a, np.log(np.maximum(errs, 1e-300)), 1)[0])
    return {"n_values": np.asarray(ns), "errors": np.asarray(errs),
            "fitted_order": order,
            "n_for_two_digits": next((n for n, e in zip(ns, errs) if e < 1e-2), None)}


def weierstrass_against_best(f, n_values=None, lo: float = -1.0, hi: float = 1.0,
                             n_probe: int = 4001) -> dict:
    """Bernstein at degree ``n`` against the **best** polynomial of the same degree.

    Both are degree ``n``. The gap is what the constructive proof gives away in exchange for
    being explicit, and it is large enough that the answer to "why not just use Bernstein" is
    settled by one table.
    """
    ns = ([2, 4, 8, 16, 32] if n_values is None else [int(v) for v in np.atleast_1d(n_values)])
    t, v = _sample(f, lo, hi, n_probe)
    rows = []
    for n in ns:
        bern = float(np.max(np.abs(bernstein(f, n, t, lo, hi) - v)))
        best = remez(f, n, lo, hi, n_probe=n_probe)["max_error"]
        rows.append((n, bern, best, bern / max(best, 1e-300)))
    return {"n_values": np.asarray([r[0] for r in rows]),
            "bernstein_error": np.asarray([r[1] for r in rows]),
            "best_error": np.asarray([r[2] for r in rows]),
            "ratio": np.asarray([r[3] for r in rows])}
