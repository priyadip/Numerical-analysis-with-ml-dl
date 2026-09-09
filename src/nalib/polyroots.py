"""Root finding for polynomials, where the extra structure buys extra methods.

A polynomial is not just a function. It has a degree, so you know how many roots to expect. Its
coefficients are finite data, so you can reason about them directly. Both facts enable things no
general root finder can do:

- **count** the real roots in an interval before finding any of them
- find **all** the roots, including the complex ones, from real starting data
- **deflate** a root away once found, and continue on a smaller problem

The price is that every method here works with **coefficients**, and lesson 12 showed that
coefficients are a badly conditioned way to describe roots. Each routine below carries a note
about how much that costs it.

Coefficient order is descending throughout, matching `numpy.polyval` and `nalib.polynomials`.

Used by lesson 13.
"""

from __future__ import annotations

import numpy as np

from .polynomials import horner_with_derivative, synthetic_division

# ---------------------------------------------------------------- counting roots


def descartes_sign_changes(coeffs) -> int:
    """Number of sign changes in the coefficient sequence, ignoring zeros."""
    signs = [np.sign(c) for c in np.asarray(coeffs, dtype=float) if c != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def descartes_bounds(coeffs) -> dict:
    """Descartes' rule of signs: bounds on the number of positive and negative roots.

    The number of positive real roots equals the number of sign changes in `p(x)`, or is less
    than it by an even number. Applying the same rule to `p(-x)` bounds the negative roots.

    This is a genuinely useful *pre*-check: it costs no function evaluations at all and it
    tells you what to look for. It gives a bound, not a count. `sturm_count` gives the exact
    number.

    >>> descartes_bounds([1, -6, 11, -6])["positive"]     # (x-1)(x-2)(x-3)
    [3, 1]
    """
    c = np.asarray(coeffs, dtype=float)
    n = c.size - 1
    # p(-x): flip the sign of every coefficient whose power is odd
    powers = np.arange(n, -1, -1)
    c_neg = c * (-1.0) ** powers

    pos = descartes_sign_changes(c)
    neg = descartes_sign_changes(c_neg)
    return {
        "positive": list(range(pos, -1, -2)),
        "negative": list(range(neg, -1, -2)),
        "sign_changes_p": pos,
        "sign_changes_p_minus_x": neg,
        "degree": n,
    }


def sturm_sequence(coeffs) -> list[np.ndarray]:
    """Build the Sturm chain of a squarefree polynomial.

        p_0 = p,   p_1 = p',   p_(k+1) = -remainder(p_(k-1), p_k)

    The chain ends when the remainder is constant. Combined with `sturm_count` this gives the
    **exact** number of distinct real roots in an interval, which no method based on sampling
    can guarantee.

    The polynomial should be squarefree. A repeated root makes the chain terminate early, and
    the count then refers to distinct roots only.
    """
    c = np.trim_zeros(np.asarray(coeffs, dtype=float), "f")
    if c.size == 0:
        raise ValueError("the zero polynomial has no Sturm chain")
    chain = [c]
    if c.size == 1:
        return chain
    chain.append(np.polyder(c))
    while chain[-1].size > 1:
        _q, rem = np.polydiv(chain[-2], chain[-1])
        rem = np.trim_zeros(np.atleast_1d(rem), "f")
        if rem.size == 0 or np.allclose(rem, 0.0):
            break
        chain.append(-rem)
    return chain


def sturm_sign_changes(chain, x: float) -> int:
    """Sign changes in the Sturm chain evaluated at x, skipping zeros."""
    vals = [float(np.polyval(p, x)) for p in chain]
    signs = [np.sign(v) for v in vals if v != 0.0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def sturm_count(coeffs, a: float, b: float) -> int:
    """Exact number of DISTINCT real roots of p in the half-open interval (a, b].

    Sturm's theorem: the count is the drop in the number of sign changes of the chain between
    a and b. This is exact, needs no sampling, and cannot miss a root however narrowly spaced.

    >>> sturm_count([1, -6, 11, -6], 0.0, 4.0)     # roots at 1, 2, 3
    3
    >>> sturm_count([1, -6, 11, -6], 1.5, 4.0)
    2
    """
    chain = sturm_sequence(coeffs)
    return sturm_sign_changes(chain, a) - sturm_sign_changes(chain, b)


def cauchy_bound(coeffs) -> float:
    """All roots satisfy |r| <= 1 + max|a_i / a_0|, real or complex.

    Gives a finite box to search in, which every bracketing method needs and no general
    function provides.
    """
    c = np.asarray(coeffs, dtype=float)
    if c[0] == 0:
        raise ValueError("leading coefficient must be nonzero")
    return 1.0 + float(np.max(np.abs(c[1:] / c[0]))) if c.size > 1 else 0.0


# ---------------------------------------------------------------- Birge-Vieta


def birge_vieta(coeffs, x0: float, tol: float = 1e-14, max_iter: int = 100) -> dict:
    """Newton's method on a polynomial, using synthetic division for value and slope.

    Also called the Newton-Horner method. The point is efficiency: one pass of synthetic
    division gives `p(x)`, and running the same recurrence on the intermediate quotients gives
    `p'(x)` at no extra cost. So a Newton step costs about two Horner passes rather than a
    separate evaluation of a symbolically differentiated polynomial.

    Converges quadratically at a simple root, exactly like Newton, because it **is** Newton.
    """
    c = np.asarray(coeffs, dtype=float)
    x = float(x0)
    xs = [x]
    for _ in range(max_iter):
        p, dp = horner_with_derivative(c, x)
        if dp == 0.0:
            return {"root": x, "iterates": np.array(xs), "converged": False,
                    "message": "derivative vanished"}
        step = p / dp
        x -= step
        xs.append(x)
        if abs(step) <= tol:
            return {"root": x, "iterates": np.array(xs), "converged": True,
                    "message": "converged"}
    return {"root": x, "iterates": np.array(xs), "converged": False,
            "message": "iteration limit reached"}


# ---------------------------------------------------------------- Bairstow


def bairstow(coeffs, r0: float = 1.0, s0: float = 1.0,
             tol: float = 1e-13, max_iter: int = 200) -> dict:
    """Extract a quadratic factor x^2 - r x - s from a real polynomial.

    The point: dividing by a **quadratic** rather than a linear factor lets you find a
    **complex conjugate pair** while staying entirely in real arithmetic. Muller reaches
    complex roots by going complex; Bairstow reaches them without ever leaving the reals.

    The method is Newton's method in two variables (r, s), driving the two remainder
    coefficients to zero. The partial derivatives come from a second synthetic division, which
    is the same trick Birge-Vieta uses one dimension down.

    Returns the quotient polynomial and the two roots of the quadratic factor.
    """
    a = np.asarray(coeffs, dtype=float)
    n = a.size - 1
    if n < 2:
        raise ValueError("bairstow needs degree at least 2")
    r, s = float(r0), float(s0)

    for it in range(max_iter):
        # first synthetic division: b holds the quotient plus the remainder
        b = np.zeros(n + 1)
        b[0] = a[0]
        b[1] = a[1] + r * b[0]
        for k in range(2, n + 1):
            b[k] = a[k] + r * b[k - 1] + s * b[k - 2]

        # second synthetic division on b: c gives the partial derivatives
        c = np.zeros(n + 1)
        c[0] = b[0]
        c[1] = b[1] + r * c[0]
        for k in range(2, n):
            c[k] = b[k] + r * c[k - 1] + s * c[k - 2]

        det = c[n - 2] ** 2 - c[n - 3] * c[n - 1] if n >= 3 else c[n - 2] ** 2
        if det == 0.0:
            return {"converged": False, "message": "singular Newton system",
                    "r": r, "s": s, "iterations": it}

        if n >= 3:
            dr = (-b[n - 1] * c[n - 2] + b[n] * c[n - 3]) / det
            ds = (-b[n] * c[n - 2] + b[n - 1] * c[n - 1]) / det
        else:
            dr = -b[n - 1] * c[n - 2] / det
            ds = -b[n] * c[n - 2] / det

        r += dr
        s += ds
        if abs(dr) + abs(ds) <= tol:
            quotient = b[: n - 1]
            disc = complex(r * r + 4 * s)
            root1 = (r + np.sqrt(disc)) / 2
            root2 = (r - np.sqrt(disc)) / 2
            return {"converged": True, "message": "converged", "r": r, "s": s,
                    "quotient": quotient, "roots": (root1, root2),
                    "iterations": it + 1}

    return {"converged": False, "message": "iteration limit reached",
            "r": r, "s": s, "iterations": max_iter}


# ---------------------------------------------------------------- Graeffe


def graeffe_step(coeffs) -> np.ndarray:
    """One Graeffe root-squaring step.

    Produces the polynomial whose roots are the **squares** of the input's roots. Repeating
    separates the roots by magnitude exponentially fast, so after m steps the ratios of
    consecutive coefficients give the root magnitudes to the power 2^m.

    The catch is severe and is the reason nobody uses this: squaring the roots squares the
    coefficients, so after a handful of steps they overflow a double. `graeffe_magnitudes`
    reports how many steps are usable.
    """
    a = np.asarray(coeffs, dtype=float)
    n = a.size - 1
    out = np.zeros(n + 1)
    for k in range(n + 1):
        total = a[k] ** 2
        j = 1
        while k - j >= 0 and k + j <= n:
            total += 2.0 * (-1.0) ** j * a[k - j] * a[k + j]
            j += 1
        out[k] = total
    return out


def graeffe_magnitudes(coeffs, steps: int = 5) -> dict:
    """Estimate root magnitudes by repeated squaring.

    After m steps, |r_k| is about |b_k / b_(k-1)| ** (1 / 2**m), where b are the coefficients
    of the m-times-squared polynomial.

    Works only for roots of distinct magnitude, and only until the coefficients overflow.
    Both limitations are reported in the result rather than hidden.
    """
    a = np.asarray(coeffs, dtype=float)
    n = a.size - 1
    history = [a]
    used = 0
    for m in range(steps):
        with np.errstate(over="ignore", invalid="ignore"):
            nxt = graeffe_step(history[-1])
        if not np.all(np.isfinite(nxt)) or np.max(np.abs(nxt)) > 1e300:
            break
        history.append(nxt)
        used = m + 1

    b = history[-1]
    power = 2.0 ** used
    with np.errstate(divide="ignore", invalid="ignore"):
        mags = np.array([abs(b[k] / b[k - 1]) ** (1.0 / power) for k in range(1, n + 1)])
    return {"magnitudes": mags, "steps_used": used, "steps_requested": steps,
            "overflowed": used < steps, "history": history}


# ---------------------------------------------------------------- companion matrix


def companion_matrix(coeffs) -> np.ndarray:
    """The companion matrix of a monic polynomial: its eigenvalues are the roots.

    This turns root finding into an **eigenvalue problem**, which is what `numpy.roots` does.
    The payoff is that all roots come out at once, real and complex, with no starting guess and
    no deflation, using the backward stable machinery of lesson 37.

    The cost is O(n^3) rather than O(n) per root, and the conditioning of the eigenvalue
    problem is not obviously the same as the conditioning of the roots. Lesson 12 exercise 5.2
    asks about exactly that.
    """
    c = np.asarray(coeffs, dtype=float)
    if c[0] == 0:
        raise ValueError("leading coefficient must be nonzero")
    c = c / c[0]
    n = c.size - 1
    C = np.zeros((n, n))
    C[0, :] = -c[1:]
    C[1:, :-1] = np.eye(n - 1)
    return C


def roots_via_companion(coeffs) -> np.ndarray:
    """All roots, as eigenvalues of the companion matrix. This is what numpy.roots does."""
    return np.linalg.eigvals(companion_matrix(coeffs))


# ---------------------------------------------------------------- deflation


def all_roots_by_deflation(coeffs, tol: float = 1e-13, start: float = 0.0) -> dict:
    """Find every real root by finding one, dividing it out, and repeating.

    Convenient, and slightly dangerous. Each root is found from the **deflated** polynomial,
    which already carries the error of every root removed before it, so accuracy degrades as
    you go. The standard mitigation is to find roots in increasing order of magnitude, and to
    **polish** each one with a Newton step on the ORIGINAL polynomial at the end.

    `start` controls the initial guess for each deflated solve, and therefore the ORDER in
    which roots are found. Starting near zero finds small roots first, which is the safe
    order. Starting large finds the big ones first, which is the order that degrades.

    Returns both the raw and the polished roots so the difference can be measured.
    """
    c = list(np.asarray(coeffs, dtype=float))
    original = np.asarray(coeffs, dtype=float)
    raw = []

    while len(c) > 1:
        deg = len(c) - 1
        if deg == 1:
            raw.append(-c[1] / c[0])
            break
        found = birge_vieta(c, start, tol=tol)
        if not found["converged"]:
            found = birge_vieta(c, start + 1.0, tol=tol)
        if not found["converged"]:
            break
        r = found["root"]
        raw.append(r)
        c, _rem = synthetic_division(c, r)

    raw = np.array(raw)
    polished = np.array([birge_vieta(original, r, tol=tol)["root"] for r in raw])
    return {"raw": raw, "polished": polished, "n_found": raw.size}
