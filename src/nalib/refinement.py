"""Condition estimation without inverting, and iterative refinement.

Two practical techniques that finish Part 3, both of which take something the earlier lessons
treated as expensive or impossible and make it routine.

**Condition estimation.** Lesson 19 said to report kappa alongside every solution, then computed
it at O(n^3) by forming the inverse, which is exactly what lesson 17 said never to do. The fix
is that you do not need kappa accurately. You need its order of magnitude, and you already have
a factorization. Hager's method finds it in O(n^2) using a few triangular solves. This is what
LAPACK's ``gecon`` does, and it is why every serious solver can afford to report a condition
estimate.

**Iterative refinement.** A backward stable solver gives an answer good to about kappa*u. If the
residual is computed in **higher precision**, one cheap correction step can recover most of the
digits kappa took away. The cost is O(n^2) per step against the O(n^3) already spent, so the
whole thing is nearly free.

The reason refinement works is worth stating plainly, because it is not obvious: the residual
of a computed solution is a difference of two nearly equal quantities, so in working precision
it is almost entirely roundoff. Computing it more accurately is what turns it back into
information.
"""

from __future__ import annotations

import numpy as np

from .linalg import matrix_norm


def _solve_with(factor, b):
    """Apply a stored factorization to a right-hand side.

    Accepts what ``nalib.pivoting.plu_factor`` returns, a ``(L, U)`` pair from
    ``nalib.lu.lu_factor``, or a callable that solves directly. Keeping this in one place lets
    every routine below work with whichever factorization the caller already has.
    """
    from .cholesky import cholesky_solve
    from .lu import lu_solve
    from .pivoting import plu_solve

    if callable(factor):
        return np.asarray(factor(b), dtype=float).ravel()
    if isinstance(factor, dict):
        if "perm" in factor:
            return plu_solve(factor, b)
        if "L" in factor and "U" in factor:
            return lu_solve(factor["L"], factor["U"], b)
        if "L" in factor:
            return cholesky_solve(factor["L"], b)
    if isinstance(factor, (tuple, list)) and len(factor) == 2:
        return lu_solve(factor[0], factor[1], b)
    raise TypeError("unrecognised factorization; pass a plu_factor dict, an (L, U) pair, "
                    "or a callable that solves A x = b")


def estimate_norm_inverse(A, factor=None, p: int = 1, max_iter: int = 8) -> float:
    """Estimate ||A^-1||_1 by Hager's method, in O(n^2) given a factorization.

    The idea is that ``||A^-1||_1`` is the maximum of ``||A^-1 x||_1`` over ``||x||_1 <= 1``,
    which is a convex maximisation over a polytope, so the maximum sits at a vertex. Hager's
    algorithm walks between vertices, at each step:

    1. solve ``A y = x``,
    2. take ``xi = sign(y)``,
    3. solve ``A^T z = xi``,
    4. move to the coordinate direction where ``|z|`` is largest, and repeat.

    Each iteration costs two triangular solve pairs, so O(n^2), and it converges in a handful of
    steps regardless of n. It **underestimates**, because it finds a local maximum over vertices,
    but in practice it is within a small factor and that is all an order-of-magnitude estimate
    needs.

    This is the core of LAPACK's ``gecon``. Supplying ``factor`` avoids refactorizing; without
    it the routine factorizes once itself.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("condition estimation needs a square matrix")
    if p != 1:
        raise ValueError("Hager's method estimates the 1-norm; use p=1")

    if factor is None:
        from .pivoting import plu_factor

        fac = plu_factor(A)
        solve = lambda rhs: _solve_with(fac, rhs)                     # noqa: E731
        facT = plu_factor(A.T)
        solveT = lambda rhs: _solve_with(facT, rhs)                   # noqa: E731
    else:
        from .pivoting import plu_factor

        solve = lambda rhs: _solve_with(factor, rhs)                  # noqa: E731
        facT = plu_factor(A.T)
        solveT = lambda rhs: _solve_with(facT, rhs)                   # noqa: E731

    x = np.full(n, 1.0 / n)
    best = 0.0
    for _ in range(max_iter):
        y = solve(x)
        best = max(best, float(np.sum(np.abs(y))))
        xi = np.sign(y)
        xi[xi == 0] = 1.0
        z = solveT(xi)
        j = int(np.argmax(np.abs(z)))
        if np.abs(z[j]) <= float(z @ x) + 1e-300:
            break
        x = np.zeros(n)
        x[j] = 1.0
    return best


def condition_estimate(A, factor=None) -> float:
    """kappa_1(A) estimated in O(n^2), given a factorization.

    ``||A||_1`` is free: it is the largest absolute column sum, read straight off the matrix.
    Only ``||A^-1||_1`` needs work, and `estimate_norm_inverse` gets it from triangular solves.

    Returns an **underestimate** of the true kappa, typically within a small factor. That is the
    right trade: you want to know whether kappa is 10 or 10^12, and a factor of two either way
    changes no decision.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    inv_norm = estimate_norm_inverse(A, factor)
    if inv_norm == 0.0:
        return float("inf")
    return float(matrix_norm(A, 1) * inv_norm)


# ---------------------------------------------------------------- refinement


def exact_residual(A, b, x) -> np.ndarray:
    """r = b - A x computed in EXACT rational arithmetic, then rounded once.

    Every float is a rational number, so `fractions.Fraction` represents A, b and x exactly and
    the products and sums carry no rounding at all. Only the final conversion back to double
    rounds.

    This is the honest version of "compute the residual in higher precision". Real
    implementations use double-double arithmetic or an extended-precision accumulator rather
    than rationals, because this costs O(n^2) rational operations and is slow. It is used here
    because it is exact, so a lesson can show what the technique achieves at its limit rather
    than at some particular precision.

    **Note that `math.fsum` is not enough.** It sums exactly, but the products ``A[i,j]*x[j]``
    are each rounded to double before it ever sees them. Lesson 22 measures the difference.
    """
    from fractions import Fraction

    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    n = A.shape[0]
    if b.size != n:
        raise ValueError(f"b has length {b.size}, but A has {n} rows")
    if x.size != A.shape[1]:
        raise ValueError(f"x has length {x.size}, but A has {A.shape[1]} columns")

    xf = [Fraction(float(v)) for v in x]
    out = np.empty(n)
    for i in range(n):
        acc = Fraction(float(b[i]))
        for j in range(A.shape[1]):
            if A[i, j] != 0.0:
                acc -= Fraction(float(A[i, j])) * xf[j]
        out[i] = float(acc)
    return out


def exact_solution(A, b) -> np.ndarray:
    """The exact solution of the STORED system A x = b, computed over the rationals.

    Needed to measure refinement honestly, and the reason is subtle enough to be worth stating.

    A lesson usually builds a test by choosing ``x_true`` and setting ``b = A @ x_true``. That
    product is computed in floating point, so the stored ``b`` is not exactly ``A x_true``, and
    the exact answer to the system you actually stored differs from ``x_true`` by about
    ``kappa * u``.

    Measuring a refined solution against ``x_true`` therefore hits a floor that has nothing to
    do with the solver: it is the rounding in forming ``b``. Measured in lesson 22 at
    ``1.1e-6`` for ``kappa = 1e12``, which is exactly where naive refinement appeared to stall.

    Against this reference, refinement converges to the exact answer.
    """
    from fractions import Fraction

    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("exact_solution needs a square system")
    if b.size != n:
        raise ValueError(f"b has length {b.size}, but A is {n} by {n}")

    rows = [[Fraction(float(A[i, j])) for j in range(n)] + [Fraction(float(b[i]))]
            for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if rows[pivot][col] == 0:
            raise np.linalg.LinAlgError("matrix is exactly singular")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r != col and rows[r][col] != 0:
                factor = rows[r][col] / rows[col][col]
                rows[r] = [rows[r][j] - factor * rows[col][j] for j in range(n + 1)]
    return np.array([float(rows[i][n] / rows[i][i]) for i in range(n)])


def iterative_refinement(A, b, factor=None, max_steps: int = 4,
                         residual: str = "exact", tol: float = 0.0) -> dict:
    """Improve a computed solution by correcting it with a residual.

    The loop is three lines, and it is Newton's method on a linear equation where the
    "derivative" is A itself and so never needs recomputing:

        r <- b - A x            (the residual)
        solve A d = r           (the correction, reusing the EXISTING factorization)
        x <- x + d

    Each step costs O(n^2) against the O(n^3) already spent factorizing, so several steps are
    nearly free.

    ``residual`` selects how ``r`` is formed, and it is the whole mechanism:

    - ``"exact"``    rational arithmetic, no rounding until the final conversion,
    - ``"fsum"``     exactly rounded summation, but the products are still rounded first,
    - ``"working"``  ordinary double precision.

    Near the solution ``A x`` and ``b`` nearly cancel, so a working-precision residual is
    mostly roundoff and carries almost no information. Lesson 22 measures all three: with an
    exact residual the error falls to zero in three steps at ``kappa = 1e12``; with a working
    precision residual it does not improve at all.

    Returns the refined solution and the history, so a lesson can watch it converge.
    """
    import math

    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[0]

    if residual not in ("exact", "fsum", "working"):
        raise ValueError("residual must be 'exact', 'fsum' or 'working'")

    if factor is None:
        from .pivoting import plu_factor

        factor = plu_factor(A)

    x = _solve_with(factor, b)
    corrections, residual_norms, iterates = [], [], [x.copy()]

    for _ in range(max_steps):
        if residual == "exact":
            r = exact_residual(A, b, x)
        elif residual == "fsum":
            r = np.array([math.fsum([b[i]] + [-A[i, j] * x[j] for j in range(n)])
                          for i in range(n)])
        else:
            r = b - A @ x

        residual_norms.append(float(np.linalg.norm(r)))
        d = _solve_with(factor, r)
        step = float(np.linalg.norm(d))
        corrections.append(step)
        x = x + d
        iterates.append(x.copy())
        if step <= tol * max(float(np.linalg.norm(x)), 1e-300):
            break

    return {"x": x,
            "corrections": np.array(corrections),
            "residual_norms": np.array(residual_norms),
            "iterates": np.array(iterates),
            "n_steps": len(corrections)}


def compare_residual_precisions(A, b, max_steps: int = 4) -> dict:
    """Run refinement at all three residual precisions and report the forward error each step.

    Measured against `exact_solution`, which is the only reference that makes the comparison
    meaningful. This is the experiment lesson 22 is built around.
    """
    from .pivoting import plu_factor

    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    x_star = exact_solution(A, b)
    scale = float(np.linalg.norm(x_star)) or 1.0
    factor = plu_factor(A)

    out = {"x_star": x_star}
    for mode in ("working", "fsum", "exact"):
        res = iterative_refinement(A, b, factor, max_steps=max_steps, residual=mode)
        out[mode] = np.array([float(np.linalg.norm(v - x_star)) / scale
                              for v in res["iterates"]])
    return out
