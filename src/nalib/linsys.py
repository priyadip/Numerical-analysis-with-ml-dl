"""Conditioning and error analysis for the linear system Ax = b.

Lesson 06 said forward error is bounded by the condition number times the backward error, and
left the condition number abstract. Lesson 15 gave it a formula. This module makes it a
measurement, for the one problem the whole of Part 3 is about.

The single result that matters:

    ||x_hat - x|| / ||x||   <=   kappa(A) * ||r|| / (||A|| ||x_hat||)
    -------------------          -------   ----------------------
      forward error              problem      backward error

**A small residual does not mean a small error.** It means the computed answer exactly solves a
nearby problem. Whether that nearby problem has a nearby answer is what kappa decides, and that
is a property of A alone, fixed before any algorithm is chosen.

The functions here separate the three quantities so a lesson can report them independently, and
so a user can never look at a residual and stop reading.

Every routine works for any square system of any size, and for any coefficient magnitudes.
"""

from __future__ import annotations

import numpy as np

from .linalg import condition_number, matrix_norm, vector_norm


def residual(A, x_hat, b) -> np.ndarray:
    """r = b - A x_hat, the vector whose size is the backward error.

    Computed in the obvious way. For an accurate residual on an ill-conditioned system you need
    **extra precision**, because b and A x_hat nearly cancel: that is lesson 22's iterative
    refinement, and it is the whole reason refinement works.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    x_hat = np.asarray(x_hat, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return b - A @ x_hat


def relative_residual(A, x_hat, b, p=2) -> float:
    """||r|| / (||A|| ||x_hat||), the scaled backward error.

    Scaling matters. A raw residual of 1e-6 means nothing on its own, because multiplying the
    whole system by 1e6 multiplies it by 1e6 without changing the problem at all. Dividing by
    ||A|| ||x_hat|| removes that freedom, and the result is the size of the smallest relative
    perturbation of A for which x_hat is the exact answer.

    A backward stable solver keeps this at O(u) **whatever the conditioning**, which is exactly
    what lesson 17 measured.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    x_hat = np.asarray(x_hat, dtype=float).ravel()
    denom = matrix_norm(A, p) * vector_norm(x_hat, p)
    r = vector_norm(residual(A, x_hat, b), p)
    if denom == 0.0:
        return float("inf") if r > 0 else 0.0
    return float(r / denom)


def forward_error(x_hat, x_exact, p=2) -> float:
    """||x_hat - x|| / ||x||, which you can only compute when the answer is already known.

    Available in a lesson, where the exact answer is constructed on purpose, and never
    available in practice. That gap is why the bound below exists.
    """
    x_hat = np.asarray(x_hat, dtype=float).ravel()
    x_exact = np.asarray(x_exact, dtype=float).ravel()
    denom = vector_norm(x_exact, p)
    num = vector_norm(x_hat - x_exact, p)
    if denom == 0.0:
        return float("inf") if num > 0 else 0.0
    return float(num / denom)


def error_bound(A, x_hat, b, p=2) -> float:
    """kappa(A) times the relative residual: the guaranteed bound on the forward error.

    This is the number to report alongside any solution, because it is computable without
    knowing the answer. It is an upper bound and usually a pessimistic one, since it assumes
    the residual points along the worst direction.
    """
    return float(condition_number(A, p) * relative_residual(A, x_hat, b, p))


def error_magnification(A, x_hat, b, x_exact, p=2) -> float:
    """The amplification actually observed on this instance: forward error / backward error.

    Sauer calls this the error magnification factor. It is a **lower bound** on kappa(A),
    because this particular right-hand side may not excite the worst direction. Taking the
    maximum over many right-hand sides approaches kappa, which is a useful way to see that the
    condition number is a worst case rather than a typical one.
    """
    back = relative_residual(A, x_hat, b, p)
    fwd = forward_error(x_hat, x_exact, p)
    if back == 0.0:
        return float("inf") if fwd > 0 else 0.0
    return float(fwd / back)


def diagnose_system(A, b, x_hat, x_exact=None, p=2) -> dict:
    """Full report on a computed solution, following lesson 06 section 9.

    Reports the residual, the scaled backward error, kappa, their product (the forward error
    bound), and a verdict. When the exact answer is supplied it also reports the true forward
    error and the observed magnification, so a lesson can show the bound holding.

    The verdict separates the two failure modes that look identical if you only read the
    residual:

    - **large backward error** means the ALGORITHM misbehaved, and lesson 18's growth factor is
      the usual cause,
    - **small backward error with a large bound** means the PROBLEM is ill conditioned, and no
      algorithm would have done better.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    u = float(np.finfo(float).eps) / 2
    n = A.shape[0]

    back = relative_residual(A, x_hat, b, p)
    kappa = condition_number(A, p)
    bound = kappa * back

    algorithm_ok = back <= 100 * n * u
    if not algorithm_ok:
        verdict = ("backward error far above n*u: the ALGORITHM lost accuracy. "
                   "suspect pivot growth (lesson 18)")
    elif bound <= 1e-10:
        verdict = "solution is accurate; both the problem and the algorithm behaved"
    elif np.isinf(kappa):
        verdict = "matrix is singular to working precision; there is no unique solution"
    else:
        verdict = (f"algorithm was fine (backward error {back:.1e}) but kappa is "
                   f"{kappa:.1e}, so the answer may be wrong by {bound:.1e}. "
                   "the PROBLEM is ill conditioned")

    out = {
        "residual_norm": float(vector_norm(residual(A, x_hat, b), p)),
        "backward_error": float(back),
        "condition_number": float(kappa),
        "forward_error_bound": float(bound),
        "backward_stable": bool(algorithm_ok),
        "verdict": verdict,
    }
    if x_exact is not None:
        out["forward_error"] = forward_error(x_hat, x_exact, p)
        out["magnification"] = error_magnification(A, x_hat, b, x_exact, p)
        out["bound_holds"] = bool(out["forward_error"] <= bound * (1 + 1e-8) or bound == 0.0)
    return out


# ---------------------------------------------------------------- test matrices


def hilbert(n: int) -> np.ndarray:
    """The n by n Hilbert matrix, H[i,j] = 1/(i+j+1).

    The standard badly conditioned example. Its condition number grows roughly like
    ``e**(3.5 n)``, so it passes 1/u at about n = 13 and by n = 20 the matrix is numerically
    singular in double precision.

    It is not artificial. It is the Gram matrix of the monomials on [0,1], so it is exactly the
    matrix you get from fitting a polynomial by least squares in the monomial basis, which is
    why lesson 32 tells you not to.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    idx = np.arange(n)
    return 1.0 / (idx[:, None] + idx[None, :] + 1.0)


def vandermonde(nodes, degree: int | None = None) -> np.ndarray:
    """The Vandermonde matrix with V[i,j] = nodes[i]**j.

    Square when ``degree`` is omitted. Also badly conditioned for equally spaced nodes, for the
    same reason as the Hilbert matrix: the monomials are nearly parallel as functions, so the
    columns are nearly dependent. Lesson 44 measures this properly.
    """
    x = np.asarray(nodes, dtype=float).ravel()
    d = x.size - 1 if degree is None else int(degree)
    return np.vander(x, d + 1, increasing=True)


def with_condition_number(n: int, kappa: float, rng=None) -> np.ndarray:
    """A random n by n matrix with an exactly prescribed 2-norm condition number.

    Built as ``U diag(s) V^T`` with logarithmically spaced singular values from 1 down to
    ``1/kappa`` and random orthogonal U, V. Since orthogonal factors do not change singular
    values, the condition number is exactly what was asked for.

    This is how every conditioning experiment in the course generates its test matrices: it
    separates the effect of kappa from every other property of the matrix, which a collection
    of named examples cannot do.
    """
    from .orthogonality import random_orthogonal

    if n < 1:
        raise ValueError("n must be at least 1")
    if kappa < 1.0:
        raise ValueError("a condition number is never below 1")
    rng = np.random.default_rng() if rng is None else rng
    if n == 1:
        # a 1 by 1 matrix has one singular value, so its condition number is 1 whatever the
        # entry is. Any nonzero value is the honest answer; kappa cannot be prescribed here.
        return np.array([[1.0]])
    s = np.logspace(0.0, -np.log10(kappa), n)
    return random_orthogonal(n, rng) @ np.diag(s) @ random_orthogonal(n, rng).T


def beam_matrix(n: int) -> np.ndarray:
    """The Euler-Bernoulli beam stiffness matrix, from Sauer's Reality Check 2.

    A clamped-free beam discretised by finite differences gives a banded matrix from the fourth
    derivative operator. The interior rows are the fourth-difference stencil
    ``[1, -4, 6, -4, 1]``, and the boundary rows are modified for the clamped end and the free
    end.

    Its condition number grows like ``n^4``, because the fourth derivative is being
    approximated, and each derivative costs a factor of ``n``. That makes it a physically real
    example of an ill-conditioned system rather than a contrived one: the ill conditioning comes
    from the mathematics of the beam, not from anybody's choice of matrix.
    """
    if n < 4:
        raise ValueError("the beam stencil needs at least 4 unknowns")
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 6.0
        if i >= 1:
            A[i, i - 1] = -4.0
        if i >= 2:
            A[i, i - 2] = 1.0
        if i + 1 < n:
            A[i, i + 1] = -4.0
        if i + 2 < n:
            A[i, i + 2] = 1.0
    # clamped left end and free right end
    A[0, 0] = 16.0
    A[0, 1] = -9.0
    A[0, 2] = 8.0 / 3.0
    A[0, 3] = -1.0 / 4.0
    A[n - 2, n - 4:] = np.array([16.0, -60.0, 72.0, -28.0]) / 17.0
    A[n - 1, n - 4:] = np.array([-12.0, 96.0, -156.0, 72.0]) / 17.0
    return A
