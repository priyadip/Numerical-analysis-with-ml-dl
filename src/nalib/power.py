"""Power iteration and its family: the simplest eigenvalue algorithms, and the fastest one.

What this module is for
-----------------------
Lesson 35 proved no finite algorithm exists, so everything here iterates. The whole family comes
from one observation: **repeatedly multiplying by ``A`` amplifies the dominant eigendirection**,
because writing a starting vector in the eigenvector basis and applying ``A`` ``k`` times scales
each component by ``lambda_i^k``, and the largest one wins.

Four methods, each fixing the previous one's limitation:

- `power_iteration` finds the **largest** eigenvalue, at rate ``|lambda_2 / lambda_1|``. Useless
  when that ratio is near 1, and it can only find one eigenvalue.
- `inverse_iteration` applies power iteration to ``(A - sigma I)^{-1}``, whose largest
  eigenvalue corresponds to the eigenvalue of ``A`` **nearest sigma**. So a shift chooses which
  eigenvalue you get.
- `rayleigh_quotient_iteration` updates the shift every step using the best current estimate.
  The rate becomes **cubic** for a symmetric matrix, which is faster than Newton's method.
- `deflate` removes a converged eigenpair so the next one can be found.

**The recurring surprise is that inverse iteration's ill conditioning is harmless**, and
`inverse_iteration` documents why. As ``sigma`` approaches an eigenvalue, ``A - sigma I``
becomes singular and the solve loses all its digits; the error lands almost entirely along the
eigenvector being sought, so the **direction** is right even when the magnitude is nonsense, and
the direction is all that is wanted. Lesson 36 measures it.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EigResult:
    """One eigenpair and enough history to measure a rate."""

    value: complex
    vector: np.ndarray
    iterations: int
    converged: bool
    message: str
    residuals: list = field(default_factory=list)     # ||A v - lambda v|| each step
    estimates: list = field(default_factory=list)     # the eigenvalue estimate each step
    shifts: list = field(default_factory=list)        # for the shifted methods


def rayleigh_quotient(A, x) -> complex:
    """``x^H A x / x^H x``, the best scalar approximation to ``A`` acting on ``x``.

    It is the least squares solution of ``x mu = A x`` in the scalar unknown ``mu``, which is
    lesson 29's normal equation with a one-column design matrix. So it is not an arbitrary
    choice: it is the value minimising ``||A x - mu x||``.

    **Its accuracy is what makes the whole family work.** For a symmetric matrix, if ``x`` is
    within ``eps`` of an eigenvector then the Rayleigh quotient is within ``eps^2`` of the
    eigenvalue: the error is **squared**, because the gradient of the quotient vanishes at an
    eigenvector. For a general matrix the error stays ``O(eps)``, and lesson 36 measures both.
    """
    A = np.atleast_2d(np.asarray(A))
    x = np.asarray(x).ravel()
    if x.size != A.shape[1]:
        raise ValueError(f"x has {x.size} entries, A has {A.shape[1]} columns")
    denom = complex(np.vdot(x, x))
    if denom == 0:
        raise ValueError("the Rayleigh quotient of the zero vector is undefined")
    return complex(np.vdot(x, A @ x)) / denom


def _residual(A, lam, v) -> float:
    return float(np.linalg.norm(A @ v - lam * v))


def _start(A, x0, rng):
    A = np.atleast_2d(np.asarray(A))
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A is {A.shape}, eigenvalues need a square matrix")
    n = A.shape[0]
    if x0 is None:
        gen = np.random.default_rng() if rng is None else rng
        x = gen.standard_normal(n)
    else:
        x = np.asarray(x0, dtype=float).ravel().astype(complex) \
            if np.iscomplexobj(x0) else np.asarray(x0, dtype=float).ravel()
        if x.size != n:
            raise ValueError(f"x0 has {x.size} entries, A is {n} by {n}")
    nrm = float(np.linalg.norm(x))
    if nrm == 0.0:
        raise ValueError("the starting vector must not be zero")
    return A, x / nrm


def power_iteration(A, x0=None, tol: float = 1e-12, max_iter: int = 1000,
                    rng=None) -> EigResult:
    """Repeatedly apply ``A`` and normalise. Converges to the eigenvector of largest modulus.

    **The rate is ``|lambda_2 / lambda_1|`` per step**, linear. Writing
    ``x = sum c_i v_i`` and applying ``A`` ``k`` times gives
    ``A^k x = lambda_1^k (c_1 v_1 + sum_{i>1} c_i (lambda_i/lambda_1)^k v_i)``, so every other
    component decays by that ratio each step.

    **Three ways it fails, all visible in the formula.** ``c_1 = 0``, a starting vector with no
    component along the dominant eigenvector (measure zero, and roundoff usually rescues it).
    ``|lambda_1| = |lambda_2|``, where nothing decays, which happens for every real matrix with
    a complex conjugate pair of dominant eigenvalues. And a ratio near 1, where it converges
    but too slowly to use.

    The eigenvalue estimate is the Rayleigh quotient rather than the ratio of successive
    entries, because the quotient is accurate to ``eps^2`` on a symmetric matrix.
    """
    A, x = _start(A, x0, rng)
    out = EigResult(value=0.0, vector=x, iterations=0, converged=False, message="")
    for k in range(int(max_iter)):
        y = A @ x
        nrm = float(np.linalg.norm(y))
        if nrm == 0.0:
            out.message = f"A x became the zero vector at step {k}; x is in the null space"
            break
        x = y / nrm
        lam = rayleigh_quotient(A, x)
        res = _residual(A, lam, x)
        out.estimates.append(lam)
        out.residuals.append(res)
        out.iterations = k + 1
        if res <= tol * max(abs(lam), 1.0):
            out.converged = True
            out.message = f"residual below tolerance after {k + 1} iterations"
            break
    else:
        out.message = f"hit the iteration limit of {max_iter}"
    out.vector = x
    out.value = rayleigh_quotient(A, x)
    return out


def power_iteration_rate(A) -> float:
    """``|lambda_2| / |lambda_1|``, the asymptotic factor by which the error shrinks each step.

    Computed from the true spectrum, so it exists to **check** a measured rate rather than to be
    used inside an algorithm. A value near 1 means power iteration is useless on this matrix;
    exactly 1 means it does not converge at all.
    """
    vals = np.sort(np.abs(np.linalg.eigvals(np.atleast_2d(np.asarray(A)))))[::-1]
    if vals.size < 2 or vals[0] == 0:
        return 0.0
    return float(vals[1] / vals[0])


def inverse_iteration(A, sigma: complex, x0=None, tol: float = 1e-12,
                      max_iter: int = 200, rng=None) -> EigResult:
    """Power iteration on ``(A - sigma I)^{-1}``: converges to the eigenvalue **nearest sigma**.

    The eigenvalues of ``(A - sigma I)^{-1}`` are ``1/(lambda_i - sigma)``, so the largest is
    the one whose ``lambda_i`` is closest to ``sigma``. The rate is

        |lambda_near - sigma| / |lambda_second_near - sigma|,

    so a good shift makes it fast **and** selects which eigenvalue is found. That is the whole
    idea, and it is why shifts appear in every later method.

    **The solve is deliberately ill conditioned, and that is not a problem.** As ``sigma``
    approaches an eigenvalue, ``A - sigma I`` approaches singular and the computed solution has
    a huge error. But the error lies almost entirely **along the eigenvector being sought**,
    since that is the direction the near-singularity amplifies. The vector is then normalised,
    so its length is discarded and only its direction is kept. Wilkinson's observation, and it
    is why the textbook worry about "never solve with a nearly singular matrix" does not apply
    here. Lesson 36 measures the residual of that solve and the accuracy of the answer side by
    side.

    One LU factorization is reused for every step, so each iteration costs ``O(n^2)``.
    """
    from scipy.linalg import lu_factor, lu_solve

    A, x = _start(A, x0, rng)
    n = A.shape[0]
    shifted = A - complex(sigma) * np.eye(n) if np.iscomplexobj(A) or np.iscomplex(sigma) \
        else A - float(np.real(sigma)) * np.eye(n)
    out = EigResult(value=0.0, vector=x, iterations=0, converged=False, message="")
    try:
        lu_piv = lu_factor(shifted)
    except (ValueError, np.linalg.LinAlgError) as exc:
        out.message = f"the shifted matrix could not be factored: {exc}"
        return out

    for k in range(int(max_iter)):
        with np.errstate(over="ignore", invalid="ignore"):
            y = lu_solve(lu_piv, x)
        nrm = float(np.linalg.norm(y))
        if not np.isfinite(nrm) or nrm == 0.0:
            out.message = (f"the solve returned a non-finite vector at step {k}; "
                           f"sigma is an exact eigenvalue to working precision")
            break
        x = y / nrm
        lam = rayleigh_quotient(A, x)
        res = _residual(A, lam, x)
        out.estimates.append(lam)
        out.residuals.append(res)
        out.iterations = k + 1
        if res <= tol * max(abs(lam), 1.0):
            out.converged = True
            out.message = f"residual below tolerance after {k + 1} iterations"
            break
    else:
        out.message = f"hit the iteration limit of {max_iter}"
    out.vector = x
    out.value = rayleigh_quotient(A, x)
    return out


def inverse_iteration_rate(A, sigma: complex) -> float:
    """The convergence factor of `inverse_iteration` at this shift, from the true spectrum."""
    vals = np.linalg.eigvals(np.atleast_2d(np.asarray(A)))
    gaps = np.sort(np.abs(vals - complex(sigma)))
    if gaps.size < 2 or gaps[1] == 0:
        return 0.0
    return float(gaps[0] / gaps[1])


def rayleigh_quotient_iteration(A, x0=None, tol: float = 1e-12, max_iter: int = 100,
                                rng=None) -> EigResult:
    """Inverse iteration with the shift updated to the Rayleigh quotient every step.

    **Cubic convergence for a symmetric matrix**, quadratic in general. The reason is the
    squaring in `rayleigh_quotient`: an eigenvector accurate to ``eps`` gives a shift accurate
    to ``eps^2``, one inverse iteration step with that shift gives a vector accurate to
    ``eps * eps^2 = eps^3``, and the loop closes.

    **The price is a fresh factorization every step**, ``O(n^3)`` instead of ``O(n^2)``, since
    the shift changes. On a Hessenberg or tridiagonal matrix that drops to ``O(n)`` and the
    method becomes the inner loop of the QR algorithm (lesson 37).

    **And it is not globally convergent**: which eigenvalue it finds depends on where it starts,
    unpredictably. That is the trade for the speed, and it is why production code uses it only
    after a cheaper method has got close.
    """
    from scipy.linalg import lu_factor, lu_solve

    A, x = _start(A, x0, rng)
    n = A.shape[0]
    out = EigResult(value=0.0, vector=x, iterations=0, converged=False, message="")
    lam = rayleigh_quotient(A, x)

    for k in range(int(max_iter)):
        out.shifts.append(lam)
        shifted = A - lam * np.eye(n) if np.iscomplexobj(A) or np.iscomplex(lam) \
            else A - float(np.real(lam)) * np.eye(n)
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                y = lu_solve(lu_factor(shifted), x)
            nrm = float(np.linalg.norm(y))
        except (ValueError, np.linalg.LinAlgError):
            nrm = 0.0
            y = x
        if not np.isfinite(nrm) or nrm == 0.0:
            # The shift hit an eigenvalue exactly to working precision. That is convergence,
            # not failure: the current x already satisfies A x = lam x to roundoff.
            out.converged = _residual(A, lam, x) <= 1e-8 * max(abs(lam), 1.0)
            out.message = (f"the shifted solve became singular at step {k}, which means the "
                           f"shift is an eigenvalue to working precision")
            break
        x = y / nrm
        lam = rayleigh_quotient(A, x)
        res = _residual(A, lam, x)
        out.estimates.append(lam)
        out.residuals.append(res)
        out.iterations = k + 1
        if res <= tol * max(abs(lam), 1.0):
            out.converged = True
            out.message = f"residual below tolerance after {k + 1} iterations"
            break
    else:
        out.message = f"hit the iteration limit of {max_iter}"
    out.vector = x
    out.value = rayleigh_quotient(A, x)
    return out


def deflate(A, value: complex, vector) -> np.ndarray:
    """Remove one eigenpair by a Hotelling deflation, so the next one becomes dominant.

    For a **symmetric** ``A`` with unit eigenvector ``v``, ``A - lambda v v^H`` has the same
    eigenvalues except that ``lambda`` is replaced by 0. That is exact, and the deflated matrix
    is still symmetric.

    **For a non-symmetric matrix this is not safe**, and the reason is lesson 35's: the left and
    right eigenvectors differ, so ``A - lambda v v^H`` does not have the intended spectrum
    unless the left eigenvector is used on the other side. The safe general route is a Schur
    deflation, which is what the QR algorithm does in lesson 37, and this routine refuses rather
    than returning something wrong.
    """
    A = np.atleast_2d(np.asarray(A))
    v = np.asarray(vector).ravel()
    if v.size != A.shape[0]:
        raise ValueError(f"the vector has {v.size} entries, A is {A.shape[0]} by {A.shape[1]}")
    asym = float(np.linalg.norm(A - np.conj(A).T))
    if asym > 1e-10 * max(float(np.linalg.norm(A)), 1.0):
        raise ValueError(
            "Hotelling deflation needs a symmetric (or Hermitian) matrix; for a general matrix "
            "the left and right eigenvectors differ and this subtraction gives the wrong "
            "spectrum. Use a Schur deflation instead (lesson 37).")
    v = v / float(np.linalg.norm(v))
    return A - complex(value) * np.outer(v, np.conj(v))


def find_k_eigenpairs(A, k: int, tol: float = 1e-12, max_iter: int = 2000,
                      rng=None) -> dict:
    """Find the ``k`` largest eigenpairs of a symmetric matrix by power iteration and deflation.

    Included to make the **failure mode** measurable rather than to be used: errors accumulate,
    because each deflation subtracts an eigenvector that is itself only accurate to the
    tolerance. Lesson 36 measures how the error grows with ``k``, and lesson 38 gives the method
    that does not have the problem.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    k = int(k)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    gen = np.random.default_rng() if rng is None else rng
    work = A.copy()
    values, vectors, steps = [], [], []
    for _ in range(k):
        out = power_iteration(work, tol=tol, max_iter=max_iter, rng=gen)
        values.append(out.value)
        vectors.append(out.vector)
        steps.append(out.iterations)
        work = deflate(work, out.value, out.vector)
    return {"values": np.array(values), "vectors": np.array(vectors).T,
            "iterations": steps}
