"""Symmetric positive definite matrices, and the Cholesky factorization.

Part 3 so far has been a sequence of things that can go wrong: zero pivots, tiny pivots, growth,
conditioning. This module covers the one class of matrix where most of that trouble simply does
not arise.

A symmetric positive definite matrix admits

    A = L L^T

with L lower triangular and a positive diagonal, and this factorization

- **needs no pivoting**, ever, and is stable without it,
- has **growth factor exactly 1**, so lesson 18's whole worry is absent,
- costs **n^3/3 flops, half of LU**, because symmetry means half the work is redundant,
- **exists if and only if A is positive definite**, so attempting it is itself the best
  numerical test for positive definiteness.

That last point is worth stating plainly: do not test positive definiteness by computing
eigenvalues at O(n^3) with a large constant, and never by checking determinants of leading
minors, which is numerically hopeless. Try Cholesky and see whether it succeeds.

Used by lessons 20 and 21, again in Part 4 for the conjugate gradient method, which requires
exactly this class of matrix, and in Part 5 for the normal equations.
"""

from __future__ import annotations

import numpy as np


def is_symmetric(A, tol: float = 1e-12) -> bool:
    """True when A equals its transpose to within a relative tolerance."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] != A.shape[1]:
        return False
    scale = np.max(np.abs(A)) if A.size else 0.0
    return bool(np.max(np.abs(A - A.T)) <= tol * max(scale, 1.0))


def cholesky(A, check_symmetry: bool = True) -> np.ndarray:
    """A = L L^T for symmetric positive definite A. Returns lower triangular L.

    Derived by writing out the product entry by entry and reading off each unknown in turn.
    Comparing entry (i, j) of ``L L^T`` with ``A[i, j]`` gives, for the diagonal,

        L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2)

    and below it

        L[i,j] = (A[i,j] - sum_{k<j} L[i,k] L[j,k]) / L[j,j].

    Every quantity on the right is already known when it is needed, so there is nothing to
    solve: the factorization is a direct computation.

    **The square root is the test.** If the quantity under it is not positive, A is not
    positive definite, and there is no real Cholesky factor. Raising there is not a failure of
    the algorithm; it is the algorithm reporting a property of the matrix.

    Raises `numpy.linalg.LinAlgError` when A is not positive definite, matching
    `numpy.linalg.cholesky`.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] != A.shape[1]:
        raise ValueError("Cholesky needs a square matrix")
    if check_symmetry and not is_symmetric(A):
        raise ValueError("Cholesky needs a symmetric matrix")

    n = A.shape[0]
    L = np.zeros((n, n))
    for j in range(n):
        pivot = A[j, j] - L[j, :j] @ L[j, :j]
        if pivot <= 0.0:
            raise np.linalg.LinAlgError(
                f"matrix is not positive definite: the pivot at index {j} is {pivot:g}"
            )
        L[j, j] = np.sqrt(pivot)
        if j + 1 < n:
            L[j + 1:, j] = (A[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    return L


def cholesky_solve(L, b) -> np.ndarray:
    """Solve A x = b given the Cholesky factor L of A.

    Two triangular solves: forward with L, then backward with L^T. Total 2n^2 flops, the same
    as an LU solve, because the saving in Cholesky is all in the factorization.
    """
    from .lu import back_substitution, forward_substitution

    L = np.atleast_2d(np.asarray(L, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    if b.size != L.shape[0]:
        raise ValueError(f"b has length {b.size}, but the factor is {L.shape[0]} square")
    y = forward_substitution(L, b)
    return back_substitution(L.T, y)


def is_positive_definite(A) -> bool:
    """Test positive definiteness by attempting Cholesky.

    This is the right test, and the alternatives are worse:

    - **eigenvalues**: O(n^3) with a much larger constant, and it computes far more than the
      yes-or-no answer you asked for,
    - **leading minor determinants** (Sylvester's criterion): mathematically correct and
      numerically hopeless, since determinants overflow, underflow, and lose all accuracy.
      Lesson 20 measures it failing on a matrix that is comfortably positive definite.

    Cholesky costs n^3/3, gives the answer, and hands you the factorization as a side effect.
    """
    try:
        cholesky(A)
        return True
    except (np.linalg.LinAlgError, ValueError):
        return False


def ldl(A, check_symmetry: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """A = L D L^T with unit lower triangular L and diagonal D. No square roots.

    The square-root-free relative of Cholesky. Two reasons to prefer it:

    - **no square roots**, which matters on hardware where they are slow, and which keeps the
      factorization exact for integer-valued input,
    - **it works for symmetric INDEFINITE matrices too**, where Cholesky cannot exist, because
      D is allowed negative entries. The signs of D then tell you the inertia of A, meaning how
      many positive, negative and zero eigenvalues it has, without computing any eigenvalue.

    A is positive definite exactly when every entry of D is positive, so this is also a
    positive definiteness test that reports *how* a matrix fails rather than only that it did.

    Without pivoting this can still break down on an indefinite matrix (a zero appears in D
    where the true factorization needs a 2 by 2 block). The Bunch-Kaufman pivoting strategy
    fixes that and is what LAPACK uses; lesson 20 exercise 5.2 develops it.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] != A.shape[1]:
        raise ValueError("LDL needs a square matrix")
    if check_symmetry and not is_symmetric(A):
        raise ValueError("LDL needs a symmetric matrix")

    n = A.shape[0]
    L = np.eye(n)
    d = np.zeros(n)
    for j in range(n):
        d[j] = A[j, j] - np.sum(d[:j] * L[j, :j] ** 2)
        if d[j] == 0.0:
            raise np.linalg.LinAlgError(
                f"zero pivot in D at index {j}; this matrix needs Bunch-Kaufman pivoting"
            )
        if j + 1 < n:
            L[j + 1:, j] = (A[j + 1:, j]
                            - (L[j + 1:, :j] * d[:j]) @ L[j, :j]) / d[j]
    return L, d


def ldl_solve(L, d, b) -> np.ndarray:
    """Solve A x = b given A = L D L^T. Forward solve, divide by D, back solve."""
    from .lu import back_substitution, forward_substitution

    y = forward_substitution(L, b, unit_diagonal=True)
    z = np.asarray(y, dtype=float) / np.asarray(d, dtype=float)
    return back_substitution(np.asarray(L, dtype=float).T, z)


def inertia(A) -> tuple[int, int, int]:
    """(positive, negative, zero) eigenvalue counts, from the signs of D in A = L D L^T.

    Sylvester's law of inertia says a congruence transformation ``A -> M A M^T`` preserves
    these counts. Since ``A = L D L^T`` is exactly such a transformation, the signs of D give
    the inertia of A directly, at n^3/3, without computing a single eigenvalue.

    Falls back to eigenvalues when the unpivoted LDL breaks down, so the answer is always
    correct even where the cheap route is unavailable.

    Rejects a nonsymmetric matrix rather than answering. Inertia is defined through Sylvester's
    law, which is a statement about symmetric matrices, and `numpy.linalg.eigvalsh` would
    silently read only one triangle and return the inertia of a **different** matrix.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if not is_symmetric(A):
        raise ValueError("inertia is defined for symmetric matrices only")
    try:
        _, d = ldl(A, check_symmetry=False)
        tol = 1e-12 * max(float(np.max(np.abs(d))), 1.0)
        return (int(np.sum(d > tol)), int(np.sum(d < -tol)), int(np.sum(np.abs(d) <= tol)))
    except (np.linalg.LinAlgError, ValueError):
        w = np.linalg.eigvalsh(A)
        tol = 1e-12 * max(float(np.max(np.abs(w))), 1.0)
        return (int(np.sum(w > tol)), int(np.sum(w < -tol)), int(np.sum(np.abs(w) <= tol)))


# ---------------------------------------------------------------- construction and cost


def random_spd(n: int, kappa: float | None = None, rng=None) -> np.ndarray:
    """A random symmetric positive definite matrix, optionally with a prescribed kappa.

    Built as ``Q diag(s) Q^T`` with positive s and orthogonal Q. That is a spectral
    decomposition, so the eigenvalues are exactly ``s``, the matrix is exactly symmetric by
    construction, and it is positive definite exactly because every s is positive.

    For a symmetric matrix the singular values are the absolute eigenvalues, so
    ``kappa_2 = max(s)/min(s)`` and prescribing it is exact.
    """
    from .orthogonality import random_orthogonal

    if n < 1:
        raise ValueError("n must be at least 1")
    rng = np.random.default_rng() if rng is None else rng
    if kappa is None:
        B = rng.standard_normal((n, n))
        return B @ B.T + n * np.eye(n)
    if kappa < 1.0:
        raise ValueError("a condition number is never below 1")
    if n == 1:
        return np.array([[1.0]])
    s = np.logspace(0.0, -np.log10(kappa), n)
    Q = random_orthogonal(n, rng)
    A = Q @ np.diag(s) @ Q.T
    return (A + A.T) / 2.0            # remove the roundoff asymmetry


def flops_cholesky(n: int) -> int:
    """Exact flop count for the Cholesky factorization.

    At step j the work is one square root, ``n-j-1`` divisions, and the inner products against
    the already-computed part. Summing gives a leading term of **n^3/3**, exactly half of LU's
    2n^3/3.

    The halving is not a trick. Symmetry means the strict upper triangle carries no information
    the lower triangle does not, so half the arithmetic in LU is recomputing what is already
    known.
    """
    total = 0
    for j in range(n):
        total += 2 * j + 1                      # the diagonal entry and its square root
        rows = n - j - 1
        total += rows * (2 * j + 1)             # each entry below the diagonal
    return total
