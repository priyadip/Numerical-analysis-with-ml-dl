"""Gaussian elimination, LU factorization, and triangular solves.

The central algorithm of numerical linear algebra, and the one that everything in Part 3 is
either building towards or repairing.

The idea is old and simple: use row operations to reduce ``Ax = b`` to a triangular system,
which can be solved by substitution. The content is in what the row operations do to the
matrix, and in what floating point does to the row operations.

**Elimination is factorization.** The multipliers you use to zero out the lower triangle,
collected in a matrix, *are* the factor ``L``. So the reduced matrix ``U`` and the multipliers
``L`` satisfy ``A = LU``, and elimination has not merely solved one system, it has produced a
decomposition that solves any number of them at ``O(n^2)`` each.

**Triangular solves are better behaved than they look.** Back substitution is backward stable
in a **componentwise** sense, which is much stronger than the usual normwise statement. The
consequence, measured in lesson 17, is that triangular systems are routinely solved far more
accurately than their condition number predicts.

Naive elimination has no pivoting and breaks on perfectly good matrices. That is deliberate:
lesson 17 shows the failure and lesson 18 fixes it in ``nalib.pivoting``.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- triangular solves


def forward_substitution(L, b, unit_diagonal: bool = False) -> np.ndarray:
    """Solve L y = b for lower triangular L, working top down.

    Row i reads ``sum_j L[i,j] y[j] = b[i]``. Every ``y[j]`` with ``j < i`` is already known,
    so one subtraction and one division give ``y[i]``.

    ``unit_diagonal=True`` skips the division, which is what you want for the ``L`` coming out
    of an LU factorization, whose diagonal is exactly 1 and is not even stored in the packed
    form.

    Cost: n^2 flops, so n^2/2 multiply-adds. Compare O(n^3) to build the factorization: **the
    factorization is the expensive part and the solve is nearly free**, which is why you
    factor once and solve many times.
    """
    L = np.atleast_2d(np.asarray(L, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    if L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square, got {L.shape}")
    if b.size != L.shape[0]:
        raise ValueError(f"b has length {b.size}, but L is {L.shape[0]} by {L.shape[1]}")
    n = b.size
    y = np.zeros(n)
    for i in range(n):
        s = b[i] - L[i, :i] @ y[:i]
        if unit_diagonal:
            y[i] = s
        else:
            if L[i, i] == 0.0:
                raise np.linalg.LinAlgError(f"zero on the diagonal at row {i}")
            y[i] = s / L[i, i]
    return y


def back_substitution(U, y) -> np.ndarray:
    """Solve U x = y for upper triangular U, working bottom up.

    The mirror image of forward substitution. The last row gives ``x[n-1]`` immediately, and
    each row above uses the entries already found.

    **This routine is backward stable in the componentwise sense** (Theorem 17.4, from
    Trefethen and Bau lecture 17): the computed ``x_hat`` satisfies exactly

        (U + dU) x_hat = y     with    |dU_ij| <= n u |U_ij|

    entry by entry, not merely ``||dU|| <= n u ||U||``. The perturbation is small **relative to
    each individual entry**, so entries that are tiny are perturbed only tinily. A normwise
    bound would allow a tiny entry to be perturbed by u times the largest entry in the matrix,
    which is a completely different and much weaker statement.

    That is why triangular systems are so often solved to far better accuracy than kappa(U)
    suggests. Lesson 17 measures it on a matrix with kappa around 1e12 and finds the forward
    error at 1e-15 rather than the 1e-4 the condition number allows.
    """
    U = np.atleast_2d(np.asarray(U, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if U.shape[0] != U.shape[1]:
        raise ValueError(f"U must be square, got {U.shape}")
    if y.size != U.shape[0]:
        raise ValueError(f"y has length {y.size}, but U is {U.shape[0]} by {U.shape[1]}")
    n = y.size
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0.0:
            raise np.linalg.LinAlgError(f"zero on the diagonal at row {i}")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


# ---------------------------------------------------------------- elimination


def naive_gaussian_elimination(A, b) -> dict:
    """Forward elimination without pivoting, then back substitution.

    The textbook algorithm, and **not** one to use. It divides by whatever happens to sit on
    the diagonal, so it fails outright when that entry is zero, and it loses accuracy when the
    entry is merely small. Lesson 18 shows the failure and fixes it.

    Returns a dict with the reduced ``U``, the modified right-hand side, the solution, and the
    list of multipliers used, so a lesson can inspect the process rather than only the answer.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).ravel()
    n = A.shape[0]
    multipliers = []

    for k in range(n - 1):
        if A[k, k] == 0.0:
            raise np.linalg.LinAlgError(
                f"zero pivot at step {k}. this is exactly what pivoting exists to avoid"
            )
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            multipliers.append((i, k, m))
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    return {
        "U": A,
        "b_reduced": b,
        "x": back_substitution(A, b),
        "multipliers": multipliers,
    }


def lu_factor(A) -> tuple[np.ndarray, np.ndarray]:
    """A = LU by Doolittle's method, no pivoting. L has a unit diagonal.

    The multipliers used in elimination are exactly the entries of L. Proving that is the whole
    content of "elimination is factorization" (Theorem 17.2), and once you have it, elimination
    stops being a way to solve one system and becomes a way to precompute a solver.

    Raises on a zero pivot, which is a genuine limitation and not a bug. Use
    ``nalib.pivoting.plu_factor`` for anything real.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for k in range(n - 1):
        if U[k, k] == 0.0:
            raise np.linalg.LinAlgError(f"zero pivot at step {k}; LU without pivoting fails")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
        U[k + 1:, k] = 0.0                 # exact zeros below the pivot
    return L, U


def lu_solve(L, U, b) -> np.ndarray:
    """Solve LUx = b: forward substitute for y, then back substitute for x.

    Two O(n^2) solves against the O(n^3) factorization, which is the point of factoring.
    """
    y = forward_substitution(L, b, unit_diagonal=True)
    return back_substitution(U, y)


def crout_factor(A) -> tuple[np.ndarray, np.ndarray]:
    """A = LU with a unit diagonal on U instead of on L.

    Doolittle and Crout are the same factorization with the diagonal scaling assigned to the
    other factor. Given Doolittle's ``A = L U``, put ``D = diag(U)``; then

        A = L D (D^-1 U)

    and ``L D`` is Crout's L while ``D^-1 U`` is Crout's U, with ones on its diagonal.

    Lesson 17 verifies numerically that the two agree, because the point worth making is that
    they are not two algorithms but one, presented twice.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.eye(n)
    for j in range(n):
        for i in range(j, n):
            L[i, j] = A[i, j] - L[i, :j] @ U[:j, j]
        if L[j, j] == 0.0:
            raise np.linalg.LinAlgError(f"zero pivot at step {j}; Crout without pivoting fails")
        for i in range(j + 1, n):
            U[j, i] = (A[j, i] - L[j, :j] @ U[:j, i]) / L[j, j]
    return L, U


def gauss_jordan(A, b=None) -> dict:
    """Reduce all the way to the identity rather than stopping at triangular.

    Gauss-Jordan eliminates **above** the pivot as well as below, so no back substitution is
    needed. That sounds like a saving and is not: it costs n^3 flops against Gaussian
    elimination's 2n^3/3, so it is **50 percent more expensive** for the same answer.

    It is still the right tool for one job, computing an explicit inverse, by carrying the
    identity along as the right-hand side. Lesson 17 measures both the cost and the fact that
    you should almost never want the inverse anyway.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    aug = np.hstack([A, np.eye(n) if b is None else np.array(b, dtype=float).reshape(n, -1)])

    for k in range(n):
        if aug[k, k] == 0.0:
            raise np.linalg.LinAlgError(f"zero pivot at step {k}")
        aug[k] = aug[k] / aug[k, k]
        for i in range(n):
            if i != k:
                aug[i] -= aug[i, k] * aug[k]

    result = aug[:, n:]
    return {"reduced": aug, "result": result if b is not None else result,
            "inverse": result if b is None else None}


def determinant_from_lu(U) -> float:
    """det(A) as the product of the diagonal of U, when A = LU with unit-diagonal L.

    Because det(A) = det(L)det(U) = 1 * prod(U_ii). This is O(n^3) via the factorization, which
    is the only sane way. Cofactor expansion is O(n!), and lesson 17 measures just how
    unusable that is.
    """
    return float(np.prod(np.diag(np.asarray(U, dtype=float))))


def cramer_solve(A, b) -> np.ndarray:
    """Solve Ax = b by Cramer's rule. Included to demonstrate that it must never be used.

    Each component is a ratio of determinants, so it needs n+1 determinants of n by n
    matrices. Even computing each determinant at the optimal O(n^3), the total is O(n^4), which
    is already worse than elimination by a factor of n. Computed by cofactor expansion as the
    rule is usually taught, it is O((n+1)!).

    It is also numerically poor, because the determinants can overflow or cancel badly even
    when the system itself is perfectly well conditioned.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[0]
    detA = np.linalg.det(A)
    if detA == 0.0:
        raise np.linalg.LinAlgError("singular matrix")
    x = np.empty(n)
    for j in range(n):
        Aj = A.copy()
        Aj[:, j] = b
        x[j] = np.linalg.det(Aj) / detA
    return x


def determinant_by_cofactor(A) -> float:
    """det(A) by recursive cofactor expansion. O(n!) and included only to be measured.

    At n = 12 this is roughly 10^9 operations. At n = 20 it is 10^18, which is decades. Lesson
    17 times it against the LU route to make the gap concrete rather than asserted.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 1:
        return float(A[0, 0])
    if n == 2:
        return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    total = 0.0
    for j in range(n):
        minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
        total += ((-1.0) ** j) * A[0, j] * determinant_by_cofactor(minor)
    return float(total)


# ---------------------------------------------------------------- cost


def flops_lu(n: int) -> int:
    """Exact flop count for LU factorization without pivoting.

    At step k there are (n-k-1) rows below the pivot, each needing one division for the
    multiplier plus (n-k) multiply-add pairs. Summing gives

        2n^3/3 - n^2/2 - n/6,

    so the leading term is **2n^3/3**. Lesson 17 verifies this against an operation counter
    rather than trusting the algebra.
    """
    return sum((n - k - 1) + 2 * (n - k - 1) * (n - k) for k in range(n - 1))


def flops_triangular_solve(n: int) -> int:
    """Exact flop count for one triangular solve: n^2 flops.

    Row i needs i multiply-add pairs and one division, so the total is
    ``sum_i (2i + 1) = n^2``. Two of these follow a factorization, so **solving costs 2n^2
    against the factorization's 2n^3/3**: for n = 1000 that is a factor of 333.
    """
    return sum(2 * i + 1 for i in range(n))


def flops_gauss_jordan(n: int) -> int:
    """Exact flop count for Gauss-Jordan reduction to the identity.

    At each of the n pivots, every one of the other n-1 rows is updated across the remaining
    columns. The leading term is n^3, against 2n^3/3 for elimination plus back substitution,
    so Gauss-Jordan costs **50 percent more**.
    """
    total = 0
    for k in range(n):
        total += n - k                                   # scaling the pivot row
        total += (n - 1) * 2 * (n - k)                   # updating every other row
    return total
