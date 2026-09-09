"""Pivoting: making Gaussian elimination usable.

Lesson 17 left naive elimination broken in two ways. It divides by whatever sits on the
diagonal, so it fails outright on a zero pivot and silently loses every digit on a small one.
Both failures are properties of the **algorithm**, not of the problem: the matrices involved
can be perfectly well conditioned.

The fix is to choose the pivot rather than accept it. At step k, look down the column, find the
largest entry in magnitude, and swap that row up. This is **partial pivoting**, and it gives
``PA = LU`` where P records the swaps.

Two facts about it are worth separating.

**Why it helps is easy.** Every multiplier becomes ``|m| <= 1``, because the pivot is the
largest available. So elimination can no longer amplify anything by a large factor at the
moment it forms a multiplier.

**Why it works is not.** The quantity that actually controls the error is the **growth factor**,
how large the entries of U become relative to those of A. Partial pivoting bounds it by
``2^(n-1)``, which is uselessly large, and Wilkinson constructed a matrix attaining it exactly.
Yet on real data growth is almost always tiny. That gap between the proven worst case and
observed behaviour is one of the genuinely open questions in the subject, and lesson 18
measures both ends of it.

Complete pivoting has a proven polynomial bound and is not used, because the extra O(n^3)
comparisons cost more than the rare bad case they prevent.
"""

from __future__ import annotations

import numpy as np


def plu_factor(A) -> dict:
    """PA = LU by Gaussian elimination with partial pivoting.

    At each step the largest-magnitude entry in the current column is brought to the pivot
    position by a row swap. Returns a dict with:

    ``P`` the permutation matrix, ``perm`` the same thing as an index array (which is how you
    should actually use it), ``L`` unit lower triangular with every ``|L_ij| <= 1``, ``U`` upper
    triangular, ``n_swaps`` for the determinant sign, and ``growth`` the growth factor
    ``max|U| / max|A|``.

    The multipliers are bounded by 1 by construction, which is the point. Compare
    ``nalib.lu.lu_factor``, where a multiplier of 1e16 is possible and does real damage.

    Cost: the same 2n^3/3 as without pivoting, plus O(n^2) comparisons and the swaps. Pivoting
    is essentially free, which is why it is always on.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    perm = np.arange(n)
    n_swaps = 0
    max_A = np.max(np.abs(A)) if A.size else 0.0

    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if U[p, k] == 0.0:
            continue                                   # the whole column is zero
        if p != k:
            U[[k, p], :] = U[[p, k], :]
            L[[k, p], :k] = L[[p, k], :k]              # swap only the finished part of L
            perm[[k, p]] = perm[[p, k]]
            n_swaps += 1
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
        U[k + 1:, k] = 0.0

    P = np.eye(n)[perm]
    growth = (np.max(np.abs(U)) / max_A) if max_A > 0 else 1.0
    return {"P": P, "perm": perm, "L": L, "U": U,
            "n_swaps": n_swaps, "growth": float(growth)}


def plu_solve(fac: dict, b) -> np.ndarray:
    """Solve Ax = b from a ``plu_factor`` result.

    PAx = Pb, and PA = LU, so solve LUx = Pb: forward substitute, then back substitute.
    Permuting b is an index operation, never a matrix multiply.
    """
    from .lu import back_substitution, forward_substitution

    b = np.asarray(b, dtype=float).ravel()
    y = forward_substitution(fac["L"], b[fac["perm"]], unit_diagonal=True)
    return back_substitution(fac["U"], y)


def complete_pivot_factor(A) -> dict:
    """PAQ = LU with complete pivoting: search the whole remaining submatrix for the pivot.

    Complete pivoting has a growth factor bound that is **polynomial-ish** rather than
    exponential (Wilkinson's bound is roughly ``n^(0.25 log n + 0.5)``), which is far better
    than partial pivoting's ``2^(n-1)``.

    Nobody uses it. The search costs O(n^3) comparisons in total, against O(n^2) for partial
    pivoting, and it destroys the column-oriented memory access that makes blocked LU fast. The
    cost is paid on every matrix; the benefit arrives on almost none. Lesson 18 measures both
    halves of that trade.

    Returns ``P``, ``Q``, ``L``, ``U``, ``row_perm``, ``col_perm`` and ``growth``.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    row_perm = np.arange(n)
    col_perm = np.arange(n)
    max_A = np.max(np.abs(A)) if A.size else 0.0

    for k in range(n - 1):
        sub = np.abs(U[k:, k:])
        i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
        i, j = i + k, j + k
        if U[i, j] == 0.0:
            continue
        if i != k:
            U[[k, i], :] = U[[i, k], :]
            L[[k, i], :k] = L[[i, k], :k]
            row_perm[[k, i]] = row_perm[[i, k]]
        if j != k:
            U[:, [k, j]] = U[:, [j, k]]
            col_perm[[k, j]] = col_perm[[j, k]]
        for r in range(k + 1, n):
            L[r, k] = U[r, k] / U[k, k]
            U[r, k:] -= L[r, k] * U[k, k:]
        U[k + 1:, k] = 0.0

    growth = (np.max(np.abs(U)) / max_A) if max_A > 0 else 1.0
    return {"P": np.eye(n)[row_perm], "Q": np.eye(n)[:, col_perm],
            "L": L, "U": U, "row_perm": row_perm, "col_perm": col_perm,
            "growth": float(growth)}


def growth_factor(A, pivoting: str = "partial") -> float:
    """max|U| / max|A| after elimination. The quantity that controls the error.

    Wilkinson's backward error bound for Gaussian elimination is

        ||dA||  <=  c n^2 rho u ||A||,

    where ``rho`` is this growth factor. Everything else in that bound is modest, so **rho is
    the entire question of whether elimination is stable on a given matrix**.

    ``pivoting`` may be ``"none"``, ``"partial"`` or ``"complete"``.

    One definitional subtlety, since it surprises people. This ratio is **not** bounded below
    by 1. If the largest entry of A sits below the diagonal, elimination zeroes it and it never
    appears in U at all, so ``max|U|`` can be smaller than ``max|A|``. Measured: about 6
    percent of random 8 by 8 Gaussian matrices give a growth factor under 1, the smallest
    around 0.69. Definitions that take the maximum over **every intermediate matrix** rather
    than over U alone are bounded below by 1. Both appear in the literature; this is the U-only
    version, which is the one Trefethen and Bau use.
    """
    A = np.asarray(A, dtype=float)
    if pivoting == "partial":
        return plu_factor(A)["growth"]
    if pivoting == "complete":
        return complete_pivot_factor(A)["growth"]
    if pivoting == "none":
        from .lu import lu_factor

        _, U = lu_factor(A)
        m = np.max(np.abs(A))
        return float(np.max(np.abs(U)) / m) if m > 0 else 1.0
    raise ValueError("pivoting must be 'none', 'partial' or 'complete'")


def wilkinson_growth_matrix(n: int) -> np.ndarray:
    """The matrix that attains partial pivoting's worst case growth of exactly 2^(n-1).

    Unit lower triangular with -1 below the diagonal, and a final column of ones:

        [ 1  0  0  0  1]
        [-1  1  0  0  1]
        [-1 -1  1  0  1]
        [-1 -1 -1  1  1]
        [-1 -1 -1 -1  1]

    Partial pivoting **never swaps** on this matrix, because the diagonal entry is already the
    largest in its column at every step. Meanwhile the last column doubles each time, ending at
    2^(n-1).

    This is why the bound is not merely a weak proof: it is attained. At n = 60 the growth is
    about 5.8e17, which is more than 1/u, so every digit is lost. Lesson 18 measures exactly
    that.
    """
    A = np.eye(n) - np.tril(np.ones((n, n)), -1)
    A[:, -1] = 1.0
    return A


def multiplier_sizes(A, pivoting: str = "partial") -> np.ndarray:
    """All the multipliers used during elimination, so their sizes can be inspected.

    With partial pivoting every one satisfies ``|m| <= 1``. Without it, a multiplier can be
    astronomically large, and that is the immediate mechanism by which a small pivot destroys
    the answer: the pivot row, scaled by a huge multiplier, overwhelms the row it is subtracted
    from, and the original information in that row is lost to rounding. That is **swamping**.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    U = A.copy()
    out = []
    for k in range(n - 1):
        if pivoting == "partial":
            p = k + int(np.argmax(np.abs(U[k:, k])))
            if p != k:
                U[[k, p], :] = U[[p, k], :]
        if U[k, k] == 0.0:
            continue
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            out.append(abs(m))
            U[i, k:] -= m * U[k, k:]
    return np.array(out)
