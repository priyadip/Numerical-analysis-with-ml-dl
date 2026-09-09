"""Banded, tridiagonal and sparse systems: getting below O(n^3).

Every method in Part 3 so far costs O(n^3) and O(n^2) memory. At n = 10^6, which is an ordinary
size for a discretised PDE, that is 10^18 flops and 8 terabytes. Neither is available.

The escape is that such matrices are almost entirely zero, and the zeros are not accidental:
they say that unknown i and unknown j are not directly coupled. Exploiting them changes the
cost by orders of magnitude rather than by constants.

**Bandedness.** When every nonzero lies within p diagonals of the main one, elimination never
touches anything outside the band, so LU costs O(n p^2) instead of O(n^3). For a tridiagonal
matrix, p = 1 and the cost is **O(n)**, which is the Thomas algorithm.

**Sparsity in general.** Elimination creates nonzeros where the matrix had zeros. That is
**fill-in**, it is the central difficulty of sparse direct methods, and it depends enormously
on the ORDER of the unknowns. Reordering is not an optimisation detail; it decides whether the
factorization fits in memory at all.

Used by lesson 21, and again in Part 4, where matrices too large even for a sparse
factorization are handled by never factorizing at all.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- structure


def bandwidths(A, tol: float = 0.0) -> tuple[int, int]:
    """(lower, upper) bandwidth: how many diagonals below and above the main one are nonzero.

    A diagonal matrix gives (0, 0), tridiagonal gives (1, 1), and a dense matrix gives
    (n-1, n-1). Entries with magnitude at or below ``tol`` count as zero.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    rows, cols = np.nonzero(np.abs(A) > tol)
    if rows.size == 0:
        return 0, 0
    diff = rows - cols
    return int(max(diff.max(), 0)), int(max((-diff).max(), 0))


def is_banded(A, lower: int, upper: int, tol: float = 0.0) -> bool:
    """True when every nonzero of A lies within the given band."""
    lo, up = bandwidths(A, tol)
    return lo <= lower and up <= upper


def density(A, tol: float = 0.0) -> float:
    """Fraction of entries that are nonzero. The reciprocal of how much sparsity buys."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    return float(np.count_nonzero(np.abs(A) > tol) / A.size) if A.size else 0.0


# ---------------------------------------------------------------- tridiagonal


def thomas(a, b, c, d) -> np.ndarray:
    """Solve a tridiagonal system in O(n) by the Thomas algorithm.

    The system is

        a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i]

    with ``b`` the main diagonal (length n), ``a`` the subdiagonal and ``c`` the superdiagonal.
    ``a`` and ``c`` may be given with length n-1, or length n with the unused end entry ignored.

    This is exactly Gaussian elimination with every operation that touches a structural zero
    removed. One forward sweep eliminates the subdiagonal, one backward sweep substitutes.

    Cost: about 8n flops and O(n) memory, against 2n^3/3 and n^2 for dense LU. At n = 10^6 that
    is the difference between a millisecond and a geological age.

    **No pivoting.** That is safe when the matrix is diagonally dominant or symmetric positive
    definite, which covers essentially every tridiagonal system arising from a discretised
    differential equation. It is NOT safe in general, and lesson 21 measures it failing.
    """
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    if n == 0:
        return np.zeros(0)
    a = np.asarray(a, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    if d.size != n:
        raise ValueError(f"right-hand side has length {d.size}, expected {n}")

    # Accept either n-1 or n for the off-diagonals, so both conventions work, and reject
    # anything else. Padding a short diagonal with zeros would silently solve a DIFFERENT
    # system and return a confident wrong answer, which is the one outcome to rule out.
    for name, v in (("a", a), ("c", c)):
        if v.size not in (max(n - 1, 0), n):
            raise ValueError(
                f"off-diagonal {name} has length {v.size}, expected {max(n - 1, 0)} or {n}")
    sub = np.zeros(n)
    sup = np.zeros(n)
    if n > 1:
        sub[1:] = a[-(n - 1):]
        sup[:-1] = c[:n - 1]

    cp = np.zeros(n)
    dp = np.zeros(n)
    beta = b[0]
    if beta == 0.0:
        raise np.linalg.LinAlgError("zero pivot at row 0; this system needs pivoting")
    cp[0] = sup[0] / beta
    dp[0] = d[0] / beta

    for i in range(1, n):
        beta = b[i] - sub[i] * cp[i - 1]
        if beta == 0.0:
            raise np.linalg.LinAlgError(
                f"zero pivot at row {i}; this system needs pivoting")
        cp[i] = sup[i] / beta
        dp[i] = (d[i] - sub[i] * dp[i - 1]) / beta

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def tridiagonal_matrix(a, b, c, n: int | None = None) -> np.ndarray:
    """Build a dense tridiagonal matrix from its three diagonals, for checking against.

    Scalars are broadcast to constant diagonals, which is the common case for a discretised
    operator. ``n`` is required when every argument is a scalar.
    """
    if n is None:
        main = np.atleast_1d(np.asarray(b, dtype=float))
        if not np.isscalar(b) and main.size >= 1:
            n = int(main.size)                    # the main diagonal fixes n, including n = 1
        else:
            for v in (a, c):
                arr = np.atleast_1d(np.asarray(v, dtype=float))
                if not np.isscalar(v) and arr.size >= 1:
                    n = int(arr.size) + 1
                    break
            else:
                raise ValueError("n is required when all three diagonals are scalars")
    main = np.full(n, float(b)) if np.isscalar(b) else np.asarray(b, dtype=float).ravel()
    sub = np.full(n - 1, float(a)) if np.isscalar(a) else np.asarray(a, dtype=float).ravel()[:n - 1]
    sup = np.full(n - 1, float(c)) if np.isscalar(c) else np.asarray(c, dtype=float).ravel()[:n - 1]
    return np.diag(main) + np.diag(sub, -1) + np.diag(sup, 1)


def second_difference(n: int, h: float = 1.0) -> np.ndarray:
    """The tridiagonal second difference operator, (-1, 2, -1)/h^2.

    The single most common matrix in all of numerical analysis. It is what discretising
    ``-u''`` gives, it appears in every part of this course from lesson 21 to Part 11, and it is
    symmetric positive definite and diagonally dominant, so Thomas needs no pivoting on it.

    Its eigenvalues are known exactly, ``4 sin^2(k pi / (2(n+1))) / h^2``, which makes it the
    standard matrix for testing anything that claims to compute eigenvalues or to converge at a
    predicted rate.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    return tridiagonal_matrix(-1.0 / h**2, 2.0 / h**2, -1.0 / h**2, n)


def second_difference_eigenvalues(n: int, h: float = 1.0) -> np.ndarray:
    """The exact eigenvalues of `second_difference`, for checking a solver against truth."""
    k = np.arange(1, n + 1)
    return 4.0 * np.sin(k * np.pi / (2 * (n + 1))) ** 2 / h**2


# ---------------------------------------------------------------- banded LU


def banded_lu(A, lower: int, upper: int) -> tuple[np.ndarray, np.ndarray]:
    """LU without pivoting, touching only the band. Returns dense L and U for inspection.

    The key structural fact: **elimination inside a band stays inside the band**. Row i has
    nonzeros only in columns ``i-lower`` to ``i+upper``, and subtracting a multiple of row k
    from row i cannot create a nonzero outside that range. So L keeps the lower bandwidth and U
    keeps the upper one, and no work is ever done on a structural zero.

    Cost: O(n * lower * upper) instead of O(n^3). For a tridiagonal matrix that is O(n).

    Pivoting would break this, which is why banded solvers avoid it when they can: a row swap
    can move a nonzero outside the band, widening it and destroying the saving. LAPACK's banded
    solver with pivoting therefore allocates ``lower`` extra superdiagonals in advance to hold
    exactly that fill.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    for k in range(n - 1):
        if U[k, k] == 0.0:
            raise np.linalg.LinAlgError(f"zero pivot at step {k}; banded LU needs pivoting")
        for i in range(k + 1, min(k + lower + 1, n)):        # only inside the band
            L[i, k] = U[i, k] / U[k, k]
            hi = min(k + upper + 1, n)
            U[i, k:hi] -= L[i, k] * U[k, k:hi]
            U[i, k] = 0.0
    return L, U


def flops_banded_lu(n: int, lower: int, upper: int) -> int:
    """Exact flop count for banded LU without pivoting.

    At step k there are at most ``lower`` rows to update, each across at most ``upper + 1``
    columns, giving roughly ``2 n * lower * upper``. Compare 2n^3/3 for the dense case: the
    saving is a factor of about ``n^2 / (3 * lower * upper)``.
    """
    total = 0
    for k in range(n - 1):
        rows = min(lower, n - 1 - k)
        cols = min(upper + 1, n - k)
        total += rows + 2 * rows * cols
    return total


# ---------------------------------------------------------------- sparse storage


def to_coo(A, tol: float = 0.0) -> dict:
    """Coordinate format: parallel arrays of row, column and value for each nonzero.

    The simplest sparse format and the easiest to build incrementally, which is why assembly
    code produces it. It is poor for arithmetic, because finding a given row means searching,
    so it is normally converted to CSR before anything is computed.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    rows, cols = np.nonzero(np.abs(A) > tol)
    return {"row": rows, "col": cols, "data": A[rows, cols], "shape": A.shape}


def to_csr(A, tol: float = 0.0) -> dict:
    """Compressed sparse row: values, their column indices, and where each row starts.

    ``indptr`` has length n+1, and row i occupies ``data[indptr[i]:indptr[i+1]]``. That single
    change from COO makes a row lookup O(1) instead of a search, which is what a matrix-vector
    product needs, and a matvec is the operation every Krylov method in Part 4 is built from.

    Storage is ``2 nnz + n + 1`` numbers against ``n^2`` dense. For the second difference matrix
    that is about ``7n`` against ``n^2``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n_rows = A.shape[0]
    data, indices, indptr = [], [], [0]
    for i in range(n_rows):
        cols = np.nonzero(np.abs(A[i]) > tol)[0]
        indices.extend(cols.tolist())
        data.extend(A[i, cols].tolist())
        indptr.append(len(data))
    return {"data": np.array(data), "indices": np.array(indices, dtype=int),
            "indptr": np.array(indptr, dtype=int), "shape": A.shape}


def csr_matvec(csr: dict, x) -> np.ndarray:
    """y = A x from CSR, touching only the nonzeros.

    Cost is ``2 nnz`` flops, independent of n except through nnz. This is the only operation a
    Krylov method needs, which is why Part 4 can solve systems that could never be factorized.
    """
    x = np.asarray(x, dtype=float).ravel()
    n_rows = csr["shape"][0]
    if x.size != csr["shape"][1]:
        raise ValueError(f"x has length {x.size}, expected {csr['shape'][1]}")
    y = np.zeros(n_rows)
    for i in range(n_rows):
        lo, hi = csr["indptr"][i], csr["indptr"][i + 1]
        if hi > lo:
            y[i] = csr["data"][lo:hi] @ x[csr["indices"][lo:hi]]
    return y


def csr_to_dense(csr: dict) -> np.ndarray:
    """Rebuild the dense matrix, for checking a CSR round trip."""
    A = np.zeros(csr["shape"])
    for i in range(csr["shape"][0]):
        lo, hi = csr["indptr"][i], csr["indptr"][i + 1]
        A[i, csr["indices"][lo:hi]] = csr["data"][lo:hi]
    return A


def storage_counts(A, tol: float = 0.0) -> dict:
    """Numbers stored by dense, COO and CSR, so the three can be compared honestly."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    nnz = int(np.count_nonzero(np.abs(A) > tol))
    n_rows = A.shape[0]
    return {"nnz": nnz, "dense": int(A.size), "coo": 3 * nnz, "csr": 2 * nnz + n_rows + 1}


# ---------------------------------------------------------------- fill-in


def fill_in(A, tol: float = 1e-14) -> dict:
    """Count the nonzeros that elimination CREATES where A had zeros.

    Fill-in is the whole difficulty of sparse direct methods. A matrix can be 99 percent zero
    and still have a completely dense factorization, in which case sparsity bought nothing and
    the factorization may not even fit in memory.

    Returns the nonzero count before and after, and the fill, meaning how many new nonzeros
    appeared.
    """
    from .lu import lu_factor

    A = np.atleast_2d(np.asarray(A, dtype=float))
    before = int(np.count_nonzero(np.abs(A) > tol))
    L, U = lu_factor(A)
    combined = np.abs(L - np.eye(A.shape[0])) + np.abs(U)
    after = int(np.count_nonzero(combined > tol))
    return {"nnz_before": before, "nnz_after": after, "fill": after - before,
            "density_before": before / A.size, "density_after": after / A.size}


def arrow_matrix(n: int, tip_first: bool = True) -> np.ndarray:
    """The arrow matrix: a full first row and column plus a diagonal, or its reversal.

    The cleanest demonstration that **ordering decides everything**. Both versions have exactly
    the same sparsity count and represent the same graph, relabelled.

    - ``tip_first=True`` puts the dense row and column first. Eliminating that first variable
      couples every remaining variable to every other, and the factorization becomes
      **completely dense**.
    - ``tip_first=False`` puts it last. Every elimination step is then independent and there is
      **zero fill**.

    Same matrix, same sparsity, opposite outcomes. That is why sparse solvers spend real effort
    on reordering before they factorize anything.
    """
    if n < 2:
        raise ValueError("an arrow matrix needs at least 2 rows")
    A = np.diag(np.arange(2.0, n + 2.0))
    if tip_first:
        A[0, :] = 1.0
        A[:, 0] = 1.0
        A[0, 0] = float(n + 2)
    else:
        A[-1, :] = 1.0
        A[:, -1] = 1.0
        A[-1, -1] = float(n + 2)
    return A


def reverse_cuthill_mckee(A, tol: float = 0.0) -> np.ndarray:
    """A permutation that reduces the bandwidth, by breadth-first search on the sparsity graph.

    Treat the matrix as the adjacency of a graph: entry (i, j) nonzero means unknown i touches
    unknown j. Cuthill-McKee visits vertices breadth first from a low-degree start, which keeps
    connected vertices close together in the new ordering, and reversing the result reduces the
    bandwidth further.

    Reordering is a **relabelling of the unknowns**, so it changes no answer at all. It changes
    only how much work and memory the factorization needs, and lesson 21 measures that the
    change can be enormous.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    adjacency = (np.abs(A) > tol) | (np.abs(A.T) > tol)
    np.fill_diagonal(adjacency, False)
    degree = adjacency.sum(axis=1)

    order: list[int] = []
    seen = np.zeros(n, dtype=bool)
    while len(order) < n:
        remaining = np.nonzero(~seen)[0]
        start = int(remaining[np.argmin(degree[remaining])])
        queue = [start]
        seen[start] = True
        while queue:
            v = queue.pop(0)
            order.append(v)
            nbrs = [w for w in np.nonzero(adjacency[v])[0] if not seen[w]]
            for w in sorted(nbrs, key=lambda z: degree[z]):
                seen[w] = True
                queue.append(int(w))
    return np.array(order[::-1], dtype=int)
