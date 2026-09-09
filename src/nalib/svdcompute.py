"""Computing the SVD: bidiagonalization, implicit sweeps, and one-sided Jacobi.

What this module is for
-----------------------
Lesson 41 said what the SVD is. This one computes it, and the whole subject is organised around
a single prohibition: **never form ``A^T A``.**

Lesson 29 measured why for least squares and lesson 41 measured it again for singular values: at
``kappa = 1e10`` the ``A^T A`` route loses the smallest singular value completely while a route
that avoids it keeps seven digits. Every algorithm here is a way of getting the eigenvalues of
``A^T A`` without ever writing ``A^T A`` down.

Three of them:

- `bidiagonalize` reduces ``A`` to bidiagonal form with Householder reflectors from both sides.
  It is finite, it costs ``4mn^2 - 4n^3/3``, and it preserves the singular values exactly. Every
  method afterwards works on two vectors instead of a matrix.
- `golub_kahan_svd` runs an implicit QR sweep on the bidiagonal matrix. The sweep is the QR
  algorithm of lesson 37 applied to ``B^T B``, arranged so that ``B^T B`` is never formed: the
  shift is computed from a ``2x2`` block and then chased down the bidiagonal as a bulge.
- `one_sided_jacobi` orthogonalizes the **columns of A** by plane rotations, which is lesson
  38's Jacobi applied to ``A^T A`` implicitly. It is the slowest and it computes small singular
  values to **high relative accuracy**, which is a guarantee the others do not offer.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import qr as _qr


@dataclass
class SVDComputation:
    """Singular values, optional vectors, and the history that makes the method measurable."""

    s: np.ndarray
    U: np.ndarray | None = None
    Vt: np.ndarray | None = None
    iterations: int = 0
    converged: bool = True
    message: str = ""
    history: list = field(default_factory=list)
    rotations: int = 0


# ----------------------------------------------------------------------------- bidiagonal


def bidiagonalize(A, compute_uv: bool = False):
    """Householder from both sides until only the diagonal and superdiagonal remain.

    ``A = U B V^T`` with ``B`` upper bidiagonal. **The singular values are preserved exactly**,
    because ``U`` and ``V`` are orthogonal and orthogonal transformations on either side leave
    the singular values alone. That identity is checkable at any size without a reference, which
    is what makes it a good foundation.

    **The right-hand reflector starts one column later than the left-hand one.** Starting it at
    the same column would destroy the zeros the left-hand reflector just created and the process
    would never terminate, which is the same one-index subtlety as lesson 37's Hessenberg
    reduction.

    Returns ``(diag, off)`` or ``(diag, off, U, Vt)``. The two diagonals are returned rather than
    a matrix because everything afterwards wants exactly those ``2n - 1`` numbers.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    if m < n:
        raise ValueError(f"A is {A.shape}; bidiagonalize expects m >= n, transpose it first")
    B = A.astype(float).copy()
    U = np.eye(m) if compute_uv else None
    V = np.eye(n) if compute_uv else None

    for k in range(n):
        v, _ = _qr.householder_vector(B[k:, k])
        vv = float(v @ v)
        if vv > 0.0:
            B[k:, k:] -= (2.0 / vv) * np.outer(v, v @ B[k:, k:])
            if U is not None:
                U[:, k:] -= (2.0 / vv) * np.outer(U[:, k:] @ v, v)
        B[k + 1:, k] = 0.0
        if k < n - 2:
            w, _ = _qr.householder_vector(B[k, k + 1:])
            ww = float(w @ w)
            if ww > 0.0:
                B[:, k + 1:] -= (2.0 / ww) * np.outer(B[:, k + 1:] @ w, w)
                if V is not None:
                    V[:, k + 1:] -= (2.0 / ww) * np.outer(V[:, k + 1:] @ w, w)
            B[k, k + 2:] = 0.0

    diag = np.diag(B[:n, :n]).copy()
    off = np.diag(B[:n, :n], 1).copy() if n > 1 else np.zeros(0)
    if compute_uv:
        return diag, off, U[:, :n], V.T
    return diag, off


def bidiagonal_matrix(diag, off) -> np.ndarray:
    """Rebuild the bidiagonal matrix from its two diagonals. For checking, not for computing."""
    d = np.asarray(diag, dtype=float).ravel()
    e = np.asarray(off, dtype=float).ravel()
    n = d.size
    if e.size != max(n - 1, 0):
        raise ValueError(f"the off-diagonal has {e.size} entries, expected {max(n - 1, 0)}")
    return np.diag(d) + (np.diag(e, 1) if n > 1 else 0.0)


# ----------------------------------------------------------------------------- implicit sweep


def wilkinson_shift_squared(diag, off) -> float:
    """The Wilkinson shift for ``B^T B``, computed from ``B`` alone.

    The trailing ``2x2`` block of ``B^T B`` is

        [[d_{n-1}^2 + e_{n-2}^2,  d_{n-1} e_{n-1}],
         [d_{n-1} e_{n-1},        d_n^2 + e_{n-1}^2]]

    and its eigenvalue nearer the bottom right entry is the shift. **Every entry is a product of
    two entries of ``B``**, so nothing is squared that was not already going to be, and the
    catastrophic loss of forming the whole of ``B^T B`` is avoided.
    """
    d = np.asarray(diag, dtype=float).ravel()
    e = np.asarray(off, dtype=float).ravel()
    n = d.size
    if n < 2:
        return float(d[-1] ** 2) if n else 0.0
    a = d[n - 2] ** 2 + (e[n - 3] ** 2 if n > 2 else 0.0)
    b = d[n - 2] * e[n - 2]
    c = d[n - 1] ** 2 + e[n - 2] ** 2
    delta = (a - c) / 2.0
    if b == 0.0:
        return float(c)
    sign = 1.0 if delta >= 0.0 else -1.0
    return float(c - b * b / (delta + sign * np.sqrt(delta * delta + b * b)))


def golub_kahan_svd(diag, off, tol: float = 1e-14, max_iter: int = 10_000,
                    zero_shift: bool = False) -> SVDComputation:
    """The implicit QR sweep on a bidiagonal matrix: the standard SVD algorithm.

    One sweep applies a Givens rotation determined by the shift to the first two columns, which
    creates a **bulge** just below the bidiagonal, then chases that bulge down with alternating
    row and column rotations until it falls off the end. The net effect is exactly one shifted QR
    step on ``B^T B``, and ``B^T B`` is never formed.

    ``zero_shift=True`` is Demmel and Kahan's variant, which uses no shift at all. It converges
    more slowly, and with no shift every rotation is a product of entries of ``B``, so no
    subtraction of nearly equal quantities occurs anywhere. Lesson 42 measures both, and measures
    that on a column-graded bidiagonal the shift is **not** what costs relative accuracy: the
    deflation thresholds are, and once those are local both variants reach 3e-15 out to a spread
    of 1e24.

    Deflation splits the problem whenever an off-diagonal entry is negligible relative to its
    neighbours, exactly as in lesson 37.
    """
    d = np.asarray(diag, dtype=float).copy()
    e = np.asarray(off, dtype=float).copy()
    n = d.size
    if e.size != max(n - 1, 0):
        raise ValueError(f"the off-diagonal has {e.size} entries, expected {max(n - 1, 0)}")
    out = SVDComputation(s=np.abs(d).copy())
    if n <= 1:
        out.s = np.abs(d)
        out.message = "nothing to do"
        return out

    high = n - 1
    steps = 0
    while high > 0 and steps < int(max_iter):
        # Deflate: a negligible off-diagonal entry splits the problem.
        if abs(e[high - 1]) <= tol * (abs(d[high]) + abs(d[high - 1])):
            e[high - 1] = 0.0
            high -= 1
            continue
        low = high
        while low > 0 and abs(e[low - 1]) > tol * (abs(d[low]) + abs(d[low - 1])):
            low -= 1

        # A negligible DIAGONAL entry is its own special case, and without it the sweep cannot
        # make progress: the shift is zero, every rotation is the identity, and the iteration
        # spins forever. Measured on diag [2, 0, 1] with off [0.5, 0.5]: 10000 iterations with
        # the off-diagonal stuck at exactly 0.5.
        #
        # The fix is the standard one: rotate from the left to chase the resulting entry along
        # the row until it falls off the end, which splits the problem in two.
        # The threshold has to be LOCAL. Comparing against the largest diagonal entry in the
        # block throws away legitimately small values on a graded matrix: at a spread of 1e16
        # with tol 1e-14, four of thirty diagonal entries trip a global test, and the computed
        # singular values then carry a relative error of 6.5e-3 instead of 3.3e-15. Comparing
        # against the neighbouring off-diagonals instead fixes that, leaves ordinary matrices
        # bit for bit unchanged in both answer and sweep count, and still catches a true zero.
        zero_at = -1
        for k in range(low, high + 1):
            local = (abs(e[k]) if k < high else 0.0) + (abs(e[k - 1]) if k > low else 0.0)
            if abs(d[k]) <= tol * local:
                zero_at = k
                break
        if zero_at >= 0 and zero_at < high:
            extra = e[zero_at]
            e[zero_at] = 0.0
            for k in range(zero_at + 1, high + 1):
                c, s = _qr.givens_rotation(d[k], extra)
                d[k] = c * d[k] + s * extra
                if k < high:
                    extra = -s * e[k]
                    e[k] = c * e[k]
            steps += 1
            continue

        mu = 0.0 if zero_shift else wilkinson_shift_squared(d[low:high + 1], e[low:high])
        # The implicit first rotation, from the shifted first column of B^T B.
        y = d[low] * d[low] - mu
        z = d[low] * e[low]
        for k in range(low, high):
            c, s = _qr.givens_rotation(y, z)
            # rotate columns k and k+1
            if k > low:
                e[k - 1] = c * y + s * z
            t1, t2 = d[k], e[k]
            d[k] = c * t1 + s * t2
            e[k] = -s * t1 + c * t2
            t3 = d[k + 1]
            z = s * t3
            d[k + 1] = c * t3
            # rotate rows k and k+1
            c, s = _qr.givens_rotation(d[k], z)
            d[k] = c * d[k] + s * z
            t1, t2 = e[k], d[k + 1]
            e[k] = c * t1 + s * t2
            d[k + 1] = -s * t1 + c * t2
            if k < high - 1:
                t3 = e[k + 1]
                y, z = e[k], s * t3
                e[k + 1] = c * t3
                y = e[k]
        steps += 1
        out.history.append(float(np.max(np.abs(e[low:high]))))

    out.iterations = steps
    out.converged = high <= 0
    out.message = ("converged" if out.converged
                   else f"hit the iteration limit of {max_iter}")
    out.s = np.sort(np.abs(d))[::-1]
    return out


def svd_via_bidiagonal(A, zero_shift: bool = False, tol: float = 1e-14) -> SVDComputation:
    """Bidiagonalize, then sweep. The two steps of the standard algorithm, in order."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    transposed = A.shape[0] < A.shape[1]
    work = A.T if transposed else A
    d, e = bidiagonalize(work)
    out = golub_kahan_svd(d, e, tol=tol, zero_shift=zero_shift)
    out.message += " (transposed first)" if transposed else ""
    return out


# ----------------------------------------------------------------------------- one-sided Jacobi


def one_sided_jacobi(A, tol: float = 1e-14, max_sweeps: int = 60,
                     compute_uv: bool = False) -> SVDComputation:
    """Orthogonalize the **columns of A** by plane rotations. The columns become ``U S``.

    This is lesson 38's Jacobi applied to ``A^T A`` without forming it. The rotation that would
    annihilate the ``(p, q)`` entry of ``A^T A`` is computed from the three inner products
    ``a_p . a_p``, ``a_p . a_q`` and ``a_q . a_q``, and applied to the **columns of A**. When the
    columns are mutually orthogonal, their norms are the singular values.

    **Its relative accuracy is the reason it survives.** Every quantity is an inner product of
    two columns, so a small singular value is computed from small numbers throughout and never
    from a difference of large ones. Lesson 38 measured the same effect for the symmetric
    eigenvalue problem, and lesson 42 measures it here.

    It is the slowest of the three and the easiest to parallelise, since rotations on disjoint
    column pairs commute.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    if m < n:
        raise ValueError(f"A is {A.shape}; one_sided_jacobi expects m >= n, transpose it first")
    W = A.astype(float).copy()
    V = np.eye(n) if compute_uv else None
    out = SVDComputation(s=np.zeros(n))

    def worst_off_diagonality(M):
        """The largest |cos| between two columns, measured on a SETTLED matrix.

        Measuring it as the sweep proceeds gives a number that can rise from one sweep to the
        next, because it mixes states from before and after the rotations. Measured on a 10 by 5
        matrix, the recorded trail went 0.530, 0.628, 0.114: not monotone, and not because
        anything was wrong. What decreases monotonically is off(A^T A) per ROTATION; this
        quantity, computed fresh at the end of each sweep, decreases per sweep.
        """
        norms = np.linalg.norm(M, axis=0)
        safe = np.where(norms > 0, norms, 1.0)
        G = (M / safe).T @ (M / safe)
        np.fill_diagonal(G, 0.0)
        return float(np.abs(G).max()) if G.size else 0.0

    for sweep in range(int(max_sweeps)):
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = float(W[:, p] @ W[:, p])
                beta = float(W[:, q] @ W[:, q])
                gamma = float(W[:, p] @ W[:, q])
                scale = np.sqrt(alpha * beta)
                if scale == 0.0:
                    continue
                if abs(gamma) <= tol * scale:
                    continue
                zeta = (beta - alpha) / (2.0 * gamma)
                sign = 1.0 if zeta >= 0.0 else -1.0
                t = sign / (abs(zeta) + np.sqrt(1.0 + zeta * zeta))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t
                col_p, col_q = W[:, p].copy(), W[:, q].copy()
                W[:, p] = c * col_p - s * col_q
                W[:, q] = s * col_p + c * col_q
                if V is not None:
                    v_p, v_q = V[:, p].copy(), V[:, q].copy()
                    V[:, p] = c * v_p - s * v_q
                    V[:, q] = s * v_p + c * v_q
                out.rotations += 1
        out.iterations = sweep + 1
        worst = worst_off_diagonality(W)
        out.history.append(worst)
        if worst <= tol:
            out.converged = True
            out.message = f"columns orthogonal after {sweep + 1} sweeps"
            break
    else:
        out.converged = False
        out.message = f"hit the sweep limit of {max_sweeps}"

    norms = np.linalg.norm(W, axis=0)
    order = np.argsort(norms)[::-1]
    out.s = norms[order]
    if compute_uv:
        safe = np.where(out.s > 0, out.s, 1.0)
        out.U = W[:, order] / safe
        out.Vt = V[:, order].T
    return out


# ----------------------------------------------------------------------------- comparison


def relative_error(computed, exact) -> float:
    """The worst relative error, which is the quantity the three methods differ on."""
    computed = np.sort(np.asarray(computed, dtype=float).ravel())[::-1]
    exact = np.sort(np.asarray(exact, dtype=float).ravel())[::-1]
    if computed.size != exact.size:
        raise ValueError(f"{computed.size} computed against {exact.size} exact")
    denom = np.where(exact > 0, exact, 1.0)
    return float(np.max(np.abs(computed - exact) / denom))


def graded_columns(m: int, n: int, spread: float, rng=None, dps: int = 60) -> dict:
    """A matrix whose columns are scaled over many decades, with a **genuine** reference.

    Built as ``B D`` with ``D`` diagonal: scaling the columns scales the singular values, and
    the small ones remain determined to high relative accuracy by the entries. That is the
    hypothesis under which one-sided Jacobi's relative accuracy theorem applies.

    The reference comes from mpmath at ``dps`` digits, not from another double precision SVD,
    for the reason lesson 38 records: an earlier comparison there measured LAPACK against
    LAPACK and scored it exactly zero.
    """
    m, n = int(m), int(n)
    if m < n:
        raise ValueError(f"expected m >= n, got ({m}, {n})")
    gen = np.random.default_rng() if rng is None else rng
    base = gen.standard_normal((m, n))
    base /= np.linalg.norm(base, axis=0)
    scales = np.geomspace(1.0, float(spread), n)
    A = base * scales

    import mpmath

    with mpmath.workdps(int(dps)):
        M = mpmath.matrix([[mpmath.mpf(float(A[i, j])) for j in range(n)] for i in range(m)])
        _, S, _ = mpmath.mp.svd_r(M)
        exact = np.sort(np.array([float(S[i]) for i in range(n)]))[::-1]
    return {"A": A, "base": base, "scales": scales, "exact": exact}
