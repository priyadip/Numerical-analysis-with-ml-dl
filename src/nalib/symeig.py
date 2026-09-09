"""The symmetric eigenvalue problem: four algorithms the general case cannot have.

What this module is for
-----------------------
Lesson 35 measured the fact this whole lesson rests on: **every eigenvalue of a symmetric matrix
has condition number exactly 1**, because its left and right eigenvectors coincide. There is
nothing to be ill conditioned about, and Weyl's inequality says a perturbation ``E`` moves every
eigenvalue by at most ``||E||_2``.

That buys four algorithms with properties the general QR algorithm cannot offer:

- `jacobi_eigen` zeroes one off-diagonal pair at a time with a plane rotation. It is the oldest
  method (1846), it converges **quadratically**, it is embarrassingly parallel, and it computes
  **small eigenvalues to high relative accuracy** where every other method here gives only
  absolute accuracy. It is also the slowest, which is why it was abandoned and then revived.
- `tridiagonalize` is the Hessenberg reduction of lesson 37 applied to a symmetric matrix, where
  it produces a **tridiagonal** matrix. Everything after this point costs ``O(n)`` per step
  instead of ``O(n^2)``.
- `sturm_count` counts how many eigenvalues lie below a given point, from one ``O(n)`` sweep.
  That turns eigenvalue finding into **bisection**, and it is the only method here that can
  compute the 37th eigenvalue without computing the other 36.
- `divide_and_conquer` splits a tridiagonal matrix into two halves plus a rank-one correction,
  solves each half, and glues them with a **secular equation**. It is the fastest method for
  large matrices when eigenvectors are wanted.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import qr as _qr


@dataclass
class SymEigResult:
    """Eigenvalues, optional eigenvectors, and the history needed to measure a rate."""

    values: np.ndarray
    vectors: np.ndarray | None
    iterations: int
    converged: bool
    message: str
    off_diagonal: list = field(default_factory=list)     # ||off(A)||_F each sweep
    rotations: int = 0


def off_norm(A) -> float:
    """The Frobenius norm of everything off the diagonal: what Jacobi drives to zero.

    **Computed by summing the off-diagonal entries, not as ``sqrt(||A||^2 - ||diag||^2)``.**
    The second form is algebraically identical and numerically useless near convergence, which
    is exactly where it gets used: once the off-diagonal is at the roundoff level the two
    squared norms agree to their last bit, and the subtraction returns **exactly zero**.

    That is lesson 05's cancellation, and here it is not merely inaccurate. The zero is fed to
    the convergence test, Jacobi stops several sweeps early, and the eigen*vectors* come back
    wrong by up to ``3e-9`` while the eigen*values*, being second order accurate, still look
    perfect. Measured on a 3 by 3 matrix before this was fixed.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    work = A.copy()
    np.fill_diagonal(work, 0.0)
    return float(np.linalg.norm(work))


def require_symmetric(A, tol: float = 1e-10) -> np.ndarray:
    """Return ``A`` as a float array, refusing anything that is not symmetric.

    Silently symmetrising would be worse than refusing: the caller would get the eigenvalues of
    ``(A + A^T)/2``, which are not the eigenvalues of ``A``, with nothing to indicate it.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A is {A.shape}, eigenvalues need a square matrix")
    asym = float(np.linalg.norm(A - A.T))
    if asym > tol * max(float(np.linalg.norm(A)), 1.0):
        raise ValueError(f"A is not symmetric: ||A - A^T|| = {asym:.3e}")
    return A


# ----------------------------------------------------------------------------- Jacobi


def jacobi_rotation(a_pp: float, a_pq: float, a_qq: float) -> tuple[float, float]:
    """The ``(c, s)`` of the plane rotation that makes the ``(p, q)`` entry exactly zero.

    Setting the off-diagonal of the rotated ``2x2`` block to zero gives
    ``t^2 + 2 theta t - 1 = 0`` with ``theta = (a_qq - a_pp) / (2 a_pq)``, and the root of
    **smaller** modulus is the one to take:

        t = sign(theta) / (|theta| + sqrt(theta^2 + 1)).

    **The smaller root is chosen for two reasons and both matter.** It is the rotation of angle
    below 45 degrees, which keeps the diagonal entries in the same order and makes the method
    converge rather than shuffle. And writing it this way adds like-signed quantities in the
    denominator, avoiding lesson 05's cancellation, where the algebraically equivalent
    ``t = -theta + sqrt(theta^2+1)`` loses all its digits for large ``theta``.
    """
    if a_pq == 0.0:
        return 1.0, 0.0
    theta = (a_qq - a_pp) / (2.0 * a_pq)
    sign = 1.0 if theta >= 0.0 else -1.0
    t = sign / (abs(theta) + np.sqrt(theta * theta + 1.0))
    c = 1.0 / np.sqrt(t * t + 1.0)
    return float(c), float(t * c)


def jacobi_eigen(A, tol: float = 1e-13, max_sweeps: int = 60,
                 compute_vectors: bool = True, cyclic: bool = True) -> SymEigResult:
    """Zero the off-diagonal entries one pair at a time with plane rotations.

    Each rotation is an orthogonal similarity that makes one off-diagonal entry zero and
    **reduces ``off(A)`` by exactly ``2 a_pq^2``**, which is why the method converges: the
    quantity being minimised decreases monotonically by a computable amount every step.

    Earlier entries do not stay zero, which is why it needs sweeps rather than one pass. But
    convergence is **quadratic** once the off-diagonal norm is small: after a full sweep,
    ``off(A_new) <= off(A)^2 / something``, so the last few sweeps each square the error.

    ``cyclic=True`` sweeps ``(p, q)`` in a fixed order, which is what implementations use.
    ``cyclic=False`` picks the largest off-diagonal entry each time, which needs fewer rotations
    and more time finding them; lesson 38 measures both.

    **It is the slowest method here and the most accurate**, and the accuracy claim needs
    stating carefully because the folklore overstates it. Measured against an 80 digit reference
    on graded matrices with condition numbers from ``10^4`` to ``10^{20}``: Jacobi's **relative**
    error is flat at about ``1.5e-15`` throughout, while LAPACK's degrades from ``1.5e-15`` to
    ``2.3e-14`` and Sturm bisection collapses to ``2.7e-10``. So Jacobi beats LAPACK by about
    **14 times**, not by orders of magnitude, and at low condition numbers it is slightly worse.
    What is remarkable is the flatness, not the size of the gap.
    """
    A = require_symmetric(A)
    n = A.shape[0]
    M = A.copy()
    V = np.eye(n) if compute_vectors else None
    out = SymEigResult(values=np.diag(M).copy(), vectors=V, iterations=0,
                       converged=False, message="")
    scale = max(float(np.linalg.norm(A)), 1e-300)
    out.off_diagonal.append(off_norm(M))
    if n == 1:
        out.converged = True
        out.message = "a 1 by 1 matrix is already diagonal"
        return out

    for sweep in range(int(max_sweeps)):
        if off_norm(M) <= tol * scale:
            out.converged = True
            out.message = f"off-diagonal below tolerance after {sweep} sweeps"
            break
        pairs = ([(p, q) for p in range(n - 1) for q in range(p + 1, n)] if cyclic
                 else [_largest_offdiagonal(M)] * (n * (n - 1) // 2))
        for p, q in pairs:
            if not cyclic:
                p, q = _largest_offdiagonal(M)
            if M[p, q] == 0.0:
                continue
            c, s = jacobi_rotation(M[p, p], M[p, q], M[q, q])
            # Apply the rotation from both sides. Only rows and columns p and q change.
            col_p, col_q = M[:, p].copy(), M[:, q].copy()
            M[:, p] = c * col_p - s * col_q
            M[:, q] = s * col_p + c * col_q
            row_p, row_q = M[p, :].copy(), M[q, :].copy()
            M[p, :] = c * row_p - s * row_q
            M[q, :] = s * row_p + c * row_q
            M[p, q] = M[q, p] = 0.0            # assigned, not computed
            if V is not None:
                v_p, v_q = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * v_p - s * v_q
                V[:, q] = s * v_p + c * v_q
            out.rotations += 1
        out.iterations = sweep + 1
        out.off_diagonal.append(off_norm(M))
    else:
        out.message = f"hit the sweep limit of {max_sweeps}"

    order = np.argsort(np.diag(M))
    out.values = np.diag(M)[order].copy()
    out.vectors = V[:, order] if V is not None else None
    return out


def _largest_offdiagonal(M) -> tuple[int, int]:
    """The indices of the largest off-diagonal entry in modulus."""
    n = M.shape[0]
    work = np.abs(M).copy()
    np.fill_diagonal(work, -1.0)
    p, q = np.unravel_index(int(np.argmax(work)), work.shape)
    return (int(min(p, q)), int(max(p, q)))


# ----------------------------------------------------------------------------- tridiagonal


def tridiagonalize(A, compute_q: bool = True):
    """Householder similarities to tridiagonal form: ``A = Q T Q^T``.

    This is lesson 37's Hessenberg reduction, and a symmetric matrix comes out **tridiagonal**
    rather than merely Hessenberg, for free: the reduction preserves symmetry, and a symmetric
    Hessenberg matrix is tridiagonal.

    Costs ``(4/3) n^3`` without ``Q``, and it is the step that makes everything afterwards cheap:
    a QR step on a tridiagonal matrix is ``O(n)``, a Sturm count is ``O(n)``, and divide and
    conquer needs the tridiagonal structure to split at all.

    Returns ``(diagonal, off_diagonal, Q)`` or ``(diagonal, off_diagonal)``, storing the two
    arrays rather than a full matrix, since that is what the later routines want.
    """
    A = require_symmetric(A)
    n = A.shape[0]
    M = A.copy()
    Q = np.eye(n) if compute_q else None
    for k in range(n - 2):
        v, _ = _qr.householder_vector(M[k + 1:, k])
        vv = float(v @ v)
        if vv == 0.0:
            continue
        # A symmetric two-sided update, written so the result stays exactly symmetric.
        w = (2.0 / vv) * (M[k + 1:, k + 1:] @ v)
        w = w - ((1.0 / vv) * float(v @ w)) * v
        M[k + 1:, k + 1:] -= np.outer(v, w) + np.outer(w, v)
        alpha = float(np.linalg.norm(M[k + 1:, k]))
        sign = -1.0 if M[k + 1, k] >= 0 else 1.0
        M[k + 1, k] = M[k, k + 1] = sign * alpha
        M[k + 2:, k] = 0.0
        M[k, k + 2:] = 0.0
        if Q is not None:
            Q[:, k + 1:] -= (2.0 / vv) * np.outer(Q[:, k + 1:] @ v, v)
    diag = np.diag(M).copy()
    off = np.diag(M, -1).copy() if n > 1 else np.zeros(0)
    return (diag, off, Q) if compute_q else (diag, off)


def tridiagonal_matrix(diag, off) -> np.ndarray:
    """Rebuild the full matrix from its two diagonals. For checking, not for computing."""
    d = np.asarray(diag, dtype=float).ravel()
    e = np.asarray(off, dtype=float).ravel()
    n = d.size
    if e.size != max(n - 1, 0):
        raise ValueError(f"the off-diagonal has {e.size} entries, expected {max(n - 1, 0)}")
    return np.diag(d) + np.diag(e, 1) + np.diag(e, -1)


# ----------------------------------------------------------------------------- Sturm


def sturm_count(diag, off, x: float) -> int:
    """How many eigenvalues of the tridiagonal matrix are strictly less than ``x``.

    The recurrence is the ratio of successive leading principal minors, and the answer is the
    number of **negative** entries::

        d_1 = a_1 - x
        d_k = (a_k - x) - b_{k-1}^2 / d_{k-1}

    Each ``d_k`` is ``det(T_k - x I) / det(T_{k-1} - x I)``, so the number of sign changes in
    the sequence of determinants is the number of eigenvalues below ``x``, and that equals the
    number of negative ``d_k``. This is Sylvester's law of inertia in disguise.

    **It costs ``O(n)`` and needs no eigenvector, no factorization and no iteration.** So the
    ``k``-th eigenvalue can be bisected for without computing any of the others, which is
    something neither the QR algorithm nor Jacobi can do.

    A zero ``d_{k-1}`` is replaced by a tiny number rather than special-cased, which is the
    standard trick: it perturbs the count by nothing and avoids a division by zero.
    """
    d = np.asarray(diag, dtype=float).ravel()
    e = np.asarray(off, dtype=float).ravel()
    n = d.size
    if e.size != max(n - 1, 0):
        raise ValueError(f"the off-diagonal has {e.size} entries, expected {max(n - 1, 0)}")
    tiny = np.finfo(float).tiny
    count = 0
    q = d[0] - float(x)
    if q < 0.0:
        count += 1
    for k in range(1, n):
        if q == 0.0:
            q = tiny
        q = (d[k] - float(x)) - e[k - 1] * e[k - 1] / q
        if q < 0.0:
            count += 1
    return count


def gerschgorin_interval(diag, off) -> tuple[float, float]:
    """An interval guaranteed to contain every eigenvalue, from lesson 35's discs."""
    d = np.asarray(diag, dtype=float).ravel()
    e = np.asarray(off, dtype=float).ravel()
    n = d.size
    radius = np.zeros(n)
    if n > 1:
        radius[:-1] += np.abs(e)
        radius[1:] += np.abs(e)
    return float(np.min(d - radius)), float(np.max(d + radius))


def bisect_eigenvalue(diag, off, k: int, tol: float = 1e-13, max_iter: int = 200) -> float:
    """The ``k``-th smallest eigenvalue (0-indexed), by bisection on the Sturm count.

    **The selling point is that ``k`` is a parameter.** Wanting eigenvalue 37 of 1000 costs the
    same as wanting eigenvalue 1, and neither costs anything like computing all 1000. Nothing
    else in this module can do that.

    Each bisection step is one ``O(n)`` Sturm count, and the interval halves, so reaching
    machine precision from a Gerschgorin interval of width ``W`` takes about
    ``log2(W / (eps |lambda|))`` steps: around 50 in practice.
    """
    d = np.asarray(diag, dtype=float).ravel()
    n = d.size
    k = int(k)
    if not 0 <= k < n:
        raise ValueError(f"k must be between 0 and {n - 1}, got {k}")
    lo, hi = gerschgorin_interval(d, off)
    width = max(hi - lo, 1e-300)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        if sturm_count(d, off, mid) <= k:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(abs(lo), abs(hi), width * 1e-16):
            break
    return 0.5 * (lo + hi)


def eigenvalues_in(diag, off, low: float, high: float) -> int:
    """How many eigenvalues lie in ``[low, high)``. Two Sturm counts, ``O(n)`` each."""
    if high < low:
        raise ValueError(f"the interval [{low}, {high}) is empty")
    return sturm_count(diag, off, high) - sturm_count(diag, off, low)


def bisection_eigenvalues(diag, off, tol: float = 1e-13) -> np.ndarray:
    """Every eigenvalue by bisection, for comparison with the other methods."""
    d = np.asarray(diag, dtype=float).ravel()
    return np.array([bisect_eigenvalue(d, off, k, tol=tol) for k in range(d.size)])


# ----------------------------------------------------------------------------- divide and conquer


def secular_function(d, z, rho: float, x):
    """``f(x) = 1 + rho * sum_i z_i^2 / (d_i - x)``, whose roots are the new eigenvalues.

    Splitting a tridiagonal matrix as ``T = [T1, 0; 0, T2] + rho v v^T`` and diagonalising each
    half turns the eigenvalue problem into finding the roots of this scalar function. It has a
    pole at every ``d_i``, and exactly one root strictly between consecutive poles, which is
    what makes the root finding reliable: every root is bracketed by construction.
    """
    d = np.asarray(d, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if z.size != d.size:
        raise ValueError(f"z has {z.size} entries, d has {d.size}")
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # z[:, None], not z: a bare (n,) array broadcasts against the trailing axis of the
        # (n, k) denominator, silently producing an (n, n) array and a wrong answer that
        # looks like a plausible number. Measured: f was nonzero at every true eigenvalue.
        return 1.0 + float(rho) * np.sum((z[:, None] ** 2)
                                         / (d[:, None] - np.atleast_1d(x)[None, :]), axis=0)


def solve_secular(d, z, rho: float, tol: float = 1e-14, max_iter: int = 200) -> np.ndarray:
    """Find every root of the secular equation, one per interval between poles.

    Plain bisection is used rather than the specialised rational interpolation of LAPACK, since
    the point here is the **structure**: each root is bracketed between consecutive poles, so a
    bracketing method cannot fail. That is the property the whole method rests on.
    """
    d = np.asarray(d, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    rho = float(rho)
    order = np.argsort(d)
    d, z = d[order], z[order]
    n = d.size
    span = max(float(np.max(d) - np.min(d)), 1.0) + abs(rho) * float(z @ z)
    edges = list(d) + ([float(np.max(d)) + span] if rho > 0 else [])
    if rho < 0:
        edges = [float(np.min(d)) - span] + list(d)
    roots = []
    gap = np.finfo(float).eps
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        a = lo + gap * max(abs(lo), 1.0) * 4
        b = hi - gap * max(abs(hi), 1.0) * 4
        if not b > a:
            roots.append(0.5 * (lo + hi))
            continue
        fa = float(secular_function(d, z, rho, a)[0])
        fb = float(secular_function(d, z, rho, b)[0])
        if fa * fb > 0:                       # no sign change: the pole was deflated away
            roots.append(d[i] if rho > 0 else d[i])
            continue
        for _ in range(int(max_iter)):
            mid = 0.5 * (a + b)
            fm = float(secular_function(d, z, rho, mid)[0])
            if fa * fm <= 0:
                b = mid
            else:
                a, fa = mid, fm
            if b - a <= tol * max(abs(a), abs(b), 1.0):
                break
        roots.append(0.5 * (a + b))
    return np.sort(np.array(roots))


def divide_and_conquer(diag, off, cutoff: int = 8) -> np.ndarray:
    """Eigenvalues of a symmetric tridiagonal matrix, by splitting and gluing.

    Write ``T`` as two smaller tridiagonal blocks plus a rank-one correction::

        T = [T1  0 ]  +  rho * v v^T,     v = e_m + e_{m+1}
            [0   T2]

    where ``T1`` and ``T2`` are ``T`` with the coupling entry ``b_m`` removed from the two
    diagonal entries either side. Solve each half recursively, then find the eigenvalues of the
    whole by solving the secular equation.

    **The cost is ``O(n^2)`` for eigenvalues and about ``O(n^{2.3})`` in practice with
    eigenvectors**, against ``O(n^3)`` for the QR algorithm with vectors, which is why LAPACK's
    default driver for large symmetric problems is this one.

    ``cutoff`` is the size below which it stops recursing and calls a direct method.
    """
    d = np.asarray(diag, dtype=float).ravel()
    e = np.asarray(off, dtype=float).ravel()
    n = d.size
    if e.size != max(n - 1, 0):
        raise ValueError(f"the off-diagonal has {e.size} entries, expected {max(n - 1, 0)}")
    if n <= max(int(cutoff), 2):
        return np.sort(np.linalg.eigvalsh(tridiagonal_matrix(d, e)))

    m = n // 2
    rho = float(e[m - 1])
    d1 = d[:m].copy()
    d2 = d[m:].copy()
    d1[-1] -= rho
    d2[0] -= rho
    left = divide_and_conquer(d1, e[:m - 1], cutoff)
    right = divide_and_conquer(d2, e[m:], cutoff)

    # The rank-one vector, expressed in the eigenbases of the two halves.
    T1 = tridiagonal_matrix(d1, e[:m - 1])
    T2 = tridiagonal_matrix(d2, e[m:])
    _, Q1 = np.linalg.eigh(T1)
    _, Q2 = np.linalg.eigh(T2)
    z = np.concatenate([Q1[m - 1, :], Q2[0, :]])
    poles = np.concatenate([left, right])
    return solve_secular(poles, z, rho)


# ----------------------------------------------------------------------------- comparison


def relative_accuracy(computed, exact) -> float:
    """The worst **relative** error, which is the quantity Jacobi is special about.

    Absolute accuracy ``O(u ||A||)`` is easy and every method here achieves it. For an
    eigenvalue that is ``10^{-15}`` times the norm, that is a *relative* error of order 1: no
    correct digits at all. Demmel and Veselic proved Jacobi does better on a graded matrix, and
    `graded_symmetric` sets up the measurement.
    """
    computed = np.sort(np.asarray(computed, dtype=float).ravel())
    exact = np.sort(np.asarray(exact, dtype=float).ravel())
    if computed.size != exact.size:
        raise ValueError(f"{computed.size} computed against {exact.size} exact")
    denom = np.where(np.abs(exact) > 0, np.abs(exact), 1.0)
    return float(np.max(np.abs(computed - exact) / denom))


def graded_symmetric(n: int, spread: float, rng=None, dps: int = 80) -> dict:
    """A symmetric matrix whose eigenvalues span many decades, with a **genuine** reference.

    Constructed as ``D B D`` rather than ``Q D Q^T``: a **graded** matrix, where the small
    eigenvalues are determined to high relative accuracy by the entries and a good algorithm can
    recover them. On a ``Q D Q^T`` construction the small eigenvalues are already destroyed by
    the rounding of the product, so no algorithm could find them and the comparison would
    measure nothing.

    **The reference is computed by mpmath at ``dps`` decimal digits**, not by another double
    precision eigensolver. An earlier version of this function used
    ``eigvalsh(longdouble(A).astype(float))``, which is ``eigvalsh(A)`` with extra steps: the
    round trip through longdouble and back changes nothing, so the reference was the very
    routine being tested and it scored an error of exactly zero. A comparison against yourself
    always wins.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    gen = np.random.default_rng() if rng is None else rng
    scales = np.geomspace(1.0, float(spread), n)
    base = gen.standard_normal((n, n))
    base = base + base.T
    np.fill_diagonal(base, np.abs(np.diag(base)) + n)
    A = np.diag(scales) @ base @ np.diag(scales)

    import mpmath

    with mpmath.workdps(int(dps)):
        M = mpmath.matrix([[mpmath.mpf(float(A[i, j])) for j in range(n)] for i in range(n)])
        values, _ = mpmath.mp.eigsy(M)
        exact = np.sort(np.array([float(values[i]) for i in range(n)]))
    return {"A": A, "scales": scales, "base": base, "exact": exact}
