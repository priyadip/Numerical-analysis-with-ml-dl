"""QR factorization: Gram-Schmidt, Householder reflectors, and Givens rotations.

Lesson 29 ended with a problem: the normal equations square the condition number, and the fix
has to avoid forming ``A^T A`` at all. Writing ``A = QR`` with ``Q`` having orthonormal columns
does exactly that, because ``||b - Ax|| = ||Q^T b - Rx||`` and the whole computation stays at
the conditioning of ``A``.

There are three ways to get there and they are not equivalent.

**Gram-Schmidt** orthogonalizes the columns one at a time, which is *triangular
orthogonalization*: it applies a sequence of triangular operations to ``A`` until the result is
orthonormal. The classical version loses orthogonality at a rate proportional to
``kappa(A)^2``; the modified version, which differs by two lines, loses it at ``kappa(A)``.
Lesson 30 measures both.

**Householder reflections** do the opposite, *orthogonal triangularization*: they apply a
sequence of orthogonal operations to ``A`` until the result is triangular. Because every step
is exactly orthogonal to roundoff, the computed ``Q`` is orthogonal to machine precision
whatever ``A`` is. That is the method every library uses.

**Givens rotations** zero one entry at a time. More work for a dense matrix and far less for a
matrix that is already nearly triangular, which is why lesson 27's GMRES used them and lesson
39's QR algorithm will.

Every routine here takes its shape from its argument. ``A`` may be any ``m`` by ``n``, tall,
square or wide, and the reduced and full forms are both available.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- measuring orthogonality


def orthogonality_error(Q) -> float:
    """||Q^T Q - I||_2, the number every comparison in lesson 30 turns on.

    A factorization can reproduce ``A`` perfectly and still have a ``Q`` whose columns are not
    orthogonal, and that is exactly what classical Gram-Schmidt does. Checking ``||A - QR||``
    alone would miss it, so both are always reported together.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    return float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), 2))


def factorization_error(A, Q, R) -> float:
    """||A - QR|| relative to ||A||. Backward error: how close QR is to the matrix given.

    Both Gram-Schmidt variants keep this small even when ``Q`` has lost orthogonality
    completely, which is the point lesson 30 makes: the product is right and the factors are
    not.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    scale = float(np.linalg.norm(A, 2)) or 1.0
    return float(np.linalg.norm(A - np.asarray(Q) @ np.asarray(R), 2)) / scale


def _dependency_floor(A, factor: float = 1.0) -> float:
    """Below this, a Gram-Schmidt residual is noise rather than a direction.

    This is the standard numerical rank threshold, ``max(m, n) * u * ||A||``. It has to be
    relative: the residual of an *exactly* dependent column is about ``m u ||A||``, not zero,
    so an ``== 0.0`` test never fires and the routine returns a garbage column in silence.

    **It cannot separate "dependent" from "very ill conditioned", and that is not a defect.**
    A column whose residual is at the noise floor is indistinguishable from a dependent one at
    working precision, whichever it actually is. So this rejects matrices with
    ``kappa`` beyond about ``1 / (m u)``, around ``1e14`` at ``m = 50``, and that is the honest
    boundary rather than a conservative one. Lesson 33's column pivoting is the routine that
    reports the numerical rank instead of refusing.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    u = float(np.finfo(float).eps) / 2
    return factor * max(A.shape) * u * float(np.linalg.norm(A, 2))


# ---------------------------------------------------------------- Gram-Schmidt


def gram_schmidt_classical(A) -> tuple[np.ndarray, np.ndarray]:
    """Classical Gram-Schmidt. Correct mathematics, and unstable.

    Column ``j`` is orthogonalized against **the original** ``q_1 ... q_{j-1}`` in one go:

        v = a_j - sum_i (q_i . a_j) q_i

    Every projection coefficient is computed from ``a_j`` itself, so they are all computed
    before any of them is subtracted. When the columns are nearly dependent, ``v`` is a tiny
    difference of large vectors, the cancellation of lesson 05 destroys it, and the resulting
    ``q_j`` is not orthogonal to what came before.

    Measured loss of orthogonality: **proportional to kappa(A)^2**, so a matrix with
    ``kappa = 1e8`` produces a ``Q`` with no orthogonality at all.

    **The dependency test has to be relative.** An exactly dependent column leaves a residual
    of about ``u`` times the column norm, not zero, so an ``== 0.0`` test never fires and the
    routine returns a garbage ``q_j`` in silence. Measured before this was fixed: a column set
    to ``a_0 + 2 a_1`` gave a residual of ``1.3e-15`` and a ``Q`` with orthogonality error
    ``0.92``, reported as success.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    floor = _dependency_floor(A)
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]          # note: A[:, j], not the running v
            v = v - R[i, j] * Q[:, i]
        R[j, j] = float(np.linalg.norm(v))
        if R[j, j] <= floor:
            raise np.linalg.LinAlgError(
                f"column {j} is numerically dependent on the earlier ones: the residual is "
                f"{R[j, j]:.3e} against a floor of {floor:.3e}. Gram-Schmidt cannot "
                "normalize it, and lesson 33's column pivoting is the routine that can")
        Q[:, j] = v / R[j, j]
    return Q, R


def gram_schmidt_modified(A) -> tuple[np.ndarray, np.ndarray]:
    """Modified Gram-Schmidt. Two lines different, and a squared improvement in stability.

    The change is to subtract each projection **as it is computed**, so the next coefficient is
    taken against the partially orthogonalized vector rather than the original:

        for i < j:  R[i,j] = q_i . v ;  v = v - R[i,j] q_i

    In exact arithmetic this is the same algorithm. In floating point it is not: the later
    coefficients are computed against a vector that has already had the earlier components
    removed, so they are small and their errors are small with them.

    Measured loss of orthogonality: **proportional to kappa(A)**, not its square. That single
    power is the whole difference, and it is worth a factor of ``kappa``.

    Still not as good as Householder, which loses nothing at all.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.astype(float).copy()
    floor = _dependency_floor(A)
    for j in range(n):
        R[j, j] = float(np.linalg.norm(V[:, j]))
        if R[j, j] <= floor:
            raise np.linalg.LinAlgError(
                f"column {j} is numerically dependent on the earlier ones: the residual is "
                f"{R[j, j]:.3e} against a floor of {floor:.3e}")
        Q[:, j] = V[:, j] / R[j, j]
        for i in range(j + 1, n):                # subtract from the REMAINING columns, now
            R[j, i] = Q[:, j] @ V[:, i]
            V[:, i] = V[:, i] - R[j, i] * Q[:, j]
    return Q, R


def gram_schmidt_reorthogonalized(A, passes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Modified Gram-Schmidt with the orthogonalization repeated.

    "Twice is enough" (Kahan and Parlett): if the first pass reduces the component along the
    existing basis by any fixed factor, the second reduces it to roundoff. The result is as
    orthogonal as Householder at roughly twice the cost of one pass, and it is what
    `nalib.nonsymmetric.gmres` uses.

    A third pass is never needed. If two do not suffice, the new vector was essentially inside
    the existing span and the right response is to declare a rank deficiency, not to
    orthogonalize again.
    """
    if passes < 1:
        raise ValueError(f"need at least one pass, got {passes}")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    floor = _dependency_floor(A)
    for j in range(n):
        v = A[:, j].astype(float).copy()
        for _ in range(int(passes)):
            for i in range(j):
                c = Q[:, i] @ v
                R[i, j] += c
                v = v - c * Q[:, i]
        R[j, j] = float(np.linalg.norm(v))
        if R[j, j] <= floor:
            raise np.linalg.LinAlgError(
                f"column {j} is numerically dependent: residual {R[j, j]:.3e} "
                f"against a floor of {floor:.3e}")
        Q[:, j] = v / R[j, j]
    return Q, R


# ---------------------------------------------------------------- Householder


def householder_vector(x) -> tuple[np.ndarray, float]:
    """The reflector ``v`` and the resulting first entry, for ``P x = alpha e_1``.

    The reflector is ``P = I - 2 v v^T / (v^T v)`` with ``v = x - alpha e_1``, and there are two
    valid choices of sign for ``alpha``. **The choice matters.** Taking
    ``alpha = +||x||`` when ``x_1 > 0`` makes ``v_1 = x_1 - ||x||`` a difference of nearly equal
    numbers, which is lesson 05's catastrophic cancellation, and the computed reflector is
    inaccurate.

    Choosing ``alpha = -sign(x_1) ||x||`` makes ``v_1 = x_1 + sign(x_1)||x||``, a sum of
    like-signed numbers, with no cancellation at any input. Lesson 31 measures both.

    Returns ``(v, alpha)`` with ``v`` unnormalized; the caller divides by ``v^T v``.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return x.copy(), 0.0
    norm = float(np.linalg.norm(x))
    if norm == 0.0:
        return np.zeros_like(x), 0.0
    # sign(0) is taken as +1, so a zero leading entry still gets a well defined reflector
    alpha = -norm if x[0] >= 0.0 else norm
    v = x.copy()
    v[0] -= alpha
    return v, alpha


def apply_householder(v, B, from_left: bool = True) -> np.ndarray:
    """Apply ``P = I - 2 v v^T / (v^T v)`` to B, in O(mn) rather than O(m^2 n).

    **Never form P.** It is ``m`` by ``m`` and applying it as a matrix product costs
    ``O(m^2 n)``; applied as a rank-one update it costs ``O(mn)``, which is the whole reason
    Householder QR is affordable.
    """
    v = np.asarray(v, dtype=float).ravel()
    B = np.atleast_2d(np.asarray(B, dtype=float)).copy()
    vtv = float(v @ v)
    if vtv == 0.0:
        return B
    if from_left:
        return B - np.outer(v, (2.0 / vtv) * (v @ B))
    return B - np.outer((2.0 / vtv) * (B @ v), v)


def householder_qr(A, mode: str = "reduced") -> tuple[np.ndarray, np.ndarray]:
    """QR by Householder reflections: orthogonal triangularization.

    Instead of orthogonalizing the columns of ``A``, apply orthogonal transformations to ``A``
    until it becomes triangular:

        P_n ... P_2 P_1 A = R,   so   A = (P_1 ... P_n) R = Q R.

    Each ``P_k`` is a reflection that zeros everything below the diagonal in column ``k``.
    Because reflections are **exactly** orthogonal up to roundoff, so is their product, and the
    computed ``Q`` satisfies ``||Q^T Q - I|| = O(u)`` **whatever kappa(A) is**. Neither
    Gram-Schmidt variant can say that.

    ``mode`` is ``"reduced"`` for the ``m`` by ``n`` factor or ``"full"`` for the ``m`` by ``m``
    orthogonal matrix.

    **It is also cheaper than Gram-Schmidt**, which surprises people who expect stability to
    cost something. Householder is ``2mn^2 - 2n^3/3`` flops against Gram-Schmidt's ``2mn^2``,
    so at ``m = n`` it is ``4n^3/3`` against ``2n^3``, a third less. What does cost extra is
    **forming Q explicitly**, another ``2mn^2 - 2n^3/3``, and that is exactly why LAPACK stores
    the reflectors instead. See `householder_qr_compact`.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    if mode not in ("reduced", "full"):
        raise ValueError("mode must be 'reduced' or 'full'")
    R = A.astype(float).copy()
    reflectors = []
    for k in range(min(m - 1, n)):
        v, _ = householder_vector(R[k:, k])
        reflectors.append(v)
        R[k:, k:] = apply_householder(v, R[k:, k:])
        R[k + 1:, k] = 0.0                        # exactly zero, not nearly

    width = m if mode == "full" else min(m, n)
    Q = np.eye(m, width)
    for k in range(len(reflectors) - 1, -1, -1):  # apply the reflectors in reverse
        Q[k:, :] = apply_householder(reflectors[k], Q[k:, :])
    return Q, (R if mode == "full" else R[:min(m, n), :])


def householder_qr_compact(A) -> dict:
    """Householder QR storing ``Q`` as its reflectors rather than as a matrix.

    This is what LAPACK does, and the reason is storage: the reflectors fit in the part of
    ``A`` the zeros of ``R`` vacate, so the whole factorization occupies the space of ``A``
    itself and nothing more. Forming ``Q`` explicitly costs another ``O(m^2 n)`` flops and
    ``O(m^2)`` storage, and for a tall thin ``A`` it is far larger than the data.

    Returns the reflectors, the triangular factor, and an ``apply`` closure so ``Q x`` and
    ``Q^T x`` are available without ever building ``Q``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    R = A.astype(float).copy()
    reflectors = []
    for k in range(min(m - 1, n)):
        v, _ = householder_vector(R[k:, k])
        reflectors.append(v)
        R[k:, k:] = apply_householder(v, R[k:, k:])
        R[k + 1:, k] = 0.0

    def apply_q(y, transpose: bool = False):
        """Q y, or Q^T y, in O(mn) without forming Q."""
        y = np.asarray(y, dtype=float).copy()
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            squeeze = True
        else:
            squeeze = False
        order = range(len(reflectors)) if transpose else range(len(reflectors) - 1, -1, -1)
        for k in order:
            y[k:, :] = apply_householder(reflectors[k], y[k:, :])
        return y.ravel() if squeeze else y

    storage = sum(v.size for v in reflectors) + min(m, n) * n
    return {"reflectors": reflectors, "R": R[:min(m, n), :], "apply_q": apply_q,
            "storage_entries": storage, "dense_q_entries": m * m}


# ---------------------------------------------------------------- Givens


def givens_rotation(a: float, b: float) -> tuple[float, float]:
    """The (c, s) sending (a, b) to (r, 0), computed without overflow.

    The same routine as `nalib.nonsymmetric.givens`, repeated here so lesson 31 can be read
    without lesson 27. Scaling by the larger of the two before squaring is what keeps it
    working from 1e-300 to 1e300.
    """
    if b == 0.0:
        return 1.0, 0.0
    if a == 0.0:
        return 0.0, 1.0
    if abs(b) > abs(a):
        t = a / b
        s = 1.0 / np.hypot(1.0, t)
        return s * t, s
    t = b / a
    c = 1.0 / np.hypot(1.0, t)
    return c, c * t


def givens_qr(A) -> tuple[np.ndarray, np.ndarray]:
    """QR by Givens rotations, zeroing one subdiagonal entry at a time.

    Each rotation acts on two rows and zeros one entry, so a dense ``m`` by ``n`` matrix needs
    about ``mn - n^2/2`` of them and the whole thing costs ``3mn^2 - n^3`` flops, roughly 50
    percent more than Householder.

    **It wins when most entries are already zero.** A Hessenberg matrix has one subdiagonal, so
    it needs ``n-1`` rotations and ``O(n^2)`` flops where Householder would do ``O(n^3)``. That
    is why lesson 27's GMRES updates its least squares problem this way, and why lesson 39's QR
    algorithm reduces to Hessenberg form first.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m)
    for j in range(min(m - 1, n)):
        for i in range(m - 1, j, -1):             # zero from the bottom up
            if R[i, j] == 0.0:
                continue
            c, s = givens_rotation(R[j, j], R[i, j])
            G = np.array([[c, s], [-s, c]])
            R[[j, i], j:] = G @ R[[j, i], j:]
            Q[:, [j, i]] = Q[:, [j, i]] @ G.T
            R[i, j] = 0.0
    return Q[:, :min(m, n)], R[:min(m, n), :]


def givens_count(A, tol: float = 0.0) -> dict:
    """How many rotations a matrix actually needs, against how many a dense one would.

    The ratio is the whole argument for using Givens on a structured matrix, and it is a
    property of the sparsity pattern rather than of the values.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    needed = int(sum(1 for j in range(min(m - 1, n)) for i in range(j + 1, m)
                     if abs(A[i, j]) > tol))
    dense = int(sum(1 for j in range(min(m - 1, n)) for i in range(j + 1, m)))
    return {"needed": needed, "dense": dense,
            "ratio": needed / dense if dense else 1.0}


# ---------------------------------------------------------------- solving with QR


def qr_solve(A, b, method: str = "householder") -> np.ndarray:
    """Least squares through QR, by the named factorization.

    ``method`` is ``"householder"``, ``"givens"``, ``"mgs"`` or ``"cgs"``. All four are correct
    in exact arithmetic and lesson 31 measures how far apart they are in floating point.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    if b.size != m:
        raise ValueError(f"b has length {b.size}, but A has {m} rows")
    if m < n:
        raise ValueError(f"underdetermined: {m} rows, {n} columns")

    builders = {"householder": householder_qr, "givens": givens_qr,
                "mgs": gram_schmidt_modified, "cgs": gram_schmidt_classical}
    if method not in builders:
        raise ValueError("method must be 'householder', 'givens', 'mgs' or 'cgs'")
    Q, R = builders[method](A)

    from .lu import back_substitution

    diag = np.abs(np.diag(R))
    if np.any(diag < 1e-14 * max(1.0, float(diag.max()))):
        raise np.linalg.LinAlgError(
            "A is numerically rank deficient; QR without pivoting cannot solve this")
    return back_substitution(R[:n, :n], (Q.T @ b)[:n])


def flops_qr(m: int, n: int, method: str = "householder") -> int:
    """Leading-order flop counts, so the three methods can be compared before timing them."""
    m, n = int(m), int(n)
    if m < 1 or n < 1:
        raise ValueError(f"need m, n at least 1, got {m}, {n}")
    counts = {
        "cgs": 2 * m * n * n,
        "mgs": 2 * m * n * n,
        "householder": 2 * m * n * n - 2 * n ** 3 // 3,
        "givens": 3 * m * n * n - n ** 3,
    }
    if method not in counts:
        raise ValueError(f"unknown method {method!r}")
    return int(counts[method])
