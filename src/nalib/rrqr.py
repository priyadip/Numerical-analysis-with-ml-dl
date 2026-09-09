"""Rank revealing QR, subset selection and total least squares.

What this module is for
-----------------------
Lesson 32 ended on an uncomfortable note: the SVD's ``rcond`` threshold decides the numerical
rank, that decision changes the answer by orders of magnitude, and QR has nothing to say about
it. This module supplies what QR can say.

Three separate ideas live here and they are easy to confuse:

**Column pivoted QR** factors ``A P = Q R`` with ``P`` a permutation chosen so that the
diagonal of ``R`` decreases. A sharp drop in that diagonal is evidence of a rank gap, and it
costs almost nothing on top of an ordinary QR. It is *evidence*, not proof, which is the whole
subject of `kahan_matrix`.

**Subset selection** uses the same permutation to answer a different question: which ``k`` of
the ``n`` columns should I keep? That is a question about the data, not about arithmetic, and
it is the reason pivoted QR survives in statistics.

**Total least squares** changes the problem rather than the algorithm. Ordinary least squares
assumes the error is all in ``b``. When ``A`` is measured too, the ordinary answer is
*biased*, systematically and predictably, and no amount of numerical care fixes it because the
arithmetic was never the problem.

Everything here derives its sizes from its input. Nothing assumes a shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import qr as _qr


# ----------------------------------------------------------------------------- pivoted QR


@dataclass
class PivotedQR:
    """The result of a column pivoted QR, ``A[:, piv] = Q @ R``.

    ``piv`` is the column order chosen, so ``A[:, piv]`` is the permuted matrix and
    ``np.argsort(piv)`` undoes it. ``rank`` is the numerical rank at the tolerance used.
    """

    Q: np.ndarray
    R: np.ndarray
    piv: np.ndarray
    rank: int
    tol: float
    diagonal: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def permutation_matrix(self) -> np.ndarray:
        """The explicit ``P`` with ``A @ P == A[:, piv]``. Built only for checking; the
        permutation itself is what the algorithms use."""
        n = self.piv.size
        P = np.zeros((n, n))
        P[self.piv, np.arange(n)] = 1.0
        return P


def qr_column_pivoted(A, tol: float | None = None) -> PivotedQR:
    """Householder QR with column pivoting: ``A[:, piv] = Q R`` with ``|r_11| >= |r_22| >= ...``.

    At step ``k`` the column with the largest remaining norm is moved into position ``k`` and
    then eliminated. So ``r_kk`` is the length of the part of that column not already explained
    by the previous ones, and the diagonal is non-increasing by construction.

    **Why the column norms are updated rather than recomputed.** Recomputing them costs
    ``O(m(n-k))`` per step and ``O(mn^2)`` overall, which would double the cost of the
    factorization. The standard downdate ``c_j^2 -= r_kj^2`` is ``O(n)`` per step, but it loses
    accuracy when the norm drops sharply, which is exactly the case this routine exists to
    detect. So the norms are recomputed from the trailing block whenever the downdated value
    has fallen below a safe fraction of its original, which is LINPACK's compromise and costs
    a recomputation only where it matters.

    ``tol`` is the relative threshold for the numerical rank; the default is the same
    ``max(m, n) * eps`` that `numpy.linalg.matrix_rank` uses, applied to ``|r_kk| / |r_11|``.

    Returns a `PivotedQR`. Any shape, tall or wide, full rank or not.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m)
    piv = np.arange(n)

    col_norm = np.linalg.norm(R, axis=0)          # current norms of the trailing parts
    original = col_norm.copy()                    # for deciding when to recompute
    steps = min(m, n)

    for k in range(steps):
        j = k + int(np.argmax(col_norm[k:]))
        if j != k:
            R[:, [k, j]] = R[:, [j, k]]
            piv[[k, j]] = piv[[j, k]]
            col_norm[[k, j]] = col_norm[[j, k]]
            original[[k, j]] = original[[j, k]]

        if col_norm[k] == 0.0:                    # nothing left anywhere: stop cleanly
            break

        v, _ = _qr.householder_vector(R[k:, k])
        R[k:, k:] = _qr.apply_householder(v, R[k:, k:])
        Q[:, k:] = _qr.apply_householder(v, Q[:, k:].T).T
        R[k + 1:, k] = 0.0

        # Downdate the trailing column norms, and recompute the ones that have collapsed.
        for j2 in range(k + 1, n):
            if col_norm[j2] == 0.0:
                continue
            ratio = abs(R[k, j2]) / col_norm[j2]
            shrunk = 1.0 - ratio ** 2
            if shrunk <= 0.0:
                col_norm[j2] = 0.0
                continue
            col_norm[j2] *= np.sqrt(shrunk)
            if col_norm[j2] < 1e-8 * original[j2]:            # the downdate is no longer safe
                col_norm[j2] = float(np.linalg.norm(R[k + 1:, j2]))
                original[j2] = col_norm[j2]

    diag = np.abs(np.diag(R))
    lead = float(diag[0]) if diag.size and diag[0] > 0 else 0.0
    if tol is None:
        tol = max(m, n) * float(np.finfo(float).eps)
    rank = int(np.sum(diag > tol * lead)) if lead > 0 else 0
    return PivotedQR(Q=Q[:, :steps], R=np.triu(R[:steps, :]), piv=piv, rank=rank,
                     tol=float(tol), diagonal=diag)


def rank_gap(values) -> dict:
    """Find the largest ratio between consecutive entries of a non-increasing sequence.

    Given the diagonal of a pivoted ``R``, or a list of singular values, this reports where the
    sequence drops hardest and by how much. A large, isolated drop is what "there is a rank
    here" means; the absence of one is what "there is no rank here" means, and lesson 32's
    exercise 4.3 showed those two cases behave completely differently.

    Returns ``{"index", "ratio", "before", "after"}`` where ``index`` is the number of entries
    *above* the gap, so it can be used directly as a rank.
    """
    v = np.abs(np.asarray(values, dtype=float).ravel())
    if v.size < 2:
        return {"index": int(v.size), "ratio": 1.0,
                "before": float(v[0]) if v.size else 0.0, "after": 0.0}
    lead = float(v[0])
    if lead <= 0.0:
        return {"index": 0, "ratio": 1.0, "before": 0.0, "after": 0.0}

    # Only ratios whose NUMERATOR still carries information are candidates. Below about
    # n * eps * v[0] the entries are roundoff, and a ratio between two roundoff values is not a
    # gap: it is noise divided by noise, and it can come out as anything at all, including
    # infinity when one of them is an exact zero. Measured on a rank one 20 by 20 matrix, the
    # unrestricted search put the "gap" at index 7 purely because that is where the first exact
    # zero happened to fall. Restricted, it puts it at 1, which is the rank.
    floor = v.size * float(np.finfo(float).eps) * lead
    usable = np.flatnonzero(v[:-1] > floor)
    if usable.size == 0:
        return {"index": 0, "ratio": 1.0, "before": lead, "after": float(v[1])}

    tiny = np.finfo(float).tiny
    ratios = v[usable] / np.maximum(v[usable + 1], tiny)
    pick = int(usable[int(np.argmax(ratios))])
    return {"index": pick + 1, "ratio": float(v[pick] / max(v[pick + 1], tiny)),
            "before": float(v[pick]), "after": float(v[pick + 1])}


def kahan_matrix(n: int, theta: float = 0.4) -> np.ndarray:
    """Kahan's matrix: upper triangular, already in pivoted order, and its rank is hidden.

    ``K = diag(1, s, s^2, ...) @ (I - c * strictly_upper_ones)`` with ``s = sin(theta)`` and
    ``c = cos(theta)``.

    Every column has the same norm, so column pivoting has **no reason to swap anything** and
    returns the matrix unchanged. The diagonal of ``R`` then decays smoothly like ``s^k`` and
    shows no gap at all, while the smallest singular value is far smaller than the smallest
    diagonal entry. So pivoted QR reports a matrix that is comfortably full rank and the SVD
    reports one that is nearly singular.

    This is the standard counterexample to "pivoted QR reveals the rank". It reveals it
    *usually*, and there is no theorem saying always. Lesson 33 measures the gap.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    c, s = float(np.cos(theta)), float(np.sin(theta))
    upper = np.triu(np.ones((n, n)), 1)
    return np.diag(s ** np.arange(n)) @ (np.eye(n) - c * upper)


def numerical_rank(A, tol: float | None = None, method: str = "svd") -> int:
    """The numerical rank of ``A``, by the SVD or by pivoted QR.

    The two disagree, and that disagreement is the point of lesson 33 rather than a defect.
    ``method="svd"`` counts singular values above ``tol * sigma_1`` and is the definition;
    ``method="qr"`` counts diagonal entries of a pivoted ``R`` above ``tol * |r_11|`` and is
    the cheap estimate.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if tol is None:
        tol = max(A.shape) * float(np.finfo(float).eps)
    if method == "svd":
        s = np.linalg.svd(A, compute_uv=False)
        return int(np.sum(s > tol * s[0])) if s.size and s[0] > 0 else 0
    if method == "qr":
        return qr_column_pivoted(A, tol=tol).rank
    raise ValueError(f"method must be 'svd' or 'qr', got {method!r}")


# ----------------------------------------------------------------------------- subset selection


def subset_selection(A, k: int) -> dict:
    """Choose ``k`` columns of ``A`` that span its range as well as possible.

    The greedy answer is the first ``k`` pivots of a column pivoted QR: at each step take the
    column furthest from the span of those already chosen. It is cheap and it comes with a
    bound, ``sigma_k(A[:, S]) >= sigma_k(A) / sqrt(k (n-k) + 1) * 2^-k`` in the worst case,
    which is weak; in practice it is close to optimal, and lesson 33 measures how close by
    checking every subset at small ``n``.

    Returns ``{"columns", "sigma_min", "sigma_min_full", "ratio", "residual"}`` where
    ``residual`` is the relative error of projecting all of ``A`` onto the chosen columns.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[1]
    k = int(k)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    cols = np.sort(qr_column_pivoted(A).piv[:k])
    sub = A[:, cols]
    s_sub = np.linalg.svd(sub, compute_uv=False)
    s_full = np.linalg.svd(A, compute_uv=False)
    proj = sub @ np.linalg.lstsq(sub, A, rcond=None)[0]
    denom = float(np.linalg.norm(A))
    return {"columns": cols,
            "sigma_min": float(s_sub[-1]),
            "sigma_min_full": float(s_full[k - 1]),
            "ratio": float(s_sub[-1] / s_full[k - 1]) if s_full[k - 1] > 0 else float("inf"),
            "residual": float(np.linalg.norm(A - proj) / denom) if denom > 0 else 0.0}


def best_subset(A, k: int) -> dict:
    """The same question answered by checking every subset. Exponential, so small ``n`` only.

    This exists to measure how good the greedy answer is, which is not something a bound can
    tell you. It refuses above a size where the enumeration would be silly.
    """
    from itertools import combinations

    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[1]
    k = int(k)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    from math import comb
    if comb(n, k) > 20000:
        raise ValueError(f"C({n},{k}) = {comb(n, k)} subsets is too many to enumerate")
    best, cols = -np.inf, None
    for c in combinations(range(n), k):
        s = np.linalg.svd(A[:, list(c)], compute_uv=False)[-1]
        if s > best:
            best, cols = float(s), list(c)
    return {"columns": np.array(cols), "sigma_min": best}


# ----------------------------------------------------------------------------- total least squares


@dataclass
class TLSResult:
    """The total least squares solution and what it needed to exist."""

    x: np.ndarray
    sigma_last: float
    sigma_n_of_A: float
    exists: bool
    correction: float
    residual_A: float
    residual_b: float
    gap: float = 0.0


def total_least_squares(A, b) -> TLSResult:
    """Minimise ``||[dA | db]||_F`` subject to ``(A + dA) x = b + db``.

    Ordinary least squares puts the whole correction in ``b`` and leaves ``A`` alone. Total
    least squares lets both move and minimises the total, which is the right question when both
    were measured.

    **The solution is the smallest right singular vector of the stacked matrix.** Write
    ``C = [A | b]``. The constraint says ``C @ [x; -1] = 0`` after correction, so the corrected
    ``C`` must be singular, and the nearest singular matrix in the Frobenius norm is the one
    with ``sigma_last`` set to zero (Eckart-Young, lesson 43). The null direction of that
    matrix is the last right singular vector ``v``, and scaling it so its last entry is ``-1``
    reads off ``x``.

    **It does not always exist.** If ``v[-1] == 0`` the last singular vector has no component
    along ``b``, so it cannot be scaled, and there is no TLS solution. The clean criterion is
    ``sigma_last(C) < sigma_min(A)``: strictly smaller, and the closer they are the worse
    conditioned the answer. This is checked and reported rather than hidden.

    Any shape with ``m >= n``. Returns a `TLSResult`.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    if b.size != m:
        raise ValueError(f"b has {b.size} entries, A has {m} rows")
    C = np.column_stack([A, b])
    U, s_c, Vt = np.linalg.svd(C, full_matrices=False)
    s_a = np.linalg.svd(A, compute_uv=False)
    sigma_last = float(s_c[-1]) if s_c.size else 0.0
    sigma_n = float(s_a[-1]) if s_a.size else 0.0
    # Interlacing gives sigma_last(C) <= sigma_n(A) always. The GAP between them is what
    # governs the conditioning: as it closes, the last singular vector stops being a
    # well-defined direction and the TLS answer stops being determined by the data.
    gap = (sigma_n - sigma_last) / sigma_n if sigma_n > 0 else 0.0

    v = Vt[-1]                                    # the smallest right singular vector of C
    tail = float(v[-1])
    # Golub and Van Loan's criterion: the TLS solution exists and is unique exactly when
    # sigma_n(A) > sigma_last(C), STRICTLY. Testing only whether the last entry of v is
    # nonzero is not enough: when b is orthogonal to range(A) that entry is zero in exact
    # arithmetic and about 1e-16 in floating point, which passes a naive test and then divides
    # by it, returning components of order 1e14. Both conditions are checked.
    floor = max(m, n + 1) * float(np.finfo(float).eps)
    exists = (gap > floor) and abs(tail) > floor * max(1.0, float(np.linalg.norm(v)))
    if not exists:
        return TLSResult(x=np.full(n, np.nan), sigma_last=sigma_last, sigma_n_of_A=sigma_n,
                         exists=False, correction=float("nan"),
                         residual_A=float("nan"), residual_b=float("nan"), gap=float(gap))

    x = -v[:n] / tail
    # The correction actually applied is the rank one piece Eckart-Young removes from C.
    dC = -sigma_last * np.outer(U[:, -1], v)
    return TLSResult(x=x, sigma_last=sigma_last, sigma_n_of_A=sigma_n, exists=True,
                     correction=float(np.linalg.norm(dC)),
                     residual_A=float(np.linalg.norm(dC[:, :n])),
                     residual_b=float(np.linalg.norm(dC[:, n])), gap=float(gap))


def tls_closed_form(A, b) -> np.ndarray:
    """``x = (A^T A - sigma_last^2 I)^{-1} A^T b``, the other standard way to write it.

    Useful only as a cross check: it shows that TLS is ordinary least squares with the
    *ridge removed* rather than added. Tikhonov adds ``+lambda^2 I`` to stabilise; TLS
    subtracts ``sigma_last^2 I`` to remove the bias, and that is why TLS is less stable than
    ordinary least squares rather than more. Lesson 33 measures both signs on one problem.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    if b.size != A.shape[0]:
        raise ValueError(f"b has {b.size} entries, A has {A.shape[0]} rows")
    s_c = np.linalg.svd(np.column_stack([A, b]), compute_uv=False)
    shift = float(s_c[-1]) ** 2
    G = A.T @ A - shift * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ b)


def errors_in_variables(n_points: int, n_par: int, noise_A: float, noise_b: float,
                        rng=None, coefficients=None) -> dict:
    """Generate data where BOTH sides carry noise, so the ordinary answer is biased.

    Returns the true coefficients, the noisy ``A`` and ``b``, and the noise-free versions, so
    an experiment can measure the bias rather than assert it. Sizes and noise levels are all
    parameters; nothing here is fixed.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_points, n_par = int(n_points), int(n_par)
    if n_points < n_par:
        raise ValueError(f"need at least {n_par} points, got {n_points}")
    if coefficients is None:
        coefficients = rng.standard_normal(n_par)
    coefficients = np.asarray(coefficients, dtype=float).ravel()
    if coefficients.size != n_par:
        raise ValueError(f"coefficients has {coefficients.size} entries, need {n_par}")
    A_true = rng.standard_normal((n_points, n_par))
    b_true = A_true @ coefficients
    return {"coefficients": coefficients,
            "A_true": A_true, "b_true": b_true,
            "A": A_true + noise_A * rng.standard_normal((n_points, n_par)),
            "b": b_true + noise_b * rng.standard_normal(n_points)}


def attenuation_factor(noise_A: float, signal_scale: float = 1.0) -> float:
    """The factor by which ordinary least squares shrinks a slope when ``A`` is noisy.

    For a single predictor with true variance ``s^2`` and independent measurement noise of
    variance ``e^2`` added to it, the ordinary estimate converges to ``beta * s^2/(s^2 + e^2)``
    rather than to ``beta``. That is *regression dilution*, and it is a bias, not a variance:
    it does not go away with more data.

    This returns ``s^2 / (s^2 + e^2)``, the predicted shrinkage, so an experiment can compare
    the measured ratio against it.
    """
    s2 = float(signal_scale) ** 2
    e2 = float(noise_A) ** 2
    if s2 + e2 <= 0.0:
        raise ValueError("signal and noise cannot both be zero")
    return s2 / (s2 + e2)


def tls_conditioning(A, b) -> dict:
    """How well determined the TLS answer is, from the two singular value sequences alone.

    The relevant quantity is **not** ``kappa(A)``. It is the gap between ``sigma_n(A)`` and
    ``sigma_{n+1}([A | b])``, because the TLS solution is read off the singular vector
    belonging to the smaller of the two, and a singular vector is only as well defined as the
    separation of its singular value from the rest of the spectrum.

    The standard bound is that the TLS condition number behaves like
    ``sigma_1(A) / (sigma_n(A) - sigma_{n+1}(C))``, which reduces to ``kappa(A)`` when the gap
    is a decent fraction of ``sigma_n`` and blows up as it closes. Lesson 33 measures both
    against the achieved sensitivity.

    Returns ``{"sigma_n_A", "sigma_last_C", "gap", "relative_gap", "kappa_A",
    "kappa_tls", "amplification"}``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    if b.size != A.shape[0]:
        raise ValueError(f"b has {b.size} entries, A has {A.shape[0]} rows")
    s_a = np.linalg.svd(A, compute_uv=False)
    s_c = np.linalg.svd(np.column_stack([A, b]), compute_uv=False)
    sn, sl, s1 = float(s_a[-1]), float(s_c[-1]), float(s_a[0])
    gap = sn - sl
    kappa_a = s1 / sn if sn > 0 else float("inf")
    kappa_tls = s1 / gap if gap > 0 else float("inf")
    return {"sigma_n_A": sn, "sigma_last_C": sl, "gap": gap,
            "relative_gap": gap / sn if sn > 0 else 0.0,
            "kappa_A": kappa_a, "kappa_tls": kappa_tls,
            "amplification": kappa_tls / kappa_a if np.isfinite(kappa_a) and kappa_a > 0
            else float("inf")}
