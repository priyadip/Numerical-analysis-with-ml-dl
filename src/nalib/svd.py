"""The singular value decomposition: what it is, and why everything else has been using it.

What this module is for
-----------------------
Parts 5 and 6 have both been drawing on the SVD without building it. Lesson 15 defined
``kappa_2 = sigma_1 / sigma_n`` without saying what a singular value is. Lesson 32 used the
pseudoinverse and the four condition numbers. Lesson 33 used Eckart-Young. This module pays
those debts.

**The geometric statement is the whole thing.** Every matrix maps the unit sphere to a
hyperellipse. The singular values are the lengths of its semi-axes, the left singular vectors
are their directions, and the right singular vectors are the preimages of those directions.
`hyperellipse` computes the picture, and `semi_axes_are_singular_values` checks the claim.

**Everything else follows from that.** The four fundamental subspaces read off ``U`` and ``V``
(`fundamental_subspaces`). The 2-norm and the condition number are ``sigma_1`` and
``sigma_1/sigma_n``. The pseudoinverse inverts what can be inverted and ignores the rest.

**Two facts distinguish it from the eigenvalue decomposition of Part 6.** It **always exists**,
for every matrix of every shape, where an eigendecomposition needs squareness and
diagonalizability. And its singular values are **perfectly conditioned**: Weyl's inequality
gives ``|sigma_i(A + E) - sigma_i(A)| <= ||E||_2`` for every matrix, not merely for symmetric
ones. That is a stronger statement than anything in lesson 35, and `weyl_singular` measures it.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SVDResult:
    """``A = U S V^T`` with the pieces and the checks that make it verifiable."""

    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray
    rank: int
    tol: float

    def reconstruct(self) -> np.ndarray:
        return (self.U * self.s) @ self.Vt

    def truncated(self, k: int) -> np.ndarray:
        """The best rank ``k`` approximation, which lesson 43 proves is best."""
        k = int(k)
        if not 0 <= k <= self.s.size:
            raise ValueError(f"k must be between 0 and {self.s.size}, got {k}")
        return (self.U[:, :k] * self.s[:k]) @ self.Vt[:k, :]


def svd(A, tol: float | None = None) -> SVDResult:
    """The thin SVD, with the numerical rank recorded alongside it.

    Uses LAPACK through numpy. Building one from scratch is lesson 42's subject, and doing it
    here would put the hardest part of Part 6 before the theory it needs.

    The rank is the count of singular values above ``tol * sigma_1``, with the same default
    ``max(m, n) * eps`` that `numpy.linalg.matrix_rank` uses. Lesson 32 measured that this
    threshold is a judgement rather than a fact, and it is recorded so the judgement is visible.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if tol is None:
        tol = max(A.shape) * float(np.finfo(float).eps)
    rank = int(np.sum(s > tol * s[0])) if s.size and s[0] > 0 else 0
    return SVDResult(U=U, s=s, Vt=Vt, rank=rank, tol=float(tol))


# ----------------------------------------------------------------------------- the geometry


def hyperellipse(A, n_points: int = 400) -> dict:
    """Map the unit circle (or sphere) through ``A`` and return the image with its axes.

    **The defining picture.** ``A`` takes the unit sphere in ``R^n`` to a hyperellipse in
    ``R^m``, and the SVD names its parts: ``sigma_i`` are the semi-axis lengths, ``u_i`` the
    axis directions, and ``v_i`` the points of the sphere that map to them.

    For a ``2 x 2`` or ``2 x n`` matrix the image can be drawn, which is why lesson 41 uses
    those sizes for the figure and general ones for the checks.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    out = svd(A)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_points))
    if n == 2:
        circle = np.vstack([np.cos(theta), np.sin(theta)])
    else:
        gen = np.random.default_rng(0)
        circle = gen.standard_normal((n, int(n_points)))
        circle /= np.linalg.norm(circle, axis=0)
    image = A @ circle
    return {"circle": circle, "image": image,
            "semi_axes": out.s, "axis_directions": out.U,
            "preimages": out.Vt.T,
            "longest": float(np.max(np.linalg.norm(image, axis=0))),
            "shortest": float(np.min(np.linalg.norm(image, axis=0)))}


def semi_axes_are_singular_values(A, n_samples: int = 200_000, rng=None) -> dict:
    """Check the geometric claim by brute force: sample the sphere and measure the image.

    The longest vector in the image is ``sigma_1``, always. **The shortest is ``sigma_n`` only
    when ``A`` is tall or square**, and this is a real distinction rather than a technicality:
    a wide matrix has a null space of dimension ``n - m``, those directions map to zero, and the
    minimum stretch over the sphere is therefore **0** rather than the smallest singular value.
    Geometrically the image of the sphere is then the *solid* ellipsoid rather than its surface.

    An earlier version asserted ``shortest >= sigma_min`` for every shape and failed on every
    wide matrix, by as much as 3.9.

    Both gaps are reported as non-negative when the sampling is honest, since a finite sample
    cannot beat the true optimum in either direction.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    gen = np.random.default_rng() if rng is None else rng
    X = gen.standard_normal((n, int(n_samples)))
    X /= np.linalg.norm(X, axis=0)
    lengths = np.linalg.norm(A @ X, axis=0)
    s = np.linalg.svd(A, compute_uv=False)
    min_theory = float(s[-1]) if m >= n else 0.0
    return {"sampled_longest": float(lengths.max()),
            "sampled_shortest": float(lengths.min()),
            "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
            "min_stretch_theory": min_theory,
            "has_null_space": bool(m < n),
            "max_gap": float(s[0] - lengths.max()),
            "min_gap": float(lengths.min() - min_theory)}


def stretch_along(A, direction) -> float:
    """``||A v|| / ||v||``: how far ``A`` stretches one direction.

    The maximum over all directions is ``sigma_1`` by definition of the 2-norm, and the minimum
    is ``sigma_n`` when ``A`` has full column rank. So the singular values are not an algebraic
    curiosity: they are the extreme stretches, and everything about conditioning follows.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    v = np.asarray(direction, dtype=float).ravel()
    if v.size != A.shape[1]:
        raise ValueError(f"the direction has {v.size} entries, A has {A.shape[1]} columns")
    nv = float(np.linalg.norm(v))
    if nv == 0.0:
        raise ValueError("the zero direction has no stretch")
    return float(np.linalg.norm(A @ v)) / nv


# ----------------------------------------------------------------------------- existence


def svd_residuals(A) -> dict:
    """Every claim the decomposition makes, checked at once.

    ``A = U S V^T``, ``U`` and ``V`` with orthonormal columns, ``s`` non-negative and
    decreasing. **All four hold for every matrix of every shape**, which is the part that
    distinguishes the SVD from the eigendecomposition of Part 6.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    out = svd(A)
    m, n = A.shape
    k = out.s.size
    scale = max(float(np.linalg.norm(A)), 1e-300)
    return {"reconstruction": float(np.linalg.norm(A - out.reconstruct()) / scale),
            "U_orthonormal": float(np.linalg.norm(out.U.T @ out.U - np.eye(k))),
            "V_orthonormal": float(np.linalg.norm(out.Vt @ out.Vt.T - np.eye(k))),
            "non_negative": bool(np.all(out.s >= 0.0)),
            "decreasing": bool(np.all(np.diff(out.s) <= 1e-12 * max(out.s[0], 1.0))),
            "shape": (m, n), "rank": out.rank}


def uniqueness_report(A, tol: float = 1e-8) -> dict:
    """What is unique about an SVD and what is not.

    **The singular values are always unique.** The singular *vectors* are unique only up to a
    sign when the corresponding singular value is simple, and not even that when it is repeated:
    a repeated singular value has a whole subspace of valid singular vectors, and any orthonormal
    basis of it will do.

    Checked by computing the decomposition twice, from ``A`` and from a copy with its columns
    reordered and put back, and comparing.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[1]
    first = svd(A)
    order = np.arange(n)[::-1]
    shuffled = svd(A[:, order])
    back = shuffled.Vt[:, np.argsort(order)]
    gaps = np.diff(first.s)
    repeated = int(np.sum(np.abs(gaps) <= tol * max(first.s[0], 1.0)))
    return {"values_agree": float(np.max(np.abs(first.s - shuffled.s))),
            "vectors_agree_up_to_sign": float(np.max(np.abs(np.abs(first.Vt)
                                                            - np.abs(back)))),
            "repeated_values": repeated,
            "singular_values": first.s}


def jordan_wielandt(A) -> np.ndarray:
    """The symmetric matrix ``[[0, A], [A^T, 0]]``, whose eigenvalues are ``+/- sigma_i``.

    **This is the bridge between Part 6 and the SVD**, and it is a better bridge than ``A^TA``.
    It is symmetric for any ``A``, its eigenvalues are the singular values with both signs plus
    ``|m - n|`` zeros, and crucially it does **not square the condition number**: forming
    ``A^TA`` loses half the digits of every small singular value, and this does not.

    Lesson 42 uses the same idea to compute the SVD without forming ``A^TA``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    out = np.zeros((m + n, m + n))
    out[:m, m:] = A
    out[m:, :m] = A.T
    return out


def singular_values_from_eigen(A, route: str = "jordan") -> np.ndarray:
    """Singular values via an eigenvalue problem, by either route, for comparison.

    ``route="gram"`` uses the eigenvalues of ``A^TA``, which is the obvious way and the one
    lesson 29 warned about: it squares the condition number, so a singular value at
    ``sqrt(u) sigma_1`` is lost entirely.

    ``route="jordan"`` uses `jordan_wielandt`, which does not.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    if route == "gram":
        w = np.linalg.eigvalsh(A.T @ A)
        return np.sort(np.sqrt(np.maximum(w, 0.0)))[::-1]
    if route == "jordan":
        # The spectrum is sigma_1..sigma_k, then |m-n| zeros, then -sigma_k..-sigma_1. So the
        # top k eigenvalues ARE the singular values. Taking the k largest by MODULUS instead
        # returns the plus and minus pair of the top k/2, which is wrong and looks plausible.
        w = np.linalg.eigvalsh(jordan_wielandt(A))
        return np.sort(w)[::-1][:min(m, n)]
    raise ValueError(f"route must be 'gram' or 'jordan', got {route!r}")


# ----------------------------------------------------------------------------- subspaces


def fundamental_subspaces(A, tol: float | None = None) -> dict:
    """Orthonormal bases for all four, read straight off the SVD.

    With rank ``r``:

    - ``range(A)``      = the first ``r`` columns of ``U``
    - ``null(A^T)``     = the rest of ``U`` (needs the FULL U, not the thin one)
    - ``range(A^T)``    = the first ``r`` columns of ``V``
    - ``null(A)``       = the rest of ``V``

    **The two pairs are orthogonal complements**, which is the fundamental theorem of linear
    algebra, and the SVD makes it a matter of reading columns rather than of proving anything.

    The full ``U`` and ``V`` are needed here, so this is the one place a non-thin SVD is used.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    if tol is None:
        tol = max(m, n) * float(np.finfo(float).eps)
    r = int(np.sum(s > tol * s[0])) if s.size and s[0] > 0 else 0
    return {"rank": r,
            "range_A": U[:, :r], "null_AT": U[:, r:],
            "range_AT": Vt[:r, :].T, "null_A": Vt[r:, :].T,
            "dimensions": {"range_A": r, "null_AT": m - r,
                           "range_AT": r, "null_A": n - r}}


def subspace_residuals(A, tol: float | None = None) -> dict:
    """Check every subspace claim: bases orthonormal, complements orthogonal, null spaces null.

    A basis of ``null(A)`` that ``A`` does not annihilate is not a basis of ``null(A)``, and
    that is the check that would catch a wrong rank.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    sub = fundamental_subspaces(A, tol=tol)
    scale = max(float(np.linalg.norm(A)), 1e-300)

    def orthonormal(M):
        return (0.0 if M.shape[1] == 0
                else float(np.linalg.norm(M.T @ M - np.eye(M.shape[1]))))

    def cross(M, N):
        return 0.0 if M.shape[1] == 0 or N.shape[1] == 0 else float(np.abs(M.T @ N).max())

    return {"rank": sub["rank"],
            "range_A_orthonormal": orthonormal(sub["range_A"]),
            "null_AT_orthonormal": orthonormal(sub["null_AT"]),
            "range_AT_orthonormal": orthonormal(sub["range_AT"]),
            "null_A_orthonormal": orthonormal(sub["null_A"]),
            "range_A_perp_null_AT": cross(sub["range_A"], sub["null_AT"]),
            "range_AT_perp_null_A": cross(sub["range_AT"], sub["null_A"]),
            "A_kills_null_A": (0.0 if sub["null_A"].shape[1] == 0
                               else float(np.linalg.norm(A @ sub["null_A"]) / scale)),
            "AT_kills_null_AT": (0.0 if sub["null_AT"].shape[1] == 0
                                 else float(np.linalg.norm(A.T @ sub["null_AT"]) / scale)),
            "rank_nullity": sub["dimensions"]["range_AT"] + sub["dimensions"]["null_A"]}


# ----------------------------------------------------------------------------- perturbation


def weyl_singular(A, E) -> dict:
    """Weyl for singular values: ``|sigma_i(A+E) - sigma_i(A)| <= ||E||_2``, for **any** matrix.

    **This is stronger than anything in lesson 35.** There, the same statement needed both
    matrices symmetric; here it needs nothing at all. Every singular value of every matrix has
    condition number 1 with respect to an absolute perturbation.

    That is the reason the SVD is the tool of choice for anything to do with rank or
    conditioning: the quantities it reports cannot be ill conditioned.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    E = np.atleast_2d(np.asarray(E, dtype=float))
    if E.shape != A.shape:
        raise ValueError(f"E is {E.shape}, A is {A.shape}")
    a = np.linalg.svd(A, compute_uv=False)
    b = np.linalg.svd(A + E, compute_uv=False)
    bound = float(np.linalg.norm(E, 2))
    moved = float(np.max(np.abs(a - b)))
    return {"bound": bound, "actual": moved,
            "ratio": moved / bound if bound > 0 else 0.0,
            "sigma": a, "perturbed": b}


def condition_from_singular_values(A) -> dict:
    """``kappa_2 = sigma_1 / sigma_n``, and the two norms it is built from.

    Lesson 15 defined this and could not say what a singular value was. The derivation is now
    one line each: ``||A||_2 = sigma_1`` because the 2-norm is the largest stretch, and
    ``||A^{-1}||_2 = 1/sigma_n`` because the inverse's largest stretch is along the direction
    ``A`` shrinks most.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    s = np.linalg.svd(A, compute_uv=False)
    smallest = float(s[-1])
    return {"norm_A": float(s[0]),
            "norm_A_inverse": 1.0 / smallest if smallest > 0 else float("inf"),
            "kappa_2": float(s[0] / smallest) if smallest > 0 else float("inf"),
            "singular_values": s,
            "agrees_with_library": float(abs(float(s[0]) - np.linalg.norm(A, 2)))}


def graded_matrix(m: int, n: int, kappa: float, rng=None) -> np.ndarray:
    """A matrix with a prescribed condition number and geometrically spaced singular values."""
    m, n = int(m), int(n)
    if m < 1 or n < 1:
        raise ValueError(f"the shape must be positive, got ({m}, {n})")
    k = min(m, n)
    gen = np.random.default_rng() if rng is None else rng
    U, _ = np.linalg.qr(gen.standard_normal((m, k)))
    V, _ = np.linalg.qr(gen.standard_normal((n, k)))
    s = np.geomspace(1.0, 1.0 / float(kappa), k) if k > 1 else np.array([1.0])
    return (U * s) @ V.T


def rank_deficient(m: int, n: int, rank: int, rng=None) -> np.ndarray:
    """An exactly rank deficient matrix, built as a product so the rank is known."""
    m, n, rank = int(m), int(n), int(rank)
    if not 0 <= rank <= min(m, n):
        raise ValueError(f"rank must be between 0 and {min(m, n)}, got {rank}")
    gen = np.random.default_rng() if rng is None else rng
    if rank == 0:
        return np.zeros((m, n))
    return gen.standard_normal((m, rank)) @ gen.standard_normal((rank, n))
