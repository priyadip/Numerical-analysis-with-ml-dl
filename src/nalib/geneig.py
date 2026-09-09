"""The generalized eigenvalue problem ``A x = lambda B x``.

What this module is for
-----------------------
A vibrating structure does not satisfy ``K x = lambda x``. It satisfies

    K x = lambda M x,

with ``K`` the stiffness matrix and ``M`` the mass matrix, and ``M`` is not the identity because
the mass is not distributed uniformly. The same shape appears whenever a differential equation
is discretised with a finite element basis that is not orthonormal, which is most of them.

**The obvious reduction is wrong.** Multiplying by ``B^{-1}`` gives ``B^{-1} A x = lambda x``, a
standard problem, and it destroys the symmetry that lesson 38 showed was worth everything:
``B^{-1} A`` is not symmetric even when ``A`` and ``B`` both are. `naive_reduction` does it so
lesson 40 can measure what it costs.

**The right reduction keeps the symmetry.** With ``B = L L^T`` positive definite,

    A x = lambda B x   <=>   (L^{-1} A L^{-T}) y = lambda y,    y = L^T x,

and ``L^{-1} A L^{-T}`` is symmetric because ``A`` is. That is a **congruence**, not a
similarity, and Sylvester's law of inertia says a congruence preserves the signs of the
eigenvalues even though it does not preserve their values. Here it happens to preserve them
exactly, which is what makes the reduction useful.

**And when ``B`` is singular the whole framing changes.** There are then **infinite**
eigenvalues, the count is no longer ``n``, and no reduction to a standard problem exists. The
QZ algorithm computes a **generalized Schur form** of the pair without inverting anything.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PencilResult:
    """Eigenvalues of a matrix pencil, in a form that can express infinity.

    ``alpha`` and ``beta`` give the eigenvalues as the ratios ``alpha / beta``. That looks like
    a needless indirection until ``beta`` is zero, which is exactly the case a single number
    cannot express: the pencil has an **infinite** eigenvalue there, and reporting ``inf`` or
    ``nan`` loses the information that ``alpha`` carries.
    """

    alpha: np.ndarray
    beta: np.ndarray
    vectors: np.ndarray | None = None
    message: str = ""

    @property
    def values(self) -> np.ndarray:
        """``alpha / beta``, with infinities where ``beta`` vanishes."""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(self.beta != 0, self.alpha / np.where(self.beta != 0, self.beta, 1),
                           np.inf)
        return out

    @property
    def finite(self) -> np.ndarray:
        """Only the finite eigenvalues, sorted. The count can be fewer than ``n``."""
        v = self.values
        return np.sort_complex(v[np.isfinite(v)])

    @property
    def n_infinite(self) -> int:
        return int(np.sum(~np.isfinite(self.values)))


def match_spectra(computed, reference) -> float:
    """The largest distance between two spectra, matched greedily rather than by sorting.

    **Sorting is not a safe way to compare complex spectra**, and this is a real trap rather
    than a nicety. ``numpy.sort_complex`` orders by real part and then by imaginary part, so a
    conjugate pair whose real parts agree to the last bit can come out in either order. Two
    correct answers then appear to differ by the full imaginary spread: measured, a QZ result
    agreeing with the reference to ``9.5e-16`` was scored as differing by **1.39**.

    Greedy nearest matching has no such failure mode, and at these sizes its cost is nothing.
    """
    a = np.asarray(computed).ravel().astype(complex)
    b = np.asarray(reference).ravel().astype(complex)
    if a.size != b.size:
        raise ValueError(f"{a.size} computed against {b.size} reference values")
    remaining = list(range(b.size))
    worst = 0.0
    for value in a:
        gaps = [abs(value - b[j]) for j in remaining]
        k = int(np.argmin(gaps))
        worst = max(worst, float(gaps[k]))
        remaining.pop(k)
    return worst


def is_definite_pencil(A, B, tol: float = 1e-12) -> dict:
    """Is ``(A, B)`` a **symmetric definite** pencil: both symmetric, ``B`` positive definite?

    That is the case where everything is easy, and checking it is cheap. When it holds, the
    eigenvalues are real, the eigenvectors are ``B``-orthogonal, and the Cholesky reduction
    applies. When it fails, none of those are guaranteed and QZ is the answer.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape != B.shape:
        raise ValueError(f"A is {A.shape} and B is {B.shape}")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"a pencil needs square matrices, got {A.shape}")
    sym_a = float(np.linalg.norm(A - A.T)) <= tol * max(float(np.linalg.norm(A)), 1.0)
    sym_b = float(np.linalg.norm(B - B.T)) <= tol * max(float(np.linalg.norm(B)), 1.0)
    eigs_b = np.linalg.eigvalsh(0.5 * (B + B.T))
    return {"A_symmetric": sym_a, "B_symmetric": sym_b,
            "B_positive_definite": bool(sym_b and eigs_b.min() > 0.0),
            "smallest_B_eigenvalue": float(eigs_b.min()),
            "kappa_B": float(np.linalg.cond(B)),
            "definite_pencil": bool(sym_a and sym_b and eigs_b.min() > 0.0)}


def cholesky_reduction(A, B, compute_vectors: bool = True) -> dict:
    """Reduce a symmetric definite pencil to a **symmetric** standard problem.

    ``B = L L^T``, then ``C = L^{-1} A L^{-T}`` is symmetric with the same eigenvalues, and the
    eigenvectors map back by ``x = L^{-T} y``.

    **Nothing is inverted.** ``L^{-1} A L^{-T}`` is computed by two triangular solves, which is
    lesson 19's rule applied here: solving is backward stable and inverting is not.

    **The eigenvectors come out ``B``-orthogonal**, ``x_i^T B x_j = delta_ij``, which is the
    natural orthogonality for this problem and the one a vibration analysis wants: it is the
    statement that the modes are dynamically independent.

    Costs ``n^3/3`` for the Cholesky plus ``O(n^3)`` for the solves, then a standard symmetric
    eigenproblem. The whole thing is a constant factor more than the standard problem.
    """
    from scipy.linalg import cholesky, solve_triangular

    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    info = is_definite_pencil(A, B)
    if not info["definite_pencil"]:
        raise ValueError(
            "the Cholesky reduction needs a symmetric definite pencil: A symmetric "
            f"({info['A_symmetric']}), B symmetric ({info['B_symmetric']}), B positive definite "
            f"({info['B_positive_definite']}, smallest eigenvalue "
            f"{info['smallest_B_eigenvalue']:.3e}). Use QZ instead.")
    L = cholesky(B, lower=True)
    # C = L^-1 A L^-T, by solves rather than by forming an inverse.
    tmp = solve_triangular(L, A, lower=True)
    C = solve_triangular(L, tmp.T, lower=True).T
    C = 0.5 * (C + C.T)                      # exactly symmetric, up to the roundoff of the solve
    values, Y = np.linalg.eigh(C)
    X = solve_triangular(L.T, Y, lower=False) if compute_vectors else None
    return {"values": values, "vectors": X, "reduced": C, "L": L,
            "symmetry_error": float(np.linalg.norm(C - C.T))}


def naive_reduction(A, B) -> dict:
    """``B^{-1} A x = lambda x``: the obvious reduction, and the one to avoid.

    It is mathematically correct and it throws away everything symmetry was worth. ``B^{-1} A``
    is **not symmetric** even for a symmetric definite pencil, so lesson 35's condition number of
    1 is gone, the eigenvalues can come out complex, and the eigenvectors are no longer
    orthogonal in any inner product.

    Included so lesson 40 can measure the damage rather than assert it. The solve is used rather
    than an explicit inverse, so what is measured is the loss of **structure**, not the extra
    error of inverting.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape != B.shape:
        raise ValueError(f"A is {A.shape} and B is {B.shape}")
    C = np.linalg.solve(B, A)
    values, X = np.linalg.eig(C)
    return {"values": values, "vectors": X, "reduced": C,
            "symmetry_error": float(np.linalg.norm(C - C.T)),
            "max_imaginary": float(np.max(np.abs(values.imag))),
            "kappa_vectors": float(np.linalg.cond(X))}


def qz_eigenvalues(A, B) -> PencilResult:
    """The generalized Schur form: ``Q^H A Z = S``, ``Q^H B Z = T``, both upper triangular.

    **Two different unitary matrices**, one on each side, which is why it is called QZ rather
    than QR: a single similarity cannot triangularize both members of a pencil at once.

    The eigenvalues are the ratios ``S_ii / T_ii``, returned as ``alpha`` and ``beta`` separately
    so that ``T_ii = 0`` can be reported as an **infinite** eigenvalue rather than as a division
    error.

    **Nothing is inverted**, which is the whole point: QZ works when ``B`` is singular, when the
    pencil is not symmetric, and when it is not definite. It is the general answer, and it costs
    about 30 times a standard eigenvalue problem of the same size.
    """
    from scipy.linalg import qz

    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape != B.shape:
        raise ValueError(f"A is {A.shape} and B is {B.shape}")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"a pencil needs square matrices, got {A.shape}")
    S, T, Q, Z = qz(A, B, output="complex")
    return PencilResult(alpha=np.diag(S).copy(), beta=np.diag(T).copy(),
                        message=f"generalized Schur form, {A.shape[0]} by {A.shape[0]}")


def qz_residuals(A, B) -> dict:
    """Check the generalized Schur factorization: both triangular, both unitary, both exact."""
    from scipy.linalg import qz

    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    n = A.shape[0]
    S, T, Q, Z = qz(A, B, output="complex")
    scale_a = max(float(np.linalg.norm(A)), 1e-300)
    scale_b = max(float(np.linalg.norm(B)), 1e-300)
    return {"A_error": float(np.linalg.norm(A - Q @ S @ np.conj(Z).T) / scale_a),
            "B_error": float(np.linalg.norm(B - Q @ T @ np.conj(Z).T) / scale_b),
            "Q_unitary": float(np.linalg.norm(np.conj(Q).T @ Q - np.eye(n))),
            "Z_unitary": float(np.linalg.norm(np.conj(Z).T @ Z - np.eye(n))),
            "S_below": float(np.abs(np.tril(S, -1)).max()) if n > 1 else 0.0,
            "T_below": float(np.abs(np.tril(T, -1)).max()) if n > 1 else 0.0}


def b_orthogonality(B, X) -> float:
    """``||X^T B X - I||``: how far the eigenvectors are from ``B``-orthonormal.

    This is the right notion of orthogonality for a symmetric definite pencil, and the plain
    ``X^T X = I`` is not: the modes of a structure are independent with respect to its mass, not
    with respect to the coordinate system it happens to be written in.
    """
    B = np.atleast_2d(np.asarray(B, dtype=float))
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != B.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows, B is {B.shape[0]} by {B.shape[1]}")
    return float(np.linalg.norm(X.T @ B @ X - np.eye(X.shape[1])))


def pencil_residual(A, B, values, vectors) -> float:
    """The worst ``||A x - lambda B x|| / (||A|| + |lambda| ||B||)``.

    The denominator is what makes this a fair test across a spectrum spanning many decades:
    dividing by ``||A||`` alone would hold a large eigenvalue to an impossible standard.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    X = np.atleast_2d(np.asarray(vectors))
    lam = np.asarray(values).ravel()
    if lam.size != X.shape[1]:
        raise ValueError(f"{lam.size} eigenvalues for {X.shape[1]} vectors")
    na, nb = float(np.linalg.norm(A)), float(np.linalg.norm(B))
    worst = 0.0
    for j in range(lam.size):
        x = X[:, j]
        r = float(np.linalg.norm(A @ x - lam[j] * (B @ x)))
        worst = max(worst, r / (na + abs(lam[j]) * nb) / max(float(np.linalg.norm(x)), 1e-300))
    return worst


# ----------------------------------------------------------------------------- test problems


def vibrating_string(n: int, density=None) -> dict:
    """A mass-spring chain: ``K x = lambda M x`` with ``K`` tridiagonal and ``M`` diagonal.

    The physical origin of the generalized problem. ``K`` is the stiffness of identical springs
    and ``M`` holds the masses, which need not be equal. With uniform masses ``M`` is a multiple
    of the identity and the problem reduces to a standard one; with varying masses it does not,
    and that is the ordinary case.

    ``lambda`` is the squared angular frequency, and the eigenvectors are the mode shapes.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    K = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    if density is None:
        masses = np.ones(n)
    else:
        masses = np.asarray(density, dtype=float).ravel()
        if masses.size != n:
            raise ValueError(f"density has {masses.size} entries, need {n}")
        if np.any(masses <= 0):
            raise ValueError("every mass must be positive")
    return {"K": K, "M": np.diag(masses), "masses": masses}


def singular_pencil(n: int, n_infinite: int, seed: int = 1) -> dict:
    """A pencil with exactly ``n_infinite`` infinite eigenvalues, built so the count is known.

    Take ``B`` with a null space of that dimension. Every null vector of ``B`` that is not a
    null vector of ``A`` gives an infinite eigenvalue, because ``A x = lambda B x`` with
    ``B x = 0`` and ``A x != 0`` forces ``lambda`` to be unbounded.

    **This is what a standard eigenvalue problem cannot express**, and why QZ returns
    ``alpha`` and ``beta`` rather than a ratio.
    """
    n = int(n)
    n_infinite = int(n_infinite)
    if not 0 <= n_infinite < n:
        raise ValueError(f"n_infinite must be between 0 and {n - 1}, got {n_infinite}")
    gen = np.random.default_rng(int(seed))
    Q, _ = np.linalg.qr(gen.standard_normal((n, n)))
    b_spectrum = np.concatenate([np.zeros(n_infinite),
                                 np.arange(1.0, n - n_infinite + 1)])
    B = Q @ np.diag(b_spectrum) @ Q.T
    A = Q @ np.diag(np.arange(1.0, n + 1)) @ Q.T          # nonsingular on B's null space
    return {"A": 0.5 * (A + A.T), "B": 0.5 * (B + B.T), "expected_infinite": n_infinite}
