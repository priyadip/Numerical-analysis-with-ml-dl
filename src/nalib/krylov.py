"""Krylov subspace methods: conjugate gradient, Arnoldi, Lanczos, GMRES.

Lesson 23's methods take a step of a size fixed in advance. Krylov methods choose the step
**optimally** over a growing subspace, and that single change turns O(kappa) iterations into
O(sqrt(kappa)).

The subspace is

    K_m(A, r) = span{r, A r, A^2 r, ..., A^(m-1) r},

which is exactly what you can build from a starting vector using only matrix-vector products.
That is the point: **a Krylov method never needs an entry of A**, only the ability to apply it.
So it works when A is a function rather than an array, which is what `LinearOperator` below is
for, and it is why these methods scale to problems no factorization could touch.

Three facts organise everything here.

**The naive basis is useless.** The vectors ``A^k r`` all converge towards the dominant
eigenvector, so they become numerically parallel and the basis loses rank. Lesson 26 measures
this. Every practical method orthogonalizes as it goes, which is what Arnoldi and Lanczos do.

**Symmetry buys a short recurrence.** For symmetric A the Arnoldi Hessenberg matrix is
tridiagonal, so each new vector only needs orthogonalizing against the previous two. That is
Lanczos, and it is why CG stores three vectors instead of all of them.

**Optimality is measured in some norm, and which one matters.** CG minimises the error in the
A-norm; GMRES minimises the residual in the 2-norm; MINRES minimises the residual for symmetric
indefinite A. The choice is forced by what the matrix allows.

Used by lessons 24 to 27, and again in Part 6 where the same subspaces produce eigenvalues.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


class LinearOperator:
    """A matrix you can apply but need not store.

    Wraps either an array or a function ``v -> A v``. Every method in this module goes through
    this interface, so none of them can accidentally depend on A being an array.

    That is not a stylistic point. A discretised PDE operator, a Jacobian accessed by
    directional differences (lesson 14), or a convolution applied by FFT are all matrices that
    exist only as functions, and Krylov methods are the reason they can be solved at all.
    """

    def __init__(self, action, shape=None, dtype=float):
        if callable(action) and not isinstance(action, np.ndarray):
            if shape is None:
                raise ValueError("a callable operator needs an explicit shape")
            self._matvec = action
            self.shape = tuple(shape)
            self._array = None
        else:
            A = np.atleast_2d(np.asarray(action, dtype=dtype))
            self._array = A
            self.shape = A.shape
            self._matvec = lambda v: A @ v
        self.n_matvec = 0

    def __matmul__(self, v):
        return self.matvec(v)

    def matvec(self, v) -> np.ndarray:
        """Apply the operator, counting the call.

        The count is the honest cost measure for every method here: a Krylov iteration's price
        is the number of matrix-vector products, not the number of iterations, exactly as
        lesson 11 counted function evaluations rather than steps.
        """
        v = np.asarray(v, dtype=float).ravel()
        if v.size != self.shape[1]:
            raise ValueError(f"operator expects length {self.shape[1]}, got {v.size}")
        self.n_matvec += 1
        return np.asarray(self._matvec(v), dtype=float).ravel()

    def to_array(self) -> np.ndarray:
        """The dense matrix, when one exists. Raises for a purely functional operator."""
        if self._array is None:
            raise ValueError("this operator is defined by a function and has no array")
        return self._array

    def __repr__(self) -> str:
        kind = "array" if self._array is not None else "function"
        return f"LinearOperator({self.shape[0]}x{self.shape[1]}, {kind})"


def as_operator(A) -> LinearOperator:
    """Accept an array, a LinearOperator, or leave one alone."""
    return A if isinstance(A, LinearOperator) else LinearOperator(A)


@dataclass
class KrylovResult:
    """The outcome of a Krylov solve, with the history that makes it measurable."""

    x: np.ndarray
    residual_norms: np.ndarray
    converged: bool
    n_iter: int
    n_matvec: int
    message: str
    iterates: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def errors(self, exact) -> np.ndarray:
        exact = np.asarray(exact, dtype=float).ravel()
        if self.iterates.size == 0:
            return np.empty(0)
        return np.linalg.norm(self.iterates - exact, axis=1)

    def a_norm_errors(self, A, exact) -> np.ndarray:
        """||x_k - x*||_A at each step, the quantity CG actually minimises.

        Defined only for symmetric positive definite A, because otherwise the square root of
        ``e^T A e`` is not real and the "norm" is not a norm (lesson 20).
        """
        A = as_operator(A)
        exact = np.asarray(exact, dtype=float).ravel()
        out = []
        for v in self.iterates:
            e = v - exact
            out.append(float(np.sqrt(max(e @ A.matvec(e), 0.0))))
        return np.array(out)

    def __repr__(self) -> str:
        return (f"KrylovResult(converged={self.converged}, n_iter={self.n_iter}, "
                f"n_matvec={self.n_matvec}, residual={self.residual_norms[-1]:.3e})")


# ---------------------------------------------------------------- Krylov basis


def krylov_basis(A, v, m: int, orthonormalize: bool = False) -> np.ndarray:
    """The first m Krylov vectors, either naive powers or an orthonormal basis.

    With ``orthonormalize=False`` this builds ``[v, Av, A^2 v, ...]`` literally, which is the
    definition and a **numerically hopeless** way to represent the subspace: by the power method
    (lesson 36) every column tends towards the dominant eigenvector, so the columns become
    parallel and the matrix loses rank.

    Lesson 26 measures the collapse. The condition number of the naive basis grows
    exponentially, and past about m = 20 in double precision it has no usable rank at all.

    With ``orthonormalize=True`` the same subspace is represented by an orthonormal basis, which
    is what Arnoldi produces and what every practical method uses.
    """
    A = as_operator(A)
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    K = np.zeros((n, m))
    w = v.copy()
    for j in range(m):
        K[:, j] = w
        if j + 1 < m:
            w = A.matvec(w)
    if orthonormalize:
        Q, _ = np.linalg.qr(K)
        return Q
    return K


def arnoldi(A, v, m: int, reorthogonalize: bool = True) -> dict:
    """The Arnoldi iteration: an orthonormal basis for K_m(A, v), and the projection of A onto it.

    Produces Q with orthonormal columns and an upper Hessenberg H satisfying the **Arnoldi
    relation**

        A Q_m = Q_(m+1) H_m,

    where H_m is (m+1) by m. That identity is the whole content of the algorithm, and lesson 26
    verifies it numerically rather than quoting it.

    The square part ``H[:m, :m] = Q_m^T A Q_m`` is the orthogonal projection of A onto the
    subspace. Its eigenvalues are the **Ritz values**, which approximate the eigenvalues of A
    (lesson 39) and control how fast GMRES converges (lesson 27).

    ``reorthogonalize`` applies modified Gram-Schmidt twice. One pass loses orthogonality in
    finite precision (lesson 30's classical against modified comparison, one level up); twice is
    enough in practice and is what production codes do.

    Stops early when the residual ``h[j+1, j]`` is negligible, which means the subspace has
    become **invariant** and the Krylov space cannot grow further. That is not a failure: it
    means the exact solution lies in the space already.
    """
    A = as_operator(A)
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    beta = float(np.linalg.norm(v))
    if beta == 0.0:
        raise ValueError("the Arnoldi starting vector must be nonzero")

    Q = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    Q[:, 0] = v / beta
    breakdown = None

    for j in range(m):
        w = A.matvec(Q[:, j])
        for _ in range(2 if reorthogonalize else 1):
            for i in range(j + 1):
                h = Q[:, i] @ w
                H[i, j] += h
                w = w - h * Q[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] <= 1e-14 * beta:
            # Breakdown: the subspace is INVARIANT, so the Krylov space cannot grow.
            # Truncate to the j+1 vectors that are genuinely orthonormal. Keeping a
            # zero column would leave Q non-orthonormal and the relation becomes the
            # square one, A Q = Q H, which is the stronger statement anyway.
            breakdown = j + 1
            Q = Q[:, :j + 1]
            H = H[:j + 1, :j + 1]
            break
        Q[:, j + 1] = w / H[j + 1, j]

    return {"Q": Q, "H": H, "beta": beta, "breakdown": breakdown,
            "m": H.shape[1], "n_matvec": A.n_matvec}


def lanczos(A, v, m: int, reorthogonalize: bool = False) -> dict:
    """The Lanczos iteration: Arnoldi for symmetric A, with a three-term recurrence.

    When A is symmetric, ``H = Q^T A Q`` is symmetric as well as Hessenberg, hence
    **tridiagonal**. So a new vector only needs orthogonalizing against the previous **two**,
    and the cost per step drops from O(mn) to O(n).

    That single fact is why conjugate gradient stores three vectors regardless of how many
    iterations it takes, while GMRES (lesson 27) stores all of them.

    **The price is loss of orthogonality.** The short recurrence enforces orthogonality against
    two neighbours only, and in finite precision the vectors drift out of orthogonality with the
    earlier ones. That produces **ghost eigenvalues**: converged Ritz values that reappear as
    spurious copies. Lesson 26 measures it, and it is why ``reorthogonalize`` exists, at the
    cost of the storage the short recurrence was meant to save.

    Returns the diagonal ``alpha``, the off-diagonal ``beta``, and the basis Q.
    """
    A = as_operator(A)
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    b0 = float(np.linalg.norm(v))
    if b0 == 0.0:
        raise ValueError("the Lanczos starting vector must be nonzero")

    Q = np.zeros((n, m + 1))
    alpha = np.zeros(m)
    beta = np.zeros(m + 1)
    Q[:, 0] = v / b0
    breakdown = None

    for j in range(m):
        w = A.matvec(Q[:, j])
        alpha[j] = Q[:, j] @ w
        w = w - alpha[j] * Q[:, j] - (beta[j] * Q[:, j - 1] if j > 0 else 0.0)
        if reorthogonalize:                      # full reorthogonalization, O(jn) per step
            for i in range(j + 1):
                w = w - (Q[:, i] @ w) * Q[:, i]
        beta[j + 1] = np.linalg.norm(w)
        if beta[j + 1] <= 1e-14 * b0:
            # invariant subspace: keep only the vectors that are genuinely orthonormal
            breakdown = j + 1
            alpha, beta, Q = alpha[:j + 1], beta[:j + 1], Q[:, :j + 1]
            break
        Q[:, j + 1] = w / beta[j + 1]

    return {"alpha": alpha, "beta": beta[1:len(alpha)], "Q": Q,
            "breakdown": breakdown, "m": len(alpha), "n_matvec": A.n_matvec}


def tridiagonal_from_lanczos(alpha, beta) -> np.ndarray:
    """Assemble the symmetric tridiagonal T from the Lanczos coefficients."""
    alpha = np.asarray(alpha, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    T = np.diag(alpha)
    if alpha.size > 1 and beta.size:
        k = min(beta.size, alpha.size - 1)
        T += np.diag(beta[:k], 1) + np.diag(beta[:k], -1)
    return T


def ritz_values(A, v, m: int, symmetric: bool = True) -> np.ndarray:
    """Eigenvalues of the projected matrix: approximations to A's own eigenvalues.

    Ritz values converge to the **extreme** eigenvalues of A first, which is exactly why Krylov
    methods work: the outliers in the spectrum are captured in a few iterations, and those
    outliers are what slow a solver down. Lesson 26 measures the convergence and lesson 39
    develops it as an eigenvalue method in its own right.
    """
    if symmetric:
        out = lanczos(A, v, m, reorthogonalize=True)
        return np.sort(np.linalg.eigvalsh(
            tridiagonal_from_lanczos(out["alpha"], out["beta"])))
    out = arnoldi(A, v, m)
    k = out["H"].shape[1]
    return np.sort(np.linalg.eigvals(out["H"][:k, :k]))


# ---------------------------------------------------------------- solvers


def steepest_descent(A, b, x0=None, tol=1e-10, max_iter=10000,
                     keep_history=True) -> KrylovResult:
    """Steepest descent: at each step move along the residual by the optimal amount.

    For symmetric positive definite A, solving ``A x = b`` is the same as minimising

        phi(x) = (1/2) x^T A x - b^T x,

    whose gradient is ``A x - b = -r``. So the residual IS the steepest descent direction, and
    the optimal step along it is ``alpha = (r.r)/(r.A r)``, found by minimising a scalar
    quadratic exactly.

    This is already better than lesson 23's methods, which take a step of a size fixed in
    advance. It is still poor: consecutive directions are orthogonal, so the path **zigzags**
    across a narrow valley and the rate is ``((kappa-1)/(kappa+1))^2`` per step, which is
    ``1 - O(1/kappa)``.

    Included so lesson 24 can measure exactly what conjugate gradient improves on.
    """
    A = as_operator(A)
    b = np.asarray(b, dtype=float).ravel()
    x = np.zeros(b.size) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    r = b - A.matvec(x)
    scale = float(np.linalg.norm(b)) or 1.0
    residuals = [float(np.linalg.norm(r))]
    history = [x.copy()] if keep_history else []

    for k in range(max_iter):
        Ar = A.matvec(r)
        denom = float(r @ Ar)
        if denom <= 0.0:
            return KrylovResult(x, np.array(residuals), False, k, A.n_matvec,
                                "matrix is not positive definite",
                                np.array(history) if keep_history else np.empty((0, 0)))
        alpha = float(r @ r) / denom
        x = x + alpha * r
        r = r - alpha * Ar
        residuals.append(float(np.linalg.norm(r)))
        if keep_history:
            history.append(x.copy())
        if residuals[-1] <= tol * scale:
            return KrylovResult(x, np.array(residuals), True, k + 1, A.n_matvec,
                                "converged",
                                np.array(history) if keep_history else np.empty((0, 0)))

    return KrylovResult(x, np.array(residuals), False, max_iter, A.n_matvec,
                        "iteration limit reached",
                        np.array(history) if keep_history else np.empty((0, 0)))


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None, M=None,
                       keep_history=True) -> KrylovResult:
    """Conjugate gradient, optionally preconditioned.

    The fix for steepest descent's zigzag: instead of searching along the residual, search along
    a direction **A-conjugate** to every previous one, meaning ``p_i^T A p_j = 0`` for i != j.

    Two consequences follow, and both are remarkable:

    - **the minimisation over the whole subspace collapses to a one-dimensional search**,
      because conjugate directions do not interfere, so no previous progress is undone,
    - **only three vectors need storing**, because the conjugacy is enforced by a three-term
      recurrence. That is Lanczos underneath: CG and Lanczos are the same recurrence written
      differently.

    CG minimises ``||x - x*||_A`` over the Krylov space at every step, and the classical bound is

        ||e_k||_A <= 2 ((sqrt(kappa)-1)/(sqrt(kappa)+1))^k ||e_0||_A,

    so the iteration count is ``O(sqrt(kappa))`` against steepest descent's ``O(kappa)``. At
    kappa = 10^4 that is 100 against 10000.

    **Requires A symmetric positive definite.** Symmetry gives the short recurrence; positive
    definiteness makes ``||.||_A`` a norm so that "minimise" means something (lesson 20).

    ``M`` is a preconditioner, supplied as an operator or a function applying ``M^-1``. Lesson
    25 builds them.
    """
    A = as_operator(A)
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    max_iter = n if max_iter is None else max_iter
    apply_M = (lambda v: v) if M is None else (
        M.matvec if isinstance(M, LinearOperator) else M)

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    r = b - A.matvec(x)
    z = np.asarray(apply_M(r), dtype=float).ravel()
    p = z.copy()
    rz = float(r @ z)
    scale = float(np.linalg.norm(b)) or 1.0
    residuals = [float(np.linalg.norm(r))]
    history = [x.copy()] if keep_history else []

    if residuals[0] <= tol * scale:
        return KrylovResult(x, np.array(residuals), True, 0, A.n_matvec,
                            "initial guess already converged",
                            np.array(history) if keep_history else np.empty((0, 0)))

    for k in range(max_iter):
        Ap = A.matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0.0:
            return KrylovResult(x, np.array(residuals), False, k, A.n_matvec,
                                "matrix is not positive definite: p^T A p <= 0",
                                np.array(history) if keep_history else np.empty((0, 0)))
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        residuals.append(float(np.linalg.norm(r)))
        if keep_history:
            history.append(x.copy())
        if residuals[-1] <= tol * scale:
            return KrylovResult(x, np.array(residuals), True, k + 1, A.n_matvec,
                                "converged",
                                np.array(history) if keep_history else np.empty((0, 0)))
        z = np.asarray(apply_M(r), dtype=float).ravel()
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new

    return KrylovResult(x, np.array(residuals), False, max_iter, A.n_matvec,
                        "iteration limit reached",
                        np.array(history) if keep_history else np.empty((0, 0)))


def cg_convergence_bound(kappa: float, k) -> np.ndarray:
    """The classical CG bound, 2 ((sqrt(k)-1)/(sqrt(k)+1))^k, as a function of iteration.

    A **bound**, not a prediction. It depends only on kappa, so it cannot see the shape of the
    spectrum, and CG routinely does far better when the eigenvalues are clustered. Lesson 24
    measures the gap and lesson 25 exploits it: clustering, not kappa, is the real goal.
    """
    k = np.asarray(k, dtype=float)
    if kappa < 1.0:
        raise ValueError("a condition number is never below 1")
    ratio = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
    return 2.0 * ratio**k


def steepest_descent_bound(kappa: float, k) -> np.ndarray:
    """The steepest descent bound, ((kappa-1)/(kappa+1))^k, for comparison.

    Note kappa where CG has sqrt(kappa). That difference in the exponent is the entire content
    of lesson 24.
    """
    k = np.asarray(k, dtype=float)
    if kappa < 1.0:
        raise ValueError("a condition number is never below 1")
    return ((kappa - 1.0) / (kappa + 1.0)) ** k


# ---------------------------------------------------------------- preconditioners


def jacobi_preconditioner(A) -> Callable:
    """M = diag(A). The cheapest preconditioner there is.

    Applying ``M^-1`` is one division per entry, and it costs nothing to build. It helps
    exactly when the trouble is **bad scaling**: if the diagonal entries span many orders of
    magnitude, dividing by them removes that spread at once.

    It does nothing at all for a matrix whose diagonal is constant, which includes the second
    difference operator. Lesson 25 measures both cases, because "the cheapest preconditioner"
    and "a useless preconditioner" are the same object on different matrices.
    """
    A = as_operator(A)
    d = np.diag(A.to_array()).copy()
    if np.any(d == 0.0):
        raise np.linalg.LinAlgError("Jacobi preconditioning needs a nonzero diagonal")
    return lambda v: np.asarray(v, dtype=float).ravel() / d


def ssor_preconditioner(A, omega: float = 1.0) -> Callable:
    """Symmetric SOR: a forward sweep and a backward sweep, so the result is symmetric.

    Plain SOR is not symmetric, and a preconditioner for CG **must** be symmetric positive
    definite, or the A-norm minimisation CG performs is no longer over an inner product. SSOR
    fixes that by sweeping both ways:

        M = (D/omega + L) (omega/(2-omega)) D^-1 (D/omega + U).

    Costs two triangular solves per application, which is O(nnz), and needs no extra storage
    beyond A itself. That last property is why it survives: it is a preconditioner you get for
    free from a matrix you already have.
    """
    A = as_operator(A).to_array()
    if not 0.0 < omega < 2.0:
        raise ValueError("SSOR needs 0 < omega < 2")
    D = np.diag(np.diag(A))
    L, U = np.tril(A, -1), np.triu(A, 1)
    if np.any(np.diag(D) == 0.0):
        raise np.linalg.LinAlgError("SSOR needs a nonzero diagonal")
    left = D / omega + L
    right = D / omega + U
    scale = (2.0 - omega) / omega

    from .lu import back_substitution, forward_substitution

    def apply(v):
        y = forward_substitution(left, np.asarray(v, dtype=float).ravel())
        z = scale * (np.diag(D) * y)
        return back_substitution(right, z)

    return apply


def incomplete_cholesky(A, drop_tol: float = 0.0) -> Callable:
    """Incomplete Cholesky: factor A but discard fill outside A's own sparsity pattern.

    Lesson 21 measured fill as the central difficulty of sparse direct methods. Incomplete
    Cholesky sidesteps it by simply **refusing to create any**: wherever A has a structural
    zero, the factor is forced to zero too.

    The result is not a factorization of A. It is an exact factorization of a nearby matrix
    ``A + E``, and it is useful precisely because ``E`` is small enough that ``(L L^T)^-1 A``
    has a clustered spectrum.

    **It can fail on a positive definite matrix.** Dropping entries can drive a pivot
    non-positive even when A is definite, and there is no way to know in advance. Production
    codes add a diagonal shift and retry, which lesson 25 measures.

    IC(0), the zero fill version, is what ``drop_tol = 0`` gives.
    """
    A = as_operator(A).to_array()
    n = A.shape[0]
    pattern = np.abs(A) > drop_tol
    L = np.zeros((n, n))
    for j in range(n):
        pivot = A[j, j] - L[j, :j] @ L[j, :j]
        if pivot <= 0.0:
            raise np.linalg.LinAlgError(
                f"incomplete Cholesky broke down at index {j}: pivot {pivot:g}. "
                "a diagonal shift is the usual remedy"
            )
        L[j, j] = np.sqrt(pivot)
        for i in range(j + 1, n):
            if pattern[i, j]:                    # only where A itself is nonzero
                L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]

    from .lu import back_substitution, forward_substitution

    def apply(v):
        y = forward_substitution(L, np.asarray(v, dtype=float).ravel())
        return back_substitution(L.T, y)

    apply.factor = L
    return apply


def preconditioned_spectrum(A, apply_M) -> np.ndarray:
    """Eigenvalues of M^-1 A, the thing a preconditioner is actually trying to improve.

    Formed densely, so this is a measuring instrument for a lesson and not something to run on
    a real problem. What it makes visible is that the goal is **clustering**, not a smaller
    condition number: lesson 24 measured two matrices with identical kappa needing 200 and 14
    iterations.
    """
    A = as_operator(A).to_array()
    n = A.shape[0]
    MinvA = np.column_stack([apply_M(A[:, j]) for j in range(n)])
    return np.sort(np.real(np.linalg.eigvals(MinvA)))
