"""Classical iterative methods: Jacobi, Gauss-Seidel and SOR.

Part 3 factorized the matrix. That costs O(n^3) and, worse, it costs memory that a large sparse
problem does not have, because factorization creates fill (lesson 21).

Iterative methods never factorize. They start from a guess and improve it, using only
matrix-vector products, so they need exactly the storage the matrix already occupies. What you
give up is finality: a direct method stops with the answer, an iterative one stops when you
decide the answer is close enough.

**Every method here is a matrix splitting.** Write ``A = M - N`` with M easy to invert, then

    A x = b  becomes  M x = N x + b  becomes  x_{k+1} = M^-1 (N x_k + b).

Different choices of M give different methods:

    Jacobi        M = D              the diagonal
    Gauss-Seidel  M = D + L          diagonal plus strict lower triangle
    SOR           M = D/omega + L    a weighted blend, tuned by omega

The iteration matrix is ``G = M^-1 N``, and the whole convergence theory is one statement:
**the iteration converges from every starting point if and only if rho(G) < 1**, and the
asymptotic rate is rho(G) itself. That is lesson 10's scalar fixed point theorem with |g'(r)|
replaced by a spectral radius, exactly as lesson 14 did for nonlinear systems.

Used by lesson 23, and again in lesson 25 where the same splittings become preconditioners and
in lesson 28 where their behaviour on different frequencies is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class IterationResult:
    """The outcome of an iterative solve, with the whole history.

    Carrying the history matters more here than for a direct method: an iterative method's
    behaviour over time IS the subject, and a convergence rate cannot be measured from a final
    answer.
    """

    x: np.ndarray
    residual_norms: np.ndarray
    converged: bool
    n_iter: int
    message: str
    iterates: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def errors(self, exact) -> np.ndarray:
        """||x_k - exact|| at every step, when the answer is known."""
        exact = np.asarray(exact, dtype=float).ravel()
        if self.iterates.size == 0:
            return np.empty(0)
        return np.linalg.norm(self.iterates - exact, axis=1)

    def observed_rate(self, last: int = 10) -> float:
        """The measured contraction factor, fitted over the final steps.

        Fitted on a log scale rather than taken as a single ratio, because the early steps are
        transient (lesson 15 section 6) and a single ratio picks up whatever noise is in the
        last two values.
        """
        r = self.residual_norms
        good = r > 0
        r = r[good]
        if r.size < 3:
            return float("nan")
        tail = r[-min(last, r.size):]
        k = np.arange(tail.size)
        slope = np.polyfit(k, np.log(tail), 1)[0]
        return float(np.exp(slope))

    def __repr__(self) -> str:
        return (f"IterationResult(converged={self.converged}, n_iter={self.n_iter}, "
                f"residual={self.residual_norms[-1]:.3e})")


# ---------------------------------------------------------------- splittings


def split(A) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (D, L, U): diagonal, strict lower and strict upper parts, with A = D + L + U.

    Note the sign convention. Many texts write ``A = D - L - U`` with L and U defined as the
    negatives; this module uses the additive form because it reads directly off the matrix and
    removes a common source of sign errors.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    D = np.diag(np.diag(A))
    return D, np.tril(A, -1), np.triu(A, 1)


def iteration_matrix(A, method: str = "jacobi", omega: float = 1.0) -> np.ndarray:
    """The matrix G with x_{k+1} = G x_k + c, for the named splitting.

    Formed explicitly only so that its spectral radius can be measured. **Never form it in a
    real solve**: it is dense even when A is sparse, and computing it costs more than the
    iteration it describes.

    Convergence is decided entirely by ``rho(G)``, which is why lesson 23 computes this at
    small sizes and then predicts behaviour at large ones.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    D, L, U = split(A)
    method = method.lower()
    if method == "jacobi":
        M = D
        if omega != 1.0:
            # Damped (weighted) Jacobi is not a plain splitting: it takes the Jacobi step and
            # then moves only a fraction omega of the way. That gives
            #     G(omega) = (1 - omega) I + omega G_J,
            # so every Jacobi eigenvalue lam becomes 1 - omega(1 - lam). Damping is what turns
            # Jacobi into a usable multigrid smoother (lesson 28), because it pulls the
            # eigenvalues near -1 in towards zero at the cost of those near +1, which the
            # coarse grid handles instead.
            if not 0.0 < omega <= 1.0:
                raise ValueError("damped Jacobi needs 0 < omega <= 1")
            if np.any(np.diag(D) == 0.0):
                raise np.linalg.LinAlgError(
                    "a zero diagonal entry makes this splitting undefined")
            G_j = np.linalg.solve(D, D - A)
            return (1.0 - omega) * np.eye(A.shape[0]) + omega * G_j
    elif method in ("gauss-seidel", "seidel", "gs"):
        M = D + L
    elif method == "sor":
        if not 0.0 < omega < 2.0:
            raise ValueError("SOR needs 0 < omega < 2 to have any chance of converging")
        M = D / omega + L
    else:
        raise ValueError("method must be 'jacobi', 'gauss-seidel' or 'sor'")
    if np.any(np.diag(D) == 0.0):
        raise np.linalg.LinAlgError("a zero diagonal entry makes this splitting undefined")
    return np.linalg.solve(M, M - A)


def spectral_radius(G) -> float:
    """rho(G), the quantity that decides everything about a stationary iteration."""
    return float(np.max(np.abs(np.linalg.eigvals(np.atleast_2d(np.asarray(G, dtype=float))))))


def converges(A, method: str = "jacobi", omega: float = 1.0) -> bool:
    """Whether the named iteration converges on A, by the only criterion that decides it."""
    try:
        return spectral_radius(iteration_matrix(A, method, omega)) < 1.0
    except (np.linalg.LinAlgError, ValueError):
        return False


# ---------------------------------------------------------------- the methods


def _run(A, b, x0, update, tol, max_iter, keep_history):
    """Shared driver. `update` performs one sweep in place and returns the new iterate."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"need a square matrix, got {A.shape}")
    if b.size != n:
        raise ValueError(f"b has length {b.size}, expected {n}")

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    scale = float(np.linalg.norm(b)) or 1.0
    history = [x.copy()] if keep_history else []
    residuals = [float(np.linalg.norm(b - A @ x))]

    for k in range(max_iter):
        x = update(x)
        r = float(np.linalg.norm(b - A @ x))
        residuals.append(r)
        if keep_history:
            history.append(x.copy())
        if not np.all(np.isfinite(x)):
            return IterationResult(x, np.array(residuals), False, k + 1, "diverged",
                                   np.array(history) if keep_history else np.empty((0, 0)))
        if r <= tol * scale:
            return IterationResult(x, np.array(residuals), True, k + 1, "converged",
                                   np.array(history) if keep_history else np.empty((0, 0)))

    return IterationResult(x, np.array(residuals), False, max_iter, "iteration limit reached",
                           np.array(history) if keep_history else np.empty((0, 0)))


def jacobi(A, b, x0=None, tol=1e-10, max_iter=10000, keep_history=True) -> IterationResult:
    """Jacobi: update every component from the OLD iterate.

        x_i^(k+1) = (b_i - sum_{j != i} a_ij x_j^(k)) / a_ii

    Every component is computed from the previous sweep, so the order of the components does
    not matter and every one can be computed at the same time. That makes Jacobi **trivially
    parallel**, which is the one thing it has over Gauss-Seidel, and it is why Jacobi survives
    as a smoother inside multigrid (lesson 28) on parallel hardware.

    Converges when A is strictly diagonally dominant, and often when it is not. Cost per sweep
    is one matrix-vector product, O(nnz).
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    d = np.diag(A).copy()
    if np.any(d == 0.0):
        raise np.linalg.LinAlgError("Jacobi needs a nonzero diagonal")
    R = A - np.diag(d)
    bb = np.asarray(b, dtype=float).ravel()
    return _run(A, b, x0, lambda x: (bb - R @ x) / d, tol, max_iter, keep_history)


def damped_jacobi(A, b, omega=2.0 / 3.0, x0=None, tol=1e-10, max_iter=10000,
                  keep_history=True) -> IterationResult:
    """Jacobi, but move only a fraction ``omega`` of the way.

        x^(k+1) = (1 - omega) x^(k) + omega * (the Jacobi step)

    As a solver this is worse than plain Jacobi on the model problem, because omega < 1 slows
    the slowest mode down. As a **smoother** it is far better, and that is the only reason it
    exists. Plain Jacobi barely touches the modes near the top of the spectrum, since its
    eigenvalues cos(k pi h) approach -1 there. Damping maps every eigenvalue lam to
    1 - omega(1 - lam), which sends -1 to 1 - 2 omega, so omega = 2/3 puts the top of the
    spectrum at -1/3 instead of -1.

    Lesson 28 needs exactly that: the smoother must kill the modes a coarse grid cannot
    represent, and leave the smooth ones to the coarse grid.

    omega must satisfy 0 < omega <= 1. omega = 1 is plain Jacobi.
    """
    if not 0.0 < omega <= 1.0:
        raise ValueError(f"damped Jacobi needs 0 < omega <= 1, got {omega}")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    d = np.diag(A).copy()
    if np.any(d == 0.0):
        raise np.linalg.LinAlgError("Jacobi needs a nonzero diagonal")
    R = A - np.diag(d)
    bb = np.asarray(b, dtype=float).ravel()
    step = lambda x: (1.0 - omega) * x + omega * ((bb - R @ x) / d)
    return _run(A, b, x0, step, tol, max_iter, keep_history)


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=10000,
                 keep_history=True) -> IterationResult:
    """Gauss-Seidel: use each new value as soon as it is available.

        x_i^(k+1) = (b_i - sum_{j<i} a_ij x_j^(k+1) - sum_{j>i} a_ij x_j^(k)) / a_ii

    The only change from Jacobi is that the first sum uses the values already updated in this
    sweep. Fresher information costs nothing extra and typically **halves** the number of
    iterations, which for the second difference matrix means exactly a factor of two in the
    asymptotic rate (Theorem 23.4).

    The price is that the sweep is inherently sequential: component i needs component i-1. It
    also makes the method depend on the ORDERING of the unknowns, which Jacobi does not.

    Guaranteed to converge for symmetric positive definite A, and for strictly diagonally
    dominant A.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    d = np.diag(A).copy()
    if np.any(d == 0.0):
        raise np.linalg.LinAlgError("Gauss-Seidel needs a nonzero diagonal")
    bb = np.asarray(b, dtype=float).ravel()

    def sweep(x):
        x = x.copy()
        for i in range(n):
            x[i] = (bb[i] - A[i, :i] @ x[:i] - A[i, i + 1:] @ x[i + 1:]) / d[i]
        return x

    return _run(A, b, x0, sweep, tol, max_iter, keep_history)


def sor(A, b, omega=1.5, x0=None, tol=1e-10, max_iter=10000,
        keep_history=True) -> IterationResult:
    """Successive over-relaxation: take the Gauss-Seidel step, then go further.

        x_i^(k+1) = (1 - omega) x_i^(k) + omega * (the Gauss-Seidel value)

    ``omega = 1`` is exactly Gauss-Seidel. ``omega > 1`` **over**-relaxes, overshooting on
    purpose, and for the right omega this changes the asymptotic rate from ``1 - O(h^2)`` to
    ``1 - O(h)``, which is a change of ORDER rather than of constant. On a 100 by 100 grid that
    is the difference between 10000 iterations and 100.

    ``omega < 1`` under-relaxes, which is occasionally needed for problems where Gauss-Seidel
    itself diverges.

    > **Kahan's theorem.** SOR can converge only if 0 < omega < 2. Outside that range
    > rho(G) >= |omega - 1| >= 1 and the iteration cannot converge for any A.

    The optimal omega depends on the spectral radius of the Jacobi iteration matrix, which is
    normally unknown. `optimal_omega` gives it for the consistently ordered case, and lesson 23
    measures how sharp the optimum is.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    if not 0.0 < omega < 2.0:
        raise ValueError("SOR needs 0 < omega < 2 (Kahan's theorem)")
    d = np.diag(A).copy()
    if np.any(d == 0.0):
        raise np.linalg.LinAlgError("SOR needs a nonzero diagonal")
    bb = np.asarray(b, dtype=float).ravel()

    def sweep(x):
        x = x.copy()
        for i in range(n):
            gs = (bb[i] - A[i, :i] @ x[:i] - A[i, i + 1:] @ x[i + 1:]) / d[i]
            x[i] = (1.0 - omega) * x[i] + omega * gs
        return x

    return _run(A, b, x0, sweep, tol, max_iter, keep_history)


def optimal_omega(A) -> float:
    """The omega minimising rho(G_SOR), for a consistently ordered matrix.

        omega* = 2 / (1 + sqrt(1 - rho_Jacobi^2))

    Valid for **consistently ordered** matrices with property A, which includes the second
    difference operator and most matrices from a regular grid. For those,
    ``rho(G_SOR) = omega* - 1``.

    Note the practical difficulty: it needs ``rho_Jacobi``, which costs an eigenvalue
    computation. In production omega is estimated adaptively or simply guessed, and lesson 23
    measures how much a wrong guess costs.
    """
    rho_j = spectral_radius(iteration_matrix(A, "jacobi"))
    if rho_j >= 1.0:
        raise ValueError("Jacobi does not converge on this matrix, so omega* is undefined")
    return float(2.0 / (1.0 + np.sqrt(1.0 - rho_j**2)))


def iterations_needed(rho: float, tol: float = 1e-10) -> float:
    """How many iterations a contraction factor rho needs to reach tol.

    From ``rho^k <= tol``, so ``k >= log(tol)/log(rho)``. This is the formula that makes the
    difference between rates visible: rho = 0.99 needs 2290 iterations for 1e-10, and
    rho = 0.9 needs 219.

    Returns infinity when rho >= 1, since the iteration never gets there, and 0 when rho is
    exactly 0, since a nilpotent iteration matrix finishes at once. A negative rho is rejected:
    rho is the modulus of an eigenvalue and cannot be negative, so a negative argument means
    the caller has passed the wrong thing, and returning a plausible number would hide it.
    """
    rho = float(rho)
    tol = float(tol)
    if rho < 0.0:
        raise ValueError(f"rho is a spectral radius and cannot be negative, got {rho}")
    if not 0.0 < tol < 1.0:
        raise ValueError(f"tol must lie strictly between 0 and 1, got {tol}")
    if rho == 0.0:
        return 0.0
    if rho >= 1.0:
        return float("inf")
    return float(np.log(tol) / np.log(rho))
