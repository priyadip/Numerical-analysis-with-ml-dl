"""Krylov solvers for nonsymmetric systems: GMRES, MINRES, BiCG and BiCGSTAB.

Lesson 26 showed that symmetry is what gives Lanczos its three-term recurrence, and hence what
gives conjugate gradient its three vectors of storage. Drop symmetry and the Hessenberg matrix
stops being tridiagonal, so orthogonalizing a new Krylov vector needs **all** the old ones.

That leaves two escapes, and this module implements both.

**Keep the orthogonality and pay for it.** GMRES stores the whole basis and minimises the
residual over it. It is optimal, its residual never increases, and its cost per step grows
linearly with the step count. Restarting bounds the cost and destroys the optimality.

**Keep the short recurrence and give up the orthogonality.** BiCG runs two Krylov sequences,
one with A and one with A^T, and makes them **biorthogonal** rather than orthogonal. That
restores a three-term recurrence and fixed storage, at the price of a residual that can
oscillate wildly and a breakdown that has no counterpart in CG. BiCGSTAB smooths the
oscillation by combining each BiCG step with a local residual minimisation.

**MINRES** sits between: it needs symmetry but not definiteness, so it keeps the short
recurrence and gives up only the A-norm optimality that needs positive definiteness.

Everything here takes its dimensions from the arguments, and every routine accepts either a
matrix or a `nalib.krylov.LinearOperator`, so a matrix is never required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .krylov import LinearOperator, as_operator


# ---------------------------------------------------------------- result type


@dataclass
class SolveResult:
    """The outcome of a nonsymmetric solve, with enough history to plot convergence.

    ``residual_norms`` is the quantity a stopping rule reads, and for GMRES it is available
    **without forming the residual**, straight out of the least squares problem. That is why
    GMRES can report progress at no cost while BiCG has to compute it.
    """

    x: np.ndarray
    residual_norms: np.ndarray
    converged: bool
    n_iter: int
    n_matvec: int
    message: str
    breakdown: bool = False
    iterates: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def relative_residuals(self, b) -> np.ndarray:
        """The residual norms divided by ||b||, which is what a tolerance is compared against."""
        scale = float(np.linalg.norm(np.asarray(b, dtype=float).ravel())) or 1.0
        return self.residual_norms / scale


# ---------------------------------------------------------------- Givens rotations


def givens(a: float, b: float) -> tuple[float, float]:
    """The cosine and sine of a rotation sending (a, b) to (r, 0), computed without overflow.

    The obvious formula ``c = a/sqrt(a^2+b^2)`` overflows when a or b exceeds the square root
    of the largest double, about 1.3e154, and underflows to 0/0 below 1e-162. Scaling by the
    larger of the two first removes both failures, and costs one comparison.

    This matters here because GMRES applies rotations to a Hessenberg matrix whose entries can
    span many orders of magnitude when A is badly scaled.
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


def apply_givens(c: float, s: float, x: float, y: float) -> tuple[float, float]:
    """Apply the rotation [[c, s], [-s, c]] to the pair (x, y)."""
    return c * x + s * y, -s * x + c * y


# ---------------------------------------------------------------- GMRES


def gmres(A, b, x0=None, tol: float = 1e-10, restart: int | None = None,
          max_iter: int | None = None, M: Callable | None = None,
          keep_history: bool = False) -> SolveResult:
    """GMRES: minimise ||b - A x|| over the Krylov space, in the 2-norm.

    At step m the iterate is ``x_0 + Q_m y`` and the Arnoldi relation
    ``A Q_m = Q_{m+1} H_m`` turns the minimisation into a small least squares problem

        min over y of  || beta e_1 - H_m y ||,

    with ``H_m`` an (m+1) by m Hessenberg matrix. Solving it by Givens rotations costs O(m)
    per step, because each new column needs only one new rotation and the previous ones are
    reapplied. The residual norm falls out of the rotation as the last entry of the transformed
    right-hand side, so **the residual is known before x is ever formed**.

    Two properties follow directly and neither has a counterpart in BiCG:

    - the residual norm is **non-increasing**, since the space searched at step m+1 contains
      the space searched at step m,
    - GMRES is **optimal** among methods whose iterate lies in ``x_0 + K_m``, in the 2-norm of
      the residual.

    ``restart = k`` throws the basis away every k steps and begins again from the current
    iterate. That bounds storage at k vectors and work at O(k) per step, and it **destroys the
    optimality**: restarted GMRES can stagnate completely on a problem full GMRES solves.
    Lesson 27 measures the stagnation.

    ``M`` applies a left preconditioner, so the system solved is ``M^-1 A x = M^-1 b`` and the
    residual minimised is the **preconditioned** one, which is not the same as the true one.

    Sizes come from ``b`` and from the operator, so nothing is assumed about the dimension.
    """
    op = as_operator(A)
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    if op.shape[0] != n:
        raise ValueError(f"b has length {n}, but the operator is {op.shape[0]} by {op.shape[1]}")
    if restart is not None and restart < 1:
        raise ValueError(f"restart must be at least 1, got {restart}")

    inner = n if restart is None else min(restart, n)
    max_iter = (n if restart is None else 10 * n) if max_iter is None else max_iter

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    if x.size != n:
        raise ValueError(f"x0 has length {x.size}, expected {n}")

    prec = (lambda v: v) if M is None else M
    scale = float(np.linalg.norm(prec(b))) or 1.0

    residuals = [float(np.linalg.norm(prec(b - op.matvec(x))))]
    history = [x.copy()] if keep_history else []
    total = 0

    while total < max_iter:
        r = prec(b - op.matvec(x))
        beta = float(np.linalg.norm(r))
        if beta <= tol * scale:
            return SolveResult(x, np.array(residuals), True, total, op.n_matvec,
                               "converged", False,
                               np.array(history) if keep_history else np.empty((0, 0)))

        Q = np.zeros((n, inner + 1))
        H = np.zeros((inner + 1, inner))
        cs = np.zeros(inner)
        sn = np.zeros(inner)
        g = np.zeros(inner + 1)
        Q[:, 0] = r / beta
        g[0] = beta
        m_used = 0

        for j in range(inner):
            if total + j >= max_iter:
                break
            w = prec(op.matvec(Q[:, j]))
            for i in range(j + 1):                       # modified Gram-Schmidt
                H[i, j] = Q[:, i] @ w
                w = w - H[i, j] * Q[:, i]
            corr = Q[:, :j + 1].T @ w                    # one reorthogonalization pass
            w = w - Q[:, :j + 1] @ corr
            H[:j + 1, j] += corr
            H[j + 1, j] = float(np.linalg.norm(w))
            # Keep the subdiagonal entry: the rotation below zeroes it by construction, so
            # testing it afterwards would report a lucky breakdown at every single step.
            h_next = H[j + 1, j]

            for i in range(j):                           # replay the earlier rotations
                H[i, j], H[i + 1, j] = apply_givens(cs[i], sn[i], H[i, j], H[i + 1, j])
            cs[j], sn[j] = givens(H[j, j], H[j + 1, j])
            H[j, j], H[j + 1, j] = apply_givens(cs[j], sn[j], H[j, j], H[j + 1, j])
            g[j], g[j + 1] = apply_givens(cs[j], sn[j], g[j], 0.0)

            m_used = j + 1
            residuals.append(abs(float(g[j + 1])))       # free: no residual is formed
            if keep_history:
                y_now = _solve_upper(H[:m_used, :m_used], g[:m_used])
                history.append(x + Q[:, :m_used] @ y_now)

            if abs(g[j + 1]) <= tol * scale:
                break
            if h_next <= 1e-14 * max(1.0, abs(H[j, j])):
                break                                     # lucky breakdown: the space closed
            Q[:, j + 1] = w / h_next

        if m_used == 0:
            break
        y = _solve_upper(H[:m_used, :m_used], g[:m_used])
        x = x + Q[:, :m_used] @ y
        total += m_used

        if residuals[-1] <= tol * scale:
            return SolveResult(x, np.array(residuals), True, total, op.n_matvec,
                               "converged", False,
                               np.array(history) if keep_history else np.empty((0, 0)))
        if restart is None:
            break

    converged = residuals[-1] <= tol * scale
    return SolveResult(x, np.array(residuals), converged, total, op.n_matvec,
                       "converged" if converged else "iteration limit reached", False,
                       np.array(history) if keep_history else np.empty((0, 0)))


def _solve_upper(R, g) -> np.ndarray:
    """Back substitution on a small upper triangular system, with a rank check."""
    R = np.atleast_2d(np.asarray(R, dtype=float))
    g = np.asarray(g, dtype=float).ravel()
    m = g.size
    y = np.zeros(m)
    for i in range(m - 1, -1, -1):
        if R[i, i] == 0.0:
            raise np.linalg.LinAlgError(f"singular least squares triangle at index {i}")
        y[i] = (g[i] - R[i, i + 1:m] @ y[i + 1:m]) / R[i, i]
    return y


def gmres_residual_polynomial(A, b, m: int):
    """The coefficients of the GMRES residual polynomial after m steps, and its roots.

    GMRES's residual is ``p(A) r_0`` for the polynomial of degree m with ``p(0) = 1`` that
    minimises the norm. Its **roots** are where GMRES has placed its zeros, and comparing them
    with the eigenvalues of A shows what the method is actually doing.

    For a normal matrix the roots migrate to the eigenvalues, which is why clustering helps.
    For a strongly nonnormal one they need not go anywhere near them, which is lesson 27's
    central negative result.
    """
    op = as_operator(A)
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    m = min(m, n)
    K = np.zeros((n, m + 1))
    K[:, 0] = b
    for j in range(1, m + 1):
        K[:, j] = op.matvec(K[:, j - 1])
    # minimise ||K c|| subject to c[0] = 1, that is  K[:,0] + K[:,1:] d  minimised over d
    d, *_ = np.linalg.lstsq(K[:, 1:], -K[:, 0], rcond=None)
    coeffs = np.concatenate([[1.0], d])              # p(t) = 1 + d1 t + d2 t^2 + ...
    roots = np.roots(coeffs[::-1]) if m >= 1 else np.array([])
    return coeffs, roots


# ---------------------------------------------------------------- MINRES


def minres(A, b, x0=None, tol: float = 1e-10, max_iter: int | None = None,
           keep_history: bool = False) -> SolveResult:
    """MINRES: minimise the residual 2-norm for a SYMMETRIC, possibly indefinite, matrix.

    Symmetry gives the Lanczos three-term recurrence, so the Hessenberg matrix is tridiagonal
    and the least squares problem can be updated with a **fixed** amount of work and storage,
    exactly as in CG. What positive definiteness would have bought is the A-norm, which is not
    a norm for an indefinite matrix, so MINRES minimises the 2-norm of the residual instead.

    That makes it the right method whenever A is symmetric and CG's assumption fails: shifted
    systems ``A - sigma I``, saddle point problems, and KKT systems.

    Implemented here through the same Givens machinery as GMRES, restricted to a tridiagonal
    H, which keeps the code honest about where the saving comes from.
    """
    op = as_operator(A)
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    max_iter = n if max_iter is None else max_iter
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()

    r = b - op.matvec(x)
    beta = float(np.linalg.norm(r))
    scale = float(np.linalg.norm(b)) or 1.0
    residuals = [beta]
    history = [x.copy()] if keep_history else []
    if beta <= tol * scale:
        return SolveResult(x, np.array(residuals), True, 0, op.n_matvec, "converged", False,
                           np.array(history) if keep_history else np.empty((0, 0)))

    # Lanczos vectors, three at a time
    q_prev = np.zeros(n)
    q = r / beta
    # the search directions, also three at a time
    w_prev2 = np.zeros(n)
    w_prev = np.zeros(n)
    g = beta
    cs_prev, sn_prev = 1.0, 0.0
    cs_prev2, sn_prev2 = 1.0, 0.0
    beta_prev = 0.0

    for k in range(1, max_iter + 1):
        v = op.matvec(q)
        alpha = float(q @ v)
        v = v - alpha * q - beta_prev * q_prev
        beta_next = float(np.linalg.norm(v))

        # apply the two previous rotations to the new tridiagonal column
        d0 = cs_prev2 * sn_prev * beta_prev + (-sn_prev2) * 0.0
        e0 = sn_prev2 * beta_prev
        h1 = cs_prev * alpha - cs_prev2 * sn_prev * beta_prev
        h0 = sn_prev * alpha + cs_prev2 * cs_prev * beta_prev
        c, s = givens(h1, beta_next)
        h1, _ = apply_givens(c, s, h1, beta_next)
        if h1 == 0.0:
            break
        w = (q - h0 * w_prev - e0 * w_prev2) / h1
        x = x + (c * g) * w
        g = -s * g
        residuals.append(abs(g))
        if keep_history:
            history.append(x.copy())
        if abs(g) <= tol * scale:
            return SolveResult(x, np.array(residuals), True, k, op.n_matvec, "converged",
                               False, np.array(history) if keep_history else np.empty((0, 0)))
        if beta_next <= 1e-14 * max(1.0, abs(alpha)):
            break
        q_prev, q = q, v / beta_next
        w_prev2, w_prev = w_prev, w
        beta_prev = beta_next
        cs_prev2, sn_prev2 = cs_prev, sn_prev
        cs_prev, sn_prev = c, s

    converged = residuals[-1] <= tol * scale
    return SolveResult(x, np.array(residuals), converged, len(residuals) - 1, op.n_matvec,
                       "converged" if converged else "iteration limit reached", False,
                       np.array(history) if keep_history else np.empty((0, 0)))


# ---------------------------------------------------------------- BiCG and BiCGSTAB


def bicg(A, b, AT=None, x0=None, tol: float = 1e-10, max_iter: int | None = None,
         shadow=None, keep_history: bool = False) -> SolveResult:
    """BiCG: a short recurrence for nonsymmetric A, bought with A^T and with stability.

    Two Krylov sequences run at once, one in ``A`` and one in ``A^T``, and the two sets of
    vectors are made **biorthogonal**, ``r_i . s_j = 0`` for i != j, rather than orthogonal.
    Biorthogonality is enough to give a three-term recurrence, so storage is fixed and each
    step costs two matrix-vector products.

    What is lost is real. The residual norm is **not** monotone and can oscillate over orders
    of magnitude. And the method can **break down**: the recurrence divides by ``s . r`` and by
    ``s_p . A p``, either of which can vanish without the solution being found. That is a
    genuine failure with no counterpart in CG, where the corresponding quantity is a norm and
    therefore positive.

    ``AT`` supplies the transpose action when A is an operator. For a matrix it is inferred.

    ``shadow`` sets the second starting vector, which defaults to ``r_0``. Any vector not
    orthogonal to ``r_0`` works in principle, and the choice changes the whole run. Setting it
    orthogonal to ``r_0`` makes the method break down on the first step, which is how lesson 27
    exhibits a breakdown on demand rather than waiting for one to occur by chance.
    """
    op = as_operator(A)
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    max_iter = 4 * n if max_iter is None else max_iter
    if AT is None:
        dense = op.to_array()
        AT = LinearOperator(lambda v, m=dense.T: m @ v, shape=op.shape)
    else:
        AT = as_operator(AT)

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    r = b - op.matvec(x)
    if shadow is None:
        s = r.copy()                                 # the shadow residual
    else:
        s = np.asarray(shadow, dtype=float).ravel().copy()
        if s.size != n:
            raise ValueError(f"shadow has length {s.size}, expected {n}")
    p, q = r.copy(), s.copy()
    scale = float(np.linalg.norm(b)) or 1.0
    residuals = [float(np.linalg.norm(r))]
    history = [x.copy()] if keep_history else []
    rho = float(s @ r)

    # Breakdown must be tested RELATIVELY. An absolute floor like 1e-300 never fires: two
    # vectors that are numerically orthogonal give an inner product around u times the product
    # of their norms, which is around 1e-16, not 1e-300. With an absolute test the method
    # limps on dividing by noise and returns a confident wrong answer.
    eps = np.finfo(float).eps

    for k in range(1, max_iter + 1):
        if abs(rho) <= eps * float(np.linalg.norm(s)) * float(np.linalg.norm(r)):
            return SolveResult(x, np.array(residuals), False, k - 1, op.n_matvec,
                               "breakdown: the shadow residual became orthogonal", True,
                               np.array(history) if keep_history else np.empty((0, 0)))
        Ap = op.matvec(p)
        denom = float(q @ Ap)
        if abs(denom) <= eps * float(np.linalg.norm(q)) * float(np.linalg.norm(Ap)):
            return SolveResult(x, np.array(residuals), False, k - 1, op.n_matvec,
                               "breakdown: a zero denominator in the step length", True,
                               np.array(history) if keep_history else np.empty((0, 0)))
        alpha = rho / denom
        x = x + alpha * p
        r = r - alpha * Ap
        s = s - alpha * AT.matvec(q)
        residuals.append(float(np.linalg.norm(r)))
        if keep_history:
            history.append(x.copy())
        if residuals[-1] <= tol * scale:
            return SolveResult(x, np.array(residuals), True, k, op.n_matvec, "converged",
                               False, np.array(history) if keep_history else np.empty((0, 0)))
        rho_new = float(s @ r)
        beta = rho_new / rho
        p = r + beta * p
        q = s + beta * q
        rho = rho_new

    return SolveResult(x, np.array(residuals), False, max_iter, op.n_matvec,
                       "iteration limit reached", False,
                       np.array(history) if keep_history else np.empty((0, 0)))


def bicgstab(A, b, x0=None, tol: float = 1e-10, max_iter: int | None = None,
             M: Callable | None = None, keep_history: bool = False) -> SolveResult:
    """BiCGSTAB: BiCG with a local residual minimisation after each step.

    Each iteration takes a BiCG step and then a one-dimensional GMRES step, choosing ``omega``
    to minimise the residual along the direction just produced. That smooths BiCG's oscillation
    while keeping the fixed storage, and it removes the need for ``A^T`` entirely, which matters
    when A is available only as a routine.

    It is not optimal and its residual is still not guaranteed monotone. It can still break
    down, now in two places: ``rho`` going to zero as in BiCG, and ``omega`` going to zero,
    which stalls the recurrence.

    Two matrix-vector products per iteration, the same as BiCG, and about the same storage.
    """
    op = as_operator(A)
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    max_iter = 4 * n if max_iter is None else max_iter
    prec = (lambda v: v) if M is None else M

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    r = b - op.matvec(x)
    r0 = r.copy()
    scale = float(np.linalg.norm(b)) or 1.0
    residuals = [float(np.linalg.norm(r))]
    history = [x.copy()] if keep_history else []
    rho = alpha = omega = 1.0
    v = np.zeros(n)
    p = np.zeros(n)

    for k in range(1, max_iter + 1):
        rho_new = float(r0 @ r)
        eps = np.finfo(float).eps
        rho_floor = eps * float(np.linalg.norm(r0)) * float(np.linalg.norm(r))
        if abs(rho_new) <= rho_floor or abs(omega) <= eps:
            return SolveResult(x, np.array(residuals), False, k - 1, op.n_matvec,
                               "breakdown: rho or omega vanished", True,
                               np.array(history) if keep_history else np.empty((0, 0)))
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        ph = prec(p)
        v = op.matvec(ph)
        denom = float(r0 @ v)
        if abs(denom) <= eps * float(np.linalg.norm(r0)) * float(np.linalg.norm(v)):
            return SolveResult(x, np.array(residuals), False, k - 1, op.n_matvec,
                               "breakdown: a zero denominator in the step length", True,
                               np.array(history) if keep_history else np.empty((0, 0)))
        alpha = rho_new / denom
        s = r - alpha * v
        if np.linalg.norm(s) <= tol * scale:                 # the BiCG half already did it
            x = x + alpha * ph
            residuals.append(float(np.linalg.norm(b - op.matvec(x))))
            if keep_history:
                history.append(x.copy())
            return SolveResult(x, np.array(residuals), True, k, op.n_matvec, "converged",
                               False, np.array(history) if keep_history else np.empty((0, 0)))
        sh = prec(s)
        t = op.matvec(sh)
        tt = float(t @ t)
        omega = 0.0 if tt == 0.0 else float(t @ s) / tt      # the local minimisation
        x = x + alpha * ph + omega * sh
        r = s - omega * t
        residuals.append(float(np.linalg.norm(r)))
        if keep_history:
            history.append(x.copy())
        if residuals[-1] <= tol * scale:
            return SolveResult(x, np.array(residuals), True, k, op.n_matvec, "converged",
                               False, np.array(history) if keep_history else np.empty((0, 0)))
        rho = rho_new

    return SolveResult(x, np.array(residuals), False, max_iter, op.n_matvec,
                       "iteration limit reached", False,
                       np.array(history) if keep_history else np.empty((0, 0)))


# ---------------------------------------------------------------- diagnostics


def convection_diffusion(m: int, pe: float = 0.0) -> np.ndarray:
    """The one dimensional convection diffusion operator on m interior points.

        -u'' + pe * u'      discretised with central differences

    ``pe`` is the mesh Peclet number. At ``pe = 0`` the matrix is the symmetric second
    difference operator; as ``pe`` grows the convection term makes it increasingly nonsymmetric
    and, beyond ``pe = 2``, increasingly **nonnormal**. That single parameter therefore sweeps
    the whole difficulty of this lesson, at any size m.
    """
    if m < 1:
        raise ValueError(f"need at least one interior point, got {m}")
    main = np.full(m, 2.0)
    lower = np.full(max(m - 1, 0), -1.0 - pe / 2.0)
    upper = np.full(max(m - 1, 0), -1.0 + pe / 2.0)
    return np.diag(main) + np.diag(lower, -1) + np.diag(upper, 1)


def departure_from_normality(A) -> float:
    """||A A^T - A^T A||_F / ||A||_F^2, zero exactly when A is normal.

    A normal matrix has orthogonal eigenvectors, so its eigenvalues describe its behaviour
    completely. The further this number is from zero, the less the eigenvalues tell you.

    **Read it with care.** Dividing by ``||A||_F^2`` makes the measure scale invariant, which is
    what you want, but it also means a family whose entries grow can look **more** normal while
    becoming less so. On the convection diffusion operator the raw commutator grows linearly in
    the mesh Peclet number while ``||A||_F^2`` grows quadratically, so this ratio peaks near
    pe = 3 and then falls, while `eigenvector_conditioning` is still reporting the truth.
    Lesson 27 measures both side by side, and the scale invariant one is the misleading one.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    denom = float(np.linalg.norm(A, "fro")) ** 2
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(A @ A.T - A.T @ A, "fro")) / denom


def eigenvector_conditioning(A) -> float:
    """kappa(V), the condition number of the eigenvector matrix. Exactly 1 when A is normal.

    The sharper companion to `departure_from_normality`, and the one that tells the truth on
    the convection diffusion family. The Bauer-Fike theorem prices an eigenvalue perturbation
    at kappa(V) times the size of the perturbation, so kappa(V) is precisely the factor by
    which the eigenvalues can mislead.

    Returns a huge number, or infinity, for a defective matrix. The convection diffusion
    operator becomes defective at a mesh Peclet number of exactly 2: the superdiagonal
    ``-1 + pe/2`` vanishes there, leaving a lower bidiagonal matrix with a constant diagonal,
    which is a single Jordan block with no eigenvector basis at all.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    try:
        _, V = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.linalg.cond(V))


def field_of_values_bounds(A, n_angles: int = 180) -> dict:
    """The extreme points of the field of values, and whether it contains the origin.

    The field of values (numerical range) is
    ``W(A) = { x^* A x / x^* x : x nonzero }``, and it is computed by maximising the real part
    of ``e^{-i theta} A`` over theta, which is an eigenvalue problem for the Hermitian part.

    It is the substitute for the spectrum when A is nonnormal. Elman's bound says that if the
    origin lies **outside** W(A), GMRES converges at a rate set by the distance, and unlike the
    eigenvalues this bound cannot be defeated by a nonnormal example.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    thetas = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    boundary = np.empty(thetas.size, dtype=complex)
    support = np.empty(thetas.size)
    for k, th in enumerate(thetas):
        B = np.exp(-1j * th) * A
        H = (B + B.conj().T) / 2.0
        lam, vec = np.linalg.eigh(H)
        v = vec[:, -1]
        boundary[k] = complex(v.conj() @ (A @ v))
        support[k] = float(lam[-1])            # the support function in direction theta

    # W(A) lies inside every half plane Re(e^{-i th} z) <= support(th). The origin satisfies
    # 0 <= support(th) for all th exactly when it is inside, so a single negative support
    # value is a separating line and proves the origin is outside.
    worst = float(np.min(support))
    return {"boundary": boundary,
            "thetas": thetas,
            "support": support,
            "contains_origin": bool(worst >= 0.0),
            "distance_to_origin": float(max(0.0, -worst)),
            "real_min": float(boundary.real.min()),
            "real_max": float(boundary.real.max())}


def stagnation_matrix(n: int) -> np.ndarray:
    """A matrix on which GMRES makes NO progress until the very last step.

    The cyclic shift: ones on the subdiagonal and a single one in the top right corner. With
    ``b = e_1`` the Krylov space never contains a better approximation than zero until step n,
    at which point the answer appears at once. Its eigenvalues are the n-th roots of unity,
    evenly spread on the unit circle, so the spectrum looks harmless.

    This is the standard counterexample to any claim that eigenvalues govern GMRES.
    """
    if n < 2:
        raise ValueError(f"the cyclic shift needs n at least 2, got {n}")
    A = np.zeros((n, n))
    A[1:, :-1] = np.eye(n - 1)
    A[0, -1] = 1.0
    return A


def same_spectrum_family(eigenvalues, gamma: float) -> np.ndarray:
    """A bidiagonal matrix with the given eigenvalues and a superdiagonal of size ``gamma``.

    Every member of the family has **exactly** the prescribed eigenvalues, because a triangular
    matrix's eigenvalues are its diagonal. What changes with ``gamma`` is the non-normality, and
    with it the GMRES convergence curve, which can be moved from immediate to complete
    stagnation while the spectrum never moves at all.

    This is the constructive form of the Greenbaum, Ptak and Strakos result: eigenvalues do not
    determine GMRES convergence for a nonsymmetric matrix. Their theorem is stronger, saying any
    non-increasing residual curve is achievable with any prescribed eigenvalues, but this family
    already settles the question and, unlike the general construction, every claim about it can
    be verified directly from the matrix.

    The size comes from ``eigenvalues``, so any spectrum at any dimension works.
    """
    lam = np.asarray(eigenvalues, dtype=float).ravel()
    n = lam.size
    if n < 1:
        raise ValueError("need at least one eigenvalue")
    A = np.diag(lam)
    if n > 1:
        A += np.diag(np.full(n - 1, float(gamma)), 1)
    return A
