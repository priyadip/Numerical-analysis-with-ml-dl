"""Krylov methods for eigenvalues: Rayleigh-Ritz, restarting, and matrices too big to store.

What this module is for
-----------------------
Lessons 37 and 38 reduce the matrix: Hessenberg, then tridiagonal, then iterate. Both cost
``O(n^3)`` and both need ``A`` in memory. **Neither is available when ``A`` is a million by a
million sparse matrix, or when ``A`` exists only as a function that multiplies vectors.**

Lesson 26 built the Krylov space ``K_m(A, v) = span(v, Av, ..., A^{m-1}v)`` and the Arnoldi and
Lanczos recurrences that produce an orthonormal basis for it. This module uses them to find
eigenvalues.

**The whole idea is Rayleigh-Ritz**, and it is lesson 36's Rayleigh quotient generalised from a
vector to a subspace. Project ``A`` onto ``K_m``, solve the small ``m x m`` eigenproblem exactly,
and map the answers back. The eigenvalues of the projection are **Ritz values** and they
approximate the eigenvalues of ``A``.

**Three facts make it practical, and each is measured in lesson 39:**

- `ritz_residual` costs nothing. The residual ``||A y - theta y||`` equals
  ``|h_{m+1,m}| |s_m|``, the last entry of the small eigenvector times one scalar already
  computed. So a rigorous error bound is available without ever forming the Ritz vector.
- **The extremes converge first.** Kaniel and Paige quantified it, and `convergence_order`
  measures it: after ``m`` steps the largest and smallest eigenvalues are accurate and the
  interior ones are not.
- **Restarting keeps the cost bounded.** The basis grows, the cost of orthogonalizing against it
  grows, and memory runs out. `restarted_arnoldi` throws most of it away and keeps the part
  pointing at the wanted eigenvalues.

Everything here derives its sizes from its input, and nothing forms a dense ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .krylov import as_operator


@dataclass
class RitzResult:
    """Ritz values, optional Ritz vectors, and the residual bound for each."""

    values: np.ndarray
    vectors: np.ndarray | None
    residuals: np.ndarray                        # ||A y - theta y|| for each pair
    basis_size: int
    matvecs: int
    converged: np.ndarray                        # which pairs met the tolerance
    history: list = field(default_factory=list)  # residual of the wanted value each restart
    restarts: int = 0
    message: str = ""


def rayleigh_ritz(A, basis, symmetric: bool = True) -> dict:
    """Project ``A`` onto the span of ``basis`` and solve the small eigenproblem exactly.

    With ``Q`` an orthonormal basis of a subspace ``S``, the projected matrix is ``H = Q^H A Q``.
    Its eigenpairs ``(theta, s)`` give **Ritz pairs** ``(theta, Q s)``, and ``theta`` is the
    Rayleigh quotient of the Ritz vector.

    **This is lesson 36's Rayleigh quotient with a subspace in place of a vector**, and it has
    the same optimality: the Ritz values are the best approximations available from ``S`` in the
    sense that the residual ``A y - theta y`` is orthogonal to ``S``. For a symmetric ``A`` they
    also interlace the true eigenvalues and bracket them from the inside.

    ``symmetric=True`` uses ``eigh`` on the symmetrised projection, which is what makes the Ritz
    values real and the vectors orthogonal.
    """
    op = as_operator(A)
    Q = np.atleast_2d(np.asarray(basis, dtype=float))
    if Q.shape[0] < Q.shape[1]:
        raise ValueError(f"the basis is {Q.shape}: more vectors than dimensions")
    AQ = np.column_stack([op.matvec(Q[:, j]) for j in range(Q.shape[1])])
    H = Q.T @ AQ
    if symmetric:
        H = 0.5 * (H + H.T)
        theta, S = np.linalg.eigh(H)
    else:
        theta, S = np.linalg.eig(H)
    Y = Q @ S
    residuals = np.array([float(np.linalg.norm(AQ @ S[:, j] - theta[j] * Y[:, j]))
                          for j in range(theta.size)])
    return {"values": theta, "vectors": Y, "residuals": residuals,
            "projected": H, "matvecs": Q.shape[1]}


def ritz_residual(h_last: float, small_vector) -> np.ndarray:
    """``|h_{m+1,m}| * |s_m|``: the exact Ritz residual, from two numbers already in hand.

    The Arnoldi relation is ``A Q_m = Q_m H_m + h_{m+1,m} q_{m+1} e_m^T``. Multiplying by the
    small eigenvector ``s`` of ``H_m``, with ``H_m s = theta s``,

        A (Q_m s) - theta (Q_m s) = h_{m+1,m} (e_m^T s) q_{m+1},

    and ``q_{m+1}`` is a unit vector, so the residual norm is exactly
    ``|h_{m+1,m}| |s_m|``.

    **So a rigorous error bound is free.** No Ritz vector need be formed, no matrix-vector
    product is needed, and for a symmetric ``A`` the residual bounds the eigenvalue error
    directly: some true eigenvalue is within ``|h_{m+1,m}| |s_m|`` of ``theta``. That is lesson
    35's Bauer-Fike with ``kappa(V) = 1``.

    **A one dimensional argument is one eigenvector and a two dimensional one is a matrix of
    them in columns**, and the distinction is made explicitly rather than guessed from the
    shape. Guessing is what an earlier version did, and for a length 3 vector it could not tell
    "one eigenvector, take its last entry" from "the last row already extracted, take all of
    it": it returned three numbers where one was wanted.
    """
    S = np.asarray(small_vector, dtype=float)
    if S.ndim == 1:
        return np.abs(float(h_last)) * np.abs(S[-1:])
    if S.ndim != 2:
        raise ValueError(f"expected one eigenvector or a matrix of them, got {S.ndim} axes")
    return np.abs(float(h_last)) * np.abs(S[-1, :])


def lanczos_eigen(A, v=None, m: int | None = None, tol: float = 1e-10,
                  reorthogonalize: bool = True, rng=None, n_wanted: int = 1) -> RitzResult:
    """Build a Krylov basis and read the eigenvalues off the projection, for symmetric ``A``.

    ``m`` is the basis size. With ``reorthogonalize=False`` this is the textbook three-term
    recurrence, which is beautiful and which **loses orthogonality catastrophically**, producing
    the ghost eigenvalues lesson 26 measured for the linear solve and lesson 39 measures here.

    Only matrix-vector products are used, so ``A`` may be a function, a sparse matrix or a dense
    one.
    """
    op = as_operator(A)
    n = op.shape[0]
    if m is None:
        m = min(n, 30)
    m = int(m)
    if not 1 <= m <= n:
        raise ValueError(f"the basis size must be between 1 and {n}, got {m}")
    gen = np.random.default_rng() if rng is None else rng
    v = gen.standard_normal(n) if v is None else np.asarray(v, dtype=float).ravel()
    if v.size != n:
        raise ValueError(f"the starting vector has {v.size} entries, A is {n} by {n}")
    nv = float(np.linalg.norm(v))
    if nv == 0.0:
        raise ValueError("the starting vector must not be zero")

    Q = np.zeros((n, m + 1))
    alpha = np.zeros(m)
    beta = np.zeros(m)
    Q[:, 0] = v / nv
    matvecs = 0
    for k in range(m):
        w = op.matvec(Q[:, k])
        matvecs += 1
        alpha[k] = float(Q[:, k] @ w)
        w = w - alpha[k] * Q[:, k] - (beta[k - 1] * Q[:, k - 1] if k > 0 else 0.0)
        if reorthogonalize:
            w = w - Q[:, :k + 1] @ (Q[:, :k + 1].T @ w)
            w = w - Q[:, :k + 1] @ (Q[:, :k + 1].T @ w)     # twice is enough, lesson 30
        beta[k] = float(np.linalg.norm(w))
        if beta[k] <= 1e-14 * max(abs(alpha[k]), 1.0):
            m = k + 1
            alpha, beta = alpha[:m], beta[:m]
            Q = Q[:, :m + 1]
            break
        Q[:, k + 1] = w / beta[k]

    T = np.diag(alpha) + np.diag(beta[:m - 1], 1) + np.diag(beta[:m - 1], -1)
    theta, S = np.linalg.eigh(T)
    residuals = ritz_residual(beta[m - 1], S)
    order = np.argsort(-np.abs(theta))
    return RitzResult(values=theta[order], vectors=Q[:, :m] @ S[:, order],
                      residuals=residuals[order], basis_size=m, matvecs=matvecs,
                      converged=residuals[order] <= tol * max(np.abs(theta).max(), 1.0),
                      message=f"basis of size {m}")


def convergence_order(A, m: int, exact=None, v=None, rng=None) -> dict:
    """How well each **true** eigenvalue is approximated, as a function of its position.

    **The extremes go first.** Kaniel and Paige's bounds say the error in the largest Ritz value
    decays like ``1 / T_{m-1}(1 + 2 gamma)^2`` with ``gamma`` the relative gap to the next
    eigenvalue and ``T`` the Chebyshev polynomial: the same expression that governed the
    conjugate gradient method in lesson 24, for the same reason. Interior eigenvalues have no
    such bound and converge last.

    **The metric matters here and an obvious one is wrong.** Measuring the distance from a
    chosen interior eigenvalue to the *nearest* Ritz value rewards accidents: with a handful of
    Ritz values spread across the spectrum, one of them lands near the middle by chance and the
    interior looks better converged than the extremes. Measured that way at ``m = 5`` the middle
    scored ``1.5e-2`` against the largest eigenvalue's ``1.5e-1``, which is the opposite of the
    truth.

    So this reports the distance from **every** true eigenvalue to its nearest Ritz value,
    indexed by position in the spectrum. The shape of that curve is the answer, and one
    favourable accident cannot hide it.

    ``exact`` may be supplied when the spectrum is known in closed form, which avoids forming a
    dense matrix and lets the measurement run at sizes where that is impossible.
    """
    op = as_operator(A)
    n = op.shape[0]
    if exact is None:
        dense = np.column_stack([op.matvec(e) for e in np.eye(n)])
        exact = np.sort(np.linalg.eigvalsh(0.5 * (dense + dense.T)))
    exact = np.sort(np.asarray(exact, dtype=float).ravel())
    out = lanczos_eigen(op, v=v, m=min(int(m), n), rng=rng)
    got = np.sort(out.values)
    distance = np.array([float(np.min(np.abs(got - lam))) for lam in exact])
    return {"exact": exact, "ritz": got, "distance": distance,
            "position": np.linspace(0.0, 1.0, exact.size),
            "matvecs": out.matvecs, "basis_size": out.basis_size}


def ghost_eigenvalues(A, v=None, m: int = 60, rng=None, tol: float = 1e-8) -> dict:
    """Count the spurious duplicate Ritz values a non-reorthogonalized Lanczos produces.

    Once orthogonality is lost, the recurrence restarts inside the space it has already
    explored, and a converged eigenvalue **reappears** as a second copy. Paige's theorem makes
    this precise: an eigenvalue is duplicated exactly when its Ritz vector has converged, so
    ghosts are a symptom of success rather than of failure.

    Returns the ghost count for both variants, and the loss of orthogonality of the basis.
    """
    op = as_operator(A)
    out = {}
    for label, reorth in (("plain", False), ("reorthogonalized", True)):
        res = lanczos_eigen(op, v=v, m=m, reorthogonalize=reorth, rng=rng)
        vals = np.sort(res.values)
        gaps = np.diff(vals)
        scale = max(float(np.abs(vals).max()), 1.0)
        out[label] = {"values": vals,
                      "duplicates": int(np.sum(gaps <= tol * scale)),
                      "distinct": int(1 + np.sum(gaps > tol * scale))}
    return out


def basis_orthogonality(A, v=None, m: int = 40, reorthogonalize: bool = False,
                        rng=None) -> np.ndarray:
    """``||Q_k^T Q_k - I||`` as the basis grows, for the two Lanczos variants."""
    op = as_operator(A)
    n = op.shape[0]
    gen = np.random.default_rng() if rng is None else rng
    v = gen.standard_normal(n) if v is None else np.asarray(v, dtype=float).ravel()
    Q = np.zeros((n, m + 1))
    Q[:, 0] = v / float(np.linalg.norm(v))
    beta_prev = 0.0
    trail = []
    for k in range(m):
        w = op.matvec(Q[:, k])
        a = float(Q[:, k] @ w)
        w = w - a * Q[:, k] - (beta_prev * Q[:, k - 1] if k > 0 else 0.0)
        if reorthogonalize:
            w = w - Q[:, :k + 1] @ (Q[:, :k + 1].T @ w)
        beta_prev = float(np.linalg.norm(w))
        if beta_prev <= 1e-14:
            break
        Q[:, k + 1] = w / beta_prev
        sub = Q[:, :k + 2]
        trail.append(float(np.linalg.norm(sub.T @ sub - np.eye(k + 2))))
    return np.array(trail)


def restarted_arnoldi(A, n_wanted: int, basis_size: int = 20, v=None,
                      tol: float = 1e-10, max_restarts: int = 200,
                      which: str = "largest", rng=None) -> RitzResult:
    """Arnoldi with **thick restarting**: keep the wanted Ritz vectors, discard the rest.

    The basis cannot grow forever. Orthogonalizing vector ``m`` against the previous ones costs
    ``O(nm)``, so building a basis of size ``m`` costs ``O(nm^2)`` and stores ``nm`` numbers.
    At ``n = 10^6`` a basis of 1000 is 8 GB.

    **Restarting throws the basis away and starts again from a better vector.** The thick
    version keeps the ``n_wanted`` best Ritz vectors and restarts from their span, so the
    information already earned is not lost. Sorensen's implicitly restarted Arnoldi does the same
    thing through a polynomial filter applied by QR steps, which is more elegant and, for the
    purposes of measuring the idea, equivalent.

    ``which`` is ``"largest"`` or ``"smallest"`` by modulus.
    """
    op = as_operator(A)
    n = op.shape[0]
    n_wanted = int(n_wanted)
    basis_size = int(basis_size)
    if not 1 <= n_wanted <= basis_size <= n:
        raise ValueError(f"need 1 <= n_wanted ({n_wanted}) <= basis_size ({basis_size}) "
                         f"<= n ({n})")
    if which not in ("largest", "smallest"):
        raise ValueError(f"which must be 'largest' or 'smallest', got {which!r}")
    gen = np.random.default_rng() if rng is None else rng
    start = gen.standard_normal(n) if v is None else np.asarray(v, dtype=float).ravel()
    if start.size != n:
        raise ValueError(f"the starting vector has {start.size} entries, A is {n} by {n}")

    out = RitzResult(values=np.zeros(0), vectors=None, residuals=np.zeros(0),
                     basis_size=basis_size, matvecs=0,
                     converged=np.zeros(0, dtype=bool))
    keep = start.reshape(n, 1) / float(np.linalg.norm(start))

    for restart in range(int(max_restarts)):
        Q = _grow_basis(op, keep, basis_size, out)
        rr = rayleigh_ritz(op, Q, symmetric=True)
        out.matvecs += rr["matvecs"]
        order = (np.argsort(-np.abs(rr["values"])) if which == "largest"
                 else np.argsort(np.abs(rr["values"])))
        picked = order[:n_wanted]
        out.values = rr["values"][picked]
        out.vectors = rr["vectors"][:, picked]
        out.residuals = rr["residuals"][picked]
        out.restarts = restart + 1
        worst = float(np.max(out.residuals))
        out.history.append(worst)
        scale = max(float(np.abs(rr["values"]).max()), 1.0)
        if worst <= tol * scale:
            out.converged = np.ones(n_wanted, dtype=bool)
            out.message = f"converged after {restart + 1} restarts"
            return out
        keep, _ = np.linalg.qr(out.vectors)
    out.converged = out.residuals <= tol * max(float(np.abs(out.values).max()), 1.0)
    out.message = f"hit the restart limit of {max_restarts}"
    return out


def _grow_basis(op, keep, basis_size: int, out: RitzResult) -> np.ndarray:
    """Extend an orthonormal block to ``basis_size`` columns by Arnoldi from its last column."""
    n = op.shape[0]
    Q = np.zeros((n, basis_size))
    k0 = keep.shape[1]
    Q[:, :k0] = keep
    for k in range(k0, basis_size):
        w = op.matvec(Q[:, k - 1])
        out.matvecs += 1
        w = w - Q[:, :k] @ (Q[:, :k].T @ w)
        w = w - Q[:, :k] @ (Q[:, :k].T @ w)
        nrm = float(np.linalg.norm(w))
        if nrm <= 1e-14:
            gen = np.random.default_rng(k)
            w = gen.standard_normal(n)
            w = w - Q[:, :k] @ (Q[:, :k].T @ w)
            nrm = float(np.linalg.norm(w))
        Q[:, k] = w / nrm
    return Q


# ----------------------------------------------------------------------------- test problems


def sparse_laplacian(size: int, dimension: int = 1):
    """The discrete Laplacian as a **function**, never as a matrix.

    In one dimension its eigenvalues are known exactly, ``4 sin^2(k pi / (2(n+1)))``, so a
    method can be checked against the truth at any size. In two dimensions the same formula
    applies to sums of pairs.

    Returns a `LinearOperator`, so ``n`` can be a million and the storage is nothing.
    """
    from .krylov import LinearOperator

    size = int(size)
    dimension = int(dimension)
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    if dimension not in (1, 2):
        raise ValueError(f"dimension must be 1 or 2, got {dimension}")

    if dimension == 1:
        n = size

        def matvec(x):
            x = np.asarray(x, dtype=float).ravel()
            out = 2.0 * x
            out[:-1] -= x[1:]
            out[1:] -= x[:-1]
            return out
    else:
        n = size * size

        def matvec(x):
            g = np.asarray(x, dtype=float).reshape(size, size)
            out = 4.0 * g
            out[:-1, :] -= g[1:, :]
            out[1:, :] -= g[:-1, :]
            out[:, :-1] -= g[:, 1:]
            out[:, 1:] -= g[:, :-1]
            return out.ravel()

    return LinearOperator(matvec, shape=(n, n))


def spiked_laplacian(size: int, spike: float):
    """The 1D Laplacian plus ``spike`` on one diagonal entry, as a function.

    **The plain Laplacian is the worst case for a Krylov eigenvalue method**, and it is worth
    knowing why before using one. Its eigenvalues are ``4 sin^2(k pi / (2(n+1)))``, which crowd
    towards 4 at the top and towards 0 at the bottom with gaps of order ``1/n^2``. At
    ``n = 10000`` the top two differ by ``3.0e-7``, so no method that separates eigenvalues by
    their gaps can make progress. Measured: 200 restarts reach only ``6e-6``.

    Adding a spike to one diagonal entry pushes a single eigenvalue clear of the rest, which is
    the situation Krylov methods are good at and the situation that actually arises in practice
    (a dominant mode, a leading principal component, a stability eigenvalue). Returns the
    operator; the exact top eigenvalue is not available in closed form, so a reference must be
    computed some other way.
    """
    from .krylov import LinearOperator

    size = int(size)
    if size < 2:
        raise ValueError(f"size must be at least 2, got {size}")
    spike = float(spike)

    def matvec(x):
        x = np.asarray(x, dtype=float).ravel()
        out = 2.0 * x
        out[:-1] -= x[1:]
        out[1:] -= x[:-1]
        out[0] += spike * x[0]
        return out

    return LinearOperator(matvec, shape=(size, size))


def operator_with_spectrum(spectrum, seed: int = 1):
    """A symmetric `LinearOperator` with exactly the given eigenvalues.

    Used to control the one thing that decides how Lanczos behaves: **the separation of the
    eigenvalue being sought**. A well separated extreme eigenvalue converges in a handful of
    steps, and by Paige's theorem the orthogonality of the basis collapses at exactly that
    moment. So the ghosts of `ghost_eigenvalues` can be turned on and off by moving one number.
    """
    from .krylov import LinearOperator

    spectrum = np.asarray(spectrum, dtype=float).ravel()
    n = spectrum.size
    if n < 1:
        raise ValueError("the spectrum must not be empty")
    Q, _ = np.linalg.qr(np.random.default_rng(int(seed)).standard_normal((n, n)))
    return LinearOperator(Q @ np.diag(spectrum) @ Q.T), np.sort(spectrum)


def separated_spectrum(n: int, gap: float) -> np.ndarray:
    """``n`` eigenvalues at ``1, 2, ..., n-1`` with one outlier at ``gap`` times the largest.

    ``gap = 1`` puts the outlier inside the cluster and Lanczos converges slowly. Raising it
    makes the top eigenvalue converge in a few steps, which is what triggers the loss of
    orthogonality that produces ghosts.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    body = np.arange(1.0, float(n))
    return np.concatenate([[float(gap) * body[-1]], body])


def laplacian_eigenvalues(size: int, dimension: int = 1) -> np.ndarray:
    """The exact spectrum of `sparse_laplacian`, from the closed form."""
    size = int(size)
    one = 4.0 * np.sin(np.arange(1, size + 1) * np.pi / (2.0 * (size + 1))) ** 2
    if int(dimension) == 1:
        return np.sort(one)
    return np.sort((one[:, None] + one[None, :]).ravel())
