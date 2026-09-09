"""The QR algorithm: the workhorse for computing every eigenvalue at once.

What this module is for
-----------------------
Lesson 36's methods find **one** eigenvalue each, and deflation stacks them up awkwardly. The QR
algorithm finds them all, and it is what every library actually calls.

The whole thing is three lines::

    A_0 = A
    A_k = Q_k R_k          (a QR factorization)
    A_{k+1} = R_k Q_k      (multiply the factors back the OTHER way)

``A_{k+1} = R_k Q_k = Q_k^T A_k Q_k``, so every step is an **orthogonal similarity**: the
eigenvalues never move, and lesson 35 section 8 showed that orthogonal is the only kind that
leaves the conditioning alone. The iterates converge to the Schur form, whose diagonal is the
answer.

**Why it works** is `simultaneous_iteration`: the unshifted QR algorithm is exactly power
iteration applied to a whole orthonormal basis at once, with re-orthogonalization each step. So
its convergence rate is inherited from lesson 36, ``|lambda_{i+1}/lambda_i|`` for each
subdiagonal entry, and everything that made power iteration slow makes this slow too.

**Three things turn that into a practical algorithm**, and each is measured in lesson 37:

- `hessenberg` reduces to almost-triangular form first, in a finite number of steps, which drops
  the cost of a QR step from ``O(n^3)`` to ``O(n^2)`` and is preserved by every later step.
- **Shifts** apply the algorithm to ``A - mu I``, turning the ratio into
  ``|lambda_{i+1} - mu| / |lambda_i - mu|``, which a good shift makes tiny.
  `wilkinson_shift` is the standard choice and it never breaks down where the obvious one does.
- **Deflation** splits the problem in two whenever a subdiagonal entry becomes negligible.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import qr as _qr


@dataclass
class QRResult:
    """The outcome of a QR iteration, with enough history to measure a rate."""

    T: np.ndarray                     # the (quasi) triangular limit
    Q: np.ndarray                     # the accumulated similarity, A = Q T Q^T
    eigenvalues: np.ndarray
    iterations: int
    converged: bool
    message: str
    subdiagonals: list = field(default_factory=list)   # |A[i+1, i]| history
    shifts: list = field(default_factory=list)
    deflations: list = field(default_factory=list)     # (step, index) each time one splits off


# ----------------------------------------------------------------------------- Hessenberg


def hessenberg(A, compute_q: bool = True):
    """Reduce to upper Hessenberg form by Householder similarities: ``A = Q H Q^T``.

    Upper Hessenberg means zero below the **first subdiagonal**. That is as close to triangular
    as a *finite* algorithm can get, and the reason is lesson 35: reaching triangular in finitely
    many steps would compute the eigenvalues in finitely many steps, which is impossible.

    **The reflector must start one row lower than it does in a QR factorization**, and that one
    index is the whole difference. In lesson 31's Householder QR the reflector for column ``k``
    acts on rows ``k`` downward and is applied only from the left. Here it must be applied from
    **both** sides to keep the eigenvalues, and a reflector acting on rows ``k`` downward would,
    when applied on the right, refill the column it just cleared. Starting at row ``k+1`` leaves
    the ``(k+1, k)`` entry alone, and that entry is exactly what survives on the subdiagonal.

    Costs ``(10/3) n^3`` with ``Q``, ``(4/3) n^3`` without. **It is finite and it is done once**,
    after which every QR step costs ``O(n^2)`` instead of ``O(n^3)``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A is {A.shape}, eigenvalues need a square matrix")
    n = A.shape[0]
    H = A.astype(float).copy()
    Q = np.eye(n) if compute_q else None

    for k in range(n - 2):
        v, _ = _qr.householder_vector(H[k + 1:, k])
        vv = float(v @ v)
        if vv == 0.0:
            continue
        # From the left on rows k+1 down, then from the right on columns k+1 across. Both are
        # needed: one alone is not a similarity and would change the eigenvalues.
        H[k + 1:, k:] -= (2.0 / vv) * np.outer(v, v @ H[k + 1:, k:])
        H[:, k + 1:] -= (2.0 / vv) * np.outer(H[:, k + 1:] @ v, v)
        H[k + 2:, k] = 0.0
        if Q is not None:
            Q[:, k + 1:] -= (2.0 / vv) * np.outer(Q[:, k + 1:] @ v, v)
    return (H, Q) if compute_q else H


def is_hessenberg(A, tol: float = 1e-12) -> bool:
    """Is everything below the first subdiagonal zero, relative to the size of the matrix?"""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] < 3:
        return True
    scale = max(float(np.linalg.norm(A, np.inf)), 1e-300)
    return float(np.abs(np.tril(A, -2)).max()) <= tol * scale


# ----------------------------------------------------------------------------- shifts


def wilkinson_shift(A) -> float:
    """The eigenvalue of the trailing ``2x2`` block closer to ``A[-1, -1]``.

    The obvious shift is ``A[-1, -1]`` itself (the Rayleigh shift), and it works well until it
    does not: on a matrix like ``[[0, 1], [1, 0]]`` it is exactly 0, the shifted matrix is
    already orthogonal, and the QR step returns the matrix **unchanged**. The iteration stalls
    forever on a problem whose eigenvalues are plus and minus one.

    Wilkinson's shift takes an eigenvalue of the trailing ``2x2`` block instead, which for that
    example is plus or minus one and converges immediately. It is provably convergent for a
    symmetric tridiagonal matrix, and cubically so.

    The formula is written to avoid cancellation, in the manner of lesson 05's quadratic
    formula: ``d + sign(d) * sqrt(d^2 + b^2)`` in the denominator adds like-signed quantities.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    if n == 1:
        return float(A[0, 0])
    a, b = float(A[n - 2, n - 2]), float(A[n - 2, n - 1])
    c, d_ = float(A[n - 1, n - 2]), float(A[n - 1, n - 1])
    delta = (a - d_) / 2.0
    bc = b * c
    if bc == 0.0:
        return d_
    disc = delta * delta + bc
    if disc < 0.0:                       # a complex pair: fall back to the Rayleigh shift
        return d_
    root = np.sqrt(disc)
    sign = 1.0 if delta >= 0.0 else -1.0
    denom = delta + sign * root
    return d_ - bc / denom if denom != 0.0 else d_ - np.sqrt(abs(bc))


def rayleigh_shift(A) -> float:
    """``A[-1, -1]``, the simplest shift. Fast when it works and it does not always work."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    return float(A[-1, -1])


# ----------------------------------------------------------------------------- the algorithm


def qr_step(A):
    """One unshifted step: factor ``A = QR``, then form ``RQ``.

    ``RQ = Q^T A Q``, so the eigenvalues are unchanged and the transformation is orthogonal.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    Q, R = np.linalg.qr(A)
    return R @ Q, Q


def qr_algorithm(A, shift: str = "wilkinson", tol: float = 1e-13, max_iter: int = 5000,
                 reduce_first: bool = True) -> QRResult:
    """The shifted QR algorithm with deflation. Returns every eigenvalue.

    ``shift`` is ``"none"``, ``"rayleigh"`` or ``"wilkinson"``. The unshifted version is included
    because it is the one whose convergence is easy to explain, not because it should be used:
    lesson 37 measures it taking hundreds of times more steps.

    **Deflation is what makes the cost bearable.** Once ``|A[i+1, i]|`` falls below the
    tolerance times the size of its neighbours, the matrix splits into two independent blocks
    and the iteration continues on the smaller one. The standard test is

        |A[i+1, i]| <= tol * (|A[i, i]| + |A[i+1, i+1]|),

    which is relative to the local scale rather than to the whole matrix, so a small eigenvalue
    is not held to the accuracy of a large one.

    **Real ``2x2`` blocks are left alone.** A real matrix with a complex conjugate pair cannot be
    triangularized over the reals, so the limit is the **real Schur form**: quasi-triangular,
    with ``2x2`` blocks on the diagonal holding the complex pairs. Lesson 36 measured the
    Rayleigh quotient iteration failing on exactly this, and this is the fix.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A is {A.shape}, eigenvalues need a square matrix")
    if shift not in ("none", "rayleigh", "wilkinson"):
        raise ValueError(f"shift must be 'none', 'rayleigh' or 'wilkinson', got {shift!r}")
    n = A.shape[0]
    if reduce_first:
        T, Q_total = hessenberg(A)
    else:
        T, Q_total = A.astype(float).copy(), np.eye(n)

    out = QRResult(T=T, Q=Q_total, eigenvalues=np.zeros(n, dtype=complex), iterations=0,
                   converged=False, message="")
    if n == 1:
        out.eigenvalues = np.array([complex(T[0, 0])])
        out.converged = True
        out.message = "a 1 by 1 matrix needs no iteration"
        return out

    high = n - 1                          # the active block is T[:high+1, :high+1]
    steps = 0
    while high > 0 and steps < int(max_iter):
        # Deflate from the bottom while the subdiagonal entry is negligible.
        local = abs(T[high, high]) + abs(T[high - 1, high - 1])
        if abs(T[high, high - 1]) <= tol * max(local, 1e-300):
            T[high, high - 1] = 0.0
            out.deflations.append((steps, high))
            high -= 1
            continue
        # A 2x2 block whose eigenvalues are complex cannot be split any further.
        if high == 1 or (high >= 2 and abs(T[high - 1, high - 2])
                         <= tol * max(abs(T[high - 1, high - 1])
                                      + abs(T[high - 2, high - 2]), 1e-300)):
            block = T[high - 1:high + 1, high - 1:high + 1]
            disc = (block[0, 0] - block[1, 1]) ** 2 + 4.0 * block[0, 1] * block[1, 0]
            if disc < 0.0:
                if high >= 2:
                    T[high - 1, high - 2] = 0.0
                out.deflations.append((steps, high))
                high -= 2
                continue

        active = slice(0, high + 1)
        mu = {"none": 0.0,
              "rayleigh": rayleigh_shift(T[active, active]),
              "wilkinson": wilkinson_shift(T[active, active])}[shift]
        out.shifts.append(mu)
        block = T[active, active] - mu * np.eye(high + 1)
        Q, R = np.linalg.qr(block)
        T[active, active] = R @ Q + mu * np.eye(high + 1)
        # The similarity must reach the OFF-block parts too, and exactly once. Applying
        # T[:, active] @ Q would hit the active block a second time, since it is leading.
        T[active, high + 1:] = Q.T @ T[active, high + 1:]
        T[high + 1:, active] = T[high + 1:, active] @ Q
        Q_total[:, active] = Q_total[:, active] @ Q
        steps += 1
        out.subdiagonals.append(float(abs(T[high, high - 1])))

    out.iterations = steps
    out.converged = high <= 0
    out.message = ("converged" if out.converged
                   else f"hit the iteration limit of {max_iter}")
    out.T = T
    out.Q = Q_total
    out.eigenvalues = quasi_triangular_eigenvalues(T)
    return out


def quasi_triangular_eigenvalues(T, tol: float = 1e-10) -> np.ndarray:
    """Read the eigenvalues off a real Schur form, handling the ``2x2`` blocks.

    A ``1x1`` block on the diagonal gives one real eigenvalue. A ``2x2`` block with a negative
    discriminant gives a complex conjugate pair, which is why a real matrix can be reduced no
    further than this over the reals.
    """
    T = np.atleast_2d(np.asarray(T, dtype=float))
    n = T.shape[0]
    vals = []
    i = 0
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > tol * max(abs(T[i, i]) + abs(T[i + 1, i + 1]),
                                                      1e-300):
            a, b = T[i, i], T[i, i + 1]
            c, d_ = T[i + 1, i], T[i + 1, i + 1]
            tr, det = a + d_, a * d_ - b * c
            disc = tr * tr / 4.0 - det
            if disc < 0.0:
                root = np.sqrt(-disc)
                vals.extend([complex(tr / 2.0, root), complex(tr / 2.0, -root)])
            else:
                root = np.sqrt(disc)
                vals.extend([complex(tr / 2.0 + root), complex(tr / 2.0 - root)])
            i += 2
        else:
            vals.append(complex(T[i, i]))
            i += 1
    return np.array(vals)


# ----------------------------------------------------------------------------- why it works


def simultaneous_iteration(A, n_vectors: int | None = None, steps: int = 50,
                           rng=None, start=None) -> dict:
    """Power iteration on a whole orthonormal basis, re-orthogonalized each step.

    ``V <- A V``, then ``V <- qr(V).Q``. Each column converges to an eigenvector at rate
    ``|lambda_{k+1}/lambda_k|``, the same rates that appear on the QR algorithm's subdiagonal.

    **With ``n_vectors = n`` and ``start`` the identity, this IS the unshifted QR algorithm**:
    the matrices ``V_k^T A V_k`` are exactly the QR algorithm's iterates ``A_k``, entry for
    entry, and lesson 37 checks that to ``1e-14``.

    **The starting basis matters for that equivalence.** A random start converges to the same
    limit along a different path, so the two agree only to the level of their remaining error.
    Both facts are worth seeing: the theorem needs the identity, and the *convergence* does not.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    p = n if n_vectors is None else int(n_vectors)
    if not 1 <= p <= n:
        raise ValueError(f"n_vectors must be between 1 and {n}, got {p}")
    if start is None:
        gen = np.random.default_rng() if rng is None else rng
        V, _ = np.linalg.qr(gen.standard_normal((n, p)))
    else:
        V = np.atleast_2d(np.asarray(start, dtype=float))
        if V.shape != (n, p):
            raise ValueError(f"start is {V.shape}, expected ({n}, {p})")
        V, _ = np.linalg.qr(V)
    history = []
    for _ in range(int(steps)):
        V, _ = np.linalg.qr(A @ V)
        history.append(V.copy())
    return {"V": V, "projected": V.T @ A @ V, "history": history}


def convergence_rates(A) -> np.ndarray:
    """``|lambda_{i+1} / lambda_i|`` for the sorted spectrum: the rate each subdiagonal entry
    of the unshifted QR algorithm decays at."""
    vals = np.sort(np.abs(np.linalg.eigvals(np.atleast_2d(np.asarray(A)))))[::-1]
    if vals.size < 2:
        return np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vals[:-1] > 0, vals[1:] / np.maximum(vals[:-1], 1e-300), 0.0)


def schur_error(A, T, Q) -> dict:
    """How well ``A = Q T Q^T`` holds, and how triangular ``T`` actually is."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    scale = max(float(np.linalg.norm(A)), 1e-300)
    below = np.abs(np.tril(T, -1))
    # A real Schur form is allowed nonzeros on the first subdiagonal, from the 2x2 blocks.
    return {"orthogonality": float(np.linalg.norm(Q.T @ Q - np.eye(n))),
            "reconstruction": float(np.linalg.norm(A - Q @ T @ Q.T) / scale),
            "below_first_subdiagonal": float(np.abs(np.tril(T, -2)).max()) if n > 2 else 0.0,
            "max_subdiagonal": float(np.max(np.diag(below, -1))) if n > 1 else 0.0}


def lr_algorithm(A, tol: float = 1e-12, max_iter: int = 2000) -> dict:
    """Rutishauser's LR algorithm: the QR algorithm with LU in place of QR.

    ``A_k = L_k U_k``, then ``A_{k+1} = U_k L_k = L_k^{-1} A_k L_k``. It came first, it is
    cheaper by a factor of about 2, and **it is not used**, for the reason lesson 35 section 8
    gives: ``L_k`` is a general similarity, not an orthogonal one, so the conditioning is free
    to degrade. Without pivoting it also breaks down whenever a leading minor is singular.

    Included so lesson 37 can measure the difference rather than assert it.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    from scipy.linalg import lu

    M = A.astype(float).copy()
    n = M.shape[0]
    growth = [float(np.linalg.cond(M))]
    for k in range(int(max_iter)):
        P, L, U = lu(M)
        if not np.allclose(P, np.eye(n)):
            return {"eigenvalues": np.diag(M).copy(), "iterations": k, "converged": False,
                    "message": f"a pivot was needed at step {k}: unpivoted LR breaks down here",
                    "condition_growth": growth}
        M = U @ L
        growth.append(float(np.linalg.cond(M)))
        off = float(np.abs(np.tril(M, -1)).max())
        if off <= tol * max(float(np.abs(M).max()), 1e-300):
            return {"eigenvalues": np.diag(M).copy(), "iterations": k + 1, "converged": True,
                    "message": "converged", "condition_growth": growth}
    return {"eigenvalues": np.diag(M).copy(), "iterations": int(max_iter), "converged": False,
            "message": f"hit the iteration limit of {max_iter}", "condition_growth": growth}
