"""Least squares: the normal equations, the projection view, and why the obvious route is bad.

Part 3 solved ``A x = b`` when a solution exists. This module is about what to do when it does
not, which is the ordinary case whenever there are more equations than unknowns.

The residual cannot be made zero, so the goal changes: find the ``x`` minimising
``||b - A x||_2``. Setting the gradient to zero gives the **normal equations**

    A^T A x = A^T b,

and the geometry says the same thing: ``A x`` is the **orthogonal projection** of ``b`` onto
the range of ``A``, so the residual is perpendicular to every column, ``A^T(b - A x) = 0``.

**And the normal equations are a bad way to compute the answer.** Forming ``A^T A`` squares the
condition number, so a problem solvable to eight digits becomes one solvable to none. Lesson 29
measures that; lessons 30 to 32 build the replacement out of orthogonality, which is the reason
lesson 16 spent so long on it.

Everything here takes its shape from its arguments: ``A`` may be any ``m`` by ``n`` with
``m >= n``, the degree of a fit comes from the caller, and no routine assumes a particular size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- the result type


@dataclass
class LSResult:
    """A least squares solution together with what it costs to trust it.

    The residual norm is not the error. For a least squares problem it is not even supposed to
    be zero, so a small residual means nothing on its own and the fields below are what a
    caller actually has to look at.
    """

    x: np.ndarray
    residual: np.ndarray
    residual_norm: float
    rank: int
    method: str
    cond_A: float = float("nan")
    cond_normal: float = float("nan")

    @property
    def is_full_rank(self) -> bool:
        return self.rank == min(*self.shape) if hasattr(self, "shape") else True


# ---------------------------------------------------------------- the normal equations


def normal_matrix(A) -> np.ndarray:
    """A^T A, formed explicitly. **Do not solve with this**; it is here to be measured.

    Forming the product squares the condition number, because the singular values of A^T A are
    the squares of those of A:

        kappa_2(A^T A) = kappa_2(A)^2.

    So a matrix with kappa = 1e8, which loses eight digits, becomes one with kappa = 1e16,
    which loses all sixteen. Lesson 29 measures the crossover.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    return A.T @ A


def solve_normal_equations(A, b, check_conditioning: bool = True) -> LSResult:
    """Solve the least squares problem by forming and solving A^T A x = A^T b.

    This is the textbook route and the one to avoid in practice. It is implemented here so the
    lessons can measure exactly how bad it is rather than asserting it, and because it is
    genuinely the right choice in one situation: when A is very well conditioned and m is
    enormous, since A^T A is only n by n and can be accumulated in one pass over the rows.

    ``check_conditioning`` computes both condition numbers, which costs an SVD and is therefore
    off the critical path in any real use.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    if b.size != m:
        raise ValueError(f"b has length {b.size}, but A has {m} rows")
    if m < n:
        raise ValueError(f"underdetermined system: {m} rows, {n} columns. "
                         "Use the pseudoinverse for the minimum norm solution")

    G = A.T @ A
    x = np.linalg.solve(G, A.T @ b)
    r = b - A @ x
    ka = kn = float("nan")
    if check_conditioning:
        ka = float(np.linalg.cond(A))
        kn = float(np.linalg.cond(G))
    return LSResult(x, r, float(np.linalg.norm(r)), n, "normal equations", ka, kn)


def solve_cholesky_normal(A, b) -> LSResult:
    """The same, but exploiting that A^T A is symmetric positive definite (lesson 20).

    Half the work of a general solve, and exactly the same accuracy problem: the damage is done
    when A^T A is formed, not when it is factored. This routine exists so a lesson can show that
    a better factorization does not rescue a badly posed reformulation.
    """
    from .cholesky import cholesky, cholesky_solve

    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    if b.size != m:
        raise ValueError(f"b has length {b.size}, but A has {m} rows")
    G = A.T @ A
    x = cholesky_solve(cholesky(G), A.T @ b)
    r = b - A @ x
    return LSResult(x, r, float(np.linalg.norm(r)), n, "Cholesky on the normal equations",
                    float(np.linalg.cond(A)), float(np.linalg.cond(G)))


# ---------------------------------------------------------------- projection


def projector_onto_range(A) -> np.ndarray:
    """P = A(A^T A)^-1 A^T, the orthogonal projector onto the range of A.

    Every property lesson 16 established is checkable here: P^2 = P, P^T = P, and
    ``||P||_2 = 1`` exactly, because an orthogonal projector cannot amplify anything.

    **Never form this to solve a problem.** It is m by m where the system is m by n, so for a
    tall thin A it is enormously larger than the data. It is a measuring instrument.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    return A @ np.linalg.solve(A.T @ A, A.T)


def residual_is_orthogonal(A, x, b, tol: float = 1e-10) -> tuple[bool, float]:
    """Check A^T(b - A x) = 0, the defining property of a least squares solution.

    Returns the verdict and the size of the violation, scaled so the tolerance means the same
    thing at every size and every scaling of the data.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    r = b - A @ x
    v = float(np.linalg.norm(A.T @ r))
    # When the residual is (nearly) zero the condition holds trivially, and dividing by ||r||
    # would turn that into a huge scaled violation. Fall back to ||b|| in that case.
    u = float(np.finfo(float).eps) / 2
    norm_A = float(np.linalg.norm(A, 2))
    scale = max(norm_A * float(np.linalg.norm(r)),
                u * norm_A * float(np.linalg.norm(b)), 1e-300)
    return v <= tol * scale, v / scale


def angle_to_range(A, b) -> float:
    """The angle theta between b and the range of A, in radians.

    This is the quantity the least squares perturbation theory turns on, and it is not the
    condition number. When theta is near zero the fit is nearly exact and the problem behaves
    like a square system; when theta is near pi/2 the data is almost orthogonal to the model,
    the residual is nearly all of b, and the sensitivity picks up a factor of
    ``kappa^2 tan(theta)``.

    Lesson 32 measures both regimes on the same matrix.

    **Computed with ``arctan2`` rather than ``arccos``, and that is not a style choice.**
    ``arccos(||proj||/||b||)`` is catastrophically inaccurate for a small angle, because the
    derivative of arccos is infinite at 1 and the argument is 1 to within roundoff. Measured:
    at a relative residual of ``3.3e-9`` the arccos form returned an angle whose sine was wrong
    by a factor of **6.8**, while ``arctan2(||r||, ||proj||)`` was exact to ``4e-16``. It is
    lesson 05's cancellation wearing a trigonometric hat.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    if float(np.linalg.norm(b)) == 0.0:
        return 0.0
    proj = A @ np.linalg.lstsq(A, b, rcond=None)[0]
    residual = b - proj
    return float(np.arctan2(float(np.linalg.norm(residual)), float(np.linalg.norm(proj))))


# ---------------------------------------------------------------- model fitting


def vandermonde_design(x, degree: int) -> np.ndarray:
    """The design matrix for a polynomial fit of the given degree. Shape (len(x), degree+1).

    Columns are ``1, x, x^2, ...``, which is the natural basis and a numerically poor one: its
    condition number grows exponentially with the degree, which is lesson 29's first
    demonstration that a well posed problem can be posed badly.
    """
    x = np.asarray(x, dtype=float).ravel()
    if degree < 0:
        raise ValueError(f"degree must be at least 0, got {degree}")
    return np.vander(x, int(degree) + 1, increasing=True)


def fourier_design(t, n_harmonics: int, period: float = 1.0) -> np.ndarray:
    """The design matrix for a constant plus ``n_harmonics`` sine and cosine pairs.

    Shape ``(len(t), 2 * n_harmonics + 1)``. Unlike the Vandermonde basis this one is nearly
    orthogonal when the samples are evenly spaced over a period, so the same fitting problem is
    well conditioned in it. **The basis, not the data, is what was wrong.**
    """
    t = np.asarray(t, dtype=float).ravel()
    k = int(n_harmonics)
    if k < 0:
        raise ValueError(f"n_harmonics must be at least 0, got {k}")
    cols = [np.ones_like(t)]
    for j in range(1, k + 1):
        w = 2.0 * np.pi * j / float(period)
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))
    return np.column_stack(cols)


def fit_polynomial(x, y, degree: int, method: str = "qr") -> LSResult:
    """Fit a polynomial of the given degree, by the named method.

    ``method`` is ``"normal"``, ``"cholesky"`` or ``"qr"``. The default is QR because the normal
    equations square the condition number, and for a Vandermonde design that is fatal at quite
    modest degrees.
    """
    A = vandermonde_design(x, degree)
    y = np.asarray(y, dtype=float).ravel()
    if y.size != A.shape[0]:
        raise ValueError(f"y has length {y.size}, but x has length {A.shape[0]}")
    if method == "normal":
        return solve_normal_equations(A, y)
    if method == "cholesky":
        return solve_cholesky_normal(A, y)
    if method == "qr":
        return solve_qr(A, y)
    raise ValueError("method must be 'normal', 'cholesky' or 'qr'")


def linearize_exponential(x, y) -> dict:
    """Fit ``y = c * exp(k x)`` by taking logs and fitting a straight line.

    Taking logs turns the model linear, which is why it is the textbook approach, and it also
    changes the problem: it minimises the error in ``log y``, not in ``y``. That reweights the
    data towards the small values, sometimes by orders of magnitude.

    Returns the linearized fit, so a lesson can compare it against a genuine nonlinear fit
    (lesson 34) and measure the difference rather than describing it.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"x has {x.size} points and y has {y.size}")
    if np.any(y <= 0.0):
        raise ValueError("linearization needs strictly positive y; log of a nonpositive "
                         "value is undefined, and the model cannot produce one")
    res = solve_qr(np.column_stack([np.ones_like(x), x]), np.log(y))
    c, k = float(np.exp(res.x[0])), float(res.x[1])
    pred = c * np.exp(k * x)
    return {"c": c, "k": k,
            "sse_original": float(np.sum((y - pred) ** 2)),
            "sse_log": float(res.residual_norm ** 2),
            "predict": lambda t: c * np.exp(k * np.asarray(t, dtype=float))}


def linearize_power(x, y) -> dict:
    """Fit ``y = c * x^p`` by taking logs of both sides. The same caveat as the exponential."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"x has {x.size} points and y has {y.size}")
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("the power model needs strictly positive x and y")
    res = solve_qr(np.column_stack([np.ones_like(x), np.log(x)]), np.log(y))
    c, p = float(np.exp(res.x[0])), float(res.x[1])
    pred = c * x ** p
    return {"c": c, "p": p,
            "sse_original": float(np.sum((y - pred) ** 2)),
            "sse_log": float(res.residual_norm ** 2),
            "predict": lambda t: c * np.asarray(t, dtype=float) ** p}


# ---------------------------------------------------------------- the good route


def solve_qr(A, b) -> LSResult:
    """Solve the least squares problem through QR, which is the right default.

    With ``A = QR`` and ``Q`` having orthonormal columns, ``||b - Ax|| = ||Q^T b - Rx||`` plus a
    part that no ``x`` can touch, so the answer comes from one triangular solve. Crucially
    **A^T A is never formed**, so the condition number is not squared.

    This routine calls `numpy.linalg.qr` deliberately: lessons 30 and 31 build QR from scratch
    three different ways and check them against this, and the point of having it here is to
    give lesson 29 a correct answer to compare the normal equations against before QR has been
    taught.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    if b.size != m:
        raise ValueError(f"b has length {b.size}, but A has {m} rows")
    if m < n:
        raise ValueError(f"underdetermined system: {m} rows, {n} columns")

    from .lu import back_substitution

    Q, R = np.linalg.qr(A, mode="reduced")
    if np.any(np.abs(np.diag(R)) < 1e-14 * max(1.0, float(np.abs(R).max()))):
        raise np.linalg.LinAlgError(
            "A is numerically rank deficient; QR without pivoting cannot solve this. "
            "Use the pseudoinverse or a rank revealing factorization")
    x = back_substitution(R, Q.T @ b)
    r = b - A @ x
    return LSResult(x, r, float(np.linalg.norm(r)), n, "QR", float(np.linalg.cond(A)),
                    float(np.linalg.cond(A)) ** 2)


def condition_squaring(A) -> dict:
    """kappa(A) against kappa(A^T A), and the digits each leaves.

    The single number that decides whether the normal equations are usable. Returns both
    condition numbers, their ratio, and how many correct digits a backward stable solve can
    hope for in each case.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    ka = float(np.linalg.cond(A))
    kn = float(np.linalg.cond(A.T @ A))
    u = float(np.finfo(float).eps) / 2
    digits = lambda k: max(0.0, -np.log10(min(1.0, k * u)))
    return {"kappa_A": ka, "kappa_normal": kn, "ratio": kn / ka if ka else float("nan"),
            "digits_qr": digits(ka), "digits_normal": digits(kn)}


def hilbert_least_squares(m: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A tall Hilbert-like design matrix, its exact coefficient vector, and the data.

    ``A[i,j] = 1/(i+j+1)`` for an m by n block, which inherits the Hilbert matrix's exponential
    conditioning while being rectangular. Returns ``(A, x_true, b)`` with ``b = A x_true``
    exactly representable where possible, so a lesson has a known answer to measure against.
    """
    m, n = int(m), int(n)
    if m < n or n < 1:
        raise ValueError(f"need m >= n >= 1, got m = {m}, n = {n}")
    A = 1.0 / (np.arange(m)[:, None] + np.arange(n)[None, :] + 1.0)
    x_true = np.ones(n)
    return A, x_true, A @ x_true


def graded_design(m: int, n: int, kappa: float, rng=None) -> np.ndarray:
    """An m by n matrix with orthonormal-ish factors and a prescribed condition number.

    Built as ``U S V^T`` with singular values spread geometrically from 1 to 1/kappa, so the
    conditioning is exactly what was asked for rather than approximately. Every dimension comes
    from the arguments.
    """
    m, n = int(m), int(n)
    if m < n or n < 1:
        raise ValueError(f"need m >= n >= 1, got m = {m}, n = {n}")
    if kappa < 1.0:
        raise ValueError(f"a condition number cannot be below 1, got {kappa}")
    rng = np.random.default_rng() if rng is None else rng
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.geomspace(1.0, 1.0 / kappa, n) if n > 1 else np.array([1.0])
    return (U * s) @ V.T


# ---------------------------------------------------------------- the SVD route


def solve_svd(A, b, rcond: float | None = None) -> LSResult:
    """Least squares through the SVD, which works when QR does not.

    With ``A = U S V^T``, the minimiser is ``V S^+ U^T b`` where ``S^+`` inverts the singular
    values above a threshold and sets the rest to zero. That threshold is the whole point: it
    is where "rank deficient" gets a definition, and below it the routine returns the
    **minimum norm** solution instead of refusing.

    ``rcond`` is the relative threshold, defaulting to ``max(m, n) * u``, the same convention
    `numpy.linalg.lstsq` uses. Singular values below ``rcond * s_max`` are treated as zero.

    Costs about ``2mn^2 + 11n^3`` flops, several times a QR, and it is the right choice
    whenever the rank is in doubt. Lesson 33 is about deciding that.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    if b.size != m:
        raise ValueError(f"b has length {b.size}, but A has {m} rows")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if rcond is None:
        rcond = max(m, n) * float(np.finfo(float).eps)
    cutoff = rcond * (s[0] if s.size else 0.0)
    keep = s > cutoff
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    x = Vt.T @ (s_inv * (U.T @ b))
    r = b - A @ x
    kappa = float(s[0] / s[keep][-1]) if np.any(keep) else float("inf")
    return LSResult(x, r, float(np.linalg.norm(r)), int(keep.sum()), "SVD",
                    kappa, kappa ** 2)


def pseudoinverse(A, rcond: float | None = None) -> np.ndarray:
    """The Moore-Penrose pseudoinverse ``A^+ = V S^+ U^T``.

    It is the unique matrix satisfying the four Penrose conditions

        A A^+ A = A,   A^+ A A^+ = A^+,   (A A^+)^T = A A^+,   (A^+ A)^T = A^+ A,

    and those four are what make it "the" generalised inverse rather than one of many. For a
    full column rank A it equals ``(A^T A)^-1 A^T``, and unlike that formula it survives rank
    deficiency.

    **Do not form it to solve a system.** Like ``inv(A)`` in lesson 19, it costs more than the
    solve and is less accurate. It is here so the conditions can be checked and so lesson 33 can
    look at what it does to a rank deficient problem.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if rcond is None:
        rcond = max(m, n) * float(np.finfo(float).eps)
    keep = s > rcond * (s[0] if s.size else 0.0)
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    return Vt.T @ np.diag(s_inv) @ U.T


def penrose_residuals(A, A_plus) -> dict:
    """How far a candidate pseudoinverse is from satisfying the four Penrose conditions.

    All four are needed. Dropping the two symmetry conditions leaves a *generalised* inverse,
    of which there are infinitely many; the symmetry is what picks out the one that also solves
    the least squares problem.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    G = np.atleast_2d(np.asarray(A_plus, dtype=float))
    scale_a = float(np.linalg.norm(A, 2)) or 1.0
    scale_g = float(np.linalg.norm(G, 2)) or 1.0
    return {
        "A G A = A": float(np.linalg.norm(A @ G @ A - A, 2)) / scale_a,
        "G A G = G": float(np.linalg.norm(G @ A @ G - G, 2)) / scale_g,
        "(A G)^T = A G": float(np.linalg.norm((A @ G).T - A @ G, 2)),
        "(G A)^T = G A": float(np.linalg.norm((G @ A).T - G @ A, 2)),
    }


def minimum_norm_solution(A, b, rcond: float | None = None) -> np.ndarray:
    """The solution of smallest norm, which is what the pseudoinverse picks out.

    When A is rank deficient or underdetermined the least squares problem has infinitely many
    minimisers, differing by anything in the null space of A. They all give the same residual,
    so the residual cannot choose between them. ``A^+ b`` picks the one with the smallest
    ``||x||``, which is a **choice** and not a consequence, and it is the standard one because
    it is the limit of Tikhonov regularization as the penalty goes to zero.
    """
    return solve_svd(A, b, rcond).x


# ---------------------------------------------------------------- conditioning, properly


def least_squares_conditioning(A, b) -> dict:
    """The condition numbers that actually govern a least squares problem.

    There are **four**, not one, because there are two things that can be perturbed (A and b)
    and two answers to watch (x and the residual r). Quoting ``kappa(A)`` alone hides three of
    them.

    Writing ``theta`` for the angle between b and the range of A, and
    ``eta = ||A|| ||x|| / ||A x||`` for how much of A's size the solution actually uses:

    ==========================  ====================================
    perturb b, watch x          ``kappa / (eta cos theta)``
    perturb b, watch r          ``1 / cos theta``
    perturb A, watch x          ``kappa + kappa^2 tan(theta) / eta``
    perturb A, watch r          ``kappa``
    ==========================  ====================================

    **The third is the one to notice.** It carries a ``kappa^2``, so a least squares problem
    with a large residual is genuinely as sensitive as the normal equations suggest, and that
    sensitivity is a property of the PROBLEM rather than of any algorithm. Lesson 32 measures
    both regimes on the same matrix.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    if b.size != m:
        raise ValueError(f"b has length {b.size}, but A has {m} rows")

    s = np.linalg.svd(A, compute_uv=False)
    kappa = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    norm_A = float(s[0])
    norm_x = float(np.linalg.norm(x))
    norm_Ax = float(np.linalg.norm(A @ x))
    theta = angle_to_range(A, b)
    cos_t = float(np.cos(theta))
    tan_t = float(np.tan(theta))
    eta = (norm_A * norm_x / norm_Ax) if norm_Ax > 0 else float("inf")

    return {"kappa": kappa, "theta": float(theta), "eta": eta,
            "cos_theta": cos_t, "tan_theta": tan_t,
            "b_to_x": kappa / (eta * cos_t) if eta and cos_t else float("inf"),
            "b_to_r": 1.0 / cos_t if cos_t else float("inf"),
            "A_to_x": kappa + kappa * kappa * tan_t / eta if eta else float("inf"),
            "A_to_r": kappa}


def measure_sensitivity(A, b, n_trials: int = 40, level: float = 1e-8, rng=None) -> dict:
    """Perturb A and b and see how far x and r actually move.

    The theory in `least_squares_conditioning` is an upper bound over perturbation directions,
    and a bound is worth checking against what a random direction does. Sizes come from A, and
    the reported figures are medians over ``n_trials`` random directions.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    rng = np.random.default_rng() if rng is None else rng
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    r0 = b - A @ x0
    nx0 = max(float(np.linalg.norm(x0)), 1e-300)
    # The residual sensitivities are normalised by ||b||, not by ||r||. When the fit is nearly
    # exact ||r|| is nearly zero, and a relative change in a nearly zero quantity is unbounded
    # for reasons that have nothing to do with conditioning.
    nr0 = max(float(np.linalg.norm(b)), 1e-300)
    out = {"b_to_x": [], "b_to_r": [], "A_to_x": [], "A_to_r": []}
    for _ in range(int(n_trials)):
        db = level * np.linalg.norm(b) * rng.standard_normal(b.size) / np.sqrt(b.size)
        rel_b = np.linalg.norm(db) / np.linalg.norm(b)
        x1 = np.linalg.lstsq(A, b + db, rcond=None)[0]
        out["b_to_x"].append((np.linalg.norm(x1 - x0) / nx0) / rel_b)
        out["b_to_r"].append((np.linalg.norm((b + db - A @ x1) - r0) / nr0) / rel_b)

        dA = level * np.linalg.norm(A, 2) * rng.standard_normal(A.shape) / np.sqrt(A.size)
        rel_a = np.linalg.norm(dA, 2) / np.linalg.norm(A, 2)
        x2 = np.linalg.lstsq(A + dA, b, rcond=None)[0]
        out["A_to_x"].append((np.linalg.norm(x2 - x0) / nx0) / rel_a)
        out["A_to_r"].append((np.linalg.norm((b - (A + dA) @ x2) - r0) / nr0) / rel_a)
    return {k: float(np.median(v)) for k, v in out.items()}


# ---------------------------------------------------------------- regularization


def tikhonov(A, b, lam: float) -> np.ndarray:
    """Solve ``min ||b - Ax||^2 + lam^2 ||x||^2``, which is always well posed.

    Adding the penalty replaces the singular values ``s_i`` by ``s_i / (s_i^2 + lam^2)`` in the
    solution, so the small ones stop being amplified. That fixes rank deficiency and ill
    conditioning at once, at the price of **biasing** the answer towards zero.

    Solved through the augmented system

        [ A     ]        [ b ]
        [ lam I ] x  ~=  [ 0 ]

    rather than by forming ``A^T A + lam^2 I``, because the augmented form is a plain least
    squares problem and lesson 29 established what forming the normal matrix costs.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    if b.size != m:
        raise ValueError(f"b has length {b.size}, but A has {m} rows")
    if lam < 0.0:
        raise ValueError(f"the penalty must be non-negative, got {lam}")
    stacked = np.vstack([A, float(lam) * np.eye(n)])
    padded = np.concatenate([b, np.zeros(n)])
    return np.linalg.lstsq(stacked, padded, rcond=None)[0]


def l_curve(A, b, lambdas=None) -> dict:
    """The residual norm against the solution norm, over a range of penalties.

    Plotted log against log this traces an **L**, and the corner is the usual choice: to its
    left the residual grows fast for no reduction in ``||x||``, to its right ``||x||`` grows
    fast for no reduction in the residual. The corner is where neither is being traded away
    cheaply.

    Returns both norms and the curvature, so the corner can be located rather than eyeballed.
    The default range of ``lambdas`` spans the singular values of A, so it adapts to the problem
    rather than assuming a scale.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    s = np.linalg.svd(A, compute_uv=False)
    if lambdas is None:
        top = float(s[0])
        bottom = max(float(s[-1]), top * 1e-14)
        lambdas = np.geomspace(bottom * 1e-2, top, 60)
    lambdas = np.asarray(lambdas, dtype=float)

    res, sol = [], []
    for lam in lambdas:
        x = tikhonov(A, b, float(lam))
        res.append(float(np.linalg.norm(b - A @ x)))
        sol.append(float(np.linalg.norm(x)))
    res, sol = np.array(res), np.array(sol)

    # curvature of the log-log curve, by finite differences
    lr = np.log(np.maximum(res, 1e-300))
    lsn = np.log(np.maximum(sol, 1e-300))
    d1r, d1s = np.gradient(lr), np.gradient(lsn)
    d2r, d2s = np.gradient(d1r), np.gradient(d1s)
    denom = np.maximum((d1r ** 2 + d1s ** 2) ** 1.5, 1e-300)
    curvature = np.abs(d1r * d2s - d1s * d2r) / denom
    corner = int(np.argmax(curvature[1:-1]) + 1) if curvature.size > 2 else 0

    return {"lambdas": lambdas, "residual_norms": res, "solution_norms": sol,
            "curvature": curvature, "corner_index": corner,
            "corner_lambda": float(lambdas[corner])}


def truncated_svd(A, b, rank: int) -> np.ndarray:
    """Keep only the largest ``rank`` singular values. The other way to regularize.

    Tikhonov damps the small singular values smoothly; truncation discards them outright. The
    two agree when the spectrum has a clear gap and differ when it does not, which is exactly
    the situation lesson 33 has to decide.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    rank = int(rank)
    if not 1 <= rank <= min(A.shape):
        raise ValueError(f"rank must lie between 1 and {min(A.shape)}, got {rank}")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[:rank].T @ ((U[:, :rank].T @ b) / s[:rank])
