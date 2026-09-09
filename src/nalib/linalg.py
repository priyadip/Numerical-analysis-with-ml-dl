"""Vector and matrix norms, and the three ways to read a matrix product.

This is the vocabulary module for Parts 3 to 6. Almost every error bound in the rest of the
course is a statement about a norm, so the definitions have to be exact and the properties
have to be things you can check numerically rather than things you remember.

Two ideas here do the heavy lifting later.

**A norm turns a bound on the input into a bound on the output.** If you know
``||dx|| <= eps`` then ``||A dx|| <= ||A|| eps``, and that single line is the machinery behind
the condition number of a linear system (lesson 19), the stability analysis of elimination
(lesson 18), and every convergence proof in Part 4.

**A matrix product can be computed in three mathematically identical ways** that touch memory
completely differently. Lesson 08 measured why that matters. Here it is stated as an identity
so later lessons can pick whichever view makes a proof easiest.

Everything is checked against ``numpy.linalg.norm`` in ``tests/test_linalg.py``, and the
axioms are verified numerically rather than asserted.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- vector norms


def vector_norm(x, p=2) -> float:
    """The p-norm of a vector, from scratch.

    ``p`` may be any real number at least 1, or ``numpy.inf``.

        ||x||_1     = sum |x_i|                 the taxicab norm
        ||x||_2     = sqrt(sum x_i^2)           the Euclidean norm
        ||x||_p     = (sum |x_i|^p)^(1/p)
        ||x||_inf   = max |x_i|                 the max norm

    The 2-norm here is computed the naive way, which can overflow for genuinely huge entries.
    `numpy.linalg.norm` scales first to avoid that. The difference is measured in lesson 15,
    and it is a clean small example of lesson 06's point that two implementations of the same
    formula are not the same algorithm.
    """
    x = np.asarray(x, dtype=float).ravel()
    if p == np.inf:
        return float(np.max(np.abs(x))) if x.size else 0.0
    if p == 1:
        return float(np.sum(np.abs(x)))
    if p < 1:
        raise ValueError("p must be at least 1, or numpy.inf, for this to be a norm")
    return float(np.sum(np.abs(x) ** p) ** (1.0 / p))


def normalize(x, p=2) -> np.ndarray:
    """x / ||x||_p. Returns the zero vector unchanged rather than dividing by zero."""
    x = np.asarray(x, dtype=float)
    n = vector_norm(x, p)
    return x.copy() if n == 0.0 else x / n


def unit_ball_points(p=2, n: int = 400) -> np.ndarray:
    """Points on the unit circle of the p-norm in R^2, for drawing.

    The shape of the unit ball *is* the norm. The 1-norm gives a diamond, the 2-norm a
    circle, the infinity-norm a square, and everything between interpolates. Lesson 15 plots
    these, because seeing the shapes makes norm equivalence obvious.
    """
    theta = np.linspace(0, 2 * np.pi, n)
    d = np.column_stack([np.cos(theta), np.sin(theta)])
    scale = np.array([vector_norm(v, p) for v in d])
    return d / scale[:, None]


# ---------------------------------------------------------------- matrix norms


def matrix_norm(A, p=2) -> float:
    """An induced (operator) matrix norm, or the Frobenius norm with ``p='fro'``.

    The induced norm is defined as the worst stretching the matrix can do:

        ||A||_p = max over x nonzero of ||Ax||_p / ||x||_p.

    Three of these have closed forms, which is why they are the ones everyone uses:

        ||A||_1    = max column sum of |a_ij|
        ||A||_inf  = max row sum of |a_ij|
        ||A||_2    = largest singular value  (the hard one, needs an SVD)
        ||A||_fro  = sqrt(sum of all a_ij^2)  (NOT an induced norm)

    The 1 and infinity norms are computable in O(n^2) by inspection. The 2-norm costs an SVD,
    which is O(n^3), and that cost is exactly why condition number *estimation* (lesson 22)
    exists.

    Frobenius is included because it appears constantly in later parts, most importantly in
    the Eckart-Young theorem (lesson 43) and in Broyden's least-change derivation (lesson 14).
    It is submultiplicative but it is not induced by any vector norm, and lesson 15 shows the
    one-line proof: an induced norm always has ||I|| = 1, and ||I||_fro = sqrt(n).
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if p == "fro":
        return float(np.sqrt(np.sum(A * A)))
    if p == 1:
        return float(np.max(np.sum(np.abs(A), axis=0)))      # max column sum
    if p == np.inf:
        return float(np.max(np.sum(np.abs(A), axis=1)))      # max row sum
    if p == 2:
        return float(np.linalg.svd(A, compute_uv=False)[0])
    raise ValueError("supported: 1, 2, numpy.inf, 'fro'")


def induced_norm_by_search(A, p=2, n_samples: int = 20000, seed: int = 42) -> float:
    """Estimate ||A||_p by sampling the definition directly.

    This exists to *demonstrate the definition*, not to compute anything. It draws random unit
    vectors, measures ||Ax||_p, and reports the largest ratio found.

    It always **underestimates**, because a random direction almost never hits the maximizing
    one, and it gets worse as the dimension grows. Lesson 15 measures that decay, which makes
    the point that the closed forms above are not a convenience but a necessity.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((A.shape[1], n_samples))
    best = 0.0
    for j in range(n_samples):
        x = X[:, j]
        nx = vector_norm(x, p)
        if nx == 0:
            continue
        best = max(best, vector_norm(A @ x, p) / nx)
    return float(best)


def spectral_radius(A) -> float:
    """rho(A), the largest eigenvalue magnitude.

    Not a norm. It fails the triangle inequality, and a nonzero matrix can have rho(A) = 0,
    which any norm forbids. But it satisfies rho(A) <= ||A|| for every induced norm, and
    Gelfand's formula says ||A^k||^(1/k) -> rho(A), so it is the quantity that decides whether
    a matrix power tends to zero.

    That is why it, and not any norm, is the convergence criterion for every stationary
    iterative method in Part 4, and for fixed point iteration on systems in lesson 14.
    """
    return float(np.max(np.abs(np.linalg.eigvals(np.atleast_2d(np.asarray(A, dtype=float))))))


def condition_number(A, p=2) -> float:
    """kappa_p(A) = ||A||_p ||A^-1||_p, the amplification factor for solving Ax = b.

    Infinite for a singular matrix. This forms the inverse explicitly, which is fine for the
    small matrices in the lessons and wrong for anything large. Lesson 22 covers estimating it
    at O(n^2) once a factorization exists, which is what LAPACK actually does.

    In the 2-norm this equals sigma_max / sigma_min, proved in lesson 41.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[0] != A.shape[1]:
        raise ValueError("condition number of a square matrix only; see lesson 32 for the rest")
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float("inf")
    return matrix_norm(A, p) * matrix_norm(Ainv, p)


# ---------------------------------------------------------------- the three views


def matvec_by_columns(A, x) -> np.ndarray:
    """Ax computed as a linear combination of the columns of A.

        Ax = x_1 a_1 + x_2 a_2 + ... + x_n a_n

    This is the single most useful way to read a matrix-vector product, and Trefethen and Bau
    open their book with it. Consequences that fall out immediately:

    - **The range of A is the span of its columns.** Ax is a combination of columns, so it
      cannot leave their span. That sentence is the whole of the least squares setup (29).
    - **Ax = b is solvable exactly when b lies in that span.**
    - **Rank is the number of independent columns**, not a fact about determinants.

    Written with an explicit loop so the definition is visible. Use ``A @ x`` in real code.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    x = np.asarray(x, dtype=float).ravel()
    out = np.zeros(A.shape[0])
    for j in range(A.shape[1]):
        out += x[j] * A[:, j]
    return out


def matmul_by_columns(A, B) -> np.ndarray:
    """AB computed one column at a time: column j of AB is A times column j of B.

    So **AB is the matrix whose columns are A applied to each column of B**. Every column of
    the product lies in the range of A, which immediately gives rank(AB) <= rank(A).
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    out = np.zeros((A.shape[0], B.shape[1]))
    for j in range(B.shape[1]):
        out[:, j] = A @ B[:, j]
    return out


def matmul_by_outer_products(A, B) -> np.ndarray:
    """AB as a sum of rank-one outer products: sum_k (column k of A)(row k of B).

        AB = a_1 b_1^T + a_2 b_2^T + ... + a_k b_k^T

    This view is the one that matters most later. It says **every matrix product is a sum of
    rank-one pieces**, which is exactly the form the SVD puts a matrix into (lesson 41), the
    form low-rank approximation truncates (lesson 43), and the form Gram-Schmidt builds up
    column by column (lesson 30).

    It is also the memory access pattern of the blocked BLAS-3 kernels measured in lesson 08.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    out = np.zeros((A.shape[0], B.shape[1]))
    for k in range(A.shape[1]):
        out += np.outer(A[:, k], B[k, :])
    return out


def matmul_by_inner_products(A, B) -> np.ndarray:
    """AB entry by entry: (AB)_ij is row i of A dotted with column j of B.

    The definition from a first linear algebra course. Correct, and the least useful of the
    three for reasoning, because it describes the answer one number at a time and hides the
    structure. It is also the worst memory access pattern of the three in row-major storage.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    out = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            out[i, j] = A[i, :] @ B[:, j]
    return out


# ---------------------------------------------------------------- norm properties


def check_norm_axioms(norm_fn, vectors, scalars=(-2.5, 0.0, 1.0, 3.7)) -> dict:
    """Verify the three norm axioms numerically on supplied vectors.

    1. **Positivity**: ||x|| >= 0, and ||x|| = 0 only when x = 0.
    2. **Absolute homogeneity**: ||c x|| = |c| ||x||.
    3. **Triangle inequality**: ||x + y|| <= ||x|| + ||y||.

    Returns the worst violation of each, so a caller can assert they are at roundoff. This is
    how lesson 15 checks that ``vector_norm`` really is a norm rather than merely asserting it.
    """
    vs = [np.asarray(v, dtype=float).ravel() for v in vectors]
    if len({v.size for v in vs}) > 1:
        raise ValueError("all vectors must have the same length, since the triangle "
                         "inequality compares x + y")
    worst_pos = 0.0
    worst_hom = 0.0
    worst_tri = 0.0

    for v in vs:
        n = norm_fn(v)
        worst_pos = max(worst_pos, -min(n, 0.0))
        if np.all(v == 0):
            worst_pos = max(worst_pos, abs(n))
        for c in scalars:
            lhs, rhs = norm_fn(c * v), abs(c) * n
            worst_hom = max(worst_hom, abs(lhs - rhs) / max(rhs, 1.0))

    for a in vs:
        for b in vs:
            slack = norm_fn(a) + norm_fn(b) - norm_fn(a + b)
            worst_tri = max(worst_tri, -min(slack, 0.0) / max(norm_fn(a + b), 1.0))

    return {"positivity": worst_pos, "homogeneity": worst_hom, "triangle": worst_tri}


def norm_equivalence_constants(n: int, p: int, q: int) -> tuple[float, float]:
    """The sharp constants in  c ||x||_q <= ||x||_p <= C ||x||_q  on R^n.

    All norms on a finite-dimensional space are equivalent, meaning each is bounded by a
    constant times any other, with constants depending only on the dimension. That is why
    "converges in norm" needs no qualifier in finite dimensions: convergence in one norm is
    convergence in all of them.

    The constants **grow with n**, which is the practical catch. Supported pairs are those
    among 1, 2 and infinity, which is all the lessons need.
    """
    table = {
        (1, 2): (1.0, np.sqrt(n)),
        (2, 1): (1.0 / np.sqrt(n), 1.0),
        (2, np.inf): (1.0, np.sqrt(n)),
        (np.inf, 2): (1.0 / np.sqrt(n), 1.0),
        (1, np.inf): (1.0, float(n)),
        (np.inf, 1): (1.0 / n, 1.0),
    }
    if (p, q) not in table:
        raise ValueError("supported pairs are drawn from 1, 2 and numpy.inf")
    return table[(p, q)]


def submultiplicativity_slack(A, B, p=2) -> float:
    """||A|| ||B|| - ||AB||, which submultiplicativity says is never negative.

    Every induced norm satisfies ||AB|| <= ||A|| ||B||, and so does Frobenius. The proof for
    induced norms is one line: ||ABx|| <= ||A|| ||Bx|| <= ||A|| ||B|| ||x||.

    The slack is usually large. Lesson 15 measures it, because the size of the gap is the
    reason repeated use of this inequality gives error bounds that are correct but very
    pessimistic, which is a theme running through the whole course.
    """
    return float(matrix_norm(A, p) * matrix_norm(B, p) - matrix_norm(np.asarray(A) @ np.asarray(B), p))
