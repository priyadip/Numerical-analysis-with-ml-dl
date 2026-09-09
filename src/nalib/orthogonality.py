"""Orthogonal vectors and matrices, and projectors.

This is the smallest module in the course and one of the most important, because it holds the
single fact that every stable algorithm in Parts 3 to 6 is built on:

    ||Qx||_2 = ||x||_2   for orthogonal Q,   so   kappa_2(Q) = 1.

An orthogonal transformation **cannot amplify error**. Multiply by anything else and whatever
error you were carrying gets scaled by up to ||A||; multiply by an orthogonal matrix and it is
carried across unchanged. That is why QR (lessons 30, 31) is built from reflections and
rotations, why the SVD (lesson 41) is the decomposition everyone trusts, and why the stable
eigenvalue algorithms (lesson 42) work by orthogonal similarity transformations.

The second half of the module is **projectors**. A projector is a matrix with P^2 = P, and
there are two kinds. Orthogonal projectors have ||P||_2 = 1 exactly. Oblique ones have no
bound at all, and their norm blows up as the two subspaces approach each other. Lesson 16
measures that blow-up, and it is the quantitative reason "project orthogonally" is a rule.

Used by lessons 16, 29, 30, 31, 32, 39 and 41.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- inner products


def inner(x, y) -> float:
    """The real Euclidean inner product x^T y, written out.

    Everything about orthogonality is a statement about this number. Two vectors are
    orthogonal when it is zero, the norm is the square root of it with y = x, and the angle
    between them is its arccosine after normalising.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("inner product needs vectors of the same length")
    return float(np.sum(x * y))


def angle_between(x, y, degrees: bool = False) -> float:
    """The angle between two vectors, from cos(theta) = x^T y / (||x|| ||y||).

    Clipped into [-1, 1] before the arccos, because roundoff can push the cosine of a
    near-parallel pair to 1 + 1e-16 and `arccos` returns nan for that. A one-line fix for a
    real failure that catches people out.
    """
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx == 0.0 or ny == 0.0:
        raise ValueError("the angle to the zero vector is undefined")
    c = np.clip(inner(x, y) / (nx * ny), -1.0, 1.0)
    theta = float(np.arccos(c))
    return np.degrees(theta) if degrees else theta


def is_orthogonal_set(vectors, tol: float = 1e-12) -> bool:
    """True when every distinct pair in the list is orthogonal to within ``tol``."""
    vs = [np.asarray(v, dtype=float).ravel() for v in vectors]
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if abs(inner(vs[i], vs[j])) > tol:
                return False
    return True


def is_orthonormal_set(vectors, tol: float = 1e-12) -> bool:
    """Orthogonal, and every vector has unit length."""
    vs = [np.asarray(v, dtype=float).ravel() for v in vectors]
    if not is_orthogonal_set(vs, tol):
        return False
    return all(abs(float(np.linalg.norm(v)) - 1.0) <= tol for v in vs)


def expand_in_basis(x, Q) -> np.ndarray:
    """Coefficients of x in the orthonormal basis given by the columns of Q.

    For an orthonormal basis the coefficients are **just inner products**:

        x = sum_j (q_j^T x) q_j,      so the coefficient vector is Q^T x.

    No linear system is solved. For a general basis you would have to solve ``B c = x`` at
    O(n^3); orthonormality reduces that to a matrix-vector product at O(n^2), and it does so
    stably. This one saving is why orthonormal bases are worth constructing.

    When the columns of Q span the whole space this reconstructs x exactly. When they span a
    subspace, ``Q @ expand_in_basis(x, Q)`` is the orthogonal projection of x onto it.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    x = np.asarray(x, dtype=float).ravel()
    return Q.T @ x


# ---------------------------------------------------------------- orthogonal matrices


def orthogonality_error(Q) -> float:
    """||Q^T Q - I||_2, how far Q is from having orthonormal columns.

    The standard measured quantity in every QR experiment in this course. For an exactly
    orthogonal matrix it is zero; for one computed in floating point it should be O(u), and
    when it is not, the algorithm that produced it is unstable. Lesson 30 uses precisely this
    to separate classical from modified Gram-Schmidt.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    return float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), 2))


def is_orthogonal_matrix(Q, tol: float = 1e-10) -> bool:
    """True when Q^T Q = I to within ``tol``. Requires at least as many rows as columns."""
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    if Q.shape[0] < Q.shape[1]:
        return False
    return orthogonality_error(Q) <= tol


def random_orthogonal(n: int, rng=None) -> np.ndarray:
    """A random orthogonal matrix, uniformly distributed over the orthogonal group.

    Built by QR of a Gaussian matrix, with the sign correction that makes the distribution
    actually uniform (Haar). Without the sign fix the result is still orthogonal but biased,
    which matters when you are averaging over many draws.

    Used throughout the tests as a source of exactly-conditioned matrices: any A = U S V^T
    built from two of these has singular values you chose, so you can construct a matrix with
    a condition number you specify exactly.
    """
    rng = np.random.default_rng() if rng is None else rng
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    return Q * np.sign(np.diag(R))


def rotation_2d(theta: float) -> np.ndarray:
    """The 2 by 2 rotation through angle theta. Determinant +1."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def reflection_2d(theta: float) -> np.ndarray:
    """Reflection in the line through the origin at angle theta. Determinant -1.

    Every 2 by 2 orthogonal matrix is one of these two, and the determinant tells you which:
    +1 is a rotation, -1 is a reflection. In n dimensions the same split holds, and the
    Householder reflector of lesson 31 is the n-dimensional version of this one.
    """
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    return np.array([[c, s], [s, -c]])


def householder_reflector(x) -> np.ndarray:
    """The reflector H = I - 2vv^T/(v^Tv) that maps x onto a multiple of e_1.

    This is the workhorse of lesson 31, included here because it is the most useful concrete
    orthogonal matrix there is, and lesson 16 wants a real example rather than a random one.

    The sign choice ``v = x + sign(x_1)||x|| e_1`` is not cosmetic. The other sign gives
    ``v = x - ||x|| e_1``, which is a difference of two nearly equal vectors when x already
    points near e_1, and that is lesson 05's catastrophic cancellation. Choosing the sign that
    adds is a one-character fix for a real instability.
    """
    x = np.asarray(x, dtype=float).ravel()
    v = x.copy()
    nx = float(np.linalg.norm(x))
    if nx == 0.0:
        return np.eye(x.size)
    v[0] += np.sign(x[0]) * nx if x[0] != 0 else nx
    vv = float(v @ v)
    if vv == 0.0:
        return np.eye(x.size)
    return np.eye(x.size) - 2.0 * np.outer(v, v) / vv


# ---------------------------------------------------------------- projectors


def is_projector(P, tol: float = 1e-10) -> bool:
    """True when P^2 = P, which is the whole definition.

    Idempotence says applying the projection twice does nothing more than applying it once,
    which is exactly what "already there" means geometrically.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    if P.shape[0] != P.shape[1]:
        return False
    return float(np.linalg.norm(P @ P - P, 2)) <= tol * max(1.0, float(np.linalg.norm(P, 2)))


def is_orthogonal_projector(P, tol: float = 1e-10) -> bool:
    """True when P is a projector **and** symmetric.

    Those two conditions together are equivalent to "projects orthogonally onto its range",
    and symmetry is the only thing separating the safe kind from the dangerous kind.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    if not is_projector(P, tol):
        return False
    return float(np.linalg.norm(P - P.T, 2)) <= tol * max(1.0, float(np.linalg.norm(P, 2)))


def orthogonal_projector(Q) -> np.ndarray:
    """P = QQ^T, the orthogonal projector onto the column space of an orthonormal Q.

    Check the two properties directly. P^2 = QQ^TQQ^T = Q(Q^TQ)Q^T = QIQ^T = P, using
    orthonormality in the middle. And P^T = (QQ^T)^T = QQ^T = P. So it is a symmetric
    projector, which is the definition of orthogonal.

    Its norm is exactly 1 whenever Q has at least one column, and exactly 0 for the empty
    case. Never more. That is the whole reason this is the safe kind.

    **Never form P explicitly in real code.** It costs O(mn^2) to build and O(m^2) per apply,
    while ``Q @ (Q.T @ x)`` costs O(mn) and is more accurate. P exists here so that lesson 16
    can measure its properties.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    return Q @ Q.T


def rank_one_projector(q) -> np.ndarray:
    """P = qq^T for a unit vector q: project onto the single direction q.

    The simplest nonzero projector there is, and the building block of Gram-Schmidt, which
    subtracts exactly this from a vector for each basis direction already found.
    """
    q = np.asarray(q, dtype=float).ravel()
    n = float(np.linalg.norm(q))
    if n == 0.0:
        raise ValueError("cannot project onto the zero direction")
    q = q / n
    return np.outer(q, q)


def complementary_projector(P) -> np.ndarray:
    """I - P, which projects onto the complementary subspace.

    If P projects onto S along T, then I - P projects onto T along S. It is a projector
    because (I-P)^2 = I - 2P + P^2 = I - 2P + P = I - P.

    Every vector splits uniquely as x = Px + (I-P)x, and when P is orthogonal those two pieces
    are orthogonal to each other, so Pythagoras applies:

        ||x||^2 = ||Px||^2 + ||(I-P)x||^2.

    That identity is the whole of least squares (lesson 29): minimising the residual means
    making the second term as small as possible, and it is smallest when the residual is the
    orthogonal complement piece.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    return np.eye(P.shape[0]) - P


def oblique_projector(A, B) -> np.ndarray:
    """The projector onto range(A) **along** the orthogonal complement of range(B).

        P = A (B^T A)^-1 B^T

    Idempotent, so it is a genuine projector, but **not symmetric** unless range(A) and
    range(B) coincide, in which case it collapses to the orthogonal projector.

    Its norm is ``1 / cos(theta)`` where theta is the largest principal angle between the two
    subspaces, so it grows without bound as they approach being orthogonal to each other.
    Lesson 16 measures exactly that.

    Oblique projectors are not a curiosity. Petrov-Galerkin methods project this way, and
    GMRES (lesson 27) is one, which is why its residual norm behaviour is harder to analyse
    than the symmetric case.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    return A @ np.linalg.solve(B.T @ A, B.T)


def projection_residual(x, Q) -> np.ndarray:
    """r = x - QQ^T x, the part of x that the subspace cannot represent.

    The defining property, proved in lesson 16 and checked in the tests: **r is orthogonal to
    every column of Q**, so Q^T r = 0. Geometrically the shortest way from a point to a
    subspace is the perpendicular one, and this is the algebraic form of that statement.

    It is also the normal equations of least squares in disguise. Q^T(x - Qc) = 0 rearranges
    to c = Q^T x, and lesson 29 does the same thing for a non-orthonormal basis.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    x = np.asarray(x, dtype=float).ravel()
    return x - Q @ (Q.T @ x)


def principal_angles(A, B) -> np.ndarray:
    """All principal angles between two subspaces, in radians, smallest first.

    Computed from the singular values of Q_A^T Q_B, where the Q's are orthonormal bases for
    the two column spaces. Those singular values are the **cosines** of the angles, sorted
    descending, so the returned angles come out ascending.

    Which end you want depends on the question:

    - the **smallest** angle says how close the subspaces come to sharing a direction,
    - the **largest** angle says how close they come to being orthogonal, and it is the one
      that controls an oblique projector's norm, since ``||P||_2 = 1 / cos(theta_max)``.

    The same quantity measures how well a Krylov subspace captures an eigenvector in lesson 39.
    """
    QA = np.linalg.qr(np.atleast_2d(np.asarray(A, dtype=float)))[0]
    QB = np.linalg.qr(np.atleast_2d(np.asarray(B, dtype=float)))[0]
    s = np.linalg.svd(QA.T @ QB, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def largest_principal_angle(A, B) -> float:
    """The largest principal angle between two subspaces, in radians.

    This is the theta in ``||P||_2 = 1 / cos(theta)`` for the oblique projector onto range(A)
    along the complement of range(B). As the subspaces approach being orthogonal, theta
    approaches pi/2, the cosine approaches zero, and the projector norm blows up.
    """
    return float(principal_angles(A, B)[-1])
