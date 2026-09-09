"""Eigenvalue theory: localization, conditioning, and why no finite algorithm exists.

What this module is for
-----------------------
Everything before Part 6 solved ``A x = b``, where the matrix acts on a vector and the answer is
another vector. Eigenvalues ask a different question: for which scalars does ``A`` act like a
scalar? That question is **not** solvable by elimination, and the reason is a theorem rather
than a gap in anyone's cleverness.

**No finite algorithm can compute eigenvalues.** Every polynomial is the characteristic
polynomial of its companion matrix, so a finite eigenvalue algorithm using only arithmetic and
roots would give a finite solution formula for every polynomial, which Abel and Galois ruled out
for degree 5 and above. `companion_matrix` and `no_finite_algorithm` make that concrete.

So every eigenvalue method is **iterative**, and the first question is where to look. This
module supplies the localization results (Gerschgorin, Brauer), the conditioning theory
(Bauer-Fike, and the individual condition number ``1/|y^H x|``), and the one decomposition that
always exists (Schur).

**The recurring theme is that eigenvalues of a symmetric matrix are perfectly conditioned and
eigenvalues of a non-symmetric one need not be.** That single fact splits Part 6 in two, and
`bauer_fike_bound` against `eigenvalue_condition_numbers` measures the gap.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------- localization


@dataclass
class Disc:
    """A Gerschgorin disc: centre ``a_ii`` and radius the off-diagonal row sum."""

    centre: complex
    radius: float
    index: int

    def contains(self, z, tol: float = 1e-9) -> bool:
        return abs(complex(z) - self.centre) <= self.radius * (1.0 + tol) + tol


def gerschgorin_discs(A, by_columns: bool = False) -> list:
    """The discs ``|z - a_ii| <= sum_{j != i} |a_ij|``, one per row.

    **Gerschgorin's theorem**: every eigenvalue lies in the union. The proof is three lines and
    worth knowing, because it explains when the bound is tight. If ``A x = lambda x`` and
    ``|x_i|`` is the largest entry, then row ``i`` of the equation gives
    ``(lambda - a_ii) x_i = sum_{j != i} a_ij x_j``, so dividing by ``x_i`` and using
    ``|x_j/x_i| <= 1`` bounds ``|lambda - a_ii|`` by the row sum.

    **So the bound is tight exactly when the eigenvector is concentrated on one entry**, which
    is why it is sharp for a nearly diagonal matrix and loose for a matrix whose eigenvectors
    are spread out.

    ``by_columns=True`` uses column sums instead, which is Gerschgorin applied to ``A^T``. Since
    ``A`` and ``A^T`` have the same eigenvalues, **both sets of discs are valid and their
    intersection is valid too**, which is free extra information that is easy to forget.
    """
    A = np.atleast_2d(np.asarray(A))
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A is {A.shape}, eigenvalues need a square matrix")
    M = A.T if by_columns else A
    n = M.shape[0]
    out = []
    for i in range(n):
        radius = float(np.sum(np.abs(M[i, :])) - abs(M[i, i]))
        out.append(Disc(centre=complex(M[i, i]), radius=radius, index=i))
    return out


def brauer_ovals(A) -> list:
    """Brauer's Cassini ovals: ``|z - a_ii| |z - a_jj| <= r_i r_j`` for each pair ``i != j``.

    **Brauer's theorem is strictly sharper than Gerschgorin**, and the containment is a theorem
    rather than an observation: the union of the ovals is contained in the union of the discs.
    The cost is that there are ``n(n-1)/2`` ovals rather than ``n`` discs, and an oval is not a
    disc, so it is harder to draw and to test membership in.

    Returns a list of ``(centre_i, centre_j, r_i, r_j)`` tuples. Use `in_brauer_oval` to test a
    point, since the region is defined by an inequality rather than by a radius.
    """
    discs = gerschgorin_discs(A)
    n = len(discs)
    if n < 2:
        return []
    return [(discs[i].centre, discs[j].centre, discs[i].radius, discs[j].radius)
            for i in range(n) for j in range(i + 1, n)]


def in_brauer_oval(z, oval, tol: float = 1e-9) -> bool:
    """Is ``z`` inside one Cassini oval, ``|z - c_i| |z - c_j| <= r_i r_j``?"""
    c_i, c_j, r_i, r_j = oval
    z = complex(z)
    lhs = abs(z - c_i) * abs(z - c_j)
    rhs = r_i * r_j
    return lhs <= rhs * (1.0 + tol) + tol


def localization_report(A) -> dict:
    """Check both theorems against the actual eigenvalues, and measure how loose each is.

    Returns the eigenvalues, whether every one lies in the corresponding union, and the **area**
    of each region, which is the honest measure of how much the bound actually says. A region
    that contains the eigenvalues and also most of the complex plane has told you nothing.
    """
    A = np.atleast_2d(np.asarray(A))
    vals = np.linalg.eigvals(A)
    discs = gerschgorin_discs(A)
    cols = gerschgorin_discs(A, by_columns=True)
    ovals = brauer_ovals(A)

    in_rows = all(any(d.contains(v) for d in discs) for v in vals)
    in_cols = all(any(d.contains(v) for d in cols) for v in vals)
    in_ovals = all(any(in_brauer_oval(v, o) for o in ovals) for v in vals) if ovals else in_rows

    # An upper bound on the area of the union: the sum of the disc areas. Overlapping discs make
    # this an overestimate, and it is still the right order of magnitude for a comparison.
    area_rows = float(np.pi * sum(d.radius ** 2 for d in discs))
    area_cols = float(np.pi * sum(d.radius ** 2 for d in cols))
    spread = float(np.max(np.abs(vals[:, None] - vals[None, :]))) if vals.size > 1 else 0.0
    return {"eigenvalues": vals,
            "rows_contain_all": bool(in_rows),
            "columns_contain_all": bool(in_cols),
            "ovals_contain_all": bool(in_ovals),
            "row_area_bound": area_rows,
            "column_area_bound": area_cols,
            "max_row_radius": max((d.radius for d in discs), default=0.0),
            "eigenvalue_spread": spread}


def disjoint_disc_count(A) -> int:
    """How many Gerschgorin discs are disjoint from all the others.

    **This is where Gerschgorin stops being a crude bound and becomes exact.** If a group of
    ``k`` discs is disjoint from the rest, that group contains **exactly** ``k`` eigenvalues,
    counted with multiplicity. So a disc disjoint from all the others isolates one eigenvalue to
    within its radius, which is a genuine error bound rather than a region.

    The proof is a continuity argument: interpolate from ``diag(A)`` to ``A``, the eigenvalues
    move continuously, and they cannot cross between disjoint components.
    """
    discs = gerschgorin_discs(A)
    n = len(discs)
    alone = 0
    for i in range(n):
        if all(abs(discs[i].centre - discs[j].centre) > discs[i].radius + discs[j].radius
               for j in range(n) if j != i):
            alone += 1
    return alone


# ----------------------------------------------------------------------------- conditioning


def eigenvalue_condition_numbers(A) -> dict:
    """The individual condition number ``1 / |y_i^H x_i|`` for each eigenvalue.

    With right eigenvector ``x_i`` and left eigenvector ``y_i``, both normalised to unit length,
    a perturbation ``E`` moves ``lambda_i`` by about ``(y_i^H E x_i) / (y_i^H x_i)``, so the
    sensitivity is ``1/|y_i^H x_i| = 1/|cos(angle between them)|``.

    **Each eigenvalue has its own condition number**, which is the point. A matrix can have one
    eigenvalue determined to sixteen digits and another to none, and a single number like
    ``kappa(V)`` cannot express that. `bauer_fike_bound` gives the single number, and the gap
    between them is measured in lesson 35.

    **For a symmetric matrix every condition number is exactly 1**, because the left and right
    eigenvectors coincide, so ``y^H x = 1``. That is why the symmetric problem is a separate
    subject.
    """
    A = np.atleast_2d(np.asarray(A))
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A is {A.shape}, eigenvalues need a square matrix")
    vals, right = np.linalg.eig(A)
    # Left eigenvectors of A are right eigenvectors of A^H, but the ORDER differs, so they must
    # be matched to the eigenvalues rather than assumed to line up.
    vals_left, left = np.linalg.eig(np.conj(A).T)
    order = []
    used = set()
    for lam in vals:
        gaps = [abs(np.conj(lv) - lam) if k not in used else np.inf
                for k, lv in enumerate(vals_left)]
        k = int(np.argmin(gaps))
        used.add(k)
        order.append(k)
    left = left[:, order]

    conds = []
    for i in range(vals.size):
        x = right[:, i] / np.linalg.norm(right[:, i])
        y = left[:, i] / np.linalg.norm(left[:, i])
        overlap = abs(complex(np.vdot(y, x)))
        conds.append(1.0 / overlap if overlap > 0 else float("inf"))
    return {"eigenvalues": vals, "condition_numbers": np.array(conds),
            "worst": float(np.max(conds)), "best": float(np.min(conds))}


def bauer_fike_bound(A, E=None) -> dict:
    """Bauer-Fike: every eigenvalue of ``A + E`` is within ``kappa(V) ||E||`` of one of ``A``.

    ``V`` is the eigenvector matrix, so the bound needs ``A`` to be diagonalizable and is
    **vacuous for a defective matrix**, where ``kappa(V)`` is infinite. That is not a weakness
    of the theorem: a defective eigenvalue really does move like ``||E||^(1/m)`` for a Jordan
    block of size ``m``, which is worse than any linear bound.

    **The bound is global and the truth is individual.** ``kappa(V)`` is one number for the whole
    matrix, so it is set by the worst-conditioned eigenvalue and applied to all of them. Lesson
    35 measures the resulting overstatement.

    Returns the bound, and when ``E`` is given, the movement that actually happened.
    """
    A = np.atleast_2d(np.asarray(A))
    vals, V = np.linalg.eig(A)
    kappa = float(np.linalg.cond(V))
    out = {"kappa_V": kappa, "eigenvalues": vals}
    if E is None:
        return out
    E = np.atleast_2d(np.asarray(E))
    if E.shape != A.shape:
        raise ValueError(f"E is {E.shape}, A is {A.shape}")
    norm_E = float(np.linalg.norm(E, 2))
    perturbed = np.linalg.eigvals(A + E)
    moved = float(max(min(abs(p - v) for v in vals) for p in perturbed))
    out.update({"norm_E": norm_E, "bound": kappa * norm_E, "actual": moved,
                "overstatement": (kappa * norm_E / moved) if moved > 0 else float("inf"),
                "perturbed": perturbed})
    return out


def defective_matrix(size: int, eigenvalue: float = 1.0) -> np.ndarray:
    """A single Jordan block: ``eigenvalue`` on the diagonal, 1 on the superdiagonal.

    It has ``size`` copies of one eigenvalue and only **one** eigenvector, so it is not
    diagonalizable. Perturbing the bottom-left corner by ``eps`` splits the eigenvalue into
    ``size`` values spread by ``eps^(1/size)``, which is the standard demonstration that a
    defective eigenvalue is not merely ill conditioned but conditioned at a different **rate**.
    """
    size = int(size)
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    return np.eye(size) * float(eigenvalue) + np.eye(size, k=1)


def jordan_perturbation_spread(size: int, eps: float, eigenvalue: float = 1.0) -> dict:
    """Perturb a Jordan block's corner and measure how far its eigenvalues move.

    Theory: the characteristic polynomial of the perturbed block is
    ``(lambda - a)^n - eps``, so the eigenvalues are ``a + eps^(1/n) w`` for the ``n``-th roots
    of unity ``w``. The movement is ``eps^(1/n)``, **not** proportional to ``eps``.

    At ``n = 10`` and ``eps = 1e-10`` that is a movement of ``0.1`` from a perturbation of
    ``1e-10``: an amplification of a billion, with no ill conditioned matrix in sight.
    """
    size = int(size)
    J = defective_matrix(size, eigenvalue)
    J[size - 1, 0] += float(eps)
    vals = np.linalg.eigvals(J)
    moved = float(np.max(np.abs(vals - eigenvalue)))
    predicted = float(abs(eps) ** (1.0 / size)) if eps != 0 else 0.0
    return {"eigenvalues": vals, "moved": moved, "predicted": predicted,
            "ratio": moved / predicted if predicted > 0 else float("nan"),
            "amplification": moved / abs(eps) if eps != 0 else float("inf")}


# ----------------------------------------------------------------------------- Schur


def schur_residuals(A) -> dict:
    """Check the Schur decomposition ``A = Q T Q^H`` with ``Q`` unitary and ``T`` triangular.

    **Every square matrix has one**, defective or not, which is what makes it the decomposition
    algorithms actually target. An eigendecomposition ``A = V D V^{-1}`` needs ``n`` independent
    eigenvectors and a defective matrix does not have them; the Schur form needs nothing.

    And its diagonal holds the eigenvalues, so it answers the question. The price is that ``T``
    is triangular rather than diagonal, so the eigen*vectors* take extra work to recover.

    Uses ``scipy.linalg.schur`` as the reference. Building one from scratch is the QR algorithm,
    which is lesson 37.
    """
    from scipy.linalg import schur

    A = np.atleast_2d(np.asarray(A))
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A is {A.shape}, the Schur form needs a square matrix")
    T, Q = schur(A, output="complex")
    n = A.shape[0]
    return {"T": T, "Q": Q,
            "unitary_error": float(np.linalg.norm(np.conj(Q).T @ Q - np.eye(n))),
            "reconstruction_error": float(np.linalg.norm(A - Q @ T @ np.conj(Q).T)
                                          / max(np.linalg.norm(A), 1e-300)),
            "below_diagonal": float(np.abs(np.tril(T, -1)).max()),
            "diagonal": np.diag(T)}


def similarity_preserves_eigenvalues(A, S) -> dict:
    """``S^{-1} A S`` has the same eigenvalues as ``A``, and possibly a very different
    conditioning.

    That second half is the part that matters for algorithms. Similarity is free to use, so
    every eigenvalue method is a sequence of similarities; but a **badly conditioned** ``S``
    changes the numerical problem even though it does not change the mathematical one. That is
    why algorithms restrict themselves to **orthogonal** similarities, where ``kappa(S) = 1``.
    """
    A = np.atleast_2d(np.asarray(A))
    S = np.atleast_2d(np.asarray(S))
    if S.shape != A.shape:
        raise ValueError(f"S is {S.shape}, A is {A.shape}")
    B = np.linalg.solve(S, A @ S)
    a = np.sort_complex(np.linalg.eigvals(A))
    b = np.sort_complex(np.linalg.eigvals(B))
    scale = float(np.max(np.abs(a))) or 1.0
    return {"kappa_S": float(np.linalg.cond(S)),
            "max_eigenvalue_shift": float(np.max(np.abs(a - b))) / scale,
            "kappa_A": float(np.linalg.cond(A)),
            "kappa_B": float(np.linalg.cond(B))}


# ----------------------------------------------------------------------------- no finite algorithm


def companion_matrix(coefficients) -> np.ndarray:
    """The companion matrix of a monic polynomial, whose eigenvalues are its roots.

    ``coefficients`` are ``[c_0, c_1, ..., c_{n-1}]`` for
    ``p(t) = t^n + c_{n-1} t^{n-1} + ... + c_0``.

    **This is the reduction that proves no finite eigenvalue algorithm exists.** Any procedure
    computing eigenvalues in finitely many arithmetic operations and root extractions would,
    applied to this matrix, produce a solution formula in radicals for ``p``. Abel and Galois
    proved no such formula exists for degree 5 and above, so no such procedure exists either.

    **It is a terrible way to find roots numerically**, which lesson 06 already measured: the
    companion matrix of a polynomial with clustered roots is hopelessly ill conditioned. The
    reduction is a proof device, not a method.
    """
    c = np.asarray(coefficients, dtype=float).ravel()
    n = c.size
    if n < 1:
        raise ValueError("a polynomial needs at least one coefficient")
    C = np.zeros((n, n))
    C[1:, :-1] = np.eye(n - 1)
    C[:, -1] = -c
    return C


def no_finite_algorithm(degree: int, rng=None) -> dict:
    """Demonstrate the Abel-Galois obstruction concretely at a given degree.

    Builds a polynomial with known roots, forms its companion matrix, and checks that the
    eigenvalues **are** the roots. The point is the equivalence, not the accuracy: for degree 5
    and above there is provably no formula in radicals for those roots, so there is no finite
    formula for those eigenvalues either.

    Also reports how badly conditioned the companion matrix is, which is lesson 06's separate
    warning and the reason nobody computes roots this way.
    """
    degree = int(degree)
    if degree < 1:
        raise ValueError(f"degree must be at least 1, got {degree}")
    gen = np.random.default_rng() if rng is None else rng
    roots = np.sort(gen.uniform(-2.0, 2.0, degree))
    # np.poly gives [1, a_{n-1}, ..., a_0], leading coefficient first. companion_matrix wants
    # [a_0, a_1, ..., a_{n-1}], so drop the leading 1 and reverse.
    coeffs = np.poly(roots)
    C = companion_matrix(coeffs[1:][::-1])
    vals = np.linalg.eigvals(C)
    ordered = np.sort(vals.real)
    return {"roots": roots, "eigenvalues": vals,
            "max_difference": float(np.max(np.abs(ordered - roots))),
            "max_imaginary_part": float(np.max(np.abs(vals.imag))),
            "kappa_companion": float(np.linalg.cond(C)),
            "solvable_in_radicals": degree <= 4}


def symmetric_eigenvalues_are_perfectly_conditioned(A, E) -> dict:
    """Weyl's inequality: for symmetric ``A`` and ``E``, every eigenvalue moves by at most
    ``||E||_2``. No condition number appears at all.

    That is the whole reason the symmetric eigenvalue problem is treated separately: the
    eigenvector matrix is orthogonal, so ``kappa(V) = 1`` and Bauer-Fike degenerates to
    ``||E||``. There is nothing to be ill conditioned about.

    Returns the bound and the movement that actually happened, sorted so the comparison is
    between corresponding eigenvalues rather than between sets.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    E = np.atleast_2d(np.asarray(E, dtype=float))
    if E.shape != A.shape:
        raise ValueError(f"E is {E.shape}, A is {A.shape}")
    sym = lambda M: float(np.linalg.norm(M - M.T))
    if sym(A) > 1e-12 * max(np.linalg.norm(A), 1.0):
        raise ValueError("A is not symmetric; Weyl's inequality does not apply")
    if sym(E) > 1e-12 * max(np.linalg.norm(E), 1.0):
        raise ValueError("E is not symmetric; Weyl's inequality does not apply")
    a = np.sort(np.linalg.eigvalsh(A))
    b = np.sort(np.linalg.eigvalsh(A + E))
    moved = float(np.max(np.abs(a - b)))
    bound = float(np.linalg.norm(E, 2))
    return {"bound": bound, "actual": moved,
            "ratio": moved / bound if bound > 0 else 0.0,
            "eigenvalues": a, "perturbed": b}
