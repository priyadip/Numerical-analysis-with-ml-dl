"""Multigrid: the only method in the course whose iteration count does not grow with n.

Every method in Part 4 so far reduces the iteration count by a constant, or at best halves its
exponent, and none of them changes the fact that a finer grid needs more iterations. Lesson 23
measured the reason: the error is a sum of modes, a smoother kills the oscillatory ones quickly
and the smooth ones barely at all, and refining the grid makes the slowest mode slower.

Multigrid takes that as an opportunity rather than a defect. A mode that is smooth on a fine
grid is **not smooth relative to a coarse one**: halve the number of points and the same
function looks twice as oscillatory. So:

1. **Smooth** on the fine grid, which removes everything oscillatory and leaves a smooth error.
2. **Restrict** the residual to a coarse grid, where the remaining error is no longer smooth.
3. **Solve** the coarse problem, recursively.
4. **Interpolate** the correction back and add it.
5. **Smooth** again, to remove what interpolation put back.

Each step handles exactly what the other cannot, and together they cover the whole spectrum.
The cost is a constant number of fine grid sweeps because the coarse grids form a geometric
series: 1 + 1/2 + 1/4 + ... = 2 in one dimension, 1 + 1/4 + 1/16 + ... = 4/3 in two.

Everything here derives its sizes from its arguments. Grid dimensions come from the vector
lengths, the number of levels is computed from the finest grid rather than fixed, and the
recursion stops when the coarse problem is small enough to solve directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------- grids and operators


def n_levels(n: int, coarsest: int = 3) -> int:
    """How many levels the standard coarsening gives, from n down to `coarsest` or below.

    With ``n = 2^k - 1`` interior points the coarsening ``n -> (n-1)//2`` is exact at every
    level, which is why those sizes are the conventional choice. Any n works; the recursion
    simply stops earlier.
    """
    if n < 1:
        raise ValueError(f"need at least one interior point, got {n}")
    levels, m = 1, int(n)
    while m > coarsest and (m - 1) // 2 >= 1:
        m = (m - 1) // 2
        levels += 1
    return levels


def poisson_1d(n: int, h: float | None = None) -> np.ndarray:
    """The 1D model operator, scaled by 1/h^2 with h = 1/(n+1) unless given.

    Scaling matters here in a way it did not in lesson 21: a multigrid cycle compares residuals
    across grids, so the fine and coarse operators must be discretisations of the SAME
    differential operator, not merely similar matrices.
    """
    if n < 1:
        raise ValueError(f"need at least one interior point, got {n}")
    h = 1.0 / (n + 1) if h is None else float(h)
    A = (np.diag(np.full(n, 2.0)) + np.diag(np.full(max(n - 1, 0), -1.0), 1)
         + np.diag(np.full(max(n - 1, 0), -1.0), -1))
    return A / (h * h)


def restrict_full_weighting(r) -> np.ndarray:
    """Fine to coarse, with the stencil (1, 2, 1)/4. Length n -> (n-1)//2.

    Full weighting is the transpose of linear interpolation up to a factor of 2, which is what
    makes the coarse grid operator symmetric when the fine one is. **Injection**, taking every
    other value, is cheaper and much worse: it ignores the neighbours, so any high frequency
    left over by the smoother is aliased straight onto the coarse grid instead of being
    averaged away. Lesson 28 measures both.
    """
    r = np.asarray(r, dtype=float).ravel()
    n = r.size
    nc = (n - 1) // 2
    if nc < 1:
        return np.zeros(0)
    out = np.empty(nc)
    for i in range(nc):
        j = 2 * i + 1                                    # the fine index this coarse one sits on
        out[i] = 0.25 * r[j - 1] + 0.5 * r[j] + 0.25 * r[j + 1]
    return out


def restrict_injection(r) -> np.ndarray:
    """Fine to coarse by taking every other value. Cheap, and aliases high frequencies."""
    r = np.asarray(r, dtype=float).ravel()
    nc = (r.size - 1) // 2
    return r[1::2][:nc].copy() if nc >= 1 else np.zeros(0)


def prolong_linear(e, n_fine: int | None = None) -> np.ndarray:
    """Coarse to fine by linear interpolation. Length nc -> 2*nc + 1.

    Coarse points map straight across; fine points between them take the average of their two
    coarse neighbours, with the boundary treated as zero. This is the transpose of full
    weighting times 2, so the pair is **variationally consistent** and the coarse grid
    correction is an A-orthogonal projection (lesson 16).
    """
    e = np.asarray(e, dtype=float).ravel()
    nc = e.size
    n = 2 * nc + 1 if n_fine is None else int(n_fine)
    out = np.zeros(n)
    for i in range(nc):
        j = 2 * i + 1
        out[j] += e[i]
        if j - 1 >= 0:
            out[j - 1] += 0.5 * e[i]
        if j + 1 < n:
            out[j + 1] += 0.5 * e[i]
    return out


def galerkin_coarse(A, nc: int) -> np.ndarray:
    """The coarse operator R A P, built from the fine one rather than rediscretised.

    For the 1D model problem this gives exactly the coarse discretisation, which is worth
    checking rather than assuming, and lesson 28 does check it. For a variable coefficient or
    distorted problem the two differ, and the Galerkin form is the one that keeps the
    variational property, which is why algebraic multigrid uses it.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    P = np.zeros((n, nc))
    for i in range(nc):
        P[:, i] = prolong_linear(np.eye(nc)[:, i], n)
    R = 0.5 * P.T                                        # full weighting = P^T / 2
    return R @ A @ P


# ---------------------------------------------------------------- smoothing


def smooth(A, b, x, sweeps: int, kind: str = "damped_jacobi", omega: float = 2.0 / 3.0):
    """Apply `sweeps` sweeps of the named smoother, in place on a copy.

    Damped Jacobi at omega = 2/3 is the standard choice, and lesson 23 measured exactly why:
    plain Jacobi's eigenvalues approach -1 at the top of the spectrum, so its worst shrinkage
    over the upper half is 0.9988 and it barely touches the modes the coarse grid cannot fix.
    Damping maps every eigenvalue lam to 1 - omega(1 - lam), sending -1 to 1 - 2 omega, and at
    omega = 2/3 the worst upper-half shrinkage becomes 0.3333.

    Gauss-Seidel is the other standard choice, at 0.4435 on the same measurement, and it is
    sequential where damped Jacobi is not.

    SOR at its optimal omega is a **bad** smoother, measured at 1.0898: it makes the high
    frequency modes grow. A good solver and a good smoother are different things.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel().copy()
    n = A.shape[0]
    d = np.diag(A).copy()
    if np.any(d == 0.0):
        raise np.linalg.LinAlgError("a smoother needs a nonzero diagonal")

    kind = kind.lower()
    if kind in ("damped_jacobi", "jacobi"):
        w = 1.0 if kind == "jacobi" else float(omega)
        if not 0.0 < w <= 1.0:
            raise ValueError(f"damped Jacobi needs 0 < omega <= 1, got {w}")
        R = A - np.diag(d)
        for _ in range(int(sweeps)):
            x = (1.0 - w) * x + w * ((b - R @ x) / d)
        return x
    if kind in ("gauss-seidel", "gauss_seidel", "gs"):
        for _ in range(int(sweeps)):
            for i in range(n):
                x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1:] @ x[i + 1:]) / d[i]
        return x
    if kind in ("red-black", "red_black", "rb"):
        colours = (np.arange(0, n, 2), np.arange(1, n, 2))
        R = A - np.diag(d)
        for _ in range(int(sweeps)):
            for idx in colours:
                x[idx] = (b[idx] - R[idx] @ x) / d[idx]
        return x
    raise ValueError("kind must be 'damped_jacobi', 'jacobi', 'gauss-seidel' or 'red-black'")


def smoothing_factor(A, kind: str = "damped_jacobi", omega: float = 2.0 / 3.0) -> dict:
    """The worst shrinkage a smoother achieves over the whole spectrum and over its upper half.

    The upper half is the number that decides whether something is a usable smoother, because
    those are exactly the modes a coarse grid with half the points cannot represent. The
    overall worst is the number that decides whether it is a usable **solver**, and the two are
    almost unrelated.

    Uses the discrete sine modes, which diagonalise the 1D model operator exactly, so this is a
    measurement rather than an estimate. The size comes from A.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    d = np.diag(A).copy()
    kind_l = kind.lower()
    if kind_l in ("damped_jacobi", "jacobi"):
        w = 1.0 if kind_l == "jacobi" else float(omega)
        G = (1.0 - w) * np.eye(n) + w * np.linalg.solve(np.diag(d), np.diag(d) - A)
    elif kind_l in ("gauss-seidel", "gauss_seidel", "gs"):
        M = np.tril(A)
        G = np.linalg.solve(M, M - A)
    elif kind_l == "sor":
        M = np.diag(d) / float(omega) + np.tril(A, -1)
        G = np.linalg.solve(M, M - A)
    else:
        raise ValueError(f"unknown smoother {kind!r}")

    h = 1.0 / (n + 1)
    modes = np.arange(1, n + 1)
    V = np.sqrt(2.0 * h) * np.sin(np.outer(modes, modes) * np.pi * h)
    shrink = np.linalg.norm(G @ V, axis=0)
    upper = shrink[n // 2:]
    return {"per_mode": shrink,
            "worst_overall": float(shrink.max()),
            "worst_upper_half": float(upper.max()),
            "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(G))))}


# ---------------------------------------------------------------- the cycles


@dataclass
class CycleResult:
    """What a multigrid run did, with enough history to measure the convergence factor."""

    x: np.ndarray
    residual_norms: np.ndarray
    converged: bool
    n_cycles: int
    work_units: float
    message: str
    levels: int = 0

    def convergence_factor(self, last: int = 5) -> float:
        """The geometric mean residual reduction per cycle, over the last few cycles.

        This is the number that should be independent of n, and measuring it over the tail
        avoids the first cycle or two, where the initial guess still matters.
        """
        r = np.asarray(self.residual_norms, dtype=float)
        r = r[r > 0]
        if r.size < 2:
            return float("nan")
        k = min(int(last), r.size - 1)
        return float((r[-1] / r[-1 - k]) ** (1.0 / k))


def coarse_operators(A, coarsest: int = 3, restrict: Callable | None = None) -> list:
    """Build the whole operator hierarchy once, from the finest level down.

    A real multigrid code separates **setup** from **solve**, because the coarse operators
    depend only on A and not on the right-hand side, so rebuilding them every cycle is pure
    waste. Doing it here also stops the O(n^3) dense Galerkin product from dominating the
    measurements and hiding the algorithm's actual cost.

    The level count follows from the size of A, so nothing about the dimension is fixed.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    levels = [A]
    while True:
        n = levels[-1].shape[0]
        nc = (n - 1) // 2
        if n <= coarsest or nc < 1:
            break
        levels.append(galerkin_coarse(levels[-1], nc))
    return levels


def v_cycle(A, b, x, nu1: int = 2, nu2: int = 2, coarsest: int = 3,
            kind: str = "damped_jacobi", omega: float = 2.0 / 3.0,
            restrict: Callable | None = None, hierarchy: list | None = None) -> np.ndarray:
    """One V-cycle: smooth, restrict, recurse, prolong, correct, smooth.

    The recursion stops when the grid has `coarsest` points or fewer, or when coarsening would
    leave nothing, and the coarse problem is then solved exactly. Nothing about the size is
    fixed: the level count follows from the length of ``b``.

    ``nu1`` and ``nu2`` are the pre and post smoothing sweeps. Two of each is the usual choice,
    and lesson 28 measures what happens with other values.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel().copy()
    n = A.shape[0]
    restrict = restrict_full_weighting if restrict is None else restrict
    nc = (n - 1) // 2

    if n <= coarsest or nc < 1:
        return np.linalg.solve(A, b)                     # exact on the coarsest grid

    x = smooth(A, b, x, nu1, kind, omega)                # 1. pre-smooth
    r_c = restrict(b - A @ x)                            # 2. restrict the residual
    if hierarchy is None or len(hierarchy) < 2:          # 3. the coarse operator
        A_c, rest = galerkin_coarse(A, r_c.size), None
    else:
        A_c, rest = hierarchy[1], hierarchy[1:]
    e_c = v_cycle(A_c, r_c, np.zeros(r_c.size), nu1, nu2, coarsest, kind, omega,
                  restrict, rest)
    x = x + prolong_linear(e_c, n)                       # 4. correct
    return smooth(A, b, x, nu2, kind, omega)             # 5. post-smooth


def w_cycle(A, b, x, nu1: int = 2, nu2: int = 2, coarsest: int = 3,
            kind: str = "damped_jacobi", omega: float = 2.0 / 3.0,
            restrict: Callable | None = None, hierarchy: list | None = None) -> np.ndarray:
    """One W-cycle: the same, but visiting the coarse level twice instead of once.

    More robust than a V-cycle when the coarse grid correction is imperfect, and more
    expensive. In one dimension the work per cycle is O(n log n) rather than O(n), because the
    geometric series 1 + 2/2 + 4/4 + ... no longer converges; in two dimensions it is still
    O(n), since 1 + 2/4 + 4/16 + ... does.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel().copy()
    n = A.shape[0]
    restrict = restrict_full_weighting if restrict is None else restrict
    nc = (n - 1) // 2

    if n <= coarsest or nc < 1:
        return np.linalg.solve(A, b)

    x = smooth(A, b, x, nu1, kind, omega)
    r_c = restrict(b - A @ x)
    if hierarchy is None or len(hierarchy) < 2:
        A_c, rest = galerkin_coarse(A, r_c.size), None
    else:
        A_c, rest = hierarchy[1], hierarchy[1:]
    e_c = np.zeros(r_c.size)
    for _ in range(2):                                   # the only difference from a V-cycle
        e_c = w_cycle(A_c, r_c, e_c, nu1, nu2, coarsest, kind, omega, restrict, rest)
    x = x + prolong_linear(e_c, n)
    return smooth(A, b, x, nu2, kind, omega)


def two_grid(A, b, x, nu1: int = 2, nu2: int = 2, kind: str = "damped_jacobi",
             omega: float = 2.0 / 3.0, restrict: Callable | None = None,
             hierarchy: list | None = None) -> np.ndarray:
    """One two-grid cycle: the coarse problem is solved exactly rather than recursively.

    Not a practical method, since the coarse solve is as hard as the original in the limit, but
    it is the object the theory analyses. Comparing it with the V-cycle separates two questions:
    whether the coarse grid correction is doing its job, and whether the recursion is.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel().copy()
    n = A.shape[0]
    restrict = restrict_full_weighting if restrict is None else restrict
    nc = (n - 1) // 2
    if nc < 1:
        return np.linalg.solve(A, b)

    x = smooth(A, b, x, nu1, kind, omega)
    r_c = restrict(b - A @ x)
    A_c = galerkin_coarse(A, nc) if hierarchy is None or len(hierarchy) < 2 else hierarchy[1]
    x = x + prolong_linear(np.linalg.solve(A_c, r_c), n)
    return smooth(A, b, x, nu2, kind, omega)


def solve(A, b, x0=None, cycle: str = "v", tol: float = 1e-10, max_cycles: int = 100,
          nu1: int = 2, nu2: int = 2, coarsest: int = 3, kind: str = "damped_jacobi",
          omega: float = 2.0 / 3.0, restrict: Callable | None = None) -> CycleResult:
    """Iterate the chosen cycle until the relative residual falls below ``tol``.

    The number that matters is ``result.convergence_factor()``, which for a well set up
    multigrid is about 0.1 and **does not change with n**. Every other method in Part 4 has a
    factor that approaches 1 as the grid refines.

    ``work_units`` counts fine grid sweeps, so a V-cycle costs about ``2 * (nu1 + nu2)`` of them
    in one dimension, the factor of 2 being the geometric series over the coarse grids.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[0]
    if b.size != n:
        raise ValueError(f"b has length {b.size}, but A is {n} by {n}")
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()

    step = {"v": v_cycle, "w": w_cycle, "two-grid": two_grid}.get(cycle.lower())
    if step is None:
        raise ValueError("cycle must be 'v', 'w' or 'two-grid'")

    scale = float(np.linalg.norm(b)) or 1.0
    residuals = [float(np.linalg.norm(b - A @ x))]
    levels = n_levels(n, coarsest)
    per_cycle = (nu1 + nu2) * (2.0 if cycle.lower() == "v" else 4.0)
    # Setup once. The coarse operators depend on A alone, so rebuilding them every cycle would
    # make the dense implementation O(n^3) PER CYCLE and hide what the algorithm really costs.
    hierarchy = coarse_operators(A, coarsest, restrict)

    for k in range(1, int(max_cycles) + 1):
        if cycle.lower() == "two-grid":
            x = step(A, b, x, nu1, nu2, kind, omega, restrict, hierarchy)
        else:
            x = step(A, b, x, nu1, nu2, coarsest, kind, omega, restrict, hierarchy)
        residuals.append(float(np.linalg.norm(b - A @ x)))
        if not np.all(np.isfinite(x)):
            return CycleResult(x, np.array(residuals), False, k, k * per_cycle,
                               "diverged", levels)
        if residuals[-1] <= tol * scale:
            return CycleResult(x, np.array(residuals), True, k, k * per_cycle,
                               "converged", levels)

    return CycleResult(x, np.array(residuals), False, int(max_cycles),
                       max_cycles * per_cycle, "cycle limit reached", levels)


def two_grid_operator(A, nu1: int = 2, nu2: int = 2, kind: str = "damped_jacobi",
                      omega: float = 2.0 / 3.0) -> np.ndarray:
    """The two-grid iteration matrix, formed explicitly so its spectral radius can be measured.

    Never do this in a real solve: it is dense and costs more than the solve. It is here
    because lesson 28's central claim is that the spectral radius stays bounded away from 1 as
    n grows, and that claim has to be checked rather than asserted.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    hierarchy = coarse_operators(A)
    G = np.empty((n, n))
    zero = np.zeros(n)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        # the error propagation of one cycle applied to error e with zero right-hand side
        G[:, j] = two_grid(A, zero, e, nu1, nu2, kind, omega, None, hierarchy)
    return G


# ---------------------------------------------------------------- the O(n) version


class StencilHierarchy:
    """A matrix-free multigrid hierarchy for the 1D model operator.

    The routines above form the coarse operators as dense matrices ``R A P``, which is clear to
    read and costs ``O(n^3)`` per cycle. That hides the property the whole method exists for, so
    this class does the same algorithm with stencils: every operation is ``O(n)``, the operators
    are never stored, and the total work per V-cycle is a constant number of fine grid sweeps.

    The level sizes come from the finest grid, so nothing about the dimension is fixed. The
    coarsening ``n -> (n-1)//2`` is exact when ``n = 2^k - 1``, which is why that family is used
    for the measurements, and any other n simply bottoms out earlier.
    """

    def __init__(self, n: int, coarsest: int = 3):
        if n < 1:
            raise ValueError(f"need at least one interior point, got {n}")
        self.sizes = [int(n)]
        while self.sizes[-1] > coarsest and (self.sizes[-1] - 1) // 2 >= 1:
            self.sizes.append((self.sizes[-1] - 1) // 2)
        #: 1/h^2 at each level, so every level discretises the SAME differential operator
        self.scales = [(size + 1) ** 2 for size in self.sizes]
        self.n_matvec = 0

    @property
    def levels(self) -> int:
        return len(self.sizes)

    def apply(self, level: int, x) -> np.ndarray:
        """The second difference operator at `level`, applied without a matrix. O(n)."""
        x = np.asarray(x, dtype=float).ravel()
        y = 2.0 * x
        y[:-1] -= x[1:]
        y[1:] -= x[:-1]
        self.n_matvec += 1
        return y * self.scales[level]

    def smooth(self, level: int, b, x, sweeps: int, omega: float = 2.0 / 3.0) -> np.ndarray:
        """Damped Jacobi, stencil form. The diagonal is 2/h^2, known rather than looked up."""
        x = np.asarray(x, dtype=float).ravel().copy()
        b = np.asarray(b, dtype=float).ravel()
        diag = 2.0 * self.scales[level]
        for _ in range(int(sweeps)):
            x = x + omega * (b - self.apply(level, x)) / diag
        return x

    def cycle(self, level: int, b, x, nu1: int = 2, nu2: int = 2,
              omega: float = 2.0 / 3.0) -> np.ndarray:
        """One V-cycle starting at `level`, entirely in O(n) operations."""
        if level == self.levels - 1:
            # the coarsest grid is tiny, so a dense solve here is O(1) in the fine grid size
            A_c = poisson_1d(self.sizes[level], 1.0 / (self.sizes[level] + 1))
            return np.linalg.solve(A_c, np.asarray(b, dtype=float).ravel())
        x = self.smooth(level, b, x, nu1, omega)
        r_c = restrict_full_weighting(np.asarray(b).ravel() - self.apply(level, x))
        e_c = self.cycle(level + 1, r_c, np.zeros(r_c.size), nu1, nu2, omega)
        x = x + prolong_linear(e_c, self.sizes[level])
        return self.smooth(level, b, x, nu2, omega)

    def solve(self, b, x0=None, tol: float = 1e-10, max_cycles: int = 100,
              nu1: int = 2, nu2: int = 2, omega: float = 2.0 / 3.0) -> CycleResult:
        """Iterate the matrix-free V-cycle. Total cost O(n) per cycle, O(n) overall."""
        b = np.asarray(b, dtype=float).ravel()
        n = self.sizes[0]
        if b.size != n:
            raise ValueError(f"b has length {b.size}, expected {n}")
        x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
        scale = float(np.linalg.norm(b)) or 1.0
        residuals = [float(np.linalg.norm(b - self.apply(0, x)))]
        for k in range(1, int(max_cycles) + 1):
            x = self.cycle(0, b, x, nu1, nu2, omega)
            residuals.append(float(np.linalg.norm(b - self.apply(0, x))))
            if residuals[-1] <= tol * scale:
                return CycleResult(x, np.array(residuals), True, k,
                                   k * (nu1 + nu2) * 2.0, "converged", self.levels)
        return CycleResult(x, np.array(residuals), False, int(max_cycles),
                           max_cycles * (nu1 + nu2) * 2.0, "cycle limit reached", self.levels)
