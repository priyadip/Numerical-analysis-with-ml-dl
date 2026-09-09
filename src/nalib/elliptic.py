"""Laplace and Poisson in two dimensions: the discretization, and the system it produces.

Where the work is
-----------------
An elliptic problem has no direction to march in. The value at every point depends on the data
everywhere, so discretizing it does not give a recurrence, it gives **one large linear system**:

    -(u_xx + u_yy) = f  on a rectangle,   u given on the boundary
    ->  A u = b  with A the five point Laplacian of lesson 77.

The discretization is the easy half. The matrix has ``N = m**2`` unknowns and five nonzeros a row,
it is symmetric positive definite, and its condition number grows like ``h**-2``. Everything Parts
3 and 4 said about such a matrix applies directly, and this module is where those parts get used
rather than restated.

What the measurements here show
-------------------------------
* The eigenvalues of the five point matrix are known exactly,
  ``4/h**2 (sin^2(p pi h/2) + sin^2(q pi h/2))``, and the measurement agrees with them to
  ``1e-13``. That makes the condition number a formula rather than an estimate:
  ``cot^2(pi h / 2)``, which is ``4/(pi**2 h**2)`` for small ``h``.
* The optimal relaxation factor for SOR is ``2/(1 + sin(pi h))``, and searching for it numerically
  lands within ``0.002`` of that at every grid size tried. The gain over Gauss-Seidel is a factor
  of 21 in iterations at ``m = 31``, and it grows.
* Four solvers on the same problem, counted honestly: Jacobi and Gauss-Seidel need ``O(N)``
  iterations, SOR with the optimal factor needs ``O(sqrt(N))``, conjugate gradients needs
  ``O(sqrt(N))`` with a much better constant, and each is measured rather than quoted.
* Piecewise linear finite elements on a right triangle mesh give **exactly** the five point
  matrix, entry for entry, to ``1e-15``. That is the two dimensional version of lesson 76's
  discovery that ``h K = tridiag(-1, 2, -1)``.
* And it goes further than lesson 76's version. With the vertex quadrature rule the **right hand
  sides match too**, so the two methods produce bit for bit the same numbers: the agreement is
  ``2e-16`` at every grid size. Six triangles of area ``h**2/2`` meet at a node, each giving it
  ``h**2/6`` of the value there, which sums to exactly the difference method's ``h**2 f``. Switch
  to the centroid rule, which is just as accurate, and the two part company, with the element
  error 67 per cent larger and both still second order. **The quadrature rule is the whole of the
  difference between the two methods here.**
* The obvious test problem is degenerate for a Krylov method. ``sin(p pi x) sin(q pi y)`` sampled
  on the grid is an exact eigenvector of the matrix, so conjugate gradients converges in **one**
  iteration at every size. Gauss-Seidel takes 147, 595 and 2387 on the same problems and barely
  notices the difference, which is the practical face of what separates a Krylov method from a
  fixed point iteration.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .iterative import (gauss_seidel, iteration_matrix, jacobi,
                        optimal_omega as omega_from_the_matrix, sor,
                        spectral_radius)
from .krylov import conjugate_gradient


# --------------------------------------------------------------------------- problems


def manufactured_problem(a: float = 0.0, b: float = 1.0, modes=(1, 1)):
    """``-(u_xx + u_yy) = f`` with ``u = sin(p pi x) sin(q pi y)`` and zero boundary values.

    The source is ``f = (p**2 + q**2) pi**2 u``, so the exact solution is known everywhere and the
    error of any method can be measured. It is also an eigenfunction of the continuous operator,
    which makes it the cleanest place to compare the discrete eigenvalues against their formula.
    """
    a, b = float(a), float(b)
    if not b > a:
        raise ValueError(f"need a < b, got a={a}, b={b}")
    p, q = int(modes[0]), int(modes[1])
    length = b - a
    wx, wy = p * math.pi / length, q * math.pi / length

    def exact(x, y):
        return (np.sin(wx * (np.asarray(x, dtype=float) - a))
                * np.sin(wy * (np.asarray(y, dtype=float) - a)))

    def source(x, y):
        return (wx ** 2 + wy ** 2) * exact(x, y)

    return {"a": a, "b": b, "exact": exact, "source": source,
            "boundary": lambda x, y: exact(x, y), "name": f"sine {p},{q}"}


def mixed_modes_problem(a: float = 0.0, b: float = 1.0):
    """``u = exp(x) sin(pi x) sin(pi y)``, which is **not** an eigenvector of the matrix.

    That matters more than it sounds. The obvious test problem, a single sine mode, is an exact
    eigenvector of the five point matrix, so its discrete right hand side lies in a one dimensional
    Krylov space and **conjugate gradients converges in one iteration** at every grid size. Any
    scaling law measured on it is a measurement of that degeneracy.

    Here the exponential factor spreads the right hand side over the whole spectrum. Putting the
    solution into the operator gives

        f = exp(x) sin(pi y) [ (2 pi**2 - 1) sin(pi x) - 2 pi cos(pi x) ] ,

    and the boundary values are still zero, so nothing else about the problem changes.
    """
    a, b = float(a), float(b)
    if not b > a:
        raise ValueError(f"need a < b, got a={a}, b={b}")

    def exact(x, y):
        gx = np.asarray(x, dtype=float)
        gy = np.asarray(y, dtype=float)
        return np.exp(gx) * np.sin(math.pi * gx) * np.sin(math.pi * gy)

    def source(x, y):
        gx = np.asarray(x, dtype=float)
        gy = np.asarray(y, dtype=float)
        return (np.exp(gx) * np.sin(math.pi * gy)
                * ((2.0 * math.pi ** 2 - 1.0) * np.sin(math.pi * gx)
                   - 2.0 * math.pi * np.cos(math.pi * gx)))

    return {"a": a, "b": b, "exact": exact, "source": source,
            "boundary": lambda x, y: exact(x, y), "name": "mixed modes"}


def harmonic_problem(a: float = 0.0, b: float = 1.0):
    """Laplace's equation, ``f = 0``, with ``u = exp(x) sin(y)`` prescribed on the boundary.

    A harmonic function is the pure elliptic case: the interior is entirely determined by the
    boundary, with no source to help. It is also the function lesson 77 used to show the nine point
    Laplacian reaching sixth order, so the two lessons can be compared directly.
    """
    a, b = float(a), float(b)

    def exact(x, y):
        return np.exp(np.asarray(x, dtype=float)) * np.sin(np.asarray(y, dtype=float))

    return {"a": a, "b": b, "exact": exact,
            "source": lambda x, y: np.zeros(np.broadcast(np.asarray(x, dtype=float),
                                                         np.asarray(y, dtype=float)).shape),
            "boundary": lambda x, y: exact(x, y), "name": "harmonic"}


def cooling_fin(length: float = 2.0, height: float = 2.0, conductivity: float = 1.68,
                thickness: float = 0.1, coefficient: float = 0.005,
                ambient: float = 20.0, power: float = 5.0, contact: float = 2.0):
    """A rectangular fin, heated on one edge and losing heat to the air everywhere else.

    The steady state of a thin fin with heat loss through both faces is

        k delta u = 2 H (u - ambient) / thickness ,

    with a prescribed flux over the contact region on the left edge and no flux elsewhere. This
    is Sauer's Reality Check 8, and it is here because it is the first problem in this part whose
    boundary conditions are **not** all Dirichlet: three edges are insulated, part of the fourth
    carries a flux, and the equation has a zeroth order term.

    Returned as a description rather than a solution, because assembling it is the exercise.
    """
    if thickness <= 0.0 or conductivity <= 0.0:
        raise ValueError("thickness and conductivity must be positive")
    return {
        "a": 0.0, "b": float(length), "height": float(height),
        "conductivity": float(conductivity), "thickness": float(thickness),
        "coefficient": float(coefficient), "ambient": float(ambient),
        "power": float(power), "contact": float(contact),
        "reaction": 2.0 * float(coefficient) / (float(conductivity) * float(thickness)),
        "flux": float(power) / (float(conductivity) * float(thickness) * float(contact)),
        "name": "cooling fin",
    }


# --------------------------------------------------------------------------- the matrix


def five_point_matrix(unknowns: int, h: float, reaction: float = 0.0):
    """The five point Laplacian on a square grid of ``unknowns`` interior points a side.

    Rows are ordered ``row * m + column``, so the matrix is block tridiagonal with bandwidth
    ``m``. Built dense on purpose: this module is about what the matrix **is**, and the solvers
    that exploit its sparsity are imported from Parts 3 and 4 rather than rewritten.

    ``reaction`` adds ``c u`` to the operator, which is what the cooling fin needs and what turns
    a singular Neumann problem into a solvable one.
    """
    m = int(unknowns)
    if m < 1:
        raise ValueError(f"need at least one unknown per side, got {m}")
    size = m * m
    scale = 1.0 / float(h) ** 2
    a = np.zeros((size, size))
    for i in range(m):
        for j in range(m):
            row = i * m + j
            a[row, row] = 4.0 * scale + float(reaction)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < m:
                    a[row, ii * m + jj] = -scale
    return a


def assemble(problem, points: int):
    """Build ``A u = b`` for the interior unknowns, folding the boundary values into ``b``.

    ``points`` counts the boundary, so there are ``(points - 2)**2`` unknowns. The boundary
    contributions are the only place the problem's data enters besides the source.
    """
    n = int(points)
    if n < 3:
        raise ValueError(f"need at least 3 points per side, got {n}")
    line = np.linspace(problem["a"], problem["b"], n)
    h = float(line[1] - line[0])
    m = n - 2
    gx, gy = np.meshgrid(line[1:-1], line[1:-1], indexing="ij")
    a = five_point_matrix(m, h)
    b = np.asarray(problem["source"](gx, gy), dtype=float).ravel()
    scale = 1.0 / h ** 2
    edge = np.asarray(problem["boundary"](
        *np.meshgrid(line, line, indexing="ij")), dtype=float)
    for i in range(m):
        for j in range(m):
            row = i * m + j
            if i == 0:
                b[row] += scale * edge[0, j + 1]
            if i == m - 1:
                b[row] += scale * edge[-1, j + 1]
            if j == 0:
                b[row] += scale * edge[i + 1, 0]
            if j == m - 1:
                b[row] += scale * edge[i + 1, -1]
    return {"A": a, "b": b, "h": h, "line": line, "unknowns": m, "x": gx, "y": gy}


def solve(problem, points: int, method: str = "direct", **options):
    """Solve the discrete problem by one of the methods of Parts 3 and 4, and report the error.

    The point of routing every method through one function is that the **matrix** is the same in
    every case. What changes is only how the system is solved, which is the honest way to compare
    them.
    """
    system = assemble(problem, points)
    a, b = system["A"], system["b"]
    out = None
    if method == "direct":
        u = np.linalg.solve(a, b)
        iterations = 1
    elif method == "jacobi":
        out = jacobi(a, b, **options)
        u, iterations = out.x, out.n_iter
    elif method == "gauss seidel":
        out = gauss_seidel(a, b, **options)
        u, iterations = out.x, out.n_iter
    elif method == "sor":
        out = sor(a, b, **options)
        u, iterations = out.x, out.n_iter
    elif method == "cg":
        out = conjugate_gradient(a, b, **options)
        u, iterations = out.x, out.n_iter
    else:
        raise ValueError(f"unknown method {method!r}")
    want = np.asarray(problem["exact"](system["x"], system["y"]), dtype=float).ravel()
    return {"u": u, "exact": want, "iterations": int(iterations),
            "converged": bool(getattr(out, "converged", True)) if method != "direct" else True, "h": system["h"],
            "unknowns": system["unknowns"], "method": method,
            "error": float(np.max(np.abs(u - want))),
            "residual": float(np.linalg.norm(a @ u - b) / max(np.linalg.norm(b), 1e-300))}


# --------------------------------------------------------------------------- the spectrum


def exact_eigenvalues(unknowns: int, h: float):
    """The eigenvalues of the five point matrix, in closed form.

    The eigenvectors are the products of the one dimensional sine modes, so the eigenvalues are

        lambda_{p,q} = (4/h**2) [ sin^2(p pi h / 2) + sin^2(q pi h / 2) ] ,

    for ``p, q = 1 ... m``. This is what makes every statement about conditioning and about
    iterative convergence in this lesson exact rather than estimated.
    """
    m = int(unknowns)
    index = np.arange(1, m + 1)
    single = 4.0 / float(h) ** 2 * np.sin(index * math.pi * float(h) / 2.0) ** 2
    return np.sort((single[:, None] + single[None, :]).ravel())


def the_spectrum_is_known_exactly(sizes=(3, 5, 9, 17)):
    """Compare the computed eigenvalues against the formula, and the condition number too.

    The condition number works out to ``cot^2(pi h / 2)``, exactly, which for small ``h`` is
    ``4/(pi**2 h**2)``. Both forms are reported, along with the error of the small ``h``
    approximation, because it is a good one and it is not exact.
    """
    rows = []
    for m in sizes:
        unknowns = int(m)
        h = 1.0 / (unknowns + 1)
        a = five_point_matrix(unknowns, h)
        computed = np.sort(np.linalg.eigvalsh(a))
        formula = exact_eigenvalues(unknowns, h)
        kappa = float(computed[-1] / computed[0])
        exact_kappa = 1.0 / math.tan(math.pi * h / 2.0) ** 2
        rows.append({
            "unknowns": unknowns,
            "h": h,
            "largest_gap": float(np.max(np.abs(computed - formula))),
            "relative_gap": float(np.max(np.abs(computed - formula)) / computed[-1]),
            "condition_number": kappa,
            "cot_squared": exact_kappa,
            "small_h_form": 4.0 / (math.pi ** 2 * h ** 2),
            "small_h_error": abs(kappa - 4.0 / (math.pi ** 2 * h ** 2)) / kappa,
        })
    hs = np.asarray([row["h"] for row in rows])
    kappas = np.asarray([row["condition_number"] for row in rows])
    return {
        "rows": rows,
        "the_formula_is_exact": all(row["relative_gap"] < 1e-12 for row in rows),
        "worst_relative_gap": max(row["relative_gap"] for row in rows),
        "cot_squared_is_exact": all(
            abs(row["condition_number"] - row["cot_squared"])
            < 1e-9 * row["condition_number"] for row in rows),
        "condition_exponent": float(np.polyfit(np.log(hs), np.log(kappas), 1)[0]),
        "the_small_h_form_is_close_not_exact": all(
            0.0 < row["small_h_error"] < 0.4 for row in rows),
    }


def optimal_omega(h: float) -> float:
    """The relaxation factor that makes SOR fastest on this matrix.

    For the five point Laplacian on a square, ``omega* = 2 / (1 + sin(pi h))``. It is one of the
    few problems where the optimal factor is known in closed form, and it is worth measuring
    against a search because the curve around it is **not** symmetric: overshooting is far less
    costly than undershooting.
    """
    return 2.0 / (1.0 + math.sin(math.pi * float(h)))


def the_optimal_omega_is_where_the_formula_says(sizes=(5, 9, 15), samples: int = 199,
                                                run_size: int = 31):
    """Find the best relaxation factor three ways and compare all three.

    For the five point Laplacian on a square the optimal factor is known in closed form,
    ``omega* = 2/(1 + sin(pi h))``. Three independent routes to it are measured here:

    1. the **formula**,
    2. a fine sweep of ``rho(G_SOR)``, the spectral radius of the iteration matrix, which is what
       actually governs the asymptotic rate,
    3. `iterative.optimal_omega`, Part 4's general expression
       ``2/(1 + sqrt(1 - rho_Jacobi**2))``, which applies to any consistently ordered matrix.

    All three agree, and route 3 agreeing with route 1 is the interesting part: the general
    formula from Part 4 reduces to the special one for this matrix, because ``rho_Jacobi`` here is
    ``cos(pi h)``.

    The **shape** of the curve is reported as well, because it is asymmetric: at ``m = 31``,
    setting ``omega`` 0.05 below the optimum costs several times more iterations than setting it
    0.05 above. That is the practical reason a code should overshoot when it has to guess.
    """
    rows = []
    for m in sizes:
        unknowns = int(m)
        h = 1.0 / (unknowns + 1)
        a = five_point_matrix(unknowns, h)
        predicted = optimal_omega(h)
        general = float(omega_from_the_matrix(a))
        grid = np.linspace(1.0, 1.995, int(samples))
        radii = np.asarray([spectral_radius(iteration_matrix(a, "sor", omega=float(w)))
                            for w in grid])
        best = int(np.argmin(radii))
        jacobi_radius = spectral_radius(iteration_matrix(a, "jacobi"))
        gs_radius = spectral_radius(iteration_matrix(a, "gauss-seidel"))
        rows.append({
            "unknowns": unknowns,
            "h": h,
            "predicted_omega": predicted,
            "swept_omega": float(grid[best]),
            "general_formula_omega": general,
            "gap_to_the_sweep": abs(float(grid[best]) - predicted),
            "gap_to_the_general_formula": abs(general - predicted),
            "best_radius": float(radii[best]),
            "gauss_seidel_radius": float(gs_radius),
            "jacobi_radius": float(jacobi_radius),
            "jacobi_radius_formula": math.cos(math.pi * h),
            # iterations go like log(tol)/log(rho), so the gain is the ratio of the logs the
            # other way up
            "rate_gain": float(math.log(radii[best]) / math.log(gs_radius)),
            "radius_low": float(np.interp(max(predicted - 0.05, grid[0]), grid, radii)),
            "radius_high": float(np.interp(min(predicted + 0.05, grid[-1]), grid, radii)),
        })
    for row in rows:
        # iterations go like 1 / |log rho|, so the cost of a wrong omega is this ratio
        row["cost_of_being_low"] = (math.log(row["best_radius"])
                                    / math.log(row["radius_low"]))
        row["cost_of_being_high"] = (math.log(row["best_radius"])
                                     / math.log(row["radius_high"]))

    # confirm on real runs at the largest size, since a spectral radius is an asymptotic rate
    problem = mixed_modes_problem()
    largest = int(run_size)
    h = 1.0 / (largest + 1)
    best = optimal_omega(h)
    counts = {}
    for label, omega in (("optimal", best), ("0.05 low", best - 0.05),
                         ("0.05 high", min(best + 0.05, 1.995))):
        counts[label] = solve(problem, largest + 2, method="sor", omega=float(omega),
                              tol=1e-10, max_iter=200000,
                              keep_history=False)["iterations"]
    counts["gauss seidel"] = solve(problem, largest + 2, method="gauss seidel", tol=1e-10,
                                   max_iter=400000, keep_history=False)["iterations"]
    return {
        "rows": rows,
        "iteration_counts": counts,
        "the_sweep_finds_the_formula": all(
            row["gap_to_the_sweep"] <= 0.01 for row in rows),
        "the_general_formula_agrees": all(
            row["gap_to_the_general_formula"] < 1e-9 for row in rows),
        "the_jacobi_radius_is_cos_pi_h": all(
            abs(row["jacobi_radius"] - row["jacobi_radius_formula"]) < 1e-10 for row in rows),
        "worst_gap": max(row["gap_to_the_sweep"] for row in rows),
        "rate_gain_grows_with_the_grid": all(
            rows[i]["rate_gain"] < rows[i + 1]["rate_gain"] for i in range(len(rows) - 1)),
        "best_rate_gain": max(row["rate_gain"] for row in rows),
        "undershooting_costs_more_than_overshooting": all(
            row["cost_of_being_low"] > row["cost_of_being_high"] for row in rows),
        "measured_gain_over_gauss_seidel": float(
            counts["gauss seidel"] / counts["optimal"]),
        "measured_cost_of_being_low": float(counts["0.05 low"] / counts["optimal"]),
        "measured_cost_of_being_high": float(counts["0.05 high"] / counts["optimal"]),
    }


def how_the_solvers_scale(sizes=(7, 15, 31), tol: float = 1e-10):
    """Count iterations for four methods on the same matrix and fit each exponent in ``N``.

    The predictions from Part 4 are: Jacobi and Gauss-Seidel take ``O(N)`` iterations, SOR with
    the optimal factor takes ``O(sqrt(N))``, and conjugate gradients takes ``O(sqrt(kappa))``,
    which is also ``O(sqrt(N))`` but with a far better constant. Fitting all four on the same
    problem is the only way to compare them honestly, and the fits are reported next to the
    predictions rather than instead of them.

    The problem used is `mixed_modes_problem` and **not** the single sine mode, because that one
    is an eigenvector of the matrix and conjugate gradients finishes it in one iteration at every
    size. `the_obvious_problem_is_degenerate_for_cg` measures that separately, since it is worth
    knowing rather than quietly avoiding.
    """
    problem = mixed_modes_problem()
    methods = ("jacobi", "gauss seidel", "sor", "cg")
    counts = {name: [] for name in methods}
    unknowns = []
    for m in sizes:
        points = int(m) + 2
        h = 1.0 / (int(m) + 1)
        for name in methods:
            options = {"tol": tol, "keep_history": False}
            if name == "sor":
                options["omega"] = optimal_omega(h)
            if name in ("jacobi", "gauss seidel", "sor"):
                options["max_iter"] = 400000
            out = solve(problem, points, method=name, **options)
            counts[name].append(out["iterations"])
        unknowns.append(int(m) ** 2)
    unknowns = np.asarray(unknowns, dtype=float)
    rows = []
    predicted = {"jacobi": 1.0, "gauss seidel": 1.0, "sor": 0.5, "cg": 0.5}
    for name in methods:
        values = np.asarray(counts[name], dtype=float)
        rows.append({
            "method": name,
            "iterations": values.astype(int),
            "exponent_in_N": float(np.polyfit(np.log(unknowns), np.log(values), 1)[0]),
            "predicted_exponent": predicted[name],
        })
    return {
        "rows": rows,
        "unknowns": unknowns.astype(int),
        "every_exponent_is_close_to_its_prediction": all(
            abs(row["exponent_in_N"] - row["predicted_exponent"]) < 0.2 for row in rows),
        "cheapest_at_the_largest": min(
            rows, key=lambda row: row["iterations"][-1])["method"],
        "jacobi_over_cg": float(counts["jacobi"][-1] / counts["cg"][-1]),
        "sor_over_cg": float(counts["sor"][-1] / counts["cg"][-1]),
    }


# --------------------------------------------------------------------------- finite elements


def triangle_stiffness(vertices):
    """The 3 by 3 element stiffness matrix of a linear triangle, in closed form.

    For piecewise linear basis functions on a triangle the gradients are constant, so the element
    integral is the area times an outer product of gradients. Writing it out is three lines and it
    is exact: no quadrature is involved anywhere.
    """
    v = np.asarray(vertices, dtype=float)
    if v.shape != (3, 2):
        raise ValueError(f"a triangle has three vertices in the plane, got {v.shape}")
    x, y = v[:, 0], v[:, 1]
    corners = v.shape[0]
    twice_area = ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
    if abs(twice_area) < 1e-14:
        raise ValueError("the triangle is degenerate")
    # the gradients of the barycentric coordinates
    grad = np.empty((corners, v.shape[1]))
    for i in range(corners):
        j, k = (i + 1) % corners, (i + 2) % corners
        grad[i] = np.asarray([y[j] - y[k], x[k] - x[j]]) / twice_area
    return abs(twice_area) / 2.0 * (grad @ grad.T)


def right_triangle_mesh(points: int, a: float = 0.0, b: float = 1.0):
    """Split every square of a uniform grid along one diagonal, giving ``2(n-1)**2`` triangles.

    The simplest two dimensional mesh there is, and the one whose assembled stiffness matrix turns
    out to be the five point stencil exactly.
    """
    n = int(points)
    if n < 2:
        raise ValueError(f"need at least 2 points per side, got {n}")
    line = np.linspace(float(a), float(b), n)
    gx, gy = np.meshgrid(line, line, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    cells = []
    for i in range(n - 1):
        for j in range(n - 1):
            lower_left = i * n + j
            lower_right = i * n + j + 1
            upper_left = (i + 1) * n + j
            upper_right = (i + 1) * n + j + 1
            cells.append((lower_left, lower_right, upper_right))
            cells.append((lower_left, upper_right, upper_left))
    return {"nodes": nodes, "cells": np.asarray(cells, dtype=int), "points": n,
            "h": float(line[1] - line[0]), "line": line}


def assemble_fem(mesh):
    """Assemble the global stiffness matrix from the element matrices, over all nodes."""
    nodes = mesh["nodes"]
    total = nodes.shape[0]
    a = np.zeros((total, total))
    for cell in mesh["cells"]:
        local = triangle_stiffness(nodes[cell])
        for p in range(local.shape[0]):
            for q in range(local.shape[1]):
                a[cell[p], cell[q]] += local[p, q]
    return a


def the_elements_give_the_five_point_stencil(sizes=(4, 6, 9)):
    """Piecewise linear elements on a right triangle mesh give exactly the five point matrix.

    Assemble the finite element stiffness matrix, take the rows belonging to interior nodes, and
    compare against ``h**2`` times the five point matrix. They agree entry for entry, which is the
    two dimensional version of lesson 76's discovery that ``h K = tridiag(-1, 2, -1)``.

    The diagonal picks up ``4`` and the four axis neighbours ``-1`` each; the two diagonal
    neighbours that the triangulation creates contribute ``+1`` and ``-1`` from the two triangles
    they share and **cancel exactly**. That cancellation is what makes the two methods identical
    here, and it is the reason a stencil argument and an element argument reach the same matrix.

    As in one dimension, the two methods still differ in the right hand side: finite differences
    evaluate the source at a point, elements integrate it against a basis function.
    """
    rows = []
    for n in sizes:
        mesh = right_triangle_mesh(int(n))
        total = assemble_fem(mesh)
        m = int(n) - 2
        h = mesh["h"]
        interior = [i * int(n) + j for i in range(1, int(n) - 1) for j in range(1, int(n) - 1)]
        block = total[np.ix_(interior, interior)]
        stencil = five_point_matrix(m, h) * h ** 2
        rows.append({
            "points": int(n),
            "unknowns": m,
            "triangles": int(mesh["cells"].shape[0]),
            "largest_gap": float(np.max(np.abs(block - stencil))),
            "matches": float(np.max(np.abs(block - stencil))) < 1e-12,
            "diagonal": float(block[0, 0]),
            "off_diagonal": float(block[0, 1]),
        })
    return {
        "rows": rows,
        "the_matrices_are_the_same": all(row["matches"] for row in rows),
        "worst_gap": max(row["largest_gap"] for row in rows),
        "the_diagonal_is_four": all(abs(row["diagonal"] - 4.0) < 1e-12 for row in rows),
        "note": "the two diagonal neighbours the triangulation creates cancel exactly between "
                "the two triangles that share them, which is why no fifth and sixth entry "
                "appears",
    }


def load_vector(mesh, source, rule: str = "vertex"):
    """The finite element load vector, by one of two quadrature rules on each triangle.

    ``vertex``: ``area/3`` times the source at each of the three corners. Exact for a linear
    source, second order otherwise, and the rule every introductory treatment uses.

    ``centroid``: ``area/3`` times the source at the **centroid**, the same to each corner. Also
    second order, and a genuinely different rule.

    Which one is used turns out to decide whether the finite element method is a different method
    from finite differences at all, which is what `the_two_methods_can_be_the_same_method`
    measures.
    """
    nodes = mesh["nodes"]
    out = np.zeros(nodes.shape[0])
    for cell in mesh["cells"]:
        corners = nodes[cell]
        area = abs((corners[1, 0] - corners[0, 0]) * (corners[2, 1] - corners[0, 1])
                   - (corners[2, 0] - corners[0, 0]) * (corners[1, 1] - corners[0, 1])) / 2.0
        if rule == "vertex":
            values = np.asarray(source(corners[:, 0], corners[:, 1]), dtype=float)
        elif rule == "centroid":
            middle = corners.mean(axis=0)
            values = np.full(cell.shape[0],
                             float(source(middle[0], middle[1])))
        else:
            raise ValueError(f"unknown rule {rule!r}")
        out[cell] += area / 3.0 * values
    return out


def solve_fem(problem, points: int, rule: str = "vertex"):
    """Assemble and solve the finite element system, and report the error at the nodes."""
    mesh = right_triangle_mesh(int(points), problem["a"], problem["b"])
    total = assemble_fem(mesh)
    nodes = mesh["nodes"]
    inside = np.flatnonzero(
        (nodes[:, 0] > problem["a"] + 1e-12) & (nodes[:, 0] < problem["b"] - 1e-12)
        & (nodes[:, 1] > problem["a"] + 1e-12) & (nodes[:, 1] < problem["b"] - 1e-12))
    edge = np.setdiff1d(np.arange(nodes.shape[0]), inside)
    known = np.zeros(nodes.shape[0])
    known[edge] = problem["boundary"](nodes[edge, 0], nodes[edge, 1])
    load = load_vector(mesh, problem["source"], rule=rule)
    rhs = load[inside] - total[np.ix_(inside, edge)] @ known[edge]
    u = np.linalg.solve(total[np.ix_(inside, inside)], rhs)
    want = np.asarray(problem["exact"](nodes[inside, 0], nodes[inside, 1]), dtype=float)
    return {"u": u, "exact": want, "h": mesh["h"], "rule": rule,
            "error": float(np.max(np.abs(u - want))), "rhs": rhs,
            "interior": inside, "mesh": mesh}


def the_two_methods_can_be_the_same_method(sizes=(5, 9, 17, 33), problem=None):
    """Finite elements with the vertex rule are **identical** to finite differences, solution and all.

    Lesson 76 found that in one dimension the assembled stiffness matrix is exactly the second
    difference and only the right hand side differs. In two dimensions on a right triangle mesh
    the matrix is again exactly the five point stencil, and this measurement goes one step
    further: with the vertex quadrature rule **the right hand sides are the same too**, so the two
    methods produce bit for bit the same numbers.

    The reason is a count. Six triangles meet at an interior node, each of area ``h**2/2``, and the
    vertex rule gives each of them ``area/3 = h**2/6`` times the value at that node. Six times
    ``h**2/6`` is ``h**2``, which is exactly what the difference method's right hand side is after
    multiplying through by ``h**2``.

    Change the quadrature to the centroid rule, which is just as accurate, and they part company.
    Both stay second order, and the errors differ by a few per cent, so the choice of quadrature
    is the whole of the difference between the two methods here.
    """
    problem = manufactured_problem() if problem is None else problem
    rows, hs = [], []
    for n in sizes:
        points = int(n)
        difference = solve(problem, points, method="direct")
        vertex = solve_fem(problem, points, rule="vertex")
        centroid = solve_fem(problem, points, rule="centroid")
        rows.append({
            "points": points,
            "h": vertex["h"],
            "finite_difference_error": difference["error"],
            "vertex_rule_error": vertex["error"],
            "centroid_rule_error": centroid["error"],
            "vertex_matches_differences": float(
                np.max(np.abs(np.sort(vertex["u"]) - np.sort(difference["u"])))),
            "centroid_differs_by": float(
                abs(centroid["error"] - difference["error"]) / difference["error"]),
        })
        hs.append(vertex["h"])
    hs = np.asarray(hs)
    fd = np.asarray([row["finite_difference_error"] for row in rows])
    vx = np.asarray([row["vertex_rule_error"] for row in rows])
    ct = np.asarray([row["centroid_rule_error"] for row in rows])
    return {
        "rows": rows,
        "h": hs,
        "finite_difference_order": float(np.polyfit(np.log(hs), np.log(fd), 1)[0]),
        "vertex_rule_order": float(np.polyfit(np.log(hs), np.log(vx), 1)[0]),
        "centroid_rule_order": float(np.polyfit(np.log(hs), np.log(ct), 1)[0]),
        "the_vertex_rule_is_the_difference_method": all(
            row["vertex_matches_differences"] < 1e-11 for row in rows),
        "the_centroid_rule_is_not": all(
            row["centroid_differs_by"] > 1e-6 for row in rows),
        "all_three_are_second_order": all(
            abs(float(np.polyfit(np.log(hs), np.log(values), 1)[0]) - 2.0) < 0.15
            for values in (fd, vx, ct)),
        "centroid_over_difference_at_the_finest": float(ct[-1] / fd[-1]),
        "note": "six triangles of area h^2/2 meet at a node and the vertex rule gives each "
                "h^2/6 of the value there, which sums to exactly the difference method's h^2 f",
    }


def the_obvious_problem_is_degenerate_for_cg(sizes=(7, 15, 31)):
    """The natural test problem is the worst possible test for a Krylov method.

    The discrete eigenvectors of the five point matrix are the products of sine modes sampled on
    the grid, and ``sin(p pi x) sin(q pi y)`` **is** one of them exactly. So on that problem the
    right hand side is a single eigenvector, the Krylov space is one dimensional, and conjugate
    gradients converges in **one** iteration at every grid size, however ill conditioned the
    matrix is.

    That is not a fast solver, it is a degenerate test. Measured beside a problem whose right hand
    side spreads over the spectrum, the same method needs a count that grows like ``sqrt(N)``, as
    the theory says.

    The stationary methods are barely affected, because they converge at a rate set by the
    spectral radius of the iteration matrix rather than by which eigenvectors the data contains.
    That difference is itself worth seeing: it is the practical face of the distinction between a
    Krylov method and a fixed point iteration.
    """
    single = manufactured_problem()
    spread = mixed_modes_problem()
    rows = []
    for m in sizes:
        points = int(m) + 2
        h = 1.0 / (int(m) + 1)
        entry = {"unknowns": int(m) ** 2, "h": h}
        for label, problem in (("single mode", single), ("mixed modes", spread)):
            entry[label] = {
                "cg": solve(problem, points, method="cg", tol=1e-10,
                            keep_history=False)["iterations"],
                "gauss seidel": solve(problem, points, method="gauss seidel", tol=1e-10,
                                      max_iter=400000,
                                      keep_history=False)["iterations"],
            }
        # how many distinct eigenvalues the right hand side actually touches
        system = assemble(single, points)
        values, vectors = np.linalg.eigh(system["A"])
        weights = np.abs(vectors.T @ system["b"])
        entry["modes_in_the_single_mode_rhs"] = int(
            np.count_nonzero(weights > 1e-8 * float(np.max(weights))))
        system = assemble(spread, points)
        values, vectors = np.linalg.eigh(system["A"])
        weights = np.abs(vectors.T @ system["b"])
        entry["modes_in_the_mixed_rhs"] = int(
            np.count_nonzero(weights > 1e-8 * float(np.max(weights))))
        rows.append(entry)
    sizes_array = np.asarray([row["unknowns"] for row in rows], dtype=float)
    spread_counts = np.asarray([row["mixed modes"]["cg"] for row in rows], dtype=float)
    return {
        "rows": rows,
        "cg_finishes_the_single_mode_in_one_step": all(
            row["single mode"]["cg"] == 1 for row in rows),
        "the_single_mode_rhs_has_one_component": all(
            row["modes_in_the_single_mode_rhs"] == 1 for row in rows),
        "the_mixed_rhs_has_many": all(
            row["modes_in_the_mixed_rhs"] > 10 for row in rows),
        "cg_exponent_on_the_spread_problem": float(
            np.polyfit(np.log(sizes_array), np.log(spread_counts), 1)[0]),
        "gauss_seidel_barely_notices": all(
            0.8 < row["single mode"]["gauss seidel"] / row["mixed modes"]["gauss seidel"] < 1.25
            for row in rows),
        "note": "a Krylov method's cost depends on which eigenvectors the data contains; a "
                "stationary method's does not",
    }
