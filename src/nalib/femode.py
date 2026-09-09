"""Collocation and finite elements: solving a differential equation by choosing a basis.

The change of view
------------------
Lesson 75 asked for the solution at a set of points. This lesson asks for a **function**, written
as ``u_h = sum c_j phi_j`` for a basis you choose, and turns the differential equation into
equations for the coefficients. Two ways to get those equations:

**Collocation** demands that the equation hold exactly at ``m`` chosen points. It is the obvious
thing to do, it needs no integrals, and with a global smooth basis on Chebyshev points it
converges faster than any power of ``1/m``. Its matrix is dense and not symmetric, and it needs
basis functions smooth enough to differentiate twice.

**Galerkin** demands instead that the residual be **orthogonal** to every basis function. That
takes one integration by parts, after which only first derivatives appear, so the basis can be
merely continuous. With piecewise linear hat functions the matrix is tridiagonal, symmetric and
positive definite, and every entry is an integral over one or two elements.

What the measurements here show
-------------------------------
* On a uniform grid with ``k = 1`` the assembled stiffness matrix is **exactly** the second
  difference: ``h K = tridiag(-1, 2, -1)``. Finite elements and finite differences produce the
  same matrix, and the only thing that differs is the right hand side.
* For ``-u'' = f`` the piecewise linear Galerkin solution is **exact at the nodes**, to machine
  precision, on a uniform grid and on an irregular one alike. Adding a ``c u`` term destroys it.
  `nodal_exactness` measures both.
* The error is second order in the ``L2`` norm and **first** order in the energy norm. Reporting
  one of those as "the" order of the method is the usual mistake, and `orders_in_two_norms`
  reports them side by side.
* A coefficient that jumps is where the weak form earns its keep, and for a sharper reason
  than "the basis needs less smoothness". The strong form contains ``k' u'``, and a jumping
  ``k`` has no derivative, so **the strong form does not exist**. Collocation returns the same
  wrong straight line at every degree from 5 unknowns to 65; Galerkin gets the answer exactly
  with 3. `collocation_cannot_do_a_kink` and `a_jumping_coefficient` measure both halves.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .banded import thomas
from .gaussquad import rule_on


# --------------------------------------------------------------------------- the hat basis


def hat(nodes, index: int, x):
    """The piecewise linear basis function that is 1 at node ``index`` and 0 at every other.

    These are the simplest functions that are continuous, have a square integrable derivative,
    and have **small support**: each one is nonzero over at most two elements. That last property
    is what makes the matrix sparse, and it is the whole reason the method scales.
    """
    v = np.asarray(nodes, dtype=float).ravel()
    i = int(index)
    if not 0 <= i < v.size:
        raise IndexError(f"node {i} is outside a grid of {v.size} points")
    t = np.asarray(x, dtype=float)
    out = np.zeros(t.shape)
    if i > 0:
        left = (t >= v[i - 1]) & (t <= v[i])
        out[left] = (t[left] - v[i - 1]) / (v[i] - v[i - 1])
    if i < v.size - 1:
        # closed at both ends on purpose. Writing this half open would leave node 0 at zero,
        # because it has no left half to set it, and the basis would stop being a partition of
        # unity at exactly one point out of every grid.
        right = (t >= v[i]) & (t <= v[i + 1])
        out[right] = (v[i + 1] - t[right]) / (v[i + 1] - v[i])
    return out


def hat_derivative(nodes, index: int, x):
    """The derivative of `hat`, which is piecewise constant and undefined at the nodes.

    It is exactly this that the weak form allows: the second derivative of a hat function does
    not exist as a function at all, and the strong form of the equation cannot be written down
    for it. After integrating by parts only this piecewise constant is needed.
    """
    v = np.asarray(nodes, dtype=float).ravel()
    i = int(index)
    if not 0 <= i < v.size:
        raise IndexError(f"node {i} is outside a grid of {v.size} points")
    t = np.asarray(x, dtype=float)
    out = np.zeros(t.shape)
    if i > 0:
        left = (t >= v[i - 1]) & (t < v[i])
        out[left] = 1.0 / (v[i] - v[i - 1])
    if i < v.size - 1:
        right = (t >= v[i]) & (t <= v[i + 1])
        out[right] = -1.0 / (v[i + 1] - v[i])
    return out


def evaluate(nodes, coefficients, x):
    """The finite element function ``sum c_j phi_j`` at arbitrary points.

    For hat functions the coefficients **are** the nodal values, so this is piecewise linear
    interpolation of ``coefficients`` on ``nodes``. That coincidence is special to this basis and
    does not hold for the quadratic or cubic ones.
    """
    v = np.asarray(nodes, dtype=float).ravel()
    c = np.asarray(coefficients, dtype=float).ravel()
    if c.size != v.size:
        raise ValueError(f"{c.size} coefficients for {v.size} nodes")
    return np.interp(np.asarray(x, dtype=float), v, c)


# --------------------------------------------------------------------------- assembly


def element_matrices(lo: float, hi: float, k=None, c=None, f=None, order: int = 6) -> dict:
    """The 2 by 2 stiffness and mass matrices and the 2 vector load for one element.

    On the element ``[lo, hi]`` only two hat functions are nonzero, so everything the element
    contributes is 2 by 2. **The whole of assembly is computing these and adding them into the
    right places**, which is why finite element codes are organised element by element rather
    than node by node: the element does not need to know where it is in the mesh.

    The integrals are done by Gauss-Legendre, which is exact for polynomial ``k``, ``c`` and
    ``f`` of degree up to ``2 order - 1``. For constant coefficients that is exact arithmetic,
    and `quadrature_that_is_too_crude` measures what happens when it is not.
    """
    a, b = float(lo), float(hi)
    h = b - a
    if not h > 0.0:
        raise ValueError(f"element [{a}, {b}] has non-positive width")
    points, weights = rule_on(int(order), a, b)
    phi = np.stack([(b - points) / h, (points - a) / h])
    dphi = np.stack([np.full(points.shape, -1.0 / h), np.full(points.shape, 1.0 / h)])
    kv = (np.ones(points.shape) if k is None
          else np.broadcast_to(np.asarray(k(points), dtype=float), points.shape))
    cv = (np.zeros(points.shape) if c is None
          else np.broadcast_to(np.asarray(c(points), dtype=float), points.shape))
    fv = (np.zeros(points.shape) if f is None
          else np.broadcast_to(np.asarray(f(points), dtype=float), points.shape))
    # how many basis functions this element carries comes from the basis, not from a literal:
    # a quadratic element would supply three rows here and nothing below would change
    count = phi.shape[0]
    stiffness = np.empty((count, count))
    mass = np.empty((count, count))
    load = np.empty(count)
    for i in range(count):
        load[i] = float(np.sum(weights * fv * phi[i]))
        for j in range(count):
            stiffness[i, j] = float(np.sum(weights * kv * dphi[i] * dphi[j]))
            mass[i, j] = float(np.sum(weights * cv * phi[i] * phi[j]))
    return {"stiffness": stiffness, "mass": mass, "load": load, "width": h,
            "local_count": int(count), "points": points, "weights": weights}


def assemble(nodes, k=None, c=None, f=None, order: int = 6) -> dict:
    """Loop over the elements, adding each one's contribution into the global system.

    The result is tridiagonal because each hat function overlaps only its two neighbours, and it
    is symmetric because the bilinear form is. Both are consequences of the basis rather than
    choices, and both are checked by `the_matrix_is_symmetric_positive_definite`.

    The three diagonals are returned separately, so the solve stays ``O(n)``.
    """
    v = np.asarray(nodes, dtype=float).ravel()
    if v.size < 3:
        raise ValueError(f"need at least one interior node, got a grid of {v.size} points")
    if np.any(np.diff(v) <= 0.0):
        raise ValueError("the nodes must be strictly increasing")
    n = v.size
    sub = np.zeros(n)
    diag = np.zeros(n)
    sup = np.zeros(n)
    load = np.zeros(n)
    for e in range(n - 1):
        local = element_matrices(v[e], v[e + 1], k, c, f, order)
        block = local["stiffness"] + local["mass"]
        count = int(local["local_count"])
        for i in range(count):
            load[e + i] += float(local["load"][i])
            diag[e + i] += float(block[i, i])
        sub[e + 1] += float(block[count - 1, 0])
        sup[e] += float(block[0, count - 1])
    return {"nodes": v, "sub": sub, "diag": diag, "sup": sup, "load": load,
            "elements": n - 1, "order": int(order)}


def dense(system: dict) -> np.ndarray:
    """The assembled system as a dense matrix, for inspecting and for checking against."""
    n = int(np.asarray(system["diag"]).size)
    A = np.zeros((n, n))
    A[np.arange(n), np.arange(n)] = system["diag"]
    A[np.arange(1, n), np.arange(n - 1)] = np.asarray(system["sub"])[1:]
    A[np.arange(n - 1), np.arange(1, n)] = np.asarray(system["sup"])[:n - 1]
    return A


def galerkin(nodes, f, k=None, c=None, alpha: float = 0.0, beta: float = 0.0,
             order: int = 6) -> dict:
    """Solve ``-(k u')' + c u = f`` with Dirichlet data, by piecewise linear Galerkin.

    Nonzero boundary values are handled by **lifting**: the boundary columns of the assembled
    matrix are multiplied by the known values and moved to the right hand side, which is the same
    thing the finite difference method does with its first and last rows.

    The interior system is tridiagonal, so the solve is Thomas and the whole method is ``O(n)``.
    """
    system = assemble(nodes, k, c, f, order)
    v = system["nodes"]
    rhs = np.asarray(system["load"], dtype=float)[1:-1].copy()
    rhs[0] -= float(system["sub"][1]) * float(alpha)
    rhs[-1] -= float(system["sup"][v.size - 2]) * float(beta)
    inner = thomas(system["sub"][2:-1], system["diag"][1:-1], system["sup"][1:-2], rhs)
    u = np.empty(v.size)
    u[0], u[-1] = float(alpha), float(beta)
    u[1:-1] = inner
    return dict(system, x=v, u=u, unknowns=inner.size, alpha=float(alpha), beta=float(beta))


# --------------------------------------------------------------------------- errors


def l2_error(nodes, u, exact, order: int = 8) -> float:
    """The ``L2`` norm of the error, integrated element by element.

    Measuring only at the nodes would report zero for ``-u'' = f``, which is true and useless:
    the finite element function is wrong **between** the nodes, and that is where its order lives.
    """
    v = np.asarray(nodes, dtype=float).ravel()
    total = 0.0
    for e in range(v.size - 1):
        points, weights = rule_on(int(order), float(v[e]), float(v[e + 1]))
        gap = evaluate(v, u, points) - np.asarray(exact(points), dtype=float)
        total += float(np.sum(weights * gap ** 2))
    return math.sqrt(total)


def energy_error(nodes, u, exact_derivative, k=None, order: int = 8) -> float:
    """The energy norm of the error, which measures the **derivative** and not the value.

    This is the norm the Galerkin method is optimal in: its answer is the closest function in the
    space, measured this way, and no other choice of coefficients does better. It is also one
    order lower than the ``L2`` norm, because differentiating a piecewise linear approximation
    costs an order.
    """
    v = np.asarray(nodes, dtype=float).ravel()
    c = np.asarray(u, dtype=float).ravel()
    total = 0.0
    for e in range(v.size - 1):
        points, weights = rule_on(int(order), float(v[e]), float(v[e + 1]))
        slope = (c[e + 1] - c[e]) / (v[e + 1] - v[e])
        gap = slope - np.asarray(exact_derivative(points), dtype=float)
        kv = (np.ones(points.shape) if k is None
              else np.broadcast_to(np.asarray(k(points), dtype=float), points.shape))
        total += float(np.sum(weights * kv * gap ** 2))
    return math.sqrt(total)


# --------------------------------------------------------------------------- collocation


def chebyshev_values(degree: int, t) -> tuple:
    """``T_j``, ``T_j'`` and ``T_j''`` for ``j = 0 .. degree`` on ``[-1, 1]``, by recurrence.

    Built from ``T_(k+1) = 2 t T_k - T_(k-1)`` and the two recurrences that come from
    differentiating it. Doing it this way keeps the whole family general in the degree, and it
    is stable for the degrees a collocation method actually uses.
    """
    m = int(degree)
    if m < 0:
        raise ValueError(f"degree must be at least 0, got {m}")
    x = np.atleast_1d(np.asarray(t, dtype=float))
    T = np.empty((m + 1, x.size))
    D1 = np.empty((m + 1, x.size))
    D2 = np.empty((m + 1, x.size))
    T[0], D1[0], D2[0] = 1.0, 0.0, 0.0
    if m >= 1:
        T[1], D1[1], D2[1] = x, 1.0, 0.0
    for j in range(1, m):
        T[j + 1] = 2.0 * x * T[j] - T[j - 1]
        D1[j + 1] = 2.0 * T[j] + 2.0 * x * D1[j] - D1[j - 1]
        D2[j + 1] = 4.0 * D1[j] + 2.0 * x * D2[j] - D2[j - 1]
    return T, D1, D2


def collocation(f, a: float, b: float, alpha: float, beta: float, degree: int = 16,
                k=None, c=None) -> dict:
    """Solve ``-(k u')' + c u = f`` by collocating a Chebyshev expansion at Lobatto points.

    The unknowns are the coefficients of ``sum_(j=0)^m c_j T_j``. Two rows impose the boundary
    conditions and ``m - 1`` rows demand the equation at the interior Lobatto points. There are
    no integrals anywhere.

    With constant ``k`` the equation is ``-k u'' + c u = f``, which is what is imposed. Note that
    the ``k' u'`` term of the product rule is **not** included, so a variable ``k`` is only
    handled correctly when it is differentiable and slowly varying; a jumping ``k`` has no strong
    form at all, and `collocation_cannot_do_a_kink` measures what that does.

    The matrix is **dense and unsymmetric**, so this costs ``O(m^3)`` against Galerkin's ``O(n)``.
    For a smooth solution it converges faster than any power of ``1/m``, so ``m`` stays small and
    the comparison is not as one sided as the exponents suggest.
    """
    m = int(degree)
    if m < 2:
        raise ValueError(f"need degree at least 2 to impose two boundary conditions, got {m}")
    lo, hi = float(a), float(b)
    half = 0.5 * (hi - lo)
    # Chebyshev-Lobatto points on [-1, 1], in increasing order, mapped to [lo, hi]
    theta = np.linspace(math.pi, 0.0, m + 1)
    t = np.cos(theta)
    x = lo + half * (t + 1.0)
    T, D1, D2 = chebyshev_values(m, t)
    kv = (np.ones(x.shape) if k is None
          else np.broadcast_to(np.asarray(k(x), dtype=float), x.shape))
    cv = (np.zeros(x.shape) if c is None
          else np.broadcast_to(np.asarray(c(x), dtype=float), x.shape))
    fv = np.broadcast_to(np.asarray(f(x), dtype=float), x.shape)
    A = np.zeros((m + 1, m + 1))
    rhs = np.zeros(m + 1)
    # the chain rule: d/dx = (1/half) d/dt
    for row in range(1, m):
        A[row] = -kv[row] * D2[:, row] / half ** 2 + cv[row] * T[:, row]
        rhs[row] = float(fv[row])
    A[0] = T[:, 0]
    rhs[0] = float(alpha)
    A[m] = T[:, m]
    rhs[m] = float(beta)
    coefficients = np.linalg.solve(A, rhs)

    def solution(query):
        s = (np.asarray(query, dtype=float) - lo) / half - 1.0
        values, _, _ = chebyshev_values(m, s)
        out = coefficients @ values
        # chebyshev_values works on at least 1-D, so a scalar query would come back as a
        # length 1 array; give back the shape that was asked for
        return out if np.ndim(query) else float(out[0])

    return {"coefficients": coefficients, "x": x, "u": solution(x), "solution": solution,
            "matrix": A, "degree": m,
            "symmetric": bool(np.allclose(A, A.T, atol=1e-12)),
            "condition_number": float(np.linalg.cond(A)),
            "unknowns": m + 1}


# --------------------------------------------------------------------------- test problems


def sine_problem(wavenumber: int = 1):
    """``-u'' = (n pi)^2 sin(n pi x)`` on ``[0, 1]`` with ``u(0) = u(1) = 0``.

    Solution ``sin(n pi x)``. Smooth, with an exact answer and an exact derivative, which is what
    the two error norms need.
    """
    w = float(wavenumber) * math.pi

    def f(x):
        return w ** 2 * np.sin(w * np.asarray(x, dtype=float))

    def exact(x):
        return np.sin(w * np.asarray(x, dtype=float))

    def exact_derivative(x):
        return w * np.cos(w * np.asarray(x, dtype=float))

    return {"f": f, "exact": exact, "exact_derivative": exact_derivative,
            "a": 0.0, "b": 1.0, "alpha": 0.0, "beta": 0.0}


def reaction_problem(reaction: float = 1.0, wavenumber: int = 1):
    """``-u'' + c u = ((n pi)^2 + c) sin(n pi x)``, the same solution with a reaction term.

    The term is what breaks nodal exactness, so this is the control for `nodal_exactness`.
    """
    w = float(wavenumber) * math.pi
    r = float(reaction)

    def f(x):
        return (w ** 2 + r) * np.sin(w * np.asarray(x, dtype=float))

    problem = sine_problem(wavenumber)
    return dict(problem, f=f, c=lambda x: np.full_like(np.asarray(x, dtype=float), r),
                reaction=r)


def jumping_coefficient_problem(left: float = 1.0, right: float = 100.0,
                                interface: float = 0.5):
    """``-(k u')' = 0`` with ``k`` jumping at an interface, ``u(0) = 0``, ``u(1) = 1``.

    The flux ``k u'`` is constant, so the solution is piecewise linear with a **kink** at the
    interface: the slope changes by the ratio of the two conductivities. It is exactly
    representable by hat functions when a node sits on the interface, so the finite element
    answer should be exact there and nowhere near exact when the interface falls inside an
    element.
    """
    k1, k2, s = float(left), float(right), float(interface)
    flux = 1.0 / (s / k1 + (1.0 - s) / k2)

    def k(x):
        v = np.asarray(x, dtype=float)
        return np.where(v < s, k1, k2)

    def exact(x):
        v = np.asarray(x, dtype=float)
        return np.where(v < s, flux * v / k1, flux * (s / k1 + (v - s) / k2))

    def exact_derivative(x):
        v = np.asarray(x, dtype=float)
        return np.where(v < s, flux / k1, flux / k2)

    return {"k": k, "f": lambda x: np.zeros_like(np.asarray(x, dtype=float)),
            "exact": exact, "exact_derivative": exact_derivative,
            "a": 0.0, "b": 1.0, "alpha": 0.0, "beta": 1.0,
            "interface": s, "flux": flux, "left": k1, "right": k2}


# --------------------------------------------------------------------------- measurements


def galerkin_orthogonality(n: int = 33, order: int = 10) -> dict:
    """The defining property: the residual is orthogonal to every basis function.

    Galerkin does not make the residual small. It makes it **invisible to the test space**, which
    is a different and weaker thing, and the whole error analysis follows from it. Here the
    residual of the weak form is formed against each interior hat function and its size is
    reported against the size of the individual terms, so cancellation is not mistaken for
    smallness.
    """
    problem = sine_problem()
    grid = np.linspace(problem["a"], problem["b"], int(n))
    out = galerkin(grid, problem["f"], alpha=problem["alpha"], beta=problem["beta"],
                   order=order)
    residuals, scales = [], []
    for i in range(1, grid.size - 1):
        total, magnitude = 0.0, 0.0
        for e in (i - 1, i):
            points, weights = rule_on(int(order), float(grid[e]), float(grid[e + 1]))
            slope = (out["u"][e + 1] - out["u"][e]) / (grid[e + 1] - grid[e])
            dphi = hat_derivative(grid, i, points)
            phi = hat(grid, i, points)
            stiff = float(np.sum(weights * slope * dphi))
            load = float(np.sum(weights * np.asarray(problem["f"](points)) * phi))
            total += stiff - load
            magnitude += abs(stiff) + abs(load)
        residuals.append(abs(total))
        scales.append(magnitude)
    residuals = np.asarray(residuals)
    scales = np.asarray(scales)
    relative = residuals / np.maximum(scales, 1e-300)
    return {"residual": residuals, "term_size": scales, "relative": relative,
            "worst": float(np.max(relative)),
            "orthogonal": bool(float(np.max(relative)) < 1e-12)}


def the_stiffness_matrix_is_the_second_difference(counts=None) -> dict:
    """The assembled matrix against ``tridiag(-1, 2, -1) / h``, entry by entry.

    For ``k = 1`` on a uniform grid the element stiffness matrix is ``[[1, -1], [-1, 1]] / h``,
    and adding them up gives exactly the second difference operator. **Finite elements and
    finite differences produce the same matrix here.** What differs is the right hand side:
    the finite difference method uses ``f(x_i)`` and Galerkin uses ``(1/h) integral f phi_i``,
    which is a weighted average of ``f`` over two elements.

    That difference is ``O(h^2)`` for smooth ``f`` and is the whole reason the two methods have
    the same order. For rough ``f`` it is not small, and the average is the better choice.
    """
    ns = ([5, 9, 17, 33, 65] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        grid = np.linspace(0.0, 1.0, n)
        h = 1.0 / (n - 1)
        system = assemble(grid, f=lambda x: np.ones_like(np.asarray(x, dtype=float)))
        diagonal_gap = float(np.max(np.abs(h * system["diag"][1:-1] - 2.0)))
        off_gap = float(np.max(np.abs(h * system["sup"][1:-2] + 1.0)))
        # the load against the finite difference right hand side h * f(x_i), for f = 1
        load_gap = float(np.max(np.abs(system["load"][1:-1] - h)))
        rows.append((n, h, diagonal_gap, off_gap, load_gap))
    return {"n": np.asarray([r[0] for r in rows]), "h": np.asarray([r[1] for r in rows]),
            "diagonal_gap": np.asarray([r[2] for r in rows]),
            "off_diagonal_gap": np.asarray([r[3] for r in rows]),
            "load_gap_for_constant_f": np.asarray([r[4] for r in rows]),
            "same_matrix": bool(max(r[2] for r in rows) < 1e-12
                                and max(r[3] for r in rows) < 1e-12),
            "same_load_for_constant_f": bool(max(r[4] for r in rows) < 1e-15)}


def the_load_vector_is_what_differs(counts=None, wavenumber: int = 3) -> dict:
    """How far the Galerkin load is from the finite difference right hand side.

    Galerkin's entry is ``integral f phi_i``, which for a uniform grid equals ``h`` times a
    weighted average of ``f`` over ``[x_(i-1), x_(i+1)]``. The finite difference method uses
    ``h f(x_i)``. For smooth ``f`` the two agree to ``O(h^3)`` per entry, which after dividing by
    ``h`` is the ``O(h^2)`` both methods already have. So they have the same order and different
    constants, and neither is a special case of the other.
    """
    ns = ([9, 17, 33, 65, 129] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    problem = sine_problem(wavenumber)
    rows = []
    for n in ns:
        grid = np.linspace(problem["a"], problem["b"], n)
        h = float(grid[1] - grid[0])
        system = assemble(grid, f=problem["f"])
        pointwise = h * np.asarray(problem["f"](grid[1:-1]), dtype=float)
        gap = np.abs(system["load"][1:-1] - pointwise)
        rows.append((n, h, float(np.max(gap)), float(np.max(np.abs(pointwise)))))
    hs = np.asarray([r[1] for r in rows])
    gaps = np.asarray([r[2] for r in rows])
    return {"n": np.asarray([r[0] for r in rows]), "h": hs, "difference": gaps,
            "load_size": np.asarray([r[3] for r in rows]),
            "relative": gaps / np.maximum(np.asarray([r[3] for r in rows]), 1e-300),
            "fitted_order": float(np.polyfit(np.log(hs), np.log(gaps), 1)[0]),
            "is_third_order_per_entry": bool(
                abs(float(np.polyfit(np.log(hs), np.log(gaps), 1)[0]) - 3.0) < 0.2)}


def nodal_exactness(counts=None, reaction: float = 1.0, seed: int = 42) -> dict:
    """The piecewise linear Galerkin answer at the nodes, with and without a reaction term.

    For ``-u'' = f`` the answer is exact at every node to machine precision, on a uniform grid
    and on a randomly jittered one alike. Measured on the sine problem:

        n                    5        9       17       33       65
        uniform grid     1.2e-16  1.2e-16  3.1e-16  3.1e-15  6.0e-15
        irregular grid   6.7e-16  6.7e-16  1.6e-15  4.7e-15  9.5e-15
        with c u = ...   4.6e-03  1.2e-03  3.0e-04  7.4e-05  1.8e-05

    The reason is that the Green's function of ``-d^2/dx^2`` is piecewise linear with a kink at
    the source point, so it lies **in the finite element space** whenever the source sits at a
    node. Adding ``c u`` makes the Green's function a hyperbolic sine, which does not, and the
    last row is ordinary second order convergence.

    **This is a property of the operator, not a property of finite elements**, and quoting it as
    a general advantage of the method would be wrong.
    """
    ns = ([5, 9, 17, 33, 65] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    plain = sine_problem()
    with_reaction = reaction_problem(reaction)
    rows = []
    for n in ns:
        uniform = np.linspace(plain["a"], plain["b"], n)
        rng = np.random.default_rng(int(seed))
        jitter = np.zeros(n)
        jitter[1:-1] = rng.uniform(-0.4, 0.4, n - 2) / (n - 1)
        irregular = np.sort(uniform + jitter)
        a = galerkin(uniform, plain["f"], alpha=plain["alpha"], beta=plain["beta"])
        b = galerkin(irregular, plain["f"], alpha=plain["alpha"], beta=plain["beta"])
        d = galerkin(uniform, with_reaction["f"], c=with_reaction["c"],
                     alpha=with_reaction["alpha"], beta=with_reaction["beta"])
        rows.append((n,
                     float(np.max(np.abs(a["u"] - plain["exact"](uniform)))),
                     float(np.max(np.abs(b["u"] - plain["exact"](irregular)))),
                     float(np.max(np.abs(d["u"] - with_reaction["exact"](uniform))))))
    uniform_errors = np.asarray([r[1] for r in rows])
    irregular_errors = np.asarray([r[2] for r in rows])
    reaction_errors = np.asarray([r[3] for r in rows])
    hs = 1.0 / (np.asarray([r[0] for r in rows], dtype=float) - 1.0)
    return {"n": np.asarray([r[0] for r in rows]),
            "uniform_nodal_error": uniform_errors,
            "irregular_nodal_error": irregular_errors,
            "with_reaction_nodal_error": reaction_errors,
            "exact_on_a_uniform_grid": bool(float(np.max(uniform_errors)) < 1e-13),
            "exact_on_an_irregular_grid": bool(float(np.max(irregular_errors)) < 1e-13),
            "reaction_term_destroys_it": bool(float(np.min(reaction_errors)) > 1e-8),
            "reaction_order": float(np.polyfit(np.log(hs), np.log(reaction_errors), 1)[0])}


def orders_in_two_norms(counts=None, wavenumber: int = 1) -> dict:
    """The ``L2`` and energy norm errors, refined together.

    Piecewise linear elements are second order in ``L2`` and **first** order in the energy norm.
    Both are correct statements about the same method and they differ by one, because the energy
    norm measures the derivative and differentiating a piecewise linear approximation costs an
    order.

    Quoting one of them as "the" order of the method is the usual mistake. Which one matters
    depends on what the answer is for: a temperature wants ``L2`` and a heat flux wants the
    energy norm.
    """
    ns = ([9, 17, 33, 65, 129, 257] if counts is None
          else [int(v) for v in np.atleast_1d(counts)])
    problem = sine_problem(wavenumber)
    rows = []
    for n in ns:
        grid = np.linspace(problem["a"], problem["b"], n)
        out = galerkin(grid, problem["f"], alpha=problem["alpha"], beta=problem["beta"])
        rows.append((n, 1.0 / (n - 1),
                     l2_error(grid, out["u"], problem["exact"]),
                     energy_error(grid, out["u"], problem["exact_derivative"]),
                     float(np.max(np.abs(out["u"] - problem["exact"](grid))))))
    hs = np.asarray([r[1] for r in rows])
    l2 = np.asarray([r[2] for r in rows])
    energy = np.asarray([r[3] for r in rows])
    l2_order = float(np.polyfit(np.log(hs), np.log(l2), 1)[0])
    energy_order = float(np.polyfit(np.log(hs), np.log(energy), 1)[0])
    return {"n": np.asarray([r[0] for r in rows]), "h": hs,
            "l2_error": l2, "energy_error": energy,
            "nodal_error": np.asarray([r[4] for r in rows]),
            "l2_order": l2_order, "energy_order": energy_order,
            "l2_is_second_order": bool(abs(l2_order - 2.0) < 0.1),
            "energy_is_first_order": bool(abs(energy_order - 1.0) < 0.1),
            "they_differ_by_one": bool(abs((l2_order - energy_order) - 1.0) < 0.15)}


def the_matrix_is_symmetric_positive_definite(counts=None) -> dict:
    """Symmetry and positive definiteness of the assembled matrix, and how its conditioning grows.

    Symmetry comes from the bilinear form being symmetric, and positive definiteness from the
    energy being positive for any nonzero function. Together they mean the system can be solved
    by Cholesky, or by conjugate gradients without any of Part 5's worries about nonsymmetric
    Krylov methods. **Collocation gives up both**, which `collocation_against_galerkin` measures.

    The condition number grows like ``h^-2``, the same as every second order differential
    operator discretised any way at all.
    """
    ns = ([9, 17, 33, 65] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        grid = np.linspace(0.0, 1.0, n)
        system = assemble(grid, f=lambda x: np.ones_like(np.asarray(x, dtype=float)))
        A = dense(system)[1:-1, 1:-1]
        eigenvalues = np.linalg.eigvalsh(0.5 * (A + A.T))
        rows.append((n, 1.0 / (n - 1), float(np.max(np.abs(A - A.T))),
                     float(np.min(eigenvalues)), float(np.linalg.cond(A))))
    hs = np.asarray([r[1] for r in rows])
    conds = np.asarray([r[4] for r in rows])
    return {"n": np.asarray([r[0] for r in rows]), "h": hs,
            "asymmetry": np.asarray([r[2] for r in rows]),
            "smallest_eigenvalue": np.asarray([r[3] for r in rows]),
            "condition_number": conds,
            "symmetric": bool(max(r[2] for r in rows) < 1e-12),
            "positive_definite": bool(min(r[3] for r in rows) > 0.0),
            "condition_order": float(np.polyfit(np.log(hs), np.log(conds), 1)[0]),
            "condition_grows_like_h_squared": bool(
                abs(float(np.polyfit(np.log(hs), np.log(conds), 1)[0]) + 2.0) < 0.2)}


def quadrature_that_is_too_crude(counts=None, wavenumber: int = 4) -> dict:
    """What happens to the order when the element integrals are not accurate enough.

    A one point Gauss rule is exact for linear integrands, which is enough for the stiffness
    matrix with constant ``k`` and not enough for the load with a varying ``f``. The question is
    whether the resulting error spoils the method's order or merely its constant.

    Measured on ``-u'' = (4 pi)^2 sin(4 pi x)``, in the ``L2`` norm:

        n                 9       17       33       65      129
        one point     0.216   0.0562   0.0142   0.0036  8.9e-04
        two point     0.149   0.0391  0.00991   0.0025  6.2e-04
        exact         0.151   0.0393  0.00992   0.0025  6.2e-04

    The fitted orders are 1.98, 1.98 and 1.98. **Crude quadrature costs a constant, not an
    order**: the one point rule is 43 per cent worse throughout and converges at the same rate,
    and the two point rule is indistinguishable from exact integration.

    That is the general rule and it has a name, the Strang first lemma: a quadrature rule exact
    for polynomials of degree ``2m - 2`` keeps the order of degree ``m`` elements. For linear
    elements that is degree 0, which the one point rule already exceeds.
    """
    ns = ([9, 17, 33, 65, 129] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    problem = sine_problem(wavenumber)
    rows = []
    for n in ns:
        grid = np.linspace(problem["a"], problem["b"], n)
        errors = []
        for order in (1, 2, 8):
            out = galerkin(grid, problem["f"], alpha=problem["alpha"], beta=problem["beta"],
                           order=order)
            errors.append(l2_error(grid, out["u"], problem["exact"]))
        rows.append((n, 1.0 / (n - 1)) + tuple(errors))
    hs = np.asarray([r[1] for r in rows])

    def fit(column):
        values = np.asarray([r[column] for r in rows])
        return float(np.polyfit(np.log(hs), np.log(values), 1)[0])

    orders = {name: fit(2 + j) for j, name in enumerate(("one_point", "two_point", "exact"))}
    return {"n": np.asarray([r[0] for r in rows]), "h": hs,
            "one_point_error": np.asarray([r[2] for r in rows]),
            "two_point_error": np.asarray([r[3] for r in rows]),
            "exact_error": np.asarray([r[4] for r in rows]),
            "one_point_order": orders["one_point"],
            "two_point_order": orders["two_point"],
            "exact_order": orders["exact"],
            "one_point_keeps_the_order": bool(abs(orders["one_point"] - 2.0) < 0.15),
            "two_point_is_enough": bool(abs(orders["two_point"] - 2.0) < 0.1)}


def a_jumping_coefficient(counts=None, left: float = 1.0, right: float = 100.0) -> dict:
    """A coefficient that jumps, with the interface on a node and off it.

    The exact solution is piecewise linear with a kink at the interface, so it lies **in the
    finite element space** when a node sits there and the answer is exact. When the interface
    falls in the middle of an element, no piecewise linear function on that mesh has a kink in
    the right place, and the error is stuck at the size of the kink.

    That is the practical rule for meshing a problem with a material interface: **put a node on
    it**. It is also where the difference from a finite difference method shows: the element
    integral ``integral k phi_i' phi_j'`` sees the jump exactly, and a stencil built from
    ``k(x_i)`` has to guess.
    """
    ns = ([5, 9, 17, 33, 65] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        # with the interface at 1/2 and an odd node count, a node lands on it exactly
        aligned = jumping_coefficient_problem(left, right, 0.5)
        grid = np.linspace(aligned["a"], aligned["b"], n)
        on_node = bool(np.min(np.abs(grid - aligned["interface"])) < 1e-14)
        first = galerkin(grid, aligned["f"], k=aligned["k"], alpha=aligned["alpha"],
                         beta=aligned["beta"], order=8)
        aligned_error = float(np.max(np.abs(first["u"] - aligned["exact"](grid))))
        # move the interface off every node by a fixed fraction of an element
        h = 1.0 / (n - 1)
        offset = jumping_coefficient_problem(left, right, 0.5 + 0.5 * h)
        second = galerkin(grid, offset["f"], k=offset["k"], alpha=offset["alpha"],
                          beta=offset["beta"], order=8)
        offset_error = float(np.max(np.abs(second["u"] - offset["exact"](grid))))
        rows.append((n, on_node, aligned_error, offset_error))
    aligned_errors = np.asarray([r[2] for r in rows])
    offset_errors = np.asarray([r[3] for r in rows])
    hs = 1.0 / (np.asarray([r[0] for r in rows], dtype=float) - 1.0)
    return {"n": np.asarray([r[0] for r in rows]),
            "interface_on_a_node": np.asarray([r[1] for r in rows]),
            "aligned_error": aligned_errors, "offset_error": offset_errors,
            "aligned_is_exact": bool(float(np.max(aligned_errors)) < 1e-12),
            "offset_is_not": bool(float(np.min(offset_errors)) > 1e-6),
            "offset_order": float(np.polyfit(np.log(hs), np.log(offset_errors), 1)[0]),
            "ratio": float(np.max(offset_errors) / max(float(np.max(aligned_errors)), 1e-300))}


def collocation_against_galerkin(degrees=None, counts=None, wavenumber: int = 2) -> dict:
    """The two methods on the same smooth problem, at matched unknown counts.

    Collocation with a global Chebyshev basis converges faster than any power, so at 17 unknowns
    it is already at the roundoff floor while piecewise linear Galerkin is at ``10^-3``. On a
    smooth problem it is simply the better method.

    What it gives up is everything structural. Its matrix is dense, unsymmetric and worse
    conditioned, it needs a basis with two derivatives, and it cannot represent a kink at all, so
    the jumping coefficient problem above is out of reach. **The comparison is not accuracy
    against accuracy, it is smoothness against generality.**
    """
    ms = ([4, 8, 12, 16, 20] if degrees is None else [int(v) for v in np.atleast_1d(degrees)])
    ns = ([5, 9, 17, 33, 65] if counts is None else [int(v) for v in np.atleast_1d(counts)])
    problem = sine_problem(wavenumber)
    probe = np.linspace(problem["a"], problem["b"], 1001)
    want = problem["exact"](probe)
    collocated = []
    for m in ms:
        out = collocation(problem["f"], problem["a"], problem["b"], problem["alpha"],
                          problem["beta"], m)
        collocated.append((m + 1, float(np.max(np.abs(out["solution"](probe) - want))),
                           bool(out["symmetric"]), float(out["condition_number"]),
                           (m + 1) ** 3))
    galerkined = []
    for n in ns:
        grid = np.linspace(problem["a"], problem["b"], n)
        out = galerkin(grid, problem["f"], alpha=problem["alpha"], beta=problem["beta"])
        system = dense(out)[1:-1, 1:-1]
        galerkined.append((n - 2, float(np.max(np.abs(evaluate(grid, out["u"], probe) - want))),
                           bool(np.max(np.abs(system - system.T)) < 1e-12),
                           float(np.linalg.cond(system)), 8 * (n - 2)))
    return {"collocation_unknowns": np.asarray([r[0] for r in collocated]),
            "collocation_error": np.asarray([r[1] for r in collocated]),
            "collocation_condition": np.asarray([r[3] for r in collocated]),
            "collocation_flops": np.asarray([r[4] for r in collocated]),
            "collocation_is_symmetric": bool(any(r[2] for r in collocated)),
            "galerkin_unknowns": np.asarray([r[0] for r in galerkined]),
            "galerkin_error": np.asarray([r[1] for r in galerkined]),
            "galerkin_condition": np.asarray([r[3] for r in galerkined]),
            "galerkin_flops": np.asarray([r[4] for r in galerkined]),
            "galerkin_is_symmetric": bool(all(r[2] for r in galerkined)),
            "collocation_reaches": float(np.min([r[1] for r in collocated])),
            "galerkin_reaches": float(np.min([r[1] for r in galerkined])),
            "collocation_wins_on_a_smooth_problem": bool(
                np.min([r[1] for r in collocated]) < np.min([r[1] for r in galerkined]))}


def collocation_cannot_do_a_kink(degrees=None, left: float = 1.0,
                                 right: float = 100.0) -> dict:
    """The other half of the comparison, and it is not the half you would expect.

    The jumping coefficient solution has a corner, so the obvious guess is that a polynomial
    approximates it slowly. What actually happens is worse and more interesting:

        unknowns          5        9       17       33       65
        collocation   0.4901   0.4901   0.4901   0.4901   0.4901

    **The error does not move at all.** Collocation works from the strong form, which expanded is
    ``-k u'' - k' u' + c u = f``. When ``k`` jumps, ``k'`` is a delta function and the strong form
    **does not exist**. What gets imposed instead is ``-k u'' = 0``, whose solution is the
    straight line from 0 to 1, and every degree returns it exactly. The 0.4901 is the gap between
    that straight line and the true kinked solution at the interface.

    Galerkin never forms ``k'``. The integration by parts moves one derivative onto the test
    function, leaving ``integral k u' v'``, which only needs ``k`` itself. With a node on the
    interface and three unknowns it gets the answer to machine precision.

    **This is why the weak form exists.** It is not a trick for lowering the smoothness
    requirement on the basis; it is the only formulation that has a solution at all when the
    coefficient is not differentiable, which in every real material problem it is not.
    """
    ms = ([4, 8, 16, 32, 64] if degrees is None else [int(v) for v in np.atleast_1d(degrees)])
    problem = jumping_coefficient_problem(left, right, 0.5)
    probe = np.linspace(problem["a"], problem["b"], 2001)
    want = problem["exact"](probe)
    rows = []
    for m in ms:
        out = collocation(problem["f"], problem["a"], problem["b"], problem["alpha"],
                          problem["beta"], m, k=problem["k"])
        rows.append((m + 1, float(np.max(np.abs(out["solution"](probe) - want)))))
    grid = np.linspace(problem["a"], problem["b"], 5)
    element = galerkin(grid, problem["f"], k=problem["k"], alpha=problem["alpha"],
                       beta=problem["beta"], order=8)
    finite_element_error = float(np.max(np.abs(evaluate(grid, element["u"], probe) - want)))
    errors = np.asarray([r[1] for r in rows])
    return {"unknowns": np.asarray([r[0] for r in rows]), "collocation_error": errors,
            "galerkin_unknowns": 3, "galerkin_error": finite_element_error,
            "collocation_stalls": bool(errors[-1] > 0.1 * errors[0]),
            "galerkin_wins": bool(finite_element_error < float(np.min(errors))),
            "how_much_better": float(np.min(errors) / max(finite_element_error, 1e-300))}
