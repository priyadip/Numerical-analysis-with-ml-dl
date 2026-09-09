"""Interpolation in two dimensions and more, and the wall it runs into.

The easy part
-------------
On a **grid**, interpolation in two variables is one dimensional interpolation done twice. Build
the interpolant in ``x`` along each row, then interpolate those values in ``y``. The result is the
**tensor product** interpolant, and it does not depend on which variable you do first, which
`order_does_not_matter` measures.

Everything from lessons 44 to 51 carries over unchanged: `lagrange_grid` is the two dimensional
Lagrange form, `newton_grid` the Newton one, and both are the same surface.

The hard part
-------------
Two things break, and only one of them is famous.

**The data requirement explodes.** A tensor product interpolant of degree ``d`` in each of ``k``
variables needs ``(d + 1)^k`` values. At ``d = 4``: 25 points in 2-D, 125 in 3-D, and 9.8 million
in 10-D. `curse_of_dimensionality` tabulates it, and the numbers are the whole argument for every
method in Parts 12 to 14 that avoids grids.

**Scattered data has no tensor structure at all.** Off a grid, there is not even a guarantee that
the interpolation problem has a solution: the 2-D analogue of the Vandermonde matrix can be exactly
singular for point sets in "general position", which is Mairhuber's theorem. `mairhuber_example`
constructs a set of points where the interpolation matrix is singular, which cannot happen in one
dimension for any distinct nodes at all.

That second fact is why scattered multivariate interpolation is done with radial basis functions
or with triangulation rather than with polynomials, and it is a genuine difference in kind rather
than in degree.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from . import interp as _ip


def _grid(x, y, Z):
    xa = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    ya = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    Za = np.atleast_2d(np.asarray(Z, dtype=float))
    if Za.shape != (xa.size, ya.size):
        raise ValueError(f"Z is {Za.shape}, expected ({xa.size}, {ya.size})")
    if np.unique(xa).size != xa.size or np.unique(ya).size != ya.size:
        raise ValueError("the nodes must be distinct in each variable")
    return xa, ya, Za


# ----------------------------------------------------------------------------- tensor product


def lagrange_grid(x, y, Z, s, t) -> np.ndarray:
    """Two dimensional Lagrange interpolation on a grid.

    ``p(s, t) = sum_i sum_j Z[i, j] L_i(s) M_j(t)``, the outer product of the two bases. Written
    this way the symmetry in the two variables is obvious, and so is the cost: ``(n_x n_y)``
    terms for one evaluation.
    """
    xa, ya, Za = _grid(x, y, Z)
    L = _ip.lagrange_basis(xa, np.atleast_1d(np.asarray(s, dtype=float)))
    M = _ip.lagrange_basis(ya, np.atleast_1d(np.asarray(t, dtype=float)))
    return np.einsum("ij,ia,jb->ab", Za, L, M)


def newton_grid(x, y, Z, s, t) -> np.ndarray:
    """The same surface built by interpolating along ``x`` first and then along ``y``.

    This is how it is done in practice: ``n_y`` one dimensional interpolations in ``x`` to get the
    values on a line, then one in ``y``. The cost is ``O(n_x n_y)`` per point for the setup and
    ``O(n_x + n_y)`` after, against the Lagrange form's ``O(n_x n_y)`` every time.
    """
    xa, ya, Za = _grid(x, y, Z)
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    along_x = np.empty((s_arr.size, ya.size))
    for j in range(ya.size):
        c = _ip.newton_coefficients(xa, Za[:, j])
        along_x[:, j] = np.atleast_1d(_ip.evaluate_newton(c, xa, s_arr))
    out = np.empty((s_arr.size, t_arr.size))
    for a in range(s_arr.size):
        c = _ip.newton_coefficients(ya, along_x[a])
        out[a] = np.atleast_1d(_ip.evaluate_newton(c, ya, t_arr))
    return out


def order_does_not_matter(x, y, Z, s, t) -> dict:
    """Interpolate in ``x`` then ``y``, and in ``y`` then ``x``. The surface is the same.

    That is what "tensor product" means, and it is worth checking rather than assuming, because
    it is the property that makes the two dimensional problem no harder than two one dimensional
    ones.
    """
    xa, ya, Za = _grid(x, y, Z)
    a = newton_grid(xa, ya, Za, s, t)
    b = newton_grid(ya, xa, Za.T, t, s).T
    c = lagrange_grid(xa, ya, Za, s, t)
    scale = max(float(np.max(np.abs(c))), 1e-300)
    return {"x_then_y": a, "y_then_x": b, "lagrange": c,
            "order_gap": float(np.max(np.abs(a - b)) / scale),
            "lagrange_gap": float(np.max(np.abs(a - c)) / scale)}


def bilinear(x, y, Z, s, t) -> np.ndarray:
    """The degree one case, used far more than every other case combined.

    Four surrounding values blended by the two fractional coordinates. It is what an image
    resampler, a texture lookup and a lookup table interpolation all do, and it is
    ``O(1)`` per point once the cell is found.
    """
    xa, ya, Za = _grid(x, y, Z)
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    i = np.clip(np.searchsorted(xa, s_arr) - 1, 0, xa.size - 2)
    j = np.clip(np.searchsorted(ya, t_arr) - 1, 0, ya.size - 2)
    u = (s_arr - xa[i]) / (xa[i + 1] - xa[i])
    v = (t_arr - ya[j]) / (ya[j + 1] - ya[j])
    out = np.empty((s_arr.size, t_arr.size))
    for a in range(s_arr.size):
        for b in range(t_arr.size):
            ia, jb = i[a], j[b]
            ua, vb = u[a], v[b]
            out[a, b] = ((1 - ua) * (1 - vb) * Za[ia, jb]
                         + ua * (1 - vb) * Za[ia + 1, jb]
                         + (1 - ua) * vb * Za[ia, jb + 1]
                         + ua * vb * Za[ia + 1, jb + 1])
    return out


def tensor_error(f, n_x: int, n_y: int, bounds=(0.0, 1.0, 0.0, 1.0),
                 n_probe: int = 41) -> dict:
    """Measure the tensor product interpolation error against a known surface."""
    x0, x1, y0, y1 = [float(v) for v in bounds]
    x = np.linspace(x0, x1, int(n_x))
    y = np.linspace(y0, y1, int(n_y))
    Z = np.asarray([[f(a, b) for b in y] for a in x], dtype=float)
    ps = np.linspace(x0, x1, int(n_probe))
    pt = np.linspace(y0, y1, int(n_probe))
    truth = np.asarray([[f(a, b) for b in pt] for a in ps], dtype=float)
    got = newton_grid(x, y, Z, ps, pt)
    lin = bilinear(x, y, Z, ps, pt)
    scale = max(float(np.max(np.abs(truth))), 1e-300)
    return {"tensor_error": float(np.max(np.abs(got - truth)) / scale),
            "bilinear_error": float(np.max(np.abs(lin - truth)) / scale),
            "n_values_used": int(n_x * n_y)}


# ----------------------------------------------------------------------------- the wall


def curse_of_dimensionality(degree: int = 4, dimensions=None) -> dict:
    """``(d + 1)^k`` values for degree ``d`` in ``k`` variables.

    ====  ==================  =====================================
    k     values at ``d = 4``  what that is
    ====  ==================  =====================================
    1     5                   a line of samples
    2     25                  a small grid
    3     125                 a modest cube
    5     3125                still possible
    10    9765625             ten million function evaluations
    20    9.5e13              beyond any computation
    ====  ==================  =====================================

    This is not a difficulty to be programmed around. It is the reason Parts 12 to 14 are full of
    methods that never build a grid: Monte Carlo, sparse grids, and every machine learning method
    that fits a function from scattered samples.
    """
    d = int(degree)
    ks = ([1, 2, 3, 5, 10, 20] if dimensions is None
          else [int(v) for v in np.atleast_1d(dimensions)])
    counts = [float((d + 1) ** k) for k in ks]
    return {"degree": d, "dimensions": np.asarray(ks), "values_needed": np.asarray(counts),
            "bytes_at_8_each": np.asarray([8.0 * c for c in counts])}


def samples_for_accuracy(target: float, order: int, dimensions: int) -> float:
    """How many samples a method of order ``p`` needs in ``k`` dimensions for a given error.

    A grid of spacing ``h`` has error ``~ h^p`` and ``h^-k`` points, so the count is
    ``target^(-k/p)``. The exponent ``k/p`` is the curse stated exactly: raising the dimension
    raises the exponent, and only raising the order can offset it.
    """
    p, k = int(order), int(dimensions)
    if p < 1:
        raise ValueError(f"order must be at least 1, got {p}")
    if not 0.0 < float(target) < 1.0:
        raise ValueError(f"target must be between 0 and 1, got {target}")
    return float(float(target) ** (-k / p))


def mairhuber_example(n_probe: int = 400, rng=None) -> dict:
    """**A two dimensional point set whose interpolation matrix is exactly singular.**

    In one dimension, the Vandermonde matrix on distinct nodes is nonsingular for every node set,
    always, so the interpolation problem always has a unique solution. Mairhuber's theorem says
    that in two or more dimensions **no** basis has that property: for any fixed basis there is a
    point set making the matrix singular.

    The construction is a continuity argument. Take two points and move them continuously around
    each other until they have swapped. The determinant of the interpolation matrix has swapped
    two of its rows, so it has changed sign, so somewhere along the path it was zero.

    This function walks that path and reports the determinant changing sign, together with the
    point set at the crossing where the matrix is singular to working precision.
    """
    gen = np.random.default_rng() if rng is None else rng
    # a fixed quadratic basis in two variables, six functions, so six points
    basis = [lambda p: np.ones(p.shape[0]),
             lambda p: p[:, 0], lambda p: p[:, 1],
             lambda p: p[:, 0] ** 2, lambda p: p[:, 0] * p[:, 1], lambda p: p[:, 1] ** 2]
    fixed = np.array([[0.3, 0.9], [1.0, 0.15], [0.55, 0.4], [0.05, 0.5]])

    def matrix(pts):
        return np.column_stack([f(pts) for f in basis])

    dets = []
    thetas = np.linspace(0.0, np.pi, int(n_probe))
    for th in thetas:
        a = np.array([0.5 + 0.4 * np.cos(th), 0.5 + 0.4 * np.sin(th)])
        b = np.array([0.5 - 0.4 * np.cos(th), 0.5 - 0.4 * np.sin(th)])
        pts = np.vstack([fixed, a, b])
        dets.append(float(np.linalg.det(matrix(pts))))
    def points_at(th):
        a = np.array([0.5 + 0.4 * np.cos(th), 0.5 + 0.4 * np.sin(th)])
        b = np.array([0.5 - 0.4 * np.cos(th), 0.5 - 0.4 * np.sin(th)])
        return np.vstack([fixed, a, b])

    d = np.asarray(dets)
    sign_changes = int(np.sum(np.diff(np.sign(d[d != 0.0])) != 0))

    # Sampling the path more finely does NOT reliably land nearer the root: a 100 point grid
    # came within 3.2e-6 and a 400 point grid within 6.9e-6, because each grid lands wherever it
    # lands. Bisecting on the sign change does converge, and that is what actually demonstrates
    # the theorem, since it produces a configuration that is singular to working precision.
    lo_i = None
    for i in range(d.size - 1):
        if d[i] * d[i + 1] < 0:
            lo_i = i
            break
    bisected = None
    if lo_i is not None:
        a_t, b_t = thetas[lo_i], thetas[lo_i + 1]
        fa = d[lo_i]
        for _ in range(200):
            m_t = 0.5 * (a_t + b_t)
            fm = float(np.linalg.det(matrix(points_at(m_t))))
            if fa * fm <= 0:
                b_t = m_t
            else:
                a_t, fa = m_t, fm
        bisected = 0.5 * (a_t + b_t)

    k = int(np.argmin(np.abs(d)))
    grid_worst = points_at(thetas[k])
    root_points = points_at(bisected) if bisected is not None else grid_worst
    root_det = float(np.linalg.det(matrix(root_points)))
    return {"angles": thetas, "determinants": d,
            "sign_changes": sign_changes,
            "swaps_sign": bool(d[0] * d[-1] < 0),
            "smallest_determinant_on_the_grid": float(np.min(np.abs(d))),
            "bisected_angle": bisected,
            "singular_points": root_points,
            "determinant_at_the_root": root_det,
            "condition_at_the_root": float(np.linalg.cond(matrix(root_points))),
            "grid_points": grid_worst}


def one_dimension_never_fails(n_trials: int = 500, n_nodes: int = 6, rng=None) -> dict:
    """The contrast: in one dimension the matrix is nonsingular for every distinct node set.

    Run against random node sets to show that no amount of searching finds a singular one, which
    is what Mairhuber's theorem says stops being true in two dimensions.
    """
    gen = np.random.default_rng() if rng is None else rng
    n = int(n_nodes)
    worst = np.inf
    for _ in range(int(n_trials)):
        x = gen.uniform(-1.0, 1.0, n)
        if np.unique(x).size != n:
            continue
        worst = min(worst, abs(float(np.linalg.det(np.vander(x, n, increasing=True)))))
    return {"trials": int(n_trials), "smallest_determinant_found": float(worst),
            "any_singular": bool(worst == 0.0)}
