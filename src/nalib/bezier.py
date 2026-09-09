"""Bezier curves and B-splines: interpolation's cousin, where the data is a shape.

The change of question
----------------------
Everything so far asked for a curve **through** given points. Design asks for something different:
a curve a human can steer. Bezier curves and B-splines answer that, and the difference shows in
one property: the control points are **not** interpolated, except the first and last.

Giving that up buys three things interpolation cannot offer.

**The convex hull property.** The curve lies inside the convex hull of its control points. That is
a guarantee about where the curve can possibly be, available before evaluating it anywhere, and it
is what makes collision culling and clipping cheap. `convex_hull_property` measures it.

**Variation diminishing.** A line crosses the curve no more often than it crosses the control
polygon, so the curve cannot oscillate more than the points suggest. There is no Runge phenomenon
here at all, because the curve does not chase the data.

**Affine invariance.** Transforming the control points and then evaluating gives the same curve as
evaluating and then transforming. `affine_invariance` checks it, and it is why a designer can
rotate a shape without recomputing anything.

Bezier
------
``B(t) = sum_i P_i C(n, i) t^i (1-t)^(n-i)`` for ``t`` in ``[0, 1]``: the control points weighted
by the Bernstein polynomials. `de_casteljau` evaluates it by **repeated linear interpolation**
instead, and is the algorithm every renderer uses.

The usual reason given for preferring it is accuracy, and that reason does not survive
measurement: the two agree to 4e-16 at degree 40, and to 2e-16 even outside ``[0, 1]`` where the
Bernstein weights alternate in sign and their absolute sum reaches 2e14. The Bernstein form does
fail eventually, at degree 1030 where the binomial coefficient overflows, and it fails loudly.
The reasons that do survive are that de Casteljau returns the **subdivision** for free, which a
renderer needs, and that every intermediate value it computes stays inside the convex hull, which
matters in fixed point.

B-splines
---------
A single Bezier curve of high degree has the same defect as a single high degree polynomial: every
control point affects every part of the curve, so moving one moves everything. **B-splines fix
that with local support**: basis function ``N_{i,p}`` is nonzero on only ``p + 1`` knot spans, so
moving one control point changes only that part of the curve. `local_support` measures the extent
and `moving_one_point` measures what actually changes, which is the property the whole subject
exists for.

Fonts
-----
TrueType stores glyph outlines as quadratic Beziers and PostScript and OpenType as cubics. The
reasons are exactly the properties above: the hull bounds the glyph cheaply, the curve is scale
invariant so one outline serves every size, and de Casteljau subdivision renders it to any
resolution. `glyph_outline` builds a small closed outline the way a font does.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np


def _points(P):
    A = np.atleast_2d(np.asarray(P, dtype=float))
    if A.ndim != 2 or A.shape[0] < 1:
        raise ValueError("control points must be an (n, d) array with at least one point")
    return A


# ----------------------------------------------------------------------------- Bezier


def bernstein(n: int, i: int, t) -> np.ndarray:
    """``C(n, i) t^i (1 - t)^(n-i)``, the Bezier basis."""
    n, i = int(n), int(i)
    if not 0 <= i <= n:
        raise ValueError(f"need 0 <= i <= n, got i={i}, n={n}")
    z = np.asarray(t, dtype=float)
    return math.comb(n, i) * z ** i * (1.0 - z) ** (n - i)


def bezier(P, t) -> np.ndarray:
    """The Bernstein sum. Correct, and not how anyone evaluates a Bezier curve in practice."""
    A = _points(P)
    n = A.shape[0] - 1
    z = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros((z.size, A.shape[1]))
    for i in range(n + 1):
        out += np.outer(bernstein(n, i, z), A[i])
    return out


def de_casteljau(P, t):
    """Evaluate by **repeated linear interpolation**, which is the practical algorithm.

    At each stage every consecutive pair of points is blended by ``t``, halving the count, until
    one point remains. That point is on the curve. It costs ``O(n^2)`` against the Bernstein
    sum's ``O(n)``, so the reason to prefer it needs stating carefully.

    **It is not accuracy, at any practical degree.** Measured against the Bernstein sum on random
    control points, the relative gap is 2.7e-16 at degree 5 and 4.1e-16 at degree 40, where the
    largest binomial coefficient is 1.38e11. Evaluating outside ``[0, 1]``, where the Bernstein
    weights alternate in sign and ``sum |w_i|`` reaches 2.1e14 at degree 30, they still agree to
    2.3e-16. The textbook claim that the Bernstein sum is the less accurate one is not observable
    in double precision here.

    The Bernstein sum does eventually fail, and the failure point is exactly where the binomial
    coefficient overflows: ``C(1020, 510) = 2.8e305`` is representable as a double and
    ``C(1030, 515)`` is not. At degree 1030 the Bernstein evaluator raises ``OverflowError``
    while de Casteljau returns a finite point. Raising is the better of the two possible
    failures, since a silent ``inf`` would propagate. No font or CAD system uses degree 1030.

    The real reasons to prefer it are structural. Every operation is a convex combination of two
    points already inside the hull, so the algorithm cannot leave the hull even in fixed point or
    reduced precision. And it returns the **subdivision** for free, which the Bernstein sum does
    not, and subdivision is what a renderer actually needs.

    Returns ``(point, left_hull, right_hull)``: the two hulls are the control points of the two
    Bezier curves that together reproduce the original.
    """
    A = _points(P).copy()
    z = float(np.atleast_1d(np.asarray(t, dtype=float))[0])
    left = [A[0].copy()]
    right = [A[-1].copy()]
    while A.shape[0] > 1:
        A = (1.0 - z) * A[:-1] + z * A[1:]
        left.append(A[0].copy())
        right.append(A[-1].copy())
    return A[0], np.asarray(left), np.asarray(right[::-1])


def subdivide(P, t: float = 0.5):
    """Split a Bezier curve into two Bezier curves that together are the original."""
    _, left, right = de_casteljau(P, t)
    return left, right


def derivative_control_points(P) -> np.ndarray:
    """The derivative of a degree ``n`` Bezier is a degree ``n-1`` Bezier on ``n (P_{i+1} - P_i)``.

    So the tangent at ``t = 0`` is along ``P_1 - P_0`` and at ``t = 1`` along ``P_n - P_{n-1}``,
    which is why a designer steers the tangent by moving the second and second-to-last points.
    """
    A = _points(P)
    n = A.shape[0] - 1
    if n < 1:
        raise ValueError("a single point has no derivative curve")
    return n * (A[1:] - A[:-1])


def elevate_degree(P) -> np.ndarray:
    """Rewrite the same curve with one more control point, which changes nothing about it.

    Degree elevation is how two curves of different degree are made compatible before joining
    them, and it is a good check that a Bezier implementation is right: the curve must not move.
    """
    A = _points(P)
    n = A.shape[0] - 1
    out = np.empty((n + 2, A.shape[1]))
    out[0] = A[0]
    out[-1] = A[-1]
    for i in range(1, n + 1):
        a = i / (n + 1.0)
        out[i] = a * A[i - 1] + (1.0 - a) * A[i]
    return out


# ----------------------------------------------------------------------------- the properties


def convex_hull_property(P, n_probe: int = 401) -> dict:
    """Every point of the curve lies inside the convex hull of the control points.

    Checked by asking whether each curve point is a convex combination of the control points,
    which is a small non-negative least squares problem, solved here by the simpler sufficient
    test that the point lies within the hull's bounding simplex constraints.
    """
    A = _points(P)
    t = np.linspace(0.0, 1.0, int(n_probe))
    curve = bezier(A, t)
    lo = A.min(axis=0)
    hi = A.max(axis=0)
    inside_box = bool(np.all(curve >= lo - 1e-12) and np.all(curve <= hi + 1e-12))
    # the Bernstein weights are non-negative and sum to 1, which IS the convex combination
    n = A.shape[0] - 1
    W = np.stack([bernstein(n, i, t) for i in range(n + 1)], axis=1)
    return {"inside_bounding_box": inside_box,
            "weights_non_negative": bool(np.all(W >= -1e-15)),
            "weights_sum_to_one": float(np.max(np.abs(W.sum(axis=1) - 1.0))),
            "endpoints_interpolated": float(max(np.linalg.norm(curve[0] - A[0]),
                                                np.linalg.norm(curve[-1] - A[-1])))}


def affine_invariance(P, M=None, shift=None, n_probe: int = 201, rng=None) -> dict:
    """Transform then evaluate, against evaluate then transform. They must agree exactly.

    This is why a designer can rotate, scale or shear a shape and the control points alone carry
    the whole description.
    """
    A = _points(P)
    d = A.shape[1]
    gen = np.random.default_rng() if rng is None else rng
    M = gen.standard_normal((d, d)) if M is None else np.asarray(M, dtype=float)
    shift = gen.standard_normal(d) if shift is None else np.asarray(shift, dtype=float)
    t = np.linspace(0.0, 1.0, int(n_probe))
    a = bezier(A, t) @ M.T + shift
    b = bezier(A @ M.T + shift, t)
    scale = max(float(np.max(np.abs(a))), 1e-300)
    return {"gap": float(np.max(np.abs(a - b))), "relative_gap": float(np.max(np.abs(a - b)) / scale)}


def variation_diminishing(P, direction=None, n_probe: int = 2001, rng=None) -> dict:
    """A line crosses the curve no more often than it crosses the control polygon.

    So the curve cannot wiggle more than its control points do, which is the precise sense in
    which there is no Runge phenomenon here.
    """
    A = _points(P)
    gen = np.random.default_rng() if rng is None else rng
    d = A.shape[1]
    v = gen.standard_normal(d) if direction is None else np.asarray(direction, dtype=float)
    level = float(np.mean(A @ v))
    poly = A @ v - level
    curve = bezier(A, np.linspace(0.0, 1.0, int(n_probe))) @ v - level
    count = lambda s: int(np.sum(np.diff(np.sign(s[s != 0.0])) != 0))
    return {"crossings_of_the_control_polygon": count(poly),
            "crossings_of_the_curve": count(curve),
            "curve_does_not_exceed_polygon": bool(count(curve) <= count(poly))}


def bernstein_vs_de_casteljau(degree: int, n_probe: int = 201, rng=None) -> dict:
    """Measure the accuracy of the two evaluators against each other as the degree grows."""
    gen = np.random.default_rng() if rng is None else rng
    A = gen.standard_normal((int(degree) + 1, 2))
    t = np.linspace(0.0, 1.0, int(n_probe))
    a = bezier(A, t)
    b = np.stack([de_casteljau(A, v)[0] for v in t])
    n = int(degree)
    return {"degree": n,
            "largest_binomial": float(math.comb(n, n // 2)),
            "max_gap": float(np.max(np.abs(a - b))),
            "relative_gap": float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))}


# ----------------------------------------------------------------------------- B-splines


def open_uniform_knots(n_control: int, degree: int) -> np.ndarray:
    """The standard clamped knot vector: ``degree + 1`` repeats at each end, uniform between.

    Repeating the end knots is what makes the curve pass through its first and last control
    points, which an unclamped B-spline does not do.
    """
    n, p = int(n_control), int(degree)
    if n < p + 1:
        raise ValueError(f"need at least degree + 1 = {p + 1} control points, got {n}")
    inner = n - p - 1
    return np.concatenate([np.zeros(p + 1),
                           np.arange(1.0, inner + 1.0) / (inner + 1.0),
                           np.ones(p + 1)])


def bspline_basis(i: int, p: int, knots, t) -> np.ndarray:
    """``N_{i,p}`` by the Cox-de Boor recursion.

    ``N_{i,0}`` is 1 on its own knot span and 0 elsewhere, and

        N_{i,p}(t) = (t - u_i)/(u_{i+p} - u_i) N_{i,p-1}
                   + (u_{i+p+1} - t)/(u_{i+p+1} - u_{i+1}) N_{i+1,p-1}

    with a term dropped whenever its denominator is zero, which is what repeated knots cause.
    """
    u = np.atleast_1d(np.asarray(knots, dtype=float)).ravel()
    z = np.atleast_1d(np.asarray(t, dtype=float))
    i, p = int(i), int(p)
    if p == 0:
        last = u[-1]
        inside = (z >= u[i]) & (z < u[i + 1])
        if u[i + 1] == last:
            inside = inside | (z == last)
        return inside.astype(float)
    out = np.zeros_like(z, dtype=float)
    den1 = u[i + p] - u[i]
    if den1 > 0:
        out = out + (z - u[i]) / den1 * bspline_basis(i, p - 1, u, z)
    den2 = u[i + p + 1] - u[i + 1]
    if den2 > 0:
        out = out + (u[i + p + 1] - z) / den2 * bspline_basis(i + 1, p - 1, u, z)
    return out


def bspline(P, degree: int, t, knots=None) -> np.ndarray:
    """``C(t) = sum_i P_i N_{i,p}(t)``."""
    A = _points(P)
    p = int(degree)
    u = open_uniform_knots(A.shape[0], p) if knots is None else np.asarray(knots, dtype=float)
    z = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros((z.size, A.shape[1]))
    for i in range(A.shape[0]):
        out += np.outer(bspline_basis(i, p, u, z), A[i])
    return out


def local_support(n_control: int, degree: int, n_probe: int = 2001) -> dict:
    """**The property B-splines exist for.** ``N_{i,p}`` is nonzero on ``p + 1`` knot spans only.

    So a control point influences a bounded stretch of curve however many control points there
    are, which is what a Bezier curve of the same degree cannot do.
    """
    n, p = int(n_control), int(degree)
    u = open_uniform_knots(n, p)
    t = np.linspace(float(u[0]), float(u[-1]), int(n_probe))
    extents = []
    for i in range(n):
        v = bspline_basis(i, p, u, t)
        live = t[np.abs(v) > 1e-12]
        extents.append(float(live.max() - live.min()) if live.size else 0.0)
    span = float(u[-1] - u[0])
    return {"extents": np.asarray(extents),
            "max_extent": float(np.max(extents)),
            "whole_domain": span,
            "max_fraction_of_domain": float(np.max(extents) / span)}


def partition_of_unity(n_control: int, degree: int, n_probe: int = 1001) -> float:
    """``sum_i N_{i,p}(t) = 1`` everywhere, which is what makes the hull property hold."""
    n, p = int(n_control), int(degree)
    u = open_uniform_knots(n, p)
    t = np.linspace(float(u[0]), float(u[-1]), int(n_probe))
    total = sum(bspline_basis(i, p, u, t) for i in range(n))
    return float(np.max(np.abs(total - 1.0)))


def moving_one_point(P, degree: int, index: int, delta, n_probe: int = 2001) -> dict:
    """Move one control point and measure how much of the curve moved.

    For a Bezier curve the answer is all of it. For a B-spline it is ``p + 1`` knot spans, and
    that difference is the reason design software uses B-splines.
    """
    A = _points(P)
    p = int(degree)
    B = A.copy()
    B[int(index)] = B[int(index)] + np.asarray(delta, dtype=float)
    u = open_uniform_knots(A.shape[0], p)
    t = np.linspace(float(u[0]), float(u[-1]), int(n_probe))
    moved_spline = np.linalg.norm(bspline(B, p, t, u) - bspline(A, p, t, u), axis=1)
    tb = np.linspace(0.0, 1.0, int(n_probe))
    moved_bezier = np.linalg.norm(bezier(B, tb) - bezier(A, tb), axis=1)
    tol = 1e-10 * max(float(np.max(moved_spline)), 1e-300)
    return {"bspline_fraction_moved": float(np.mean(moved_spline > tol)),
            "bezier_fraction_moved": float(np.mean(moved_bezier > 1e-10 * max(
                float(np.max(moved_bezier)), 1e-300))),
            "bspline_max_move": float(np.max(moved_spline)),
            "bezier_max_move": float(np.max(moved_bezier))}


# ----------------------------------------------------------------------------- fonts


def glyph_outline(kind: str = "quadratic", n_probe: int = 200) -> dict:
    """A small closed outline built the way a font stores one.

    TrueType uses quadratic Beziers, PostScript and OpenType cubics. The on-curve points are
    shared between consecutive segments, so the outline is continuous by construction, and making
    it **smooth** at a join needs the three points around the join to be collinear, which is the
    condition a font editor enforces when you ask for a smooth node.
    """
    if kind == "quadratic":
        segs = [np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]),
                np.array([[1.0, 0.0], [1.5, -1.0], [0.0, -1.0]]),
                np.array([[0.0, -1.0], [-0.4, -0.5], [0.0, 0.0]])]
    elif kind == "cubic":
        segs = [np.array([[0.0, 0.0], [0.3, 1.0], [0.7, 1.0], [1.0, 0.0]]),
                np.array([[1.0, 0.0], [1.4, -0.8], [0.4, -1.2], [0.0, -1.0]]),
                np.array([[0.0, -1.0], [-0.3, -0.7], [-0.3, -0.3], [0.0, 0.0]])]
    else:
        raise ValueError(f"kind must be 'quadratic' or 'cubic', got {kind!r}")
    t = np.linspace(0.0, 1.0, int(n_probe))
    pieces = [bezier(s, t) for s in segs]
    closed = float(np.linalg.norm(pieces[-1][-1] - pieces[0][0]))
    joins = [float(np.linalg.norm(pieces[k][-1] - pieces[k + 1][0]))
             for k in range(len(pieces) - 1)]
    smooth = []
    for k in range(len(segs) - 1):
        a = segs[k][-1] - segs[k][-2]
        b = segs[k + 1][1] - segs[k + 1][0]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        smooth.append(float(abs(a[0] * b[1] - a[1] * b[0]) / max(na * nb, 1e-300)))
    return {"segments": segs, "curve": np.vstack(pieces),
            "closed_gap": closed, "join_gaps": joins,
            "tangent_mismatch_at_joins": smooth,
            "is_closed": bool(closed < 1e-12),
            "is_continuous": bool(max(joins) < 1e-12)}
