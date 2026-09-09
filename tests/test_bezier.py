"""Tests for `nalib.bezier`.

The three properties that make Bezier curves worth using instead of interpolants are each tested
directly: the convex hull, affine invariance, and variation diminishing. Local support is tested
by measuring how much of the curve actually moves when one control point does, which is the
practical form of the claim.

The comparison between the two evaluators is measured rather than asserted, and the measurement
contradicts the usual textbook framing, which is recorded here rather than smoothed over.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nalib import bezier as bz


DEGREES = [1, 2, 3, 5, 8, 15]
DIMS = [1, 2, 3]


def control(n, d, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d))


# ---------------------------------------------------------------- Bezier basics


@pytest.mark.parametrize("n", DEGREES)
def test_the_bernstein_basis_is_a_partition_of_unity(n):
    """Non-negative on ``[0, 1]`` and summing to 1, which is exactly what makes every point of
    the curve a convex combination of the control points."""
    t = np.linspace(0.0, 1.0, 501)
    W = np.stack([bz.bernstein(n, i, t) for i in range(n + 1)], axis=1)
    assert np.all(W >= -1e-15)
    assert float(np.max(np.abs(W.sum(axis=1) - 1.0))) < 1e-12


@pytest.mark.parametrize("n", DEGREES)
@pytest.mark.parametrize("d", DIMS)
def test_the_endpoints_are_interpolated_and_nothing_else_need_be(n, d):
    """The one thing a Bezier curve does interpolate. The interior control points are steering
    handles and the curve generally misses them, which is the point."""
    P = control(n + 1, d, seed=n * 10 + d)
    curve = bz.bezier(P, np.linspace(0.0, 1.0, 101))
    assert float(np.linalg.norm(curve[0] - P[0])) < 1e-12
    assert float(np.linalg.norm(curve[-1] - P[-1])) < 1e-12


@pytest.mark.parametrize("n", DEGREES)
@pytest.mark.parametrize("d", DIMS)
def test_de_casteljau_agrees_with_the_bernstein_sum(n, d):
    P = control(n + 1, d, seed=n + d)
    for t in (0.0, 0.17, 0.5, 0.83, 1.0):
        a = bz.bezier(P, [t])[0]
        b = bz.de_casteljau(P, t)[0]
        assert float(np.max(np.abs(a - b))) < 1e-12 * max(float(np.max(np.abs(b))), 1.0)


@pytest.mark.parametrize("n", [2, 3, 5, 8])
@pytest.mark.parametrize("t", [0.25, 0.5, 0.75])
def test_subdivision_reproduces_the_original_curve(n, t):
    """Splitting gives two Bezier curves that together are the original, which is what a renderer
    uses to draw a curve to any resolution."""
    P = control(n + 1, 2, seed=n)
    left, right = bz.subdivide(P, t)
    for s in np.linspace(0.0, 1.0, 41):
        assert float(np.max(np.abs(bz.bezier(left, [s])[0]
                                   - bz.bezier(P, [s * t])[0]))) < 1e-10
        assert float(np.max(np.abs(bz.bezier(right, [s])[0]
                                   - bz.bezier(P, [t + s * (1.0 - t)])[0]))) < 1e-10


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_degree_elevation_does_not_move_the_curve(n):
    """Adding a control point without changing the curve is how two curves of different degree
    are made compatible, and it is a strong check that the basis is right."""
    P = control(n + 1, 2, seed=n + 3)
    Q = bz.elevate_degree(P)
    assert Q.shape[0] == P.shape[0] + 1
    t = np.linspace(0.0, 1.0, 201)
    assert float(np.max(np.abs(bz.bezier(P, t) - bz.bezier(Q, t)))) < 1e-11


def test_a_single_point_has_no_derivative_curve():
    with pytest.raises(ValueError):
        bz.derivative_control_points([[1.0, 2.0]])


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_the_end_tangents_point_along_the_first_and_last_control_edges(n):
    """Which is why a designer steers the tangent with the second and second-to-last points."""
    P = control(n + 1, 2, seed=n + 7)
    D = bz.derivative_control_points(P)
    start = bz.bezier(D, [0.0])[0]
    end = bz.bezier(D, [1.0])[0]
    edge0 = P[1] - P[0]
    edge1 = P[-1] - P[-2]
    cross = lambda a, b: abs(a[0] * b[1] - a[1] * b[0]) / max(
        float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-300)
    assert cross(start, edge0) < 1e-10
    assert cross(end, edge1) < 1e-10


def test_a_bad_bernstein_index_is_rejected():
    with pytest.raises(ValueError):
        bz.bernstein(3, 5, 0.5)


def test_empty_control_points_are_rejected():
    with pytest.raises(ValueError):
        bz.bezier(np.empty((0, 2)), [0.5])


# ---------------------------------------------------------------- the three properties


@pytest.mark.parametrize("n", DEGREES)
@pytest.mark.parametrize("d", DIMS)
def test_the_curve_stays_inside_the_convex_hull(n, d):
    """A guarantee about where the curve can be, available before evaluating it anywhere."""
    out = bz.convex_hull_property(control(n + 1, d, seed=n * 3 + d))
    assert out["inside_bounding_box"]
    assert out["weights_non_negative"]
    assert out["weights_sum_to_one"] < 1e-12


@pytest.mark.parametrize("n", DEGREES)
@pytest.mark.parametrize("d", [2, 3])
def test_the_curve_is_affine_invariant(n, d):
    """Transform the control points or transform the curve: the same answer. It is why a shape
    can be rotated without recomputing anything."""
    out = bz.affine_invariance(control(n + 1, d, seed=n + d * 5),
                               rng=np.random.default_rng(n + d))
    assert out["relative_gap"] < 1e-12


@pytest.mark.parametrize("n", [3, 5, 8, 12])
def test_the_curve_does_not_oscillate_more_than_its_control_polygon(n):
    """Variation diminishing, which is the precise sense in which there is no Runge phenomenon."""
    out = bz.variation_diminishing(control(n + 1, 2, seed=n * 7),
                                   rng=np.random.default_rng(n))
    assert out["curve_does_not_exceed_polygon"]


# ---------------------------------------------------------------- the two evaluators


def test_the_bernstein_sum_is_as_accurate_as_de_casteljau_at_practical_degrees():
    """**Measured, and it contradicts the usual framing.** Relative gap between the two:

    ========  ===================  ==============
    degree    largest binomial     relative gap
    ========  ===================  ==============
    5         1.00e1               2.70e-16
    20        1.85e5               3.62e-16
    40        1.38e11              4.05e-16
    ========  ===================  ==============

    Outside ``[0, 1]``, where the Bernstein weights alternate in sign and ``sum |w_i|`` reaches
    2.1e14 at degree 30, they still agree to 2.3e-16. There is no measurable accuracy advantage
    to de Casteljau in double precision, and the reasons to prefer it are subdivision and the
    hull guarantee instead.
    """
    for d in (5, 10, 20, 30, 40):
        out = bz.bernstein_vs_de_casteljau(d, rng=np.random.default_rng(1))
        assert out["relative_gap"] < 1e-14


def test_the_bernstein_sum_does_eventually_fail_where_the_binomial_overflows():
    """``C(1020, 510) = 2.8e305`` is representable as a double and ``C(1030, 515)`` is not.

    At degree 1030 the Bernstein evaluator raises ``OverflowError`` while de Casteljau returns a
    finite point. Raising is the better of the two possible failures, since a silent ``inf``
    would propagate. No font or CAD system uses degree 1030, which is why this is a footnote
    rather than a reason to choose an evaluator.
    """
    assert math.comb(1020, 510) < 10 ** 308
    assert math.comb(1030, 515) > 10 ** 308
    P = np.random.default_rng(2).standard_normal((1031, 2))
    with pytest.raises(OverflowError):
        bz.bezier(P, [0.5])
    assert np.all(np.isfinite(bz.de_casteljau(P, 0.5)[0]))

    ok = np.random.default_rng(2).standard_normal((1021, 2))
    assert np.all(np.isfinite(bz.bezier(ok, [0.5])))


# ---------------------------------------------------------------- B-splines


@pytest.mark.parametrize("n,p", [(4, 2), (6, 2), (6, 3), (10, 3), (16, 4)])
def test_the_bspline_basis_is_a_partition_of_unity(n, p):
    assert bz.partition_of_unity(n, p) < 1e-10


@pytest.mark.parametrize("n,p", [(6, 2), (10, 2), (10, 3), (16, 3), (30, 3)])
def test_a_bspline_basis_function_has_local_support(n, p):
    """**The property B-splines exist for.** ``N_{i,p}`` is nonzero on ``p + 1`` knot spans only,
    so its extent is ``(p + 1)/(inner + 1)`` of the domain, measured to three digits:

    ==========  ========  ==============  ============
    controls    degree    max extent      predicted
    ==========  ========  ==============  ============
    10          2         0.3740          0.3750
    16          3         0.3075          0.3077
    30          3         0.1480          0.1481
    ==========  ========  ==============  ============
    """
    out = bz.local_support(n, p)
    inner = n - p - 1
    predicted = (p + 1) / (inner + 1.0) if inner > 0 else 1.0
    assert out["max_fraction_of_domain"] <= min(predicted, 1.0) + 0.01


def test_more_control_points_shrink_the_support_of_each_one():
    fractions = [bz.local_support(n, 3)["max_fraction_of_domain"] for n in (10, 16, 24, 30)]
    assert fractions == sorted(fractions, reverse=True)


def test_moving_one_control_point_moves_a_bounded_stretch_of_a_bspline():
    """**The comparison that settles it.** Fraction of the curve that moves when one control
    point is moved, degree 3:

    ==========  ==============  ============
    controls    B-spline        Bezier
    ==========  ==============  ============
    8           0.799           0.999
    12          0.444           0.991
    20          0.235           0.954
    30          0.148           0.891
    ==========  ==============  ============

    The B-spline fraction falls toward zero as the curve grows and the Bezier fraction stays
    near 1. That is why design software uses B-splines.
    """
    g = np.random.default_rng(3)
    fracs = []
    for n in (8, 12, 20, 30):
        P = np.column_stack([np.arange(n, dtype=float), g.standard_normal(n)])
        out = bz.moving_one_point(P, 3, n // 2, [0.0, 1.0])
        fracs.append(out["bspline_fraction_moved"])
        assert out["bspline_fraction_moved"] < out["bezier_fraction_moved"]
        assert out["bezier_fraction_moved"] > 0.85
    assert fracs == sorted(fracs, reverse=True)
    assert fracs[-1] < 0.2


@pytest.mark.parametrize("n,p", [(4, 3), (6, 3), (8, 2)])
def test_a_clamped_bspline_passes_through_its_first_and_last_control_points(n, p):
    P = control(n, 2, seed=n * p)
    u = bz.open_uniform_knots(n, p)
    curve = bz.bspline(P, p, np.linspace(float(u[0]), float(u[-1]), 401), u)
    assert float(np.linalg.norm(curve[0] - P[0])) < 1e-10
    assert float(np.linalg.norm(curve[-1] - P[-1])) < 1e-10


def test_too_few_control_points_for_the_degree_is_rejected():
    with pytest.raises(ValueError):
        bz.open_uniform_knots(3, 5)


# ---------------------------------------------------------------- fonts


@pytest.mark.parametrize("kind", ["quadratic", "cubic"])
def test_a_glyph_outline_is_closed_and_continuous(kind):
    """TrueType stores quadratics and PostScript cubics. Both are closed by sharing the on-curve
    points between consecutive segments, so continuity is structural rather than enforced."""
    out = bz.glyph_outline(kind)
    assert out["is_closed"]
    assert out["is_continuous"]
    # the curve lives in whatever space the control points do, taken from them rather than
    # assumed, even though a glyph is always planar
    assert out["curve"].shape[1] == out["segments"][0].shape[1]


def test_a_bad_glyph_kind_is_rejected():
    with pytest.raises(ValueError):
        bz.glyph_outline("septic")