"""Tests for `nalib.splines`.

The smoothness class is established rather than assumed: ``S``, ``S'`` and ``S''`` are checked
continuous and ``S'''`` is checked **discontinuous**, because a cubic spline is ``C^2`` and no
more, and a test that only checked the first half would pass for a method that was accidentally
smoother or accidentally not a spline at all.

The four end conditions are compared on convergence order, split between the ends and the
interior, since the end conditions only affect the ends and one number for the whole interval
would hide which half is responsible.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import splines as sp


SIZES = [3, 4, 5, 8, 16, 33]
SPANS = [(0.0, 1.0), (-2.0, 3.0), (10.0, 10.5)]
BUILDERS = ["natural", "parabolic", "not-a-knot"]


def data(lo, hi, n, f=np.exp):
    x = np.linspace(lo, hi, n)
    return x, f(x)


# ---------------------------------------------------------------- construction


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("lo,hi", SPANS)
@pytest.mark.parametrize("name", BUILDERS)
def test_every_spline_interpolates_its_data(n, lo, hi, name):
    x, y = data(lo, hi, n)
    assert sp.interpolates(sp.BUILDERS[name](x, y)) < 1e-12


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("lo,hi", SPANS)
def test_the_clamped_spline_interpolates_and_matches_the_given_slopes(n, lo, hi):
    x, y = data(lo, hi, n)
    s = sp.clamped(x, y, float(np.exp(lo)), float(np.exp(hi)))
    assert sp.interpolates(s) < 1e-12
    assert abs(float(s.derivative(lo)) - np.exp(lo)) < 1e-8 * np.exp(hi)
    assert abs(float(s.derivative(hi)) - np.exp(hi)) < 1e-8 * np.exp(hi)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_too_few_knots_is_rejected(n):
    with pytest.raises(ValueError):
        sp.natural(np.arange(n, dtype=float), np.arange(n, dtype=float))


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        sp.natural([0.0, 1.0, 2.0], [1.0, 2.0])


def test_repeated_knots_are_rejected():
    with pytest.raises(ValueError):
        sp.natural([0.0, 1.0, 1.0], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("name", BUILDERS)
def test_unsorted_knots_are_accepted_and_sorted(name):
    """A caller with data in any order should get the same spline."""
    x = np.array([0.4, 0.0, 1.0, 0.7])
    y = np.exp(x)
    s = sp.BUILDERS[name](x, y)
    assert np.all(np.diff(s.x) > 0)
    assert sp.interpolates(s) < 1e-12


# ---------------------------------------------------------------- smoothness


@pytest.mark.parametrize("n", [4, 6, 10, 20])
@pytest.mark.parametrize("name", BUILDERS)
def test_a_cubic_spline_is_c_two(n, name):
    """``S``, ``S'`` and ``S''`` continuous across every interior knot."""
    x, y = data(0.0, 1.0, n)
    out = sp.smoothness_report(sp.BUILDERS[name](x, y))
    for k in (0, 1, 2):
        assert out[f"jump_in_derivative_{k}"] < 1e-12


@pytest.mark.parametrize("n", [5, 9, 17])
@pytest.mark.parametrize("name", BUILDERS)
def test_and_no_smoother_than_c_two(n, name):
    """**The other half.** ``S'''`` is piecewise constant and jumps at every interior knot.
    Measured at 0.15 to 1.7 while the lower derivatives are at 1e-15."""
    x, y = data(0.0, 1.0, n)
    out = sp.smoothness_report(sp.BUILDERS[name](x, y))
    assert out["jump_in_derivative_3"] > 1e-3


@pytest.mark.parametrize("n", [4, 8, 16])
def test_the_natural_spline_has_zero_second_derivative_at_the_ends(n):
    x, y = data(0.0, 1.0, n)
    s = sp.natural(x, y)
    assert abs(float(s.derivative(0.0, order=2))) < 1e-10
    assert abs(float(s.derivative(1.0, order=2))) < 1e-10


@pytest.mark.parametrize("n", [4, 8, 16])
def test_the_parabolic_spline_has_equal_end_moments(n):
    x, y = data(0.0, 1.0, n)
    M = sp.parabolic(x, y).moments
    assert abs(M[0] - M[1]) < 1e-10 * max(abs(M[1]), 1.0)
    assert abs(M[-1] - M[-2]) < 1e-10 * max(abs(M[-2]), 1.0)


# ---------------------------------------------------------------- exactness


@pytest.mark.parametrize("n", [4, 6, 10])
def test_a_cubic_is_reproduced_exactly_by_the_two_accurate_end_conditions(n):
    """A cubic spline of a cubic should be that cubic. Natural cannot do it, because the cubic
    does not have zero second derivative at the ends, and that is the end condition's error
    showing up in the cleanest possible case."""
    cubic = lambda t: 1.0 + 2.0 * t - 0.5 * t ** 2 + 0.75 * t ** 3
    dcubic = lambda t: 2.0 - t + 2.25 * t ** 2
    x = np.linspace(0.0, 1.0, n)
    y = cubic(x)
    probe = np.linspace(0.0, 1.0, 201)
    truth = cubic(probe)
    scale = float(np.max(np.abs(truth)))
    assert float(np.max(np.abs(np.atleast_1d(sp.not_a_knot(x, y)(probe)) - truth))) < 1e-10 * scale
    s = sp.clamped(x, y, float(dcubic(0.0)), float(dcubic(1.0)))
    assert float(np.max(np.abs(np.atleast_1d(s(probe)) - truth))) < 1e-10 * scale
    nat = float(np.max(np.abs(np.atleast_1d(sp.natural(x, y)(probe)) - truth)))
    assert nat > 1e-6 * scale


# ---------------------------------------------------------------- the tridiagonal system


@pytest.mark.parametrize("n", [4, 10, 50, 200])
def test_the_equispaced_system_is_well_conditioned_at_every_size(n):
    """**The reason splines are cheap.** The stencil is ``[1, 4, 1]`` for every row, its
    eigenvalues lie in ``(2, 6)``, so the condition number is below 3 however many knots there
    are. No pivoting, no growth, and an ``O(n)`` solve."""
    out = sp.equispaced_system(n, 1.0 / n)
    assert np.array_equal(out["stencil"], np.array([1.0, 4.0, 1.0]))
    assert out["bounded_independently_of_n"]
    assert 2.0 < out["eigenvalue_range"][0] and out["eigenvalue_range"][1] < 6.0
    assert out["condition_number"] < 3.0


def test_the_equispaced_system_rejects_a_degenerate_size():
    with pytest.raises(ValueError):
        sp.equispaced_system(1, 0.5)


@pytest.mark.parametrize("n", [5, 20, 100])
def test_the_condition_number_does_not_grow_with_the_number_of_knots(n):
    small = sp.equispaced_system(4, 0.25)["condition_number"]
    big = sp.equispaced_system(n, 1.0 / n)["condition_number"]
    assert big < 1.5 * small + 0.5


# ---------------------------------------------------------------- convergence


def test_the_end_conditions_have_the_orders_the_theory_gives():
    """**Measured on ``exp(2x)``**, fitting the order separately at the ends and in the middle:

    ==============  ==========  ==========  ==========
    end condition   overall     at ends     middle
    ==============  ==========  ==========  ==========
    natural         2.00        2.00        5.97
    parabolic       2.95        2.95        5.41
    not-a-knot      3.92        3.92        4.58
    clamped         3.98        3.98        3.93
    ==============  ==========  ==========  ==========

    The theory gives 2, 3, 4, 4 and the measurement gives 2.00, 2.95, 3.92, 3.98.

    The split is the useful part: the natural spline is ``O(h^2)`` at the ends and ``O(h^6)`` in
    the middle, so its whole penalty is local. Quoting one number would suggest the method is bad
    everywhere, and it is not.
    """
    out = sp.convergence_by_end_condition(lambda t: np.exp(2.0 * t),
                                          df=lambda t: 2.0 * np.exp(2.0 * t),
                                          n_values=[8, 16, 32, 64, 128])
    order = out["order_overall"]
    assert abs(order["natural"] - 2.0) < 0.2
    assert abs(order["parabolic"] - 3.0) < 0.2
    assert abs(order["not-a-knot"] - 4.0) < 0.2
    assert abs(order["clamped"] - 4.0) < 0.2
    assert out["order_in_the_middle"]["natural"] > 4.0
    assert abs(out["order_at_the_ends"]["natural"] - 2.0) < 0.2


@pytest.mark.parametrize("name", BUILDERS)
def test_every_end_condition_converges_on_runge(name):
    """Splines have no Runge phenomenon, for the same reason piecewise linear does not: the
    degree is fixed at 3 forever."""
    from nalib import interperror as ie

    probe = np.linspace(-1.0, 1.0, 2001)
    truth = ie.runge(probe)
    errs = []
    for n in (8, 16, 32, 64):
        x = np.linspace(-1.0, 1.0, n + 1)
        s = sp.BUILDERS[name](x, ie.runge(x))
        errs.append(float(np.max(np.abs(np.atleast_1d(s(probe)) - truth))))
    assert np.all(np.diff(errs) < 0.0)
    assert errs[-1] < 1e-2


# ---------------------------------------------------------------- the minimum curvature theorem


@pytest.mark.parametrize("n", [5, 8, 12])
def test_the_natural_spline_minimises_the_bending_energy(n):
    """**The theorem the word "spline" comes from.** A draughtsman's flexible strip takes the
    shape minimising ``integral (g'')^2``, and the natural cubic spline is exactly that shape."""
    x, y = data(0.0, 1.0, n, f=lambda t: np.sin(4.0 * t))
    out = sp.minimum_curvature(sp.natural(x, y), n_trials=120,
                               rng=np.random.default_rng(5))
    assert out["is_minimum"]
    assert out["perturbations_with_more_energy"] == out["trials"]