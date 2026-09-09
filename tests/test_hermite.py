"""Tests for `nalib.hermite`.

The Hermite polynomial is built two independent ways and the two are compared, so agreement is
evidence. The piecewise bound is checked as a bound and its sharpness is measured, since the
whole argument for piecewise methods is that the bound is unconditional.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nalib import hermite as hm, interp as ip


SIZES = [2, 3, 4, 6, 8]
SPANS = [(0.0, 1.0), (-1.0, 1.0), (2.0, 5.0)]


# ---------------------------------------------------------------- Hermite


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("lo,hi", SPANS)
def test_the_two_hermite_constructions_agree(n, lo, hi):
    """The confluent Newton table and the classical basis are independent computations."""
    x = np.linspace(lo, hi, n)
    y, dy = np.exp(x), np.exp(x)
    c, z = hm.hermite_newton(x, y, dy)
    probe = np.linspace(lo, hi, 201)
    a = ip.evaluate_newton(c, z, probe)
    b = hm.hermite_basis(x, y, dy, probe)
    assert float(np.max(np.abs(a - b))) < 1e-9 * max(float(np.max(np.abs(b))), 1.0)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("lo,hi", SPANS)
def test_hermite_matches_both_the_value_and_the_derivative(n, lo, hi):
    """**The defining conditions**, and the reason the degree is ``2n - 1`` rather than ``n - 1``."""
    x = np.linspace(lo, hi, n)
    y, dy = np.exp(x), np.exp(x)
    c, z = hm.hermite_newton(x, y, dy)
    h = 1e-6
    vals = np.atleast_1d(ip.evaluate_newton(c, z, x))
    ders = (np.atleast_1d(ip.evaluate_newton(c, z, x + h))
            - np.atleast_1d(ip.evaluate_newton(c, z, x - h))) / (2.0 * h)
    scale = float(np.max(np.abs(y)))
    assert float(np.max(np.abs(vals - y))) < 1e-10 * scale
    assert float(np.max(np.abs(ders - dy))) < 1e-6 * scale


@pytest.mark.parametrize("n", SIZES)
def test_the_hermite_polynomial_has_degree_two_n_minus_one(n):
    c, z = hm.hermite_newton(np.linspace(0.0, 1.0, n), np.exp(np.linspace(0.0, 1.0, n)),
                             np.exp(np.linspace(0.0, 1.0, n)))
    assert c.size == 2 * n
    assert z.size == 2 * n


def test_hermite_rejects_mismatched_derivative_data():
    with pytest.raises(ValueError):
        hm.hermite_newton([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0])


def test_hermite_rejects_repeated_nodes():
    with pytest.raises(ValueError):
        hm.hermite_newton([0.0, 0.0], [1.0, 1.0], [1.0, 1.0])


def test_lagrange_beats_hermite_at_the_same_number_of_data_items():
    """**Counterintuitive, and measured.** Hermite is often introduced as the more accurate
    method. At the same total count of data items it is not:

    ==============  =====================  =======================  =======
    data items      Hermite (n nodes)      Lagrange (2n nodes)      ratio
    ==============  =====================  =======================  =======
    4               4.37e-3                9.24e-4                  4.73
    6               5.58e-6                2.65e-6                  2.10
    8               6.54e-9                4.81e-9                  1.36
    10              5.96e-12               5.86e-12                 1.02
    ==============  =====================  =======================  =======

    Both are degree ``2n - 1``, so this is not a degree difference: it is that spreading the
    conditions over twice as many *locations* samples the function better than doubling up at
    half as many. Hermite earns its place when derivative data is what you have, or when matching
    the derivative is itself the requirement, as in the splines of lesson 51.
    """
    probe = np.linspace(0.0, 1.0, 401)
    truth = np.exp(probe)
    for n in (2, 3, 4, 5):
        x = np.linspace(0.0, 1.0, n)
        c, z = hm.hermite_newton(x, np.exp(x), np.exp(x))
        eh = float(np.max(np.abs(np.atleast_1d(ip.evaluate_newton(c, z, probe)) - truth)))
        xl = np.linspace(0.0, 1.0, 2 * n)
        el = float(np.max(np.abs(ip.evaluate_barycentric(xl, np.exp(xl), probe) - truth)))
        assert eh >= el


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_the_hermite_error_bound_uses_the_squared_node_polynomial(n):
    """That squaring is the whole difference from lesson 46's formula."""
    from nalib import interperror as ie

    x = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 1.0, 101)
    b = hm.hermite_error_bound(1.0, x, t)
    w = np.asarray(ie.node_polynomial(x, t))
    assert np.all(b >= 0.0)
    assert float(np.max(np.abs(b * math.factorial(2 * n) - w ** 2))) < 1e-10


# ---------------------------------------------------------------- osculating


@pytest.mark.parametrize("m", [0, 1, 2, 3, 5])
def test_one_node_repeated_gives_the_taylor_polynomial(m):
    """The osculating polynomial with a single node is Taylor's polynomial, which is a genuine
    special case and a good check that the general construction is right."""
    derivs = [np.ones(m + 1)]                       # every derivative of exp at 0 is 1
    c, z = hm.osculating([0.0], derivs)
    probe = np.linspace(-0.4, 0.4, 81)
    got = np.atleast_1d(ip.evaluate_newton(c, z, probe))
    taylor = sum(probe ** k / math.factorial(k) for k in range(m + 1))
    assert float(np.max(np.abs(got - taylor))) < 1e-10


@pytest.mark.parametrize("n", [2, 3, 5])
def test_every_multiplicity_one_gives_plain_lagrange_interpolation(n):
    """The other special case, at the opposite end."""
    x = np.linspace(0.0, 1.0, n)
    y = np.exp(x)
    c, z = hm.osculating(x, [[float(v)] for v in y])
    probe = np.linspace(0.0, 1.0, 61)
    got = np.atleast_1d(ip.evaluate_newton(c, z, probe))
    ref = ip.evaluate_barycentric(x, y, probe)
    assert float(np.max(np.abs(got - ref))) < 1e-9


def test_osculating_rejects_mismatched_input():
    with pytest.raises(ValueError):
        hm.osculating([0.0, 1.0], [[1.0]])


def test_osculating_rejects_an_empty_derivative_list():
    with pytest.raises(ValueError):
        hm.osculating([0.0, 1.0], [[1.0], []])


# ---------------------------------------------------------------- piecewise


@pytest.mark.parametrize("n", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("lo,hi", SPANS)
def test_piecewise_linear_reproduces_the_data(n, lo, hi, rng):
    x = np.linspace(lo, hi, n)
    y = rng.standard_normal(n)
    assert float(np.max(np.abs(hm.piecewise_linear(x, y, x) - y))) < 1e-14


@pytest.mark.parametrize("h", [0.5, 0.1, 0.01])
@pytest.mark.parametrize("m2", [0.5, 2.0, 25.0])
def test_the_piecewise_bound_is_h_squared_over_eight(h, m2):
    assert abs(hm.piecewise_linear_bound(m2, h) - m2 * h * h / 8.0) < 1e-15


def test_piecewise_linear_converges_on_runge_where_polynomials_diverged():
    """**The trade the rest of Part 7 is built on.** Measured on Runge's function:

    ========  ===========  ==============  ==========
    pieces    error        ``h^2/8`` bound  bound/err
    ========  ===========  ==============  ==========
    8         6.39e-2      3.91e-1         6.11
    32        2.07e-2      2.44e-2         1.18
    128       1.51e-3      1.53e-3         1.01
    ========  ===========  ==============  ==========

    The fitted order on the asymptotic tail is 1.889 against the theoretical 2, and the bound
    tightens to within 1 percent of the error. There is no divergence at any refinement, because
    the degree never grows. Fitting all six points instead gives 1.32, since the coarse grids
    have not resolved the peak yet, and that is why the tail is fitted.
    """
    out = hm.runge_by_pieces()
    assert out.fitted_order > 1.8
    assert out.fitted_order_all_points < out.fitted_order
    assert np.all(np.diff(out.errors) < 0.0)
    assert np.all(out.errors <= out.bounds * (1.0 + 1e-9))
    assert out.bounds[-1] / out.errors[-1] < 1.1


@pytest.mark.parametrize("a", [1.0, 25.0, 100.0])
def test_piecewise_linear_converges_for_every_pole_position(a):
    """Unlike polynomial interpolation, this does not care how close the poles are."""
    out = hm.runge_by_pieces(n_pieces=[16, 32, 64, 128, 256], a=a)
    assert np.all(np.diff(out.errors) < 0.0)
    assert out.fitted_order > 1.7


# ---------------------------------------------------------------- Weierstrass


def test_weierstrass_approximation_converges_where_interpolation_does_not():
    """**The precise content of the distinction.** Weierstrass promises that *some* sequence of
    polynomials converges uniformly to any continuous function. It promises nothing about the
    interpolants at prescribed nodes, and on Runge's function the two part company: the near best
    approximation converges while the equally spaced interpolant diverges, on the same function
    at the same degrees.
    """
    from nalib import interperror as ie

    out = hm.weierstrass_gap(lambda t: ie.runge(t), degrees=[4, 8, 12, 16, 20, 24])
    assert out["approximation_converges"]
    assert not out["interpolation_converges"]
    assert out["near_best_approximation"][-1] < 1e-2
    assert out["equally_spaced_interpolation"][-1] > 1.0


def test_both_converge_on_a_function_without_nearby_poles():
    """The distinction is about Runge's function, not about interpolation in general."""
    out = hm.weierstrass_gap(np.exp, degrees=[4, 6, 8, 10, 12])
    assert out["approximation_converges"]
    assert out["interpolation_converges"]