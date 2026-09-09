"""Tests for `nalib.interperror`.

The error formula has three factors and they are tested separately, because the whole argument
of lessons 46 and 47 is that only one of them is under your control.

The Runge phenomenon is tested as a **divergence**, not as a large error. A large error at one
degree could be anything; an error that grows without bound as nodes are added is the thing that
makes equally spaced high degree interpolation unusable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nalib import interperror as ie


DEGREES = [2, 4, 6, 8, 12]
SPANS = [(-1.0, 1.0), (0.0, 1.0), (-2.0, 5.0)]


# ---------------------------------------------------------------- the node polynomial


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9])
@pytest.mark.parametrize("lo,hi", SPANS)
def test_the_node_polynomial_vanishes_exactly_at_every_node(n, lo, hi):
    """``w(x_i) = 0`` by construction, which is why the error formula has no term at a node."""
    x = np.linspace(lo, hi, n)
    assert np.all(ie.node_polynomial(x, x) == 0.0)


@pytest.mark.parametrize("n", [2, 4, 7, 11])
def test_the_node_polynomial_is_monic_of_the_right_degree(n, rng):
    """It is ``prod (t - x_i)``, so it is monic of degree ``n``. Checked by the ratio to ``t^n``
    far from the nodes, where the leading term dominates."""
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    t = 1e6
    assert abs(ie.node_polynomial(x, t) / t ** n - 1.0) < 1e-4


@pytest.mark.parametrize("n", [4, 6, 8, 12, 16])
def test_equally_spaced_nodes_put_the_worst_of_the_node_polynomial_at_the_ends(n):
    """**The mechanism of the Runge phenomenon.** The one factor you control is largest exactly
    where you can least afford it, and the imbalance grows with the degree."""
    out = ie.node_polynomial_extremes(np.linspace(-1.0, 1.0, n))
    assert out["distance_from_nearest_end"] < 0.5
    assert out["end_to_middle_ratio"] > 1.0


def test_the_end_to_middle_imbalance_grows_with_the_degree():
    ratios = [ie.node_polynomial_extremes(np.linspace(-1.0, 1.0, n))["end_to_middle_ratio"]
              for n in (4, 8, 16, 24)]
    assert ratios[-1] > ratios[0]
    assert ratios == sorted(ratios)


# ---------------------------------------------------------------- the error formula


@pytest.mark.parametrize("n", DEGREES)
@pytest.mark.parametrize("cheb", [False, True])
def test_the_error_formula_bounds_the_actual_error(n, cheb):
    """The bound must hold. It is a worst case over ``xi``, so it will overstate, and by how
    much is recorded rather than asserted away."""
    out = ie.error_formula_report(np.exp, lambda k, t: np.exp(t), n,
                                  lo=-1.0, hi=1.0, chebyshev=cheb)
    assert out.measured_error <= out.predicted_bound * (1.0 + 1e-9)
    assert out.bound_overstates >= 1.0


@pytest.mark.parametrize("n", [3, 5, 7])
def test_a_polynomial_of_low_enough_degree_has_zero_error_and_a_zero_bound(n):
    """The ``(n+1)``-th derivative of a degree ``n-1`` polynomial is identically zero, so both
    the error and its bound must vanish. A bound that did not would be wrong, not merely loose."""
    cubic = lambda t: 2.0 - t + 0.5 * t ** 2
    dfn = lambda k, t: (1.0 if k == 0 else (-1.0 if k == 1 else (1.0 if k == 2 else 0.0)))
    out = ie.error_formula_report(cubic, lambda k, t: 0.0, n, lo=-1.0, hi=1.0)
    assert out.derivative_bound == 0.0
    assert out.predicted_bound == 0.0
    assert out.measured_error < 1e-10


def test_the_error_report_rejects_a_degenerate_size():
    with pytest.raises(ValueError):
        ie.error_formula_report(np.exp, lambda k, t: np.exp(t), 0)


# ---------------------------------------------------------------- Runge


def test_equally_spaced_interpolation_of_runge_diverges():
    """**The phenomenon, as a divergence.** Measured maximum error on ``[-1, 1]``:

    ====  ================  ============
    n     equally spaced    Chebyshev
    ====  ================  ============
    8     2.47e-1           3.92e-1
    12    5.57e-1           1.83e-1
    20    8.58e+0           3.76e-2
    30    3.34e+2           5.16e-3
    ====  ================  ============

    The equally spaced error bottoms out near ``n = 8`` and then grows like ``1.31^n``, reaching
    334 at 30 nodes on a function bounded by 1. The Chebyshev error falls geometrically the whole
    way. Both node sets interpolate the same smooth function.
    """
    out = ie.runge_divergence([8, 12, 20, 30])
    assert out["diverges"]
    assert out["log_growth_per_node"] > 0.1
    assert out["errors"][-1] > 100.0


def test_chebyshev_nodes_make_runge_converge():
    out = ie.runge_divergence([8, 12, 20, 30], chebyshev=True)
    assert not out["diverges"]
    assert out["log_growth_per_node"] < 0.0
    assert out["errors"][-1] < 1e-2
    assert np.all(np.diff(out["errors"]) < 0.0)


@pytest.mark.parametrize("n", [16, 20, 24, 30])
def test_the_runge_error_concentrates_at_the_ends(n):
    """It is entirely an end effect. The middle of the interval is interpolated well at every
    degree, and quoting one maximum hides that."""
    out = ie.where_the_error_lives(n)
    assert out["max_outer_quarters"] > out["max_middle_half"]
    assert out["ratio"] > 10.0


@pytest.mark.parametrize("n", [12, 20, 28])
def test_the_location_of_the_worst_error_approaches_the_endpoints(n):
    out = ie.runge_divergence([n])
    assert abs(float(out["argmax"][0])) > 0.9


@pytest.mark.parametrize("a", [1.0, 5.0, 25.0, 100.0])
def test_moving_the_poles_closer_makes_the_divergence_faster(a):
    """The poles of ``1/(1 + a t^2)`` sit at ``+- i/sqrt(a)``. Raising ``a`` moves them toward
    the real interval, which is where the difficulty actually lives."""
    out = ie.runge_divergence([8, 14, 20, 26], a=a)
    if a >= 25.0:
        assert out["diverges"]
    assert np.isfinite(out["log_growth_per_node"])


def test_runge_is_bounded_and_smooth_on_the_interval():
    """The function itself is harmless, which is the point: nothing on the interval predicts the
    failure, and the explanation is in the complex plane."""
    t = np.linspace(-1.0, 1.0, 1001)
    v = ie.runge(t)
    assert np.all(np.isfinite(v))
    assert 0.0 < float(np.min(v)) and float(np.max(v)) <= 1.0