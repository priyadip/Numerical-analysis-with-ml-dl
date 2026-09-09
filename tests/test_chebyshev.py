"""Tests for `nalib.chebyshev`.

The minimax theorem is the reason this module exists, so it is checked against the exact value
``2^(1-n)`` and against both alternatives it claims to beat: equally spaced nodes and the best of
several hundred random node sets. A theorem that says "smallest" should beat everything offered.

The Lebesgue constant growth is fitted to the form the theory predicts for each family, since
fitting the wrong form to either would hide the difference that is the whole point.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import chebyshev as cb


DEGREES = [1, 2, 3, 5, 8, 13, 20]
SPANS = [(-1.0, 1.0), (0.0, 1.0), (-3.0, 7.0), (100.0, 101.0)]


# ---------------------------------------------------------------- the polynomials


@pytest.mark.parametrize("k", [0, 1, 2, 3, 5, 9, 16])
def test_the_recurrence_and_the_cosine_identity_agree(k):
    """``T_k(cos θ) = cos(kθ)`` and ``T_{k+1} = 2 t T_k - T_{k-1}`` are independent routes."""
    out = cb.recurrence_vs_cosine(k)
    assert out["relative_gap"] < 1e-12


@pytest.mark.parametrize("k", [1, 2, 3, 5, 9, 16])
def test_chebyshev_polynomials_are_bounded_by_one_on_the_interval(k):
    t = np.linspace(-1.0, 1.0, 2001)
    assert float(np.max(np.abs(cb.recurrence(k, t)))) <= 1.0 + 1e-12


@pytest.mark.parametrize("k", [1, 2, 3, 5, 9])
def test_the_extrema_are_plus_and_minus_one_and_there_are_k_plus_one_of_them(k):
    """``T_k`` equioscillates, which is what makes it the minimax polynomial."""
    x = cb.extrema_nodes(k + 1)
    v = cb.recurrence(k, x)
    assert np.max(np.abs(np.abs(v) - 1.0)) < 1e-12
    assert np.all(np.abs(np.diff(np.sign(v))) > 1.0)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 9])
def test_the_nodes_are_the_roots(k):
    assert float(np.max(np.abs(cb.recurrence(k, cb.nodes(k))))) < 1e-12


@pytest.mark.parametrize("k", [0, 1, 2, 3, 6])
def test_the_power_coefficients_reproduce_the_polynomial(k):
    """Included to be inspected, not to be evaluated with. The check is that they are right."""
    t = np.linspace(-1.0, 1.0, 501)
    a = cb.coefficients(k)
    got = np.polyval(a[::-1], t)
    assert float(np.max(np.abs(got - cb.recurrence(k, t)))) < 1e-10


@pytest.mark.parametrize("k", [-1, -5])
def test_a_negative_degree_is_rejected(k):
    with pytest.raises(ValueError):
        cb.recurrence(k, 0.0)
    with pytest.raises(ValueError):
        cb.coefficients(k)


# ---------------------------------------------------------------- the nodes


@pytest.mark.parametrize("n", DEGREES)
@pytest.mark.parametrize("lo,hi", SPANS)
def test_the_nodes_land_inside_the_requested_interval_and_are_sorted(n, lo, hi):
    x = cb.nodes(n, lo, hi)
    assert x.size == n
    assert np.all(np.diff(x) > 0) if n > 1 else True
    assert lo - 1e-12 <= float(x[0]) and float(x[-1]) <= hi + 1e-12


@pytest.mark.parametrize("n", [2, 3, 5, 8, 13])
@pytest.mark.parametrize("lo,hi", SPANS)
def test_the_lobatto_points_include_both_endpoints(n, lo, hi):
    """Which is why a practical code uses them: no extrapolation at the ends."""
    x = cb.extrema_nodes(n, lo, hi)
    assert abs(float(x[0]) - lo) < 1e-12
    assert abs(float(x[-1]) - hi) < 1e-12


@pytest.mark.parametrize("lo,hi", SPANS)
def test_the_change_of_interval_round_trips(lo, hi):
    t = np.linspace(-1.0, 1.0, 101)
    back = cb.inverse_change_of_interval(cb.change_of_interval(t, lo, hi), lo, hi)
    assert float(np.max(np.abs(back - t))) < 1e-12


def test_a_degenerate_interval_is_rejected():
    with pytest.raises(ValueError):
        cb.inverse_change_of_interval(0.5, 1.0, 1.0)


@pytest.mark.parametrize("n", [0, -3])
def test_a_degenerate_node_count_is_rejected(n):
    with pytest.raises(ValueError):
        cb.nodes(n)


def test_lobatto_needs_at_least_two_points():
    with pytest.raises(ValueError):
        cb.extrema_nodes(1)


# ---------------------------------------------------------------- the theorem


@pytest.mark.parametrize("n", [2, 4, 6, 8, 10, 14])
def test_the_chebyshev_node_polynomial_attains_the_theoretical_minimum(n):
    """The monic polynomial of degree ``n`` with the smallest maximum on ``[-1, 1]`` is
    ``T_n / 2^(n-1)``, whose maximum is exactly ``2^(1-n)``."""
    out = cb.minimax_property(n, n_random=1, rng=np.random.default_rng(0))
    assert out["chebyshev_matches_theory"] < 1e-12


@pytest.mark.parametrize("n", [4, 6, 8, 10, 14])
def test_chebyshev_beats_equally_spaced_and_the_best_of_many_random_node_sets(n):
    """A theorem that says "smallest" has to beat everything offered, not just the obvious
    alternative. Measured, ``equal/cheb`` grows from 1.58 at ``n = 4`` to 22.71 at ``n = 14``."""
    out = cb.minimax_property(n, n_random=300, rng=np.random.default_rng(2))
    assert out["beats_equally_spaced_by"] > 1.0
    assert out["beats_random_by"] > 1.0
    assert out["chebyshev"] <= out["best_of_random"] * (1.0 + 1e-12)


def test_the_advantage_over_equally_spaced_grows_with_the_degree():
    ratios = [cb.minimax_property(n, n_random=1,
                                  rng=np.random.default_rng(0))["beats_equally_spaced_by"]
              for n in (4, 8, 12, 16)]
    assert ratios == sorted(ratios)


# ---------------------------------------------------------------- Lebesgue constants


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_the_lebesgue_function_is_one_at_every_node(n):
    """At a node, one basis polynomial is 1 and the rest are 0, so the sum of absolute values
    is exactly 1. The constant is the maximum between the nodes, not at them."""
    x = cb.nodes(n)
    assert float(np.max(np.abs(cb.lebesgue_function(x, x) - 1.0))) < 1e-10


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_the_lebesgue_constant_is_at_least_one(n):
    assert cb.lebesgue_constant(cb.nodes(n)) >= 1.0 - 1e-12


def test_the_two_lebesgue_growth_rates_are_the_whole_argument():
    """**Lesson 46 in one table.** Measured on ``[-1, 1]``:

    ====  ================  ============  ==========
    n     equally spaced    Chebyshev     ratio
    ====  ================  ============  ==========
    8     6.93e+0           1.867         3.71e+0
    12    5.12e+1           2.124         2.41e+1
    16    5.12e+2           2.306         2.22e+2
    20    5.89e+3           2.448         2.41e+3
    ====  ================  ============  ==========

    Equally spaced fits ``C b^n`` with ``b = 1.676``, so it is exponential. Chebyshev fits
    ``C + s log n`` with ``s = 0.6326``, against the theoretical ``2/π = 0.6366``, a 0.6 percent
    agreement. Since ``||f - p|| <= (1 + Λ_n) ||f - best||``, that is the entire difference
    between a method that works at high degree and one that does not.
    """
    g = cb.lebesgue_growth([4, 6, 8, 10, 12, 14, 16, 20])
    assert g.equal_fitted_base > 1.5
    assert abs(g.chebyshev_fitted_log_slope - 2.0 / np.pi) < 0.05
    assert g.equally_spaced[-1] / g.chebyshev[-1] > 1000.0
    assert g.chebyshev[-1] < 3.0


# ---------------------------------------------------------------- barycentric weights


@pytest.mark.parametrize("n", [2, 3, 5, 8, 13, 20])
@pytest.mark.parametrize("lobatto", [False, True])
def test_the_closed_form_weights_match_the_computed_ones(n, lobatto):
    """The second reason to use these nodes: the ``O(n^2)`` weight computation becomes an
    ``O(n)`` formula. Barycentric weights are defined up to a common factor, so the ratio is
    what must be constant."""
    from nalib import interp as ip

    x = cb.extrema_nodes(n, ) if lobatto else cb.nodes(n)
    ratio = ip.barycentric_weights(x) / cb.barycentric_weights(n, lobatto=lobatto)
    assert float(np.max(np.abs(ratio - ratio[0]))) < 1e-10 * abs(float(ratio[0]))


@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("lobatto", [False, True])
def test_the_closed_form_weights_stay_bounded(n, lobatto):
    """General barycentric weights can overflow. These are bounded by 1 at every size."""
    w = cb.barycentric_weights(n, lobatto=lobatto)
    assert float(np.max(np.abs(w))) <= 1.0 + 1e-12


# ---------------------------------------------------------------- interpolating with them


@pytest.mark.parametrize("n", [16, 24, 32, 48])
@pytest.mark.parametrize("lobatto", [False, True])
def test_chebyshev_interpolation_of_a_smooth_function_reaches_machine_precision(n, lobatto):
    """From 16 nodes upward. Below that the limit is approximation, not arithmetic: 8 nodes give
    4.0e-7 on ``exp``, which is what a degree 7 polynomial can do and not a defect."""
    p, x, y = cb.interpolate(np.exp, n, lo=-1.0, hi=1.0, lobatto=lobatto)
    probe = np.linspace(-1.0, 1.0, 1001)
    assert float(np.max(np.abs(p(probe) - np.exp(probe)))) < 1e-13


@pytest.mark.parametrize("n", [4, 6, 8, 10, 12])
def test_the_error_falls_geometrically_before_it_reaches_rounding(n):
    """Below machine precision the convergence is geometric in the degree, which is what
    "spectral accuracy" means and what equally spaced nodes never achieve."""
    probe = np.linspace(-1.0, 1.0, 1001)
    err = lambda k: float(np.max(np.abs(cb.interpolate(np.exp, k)[0](probe) - np.exp(probe))))
    assert err(n + 2) < 0.5 * err(n)


@pytest.mark.parametrize("lo,hi", SPANS)
def test_chebyshev_interpolation_works_on_any_interval(lo, hi):
    p, x, y = cb.interpolate(np.sin, 24, lo=lo, hi=hi)
    probe = np.linspace(lo, hi, 501)
    assert float(np.max(np.abs(p(probe) - np.sin(probe)))) < 1e-10


@pytest.mark.parametrize("n", [10, 20, 40])
def test_chebyshev_interpolation_reproduces_its_own_nodes(n):
    p, x, y = cb.interpolate(np.cos, n)
    assert float(np.max(np.abs(p(x) - y))) < 1e-13