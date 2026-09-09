"""Tests for nalib.chebapprox.

Chebyshev approximation has a lot of exactly known answers, so most of these check against a
closed form rather than against a tolerance: the coefficients of ``T_n`` itself, the power
expansions of the first few members, and the identity that Clenshaw and a direct basis sum agree.

Every test sweeps degrees, intervals and functions.
"""
import math

import numpy as np
import pytest

from nalib import chebapprox as ca

INTERVALS = [(-1.0, 1.0), (0.0, 1.0), (-2.0, 3.0), (1.5, 1.75)]
FUNCTIONS = [("exp", np.exp),
             ("sin3", lambda t: np.sin(3.0 * t)),
             ("runge", lambda t: 1.0 / (1.0 + 25.0 * t * t)),
             ("cosh", np.cosh),
             ("smoothed-abs", lambda t: np.sqrt(t * t + 0.01))]


def taylor_exp(k):
    return np.asarray([1.0 / math.factorial(i) for i in range(k)])


# --------------------------------------------------------------------------- Clenshaw


@pytest.mark.parametrize("n", [0, 1, 2, 5, 9, 20])
@pytest.mark.parametrize("lo,hi", INTERVALS)
def test_clenshaw_agrees_with_a_direct_basis_sum(n, lo, hi):
    rng = np.random.default_rng(n + 1)
    a = rng.standard_normal(n + 1)
    x = np.linspace(lo, hi, 97)
    u = np.clip((2.0 * x - (hi + lo)) / (hi - lo), -1.0, 1.0)
    theta = np.arccos(u)
    direct = sum(a[k] * np.cos(k * theta) for k in range(n + 1))
    got = ca.evaluate_series(a, x, lo, hi)
    assert np.max(np.abs(got - direct)) < 1e-12 * max(float(np.max(np.abs(a))), 1.0)


@pytest.mark.parametrize("n", [0, 1, 4, 8])
def test_a_single_coefficient_gives_that_chebyshev_polynomial(n):
    a = np.zeros(n + 1)
    a[n] = 1.0
    x = np.linspace(-1.0, 1.0, 101)
    assert np.max(np.abs(ca.evaluate_series(a, x) - np.cos(n * np.arccos(x)))) < 1e-12


def test_an_empty_series_is_rejected():
    with pytest.raises(ValueError, match="at least one coefficient"):
        ca.evaluate_series(np.zeros(0), np.array([0.5]))


@pytest.mark.parametrize("lo,hi", [(1.0, 1.0), (2.0, -1.0)])
def test_a_degenerate_interval_is_rejected(lo, hi):
    with pytest.raises(ValueError, match="need lo < hi"):
        ca.evaluate_series(np.array([1.0]), np.array([0.5]), lo, hi)


# --------------------------------------------------------------------------- power basis


@pytest.mark.parametrize("n,expected", [
    (0, [1.0]),
    (1, [0.0, 1.0]),
    (2, [-1.0, 0.0, 2.0]),
    (3, [0.0, -3.0, 0.0, 4.0]),
    (4, [1.0, 0.0, -8.0, 0.0, 8.0]),
    (5, [0.0, 5.0, 0.0, -20.0, 0.0, 16.0]),
])
def test_the_power_expansion_of_each_chebyshev_polynomial_is_the_textbook_one(n, expected):
    a = np.zeros(n + 1)
    a[n] = 1.0
    assert np.max(np.abs(ca.to_power_basis(a) - np.asarray(expected))) < 1e-12


@pytest.mark.parametrize("n", [1, 3, 6, 10])
def test_the_leading_power_coefficient_is_two_to_the_n_minus_one(n):
    a = np.zeros(n + 1)
    a[n] = 1.0
    assert ca.to_power_basis(a)[n] == pytest.approx(2.0 ** (n - 1), rel=1e-12)


@pytest.mark.parametrize("n", [2, 5, 10, 20, 30])
def test_the_conversion_round_trip_is_accurate_while_the_coefficients_decay(n):
    out = ca.conversion_cost(np.exp, n)
    assert out["relative_disagreement"] < 1e-10


@pytest.mark.parametrize("n", [40, 50, 60])
def test_the_conversion_fails_when_nothing_decays(n):
    a = np.zeros(n + 1)
    a[n] = 1.0
    p = ca.to_power_basis(a)
    x = np.linspace(-1.0, 1.0, 501)
    direct = ca.evaluate_series(a, x)
    power = sum(p[k] * x ** k for k in range(p.size))
    assert float(np.max(np.abs(direct - power))) > 1e-4
    assert float(np.max(np.abs(p))) > 1e12


# --------------------------------------------------------------------------- coefficients


@pytest.mark.parametrize("n", [12, 15, 20])
def test_the_three_coefficient_routes_agree_once_the_degree_is_high_enough(n):
    q = ca.coefficients_by_quadrature(np.exp, n, n_quad=200001)
    i = ca.coefficients_by_interpolation(np.exp, n)
    r = ca.coefficients_by_interpolation(np.exp, n, kind="roots")
    assert np.max(np.abs(q - i)) < 1e-9
    assert np.max(np.abs(q - r)) < 1e-9


def test_the_routes_disagree_at_low_degree_by_exactly_the_aliasing():
    """They are different objects, and the difference shrinks as the grid resolves more.

    At degree 2 the Lobatto interpolation coefficients differ from the true series by 4.5e-2,
    at degree 5 by 4.5e-5 and at degree 9 by 5.5e-10. The Chebyshev roots grid aliases less at
    every degree, because it has no repeated endpoint.
    """
    gaps_extrema, gaps_roots = [], []
    for n in (2, 3, 5, 9):
        q = ca.coefficients_by_quadrature(np.exp, n, n_quad=200001)
        gaps_extrema.append(float(np.max(np.abs(
            q - ca.coefficients_by_interpolation(np.exp, n)))))
        gaps_roots.append(float(np.max(np.abs(
            q - ca.coefficients_by_interpolation(np.exp, n, kind="roots")))))
    assert all(b < a for a, b in zip(gaps_extrema, gaps_extrema[1:]))
    assert all(b < a for a, b in zip(gaps_roots, gaps_roots[1:]))
    assert all(r < e for r, e in zip(gaps_roots, gaps_extrema))


def test_the_leading_coefficients_of_exp_match_the_known_values():
    """a_k = I_k(1) times 2, the modified Bessel functions, tabulated to eight digits."""
    a = ca.coefficients_by_quadrature(np.exp, 4, n_quad=200001)
    exact = [1.26606588, 1.13031821, 0.27149534, 0.04433685, 0.00547424]
    for k, v in enumerate(exact):
        assert a[k] == pytest.approx(v, abs=1e-7)


@pytest.mark.parametrize("kind", ["extrema", "roots"])
@pytest.mark.parametrize("n", [0, 1, 3, 7])
@pytest.mark.parametrize("lo,hi", INTERVALS)
def test_the_interpolant_reproduces_the_data_at_its_own_nodes(kind, n, lo, hi):
    f = lambda t: np.exp(0.5 * t)
    a = ca.coefficients_by_interpolation(f, n, lo, hi, kind=kind)
    if kind == "extrema":
        theta = np.arange(n + 1) * math.pi / n if n else np.array([0.0])
    else:
        theta = (2.0 * np.arange(n + 1) + 1.0) * math.pi / (2.0 * (n + 1))
    x = 0.5 * (hi + lo) + 0.5 * (hi - lo) * np.cos(theta)
    assert np.max(np.abs(ca.evaluate_series(a, x, lo, hi) - f(x))) < 1e-10


@pytest.mark.parametrize("kind", ["extrema", "roots"])
@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_a_polynomial_is_reproduced_at_or_above_its_degree(kind, n):
    rng = np.random.default_rng(n + 11)
    c = rng.standard_normal(n + 1)
    poly = lambda t: sum(c[k] * np.asarray(t) ** k for k in range(n + 1))
    a = ca.coefficients_by_interpolation(poly, n, -1.0, 1.0, kind=kind)
    x = np.linspace(-1.0, 1.0, 301)
    assert np.max(np.abs(ca.evaluate_series(a, x) - poly(x))) < 1e-9 * max(
        float(np.max(np.abs(c))), 1.0)


@pytest.mark.parametrize("bad", ["gauss", "nodes", ""])
def test_an_unknown_node_kind_is_rejected(bad):
    with pytest.raises(ValueError, match="kind must be"):
        ca.coefficients_by_interpolation(np.exp, 4, kind=bad)


@pytest.mark.parametrize("n", [-1, -4])
def test_a_negative_degree_is_rejected(n):
    with pytest.raises(ValueError, match="degree must be non-negative"):
        ca.coefficients_by_interpolation(np.exp, n)
    with pytest.raises(ValueError, match="degree must be non-negative"):
        ca.coefficients_by_quadrature(np.exp, n)


@pytest.mark.parametrize("n", [6, 8, 10])
def test_fitting_scattered_samples_recovers_the_interpolation_coefficients(n):
    rng = np.random.default_rng(n)
    x = np.sort(rng.uniform(-1.0, 1.0, 200))
    f = lambda t: np.exp(t)
    out = ca.coefficients_by_fit(x, f(x), n, -1.0, 1.0)
    reference = ca.coefficients_by_interpolation(f, n)
    assert np.max(np.abs(out["coefficients"] - reference)) < 1e-5


def test_the_scattered_fit_approaches_the_interpolation_coefficients_with_degree():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(-1.0, 1.0, 200))
    f = lambda t: np.exp(t)
    gaps = [float(np.max(np.abs(ca.coefficients_by_fit(x, f(x), n, -1.0, 1.0)["coefficients"]
                                - ca.coefficients_by_interpolation(f, n))))
            for n in (2, 4, 6, 8)]
    assert all(b < a for a, b in zip(gaps, gaps[1:]))


def test_fitting_rejects_mismatched_or_insufficient_samples():
    with pytest.raises(ValueError, match="nodes against"):
        ca.coefficients_by_fit(np.linspace(0, 1, 5), np.zeros(4), 2)
    with pytest.raises(ValueError, match="at least"):
        ca.coefficients_by_fit(np.linspace(0, 1, 3), np.zeros(3), 5)


# --------------------------------------------------------------------------- decay


def test_an_entire_function_decays_faster_than_any_geometric_rate():
    """exp has |a_k| ~ 1/(2^k k!), so the local ratio keeps growing and no fixed rho fits."""
    out = ca.coefficient_decay(np.exp, 40)
    assert out["accelerating"]
    assert out["classification"].startswith("faster than geometric")
    assert not out["looks_geometric"]


@pytest.mark.parametrize("a,rho", [(1.0, (1.0 + math.sqrt(2.0)) / 1.0),
                                   (2.0, (1.0 + math.sqrt(5.0)) / 2.0),
                                   (5.0, (1.0 + math.sqrt(26.0)) / 5.0)])
def test_the_fitted_rate_recovers_the_bernstein_ellipse_parameter(a, rho):
    """Poles at +/- i/a give rho = (1 + sqrt(1 + a^2)) / a, which the fit should recover."""
    f = lambda t: 1.0 / (1.0 + a * a * t * t)
    out = ca.coefficient_decay(f, 60)
    assert out["looks_geometric"]
    assert abs(out["geometric_rate"] - rho) / rho < 0.05


def test_a_function_with_a_nearby_singularity_decays_more_slowly():
    far = ca.coefficient_decay(lambda t: 1.0 / (1.0 + t * t), 60)["geometric_rate"]
    near = ca.coefficient_decay(lambda t: 1.0 / (1.0 + 25.0 * t * t), 60)["geometric_rate"]
    assert near < far


def test_a_non_smooth_function_is_classified_algebraic_with_a_rate_tending_to_one():
    """The tell for algebraic decay is that the fitted geometric rate collapses toward 1."""
    rates = [ca.coefficient_decay(np.abs, n)["geometric_rate"] for n in (40, 80, 160, 300)]
    assert all(b < a for a, b in zip(rates, rates[1:]))
    assert rates[-1] < 1.05
    assert ca.coefficient_decay(np.abs, 300)["classification"].startswith("algebraic")


def test_the_classification_needs_enough_coefficients_to_settle():
    """sqrt(t^2 + 0.01) is analytic with poles at +/- 0.1i, so it IS geometric, and it is
    misclassified at low degree because the asymptotic regime has not started."""
    f = lambda t: np.sqrt(t * t + 0.01)
    assert not ca.coefficient_decay(f, 20)["looks_geometric"]
    assert ca.coefficient_decay(f, 160)["looks_geometric"]
    rho = (1.0 + math.sqrt(101.0)) / 10.0
    assert ca.coefficient_decay(f, 300)["geometric_rate"] < 1.2
    assert ca.coefficient_decay(f, 300)["geometric_rate"] > rho


def test_the_decay_fit_skips_coefficients_at_the_roundoff_floor():
    """An even function has every odd coefficient exactly zero. Including them halves the
    apparent rate: the Runge function then fits 0.985 instead of its true 1.22."""
    out = ca.coefficient_decay(lambda t: 1.0 / (1.0 + 25.0 * t * t), 60)
    assert out["kept"] < 60
    assert out["kept"] > 10


@pytest.mark.parametrize("name,f", FUNCTIONS)
def test_the_coefficient_magnitudes_broadly_decrease(name, f):
    # Compared in blocks, because an even or odd function has exactly zero coefficients of the
    # opposite parity and a single entry can be roundoff rather than a measurement.
    mag = ca.coefficient_decay(f, 24)["magnitudes"]
    assert float(np.max(mag[-4:])) < float(np.max(mag[:4]))


# --------------------------------------------------------------------------- truncation


@pytest.mark.parametrize("name,f", FUNCTIONS)
@pytest.mark.parametrize("n", [2, 4, 8, 12])
def test_the_tail_bound_holds_for_the_truncated_series(name, f, n):
    out = ca.truncation_error(f, n)
    assert out["bound_holds"]


@pytest.mark.parametrize("name,f", FUNCTIONS)
@pytest.mark.parametrize("n", [2, 4, 8, 12])
def test_the_interpolant_needs_twice_the_bound(name, f, n):
    out = ca.truncation_error(f, n)
    assert out["bound_holds_for_the_interpolant"]


@pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
def test_the_tail_bound_is_tight_on_a_geometrically_decaying_series(n):
    out = ca.truncation_error(np.exp, n)
    assert 0.4 < out["bound_tightness"] <= 1.0 + 1e-6


@pytest.mark.parametrize("name,f", FUNCTIONS)
@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_the_chebyshev_truncation_is_near_minimax(name, f, n):
    out = ca.near_minimax_factor(f, n)
    assert out["factor"] >= 1.0 - 1e-9
    assert out["factor"] < 8.0


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_the_measured_factor_stays_below_the_classical_bound(n):
    out = ca.near_minimax_factor(np.exp, n)
    assert out["factor"] < out["classical_bound"]


@pytest.mark.parametrize("n", [2, 4, 6, 8])
@pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (-3.0, 3.0)])
def test_chebyshev_beats_taylor_of_the_same_degree_on_an_interval(n, lo, hi):
    out = ca.against_taylor(np.exp, taylor_exp(n + 1), n, lo, hi)
    assert out["chebyshev_error"] < out["taylor_error"]
    assert out["ratio"] > 1.0


def test_the_taylor_advantage_is_set_by_the_degree_not_by_the_width():
    """At a fixed degree the ratio barely moves with the interval: 34.5, 35.6, 36.7, 35.1 for
    half-widths 0.5, 1, 2 and 4. Both errors scale together, so the ratio does not."""
    ratios = [ca.against_taylor(np.exp, taylor_exp(9), 6, -w, w)["ratio"]
              for w in (0.5, 1.0, 2.0, 4.0)]
    assert max(ratios) / min(ratios) < 1.2


def test_the_taylor_advantage_grows_geometrically_with_the_degree():
    """Roughly a factor of 4 per two degrees, which is the 2^(n-1) of the minimax property.

    Stopped at degree 12: past that both errors are at machine precision and the ratio is
    measuring roundoff.
    """
    tc = taylor_exp(25)
    ratios = [ca.against_taylor(np.exp, tc, n, -1.0, 1.0)["ratio"]
              for n in (2, 4, 6, 8, 10, 12)]
    assert all(b > a for a, b in zip(ratios, ratios[1:]))
    steps = [b / a for a, b in zip(ratios, ratios[1:])]
    assert all(3.0 < s < 5.0 for s in steps[1:])


def test_against_taylor_needs_enough_coefficients():
    with pytest.raises(ValueError, match="Taylor coefficients"):
        ca.against_taylor(np.exp, taylor_exp(3), 6)


# --------------------------------------------------------------------------- aliasing


@pytest.mark.parametrize("n", [4, 8, 12])
def test_the_interpolation_coefficients_differ_from_the_true_ones(n):
    out = ca.aliasing_report(np.exp, n)
    assert out["relative_difference"] > 0.0


@pytest.mark.parametrize("n", [4, 6, 8])
def test_the_aliasing_difference_shrinks_as_the_degree_grows(n):
    a = ca.aliasing_report(np.exp, n)["relative_difference"]
    b = ca.aliasing_report(np.exp, n + 4)["relative_difference"]
    assert b < a


# --------------------------------------------------------------------------- economization


@pytest.mark.parametrize("start", [4, 6, 10])
@pytest.mark.parametrize("drop", [0, 1, 2])
def test_economization_lowers_the_degree_by_exactly_the_amount_asked(start, drop):
    out = ca.economize(taylor_exp(start + 1), drop)
    assert out["degree"] == start - drop


def test_dropping_nothing_leaves_the_series_alone():
    c = taylor_exp(7)
    out = ca.economize(c, 0)
    assert np.max(np.abs(out["coefficients"] - c)) == 0.0
    assert out["guaranteed_error_added"] == 0.0


@pytest.mark.parametrize("drop", [-1, -5])
def test_dropping_a_negative_number_is_rejected(drop):
    with pytest.raises(ValueError, match="negative number of terms"):
        ca.economize(taylor_exp(6), drop)


@pytest.mark.parametrize("drop", [6, 9])
def test_dropping_more_terms_than_exist_is_rejected(drop):
    with pytest.raises(ValueError, match="cannot drop"):
        ca.economize(taylor_exp(6), drop)


@pytest.mark.parametrize("start,target", [(10, 5), (12, 6), (8, 4), (14, 7)])
def test_economization_beats_plain_truncation(start, target):
    out = ca.economization_report(np.exp, taylor_exp(start + 1), target)
    assert out["economized_error"] < out["truncated_taylor_error"]


@pytest.mark.parametrize("start,target", [(10, 5), (12, 6), (14, 7)])
def test_economization_recovers_most_of_the_gap_to_minimax(start, target):
    out = ca.economization_report(np.exp, taylor_exp(start + 1), target)
    assert out["fraction_of_the_gap_recovered"] > 0.9


@pytest.mark.parametrize("start,target", [(10, 5), (12, 6)])
def test_the_guaranteed_bound_on_the_added_error_holds(start, target):
    out = ca.economization_report(np.exp, taylor_exp(start + 1), target)
    assert out["economized_error"] <= out["full_taylor_error"] + out["guaranteed_error_added"] \
        + 1e-12


def test_economizing_to_a_degree_that_is_not_lower_is_rejected():
    with pytest.raises(ValueError, match="not below"):
        ca.economization_report(np.exp, taylor_exp(6), 5)
