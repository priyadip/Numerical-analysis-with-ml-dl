"""Tests for nalib.montecarlo.

Four groups. The rate group asserts the exponent and, more importantly, asserts that the exponent
does not move with the dimension. A test of the exponent alone would pass for any method; the
constancy across dimensions is the property that distinguishes Monte Carlo from every quadrature
rule in Part 9.

The honesty group checks that the reported error bar is one the estimator can back up, by measuring
the spread of the estimates independently and comparing. An estimator that reports a bar it cannot
support is worse than one that reports none.

The variance reduction group asserts the closed form ratios, ``1/(1 + rho)`` for antithetic and
``1/(1 - rho**2)`` for a control variate, against the measured ones. These are identities, so the
bars are tight, and the antithetic group deliberately includes the case where the technique **costs**
a factor of two.

The quasi-random group asserts the negative result on purpose: by sixteen dimensions Halton is worse
than a random sample on this integrand. A reader who takes "low discrepancy is better" as
unconditional has to get past a test that says where it stops.
"""
import functools
import math

import numpy as np
import pytest

from nalib import montecarlo as mc


@functools.lru_cache(maxsize=None)
def rate():
    return mc.the_error_falls_like_one_over_root_n()


@functools.lru_cache(maxsize=None)
def bar():
    return mc.the_error_bar_is_honest()


@functools.lru_cache(maxsize=None)
def crossover():
    return mc.where_monte_carlo_overtakes_a_grid()


@functools.lru_cache(maxsize=None)
def paired():
    return mc.antithetic_helps_only_a_monotone_integrand()


@functools.lru_cache(maxsize=None)
def controls():
    return mc.a_control_variate_achieves_one_minus_rho_squared()


@functools.lru_cache(maxsize=None)
def quasi():
    return mc.halton_beats_sampling_then_stops()


@functools.lru_cache(maxsize=None)
def spread():
    return mc.the_discrepancy_is_the_reason()


@functools.lru_cache(maxsize=None)
def strata():
    return mc.stratification_pays_where_the_integrand_is_smooth()


# ------------------------------------------------------------------ the problems


@pytest.mark.parametrize("dimension", [1, 2, 5, 9])
@pytest.mark.parametrize("key", ["smooth", "gaussian", "oscillating", "symmetric", "ball"])
def test_every_stated_value_is_the_integral(key, dimension):
    f, exact, _ = mc.problems(dimension)[key]
    out = mc.integrate(f, dimension, 200000, seed=7)
    assert abs(out["estimate"] - exact) < 8.0 * out["standard_error"] + 1e-12


def test_the_problem_rejects_a_bad_dimension():
    with pytest.raises(ValueError):
        mc.problems(0)


def test_the_estimator_rejects_bad_sizes():
    f = mc.problems(2)["smooth"][0]
    with pytest.raises(ValueError):
        mc.integrate(f, 0, 100)
    with pytest.raises(ValueError):
        mc.integrate(f, 2, 1)


# ------------------------------------------------------------------ the rate


def test_the_exponent_is_minus_one_half():
    out = rate()
    assert out["every_power_is_minus_a_half"]
    assert out["worst_deviation"] < 0.06


def test_the_exponent_does_not_move_with_the_dimension():
    # the whole reason Monte Carlo exists: no quadrature rule can say this
    assert rate()["spread_across_dimensions"] < 0.06


def test_the_rate_was_measured_across_a_real_range_of_dimensions():
    dimensions = [row["dimension"] for row in rate()["rows"]]
    assert max(dimensions) >= 20
    assert min(dimensions) == 1


def test_the_error_bar_matches_the_spread():
    out = bar()
    assert out["the_bar_is_honest"]
    assert abs(out["ratio"] - 1.0) < 0.1


def test_the_coverage_is_the_normal_one():
    out = bar()
    assert out["coverage_matches"]
    assert abs(out["fraction_within_one_bar"] - out["predicted_fraction"]) < 0.05


# ------------------------------------------------------------------ against a grid


def test_the_grid_never_loses_on_a_separable_integrand():
    # the textbook crossover argument assumes a generic integrand and this one is not
    out = crossover()
    assert out["the_grid_never_loses_on_the_separable_one"]
    assert out["separable_crossover"] is None


def test_monte_carlo_wins_early_on_a_discontinuous_one():
    out = crossover()
    assert out["monte_carlo_wins_early_on_the_rough_one"]
    assert out["rough_crossover"] <= 4


def test_the_grid_is_essentially_exact_in_one_dimension():
    row = next(r for r in crossover()["rows"]
               if r["dimension"] == 1 and r["integrand"] == crossover()["separable_name"])
    assert row["grid_error"] < 1e-12


# ------------------------------------------------------------------ variance reduction


def test_the_antithetic_ratio_is_one_over_one_plus_rho():
    out = paired()
    assert out["the_prediction_holds"]
    assert out["worst_relative_miss"] < 0.02


def test_a_linear_integrand_is_integrated_exactly():
    out = paired()
    assert out["linear_is_exactly_minus_one"]
    assert out["linear_error"] == 0.0


def test_antithetic_costs_a_factor_of_two_on_a_symmetric_integrand():
    # not a defect, and worth a test: pairing a symmetric integrand wastes half the evaluations
    out = paired()
    assert out["it_costs_a_factor_of_two_on_the_symmetric_one"]
    assert abs(out["symmetric_correlation"] - 1.0) < 1e-12


def test_antithetic_helps_a_monotone_integrand():
    assert paired()["it_helps_the_monotone_one"]


def test_antithetic_needs_an_even_count():
    f = mc.problems(2)["smooth"][0]
    with pytest.raises(ValueError):
        mc.antithetic(f, 2, 101)


def test_the_control_variate_ratio_is_one_over_one_minus_rho_squared():
    out = controls()
    assert out["the_prediction_holds"]
    assert out["worst_relative_miss"] < 0.05


def test_a_better_correlated_control_saves_more():
    rows = sorted(controls()["rows"], key=lambda r: abs(r["correlation"]))
    ratios = [r["measured_variance_ratio"] for r in rows]
    assert ratios == sorted(ratios)


def test_stratification_pays_more_on_a_smooth_integrand():
    out = strata()
    assert out["it_pays_more_on_the_smooth_one"]
    assert out["smooth_gain"] > 10.0 * 1.0


def test_stratification_rejects_impossible_sizes():
    f = mc.problems(2)["smooth"][0]
    with pytest.raises(ValueError):
        mc.stratified(f, 2, 0, 4)
    with pytest.raises(ValueError):
        mc.stratified(f, 12, 8, 2)


# ------------------------------------------------------------------ quasi-random


@pytest.mark.parametrize("base", [2, 3, 7])
@pytest.mark.parametrize("count", [1, 16, 500])
def test_van_der_corput_stays_in_the_unit_interval(base, count):
    values = mc.van_der_corput(count, base)
    assert values.shape == (count,)
    assert float(np.min(values)) >= 0.0
    assert float(np.max(values)) < 1.0


def test_van_der_corput_base_two_is_the_bit_reversal():
    # 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8 is the definition, so it is checkable by hand
    expected = np.array([0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875])
    assert float(np.max(np.abs(mc.van_der_corput(7, 2) - expected))) == 0.0


def test_the_base_must_be_at_least_two():
    with pytest.raises(ValueError):
        mc.van_der_corput(10, 1)


@pytest.mark.parametrize("dimension", [1, 3, 12])
def test_halton_has_the_right_shape_and_range(dimension):
    points = mc.halton(200, dimension)
    assert points.shape == (200, dimension)
    assert float(np.min(points)) >= 0.0
    assert float(np.max(points)) < 1.0


def test_halton_runs_out_of_tabulated_bases():
    with pytest.raises(ValueError):
        mc.halton(10, len(mc.PRIMES) + 1)


def test_the_discrepancy_exponents_are_what_they_should_be():
    out = spread()
    assert out["random_is_root_n"]
    assert out["quasi_is_much_faster"]
    assert out["quasi_power"] < -0.9


def test_the_discrepancy_advantage_is_large():
    assert spread()["advantage_at_the_largest"] > 20.0


def test_the_discrepancy_of_a_single_point_is_defined():
    assert 0.0 <= mc.star_discrepancy_1d(np.array([0.5])) <= 1.0
    with pytest.raises(ValueError):
        mc.star_discrepancy_1d(np.array([]))


def test_halton_wins_in_low_dimensions():
    out = quasi()
    assert out["quasi_wins_in_two_dimensions"]
    assert out["best_advantage"] > 10.0


def test_halton_stops_winning_in_high_dimensions():
    # the negative half of the result, and the half that gets left out of the usual summary
    out = quasi()
    assert out["the_advantage_shrinks"]
    assert out["worst_advantage"] < 1.0


def test_the_quasi_exponent_degrades_with_the_dimension():
    powers = [row["quasi_power"] for row in quasi()["rows"]]
    assert powers[0] < -0.9
    assert powers[-1] > powers[0]
