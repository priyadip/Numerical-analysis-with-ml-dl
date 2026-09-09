"""Tests for nalib.sde.

Four groups. The path group asserts the two defining properties of Brownian motion, that the
variance at time ``t`` is ``t`` and that the increments are uncorrelated, against the sampling bar
rather than against a guess. It also asserts the quadratic variation both ways: it converges to the
elapsed time for a Brownian path and to zero for a differentiable one, which is the contrast that
makes ``dW**2 = dt`` a measurement rather than a definition.

The order group asserts the strong and weak orders separately, over the step ranges where each one
is measurable. It also asserts that they **differ**, which is the point: a test of either alone would
pass for a method that had the same order in both senses.

The control group asserts an exact equality on purpose. With additive noise Milstein's extra term is
identically zero, so the two methods must agree bit for bit, and any difference at all would be an
implementation error rather than a numerical one.

The pricing group checks the simulation against a closed form, and asserts that antithetic pairing
changes the constant and not the exponent, which is what a variance reduction technique does.
"""
import functools
import math

import numpy as np
import pytest

from nalib import sde


@functools.lru_cache(maxsize=None)
def walk():
    return sde.the_walk_becomes_brownian_motion()


@functools.lru_cache(maxsize=None)
def variation():
    return sde.the_quadratic_variation_is_the_time()


@functools.lru_cache(maxsize=None)
def orders():
    return sde.euler_maruyama_is_half_strong_and_one_weak()


@functools.lru_cache(maxsize=None)
def compared():
    return sde.milstein_doubles_the_strong_order_only()


@functools.lru_cache(maxsize=None)
def additive():
    return sde.the_extra_term_vanishes_for_additive_noise()


@functools.lru_cache(maxsize=None)
def correction():
    return sde.the_ito_correction_is_measurable()


@functools.lru_cache(maxsize=None)
def pricing():
    return sde.black_scholes_by_simulation()


# ------------------------------------------------------------------ the paths


@pytest.mark.parametrize("steps", [0, 1, 50])
@pytest.mark.parametrize("paths", [1, 7])
def test_the_walk_has_the_right_shape_and_starts_at_zero(steps, paths):
    out = sde.random_walk(steps, paths=paths)
    assert out.shape == (paths, steps + 1)
    assert float(np.max(np.abs(out[:, 0]))) == 0.0


def test_the_walk_only_steps_by_one():
    jumps = np.diff(sde.random_walk(500, paths=20), axis=1)
    assert set(np.unique(jumps).tolist()) <= {-1.0, 1.0}


@pytest.mark.parametrize("steps", [1, 32, 400])
@pytest.mark.parametrize("horizon", [0.25, 1.0, 6.0])
def test_brownian_motion_has_the_right_shape_and_times(steps, horizon):
    out = sde.brownian(steps, horizon=horizon, paths=3)
    assert out["values"].shape == (3, steps + 1)
    assert out["times"].shape == (steps + 1,)
    assert abs(float(out["times"][-1]) - horizon) < 1e-12
    assert float(np.max(np.abs(out["values"][:, 0]))) == 0.0


def test_brownian_motion_rejects_bad_sizes():
    with pytest.raises(ValueError):
        sde.brownian(0)
    with pytest.raises(ValueError):
        sde.brownian(10, horizon=0.0)
    with pytest.raises(ValueError):
        sde.random_walk(-1)


def test_the_variance_at_time_t_is_t():
    out = walk()
    assert out["variance_reaches_one"]
    assert out["variance_is_linear_in_time"]


def test_the_increments_are_uncorrelated():
    assert walk()["increments_are_uncorrelated"]


def test_the_quadratic_variation_is_the_elapsed_time():
    out = variation()
    assert out["brownian_reaches_the_horizon"]
    assert abs(out["rows"][-1]["brownian"] - out["horizon"]) < 0.05


def test_the_quadratic_variation_of_a_smooth_path_vanishes():
    # the contrast, and the whole reason Ito calculus is not ordinary calculus
    out = variation()
    assert out["the_smooth_one_goes_to_zero"]
    assert out["smooth_power_is_minus_one"]


def test_the_quadratic_variation_checks_its_input():
    with pytest.raises(ValueError):
        sde.quadratic_variation(np.zeros(5), np.zeros(4))


# ------------------------------------------------------------------ the orders


def test_euler_maruyama_is_strong_order_a_half():
    out = orders()
    assert out["strong_is_a_half"]
    assert abs(out["strong_order"] - 0.5) < 0.08


def test_euler_maruyama_is_weak_order_one():
    out = orders()
    assert out["weak_is_one"]
    assert abs(out["weak_order"] - 1.0) < 0.1


def test_the_two_orders_are_different_numbers():
    # a test of either alone would pass for a method whose two orders coincided
    assert orders()["they_differ"]


def test_the_two_fits_use_different_step_ranges():
    out = orders()
    assert min(out["weak_steps"]) > min(out["strong_steps"])


def test_milstein_is_strong_order_one():
    out = compared()
    assert out["milstein_is_strong_order_one"]
    assert out["euler_is_strong_order_a_half"]


def test_milstein_is_much_more_accurate_in_the_strong_sense():
    assert compared()["strong_gain"] > 10.0


def test_milstein_buys_nothing_in_the_weak_sense():
    # the result that decides which method to use, and the opposite of the usual instinct
    out = compared()
    assert out["both_are_weak_order_one"]
    assert out["the_extra_term_buys_nothing_weakly"]
    assert out["weak_gain"] < 1.5


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_coarsening_preserves_the_brownian_path(factor):
    fine = sde.brownian(64, paths=5, seed=3)["increments"]
    coarse = sde.coarsen(fine, factor)
    assert coarse.shape == (5, 64 // factor)
    assert float(np.max(np.abs(coarse.sum(axis=1) - fine.sum(axis=1)))) < 1e-12


def test_coarsening_rejects_an_impossible_grouping():
    fine = sde.brownian(10, paths=2)["increments"]
    with pytest.raises(ValueError):
        sde.coarsen(fine, 3)
    with pytest.raises(ValueError):
        sde.coarsen(fine, 0)


@pytest.mark.parametrize("method", [sde.euler_maruyama, sde.milstein])
@pytest.mark.parametrize("steps", [1, 16, 200])
def test_both_methods_run_at_any_step_count(method, steps):
    problem = sde.geometric_brownian()
    out = method(problem, steps, paths=4)
    assert out["path"].shape == (4, steps + 1)
    assert np.all(np.isfinite(out["x"]))
    assert float(np.max(np.abs(out["path"][:, 0] - problem["start"]))) == 0.0


# ------------------------------------------------------------------ the control


def test_the_two_methods_agree_exactly_on_additive_noise():
    out = additive()
    assert out["identical"]
    assert out["worst_difference"] == 0.0


def test_the_additive_problem_has_the_right_moments():
    assert additive()["the_solution_has_the_right_moments"]


def test_the_ito_correction_is_the_gap_between_mean_and_median():
    out = correction()
    assert out["the_correction_matches"]
    assert abs(out["mean_over_median"] / out["predicted_ratio"] - 1.0) < 0.02


def test_the_naive_chain_rule_gives_the_wrong_answer():
    assert correction()["the_naive_answer_is_the_wrong_one"]


@pytest.mark.parametrize("horizon", [0.5, 1.0, 3.0])
def test_the_exact_solution_matches_its_own_moments(horizon):
    problem = sde.geometric_brownian()
    rng = np.random.default_rng(11)
    values = problem["exact"](horizon, math.sqrt(horizon) * rng.standard_normal(400000))
    assert abs(float(np.mean(values)) / problem["mean"](horizon) - 1.0) < 0.01
    assert abs(float(np.var(values, ddof=1)) / problem["variance"](horizon) - 1.0) < 0.05


# ------------------------------------------------------------------ pricing


@pytest.mark.parametrize("spot", [80.0, 100.0, 130.0])
@pytest.mark.parametrize("call", [True, False])
def test_the_closed_form_satisfies_put_call_parity(spot, call):
    strike, rate, maturity = 100.0, 0.05, 1.0
    price_call = sde.black_scholes(spot, strike, rate, 0.2, maturity, call=True)
    price_put = sde.black_scholes(spot, strike, rate, 0.2, maturity, call=False)
    parity = spot - strike * math.exp(-rate * maturity)
    assert abs((price_call - price_put) - parity) < 1e-10


def test_the_closed_form_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        sde.black_scholes(maturity=0.0)
    with pytest.raises(ValueError):
        sde.black_scholes(volatility=0.0)


@pytest.mark.parametrize("call", [True, False])
def test_the_simulation_agrees_with_the_closed_form(call):
    exact = sde.black_scholes(call=call)
    out = sde.price_by_simulation(400000, call=call)
    assert abs(out["price"] - exact) < 4.0 * out["standard_error"]


def test_the_price_converges_at_one_over_root_n():
    out = pricing()
    assert out["both_are_root_n"]
    assert abs(out["plain_power"] + 0.5) < 0.08


def test_antithetic_pairing_helps_the_price():
    assert pricing()["antithetic_helps"]


def test_the_antithetic_gain_is_the_one_lesson_92_predicts():
    # the payoff is flat below the strike, so the pair correlation is about -1/2 and not -1
    out = pricing()
    assert out["the_prediction_holds"]
    assert -0.6 < out["pair_correlation"] < -0.4


def test_antithetic_pairing_changes_the_constant_and_not_the_rate():
    # a variance reduction technique moves the constant; nothing here changes the exponent
    out = pricing()
    assert abs(out["antithetic_power"] - out["plain_power"]) < 0.15


def test_the_simulation_rejects_too_few_paths():
    with pytest.raises(ValueError):
        sde.price_by_simulation(1)
