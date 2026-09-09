"""Tests for nalib.stochastic.

Four groups. The noise group asserts two exponents that are properties of sampling rather than of
any algorithm: the noise ball grows like the square root of the step, and the gradient noise falls
like one over the square root of the batch. Both are fitted rather than compared against a stored
number.

The schedule group asserts that **each** Robbins-Monro condition has a failing case here, including
the one that surprises people: a schedule that decays too fast is the worst of the four.

The acceleration group asserts the rates against their closed forms, and asserts the exponent ratio
rather than the exponents on Nesterov's worst function, because a measurement on one instance and a
worst case bound over a class are different statements.

The adaptive group asserts a negative as well as a positive: the plain methods diverge on the badly
scaled problem with a step chosen blind, and they converge and are still beaten when given the best
step available to them.
"""
import functools
import math

import numpy as np
import pytest

from nalib import stochastic as st


@functools.lru_cache(maxsize=None)
def ball():
    return st.the_noise_ball_grows_like_the_square_root_of_the_step()


@functools.lru_cache(maxsize=None)
def schedules():
    return st.the_robbins_monro_conditions_decide_convergence()


@functools.lru_cache(maxsize=None)
def batches():
    return st.the_noise_falls_like_one_over_root_batch()


@functools.lru_cache(maxsize=None)
def rates():
    return st.momentum_turns_the_condition_number_into_its_square_root()


@functools.lru_cache(maxsize=None)
def accelerated():
    return st.acceleration_doubles_the_exponent()


@functools.lru_cache(maxsize=None)
def adaptive():
    return st.the_adaptive_methods_rescale_the_coordinates()


@functools.lru_cache(maxsize=None)
def decoupled():
    return st.adam_and_adamw_differ()


@functools.lru_cache(maxsize=None)
def picked():
    return st.which_schedule()


# ------------------------------------------------------------------ the problem


@pytest.mark.parametrize("variables", [1, 3, 10, 25])
def test_the_problem_knows_its_own_answer(variables):
    problem = st.least_squares_problem(variables=variables)
    slope = np.asarray(problem["gradient"](problem["minimizer"]), dtype=float)
    assert slope.shape == (variables,)
    assert float(np.max(np.abs(slope))) < 1e-8
    assert problem["f"](problem["minimizer"]) == problem["minimum"]


@pytest.mark.parametrize("scaling", [1.0, 100.0, 10000.0])
def test_the_scaling_dial_moves_the_condition_number(scaling):
    problem = st.least_squares_problem(scaling=scaling)
    assert problem["condition"] > 1.0
    if scaling > 1.0:
        plain = st.least_squares_problem(scaling=1.0)
        assert problem["condition"] > 100.0 * plain["condition"]


def test_the_problem_refuses_an_empty_size():
    with pytest.raises(ValueError):
        st.least_squares_problem(samples=0)
    with pytest.raises(ValueError):
        st.least_squares_problem(variables=0)


@pytest.mark.parametrize("batch", [1, 8, 64])
def test_the_batch_gradient_averages_to_the_full_one(batch):
    problem = st.least_squares_problem(samples=200, variables=4)
    rng = np.random.default_rng(3)
    where = problem["minimizer"] + rng.normal(size=4)
    exact = np.asarray(problem["gradient"](where), dtype=float)
    total = np.zeros(4)
    trials = 4000
    for _ in range(trials):
        rows = rng.integers(0, problem["samples"], size=batch)
        total += np.asarray(problem["batch_gradient"](where, rows), dtype=float)
    assert float(np.max(np.abs(total / trials - exact))) < 0.1 * float(
        np.max(np.abs(exact)))


# ------------------------------------------------------------------ the noise


def test_the_noise_ball_grows_like_the_square_root_of_the_step():
    out = ball()
    assert out["the_power_is_a_half"]
    assert abs(out["fitted_power_of_the_step"] - 0.5) < 0.08


def test_the_constant_in_the_noise_ball_barely_moves():
    assert ball()["spread_in_the_constant"] < 1.2


def test_a_bigger_step_gives_a_bigger_ball():
    rows = sorted(ball()["rows"], key=lambda r: r["step"])
    radii = [r["tail_distance"] for r in rows]
    assert radii == sorted(radii)


def test_the_gradient_noise_falls_like_one_over_root_batch():
    out = batches()
    assert out["the_power_is_minus_a_half"]
    assert out["constant_above_four"] < 1.15


# ------------------------------------------------------------------ schedules


def test_the_best_schedule_satisfies_both_conditions():
    out = schedules()
    assert out["the_best_satisfies_both"]


def test_the_worst_schedule_is_the_one_that_decays_too_fast():
    # this is the surprising half: a more cautious looking schedule is broken
    out = schedules()
    assert out["the_worst_decays_too_fast"]
    assert out["spread"] > 50.0


def test_each_condition_has_a_failing_case():
    rows = {r["schedule"]: r for r in schedules()["rows"]}
    assert rows["one over k to the 1.5"]["sum_of_steps"] < 10.0
    assert rows["constant"]["sum_of_squares"] > 1.0
    assert rows["one over k"]["sum_of_squares"] < 1.0


def test_an_unknown_schedule_is_refused():
    with pytest.raises(ValueError):
        st.schedule("made up", 0.1)


def test_the_cosine_schedule_warms_up_then_decays():
    rule = st.cosine_schedule(0.1, 1000, warmup=100)
    assert rule(0) < rule(50) < rule(99)
    assert rule(150) > rule(500) > rule(950)
    assert rule(999) < 1e-4


def test_any_decaying_schedule_beats_the_constant_one():
    out = picked()
    assert out["constant_is_worst"]
    assert out["best_over_constant"] > 5.0


# ------------------------------------------------------------------ acceleration


def test_plain_descent_matches_its_closed_form_rate():
    assert rates()["plain_matches_its_bound"]


def test_heavy_ball_matches_the_square_root_rate():
    assert rates()["heavy_matches_its_bound"]


def test_the_speedup_grows_with_the_condition_number():
    rows = sorted(rates()["rows"], key=lambda r: r["condition"])
    speedups = [r["speedup"] for r in rows]
    assert speedups == sorted(speedups)
    assert rates()["biggest_speedup"] > 10.0


def test_acceleration_doubles_the_exponent():
    out = accelerated()
    assert out["the_ratio_is_two"]
    assert abs(out["exponent_ratio"] - 2.0) < 0.1


def test_and_neither_exponent_is_the_quoted_bound():
    # asserting the honest reading: one instance is not a worst case over a class
    assert accelerated()["neither_matches_the_quoted_bound"]


def test_the_worst_function_needs_room():
    with pytest.raises(ValueError):
        st.acceleration_doubles_the_exponent(variables=100, budget=800)


# ------------------------------------------------------------------ the adaptive methods


@pytest.mark.parametrize("method", ["sgd", "momentum", "nesterov", "adagrad",
                                    "rmsprop", "adam", "adamw"])
def test_every_method_runs_and_improves(method):
    problem = st.least_squares_problem(samples=400, variables=5)
    out = st.stochastic_descent(problem, method=method, step=0.01, batch=16,
                                rounds=3000)
    assert out["x"].shape == (5,)
    assert out["final_distance"] < float(np.linalg.norm(problem["minimizer"]))


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError):
        st.stochastic_descent(st.least_squares_problem(samples=100, variables=3),
                              method="made up", rounds=10)


def test_only_the_plain_methods_diverged():
    out = adaptive()
    assert out["methods_that_diverged"]
    assert out["everything_that_diverged_was_non_adaptive"]


def test_an_adaptive_method_wins_when_the_scaling_is_bad():
    out = adaptive()
    assert out["adaptive_wins_when_badly_scaled"]
    assert out["margin_when_badly_scaled"] > out["margin_when_well_scaled"]


def test_the_margin_is_much_larger_when_badly_scaled():
    out = adaptive()
    assert out["margin_when_badly_scaled"] > 10.0
    assert out["margin_when_well_scaled"] < 10.0


def test_adam_and_adamw_are_the_same_algorithm_without_decay():
    out = decoupled()
    assert out["identical_without_decay"]
    assert out["difference_without_decay"] == 0.0


def test_and_differ_with_decay_unevenly():
    out = decoupled()
    assert out["different_with_decay"]
    assert out["the_difference_is_uneven_across_coordinates"]
    assert out["largest_coordinate_difference"] > 10.0 * out["smallest_coordinate_difference"]
