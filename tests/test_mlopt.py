"""Tests for nalib.mlopt.

Four groups. The flow group asserts that gradient descent really is Euler's method on the gradient
flow, and that heavy ball really is a damped oscillator, by fitting the order of convergence to the
continuous trajectory. A first order fit is the assertion; anything else would mean one of the two
implementations does not solve what it claims to.

The stability group asserts ``h < 2/L``, twice. Once globally on a quadratic, where the bisection
locates the boundary to 12 digits, and once locally at each of two minima with different curvature,
where the same condition explains a preference between them that is usually attributed to noise.

The noise group asserts the finite population variance formula, including its exact zero at the full
batch, and asserts that the batch size changes which minimum a run ends in only inside the window
the stability group identified. Both directions are checked: the full batch never leaves the sharp
well, and a batch of one always does.

The second order group asserts the negative result. Newton is exact in one step and cheap in flops,
and it still loses to plain descent at a small batch, because the curvature estimate is singular and
gets inverted. The group also asserts the two exact identities that survive: the natural gradient's
invariance under reparametrization, and ``F = J'J`` for a Gaussian model.
"""
import functools
import math

import numpy as np
import pytest

from nalib import mlopt


@functools.lru_cache(maxsize=None)
def euler():
    return mlopt.gradient_descent_is_explicit_euler()


@functools.lru_cache(maxsize=None)
def limit():
    return mlopt.the_step_limit_is_the_stability_limit()


@functools.lru_cache(maxsize=None)
def momentum():
    return mlopt.momentum_is_a_second_order_ode()


@functools.lru_cache(maxsize=None)
def counts():
    return mlopt.conditioning_decides_the_iteration_count()


@functools.lru_cache(maxsize=None)
def variance():
    return mlopt.the_gradient_noise_falls_like_the_batch_size()


@functools.lru_cache(maxsize=None)
def scaling():
    return mlopt.the_linear_scaling_rule_stops_at_the_stability_limit()


@functools.lru_cache(maxsize=None)
def wells():
    return mlopt.a_large_step_cannot_sit_in_a_sharp_minimum()


@functools.lru_cache(maxsize=None)
def emptying():
    return mlopt.minibatch_noise_empties_the_sharp_well_first()


@functools.lru_cache(maxsize=None)
def flatness():
    return mlopt.sharpness_predicts_the_damage()


@functools.lru_cache(maxsize=None)
def noisy_newton():
    return mlopt.a_noisy_hessian_makes_newton_worse()


@functools.lru_cache(maxsize=None)
def flops():
    return mlopt.cost_is_not_the_reason_second_order_is_rare()


@functools.lru_cache(maxsize=None)
def invariance():
    return mlopt.the_natural_gradient_ignores_the_parametrization()


@functools.lru_cache(maxsize=None)
def fisher():
    return mlopt.gauss_newton_is_the_fisher_information()


# ------------------------------------------------------------------ the problems


def test_a_quadratic_has_the_curvatures_it_was_given():
    values = np.array([0.5, 2.0, 7.0])
    problem = mlopt.quadratic(values, seed=3)
    assert np.allclose(np.sort(np.linalg.eigvalsh(problem["hessian"])), np.sort(values),
                       rtol=1e-12)
    assert problem["condition"] == pytest.approx(values.max() / values.min(), rel=1e-12)


def test_a_quadratic_is_zero_at_its_minimum():
    problem = mlopt.quadratic([1.0, 4.0], seed=1)
    assert problem["f"](problem["minimum"]) == 0.0
    assert np.allclose(problem["gradient"](problem["minimum"]), 0.0)


def test_a_quadratic_rejects_a_negative_curvature():
    with pytest.raises(ValueError):
        mlopt.quadratic([1.0, -1.0])


def test_the_two_wells_end_up_exactly_equally_deep():
    problem = mlopt.two_wells(samples=200, seed=5)
    assert problem["flat_value"] == pytest.approx(problem["sharp_value"], abs=1e-12)
    assert problem["sharp_curvature"] > 4.0 * problem["flat_curvature"]


def test_two_wells_rejects_an_empty_sum():
    with pytest.raises(ValueError):
        mlopt.two_wells(samples=0)


def test_a_full_batch_gradient_is_the_average_of_the_terms():
    problem = mlopt.least_squares_model(60, 5, noise=0.1, seed=7)
    point = np.ones(5)
    full = problem["gradient"](point)
    parts = np.mean([problem["gradient"](point, np.array([i])) for i in range(60)], axis=0)
    assert np.allclose(full, parts, atol=1e-12)


def test_the_least_squares_model_solves_its_own_problem():
    problem = mlopt.least_squares_model(80, 6, noise=0.05, condition=10.0, seed=9)
    assert np.max(np.abs(problem["gradient"](problem["solution"]))) < 1e-10


def test_the_least_squares_model_rejects_an_empty_shape():
    with pytest.raises(ValueError):
        mlopt.least_squares_model(0, 3)


# ------------------------------------------------------------------ the flow


def test_gradient_descent_is_first_order_on_the_flow():
    assert euler()["it_is_first_order"]
    assert abs(euler()["order"] - 1.0) < 0.05


def test_the_flow_error_halves_when_the_step_does():
    errors = [row["error"] for row in euler()["rows"]]
    for i in range(len(errors) - 1):
        assert errors[i] / errors[i + 1] == pytest.approx(2.0, rel=0.05)


def test_heavy_ball_tracks_the_damped_oscillator():
    assert momentum()["it_tracks_the_ode"]
    assert abs(momentum()["order"] - 1.0) < 0.1


def test_the_flow_solver_stays_at_a_minimum():
    problem = mlopt.quadratic([1.0, 3.0], seed=2)
    out = mlopt.gradient_flow(problem, problem["minimum"], 1.0, steps=64)
    assert np.max(np.abs(out["x"])) < 1e-14


# ------------------------------------------------------------------ stability


def test_the_step_limit_is_two_over_the_largest_curvature():
    assert limit()["the_limit_is_two_over_l"]
    assert limit()["relative_gap"] < 1e-6


def test_the_amplification_crosses_one_at_the_limit():
    for row in limit()["rows"]:
        assert (row["amplification"] > 1.0) == (row["factor"] > 1.0)


def test_the_iteration_count_is_the_condition_number():
    assert counts()["the_prediction_holds"]
    assert counts()["worst_ratio"] < 1.5


def test_preconditioning_removes_the_condition_number():
    assert counts()["whitening_removes_it"]
    assert counts()["largest_saving"] > 1000.0


def test_each_well_has_its_own_step_limit():
    assert wells()["both_limits_are_two_over_the_curvature"]
    assert wells()["worst_gap"] < 0.02


def test_the_sharp_well_loses_its_step_first():
    assert wells()["there_is_a_window"]
    assert wells()["limits"]["sharp"]["destabilizes"] < wells()["limits"]["flat"]["destabilizes"]


def test_leaving_a_well_needs_a_much_larger_step_than_destabilizing_it():
    assert wells()["ejection_comes_much_later"]
    assert wells()["ejection_over_destabilization"] > 3.0


def test_the_swing_appears_only_past_the_limit():
    for row in wells()["rows"]:
        if row["over_the_sharp_limit"] < 1.0:
            assert row["sharp_swing"] < 1e-10


# ------------------------------------------------------------------ noise


def test_the_gradient_variance_follows_the_finite_population_formula():
    assert variance()["the_formula_holds"]
    assert variance()["worst_ratio"] < 1.1


def test_the_full_batch_gradient_has_no_noise_at_all():
    assert variance()["the_full_batch_is_exact"]
    assert variance()["full_batch_variance"] < 1e-25


def test_the_variance_falls_with_the_batch_size():
    measured = [row["measured"] for row in variance()["rows"]]
    assert all(measured[i] > measured[i + 1] for i in range(len(measured) - 1))


def test_the_linear_scaling_rule_holds_while_the_step_is_stable():
    assert scaling()["the_rule_holds_while_stable"]
    assert scaling()["largest_stable_batch"] >= 8


def test_the_linear_scaling_rule_breaks_at_the_stability_limit():
    assert scaling()["it_breaks_at_the_stability_limit"]
    for row in scaling()["rows"]:
        assert row["stable"] == (row["step_over_limit"] < 1.0)


def test_the_wells_are_equally_deep_before_anything_is_measured():
    assert emptying()["the_wells_are_equally_deep"]
    assert emptying()["curvature_ratio"] > 4.0


def test_a_small_step_holds_every_batch_size():
    assert emptying()["a_small_step_holds_every_batch"]


def test_the_full_batch_never_leaves_the_sharp_well():
    assert emptying()["the_full_batch_never_leaves"]


def test_a_small_batch_empties_it_where_the_full_batch_cannot():
    assert emptying()["noise_empties_it_where_the_full_batch_cannot"]
    assert emptying()["largest_noise_advantage"] > 0.5


def test_smaller_batches_leave_sooner():
    assert emptying()["smaller_batches_leave_sooner"]


# ------------------------------------------------------------------ curvature


def test_sharpness_matches_the_trace_prediction():
    assert flatness()["the_prediction_is_exact"]
    assert flatness()["worst_ratio"] < 1.05


def test_sharpness_scales_with_the_square_of_the_radius():
    grouped = {}
    for row in flatness()["rows"]:
        grouped.setdefault(row["condition"], []).append(row)
    for entries in grouped.values():
        entries = sorted(entries, key=lambda r: r["radius"])
        for i in range(len(entries) - 1):
            factor = (entries[i + 1]["radius"] / entries[i]["radius"]) ** 2
            assert entries[i + 1]["measured"] / entries[i]["measured"] == pytest.approx(
                factor, rel=1e-6)


def test_the_worst_case_exceeds_the_average_when_the_problem_is_ill_conditioned():
    assert flatness()["worst_case_over_average"] > 2.0
    for row in flatness()["rows"]:
        assert row["worst_case"] >= row["measured"] - 1e-15


def test_a_sampled_curvature_is_singular_below_the_dimension():
    assert noisy_newton()["the_curvature_is_singular_below_the_dimension"]


def test_newton_diverges_at_the_smallest_batch():
    assert noisy_newton()["newton_diverges_at_the_smallest_batch"]


def test_newton_is_worse_than_gradient_descent_at_a_small_batch():
    assert noisy_newton()["newton_is_worse_at_small_batches"]
    assert noisy_newton()["worst_ratio"] > 2.0


def test_newton_wins_again_at_a_large_batch():
    assert noisy_newton()["crossover_batch"] is not None
    assert noisy_newton()["rows"][-1]["ratio"] < 1.0


# ------------------------------------------------------------------ cost and invariance


def test_newton_solves_a_quadratic_in_one_step():
    assert flops()["newton_converges_in_one_step"]


def test_newton_wins_the_flop_count_at_every_dimension_tested():
    assert flops()["newton_wins_everywhere_tested"]
    assert flops()["worst_ratio"] > 1.0


def test_the_flop_advantage_falls_like_one_over_the_dimension():
    assert flops()["the_advantage_falls_like_one_over_d"]
    assert abs(flops()["slope"] + 1.0) < 0.2


def test_the_flop_crossover_is_the_iteration_count():
    assert flops()["the_crossover_is_the_iteration_count"]
    assert abs(flops()["crossover_ratio"] - 1.0) < 0.3


def test_the_break_even_count_is_about_the_dimension():
    # d + 1 gradient steps for the d by d solve and the extra pass, plus d/(3*samples per
    # parameter) for the cubic term, so the count is the dimension and a little
    for row in flops()["rows"]:
        assert row["dimension"] <= row["break_even"] <= 1.5 * row["dimension"] + 2.0


def test_the_natural_gradient_does_not_move_under_a_change_of_variables():
    assert invariance()["natural_is_invariant"]
    assert invariance()["worst_natural_drift"] < 1e-10


def test_plain_gradient_descent_does_move():
    assert invariance()["plain_is_not"]
    assert invariance()["worst_plain_drift"] > 1e-3


def test_a_natural_gradient_step_solves_a_linear_model_in_one_step():
    problem = mlopt.least_squares_model(50, 4, condition=100.0, seed=13)
    start = problem["solution"] + np.ones(4)
    moved = start - mlopt.natural_gradient_step(problem["design"], start, problem["target"])
    assert np.max(np.abs(moved - problem["solution"])) < 1e-8


def test_gauss_newton_is_exactly_the_fisher_information():
    assert fisher()["fisher_equals_gauss_newton"]


def test_they_are_the_hessian_only_at_a_zero_residual():
    assert fisher()["they_agree_at_zero_residual"]
    assert fisher()["they_disagree_at_a_large_one"]
    assert fisher()["largest_hessian_gap"] > 0.1


def test_the_hessian_gap_grows_with_the_residual():
    gaps = [row["hessian_gap"] for row in fisher()["rows"]]
    assert all(gaps[i] <= gaps[i + 1] for i in range(len(gaps) - 1))


def test_every_measurement_carries_a_note():
    for out in (euler(), limit(), momentum(), counts(), variance(), scaling(), wells(),
                emptying(), flatness(), noisy_newton(), flops(), invariance(), fisher()):
        assert isinstance(out["note"], str) and out["note"]
