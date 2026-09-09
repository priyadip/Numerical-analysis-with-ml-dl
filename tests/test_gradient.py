"""Tests for nalib.gradient.

Four groups. The line search group asserts each condition separately, including that backtracking
rejects an uphill direction rather than quietly halving forever, and that the strong Wolfe search
extends a step that Armijo would have accepted far too short.

The rate group is the one worth the most. It asserts the Kantorovich bound is attained at ten
variables **and not** at two, so a later reader cannot "simplify" the measurement down to a two
variable problem without breaking a test that says two variables is not enough.

The Newton group asserts three failures on purpose: that the gradient rises before it falls, that
the plain iteration stops at a saddle at four variables, and that the safeguards make two runs
worse. Each is a real property of the method and each would look like a bug to someone tidying up.
"""
import functools
import math

import numpy as np
import pytest

from nalib import gradient as gr
from nalib.optimize import quadratic, rosenbrock


@functools.lru_cache(maxsize=None)
def rates():
    return gr.the_kantorovich_bound_is_sharp()


@functools.lru_cache(maxsize=None)
def counts():
    return gr.the_step_count_grows_with_the_condition_number()


@functools.lru_cache(maxsize=None)
def degenerate():
    return gr.an_eigenvector_start_is_degenerate()


@functools.lru_cache(maxsize=None)
def quadratic_newton():
    return gr.newton_is_quadratic_and_not_monotone()


@functools.lru_cache(maxsize=None)
def wrong():
    return gr.plain_newton_can_find_the_wrong_minimum()


@functools.lru_cache(maxsize=None)
def safeguards():
    return gr.the_safeguards_help_and_hurt()


@functools.lru_cache(maxsize=None)
def searches():
    return gr.wolfe_is_stricter_than_armijo()


@functools.lru_cache(maxsize=None)
def cost():
    return gr.gradient_against_newton()


# ------------------------------------------------------------------ line searches


@pytest.mark.parametrize("dimension", [1, 2, 5])
def test_backtracking_gives_a_step_that_satisfies_armijo(dimension):
    problem = quadratic(condition=50.0, dimension=dimension)
    f, g = problem["f"], problem["gradient"]
    x = problem["start"]
    slope = np.asarray(g(x), dtype=float)
    out = gr.backtracking(f, float(f(x)), slope, x, -slope)
    assert out["accepted"]
    assert out["armijo"]
    assert out["f"] <= float(f(x)) - 1e-4 * out["step"] * float(slope @ slope) + 1e-15


def test_backtracking_refuses_an_uphill_direction():
    problem = quadratic(dimension=2)
    slope = np.asarray(problem["gradient"](problem["start"]), dtype=float)
    with pytest.raises(ValueError):
        gr.backtracking(problem["f"], float(problem["f"](problem["start"])),
                        slope, problem["start"], slope)


def test_strong_wolfe_refuses_an_uphill_direction():
    problem = quadratic(dimension=2)
    slope = np.asarray(problem["gradient"](problem["start"]), dtype=float)
    with pytest.raises(ValueError):
        gr.strong_wolfe(problem["f"], problem["gradient"], problem["start"], slope)


@pytest.mark.parametrize("dimension", [1, 2, 4])
def test_strong_wolfe_satisfies_both_conditions(dimension):
    problem = quadratic(condition=30.0, dimension=dimension)
    f, g = problem["f"], problem["gradient"]
    x = problem["start"]
    slope = np.asarray(g(x), dtype=float)
    out = gr.strong_wolfe(f, g, x, -slope)
    assert out["accepted"]
    assert out["armijo"]
    assert out["curvature"]


def test_the_curvature_condition_is_free_on_rosenbrock():
    assert searches()["on_rosenbrock_curvature_is_free"]


def test_and_essential_on_a_flat_quadratic():
    out = searches()
    assert out["on_the_flat_quadratic_armijo_is_too_short"]
    assert out["biggest_step_ratio"] > 100.0


def test_the_exact_step_needs_a_positive_definite_matrix():
    with pytest.raises(ValueError):
        gr.exact_step_on_a_quadratic(-np.eye(2), np.ones(2))


@pytest.mark.parametrize("dimension", [1, 3, 6])
def test_the_exact_step_lands_on_the_minimum_along_the_gradient(dimension):
    problem = quadratic(condition=20.0, dimension=dimension)
    matrix, centre = problem["matrix"], problem["minimizers"][0]
    gap = problem["start"] - centre
    slope = matrix @ gap
    step = gr.exact_step_on_a_quadratic(matrix, slope)
    after = gap - step * slope
    # the new gradient is orthogonal to the direction just taken, which is what exact means
    assert abs(float((matrix @ after) @ slope)) <= 1e-8 * float(np.linalg.norm(slope)) ** 2


# ------------------------------------------------------------------ the rate


def test_the_kantorovich_bound_is_attained_in_enough_dimensions():
    assert rates()["sharp_at_the_largest_dimension"]


def test_two_variables_is_not_enough_to_see_the_worst_case():
    out = rates()
    assert out["not_sharp_at_the_smallest"]
    assert out["worst_fraction_at_the_smallest"] < 0.9


def test_no_measured_rate_ever_exceeds_the_bound():
    for row in rates()["rows"]:
        assert row["measured_rate"] <= row["bound"] * (1.0 + 1e-6)


def test_the_step_count_grows_linearly_in_the_condition_number():
    out = counts()
    assert out["the_growth_is_linear"]
    assert abs(out["fitted_power_of_the_condition_number"] - 1.0) < 0.1


def test_an_eigenvector_start_finishes_in_one_step():
    out = degenerate()
    assert out["one_step_is_enough"]
    assert out["a_spread_start_takes"] > 1000


def test_what_is_left_after_that_step_is_rounding_times_the_condition_number():
    assert degenerate()["the_residue_is_rounding_times_the_condition_number"]


@pytest.mark.parametrize("dimension", [2, 4, 8])
def test_steepest_descent_converges_on_a_quadratic_at_any_size(dimension):
    problem = quadratic(condition=25.0, dimension=dimension)
    out = gr.steepest_descent(problem, tol=1e-9, search="exact")
    assert out["converged"]
    assert float(np.linalg.norm(out["x"] - problem["minimizers"][0])) < 1e-6


def test_an_exact_line_search_needs_a_quadratic():
    with pytest.raises(ValueError):
        gr.steepest_descent(rosenbrock(2), search="exact")


# ------------------------------------------------------------------ Newton


def test_newton_is_quadratic_by_the_constant_and_not_by_a_fitted_order():
    out = quadratic_newton()
    assert out["the_constant_is_size_independent"]
    assert out["spread_in_the_constant"] < 3.0


def test_the_gradient_rises_before_it_falls():
    out = quadratic_newton()
    assert out["every_run_rises_first"]
    assert out["biggest_rise_anywhere"] > 100.0


def test_the_plain_iteration_stops_where_the_gradient_vanishes_and_no_further():
    out = wrong()
    assert out["it_happens"]
    assert out["every_wrong_stop_converged_cleanly"]
    assert out["the_wrong_stops_have_a_larger_value"]


def test_it_stops_at_a_saddle_at_least_once():
    out = wrong()
    assert out["it_stops_at_saddles_too"]
    assert out["wrong_stops_that_are_saddles"]


def test_the_safeguards_rescue_the_saddle():
    out = safeguards()
    assert out["it_rescues_the_saddle"]
    assert 4 in out["rescued"]


def test_the_safeguards_also_make_things_worse():
    # not a defect to be tidied away: guaranteed descent is not descent to the right place
    out = safeguards()
    assert out["it_also_makes_things_worse"]
    assert out["spoiled"]


def test_the_hessian_is_often_indefinite():
    out = gr.the_hessian_is_not_always_positive_definite(dimension=2)
    assert 0.05 < out["indefinite_in_the_box"] < 0.95
    bigger = gr.the_hessian_is_not_always_positive_definite(dimension=5)
    assert bigger["indefinite_in_the_box"] > out["indefinite_in_the_box"]


@pytest.mark.parametrize("dimension", [1, 2, 4, 7])
def test_newton_solves_a_quadratic_in_one_step(dimension):
    problem = quadratic(condition=1e4, dimension=dimension)
    out = gr.newton(problem, tol=1e-8)
    assert out["steps"] <= 1
    assert float(np.linalg.norm(out["x"] - problem["minimizers"][0])) < 1e-8


def test_the_cost_comparison_is_made_where_both_agree():
    out = cost()
    assert out["newton_solves_a_quadratic_in_one_step"]
    assert out["smallest_evaluation_ratio"] > 1.0


def test_newton_beats_steepest_descent_by_more_as_the_condition_number_grows():
    rows = [r for r in cost()["rows"] if r["variables"] == 10]
    rows.sort(key=lambda r: r["condition"])
    ratios = [r["evaluation_ratio"] for r in rows]
    assert ratios == sorted(ratios)
    assert ratios[-1] / ratios[0] > 10.0


# ------------------------------------------------------------------ generality


@pytest.mark.parametrize("dimension", [1, 2, 3, 6])
def test_every_routine_works_at_any_size(dimension):
    problem = quadratic(condition=40.0, dimension=dimension)
    slow = gr.steepest_descent(problem, tol=1e-8, search="exact")
    fast = gr.newton(problem, tol=1e-8)
    safe = gr.newton(problem, tol=1e-8, safeguard=True)
    for out in (slow, fast, safe):
        assert out["x"].shape == (dimension,)
        assert float(np.linalg.norm(out["x"] - problem["minimizers"][0])) < 1e-6
