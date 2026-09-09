"""Tests for nalib.constrained.

Four groups. The conditions group checks the test problems against their own optimality conditions
before any method touches them, because a stated answer that fails its own KKT residual would make
every later measurement worthless. The four conditions are asserted separately, since they fail for
different reasons.

The penalty group asserts both halves of the trade at once, that the error is ``1/mu`` and the
condition number is ``mu``, and asserts that the method **fails** above a weight it also measures.
The augmented Lagrangian group asserts the weight never grows, which is the whole claim.

The projection group asserts the defining inequality rather than feasibility, because feasibility
is necessary and not sufficient: a projection that returned any feasible point would pass a
feasibility check and fail this one.
"""
import functools
import math

import numpy as np
import pytest

from nalib import constrained as cs


@functools.lru_cache(maxsize=None)
def conditions():
    return cs.the_conditions_hold_at_the_answer()


@functools.lru_cache(maxsize=None)
def complementarity():
    return cs.strict_complementarity_is_not_automatic()


@functools.lru_cache(maxsize=None)
def penalty():
    return cs.the_penalty_trades_accuracy_for_conditioning()


@functools.lru_cache(maxsize=None)
def augmented():
    return cs.the_augmented_lagrangian_keeps_the_weight_bounded()


@functools.lru_cache(maxsize=None)
def path():
    return cs.the_central_path()


@functools.lru_cache(maxsize=None)
def paths():
    return cs.the_path_slows_where_complementarity_is_not_strict()


@functools.lru_cache(maxsize=None)
def projections():
    return cs.the_projections_are_projections()


@functools.lru_cache(maxsize=None)
def feasible():
    return cs.projected_gradient_stays_feasible()


# ------------------------------------------------------------------ the conditions


def test_every_stated_answer_satisfies_its_own_conditions():
    out = conditions()
    assert out["every_answer_satisfies_them"]
    assert out["worst_residual"] < 1e-12


@pytest.mark.parametrize("dimension", [1, 2, 5, 9])
def test_the_equality_problem_has_the_answer_it_claims(dimension):
    problem = cs.equality_problem(dimension)
    out = cs.kkt_residual(problem, problem["minimizer"], problem["multipliers"])
    assert out["worst"] < 1e-12
    assert abs(float(np.sum(problem["minimizer"])) - 1.0) < 1e-15
    assert abs(problem["multipliers"][0] + 2.0 / dimension) < 1e-15


def test_the_conditions_are_reported_one_at_a_time():
    problem = cs.equality_problem(2)
    out = cs.kkt_residual(problem, np.array([1.0, 1.0]), problem["multipliers"])
    # a point that is neither feasible nor stationary should fail both, separately
    assert out["stationarity"] > 1e-6
    assert out["feasibility"] > 1e-6


def test_a_negative_multiplier_is_caught_on_an_inequality():
    problem = cs.triangle_problem((1.5, 1.5))
    bad = problem["multipliers"].copy()
    bad[1] = -1.0
    out = cs.kkt_residual(problem, problem["minimizer"], bad)
    assert out["nonnegativity"] > 0.5


def test_complementary_slackness_is_checked_separately():
    problem = cs.triangle_problem((1.5, 1.5))
    bad = problem["multipliers"].copy()
    bad[1] = 1.0            # a multiplier on a constraint that is not active
    out = cs.kkt_residual(problem, problem["minimizer"], bad)
    assert out["complementary_slackness"] > 0.1


def test_strict_complementarity_can_fail():
    out = complementarity()
    assert out["one_of_each"]
    loose = [r for r in out["rows"] if not r["strict"]]
    assert loose
    assert min(abs(v) for v in loose[0]["multipliers_there"]) < 1e-12


def test_the_triangle_problem_is_two_dimensional():
    with pytest.raises(ValueError):
        cs.triangle_problem((1.0, 2.0, 3.0))


def test_the_equality_problem_needs_a_variable():
    with pytest.raises(ValueError):
        cs.equality_problem(0)


# ------------------------------------------------------------------ penalty


def test_the_penalty_error_is_one_over_the_weight():
    assert penalty()["the_error_is_one_over_mu"]


def test_and_the_condition_number_is_the_weight():
    assert penalty()["the_condition_is_mu"]


def test_the_penalty_method_stops_working_above_a_weight_it_can_measure():
    out = penalty()
    assert out["failures"]
    assert out["largest_weight_that_worked"] < min(out["failures"])


def test_the_reachable_violation_is_capped():
    out = penalty()
    assert out["best_violation_reached"] > 1e-13
    assert out["best_violation_reached"] < 1e-9


def test_the_penalty_weight_must_be_positive():
    with pytest.raises(ValueError):
        cs.penalty_problem(cs.equality_problem(2), 0.0)


@pytest.mark.parametrize("dimension", [1, 2, 4])
def test_the_penalty_subproblem_is_built_at_any_size(dimension):
    problem = cs.equality_problem(dimension)
    inner = cs.penalty_problem(problem, 100.0)
    where = problem["start"]
    assert np.asarray(inner["gradient"](where)).shape == (dimension,)
    assert np.asarray(inner["hessian"](where)).shape == (dimension, dimension)


# ------------------------------------------------------------------ augmented Lagrangian


def test_the_weight_never_grows():
    out = augmented()
    assert out["weight_never_grew"]
    assert out["final_weight"] == 10.0


def test_the_condition_number_stays_small():
    out = augmented()
    assert out["condition_stayed_at"] < 20.0
    assert out["penalty_condition_for_the_same_violation"] > 1e8


def test_the_multiplier_converges_to_the_exact_one():
    out = augmented()
    assert out["multiplier_error"] < 1e-7
    assert abs(out["exact_multipliers"][0] + 1.0) < 1e-15


def test_the_violation_falls_by_about_ten_each_round():
    rows = augmented()["rows"]
    ratios = [rows[k]["violation"] / rows[k - 1]["violation"]
              for k in range(1, len(rows))]
    assert max(ratios) < 0.2
    assert min(ratios) > 0.01


@pytest.mark.parametrize("dimension", [1, 3, 6])
def test_the_augmented_lagrangian_works_at_any_size(dimension):
    # the bar is 1e-5 and not 1e-6 on purpose: the multiplier update converges at a rate set by
    # the weight against the curvature of the problem, and that ratio changes with the size, so a
    # single tight absolute bar across sizes would be testing the arithmetic of one of them
    problem = cs.equality_problem(dimension)
    out = cs.augmented_lagrangian(problem)
    assert out["error"] < 1e-5
    assert out["multiplier_error"] < 1e-5


# ------------------------------------------------------------------ barrier


def test_every_point_of_the_central_path_is_strictly_feasible():
    assert path()["every_point_strictly_feasible"]


def test_the_path_ends_at_the_solution():
    out = path()
    assert out["final_distance"] < 1e-4


def test_the_multipliers_come_free_from_the_path():
    out = path()
    assert out["the_multipliers_come_free"]
    last = out["rows"][-1]["multiplier_estimate"]
    assert abs(last[0] - 2.0) < 1e-3


def test_the_path_is_more_accurate_where_complementarity_is_strict():
    assert paths()["the_strict_case_is_more_accurate"]


def test_and_the_difference_is_large():
    rows = paths()["rows"]
    strict = [r for r in rows if r["strict_complementarity"]][0]
    loose = [r for r in rows if not r["strict_complementarity"]][0]
    assert loose["final_multiplier_error"] > 100 * strict["final_multiplier_error"]


# ------------------------------------------------------------------ projection


def test_each_projection_satisfies_the_defining_inequality():
    out = projections()
    assert out["all_are_projections"]
    assert out["all_idempotent"]


@pytest.mark.parametrize("size", [1, 2, 5, 17])
def test_the_simplex_projection_lands_on_the_simplex(size):
    rng = np.random.default_rng(42)
    point = rng.normal(scale=3.0, size=size)
    out = cs.project_onto_simplex(point)
    assert out.shape == (size,)
    assert abs(float(np.sum(out)) - 1.0) < 1e-12
    assert float(np.min(out)) >= -1e-15


@pytest.mark.parametrize("size", [1, 3, 8])
def test_a_point_already_inside_is_left_alone(size):
    rng = np.random.default_rng(7)
    inside = rng.dirichlet(np.ones(size))
    assert float(np.max(np.abs(cs.project_onto_simplex(inside) - inside))) < 1e-12
    small = 0.3 * rng.normal(size=size) / max(size, 1)
    assert float(np.max(np.abs(cs.project_onto_box(small, -1.0, 1.0) - small))) < 1e-15


@pytest.mark.parametrize("size", [1, 2, 6])
def test_the_ball_projection_lands_on_the_sphere_when_outside(size):
    rng = np.random.default_rng(42)
    point = 10.0 * rng.normal(size=size)
    out = cs.project_onto_ball(point, 2.0)
    assert abs(float(np.linalg.norm(out)) - 2.0) < 1e-12


def test_the_simplex_projection_refuses_an_empty_vector():
    with pytest.raises(ValueError):
        cs.project_onto_simplex(np.zeros(0))


def test_projected_gradient_keeps_every_iterate_feasible():
    out = feasible()
    assert out["always_feasible"]
    assert out["sums_are_one"]
    assert out["nothing_negative"]


def test_it_agrees_with_the_closed_form_projection():
    assert feasible()["matches_the_direct_projection"]


@pytest.mark.parametrize("size", [2, 4, 9])
def test_projected_gradient_on_a_box_at_any_size(size):
    rng = np.random.default_rng(42)
    target = 3.0 * rng.normal(size=size)

    def f(v):
        gap = np.asarray(v, dtype=float) - target
        return float(gap @ gap)

    def gradient(v):
        return 2.0 * (np.asarray(v, dtype=float) - target)

    out = cs.projected_gradient(f, gradient,
                                lambda p: cs.project_onto_box(p, -1.0, 1.0),
                                np.zeros(size), step=0.25)
    assert out["converged"]
    assert float(np.max(np.abs(out["x"]))) <= 1.0 + 1e-12
    assert float(np.max(np.abs(out["x"] - np.clip(target, -1.0, 1.0)))) < 1e-6
