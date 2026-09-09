"""Tests for nalib.optimize.

Five groups. The test problems are checked against their own closed form derivatives by finite
differences, with separate tolerances for the gradient and the Hessian because the two formulas
have different accuracies and a single tolerance would hide that. The classification group asserts
that three cases come back **inconclusive**, which is the answer a correct second order test must
give, and asserts separately that the two fourth power cases really do differ so that the
inconclusive verdict cannot be tightened away.

The conditioning group is the one worth the most. It asserts the flat region's width against a
formula rather than against a stored number, asserts that the width vanishes when the value at the
minimum does, and asserts that the same three problems solved as root problems reach a precision
the minimizations cannot. A later reader tempted to "fix" the minimization accuracy has to get past
all three.

The convexity group asserts a negative: that sampling's success is a Poisson coin flip with no
sharp threshold, so no sample size can be quoted as sufficient.
"""
import functools
import math

import numpy as np
import pytest

from nalib import optimize as op


@functools.lru_cache(maxsize=None)
def classification():
    return op.stationary_points_are_classified_right()


@functools.lru_cache(maxsize=None)
def flat():
    return op.the_flat_region_is_as_wide_as_predicted()


@functools.lru_cache(maxsize=None)
def halves():
    return op.you_only_get_half_the_digits()


@functools.lru_cache(maxsize=None)
def shifted():
    return op.the_loss_is_set_by_the_value_at_the_minimum()


@functools.lru_cache(maxsize=None)
def sampled():
    return op.sampling_cannot_prove_convexity()


@functools.lru_cache(maxsize=None)
def conditioning():
    return op.the_conditioning_of_every_test_problem()


# ------------------------------------------------------------------ the test problems


@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_every_problem_reports_a_true_minimizer(dimension):
    for problem in op.all_problems(dimension):
        for where in problem["minimizers"]:
            assert op.is_stationary(problem, where)
            assert abs(problem["f"](where) - problem["minimum"]) < 1e-9
            assert op.classify(problem["hessian"](where)) == "minimum"


@pytest.mark.parametrize("dimension", [1, 2, 4, 7])
def test_the_quadratic_has_the_condition_number_it_was_asked_for(dimension):
    for wanted in (1.0, 10.0, 1e3):
        problem = op.quadratic(condition=wanted, dimension=dimension)
        got = problem["condition"]
        expected = wanted if dimension > 1 else 1.0
        assert abs(got - expected) <= 1e-8 * max(expected, 1.0)


@pytest.mark.parametrize("dimension", [2, 3, 6])
def test_derivatives_match_their_closed_forms(dimension):
    for problem in op.all_problems(dimension):
        out = op.derivatives_are_right(problem)
        assert out["gradient_agrees"]
        assert out["hessian_agrees"]
        assert out["the_hessian_is_symmetric"]


def test_the_gradient_formula_is_more_accurate_than_the_hessian_formula():
    # not a stylistic preference: h**(1/3) against h**(1/4), so the gap is structural
    out = op.derivatives_are_right(op.rosenbrock(4))
    assert out["worst_gradient_gap"] < out["worst_hessian_gap"]


def test_himmelblau_really_has_four_distinct_minima():
    problem = op.himmelblau()
    corners = problem["minimizers"]
    assert len(corners) == 4
    for a in range(len(corners)):
        for b in range(a + 1, len(corners)):
            assert float(np.linalg.norm(corners[a] - corners[b])) > 1.0


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_rastrigin_is_perfectly_conditioned_and_still_hard(dimension):
    problem = op.rastrigin(dimension)
    values = np.linalg.eigvalsh(problem["hessian"](np.zeros(dimension)))
    assert abs(float(values[-1] / values[0]) - 1.0) < 1e-12
    # and yet one coordinate line already crosses many other stationary points
    line = np.linspace(-2.0, 2.0, 4001)
    slope = np.asarray([float(problem["gradient"](np.full(dimension, t))[0]) for t in line])
    crossings = int(np.sum(np.diff(np.sign(slope)) != 0))
    assert crossings > 8


def test_the_problems_reject_impossible_sizes():
    with pytest.raises(ValueError):
        op.rosenbrock(1)
    with pytest.raises(ValueError):
        op.quadratic(dimension=0)
    with pytest.raises(ValueError):
        op.quadratic(condition=0.5)
    with pytest.raises(ValueError):
        op.rastrigin(0)


# ------------------------------------------------------------------ optimality


def test_every_stationary_point_is_classified_as_expected():
    out = classification()
    assert out["all_stationary"]
    assert out["all_right"]


def test_the_test_says_inconclusive_where_it_must():
    out = classification()
    assert out["inconclusive_count"] == 3
    names = [r["case"] for r in out["rows"] if r["verdict"] == "inconclusive"]
    assert any("x**4" in n and not n.startswith("-") for n in names)
    assert any(n.startswith("-x**4") for n in names)


def test_the_inconclusive_verdict_cannot_be_tightened():
    out = op.the_fourth_power_cases_really_differ()
    assert out["gradients_agree"] and out["hessians_agree"]
    assert out["one_rises_everywhere"]
    assert out["the_other_falls_everywhere"]


@pytest.mark.parametrize("size", [1, 2, 5])
def test_classify_scales_with_the_matrix(size):
    identity = np.eye(size)
    assert op.classify(identity) == "minimum"
    assert op.classify(-identity) == "maximum"
    assert op.classify(np.zeros((size, size))) == "inconclusive"
    assert op.classify(1e12 * identity) == "minimum"
    assert op.classify(1e-12 * identity) == "minimum"


def test_classify_rejects_a_non_square_matrix():
    with pytest.raises(ValueError):
        op.classify(np.zeros((2, 3)))


# ------------------------------------------------------------------ conditioning


def test_the_flat_region_has_the_width_the_formula_predicts():
    out = flat()
    assert out["ratios_agree"]
    assert out["the_prediction_is_right"]


@pytest.mark.parametrize("curvature", [1e-3, 1.0, 1e3])
def test_the_width_scales_as_one_over_the_square_root_of_the_curvature(curvature):
    wide = op.attainable_accuracy(curvature, 1.0)
    narrow = op.attainable_accuracy(4.0 * curvature, 1.0)
    assert abs(wide / narrow - 2.0) < 1e-12


def test_attainable_accuracy_rejects_a_flat_or_negative_curvature():
    with pytest.raises(ValueError):
        op.attainable_accuracy(0.0)
    with pytest.raises(ValueError):
        op.attainable_accuracy(-1.0)


def test_minimizing_loses_half_the_digits_and_root_finding_does_not():
    out = halves()
    assert out["every_minimum_is_near_root_eps"]
    assert out["every_root_is_near_eps"]
    assert out["worst_ratio"] > 1e5


def test_the_loss_follows_the_square_root_of_the_value_at_the_minimum():
    out = shifted()
    assert out["the_power_is_a_half"]
    assert abs(out["fitted_power_of_the_value"] - 0.5) < 0.05


def test_a_zero_minimum_value_costs_nothing():
    out = shifted()
    assert out["zero_value_is_exact"]


def test_the_aspect_ratio_is_the_square_root_of_the_condition_number():
    for condition in (1e2, 1e4):
        out = op.the_condition_number_is_an_aspect_ratio(
            op.quadratic(condition=condition, dimension=2))
        assert out["there_is_a_flat_region"]
        assert out["it_is_the_square_root"]


def test_a_zero_minimum_value_has_no_flat_region_to_measure():
    out = op.the_condition_number_is_an_aspect_ratio(op.rosenbrock(2))
    assert not out["there_is_a_flat_region"]
    assert math.isnan(out["measured_aspect_ratio"])


def test_flat_region_width_needs_a_real_direction():
    with pytest.raises(ValueError):
        op.flat_region_width(lambda x: float(np.sum(np.asarray(x) ** 2)),
                             np.zeros(2), np.zeros(2))


@pytest.mark.parametrize("dimension", [2, 4])
def test_the_conditioning_table_is_consistent(dimension):
    out = op.the_conditioning_of_every_test_problem(dimension)
    assert out["every_gradient_vanishes"]
    assert out["every_point_is_a_minimum"]
    # Rastrigin is the best conditioned and the hardest, which is the point of the table
    assert "Rastrigin" in out["best_conditioned"]


# ------------------------------------------------------------------ convexity


def test_the_sampled_functions_are_all_genuinely_not_convex():
    assert sampled()["every_function_here_is_not_convex"]


def test_sampling_has_no_sharp_threshold():
    out = sampled()
    # a hit below one expected hit and a miss above it both occur, so no cutoff can be quoted
    assert out["smallest_expectation_that_found_it"] < out["largest_expectation_that_missed"]
    assert out["every_outcome_is_ordinary"]


def test_a_sample_that_finds_the_dip_can_still_understate_it():
    out = sampled()
    assert out["shallowest_fraction_of_the_dip_found"] < 0.5
    assert out["deepest_fraction_of_the_dip_found"] <= 1.0


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_a_convex_problem_yields_no_counterexample(dimension):
    out = op.smallest_curvature(op.quadratic(dimension=dimension), samples=50)
    assert out["smallest_eigenvalue_found"] > 0.0
    assert out["negative_samples"] == 0
    assert out["actually_convex"]


@pytest.mark.parametrize("dimension", [2, 3])
def test_a_non_convex_problem_is_caught(dimension):
    out = op.smallest_curvature(op.rosenbrock(dimension), samples=200)
    assert out["smallest_eigenvalue_found"] < 0.0
    assert out["verdict"] == "not convex"
    assert not out["actually_convex"]


# ------------------------------------------------------------------ generality


@pytest.mark.parametrize("dimension", [1, 2, 3, 5, 8])
def test_the_gradient_and_hessian_helpers_work_at_any_size(dimension):
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(dimension, dimension))
    matrix = matrix @ matrix.T + dimension * np.eye(dimension)
    shift = rng.normal(size=dimension)

    def f(x):
        v = np.asarray(x, dtype=float).ravel() - shift
        return float(0.5 * v @ matrix @ v)

    where = rng.normal(size=dimension)
    exact_g = matrix @ (where - shift)
    got_g = op.gradient(f, where)
    got_h = op.hessian(f, where)
    assert got_g.shape == (dimension,)
    assert got_h.shape == (dimension, dimension)
    scale = max(float(np.max(np.abs(exact_g))), 1.0)
    assert float(np.max(np.abs(got_g - exact_g))) / scale < 1e-6
    assert float(np.max(np.abs(got_h - matrix))) / float(np.max(np.abs(matrix))) < 1e-4
