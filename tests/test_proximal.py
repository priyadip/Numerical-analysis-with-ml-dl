"""Tests for nalib.proximal.

Four groups. The operator group checks the two defining properties rather than the answers: that
the prox of an indicator **is** lesson 88's projection, exactly, and that every prox here is firmly
nonexpansive. Both are identities, so a tolerance would be the wrong instrument.

The sparsity group asserts a comparison rather than a number: only the L1 prox produces exact
zeros, and the subgradient method with the same penalty and the ridge prox with the same method
each produce none. That rules out both wrong explanations at once.

The rate group asserts the subgradient exponent against ``-1/2`` and asserts that ISTA and FISTA
do **not** show their quoted rates on these instances, which is the third time in this part that a
worst case bound over a class has been separated from a measurement on an instance.

The FISTA group asserts non-monotonicity on purpose. It is a property of the method and it breaks
any stopping test built on the objective, so a later reader tempted to "fix" it has to get past a
test that says it is not a fault.
"""
import functools
import math

import numpy as np
import pytest

from nalib import proximal as px


@functools.lru_cache(maxsize=None)
def zeros():
    return px.only_soft_thresholding_makes_exact_zeros()


@functools.lru_cache(maxsize=None)
def threshold():
    return px.the_threshold_where_everything_vanishes()


@functools.lru_cache(maxsize=None)
def path():
    return px.the_regularization_path()


@functools.lru_cache(maxsize=None)
def rough():
    return px.subgradient_is_slower_and_never_sparse()


@functools.lru_cache(maxsize=None)
def accelerated():
    return px.fista_is_faster_and_not_monotone()


@functools.lru_cache(maxsize=None)
def rates():
    return px.the_quoted_rates_are_worst_case()


@functools.lru_cache(maxsize=None)
def recovery():
    return px.recovery_needs_enough_samples()


# ------------------------------------------------------------------ the operators


def test_the_prox_of_an_indicator_is_exactly_a_projection():
    out = px.the_prox_of_an_indicator_is_a_projection()
    assert out["they_are_the_same_operator"]
    assert out["worst_difference"] == 0.0


def test_every_prox_is_firmly_nonexpansive():
    out = px.every_prox_is_firmly_nonexpansive()
    assert out["all_nonexpansive"]
    assert out["worst_ratio_anywhere"] <= 1.0 + 1e-12


@pytest.mark.parametrize("size", [1, 3, 12, 50])
def test_soft_thresholding_zeroes_what_it_should(size):
    rng = np.random.default_rng(42)
    point = rng.normal(scale=2.0, size=size)
    amount = 1.0
    out = px.soft_threshold(point, amount)
    small = np.abs(point) <= amount
    assert np.all(out[small] == 0.0)
    kept = out[~small]
    if kept.size:                       # a small sample can be entirely below the threshold
        assert np.all(np.abs(kept) > 0.0)
        assert float(np.max(np.abs(np.abs(kept)
                                   - (np.abs(point[~small]) - amount)))) < 1e-15


@pytest.mark.parametrize("size", [1, 4, 20])
def test_the_ridge_prox_never_reaches_zero(size):
    rng = np.random.default_rng(7)
    point = rng.normal(size=size) + 1e-3
    out = px.prox_ridge(point, 1000.0)
    assert np.all(out != 0.0)
    assert float(np.max(np.abs(out))) < float(np.max(np.abs(point)))


def test_the_group_prox_removes_whole_groups():
    point = np.array([0.1, 0.1, 5.0, 5.0])
    out = px.prox_group(point, 1.0, [[0, 1], [2, 3]])
    assert out[0] == 0.0 and out[1] == 0.0
    assert out[2] > 0.0 and out[3] > 0.0


def test_a_zero_amount_leaves_everything_alone():
    rng = np.random.default_rng(1)
    point = rng.normal(size=9)
    assert float(np.max(np.abs(px.soft_threshold(point, 0.0) - point))) == 0.0
    assert float(np.max(np.abs(px.prox_ridge(point, 0.0) - point))) == 0.0


# ------------------------------------------------------------------ sparsity


def test_only_the_l1_prox_produces_exact_zeros():
    out = zeros()
    assert out["only_the_l1_prox_is_sparse"]
    rows = {r["method"]: r for r in out["rows"]}
    assert rows["FISTA on the L1 objective"]["nonzeros"] < out["variables"]
    assert rows["subgradient on the L1 objective"]["nonzeros"] == out["variables"]
    assert rows["ridge penalty, proximal"]["nonzeros"] == out["variables"]


def test_the_closed_form_threshold_is_exact():
    out = threshold()
    assert out["the_formula_is_exact"]
    assert out["everything_above_is_zero"]
    assert out["everything_below_is_not"]


def test_the_support_grows_as_the_penalty_falls():
    out = path()
    assert out["the_support_grows"]
    assert out["true_ones_found_there"] == out["true_nonzeros"]


def test_the_path_starts_empty_and_ends_full():
    rows = path()["rows"]
    assert rows[0]["nonzeros"] == 0
    assert rows[-1]["nonzeros"] > path()["true_nonzeros"]


@pytest.mark.parametrize("sparsity", [0, 1, 5])
def test_the_problem_accepts_any_sparsity(sparsity):
    problem = px.lasso_problem(samples=40, variables=30, sparsity=sparsity)
    assert int(np.count_nonzero(problem["truth"])) == sparsity
    assert problem["lambda_max"] > 0.0


def test_the_problem_rejects_impossible_sizes():
    with pytest.raises(ValueError):
        px.lasso_problem(samples=0)
    with pytest.raises(ValueError):
        px.lasso_problem(variables=10, sparsity=11)


# ------------------------------------------------------------------ rates


def test_the_subgradient_rate_is_one_over_root_k():
    out = rough()
    assert out["the_exponent_is_minus_a_half"]
    assert abs(out["fitted_exponent"] + 0.5) < 0.15


def test_the_subgradient_iterate_is_never_sparse():
    out = rough()
    assert out["never_sparse"]
    assert out["nonzeros"] == out["variables"]


def test_the_subgradient_objective_is_not_monotone():
    assert rough()["raw_history_is_not_monotone"]


def test_the_quoted_composite_rates_are_not_what_happens_here():
    out = rates()
    assert out["neither_late_exponent_is_the_quoted_one"]
    assert out["the_late_exponents_are_steeper"]


def test_fista_settles_its_support_much_sooner():
    rows = {r["method"]: r for r in rates()["rows"]}
    assert rows["FISTA"]["support_settles_at"] < rows["ISTA"]["support_settles_at"]


# ------------------------------------------------------------------ FISTA


def test_fista_reaches_the_floor_in_fewer_steps():
    out = accelerated()
    assert out["speedup_in_steps"] > 1.5


def test_and_by_a_large_margin_at_a_fixed_budget():
    assert accelerated()["accuracy_gain_at_1000"] > 1e6


def test_ista_is_monotone_and_fista_is_not():
    # not a defect: acceleration gives up monotonicity, and a stopping test has to know
    out = accelerated()
    assert out["ista_is_monotone"]
    assert out["fista_is_not"]


def test_they_reach_the_same_support():
    assert accelerated()["they_agree_on_the_support"]


@pytest.mark.parametrize("accelerate", [False, True])
@pytest.mark.parametrize("variables", [5, 40])
def test_proximal_gradient_runs_at_any_size(accelerate, variables):
    problem = px.lasso_problem(samples=30, variables=variables, sparsity=3)
    lam = 0.1 * problem["lambda_max"]
    out = px.proximal_gradient(problem, lam, accelerated=accelerate, rounds=2000)
    assert out["x"].shape == (variables,)
    assert out["objective"] <= px.objective(problem, problem["start"], lam) + 1e-12


def test_a_penalty_above_the_maximum_gives_the_zero_vector():
    problem = px.lasso_problem(samples=40, variables=60, sparsity=4)
    out = px.proximal_gradient(problem, 2.0 * problem["lambda_max"],
                               accelerated=True, rounds=2000, record=False)
    assert out["nonzeros"] == 0


# ------------------------------------------------------------------ recovery


def test_the_sample_count_rule_separates_the_two_regimes():
    out = recovery()
    assert out["the_rule_separates_them"]
    assert out["worst_fraction_below_the_rule"] < 0.5
    assert out["best_fraction_above_the_rule"] > 0.8


def test_the_false_positives_fall_with_more_samples():
    out = recovery()
    assert out["false_positives_fall_with_samples"]
    assert out["rows"][-1]["false_ones"] == 0


def test_the_best_is_not_at_the_largest_sample_count():
    # the penalty is a fixed multiple of lambda_max, which is not the right normalization
    out = recovery()
    assert out["samples_at_the_best"] < max(r["samples"] for r in out["rows"])
