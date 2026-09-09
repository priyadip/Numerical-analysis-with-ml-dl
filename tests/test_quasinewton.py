"""Tests for nalib.quasinewton.

Four groups. The BFGS group asserts the secant condition as an identity, to rounding, and asserts
positive curvature separately, because one is a property of the algebra and the other a property of
the line search: a single combined assertion would hide which had broken.

The rate group asserts that BFGS is **not** quadratic and that whether it is superlinear cannot be
settled in double precision, which is the third time in this part that a fast method outruns the
precision before its rate can be fitted. A later reader who "fixes" that has to break a test saying
it is not fixable here.

The trust region group asserts the escape from the saddle that lesson 86's plain Newton fell into,
and asserts as well that the trust region makes two other sizes worse, because reporting only the
rescue would be the easy half of the truth.
"""
import functools
import math

import numpy as np
import pytest

from nalib import quasinewton as qn
from nalib.optimize import quadratic, rosenbrock


@functools.lru_cache(maxsize=None)
def secant():
    return qn.the_secant_condition_holds_and_needs_positive_curvature()


@functools.lru_cache(maxsize=None)
def rate():
    return qn.superlinear_but_not_quadratic()


@functools.lru_cache(maxsize=None)
def steps():
    return qn.the_n_step_property_needs_an_exact_line_search()


@functools.lru_cache(maxsize=None)
def searches():
    return qn.the_line_search_decides_the_rate()


@functools.lru_cache(maxsize=None)
def memory():
    return qn.limited_memory_costs_little()


@functools.lru_cache(maxsize=None)
def region():
    return qn.a_trust_region_escapes_the_saddle()


@functools.lru_cache(maxsize=None)
def betas():
    return qn.which_beta_for_nonlinear_cg()


@functools.lru_cache(maxsize=None)
def cost():
    return qn.the_cost_of_curvature()


# ------------------------------------------------------------------ BFGS


def test_the_secant_condition_is_an_identity():
    out = secant()
    assert out["the_secant_condition_holds"]
    assert out["worst_secant_residual"] < 1e-8


def test_the_curvature_stays_positive_which_is_the_wolfe_condition():
    out = secant()
    assert out["every_curvature_positive"]


def test_the_update_is_skipped_rather_than_applied_when_the_curvature_is_tiny():
    out = secant()
    assert out["skipped_updates"] == out["curvatures_below_the_update_threshold"]


@pytest.mark.parametrize("dimension", [1, 2, 4, 8])
def test_bfgs_converges_on_a_quadratic_at_any_size(dimension):
    problem = quadratic(condition=100.0, dimension=dimension)
    out = qn.bfgs(problem, tol=1e-9)
    assert out["converged"]
    assert out["x"].shape == (dimension,)
    assert float(np.linalg.norm(out["x"] - problem["minimizers"][0])) < 1e-6


@pytest.mark.parametrize("dimension", [2, 5])
def test_bfgs_and_lbfgs_reach_the_same_answer(dimension):
    problem = rosenbrock(dimension)
    full = qn.bfgs(problem, tol=1e-8)
    limited = qn.lbfgs(problem, memory=10, tol=1e-8)
    assert full["converged"] and limited["converged"]
    assert float(np.linalg.norm(full["x"] - limited["x"])) < 1e-5


def test_lbfgs_needs_at_least_one_stored_pair():
    with pytest.raises(ValueError):
        qn.lbfgs(rosenbrock(2), memory=0)


# ------------------------------------------------------------------ the rate


def test_bfgs_is_not_quadratic():
    out = rate()
    assert out["square_ratios_grow"]
    assert out["square_ratio_growth"] > 1e4


def test_newtons_constant_stays_where_lesson_86_measured_it():
    out = rate()
    assert 50.0 < out["newton_square_constant"] < 500.0


def test_the_per_step_ratio_is_far_below_one():
    out = rate()
    assert out["ratios_are_far_below_one"]
    assert out["biggest_linear_ratio"] < 0.1


def test_whether_it_is_superlinear_cannot_be_settled_here():
    # asserting the limitation, so nobody later reports a fitted rate from six points
    out = rate()
    assert out["whether_it_tends_to_zero_is_not_resolvable"]
    assert out["usable_points"] <= 8


def test_the_n_step_property_needs_a_tight_line_search():
    out = steps()
    assert out["tightening_never_costs_steps"]
    assert out["tightest_is_within_three_of_n"]


def test_the_loose_search_costs_more_than_n_steps():
    out = steps()
    biggest = max(out["rows"], key=lambda r: r["n"])
    assert biggest["c2=0.9"] > biggest["n"]


def test_a_tighter_line_search_builds_a_better_curvature_model():
    out = searches()
    assert out["tightening_improves_the_model"]
    assert out["tightening_reduces_the_steps"]
    assert out["model_error_improvement"] > 2.0


def test_the_line_search_can_change_which_minimum_is_found():
    assert searches()["someone_found_a_different_minimum"]


# ------------------------------------------------------------------ limited memory


def test_a_few_pairs_are_nearly_as_good_as_the_whole_matrix():
    out = memory()
    assert out["spread_above_five_pairs"] < 1.5
    assert out["cheapest_that_beats_full"] is not None


def test_one_pair_is_not_enough():
    out = memory()
    one = [r for r in out["rows"] if r["memory"] == 1][0]
    best = min(r["steps"] for r in out["rows"])
    assert one["steps"] > 3 * best


def test_the_storage_saving_is_real_only_when_the_memory_is_small():
    out = memory()
    assert out["storage_saving_at_five"] > 5.0
    for row in out["rows"]:
        assert row["stored_numbers"] == 2 * row["memory"] * out["variables"]


# ------------------------------------------------------------------ trust region


def test_the_trust_region_escapes_the_saddle():
    out = region()
    assert out["newton_saddles"]
    assert out["it_escapes_every_saddle"]


def test_and_it_rescues_some_sizes_and_spoils_others():
    out = region()
    assert out["rescued"]
    rows = out["rows"]
    spoiled = [r for r in rows
               if r["newton_distance"] < 1e-6 and r["trust_distance"] > 1e-6]
    assert spoiled


def test_negative_curvature_is_used_rather_than_repaired():
    out = region()
    kinds = set()
    for row in out["rows"]:
        kinds.update(row["step_kinds"])
    assert "cauchy" in kinds or "edge" in kinds


def test_the_dogleg_takes_the_newton_step_when_it_fits():
    matrix = np.diag([1.0, 2.0])
    slope = np.array([0.1, 0.1])
    out = qn.dogleg(slope, matrix, radius=10.0)
    assert out["kind"] == "newton"
    assert float(np.linalg.norm(out["step"] + np.linalg.solve(matrix, slope))) < 1e-12


def test_the_dogleg_goes_to_the_edge_on_negative_curvature():
    matrix = np.diag([-1.0, -2.0])
    out = qn.dogleg(np.array([1.0, 0.0]), matrix, radius=0.5)
    assert out["kind"] == "edge"
    assert abs(float(np.linalg.norm(out["step"])) - 0.5) < 1e-12


@pytest.mark.parametrize("dimension", [1, 2, 4])
def test_the_trust_region_solves_a_quadratic_at_any_size(dimension):
    problem = quadratic(condition=100.0, dimension=dimension)
    out = qn.trust_region(problem, tol=1e-9)
    assert out["converged"]
    assert float(np.linalg.norm(out["x"] - problem["minimizers"][0])) < 1e-6


# ------------------------------------------------------------------ nonlinear CG


def test_polak_ribiere_plus_beats_fletcher_reeves():
    out = betas()
    assert out["best_rule"] == "polak-ribiere-plus"
    assert (out["by_rule"]["polak-ribiere-plus"]["total_steps"]
            < out["by_rule"]["fletcher-reeves"]["total_steps"])


def test_fletcher_reeves_fails_to_converge_somewhere():
    out = betas()
    assert out["by_rule"]["fletcher-reeves"]["converged"] < len(
        {r["variables"] for r in out["rows"]})


def test_an_unknown_beta_rule_is_refused():
    with pytest.raises(ValueError):
        qn.nonlinear_cg(rosenbrock(2), rule="made up")


@pytest.mark.parametrize("rule", ["fletcher-reeves", "polak-ribiere",
                                  "polak-ribiere-plus"])
def test_every_rule_solves_a_quadratic(rule):
    problem = quadratic(condition=50.0, dimension=4)
    out = qn.nonlinear_cg(problem, rule=rule, tol=1e-9, max_steps=5000)
    assert out["converged"]
    assert float(np.linalg.norm(out["x"] - problem["minimizers"][0])) < 1e-6


# ------------------------------------------------------------------ cost


def test_bfgs_is_cheaper_than_newton_once_the_hessian_costs_enough():
    out = cost()
    assert out["bfgs_is_cheaper_somewhere"]
    assert out["sizes_where_bfgs_is_cheaper"]


def test_and_newton_is_cheaper_at_the_smallest_size():
    out = cost()
    smallest = min(out["rows"], key=lambda r: r["variables"])
    assert smallest["bfgs_over_newton"] > 1.0


def test_the_advantage_grows_with_the_dimension():
    rows = sorted(cost()["rows"], key=lambda r: r["variables"])
    ratios = [r["bfgs_over_newton"] for r in rows]
    assert ratios == sorted(ratios, reverse=True)


# ------------------------------------------------------------------ the cluster


@functools.lru_cache(maxsize=None)
def cluster():
    return qn.a_cluster_has_many_minima()


@pytest.mark.parametrize("atoms", [2, 3, 5])
def test_the_cluster_gradient_matches_finite_differences(atoms):
    from nalib.optimize import gradient as by_differences
    problem = qn.lennard_jones(atoms)
    rng = np.random.default_rng(42)
    where = problem["start"] + 0.1 * rng.normal(size=problem["dimension"])
    exact = np.asarray(problem["gradient"](where), dtype=float)
    numeric = by_differences(problem["f"], where)
    scale = max(float(np.max(np.abs(exact))), 1.0)
    assert float(np.max(np.abs(exact - numeric))) / scale < 1e-5


def test_the_cluster_needs_at_least_two_atoms():
    with pytest.raises(ValueError):
        qn.lennard_jones(1)


def test_the_lowest_energy_found_is_the_published_one():
    assert cluster()["matches_the_published_values"]


def test_the_same_problem_has_more_minima_as_it_grows():
    out = cluster()
    assert out["the_count_grows_with_the_atoms"]
    assert max(r["distinct_energies"] for r in out["rows"]) > 2


def test_and_the_best_one_gets_harder_to_find():
    out = cluster()
    rows = sorted(out["rows"], key=lambda r: r["atoms"])
    assert rows[-1]["how_often_the_best_was_found"] < rows[0]["how_often_the_best_was_found"]
    assert out["worst_success_rate"] < 0.5
