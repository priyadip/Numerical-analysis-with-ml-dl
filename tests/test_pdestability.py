"""Tests for nalib.pdestability.

Four groups, one per property plus the theorem. Consistency is checked against a closed form for
the truncation constant, which is a much sharper test than fitting an order: an order fit tolerates
a wrong constant, and the constant is what separates the members of the family. Stability is
checked two ways against each other, and the case where they agree exactly is asserted to agree
exactly rather than approximately, because that is a theorem about symmetric matrices and not a
numerical accident. The Lax group asserts the two failing combinations fail, since a test on the
working combination alone proves nothing.

The test worth the most here is the one on the inconsistent scheme: it is stable, it runs, it
produces a smooth answer, and only a comparison against the exact solution shows it is wrong.
"""
import math

import numpy as np
import pytest

from nalib import parabolic as pb
from nalib import pdestability as ps

THETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
UNKNOWNS = [5, 9, 17]


# --------------------------------------------------------------------------- consistency


@pytest.mark.parametrize("theta", THETAS)
def test_every_member_of_the_family_is_second_order_in_space(theta):
    out = ps.truncation_order(pb.sine_problem(), theta=theta, ratio=0.4)
    if abs((0.5 - theta) * 0.4 - 1.0 / 12.0) < 1e-12:
        pytest.skip("this theta cancels the leading term at this ratio")
    assert out["order_in_h"] == pytest.approx(2.0, abs=0.2)
    assert out["consistent"]


def test_the_truncation_constant_matches_its_closed_form():
    out = ps.consistency_of_the_family()
    assert out["the_formula_predicts_every_constant"]
    assert out["every_member_is_consistent"]


def test_the_leading_term_cancels_at_one_theta_and_the_order_jumps():
    out = ps.consistency_of_the_family(ratio=0.4)
    assert out["cancelling_theta"] == pytest.approx(0.5 - 1.0 / (12.0 * 0.4))
    assert out["order_at_the_cancelling_theta"] == pytest.approx(4.0, abs=0.2)


def test_crank_nicolson_is_not_the_most_accurate_member_at_a_fixed_ratio():
    out = ps.consistency_of_the_family(ratio=0.4)
    assert out["crank_nicolson_is_not_the_most_accurate_at_this_ratio"]
    assert out["crank_nicolson_over_the_best"] > 2.0


@pytest.mark.parametrize("ratio", [0.2, 0.3, 0.5])
def test_the_cancelling_theta_moves_with_the_ratio(ratio):
    out = ps.consistency_of_the_family(ratio=ratio)
    assert out["cancelling_theta"] == pytest.approx(0.5 - 1.0 / (12.0 * ratio))
    assert out["order_at_the_cancelling_theta"] > 3.5


def test_the_truncation_residual_of_the_exact_solution_shrinks_with_the_grid():
    problem = pb.sine_problem()
    previous = None
    for n in (11, 21, 41, 81):
        h = 1.0 / (n - 1)
        k = 0.4 * h ** 2
        steps = max(int(round(0.05 / k)), 1)
        out = ps.truncation_residual(problem, n, steps, steps * k, theta=0.0)
        if previous is not None:
            assert out["residual"] < previous
        previous = out["residual"]


# --------------------------------------------------------------------------- the matrix method


@pytest.mark.parametrize("m", UNKNOWNS)
@pytest.mark.parametrize("theta", THETAS)
def test_the_amplification_matrix_reproduces_one_step_of_the_scheme(theta, m):
    rng = np.random.default_rng(42)
    inner = rng.normal(size=m)
    padded = np.concatenate([[0.0], inner, [0.0]])
    r = 0.37
    stepped = pb.theta_step(padded, r, theta)
    matrix = ps.amplification_matrix(theta, r, m)
    assert np.allclose(matrix @ inner, stepped[1:-1], atol=1e-11)


@pytest.mark.parametrize("m", UNKNOWNS)
def test_the_matrix_is_symmetric_for_the_heat_equation(m):
    for theta in THETAS:
        g = ps.amplification_matrix(theta, 0.4, m)
        assert np.allclose(g, g.T, atol=1e-12)


def test_the_two_stability_tests_are_the_same_statement_here():
    out = ps.the_two_tests_agree_on_the_heat_equation()
    assert out["the_two_tests_are_the_same_statement"]
    assert out["every_matrix_is_normal"]
    assert out["radius_equals_norm_everywhere"]
    assert out["worst_eigenvalue_gap"] < 1e-10
    assert out["cases"] == 48


def test_both_tests_agree_with_the_known_stability_limit():
    out = ps.the_two_tests_agree_on_the_heat_equation()
    assert out["and_they_both_say_the_same_about_stability"]


@pytest.mark.parametrize("m", UNKNOWNS)
def test_the_grid_carries_exactly_the_phases_the_symbols_use(m):
    symbols = ps.von_neumann_symbols(0.0, 0.4, m)
    assert symbols.size == m
    values = np.sort(np.real(np.linalg.eigvals(ps.amplification_matrix(0.0, 0.4, m))))
    assert np.allclose(values, np.sort(symbols), atol=1e-12)


def test_an_empty_grid_is_rejected():
    with pytest.raises(ValueError):
        ps.amplification_matrix(0.5, 0.4, 0)


# --------------------------------------------------------------------------- non-normality


@pytest.mark.parametrize("peclet,r,expected", [(0.5, 0.4, True), (2.0, 0.45, True),
                                               (2.0, 0.6, False), (8.0, 0.1, False)])
def test_the_convection_stability_conditions(peclet, r, expected):
    assert ps.convection_symbol_is_stable(peclet, r) == expected


def test_convection_makes_the_matrix_not_normal():
    plain = ps.convection_diffusion_matrix(0.0, 0.4, 20)
    convected = ps.convection_diffusion_matrix(1.9, 0.4, 20)
    assert np.allclose(plain, plain.T, atol=1e-12)
    assert not np.allclose(convected, convected.T, atol=1e-12)


def test_nothing_grows_anywhere_in_the_stable_region():
    out = ps.the_spectral_radius_is_only_the_limit()
    assert out["nothing_grows_anywhere"]
    assert out["every_run_is_monotone"]


def test_a_normal_matrix_decays_at_its_spectral_radius_from_the_first_step():
    out = ps.the_spectral_radius_is_only_the_limit()
    assert out["the_normal_one_hits_its_rate_at_once"]


def test_a_non_normal_one_takes_hundreds_of_steps_to_get_there():
    out = ps.the_spectral_radius_is_only_the_limit()
    assert out["the_others_take_longer"]
    assert out["worst_delay"] > 100
    assert out["some_never_get_there"]


def test_an_unstable_pair_is_refused():
    with pytest.raises(ValueError):
        ps.the_spectral_radius_is_only_the_limit(cases=((8.0, 0.4),))


def test_a_one_point_convection_grid_is_rejected():
    with pytest.raises(ValueError):
        ps.convection_diffusion_matrix(1.0, 0.4, 1)


# --------------------------------------------------------------------------- Lax


def test_only_the_consistent_and_stable_scheme_converges():
    out = ps.lax_in_three_runs()
    assert out["only_the_first_converges"]


def test_the_unstable_scheme_gets_worse_as_the_grid_is_refined():
    out = ps.lax_in_three_runs()
    unstable = out["cases"][1]
    assert unstable["consistent"]
    assert not unstable["stable"]
    assert unstable["gets_worse_when_refined"]


def test_the_inconsistent_scheme_is_stable_and_settles_on_a_wrong_answer():
    out = ps.lax_in_three_runs()
    wrong = out["cases"][2]
    assert wrong["stable"]
    assert not wrong["consistent"]
    assert wrong["settles_on"] > 1e-3
    assert bool(np.all(np.isfinite(wrong["error"])))


def test_the_converging_case_actually_converges_at_the_right_rate():
    out = ps.lax_in_three_runs()
    good = out["cases"][0]
    ratios = good["error"][:-1] / good["error"][1:]
    assert bool(np.all(ratios > 1.7))


# --------------------------------------------------------------------------- designing one


def test_the_consistency_conditions_are_met_by_design():
    out = ps.build_your_own_scheme()
    assert out["the_conditions_are_linear_and_are_met_by_four_of_five"]
    assert out["every_scheme_is_solvable"]


def test_consistency_and_stability_are_independent():
    out = ps.build_your_own_scheme()
    assert out["consistency_and_stability_are_independent"]
    assert out["a_consistent_scheme_can_be_unstable"]
    assert out["an_inconsistent_scheme_can_be_stable"]


def test_the_wrong_second_moment_solves_a_different_equation():
    out = ps.build_your_own_scheme()
    wrong = next(row for row in out["rows"] if not row["consistent"])
    assert wrong["stable"]
    assert not wrong["residual_falls"]
    assert wrong["second_moment_condition"] != pytest.approx(
        wrong["second_moment_should_be"])


def test_the_designed_fourth_order_member_reaches_fourth_order():
    out = ps.build_your_own_scheme(r=0.25)
    designed = next(row for row in out["rows"] if "theta = " in row["scheme"])
    assert designed["consistent"]
    assert designed["stable"]
    assert designed["measured_order"] == pytest.approx(4.0, abs=0.2)


@pytest.mark.parametrize("theta", THETAS)
def test_the_symbol_of_the_family_matches_the_closed_form(theta):
    r = 0.37
    weights = (np.asarray([(1.0 - theta) * r, 1.0 - 2.0 * (1.0 - theta) * r,
                           (1.0 - theta) * r]),
               np.asarray([-theta * r, 1.0 + 2.0 * theta * r, -theta * r]))
    phases = np.linspace(0.0, math.pi, 51)
    top, bottom = ps.symbol_of(weights, phases)
    assert np.allclose(top / bottom, pb.growth_factor(r, phases, theta), atol=1e-12)


def test_weight_rows_of_different_or_even_length_are_rejected():
    with pytest.raises(ValueError):
        ps.symbol_of((np.ones(3), np.ones(5)), np.zeros(3))
    with pytest.raises(ValueError):
        ps.symbol_of((np.ones(4), np.ones(4)), np.zeros(3))
