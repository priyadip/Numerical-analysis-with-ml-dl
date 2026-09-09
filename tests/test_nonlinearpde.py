"""Tests for nalib.nonlinearpde.

Four groups. The residual and Jacobian are checked against each other by finite differences, which
is the only test that catches a Jacobian that is merely close. Newton's convergence is checked in
its second phase only, and the first phase is asserted to be the non quadratic one, since fitting
the whole history reads 1.75 and a careless test would call that a failure. The Bratu group asserts
the fold from two independent directions, a closed form and a bisection, and asserts that the gap
between them is the discretization error by fitting its order. The method of lines group asserts an
exact zero, that Euler on the semi-discrete system is bit for bit lesson 78's scheme.

The test worth the most here is the error decomposition: RK4 is more accurate in time and less
accurate overall, and asserting both stops a later reader from "fixing" the wrong one.
"""
import functools
import math

import numpy as np
import pytest

from nalib import nonlinearpde as nl


@functools.lru_cache(maxsize=None)
def fold():
    return nl.the_bratu_problem_has_a_fold()


@functools.lru_cache(maxsize=None)
def newton():
    return nl.newton_is_quadratic()


@functools.lru_cache(maxsize=None)
def limit():
    return nl.the_step_limit_comes_from_the_ode_side()


@functools.lru_cache(maxsize=None)
def adaptive():
    return nl.an_adaptive_solver_finds_the_limit_by_itself()


@functools.lru_cache(maxsize=None)
def lines():
    return nl.lines_against_a_direct_scheme()


@functools.lru_cache(maxsize=None)
def fisher():
    return nl.a_nonlinear_time_dependent_problem()


# --------------------------------------------------------------------------- residual


@pytest.mark.parametrize("n", [11, 21, 41])
@pytest.mark.parametrize("weight", [1.0, 10.0])
def test_the_jacobian_is_the_derivative_of_the_residual(n, weight):
    problem = nl.cubic_problem(weight=weight)
    pieces = nl.residual_and_jacobian(problem, n)
    rng = np.random.default_rng(42)
    u = rng.normal(size=pieces["unknowns"])
    analytic = pieces["jacobian"](u)
    step = 1e-6
    numeric = np.empty_like(analytic)
    for j in range(u.size):
        bump = u.copy()
        bump[j] += step
        drop = u.copy()
        drop[j] -= step
        numeric[:, j] = (pieces["residual"](bump) - pieces["residual"](drop)) / (2.0 * step)
    scale = float(np.max(np.abs(analytic)))
    assert float(np.max(np.abs(analytic - numeric))) < 1e-5 * scale


@pytest.mark.parametrize("n", [11, 21, 41])
def test_the_jacobian_is_tridiagonal(n):
    pieces = nl.residual_and_jacobian(nl.cubic_problem(), n)
    j = pieces["jacobian"](np.ones(pieces["unknowns"]))
    for row in range(j.shape[0]):
        for col in range(j.shape[1]):
            if abs(row - col) > 1:
                assert j[row, col] == 0.0


def test_the_residual_vanishes_at_the_exact_solution_to_the_discretization_error():
    problem = nl.cubic_problem()
    previous = None
    for n in (21, 41, 81):
        pieces = nl.residual_and_jacobian(problem, n)
        value = float(np.max(np.abs(
            pieces["residual"](problem["exact"](pieces["interior"])))))
        if previous is not None:
            assert value < 0.4 * previous
        previous = value


def test_a_grid_or_guess_of_the_wrong_size_is_rejected():
    with pytest.raises(ValueError):
        nl.residual_and_jacobian(nl.cubic_problem(), 2)
    with pytest.raises(ValueError):
        nl.solve_nonlinear(nl.cubic_problem(), 11, guess=np.zeros(3))
    with pytest.raises(ValueError):
        nl.cubic_problem(a=1.0, b=0.0)


@pytest.mark.parametrize("n", [21, 41, 81])
def test_the_nonlinear_solve_converges_to_the_exact_solution(n):
    run = nl.solve_nonlinear(nl.cubic_problem(), n)
    assert run["converged"]
    assert run["final_residual"] < 1e-8
    assert run["error"] < 5.0 / n ** 2


# --------------------------------------------------------------------------- Newton


def test_newton_is_quadratic_in_its_second_phase():
    out = newton()
    assert out["quadratic"]
    assert out["exponent"] == pytest.approx(2.0, abs=0.15)


def test_the_whole_history_is_not_quadratic_and_says_so():
    out = newton()
    assert out["the_residual_rises_before_it_falls"]
    assert out["exponent_over_the_whole_history"] < 1.9


def test_the_jacobian_is_mostly_zeros():
    out = newton()
    assert out["sparsity"] > 0.9
    assert out["jacobian_nonzeros"] == 3 * int(math.isqrt(out["jacobian_entries"])) - 2


# --------------------------------------------------------------------------- Bratu


def test_the_closed_form_fold_is_the_published_value():
    assert nl.bratu_critical_value() == pytest.approx(3.513830719, abs=1e-8)


def test_the_measured_fold_agrees_with_the_closed_form():
    out = fold()
    assert out["agrees_to_four_digits"]
    assert out["solvable_below"]
    assert out["unsolvable_above"]


def test_the_gap_to_the_closed_form_is_the_discretization_error():
    out = fold()
    assert out["the_gap_is_the_discretization_error"]
    assert out["gap_order_in_h"] == pytest.approx(2.0, abs=0.15)


def test_the_measured_fold_rises_towards_the_closed_form_as_the_grid_refines():
    out = fold()
    values = [value for _points, value in out["refinement"]]
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))
    assert values[-1] < out["closed_form_fold"]


def test_there_are_two_branches_below_the_fold():
    out = fold()
    assert out["two_branches_found"]
    assert out["upper_peak"] > out["lower_peak"] + 0.5


@pytest.mark.parametrize("lam", [1.0, 2.0, 3.0])
def test_the_lower_branch_solves_its_own_equation(lam):
    problem = nl.bratu_problem(lam)
    run = nl.solve_nonlinear(problem, 81, tol=1e-12)
    assert run["converged"]
    assert run["final_residual"] < 1e-8
    assert float(np.max(run["u"])) > 0.0


# --------------------------------------------------------------------------- method of lines


@pytest.mark.parametrize("n", [11, 21, 41])
def test_the_semi_discrete_eigenvalues_match_the_matrix(n):
    system = nl.semi_discrete_heat(n)
    computed = np.sort(np.linalg.eigvalsh(system["matrix"]))
    assert np.allclose(computed, system["eigenvalues"], rtol=1e-12)
    assert bool(np.all(system["eigenvalues"] < 0.0))


@pytest.mark.parametrize("n", [11, 21, 41])
def test_the_semi_discrete_right_hand_side_is_the_matrix(n):
    system = nl.semi_discrete_heat(n)
    rng = np.random.default_rng(42)
    u = rng.normal(size=system["unknowns"])
    assert np.allclose(system["rhs"](0.0, u), system["matrix"] @ u)


def test_a_grid_too_small_is_rejected():
    with pytest.raises(ValueError):
        nl.semi_discrete_heat(2)


def test_the_ode_derivation_reproduces_the_mesh_ratio_limit():
    out = limit()
    assert out["the_two_derivations_agree"]
    assert out["the_ode_side_allows_slightly_more"]
    assert out["the_gap_is_second_order"]


def test_the_semi_discrete_problem_gets_stiffer_as_the_grid_refines():
    out = limit()
    assert out["stiffness_grows"]
    assert out["rows"][-1]["stiffness_ratio"] > out["rows"][0]["stiffness_ratio"]


def test_the_adaptive_solver_is_held_by_stability_not_accuracy():
    out = adaptive()
    assert out["the_tolerance_barely_matters"]
    assert out["it_lands_near_the_stability_limit"]
    assert out["tolerance_span"] >= 1e6
    assert out["step_spread"] < 2.0


def test_the_dormand_prince_reach_is_longer_than_eulers():
    assert nl._real_axis_limit("dormand prince") > 2.0
    assert nl._real_axis_limit("dormand prince") < 4.0


def test_the_lines_with_euler_is_the_explicit_scheme():
    out = lines()
    assert out["identical_to_rounding"]
    assert out["they_are_the_same_scheme"] == 0.0


def test_rk4_removes_the_time_error_and_the_cancellation_with_it():
    out = lines()
    assert out["rk4_beats_euler_in_time_by"] > 1e6
    assert out["rk4_total_over_euler_total"] > 1.5
    assert out["the_two_errors_have_opposite_signs"]
    assert out["rk4_is_the_space_error"]


def test_the_semi_discrete_rate_is_slower_than_the_continuous_one():
    out = lines()
    assert out["semi_discrete_rate"] > out["continuous_rate"]


# --------------------------------------------------------------------------- Fisher


@pytest.mark.parametrize("d,r,a,expected", [(1e-2, 1.0, 2.0, 0.52),
                                            (1e-2, 1.0, 5.0, 0.25),
                                            (1e-2, 1.0, 20.0, 0.2),
                                            (1e-2, 1.0, 50.0, 0.2)])
def test_the_front_speed_formula_switches_at_the_critical_decay(d, r, a, expected):
    assert nl.fisher_front_speed(d, r, a) == pytest.approx(expected)


def test_the_front_speed_formula_rejects_nonsense():
    for bad in ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)):
        with pytest.raises(ValueError):
            nl.fisher_front_speed(*bad)


def test_both_fronts_travel_at_the_speed_their_decay_rate_predicts():
    out = fisher()
    assert out["every_late_speed_matches_its_prediction"]


def test_the_textbook_minimum_is_wrong_for_a_shallow_front():
    out = fisher()
    assert out["the_minimum_is_wrong_for_a_shallow_front"]
    assert out["how_wrong_the_minimum_is"] > 1.0


def test_the_steep_front_approaches_the_minimum_slowly_and_bramson_says_how_slowly():
    out = fisher()
    assert out["the_steep_front_speeds_up_towards_the_minimum"]
    assert out["bramson_matches_every_window"]
    assert out["the_last_window_matches_to"] < 0.01


def test_the_run_stops_measuring_when_the_front_nears_the_boundary():
    out = fisher()
    shallow = next(row for row in out["rows"] if not row["steep"])
    assert shallow["left_the_domain"]
    assert shallow["last_usable_time"] < 40.0
