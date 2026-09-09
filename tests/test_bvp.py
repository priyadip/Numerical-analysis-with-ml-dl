"""Tests for nalib.bvp.

Four groups. The test problems are checked by **substitution**: the exact solution is put back
into the differential equation and into the boundary conditions, so a typo in a constant is
caught before any solver runs. The solvers are checked against those exact solutions and against
each other. The orders are checked by refinement, on the part of the sweep where the order is
actually measurable. The last group is the failures, which are the reason this lesson has two
methods in it rather than one.

The sharpest tests here are the ones that would pass for the wrong reason if written carelessly:
the shooting sensitivity against ``sinh(lam)/lam``, the condition number against the **discrete**
resonance rather than ``pi^2``, and the truncation order in the rms norm rather than the max.
"""
import math

import numpy as np
import pytest

from nalib import bvp

RATES = [1.0, 5.0, 10.0, 20.0]
COUNTS = [20, 40, 80, 160]


# --------------------------------------------------------------------------- the problems


def test_the_linear_test_problem_solves_its_own_equation():
    """Substitute the exact solution back in, by differentiating it numerically to high order."""
    problem = bvp.textbook_linear_problem()
    x = np.linspace(1.05, 1.95, 25)
    h = 1e-4
    y = problem["exact"](x)
    dy = (problem["exact"](x + h) - problem["exact"](x - h)) / (2.0 * h)
    d2 = (problem["exact"](x + h) - 2.0 * y + problem["exact"](x - h)) / h ** 2
    want = problem["p"](x) * dy + problem["q"](x) * y + problem["r"](x)
    assert float(np.max(np.abs(d2 - want))) < 1e-6


def test_the_linear_test_problem_meets_its_boundary_conditions_exactly():
    problem = bvp.textbook_linear_problem()
    assert float(problem["exact"](problem["a"])) == pytest.approx(problem["alpha"], abs=1e-14)
    assert float(problem["exact"](problem["b"])) == pytest.approx(problem["beta"], abs=1e-14)


@pytest.mark.parametrize("rate", RATES)
def test_the_exponential_problem_solves_its_own_equation(rate):
    problem = bvp.exponential_problem(rate)
    x = np.linspace(0.1, 0.9, 15)
    h = 1e-5
    y = problem["exact"](x)
    d2 = (problem["exact"](x + h) - 2.0 * y + problem["exact"](x - h)) / h ** 2
    assert np.allclose(d2, rate ** 2 * y, rtol=1e-5, atol=1e-7)
    assert float(problem["exact"](0.0)) == pytest.approx(1.0, abs=1e-14)
    assert float(problem["exact"](1.0)) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("eps", [0.1, 0.01, 0.001])
def test_the_convection_diffusion_solution_is_monotone_and_hits_its_ends(eps):
    problem = bvp.convection_diffusion_problem(eps)
    x = np.linspace(0.0, 1.0, 2001)
    y = problem["exact"](x)
    assert float(y[0]) == pytest.approx(0.0, abs=1e-14)
    assert float(y[-1]) == pytest.approx(1.0, abs=1e-14)
    assert np.all(np.diff(y) >= -1e-15)


def test_the_nonlinear_test_problem_solves_its_own_equation():
    """x^2 + 16/x satisfies y'' = (32 + 2x^3 - y y')/8 identically, which is checked here."""
    problem = bvp.textbook_nonlinear_problem()
    x = np.linspace(1.0, 3.0, 21)
    y = x ** 2 + 16.0 / x
    dy = 2.0 * x - 16.0 / x ** 2
    d2 = 2.0 + 32.0 / x ** 3
    assert np.allclose(d2, problem["f"](x, y, dy), rtol=1e-14, atol=1e-13)
    assert np.allclose(problem["exact"](x), y, rtol=1e-14)


def test_the_nonlinear_partial_derivatives_match_finite_differences():
    """A wrong Jacobian still converges, so it has to be checked directly."""
    problem = bvp.textbook_nonlinear_problem()
    rng = np.random.default_rng(42)
    for _ in range(20):
        x = float(rng.uniform(1.0, 3.0))
        y = float(rng.uniform(10.0, 20.0))
        dy = float(rng.uniform(-15.0, 5.0))
        h = 1e-6
        by_y = (problem["f"](x, y + h, dy) - problem["f"](x, y - h, dy)) / (2.0 * h)
        by_dy = (problem["f"](x, y, dy + h) - problem["f"](x, y, dy - h)) / (2.0 * h)
        assert float(problem["f_y"](x, y, dy)) == pytest.approx(float(by_y), abs=1e-8)
        assert float(problem["f_dy"](x, y, dy)) == pytest.approx(float(by_dy), abs=1e-8)


# --------------------------------------------------------------------------- shooting


def test_linear_shooting_needs_no_iteration_and_hits_both_ends():
    problem = bvp.textbook_linear_problem()
    out = bvp.linear_shooting(problem["p"], problem["q"], problem["r"], problem["a"],
                              problem["b"], problem["alpha"], problem["beta"], 200)
    assert out["iterations"] == 0
    assert float(out["y"][0]) == pytest.approx(problem["alpha"], abs=1e-14)
    assert float(out["y"][-1]) == pytest.approx(problem["beta"], abs=1e-10)
    assert float(np.max(np.abs(out["y"] - problem["exact"](out["x"])))) < 1e-10


@pytest.mark.parametrize("steps", [50, 100, 200, 400])
def test_linear_shooting_inherits_the_integrator_order(steps):
    """Shooting is as accurate as the IVP solver inside it, which for RK4 is fourth order."""
    problem = bvp.textbook_linear_problem()
    coarse = bvp.linear_shooting(problem["p"], problem["q"], problem["r"], problem["a"],
                                 problem["b"], problem["alpha"], problem["beta"], steps)
    fine = bvp.linear_shooting(problem["p"], problem["q"], problem["r"], problem["a"],
                               problem["b"], problem["alpha"], problem["beta"], 2 * steps)
    a = float(np.max(np.abs(coarse["y"] - problem["exact"](coarse["x"]))))
    b = float(np.max(np.abs(fine["y"] - problem["exact"](fine["x"]))))
    if a > 1e-12:
        assert a / max(b, 1e-300) > 8.0


def test_linear_shooting_flags_a_problem_with_no_unique_solution():
    """It does not divide by zero. It divides by the integrator's truncation error.

    At the resonance the homogeneous solution vanishes at b in exact arithmetic, and what RK4
    returns instead is its own O(h^4) error. So there is no exception to catch, and the only
    protection is the reported relative end value.
    """
    out = bvp.linear_shooting(lambda x: 0.0 * x, lambda x: -math.pi ** 2 + 0.0 * x,
                              lambda x: np.ones_like(np.asarray(x, dtype=float)),
                              0.0, 1.0, 0.0, 0.0, 2000)
    assert out["near_a_resonance"]
    assert out["relative_end"] < 1e-9
    assert float(np.max(np.abs(out["y"]))) > 1e6


def test_a_problem_away_from_a_resonance_is_not_flagged():
    problem = bvp.textbook_linear_problem()
    out = bvp.linear_shooting(problem["p"], problem["q"], problem["r"], problem["a"],
                              problem["b"], problem["alpha"], problem["beta"], 200)
    assert not out["near_a_resonance"]
    assert out["relative_end"] > 0.1


def test_refining_the_step_at_a_resonance_makes_the_answer_bigger():
    """More work, worse answer, and no complaint from anything in the run."""
    out = bvp.at_a_resonance_refining_makes_it_worse()
    assert out["answer_grows_with_refinement"]
    assert out["every_run_is_flagged"]
    assert out["end_value_order"] == pytest.approx(4.0, abs=0.2)
    assert float(out["largest_value"][-1]) > 1e4 * float(out["largest_value"][0])


def test_nonlinear_shooting_finds_the_slope_the_exact_solution_has():
    """The exact solution has y'(1) = 2 - 16 = -14, so the secant method must find -14."""
    problem = bvp.textbook_nonlinear_problem()

    def f(t, state):
        return np.asarray([state[1], problem["f"](t, state[0], state[1])])

    out = bvp.shooting(f, problem["a"], problem["b"], problem["alpha"], problem["beta"])
    assert out["converged"]
    assert out["slope"] == pytest.approx(-14.0, abs=1e-6)
    assert float(np.max(np.abs(out["y"] - problem["exact"](out["x"])))) < 1e-7


def test_the_secant_method_costs_one_solve_per_step():
    problem = bvp.textbook_nonlinear_problem()

    def f(t, state):
        return np.asarray([state[1], problem["f"](t, state[0], state[1])])

    out = bvp.shooting(f, problem["a"], problem["b"], problem["alpha"], problem["beta"])
    assert out["solves"] == out["iterations"] + 2
    assert out["iterations"] < 15


@pytest.mark.parametrize("guess", [-30.0, -5.0, 0.0, 10.0])
def test_nonlinear_shooting_reaches_the_same_answer_from_any_start(guess):
    problem = bvp.textbook_nonlinear_problem()

    def f(t, state):
        return np.asarray([state[1], problem["f"](t, state[0], state[1])])

    out = bvp.shooting(f, problem["a"], problem["b"], problem["alpha"], problem["beta"],
                       guesses=(guess, guess + 1.0))
    assert out["converged"]
    assert out["slope"] == pytest.approx(-14.0, abs=1e-5)


def test_the_miss_function_is_affine_for_a_linear_problem():
    """Which is why linear shooting needs no root finder: two points determine the whole map."""
    problem = bvp.exponential_problem(3.0)

    def f(t, state):
        return np.asarray([state[1], 9.0 * state[0]])

    values = [bvp.miss(f, 0.0, 1.0, 1.0, s, 400) for s in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    second = np.diff(np.diff(values))
    assert float(np.max(np.abs(second))) < 1e-9


# --------------------------------------------------------------------------- conditioning


def test_the_shooting_sensitivity_is_sinh_over_lambda():
    """Measured against the closed form, which is the sharpest check available."""
    out = bvp.shooting_amplifies_the_guess()
    for lam, got, want in zip(out["rate"], out["measured_sensitivity"],
                              out["closed_form_sensitivity"]):
        assert float(got) == pytest.approx(float(want), rel=1e-6), lam
        assert float(want) == pytest.approx(math.sinh(lam) / lam, rel=1e-12)


def test_the_sensitivity_grows_and_the_error_grows_with_it():
    out = bvp.shooting_amplifies_the_guess()
    assert out["grows_exponentially"]
    errors = np.asarray(out["shooting_error"], dtype=float)
    assert errors[-1] > 1.0
    assert errors[0] < 1e-12
    # the error tracks the unavoidable floor set by the sensitivity, within a factor of 100
    for got, floor in zip(errors[2:], np.asarray(out["unavoidable_slope_error"])[2:]):
        assert got > 0.1 * floor
        assert got < 200.0 * floor


def test_finite_differences_grow_only_polynomially_where_shooting_grows_exponentially():
    out = bvp.shooting_against_finite_differences()
    assert out["finite_differences_grow_polynomially"]
    assert out["shooting_grows_faster_than_any_power"]
    assert out["finite_difference_exponent_in_the_rate"] == pytest.approx(2.0, abs=0.2)


def test_shooting_is_the_better_method_on_a_mild_problem():
    """The honest half of the comparison: order beats conditioning when there is no layer."""
    out = bvp.shooting_against_finite_differences()
    assert out["shooting_wins_at_the_smallest_rate"]
    assert out["crossover_rate"] is not None
    assert out["crossover_rate"] > 10.0


def test_splitting_the_interval_recovers_what_the_amplification_lost():
    out = bvp.multiple_shooting_recovers_it()
    assert out["improves"]
    assert out["gain_from_the_first_split"] > 1e4
    assert out["enough_pieces"] <= 4
    assert bool(np.all(out["converged"]))


@pytest.mark.parametrize("pieces", [1, 2, 3, 5])
def test_multiple_shooting_solves_the_nonlinear_problem_too(pieces):
    problem = bvp.textbook_nonlinear_problem()

    def f(t, state):
        return np.asarray([state[1], problem["f"](t, state[0], state[1])])

    out = bvp.multiple_shooting(f, problem["a"], problem["b"], problem["alpha"],
                                problem["beta"], pieces)
    assert out["converged"]
    assert out["slope"] == pytest.approx(-14.0, abs=1e-5)
    assert float(np.max(np.abs(out["y"] - problem["exact"](out["x"])))) < 1e-6


def test_a_piece_count_below_one_is_refused():
    with pytest.raises(ValueError):
        bvp.multiple_shooting(lambda t, y: np.asarray([y[1], 0.0]), 0.0, 1.0, 0.0, 1.0, 0)


# --------------------------------------------------------------------------- finite differences


def test_the_finite_difference_solution_is_second_order():
    out = bvp.order_of_finite_differences()
    assert out["second_order"]
    assert np.all(np.abs(out["ratio_per_halving"] - 4.0) < 0.3)


@pytest.mark.parametrize("rate", RATES)
def test_finite_differences_are_second_order_whatever_the_growth_rate(rate):
    out = bvp.order_of_finite_differences(problem=bvp.exponential_problem(rate),
                                          counts=[40, 80, 160, 320, 640])
    assert out["fitted_order"] == pytest.approx(2.0, abs=0.2)


def test_the_solution_meets_both_boundary_conditions_exactly():
    problem = bvp.textbook_linear_problem()
    for n in COUNTS:
        out = bvp.finite_difference_linear(problem["p"], problem["q"], problem["r"],
                                           problem["a"], problem["b"], problem["alpha"],
                                           problem["beta"], n)
        assert float(out["y"][0]) == problem["alpha"]
        assert float(out["y"][-1]) == problem["beta"]
        assert out["unknowns"] == n - 1


def test_a_grid_of_fewer_than_three_points_is_refused():
    with pytest.raises(ValueError):
        bvp.finite_difference_matrix(lambda x: 0.0 * x, lambda x: 0.0 * x, [0.0, 1.0], 0.0, 1.0)


def test_a_grid_that_is_not_increasing_is_refused():
    with pytest.raises(ValueError):
        bvp.finite_difference_matrix(lambda x: 0.0 * x, lambda x: 0.0 * x,
                                     [0.0, 0.5, 0.4, 1.0], 0.0, 1.0)


def test_the_tridiagonal_solve_agrees_with_a_dense_solve():
    """Thomas has no pivoting, so it has to be checked against something that does."""
    from nalib.banded import tridiagonal_matrix

    problem = bvp.textbook_linear_problem()
    for n in (20, 50, 111):
        out = bvp.finite_difference_linear(problem["p"], problem["q"], problem["r"],
                                           problem["a"], problem["b"], problem["alpha"],
                                           problem["beta"], n)
        A = tridiagonal_matrix(out["sub"][1:], out["diag"], out["sup"][:-1])
        rhs = np.asarray(problem["r"](out["interior"]), dtype=float).copy()
        rhs[0] -= out["sub"][0] * problem["alpha"]
        rhs[-1] -= out["sup"][-1] * problem["beta"]
        dense = np.linalg.solve(A, rhs)
        assert np.allclose(dense, out["y"][1:-1], rtol=1e-10, atol=1e-12)


def test_the_nonlinear_solver_reaches_the_exact_solution():
    problem = bvp.textbook_nonlinear_problem()
    out = bvp.finite_difference_nonlinear(problem["f"], problem["f_y"], problem["f_dy"],
                                          problem["a"], problem["b"], problem["alpha"],
                                          problem["beta"], 80)
    assert out["converged"]
    assert float(np.max(np.abs(out["y"] - problem["exact"](out["x"])))) < 2e-4


@pytest.mark.parametrize("n", [20, 40, 80, 160])
def test_the_nonlinear_solver_is_second_order_too(n):
    problem = bvp.textbook_nonlinear_problem()
    coarse = bvp.finite_difference_nonlinear(problem["f"], problem["f_y"], problem["f_dy"],
                                             problem["a"], problem["b"], problem["alpha"],
                                             problem["beta"], n)
    fine = bvp.finite_difference_nonlinear(problem["f"], problem["f_y"], problem["f_dy"],
                                           problem["a"], problem["b"], problem["alpha"],
                                           problem["beta"], 2 * n)
    a = float(np.max(np.abs(coarse["y"] - problem["exact"](coarse["x"]))))
    b = float(np.max(np.abs(fine["y"] - problem["exact"](fine["x"]))))
    assert a / max(b, 1e-300) == pytest.approx(4.0, abs=0.4)


def test_newton_squares_the_residual_and_the_chord_method_does_not():
    out = bvp.newton_squares_the_residual()
    assert out["quadratic"]
    assert out["chord_is_linear"]
    assert out["both_reach_the_same_answer"]
    assert out["chord_iterations"] > 3 * out["iterations"]


def test_the_residual_cannot_be_driven_below_the_roundoff_floor():
    """Which is why the iteration stops on stagnation as well as on the tolerance."""
    out = bvp.newton_squares_the_residual()
    assert float(out["residuals"][-1]) > 0.1 * out["roundoff_floor"]
    assert float(out["residuals"][-1]) < 100.0 * out["roundoff_floor"]


# --------------------------------------------------------------------------- what goes wrong


def test_the_matrix_becomes_singular_at_the_discrete_resonance():
    out = bvp.existence_can_fail()
    assert out["worst_is_nearest_the_resonance"]
    assert out["condition_grows_like_one_over_the_gap"]
    assert float(np.max(out["condition_number"])) > 1e8


def test_the_discrete_resonance_sits_below_pi_squared_by_order_h_squared():
    """Measuring from pi^2 instead makes the 1/gap law look broken, and it is not."""
    out = bvp.existence_can_fail()
    assert out["discrete_resonance"] < out["resonance"]
    assert out["shift"] == pytest.approx(out["predicted_shift"], rel=1e-3)
    assert out["spread_using_the_discrete_resonance"] < out["spread_using_pi_squared"]
    assert out["spread_using_the_discrete_resonance"] == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("n", [50, 100, 400])
def test_the_shift_in_the_resonance_is_second_order_in_the_step(n):
    out = bvp.existence_can_fail(coefficients=[5.0], n=n)
    assert out["shift"] == pytest.approx(out["predicted_shift"], rel=1e-2)


def test_the_wiggle_appears_exactly_when_the_cell_number_exceeds_one():
    out = bvp.oscillation_when_the_cell_number_is_too_big()
    assert out["overshoots_exactly_when_the_cell_number_exceeds_one"]
    assert out["the_m_matrix_condition_is_the_cell_number"]
    assert float(np.max(out["overshoot"])) > 0.5


@pytest.mark.parametrize("eps", [0.02, 0.01, 0.002])
def test_the_cell_number_threshold_does_not_depend_on_the_viscosity(eps):
    """The counts are chosen so the cell number lands on the same values whatever eps is.

    Exactly 1 is skipped on purpose: there the off-diagonal is exactly zero, which is neither
    the M-matrix case nor the oscillating one.
    """
    counts = [int(round(1.0 / (2.0 * eps * c))) for c in (4.0, 2.0, 1.2, 0.6, 0.3)]
    out = bvp.oscillation_when_the_cell_number_is_too_big(viscosity=eps, counts=counts)
    assert out["overshoots_exactly_when_the_cell_number_exceeds_one"]
    assert out["the_m_matrix_condition_is_the_cell_number"]


def test_upwinding_removes_the_overshoot_and_costs_an_order():
    out = bvp.upwinding_removes_the_wiggle()
    assert out["upwind_never_overshoots"]
    assert out["central_is_second_order"]
    assert out["upwind_is_first_order"]
    assert out["upwind_costs_an_order"]


def test_fitting_the_unresolved_part_of_the_sweep_would_give_the_wrong_order():
    """The head of the sweep is a different regime, not a slow start."""
    out = bvp.upwinding_removes_the_wiggle()
    assert out["fitting_the_whole_sweep_would_be_wrong"]
    assert abs(out["upwind_order_over_the_whole_sweep"] - 1.0) > 0.3
    assert out["upwind_order"] == pytest.approx(1.0, abs=0.1)


def test_upwinding_wins_only_where_the_wiggle_is_large():
    out = bvp.upwinding_removes_the_wiggle()
    assert out["upwind_only_wins_on_coarse_grids"]
    assert out["upwind_error"][-1] > 10.0 * out["central_error"][-1]


def test_grading_the_grid_into_the_layer_helps_and_grading_away_from_it_hurts():
    good = bvp.graded_grid_for_a_layer(towards="a")
    bad = bvp.graded_grid_for_a_layer(towards="b")
    assert good["graded_wins"]
    assert good["improvement"] > 5.0
    assert not bad["graded_wins"]
    assert good["unknowns"] == bad["unknowns"]


def test_an_unknown_grading_side_is_refused():
    with pytest.raises(ValueError):
        bvp.graded_grid_for_a_layer(towards="middle")


@pytest.mark.parametrize("pattern", ["alternating", "random"])
def test_the_solution_is_second_order_on_an_irregular_grid(pattern):
    out = bvp.supraconvergence_on_an_unequal_grid(pattern=pattern)
    assert out["truncation_is_first_order"]
    assert out["solution_is_second_order"]
    assert out["solution_beats_consistency"]


def test_a_uniform_grid_has_no_gap_between_consistency_and_convergence():
    """The control: on an equal grid both are second order and there is nothing to explain."""
    out = bvp.supraconvergence_on_an_unequal_grid(pattern="uniform")
    assert out["truncation_order"] == pytest.approx(2.0, abs=0.15)
    assert out["solution_order"] == pytest.approx(2.0, abs=0.15)
    assert not out["solution_beats_consistency"]


def test_the_max_norm_would_hide_the_first_order_truncation():
    out = bvp.supraconvergence_on_an_unequal_grid(pattern="alternating")
    assert out["the_max_norm_hides_it"]
    assert out["truncation_order"] == pytest.approx(1.0, abs=0.1)
    assert abs(out["truncation_order_in_the_max_norm"] - 1.0) > 0.1


@pytest.mark.parametrize("pattern", ["alternating", "random", "uniform"])
@pytest.mark.parametrize("n", [5, 17, 64])
def test_every_grid_pattern_spans_the_interval_and_increases(pattern, n):
    grid = bvp.unequal_grid(2.0, 5.0, n, pattern)
    assert grid.size == n + 1
    assert float(grid[0]) == pytest.approx(2.0, abs=1e-14)
    assert float(grid[-1]) == pytest.approx(5.0, abs=1e-14)
    assert np.all(np.diff(grid) > 0.0)


def test_an_unknown_grid_pattern_is_refused():
    with pytest.raises(ValueError):
        bvp.unequal_grid(0.0, 1.0, 10, "chebyshev")


def test_a_grid_with_fewer_than_two_intervals_is_refused():
    with pytest.raises(ValueError):
        bvp.unequal_grid(0.0, 1.0, 1, "alternating")


def test_the_two_methods_cost_about_the_same_on_a_mild_problem():
    """So the reason to prefer finite differences is conditioning, not cost."""
    out = bvp.cost_of_the_two_methods()
    ratio = (np.asarray(out["shooting_evaluations"], dtype=float)
             / np.asarray(out["finite_difference_evaluations"], dtype=float))
    assert np.all(ratio < 10.0)
    assert out["shooting_is_more_accurate_here"]
