"""Tests for nalib.adi.

Five groups. The two problems are checked against the equation they claim to solve, including the
one whose alpha was chosen to make a given function exact, because that choice is easy to get wrong
and nothing else would catch it. The operators are checked against direct loops. The stability
group checks the dimensional limit both through the growth factor and through runs, and the runs
need the worst mode seeded or they measure nothing. The order group checks that the sweeps hold the
final time fixed, which they did not at first and which made the coarse grid solve a different
problem. The last group is the cost, where the exponent is asserted against the right variable.

The test worth the most here is the one asserting the half step boundary makes no difference: it
records a negative result, and without it a later reader would assume the two treatments were never
compared.
"""
import math

import numpy as np
import pytest

from nalib import adi

POINTS = [9, 13, 17]


# --------------------------------------------------------------------------- the problems


@pytest.mark.parametrize("modes", [(1, 1), (2, 1), (3, 2)])
def test_the_separable_solution_satisfies_the_equation(modes):
    problem = adi.separable_problem(modes=modes)
    mesh = adi.grid_of(problem, 201)
    t, dt, d = 0.02, 1e-6, 1e-4
    x, y = mesh["x"][1:-1, 1:-1], mesh["y"][1:-1, 1:-1]
    u_t = (problem["exact"](x, y, t + dt) - problem["exact"](x, y, t - dt)) / (2.0 * dt)
    lap = ((problem["exact"](x + d, y, t) - 2.0 * problem["exact"](x, y, t)
            + problem["exact"](x - d, y, t))
           + (problem["exact"](x, y + d, t) - 2.0 * problem["exact"](x, y, t)
              + problem["exact"](x, y - d, t))) / d ** 2
    assert float(np.max(np.abs(u_t - problem["alpha"] * lap))) < 1e-4


def test_the_moving_boundary_solution_satisfies_the_equation_with_its_own_alpha():
    problem = adi.moving_boundary_problem()
    assert problem["alpha"] == pytest.approx(1.0 / (2.0 * math.pi ** 2))
    mesh = adi.grid_of(problem, 201)
    t, dt, d = 0.02, 1e-6, 1e-4
    x, y = mesh["x"][1:-1, 1:-1], mesh["y"][1:-1, 1:-1]
    u_t = (problem["exact"](x, y, t + dt) - problem["exact"](x, y, t - dt)) / (2.0 * dt)
    lap = ((problem["exact"](x + d, y, t) - 2.0 * problem["exact"](x, y, t)
            + problem["exact"](x - d, y, t))
           + (problem["exact"](x, y + d, t) - 2.0 * problem["exact"](x, y, t)
              + problem["exact"](x, y - d, t))) / d ** 2
    assert float(np.max(np.abs(u_t - problem["alpha"] * lap))) < 1e-4


def test_only_one_problem_has_moving_boundary_data():
    still = adi.separable_problem()
    moving = adi.moving_boundary_problem()
    mesh = adi.grid_of(still, 21)
    edge = still["exact"](mesh["x"][0, :], mesh["y"][0, :], 0.03)
    assert float(np.max(np.abs(edge))) < 1e-14
    assert not still["boundary_moves"]
    mesh = adi.grid_of(moving, 21)
    early = moving["exact"](mesh["x"][0, :], mesh["y"][0, :], 0.0)
    later = moving["exact"](mesh["x"][0, :], mesh["y"][0, :], 0.5)
    assert float(np.max(np.abs(early - later))) > 0.1
    assert moving["boundary_moves"]


def test_a_grid_too_small_is_rejected():
    with pytest.raises(ValueError):
        adi.grid_of(adi.separable_problem(), 2)
    with pytest.raises(ValueError):
        adi.moving_boundary_problem(a=1.0, b=0.0)


# --------------------------------------------------------------------------- operators


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("shape", [(7, 9), (11, 11), (5, 13)])
def test_the_second_difference_matches_a_direct_loop(axis, shape):
    rng = np.random.default_rng(42)
    values = rng.normal(size=shape)
    got = adi.second_difference(values, axis, 0.25)
    want = np.empty_like(got)
    for i in range(want.shape[0]):
        for j in range(want.shape[1]):
            index = [i, j]
            index[axis] += 1
            lo, hi = list(index), list(index)
            lo[axis] -= 1
            hi[axis] += 1
            want[i, j] = (values[tuple(hi)] - 2.0 * values[tuple(index)]
                          + values[tuple(lo)]) / 0.25 ** 2
    assert np.allclose(got, want)


@pytest.mark.parametrize("lines", [1, 3, 7])
@pytest.mark.parametrize("m", [3, 6, 11])
def test_the_line_solver_solves_the_system_it_claims(m, lines):
    rng = np.random.default_rng(7)
    rhs = rng.normal(size=(m, lines))
    low = rng.normal(size=lines)
    high = rng.normal(size=lines)
    c = 0.37
    out = adi.tridiagonal_lines(rhs, c, low, high)
    for line in range(lines):
        x = out[:, line]
        residual = np.empty(m)
        for i in range(m):
            left = x[i - 1] if i > 0 else low[line]
            right = x[i + 1] if i + 1 < m else high[line]
            residual[i] = -c * left + (1.0 + 2.0 * c) * x[i] - c * right - rhs[i, line]
        assert float(np.max(np.abs(residual))) < 1e-11


def test_the_explicit_step_is_the_five_point_stencil():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(9, 9))
    rx, ry = 0.2, 0.3
    got = adi.explicit_step(values, rx, ry)
    centre = values[1:-1, 1:-1]
    want = (centre
            + rx * (values[2:, 1:-1] - 2.0 * centre + values[:-2, 1:-1])
            + ry * (values[1:-1, 2:] - 2.0 * centre + values[1:-1, :-2]))
    assert np.allclose(got, want)


# --------------------------------------------------------------------------- stability


@pytest.mark.parametrize("r,stable", [(0.2, True), (0.25, True), (0.26, False), (1.0, False)])
def test_the_two_dimensional_explicit_limit_is_a_quarter(r, stable):
    worst = abs(float(adi.growth_factor_2d(r, r, math.pi, math.pi, "explicit")))
    assert (worst <= 1.0 + 1e-12) == stable


def test_the_limit_on_the_sum_is_a_half():
    for rx, ry in ((0.4, 0.09), (0.1, 0.39), (0.25, 0.25)):
        worst = abs(float(adi.growth_factor_2d(rx, ry, math.pi, math.pi, "explicit")))
        assert (worst <= 1.0 + 1e-12) == (rx + ry <= 0.5 + 1e-12)


def test_the_limit_measured_from_runs_agrees_with_the_growth_factor():
    out = adi.the_explicit_limit_tightens()
    assert out["the_growth_passes_one_at_a_quarter"]
    assert out["the_prediction_is_right_in_every_row"]
    assert out["nothing_below_the_limit_blew_up"]
    assert out["rows_checked"] == 10


def test_smooth_data_hides_the_two_dimensional_instability_and_a_seed_shows_it():
    out = adi.the_explicit_limit_tightens()
    assert out["the_smooth_data_hides_it"]
    assert out["the_seeded_data_shows_it"]


def test_the_limit_in_d_dimensions_is_one_over_two_d():
    out = adi.the_explicit_limit_tightens()
    assert out["in_d_dimensions"][1] == pytest.approx(0.5)
    assert out["in_d_dimensions"][2] == pytest.approx(0.25)
    assert out["in_d_dimensions"][3] == pytest.approx(1.0 / 6.0)


def test_adi_is_stable_at_every_ratio_tried():
    out = adi.adi_is_unconditionally_stable()
    assert out["adi_is_stable_at_every_ratio"]
    assert out["explicit_is_not"]
    assert out["largest_adi_factor_seen"] <= 1.0 + 1e-12


@pytest.mark.parametrize("r", [0.3, 3.0, 300.0, 3e5])
def test_the_adi_factor_is_a_product_of_two_bounded_factors(r):
    phases = np.linspace(0.0, math.pi, 37)
    px, py = np.meshgrid(phases, phases, indexing="ij")
    sx = np.sin(px / 2.0) ** 2
    sy = np.sin(py / 2.0) ** 2
    first = (1.0 - 2.0 * r * sy) / (1.0 + 2.0 * r * sx)
    second = (1.0 - 2.0 * r * sx) / (1.0 + 2.0 * r * sy)
    assert np.allclose(adi.growth_factor_2d(r, r, px, py, "adi"), first * second)
    assert float(np.max(np.abs(first * second))) <= 1.0 + 1e-12


def test_an_unknown_scheme_name_is_rejected():
    with pytest.raises(ValueError):
        adi.growth_factor_2d(0.4, 0.4, 1.0, 1.0, "leapfrog")
    with pytest.raises(ValueError):
        adi.solve(adi.separable_problem(), 9, 4, 0.01, scheme="leapfrog")
    with pytest.raises(ValueError):
        adi.solve(adi.separable_problem(), 9, 4, 0.01, half_step="guess")


# --------------------------------------------------------------------------- accuracy


def test_adi_is_second_order_well_above_the_explicit_limit():
    out = adi.adi_is_second_order()
    assert out["second_order"]
    assert out["ratio_over_the_explicit_limit"] > 4.0


def test_the_order_sweep_holds_the_final_time_fixed():
    problem = adi.separable_problem()
    for n in (9, 17, 33):
        h = 1.0 / (n - 1)
        steps = adi.steps_for(problem, h, 2.0, 0.02)
        run = adi.solve(problem, n, steps, 0.02, scheme="adi")
        assert run["k"] * run["steps"] == pytest.approx(0.02)
        assert run["r"] <= 2.0 + 1e-12


@pytest.mark.parametrize("n,steps", [(9, 20), (17, 80), (33, 200)])
def test_adi_and_the_explicit_scheme_agree_where_both_are_valid(n, steps):
    problem = adi.separable_problem()
    left = adi.solve(problem, n, steps, 0.02, scheme="adi")
    right = adi.solve(problem, n, steps, 0.02, scheme="explicit")
    assert left["r"] < 0.25
    assert float(np.max(np.abs(left["u"] - right["u"]))) < 5.0 * left["error"]


def test_both_half_step_treatments_are_second_order():
    out = adi.the_half_step_boundary_costs_less_than_advertised()
    assert out["both_treatments_are_second_order"]
    assert out["the_warning_is_not_reproduced"]


def test_the_two_candidate_boundary_values_differ_by_k_squared_h_squared():
    out = adi.the_half_step_boundary_costs_less_than_advertised()
    assert out["the_gap_is_k_squared_h_squared"]
    assert out["exponent_in_k_over_the_small_steps"] == pytest.approx(2.0, abs=0.15)
    assert out["exponent_in_h"] == pytest.approx(2.0, abs=0.15)


def test_the_zero_boundary_problem_cannot_show_the_difference_at_all():
    out = adi.the_half_step_boundary_costs_less_than_advertised()
    assert out["the_zero_boundary_problem_has_nothing_to_get_wrong"]


# --------------------------------------------------------------------------- cost


@pytest.mark.parametrize("n", POINTS)
def test_the_crank_nicolson_matrix_has_the_bandwidth_claimed(n):
    system = adi.crank_nicolson_matrix(n, 0.3, 0.3)
    assert system["bandwidth"] == n - 2
    assert system["unknowns"] == (n - 2) ** 2
    left = system["left"]
    band = np.max([abs(i - j) for i in range(left.shape[0]) for j in range(left.shape[1])
                   if abs(left[i, j]) > 1e-14])
    assert band == n - 2


def test_the_crank_nicolson_matrix_is_symmetric_and_diagonally_dominant():
    system = adi.crank_nicolson_matrix(11, 0.4, 0.4)
    left = system["left"]
    assert np.allclose(left, left.T)
    off = np.sum(np.abs(left), axis=1) - np.abs(np.diag(left))
    assert bool(np.all(np.abs(np.diag(left)) >= off))


def test_a_matrix_with_no_interior_is_rejected():
    with pytest.raises(ValueError):
        adi.crank_nicolson_matrix(2, 0.3, 0.3)


def test_the_cost_exponents_are_four_and_two():
    out = adi.the_cost_of_a_full_solve()
    assert out["banded_exponent"] == pytest.approx(4.0, abs=0.05)
    assert out["adi_exponent"] == pytest.approx(2.0, abs=0.05)
    assert out["two_orders_apart"]
    assert out["the_gap_grows"]


def test_fitting_against_the_point_count_gives_the_wrong_exponents():
    out = adi.the_cost_of_a_full_solve()
    assert out["banded_exponent_against_the_point_count"] > 4.3
    assert out["adi_exponent_against_the_point_count"] > 2.15


def test_adi_is_cheaper_than_the_explicit_scheme_at_equal_accuracy():
    out = adi.adi_against_explicit_at_equal_accuracy()
    assert out["adi_is_cheaper"]
    assert out["explicit_over_adi"] > 1.0
    assert out["explicit_steps_over_adi_steps"] > 1.0
