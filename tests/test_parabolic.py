"""Tests for nalib.parabolic.

Five groups. The problems are checked against the equation they claim to solve, by putting the
exact solution into a difference quotient, because a wrong exact solution makes every error
measurement below meaningless and nothing else would catch it. The stepper is checked against the
weighted formula written out directly, at several ``theta``, and against the amplification factor,
which is the same statement in Fourier space. The stability results are checked on both sides of
the limit and on both kinds of data. The orders are fitted twice, once in a sweep that can see the
time order and once in a sweep that cannot, and the second one is asserted to be blind on purpose.
The last group is the two schemes that fail, where the assertion is that they fail in the specific
way claimed rather than merely that they are bad.

The tests worth the most here are the ones that would pass for the wrong reason if written
carelessly. An unstable scheme looks stable on a smooth initial condition for a long time, so the
stability tests use the amplitude of the mode that actually grows. Crank-Nicolson's second order in
time is invisible in a fixed-ratio sweep, so the sweep that hides it is tested for hiding it.
"""
import math

import numpy as np
import pytest

from nalib import parabolic as pb

POINTS = [11, 21, 41]
THETAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# --------------------------------------------------------------------------- the problems


@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_the_sine_solution_satisfies_the_heat_equation(mode, alpha):
    problem = pb.sine_problem(alpha=alpha, mode=mode)
    x = np.linspace(problem["a"], problem["b"], 401)[1:-1]
    t, dt, dx = 0.03, 1e-6, 1e-4
    u_t = (problem["exact"](x, t + dt) - problem["exact"](x, t - dt)) / (2.0 * dt)
    u_xx = (problem["exact"](x + dx, t) - 2.0 * problem["exact"](x, t)
            + problem["exact"](x - dx, t)) / dx ** 2
    assert float(np.max(np.abs(u_t - alpha * u_xx))) < 1e-5


def test_the_step_solution_satisfies_the_heat_equation():
    problem = pb.step_problem()
    x = np.linspace(problem["a"], problem["b"], 201)[1:-1]
    t, dt, dx = 0.02, 1e-6, 1e-4
    u_t = (problem["exact"](x, t + dt) - problem["exact"](x, t - dt)) / (2.0 * dt)
    u_xx = (problem["exact"](x + dx, t) - 2.0 * problem["exact"](x, t)
            + problem["exact"](x - dx, t)) / dx ** 2
    assert float(np.max(np.abs(u_t - u_xx))) < 1e-4


def test_the_step_solution_starts_as_a_step():
    problem = pb.step_problem(terms=2000)
    x = np.asarray([0.1, 0.3, 0.7, 0.9])
    early = problem["exact"](x, 1e-6)
    assert early[0] == pytest.approx(1.0, abs=1e-3)
    assert early[1] == pytest.approx(1.0, abs=1e-3)
    assert early[2] == pytest.approx(0.0, abs=1e-3)
    assert early[3] == pytest.approx(0.0, abs=1e-3)


def test_both_problems_keep_their_boundary_values():
    for problem in (pb.sine_problem(), pb.step_problem()):
        ends = np.asarray([problem["a"], problem["b"]])
        assert np.allclose(problem["exact"](ends, 0.05), 0.0, atol=1e-12)


def test_a_backwards_interval_is_rejected():
    with pytest.raises(ValueError):
        pb.sine_problem(a=1.0, b=0.0)


def test_a_mode_below_one_is_rejected():
    with pytest.raises(ValueError):
        pb.sine_problem(mode=0)


# --------------------------------------------------------------------------- the stepper


@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("n", POINTS)
def test_one_step_satisfies_the_weighted_formula(theta, n):
    rng = np.random.default_rng(42)
    u = rng.normal(size=n)
    u[0] = u[-1] = 0.0
    r = 0.3
    new = pb.theta_step(u, r, theta)
    old_d2 = u[2:] - 2.0 * u[1:-1] + u[:-2]
    new_d2 = new[2:] - 2.0 * new[1:-1] + new[:-2]
    residual = (new[1:-1] - u[1:-1]) - r * (theta * new_d2 + (1.0 - theta) * old_d2)
    assert float(np.max(np.abs(residual))) < 1e-12


@pytest.mark.parametrize("theta", THETAS)
def test_a_single_mode_is_multiplied_by_the_growth_factor(theta):
    n = 65
    index = np.arange(n)
    for mode in (1, 5, 17, n - 2):
        phase = mode * math.pi / (n - 1)
        u = np.sin(phase * index)
        u[0] = u[-1] = 0.0
        r = 0.35
        new = pb.theta_step(u, r, theta)
        want = pb.growth_factor(r, phase, theta) * u
        assert float(np.max(np.abs(new - want))) < 1e-11


def test_the_stepper_keeps_the_boundary_values_it_is_given():
    u = np.zeros(9)
    out = pb.theta_step(u, 0.4, 0.5, left=2.0, right=-3.0)
    assert out[0] == 2.0
    assert out[-1] == -3.0


def test_a_theta_outside_zero_to_one_is_rejected():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            pb.theta_step(np.zeros(5), 0.4, bad)


def test_too_few_points_is_rejected():
    with pytest.raises(ValueError):
        pb.theta_step(np.zeros(2), 0.4, 0.5)
    with pytest.raises(ValueError):
        pb.solve(pb.sine_problem(), 2, 4, 0.01)
    with pytest.raises(ValueError):
        pb.solve(pb.sine_problem(), 11, 0, 0.01)


@pytest.mark.parametrize("alpha,k,h", [(1.0, 0.01, 0.1), (2.5, 0.004, 0.05), (0.1, 1.0, 2.0)])
def test_the_mesh_ratio_is_what_it_says(alpha, k, h):
    assert pb.mesh_ratio(alpha, k, h) == pytest.approx(alpha * k / h ** 2)


# --------------------------------------------------------------------------- stability


@pytest.mark.parametrize("theta", THETAS)
def test_the_stability_limit_matches_the_growth_factor(theta):
    limit = pb.stability_limit(theta)
    if math.isinf(limit):
        for r in (0.5, 5.0, 500.0, 5e6):
            assert abs(pb.growth_factor(r, math.pi, theta)) <= 1.0
    else:
        assert abs(pb.growth_factor(limit, math.pi, theta)) == pytest.approx(1.0, abs=1e-12)
        assert abs(pb.growth_factor(1.01 * limit, math.pi, theta)) > 1.0


def test_the_explicit_limit_is_one_half():
    assert pb.stability_limit(0.0) == pytest.approx(0.5)


def test_every_member_at_or_above_a_half_is_unconditionally_stable():
    for theta in (0.5, 0.6, 0.9, 1.0):
        assert math.isinf(pb.stability_limit(theta))
    for theta in (0.0, 0.1, 0.4, 0.49):
        assert math.isfinite(pb.stability_limit(theta))


def test_the_prediction_matches_the_run_in_every_sweep():
    out = pb.the_explicit_limit_is_sharp()
    assert out["the_prediction_is_right_in_every_row"]
    assert out["rows_checked"] == 21
    assert out["nothing_below_the_limit_blew_up"]


def test_a_smooth_short_run_hides_the_instability_and_a_jump_does_not():
    out = pb.the_explicit_limit_is_sharp()
    assert out["a_short_run_on_a_smooth_mode_hides_it"]
    assert out["the_long_run_finds_it"]
    assert out["the_jump_finds_it_in_the_short_run"]


def test_the_seed_of_a_smooth_mode_is_rounding_and_of_a_jump_is_not():
    out = pb.the_explicit_limit_is_sharp()
    eps = float(np.finfo(float).eps)
    assert all(row["seed"] == eps for row in out["short_run"])
    assert all(row["seed"] > 1e-3 for row in out["jump_data"])


@pytest.mark.parametrize("n", [9, 17, 33])
def test_the_worst_mode_reader_finds_a_mode_it_is_given(n):
    index = np.arange(n)
    inner = n - 2
    u = np.sin(inner * math.pi * index / (inner + 1))
    u[0] = u[-1] = 0.0
    assert pb.worst_mode_amplitude(u) == pytest.approx(1.0, abs=1e-10)
    smooth = np.sin(math.pi * index / (n - 1))
    smooth[0] = smooth[-1] = 0.0
    assert pb.worst_mode_amplitude(smooth) < 1e-12


# --------------------------------------------------------------------------- Bender-Schmidt


def test_bender_schmidt_is_the_average_to_one_rounding_but_not_exactly():
    out = pb.bender_schmidt_is_an_average()
    assert out["agrees_to_one_rounding"]
    assert not out["exactly_the_average"]
    assert bool(np.all(out["roundings"] < 4.0))


def test_halving_the_space_step_quadruples_the_bender_schmidt_step_count():
    out = pb.bender_schmidt_is_an_average()
    assert np.allclose(out["step_growth"], 4.0)


def test_bender_schmidt_runs_at_exactly_half():
    out = pb.bender_schmidt_is_an_average()
    assert np.allclose(out["r"], 0.5)


def test_bender_schmidt_refuses_a_horizon_shorter_than_one_step():
    with pytest.raises(ValueError):
        pb.bender_schmidt(pb.sine_problem(), 11, 1e-9)


# --------------------------------------------------------------------------- orders


def test_the_joint_sweep_cannot_tell_the_members_apart():
    out = pb.orders_of_the_weighted_family()
    assert out["the_joint_sweep_measures_two_for_everything"]
    assert out["and_one_in_k_for_everything"]


def test_the_fine_grid_sweep_can():
    out = pb.orders_of_the_weighted_family()
    assert out["crank_nicolson_is_second_order_in_time"]
    assert out["backward_is_first_order_in_time"]


def test_the_explicit_scheme_has_no_independent_time_order_to_measure():
    out = pb.orders_of_the_weighted_family()
    assert out["the_explicit_scheme_cannot_be_measured_this_way"]
    explicit = next(row for row in out["in_time"] if row["theta"] == 0.0)
    assert explicit["largest_stable_k"] < explicit["smallest_k_in_the_sweep"]


def test_the_explicit_scheme_is_fourth_order_at_one_sixth_and_nowhere_else():
    out = pb.the_lucky_ratio()
    assert out["fourth_order_at_one_sixth"]
    assert out["second_order_everywhere_else"]
    assert out["the_cancelling_factor_is_zero_only_there"]
    assert out["gain_at_the_finest_grid"] > 100.0


# --------------------------------------------------------------------------- ringing


def test_crank_nicolson_flips_the_worst_mode_and_backward_never_does():
    out = pb.crank_nicolson_rings_on_a_step()
    assert out["crank_nicolson_flips_the_mode_every_step_above_the_half"]
    assert out["the_backward_scheme_never_flips"]


def test_the_growth_factor_signs_are_what_the_formulas_say():
    out = pb.crank_nicolson_rings_on_a_step()
    assert out["crank_nicolson_growth_turns_negative_above_r_half"]
    assert out["backward_growth_is_always_positive"]
    assert out["both_stay_bounded"]


def test_the_profile_only_dips_when_the_factor_is_near_minus_one():
    out = pb.crank_nicolson_rings_on_a_step()
    assert out["the_profile_only_dips_when_the_factor_is_near_minus_one"]
    assert out["worst_undershoot"] > 0.1


@pytest.mark.parametrize("r", [1.0, 5.0, 25.0])
def test_the_crank_nicolson_factor_matches_its_closed_form(r):
    assert pb.growth_factor(r, math.pi, 0.5) == pytest.approx((1.0 - 2.0 * r) / (1.0 + 2.0 * r))
    assert pb.growth_factor(r, math.pi, 1.0) == pytest.approx(1.0 / (1.0 + 4.0 * r))


# --------------------------------------------------------------------------- the failures


def test_richardson_has_a_root_outside_the_circle_at_every_ratio():
    out = pb.richardson_cannot_be_saved()
    assert out["the_product_of_the_roots_is_always_minus_one"]
    assert out["a_root_is_outside_at_every_ratio"]
    assert out["every_run_grows"]


def test_richardson_takes_over_exactly_when_the_seed_says_it_will():
    # the parasitic root is 1 + sqrt(2) at r = 1/4, growing from a rounding seed, so it needs
    # log(1/eps) / log(root) steps to reach 1. Before that the physical decay is still winning
    # and the run looks like it is behaving, which is the same trap as the explicit scheme.
    problem = pb.sine_problem()
    n, r = 21, 0.25
    h = 1.0 / (n - 1)
    k = r * h ** 2
    root = -4.0 * r + math.sqrt(16.0 * r ** 2 + 1.0)
    largest = max(abs(root), abs(1.0 / root))
    crossover = math.log(1.0 / float(np.finfo(float).eps)) / math.log(largest)
    early = pb.richardson(problem, n, int(0.5 * crossover), int(0.5 * crossover) * k)
    late = pb.richardson(problem, n, int(3.0 * crossover), int(3.0 * crossover) * k)
    assert early["grew_by"] < 1.0
    assert late["grew_by"] > 1e6
    assert 20.0 < crossover < 60.0


def test_dufort_frankel_stops_converging_when_k_over_h_is_held_fixed():
    out = pb.dufort_frankel_solves_the_wrong_equation()
    assert out["the_error_stops_falling_at_fixed_k_over_h"]
    assert out["it_converges_once_k_over_h_goes_to_zero"]


def test_the_dufort_frankel_limit_scales_like_the_square_of_k_over_h():
    out = pb.dufort_frankel_solves_the_wrong_equation()
    assert out["the_limit_scales_like_the_square"]
    assert out["exponent_over_the_small_ratios"] == pytest.approx(2.0, abs=0.1)


def test_fitting_the_whole_dufort_frankel_sweep_gives_the_wrong_exponent():
    out = pb.dufort_frankel_solves_the_wrong_equation()
    assert abs(out["exponent_over_the_whole_sweep"] - 2.0) > 0.2


def test_dufort_frankel_stays_finite_where_the_explicit_scheme_would_not():
    problem = pb.sine_problem()
    n, steps = 21, 40
    h = 1.0 / (n - 1)
    k = 5.0 * h ** 2                              # r = 5, ten times the explicit limit
    out = pb.dufort_frankel(problem, n, steps, steps * k)
    assert out["finite"]
    assert out["r"] == pytest.approx(5.0)


# --------------------------------------------------------------------------- cost


def test_crank_nicolson_is_the_cheapest_at_equal_accuracy():
    out = pb.cost_at_equal_accuracy()
    assert out["every_scheme_met_the_target"]
    assert out["crank_nicolson_is_cheapest"]
    assert out["explicit_over_crank_nicolson"] > 1.0
    assert out["backward_over_crank_nicolson"] > 1.0


def test_the_cheapest_runs_land_where_the_error_balance_says():
    out = pb.cost_at_equal_accuracy()
    # a first order scheme in time balances at k ~ h^2, a second order one at k ~ h
    assert out["k_over_h"]["crank nicolson"] > 10.0 * out["k_over_h"]["explicit"]
    assert out["k_over_h_squared"]["explicit"] < 1.0
