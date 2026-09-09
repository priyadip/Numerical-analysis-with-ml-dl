"""Tests for nalib.hyperbolic.

Five groups. The problems are checked against the wave equation and against their own boundary
values, because the travelling bump's boundary data is not zero and assuming it was cost an error
of 3e-2 that hid everything else. The schemes are checked against the formulas written out and
against the growth roots, whose product must be exactly 1 for a wave problem. The Courant group
checks the limit from the roots and from runs, and asserts the countdown that makes a violated run
look clean for a while. The exactness at lam = 1 is asserted at the rounding level, since it is an
identity and not an approximation. The last group is dispersion, where the interesting assertion is
the sign: every stable scheme is too slow, never too fast.

The test worth the most here is the domain of dependence one. It asserts an **exact zero**: the
computed answer is bit for bit identical with and without a bump the true answer depends on, which
is a much stronger statement than any error bound.
"""
import math

import numpy as np
import pytest

from nalib import hyperbolic as hy

POINTS = [21, 41, 81]
LAMS = [0.25, 0.5, 0.9, 1.0]


# --------------------------------------------------------------------------- the problems


@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_the_standing_wave_satisfies_the_wave_equation(mode, speed):
    problem = hy.standing_wave(speed=speed, mode=mode)
    x = np.linspace(0.1, 0.9, 101)
    t, dt, dx = 0.13, 1e-5, 1e-4
    u_tt = (problem["exact"](x, t + dt) - 2.0 * problem["exact"](x, t)
            + problem["exact"](x, t - dt)) / dt ** 2
    u_xx = (problem["exact"](x + dx, t) - 2.0 * problem["exact"](x, t)
            + problem["exact"](x - dx, t)) / dx ** 2
    assert float(np.max(np.abs(u_tt - speed ** 2 * u_xx))) < 1e-3


def test_the_travelling_bump_satisfies_the_wave_equation():
    problem = hy.travelling_bump()
    x = np.linspace(0.3, 1.2, 101)
    t, dt, dx = 0.2, 1e-5, 1e-5
    u_tt = (problem["exact"](x, t + dt) - 2.0 * problem["exact"](x, t)
            + problem["exact"](x, t - dt)) / dt ** 2
    u_xx = (problem["exact"](x + dx, t) - 2.0 * problem["exact"](x, t)
            + problem["exact"](x - dx, t)) / dx ** 2
    assert float(np.max(np.abs(u_tt - u_xx))) < 1e-2 * float(np.max(np.abs(u_xx)))


def test_both_problems_start_from_rest():
    # a forward difference of a function starting from rest is O(dt), not zero, so the test is
    # that it halves when dt halves rather than that it is below some threshold
    for problem in (hy.standing_wave(), hy.travelling_bump()):
        x = np.linspace(problem["a"], problem["b"], 41)
        assert np.allclose(problem["velocity"](x), 0.0)
        rates = []
        for dt in (4e-6, 2e-6, 1e-6):
            rates.append(float(np.max(np.abs(
                (problem["exact"](x, dt) - problem["exact"](x, 0.0)) / dt))))
        assert rates[0] / rates[1] == pytest.approx(2.0, abs=0.05)
        assert rates[1] / rates[2] == pytest.approx(2.0, abs=0.05)


def test_the_bump_boundary_values_are_the_exact_ones_not_zero():
    problem = hy.travelling_bump()
    assert problem["left"](0.4) == pytest.approx(
        float(problem["exact"](problem["a"], 0.4)))
    assert problem["right"](0.4) == pytest.approx(
        float(problem["exact"](problem["b"], 0.4)))
    # and they are not zero, which is the reason this matters
    assert abs(problem["left"](0.4)) > 1e-4


def test_the_standing_wave_boundaries_are_zero_for_all_time():
    problem = hy.standing_wave()
    for t in (0.0, 0.3, 1.7):
        assert abs(float(problem["exact"](problem["a"], t))) < 1e-14
        assert abs(float(problem["exact"](problem["b"], t))) < 1e-14


def test_bad_problem_arguments_are_rejected():
    with pytest.raises(ValueError):
        hy.standing_wave(a=1.0, b=0.0)
    with pytest.raises(ValueError):
        hy.standing_wave(mode=0)
    with pytest.raises(ValueError):
        hy.travelling_bump(width=0.0)


@pytest.mark.parametrize("c,k,h", [(1.0, 0.01, 0.01), (2.0, 0.005, 0.02), (0.5, 0.4, 0.1)])
def test_the_courant_number_is_what_it_says(c, k, h):
    assert hy.courant(c, k, h) == pytest.approx(c * k / h)


# --------------------------------------------------------------------------- the schemes


@pytest.mark.parametrize("lam", LAMS)
@pytest.mark.parametrize("n", POINTS)
def test_the_explicit_step_matches_the_formula(lam, n):
    rng = np.random.default_rng(42)
    old, now = rng.normal(size=n), rng.normal(size=n)
    old[0] = old[-1] = now[0] = now[-1] = 0.0
    got = hy.explicit_step(old, now, lam)
    want = (2.0 * now[1:-1] - old[1:-1]
            + lam ** 2 * (now[2:] - 2.0 * now[1:-1] + now[:-2]))
    assert np.allclose(got[1:-1], want)


@pytest.mark.parametrize("theta", [0.0, 0.1, 0.25, 0.5])
def test_the_implicit_step_satisfies_its_own_equation(theta):
    rng = np.random.default_rng(7)
    n, lam = 33, 1.7
    old, now = rng.normal(size=n), rng.normal(size=n)
    old[0] = old[-1] = now[0] = now[-1] = 0.0
    new = hy.implicit_step(old, now, lam, theta)
    def d2(v):
        return v[2:] - 2.0 * v[1:-1] + v[:-2]
    residual = ((new[1:-1] - 2.0 * now[1:-1] + old[1:-1])
                - lam ** 2 * (theta * d2(new) + (1.0 - 2.0 * theta) * d2(now)
                              + theta * d2(old)))
    assert float(np.max(np.abs(residual))) < 1e-10


def test_the_implicit_step_at_theta_zero_is_the_explicit_one():
    rng = np.random.default_rng(11)
    n, lam = 25, 0.7
    old, now = rng.normal(size=n), rng.normal(size=n)
    old[0] = old[-1] = now[0] = now[-1] = 0.0
    assert np.allclose(hy.implicit_step(old, now, lam, 0.0),
                       hy.explicit_step(old, now, lam))


@pytest.mark.parametrize("lam", LAMS)
def test_a_single_mode_is_multiplied_by_a_root(lam):
    n = 65
    index = np.arange(n)
    for mode in (1, 7, n - 2):
        phase = mode * math.pi / (n - 1)
        u = np.sin(phase * index)
        u[0] = u[-1] = 0.0
        first, second = hy.growth_roots(lam, phase)
        # start on the first root's eigen-solution and one step must reproduce it
        old = u
        now = float(np.real(first[0])) * u
        new = hy.explicit_step(old, now, lam)
        assert np.allclose(new, float(np.real(first[0] ** 2)) * u, atol=1e-9)


@pytest.mark.parametrize("lam", [0.3, 0.9, 1.0, 1.4, 3.0])
@pytest.mark.parametrize("theta", [0.0, 0.25, 0.5])
def test_the_two_roots_multiply_to_exactly_one(lam, theta):
    phases = np.linspace(1e-6, math.pi, 41)
    first, second = hy.growth_roots(lam, phases, theta)
    assert np.allclose(np.abs(first * second), 1.0, atol=1e-10)


def test_bad_scheme_arguments_are_rejected():
    with pytest.raises(ValueError):
        hy.first_step(np.zeros(5), np.zeros(5), 0.5, 0.1, order=3)
    with pytest.raises(ValueError):
        hy.explicit_step(np.zeros(5), np.zeros(7), 0.5)
    with pytest.raises(ValueError):
        hy.explicit_step(np.zeros(2), np.zeros(2), 0.5)
    with pytest.raises(ValueError):
        hy.implicit_step(np.zeros(9), np.zeros(9), 0.5, theta=0.9)
    with pytest.raises(ValueError):
        hy.solve(hy.standing_wave(), 9, 1, 0.1)
    with pytest.raises(ValueError):
        hy.solve(hy.standing_wave(), 2, 10, 0.1)


# --------------------------------------------------------------------------- the Courant limit


@pytest.mark.parametrize("lam,stable", [(0.5, True), (0.999, True), (1.0, True),
                                        (1.001, False), (2.0, False)])
def test_the_root_leaves_the_circle_exactly_at_one(lam, stable):
    assert (hy.largest_root(lam, 0.0) <= 1.0 + 1e-9) == stable


def test_the_courant_sweep_agrees_with_the_roots():
    out = hy.the_courant_condition()
    assert out["the_root_leaves_the_circle_at_one"]
    assert out["the_prediction_is_right_in_every_row"]
    assert out["nothing_below_the_limit_blew_up"]
    assert out["rows_checked"] == 12


def test_a_violated_run_can_look_clean_for_a_while():
    out = hy.the_courant_condition()
    assert out["the_seed_is_rounding"]
    assert out["a_longer_run_finds_more_of_them"]
    assert 1.001 in out["violations_that_did_not"]


def test_the_implicit_threshold_is_exactly_a_quarter():
    out = hy.the_implicit_threshold_is_a_quarter()
    assert out["unconditional_above_a_quarter"]
    assert out["threshold"] == 0.25
    assert 0.24 in out["conditional_ones"]
    assert 0.25 not in out["conditional_ones"]


# --------------------------------------------------------------------------- domains


def test_the_scheme_cannot_see_data_the_answer_depends_on():
    out = hy.changing_data_it_cannot_see()
    assert out["bump_is_inside_the_true_domain"]
    assert out["bump_is_outside_the_numerical_domain"]
    assert out["the_scheme_did_not_notice"]
    assert out["computed_change"] == 0.0
    assert out["the_truth_did"]
    assert out["true_change"] > 0.1


def test_the_two_reaches_are_in_the_ratio_of_the_courant_number():
    out = hy.changing_data_it_cannot_see()
    assert out["true_reach"] / out["numerical_reach"] == pytest.approx(out["lam"])


def test_a_bump_inside_the_numerical_domain_is_refused():
    with pytest.raises(ValueError):
        hy.changing_data_it_cannot_see(bump_offset=-0.1)


# --------------------------------------------------------------------------- accuracy


def test_the_scheme_is_exact_at_courant_one():
    out = hy.the_magic_step()
    assert out["exact_to_rounding"]
    assert out["error_at_one"] < 1e-12
    assert out["gain"] > 1e9


@pytest.mark.parametrize("n", [101, 201, 401])
def test_the_magic_step_holds_at_any_grid_size(n):
    problem = hy.travelling_bump()
    h = (problem["b"] - problem["a"]) / (n - 1)
    k = h / problem["speed"]
    steps = max(int(round(0.3 / k)), 2)
    run = hy.solve(problem, n, steps, steps * k, theta=0.0)
    assert run["lam"] == pytest.approx(1.0)
    assert run["error"] < 1e-12


def test_the_first_step_sets_the_order_of_the_whole_run():
    out = hy.the_first_step_sets_the_order()
    assert out["one_row_sets_the_order"]
    assert out["first_order_start"] == pytest.approx(1.0, abs=0.15)
    assert out["second_order_start"] == pytest.approx(2.0, abs=0.15)
    assert out["gain_at_the_finest"] > 100.0


def test_every_stable_scheme_propagates_short_waves_too_slowly():
    out = hy.dispersion_is_worst_for_short_waves()
    assert out["every_scheme_is_too_slow"]
    assert out["exact_at_lam_one"]
    assert out["worst_grows_as_lam_falls"]


@pytest.mark.parametrize("lam", [0.2, 0.5, 0.9])
def test_the_numerical_speed_formula_matches_the_scheme(lam):
    # a mode of phase phi should come back rotated by exp(i omega k) with sin(omega k/2) =
    # lam sin(phi/2), which is what the speed formula encodes
    phases = np.linspace(0.05, math.pi - 0.05, 25)
    speeds = hy.numerical_wave_speed(lam, phases)
    omega_k = 2.0 * np.arcsin(lam * np.sin(phases / 2.0))
    assert np.allclose(speeds, omega_k / (lam * phases))
    # the two roots are complex conjugates when stable, so the angle of one is the negative of
    # the other's and only the magnitude is the frequency
    first, second = hy.growth_roots(lam, phases)
    assert np.allclose(np.abs(np.angle(first)), omega_k, atol=1e-9)
    assert np.allclose(np.angle(second), -np.angle(first), atol=1e-9)


def test_the_speed_is_exactly_right_at_courant_one():
    phases = np.linspace(1e-6, math.pi, 51)
    assert np.allclose(hy.numerical_wave_speed(1.0, phases), 1.0, atol=1e-12)


def test_the_implicit_scheme_stays_stable_and_loses_accuracy():
    out = hy.the_implicit_scheme_buys_stability_and_pays_in_phase()
    assert out["every_run_is_stable"]
    assert out["every_root_is_on_the_circle"]
    assert out["the_error_grows_with_the_step"]
    assert out["error_span"] > 100.0


def test_the_implicit_error_grows_at_second_order_only_at_small_steps():
    # the scheme is second order in k, and the sweep runs far past where that describes it: the
    # first doubling of lam multiplies the error by 4, and the last one by 15
    out = hy.the_implicit_scheme_buys_stability_and_pays_in_phase()
    errors = np.asarray([row["error"] for row in out["rows"]])
    assert errors[1] / errors[0] == pytest.approx(4.0, abs=0.3)
    assert errors[-1] / errors[-2] > 8.0
    lams = np.asarray([row["lam"] for row in out["rows"]])
    whole = float(np.polyfit(np.log(lams), np.log(errors), 1)[0])
    assert whole > 2.5
