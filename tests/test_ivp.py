"""Tests for nalib.ivp.

Three groups. The driver and the steps are checked against problems with closed form solutions,
and against the properties any consistent method must have. The hypotheses of the existence
theorem are checked by measuring them: the Lipschitz estimate has a fitted growth exponent that
separates the well posed cases from the ill posed one exactly. The error behaviour is checked by
measuring local against global order and by comparing the classical bound against what actually
happens.

Every test sweeps step counts, problems and dimensions.
"""
import math

import numpy as np
import pytest

from nalib import ivp

STEP_COUNTS = [10, 20, 40, 80, 160, 320]

#: Problems with a closed form solution, as (f, exact, y0, interval).
PROBLEMS = {
    "decay": (lambda t, y: -2.0 * np.asarray(y, dtype=float),
              lambda t: np.atleast_1d(np.exp(-2.0 * t)), 1.0, (0.0, 1.0)),
    "growth": (lambda t, y: np.asarray(y, dtype=float),
               lambda t: np.atleast_1d(np.exp(t)), 1.0, (0.0, 1.0)),
    "linear forcing": (lambda t, y: -2.0 * np.asarray(y, dtype=float) + t,
                       lambda t: np.atleast_1d(t / 2.0 - 0.25 + 1.25 * np.exp(-2.0 * t)),
                       1.0, (0.0, 1.0)),
    # started at 0.2, not 0.5: the logistic's inflection point is y = 1/2, where y'' = 0 and
    # Euler's local error constant vanishes, which makes it a degenerate test case rather than
    # a representative one. That degeneracy has its own test below.
    "logistic": (lambda t, y: np.asarray(y, dtype=float) * (1.0 - np.asarray(y, dtype=float)),
                 lambda t: np.atleast_1d(1.0 / (1.0 + 4.0 * np.exp(-t))),
                 0.2, (0.0, 2.0)),
}


def harmonic():
    """y'' = -y as a system, whose solution is (cos t, -sin t)."""
    def f(t, y):
        v = np.atleast_1d(np.asarray(y, dtype=float))
        return np.asarray([v[1], -v[0]])

    def exact(t):
        return np.asarray([math.cos(t), -math.sin(t)])

    return f, exact, [1.0, 0.0], (0.0, 1.0)


# --------------------------------------------------------------------------- the driver


@pytest.mark.parametrize("n", STEP_COUNTS)
@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_the_driver_returns_a_consistent_grid(n, kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]
    out = ivp.integrate(f, a, y0, b, n)
    assert out["t"].size == n + 1
    assert out["y"].shape == (n + 1, 1)
    assert float(out["t"][0]) == pytest.approx(a)
    assert float(out["t"][-1]) == pytest.approx(b)
    assert out["step"] == pytest.approx((b - a) / n)
    assert np.max(np.abs(np.diff(out["t"]) - out["step"])) < 1e-12


@pytest.mark.parametrize("n", STEP_COUNTS)
def test_the_driver_handles_a_system(n):
    f, exact, y0, (a, b) = harmonic()
    out = ivp.integrate(f, a, y0, b, n)
    assert out["y"].shape == (n + 1, 2)
    assert np.max(np.abs(out["y"][0] - np.asarray(y0))) == 0.0


def test_zero_steps_is_rejected():
    with pytest.raises(ValueError):
        ivp.integrate(np.sin, 0.0, 1.0, 1.0, 0)


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_euler_counts_one_evaluation_per_step(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]
    counter = [0]
    ivp.integrate(f, a, y0, b, 37, None, counter)
    assert counter[0] == 37


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_euler_reproduces_the_first_step_by_hand(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]
    h = (b - a) / 10.0
    out = ivp.integrate(f, a, y0, b, 10)
    by_hand = ivp.as_state(y0) + h * ivp.as_state(f(a, y0))
    assert np.max(np.abs(out["y"][1] - by_hand)) == 0.0


@pytest.mark.parametrize("n", [5, 20, 80])
def test_backward_euler_solves_its_own_equation(n):
    """The implicit step must satisfy y_new = y + h f(t_new, y_new), not merely be near it."""
    f, exact, y0, (a, b) = PROBLEMS["decay"]
    h = (b - a) / n
    y = ivp.as_state(y0)
    y_new = ivp.backward_euler_step(f, a, y, h)
    residual = y_new - y - h * ivp.as_state(f(a + h, y_new))
    assert float(np.max(np.abs(residual))) < 1e-10


@pytest.mark.parametrize("n", STEP_COUNTS)
def test_backward_euler_is_first_order_too(n):
    f, exact, y0, (a, b) = PROBLEMS["decay"]
    out = ivp.error_against_step(f, exact, a, y0, b, STEP_COUNTS,
                                 ivp.backward_euler_step)
    assert out["fitted_order"] == pytest.approx(1.0, abs=0.25)


def test_backward_euler_is_stable_where_forward_euler_is_not():
    """The whole reason to pay for an implicit solve, previewed here and taken up in lesson 72."""
    lam = -100.0
    f = lambda t, y: lam * ivp.as_state(y)
    # h = 0.1 gives |lam h| = 10, far outside forward Euler's stability limit of 2
    forward = ivp.integrate(f, 0.0, 1.0, 1.0, 10)
    backward = ivp.integrate(f, 0.0, 1.0, 1.0, 10, ivp.backward_euler_step)
    assert abs(float(forward["y"][-1, 0])) > 1e3
    assert abs(float(backward["y"][-1, 0])) < 1.0


# --------------------------------------------------------------------------- the hypotheses


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_the_lipschitz_estimate_does_not_blow_up_for_a_lipschitz_problem(kind):
    """The exponent must be non negative, and it need not be zero.

    A zero exponent means L settles to a constant, which is the usual case. A positive one means
    L shrinks as the box does, which is better still and happens wherever df/dy vanishes at the
    point. The logistic at y = 1/2 is such a place, so requiring the exponent to be near zero
    would reject a perfectly well posed problem.
    """
    f, exact, y0, (a, b) = PROBLEMS[kind]
    out = ivp.lipschitz_near(f, a, [y0])
    assert not out["grows_without_bound"]
    assert out["growth_exponent"] > -0.05


def test_the_estimate_shrinks_where_the_derivative_vanishes():
    """The logistic at its inflection point, where df/dy = 1 - 2y is zero."""
    logistic = PROBLEMS["logistic"][0]
    out = ivp.lipschitz_near(logistic, 0.0, [0.5])
    assert out["growth_exponent"] == pytest.approx(1.0, abs=0.1)
    assert not out["grows_without_bound"]


@pytest.mark.parametrize("power,expected", [(2.0 / 3.0, -1.0 / 3.0), (0.5, -0.5),
                                            (0.25, -0.75)])
def test_the_estimate_grows_at_the_right_rate_when_the_condition_fails(power, expected):
    """|y|^p has L ~ r^(p-1) near the origin, and the fitted exponent finds it exactly."""
    def f(t, y):
        v = ivp.as_state(y)
        return np.sign(v) * np.abs(v) ** power

    out = ivp.lipschitz_near(f, 0.0, [0.0])
    assert out["grows_without_bound"]
    assert out["growth_exponent"] == pytest.approx(expected, abs=0.02)


def test_a_ratio_threshold_would_not_have_caught_it():
    """Why the growth is reported as an exponent and not as a ratio.

    Over five decades of radius, y^(2/3) raises the estimate by a factor of only 46, so any
    threshold set above that calls an ill posed problem well posed.
    """
    def f(t, y):
        v = ivp.as_state(y)
        return np.sign(v) * np.abs(v) ** (2.0 / 3.0)

    out = ivp.lipschitz_near(f, 0.0, [0.0])
    values = np.asarray(out["estimate"], dtype=float)
    assert values[-1] / values[0] < 100.0
    assert out["grows_without_bound"]


def test_the_lipschitz_estimate_is_a_lower_bound():
    """Sampling can only find pairs it happens to draw, so it never overstates L."""
    out = ivp.lipschitz_constant(lambda t, y: -3.0 * ivp.as_state(y), 0.0, [-1.0], [1.0])
    assert out["estimate"] <= 3.0 + 1e-9
    assert out["estimate"] == pytest.approx(3.0, rel=1e-9)
    assert "lower bound" in out["note"]


def test_non_uniqueness_exhibits_genuine_solutions():
    out = ivp.uniqueness_fails_on()
    assert out["count"] >= 3
    assert out["all_start_at_zero"]
    # each curve satisfies the equation to the accuracy of the difference used to check it
    assert float(np.max(out["worst_residual"])) < 1e-4


def test_refining_the_step_converges_on_the_blow_up_time():
    f = lambda t, y: ivp.as_state(y) ** 2
    out = ivp.blows_up_at(f, 0.0, 1.0, 1.5)
    where = np.asarray(out["first_exceeds_threshold"], dtype=float)
    assert np.all(np.diff(where) < 0.0)
    assert float(where[-1]) == pytest.approx(1.0, abs=0.05)
    assert not bool(np.any(out["all_finite"]))


# --------------------------------------------------------------------------- error


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_euler_is_first_order(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]
    out = ivp.error_against_step(f, exact, a, y0, b)
    assert out["fitted_order"] == pytest.approx(1.0, abs=0.15)
    assert out["points_used_in_the_fit"] >= 3


def test_euler_is_first_order_on_a_system():
    f, exact, y0, (a, b) = harmonic()
    out = ivp.error_against_step(f, exact, a, y0, b)
    assert out["fitted_order"] == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_the_local_order_is_one_higher_than_the_global(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]
    out = ivp.local_against_global(f, exact, a, y0, b)
    assert out["local_order"] == pytest.approx(2.0, abs=0.2)
    assert out["global_order"] == pytest.approx(1.0, abs=0.2)
    assert out["difference"] == pytest.approx(1.0, abs=0.2)


def test_the_local_order_is_higher_at_a_degenerate_starting_point():
    """Euler's local error is (h^2/2) y'', so where y'' vanishes the local order rises.

    The logistic has y'' = y'(1 - 2y), which is zero at the inflection point y = 1/2. Starting
    there, the one step error is O(h^3) rather than O(h^2), and a test suite that used that
    initial condition would measure a local order of 3 for a first order method.

    The global order is unaffected, because y leaves the inflection point immediately.
    """
    logistic = PROBLEMS["logistic"][0]
    exact = lambda t: np.atleast_1d(1.0 / (1.0 + np.exp(-t)))
    out = ivp.local_against_global(logistic, exact, 0.0, 0.5, 2.0)
    assert out["local_order"] == pytest.approx(3.0, abs=0.2)
    assert out["global_order"] == pytest.approx(1.0, abs=0.2)


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_the_classical_bound_holds_and_is_pessimistic(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]
    out = ivp.error_bound(f, exact, a, y0, b, 2.0, 4.0)
    assert out["bound_holds"]
    ratios = np.asarray(out["bound_over_observed"], dtype=float)
    assert float(np.min(ratios)) > 2.0


def test_the_bound_is_pessimistic_by_a_factor_that_does_not_shrink():
    """Both sides are O(h), so the ratio is constant: refining does not close the gap."""
    f, exact, y0, (a, b) = PROBLEMS["decay"]
    out = ivp.error_bound(f, exact, a, y0, b, 2.0, 4.0)
    ratios = np.asarray(out["bound_over_observed"], dtype=float)
    assert float(np.max(ratios) / np.min(ratios)) < 1.2


def test_euler_is_still_first_order_at_the_smallest_step_worth_trying():
    """The textbook floor at sqrt(eps) is a worst case that does not happen here.

    Rounding accumulates like sqrt(n) rather than n, because the per step errors are not
    systematically aligned, so the truncation term still dominates at h = sqrt(eps).
    """
    f, exact, y0, (a, b) = PROBLEMS["decay"]
    out = ivp.euler_step_size_floor(f, exact, a, y0, b, [2 ** k for k in range(4, 21)])
    assert out["fitted_order_over_the_sweep"] == pytest.approx(1.0, abs=0.05)
    assert not out["reached_the_minimum"]
    assert out["best_error"] < 1e-6
