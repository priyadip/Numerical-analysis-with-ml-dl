"""Tests for nalib.taylorode.

Two groups. Picard is checked against the one case where its iterates are known in closed form,
against the failure of the contraction argument when the Lipschitz condition does not hold, and
for the property that makes it a proof rather than a method. The Taylor methods are checked for
the order they are built for, on several problems and at several orders, and against Euler at
equal cost.

Every test sweeps orders, iteration counts and problems.
"""
import math

import numpy as np
import pytest

from nalib import taylorode as ta
from nalib.ivp import as_state

ORDERS = [1, 2, 3, 4]
STEP_COUNTS = [10, 20, 40, 80, 160, 320]


def linear_forcing():
    """y' = -2y + t, y(0) = 1, with every total derivative in closed form."""
    def f(t, y):
        return -2.0 * as_state(y) + t

    def derivatives(t, y, k):
        # y' = -2y + t, y'' = -2y' + 1, y^(k) = -2 y^(k-1) for k >= 3
        first = -2.0 * as_state(y) + t
        if k == 1:
            return first
        current = -2.0 * first + 1.0
        for _ in range(k - 2):
            current = -2.0 * current
        return current

    def exact(t):
        return np.atleast_1d(t / 2.0 - 0.25 + 1.25 * np.exp(-2.0 * t))

    return f, derivatives, exact, 1.0, (0.0, 1.0)


def pure_exponential():
    """y' = a y, whose kth derivative along the solution is a^k y."""
    a = -1.5

    def f(t, y):
        return a * as_state(y)

    def derivatives(t, y, k):
        return (a ** k) * as_state(y)

    def exact(t):
        return np.atleast_1d(np.exp(a * t))

    return f, derivatives, exact, 1.0, (0.0, 1.0)


PROBLEMS = {"linear forcing": linear_forcing, "pure exponential": pure_exponential}


# --------------------------------------------------------------------------- Picard


@pytest.mark.parametrize("t_end", [0.4, 0.8, 1.2])
def test_the_picard_iterates_are_the_taylor_partial_sums(t_end):
    """On y' = y the kth iterate is the degree k Taylor polynomial of exp, exactly."""
    out = ta.picard_matches_the_series(t_end, iterations=6)
    assert out["iterates_are_partial_sums"]
    assert float(np.max(out["gap_to_the_taylor_partial_sum"])) < 1e-9


@pytest.mark.parametrize("iterations", [2, 4, 6, 8])
def test_picard_converges_to_the_solution(iterations):
    """The gap falls like (Lt)^k / k!, so how far it gets depends on how many iterations ran."""
    out = ta.picard_matches_the_series(0.8, iterations)
    gaps = np.asarray(out["gap_to_the_exact_solution"], dtype=float)
    assert np.all(np.diff(gaps) < 0.0)
    assert gaps[-1] < gaps[0] / (2.0 ** iterations)


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_picard_reproduces_the_initial_condition_at_every_iteration(kind):
    f, derivatives, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ta.picard(f, a, y0, b, iterations=4)
    for k in range(out["iterates"].shape[0]):
        assert float(np.max(np.abs(out["iterates"][k][0] - as_state(y0)))) < 1e-14


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_picard_converges_on_a_lipschitz_problem(kind):
    f, derivatives, exact, y0, (a, b) = PROBLEMS[kind]()
    # 12 iterations, because the convergence is like (Lt)^k / k! and with L = 2 on [0, 1]
    # eight iterations leaves 1e-03, which is convergence and is not yet accuracy
    out = ta.picard(f, a, y0, b, iterations=12)
    assert out["converging"]
    changes = np.asarray(out["change_per_iteration"], dtype=float)
    assert changes[-1] < changes[0] / 100.0
    want = np.stack([exact(float(v)) for v in out["t"]])
    assert float(np.max(np.abs(out["final"] - want))) < 1e-4


def test_two_grid_points_is_the_minimum():
    with pytest.raises(ValueError):
        ta.picard(lambda t, y: as_state(y), 0.0, 1.0, 1.0, nodes=1)


def test_picard_cannot_choose_between_the_non_unique_solutions():
    """The contraction argument needs the Lipschitz condition, and here it does not hold."""
    out = ta.picard_needs_lipschitz()
    assert out["from_zero_stays_zero"]
    assert not out["from_small_settles"]
    # started slightly off zero the iterates keep moving rather than settling
    changes = np.asarray(out["from_small_changes"], dtype=float)
    assert float(np.max(changes)) > 10.0 * float(changes[0])


# --------------------------------------------------------------------------- Taylor methods


@pytest.mark.parametrize("order", ORDERS)
def test_a_taylor_step_of_order_one_is_euler(order):
    from nalib.ivp import euler_step

    f, derivatives, exact, y0, (a, b) = linear_forcing()
    step = ta.taylor_step(derivatives, 1)
    h = 0.1
    assert float(np.max(np.abs(step(f, a, as_state(y0), h)
                               - euler_step(f, a, as_state(y0), h)))) < 1e-15


def test_order_zero_is_rejected():
    f, derivatives, exact, y0, (a, b) = linear_forcing()
    with pytest.raises(ValueError):
        ta.taylor_step(derivatives, 0)


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_taylor_methods_reach_the_order_they_were_built_for(kind):
    f, derivatives, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ta.taylor_orders_are_exact(f, derivatives, exact, a, y0, b, ORDERS, STEP_COUNTS)
    assert out["all_match"]
    for claimed, fitted in zip(out["order"], out["fitted_order"]):
        assert fitted == pytest.approx(claimed, abs=0.3)


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_a_higher_order_taylor_method_is_more_accurate_at_the_same_step_count(kind):
    f, derivatives, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ta.taylor_orders_are_exact(f, derivatives, exact, a, y0, b, ORDERS, STEP_COUNTS)
    errors = np.asarray(out["finest_error"], dtype=float)
    assert np.all(np.diff(errors) < 0.0)


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
@pytest.mark.parametrize("order", ORDERS)
def test_the_cost_is_one_derivative_evaluation_per_order_per_step(kind, order):
    f, derivatives, exact, y0, (a, b) = PROBLEMS[kind]()
    from nalib.ivp import integrate

    counter = [0]
    integrate(f, a, y0, b, 25, ta.taylor_step(derivatives, order), counter)
    assert counter[0] == 25 * order


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_taylor_beats_euler_at_equal_cost(kind):
    f, derivatives, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ta.taylor_against_euler_at_equal_cost(f, derivatives, exact, a, y0, b)
    assert out["taylor_wins"]
    advantage = np.asarray(out["advantage"], dtype=float)
    assert np.all(np.diff(advantage) > 0.0)
    assert float(advantage[-1]) > 1e5


def test_a_wrong_derivative_shows_up_as_a_missing_order():
    """Why the order is measured and not assumed: the formulas are the thing that can be wrong."""
    f, derivatives, exact, y0, (a, b) = linear_forcing()

    def broken(t, y, k):
        # correct at k = 1, wrong sign at k = 2 and beyond
        return derivatives(t, y, k) if k == 1 else -derivatives(t, y, k)

    good = ta.taylor_orders_are_exact(f, derivatives, exact, a, y0, b, [3], STEP_COUNTS)
    bad = ta.taylor_orders_are_exact(f, broken, exact, a, y0, b, [3], STEP_COUNTS)
    assert float(good["fitted_order"][0]) == pytest.approx(3.0, abs=0.3)
    assert float(bad["fitted_order"][0]) < 2.0


# --------------------------------------------------------------------------- the cost argument


def test_the_term_count_is_the_rooted_tree_count():
    out = ta.taylor_term_count([1, 2, 3, 4, 5, 6, 7])
    assert list(out["new_terms"]) == [1, 1, 2, 4, 9, 20, 48]
    assert list(out["cumulative_terms"]) == [1, 2, 4, 8, 17, 37, 85]


def test_the_term_count_grows_faster_than_the_order():
    out = ta.taylor_term_count([1, 2, 3, 4, 5, 6, 7, 8])
    counts = np.asarray(out["cumulative_terms"], dtype=float)
    orders = np.asarray(out["order"], dtype=float)
    ratios = counts / orders
    # the first two orders tie at 1 term per order; from there the ratio rises without limit
    assert np.all(np.diff(ratios) >= 0.0)
    assert np.all(np.diff(ratios[1:]) > 0.0)
    assert float(ratios[-1]) > 20.0
    assert float(counts[-1]) > 100.0
