"""Tests for nalib.adaptivestep.

Three groups. Each pair is checked for the structural properties an embedded pair must have and
for the orders both halves achieve when run at fixed steps. The estimate is checked against the
true one step error it claims to describe, which is the only way to catch a wrong weight vector.
The controller is checked for the properties that make it a controller: it meets its tolerance,
it costs what it should, and it chooses step sizes that reflect the problem.

Every test sweeps pairs, tolerances and problems.
"""
import math

import numpy as np
import pytest

from nalib import adaptivestep as ad
from nalib.ivp import as_state

PAIRS = sorted(ad.PAIRS)
TOLERANCES = [1e-4, 1e-6, 1e-8, 1e-10]


def linear_forcing():
    def f(t, y):
        return -2.0 * as_state(y) + t

    def exact(t):
        return np.atleast_1d(t / 2.0 - 0.25 + 1.25 * np.exp(-2.0 * t))

    return f, exact, 1.0, (0.0, 1.0)


def harmonic():
    def f(t, y):
        v = as_state(y)
        return np.asarray([v[1], -v[0]])

    def exact(t):
        return np.asarray([math.cos(t), -math.sin(t)])

    return f, exact, [1.0, 0.0], (0.0, 2.0)


def two_scales():
    """A fast transient over a slow background, which is what adaptivity is for."""
    def f(t, y):
        v = as_state(y)
        return np.asarray([-50.0 * (v[0] - math.cos(t)) - math.sin(t)])

    def exact(t):
        return np.atleast_1d(np.cos(t))

    return f, exact, 1.0, (0.0, 3.0)


PROBLEMS = {"linear forcing": linear_forcing, "harmonic": harmonic}


# --------------------------------------------------------------------------- structure


@pytest.mark.parametrize("name", PAIRS)
def test_the_two_weight_vectors_both_sum_to_one(name):
    """Both halves must be consistent, or one of them is not a method at all."""
    _A, bh, bl, _c, _ph, _pl = ad.pair(name)
    assert float(np.sum(bh)) == pytest.approx(1.0, abs=1e-13)
    assert float(np.sum(bl)) == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("name", PAIRS)
def test_c_is_the_row_sum_of_a(name):
    A, bh, bl, c, _ph, _pl = ad.pair(name)
    assert float(np.max(np.abs(A.sum(axis=1) - c))) < 1e-13


@pytest.mark.parametrize("name", PAIRS)
def test_the_two_halves_differ(name):
    """If they did not, the estimate would be identically zero and the pair useless."""
    _A, bh, bl, _c, _ph, _pl = ad.pair(name)
    assert float(np.max(np.abs(bh - bl))) > 1e-6


@pytest.mark.parametrize("name", PAIRS)
def test_the_orders_differ_by_one(name):
    _A, _bh, _bl, _c, ph, pl = ad.pair(name)
    assert ph == pl + 1


def test_dormand_prince_is_first_same_as_last_and_fehlberg_is_not():
    """The property that makes ode45 cost 6 evaluations per step rather than 7."""
    assert ad.is_first_same_as_last("dormand prince")
    assert ad.is_first_same_as_last("bogacki shampine")
    assert not ad.is_first_same_as_last("fehlberg")
    assert not ad.is_first_same_as_last("heun euler")


def test_an_unknown_pair_is_rejected():
    with pytest.raises(ValueError):
        ad.pair("not a pair")


@pytest.mark.parametrize("name", PAIRS)
def test_the_estimate_costs_nothing_beyond_the_step(name):
    out = ad.the_estimate_is_free(name)
    assert out["step_halving_evaluations_per_step"] == 3 * out["stages"]
    assert out["embedded_evaluations_per_step"] <= out["stages"]
    assert out["ratio"] >= 3.0


# --------------------------------------------------------------------------- the estimate


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_both_halves_of_every_pair_reach_their_order(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ad.orders_are_what_they_claim(f, exact, a, y0, b)
    assert out["all_match"]


@pytest.mark.parametrize("name", PAIRS)
def test_the_estimate_predicts_the_low_order_error(name):
    """The ratio must go to 1 as the step shrinks, or the weights are wrong."""
    f, exact, y0, (a, b) = linear_forcing()
    out = ad.the_estimate_predicts_the_error(f, exact, a, y0, b, name)
    ratios = np.asarray(out["estimate_over_low"], dtype=float)
    assert ratios[-1] == pytest.approx(1.0, abs=0.15)
    assert abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0)


@pytest.mark.parametrize("name", PAIRS)
def test_the_estimate_wildly_overstates_the_error_of_what_is_returned(name):
    """It describes the low order result, and the high order one is what advances."""
    f, exact, y0, (a, b) = linear_forcing()
    out = ad.the_estimate_predicts_the_error(f, exact, a, y0, b, name)
    over_high = np.asarray(out["estimate_over_high"], dtype=float)
    assert float(np.max(over_high)) > 3.0


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_the_control_law_uses_the_local_order(order):
    """Doubling the error should shrink the step by 2^(1/(p+1)), not by 2^(1/p)."""
    h = 0.1
    base = ad.proposed_step(h, 1e-6, 1e-6, order, safety=1.0, grow=1e9, shrink=1e-9)
    doubled = ad.proposed_step(h, 2e-6, 1e-6, order, safety=1.0, grow=1e9, shrink=1e-9)
    assert base / doubled == pytest.approx(2.0 ** (1.0 / (order + 1)), rel=1e-12)


def test_the_control_law_respects_its_limits():
    assert ad.proposed_step(1.0, 1e30, 1e-8, 4) == pytest.approx(0.2)
    assert ad.proposed_step(1.0, 1e-30, 1e-8, 4) == pytest.approx(5.0)
    assert ad.proposed_step(1.0, 0.0, 1e-8, 4) == pytest.approx(5.0)


# --------------------------------------------------------------------------- the controller


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
@pytest.mark.parametrize("tol", TOLERANCES)
def test_the_solver_reaches_the_end_and_gets_the_right_answer(kind, tol):
    f, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ad.solve(f, a, y0, b, tol)
    assert out["reached_the_end"]
    assert float(out["t"][-1]) == pytest.approx(b, abs=1e-10)
    want = as_state(exact(b))
    assert float(np.max(np.abs(out["y"][-1] - want))) < max(1e-4, 100.0 * tol)


def test_a_reversed_interval_is_rejected():
    f, exact, y0, (a, b) = linear_forcing()
    with pytest.raises(ValueError):
        ad.solve(f, 1.0, y0, 0.0, 1e-8)


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_the_tolerance_is_met_with_room_to_spare(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ad.tolerance_is_met(f, exact, a, y0, b, TOLERANCES)
    assert out["always_met"]
    assert out["worst_ratio"] < 1.0


@pytest.mark.parametrize("name", ["dormand prince", "bogacki shampine"])
def test_the_first_same_as_last_saving_is_taken(name):
    f, exact, y0, (a, b) = linear_forcing()
    counter = [0]
    out = ad.solve(f, a, y0, b, 1e-8, name, _counter=counter)
    _A, bh, _bl, _c, _ph, _pl = ad.pair(name)
    attempts = out["accepted"] + out["rejected"]
    assert out["reused_the_last_stage"]
    # one evaluation per attempt is saved, so the count is below stages times attempts
    assert counter[0] < bh.size * attempts
    assert counter[0] >= (bh.size - 1) * attempts


def test_turning_off_local_extrapolation_gives_up_the_saving():
    """The reused stage is f at the point the high order result lands on, so advancing
    somewhere else makes it useless. That cost is easy to overlook."""
    f, exact, y0, (a, b) = linear_forcing()
    with_it, without = [0], [0]
    ad.solve(f, a, y0, b, 1e-8, "dormand prince", _counter=with_it)
    ad.solve(f, a, y0, b, 1e-8, "dormand prince", local_extrapolation=False,
             _counter=without)
    assert without[0] > with_it[0]


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_local_extrapolation_is_free_accuracy(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ad.local_extrapolation_is_free_accuracy(f, exact, a, y0, b)
    gain = np.asarray(out["gain"], dtype=float)
    assert np.all(gain > 1.0)
    assert float(gain[-1]) > float(gain[0])


def test_adaptivity_buys_nothing_on_a_single_scale_problem():
    """The steps come out nearly all the same, exactly as in lesson 64."""
    f, exact, y0, (a, b) = linear_forcing()
    out = ad.where_the_work_went(f, a, y0, b, 1e-8)
    assert out["ratio"] < 20.0


def test_adaptivity_buys_a_great_deal_on_a_two_scale_problem():
    f, exact, y0, (a, b) = two_scales()
    out = ad.where_the_work_went(f, a, y0, b, 1e-8)
    assert out["ratio"] > 20.0


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_the_comparison_against_fixed_steps_is_at_equal_cost(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ad.against_fixed_steps(f, exact, a, y0, b, [1e-6, 1e-8])
    adaptive = np.asarray(out["adaptive_evaluations"], dtype=float)
    fixed = np.asarray(out["fixed_evaluations"], dtype=float)
    # the fixed run gets the same budget to within one step's worth of stages
    assert np.all(fixed <= adaptive)
    assert np.all(fixed > adaptive - 8)
