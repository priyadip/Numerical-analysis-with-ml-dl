"""Tests for nalib.multistep.

Three groups. The coefficients are checked against the classical fractions in exact rational
arithmetic. The order and root conditions are checked for every named method, including the one
that is high order and useless, whose failure is the whole reason the root condition exists. The
methods are then run, and the start up dependence is measured because it is the quiet way a
correct method delivers the wrong order.

Every test sweeps methods, orders and starters.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from nalib import multistep as ms
from nalib.ivp import as_state

NAMES = sorted(ms.NAMED)
EXPLICIT = [n for n in NAMES if ms.is_explicit(n) and n != "unstable order two"]
STEP_COUNTS = [20, 40, 80, 160, 320, 640]

#: The coefficients everyone has seen, as strings so an exact comparison is possible.
CLASSICAL_AB = {1: ["1"], 2: ["3/2", "-1/2"], 3: ["23/12", "-4/3", "5/12"],
                4: ["55/24", "-59/24", "37/24", "-3/8"]}
CLASSICAL_AM = {0: ["1"], 1: ["1/2", "1/2"], 2: ["5/12", "2/3", "-1/12"],
                3: ["3/8", "19/24", "-5/24", "1/24"]}


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

    return f, exact, [1.0, 0.0], (0.0, 1.0)


PROBLEMS = {"linear forcing": linear_forcing, "harmonic": harmonic}


# --------------------------------------------------------------------------- coefficients


@pytest.mark.parametrize("k", sorted(CLASSICAL_AB))
def test_adams_bashforth_matches_the_classical_fractions(k):
    got = ms.adams_bashforth_coefficients(k)
    assert [str(v) for v in got] == CLASSICAL_AB[k]


@pytest.mark.parametrize("k", sorted(CLASSICAL_AM))
def test_adams_moulton_matches_the_classical_fractions(k):
    got = ms.adams_moulton_coefficients(k)
    assert [str(v) for v in got] == CLASSICAL_AM[k]


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_both_families_have_coefficients_summing_to_one(k):
    """The method must reproduce the integral of a constant, whatever its order."""
    assert sum(ms.adams_bashforth_coefficients(k)) == Fraction(1)
    assert sum(ms.adams_moulton_coefficients(k)) == Fraction(1)


def test_the_one_point_adams_moulton_rule_is_the_trapezoid_rule():
    assert [str(v) for v in ms.adams_moulton_coefficients(1)] == ["1/2", "1/2"]


def test_zero_past_points_is_rejected_for_the_explicit_family():
    with pytest.raises(ValueError):
        ms.adams_bashforth_coefficients(0)
    with pytest.raises(ValueError):
        ms.adams_moulton_coefficients(-1)


# --------------------------------------------------------------------------- order and roots


@pytest.mark.parametrize("name", NAMES)
def test_every_named_method_has_the_order_it_claims(name):
    alpha, beta, claimed = ms.method(name)
    out = ms.order_conditions(alpha, beta)
    assert out["order"] == claimed


@pytest.mark.parametrize("name", NAMES)
def test_every_named_method_is_consistent(name):
    """Order at least 1 means the method solves the right differential equation."""
    alpha, beta, _claimed = ms.method(name)
    out = ms.order_conditions(alpha, beta)
    assert out["order"] >= 1
    assert bool(out["holds"][0]) and bool(out["holds"][1])


@pytest.mark.parametrize("name", NAMES)
def test_consistency_puts_a_root_at_one(name):
    alpha, _beta, _claimed = ms.method(name)
    out = ms.root_condition(alpha)
    assert out["has_the_principal_root"]


@pytest.mark.parametrize("name", [n for n in NAMES if n != "unstable order two"])
def test_every_usable_method_is_zero_stable(name):
    alpha, _beta, _claimed = ms.method(name)
    assert ms.root_condition(alpha)["zero_stable"]


def test_the_unstable_example_is_high_order_and_useless():
    """Order says nothing about convergence without zero-stability."""
    out = ms.unstable_example()
    assert out["consistent"]
    assert out["measured_order"] == 3
    assert not out["zero_stable"]
    assert out["largest_modulus"] == pytest.approx(5.0, abs=1e-9)


def test_the_unstable_method_gets_worse_when_refined():
    """No convergent method does this, which makes it the clearest possible symptom."""
    out = ms.unstable_method_diverges()
    assert out["gets_worse_with_refinement"]
    errors = np.asarray(out["errors"], dtype=float)
    assert float(errors[-1]) > 1e30


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        ms.method("not a method")


def test_the_first_barrier_is_k_plus_one_or_two():
    out = ms.first_dahlquist_barrier(8)
    for k, maximum in zip(out["steps"], out["maximum_order"]):
        assert maximum == (k + 2 if k % 2 == 0 else k + 1)


@pytest.mark.parametrize("name", NAMES)
def test_no_zero_stable_method_here_exceeds_its_barrier(name):
    alpha, beta, claimed = ms.method(name)
    if not ms.root_condition(alpha)["zero_stable"]:
        pytest.skip("the barrier applies to zero-stable methods")
    k = max(alpha.size, beta.size) - 1
    assert claimed <= k + (2 if k % 2 == 0 else 1)


# --------------------------------------------------------------------------- running them


@pytest.mark.parametrize("name", EXPLICIT)
@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_each_explicit_method_achieves_its_order(name, kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ms.measured_order(f, exact, a, y0, b, name, STEP_COUNTS)
    assert out["matches"], f"{name} on {kind}: fitted {out['fitted_order']:.3f}"


@pytest.mark.parametrize("name", EXPLICIT)
def test_one_evaluation_per_step_after_the_start_up(name):
    """The reason multistep methods exist: the order is free after the first few points."""
    f, exact, y0, (a, b) = linear_forcing()
    alpha, beta, _order = ms.method(name)
    k = max(alpha.size, beta.size) - 1
    counter = [0]
    out = ms.solve_explicit(f, a, y0, b, 50, name, None, counter)
    # k start up points, then one per remaining step, plus the RK4 starter's four per point
    assert counter[0] <= 50 + 5 * k
    assert out["start_up_points"] == k


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_the_starting_order_decides_the_achieved_order(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ms.starting_values_matter(f, exact, a, y0, b, "ab4", STEP_COUNTS)
    assert out["matches_min_p_q_plus_one"]
    fitted = np.asarray(out["fitted_order"], dtype=float)
    assert float(fitted[0]) < 2.5
    assert float(fitted[-1]) == pytest.approx(4.0, abs=0.35)


def test_the_predictor_corrector_costs_two_evaluations_per_step():
    f, exact, y0, (a, b) = linear_forcing()
    counter = [0]
    ms.solve_predictor_corrector(f, a, y0, b, 50, "ab4", "am4", 1, None, counter)
    assert 2 * 50 <= counter[0] <= 2 * 50 + 25


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_the_predictor_corrector_reaches_the_correctors_order(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]()
    want = as_state(exact(b))
    errors, counts = [], []
    for n in STEP_COUNTS:
        out = ms.solve_predictor_corrector(f, a, y0, b, n, "ab4", "am4")
        errors.append(float(np.max(np.abs(out["y"][-1] - want))))
        counts.append(float(n))
    errors = np.asarray(errors)
    keep = errors > 1e-13
    order = float(-np.polyfit(np.log(np.asarray(counts)[keep]), np.log(errors[keep]), 1)[0])
    assert order == pytest.approx(4.0, abs=0.35)


def test_an_implicit_method_cannot_be_run_explicitly():
    f, exact, y0, (a, b) = linear_forcing()
    with pytest.raises(ValueError):
        ms.solve_explicit(f, a, y0, b, 20, "am4")


def test_the_implicit_error_constant_is_the_smaller_one():
    out = ms.implicit_error_constant([2, 3, 4, 5])
    assert out["implicit_is_smaller"]
    ratios = np.asarray(out["ratio"], dtype=float)
    assert np.all(np.diff(ratios) > 0.0)
    assert float(ratios[0]) == pytest.approx(5.0, rel=0.02)


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_multistep_beats_runge_kutta_at_equal_cost(kind):
    """AB4 takes four times as many steps as RK4 for the same evaluations, and both are order 4."""
    f, exact, y0, (a, b) = PROBLEMS[kind]()
    out = ms.against_runge_kutta(f, exact, a, y0, b)
    assert np.all(np.asarray(out["ab4_error"]) < np.asarray(out["rk4_error"]))


# ------------------------------------------------------------------ the BDF family


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_the_bdf_generator_reproduces_the_named_methods(k):
    """BDF1 is backward Euler and BDF2 and BDF3 are already in the table, so they must match."""
    alpha, beta = ms.bdf_coefficients(k)
    known = {1: "backward euler", 2: "bdf2", 3: "bdf3"}
    if k not in known:
        return
    want_alpha, want_beta, order = ms.method(known[k])
    assert len(alpha) == len(want_alpha)
    for got, want in zip(alpha, want_alpha):
        assert float(got) == pytest.approx(float(want), rel=1e-14)
    for got, want in zip(beta, list(want_beta) + [0] * len(alpha)):
        assert float(got) == pytest.approx(float(want), abs=1e-14)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7])
def test_every_bdf_method_has_order_equal_to_its_step_count(k):
    alpha, beta = ms.bdf_coefficients(k)
    out = ms.order_conditions(alpha, beta, up_to=k + 2)
    assert out["order"] == k


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7])
def test_only_the_first_beta_of_a_bdf_method_is_nonzero(k):
    """Which is what makes it implicit in one point and cheap to solve."""
    alpha, beta = ms.bdf_coefficients(k)
    assert beta[0] != 0
    assert all(v == 0 for v in beta[1:])


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_the_bdf_methods_up_to_six_are_zero_stable_and_seven_is_not(k):
    alpha, beta = ms.bdf_coefficients(k)
    assert ms.root_condition(alpha)["zero_stable"]
    seven, _ = ms.bdf_coefficients(7)
    assert not ms.root_condition(seven)["zero_stable"]


def test_a_bdf_of_zero_steps_is_refused():
    with pytest.raises(ValueError):
        ms.bdf_coefficients(0)


def test_the_bdf_coefficients_are_exact_rationals():
    alpha, beta = ms.bdf_coefficients(3)
    assert all(isinstance(v, Fraction) for v in alpha)
    assert alpha[1] == Fraction(-18, 11)
    assert alpha[2] == Fraction(9, 11)
    assert alpha[3] == Fraction(-2, 11)
    assert beta[0] == Fraction(6, 11)
