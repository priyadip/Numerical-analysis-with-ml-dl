"""Tests for nalib.rungekutta.

Three groups. The tableaux are checked against their order conditions in exact arithmetic, which
is what catches a mistyped coefficient, and then against the order they actually achieve when
run, which is a different question and catches a different mistake. The classical structural
facts, that the weights sum to 1 and that c is the row sum of A, are checked for every tableau.
The stability limits are found by bisecting the growth polynomial and confirmed against the step
at which a real integration diverges.

Every test sweeps tableaux, problems and step counts.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from nalib import rungekutta as rk
from nalib.ivp import as_state, integrate

NAMES = sorted(rk.TABLEAUX)
STEP_COUNTS = [10, 20, 40, 80, 160, 320]

#: The classical stability limits on the negative real axis, to four places.
STABILITY = {1: 2.0, 2: 2.0, 3: 2.512745, 4: 2.785294}


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


# --------------------------------------------------------------------------- the tableaux


@pytest.mark.parametrize("name", NAMES)
def test_every_tableau_is_explicit(name):
    A, b, c, _order = rk.tableau(name)
    assert rk.is_explicit(A)
    assert A.shape == (b.size, b.size)
    assert c.size == b.size


@pytest.mark.parametrize("name", NAMES)
def test_the_weights_sum_to_one_exactly(name):
    """In exact rational arithmetic, so a rounded coefficient cannot slip through."""
    _A, b, _c, _order = rk.TABLEAUX[name]
    assert sum(Fraction(v) for v in b) == Fraction(1)


@pytest.mark.parametrize("name", NAMES)
def test_c_is_the_row_sum_of_a(name):
    """The consistency condition: stage i is evaluated at t + c_i h, so c_i must be sum_j a_ij."""
    A, b, c, _order = rk.TABLEAUX[name]
    for i, row in enumerate(A):
        assert sum(Fraction(v) for v in row) == Fraction(c[i])


@pytest.mark.parametrize("name", NAMES)
def test_every_tableau_satisfies_the_conditions_for_its_order(name):
    A, b, c, order = rk.tableau(name)
    out = rk.order_conditions(A, b, c, up_to=min(order, 4))
    assert out["achieved_order"] >= min(order, 4)
    assert bool(np.all(out["holds"]))


def test_the_whole_family_is_what_it_claims():
    out = rk.all_tableaux_are_what_they_claim()
    assert out["all_agree"]
    assert bool(np.all(out["explicit"]))
    for stages, order in zip(out["stages"], out["claimed_order"]):
        assert order <= stages


@pytest.mark.parametrize("name", NAMES)
def test_a_perturbed_coefficient_breaks_a_condition(name):
    """The order conditions exist to catch this, so it has to be shown that they do."""
    A, b, c, order = rk.tableau(name)
    if b.size < 2:
        pytest.skip("Euler has a single weight and nothing to perturb against")
    bad = b.copy()
    bad[0] += 0.01
    bad[1] -= 0.01
    out = rk.order_conditions(A, bad, c, up_to=min(order, 4))
    assert out["achieved_order"] < min(order, 4)


def test_an_unknown_tableau_is_rejected():
    with pytest.raises(ValueError):
        rk.tableau("not a method")


def test_an_implicit_tableau_is_refused():
    A = np.asarray([[0.5]])
    with pytest.raises(ValueError):
        rk.step_from(A, np.asarray([1.0]), np.asarray([0.5]))


def test_rk4_has_the_coefficients_everyone_knows():
    _A, b, c, order = rk.TABLEAUX["rk4"]
    assert [str(v) for v in b] == ["1/6", "1/3", "1/3", "1/6"]
    assert [str(v) for v in c] == ["0", "1/2", "1/2", "1"]
    assert order == 4


# --------------------------------------------------------------------------- measured order


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_each_method_achieves_its_order_when_run(name, kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]()
    out = rk.verified_order(f, exact, a, y0, b, name, STEP_COUNTS)
    assert out["matches"], f"{name} on {kind}: fitted {out['fitted_order']:.3f}"


@pytest.mark.parametrize("name", NAMES)
def test_the_cost_is_one_evaluation_per_stage_per_step(name):
    f, exact, y0, (a, b) = linear_forcing()
    A, weights, c, _order = rk.tableau(name)
    counter = [0]
    integrate(f, a, y0, b, 23, rk.named_step(name), counter)
    assert counter[0] == 23 * weights.size


@pytest.mark.parametrize("name", NAMES)
def test_every_method_reproduces_a_polynomial_of_its_own_degree(name):
    """y' = k t^(k-1) with y(0) = 0 has solution t^k, which a method of order k integrates
    exactly, because its error term involves the (k+1)th derivative and that is zero."""
    A, b, c, order = rk.tableau(name)
    k = order

    def f(t, y):
        return np.atleast_1d(k * t ** (k - 1)) if k > 1 else np.atleast_1d(1.0)

    out = integrate(f, 0.0, 0.0, 1.0, 7, rk.named_step(name))
    assert float(abs(out["y"][-1, 0] - 1.0)) < 1e-12


@pytest.mark.parametrize("kind", sorted(PROBLEMS))
def test_higher_order_wins_at_equal_cost(kind):
    f, exact, y0, (a, b) = PROBLEMS[kind]()
    out = rk.compare_at_equal_cost(f, exact, a, y0, b)
    assert out["highest_order_always_wins"]
    # and the margin grows with the budget
    ratios = out["errors"]["euler"] / np.maximum(out["errors"]["rk4"], 1e-300)
    assert np.all(np.diff(ratios) > 0.0)


def test_two_tableaux_of_the_same_order_are_indistinguishable_by_order_alone():
    """`rk4` and `three eighths` both have order 4 and different coefficients.

    The measured order cannot tell them apart, which is the point: order is not a complete
    description of a method. Their stability limits are also equal, and lesson 70 finds the
    property that does separate methods of the same order, which is the error estimate an
    embedded pair provides.
    """
    f, exact, y0, (a, b) = linear_forcing()
    one = rk.verified_order(f, exact, a, y0, b, "rk4", STEP_COUNTS)
    two = rk.verified_order(f, exact, a, y0, b, "three eighths", STEP_COUNTS)
    assert abs(one["fitted_order"] - two["fitted_order"]) < 0.05
    assert rk.stability_limit("rk4") == pytest.approx(rk.stability_limit("three eighths"))


# --------------------------------------------------------------------------- stability


@pytest.mark.parametrize("name", NAMES)
def test_the_stability_limit_matches_the_classical_value(name):
    _A, b, _c, order = rk.tableau(name)
    assert rk.stability_limit(name) == pytest.approx(STABILITY[order], abs=1e-5)


@pytest.mark.parametrize("name", ["euler", "heun", "kutta third", "rk4"])
def test_the_predicted_limit_is_where_the_solver_actually_fails(name):
    out = rk.stability_limit_is_where_the_solver_fails(name, lam=-50.0)
    assert out["agrees"]
    # the largest step that stays stable is within one grid spacing of the prediction
    assert out["largest_observed_stable_lam_h"] == pytest.approx(out["predicted_limit"],
                                                                rel=0.06)


def test_raising_the_order_buys_almost_nothing_in_stability():
    """The whole argument for lesson 72, as one number.

    Going from Euler to RK4 multiplies the stable step by 1.39 and the cost per step by 4, so on
    a stiff problem RK4 needs nearly three times as many evaluations as Euler to reach the same
    time. No explicit method escapes this.
    """
    out = rk.explicit_stability_is_not_the_answer(lam=-1000.0)
    assert out["spread"] == pytest.approx(1.3926, abs=0.01)
    evaluations = np.asarray(out["evaluations_needed"], dtype=float)
    names = list(out["names"])
    assert evaluations[names.index("rk4")] > evaluations[names.index("euler")]
    assert evaluations[names.index("rk4")] / evaluations[names.index("euler")] > 2.5


@pytest.mark.parametrize("lam", [-10.0, -100.0, -1000.0])
def test_the_stable_step_scales_inversely_with_the_stiffness(lam):
    out = rk.explicit_stability_is_not_the_answer(lam=lam)
    limits = np.asarray(out["stability_limit"], dtype=float)
    steps = np.asarray(out["largest_stable_step"], dtype=float)
    assert np.max(np.abs(steps * abs(lam) - limits)) < 1e-9


# ------------------------------------------------------------------ exact order conditions


@pytest.mark.parametrize("name", sorted(rk.TABLEAUX))
def test_the_exact_tableau_has_the_same_numbers_as_the_float_one(name):
    A, b, c, order = rk.tableau_exact(name)
    fA, fb, fc, forder = rk.tableau(name)
    assert order == forder
    assert np.allclose([[float(v) for v in row] for row in A], fA, rtol=0, atol=0)
    assert np.allclose([float(v) for v in b], fb, rtol=0, atol=0)
    assert np.allclose([float(v) for v in c], fc, rtol=0, atol=0)


def test_an_unknown_exact_tableau_is_refused():
    with pytest.raises(ValueError):
        rk.tableau_exact("dormand prince")


@pytest.mark.parametrize("name", sorted(rk.TABLEAUX))
def test_the_conditions_a_tableau_satisfies_have_an_exactly_zero_miss(name):
    """Not a small miss. Zero, as a Fraction, which floating point cannot report."""
    A, b, c, order = rk.tableau_exact(name)
    out = rk.order_conditions(A, b, c, up_to=5)
    assert out["exact"]
    for level, miss, holds in zip(out["order"], out["residual"], out["holds"]):
        if level <= order:
            assert holds
            assert miss == 0
            assert isinstance(miss, Fraction)


@pytest.mark.parametrize("name", sorted(rk.TABLEAUX))
def test_the_first_condition_a_tableau_fails_misses_by_an_exact_nonzero(name):
    A, b, c, order = rk.tableau_exact(name)
    out = rk.order_conditions(A, b, c, up_to=5)
    if order >= 5:
        return
    failing = [m for lvl, m, h in zip(out["order"], out["residual"], out["holds"])
               if lvl == order + 1 and not h]
    assert failing, f"{name} claims order {order} but satisfies order {order + 1} too"
    for miss in failing:
        assert miss != 0


def test_rk4_misses_the_order_five_conditions_by_a_twentieth_of_a_fourth():
    """The worst miss is exactly -1/80, which is not a rounding artefact."""
    out = rk.order_conditions(*rk.tableau_exact("rk4")[:3], up_to=5)
    fifth = [m for lvl, m in zip(out["order"], out["residual"]) if lvl == 5]
    assert len(fifth) == 9
    assert all(m != 0 for m in fifth)
    assert min(fifth) == Fraction(-1, 80)
    assert out["achieved_order"] == 4


def test_the_order_five_conditions_hold_for_the_order_five_pairs():
    """The check on the conditions themselves: a genuine order 5 method must satisfy all nine."""
    from nalib import adaptivestep as ad

    for name in ("dormand prince", "fehlberg"):
        A, bh, bl, c, ph, pl = ad.pair(name)
        assert ph == 5
        out = rk.order_conditions(A, bh, c, up_to=5)
        assert not out["exact"]              # the pairs are stored as floats
        assert out["achieved_order"] == 5
        assert float(np.max(np.abs([float(v) for v in out["residual"]]))) < 1e-14


@pytest.mark.parametrize("name", ["heun euler", "bogacki shampine"])
def test_the_lower_order_pairs_do_not_reach_five(name):
    from nalib import adaptivestep as ad

    A, bh, bl, c, ph, pl = ad.pair(name)
    out = rk.order_conditions(A, bh, c, up_to=5)
    assert out["achieved_order"] == ph


def test_floating_point_cannot_report_an_exact_zero():
    """Why the exact path exists: RK4 sums its weights to 0.9999999999999999."""
    fA, fb, fc, _ = rk.tableau("rk4")
    out = rk.order_conditions(fA, fb, fc, up_to=4)
    assert not out["exact"]
    first = float(out["computed"][0])
    assert first != 1.0
    assert abs(first - 1.0) < 1e-15
    # and the exact path gets it right
    exact = rk.order_conditions(*rk.tableau_exact("rk4")[:3], up_to=4)
    assert exact["computed"][0] == 1


def test_both_paths_reach_the_same_verdict_on_every_tableau():
    out = rk.all_tableaux_are_what_they_claim()
    assert out["all_agree"]
    assert out["exact_and_float_agree"]
    assert bool(np.all(out["checked_exactly"]))


def test_the_worst_float_residual_does_not_separate_pass_from_fail():
    """The number a float check would have to threshold on spans 0.5 down to 1e-16."""
    out = rk.all_tableaux_are_what_they_claim()
    residuals = np.asarray(out["worst_float_residual"], dtype=float)
    assert float(np.max(residuals)) > 0.4
    assert float(np.min(residuals)) < 1e-15


def test_the_cheapest_method_at_the_stability_limit_is_the_lowest_order_one():
    out = rk.explicit_stability_is_not_the_answer(lam=-1000.0, t_end=1.0)
    assert out["cheapest"] == "euler"
    assert out["cheapest_is_the_lowest_order"]
    assert out["spread"] == pytest.approx(2.785294 / 2.0, rel=1e-4)
    assert out["evaluation_spread"] == pytest.approx(1440.0 / 500.0, rel=1e-6)
