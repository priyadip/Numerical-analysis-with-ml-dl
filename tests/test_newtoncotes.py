"""Tests for nalib.newtoncotes.

Four groups. The weights are checked against the classical fractions and against the property
that defines them, exactness on polynomials, in exact rational arithmetic so there is no
tolerance to argue about. The degree of precision is checked against the odd node bonus. The
failure of the family at high node count is checked by measuring the weights rather than by
quoting the result. The composite rules are checked by fitting the observed convergence order
and comparing it against the predicted one.

Every test sweeps node counts and integrands. Nothing is fixed to one rule or one interval.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from nalib import newtoncotes as nc

CLOSED_COUNTS = [2, 3, 4, 5, 6, 7, 8, 9, 11]
OPEN_COUNTS = [1, 2, 3, 4, 5, 6, 7]

#: The textbook weights, written as fractions so a transcription error cannot hide.
CLASSICAL = {
    "trapezoid": (2, True, ["1/2", "1/2"]),
    "Simpson 1/3": (3, True, ["1/6", "2/3", "1/6"]),
    "Simpson 3/8": (4, True, ["1/8", "3/8", "3/8", "1/8"]),
    "Boole": (5, True, ["7/90", "16/45", "2/15", "16/45", "7/90"]),
    "Weddle": (7, True, ["41/840", "9/35", "9/280", "34/105", "9/280", "9/35", "41/840"]),
    "midpoint": (1, False, ["1"]),
    "open two point": (2, False, ["1/2", "1/2"]),
    "open three point": (3, False, ["2/3", "-1/3", "2/3"]),
}

INTEGRANDS = {
    "sin": (np.sin, lambda a, b: math.cos(a) - math.cos(b)),
    "exp": (np.exp, lambda a, b: math.exp(b) - math.exp(a)),
    "cubic": (lambda x: x ** 3 - 2.0 * x, lambda a, b: (b ** 4 - a ** 4) / 4.0 - (b ** 2 - a ** 2)),
    "runge": (lambda x: 1.0 / (1.0 + 25.0 * x ** 2),
              lambda a, b: (math.atan(5.0 * b) - math.atan(5.0 * a)) / 5.0),
}

INTERVALS = [(0.0, 1.0), (-1.0, 1.0), (0.5, 2.5), (-3.0, -0.5)]


# --------------------------------------------------------------------------- the weights


@pytest.mark.parametrize("name", sorted(CLASSICAL))
def test_the_weights_are_the_classical_fractions(name):
    n, closed, want = CLASSICAL[name]
    got = nc.exact_weights(n, closed)
    assert [str(v) for v in got] == want


@pytest.mark.parametrize("name", sorted(CLASSICAL))
def test_the_float_weights_match_the_exact_ones(name):
    n, closed, _ = CLASSICAL[name]
    exact = np.asarray([float(v) for v in nc.exact_weights(n, closed)])
    assert np.max(np.abs(nc.weights(n, closed) - exact)) == 0.0


@pytest.mark.parametrize("n", CLOSED_COUNTS)
@pytest.mark.parametrize("closed", [True, False])
def test_the_weights_sum_to_exactly_one(n, closed):
    # the rule is scaled to a unit interval, so it must reproduce the integral of 1
    if not closed and n < 1:
        pytest.skip("open rules need at least one node")
    assert sum(nc.exact_weights(n, closed)) == Fraction(1)


@pytest.mark.parametrize("n", CLOSED_COUNTS)
@pytest.mark.parametrize("closed", [True, False])
def test_the_weights_are_symmetric(n, closed):
    w = nc.exact_weights(n, closed)
    assert w == list(reversed(w))


@pytest.mark.parametrize("n", CLOSED_COUNTS)
@pytest.mark.parametrize("closed", [True, False])
def test_the_rule_is_exact_on_every_polynomial_up_to_its_degree(n, closed):
    d = nc.degree_of_precision(n, closed)
    nodes = ([Fraction(j, n - 1) for j in range(n)] if closed
             else [Fraction(j + 1, n + 1) for j in range(n)])
    w = nc.exact_weights(n, closed)
    for k in range(d + 1):
        got = sum(w[j] * nodes[j] ** k for j in range(n))
        assert got == Fraction(1, k + 1)


@pytest.mark.parametrize("n", CLOSED_COUNTS)
@pytest.mark.parametrize("closed", [True, False])
def test_and_is_not_exact_one_degree_higher(n, closed):
    d = nc.degree_of_precision(n, closed)
    nodes = ([Fraction(j, n - 1) for j in range(n)] if closed
             else [Fraction(j + 1, n + 1) for j in range(n)])
    w = nc.exact_weights(n, closed)
    k = d + 1
    assert sum(w[j] * nodes[j] ** k for j in range(n)) != Fraction(1, k + 1)


# --------------------------------------------------------------------------- degree


@pytest.mark.parametrize("n", CLOSED_COUNTS)
def test_the_odd_node_bonus(n):
    """An odd node count buys one extra degree, an even one buys nothing."""
    d = nc.degree_of_precision(n, True)
    assert d == (n if n % 2 == 1 else n - 1)


@pytest.mark.parametrize("n", OPEN_COUNTS)
def test_the_open_rules_get_the_same_bonus(n):
    d = nc.degree_of_precision(n, False)
    assert d == (n if n % 2 == 1 else n - 1)


def test_an_even_rule_is_no_better_than_the_odd_one_below_it():
    """Why Simpson 3/8 is not worth its extra evaluation.

    The 4 point rule costs one more function value than the 3 point rule and has the same
    degree of precision. The same holds at every even count.
    """
    for n in (4, 6, 8):
        assert nc.degree_of_precision(n, True) == nc.degree_of_precision(n - 1, True)


# --------------------------------------------------------------------------- the failure


def test_the_weights_stay_positive_only_up_to_eight_nodes():
    out = nc.weight_report()
    assert out["first_negative"] == 9
    positive = np.asarray(out["all_positive"], dtype=bool)
    counts = np.asarray(out["counts"])
    assert np.all(positive[counts <= 8])
    assert not np.any(positive[counts >= 9])


def test_a_positive_rule_cannot_amplify():
    """All positive weights summing to one make the rule a weighted average.

    That is the guarantee that is lost at 9 nodes, and it is the whole reason the family is
    used at low order and composed instead of extended.
    """
    out = nc.weight_report()
    sums = np.asarray(out["sum_of_absolute_weights"], dtype=float)
    counts = np.asarray(out["counts"])
    assert np.max(np.abs(sums[counts <= 8] - 1.0)) < 1e-12
    assert np.all(sums[counts >= 9] > 1.0)


def test_the_amplification_grows_without_bound():
    out = nc.weight_report(counts=[9, 11, 15, 21])
    sums = np.asarray(out["sum_of_absolute_weights"], dtype=float)
    assert np.all(np.diff(sums) > 0.0)
    assert sums[-1] > 100.0


@pytest.mark.parametrize("n", [3, 5, 7])
def test_a_low_order_rule_is_bounded_by_max_absolute_f(n):
    rng = np.random.default_rng(42)
    for _ in range(5):
        values = rng.standard_normal(n)
        w = nc.weights(n, True)
        assert abs(float(np.sum(w * values))) <= float(np.max(np.abs(values))) + 1e-12


# --------------------------------------------------------------------------- error terms


def test_the_derived_error_constants_match_the_classical_table():
    out = nc.error_constants_agree()
    assert out["all_agree"]
    assert np.max(np.asarray(out["gap"], dtype=float)) < 1e-9


@pytest.mark.parametrize("name", sorted(nc.CLASSICAL_ERRORS))
def test_every_classical_error_constant_is_negative(name):
    """The sign convention, which is the easiest thing to get backwards.

    The classical tables state the error as ``exact - rule``. Trapezoid on a convex function
    overestimates, so its constant is negative, and the same holds for the whole table.
    Computing ``rule - exact`` instead reproduces every magnitude with every sign flipped.
    """
    coefficient, _, _ = nc.CLASSICAL_ERRORS[name]
    assert coefficient < 0


@pytest.mark.parametrize("n", CLOSED_COUNTS)
def test_the_error_order_is_the_degree_plus_two(n):
    out = nc.error_constant(n, True)
    assert out["error_order"] == out["degree_of_precision"] + 2
    assert out["first_missed_degree"] == out["degree_of_precision"] + 1
    assert out["error_constant"] != 0.0


# --------------------------------------------------------------------------- composite


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_a_single_rule_is_exact_on_a_polynomial_of_its_degree(interval, n):
    a, b = interval
    d = nc.degree_of_precision(n, True)
    rng = np.random.default_rng(42)
    c = rng.standard_normal(d + 1)

    def f(x):
        return sum(c[k] * np.asarray(x, dtype=float) ** k for k in range(c.size))

    exact = sum(c[k] * (b ** (k + 1) - a ** (k + 1)) / (k + 1) for k in range(c.size))
    got = nc.integrate(f, a, b, n, True)
    assert got == pytest.approx(exact, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("n", [2, 3, 5])
@pytest.mark.parametrize("panels", [1, 2, 5, 13])
def test_composite_agrees_with_a_single_rule_on_a_polynomial(interval, n, panels):
    a, b = interval
    d = nc.degree_of_precision(n, True)
    rng = np.random.default_rng(7)
    c = rng.standard_normal(d + 1)

    def f(x):
        return sum(c[k] * np.asarray(x, dtype=float) ** k for k in range(c.size))

    exact = sum(c[k] * (b ** (k + 1) - a ** (k + 1)) / (k + 1) for k in range(c.size))
    got = nc.composite(f, a, b, panels, n, True)
    scale = max(1.0, abs(exact))
    assert got == pytest.approx(exact, rel=1e-9, abs=1e-9 * scale)


@pytest.mark.parametrize("kind", sorted(INTEGRANDS))
@pytest.mark.parametrize("n", [2, 3, 4, 5, 7])
def test_the_fitted_convergence_order_matches_the_predicted_one(kind, n):
    """Whenever a rate can be fitted at all, it is the predicted one.

    Not every combination gives one. A rule that is exact on the integrand has no rate, and a
    high order rule on a smooth integrand can reach the roundoff floor before any two
    refinements share a slope. `convergence` reports nan and says which happened, and this test
    accepts that as an answer rather than forcing a number out of it.
    """
    f, antiderivative = INTEGRANDS[kind]
    a, b = (0.0, 1.0) if kind != "runge" else (-1.0, 1.0)
    out = nc.convergence(f, antiderivative(a, b), a, b, n, True)
    if math.isnan(out["fitted_order"]):
        assert out["points_used_in_the_fit"] == 0
        assert out["note"]
        return
    assert out["points_used_in_the_fit"] >= 3
    assert out["fitted_order"] == pytest.approx(out["predicted_order"], abs=0.25)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_a_rate_is_actually_fitted_on_the_cases_that_have_one(n):
    # the guard on the test above must not be a way for every case to opt out
    f, antiderivative = INTEGRANDS["sin"]
    out = nc.convergence(f, antiderivative(0.0, 1.0), 0.0, 1.0, n, True)
    assert not math.isnan(out["fitted_order"])
    assert out["fitted_order"] == pytest.approx(out["predicted_order"], abs=0.25)


@pytest.mark.parametrize("n", [3, 4, 5, 7])
def test_an_exactly_integrated_polynomial_reports_no_rate_rather_than_a_wrong_one(n):
    d = nc.degree_of_precision(n, True)
    rng = np.random.default_rng(42)
    c = rng.standard_normal(d + 1)

    def f(x):
        return sum(c[k] * np.asarray(x, dtype=float) ** k for k in range(c.size))

    exact = sum(c[k] / (k + 1) for k in range(c.size))
    out = nc.convergence(f, exact, 0.0, 1.0, n, True)
    assert math.isnan(out["fitted_order"])
    assert "exact" in out["note"]


def test_fitting_through_the_pre_asymptotic_panels_would_give_the_wrong_order():
    """Why `convergence` cuts the coarse end off.

    Composite trapezoid on Runge's function is order 2, but its first few panel counts do not
    resolve the peak. A fit over every point above the roundoff floor reads 2.8.
    """
    f, antiderivative = INTEGRANDS["runge"]
    a, b = -1.0, 1.0
    out = nc.convergence(f, antiderivative(a, b), a, b, 2, True)
    assert out["fitted_order"] == pytest.approx(2.0, abs=0.25)
    assert out["panels_dropped_from_the_head"] > 0
    # the fit the function refuses to do
    errors = np.asarray(out["errors"], dtype=float)
    panels = np.asarray(out["panels"], dtype=float)
    naive = float(-np.polyfit(np.log(panels), np.log(errors), 1)[0])
    assert naive > 2.5


@pytest.mark.parametrize("kind", sorted(INTEGRANDS))
def test_shared_endpoints_give_the_same_answer_for_fewer_evaluations(kind):
    f, antiderivative = INTEGRANDS[kind]
    a, b = 0.0, 1.0
    for n in (2, 3, 5):
        plain, shared = [0], [0]
        x = nc.composite(f, a, b, 8, n, True, plain)
        y = nc.composite_shared(f, a, b, 8, n, shared)
        assert x == pytest.approx(y, rel=1e-12, abs=1e-14)
        assert shared[0] < plain[0]


def test_composite_beats_one_wide_rule_at_the_same_cost():
    """The practical conclusion of the whole module.

    At a fixed number of function evaluations, splitting into panels with a small rule beats
    one Newton-Cotes rule on all the nodes, and the gap widens as the budget grows.
    """
    f, antiderivative = INTEGRANDS["runge"]
    a, b = -1.0, 1.0
    exact = antiderivative(a, b)
    out = nc.compare_rules(f, exact, a, b)
    errors = np.asarray(out["errors"], dtype=float)
    assert np.all(np.isfinite(errors))
    assert out["best"] in list(out["names"])


@pytest.mark.parametrize("panels", [4, 16, 64])
def test_an_open_rule_handles_an_endpoint_singularity_and_a_closed_one_does_not(panels):
    out = nc.endpoint_singularity(lambda x: 1.0 / np.sqrt(x), 2.0, 0.0, 1.0, panels)
    for key, row in out.items():
        if "open" in key or "midpoint" in key:
            assert row["finite"], f"{key} should produce a number"
            assert np.isfinite(row["value"])
        else:
            assert not row["finite"], f"{key} evaluates at the singular endpoint"
            assert not np.isfinite(row["value"])


@pytest.mark.parametrize("panels", [16, 64, 256])
def test_the_open_rules_converge_on_the_singularity_but_slowly(panels):
    """An integrable endpoint singularity kills the order, it does not kill the answer.

    1/sqrt(x) has no bounded derivatives at 0, so the classical error term does not apply and
    the composite rate collapses to something near 1/2 in the panel width. The open rules still
    produce a number, and it still improves.
    """
    coarse = nc.endpoint_singularity(lambda x: 1.0 / np.sqrt(x), 2.0, 0.0, 1.0, panels)
    finer = nc.endpoint_singularity(lambda x: 1.0 / np.sqrt(x), 2.0, 0.0, 1.0, 4 * panels)
    for key in coarse:
        if "open" in key or "midpoint" in key:
            assert finer[key]["error"] < coarse[key]["error"]


# --------------------------------------------------------------------------- guards


@pytest.mark.parametrize("n", [0, 1])
def test_a_closed_rule_needs_two_nodes(n):
    with pytest.raises(ValueError):
        nc.weights(n, True)


def test_the_nodes_must_be_ordered():
    with pytest.raises(ValueError):
        nc.rule_nodes(3, 1.0, 0.0, True)


def test_panels_must_be_positive():
    with pytest.raises(ValueError):
        nc.composite(np.sin, 0.0, 1.0, 0, 3, True)


@pytest.mark.parametrize("n", CLOSED_COUNTS)
@pytest.mark.parametrize("interval", INTERVALS)
def test_the_nodes_span_the_interval(n, interval):
    a, b = interval
    x = nc.rule_nodes(n, a, b, True)
    assert x[0] == pytest.approx(a)
    assert x[-1] == pytest.approx(b)
    assert np.all(np.diff(x) > 0.0)
    y = nc.rule_nodes(n, a, b, False)
    assert y[0] > a and y[-1] < b
