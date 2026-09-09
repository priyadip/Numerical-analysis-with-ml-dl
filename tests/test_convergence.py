"""Tests for nalib.convergence.

The important tests are the ones that feed in sequences with a KNOWN order and check that
the measurement recovers it. Everything in this course that claims a convergence rate is
checked by these routines, so they need to be right.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nalib import convergence as cv


# ---------------------------------------------------------------- helpers


def synthetic_errors(p, C, e0=0.1, n=8, floor=0.0):
    """Build a sequence with exactly the given order and rate: e_{k+1} = C e_k^p."""
    out = [e0]
    for _ in range(n - 1):
        nxt = C * out[-1] ** p
        if nxt < floor:
            break
        out.append(nxt)
    return np.array(out)


# ---------------------------------------------------------------- order estimation


@pytest.mark.parametrize(
    "p, C",
    [(1.0, 0.5), (1.0, 0.9), (1.5, 0.8), (1.618, 1.0), (2.0, 1.0), (2.0, 0.3), (3.0, 1.0)],
)
def test_observed_order_recovers_a_synthetic_sequence(p, C):
    errs = synthetic_errors(p, C, e0=0.1, n=8)
    orders = cv.observed_order(errs)
    assert orders.size >= 1
    assert float(orders[-1]) == pytest.approx(p, rel=1e-6)


def test_observed_order_needs_three_points():
    assert cv.observed_order([1.0]).size == 0
    assert cv.observed_order([1.0, 0.5]).size == 0
    assert cv.observed_order([1.0, 0.5, 0.25]).size == 1


def test_observed_order_returns_one_estimate_per_triple():
    errs = synthetic_errors(2.0, 1.0, e0=0.5, n=6)
    assert cv.observed_order(errs).size == errs.size - 2


def test_observed_order_truncates_at_a_zero():
    """Everything after the error hits zero is roundoff noise and must be discarded."""
    errs = [0.1, 0.01, 1e-4, 1e-8, 0.0, 1e-16, 0.0]
    orders = cv.observed_order(errs)
    # only the first four entries are usable, giving two estimates
    assert orders.size == 2
    assert np.allclose(orders, 2.0, rtol=1e-9)


def test_observed_order_truncates_at_non_finite():
    errs = [0.1, 0.01, 1e-4, np.nan, 1e-8]
    assert cv.observed_order(errs).size == 1


def test_observed_order_handles_all_zeros():
    assert cv.observed_order([0.0, 0.0, 0.0]).size == 0


def test_observed_order_from_iterates():
    """Using successive differences instead of true errors should give the same order.

    Only 5 Newton steps are taken on purpose. Run it longer and the iterates stop changing
    except by one ulp, so the differences plateau near 2.2e-16 and oscillate instead of
    reaching exactly zero. `_usable_prefix` cannot truncate an oscillating plateau, so the
    final estimates become noise. That is the "late steps lie" warning from lesson 07, and
    the cure is to stop measuring before the roundoff floor, which is what this does.
    """
    x, iterates = 1.0, []
    for _ in range(5):
        x = x - (x * x - 2.0) / (2.0 * x)
        iterates.append(x)
    orders = cv.observed_order_from_iterates([1.0] + iterates)
    assert float(orders[-1]) == pytest.approx(2.0, abs=0.05)


def test_observed_order_degrades_once_the_plateau_is_reached():
    """Documenting the failure mode, so nobody is surprised by it later."""
    x, iterates = 1.0, []
    for _ in range(9):
        x = x - (x * x - 2.0) / (2.0 * x)
        iterates.append(x)
    orders = cv.observed_order_from_iterates([1.0] + iterates)
    # the middle estimates are good
    assert float(orders[2]) == pytest.approx(2.0, abs=0.05)
    # the last ones are not: on an exactly flat plateau the ratio is 1, so the formula
    # computes log(1)/log(1) = 0/0 and returns NaN. NaN is the honest answer here, since
    # a flat sequence carries no information about a rate.
    last = float(orders[-1])
    assert (not math.isfinite(last)) or abs(last - 2.0) > 0.5
    assert not np.isfinite(orders[-2:]).all(), "the plateau should produce NaN estimates"


# ---------------------------------------------------------------- real methods


def test_newton_measures_quadratic():
    target = math.sqrt(2.0)
    x, errs = 1.0, []
    for _ in range(6):
        x = x - (x * x - 2.0) / (2.0 * x)
        errs.append(abs(x - target))
    orders = cv.observed_order(errs)
    assert float(orders[-1]) == pytest.approx(2.0, abs=0.05)


def test_secant_measures_golden_ratio():
    target = math.sqrt(2.0)
    f = lambda t: t * t - 2.0
    x0, x1, errs = 1.0, 2.0, []
    for _ in range(10):
        f0, f1 = f(x0), f(x1)
        if f1 == f0:
            break
        x0, x1 = x1, x1 - f1 * (x1 - x0) / (f1 - f0)
        errs.append(abs(x1 - target))
    orders = cv.observed_order(errs)
    golden = (1 + math.sqrt(5)) / 2
    assert float(orders[-1]) == pytest.approx(golden, abs=0.1)


def test_bisection_bracket_halves_exactly():
    target = math.sqrt(2.0)
    f = lambda t: t * t - 2.0
    a, b, bounds = 1.0, 2.0, []
    for _ in range(50):
        m = 0.5 * (a + b)
        bounds.append(0.5 * (b - a))
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    ratios = np.array(bounds[1:]) / np.array(bounds[:-1])
    np.testing.assert_allclose(ratios, 0.5)
    assert cv.linear_rate(bounds) == pytest.approx(0.5, abs=1e-6)


def test_bisection_error_is_not_monotone():
    """The reason linear_rate fits instead of taking a ratio. Documented behaviour."""
    target = math.sqrt(2.0)
    f = lambda t: t * t - 2.0
    a, b, errs = 1.0, 2.0, []
    for _ in range(30):
        m = 0.5 * (a + b)
        errs.append(abs(m - target))
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    errs = np.array(errs)
    ratios = errs[1:20] / errs[:19]
    assert ratios.max() > 1.5, "some bisection steps make the error worse"
    assert ratios.min() < 0.1, "and some make it much better"
    # but the fitted decay is still 0.5
    assert cv.linear_rate(errs) == pytest.approx(0.5, abs=0.05)


def test_fixed_point_rate_matches_derivative():
    """g(x) = 0.7x + 0.6/x has g'(sqrt 2) = 0.4, so the linear rate must be 0.4."""
    target = math.sqrt(2.0)
    x, errs = 1.0, []
    for _ in range(60):
        x = 0.7 * x + 0.6 / x
        errs.append(abs(x - target))
    assert cv.linear_rate(errs) == pytest.approx(0.4, abs=0.01)


def test_linear_rate_on_a_clean_geometric_sequence():
    for C in [0.1, 0.25, 0.5, 0.75, 0.9]:
        errs = [C**k for k in range(30)]
        assert cv.linear_rate(errs) == pytest.approx(C, rel=1e-9)


def test_linear_rate_ignores_a_stalled_tail():
    """Once the errors flatten out at the roundoff floor they carry no information."""
    clean = [0.5**k for k in range(30)]
    stalled = clean + [clean[-1]] * 20
    assert cv.linear_rate(stalled) == pytest.approx(0.5, abs=0.02)


def test_linear_rate_needs_enough_points():
    assert math.isnan(cv.linear_rate([1.0]))
    assert math.isnan(cv.linear_rate([1.0, 0.5]))


# ---------------------------------------------------------------- constants and tables


def test_asymptotic_constant():
    errs = synthetic_errors(2.0, 0.3, e0=0.1, n=6)
    assert cv.asymptotic_constant(errs, 2.0) == pytest.approx(0.3, rel=1e-9)


def test_asymptotic_constant_insufficient_data():
    assert math.isnan(cv.asymptotic_constant([1.0], 2.0))


def test_steps_to_tolerance():
    errs = [1.0, 0.1, 0.01, 1e-3, 1e-4]
    assert cv.steps_to_tolerance(errs, 1e-3) == 3
    assert cv.steps_to_tolerance(errs, 1.0) == 0
    assert cv.steps_to_tolerance(errs, 1e-9) == -1


def test_convergence_table_is_readable():
    errs = synthetic_errors(2.0, 1.0, e0=0.1, n=5)
    text = cv.convergence_table(errs, exact_order=2)
    assert "observed order" in text
    assert "theory: 2" in text
    assert len(text.splitlines()) == len(errs) + 2      # header, rule, one row each


# ---------------------------------------------------------------- power law fitting


def test_fit_power_law_exact():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = 3.0 * x**2.5
    p, C = cv.fit_power_law(x, y)
    assert p == pytest.approx(2.5, rel=1e-9)
    assert C == pytest.approx(3.0, rel=1e-9)


def test_fit_power_law_ignores_non_positive_points():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    y = np.array([1.0, 4.0, 16.0, 0.0, -1.0])
    p, _C = cv.fit_power_law(x, y)
    assert p == pytest.approx(2.0, rel=1e-9)


def test_fit_power_law_insufficient_data():
    p, C = cv.fit_power_law([1.0], [1.0])
    assert math.isnan(p) and math.isnan(C)


def test_refinement_order_forward_difference():
    """Forward difference is first order in h."""
    hs = np.array([2.0**-k for k in range(2, 14)])
    errs = np.array([abs((np.exp(1 + h) - np.exp(1)) / h - np.exp(1)) for h in hs])
    assert cv.refinement_order(hs, errs) == pytest.approx(1.0, abs=0.02)


def test_refinement_order_central_difference():
    """Central difference is second order in h."""
    hs = np.array([2.0**-k for k in range(2, 14)])
    errs = np.array(
        [abs((np.exp(1 + h) - np.exp(1 - h)) / (2 * h) - np.exp(1)) for h in hs]
    )
    assert cv.refinement_order(hs, errs) == pytest.approx(2.0, abs=0.02)


def test_refinement_order_on_a_fourth_order_rule():
    """Simpson's rule on a smooth integrand is fourth order."""

    def simpson(f, a, b, n):
        x = np.linspace(a, b, n + 1)
        y = f(x)
        h = (b - a) / n
        return h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())

    exact = np.exp(1.0) - 1.0
    ns = np.array([4, 8, 16, 32, 64])
    hs = 1.0 / ns
    errs = np.array([abs(simpson(np.exp, 0.0, 1.0, int(n)) - exact) for n in ns])
    assert cv.refinement_order(hs, errs) == pytest.approx(4.0, abs=0.05)
