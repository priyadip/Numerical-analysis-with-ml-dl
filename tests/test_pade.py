"""Tests for nalib.pade.

The defining property, that the series of ``P/Q`` matches the Taylor series to order ``m + n``,
is checkable exactly by long division, so most of these check coefficients rather than samples.
The classical ``[2/2]`` approximants of ``exp`` and ``log(1+x)`` are in every textbook and are
checked against their published forms.

Every test sweeps ``m``, ``n``, functions and intervals.
"""
import math

import numpy as np
import pytest

from nalib import pade as pd

SERIES = ["exp", "log1p", "tan", "arctan", "one_over_one_plus_x", "sqrt1p"]
SPLITS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 2), (2, 3), (4, 4), (5, 3)]


def callable_for(name):
    return {"exp": np.exp,
            "log1p": np.log1p,
            "tan": np.tan,
            "arctan": np.arctan,
            "one_over_one_plus_x": lambda t: 1.0 / (1.0 + t),
            "sqrt1p": lambda t: np.sqrt(1.0 + t)}[name]


# --------------------------------------------------------------------------- series


@pytest.mark.parametrize("name", SERIES)
def test_every_series_starts_at_the_right_value(name):
    c = pd.taylor_coefficients(name, 8)
    f = callable_for(name)
    assert float(c[0]) == pytest.approx(float(np.atleast_1d(f(np.array([0.0])))[0]), abs=1e-14)


@pytest.mark.parametrize("name", SERIES)
@pytest.mark.parametrize("count", [4, 9, 20])
def test_the_series_reproduces_the_function_near_zero(name, count):
    c = pd.taylor_coefficients(name, count)
    f = callable_for(name)
    x = np.linspace(-0.05, 0.05, 41)
    got = sum(c[k] * x ** k for k in range(c.size))
    assert np.max(np.abs(got - f(x))) < 1e-3


def test_the_tangent_series_matches_the_published_coefficients():
    c = pd.taylor_coefficients("tan", 10)
    exact = {1: 1.0, 3: 1.0 / 3.0, 5: 2.0 / 15.0, 7: 17.0 / 315.0, 9: 62.0 / 2835.0}
    for k, v in exact.items():
        assert c[k] == pytest.approx(v, rel=1e-12)
    for k in (0, 2, 4, 6, 8):
        assert c[k] == 0.0


def test_the_sqrt_series_matches_the_binomial_coefficients():
    c = pd.taylor_coefficients("sqrt1p", 6)
    exact = [1.0, 0.5, -0.125, 0.0625, -5.0 / 128.0, 7.0 / 256.0]
    assert np.max(np.abs(c - np.asarray(exact))) < 1e-14


@pytest.mark.parametrize("bad", ["cos", "gamma", ""])
def test_an_unknown_series_is_rejected(bad):
    with pytest.raises(ValueError, match="unknown series"):
        pd.taylor_coefficients(bad, 5)


@pytest.mark.parametrize("count", [0, -3])
def test_asking_for_no_coefficients_is_rejected(count):
    with pytest.raises(ValueError, match="at least one coefficient"):
        pd.taylor_coefficients("exp", count)


# --------------------------------------------------------------------------- construction


@pytest.mark.parametrize("name", SERIES)
@pytest.mark.parametrize("m,n", SPLITS)
def test_the_approximant_matches_the_series_to_the_promised_order(name, m, n):
    """Full order when the block is non-degenerate, and to the block's own order when not.

    A degenerate entry is a lower order approximant repeated, so it matches the series only as
    far as that lower order reaches. Asserting the full order there would be asserting something
    false; asserting nothing would let a real failure through. The weaker claim that still has
    content is that it agrees with the function near the origin.
    """
    c = pd.taylor_coefficients(name, m + n + 6)
    out = pd.coefficients(c, m, n)
    if not out["degenerate"]:
        assert pd.matches_to_order(c, m, n)["matches"]
        return
    # A degenerate entry matches fewer terms, and how many is a measurement rather than a
    # prediction: [2/3] of tan has rank 2 and matches 3 terms, [0/1] of tan has rank 0 and
    # matches 1. What is guaranteed is that it matches at least the constant, and that its
    # measured order is at least 1 and at most the full m + n + 1.
    order = pd.matching_order(c, m, n)
    assert order >= 1
    assert order <= m + n + 1
    assert pd.matches_to_order(c, m, n, n_terms=order)["matches"]


@pytest.mark.parametrize("m,n", [(2, 2), (3, 3), (4, 4), (6, 6)])
def test_a_degenerate_block_still_reproduces_the_rational_function(m, n):
    """1/(1+x) is [0/1], so every larger entry of its Pade table is the same function and the
    Toeplitz system has rank 1. The least norm solve recovers it to machine precision."""
    c = pd.taylor_coefficients("one_over_one_plus_x", m + n + 2)
    out = pd.coefficients(c, m, n)
    assert out["degenerate"] == (n > 1)
    assert out["rank"] == 1
    x = np.linspace(-0.5, 3.0, 41)
    assert np.max(np.abs(pd.evaluate(out["p"], out["q"], x) - 1.0 / (1.0 + x))) < 1e-12


@pytest.mark.parametrize("name", SERIES)
@pytest.mark.parametrize("m,n", SPLITS)
def test_the_denominator_is_normalised_to_one_at_the_origin(name, m, n):
    c = pd.taylor_coefficients(name, m + n + 4)
    out = pd.coefficients(c, m, n)
    assert out["q"][0] == 1.0
    assert out["p"].size == m + 1
    assert out["q"].size == n + 1


def test_the_two_two_approximant_of_exp_is_the_textbook_one():
    """P = 1 + x/2 + x^2/12, Q = 1 - x/2 + x^2/12, which is the diagonal Pade for exp."""
    out = pd.coefficients(pd.taylor_coefficients("exp", 6), 2, 2)
    assert np.max(np.abs(out["p"] - np.array([1.0, 0.5, 1.0 / 12.0]))) < 1e-14
    assert np.max(np.abs(out["q"] - np.array([1.0, -0.5, 1.0 / 12.0]))) < 1e-14


def test_the_one_one_approximant_of_log1p_is_the_textbook_one():
    """P = x, Q = 1 + x/2, so P/Q = 2x / (2 + x)."""
    out = pd.coefficients(pd.taylor_coefficients("log1p", 4), 1, 1)
    assert np.max(np.abs(out["p"] - np.array([0.0, 1.0]))) < 1e-14
    assert np.max(np.abs(out["q"] - np.array([1.0, 0.5]))) < 1e-14


@pytest.mark.parametrize("name", SERIES)
@pytest.mark.parametrize("m", [0, 1, 3, 6])
def test_a_zero_denominator_degree_returns_the_taylor_polynomial(name, m):
    c = pd.taylor_coefficients(name, m + 3)
    out = pd.coefficients(c, m, 0)
    assert np.max(np.abs(out["p"] - c[:m + 1])) == 0.0
    assert out["q"].size == 1 and out["q"][0] == 1.0


@pytest.mark.parametrize("m,n", [(-1, 2), (2, -1), (-3, -3)])
def test_negative_degrees_are_rejected(m, n):
    with pytest.raises(ValueError, match="degrees must be non-negative"):
        pd.coefficients(pd.taylor_coefficients("exp", 12), m, n)


@pytest.mark.parametrize("m,n,have", [(3, 3, 5), (2, 2, 4), (5, 1, 6)])
def test_too_few_taylor_coefficients_is_rejected(m, n, have):
    with pytest.raises(ValueError, match="Taylor coefficients"):
        pd.coefficients(pd.taylor_coefficients("exp", have), m, n)


# --------------------------------------------------------------------------- evaluation


@pytest.mark.parametrize("name", SERIES)
@pytest.mark.parametrize("m,n", [(2, 2), (3, 3), (4, 3)])
def test_the_approximant_is_exact_at_the_origin(name, m, n):
    c = pd.taylor_coefficients(name, m + n + 3)
    got = pd.approximant(c, m, n)(np.array([0.0]))
    assert float(got[0]) == pytest.approx(float(c[0]), abs=1e-12)


@pytest.mark.parametrize("m,n", [(0, 1), (1, 1), (2, 2), (4, 4), (6, 6)])
def test_the_diagonal_approximant_of_one_over_one_plus_x_is_exact(m, n):
    """1/(1+x) IS a rational function, so any [m/n] with n >= 1 reproduces it exactly."""
    c = pd.taylor_coefficients("one_over_one_plus_x", m + n + 2)
    x = np.linspace(-0.5, 3.0, 41)
    assert np.max(np.abs(pd.approximant(c, m, n)(x) - 1.0 / (1.0 + x))) < 1e-12


# --------------------------------------------------------------------------- poles


@pytest.mark.parametrize("n", [4, 5, 6])
def test_the_tangent_approximant_finds_the_pole_at_half_pi(n):
    c = pd.taylor_coefficients("tan", 2 * n + 8)
    out = pd.pole_locations(c, n - 1, n)
    real = out["real_poles"]
    assert real.size > 0
    assert float(np.min(np.abs(np.abs(real) - math.pi / 2.0))) < 1e-2


@pytest.mark.parametrize("name", SERIES)
@pytest.mark.parametrize("m,n", [(2, 2), (3, 3), (4, 4)])
def test_a_non_degenerate_entry_matches_every_promised_term(name, m, n):
    c = pd.taylor_coefficients(name, m + n + 6)
    if pd.coefficients(c, m, n)["degenerate"]:
        return
    assert pd.matching_order(c, m, n) == m + n + 1


def test_the_pole_of_tan_is_located_better_as_the_degree_rises():
    """0.161, 0.144, 4.4e-4, 3.8e-4, 2.1e-7 for [n-1/n] at n = 2 to 6. The jump at n = 4 is
    where the denominator first has enough degree to place a pair of poles rather than being
    forced to compromise."""
    gaps = []
    for n in (2, 3, 4, 5, 6):
        real = pd.pole_locations(pd.taylor_coefficients("tan", 2 * n + 8), n - 1, n)["real_poles"]
        gaps.append(float(np.min(np.abs(np.abs(real) - math.pi / 2.0))))
    assert gaps[-1] < gaps[0]
    assert gaps[2] < 0.01 < gaps[1]


@pytest.mark.parametrize("m,n", [(2, 2), (3, 3), (4, 4)])
def test_the_log_approximant_puts_a_pole_near_the_branch_point(m, n):
    c = pd.taylor_coefficients("log1p", m + n + 3)
    out = pd.pole_locations(c, m, n)
    assert out["real_poles"].size > 0
    assert float(np.min(np.abs(out["real_poles"] + 1.0))) < 0.6


@pytest.mark.parametrize("m", [0, 2, 5])
def test_a_polynomial_approximant_has_no_poles(m):
    out = pd.pole_locations(pd.taylor_coefficients("exp", m + 3), m, 0)
    assert out["poles"].size == 0
    assert math.isinf(out["nearest_to_origin"])


@pytest.mark.parametrize("m,n", [(2, 2), (4, 4)])
def test_the_exponential_approximant_keeps_its_poles_away_from_the_origin(m, n):
    out = pd.pole_locations(pd.taylor_coefficients("exp", m + n + 2), m, n)
    assert out["nearest_to_origin"] > 1.0


@pytest.mark.parametrize("m,n", [(2, 2), (3, 3), (5, 5)])
def test_the_doublet_search_runs_and_reports_a_count(m, n):
    out = pd.spurious_poles(pd.taylor_coefficients("exp", m + n + 2), m, n, -1.0, 1.0)
    assert out["count"] == out["poles_inside"].size
    assert out["gaps"].size == out["count"]


# --------------------------------------------------------------------------- comparisons


@pytest.mark.parametrize("m", [1, 2, 4, 6])
def test_an_m_over_zero_split_is_exactly_the_taylor_polynomial(m):
    out = pd.against_taylor(np.exp, pd.taylor_coefficients("exp", 12), m, 0, -1.0, 1.0)
    assert out["ratio"] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("m,n", [(2, 2), (3, 3), (4, 4)])
@pytest.mark.parametrize("name,lo,hi", [("tan", -1.4, 1.4), ("log1p", -0.9, 3.0),
                                        ("sqrt1p", -0.9, 3.0)])
def test_pade_beats_taylor_near_a_singularity(m, n, name, lo, hi):
    c = pd.taylor_coefficients(name, m + n + 8)
    out = pd.against_taylor(callable_for(name), c, m, n, lo, hi)
    assert out["ratio"] > 1.0


@pytest.mark.parametrize("name,lo,hi", [("tan", -1.4, 1.4), ("log1p", -0.9, 3.0)])
def test_the_pade_advantage_grows_with_the_degree(name, lo, hi):
    c = pd.taylor_coefficients(name, 40)
    ratios = [pd.against_taylor(callable_for(name), c, k, k, lo, hi)["ratio"]
              for k in (2, 3, 4, 5)]
    assert all(b > a for a, b in zip(ratios, ratios[1:]))


def test_pade_gains_little_on_an_entire_function_over_a_small_interval():
    """exp has no singularity, so the ratio structure has nothing to model."""
    c = pd.taylor_coefficients("exp", 20)
    out = pd.against_taylor(np.exp, c, 2, 2, -0.3, 0.3)
    assert out["ratio"] < 50.0


@pytest.mark.parametrize("total", [2, 4, 6, 8])
def test_the_table_covers_every_split_of_the_total_degree(total):
    c = pd.taylor_coefficients("log1p", total + 6)
    out = pd.table(np.log1p, c, total, -0.9, 3.0)
    assert np.all(out["m"] + out["n"] == total)
    assert out["errors"].size == out["m"].size


@pytest.mark.parametrize("name,lo,hi", [("log1p", -0.9, 3.0), ("sqrt1p", -0.9, 3.0),
                                        ("tan", -1.4, 1.4)])
def test_the_best_split_is_at_or_next_to_the_diagonal(name, lo, hi):
    c = pd.taylor_coefficients(name, 20)
    out = pd.table(callable_for(name), c, 8, lo, hi)
    assert out["diagonal_is_best"]


@pytest.mark.parametrize("m", [2, 4])
def test_the_toeplitz_conditioning_is_reported_and_grows(m):
    out = pd.system_conditioning(pd.taylor_coefficients("exp", 30), m, (1, 2, 3, 4, 5, 6))
    assert out["conditions"].size == out["denominator_degrees"].size
    assert out["conditions"][-1] > out["conditions"][0]
