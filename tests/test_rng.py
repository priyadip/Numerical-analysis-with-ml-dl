"""Tests for nalib.rng.

Four groups. The recurrence group asserts integer identities rather than statistics: the period is
an exact count, the Hull-Dobell conditions are an equivalence, and the RANDU relation holds with a
residual of exactly zero. Anything above zero there is a bug, not a tolerance question.

The lattice group asserts the plane count and, more usefully, asserts that the same bounded search
finds **no** short relation for the other generators. A test that only confirmed RANDU's 15 planes
would pass for a search that always returns 15.

The transform group asserts against the distribution function each transform claims to produce,
through a Kolmogorov-Smirnov statistic scaled by ``sqrt(n)`` so the bar means the same thing at
every sample size.

The comparison group asserts the negative result on purpose: every generator here, including RANDU,
passes a one dimensional uniformity test and a lag one correlation test. A later reader tempted to
treat those as sufficient has to get past a test that says they are not.
"""
import functools
import math

import numpy as np
import pytest

from nalib import rng as rg


@functools.lru_cache(maxsize=None)
def planes():
    return rg.randu_lies_on_a_few_planes()


@functools.lru_cache(maxsize=None)
def conditions():
    return rg.the_hull_dobell_conditions_are_exact()


@functools.lru_cache(maxsize=None)
def low_bits():
    return rg.the_low_bits_are_worse()


@functools.lru_cache(maxsize=None)
def dimensions():
    return rg.one_test_is_not_enough()


@functools.lru_cache(maxsize=None)
def transforms():
    return rg.the_transforms_are_exact()


@functools.lru_cache(maxsize=None)
def polar():
    return rg.the_polar_method_wastes_a_known_fraction()


@functools.lru_cache(maxsize=None)
def compared():
    return rg.the_named_generators_compared()


# ------------------------------------------------------------------ the recurrence


@pytest.mark.parametrize("modulus", [16, 64, 256])
def test_the_period_never_exceeds_the_modulus(modulus):
    rng = np.random.default_rng(42)
    for _ in range(20):
        a = int(rng.integers(0, modulus))
        c = int(rng.integers(0, modulus))
        assert rg.period(a, c, modulus) <= modulus


def test_the_hull_dobell_conditions_are_an_equivalence():
    out = conditions()
    assert out["the_conditions_are_exact"]
    assert out["disagreements"] == []
    assert out["agreements"] == out["combinations"]


def test_some_combinations_do_have_full_period():
    # a sweep where nothing reached full period would make the previous test vacuous
    assert conditions()["full_period_count"] > 0


@pytest.mark.parametrize("name", sorted(rg.FAMOUS))
def test_every_named_generator_stays_in_range(name):
    values = rg.lcg_uniform(500, name)
    assert values.size == 500
    assert float(np.min(values)) >= 0.0
    assert float(np.max(values)) < 1.0


def test_the_generator_is_deterministic():
    first = rg.lcg_uniform(200, "minstd", seed=7)
    second = rg.lcg_uniform(200, "minstd", seed=7)
    assert float(np.max(np.abs(first - second))) == 0.0
    other = rg.lcg_uniform(200, "minstd", seed=8)
    assert float(np.max(np.abs(first - other))) > 0.0


def test_a_bad_modulus_is_rejected():
    with pytest.raises(ValueError):
        rg.lcg(10, 5, 1, 1)
    with pytest.raises(ValueError):
        rg.lcg(-1, 5, 1, 16)
    with pytest.raises(ValueError):
        rg.lcg_uniform(10, "not a generator")


# ------------------------------------------------------------------ the lattice


def test_randu_lies_on_exactly_fifteen_planes():
    out = planes()
    assert out["planes"] == 15
    assert out["normal"] == [9, -6, 1]


def test_the_relation_is_an_exact_integer_identity():
    out = planes()
    assert out["relation_is_exact"]
    assert out["worst_residual"] == 0


def test_a_modern_generator_puts_nothing_on_those_planes():
    out = planes()
    assert out["the_relation_does_not_hold_for_pcg64"]
    assert out["pcg64_triples_on_the_same_planes"] == 0


def test_only_randu_has_a_short_relation():
    # without this the plane count would pass for a search that always returns the same answer
    out = planes()
    assert out["only_randu_has_a_short_relation"]
    for row in out["rows"]:
        if row["generator"] != "randu":
            assert row["normal"] is None


def test_the_relation_search_is_verifiable():
    found = rg.shortest_relation(65539, 2 ** 31)
    assert found["found"]
    p, q, r = found["normal"]
    assert (p + q * 65539 + r * 65539 ** 2) % 2 ** 31 == 0


@pytest.mark.parametrize("bound", [1, 2, 4])
def test_a_small_search_box_finds_nothing(bound):
    assert not rg.shortest_relation(65539, 2 ** 31, bound=bound)["found"]


def test_the_low_bits_have_the_predicted_periods():
    out = low_bits()
    assert out["every_bit_matches"]
    for row in out["rows"]:
        assert row["measured_period"] == 2 ** (row["bit"] + 1)


def test_randu_passes_in_one_dimension_and_fails_in_three():
    out = dimensions()
    assert out["flat_passes"]
    assert out["randu_fails_in_three_dimensions"]
    assert out["pcg64_passes_in_three_dimensions"]
    assert out["cube_ratio"] > 2.0 * out["pcg64_cube_ratio"]


# ------------------------------------------------------------------ the transforms


def test_both_transforms_match_their_distributions():
    out = transforms()
    assert out["both_pass"]
    assert out["exponential_scaled"] < 1.63
    assert out["normal_scaled"] < 1.63


def test_the_normals_have_the_right_moments():
    out = transforms()
    assert abs(out["normal_mean"]) < 0.01
    assert abs(out["normal_variance"] - 1.0) < 0.01


@pytest.mark.parametrize("rate", [0.25, 1.0, 7.5])
def test_the_exponential_mean_is_one_over_the_rate(rate):
    rng = np.random.default_rng(42)
    values = rg.inverse_transform_exponential(rng.random(200000), rate=rate)
    assert abs(float(np.mean(values)) * rate - 1.0) < 0.02
    assert float(np.min(values)) >= 0.0


def test_the_transform_rejects_a_bad_rate():
    with pytest.raises(ValueError):
        rg.inverse_transform_exponential(np.array([0.5]), rate=0.0)


@pytest.mark.parametrize("size", [2, 10, 1000])
def test_box_muller_returns_as_many_as_it_consumes(size):
    rng = np.random.default_rng(42)
    out = rg.box_muller(rng.random(size))
    assert out.shape == (size,)
    assert np.all(np.isfinite(out))


def test_box_muller_needs_pairs():
    with pytest.raises(ValueError):
        rg.box_muller(np.array([0.5, 0.25, 0.75]))


def test_the_polar_method_accepts_pi_over_four():
    out = polar()
    assert out["matches_pi_over_four"]
    assert abs(out["ratio"] - 1.0) < 0.01


def test_the_polar_output_is_normal():
    assert polar()["passes"]


@pytest.mark.parametrize("draws", [0, 1, 5, 1000])
def test_the_polar_method_returns_what_was_asked_for(draws):
    out = rg.polar_normal(draws)
    assert out["values"].shape == (draws,)


# ------------------------------------------------------------------ the comparison


def test_every_generator_passes_the_easy_tests():
    # the point of the lesson: uniformity and low order correlation catch nothing at all
    out = compared()
    assert out["every_generator_passes_uniformity"]
    assert out["worst_lcg_correlation"] < 0.01


def test_randu_is_not_separated_by_correlation_alone():
    out = compared()
    randu = next(r for r in out["rows"] if r["generator"] == "randu")
    modern = next(r for r in out["rows"] if r["generator"] == "pcg64")
    assert abs(randu["lag_one_correlation"]) < 10.0 * abs(modern["lag_one_correlation"])


def test_the_chi_square_statistic_is_near_its_degrees_of_freedom():
    rng = np.random.default_rng(42)
    out = rg.chi_square_uniformity(rng.random(200000), bins=50)
    assert out["degrees_of_freedom"] == 49
    assert out["looks_uniform"]


def test_a_deliberately_skewed_sample_is_caught():
    rng = np.random.default_rng(42)
    skewed = rng.random(200000) ** 2
    assert not rg.chi_square_uniformity(skewed, bins=50)["looks_uniform"]


def test_the_chi_square_rejects_bad_input():
    with pytest.raises(ValueError):
        rg.chi_square_uniformity(np.array([0.5]), bins=1)
    with pytest.raises(ValueError):
        rg.chi_square_uniformity(np.array([]))


def test_a_deliberately_correlated_sequence_is_caught():
    rng = np.random.default_rng(42)
    base = rng.random(100000)
    smoothed = 0.5 * (base[:-1] + base[1:])
    assert abs(rg.serial_correlation(smoothed, 1)) > 0.4
    assert abs(rg.serial_correlation(base, 1)) < 0.01


def test_the_correlation_rejects_a_bad_lag():
    with pytest.raises(ValueError):
        rg.serial_correlation(np.arange(10.0), lag=0)
    with pytest.raises(ValueError):
        rg.serial_correlation(np.arange(10.0), lag=10)


@pytest.mark.parametrize("size", [10, 500, 20000])
def test_the_ks_statistic_falls_like_one_over_root_n(size):
    rng = np.random.default_rng(42)
    out = rg.kolmogorov_smirnov(rng.random(size))
    assert out["samples"] == size
    assert 0.0 <= out["gap"] <= 1.0
    assert out["scaled"] < 3.0


def test_the_ks_statistic_catches_the_wrong_distribution():
    rng = np.random.default_rng(42)
    out = rg.kolmogorov_smirnov(rng.random(20000) ** 2)
    assert not out["passes_at_one_percent"]
