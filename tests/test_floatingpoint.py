"""Tests for nalib.floatingpoint.

The key tests here check the standard model and the exactness claims, since those are what
the rest of the course relies on.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction

import numpy as np
import pytest


from nalib import floatingpoint as fp


def subtraction_is_exact(a: float, b: float) -> bool:
    """Is the computed a - b exactly equal to the true difference?

    Uses `fractions.Fraction`, which is exact. `np.longdouble` is NOT a valid
    higher-precision reference: on Windows with MSVC it is an alias for float64, so
    comparing against it would compare a value with itself and always pass.
    """
    return Fraction(a) - Fraction(b) == Fraction(a - b)



# ---------------------------------------------------------------- constants


def test_machine_epsilon_matches_numpy():
    assert fp.machine_epsilon() == np.finfo(np.float64).eps
    assert fp.machine_epsilon() == 2.0**-52
    assert fp.EPS_DOUBLE == 2.0**-52
    assert fp.UNIT_ROUNDOFF == 2.0**-53


def test_machine_epsilon_single_precision():
    assert fp.machine_epsilon(np.float32) == float(np.finfo(np.float32).eps)


def test_epsilon_is_the_boundary():
    eps = fp.EPS_DOUBLE
    assert 1.0 + eps != 1.0
    assert 1.0 + eps / 2 == 1.0


# ---------------------------------------------------------------- inspection


def test_float_bits_length_and_content():
    bits = fp.float_bits(1.0)
    assert len(bits) == 64
    assert set(bits) <= {"0", "1"}
    # 1.0 is sign 0, exponent 1023 = 01111111111, mantissa all zero
    assert bits == "0" + "01111111111" + "0" * 52


@pytest.mark.parametrize("value", [1.0, 2.0, 9.0, 0.5, -3.25, 1e-8, 1e300, -1e-300])
def test_decompose_reconstructs_exactly(value):
    d = fp.decompose(value)
    assert d["kind"] == "normal"
    rebuilt = d["sign"] * d["significand"] * 2.0 ** d["exponent"]
    assert rebuilt == value


def test_decompose_nine():
    """9 = 1001 in binary = 1.001 x 2^3, so significand 1.125 and exponent 3."""
    d = fp.decompose(9.0)
    assert d["exponent"] == 3
    assert d["significand"] == 1.125
    assert d["sign"] == 1
    assert d["raw_exponent"] == 3 + 1023


def test_decompose_special_values():
    assert fp.decompose(0.0)["kind"] == "zero"
    assert fp.decompose(-0.0)["kind"] == "zero"
    assert fp.decompose(float("inf"))["kind"] == "inf"
    assert fp.decompose(float("-inf"))["kind"] == "inf"
    assert fp.decompose(float("nan"))["kind"] == "nan"
    assert fp.decompose(5e-324)["kind"] == "subnormal"
    assert fp.decompose(float(np.finfo(float).tiny) / 2)["kind"] == "subnormal"
    assert fp.decompose(float(np.finfo(float).tiny))["kind"] == "normal"


def test_decompose_random_reconstruction(rng):
    for _ in range(5000):
        value = float(rng.standard_normal()) * 10.0 ** int(rng.integers(-200, 200))
        if value == 0.0 or not math.isfinite(value):
            continue
        d = fp.decompose(value)
        if d["kind"] != "normal":
            continue
        assert d["sign"] * d["significand"] * 2.0 ** d["exponent"] == value


# ---------------------------------------------------------------- spacing


def test_ulp_at_one_is_epsilon():
    assert fp.ulp(1.0) == fp.EPS_DOUBLE


def test_ulp_matches_nextafter(rng):
    for _ in range(2000):
        x = abs(float(rng.standard_normal())) * 10.0 ** int(rng.integers(-100, 100))
        if x == 0.0:
            continue
        assert fp.ulp(x) == math.nextafter(x, math.inf) - x


def test_relative_spacing_stays_in_the_expected_band(rng):
    """ulp(x)/x must lie between eps/2 and eps for every normal x."""
    lo, hi = fp.EPS_DOUBLE / 2, fp.EPS_DOUBLE
    for _ in range(5000):
        x = abs(float(rng.standard_normal())) * 10.0 ** int(rng.integers(-250, 250))
        if x == 0.0 or not math.isfinite(x) or x < float(np.finfo(float).tiny):
            continue
        ratio = fp.ulp(x) / x
        assert lo <= ratio <= hi * (1 + 1e-12), (x, ratio)


def test_count_floats_between():
    assert fp.count_floats_between(1.0, 1.0) == 0
    assert fp.count_floats_between(1.0, math.nextafter(1.0, math.inf)) == 1
    assert fp.count_floats_between(1.0, math.nextafter(1.0, -math.inf)) == 1
    assert fp.count_floats_between(0.0, -0.0) == 0
    # symmetry
    a, b = 1.5, 2.5
    assert fp.count_floats_between(a, b) == fp.count_floats_between(b, a)


def test_count_floats_between_is_monotone():
    """Stepping n times must report exactly n floats of distance."""
    x = 3.7
    y = x
    for n in range(1, 50):
        y = math.nextafter(y, math.inf)
        assert fp.count_floats_between(x, y) == n


def test_count_floats_between_rejects_non_finite():
    with pytest.raises(ValueError):
        fp.count_floats_between(1.0, float("inf"))


# ---------------------------------------------------------------- simulated precision


def test_round_to_precision_is_identity_at_full_precision(rng):
    for _ in range(2000):
        x = float(rng.standard_normal()) * 10.0 ** int(rng.integers(-50, 50))
        assert fp.round_to_precision(x, 52) == x


def test_round_to_precision_known_cases():
    # 1/3 = 0.0101010101..._2 = 1.0101..._2 x 2^-2
    # with 2 mantissa bits: 1.01_2 x 2^-2 = 0.3125
    assert fp.round_to_precision(1.0 / 3.0, 2) == 0.3125
    # powers of two are exact at any precision
    for bits in range(0, 53):
        assert fp.round_to_precision(0.5, bits) == 0.5
        assert fp.round_to_precision(8.0, bits) == 8.0


def test_round_to_precision_handles_zero_and_specials():
    assert fp.round_to_precision(0.0, 4) == 0.0
    assert math.isinf(fp.round_to_precision(float("inf"), 4))
    assert math.isnan(fp.round_to_precision(float("nan"), 4))


def test_round_to_precision_rejects_negative_bits():
    with pytest.raises(ValueError):
        fp.round_to_precision(1.0, -1)


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16, 24, 40, 52])
def test_standard_model_bound_holds(bits, rng):
    """|fl(x) - x| / |x| <= u for the simulated precision. This is THE model."""
    u = fp.simulated_unit_roundoff(bits)
    worst = 0.0
    for _ in range(3000):
        x = float(rng.standard_normal()) * 10.0 ** int(rng.integers(-30, 30))
        if x == 0.0:
            continue
        delta = abs(fp.relative_rounding_error(x, bits))
        worst = max(worst, delta)
        assert delta <= u * (1 + 1e-12), (bits, x, delta, u)
    if bits < 52:
        # the bound should be close to tight, not wildly pessimistic
        assert worst > u / 4, (bits, worst, u)


def test_simulated_unit_roundoff_matches_double():
    assert fp.simulated_unit_roundoff(52) == fp.UNIT_ROUNDOFF


def test_relative_rounding_error_of_zero():
    assert fp.relative_rounding_error(0.0, 8) == 0.0


# ---------------------------------------------------------------- summation


def test_all_summation_methods_agree_on_easy_data(rng):
    values = rng.uniform(-1, 1, 500)
    exact = math.fsum(values.tolist())
    for name, func in fp.summation_methods().items():
        assert abs(func(values) - exact) < 1e-12, name


def test_summation_of_empty_and_single():
    assert fp.naive_sum([]) == 0.0
    assert fp.pairwise_sum([]) == 0.0
    assert fp.kahan_sum([]) == 0.0
    assert fp.neumaier_sum([]) == 0.0
    assert fp.naive_sum([3.5]) == 3.5
    assert fp.pairwise_sum([3.5]) == 3.5
    assert fp.kahan_sum([3.5]) == 3.5
    assert fp.neumaier_sum([3.5]) == 3.5


def test_compensated_summation_beats_naive_on_the_hard_case():
    """One large value then many tiny ones. Naive loses all of them."""
    n = 100_000
    values = np.concatenate([[1.0], np.full(n, 1e-16)])
    exact = 1.0 + n * 1e-16

    naive = fp.naive_sum(values)
    kahan = fp.kahan_sum(values)
    neumaier = fp.neumaier_sum(values)

    assert naive == 1.0, "naive summation should lose every small value here"
    assert abs(kahan - exact) / exact < 1e-15
    assert abs(neumaier - exact) / exact < 1e-15


def test_neumaier_handles_the_case_kahan_misses():
    """Kahan loses the correction when the running total is smaller than the addend.

    The classic example: 1.0, 1e100, 1.0, -1e100. The exact answer is 2.0.
    """
    values = [1.0, 1e100, 1.0, -1e100]
    assert fp.neumaier_sum(values) == 2.0
    assert fp.naive_sum(values) == 0.0        # the two ones are annihilated


def test_pairwise_matches_numpy_closely(rng):
    values = rng.uniform(0, 1, 10_000)
    assert abs(fp.pairwise_sum(values) - float(np.sum(values))) < 1e-12


def test_summation_error_ordering_on_wide_range(rng):
    """On data spanning many magnitudes, compensated beats pairwise beats naive."""
    values = rng.uniform(0.5, 1.5, 50_000) * np.logspace(-8, 8, 50_000)
    exact = math.fsum(np.sort(values).tolist())   # fsum is exactly rounded

    e_naive = abs(fp.naive_sum(values) - exact) / exact
    e_pairwise = abs(fp.pairwise_sum(values) - exact) / exact
    e_neumaier = abs(fp.neumaier_sum(values) - exact) / exact

    assert e_neumaier <= e_pairwise
    assert e_pairwise < e_naive


# ---------------------------------------------------------------- the Sterbenz lemma


def test_sterbenz_lemma(rng):
    """If a/2 <= b <= 2a then a - b is computed exactly."""
    checked = 0
    for _ in range(50_000):
        a = float(rng.uniform(0.01, 100.0))
        b = float(rng.uniform(a / 2, 2 * a))
        if not (a / 2 <= b <= 2 * a):
            continue
        checked += 1
        assert subtraction_is_exact(a, b), (a, b)
    assert checked > 40_000


def test_sterbenz_can_fail_outside_its_range(rng):
    """Outside a/2 <= b <= 2a, subtraction is not guaranteed exact. Find a case."""
    found = False
    for _ in range(200_000):
        a = float(rng.uniform(1.0, 2.0))
        b = float(rng.uniform(1e-20, 1e-18))       # far outside the Sterbenz band
        if not subtraction_is_exact(a, b):
            found = True
            break
    assert found, "expected to find an inexact subtraction outside the Sterbenz range"
