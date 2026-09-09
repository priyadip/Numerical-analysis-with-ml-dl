"""Tests for nalib.numbersystems.

Covers normal cases, edge cases (zero, one, negatives, base limits), known analytic
results, and randomised round trips with a fixed seed.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from nalib import numbersystems as ns


# ---------------------------------------------------------------- integers


@pytest.mark.parametrize(
    "n, base, expected",
    [
        (0, 2, "0"),
        (1, 2, "1"),
        (2, 2, "10"),
        (13, 2, "1101"),
        (53, 2, "110101"),
        (255, 16, "FF"),
        (255, 2, "11111111"),
        (511, 8, "777"),
        (3705, 10, "3705"),
        (4095, 16, "FFF"),
    ],
)
def test_int_to_base_known_values(n, base, expected):
    assert ns.int_to_base(n, base) == expected


def test_int_to_base_matches_python_builtins():
    for n in range(0, 5000, 7):
        assert ns.int_to_base(n, 2) == format(n, "b")
        assert ns.int_to_base(n, 8) == format(n, "o")
        assert ns.int_to_base(n, 16) == format(n, "X")


def test_int_to_base_negative():
    assert ns.int_to_base(-13, 2) == "-1101"
    assert ns.int_from_base("-1101", 2) == -13


def test_int_round_trip_all_bases(rng):
    for _ in range(500):
        n = int(rng.integers(0, 10**9))
        for base in range(2, 17):
            assert ns.int_from_base(ns.int_to_base(n, base), base) == n


def test_int_to_base_steps_reconstructs_the_digits():
    for n in [0, 1, 13, 53, 1000, 65535]:
        for base in (2, 8, 10, 16):
            steps = ns.int_to_base_steps(n, base)
            digits = "".join(ns.DIGITS[r] for _, _, r in reversed(steps))
            assert digits == ns.int_to_base(n, base)


def test_int_to_base_steps_rejects_negative():
    with pytest.raises(ValueError):
        ns.int_to_base_steps(-1, 2)


# ---------------------------------------------------------------- fractions


@pytest.mark.parametrize(
    "value, base, ndigits, expected_digits, expected_exact",
    [
        (Fraction(1, 2), 2, 8, "1", True),
        (Fraction(1, 4), 2, 8, "01", True),
        (Fraction(3, 4), 2, 8, "11", True),
        (Fraction(5, 8), 2, 8, "101", True),
        (Fraction(1, 10), 2, 12, "000110011001", False),
        (Fraction(1, 3), 10, 6, "333333", False),
        (Fraction(1, 2), 10, 4, "5", True),
    ],
)
def test_frac_to_base_known(value, base, ndigits, expected_digits, expected_exact):
    digits, exact = ns.frac_to_base(value, base, ndigits)
    assert digits == expected_digits
    assert exact is expected_exact


def test_frac_to_base_rejects_out_of_range():
    with pytest.raises(ValueError):
        ns.frac_to_base(Fraction(3, 2), 2, 8)
    with pytest.raises(ValueError):
        ns.frac_to_base(Fraction(-1, 2), 2, 8)


def test_one_tenth_binary_expansion_repeats_with_period_four():
    """1/10 in binary is 0.0(0011) repeating. Check the pattern directly."""
    digits, exact = ns.frac_to_base(Fraction(1, 10), 2, 33)
    assert not exact
    # after the leading 0, the block 0011 repeats
    assert digits[0] == "0"
    body = digits[1:33]
    assert body == "0011" * 8


def test_frac_to_base_steps_remainders_cycle_for_one_tenth():
    """Once a remainder repeats, the digits must repeat. That is why 1/10 never ends."""
    steps = ns.frac_to_base_steps(Fraction(1, 10), 2, 12)
    remainders = [after for _before, after, _d in steps]
    # the value 1/5 shows up more than once, which forces the cycle
    assert remainders.count(Fraction(1, 5)) >= 2


# ---------------------------------------------------------------- whole numbers


@pytest.mark.parametrize(
    "value, base, expected",
    [
        (Fraction(53, 8), 2, "110.101"),
        (Fraction(0), 2, "0"),
        (Fraction(1), 2, "1"),
        (Fraction(-53, 8), 2, "-110.101"),
        (Fraction(255), 16, "FF"),
    ],
)
def test_to_base_known(value, base, expected):
    assert ns.to_base(value, base) == expected


def test_to_base_marks_non_terminating_expansions():
    text = ns.to_base(Fraction(1, 10), 2, 12)
    assert text.endswith("...")
    assert text.startswith("0.0001100")


def test_round_trip_exact_values(rng):
    """Anything that terminates in a base must survive a round trip exactly."""
    checked = 0
    for _ in range(1500):
        num = int(rng.integers(-10_000, 10_000))
        den = int(rng.integers(1, 500))
        value = Fraction(num, den)
        for base in (2, 3, 8, 10, 16):
            if not ns.is_exact_in_base(value, base):
                continue
            text = ns.to_base(value, base, 80)
            assert ns.from_base(text, base) == value
            checked += 1
    assert checked > 150, "the test should have exercised many exact conversions"


def test_convert_between_bases():
    assert ns.convert("FF", 16, 2) == "11111111"
    assert ns.convert("11111111", 2, 16) == "FF"
    assert ns.convert("266", 8, 16) == "B6"
    assert ns.convert("1101.101", 2, 10) == "13.625"


# ---------------------------------------------------------------- the theorem


@pytest.mark.parametrize(
    "value, base, expected",
    [
        (Fraction(1, 10), 10, True),
        (Fraction(1, 10), 2, False),
        (Fraction(1, 10), 16, False),
        (Fraction(1, 2), 2, True),
        (Fraction(1, 8), 2, True),
        (Fraction(1, 3), 2, False),
        (Fraction(1, 3), 3, True),
        (Fraction(1, 3), 6, True),
        (Fraction(1, 3), 9, True),
        (Fraction(1, 7), 14, True),
        (Fraction(1, 7), 10, False),
        (Fraction(5), 2, True),          # an integer terminates in every base
    ],
)
def test_is_exact_in_base_known(value, base, expected):
    assert ns.is_exact_in_base(value, base) is expected


def test_terminating_in_base_two_means_power_of_two_denominator(rng):
    """Theorem 2.3 specialised to base 2, checked over many random fractions."""
    for _ in range(2000):
        num = int(rng.integers(1, 1000))
        den = int(rng.integers(1, 1000))
        value = Fraction(num, den)
        q = value.denominator
        is_power_of_two = (q & (q - 1)) == 0
        assert ns.is_exact_in_base(value, 2) == is_power_of_two


def test_terminating_agrees_with_direct_expansion(rng):
    """is_exact_in_base must agree with actually trying the expansion."""
    for _ in range(300):
        num = int(rng.integers(0, 100))
        den = int(rng.integers(1, 100))
        value = Fraction(num, den)
        frac_part = value - int(value)
        for base in (2, 3, 5, 10, 12):
            predicted = ns.is_exact_in_base(value, base)
            _digits, actually_ended = ns.frac_to_base(frac_part, base, 400)
            assert predicted == actually_ended


# ---------------------------------------------------------------- edge cases


def test_base_validation():
    for bad in (0, 1, 17, -2, 2.5, "2"):
        with pytest.raises(ValueError):
            ns.int_to_base(5, bad)


def test_bad_digit_for_base():
    with pytest.raises(ValueError):
        ns.int_from_base("2", 2)
    with pytest.raises(ValueError):
        ns.int_from_base("G", 16)
    with pytest.raises(ValueError):
        ns.int_from_base("9", 8)


def test_empty_digit_string():
    with pytest.raises(ValueError):
        ns.int_from_base("", 2)


def test_digit_value():
    assert ns.digit_value("F", 16) == 15
    assert ns.digit_value("f", 16) == 15
    assert ns.digit_value("0", 2) == 0
