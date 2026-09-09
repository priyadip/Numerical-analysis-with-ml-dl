"""Tests for nalib.mixedprecision.

Four groups. The format group checks the simulated ``bfloat16`` against things that can be verified
without it: that rounding is idempotent, that a value already in the format survives untouched, that
the round trip through the top 16 bits agrees with an independent bit-level construction, and that
the tabulated epsilon and largest value match what the arithmetic actually does. Everything else in
this module rests on that simulation being right.

The overflow group asserts that the softmax threshold is ``log`` of the format's largest value, in
three formats, and that the shift by the maximum is an identity rather than an approximation. The
second half is the important one: an identity that is only approximately an identity would be a
trade, and it is not.

The underflow group asserts the other end of the same exponent field. It checks that a float16
gradient at 1e-08 is mostly exactly zero, that bfloat16 loses nothing there, and that loss scaling
recovers the gradient inside a window with a floor and a ceiling that are both located.

The summation group asserts three exact integers and one fitted exponent: the stagnation point is
``1/u`` to the digit in two formats, a wider accumulator removes it, the spread over random orders
follows the square root law for sequential summation, and pairwise summation beats that law rather
than obeying it.
"""
import functools
import math

import numpy as np
import pytest

from nalib import mixedprecision as mp


@functools.lru_cache(maxsize=None)
def formats():
    return mp.the_same_bits_spent_differently()


@functools.lru_cache(maxsize=None)
def overflow():
    return mp.a_softmax_overflows_where_the_exponent_runs_out()


@functools.lru_cache(maxsize=None)
def shifting():
    return mp.the_shift_is_exact_and_it_fixes_the_overflow()


@functools.lru_cache(maxsize=None)
def underflow():
    return mp.gradients_underflow_before_activations_do()


@functools.lru_cache(maxsize=None)
def scaling():
    return mp.loss_scaling_has_a_window()


@functools.lru_cache(maxsize=None)
def stagnation():
    return mp.a_low_precision_sum_stagnates()


@functools.lru_cache(maxsize=None)
def ordering():
    return mp.the_order_of_a_reduction_changes_the_answer()


@functools.lru_cache(maxsize=None)
def chunking():
    return mp.chunking_is_what_makes_a_reduction_nondeterministic()


@functools.lru_cache(maxsize=None)
def products():
    return mp.a_half_precision_product_needs_a_wider_accumulator()


# ------------------------------------------------------------------ the formats


def test_rounding_to_a_format_is_idempotent():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(500) * 1e3
    for name in ("float32", "float16", "bfloat16"):
        once = mp.cast(values, name)
        assert np.array_equal(np.asarray(mp.cast(once, name), dtype=np.float64),
                              np.asarray(once, dtype=np.float64))


def test_a_power_of_two_survives_every_format():
    for power in (-8, -1, 0, 3, 10):
        value = 2.0 ** power
        for name in ("float32", "float16", "bfloat16"):
            assert float(np.asarray(mp.cast(value, name), dtype=np.float64)) == value


def test_bfloat16_keeps_the_top_sixteen_bits():
    rng = np.random.default_rng(1)
    values = np.asarray(rng.standard_normal(200), dtype=np.float32)
    rounded = np.asarray(mp.to_bfloat16(values), dtype=np.float32)
    low = rounded.view(np.uint32) & np.uint32(0xFFFF)
    assert np.all(low == 0)


def test_bfloat16_rounding_is_closer_than_truncation():
    rng = np.random.default_rng(2)
    values = np.asarray(rng.standard_normal(2000), dtype=np.float32)
    truncated = (values.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    rounded = np.asarray(mp.to_bfloat16(values), dtype=np.float32)
    assert np.sum(np.abs(rounded - values)) < np.sum(np.abs(truncated - values))


def test_the_tabulated_epsilon_is_the_one_the_arithmetic_uses():
    for name in ("float32", "float16"):
        spec = mp.describe(name)
        one = float(np.asarray(mp.cast(1.0, name), dtype=np.float64))
        just_over = float(np.asarray(mp.cast(1.0 + spec["epsilon"], name), dtype=np.float64))
        just_under = float(np.asarray(mp.cast(1.0 + spec["epsilon"] / 4.0, name),
                                      dtype=np.float64))
        assert just_over > one
        assert just_under == one


def test_the_tabulated_largest_value_is_the_one_that_overflows():
    for name in ("float32", "float16"):
        spec = mp.describe(name)
        with np.errstate(over="ignore"):
            assert np.isfinite(float(np.asarray(mp.cast(spec["largest"], name),
                                                dtype=np.float64)))
            assert not np.isfinite(float(np.asarray(mp.cast(spec["largest"] * 4.0, name),
                                                    dtype=np.float64)))


def test_the_two_halves_are_the_same_width():
    assert formats()["the_two_halves_have_the_same_width"]
    assert formats()["one_is_more_precise_and_one_has_more_range"]


def test_the_trade_is_eight_against_ten_to_the_thirty_three():
    assert formats()["precision_ratio"] == pytest.approx(8.0, rel=1e-12)
    assert formats()["range_ratio"] > 1e30


def test_an_unknown_format_is_refused():
    with pytest.raises(ValueError):
        mp.cast(1.0, "float8")
    with pytest.raises(ValueError):
        mp.describe("float8")


# ------------------------------------------------------------------ overflow


def test_the_softmax_threshold_is_log_of_the_largest_value():
    assert overflow()["the_threshold_is_log_of_the_largest"]
    assert overflow()["worst_ratio"] < 1.01


def test_float16_is_the_one_that_overflows_first():
    assert overflow()["float16_is_the_tight_one"]


def test_softmax_is_invariant_under_a_constant_shift():
    assert shifting()["the_identity_is_exact"]
    assert shifting()["the_only_error_is_the_shift_itself"]


def test_the_direct_form_overflows_and_the_shifted_one_does_not():
    assert shifting()["the_direct_form_does"]
    assert shifting()["the_shifted_form_never_overflows"]


def test_the_shifted_softmax_still_sums_to_one():
    assert shifting()["it_still_sums_to_one"]


def test_the_two_logsumexp_forms_agree_where_both_work():
    rng = np.random.default_rng(3)
    scores = rng.standard_normal(20)
    assert mp.logsumexp_direct(scores) == pytest.approx(mp.logsumexp_shifted(scores), rel=1e-14)


def test_only_the_shifted_logsumexp_survives_a_large_input():
    scores = np.array([1000.0, 0.0, -3.0])
    with np.errstate(over="ignore"):
        assert not math.isfinite(mp.logsumexp_direct(scores))
    assert mp.logsumexp_shifted(scores) == pytest.approx(1000.0, abs=1e-9)


# ------------------------------------------------------------------ underflow


def test_a_small_gradient_vanishes_in_float16():
    assert underflow()["float16_fails"]
    assert underflow()["rows"][-1]["float16 zeros"] == 1.0


def test_the_same_gradient_survives_in_bfloat16():
    assert underflow()["bfloat16_survives"]


def test_the_smallest_numbers_differ_by_thirty_orders_of_magnitude():
    assert underflow()["float16_smallest"] / underflow()["bfloat16_smallest"] > 1e30


def test_loss_scaling_is_needed_in_float16_and_not_in_bfloat16():
    assert scaling()["float16_needs_scaling"]
    assert scaling()["bfloat16_does_not"]


def test_the_loss_scaling_window_has_a_floor_and_a_ceiling():
    assert scaling()["the_window_is_wide"]
    assert scaling()["the_window_has_a_top"]
    window = scaling()["windows"]["float16"]
    assert window["lowest"] < window["highest"] < window["overflows_above"]


def test_scaling_by_a_power_of_two_does_not_round():
    rng = np.random.default_rng(4)
    gradient = rng.standard_normal(64) * 1e-4
    plain = mp.loss_scaled_gradient(gradient, 1.0, "float16")
    scaled = mp.loss_scaled_gradient(gradient, 2.0 ** 8, "float16")
    assert scaled["relative_error"] <= plain["relative_error"]


# ------------------------------------------------------------------ summation


def test_the_stagnation_point_is_one_over_the_unit_roundoff():
    assert stagnation()["the_stagnation_point_is_one_over_u"]
    for row in stagnation()["rows"]:
        if row["stagnated"]:
            assert row["terms"] == int(round(row["predicted"]))


def test_half_precision_stagnates_and_single_does_not():
    assert stagnation()["half_precision_stagnates"]
    assert not stagnation()["rows"][-1]["stagnated"]


def test_a_wider_accumulator_removes_the_stagnation():
    assert stagnation()["mixed_precision_is_exact"]


def test_accumulating_in_the_storage_format_loses_the_tail():
    values = np.full(4096, 1.0)
    assert mp.accumulate(values, "float16", "float16") < values.size
    assert mp.accumulate(values, "float16", "float32") == pytest.approx(float(values.size))


def test_a_reduction_has_more_than_one_answer():
    assert ordering()["the_answers_differ"]
    for row in ordering()["rows"]:
        assert row["distinct"] > 1


def test_the_sequential_spread_follows_the_square_root_law():
    assert ordering()["the_sequential_spread_grows_like_the_square_root"]
    assert abs(ordering()["sequential_slope"] - 0.5) < 0.15


def test_pairwise_summation_beats_that_law():
    assert ordering()["pairwise_is_much_flatter"]
    assert ordering()["largest_gain_from_pairwise"] > 10.0


def test_the_classic_bound_is_loose_on_random_data():
    assert ordering()["the_bound_holds"]
    assert ordering()["worst_bound_slack"] > 10.0


def test_chunking_changes_the_answer():
    assert chunking()["chunking_changes_the_answer"]
    assert chunking()["distinct"] > 1


def test_more_chunks_is_more_accurate_here():
    assert chunking()["more_chunks_is_usually_better"]


def test_every_chunking_is_still_close_to_the_right_answer():
    for row in chunking()["rows"]:
        assert row["error"] < 1e-4


def test_chunked_sum_with_one_chunk_is_sequential():
    values = np.arange(1.0, 17.0)
    assert mp.chunked_sum(values, 1, "float64") == pytest.approx(float(np.sum(values)))


# ------------------------------------------------------------------ products


def test_a_half_accumulator_gets_worse_with_the_inner_dimension():
    assert products()["the_narrow_one_grows"]
    assert products()["excess_slope"] > 0.4


def test_a_single_accumulator_does_not():
    assert products()["the_wide_one_is_flat"]
    assert abs(products()["wide_slope"]) < 0.25


def test_the_wider_accumulator_is_always_at_least_as_good():
    assert products()["the_wide_one_is_always_better"]
    assert products()["worst_gain"] > 2.0


def test_a_product_in_double_precision_is_the_reference():
    rng = np.random.default_rng(5)
    left = rng.standard_normal((3, 4))
    right = rng.standard_normal((4, 2))
    mine = np.asarray(mp.matmul(left, right, "float64", "float64"), dtype=np.float64)
    assert np.allclose(mine, left @ right, atol=1e-12)


def test_a_product_refuses_shapes_that_do_not_multiply():
    with pytest.raises(ValueError):
        mp.matmul(np.ones((2, 3)), np.ones((4, 2)))


def test_every_measurement_carries_a_note():
    for out in (formats(), overflow(), shifting(), underflow(), scaling(), stagnation(),
                ordering(), chunking(), products()):
        assert isinstance(out["note"], str) and out["note"]
