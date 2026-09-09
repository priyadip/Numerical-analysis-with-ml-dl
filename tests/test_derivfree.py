"""Tests for nalib.derivfree.

Four groups. The golden section group asserts the reduction factor against ``1/phi`` and the call
count against a closed form, so a change to the point placement cannot pass unnoticed. The
parabolic group asserts the two failure modes on purpose, collinear points and a vertex that is a
maximum, because the safeguards other libraries add exist for exactly those and hiding them would
hide the reason.

The order group is the one worth the most. It asserts that the plastic number **cannot** be
recovered in double precision and **can** in extended precision, which is a direct test that lesson
84's barrier is doing what lesson 84 says. A later reader who "fixes" the double precision fit has
to break an assertion that says it should not fit.

The Nelder-Mead group asserts a failure: on a strictly convex function the method stops, with a
simplex at rounding, at a point whose gradient is 1. Convexity is checked separately by the chord
inequality so the failure cannot be explained away as a second minimum.
"""
import functools
import math

import numpy as np
import pytest

from nalib import derivfree as df


@functools.lru_cache(maxsize=None)
def golden():
    return df.the_reduction_is_the_golden_ratio()


@functools.lru_cache(maxsize=None)
def race():
    return df.parabolas_beat_golden_section()


@functools.lru_cache(maxsize=None)
def order():
    return df.the_order_is_hidden_by_the_barrier()


@functools.lru_cache(maxsize=None)
def wells():
    return df.unimodality_is_required()


@functools.lru_cache(maxsize=None)
def mckinnon():
    return df.nelder_mead_can_converge_to_a_non_minimum()


@functools.lru_cache(maxsize=None)
def scaling():
    return df.how_nelder_mead_scales()


# ------------------------------------------------------------------ golden section


def test_the_reduction_factor_is_the_golden_ratio():
    out = golden()
    assert out["the_ratio_is_golden"]
    assert abs(out["mean_ratio"] - df.GOLDEN) < 1e-6


def test_one_new_evaluation_per_step():
    out = golden()
    assert out["one_new_call_per_step"]


def test_the_call_count_matches_the_closed_form():
    out = golden()
    assert out["steps_match_the_prediction"]


@pytest.mark.parametrize("span", [(0.0, 1.0), (-5.0, 5.0), (2.0, 2.5)])
def test_golden_section_brackets_a_minimum_wherever_it_starts(span):
    lo, hi = span
    centre = 0.5 * (lo + hi)
    out = df.golden_section(lambda t: (t - centre) ** 2, lo, hi, tol=1e-10)
    assert abs(out["x"] - centre) < 1e-6
    assert out["bracket"][0] <= out["x"] <= out["bracket"][1]
    assert out["widths"][-1] <= 1e-10


def test_golden_section_rejects_a_backwards_bracket():
    with pytest.raises(ValueError):
        df.golden_section(math.cos, 4.0, 2.0)


@pytest.mark.parametrize("tol", [1e-4, 1e-8, 1e-12])
def test_the_bracket_reaches_the_tolerance_it_was_given(tol):
    out = df.golden_section(math.cos, 2.0, 4.0, tol=tol)
    assert out["widths"][-1] <= tol


# ------------------------------------------------------------------ parabolic interpolation


def test_parabolic_interpolation_is_exact_on_a_parabola():
    run = df.successive_parabolic(lambda t: (t - math.sqrt(2.0)) ** 2 + 1.0,
                                  (0.0, 1.0, 3.0), rounds=3)
    assert abs(run["history"][0] - math.sqrt(2.0)) < 1e-12


def test_parabolic_interpolation_beats_golden_section_on_evaluations():
    out = race()
    assert out["parabolic_calls_to_the_same_accuracy"] < out["golden_calls"]
    assert out["speedup"] > 2.0


def test_it_stops_when_the_three_points_are_collinear():
    run = df.successive_parabolic(lambda t: 0.0 * t + 1.0, (0.0, 1.0, 2.0))
    assert run["calls"] == 0
    assert "collinear" in run["stopped_because"]


def test_it_walks_to_a_maximum_when_the_parabola_opens_downward():
    # the vertex of a downward parabola is a maximum, and nothing here prevents jumping to it
    run = df.successive_parabolic(math.cos, (-1.0, 0.0, 1.0), rounds=5)
    assert run["calls"] >= 1
    assert abs(run["history"][0]) < 1e-9      # the maximum of cos, not a minimum


def test_it_needs_exactly_three_seeds():
    with pytest.raises(ValueError):
        df.successive_parabolic(math.cos, (0.0, 1.0))


# ------------------------------------------------------------------ the hidden order


def test_double_precision_cannot_recover_the_order():
    out = order()
    assert out["double_precision_cannot_see_it"]
    assert out["usable_points_in_double"] < 8


def test_extended_precision_can():
    out = order()
    assert out["extended_precision_can"]
    assert abs(out["high_precision_order"] - df.PLASTIC) < 0.05


def test_the_plastic_number_is_the_root_it_claims_to_be():
    t = df.PLASTIC
    assert abs(t ** 3 - t - 1.0) < 1e-14


@pytest.mark.parametrize("digits", [60, 120, 200])
def test_more_digits_do_not_change_the_order(digits):
    out = df.the_order_is_hidden_by_the_barrier(digits=digits)
    assert abs(out["high_precision_order"] - df.PLASTIC) < 0.1


def test_the_barrier_is_where_lesson_84_says_it_is():
    out = order()
    floor = math.sqrt(float(np.finfo(float).eps))
    reached = out["double_errors"][out["steps_to_reach_the_floor"] - 1]
    assert reached < floor


# ------------------------------------------------------------------ unimodality


def test_the_wells_are_found_not_quoted():
    out = wells()
    assert len(out["wells"]) == 2
    assert out["wells"][0] < 0.0 < out["wells"][1]


def test_every_bracket_converges_and_they_disagree():
    out = wells()
    assert out["every_run_converged"]
    assert out["different_brackets_give_different_answers"]


def test_a_bracket_can_deliver_the_shallower_well():
    assert wells()["a_run_found_the_shallower_well"]


# ------------------------------------------------------------------ Nelder-Mead


def test_nelder_mead_solves_rosenbrock_at_small_sizes():
    for row in scaling()["rows"]:
        if row["variables"] <= 6:
            assert row["solved"]


def test_the_cost_grows_like_a_power_of_the_dimension():
    out = scaling()
    assert 1.4 < out["fitted_power_of_n"] < 2.6


def test_it_fails_somewhere_and_not_at_the_largest_size():
    out = scaling()
    assert out["failures"]
    assert out["the_failure_is_not_the_largest_size"]


def test_mckinnon_stops_at_a_non_stationary_point():
    out = mckinnon()
    assert out["it_stopped"]
    assert out["it_is_not_stationary"]
    assert abs(out["gradient_norm"] - 1.0) < 1e-9


def test_the_point_it_stops_at_is_not_the_minimum():
    out = mckinnon()
    assert out["the_gap_in_value"] > 0.2
    assert float(np.linalg.norm(out["stopped_at"] - out["true_minimizer"])) > 0.4


def test_it_got_there_by_contracting_every_step():
    out = mckinnon()
    assert out["moves"]["contract"] > 0
    assert out["moves"]["reflect"] == 0
    assert out["moves"]["expand"] == 0


def test_the_failure_is_not_a_second_minimum():
    out = df.the_function_really_is_convex()
    assert out["no_counterexample_found"]
    assert out["worst_chord_violation"] <= 1e-12


@pytest.mark.parametrize("dimension", [1, 2, 3, 5])
def test_nelder_mead_works_at_any_size_on_a_bowl(dimension):
    rng = np.random.default_rng(42)
    centre = rng.normal(size=dimension)

    def f(v):
        d = np.asarray(v, dtype=float).ravel() - centre
        return float(d @ d)

    out = df.nelder_mead(f, np.zeros(dimension))
    assert float(np.linalg.norm(out["x"] - centre)) < 1e-6
    assert out["x"].shape == (dimension,)
    assert out["simplex"].shape == (dimension + 1, dimension)


def test_nelder_mead_rejects_an_empty_start():
    with pytest.raises(ValueError):
        df.nelder_mead(lambda v: 0.0, np.zeros(0))
