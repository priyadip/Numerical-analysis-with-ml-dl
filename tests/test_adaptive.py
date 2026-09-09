"""Tests for nalib.adaptive.

Four groups. The local rule and its error estimate are checked against the exact answer and
against the true error they claim to predict, which is the only way to catch a wrong Richardson
divisor. The recursion is checked for the properties it must have: the intervals tile the domain
exactly, the tolerance splits, and the correction costs nothing. The comparison against a uniform
rule is made at equal evaluation counts. The failure modes are checked by measuring a miss, not
by assuming one.

Every test sweeps integrands, intervals and tolerances.
"""
import math

import numpy as np
import pytest

from nalib import adaptive as ad

INTERVALS = [(0.0, 1.0), (-1.0, 2.0), (0.5, 3.0), (-2.5, -0.25)]
TOLERANCES = [1e-4, 1e-6, 1e-8, 1e-10]

SMOOTH = {
    "sin": (np.sin, lambda a, b: math.cos(a) - math.cos(b)),
    "exp": (np.exp, lambda a, b: math.exp(b) - math.exp(a)),
    "cubic": (lambda x: np.asarray(x, dtype=float) ** 3 - 2.0 * np.asarray(x, dtype=float),
              lambda a, b: (b ** 4 - a ** 4) / 4.0 - (b ** 2 - a ** 2)),
}


def sqrt_case():
    return (lambda x: np.sqrt(np.asarray(x, dtype=float))), 2.0 / 3.0, 0.0, 1.0


def lorentzian(half_width):
    w = float(half_width)

    def f(x):
        return 1.0 / (w ** 2 + (np.asarray(x, dtype=float) - 0.3) ** 2)

    exact = (math.atan(0.7 / w) - math.atan(-0.3 / w)) / w
    return f, exact, 0.0, 1.0


# --------------------------------------------------------------------------- the local rule


@pytest.mark.parametrize("interval", INTERVALS)
def test_simpson_on_one_interval_is_exact_on_a_cubic(interval):
    a, b = interval
    f, antiderivative = SMOOTH["cubic"]
    assert ad.simpson(f, a, b) == pytest.approx(antiderivative(a, b), rel=1e-12, abs=1e-12)


def test_simpson_costs_three_evaluations():
    counter = [0]
    ad.simpson(np.sin, 0.0, 1.0, counter)
    assert counter[0] == 3


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_the_divisor_is_two_to_the_order_minus_one(order):
    out = ad.local_error_estimate(np.sin, 0.0, 1.0, order)
    assert out["divisor"] == 2.0 ** order - 1.0


def test_the_simpson_divisor_is_fifteen_not_seven():
    """The off by one that does not announce itself.

    Simpson's local error is ``C h^5``, so halving gives a ratio of 16 and a divisor of 15.
    Using the composite order 4 in ``2^(p-1) - 1`` gives 7, an estimate 15/7 too large. Nothing
    fails: the routine just becomes conservative and spends about twice the evaluations.
    """
    out = ad.local_error_estimate(np.sin, 0.0, 1.0, 4)
    assert out["divisor"] == 15.0


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("kind", ["sin", "exp"])
def test_the_estimate_predicts_the_true_error_of_the_fine_rule(interval, kind):
    """The check a wrong divisor cannot survive.

    The estimate claims to be ``I - fine``. On a smooth integrand over a modest interval the
    ratio of the two is within a few percent of 1, and a divisor of 7 instead of 15 would make
    it about 2.14.
    """
    a, b = interval
    f, antiderivative = SMOOTH[kind]
    edges = np.linspace(a, b, 5)
    checked = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        out = ad.local_error_estimate(f, float(lo), float(hi))
        true_error = antiderivative(float(lo), float(hi)) - out["fine"]
        if abs(true_error) < 1e-14:
            continue
        assert out["estimate"] / true_error == pytest.approx(1.0, abs=0.15)
        checked += 1
    assert checked >= 3


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("kind", ["sin", "exp"])
def test_the_correction_beats_the_fine_rule(interval, kind):
    a, b = interval
    f, antiderivative = SMOOTH[kind]
    lo = a
    hi = a + 0.25 * (b - a)
    want = antiderivative(lo, hi)
    out = ad.local_error_estimate(f, lo, hi)
    assert abs(out["corrected"] - want) < abs(out["fine"] - want)
    assert abs(out["fine"] - want) < abs(out["coarse"] - want)


# --------------------------------------------------------------------------- the recursion


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("tol", TOLERANCES)
def test_the_accepted_intervals_tile_the_domain_exactly(interval, tol):
    a, b = interval
    out = ad.integrate(np.sin, a, b, tol)
    edges = out["intervals"]
    assert float(edges[0, 0]) == pytest.approx(a)
    assert float(edges[-1, 1]) == pytest.approx(b)
    # no gaps and no overlaps
    assert np.max(np.abs(edges[1:, 0] - edges[:-1, 1])) < 1e-12
    assert float(np.sum(edges[:, 1] - edges[:, 0])) == pytest.approx(b - a, rel=1e-12)


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("kind", sorted(SMOOTH))
@pytest.mark.parametrize("tol", TOLERANCES)
def test_it_gets_the_right_answer_on_a_smooth_integrand(interval, kind, tol):
    a, b = interval
    f, antiderivative = SMOOTH[kind]
    want = antiderivative(a, b)
    out = ad.integrate(f, a, b, tol)
    assert out["value"] == pytest.approx(want, rel=1e-8, abs=max(tol, 1e-12))
    assert not out["hit_the_depth_limit"]


@pytest.mark.parametrize("min_depth", [0, 1, 3, 5])
def test_min_depth_forces_at_least_that_many_subdivisions(min_depth):
    out = ad.integrate(np.sin, 0.0, 1.0, 1e-6, min_depth=min_depth)
    assert int(np.min(out["depths"])) >= min_depth
    assert out["panel_count"] >= 2 ** min_depth


def test_a_reversed_interval_is_rejected():
    with pytest.raises(ValueError):
        ad.integrate(np.sin, 1.0, 0.0, 1e-8)


def test_the_depth_limit_is_reported_when_it_is_hit():
    f, exact, a, b = lorentzian(1e-4)
    out = ad.integrate(f, a, b, 1e-12, max_depth=8)
    assert out["hit_the_depth_limit"]
    assert out["deepest"] == 8


def test_a_smooth_integrand_never_hits_the_depth_limit():
    out = ad.integrate(np.sin, 0.0, 1.0, 1e-12, max_depth=30)
    assert not out["hit_the_depth_limit"]
    assert out["deepest"] < 30


@pytest.mark.parametrize("tol", TOLERANCES)
def test_a_tighter_tolerance_costs_more_and_delivers_more(tol):
    f, exact, a, b = sqrt_case()
    loose = [0]
    tight = [0]
    a_out = ad.integrate(f, a, b, tol, _counter=loose)
    b_out = ad.integrate(f, a, b, tol * 1e-2, _counter=tight)
    assert tight[0] > loose[0]
    assert abs(b_out["value"] - exact) <= abs(a_out["value"] - exact)


# --------------------------------------------------------------------------- against uniform


def test_adaptivity_is_worth_nothing_on_a_smooth_integrand():
    """The panels come out all the same width, which is the honest negative result."""
    out = ad.where_the_work_went(np.sin, 0.0, 1.0, 1e-10)
    assert out["width_ratio"] <= 2.0


def test_adaptivity_is_worth_a_great_deal_on_an_endpoint_singularity():
    f, exact, a, b = sqrt_case()
    out = ad.where_the_work_went(f, a, b, 1e-10)
    assert out["width_ratio"] > 1e6


@pytest.mark.parametrize("case", ["sqrt", "narrow"])
def test_adaptive_beats_uniform_where_the_integrand_is_local(case):
    if case == "sqrt":
        f, exact, a, b = sqrt_case()
    else:
        f, exact, a, b = lorentzian(1e-4)
    out = ad.against_uniform(f, exact, a, b, [1e-4, 1e-6, 1e-8])
    assert np.all(np.asarray(out["adaptive_evaluations"])
                  == np.asarray(out["uniform_evaluations"]))
    assert bool(np.all(out["adaptive_wins"]))


def test_uniform_beats_adaptive_when_the_integrand_is_resolved_everywhere():
    """The negative result that keeps the comparison honest.

    A Lorentzian of half width 1e-2 on [0, 1] is resolved by composite Simpson at the budgets
    adaptivity spends, so the bookkeeping overhead loses. Adaptivity is not free and it is not
    always right.
    """
    f, exact, a, b = lorentzian(1e-2)
    out = ad.against_uniform(f, exact, a, b, [1e-4, 1e-6])
    assert not bool(np.any(out["adaptive_wins"]))


# --------------------------------------------------------------------------- honesty


@pytest.mark.parametrize("kind", sorted(SMOOTH))
def test_the_tolerance_is_met_with_room_to_spare_on_smooth_integrands(kind):
    f, antiderivative = SMOOTH[kind]
    out = ad.tolerance_is_met(f, antiderivative(0.0, 1.0), 0.0, 1.0, TOLERANCES)
    assert out["always_met"]
    assert out["worst_ratio"] < 1.0


def test_the_tolerance_is_not_met_when_it_is_below_the_roundoff_floor():
    """Asking for more than the arithmetic can give, and not being told.

    A Lorentzian of half width 1e-4 at 1e-12 needs millions of evaluations, and the accumulated
    roundoff then exceeds the tolerance by a wide margin. The routine returns without any
    complaint, so a caller who does not check the answer against something else has no warning.
    """
    f, exact, a, b = lorentzian(1e-4)
    out = ad.tolerance_is_met(f, exact, a, b, [1e-10, 1e-12])
    assert not out["always_met"]
    assert out["worst_ratio"] > 10.0


@pytest.mark.parametrize("kind", ["sin", "exp"])
def test_the_correction_is_free_and_always_helps(kind):
    f, antiderivative = SMOOTH[kind]
    out = ad.correction_is_free_accuracy(f, antiderivative(0.0, 1.0), 0.0, 1.0, TOLERANCES)
    assert out["same_cost"]
    corrected = np.asarray(out["corrected_error"], dtype=float)
    uncorrected = np.asarray(out["uncorrected_error"], dtype=float)
    assert np.all(corrected <= uncorrected)


def test_the_correction_is_worth_orders_of_magnitude_on_a_hard_integrand():
    f, exact, a, b = sqrt_case()
    out = ad.correction_is_free_accuracy(f, exact, a, b, [1e-6, 1e-8, 1e-10])
    corrected = np.asarray(out["corrected_error"], dtype=float)
    uncorrected = np.asarray(out["uncorrected_error"], dtype=float)
    assert out["same_cost"]
    assert float(np.min(uncorrected / np.maximum(corrected, 1e-300))) > 10.0


# --------------------------------------------------------------------------- the spike


def test_a_spike_on_a_sampled_point_is_found_and_one_between_them_is_not():
    """The result a badly chosen demonstration hides.

    The first accept decision sees five points: both ends, the midpoint and the two quarter
    points. A spike on one of them is found immediately. A spike anywhere else is missed
    completely, and the returned value is essentially zero.
    """
    out = ad.spike_centre_decides_everything(1e-3)
    found = np.asarray(out["found"], dtype=bool)
    errors = np.asarray(out["relative_error"], dtype=float)
    centres = np.asarray(out["centre_fraction"], dtype=float)
    assert bool(np.any(found)) and not bool(np.all(found))
    for c, e in zip(centres, errors):
        if abs(c - 0.5) < 1e-12 or abs(c - 0.25) < 1e-12:
            assert e < 1e-3
        else:
            assert e > 0.5


def test_the_missed_spike_is_missed_entirely_not_slightly():
    out = ad.fooled_by_a_narrow_spike(1e-3)
    assert float(out["relative_error"][0]) == pytest.approx(1.0, abs=1e-6)
    assert int(out["evaluations"][0]) < 20


def test_forced_depth_eventually_finds_it():
    out = ad.fooled_by_a_narrow_spike(1e-3)
    depth = out["first_depth_that_finds_it"]
    assert depth is not None
    assert depth == 5
    index = list(out["min_depth"]).index(depth)
    assert float(out["relative_error"][index]) < 1e-6
    # and it was still missed at every shallower depth
    assert np.all(np.asarray(out["relative_error"][:index], dtype=float) > 0.5)


def test_each_halving_of_the_width_costs_about_one_more_level():
    out = ad.spike_width_against_depth([1e-1, 1e-2, 1e-3])
    depths = [d for d in out["depth_needed"] if d is not None]
    assert len(depths) >= 2
    assert depths == sorted(depths)


def test_no_finite_depth_protects_against_every_spike():
    """The limitation stated plainly rather than left implicit.

    At the depths this sweep forces, a spike of width 1e-4 is still missed. That is not a bug
    to fix: a quadrature routine cannot find structure it never samples, and the caller who
    knows the spike is there has to say where.
    """
    out = ad.spike_width_against_depth([1e-3, 1e-4])
    assert out["depth_needed"][-1] is None
    assert out["note"]
    assert max(out["depths_tried"]) >= 8


def test_a_zero_width_spike_is_rejected():
    with pytest.raises(ValueError):
        ad.fooled_by_a_narrow_spike(0.0)
