"""Tests for nalib.multiquad.

Four groups. The transformations are checked by integrating something with a known answer both
ways. The double exponential rule is checked on integrands that are singular at one end, at both
ends, and at neither, and the difference between the two spellings of a both-ends integrand is
checked because that difference is the whole reason the rule returns endpoint distances. The
tensor product machinery is checked in several dimensions at once, never only in two. The
dimension arguments are checked by measuring the exponents rather than quoting them.

Every test sweeps dimensions, levels and integrands.
"""
import math

import numpy as np
import pytest

from nalib import multiquad as mq

LEVELS = [1, 2, 3, 4, 5]
DIMENSIONS = [1, 2, 3, 4]


def sqrt_at_left():
    """1/sqrt(x) on (0, 1): singular at one end only, integral 2."""
    return (lambda x: 1.0 / np.sqrt(np.asarray(x, dtype=float))), 2.0, 0.0, 1.0


def log_at_left():
    """log(x) on (0, 1): singular at one end, integral -1."""
    return (lambda x: np.log(np.asarray(x, dtype=float))), -1.0, 0.0, 1.0


def arcsine():
    """1/sqrt(x(1-x)) on (0, 1): singular at both ends, integral pi."""
    def f(x):
        t = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / np.sqrt(t * (1.0 - t))

    def f_stable(x, d_lo, d_hi):
        return 1.0 / np.sqrt(d_lo * d_hi)

    return f, f_stable, math.pi, 0.0, 1.0


SMOOTH = {
    "sin": (np.sin, lambda a, b: math.cos(a) - math.cos(b)),
    "exp": (np.exp, lambda a, b: math.exp(b) - math.exp(a)),
}


# --------------------------------------------------------------------------- transformations


@pytest.mark.parametrize("kind", ["rational", "exponential"])
def test_an_infinite_range_transforms_to_a_finite_one(kind):
    """exp(-x) over [0, infinity) is 1, whichever map is used to get there."""
    g, lo, hi = mq.transform_to_finite(lambda x: np.exp(-np.asarray(x, dtype=float)), 0.0, kind)
    out = mq.tanh_sinh(g, lo, hi, 6)
    assert out["value"] == pytest.approx(1.0, abs=1e-10)


def test_an_unknown_transform_is_rejected():
    with pytest.raises(ValueError):
        mq.transform_to_finite(np.exp, 0.0, "not a transform")


def test_truncation_needs_a_tail_bound_to_be_honest():
    """The values settle long before the tail is negligible, and only the bound shows it."""
    out = mq.truncation_error(lambda x: np.exp(-np.asarray(x, dtype=float)),
                              lambda t: math.exp(-t), 0.0)
    values = np.asarray(out["value"], dtype=float)
    bounds = np.asarray(out["tail_bound"], dtype=float)
    assert np.all(np.diff(bounds) < 0.0)
    # the tail bound is a real bound on what was discarded
    assert np.all(np.abs(values - 1.0) <= bounds * 1.5 + 1e-12)


@pytest.mark.parametrize("power", [0.25, 0.5, 0.75])
def test_the_substitution_removes_an_algebraic_singularity(power):
    p = float(power)
    exact = 1.0 / (1.0 - p)

    def f(x):
        return np.asarray(x, dtype=float) ** (-p)

    g, lo, hi = mq.remove_endpoint_singularity(f, 0.0, 1.0, p)
    # the transformed integrand is bounded, so a modest Gauss rule nails it
    from nalib import gaussquad as gq
    assert gq.integrate(g, lo, hi, 12) == pytest.approx(exact, rel=1e-10)


def test_a_power_outside_the_valid_range_is_rejected():
    with pytest.raises(ValueError):
        mq.remove_endpoint_singularity(np.sqrt, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        mq.remove_endpoint_singularity(np.sqrt, 0.0, 1.0, -0.5)


def test_substituting_beats_refining_at_the_same_cost():
    f, exact, a, b = sqrt_at_left()
    out = mq.substitution_beats_brute_force(f, exact, a, b, 0.5)
    assert bool(np.all(out["substitution_wins"]))
    assert float(out["substituted_error"][-1]) < 1e-12
    assert float(out["raw_error"][-1]) > 1e-6


# --------------------------------------------------------------------------- double exponential


@pytest.mark.parametrize("interval", [(0.0, 1.0), (-1.0, 2.0), (0.5, 3.0), (-2.5, -0.25)])
@pytest.mark.parametrize("kind", sorted(SMOOTH))
def test_it_gets_a_smooth_integrand_right(interval, kind):
    a, b = interval
    f, antiderivative = SMOOTH[kind]
    out = mq.tanh_sinh(f, a, b, 5)
    assert out["value"] == pytest.approx(antiderivative(a, b), rel=1e-12, abs=1e-13)


@pytest.mark.parametrize("case", ["sqrt", "log"])
def test_a_singularity_at_one_end_reaches_machine_precision(case):
    f, exact, a, b = sqrt_at_left() if case == "sqrt" else log_at_left()
    for level in (3, 4, 5):
        out = mq.tanh_sinh(f, a, b, level)
        assert out["value"] == pytest.approx(exact, abs=1e-14)


def test_the_nodes_are_never_the_endpoints():
    out = mq.tanh_sinh(np.sin, 0.0, 1.0, 4)
    assert out["smallest_distance_to_an_endpoint"] > 0.0
    assert np.all(np.asarray(out["distance_to_lo"]) > 0.0)
    assert np.all(np.asarray(out["distance_to_hi"]) > 0.0)


def test_the_distances_are_accurate_where_the_node_position_is_not():
    """The point of returning distances at all.

    Past a few units of ``t`` the node is closer to the endpoint than machine epsilon, so ``x``
    rounds to the endpoint exactly and ``hi - x`` computed from it is zero or garbage. The
    distance the rule computed is still correct to full relative precision.
    """
    out = mq.tanh_sinh(np.sin, 0.0, 1.0, 4)
    x = np.asarray(out["nodes"])
    d_hi = np.asarray(out["distance_to_hi"])
    saturated = x >= 1.0
    assert bool(np.any(saturated)), "expected some nodes to round to the endpoint"
    # those nodes have a positive, meaningful distance even though 1 - x is zero
    assert np.all(d_hi[saturated] > 0.0)
    assert float(np.min(d_hi)) < 1e-30


@pytest.mark.parametrize("level", LEVELS)
def test_the_point_count_roughly_doubles_with_each_level(level):
    coarse = mq.tanh_sinh(np.sin, 0.0, 1.0, level)["points"]
    fine = mq.tanh_sinh(np.sin, 0.0, 1.0, level + 1)["points"]
    assert 1.8 * coarse <= fine <= 2.2 * coarse + 2


def test_a_reversed_interval_and_a_negative_level_are_rejected():
    with pytest.raises(ValueError):
        mq.tanh_sinh(np.sin, 1.0, 0.0, 4)
    with pytest.raises(ValueError):
        mq.tanh_sinh(np.sin, 0.0, 1.0, -1)


def test_a_both_ends_singularity_needs_the_distances():
    """The finding that a plain implementation hides.

    Written in terms of ``x``, the integrand forms ``1 - x`` itself, which has no digits left at
    the outer nodes, and the rule stalls at 1.5e-08 forever. Given the distances, the same rule
    on the same nodes reaches exactly zero error.
    """
    f, f_stable, exact, a, b = arcsine()
    naive = [abs(mq.tanh_sinh(f, a, b, k)["value"] - exact) for k in (2, 3, 4, 5)]
    stable = [abs(mq.tanh_sinh(f_stable, a, b, k, None, True)["value"] - exact)
              for k in (2, 3, 4, 5)]
    # the naive version never improves
    assert min(naive) > 1e-10
    assert max(naive) < 1e-6
    # the distance aware one converges to nothing
    assert stable[-1] < 1e-14
    assert stable[-1] < naive[-1] / 1e6


def test_the_comparison_against_gauss_reports_both_spellings():
    out = mq.handles_a_singularity_without_being_told()
    assert bool(np.all(out["double_exponential_wins"]))
    assert bool(np.all(out["distances_help"][1:]))
    assert float(np.min(out["distance_aware_error"])) < 1e-14
    assert float(np.min(out["double_exponential_error"])) > 1e-10


def test_the_convergence_on_a_smooth_integrand_is_not_a_power_law():
    f, antiderivative = SMOOTH["sin"]
    out = mq.tanh_sinh_convergence(f, antiderivative(0.0, 1.0), 0.0, 1.0, [1, 2, 3])
    errors = np.asarray(out["errors"], dtype=float)
    # three levels take it from 1e-05 to machine precision, which no algebraic order does
    assert errors[0] > 1e-6
    assert errors[-1] < 1e-15


# --------------------------------------------------------------------------- tensor products


@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("n", [2, 3, 5])
def test_the_tensor_rule_has_the_right_shape_and_total_weight(d, n):
    points, weights = mq.tensor_rule(mq.gauss_rule_1d, [n] * d, [(0.0, 2.0)] * d)
    assert points.shape == (n ** d, d)
    assert weights.shape == (n ** d,)
    assert float(np.sum(weights)) == pytest.approx(2.0 ** d, rel=1e-12)


@pytest.mark.parametrize("d", DIMENSIONS)
def test_a_single_node_count_is_broadcast_across_the_axes(d):
    points, _ = mq.tensor_rule(mq.gauss_rule_1d, [4], [(0.0, 1.0)] * d)
    assert points.shape == (4 ** d, d)


def test_mismatched_counts_and_boxes_are_rejected():
    with pytest.raises(ValueError):
        mq.tensor_rule(mq.gauss_rule_1d, [3, 4, 5], [(0.0, 1.0)] * 2)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_composite_simpson_as_a_one_dimensional_rule_needs_an_odd_count(n):
    with pytest.raises(ValueError):
        mq.simpson_rule_1d(n, 0.0, 1.0)


@pytest.mark.parametrize("n", [3, 5, 9, 17])
def test_the_simpson_axis_rule_integrates_a_cubic_exactly(n):
    x, w = mq.simpson_rule_1d(n, 0.0, 2.0)
    assert x.size == n and w.size == n
    assert float(np.sum(w)) == pytest.approx(2.0, rel=1e-12)
    assert float(np.sum(w * x ** 3)) == pytest.approx(4.0, rel=1e-12)


@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("rule", ["gauss", "simpson"])
def test_a_box_integral_is_right_in_every_dimension(d, rule):
    f, exact = mq.cube_integrand("smooth", d)
    # Simpson is a fixed order 4 rule, so it needs more nodes than Gauss to reach the same
    # accuracy; asking both for 1e-06 at 9 nodes is a test of the tolerance, not of the rule
    counts = [6] * d if rule == "gauss" else [17] * d
    rule_1d = mq.gauss_rule_1d if rule == "gauss" else mq.simpson_rule_1d
    counter = [0]
    got = mq.integrate_box(f, [(0.0, 1.0)] * d, counts, rule_1d, counter)
    assert got == pytest.approx(exact, rel=1e-6)
    assert counter[0] == counts[0] ** d


def test_an_unknown_cube_integrand_is_rejected():
    with pytest.raises(ValueError):
        mq.cube_integrand("not an integrand", 2)


def test_an_iterated_integral_handles_a_triangle():
    # integral of x*y over 0 <= y <= x <= 1 is 1/8
    got = mq.iterated(lambda p: p[:, 0] * p[:, 1], 0.0, 1.0,
                      lambda x: 0.0, lambda x: x, 8)
    assert got == pytest.approx(0.125, abs=1e-13)


def test_an_iterated_integral_handles_a_curved_boundary():
    # the area of a quarter disc; the boundary has a vertical tangent, so this is not exact
    got = mq.iterated(lambda p: np.ones(p.shape[0]), 0.0, 1.0,
                      lambda x: 0.0, lambda x: math.sqrt(max(0.0, 1.0 - x * x)), 40)
    assert got == pytest.approx(math.pi / 4.0, abs=1e-5)


# --------------------------------------------------------------------------- dimension


@pytest.mark.parametrize("d", DIMENSIONS)
def test_the_order_per_evaluation_is_the_order_per_axis_over_d(d):
    f, exact = mq.cube_integrand("smooth", d)
    out = mq.tensor_convergence(f, exact, [(0.0, 1.0)] * d, [3, 5, 9, 17], mq.simpson_rule_1d)
    assert out["dimension"] == d
    assert out["order_per_evaluation"] == pytest.approx(out["order_per_axis"] / d, rel=1e-6)


def test_a_fixed_order_rule_keeps_its_order_per_axis_and_loses_it_per_evaluation():
    """The curse, stated as two numbers rather than as a warning.

    Composite Simpson has order 4 in every dimension. Divided by the dimension it is 1.2 by
    four dimensions, so a fourth order rule is behaving like a first order one.
    """
    out = mq.curse_of_dimensionality()
    per_axis = np.asarray(out["order_per_axis"], dtype=float)
    per_evaluation = np.asarray(out["order_per_evaluation"], dtype=float)
    d = np.asarray(out["dimension"], dtype=float)
    assert np.max(per_axis) - np.min(per_axis) < 0.05
    assert np.all(np.diff(per_evaluation) < 0.0)
    assert np.max(np.abs(per_evaluation - per_axis / d)) < 1e-6


def test_a_geometric_rule_cannot_be_used_to_show_the_curse():
    """Why `curse_of_dimensionality` uses Simpson and not Gauss.

    Gauss on an analytic integrand converges geometrically, so a power law fit returns whatever
    the fitting range happens to give, around 21 here. That number is not an order and dividing
    it by the dimension demonstrates nothing.
    """
    f, exact = mq.cube_integrand("smooth", 1)
    out = mq.tensor_convergence(f, exact, [(0.0, 1.0)], [2, 3, 4, 5, 6], mq.gauss_rule_1d)
    assert out["order_per_axis"] > 10.0


@pytest.mark.parametrize("d", [1, 3, 6, 10])
def test_monte_carlo_converges_at_one_half_in_every_dimension(d):
    f, exact = mq.cube_integrand("smooth", d)
    out = mq.monte_carlo_rate(f, exact, [(0.0, 1.0)] * d)
    assert out["fitted_rate"] == pytest.approx(0.5, abs=0.08)


@pytest.mark.parametrize("d", [1, 2, 5])
def test_the_monte_carlo_standard_error_is_the_right_size(d):
    f, exact = mq.cube_integrand("smooth", d)
    out = mq.monte_carlo(f, [(0.0, 1.0)] * d, 20000, seed=42)
    assert abs(out["value"] - exact) < 4.0 * out["standard_error"]
    assert out["dimension"] == d
    assert out["volume"] == pytest.approx(1.0)


def test_monte_carlo_needs_at_least_two_samples():
    with pytest.raises(ValueError):
        mq.monte_carlo(lambda p: np.ones(p.shape[0]), [(0.0, 1.0)], 1)


def test_monte_carlo_is_reproducible():
    f, exact = mq.cube_integrand("smooth", 3)
    a = mq.monte_carlo(f, [(0.0, 1.0)] * 3, 5000, seed=7)
    b = mq.monte_carlo(f, [(0.0, 1.0)] * 3, 5000, seed=7)
    assert a["value"] == b["value"]


def test_the_crossover_depends_on_the_integrand_not_only_on_the_dimension():
    """The comparison everyone gets to come out however they like.

    On a product of cosines, analytic and separable, the tensor rule is ahead at every dimension
    tested. On a product of square roots of distances, which merely has a kink on each axis,
    Monte Carlo is ahead from three dimensions on. Same budget, same method, opposite answer.
    """
    smooth = mq.monte_carlo_crossover(kind="smooth")
    rough = mq.monte_carlo_crossover(kind="rough")
    assert smooth["first_dimension_monte_carlo_wins"] is None
    assert rough["first_dimension_monte_carlo_wins"] == 3
    assert not bool(np.any(smooth["monte_carlo_wins"]))
    assert bool(np.all(rough["monte_carlo_wins"][2:]))


def test_both_methods_get_the_same_budget():
    out = mq.monte_carlo_crossover(dimensions=[2, 3, 4], budget=2401)
    evaluations = np.asarray(out["evaluations"])
    per_axis = np.asarray(out["nodes_per_axis"])
    d = np.asarray(out["dimension"])
    assert np.all(evaluations == per_axis ** d)
