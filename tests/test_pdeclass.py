"""Tests for nalib.pdeclass.

Four groups. The classification is checked against the five standard equations and against the
one equation whose type changes across the domain, because a classifier that gets a constant
coefficient equation right can still be reading the wrong thing. The stencils are checked against
their Taylor moments, which is a statement about the weights rather than about a test function, so
a stencil that is not what its name says fails here rather than in a plot. The two Laplacians are
checked on harmonic polynomials, where the answer is an exact zero and there is no tolerance to
choose. The last group is well posedness, which is the part of the classification that actually
constrains what a method may do.

The tests worth the most here are the ones that would pass for the wrong reason if written
carelessly: the nine point rule looks fourth order if you only fit a convergence sweep and stop
before the rounding floor, and the Neumann comparison looks fine if you measure the interior
instead of the solve.
"""
import math

import numpy as np
import pytest

from nalib import pdeclass as pc

SIZES = [5, 9, 17, 33]
WIDTHS = [3, 5, 7, 9, 11]


# --------------------------------------------------------------------------- classification


def test_the_five_standard_equations_are_classified_right():
    out = pc.every_standard_equation_is_classified_right()
    assert out["all_agree"]
    assert out["two_real_characteristics_exactly_when_hyperbolic"]
    assert len(out["rows"]) == len(pc.standard_equations())


@pytest.mark.parametrize("name,expected", [("laplace", "elliptic"), ("poisson", "elliptic"),
                                           ("heat", "parabolic"), ("wave", "hyperbolic"),
                                           ("advection", "hyperbolic")])
def test_each_equation_by_name(name, expected):
    a, b, c = pc.standard_equations()[name]["coefficients"]
    assert pc.classify(a, b, c) == expected


def test_the_discriminant_sign_and_the_name_agree_on_random_coefficients():
    rng = np.random.default_rng(42)
    for _ in range(200):
        a, b, c = rng.normal(size=3) * rng.choice([1.0, 1e3, 1e-3])
        d = float(pc.discriminant(a, b, c))
        name = pc.classify(a, b, c)
        if name == "hyperbolic":
            assert d > 0.0
        elif name == "elliptic":
            assert d < 0.0
        else:
            assert abs(d) <= 1e-9 * max(abs(b * b), abs(4.0 * a * c), 1.0)


def test_scaling_the_equation_does_not_change_its_type():
    rng = np.random.default_rng(7)
    for _ in range(100):
        a, b, c = rng.normal(size=3)
        first = pc.classify(a, b, c)
        for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6):
            assert pc.classify(scale * a, scale * b, scale * c) == first


def test_classify_broadcasts_over_arrays_of_any_shape():
    for shape in [(4,), (3, 5), (2, 3, 4)]:
        a = np.ones(shape)
        b = np.zeros(shape)
        c = np.linspace(-1.0, 1.0, int(np.prod(shape))).reshape(shape)
        names = pc.classify(a, b, c)
        assert names.shape == shape
        assert set(np.unique(names)) <= set(pc.TYPES)


def test_a_scalar_gives_a_string_and_an_array_gives_an_array():
    assert isinstance(pc.classify(1.0, 0.0, 1.0), str)
    assert isinstance(pc.classify([1.0, 1.0], [0.0, 0.0], [1.0, -1.0]), np.ndarray)


@pytest.mark.parametrize("n", [11, 21, 41])
def test_tricomi_holds_all_three_types_and_puts_them_in_the_right_places(n):
    out = pc.the_type_can_change_across_the_domain(points=n)
    assert out["all_three_appear"]
    assert out["hyperbolic_is_below_the_axis"]
    assert out["elliptic_is_above_it"]
    assert out["the_parabolic_line_is_measure_zero"]


def test_the_parabolic_set_shrinks_when_the_grid_misses_the_axis():
    out = pc.the_type_can_change_across_the_domain(points=41)
    assert out["grid_through_the_axis"]["parabolic"] > 0.0
    assert out["grid_missing_the_axis"]["parabolic"] == 0.0


def test_characteristic_slopes_are_real_exactly_for_hyperbolic():
    rng = np.random.default_rng(11)
    for _ in range(200):
        a, b, c = rng.normal(size=3)
        slopes = pc.characteristic_slopes(a, b, c)
        real = all(abs(s.imag) < 1e-12 for s in slopes)
        assert real == (pc.classify(a, b, c) != "elliptic")


def test_a_characteristic_slope_solves_its_own_quadratic():
    rng = np.random.default_rng(3)
    for _ in range(100):
        a, b, c = rng.normal(size=3)
        if abs(a) < 1e-3:
            continue
        for m in pc.characteristic_slopes(a, b, c):
            assert abs(a * m * m - b * m + c) < 1e-10 * (abs(a) + abs(b) + abs(c))


def test_conditions_are_defined_for_every_type_and_only_elliptic_is_not_marched():
    out = pc.the_condition_count_matches_the_type()
    assert out["only_elliptic_is_not_marched"]
    assert set(out["types_covered"]) <= set(pc.TYPES)
    for kind in pc.TYPES:
        assert pc.conditions_for(kind)["needs"]


def test_an_unknown_type_is_rejected():
    with pytest.raises(ValueError):
        pc.conditions_for("parahyperbolic")


# --------------------------------------------------------------------------- stencils


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("order", [1, 2])
def test_a_centred_stencil_has_the_order_its_width_allows(width, order):
    # a centred stencil on w points is accurate to w - k, rounded up to the next even number
    # because the odd term cancels by symmetry. For odd w that is w - 1 for both k = 1 and k = 2,
    # so a second derivative gets its extra order for free.
    half = width // 2
    off = np.arange(-half, half + 1)
    plain = width - order
    assert pc.stencil_order(off, order) == plain + plain % 2
    assert pc.stencil_order(off, order) == width - 1


@pytest.mark.parametrize("width", WIDTHS)
def test_every_stencil_annihilates_a_constant(width):
    half = width // 2
    off = np.arange(-half, half + 1)
    for order in (1, 2):
        assert abs(float(np.sum(pc.stencil(off, order)))) < 1e-12


def test_the_three_point_second_difference_is_the_familiar_one():
    got = pc.stencil([-1, 0, 1], 2)
    assert np.allclose(got, [1.0, -2.0, 1.0])


def test_the_three_point_first_difference_is_the_familiar_one():
    got = pc.stencil([-1, 0, 1], 1)
    assert np.allclose(got, [-0.5, 0.0, 0.5])


def test_a_one_sided_stencil_needs_the_same_width_for_the_same_order():
    # the usual shorthand is that one sided rules are worse. What is true is narrower: at equal
    # WIDTH a one sided first difference matches the centred one, and it is the two point rule
    # a boundary is usually written with that costs the order.
    assert pc.stencil_order([-1, 0, 1], 1) == 2
    assert pc.stencil_order([0, 1, 2], 1) == 2
    assert pc.stencil_order([0, 1], 1) == 1


def test_too_few_offsets_is_an_error():
    with pytest.raises(ValueError):
        pc.stencil([0, 1], 2)


def test_the_reported_order_is_measured_and_not_assumed():
    # a deliberately unbalanced stencil: still exact for a second derivative, but only order 1
    assert pc.stencil_order([-1, 0, 1, 2], 2) == 2
    assert pc.stencil_order([0, 1, 2], 2) == 1


@pytest.mark.parametrize("n", [9, 17, 33])
def test_partial_reproduces_a_known_derivative(n):
    x = np.linspace(0.0, 1.0, n)
    h = float(x[1] - x[0])
    values = np.sin(3.0 * x)
    got = pc.partial(values, axis=0, order=2, h=h, accuracy=2)
    want = -9.0 * np.sin(3.0 * x[1:-1])
    assert float(np.max(np.abs(got - want))) < 20.0 * h ** 2


def test_partial_refuses_an_odd_accuracy_order():
    with pytest.raises(ValueError):
        pc.partial(np.zeros(9), axis=0, order=2, h=0.1, accuracy=3)


def test_apply_stencil_works_along_any_axis_of_any_rank():
    rng = np.random.default_rng(42)
    for shape in [(9,), (7, 9), (5, 7, 9)]:
        values = rng.normal(size=shape)
        for axis in range(len(shape)):
            out = pc.apply_stencil(values, [-1, 0, 1], [1.0, -2.0, 1.0], axis=axis, h=1.0)
            expected = list(shape)
            expected[axis] -= 2
            assert out.shape == tuple(expected)


def test_apply_stencil_rejects_an_axis_too_short_for_the_stencil():
    with pytest.raises(ValueError):
        pc.apply_stencil(np.zeros(2), [-1, 0, 1], [1.0, -2.0, 1.0], axis=0)


def test_apply_stencil_rejects_a_weight_count_that_does_not_match():
    with pytest.raises(ValueError):
        pc.apply_stencil(np.zeros(9), [-1, 0, 1], [1.0, -2.0], axis=0)


# --------------------------------------------------------------------------- the Laplacians


@pytest.mark.parametrize("n", SIZES)
def test_both_laplacians_are_exact_on_a_harmonic_polynomial_of_low_degree(n):
    grid = np.linspace(0.0, 1.0, n)
    h = float(grid[1] - grid[0])
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    for degree in range(4):
        values = np.real((gx + 1j * gy) ** degree)
        scale = max(float(np.max(np.abs(values))), 1.0) / h ** 2
        assert float(np.max(np.abs(pc.five_point(values, h, h)))) < 1e-10 * scale
        assert float(np.max(np.abs(pc.nine_point(values, h)))) < 1e-10 * scale


def test_the_nine_point_rule_is_exact_four_degrees_further_than_the_five_point_one():
    out = pc.exact_on_harmonic_polynomials()
    assert out["five_point_exact_up_to"] == 3
    assert out["nine_point_exact_up_to"] == 7
    assert out["no_gaps"]
    assert out["nine_point_order_implied"] == 6
    assert out["five_point_order_implied"] == 2


@pytest.mark.parametrize("points", [7, 9, 13])
def test_the_exactness_degrees_do_not_depend_on_the_grid(points):
    out = pc.exact_on_harmonic_polynomials(points=points)
    assert out["five_point_exact_up_to"] == 3
    assert out["nine_point_exact_up_to"] == 7


def test_the_nine_point_rule_is_only_second_order_on_a_general_function():
    out = pc.orders_of_the_two_laplacians()
    assert out["nine_point_is_only_second_order_otherwise"]
    assert out["five_point_is_second_order_either_way"]
    assert out["nine_point_costs_twice_the_error_when_it_does_not_help"] == pytest.approx(
        2.0, abs=0.05)


def test_the_nine_point_rule_beats_fourth_order_on_a_harmonic_function():
    out = pc.orders_of_the_two_laplacians()
    assert out["nine_point_beats_fourth_order_on_a_harmonic_function"]
    harmonic = out["rows"][0]
    assert harmonic["nine_point_order_above_the_floor"] > 5.0


def test_fitting_past_the_rounding_floor_gives_the_wrong_order():
    out = pc.orders_of_the_two_laplacians()
    harmonic = out["rows"][0]
    assert out["fitting_the_whole_sweep_would_say"] < 3.0
    assert harmonic["nine_point_rows_used"] < harmonic["h"].size


def test_the_rounding_floor_grows_as_the_grid_is_refined():
    out = pc.orders_of_the_two_laplacians()
    floors = out["rows"][0]["rounding_floor"]
    assert bool(np.all(np.diff(floors) > 0.0))


@pytest.mark.parametrize("n", [11, 21])
def test_the_five_point_laplacian_matches_two_second_differences(n):
    rng = np.random.default_rng(42)
    values = rng.normal(size=(n, n + 2))
    hx, hy = 0.3, 0.7
    got = pc.five_point(values, hx, hy)
    xx = pc.partial(values, axis=0, order=2, h=hx)[:, 1:-1]
    yy = pc.partial(values, axis=1, order=2, h=hy)[1:-1, :]
    assert np.allclose(got, xx + yy)


def test_the_laplacians_reject_arrays_that_are_not_two_dimensional():
    with pytest.raises(ValueError):
        pc.five_point(np.zeros(5), 0.1, 0.1)
    with pytest.raises(ValueError):
        pc.nine_point(np.zeros((3, 3, 3)), 0.1)


def test_the_mehrstellen_correction_gains_two_orders():
    out = pc.the_correction_restores_fourth_order()
    assert out["gains_two_orders"]
    assert out["plain_order"] == pytest.approx(2.0, abs=0.15)
    assert out["corrected_order"] == pytest.approx(4.0, abs=0.2)
    assert out["mehrstellen_order"] == pytest.approx(4.0, abs=0.2)


def test_the_discrete_correction_is_no_worse_than_the_exact_one():
    out = pc.the_correction_restores_fourth_order()
    assert out["the_discrete_correction_is_smaller_by"] > 1.0


def test_mehrstellen_rejects_a_source_of_the_wrong_shape():
    with pytest.raises(ValueError):
        pc.mehrstellen(np.zeros((5, 5)), np.zeros((5, 6)), 0.1)


# --------------------------------------------------------------------------- boundaries


def test_a_ghost_point_keeps_second_order_and_one_sided_does_not():
    out = pc.a_ghost_point_keeps_the_order()
    assert out["one_sided_order"] == pytest.approx(1.0, abs=0.1)
    assert out["ghost_point_order"] == pytest.approx(2.0, abs=0.1)
    assert out["ratio_at_the_finest"] > 100.0


def test_the_ghost_point_error_falls_by_four_when_the_grid_halves():
    out = pc.a_ghost_point_keeps_the_order()
    ratios = out["ghost_point_error"][:-1] / out["ghost_point_error"][1:]
    assert bool(np.all(np.abs(ratios - 4.0) < 0.2))


def test_the_one_sided_error_only_halves():
    out = pc.a_ghost_point_keeps_the_order()
    ratios = out["one_sided_error"][:-1] / out["one_sided_error"][1:]
    assert bool(np.all(np.abs(ratios - 2.0) < 0.1))


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_ghost_point_formula_reproduces_a_known_second_derivative(side):
    # u = cos(pi x) on [0, 1] has u'(0) = 0 and u'(1) = 0, so both ends are pure Neumann
    for n in (33, 65, 129):
        x = np.linspace(0.0, 1.0, n)
        h = float(x[1] - x[0])
        values = np.cos(math.pi * x)
        got = pc.ghost_point_neumann(values, h, 0.0, side=side)
        want = -math.pi ** 2 * (1.0 if side == "left" else -1.0)
        assert abs(got - want) < 30.0 * h ** 2


def test_ghost_point_rejects_an_unknown_side():
    with pytest.raises(ValueError):
        pc.ghost_point_neumann(np.zeros(5), 0.1, 0.0, side="top")


# --------------------------------------------------------------------------- well posedness


def test_the_backward_heat_equation_amplifies_every_mode():
    out = pc.the_backward_problem_is_ill_posed()
    assert out["forward_is_a_contraction"]
    assert out["backward_grows_without_bound"]
    assert out["exponent_is_two"]


def test_the_forward_and_backward_factors_are_reciprocal():
    out = pc.the_backward_problem_is_ill_posed()
    assert np.allclose(out["forward_factor"] * out["backward_factor"], 1.0)


@pytest.mark.parametrize("t", [0.01, 0.05, 0.2])
def test_the_amplification_exponent_is_two_whatever_the_time(t):
    out = pc.the_backward_problem_is_ill_posed(time=t)
    assert out["fitted_exponent_in_k"] == pytest.approx(2.0, abs=1e-6)


def test_a_finer_grid_supports_a_larger_wavenumber_and_a_larger_amplification():
    out = pc.the_backward_problem_is_ill_posed()
    assert bool(np.all(np.diff(out["grid_that_supports_k"]) < 0.0))
    assert bool(np.all(np.diff(out["backward_factor"]) > 0.0))
