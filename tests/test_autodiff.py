"""Tests for nalib.autodiff.

Four groups. The algebra group checks the dual number rules one at a time against derivatives that
can be written down: the product rule, the quotient rule, the chain rule through each elementary
function, and the fact that a constant carries a zero derivative. If any of those is wrong every
number in this module is wrong, and nothing else here would catch it.

The exactness group asserts what makes automatic differentiation a different object from a finite
difference: the answer matches an analytic derivative to rounding, with no step size anywhere, and
the two modes agree with each other. The finite difference comparison is included so the claim is
a comparison and not an assertion.

The cost group asserts the two counts that decide which mode to use. Reverse mode's operation count
does not depend on the input dimension and forward mode's is exactly proportional to it, both
checked as counted integers rather than as timings, and the tape length is exactly the number of
elementary operations.

The honesty group asserts the failures. At a kink the answer depends on how the source was written,
and three different values come back for one mathematical function. Differentiating an iterative
solver by unrolling it gives the derivative of the approximation, which is only the right answer
once the approximation has converged.
"""
import functools
import math

import numpy as np
import pytest

from nalib import autodiff as ad


@functools.lru_cache(maxsize=None)
def exactness():
    return ad.forward_mode_is_exact_with_no_step_size()


@functools.lru_cache(maxsize=None)
def floors():
    return ad.a_finite_difference_has_a_floor_and_this_does_not()


@functools.lru_cache(maxsize=None)
def costs():
    return ad.a_gradient_costs_one_reverse_sweep()


@functools.lru_cache(maxsize=None)
def tapes():
    return ad.the_tape_is_the_price()


@functools.lru_cache(maxsize=None)
def checkpoints():
    return ad.checkpointing_trades_memory_for_time()


@functools.lru_cache(maxsize=None)
def shapes():
    return ad.which_mode_wins_is_only_the_shape()


@functools.lru_cache(maxsize=None)
def kinks():
    return ad.it_differentiates_the_program_not_the_function()


@functools.lru_cache(maxsize=None)
def unrolling():
    return ad.unrolling_a_solver_converges_to_the_implicit_answer()


@functools.lru_cache(maxsize=None)
def second():
    return ad.forward_over_reverse_gives_an_exact_hessian()


# ------------------------------------------------------------------ the algebra


def test_a_seeded_dual_carries_its_own_derivative():
    x = ad.Dual(3.0, 1.0)
    assert x.value == 3.0 and x.derivative == 1.0
    assert ad.Dual(3.0).derivative == 0.0


def test_the_product_rule_falls_out_of_the_algebra():
    x, y = ad.Dual(2.0, 1.0), ad.Dual(5.0, 0.0)
    product = x * y
    assert product.value == 10.0
    assert product.derivative == 5.0
    both = ad.Dual(2.0, 1.0) * ad.Dual(5.0, 1.0)
    assert both.derivative == 5.0 + 2.0


def test_the_quotient_rule_too():
    x, y = ad.Dual(2.0, 1.0), ad.Dual(5.0, 1.0)
    out = x / y
    assert out.value == pytest.approx(0.4)
    assert out.derivative == pytest.approx((1.0 * 5.0 - 2.0 * 1.0) / 25.0)


def test_the_power_rule():
    out = ad.Dual(3.0, 1.0) ** 4
    assert out.value == pytest.approx(81.0)
    assert out.derivative == pytest.approx(4.0 * 27.0)


def test_a_constant_has_a_zero_derivative():
    out = 7.0 * ad.Dual(2.0, 1.0) + 3.0
    assert out.value == 17.0
    assert out.derivative == 7.0


def test_subtraction_works_from_both_sides():
    assert (5.0 - ad.Dual(2.0, 1.0)).derivative == -1.0
    assert (ad.Dual(2.0, 1.0) - 5.0).derivative == 1.0
    assert (5.0 / ad.Dual(2.0, 1.0)).derivative == pytest.approx(-5.0 / 4.0)


def test_each_elementary_function_carries_its_own_chain_rule():
    at = 0.7
    checks = ((ad.sin, math.cos(at)), (ad.cos, -math.sin(at)), (ad.exp, math.exp(at)),
              (ad.log, 1.0 / at), (ad.sqrt, 0.5 / math.sqrt(at)),
              (ad.tanh, 1.0 - math.tanh(at) ** 2))
    for f, wanted in checks:
        assert f(ad.Dual(at, 1.0)).derivative == pytest.approx(wanted, rel=1e-14)


def test_the_elementary_functions_still_work_on_plain_floats():
    for f, reference in ((ad.sin, math.sin), (ad.cos, math.cos), (ad.exp, math.exp),
                         (ad.log, math.log), (ad.sqrt, math.sqrt), (ad.tanh, math.tanh)):
        assert f(0.9) == pytest.approx(reference(0.9), rel=1e-15)


def test_a_tape_records_one_entry_per_operation():
    tape = ad.Tape()
    x = tape.variable(2.0)
    y = tape.variable(3.0)
    out = x * y + x
    assert len(tape) == 4
    adjoint = tape.backward(out.index)
    assert adjoint[x.index] == pytest.approx(4.0)
    assert adjoint[y.index] == pytest.approx(2.0)


def test_the_tape_handles_reflected_operators():
    tape = ad.Tape()
    x = tape.variable(4.0)
    out = 10.0 - x + 3.0 * x - 2.0 / x
    adjoint = tape.backward(out.index)
    assert adjoint[x.index] == pytest.approx(-1.0 + 3.0 + 2.0 / 16.0)


def test_a_variable_used_twice_accumulates_both_paths():
    tape = ad.Tape()
    x = tape.variable(3.0)
    out = x * x
    assert tape.backward(out.index)[x.index] == pytest.approx(6.0)


# ------------------------------------------------------------------ exactness


def test_both_modes_match_the_analytic_derivative():
    assert exactness()["both_are_exact"]
    assert exactness()["worst_forward"] < 1e-14
    assert exactness()["worst_reverse"] < 1e-14


def test_the_two_modes_agree_with_each_other():
    assert exactness()["the_modes_agree"]


def test_automatic_differentiation_beats_every_finite_difference_step():
    assert floors()["automatic_beats_both"]
    assert floors()["automatic_is_exact"]


def test_the_finite_difference_optimum_is_where_lesson_68_said():
    assert floors()["the_best_step_is_where_it_was_predicted"]
    assert 0.2 < floors()["forward_step_over_the_detailed_one"] < 5.0


def test_the_central_difference_beats_the_forward_one():
    assert floors()["best_central"]["central"] < floors()["best_forward"]["forward"]


def test_the_plain_root_eps_rule_is_only_a_scale():
    assert floors()["the_constants_matter"]


def test_the_finite_difference_curve_is_a_v():
    rows = floors()["rows"]
    smallest = min(range(len(rows)), key=lambda i: rows[i]["forward"])
    assert 0 < smallest < len(rows) - 1
    assert rows[0]["forward"] > rows[smallest]["forward"]
    assert rows[-1]["forward"] > rows[smallest]["forward"]


# ------------------------------------------------------------------ cost


def test_reverse_mode_does_not_care_about_the_input_dimension():
    assert costs()["reverse_is_flat"]
    assert costs()["worst_reverse_ratio"] < 4.0


def test_forward_mode_costs_one_sweep_per_input():
    assert costs()["forward_grows_with_the_dimension"]
    for row in costs()["rows"]:
        assert row["forward_ratio"] == pytest.approx(row["inputs"], rel=1e-9)


def test_the_saving_grows_with_the_dimension():
    assert costs()["largest_saving"] > 10.0
    assert costs()["the_modes_agree"]


def test_the_tape_length_is_the_operation_count():
    assert tapes()["the_tape_grows_linearly"]
    for row in tapes()["rows"]:
        assert row["tape"] == 3 * row["depth"] + 1


def test_forward_mode_memory_does_not_grow():
    assert tapes()["forward_memory_is_constant"]


def test_checkpointing_does_not_change_the_answer():
    assert checkpoints()["answers_are_unchanged"]
    for row in checkpoints()["rows"]:
        assert row["error"] < 1e-10


def test_checkpointing_costs_exactly_one_extra_forward_pass():
    assert checkpoints()["the_work_is_two_passes_whatever_the_gap"]
    assert checkpoints()["extra_work"] == pytest.approx(2.0, abs=1e-9)


def test_the_memory_optimum_is_near_the_square_root_of_the_depth():
    assert checkpoints()["the_memory_has_an_interior_minimum"]
    assert checkpoints()["the_optimum_is_near_the_square_root"]
    assert checkpoints()["largest_saving"] > 5.0


def test_forward_mode_wins_when_there_are_few_inputs():
    assert shapes()["forward_wins_tall"]
    assert shapes()["largest_forward_advantage"] > 5.0


def test_reverse_mode_wins_when_there_are_few_outputs():
    assert shapes()["reverse_wins_wide"]
    assert shapes()["largest_reverse_advantage"] > 5.0


def test_the_two_modes_produce_the_same_jacobian():
    assert shapes()["the_modes_agree"]


def test_the_crossover_is_near_the_square_jacobian():
    square = [row for row in shapes()["rows"] if row["inputs"] == row["outputs"]]
    assert square
    for row in square:
        assert 0.5 < row["ratio"] < 2.0


# ------------------------------------------------------------------ honesty


def test_a_kink_returns_a_number_without_complaining():
    assert kinks()["it_answers_without_complaining"]


def test_the_answer_at_a_kink_depends_on_how_it_was_written():
    assert kinks()["the_answer_depends_on_how_it_was_written"]
    assert kinks()["distinct_answers"] >= 3


def test_the_derivative_jumps_across_the_kink():
    assert kinks()["it_jumps_across_the_kink"]
    assert abs(kinks()["nearby"][0]["derivative"] - kinks()["nearby"][-1]["derivative"]) == 2.0


def test_unrolling_one_newton_step_gives_the_wrong_derivative():
    assert unrolling()["one_step_is_wrong"]
    assert unrolling()["rows"][0]["derivative_error"] > 0.1


def test_unrolling_converges_to_the_implicit_derivative():
    assert unrolling()["it_converges"]
    assert unrolling()["final_error"] < 1e-12


def test_the_derivative_error_falls_faster_than_the_solution_error():
    rows = unrolling()["rows"]
    assert all(rows[i]["derivative_error"] >= rows[i + 1]["derivative_error"] - 1e-15
               for i in range(len(rows) - 1))


def test_the_tape_grows_with_the_iteration_count_and_the_implicit_route_does_not():
    assert unrolling()["the_tape_grows_with_the_iterations"]
    assert unrolling()["implicit_tape"] < unrolling()["rows"][-1]["tape"]


def test_the_implicit_derivative_satisfies_the_implicit_function_theorem():
    root, slope = unrolling()["root"], unrolling()["implicit_derivative"]
    parameter = 0.7
    assert root ** 3 + parameter * root - 1.0 == pytest.approx(0.0, abs=1e-12)
    assert slope == pytest.approx(-root / (3.0 * root ** 2 + parameter), rel=1e-12)


# ------------------------------------------------------------------ second derivatives


def test_forward_over_reverse_is_exact():
    assert second()["it_is_exact"]
    assert second()["error"] < 1e-12


def test_the_hessian_comes_out_symmetric_without_being_told_to():
    assert second()["it_comes_out_symmetric_on_its_own"]
    assert second()["asymmetry"] < 1e-12


def test_a_hessian_vector_product_costs_two_sweeps():
    assert second()["the_product_is_cheaper"]
    assert second()["sweeps_for_one_product"] == 2
    assert second()["the_product_is_exact"]


def test_a_finite_difference_hessian_is_not_exact():
    assert second()["finite_differences_are_not"]
    assert second()["rosenbrock_gap"] < 1e-3


def test_the_hessian_of_a_known_quadratic():
    point = np.array([0.2, -1.1, 0.5])
    out = ad.hessian(ad.sum_of_squares, point)
    assert np.allclose(out["hessian"], np.diag([2.0, 4.0, 6.0]), atol=1e-12)


def test_the_hessian_vector_product_matches_the_full_hessian():
    point = np.linspace(-0.5, 0.9, 4)
    direction = np.array([1.0, -2.0, 0.5, 3.0])
    full = ad.hessian(ad.rosenbrock, point)["hessian"]
    product = ad.hessian_vector(ad.rosenbrock, point, direction)["product"]
    assert np.allclose(product, full @ direction, rtol=1e-12)


def test_every_measurement_carries_a_note():
    for out in (exactness(), floors(), costs(), tapes(), checkpoints(), shapes(), kinks(),
                unrolling(), second()):
        assert isinstance(out["note"], str) and out["note"]
