"""Tests for nalib.sciml.

Four groups. The operator group checks the test problems before anything is concluded from them:
that the blur is symmetric and positive, that its spectrum really does decay geometrically, that
the second difference operator reproduces a second derivative, and that the boundary value problem's
written-down solution actually solves it. Every later claim rests on those.

The regularization group asserts that Tikhonov is ridge, that the filter factors are the ones lesson
94 measured, and that the two ways of choosing a penalty disagree about the penalty far more than
they disagree about the answer. It also asserts semiconvergence in both directions: the error rises
after its minimum, and the minimum moves earlier as the noise grows.

The collocation group asserts the comparison that makes the physics-informed claim checkable. The
same problem is solved with a linear basis, with a fixed nonlinear basis and with a trained one, and
the assertions are on the ordering of the three, not on any one of them.

The adjoint group asserts what lesson 97 predicted: two routes to a gradient that agree only in the
limit, at a memory cost that differs by two orders of magnitude, with the gap falling like the step.
"""
import functools
import math

import numpy as np
import pytest

from nalib import sciml


@functools.lru_cache(maxsize=None)
def grids():
    return sciml.refining_the_grid_makes_an_inverse_problem_worse()


@functools.lru_cache(maxsize=None)
def picard():
    return sciml.the_picard_condition_locates_the_usable_components()


@functools.lru_cache(maxsize=None)
def ridge():
    return sciml.tikhonov_is_the_ridge_of_lesson_94()


@functools.lru_cache(maxsize=None)
def penalties():
    return sciml.a_derivative_penalty_knows_something_the_identity_does_not()


@functools.lru_cache(maxsize=None)
def stopping():
    return sciml.stopping_early_is_regularization()


@functools.lru_cache(maxsize=None)
def collocation():
    return sciml.collocation_beats_the_trained_basis()


@functools.lru_cache(maxsize=None)
def adjoint():
    return sciml.the_adjoint_gives_the_same_gradient_at_constant_memory()


@functools.lru_cache(maxsize=None)
def crossover():
    return sciml.where_a_trained_basis_starts_to_win()


# ------------------------------------------------------------------ the operators


def test_the_blur_is_symmetric_and_positive():
    matrix = sciml.blur(40)["matrix"]
    assert np.allclose(matrix, matrix.T, atol=1e-14)
    assert np.all(matrix > 0.0)


def test_the_blur_smooths_a_spike():
    problem = sciml.blur(64)
    spike = np.zeros(64)
    spike[32] = 1.0
    blurred = problem["matrix"] @ spike
    assert float(np.max(np.abs(np.diff(blurred)))) < float(np.max(np.abs(np.diff(spike))))
    assert int(np.argmax(blurred)) == 32


def test_the_blur_spectrum_decays_geometrically():
    s = np.linalg.svd(sciml.blur(64)["matrix"], compute_uv=False)
    ratios = s[1:20] / s[:19]
    # the first few ratios are close to 1 because the leading singular values are nearly equal;
    # what makes the decay geometric is that the ratio settles, not that it starts small
    assert float(np.max(ratios)) < 0.99
    assert float(np.mean(ratios)) < 0.85
    assert float(np.min(ratios)) > 0.3
    assert s[19] / s[0] < 0.05


def test_the_blur_rejects_a_grid_of_one():
    with pytest.raises(ValueError):
        sciml.blur(1)


def test_the_second_difference_reproduces_a_second_derivative():
    n = 60
    matrix = sciml.second_difference(n)
    h = 1.0 / (n + 1)
    grid = np.arange(1, n + 1) * h
    values = np.sin(math.pi * grid)
    assert np.allclose(matrix @ values, -math.pi ** 2 * values, rtol=2e-3)


def test_the_three_test_signals_differ_in_the_way_they_are_meant_to():
    grid = sciml.blur(96)["grid"]
    smooth = sciml.test_signal(grid, "smooth")
    box = sciml.test_signal(grid, "box")
    assert float(np.max(np.abs(np.diff(box)))) > 10.0 * float(np.max(np.abs(np.diff(smooth))))
    with pytest.raises(ValueError):
        sciml.test_signal(grid, "wishful")


def test_the_boundary_value_problem_solves_itself():
    problem = sciml.boundary_value_problem()
    grid = np.linspace(0.0, 1.0, 2001)
    exact = problem["exact"](grid)
    step = grid[1] - grid[0]
    second = (exact[2:] - 2.0 * exact[1:-1] + exact[:-2]) / step ** 2
    assert np.allclose(second, problem["source"](grid[1:-1]), atol=1e-5)
    assert abs(exact[0]) < 1e-14 and abs(exact[-1]) < 1e-14


def test_the_chebyshev_derivatives_match_finite_differences():
    points = np.linspace(0.15, 0.85, 9)
    step = 1e-5
    basis = sciml.chebyshev_basis(points, 6)
    ahead = sciml.chebyshev_basis(points + step, 6)["values"]
    behind = sciml.chebyshev_basis(points - step, 6)["values"]
    middle = basis["values"]
    assert np.allclose(basis["second"], (ahead - 2.0 * middle + behind) / step ** 2, atol=2e-2)


def test_the_tanh_derivatives_match_finite_differences():
    points = np.linspace(-0.4, 0.9, 11)
    weights = np.array([1.5, -2.0, 0.7])
    offsets = np.array([0.2, -0.5, 1.1])
    step = 1e-5
    basis = sciml.tanh_basis(points, weights, offsets)
    ahead = sciml.tanh_basis(points + step, weights, offsets)["values"]
    behind = sciml.tanh_basis(points - step, weights, offsets)["values"]
    assert np.allclose(basis["first"], (ahead - behind) / (2.0 * step), atol=1e-7)
    assert np.allclose(basis["second"],
                       (ahead - 2.0 * basis["values"] + behind) / step ** 2, atol=1e-3)


def test_tikhonov_rejects_an_order_it_does_not_have():
    problem = sciml.blur(20)
    with pytest.raises(ValueError):
        sciml.tikhonov(problem, np.ones(20), 1e-3, order=1)


# ------------------------------------------------------------------ regularization


def test_refining_the_grid_exhausts_double_precision():
    assert grids()["the_blur_exhausts_double_precision"]
    assert grids()["saturates_at"] is not None


def test_the_extra_grid_points_are_wasted():
    assert grids()["the_extra_grid_points_are_wasted"]
    assert grids()["usable_gain_from_the_last_doubling"] < 10


def test_the_second_difference_operator_is_merely_ill_conditioned():
    assert grids()["the_difference_operator_grows_like_n_squared"]
    assert grids()["rows"][-1]["difference_condition"] < 1e6


def test_the_number_of_usable_components_falls_with_the_noise():
    assert picard()["usable_falls_with_noise"]
    assert picard()["the_pseudoinverse_fails"]


def test_exact_data_is_far_better_but_not_perfect():
    assert picard()["exact_data_is_far_better"]
    assert picard()["even_exact_data_has_a_floor"]


def test_tikhonov_is_the_same_filter_as_ridge():
    assert ridge()["it_is_the_same_filter"]
    assert ridge()["filter_gap"] < 1e-8


def test_regularization_helps_a_lot():
    assert ridge()["regularization_helps"]
    assert ridge()["oracle_error"] < 1.0


def test_the_l_curve_finds_a_good_answer_from_a_bad_penalty():
    assert ridge()["the_corner_is_close_to_the_oracle"]
    assert ridge()["penalty_ratio"] > 3.0
    assert ridge()["error_ratio"] < 1.5


def test_the_two_penalty_operators_nearly_agree():
    assert penalties()["the_two_penalties_nearly_agree"]
    assert penalties()["largest_gain"] < 1.1


def test_the_smoothness_prior_helps_more_on_a_smooth_answer():
    assert penalties()["it_wins_by_less_on_a_discontinuous_one"]
    assert penalties()["smooth_gain"] > penalties()["box_gain"]


def test_the_operator_orders_its_own_directions_by_roughness():
    assert penalties()["the_operator_sorts_its_own_directions_by_smoothness"]
    assert penalties()["roughness_range"] > 10.0


def test_the_ordering_gives_out_where_the_singular_values_do():
    assert penalties()["the_ordering_stops_where_the_singular_values_hit_rounding"]


def test_conjugate_gradient_gets_worse_after_its_minimum():
    assert stopping()["it_falls_then_rises"]
    assert stopping()["worst_rise"] > 100.0


def test_the_best_iteration_moves_earlier_as_the_noise_grows():
    assert stopping()["the_minimum_moves_earlier_with_noise"]
    assert stopping()["rows"][0]["best_iteration"] > stopping()["rows"][-1]["best_iteration"]


def test_early_stopping_matches_tikhonov():
    assert stopping()["it_is_competitive_with_tikhonov"]
    for row in stopping()["rows"]:
        assert row["krylov_over_tikhonov"] < 1.5


def test_conjugate_gradient_only_needs_a_matrix_vector_product():
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    target = np.array([1.0, 2.0])
    run = sciml.conjugate_gradient(lambda v: matrix @ v, target, 10)
    assert np.allclose(run["x"], np.linalg.solve(matrix, target), atol=1e-12)


# ------------------------------------------------------------------ collocation


def test_the_chebyshev_error_falls_spectrally():
    assert collocation()["the_chebyshev_error_falls_spectrally"]
    errors = [row["error"] for row in collocation()["linear"]]
    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))


def test_training_the_nonlinear_basis_helps():
    assert collocation()["training_helps"]
    assert collocation()["training_gain"] > 2.0


def test_the_loss_actually_fell():
    assert collocation()["loss_fell"] > 100.0


def test_a_linear_basis_of_the_same_size_matches_the_trained_one():
    assert collocation()["the_linear_basis_matches_it_in_one_solve"]


def test_a_larger_linear_basis_beats_it_outright():
    assert collocation()["a_bigger_linear_basis_beats_it_outright"]
    assert collocation()["best_chebyshev_error"] < 1e-8


def test_the_trained_fit_costs_many_evaluations_and_the_solve_costs_one():
    assert collocation()["evaluations"] > 10000
    for row in collocation()["linear"]:
        assert row["solves"] == 1


# ------------------------------------------------------------------ the adjoint


def test_the_two_gradients_converge_to_each_other():
    assert adjoint()["they_converge"]
    assert adjoint()["fine_gap"] < 0.1 * adjoint()["coarse_gap"]


def test_the_gap_falls_like_the_step():
    assert adjoint()["the_gap_falls_with_the_step"]
    assert abs(adjoint()["order"] - 1.0) < 0.3


def test_the_adjoint_memory_does_not_depend_on_the_step_count():
    assert adjoint()["adjoint_memory_is_constant"]
    assert adjoint()["unrolled_memory_grows"]
    assert adjoint()["largest_memory_ratio"] > 50.0


def test_the_integrator_reproduces_a_known_flow():
    problem = sciml.flow_problem(2, seed=1)
    theta = problem["parameters"]
    coarse = sciml.integrate(problem, theta, 32)["z"]
    fine = sciml.integrate(problem, theta, 2048)["z"]
    assert float(np.linalg.norm(coarse - fine)) < 1e-6


def test_the_loss_is_zero_at_the_target():
    problem = sciml.flow_problem(2, seed=2)
    endpoint = sciml.integrate(problem, problem["parameters"], 64)["z"]
    problem["target"] = endpoint
    assert sciml.loss(problem, problem["parameters"], 64) < 1e-24


def test_every_measurement_carries_a_note():
    for out in (grids(), picard(), ridge(), penalties(), stopping(), collocation(), adjoint(),
                crossover()):
        assert isinstance(out["note"], str) and out["note"]


# ------------------------------------------------------------------ many dimensions


def test_the_manufactured_solution_solves_its_own_equation():
    for dimension in (1, 2, 3):
        problem = sciml.poisson(dimension)
        rng = np.random.default_rng(dimension)
        points = rng.random((30, dimension)) * 0.8 + 0.1
        step = 1e-4
        laplacian = np.zeros(points.shape[0])
        for axis in range(dimension):
            shift = np.zeros(dimension)
            shift[axis] = step
            laplacian += (problem["exact"](points + shift)
                          - 2.0 * problem["exact"](points)
                          + problem["exact"](points - shift)) / step ** 2
        assert np.allclose(laplacian, problem["source"](points), atol=1e-5)


def test_the_manufactured_solution_vanishes_on_the_boundary():
    for dimension in (1, 2, 4):
        problem = sciml.poisson(dimension)
        rng = np.random.default_rng(dimension)
        points = rng.random((20, dimension))
        points[:, 0] = 0.0
        assert np.max(np.abs(problem["exact"](points))) < 1e-15
        points[:, 0] = 1.0
        assert np.max(np.abs(problem["exact"](points))) < 1e-15


def test_poisson_rejects_a_zero_dimension():
    with pytest.raises(ValueError):
        sciml.poisson(0)


def test_the_tensor_index_set_has_the_size_it_should():
    for dimension in (1, 2, 3, 4):
        assert sciml.tensor_indices(dimension, 4).shape == (4 ** dimension, dimension)


def test_the_sparse_index_set_has_the_binomial_size():
    for dimension in (1, 2, 3, 5):
        for level in (0, 2, 5):
            indices = sciml.sparse_indices(dimension, level)
            assert indices.shape == (math.comb(level + dimension, dimension), dimension)
            assert np.all(indices.sum(axis=1) <= level)


def test_the_sparse_set_is_smaller_than_the_tensor_set_past_two_dimensions():
    for dimension in (3, 4, 5):
        level = 6
        assert (sciml.sparse_indices(dimension, level).shape[0]
                < sciml.tensor_indices(dimension, level + 1).shape[0])


def test_the_index_sets_reject_a_bad_shape():
    with pytest.raises(ValueError):
        sciml.tensor_indices(0, 3)
    with pytest.raises(ValueError):
        sciml.sparse_indices(2, -1)


def test_the_chebyshev_laplacian_matches_finite_differences():
    dimension = 3
    indices = sciml.sparse_indices(dimension, 4)
    rng = np.random.default_rng(0)
    points = rng.random((6, dimension)) * 0.8 + 0.1
    field = sciml.chebyshev_field(points, indices)
    step = 1e-4
    numeric = np.zeros_like(field["laplacian"])
    for axis in range(dimension):
        shift = np.zeros(dimension)
        shift[axis] = step
        numeric += (sciml.chebyshev_field(points + shift, indices)["value"]
                    - 2.0 * field["value"]
                    + sciml.chebyshev_field(points - shift, indices)["value"]) / step ** 2
    scale = np.maximum(np.abs(field["laplacian"]), 1.0)
    assert np.max(np.abs(numeric - field["laplacian"]) / scale) < 1e-5


def test_the_chebyshev_table_sizes_itself_from_the_indices():
    indices = sciml.sparse_indices(2, 7)
    assert sciml.chebyshev_field(np.array([[0.3, 0.4]]), indices)["size"] == indices.shape[0]
    with pytest.raises(ValueError):
        sciml.chebyshev_field(np.array([[0.3, 0.4]]), indices, degree=3)


def test_the_chebyshev_field_rejects_a_width_mismatch():
    with pytest.raises(ValueError):
        sciml.chebyshev_field(np.array([[0.3, 0.4]]), sciml.sparse_indices(3, 2))


def test_the_tanh_laplacian_matches_finite_differences():
    dimension = 4
    rng = np.random.default_rng(1)
    weights = rng.standard_normal((5, dimension))
    offsets = rng.standard_normal(5)
    points = rng.random((7, dimension))
    field = sciml.tanh_field(points, weights, offsets)
    step = 1e-4
    numeric = np.zeros_like(field["laplacian"])
    for axis in range(dimension):
        shift = np.zeros(dimension)
        shift[axis] = step
        numeric += (sciml.tanh_field(points + shift, weights, offsets)["value"]
                    - 2.0 * field["value"]
                    + sciml.tanh_field(points - shift, weights, offsets)["value"]) / step ** 2
    assert np.allclose(numeric, field["laplacian"], atol=1e-5)


def test_the_tanh_shape_slope_is_the_derivative_it_claims_to_be():
    rng = np.random.default_rng(2)
    weights = rng.standard_normal((4, 2))
    offsets = rng.standard_normal(4)
    points = rng.random((5, 2))
    step = 1e-6
    field = sciml.tanh_field(points, weights, offsets)
    ahead = sciml.tanh_field(points, weights, offsets + step)
    behind = sciml.tanh_field(points, weights, offsets - step)
    assert np.allclose((ahead["shape"] - behind["shape"]) / (2.0 * step),
                       field["shape_slope"], atol=1e-6)


def test_the_cube_points_are_inside_and_on_the_faces():
    out = sciml.cube_points(3, 50, 30, seed=4)
    assert out["interior"].shape == (50, 3)
    assert out["boundary"].shape == (30, 3)
    assert np.all(out["interior"] > 0.0) and np.all(out["interior"] < 1.0)
    on_face = np.logical_or(np.isclose(out["boundary"], 0.0), np.isclose(out["boundary"], 1.0))
    assert np.all(on_face.sum(axis=1) >= 1)


def test_cube_points_rejects_an_empty_request():
    with pytest.raises(ValueError):
        sciml.cube_points(2, 0, 5)


def test_the_training_gradients_agree_with_finite_differences():
    dimension, units, weight = 3, 5, 10.0
    problem = sciml.poisson(dimension)
    points = sciml.cube_points(dimension, 40, 20, seed=3)
    inside, edge = points["interior"], points["boundary"]
    source = problem["source"](inside)
    total = inside.shape[0] + edge.shape[0]
    rng = np.random.default_rng(11)
    w = rng.standard_normal((units, dimension))
    b = rng.standard_normal(units)
    c = rng.standard_normal(units) * 0.3

    def objective(w, b, c):
        home = sciml.tanh_field(inside, w, b)
        wall = sciml.tanh_field(edge, w, b)
        r = home["laplacian"] @ c - source
        e = weight * (wall["value"] @ c)
        return 0.5 * float(r @ r + e @ e) / total

    home = sciml.tanh_field(inside, w, b)
    wall = sciml.tanh_field(edge, w, b)
    residual = home["laplacian"] @ c - source
    edge_residual = weight * (wall["value"] @ c)
    grad_c = (home["laplacian"].T @ residual
              + weight * wall["value"].T @ edge_residual) / total
    inner = residual[:, None] * c[None, :]
    grad_w = 2.0 * w * np.sum(inner * home["shape"], axis=0)[:, None]
    grad_w = grad_w + home["norms"][:, None] * ((inner * home["shape_slope"]).T @ inside)
    grad_b = home["norms"] * np.sum(inner * home["shape_slope"], axis=0)
    outer = edge_residual[:, None] * c[None, :] * weight * wall["sech"]
    grad_w = (grad_w + outer.T @ edge) / total
    grad_b = (grad_b + np.sum(outer, axis=0)) / total

    step = 1e-6
    for values, analytic in ((c, grad_c), (b, grad_b)):
        numeric = np.zeros_like(values)
        for i in range(values.size):
            ahead, behind = values.copy(), values.copy()
            ahead[i] += step
            behind[i] -= step
            if values is c:
                numeric[i] = (objective(w, b, ahead) - objective(w, b, behind)) / (2.0 * step)
            else:
                numeric[i] = (objective(w, ahead, c) - objective(w, behind, c)) / (2.0 * step)
        assert np.max(np.abs(analytic - numeric)) < 1e-6 * max(np.max(np.abs(numeric)), 1.0)


def test_training_reduces_its_own_objective():
    problem = sciml.poisson(2)
    points = sciml.cube_points(2, 200, 60, seed=6)
    run = sciml.train_tanh_collocation(problem, points, 20, 400, seed=1)
    assert run["losses"][-1] < 0.05 * run["losses"][0]
    assert run["parameters"] == 20 * 4


def test_a_linear_collocation_solve_beats_a_zero_answer():
    problem = sciml.poisson(2)
    points = sciml.cube_points(2, 400, 120, seed=8)
    indices = sciml.sparse_indices(2, 10)
    out = sciml.linear_collocation(problem, points,
                                   sciml.chebyshev_field(points["interior"], indices),
                                   sciml.chebyshev_field(points["boundary"], indices))
    error = sciml.field_error(problem, lambda t, i=indices, c=out["coefficients"]:
                              sciml.chebyshev_field(t, i)["value"] @ c)
    assert error < 1e-6


def test_the_classical_basis_wins_in_low_dimensions():
    assert crossover()["the_classical_basis_wins_in_low_dimensions"]
    assert crossover()["largest_classical_win"] > 1e6


def test_the_trained_basis_wins_in_high_dimensions():
    assert crossover()["the_trained_basis_wins_somewhere"]
    assert crossover()["largest_trained_win"] > 1.5


def test_the_crossover_is_the_same_at_both_budgets():
    assert crossover()["the_crossover_agrees_across_budgets"]
    assert set(crossover()["crossovers"].values()) == {4}


def test_the_trained_error_grows_more_slowly_with_the_dimension():
    assert crossover()["the_trained_error_is_flatter"]
    for budget, pair in crossover()["slopes"].items():
        assert pair["grid"] > 2.0
        assert pair["trained"] < 1.0


def test_neither_method_is_accurate_past_the_crossover():
    assert crossover()["neither_is_usable_past_the_crossover"]
    assert crossover()["worst_error_past_the_crossover"] > 0.1


def test_the_classical_win_is_far_larger_than_the_trained_one():
    assert crossover()["largest_classical_win"] > 1e6 * crossover()["largest_trained_win"]


def test_the_sparse_level_collapses_as_the_dimension_grows():
    for budget in crossover()["crossovers"]:
        levels = [r["level"] for r in crossover()["rows"] if r["budget"] == budget]
        assert levels[0] >= levels[-1]
        assert levels[-1] < levels[0]
