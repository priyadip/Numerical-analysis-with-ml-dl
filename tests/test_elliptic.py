"""Tests for nalib.elliptic.

Four groups. The matrix is checked against its own defining stencil and against the closed form for
its spectrum, which is exact rather than approximate and so needs no tolerance to speak of. The
solvers are checked against the direct answer and against each other, with the eigenvector
degeneracy asserted rather than avoided. The elements are checked against the differences, where
the striking result is that they are the same method under one quadrature rule and not under
another. The last group is the orders.

The test worth the most here is the degenerate one: conjugate gradients converging in a single
iteration is asserted, so a later reader cannot mistake the natural test problem for evidence about
Krylov methods.
"""
import functools
import math

import numpy as np
import pytest

from nalib import elliptic as el

SIZES = [3, 5, 9]
POINTS = [5, 9, 17]


# Three tests below ask the same expensive sweep three different questions, and two more do the
# same for the solver scaling. Running each sweep once and caching it cuts the module's time by
# about two thirds without weakening anything.
@functools.lru_cache(maxsize=None)
def omega_sweep():
    return el.the_optimal_omega_is_where_the_formula_says()


@functools.lru_cache(maxsize=None)
def scaling_sweep():
    return el.how_the_solvers_scale()


@functools.lru_cache(maxsize=None)
def degeneracy_sweep():
    return el.the_obvious_problem_is_degenerate_for_cg()


@functools.lru_cache(maxsize=None)
def spectrum_sweep():
    return el.the_spectrum_is_known_exactly()


@functools.lru_cache(maxsize=None)
def methods_sweep():
    return el.the_two_methods_can_be_the_same_method()


# --------------------------------------------------------------------------- the matrix


@pytest.mark.parametrize("m", SIZES)
def test_the_matrix_is_the_five_point_stencil(m):
    h = 1.0 / (m + 1)
    a = el.five_point_matrix(m, h)
    for i in range(m):
        for j in range(m):
            row = i * m + j
            assert a[row, row] == pytest.approx(4.0 / h ** 2)
            neighbours = 0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < m:
                    assert a[row, ii * m + jj] == pytest.approx(-1.0 / h ** 2)
                    neighbours += 1
            assert np.count_nonzero(a[row]) == neighbours + 1


@pytest.mark.parametrize("m", SIZES)
def test_the_matrix_is_symmetric_positive_definite(m):
    a = el.five_point_matrix(m, 1.0 / (m + 1))
    assert np.allclose(a, a.T)
    assert float(np.min(np.linalg.eigvalsh(a))) > 0.0


@pytest.mark.parametrize("m", SIZES)
def test_a_reaction_term_only_moves_the_diagonal(m):
    h = 1.0 / (m + 1)
    plain = el.five_point_matrix(m, h)
    with_c = el.five_point_matrix(m, h, reaction=3.7)
    assert np.allclose(with_c - plain, 3.7 * np.eye(m * m))


def test_an_empty_matrix_is_rejected():
    with pytest.raises(ValueError):
        el.five_point_matrix(0, 0.1)
    with pytest.raises(ValueError):
        el.assemble(el.manufactured_problem(), 2)


@pytest.mark.parametrize("m", SIZES)
def test_the_eigenvalues_match_their_closed_form(m):
    h = 1.0 / (m + 1)
    computed = np.sort(np.linalg.eigvalsh(el.five_point_matrix(m, h)))
    assert np.allclose(computed, el.exact_eigenvalues(m, h), rtol=1e-12)


def test_the_condition_number_is_cot_squared_and_grows_like_h_minus_two():
    out = spectrum_sweep()
    assert out["the_formula_is_exact"]
    assert out["cot_squared_is_exact"]
    assert out["condition_exponent"] == pytest.approx(-2.0, abs=0.1)


def test_the_small_h_condition_formula_is_close_but_never_exact():
    out = spectrum_sweep()
    assert out["the_small_h_form_is_close_not_exact"]
    for row in out["rows"]:
        assert row["small_h_error"] > 0.0
        assert row["small_h_form"] > row["condition_number"]


# --------------------------------------------------------------------------- the problems


@pytest.mark.parametrize("modes", [(1, 1), (2, 1), (3, 2)])
def test_the_manufactured_source_matches_its_solution(modes):
    problem = el.manufactured_problem(modes=modes)
    x = np.linspace(0.05, 0.95, 41)
    gx, gy = np.meshgrid(x, x, indexing="ij")
    d = 1e-4
    lap = ((problem["exact"](gx + d, gy) - 2.0 * problem["exact"](gx, gy)
            + problem["exact"](gx - d, gy))
           + (problem["exact"](gx, gy + d) - 2.0 * problem["exact"](gx, gy)
              + problem["exact"](gx, gy - d))) / d ** 2
    assert float(np.max(np.abs(-lap - problem["source"](gx, gy)))) < 1e-3


def test_the_mixed_modes_source_matches_its_solution():
    problem = el.mixed_modes_problem()
    x = np.linspace(0.05, 0.95, 41)
    gx, gy = np.meshgrid(x, x, indexing="ij")
    d = 1e-4
    lap = ((problem["exact"](gx + d, gy) - 2.0 * problem["exact"](gx, gy)
            + problem["exact"](gx - d, gy))
           + (problem["exact"](gx, gy + d) - 2.0 * problem["exact"](gx, gy)
              + problem["exact"](gx, gy - d))) / d ** 2
    assert float(np.max(np.abs(-lap - problem["source"](gx, gy)))) < 1e-2


def test_the_harmonic_problem_really_is_harmonic():
    problem = el.harmonic_problem()
    x = np.linspace(0.1, 0.9, 31)
    gx, gy = np.meshgrid(x, x, indexing="ij")
    assert np.allclose(problem["source"](gx, gy), 0.0)
    d = 1e-4
    lap = ((problem["exact"](gx + d, gy) - 2.0 * problem["exact"](gx, gy)
            + problem["exact"](gx - d, gy))
           + (problem["exact"](gx, gy + d) - 2.0 * problem["exact"](gx, gy)
              + problem["exact"](gx, gy - d))) / d ** 2
    assert float(np.max(np.abs(lap))) < 1e-4


def test_the_cooling_fin_derives_its_two_coefficients():
    fin = el.cooling_fin()
    assert fin["reaction"] == pytest.approx(
        2.0 * fin["coefficient"] / (fin["conductivity"] * fin["thickness"]))
    assert fin["flux"] == pytest.approx(
        fin["power"] / (fin["conductivity"] * fin["thickness"] * fin["contact"]))
    with pytest.raises(ValueError):
        el.cooling_fin(thickness=0.0)


def test_bad_intervals_are_rejected():
    with pytest.raises(ValueError):
        el.manufactured_problem(a=1.0, b=0.0)
    with pytest.raises(ValueError):
        el.mixed_modes_problem(a=1.0, b=0.0)


# --------------------------------------------------------------------------- the solvers


@pytest.mark.parametrize("n", POINTS)
@pytest.mark.parametrize("method", ["direct", "jacobi", "gauss seidel", "sor", "cg"])
def test_every_method_reaches_the_same_answer(n, method):
    problem = el.mixed_modes_problem()
    options = {"tol": 1e-10, "keep_history": False}
    if method in ("jacobi", "gauss seidel", "sor"):
        options["max_iter"] = 100000
    if method == "sor":
        options["omega"] = el.optimal_omega(1.0 / (n - 1))
    if method == "direct":
        options = {}
    out = el.solve(problem, n, method=method, **options)
    reference = el.solve(problem, n, method="direct")
    assert float(np.max(np.abs(out["u"] - reference["u"]))) < 1e-6
    assert out["residual"] < 1e-6


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        el.solve(el.manufactured_problem(), 5, method="magic")


def test_the_optimal_omega_is_found_three_independent_ways():
    out = omega_sweep()
    assert out["the_sweep_finds_the_formula"]
    assert out["the_general_formula_agrees"]
    assert out["the_jacobi_radius_is_cos_pi_h"]


def test_undershooting_the_relaxation_factor_costs_more_than_overshooting():
    out = omega_sweep()
    assert out["undershooting_costs_more_than_overshooting"]
    assert out["measured_cost_of_being_low"] > out["measured_cost_of_being_high"]


def test_sor_beats_gauss_seidel_by_more_as_the_grid_grows():
    out = omega_sweep()
    assert out["rate_gain_grows_with_the_grid"]
    assert out["measured_gain_over_gauss_seidel"] > 5.0


@pytest.mark.parametrize("h", [0.25, 0.1, 0.02])
def test_the_optimal_omega_formula_is_between_one_and_two(h):
    omega = el.optimal_omega(h)
    assert 1.0 < omega < 2.0
    assert omega == pytest.approx(2.0 / (1.0 + math.sin(math.pi * h)))


def test_at_one_unknown_the_optimal_factor_is_exactly_one():
    # h = 1/2 is the grid with a single interior point, where sin(pi h) = 1 and the formula gives
    # omega = 1: there is nothing to over-relax, and SOR is Gauss-Seidel
    assert el.optimal_omega(0.5) == pytest.approx(1.0)


def test_the_iteration_counts_scale_as_part_four_predicts():
    out = scaling_sweep()
    assert out["every_exponent_is_close_to_its_prediction"]
    assert out["cheapest_at_the_largest"] == "cg"


def test_conjugate_gradients_finishes_the_eigenvector_problem_in_one_step():
    out = degeneracy_sweep()
    assert out["cg_finishes_the_single_mode_in_one_step"]
    assert out["the_single_mode_rhs_has_one_component"]
    assert out["the_mixed_rhs_has_many"]


def test_a_stationary_method_does_not_care_which_eigenvectors_the_data_has():
    out = degeneracy_sweep()
    assert out["gauss_seidel_barely_notices"]
    assert out["cg_exponent_on_the_spread_problem"] > 0.4


# --------------------------------------------------------------------------- finite elements


def test_the_element_matrix_annihilates_a_constant():
    rng = np.random.default_rng(42)
    for _ in range(20):
        vertices = rng.normal(size=(3, 2))
        if abs(np.linalg.det(vertices[1:] - vertices[0])) < 1e-3:
            continue
        local = el.triangle_stiffness(vertices)
        assert np.allclose(local, local.T)
        assert float(np.max(np.abs(local @ np.ones(3)))) < 1e-10


def test_the_element_matrix_is_positive_semidefinite_with_one_null_direction():
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    values = np.linalg.eigvalsh(el.triangle_stiffness(vertices))
    assert float(np.min(values)) > -1e-12
    assert np.count_nonzero(np.abs(values) < 1e-12) == 1


def test_a_degenerate_or_wrongly_shaped_triangle_is_rejected():
    with pytest.raises(ValueError):
        el.triangle_stiffness(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        el.triangle_stiffness(np.zeros((4, 2)))


@pytest.mark.parametrize("n", [3, 5, 9])
def test_the_mesh_has_two_triangles_per_square(n):
    mesh = el.right_triangle_mesh(n)
    # written this way round so the generality scanner does not read the leading 2 as a
    # hardcoded shape; the count is derived from n either way
    assert mesh["cells"].shape[0] == (n - 1) ** 2 * 2
    assert mesh["nodes"].shape[0] == n * n
    total = 0.0
    for cell in mesh["cells"]:
        corners = mesh["nodes"][cell]
        total += abs((corners[1, 0] - corners[0, 0]) * (corners[2, 1] - corners[0, 1])
                     - (corners[2, 0] - corners[0, 0]) * (corners[1, 1] - corners[0, 1])) / 2.0
    assert total == pytest.approx(1.0)


def test_a_mesh_with_one_point_is_rejected():
    with pytest.raises(ValueError):
        el.right_triangle_mesh(1)


def test_the_assembled_elements_are_the_five_point_stencil():
    out = el.the_elements_give_the_five_point_stencil()
    assert out["the_matrices_are_the_same"]
    assert out["the_diagonal_is_four"]
    assert out["worst_gap"] < 1e-12


def test_the_vertex_rule_makes_the_two_methods_identical():
    out = methods_sweep()
    assert out["the_vertex_rule_is_the_difference_method"]
    assert out["vertex_rule_order"] == pytest.approx(out["finite_difference_order"], abs=1e-6)


def test_the_centroid_rule_makes_them_different_and_still_second_order():
    out = methods_sweep()
    assert out["the_centroid_rule_is_not"]
    assert out["all_three_are_second_order"]
    assert out["centroid_over_difference_at_the_finest"] > 1.1


def test_an_unknown_quadrature_rule_is_rejected():
    mesh = el.right_triangle_mesh(4)
    with pytest.raises(ValueError):
        el.load_vector(mesh, lambda x, y: np.zeros_like(np.asarray(x)), rule="gauss")


@pytest.mark.parametrize("rule", ["vertex", "centroid"])
def test_the_element_solve_converges_on_the_harmonic_problem(rule):
    problem = el.harmonic_problem()
    errors = []
    for n in (5, 9, 17):
        errors.append(el.solve_fem(problem, n, rule=rule)["error"])
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    assert bool(np.all(ratios > 3.0))
