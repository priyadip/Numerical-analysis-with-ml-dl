"""Tests for nalib.femode.

Four groups. The basis is checked against the properties that define it: a hat function is 1 at
its own node and 0 at every other, the hats sum to 1 everywhere, and the derivative is the actual
derivative. The assembly is checked against the **exact** element matrices, which for constant
coefficients are known in closed form, and against the second difference operator, which it turns
out to equal exactly. The solutions are checked against exact solutions in three norms, because
the orders differ between them. The last group is what separates the two methods.

The most valuable tests here are the ones that could easily be written to pass for the wrong
reason: nodal exactness needs the reaction control beside it or it looks like a general property
of the method, and the two error norms have to be measured separately or the first order one
hides behind the second order one.
"""
import math

import numpy as np
import pytest

from nalib import femode as fe

COUNTS = [5, 9, 17, 33]
SIZES = [3, 5, 8, 21]


# --------------------------------------------------------------------------- the basis


@pytest.mark.parametrize("n", SIZES)
def test_a_hat_is_one_at_its_own_node_and_zero_at_the_others(n):
    nodes = np.linspace(0.0, 1.0, n)
    for i in range(n):
        values = fe.hat(nodes, i, nodes)
        want = np.zeros(n)
        want[i] = 1.0
        assert np.allclose(values, want, atol=1e-14), i


@pytest.mark.parametrize("n", SIZES)
def test_the_hats_sum_to_one_everywhere(n):
    """A partition of unity, which is what lets the basis represent a constant exactly."""
    nodes = np.linspace(-1.0, 2.0, n)
    x = np.linspace(-1.0, 2.0, 501)
    total = sum(fe.hat(nodes, i, x) for i in range(n))
    assert np.allclose(total, 1.0, atol=1e-13)


@pytest.mark.parametrize("n", SIZES)
def test_the_hats_reproduce_any_linear_function_exactly(n):
    nodes = np.linspace(0.0, 3.0, n)
    x = np.linspace(0.0, 3.0, 401)
    for slope, offset in ((1.0, 0.0), (-2.5, 4.0), (0.0, 7.0)):
        built = sum((slope * nodes[i] + offset) * fe.hat(nodes, i, x) for i in range(n))
        assert np.allclose(built, slope * x + offset, atol=1e-12)


@pytest.mark.parametrize("n", SIZES)
def test_the_hat_derivative_is_the_derivative_of_the_hat(n):
    """Checked away from the nodes, where the hat is differentiable."""
    nodes = np.linspace(0.0, 1.0, n)
    h = 1e-7
    rng = np.random.default_rng(42)
    for i in range(n):
        for _ in range(10):
            x = float(rng.uniform(0.02, 0.98))
            if np.min(np.abs(nodes - x)) < 1e-3:
                continue
            got = float(fe.hat_derivative(nodes, i, np.asarray([x]))[0])
            want = float((fe.hat(nodes, i, np.asarray([x + h]))[0]
                          - fe.hat(nodes, i, np.asarray([x - h]))[0]) / (2.0 * h))
            assert got == pytest.approx(want, abs=1e-5)


def test_a_hat_outside_the_grid_is_refused():
    nodes = np.linspace(0.0, 1.0, 5)
    for bad in (-1, 5, 9):
        with pytest.raises(IndexError):
            fe.hat(nodes, bad, np.asarray([0.5]))
        with pytest.raises(IndexError):
            fe.hat_derivative(nodes, bad, np.asarray([0.5]))


@pytest.mark.parametrize("n", SIZES)
def test_evaluating_reproduces_the_nodal_values(n):
    nodes = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(42)
    coefficients = rng.normal(size=n)
    assert np.allclose(fe.evaluate(nodes, coefficients, nodes), coefficients, atol=1e-14)


def test_a_coefficient_count_that_does_not_match_the_nodes_is_refused():
    with pytest.raises(ValueError):
        fe.evaluate(np.linspace(0.0, 1.0, 5), np.zeros(4), 0.5)


# --------------------------------------------------------------------------- assembly


@pytest.mark.parametrize("width", [0.1, 1.0, 7.5])
def test_the_element_stiffness_matrix_is_the_closed_form(width):
    """For constant k it is (k/h) [[1, -1], [-1, 1]], which is exactly reproducible."""
    for k in (1.0, 3.7):
        local = fe.element_matrices(2.0, 2.0 + width, k=lambda x: k + 0.0 * x)
        want = (k / width) * np.asarray([[1.0, -1.0], [-1.0, 1.0]])
        assert np.allclose(local["stiffness"], want, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("width", [0.1, 1.0, 7.5])
def test_the_element_mass_matrix_is_the_closed_form(width):
    """For constant c it is (c h / 6) [[2, 1], [1, 2]]."""
    for c in (1.0, 0.25):
        local = fe.element_matrices(-1.0, -1.0 + width, c=lambda x: c + 0.0 * x)
        want = (c * width / 6.0) * np.asarray([[2.0, 1.0], [1.0, 2.0]])
        assert np.allclose(local["mass"], want, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("width", [0.1, 1.0, 7.5])
def test_the_element_load_for_a_constant_source_splits_evenly(width):
    local = fe.element_matrices(0.0, width, f=lambda x: 2.0 + 0.0 * x)
    assert np.allclose(local["load"], width, rtol=1e-13)


def test_an_element_of_zero_or_negative_width_is_refused():
    with pytest.raises(ValueError):
        fe.element_matrices(1.0, 1.0)
    with pytest.raises(ValueError):
        fe.element_matrices(1.0, 0.5)


def test_a_grid_with_no_interior_node_is_refused():
    with pytest.raises(ValueError):
        fe.assemble([0.0, 1.0], f=lambda x: np.ones_like(x))


def test_a_grid_that_does_not_increase_is_refused():
    with pytest.raises(ValueError):
        fe.assemble([0.0, 0.6, 0.4, 1.0], f=lambda x: np.ones_like(x))


def test_the_assembled_matrix_is_exactly_the_second_difference():
    out = fe.the_stiffness_matrix_is_the_second_difference()
    assert out["same_matrix"]
    assert out["same_load_for_constant_f"]
    assert float(np.max(out["diagonal_gap"])) == 0.0
    assert float(np.max(out["off_diagonal_gap"])) == 0.0


def test_the_load_vector_is_where_the_two_methods_differ():
    """And it differs by O(h^3) per entry, which is why they have the same order."""
    out = fe.the_load_vector_is_what_differs()
    assert out["is_third_order_per_entry"]
    assert out["fitted_order"] == pytest.approx(3.0, abs=0.15)


@pytest.mark.parametrize("n", COUNTS)
def test_the_assembled_matrix_is_symmetric_at_every_size(n):
    grid = np.linspace(0.0, 1.0, n)
    system = fe.assemble(grid, f=lambda x: np.ones_like(np.asarray(x, dtype=float)))
    A = fe.dense(system)
    assert np.allclose(A, A.T, atol=1e-13)


def test_the_matrix_is_symmetric_positive_definite_and_badly_conditioned():
    out = fe.the_matrix_is_symmetric_positive_definite()
    assert out["symmetric"]
    assert out["positive_definite"]
    assert out["condition_grows_like_h_squared"]
    assert out["condition_order"] == pytest.approx(-2.0, abs=0.15)


@pytest.mark.parametrize("n", COUNTS)
def test_assembly_works_on_an_unequal_grid_too(n):
    rng = np.random.default_rng(42)
    grid = np.linspace(0.0, 1.0, n)
    grid[1:-1] += rng.uniform(-0.4, 0.4, n - 2) / (n - 1)
    grid = np.sort(grid)
    system = fe.assemble(grid, f=lambda x: np.ones_like(np.asarray(x, dtype=float)))
    A = fe.dense(system)
    assert np.allclose(A, A.T, atol=1e-12)
    # each row of the stiffness part sums to zero, because a constant has zero energy
    interior = A[1:-1]
    assert float(np.max(np.abs(np.sum(interior, axis=1)))) < 1e-11


# --------------------------------------------------------------------------- solutions


def test_the_residual_is_orthogonal_to_every_basis_function():
    out = fe.galerkin_orthogonality()
    assert out["orthogonal"]
    assert out["worst"] < 1e-12


@pytest.mark.parametrize("n", COUNTS)
def test_the_solution_meets_both_boundary_conditions_exactly(n):
    problem = fe.sine_problem()
    grid = np.linspace(problem["a"], problem["b"], n)
    out = fe.galerkin(grid, problem["f"], alpha=2.5, beta=-1.0)
    assert float(out["u"][0]) == 2.5
    assert float(out["u"][-1]) == -1.0
    assert out["unknowns"] == n - 2


def test_the_solution_is_exact_at_the_nodes_and_the_reaction_term_breaks_it():
    out = fe.nodal_exactness()
    assert out["exact_on_a_uniform_grid"]
    assert out["exact_on_an_irregular_grid"]
    assert out["reaction_term_destroys_it"]
    assert out["reaction_order"] == pytest.approx(2.0, abs=0.15)


@pytest.mark.parametrize("wavenumber", [1, 2, 4])
def test_nodal_exactness_holds_for_any_smooth_source(wavenumber):
    problem = fe.sine_problem(wavenumber)
    grid = np.linspace(problem["a"], problem["b"], 21)
    out = fe.galerkin(grid, problem["f"], alpha=problem["alpha"], beta=problem["beta"])
    assert float(np.max(np.abs(out["u"] - problem["exact"](grid)))) < 1e-12


def test_the_two_error_norms_have_different_orders():
    out = fe.orders_in_two_norms()
    assert out["l2_is_second_order"]
    assert out["energy_is_first_order"]
    assert out["they_differ_by_one"]


def test_the_energy_error_is_much_larger_than_the_l2_error():
    """Which is the practical consequence: a good temperature is not a good flux."""
    out = fe.orders_in_two_norms()
    assert np.all(out["energy_error"] > 10.0 * out["l2_error"])


def test_the_nodal_error_would_report_the_wrong_thing_entirely():
    """It is at roundoff at every size, so it says nothing about the order of anything."""
    out = fe.orders_in_two_norms()
    assert float(np.max(out["nodal_error"])) < 1e-12
    assert float(np.min(out["l2_error"])) > 1e-6


def test_crude_quadrature_costs_a_constant_and_not_an_order():
    out = fe.quadrature_that_is_too_crude()
    assert out["one_point_keeps_the_order"]
    assert out["two_point_is_enough"]
    # and it really is worse, so the test is not passing because nothing changed
    assert np.all(out["one_point_error"] > 1.2 * out["exact_error"])
    assert np.allclose(out["two_point_error"], out["exact_error"], rtol=0.05)


# --------------------------------------------------------------------------- collocation


@pytest.mark.parametrize("degree", [0, 1, 5, 12])
def test_the_chebyshev_recurrence_matches_the_cosine_form(degree):
    """T_k(cos theta) = cos(k theta), which is the sharpest available check."""
    theta = np.linspace(0.05, math.pi - 0.05, 41)
    t = np.cos(theta)
    T, D1, D2 = fe.chebyshev_values(degree, t)
    for j in range(degree + 1):
        assert np.allclose(T[j], np.cos(j * theta), atol=1e-11), j


@pytest.mark.parametrize("degree", [2, 5, 9])
def test_the_chebyshev_derivatives_match_finite_differences(degree):
    x = np.linspace(-0.8, 0.8, 21)
    h = 1e-5
    T, D1, D2 = fe.chebyshev_values(degree, x)
    plus, _, _ = fe.chebyshev_values(degree, x + h)
    minus, _, _ = fe.chebyshev_values(degree, x - h)
    for j in range(degree + 1):
        assert np.allclose(D1[j], (plus[j] - minus[j]) / (2.0 * h), atol=1e-5), j
        assert np.allclose(D2[j], (plus[j] - 2.0 * T[j] + minus[j]) / h ** 2, atol=1e-3), j


def test_a_negative_chebyshev_degree_is_refused():
    with pytest.raises(ValueError):
        fe.chebyshev_values(-1, 0.0)


def test_a_collocation_degree_below_two_is_refused():
    with pytest.raises(ValueError):
        fe.collocation(lambda x: np.ones_like(x), 0.0, 1.0, 0.0, 0.0, 1)


@pytest.mark.parametrize("degree", [8, 12, 20])
def test_collocation_meets_its_boundary_conditions(degree):
    problem = fe.sine_problem()
    out = fe.collocation(problem["f"], problem["a"], problem["b"], 1.5, -0.5, degree)
    assert float(out["solution"](problem["a"])) == pytest.approx(1.5, abs=1e-10)
    assert float(out["solution"](problem["b"])) == pytest.approx(-0.5, abs=1e-10)


def test_collocation_converges_faster_than_any_power():
    """Every extra pair of unknowns multiplies the accuracy, rather than adding to it."""
    out = fe.collocation_against_galerkin()
    errors = np.asarray(out["collocation_error"], dtype=float)
    ratios = errors[:-1] / np.maximum(errors[1:], 1e-300)
    assert np.all(ratios > 100.0)
    assert out["collocation_reaches"] < 1e-12


def test_collocation_gives_up_symmetry_and_conditioning():
    out = fe.collocation_against_galerkin()
    assert not out["collocation_is_symmetric"]
    assert out["galerkin_is_symmetric"]
    assert float(np.max(out["collocation_condition"])) > float(
        np.max(out["galerkin_condition"]))


def test_collocation_is_the_better_method_on_a_smooth_problem():
    out = fe.collocation_against_galerkin()
    assert out["collocation_wins_on_a_smooth_problem"]
    assert out["galerkin_reaches"] > 1e-4


def test_collocation_returns_the_same_wrong_answer_at_every_degree_on_a_kink():
    """Not slow convergence. No convergence, because the strong form does not exist."""
    out = fe.collocation_cannot_do_a_kink()
    errors = np.asarray(out["collocation_error"], dtype=float)
    assert out["collocation_stalls"]
    assert float(np.max(errors) - np.min(errors)) < 1e-10
    assert out["galerkin_wins"]
    assert out["how_much_better"] > 1e10


def test_the_straight_line_is_exactly_what_collocation_returns():
    """The measured 0.4901 is the gap between u = x and the true kinked solution."""
    problem = fe.jumping_coefficient_problem(1.0, 100.0, 0.5)
    out = fe.collocation_cannot_do_a_kink(degrees=[8])
    probe = np.linspace(0.0, 1.0, 2001)
    straight_line_gap = float(np.max(np.abs(probe - problem["exact"](probe))))
    assert float(out["collocation_error"][0]) == pytest.approx(straight_line_gap, rel=1e-6)


# --------------------------------------------------------------------------- interfaces


@pytest.mark.parametrize("ratio", [10.0, 100.0, 1000.0])
def test_the_jumping_coefficient_solution_has_a_constant_flux(ratio):
    """Which is the physics: whatever the conductivity does, the flux k u' is continuous."""
    problem = fe.jumping_coefficient_problem(1.0, ratio, 0.5)
    x = np.concatenate([np.linspace(0.01, 0.49, 20), np.linspace(0.51, 0.99, 20)])
    flux = problem["k"](x) * problem["exact_derivative"](x)
    assert np.allclose(flux, problem["flux"], rtol=1e-12)
    assert float(problem["exact"](0.0)) == pytest.approx(0.0, abs=1e-14)
    assert float(problem["exact"](1.0)) == pytest.approx(1.0, abs=1e-14)


def test_a_node_on_the_interface_makes_the_answer_exact():
    out = fe.a_jumping_coefficient()
    assert out["aligned_is_exact"]
    assert bool(np.all(out["interface_on_a_node"]))
    assert float(np.max(out["aligned_error"])) < 1e-12


def test_missing_the_interface_costs_an_order_and_a_lot_of_accuracy():
    out = fe.a_jumping_coefficient()
    assert out["offset_is_not"]
    assert out["offset_order"] == pytest.approx(1.0, abs=0.2)
    assert out["ratio"] > 1e8


@pytest.mark.parametrize("ratio", [2.0, 50.0, 500.0])
def test_the_aligned_answer_is_exact_whatever_the_jump(ratio):
    problem = fe.jumping_coefficient_problem(1.0, ratio, 0.5)
    grid = np.linspace(0.0, 1.0, 9)
    out = fe.galerkin(grid, problem["f"], k=problem["k"], alpha=problem["alpha"],
                      beta=problem["beta"], order=8)
    assert float(np.max(np.abs(out["u"] - problem["exact"](grid)))) < 1e-12
