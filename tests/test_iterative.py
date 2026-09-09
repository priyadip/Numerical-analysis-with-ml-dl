"""Tests for `nalib.iterative`.

Everything about a stationary method is decided by rho(G), so the tests check that the code and
the theory agree on rho, and then that convergence follows. Sizes run from a single unknown
upward and the matrices are drawn from several families, because the second difference matrix
is unusually well behaved and a routine tuned to it can fail everywhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import iterative as it
from nalib.banded import second_difference
from nalib.cholesky import random_spd


SIZES = [1, 2, 3, 5, 9, 20, 47]
SCALES = [1e-9, 1e-3, 1.0, 1e4, 1e9]
TOL = 1e-10


def dominant(n, rng, strength=3.0):
    """A strictly diagonally dominant matrix at any size, so every method here converges."""
    A = rng.standard_normal((n, n))
    A += np.diag(strength * n * np.ones(n))
    return A


# ---------------------------------------------------------------- the splitting


@pytest.mark.parametrize("n", SIZES)
def test_split_reassembles_the_matrix(n, rng):
    A = dominant(n, rng)
    D, L, U = it.split(A)
    np.testing.assert_allclose(D + L + U, A, atol=0.0)


@pytest.mark.parametrize("n", SIZES)
def test_split_pieces_have_the_right_shape(n, rng):
    D, L, U = it.split(dominant(n, rng))
    assert np.allclose(D, np.diag(np.diag(D)))
    assert np.allclose(np.triu(L), 0.0)
    assert np.allclose(np.tril(U), 0.0)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("method", ["jacobi", "gauss-seidel", "sor"])
def test_iteration_matrix_has_the_defining_property(n, rng, method):
    """G = M^-1 N with A = M - N, so (I - G) = M^-1 A and the fixed point of the iteration is
    the solution. This checks that identity rather than the formula that produced it."""
    A = dominant(n, rng)
    omega = 1.3 if method == "sor" else 1.0
    G = it.iteration_matrix(A, method, omega)
    D, L, U = it.split(A)
    M = {"jacobi": D, "gauss-seidel": D + L, "sor": D / omega + L}[method]
    np.testing.assert_allclose(np.eye(n) - G, np.linalg.solve(M, A), atol=1e-10)


@pytest.mark.parametrize("n", SIZES)
def test_jacobi_iteration_matrix_has_a_zero_diagonal(n, rng):
    G = it.iteration_matrix(dominant(n, rng), "jacobi")
    np.testing.assert_allclose(np.diag(G), np.zeros(n), atol=1e-14)


def test_iteration_matrix_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        it.iteration_matrix(np.eye(3), "chebyshev")


@pytest.mark.parametrize("omega", [-0.5, 0.0, 2.0, 3.0])
def test_sor_rejects_omega_outside_kahans_interval(omega):
    with pytest.raises(ValueError):
        it.iteration_matrix(second_difference(5), "sor", omega)


def test_iteration_matrix_rejects_a_zero_diagonal():
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(np.linalg.LinAlgError):
        it.iteration_matrix(A, "jacobi")


# ---------------------------------------------------------------- the model problem


@pytest.mark.parametrize("n", [3, 7, 15, 31, 63])
def test_jacobi_rho_is_cos_pi_h(n):
    h = 1.0 / (n + 1)
    got = it.spectral_radius(it.iteration_matrix(second_difference(n), "jacobi"))
    assert abs(got - np.cos(np.pi * h)) < 1e-11, f"{got:.12f} against {np.cos(np.pi*h):.12f}"


@pytest.mark.parametrize("n", [3, 7, 15, 31, 63])
def test_gauss_seidel_rho_is_jacobi_squared(n):
    """Theorem 23.4 for a consistently ordered matrix. Exact to eight decimal places, not
    approximate, and it is the whole content of 'Gauss-Seidel is twice as fast'."""
    A = second_difference(n)
    rj = it.spectral_radius(it.iteration_matrix(A, "jacobi"))
    rg = it.spectral_radius(it.iteration_matrix(A, "gauss-seidel"))
    assert abs(rg - rj ** 2) < 1e-9, f"{rg:.12f} against {rj ** 2:.12f}"


@pytest.mark.parametrize("n", [3, 7, 15, 31, 63])
def test_optimal_sor_rho_equals_omega_minus_one(n):
    """Kahan's bound is attained exactly at the optimum, so the inequality is also an equality
    there. Both facts are checked."""
    A = second_difference(n)
    w = it.optimal_omega(A)
    rho = it.spectral_radius(it.iteration_matrix(A, "sor", w))
    assert 1.0 < w < 2.0
    assert rho >= abs(w - 1.0) - 1e-12                 # Kahan's theorem
    assert abs(rho - (w - 1.0)) < 1e-7, f"{rho:.10f} against {w - 1.0:.10f}"


@pytest.mark.parametrize("n", [3, 7, 15, 31])
def test_optimal_omega_matches_the_closed_form(n):
    h = 1.0 / (n + 1)
    want = 2.0 / (1.0 + np.sin(np.pi * h))
    assert abs(it.optimal_omega(second_difference(n)) - want) < 1e-9


@pytest.mark.parametrize("n", [7, 15, 31])
def test_optimal_omega_really_is_optimal(n):
    """Sweeping omega must never find anything better than what optimal_omega returns."""
    A = second_difference(n)
    best = it.optimal_omega(A)
    rho_best = it.spectral_radius(it.iteration_matrix(A, "sor", best))
    for w in np.linspace(1.0, 1.99, 60):
        assert it.spectral_radius(it.iteration_matrix(A, "sor", w)) >= rho_best - 1e-9


@pytest.mark.parametrize("n", [3, 8, 20])
def test_kahans_bound_holds_for_every_omega(n):
    A = second_difference(n)
    for w in np.linspace(0.05, 1.95, 40):
        rho = it.spectral_radius(it.iteration_matrix(A, "sor", w))
        assert rho >= abs(w - 1.0) - 1e-11, f"rho {rho:.6f} below |w-1| at w={w:.3f}"


# ---------------------------------------------------------------- convergence


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("method", ["jacobi", "damped_jacobi", "gauss_seidel", "sor"])
def test_every_method_solves_a_dominant_system(n, rng, method):
    A = dominant(n, rng)
    x_true = rng.standard_normal(n)
    b = A @ x_true
    run = {"jacobi": lambda: it.jacobi(A, b, tol=TOL, max_iter=20000, keep_history=False),
           "damped_jacobi": lambda: it.damped_jacobi(A, b, 2.0 / 3.0, tol=TOL,
                                                     max_iter=20000, keep_history=False),
           "gauss_seidel": lambda: it.gauss_seidel(A, b, tol=TOL, max_iter=20000,
                                                   keep_history=False),
           "sor": lambda: it.sor(A, b, 1.1, tol=TOL, max_iter=20000, keep_history=False)}[method]
    r = run()
    assert r.converged, r.message
    np.testing.assert_allclose(r.x, x_true, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("scale", SCALES)
def test_convergence_is_scale_invariant(scale, rng):
    n = 12
    A = scale * dominant(n, rng)
    x_true = rng.standard_normal(n)
    r = it.gauss_seidel(A, A @ x_true, tol=TOL, max_iter=20000, keep_history=False)
    assert r.converged
    np.testing.assert_allclose(r.x, x_true, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n", [7, 15, 31])
def test_gauss_seidel_needs_about_half_of_jacobis_sweeps(n):
    A = second_difference(n)
    b = A @ np.ones(n)
    j = it.jacobi(A, b, tol=1e-8, max_iter=200000, keep_history=False)
    g = it.gauss_seidel(A, b, tol=1e-8, max_iter=200000, keep_history=False)
    assert j.converged and g.converged
    ratio = j.n_iter / g.n_iter
    assert 1.7 < ratio < 2.3, f"ratio {ratio:.3f} at n={n}"


@pytest.mark.parametrize("n", [15, 31, 63])
def test_optimal_sor_beats_gauss_seidel_by_more_as_n_grows(n):
    A = second_difference(n)
    b = A @ np.ones(n)
    g = it.gauss_seidel(A, b, tol=1e-8, max_iter=400000, keep_history=False)
    s = it.sor(A, b, it.optimal_omega(A), tol=1e-8, max_iter=400000, keep_history=False)
    assert g.converged and s.converged
    assert s.n_iter < g.n_iter


def test_optimal_sor_advantage_grows_with_n():
    ratios = []
    for n in (15, 31, 63):
        A = second_difference(n)
        b = A @ np.ones(n)
        g = it.gauss_seidel(A, b, tol=1e-8, max_iter=400000, keep_history=False)
        s = it.sor(A, b, it.optimal_omega(A), tol=1e-8, max_iter=400000, keep_history=False)
        ratios.append(g.n_iter / s.n_iter)
    assert ratios[0] < ratios[1] < ratios[2], f"ratios {ratios}"


@pytest.mark.parametrize("n", SIZES)
def test_converges_agrees_with_the_actual_run(n, rng):
    A = dominant(n, rng)
    b = rng.standard_normal(n)
    for method in ("jacobi", "gauss-seidel"):
        predicted = it.converges(A, method)
        run = (it.jacobi if method == "jacobi" else it.gauss_seidel)(
            A, b, tol=TOL, max_iter=20000, keep_history=False)
        assert predicted == run.converged, f"{method} at n={n}"


def test_converges_reports_false_for_a_divergent_splitting():
    """Collatz's matrix: Jacobi converges in three sweeps and Gauss-Seidel diverges. The two
    methods do not dominate each other, and this pair of assertions is the proof."""
    A = np.array([[1.0, 2.0, -2.0], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]])
    assert it.converges(A, "jacobi")
    assert not it.converges(A, "gauss-seidel")


def test_the_reverse_counterexample_also_holds():
    A = np.array([[2.0, -1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, -1.0, 2.0]])
    assert not it.converges(A, "jacobi")
    assert it.converges(A, "gauss-seidel")


@pytest.mark.parametrize("n", [4, 10])
def test_a_divergent_run_is_reported_not_hidden(n, rng):
    A = np.array([[2.0, -1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, -1.0, 2.0]])
    r = it.jacobi(A, np.ones(3), tol=TOL, max_iter=400, keep_history=False)
    assert not r.converged
    assert r.message


# ---------------------------------------------------------------- damped Jacobi


@pytest.mark.parametrize("omega", [0.25, 0.5, 2.0 / 3.0, 1.0])
def test_damped_jacobi_eigenvalues_follow_the_damping_formula(omega):
    """G(omega) = (1-omega) I + omega G_J, so every eigenvalue lam becomes 1 - omega(1-lam)."""
    n = 21
    A = second_difference(n)
    plain = np.linalg.eigvals(it.iteration_matrix(A, "jacobi"))
    damped = np.linalg.eigvals(it.iteration_matrix(A, "jacobi", omega))
    want = np.sort(1.0 - omega * (1.0 - plain.real))
    np.testing.assert_allclose(np.sort(damped.real), want, atol=1e-10)


@pytest.mark.parametrize("omega", [0.0, -0.5, 1.5, 2.5])
def test_damped_jacobi_rejects_omega_outside_zero_to_one(omega):
    with pytest.raises(ValueError):
        it.damped_jacobi(second_difference(5), np.ones(5), omega)


@pytest.mark.parametrize("n", [15, 31, 63])
def test_damped_jacobi_is_a_better_smoother_and_a_worse_solver(n):
    """The whole reason it exists. Plain Jacobi barely touches the top of the spectrum because
    its eigenvalues approach -1 there; damping at 2/3 brings the worst upper-half shrinkage to
    1/3. The price is a slower solve, since the slowest mode is slowed too."""
    A = second_difference(n)
    h = 1.0 / (n + 1)
    modes = np.arange(1, n + 1)
    V = np.sqrt(2 * h) * np.sin(np.outer(modes, modes) * np.pi * h)
    worst = {}
    for w in (1.0, 2.0 / 3.0):
        G = it.iteration_matrix(A, "jacobi", w)
        worst[w] = np.linalg.norm(G @ V, axis=0)[n // 2:].max()
    assert worst[2.0 / 3.0] < 0.4, f"damped upper-half shrinkage {worst[2.0/3.0]:.4f}"
    assert worst[1.0] > 0.9, f"plain upper-half shrinkage {worst[1.0]:.4f}"
    # and as a SOLVER the damped version is slower, because rho is closer to 1
    assert (it.spectral_radius(it.iteration_matrix(A, "jacobi", 2.0 / 3.0))
            > it.spectral_radius(it.iteration_matrix(A, "jacobi")))


@pytest.mark.parametrize("n", SIZES)
def test_damped_jacobi_at_omega_one_is_plain_jacobi(n, rng):
    A = dominant(n, rng)
    b = rng.standard_normal(n)
    a = it.jacobi(A, b, tol=TOL, max_iter=20000, keep_history=False)
    d = it.damped_jacobi(A, b, 1.0, tol=TOL, max_iter=20000, keep_history=False)
    assert a.n_iter == d.n_iter
    np.testing.assert_allclose(a.x, d.x, atol=1e-14)


# ---------------------------------------------------------------- rate and bookkeeping


@pytest.mark.parametrize("rho", [0.5, 0.9, 0.99, 0.999, 0.9999])
def test_iterations_needed_inverts_the_rate(rho):
    k = it.iterations_needed(rho, 1e-10)
    assert rho ** k <= 1e-10 * (1.0 + 1e-9)
    assert rho ** max(k - 1.0, 0.0) >= 1e-10 * (1.0 - 1e-9)


@pytest.mark.parametrize("rho", [1.0, 1.5, 2.0])
def test_iterations_needed_is_infinite_for_a_non_convergent_rate(rho):
    assert it.iterations_needed(rho, 1e-10) == float("inf")


def test_iterations_needed_is_zero_for_a_nilpotent_iteration():
    assert it.iterations_needed(0.0, 1e-10) == 0.0


@pytest.mark.parametrize("bad", [-0.2, -1.0])
def test_iterations_needed_rejects_a_negative_rate(bad):
    """rho is the modulus of an eigenvalue, so a negative value means the caller passed the
    wrong quantity. Returning a plausible number would hide the mistake."""
    with pytest.raises(ValueError):
        it.iterations_needed(bad, 1e-10)


@pytest.mark.parametrize("tol", [0.0, 1.0, 2.0, -1e-3])
def test_iterations_needed_rejects_a_tolerance_outside_zero_to_one(tol):
    with pytest.raises(ValueError):
        it.iterations_needed(0.5, tol)


@pytest.mark.parametrize("n", [7, 15, 31])
def test_observed_rate_matches_the_spectral_radius(n):
    A = second_difference(n)
    r = it.jacobi(A, A @ np.ones(n), tol=1e-12, max_iter=200000)
    rho = it.spectral_radius(it.iteration_matrix(A, "jacobi"))
    assert abs(r.observed_rate() - rho) < 0.02, f"{r.observed_rate():.5f} against {rho:.5f}"


@pytest.mark.parametrize("n", [4, 12])
def test_history_is_kept_only_when_asked(n, rng):
    A = dominant(n, rng)
    b = rng.standard_normal(n)
    with_hist = it.jacobi(A, b, tol=TOL, max_iter=5000)
    without = it.jacobi(A, b, tol=TOL, max_iter=5000, keep_history=False)
    assert with_hist.iterates.shape == (with_hist.n_iter + 1, n)
    assert without.iterates.size == 0
    np.testing.assert_allclose(with_hist.x, without.x, atol=1e-14)


@pytest.mark.parametrize("n", [4, 12])
def test_errors_helper_matches_a_manual_computation(n, rng):
    A = dominant(n, rng)
    x_true = rng.standard_normal(n)
    r = it.jacobi(A, A @ x_true, tol=TOL, max_iter=5000)
    manual = np.linalg.norm(r.iterates - x_true, axis=1)
    np.testing.assert_allclose(r.errors(x_true), manual, atol=1e-14)


@pytest.mark.parametrize("n", [3, 9])
def test_a_length_mismatch_is_rejected(n, rng):
    A = dominant(n, rng)
    for f in (it.jacobi, it.gauss_seidel):
        with pytest.raises(ValueError):
            f(A, np.ones(n + 1))


def test_a_non_square_matrix_is_rejected():
    with pytest.raises(ValueError):
        it.jacobi(np.ones((2, 3)), np.ones(2))


@pytest.mark.parametrize("n", SIZES)
def test_residual_norms_start_from_the_initial_guess(n, rng):
    A = dominant(n, rng)
    b = rng.standard_normal(n)
    r = it.jacobi(A, b, tol=TOL, max_iter=5000, keep_history=False)
    assert abs(r.residual_norms[0] - np.linalg.norm(b)) < 1e-12 * max(1.0, np.linalg.norm(b))
    assert len(r.residual_norms) == r.n_iter + 1


@pytest.mark.parametrize("n", SIZES)
def test_starting_from_the_answer_stops_immediately(n, rng):
    A = dominant(n, rng)
    x_true = rng.standard_normal(n)
    r = it.jacobi(A, A @ x_true, x0=x_true, tol=1e-8, max_iter=5000, keep_history=False)
    assert r.converged and r.n_iter <= 1
