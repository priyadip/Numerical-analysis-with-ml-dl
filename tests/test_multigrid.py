"""Tests for `nalib.multigrid`.

The claim multigrid makes is stronger than any other in the course, so it is tested more
strictly: the convergence factor must be essentially the same at every size, not merely
bounded, and the two-grid spectral radius must be identical to many digits rather than close.
Sizes run from a single unknown to a million, and the smoother sweep exists to record that the
wrong smoother makes the whole thing fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import multigrid as mg


DYADIC = [7, 15, 31, 63, 127]
SCALES = [1e-6, 1.0, 1e6]


# ---------------------------------------------------------------- grids


@pytest.mark.parametrize("n", [1, 3, 7, 15, 31, 63, 127, 255, 1023])
def test_level_count_matches_the_coarsening(n):
    levels = mg.n_levels(n)
    m = n
    counted = 1
    while m > 3 and (m - 1) // 2 >= 1:
        m = (m - 1) // 2
        counted += 1
    assert levels == counted


def test_level_count_rejects_an_empty_grid():
    with pytest.raises(ValueError):
        mg.n_levels(0)


@pytest.mark.parametrize("n", [1, 2, 5, 15, 64])
def test_poisson_has_the_right_shape_and_scaling(n):
    A = mg.poisson_1d(n)
    assert A.shape == (n, n)
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(A) > 0.0)
    h = 1.0 / (n + 1)
    np.testing.assert_allclose(np.diag(A), np.full(n, 2.0 / h ** 2), rtol=1e-12)


@pytest.mark.parametrize("n", [3, 9, 20])
@pytest.mark.parametrize("h", [0.5, 1.0, 2.0])
def test_poisson_scales_with_h_squared(n, h):
    np.testing.assert_allclose(mg.poisson_1d(n, h) * h * h,
                               mg.poisson_1d(n, 1.0), rtol=1e-12)


def test_poisson_rejects_an_empty_grid():
    with pytest.raises(ValueError):
        mg.poisson_1d(0)


# ---------------------------------------------------------------- transfers


@pytest.mark.parametrize("n", [3, 7, 15, 31, 63])
def test_the_transfers_have_the_right_lengths(n):
    r = np.arange(1.0, n + 1)
    nc = (n - 1) // 2
    assert mg.restrict_full_weighting(r).size == nc
    assert mg.restrict_injection(r).size == nc
    assert mg.prolong_linear(np.ones(nc)).size == 2 * nc + 1
    assert mg.prolong_linear(np.ones(nc), n).size == n


@pytest.mark.parametrize("n", [3, 7, 15, 31, 63])
def test_full_weighting_is_prolongation_transposed_over_two(n):
    """The variational condition. Without it the coarse operator is not symmetric and the
    coarse grid correction is not an A-orthogonal projection."""
    nc = (n - 1) // 2
    P = np.column_stack([mg.prolong_linear(np.eye(nc)[:, i], n) for i in range(nc)])
    R = np.vstack([mg.restrict_full_weighting(np.eye(n)[:, j]) for j in range(n)]).T
    np.testing.assert_allclose(R, 0.5 * P.T, atol=1e-14)


@pytest.mark.parametrize("n", [7, 15, 31, 63])
def test_full_weighting_damps_the_roughest_mode_and_injection_does_not(n):
    """Injection samples a high frequency and keeps its size, putting a frequency on the coarse
    grid that cannot exist there. That is aliasing."""
    grid = np.arange(1, n + 1) / (n + 1)
    rough = np.sin(np.pi * n * grid)
    fw = np.linalg.norm(mg.restrict_full_weighting(rough))
    inj = np.linalg.norm(mg.restrict_injection(rough))
    assert fw < 0.3 * inj, f"full weighting {fw:.4f}, injection {inj:.4f}"


@pytest.mark.parametrize("n", [7, 15, 31, 63])
def test_full_weighting_preserves_the_smoothest_mode(n):
    grid = np.arange(1, n + 1) / (n + 1)
    smooth_mode = np.sin(np.pi * grid)
    coarse = mg.restrict_full_weighting(smooth_mode)
    nc = coarse.size
    expected = np.sin(np.pi * np.arange(1, nc + 1) / (nc + 1))
    corr = abs(float(coarse @ expected)) / (np.linalg.norm(coarse) * np.linalg.norm(expected))
    assert corr > 0.999, f"correlation with the coarse mode 1 was only {corr:.4f}"


@pytest.mark.parametrize("nc", [1, 3, 7, 15])
def test_prolongation_reproduces_a_linear_function(nc):
    """Linear interpolation is exact on linear data, away from the boundary where the transfer
    assumes a zero value."""
    coarse = np.arange(1, nc + 1, dtype=float)
    fine = mg.prolong_linear(coarse)
    for i in range(nc):
        assert abs(fine[2 * i + 1] - coarse[i]) < 1e-14
    for i in range(nc - 1):
        mid = fine[2 * i + 2]
        assert abs(mid - 0.5 * (coarse[i] + coarse[i + 1])) < 1e-14


def test_restriction_of_a_grid_too_small_returns_nothing():
    assert mg.restrict_full_weighting(np.ones(1)).size == 0
    assert mg.restrict_injection(np.ones(1)).size == 0


# ---------------------------------------------------------------- the coarse operator


@pytest.mark.parametrize("n", [7, 15, 31, 63, 127])
def test_galerkin_equals_the_rediscretisation_for_the_model_problem(n):
    """Exactly, to the last bit. Worth checking rather than assuming, because it is false for a
    variable coefficient problem and the lesson relies on knowing which case it is in."""
    nc = (n - 1) // 2
    np.testing.assert_allclose(mg.galerkin_coarse(mg.poisson_1d(n), nc),
                               mg.poisson_1d(nc), atol=0.0)


@pytest.mark.parametrize("n", [7, 15, 31])
def test_the_coarse_operator_is_symmetric_positive_definite(n, rng):
    """It has to be, or the recursion is solving a different kind of problem at each level."""
    B = rng.standard_normal((n, n))
    A = B @ B.T + n * np.eye(n)
    Ac = mg.galerkin_coarse(A, (n - 1) // 2)
    np.testing.assert_allclose(Ac, Ac.T, atol=1e-10 * np.abs(Ac).max())
    assert np.all(np.linalg.eigvalsh((Ac + Ac.T) / 2) > 0.0)


@pytest.mark.parametrize("n", [7, 15, 31])
def test_the_coarse_operator_differs_for_a_variable_coefficient(n, rng):
    """The case where RAP is NOT the rediscretisation, which is exactly why it is built from A
    rather than written down afresh."""
    coeff = np.ones(n + 1)
    coeff[(n + 1) // 2:] = 100.0
    A = ((np.diag(coeff[:-1] + coeff[1:]) - np.diag(coeff[1:-1], 1)
          - np.diag(coeff[1:-1], -1)) * (n + 1) ** 2)
    nc = (n - 1) // 2
    if nc >= 1:
        assert np.abs(mg.galerkin_coarse(A, nc) - mg.poisson_1d(nc)).max() > 1.0


# ---------------------------------------------------------------- smoothing


@pytest.mark.parametrize("n", [1, 5, 15, 63])
@pytest.mark.parametrize("kind", ["jacobi", "damped_jacobi", "gauss-seidel", "red-black"])
def test_every_smoother_reduces_the_error_on_the_model_problem(n, kind, rng):
    A = mg.poisson_1d(n)
    x_true = rng.standard_normal(n)
    b = A @ x_true
    before = np.linalg.norm(x_true)
    after = np.linalg.norm(mg.smooth(A, b, np.zeros(n), 5, kind) - x_true)
    assert after < before


@pytest.mark.parametrize("n", [15, 63])
def test_zero_sweeps_changes_nothing(n, rng):
    A = mg.poisson_1d(n)
    x = rng.standard_normal(n)
    np.testing.assert_allclose(mg.smooth(A, rng.standard_normal(n), x, 0), x, atol=0.0)


@pytest.mark.parametrize("omega", [0.0, -0.5, 1.5, 2.0])
def test_damped_jacobi_rejects_omega_outside_zero_to_one(omega):
    with pytest.raises(ValueError):
        mg.smooth(mg.poisson_1d(7), np.ones(7), np.zeros(7), 1, "damped_jacobi", omega)


def test_an_unknown_smoother_is_rejected():
    with pytest.raises(ValueError):
        mg.smooth(mg.poisson_1d(7), np.ones(7), np.zeros(7), 1, "chebyshev")


@pytest.mark.parametrize("n", [31, 63, 127])
def test_damped_jacobi_at_two_thirds_smooths_and_plain_jacobi_does_not(n):
    """The measurement lesson 23 made, and the reason multigrid uses damping. The upper half of
    the spectrum is exactly what a coarse grid with half the points cannot represent."""
    A = mg.poisson_1d(n)
    plain = mg.smoothing_factor(A, "jacobi")
    damped = mg.smoothing_factor(A, "damped_jacobi", 2.0 / 3.0)
    assert plain["worst_upper_half"] > 0.99, f"{plain['worst_upper_half']:.4f}"
    assert abs(damped["worst_upper_half"] - 1.0 / 3.0) < 0.01
    # and both are hopeless as solvers, which is the point
    assert plain["spectral_radius"] > 0.99 and damped["spectral_radius"] > 0.99


@pytest.mark.parametrize("n", [31, 63])
def test_two_thirds_is_the_best_damping_for_smoothing(n):
    A = mg.poisson_1d(n)
    best = mg.smoothing_factor(A, "damped_jacobi", 2.0 / 3.0)["worst_upper_half"]
    for w in np.linspace(0.05, 1.0, 40):
        assert mg.smoothing_factor(A, "damped_jacobi", w)["worst_upper_half"] >= best - 1e-9


@pytest.mark.parametrize("n", [31, 63])
def test_the_spectral_radius_improves_as_the_smoothing_gets_worse(n):
    """The two criteria point in opposite directions, which is the whole reason a good solver
    makes a bad smoother."""
    A = mg.poisson_1d(n)
    omegas = [0.4, 0.6, 0.8, 1.0]
    rhos = [mg.smoothing_factor(A, "damped_jacobi", w)["spectral_radius"] for w in omegas]
    assert all(a > b for a, b in zip(rhos, rhos[1:])), f"rho did not fall: {rhos}"
    tops = [mg.smoothing_factor(A, "damped_jacobi", w)["worst_upper_half"] for w in omegas]
    assert tops[-1] > tops[1], "the smoothing factor was expected to get worse"


@pytest.mark.parametrize("n", [31, 63])
def test_optimal_sor_makes_high_frequencies_grow(n):
    from nalib.iterative import optimal_omega

    A = mg.poisson_1d(n)
    w = optimal_omega(A)
    s = mg.smoothing_factor(A, "sor", w)
    assert s["spectral_radius"] < 0.95, "SOR should be the best solver of the group"
    assert s["worst_upper_half"] > 1.0, "and the worst smoother, amplifying the top"


@pytest.mark.parametrize("n", [31, 63])
def test_smoothing_removes_the_high_half_and_leaves_the_low(n):
    A = mg.poisson_1d(n)
    h = 1.0 / (n + 1)
    modes = np.arange(1, n + 1)
    V = np.sqrt(2 * h) * np.sin(np.outer(modes, modes) * np.pi * h)
    e0 = V @ np.ones(n)
    e0 = e0 / np.linalg.norm(e0)
    e8 = mg.smooth(A, np.zeros(n), e0, 8)
    c0, c8 = V.T @ e0, V.T @ e8
    hi0, hi8 = np.linalg.norm(c0[n // 2:]), np.linalg.norm(c8[n // 2:])
    lo0, lo8 = np.linalg.norm(c0[:n // 2]), np.linalg.norm(c8[:n // 2])
    assert hi8 < 0.01 * hi0, f"the high half only fell from {hi0:.3f} to {hi8:.3f}"
    assert lo8 > 0.4 * lo0, f"the low half fell from {lo0:.3f} to {lo8:.3f}, too much"


# ---------------------------------------------------------------- the cycles


@pytest.mark.parametrize("n", [1, 3, 7, 15, 31, 63, 127, 255])
def test_the_v_cycle_solves_the_model_problem(n, rng):
    A = mg.poisson_1d(n)
    x_true = rng.standard_normal(n)
    r = mg.solve(A, A @ x_true, tol=1e-10, max_cycles=200)
    assert r.converged, r.message
    np.testing.assert_allclose(r.x, x_true, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("cycle", ["v", "w", "two-grid"])
@pytest.mark.parametrize("n", [7, 31, 127])
def test_every_cycle_type_solves_it(cycle, n, rng):
    A = mg.poisson_1d(n)
    x_true = rng.standard_normal(n)
    r = mg.solve(A, A @ x_true, cycle=cycle, tol=1e-10, max_cycles=200)
    assert r.converged, f"{cycle}: {r.message}"
    np.testing.assert_allclose(r.x, x_true, rtol=1e-6, atol=1e-8)


def test_an_unknown_cycle_is_rejected():
    with pytest.raises(ValueError):
        mg.solve(mg.poisson_1d(7), np.ones(7), cycle="u")


def test_the_solver_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        mg.solve(mg.poisson_1d(7), np.ones(8))


def test_convergence_is_mesh_independent():
    """The claim the whole method exists for, asserted as a spread rather than a bound: the
    factor must be essentially the SAME at every size, not merely below something."""
    factors, cycles = [], []
    for k in range(3, 10):
        n = 2 ** k - 1
        A = mg.poisson_1d(n)
        gen = np.random.default_rng(7)
        r = mg.solve(A, A @ gen.standard_normal(n), tol=1e-10, max_cycles=200)
        assert r.converged
        factors.append(r.convergence_factor())
        cycles.append(r.n_cycles)
    assert max(factors) - min(factors) < 0.03, f"factors ranged over {factors}"
    assert max(cycles) - min(cycles) <= 2, f"cycle counts {cycles}"
    assert max(factors) < 0.15, f"worst factor {max(factors):.4f}"


@pytest.mark.parametrize("n", DYADIC)
def test_the_two_grid_spectral_radius_is_exactly_five_over_eightyone(n):
    """Not approximately mesh independent: identical to twelve digits, while the smoother's own
    spectral radius climbs towards 1."""
    rho = float(np.max(np.abs(np.linalg.eigvals(mg.two_grid_operator(mg.poisson_1d(n))))))
    assert abs(rho - 5.0 / 81.0) < 1e-10, f"rho = {rho:.14f} at n = {n}"


@pytest.mark.parametrize("n", DYADIC)
def test_the_smoother_alone_gets_worse_while_the_cycle_does_not(n):
    sm = mg.smoothing_factor(mg.poisson_1d(n))["spectral_radius"]
    rho = float(np.max(np.abs(np.linalg.eigvals(mg.two_grid_operator(mg.poisson_1d(n))))))
    assert sm > 0.9, f"the smoother is doing suspiciously well at n = {n}"
    assert rho < 0.1


def test_the_smoother_spectral_radius_approaches_one_with_n():
    rhos = [mg.smoothing_factor(mg.poisson_1d(n))["spectral_radius"] for n in DYADIC]
    assert all(a < b for a, b in zip(rhos, rhos[1:])), f"{rhos}"
    assert rhos[-1] > 0.999


@pytest.mark.parametrize("scale", SCALES)
def test_the_v_cycle_is_scale_invariant(scale, rng):
    n = 63
    A = mg.poisson_1d(n)
    x_true = rng.standard_normal(n)
    plain = mg.solve(A, A @ x_true, tol=1e-10, max_cycles=200)
    scaled = mg.solve(scale * A, scale * (A @ x_true), tol=1e-10, max_cycles=200)
    assert plain.converged and scaled.converged
    assert plain.n_cycles == scaled.n_cycles
    np.testing.assert_allclose(scaled.x, plain.x, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------- the smoother decides


def test_plain_jacobi_makes_multigrid_fail_completely():
    """Not slower: it does not converge at any size. The single constant omega is the whole
    difference, and lesson 23 measured why."""
    for n in (31, 63, 127):
        A = mg.poisson_1d(n)
        gen = np.random.default_rng(11)
        b = A @ gen.standard_normal(n)
        assert not mg.solve(A, b, tol=1e-10, max_cycles=400, kind="jacobi").converged
        assert mg.solve(A, b, tol=1e-10, max_cycles=400, kind="damped_jacobi").converged


def test_the_cycle_count_tracks_the_smoothing_factor_not_the_spectral_radius():
    n = 127
    A = mg.poisson_1d(n)
    gen = np.random.default_rng(13)
    b = A @ gen.standard_normal(n)
    rows = []
    for w in (0.3, 0.5, 2.0 / 3.0, 0.9):
        sf = mg.smoothing_factor(A, "damped_jacobi", w)
        r = mg.solve(A, b, tol=1e-10, max_cycles=400, omega=w)
        rows.append((sf["worst_upper_half"], sf["spectral_radius"],
                     r.n_cycles if r.converged else 10 ** 6))
    # the best cycle count belongs to the best smoothing factor
    best_by_cycles = min(rows, key=lambda t: t[2])
    best_by_smoothing = min(rows, key=lambda t: t[0])
    assert best_by_cycles is best_by_smoothing
    # and NOT to the best spectral radius
    best_by_rho = min(rows, key=lambda t: t[1])
    assert best_by_rho is not best_by_cycles


@pytest.mark.parametrize("n", [15, 31, 63])
def test_red_black_solves_the_model_problem_in_one_cycle(n, rng):
    """Exact, not lucky: the coarse points sit at the odd fine indices, so after the exact
    coarse solve the odd points carry no error, and an even point couples only to odd
    neighbours."""
    A = mg.poisson_1d(n)
    x_true = rng.standard_normal(n)
    r = mg.solve(A, A @ x_true, tol=1e-10, max_cycles=20, kind="red-black")
    assert r.n_cycles == 1
    np.testing.assert_allclose(r.x, x_true, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n", [15, 31])
def test_one_red_black_post_sweep_annihilates_the_two_grid_operator(n):
    G = np.column_stack([mg.two_grid(mg.poisson_1d(n), np.zeros(n), np.eye(n)[:, j],
                                     0, 1, "red-black") for j in range(n)])
    assert np.linalg.norm(G, 2) < 1e-12, f"||G|| = {np.linalg.norm(G, 2):.3e}"


@pytest.mark.parametrize("n", [15, 31])
def test_pre_smoothing_alone_does_not_annihilate_it(n):
    """The correction comes last and reintroduces error, so the order matters."""
    G = np.column_stack([mg.two_grid(mg.poisson_1d(n), np.zeros(n), np.eye(n)[:, j],
                                     2, 0, "red-black") for j in range(n)])
    assert np.linalg.norm(G, 2) > 1e-3


# ---------------------------------------------------------------- cycle choices


@pytest.mark.parametrize("n", [31, 63, 127])
def test_the_w_cycle_recovers_the_two_grid_factor(n, rng):
    A = mg.poisson_1d(n)
    b = A @ rng.standard_normal(n)
    tg = mg.solve(A, b, cycle="two-grid", tol=1e-10, max_cycles=200)
    w = mg.solve(A, b, cycle="w", tol=1e-10, max_cycles=200)
    v = mg.solve(A, b, cycle="v", tol=1e-10, max_cycles=200)
    assert abs(w.convergence_factor() - tg.convergence_factor()) < 0.005
    assert v.convergence_factor() >= w.convergence_factor() - 1e-9


@pytest.mark.parametrize("n", [63, 127])
def test_more_smoothing_gives_a_better_factor(n, rng):
    A = mg.poisson_1d(n)
    b = A @ rng.standard_normal(n)
    factors = [mg.solve(A, b, tol=1e-10, max_cycles=300,
                        nu1=k, nu2=k).convergence_factor() for k in (1, 2, 3, 4)]
    assert all(a > b for a, b in zip(factors, factors[1:])), f"{factors}"


@pytest.mark.parametrize("n", [31, 63, 127])
def test_injection_is_worse_than_full_weighting_but_still_mesh_independent(n, rng):
    A = mg.poisson_1d(n)
    b = A @ rng.standard_normal(n)
    fw = mg.solve(A, b, tol=1e-10, max_cycles=300)
    inj = mg.solve(A, b, tol=1e-10, max_cycles=300, restrict=mg.restrict_injection)
    assert fw.converged and inj.converged
    assert inj.convergence_factor() > fw.convergence_factor()
    assert inj.convergence_factor() < 0.2, "injection should degrade, not break"


# ---------------------------------------------------------------- the matrix-free version


@pytest.mark.parametrize("n", [7, 31, 127, 1023])
def test_the_stencil_hierarchy_solves_it(n, rng):
    H = mg.StencilHierarchy(n)
    x_true = rng.standard_normal(n)
    r = H.solve(H.apply(0, x_true), tol=1e-10, max_cycles=100)
    assert r.converged
    np.testing.assert_allclose(r.x, x_true, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("n", [7, 31, 127, 511])
def test_the_stencil_apply_matches_the_matrix(n, rng):
    H = mg.StencilHierarchy(n)
    x = rng.standard_normal(n)
    np.testing.assert_allclose(H.apply(0, x), mg.poisson_1d(n) @ x, rtol=1e-11, atol=1e-9)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_every_level_discretises_the_same_operator(level):
    H = mg.StencilHierarchy(255)
    n_l = H.sizes[level]
    x = np.ones(n_l)
    got = H.apply(level, x)
    want = mg.poisson_1d(n_l, 1.0 / (n_l + 1)) @ x
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-8)


def test_the_stencil_version_is_mesh_independent_at_large_n():
    cycles = []
    for k in (10, 12, 14, 16):
        n = 2 ** k - 1
        H = mg.StencilHierarchy(n)
        gen = np.random.default_rng(3)
        r = H.solve(H.apply(0, gen.standard_normal(n)), tol=1e-10, max_cycles=100)
        assert r.converged
        cycles.append(r.n_cycles)
    assert max(cycles) - min(cycles) <= 1, f"cycle counts {cycles}"


def test_the_stencil_and_dense_versions_agree():
    n = 127
    gen = np.random.default_rng(17)
    x_true = gen.standard_normal(n)
    A = mg.poisson_1d(n)
    b = A @ x_true
    dense = mg.solve(A, b, tol=1e-10, max_cycles=100)
    free = mg.StencilHierarchy(n).solve(b, tol=1e-10, max_cycles=100)
    assert dense.n_cycles == free.n_cycles
    np.testing.assert_allclose(free.x, dense.x, rtol=1e-6, atol=1e-8)


def test_the_stencil_hierarchy_rejects_bad_input():
    with pytest.raises(ValueError):
        mg.StencilHierarchy(0)
    with pytest.raises(ValueError):
        mg.StencilHierarchy(15).solve(np.ones(16))


# ---------------------------------------------------------------- robustness


@pytest.mark.parametrize("jump", [1.0, 1e2, 1e4, 1e6])
def test_a_jumping_coefficient_does_not_move_the_cycle_count(jump, rng):
    """The Galerkin coarse operator adapts to the coefficient by itself, which is why RAP is
    built from A rather than rediscretised."""
    n = 127
    coeff = np.ones(n + 1)
    coeff[(n + 1) // 2:] = jump
    A = ((np.diag(coeff[:-1] + coeff[1:]) - np.diag(coeff[1:-1], 1)
          - np.diag(coeff[1:-1], -1)) * (n + 1) ** 2)
    r = mg.solve(A, A @ rng.standard_normal(n), tol=1e-10, max_cycles=500)
    assert r.converged
    assert r.n_cycles <= 12, f"{r.n_cycles} cycles at a jump of {jump:.0e}"


def test_multigrid_beats_cg_by_a_growing_margin():
    from nalib.krylov import conjugate_gradient

    ratios = []
    for n in (63, 127, 255):
        A = mg.poisson_1d(n)
        b = A @ np.ones(n)
        cg = conjugate_gradient(A, b, tol=1e-10, max_iter=10 * n, keep_history=False)
        v = mg.solve(A, b, tol=1e-10, max_cycles=200)
        assert cg.converged and v.converged
        ratios.append(cg.n_iter / v.n_cycles)
    assert all(a < b for a, b in zip(ratios, ratios[1:])), f"ratios {ratios}"


# ---------------------------------------------------------------- the result type


@pytest.mark.parametrize("n", [31, 127])
def test_the_result_reports_what_it_did(n, rng):
    A = mg.poisson_1d(n)
    r = mg.solve(A, A @ rng.standard_normal(n), tol=1e-10, max_cycles=200)
    assert r.levels == mg.n_levels(n)
    assert len(r.residual_norms) == r.n_cycles + 1
    assert r.work_units == r.n_cycles * 8.0            # (nu1 + nu2) * 2 with the defaults
    assert 0.0 < r.convergence_factor() < 1.0


def test_a_cycle_limit_is_reported_rather_than_hidden():
    n = 63
    A = mg.poisson_1d(n)
    r = mg.solve(A, A @ np.ones(n), tol=1e-14, max_cycles=1)
    assert not r.converged
    assert "limit" in r.message
