"""Tests for nalib.lu and nalib.pivoting.

Factorizations are checked against their defining identity (``A = LU``, ``PA = LU``,
``PAQ = LU``) and solutions against `numpy.linalg.solve`. The flop count formulas are checked
against their leading terms, and the stability claims of lessons 17 and 18 are verified rather
than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import linalg as la, lu, orthogonality as og, pivoting as pv

U_ROUND = np.finfo(float).eps / 2


def well_conditioned(n, rng):
    """A matrix that naive elimination can handle, for testing the no-pivot routines."""
    return rng.standard_normal((n, n)) + n * np.eye(n)


# ---------------------------------------------------------------- triangular solves


def test_back_substitution_on_a_hand_case():
    U = np.array([[2.0, -1.0, 3.0], [0.0, 4.0, -2.0], [0.0, 0.0, 5.0]])
    y = np.array([9.0, 6.0, 15.0])
    np.testing.assert_allclose(lu.back_substitution(U, y), [1.5, 3.0, 3.0], atol=1e-13)


def test_forward_substitution_on_a_hand_case():
    L = np.array([[2.0, 0.0], [3.0, 4.0]])
    b = np.array([4.0, 18.0])
    np.testing.assert_allclose(lu.forward_substitution(L, b), [2.0, 3.0], atol=1e-13)


def test_forward_substitution_with_unit_diagonal(rng):
    for n in (3, 10, 40):
        L = np.tril(rng.standard_normal((n, n)), -1) + np.eye(n)
        y_true = rng.standard_normal(n)
        b = L @ y_true
        np.testing.assert_allclose(lu.forward_substitution(L, b, unit_diagonal=True),
                                   y_true, atol=1e-9)


@pytest.mark.parametrize("n", [2, 5, 25, 80])
def test_triangular_solves_match_numpy(n, rng):
    U = np.triu(rng.standard_normal((n, n))) + n * np.eye(n)
    L = np.tril(rng.standard_normal((n, n))) + n * np.eye(n)
    b = rng.standard_normal(n)
    np.testing.assert_allclose(lu.back_substitution(U, b), np.linalg.solve(U, b), atol=1e-9)
    np.testing.assert_allclose(lu.forward_substitution(L, b), np.linalg.solve(L, b), atol=1e-9)


def test_triangular_solve_raises_on_a_zero_diagonal():
    U = np.array([[1.0, 2.0], [0.0, 0.0]])
    with pytest.raises(np.linalg.LinAlgError):
        lu.back_substitution(U, np.ones(2))
    with pytest.raises(np.linalg.LinAlgError):
        lu.forward_substitution(U.T, np.ones(2))


def test_back_substitution_is_componentwise_backward_stable(rng):
    """Theorem 17.4. The componentwise backward error must stay below n*u."""
    for n in (10, 20, 40, 60):
        U = np.triu(-np.ones((n, n)), 1) + np.eye(n)
        x_true = rng.standard_normal(n)
        y = U @ x_true
        x_hat = lu.back_substitution(U, y)
        r = y - U @ x_hat
        denom = np.abs(U) @ np.abs(x_hat)
        omega = np.max(np.abs(r) / np.where(denom > 0, denom, np.inf))
        assert omega <= n * U_ROUND, f"n={n}: componentwise backward error {omega:.2e}"


def test_triangular_solves_beat_their_condition_number(rng):
    """The measured consequence of Theorem 17.4: far better than kappa*u."""
    ratios = []
    for n in (20, 30, 40):
        U = np.triu(-np.ones((n, n)), 1) + np.eye(n)
        x_true = rng.standard_normal(n)
        x_hat = lu.back_substitution(U, U @ x_true)
        kappa = la.condition_number(U, 2)
        fwd = np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)
        ratios.append(kappa * U_ROUND / fwd)
    assert min(ratios) > 20, f"expected a large margin, got {ratios}"


# ---------------------------------------------------------------- LU without pivoting


@pytest.mark.parametrize("n", [2, 4, 12, 40])
def test_lu_factor_reproduces_the_matrix(n, rng):
    A = well_conditioned(n, rng)
    L, U = lu.lu_factor(A)
    np.testing.assert_allclose(L @ U, A, atol=1e-10)


def test_lu_factor_produces_genuine_triangular_factors(rng):
    L, U = lu.lu_factor(well_conditioned(9, rng))
    assert np.abs(np.triu(L, 1)).max() == 0.0
    assert np.abs(np.tril(U, -1)).max() == 0.0
    np.testing.assert_allclose(np.diag(L), np.ones(9))


def test_l_entries_are_the_elimination_multipliers(rng):
    """Theorem 17.2: L is free, it is the multipliers written down."""
    A = well_conditioned(7, rng)
    L, _ = lu.lu_factor(A)
    res = lu.naive_gaussian_elimination(A, np.zeros(7))
    for i, k, m in res["multipliers"]:
        assert L[i, k] == pytest.approx(m, abs=1e-13)


def test_lu_solve_matches_numpy(rng):
    for n in (3, 10, 50):
        A = well_conditioned(n, rng)
        b = rng.standard_normal(n)
        L, U = lu.lu_factor(A)
        np.testing.assert_allclose(lu.lu_solve(L, U, b), np.linalg.solve(A, b), atol=1e-9)


def test_lu_factor_raises_on_a_zero_pivot():
    """A matrix with determinant -1 that naive elimination cannot handle."""
    Z = np.array([[0.0, 1.0], [1.0, 1.0]])
    assert abs(np.linalg.det(Z)) == pytest.approx(1.0)
    with pytest.raises(np.linalg.LinAlgError):
        lu.lu_factor(Z)


def test_naive_elimination_loses_everything_on_a_tiny_pivot():
    """Swamping, on a matrix with kappa near 2.6."""
    eps = 1e-16
    A = np.array([[eps, 1.0], [1.0, 1.0]])
    b = np.array([1.0 + eps, 2.0])
    assert la.condition_number(A, 2) < 10, "the problem itself must be easy"
    L, U = lu.lu_factor(A)
    x = lu.lu_solve(L, U, b)
    assert np.max(np.abs(x - np.ones(2))) > 0.5, "expected a first-digit error"
    # and pivoting fixes it completely
    x_piv = pv.plu_solve(pv.plu_factor(A), b)
    np.testing.assert_allclose(x_piv, np.ones(2), atol=1e-10)


# ---------------------------------------------------------------- Crout


def test_crout_reproduces_the_matrix(rng):
    for n in (3, 8, 25):
        A = well_conditioned(n, rng)
        L, U = lu.crout_factor(A)
        np.testing.assert_allclose(L @ U, A, atol=1e-10)
        np.testing.assert_allclose(np.diag(U), np.ones(n), atol=1e-13)


def test_crout_and_doolittle_are_the_same_factorization(rng):
    """L_D D = L_C and D^-1 U_D = U_C, with D = diag(U_D)."""
    A = well_conditioned(10, rng)
    Ld, Ud = lu.lu_factor(A)
    Lc, Uc = lu.crout_factor(A)
    D = np.diag(np.diag(Ud))
    np.testing.assert_allclose(Ld @ D, Lc, atol=1e-10)
    np.testing.assert_allclose(np.linalg.inv(D) @ Ud, Uc, atol=1e-10)


def test_crout_raises_on_a_zero_pivot():
    with pytest.raises(np.linalg.LinAlgError):
        lu.crout_factor(np.array([[0.0, 1.0], [1.0, 1.0]]))


# ---------------------------------------------------------------- Gauss-Jordan, determinants


def test_gauss_jordan_computes_the_inverse(rng):
    for n in (2, 6, 20):
        A = well_conditioned(n, rng)
        np.testing.assert_allclose(lu.gauss_jordan(A)["inverse"], np.linalg.inv(A), atol=1e-9)


def test_gauss_jordan_solves_a_system(rng):
    A = well_conditioned(8, rng)
    b = rng.standard_normal(8).reshape(-1, 1)
    np.testing.assert_allclose(lu.gauss_jordan(A, b)["result"].ravel(),
                               np.linalg.solve(A, b.ravel()), atol=1e-9)


def test_determinant_from_lu_matches_numpy(rng):
    for n in (2, 5, 15):
        A = well_conditioned(n, rng)
        _, U = lu.lu_factor(A)
        assert lu.determinant_from_lu(U) == pytest.approx(np.linalg.det(A), rel=1e-8)


def test_cofactor_determinant_agrees_with_lu(rng):
    for n in (1, 2, 4, 7):
        A = rng.standard_normal((n, n))
        assert lu.determinant_by_cofactor(A) == pytest.approx(np.linalg.det(A), rel=1e-8)


def test_cramer_agrees_with_numpy(rng):
    for n in (2, 4, 8):
        A = well_conditioned(n, rng)
        b = rng.standard_normal(n)
        np.testing.assert_allclose(lu.cramer_solve(A, b), np.linalg.solve(A, b), atol=1e-9)


def test_cramer_rejects_a_singular_matrix():
    with pytest.raises(np.linalg.LinAlgError):
        lu.cramer_solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.ones(2))


# ---------------------------------------------------------------- flop counts


def test_lu_flop_count_approaches_two_thirds_n_cubed():
    for n in (100, 1000, 5000):
        assert lu.flops_lu(n) / (2 * n**3 / 3) == pytest.approx(1.0, abs=0.02)


def test_triangular_solve_flop_count_is_n_squared():
    for n in (10, 100, 1000):
        assert lu.flops_triangular_solve(n) == n * n


def test_gauss_jordan_costs_fifty_percent_more():
    """The exact claim of lesson 17 section 5."""
    for n in (500, 2000, 5000):
        assert lu.flops_gauss_jordan(n) / lu.flops_lu(n) == pytest.approx(1.5, abs=0.01)


def test_factorization_dominates_the_solve():
    """At n = 1000 the factorization is over 300 times a single solve."""
    n = 1000
    assert lu.flops_lu(n) / (2 * lu.flops_triangular_solve(n)) > 300


# ---------------------------------------------------------------- partial pivoting


@pytest.mark.parametrize("n", [2, 5, 20, 60])
def test_plu_factor_satisfies_pa_equals_lu(n, rng):
    A = rng.standard_normal((n, n))
    f = pv.plu_factor(A)
    np.testing.assert_allclose(f["P"] @ A, f["L"] @ f["U"], atol=1e-10)
    np.testing.assert_allclose(A[f["perm"]], f["L"] @ f["U"], atol=1e-10)


def test_partial_pivoting_bounds_every_multiplier_by_one(rng):
    for n in (5, 20, 60):
        for _ in range(20):
            f = pv.plu_factor(rng.standard_normal((n, n)))
            assert np.abs(f["L"]).max() <= 1.0 + 1e-14


def test_plu_handles_a_zero_pivot(rng):
    Z = np.array([[0.0, 1.0], [1.0, 1.0]])
    f = pv.plu_factor(Z)
    np.testing.assert_allclose(f["P"] @ Z, f["L"] @ f["U"], atol=1e-13)
    np.testing.assert_allclose(pv.plu_solve(f, np.array([1.0, 2.0])),
                               np.linalg.solve(Z, np.array([1.0, 2.0])), atol=1e-12)


def test_plu_solve_matches_numpy(rng):
    for n in (3, 15, 60):
        A = rng.standard_normal((n, n))
        b = rng.standard_normal(n)
        np.testing.assert_allclose(pv.plu_solve(pv.plu_factor(A), b),
                                   np.linalg.solve(A, b), atol=1e-8)


def test_permutation_matrix_is_orthogonal(rng):
    f = pv.plu_factor(rng.standard_normal((12, 12)))
    assert og.orthogonality_error(f["P"]) < 1e-14
    assert la.condition_number(f["P"], 2) == pytest.approx(1.0, abs=1e-10)


def test_determinant_from_plu_with_the_swap_sign(rng):
    for n in (3, 8, 20):
        A = rng.standard_normal((n, n))
        f = pv.plu_factor(A)
        det = (-1) ** f["n_swaps"] * np.prod(np.diag(f["U"]))
        assert det == pytest.approx(np.linalg.det(A), rel=1e-8)


def test_index_permutation_equals_matrix_permutation(rng):
    A = rng.standard_normal((30, 30))
    f = pv.plu_factor(A)
    b = rng.standard_normal(30)
    np.testing.assert_allclose(f["P"] @ b, b[f["perm"]], atol=1e-15)


# ---------------------------------------------------------------- growth factor


def test_wilkinson_matrix_attains_the_bound_exactly():
    """Theorem 18.3's bound is attained, not merely approached."""
    for n in (5, 10, 20, 40, 60):
        W = pv.wilkinson_growth_matrix(n)
        assert pv.growth_factor(W, "partial") == pytest.approx(2.0 ** (n - 1), rel=1e-9)


def test_partial_pivoting_never_swaps_on_wilkinsons_matrix():
    for n in (5, 20, 50):
        assert pv.plu_factor(pv.wilkinson_growth_matrix(n))["n_swaps"] == 0


def test_wilkinson_matrix_is_well_conditioned():
    """The point of the example: an easy problem the algorithm gets wrong."""
    for n in (20, 40, 60):
        assert la.condition_number(pv.wilkinson_growth_matrix(n), 2) < 100


def test_growth_destroys_accuracy_on_wilkinsons_matrix(rng):
    n = 60
    W = pv.wilkinson_growth_matrix(n)
    x_true = rng.standard_normal(n)
    x_hat = pv.plu_solve(pv.plu_factor(W), W @ x_true)
    fwd = np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)
    assert fwd > 1e-3, "expected total loss of accuracy despite kappa being small"


def test_growth_is_small_on_random_matrices(rng):
    """The empirical fact with no proof."""
    for n in (16, 32, 64):
        g = np.array([pv.growth_factor(rng.standard_normal((n, n)), "partial")
                      for _ in range(200)])
        assert np.median(g) < 10
        assert g.max() < 50
        assert g.max() * 1e3 < 2.0 ** (n - 1), "nowhere near the bound"


def test_growth_factor_can_dip_below_one(rng):
    """max|U| / max|A| is NOT bounded below by 1.

    If the largest entry of A sits below the diagonal, elimination zeroes it and it never
    appears in U at all, so the ratio can come out under 1. Measured: about 6 percent of
    random 8x8 matrices. Definitions that take the maximum over every intermediate matrix,
    rather than over U alone, are bounded below by 1; this one is not.
    """
    g = np.array([pv.growth_factor(rng.standard_normal((8, 8)), "partial")
                  for _ in range(2000)])
    assert g.min() < 1.0, "some matrices should shrink"
    assert g.min() > 0.1, "but not by very much"
    assert np.median(g) > 1.0, "the typical case still grows"


def test_growth_factor_rejects_an_unknown_strategy():
    with pytest.raises(ValueError):
        pv.growth_factor(np.eye(3), "sideways")


def test_no_pivoting_growth_can_be_enormous():
    A = np.array([[1e-16, 1.0], [1.0, 1.0]])
    assert pv.growth_factor(A, "none") > 1e10
    assert pv.growth_factor(A, "partial") < 10


# ---------------------------------------------------------------- complete pivoting


@pytest.mark.parametrize("n", [3, 10, 30])
def test_complete_pivoting_satisfies_paq_equals_lu(n, rng):
    A = rng.standard_normal((n, n))
    f = pv.complete_pivot_factor(A)
    np.testing.assert_allclose(f["P"] @ A @ f["Q"], f["L"] @ f["U"], atol=1e-10)


def test_complete_pivoting_tames_wilkinsons_matrix():
    for n in (20, 30):
        assert pv.growth_factor(pv.wilkinson_growth_matrix(n), "partial") > 1e5
        assert pv.growth_factor(pv.wilkinson_growth_matrix(n), "complete") < 100


def test_complete_pivoting_bounds_multipliers_by_one(rng):
    f = pv.complete_pivot_factor(rng.standard_normal((20, 20)))
    assert np.abs(f["L"]).max() <= 1.0 + 1e-14


# ---------------------------------------------------------------- multipliers


def test_multiplier_sizes_are_bounded_with_pivoting(rng):
    for _ in range(20):
        m = pv.multiplier_sizes(rng.standard_normal((15, 15)), "partial")
        assert m.max() <= 1.0 + 1e-14


def test_multiplier_sizes_are_unbounded_without_pivoting():
    A = np.array([[1e-16, 1.0], [1.0, 1.0]])
    assert pv.multiplier_sizes(A, "none").max() > 1e15
