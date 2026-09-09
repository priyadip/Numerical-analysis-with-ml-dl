"""Tests for `nalib.cholesky`.

The theme is that Cholesky is the one factorization that needs no pivoting, and the tests are
built around the properties that make that true: positive pivots, growth exactly 1, and a
factor determined uniquely by A. Every size is a parameter, and the value ranges are swept
deliberately, because a routine that works at n = 3 with entries near 1 tells you nothing about
n = 200 with entries near 1e-8.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import cholesky as ch


SIZES = [1, 2, 3, 5, 8, 13, 30, 64]
SCALES = [1e-10, 1e-3, 1.0, 1e3, 1e10]


# ---------------------------------------------------------------- symmetry


@pytest.mark.parametrize("n", SIZES)
def test_is_symmetric_accepts_symmetric_and_rejects_the_rest(n, rng):
    B = rng.standard_normal((n, n))
    assert ch.is_symmetric(B + B.T)
    if n > 1:                                    # a 1x1 matrix is always symmetric
        C = B + B.T
        C[0, n - 1] += 1.0
        assert not ch.is_symmetric(C)


def test_is_symmetric_rejects_non_square():
    assert not ch.is_symmetric(np.ones((2, 3)))


# ---------------------------------------------------------------- the factorization


@pytest.mark.parametrize("n", SIZES)
def test_cholesky_reproduces_the_matrix(n, rng):
    A = ch.random_spd(n, rng=rng)
    L = ch.cholesky(A)
    np.testing.assert_allclose(L @ L.T, A, rtol=1e-11, atol=1e-11 * np.abs(A).max())


@pytest.mark.parametrize("n", SIZES)
def test_cholesky_factor_is_lower_triangular_with_positive_diagonal(n, rng):
    L = ch.cholesky(ch.random_spd(n, rng=rng))
    assert np.allclose(np.triu(L, 1), 0.0)
    assert np.all(np.diag(L) > 0.0)


@pytest.mark.parametrize("scale", SCALES)
def test_cholesky_holds_across_ten_orders_of_magnitude(scale, rng):
    n = 12
    A = scale * ch.random_spd(n, rng=rng)
    L = ch.cholesky(A)
    err = np.abs(L @ L.T - A).max() / np.abs(A).max()
    assert err < 1e-12, f"relative error {err:.2e} at scale {scale:.0e}"


@pytest.mark.parametrize("kappa", [1.0, 1e2, 1e6, 1e10])
def test_cholesky_accuracy_degrades_only_as_kappa_allows(kappa, rng):
    n = 20
    A = ch.random_spd(n, kappa=kappa, rng=rng)
    L = ch.cholesky(A)
    backward = np.abs(L @ L.T - A).max() / np.abs(A).max()
    # Cholesky is backward stable with NO growth, so the backward error is O(u) whatever kappa
    assert backward < 1e-13, f"backward error {backward:.2e} at kappa {kappa:.0e}"


@pytest.mark.parametrize("n", SIZES)
def test_cholesky_matches_numpy(n, rng):
    A = ch.random_spd(n, rng=rng)
    np.testing.assert_allclose(ch.cholesky(A), np.linalg.cholesky(A), atol=1e-11)


def test_cholesky_rejects_a_non_positive_definite_matrix():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])       # eigenvalues 3 and -1
    assert not ch.is_positive_definite(A)
    with pytest.raises(np.linalg.LinAlgError):
        ch.cholesky(A)


def test_cholesky_rejects_a_nonsymmetric_matrix():
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        ch.cholesky(np.array([[2.0, 1.0], [0.0, 2.0]]))


def test_cholesky_rejects_a_semidefinite_matrix():
    A = np.array([[1.0, 1.0], [1.0, 1.0]])       # rank 1, smallest eigenvalue exactly 0
    with pytest.raises(np.linalg.LinAlgError):
        ch.cholesky(A)


# ---------------------------------------------------------------- solving


@pytest.mark.parametrize("n", SIZES)
def test_cholesky_solve_recovers_a_known_answer(n, rng):
    A = ch.random_spd(n, rng=rng)
    x = rng.standard_normal(n)
    b = A @ x
    got = ch.cholesky_solve(ch.cholesky(A), b)
    np.testing.assert_allclose(got, x, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("scale", SCALES)
def test_cholesky_solve_is_scale_invariant(scale, rng):
    n = 15
    A = ch.random_spd(n, rng=rng)
    x = rng.standard_normal(n)
    plain = ch.cholesky_solve(ch.cholesky(A), A @ x)
    scaled = ch.cholesky_solve(ch.cholesky(scale * A), scale * (A @ x))
    np.testing.assert_allclose(plain, scaled, rtol=1e-8, atol=1e-10)


def test_cholesky_solve_rejects_a_length_mismatch(rng):
    L = ch.cholesky(ch.random_spd(4, rng=rng))
    with pytest.raises(ValueError):
        ch.cholesky_solve(L, np.ones(5))


# ---------------------------------------------------------------- LDL


@pytest.mark.parametrize("n", SIZES)
def test_ldl_reproduces_a_definite_matrix(n, rng):
    A = ch.random_spd(n, rng=rng)
    L, d = ch.ldl(A)
    np.testing.assert_allclose(L @ np.diag(d) @ L.T, A, rtol=1e-10,
                               atol=1e-10 * np.abs(A).max())


@pytest.mark.parametrize("n", SIZES)
def test_ldl_reproduces_an_indefinite_matrix(n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = rng.standard_normal(n)
    lam[np.abs(lam) < 0.2] = 0.5                 # keep it away from singular
    A = (Q * lam) @ Q.T
    A = (A + A.T) / 2
    L, d = ch.ldl(A)
    np.testing.assert_allclose(L @ np.diag(d) @ L.T, A, rtol=1e-7,
                               atol=1e-8 * np.abs(A).max())


@pytest.mark.parametrize("n", SIZES)
def test_ldl_has_a_unit_diagonal(n, rng):
    L, _ = ch.ldl(ch.random_spd(n, rng=rng))
    np.testing.assert_allclose(np.diag(L), np.ones(n), atol=1e-14)


@pytest.mark.parametrize("n", SIZES)
def test_ldl_d_is_positive_exactly_when_the_matrix_is(n, rng):
    _, d = ch.ldl(ch.random_spd(n, rng=rng))
    assert np.all(d > 0.0)


@pytest.mark.parametrize("n", SIZES)
def test_ldl_solve_recovers_a_known_answer(n, rng):
    A = ch.random_spd(n, rng=rng)
    x = rng.standard_normal(n)
    L, d = ch.ldl(A)
    np.testing.assert_allclose(ch.ldl_solve(L, d, A @ x), x, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("n", SIZES)
def test_ldl_relates_to_cholesky_by_the_square_root_of_d(n, rng):
    A = ch.random_spd(n, rng=rng)
    L, d = ch.ldl(A)
    np.testing.assert_allclose(L * np.sqrt(d), ch.cholesky(A), atol=1e-10)


# ---------------------------------------------------------------- inertia


@pytest.mark.parametrize("n", SIZES)
def test_inertia_counts_the_signs_of_the_eigenvalues(n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = rng.standard_normal(n)
    lam[np.abs(lam) < 0.3] = 0.7
    A = (Q * lam) @ Q.T
    A = (A + A.T) / 2
    pos, neg, zero = ch.inertia(A)
    assert (pos, neg, zero) == (int(np.sum(lam > 0)), int(np.sum(lam < 0)), 0)


@pytest.mark.parametrize("n", SIZES)
def test_inertia_totals_the_dimension(n, rng):
    A = ch.random_spd(n, rng=rng)
    assert sum(ch.inertia(A)) == n


@pytest.mark.parametrize("n", [2, 5, 11])
def test_sylvesters_law_inertia_survives_congruence(n, rng):
    """X^T A X has the same inertia as A for any invertible X. This is the theorem that makes
    inertia computable from LDL at all."""
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.sign(rng.standard_normal(n)) * rng.uniform(0.5, 3.0, n)
    A = (Q * lam) @ Q.T
    A = (A + A.T) / 2
    X = rng.standard_normal((n, n)) + 3.0 * np.eye(n)     # well away from singular
    B = X.T @ A @ X
    assert ch.inertia(A) == ch.inertia((B + B.T) / 2)


def test_inertia_refuses_a_nonsymmetric_matrix():
    with pytest.raises(ValueError):
        ch.inertia(np.array([[1.0, 2.0], [0.0, 1.0]]))


# ---------------------------------------------------------------- random_spd


@pytest.mark.parametrize("n", SIZES)
def test_random_spd_is_symmetric_and_definite(n, rng):
    A = ch.random_spd(n, rng=rng)
    assert ch.is_symmetric(A)
    assert ch.is_positive_definite(A)


@pytest.mark.parametrize("kappa", [1.0, 10.0, 1e4, 1e8, 1e12])
def test_random_spd_hits_the_requested_condition_number(kappa, rng):
    A = ch.random_spd(25, kappa=kappa, rng=rng)
    got = np.linalg.cond(A)
    assert abs(got - kappa) <= 1e-6 * kappa + 1e-8, f"asked {kappa:.0e}, got {got:.3e}"


def test_random_spd_rejects_a_bad_size():
    with pytest.raises(ValueError):
        ch.random_spd(0)


def test_random_spd_rejects_a_condition_number_below_one():
    with pytest.raises(ValueError):
        ch.random_spd(4, kappa=0.5)


# ---------------------------------------------------------------- cost


@pytest.mark.parametrize("n", SIZES)
def test_cholesky_costs_about_half_of_lu(n):
    from nalib.lu import flops_lu

    if n >= 8:                                   # the ratio is asymptotic
        ratio = flops_lu(n) / ch.flops_cholesky(n)
        assert 1.6 < ratio < 2.4, f"ratio {ratio:.2f} at n={n}"


@pytest.mark.parametrize("n", [10, 50, 200, 1000])
def test_cholesky_flops_follow_n_cubed_over_three(n):
    assert abs(ch.flops_cholesky(n) / (n ** 3 / 3.0) - 1.0) < 0.35


# ---------------------------------------------------------------- growth


@pytest.mark.parametrize("n", [3, 8, 20, 50])
def test_cholesky_growth_is_exactly_one(n, rng):
    """The reason Cholesky needs no pivoting: no intermediate entry can exceed the largest
    entry of A, because max|L_ij|^2 <= a_ii and a_ii <= max|a_ij| for an SPD matrix."""
    A = ch.random_spd(n, kappa=1e6, rng=rng)
    L = ch.cholesky(A)
    growth = np.abs(L).max() ** 2 / np.abs(A).max()
    assert growth <= 1.0 + 1e-12, f"growth {growth:.6f} at n={n}"


@pytest.mark.parametrize("n", [1, 4, 17])
def test_cholesky_is_unique(n, rng):
    """Running it twice on the same matrix gives the same factor, bit for bit. Uniqueness is a
    theorem (positive diagonal fixes the sign of each column) and this checks the code has no
    hidden state."""
    A = ch.random_spd(n, rng=rng)
    np.testing.assert_array_equal(ch.cholesky(A), ch.cholesky(A))


# ---------------------------------------------------------------- the 1x1 edge case


def test_everything_works_at_size_one():
    A = np.array([[9.0]])
    L = ch.cholesky(A)
    np.testing.assert_allclose(L, [[3.0]])
    np.testing.assert_allclose(ch.cholesky_solve(L, np.array([18.0])), [2.0])
    Ld, d = ch.ldl(A)
    np.testing.assert_allclose(Ld, [[1.0]])
    np.testing.assert_allclose(d, [9.0])
    assert ch.inertia(A) == (1, 0, 0)
