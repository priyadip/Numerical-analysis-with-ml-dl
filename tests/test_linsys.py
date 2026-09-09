"""Tests for nalib.linsys, nalib.cholesky, nalib.banded and nalib.refinement.

Every factorization is checked against its defining identity, every solve against
`numpy.linalg.solve`, and every claim lessons 19 to 22 make from a measurement is re-checked
here rather than taken on trust.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import (banded as bd, cholesky as ch, linalg as la, linsys as ls, lu,
                   pivoting as pv, refinement as rf)

U_ROUND = np.finfo(float).eps / 2


# ---------------------------------------------------------------- linsys


def test_residual_matches_the_definition(rng):
    for n in (1, 3, 17):
        A = ls.with_condition_number(n, 1e3, rng)
        x, b = rng.standard_normal(n), rng.standard_normal(n)
        np.testing.assert_allclose(ls.residual(A, x, b), b - A @ x, atol=1e-12)


def test_relative_residual_is_invariant_under_scaling(rng):
    """The whole reason to scale: the same problem must give the same number."""
    A = ls.with_condition_number(12, 1e6, rng)
    x_true = rng.standard_normal(12)
    values = []
    for scale in (1e-8, 1.0, 1e8):
        A_s, b_s = scale * A, scale * (A @ x_true)
        values.append(ls.relative_residual(A_s, np.linalg.solve(A_s, b_s), b_s))
    assert max(values) < 1e-14
    assert max(values) / max(min(values), 1e-300) < 100, "should barely move"


def test_raw_residual_is_not_invariant_under_scaling(rng):
    """The contrast that makes the previous test meaningful."""
    A = ls.with_condition_number(12, 1e6, rng)
    x_true = rng.standard_normal(12)
    raws = []
    for scale in (1e-8, 1e8):
        A_s, b_s = scale * A, scale * (A @ x_true)
        raws.append(np.linalg.norm(ls.residual(A_s, np.linalg.solve(A_s, b_s), b_s)))
    assert max(raws) / max(min(raws), 1e-300) > 1e10


def test_forward_error_is_zero_for_the_exact_answer(rng):
    x = rng.standard_normal(9)
    assert ls.forward_error(x, x) == pytest.approx(0.0, abs=1e-15)


def test_error_bound_holds(rng):
    """forward error <= kappa * backward error, the governing inequality."""
    for kappa in (1e2, 1e6, 1e10, 1e14):
        n = 20
        A = ls.with_condition_number(n, kappa, rng)
        for _ in range(20):
            x_true = rng.standard_normal(n)
            b = A @ x_true
            x_hat = np.linalg.solve(A, b)
            bound = ls.error_bound(A, x_hat, b)
            assert ls.forward_error(x_hat, x_true) <= bound * (1 + 1e-6) + 1e-14


def test_error_magnification_never_exceeds_kappa(rng):
    for kappa in (1e4, 1e8):
        n = 15
        A = ls.with_condition_number(n, kappa, rng)
        k = la.condition_number(A, 2)
        for _ in range(200):
            x_true = rng.standard_normal(n)
            b = A @ x_true
            mag = ls.error_magnification(A, np.linalg.solve(A, b), b, x_true)
            if np.isfinite(mag):
                assert mag <= k * (1 + 1e-6)


def test_with_condition_number_is_exact(rng):
    for n in (2, 5, 34):
        for kappa in (1.0, 1e3, 1e10):
            A = ls.with_condition_number(n, kappa, rng)
            assert la.condition_number(A, 2) == pytest.approx(kappa, rel=1e-6)


def test_with_condition_number_rejects_bad_input(rng):
    with pytest.raises(ValueError):
        ls.with_condition_number(5, 0.5, rng)
    with pytest.raises(ValueError):
        ls.with_condition_number(0, 10.0, rng)


def test_hilbert_is_symmetric_and_ill_conditioned():
    for n in (1, 2, 5, 12):
        H = ls.hilbert(n)
        assert ch.is_symmetric(H)
        assert H.shape == (n, n)
        assert H[0, 0] == pytest.approx(1.0)
    assert la.condition_number(ls.hilbert(12), 2) > 1e15
    assert la.condition_number(ls.hilbert(2), 2) < 100


def test_hilbert_entries_match_the_formula():
    H = ls.hilbert(6)
    for i in range(6):
        for j in range(6):
            assert H[i, j] == pytest.approx(1.0 / (i + j + 1))


def test_beam_condition_grows_like_n_to_the_fourth():
    sizes = [8, 16, 32, 64, 128]
    kappas = [la.condition_number(ls.beam_matrix(n), 2) for n in sizes]
    slope = np.polyfit(np.log(sizes), np.log(kappas), 1)[0]
    assert 3.5 < slope < 4.5, f"fitted exponent {slope:.2f}, expected about 4"


def test_beam_matrix_rejects_tiny_sizes():
    with pytest.raises(ValueError):
        ls.beam_matrix(3)


def test_diagnose_system_separates_the_two_failure_modes(rng):
    # ill conditioned problem, stable algorithm
    H = ls.hilbert(13)
    x_true = np.ones(13)
    b = H @ x_true
    d = ls.diagnose_system(H, b, np.linalg.solve(H, b), x_true)
    assert d["backward_stable"], "the algorithm is fine on Hilbert"
    assert d["condition_number"] > 1e12

    # well conditioned problem, unstable algorithm
    W = pv.wilkinson_growth_matrix(60)
    xw = rng.standard_normal(60)
    bw = W @ xw
    dw = ls.diagnose_system(W, bw, pv.plu_solve(pv.plu_factor(W), bw), xw)
    assert dw["condition_number"] < 1e3, "the problem is easy"
    assert not dw["backward_stable"], "but the algorithm failed"


# ---------------------------------------------------------------- cholesky


@pytest.mark.parametrize("n", [1, 2, 3, 7, 23])
def test_cholesky_reproduces_the_matrix(n, rng):
    A = ch.random_spd(n, rng=rng)
    L = ch.cholesky(A)
    np.testing.assert_allclose(L @ L.T, A, atol=1e-9 * max(1.0, np.abs(A).max()))
    assert np.abs(np.triu(L, 1)).max() == 0.0
    assert np.all(np.diag(L) > 0)


@pytest.mark.parametrize("n", [1, 2, 5, 23])
def test_cholesky_matches_numpy(n, rng):
    A = ch.random_spd(n, rng=rng)
    np.testing.assert_allclose(ch.cholesky(A), np.linalg.cholesky(A), atol=1e-9)


def test_cholesky_on_a_hand_case():
    A = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])
    expected = np.array([[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]])
    np.testing.assert_allclose(ch.cholesky(A), expected, atol=1e-12)


def test_cholesky_solve_matches_numpy(rng):
    for n in (1, 4, 20):
        A = ch.random_spd(n, rng=rng)
        b = rng.standard_normal(n)
        np.testing.assert_allclose(ch.cholesky_solve(ch.cholesky(A), b),
                                   np.linalg.solve(A, b), atol=1e-8)


def test_cholesky_refuses_a_non_definite_matrix():
    for M in (np.diag([1.0, -2.0, 3.0]), -np.eye(4), np.ones((3, 3))):
        with pytest.raises(np.linalg.LinAlgError):
            ch.cholesky(M)


def test_cholesky_refuses_a_nonsymmetric_matrix():
    with pytest.raises(ValueError):
        ch.cholesky(np.array([[2.0, 1.0], [0.0, 2.0]]))


def test_is_positive_definite_agrees_with_eigenvalues(rng):
    for _ in range(30):
        n = int(rng.integers(1, 8))
        A = rng.standard_normal((n, n))
        A = (A + A.T) / 2 + rng.uniform(-2, 4) * np.eye(n)
        expected = bool(np.all(np.linalg.eigvalsh(A) > 1e-12))
        assert ch.is_positive_definite(A) == expected


def test_entries_of_l_are_bounded_by_sqrt_of_the_diagonal(rng):
    """The bound that gives Cholesky a growth factor of exactly 1."""
    for n in (5, 20, 60):
        A = ch.random_spd(n, kappa=1e6, rng=rng)
        L = ch.cholesky(A)
        assert np.abs(L).max() <= np.sqrt(np.max(np.diag(A))) + 1e-10


def test_cholesky_costs_half_of_lu():
    for n in (500, 2000, 5000):
        assert ch.flops_cholesky(n) / lu.flops_lu(n) == pytest.approx(0.5, abs=0.01)
    assert ch.flops_cholesky(3000) / (3000**3 / 3) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("n", [1, 2, 5, 13])
def test_ldl_reproduces_the_matrix(n, rng):
    A = ch.random_spd(n, rng=rng)
    L, d = ch.ldl(A)
    np.testing.assert_allclose(L @ np.diag(d) @ L.T, A,
                               atol=1e-9 * max(1.0, np.abs(A).max()))
    np.testing.assert_allclose(np.diag(L), np.ones(n))
    assert np.all(d > 0)


def test_ldl_relates_to_cholesky(rng):
    A = ch.random_spd(9, rng=rng)
    L, d = ch.ldl(A)
    np.testing.assert_allclose(L @ np.diag(np.sqrt(d)), ch.cholesky(A), atol=1e-9)


def test_ldl_handles_an_indefinite_matrix():
    M = np.array([[2.0, 1.0, 0.0], [1.0, -3.0, 1.0], [0.0, 1.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        ch.cholesky(M)
    L, d = ch.ldl(M)
    np.testing.assert_allclose(L @ np.diag(d) @ L.T, M, atol=1e-12)
    assert np.any(d < 0), "an indefinite matrix must produce a negative entry in D"


def test_ldl_solve_matches_numpy(rng):
    for n in (2, 6, 15):
        A = ch.random_spd(n, rng=rng)
        b = rng.standard_normal(n)
        L, d = ch.ldl(A)
        np.testing.assert_allclose(ch.ldl_solve(L, d, b), np.linalg.solve(A, b), atol=1e-8)


def test_inertia_matches_the_eigenvalues(rng):
    cases = [ch.random_spd(8, rng=rng), -ch.random_spd(5, rng=rng),
             np.diag([3.0, -1.0, 2.0, -5.0]), np.zeros((3, 3)),
             np.array([[0.0, 1.0], [1.0, 0.0]])]
    for A in cases:
        got = ch.inertia(A)
        w = np.linalg.eigvalsh(A)
        tol = 1e-10 * max(np.abs(w).max(), 1.0)
        expected = (int(np.sum(w > tol)), int(np.sum(w < -tol)),
                    int(np.sum(np.abs(w) <= tol)))
        assert got == expected


def test_inertia_rejects_a_nonsymmetric_matrix():
    """It would otherwise silently report the inertia of a different matrix."""
    with pytest.raises(ValueError):
        ch.inertia(np.array([[2.0, 1.0], [0.0, 2.0]]))


def test_sylvester_criterion_fails_where_cholesky_does_not():
    """Lesson 20's measurement: the minors underflow on a definite matrix."""
    A = ch.random_spd(120, kappa=1e6, rng=np.random.default_rng(1))
    dets = [np.linalg.det(A[:j, :j]) for j in range(1, 121)]
    assert min(dets) == 0.0, "the determinants should underflow"
    assert not all(d > 0 for d in dets), "so Sylvester gets it wrong"
    assert ch.is_positive_definite(A), "while Cholesky gets it right"


def test_random_spd_has_the_requested_condition_number(rng):
    for n in (2, 8, 30):
        for kappa in (1e2, 1e8):
            A = ch.random_spd(n, kappa=kappa, rng=rng)
            assert ch.is_symmetric(A)
            assert la.condition_number(A, 2) == pytest.approx(kappa, rel=1e-6)


# ---------------------------------------------------------------- banded


@pytest.mark.parametrize("n", [1, 2, 3, 10, 200])
def test_thomas_matches_a_dense_solve(n, rng):
    A = bd.second_difference(n)
    x_true = rng.standard_normal(n)
    d = A @ x_true
    got = bd.thomas(np.full(max(n - 1, 0), -1.0), np.full(n, 2.0),
                    np.full(max(n - 1, 0), -1.0), d)
    np.testing.assert_allclose(got, x_true, atol=1e-8)


def test_thomas_rejects_a_mismatched_right_hand_side():
    with pytest.raises(ValueError):
        bd.thomas(np.ones(2), np.ones(3), np.ones(2), np.ones(4))


def test_thomas_raises_on_a_zero_pivot():
    with pytest.raises(np.linalg.LinAlgError):
        bd.thomas(np.array([1.0]), np.array([0.0, 1.0]), np.array([1.0]), np.ones(2))


def test_second_difference_eigenvalues_are_exact():
    for n in (1, 4, 20, 50):
        A = bd.second_difference(n)
        got = np.sort(np.linalg.eigvalsh(A))
        expected = np.sort(bd.second_difference_eigenvalues(n))
        np.testing.assert_allclose(got, expected, atol=1e-10)


def test_second_difference_is_spd_and_tridiagonal():
    for n in (1, 5, 40):
        A = bd.second_difference(n)
        assert ch.is_positive_definite(A)
        assert bd.bandwidths(A) == (min(1, n - 1), min(1, n - 1))


def test_bandwidths_on_known_shapes():
    assert bd.bandwidths(np.eye(5)) == (0, 0)
    assert bd.bandwidths(bd.second_difference(6)) == (1, 1)
    assert bd.bandwidths(np.ones((5, 5))) == (4, 4)


@pytest.mark.parametrize("n,p,q", [(6, 1, 1), (12, 2, 3), (20, 3, 1), (5, 4, 4)])
def test_banded_lu_reproduces_the_matrix_and_keeps_the_band(n, p, q, rng):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - p), min(n, i + q + 1)):
            M[i, j] = rng.standard_normal()
    M += n * np.eye(n)
    L, U = bd.banded_lu(M, p, q)
    np.testing.assert_allclose(L @ U, M, atol=1e-9 * max(1.0, np.abs(M).max()))
    assert bd.bandwidths(L, tol=1e-13)[0] <= p
    assert bd.bandwidths(U, tol=1e-13)[1] <= q


def test_banded_flops_beat_dense_hugely():
    assert lu.flops_lu(10000) / bd.flops_banded_lu(10000, 1, 1) > 1e6


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (7, 3), (20, 20)])
def test_csr_round_trip_and_matvec(shape, rng):
    A = rng.standard_normal(shape)
    A[np.abs(A) < 0.8] = 0.0
    csr = bd.to_csr(A)
    np.testing.assert_allclose(bd.csr_to_dense(csr), A, atol=1e-15)
    x = rng.standard_normal(shape[1])
    np.testing.assert_allclose(bd.csr_matvec(csr, x), A @ x, atol=1e-12)


def test_csr_matvec_rejects_a_wrong_length_vector(rng):
    csr = bd.to_csr(rng.standard_normal((4, 6)))
    with pytest.raises(ValueError):
        bd.csr_matvec(csr, np.ones(4))


def test_csr_stores_far_less_than_dense():
    counts = bd.storage_counts(bd.second_difference(1000))
    assert counts["csr"] < counts["dense"] / 100


def test_coo_finds_every_nonzero(rng):
    A = rng.standard_normal((9, 5))
    A[np.abs(A) < 1.0] = 0.0
    coo = bd.to_coo(A)
    assert coo["row"].size == np.count_nonzero(A)
    for r, c, v in zip(coo["row"], coo["col"], coo["data"]):
        assert A[r, c] == v


def test_arrow_ordering_decides_the_fill():
    """The headline of lesson 21: same matrix, opposite outcomes."""
    for n in (6, 12, 25):
        assert bd.fill_in(bd.arrow_matrix(n, tip_first=False))["fill"] == 0
        assert bd.fill_in(bd.arrow_matrix(n, tip_first=True))["fill"] > 0
        assert bd.fill_in(bd.arrow_matrix(n, tip_first=True))["density_after"] > 0.99


def test_tridiagonal_has_no_fill():
    for n in (5, 20, 60):
        assert bd.fill_in(bd.second_difference(n))["fill"] == 0


def test_reverse_cuthill_mckee_returns_a_permutation(rng):
    for n in (1, 2, 7, 30):
        A = bd.second_difference(n) if n > 1 else np.ones((1, 1))
        perm = bd.reverse_cuthill_mckee(A)
        assert sorted(perm.tolist()) == list(range(n))


def test_reverse_cuthill_mckee_recovers_a_shuffled_tridiagonal(rng):
    """It finds the good ordering without being told the matrix was ever banded."""
    n = 25
    T = bd.second_difference(n)
    shuffle = rng.permutation(n)
    shuffled = T[np.ix_(shuffle, shuffle)]
    assert max(bd.bandwidths(shuffled)) > 5, "the shuffle should destroy the band"

    perm = bd.reverse_cuthill_mckee(shuffled)
    recovered = shuffled[np.ix_(perm, perm)]
    assert max(bd.bandwidths(recovered)) == 1
    assert bd.fill_in(recovered)["fill"] == 0


def test_density_of_a_tridiagonal_matrix_falls_with_n():
    assert bd.density(bd.second_difference(1000)) < bd.density(bd.second_difference(10))


# ---------------------------------------------------------------- refinement


def test_condition_estimate_is_close_to_the_truth(rng):
    ratios = []
    for n in (5, 20, 60):
        for kappa in (1e2, 1e6, 1e10):
            A = ls.with_condition_number(n, kappa, rng)
            ratios.append(rf.condition_estimate(A) / la.condition_number(A, 1))
    ratios = np.array(ratios)
    assert ratios.max() < 1.01, "Hager must not meaningfully overestimate"
    assert ratios.min() > 0.3, "and must stay within a small factor"


def test_condition_estimate_of_the_identity_is_one():
    for n in (1, 5, 30):
        assert rf.condition_estimate(np.eye(n)) == pytest.approx(1.0, rel=1e-9)


def test_condition_estimate_rejects_nonsquare_and_wrong_norm(rng):
    with pytest.raises(ValueError):
        rf.estimate_norm_inverse(rng.standard_normal((3, 5)))
    with pytest.raises(ValueError):
        rf.estimate_norm_inverse(np.eye(3), p=2)


def test_exact_solution_really_solves_the_stored_system(rng):
    for n in (1, 3, 12):
        A = ls.with_condition_number(n, 1e6, rng)
        b = rng.standard_normal(n)
        x = rf.exact_solution(A, b)
        assert ls.relative_residual(A, x, b) < 10 * U_ROUND


def test_exact_residual_beats_the_working_precision_one(rng):
    A = ls.with_condition_number(20, 1e12, rng)
    b = A @ rng.standard_normal(20)
    x = np.linalg.solve(A, b)
    r_exact = rf.exact_residual(A, b, x)
    r_working = b - A @ x
    # the exact residual is a genuinely different vector, not a rounding of the same one
    assert np.linalg.norm(r_exact - r_working) > 0.0


def test_refinement_recovers_digits_with_an_exact_residual(rng):
    """The headline of lesson 22."""
    for kappa in (1e8, 1e12):
        n = 15
        A = ls.with_condition_number(n, kappa, rng)
        b = A @ rng.standard_normal(n)
        res = rf.compare_residual_precisions(A, b, max_steps=4)
        assert res["exact"][-1] < res["exact"][0] / 1e4, "exact residual must gain digits"
        assert res["exact"][-1] < 1e-13


def test_refinement_achieves_nothing_in_working_precision(rng):
    """The contrast that identifies the mechanism."""
    A = ls.with_condition_number(20, 1e12, rng)
    b = A @ rng.standard_normal(20)
    res = rf.compare_residual_precisions(A, b, max_steps=4)
    assert res["working"][-1] > res["exact"][-1] * 1e6


def test_fsum_alone_is_not_enough(rng):
    """Summing exactly does not help while the products are still rounded."""
    A = ls.with_condition_number(20, 1e12, rng)
    b = A @ rng.standard_normal(20)
    res = rf.compare_residual_precisions(A, b, max_steps=4)
    assert res["fsum"][-1] > res["exact"][-1] * 1e6


def test_refinement_repairs_pivot_growth(rng):
    """Wilkinson's matrix: easy problem, unstable algorithm, fixable by refinement."""
    W = pv.wilkinson_growth_matrix(60)
    b = W @ rng.standard_normal(60)
    x_star = rf.exact_solution(W, b)
    factor = pv.plu_factor(W)
    before = np.linalg.norm(pv.plu_solve(factor, b) - x_star)
    after = np.linalg.norm(
        rf.iterative_refinement(W, b, factor, max_steps=4, residual="exact")["x"] - x_star)
    assert after < before / 100


def test_refinement_rejects_an_unknown_residual_mode(rng):
    A = ls.with_condition_number(5, 10.0, rng)
    with pytest.raises(ValueError):
        rf.iterative_refinement(A, np.ones(5), residual="approximate")


def test_refinement_accepts_several_factorization_forms(rng):
    A = ch.random_spd(10, rng=rng)
    b = rng.standard_normal(10)
    expected = np.linalg.solve(A, b)
    for factor in (pv.plu_factor(A), lu.lu_factor(A), {"L": ch.cholesky(A)},
                   lambda rhs: np.linalg.solve(A, rhs)):
        got = rf.iterative_refinement(A, b, factor, max_steps=1, residual="exact")["x"]
        np.testing.assert_allclose(got, expected, atol=1e-8)


def test_refinement_rejects_an_unrecognised_factorization(rng):
    with pytest.raises(TypeError):
        rf._solve_with("not a factorization", np.ones(3))
