"""Tests for `nalib.qr`.

Four factorizations that agree in exact arithmetic and disagree sharply in floating point, so
the tests check two separate things for each: that ``QR`` reproduces ``A``, which they all do,
and that ``Q`` is orthogonal, which only some do. Conflating those is exactly the mistake
lesson 30 exists to prevent, so no test here checks one without the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import leastsquares as ls
from nalib import qr


SHAPES = [(1, 1), (2, 1), (3, 3), (5, 2), (12, 5), (40, 12), (200, 40)]
WIDE = [(2, 5), (3, 8), (10, 25)]
FACTORIZERS = ["cgs", "mgs", "mgs2", "householder", "givens"]
STABLE = ["mgs2", "householder", "givens"]


def factorize(name, A):
    return {
        "cgs": qr.gram_schmidt_classical,
        "mgs": qr.gram_schmidt_modified,
        "mgs2": lambda M: qr.gram_schmidt_reorthogonalized(M, 2),
        "householder": qr.householder_qr,
        "givens": qr.givens_qr,
    }[name](A)


# ---------------------------------------------------------------- every method factorizes


@pytest.mark.parametrize("name", FACTORIZERS)
@pytest.mark.parametrize("m,n", SHAPES)
def test_qr_reproduces_the_matrix(name, m, n, rng):
    A = rng.standard_normal((m, n))
    Q, R = factorize(name, A)
    assert qr.factorization_error(A, Q, R) < 1e-13


@pytest.mark.parametrize("name", FACTORIZERS)
@pytest.mark.parametrize("m,n", SHAPES)
def test_r_is_upper_triangular(name, m, n, rng):
    A = rng.standard_normal((m, n))
    _, R = factorize(name, A)
    assert np.abs(np.tril(R, -1)).max() < 1e-13


@pytest.mark.parametrize("name", ["householder", "givens"])
@pytest.mark.parametrize("m,n", SHAPES)
def test_the_orthogonal_methods_zero_r_exactly(name, m, n, rng):
    """Not nearly zero: the entries are assigned, so they are bit-for-bit zero. Gram-Schmidt
    cannot promise that, because it never touches them."""
    A = rng.standard_normal((m, n))
    _, R = factorize(name, A)
    assert np.abs(np.tril(R, -1)).max() == 0.0


@pytest.mark.parametrize("name", FACTORIZERS)
@pytest.mark.parametrize("m,n", SHAPES)
def test_q_has_the_right_shape(name, m, n, rng):
    A = rng.standard_normal((m, n))
    Q, R = factorize(name, A)
    assert Q.shape == (m, min(m, n))
    assert R.shape == (min(m, n), n)


@pytest.mark.parametrize("m,n", WIDE)
def test_householder_handles_a_wide_matrix(m, n, rng):
    A = rng.standard_normal((m, n))
    Q, R = qr.householder_qr(A)
    assert qr.factorization_error(A, Q, R) < 1e-13
    assert qr.orthogonality_error(Q) < 1e-13


@pytest.mark.parametrize("scale", [1e-10, 1e-3, 1.0, 1e5, 1e10])
@pytest.mark.parametrize("name", STABLE)
def test_factorization_is_scale_invariant(scale, name, rng):
    A = rng.standard_normal((30, 8))
    Q1, R1 = factorize(name, A)
    Q2, R2 = factorize(name, scale * A)
    np.testing.assert_allclose(np.abs(Q2), np.abs(Q1), atol=1e-10)
    np.testing.assert_allclose(np.abs(R2) / scale, np.abs(R1), rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------- but only some are orthogonal


@pytest.mark.parametrize("name", STABLE)
@pytest.mark.parametrize("exponent", [1, 4, 8, 12])
def test_the_stable_methods_stay_orthogonal_at_every_kappa(name, exponent, rng):
    """The whole point. Orthogonality is a property of the operations, so it cannot depend on
    the matrix they are applied to."""
    A = ls.graded_design(50, 10, 10.0 ** exponent, rng)
    Q, _ = factorize(name, A)
    assert qr.orthogonality_error(Q) < 1e-12, f"{name} at kappa 1e{exponent}"


def test_classical_gram_schmidt_loses_orthogonality_completely():
    """Not a defect of the implementation: it is the algorithm, and a suite that never saw it
    would mean the experiment is not reproducing the known behaviour."""
    A = ls.graded_design(50, 10, 1e11, np.random.default_rng(7))
    assert qr.orthogonality_error(qr.gram_schmidt_classical(A)[0]) > 0.5


def test_the_two_gram_schmidt_variants_have_different_exponents():
    """Classical loses kappa^2 and modified loses kappa. Checked as a fitted slope, because a
    single comparison would not distinguish the two claims."""
    exponents = np.arange(1, 13)
    slopes = {}
    for name in ("cgs", "mgs"):
        losses = []
        for e in exponents:
            A = ls.graded_design(50, 10, 10.0 ** float(e), np.random.default_rng(7))
            losses.append(np.log10(max(qr.orthogonality_error(factorize(name, A)[0]), 1e-17)))
        losses = np.array(losses)
        usable = (losses > -13.0) & (losses < -0.5)
        slopes[name] = np.polyfit(exponents[usable], losses[usable], 1)[0]
    assert 1.5 < slopes["cgs"] < 2.3, f"classical slope {slopes['cgs']:.3f}"
    assert 0.7 < slopes["mgs"] < 1.3, f"modified slope {slopes['mgs']:.3f}"
    assert slopes["cgs"] > slopes["mgs"] + 0.5


@pytest.mark.parametrize("exponent", [4, 8, 12])
def test_modified_beats_classical_at_every_kappa(exponent, rng):
    A = ls.graded_design(50, 10, 10.0 ** exponent, rng)
    loss_c = qr.orthogonality_error(qr.gram_schmidt_classical(A)[0])
    loss_m = qr.orthogonality_error(qr.gram_schmidt_modified(A)[0])
    assert loss_m < loss_c


@pytest.mark.parametrize("exponent", [4, 8, 12])
def test_the_backward_error_sees_nothing(exponent, rng):
    """||A - QR|| is small for every method at every kappa, which is why checking it alone
    would report all five as equally good."""
    A = ls.graded_design(50, 10, 10.0 ** exponent, rng)
    for name in FACTORIZERS:
        Q, R = factorize(name, A)
        assert qr.factorization_error(A, Q, R) < 1e-13, f"{name} at 1e{exponent}"


@pytest.mark.parametrize("passes", [1, 2, 3])
def test_reorthogonalization_reaches_roundoff_at_two_passes(passes, rng):
    A = ls.graded_design(50, 10, 1e12, rng)
    loss = qr.orthogonality_error(qr.gram_schmidt_reorthogonalized(A, passes)[0])
    if passes == 1:
        assert loss > 1e-10, "one pass should NOT be enough at kappa 1e12"
    else:
        assert loss < 1e-13, f"{passes} passes left {loss:.2e}"


def test_a_third_pass_adds_nothing():
    A = ls.graded_design(50, 10, 1e12, np.random.default_rng(3))
    two = qr.orthogonality_error(qr.gram_schmidt_reorthogonalized(A, 2)[0])
    three = qr.orthogonality_error(qr.gram_schmidt_reorthogonalized(A, 3)[0])
    assert abs(np.log10(two) - np.log10(three)) < 1.0


def test_reorthogonalization_rejects_zero_passes():
    with pytest.raises(ValueError):
        qr.gram_schmidt_reorthogonalized(np.eye(3), 0)


@pytest.mark.parametrize("builder", [qr.gram_schmidt_classical, qr.gram_schmidt_modified,
                                     qr.gram_schmidt_reorthogonalized])
def test_gram_schmidt_rejects_a_dependent_column(builder, rng):
    """The test has to be RELATIVE. An exactly dependent column leaves a residual of about
    u times the column norm, not zero, so an == 0.0 guard never fires. Before this was fixed
    the routines returned a Q with orthogonality error 0.92 and reported success."""
    A = rng.standard_normal((10, 3))
    A[:, 2] = A[:, 0] + 2.0 * A[:, 1]
    with pytest.raises(np.linalg.LinAlgError):
        builder(A)


@pytest.mark.parametrize("builder", [qr.gram_schmidt_classical, qr.gram_schmidt_modified,
                                     qr.gram_schmidt_reorthogonalized])
def test_the_dependency_guard_does_not_fire_on_a_merely_hard_matrix(builder, rng):
    """It must catch dependence without refusing every ill conditioned matrix, or it would be
    useless. kappa = 1e12 is hard and not dependent."""
    A = ls.graded_design(30, 6, 1e12, rng)
    Q, R = builder(A)
    assert qr.factorization_error(A, Q, R) < 1e-12


# ---------------------------------------------------------------- Householder pieces


@pytest.mark.parametrize("n", [1, 2, 3, 8, 50])
def test_the_reflector_sends_x_onto_the_axis(n, rng):
    x = rng.standard_normal(n)
    v, alpha = qr.householder_vector(x)
    reflected = qr.apply_householder(v, x.reshape(-1, 1)).ravel()
    assert abs(abs(alpha) - np.linalg.norm(x)) < 1e-12 * max(1.0, np.linalg.norm(x))
    if n > 1:
        assert np.abs(reflected[1:]).max() < 1e-12 * max(1.0, np.linalg.norm(x))
    assert abs(reflected[0] - alpha) < 1e-12 * max(1.0, np.linalg.norm(x))


@pytest.mark.parametrize("n", [2, 5, 20])
def test_the_reflector_matrix_is_symmetric_orthogonal_and_involutive(n, rng):
    v, _ = qr.householder_vector(rng.standard_normal(n))
    P = np.eye(n) - 2.0 * np.outer(v, v) / (v @ v)
    np.testing.assert_allclose(P, P.T, atol=1e-14)
    np.testing.assert_allclose(P.T @ P, np.eye(n), atol=1e-13)
    np.testing.assert_allclose(P @ P, np.eye(n), atol=1e-13)


@pytest.mark.parametrize("x1", [1.0, 1e-8, 1e4, 1e8, 1e12, -1.0, -1e8])
def test_the_sign_choice_leaves_nothing_behind(x1):
    """The good choice zeroes the tail exactly, at every magnitude of the leading entry."""
    x = np.concatenate([[x1], np.ones(4)])
    v, _ = qr.householder_vector(x)
    reflected = qr.apply_householder(v, x.reshape(-1, 1)).ravel()
    assert np.abs(reflected[1:]).max() == 0.0 or \
        np.abs(reflected[1:]).max() < 1e-15 * np.linalg.norm(x)


def test_the_wrong_sign_cancels_completely():
    """At x_1 = 1e8 the subtraction x_1 - ||x|| gives exactly zero, so the reflector loses its
    leading component. The reflection then still happens and accomplishes nothing: the tail it
    was supposed to zero comes back at exactly the same size, negated."""
    x = np.concatenate([[1e8], np.ones(4)])
    bad = x.copy()
    bad[0] -= np.linalg.norm(x)                     # the wrong sign
    assert bad[0] == 0.0, "expected total cancellation in v_1"
    reflected = qr.apply_householder(bad, x.reshape(-1, 1)).ravel()
    # the tail is negated, not removed: exactly as large as before
    assert abs(np.linalg.norm(reflected[1:]) - np.linalg.norm(x[1:])) < 1e-12
    np.testing.assert_allclose(reflected[1:], -x[1:], atol=1e-12)
    # while the correct sign zeroes it outright
    good, _ = qr.householder_vector(x)
    assert np.abs(qr.apply_householder(good, x.reshape(-1, 1)).ravel()[1:]).max() == 0.0


@pytest.mark.parametrize("first", [0.0, -0.0])
def test_a_zero_leading_entry_still_gives_a_valid_reflector(first):
    x = np.concatenate([[first], np.array([3.0, 4.0])])
    v, alpha = qr.householder_vector(x)
    reflected = qr.apply_householder(v, x.reshape(-1, 1)).ravel()
    assert abs(abs(alpha) - 5.0) < 1e-13
    assert np.abs(reflected[1:]).max() < 1e-13


def test_the_reflector_of_a_zero_vector_is_the_identity():
    v, alpha = qr.householder_vector(np.zeros(5))
    assert alpha == 0.0
    np.testing.assert_allclose(qr.apply_householder(v, np.eye(5)), np.eye(5), atol=0.0)


def test_the_reflector_of_an_empty_vector_is_handled():
    v, alpha = qr.householder_vector(np.zeros(0))
    assert v.size == 0 and alpha == 0.0


@pytest.mark.parametrize("m,n", [(10, 4), (50, 12), (200, 30)])
def test_applying_a_reflector_matches_forming_it(m, n, rng):
    v = rng.standard_normal(m)
    B = rng.standard_normal((m, n))
    P = np.eye(m) - 2.0 * np.outer(v, v) / (v @ v)
    np.testing.assert_allclose(qr.apply_householder(v, B), P @ B, atol=1e-11)
    np.testing.assert_allclose(qr.apply_householder(v, B.T, from_left=False), B.T @ P,
                               atol=1e-11)


@pytest.mark.parametrize("m,n", [(6, 3), (30, 8)])
def test_the_full_and_reduced_forms_agree_where_they_overlap(m, n, rng):
    A = rng.standard_normal((m, n))
    Qr, Rr = qr.householder_qr(A, "reduced")
    Qf, Rf = qr.householder_qr(A, "full")
    assert Qr.shape == (m, n) and Rr.shape == (n, n)
    assert Qf.shape == (m, m) and Rf.shape == (m, n)
    np.testing.assert_allclose(Qf[:, :n], Qr, atol=1e-12)
    np.testing.assert_allclose(Rf[:n, :], Rr, atol=1e-12)
    assert np.abs(Rf[n:, :]).max() == 0.0
    np.testing.assert_allclose(Qf.T @ Qf, np.eye(m), atol=1e-12)


def test_householder_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        qr.householder_qr(np.eye(3), "economy")


@pytest.mark.parametrize("m,n", [(20, 4), (200, 8), (2000, 5)])
def test_the_compact_form_applies_q_without_building_it(m, n, rng):
    A = rng.standard_normal((m, n))
    compact = qr.householder_qr_compact(A)
    y = rng.standard_normal(m)
    # Q^T Q y = y needs no reference, so it is checkable at any size
    round_trip = compact["apply_q"](compact["apply_q"](y), transpose=True)
    assert np.linalg.norm(round_trip - y) / np.linalg.norm(y) < 1e-13
    assert compact["storage_entries"] < compact["dense_q_entries"]


@pytest.mark.parametrize("m,n", [(20, 4), (200, 8)])
def test_the_compact_form_matches_the_dense_one(m, n, rng):
    A = rng.standard_normal((m, n))
    compact = qr.householder_qr_compact(A)
    Q, R = qr.householder_qr(A, "full")
    y = rng.standard_normal(m)
    np.testing.assert_allclose(compact["apply_q"](y, transpose=True), Q.T @ y, atol=1e-11)
    np.testing.assert_allclose(compact["apply_q"](y), Q @ y, atol=1e-11)
    np.testing.assert_allclose(compact["R"], R[:min(m, n), :], atol=1e-12)


@pytest.mark.parametrize("m,n", [(100, 5), (1000, 10)])
def test_storing_reflectors_beats_storing_q_by_a_growing_margin(m, n, rng):
    compact = qr.householder_qr_compact(rng.standard_normal((m, n)))
    assert compact["dense_q_entries"] / compact["storage_entries"] > m / (4.0 * n)


# ---------------------------------------------------------------- Givens


@pytest.mark.parametrize("a,b", [(3.0, 4.0), (0.0, 1.0), (1.0, 0.0), (-2.0, 5.0),
                                 (1e200, 1e200), (1e-200, 1e-200), (1e300, 1.0)])
def test_the_rotation_is_orthogonal_across_the_exponent_range(a, b):
    c, s = qr.givens_rotation(a, b)
    assert np.isfinite(c) and np.isfinite(s)
    assert abs(c * c + s * s - 1.0) < 1e-13, f"c^2+s^2 = {c*c+s*s} at ({a:.0e}, {b:.0e})"


@pytest.mark.parametrize("a,b", [(3.0, 4.0), (-1.0, 2.0), (5.0, -0.5), (1e-30, 1e30)])
def test_the_rotation_zeroes_the_second_entry(a, b):
    c, s = qr.givens_rotation(a, b)
    zeroed = -s * a + c * b
    assert abs(zeroed) < 1e-13 * max(1.0, abs(a), abs(b))


@pytest.mark.parametrize("n", [4, 10, 40, 100])
def test_a_hessenberg_matrix_needs_n_minus_one_rotations(n, rng):
    H = np.triu(rng.standard_normal((n, n)), -1)
    assert qr.givens_count(H)["needed"] == n - 1


@pytest.mark.parametrize("n", [10, 40, 100])
def test_the_hessenberg_saving_is_two_over_n(n, rng):
    H = np.triu(rng.standard_normal((n, n)), -1)
    info = qr.givens_count(H)
    assert abs(info["ratio"] - 2.0 / n) < 0.01 * (2.0 / n) + 1e-6


@pytest.mark.parametrize("n", [5, 20, 60])
def test_givens_matches_householder_on_a_hessenberg_matrix(n, rng):
    H = np.triu(rng.standard_normal((n, n)), -1)
    Qg, Rg = qr.givens_qr(H)
    Qh, Rh = qr.householder_qr(H)
    assert qr.factorization_error(H, Qg, Rg) < 1e-12
    np.testing.assert_allclose(np.abs(Rg), np.abs(Rh), atol=1e-10)


@pytest.mark.parametrize("n", [4, 12])
def test_givens_count_of_a_dense_matrix_is_the_full_triangle(n, rng):
    A = rng.standard_normal((n, n))
    info = qr.givens_count(A)
    assert info["needed"] == info["dense"] == n * (n - 1) // 2
    assert info["ratio"] == 1.0


def test_givens_count_of_a_diagonal_matrix_is_zero():
    info = qr.givens_count(np.diag([1.0, 2.0, 3.0, 4.0]))
    assert info["needed"] == 0


# ---------------------------------------------------------------- solving


@pytest.mark.parametrize("name", FACTORIZERS[:1] + STABLE)
@pytest.mark.parametrize("m,n", [(5, 2), (20, 6), (80, 20)])
def test_qr_solve_gets_a_consistent_system_right(name, m, n, rng):
    method = {"cgs": "cgs", "mgs2": "mgs", "householder": "householder",
              "givens": "givens"}[name]
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    got = qr.qr_solve(A, A @ x_true, method=method)
    np.testing.assert_allclose(got, x_true, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("method", ["householder", "givens", "mgs", "cgs"])
@pytest.mark.parametrize("m,n", [(10, 3), (40, 8)])
def test_qr_solve_matches_numpy_lstsq(method, m, n, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    want = np.linalg.lstsq(A, b, rcond=None)[0]
    np.testing.assert_allclose(qr.qr_solve(A, b, method), want, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("exponent", [4, 8, 10, 12])
def test_the_orthogonal_methods_survive_where_the_others_fail(exponent, rng):
    m, n = 60, 8
    A = ls.graded_design(m, n, 10.0 ** exponent, rng)
    x_true = rng.standard_normal(n)
    b = A @ x_true
    rel = lambda z: np.linalg.norm(z - x_true) / np.linalg.norm(x_true)
    for method in ("householder", "givens"):
        assert rel(qr.qr_solve(A, b, method)) < 1e-4, f"{method} at 1e{exponent}"
    if exponent >= 10:
        assert rel(qr.qr_solve(A, b, "cgs")) > 1e-2, "classical GS should have failed here"


def test_householder_and_givens_agree_closely():
    """They are the same idea built from different orthogonal pieces, so they should track
    each other at every condition number."""
    for exponent in (2, 6, 10, 12):
        gen = np.random.default_rng(9)
        A = ls.graded_design(60, 8, 10.0 ** exponent, gen)
        x_true = gen.standard_normal(8)
        b = A @ x_true
        e_h = np.linalg.norm(qr.qr_solve(A, b, "householder") - x_true)
        e_g = np.linalg.norm(qr.qr_solve(A, b, "givens") - x_true)
        assert 0.05 < e_g / max(e_h, 1e-300) < 20.0, f"{e_h:.2e} against {e_g:.2e}"


@pytest.mark.parametrize("m,n", [(4, 2), (20, 5)])
def test_qr_solve_rejects_bad_input(m, n, rng):
    A = rng.standard_normal((m, n))
    with pytest.raises(ValueError):
        qr.qr_solve(A, np.ones(m + 1))
    with pytest.raises(ValueError):
        qr.qr_solve(A, np.ones(m), method="svd")
    with pytest.raises(ValueError):
        qr.qr_solve(rng.standard_normal((n, m)), np.ones(n))     # underdetermined


def test_qr_solve_rejects_a_rank_deficient_matrix(rng):
    A = rng.standard_normal((10, 3))
    A[:, 2] = A[:, 0] - A[:, 1]
    with pytest.raises(np.linalg.LinAlgError):
        qr.qr_solve(A, rng.standard_normal(10), "householder")


# ---------------------------------------------------------------- cost


@pytest.mark.parametrize("m,n", [(100, 10), (1000, 50), (1000, 1000)])
def test_householder_costs_less_than_gram_schmidt(m, n):
    """Which surprises people who expect stability to cost something."""
    assert qr.flops_qr(m, n, "householder") <= qr.flops_qr(m, n, "cgs")


@pytest.mark.parametrize("m,n", [(100, 10), (1000, 50), (1000, 500)])
def test_givens_costs_fifty_percent_more_than_householder_on_a_dense_matrix(m, n):
    ratio = qr.flops_qr(m, n, "givens") / qr.flops_qr(m, n, "householder")
    assert 1.4 < ratio < 1.6, f"ratio {ratio:.3f}"


def test_the_two_gram_schmidt_variants_cost_the_same():
    """Which is why there is never a reason to use the classical one."""
    for m, n in ((100, 10), (1000, 50)):
        assert qr.flops_qr(m, n, "cgs") == qr.flops_qr(m, n, "mgs")


def test_flops_rejects_bad_arguments():
    with pytest.raises(ValueError):
        qr.flops_qr(0, 5)
    with pytest.raises(ValueError):
        qr.flops_qr(10, 5, "svd")


# ---------------------------------------------------------------- the measuring helpers


@pytest.mark.parametrize("n", [2, 5, 20])
def test_orthogonality_error_is_zero_for_an_orthogonal_matrix(n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    assert qr.orthogonality_error(Q) < 1e-14


@pytest.mark.parametrize("n", [3, 8])
def test_orthogonality_error_grows_with_the_perturbation(n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    E = rng.standard_normal((n, n))
    losses = [qr.orthogonality_error(Q + level * E) for level in (1e-8, 1e-5, 1e-2)]
    assert all(a < b for a, b in zip(losses, losses[1:])), f"{losses}"


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_the_factorization_error_is_relative(scale, rng):
    A = rng.standard_normal((20, 5))
    Q, R = qr.householder_qr(A)
    plain = qr.factorization_error(A, Q, R)
    Qs, Rs = qr.householder_qr(scale * A)
    assert abs(np.log10(max(qr.factorization_error(scale * A, Qs, Rs), 1e-300))
               - np.log10(max(plain, 1e-300))) < 1.5


def test_the_guard_marks_the_numerical_rank_boundary():
    """It rejects beyond kappa ~ 1/(m u), which is where a column's residual becomes
    indistinguishable from noise. That boundary is real, not conservative: at working precision
    such a column carries no reliable direction whichever it is."""
    accepted, rejected = [], []
    for exponent in (8, 10, 12, 14, 16):
        A = ls.graded_design(50, 10, 10.0 ** exponent, np.random.default_rng(4))
        try:
            qr.gram_schmidt_modified(A)
            accepted.append(exponent)
        except np.linalg.LinAlgError:
            rejected.append(exponent)
    assert accepted and rejected, f"accepted {accepted}, rejected {rejected}"
    assert max(accepted) < min(rejected), "the boundary should be monotone"
    assert max(accepted) >= 12, f"rejected a usable matrix at 1e{min(rejected)}"


def test_householder_has_no_such_boundary(rng):
    """It never divides by a small residual, so it factorizes anything. Whether the result is
    USEFUL is a separate question, and lesson 33's answer."""
    for exponent in (12, 14, 16, 18):
        A = ls.graded_design(50, 10, 10.0 ** exponent, rng)
        Q, R = qr.householder_qr(A)
        assert qr.factorization_error(A, Q, R) < 1e-12
        assert qr.orthogonality_error(Q) < 1e-12
