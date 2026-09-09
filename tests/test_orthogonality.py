"""Tests for nalib.orthogonality.

The defining identities are checked directly rather than assumed: ``Q^T Q = I``,
``||Qx|| = ||x||``, ``P^2 = P``, ``P^T = P``, and ``Q^T(x - Px) = 0``.

The two quantitative claims of lesson 16 are also verified: an orthogonal projector has norm
exactly 1, and an oblique one has norm ``1/cos(theta_max)`` with no upper bound.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import linalg as la, orthogonality as og


# ---------------------------------------------------------------- inner products


def test_inner_matches_numpy_dot(rng):
    for _ in range(30):
        x, y = rng.standard_normal(8), rng.standard_normal(8)
        assert og.inner(x, y) == pytest.approx(float(np.dot(x, y)), rel=1e-13)


def test_inner_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        og.inner(np.ones(3), np.ones(4))


def test_angle_between_known_cases():
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert og.angle_between(e1, e2, degrees=True) == pytest.approx(90.0)
    assert og.angle_between(e1, e1, degrees=True) == pytest.approx(0.0, abs=1e-7)
    assert og.angle_between(e1, -e1, degrees=True) == pytest.approx(180.0)
    assert og.angle_between(e1, np.array([1.0, 1.0]), degrees=True) == pytest.approx(45.0)


def test_angle_between_does_not_return_nan_for_parallel_vectors(rng):
    """Roundoff can push the cosine past 1. Without clipping, arccos gives nan."""
    for _ in range(200):
        x = rng.standard_normal(50)
        assert not np.isnan(og.angle_between(x, x.copy()))
        assert not np.isnan(og.angle_between(x, 3.7 * x))


def test_angle_to_the_zero_vector_is_rejected():
    with pytest.raises(ValueError):
        og.angle_between(np.zeros(3), np.ones(3))


def test_is_orthogonal_set(rng):
    Q = og.random_orthogonal(6, rng)
    assert og.is_orthogonal_set([Q[:, 0], Q[:, 1], Q[:, 2]])
    assert og.is_orthonormal_set([Q[:, 0], Q[:, 1], Q[:, 2]])
    assert og.is_orthogonal_set([2 * Q[:, 0], 5 * Q[:, 1]])
    assert not og.is_orthonormal_set([2 * Q[:, 0], 5 * Q[:, 1]])
    assert not og.is_orthogonal_set([Q[:, 0], Q[:, 0] + Q[:, 1]])


def test_expand_in_basis_is_the_same_as_solving(rng):
    """The point of orthonormality: coefficients are inner products, not a linear solve."""
    for n in (4, 10, 25):
        Q = og.random_orthogonal(n, rng)
        x = rng.standard_normal(n)
        np.testing.assert_allclose(og.expand_in_basis(x, Q),
                                   np.linalg.solve(Q, x), atol=1e-12)


def test_expand_in_basis_reconstructs_exactly(rng):
    for _ in range(20):
        Q = og.random_orthogonal(9, rng)
        x = rng.standard_normal(9)
        np.testing.assert_allclose(Q @ og.expand_in_basis(x, Q), x, atol=1e-13)


# ---------------------------------------------------------------- orthogonal matrices


def test_random_orthogonal_really_is_orthogonal(rng):
    for n in (2, 5, 20, 60):
        Q = og.random_orthogonal(n, rng)
        assert og.orthogonality_error(Q) < 1e-13
        assert og.is_orthogonal_matrix(Q)


def test_random_orthogonal_has_determinant_plus_or_minus_one(rng):
    dets = [np.linalg.det(og.random_orthogonal(8, rng)) for _ in range(200)]
    assert all(abs(abs(d) - 1.0) < 1e-10 for d in dets)
    assert sum(d > 0 for d in dets) > 50, "both signs should appear"
    assert sum(d < 0 for d in dets) > 50


def test_orthogonal_matrices_preserve_the_two_norm(rng):
    """Theorem 16.3, the fact the rest of the course rests on."""
    for n in (3, 20, 80):
        Q = og.random_orthogonal(n, rng)
        for _ in range(200):
            x = rng.standard_normal(n)
            assert np.linalg.norm(Q @ x) == pytest.approx(np.linalg.norm(x), rel=1e-13)


def test_orthogonal_matrices_preserve_inner_products(rng):
    Q = og.random_orthogonal(10, rng)
    for _ in range(200):
        x, y = rng.standard_normal(10), rng.standard_normal(10)
        assert og.inner(Q @ x, Q @ y) == pytest.approx(og.inner(x, y), abs=1e-12)


def test_condition_number_of_an_orthogonal_matrix_is_exactly_one(rng):
    """Corollary 16.4."""
    for n in (2, 10, 50):
        Q = og.random_orthogonal(n, rng)
        assert la.condition_number(Q, 2) == pytest.approx(1.0, abs=1e-10)
        assert la.matrix_norm(Q, 2) == pytest.approx(1.0, abs=1e-10)


def test_error_is_carried_unchanged_through_many_orthogonal_steps(rng):
    """The architectural claim of lesson 16: 200 steps, no change at all."""
    n = 15
    e = rng.standard_normal(n)
    e = e / np.linalg.norm(e)
    for _ in range(200):
        e = og.random_orthogonal(n, rng) @ e
    assert np.linalg.norm(e) == pytest.approx(1.0, abs=1e-11)


def test_product_of_orthogonal_matrices_is_orthogonal(rng):
    Q1, Q2, Q3 = (og.random_orthogonal(7, rng) for _ in range(3))
    assert og.orthogonality_error(Q1 @ Q2 @ Q3) < 1e-12


def test_is_orthogonal_matrix_rejects_a_wide_matrix(rng):
    assert not og.is_orthogonal_matrix(rng.standard_normal((3, 5)))


def test_tall_q_has_orthonormal_columns_but_qqt_is_not_the_identity(rng):
    """The mistake lesson 16 section 9 warns about."""
    Q = og.random_orthogonal(8, rng)[:, :3]
    rows, cols = Q.shape
    np.testing.assert_allclose(Q.T @ Q, np.eye(cols), atol=1e-13)
    assert np.abs(Q @ Q.T - np.eye(rows)).max() > 0.5
    assert np.linalg.matrix_rank(Q @ Q.T) == cols


# ---------------------------------------------------------------- rotations, reflections


def test_rotation_is_orthogonal_with_determinant_plus_one(rng):
    for th in rng.uniform(0, 2 * np.pi, 40):
        R = og.rotation_2d(th)
        assert og.orthogonality_error(R) < 1e-14
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_reflection_is_orthogonal_with_determinant_minus_one(rng):
    for th in rng.uniform(0, 2 * np.pi, 40):
        F = og.reflection_2d(th)
        assert og.orthogonality_error(F) < 1e-14
        assert np.linalg.det(F) == pytest.approx(-1.0)


def test_a_reflection_is_its_own_inverse(rng):
    for th in rng.uniform(0, np.pi, 20):
        F = og.reflection_2d(th)
        np.testing.assert_allclose(F @ F, np.eye(F.shape[0]), atol=1e-13)


def test_rotations_compose_by_adding_angles():
    a, b = 0.3, 1.1
    np.testing.assert_allclose(og.rotation_2d(a) @ og.rotation_2d(b),
                               og.rotation_2d(a + b), atol=1e-13)


# ---------------------------------------------------------------- Householder


def test_householder_maps_x_onto_the_first_axis(rng):
    for n in (2, 5, 30):
        for _ in range(20):
            x = rng.standard_normal(n)
            H = og.householder_reflector(x)
            Hx = H @ x
            assert np.abs(Hx[1:]).max() < 1e-13 * np.linalg.norm(x)
            assert abs(abs(Hx[0]) - np.linalg.norm(x)) < 1e-13 * np.linalg.norm(x)


def test_householder_is_orthogonal_and_symmetric(rng):
    for _ in range(30):
        x = rng.standard_normal(9)
        H = og.householder_reflector(x)
        assert og.orthogonality_error(H) < 1e-13
        assert np.abs(H - H.T).max() < 1e-14
        np.testing.assert_allclose(H @ H, np.eye(H.shape[0]), atol=1e-12)


def test_householder_has_determinant_minus_one(rng):
    for _ in range(20):
        H = og.householder_reflector(rng.standard_normal(6))
        assert np.linalg.det(H) == pytest.approx(-1.0, abs=1e-10)


def test_householder_survives_a_vector_already_on_the_axis():
    """The case where the other sign choice divides by zero."""
    for n in (2, 5, 20):
        for sign in (1.0, -1.0):
            x = np.zeros(n)
            x[0] = sign * 3.0
            H = og.householder_reflector(x)
            assert og.orthogonality_error(H) < 1e-13
            assert np.abs((H @ x)[1:]).max() < 1e-14


def test_householder_handles_the_zero_vector():
    n = 4
    np.testing.assert_allclose(og.householder_reflector(np.zeros(n)), np.eye(n))


def test_the_bad_sign_choice_divides_by_zero_on_the_axis():
    """Documents exactly why nalib picks sign(x_1). Not a test of nalib, but of the claim."""
    x = np.array([1.0, 0.0, 0.0])
    v_bad = x.copy()
    v_bad[0] -= np.linalg.norm(x)
    assert float(v_bad @ v_bad) == 0.0, "the bad sign really does give a zero denominator"


# ---------------------------------------------------------------- projectors


def test_orthogonal_projector_satisfies_both_defining_properties(rng):
    for m, k in [(6, 2), (10, 5), (20, 1)]:
        Q = og.random_orthogonal(m, rng)[:, :k]
        P = og.orthogonal_projector(Q)
        assert og.is_projector(P)
        assert og.is_orthogonal_projector(P)
        np.testing.assert_allclose(P @ P, P, atol=1e-12)
        np.testing.assert_allclose(P, P.T, atol=1e-13)


def test_orthogonal_projector_has_norm_exactly_one(rng):
    for m, k in [(5, 1), (12, 4), (30, 17)]:
        Q = og.random_orthogonal(m, rng)[:, :k]
        assert la.matrix_norm(og.orthogonal_projector(Q), 2) == pytest.approx(1.0, abs=1e-10)


def test_projector_trace_equals_its_rank(rng):
    for m, k in [(8, 3), (15, 7)]:
        P = og.orthogonal_projector(og.random_orthogonal(m, rng)[:, :k])
        assert np.trace(P) == pytest.approx(k, abs=1e-10)
        assert np.linalg.matrix_rank(P) == k


def test_projector_eigenvalues_are_zero_and_one_only(rng):
    P = og.orthogonal_projector(og.random_orthogonal(10, rng)[:, :4])
    ev = np.sort(np.linalg.eigvals(P).real)
    assert np.abs(ev[:6]).max() < 1e-12
    assert np.abs(ev[6:] - 1.0).max() < 1e-12


def test_complementary_projector_is_a_projector(rng):
    P = og.orthogonal_projector(og.random_orthogonal(9, rng)[:, :4])
    C = og.complementary_projector(P)
    assert og.is_projector(C)
    assert og.is_orthogonal_projector(C)
    assert np.linalg.matrix_rank(C) == 5


def test_the_two_pieces_sum_to_the_original_and_are_orthogonal(rng):
    Q = og.random_orthogonal(11, rng)[:, :4]
    P = og.orthogonal_projector(Q)
    C = og.complementary_projector(P)
    for _ in range(50):
        x = rng.standard_normal(11)
        np.testing.assert_allclose(P @ x + C @ x, x, atol=1e-12)
        assert og.inner(P @ x, C @ x) == pytest.approx(0.0, abs=1e-12)


def test_rank_one_projector(rng):
    for _ in range(20):
        q = rng.standard_normal(7)
        P = og.rank_one_projector(q)
        assert og.is_orthogonal_projector(P)
        assert np.linalg.matrix_rank(P) == 1
        assert la.matrix_norm(P, 2) == pytest.approx(1.0, abs=1e-10)
        # projecting q itself leaves it unchanged
        np.testing.assert_allclose(P @ q, q, atol=1e-12)


def test_rank_one_projector_rejects_the_zero_direction():
    with pytest.raises(ValueError):
        og.rank_one_projector(np.zeros(4))


def test_is_projector_rejects_a_nonsquare_matrix(rng):
    assert not og.is_projector(rng.standard_normal((3, 5)))


def test_a_general_matrix_is_not_a_projector(rng):
    assert not og.is_projector(rng.standard_normal((5, 5)))


# ---------------------------------------------------------------- projection residual


def test_residual_is_orthogonal_to_the_subspace(rng):
    """Theorem 16.9(1), the defining property."""
    for m, k in [(8, 3), (20, 9), (50, 25)]:
        Q = og.random_orthogonal(m, rng)[:, :k]
        for _ in range(30):
            x = rng.standard_normal(m)
            r = og.projection_residual(x, Q)
            assert np.abs(Q.T @ r).max() < 1e-12


def test_pythagoras_holds_for_orthogonal_projection(rng):
    """Theorem 16.9(2)."""
    Q = og.random_orthogonal(14, rng)[:, :5]
    for _ in range(50):
        x = rng.standard_normal(14)
        Px = Q @ (Q.T @ x)
        r = og.projection_residual(x, Q)
        assert (np.linalg.norm(Px) ** 2 + np.linalg.norm(r) ** 2
                == pytest.approx(np.linalg.norm(x) ** 2, rel=1e-11))


def test_projection_is_the_closest_point_in_the_subspace(rng):
    """Theorem 16.9(3), the least squares theorem. No competitor may beat it."""
    m, k = 12, 4
    Q = og.random_orthogonal(m, rng)[:, :k]
    x = rng.standard_normal(m)
    best = np.linalg.norm(og.projection_residual(x, Q))
    for _ in range(2000):
        y = Q @ rng.standard_normal(k)
        assert np.linalg.norm(x - y) >= best - 1e-12


def test_projecting_a_vector_already_in_the_subspace_changes_nothing(rng):
    Q = og.random_orthogonal(10, rng)[:, :4]
    for _ in range(30):
        y = Q @ rng.standard_normal(4)
        np.testing.assert_allclose(og.projection_residual(y, Q),
                                   np.zeros(Q.shape[0]), atol=1e-12)


# ---------------------------------------------------------------- oblique projectors


def _closing_subspaces(eps):
    A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    B = np.array([[1.0, 0.0], [0.0, eps], [0.0, np.sqrt(1 - eps**2)]])
    return A, B


def test_oblique_projector_is_a_projector_but_not_symmetric():
    A, B = _closing_subspaces(0.3)
    P = og.oblique_projector(A, B)
    assert og.is_projector(P)
    assert not og.is_orthogonal_projector(P)


def test_oblique_projector_norm_is_one_over_cos_of_the_largest_angle():
    """Theorem 16.10, checked over five orders of magnitude."""
    for eps in [1.0, 0.5, 0.1, 0.01, 1e-3, 1e-5]:
        A, B = _closing_subspaces(eps)
        P = og.oblique_projector(A, B)
        theta = og.largest_principal_angle(A, B)
        assert la.matrix_norm(P, 2) == pytest.approx(1.0 / np.cos(theta), rel=1e-8)


def test_oblique_projector_norm_is_unbounded():
    A, B = _closing_subspaces(1e-6)
    assert la.matrix_norm(og.oblique_projector(A, B), 2) > 1e5


def test_oblique_projector_reduces_to_the_orthogonal_one_when_the_subspaces_agree():
    A, B = _closing_subspaces(1.0)
    P = og.oblique_projector(A, B)
    assert og.is_orthogonal_projector(P)
    assert la.matrix_norm(P, 2) == pytest.approx(1.0, abs=1e-10)


def test_orthogonal_projector_never_amplifies_and_oblique_can(rng):
    """The numerical consequence: the same error, two very different outcomes."""
    A, B = _closing_subspaces(1e-4)
    P_ob = og.oblique_projector(A, B)
    P_or = og.orthogonal_projector(np.linalg.qr(A)[0])
    worst_or = worst_ob = 0.0
    for _ in range(500):
        d = rng.standard_normal(3)
        d = d / np.linalg.norm(d)
        worst_or = max(worst_or, np.linalg.norm(P_or @ d))
        worst_ob = max(worst_ob, np.linalg.norm(P_ob @ d))
    assert worst_or <= 1.0 + 1e-9
    assert worst_ob > 100


# ---------------------------------------------------------------- principal angles


def test_principal_angles_of_identical_subspaces_are_zero(rng):
    A = rng.standard_normal((10, 4))
    assert np.abs(og.principal_angles(A, A.copy())).max() < 1e-7


def test_principal_angles_of_orthogonal_subspaces_are_ninety_degrees(rng):
    Q = og.random_orthogonal(10, rng)
    angles = og.principal_angles(Q[:, :3], Q[:, 3:6])
    assert np.abs(angles - np.pi / 2).max() < 1e-7


def test_principal_angles_are_sorted_ascending(rng):
    A, B = rng.standard_normal((12, 5)), rng.standard_normal((12, 5))
    angles = og.principal_angles(A, B)
    assert np.all(np.diff(angles) >= -1e-12)
    assert og.largest_principal_angle(A, B) == pytest.approx(angles[-1])


def test_principal_angles_lie_in_the_valid_range(rng):
    for _ in range(30):
        A, B = rng.standard_normal((9, 3)), rng.standard_normal((9, 4))
        angles = og.principal_angles(A, B)
        assert np.all(angles >= -1e-12) and np.all(angles <= np.pi / 2 + 1e-12)
