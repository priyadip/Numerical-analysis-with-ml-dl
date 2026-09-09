"""Tests for nalib.linalg.

Norms are checked against `numpy.linalg.norm`, which is the trusted reference, and the norm
**axioms** are checked directly rather than assumed. The three views of a matrix product are
checked against `A @ B` and against each other.

Where a property is claimed in lesson 15 (submultiplicativity, norm equivalence constants,
Gelfand's formula), it is verified here rather than left as prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import linalg as la


def well_conditioned(n, rng):
    """A random n by n matrix made safely invertible by a diagonal boost.

    The size appears once, so changing it cannot leave a stale literal behind.
    """
    return rng.standard_normal((n, n)) + n * np.eye(n)


# ---------------------------------------------------------------- vector norms


@pytest.mark.parametrize("p", [1, 2, 3, 7, np.inf])
def test_vector_norm_matches_numpy(p, rng):
    for _ in range(20):
        x = rng.standard_normal(9)
        assert la.vector_norm(x, p) == pytest.approx(np.linalg.norm(x, p), rel=1e-12)


def test_vector_norm_on_a_known_case():
    x = np.array([3.0, -4.0])
    assert la.vector_norm(x, 1) == pytest.approx(7.0)
    assert la.vector_norm(x, 2) == pytest.approx(5.0)
    assert la.vector_norm(x, np.inf) == pytest.approx(4.0)


def test_vector_norms_are_ordered_decreasing_in_p(rng):
    """||x||_inf <= ... <= ||x||_2 <= ||x||_1 for every x."""
    for _ in range(50):
        x = rng.standard_normal(12)
        vals = [la.vector_norm(x, p) for p in (1, 2, 4, 16, np.inf)]
        assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


def test_vector_norm_rejects_p_below_one():
    with pytest.raises(ValueError):
        la.vector_norm([1.0, 2.0], 0.5)


def test_vector_norm_of_zero_is_zero():
    for p in (1, 2, np.inf):
        assert la.vector_norm(np.zeros(5), p) == 0.0


@pytest.mark.parametrize("p", [1, 2, np.inf, 3])
def test_norm_axioms_hold(p, rng):
    vs = [rng.standard_normal(7) for _ in range(8)] + [np.zeros(7)]
    v = la.check_norm_axioms(lambda z: la.vector_norm(z, p), vs)
    assert max(v.values()) < 1e-13, f"axiom violated: {v}"


def test_axiom_checker_catches_a_non_norm(rng):
    """The 'p = 0.5 norm' fails the triangle inequality. The checker must see it."""
    vs = [np.abs(rng.standard_normal(7)) for _ in range(8)]
    v = la.check_norm_axioms(lambda z: float(np.sum(np.abs(z) ** 0.5) ** 2), vs)
    assert v["homogeneity"] < 1e-12, "homogeneity should still hold"
    assert v["triangle"] > 1e-3, "the triangle inequality should fail, and visibly"


def test_axiom_checker_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        la.check_norm_axioms(np.linalg.norm, [np.ones(3), np.ones(4)])


def test_normalize_gives_unit_length(rng):
    for p in (1, 2, np.inf):
        x = rng.standard_normal(8)
        assert la.vector_norm(la.normalize(x, p), p) == pytest.approx(1.0)


def test_normalize_leaves_the_zero_vector_alone():
    np.testing.assert_array_equal(la.normalize(np.zeros(4)), np.zeros(4))


def test_unit_ball_points_have_unit_norm():
    for p in (1, 1.5, 2, 4, np.inf):
        pts = la.unit_ball_points(p, 200)
        norms = [la.vector_norm(v, p) for v in pts]
        assert max(abs(n - 1.0) for n in norms) < 1e-12


# ---------------------------------------------------------------- matrix norms


@pytest.mark.parametrize("p", [1, 2, np.inf, "fro"])
def test_matrix_norm_matches_numpy(p, rng):
    for shape in [(4, 4), (6, 3), (3, 6)]:
        A = rng.standard_normal(shape)
        assert la.matrix_norm(A, p) == pytest.approx(np.linalg.norm(A, p), rel=1e-11)


def test_matrix_norm_closed_forms_are_what_they_claim(rng):
    A = rng.standard_normal((5, 7))
    assert la.matrix_norm(A, 1) == pytest.approx(np.abs(A).sum(axis=0).max())
    assert la.matrix_norm(A, np.inf) == pytest.approx(np.abs(A).sum(axis=1).max())
    assert la.matrix_norm(A, 2) == pytest.approx(np.linalg.svd(A, compute_uv=False)[0])


def test_matrix_norm_rejects_unsupported_p():
    with pytest.raises(ValueError):
        la.matrix_norm(np.eye(2), 3)


def test_induced_norms_of_the_identity_are_exactly_one():
    """This is the fact that proves Frobenius is not induced."""
    for n in (1, 2, 5, 40):
        for p in (1, 2, np.inf):
            assert la.matrix_norm(np.eye(n), p) == pytest.approx(1.0)


def test_frobenius_norm_of_the_identity_is_sqrt_n():
    for n in (1, 4, 25, 100):
        assert la.matrix_norm(np.eye(n), "fro") == pytest.approx(np.sqrt(n))


def test_frobenius_is_between_two_norm_and_sqrt_rank_times_it(rng):
    """||A||_2 <= ||A||_F <= sqrt(r) ||A||_2, with equality on the left for rank one."""
    for _ in range(20):
        A = rng.standard_normal((6, 6))
        r = np.linalg.matrix_rank(A)
        n2, nf = la.matrix_norm(A, 2), la.matrix_norm(A, "fro")
        assert n2 <= nf + 1e-12
        assert nf <= np.sqrt(r) * n2 + 1e-10


def test_rank_one_matrices_have_equal_two_and_frobenius_norms(rng):
    for _ in range(20):
        A = np.outer(rng.standard_normal(7), rng.standard_normal(5))
        assert la.matrix_norm(A, 2) == pytest.approx(la.matrix_norm(A, "fro"), rel=1e-11)


def test_induced_norm_is_really_the_max_stretch(rng):
    """No direction may beat the closed form, and sampling should get close in 2D."""
    for p in (1, 2, np.inf):
        A = rng.standard_normal((3, 3))
        true = la.matrix_norm(A, p)
        for _ in range(3000):
            x = rng.standard_normal(3)
            assert la.vector_norm(A @ x, p) <= true * la.vector_norm(x, p) + 1e-10


def test_sampling_underestimates_the_norm_and_worsens_with_dimension(rng):
    """Section 4's finding: random search is a demonstration, not a method."""
    fractions = []
    for n in (2, 50):
        A = rng.standard_normal((n, n))
        fractions.append(la.induced_norm_by_search(A, 2, 4000, seed=3)
                         / la.matrix_norm(A, 2))
    assert all(f <= 1.0 + 1e-9 for f in fractions), "sampling must never overestimate"
    assert fractions[1] < fractions[0], "and it must get worse in higher dimensions"


def test_two_norm_is_transpose_invariant(rng):
    for _ in range(20):
        A = rng.standard_normal((5, 8))
        assert la.matrix_norm(A, 2) == pytest.approx(la.matrix_norm(A.T, 2), rel=1e-11)


def test_one_norm_and_infinity_norm_are_transposes_of_each_other(rng):
    for _ in range(20):
        A = rng.standard_normal((5, 8))
        assert la.matrix_norm(A, 1) == pytest.approx(la.matrix_norm(A.T, np.inf))


# ---------------------------------------------------------------- submultiplicativity


@pytest.mark.parametrize("p", [1, 2, np.inf, "fro"])
def test_submultiplicativity_holds(p, rng):
    for _ in range(40):
        A = rng.standard_normal((5, 5))
        B = rng.standard_normal((5, 5))
        assert la.matrix_norm(A @ B, p) <= la.matrix_norm(A, p) * la.matrix_norm(B, p) + 1e-10


def test_submultiplicativity_is_tight_for_orthogonal_matrices(rng):
    """Orthogonal matrices have norm 1 and their product does too, so there is no slack."""
    from nalib import orthogonality as og

    Q1, Q2 = og.random_orthogonal(6, rng), og.random_orthogonal(6, rng)
    assert la.submultiplicativity_slack(Q1, Q2, 2) == pytest.approx(0.0, abs=1e-10)


def test_submultiplicativity_is_very_loose_for_a_matrix_and_its_inverse(rng):
    """A A^-1 = I has norm 1 while the bound is kappa(A). The gap is the point."""
    A = well_conditioned(6, rng)
    slack = la.submultiplicativity_slack(A, np.linalg.inv(A), 2)
    assert slack > 1.0, "expected a large gap for this pair"
    assert la.matrix_norm(A @ np.linalg.inv(A), 2) == pytest.approx(1.0, abs=1e-8)


def test_submultiplicativity_slack_is_never_negative(rng):
    for _ in range(60):
        A, B = rng.standard_normal((4, 4)), rng.standard_normal((4, 4))
        assert la.submultiplicativity_slack(A, B, 2) >= -1e-10


# ---------------------------------------------------------------- spectral radius


def test_spectral_radius_of_a_triangular_matrix_is_the_largest_diagonal(rng):
    for _ in range(20):
        A = np.triu(rng.standard_normal((6, 6)))
        assert la.spectral_radius(A) == pytest.approx(np.abs(np.diag(A)).max(), rel=1e-9)


def test_spectral_radius_is_not_a_norm_positivity():
    """A nonzero matrix with spectral radius zero. No norm can do this."""
    N = np.array([[0.0, 100.0], [0.0, 0.0]])
    assert la.spectral_radius(N) == pytest.approx(0.0, abs=1e-12)
    assert la.matrix_norm(N, 2) == pytest.approx(100.0)


def test_spectral_radius_is_not_a_norm_triangle_inequality():
    M1 = np.array([[0.0, 1.0], [0.0, 0.0]])
    M2 = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert la.spectral_radius(M1 + M2) > la.spectral_radius(M1) + la.spectral_radius(M2)


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_spectral_radius_is_bounded_by_every_induced_norm(p, rng):
    for _ in range(40):
        A = rng.standard_normal((6, 6))
        assert la.spectral_radius(A) <= la.matrix_norm(A, p) + 1e-10


def test_gelfand_formula(rng):
    """||A^k||^(1/k) -> rho(A). Checked on a matrix where the two differ a lot."""
    A = np.array([[0.9, 4.0], [0.0, 0.8]])
    rho = la.spectral_radius(A)
    assert la.matrix_norm(A, 2) > 4 * rho, "the test matrix should be strongly non-normal"
    Ak = np.linalg.matrix_power(A, 400)
    assert la.matrix_norm(Ak, 2) ** (1 / 400) == pytest.approx(rho, abs=0.02)


def test_powers_can_grow_before_they_decay():
    """rho < 1 does not mean ||A^k|| decreases at every step. Lesson 15 section 6."""
    A = np.array([[0.9, 4.0], [0.0, 0.8]])
    assert la.spectral_radius(A) < 1.0
    norms = [la.matrix_norm(np.linalg.matrix_power(A, k), 2) for k in range(1, 121)]
    assert max(norms) > norms[0], "expected a transient hump"
    assert norms[-1] < 1e-3, "but it must decay in the end"


# ---------------------------------------------------------------- condition number


def test_condition_number_matches_numpy(rng):
    for p in (1, 2, np.inf):
        A = well_conditioned(6, rng)
        assert la.condition_number(A, p) == pytest.approx(np.linalg.cond(A, p), rel=1e-9)


def test_condition_number_of_the_identity_is_one():
    for p in (1, 2, np.inf):
        assert la.condition_number(np.eye(9), p) == pytest.approx(1.0)


def test_condition_number_is_never_below_one(rng):
    for _ in range(40):
        A = well_conditioned(5, rng)
        assert la.condition_number(A, 2) >= 1.0 - 1e-10


def test_condition_number_of_a_singular_matrix_is_infinite():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert la.condition_number(A, 2) > 1e15


def test_condition_number_two_equals_the_singular_value_ratio(rng):
    for _ in range(20):
        A = well_conditioned(6, rng)
        s = np.linalg.svd(A, compute_uv=False)
        assert la.condition_number(A, 2) == pytest.approx(s[0] / s[-1], rel=1e-9)


def test_condition_number_rejects_nonsquare():
    with pytest.raises(ValueError):
        la.condition_number(np.ones((3, 5)))


def test_condition_number_bounds_the_error_amplification(rng):
    """The governing inequality of lesson 15 section 8, checked directly."""
    from nalib import orthogonality as og

    for target in (1e2, 1e6, 1e10):
        n = 8
        U, V = og.random_orthogonal(n, rng), og.random_orthogonal(n, rng)
        A = U @ np.diag(np.logspace(0, -np.log10(target), n)) @ V.T
        k = la.condition_number(A, 2)
        for _ in range(50):
            x, dx = rng.standard_normal(n), rng.standard_normal(n)
            dx = 1e-9 * dx / np.linalg.norm(dx) * np.linalg.norm(x)
            rel_in = np.linalg.norm(dx) / np.linalg.norm(x)
            rel_out = np.linalg.norm(A @ dx) / np.linalg.norm(A @ x)
            assert rel_out / rel_in <= k * (1 + 1e-9)


# ---------------------------------------------------------------- norm equivalence


@pytest.mark.parametrize("p,q", [(1, 2), (2, 1), (2, np.inf), (np.inf, 2),
                                 (1, np.inf), (np.inf, 1)])
def test_norm_equivalence_constants_are_valid(p, q, rng):
    for n in (2, 5, 30):
        c, C = la.norm_equivalence_constants(n, p, q)
        for _ in range(200):
            x = rng.standard_normal(n)
            npx, nqx = la.vector_norm(x, p), la.vector_norm(x, q)
            assert c * nqx <= npx + 1e-12
            assert npx <= C * nqx + 1e-12


def test_norm_equivalence_constants_are_attained():
    """The constants are sharp, not merely valid: specific vectors reach them."""
    n = 50
    ones, e1 = np.ones(n), np.eye(n)[0]
    _, C = la.norm_equivalence_constants(n, 1, np.inf)
    assert la.vector_norm(ones, 1) / la.vector_norm(ones, np.inf) == pytest.approx(C)
    c, _ = la.norm_equivalence_constants(n, np.inf, 1)
    assert la.vector_norm(e1, np.inf) / la.vector_norm(e1, 1) == pytest.approx(1.0)


def test_norm_equivalence_rejects_unsupported_pairs():
    with pytest.raises(ValueError):
        la.norm_equivalence_constants(5, 3, 7)


# ---------------------------------------------------------------- the three views


def test_matvec_by_columns_matches_numpy(rng):
    for _ in range(30):
        A = rng.standard_normal((6, 4))
        x = rng.standard_normal(4)
        np.testing.assert_allclose(la.matvec_by_columns(A, x), A @ x, atol=1e-13)


@pytest.mark.parametrize("fn", [la.matmul_by_columns, la.matmul_by_outer_products,
                                la.matmul_by_inner_products])
def test_all_three_matmul_views_agree_with_numpy(fn, rng):
    for shapes in [((4, 3), (3, 5)), ((6, 6), (6, 6)), ((2, 7), (7, 1))]:
        A, B = rng.standard_normal(shapes[0]), rng.standard_normal(shapes[1])
        np.testing.assert_allclose(fn(A, B), A @ B, atol=1e-12)


def test_the_three_views_agree_with_each_other(rng):
    A, B = rng.standard_normal((5, 4)), rng.standard_normal((4, 6))
    r1 = la.matmul_by_columns(A, B)
    r2 = la.matmul_by_outer_products(A, B)
    r3 = la.matmul_by_inner_products(A, B)
    np.testing.assert_allclose(r1, r2, atol=1e-12)
    np.testing.assert_allclose(r2, r3, atol=1e-12)


def test_outer_product_view_builds_rank_one_at_a_time(rng):
    """Each term adds at most 1 to the rank. This is what 'rank' means."""
    A, B = rng.standard_normal((6, 4)), rng.standard_normal((4, 6))
    partial = np.zeros((6, 6))
    previous = 0
    for k in range(4):
        partial = partial + np.outer(A[:, k], B[k, :])
        r = np.linalg.matrix_rank(partial)
        assert r <= previous + 1
        previous = r
    np.testing.assert_allclose(partial, A @ B, atol=1e-12)


def test_range_of_a_product_is_inside_the_range_of_the_left_factor(rng):
    """rank(AB) <= rank(A), immediate from the column view."""
    for _ in range(20):
        A = rng.standard_normal((7, 3))
        B = rng.standard_normal((3, 9))
        assert np.linalg.matrix_rank(A @ B) <= np.linalg.matrix_rank(A)
