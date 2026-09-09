"""Tests for `nalib.svd`.

The SVD's two distinguishing claims are that it **always exists** and that its singular values
are **perfectly conditioned for any matrix**, so both are tested across shapes rather than on
one example: tall, wide, square, single row, single column, rank deficient, exactly zero.

The geometric claim is tested by brute force sampling, and the test records where that stops
working: in ten dimensions a random sample misses the true maximum by 2 percent, which is the
curse of dimensionality and the reason the algebra is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import svd as sv


SHAPES = [(1, 1), (1, 5), (5, 1), (3, 3), (3, 7), (7, 3), (20, 20), (50, 8), (8, 50)]


# ---------------------------------------------------------------- existence


@pytest.mark.parametrize("m,n", SHAPES)
def test_every_matrix_has_an_svd(m, n, rng):
    """**The claim that distinguishes it from Part 6.** No squareness, no diagonalizability, no
    conditions of any kind."""
    out = sv.svd_residuals(rng.standard_normal((m, n)))
    assert out["reconstruction"] < 1e-13
    assert out["U_orthonormal"] < 1e-13
    assert out["V_orthonormal"] < 1e-13
    assert out["non_negative"] and out["decreasing"]


@pytest.mark.parametrize("m,n", [(4, 4), (6, 3), (3, 6)])
def test_the_zero_matrix_has_one_too(m, n):
    out = sv.svd(np.zeros((m, n)))
    assert out.rank == 0
    assert np.allclose(out.s, 0.0)
    assert np.linalg.norm(out.reconstruct()) < 1e-15


@pytest.mark.parametrize("m,n,r", [(8, 5, 3), (6, 9, 4), (10, 10, 1), (7, 4, 0)])
def test_a_rank_deficient_matrix_has_one(m, n, r, rng):
    A = sv.rank_deficient(m, n, r, rng=rng)
    out = sv.svd_residuals(A)
    assert out["rank"] == r
    assert out["reconstruction"] < 1e-13 or r == 0


def test_rank_deficient_rejects_an_impossible_rank():
    for bad in (-1, 6):
        with pytest.raises(ValueError):
            sv.rank_deficient(5, 5, bad)


def test_graded_matrix_has_the_requested_condition_number():
    for kappa in (1e2, 1e6, 1e12):
        A = sv.graded_matrix(30, 10, kappa, rng=np.random.default_rng(4))
        assert abs(np.linalg.cond(A) / kappa - 1.0) < 1e-6


def test_graded_matrix_rejects_a_bad_shape():
    with pytest.raises(ValueError):
        sv.graded_matrix(0, 5, 10.0)


# ---------------------------------------------------------------- the geometry


@pytest.mark.parametrize("m,n", [(2, 2), (3, 2)])
def test_the_semi_axes_are_the_singular_values_in_low_dimensions(m, n, rng):
    """Brute force: sample the unit sphere, measure the image, and the extremes are the
    singular values. Measured agreement 1e-11 at n = 2."""
    A = rng.standard_normal((m, n))
    out = sv.semi_axes_are_singular_values(A, n_samples=200_000,
                                           rng=np.random.default_rng(1))
    assert not out["has_null_space"]
    assert abs(out["sampled_longest"] - out["sigma_max"]) < 1e-6
    assert abs(out["sampled_shortest"] - out["sigma_min"]) < 1e-6


@pytest.mark.parametrize("m,n", [(2, 3), (3, 7), (8, 50)])
def test_a_wide_matrix_has_a_minimum_stretch_of_zero(m, n, rng):
    """**Not sigma_min.** With n - m null directions the sphere maps onto the SOLID ellipsoid,
    so the image contains points arbitrarily close to the origin. An earlier version of this
    test asserted otherwise and was wrong by as much as 3.9."""
    A = rng.standard_normal((m, n))
    out = sv.semi_axes_are_singular_values(A, n_samples=200_000,
                                           rng=np.random.default_rng(1))
    assert out["has_null_space"]
    assert out["min_stretch_theory"] == 0.0
    assert out["sampled_shortest"] < out["sigma_min"], \
        "a wide matrix should reach below its smallest singular value"


@pytest.mark.parametrize("m,n", SHAPES)
def test_the_sampled_extremes_never_beat_the_true_ones(m, n, rng):
    """**The one-sidedness is the check that the sampling is honest.** A finite sample cannot
    exceed the true maximum or undercut the true minimum, at any dimension."""
    A = rng.standard_normal((m, n))
    out = sv.semi_axes_are_singular_values(A, n_samples=20_000,
                                           rng=np.random.default_rng(1))
    assert out["max_gap"] >= -1e-12, "the sample beat sigma_max, which is impossible"
    assert out["min_gap"] >= -1e-12, "the sample undercut sigma_min, which is impossible"


def test_brute_force_stops_working_in_high_dimensions():
    """**The curse of dimensionality, and the reason the algebra is needed.** At n = 2 the
    sample finds sigma_1 to 1e-11; at n = 10 it misses by 0.10 and misses sigma_min by 0.52,
    because a random direction in ten dimensions is nowhere near any particular one."""
    gen = np.random.default_rng(0)
    small = sv.semi_axes_are_singular_values(gen.standard_normal((2, 2)),
                                             n_samples=200_000,
                                             rng=np.random.default_rng(1))
    large = sv.semi_axes_are_singular_values(gen.standard_normal((10, 10)),
                                             n_samples=200_000,
                                             rng=np.random.default_rng(1))
    assert small["max_gap"] < 1e-8
    assert large["max_gap"] > 1e-3, f"the curse did not bite: {large['max_gap']:.2e}"


@pytest.mark.parametrize("m,n", [(4, 4), (7, 3)])
def test_the_stretch_is_maximal_along_the_first_right_singular_vector(m, n, rng):
    A = rng.standard_normal((m, n))
    out = sv.svd(A)
    assert abs(sv.stretch_along(A, out.Vt[0]) - out.s[0]) < 1e-12
    assert abs(sv.stretch_along(A, out.Vt[-1]) - out.s[-1]) < 1e-12


def test_stretch_rejects_bad_input(rng):
    A = rng.standard_normal((4, 3))
    with pytest.raises(ValueError):
        sv.stretch_along(A, np.ones(4))
    with pytest.raises(ValueError):
        sv.stretch_along(A, np.zeros(3))


@pytest.mark.parametrize("n", [2, 5])
def test_the_hyperellipse_image_has_the_right_extremes(n, rng):
    A = rng.standard_normal((3, n))
    out = sv.hyperellipse(A, n_points=2000)
    assert out["longest"] <= out["semi_axes"][0] * (1.0 + 1e-9)


# ---------------------------------------------------------------- uniqueness


@pytest.mark.parametrize("m,n", [(6, 4), (10, 10), (4, 9)])
def test_the_singular_values_are_unique(m, n, rng):
    A = rng.standard_normal((m, n))
    out = sv.uniqueness_report(A)
    assert out["values_agree"] < 1e-12


@pytest.mark.parametrize("m,n", [(6, 4), (10, 6)])
def test_the_vectors_are_unique_up_to_sign_when_the_values_are_distinct(m, n, rng):
    A = rng.standard_normal((m, n))
    out = sv.uniqueness_report(A)
    assert out["repeated_values"] == 0
    assert out["vectors_agree_up_to_sign"] < 1e-10


def test_a_repeated_singular_value_has_a_whole_subspace_of_vectors():
    """So the vectors are NOT unique even up to sign, and any orthonormal basis of the subspace
    is equally correct."""
    A = np.diag([3.0, 3.0, 1.0, 1.0])
    out = sv.uniqueness_report(A)
    assert out["repeated_values"] == 2
    assert out["vectors_agree_up_to_sign"] > 0.5, "the vectors happened to agree; try again"


# ---------------------------------------------------------------- the eigenvalue routes


@pytest.mark.parametrize("m,n", [(6, 6), (10, 4), (4, 10)])
def test_the_jordan_wielandt_matrix_is_symmetric_with_the_right_spectrum(m, n, rng):
    A = rng.standard_normal((m, n))
    J = sv.jordan_wielandt(A)
    assert J.shape == (m + n, m + n)
    assert np.linalg.norm(J - J.T) < 1e-14
    w = np.sort(np.linalg.eigvalsh(J))
    s = np.linalg.svd(A, compute_uv=False)
    k = min(m, n)
    assert np.max(np.abs(np.sort(w)[::-1][:k] - s)) < 1e-11
    assert abs(int(np.sum(np.abs(w) < 1e-10)) - abs(m - n)) <= 1


@pytest.mark.parametrize("route", ["gram", "jordan"])
@pytest.mark.parametrize("m,n", [(8, 5), (20, 8)])
def test_both_routes_agree_on_a_well_conditioned_matrix(route, m, n, rng):
    A = rng.standard_normal((m, n))
    got = sv.singular_values_from_eigen(A, route=route)
    assert np.max(np.abs(got - np.linalg.svd(A, compute_uv=False))) < 1e-10


@pytest.mark.parametrize("kappa", [1e6, 1e10, 1e14])
def test_the_gram_route_squares_the_condition_number_and_jordan_does_not(kappa):
    """**Lesson 29's warning, measured on the SVD.** At kappa = 1e10 the Gram route's relative
    error reaches 1.00, meaning the smallest singular value has no correct digits at all, while
    Jordan-Wielandt still has seven. This is why lesson 42 never forms A^T A."""
    A = sv.graded_matrix(40, 8, kappa, rng=np.random.default_rng(3))
    exact = np.linalg.svd(A, compute_uv=False)
    rel = lambda v: float(np.max(np.abs(v - exact) / np.maximum(exact, 1e-300)))
    gram = rel(sv.singular_values_from_eigen(A, "gram"))
    jordan = rel(sv.singular_values_from_eigen(A, "jordan"))
    assert gram > 100.0 * jordan, f"gram {gram:.2e}, jordan {jordan:.2e}"


def test_the_jordan_route_takes_the_top_eigenvalues_not_the_largest_moduli():
    """**The bug this pins.** The spectrum is +sigma_1..+sigma_k, zeros, -sigma_k..-sigma_1, so
    the k largest by MODULUS are the plus and minus pair of the top k/2. That returned plausible
    numbers that were wrong by a factor of 1e9 at high condition numbers."""
    A = sv.graded_matrix(12, 4, 1e3, rng=np.random.default_rng(5))
    w = np.linalg.eigvalsh(sv.jordan_wielandt(A))
    by_modulus = np.sort(np.abs(w))[::-1][:4]
    correct = sv.singular_values_from_eigen(A, "jordan")
    exact = np.linalg.svd(A, compute_uv=False)
    assert np.max(np.abs(correct - exact)) < 1e-10
    assert np.max(np.abs(by_modulus - exact)) > 1e-3, "the wrong extraction was not wrong"


def test_singular_values_from_eigen_rejects_a_bad_route(rng):
    with pytest.raises(ValueError):
        sv.singular_values_from_eigen(rng.standard_normal((4, 3)), route="lanczos")


# ---------------------------------------------------------------- the four subspaces


@pytest.mark.parametrize("m,n,r", [(8, 5, 3), (6, 9, 4), (10, 10, 10), (7, 4, 0), (5, 5, 5)])
def test_the_four_subspaces_have_the_right_dimensions(m, n, r, rng):
    A = sv.rank_deficient(m, n, r, rng=rng)
    sub = sv.fundamental_subspaces(A)
    assert sub["rank"] == r
    assert sub["dimensions"] == {"range_A": r, "null_AT": m - r,
                                 "range_AT": r, "null_A": n - r}


@pytest.mark.parametrize("m,n,r", [(8, 5, 3), (6, 9, 4), (12, 7, 5)])
def test_every_subspace_claim_holds(m, n, r, rng):
    A = sv.rank_deficient(m, n, r, rng=rng)
    out = sv.subspace_residuals(A)
    assert out["range_A_orthonormal"] < 1e-12
    assert out["null_A_orthonormal"] < 1e-12
    assert out["range_A_perp_null_AT"] < 1e-12
    assert out["range_AT_perp_null_A"] < 1e-12
    assert out["A_kills_null_A"] < 1e-12
    assert out["AT_kills_null_AT"] < 1e-12


@pytest.mark.parametrize("m,n,r", [(8, 5, 3), (6, 9, 4), (10, 4, 4)])
def test_rank_plus_nullity_is_the_number_of_columns(m, n, r, rng):
    A = sv.rank_deficient(m, n, r, rng=rng)
    assert sv.subspace_residuals(A)["rank_nullity"] == n


@pytest.mark.parametrize("m,n,r", [(8, 5, 3), (6, 9, 4)])
def test_the_range_basis_really_spans_the_range(m, n, r, rng):
    """Projecting A onto the claimed range must leave it unchanged."""
    A = sv.rank_deficient(m, n, r, rng=rng)
    U_r = sv.fundamental_subspaces(A)["range_A"]
    assert np.linalg.norm(A - U_r @ (U_r.T @ A)) < 1e-11 * max(np.linalg.norm(A), 1.0)


# ---------------------------------------------------------------- perturbation


@pytest.mark.parametrize("m,n", SHAPES)
def test_weyl_holds_for_every_shape(m, n, rng):
    """**Stronger than anything in lesson 35.** There the same statement needed both matrices
    symmetric; here it needs nothing at all."""
    A = rng.standard_normal((m, n))
    E = rng.standard_normal((m, n))
    E = E * (1e-6 / max(np.linalg.norm(E, 2), 1e-300))
    out = sv.weyl_singular(A, E)
    assert out["ratio"] <= 1.0 + 1e-9, f"ratio {out['ratio']:.6f} exceeded 1"


@pytest.mark.parametrize("m,n", [(10, 10), (30, 12)])
def test_weyl_is_not_attained_by_a_random_perturbation(m, n, rng):
    A = rng.standard_normal((m, n))
    E = rng.standard_normal((m, n))
    E = E * (1e-6 / np.linalg.norm(E, 2))
    assert sv.weyl_singular(A, E)["ratio"] < 0.95


def test_weyl_is_attained_by_a_rank_one_perturbation(rng):
    """The bound is sharp, not merely true, and the worst case is easy to build."""
    A = rng.standard_normal((10, 6))
    out = sv.svd(A)
    E = 1e-6 * np.outer(out.U[:, 0], out.Vt[0])
    assert sv.weyl_singular(A, E)["ratio"] > 0.999


def test_weyl_holds_even_for_a_rank_deficient_matrix(rng):
    A = sv.rank_deficient(9, 6, 3, rng=rng)
    E = rng.standard_normal((9, 6))
    E = E * (1e-6 / np.linalg.norm(E, 2))
    assert sv.weyl_singular(A, E)["ratio"] <= 1.0 + 1e-9


def test_weyl_rejects_a_shape_mismatch(rng):
    with pytest.raises(ValueError):
        sv.weyl_singular(rng.standard_normal((4, 3)), rng.standard_normal((3, 4)))


# ---------------------------------------------------------------- conditioning


@pytest.mark.parametrize("kappa", [1e2, 1e6, 1e12])
def test_kappa_is_the_ratio_of_the_extreme_singular_values(kappa):
    """Lesson 15 defined this without saying what a singular value was. The debt is now paid."""
    A = sv.graded_matrix(30, 10, kappa, rng=np.random.default_rng(4))
    out = sv.condition_from_singular_values(A)
    assert abs(out["kappa_2"] / np.linalg.cond(A) - 1.0) < 1e-9
    assert out["agrees_with_library"] < 1e-10


@pytest.mark.parametrize("m,n", [(6, 6), (12, 5)])
def test_the_two_norm_is_the_largest_singular_value(m, n, rng):
    A = rng.standard_normal((m, n))
    out = sv.condition_from_singular_values(A)
    assert abs(out["norm_A"] - np.linalg.norm(A, 2)) < 1e-12


def test_a_singular_matrix_has_infinite_condition_number():
    A = sv.rank_deficient(6, 6, 3, rng=np.random.default_rng(2))
    A = A - A                                    # exactly zero, so sigma_min is exactly zero
    assert np.isinf(sv.condition_from_singular_values(A)["kappa_2"])


# ---------------------------------------------------------------- truncation


@pytest.mark.parametrize("m,n,k", [(10, 6, 0), (10, 6, 3), (10, 6, 6), (5, 9, 4)])
def test_truncation_has_the_right_rank_and_error(m, n, k, rng):
    A = rng.standard_normal((m, n))
    out = sv.svd(A)
    Ak = out.truncated(k)
    assert np.linalg.matrix_rank(Ak, tol=1e-10 * max(out.s[0], 1.0)) == k
    # Eckart-Young, proved in lesson 43: the error is exactly the next singular value.
    expected = out.s[k] if k < out.s.size else 0.0
    assert abs(np.linalg.norm(A - Ak, 2) - expected) < 1e-10 * max(out.s[0], 1.0)


def test_truncation_rejects_an_impossible_rank(rng):
    out = sv.svd(rng.standard_normal((8, 5)))
    for bad in (-1, 6):
        with pytest.raises(ValueError):
            out.truncated(bad)
