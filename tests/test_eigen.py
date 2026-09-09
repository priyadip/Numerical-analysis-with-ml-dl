"""Tests for `nalib.eigen`.

Localization theorems are **containment** claims, so the tests check containment at many shapes
and matrix families rather than checking one example. They also check that the bounds are
sometimes loose, because a test suite that only demonstrates a bound holding would leave the
impression that it is sharp.

The conditioning tests are the heart of it. A symmetric matrix must come out with every
condition number exactly 1, and a non-normal one must not, and Bauer-Fike must sit above what
actually happens without being confused for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import eigen


SIZES = [1, 2, 3, 5, 8, 20]


def symmetrise(M):
    return M + M.T


def strongly_non_normal(n, strength=8.0):
    """Distinct eigenvalues on the diagonal and a large strictly upper triangle, so the
    eigenvectors are nearly parallel."""
    return np.triu(np.ones((n, n)), 1) * strength + np.diag(np.arange(1.0, n + 1))


# ---------------------------------------------------------------- Gerschgorin


@pytest.mark.parametrize("n", SIZES)
def test_gerschgorin_contains_every_eigenvalue(n, rng):
    A = rng.standard_normal((n, n))
    discs = eigen.gerschgorin_discs(A)
    for lam in np.linalg.eigvals(A):
        assert any(d.contains(lam) for d in discs), f"{lam} escaped every disc"


@pytest.mark.parametrize("n", SIZES)
def test_gerschgorin_by_columns_also_contains_them(n, rng):
    """A and A^T have the same eigenvalues, so the column discs are equally valid. Two
    independent regions for the price of one transpose."""
    A = rng.standard_normal((n, n))
    discs = eigen.gerschgorin_discs(A, by_columns=True)
    for lam in np.linalg.eigvals(A):
        assert any(d.contains(lam) for d in discs)


@pytest.mark.parametrize("n", [3, 6, 12])
def test_the_two_disc_sets_differ(n, rng):
    """If they were always the same there would be no point computing both."""
    A = rng.standard_normal((n, n))
    rows = sorted(d.radius for d in eigen.gerschgorin_discs(A))
    cols = sorted(d.radius for d in eigen.gerschgorin_discs(A, by_columns=True))
    assert not np.allclose(rows, cols), "the row and column radii coincided"


@pytest.mark.parametrize("n", [2, 5, 10])
def test_a_diagonal_matrix_has_radius_zero_discs(n, rng):
    """Then Gerschgorin is exact: each disc is a point and it is an eigenvalue."""
    d = rng.standard_normal(n)
    discs = eigen.gerschgorin_discs(np.diag(d))
    assert all(disc.radius == 0.0 for disc in discs)
    assert np.allclose(sorted(complex(disc.centre).real for disc in discs), sorted(d))


@pytest.mark.parametrize("n", [4, 9])
def test_gerschgorin_is_loose_when_the_matrix_is_not_diagonally_dominant(n, rng):
    """The bound holds and says very little. Measured on random matrices: the union's area
    bound is far larger than the spread of the eigenvalues themselves."""
    A = rng.standard_normal((n, n))
    report = eigen.localization_report(A)
    assert report["rows_contain_all"]
    assert report["max_row_radius"] > 0.3 * report["eigenvalue_spread"], \
        "this matrix was too close to diagonal to make the point"


def test_gerschgorin_rejects_a_non_square_matrix(rng):
    with pytest.raises(ValueError):
        eigen.gerschgorin_discs(rng.standard_normal((3, 5)))


# ---------------------------------------------------------------- Brauer


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_brauer_ovals_contain_every_eigenvalue(n, rng):
    A = rng.standard_normal((n, n))
    ovals = eigen.brauer_ovals(A)
    for lam in np.linalg.eigvals(A):
        assert any(eigen.in_brauer_oval(lam, o) for o in ovals), f"{lam} escaped every oval"


@pytest.mark.parametrize("n", [3, 5, 8])
def test_brauer_is_contained_in_gerschgorin(n, rng):
    """The theorem says the Brauer union is inside the Gerschgorin union. Checked by sampling:
    every point in some oval must also be in some disc."""
    A = rng.standard_normal((n, n))
    discs = eigen.gerschgorin_discs(A)
    ovals = eigen.brauer_ovals(A)
    gen = np.random.default_rng(11)
    centres = np.array([d.centre for d in discs])
    span = max(float(np.max(np.abs(centres))), 1.0) + max(d.radius for d in discs)
    tested = 0
    for _ in range(4000):
        z = complex(gen.uniform(-span, span), gen.uniform(-span, span))
        if any(eigen.in_brauer_oval(z, o) for o in ovals):
            tested += 1
            assert any(d.contains(z, tol=1e-7) for d in discs), \
                f"{z} is in a Brauer oval but no Gerschgorin disc"
    assert tested > 20, f"only {tested} sample points landed in an oval; the test proved little"


def test_brauer_has_no_ovals_for_a_one_by_one_matrix():
    assert eigen.brauer_ovals([[3.0]]) == []


def test_localization_report_falls_back_for_a_single_eigenvalue():
    r = eigen.localization_report([[2.5]])
    assert r["rows_contain_all"] and r["ovals_contain_all"]
    assert r["eigenvalue_spread"] == 0.0


# ---------------------------------------------------------------- disjoint discs


@pytest.mark.parametrize("n", [3, 5, 8])
def test_a_disjoint_disc_isolates_exactly_one_eigenvalue(n, rng):
    """Widely separated diagonal entries with small off-diagonals: every disc is disjoint, so
    each contains exactly one eigenvalue and the radius is a true error bound."""
    A = np.diag(np.arange(n, dtype=float) * 100.0) + 0.5 * rng.standard_normal((n, n))
    np.fill_diagonal(A, np.arange(n, dtype=float) * 100.0)
    assert eigen.disjoint_disc_count(A) == n
    discs = eigen.gerschgorin_discs(A)
    vals = np.linalg.eigvals(A)
    for d in discs:
        inside = [v for v in vals if d.contains(v)]
        assert len(inside) == 1, f"disc {d.index} caught {len(inside)} eigenvalues"


def test_overlapping_discs_are_not_counted_as_disjoint(rng):
    A = np.ones((4, 4)) + np.diag([1.0, 1.1, 1.2, 1.3])
    assert eigen.disjoint_disc_count(A) == 0


# ---------------------------------------------------------------- conditioning


@pytest.mark.parametrize("n", [2, 4, 9, 20])
def test_symmetric_eigenvalues_have_condition_number_one(n, rng):
    """Left and right eigenvectors coincide, so y^H x = 1 exactly. This is the fact that makes
    the symmetric problem a separate and much easier subject."""
    A = symmetrise(rng.standard_normal((n, n)))
    conds = eigen.eigenvalue_condition_numbers(A)["condition_numbers"]
    assert np.allclose(conds, 1.0, atol=1e-8), f"worst was {conds.max():.6f}"


@pytest.mark.parametrize("n", [4, 8, 12])
def test_a_non_normal_matrix_has_condition_numbers_above_one(n):
    A = strongly_non_normal(n)
    conds = eigen.eigenvalue_condition_numbers(A)["condition_numbers"]
    assert conds.max() > 100.0, f"worst was only {conds.max():.2f}"


@pytest.mark.parametrize("n", [6, 10])
def test_the_condition_numbers_differ_from_each_other(n):
    """**The point of computing them individually.** A single kappa(V) cannot say that one
    eigenvalue is well determined and another is not. Measured at n=6: the individual numbers
    span a factor of 33 on the same matrix."""
    conds = eigen.eigenvalue_condition_numbers(strongly_non_normal(n))["condition_numbers"]
    assert conds.max() / conds.min() > 5.0, \
        f"they only spanned a factor of {conds.max() / conds.min():.2f}"


@pytest.mark.parametrize("n", [4, 8])
def test_bauer_fike_is_an_upper_bound(n, rng):
    A = strongly_non_normal(n)
    E = rng.standard_normal((n, n))
    E = E * (1e-8 / np.linalg.norm(E, 2))
    out = eigen.bauer_fike_bound(A, E)
    assert out["actual"] <= out["bound"] * (1.0 + 1e-9), \
        f"actual {out['actual']:.3e} exceeded the bound {out['bound']:.3e}"


@pytest.mark.parametrize("n", [4, 8])
def test_bauer_fike_overstates_a_random_perturbation(n, rng):
    """As everywhere else in this course: a bound is a worst case over directions, and a random
    direction is not the worst one. Measured overstatement 2.9x to 22x."""
    A = strongly_non_normal(n)
    E = rng.standard_normal((n, n))
    E = E * (1e-8 / np.linalg.norm(E, 2))
    out = eigen.bauer_fike_bound(A, E)
    assert out["overstatement"] > 1.5, f"only {out['overstatement']:.2f}x"


@pytest.mark.parametrize("n", [3, 6, 15])
def test_kappa_v_is_one_for_a_symmetric_matrix(n, rng):
    A = symmetrise(rng.standard_normal((n, n)))
    assert abs(eigen.bauer_fike_bound(A)["kappa_V"] - 1.0) < 1e-8


def test_bauer_fike_rejects_a_mismatched_perturbation(rng):
    with pytest.raises(ValueError):
        eigen.bauer_fike_bound(rng.standard_normal((4, 4)), rng.standard_normal((5, 5)))


# ---------------------------------------------------------------- Weyl


@pytest.mark.parametrize("n", [2, 5, 12, 30])
def test_weyl_holds_for_symmetric_matrices(n, rng):
    A = symmetrise(rng.standard_normal((n, n)))
    E = symmetrise(rng.standard_normal((n, n)))
    E = E * (1e-6 / np.linalg.norm(E, 2))
    out = eigen.symmetric_eigenvalues_are_perfectly_conditioned(A, E)
    assert out["ratio"] <= 1.0 + 1e-9, f"ratio {out['ratio']:.6f} exceeded 1"


@pytest.mark.parametrize("n", [10, 30])
def test_weyl_is_not_attained_by_a_random_perturbation(n, rng):
    """Attained only when E acts along a single eigenvector. Measured ratio falls with n:
    0.71 at n=4, 0.56 at n=10, 0.32 at n=30."""
    A = symmetrise(rng.standard_normal((n, n)))
    E = symmetrise(rng.standard_normal((n, n)))
    E = E * (1e-6 / np.linalg.norm(E, 2))
    assert eigen.symmetric_eigenvalues_are_perfectly_conditioned(A, E)["ratio"] < 0.95


def test_weyl_is_attained_by_a_rank_one_perturbation_along_an_eigenvector(rng):
    """The worst case exists and is easy to build, which is what makes the bound sharp rather
    than merely true."""
    n = 8
    A = symmetrise(rng.standard_normal((n, n)))
    _, V = np.linalg.eigh(A)
    v = V[:, 0]
    E = 1e-6 * np.outer(v, v)
    out = eigen.symmetric_eigenvalues_are_perfectly_conditioned(A, E)
    assert out["ratio"] > 0.999, f"ratio only {out['ratio']:.6f}"


def test_weyl_refuses_a_non_symmetric_argument(rng):
    A = symmetrise(rng.standard_normal((5, 5)))
    with pytest.raises(ValueError):
        eigen.symmetric_eigenvalues_are_perfectly_conditioned(rng.standard_normal((5, 5)),
                                                              A * 1e-6)
    with pytest.raises(ValueError):
        eigen.symmetric_eigenvalues_are_perfectly_conditioned(A, rng.standard_normal((5, 5)))


# ---------------------------------------------------------------- defective matrices


@pytest.mark.parametrize("size", [2, 3, 5, 10])
def test_a_jordan_block_has_one_eigenvector(size):
    J = eigen.defective_matrix(size)
    if size == 1:
        pytest.skip("a 1x1 block is not defective")
    assert np.linalg.matrix_rank(J - np.eye(size)) == size - 1


@pytest.mark.parametrize("size", [2, 4, 10])
@pytest.mark.parametrize("eps", [1e-10, 1e-14])
def test_the_jordan_spread_follows_the_nth_root(size, eps):
    """**Not proportional to eps.** The characteristic polynomial is (lambda - a)^n - eps, so
    the eigenvalues move by eps^(1/n). Measured ratio 1.0000 at every size and eps tested.

    The tolerance cannot be tighter than about 1e-4. Roundoff perturbs the corner by around u
    on top of eps, so the effective perturbation is eps*(1 + u/eps), and the n-th root turns
    a relative error of u/eps into about u/(n*eps). At eps = 1e-14 that is 1e-2/n, so the
    measured agreement of 1.2e-6 is already far better than the arithmetic promises."""
    out = eigen.jordan_perturbation_spread(size, eps)
    assert abs(out["ratio"] - 1.0) < 1e-4, \
        f"moved {out['moved']:.4e} against a predicted {out['predicted']:.4e}"


def test_the_jordan_amplification_grows_with_the_block_size():
    """At n=10 and eps=1e-14 a perturbation of 1e-14 moves the eigenvalues by 0.04, an
    amplification of 4e12, with no ill conditioned matrix anywhere in sight."""
    small = eigen.jordan_perturbation_spread(2, 1e-14)["amplification"]
    large = eigen.jordan_perturbation_spread(10, 1e-14)["amplification"]
    assert large > small * 1e4, f"{small:.2e} against {large:.2e}"


def test_bauer_fike_is_vacuous_for_a_defective_matrix():
    """kappa(V) is astronomically large or infinite, so the bound says nothing. That is
    correct: the true movement is eps^(1/n), which no linear bound can describe."""
    J = eigen.defective_matrix(6)
    assert eigen.bauer_fike_bound(J)["kappa_V"] > 1e10


def test_defective_matrix_rejects_a_bad_size():
    with pytest.raises(ValueError):
        eigen.defective_matrix(0)


# ---------------------------------------------------------------- Schur


@pytest.mark.parametrize("n", [1, 2, 5, 12, 30])
def test_schur_reconstructs_any_matrix(n, rng):
    out = eigen.schur_residuals(rng.standard_normal((n, n)))
    assert out["unitary_error"] < 1e-12
    assert out["reconstruction_error"] < 1e-12
    assert out["below_diagonal"] < 1e-12


@pytest.mark.parametrize("size", [3, 6, 10])
def test_schur_exists_for_a_defective_matrix(size):
    """**The reason it is the target of every algorithm.** An eigendecomposition needs n
    independent eigenvectors; a Jordan block has one; the Schur form needs nothing."""
    out = eigen.schur_residuals(eigen.defective_matrix(size))
    assert out["reconstruction_error"] < 1e-12
    assert out["unitary_error"] < 1e-12


@pytest.mark.parametrize("n", [4, 9])
def test_the_schur_diagonal_holds_the_eigenvalues(n, rng):
    A = rng.standard_normal((n, n))
    diag = np.sort_complex(eigen.schur_residuals(A)["diagonal"])
    vals = np.sort_complex(np.linalg.eigvals(A))
    assert np.max(np.abs(diag - vals)) < 1e-10


def test_schur_rejects_a_non_square_matrix(rng):
    with pytest.raises(ValueError):
        eigen.schur_residuals(rng.standard_normal((4, 7)))


# ---------------------------------------------------------------- similarity


@pytest.mark.parametrize("n", [3, 6, 12])
def test_similarity_preserves_eigenvalues(n, rng):
    A = rng.standard_normal((n, n))
    S = rng.standard_normal((n, n))
    out = eigen.similarity_preserves_eigenvalues(A, S)
    assert out["max_eigenvalue_shift"] < 1e-8, f"{out['max_eigenvalue_shift']:.2e}"


@pytest.mark.parametrize("n", [5, 10])
def test_an_orthogonal_similarity_leaves_the_conditioning_alone(n, rng):
    """Which is exactly why algorithms use only orthogonal similarities."""
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    out = eigen.similarity_preserves_eigenvalues(A, Q)
    assert abs(out["kappa_S"] - 1.0) < 1e-8
    assert abs(out["kappa_B"] / out["kappa_A"] - 1.0) < 1e-6


@pytest.mark.parametrize("n", [5, 10])
def test_a_badly_conditioned_similarity_changes_the_numerical_problem(n, rng):
    """Same eigenvalues, very different matrix. This is the trap the orthogonality restriction
    exists to avoid."""
    A = rng.standard_normal((n, n))
    S = np.diag(np.geomspace(1.0, 1e8, n))
    out = eigen.similarity_preserves_eigenvalues(A, S)
    assert out["max_eigenvalue_shift"] < 1e-6
    assert out["kappa_B"] > 100.0 * out["kappa_A"], \
        f"kappa went {out['kappa_A']:.2e} -> {out['kappa_B']:.2e}"


def test_similarity_rejects_a_shape_mismatch(rng):
    with pytest.raises(ValueError):
        eigen.similarity_preserves_eigenvalues(rng.standard_normal((4, 4)),
                                               rng.standard_normal((5, 5)))


# ---------------------------------------------------------------- no finite algorithm


@pytest.mark.parametrize("degree", [1, 2, 3, 5, 8, 12])
def test_the_companion_matrix_has_the_polynomial_roots_as_eigenvalues(degree):
    """**The reduction that proves no finite eigenvalue algorithm exists.** Measured: exact at
    degree 2, and agreeing to 1.7e-13 at degree 12."""
    out = eigen.no_finite_algorithm(degree, np.random.default_rng(1))
    assert out["max_difference"] < 1e-9, f"degree {degree}: {out['max_difference']:.2e}"
    assert out["max_imaginary_part"] < 1e-9


def test_a_companion_matrix_of_a_known_quadratic():
    """t^2 - 3t + 2 has roots 1 and 2."""
    C = eigen.companion_matrix([2.0, -3.0])
    assert np.allclose(np.sort(np.linalg.eigvals(C).real), [1.0, 2.0])


def test_the_companion_conditioning_grows_with_the_degree():
    """Lesson 06's separate warning: this is a proof device, not a root finder."""
    low = eigen.no_finite_algorithm(3, np.random.default_rng(1))["kappa_companion"]
    high = eigen.no_finite_algorithm(12, np.random.default_rng(1))["kappa_companion"]
    assert high > 100.0 * low, f"{low:.2e} against {high:.2e}"


@pytest.mark.parametrize("degree,expected", [(1, True), (4, True), (5, False), (9, False)])
def test_the_radical_solvability_boundary_is_at_degree_five(degree, expected):
    assert eigen.no_finite_algorithm(degree,
                                     np.random.default_rng(2))["solvable_in_radicals"] is expected


def test_companion_matrix_rejects_empty_coefficients():
    with pytest.raises(ValueError):
        eigen.companion_matrix([])


def test_no_finite_algorithm_rejects_a_bad_degree():
    with pytest.raises(ValueError):
        eigen.no_finite_algorithm(0)
