"""Tests for `nalib.svdcompute`.

The organising claim of this module is that ``A^T A`` must never be formed, so the tests measure
that directly: on a matrix whose small singular values are not protected by structure, the Gram
route reaches a relative error of 1.79 at kappa = 1e10 while the bidiagonal route reaches 3e-8.

Two things a careless suite would get wrong are pinned here. The relative accuracy advantage of
one-sided Jacobi does **not** show up on every graded matrix, and claiming it in general would be
false; and the reference for a relative accuracy test must not be another double precision SVD.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import svd as sv
from nalib import svdcompute as sc


TALL = [(2, 1), (3, 3), (5, 3), (20, 8), (50, 15), (12, 12)]


# ---------------------------------------------------------------- bidiagonalization


@pytest.mark.parametrize("m,n", TALL)
def test_bidiagonalization_preserves_the_singular_values(m, n, rng):
    """Orthogonal on both sides, so the singular values cannot move. Checkable at any size with
    no reference at all, which is what makes it a good foundation."""
    A = rng.standard_normal((m, n))
    d, e = sc.bidiagonalize(A)
    B = sc.bidiagonal_matrix(d, e)
    assert np.max(np.abs(np.linalg.svd(B, compute_uv=False)
                         - np.linalg.svd(A, compute_uv=False))) < 1e-12


@pytest.mark.parametrize("m,n", TALL)
def test_the_result_is_genuinely_bidiagonal(m, n, rng):
    A = rng.standard_normal((m, n))
    d, e = sc.bidiagonalize(A)
    assert d.size == n and e.size == max(n - 1, 0)


@pytest.mark.parametrize("m,n", [(6, 4), (20, 8), (30, 12)])
def test_bidiagonalization_with_vectors_reconstructs(m, n, rng):
    A = rng.standard_normal((m, n))
    d, e, U, Vt = sc.bidiagonalize(A, compute_uv=True)
    B = sc.bidiagonal_matrix(d, e)
    assert np.linalg.norm(U.T @ U - np.eye(n)) < 1e-11
    assert np.linalg.norm(Vt @ Vt.T - np.eye(n)) < 1e-11
    assert np.linalg.norm(A - U @ B @ Vt) < 1e-11 * max(np.linalg.norm(A), 1.0)


@pytest.mark.parametrize("m,n", [(20, 8), (60, 20)])
def test_the_storage_collapses(m, n, rng):
    """2n - 1 numbers instead of mn, which is what makes every later step cheap."""
    d, e = sc.bidiagonalize(rng.standard_normal((m, n)))
    assert d.size + e.size == 2 * n - 1
    assert d.size + e.size < m * n


def test_bidiagonalize_refuses_a_wide_matrix(rng):
    with pytest.raises(ValueError, match="transpose"):
        sc.bidiagonalize(rng.standard_normal((4, 9)))


def test_bidiagonal_matrix_rejects_a_mismatched_offdiagonal():
    with pytest.raises(ValueError):
        sc.bidiagonal_matrix([1.0, 2.0, 3.0], [1.0])


# ---------------------------------------------------------------- the sweep


@pytest.mark.parametrize("m,n", TALL)
@pytest.mark.parametrize("zero_shift", [False, True])
def test_the_sweep_finds_the_singular_values(m, n, zero_shift, rng):
    A = rng.standard_normal((m, n))
    out = sc.svd_via_bidiagonal(A, zero_shift=zero_shift)
    assert out.converged, out.message
    assert np.max(np.abs(out.s - np.linalg.svd(A, compute_uv=False))) < 1e-11


@pytest.mark.parametrize("m,n", [(4, 9), (8, 30)])
def test_a_wide_matrix_is_transposed_first(m, n, rng):
    """The singular values of A and A^T are the same, so transposing costs nothing."""
    A = rng.standard_normal((m, n))
    out = sc.svd_via_bidiagonal(A)
    assert "transposed" in out.message
    assert np.max(np.abs(out.s - np.linalg.svd(A, compute_uv=False))) < 1e-11


def test_the_sweep_handles_a_one_by_one():
    out = sc.golub_kahan_svd(np.array([3.5]), np.zeros(0))
    assert np.allclose(out.s, [3.5])


def test_the_sweep_handles_an_exactly_diagonal_matrix():
    d = np.array([5.0, 3.0, 1.0])
    out = sc.golub_kahan_svd(d, np.zeros(2))
    assert out.iterations == 0
    assert np.allclose(out.s, [5.0, 3.0, 1.0])


def test_the_sweep_handles_a_zero_singular_value():
    d = np.array([2.0, 0.0, 1.0])
    e = np.array([0.5, 0.5])
    out = sc.golub_kahan_svd(d, e)
    assert out.converged
    exact = np.linalg.svd(sc.bidiagonal_matrix(d, e), compute_uv=False)
    assert np.max(np.abs(out.s - exact)) < 1e-11


def test_the_sweep_rejects_a_mismatched_offdiagonal():
    with pytest.raises(ValueError):
        sc.golub_kahan_svd([1.0, 2.0, 3.0], [1.0])


@pytest.mark.parametrize("n", [1, 2, 5])
def test_the_shift_is_an_eigenvalue_of_the_trailing_block(n, rng):
    """Computed from B alone, never from B^T B."""
    d = np.abs(rng.standard_normal(n)) + 0.5
    e = rng.standard_normal(max(n - 1, 0))
    mu = sc.wilkinson_shift_squared(d, e)
    if n < 2:
        assert abs(mu - d[-1] ** 2) < 1e-12
        return
    B = sc.bidiagonal_matrix(d, e)
    block = (B.T @ B)[n - 2:, n - 2:]
    assert min(abs(mu - v) for v in np.linalg.eigvalsh(block)) < 1e-9


def test_the_shift_of_an_empty_matrix_is_zero():
    assert sc.wilkinson_shift_squared(np.zeros(0), np.zeros(0)) == 0.0


# ---------------------------------------------------------------- the speed difference


@pytest.mark.parametrize("m,n", [(20, 8), (50, 15)])
def test_the_shift_is_worth_a_large_factor_in_sweeps(m, n):
    """**The measurable difference between the two variants is speed, not accuracy.**
    Measured: 29 sweeps shifted against 4785 unshifted at 50x15, a factor of 165."""
    A = np.random.default_rng(1).standard_normal((m, n))
    shifted = sc.svd_via_bidiagonal(A)
    unshifted = sc.svd_via_bidiagonal(A, zero_shift=True)
    assert unshifted.iterations > 20 * shifted.iterations, \
        f"{shifted.iterations} against {unshifted.iterations}"


# ---------------------------------------------------------------- one-sided Jacobi


@pytest.mark.parametrize("m,n", TALL)
def test_one_sided_jacobi_finds_the_singular_values(m, n, rng):
    A = rng.standard_normal((m, n))
    out = sc.one_sided_jacobi(A)
    assert out.converged, out.message
    assert np.max(np.abs(out.s - np.linalg.svd(A, compute_uv=False))) < 1e-11


@pytest.mark.parametrize("m,n", [(6, 4), (20, 8), (40, 12)])
def test_one_sided_jacobi_reconstructs_the_matrix(m, n, rng):
    A = rng.standard_normal((m, n))
    out = sc.one_sided_jacobi(A, compute_uv=True)
    assert np.linalg.norm(out.U.T @ out.U - np.eye(n)) < 1e-11
    assert np.linalg.norm(out.Vt @ out.Vt.T - np.eye(n)) < 1e-11
    assert np.linalg.norm(A - (out.U * out.s) @ out.Vt) < 1e-11 * max(np.linalg.norm(A), 1.0)


@pytest.mark.parametrize("m,n", [(10, 5), (30, 10)])
def test_the_column_orthogonality_decreases_every_sweep(m, n, rng):
    A = rng.standard_normal((m, n))
    trail = sc.one_sided_jacobi(A).history
    assert all(b <= a for a, b in zip(trail, trail[1:])), trail


def test_a_matrix_with_orthogonal_columns_needs_no_rotations():
    Q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((10, 4)))
    out = sc.one_sided_jacobi(Q * np.array([4.0, 3.0, 2.0, 1.0]))
    assert out.rotations == 0
    assert np.allclose(out.s, [4.0, 3.0, 2.0, 1.0])


def test_one_sided_jacobi_refuses_a_wide_matrix(rng):
    with pytest.raises(ValueError, match="transpose"):
        sc.one_sided_jacobi(rng.standard_normal((4, 9)))


# ---------------------------------------------------------------- never form A^T A


@pytest.mark.parametrize("kappa", [1e6, 1e10, 1e14])
def test_forming_the_gram_matrix_is_catastrophic(kappa):
    """**The prohibition this module is organised around.** On a matrix whose small singular
    values are not protected by structure, the Gram route tracks ``u kappa^2`` and the
    bidiagonal route tracks ``u kappa``. Measured at kappa = 1e10: relative error 1.79 against
    3.2e-8."""
    A = sv.graded_matrix(30, 8, kappa, rng=np.random.default_rng(3))
    exact = np.linalg.svd(A, compute_uv=False)
    gram = np.sqrt(np.maximum(np.sort(np.linalg.eigvalsh(A.T @ A))[::-1], 0.0))
    bidiag = sc.svd_via_bidiagonal(A).s
    assert sc.relative_error(gram, exact) > 100.0 * sc.relative_error(bidiag, exact)


@pytest.mark.parametrize("kappa", [1e2, 1e6, 1e10])
def test_the_bidiagonal_route_tracks_u_kappa(kappa):
    A = sv.graded_matrix(30, 8, kappa, rng=np.random.default_rng(3))
    exact = np.linalg.svd(A, compute_uv=False)
    err = sc.relative_error(sc.svd_via_bidiagonal(A).s, exact)
    assert err < 1000.0 * np.finfo(float).eps * kappa, f"{err:.2e} at kappa {kappa:.0e}"


# ---------------------------------------------------------------- relative accuracy, honestly


@pytest.mark.parametrize("spread", [1e-2, 1e-5, 1e-8, 1e-11])
def test_every_method_is_accurate_on_a_column_graded_matrix(spread):
    """**And that is the honest finding, not a Jacobi win.** When ``A = B D`` with ``B`` well
    conditioned, the small singular values ARE determined to high relative accuracy by the
    entries, and every method here recovers them: measured 1e-15 for all four, up to
    kappa = 1.3e11. Claiming a Jacobi advantage in general would be false."""
    g = sc.graded_columns(30, 8, spread, rng=np.random.default_rng(2))
    A, exact = g["A"], g["exact"]
    for name, got in (("LAPACK", np.linalg.svd(A, compute_uv=False)),
                      ("bidiagonal", sc.svd_via_bidiagonal(A).s),
                      ("zero shift", sc.svd_via_bidiagonal(A, zero_shift=True,
                                                           tol=1e-15).s),
                      ("Jacobi", sc.one_sided_jacobi(A, tol=1e-15, max_sweeps=100).s)):
        assert sc.relative_error(got, exact) < 1e-13, \
            f"{name} at spread {spread}: {sc.relative_error(got, exact):.2e}"


def test_the_graded_base_is_well_conditioned():
    """Which is the hypothesis of the relative accuracy theorem. Without it the theorem says
    nothing, and the previous test would be measuring luck."""
    g = sc.graded_columns(30, 8, 1e-11, rng=np.random.default_rng(2))
    assert np.linalg.cond(g["base"]) < 10.0
    assert g["exact"][0] / g["exact"][-1] > 1e9


def test_the_reference_is_not_another_double_precision_svd():
    """The mpmath reference must disagree with LAPACK somewhere, or it IS LAPACK."""
    g = sc.graded_columns(30, 8, 1e-11, rng=np.random.default_rng(2))
    lapack = np.linalg.svd(g["A"], compute_uv=False)
    assert sc.relative_error(lapack, g["exact"]) > 0.0


def test_relative_error_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        sc.relative_error([1.0, 2.0], [1.0, 2.0, 3.0])


def test_graded_columns_refuses_a_wide_shape():
    with pytest.raises(ValueError):
        sc.graded_columns(4, 9, 1e-3)


# ---------------------------------------------------------------- the three agree


@pytest.mark.parametrize("m,n", [(10, 6), (30, 12), (60, 20)])
def test_all_three_methods_agree_with_lapack(m, n, rng):
    A = rng.standard_normal((m, n))
    reference = np.linalg.svd(A, compute_uv=False)
    answers = {"bidiagonal + shift": sc.svd_via_bidiagonal(A).s,
               "bidiagonal, zero shift": sc.svd_via_bidiagonal(A, zero_shift=True).s,
               "one-sided Jacobi": sc.one_sided_jacobi(A).s}
    for name, got in answers.items():
        assert np.max(np.abs(got - reference)) < 1e-10, f"{name} disagreed"


@pytest.mark.parametrize("m,n,r", [(10, 6, 3), (20, 8, 5)])
def test_they_agree_on_a_rank_deficient_matrix(m, n, r, rng):
    A = sv.rank_deficient(m, n, r, rng=rng)
    reference = np.linalg.svd(A, compute_uv=False)
    for got in (sc.svd_via_bidiagonal(A).s, sc.one_sided_jacobi(A).s):
        assert np.max(np.abs(got - reference)) < 1e-10
        assert int(np.sum(got > 1e-10 * max(got[0], 1.0))) == r
