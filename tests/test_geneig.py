"""Tests for `nalib.geneig`.

The generalized problem has one structural fact that a standard eigensolver cannot express: a
singular ``B`` gives **infinite** eigenvalues, and the count of finite ones is then fewer than
``n``. So several tests build pencils with a known number of infinite eigenvalues and check the
count, not just the values.

`match_spectra` is used rather than sorting throughout, because sorting complex spectra is a
trap this module documents: a correct QZ result was scored as differing by 1.39 when it agreed
to 9.5e-16.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from nalib import geneig as ge


SIZES = [1, 2, 4, 10, 30]


def definite_pencil(n, seed=1, kappa_b=10.0):
    """A symmetric definite pencil with a controllable ``kappa(B)``."""
    gen = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(gen.standard_normal((n, n)))
    B = Q @ np.diag(np.geomspace(1.0, 1.0 / kappa_b, n)) @ Q.T
    A = gen.standard_normal((n, n))
    return 0.5 * (A + A.T), 0.5 * (B + B.T)


# ---------------------------------------------------------------- comparing spectra


def test_sorting_complex_spectra_is_a_trap():
    """**The bug this pins.** Two identical conjugate pairs can sort in either order when their
    real parts agree to the last bit, so a correct answer scores as wrong by the full imaginary
    spread. Measured: a QZ result agreeing to 9.5e-16 was scored 1.39 by sorting."""
    gen = np.random.default_rng(0)
    n = 10
    A = gen.standard_normal((n, n))
    B = gen.standard_normal((n, n))
    mine = ge.qz_eigenvalues(A, B).values
    ref = sla.eig(A, B, right=False)
    by_sorting = float(np.max(np.abs(np.sort_complex(mine) - np.sort_complex(ref))))
    by_matching = ge.match_spectra(mine, ref)
    assert by_matching < 1e-11, f"the QZ answer is wrong: {by_matching:.2e}"
    assert by_sorting > 1e-3, "the sorting trap did not trigger; the test proves nothing"


def test_match_spectra_is_zero_for_identical_sets():
    v = np.array([1.0 + 2.0j, 1.0 - 2.0j, -3.0])
    assert ge.match_spectra(v, v[::-1]) == 0.0


def test_match_spectra_rejects_different_counts():
    with pytest.raises(ValueError):
        ge.match_spectra([1.0, 2.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------- classifying the pencil


@pytest.mark.parametrize("n", SIZES)
def test_a_definite_pencil_is_recognised(n):
    A, B = definite_pencil(n, seed=n)
    info = ge.is_definite_pencil(A, B)
    assert info["definite_pencil"]
    assert info["smallest_B_eigenvalue"] > 0.0


def test_an_asymmetric_pencil_is_rejected(rng):
    A = rng.standard_normal((6, 6))
    B = np.eye(6)
    assert not ge.is_definite_pencil(A, B)["definite_pencil"]
    assert not ge.is_definite_pencil(A, B)["A_symmetric"]


def test_an_indefinite_b_is_rejected():
    A = np.eye(4)
    B = np.diag([1.0, 1.0, -1.0, 1.0])
    info = ge.is_definite_pencil(A, B)
    assert info["B_symmetric"] and not info["B_positive_definite"]


def test_classification_rejects_mismatched_shapes(rng):
    with pytest.raises(ValueError):
        ge.is_definite_pencil(rng.standard_normal((4, 4)), rng.standard_normal((5, 5)))
    with pytest.raises(ValueError):
        ge.is_definite_pencil(rng.standard_normal((3, 5)), rng.standard_normal((3, 5)))


# ---------------------------------------------------------------- the Cholesky reduction


@pytest.mark.parametrize("n", SIZES)
def test_the_cholesky_reduction_gets_the_eigenvalues(n):
    A, B = definite_pencil(n, seed=n)
    out = ge.cholesky_reduction(A, B)
    ref = np.sort(sla.eigh(A, B, eigvals_only=True))
    assert np.max(np.abs(np.sort(out["values"]) - ref)) < 1e-9 * max(np.abs(ref).max(), 1.0)


@pytest.mark.parametrize("n", [4, 10, 30])
def test_the_reduced_matrix_is_exactly_symmetric(n):
    """Which is the whole reason for using this reduction rather than the obvious one."""
    A, B = definite_pencil(n, seed=n)
    out = ge.cholesky_reduction(A, B)
    assert out["symmetry_error"] == 0.0


@pytest.mark.parametrize("n", [4, 10, 30])
def test_the_eigenvectors_are_B_orthogonal(n):
    """``x_i^T B x_j = delta_ij``: the natural orthogonality for this problem, and the one a
    vibration analysis wants. Plain ``X^T X = I`` is not what holds."""
    A, B = definite_pencil(n, seed=n)
    out = ge.cholesky_reduction(A, B)
    assert ge.b_orthogonality(B, out["vectors"]) < 1e-10


@pytest.mark.parametrize("n", [4, 10, 30])
def test_the_pencil_residual_is_tiny(n):
    A, B = definite_pencil(n, seed=n)
    out = ge.cholesky_reduction(A, B)
    assert ge.pencil_residual(A, B, out["values"], out["vectors"]) < 1e-12


def test_the_cholesky_reduction_refuses_an_indefinite_pencil():
    p = ge.singular_pencil(8, 3)
    with pytest.raises(ValueError, match="definite"):
        ge.cholesky_reduction(p["A"], p["B"])


def test_it_refuses_an_asymmetric_A(rng):
    with pytest.raises(ValueError, match="definite"):
        ge.cholesky_reduction(rng.standard_normal((5, 5)), np.eye(5))


# ---------------------------------------------------------------- the naive reduction


@pytest.mark.parametrize("n", [10, 30])
def test_the_naive_reduction_destroys_symmetry(n):
    A, B = definite_pencil(n, seed=n)
    out = ge.naive_reduction(A, B)
    assert out["symmetry_error"] > 1e-3 * max(np.linalg.norm(A), 1.0)


@pytest.mark.parametrize("kappa_b", [1e2, 1e6, 1e10, 1e14])
def test_the_naive_reduction_degrades_with_kappa_B_and_cholesky_does_not(kappa_b):
    """**The measured result, and it is dramatic.** At kappa(B) = 1e14 the naive route gives a
    relative error of 1.3e-4 and the Cholesky route gives 2.3e-14. The Cholesky error does not
    depend on kappa(B) at all."""
    n = 10
    A, B = definite_pencil(n, seed=3, kappa_b=kappa_b)
    ref = np.sort(sla.eigh(A, B, eigvals_only=True))
    scale = max(np.abs(ref).max(), 1.0)
    naive = np.max(np.abs(np.sort(ge.naive_reduction(A, B)["values"].real) - ref)) / scale
    chol = np.max(np.abs(np.sort(ge.cholesky_reduction(A, B)["values"]) - ref)) / scale
    assert chol < 1e-12, f"the Cholesky route should be insensitive to kappa(B): {chol:.2e}"
    if kappa_b >= 1e6:
        assert naive > 100.0 * chol, f"naive {naive:.2e} against Cholesky {chol:.2e}"


def test_the_naive_reduction_rejects_mismatched_shapes(rng):
    with pytest.raises(ValueError):
        ge.naive_reduction(rng.standard_normal((4, 4)), rng.standard_normal((5, 5)))


# ---------------------------------------------------------------- QZ


@pytest.mark.parametrize("n", SIZES)
def test_the_generalized_schur_form_is_exact(n, rng):
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    r = ge.qz_residuals(A, B)
    assert r["A_error"] < 1e-12 and r["B_error"] < 1e-12
    assert r["Q_unitary"] < 1e-12 and r["Z_unitary"] < 1e-12
    assert r["S_below"] < 1e-12 and r["T_below"] < 1e-12


@pytest.mark.parametrize("n", [2, 4, 10, 40])
def test_qz_finds_the_eigenvalues(n, rng):
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    out = ge.qz_eigenvalues(A, B)
    assert ge.match_spectra(out.values, sla.eig(A, B, right=False)) < 1e-10


@pytest.mark.parametrize("n", [4, 12])
def test_qz_agrees_with_the_cholesky_route_on_a_definite_pencil(n):
    """Two very different algorithms on the same problem must give the same answer."""
    A, B = definite_pencil(n, seed=n)
    qz_vals = np.sort(ge.qz_eigenvalues(A, B).values.real)
    chol = np.sort(ge.cholesky_reduction(A, B)["values"])
    assert np.max(np.abs(qz_vals - chol)) < 1e-9 * max(np.abs(chol).max(), 1.0)


def test_qz_rejects_mismatched_shapes(rng):
    with pytest.raises(ValueError):
        ge.qz_eigenvalues(rng.standard_normal((4, 4)), rng.standard_normal((5, 5)))
    with pytest.raises(ValueError):
        ge.qz_eigenvalues(rng.standard_normal((3, 5)), rng.standard_normal((3, 5)))


# ---------------------------------------------------------------- infinite eigenvalues


@pytest.mark.parametrize("n,k", [(6, 2), (10, 3), (20, 7), (5, 0), (8, 7)])
def test_a_singular_B_gives_exactly_that_many_infinite_eigenvalues(n, k):
    """**What a standard eigenvalue problem cannot express.** With B singular the pencil has
    infinite eigenvalues, the finite count is fewer than n, and no reduction to a standard
    problem exists."""
    p = ge.singular_pencil(n, k)
    out = ge.qz_eigenvalues(p["A"], p["B"])
    assert out.n_infinite == k, f"expected {k} infinite, got {out.n_infinite}"
    assert out.finite.size == n - k


@pytest.mark.parametrize("n,k", [(6, 2), (12, 4)])
def test_the_beta_values_are_exactly_zero_for_the_infinite_ones(n, k):
    """Which is why QZ returns alpha and beta separately: a single ratio would be inf or nan
    and would lose the information alpha carries."""
    p = ge.singular_pencil(n, k)
    out = ge.qz_eigenvalues(p["A"], p["B"])
    assert int(np.sum(np.abs(out.beta) < 1e-12 * max(np.abs(out.beta).max(), 1.0))) == k


def test_the_pencil_result_reports_finite_values_correctly():
    out = ge.PencilResult(alpha=np.array([2.0, 3.0, 1.0]), beta=np.array([1.0, 0.0, 4.0]))
    assert out.n_infinite == 1
    assert out.finite.size == 2
    assert np.allclose(np.sort(out.finite.real), [0.25, 2.0])


def test_singular_pencil_rejects_a_bad_count():
    for bad in (-1, 6):
        with pytest.raises(ValueError):
            ge.singular_pencil(6, bad)


# ---------------------------------------------------------------- the physical problem


@pytest.mark.parametrize("n", [3, 10, 40])
def test_the_vibrating_string_with_uniform_mass_is_a_standard_problem(n):
    """When M is the identity, the generalized problem collapses to the standard one, and the
    eigenvalues are the Laplacian's."""
    p = ge.vibrating_string(n)
    out = ge.cholesky_reduction(p["K"], p["M"])
    exact = 4.0 * np.sin(np.arange(1, n + 1) * np.pi / (2.0 * (n + 1))) ** 2
    assert np.max(np.abs(np.sort(out["values"]) - np.sort(exact))) < 1e-10


@pytest.mark.parametrize("n", [10, 30])
def test_varying_the_mass_changes_the_frequencies(n):
    """A heavier chain vibrates more slowly, which is a fact the eigenvalues must reproduce."""
    light = ge.vibrating_string(n)
    heavy = ge.vibrating_string(n, density=np.full(n, 4.0))
    w_light = ge.cholesky_reduction(light["K"], light["M"])["values"]
    w_heavy = ge.cholesky_reduction(heavy["K"], heavy["M"])["values"]
    assert np.allclose(w_heavy, w_light / 4.0, rtol=1e-10)


@pytest.mark.parametrize("n", [8, 25])
def test_the_mode_shapes_are_mass_orthogonal(n):
    p = ge.vibrating_string(n, density=np.geomspace(1.0, 10.0, n))
    out = ge.cholesky_reduction(p["K"], p["M"])
    assert ge.b_orthogonality(p["M"], out["vectors"]) < 1e-10


def test_the_frequencies_are_positive():
    """A stable structure has real positive squared frequencies, which is what a definite
    pencil guarantees and an indefinite one does not."""
    p = ge.vibrating_string(20, density=np.geomspace(1.0, 5.0, 20))
    values = ge.cholesky_reduction(p["K"], p["M"])["values"]
    assert np.all(values > 0.0)


def test_vibrating_string_rejects_bad_input():
    with pytest.raises(ValueError):
        ge.vibrating_string(0)
    with pytest.raises(ValueError):
        ge.vibrating_string(5, density=np.ones(3))
    with pytest.raises(ValueError):
        ge.vibrating_string(5, density=[1.0, 1.0, 0.0, 1.0, 1.0])


# ---------------------------------------------------------------- the helpers


def test_b_orthogonality_of_the_identity():
    assert ge.b_orthogonality(np.eye(4), np.eye(4)) == 0.0


def test_b_orthogonality_rejects_a_shape_mismatch():
    with pytest.raises(ValueError):
        ge.b_orthogonality(np.eye(4), np.eye(5))


def test_pencil_residual_rejects_a_count_mismatch(rng):
    with pytest.raises(ValueError):
        ge.pencil_residual(np.eye(4), np.eye(4), [1.0, 2.0], np.eye(4))


@pytest.mark.parametrize("n", [5, 15])
def test_the_pencil_residual_is_scale_fair(n):
    """Dividing by ||A|| alone would hold a large eigenvalue to an impossible standard, so the
    denominator includes |lambda| ||B||. Checked by scaling the whole pencil."""
    A, B = definite_pencil(n, seed=n)
    out = ge.cholesky_reduction(A, B)
    plain = ge.pencil_residual(A, B, out["values"], out["vectors"])
    scaled = ge.pencil_residual(1e6 * A, B, 1e6 * out["values"], out["vectors"])
    assert abs(plain - scaled) < 10.0 * max(plain, 1e-16)
