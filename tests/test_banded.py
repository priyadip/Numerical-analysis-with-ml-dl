"""Tests for `nalib.banded`.

Structure is the subject, so the tests vary the structure as well as the size: bandwidths from
0 (diagonal) up to n-1 (dense), tridiagonal systems from 1 unknown upward, and sparse storage
on matrices whose density ranges from a few percent to full. A routine that only handles the
tridiagonal case is easy to write by accident, and these tests are arranged to catch that.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import banded as bd


SIZES = [1, 2, 3, 5, 9, 16, 40, 75]
SCALES = [1e-8, 1e-2, 1.0, 1e5, 1e9]


# ---------------------------------------------------------------- bandwidths


@pytest.mark.parametrize("n", SIZES)
def test_diagonal_has_zero_bandwidth(n):
    assert bd.bandwidths(np.diag(np.arange(1.0, n + 1))) == (0, 0)


@pytest.mark.parametrize("n", [2, 5, 12, 40])
@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_bandwidths_recovers_what_was_built(n, p, rng):
    p = min(p, n - 1)
    A = np.zeros((n, n))
    for k in range(-p, p + 1):
        A += np.diag(rng.standard_normal(n - abs(k)) + 1.0, k)
    assert bd.bandwidths(A) == (p, p)


@pytest.mark.parametrize("n", [3, 7, 20])
def test_bandwidths_can_be_asymmetric(n, rng):
    A = np.triu(np.ones((n, n)))                 # upper bandwidth n-1, lower 0
    assert bd.bandwidths(A) == (0, n - 1)


@pytest.mark.parametrize("n", SIZES)
def test_dense_matrix_has_full_bandwidth(n, rng):
    A = rng.standard_normal((n, n)) + 5.0
    assert bd.bandwidths(A) == (n - 1, n - 1)


@pytest.mark.parametrize("n", [4, 11])
def test_bandwidths_respects_the_tolerance(n):
    A = np.eye(n)
    A[0, n - 1] = 1e-14
    assert bd.bandwidths(A) == (0, n - 1)        # a nonzero is a nonzero
    assert bd.bandwidths(A, tol=1e-12) == (0, 0)  # unless you say otherwise


@pytest.mark.parametrize("n", SIZES)
def test_is_banded_agrees_with_bandwidths(n, rng):
    A = bd.second_difference(n)
    lo, up = bd.bandwidths(A)
    assert bd.is_banded(A, lo, up)
    if n > 2:
        assert not bd.is_banded(A, 0, 0)


@pytest.mark.parametrize("n", SIZES)
def test_density_is_between_zero_and_one(n, rng):
    for A in (np.eye(n), bd.second_difference(n), rng.standard_normal((n, n)) + 3.0):
        d = bd.density(A)
        assert 0.0 < d <= 1.0
    assert bd.density(np.zeros((n, n))) == 0.0


# ---------------------------------------------------------------- Thomas


@pytest.mark.parametrize("n", SIZES)
def test_thomas_matches_a_dense_solve(n, rng):
    b = rng.standard_normal(n) + 4.0             # diagonally dominant, so it is solvable
    a = rng.standard_normal(max(n - 1, 0))
    c = rng.standard_normal(max(n - 1, 0))
    A = bd.tridiagonal_matrix(a, b, c)
    d = rng.standard_normal(n)
    np.testing.assert_allclose(bd.thomas(a, b, c, d), np.linalg.solve(A, d),
                               rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("scale", SCALES)
def test_thomas_holds_across_scales(scale, rng):
    n = 20
    b = scale * (rng.standard_normal(n) + 4.0)
    a = scale * rng.standard_normal(n - 1)
    c = scale * rng.standard_normal(n - 1)
    A = bd.tridiagonal_matrix(a, b, c)
    x = rng.standard_normal(n)
    got = bd.thomas(a, b, c, A @ x)
    np.testing.assert_allclose(got, x, rtol=1e-7, atol=1e-9 * max(1.0, np.abs(x).max()))


@pytest.mark.parametrize("n", SIZES)
def test_thomas_on_the_second_difference_operator(n):
    A = bd.second_difference(n)
    x = np.arange(1.0, n + 1)
    d = A @ x
    got = bd.thomas(np.full(max(n - 1, 0), -1.0), np.full(n, 2.0),
                    np.full(max(n - 1, 0), -1.0), d)
    np.testing.assert_allclose(got, x, rtol=1e-9, atol=1e-9)


def test_thomas_works_at_size_one():
    np.testing.assert_allclose(bd.thomas([], [4.0], [], [8.0]), [2.0])


@pytest.mark.parametrize("bad", ["short_a", "short_d"])
def test_thomas_rejects_mismatched_lengths(bad):
    n = 5
    a = np.full(n - 1, -1.0)
    b = np.full(n, 2.0)
    c = np.full(n - 1, -1.0)
    d = np.ones(n)
    if bad == "short_a":
        a = a[:-1]
    else:
        d = d[:-1]
    with pytest.raises(ValueError):
        bd.thomas(a, b, c, d)


def test_thomas_reports_a_zero_pivot():
    # 2x2 with a zero pivot after the first elimination step
    with pytest.raises(np.linalg.LinAlgError):
        bd.thomas([1.0], [1.0, 1.0], [1.0], [1.0, 1.0])


# ---------------------------------------------------------------- the model operator


@pytest.mark.parametrize("n", SIZES)
def test_second_difference_eigenvalues_are_exact(n):
    A = bd.second_difference(n)
    got = np.sort(bd.second_difference_eigenvalues(n))
    want = np.sort(np.linalg.eigvalsh(A))
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("n", [3, 8, 25])
@pytest.mark.parametrize("h", [0.25, 1.0, 3.0])
def test_second_difference_scales_with_h_squared(n, h):
    np.testing.assert_allclose(bd.second_difference(n, h),
                               bd.second_difference(n, 1.0) / h ** 2, rtol=1e-12)


@pytest.mark.parametrize("n", SIZES)
def test_second_difference_is_symmetric_positive_definite(n):
    A = bd.second_difference(n)
    assert np.allclose(A, A.T)
    assert np.all(np.linalg.eigvalsh(A) > 0.0)


@pytest.mark.parametrize("n", [2, 6, 30, 120])
def test_second_difference_condition_number_grows_like_n_squared(n):
    kappa = np.linalg.cond(bd.second_difference(n))
    predicted = 4.0 / (np.pi / (n + 1)) ** 2 / 4.0 * 4.0     # ~ (2/(pi h))^2 * pi^2/...
    # the clean statement: kappa ~ (2(n+1)/pi)^2, so the ratio is bounded
    ratio = kappa / ((2.0 * (n + 1) / np.pi) ** 2)
    assert 0.5 < ratio < 1.5, f"kappa {kappa:.1f}, ratio {ratio:.3f} at n={n}"


# ---------------------------------------------------------------- banded LU


@pytest.mark.parametrize("n", [2, 4, 9, 20, 45])
@pytest.mark.parametrize("p", [1, 2, 3])
def test_banded_lu_reproduces_the_matrix(n, p, rng):
    p = min(p, max(n - 1, 1))
    A = np.diag(rng.standard_normal(n) + 6.0 * p)
    for k in range(1, p + 1):
        A += np.diag(rng.standard_normal(n - k), k) + np.diag(rng.standard_normal(n - k), -k)
    L, U = bd.banded_lu(A, p, p)
    np.testing.assert_allclose(L @ U, A, rtol=1e-9, atol=1e-9 * np.abs(A).max())


@pytest.mark.parametrize("n", [3, 10, 30])
@pytest.mark.parametrize("p", [1, 2])
def test_banded_lu_preserves_the_bandwidth(n, p, rng):
    p = min(p, n - 1)
    A = np.diag(rng.standard_normal(n) + 6.0 * p)
    for k in range(1, p + 1):
        A += np.diag(rng.standard_normal(n - k), k) + np.diag(rng.standard_normal(n - k), -k)
    L, U = bd.banded_lu(A, p, p)
    assert bd.bandwidths(L, tol=1e-13)[0] <= p
    assert bd.bandwidths(U, tol=1e-13)[1] <= p


@pytest.mark.parametrize("n", [10, 50, 200, 1000])
@pytest.mark.parametrize("p", [1, 3, 10])
def test_banded_lu_costs_far_less_than_dense(n, p):
    from nalib.lu import flops_lu

    if p < n:
        assert bd.flops_banded_lu(n, p, p) < flops_lu(n)


@pytest.mark.parametrize("n", [20, 100, 500])
def test_banded_lu_flops_are_linear_in_n_for_fixed_bandwidth(n):
    p = 2
    per_row = bd.flops_banded_lu(n, p, p) / n
    assert 1.0 <= per_row <= 4.0 * (p + 1) ** 2, f"{per_row:.1f} flops per row at n={n}"


# ---------------------------------------------------------------- sparse storage


@pytest.mark.parametrize("n", SIZES)
def test_coo_and_csr_round_trip(n, rng):
    A = bd.second_difference(n)
    np.testing.assert_allclose(bd.csr_to_dense(bd.to_csr(A)), A, atol=0.0)
    coo = bd.to_coo(A)
    B = np.zeros_like(A)
    B[coo["row"], coo["col"]] = coo["data"]
    np.testing.assert_allclose(B, A, atol=0.0)


@pytest.mark.parametrize("n", SIZES)
def test_csr_matvec_matches_the_dense_product(n, rng):
    for A in (bd.second_difference(n), rng.standard_normal((n, n)),
              np.diag(rng.standard_normal(n))):
        x = rng.standard_normal(n)
        np.testing.assert_allclose(bd.csr_matvec(bd.to_csr(A), x), A @ x,
                                   rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("scale", SCALES)
def test_csr_matvec_holds_across_scales(scale, rng):
    n = 30
    A = scale * bd.second_difference(n)
    x = rng.standard_normal(n) / scale
    np.testing.assert_allclose(bd.csr_matvec(bd.to_csr(A), x), A @ x, rtol=1e-12,
                               atol=1e-12 * np.abs(A @ x).max())


@pytest.mark.parametrize("n", [1, 4, 25])
def test_csr_of_the_zero_matrix_is_empty(n):
    csr = bd.to_csr(np.zeros((n, n)))
    assert len(csr["data"]) == 0
    np.testing.assert_allclose(bd.csr_matvec(csr, np.ones(n)), np.zeros(n))


@pytest.mark.parametrize("n", [5, 20, 60])
def test_storage_counts_favour_sparse_exactly_when_the_matrix_is_sparse(n):
    sparse = bd.storage_counts(bd.second_difference(n))
    dense = bd.storage_counts(np.ones((n, n)))
    assert sparse["dense"] == n * n
    assert dense["dense"] == n * n
    if n >= 20:
        assert sparse["csr"] < sparse["dense"]
    assert dense["csr"] > dense["dense"]         # CSR loses on a full matrix, as it must


# ---------------------------------------------------------------- fill-in and reordering


@pytest.mark.parametrize("n", [4, 9, 25, 60])
def test_arrow_matrix_tip_first_fills_completely(n):
    """The classic: an arrow with the tip first fills in entirely, and the same matrix with the
    tip last has no fill at all. Same matrix, different ordering."""
    first = bd.fill_in(bd.arrow_matrix(n, tip_first=True))
    last = bd.fill_in(bd.arrow_matrix(n, tip_first=False))
    assert first["fill"] > last["fill"]
    assert last["fill"] == 0


@pytest.mark.parametrize("n", SIZES)
def test_fill_in_never_reports_negative_fill(n, rng):
    A = bd.second_difference(n) if n > 1 else np.array([[2.0]])
    r = bd.fill_in(A)
    assert r["fill"] >= 0
    assert r["fill"] == r["nnz_after"] - r["nnz_before"]
    assert 0.0 < r["density_after"] <= 1.0
    assert r["density_after"] >= r["density_before"]


@pytest.mark.parametrize("n", [2, 5, 17, 40])
def test_arrow_matrix_is_symmetric_positive_definite(n):
    for tip_first in (True, False):
        A = bd.arrow_matrix(n, tip_first=tip_first)
        assert np.allclose(A, A.T)
        assert np.all(np.linalg.eigvalsh(A) > 0.0)


@pytest.mark.parametrize("n", [3, 8, 20, 50])
def test_rcm_is_a_permutation(n, rng):
    A = bd.arrow_matrix(n, tip_first=True)
    p = bd.reverse_cuthill_mckee(A)
    assert sorted(p.tolist()) == list(range(n))


@pytest.mark.parametrize("n", [8, 20, 50])
def test_rcm_does_not_increase_the_bandwidth(n):
    A = bd.arrow_matrix(n, tip_first=True)
    p = bd.reverse_cuthill_mckee(A)
    before = max(bd.bandwidths(A))
    after = max(bd.bandwidths(A[np.ix_(p, p)]))
    assert after <= before, f"bandwidth went from {before} to {after}"


@pytest.mark.parametrize("n", [10, 30])
def test_rcm_reduces_fill_on_the_arrow(n):
    A = bd.arrow_matrix(n, tip_first=True)
    p = bd.reverse_cuthill_mckee(A)
    assert bd.fill_in(A[np.ix_(p, p)])["fill"] <= bd.fill_in(A)["fill"]


def test_rcm_of_a_diagonal_matrix_changes_nothing_structural():
    n = 7
    A = np.diag(np.arange(1.0, n + 1))
    p = bd.reverse_cuthill_mckee(A)
    assert bd.bandwidths(A[np.ix_(p, p)]) == (0, 0)
