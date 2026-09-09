"""Tests for `nalib.krylov_eig`.

The point of these methods is that they work when the matrix cannot be stored, so the tests use
`LinearOperator` throughout and several of them run at sizes where a dense matrix would not fit.

Two things are checked that a correctness-only suite would skip. That the **free** residual
bound ``|h_{m+1,m}| |s_m|`` equals the true residual, since the whole stopping rule rests on it.
And that the extremes converge before the interior, since that is what decides whether the
method is usable on a given problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import krylov_eig as ke
from nalib.krylov import LinearOperator


# ---------------------------------------------------------------- Rayleigh-Ritz


@pytest.mark.parametrize("n,m", [(10, 3), (20, 7), (50, 20), (30, 30)])
def test_rayleigh_ritz_residuals_are_orthogonal_to_the_subspace(n, m, rng):
    """The defining property of a Ritz pair: ``A y - theta y`` is perpendicular to the space it
    was drawn from. Everything else about the method follows from that."""
    A = rng.standard_normal((n, n))
    A = A + A.T
    Q, _ = np.linalg.qr(rng.standard_normal((n, m)))
    out = ke.rayleigh_ritz(A, Q)
    for j in range(m):
        r = A @ out["vectors"][:, j] - out["values"][j] * out["vectors"][:, j]
        assert np.linalg.norm(Q.T @ r) < 1e-11 * max(np.linalg.norm(A), 1.0)


@pytest.mark.parametrize("n", [5, 12, 30])
def test_a_full_subspace_gives_the_exact_spectrum(n, rng):
    A = rng.standard_normal((n, n))
    A = A + A.T
    out = ke.rayleigh_ritz(A, np.eye(n))
    assert np.max(np.abs(np.sort(out["values"]) - np.sort(np.linalg.eigvalsh(A)))) < 1e-11


@pytest.mark.parametrize("n", [10, 25])
def test_ritz_values_interlace_the_true_spectrum(n, rng):
    """For a symmetric matrix the Ritz values always lie inside the true spectrum, so they
    bracket it from within and can never overshoot."""
    A = rng.standard_normal((n, n))
    A = A + A.T
    exact = np.sort(np.linalg.eigvalsh(A))
    for m in (2, n // 2, n - 1):
        Q, _ = np.linalg.qr(rng.standard_normal((n, m)))
        theta = np.sort(ke.rayleigh_ritz(A, Q)["values"])
        assert theta[0] >= exact[0] - 1e-10
        assert theta[-1] <= exact[-1] + 1e-10


def test_rayleigh_ritz_rejects_more_vectors_than_dimensions(rng):
    with pytest.raises(ValueError):
        ke.rayleigh_ritz(rng.standard_normal((4, 4)), rng.standard_normal((4, 6)))


# ---------------------------------------------------------------- the free residual bound


@pytest.mark.parametrize("n,m", [(50, 10), (100, 25), (200, 40)])
def test_the_cheap_bound_equals_the_true_residual(n, m):
    """**The whole stopping rule rests on this.** Measured ratio 1.0000 at every Ritz value:
    the residual is exactly ``|h_{m+1,m}| |s_m|``, so no Ritz vector need ever be formed."""
    op = ke.sparse_laplacian(n)
    out = ke.lanczos_eigen(op, m=m, rng=np.random.default_rng(2))
    for j in range(out.values.size):
        y = out.vectors[:, j]
        true = float(np.linalg.norm(op.matvec(y) - out.values[j] * y))
        assert abs(true - out.residuals[j]) < 1e-9 * max(true, 1e-12), \
            f"Ritz {j}: cheap {out.residuals[j]:.3e}, true {true:.3e}"


def test_a_one_dimensional_argument_is_one_eigenvector():
    """So its LAST entry is what matters, not the whole thing. An earlier version guessed from
    the shape and could not tell this case from an already-extracted last row."""
    s = np.array([0.1, 0.2, 0.7])
    got = ke.ritz_residual(2.0, s)
    assert got.shape == (1,)
    assert np.allclose(got, 2.0 * 0.7)


def test_the_two_shapes_agree_on_a_single_column():
    s = np.array([0.1, 0.2, 0.7])
    assert np.allclose(ke.ritz_residual(2.0, s), ke.ritz_residual(2.0, s[:, None]))


def test_ritz_residual_rejects_a_three_dimensional_argument():
    with pytest.raises(ValueError):
        ke.ritz_residual(1.0, np.zeros((2, 2, 2)))


def test_the_bound_reads_the_last_row_of_a_matrix():
    S = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, -0.25]])
    assert np.allclose(ke.ritz_residual(-3.0, S), [1.5, 0.75])


# ---------------------------------------------------------------- Lanczos


@pytest.mark.parametrize("n", [20, 100, 1000])
def test_lanczos_finds_the_extremes_of_the_laplacian(n):
    """The exact spectrum is known in closed form, so this needs no dense reference at all."""
    op = ke.sparse_laplacian(n)
    exact = ke.laplacian_eigenvalues(n)
    out = ke.lanczos_eigen(op, m=min(40, n), rng=np.random.default_rng(1))
    got = np.sort(out.values)
    assert abs(got[-1] - exact[-1]) < 0.05
    assert abs(got[0] - exact[0]) < 0.05


@pytest.mark.parametrize("n", [10, 30])
def test_a_full_basis_gives_the_exact_answer(n):
    op = ke.sparse_laplacian(n)
    out = ke.lanczos_eigen(op, m=n, rng=np.random.default_rng(1))
    assert np.max(np.abs(np.sort(out.values) - ke.laplacian_eigenvalues(n))) < 1e-9


def test_lanczos_counts_its_matvecs():
    op = ke.sparse_laplacian(60)
    out = ke.lanczos_eigen(op, m=15, rng=np.random.default_rng(1))
    assert out.matvecs == 15


def test_lanczos_stops_early_on_an_invariant_subspace():
    """Starting inside an invariant subspace exhausts the Krylov space, and the recurrence must
    stop cleanly rather than divide by a zero beta."""
    spectrum = np.array([5.0, 3.0, 1.0, -2.0])
    op, _ = ke.operator_with_spectrum(spectrum, seed=1)
    dense = op.to_array()
    v = np.linalg.eigh(dense)[1][:, 0]
    out = ke.lanczos_eigen(op, v=v, m=4, rng=np.random.default_rng(1))
    assert out.basis_size == 1
    assert abs(out.values[0] - np.min(spectrum)) < 1e-10


@pytest.mark.parametrize("bad", [0, -1, 999])
def test_lanczos_rejects_a_bad_basis_size(bad):
    with pytest.raises(ValueError):
        ke.lanczos_eigen(ke.sparse_laplacian(20), m=bad)


def test_lanczos_rejects_a_zero_start():
    with pytest.raises(ValueError):
        ke.lanczos_eigen(ke.sparse_laplacian(10), v=np.zeros(10))


def test_lanczos_rejects_a_start_of_the_wrong_length():
    with pytest.raises(ValueError):
        ke.lanczos_eigen(ke.sparse_laplacian(10), v=np.ones(7))


# ---------------------------------------------------------------- which converge first


@pytest.mark.parametrize("m", [10, 20, 40, 80])
def test_the_extremes_converge_before_the_interior(m):
    """**The metric matters and an obvious one is wrong.** Measuring one interior eigenvalue's
    distance to the NEAREST Ritz value rewards accidents: at m=5 that scored the middle better
    than the extremes, which is the opposite of the truth. This uses medians over bands.

    Measured on a 200 point Laplacian: the interior is 4 to 6 times worse at every m."""
    op = ke.sparse_laplacian(200)
    out = ke.convergence_order(op, m, exact=ke.laplacian_eigenvalues(200),
                               rng=np.random.default_rng(3))
    d, p = out["distance"], out["position"]
    outer = float(np.median(d[(p < 0.15) | (p > 0.85)]))
    inner = float(np.median(d[(p > 0.35) & (p < 0.65)]))
    assert inner > 2.0 * outer, f"outer {outer:.2e}, inner {inner:.2e}"


def test_convergence_order_accepts_a_known_spectrum():
    """So the measurement can run where forming a dense matrix is impossible."""
    op = ke.sparse_laplacian(5000)
    out = ke.convergence_order(op, 20, exact=ke.laplacian_eigenvalues(5000),
                               rng=np.random.default_rng(3))
    assert out["distance"].size == 5000
    assert out["matvecs"] == 20


# ---------------------------------------------------------------- ghosts


@pytest.mark.parametrize("gap,expect_ghosts", [(1.0, False), (10.0, True), (30.0, True)])
def test_ghosts_appear_exactly_when_an_eigenvalue_converges(gap, expect_ghosts):
    """**Paige's theorem, measured.** Orthogonality is lost at the moment a Ritz value
    converges, so ghosts are a symptom of success. Moving one number turns them on:
    gap 1 gives 0 duplicates and an orthogonality loss of 1.6e-7; gap 10 gives 3 duplicates and
    a loss of 3.5."""
    op, _ = ke.operator_with_spectrum(ke.separated_spectrum(40, gap), seed=1)
    out = ke.ghost_eigenvalues(op, m=35, rng=np.random.default_rng(4), tol=1e-6)
    if expect_ghosts:
        assert out["plain"]["duplicates"] > 0, "no ghosts where they were expected"
    else:
        assert out["plain"]["duplicates"] == 0, "ghosts where none were expected"
    assert out["reorthogonalized"]["duplicates"] == 0, "reorthogonalization should prevent them"


def test_ghosts_do_not_damage_the_answer():
    """They duplicate a converged eigenvalue rather than inventing a wrong one, so the top
    eigenvalue is still right to 1e-13 even with three ghosts present."""
    op, exact = ke.operator_with_spectrum(ke.separated_spectrum(40, 10.0), seed=1)
    out = ke.lanczos_eigen(op, m=35, reorthogonalize=False, rng=np.random.default_rng(4))
    assert abs(np.max(out.values) - exact[-1]) < 1e-9


@pytest.mark.parametrize("gap", [1.0, 10.0])
def test_reorthogonalization_keeps_the_basis_orthogonal(gap):
    op, _ = ke.operator_with_spectrum(ke.separated_spectrum(40, gap), seed=1)
    plain = ke.basis_orthogonality(op, m=35, reorthogonalize=False,
                                   rng=np.random.default_rng(5))
    reorth = ke.basis_orthogonality(op, m=35, reorthogonalize=True,
                                    rng=np.random.default_rng(5))
    assert reorth[-1] < 1e-12, f"reorthogonalized basis lost orthogonality: {reorth[-1]:.2e}"
    assert reorth[-1] < plain[-1]


def test_the_orthogonality_trail_grows_monotonically_enough():
    op, _ = ke.operator_with_spectrum(ke.separated_spectrum(40, 10.0), seed=1)
    trail = ke.basis_orthogonality(op, m=35, reorthogonalize=False,
                                   rng=np.random.default_rng(5))
    assert trail[-1] > 1e6 * trail[0], f"{trail[0]:.2e} to {trail[-1]:.2e}"


def test_separated_spectrum_rejects_a_tiny_size():
    with pytest.raises(ValueError):
        ke.separated_spectrum(1, 10.0)


def test_operator_with_spectrum_has_that_spectrum():
    spectrum = np.array([7.0, 2.0, -3.0, 0.5])
    op, exact = ke.operator_with_spectrum(spectrum, seed=2)
    assert np.max(np.abs(np.sort(np.linalg.eigvalsh(op.to_array())) - np.sort(spectrum))) < 1e-12
    assert np.allclose(exact, np.sort(spectrum))


def test_operator_with_spectrum_rejects_an_empty_spectrum():
    with pytest.raises(ValueError):
        ke.operator_with_spectrum([])


# ---------------------------------------------------------------- restarting


@pytest.mark.parametrize("n_wanted", [1, 2, 4])
def test_restarted_arnoldi_finds_the_wanted_eigenvalues(n_wanted):
    op, exact = ke.operator_with_spectrum(np.arange(1.0, 31.0) ** 1.5, seed=3)
    out = ke.restarted_arnoldi(op, n_wanted=n_wanted, basis_size=12, tol=1e-10,
                               max_restarts=400, rng=np.random.default_rng(7))
    assert out.converged.all(), out.message
    got = np.sort(out.values)[::-1]
    assert np.max(np.abs(got - exact[::-1][:n_wanted])) < 1e-8


def test_restarting_can_ask_for_the_smallest():
    op, exact = ke.operator_with_spectrum(np.arange(1.0, 31.0) ** 1.5, seed=3)
    out = ke.restarted_arnoldi(op, n_wanted=2, basis_size=12, which="smallest",
                               tol=1e-10, max_restarts=400, rng=np.random.default_rng(7))
    assert out.converged.all(), out.message
    assert np.max(np.abs(np.sort(out.values) - exact[:2])) < 1e-8


def test_the_residual_history_decreases():
    op, _ = ke.operator_with_spectrum(np.arange(1.0, 31.0) ** 1.5, seed=3)
    out = ke.restarted_arnoldi(op, n_wanted=1, basis_size=10, tol=1e-12,
                               max_restarts=200, rng=np.random.default_rng(7))
    assert out.history[-1] < out.history[0]


@pytest.mark.parametrize("n", [1000, 10000, 100000])
def test_it_works_at_sizes_where_a_dense_matrix_would_not_fit(n):
    """**The reason the method exists.** At n = 100000 a dense matrix is 80 GB. Measured: one
    restart, 39 matvecs, a residual of 7e-14."""
    op = ke.spiked_laplacian(n, spike=20.0)
    out = ke.restarted_arnoldi(op, n_wanted=1, basis_size=20, tol=1e-10,
                               max_restarts=200, rng=np.random.default_rng(7))
    assert out.converged.all(), out.message
    assert out.matvecs < 200, f"{out.matvecs} matvecs is too many for this problem"
    y = out.vectors[:, 0]
    assert np.linalg.norm(op.matvec(y) - out.values[0] * y) < 1e-9


def test_restarting_rejects_impossible_sizes():
    op = ke.sparse_laplacian(20)
    for wanted, basis in ((0, 10), (5, 3), (5, 50)):
        with pytest.raises(ValueError):
            ke.restarted_arnoldi(op, n_wanted=wanted, basis_size=basis)


def test_restarting_rejects_a_bad_which():
    with pytest.raises(ValueError):
        ke.restarted_arnoldi(ke.sparse_laplacian(20), n_wanted=1, which="biggest")


def test_restarting_rejects_a_start_of_the_wrong_length():
    with pytest.raises(ValueError):
        ke.restarted_arnoldi(ke.sparse_laplacian(20), n_wanted=1, v=np.ones(5))


# ---------------------------------------------------------------- the test operators


@pytest.mark.parametrize("n", [2, 5, 50, 500])
def test_the_laplacian_matches_its_closed_form(n):
    op = ke.sparse_laplacian(n)
    dense = np.column_stack([op.matvec(e) for e in np.eye(n)])
    assert np.max(np.abs(np.sort(np.linalg.eigvalsh(dense))
                         - ke.laplacian_eigenvalues(n))) < 1e-10


@pytest.mark.parametrize("size", [3, 6, 12])
def test_the_two_dimensional_laplacian_matches_its_closed_form(size):
    op = ke.sparse_laplacian(size, dimension=2)
    n = size * size
    dense = np.column_stack([op.matvec(e) for e in np.eye(n)])
    assert np.max(np.abs(np.sort(np.linalg.eigvalsh(dense))
                         - ke.laplacian_eigenvalues(size, dimension=2))) < 1e-10


def test_the_laplacian_is_symmetric():
    n = 30
    op = ke.sparse_laplacian(n)
    dense = np.column_stack([op.matvec(e) for e in np.eye(n)])
    assert np.linalg.norm(dense - dense.T) < 1e-14


def test_the_spiked_laplacian_separates_one_eigenvalue():
    n = 60
    plain = ke.sparse_laplacian(n)
    spiked = ke.spiked_laplacian(n, spike=20.0)
    d_plain = np.column_stack([plain.matvec(e) for e in np.eye(n)])
    d_spiked = np.column_stack([spiked.matvec(e) for e in np.eye(n)])
    w_plain = np.sort(np.linalg.eigvalsh(d_plain))
    w_spiked = np.sort(np.linalg.eigvalsh(d_spiked))
    assert (w_spiked[-1] - w_spiked[-2]) > 100.0 * (w_plain[-1] - w_plain[-2])


def test_the_laplacian_rejects_bad_arguments():
    with pytest.raises(ValueError):
        ke.sparse_laplacian(0)
    with pytest.raises(ValueError):
        ke.sparse_laplacian(10, dimension=3)
    with pytest.raises(ValueError):
        ke.spiked_laplacian(1, 1.0)


def test_the_operator_never_builds_a_dense_matrix():
    """A million by million operator must be constructible and applicable without allocating
    anything of size n squared."""
    op = ke.sparse_laplacian(1_000_000)
    x = np.ones(1_000_000)
    y = op.matvec(x)
    assert y.shape == (1_000_000,)
    assert abs(y[0] - 1.0) < 1e-12 and abs(y[500_000]) < 1e-12
    with pytest.raises(ValueError):
        op.to_array()
