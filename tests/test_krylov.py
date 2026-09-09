"""Tests for `nalib.krylov`.

Krylov methods are checked against their defining identities rather than against a reference
implementation: the Arnoldi relation, conjugacy of the CG directions, the finite termination
property, and the convergence bound. Those hold at every size and for every spectrum, so the
tests sweep both. Where floating point breaks an identity, as it does for CG conjugacy at large
kappa, the test says so explicitly rather than loosening the tolerance until it passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import krylov as kr
from nalib.banded import second_difference
from nalib.cholesky import random_spd


SIZES = [1, 2, 3, 5, 9, 20, 47]
KAPPAS = [1.0, 10.0, 1e2, 1e4]
SCALES = [1e-8, 1e-2, 1.0, 1e5, 1e9]
TOL = 1e-10


def spd_with(lams, rng):
    """An SPD matrix with exactly the requested eigenvalues. The size comes from lams."""
    lams = np.asarray(lams, dtype=float)
    n = lams.size
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * lams) @ Q.T
    return (A + A.T) / 2


# ---------------------------------------------------------------- LinearOperator


@pytest.mark.parametrize("n", SIZES)
def test_operator_matches_the_matrix_it_wraps(n, rng):
    A = rng.standard_normal((n, n))
    op = kr.as_operator(A)
    x = rng.standard_normal(n)
    np.testing.assert_allclose(op.matvec(x), A @ x, rtol=1e-13, atol=1e-14)
    assert op.shape == (n, n)


@pytest.mark.parametrize("n", SIZES)
def test_operator_counts_its_matvecs(n, rng):
    op = kr.as_operator(rng.standard_normal((n, n)))
    assert op.n_matvec == 0
    for k in range(1, 4):
        op.matvec(rng.standard_normal(n))
        assert op.n_matvec == k


@pytest.mark.parametrize("n", [1, 4, 17, 60])
def test_a_matrix_free_operator_needs_no_matrix(n, rng):
    """The point of the interface: a stencil in code beats an array. This one is the second
    difference operator, and n is free."""
    def apply(v):
        y = 2.0 * v
        y[:-1] -= v[1:]
        y[1:] -= v[:-1]
        return y

    op = kr.LinearOperator(apply, shape=(n, n))
    x = rng.standard_normal(n)
    np.testing.assert_allclose(op.matvec(x), second_difference(n) @ x, rtol=1e-12, atol=1e-13)


def test_operator_rejects_a_wrong_length_vector():
    op = kr.LinearOperator(lambda v: v, shape=(4, 4))
    with pytest.raises(ValueError):
        op.matvec(np.ones(5))


# ---------------------------------------------------------------- the naive basis


@pytest.mark.parametrize("n", [5, 12, 30])
@pytest.mark.parametrize("m", [1, 2, 4])
def test_krylov_basis_columns_are_the_powers(n, m, rng):
    A = rng.standard_normal((n, n))
    v = rng.standard_normal(n)
    K = kr.krylov_basis(A, v, m)
    assert K.shape == (n, m)
    for j in range(m):
        np.testing.assert_allclose(K[:, j], np.linalg.matrix_power(A, j) @ v,
                                   rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("m", [4, 8, 12, 16])
def test_the_naive_basis_gets_worse_as_m_grows(m, rng):
    """The power method in disguise: every column tilts further toward the dominant
    eigenvector, so the condition number grows without bound."""
    n = 40
    A = spd_with(np.linspace(1.0, 20.0, n), rng)
    v = rng.standard_normal(n)
    assert np.linalg.cond(kr.krylov_basis(A, v, m)) > np.linalg.cond(
        kr.krylov_basis(A, v, m - 2))


@pytest.mark.parametrize("n", [6, 20])
def test_orthonormalized_naive_basis_is_orthonormal(n, rng):
    A = spd_with(np.linspace(1.0, 5.0, n), rng)
    Q = kr.krylov_basis(A, rng.standard_normal(n), min(4, n), orthonormalize=True)
    np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-12)


# ---------------------------------------------------------------- Arnoldi


@pytest.mark.parametrize("n", [2, 5, 12, 30])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_arnoldi_relation_holds(n, m, rng):
    """A Q_m = Q_{m+1} H_m, the identity the whole method is built on."""
    m = min(m, n)
    A = rng.standard_normal((n, n))
    d = kr.arnoldi(A, rng.standard_normal(n), m)
    Q, H = d["Q"], d["H"]
    k = H.shape[1]
    np.testing.assert_allclose(A @ Q[:, :k], Q[:, :H.shape[0]] @ H, atol=1e-9)


@pytest.mark.parametrize("n", [2, 5, 12, 30])
def test_arnoldi_basis_is_orthonormal(n, rng):
    m = min(6, n)
    Q = kr.arnoldi(rng.standard_normal((n, n)), rng.standard_normal(n), m)["Q"]
    np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)


@pytest.mark.parametrize("n", [4, 10, 25])
def test_arnoldi_H_is_upper_hessenberg(n, rng):
    H = kr.arnoldi(rng.standard_normal((n, n)), rng.standard_normal(n), min(6, n))["H"]
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if i > j + 1:
                assert abs(H[i, j]) < 1e-12, f"H[{i},{j}] = {H[i, j]:.3e} is below the band"


@pytest.mark.parametrize("n", [4, 10, 25])
def test_arnoldi_H_is_tridiagonal_for_a_symmetric_matrix(n, rng):
    A = spd_with(np.linspace(1.0, 8.0, n), rng)
    H = kr.arnoldi(A, rng.standard_normal(n), min(6, n))["H"]
    k = min(H.shape)
    for i in range(k):
        for j in range(k):
            if abs(i - j) > 1:
                assert abs(H[i, j]) < 1e-9, f"H[{i},{j}] = {H[i, j]:.3e} outside the tridiagonal"


@pytest.mark.parametrize("n", [3, 8, 20])
def test_arnoldi_at_m_equals_n_recovers_the_eigenvalues(n, rng):
    A = rng.standard_normal((n, n))
    d = kr.arnoldi(A, rng.standard_normal(n), n)
    H = d["H"]
    k = min(H.shape)
    got = np.sort_complex(np.linalg.eigvals(H[:k, :k]))
    want = np.sort_complex(np.linalg.eigvals(A))
    np.testing.assert_allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("n", [4, 9])
def test_arnoldi_breakdown_leaves_an_orthonormal_basis(n, rng):
    """A vector inside a small invariant subspace makes the method break down early. The basis
    returned must still be orthonormal, which means the zero column has to be dropped."""
    lams = np.arange(1.0, n + 1)
    Q0, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q0 * lams) @ Q0.T
    v = Q0[:, 0] + Q0[:, 1]                      # lives in a 2-dimensional invariant subspace
    d = kr.arnoldi(A, v, n)
    Q = d["Q"]
    np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)
    assert Q.shape[1] <= 3, f"expected an early stop, got {Q.shape[1]} columns"


@pytest.mark.parametrize("n", [4, 9])
def test_arnoldi_breakdown_gives_exact_eigenvalues(n, rng):
    lams = np.arange(1.0, n + 1)
    Q0, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q0 * lams) @ Q0.T
    v = Q0[:, 0] + Q0[:, 1]
    H = kr.arnoldi(A, v, n)["H"]
    k = min(H.shape)
    got = np.sort(np.real(np.linalg.eigvals(H[:k, :k])))
    for g in got:
        assert np.min(np.abs(lams - g)) < 1e-8, f"Ritz value {g:.6f} is not an eigenvalue"


# ---------------------------------------------------------------- Lanczos


@pytest.mark.parametrize("n", [2, 5, 12, 30])
def test_lanczos_tridiagonal_matches_the_rayleigh_quotient(n, rng):
    m = min(5, n)
    A = spd_with(np.linspace(1.0, 9.0, n), rng)
    d = kr.lanczos(A, rng.standard_normal(n), m, reorthogonalize=True)
    T = kr.tridiagonal_from_lanczos(d["alpha"], d["beta"])
    Q = d["Q"][:, :T.shape[0]]
    np.testing.assert_allclose(T, Q.T @ A @ Q, atol=1e-8)


@pytest.mark.parametrize("n", [5, 15, 40])
def test_lanczos_with_reorthogonalization_stays_orthonormal(n, rng):
    A = spd_with(np.geomspace(1.0, 1e4, n), rng)
    Q = kr.lanczos(A, rng.standard_normal(n), n, reorthogonalize=True)["Q"]
    np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-9)


def test_lanczos_without_reorthogonalization_loses_orthogonality():
    """Not a defect: it is the phenomenon lesson 26 is about, and a test that never sees it
    would mean the experiment is not reproducing the known behaviour."""
    rng = np.random.default_rng(0)
    n = 80
    A = spd_with(np.concatenate([[30.0], np.linspace(1.0, 10.0, n - 1)]), rng)
    v = rng.standard_normal(n)
    Q = kr.lanczos(A, v, 40, reorthogonalize=False)["Q"]
    loss = np.max(np.abs(Q.T @ Q - np.eye(Q.shape[1])))
    assert loss > 1e-4, f"orthogonality loss was only {loss:.2e}, expected a real collapse"


@pytest.mark.parametrize("n", [3, 8, 20])
def test_tridiagonal_from_lanczos_has_the_right_shape_and_symmetry(n, rng):
    d = kr.lanczos(spd_with(np.linspace(1.0, 5.0, n), rng),
                   rng.standard_normal(n), n, reorthogonalize=True)
    T = kr.tridiagonal_from_lanczos(d["alpha"], d["beta"])
    assert T.shape[0] == T.shape[1]
    np.testing.assert_allclose(T, T.T, atol=1e-14)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if abs(i - j) > 1:
                assert T[i, j] == 0.0


@pytest.mark.parametrize("n", [4, 10, 25])
def test_ritz_values_lie_inside_the_spectrum(n, rng):
    """A Rayleigh quotient on a subspace cannot leave the range of the eigenvalues, so this
    holds at every m and for every starting vector."""
    lams = np.linspace(-3.0, 7.0, n)
    Q0, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q0 * lams) @ Q0.T
    A = (A + A.T) / 2
    for m in range(1, n + 1):
        r = kr.ritz_values(A, rng.standard_normal(n), m)
        assert r.min() >= lams.min() - 1e-8
        assert r.max() <= lams.max() + 1e-8


@pytest.mark.parametrize("n", [6, 15, 30])
def test_ritz_values_converge_to_the_extremes_first(n, rng):
    lams = np.linspace(1.0, 100.0, n)
    A = spd_with(lams, rng)
    v = rng.standard_normal(n)
    few = kr.ritz_values(A, v, max(2, n // 4))
    many = kr.ritz_values(A, v, n)
    assert abs(many.max() - lams.max()) <= abs(few.max() - lams.max()) + 1e-9
    assert abs(many.min() - lams.min()) <= abs(few.min() - lams.min()) + 1e-9


# ---------------------------------------------------------------- steepest descent


@pytest.mark.parametrize("n", SIZES)
def test_steepest_descent_solves_an_spd_system(n, rng):
    A = spd_with(np.linspace(1.0, 4.0, n), rng)
    x_true = rng.standard_normal(n)
    r = kr.steepest_descent(A, A @ x_true, tol=TOL, max_iter=20000, keep_history=False)
    assert r.converged, r.message
    np.testing.assert_allclose(r.x, x_true, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n", [5, 15, 30])
def test_consecutive_steepest_descent_directions_are_orthogonal(n, rng):
    """Exactly 90 degrees, every step. This is the zigzag, and it is a theorem, so the
    tolerance is roundoff rather than something generous."""
    A = spd_with(np.geomspace(1.0, 1e3, n), rng)
    b = rng.standard_normal(n)
    x = np.zeros(n)
    r = b - A @ x
    for _ in range(min(12, 3 * n)):
        Ar = A @ r
        alpha = (r @ r) / (r @ Ar)
        x = x + alpha * r
        r_new = r - alpha * Ar
        denom = np.linalg.norm(r) * np.linalg.norm(r_new)
        if denom < 1e-280:
            break
        assert abs(r @ r_new) / denom < 1e-9
        r = r_new


def test_steepest_descent_reports_an_indefinite_matrix():
    A = np.array([[1.0, 0.0], [0.0, -1.0]])
    r = kr.steepest_descent(A, np.array([1.0, 1.0]), max_iter=10, keep_history=False)
    assert not r.converged
    assert "positive definite" in r.message


# ---------------------------------------------------------------- conjugate gradient


@pytest.mark.parametrize("n", SIZES)
def test_cg_solves_an_spd_system(n, rng):
    A = random_spd(n, rng=rng)
    x_true = rng.standard_normal(n)
    r = kr.conjugate_gradient(A, A @ x_true, tol=TOL, max_iter=10 * n, keep_history=False)
    assert r.converged, r.message
    np.testing.assert_allclose(r.x, x_true, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("scale", SCALES)
def test_cg_is_scale_invariant(scale, rng):
    n = 15
    A = spd_with(np.linspace(1.0, 20.0, n), rng)
    x_true = rng.standard_normal(n)
    plain = kr.conjugate_gradient(A, A @ x_true, tol=TOL, max_iter=10 * n, keep_history=False)
    scaled = kr.conjugate_gradient(scale * A, scale * (A @ x_true), tol=TOL,
                                   max_iter=10 * n, keep_history=False)
    assert plain.converged and scaled.converged
    assert abs(plain.n_iter - scaled.n_iter) <= 1
    np.testing.assert_allclose(scaled.x, plain.x, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n", [3, 8, 20, 40])
def test_cg_terminates_within_n_steps_on_a_well_conditioned_system(n, rng):
    A = spd_with(np.linspace(1.0, 10.0, n), rng)
    r = kr.conjugate_gradient(A, rng.standard_normal(n), tol=1e-10, max_iter=10 * n,
                              keep_history=False)
    assert r.converged and r.n_iter <= n, f"{r.n_iter} steps at n={n}"


@pytest.mark.parametrize("clusters", [1, 2, 3, 5])
def test_cg_finishes_in_about_as_many_steps_as_there_are_clusters(clusters, rng):
    """The result the whole of lesson 25 is built on: the count follows the number of distinct
    eigenvalues, not kappa. Here kappa is 1e6 in every case."""
    n = 60
    lams = np.repeat(np.geomspace(1.0, 1e6, clusters),
                     int(np.ceil(n / clusters)))[:n]
    op = kr.LinearOperator(lambda v, l=lams: l * v, shape=(n, n))
    r = kr.conjugate_gradient(op, rng.standard_normal(n), tol=1e-10, max_iter=10 * n,
                              keep_history=False)
    assert r.converged
    assert r.n_iter <= clusters + 2, f"{r.n_iter} steps for {clusters} clusters"


@pytest.mark.parametrize("kappa", KAPPAS)
def test_cg_never_violates_its_own_bound(kappa, rng):
    n = 40
    A = spd_with(np.geomspace(1.0, kappa, n), rng) if kappa > 1 else np.eye(n)
    b = rng.standard_normal(n)
    r = kr.conjugate_gradient(A, b, tol=1e-12, max_iter=10 * n)
    x_star = np.linalg.solve(A, b)
    e0 = x_star
    norm_a = lambda v: float(np.sqrt(max(v @ (A @ v), 0.0)))
    bound = kr.cg_convergence_bound(max(kappa, 1.0 + 1e-12), np.arange(len(r.iterates)))
    for k, xk in enumerate(r.iterates):
        rel = norm_a(xk - x_star) / max(norm_a(e0), 1e-300)
        assert rel <= bound[k] * (1.0 + 1e-6) + 1e-12, f"bound broken at step {k}"


@pytest.mark.parametrize("kappa", [10.0, 1e2])
def test_cg_directions_are_conjugate_at_modest_kappa(rng, kappa):
    """Holds to roundoff while kappa is small. It degrades with kappa, which the next test
    records rather than hides."""
    n = 30
    A = spd_with(np.geomspace(1.0, kappa, n), rng)
    b = rng.standard_normal(n)
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    P = [p.copy()]
    rs = r @ r
    for _ in range(min(10, n - 1)):
        Ap = A @ p
        al = rs / (p @ Ap)
        x = x + al * p
        r = r - al * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < 1e-13 * np.linalg.norm(b):
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
        P.append(p.copy())
    P = np.array(P)
    G = P @ A @ P.T
    scale = np.sqrt(np.outer(np.diag(G), np.diag(G)))
    off = np.abs(G / scale)
    np.fill_diagonal(off, 0.0)
    assert off.max() < 1e-6, f"worst conjugacy {off.max():.2e} at kappa {kappa:.0e}"


def test_cg_conjugacy_degrades_as_kappa_grows():
    """The measured fact from lesson 24 section 8, asserted as a trend so that a change in the
    implementation that quietly fixed or worsened it would be noticed."""
    worst = []
    for e in (1, 3, 6):
        gen = np.random.default_rng(7)
        n = 40
        A = spd_with(np.geomspace(1.0, 10.0 ** e, n), gen)
        b = gen.standard_normal(n)
        x = np.zeros(n)
        r = b - A @ x
        p = r.copy()
        P = [p.copy()]
        rs = r @ r
        for _ in range(20):
            Ap = A @ p
            al = rs / (p @ Ap)
            x = x + al * p
            r = r - al * Ap
            rs_new = r @ r
            if np.sqrt(rs_new) < 1e-13 * np.linalg.norm(b):
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
            P.append(p.copy())
        P = np.array(P)
        G = P @ A @ P.T
        scale = np.sqrt(np.outer(np.diag(G), np.diag(G)))
        off = np.abs(G / scale)
        np.fill_diagonal(off, 0.0)
        worst.append(off.max())
    assert worst[0] < worst[-1], f"conjugacy did not degrade with kappa: {worst}"


@pytest.mark.parametrize("n", [10, 30, 60])
def test_cg_beats_steepest_descent(n, rng):
    """Steepest descent needs O(kappa) steps and CG needs O(sqrt(kappa)), so at kappa = 1e3 the
    gap is two orders of magnitude and the budget has to be generous enough to let steepest
    descent actually finish. Capping it early would turn the comparison into a tautology."""
    A = spd_with(np.geomspace(1.0, 1e3, n), rng)
    b = rng.standard_normal(n)
    c = kr.conjugate_gradient(A, b, tol=1e-8, max_iter=200000, keep_history=False)
    s = kr.steepest_descent(A, b, tol=1e-8, max_iter=200000, keep_history=False)
    assert c.converged and s.converged
    assert c.n_iter * 10 < s.n_iter, f"CG {c.n_iter}, steepest descent {s.n_iter}"


def test_steepest_descent_count_tracks_kappa_while_cg_tracks_its_square_root():
    """The single most important comparison in Part 4, checked as an exponent rather than as a
    pair of numbers, so it cannot pass by accident on one problem."""
    n = 50
    counts = {"cg": [], "sd": []}
    kappas = [1e2, 1e3, 1e4]
    for kappa in kappas:
        gen = np.random.default_rng(11)
        A = spd_with(np.geomspace(1.0, kappa, n), gen)
        b = gen.standard_normal(n)
        counts["cg"].append(kr.conjugate_gradient(A, b, tol=1e-8, max_iter=400000,
                                                  keep_history=False).n_iter)
        counts["sd"].append(kr.steepest_descent(A, b, tol=1e-8, max_iter=400000,
                                                keep_history=False).n_iter)
    lk = np.log10(kappas)
    e_cg = np.polyfit(lk, np.log10(counts["cg"]), 1)[0]
    e_sd = np.polyfit(lk, np.log10(counts["sd"]), 1)[0]
    assert e_cg < e_sd - 0.2, f"CG exponent {e_cg:.3f}, steepest descent {e_sd:.3f}"
    assert e_sd > 2.0 * e_cg - 0.3, f"CG {e_cg:.3f} against steepest descent {e_sd:.3f}"


@pytest.mark.parametrize("kappa", [1e2, 1e4, 1e6])
def test_cg_convergence_bound_is_a_decreasing_sequence(kappa):
    b = kr.cg_convergence_bound(kappa, np.arange(30))
    assert b[0] >= 1.0
    assert np.all(np.diff(b) <= 0.0)


@pytest.mark.parametrize("kappa,cross_below", [(1e2, 10), (1e4, 60), (1e6, 500)])
def test_cg_bound_overtakes_the_steepest_descent_bound(kappa, cross_below):
    """CG's bound carries a leading factor of 2, so for the first few steps it is LARGER than
    steepest descent's, and it can even exceed 1. What matters is that it overtakes quickly and
    then wins by a growing margin, so the test checks the crossover point rather than assuming
    CG dominates from step 1."""
    k = np.arange(1, 200000)
    cg = kr.cg_convergence_bound(kappa, k)
    sd = kr.steepest_descent_bound(kappa, k)
    assert np.any(cg <= sd), "the bounds never cross"
    crossover = int(k[np.argmax(cg <= sd)])
    assert crossover <= cross_below, f"crossover at {crossover} for kappa {kappa:.0e}"
    assert np.all(cg[crossover:] <= sd[crossover:] * (1.0 + 1e-12))


@pytest.mark.parametrize("kappa", [1e2, 1e4])
def test_the_cg_bound_reaches_a_target_far_sooner(kappa):
    """The bounds expressed the way they are used: how many steps to a given accuracy. CG needs
    about sqrt(kappa) and steepest descent about kappa, so the ratio grows like sqrt(kappa)."""
    k = np.arange(1, 400000)
    steps = lambda b: int(k[np.argmax(b <= 1e-6)])
    n_cg = steps(kr.cg_convergence_bound(kappa, k))
    n_sd = steps(kr.steepest_descent_bound(kappa, k))
    assert n_cg > 0 and n_sd > 0
    assert n_sd / n_cg > 0.3 * np.sqrt(kappa), f"ratio {n_sd / n_cg:.1f} at kappa {kappa:.0e}"


def test_cg_counts_one_matvec_per_iteration(rng):
    n = 25
    A = spd_with(np.linspace(1.0, 30.0, n), rng)
    op = kr.as_operator(A)
    r = kr.conjugate_gradient(op, rng.standard_normal(n), tol=1e-10, max_iter=10 * n,
                              keep_history=False)
    assert abs(op.n_matvec - r.n_iter) <= 2, f"{op.n_matvec} matvecs for {r.n_iter} iterations"


@pytest.mark.parametrize("n", [4, 12])
def test_cg_rejects_a_length_mismatch(n, rng):
    A = random_spd(n, rng=rng)
    with pytest.raises(ValueError):
        kr.conjugate_gradient(A, np.ones(n + 1))


@pytest.mark.parametrize("n", SIZES)
def test_cg_starting_from_the_answer_stops_at_once(n, rng):
    A = random_spd(n, rng=rng)
    x_true = rng.standard_normal(n)
    r = kr.conjugate_gradient(A, A @ x_true, x0=x_true, tol=1e-8, max_iter=10 * n,
                              keep_history=False)
    assert r.converged and r.n_iter <= 1


# ---------------------------------------------------------------- preconditioners


@pytest.mark.parametrize("n", [4, 12, 30])
@pytest.mark.parametrize("name", ["jacobi", "ssor", "ic"])
def test_every_preconditioner_is_symmetric_and_positive_definite(n, rng, name):
    """PCG needs M SPD, so this checks the operator each one applies, not the formula."""
    A = random_spd(n, rng=rng)
    M = {"jacobi": lambda: kr.jacobi_preconditioner(A),
         "ssor": lambda: kr.ssor_preconditioner(A, 1.2),
         "ic": lambda: kr.incomplete_cholesky(A)}[name]()
    applied = np.column_stack([M(np.eye(n)[:, k]) for k in range(n)])
    np.testing.assert_allclose(applied, applied.T, atol=1e-9 * np.abs(applied).max())
    assert np.all(np.linalg.eigvalsh((applied + applied.T) / 2) > 0.0)


@pytest.mark.parametrize("n", [4, 12, 30])
def test_jacobi_preconditioner_is_the_inverse_diagonal(n, rng):
    A = random_spd(n, rng=rng)
    M = kr.jacobi_preconditioner(A)
    v = rng.standard_normal(n)
    np.testing.assert_allclose(M(v), v / np.diag(A), rtol=1e-12)


def test_jacobi_preconditioner_does_nothing_to_a_constant_diagonal(rng):
    """The second difference matrix has diagonal 2 everywhere, so Jacobi is a scalar multiple
    of the identity and CG is invariant under it. Identical counts, not merely similar."""
    n = 31
    A = second_difference(n)
    b = rng.standard_normal(n)
    plain = kr.conjugate_gradient(A, b, tol=TOL, max_iter=10 * n, keep_history=False)
    jac = kr.conjugate_gradient(A, b, tol=TOL, max_iter=10 * n,
                                M=kr.jacobi_preconditioner(A), keep_history=False)
    assert plain.n_iter == jac.n_iter


@pytest.mark.parametrize("n", [10, 25, 50])
@pytest.mark.parametrize("name", ["jacobi", "ssor", "ic"])
def test_preconditioned_cg_still_gets_the_right_answer(n, rng, name):
    A = random_spd(n, kappa=1e4, rng=rng)
    x_true = rng.standard_normal(n)
    M = {"jacobi": lambda: kr.jacobi_preconditioner(A),
         "ssor": lambda: kr.ssor_preconditioner(A, 1.0),
         "ic": lambda: kr.incomplete_cholesky(A)}[name]()
    r = kr.conjugate_gradient(A, A @ x_true, tol=1e-11, max_iter=20 * n, M=M,
                              keep_history=False)
    assert r.converged, f"{name} did not converge: {r.message}"
    np.testing.assert_allclose(r.x, x_true, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("n", [10, 25, 50])
def test_preconditioning_reduces_the_iteration_count(n, rng):
    A = random_spd(n, kappa=1e5, rng=rng)
    b = rng.standard_normal(n)
    plain = kr.conjugate_gradient(A, b, tol=1e-10, max_iter=40 * n, keep_history=False)
    pre = kr.conjugate_gradient(A, b, tol=1e-10, max_iter=40 * n,
                                M=kr.ssor_preconditioner(A, 1.0), keep_history=False)
    assert pre.converged and plain.converged
    assert pre.n_iter < plain.n_iter, f"{pre.n_iter} against {plain.n_iter}"


@pytest.mark.parametrize("n", [6, 20])
def test_preconditioned_spectrum_is_the_eigenvalues_of_m_inverse_a(n, rng):
    A = random_spd(n, rng=rng)
    M = kr.jacobi_preconditioner(A)
    got = np.sort(kr.preconditioned_spectrum(A, M))
    want = np.sort(np.real(np.linalg.eigvals(np.diag(1.0 / np.diag(A)) @ A)))
    np.testing.assert_allclose(got, want, atol=1e-8)


@pytest.mark.parametrize("n", [6, 20, 40])
def test_a_preconditioner_lowers_the_condition_number_it_is_built_for(n, rng):
    A = random_spd(n, kappa=1e5, rng=rng)
    before = np.linalg.cond(A)
    sp = kr.preconditioned_spectrum(A, kr.ssor_preconditioner(A, 1.0))
    assert sp.max() / sp.min() < before


def test_incomplete_cholesky_reports_a_breakdown_rather_than_returning_nonsense():
    """Kershaw's matrix is SPD and IC(0) fails on it with a negative pivot. The library must
    say so, because a factor built from a negative pivot is not a preconditioner."""
    K = np.array([[3.0, -2.0, 0.0, 2.0],
                  [-2.0, 3.0, -2.0, 0.0],
                  [0.0, -2.0, 3.0, -2.0],
                  [2.0, 0.0, -2.0, 3.0]])
    assert np.all(np.linalg.eigvalsh(K) > 0.0)
    with pytest.raises(np.linalg.LinAlgError):
        kr.incomplete_cholesky(K)


@pytest.mark.parametrize("n", [8, 20])
def test_incomplete_cholesky_keeps_the_sparsity_pattern(n):
    A = second_difference(n)
    L = kr.incomplete_cholesky(A).factor
    outside = (np.abs(np.tril(A)) == 0.0)
    assert np.all(np.abs(L[outside]) == 0.0), "IC(0) created fill it should have dropped"


@pytest.mark.parametrize("n", [8, 20, 40])
def test_ic0_on_a_tridiagonal_matrix_is_exact_cholesky(n, rng):
    """A bidiagonal factor already fits the tridiagonal pattern, so there is no fill to drop
    and IC(0) is the complete factorization. PCG then finishes in one step, which is a useful
    warning against demonstrating preconditioners on tridiagonal problems."""
    A = second_difference(n)
    sp = kr.preconditioned_spectrum(A, kr.incomplete_cholesky(A))
    assert abs(sp.max() / sp.min() - 1.0) < 1e-8
    r = kr.conjugate_gradient(A, rng.standard_normal(n), tol=1e-10, max_iter=10 * n,
                              M=kr.incomplete_cholesky(A), keep_history=False)
    assert r.n_iter <= 2, f"{r.n_iter} steps, expected 1"


@pytest.mark.parametrize("omega", [0.5, 1.0, 1.5, 1.9])
def test_ssor_works_across_kahans_interval(omega, rng):
    n = 20
    A = random_spd(n, kappa=1e3, rng=rng)
    x_true = rng.standard_normal(n)
    r = kr.conjugate_gradient(A, A @ x_true, tol=1e-11, max_iter=20 * n,
                              M=kr.ssor_preconditioner(A, omega), keep_history=False)
    assert r.converged
    np.testing.assert_allclose(r.x, x_true, rtol=1e-5, atol=1e-7)
