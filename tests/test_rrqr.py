"""Tests for `nalib.rrqr`.

Three separate claims are tested here and they must not be conflated.

**Pivoted QR factorizes**, exactly as ordinary QR does, and its diagonal decreases. That part
is a theorem and is asserted at every shape.

**Pivoted QR reveals the rank**, which is a heuristic. So it is tested on matrices where it
works and on `kahan_matrix`, where it provably does not, and the test asserts the *failure*
there. A test suite that only covered the working case would be telling a lie by omission.

**Total least squares is a different problem**, not a better algorithm. The tests check that it
is unbiased where ordinary least squares is biased, and that it is worse conditioned, because
both are true and only the first is usually mentioned.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import leastsquares as ls
from nalib import rrqr


SHAPES = [(1, 1), (3, 1), (5, 3), (8, 8), (30, 8), (60, 12), (150, 30)]
WIDE = [(3, 7), (6, 12), (10, 25)]


# ---------------------------------------------------------------- the factorization itself


@pytest.mark.parametrize("m,n", SHAPES + WIDE)
def test_pivoted_qr_reproduces_the_permuted_matrix(m, n, rng):
    """``A[:, piv] == Q R`` to roundoff, at every shape including wide."""
    A = rng.standard_normal((m, n))
    out = rrqr.qr_column_pivoted(A)
    err = np.linalg.norm(A[:, out.piv] - out.Q @ out.R) / np.linalg.norm(A)
    assert err < 1e-13, f"{m}x{n}: relative factorization error {err:.2e}"


@pytest.mark.parametrize("m,n", SHAPES + WIDE)
def test_pivoted_qr_has_an_orthogonal_q(m, n, rng):
    """Pivoting does not cost orthogonality: it is still Householder underneath."""
    A = rng.standard_normal((m, n))
    Q = rrqr.qr_column_pivoted(A).Q
    err = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))
    assert err < 1e-13, f"{m}x{n}: ||Q^T Q - I|| = {err:.2e}"


@pytest.mark.parametrize("m,n", SHAPES + WIDE)
def test_the_diagonal_is_non_increasing(m, n, rng):
    """This is what "pivoted" means, and it holds by construction rather than by luck."""
    A = rng.standard_normal((m, n))
    d = np.abs(np.diag(rrqr.qr_column_pivoted(A).R))
    if d.size < 2:
        pytest.skip("nothing to compare at a single diagonal entry")
    worst = float(np.max(np.diff(d)))
    assert worst <= 1e-12 * d[0], f"{m}x{n}: diagonal rose by {worst:.2e}"


@pytest.mark.parametrize("m,n", SHAPES)
def test_r_is_exactly_zero_below_the_diagonal(m, n, rng):
    """Assigned, not computed, exactly as in lesson 31."""
    A = rng.standard_normal((m, n))
    R = rrqr.qr_column_pivoted(A).R
    assert np.all(np.tril(R, -1) == 0.0)


@pytest.mark.parametrize("m,n", [(30, 8), (60, 12), (5, 3)])
def test_the_permutation_matrix_agrees_with_the_index_vector(m, n, rng):
    """Two descriptions of the same permutation, checked against each other."""
    A = rng.standard_normal((m, n))
    out = rrqr.qr_column_pivoted(A)
    assert np.allclose(A @ out.permutation_matrix(), A[:, out.piv])


@pytest.mark.parametrize("m,n", [(20, 6), (40, 10)])
def test_a_zero_column_is_pivoted_to_the_end(m, n, rng):
    """A column of zeros carries no information, so it must end up last, not crash."""
    A = rng.standard_normal((m, n))
    A[:, n // 2] = 0.0
    out = rrqr.qr_column_pivoted(A)
    assert out.piv[-1] == n // 2, f"the zero column landed at position {list(out.piv).index(n // 2)}"
    assert out.rank == n - 1


def test_an_all_zero_matrix_is_handled(rng):
    """Rank zero, no exception, and the factorization identity still holds."""
    A = np.zeros((10, 4))
    out = rrqr.qr_column_pivoted(A)
    assert out.rank == 0
    assert np.linalg.norm(A[:, out.piv] - out.Q @ out.R) < 1e-15


# ---------------------------------------------------------------- rank revealing, when it works


@pytest.mark.parametrize("m,n,r", [(40, 10, 4), (60, 12, 7), (20, 20, 1), (25, 6, 5),
                                   (100, 15, 9)])
def test_pivoted_qr_finds_the_rank_of_a_product(m, n, r, rng):
    """``A = B C`` with ``B`` of ``r`` columns has rank ``r``, and both methods should say so."""
    A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    assert rrqr.numerical_rank(A, method="qr") == r
    assert rrqr.numerical_rank(A, method="svd") == r


@pytest.mark.parametrize("m,n,r", [(40, 10, 4), (60, 12, 7), (100, 15, 9)])
def test_the_rank_gap_lands_on_the_true_rank(m, n, r, rng):
    """The largest drop in the diagonal is at the rank, and it is a large drop."""
    A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    gap = rrqr.rank_gap(np.diag(rrqr.qr_column_pivoted(A).R))
    assert gap["index"] == r, f"gap at {gap['index']}, true rank {r}"
    assert gap["ratio"] > 1e6, f"the drop was only a factor of {gap['ratio']:.1e}"


def test_rank_gap_on_a_short_sequence():
    """One entry cannot have a gap, and the routine must say so rather than index past the end."""
    assert rrqr.rank_gap([3.0])["index"] == 1
    assert rrqr.rank_gap([])["index"] == 0


def test_rank_gap_finds_no_large_drop_in_a_smooth_spectrum():
    """A geometrically spaced spectrum has no rank, so the largest ratio is the common ratio
    and nothing more. Reporting an index here would be reporting noise."""
    v = np.geomspace(1.0, 1e-8, 20)
    gap = rrqr.rank_gap(v)
    common = float(v[0] / v[1])
    assert abs(gap["ratio"] - common) < 1e-9 * common, \
        f"ratio {gap['ratio']:.4f} against a common ratio of {common:.4f}"


# ---------------------------------------------------------------- rank revealing, when it fails


@pytest.mark.parametrize("n", [10, 20, 30, 40])
def test_kahan_defeats_the_pivoting(n):
    """**Zero swaps.** Every column of the Kahan matrix has the same norm, so pivoting has no
    reason to move anything, and the matrix comes back exactly as it went in."""
    K = rrqr.kahan_matrix(n)
    out = rrqr.qr_column_pivoted(K)
    assert np.array_equal(out.piv, np.arange(n)), f"n={n}: pivoting reordered the columns"


@pytest.mark.parametrize("n", [20, 30, 40])
def test_kahan_hides_its_conditioning_from_pivoted_qr(n):
    """The diagonal of R understates the true condition number by orders of magnitude, and the
    understatement grows with n. This is the counterexample, asserted as a counterexample."""
    K = rrqr.kahan_matrix(n)
    d = np.abs(np.diag(rrqr.qr_column_pivoted(K).R))
    s = np.linalg.svd(K, compute_uv=False)
    kappa_qr = d[0] / d[-1]
    kappa_true = s[0] / s[-1]
    assert kappa_true > 1e5 * kappa_qr, \
        f"n={n}: QR said {kappa_qr:.2e}, the truth is {kappa_true:.2e}"


@pytest.mark.parametrize("n", [24, 28, 32])
def test_kahan_makes_the_two_rank_verdicts_disagree(n):
    """At these sizes pivoted QR calls the matrix full rank and the SVD does not."""
    K = rrqr.kahan_matrix(n)
    assert rrqr.numerical_rank(K, method="qr") > rrqr.numerical_rank(K, method="svd")


def test_kahan_rejects_a_bad_size():
    with pytest.raises(ValueError):
        rrqr.kahan_matrix(0)


def test_numerical_rank_rejects_an_unknown_method(rng):
    with pytest.raises(ValueError):
        rrqr.numerical_rank(rng.standard_normal((5, 3)), method="pivoted")


# ---------------------------------------------------------------- subset selection


@pytest.mark.parametrize("m,n,k", [(30, 8, 3), (40, 10, 5), (25, 9, 4), (50, 7, 2), (12, 4, 4)])
def test_subset_selection_returns_k_distinct_columns(m, n, k, rng):
    A = rng.standard_normal((m, n))
    out = rrqr.subset_selection(A, k)
    assert out["columns"].size == k
    assert len(set(out["columns"].tolist())) == k
    assert np.all((0 <= out["columns"]) & (out["columns"] < n))


@pytest.mark.parametrize("m,n,k", [(30, 8, 3), (40, 10, 5), (25, 9, 4)])
def test_selecting_every_column_is_the_whole_matrix(m, n, k, rng):
    """Taking all n columns must reproduce A exactly, so the residual is zero."""
    A = rng.standard_normal((m, n))
    out = rrqr.subset_selection(A, n)
    assert out["residual"] < 1e-13, f"residual {out['residual']:.2e} when nothing was dropped"


@pytest.mark.parametrize("m,n,k", [(30, 8, 3), (25, 9, 4), (20, 7, 2)])
def test_greedy_is_never_better_than_exhaustive(m, n, k, rng):
    """It cannot be, by definition, and a violation would mean one of the two is wrong."""
    A = rng.standard_normal((m, n))
    greedy = rrqr.subset_selection(A, k)["sigma_min"]
    best = rrqr.best_subset(A, k)["sigma_min"]
    assert greedy <= best * (1.0 + 1e-12), f"greedy {greedy:.6e} beat the optimum {best:.6e}"


def test_greedy_is_sometimes_strictly_worse(rng):
    """On correlated columns the greedy choice is often not optimal, and pretending otherwise
    would misrepresent what pivoted QR promises. Measured: optimal in 3 of 60 trials, with a
    median sigma_min ratio of 0.90."""
    worse = 0
    for trial in range(30):
        gen = np.random.default_rng(4000 + trial)
        B = gen.standard_normal((40, 3))
        A = B @ gen.standard_normal((3, 10)) + 0.05 * gen.standard_normal((40, 10))
        if rrqr.subset_selection(A, 4)["sigma_min"] < 0.999 * rrqr.best_subset(A, 4)["sigma_min"]:
            worse += 1
    assert worse > 10, f"greedy was suboptimal in only {worse} of 30 correlated trials"


@pytest.mark.parametrize("k", [0, -1, 20])
def test_subset_selection_rejects_a_bad_k(k, rng):
    with pytest.raises(ValueError):
        rrqr.subset_selection(rng.standard_normal((30, 8)), k)


def test_best_subset_refuses_an_absurd_enumeration(rng):
    with pytest.raises(ValueError):
        rrqr.best_subset(rng.standard_normal((60, 40)), 20)


# ---------------------------------------------------------------- total least squares


@pytest.mark.parametrize("m,n", [(20, 3), (40, 5), (100, 8), (6, 5), (30, 1)])
def test_tls_solves_a_consistent_system_exactly(m, n, rng):
    """With no noise anywhere, TLS and ordinary least squares must agree with the truth."""
    A = rng.standard_normal((m, n))
    x = rng.standard_normal(n)
    out = rrqr.total_least_squares(A, A @ x)
    assert out.exists
    assert np.linalg.norm(out.x - x) / np.linalg.norm(x) < 1e-10


@pytest.mark.parametrize("m,n", [(40, 5), (100, 8), (25, 4)])
def test_the_closed_form_agrees_with_the_svd_form(m, n, rng):
    """Two derivations of the same quantity, so any disagreement is a bug in one of them."""
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    out = rrqr.total_least_squares(A, b)
    if not out.exists:
        pytest.skip("no TLS solution for this draw")
    rel = (np.linalg.norm(out.x - rrqr.tls_closed_form(A, b))
           / max(np.linalg.norm(out.x), 1e-300))
    assert rel < 1e-6, f"the two forms differ by {rel:.2e}"


@pytest.mark.parametrize("m,n", [(20, 3), (40, 5), (100, 8)])
def test_the_correction_is_the_smallest_singular_value(m, n, rng):
    """Eckart-Young: the Frobenius norm of the rank one correction that makes [A|b] singular
    is exactly sigma_last. This is an identity, so it is asserted tightly."""
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    out = rrqr.total_least_squares(A, b)
    if not out.exists:
        pytest.skip("no TLS solution for this draw")
    assert abs(out.correction - out.sigma_last) < 1e-12 * max(out.sigma_last, 1.0)
    combined = np.hypot(out.residual_A, out.residual_b)
    assert abs(combined - out.correction) < 1e-12 * max(out.correction, 1.0)


@pytest.mark.parametrize("m,n", [(20, 3), (40, 5), (100, 8), (12, 6)])
def test_interlacing_holds(m, n, rng):
    """sigma_last([A|b]) <= sigma_n(A) always, which is Cauchy interlacing. If this ever fails
    the singular values are being read off the wrong matrix."""
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    out = rrqr.total_least_squares(A, b)
    assert out.sigma_last <= out.sigma_n_of_A * (1.0 + 1e-12)


def test_tls_reports_no_solution_when_b_is_orthogonal_to_the_range(rng):
    """Then the smallest singular vector has no component along b, so it cannot be scaled.
    The routine must say so rather than divide by a number that is zero."""
    A = rng.standard_normal((40, 5))
    b = rng.standard_normal(40)
    Q, _ = np.linalg.qr(A)
    b = b - Q @ (Q.T @ b)
    out = rrqr.total_least_squares(A, b)
    assert not out.exists
    assert np.all(np.isnan(out.x))


def test_tls_reports_no_solution_when_a_is_rank_deficient(rng):
    A = rng.standard_normal((40, 3)) @ rng.standard_normal((3, 5))
    out = rrqr.total_least_squares(A, rng.standard_normal(40))
    assert not out.exists


def test_tls_rejects_a_length_mismatch(rng):
    with pytest.raises(ValueError):
        rrqr.total_least_squares(rng.standard_normal((20, 4)), rng.standard_normal(19))


def test_tls_closed_form_rejects_a_length_mismatch(rng):
    with pytest.raises(ValueError):
        rrqr.tls_closed_form(rng.standard_normal((20, 4)), rng.standard_normal(21))


# ---------------------------------------------------------------- the bias TLS exists to remove


@pytest.mark.parametrize("noise", [0.1, 0.25, 0.5, 1.0])
def test_the_attenuation_formula_predicts_the_ordinary_bias(noise):
    """Ordinary least squares converges to ``beta * s^2/(s^2+e^2)``, not to ``beta``. Measured
    agreement to three decimal places at every noise level tested."""
    slopes = []
    for trial in range(120):
        gen = np.random.default_rng(9000 + trial)
        data = rrqr.errors_in_variables(400, 1, noise, 0.05, gen, coefficients=[2.0])
        slopes.append(ls.solve_qr(data["A"], data["b"]).x[0])
    measured = float(np.mean(slopes)) / 2.0
    predicted = rrqr.attenuation_factor(noise, 1.0)
    assert abs(measured - predicted) < 0.02, \
        f"noise {noise}: measured shrinkage {measured:.4f}, predicted {predicted:.4f}"


@pytest.mark.parametrize("m", [200, 2000])
def test_more_data_does_not_remove_the_bias(m):
    """The whole point: it is a bias, not a variance. More data makes it more precisely wrong."""
    slopes = []
    for trial in range(40):
        gen = np.random.default_rng(9500 + trial)
        data = rrqr.errors_in_variables(m, 1, 0.5, 0.05, gen, coefficients=[2.0])
        slopes.append(ls.solve_qr(data["A"], data["b"]).x[0])
    assert abs(float(np.mean(slopes)) - 1.6) < 0.05, \
        f"m={m}: mean slope {np.mean(slopes):.4f}, expected the biased limit 1.6"


@pytest.mark.parametrize("m,n,noise", [(200, 1, 0.5), (200, 3, 0.3), (500, 2, 0.2)])
def test_tls_beats_ordinary_least_squares_when_a_is_noisy(m, n, noise):
    """Measured: TLS is closer in 98 percent of trials at these noise levels."""
    better = 0
    for trial in range(40):
        gen = np.random.default_rng(9800 + trial)
        data = rrqr.errors_in_variables(m, n, noise, 0.1, gen)
        c = data["coefficients"]
        e_ols = np.linalg.norm(ls.solve_qr(data["A"], data["b"]).x - c)
        out = rrqr.total_least_squares(data["A"], data["b"])
        if out.exists and np.linalg.norm(out.x - c) < e_ols:
            better += 1
    assert better >= 30, f"TLS was closer in only {better} of 40 trials"


def test_errors_in_variables_rejects_bad_sizes(rng):
    with pytest.raises(ValueError):
        rrqr.errors_in_variables(3, 5, 0.1, 0.1, rng)
    with pytest.raises(ValueError):
        rrqr.errors_in_variables(30, 4, 0.1, 0.1, rng, coefficients=[1.0, 2.0])


def test_attenuation_factor_rejects_all_zero():
    with pytest.raises(ValueError):
        rrqr.attenuation_factor(0.0, 0.0)


# ---------------------------------------------------------------- TLS conditioning


@pytest.mark.parametrize("m,n", [(40, 5), (100, 8), (25, 4)])
def test_tls_conditioning_reports_a_gap_within_its_bounds(m, n, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    c = rrqr.tls_conditioning(A, b)
    assert c["gap"] >= -1e-12
    assert 0.0 <= c["relative_gap"] <= 1.0 + 1e-12
    assert c["kappa_tls"] >= c["kappa_A"] * (1.0 - 1e-9)


def test_tls_is_worse_conditioned_than_the_matrix_it_came_from(rng):
    """A perfectly conditioned A can still give a hopeless TLS problem, because what matters is
    the gap between sigma_n(A) and sigma_last([A|b]), not kappa(A). Measured on one draw:
    kappa(A) = 2.0 and kappa_tls = 1.6e5, an amplification of 7.8e4."""
    gen = np.random.default_rng(9)
    A = gen.standard_normal((40, 5))
    b = gen.standard_normal(40)
    c = rrqr.tls_conditioning(A, b)
    assert c["kappa_A"] < 10.0, f"kappa(A) = {c['kappa_A']:.2f}, expected a well conditioned A"
    assert c["amplification"] > 1e3, f"amplification only {c['amplification']:.2e}"


def test_tls_conditioning_rejects_a_length_mismatch(rng):
    with pytest.raises(ValueError):
        rrqr.tls_conditioning(rng.standard_normal((20, 4)), rng.standard_normal(21))
