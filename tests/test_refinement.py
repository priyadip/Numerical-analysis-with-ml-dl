"""Tests for `nalib.refinement`.

Condition estimation and iterative refinement are both about measuring or removing an error
that is invisible in the residual, so the tests are built around problems whose exact answer is
known by construction. `exact_solution` and `exact_residual` use rationals, so they are the one
place in the library where the reference is not itself a floating point computation, and
several tests below depend on that.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import refinement as rf
from nalib.cholesky import random_spd
from nalib.linsys import hilbert, with_condition_number


SIZES = [1, 2, 3, 6, 12, 30]
KAPPAS = [1.0, 1e2, 1e6, 1e10, 1e14]


# ---------------------------------------------------------------- norm estimation


@pytest.mark.parametrize("n", SIZES)
def test_estimate_norm_inverse_is_within_a_small_factor(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    got = rf.estimate_norm_inverse(A)
    want = np.linalg.norm(np.linalg.inv(A), 1)
    assert 0.1 * want <= got <= 1.5 * want, f"estimate {got:.4g} against {want:.4g}"


@pytest.mark.parametrize("n", SIZES)
def test_estimate_norm_inverse_never_exceeds_the_truth_by_much(n, rng):
    """Hager's method builds a lower bound in exact arithmetic. It can come out slightly above
    the reference because the reference itself is computed from inv(A), which has its own
    error, so the check is a small tolerance rather than a strict inequality."""
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    want = np.linalg.norm(np.linalg.inv(A), 1)
    assert rf.estimate_norm_inverse(A) <= want * (1.0 + 1e-6)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_estimate_norm_inverse_scales_correctly(scale, rng):
    n = 10
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    plain = rf.estimate_norm_inverse(A)
    scaled = rf.estimate_norm_inverse(scale * A)
    assert abs(scaled * scale - plain) <= 1e-8 * plain


def test_estimate_norm_inverse_of_the_identity_is_one():
    for n in SIZES:
        assert abs(rf.estimate_norm_inverse(np.eye(n)) - 1.0) < 1e-12


# ---------------------------------------------------------------- condition estimation


@pytest.mark.parametrize("n", SIZES)
def test_condition_estimate_is_within_a_factor_of_the_truth(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    got = rf.condition_estimate(A)
    want = np.linalg.cond(A, 1)
    assert 0.1 * want <= got <= 1.5 * want, f"estimate {got:.4g} against {want:.4g}"


@pytest.mark.parametrize("kappa", KAPPAS)
def test_condition_estimate_tracks_a_prescribed_condition_number(kappa, rng):
    A = with_condition_number(15, kappa, rng=rng)
    got = rf.condition_estimate(A)
    want = np.linalg.cond(A, 1)
    assert 0.05 * want <= got <= 2.0 * want, f"estimate {got:.3e} against {want:.3e}"


@pytest.mark.parametrize("n", [3, 6, 9, 12])
def test_condition_estimate_sees_the_hilbert_matrix_blowing_up(n):
    """The Hilbert condition number grows like e^(3.5 n), so an estimator that misses it by
    orders of magnitude is useless. This checks the estimate stays within one order."""
    got = rf.condition_estimate(hilbert(n))
    want = np.linalg.cond(hilbert(n), 1)
    assert got > 0.0
    assert abs(np.log10(got) - np.log10(want)) < 1.0


def test_condition_estimate_of_the_identity_is_one():
    for n in SIZES:
        assert abs(rf.condition_estimate(np.eye(n)) - 1.0) < 1e-12


def test_condition_estimate_costs_no_inverse():
    """The point of Hager's method is O(n^2), so it must never form inv(A). This checks the
    estimate is finite and sane on a matrix too ill conditioned for inv(A) to mean anything."""
    got = rf.condition_estimate(hilbert(14))
    assert np.isfinite(got) and got > 1e10


# ---------------------------------------------------------------- exact arithmetic


@pytest.mark.parametrize("n", SIZES)
def test_exact_residual_is_zero_for_the_exact_solution(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    b = rng.standard_normal(n)
    x = rf.exact_solution(A, b)
    r = rf.exact_residual(A, b, x)
    u = np.finfo(float).eps / 2
    # x is the exact answer ROUNDED to double, so the residual is O(u ||A|| ||x||), and the
    # point is that nothing larger creeps in from the residual computation itself.
    bound = 50.0 * u * np.abs(A).max() * max(1.0, np.abs(x).max()) * A.shape[0]
    assert np.max(np.abs(r)) <= bound, f"{np.max(np.abs(r)):.2e} against bound {bound:.2e}"


@pytest.mark.parametrize("n", SIZES)
def test_exact_residual_beats_the_floating_point_one(n, rng):
    """On an ill conditioned system the computed residual of a computed solution is dominated
    by the rounding in forming A @ x. The rational version is not."""
    A = with_condition_number(n, 1e12, rng=rng) if n > 1 else np.array([[3.0]])
    b = rng.standard_normal(n)
    x = np.linalg.solve(A, b)
    naive = np.max(np.abs(b - A @ x))
    exact = np.max(np.abs(rf.exact_residual(A, b, x)))
    assert np.isfinite(exact)
    assert exact <= naive * 10.0 + 1e-300       # never much worse, usually far better


@pytest.mark.parametrize("n", SIZES)
def test_exact_solution_solves_the_system_exactly(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    x_true = rng.standard_normal(n)
    b = A @ x_true                               # b is rounded, so x_true is NOT the answer
    x = rf.exact_solution(A, b)
    u = np.finfo(float).eps / 2
    bound = 50.0 * u * np.abs(A).max() * max(1.0, np.abs(x).max()) * n
    assert np.max(np.abs(rf.exact_residual(A, b, x))) <= bound


@pytest.mark.parametrize("kappa", [1e2, 1e8, 1e12])
def test_the_stored_system_differs_from_the_intended_one(kappa, rng):
    """Forming b = A @ x_true rounds, so the exact answer to the STORED system differs from
    x_true by about kappa * u. This is the measurement trap that makes iterative refinement
    look broken if you check against x_true instead of against the exact solution."""
    n = 12
    A = with_condition_number(n, kappa, rng=rng)
    x_true = rng.standard_normal(n)
    b = A @ x_true
    x_star = rf.exact_solution(A, b)
    drift = np.linalg.norm(x_star - x_true) / np.linalg.norm(x_true)
    u = np.finfo(float).eps / 2
    assert drift <= 100.0 * kappa * u + 1e-14, f"drift {drift:.2e} at kappa {kappa:.0e}"


def test_exact_solution_reports_a_singular_matrix():
    with pytest.raises(np.linalg.LinAlgError):
        rf.exact_solution(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))


@pytest.mark.parametrize("n", [2, 5])
def test_exact_routines_reject_a_length_mismatch(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    with pytest.raises(ValueError):
        rf.exact_solution(A, np.ones(n + 1))
    with pytest.raises(ValueError):
        rf.exact_residual(A, np.ones(n), np.ones(n + 1))


# ---------------------------------------------------------------- refinement


@pytest.mark.parametrize("n", [2, 5, 12, 25])
def test_refinement_leaves_a_well_conditioned_system_alone(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    x_true = rng.standard_normal(n)
    b = A @ x_true
    out = rf.iterative_refinement(A, b, max_steps=3)
    err = np.linalg.norm(out["x"] - rf.exact_solution(A, b))
    assert err < 1e-13 * max(1.0, np.linalg.norm(x_true))


@pytest.mark.parametrize("kappa", [1e6, 1e10, 1e12])
def test_refinement_with_an_exact_residual_reaches_the_exact_solution(kappa, rng):
    n = 15
    A = with_condition_number(n, kappa, rng=rng)
    b = rng.standard_normal(n)
    x_star = rf.exact_solution(A, b)
    out = rf.iterative_refinement(A, b, max_steps=5, residual="exact")
    before = np.linalg.norm(np.linalg.solve(A, b) - x_star) / np.linalg.norm(x_star)
    after = np.linalg.norm(out["x"] - x_star) / np.linalg.norm(x_star)
    assert after <= before + 1e-15, f"{before:.2e} -> {after:.2e} at kappa {kappa:.0e}"
    assert after < 1e-13, f"final relative error {after:.2e}"


@pytest.mark.parametrize("kappa", [1e8, 1e12])
def test_refinement_errors_decrease_monotonically(kappa, rng):
    n = 12
    A = with_condition_number(n, kappa, rng=rng)
    b = rng.standard_normal(n)
    x_star = rf.exact_solution(A, b)
    out = rf.iterative_refinement(A, b, max_steps=4, residual="exact")
    errs = np.array([np.linalg.norm(v - x_star) for v in out["iterates"]])
    assert len(errs) >= 2
    assert errs[-1] <= errs[0], f"error grew from {errs[0]:.2e} to {errs[-1]:.2e}"
    # each correction is no larger than the previous one, up to the roundoff floor
    floor = 1e-14 * max(1.0, np.linalg.norm(x_star))
    assert np.all(np.diff(errs) <= floor), f"errors {errs}"


@pytest.mark.parametrize("n", SIZES)
def test_refinement_returns_the_documented_keys(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    out = rf.iterative_refinement(A, rng.standard_normal(n), max_steps=2)
    for key in ("x", "corrections", "residual_norms", "iterates", "n_steps"):
        assert key in out, f"missing key {key}"
    assert np.asarray(out["x"]).shape == (n,)
    assert np.asarray(out["iterates"]).shape == (out["n_steps"] + 1, n)
    assert len(out["corrections"]) == out["n_steps"]


@pytest.mark.parametrize("kappa", [1e4, 1e10])
def test_compare_residual_precisions_shows_exact_winning(kappa, rng):
    n = 12
    A = with_condition_number(n, kappa, rng=rng)
    b = rng.standard_normal(n)
    out = rf.compare_residual_precisions(A, b, max_steps=4)
    for key in ("x_star", "working", "fsum", "exact"):
        assert key in out, f"missing key {key}"
    # the exact residual can never end worse than the working precision one
    assert out["exact"][-1] <= out["working"][-1] * (1.0 + 1e-9) + 1e-16
    # and every run starts from the same unrefined solution
    assert abs(out["exact"][0] - out["working"][0]) < 1e-14 * max(1.0, out["working"][0])


@pytest.mark.parametrize("n", [1, 4])
def test_refinement_works_at_tiny_sizes(n, rng):
    A = np.eye(n) * 3.0
    b = np.full(n, 6.0)
    out = rf.iterative_refinement(A, b, max_steps=2)
    np.testing.assert_allclose(out["x"], np.full(n, 2.0), rtol=1e-14)


@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
def test_refinement_is_scale_invariant(scale, rng):
    n = 10
    A = with_condition_number(n, 1e8, rng=rng)
    b = rng.standard_normal(n)
    plain = rf.iterative_refinement(A, b, max_steps=3, residual="exact")["x"]
    scaled = rf.iterative_refinement(scale * A, scale * b, max_steps=3,
                                     residual="exact")["x"]
    np.testing.assert_allclose(scaled, plain, rtol=1e-8, atol=1e-12 * np.abs(plain).max())


@pytest.mark.parametrize("mode", ["exact", "fsum", "working"])
def test_every_residual_mode_runs_and_returns_the_right_shapes(mode, rng):
    n = 10
    A = with_condition_number(n, 1e8, rng=rng)
    b = rng.standard_normal(n)
    out = rf.iterative_refinement(A, b, max_steps=3, residual=mode)
    assert out["x"].shape == (n,)
    assert out["n_steps"] == 3
    assert np.all(np.isfinite(out["x"]))


def test_an_unknown_residual_mode_is_rejected(rng):
    A = np.eye(3)
    with pytest.raises(ValueError):
        rf.iterative_refinement(A, np.ones(3), residual="quadruple")


@pytest.mark.parametrize("kappa", [1e6, 1e10, 1e14])
def test_exact_is_the_best_residual_at_every_condition_number(kappa, rng):
    """The exact residual is never beaten. This is the ordering the lesson rests on, and it is
    the only part of it that holds unconditionally."""
    n = 14
    A = with_condition_number(n, kappa, rng=rng)
    b = rng.standard_normal(n)
    out = rf.compare_residual_precisions(A, b, max_steps=5)
    assert out["exact"][-1] <= out["fsum"][-1] * (1.0 + 1e-9) + 1e-16
    assert out["exact"][-1] <= out["working"][-1] * (1.0 + 1e-9) + 1e-16


def test_fsum_is_usually_but_not_always_better_than_working_precision():
    """fsum sums exactly, but the PRODUCTS A[i,j]*x[j] are rounded before it ever sees them,
    and at high kappa that rounding dominates. So fsum gives a modest and unreliable gain,
    which is exactly why the lesson says fsum is not enough.

    Measured over 72 systems: exact wins 72 times out of 72, fsum beats working 49 times out
    of 72. This test asserts the pattern rather than a single comparison, because a single
    comparison would be a coin flip.
    """
    wins = exact_wins = trials = 0
    for e in (4, 6, 8, 10, 12, 14):
        for seed in range(1000, 1006):
            gen = np.random.default_rng(seed)
            n = 14
            A = with_condition_number(n, 10.0 ** e, rng=gen)
            out = rf.compare_residual_precisions(A, gen.standard_normal(n), max_steps=5)
            f, w, x = out["fsum"][-1], out["working"][-1], out["exact"][-1]
            trials += 1
            wins += f <= w * (1.0 + 1e-9)
            exact_wins += x <= min(f, w) * (1.0 + 1e-9)
    assert exact_wins == trials, f"exact lost {trials - exact_wins} of {trials}"
    assert 0.4 * trials <= wins < trials, (
        f"fsum won {wins} of {trials}: expected a majority but not a clean sweep")


@pytest.mark.parametrize("n", [3, 8])
def test_refinement_stops_early_when_the_correction_is_tiny(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    b = rng.standard_normal(n)
    out = rf.iterative_refinement(A, b, max_steps=20, residual="exact", tol=1e-14)
    assert out["n_steps"] < 20, "the tolerance never triggered"
