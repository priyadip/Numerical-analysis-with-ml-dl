"""Tests for `nalib.nonsymmetric`.

The two properties that separate GMRES from everything else, optimality and monotonicity, are
theorems, so they are tested as theorems: at every size, on several matrix families, to
roundoff rather than to a generous tolerance. The short-recurrence methods have no such
guarantees, and the tests record what they actually do instead of pretending otherwise, which
includes recording that they break down.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import nonsymmetric as ns
from nalib.banded import second_difference
from nalib.cholesky import random_spd
from nalib.krylov import LinearOperator


SIZES = [1, 2, 3, 5, 9, 20, 47]
SCALES = [1e-8, 1e-2, 1.0, 1e5, 1e9]
TOL = 1e-10


def well_posed(n, rng):
    """A nonsymmetric matrix that every method here can solve, at any size."""
    return rng.standard_normal((n, n)) + (n + 2.0) * np.eye(n)


# ---------------------------------------------------------------- Givens rotations


@pytest.mark.parametrize("a,b", [(3.0, 4.0), (0.0, 1.0), (1.0, 0.0), (-2.0, 5.0),
                                 (1e-30, 1e-30), (7.0, -7.0)])
def test_givens_rotates_the_pair_onto_the_axis(a, b):
    c, s = ns.givens(a, b)
    x, y = ns.apply_givens(c, s, a, b)
    assert abs(y) <= 1e-14 * max(1.0, abs(a), abs(b))
    assert abs(abs(x) - np.hypot(a, b)) <= 1e-13 * max(1.0, np.hypot(a, b))


@pytest.mark.parametrize("a,b", [(3.0, 4.0), (1e200, 1e200), (1e-200, 1e-200),
                                 (1e300, 1.0), (1.0, 1e-300)])
def test_givens_survives_the_whole_exponent_range(a, b):
    """The naive formula forms a^2 + b^2 and so overflows above 1.3e154 and underflows below
    1e-162. Scaling first removes both, and this checks the identity c^2 + s^2 = 1 rather than
    a reference value, since the reference itself would overflow."""
    c, s = ns.givens(a, b)
    assert np.isfinite(c) and np.isfinite(s)
    assert abs(c * c + s * s - 1.0) < 1e-13, f"c^2+s^2 = {c*c+s*s} at ({a:.0e}, {b:.0e})"


@pytest.mark.parametrize("a,b", [(3.0, 4.0), (-1.0, 2.0), (5.0, -0.5)])
def test_apply_givens_is_an_isometry(a, b):
    c, s = ns.givens(a, b)
    for x, y in [(1.0, 0.0), (0.0, 1.0), (2.5, -3.5)]:
        u, v = ns.apply_givens(c, s, x, y)
        assert abs(np.hypot(u, v) - np.hypot(x, y)) < 1e-13 * max(1.0, np.hypot(x, y))


# ---------------------------------------------------------------- GMRES correctness


@pytest.mark.parametrize("n", SIZES)
def test_gmres_solves_a_nonsymmetric_system(n, rng):
    A = well_posed(n, rng)
    x_true = rng.standard_normal(n)
    r = ns.gmres(A, A @ x_true, tol=1e-12, max_iter=4 * n)
    assert r.converged, r.message
    np.testing.assert_allclose(r.x, x_true, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("scale", SCALES)
def test_gmres_is_scale_invariant(scale, rng):
    n = 20
    A = well_posed(n, rng)
    x_true = rng.standard_normal(n)
    plain = ns.gmres(A, A @ x_true, tol=1e-11, max_iter=4 * n)
    scaled = ns.gmres(scale * A, scale * (A @ x_true), tol=1e-11, max_iter=4 * n)
    assert plain.converged and scaled.converged
    assert abs(plain.n_iter - scaled.n_iter) <= 1
    np.testing.assert_allclose(scaled.x, plain.x, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n", [3, 8, 20, 40])
def test_gmres_terminates_within_n_steps(n, rng):
    """Exact in exact arithmetic, since K_n is the whole space. Allow one extra step for the
    convergence test to notice."""
    A = well_posed(n, rng)
    r = ns.gmres(A, rng.standard_normal(n), tol=1e-10, max_iter=2 * n)
    assert r.converged and r.n_iter <= n + 1, f"{r.n_iter} steps at n = {n}"


@pytest.mark.parametrize("n", [5, 15, 40])
def test_gmres_residual_never_increases(n, rng):
    """Proposition 27.2. The searched space only grows, so the minimum only falls. This is a
    theorem, so the tolerance is roundoff."""
    for A in (well_posed(n, rng), second_difference(n),
              ns.convection_diffusion(n, 3.0),
              ns.same_spectrum_family(np.arange(1.0, n + 1), 5.0)):
        r = ns.gmres(A, rng.standard_normal(n), tol=1e-12, max_iter=n)
        assert np.all(np.diff(r.residual_norms) <= 1e-12 * r.residual_norms[0])


@pytest.mark.parametrize("n", [5, 15, 40])
def test_the_reported_residual_is_the_true_one(n, rng):
    """It comes out of the Givens rotation without forming a residual, so it could easily drift
    from reality. It does not."""
    A = well_posed(n, rng)
    b = rng.standard_normal(n)
    r = ns.gmres(A, b, tol=1e-13, max_iter=n, keep_history=True)
    true = np.array([np.linalg.norm(b - A @ xk) for xk in r.iterates])
    got = r.residual_norms[:true.size]
    assert np.abs(true - got).max() < 1e-9 * max(1.0, true[0])


@pytest.mark.parametrize("n", [4, 12, 30])
def test_gmres_beats_or_matches_any_other_krylov_iterate(n, rng):
    """Proposition 27.3. Compare against the best residual any vector in x_0 + K_m can give,
    computed independently by least squares on the naive Krylov basis."""
    A = well_posed(n, rng)
    b = rng.standard_normal(n)
    r = ns.gmres(A, b, tol=1e-14, max_iter=n, keep_history=True)
    for m in range(1, min(6, n) + 1):
        K = np.zeros((n, m))
        K[:, 0] = b
        for j in range(1, m):
            K[:, j] = A @ K[:, j - 1]
        AK = A @ K
        y, *_ = np.linalg.lstsq(AK, b, rcond=None)
        best = float(np.linalg.norm(b - AK @ y))
        assert r.residual_norms[m] <= best * (1.0 + 1e-6) + 1e-10, (
            f"GMRES gave {r.residual_norms[m]:.3e} where {best:.3e} was reachable at m={m}")


@pytest.mark.parametrize("n", [4, 12])
def test_gmres_rejects_bad_arguments(n, rng):
    A = well_posed(n, rng)
    with pytest.raises(ValueError):
        ns.gmres(A, np.ones(n + 1))
    with pytest.raises(ValueError):
        ns.gmres(A, np.ones(n), x0=np.ones(n + 1))
    with pytest.raises(ValueError):
        ns.gmres(A, np.ones(n), restart=0)


@pytest.mark.parametrize("n", SIZES)
def test_gmres_starting_from_the_answer_stops_at_once(n, rng):
    A = well_posed(n, rng)
    x_true = rng.standard_normal(n)
    r = ns.gmres(A, A @ x_true, x0=x_true, tol=1e-8, max_iter=4 * n)
    assert r.converged and r.n_iter == 0


@pytest.mark.parametrize("n", [6, 20, 40])
def test_gmres_works_matrix_free(n, rng):
    def apply(v):
        y = 2.0 * v
        y[:-1] -= v[1:]
        y[1:] -= v[:-1]
        return y

    op = LinearOperator(apply, shape=(n, n))
    x_true = rng.standard_normal(n)
    b = second_difference(n) @ x_true
    r = ns.gmres(op, b, tol=1e-11, max_iter=2 * n)
    assert r.converged
    np.testing.assert_allclose(r.x, x_true, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------- restarting


@pytest.mark.parametrize("k", [1, 2, 5, 10])
@pytest.mark.parametrize("n", [12, 30])
def test_restarted_gmres_gets_the_right_answer_when_it_converges(k, n, rng):
    A = well_posed(n, rng)
    x_true = rng.standard_normal(n)
    r = ns.gmres(A, A @ x_true, tol=1e-11, restart=k, max_iter=100 * n)
    if r.converged:
        np.testing.assert_allclose(r.x, x_true, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n", [12, 30])
def test_restarted_gmres_is_still_monotone(n, rng):
    """Restarting destroys optimality but not monotonicity: each cycle starts from the current
    iterate, so the residual cannot jump up."""
    A = ns.convection_diffusion(n, 2.5)
    r = ns.gmres(A, rng.standard_normal(n), tol=1e-10, restart=5, max_iter=50 * n)
    assert np.all(np.diff(r.residual_norms) <= 1e-9 * r.residual_norms[0])


@pytest.mark.parametrize("n", [10, 20, 40])
def test_restarting_stagnates_on_the_cyclic_shift(n):
    """Full GMRES makes no progress for n-1 steps and then solves exactly, so every restarted
    version is thrown out before the one useful step arrives."""
    A = ns.stagnation_matrix(n)
    b = np.zeros(n)
    b[0] = 1.0
    full = ns.gmres(A, b, tol=1e-10, max_iter=n)
    assert full.converged and full.n_iter == n
    assert np.allclose(full.residual_norms[:n], 1.0, atol=1e-9)
    for k in (2, 5, max(2, n // 2)):
        rr = ns.gmres(A, b, tol=1e-10, restart=k, max_iter=20 * n)
        assert not rr.converged
        assert abs(rr.residual_norms[-1] - 1.0) < 1e-9


@pytest.mark.parametrize("n", [3, 9, 25])
def test_the_cyclic_shift_has_unit_eigenvalues(n):
    """Its spectrum is the n-th roots of unity, evenly spread on the unit circle, so nothing
    about the eigenvalues warns you about the stagnation."""
    A = ns.stagnation_matrix(n)
    np.testing.assert_allclose(np.abs(np.linalg.eigvals(A)), np.ones(n), atol=1e-10)


def test_stagnation_matrix_rejects_a_size_below_two():
    with pytest.raises(ValueError):
        ns.stagnation_matrix(1)


@pytest.mark.parametrize("n", [30, 60])
def test_a_bigger_restart_is_never_worse(n, rng):
    A = ns.convection_diffusion(n, 0.0)
    b = np.ones(n)
    counts = []
    for k in (5, 10, 20):
        r = ns.gmres(A, b, tol=1e-8, restart=k, max_iter=200 * n)
        counts.append(r.n_iter if r.converged else 10 ** 9)
    assert counts[0] >= counts[1] >= counts[2], f"counts {counts}"


# ---------------------------------------------------------------- non-normality


@pytest.mark.parametrize("gamma", [0.0, 1.0, 5.0, 30.0])
@pytest.mark.parametrize("n", [5, 20, 40])
def test_the_family_really_does_keep_its_eigenvalues(gamma, n):
    lam = np.arange(1.0, n + 1)
    A = ns.same_spectrum_family(lam, gamma)
    got = np.sort(np.real(np.linalg.eigvals(A)))
    np.testing.assert_allclose(got, lam, atol=1e-8)


def test_the_same_spectrum_gives_different_gmres_curves():
    """The central negative result: eigenvalues do not determine GMRES convergence. Both
    matrices below have eigenvalues 1..40 exactly."""
    n = 40
    lam = np.arange(1.0, n + 1)
    b = np.ones(n)
    normal = ns.gmres(ns.same_spectrum_family(lam, 0.0), b, tol=1e-8, max_iter=n)
    skewed = ns.gmres(ns.same_spectrum_family(lam, 30.0), b, tol=1e-8, max_iter=n)
    rel_n = normal.residual_norms / normal.residual_norms[0]
    rel_s = skewed.residual_norms / skewed.residual_norms[0]
    assert rel_n[30] < 1e-6, f"the normal one reached only {rel_n[30]:.2e} by step 30"
    assert rel_s[30] > 1e-3, f"the skewed one reached {rel_s[30]:.2e} by step 30"
    assert rel_s[30] / rel_n[30] > 1000.0


def test_mild_non_normality_can_help():
    """A story saying 'further from normal is slower' would be wrong, and this records the
    counterexample rather than leaving it out."""
    n = 40
    lam = np.arange(1.0, n + 1)
    b = np.ones(n)
    base = ns.gmres(ns.same_spectrum_family(lam, 0.0), b, tol=1e-8, max_iter=n)
    mild = ns.gmres(ns.same_spectrum_family(lam, 10.0), b, tol=1e-8, max_iter=n)
    assert ns.eigenvector_conditioning(ns.same_spectrum_family(lam, 10.0)) > 1e6
    assert mild.n_iter < base.n_iter, f"{mild.n_iter} against {base.n_iter}"


@pytest.mark.parametrize("n", [5, 20, 50])
def test_normal_matrices_have_conditioning_one_and_zero_departure(n, rng):
    A = random_spd(n, rng=rng)
    assert abs(ns.eigenvector_conditioning(A) - 1.0) < 1e-6
    assert ns.departure_from_normality(A) < 1e-12


@pytest.mark.parametrize("n", [10, 30])
def test_a_skew_matrix_is_normal(n, rng):
    S = rng.standard_normal((n, n))
    S = S - S.T
    assert ns.departure_from_normality(S) < 1e-12


def test_departure_from_normality_is_scale_invariant(rng):
    n = 20
    A = ns.convection_diffusion(n, 3.0)
    for scale in (1e-6, 1.0, 1e6):
        assert abs(ns.departure_from_normality(scale * A)
                   - ns.departure_from_normality(A)) < 1e-10


def test_the_normalised_measure_can_mislead_where_conditioning_does_not():
    """Measured: the raw commutator grows linearly in the Peclet number while ||A||_F^2 grows
    quadratically, so the scale invariant ratio peaks and then falls. kappa(V) does not."""
    pes = [0.0, 1.0, 3.0, 10.0, 30.0]
    ratios = [ns.departure_from_normality(ns.convection_diffusion(30, pe)) for pe in pes]
    assert ratios[-1] < max(ratios), "the normalised ratio was expected to peak and fall"
    assert ns.eigenvector_conditioning(ns.convection_diffusion(30, 0.0)) < 1.001


def test_the_operator_is_defective_at_peclet_two():
    """At pe = 2 the superdiagonal -1 + pe/2 vanishes, leaving a single Jordan block."""
    A = ns.convection_diffusion(20, 2.0)
    assert np.allclose(np.triu(A, 1), 0.0)
    assert ns.eigenvector_conditioning(A) > 1e10


@pytest.mark.parametrize("pe", [0.0, 1.0, 5.0])
@pytest.mark.parametrize("m", [1, 4, 15, 40])
def test_convection_diffusion_has_the_right_shape(m, pe):
    A = ns.convection_diffusion(m, pe)
    assert A.shape == (m, m)
    np.testing.assert_allclose(np.diag(A), np.full(m, 2.0))
    if pe == 0.0:
        np.testing.assert_allclose(A, A.T, atol=1e-14)


def test_convection_diffusion_rejects_an_empty_grid():
    with pytest.raises(ValueError):
        ns.convection_diffusion(0)


def test_same_spectrum_family_rejects_an_empty_spectrum():
    with pytest.raises(ValueError):
        ns.same_spectrum_family([], 1.0)


# ---------------------------------------------------------------- field of values


@pytest.mark.parametrize("n", [3, 10, 25])
def test_the_field_of_values_contains_the_eigenvalues(n, rng):
    A = rng.standard_normal((n, n))
    info = ns.field_of_values_bounds(A, 240)
    for lam in np.linalg.eigvals(A):
        assert info["real_min"] - 1e-8 <= lam.real <= info["real_max"] + 1e-8


@pytest.mark.parametrize("c", [1.0, 5.0, 20.0])
def test_the_origin_sits_outside_a_shifted_skew_matrix(c, rng):
    n = 20
    S = rng.standard_normal((n, n))
    S = S - S.T
    info = ns.field_of_values_bounds(c * np.eye(n) + S, 200)
    assert not info["contains_origin"]
    assert abs(info["distance_to_origin"] - c) < 1e-6


def test_the_origin_sits_inside_an_indefinite_field_of_values():
    info = ns.field_of_values_bounds(np.diag([-2.0, 1.0, 3.0]), 200)
    assert info["contains_origin"]
    assert info["distance_to_origin"] == 0.0


@pytest.mark.parametrize("n", [8, 24])
def test_elmans_bound_is_never_violated(n, rng):
    """Only where it applies, that is where the origin lies outside the field of values."""
    S = rng.standard_normal((n, n))
    S = S - S.T
    for c in (2.0, 5.0, 15.0):
        A = c * np.eye(n) + S
        info = ns.field_of_values_bounds(A, 200)
        nrm = float(np.linalg.norm(A, 2))
        rate = np.sqrt(max(0.0, 1.0 - (info["distance_to_origin"] / nrm) ** 2))
        if rate >= 1.0:
            continue
        got = ns.gmres(A, np.ones(n), tol=1e-8, max_iter=n)
        if got.converged:
            predicted = np.ceil(np.log(1e-8) / np.log(rate))
            assert got.n_iter <= predicted, f"{got.n_iter} steps against a bound of {predicted}"


# ---------------------------------------------------------------- MINRES


@pytest.mark.parametrize("n", [1, 2, 5, 15, 40])
def test_minres_solves_a_symmetric_indefinite_system(n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.sign(rng.standard_normal(n)) * rng.uniform(0.5, 3.0, n)
    A = (Q * lam) @ Q.T
    A = (A + A.T) / 2
    x_true = rng.standard_normal(n)
    r = ns.minres(A, A @ x_true, tol=1e-11, max_iter=4 * n)
    assert r.converged, r.message
    np.testing.assert_allclose(r.x, x_true, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n", [5, 20, 50])
def test_minres_also_solves_definite_systems(n, rng):
    A = random_spd(n, rng=rng)
    x_true = rng.standard_normal(n)
    r = ns.minres(A, A @ x_true, tol=1e-11, max_iter=4 * n)
    assert r.converged
    np.testing.assert_allclose(r.x, x_true, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n", [10, 30])
def test_minres_agrees_with_gmres_on_a_symmetric_matrix(n, rng):
    """They minimise the same thing, so on a symmetric matrix they must produce the same
    iterates. MINRES gets there with three vectors instead of n."""
    A = random_spd(n, kappa=100.0, rng=rng)
    b = rng.standard_normal(n)
    g = ns.gmres(A, b, tol=1e-11, max_iter=2 * n)
    m = ns.minres(A, b, tol=1e-11, max_iter=2 * n)
    assert g.converged and m.converged
    np.testing.assert_allclose(m.x, g.x, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("n", [5, 20])
def test_minres_residual_is_essentially_monotone(n, rng):
    A = random_spd(n, kappa=1e3, rng=rng)
    r = ns.minres(A, rng.standard_normal(n), tol=1e-11, max_iter=4 * n)
    assert np.all(np.diff(r.residual_norms) <= 1e-8 * r.residual_norms[0])


# ---------------------------------------------------------------- BiCG and BiCGSTAB


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("method", ["bicg", "bicgstab"])
def test_the_short_recurrence_methods_solve_an_easy_system(n, rng, method):
    A = well_posed(n, rng)
    x_true = rng.standard_normal(n)
    fn = ns.bicg if method == "bicg" else ns.bicgstab
    r = fn(A, A @ x_true, tol=1e-11, max_iter=8 * n)
    assert r.converged, f"{method}: {r.message}"
    np.testing.assert_allclose(r.x, x_true, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("method", ["bicg", "bicgstab"])
def test_the_short_recurrence_methods_are_scale_invariant(scale, rng, method):
    n = 20
    A = well_posed(n, rng)
    x_true = rng.standard_normal(n)
    fn = ns.bicg if method == "bicg" else ns.bicgstab
    plain = fn(A, A @ x_true, tol=1e-11, max_iter=8 * n)
    scaled = fn(scale * A, scale * (A @ x_true), tol=1e-11, max_iter=8 * n)
    assert plain.converged and scaled.converged
    np.testing.assert_allclose(scaled.x, plain.x, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("n", [6, 12, 24])
def test_an_orthogonal_shadow_breaks_bicg_down_at_once(n, rng):
    """Breakdown has no counterpart in CG, where the corresponding quantity is a norm. Here it
    is arranged deliberately, so the test does not depend on hitting one by chance."""
    A = well_posed(n, rng)
    b = rng.standard_normal(n)
    shadow = rng.standard_normal(n)
    shadow = shadow - (shadow @ b) / (b @ b) * b
    r = ns.bicg(A, b, tol=1e-10, max_iter=6 * n, shadow=shadow)
    assert r.breakdown
    assert not r.converged
    assert "orthogonal" in r.message


def test_a_relative_breakdown_test_is_what_catches_it():
    """A numerically zero inner product is about u times the product of the norms, roughly
    1e-16. An absolute floor of 1e-300 never fires, and the method divides by noise."""
    rng = np.random.default_rng(3)
    n = 12
    b = rng.standard_normal(n)
    shadow = rng.standard_normal(n)
    shadow = shadow - (shadow @ b) / (b @ b) * b
    product = abs(float(shadow @ b))
    relative = np.finfo(float).eps * np.linalg.norm(shadow) * np.linalg.norm(b)
    assert product > 1e-300, "an absolute floor would not have fired"
    assert product <= relative, "a relative floor does fire"


@pytest.mark.parametrize("n", [8, 20])
def test_a_useful_shadow_does_not_break_down(n, rng):
    A = well_posed(n, rng)
    b = rng.standard_normal(n)
    shadow = b + 0.1 * rng.standard_normal(n)
    r = ns.bicg(A, b, tol=1e-10, max_iter=8 * n, shadow=shadow)
    assert not r.breakdown


@pytest.mark.parametrize("n", [4, 12])
def test_bicg_rejects_a_wrong_length_shadow(n, rng):
    with pytest.raises(ValueError):
        ns.bicg(well_posed(n, rng), np.ones(n), shadow=np.ones(n + 1))


def test_bicg_residual_is_not_monotone():
    """Not a defect: it is the price of dropping the minimisation, and a test suite that never
    saw it would mean the experiment is not reproducing the known behaviour."""
    rng = np.random.default_rng(5)
    m = 120
    A = ns.convection_diffusion(m, 3.0)
    r = ns.bicg(A, rng.standard_normal(m), tol=1e-9, max_iter=8 * m)
    rel = r.residual_norms / r.residual_norms[0]
    assert rel.max() > 5.0, f"peak was only {rel.max():.2f}, expected a real excursion"


def test_gmres_is_the_reliable_one():
    """Over the same thirty problems: GMRES finished all of them, the short-recurrence methods
    did not. This is the practical conclusion of lesson 27, asserted rather than asserted-to."""
    m = 120
    wins = {"gmres": 0, "bicg": 0, "bicgstab": 0}
    for pe in (1.0, 3.0, 8.0):
        A = ns.convection_diffusion(m, pe)
        for trial in range(10):
            gen = np.random.default_rng(100 + trial)
            b = gen.standard_normal(m)
            wins["gmres"] += ns.gmres(A, b, tol=1e-9, max_iter=m).converged
            wins["bicg"] += ns.bicg(A, b, tol=1e-9, max_iter=8 * m).converged
            wins["bicgstab"] += ns.bicgstab(A, b, tol=1e-9, max_iter=8 * m).converged
    assert wins["gmres"] == 30, f"GMRES failed {30 - wins['gmres']} times"
    assert wins["bicg"] < 30 and wins["bicgstab"] < 30, (
        f"expected the short-recurrence methods to fail sometimes, got {wins}")


def test_neither_short_recurrence_method_dominates():
    """BiCGSTAB fails every time where BiCG succeeds every time, so 'BiCGSTAB replaced BiCG' is
    too simple a story."""
    m = 120
    A = ns.convection_diffusion(m, 8.0)
    bicg_wins = bicgstab_wins = 0
    for trial in range(6):
        gen = np.random.default_rng(100 + trial)
        b = gen.standard_normal(m)
        bicg_wins += ns.bicg(A, b, tol=1e-9, max_iter=8 * m).converged
        bicgstab_wins += ns.bicgstab(A, b, tol=1e-9, max_iter=8 * m).converged
    assert bicg_wins > bicgstab_wins, f"BiCG {bicg_wins}, BiCGSTAB {bicgstab_wins}"


@pytest.mark.parametrize("n", [10, 30])
def test_every_failure_is_reported_not_silent(n, rng):
    """The one outcome to rule out is a confident wrong answer. Whenever a method says it did
    not converge, the returned iterate must indeed have a large residual, and whenever it says
    it did, the residual must indeed be small."""
    for pe in (1.0, 5.0, 12.0):
        A = ns.convection_diffusion(n, pe)
        for fn in (ns.gmres, ns.bicg, ns.bicgstab):
            b = rng.standard_normal(n)
            r = fn(A, b, tol=1e-10, max_iter=4 * n)
            actual = np.linalg.norm(b - A @ r.x) / np.linalg.norm(b)
            if r.converged:
                assert actual < 1e-6, f"{fn.__name__} claimed success at {actual:.2e}"
            else:
                assert r.message, f"{fn.__name__} failed without saying why"


@pytest.mark.parametrize("n", [10, 25])
def test_bicgstab_needs_two_operator_applications_per_step(n, rng):
    A = well_posed(n, rng)
    r = ns.bicgstab(A, rng.standard_normal(n), tol=1e-10, max_iter=8 * n)
    assert abs(r.n_matvec - 2 * r.n_iter) <= 3


@pytest.mark.parametrize("n", [10, 25])
def test_bicgstab_accepts_a_preconditioner(n, rng):
    from nalib.krylov import jacobi_preconditioner

    A = well_posed(n, rng)
    x_true = rng.standard_normal(n)
    r = ns.bicgstab(A, A @ x_true, tol=1e-11, max_iter=8 * n, M=jacobi_preconditioner(A))
    assert r.converged
    np.testing.assert_allclose(r.x, x_true, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------- the result type


@pytest.mark.parametrize("n", [5, 20])
def test_relative_residuals_divide_by_the_right_thing(n, rng):
    A = well_posed(n, rng)
    b = rng.standard_normal(n)
    r = ns.gmres(A, b, tol=1e-10, max_iter=2 * n)
    np.testing.assert_allclose(r.relative_residuals(b),
                               r.residual_norms / np.linalg.norm(b), rtol=1e-14)


@pytest.mark.parametrize("n", [5, 20])
def test_history_is_kept_only_when_asked(n, rng):
    A = well_posed(n, rng)
    b = rng.standard_normal(n)
    with_hist = ns.gmres(A, b, tol=1e-10, max_iter=2 * n, keep_history=True)
    without = ns.gmres(A, b, tol=1e-10, max_iter=2 * n)
    assert with_hist.iterates.size > 0
    assert without.iterates.size == 0
    np.testing.assert_allclose(with_hist.x, without.x, atol=1e-12)
