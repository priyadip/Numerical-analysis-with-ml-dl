"""Tests for `nalib.leastsquares`.

The defining property of a least squares solution is that the residual is orthogonal to every
column, so that is what most of these check, rather than comparing against a reference that
could be wrong in the same way. The conditioning tests exist to record that the normal
equations really are as bad as the lesson says, at sizes and condition numbers swept rather
than picked.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import leastsquares as ls


SHAPES = [(1, 1), (2, 1), (3, 2), (10, 3), (40, 12), (200, 40)]
KAPPAS = [1.0, 10.0, 1e3, 1e6]
SCALES = [1e-8, 1e-2, 1.0, 1e5, 1e9]


# ---------------------------------------------------------------- design matrices


@pytest.mark.parametrize("n", [1, 2, 5, 20, 100])
@pytest.mark.parametrize("degree", [0, 1, 3, 7])
def test_vandermonde_has_the_right_shape_and_columns(n, degree, rng):
    x = rng.standard_normal(n)
    A = ls.vandermonde_design(x, degree)
    assert A.shape == (n, degree + 1)
    for j in range(degree + 1):
        np.testing.assert_allclose(A[:, j], x ** j, rtol=1e-12, atol=1e-14)


def test_vandermonde_rejects_a_negative_degree():
    with pytest.raises(ValueError):
        ls.vandermonde_design([1.0, 2.0], -1)


@pytest.mark.parametrize("n", [4, 12, 60])
@pytest.mark.parametrize("k", [0, 1, 3, 5])
def test_fourier_design_has_the_right_shape(n, k):
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    A = ls.fourier_design(t, k)
    assert A.shape == (n, 2 * k + 1)
    np.testing.assert_allclose(A[:, 0], np.ones(n))


@pytest.mark.parametrize("k", [1, 3, 8, 12])
def test_the_fourier_design_stays_well_conditioned(k):
    """Evenly spaced samples over a period make the columns nearly orthogonal, so the
    condition number does not move. This is the whole point of choosing the basis."""
    t = np.linspace(0.0, 1.0, 4 * k + 20, endpoint=False)
    assert np.linalg.cond(ls.fourier_design(t, k)) < 2.0


def test_the_vandermonde_condition_number_grows_exponentially():
    """Not merely "large": log(kappa) rises linearly in the degree, so kappa rises like a
    constant to the power of the degree. Asserting a threshold instead would be arbitrary."""
    t = np.linspace(0.0, 1.0, 60, endpoint=False)
    degrees = np.array([2, 4, 6, 8, 10, 12])
    logs = np.array([np.log10(np.linalg.cond(ls.vandermonde_design(t, d))) for d in degrees])
    slope, _ = np.polyfit(degrees, logs, 1)
    assert slope > 0.5, f"log10(kappa) grew only {slope:.3f} per degree"
    resid = np.max(np.abs(np.polyval(np.polyfit(degrees, logs, 1), degrees) - logs))
    assert resid < 0.6, f"the growth was not close to exponential, residual {resid:.3f}"


@pytest.mark.parametrize("degree", [4, 8, 12])
def test_the_vandermonde_design_is_far_worse_than_the_fourier_one(degree):
    t = np.linspace(0.0, 1.0, 60, endpoint=False)
    vand = np.linalg.cond(ls.vandermonde_design(t, degree))
    four = np.linalg.cond(ls.fourier_design(t, degree))
    assert vand > 100.0 * four, f"Vandermonde {vand:.3e}, Fourier {four:.3e}"


def test_the_vandermonde_condition_number_grows_with_the_degree():
    t = np.linspace(0.0, 1.0, 60, endpoint=False)
    conds = [np.linalg.cond(ls.vandermonde_design(t, d)) for d in (2, 4, 8, 12)]
    assert all(a < b for a, b in zip(conds, conds[1:])), f"{conds}"


@pytest.mark.parametrize("period", [0.5, 1.0, 7.0])
def test_the_fourier_design_respects_its_period(period):
    t = np.linspace(0.0, period, 40, endpoint=False)
    A = ls.fourier_design(t, 3, period=period)
    # column 1 is cos(2 pi t / period), which completes exactly one cycle
    np.testing.assert_allclose(A[:, 1], np.cos(2 * np.pi * t / period), atol=1e-12)


def test_fourier_rejects_a_negative_harmonic_count():
    with pytest.raises(ValueError):
        ls.fourier_design([0.0, 0.5], -1)


@pytest.mark.parametrize("m,n", SHAPES)
@pytest.mark.parametrize("kappa", [1.0, 1e3, 1e8])
def test_graded_design_hits_the_condition_number_it_was_asked_for(m, n, kappa, rng):
    A = ls.graded_design(m, n, kappa, rng)
    assert A.shape == (m, n)
    if n > 1:
        assert abs(np.linalg.cond(A) / kappa - 1.0) < 1e-6


def test_graded_design_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ls.graded_design(3, 5, 10.0)            # wide, not tall
    with pytest.raises(ValueError):
        ls.graded_design(5, 0, 10.0)
    with pytest.raises(ValueError):
        ls.graded_design(5, 2, 0.5)             # a condition number below 1


@pytest.mark.parametrize("m,n", [(3, 2), (10, 4), (30, 8)])
def test_hilbert_least_squares_is_consistent_by_construction(m, n):
    A, x_true, b = ls.hilbert_least_squares(m, n)
    assert A.shape == (m, n)
    np.testing.assert_allclose(b, A @ x_true, rtol=1e-14)
    assert np.linalg.cond(A) > 10.0 ** (n / 2)


def test_hilbert_rejects_a_wide_shape():
    with pytest.raises(ValueError):
        ls.hilbert_least_squares(3, 5)


# ---------------------------------------------------------------- the answer


@pytest.mark.parametrize("m,n", SHAPES)
def test_qr_residual_is_orthogonal_to_every_column(m, n, rng):
    """The defining property. Checked directly rather than against a reference."""
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    out = ls.solve_qr(A, b)
    ok, violation = ls.residual_is_orthogonal(A, out.x, b)
    assert ok, f"scaled violation {violation:.2e}"


@pytest.mark.parametrize("m,n", SHAPES)
def test_qr_matches_numpy_lstsq(m, n, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    want = np.linalg.lstsq(A, b, rcond=None)[0]
    np.testing.assert_allclose(ls.solve_qr(A, b).x, want, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("m,n", SHAPES)
def test_a_consistent_system_gives_a_zero_residual(m, n, rng):
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    out = ls.solve_qr(A, A @ x_true)
    assert out.residual_norm < 1e-10 * max(1.0, np.linalg.norm(A @ x_true))
    np.testing.assert_allclose(out.x, x_true, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("m,n", [(5, 2), (20, 6), (80, 20)])
def test_no_nearby_x_does_better(m, n, rng):
    """A minimum is a minimum: perturbing in any direction must not reduce the residual."""
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    out = ls.solve_qr(A, b)
    best = out.residual_norm
    for _ in range(10):
        step = 1e-3 * rng.standard_normal(n)
        assert np.linalg.norm(b - A @ (out.x + step)) >= best - 1e-12


@pytest.mark.parametrize("scale", SCALES)
def test_the_solution_is_scale_invariant(scale, rng):
    m, n = 30, 6
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    plain = ls.solve_qr(A, b).x
    scaled = ls.solve_qr(scale * A, scale * b).x
    np.testing.assert_allclose(scaled, plain, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("m,n", [(4, 2), (20, 5)])
def test_every_solver_rejects_a_length_mismatch(m, n, rng):
    A = rng.standard_normal((m, n))
    for solver in (ls.solve_qr, ls.solve_normal_equations, ls.solve_cholesky_normal):
        with pytest.raises(ValueError):
            solver(A, np.ones(m + 1))


def test_an_underdetermined_system_is_rejected_rather_than_guessed(rng):
    """Fewer equations than unknowns has infinitely many solutions, and picking one silently
    would hide that. The pseudoinverse (lesson 32) is the routine that answers this."""
    A = rng.standard_normal((3, 5))
    with pytest.raises(ValueError):
        ls.solve_qr(A, np.ones(3))
    with pytest.raises(ValueError):
        ls.solve_normal_equations(A, np.ones(3))


def test_a_rank_deficient_matrix_is_rejected_rather_than_solved(rng):
    A = rng.standard_normal((10, 3))
    A[:, 2] = A[:, 0] + 2.0 * A[:, 1]          # exactly dependent
    with pytest.raises(np.linalg.LinAlgError):
        ls.solve_qr(A, rng.standard_normal(10))


# ---------------------------------------------------------------- the projector


@pytest.mark.parametrize("m,n", SHAPES)
def test_the_projector_is_idempotent_symmetric_and_norm_one(m, n, rng):
    A = rng.standard_normal((m, n))
    P = ls.projector_onto_range(A)
    assert P.shape == (m, m)
    np.testing.assert_allclose(P @ P, P, atol=1e-11)
    np.testing.assert_allclose(P, P.T, atol=1e-11)
    assert abs(np.linalg.norm(P, 2) - 1.0) < 1e-9
    assert np.linalg.matrix_rank(P) == n


@pytest.mark.parametrize("m,n", SHAPES)
def test_the_projector_fixes_the_range_and_kills_its_complement(m, n, rng):
    A = rng.standard_normal((m, n))
    P = ls.projector_onto_range(A)
    inside = A @ rng.standard_normal(n)
    np.testing.assert_allclose(P @ inside, inside, rtol=1e-8, atol=1e-10)
    outside = (np.eye(m) - P) @ rng.standard_normal(m)
    assert np.linalg.norm(P @ outside) < 1e-9 * max(1.0, np.linalg.norm(outside))


@pytest.mark.parametrize("m,n", SHAPES)
def test_pythagoras_holds_for_the_split(m, n, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    P = ls.projector_onto_range(A)
    explained = P @ b
    left = b - explained
    np.testing.assert_allclose(b @ b, explained @ explained + left @ left, rtol=1e-11)
    assert abs(explained @ left) < 1e-10 * max(1.0, b @ b)


@pytest.mark.parametrize("m,n", [(6, 2), (25, 7)])
def test_the_projected_vector_is_what_the_solver_produces(m, n, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    np.testing.assert_allclose(A @ ls.solve_qr(A, b).x,
                               ls.projector_onto_range(A) @ b, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------- the angle


@pytest.mark.parametrize("m,n", [(10, 3), (40, 8), (100, 20)])
def test_the_angle_is_zero_for_data_inside_the_range(m, n, rng):
    A = rng.standard_normal((m, n))
    assert ls.angle_to_range(A, A @ rng.standard_normal(n)) < 1e-6


@pytest.mark.parametrize("m,n", [(10, 3), (40, 8)])
def test_sin_of_the_angle_is_the_relative_residual(m, n, rng):
    """The identity that makes the angle measurable without knowing the answer."""
    A = rng.standard_normal((m, n))
    for level in (1e-3, 0.1, 1.0, 10.0):
        b = A @ rng.standard_normal(n) + level * rng.standard_normal(m)
        theta = ls.angle_to_range(A, b)
        rel = ls.solve_qr(A, b).residual_norm / np.linalg.norm(b)
        assert abs(np.sin(theta) - rel) < 1e-8, f"{np.sin(theta):.6f} against {rel:.6f}"


@pytest.mark.parametrize("m,n", [(20, 4), (60, 10)])
def test_the_angle_grows_with_the_noise(m, n, rng):
    A = rng.standard_normal((m, n))
    base = A @ rng.standard_normal(n)
    noise = rng.standard_normal(m)
    angles = [ls.angle_to_range(A, base + lev * noise) for lev in (1e-3, 1e-2, 1e-1, 1.0)]
    assert all(a < b for a, b in zip(angles, angles[1:])), f"{angles}"


def test_the_angle_of_the_zero_vector_is_zero(rng):
    assert ls.angle_to_range(rng.standard_normal((5, 2)), np.zeros(5)) == 0.0


# ---------------------------------------------------------------- conditioning


@pytest.mark.parametrize("kappa", [1.0, 1e2, 1e4, 1e6, 1e8])
def test_the_condition_number_really_is_squared(kappa, rng):
    A = ls.graded_design(50, 6, kappa, rng)
    info = ls.condition_squaring(A)
    if kappa ** 2 < 1e15:                       # above that the reference itself is unreliable
        assert abs(info["kappa_normal"] / info["kappa_A"] ** 2 - 1.0) < 1e-4


@pytest.mark.parametrize("kappa", [1e2, 1e4, 1e6])
def test_the_digit_counts_differ_by_a_factor_of_two(kappa, rng):
    info = ls.condition_squaring(ls.graded_design(40, 6, kappa, rng))
    lost_qr = 16.0 - info["digits_qr"]
    lost_normal = 16.0 - info["digits_normal"]
    assert abs(lost_normal / lost_qr - 2.0) < 0.2, f"{lost_qr:.2f}, {lost_normal:.2f}"


@pytest.mark.parametrize("kappa", [1e4, 1e6, 1e8])
def test_qr_beats_the_normal_equations_where_it_should(kappa, rng):
    m, n = 60, 8
    A = ls.graded_design(m, n, kappa, rng)
    x_true = rng.standard_normal(n)
    b = A @ x_true
    rel = lambda z: np.linalg.norm(z - x_true) / np.linalg.norm(x_true)
    e_qr = rel(ls.solve_qr(A, b).x)
    e_normal = rel(ls.solve_normal_equations(A, b).x)
    assert e_qr < e_normal, f"QR {e_qr:.2e}, normal {e_normal:.2e}"
    assert e_qr < 1e-6, f"QR should still be usable at kappa {kappa:.0e}, got {e_qr:.2e}"


def test_the_normal_equations_lose_everything_at_high_kappa():
    """Not merely worse: no correct digits. This is the measurement the lesson rests on."""
    gen = np.random.default_rng(11)
    m, n = 60, 8
    A = ls.graded_design(m, n, 1e10, gen)
    x_true = gen.standard_normal(n)
    b = A @ x_true
    e_normal = (np.linalg.norm(ls.solve_normal_equations(A, b).x - x_true)
                / np.linalg.norm(x_true))
    e_qr = np.linalg.norm(ls.solve_qr(A, b).x - x_true) / np.linalg.norm(x_true)
    assert e_normal > 0.1, f"expected total failure, got {e_normal:.2e}"
    assert e_qr < 1e-5, f"expected QR to survive, got {e_qr:.2e}"


@pytest.mark.parametrize("kappa", [1.0, 10.0, 100.0])
def test_the_normal_equations_are_fine_when_the_design_is(kappa, rng):
    A = ls.graded_design(80, 10, kappa, rng)
    b = rng.standard_normal(80)
    np.testing.assert_allclose(ls.solve_normal_equations(A, b).x, ls.solve_qr(A, b).x,
                               rtol=1e-8, atol=1e-10)


def test_cholesky_on_the_normal_equations_eventually_breaks_down():
    """The reformulation, not the factorization, is what fails. Recorded as a rate rather than
    a single case, because whether a given draw fails is a coin flip near the threshold."""
    failures = {}
    for exponent in (6.0, 8.0, 9.0):
        n_fail = 0
        for trial in range(20):
            gen = np.random.default_rng(300 + trial)
            A = ls.graded_design(40, 6, 10.0 ** exponent, gen)
            try:
                ls.solve_cholesky_normal(A, A @ np.ones(A.shape[1]))
            except np.linalg.LinAlgError:
                n_fail += 1
        failures[exponent] = n_fail
    assert failures[6.0] == 0, "Cholesky should not fail at kappa = 1e6"
    assert failures[9.0] > 10, f"expected most draws to fail at 1e9, got {failures[9.0]}/20"
    assert failures[6.0] <= failures[8.0] <= failures[9.0]


@pytest.mark.parametrize("m,n", [(20, 5), (60, 12)])
def test_cholesky_and_the_general_solve_agree_where_both_work(m, n, rng):
    A = ls.graded_design(m, n, 1e4, rng)
    b = rng.standard_normal(m)
    np.testing.assert_allclose(ls.solve_cholesky_normal(A, b).x,
                               ls.solve_normal_equations(A, b).x, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("m,n", SHAPES)
def test_normal_matrix_is_the_gram_matrix(m, n, rng):
    A = rng.standard_normal((m, n))
    G = ls.normal_matrix(A)
    assert G.shape == (n, n)
    np.testing.assert_allclose(G, G.T, atol=1e-13)
    for i in range(n):
        for j in range(n):
            assert abs(G[i, j] - A[:, i] @ A[:, j]) < 1e-11


# ---------------------------------------------------------------- fitting


@pytest.mark.parametrize("degree", [0, 1, 2, 5])
@pytest.mark.parametrize("method", ["qr", "normal", "cholesky"])
def test_polynomial_fit_recovers_an_exact_polynomial(degree, method, rng):
    x = np.linspace(-1.0, 1.0, 4 * degree + 12)
    coeffs = rng.standard_normal(degree + 1)
    y = ls.vandermonde_design(x, degree) @ coeffs
    out = ls.fit_polynomial(x, y, degree, method=method)
    np.testing.assert_allclose(out.x, coeffs, rtol=1e-6, atol=1e-8)
    assert out.residual_norm < 1e-8 * max(1.0, np.linalg.norm(y))


@pytest.mark.parametrize("degree", [1, 2, 4])
def test_polynomial_fit_matches_numpy_polyfit(degree, rng):
    x = np.linspace(0.0, 3.0, 25)
    y = rng.standard_normal(25)
    np.testing.assert_allclose(ls.fit_polynomial(x, y, degree).x,
                               np.polyfit(x, y, degree)[::-1], rtol=1e-6, atol=1e-9)


def test_polynomial_fit_rejects_an_unknown_method(rng):
    with pytest.raises(ValueError):
        ls.fit_polynomial([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1, method="svd")


def test_polynomial_fit_rejects_mismatched_data():
    with pytest.raises(ValueError):
        ls.fit_polynomial([1.0, 2.0, 3.0], [1.0, 2.0], 1)


def test_the_residual_falls_with_every_extra_parameter(rng):
    """Which is exactly why it cannot be used to choose a model."""
    x = np.linspace(-1.0, 1.0, 25)
    y = 3.0 - 2.0 * x + 0.5 * x ** 2 + 0.05 * rng.standard_normal(25)
    res = [ls.fit_polynomial(x, y, d).residual_norm for d in (0, 1, 2, 3, 6, 12)]
    assert all(a >= b - 1e-12 for a, b in zip(res, res[1:])), f"{res}"


# ---------------------------------------------------------------- linearization


@pytest.mark.parametrize("c,k", [(1.0, 1.0), (2.0, 0.8), (0.3, -1.5), (100.0, 0.05)])
def test_the_exponential_linearization_is_exact_on_exact_data(c, k):
    x = np.linspace(0.2, 3.0, 15)
    out = ls.linearize_exponential(x, c * np.exp(k * x))
    assert abs(out["c"] - c) < 1e-9 * max(1.0, abs(c))
    assert abs(out["k"] - k) < 1e-9 * max(1.0, abs(k))
    assert out["sse_log"] < 1e-18


@pytest.mark.parametrize("c,p", [(1.0, 2.0), (3.0, -0.5), (0.1, 1.5)])
def test_the_power_linearization_is_exact_on_exact_data(c, p):
    x = np.linspace(0.5, 5.0, 15)
    out = ls.linearize_power(x, c * x ** p)
    assert abs(out["c"] - c) < 1e-9 * max(1.0, abs(c))
    assert abs(out["p"] - p) < 1e-9 * max(1.0, abs(p))


@pytest.mark.parametrize("n", [8, 30, 120])
def test_the_predict_helper_reproduces_the_fit(n, rng):
    x = np.linspace(0.5, 4.0, n)
    y = 2.0 * np.exp(0.7 * x) * (1.0 + 0.05 * rng.standard_normal(n))
    out = ls.linearize_exponential(x, y)
    np.testing.assert_allclose(out["predict"](x), out["c"] * np.exp(out["k"] * x), rtol=1e-12)


def test_linearization_rejects_nonpositive_data():
    """A nan would be worse than a refusal: the model cannot produce a nonpositive value, so
    the data does not fit the model and saying so is the only honest option."""
    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        ls.linearize_exponential(x, np.array([1.0, -2.0, 3.0]))
    with pytest.raises(ValueError):
        ls.linearize_exponential(x, np.array([1.0, 0.0, 3.0]))
    with pytest.raises(ValueError):
        ls.linearize_power(np.array([1.0, 0.0, 3.0]), np.array([1.0, 2.0, 3.0]))


def test_linearization_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ls.linearize_exponential([1.0, 2.0], [1.0, 2.0, 3.0])


def test_the_log_fit_and_the_direct_fit_optimise_different_things():
    """Not a defect of either: each is the minimiser of its own criterion. Getting this
    backwards is how people fit the wrong model without noticing."""
    from scipy.optimize import curve_fit

    gen = np.random.default_rng(5)
    x = np.linspace(0.5, 4.0, 12)
    y = 2.0 * np.exp(0.8 * x) * (1.0 + 0.15 * gen.standard_normal(x.size))

    log_fit = ls.linearize_exponential(x, y)
    params, _ = curve_fit(lambda t, c, k: c * np.exp(k * t), x, y, p0=[1.0, 1.0])
    direct = params[0] * np.exp(params[1] * x)

    # the direct fit wins on the y residual, which is what it minimises
    assert np.sum((y - direct) ** 2) <= log_fit["sse_original"] * (1.0 + 1e-9)
    # and the log fit wins on the log residual, which is what IT minimises
    assert log_fit["sse_log"] <= np.sum((np.log(y) - np.log(direct)) ** 2) * (1.0 + 1e-9)


# ---------------------------------------------------------------- the result type


@pytest.mark.parametrize("m,n", [(10, 3), (50, 8)])
def test_the_result_carries_a_consistent_residual(m, n, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    for solver in (ls.solve_qr, ls.solve_normal_equations, ls.solve_cholesky_normal):
        out = solver(A, b)
        np.testing.assert_allclose(out.residual, b - A @ out.x, atol=1e-12)
        assert abs(out.residual_norm - np.linalg.norm(out.residual)) < 1e-12
        assert out.rank == n
        assert out.method


@pytest.mark.parametrize("m,n", [(10, 3), (50, 8)])
def test_conditioning_is_reported_when_asked_and_skipped_when_not(m, n, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    with_check = ls.solve_normal_equations(A, b, check_conditioning=True)
    without = ls.solve_normal_equations(A, b, check_conditioning=False)
    assert np.isfinite(with_check.cond_A) and np.isfinite(with_check.cond_normal)
    assert np.isnan(without.cond_A) and np.isnan(without.cond_normal)
    np.testing.assert_allclose(with_check.x, without.x, atol=0.0)


# ---------------------------------------------------------------- the SVD route


@pytest.mark.parametrize("m,n", SHAPES)
def test_svd_solves_the_same_problem_as_qr(m, n, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    np.testing.assert_allclose(ls.solve_svd(A, b).x, ls.solve_qr(A, b).x,
                               rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("m,n", SHAPES)
def test_svd_residual_is_orthogonal(m, n, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    out = ls.solve_svd(A, b)
    ok, violation = ls.residual_is_orthogonal(A, out.x, b)
    assert ok, f"scaled violation {violation:.2e}"


@pytest.mark.parametrize("m,n", [(3, 8), (5, 20), (10, 25)])
def test_svd_handles_an_underdetermined_system_where_qr_refuses(m, n, rng):
    """Fewer equations than unknowns has infinitely many exact solutions. QR refuses; the SVD
    returns the smallest one, which is a choice it states rather than a guess it hides."""
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    with pytest.raises(ValueError):
        ls.solve_qr(A, b)
    out = ls.solve_svd(A, b)
    assert out.residual_norm < 1e-10 * max(1.0, np.linalg.norm(b))
    # and it really is the minimum norm one
    for _ in range(5):
        other = out.x + rng.standard_normal(n) * 1e-3
        if np.linalg.norm(b - A @ other) <= out.residual_norm + 1e-12:
            assert np.linalg.norm(other) >= np.linalg.norm(out.x) - 1e-12


@pytest.mark.parametrize("m,n,r", [(10, 4, 2), (20, 6, 3), (40, 10, 7)])
def test_svd_reports_the_rank(m, n, r, rng):
    A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    assert ls.solve_svd(A, rng.standard_normal(m)).rank == r


@pytest.mark.parametrize("scale", SCALES)
def test_svd_is_scale_invariant(scale, rng):
    A = rng.standard_normal((30, 6))
    b = rng.standard_normal(30)
    np.testing.assert_allclose(ls.solve_svd(scale * A, scale * b).x,
                               ls.solve_svd(A, b).x, rtol=1e-8, atol=1e-10)


def test_svd_rejects_a_length_mismatch(rng):
    with pytest.raises(ValueError):
        ls.solve_svd(rng.standard_normal((6, 3)), np.ones(7))


# ---------------------------------------------------------------- the pseudoinverse


@pytest.mark.parametrize("m,n,r", [(8, 3, 3), (20, 6, 6), (10, 4, 2), (4, 7, 3), (30, 10, 5)])
def test_all_four_penrose_conditions_hold(m, n, r, rng):
    """All four, not two. Dropping the symmetry conditions leaves a generalised inverse, of
    which there are infinitely many."""
    A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    for name, value in ls.penrose_residuals(A, ls.pseudoinverse(A)).items():
        assert value < 1e-11, f"{name} violated by {value:.2e} at {m}x{n} rank {r}"


@pytest.mark.parametrize("m,n", [(6, 3), (15, 4), (40, 12)])
def test_the_pseudoinverse_matches_the_normal_equations_formula(m, n, rng):
    A = rng.standard_normal((m, n))
    np.testing.assert_allclose(ls.pseudoinverse(A),
                               np.linalg.solve(A.T @ A, A.T), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("m,n", [(6, 3), (20, 5)])
def test_the_pseudoinverse_matches_numpy(m, n, rng):
    A = rng.standard_normal((m, n))
    np.testing.assert_allclose(ls.pseudoinverse(A), np.linalg.pinv(A), atol=1e-10)


@pytest.mark.parametrize("n", [2, 5, 12])
def test_the_pseudoinverse_of_an_invertible_matrix_is_its_inverse(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    np.testing.assert_allclose(ls.pseudoinverse(A), np.linalg.inv(A), rtol=1e-8, atol=1e-10)


def test_the_pseudoinverse_of_the_zero_matrix_is_the_zero_matrix():
    Z = np.zeros((5, 3))
    np.testing.assert_allclose(ls.pseudoinverse(Z), np.zeros((3, 5)), atol=0.0)


@pytest.mark.parametrize("m,n,r", [(12, 5, 4), (20, 6, 4)])
def test_the_minimum_norm_solution_is_minimal(m, n, r, rng):
    """Every point of the solution family has the SAME residual, so the residual does not
    choose between them. The pseudoinverse picks the smallest, and that is a choice."""
    A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    b = rng.standard_normal(m)
    x_min = ls.minimum_norm_solution(A, b)
    base = np.linalg.norm(b - A @ x_min)

    # find a null vector by the SVD, then walk along it
    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    null = Vt[-1]
    assert np.linalg.norm(A @ null) < 1e-10 * max(1.0, np.linalg.norm(A, 2))
    for t in (0.5, 1.0, 10.0):
        moved = x_min + t * null
        assert abs(np.linalg.norm(b - A @ moved) - base) < 1e-9 * max(1.0, base)
        assert np.linalg.norm(moved) > np.linalg.norm(x_min) - 1e-12


# ---------------------------------------------------------------- the four condition numbers


@pytest.mark.parametrize("m,n", [(20, 5), (60, 12)])
@pytest.mark.parametrize("kappa", [10.0, 1e4, 1e8])
def test_the_residual_sensitivity_to_a_equals_kappa(m, n, kappa, rng):
    """One of the four is exactly kappa, and it is the one nobody quotes."""
    A = ls.graded_design(m, n, kappa, rng)
    b = rng.standard_normal(m)
    info = ls.least_squares_conditioning(A, b)
    assert abs(info["A_to_r"] / kappa - 1.0) < 1e-6


@pytest.mark.parametrize("m,n", [(20, 5), (60, 12)])
def test_sin_theta_is_the_relative_residual(m, n, rng):
    A = rng.standard_normal((m, n))
    for level in (1e-6, 1e-2, 1.0, 10.0):
        b = A @ rng.standard_normal(n) + level * rng.standard_normal(m)
        info = ls.least_squares_conditioning(A, b)
        rel = ls.solve_qr(A, b).residual_norm / np.linalg.norm(b)
        assert abs(np.sin(info["theta"]) - rel) < 1e-9


@pytest.mark.parametrize("m,n", [(30, 6)])
def test_the_four_sensitivities_do_not_move_together(m, n, rng):
    """The whole reason there are four. As theta grows, b->x falls and A->x rises."""
    A = ls.graded_design(m, n, 1e6, rng)
    seen = []
    for level in (1e-10, 1e-3, 1.0):
        b = A @ rng.standard_normal(n) + level * rng.standard_normal(m)
        info = ls.least_squares_conditioning(A, b)
        seen.append((info["theta"], info["b_to_x"], info["A_to_x"]))
    seen.sort()
    assert seen[0][1] > seen[-1][1] * 100, "b->x should FALL as theta grows"
    assert seen[-1][2] > seen[0][2], "A->x should RISE as theta grows"


@pytest.mark.parametrize("m,n", [(20, 5), (50, 10)])
def test_eta_lies_between_one_and_kappa(m, n, rng):
    A = ls.graded_design(m, n, 1e5, rng)
    b = rng.standard_normal(m)
    info = ls.least_squares_conditioning(A, b)
    assert 1.0 - 1e-9 <= info["eta"] <= info["kappa"] * (1.0 + 1e-6)


@pytest.mark.parametrize("m,n", [(30, 6), (60, 10)])
def test_the_bound_is_never_violated_by_a_random_perturbation(m, n, rng):
    A = ls.graded_design(m, n, 1e5, rng)
    for level in (1e-6, 1.0):
        b = A @ rng.standard_normal(n) + level * rng.standard_normal(m)
        bound = ls.least_squares_conditioning(A, b)
        seen = ls.measure_sensitivity(A, b, n_trials=20, level=1e-9,
                                      rng=np.random.default_rng(2))
        for key in ("b_to_x", "b_to_r", "A_to_x", "A_to_r"):
            assert seen[key] <= bound[key] * 1.5 + 1e-9, f"{key} exceeded its bound"


def test_conditioning_rejects_a_length_mismatch(rng):
    with pytest.raises(ValueError):
        ls.least_squares_conditioning(rng.standard_normal((6, 3)), np.ones(7))


# ---------------------------------------------------------------- regularization


@pytest.mark.parametrize("m,n", [(20, 5), (40, 10)])
def test_zero_penalty_reproduces_the_plain_solution(m, n, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    np.testing.assert_allclose(ls.tikhonov(A, b, 0.0), ls.solve_qr(A, b).x,
                               rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("m,n", [(20, 5), (40, 10)])
def test_a_larger_penalty_gives_a_smaller_solution(m, n, rng):
    A = ls.graded_design(m, n, 1e6, rng)
    b = rng.standard_normal(m)
    norms = [np.linalg.norm(ls.tikhonov(A, b, lam))
             for lam in (1e-10, 1e-6, 1e-3, 1e-1, 1.0)]
    assert all(a >= b_ - 1e-12 for a, b_ in zip(norms, norms[1:])), f"{norms}"


@pytest.mark.parametrize("m,n", [(20, 5), (40, 10)])
def test_a_larger_penalty_gives_a_larger_residual(m, n, rng):
    A = ls.graded_design(m, n, 1e6, rng)
    b = rng.standard_normal(m)
    res = [np.linalg.norm(b - A @ ls.tikhonov(A, b, lam))
           for lam in (1e-10, 1e-6, 1e-3, 1e-1, 1.0)]
    assert all(a <= b_ + 1e-12 for a, b_ in zip(res, res[1:])), f"{res}"


@pytest.mark.parametrize("m,n,r", [(20, 6, 4), (30, 8, 5)])
def test_tikhonov_works_where_qr_refuses(m, n, r, rng):
    """Regularization makes a rank deficient problem well posed, which is its whole point."""
    A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    b = rng.standard_normal(m)
    with pytest.raises(np.linalg.LinAlgError):
        ls.solve_qr(A, b)
    x = ls.tikhonov(A, b, 1e-6)
    assert np.all(np.isfinite(x))


def test_tikhonov_approaches_the_minimum_norm_solution_and_then_diverges():
    """The mathematical limit as lambda -> 0 is the minimum norm solution. In floating point it
    is approached and then LOST, because the singular values that should be zero are actually
    about u ||A||, and their filter factor sigma/(sigma^2 + lambda^2) grows once
    lambda drops below sqrt(sigma). Measured: the error falls to 2.7e-8 at lambda = 1e-3 and is
    back up to 2e5 by lambda = 1e-10."""
    gen = np.random.default_rng(5)
    A = gen.standard_normal((20, 6)) @ gen.standard_normal((6, 8))
    b = gen.standard_normal(20)
    target = ls.minimum_norm_solution(A, b)
    lambdas = [1e-1, 1e-2, 1e-3, 1e-6, 1e-8, 1e-10]
    errors = [np.linalg.norm(ls.tikhonov(A, b, lam) - target) for lam in lambdas]
    # it approaches
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-6 * max(1.0, np.linalg.norm(target))
    # and then it diverges
    assert errors[-1] > errors[2] * 1e6, f"expected divergence, got {errors}"


def test_the_useful_range_of_the_penalty_is_bounded_below():
    """Below about sqrt(u ||A||) the penalty stops helping and starts hurting, so 'as small as
    possible' is the wrong instruction."""
    gen = np.random.default_rng(5)
    A = gen.standard_normal((20, 6)) @ gen.standard_normal((6, 8))
    b = gen.standard_normal(20)
    target = ls.minimum_norm_solution(A, b)
    sweep = np.geomspace(1e-12, 1.0, 40)
    errors = np.array([np.linalg.norm(ls.tikhonov(A, b, lam) - target) for lam in sweep])
    best = sweep[int(np.argmin(errors))]
    floor = np.sqrt(np.finfo(float).eps * np.linalg.norm(A, 2))
    assert best > floor / 100.0, f"best lambda {best:.2e} against a floor of {floor:.2e}"


def test_tikhonov_rejects_a_negative_penalty(rng):
    with pytest.raises(ValueError):
        ls.tikhonov(rng.standard_normal((6, 3)), np.ones(6), -1.0)


def test_tikhonov_rejects_a_length_mismatch(rng):
    with pytest.raises(ValueError):
        ls.tikhonov(rng.standard_normal((6, 3)), np.ones(7), 1.0)


@pytest.mark.parametrize("m,n", [(30, 8), (60, 12)])
def test_the_l_curve_is_monotone_in_both_norms(m, n, rng):
    A = ls.graded_design(m, n, 1e6, rng)
    b = rng.standard_normal(m)
    curve = ls.l_curve(A, b)
    assert np.all(np.diff(curve["residual_norms"]) >= -1e-12), "residual should not fall"
    assert np.all(np.diff(curve["solution_norms"]) <= 1e-12), "||x|| should not rise"
    assert 0 < curve["corner_index"] < curve["lambdas"].size - 1


def test_the_l_curve_corner_is_close_to_the_best_penalty():
    """It uses only the two norms, both computable, and lands near a choice that needed the
    answer. Recorded as a factor rather than a single number, because the corner is a heuristic
    and the point is that it is a good one."""
    gen = np.random.default_rng(13)
    A = ls.graded_design(40, 10, 1e8, gen)
    x_true = gen.standard_normal(A.shape[1])
    b = A @ x_true + 1e-6 * gen.standard_normal(A.shape[0])
    curve = ls.l_curve(A, b)
    err = lambda lam: (np.linalg.norm(ls.tikhonov(A, b, lam) - x_true)
                       / np.linalg.norm(x_true))
    best = min(err(lam) for lam in np.geomspace(1e-12, 1.0, 60))
    assert err(curve["corner_lambda"]) < 2.0 * best
    assert err(curve["corner_lambda"]) < 0.1 * err(0.0)


@pytest.mark.parametrize("m,n", [(20, 6), (40, 10)])
def test_truncated_svd_at_full_rank_is_the_plain_solution(m, n, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    np.testing.assert_allclose(ls.truncated_svd(A, b, n), ls.solve_qr(A, b).x,
                               rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("m,n", [(30, 8)])
def test_truncating_further_gives_a_smaller_solution(m, n, rng):
    A = ls.graded_design(m, n, 1e8, rng)
    b = rng.standard_normal(m)
    norms = [np.linalg.norm(ls.truncated_svd(A, b, k)) for k in range(1, n + 1)]
    assert norms[-1] > norms[0]
    assert all(a <= b_ * (1.0 + 1e-9) for a, b_ in zip(norms, norms[1:])), f"{norms}"


def test_truncated_svd_rejects_a_bad_rank(rng):
    A = rng.standard_normal((10, 4))
    for bad in (0, -1, 5):
        with pytest.raises(ValueError):
            ls.truncated_svd(A, np.ones(10), bad)


def test_truncation_and_tikhonov_agree_when_the_spectrum_has_a_gap(rng):
    """And they diverge when it does not, which is why lesson 33 has to decide whether it does."""
    m, n, r = 40, 8, 4
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    spread = np.concatenate([np.ones(r), np.full(n - r, 1e-12)])   # a clear gap
    A = (U * spread) @ V.T
    b = rng.standard_normal(m)
    x_trunc = ls.truncated_svd(A, b, r)
    # The penalty has to sit between sqrt(sigma_small) and sigma_large, NOT merely inside the
    # gap: the filter factor for a tiny sigma is sigma / lambda^2, so lambda must exceed
    # sqrt(sigma) for that direction to be suppressed rather than amplified.
    small, large = float(spread[-1]), float(spread[0])
    lam = float(np.sqrt(np.sqrt(small) * large))            # the geometric middle
    x_tik = ls.tikhonov(A, b, lam)
    rel = np.linalg.norm(x_trunc - x_tik) / max(np.linalg.norm(x_trunc), 1e-300)
    assert rel < 1e-4, f"they differ by {rel:.2e} at lambda = {lam:.1e} despite a clear gap"
    # and a penalty inside the gap but below sqrt(sigma_small) does NOT match
    assert (np.linalg.norm(x_trunc - ls.tikhonov(A, b, small * 10.0))
            / np.linalg.norm(x_trunc)) > 1.0
