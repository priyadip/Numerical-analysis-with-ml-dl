"""Tests for `nalib.nlls`.

The claims tested here are about **rates**, not just answers, because the whole content of
Gauss-Newton is that its rate depends on the residual at the solution. So there are tests that
the rate is quadratic on a zero residual problem, tests that it degrades in step with the
dropped Hessian term, and a test that it stops converging altogether once that term approaches
the term it was dropped from.

Levenberg-Marquardt is not tested as "better than Gauss-Newton", because measured, it is not
always. It is tested where it is better, and there is a test recording a case where it is worse.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import nlls


# ---------------------------------------------------------------- the zero residual case


@pytest.mark.parametrize("n_points", [5, 20, 60, 200])
@pytest.mark.parametrize("truth", [[2.0, -0.7], [0.5, 0.4], [-3.0, -1.2]])
def test_gauss_newton_solves_an_exact_exponential_fit(n_points, truth):
    """No noise, so the residual at the solution is zero and both parameters are recoverable."""
    prob = nlls.exponential_model(n_points, truth)
    start = np.asarray(truth, dtype=float) * 0.5
    out = nlls.gauss_newton(prob["residual"], prob["jacobian"], start, max_iter=200)
    assert out.converged, out.message
    assert np.linalg.norm(out.x - np.asarray(truth)) < 1e-8, f"got {out.x}, wanted {truth}"


@pytest.mark.parametrize("n_points", [20, 60])
def test_convergence_is_quadratic_when_the_residual_is_zero(n_points):
    """The dropped term S is r_i times a Hessian, so a zero residual drops nothing and
    Gauss-Newton IS Newton. Measured order: 1.90 on 30 points."""
    prob = nlls.exponential_model(n_points, [2.0, -0.7])
    out = nlls.gauss_newton(prob["residual"], prob["jacobian"], [1.0, -0.2], max_iter=100)
    rate = nlls.observed_rate(out.history)
    assert rate["order"] > 1.6, f"order {rate['order']:.2f}, expected close to 2"
    assert out.iterations < 12, f"{out.iterations} iterations for a quadratic method"


@pytest.mark.parametrize("n_points", [20, 60])
def test_the_dropped_term_vanishes_at_a_zero_residual_solution(n_points):
    prob = nlls.exponential_model(n_points, [2.0, -0.7])
    info = nlls.dropped_term(prob["residual"], prob["jacobian"], prob["hessians"],
                             prob["truth"])
    assert info["residual_norm"] < 1e-12
    assert info["ratio"] < 1e-12, f"S/JtJ = {info['ratio']:.2e} at an exact solution"


# ---------------------------------------------------------------- the large residual case


@pytest.mark.parametrize("offset,expected_ratio", [(0.0, 0.0), (1.0, 0.05), (3.0, 0.17),
                                                   (6.0, 0.28)])
def test_the_dropped_term_grows_with_the_residual(offset, expected_ratio):
    """The knob is the unfittable offset, and it moves ||S||/||J^T J|| in step."""
    prob = nlls.large_residual_problem(40, offset)
    out = nlls.gauss_newton(prob["residual"], prob["jacobian"], [1.5, 1.1], max_iter=300)
    info = nlls.dropped_term(prob["residual"], prob["jacobian"], prob["hessians"], out.x)
    assert abs(info["ratio"] - expected_ratio) < 0.03, \
        f"offset {offset}: ratio {info['ratio']:.3f}, expected about {expected_ratio}"


def test_the_iteration_count_grows_with_the_dropped_term():
    """Measured: 5, 8, 12, 16 iterations as the ratio goes 0.00, 0.05, 0.17, 0.28."""
    counts = []
    for offset in (0.0, 1.0, 3.0, 6.0):
        prob = nlls.large_residual_problem(40, offset)
        counts.append(nlls.gauss_newton(prob["residual"], prob["jacobian"], [1.5, 1.1],
                                        max_iter=300).iterations)
    assert counts == sorted(counts), f"iteration counts not monotone: {counts}"
    assert counts[-1] >= 3 * counts[0], f"{counts[0]} to {counts[-1]} is not much of a slowdown"


def test_gauss_newton_stalls_when_the_dropped_term_approaches_one():
    """At ratio 0.987 neither method converges in 300 iterations. That is the theory's
    prediction, not a bug: Gauss-Newton's linear factor is roughly ||S||/||J^T J||, so a ratio
    near 1 is no contraction at all."""
    prob = nlls.large_residual_problem(40, 3.0)
    out = nlls.gauss_newton(prob["residual"], prob["jacobian"], [0.3, 3.0], max_iter=300)
    info = nlls.dropped_term(prob["residual"], prob["jacobian"], prob["hessians"], out.x)
    assert info["ratio"] > 0.9, f"expected a ratio near 1, got {info['ratio']:.3f}"
    assert not out.converged, "this problem is supposed to be the hard one"


# ---------------------------------------------------------------- Levenberg-Marquardt


@pytest.mark.parametrize("n_points", [20, 60])
@pytest.mark.parametrize("scaled", [True, False])
def test_levenberg_marquardt_solves_the_exact_fit(n_points, scaled):
    prob = nlls.exponential_model(n_points, [2.0, -0.7])
    out = nlls.levenberg_marquardt(prob["residual"], prob["jacobian"], [1.0, -0.2],
                                   scaled=scaled, max_iter=300)
    assert out.converged, out.message
    assert np.linalg.norm(out.x - prob["truth"]) < 1e-7


def test_levenberg_marquardt_beats_gauss_newton_on_a_hard_residual():
    """Measured at offset 10: Gauss-Newton takes 79 iterations to reach ||r|| = 65.84 and
    Levenberg-Marquardt takes 20 to reach 60.19, which is a BETTER minimum as well as faster."""
    prob = nlls.large_residual_problem(40, 10.0)
    gn = nlls.gauss_newton(prob["residual"], prob["jacobian"], [1.5, 1.1], max_iter=300)
    lm = nlls.levenberg_marquardt(prob["residual"], prob["jacobian"], [1.5, 1.1], max_iter=300)
    assert lm.iterations < gn.iterations / 2, f"LM {lm.iterations}, GN {gn.iterations}"
    assert lm.residual_norm < gn.residual_norm


def test_levenberg_marquardt_is_sometimes_worse():
    """Damping keeps the iterate near where it started, so it finds the NEAREST minimum. From
    a start where the undamped long step escapes to a better basin, that is a disadvantage.
    Measured: Gauss-Newton reaches ||r|| = 5.99 and Levenberg-Marquardt settles at 12.16."""
    prob = nlls.large_residual_problem(40, 1.0)
    gn = nlls.gauss_newton(prob["residual"], prob["jacobian"], [0.1, 4.5], max_iter=300)
    lm = nlls.levenberg_marquardt(prob["residual"], prob["jacobian"], [0.1, 4.5], max_iter=300)
    assert gn.residual_norm < lm.residual_norm, \
        f"expected GN {gn.residual_norm:.3f} to beat LM {lm.residual_norm:.3f} here"


def test_the_damping_history_falls_when_steps_succeed():
    """lam is divided by `down` on every accepted step, so on an easy problem it decays."""
    prob = nlls.exponential_model(40, [2.0, -0.7])
    out = nlls.levenberg_marquardt(prob["residual"], prob["jacobian"], [1.0, -0.2],
                                   lam0=1.0, max_iter=100)
    assert out.lambdas, "no damping history recorded"
    assert out.lambdas[-1] < out.lambdas[0], f"lam went {out.lambdas[0]} -> {out.lambdas[-1]}"


def test_large_damping_gives_a_short_step_along_the_gradient():
    """As lam -> infinity the step tends to -J^T r / lam, which is steepest descent. Checked by
    comparing directions, since the lengths differ by construction."""
    prob = nlls.exponential_model(30, [2.0, -0.7])
    x = np.array([1.0, -0.2])
    J = prob["jacobian"](x)
    r = prob["residual"](x)
    grad = -(J.T @ r)
    lam = 1e10
    d = np.sqrt(np.maximum(np.sum(J * J, axis=0), 1e-300))
    stacked = np.vstack([J, np.sqrt(lam) * np.diag(d)])
    p = np.linalg.lstsq(stacked, np.concatenate([-r, np.zeros(x.size)]), rcond=None)[0]
    # Marquardt scaling changes the metric, so compare against the scaled gradient.
    scaled_grad = grad / d ** 2
    cosine = float(p @ scaled_grad / (np.linalg.norm(p) * np.linalg.norm(scaled_grad)))
    assert cosine > 0.999, f"cosine to the scaled gradient is only {cosine:.6f}"


# ---------------------------------------------------------------- full Newton, for comparison


@pytest.mark.parametrize("offset", [1.0, 3.0, 6.0])
def test_full_newton_converges_to_a_WORSE_point(offset):
    """**Keeping the exact Hessian is not simply better, and this is the important test here.**

    ``J^T J`` is positive semidefinite whatever the problem, so the Gauss-Newton step is always
    a descent direction. The exact Hessian ``J^T J + S`` can be indefinite, and Newton
    converges to the nearest *stationary point*, which may be a saddle or a degenerate minimum.

    Measured at offset 3: Gauss-Newton reaches ``||r|| = 18.01`` and Newton settles at
    ``22.87``, at the point ``a = 2.8e-16``, where the model has collapsed to zero and the
    residual no longer depends on ``b`` at all. Newton reports convergence, correctly: the
    gradient really is zero there.
    """
    prob = nlls.large_residual_problem(40, offset)
    gn = nlls.gauss_newton(prob["residual"], prob["jacobian"], [1.5, 1.1], max_iter=300)
    nt = nlls.newton_nlls(prob["residual"], prob["jacobian"], prob["hessians"],
                          [1.5, 1.1], max_iter=100)
    assert nt.converged, "Newton is supposed to reach a stationary point here"
    assert nt.residual_norm > gn.residual_norm, \
        f"offset {offset}: Newton {nt.residual_norm:.3f}, Gauss-Newton {gn.residual_norm:.3f}"


def test_gauss_newtons_approximation_is_always_positive_semidefinite():
    """The reason for the previous test, checked directly: J^T J has no negative eigenvalues at
    any point, while J^T J + S does at the points these iterations pass through."""
    prob = nlls.large_residual_problem(40, 6.0)
    gen = np.random.default_rng(3)
    saw_indefinite = 0
    for _ in range(30):
        x = np.array([gen.uniform(-4.0, 4.0), gen.uniform(0.2, 3.0)])
        J = prob["jacobian"](x)
        r = prob["residual"](x)
        S = np.einsum("i,ijk->jk", r, prob["hessians"](x))
        assert np.min(np.linalg.eigvalsh(J.T @ J)) >= -1e-10, "J^T J went indefinite"
        if np.min(np.linalg.eigvalsh(J.T @ J + S)) < 0.0:
            saw_indefinite += 1
    assert saw_indefinite > 5, f"the exact Hessian was indefinite at only {saw_indefinite}/30 points"


def test_full_newton_takes_fewer_iterations_when_the_residual_is_large():
    """It keeps the term Gauss-Newton drops, so its rate does not degrade with the residual.
    Measured at offset 6: 7 iterations against 16. It just arrives somewhere worse."""
    prob = nlls.large_residual_problem(40, 6.0)
    gn = nlls.gauss_newton(prob["residual"], prob["jacobian"], [1.5, 1.1], max_iter=300)
    nt = nlls.newton_nlls(prob["residual"], prob["jacobian"], prob["hessians"],
                          [1.5, 1.1], max_iter=100)
    assert nt.iterations < gn.iterations, f"Newton {nt.iterations}, Gauss-Newton {gn.iterations}"


def test_newton_rejects_a_wrong_hessian_shape():
    prob = nlls.large_residual_problem(20, 1.0)
    with pytest.raises(ValueError):
        nlls.newton_nlls(prob["residual"], prob["jacobian"],
                         lambda x: np.zeros((3, 2, 2)), [1.5, 1.1])


def test_dropped_term_rejects_a_wrong_hessian_shape():
    prob = nlls.large_residual_problem(20, 1.0)
    with pytest.raises(ValueError):
        nlls.dropped_term(prob["residual"], prob["jacobian"],
                          lambda x: np.zeros((3, 2, 2)), [1.5, 1.1])


# ---------------------------------------------------------------- the finite difference fallback


@pytest.mark.parametrize("n_points", [20, 50])
def test_a_missing_jacobian_is_supplied_by_finite_differences(n_points):
    """The same answer, at a looser tolerance, and many more residual evaluations."""
    prob = nlls.exponential_model(n_points, [2.0, -0.7])
    exact = nlls.gauss_newton(prob["residual"], prob["jacobian"], [1.0, -0.2], max_iter=100)
    approx = nlls.gauss_newton(prob["residual"], None, [1.0, -0.2], max_iter=100)
    assert approx.converged, approx.message
    assert np.linalg.norm(approx.x - exact.x) < 1e-6
    assert approx.n_feval > exact.n_feval, "finite differences should cost extra evaluations"


def test_a_wrong_jacobian_shape_is_rejected():
    prob = nlls.exponential_model(20, [2.0, -0.7])
    with pytest.raises(ValueError):
        nlls.gauss_newton(prob["residual"], lambda x: np.zeros((5, 2)), [1.0, -0.2])


def test_an_underdetermined_problem_is_rejected():
    """Two residuals cannot determine three parameters, and least squares does not fix that."""
    with pytest.raises(ValueError):
        nlls.gauss_newton(lambda x: np.zeros(2), None, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------- GPS


@pytest.mark.parametrize("k", [4, 6, 9, 12])
def test_gps_recovers_the_position_from_a_cold_start(k):
    """Starting from the centre of the Earth with no clock knowledge at all."""
    gen = np.random.default_rng(100 + k)
    radius = 26.56e6
    ang = gen.uniform(0.0, 2.0 * np.pi, (k, 2))
    S = np.column_stack([radius * np.sin(ang[:, 0]) * np.cos(ang[:, 1]),
                         radius * np.sin(ang[:, 0]) * np.sin(ang[:, 1]),
                         radius * np.cos(ang[:, 0])])
    p = np.array([4.0e6, 1.0e6, 4.5e6])
    prob = nlls.gps_problem(S, p, clock_bias=3.0e4, noise=5.0, rng=np.random.default_rng(1))
    out = nlls.levenberg_marquardt(prob["residual"], prob["jacobian"],
                                   np.zeros(4), max_iter=300)
    assert out.converged, out.message
    assert np.linalg.norm(out.x[:3] - p) < 100.0, \
        f"{k} satellites: position error {np.linalg.norm(out.x[:3] - p):.1f} m"


def test_gps_is_exact_without_noise():
    gen = np.random.default_rng(5)
    S = gen.standard_normal((6, 3)) * 2.0e7
    p = np.array([1.0e6, -2.0e6, 3.0e6])
    prob = nlls.gps_problem(S, p, clock_bias=1.0e4, noise=0.0)
    out = nlls.levenberg_marquardt(prob["residual"], prob["jacobian"], np.zeros(4),
                                   max_iter=300)
    assert np.linalg.norm(out.x - prob["truth"]) < 1e-4 * np.linalg.norm(prob["truth"])


@pytest.mark.parametrize("d,k", [(2, 4), (2, 7), (3, 5), (4, 8)])
def test_gps_works_in_any_dimension(d, k):
    """The dimension is read from the satellite array, not assumed to be 3."""
    gen = np.random.default_rng(20 + d * k)
    S = gen.standard_normal((k, d)) * 1000.0
    p = gen.standard_normal(d) * 50.0
    prob = nlls.gps_problem(S, p, clock_bias=7.5, noise=0.0)
    assert prob["dimension"] == d
    out = nlls.levenberg_marquardt(prob["residual"], prob["jacobian"], np.zeros(d + 1),
                                   max_iter=400)
    assert np.linalg.norm(out.x[:d] - p) < 1e-4 * max(np.linalg.norm(p), 1.0)


def test_gps_rejects_too_few_satellites():
    """d coordinates plus a clock bias need d+1 measurements. Three satellites is not enough
    for a three dimensional fix, which is why receivers wait for a fourth."""
    with pytest.raises(ValueError):
        nlls.gps_problem(np.zeros((3, 3)), np.zeros(3))


def test_gps_rejects_a_dimension_mismatch():
    with pytest.raises(ValueError):
        nlls.gps_problem(np.zeros((6, 3)), np.zeros(2))


# ---------------------------------------------------------------- variable projection


@pytest.mark.parametrize("n_points", [20, 50, 120])
def test_variable_projection_finds_the_exact_linear_part(n_points):
    """With the nonlinear parameter correct, the linear ones come from one lstsq and the
    residual is zero."""
    t = np.linspace(0.0, 3.0, n_points)
    basis = lambda tt, b: np.column_stack([np.exp(b * tt), np.ones_like(tt)])
    y = 2.5 * np.exp(-0.8 * t) + 1.25
    out = nlls.separable_fit(t, y, basis, -0.8)
    assert out["residual_norm"] < 1e-10
    assert np.allclose(out["coefficients"], [2.5, 1.25], atol=1e-8)


def test_variable_projection_residual_is_minimal_at_the_true_parameter():
    """Sweeping the nonlinear parameter, the reduced residual has its minimum at the truth."""
    t = np.linspace(0.0, 3.0, 60)
    basis = lambda tt, b: np.column_stack([np.exp(b * tt), np.ones_like(tt)])
    y = 2.5 * np.exp(-0.8 * t) + 1.25
    sweep = np.linspace(-2.0, -0.2, 91)
    norms = [nlls.separable_fit(t, y, basis, b)["residual_norm"] for b in sweep]
    assert abs(sweep[int(np.argmin(norms))] - (-0.8)) < 0.03


def test_separable_fit_rejects_mismatched_inputs():
    basis = lambda tt, b: np.column_stack([np.exp(b * tt), np.ones_like(tt)])
    with pytest.raises(ValueError):
        nlls.separable_fit(np.arange(10.0), np.arange(9.0), basis, -0.5)
    with pytest.raises(ValueError):
        nlls.separable_fit(np.arange(10.0), np.arange(10.0),
                           lambda tt, b: np.zeros((5, 2)), -0.5)


# ---------------------------------------------------------------- rate measurement


def test_observed_rate_recognises_a_quadratic_sequence():
    e = [1e-1]
    for _ in range(5):
        e.append(e[-1] ** 2)
    assert abs(nlls.observed_rate(e)["order"] - 2.0) < 0.05


def test_observed_rate_recognises_a_linear_sequence():
    e = [0.5 ** k for k in range(10)]
    rate = nlls.observed_rate(e)
    assert abs(rate["order"] - 1.0) < 0.05
    assert abs(rate["linear_factor"] - 0.5) < 0.01


def test_observed_rate_refuses_a_sequence_too_short_to_fit():
    assert np.isnan(nlls.observed_rate([1.0, 0.5])["order"])


# ---------------------------------------------------------------- problem generators


@pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0]])
def test_exponential_model_rejects_the_wrong_parameter_count(bad):
    with pytest.raises(ValueError):
        nlls.exponential_model(20, bad)


def test_exponential_model_rejects_too_few_points():
    with pytest.raises(ValueError):
        nlls.exponential_model(1, [2.0, -0.7])


def test_large_residual_problem_rejects_too_few_points():
    with pytest.raises(ValueError):
        nlls.large_residual_problem(2, 1.0)


@pytest.mark.parametrize("n_points", [10, 40, 100])
def test_the_analytic_jacobians_match_finite_differences(n_points):
    """Every generator ships a hand-derived Jacobian, and a wrong one would quietly change the
    convergence rate rather than raise, so it is checked."""
    from nalib.nlsystems import numerical_jacobian

    for prob, point in ((nlls.exponential_model(n_points, [2.0, -0.7]), [1.4, -0.5]),
                        (nlls.large_residual_problem(n_points, 2.0), [1.7, 1.1])):
        exact = prob["jacobian"](point)
        approx = numerical_jacobian(prob["residual"], point)
        rel = np.linalg.norm(exact - approx) / max(np.linalg.norm(exact), 1e-300)
        assert rel < 1e-6, f"analytic and numerical Jacobians differ by {rel:.2e}"


@pytest.mark.parametrize("n_points", [10, 40])
def test_the_analytic_hessians_match_finite_differences(n_points):
    """Same argument, one derivative further up, where mistakes are easier to make."""
    for prob, point in ((nlls.exponential_model(n_points, [2.0, -0.7]), np.array([1.4, -0.5])),
                        (nlls.large_residual_problem(n_points, 2.0), np.array([1.7, 1.1]))):
        exact = prob["hessians"](point)
        h = 1e-5
        approx = np.zeros_like(exact)
        for j in range(point.size):
            step = np.zeros(point.size)
            step[j] = h
            approx[:, :, j] = (prob["jacobian"](point + step)
                               - prob["jacobian"](point - step)) / (2 * h)
        rel = np.abs(exact - approx).max() / max(np.abs(exact).max(), 1e-300)
        assert rel < 1e-5, f"analytic and numerical Hessians differ by {rel:.2e}"
