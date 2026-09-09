"""Tests for nalib.nlsystems.

Every solver here is checked three ways:

1. the returned point really is a root, so ``||F(x)||`` is at the noise level,
2. the returned point matches an independently known exact root,
3. the claimed convergence behaviour actually shows up in the iteration history.

Where a library equivalent exists (`scipy.optimize.root`), the answer is compared against it as
well. The defining identity for Broyden, the secant condition, is checked directly on the update
formula rather than being taken on trust.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import nlsystems as ns

TOL = 1e-12


# ---------------------------------------------------------------- test systems


def circle_line():
    """x^2 + y^2 = 4 and y = x. Exact roots (+-sqrt(2), +-sqrt(2))."""

    def F(v):
        x, y = v
        return np.array([x * x + y * y - 4.0, y - x])

    def J(v):
        x, y = v
        return np.array([[2 * x, 2 * y], [-1.0, 1.0]])

    r = np.sqrt(2.0)
    return F, J, np.array([r, r]), np.array([-r, -r])


def rosenbrock_gradient():
    """Gradient of the Rosenbrock function. Single root at (1, 1), badly scaled."""

    def F(v):
        x, y = v
        return np.array([-400.0 * x * (y - x * x) - 2.0 * (1.0 - x),
                         200.0 * (y - x * x)])

    def J(v):
        x, y = v
        return np.array([[-400.0 * y + 1200.0 * x * x + 2.0, -400.0 * x],
                         [-400.0 * x, 200.0]])

    return F, J, np.array([1.0, 1.0])


def exp_system():
    """A 3x3 system with no closed form, used to test against scipy."""

    def F(v):
        x, y, z = v
        return np.array([3 * x - np.cos(y * z) - 0.5,
                         x * x - 81 * (y + 0.1) ** 2 + np.sin(z) + 1.06,
                         np.exp(-x * y) + 20 * z + (10 * np.pi - 3) / 3])

    return F


# ---------------------------------------------------------------- Jacobian


def test_numerical_jacobian_matches_the_analytic_one():
    F, J, _, _ = circle_line()
    for x in [np.array([1.0, 2.0]), np.array([-3.0, 0.5]), np.array([0.1, -0.1])]:
        np.testing.assert_allclose(ns.numerical_jacobian(F, x), J(x), rtol=1e-7, atol=1e-8)


def test_numerical_jacobian_is_accurate_to_about_eps_two_thirds():
    """Central differences with the eps^(1/3) step should give about 1e-11 relative error."""
    F, J, _, _ = circle_line()
    x = np.array([1.3, -0.7])
    err = np.max(np.abs(ns.numerical_jacobian(F, x) - J(x)) / (1 + np.abs(J(x))))
    assert err < 1e-9, f"central difference Jacobian error {err:.2e} is worse than expected"


def test_numerical_jacobian_handles_a_nonsquare_map():
    def F(v):
        x, y = v
        return np.array([x + y, x - y, x * y])

    Jn = ns.numerical_jacobian(F, np.array([2.0, 3.0]))
    assert Jn.shape == (3, 2)
    np.testing.assert_allclose(Jn, [[1, 1], [1, -1], [3, 2]], rtol=1e-6, atol=1e-7)


def test_numerical_jacobian_costs_two_n_evaluations():
    calls = {"n": 0}

    def F(v):
        calls["n"] += 1
        return np.asarray(v, dtype=float) ** 2

    ns.numerical_jacobian(F, np.ones(5))
    assert calls["n"] == 2 * 5 + 1        # 2n differences plus the base point


# ---------------------------------------------------------------- spectral radius


def test_spectral_radius_of_a_diagonal_matrix_is_the_largest_entry():
    assert ns.spectral_radius(np.diag([0.3, -0.9, 0.5])) == pytest.approx(0.9)


def test_spectral_radius_ignores_the_norm():
    """A matrix can have norm above 1 and spectral radius below it. That is the whole point."""
    A = np.array([[0.0, 100.0], [0.0, 0.0]])
    assert np.linalg.norm(A) == pytest.approx(100.0)
    assert ns.spectral_radius(A) == pytest.approx(0.0, abs=1e-12)


def test_spectral_radius_matches_numpy_on_random_matrices(rng):
    for _ in range(10):
        A = rng.standard_normal((6, 6))
        assert ns.spectral_radius(A) == pytest.approx(
            float(np.max(np.abs(np.linalg.eigvals(A)))), rel=1e-12
        )


# ---------------------------------------------------------------- fixed point


def fixed_point_pair():
    """G with a fixed point at (0.5, 0.25) and a contracting Jacobian there."""

    def G(v):
        x, y = v
        return np.array([(y + 0.5) / 3.0 + 0.25, (x * x + 1.0) / 5.0])

    return G


def test_fixed_point_system_converges_to_a_genuine_fixed_point():
    G = fixed_point_pair()
    res = ns.fixed_point_system(G, [0.0, 0.0], tol=TOL)
    assert res.converged
    np.testing.assert_allclose(G(res.root), res.root, atol=1e-11)


def test_fixed_point_converges_exactly_when_the_spectral_radius_is_below_one():
    G = fixed_point_pair()
    res = ns.fixed_point_system(G, [0.0, 0.0], tol=TOL)
    JG = ns.numerical_jacobian(G, res.root)
    assert ns.spectral_radius(JG) < 1.0
    assert res.converged


def test_fixed_point_diverges_when_the_spectral_radius_exceeds_one():
    def G(v):
        x, y = v
        return np.array([2.0 * y + 0.1, 2.0 * x + 0.1])

    assert ns.spectral_radius(ns.numerical_jacobian(G, [0.0, 0.0])) > 1.0
    res = ns.fixed_point_system(G, [1.0, 1.0], tol=TOL, max_iter=200)
    assert not res.converged


def test_fixed_point_error_decays_at_the_rate_the_spectral_radius_predicts():
    G = fixed_point_pair()
    res = ns.fixed_point_system(G, [0.0, 0.0], tol=1e-14)
    rho = ns.spectral_radius(ns.numerical_jacobian(G, res.root))
    e = res.errors(res.root)
    good = e > 1e-13
    ratios = e[1:][good[1:]] / e[:-1][good[1:]]
    observed = float(np.mean(ratios[-5:])) if ratios.size >= 5 else float(np.mean(ratios))
    assert observed < 1.0
    assert observed == pytest.approx(rho, abs=0.25), (
        f"observed contraction {observed:.4f} against rho = {rho:.4f}"
    )


def test_seidel_beats_jacobi_on_the_same_system():
    """Fresher information should not cost more iterations, and here it costs fewer."""
    G = fixed_point_pair()
    j = ns.fixed_point_system(G, [0.0, 0.0], tol=TOL)
    s = ns.seidel_system(G, [0.0, 0.0], tol=TOL)
    assert j.converged and s.converged
    np.testing.assert_allclose(j.root, s.root, atol=1e-10)
    assert s.n_iter <= j.n_iter


# ---------------------------------------------------------------- Newton


def test_newton_system_finds_the_nearest_root():
    F, J, rp, rn = circle_line()
    np.testing.assert_allclose(ns.newton_system(F, J, [1.0, 1.5], tol=TOL).root, rp, atol=1e-12)
    np.testing.assert_allclose(ns.newton_system(F, J, [-1.0, -1.5], tol=TOL).root, rn, atol=1e-12)


def test_newton_residual_reaches_the_noise_level():
    F, J, _, _ = circle_line()
    res = ns.newton_system(F, J, [1.0, 1.5], tol=TOL)
    assert res.converged
    assert np.linalg.norm(F(res.root)) < 1e-14


def test_newton_with_a_numerical_jacobian_agrees_with_the_analytic_one():
    F, J, rp, _ = circle_line()
    a = ns.newton_system(F, J, [1.0, 1.5], tol=1e-10)
    b = ns.newton_system(F, None, [1.0, 1.5], tol=1e-10)
    np.testing.assert_allclose(a.root, b.root, atol=1e-9)
    assert b.n_feval > a.n_feval, "the difference Jacobian must cost more F evaluations"


def test_newton_converges_quadratically():
    """Each error should be about the square of the previous one."""
    F, J, rp, _ = circle_line()
    res = ns.newton_system(F, J, [1.0, 1.6], tol=1e-15, max_iter=50)
    e = res.errors(rp)
    e = e[e > 1e-13]
    assert e.size >= 3, "need at least three clean errors to see the rate"
    orders = np.log(e[2:]) / np.log(e[1:-1])
    assert np.max(orders) > 1.7, f"orders {orders} do not look quadratic"


def test_newton_reports_a_singular_jacobian_instead_of_crashing():
    def F(v):
        x, y = v
        return np.array([x * x - y, 2.0 * (x * x - y)])       # rows are dependent

    def J(v):
        x, _ = v
        return np.array([[2 * x, -1.0], [4 * x, -2.0]])

    res = ns.newton_system(F, J, [1.0, 1.0], tol=TOL)
    assert not res.converged
    assert "singular" in res.message


def test_newton_matches_scipy_on_a_three_by_three_system():
    scipy_opt = pytest.importorskip("scipy.optimize")
    F = exp_system()
    mine = ns.newton_system(F, None, [0.1, 0.1, -0.1], tol=1e-13)
    theirs = scipy_opt.root(F, [0.1, 0.1, -0.1], tol=1e-13)
    assert mine.converged and theirs.success
    np.testing.assert_allclose(mine.root, theirs.x, atol=1e-9)


def test_newton_counts_one_jacobian_per_iteration():
    F, J, _, _ = circle_line()
    res = ns.newton_system(F, J, [1.0, 1.5], tol=TOL)
    assert res.n_jeval == res.n_iter


# ---------------------------------------------------------------- damped Newton


def overshoot_system():
    """x^2 + y^2 = 4 and e^x + y = 1. Plain Newton overshoots from a start far below."""

    def F(v):
        x, y = v
        return np.array([x * x + y * y - 4.0, np.exp(x) + y - 1.0])

    def J(v):
        x, y = v
        return np.array([[2 * x, 2 * y], [np.exp(x), 1.0]])

    return F, J


def test_plain_newton_fails_where_damped_newton_succeeds():
    """The reason damping exists: a start where the full step lands somewhere worse."""
    F, J = overshoot_system()
    start = [-1.5, -8.0]
    with np.errstate(over="ignore", invalid="ignore"):
        plain = ns.newton_system(F, J, start, tol=1e-12, max_iter=60)
        damped = ns.damped_newton(F, J, start, tol=1e-12, max_iter=60)
    assert not plain.converged, "this start is meant to defeat plain Newton"
    assert damped.converged
    assert np.linalg.norm(F(damped.root)) < 1e-10


def test_damping_can_be_far_slower_than_plain_newton():
    """The limitation nobody mentions. Both find the root; damping needs hundreds of steps.

    The Rosenbrock valley is nearly flat in ||F||, so every full Newton step from inside it
    raises the residual and gets rejected. Plain Newton accepts that rise and is done quickly.
    Monotone decrease of a merit function is a weaker property than it sounds.
    """
    F, J, exact = rosenbrock_gradient()
    start = [-3.0, -4.0]
    plain = ns.newton_system(F, J, start, tol=1e-12, max_iter=3000)
    damped = ns.damped_newton(F, J, start, tol=1e-12, max_iter=3000)

    assert plain.converged and damped.converged
    np.testing.assert_allclose(plain.root, exact, atol=1e-10)
    np.testing.assert_allclose(damped.root, exact, atol=1e-8)
    assert damped.n_iter > 20 * plain.n_iter, (
        f"expected damping to be far slower: plain {plain.n_iter}, damped {damped.n_iter}"
    )


def test_damped_newton_never_increases_the_residual():
    """This is the property the line search enforces, so it must hold at every single step."""
    F, J, _ = rosenbrock_gradient()
    res = ns.damped_newton(F, J, [-3.0, -4.0], tol=1e-12, max_iter=80)
    r = res.residuals
    assert np.all(np.diff(r) < 0), f"residual increased somewhere: {r}"


def test_damped_newton_takes_full_steps_near_the_root():
    """Damping must switch itself off close in, or the quadratic rate is lost."""
    F, J, exact = circle_line()[0], circle_line()[1], circle_line()[2]
    near = ns.damped_newton(F, J, exact + 1e-3, tol=1e-14)
    plain = ns.newton_system(F, J, exact + 1e-3, tol=1e-14)
    assert near.n_iter <= plain.n_iter + 1


def test_damped_newton_reports_a_failed_line_search():
    def F(v):
        return np.array([np.exp(v[0]) + 1.0, v[1]])       # exp(x) + 1 has no real root

    res = ns.damped_newton(F, None, [0.0, 0.0], tol=TOL, max_iter=200)
    assert not res.converged


# ---------------------------------------------------------------- Broyden


def test_broyden_secant_condition_holds_after_the_update():
    """B_new s = y is the defining identity. Check it on the update formula directly."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = 4
        B = rng.standard_normal((n, n))
        s = rng.standard_normal(n)
        y = rng.standard_normal(n)
        B_new = B + np.outer(y - B @ s, s) / float(s @ s)
        np.testing.assert_allclose(B_new @ s, y, rtol=1e-12, atol=1e-12)


def test_broyden_update_is_the_smallest_change_that_does_it():
    """Any other matrix meeting the secant condition is at least as far from B in Frobenius."""
    rng = np.random.default_rng(7)
    n = 4
    B = rng.standard_normal((n, n))
    s = rng.standard_normal(n)
    y = rng.standard_normal(n)
    B_broyden = B + np.outer(y - B @ s, s) / float(s @ s)
    d_broyden = np.linalg.norm(B_broyden - B, "fro")

    # any C with C s = y can be written as B_broyden + M where M s = 0
    for _ in range(50):
        M = rng.standard_normal((n, n))
        M = M - np.outer(M @ s, s) / float(s @ s)         # project so that M s = 0
        np.testing.assert_allclose(M @ s, 0.0, atol=1e-10)
        C = B_broyden + M
        np.testing.assert_allclose(C @ s, y, atol=1e-10)
        assert np.linalg.norm(C - B, "fro") >= d_broyden - 1e-12


def test_broyden_finds_the_root():
    F, _, rp, _ = circle_line()
    res = ns.broyden(F, [1.0, 1.5], tol=1e-13)
    assert res.converged
    np.testing.assert_allclose(res.root, rp, atol=1e-10)


def test_broyden_uses_fewer_f_evaluations_than_difference_newton():
    """The point of Broyden: no Jacobian after the first one."""
    F, _, _, _ = circle_line()
    b = ns.broyden(F, [1.0, 1.5], tol=1e-12)
    n = ns.newton_system(F, None, [1.0, 1.5], tol=1e-12)
    assert b.converged and n.converged
    assert b.n_feval < n.n_feval, f"Broyden used {b.n_feval}, Newton used {n.n_feval}"


def test_broyden_is_superlinear_but_slower_than_newton():
    F, J, rp, _ = circle_line()
    b = ns.broyden(F, [1.0, 1.6], tol=1e-14, max_iter=100)
    n = ns.newton_system(F, J, [1.0, 1.6], tol=1e-14, max_iter=100)
    assert b.n_iter > n.n_iter, "Broyden should need more steps than Newton"

    e = b.errors(rp)
    e = e[e > 1e-13]
    ratios = e[1:] / e[:-1]
    assert ratios[-1] < ratios[0], "the ratio must keep shrinking, which is superlinear"


def test_broyden_accepts_a_supplied_starting_matrix():
    F, J, rp, _ = circle_line()
    res = ns.broyden(F, [1.0, 1.5], B0=J([1.0, 1.5]), tol=1e-13)
    assert res.converged
    np.testing.assert_allclose(res.root, rp, atol=1e-10)
    assert res.n_jeval == 0, "Broyden never evaluates a Jacobian"


def test_broyden_with_the_identity_start_still_works_or_reports_failure():
    """A poor B0 is allowed to fail, but it must never claim a wrong answer."""
    F, _, rp, rn = circle_line()
    res = ns.broyden(F, [1.0, 1.5], B0=np.eye(2), tol=1e-12, max_iter=300)
    if res.converged:
        assert np.linalg.norm(F(res.root)) < 1e-9


# ---------------------------------------------------------------- shared contract


ALL_SOLVERS = ["newton", "damped", "broyden"]


def run(name, start, tol=1e-12):
    F, J, _, _ = circle_line()
    if name == "newton":
        return ns.newton_system(F, J, start, tol=tol)
    if name == "damped":
        return ns.damped_newton(F, J, start, tol=tol)
    return ns.broyden(F, start, tol=tol)


@pytest.mark.parametrize("name", ALL_SOLVERS)
def test_every_solver_returns_a_consistent_history(name):
    res = run(name, [1.0, 1.5])
    assert res.iterates.shape == (res.n_iter + 1, 2)
    assert res.residuals.size == res.n_iter + 1
    np.testing.assert_allclose(res.root, res.iterates[-1])
    assert res.n_feval > 0


@pytest.mark.parametrize("name", ALL_SOLVERS)
def test_every_solver_that_claims_convergence_really_converged(name):
    F, _, _, _ = circle_line()
    res = run(name, [1.0, 1.5])
    assert res.converged
    assert np.linalg.norm(F(res.root)) < 1e-10, "claimed convergence with a large residual"


@pytest.mark.parametrize("name", ALL_SOLVERS)
def test_every_solver_agrees_on_the_same_root(name):
    _, _, rp, _ = circle_line()
    np.testing.assert_allclose(run(name, [1.0, 1.5]).root, rp, atol=1e-9)


@pytest.mark.parametrize("name", ALL_SOLVERS)
def test_errors_helper_is_zero_at_the_root(name):
    res = run(name, [1.0, 1.5])
    assert res.errors(res.root)[-1] == pytest.approx(0.0, abs=1e-15)


def test_result_repr_is_readable():
    text = repr(run("newton", [1.0, 1.5]))
    assert "SystemResult" in text and "converged=True" in text


# ---------------------------------------------------------------- scaling


def test_newton_works_in_higher_dimensions():
    """A 20-variable system, to show nothing in the code is hard-wired to n = 2."""
    n = 20
    rng = np.random.default_rng(42)
    target = rng.standard_normal(n)
    A = rng.standard_normal((n, n)) + n * np.eye(n)        # well conditioned

    def F(v):
        v = np.asarray(v, dtype=float)
        return A @ (v - target) + 0.1 * (v - target) ** 3

    def J(v):
        v = np.asarray(v, dtype=float)
        return A + np.diag(0.3 * (v - target) ** 2)

    res = ns.newton_system(F, J, target + 0.3, tol=1e-13)
    assert res.converged
    np.testing.assert_allclose(res.root, target, atol=1e-10)


def test_broyden_works_in_higher_dimensions():
    n = 10
    rng = np.random.default_rng(42)
    target = rng.standard_normal(n)
    A = rng.standard_normal((n, n)) + n * np.eye(n)

    def F(v):
        v = np.asarray(v, dtype=float)
        return A @ (v - target) + 0.1 * (v - target) ** 3

    res = ns.broyden(F, target + 0.2, tol=1e-12, max_iter=200)
    assert res.converged
    np.testing.assert_allclose(res.root, target, atol=1e-9)
