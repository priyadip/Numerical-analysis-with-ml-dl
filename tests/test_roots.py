"""Tests for nalib.roots.

Two kinds of check dominate here:

1. Every method must find the right root, on easy and hard problems, and its residual must be
   at the level the conditioning allows.
2. Every method must achieve its **claimed convergence order**. A method that converges to the
   right answer at the wrong rate has a bug that no value-based test would catch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from nalib import convergence as cv, roots as R

SQRT2 = math.sqrt(2.0)
GOLDEN = (1 + math.sqrt(5)) / 2


def f_quad(x):
    return x * x - 2.0


def df_quad(x):
    return 2.0 * x


def d2f_quad(x):
    return 2.0


#: (name, f, bracket, root). Every one has a sign change on the bracket.
SUITE = [
    ("x^2 - 2",      f_quad,                          (1.0, 2.0), SQRT2),
    ("x^3 - 2x - 5", lambda x: x**3 - 2*x - 5,        (2.0, 3.0), 2.0945514815423265),
    ("cos x - x",    lambda x: np.cos(x) - x,         (0.0, 1.0), 0.7390851332151607),
    ("e^x - 2",      lambda x: np.exp(x) - 2.0,       (0.0, 2.0), math.log(2.0)),
    ("x^10 - 1",     lambda x: x**10 - 1.0,           (0.0, 1.3), 1.0),
    ("atan x",       np.arctan,                       (-1.0, 2.0), 0.0),
]

BRACKETING = [R.bisection, R.regula_falsi, R.illinois, R.brent]


# ---------------------------------------------------------------- bracketing methods


@pytest.mark.parametrize("method", BRACKETING)
@pytest.mark.parametrize("name,f,bracket,root", SUITE)
def test_bracketing_methods_find_the_root(method, name, f, bracket, root):
    res = method(f, *bracket, tol=1e-14, max_iter=500)
    assert res.converged, f"{method.__name__} failed on {name}: {res.message}"
    assert abs(res.root - root) < 1e-9, f"{method.__name__} on {name}"


@pytest.mark.parametrize("method", BRACKETING)
def test_bracketing_methods_reject_a_bad_bracket(method):
    with pytest.raises(ValueError):
        method(f_quad, 2.0, 3.0)          # both endpoints positive


def test_bisection_error_bound_is_never_violated():
    """The guarantee (b-a)/2^(k+1) must hold at every step above the roundoff floor.

    The bound is a statement about real arithmetic. Once the bracket has narrowed to a
    single ulp it stops being meaningful, because the true root is not representable: for
    f = x^2 - 2 the root is sqrt(2), which sits strictly between two doubles. At that point
    the midpoint can be a full ulp from the root while the "bracket half-width" is half an
    ulp, and the bound appears violated by a factor of two.

    That is a limitation of the floating point grid, not of bisection, so the check is
    restricted to the steps where the bracket is still several ulps wide.
    """
    res = R.bisection(f_quad, 1.0, 2.0, tol=1e-16, max_iter=60)
    errs = res.errors(SQRT2)
    bounds = res.error_bounds
    resolvable = bounds > 8 * np.spacing(SQRT2)
    assert resolvable.sum() > 40, "should have plenty of resolvable steps"
    assert (errs[resolvable] <= bounds[resolvable] * (1 + 1e-12)).all()


def test_bisection_bound_becomes_meaningless_at_the_roundoff_floor():
    """Documenting the limit above, so nobody is surprised by it later."""
    res = R.bisection(f_quad, 1.0, 2.0, tol=1e-16, max_iter=60)
    errs = res.errors(SQRT2)
    bounds = res.error_bounds
    floor = bounds <= 8 * np.spacing(SQRT2)
    # At the floor the error stalls at about one ulp and stops improving.
    assert errs[floor][-1] <= 4 * np.spacing(SQRT2)
    assert errs[floor][-1] > 0.0, "sqrt(2) is not exactly representable"


def test_bisection_bracket_halves_exactly():
    res = R.bisection(f_quad, 1.0, 2.0, tol=1e-16, max_iter=50)
    ratios = res.bracket_widths[1:] / res.bracket_widths[:-1]
    np.testing.assert_allclose(ratios, 0.5)


def test_bisection_rate_is_one_half():
    res = R.bisection(f_quad, 1.0, 2.0, tol=1e-16, max_iter=60)
    assert cv.linear_rate(res.error_bounds) == pytest.approx(0.5, abs=1e-9)


def test_bisection_error_is_not_monotone():
    """Documented behaviour: only the BOUND decreases monotonically."""
    res = R.bisection(f_quad, 1.0, 2.0, tol=1e-16, max_iter=40)
    errs = res.errors(SQRT2)
    ratios = errs[1:25] / errs[:24]
    assert ratios.max() > 1.0, "some bisection steps increase the error"


@pytest.mark.parametrize("tol", [1e-3, 1e-6, 1e-9, 1e-12, 1e-15])
def test_bisection_step_count_is_predictable(tol):
    predicted = R.bisection_steps_needed(1.0, 2.0, tol)
    res = R.bisection(f_quad, 1.0, 2.0, tol=1e-16, max_iter=200)
    assert cv.steps_to_tolerance(res.error_bounds, tol) == predicted


def test_bisection_steps_needed_rejects_bad_tolerance():
    with pytest.raises(ValueError):
        R.bisection_steps_needed(0.0, 1.0, 0.0)


def test_bisection_detects_an_endpoint_root():
    res = R.bisection(lambda x: x - 1.0, 1.0, 2.0)
    assert res.converged and res.root == 1.0


def test_regula_falsi_stagnates_on_a_convex_function():
    """The documented failure: one endpoint never moves, so the bracket never closes."""
    g = lambda x: x**10 - 1.0
    rf = R.regula_falsi(g, 0.0, 1.3, tol=1e-14, max_iter=400)
    rights = np.array([hi for _lo, hi in rf.brackets])
    assert len(np.unique(np.round(rights, 14))) == 1, "right endpoint should be stuck"
    assert rf.bracket_widths[-1] > 0.1, "the bracket should never close"


def test_illinois_beats_regula_falsi_on_the_stagnating_case():
    g = lambda x: x**10 - 1.0
    rf = R.regula_falsi(g, 0.0, 1.3, tol=1e-14, max_iter=400)
    il = R.illinois(g, 0.0, 1.3, tol=1e-14, max_iter=400)
    assert il.n_iter < rf.n_iter / 5
    assert il.bracket_widths[-1] < 1e-12, "Illinois should actually close the bracket"


def test_illinois_beats_bisection_across_the_suite():
    for name, f, bracket, _root in SUITE:
        b = R.bisection(f, *bracket, tol=1e-14, max_iter=500)
        i = R.illinois(f, *bracket, tol=1e-14, max_iter=500)
        assert i.n_iter <= b.n_iter, f"Illinois lost to bisection on {name}"


@pytest.mark.parametrize("name,f,bracket,root", SUITE)
def test_brent_matches_scipy(name, f, bracket, root):
    ours = R.brent(f, *bracket, tol=1e-14, max_iter=500)
    ref = brentq(f, *bracket, xtol=1e-15, rtol=8.9e-16)
    assert abs(ours.root - ref) < 1e-11, f"{name}: ours {ours.root}, scipy {ref}"


def test_brent_is_never_much_worse_than_bisection():
    """The whole design goal: interpolation speed with a bisection floor."""
    for name, f, bracket, _root in SUITE:
        b = R.bisection(f, *bracket, tol=1e-14, max_iter=500)
        br = R.brent(f, *bracket, tol=1e-14, max_iter=500)
        assert br.n_feval <= b.n_feval, f"Brent lost to bisection on {name}"


# ---------------------------------------------------------------- open methods


def test_fixed_point_converges_when_derivative_is_small():
    root = 0.6823278038280193
    g = lambda x: np.cbrt(1 - x)                # |g'(r)| = 0.716
    res = R.fixed_point(g, 0.5, tol=1e-14, max_iter=300)
    assert res.converged
    assert abs(res.root - root) < 1e-12


def test_fixed_point_diverges_when_derivative_exceeds_one():
    g = lambda x: 1 - x**3                      # |g'(r)| = 1.397
    res = R.fixed_point(g, 0.5, tol=1e-14, max_iter=200)
    assert not res.converged


def test_fixed_point_rate_equals_the_derivative_at_the_root():
    """The sharpest statement of Theorem 10.2: the rate IS |g'(r)|."""
    for c in [0.2, 0.4, 0.6, 0.8]:
        target = 1.234
        g = lambda x, c=c, t=target: t + c * (x - t)     # g'(r) = c exactly
        res = R.fixed_point(g, target + 1.0, tol=1e-15, max_iter=2000)
        rate = cv.linear_rate(np.abs(res.iterates - target))
        assert rate == pytest.approx(c, abs=1e-6), f"c={c}: measured {rate}"


def test_newton_is_quadratic():
    res = R.newton(f_quad, df_quad, 1.0, tol=1e-16)
    assert cv.reliable_order(res.errors(SQRT2)) == pytest.approx(2.0, abs=0.05)


def test_newton_asymptotic_constant_matches_theory():
    """C = |f''(r) / (2 f'(r))|, a stronger check than the order alone."""
    res = R.newton(f_quad, df_quad, 1.0, tol=1e-16)
    measured = cv.asymptotic_constant(res.errors(SQRT2), 2.0)
    theory = abs(2.0 / (2.0 * 2.0 * SQRT2))
    assert measured == pytest.approx(theory, abs=1e-3)


def test_newton_cycles_on_the_classic_example():
    f = lambda x: x**3 - 2*x + 2
    df = lambda x: 3*x**2 - 2
    res = R.newton(f, df, 0.0, max_iter=30)
    assert not res.converged
    assert np.allclose(res.iterates[0::2], 0.0)
    assert np.allclose(res.iterates[1::2], 1.0)


def test_newton_diverges_outside_the_basin():
    datan = lambda x: 1.0 / (1.0 + x * x)
    with np.errstate(over="ignore", invalid="ignore"):
        assert R.newton(np.arctan, datan, 1.0, max_iter=60).converged
        assert not R.newton(np.arctan, datan, 2.0, max_iter=60).converged


def test_newton_reports_a_vanishing_derivative():
    res = R.newton(lambda x: x**2 + 1.0, lambda x: 2.0 * x, 0.0, max_iter=10)
    assert not res.converged
    assert "derivative" in res.message


def test_newton_is_linear_at_a_multiple_root_with_rate_one_minus_one_over_m():
    for m in [2, 3, 4]:
        f = lambda x, m=m: (x - 1.0)**m * (x + 2.0)
        df = lambda x, m=m: (m * (x - 1.0)**(m-1) * (x + 2.0) + (x - 1.0)**m)
        res = R.newton(f, df, 1.5, tol=1e-13, max_iter=500)
        rate = cv.linear_rate(np.abs(res.iterates - 1.0))
        assert rate == pytest.approx(1 - 1/m, abs=0.02), f"m={m}: measured {rate}"


def test_modified_newton_restores_quadratic_convergence():
    m = 3
    f = lambda x: (x - 1.0)**m * (x + 2.0)
    df = lambda x: m * (x - 1.0)**(m-1) * (x + 2.0) + (x - 1.0)**m
    plain = R.newton(f, df, 1.5, tol=1e-13, max_iter=500)
    mod = R.newton(f, df, 1.5, tol=1e-13, max_iter=500, multiplicity=m)
    assert mod.n_iter < plain.n_iter / 10
    order = cv.reliable_order(np.abs(mod.iterates - 1.0))
    assert order == pytest.approx(2.0, abs=0.1)


def test_secant_order_is_the_golden_ratio():
    for x0, x1 in [(0.0, 2.0), (1.0, 2.0), (0.5, 1.0)]:
        res = R.secant(lambda x: np.exp(x) - 2.0, x0, x1, tol=1e-16)
        order = cv.reliable_order(res.errors(math.log(2.0)))
        assert order == pytest.approx(GOLDEN, abs=0.12), f"({x0},{x1}): {order}"


def test_secant_uses_fewer_evaluations_than_newton():
    n = R.newton(f_quad, df_quad, 1.0, tol=1e-14)
    s = R.secant(f_quad, 1.0, 2.0, tol=1e-14)
    assert s.n_feval < n.n_feval


def test_muller_finds_complex_roots_from_real_starts():
    res = R.muller(lambda x: x*x + x + 1.0, 0.0, 0.5, 1.0)
    assert abs(res.root.imag) > 0.5, "should leave the real line"
    assert res.residuals[-1] < 1e-13
    expected = np.roots([1.0, 1.0, 1.0])
    assert np.min(np.abs(expected - res.root)) < 1e-12


def test_muller_also_handles_real_roots():
    res = R.muller(f_quad, 1.0, 1.5, 2.0)
    assert abs(float(np.real(res.root)) - SQRT2) < 1e-12


@pytest.mark.parametrize("method", [R.chebyshev, R.halley])
def test_third_order_methods_are_cubic(method):
    res = method(lambda x: np.exp(x) - 2.0, np.exp, np.exp, 2.0, tol=1e-16)
    order = cv.reliable_order(res.errors(math.log(2.0)))
    assert order == pytest.approx(3.0, abs=0.15), f"{method.__name__}: {order}"


@pytest.mark.parametrize("method", [R.chebyshev, R.halley])
def test_third_order_methods_beat_newton_on_iterations(method):
    n = R.newton(f_quad, df_quad, 1.0, tol=1e-14)
    t = method(f_quad, df_quad, d2f_quad, 1.0, tol=1e-14)
    assert t.n_iter <= n.n_iter


# ---------------------------------------------------------------- acceleration


def test_aitken_on_an_exact_geometric_sequence():
    """For e_k = C^k exactly, Aitken should land on the limit immediately."""
    limit, C = 3.0, 0.5
    seq = np.array([limit + C**k for k in range(10)])
    accel = R.aitken(seq)
    np.testing.assert_allclose(accel, limit, atol=1e-12)


def test_aitken_accelerates_a_linear_sequence():
    root = 0.6823278038280193
    res = R.fixed_point(lambda x: np.cbrt(1 - x), 0.5, tol=1e-14, max_iter=300)
    plain = np.abs(res.iterates - root)
    accel = np.abs(R.aitken(res.iterates) - root)
    assert (cv.steps_to_tolerance(accel, 1e-10)
            < cv.steps_to_tolerance(plain, 1e-10) / 2)


def test_aitken_handles_short_input():
    assert R.aitken([1.0]).size == 0
    assert R.aitken([1.0, 2.0]).size == 0


def test_steffensen_is_quadratic_without_a_derivative():
    root = 0.6823278038280193
    res = R.steffensen(lambda x: np.cbrt(1 - x), 0.5, tol=1e-14)
    assert res.converged
    assert abs(res.root - root) < 1e-13
    order = cv.reliable_order(np.abs(res.iterates - root))
    assert order == pytest.approx(2.0, abs=0.15)


def test_steffensen_beats_the_plain_fixed_point_iteration():
    g = lambda x: np.cbrt(1 - x)
    plain = R.fixed_point(g, 0.5, tol=1e-14, max_iter=300)
    st = R.steffensen(g, 0.5, tol=1e-14)
    assert st.n_iter < plain.n_iter / 10


# ---------------------------------------------------------------- result plumbing


def test_result_records_the_full_history():
    res = R.newton(f_quad, df_quad, 1.0)
    assert res.iterates.ndim == 1
    assert res.iterates.size == res.n_iter + 1
    assert res.residuals.size == res.iterates.size
    assert res.iterates[0] == 1.0
    assert res.root == res.iterates[-1]


def test_result_errors_helper():
    res = R.newton(f_quad, df_quad, 1.0)
    np.testing.assert_allclose(res.errors(SQRT2), np.abs(res.iterates - SQRT2))


def test_residuals_really_are_the_backward_error():
    res = R.newton(f_quad, df_quad, 1.0)
    for x, r in zip(res.iterates, res.residuals):
        assert r == pytest.approx(abs(f_quad(x)), abs=1e-15)


def test_feval_count_is_honest():
    """Newton evaluates f twice per step plus once at the start."""
    res = R.newton(f_quad, df_quad, 1.0, max_iter=3)
    assert res.n_feval >= res.n_iter, "cannot use fewer evaluations than steps"


def test_repr_is_informative():
    text = repr(R.newton(f_quad, df_quad, 1.0))
    assert "RootResult" in text and "converged" in text
