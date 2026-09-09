"""Tests for nalib.errors.

Condition numbers are checked against analytic values wherever one is known, and the
governing inequality (forward <= condition x backward) is checked on real cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nalib import errors as err


# ---------------------------------------------------------------- basic measures


def test_absolute_and_relative_error_scalars():
    assert err.absolute_error(2.0, 2.5) == 0.5
    assert err.relative_error(2.0, 2.5) == 0.25
    assert err.percentage_error(2.0, 2.5) == 25.0


def test_errors_are_zero_for_an_exact_answer():
    assert err.absolute_error(3.14, 3.14) == 0.0
    assert err.relative_error(3.14, 3.14) == 0.0
    assert err.significant_digits(3.14, 3.14) == 16.0


def test_relative_error_at_zero():
    assert err.relative_error(0.0, 0.0) == 0.0
    assert math.isinf(err.relative_error(0.0, 1e-12))


def test_errors_on_arrays():
    exact = np.array([1.0, 2.0, 3.0])
    approx = np.array([1.0, 2.0, 3.3])
    assert err.absolute_error(exact, approx) == pytest.approx(0.3)
    assert err.relative_error(exact, approx) == pytest.approx(0.1)


def test_significant_digits_known_values():
    assert err.significant_digits(1.0, 1.0 + 1e-9) == pytest.approx(9.0, abs=0.01)
    assert err.significant_digits(1.0, 1.0 + 1e-3) == pytest.approx(3.0, abs=0.01)
    assert err.significant_digits(1.0, 2.0) == 0.0        # relative error 1, no digits
    assert err.significant_digits(1.0, 100.0) == 0.0      # worse than useless


def test_forward_error_is_absolute_error():
    assert err.forward_error(5.0, 4.5) == err.absolute_error(5.0, 4.5)


# ---------------------------------------------------------------- backward error


def test_backward_error_of_an_exact_root_is_zero():
    f = lambda x: x * x - 4.0
    assert err.backward_error_root(f, 2.0) == 0.0


def test_backward_error_of_a_root_is_the_residual():
    f = lambda x: x * x - 2.0
    x_hat = 1.4142
    assert err.backward_error_root(f, x_hat) == abs(f(x_hat))


def test_backward_error_linear_system_zero_for_exact_solution(rng):
    for _ in range(200):
        n = int(rng.integers(2, 8))
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        x = rng.standard_normal(n)
        b = A @ x
        # the computed solution from a stable solver should have tiny backward error
        x_hat = np.linalg.solve(A, b)
        be = err.backward_error_linear_system(A, b, x_hat)
        assert be < 100 * np.finfo(float).eps


def test_backward_error_grows_with_a_deliberately_wrong_solution(rng):
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([3.0, 4.0])
    x_true = np.linalg.solve(A, b)
    good = err.backward_error_linear_system(A, b, x_true)
    bad = err.backward_error_linear_system(A, b, x_true + 1.0)
    assert bad > 1000 * max(good, np.finfo(float).eps)


def test_residual_matches_definition(rng):
    A = rng.standard_normal((5, 5))
    x = rng.standard_normal(5)
    b = rng.standard_normal(5)
    np.testing.assert_allclose(err.residual(A, b, x), b - A @ x)


# ---------------------------------------------------------------- conditioning


@pytest.mark.parametrize(
    "f, x, expected",
    [
        (np.sqrt, 4.0, 0.5),            # kappa = |x f'/f| = 1/2 for sqrt, all x
        (np.sqrt, 100.0, 0.5),
        (np.exp, 1.0, 1.0),             # kappa = |x| for exp
        (np.exp, 20.0, 20.0),
        (np.exp, 0.5, 0.5),
        (lambda t: t**3, 2.0, 3.0),     # kappa = n for x^n
        (lambda t: t**5, 7.0, 5.0),
        (lambda t: 1.0 / t, 3.0, 1.0),  # kappa = 1 for reciprocal
    ],
)
def test_condition_number_scalar_against_analytic(f, x, expected):
    assert err.condition_number_scalar(f, x) == pytest.approx(expected, rel=1e-4)


def test_condition_number_of_log_blows_up_near_one():
    """kappa = 1/|ln x| for the logarithm, so it explodes as x approaches 1."""
    for delta in [1e-2, 1e-4, 1e-6]:
        x = 1.0 + delta
        expected = 1.0 / abs(np.log(x))
        assert err.condition_number_scalar(np.log, x) == pytest.approx(expected, rel=1e-3)


def test_condition_number_matches_measured_amplification():
    """The condition number must be the factor by which input error is multiplied.

    The perturbation has to be large enough that the OUTPUT change is resolvable. A single
    ulp of the input is not enough when kappa < 1: for f = sqrt at x = 4, a one-ulp step in
    x produces a half-ulp step in f, which rounds straight back and measures an
    amplification of exactly zero. A relative perturbation of 1e-8 sits comfortably above
    the roundoff floor and well below the range where the second derivative matters.
    """
    delta = 1e-8
    for f, x in [(np.sqrt, 4.0), (np.exp, 2.0), (np.log, 5.0), (np.sin, 1.0),
                 (np.sqrt, 100.0), (np.exp, 0.5)]:
        kappa = err.condition_number_scalar(f, x)
        x_pert = x * (1 + delta)
        rel_in = abs(x_pert - x) / abs(x)
        rel_out = abs(float(f(x_pert)) - float(f(x))) / abs(float(f(x)))
        measured = rel_out / rel_in
        assert measured == pytest.approx(kappa, rel=0.01), (f, x, kappa, measured)


def test_condition_number_root_analytic():
    """Sensitivity of a root is 1/|f'(r)|."""
    f = lambda x: x * x - 2.0
    r = math.sqrt(2.0)
    expected = 1.0 / abs(2.0 * r)
    assert err.condition_number_root(f, r) == pytest.approx(expected, rel=1e-4)


def test_condition_number_root_infinite_at_a_multiple_root():
    """A double root has f'(r) = 0, so it cannot be located accurately."""
    f = lambda x: (x - 1.0) ** 2
    assert err.condition_number_root(f, 1.0) > 1e6


def test_error_magnification():
    assert err.error_magnification(1.0, 1.0 + 1e-6, 1e-12) == pytest.approx(1e6, rel=1e-3)
    assert err.error_magnification(1.0, 1.0, 0.0) == 0.0
    assert math.isinf(err.error_magnification(1.0, 2.0, 0.0))


# ---------------------------------------------------------------- propagation


def test_propagate_absolute_one_variable():
    """|df| = |f'(x)| |dx| for a single variable."""
    f = lambda x: x**3
    x, dx = 2.0, 0.01
    assert err.propagate_absolute(f, [x], [dx]) == pytest.approx(3 * x**2 * dx, rel=1e-5)


def test_propagate_absolute_cylinder_volume():
    """V = pi r^2 h, so dV = |2 pi r h| dr + |pi r^2| dh."""
    volume = lambda r, h: np.pi * r**2 * h
    r, h, dr, dh = 5.0, 12.0, 0.01, 0.02
    expected = abs(2 * np.pi * r * h) * dr + abs(np.pi * r**2) * dh
    assert err.propagate_absolute(volume, [r, h], [dr, dh]) == pytest.approx(
        expected, rel=1e-5
    )


def test_propagate_relative_for_a_product():
    """Relative errors add under multiplication."""
    prod = lambda a, b: a * b
    got = err.propagate_relative(prod, [2.0, 3.0], [1e-3, 2e-3])
    assert got == pytest.approx(3e-3, rel=1e-4)


def test_propagate_relative_for_a_power():
    """f = x^n multiplies relative error by n."""
    for n in [2, 3, 5]:
        f = lambda x, n=n: x**n
        got = err.propagate_relative(f, [3.0], [1e-6])
        assert got == pytest.approx(n * 1e-6, rel=1e-4)


def test_propagate_shape_mismatch_raises():
    with pytest.raises(ValueError):
        err.propagate_absolute(lambda a, b: a + b, [1.0, 2.0], [0.1])


def test_first_order_estimate_is_close_to_a_direct_perturbation():
    """For small perturbations the linear estimate should be accurate."""
    volume = lambda r, h: np.pi * r**2 * h
    r, h, dr, dh = 5.0, 12.0, 0.001, 0.002
    estimate = err.propagate_absolute(volume, [r, h], [dr, dh])
    worst = max(
        abs(volume(r + sr * dr, h + sh * dh) - volume(r, h))
        for sr in (-1, 1)
        for sh in (-1, 1)
    )
    assert abs(worst - estimate) / worst < 0.01


# ---------------------------------------------------------------- arithmetic rules


def test_relative_error_rules_for_product_and_quotient():
    assert err.rel_error_product(1e-6, 2e-6) == 3e-6
    assert err.rel_error_quotient(1e-6, 2e-6) == 3e-6


def test_rel_error_sum_bound_holds(rng):
    """The derived bound must actually bound the realised error.

    The realised error is computed with exact rational arithmetic. Doing it in floating
    point would measure the rounding of the test itself rather than the inequality, and
    near cancellation that rounding can exceed the bound.
    """
    from fractions import Fraction

    for _ in range(3000):
        a = float(rng.standard_normal())
        b = float(rng.standard_normal())
        if a + b == 0.0:
            continue
        ea, eb = 1e-8, 2e-8
        bound = err.rel_error_sum(a, b, ea, eb)

        A, B = Fraction(a), Fraction(b)
        Ea, Eb = Fraction(ea), Fraction(eb)
        perturbed = A * (1 + Ea) + B * (1 + Eb)
        realised = abs(perturbed - (A + B)) / abs(A + B)

        assert float(realised) <= bound * (1 + 1e-12), (a, b, float(realised), bound)


def test_rel_error_sum_explodes_under_cancellation():
    """Subtracting nearly equal values multiplies the input relative error enormously."""
    input_error = 1e-16
    for gap in [1e-10, 1e-12, 1e-15]:
        a, b = 1.0, -1.0 + gap
        bound = err.rel_error_sum(a, b, input_error, input_error)
        amplification = bound / input_error
        # the amplification is about 2/gap
        assert amplification == pytest.approx(2.0 / gap, rel=0.05)
        assert amplification > 1e9


def test_cancellation_factor():
    assert err.cancellation_factor(1.0, 0.0) == 1.0
    assert err.cancellation_factor(1.0, 0.9) == pytest.approx(19.0, rel=1e-9)
    assert math.isinf(err.cancellation_factor(1.0, 1.0))


def test_cancellation_factor_predicts_lost_digits():
    """Agreeing in k digits should give a factor of roughly 2 x 10^k."""
    for k in [2, 4, 8, 12]:
        a, b = 1.0, 1.0 - 10.0**-k
        factor = err.cancellation_factor(a, b)
        assert abs(np.log10(factor) - (k + np.log10(2.0))) < 0.01


# ---------------------------------------------------------------- governing inequality


def test_forward_bounded_by_condition_times_backward():
    """forward error is about kappa x backward error, at a SIMPLE root.

    The inequality is a first order statement, so it needs f'(r) != 0. At a multiple root
    the relation is genuinely nonlinear (for a root of multiplicity m the forward error
    goes like the backward error to the power 1/m) and the linear bound does not apply.
    That case is tested separately below.
    """
    f = lambda x: x * x - 2.0
    root = math.sqrt(2.0)
    kappa = err.condition_number_root(f, root)          # = 1 / |f'(r)| = 1/(2 sqrt 2)

    for offset in [1e-4, 1e-6, 1e-8, -1e-5]:
        x_hat = root + offset
        forward = abs(x_hat - root)
        backward = err.backward_error_root(f, x_hat)
        predicted = kappa * backward
        assert predicted == pytest.approx(forward, rel=1e-3), (offset, forward, predicted)


def test_multiple_root_breaks_the_linear_relation():
    """At a triple root the forward error goes like the cube root of the backward error."""
    f = lambda x: (x - 2.0) ** 3
    for offset in [1e-2, 1e-3, 1e-4]:
        x_hat = 2.0 + offset
        forward = abs(x_hat - 2.0)
        backward = err.backward_error_root(f, x_hat)
        # backward = offset^3, so forward = backward^(1/3)
        assert forward == pytest.approx(backward ** (1.0 / 3.0), rel=1e-6)
        # and the forward error is enormously larger than the backward error
        assert forward / backward > 1e3


# ---------------------------------------------------------------- diagnose_root


def test_diagnose_root_reports_the_noise_floor_for_a_good_root():
    d = err.diagnose_root(lambda x: x * x - 2.0, np.sqrt(2.0), lambda x: 2 * x)
    assert d["at_noise_floor"]
    assert d["residual"] < 1e-15
    assert d["condition"] == pytest.approx(1 / (2 * np.sqrt(2)), rel=1e-9)


def test_diagnose_root_catches_a_small_residual_hiding_a_large_error():
    """The whole reason the function exists: residual 1e-15, forward error 1e-9."""
    true_error = 1e-9
    d = err.diagnose_root(lambda x: 1e-6 * (x - 1.0), 1.0 + true_error, lambda x: 1e-6)
    assert d["residual"] < 1e-14, "the residual really is tiny"
    assert not d["at_noise_floor"]
    assert d["forward_bound"] == pytest.approx(true_error, rel=1e-6), (
        "the bound must predict the actual error, not merely exceed it"
    )


def test_diagnose_root_bound_is_never_smaller_than_the_true_error():
    """A bound that undershoots is worse than no bound. Check on a range of slopes."""
    rng = np.random.default_rng(42)
    for slope in [1e-8, 1e-4, 1.0, 1e3]:
        for offset in [1e-12, 1e-9, 1e-6]:
            r_true = 1.0
            d = err.diagnose_root(lambda x, s=slope: s * (x - r_true),
                                r_true + offset, lambda x, s=slope: s)
            assert d["forward_bound"] >= offset * (1 - 1e-9), (
                f"bound {d['forward_bound']:.3e} below true error {offset:.3e}"
            )


def test_diagnose_root_flags_a_multiple_root_separately():
    d = err.diagnose_root(lambda x: (x - 1.0) ** 2, 1.0, lambda x: 2 * (x - 1.0))
    assert np.isinf(d["condition"])
    assert "multiple root" in d["verdict"]
    assert not d["at_noise_floor"], "an infinite bound is not the noise floor"


def test_diagnose_root_works_without_an_analytic_derivative():
    with_df = err.diagnose_root(lambda x: x * x - 2.0, np.sqrt(2.0), lambda x: 2 * x)
    without = err.diagnose_root(lambda x: x * x - 2.0, np.sqrt(2.0))
    assert without["condition"] == pytest.approx(with_df["condition"], rel=1e-6)


def test_diagnose_root_agrees_with_condition_number_root():
    """The two entry points must not disagree about the same quantity."""
    f = lambda x: np.exp(x) - 2.0
    r = np.log(2.0)
    d = err.diagnose_root(f, r)
    assert d["condition"] == pytest.approx(err.condition_number_root(f, r), rel=1e-9)
