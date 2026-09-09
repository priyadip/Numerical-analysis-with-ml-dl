"""Tests for nalib.polynomials.

Every evaluation routine is checked against `numpy.polyval`, and the operation counts are
checked against a real measurement rather than against the derivation that produced them.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import cost, polynomials as poly


# ---------------------------------------------------------------- evaluation


def test_horner_known_value():
    # p(x) = 2x^3 - 6x^2 + 2x - 1 at x = 3  ->  54 - 54 + 6 - 1 = 5
    assert float(poly.horner([2, -6, 2, -1], 3.0)) == 5.0


def test_horner_constant_polynomial():
    assert float(poly.horner([7.0], 123.456)) == 7.0


def test_horner_rejects_empty():
    with pytest.raises(ValueError):
        poly.horner([], 1.0)


def test_horner_matches_polyval_bit_for_bit(rng):
    """Same algorithm, so the results must agree exactly, not just closely."""
    for _ in range(3000):
        deg = int(rng.integers(0, 15))
        c = rng.standard_normal(deg + 1)
        x = float(rng.standard_normal()) * 10 ** int(rng.integers(-3, 4))
        assert float(poly.horner(c, x)) == float(np.polyval(c, x))


def test_horner_works_elementwise_on_arrays(rng):
    c = rng.standard_normal(6)
    xs = rng.standard_normal(50)
    got = poly.horner(c, xs)
    assert got.shape == xs.shape
    np.testing.assert_array_equal(got, np.polyval(c, xs))


def test_scalar_variants_agree_with_horner(rng):
    for _ in range(500):
        deg = int(rng.integers(0, 10))
        c = list(rng.standard_normal(deg + 1))
        x = float(rng.standard_normal())
        assert poly.horner_scalar(c, x) == float(poly.horner(c, x))


def test_naive_and_horner_agree_on_well_behaved_input(rng):
    """Algebraically identical, so on benign data they should be very close."""
    for _ in range(500):
        deg = int(rng.integers(0, 8))
        c = list(rng.uniform(-2, 2, deg + 1))
        x = float(rng.uniform(-2, 2))
        h = poly.horner_scalar(c, x)
        n = poly.naive_scalar(c, x)
        assert abs(h - n) <= 1e-12 * max(1.0, abs(h))


def test_naive_eval_matches_polyval(rng):
    for _ in range(300):
        c = rng.standard_normal(int(rng.integers(1, 8)))
        x = float(rng.uniform(-2, 2))
        assert abs(float(poly.naive_eval(c, x)) - float(np.polyval(c, x))) < 1e-10


# ---------------------------------------------------------------- derivatives


def test_horner_with_derivative_known():
    # p(x) = x^2 - 2, p(1.5) = 0.25, p'(1.5) = 3.0
    p, dp = poly.horner_with_derivative([1, 0, -2], 1.5)
    assert p == 0.25
    assert dp == 3.0


def test_horner_with_derivative_returns_plain_floats():
    p, dp = poly.horner_with_derivative([1, 0, -2], 1.5)
    assert type(p) is float
    assert type(dp) is float


def test_horner_with_derivative_matches_numpy(rng):
    for _ in range(2000):
        deg = int(rng.integers(1, 12))
        c = rng.standard_normal(deg + 1)
        x = float(rng.standard_normal())
        p, dp = poly.horner_with_derivative(c, x)
        assert abs(p - float(np.polyval(c, x))) < 1e-9 * max(1.0, abs(p))
        assert abs(dp - float(np.polyval(np.polyder(c), x))) < 1e-9 * max(1.0, abs(dp))


def test_derivative_coeffs():
    np.testing.assert_allclose(poly.derivative_coeffs([2, -6, 2, -1]), [6, -12, 2])
    np.testing.assert_allclose(poly.derivative_coeffs([5.0]), [0.0])


def test_derivative_coeffs_matches_polyder(rng):
    for _ in range(500):
        c = rng.standard_normal(int(rng.integers(2, 12)))
        np.testing.assert_allclose(poly.derivative_coeffs(c), np.polyder(c))


def test_horner_all_derivatives_matches_repeated_polyder(rng):
    for _ in range(200):
        deg = int(rng.integers(1, 8))
        c = rng.standard_normal(deg + 1)
        x = float(rng.uniform(-2, 2))
        got = poly.horner_all_derivatives(c, x, deg)
        current = np.asarray(c, dtype=float)
        for k in range(deg + 1):
            expected = float(np.polyval(current, x))
            assert abs(got[k] - expected) < 1e-7 * max(1.0, abs(expected)), (k, deg)
            current = np.polyder(current) if current.size > 1 else np.array([0.0])


# ---------------------------------------------------------------- division


def test_synthetic_division_known():
    q, r = poly.synthetic_division([1, -6, 11, -6], 1.0)     # roots 1, 2, 3
    np.testing.assert_allclose(q, [1.0, -5.0, 6.0])
    assert r == 0.0


def test_remainder_theorem(rng):
    """The remainder of p / (x - r) must equal p(r). That is the whole trick."""
    for _ in range(2000):
        c = rng.standard_normal(int(rng.integers(1, 12)))
        r = float(rng.standard_normal())
        _q, remainder = poly.synthetic_division(c, r)
        assert abs(remainder - float(np.polyval(c, r))) < 1e-9 * max(1.0, abs(remainder))


def test_synthetic_division_reconstructs_the_polynomial(rng):
    """p(x) = (x - r) q(x) + remainder, exactly as an identity."""
    for _ in range(500):
        c = rng.standard_normal(int(rng.integers(2, 10)))
        r = float(rng.uniform(-2, 2))
        q, rem = poly.synthetic_division(c, r)
        rebuilt = np.polyadd(np.polymul([1.0, -r], q), [rem])
        # pad to the same length before comparing
        rebuilt = np.concatenate([np.zeros(len(c) - len(rebuilt)), rebuilt])
        np.testing.assert_allclose(rebuilt, c, atol=1e-10)


def test_synthetic_division_rejects_empty():
    with pytest.raises(ValueError):
        poly.synthetic_division([], 1.0)


def test_deflate_removes_a_root():
    c = poly.from_roots([1.0, 2.0, 3.0])
    reduced = poly.deflate(c, 1.0)
    remaining = np.sort(np.roots(reduced))
    np.testing.assert_allclose(remaining, [2.0, 3.0], atol=1e-10)


# ---------------------------------------------------------------- construction


def test_from_roots_matches_numpy_poly(rng):
    for _ in range(500):
        roots = rng.uniform(-3, 3, int(rng.integers(1, 8)))
        np.testing.assert_allclose(poly.from_roots(roots), np.poly(roots), atol=1e-10)


def test_from_roots_known():
    np.testing.assert_allclose(poly.from_roots([1, 2, 3]), [1, -6, 11, -6])
    np.testing.assert_allclose(poly.from_roots([2.0] * 5),
                               [1, -10, 40, -80, 80, -32])


def test_from_roots_then_evaluate_gives_zero_at_the_roots(rng):
    for _ in range(200):
        roots = rng.uniform(-2, 2, 4)
        c = poly.from_roots(roots)
        for r in roots:
            assert abs(float(poly.horner(c, r))) < 1e-9


# ---------------------------------------------------------------- operation counts


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10, 25, 50, 100, 200])
def test_measured_flop_counts_match_the_formulas(n):
    """The derivation is only a claim until the operations are actually counted."""
    c = [1.0] * (n + 1)

    _, counter_h = cost.count_ops(lambda t: poly.horner_scalar(c, t), 1.1)
    assert (counter_h.adds, counter_h.muls) == poly.flop_count_horner(n)

    _, counter_n = cost.count_ops(lambda t: poly.naive_scalar(c, t), 1.1)
    assert (counter_n.adds, counter_n.muls) == poly.flop_count_naive(n)


def test_horner_uses_exactly_n_of_each():
    for n in [1, 7, 33]:
        adds, muls = poly.flop_count_horner(n)
        assert adds == n and muls == n


def test_naive_multiplications_grow_quadratically():
    counts = [poly.flop_count_naive(n)[1] for n in [10, 20, 40]]
    # quadrupling n should roughly quadruple... no, quadratic means x4 for x2 in n
    assert counts[1] / counts[0] > 3.4
    assert counts[2] / counts[1] > 3.7


# ---------------------------------------------------------------- ill-conditioned case


def test_expanded_multiple_root_is_ill_conditioned():
    """The lesson 01 and 06 experiment, pinned down as a regression test.

    (x-1)^6 expanded and evaluated near x = 1 must produce values far larger than the
    true ones, and must produce negative values for an even power. If this ever stops
    happening, the lessons that rely on it need revisiting.
    """
    expanded = poly.from_roots([1.0] * 6)
    xs = np.linspace(0.998, 1.002, 401)
    exact = (xs - 1.0) ** 6

    by_horner = poly.horner(expanded, xs)
    peak = exact.max()

    assert (exact >= 0).all()
    assert by_horner.min() < 0, "expected impossible negative values"
    assert np.abs(by_horner - exact).max() > 10 * peak


def test_factored_form_is_accurate_where_expanded_form_is_not():
    xs = np.linspace(0.998, 1.002, 401)
    factored = (xs - 1.0) ** 6
    expanded = poly.horner(poly.from_roots([1.0] * 6), xs)
    # the factored form is exact to roundoff by construction, the expanded one is not
    assert np.abs(expanded - factored).max() > 1e3 * np.finfo(float).eps * factored.max()
