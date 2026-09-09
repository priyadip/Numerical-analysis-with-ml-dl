"""Tests for `nalib.finitediff`.

The operator identities are checked on real data at several lengths, because the whole point of
the algebra is that it holds as an identity and can therefore be rearranged freely, which is what
lesson 49 does with it.

The error propagation is tested both for the pattern it produces and for the precondition the
technique needs, since the technique returns a confident wrong answer when that precondition
fails and a test that only used well behaved data would never notice.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nalib import finitediff as fd


LENGTHS = [2, 3, 5, 8, 13, 21]
ORDERS = [0, 1, 2, 3, 5]


# ---------------------------------------------------------------- the operators


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("k", ORDERS)
def test_each_difference_shortens_the_sequence_by_its_order(n, k, rng):
    y = rng.standard_normal(n)
    assert fd.forward(y, k).size == max(n - k, 0)
    assert fd.backward(y, k).size == max(n - k, 0)


@pytest.mark.parametrize("n", LENGTHS)
def test_the_zeroth_difference_is_the_sequence(n, rng):
    y = rng.standard_normal(n)
    assert np.array_equal(fd.forward(y, 0), y)


@pytest.mark.parametrize("n", LENGTHS)
def test_the_mean_operator_averages_neighbours(n, rng):
    y = rng.standard_normal(n)
    m = fd.mean(y)
    assert m.size == max(n - 1, 0)
    if m.size:
        assert float(np.max(np.abs(m - 0.5 * (y[1:] + y[:-1])))) < 1e-15


@pytest.mark.parametrize("k", [-1, -3])
def test_a_negative_order_is_rejected(k):
    with pytest.raises(ValueError):
        fd.forward([1.0, 2.0, 3.0], k)


def test_an_empty_sequence_is_rejected():
    with pytest.raises(ValueError):
        fd.forward([])


# ---------------------------------------------------------------- the algebra


@pytest.mark.parametrize("n", [4, 7, 12, 20])
def test_the_operators_satisfy_their_interrelations(n, rng):
    """``Δ = E - 1``, ``∇ = 1 - E^-1``, ``Δ = ∇E``. Each is checked where both sides exist."""
    out = fd.interrelations(rng.standard_normal(n))
    for name, gap in out.items():
        assert gap < 1e-12, f"{name} failed by {gap:.2e}"


@pytest.mark.parametrize("j,k", [(1, 1), (1, 2), (2, 3), (3, 2)])
@pytest.mark.parametrize("n", [8, 14, 20])
def test_the_operators_commute(j, k, n, rng):
    """They are all series in ``E``, so they commute, which is the licence to rearrange."""
    out = fd.commutes(rng.standard_normal(n), j=j, k=k)
    assert out["length"] > 0
    assert out["relative_gap"] < 1e-10


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("n", [8, 15])
def test_the_operators_are_linear(k, n, rng):
    out = fd.is_linear(rng.standard_normal(n), rng.standard_normal(n), k=k)
    assert out["relative_gap"] < 1e-10


@pytest.mark.parametrize("h", [0.5, 0.1, 0.01])
def test_the_shift_operator_is_the_exponential_of_the_differential_one(h):
    """``E = e^(hD)`` is Taylor's theorem as an operator identity, and it is where every finite
    difference formula for a derivative comes from."""
    out = fd.operator_identity_report(np.exp, 0.0, h, n_terms=14)
    assert out["relative_gap"] < 1e-12


@pytest.mark.parametrize("n_terms", [1, 2, 3, 4])
def test_more_terms_of_the_log_series_give_a_better_derivative(n_terms):
    """``hD = log(1 + Δ)``, truncated at ``n`` terms, is an ``O(h^n)`` approximation."""
    h = 0.01
    x = h * np.arange(12)
    y = np.exp(x)
    got = fd.derivative_from_differences(y, h, n_terms=n_terms)
    err = float(np.max(np.abs(got - np.exp(x[:got.size]))))
    assert err < 5.0 * h ** n_terms


def test_the_log_series_rejects_a_sequence_too_short_for_the_order():
    with pytest.raises(ValueError):
        fd.derivative_from_differences([1.0, 2.0, 3.0], 0.1, n_terms=5)


# ---------------------------------------------------------------- tables


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("h", [1.0, 0.5, 0.25])
def test_a_polynomial_has_constant_differences_at_its_own_degree(degree, h, rng):
    """The classical test for whether tabulated data is polynomial, and the order at which the
    column goes constant is the degree."""
    coeffs = rng.standard_normal(degree + 1)
    coeffs[-1] = abs(coeffs[-1]) + 0.5           # keep the leading term away from zero
    out = fd.table_of_a_polynomial(coeffs, x0=0.0, h=h, n_points=degree + 6)
    assert out["constant_at_order"] == degree
    col = fd.forward(out["values"], degree)
    assert float(np.max(np.abs(col - out["d_th_difference"]))) < 1e-6 * max(
        abs(out["d_th_difference"]), 1.0)


@pytest.mark.parametrize("n", [4, 8, 15])
def test_the_unreachable_corner_of_the_table_is_nan(n, rng):
    T = fd.difference_table(rng.standard_normal(n))
    assert np.isnan(T[-1, -1])
    assert np.all(np.isfinite(T[:, 0]))


# ---------------------------------------------------------------- error propagation


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_a_single_fault_spreads_in_the_binomial_pattern(k):
    """``+1, -2, +1`` at order 2, ``+1, -3, +3, -1`` at order 3, and so on."""
    y = np.polyval([1.0, -2.0, 0.5], np.arange(16.0))     # a quadratic, so Δ^k is 0 for k > 2
    out = fd.error_propagation(y, index=8, size=1.0, order=k)
    nonzero = out.pattern[np.abs(out.pattern) > 1e-9]
    expected = np.asarray([((-1) ** j) * math.comb(k, j) for j in range(k + 1)], dtype=float)
    assert nonzero.size == k + 1
    assert float(np.max(np.abs(np.abs(nonzero) - np.abs(expected)))) < 1e-9


@pytest.mark.parametrize("k,amp", [(2, 2), (3, 3), (4, 6), (5, 10), (6, 20)])
def test_the_amplification_is_the_central_binomial_coefficient_not_two_to_the_k(k, amp):
    """**Worth being exact about.** Measured amplification is ``C(k, floor(k/2))``: 2, 3, 6, 10,
    20 for ``k = 2..6``, against ``2^k`` of 4, 8, 16, 32, 64. The two differ by roughly
    ``sqrt(k)``, and at ``k = 6`` that is 20 rather than 64."""
    y = np.polyval([1.0, -2.0, 0.5], np.arange(16.0))
    out = fd.error_propagation(y, index=8, size=1.0, order=k)
    assert abs(out.amplification - amp) < 1e-9
    assert out.amplification < 2.0 ** k


@pytest.mark.parametrize("index", [3, 5, 9])
@pytest.mark.parametrize("size", [0.02, -0.05, 0.1])
def test_a_fault_is_located_when_the_data_is_smooth_enough(index, size):
    """The precondition is that the data's own ``Δ^k`` is below the fault. On ``sin(0.2k)`` at
    order 4 that ratio is 0.08, so the technique works."""
    y = np.sin(0.2 * np.arange(14))
    bad = y.copy()
    bad[index] += size
    out = fd.locate_a_bad_value(bad, order=4)
    assert out["reliable"]
    assert out["estimated_index"] == index
    assert abs(out["estimated_size"] - size) < 0.2 * abs(size)


def test_the_locator_reports_when_it_cannot_be_trusted():
    """**The failure case, and the guard for it.** Measured, planting 0.02 at index 5:

    ====================  =================  ===============  =======
    data                  its own max |Δ⁴|   ratio to fault   found?
    ====================  =================  ===============  =======
    ``exp(0.3 k)``        2.23e-1            11.1             no, 11
    ``exp(0.05 k)``       1.08e-5            0.001            yes, 5
    ``sin(0.2 k)``        1.59e-3            0.08             yes, 5
    ====================  =================  ===============  =======

    On ``exp(0.3 k)`` the technique returns index 11 for a fault at index 5. The binomial pattern
    check catches it: ``reliable`` is False for exactly the cases where the answer is wrong, in
    all eight combinations measured.
    """
    steep = np.exp(0.3 * np.arange(14))
    bad = steep.copy()
    bad[5] += 0.02
    out = fd.locate_a_bad_value(bad, order=4)
    assert not out["reliable"]
    assert out["fits_binomial_pattern"] > 0.25

    gentle = np.exp(0.05 * np.arange(14))
    ok = gentle.copy()
    ok[5] += 0.02
    good = fd.locate_a_bad_value(ok, order=4)
    assert good["reliable"] and good["estimated_index"] == 5


def test_the_error_propagation_rejects_a_bad_index():
    with pytest.raises(ValueError):
        fd.error_propagation(np.arange(6.0), index=99, size=1.0)


def test_the_error_propagation_rejects_order_zero():
    with pytest.raises(ValueError):
        fd.error_propagation(np.arange(6.0), index=2, size=1.0, order=0)