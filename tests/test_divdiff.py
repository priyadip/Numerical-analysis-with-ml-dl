"""Tests for `nalib.divdiff`.

The recursion is checked against the closed form, which is an independent computation, rather
than against itself. The four properties are each tested across a range of orders, because the
interesting fact about divided differences is that the properties hold mathematically at every
order and hold **numerically** only up to about order 10.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nalib import divdiff as dd


ORDERS = [1, 2, 3, 5, 8]
SPANS = [(0.0, 1.0), (-1.0, 1.0), (-3.0, 7.0), (10.0, 10.5)]


def equally_spaced(lo, hi, n):
    return np.linspace(lo, hi, n)


# ---------------------------------------------------------------- the recursion


@pytest.mark.parametrize("k", ORDERS)
@pytest.mark.parametrize("lo,hi", SPANS)
def test_the_recursion_matches_the_closed_form(k, lo, hi, rng):
    """Two independent routes to the same number: the recursion, and
    ``sum_i y_i / prod_{j != i} (x_i - x_j)``."""
    x = equally_spaced(lo, hi, k + 1)
    y = rng.standard_normal(k + 1)
    a = float(dd.coefficients(x, y)[-1])
    b = dd.from_definition(x, y)
    scale = max(abs(a), abs(b), 1e-300)
    assert abs(a - b) / scale < 1e-9


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_the_table_matches_the_naive_recursive_definition(k, rng):
    """`recursive` is exponential and written only for checking. It must agree with the table."""
    x = np.sort(rng.uniform(0.0, 1.0, k + 1))
    if np.unique(x).size != x.size:
        pytest.skip("duplicate nodes drawn")
    y = rng.standard_normal(k + 1)
    a = float(dd.coefficients(x, y)[-1])
    b = dd.recursive(x, y)
    scale = max(abs(a), abs(b), 1e-300)
    assert abs(a - b) / scale < 1e-9


@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_the_zeroth_differences_are_the_data(n, rng):
    x = np.linspace(0.0, 1.0, n)
    y = rng.standard_normal(n)
    assert np.array_equal(dd.table(x, y)[:, 0], y)


def test_the_unused_corner_of_the_table_is_nan():
    """A caller reading an entry that was never computed gets a nan, not a plausible number."""
    x = np.linspace(0.0, 1.0, 4)
    T = dd.table(x, np.exp(x))
    assert np.isnan(T[3, 1]) and np.isnan(T[2, 2]) and np.isnan(T[1, 3])
    assert np.all(np.isfinite(np.diag(np.fliplr(T))))


# ---------------------------------------------------------------- the four properties


@pytest.mark.parametrize("k", [1, 2, 3, 4, 6, 8])
def test_a_divided_difference_is_symmetric_in_its_arguments(k):
    """Mathematically exact at every order. Numerically it holds to about order 10 and then
    fails completely, which is measured in the next test rather than hidden here."""
    x = np.linspace(0.0, 1.0, k + 1)
    y = np.exp(x)
    out = dd.symmetry_report(x, y, n_orders=15, rng=np.random.default_rng(3))
    assert out["relative_spread"] < 1e-6


def test_symmetry_fails_numerically_at_high_order():
    """**The central practical fact about divided differences.** Measured on equally spaced
    nodes, ``f = exp``, spread across 21 node orderings:

    ====  ================
    k     relative spread
    ====  ================
    2     0.00e+00
    4     1.63e-13
    7     8.15e-09
    10    9.01e-04
    13    1.91e+00
    16    1.93e+00
    ====  ================

    By order 13 the value depends on the ordering by 191 percent, so it carries no information
    at all. The nodes are equally spaced, so this is not node clustering: it is the recursion's
    own cancellation, and it is the reason every practical scheme in the rest of Part 7 keeps
    the order low and uses many pieces instead.
    """
    spreads = {}
    for k in (4, 7, 10, 13):
        x = np.linspace(0.0, 1.0, k + 1)
        out = dd.symmetry_report(x, np.exp(x), n_orders=20, rng=np.random.default_rng(7))
        spreads[k] = out["relative_spread"]
    assert spreads[4] < 1e-10
    assert spreads[7] > spreads[4]
    assert spreads[10] > spreads[7]
    assert spreads[13] > 1e-2


@pytest.mark.parametrize("n", [2, 3, 6, 9, 12, 15])
def test_it_is_the_leading_coefficient_of_the_interpolating_polynomial(n):
    """Checked against the power form, which reaches the same number a different way.

    The comparison is scaled by the table rather than by the answer, because the answer is
    exactly zero for an even function at symmetric nodes and odd degree.
    """
    x = np.linspace(-1.0, 1.0, n)
    out = dd.is_leading_coefficient(x, np.cos(2.0 * x))
    assert out["relative_gap"] < 1e-10


@pytest.mark.parametrize("n", [2, 4, 7, 10])
def test_divided_differences_are_linear_in_the_data(n, rng):
    x = np.linspace(0.0, 1.0, n)
    out = dd.linearity_report(x, rng.standard_normal(n), rng.standard_normal(n))
    assert out["relative_gap"] < 1e-9


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_a_divided_difference_lies_in_the_range_of_the_scaled_derivative(k):
    """``f[x_0, ..., x_k] = f^(k)(xi)/k!`` for some ``xi`` inside the interval, so the value must
    lie between the smallest and largest of ``f^(k)/k!`` there."""
    out = dd.derivative_connection(np.exp, np.exp, 0.0, 1.0, k, n_trials=30,
                                   rng=np.random.default_rng(11))
    assert out["trials"] > 0
    assert out["inside"] == out["trials"]


@pytest.mark.parametrize("k", [0, 1, 2, 3, 5])
def test_the_confluent_limit_is_the_taylor_coefficient(k):
    """All nodes coalescing gives ``f^(k)(x)/k!``, which is the bridge to Hermite in lesson 50."""
    derivs = np.ones(k + 1)                       # every derivative of exp at 0 is 1
    got = dd.confluent(0.0, derivs)
    assert abs(got - 1.0 / float(math.factorial(k))) < 1e-14


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_crowding_the_nodes_approaches_the_confluent_limit_at_first(k):
    """While truncation dominates, halving the spacing halves the error."""
    target = 1.0 / float(math.factorial(k))
    coarse = dd.conditioning_report(np.exp, k=k, separations=[1e-1], centre=0.0).errors[0]
    finer = dd.conditioning_report(np.exp, k=k, separations=[1e-2], centre=0.0).errors[0]
    assert abs(finer - target) < abs(coarse - target)


@pytest.mark.parametrize("k,h_best", [(2, 1e-5), (3, 1e-4), (4, 1e-3), (6, 1e-2)])
def test_crowding_the_nodes_too_far_destroys_the_divided_difference(k, h_best):
    """**A U-curve, and the bottom of it moves with the order.**

    Relative error of the ``k``-th divided difference of ``exp`` at 0, nodes spaced ``h``:

    =======  ========  ========  ========  ========  ========
    h        k=1       k=2       k=3       k=4       k=6
    =======  ========  ========  ========  ========  ========
    1e-2     5.02e-3   1.01e-2   1.51e-2   2.02e-2   2.83e-2
    1e-3     5.00e-4   1.00e-3   1.50e-3   1.87e-3   4.43e+2
    1e-4     5.00e-5   1.00e-4   8.89e-5   3.44e+0   2.89e+9
    1e-6     5.00e-7   8.89e-5   2.23e+2   4.44e+8   6.66e+20
    =======  ========  ========  ========  ========  ========

    Truncation falls like ``h`` and rounding rises like ``eps / h^k``, so the best spacing is
    near ``eps^(1/(k+1))``: 1.8e-3 for ``k = 4`` and 5.6e-3 for ``k = 6``, which is where the
    table bottoms out. Past it the error grows by **twenty orders of magnitude**. This is the
    optimal step size argument that Part 9 will meet again in numerical differentiation, and it
    is the same phenomenon as the interpolation degree curve in `test_interp`.
    """
    target = 1.0 / float(math.factorial(k))
    rel = lambda h: abs(dd.conditioning_report(np.exp, k=k, separations=[h],
                                               centre=0.0).errors[0] - target) / target
    predicted = np.finfo(float).eps ** (1.0 / (k + 1))
    assert 0.02 * predicted < h_best < 50.0 * predicted
    assert rel(h_best) < rel(h_best * 1e-2), "past the optimum the error must grow"
    assert rel(1e-6) > rel(h_best)


# ---------------------------------------------------------------- input handling


@pytest.mark.parametrize("bad", [[1.0, 1.0], [0.0, 2.0, 0.0]])
def test_repeated_nodes_are_rejected_with_a_pointer_to_the_confluent_form(bad):
    with pytest.raises(ValueError, match="confluent"):
        dd.table(bad, np.arange(len(bad), dtype=float))


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        dd.table([0.0, 1.0, 2.0], [1.0, 2.0])


def test_an_empty_input_is_rejected():
    with pytest.raises(ValueError):
        dd.table([], [])


# ---------------------------------------------------------------- the cost argument


@pytest.mark.parametrize("n", [4, 16, 64, 256])
def test_adding_a_point_is_cheaper_than_rebuilding(n):
    """The reason an adaptive scheme carries divided differences rather than Lagrange."""
    c = dd.advantage_over_lagrange(n)
    assert c["newton_add_one_point"] < c["lagrange_rebuild"]
    assert c["lagrange_rebuild"] / c["newton_add_one_point"] == n
    assert c["newton_evaluate"] < c["lagrange_evaluate"]