"""Tests for nalib.cost.

The operation counter is the tool that turns every complexity claim in this course from an
assertion into a measurement, so it needs to count correctly on cases where the answer is
known by hand.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nalib import cost


# ---------------------------------------------------------------- the counter


def test_counter_starts_empty():
    c = cost.OpCounter()
    assert c.adds == c.muls == c.divs == c.others == c.total == 0


def test_counter_counts_each_operation_once():
    c = cost.OpCounter()
    x = c.wrap(3.0)
    _ = x + x
    assert (c.adds, c.muls, c.divs) == (1, 0, 0)
    _ = x * x
    assert (c.adds, c.muls, c.divs) == (1, 1, 0)
    _ = x / x
    assert (c.adds, c.muls, c.divs) == (1, 1, 1)
    _ = x - x
    assert c.adds == 2
    assert c.total == 4          # 2 adds + 1 mul + 1 div


def test_counter_handles_reflected_operations():
    """Plain float on the left must still be counted, through the r-methods."""
    c = cost.OpCounter()
    x = c.wrap(2.0)
    assert float(1.0 + x) == 3.0
    assert float(1.0 - x) == -1.0
    assert float(3.0 * x) == 6.0
    assert float(8.0 / x) == 4.0
    assert (c.adds, c.muls, c.divs) == (2, 1, 1)


def test_counter_values_are_correct():
    c = cost.OpCounter()
    x = c.wrap(3.0)
    assert float((x + 1.0) * 2.0 - 4.0) == 4.0
    assert float(x**2) == 9.0
    assert float(-x) == -3.0


def test_power_counts_as_other():
    c = cost.OpCounter()
    x = c.wrap(2.0)
    _ = x**3
    assert c.others == 1
    assert c.total == 1


def test_negation_is_free():
    c = cost.OpCounter()
    x = c.wrap(2.0)
    _ = -x
    assert c.total == 0


def test_counter_reset():
    c = cost.OpCounter()
    x = c.wrap(1.0)
    _ = x + x + x
    assert c.total == 2
    c.reset()
    assert c.total == 0


def test_counter_summary_is_readable():
    c = cost.OpCounter()
    x = c.wrap(1.0)
    _ = x * x + x
    text = c.summary()
    assert "adds=1" in text and "muls=1" in text and "total=2" in text


def test_count_ops_returns_value_and_counter():
    value, counter = cost.count_ops(lambda t: t * t + t, 3.0)
    assert value == 12.0
    assert (counter.adds, counter.muls) == (1, 1)


def test_count_ops_on_a_known_loop():
    """Summing n wrapped values takes exactly n additions."""
    for n in [1, 5, 20]:
        c = cost.OpCounter()
        x = c.wrap(1.0)
        total = 0.0
        for _ in range(n):
            total = total + x
        assert c.adds == n


# ---------------------------------------------------------------- known algorithms


def test_dot_product_operation_count():
    """A dot product of length n is n multiplications and n-1 additions."""
    for n in [1, 3, 10, 50]:
        c = cost.OpCounter()
        xs = [c.wrap(float(i + 1)) for i in range(n)]
        ys = [float(i + 2) for i in range(n)]
        total = xs[0] * ys[0]
        for i in range(1, n):
            total = total + xs[i] * ys[i]
        assert c.muls == n
        assert c.adds == n - 1


def test_back_substitution_operation_count():
    """Back substitution on an n by n triangular system costs about n^2 flops.

    Precisely: n divisions, n(n-1)/2 multiplications and n(n-1)/2 subtractions.
    """
    for n in [2, 3, 5, 10]:
        c = cost.OpCounter()
        U = [[c.wrap(2.0 if i == j else 1.0) for j in range(n)] for i in range(n)]
        b = [c.wrap(1.0) for _ in range(n)]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = b[i]
            for j in range(i + 1, n):
                s = s - U[i][j] * x[j]
            x[i] = s / U[i][i]
        assert c.divs == n
        assert c.muls == n * (n - 1) // 2
        assert c.adds == n * (n - 1) // 2


# ---------------------------------------------------------------- timing


def test_time_once_returns_result_and_positive_time():
    out, dt = cost.time_once(lambda: sum(range(100_000)))
    assert out == sum(range(100_000))
    assert dt > 0.0


def test_time_best_of_is_at_most_a_single_run():
    def work():
        return np.linalg.norm(np.ones(100_000))

    best = cost.time_best_of(work, repeats=5)
    _out, single = cost.time_once(work)
    assert best > 0.0
    assert best <= single * 5, "the minimum should not exceed a typical single run by much"


def test_time_scaling_shapes():
    sizes = [100, 200, 400]
    ns, ts = cost.time_scaling(lambda n: np.ones(n).sum(), sizes, repeats=2)
    assert list(ns) == sizes
    assert ts.shape == (3,)
    assert (ts > 0).all()


# ---------------------------------------------------------------- exponent fitting


def test_fit_exponent_exact_power_law():
    sizes = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    for p in [1.0, 2.0, 3.0, 2.5, 0.5]:
        times = 1e-6 * sizes**p
        assert cost.fit_exponent(sizes, times) == pytest.approx(p, rel=1e-9)


def test_fit_exponent_ignores_bad_points():
    sizes = np.array([1.0, 2.0, 4.0, 8.0])
    times = np.array([1.0, 4.0, 16.0, 0.0])
    assert cost.fit_exponent(sizes, times) == pytest.approx(2.0, rel=1e-9)


def test_fit_exponent_insufficient_data():
    assert math.isnan(cost.fit_exponent([1.0], [1.0]))
    assert math.isnan(cost.fit_exponent([], []))


def test_fit_exponent_on_real_matvec_timings():
    """A matrix vector product is memory bound, so its exponent should come out near 2."""
    sizes = np.array([500, 1000, 1500, 2000])

    def matvec(n):
        r = np.random.default_rng(0)
        A = r.standard_normal((n, n))
        v = r.standard_normal(n)
        return A @ v

    ns, ts = cost.time_scaling(matvec, sizes, repeats=3)
    p = cost.fit_exponent(ns, ts)
    assert 1.5 < p < 2.5, f"expected roughly quadratic, measured {p}"


def test_scaling_table_is_readable():
    sizes = np.array([100, 200, 400])
    times = np.array([1e-3, 4e-3, 16e-3])
    text = cost.scaling_table(sizes, times, expected_exponent=2)
    assert "ratio to previous" in text
    assert "4.00" in text
    assert len(text.splitlines()) == 5      # header, rule, three rows
