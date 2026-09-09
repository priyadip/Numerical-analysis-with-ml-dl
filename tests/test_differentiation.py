"""Tests for nalib.differentiation.

Four groups. The stencil weights are checked against the exact answer on polynomials, which is
the definition, and the two independent derivations are checked against each other. The step
size behaviour is checked by measuring the fitted order rather than by trusting the table. The
accelerations, Richardson and the complex step, are checked against the accuracy they claim and
against the cases where the complex step genuinely fails. The matrices are checked against the
stencils they are built from and against the eigenvalues of the operator they discretise.

Every test sweeps stencils, step sizes and node counts. Nothing is fixed to one example.
"""
import math

import numpy as np
import pytest

from nalib import differentiation as df

STENCILS = sorted(df.STENCILS)
NODE_COUNTS = [4, 7, 11, 16, 25]


def polynomial(coefficients):
    """A polynomial and its exact derivatives, built from whatever coefficients are given."""
    c = np.asarray(coefficients, dtype=float)

    def value(x, order=0):
        d = c.copy()
        for _ in range(order):
            if d.size <= 1:
                return np.zeros_like(np.asarray(x, dtype=float))
            d = d[1:] * np.arange(1, d.size)
        return sum(d[k] * np.asarray(x, dtype=float) ** k for k in range(d.size))

    return value


# --------------------------------------------------------------------------- the weights


@pytest.mark.parametrize("name", STENCILS)
def test_every_stencil_reproduces_its_own_derivative_order(name):
    offsets, order = df.STENCILS[name]
    c = df.from_taylor(offsets, order)
    o = np.asarray(offsets, dtype=float)
    # sum c_j o_j^k must be m! at k = m and zero at every other k below the accuracy limit
    for k in range(order + 1):
        got = float(np.sum(c * o ** k))
        want = float(math.factorial(order)) if k == order else 0.0
        assert got == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("name", STENCILS)
def test_the_two_derivations_agree(name):
    offsets, order = df.STENCILS[name]
    out = df.two_derivations_agree(offsets, order)
    assert out["agree"]
    assert out["relative_gap"] < 1e-9


@pytest.mark.parametrize("name", STENCILS)
def test_weights_sum_to_zero_for_every_derivative(name):
    # a derivative of any order annihilates constants, so the weights must cancel
    offsets, order = df.STENCILS[name]
    c = df.from_taylor(offsets, order)
    assert float(np.sum(c)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", STENCILS)
def test_the_stated_accuracy_order_is_at_least_the_node_count_bound(name):
    offsets, order = df.STENCILS[name]
    p = df.accuracy_order(offsets, order)
    o = np.asarray(offsets, dtype=float)
    assert p >= 1
    minimum = o.size - order
    assert p >= minimum
    symmetric = bool(np.allclose(np.sort(o), -np.sort(o)[::-1]))
    if p > minimum:
        # only a symmetric stencil can beat the node count bound, and only by one
        assert symmetric
        assert p == minimum + 1


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_a_stencil_is_exact_on_polynomials_up_to_its_degree(n, order):
    if n <= order:
        pytest.skip("need more nodes than the derivative order")
    offsets = np.arange(n) - n // 2
    rng = np.random.default_rng(42)
    f = polynomial(rng.standard_normal(n))
    x = np.linspace(-1.0, 1.0, 7)
    got = df.differentiate(lambda t: f(t, 0), x, 0.25, offsets, order)
    want = f(x, order)
    scale = max(1.0, float(np.max(np.abs(want))))
    assert np.max(np.abs(got - want)) == pytest.approx(0.0, abs=1e-7 * scale)


# --------------------------------------------------------------------------- step size


@pytest.mark.parametrize("name", ["forward first", "central first", "central second",
                                  "central first, 4th order"])
def test_the_fitted_order_matches_the_predicted_one(name):
    offsets, order = df.STENCILS[name]
    predicted = df.accuracy_order(offsets, order)
    exact = math.cos(1.0) if order == 1 else -math.sin(1.0)
    out = df.error_against_step(np.sin, exact, 1.0, offsets, order)
    assert out["accuracy_order"] == pytest.approx(predicted, abs=0.35)


@pytest.mark.parametrize("name", ["forward first", "central first", "central second"])
def test_shrinking_the_step_stops_helping(name):
    offsets, order = df.STENCILS[name]
    exact = math.cos(1.0) if order == 1 else -math.sin(1.0)
    out = df.error_against_step(np.sin, exact, 1.0, offsets, order,
                                steps=np.logspace(-1, -14, 40))
    # the best step is interior, not the smallest: that is the whole point
    assert out["best_step"] > float(np.min(out["steps"])) * 10.0
    smallest = float(out["errors"][int(np.argmin(out["steps"]))])
    assert smallest > out["best_error"] * 10.0


@pytest.mark.parametrize("p", [1, 2, 4, 6])
@pytest.mark.parametrize("m", [1, 2])
def test_the_optimal_step_formula_is_self_consistent(p, m):
    out = df.optimal_step(p, m)
    assert out["step_exponent"] == pytest.approx(1.0 / (p + m))
    assert out["error_exponent"] == pytest.approx(p / (p + m))
    # At the optimum the two error sources are NOT equal. Setting d/dh of C h^p + eps h^-m to
    # zero gives p * truncation = m * roundoff, so they are equal only when p = m. Asserting
    # equality passes for the four cases with p = m and fails for the rest, which is how the
    # difference showed up here.
    assert p * out["truncation"] == pytest.approx(m * out["roundoff"], rel=1e-6)


def test_higher_order_keeps_more_digits():
    out = df.digits_lost()
    kept = np.asarray(out["digits_kept"], dtype=float)
    assert np.all(np.diff(kept) > 0.0)


# --------------------------------------------------------------------------- acceleration


@pytest.mark.parametrize("h", [1.0, 0.5, 0.25])
@pytest.mark.parametrize("central", [True, False])
def test_richardson_gains_the_order_it_claims(h, central):
    out = df.richardson(np.sin, 1.0, h, levels=4, central=central)
    orders = np.asarray(out["orders"], dtype=float)
    assert np.all(np.isfinite(orders))
    first = 2.0 if central else 1.0
    assert orders[0] == pytest.approx(first, abs=0.6)


def test_richardson_beats_the_plain_stencil_at_the_same_step():
    exact = math.cos(1.0)
    out = df.richardson_report(np.sin, exact, 1.0, 0.5, levels=5)
    for kind in ("forward", "central"):
        errors = np.asarray(out[kind]["errors"], dtype=float)
        assert errors[-1] < errors[0]


@pytest.mark.parametrize("x", [0.3, 1.0, 2.5])
def test_the_complex_step_is_exact_where_the_central_difference_is_not(x):
    out = df.complex_step_report(np.sin, math.cos(x), x)
    assert out["complex_step_best"] <= out["central_best"]
    assert out["complex_step_best"] < 1e-14


@pytest.mark.parametrize("h", [1e-8, 1e-100, 1e-200])
def test_the_complex_step_does_not_depend_on_the_step(h):
    # no subtraction means no cancellation, so any small h gives the same answer
    got = float(np.ravel(df.complex_step(np.sin, 1.0, h))[0])
    assert got == pytest.approx(math.cos(1.0), abs=1e-12)


@pytest.mark.parametrize("name", ["abs", "real", "conjugate"])
def test_the_complex_step_fails_on_nowhere_holomorphic_functions(name):
    out = df.complex_step_fails_on(name)
    assert out["silently_wrong"]
    assert out["complex_step_error"] > out["central_error"]


@pytest.mark.parametrize("name", ["sqrt of a square", "maximum"])
def test_those_two_cases_work_away_from_their_kink(name):
    # they look non-holomorphic but agree with a holomorphic function on a neighbourhood
    out = df.complex_step_fails_on(name)
    assert not out["silently_wrong"]
    assert out["complex_step_error"] <= out["central_error"]


# --------------------------------------------------------------------------- matrices


@pytest.mark.parametrize("n", NODE_COUNTS)
@pytest.mark.parametrize("order", [1, 2])
def test_the_differentiation_matrix_is_exact_on_low_polynomials(n, order):
    nodes = np.linspace(-1.0, 1.0, n)
    D = df.differentiation_matrix(nodes, order)
    assert D.shape == (n, n)
    rng = np.random.default_rng(42)
    f = polynomial(rng.standard_normal(n))
    values = f(nodes, 0)
    got = D @ values
    want = f(nodes, order)
    # The matrix is exact in exact arithmetic. In floating point the product carries roundoff
    # of size eps * max|D| * max|f|, and max|D| reaches 1.6e7 by n = 25 on these nodes, so a
    # fixed absolute tolerance fails for reasons that have nothing to do with the weights.
    # Chebyshev nodes do not rescue it: the growth is in the derivative operator itself.
    roundoff = (np.finfo(float).eps * float(np.max(np.abs(D)))
                * max(1.0, float(np.max(np.abs(values)))) * nodes.size)
    scale = max(1.0, float(np.max(np.abs(want))))
    assert np.max(np.abs(got - want)) < max(1e-9 * scale, 50.0 * roundoff)


@pytest.mark.parametrize("n", NODE_COUNTS)
@pytest.mark.parametrize("order", [1, 2])
def test_the_differentiation_matrix_annihilates_constants(n, order):
    nodes = np.linspace(0.0, 3.0, n)
    D = df.differentiation_matrix(nodes, order)
    # measured relative to the size of the entries doing the cancelling, which is the only
    # scale a row sum can be judged against
    assert np.max(np.abs(D @ np.ones(n))) < 1e-13 * float(np.max(np.abs(D))) * n


@pytest.mark.parametrize("n", [5, 10, 20, 40])
def test_the_second_derivative_matrix_has_the_known_eigenvalues(n):
    h = 1.0 / (n + 1)
    A = df.second_derivative_matrix(n, h, "dirichlet")
    got = np.sort(np.linalg.eigvalsh(A))
    k = np.arange(1, n + 1)
    want = np.sort(-4.0 / h ** 2 * np.sin(k * math.pi * h / 2.0) ** 2)
    scale = float(np.max(np.abs(want)))
    assert np.max(np.abs(got - want)) == pytest.approx(0.0, abs=1e-9 * scale)


@pytest.mark.parametrize("n", [5, 10, 20])
def test_the_second_derivative_matrix_is_symmetric_and_negative_definite(n):
    A = df.second_derivative_matrix(n, 1.0 / (n + 1), "dirichlet")
    assert np.max(np.abs(A - A.T)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.max(np.linalg.eigvalsh(A))) < 0.0


def test_spectral_differentiation_beats_finite_differences_before_roundoff():
    out = df.spectral_against_finite_difference(np.sin, np.cos)
    spectral = np.asarray(out["spectral_error"], dtype=float)
    finite = np.asarray(out["finite_difference_error"], dtype=float)
    assert float(np.min(spectral)) < float(np.min(finite))
    # and it does bottom out: the minimum is not at the largest node count
    assert int(np.argmin(spectral)) < spectral.size - 1


@pytest.mark.parametrize("order", [1, 2])
def test_the_matrix_entries_grow_but_the_relative_accuracy_does_not_move(order):
    """The differentiation matrix is exact on paper and loud in floating point.

    max|D| grows roughly like n^(2*order), so by 33 nodes the second derivative matrix has
    entries near 1.3e8 and any answer it produces carries that much amplified roundoff. What
    stays constant is the accuracy relative to those entries: the row sums, which are exactly
    zero on paper, come out at a few rounding units of max|D| at every size.
    """
    sizes = [11, 17, 25, 33]
    biggest = []
    relative = []
    for n in sizes:
        D = df.differentiation_matrix(np.linspace(0.0, 3.0, n), order)
        biggest.append(float(np.max(np.abs(D))))
        relative.append(float(np.max(np.abs(D @ np.ones(n)))) / biggest[-1])
    assert np.all(np.diff(biggest) > 0.0)
    assert max(relative) < 1e-13


# ------------------------------------------------- the three routes and their conditioning


@pytest.mark.parametrize("name", STENCILS)
def test_all_three_derivations_agree_on_the_named_stencils(name):
    offsets, order = df.STENCILS[name]
    out = df.derivations_agree(offsets, order)
    assert out["agree"]
    assert out["taylor_gap"] < 1e-10
    assert out["interpolation_gap"] < 1e-10


@pytest.mark.parametrize("n", [3, 5, 7, 9, 11])
@pytest.mark.parametrize("order", [1, 2])
def test_fornberg_matches_the_vandermonde_solve_while_that_solve_is_still_good(n, order):
    offsets = np.arange(n) - n // 2
    a = df.from_taylor(offsets, order)
    b = df.fornberg(offsets, order)
    scale = float(np.max(np.abs(b)))
    assert np.max(np.abs(a - b)) < 1e-9 * scale


@pytest.mark.parametrize("order", [1, 2])
def test_the_vandermonde_solve_falls_apart_and_the_other_two_do_not(order):
    """The reason `differentiation_matrix` does not use `from_taylor`.

    This is forward against backward error, in the one place in this module where the two come
    apart completely. `np.linalg.solve` is backward stable, so the weights it returns satisfy
    the Taylor conditions to rounding no matter how wide the stencil gets. They are still wrong.
    The residual measures the wrong thing, and the only way to see it is to compare against a
    method that does not go through the ill conditioned matrix at all.

    Checking "the weights sum to zero" therefore proves nothing here: it holds to 5e-16 for the
    broken weights too.
    """
    worst_residual = 0.0
    worst_error = 0.0
    for n in (11, 21, 31, 41):
        half = n // 2
        offsets = (np.arange(n) - half) / float(half)
        stable = df.fornberg(offsets, order)
        lagrange = df.from_interpolation(offsets, order)
        taylor = df.from_taylor(offsets, order)
        scale = float(np.max(np.abs(stable)))
        # the two stable routes agree with each other, so they pin down the true answer
        assert np.max(np.abs(lagrange - stable)) < 1e-12 * scale
        worst_error = max(worst_error, float(np.max(np.abs(taylor - stable))) / scale)
        # the residual of the broken weights, on the very conditions they were built to satisfy
        for k in range(min(n, 6)):
            got = float(np.sum(taylor * offsets ** k / math.factorial(k)))
            want = 1.0 if k == order else 0.0
            worst_residual = max(worst_residual, abs(got - want))
    assert worst_residual < 1e-9
    assert worst_error > 1e-3


@pytest.mark.parametrize("n", [5, 9, 15, 21])
def test_fornberg_reproduces_the_derivative_of_a_polynomial_it_can_represent(n):
    offsets = np.arange(n) - n // 2
    rng = np.random.default_rng(42)
    f = polynomial(rng.standard_normal(n))
    for order in (1, 2):
        c = df.fornberg(offsets, order)
        h = 0.1
        got = float(np.sum(c * f(offsets * h, 0))) / h ** order
        want = float(f(0.0, order))
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)
