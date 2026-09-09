"""Tests for nalib.orthopoly.

The classical families have closed forms, so every structural claim can be checked against
something independent: Rodrigues formulas and explicit low-degree expressions for the values,
the defining integral for orthogonality, and published Gauss nodes for `golub_welsch`.

Every test sweeps degrees and families rather than fixing one.
"""
import math

import numpy as np
import pytest

from nalib import orthopoly as op

FINITE = ["legendre", "chebyshev_t", "chebyshev_u"]
ALL = ["legendre", "chebyshev_t", "chebyshev_u", "laguerre", "hermite"]
DEGREES = [0, 1, 2, 3, 5, 8]


# --------------------------------------------------------------------------- values


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6])
def test_legendre_matches_the_explicit_low_degree_formulas(n):
    x = np.linspace(-1.0, 1.0, 41)
    exact = {0: np.ones_like(x),
             1: x,
             2: 0.5 * (3 * x ** 2 - 1),
             3: 0.5 * (5 * x ** 3 - 3 * x),
             4: (35 * x ** 4 - 30 * x ** 2 + 3) / 8.0,
             5: (63 * x ** 5 - 70 * x ** 3 + 15 * x) / 8.0,
             6: (231 * x ** 6 - 315 * x ** 4 + 105 * x ** 2 - 5) / 16.0}[n]
    got = op.evaluate("legendre", n, x, monic=False)
    assert np.max(np.abs(got - exact)) < 1e-12


@pytest.mark.parametrize("n", list(range(9)))
def test_chebyshev_t_matches_the_cosine_definition(n):
    x = np.linspace(-1.0, 1.0, 61)
    got = op.evaluate("chebyshev_t", n, x, monic=False)
    assert np.max(np.abs(got - np.cos(n * np.arccos(x)))) < 1e-11


@pytest.mark.parametrize("n", list(range(1, 9)))
def test_chebyshev_u_matches_the_sine_definition(n):
    theta = np.linspace(0.05, math.pi - 0.05, 61)
    x = np.cos(theta)
    got = op.evaluate("chebyshev_u", n, x, monic=False)
    assert np.max(np.abs(got - np.sin((n + 1) * theta) / np.sin(theta))) < 1e-10


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_hermite_matches_the_physicists_formulas(n):
    x = np.linspace(-2.0, 2.0, 41)
    exact = {0: np.ones_like(x), 1: 2 * x, 2: 4 * x ** 2 - 2,
             3: 8 * x ** 3 - 12 * x, 4: 16 * x ** 4 - 48 * x ** 2 + 12}[n]
    assert np.max(np.abs(op.evaluate("hermite", n, x, monic=False) - exact)) < 1e-10


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_laguerre_matches_the_explicit_formulas(n):
    x = np.linspace(0.0, 6.0, 41)
    exact = {0: np.ones_like(x), 1: 1 - x, 2: (x ** 2 - 4 * x + 2) / 2.0,
             3: (-x ** 3 + 9 * x ** 2 - 18 * x + 6) / 6.0,
             4: (x ** 4 - 16 * x ** 3 + 72 * x ** 2 - 96 * x + 24) / 24.0}[n]
    assert np.max(np.abs(op.evaluate("laguerre", n, x, monic=False) - exact)) < 1e-9


@pytest.mark.parametrize("family", ALL)
@pytest.mark.parametrize("n", DEGREES)
def test_the_monic_member_has_leading_coefficient_one(family, n):
    """By the top divided difference, which equals the leading coefficient exactly.

    Comparing ``p_n(x) / x^n`` at a large ``x`` instead is only asymptotic, and for Laguerre,
    whose ``a_k = 2k + 1`` sum to 64 at degree 8, the next term is still 0.6 percent of the
    leading one at ``x = 1e4``. The divided difference has no such error.
    """
    nodes = np.linspace(-1.0, 1.0, n + 1) if family != "laguerre" else np.linspace(0.5, 3.0,
                                                                                   n + 1)
    values = op.evaluate(family, n, nodes)
    for k in range(1, n + 1):
        values = (values[1:] - values[:-1]) / (nodes[k:] - nodes[:-k])
    assert float(values[0]) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("family", ALL)
@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_the_recurrence_holds_pointwise(family, n):
    assert op.three_term_holds(family, n) < 1e-12


@pytest.mark.parametrize("family", ALL)
@pytest.mark.parametrize("n", [-1, -4])
def test_a_negative_degree_is_rejected(family, n):
    with pytest.raises(ValueError, match="degree must be non-negative"):
        op.evaluate(family, n, np.array([0.5]))


@pytest.mark.parametrize("bad", ["chebyshev", "jacobi", "", "LEGENDR"])
def test_an_unknown_family_is_rejected(bad):
    with pytest.raises(ValueError, match="unknown family"):
        op.recurrence_coefficients(bad, 3)


@pytest.mark.parametrize("n", [0, -2])
def test_asking_for_no_coefficients_is_rejected(n):
    with pytest.raises(ValueError, match="at least one coefficient"):
        op.recurrence_coefficients("legendre", n)


# --------------------------------------------------------------------------- derivatives


@pytest.mark.parametrize("family", ALL)
@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_the_derivative_matches_a_central_difference(family, n):
    lo, hi = (0.5, 4.0) if family == "laguerre" else (-0.8, 0.8)
    x = np.linspace(lo, hi, 17)
    h = 1e-5
    fd = (op.evaluate(family, n, x + h) - op.evaluate(family, n, x - h)) / (2 * h)
    got = op.evaluate_derivative(family, n, x)
    scale = max(float(np.max(np.abs(got))), 1.0)
    assert np.max(np.abs(got - fd)) < 1e-6 * scale


def test_the_derivative_of_a_constant_member_is_zero():
    x = np.linspace(-1.0, 1.0, 11)
    assert np.max(np.abs(op.evaluate_derivative("legendre", 0, x))) == 0.0


# --------------------------------------------------------------------------- orthogonality


@pytest.mark.parametrize("family", ALL)
@pytest.mark.parametrize("n", [2, 4, 6])
def test_the_family_is_orthogonal_in_its_own_weight(family, n):
    out = op.orthogonality_report(family, n)
    assert out["is_orthogonal"]
    assert out["relative"] < 1e-5


@pytest.mark.parametrize("family", ALL)
def test_the_quadrature_grid_carries_the_right_total_mass(family):
    exact = {"legendre": 2.0, "chebyshev_t": math.pi, "chebyshev_u": 0.5 * math.pi,
             "laguerre": 1.0, "hermite": math.sqrt(math.pi)}[family]
    _, weights = op.weighted_grid(family, 40001)
    assert float(np.sum(weights)) == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize("n_probe", [0, 1, 2, -5])
def test_a_grid_too_small_to_integrate_is_rejected(n_probe):
    with pytest.raises(ValueError, match="at least three grid points"):
        op.weighted_grid("legendre", n_probe)


@pytest.mark.parametrize("family", ALL)
@pytest.mark.parametrize("n", [2, 4, 6])
def test_stieltjes_recovers_the_closed_form_coefficients(family, n):
    out = op.stieltjes(n, family)
    assert out["alpha_gap"] < 1e-4
    assert out["beta_gap"] < 1e-4
    assert out["inner_products_used"] == 2 * n


@pytest.mark.parametrize("n", [1, 3, 6, 10])
def test_stieltjes_costs_o_n_while_gram_schmidt_costs_o_n_squared(n):
    linear = op.stieltjes(n, "legendre")["inner_products_used"]
    quadratic = op.gram_schmidt(n, "legendre")["inner_products_used"]
    assert linear == 2 * n
    assert quadratic == (n + 1) * (n + 2) // 2
    if n >= 6:
        assert quadratic > linear


@pytest.mark.parametrize("n", [0, -1])
def test_stieltjes_needs_at_least_one_step(n):
    with pytest.raises(ValueError, match="at least one step"):
        op.stieltjes(n, "legendre")


@pytest.mark.parametrize("family", FINITE)
@pytest.mark.parametrize("n", [2, 5, 8])
def test_gram_schmidt_produces_an_orthogonal_set(family, n):
    out = op.gram_schmidt(n, family)
    assert out["worst_off_diagonal"] < 1e-8


@pytest.mark.parametrize("n", [4, 8, 12, 16, 20, 24, 28])
def test_modified_gram_schmidt_keeps_its_orthogonality_and_classical_does_not(n):
    """The usual claim that Gram-Schmidt loses orthogonality is about the CLASSICAL form.

    Measured on the Legendre weight, the modified form stays at 2e-16 out to degree 28 while
    the classical form reaches 7.8e-7, a factor of 3.8e9. The cost argument against
    Gram-Schmidt, O(n^2) inner products against Stieltjes's O(n), holds for both.
    """
    modified = op.gram_schmidt(n, "legendre", n_probe=20001)["worst_off_diagonal"]
    classical = op.gram_schmidt(n, "legendre", n_probe=20001,
                                classical=True)["worst_off_diagonal"]
    assert modified < 1e-14
    assert classical >= modified - 1e-300
    if n >= 20:
        assert classical > 1000.0 * modified


def test_the_classical_loss_grows_with_the_degree_and_the_modified_one_does_not():
    classical = [op.gram_schmidt(n, "legendre", n_probe=20001,
                                 classical=True)["worst_off_diagonal"]
                 for n in (12, 16, 20, 24, 28)]
    modified = [op.gram_schmidt(n, "legendre", n_probe=20001)["worst_off_diagonal"]
                for n in (12, 16, 20, 24, 28)]
    assert all(b > a for a, b in zip(classical, classical[1:]))
    assert max(modified) < 1e-14


@pytest.mark.parametrize("family", ["laguerre", "hermite"])
def test_gram_schmidt_refuses_an_unbounded_interval(family):
    with pytest.raises(ValueError, match="finite interval"):
        op.gram_schmidt(4, family)


@pytest.mark.parametrize("n", [-1, -3])
def test_gram_schmidt_rejects_a_negative_degree(n):
    with pytest.raises(ValueError, match="degree must be non-negative"):
        op.gram_schmidt(n, "legendre")


# --------------------------------------------------------------------------- identities


@pytest.mark.parametrize("family", ALL)
@pytest.mark.parametrize("n", [1, 3, 5])
def test_christoffel_darboux_holds_away_from_the_diagonal(family, n):
    lo, hi = (0.5, 3.0) if family == "laguerre" else (-0.8, 0.8)
    x = np.linspace(lo, hi, 11)
    y = float(lo) + 0.37 * (float(hi) - float(lo))
    out = op.christoffel_darboux(family, n, x, y)
    assert out["relative_gap"] < 1e-8


@pytest.mark.parametrize("family", FINITE)
@pytest.mark.parametrize("n", [2, 4])
def test_the_confluent_branch_is_what_saves_the_identity_on_the_diagonal(family, n):
    x = np.linspace(-0.9, 0.9, 7)
    out = op.christoffel_darboux(family, n, x, x[4])
    assert bool(np.any(out["used_confluent_branch"]))
    assert out["relative_gap"] < 1e-8
    assert out["relative_gap_without_the_confluent_branch"] > out["relative_gap"]


# --------------------------------------------------------------------------- Golub-Welsch


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 12])
def test_gauss_legendre_nodes_match_numpy(n):
    got = op.golub_welsch("legendre", n)
    ref_x, ref_w = np.polynomial.legendre.leggauss(n)
    assert np.max(np.abs(got["nodes"] - ref_x)) < 1e-12
    assert np.max(np.abs(got["weights"] - ref_w)) < 1e-12


@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_gauss_chebyshev_nodes_match_the_closed_form(n):
    got = op.golub_welsch("chebyshev_t", n)
    exact = np.sort(np.cos((2 * np.arange(n) + 1) * math.pi / (2 * n)))
    assert np.max(np.abs(got["nodes"] - exact)) < 1e-12
    assert np.max(np.abs(got["weights"] - math.pi / n)) < 1e-12


@pytest.mark.parametrize("family", ALL)
@pytest.mark.parametrize("n", [1, 3, 6, 10])
def test_the_gauss_weights_sum_to_the_total_mass(family, n):
    got = op.golub_welsch(family, n)
    assert got["total_weight"] == pytest.approx(got["expected_total_weight"], rel=1e-12)
    assert np.all(got["weights"] > 0.0)


@pytest.mark.parametrize("family", ALL)
@pytest.mark.parametrize("n", [2, 4, 6])
def test_gauss_quadrature_is_exact_to_degree_two_n_minus_one(family, n):
    """The defining property, checked against `weighted_grid` rather than assumed."""
    got = op.golub_welsch(family, n)
    fine, dmu = op.weighted_grid(family, 200001, degree=2 * n)
    for k in range(2 * n):
        rule = float(np.sum(got["weights"] * got["nodes"] ** k))
        ref = float(np.sum(fine ** k * dmu))
        assert rule == pytest.approx(ref, rel=1e-4, abs=1e-9)


@pytest.mark.parametrize("family", ALL)
@pytest.mark.parametrize("n", [0, -2])
def test_asking_for_no_gauss_nodes_is_rejected(family, n):
    with pytest.raises(ValueError, match="at least one node"):
        op.golub_welsch(family, n)


@pytest.mark.parametrize("family", ALL)
@pytest.mark.parametrize("n", [2, 5, 9])
def test_the_gauss_nodes_are_the_roots_of_the_next_member(family, n):
    got = op.golub_welsch(family, n)
    values = op.evaluate(family, n, got["nodes"])
    scale = max(float(np.max(np.abs(op.evaluate(family, n, got["nodes"] + 0.3)))), 1.0)
    assert np.max(np.abs(values)) < 1e-8 * scale


# --------------------------------------------------------------------------- fitting


@pytest.mark.parametrize("family", FINITE)
@pytest.mark.parametrize("degree", [0, 1, 3, 6])
def test_a_polynomial_is_reproduced_at_or_above_its_degree(family, degree):
    rng = np.random.default_rng(degree + 5)
    c = rng.standard_normal(degree + 1)
    poly = lambda t: sum(c[k] * np.asarray(t) ** k for k in range(degree + 1))
    out = op.least_squares_by_orthogonality(poly, degree, family)
    assert out["weighted_l2_error"] < 1e-6 * max(float(np.max(np.abs(c))), 1.0)


@pytest.mark.parametrize("family", FINITE)
@pytest.mark.parametrize("degree", [2, 5, 8])
def test_truncating_the_series_leaves_the_coefficients_alone(family, degree):
    out = op.truncation_is_optimal(lambda t: np.exp(t), degree, family)
    assert out["coefficients_are_stable"]
    assert out["worst"] < 1e-8


@pytest.mark.parametrize("family", FINITE)
def test_raising_the_degree_never_increases_the_weighted_error(family):
    f = lambda t: np.exp(t)
    errs = [op.least_squares_by_orthogonality(f, d, family)["weighted_l2_error"]
            for d in range(0, 9, 2)]
    assert all(b <= a * (1.0 + 1e-5) + 1e-12 for a, b in zip(errs, errs[1:]))


@pytest.mark.parametrize("degree", [4, 8, 12, 16])
def test_the_orthonormal_basis_is_perfectly_conditioned_and_the_monomials_are_not(degree):
    out = op.conditioning_against_monomials(degree)
    assert out["orthonormal_condition"] == pytest.approx(1.0, abs=1e-2)
    assert out["monomial_condition"] > out["orthonormal_condition"]
    if degree >= 8:
        assert out["ratio"] > 1e3


def test_the_monomial_conditioning_grows_with_the_degree():
    vals = [op.conditioning_against_monomials(d)["monomial_condition"] for d in (2, 6, 10, 14)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("family", ["laguerre", "hermite"])
def test_the_conditioning_comparison_refuses_an_unbounded_interval(family):
    with pytest.raises(ValueError, match="finite interval"):
        op.conditioning_against_monomials(4, family)


@pytest.mark.parametrize("family", FINITE)
@pytest.mark.parametrize("degree", [1, 4])
def test_the_orthogonal_fit_agrees_with_a_direct_least_squares_solve(family, degree):
    f = lambda t: np.exp(t)
    out = op.least_squares_by_orthogonality(f, degree, family)
    x, dmu = op.weighted_grid(family, 20001, degree=degree)
    A = np.stack([op.evaluate(family, k, x) for k in range(degree + 1)], axis=1)
    root = np.sqrt(np.maximum(dmu, 0.0))
    direct, *_ = np.linalg.lstsq(A * root[:, None], f(x) * root, rcond=None)
    scale = max(float(np.max(np.abs(direct))), 1.0)
    assert np.max(np.abs(direct - out["coefficients"])) < 1e-6 * scale
