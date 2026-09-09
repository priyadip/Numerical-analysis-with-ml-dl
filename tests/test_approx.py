"""Tests for nalib.approx.

Every test sweeps sizes, intervals and functions rather than checking one case. The three
groups are: the norms agree with independent formulas, the Remez output satisfies Chebyshev's
equioscillation characterisation, and the Weierstrass rate really is the poor one the theory
predicts.
"""
import math

import numpy as np
import pytest

from nalib import approx as ap

INTERVALS = [(-1.0, 1.0), (0.0, 1.0), (-3.0, 2.0), (2.0, 2.5), (-0.25, 0.25)]
FUNCTIONS = [("exp", np.exp),
             ("sin", lambda t: np.sin(3.0 * t)),
             ("abs-shifted", lambda t: np.abs(t - 0.1)),
             ("runge", lambda t: 1.0 / (1.0 + 25.0 * t * t)),
             ("cosh", np.cosh)]


# --------------------------------------------------------------------------- norms


@pytest.mark.parametrize("lo,hi", INTERVALS)
def test_norm_of_a_constant_is_the_length_times_the_constant(lo, hi):
    for c in (1.0, -2.5, 0.125):
        assert ap.function_norm(lambda t: c * np.ones_like(t), lo, hi, 1) == pytest.approx(
            abs(c) * (hi - lo), rel=1e-10)
        assert ap.function_norm(lambda t: c * np.ones_like(t), lo, hi, 2) == pytest.approx(
            abs(c) * math.sqrt(hi - lo), rel=1e-10)
        assert ap.function_norm(lambda t: c * np.ones_like(t), lo, hi, np.inf) == pytest.approx(
            abs(c), rel=1e-12)


@pytest.mark.parametrize("lo,hi", INTERVALS)
def test_l2_norm_of_a_monomial_matches_the_exact_integral(lo, hi):
    for k in range(1, 6):
        got = ap.function_norm(lambda t: t ** k, lo, hi, 2, n_probe=200001)
        exact = math.sqrt(abs(hi ** (2 * k + 1) - lo ** (2 * k + 1)) / (2 * k + 1))
        assert got == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0, 8.0])
def test_higher_p_norms_of_a_bounded_function_approach_the_max(p):
    f = lambda t: np.exp(t)
    lo, hi = 0.0, 1.0
    v = ap.function_norm(f, lo, hi, p)
    assert v <= ap.function_norm(f, lo, hi, np.inf) * (hi - lo) ** (1.0 / p) + 1e-9


def test_the_p_norm_increases_toward_the_max_on_a_unit_interval():
    f = lambda t: np.exp(t)
    top = ap.function_norm(f, 0.0, 1.0, np.inf)
    ps = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
    vals = [ap.function_norm(f, 0.0, 1.0, p) for p in ps]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
    assert all(v <= top + 1e-9 for v in vals)
    gaps = [top - v for v in vals]
    assert all(b < a for a, b in zip(gaps, gaps[1:]))
    # the approach is slow: at p = 512 the gap is still above one percent
    assert 1e-2 < gaps[-1] / top < 5e-2


@pytest.mark.parametrize("p", [0.0, -1.0, -0.5])
def test_a_non_positive_p_is_rejected(p):
    with pytest.raises(ValueError, match="p must be positive"):
        ap.function_norm(np.exp, 0.0, 1.0, p)


@pytest.mark.parametrize("lo,hi", [(1.0, 1.0), (2.0, -1.0), (0.5, 0.5)])
def test_an_empty_or_reversed_interval_is_rejected(lo, hi):
    with pytest.raises(ValueError, match="need lo < hi"):
        ap.function_norm(np.exp, lo, hi, 2)


@pytest.mark.parametrize("name,f", FUNCTIONS)
def test_norm_comparison_of_a_function_with_itself_is_zero(name, f):
    out = ap.norm_comparison(f, f, -1.0, 1.0)
    assert out["l1"] == 0.0 and out["l2"] == 0.0 and out["linf"] == 0.0


@pytest.mark.parametrize("lo,hi", INTERVALS)
def test_inner_product_is_symmetric_and_bilinear(lo, hi):
    f = lambda t: np.exp(t)
    g = lambda t: np.sin(2.0 * t)
    a = ap.weighted_inner_product(f, g, lo, hi)
    b = ap.weighted_inner_product(g, f, lo, hi)
    assert a == pytest.approx(b, rel=1e-12, abs=1e-14)
    scaled = ap.weighted_inner_product(lambda t: 3.0 * f(t), g, lo, hi)
    assert scaled == pytest.approx(3.0 * a, rel=1e-10, abs=1e-13)


# --------------------------------------------------------------------------- best L2


@pytest.mark.parametrize("degree", [0, 1, 2, 5, 9])
@pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (0.0, 2.0), (-4.0, -1.0)])
def test_a_polynomial_is_reproduced_exactly_at_or_above_its_degree(degree, lo, hi):
    rng = np.random.default_rng(degree * 17 + abs(int(hi)))
    c = rng.standard_normal(degree + 1)
    poly = lambda t: sum(c[k] * t ** k for k in range(degree + 1))
    out = ap.best_l2_polynomial(poly, degree, lo, hi)
    scale = max(float(np.max(np.abs([poly(np.array([lo]))[0], poly(np.array([hi]))[0]]))), 1.0)
    assert out["linf_error"] < 1e-8 * scale


@pytest.mark.parametrize("name,f", FUNCTIONS)
def test_raising_the_degree_never_increases_the_l2_error(name, f):
    errs = [ap.best_l2_polynomial(f, d, -1.0, 1.0)["l2_error"] for d in range(0, 9, 2)]
    assert all(b <= a * (1.0 + 1e-6) + 1e-14 for a, b in zip(errs, errs[1:]))


@pytest.mark.parametrize("degree", [2, 6, 10, 14])
def test_the_monomial_gram_matrix_is_far_worse_conditioned_than_the_chebyshev_one(degree):
    out = ap.basis_conditioning(degree)
    assert out["chebyshev_condition"] < out["monomial_condition"]
    if degree >= 6:
        assert out["monomial_condition"] > 100.0 * out["chebyshev_condition"]


def test_the_monomial_conditioning_grows_and_the_chebyshev_one_stays_small():
    # Only while it is measurable. Past about degree 12 the Hilbert matrix condition number
    # exceeds 1/eps, and what numpy reports there is a floor of its own arithmetic, not a
    # measurement, so it stops being monotone.
    measurable = [ap.basis_conditioning(d)["monomial_condition"] for d in (2, 4, 6, 8, 10)]
    assert all(b > a for a, b in zip(measurable, measurable[1:]))
    assert ap.basis_conditioning(14)["monomial_condition"] > 1e15
    cheb = [ap.basis_conditioning(d)["chebyshev_condition"] for d in (4, 8, 12, 16, 20)]
    assert all(b > a for a, b in zip(cheb, cheb[1:]))
    assert max(cheb) < 1e2


@pytest.mark.parametrize("degree", [-1, -5])
def test_a_negative_degree_is_rejected(degree):
    with pytest.raises(ValueError, match="degree must be non-negative"):
        ap.best_l2_polynomial(np.exp, degree)
    with pytest.raises(ValueError, match="degree must be non-negative"):
        ap.remez(np.exp, degree)


def test_a_negative_weight_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ap.best_l2_polynomial(np.exp, 3, weight=lambda t: t)


# --------------------------------------------------------------------------- Remez


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 5, 8])
@pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (0.0, 1.0), (-2.0, 3.0)])
def test_remez_equioscillates_the_required_number_of_times(degree, lo, hi):
    f = lambda t: np.exp(t)
    out = ap.remez(f, degree, lo, hi)
    report = ap.equioscillation_report(f, out["evaluate"], lo, hi, degree)
    assert report["alternations"] >= degree + 2
    assert report["is_best"]


@pytest.mark.parametrize("degree", [1, 2, 4, 7])
@pytest.mark.parametrize("name,f", FUNCTIONS[:2] + FUNCTIONS[3:])
def test_remez_levels_the_error(degree, name, f):
    out = ap.remez(f, degree, -1.0, 1.0)
    assert out["levelled_error"] == pytest.approx(out["max_error"], rel=1e-3)


@pytest.mark.parametrize("degree", [1, 3, 6])
def test_remez_beats_the_least_squares_fit_in_the_max_norm(degree):
    f = lambda t: np.exp(t)
    out = ap.minimax_vs_least_squares(f, degree, -1.0, 1.0)
    assert out["minimax"]["linf"] <= out["least_squares"]["linf"] * (1.0 + 1e-9)
    assert out["linf_ratio"] >= 1.0 - 1e-9


@pytest.mark.parametrize("degree", [1, 3, 6])
def test_least_squares_beats_remez_in_the_l2_norm(degree):
    f = lambda t: np.exp(t)
    out = ap.minimax_vs_least_squares(f, degree, -1.0, 1.0)
    assert out["least_squares"]["l2"] <= out["minimax"]["l2"] * (1.0 + 1e-6)
    assert out["l2_ratio"] <= 1.0 + 1e-6


@pytest.mark.parametrize("degree", [0, 2, 4])
def test_remez_reproduces_a_polynomial_of_the_same_degree(degree):
    rng = np.random.default_rng(degree + 3)
    c = rng.standard_normal(degree + 1)
    poly = lambda t: sum(c[k] * np.asarray(t) ** k for k in range(degree + 1))
    out = ap.remez(poly, degree, -1.0, 1.0)
    assert out["max_error"] < 1e-10 * max(float(np.max(np.abs(c))), 1.0)


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_the_minimax_error_falls_as_the_degree_rises(degree):
    f = lambda t: np.exp(t)
    a = ap.remez(f, degree, -1.0, 1.0)["max_error"]
    b = ap.remez(f, degree + 1, -1.0, 1.0)["max_error"]
    assert b < a


def test_equioscillation_of_a_deliberately_bad_approximation_falls_short():
    f = lambda t: np.exp(t)
    bad = lambda t: np.zeros_like(np.atleast_1d(np.asarray(t, dtype=float)))
    report = ap.equioscillation_report(f, bad, -1.0, 1.0, 3)
    assert report["alternations"] < report["required"]
    assert not report["is_best"]


def test_equioscillation_of_an_exact_fit_reports_zero_error():
    f = lambda t: 2.0 * np.asarray(t) + 1.0
    report = ap.equioscillation_report(f, f, -1.0, 1.0, 1)
    assert report["max_error"] == 0.0 and report["is_best"]


@pytest.mark.parametrize("degree", [1, 3, 5])
def test_the_remez_minimum_is_strict(degree):
    out = ap.uniqueness_report(lambda t: np.exp(t), degree, -1.0, 1.0, n_trials=80,
                               rng=np.random.default_rng(degree))
    assert out["is_strict_minimum"]
    assert out["smallest_increase"] >= 0.0


# --------------------------------------------------------------------------- existence


@pytest.mark.parametrize("name,f", FUNCTIONS[:3])
def test_the_error_surface_has_an_interior_minimum_that_grows_at_the_edge(name, f):
    out = ap.existence_report(f, 1, -1.0, 1.0, span=8.0, n_grid=41, n_probe=101)
    assert out["minimum_is_interior"]
    assert out["grows_at_the_edge"]


@pytest.mark.parametrize("degree", [0, 2, 3])
def test_the_existence_picture_refuses_a_degree_it_cannot_draw(degree):
    with pytest.raises(ValueError, match="drawn for degree 1"):
        ap.existence_report(np.exp, degree)


# --------------------------------------------------------------------------- Weierstrass


@pytest.mark.parametrize("n", [1, 2, 5, 20, 64])
@pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-1.0, 1.0), (2.0, 5.0)])
def test_bernstein_reproduces_the_endpoints_exactly(n, lo, hi):
    f = lambda t: np.exp(t)
    got = ap.bernstein(f, n, np.array([lo, hi]), lo, hi)
    assert got[0] == pytest.approx(math.exp(lo), rel=1e-12)
    assert got[1] == pytest.approx(math.exp(hi), rel=1e-12)


@pytest.mark.parametrize("n", [1, 3, 10, 40])
@pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-2.0, 1.0)])
def test_bernstein_reproduces_affine_functions_at_every_n(n, lo, hi):
    f = lambda t: 3.0 * np.asarray(t) - 1.0
    t = np.linspace(lo, hi, 37)
    assert np.max(np.abs(ap.bernstein(f, n, t, lo, hi) - f(t))) < 1e-11


@pytest.mark.parametrize("n", [2, 8, 32])
def test_bernstein_never_leaves_the_range_of_the_data(n):
    rng = np.random.default_rng(n)
    values = rng.standard_normal(n + 1)
    f = lambda t: np.interp(np.asarray(t), np.linspace(0.0, 1.0, values.size), values)
    t = np.linspace(0.0, 1.0, 401)
    got = ap.bernstein(f, n, t)
    assert got.min() >= values.min() - 1e-12
    assert got.max() <= values.max() + 1e-12


@pytest.mark.parametrize("name,f", FUNCTIONS)
def test_the_weierstrass_rate_is_first_order(name, f):
    out = ap.weierstrass_rate(f, (8, 16, 32, 64, 128), 0.0, 1.0)
    assert 0.4 < out["fitted_order"] < 1.6


@pytest.mark.parametrize("n", [0, -3])
def test_bernstein_rejects_a_degenerate_subdivision(n):
    with pytest.raises(ValueError, match="at least one subdivision"):
        ap.bernstein(np.exp, n, np.array([0.5]))


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_bernstein_is_far_worse_than_the_best_polynomial_of_the_same_degree(degree):
    out = ap.weierstrass_against_best(lambda t: np.exp(t), (degree,), -1.0, 1.0)
    assert out["ratio"][0] > 5.0


def test_the_bernstein_gap_widens_with_the_degree():
    out = ap.weierstrass_against_best(lambda t: np.exp(t), (2, 4, 8, 16), -1.0, 1.0)
    assert out["ratio"][-1] > out["ratio"][0]
