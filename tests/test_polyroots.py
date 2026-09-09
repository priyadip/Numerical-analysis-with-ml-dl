"""Tests for nalib.polyroots.

The counting routines (Descartes, Sturm) are checked against the true root counts, which are
known exactly because the test polynomials are built from their roots. The solvers are checked
against `numpy.roots`.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import polynomials as poly, polyroots as pr


def build(roots):
    return poly.from_roots(np.asarray(roots, dtype=float))


# ---------------------------------------------------------------- Descartes


@pytest.mark.parametrize(
    "roots, n_pos, n_neg",
    [
        ([1.0, 2.0, 3.0],       3, 0),
        ([-1.0, -2.0, -3.0],    0, 3),
        ([1.0, -2.0, 3.0],      2, 1),
        ([1.0, 2.0, -3.0, -4.0], 2, 2),
    ],
)
def test_descartes_bounds_the_true_counts(roots, n_pos, n_neg):
    d = pr.descartes_bounds(build(roots))
    assert n_pos in d["positive"], f"true count {n_pos} not in {d['positive']}"
    assert n_neg in d["negative"], f"true count {n_neg} not in {d['negative']}"
    assert max(d["positive"]) >= n_pos
    assert max(d["negative"]) >= n_neg


def test_descartes_proves_no_real_roots():
    for c in [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0]]:      # x^2+1, x^4+1
        d = pr.descartes_bounds(c)
        assert d["positive"] == [0] and d["negative"] == [0]
        assert np.all(np.abs(np.roots(c).imag) > 1e-12)


def test_descartes_sign_changes_ignores_zeros():
    assert pr.descartes_sign_changes([1.0, 0.0, -1.0]) == 1
    assert pr.descartes_sign_changes([1.0, 0.0, 0.0, 1.0]) == 0
    assert pr.descartes_sign_changes([1.0, -1.0, 1.0, -1.0]) == 3


# ---------------------------------------------------------------- Sturm


@pytest.mark.parametrize(
    "roots",
    [
        [1.0, 2.0, 3.0],
        [-2.0, 0.5, 1.0, 3.0, 7.0],
        [-5.0, -1.0, 0.25],
        [0.1, 0.2, 0.3, 0.4],
    ],
)
def test_sturm_count_is_exact_over_many_intervals(roots):
    roots = np.sort(np.asarray(roots, dtype=float))
    c = build(roots)
    for a, b in [(-20, 20), (0, 20), (-20, 0), (0.15, 5.0), (1.5, 2.5), (-1, 1)]:
        expected = int(np.sum((roots > a) & (roots <= b)))
        assert pr.sturm_count(c, a, b) == expected, f"({a},{b}] for roots {roots}"


def test_sturm_separates_close_roots():
    """Sturm resolves roots far closer than any practical sampling grid would."""
    gap = 1e-3
    c = build([1.0, 1.0 + gap, 5.0])
    assert pr.sturm_count(c, 0.0, 3.0) == 2
    assert pr.sturm_count(c, 0.0, 1.0 + gap / 2) == 1
    assert pr.sturm_count(c, 1.0 + gap / 2, 3.0) == 1


def test_sturm_has_a_floating_point_resolution_limit():
    """Sturm's theorem is exact in EXACT arithmetic. The chain is not exact in doubles.

    Building the chain needs repeated polynomial remainders, and each one loses accuracy.
    Below a gap of roughly 1e-4 the computed chain can no longer tell two nearby roots from
    one double root, and reports the wrong count. This is a limitation of the implementation
    in floating point, not of the theorem, and it is documented here so nobody relies on
    exactness that is not there.
    """
    resolved, lost = [], []
    for gap in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        c = build([1.0, 1.0 + gap, 5.0])
        (resolved if pr.sturm_count(c, 0.0, 3.0) == 2 else lost).append(gap)
    assert min(resolved) <= 1e-3, "should resolve gaps down to at least 1e-3"
    assert max(lost) >= 1e-5, "and should lose the smallest gaps"


def test_sturm_finds_no_roots_where_there_are_none():
    c = build([1.0, 2.0, 3.0])
    assert pr.sturm_count(c, 3.5, 100.0) == 0
    assert pr.sturm_count(c, -100.0, 0.5) == 0


def test_sturm_sequence_degrees_decrease():
    chain = pr.sturm_sequence(build([-2.0, 0.5, 1.0, 3.0, 7.0]))
    degrees = [len(p) - 1 for p in chain]
    assert degrees[0] == 5 and degrees[1] == 4
    assert all(a > b for a, b in zip(degrees, degrees[1:]))


def test_sturm_rejects_the_zero_polynomial():
    with pytest.raises(ValueError):
        pr.sturm_sequence([0.0, 0.0])


def test_cauchy_bound_contains_every_root():
    for roots in [[1.0, 2.0, 3.0], [-7.0, 0.5], [10.0, -10.0, 1.0]]:
        c = build(roots)
        b = pr.cauchy_bound(c)
        assert np.all(np.abs(np.roots(c)) <= b + 1e-10)


def test_cauchy_bound_rejects_zero_leading_coefficient():
    with pytest.raises(ValueError):
        pr.cauchy_bound([0.0, 1.0, 2.0])


# ---------------------------------------------------------------- Birge-Vieta


def test_birge_vieta_finds_a_root():
    c = build([1.0, 2.0, 3.0])
    for x0 in [0.5, 1.8, 2.6, 5.0]:
        res = pr.birge_vieta(c, x0)
        assert res["converged"]
        assert abs(np.polyval(c, res["root"])) < 1e-10
        assert np.min(np.abs(np.array([1.0, 2.0, 3.0]) - res["root"])) < 1e-10


def test_birge_vieta_matches_plain_newton():
    """It IS Newton, so it must produce the identical iterates."""
    from nalib import roots as R
    c = build([1.0, 2.0, 3.0])
    dc = poly.derivative_coeffs(c)
    bv = pr.birge_vieta(c, 5.0, tol=1e-14)
    nw = R.newton(lambda x: np.polyval(c, x), lambda x: np.polyval(dc, x),
                  5.0, tol=1e-14)
    k = min(bv["iterates"].size, nw.iterates.size)
    np.testing.assert_allclose(bv["iterates"][:k], nw.iterates[:k], atol=1e-12)


def test_birge_vieta_reports_a_vanishing_derivative():
    res = pr.birge_vieta([1.0, 0.0, 1.0], 0.0)      # x^2+1, p'(0) = 0
    assert not res["converged"]


# ---------------------------------------------------------------- Bairstow


@pytest.mark.parametrize(
    "coeffs, r0, s0",
    [
        (np.array([1.0, 0, 0, 0, 1.0]),                          1.0, -1.0),
        (np.polymul([1.0, -2.0], [1.0, 1.0, 1.0]),              -1.0, -1.0),
        (np.polymul([1.0, 0, 4.0], [1.0, 0, 9.0]),               0.1, -1.0),
        (np.polymul([1.0, -1.0, 2.0], [1.0, 3.0, 5.0]),          1.0, -2.0),
    ],
)
def test_bairstow_extracts_a_genuine_quadratic_factor(coeffs, r0, s0):
    res = pr.bairstow(coeffs, r0, s0)
    assert res["converged"], res["message"]
    got = np.array(res["roots"])
    assert np.abs(np.polyval(coeffs, got)).max() < 1e-9


def test_bairstow_finds_complex_roots_using_only_real_arithmetic():
    c = np.array([1.0, 0.0, 0.0, 0.0, 1.0])          # x^4 + 1, all roots complex
    res = pr.bairstow(c, 1.0, -1.0)
    assert res["converged"]
    assert isinstance(res["r"], float) and isinstance(res["s"], float)
    assert abs(np.array(res["roots"])[0].imag) > 0.1


def test_bairstow_quotient_has_the_right_degree():
    c = np.polymul([1.0, -2.0], [1.0, 1.0, 1.0])     # degree 3
    res = pr.bairstow(c, -1.0, -1.0)
    assert res["converged"]
    assert len(res["quotient"]) == 2                  # degree 3 minus 2


def test_bairstow_rejects_low_degree():
    with pytest.raises(ValueError):
        pr.bairstow([1.0, -1.0], 1.0, 1.0)


# ---------------------------------------------------------------- Graeffe


def test_graeffe_recovers_well_separated_magnitudes():
    g = pr.graeffe_magnitudes(build([1.0, 2.0, 4.0]), steps=6)
    np.testing.assert_allclose(np.sort(g["magnitudes"]), [1.0, 2.0, 4.0], rtol=1e-6)


def test_graeffe_overflows_and_says_so():
    """The documented failure mode: coefficients square every step."""
    g = pr.graeffe_magnitudes(build([1.0, 2.0, 4.0, 8.0, 16.0]), steps=30)
    assert g["overflowed"]
    assert g["steps_used"] < 30
    assert g["steps_used"] >= 3, "it should manage at least a few steps"


def test_graeffe_step_doubles_the_log_of_the_coefficients():
    c = build([1.0, 2.0, 4.0])
    a = np.max(np.abs(c))
    b = np.max(np.abs(pr.graeffe_step(c)))
    d = np.max(np.abs(pr.graeffe_step(pr.graeffe_step(c))))
    # log grows roughly geometrically, which is why overflow comes so fast
    assert np.log(d) > 1.5 * np.log(b) > 1.5 * np.log(a)


# ---------------------------------------------------------------- companion matrix


@pytest.mark.parametrize(
    "roots",
    [[1.0, 2.0, 3.0], [-1.0, 0.5], [2.0, 2.0, 5.0], [-3.0, 0.25, 1.0, 4.0]],
)
def test_companion_eigenvalues_are_the_roots(roots):
    c = build(roots)
    got = np.sort(pr.roots_via_companion(c).real)
    np.testing.assert_allclose(got, np.sort(roots), atol=1e-9)


def test_companion_characteristic_polynomial_is_the_input():
    c = build([1.0, 2.0, 3.0])
    C = pr.companion_matrix(c)
    np.testing.assert_allclose(np.poly(C), c / c[0], atol=1e-10)


def test_companion_matches_numpy_roots():
    rng = np.random.default_rng(7)
    for _ in range(50):
        roots = rng.uniform(-3, 3, int(rng.integers(2, 7)))
        c = build(roots)
        a = np.sort_complex(pr.roots_via_companion(c))
        b = np.sort_complex(np.roots(c))
        assert np.abs(a - b).max() < 1e-8


def test_companion_rejects_zero_leading_coefficient():
    with pytest.raises(ValueError):
        pr.companion_matrix([0.0, 1.0, 1.0])


# ---------------------------------------------------------------- deflation


def test_deflation_finds_every_root_when_the_order_is_good():
    for n in [3, 5, 6, 8]:
        true = np.arange(1.0, n + 1)
        res = pr.all_roots_by_deflation(build(true), start=0.0)
        assert res["n_found"] == n, f"n={n}: found only {res['n_found']}"
        np.testing.assert_allclose(np.sort(res["raw"]), true, atol=1e-8)


def test_deflation_degrades_in_the_wrong_order():
    """Finding large roots first corrupts the deflated polynomial faster."""
    n = 6
    true = np.arange(1.0, n + 1)
    c = build(true)
    small = pr.all_roots_by_deflation(c, start=0.0)
    large = pr.all_roots_by_deflation(c, start=float(n) + 2)
    e_small = np.abs(np.sort(small["raw"]) - true).max()
    e_large = np.abs(np.sort(large["raw"]) - true).max()
    assert e_large > 3 * e_small, f"small {e_small:.2e}, large {e_large:.2e}"


def test_deflation_can_break_down_completely():
    """At degree 10 in the wrong order it stops finding roots at all."""
    res = pr.all_roots_by_deflation(build(np.arange(1.0, 11.0)), start=12.0)
    assert res["n_found"] < 10


def test_polished_roots_satisfy_the_original_polynomial():
    true = np.arange(1.0, 7.0)
    c = build(true)
    res = pr.all_roots_by_deflation(c, start=0.0)
    residuals = np.abs(np.polyval(c, res["polished"]))
    scale = np.abs(np.vander(res["polished"], len(c))) @ np.abs(c)
    assert np.max(residuals / scale) < 1e-13, "polished roots should be backward stable"
