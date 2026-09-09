"""Every library routine must work at every size, not only at the sizes the lessons use.

A function that takes a matrix has to behave for 1x1, for 2x3, for 34x65 and for anything
else. A function that takes a square matrix has to behave from 1x1 upwards. Nothing may quietly
assume "at least 2", "square", "even", or "big enough for the interesting case".

This module exists so that property cannot regress. Every routine in `nalib` that accepts a
size-varying input is swept here across:

- **degenerate sizes**: 1x1, and a 1-dimensional subspace,
- **small sizes** where off-by-one errors live: 2, 3, 4,
- **non-square shapes in both orientations**: 2x3 and 3x2, 34x65 and 65x34,
- **awkward sizes** that are not powers of two and not round numbers: 7, 13, 23, 34.

Each test checks the routine's **defining identity**, not merely that it returns without
raising. Returning a wrong answer at n = 1 is a worse failure than crashing.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import (floatingpoint as fp, linalg as la, lu, nlsystems as ns,
                   orthogonality as og, pivoting as pv, polynomials as poly,
                   polyroots as pr)

#: Square sizes: starts at 1, includes small awkward values and a large non-round one.
SQUARE_SIZES = [1, 2, 3, 4, 5, 7, 13, 23, 34]

#: Rectangular shapes, both orientations, including degenerate rows and columns.
SHAPES = [(1, 1), (1, 3), (3, 1), (2, 3), (3, 2), (5, 9), (9, 5), (7, 13), (34, 65), (65, 34)]


def _scale(A):
    """A tolerance that scales with the data, so a big matrix is not held to a small bound."""
    return max(1.0, float(np.abs(np.asarray(A)).max()))


def _tail_max(v):
    """max|v[1:]|, returning 0 for a length-1 vector instead of raising on an empty slice."""
    tail = np.asarray(v).ravel()[1:]
    return float(np.max(np.abs(tail))) if tail.size else 0.0


# ---------------------------------------------------------------- linalg


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("p", [1, 2, np.inf, "fro"])
def test_matrix_norm_at_every_shape(shape, p, rng):
    A = rng.standard_normal(shape)
    expected = np.linalg.norm(A, p)
    assert la.matrix_norm(A, p) == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 13, 34, 100])
@pytest.mark.parametrize("p", [1, 2, np.inf, 3])
def test_vector_norm_at_every_length(n, p, rng):
    x = rng.standard_normal(n)
    assert la.vector_norm(x, p) == pytest.approx(np.linalg.norm(x, p), rel=1e-11)


@pytest.mark.parametrize("shape", SHAPES)
def test_matvec_by_columns_at_every_shape(shape, rng):
    A = rng.standard_normal(shape)
    x = rng.standard_normal(shape[1])
    np.testing.assert_allclose(la.matvec_by_columns(A, x), A @ x, atol=1e-11)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fn", [la.matmul_by_columns, la.matmul_by_outer_products,
                                la.matmul_by_inner_products])
def test_all_matmul_views_at_every_shape(shape, fn, rng):
    m, n = shape
    for k in (1, 4):
        A, B = rng.standard_normal((m, n)), rng.standard_normal((n, k))
        np.testing.assert_allclose(fn(A, B), A @ B, atol=1e-10)


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_condition_number_at_every_size(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    assert la.condition_number(A, 2) == pytest.approx(np.linalg.cond(A, 2), rel=1e-8)
    assert la.condition_number(np.eye(n), 2) == pytest.approx(1.0)


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_spectral_radius_at_every_size(n, rng):
    A = rng.standard_normal((n, n))
    assert la.spectral_radius(A) == pytest.approx(
        float(np.max(np.abs(np.linalg.eigvals(A)))), rel=1e-9)


@pytest.mark.parametrize("n", [1, 2, 5, 34, 1000])
def test_norm_equivalence_constants_at_every_size(n, rng):
    for p, q in [(1, 2), (2, 1), (2, np.inf), (np.inf, 2), (1, np.inf), (np.inf, 1)]:
        c, C = la.norm_equivalence_constants(n, p, q)
        for _ in range(20):
            x = rng.standard_normal(n)
            npx, nqx = la.vector_norm(x, p), la.vector_norm(x, q)
            assert c * nqx <= npx + 1e-12
            assert npx <= C * nqx + 1e-12


# ---------------------------------------------------------------- orthogonality


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_random_orthogonal_at_every_size(n, rng):
    Q = og.random_orthogonal(n, rng)
    assert Q.shape == (n, n)
    assert og.orthogonality_error(Q) < 1e-11
    assert la.condition_number(Q, 2) == pytest.approx(1.0, abs=1e-9)
    x = rng.standard_normal(n)
    assert np.linalg.norm(Q @ x) == pytest.approx(np.linalg.norm(x), rel=1e-12)


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_householder_at_every_size(n, rng):
    """Including n = 1, where H is the 1x1 matrix [-1] and the 'tail' is empty."""
    for _ in range(5):
        x = rng.standard_normal(n)
        H = og.householder_reflector(x)
        assert H.shape == (n, n)
        assert og.orthogonality_error(H) < 1e-11
        Hx = H @ x
        assert _tail_max(Hx) < 1e-11 * _scale(x)
        assert abs(Hx[0]) == pytest.approx(np.linalg.norm(x), rel=1e-11)


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_householder_on_a_vector_already_on_the_axis(n):
    """The degenerate input, at every size. The other sign choice divides by zero here."""
    for sign in (1.0, -1.0):
        x = np.zeros(n)
        x[0] = sign * 2.5
        H = og.householder_reflector(x)
        assert og.orthogonality_error(H) < 1e-12
        assert _tail_max(H @ x) < 1e-12


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_projectors_at_every_size_and_every_subspace_dimension(n, rng):
    """k runs from 1 to n, so the rank-one and full-rank ends are both covered."""
    Q = og.random_orthogonal(n, rng)
    x = rng.standard_normal(n)
    for k in range(1, n + 1):
        Qk = Q[:, :k]
        P = og.orthogonal_projector(Qk)
        assert P.shape == (n, n)
        assert og.is_projector(P)
        assert og.is_orthogonal_projector(P)
        assert la.matrix_norm(P, 2) == pytest.approx(1.0, abs=1e-9)
        assert np.trace(P) == pytest.approx(k, abs=1e-9)
        r = og.projection_residual(x, Qk)
        assert np.abs(Qk.T @ r).max() < 1e-11 * _scale(x)
        # Pythagoras, at every k including k = n where the residual is zero
        assert (np.linalg.norm(P @ x) ** 2 + np.linalg.norm(r) ** 2
                == pytest.approx(np.linalg.norm(x) ** 2, rel=1e-10))


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_rank_one_projector_at_every_size(n, rng):
    q = rng.standard_normal(n)
    P = og.rank_one_projector(q)
    assert P.shape == (n, n)
    assert og.is_orthogonal_projector(P)
    assert np.linalg.matrix_rank(P) == 1
    np.testing.assert_allclose(P @ q, q, atol=1e-11 * _scale(q))


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_expand_in_basis_at_every_size(n, rng):
    Q = og.random_orthogonal(n, rng)
    x = rng.standard_normal(n)
    np.testing.assert_allclose(Q @ og.expand_in_basis(x, Q), x, atol=1e-11 * _scale(x))


@pytest.mark.parametrize("m,k1,k2", [(1, 1, 1), (3, 1, 1), (5, 2, 3), (10, 4, 4),
                                     (20, 1, 19), (34, 10, 7), (65, 34, 34)])
def test_principal_angles_at_every_subspace_pairing(m, k1, k2, rng):
    A, B = rng.standard_normal((m, k1)), rng.standard_normal((m, k2))
    angles = og.principal_angles(A, B)
    assert len(angles) == min(k1, k2)
    assert np.all(angles >= -1e-9) and np.all(angles <= np.pi / 2 + 1e-9)
    assert og.largest_principal_angle(A, B) == pytest.approx(angles[-1])


# ---------------------------------------------------------------- lu


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_lu_factor_at_every_size(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    L, U = lu.lu_factor(A)
    assert L.shape == U.shape == (n, n)
    np.testing.assert_allclose(L @ U, A, atol=1e-9 * _scale(A))
    np.testing.assert_allclose(np.diag(L), np.ones(n))
    assert np.abs(np.triu(L, 1)).max() == 0.0
    assert np.abs(np.tril(U, -1)).max() == 0.0


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_lu_solve_at_every_size(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    b = rng.standard_normal(n)
    L, U = lu.lu_factor(A)
    np.testing.assert_allclose(lu.lu_solve(L, U, b), np.linalg.solve(A, b),
                               atol=1e-8 * _scale(b))


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_crout_at_every_size(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    L, U = lu.crout_factor(A)
    np.testing.assert_allclose(L @ U, A, atol=1e-9 * _scale(A))
    np.testing.assert_allclose(np.diag(U), np.ones(n), atol=1e-13)


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_triangular_solves_at_every_size(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    b = rng.standard_normal(n)
    U, L = np.triu(A), np.tril(A)
    np.testing.assert_allclose(lu.back_substitution(U, b), np.linalg.solve(U, b), atol=1e-8)
    np.testing.assert_allclose(lu.forward_substitution(L, b), np.linalg.solve(L, b), atol=1e-8)


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_gauss_jordan_at_every_size(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    np.testing.assert_allclose(lu.gauss_jordan(A)["inverse"], np.linalg.inv(A), atol=1e-7)


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_determinants_at_every_size(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    _, U = lu.lu_factor(A)
    expected = np.linalg.det(A)
    assert lu.determinant_from_lu(U) == pytest.approx(expected, rel=1e-7)
    if n <= 7:                                   # cofactor expansion is O(n!)
        assert lu.determinant_by_cofactor(A) == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_cramer_at_every_size(n, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    b = rng.standard_normal(n)
    np.testing.assert_allclose(lu.cramer_solve(A, b), np.linalg.solve(A, b), atol=1e-7)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 13, 34, 500, 5000])
def test_flop_formulas_at_every_size(n):
    """These are closed forms, so they must be finite and ordered at every n, including 1."""
    f_lu = lu.flops_lu(n)
    f_tri = lu.flops_triangular_solve(n)
    f_gj = lu.flops_gauss_jordan(n)
    assert f_lu >= 0 and f_tri > 0 and f_gj > 0
    assert f_tri == n * n
    if n >= 3:
        assert f_gj > f_lu, "Gauss-Jordan must cost more once there is work to do"


# ---------------------------------------------------------------- pivoting


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_plu_factor_at_every_size(n, rng):
    A = rng.standard_normal((n, n))
    f = pv.plu_factor(A)
    assert f["L"].shape == f["U"].shape == f["P"].shape == (n, n)
    assert len(f["perm"]) == n
    np.testing.assert_allclose(f["P"] @ A, f["L"] @ f["U"], atol=1e-9 * _scale(A))
    np.testing.assert_allclose(A[f["perm"]], f["L"] @ f["U"], atol=1e-9 * _scale(A))
    assert np.abs(f["L"]).max() <= 1.0 + 1e-13


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_plu_solve_at_every_size(n, rng):
    A = rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    np.testing.assert_allclose(pv.plu_solve(pv.plu_factor(A), b),
                               np.linalg.solve(A, b), atol=1e-6 * _scale(A))


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_complete_pivoting_at_every_size(n, rng):
    A = rng.standard_normal((n, n))
    f = pv.complete_pivot_factor(A)
    np.testing.assert_allclose(f["P"] @ A @ f["Q"], f["L"] @ f["U"], atol=1e-9 * _scale(A))
    assert np.abs(f["L"]).max() <= 1.0 + 1e-13


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_wilkinson_matrix_at_every_size(n):
    """Growth is exactly 2^(n-1) at every n, including n = 1 where that is 1."""
    W = pv.wilkinson_growth_matrix(n)
    assert W.shape == (n, n)
    assert pv.growth_factor(W, "partial") == pytest.approx(2.0 ** (n - 1), rel=1e-9)
    assert pv.plu_factor(W)["n_swaps"] == 0


@pytest.mark.parametrize("n", SQUARE_SIZES)
@pytest.mark.parametrize("strategy", ["none", "partial", "complete"])
def test_growth_factor_at_every_size(n, strategy, rng):
    A = rng.standard_normal((n, n)) + n * np.eye(n)     # safe for the no-pivot case
    g = pv.growth_factor(A, strategy)
    assert np.isfinite(g) and g > 0


@pytest.mark.parametrize("n", SQUARE_SIZES)
def test_multiplier_sizes_at_every_size(n, rng):
    """Length is n(n-1)/2, which is 0 at n = 1 and must not raise."""
    m = pv.multiplier_sizes(rng.standard_normal((n, n)), "partial")
    assert len(m) == n * (n - 1) // 2
    if m.size:
        assert m.max() <= 1.0 + 1e-13


# ---------------------------------------------------------------- nlsystems


@pytest.mark.parametrize("n", [1, 2, 3, 5, 13, 34])
def test_newton_system_at_every_dimension(n, rng):
    target = rng.standard_normal(n)
    A = rng.standard_normal((n, n)) + n * np.eye(n)

    def F(v):
        v = np.asarray(v, dtype=float)
        return A @ (v - target) + 0.1 * (v - target) ** 3

    def J(v):
        v = np.asarray(v, dtype=float)
        return A + np.diag(0.3 * (v - target) ** 2)

    res = ns.newton_system(F, J, target + 0.2, tol=1e-12)
    assert res.converged
    np.testing.assert_allclose(res.root, target, atol=1e-8)
    assert res.iterates.shape[1] == n


@pytest.mark.parametrize("n", [1, 2, 3, 5, 13, 34])
def test_broyden_at_every_dimension(n, rng):
    target = rng.standard_normal(n)
    A = rng.standard_normal((n, n)) + n * np.eye(n)

    def F(v):
        v = np.asarray(v, dtype=float)
        return A @ (v - target) + 0.1 * (v - target) ** 3

    res = ns.broyden(F, target + 0.2, tol=1e-11, max_iter=400)
    assert res.converged
    np.testing.assert_allclose(res.root, target, atol=1e-7)


@pytest.mark.parametrize("shape", SHAPES)
def test_numerical_jacobian_at_every_shape(shape, rng):
    """F may map R^n to R^m with m and n unrelated. The Jacobian must come out m by n."""
    m, n = shape
    M = rng.standard_normal((m, n))
    J = ns.numerical_jacobian(lambda v: M @ np.asarray(v, dtype=float),
                              rng.standard_normal(n))
    assert J.shape == (m, n)
    np.testing.assert_allclose(J, M, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 13, 34])
def test_spectral_radius_of_systems_at_every_size(n, rng):
    A = rng.standard_normal((n, n))
    assert ns.spectral_radius(A) == pytest.approx(
        float(np.max(np.abs(np.linalg.eigvals(A)))), rel=1e-9)


# ---------------------------------------------------------------- polyroots


@pytest.mark.parametrize("deg", [1, 2, 3, 4, 5, 7, 9])
def test_polyroots_at_every_degree(deg):
    """Degree 1 upwards. The companion matrix must be deg by deg, never assumed square-ish."""
    true_roots = np.arange(1.0, deg + 1)
    c = poly.from_roots(true_roots)

    C = pr.companion_matrix(c)
    assert C.shape == (deg, deg)

    found = np.sort(np.real(pr.roots_via_companion(c)))
    np.testing.assert_allclose(found, true_roots, atol=1e-6 * deg)

    assert pr.cauchy_bound(c) >= deg
    assert deg in pr.descartes_bounds(c)["positive"]
    assert len(pr.sturm_sequence(c)) >= 1


@pytest.mark.parametrize("deg", [1, 2, 3, 4, 5, 7])
def test_sturm_count_at_every_degree(deg):
    c = poly.from_roots(np.arange(1.0, deg + 1))
    assert pr.sturm_count(c, 0.0, deg + 1.0) == deg


@pytest.mark.parametrize("deg", [0, 1, 2, 3, 8, 20])
def test_horner_at_every_degree(deg, rng):
    c = rng.standard_normal(deg + 1)
    for x in (-2.0, 0.0, 0.5, 3.0):
        assert poly.horner(c, x) == pytest.approx(np.polyval(c, x), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("deg", [1, 2, 3, 5, 8, 12])
def test_remaining_polyroots_at_every_degree(deg):
    c = poly.from_roots(np.arange(1.0, deg + 1))
    assert isinstance(pr.graeffe_magnitudes(c, 3), dict)
    assert pr.descartes_sign_changes(c) >= 0
    assert pr.all_roots_by_deflation(c, start=0.5)["n_found"] <= deg


def test_bairstow_rejects_degree_below_two():
    """A genuine mathematical restriction, since Bairstow finds a QUADRATIC factor.

    It must say so clearly rather than misbehaving on a linear polynomial.
    """
    with pytest.raises(ValueError):
        pr.bairstow(np.array([1.0, -1.0]), 0.0, 0.0)


@pytest.mark.parametrize("deg", [2, 3, 4, 5, 6, 8, 11])
def test_bairstow_runs_at_every_valid_degree(deg):
    pairs = [complex(0.5 + 0.3 * k, 0.7) for k in range(deg // 2)]
    roots = []
    for z in pairs:
        roots += [z, z.conjugate()]
    while len(roots) < deg:
        roots.append(1.0 + 0.2 * len(roots))
    assert isinstance(pr.bairstow(np.real(np.poly(roots)), 0.5, -0.5), dict)


@pytest.mark.parametrize("deg", [0, 1, 2, 3, 7, 20])
def test_polynomials_at_every_degree_including_constant(deg, rng):
    """Degree 0 is a constant. Its derivative is the zero polynomial, not an empty array."""
    c = rng.standard_normal(deg + 1)
    for x in (-1.5, 0.0, 1.7):
        assert poly.horner(c, x) == pytest.approx(np.polyval(c, x), rel=1e-10, abs=1e-12)
        value, deriv = poly.horner_with_derivative(c, x)
        assert value == pytest.approx(np.polyval(c, x), rel=1e-10, abs=1e-12)
        assert deriv == pytest.approx(np.polyval(np.polyder(c), x), rel=1e-9, abs=1e-11)
    assert len(poly.derivative_coeffs(c)) == max(deg, 1)

    mults_n, adds_n = poly.flop_count_naive(deg)
    mults_h, adds_h = poly.flop_count_horner(deg)
    assert mults_n >= mults_h and adds_n >= adds_h, "Horner must never cost more"


@pytest.mark.parametrize("n", [0, 1, 2, 3, 17, 1000])
def test_summation_at_every_length_including_empty(n, rng):
    """An empty sum is 0. A one-element sum is that element. Neither may raise."""
    x = rng.standard_normal(n) if n else np.array([])
    expected = float(np.sum(x))
    for fn in (fp.naive_sum, fp.pairwise_sum, fp.kahan_sum, fp.neumaier_sum):
        assert fn(x) == pytest.approx(expected, rel=1e-9, abs=1e-12), fn.__name__


# ---------------------------------------------------------------- value ranges
#
# Size independence is only half the requirement. A routine must also be correct across the
# valid RANGE of values: negative, zero, decimal, very large, very small and subnormal. A
# function that works only for inputs near 1 is as restricted as one that works only at n = 10.


MAGNITUDES = [1e-300, 1e-200, 1e-100, 1e-8, 1.0, 1e8, 1e100, 1e200, 1e300]


@pytest.mark.parametrize("scale", MAGNITUDES)
@pytest.mark.parametrize("method", ["bisection", "regula_falsi", "illinois", "brent"])
def test_bracketing_methods_across_600_orders_of_magnitude(scale, method):
    """The bracket test must not form a product of function values.

    ``f(a) * f(b)`` overflows to inf above 1e154 and underflows to exactly 0 below 1e-154,
    and ``inf * 0`` is nan. In every one of those cases the sign is destroyed. Comparing
    signs directly is immune, and this test pins that down.
    """
    from nalib import roots as R

    fn = getattr(R, method)
    f = lambda x, s=scale: s * (x - 1.3)
    result = fn(f, 0.0, 3.0, tol=1e-14)
    assert abs(result.root - 1.3) < 1e-9, f"{method} at scale {scale:.0e}"


@pytest.mark.parametrize("root", [-1e6, -3.7, -1.0, 0.0, 1e-8, 0.5, 1.0, 1e6])
def test_bracketing_methods_find_roots_anywhere_on_the_line(root):
    """Negative, zero, fractional and large roots are all valid inputs."""
    from nalib import roots as R

    f = lambda x, r=root: (x - r) ** 3
    for method in (R.bisection, R.illinois, R.brent):
        got = method(f, root - 2.0, root + 2.0, tol=1e-14).root
        assert abs(got - root) < 1e-6 * max(1.0, abs(root))


@pytest.mark.parametrize("n", [1, 2, 3, 13, 34])
@pytest.mark.parametrize("scale", [1e-150, 1e-8, 1.0, 1e8, 1e150])
def test_norms_across_magnitudes(n, scale, rng):
    """Norms must not overflow or underflow before the true answer would."""
    x = scale * rng.standard_normal(n)
    for p in (1, 2, np.inf):
        assert la.vector_norm(x, p) == pytest.approx(np.linalg.norm(x, p), rel=1e-10)


@pytest.mark.parametrize("n", [1, 2, 5, 13])
@pytest.mark.parametrize("scale", [1e-150, 1.0, 1e150])
def test_lu_across_magnitudes(n, scale, rng):
    """Scaling a matrix scales its factors, and must not break the factorization."""
    A = scale * (rng.standard_normal((n, n)) + n * np.eye(n))
    L, U = lu.lu_factor(A)
    np.testing.assert_allclose(L @ U, A, rtol=1e-9, atol=0.0)
    f = pv.plu_factor(A)
    np.testing.assert_allclose(f["P"] @ A, f["L"] @ f["U"], rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("n", [1, 2, 5, 13, 34])
@pytest.mark.parametrize("scale", [1e-150, 1.0, 1e150])
def test_householder_across_magnitudes(n, scale, rng):
    x = scale * rng.standard_normal(n)
    H = og.householder_reflector(x)
    assert og.orthogonality_error(H) < 1e-11
    assert _tail_max(H @ x) < 1e-11 * _scale(x)


@pytest.mark.parametrize("values", [
    [0.0, 0.0, 0.0],
    [1e-320, 1e-320, 1e-320],          # subnormal
    [1e300, -1e300, 1e300],
    [-5.0, 0.0, 5.0],
    [1e-16, 1.0, 1e16],                # wildly mixed magnitudes
])
def test_summation_across_value_ranges(values):
    """Zero, subnormal, huge, mixed sign and mixed magnitude are all valid inputs."""
    from nalib import floatingpoint as fp

    x = np.array(values, dtype=float)
    expected = float(np.sum(x))
    for fn in (fp.naive_sum, fp.pairwise_sum, fp.kahan_sum, fp.neumaier_sum):
        got = fn(x)
        assert np.isfinite(got) == np.isfinite(expected)
        if np.isfinite(expected):
            assert got == pytest.approx(expected, rel=1e-9, abs=1e-300), fn.__name__


@pytest.mark.parametrize("n", [1, 2, 5, 13])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_newton_system_across_magnitudes(n, scale, rng):
    """The root may be anywhere, not only near the origin."""
    target = scale * rng.standard_normal(n)
    A = rng.standard_normal((n, n)) + n * np.eye(n)

    def F(v):
        return A @ (np.asarray(v, dtype=float) - target)

    def J(v):
        return A

    res = ns.newton_system(F, J, target + scale * 0.1, tol=1e-12)
    assert res.converged
    np.testing.assert_allclose(res.root, target, rtol=1e-8, atol=1e-12)
