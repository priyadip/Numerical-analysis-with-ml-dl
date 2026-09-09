"""Tests for `nalib.power`.

These are **rate** tests as much as correctness tests, because the content of the power method
family is entirely in how fast each one converges and what controls it. So there are tests that
the measured factor equals ``|lambda_2/lambda_1|``, that a shift changes which eigenvalue is
found, and that the Rayleigh quotient iteration is cubic on a symmetric matrix.

The failure modes are tested as failures, not skipped: equal-modulus eigenvalues stop power
iteration dead, and a real starting vector cannot reach a complex eigenvector.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import power


def symmetric_with_spectrum(spectrum, seed=1):
    """A symmetric matrix with exactly the given eigenvalues."""
    spectrum = np.asarray(spectrum, dtype=float)
    n = spectrum.size
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))
    return Q @ np.diag(spectrum) @ Q.T


def nonsymmetric_with_spectrum(spectrum, seed=1):
    """A diagonalizable non-symmetric matrix with exactly the given real eigenvalues."""
    spectrum = np.asarray(spectrum, dtype=float)
    n = spectrum.size
    S = np.random.default_rng(seed).standard_normal((n, n))
    return S @ np.diag(spectrum) @ np.linalg.inv(S)


# ---------------------------------------------------------------- the Rayleigh quotient


@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_the_quotient_of_an_eigenvector_is_its_eigenvalue(n, rng):
    A = symmetric_with_spectrum(np.arange(1.0, n + 1), seed=n)
    vals, vecs = np.linalg.eigh(A)
    for i in range(n):
        got = power.rayleigh_quotient(A, vecs[:, i])
        assert abs(complex(got).real - vals[i]) < 1e-10


@pytest.mark.parametrize("n", [5, 12])
def test_the_quotient_squares_the_error_for_a_symmetric_matrix(n):
    """**The fact the whole family rests on.** An eigenvector accurate to eps gives an
    eigenvalue accurate to eps^2, because the gradient of the quotient vanishes there."""
    A = symmetric_with_spectrum(np.arange(1.0, n + 1), seed=n)
    vals, vecs = np.linalg.eigh(A)
    gen = np.random.default_rng(3)
    direction = gen.standard_normal(n)
    direction -= vecs[:, -1] * (vecs[:, -1] @ direction)
    direction /= np.linalg.norm(direction)
    for eps in (1e-2, 1e-3, 1e-4):
        x = vecs[:, -1] + eps * direction
        err = abs(complex(power.rayleigh_quotient(A, x)).real - vals[-1])
        assert err < 20.0 * eps ** 2, f"eps={eps}: error {err:.3e} against eps^2 {eps ** 2:.3e}"


def test_the_quotient_does_not_square_the_error_for_a_nonsymmetric_matrix():
    """The squaring needs symmetry. Without it the error stays first order, which is why the
    non-symmetric Rayleigh quotient iteration is quadratic rather than cubic."""
    n = 6
    A = nonsymmetric_with_spectrum(np.arange(1.0, n + 1), seed=2)
    vals, vecs = np.linalg.eig(A)
    top = int(np.argmax(vals.real))
    v = vecs[:, top].real
    v = v / np.linalg.norm(v)
    gen = np.random.default_rng(4)
    d = gen.standard_normal(n)
    d -= v * (v @ d)
    d /= np.linalg.norm(d)
    eps = 1e-4
    err = abs(complex(power.rayleigh_quotient(A, v + eps * d)).real - vals[top].real)
    assert err > 10.0 * eps ** 2, f"error {err:.3e} was as small as the symmetric case"


def test_the_quotient_rejects_the_zero_vector(rng):
    with pytest.raises(ValueError):
        power.rayleigh_quotient(rng.standard_normal((4, 4)), np.zeros(4))


def test_the_quotient_rejects_a_length_mismatch(rng):
    with pytest.raises(ValueError):
        power.rayleigh_quotient(rng.standard_normal((4, 4)), np.ones(5))


# ---------------------------------------------------------------- power iteration


@pytest.mark.parametrize("spectrum", [[10.0, 5.0, 2.0, 1.0],
                                      [8.0, 1.0],
                                      [20.0, 4.0, 3.0, 2.0, 1.0],
                                      [-9.0, 3.0, 1.0]])
def test_power_iteration_finds_the_dominant_eigenvalue(spectrum):
    A = symmetric_with_spectrum(spectrum, seed=len(spectrum))
    out = power.power_iteration(A, tol=1e-12, max_iter=5000,
                                rng=np.random.default_rng(2))
    assert out.converged, out.message
    biggest = max(spectrum, key=abs)
    assert abs(complex(out.value).real - biggest) < 1e-8


@pytest.mark.parametrize("spectrum", [[10.0, 5.0, 2.0, 1.0],
                                      [10.0, 9.5, 3.0, 1.0],
                                      [10.0, 2.0, 1.0]])
def test_the_measured_rate_equals_the_eigenvalue_ratio(spectrum):
    """Measured to four decimal places: 0.5000 against a predicted 0.5000, and 0.9500 against
    0.9500."""
    A = symmetric_with_spectrum(spectrum, seed=len(spectrum))
    out = power.power_iteration(A, tol=1e-13, max_iter=5000,
                                rng=np.random.default_rng(2))
    r = np.array(out.residuals)
    usable = r[(r > 1e-11) & (r < r[0])]
    assert usable.size > 4, "not enough usable residuals to fit a rate"
    measured = float(np.median(usable[1:] / usable[:-1]))
    predicted = power.power_iteration_rate(A)
    assert abs(measured - predicted) < 0.01, \
        f"measured {measured:.4f} against predicted {predicted:.4f}"


def test_power_iteration_stalls_when_the_ratio_is_near_one():
    """It converges in theory and not in practice. Measured: 5000 iterations at a ratio of
    0.999 without reaching the tolerance."""
    A = symmetric_with_spectrum([10.0, 9.99, 3.0, 1.0], seed=4)
    out = power.power_iteration(A, tol=1e-13, max_iter=5000,
                                rng=np.random.default_rng(2))
    assert not out.converged
    assert power.power_iteration_rate(A) > 0.99


def test_power_iteration_fails_on_equal_moduli():
    """**The failure that is not a matter of speed.** With eigenvalues +5 and -5 nothing decays
    at all, and the iterate oscillates forever. Every real matrix with a dominant complex
    conjugate pair has this problem."""
    A = symmetric_with_spectrum([5.0, -5.0, 2.0, 1.0], seed=5)
    assert abs(power.power_iteration_rate(A) - 1.0) < 1e-12
    out = power.power_iteration(A, tol=1e-13, max_iter=2000,
                                rng=np.random.default_rng(2))
    assert not out.converged


def test_power_iteration_rate_of_a_single_eigenvalue():
    assert power.power_iteration_rate([[3.0]]) == 0.0


@pytest.mark.parametrize("n", [2, 5, 15])
def test_power_iteration_works_on_a_nonsymmetric_matrix(n):
    A = nonsymmetric_with_spectrum(np.arange(1.0, n + 1), seed=n)
    out = power.power_iteration(A, tol=1e-11, max_iter=5000,
                                rng=np.random.default_rng(6))
    assert out.converged, out.message
    assert abs(complex(out.value).real - float(n)) < 1e-6


def test_power_iteration_rejects_a_non_square_matrix(rng):
    with pytest.raises(ValueError):
        power.power_iteration(rng.standard_normal((3, 5)))


def test_power_iteration_rejects_a_zero_start(rng):
    with pytest.raises(ValueError):
        power.power_iteration(rng.standard_normal((4, 4)), x0=np.zeros(4))


def test_power_iteration_rejects_a_start_of_the_wrong_length(rng):
    with pytest.raises(ValueError):
        power.power_iteration(rng.standard_normal((4, 4)), x0=np.ones(5))


# ---------------------------------------------------------------- inverse iteration


@pytest.mark.parametrize("shift,expected", [(0.5, 1.0), (2.9, 3.0), (7.2, 7.0), (9.5, 10.0),
                                            (-1.0, 1.0)])
def test_the_shift_chooses_which_eigenvalue_is_found(shift, expected):
    """**The idea that makes every later method work.** Measured: shifts of 0.5, 2.9, 7.2 and
    9.5 find 1, 3, 7 and 10."""
    A = symmetric_with_spectrum([10.0, 7.0, 3.0, 1.0], seed=1)
    out = power.inverse_iteration(A, shift, tol=1e-12, max_iter=500,
                                  rng=np.random.default_rng(3))
    assert out.converged, out.message
    assert abs(complex(out.value).real - expected) < 1e-8, \
        f"shift {shift} found {out.value} rather than {expected}"


def test_a_distant_shift_picks_the_right_eigenvalue_and_crawls_towards_it():
    """**A shift outside the spectrum still selects, it just does not accelerate.** At sigma =
    100 the rate is |10-100|/|7-100| = 0.968, so 500 steps buy about seven digits and no more.
    Measured: the value is right to 13 digits and the residual test never fires.

    Which is the practical rule the later methods all follow: a shift is worth having only when
    it is CLOSE, and getting it close is what the Rayleigh quotient does."""
    A = symmetric_with_spectrum([10.0, 7.0, 3.0, 1.0], seed=1)
    predicted = power.inverse_iteration_rate(A, 100.0)
    assert predicted > 0.95, f"expected a slow rate, got {predicted:.4f}"
    out = power.inverse_iteration(A, 100.0, tol=1e-12, max_iter=500,
                                  rng=np.random.default_rng(3))
    assert not out.converged, "500 steps at rate 0.968 should not reach 1e-12"
    assert abs(complex(out.value).real - 10.0) < 1e-9, \
        f"it should still be heading for 10, got {out.value}"


@pytest.mark.parametrize("shift", [0.5, 2.9, 7.2, 9.5])
def test_the_inverse_iteration_rate_matches_the_gap_ratio(shift):
    A = symmetric_with_spectrum([10.0, 7.0, 3.0, 1.0], seed=1)
    out = power.inverse_iteration(A, shift, tol=1e-14, max_iter=500,
                                  rng=np.random.default_rng(3))
    r = np.array(out.residuals)
    usable = r[(r > 1e-12) & (r < r[0])]
    if usable.size < 4:
        pytest.skip("converged too fast to fit a rate")
    measured = float(np.median(usable[1:] / usable[:-1]))
    predicted = power.inverse_iteration_rate(A, shift)
    assert abs(measured - predicted) < 0.05, \
        f"measured {measured:.4f} against predicted {predicted:.4f}"


def test_a_closer_shift_converges_faster():
    A = symmetric_with_spectrum([10.0, 7.0, 3.0, 1.0], seed=1)
    far = power.inverse_iteration(A, 5.0, tol=1e-12, rng=np.random.default_rng(3))
    near = power.inverse_iteration(A, 6.9, tol=1e-12, rng=np.random.default_rng(3))
    assert near.iterations < far.iterations, f"{near.iterations} against {far.iterations}"


def test_a_nearly_exact_shift_still_gives_the_right_direction():
    """**Wilkinson's point, and the one people find hardest to believe.** The shifted matrix is
    numerically singular, so the solve has no correct digits in its magnitude. The error lies
    along the eigenvector being sought, and normalising throws the magnitude away, so the
    answer is excellent."""
    A = symmetric_with_spectrum([10.0, 7.0, 3.0, 1.0], seed=1)
    shift = 7.0 + 1e-13
    shifted = A - shift * np.eye(4)
    assert np.linalg.cond(shifted) > 1e11, \
        f"the test matrix was not ill conditioned enough: {np.linalg.cond(shifted):.2e}"
    out = power.inverse_iteration(A, shift, tol=1e-10, max_iter=100,
                                  rng=np.random.default_rng(3))
    assert abs(complex(out.value).real - 7.0) < 1e-9, f"got {out.value}"


def test_inverse_iteration_rejects_a_non_square_matrix(rng):
    with pytest.raises(ValueError):
        power.inverse_iteration(rng.standard_normal((3, 5)), 1.0)


# ---------------------------------------------------------------- Rayleigh quotient iteration


@pytest.mark.parametrize("n", [3, 5, 10, 25])
def test_rayleigh_quotient_iteration_converges_on_a_symmetric_matrix(n):
    A = symmetric_with_spectrum(np.arange(1.0, n + 1), seed=n)
    out = power.rayleigh_quotient_iteration(A, tol=1e-13, max_iter=100,
                                            rng=np.random.default_rng(11))
    assert out.converged, out.message
    assert np.min(np.abs(np.arange(1.0, n + 1) - complex(out.value).real)) < 1e-9


@pytest.mark.parametrize("n", [5, 10, 25])
def test_the_symmetric_rate_is_cubic(n):
    """Measured at n=10: residuals 1.0e+00, 1.8e-02, 4.6e-07, 8.3e-16. Each exponent is about
    three times the previous one, which is what cubic means."""
    A = symmetric_with_spectrum(np.arange(1.0, n + 1), seed=n)
    out = power.rayleigh_quotient_iteration(A, tol=1e-13, max_iter=100,
                                            rng=np.random.default_rng(11))
    r = np.array(out.residuals)
    usable = r[r > 1e-15]
    assert usable.size >= 3, "converged too fast to see a rate"
    logs = np.log10(usable)
    orders = [(logs[i + 1] - logs[0]) / (logs[i] - logs[0]) for i in range(1, logs.size - 1)]
    assert max(orders) > 2.5, f"orders {orders} never reached cubic"


@pytest.mark.parametrize("n", [4, 6, 10])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_it_converges_on_a_nonsymmetric_matrix_with_real_eigenvalues(n, seed):
    A = nonsymmetric_with_spectrum(np.arange(1.0, n + 1), seed=7)
    out = power.rayleigh_quotient_iteration(A, tol=1e-12, max_iter=100,
                                            rng=np.random.default_rng(seed))
    assert out.converged, out.message
    assert np.min(np.abs(np.arange(1.0, n + 1) - complex(out.value).real)) < 1e-7


def test_which_eigenvalue_it_finds_depends_on_where_it_starts():
    """**Not globally convergent, and that is the price of the speed.** Measured at n=10:
    three starting vectors found 5, 7 and 10."""
    A = nonsymmetric_with_spectrum(np.arange(1.0, 11.0), seed=7)
    found = set()
    for seed in (1, 2, 3, 4, 5):
        out = power.rayleigh_quotient_iteration(A, tol=1e-12, max_iter=100,
                                                rng=np.random.default_rng(seed))
        if out.converged:
            found.add(round(complex(out.value).real))
    assert len(found) > 1, f"every start found the same eigenvalue: {found}"


def test_real_arithmetic_cannot_reach_a_complex_eigenvector():
    """A real matrix with only complex eigenvalues has no real eigenvector, so a real starting
    vector cannot converge. Measured: the residual cycles between 0.63 and 0.35 forever."""
    A = np.random.default_rng(4).standard_normal((6, 6))
    assert np.max(np.abs(np.linalg.eigvals(A).imag)) > 0.1, "this matrix has real eigenvalues"
    out = power.rayleigh_quotient_iteration(A, tol=1e-12, max_iter=100,
                                            rng=np.random.default_rng(5))
    assert not out.converged
    assert len(out.residuals) > 50


def test_rqi_records_its_shifts():
    A = symmetric_with_spectrum(np.arange(1.0, 7.0), seed=6)
    out = power.rayleigh_quotient_iteration(A, tol=1e-13, rng=np.random.default_rng(11))
    assert len(out.shifts) == out.iterations or len(out.shifts) == out.iterations + 1
    assert abs(complex(out.shifts[-1]) - complex(out.value)) < 1e-6


# ---------------------------------------------------------------- deflation


@pytest.mark.parametrize("n", [3, 6, 12])
def test_deflation_removes_exactly_one_eigenvalue(n):
    A = symmetric_with_spectrum(np.arange(1.0, n + 1), seed=n)
    vals, vecs = np.linalg.eigh(A)
    B = power.deflate(A, vals[-1], vecs[:, -1])
    after = np.sort(np.linalg.eigvalsh(B))
    expected = np.sort(np.concatenate([vals[:-1], [0.0]]))
    assert np.max(np.abs(after - expected)) < 1e-10


def test_deflation_keeps_the_matrix_symmetric(rng):
    A = symmetric_with_spectrum([5.0, 3.0, 1.0], seed=2)
    vals, vecs = np.linalg.eigh(A)
    B = power.deflate(A, vals[-1], vecs[:, -1])
    assert np.linalg.norm(B - B.T) < 1e-12


def test_deflation_refuses_a_nonsymmetric_matrix(rng):
    """**It would silently give the wrong spectrum**, because the left and right eigenvectors
    differ. Refusing is the only safe behaviour."""
    A = nonsymmetric_with_spectrum([3.0, 2.0, 1.0], seed=1)
    vals, vecs = np.linalg.eig(A)
    with pytest.raises(ValueError, match="symmetric"):
        power.deflate(A, vals[0], vecs[:, 0])


def test_deflation_rejects_a_length_mismatch(rng):
    with pytest.raises(ValueError):
        power.deflate(np.eye(4), 1.0, np.ones(5))


@pytest.mark.parametrize("n,k", [(6, 3), (10, 4), (5, 5)])
def test_finding_several_eigenpairs_by_deflation(n, k):
    A = symmetric_with_spectrum(np.arange(1.0, n + 1) * 2.0, seed=n)
    out = power.find_k_eigenpairs(A, k, tol=1e-12, max_iter=5000,
                                  rng=np.random.default_rng(9))
    expected = np.sort(np.arange(1.0, n + 1) * 2.0)[::-1][:k]
    got = np.abs(np.real(out["values"]))
    assert np.max(np.abs(np.sort(got)[::-1] - expected)) < 1e-6, \
        f"got {got}, expected {expected}"


@pytest.mark.parametrize("tol", [1e-13, 1e-9, 1e-6, 1e-4])
def test_the_deflation_error_does_NOT_accumulate(tol):
    """**The folklore says it should, and measured it does not.**

    The expectation is that each deflation subtracts an eigenvector accurate only to the
    tolerance, so the errors compound. Measured at four tolerances and two spectra, the worst
    error is flat in k: 7.1e-15 at k=1 and 8.9e-15 at k=10.

    Symmetry is why, and it is the same mechanism that makes the Rayleigh quotient square its
    error. The eigenvectors are orthogonal, so an error in v lies perpendicular to v to first
    order and the deflation term is wrong by O(delta^2) rather than O(delta)."""
    n = 12
    A = symmetric_with_spectrum(np.arange(1.0, n + 1) * 2.0, seed=n)
    exact = np.sort(np.arange(1.0, n + 1) * 2.0)[::-1]
    errors = []
    for k in (1, 4, 10):
        out = power.find_k_eigenpairs(A, k, tol=tol, max_iter=8000,
                                      rng=np.random.default_rng(9))
        errors.append(float(np.max(np.abs(np.real(out["values"]) - exact[:k]))))
    # An absolute floor, not a ratio: errors[0] is sometimes exactly zero, and a ratio
    # against zero says nothing. Not accumulating means staying at the level the
    # tolerance sets, whatever k is.
    floor = max(20.0 * tol, 1e-13 * float(np.linalg.norm(A, 2)))
    assert errors[-1] <= floor, f"{errors} exceeded the floor {floor:.1e}"


def test_the_deflation_COST_does_grow():
    """What actually degrades is the speed, because each deflation leaves a spectrum whose
    ratios are closer to 1. Measured: 339 total iterations at k=1 and 1990 at k=10."""
    n = 12
    A = symmetric_with_spectrum(np.arange(1.0, n + 1) * 2.0, seed=n)
    small = sum(power.find_k_eigenpairs(A, 1, tol=1e-13, max_iter=8000,
                                        rng=np.random.default_rng(9))["iterations"])
    large = sum(power.find_k_eigenpairs(A, 10, tol=1e-13, max_iter=8000,
                                        rng=np.random.default_rng(9))["iterations"])
    assert large > 3 * small, f"{small} against {large}"


def test_find_k_rejects_a_bad_k(rng):
    A = symmetric_with_spectrum([3.0, 2.0, 1.0], seed=1)
    for bad in (0, -1, 4):
        with pytest.raises(ValueError):
            power.find_k_eigenpairs(A, bad)
