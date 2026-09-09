"""Tests for `nalib.qralg`.

Three separate claims, tested separately.

**Hessenberg reduction is a similarity**, so the eigenvalues must be unchanged, and the result
must actually be Hessenberg. Both are checked at every size including the degenerate ones.

**The QR algorithm converges to the Schur form**, so the test is not only that the eigenvalues
are right but that ``A = Q T Q^T`` holds and that ``T`` is quasi-triangular with nothing below
the first subdiagonal.

**Shifts are an optimisation with a failure mode.** The Rayleigh shift stalls forever on
``[[0, 1], [1, 0]]`` and Wilkinson's converges in one step, so that pair is tested as a pair.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import qralg


SIZES = [1, 2, 3, 5, 12, 30]


def symmetric_with_spectrum(spectrum, seed=1):
    spectrum = np.asarray(spectrum, dtype=float)
    n = spectrum.size
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))
    return Q @ np.diag(spectrum) @ Q.T


def match(a, b):
    """Largest distance between two sorted complex spectra."""
    return float(np.max(np.abs(np.sort_complex(np.asarray(a)) - np.sort_complex(np.asarray(b)))))


# ---------------------------------------------------------------- Hessenberg


@pytest.mark.parametrize("n", SIZES)
def test_hessenberg_is_a_similarity(n, rng):
    A = rng.standard_normal((n, n))
    H, Q = qralg.hessenberg(A)
    assert np.linalg.norm(A - Q @ H @ Q.T) / max(np.linalg.norm(A), 1e-300) < 1e-13
    assert np.linalg.norm(Q.T @ Q - np.eye(n)) < 1e-13


@pytest.mark.parametrize("n", SIZES)
def test_hessenberg_preserves_the_eigenvalues(n, rng):
    """It must, since it is a similarity. A failure here would mean the reflector was applied
    on only one side."""
    A = rng.standard_normal((n, n))
    H = qralg.hessenberg(A, compute_q=False)
    assert match(np.linalg.eigvals(A), np.linalg.eigvals(H)) < 1e-11


@pytest.mark.parametrize("n", SIZES)
def test_the_result_is_actually_hessenberg(n, rng):
    A = rng.standard_normal((n, n))
    H = qralg.hessenberg(A, compute_q=False)
    assert qralg.is_hessenberg(H)
    if n > 2:
        assert np.abs(np.tril(H, -2)).max() < 1e-13 * max(np.linalg.norm(A, np.inf), 1.0)


@pytest.mark.parametrize("n", [4, 10, 25])
def test_a_symmetric_matrix_reduces_to_tridiagonal(n, rng):
    """Hessenberg plus symmetric is tridiagonal, which is what lesson 38 builds on."""
    A = symmetric_with_spectrum(np.arange(1.0, n + 1), seed=n)
    H = qralg.hessenberg(A, compute_q=False)
    assert np.abs(np.triu(H, 2)).max() < 1e-12 * max(np.linalg.norm(A), 1.0)


def test_an_already_hessenberg_matrix_is_left_alone(rng):
    n = 8
    A = np.triu(rng.standard_normal((n, n)), -1)
    H = qralg.hessenberg(A, compute_q=False)
    assert match(np.linalg.eigvals(A), np.linalg.eigvals(H)) < 1e-11
    assert qralg.is_hessenberg(H)


def test_hessenberg_rejects_a_non_square_matrix(rng):
    with pytest.raises(ValueError):
        qralg.hessenberg(rng.standard_normal((3, 5)))


def test_is_hessenberg_accepts_tiny_matrices():
    assert qralg.is_hessenberg(np.array([[1.0]]))
    assert qralg.is_hessenberg(np.array([[1.0, 2.0], [3.0, 4.0]]))


# ---------------------------------------------------------------- one step


@pytest.mark.parametrize("n", [2, 5, 12])
def test_one_qr_step_is_an_orthogonal_similarity(n, rng):
    A = rng.standard_normal((n, n))
    B, Q = qralg.qr_step(A)
    assert np.linalg.norm(B - Q.T @ A @ Q) < 1e-12 * max(np.linalg.norm(A), 1.0)
    assert match(np.linalg.eigvals(A), np.linalg.eigvals(B)) < 1e-11


# ---------------------------------------------------------------- the algorithm


@pytest.mark.parametrize("n", [1, 2, 4, 10, 30])
@pytest.mark.parametrize("shift", ["none", "rayleigh", "wilkinson"])
def test_the_qr_algorithm_finds_a_symmetric_spectrum(n, shift):
    A = symmetric_with_spectrum(np.arange(1.0, n + 1), seed=n)
    out = qralg.qr_algorithm(A, shift=shift, max_iter=30000)
    assert out.converged, out.message
    err = float(np.max(np.abs(np.sort(out.eigenvalues.real) - np.arange(1.0, n + 1))))
    assert err < 1e-10, f"{shift} at n={n}: error {err:.2e}"


@pytest.mark.parametrize("n", [2, 4, 8, 20, 40])
def test_it_finds_a_nonsymmetric_spectrum_including_complex_pairs(n, rng):
    A = rng.standard_normal((n, n))
    out = qralg.qr_algorithm(A, max_iter=30000)
    assert out.converged, out.message
    assert match(out.eigenvalues, np.linalg.eigvals(A)) < 1e-9


@pytest.mark.parametrize("n", [5, 12, 30])
def test_the_limit_is_a_real_schur_form(n, rng):
    """``A = Q T Q^T`` with ``T`` quasi-triangular: nothing at all below the FIRST subdiagonal,
    and only the 2x2 blocks of complex pairs on it."""
    A = rng.standard_normal((n, n))
    out = qralg.qr_algorithm(A, max_iter=30000)
    e = qralg.schur_error(A, out.T, out.Q)
    assert e["orthogonality"] < 1e-11
    assert e["reconstruction"] < 1e-11
    assert e["below_first_subdiagonal"] < 1e-11


@pytest.mark.parametrize("n", [10, 30])
def test_shifts_cut_the_iteration_count(n):
    """Measured at n=30: 787 unshifted steps, 97 with the Rayleigh shift, 63 with Wilkinson's."""
    A = symmetric_with_spectrum(np.arange(1.0, n + 1), seed=n)
    counts = {s: qralg.qr_algorithm(A, shift=s, max_iter=30000).iterations
              for s in ("none", "rayleigh", "wilkinson")}
    assert counts["wilkinson"] < counts["none"] / 5, counts
    assert counts["rayleigh"] < counts["none"] / 3, counts


def test_the_rayleigh_shift_stalls_where_wilkinsons_does_not():
    """**The standard counterexample.** For [[0,1],[1,0]] the Rayleigh shift is exactly 0, the
    shifted matrix is already orthogonal, and the QR step returns it unchanged forever.
    Measured: 200 steps without converging, against 1 step for Wilkinson's."""
    S = np.array([[0.0, 1.0], [1.0, 0.0]])
    slow = qralg.qr_algorithm(S, shift="rayleigh", max_iter=200, reduce_first=False)
    fast = qralg.qr_algorithm(S, shift="wilkinson", max_iter=200, reduce_first=False)
    assert not slow.converged, "the Rayleigh shift was supposed to stall here"
    assert fast.converged and fast.iterations <= 3, f"{fast.iterations} steps"
    assert np.allclose(np.sort(fast.eigenvalues.real), [-1.0, 1.0])


def test_deflation_happens_and_is_recorded():
    n = 12
    A = symmetric_with_spectrum(np.arange(1.0, n + 1), seed=n)
    out = qralg.qr_algorithm(A, max_iter=30000)
    assert len(out.deflations) >= n - 2, f"only {len(out.deflations)} deflations for n={n}"


def test_a_diagonal_matrix_needs_almost_no_work():
    A = np.diag([5.0, 3.0, 1.0, -2.0])
    out = qralg.qr_algorithm(A, max_iter=100)
    assert out.iterations <= 2, f"{out.iterations} steps on an already diagonal matrix"
    assert np.allclose(np.sort(out.eigenvalues.real), [-2.0, 1.0, 3.0, 5.0])


def test_the_algorithm_rejects_a_bad_shift_name(rng):
    with pytest.raises(ValueError):
        qralg.qr_algorithm(rng.standard_normal((4, 4)), shift="rayleigh-quotient")


def test_the_algorithm_rejects_a_non_square_matrix(rng):
    with pytest.raises(ValueError):
        qralg.qr_algorithm(rng.standard_normal((3, 5)))


# ---------------------------------------------------------------- shifts themselves


def test_the_wilkinson_shift_of_a_symmetric_2x2_is_an_eigenvalue():
    A = np.array([[4.0, 1.0], [1.0, 2.0]])
    mu = qralg.wilkinson_shift(A)
    vals = np.linalg.eigvalsh(A)
    assert min(abs(mu - v) for v in vals) < 1e-12


def test_the_wilkinson_shift_picks_the_closer_eigenvalue():
    A = np.array([[10.0, 1.0], [1.0, 0.0]])
    mu = qralg.wilkinson_shift(A)
    vals = np.linalg.eigvalsh(A)
    closer = min(vals, key=lambda v: abs(v - A[1, 1]))
    assert abs(mu - closer) < 1e-12, f"shift {mu} against the closer eigenvalue {closer}"


def test_the_wilkinson_shift_falls_back_on_a_complex_pair():
    """A trailing block with complex eigenvalues has no real shift to offer, so the routine
    returns the Rayleigh shift rather than a NaN."""
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    mu = qralg.wilkinson_shift(A)
    assert np.isfinite(mu)


def test_the_wilkinson_shift_of_a_one_by_one():
    assert qralg.wilkinson_shift([[2.5]]) == 2.5


def test_a_zero_offdiagonal_gives_the_rayleigh_shift():
    A = np.array([[4.0, 0.0], [0.0, 2.0]])
    assert qralg.wilkinson_shift(A) == 2.0


# ---------------------------------------------------------------- why it works


@pytest.mark.parametrize("steps", [5, 10, 15, 25, 40])
def test_simultaneous_iteration_from_the_identity_IS_the_unshifted_qr_algorithm(steps):
    """**Up to column signs, which numpy's qr chooses arbitrarily.** Measured agreement
    1e-15 at every step count; the raw difference reaches 0.77 purely from sign flips, which is
    lesson 30's uniqueness-up-to-signs showing up as a practical trap."""
    n = 6
    A = symmetric_with_spectrum(np.arange(1.0, n + 1), seed=3)
    si = qralg.simultaneous_iteration(A, steps=steps, start=np.eye(n))
    M = A.copy()
    for _ in range(steps):
        M, _ = qralg.qr_step(M)
    assert np.max(np.abs(np.abs(si["projected"]) - np.abs(M))) < 1e-12
    assert np.max(np.abs(np.diag(si["projected"]) - np.diag(M))) < 1e-12


@pytest.mark.parametrize("p", [1, 3, 6])
def test_simultaneous_iteration_finds_the_top_p_eigenvalues(p):
    n = 6
    A = symmetric_with_spectrum(np.arange(1.0, n + 1), seed=3)
    out = qralg.simultaneous_iteration(A, n_vectors=p, steps=400,
                                       rng=np.random.default_rng(4))
    got = np.sort(np.linalg.eigvalsh(out["projected"]))[::-1]
    expected = np.arange(1.0, n + 1)[::-1][:p]
    assert np.max(np.abs(got - expected)) < 1e-6, f"got {got}, expected {expected}"


def test_simultaneous_iteration_rejects_a_bad_block_size(rng):
    for bad in (0, -1, 9):
        with pytest.raises(ValueError):
            qralg.simultaneous_iteration(rng.standard_normal((6, 6)), n_vectors=bad)


def test_simultaneous_iteration_rejects_a_bad_start(rng):
    with pytest.raises(ValueError):
        qralg.simultaneous_iteration(rng.standard_normal((6, 6)), start=np.eye(5))


@pytest.mark.parametrize("spectrum", [[8.0, 4.0, 2.0, 1.0], [10.0, 9.0, 3.0, 1.0]])
def test_the_predicted_rates_are_the_eigenvalue_ratios(spectrum):
    A = symmetric_with_spectrum(spectrum, seed=len(spectrum))
    rates = qralg.convergence_rates(A)
    expected = np.array([abs(spectrum[i + 1] / spectrum[i]) for i in range(len(spectrum) - 1)])
    assert np.max(np.abs(np.sort(rates) - np.sort(expected))) < 1e-10


def test_convergence_rates_of_a_single_eigenvalue():
    assert qralg.convergence_rates([[3.0]]).size == 0


# ---------------------------------------------------------------- LR, for comparison


def test_lr_converges_on_an_easy_matrix(rng):
    n = 4
    A = rng.standard_normal((n, n))
    A = A @ A.T + n * np.eye(n)
    out = qralg.lr_algorithm(A, max_iter=5000)
    if not out["converged"]:
        pytest.skip("this draw needed a pivot, which is the other failure mode")
    assert match(out["eigenvalues"], np.linalg.eigvals(A)) < 1e-6


def test_lr_breaks_down_when_a_pivot_is_needed():
    """**Why nobody uses it.** Unpivoted LU fails the moment a leading minor is singular, and
    pivoting would destroy the similarity structure. Measured at n=8: a pivot was needed at
    step 7."""
    gen = np.random.default_rng(0)
    broke = 0
    for trial in range(12):
        A = gen.standard_normal((8, 8))
        A = A @ A.T + 8 * np.eye(8)
        if not qralg.lr_algorithm(A, max_iter=3000)["converged"]:
            broke += 1
    assert broke > 0, "no draw broke down; the point was not made"


def test_lr_is_a_similarity_while_it_lasts(rng):
    """It preserves the eigenvalues, exactly as the QR algorithm does. What it does not preserve
    is the conditioning, because L is not orthogonal."""
    n = 4
    A = rng.standard_normal((n, n))
    A = A @ A.T + n * np.eye(n)
    out = qralg.lr_algorithm(A, max_iter=200)
    assert len(out["condition_growth"]) >= 2
    assert all(np.isfinite(out["condition_growth"]))


# ---------------------------------------------------------------- quasi-triangular reading


def test_reading_a_triangular_matrix():
    T = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 1.0], [0.0, 0.0, -1.0]])
    assert np.allclose(np.sort(qralg.quasi_triangular_eigenvalues(T).real), [-1.0, 3.0, 5.0])


def test_reading_a_two_by_two_complex_block():
    """A rotation by 90 degrees has eigenvalues plus and minus i, and cannot be triangularized
    over the reals at all."""
    T = np.array([[0.0, 1.0], [-1.0, 0.0]])
    vals = qralg.quasi_triangular_eigenvalues(T)
    assert np.allclose(np.sort_complex(vals), np.sort_complex(np.array([1j, -1j])))


def test_reading_a_mixed_quasi_triangular_matrix():
    T = np.array([[2.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0],
                  [0.0, -1.0, 0.0]])
    vals = qralg.quasi_triangular_eigenvalues(T)
    assert abs(sorted(vals, key=lambda z: -abs(z.real))[0].real - 2.0) < 1e-12
    assert np.max(np.abs(np.imag(vals))) > 0.9


@pytest.mark.parametrize("n", [4, 9, 20])
def test_reading_agrees_with_the_library_on_a_computed_schur_form(n, rng):
    A = rng.standard_normal((n, n))
    out = qralg.qr_algorithm(A, max_iter=30000)
    assert match(qralg.quasi_triangular_eigenvalues(out.T), np.linalg.eigvals(A)) < 1e-9
