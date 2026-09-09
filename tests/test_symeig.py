"""Tests for `nalib.symeig`.

The symmetric problem is the one where accuracy claims can be checked properly, so these tests
check them properly. In particular the relative accuracy comparison uses an **80 digit mpmath
reference**, not another double precision eigensolver: an earlier version of `graded_symmetric`
compared LAPACK against LAPACK and scored it a perfect zero.

Two bugs found while writing these are pinned here as tests: `off_norm` computed by subtraction
returns exactly zero while the off-diagonal is still there, and a bare ``z ** 2`` in the secular
function broadcasts against the wrong axis.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import symeig


SIZES = [1, 2, 3, 5, 12, 30]


def sym(n, seed):
    A = np.random.default_rng(seed).standard_normal((n, n))
    return A + A.T


def with_spectrum(spectrum, seed=1):
    spectrum = np.asarray(spectrum, dtype=float)
    n = spectrum.size
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))
    return Q @ np.diag(spectrum) @ Q.T


# ---------------------------------------------------------------- off_norm and guards


def test_off_norm_of_a_diagonal_matrix_is_zero():
    assert symeig.off_norm(np.diag([3.0, -1.0, 7.0])) == 0.0


def test_off_norm_survives_the_cancellation_that_broke_it():
    """**The bug this pins.** ``sqrt(||A||^2 - ||diag||^2)`` returns exactly 0.0 once the
    off-diagonal is at the roundoff level, because the two squared norms agree to their last
    bit. Fed to a convergence test that stops Jacobi several sweeps early, leaving the
    eigenvectors wrong by 3e-9 while the eigenvalues still look perfect."""
    n = 6
    A = np.diag(np.arange(1.0, n + 1) * 1e3)
    A[0, 1] = A[1, 0] = 1e-9
    naive = np.sqrt(max(np.linalg.norm(A) ** 2 - np.linalg.norm(np.diag(A)) ** 2, 0.0))
    honest = symeig.off_norm(A)
    assert naive == 0.0, "the cancellation did not trigger; the test proves nothing"
    assert abs(honest - np.sqrt(2.0) * 1e-9) < 1e-20, f"got {honest:.3e}"


@pytest.mark.parametrize("n", SIZES)
def test_require_symmetric_accepts_symmetric(n):
    A = sym(n, n)
    assert symeig.require_symmetric(A).shape == (n, n)


def test_require_symmetric_refuses_asymmetric(rng):
    with pytest.raises(ValueError, match="not symmetric"):
        symeig.require_symmetric(rng.standard_normal((5, 5)))


def test_require_symmetric_refuses_non_square(rng):
    with pytest.raises(ValueError):
        symeig.require_symmetric(rng.standard_normal((3, 5)))


# ---------------------------------------------------------------- Jacobi


@pytest.mark.parametrize("n", SIZES)
def test_jacobi_finds_the_eigenvalues(n):
    A = sym(n, n)
    out = symeig.jacobi_eigen(A)
    assert out.converged, out.message
    assert np.max(np.abs(out.values - np.sort(np.linalg.eigvalsh(A)))) < 1e-11


@pytest.mark.parametrize("n", [2, 5, 12, 30])
def test_jacobi_produces_real_eigenvectors(n):
    """``A V = V D`` and ``V`` orthogonal. Checking only the eigenvalues would have missed the
    `off_norm` bug entirely, because the eigenvalues stayed accurate while the vectors did not."""
    A = sym(n, n)
    out = symeig.jacobi_eigen(A)
    V, w = out.vectors, out.values
    assert np.linalg.norm(V.T @ V - np.eye(n)) < 1e-12
    assert np.linalg.norm(A @ V - V @ np.diag(w)) < 1e-11 * max(np.linalg.norm(A), 1.0)
    assert np.linalg.norm(A - V @ np.diag(w) @ V.T) < 1e-11 * max(np.linalg.norm(A), 1.0)


@pytest.mark.parametrize("n", [8, 20, 40])
def test_the_off_diagonal_norm_decreases_every_sweep(n):
    A = sym(n, n)
    trail = symeig.jacobi_eigen(A, compute_vectors=False).off_diagonal
    assert all(b < a for a, b in zip(trail, trail[1:])), trail


@pytest.mark.parametrize("n", [8, 20, 40])
def test_jacobi_converges_quadratically(n):
    """Measured at n=40: 5.7e1, 3.1e1, 1.4e1, 4.9, 1.0, 3.5e-2, 2.7e-5, 4.4e-12. Each of the
    last three sweeps roughly squares the one before."""
    A = sym(n, n)
    trail = np.array(symeig.jacobi_eigen(A, compute_vectors=False).off_diagonal)
    usable = trail[(trail > 1e-14) & (trail < 1.0)]
    assert usable.size >= 2, f"not enough small sweeps to see a rate: {trail}"
    ratios = np.log10(usable[1:]) / np.log10(usable[:-1])
    assert float(np.max(ratios)) > 1.7, f"never squared: log ratios {ratios}"


@pytest.mark.parametrize("n", [4, 8, 15])
def test_the_greedy_ordering_needs_fewer_rotations(n):
    """Picking the largest entry each time uses fewer rotations than a fixed sweep, and spends
    more time looking for them. Both are worth knowing."""
    A = sym(n, n)
    cyclic = symeig.jacobi_eigen(A, compute_vectors=False, cyclic=True)
    greedy = symeig.jacobi_eigen(A, compute_vectors=False, cyclic=False, max_sweeps=200)
    assert greedy.converged, greedy.message
    assert np.max(np.abs(greedy.values - cyclic.values)) < 1e-10


def test_a_diagonal_matrix_needs_no_rotations():
    out = symeig.jacobi_eigen(np.diag([4.0, 1.0, -2.0]))
    assert out.rotations == 0
    assert np.allclose(out.values, [-2.0, 1.0, 4.0])


def test_jacobi_on_a_one_by_one():
    out = symeig.jacobi_eigen([[3.5]])
    assert out.converged and np.allclose(out.values, [3.5])


def test_jacobi_refuses_an_asymmetric_matrix(rng):
    with pytest.raises(ValueError, match="not symmetric"):
        symeig.jacobi_eigen(rng.standard_normal((4, 4)))


# ---------------------------------------------------------------- the rotation itself


@pytest.mark.parametrize("a,b,c_", [(1.0, 2.0, 3.0), (5.0, -1.0, 5.0), (0.0, 1.0, 0.0),
                                    (1e8, 1.0, -1e8)])
def test_the_rotation_zeroes_the_entry(a, b, c_):
    c, s = symeig.jacobi_rotation(a, b, c_)
    J = np.array([[c, s], [-s, c]])
    B = np.array([[a, b], [b, c_]])
    assert abs(c * c + s * s - 1.0) < 1e-15
    assert abs((J.T @ B @ J)[0, 1]) < 1e-12 * max(abs(a), abs(b), abs(c_), 1.0)


def test_the_rotation_takes_the_smaller_angle():
    """Below 45 degrees, so the diagonal entries keep their order and the method converges
    rather than shuffling."""
    c, s = symeig.jacobi_rotation(1.0, 2.0, 3.0)
    assert abs(s / c) <= 1.0 + 1e-12, f"tan = {s / c}"


def test_the_rotation_avoids_cancellation_for_a_large_theta():
    """``-theta + sqrt(theta^2 + 1)`` is algebraically the same and loses every digit here."""
    a, b, c_ = -1e8, 1.0, 1e8
    theta = (c_ - a) / (2.0 * b)
    naive = -theta + np.sqrt(theta * theta + 1.0)
    c, s = symeig.jacobi_rotation(a, b, c_)
    stable = s / c
    assert abs(naive - stable) / abs(stable) > 1e-3, \
        f"the naive form was not damaged enough: {naive:.6e} against {stable:.6e}"
    B = np.array([[a, b], [b, c_]])
    J = np.array([[c, s], [-s, c]])
    assert abs((J.T @ B @ J)[0, 1]) < 1e-6


def test_a_zero_offdiagonal_gives_the_identity_rotation():
    assert symeig.jacobi_rotation(3.0, 0.0, 5.0) == (1.0, 0.0)


# ---------------------------------------------------------------- tridiagonalization


@pytest.mark.parametrize("n", SIZES)
def test_tridiagonalize_is_a_similarity(n):
    A = sym(n, n)
    d, e, Q = symeig.tridiagonalize(A)
    T = symeig.tridiagonal_matrix(d, e)
    assert np.linalg.norm(Q.T @ Q - np.eye(n)) < 1e-12
    assert np.linalg.norm(A - Q @ T @ Q.T) < 1e-11 * max(np.linalg.norm(A), 1.0)


@pytest.mark.parametrize("n", SIZES)
def test_tridiagonalize_preserves_the_eigenvalues(n):
    A = sym(n, n)
    d, e = symeig.tridiagonalize(A, compute_q=False)
    T = symeig.tridiagonal_matrix(d, e)
    assert np.max(np.abs(np.sort(np.linalg.eigvalsh(A))
                         - np.sort(np.linalg.eigvalsh(T)))) < 1e-11


@pytest.mark.parametrize("n", [4, 12, 30])
def test_the_result_is_genuinely_tridiagonal(n):
    A = sym(n, n)
    d, e = symeig.tridiagonalize(A, compute_q=False)
    assert d.size == n and e.size == n - 1


def test_tridiagonal_matrix_rejects_a_mismatched_offdiagonal():
    with pytest.raises(ValueError):
        symeig.tridiagonal_matrix([1.0, 2.0, 3.0], [1.0])


def test_tridiagonalize_refuses_an_asymmetric_matrix(rng):
    with pytest.raises(ValueError, match="not symmetric"):
        symeig.tridiagonalize(rng.standard_normal((5, 5)))


# ---------------------------------------------------------------- Sturm


@pytest.mark.parametrize("n", [1, 2, 5, 20, 60])
def test_the_sturm_count_is_right_everywhere(n):
    A = sym(n, n)
    d, e = symeig.tridiagonalize(A, compute_q=False)
    vals = np.sort(np.linalg.eigvalsh(A))
    lo, hi = symeig.gerschgorin_interval(d, e)
    for x in np.linspace(lo, hi, 25):
        assert symeig.sturm_count(d, e, x) == int(np.sum(vals < x)), f"at x={x}"


@pytest.mark.parametrize("n", [5, 20, 60])
def test_the_count_is_zero_below_and_n_above(n):
    A = sym(n, n)
    d, e = symeig.tridiagonalize(A, compute_q=False)
    lo, hi = symeig.gerschgorin_interval(d, e)
    assert symeig.sturm_count(d, e, lo - 1.0) == 0
    assert symeig.sturm_count(d, e, hi + 1.0) == n


@pytest.mark.parametrize("n", [5, 20, 50])
def test_bisection_finds_every_eigenvalue(n):
    A = sym(n, n)
    d, e = symeig.tridiagonalize(A, compute_q=False)
    got = symeig.bisection_eigenvalues(d, e)
    assert np.max(np.abs(got - np.sort(np.linalg.eigvalsh(A)))) < 1e-10


@pytest.mark.parametrize("k", [0, 3, 9, 19])
def test_one_eigenvalue_can_be_found_without_the_others(k):
    """**The property nothing else here has.** Eigenvalue 19 of 20 costs exactly one bisection,
    the same as eigenvalue 0."""
    n = 20
    A = sym(n, n)
    d, e = symeig.tridiagonalize(A, compute_q=False)
    exact = np.sort(np.linalg.eigvalsh(A))
    assert abs(symeig.bisect_eigenvalue(d, e, k) - exact[k]) < 1e-10


def test_counting_eigenvalues_in_an_interval():
    n = 15
    A = sym(n, n)
    d, e = symeig.tridiagonalize(A, compute_q=False)
    vals = np.sort(np.linalg.eigvalsh(A))
    first, last = n // 5, 3 * n // 5          # derived from n, not written in
    lo, hi = float(vals[first]) - 1e-9, float(vals[last]) + 1e-9
    assert symeig.eigenvalues_in(d, e, lo, hi) == last - first + 1


def test_eigenvalues_in_rejects_a_backwards_interval():
    with pytest.raises(ValueError):
        symeig.eigenvalues_in([1.0, 2.0], [0.5], 5.0, 1.0)


def test_bisect_rejects_a_bad_index():
    for bad in (-1, 3):
        with pytest.raises(ValueError):
            symeig.bisect_eigenvalue([1.0, 2.0, 3.0], [0.5, 0.5], bad)


def test_sturm_rejects_a_mismatched_offdiagonal():
    with pytest.raises(ValueError):
        symeig.sturm_count([1.0, 2.0, 3.0], [1.0], 0.0)


def test_sturm_survives_a_zero_pivot():
    """``d_{k-1} = 0`` would divide by zero. The tiny-substitution keeps the count right.

    The matrix here has eigenvalues -sqrt(2), 0, +sqrt(2) exactly, so x = 0 is skipped: the
    count of eigenvalues STRICTLY below 0 is 1, and a reference built from a computed
    eigenvalue of -8e-17 says 2. The Sturm count is right and the reference is not, which is
    itself the reason this routine exists."""
    d = np.array([0.0, 0.0, 0.0])
    e = np.array([1.0, 1.0])
    for x, expected in ((-2.0, 0), (-0.5, 1), (0.5, 2), (2.0, 3)):
        assert symeig.sturm_count(d, e, x) == expected, f"at x={x}"
    # And exactly on the eigenvalue, the count is of those STRICTLY below it.
    assert symeig.sturm_count(d, e, 0.0) == 1


# ---------------------------------------------------------------- divide and conquer


@pytest.mark.parametrize("n", [1, 2, 4, 10, 25, 60, 120])
def test_divide_and_conquer_finds_the_eigenvalues(n):
    A = sym(n, n)
    d, e = symeig.tridiagonalize(A, compute_q=False)
    got = symeig.divide_and_conquer(d, e)
    assert np.max(np.abs(got - np.sort(np.linalg.eigvalsh(A)))) < 1e-10


@pytest.mark.parametrize("cutoff", [2, 4, 16])
def test_the_cutoff_does_not_change_the_answer(cutoff):
    n = 40
    A = sym(n, n)
    d, e = symeig.tridiagonalize(A, compute_q=False)
    got = symeig.divide_and_conquer(d, e, cutoff=cutoff)
    assert np.max(np.abs(got - np.sort(np.linalg.eigvalsh(A)))) < 1e-10


def test_the_secular_function_vanishes_at_the_eigenvalues():
    """**The bug this pins.** A bare ``z ** 2`` broadcasts against the trailing axis of the
    ``(n, k)`` denominator, silently building an ``(n, n)`` array. The function then returned a
    plausible finite number that was nonzero at every true eigenvalue."""
    gen = np.random.default_rng(3)
    n = 7
    d = np.sort(gen.standard_normal(n))
    z = gen.standard_normal(n)
    rho = 1.4
    exact = np.sort(np.linalg.eigvalsh(np.diag(d) + rho * np.outer(z, z)))
    for lam in exact:
        assert abs(float(symeig.secular_function(d, z, rho, lam)[0])) < 1e-6, \
            f"f({lam}) was not zero"


@pytest.mark.parametrize("rho", [1.4, -1.4, 0.3])
def test_solving_the_secular_equation(rho):
    gen = np.random.default_rng(5)
    n = 6
    d = np.sort(gen.standard_normal(n))
    z = gen.standard_normal(n)
    got = symeig.solve_secular(d, z, rho)
    exact = np.sort(np.linalg.eigvalsh(np.diag(d) + rho * np.outer(z, z)))
    assert np.max(np.abs(got - exact)) < 1e-8, f"got {got}, exact {exact}"


def test_the_roots_interlace_the_poles():
    """Every root lies strictly between consecutive poles, which is what makes a bracketing
    method unable to fail."""
    gen = np.random.default_rng(7)
    n = 8
    d = np.sort(gen.standard_normal(n))
    z = gen.standard_normal(n)
    roots = symeig.solve_secular(d, z, 1.0)
    for i in range(n - 1):
        assert d[i] - 1e-9 <= roots[i] <= d[i + 1] + 1e-9, \
            f"root {i} = {roots[i]} escaped ({d[i]}, {d[i + 1]})"


def test_secular_function_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        symeig.secular_function([1.0, 2.0], [1.0], 1.0, 0.0)


def test_divide_and_conquer_rejects_a_mismatched_offdiagonal():
    with pytest.raises(ValueError):
        symeig.divide_and_conquer([1.0, 2.0, 3.0], [1.0])


# ---------------------------------------------------------------- relative accuracy


@pytest.mark.parametrize("n,spread", [(6, 1e-4), (10, 1e-8)])
def test_the_reference_is_not_the_thing_being_tested(n, spread):
    """**The bug this pins.** The old reference was
    ``eigvalsh(longdouble(A).astype(float))``, which is ``eigvalsh(A)``: the round trip changes
    nothing. LAPACK was then measured against LAPACK and scored exactly zero."""
    g = symeig.graded_symmetric(n, spread, rng=np.random.default_rng(2))
    lapack = np.sort(np.linalg.eigvalsh(g["A"]))
    assert symeig.relative_accuracy(lapack, g["exact"]) > 0.0, \
        "the reference agreed with LAPACK exactly, so it IS LAPACK"


@pytest.mark.parametrize("spread", [1e-4, 1e-6, 1e-8, 1e-10])
def test_jacobi_relative_accuracy_is_flat_in_the_condition_number(spread):
    """**The real result, and it is the flatness rather than the size of the gap.** Measured
    across condition numbers from 1e8 to 1e20, Jacobi's relative error stays at about 1.5e-15
    while LAPACK's drifts from 1.2e-14 to 2.3e-14."""
    n = 10
    g = symeig.graded_symmetric(n, spread, rng=np.random.default_rng(2))
    jac = symeig.jacobi_eigen(g["A"], compute_vectors=False, max_sweeps=100).values
    assert symeig.relative_accuracy(jac, g["exact"]) < 1e-13, \
        f"spread {spread}: {symeig.relative_accuracy(jac, g['exact']):.2e}"


@pytest.mark.parametrize("spread", [1e-6, 1e-8, 1e-10])
def test_jacobi_beats_lapack_relatively_on_a_graded_matrix(spread):
    """By about 12 to 15 times, not by orders of magnitude. The folklore overstates this."""
    n = 10
    g = symeig.graded_symmetric(n, spread, rng=np.random.default_rng(2))
    jac = symeig.jacobi_eigen(g["A"], compute_vectors=False, max_sweeps=100).values
    lapack = np.sort(np.linalg.eigvalsh(g["A"]))
    r_j = symeig.relative_accuracy(jac, g["exact"])
    r_l = symeig.relative_accuracy(lapack, g["exact"])
    assert r_l > 3.0 * r_j, f"Jacobi {r_j:.2e} against LAPACK {r_l:.2e}"


def test_jacobi_does_NOT_win_at_a_low_condition_number():
    """At kappa 1e4 it is slightly worse. The advantage is specific to graded matrices with a
    large spread, and claiming it in general would be wrong."""
    g = symeig.graded_symmetric(10, 1e-2, rng=np.random.default_rng(2))
    jac = symeig.jacobi_eigen(g["A"], compute_vectors=False, max_sweeps=100).values
    lapack = np.sort(np.linalg.eigvalsh(g["A"]))
    assert (symeig.relative_accuracy(jac, g["exact"])
            >= 0.5 * symeig.relative_accuracy(lapack, g["exact"]))


def test_bisection_loses_relative_accuracy_worst_of_all():
    """Measured at kappa 1e20: 2.7e-10 against Jacobi's 1.6e-15, a factor of 160000. The
    tridiagonal reduction destroys the grading that Jacobi works directly with."""
    g = symeig.graded_symmetric(10, 1e-10, rng=np.random.default_rng(2))
    d, e = symeig.tridiagonalize(g["A"], compute_q=False)
    bis = symeig.bisection_eigenvalues(d, e)
    jac = symeig.jacobi_eigen(g["A"], compute_vectors=False, max_sweeps=100).values
    assert (symeig.relative_accuracy(bis, g["exact"])
            > 100.0 * symeig.relative_accuracy(jac, g["exact"]))


def test_relative_accuracy_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        symeig.relative_accuracy([1.0, 2.0], [1.0, 2.0, 3.0])


def test_graded_symmetric_rejects_a_bad_size():
    with pytest.raises(ValueError):
        symeig.graded_symmetric(0, 1e-3)


@pytest.mark.parametrize("n,spread", [(4, 1e-3), (8, 1e-6)])
def test_the_graded_base_is_well_conditioned(n, spread):
    """The whole construction only means something if ``A = D B D`` has a well conditioned
    ``B``: that is the hypothesis of the Demmel-Veselic theorem."""
    g = symeig.graded_symmetric(n, spread, rng=np.random.default_rng(2))
    assert np.linalg.cond(g["base"]) < 20.0
    assert np.linalg.cond(g["A"]) > 100.0 * np.linalg.cond(g["base"])


# ---------------------------------------------------------------- the methods agree


@pytest.mark.parametrize("n", [6, 15, 40])
def test_all_four_methods_agree(n):
    A = sym(n, n)
    d, e = symeig.tridiagonalize(A, compute_q=False)
    answers = {"jacobi": symeig.jacobi_eigen(A, compute_vectors=False).values,
               "bisection": symeig.bisection_eigenvalues(d, e),
               "divide and conquer": symeig.divide_and_conquer(d, e),
               "lapack": np.sort(np.linalg.eigvalsh(A))}
    reference = answers["lapack"]
    for name, got in answers.items():
        assert np.max(np.abs(got - reference)) < 1e-10, f"{name} disagreed"
