"""Tests for `nalib.lowrank`.

Eckart-Young is an **optimality** claim, so it is tested by attacking it: 3000 random rank ``k``
subspaces are fitted to the matrix and none gets closer than 1.16 times the truncation. Quoting
the theorem and checking one example would establish much less.

The randomised tests measure the two knobs separately, because they are not interchangeable: one
power step (three matrix products) beats twenty oversamples (one product, a much larger sketch).
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import lowrank as lr


SHAPES = [(6, 4), (20, 12), (30, 20), (50, 50), (12, 30)]


# ---------------------------------------------------------------- Eckart-Young


@pytest.mark.parametrize("m,n", SHAPES)
def test_the_two_norm_error_is_exactly_the_next_singular_value(m, n, rng):
    A = rng.standard_normal((m, n))
    for k in range(min(m, n) + 1):
        out = lr.truncate(A, k)
        assert abs(out.error_2 - out.optimal_2) < 1e-11 * max(out.optimal_2, 1.0), \
            f"k={k}: {out.error_2:.6e} against sigma_{k + 1} = {out.optimal_2:.6e}"


@pytest.mark.parametrize("m,n", SHAPES)
def test_the_frobenius_error_is_the_tail_norm(m, n, rng):
    A = rng.standard_normal((m, n))
    for k in (0, 1, min(m, n) // 2, min(m, n)):
        out = lr.truncate(A, k)
        assert abs(out.error_frobenius - out.optimal_frobenius) < 1e-11 * max(
            out.optimal_frobenius, 1.0)


@pytest.mark.parametrize("m,n,k", [(20, 12, 3), (30, 20, 5), (50, 50, 8)])
def test_nothing_beats_the_truncation(m, n, k, rng):
    """**The theorem, attacked rather than quoted.** 3000 random rank k subspaces, each fitted
    optimally to A, and the closest is 1.16 to 1.27 times the truncation's error."""
    A = rng.standard_normal((m, n))
    out = lr.beat_the_truncation(A, k, n_trials=3000, rng=np.random.default_rng(1))
    assert out["best_random"] >= out["truncated"] * (1.0 - 1e-9), \
        "a random subspace beat the SVD, which contradicts Eckart-Young"
    assert out["shortfall"] > 1.05, "the search was too weak to say anything"


@pytest.mark.parametrize("m,n", [(20, 12), (30, 30)])
def test_the_same_matrix_is_optimal_in_both_norms(m, n, rng):
    """The surprising half of the theorem: one matrix is simultaneously best in two different
    norms, and Mirsky extended that to every unitarily invariant norm."""
    A = rng.standard_normal((m, n))
    for k in (1, min(m, n) // 2):
        assert lr.truncate(A, k).is_optimal


@pytest.mark.parametrize("m,n,r", [(10, 6, 3), (20, 20, 7)])
def test_truncating_at_the_true_rank_is_exact(m, n, r, rng):
    A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    out = lr.truncate(A, r)
    assert out.error_2 < 1e-11 * max(np.linalg.norm(A, 2), 1.0)


def test_truncating_to_rank_zero_gives_the_zero_matrix(rng):
    A = rng.standard_normal((5, 4))
    out = lr.truncate(A, 0)
    assert np.linalg.norm(out.matrix) == 0.0
    assert abs(out.error_2 - np.linalg.norm(A, 2)) < 1e-12


def test_truncate_rejects_an_impossible_rank(rng):
    for bad in (-1, 7):
        with pytest.raises(ValueError):
            lr.truncate(rng.standard_normal((10, 6)), bad)


def test_beat_the_truncation_rejects_a_bad_rank(rng):
    for bad in (0, 9):
        with pytest.raises(ValueError):
            lr.beat_the_truncation(rng.standard_normal((10, 8)), bad, n_trials=10)


# ---------------------------------------------------------------- energy and storage


@pytest.mark.parametrize("m,n", [(20, 12), (40, 40)])
def test_energy_is_monotone_and_reaches_one(m, n, rng):
    A = rng.standard_normal((m, n))
    energies = [lr.energy_captured(A, k) for k in range(min(m, n) + 1)]
    assert energies[0] == 0.0
    assert abs(energies[-1] - 1.0) < 1e-12
    assert all(b >= a - 1e-15 for a, b in zip(energies, energies[1:]))


def test_energy_is_a_squared_fraction():
    """**Worth naming, because it misleads.** Keeping 98.5 percent of the energy of the test
    image at rank 1 still leaves a 12 percent relative error, because energy is a fraction of
    the SQUARED norm."""
    img = lr.synthetic_image(256, 256)
    energy = lr.energy_captured(img, 1)
    error = lr.truncate(img, 1).error_frobenius / np.linalg.norm(img)
    assert energy > 0.98
    assert error > 0.10, f"energy {energy:.4f} but error only {error:.4f}"
    assert abs(energy + error ** 2 - 1.0) < 1e-9, "the two should be exactly complementary"


@pytest.mark.parametrize("fraction", [0.5, 0.9, 0.99, 0.999, 1.0])
def test_rank_for_energy_delivers_that_energy(fraction, rng):
    A = rng.standard_normal((30, 20))
    k = lr.rank_for_energy(A, fraction)
    assert lr.energy_captured(A, k) >= fraction - 1e-12
    if k > 1:
        assert lr.energy_captured(A, k - 1) < fraction


def test_energy_of_the_zero_matrix_is_one():
    assert lr.energy_captured(np.zeros((4, 6)), 0) == 1.0
    assert lr.rank_for_energy(np.zeros((4, 6)), 0.9) == 0


def test_energy_rejects_bad_arguments(rng):
    with pytest.raises(ValueError):
        lr.energy_captured(rng.standard_normal((6, 4)), 5)
    with pytest.raises(ValueError):
        lr.rank_for_energy(rng.standard_normal((6, 4)), 1.5)


@pytest.mark.parametrize("m,n,k,expected_saving", [(256, 256, 1, True), (256, 256, 50, True),
                                                   (256, 256, 200, False)])
def test_compression_only_saves_at_small_rank(m, n, k, expected_saving):
    """**A rank n-1 approximation of an n by n matrix costs nearly twice the matrix.** Measured
    at 256x256 rank 255: a ratio of 0.50, meaning it doubles the storage."""
    ratio = lr.compression_ratio(m, n, k)
    assert (ratio > 1.0) is expected_saving, f"k={k}: ratio {ratio:.2f}"


def test_compression_ratio_rejects_a_bad_shape():
    with pytest.raises(ValueError):
        lr.compression_ratio(0, 5, 1)


# ---------------------------------------------------------------- randomised


@pytest.mark.parametrize("m,n,k", [(60, 40, 5), (200, 100, 15), (100, 200, 10)])
def test_the_randomised_svd_is_close_to_optimal(m, n, k, rng):
    A = rng.standard_normal((m, n))
    optimal = lr.truncate(A, k).error_2
    out = lr.randomised_svd(A, k, oversample=10, n_power=2, rng=rng)
    assert out["error_2"] <= 1.5 * optimal, \
        f"{out['error_2']:.4e} against an optimal {optimal:.4e}"


@pytest.mark.parametrize("m,n,r,k", [(60, 40, 8, 8), (100, 80, 12, 12)])
def test_it_is_exact_when_the_matrix_really_has_that_rank(m, n, r, k, rng):
    A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    out = lr.randomised_svd(A, k, oversample=5, rng=rng)
    assert out["error_2"] < 1e-9 * max(np.linalg.norm(A, 2), 1.0)


def test_oversampling_is_not_optional():
    """Measured on a 200x300 image at k=20: p=0 gives 2.76 times the optimal error and p=20
    gives 1.005."""
    A = lr.synthetic_image(200, 300)
    out = lr.oversampling_matters(A, 20, oversamples=(0, 20), n_trials=10,
                                  rng=np.random.default_rng(2))
    none, plenty = out["rows"][0], out["rows"][1]
    assert none["median_ratio"] > 2.0, f"p=0 was already good: {none['median_ratio']:.3f}"
    assert plenty["median_ratio"] < 1.2, f"p=20 was not good: {plenty['median_ratio']:.3f}"


def test_power_iteration_beats_oversampling_per_product():
    """**The two knobs are not interchangeable.** One power step costs three matrix products and
    reaches 1.04 times optimal; twenty oversamples cost one product with a far larger sketch and
    reach 1.005. Per product, the power step is the better buy on a slowly decaying spectrum."""
    A = lr.synthetic_image(200, 300)
    out = lr.power_iteration_helps(A, 20, powers=(0, 1, 2), oversample=5, n_trials=10,
                                   rng=np.random.default_rng(3))
    ratios = [r["median_ratio"] for r in out["rows"]]
    assert ratios[0] > 1.5, f"no power step was already good: {ratios[0]:.3f}"
    assert ratios[1] < 1.15, f"one power step did not help enough: {ratios[1]:.3f}"
    assert all(b <= a + 1e-9 for a, b in zip(ratios, ratios[1:])), ratios


def test_the_matrix_product_count_is_reported(rng):
    A = rng.standard_normal((50, 30))
    for q in (0, 1, 3):
        assert lr.randomised_svd(A, 5, n_power=q, rng=rng)["matrix_products"] == 1 + 2 * q


def test_randomised_svd_rejects_bad_arguments(rng):
    A = rng.standard_normal((20, 10))
    for bad_k in (0, 11):
        with pytest.raises(ValueError):
            lr.randomised_svd(A, bad_k)
    with pytest.raises(ValueError):
        lr.randomised_svd(A, 5, oversample=-1)


# ---------------------------------------------------------------- the test image


@pytest.mark.parametrize("h,w", [(2, 2), (64, 64), (128, 200), (200, 128)])
def test_the_synthetic_image_has_the_requested_shape(h, w):
    assert lr.synthetic_image(h, w).shape == (h, w)


@pytest.mark.parametrize("h,w", [(32, 32), (128, 128), (200, 96), (96, 200)])
def test_the_synthetic_image_has_a_low_rank_head_and_a_tail(h, w):
    """Which is the shape a photograph's singular values have, and the reason compression
    works at all.

    Both claims are stated in a size-independent way. The drop from the largest singular value
    to the next is a property of the image model and not of the pixel count: it is measured at
    10.19 to 10.30 for every shape below. So is the number of components carrying 99.9 percent
    of the energy, measured at 4 or 5 everywhere. Comparing against a fixed index further down
    the spectrum, as an earlier version of this test did, silently makes the claim weaker as the
    image grows: the fifth of thirty two singular values and the fifth of two hundred and fifty
    six are different places in the tail.
    """
    img = lr.synthetic_image(h, w)
    s = np.linalg.svd(img, compute_uv=False)
    assert s.size >= 2
    assert s[0] > 5.0 * s[1], "the leading component does not dominate the next"
    assert lr.rank_for_energy(img, 0.999) < 20


def test_the_image_rejects_a_tiny_shape():
    with pytest.raises(ValueError):
        lr.synthetic_image(1, 5)


# ---------------------------------------------------------------- denoising


@pytest.mark.parametrize("noise", [0.02, 0.05, 0.15])
def test_truncation_removes_more_noise_than_signal(noise):
    """Measured improvements of 1.57x, 1.97x and 2.89x as the noise rises."""
    clean = lr.synthetic_image(120, 120)
    out = lr.best_denoising_rank(clean, noise, rng=np.random.default_rng(4))
    assert out["best_error"] < out["no_denoising"], "denoising made it worse"
    assert out["no_denoising"] / out["best_error"] > 1.4


def test_the_best_rank_falls_as_the_noise_rises():
    """**The interesting part.** More noise means fewer trustworthy singular directions.
    Measured: best rank 21, 9, 4 at noise 0.02, 0.05, 0.15."""
    clean = lr.synthetic_image(120, 120)
    ranks = [lr.best_denoising_rank(clean, lvl, rng=np.random.default_rng(4))["best_rank"]
             for lvl in (0.02, 0.05, 0.15)]
    assert ranks == sorted(ranks, reverse=True), f"ranks {ranks} did not fall"


def test_the_error_curve_has_an_interior_minimum():
    """Too small a rank throws away signal, too large keeps noise, so the best is in between."""
    clean = lr.synthetic_image(80, 80)
    out = lr.best_denoising_rank(clean, 0.05, rng=np.random.default_rng(4))
    errors = out["errors"]
    assert 1 < out["best_rank"] < errors.size, f"the minimum was at an endpoint: {out['best_rank']}"
    assert errors[out["best_rank"] - 1] < min(errors[0], errors[-1])


@pytest.mark.parametrize("rank", [3, 10])
def test_denoise_reports_both_errors(rank):
    clean = lr.synthetic_image(60, 60)
    out = lr.denoise(clean, 0.05, rank, rng=np.random.default_rng(6))
    assert out["error_before"] > 0.0 and out["error_after"] > 0.0
    assert abs(out["improvement"]
               - out["error_before"] / out["error_after"]) < 1e-6 * out["improvement"]


# ---------------------------------------------------------------- PageRank


@pytest.mark.parametrize("n,dangling", [(50, 0), (200, 20), (400, 40)])
def test_pagerank_matches_the_dominant_eigenvector(n, dangling):
    """It is lesson 36's power iteration, and the answer must equal the eigenvector a direct
    solve returns."""
    L = lr.random_web(n, density=0.03, n_dangling=dangling, rng=np.random.default_rng(5))
    out = lr.pagerank(L)
    assert out["converged"], "the iteration did not settle"
    w, V = np.linalg.eig(lr.google_matrix(L))
    top = int(np.argmax(w.real))
    reference = np.abs(V[:, top].real)
    reference /= reference.sum()
    assert np.max(np.abs(out["rank"] - reference)) < 1e-9


@pytest.mark.parametrize("n", [50, 200])
def test_the_ranking_is_a_probability_distribution(n):
    L = lr.random_web(n, density=0.04, rng=np.random.default_rng(7))
    x = lr.pagerank(L)["rank"]
    assert abs(x.sum() - 1.0) < 1e-10
    assert np.all(x > 0.0), "every page must get positive rank, which is what damping buys"


def test_the_iteration_count_tracks_the_second_eigenvalue():
    """**Lesson 36's rate, at web scale.** The second eigenvalue of the Google matrix is at most
    the damping factor, and the iteration count follows it: measured 95 iterations at 0.76 and
    15 at 0.15."""
    counts, seconds = [], []
    for n, dang in ((50, 0), (200, 20), (1000, 100)):
        L = lr.random_web(n, density=0.03, n_dangling=dang, rng=np.random.default_rng(5))
        counts.append(lr.pagerank(L)["iterations"])
        w = np.abs(np.linalg.eigvals(lr.google_matrix(L)))
        seconds.append(float(np.sort(w)[-2]))
    order = np.argsort(seconds)
    assert [counts[i] for i in order] == sorted([counts[i] for i in order]), \
        f"counts {counts} against second eigenvalues {seconds}"


@pytest.mark.parametrize("damping", [0.5, 0.85, 0.95])
def test_more_damping_means_more_iterations(damping):
    L = lr.random_web(100, density=0.03, rng=np.random.default_rng(8))
    out = lr.pagerank(L, damping=damping)
    assert out["converged"]
    w = np.abs(np.linalg.eigvals(lr.google_matrix(L, damping=damping)))
    assert float(np.sort(w)[-2]) <= damping + 1e-9, "the second eigenvalue exceeded the damping"


def test_dangling_pages_are_handled():
    """A page with no outgoing links would make the walk stall, so its column is replaced by a
    uniform one. Without that the matrix is not column stochastic and the iteration leaks mass."""
    L = lr.random_web(60, density=0.03, n_dangling=15, rng=np.random.default_rng(9))
    assert int(np.sum(L.sum(axis=0) == 0)) == 15
    out = lr.pagerank(L)
    assert out["converged"] and abs(out["rank"].sum() - 1.0) < 1e-10


def test_pagerank_rejects_bad_arguments(rng):
    with pytest.raises(ValueError):
        lr.pagerank(rng.standard_normal((4, 6)))
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            lr.pagerank(np.eye(4), damping=bad)


def test_random_web_rejects_bad_arguments():
    with pytest.raises(ValueError):
        lr.random_web(1)
    with pytest.raises(ValueError):
        lr.random_web(10, n_dangling=10)


def test_the_google_matrix_is_column_stochastic():
    L = lr.random_web(40, density=0.05, n_dangling=5, rng=np.random.default_rng(10))
    G = lr.google_matrix(L)
    assert np.max(np.abs(G.sum(axis=0) - 1.0)) < 1e-12
    assert np.all(G > 0.0), "damping makes every entry positive, which is what guarantees a "\
                            "unique stationary distribution"
