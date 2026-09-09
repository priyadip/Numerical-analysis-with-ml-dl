"""Tests for nalib.mllinalg.

Four groups. The identity group asserts that the machine learning names really are the course's
objects: that PCA reproduces the singular values it was built from, that the explained variance is
the squared singular value share, that reconstruction from all components is the identity map, and
that ridge with a zero penalty is ordinary least squares. If any of those failed, the lesson's whole
claim would be wrong.

The conditioning group asserts the thing the identity group hides. The three routes to PCA are the
same mathematics, so the only way to tell them apart is to measure how they degrade, and the two
fitted slopes, 1 for the SVD route and 2 for the covariance route, are the assertion. The same group
checks that ``lam = 0`` is not always the right answer and, in the other direction, that it sometimes
is.

The invariance group asserts two properties that pull in opposite directions: whitening produces the
identity covariance for any orthogonal factor, so it is not unique, and the sample geometry it
leaves behind does not depend on the units the features arrived in. Both are exact statements and
both are checked as exact.

The low-rank group asserts Eckart-Young by trying to beat it and failing, and asserts the rank bound
on a trained update, which is an exact integer and not a tolerance. The contrast against a random
matrix of the same shape is what keeps the statement from being circular.
"""
import functools
import math

import numpy as np
import pytest

from nalib import mllinalg


@functools.lru_cache(maxsize=None)
def routes():
    return mllinalg.the_three_routes_to_pca_agree_until_they_do_not()


@functools.lru_cache(maxsize=None)
def scaling():
    return mllinalg.pca_is_not_scale_invariant()


@functools.lru_cache(maxsize=None)
def whitening():
    return mllinalg.whitening_is_not_unique()


@functools.lru_cache(maxsize=None)
def generalizing():
    return mllinalg.whitening_needs_a_penalty_to_generalize()


@functools.lru_cache(maxsize=None)
def filtering():
    return mllinalg.ridge_is_a_spectral_filter()


@functools.lru_cache(maxsize=None)
def stability():
    return mllinalg.the_augmented_form_is_the_stable_one()


@functools.lru_cache(maxsize=None)
def penalties():
    return mllinalg.ridge_does_not_always_help()


@functools.lru_cache(maxsize=None)
def ranks():
    return mllinalg.a_decaying_spectrum_has_no_single_rank()


@functools.lru_cache(maxsize=None)
def approximation():
    return mllinalg.nothing_beats_the_truncated_svd()


@functools.lru_cache(maxsize=None)
def adaptation():
    return mllinalg.a_trained_update_is_low_rank_and_a_random_one_is_not()


# ------------------------------------------------------------------ the objects are the same


def test_random_orthogonal_is_orthogonal():
    for size in (2, 5, 13):
        q = mllinalg.random_orthogonal(size, seed=size)
        assert np.allclose(q.T @ q, np.eye(size), atol=1e-13)


def test_random_orthogonal_rejects_an_empty_size():
    with pytest.raises(ValueError):
        mllinalg.random_orthogonal(0)


def test_spectrum_is_decreasing_and_normalized():
    for kind in ("power", "exponential", "gap"):
        values = mllinalg.spectrum(20, kind, 1.5 if kind != "gap" else 0.01)
        assert values[0] == pytest.approx(1.0)
        assert np.all(np.diff(values) <= 1e-15)


def test_spectrum_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        mllinalg.spectrum(4, "wishful")


def test_matrix_with_spectrum_has_that_spectrum():
    wanted = np.array([4.0, 2.0, 0.5, 0.125])
    made = mllinalg.matrix_with_spectrum(wanted, rows=17, seed=3)
    assert np.allclose(np.linalg.svd(made, compute_uv=False), wanted, rtol=1e-12)


def test_the_centred_form_survives_centring():
    wanted = np.geomspace(1.0, 1e-3, 6)
    made = mllinalg.matrix_with_spectrum(wanted, rows=50, seed=4, centred=True)
    assert np.max(np.abs(made.mean(axis=0))) < 1e-14
    assert np.allclose(np.linalg.svd(made - made.mean(axis=0), compute_uv=False), wanted,
                       rtol=1e-12)


def test_matrix_with_spectrum_needs_enough_rows():
    with pytest.raises(ValueError):
        mllinalg.matrix_with_spectrum(np.ones(5), rows=3)


def test_pca_recovers_the_singular_values_it_was_built_from():
    wanted = np.geomspace(1.0, 1e-4, 8)
    data = mllinalg.matrix_with_spectrum(wanted, rows=120, seed=11, centred=True)
    assert np.allclose(mllinalg.pca(data)["singular_values"], wanted, rtol=1e-10)


def test_explained_variance_is_the_squared_singular_value_share():
    data = mllinalg.dataset(80, 7, seed=5)["data"]
    model = mllinalg.pca(data)
    s = model["singular_values"]
    assert np.allclose(model["explained"], s ** 2 / np.sum(s ** 2), atol=1e-14)
    assert model["explained"].sum() == pytest.approx(1.0)


def test_reconstruction_with_every_component_is_the_identity():
    data = mllinalg.dataset(60, 9, seed=6)["data"]
    model = mllinalg.pca(data)
    assert np.allclose(mllinalg.reconstruct(model), data, atol=1e-12)


def test_pca_directions_are_orthonormal():
    data = mllinalg.dataset(70, 6, seed=8)["data"]
    directions = mllinalg.pca(data)["directions"]
    assert np.allclose(directions @ directions.T, np.eye(directions.shape[0]), atol=1e-13)


def test_pca_rejects_a_single_row():
    with pytest.raises(ValueError):
        mllinalg.pca(np.ones((1, 4)))
    with pytest.raises(ValueError):
        mllinalg.pca_by_covariance(np.ones((1, 4)))
    with pytest.raises(ValueError):
        mllinalg.pca_by_gram(np.ones((1, 4)))


def test_the_three_routes_agree_when_the_data_is_easy():
    data = mllinalg.dataset(90, 5, decay=0.3, seed=12)["data"]
    by_svd = mllinalg.pca(data)["singular_values"]
    assert np.allclose(mllinalg.pca_by_covariance(data)["singular_values"], by_svd, rtol=1e-10)
    assert np.allclose(mllinalg.pca_by_gram(data)["singular_values"][:by_svd.size], by_svd,
                       rtol=1e-10)


def test_subspace_angle_is_zero_for_the_same_subspace():
    basis = mllinalg.random_orthogonal(6, seed=2)[:3]
    # arccos near 1 halves the digits, so 1e-4 degrees is as close to zero as this gets
    assert mllinalg.subspace_angle(basis, basis) < 1e-4
    assert mllinalg.subspace_angle(basis[:1], mllinalg.random_orthogonal(6, seed=2)[3:4]) > 89.0


def test_ridge_at_zero_penalty_is_least_squares():
    data = mllinalg.dataset(40, 6, seed=14)["data"]
    rng = np.random.default_rng(0)
    target = rng.standard_normal(data.shape[0])
    reference = np.linalg.lstsq(data, target, rcond=None)[0]
    assert np.allclose(mllinalg.ridge_svd(data, target, 0.0)["solution"], reference, atol=1e-8)


def test_the_three_ridge_routes_agree_at_a_healthy_penalty():
    matrix = mllinalg.matrix_with_spectrum(np.geomspace(1.0, 1e-2, 5), rows=30, seed=15)
    rng = np.random.default_rng(1)
    target = rng.standard_normal(matrix.shape[0])
    exact = mllinalg.ridge_svd(matrix, target, 1e-2)["solution"]
    assert np.allclose(mllinalg.ridge_normal(matrix, target, 1e-2)["solution"], exact, atol=1e-12)
    assert np.allclose(mllinalg.ridge_augmented(matrix, target, 1e-2)["solution"], exact,
                       atol=1e-12)


def test_ridge_filters_are_between_zero_and_one():
    values = np.geomspace(1.0, 1e-8, 12)
    factors = mllinalg.ridge_filters(values, 1e-6)
    assert np.all(factors > 0.0) and np.all(factors < 1.0)
    assert np.all(np.diff(factors) < 0.0)


def test_the_ridge_solution_norm_falls_with_the_penalty():
    matrix = mllinalg.matrix_with_spectrum(np.geomspace(1.0, 1e-4, 8), rows=40, seed=16)
    rng = np.random.default_rng(2)
    path = mllinalg.ridge_path(matrix, rng.standard_normal(40), np.geomspace(1e-10, 1.0, 9))
    norms = [row["norm"] for row in path["rows"]]
    assert all(norms[i + 1] <= norms[i] + 1e-12 for i in range(len(norms) - 1))


def test_whiten_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        mllinalg.whiten(np.eye(4), "bleach")


# ------------------------------------------------------------------ conditioning


def test_the_svd_route_loses_one_digit_per_digit_of_conditioning():
    assert routes()["svd_grows_like_the_condition_number"]
    assert abs(routes()["svd_slope"] - 1.0) < 0.2


def test_the_covariance_route_loses_two():
    assert routes()["covariance_grows_like_its_square"]
    assert abs(routes()["covariance_slope"] - 2.0) < 0.2


def test_the_two_routes_agree_while_the_data_is_easy():
    assert routes()["agree_when_well_conditioned"]
    assert routes()["rows"][0]["covariance_error"] < 1e-10


def test_the_covariance_route_eventually_fails_outright():
    assert routes()["covariance_route_fails"]
    assert routes()["some_eigenvalue_goes_negative"]


def test_the_gram_route_is_not_a_safe_substitute():
    worst = max(row["gram_error"] for row in routes()["rows"])
    assert worst > 1.0


def test_the_augmented_route_beats_the_normal_equations():
    assert stability()["augmented_is_better"]
    assert stability()["digits_lost"] > 3.0


def test_the_gram_condition_number_is_the_square_of_the_stacked_one():
    assert stability()["condition_is_squared"]
    assert stability()["worst_condition_mismatch"] < 1e-5


def test_ridge_is_exactly_a_spectral_filter():
    assert filtering()["filters_are_exact"]
    assert filtering()["condition_formula_holds"]


def test_the_penalty_switches_directions_off_one_at_a_time():
    kept = [row["kept"] for row in filtering()["rows"]]
    assert all(kept[i + 1] <= kept[i] for i in range(len(kept) - 1))
    assert kept[0] > kept[-1]


def test_a_clean_problem_wants_no_penalty():
    assert penalties()["no_noise_wants_no_penalty"]
    assert penalties()["rows"][0]["gain"] == pytest.approx(1.0)


def test_a_noisy_problem_wants_one():
    assert penalties()["noise_wants_a_penalty"]
    assert penalties()["largest_gain"] > 100.0


def test_the_best_penalty_grows_with_the_noise():
    assert penalties()["best_penalty_grows_with_noise"]


# ------------------------------------------------------------------ invariance


def test_both_whiteners_whiten():
    assert whitening()["both_whiten"]
    assert whitening()["pca_covariance_error"] < 1e-12
    assert whitening()["zca_covariance_error"] < 1e-12


def test_they_differ_by_an_orthogonal_factor():
    assert whitening()["they_differ_by_a_rotation"]
    assert whitening()["rotation_orthogonality"] < 1e-8


def test_zca_is_the_one_that_moves_the_data_least():
    assert whitening()["zca_is_closer_to_the_identity"]
    assert whitening()["zca_distance"] < whitening()["pca_distance"]


def test_pca_directions_move_when_a_feature_is_rescaled():
    assert scaling()["directions_move"]
    assert scaling()["the_scaled_feature_takes_over"]


def test_the_whitened_geometry_does_not_move():
    assert scaling()["whitening_is_invariant"]
    assert max(row["whitened_gram_change"] for row in scaling()["rows"]) < 1e-8


def test_a_whitener_is_exact_on_its_own_sample_and_wrong_on_fresh_data():
    assert generalizing()["training_error_says_nothing"]
    assert generalizing()["rows"][0]["held_out_error"] > 0.5


def test_the_whitening_penalty_has_an_interior_optimum():
    assert generalizing()["the_penalty_helps"]
    assert generalizing()["too_much_penalty_hurts"]
    assert generalizing()["gain"] > 1.5


def test_the_penalty_caps_the_amplification():
    amplifications = [row["amplification"] for row in generalizing()["rows"]]
    assert all(amplifications[i + 1] <= amplifications[i] for i in range(len(amplifications) - 1))


# ------------------------------------------------------------------ low rank


def test_truncation_error_is_the_next_singular_value():
    values = np.geomspace(1.0, 1e-3, 10)
    matrix = mllinalg.matrix_with_spectrum(values, rows=25, seed=21)
    for rank in (1, 4, 9):
        cut = mllinalg.truncate(matrix, rank)
        assert cut["spectral_error"] == pytest.approx(values[rank], rel=1e-10)
        assert cut["frobenius_error"] == pytest.approx(
            float(np.linalg.norm(values[rank:])), rel=1e-10)


def test_truncating_at_full_rank_is_exact():
    matrix = mllinalg.matrix_with_spectrum(np.geomspace(1.0, 1e-2, 6), rows=20, seed=22)
    cut = mllinalg.truncate(matrix, 6)
    assert cut["relative_error"] < 1e-14


def test_energy_captured_is_monotone_and_reaches_one():
    values = mllinalg.spectrum(30, "power", 1.0)
    shares = [mllinalg.energy_captured(values, r) for r in range(values.size + 1)]
    assert all(shares[i + 1] >= shares[i] for i in range(len(shares) - 1))
    assert shares[-1] == pytest.approx(1.0)
    assert shares[0] == 0.0


def test_the_spectral_summary_brackets_the_ranks():
    values = mllinalg.spectrum(64, "power", 1.0)
    summary = mllinalg.spectral_summary(values)
    assert summary["stable_rank"] <= summary["effective_rank"] <= summary["numerical_rank"]
    assert summary["condition"] == pytest.approx(values[0] / values[-1], rel=1e-12)


def test_a_flat_spectrum_has_effective_rank_equal_to_its_size():
    summary = mllinalg.spectral_summary(np.ones(17))
    assert summary["effective_rank"] == pytest.approx(17.0, rel=1e-10)
    assert summary["stable_rank"] == pytest.approx(17.0, rel=1e-10)


def test_the_three_ranks_disagree_on_a_power_law():
    assert ranks()["the_three_ranks_disagree"]
    assert ranks()["all_are_full_rank"]
    assert ranks()["largest_spread"] > 10.0


def test_steeper_decay_lowers_the_effective_rank():
    assert ranks()["steeper_decay_means_lower_effective_rank"]


def test_nothing_beats_the_truncated_svd():
    assert approximation()["optimal_is_optimal"]
    assert approximation()["worst_random_ratio"] > 1.0
    assert approximation()["worst_column_ratio"] > 1.0


def test_oversampling_closes_most_of_the_randomized_gap():
    assert approximation()["oversampling_helps"]
    assert approximation()["worst_oversampled_ratio"] < approximation()["worst_random_ratio"]


def test_the_alternatives_are_still_within_a_small_factor():
    assert approximation()["worst_oversampled_ratio"] < 1.5


def test_an_adapter_starts_at_zero():
    made = mllinalg.adapter(12, 9, 3, seed=1)
    assert np.max(np.abs(made["update"])) == 0.0
    assert made["parameters"] == 3 * (12 + 9)
    assert made["dense_parameters"] == 12 * 9


def test_an_adapter_rejects_a_zero_rank():
    with pytest.raises(ValueError):
        mllinalg.adapter(4, 4, 0)


def test_fitting_an_adapter_reproduces_the_truncated_svd():
    matrix = mllinalg.matrix_with_spectrum(np.geomspace(1.0, 1e-2, 8), rows=30, seed=23)
    fitted = mllinalg.fit_adapter(matrix, 3)
    assert np.allclose(fitted["up"] @ fitted["down"],
                       mllinalg.truncate(matrix, 3)["approximation"], atol=1e-12)


def test_training_reduces_the_loss():
    run = mllinalg.train_linear_layer(20, 5, 200, 40, 8, seed=31)
    assert run["losses"][-1] < run["losses"][0]


def test_the_update_rank_is_exactly_the_step_bound():
    for steps, batch in ((1, 1), (3, 2), (5, 1)):
        run = mllinalg.train_linear_layer(40, 30, 100, steps, batch, seed=32)
        assert run["numerical_rank"] == run["rank_bound"] == min(steps * batch, 30)


def test_training_rejects_an_empty_shape():
    with pytest.raises(ValueError):
        mllinalg.train_linear_layer(0, 4, 10, 3, 2)


def test_the_measured_rank_bound_is_exact():
    assert adaptation()["the_bound_is_exact"]


def test_an_adapter_captures_a_trained_update_and_not_a_random_matrix():
    assert adaptation()["the_adapter_captures_what_fits"]
    assert adaptation()["it_does_not_capture_noise"]
    assert adaptation()["advantage_over_noise"] > 3.0


def test_a_random_matrix_is_not_flat_either():
    assert adaptation()["noise_is_not_flat_either"]
    assert adaptation()["random_numerical_rank"] > adaptation()["adapter_rank"]


def test_the_adapter_saves_the_parameters_it_claims_to():
    assert adaptation()["parameter_ratio"] > 1.0


def test_every_measurement_carries_a_note():
    for out in (routes(), scaling(), whitening(), generalizing(), filtering(), stability(),
                penalties(), ranks(), approximation(), adaptation()):
        assert isinstance(out["note"], str) and out["note"]
