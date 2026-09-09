"""Tests for nalib.dft.

The DFT has more exactly known answers than anything else in this course, so almost nothing here
is checked against a tolerance alone: the transform of a pure tone is a single spike at a known
bin, the transform matrix is exactly unitary up to a factor, and numpy's `fft` is an independent
implementation to compare against.

Every test sweeps sizes, including odd, prime and 1.
"""
import math

import numpy as np
import pytest

from nalib import dft as dt

SIZES = [1, 2, 3, 4, 5, 7, 8, 16, 17, 32, 64]


def signal(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


# --------------------------------------------------------------------------- roots of unity


@pytest.mark.parametrize("n", SIZES)
def test_the_roots_lie_on_the_unit_circle_and_are_nth_roots(n):
    out = dt.root_properties(n)
    assert out["on_the_unit_circle"] < 1e-14
    assert out["each_is_an_nth_root"] < 1e-13


@pytest.mark.parametrize("n", SIZES)
def test_the_roots_sum_to_zero_except_at_size_one(n):
    out = dt.root_properties(n)
    assert out["sum_is_zero"]
    if n == 1:
        assert abs(out["sum"] - 1.0) < 1e-14
    else:
        assert abs(out["sum"]) < 1e-12 * n


@pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64])
def test_the_half_turn_is_minus_one_which_is_what_the_fft_uses(n):
    out = dt.root_properties(n)
    assert abs(out["half_turn_is_minus_one"] + 1.0) < 1e-14


@pytest.mark.parametrize("n", [3, 5, 7, 17])
def test_there_is_no_half_turn_at_an_odd_size(n):
    assert dt.root_properties(n)["half_turn_is_minus_one"] is None


@pytest.mark.parametrize("n", SIZES)
def test_the_exponents_wrap_at_n(n):
    assert dt.root_properties(n)["wraps_at_n"] < 1e-13


@pytest.mark.parametrize("n", [0, -1, -8])
def test_a_non_positive_size_is_rejected(n):
    with pytest.raises(ValueError, match="at least one root"):
        dt.roots_of_unity(n)


@pytest.mark.parametrize("sign", [0, 2, -3])
def test_an_invalid_sign_is_rejected(sign):
    with pytest.raises(ValueError, match="sign must be"):
        dt.roots_of_unity(8, sign)


# --------------------------------------------------------------------------- transform


@pytest.mark.parametrize("n", SIZES)
def test_the_transform_agrees_with_numpy(n):
    x = signal(n, n)
    got = dt.dft(x)
    ref = np.fft.fft(x)
    assert np.max(np.abs(got - ref)) < 1e-11 * max(float(np.max(np.abs(ref))), 1.0)


@pytest.mark.parametrize("n", SIZES)
def test_the_inverse_agrees_with_numpy(n):
    X = signal(n, n + 3) + 1j * signal(n, n + 9)
    got = dt.inverse_dft(X)
    ref = np.fft.ifft(X)
    assert np.max(np.abs(got - ref)) < 1e-11 * max(float(np.max(np.abs(ref))), 1.0)


@pytest.mark.parametrize("n", SIZES)
def test_the_round_trip_returns_the_data(n):
    assert dt.round_trip(signal(n, n)) < 1e-12


@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_a_pure_tone_is_one_spike_at_a_known_bin(k, n):
    """The whole basis for reading a spectrum: bin k means frequency k cycles per block."""
    j = np.arange(n)
    x = np.exp(2j * math.pi * k * j / n)
    X = dt.dft(x)
    assert abs(X[k % n] - n) < 1e-9 * n
    others = np.delete(np.abs(X), k % n)
    assert float(np.max(others)) < 1e-9 * n


@pytest.mark.parametrize("n", SIZES)
def test_the_transform_of_a_constant_is_a_single_dc_spike(n):
    X = dt.dft(np.ones(n))
    assert abs(X[0] - n) < 1e-10 * n
    if n > 1:
        assert float(np.max(np.abs(X[1:]))) < 1e-10 * n


def test_an_empty_signal_is_rejected():
    with pytest.raises(ValueError, match="empty signal"):
        dt.dft(np.zeros(0))
    with pytest.raises(ValueError, match="empty spectrum"):
        dt.inverse_dft(np.zeros(0))


@pytest.mark.parametrize("n", [0, -4])
def test_a_matrix_of_non_positive_size_is_rejected(n):
    with pytest.raises(ValueError, match="positive size"):
        dt.dft_matrix(n)


@pytest.mark.parametrize("n", SIZES)
def test_the_matrix_and_the_sum_agree(n):
    x = signal(n, n)
    assert np.max(np.abs(dt.dft_matrix(n) @ x - dt.dft(x))) < 1e-11 * max(n, 1.0)


# --------------------------------------------------------------------------- orthogonality


@pytest.mark.parametrize("n", SIZES)
def test_the_columns_are_orthogonal_with_squared_norm_n(n):
    out = dt.orthogonality(n)
    assert out["is_orthogonal"]
    assert out["diagonal_is_n"] < 1e-10 * n


@pytest.mark.parametrize("n", SIZES)
def test_the_scaled_transform_has_condition_number_one(n):
    F = dt.dft_matrix(n) / math.sqrt(n)
    assert float(np.linalg.cond(F)) == pytest.approx(1.0, abs=1e-9)


def test_the_dft_is_perfectly_conditioned_where_the_vandermonde_is_not():
    out = dt.conditioning((4, 8, 16, 20))
    assert np.max(np.abs(out["dft_condition"] - 1.0)) < 1e-9
    assert out["vandermonde_condition"][-1] > 1e12
    assert np.all(np.diff(out["vandermonde_condition"]) > 0)


@pytest.mark.parametrize("n", SIZES)
def test_parseval_holds(n):
    assert dt.parseval(signal(n, n))["relative_gap"] < 1e-12


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_parseval_holds_for_complex_data_too(n):
    x = signal(n, n) + 1j * signal(n, n + 100)
    assert dt.parseval(x)["relative_gap"] < 1e-12


# --------------------------------------------------------------------------- interpolation


@pytest.mark.parametrize("n", SIZES)
def test_both_conventions_interpolate_the_data_exactly(n):
    out = dt.interpolation_theorem(signal(n, n))
    assert out["both_interpolate"]
    assert out["naive_gap"] < 1e-11
    assert out["centred_gap"] < 1e-11


@pytest.mark.parametrize("n", [8, 16, 32, 64])
def test_only_the_centred_convention_gives_a_real_interpolant(n):
    f = lambda t: np.sin(2.0 * math.pi * 3.0 * t) + 0.5 * np.cos(2.0 * math.pi * t)
    out = dt.naive_versus_centred(f, n)
    assert out["centred_max_imaginary"] < 1e-10
    assert out["naive_max_imaginary"] > 0.1
    assert out["centred_error"] < 1e-10
    assert out["naive_error"] > 0.1


@pytest.mark.parametrize("n", [16, 32, 64])
def test_the_centred_interpolant_reproduces_a_band_limited_signal_everywhere(n):
    f = lambda t: np.sin(2.0 * math.pi * 2.0 * t) + 0.3 * np.cos(2.0 * math.pi * 5.0 * t)
    out = dt.naive_versus_centred(f, n)
    assert out["centred_error"] < 1e-10


@pytest.mark.parametrize("n", [4, 5, 8, 16, 17])
def test_a_real_signal_has_a_hermitian_spectrum(n):
    out = dt.hermitian_symmetry(signal(n, n))
    assert out["holds"]
    assert out["independent_values"] == n // 2 + 1


@pytest.mark.parametrize("n", [8, 16, 32])
def test_the_real_form_reconstructs_the_signal(n):
    x = signal(n, n)
    out = dt.real_form(dt.dft(x))
    t = np.arange(n) / n
    got = np.full(n, out["cosine"][0] / 2.0)
    for k in out["frequencies"][1:]:
        got = got + out["cosine"][k] * np.cos(2 * math.pi * k * t) \
            + out["sine"][k] * np.sin(2 * math.pi * k * t)
    assert np.max(np.abs(got - x)) < 1e-10 * max(float(np.max(np.abs(x))), 1.0)


# --------------------------------------------------------------------------- aliasing


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_frequencies_a_multiple_of_n_apart_are_identical_on_the_grid(n):
    out = dt.aliasing(n, [1, 1 + n, 1 + 2 * n, 1 + 5 * n])
    assert np.all(out["gap_on_the_grid"] < 1e-10)
    assert np.all(out["folded_to"] == 1)


@pytest.mark.parametrize("n", [8, 16, 32])
def test_those_same_frequencies_are_completely_different_off_the_grid(n):
    out = dt.aliasing(n, [1, 1 + n, 1 + 2 * n])
    assert np.all(out["gap_off_the_grid"][1:] > 0.5)


@pytest.mark.parametrize("freq", [1.0, 3.0, 7.0])
def test_a_tone_below_nyquist_is_reconstructed_and_above_it_is_not(freq):
    tone = lambda t, f: np.sin(2.0 * math.pi * f * t)
    out = dt.nyquist(tone, freq, sizes=(4, 8, 16, 32, 64))
    below = ~out["above_nyquist"]
    assert np.all(out["reconstruction_error"][below] < 1e-9)
    if np.any(out["above_nyquist"]):
        assert np.max(out["reconstruction_error"][out["above_nyquist"]]) > 1e-3


def test_the_folded_frequency_is_what_the_grid_thinks_it_saw():
    tone = lambda t, f: np.sin(2.0 * math.pi * f * t)
    out = dt.nyquist(tone, 7.0, sizes=(8,))
    assert int(out["folded_frequency"][0]) == 1


# --------------------------------------------------------------------------- theorems


@pytest.mark.parametrize("n", [4, 8, 16, 17, 32])
@pytest.mark.parametrize("shift", [0, 1, 3, -2])
def test_shifting_multiplies_the_spectrum_by_a_phase(n, shift):
    out = dt.shift_theorem(signal(n, n), shift)
    assert out["relative_gap"] < 1e-11


@pytest.mark.parametrize("n", [4, 8, 16, 17])
@pytest.mark.parametrize("shift", [1, 5, -3])
def test_a_shift_does_not_change_the_magnitude_spectrum(n, shift):
    assert dt.shift_theorem(signal(n, n), shift)["magnitudes_unchanged"] < 1e-11


@pytest.mark.parametrize("n", [2, 4, 5, 8, 16])
def test_convolution_becomes_multiplication(n):
    out = dt.convolution_theorem(signal(n, n), signal(n, n + 50))
    assert out["relative_gap"] < 1e-11


def test_convolution_needs_equal_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        dt.convolution_theorem(np.zeros(4), np.zeros(5))


# --------------------------------------------------------------------------- twiddles


@pytest.mark.parametrize("n", [8, 64, 256, 1024])
@pytest.mark.parametrize("method", ["naive", "stable", "direct"])
def test_every_twiddle_method_gets_the_values_roughly_right(n, method):
    out = dt.twiddle_by_recurrence(n, method)
    assert out["max_error"] < 1e-9
    assert out["cos"].size == n and out["sin"].size == n


@pytest.mark.parametrize("n", [256, 1024, 4096, 16384])
def test_the_stable_recurrence_beats_the_naive_one(n):
    naive = dt.twiddle_by_recurrence(n, "naive")["max_error"]
    stable = dt.twiddle_by_recurrence(n, "stable")["max_error"]
    assert stable < naive


def test_the_naive_error_grows_with_the_length_and_the_stable_one_barely_does():
    out = dt.recurrence_comparison((64, 256, 1024, 4096, 16384))
    naive_growth = out["naive_error"][-1] / out["naive_error"][0]
    stable_growth = out["stable_error"][-1] / out["stable_error"][0]
    assert naive_growth > 50.0
    assert stable_growth < 20.0
    assert naive_growth > 5.0 * stable_growth


@pytest.mark.parametrize("n", [1024, 4096])
def test_the_naive_recurrence_lets_the_modulus_drift_and_the_stable_one_does_not(n):
    assert dt.twiddle_by_recurrence(n, "naive")["worst_modulus_drift"] > \
        10.0 * dt.twiddle_by_recurrence(n, "stable")["worst_modulus_drift"]


@pytest.mark.parametrize("bad", ["fast", "", "STABLE2"])
def test_an_unknown_twiddle_method_is_rejected(bad):
    with pytest.raises(ValueError, match="method must be"):
        dt.twiddle_by_recurrence(64, bad)


@pytest.mark.parametrize("n", [0, -5])
def test_a_twiddle_run_of_non_positive_length_is_rejected(n):
    with pytest.raises(ValueError, match="positive size"):
        dt.twiddle_by_recurrence(n)
