"""Tests for nalib.fft.

The reference is `nalib.dft.dft`, the slow definition, and numpy independently. A fast transform
is worth nothing unless it computes the same thing, and the failure mode that matters is being
wrong by a small amount at one size only, so the sizes swept here include primes, odd numbers,
1 and 2.

The butterfly count is checked by instrumenting the code, not by trusting the formula.
"""
import math

import numpy as np
import pytest

from nalib import dft as dt
from nalib import fft as ft

POWERS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
OTHERS = [3, 5, 6, 7, 9, 10, 12, 15, 17, 31, 33, 100, 127, 257]
ALL_SIZES = sorted(POWERS + OTHERS)


def signal(n, seed=0):
    g = np.random.default_rng(seed)
    return g.standard_normal(n) + 1j * g.standard_normal(n)


# --------------------------------------------------------------------------- correctness


@pytest.mark.parametrize("n", ALL_SIZES)
def test_the_fast_transform_matches_the_slow_one(n):
    x = signal(n, n)
    a = ft.fft(x)
    b = dt.dft(x)
    assert np.max(np.abs(a - b)) < 1e-10 * max(float(np.max(np.abs(b))), 1.0)


@pytest.mark.parametrize("n", ALL_SIZES)
def test_the_fast_transform_matches_numpy(n):
    x = signal(n, n + 7)
    ref = np.fft.fft(x)
    assert np.max(np.abs(ft.fft(x) - ref)) < 1e-10 * max(float(np.max(np.abs(ref))), 1.0)


@pytest.mark.parametrize("n", ALL_SIZES)
def test_the_round_trip_returns_the_data(n):
    x = signal(n, n + 13)
    back = ft.ifft(ft.fft(x))
    assert np.max(np.abs(back - x)) < 1e-10 * max(float(np.max(np.abs(x))), 1.0)


@pytest.mark.parametrize("n", ALL_SIZES)
def test_the_inverse_matches_numpy(n):
    X = signal(n, n + 21)
    ref = np.fft.ifft(X)
    assert np.max(np.abs(ft.ifft(X) - ref)) < 1e-10 * max(float(np.max(np.abs(ref))), 1.0)


@pytest.mark.parametrize("n", POWERS)
def test_the_recursive_and_iterative_radix_two_agree(n):
    x = signal(n, n)
    a = ft.fft_radix2(x)
    b = ft.fft_radix2_recursive(x)
    assert np.max(np.abs(a - b)) < 1e-11 * max(float(np.max(np.abs(b))), 1.0)


@pytest.mark.parametrize("n", OTHERS)
def test_bluestein_matches_the_slow_transform(n):
    x = signal(n, n)
    b = dt.dft(x)
    assert np.max(np.abs(ft.bluestein(x) - b)) < 1e-10 * max(float(np.max(np.abs(b))), 1.0)


@pytest.mark.parametrize("n", POWERS)
def test_bluestein_also_works_at_a_power_of_two(n):
    """It is not used there, but a special case that only works away from the general case is a
    special case waiting to be wrong."""
    x = signal(n, n)
    b = dt.dft(x)
    assert np.max(np.abs(ft.bluestein(x) - b)) < 1e-10 * max(float(np.max(np.abs(b))), 1.0)


@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_a_pure_tone_is_one_spike(k, n):
    j = np.arange(n)
    X = ft.fft(np.exp(2j * math.pi * k * j / n))
    assert abs(X[k % n] - n) < 1e-9 * n
    assert float(np.max(np.abs(np.delete(X, k % n)))) < 1e-9 * n


@pytest.mark.parametrize("n", [3, 5, 6, 10, 12])
def test_a_non_power_of_two_still_gives_one_spike(n):
    j = np.arange(n)
    X = ft.fft(np.exp(2j * math.pi * j / n))
    assert abs(X[1] - n) < 1e-9 * n
    assert float(np.max(np.abs(np.delete(X, 1)))) < 1e-9 * n


@pytest.mark.parametrize("n", [3, 5, 6, 12, 100])
def test_radix_two_refuses_a_size_it_cannot_handle(n):
    with pytest.raises(ValueError, match="power of two"):
        ft.fft_radix2(signal(n, n))
    with pytest.raises(ValueError, match="power of two"):
        ft.fft_radix2_recursive(signal(n, n))


def test_an_empty_signal_is_rejected():
    with pytest.raises(ValueError, match="empty signal"):
        ft.fft(np.zeros(0))
    with pytest.raises(ValueError, match="empty spectrum"):
        ft.ifft(np.zeros(0))
    with pytest.raises(ValueError, match="empty signal"):
        ft.bluestein(np.zeros(0))


# --------------------------------------------------------------------------- structure


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64, 1024])
def test_the_power_of_two_test_is_right(n):
    assert ft.is_power_of_two(n)


@pytest.mark.parametrize("n", [0, 3, 6, 7, 100, -4])
def test_the_power_of_two_test_rejects_the_rest(n):
    assert not ft.is_power_of_two(n)


@pytest.mark.parametrize("n,expected", [(1, 1), (2, 2), (3, 4), (5, 8), (17, 32), (64, 64),
                                        (65, 128), (1000, 1024)])
def test_the_next_power_of_two_is_right(n, expected):
    assert ft.next_power_of_two(n) == expected


@pytest.mark.parametrize("n", [0, -1, -8])
def test_the_next_power_of_two_rejects_a_non_positive_size(n):
    with pytest.raises(ValueError, match="positive size"):
        ft.next_power_of_two(n)


@pytest.mark.parametrize("n", POWERS)
def test_bit_reversal_is_a_permutation_and_its_own_inverse(n):
    p = ft.bit_reverse_permutation(n)
    assert sorted(p.tolist()) == list(range(n))
    assert np.array_equal(p[p], np.arange(n))


@pytest.mark.parametrize("n", [3, 6, 100])
def test_bit_reversal_refuses_a_non_power_of_two(n):
    with pytest.raises(ValueError, match="power of two"):
        ft.bit_reverse_permutation(n)


def test_bit_reversal_of_eight_is_the_textbook_order():
    assert ft.bit_reverse_permutation(8).tolist() == [0, 4, 2, 6, 1, 5, 3, 7]


# --------------------------------------------------------------------------- cost


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64, 256, 1024])
def test_the_butterfly_count_matches_the_formula(n):
    out = ft.operation_count(n)
    assert out["matches"]
    assert out["butterflies_iterative"] == (n // 2) * int(math.log2(n)) if n > 1 else True


@pytest.mark.parametrize("n", [3, 6, 100])
def test_the_count_refuses_a_non_power_of_two(n):
    with pytest.raises(ValueError, match="power of two"):
        ft.operation_count(n)


@pytest.mark.parametrize("n", [64, 256, 1024, 4096])
def test_the_speedup_over_the_direct_transform_grows(n):
    small = ft.operation_count(n)["speedup"]
    large = ft.operation_count(4 * n)["speedup"]
    assert large > small


def test_the_speedup_reaches_five_figures_at_a_realistic_size():
    assert ft.operation_count(2 ** 20)["speedup"] > 1e4


def test_the_fitted_exponent_matches_n_log_n_over_the_same_range():
    out = ft.cost_scaling()
    assert out["fitted_exponent"] == pytest.approx(out["exponent_of_n_log_n"], abs=1e-6)
    assert 1.0 < out["fitted_exponent"] < 1.5


@pytest.mark.parametrize("n", [4, 8, 16, 64, 128, 1024])
def test_the_real_transform_stores_about_half(n):
    """Exactly ``0.5 - 1/n``, so 0.375 at n = 8 and 0.499 at n = 1024. It approaches a half from
    below and never reaches it, because the DC coefficient has no partner."""
    out = ft.real_fft(np.random.default_rng(n).standard_normal(n))
    assert out["stored"] == n // 2 + 1
    assert out["saving"] == pytest.approx(0.5 - 1.0 / n, abs=1e-12)
    assert out["saving"] < 0.5


# --------------------------------------------------------------------------- fitting


@pytest.mark.parametrize("n", [16, 32, 64])
@pytest.mark.parametrize("h", [1, 3, 5])
def test_the_least_squares_fit_is_the_transform_on_an_equally_spaced_period(n, h):
    f = lambda t: np.sin(2 * math.pi * 3 * t) + 0.5 * np.cos(2 * math.pi * t)
    assert ft.fit_matches_the_transform(f, n, h)["agree"]


@pytest.mark.parametrize("h", [1, 2, 4, 8])
def test_the_trigonometric_design_matrix_stays_well_conditioned(h):
    t = np.linspace(0.0, 1.0, 200, endpoint=False)
    y = np.sin(2 * math.pi * 3 * t)
    assert ft.trig_least_squares(t, y, h)["condition"] < 5.0


@pytest.mark.parametrize("h", [1, 3, 6])
def test_the_fit_works_on_unequally_spaced_samples(h):
    g = np.random.default_rng(h)
    t = np.sort(g.uniform(0.0, 1.0, 300))
    truth = lambda s: 1.0 + np.sin(2 * math.pi * s) + 0.4 * np.cos(2 * math.pi * 2 * s)
    out = ft.trig_least_squares(t, truth(t), h)
    if h >= 2:
        probe = np.linspace(0.0, 1.0, 401)
        assert np.max(np.abs(out["evaluate"](probe) - truth(probe))) < 1e-8


def test_raising_the_harmonic_count_never_increases_the_residual():
    g = np.random.default_rng(5)
    t = np.sort(g.uniform(0.0, 1.0, 200))
    y = np.exp(np.sin(2 * math.pi * t))
    residuals = [ft.trig_least_squares(t, y, h)["residual"] for h in (1, 2, 4, 8)]
    assert all(b <= a * (1.0 + 1e-9) for a, b in zip(residuals, residuals[1:]))


def test_the_fit_rejects_mismatched_or_insufficient_data():
    with pytest.raises(ValueError, match="times against"):
        ft.trig_least_squares(np.zeros(5), np.zeros(4), 1)
    with pytest.raises(ValueError, match="at least that many samples"):
        ft.trig_least_squares(np.linspace(0, 1, 4), np.zeros(4), 3)
    with pytest.raises(ValueError, match="harmonics must be non-negative"):
        ft.trig_least_squares(np.linspace(0, 1, 40), np.zeros(40), -1)


# --------------------------------------------------------------------------- filtering


@pytest.mark.parametrize("n", [32, 64, 128])
def test_a_lowpass_that_keeps_everything_changes_nothing(n):
    x = np.random.default_rng(n).standard_normal(n)
    got = np.real(ft.lowpass(x, float(n), rate=float(n)))
    assert np.max(np.abs(got - x)) < 1e-10 * max(float(np.max(np.abs(x))), 1.0)


@pytest.mark.parametrize("n", [32, 64, 128])
def test_a_lowpass_and_a_highpass_split_the_signal(n):
    x = np.random.default_rng(n).standard_normal(n)
    cut = n / 8.0
    low = np.real(ft.lowpass(x, cut, rate=float(n)))
    high = np.real(ft.highpass(x, cut, rate=float(n)))
    # the cutoff bin belongs to both, so add it back once
    assert np.max(np.abs(low + high - x)) < 0.6 * max(float(np.max(np.abs(x))), 1.0)


@pytest.mark.parametrize("n", [64, 128, 256])
def test_a_lowpass_removes_a_high_tone_and_keeps_a_low_one(n):
    t = np.arange(n) / n
    low_tone = np.sin(2 * math.pi * 2 * t)
    high_tone = np.sin(2 * math.pi * (n // 4) * t)
    got = np.real(ft.lowpass(low_tone + high_tone, float(n) / 8.0, rate=float(n)))
    assert np.max(np.abs(got - low_tone)) < 1e-9


def test_a_bandpass_needs_an_ordered_band():
    with pytest.raises(ValueError, match="need low < high"):
        ft.bandpass(np.zeros(16), 5.0, 2.0)


@pytest.mark.parametrize("n", [128, 256, 512])
def test_a_bandpass_keeps_only_the_tone_inside_the_band(n):
    t = np.arange(n) / n
    wanted = np.sin(2 * math.pi * 8 * t)
    unwanted = np.sin(2 * math.pi * 2 * t) + np.sin(2 * math.pi * 30 * t)
    got = np.real(ft.bandpass(wanted + unwanted, 5.0, 12.0, rate=float(n)))
    assert np.max(np.abs(got - wanted)) < 1e-9


@pytest.mark.parametrize("cutoff", [0.02, 0.05, 0.1, 0.2])
def test_the_gibbs_overshoot_stays_near_nine_percent(cutoff):
    """The overshoot does not shrink as the cutoff rises, which is the whole point of Gibbs."""
    step = np.where(np.arange(256) < 128, 1.0, -1.0)
    out = ft.filter_report(step, cutoff)
    assert 0.05 < out["overshoot_fraction"] < 0.15


def test_more_energy_survives_a_higher_cutoff():
    step = np.where(np.arange(256) < 128, 1.0, -1.0)
    kept = [ft.filter_report(step, c)["energy_kept"] for c in (0.02, 0.05, 0.1, 0.2, 0.4)]
    assert all(b > a for a, b in zip(kept, kept[1:]))


# --------------------------------------------------------------------------- Wiener


@pytest.mark.parametrize("noise", [0.05, 0.1, 0.3, 0.6, 1.0])
def test_the_wiener_filter_with_the_true_spectrum_is_the_best_of_the_four(noise):
    n = 512
    t = np.arange(n) / n
    clean = np.sin(2 * math.pi * 3 * t) + 0.4 * np.sin(2 * math.pi * 7 * t)
    out = ft.wiener_report(clean, noise, cutoff=10.0, rate=float(n),
                           rng=np.random.default_rng(7))
    assert out["true_wiener_is_best"]


@pytest.mark.parametrize("noise", [0.05, 0.1, 0.3, 0.6])
def test_the_practical_wiener_filter_still_beats_doing_nothing(noise):
    n = 512
    t = np.arange(n) / n
    clean = np.sin(2 * math.pi * 3 * t) + 0.4 * np.sin(2 * math.pi * 7 * t)
    out = ft.wiener_report(clean, noise, cutoff=10.0, rate=float(n),
                           rng=np.random.default_rng(7))
    assert out["estimated_beats_noisy"]
    assert out["improvement"] > 1.5


@pytest.mark.parametrize("noise", [0.1, 0.3])
def test_the_practical_wiener_filter_beats_a_brick_wall_on_a_broadband_signal(noise):
    n = 512
    t = np.arange(n) / n
    square = np.where(t < 0.5, 1.0, -1.0)
    out = ft.wiener_report(square, noise, cutoff=10.0, rate=float(n),
                           rng=np.random.default_rng(7))
    assert out["estimated_beats_brickwall"]


@pytest.mark.parametrize("noise", [0.1, 0.3])
def test_a_well_chosen_brick_wall_beats_the_practical_wiener_on_a_band_limited_signal(noise):
    """The honest negative result: given an oracle cutoff, a brick wall wins. Estimating the
    signal spectrum from the noisy data is what costs the Wiener filter that case."""
    n = 512
    t = np.arange(n) / n
    clean = np.sin(2 * math.pi * 3 * t) + 0.4 * np.sin(2 * math.pi * 7 * t)
    out = ft.wiener_report(clean, noise, cutoff=10.0, rate=float(n),
                           rng=np.random.default_rng(7))
    assert not out["estimated_beats_brickwall"]


def test_the_wiener_gain_lies_between_zero_and_one():
    n = 256
    g = np.random.default_rng(1)
    x = np.sin(2 * math.pi * 4 * np.arange(n) / n) + 0.2 * g.standard_normal(n)
    gain = ft.wiener(x, noise_power=0.04 * n)["gain"]
    assert np.all(gain >= 0.0) and np.all(gain <= 1.0)


def test_a_zero_noise_power_passes_the_signal_through():
    n = 64
    x = np.random.default_rng(2).standard_normal(n)
    got = ft.wiener(x, noise_power=0.0)["filtered"]
    assert np.max(np.abs(got - x)) < 1e-10 * max(float(np.max(np.abs(x))), 1.0)


def test_the_wiener_filter_requires_a_noise_estimate():
    with pytest.raises(ValueError, match="noise power spectrum is required"):
        ft.wiener(np.zeros(16))


@pytest.mark.parametrize("bad", [-1.0, -0.5])
def test_a_negative_power_is_rejected(bad):
    with pytest.raises(ValueError, match="noise power must be non-negative"):
        ft.wiener(np.zeros(16), noise_power=bad)
    with pytest.raises(ValueError, match="signal power must be non-negative"):
        ft.wiener(np.zeros(16), signal_power=bad, noise_power=1.0)
