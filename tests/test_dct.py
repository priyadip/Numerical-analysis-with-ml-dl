"""Tests for nalib.dct.

Three groups. The transform is checked against its defining matrix identity and against scipy
where scipy is available. The compression stages are checked against the properties that make
them work: quantization is the only lossy one, Huffman is prefix free and within one bit of the
entropy. The MDCT is checked by showing a single block is genuinely lossy and that overlapping
recovers the signal anyway, because that is the surprising part.

Every test sweeps block sizes and signal types.
"""
import math

import numpy as np
import pytest

from nalib import dct as dc

SIZES = [1, 2, 3, 4, 8, 16, 32, 64]


def signals(n, seed=0):
    t = np.arange(n) / n
    g = np.random.default_rng(seed)
    return {"ramp": t,
            "constant": np.ones(n),
            "smooth bump": np.exp(-((t - 0.5) ** 2) / (2 * 0.15 ** 2)),
            "tone": np.cos(2 * math.pi * 2 * t),
            "random": g.standard_normal(n)}


# --------------------------------------------------------------------------- the transform


@pytest.mark.parametrize("n", SIZES)
def test_the_matrix_is_orthonormal_with_condition_number_one(n):
    out = dc.orthonormality(n)
    assert out["is_orthonormal"]
    assert out["condition"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n", SIZES)
def test_the_round_trip_returns_the_block(n):
    for name, v in signals(n, n).items():
        assert dc.round_trip(v) < 1e-12, name


@pytest.mark.parametrize("n", SIZES)
def test_the_matrix_and_the_function_agree(n):
    v = signals(n, n)["random"]
    assert np.max(np.abs(dc.dct_matrix(n) @ v - dc.dct(v))) < 1e-12 * max(
        float(np.max(np.abs(v))), 1.0)


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_the_transform_matches_scipy_when_scipy_is_present(n):
    scipy_fft = pytest.importorskip("scipy.fft")
    v = signals(n, n)["random"]
    assert np.max(np.abs(dc.dct(v) - scipy_fft.dct(v, type=2, norm="ortho"))) < 1e-11


@pytest.mark.parametrize("n", SIZES)
def test_the_transform_of_a_constant_is_a_single_dc_term(n):
    X = dc.dct(np.ones(n))
    assert abs(X[0] - math.sqrt(n)) < 1e-10 * math.sqrt(n)
    if n > 1:
        assert float(np.max(np.abs(X[1:]))) < 1e-10 * math.sqrt(n)


@pytest.mark.parametrize("n", SIZES)
def test_the_transform_preserves_energy(n):
    v = signals(n, n)["random"]
    assert float(np.sum(dc.dct(v) ** 2)) == pytest.approx(float(np.sum(v ** 2)), rel=1e-11)


@pytest.mark.parametrize("shape", [(4, 4), (8, 8), (8, 16), (3, 5), (1, 7)])
def test_the_two_dimensional_round_trip_returns_the_block(shape):
    A = np.random.default_rng(sum(shape)).standard_normal(shape)
    assert dc.round_trip(A) < 1e-12


@pytest.mark.parametrize("shape", [(4, 4), (8, 8), (6, 10)])
def test_the_two_dimensional_transform_is_the_tensor_product(shape):
    """Rows then columns, which is lesson 53's separability, checked rather than assumed."""
    A = np.random.default_rng(sum(shape)).standard_normal(shape)
    rows = np.stack([dc.dct(A[i]) for i in range(A.shape[0])])
    both = np.stack([dc.dct(rows[:, j]) for j in range(A.shape[1])], axis=1)
    assert np.max(np.abs(both - dc.dct2(A))) < 1e-11 * max(float(np.max(np.abs(A))), 1.0)


def test_an_empty_block_is_rejected():
    with pytest.raises(ValueError, match="empty block"):
        dc.dct(np.zeros(0))
    with pytest.raises(ValueError, match="empty block"):
        dc.idct(np.zeros(0))


@pytest.mark.parametrize("n", [0, -3])
def test_a_matrix_of_non_positive_size_is_rejected(n):
    with pytest.raises(ValueError, match="positive size"):
        dc.dct_matrix(n)


# --------------------------------------------------------------------------- why DCT


@pytest.mark.parametrize("n", [32, 64, 128])
def test_the_even_extension_has_no_jump_and_the_periodic_one_does(n):
    out = dc.boundary_jump(n=n)
    assert out["even_extension_jump"] == 0.0
    assert out["periodic_extension_jump"] > 0.9


@pytest.mark.parametrize("n", [64, 128, 256])
def test_the_cosine_coefficients_decay_a_full_order_faster_on_a_ramp(n):
    out = dc.boundary_jump(n=n)
    assert out["dct_decay_order"] > out["dft_decay_order"] + 0.8
    assert 0.7 < out["dft_decay_order"] < 1.3
    assert 1.7 < out["dct_decay_order"] < 2.5


@pytest.mark.parametrize("n", [64, 128])
def test_the_cosine_tail_carries_far_less_energy_on_a_ramp(n):
    out = dc.boundary_jump(n=n)
    assert out["dct_tail_energy"] < 1e-3 * out["dft_tail_energy"]


def test_the_two_transforms_are_tied_when_the_block_ends_already_match():
    """A bump whose ends are both near zero has no seam, so there is nothing for the even
    extension to fix and the DCT's advantage disappears."""
    n = 64
    t = np.arange(n) / n
    bump = np.exp(-((t - 0.5) ** 2) / (2 * 0.15 ** 2))
    out = dc.energy_compaction(bump, (4,))
    assert abs(out["dct_energy_fraction"][0] - out["dft_energy_fraction"][0]) < 1e-3


def test_the_cosine_transform_wins_decisively_on_a_ramp():
    n = 64
    out = dc.energy_compaction(np.arange(n) / n, (4, 8, 16))
    assert np.all(out["dct_energy_fraction"] > out["dft_energy_fraction"])
    assert out["dct_energy_fraction"][0] > 0.99
    assert out["dft_energy_fraction"][0] < 0.95


def test_neither_transform_compacts_random_data():
    n = 64
    out = dc.energy_compaction(np.random.default_rng(4).standard_normal(n), (4, 8))
    assert np.all(out["dct_energy_fraction"] < 0.3)
    assert np.all(out["dft_energy_fraction"] < 0.3)


@pytest.mark.parametrize("n", [8, 16, 32])
def test_keeping_everything_keeps_all_the_energy(n):
    for name, v in signals(n, n).items():
        out = dc.energy_compaction(v, (n,))
        assert out["dct_energy_fraction"][0] == pytest.approx(1.0, abs=1e-9), name
        assert out["dft_energy_fraction"][0] == pytest.approx(1.0, abs=1e-9), name


@pytest.mark.parametrize("rho", [0.5, 0.9, 0.95, 0.99])
@pytest.mark.parametrize("n", [4, 8, 16])
def test_the_dct_is_close_to_the_optimal_transform(rho, n):
    out = dc.against_kl(rho, n)
    assert out["worst_shortfall"] >= -1e-12
    assert out["worst_shortfall"] < 0.02


@pytest.mark.parametrize("rho", [0.9, 0.95])
def test_the_kl_transform_is_never_beaten_which_is_what_optimal_means(rho):
    out = dc.against_kl(rho, 8)
    assert np.all(out["kl_fraction"] >= out["dct_fraction"] - 1e-12)


@pytest.mark.parametrize("bad", [-1.0, 1.0, 1.5])
def test_an_invalid_correlation_is_rejected(bad):
    with pytest.raises(ValueError, match="strictly in"):
        dc.against_kl(bad, 8)


# --------------------------------------------------------------------------- quantization


@pytest.mark.parametrize("step", [0.001, 0.01, 0.1, 1.0])
@pytest.mark.parametrize("n", [8, 16, 64])
def test_quantization_error_never_exceeds_half_a_step(step, n):
    c = dc.dct(signals(n, n)["random"])
    back = dc.dequantize(dc.quantize(c, step), step)
    assert float(np.max(np.abs(back - c))) <= 0.5 * step + 1e-12


@pytest.mark.parametrize("n", [16, 64])
def test_a_finer_step_gives_a_smaller_error_and_fewer_zeros(n):
    out = dc.quantization_report(signals(n, n)["smooth bump"],
                                 (0.0001, 0.001, 0.01, 0.1, 1.0))
    assert all(b > a for a, b in zip(out["relative_rms_error"], out["relative_rms_error"][1:]))
    assert all(b >= a for a, b in zip(out["zero_fraction"], out["zero_fraction"][1:]))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_step_is_rejected(bad):
    with pytest.raises(ValueError, match="must be positive"):
        dc.quantize(np.ones(8), bad)


@pytest.mark.parametrize("quality", [1, 25, 50, 75, 100])
def test_the_jpeg_table_is_the_right_shape_and_stays_in_range(quality):
    table = dc.jpeg_luminance_table(quality)
    assert table.shape == (8, 8)
    assert np.all(table >= 1.0) and np.all(table <= 255.0)


def test_a_higher_quality_gives_a_finer_table():
    coarse = dc.jpeg_luminance_table(10)
    fine = dc.jpeg_luminance_table(90)
    assert float(np.mean(fine)) < float(np.mean(coarse))


def test_the_table_is_coarser_at_high_frequencies():
    table = dc.jpeg_luminance_table(50)
    assert table[7, 7] > table[0, 0]


@pytest.mark.parametrize("bad", [0, 101, -5])
def test_an_invalid_quality_is_rejected(bad):
    with pytest.raises(ValueError, match="quality must be"):
        dc.jpeg_luminance_table(bad)


# --------------------------------------------------------------------------- entropy coding


@pytest.mark.parametrize("n", [8, 64, 256])
def test_a_uniform_alphabet_has_entropy_log_two_of_its_size(n):
    symbols = list(range(n))
    assert dc.entropy(symbols) == pytest.approx(math.log2(n), abs=1e-12)


def test_a_single_repeated_symbol_has_zero_entropy():
    assert dc.entropy([7] * 100) == pytest.approx(0.0, abs=1e-12)


def test_measuring_the_entropy_of_nothing_is_rejected():
    with pytest.raises(ValueError, match="entropy of nothing"):
        dc.entropy([])


def test_building_a_code_for_nothing_is_rejected():
    with pytest.raises(ValueError, match="code for nothing"):
        dc.huffman([])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_code_is_prefix_free(seed):
    g = np.random.default_rng(seed)
    symbols = g.integers(-8, 9, 500).tolist()
    assert dc.is_prefix_free(dc.huffman(symbols)["code"])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_code_is_within_one_bit_of_the_entropy(seed):
    g = np.random.default_rng(seed)
    symbols = g.integers(-8, 9, 2000).tolist()
    out = dc.huffman_report(symbols)
    assert out["within_one_bit"]
    assert out["bits_per_symbol"] >= out["entropy"] - 1e-9


def test_a_skewed_distribution_compresses_and_a_uniform_one_barely_does():
    g = np.random.default_rng(9)
    skewed = (g.random(4000) < 0.95).astype(int).tolist()
    uniform = g.integers(0, 256, 4000).tolist()
    assert dc.huffman_report(skewed)["bits_per_symbol"] < 1.5
    assert dc.huffman_report(uniform)["bits_per_symbol"] > 7.0


def test_a_single_symbol_alphabet_gets_a_one_bit_code():
    out = dc.huffman_report([3] * 50)
    assert out["bits_per_symbol"] == 1.0
    assert out["entropy"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- pipelines


@pytest.mark.parametrize("n", [16, 64, 128])
@pytest.mark.parametrize("step", [0.01, 0.1, 1.0])
def test_the_one_dimensional_pipeline_loses_nothing_in_the_transform(n, step):
    out = dc.compress_1d(signals(n, n)["smooth bump"], step)
    assert out["transform_loss"] < 1e-12


@pytest.mark.parametrize("n", [64, 128])
def test_a_coarser_step_compresses_more_and_costs_more_error(n):
    v = signals(n, n)["smooth bump"]
    fine = dc.compress_1d(v, 0.001)
    coarse = dc.compress_1d(v, 0.5)
    assert coarse["compression"] > fine["compression"]
    assert coarse["relative_rms_error"] > fine["relative_rms_error"]


@pytest.mark.parametrize("quality", [10, 50, 90])
def test_the_image_pipeline_runs_and_reconstructs_something_close(quality):
    g = np.random.default_rng(quality)
    x, y = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
    image = 128.0 + 100.0 * np.sin(4 * x) * np.cos(3 * y)
    out = dc.compress_2d(image, quality)
    assert out["reconstructed"].shape == image.shape
    assert out["relative_rms_error"] < 0.2
    assert out["blocks"] == 16


def test_a_higher_quality_gives_a_smaller_error_and_a_larger_file():
    x, y = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
    image = 128.0 + 100.0 * np.sin(6 * x) * np.cos(5 * y)
    low = dc.compress_2d(image, 10)
    high = dc.compress_2d(image, 90)
    assert high["relative_rms_error"] < low["relative_rms_error"]
    assert high["bits"] > low["bits"]
    assert low["compression"] > high["compression"]


@pytest.mark.parametrize("shape", [(16, 16), (17, 23), (8, 8)])
def test_the_image_pipeline_handles_sizes_that_do_not_divide_the_block(shape):
    g = np.random.default_rng(sum(shape))
    image = 128.0 + 20.0 * g.standard_normal(shape)
    out = dc.compress_2d(image, 50)
    assert out["reconstructed"].shape == shape


@pytest.mark.parametrize("bad", [0, -2])
def test_a_non_positive_block_size_is_rejected(bad):
    with pytest.raises(ValueError, match="block size must be positive"):
        dc.compress_2d(np.zeros((8, 8)), 50, bad)


# --------------------------------------------------------------------------- MDCT


@pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
def test_the_window_satisfies_the_princen_bradley_condition(n):
    w = dc.sine_window(n)
    assert w.size == 2 * n
    assert float(np.max(np.abs(w[:n] ** 2 + w[n:] ** 2 - 1.0))) < 1e-14


@pytest.mark.parametrize("n", [0, -4])
def test_a_non_positive_half_length_is_rejected(n):
    with pytest.raises(ValueError, match="positive half length"):
        dc.sine_window(n)


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_the_mdct_halves_the_count(n):
    block = np.random.default_rng(n).standard_normal(2 * n)
    assert dc.mdct(block).size == n


@pytest.mark.parametrize("n", [4, 8, 16])
def test_a_single_block_is_genuinely_lossy(n):
    """It must be. It maps 2n numbers to n, so it cannot be invertible, and the tests should say
    so rather than only checking that the overlapped version works."""
    out = dc.tdac_report(n, rng=np.random.default_rng(n))
    assert out["single_block_is_lossy"]
    assert out["single_block_error"] > 1e-3


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_overlapping_and_adding_reconstructs_exactly(n):
    out = dc.tdac_report(n, rng=np.random.default_rng(n))
    assert out["overlap_added_error"] < 1e-12
    assert out["princen_bradley"] < 1e-14


@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("length", [40, 64, 100, 127])
def test_the_round_trip_works_at_any_signal_length(n, length):
    x = np.random.default_rng(n + length).standard_normal(length)
    out = dc.mdct_roundtrip(x, n)
    assert out["relative_error"] < 1e-11


@pytest.mark.parametrize("n", [8, 16, 32])
def test_the_lapped_transform_stores_about_one_coefficient_per_sample(n):
    x = np.random.default_rng(n).standard_normal(16 * n)
    out = dc.mdct_roundtrip(x, n)
    assert 1.0 <= out["expansion"] < 1.3


def test_an_odd_block_is_rejected():
    with pytest.raises(ValueError, match="even number of samples"):
        dc.mdct(np.zeros(7))


def test_a_mismatched_window_is_rejected():
    with pytest.raises(ValueError, match="window of"):
        dc.mdct(np.zeros(8), window=np.ones(6))
    with pytest.raises(ValueError, match="window of"):
        dc.imdct(np.zeros(4), window=np.ones(6))
