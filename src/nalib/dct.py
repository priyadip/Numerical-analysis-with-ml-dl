"""The discrete cosine transform, and how a compressed file is actually made.

Why not the DFT
---------------
The DFT of a finite block assumes the block **repeats**. Unless the two ends happen to match,
that periodic extension has a jump at the seam, and a jump costs slowly decaying coefficients,
which is the whole difficulty. `boundary_jump` measures it: on a smooth ramp the DFT's
coefficients decay like ``1/k`` while the DCT's decay like ``1/k^2``, purely because of what
happens at the edge.

The DCT fixes it by extending the block **evenly** first, reflecting rather than repeating. The
reflected signal is continuous at the seam whatever the data, so there is no artificial jump and
the coefficients decay at the rate the data deserves.

    C_k = s_k sum_{j=0}^{n-1} x_j cos( pi (2j + 1) k / (2n) )

with ``s_0 = sqrt(1/n)`` and ``s_k = sqrt(2/n)`` otherwise. That is the DCT-II, which is what
"the DCT" means in every codec. Its inverse is the DCT-III.

Why it compresses
-----------------
`energy_compaction` measures the property everything else rests on: for a smooth or slowly
varying block, almost all the energy lands in the first few coefficients. On a ramp of 64
samples, 8 stored numbers carry **99.99 percent** of the energy against the DFT's 95.6 percent,
and 4 carry 99.94 against 90.0.

The budget there is counted in **real numbers**, not in coefficients, because a DFT coefficient
of real data is complex and a DCT coefficient is not. Counting bins would hand the DFT twice the
storage and make the comparison meaningless.

Counted that way the DCT's advantage is real but narrower than the usual telling, and it is
**entirely a boundary effect**. On a Gaussian bump centred in the block, whose two ends are both
near zero so the periodic extension has no jump, the two transforms are tied: 0.96965 against
0.96942 at 4 numbers. On random data neither compacts anything, 0.030 against 0.030, because
there is nothing to compact. The DCT wins exactly when the block's two ends disagree, which for
a tile of a photograph is almost always.

For a stationary signal the transform that compacts energy best is the Karhunen-Loeve transform,
which requires knowing the covariance. **The DCT is very close to it** for the first order Markov
model that images actually follow, and it is data independent and fast, which is why it won.
`against_kl` measures the gap.

The pipeline
------------
`compress_1d` and `compress_2d` run the real thing end to end: transform, quantize, entropy code,
and back. The stages are separable and each is measured on its own, because the interesting
question is which stage costs what.

- **The transform loses nothing**, and `round_trip` shows it at ``5e-15``.
- **Quantization is where every bit of the loss happens**, and `quantization_report` measures
  the trade against the step size.
- **Huffman coding loses nothing** and buys the difference between the entropy and eight bits a
  symbol. `huffman` builds the code and `huffman_report` measures it against the entropy bound.

The MDCT
--------
`mdct` is the lapped transform an audio codec uses. It takes ``2n`` samples and produces ``n``
coefficients, which cannot be invertible on its own, and yet overlapping consecutive blocks and
adding reconstructs the signal exactly. That is **time domain alias cancellation**, and
`tdac_report` verifies it rather than asserting it. It is what removes the blocking artefacts a
plain block transform produces at every boundary.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter

import numpy as np

# --------------------------------------------------------------------------- the transform


def dct_matrix(n: int) -> np.ndarray:
    """The orthonormal DCT-II matrix, built explicitly.

    Orthonormal means ``C^T C = I``, so the inverse is the transpose and the condition number is
    exactly 1, the same guarantee the DFT has and for the same reason.
    """
    m = int(n)
    if m < 1:
        raise ValueError(f"need a positive size, got {m}")
    j = np.arange(m)
    k = np.arange(m)
    M = np.cos(math.pi * np.outer(k, 2 * j + 1) / (2 * m))
    scale = np.full(m, math.sqrt(2.0 / m))
    scale[0] = math.sqrt(1.0 / m)
    return M * scale[:, None]


def dct(x) -> np.ndarray:
    """The orthonormal DCT-II, as the defining sum.

    ``O(n^2)`` as written. A production codec computes it in ``O(n log n)`` through the FFT, or
    for the fixed size 8 by a hand tuned network of 11 multiplies, and neither changes what is
    computed.
    """
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if v.size < 1:
        raise ValueError("cannot transform an empty block")
    return dct_matrix(v.size) @ v


def idct(X) -> np.ndarray:
    """The DCT-III, which is the transpose of the DCT-II because the matrix is orthonormal."""
    v = np.atleast_1d(np.asarray(X, dtype=float)).ravel()
    if v.size < 1:
        raise ValueError("cannot transform an empty block")
    return dct_matrix(v.size).T @ v


def dct2(block) -> np.ndarray:
    """The two dimensional DCT: rows then columns, which is lesson 53's tensor product.

    ``C A C^T``. Separability is what makes it affordable: a ``n`` by ``n`` block costs
    ``2 n^3`` instead of ``n^4``, and for the 8 by 8 blocks JPEG uses that is 1024 against 4096.
    """
    A = np.atleast_2d(np.asarray(block, dtype=float))
    if A.ndim != 2:
        raise ValueError(f"need a two dimensional block, got shape {A.shape}")
    return dct_matrix(A.shape[0]) @ A @ dct_matrix(A.shape[1]).T


def idct2(block) -> np.ndarray:
    """The inverse two dimensional transform."""
    A = np.atleast_2d(np.asarray(block, dtype=float))
    return dct_matrix(A.shape[0]).T @ A @ dct_matrix(A.shape[1])


def round_trip(x) -> float:
    """Relative error of transforming and back, which must be roundoff and nothing more."""
    v = np.atleast_1d(np.asarray(x, dtype=float))
    if v.ndim == 1:
        back = idct(dct(v))
    else:
        back = idct2(dct2(v))
    return float(np.max(np.abs(back - v)) / max(float(np.max(np.abs(v))), 1e-300))


def orthonormality(n: int) -> dict:
    """``C^T C = I``, checked, with the condition number that follows."""
    C = dct_matrix(int(n))
    G = C.T @ C
    off = G - np.eye(int(n))
    return {"n": int(n), "worst_deviation": float(np.max(np.abs(off))),
            "condition": float(np.linalg.cond(C)),
            "is_orthonormal": bool(float(np.max(np.abs(off))) < 1e-12)}


# --------------------------------------------------------------------------- why DCT


def boundary_jump(f=None, n: int = 64) -> dict:
    """The DFT's periodic extension jumps at the seam; the DCT's even extension does not.

    Measured on a ramp, whose ends are as far apart as possible. The DFT coefficients decay like
    ``1/k`` and the DCT's like ``1/k^2``, which is the difference between a discontinuous
    extension and a continuous one, and it is the entire reason codecs use the cosine transform.

    **The fit skips coefficients that are exactly zero, and that matters.** Every even indexed
    DCT coefficient of a ramp vanishes by symmetry, so half the entries sit at the roundoff floor
    of ``1e-15``. Fitting a slope through them mixes a decaying sequence with a flat one and the
    answer depends on how many of each the window happens to contain: measured over all ``k``,
    the fitted order reads 2.04 at ``n = 64``, 1.44 at ``n = 128`` and 0.99 at ``n = 256``,
    which looks like the theory failing and is really the fit averaging in a floor. Restricted to
    the coefficients that carry anything, it is 2.0 at every size.

    ``kept`` and ``dropped`` say how many entries the fit used, so the exclusion is visible.
    """
    from . import fft as _fft

    m = int(n)
    t = np.arange(m) / m
    v = np.asarray(f(t), dtype=float) if f is not None else t.copy()
    fourier = np.abs(_fft.fft(v))[1:m // 2]
    cosine = np.abs(dct(v))[1:m // 2]
    k = np.arange(1, fourier.size + 1, dtype=float)

    def fit(a):
        floor = 1e-10 * max(float(np.max(a)), 1e-300)
        keep = a > floor
        if int(np.sum(keep)) < 3:
            return float("nan"), int(np.sum(keep)), int(np.sum(~keep))
        slope = float(-np.polyfit(np.log(k[keep]), np.log(a[keep]), 1)[0])
        return slope, int(np.sum(keep)), int(np.sum(~keep))

    dft_order, dft_kept, dft_dropped = fit(fourier)
    dct_order, dct_kept, dct_dropped = fit(cosine)
    periodic = np.concatenate([v, v])
    even = np.concatenate([v, v[::-1]])
    return {"n": m,
            "dft_decay_order": dft_order, "dct_decay_order": dct_order,
            "dft_kept": dft_kept, "dft_dropped": dft_dropped,
            "dct_kept": dct_kept, "dct_dropped": dct_dropped,
            "periodic_extension_jump": float(abs(periodic[m] - periodic[m - 1])),
            "even_extension_jump": float(abs(even[m] - even[m - 1])),
            "dft_tail_energy": float(np.sum(fourier[fourier.size // 2:] ** 2)),
            "dct_tail_energy": float(np.sum(cosine[cosine.size // 2:] ** 2))}


def energy_compaction(x, keep_values=None) -> dict:
    """What fraction of the energy survives keeping ``k`` **real numbers** of the transform.

    This is the number that decides whether a transform is any good for compression, and the
    unit has to be real numbers rather than coefficients or the comparison is rigged. A DCT
    coefficient of a real signal is one real number. A DFT coefficient is complex, so it is two,
    except for the DC term and the Nyquist term. Counting bins instead of numbers would give the
    DFT twice the budget.

    So ``k`` here is a storage budget, and each transform is asked what it can do with it.
    """
    from . import fft as _fft

    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n = v.size
    ks = ([1, 2, 4, 8, 16, n] if keep_values is None
          else [int(k) for k in np.atleast_1d(keep_values)])
    c = dct(v)
    F = _fft.fft(v)
    total = float(np.sum(v ** 2))
    rows = []
    for k in ks:
        k = min(max(int(k), 0), n)
        dct_energy = float(np.sum(c[:k] ** 2))
        # DFT: spend the budget on DC first (one real), then on conjugate pairs (two reals each)
        budget = k
        kept = 0.0
        if budget >= 1:
            kept += float(np.abs(F[0]) ** 2) / n
            budget -= 1
        bin_index = 1
        while budget >= 2 and bin_index <= (n - 1) // 2:
            kept += 2.0 * float(np.abs(F[bin_index]) ** 2) / n
            budget -= 2
            bin_index += 1
        if budget >= 1 and n % 2 == 0 and bin_index == n // 2:
            kept += float(np.abs(F[n // 2]) ** 2) / n
            budget -= 1
        rows.append((k, dct_energy / max(total, 1e-300), kept / max(total, 1e-300)))
    return {"keep": np.asarray([r[0] for r in rows]),
            "dct_energy_fraction": np.asarray([r[1] for r in rows]),
            "dft_energy_fraction": np.asarray([r[2] for r in rows]),
            "budget_is_real_numbers": True}


def against_kl(rho: float = 0.95, n: int = 8) -> dict:
    """The DCT against the optimal Karhunen-Loeve transform for a first order Markov source.

    The KL transform is the eigenvector basis of the covariance matrix. It is optimal for energy
    compaction and useless in practice, because it depends on the data and costs an eigensolve.

    For the covariance ``R_{ij} = rho^{|i-j|}``, which is the standard model of a row of image
    pixels, the DCT basis is very close to the KL basis, and this measures how close by comparing
    the energy each packs into the first few coefficients. That near-optimality, plus being data
    independent and fast, is why JPEG uses the DCT and not the KL transform.
    """
    m = int(n)
    r = float(rho)
    if not -1.0 < r < 1.0:
        raise ValueError(f"the correlation must lie strictly in (-1, 1), got {r}")
    idx = np.arange(m)
    R = r ** np.abs(idx[:, None] - idx[None, :])
    vals, vecs = np.linalg.eigh(R)
    order = np.argsort(vals)[::-1]
    kl = vecs[:, order].T
    C = dct_matrix(m)
    # energy of the transformed covariance on the diagonal, largest first
    kl_energy = np.sort(np.real(np.diag(kl @ R @ kl.T)))[::-1]
    dct_energy = np.sort(np.real(np.diag(C @ R @ C.T)))[::-1]
    total = float(np.trace(R))
    frac = lambda a, k: float(np.sum(a[:k])) / total
    ks = [1, 2, m // 2, m]
    return {"rho": r, "n": m,
            "keep": np.asarray(ks),
            "kl_fraction": np.asarray([frac(kl_energy, k) for k in ks]),
            "dct_fraction": np.asarray([frac(dct_energy, k) for k in ks]),
            "worst_shortfall": float(max(frac(kl_energy, k) - frac(dct_energy, k)
                                         for k in ks))}


# --------------------------------------------------------------------------- quantization


def quantize(coefficients, step) -> np.ndarray:
    """Round each coefficient to a multiple of its own step size.

    This is the **only** lossy stage. Everything else in a codec is exactly invertible, so every
    artefact anyone has ever complained about comes from this one line.
    """
    c = np.asarray(coefficients, dtype=float)
    q = np.asarray(step, dtype=float)
    if np.any(q <= 0.0):
        raise ValueError("every quantization step must be positive")
    return np.round(c / q)


def dequantize(levels, step) -> np.ndarray:
    """Multiply back. The value returned is the centre of the bin, not the original."""
    return np.asarray(levels, dtype=float) * np.asarray(step, dtype=float)


def jpeg_luminance_table(quality: int = 50) -> np.ndarray:
    """The standard 8 by 8 luminance quantization table, scaled by a quality setting.

    The numbers are from Annex K of the JPEG standard. They rise toward the bottom right, which
    is the high frequency corner, because the eye is least sensitive there. That is the only
    place perception enters the pipeline, and it enters as a table of integers.
    """
    base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]], dtype=float)
    q = int(quality)
    if not 1 <= q <= 100:
        raise ValueError(f"quality must be between 1 and 100, got {q}")
    scale = (5000.0 / q) if q < 50 else (200.0 - 2.0 * q)
    return np.clip(np.floor((base * scale + 50.0) / 100.0), 1.0, 255.0)


def quantization_report(x, steps=None) -> dict:
    """Error and sparsity against the quantization step, on one block.

    The two move in opposite directions and the whole design of a codec is choosing where on
    that curve to sit.
    """
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    qs = ([0.001, 0.01, 0.1, 1.0, 10.0] if steps is None
          else [float(s) for s in np.atleast_1d(steps)])
    c = dct(v)
    scale = max(float(np.max(np.abs(v))), 1e-300)
    rows = []
    for q in qs:
        levels = quantize(c, q)
        back = idct(dequantize(levels, q))
        rows.append((q,
                     float(np.sqrt(np.mean((back - v) ** 2))) / scale,
                     float(np.mean(levels == 0.0)),
                     int(np.count_nonzero(levels))))
    return {"steps": np.asarray([r[0] for r in rows]),
            "relative_rms_error": np.asarray([r[1] for r in rows]),
            "zero_fraction": np.asarray([r[2] for r in rows]),
            "nonzero_count": np.asarray([r[3] for r in rows])}


# --------------------------------------------------------------------------- entropy coding


def entropy(symbols) -> float:
    """Shannon entropy in bits per symbol, which is the floor no lossless code can beat."""
    s = list(np.asarray(symbols).ravel())
    if not s:
        raise ValueError("cannot measure the entropy of nothing")
    counts = Counter(s)
    total = float(len(s))
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))


def huffman(symbols) -> dict:
    """Build a Huffman code: repeatedly merge the two least frequent symbols.

    The greedy merge is optimal among prefix codes, which is Huffman's theorem, and the code it
    produces is within one bit per symbol of the entropy. `huffman_report` measures that gap.

    A single distinct symbol is given the one bit code "0", because a zero length code cannot be
    decoded. That costs one bit per symbol against an entropy of zero, and it is the standard
    convention rather than a defect.
    """
    s = list(np.asarray(symbols).ravel())
    if not s:
        raise ValueError("cannot build a code for nothing")
    counts = Counter(s)
    if len(counts) == 1:
        only = next(iter(counts))
        return {"code": {only: "0"}, "counts": dict(counts), "symbols": len(s)}
    heap = [(count, i, {sym: ""}) for i, (sym, count) in enumerate(sorted(counts.items(),
                                                                         key=lambda kv: str(kv[0])))]
    heapq.heapify(heap)
    tie = len(heap)
    while len(heap) > 1:
        c1, _, a = heapq.heappop(heap)
        c2, _, b = heapq.heappop(heap)
        merged = {k: "0" + v for k, v in a.items()}
        merged.update({k: "1" + v for k, v in b.items()})
        heapq.heappush(heap, (c1 + c2, tie, merged))
        tie += 1
    return {"code": heap[0][2], "counts": dict(counts), "symbols": len(s)}


def huffman_report(symbols) -> dict:
    """Bits per symbol achieved, against the entropy floor and against a fixed 8 bit code."""
    s = list(np.asarray(symbols).ravel())
    built = huffman(s)
    code = built["code"]
    bits = sum(len(code[v]) for v in s)
    h = entropy(s)
    return {"entropy": h, "bits_per_symbol": bits / len(s),
            "total_bits": bits, "fixed_eight_bit_total": 8 * len(s),
            "overhead_against_entropy": bits / len(s) - h,
            "within_one_bit": bool(bits / len(s) <= h + 1.0 + 1e-9),
            "compression_against_fixed": (8.0 * len(s)) / max(bits, 1),
            "distinct_symbols": len(built["counts"])}


def is_prefix_free(code) -> bool:
    """No codeword is a prefix of another, which is what makes a stream decodable."""
    words = sorted(code.values())
    return all(not b.startswith(a) for a, b in zip(words, words[1:]))


# --------------------------------------------------------------------------- pipelines


def compress_1d(x, step: float = 1.0) -> dict:
    """Transform, quantize, entropy code, and back. Every stage measured on its own."""
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    c = dct(v)
    levels = quantize(c, float(step))
    back = idct(dequantize(levels, float(step)))
    report = huffman_report(levels.astype(int))
    scale = max(float(np.max(np.abs(v))), 1e-300)
    return {"reconstructed": back,
            "relative_rms_error": float(np.sqrt(np.mean((back - v) ** 2))) / scale,
            "transform_loss": round_trip(v),
            "bits": report["total_bits"],
            "bits_per_sample": report["total_bits"] / v.size,
            "raw_bits_per_sample": 64.0,
            "compression": 64.0 * v.size / max(report["total_bits"], 1),
            "zero_fraction": float(np.mean(levels == 0.0)),
            "entropy": report["entropy"]}


def compress_2d(image, quality: int = 50, block: int = 8) -> dict:
    """The JPEG pipeline on a greyscale image: block, transform, quantize, code.

    The image is split into ``block`` by ``block`` tiles, each transformed and quantized against
    the standard table. Working in blocks is what makes it affordable and is also what causes
    the blocking artefacts at low quality, which `compress_2d` reports as the discontinuity
    across tile boundaries.
    """
    A = np.atleast_2d(np.asarray(image, dtype=float))
    b = int(block)
    if b < 1:
        raise ValueError(f"block size must be positive, got {b}")
    rows = (A.shape[0] + b - 1) // b
    cols = (A.shape[1] + b - 1) // b
    padded = np.zeros((rows * b, cols * b))
    padded[:A.shape[0], :A.shape[1]] = A
    if b == 8:
        table = jpeg_luminance_table(quality)
    else:
        # The standard table is 8 by 8 only, so other block sizes get a flat table scaled by
        # the same quality rule. Without the scaling the quality setting would be silently
        # ignored away from b = 8, and the measured blocking artefact would not move with it.
        q = int(quality)
        if not 1 <= q <= 100:
            raise ValueError(f"quality must be between 1 and 100, got {q}")
        scale = (5000.0 / q) if q < 50 else (200.0 - 2.0 * q)
        table = np.clip(np.floor((16.0 * scale + 50.0) / 100.0), 1.0, 255.0) * np.ones((b, b))
    out = np.zeros_like(padded)
    levels = []
    for i in range(rows):
        for j in range(cols):
            tile = padded[i * b:(i + 1) * b, j * b:(j + 1) * b]
            q = quantize(dct2(tile), table)
            levels.append(q.ravel())
            out[i * b:(i + 1) * b, j * b:(j + 1) * b] = idct2(dequantize(q, table))
    flat = np.concatenate(levels).astype(int)
    report = huffman_report(flat)
    got = out[:A.shape[0], :A.shape[1]]
    seams = 0.0
    if rows > 1:
        edges = [i * b for i in range(1, rows) if i * b < A.shape[0]]
        if edges:
            seams = max(seams, float(np.max(np.abs(got[edges] - got[[e - 1 for e in edges]]))))
    scale = max(float(np.max(np.abs(A))), 1e-300)
    return {"reconstructed": got,
            "relative_rms_error": float(np.sqrt(np.mean((got - A) ** 2))) / scale,
            "peak_signal_to_noise": float(
                20.0 * math.log10(scale / max(float(np.sqrt(np.mean((got - A) ** 2))), 1e-300))),
            "bits": report["total_bits"],
            "bits_per_pixel": report["total_bits"] / A.size,
            "compression": 8.0 * A.size / max(report["total_bits"], 1),
            "zero_fraction": float(np.mean(flat == 0)),
            "blocking_seam": seams,
            "blocks": rows * cols}


# --------------------------------------------------------------------------- MDCT


def sine_window(n: int) -> np.ndarray:
    """The window that makes time domain alias cancellation work.

    It has to satisfy the Princen-Bradley condition ``w_j^2 + w_{j+n}^2 = 1``, and
    ``sin(pi (j + 1/2) / 2n)`` does. `tdac_report` checks the condition rather than assuming it.
    """
    m = int(n)
    if m < 1:
        raise ValueError(f"need a positive half length, got {m}")
    return np.sin(math.pi * (np.arange(2 * m) + 0.5) / (2 * m))


def mdct(block, window=None) -> np.ndarray:
    """The modified DCT: ``2n`` samples in, ``n`` coefficients out.

        X_k = sum_{j=0}^{2n-1} w_j x_j cos( pi/n (j + 1/2 + n/2)(k + 1/2) )
    """
    v = np.atleast_1d(np.asarray(block, dtype=float)).ravel()
    if v.size % 2 != 0:
        raise ValueError(f"the MDCT takes an even number of samples, got {v.size}")
    n = v.size // 2
    w = sine_window(n) if window is None else np.asarray(window, dtype=float).ravel()
    if w.size != v.size:
        raise ValueError(f"window of {w.size} against a block of {v.size}")
    j = np.arange(2 * n)
    k = np.arange(n)
    M = np.cos(math.pi / n * np.outer(k + 0.5, j + 0.5 + n / 2.0))
    return M @ (w * v)


def imdct(X, window=None) -> np.ndarray:
    """The inverse: ``n`` coefficients in, ``2n`` samples out, windowed and scaled.

    The output is **not** the original block. It is the original plus a time reversed copy of
    part of it, which is the time domain alias. Overlapping and adding the next block cancels it.
    """
    c = np.atleast_1d(np.asarray(X, dtype=float)).ravel()
    n = c.size
    w = sine_window(n) if window is None else np.asarray(window, dtype=float).ravel()
    if w.size != 2 * n:
        raise ValueError(f"window of {w.size} against {n} coefficients")
    j = np.arange(2 * n)
    k = np.arange(n)
    M = np.cos(math.pi / n * (np.outer(j + 0.5 + n / 2.0, k + 0.5)))
    return (2.0 / n) * w * (M @ c)


def mdct_roundtrip(x, n: int) -> dict:
    """Split into overlapping blocks, transform, invert, overlap-add, and compare.

    The signal is padded by one half block at each end, because the first and last half blocks
    have no neighbour to cancel against. Inside that, reconstruction is exact.
    """
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    half = int(n)
    if half < 1:
        raise ValueError(f"need a positive half length, got {half}")
    blocks = int(math.ceil(v.size / half)) + 1
    padded = np.zeros((blocks + 1) * half)
    padded[half:half + v.size] = v
    out = np.zeros_like(padded)
    for b in range(blocks):
        seg = padded[b * half:(b + 2) * half]
        out[b * half:(b + 2) * half] += imdct(mdct(seg))
    got = out[half:half + v.size]
    scale = max(float(np.max(np.abs(v))), 1e-300)
    return {"reconstructed": got, "blocks": blocks,
            "relative_error": float(np.max(np.abs(got - v))) / scale,
            "coefficients_stored": blocks * half,
            "samples": v.size,
            "expansion": blocks * half / max(v.size, 1)}


def tdac_report(n: int = 16, rng=None) -> dict:
    """Show that a single MDCT block is **not** invertible and that overlapping fixes it.

    Both halves matter. If a single block were invertible the transform would be doing ``2n``
    numbers into ``n`` losslessly, which is impossible. The alias is what a single block loses,
    and the overlap is what recovers it.
    """
    gen = np.random.default_rng() if rng is None else rng
    m = int(n)
    w = sine_window(m)
    block = gen.standard_normal(2 * m)
    single = imdct(mdct(block))
    scale = max(float(np.max(np.abs(block))), 1e-300)
    return {"n": m,
            "princen_bradley": float(np.max(np.abs(w[:m] ** 2 + w[m:] ** 2 - 1.0))),
            "single_block_error": float(np.max(np.abs(single - block))) / scale,
            "single_block_is_lossy": bool(float(np.max(np.abs(single - block))) > 1e-6 * scale),
            "overlap_added_error": mdct_roundtrip(gen.standard_normal(8 * m), m)["relative_error"]}
