"""The fast Fourier transform, and what having one makes possible.

The idea in one line
--------------------
Split the sum by the parity of the index. With ``n`` even,

    X_k = sum_{j even} x_j w^{jk}  +  sum_{j odd} x_j w^{jk}
        = E_k  +  w^k O_k

where ``E`` and ``O`` are the ``n/2`` point transforms of the even and odd entries. The saving is
that ``E`` and ``O`` are periodic with period ``n/2``, so the **same** two half transforms give
both ``X_k`` and ``X_{k+n/2}``:

    X_k          =  E_k + w^k O_k
    X_{k+n/2}    =  E_k - w^k O_k

because ``w^{k+n/2} = -w^k``. Two outputs from one multiplication. That is the butterfly, and
recursing gives ``T(n) = 2 T(n/2) + O(n)``, so ``T(n) = O(n log n)``.

The cost, exactly
-----------------
A radix-2 transform of length ``n = 2^m`` does ``(n/2) log2(n)`` butterflies, each one complex
multiply and two complex adds. `operation_count` counts them by instrumenting the code rather
than by quoting the formula, and the count matches. Against the DFT's ``n^2``, at ``n = 2^20``
that is ``1.0e7`` against ``1.1e12``: a factor of **105000**.

Sizes that are not powers of two
--------------------------------
`fft` handles any length. Powers of two go through `fft_radix2`. Everything else goes through
**Bluestein's algorithm**, which rewrites the transform as a convolution using the identity
``jk = (j^2 + k^2 - (j-k)^2)/2``, pads that convolution to a power of two, and does it with three
radix-2 transforms. So the cost is ``O(n log n)`` for **every** ``n``, including primes, which
the textbook "pad to the next power of two" advice does not achieve without changing the answer.

What the transform is used for here
-----------------------------------
- `trig_least_squares` fits a truncated Fourier series to unequally spaced or noisy data, which
  is Part 5's least squares problem with a design matrix that happens to be almost orthogonal.
- `lowpass`, `highpass` and `bandpass` filter by zeroing coefficients, and `filter_report`
  measures the ringing that causes, which is the Gibbs phenomenon of lesson 58's aliasing seen
  from the other side.
- `wiener` is the filter that is optimal in the mean square sense given the signal and noise
  power spectra. `wiener_report` measures it against a brick wall filter on the same data, and
  the answer is not the simple one: with the **true** signal spectrum Wiener wins every case,
  but with the spectrum estimated from the noisy data it loses to a well chosen brick wall on a
  band limited signal and beats it by a factor of 3 on a square wave.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- helpers


def is_power_of_two(n: int) -> bool:
    m = int(n)
    return m > 0 and (m & (m - 1)) == 0


def next_power_of_two(n: int) -> int:
    """The smallest power of two at least ``n``."""
    m = int(n)
    if m < 1:
        raise ValueError(f"need a positive size, got {m}")
    return 1 << (m - 1).bit_length()


def bit_reverse_permutation(n: int) -> np.ndarray:
    """The index permutation the iterative radix-2 transform starts from.

    Splitting by parity repeatedly sorts the inputs by the **reversed** bits of their index. The
    recursive version does this implicitly by slicing; the iterative version has to do it once,
    up front, and then works in place.
    """
    m = int(n)
    if not is_power_of_two(m):
        raise ValueError(f"bit reversal needs a power of two, got {m}")
    bits = m.bit_length() - 1
    idx = np.arange(m)
    out = np.zeros(m, dtype=int)
    for b in range(bits):
        out |= ((idx >> b) & 1) << (bits - 1 - b)
    return out


# --------------------------------------------------------------------------- radix 2


def fft_radix2_recursive(x, _counter=None) -> np.ndarray:
    """Cooley-Tukey, decimation in time, written the way the derivation reads.

    Clear and wasteful: every level allocates new arrays. `fft_radix2` is the in-place version
    and is what should actually be called.
    """
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    n = v.size
    if not is_power_of_two(n):
        raise ValueError(f"radix-2 needs a power of two, got {n}")
    if n == 1:
        return v.copy()
    even = fft_radix2_recursive(v[0::2], _counter)
    odd = fft_radix2_recursive(v[1::2], _counter)
    k = np.arange(n // 2)
    twiddle = np.exp(-2j * math.pi * k / n) * odd
    if _counter is not None:
        _counter[0] += n // 2
    return np.concatenate([even + twiddle, even - twiddle])


def fft_radix2(x, inverse: bool = False, _counter=None) -> np.ndarray:
    """The iterative in-place radix-2 transform: bit reverse, then log2(n) passes of butterflies.

    This is the shape every real implementation has. The outer loop doubles the transform size,
    the middle loop walks the blocks at that size, and the inner loop runs the butterflies.

    **The copy in ``top`` is load bearing.** ``v[start:start + half]`` is a numpy *view*, so
    writing the first half of the butterfly result overwrites the values the second half still
    needs. Without the copy the transform is silently wrong by an ``O(1)`` amount at every size,
    which is exactly what a correctness test against `nalib.dft.dft` catches and what a test of
    the butterfly count alone would not.
    """
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel().copy()
    n = v.size
    if not is_power_of_two(n):
        raise ValueError(f"radix-2 needs a power of two, got {n}")
    v = v[bit_reverse_permutation(n)]
    sign = 1.0 if inverse else -1.0
    size = 2
    while size <= n:
        half = size // 2
        step = np.exp(sign * 2j * math.pi * np.arange(half) / size)
        for start in range(0, n, size):
            top = v[start:start + half].copy()
            bottom = v[start + half:start + size] * step
            v[start:start + half] = top + bottom
            v[start + half:start + size] = top - bottom
            if _counter is not None:
                _counter[0] += half
        size *= 2
    return v / n if inverse else v


# --------------------------------------------------------------------------- any length


def bluestein(x, inverse: bool = False) -> np.ndarray:
    """The chirp z-transform: a DFT of any length as a convolution of length a power of two.

    Using ``jk = (j^2 + k^2 - (j-k)^2) / 2``,

        X_k = w^{k^2/2} sum_j ( x_j w^{j^2/2} ) w^{-(j-k)^2/2}

    which is a convolution of the chirped input with a chirp kernel. Pad both to a power of two
    at least ``2n - 1``, convolve with three radix-2 transforms, and the cost is ``O(n log n)``
    for every ``n``, primes included.

    The alternative advice, "pad the signal with zeros to a power of two", computes a
    **different** transform of a longer signal, which is fine for a spectrogram and wrong if you
    wanted the length ``n`` transform.
    """
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    n = v.size
    if n < 1:
        raise ValueError("cannot transform an empty signal")
    sign = 1.0 if inverse else -1.0
    j = np.arange(n)
    chirp = np.exp(sign * 1j * math.pi * (j * j) / n)
    m = next_power_of_two(2 * n - 1)
    a = np.zeros(m, dtype=complex)
    a[:n] = v * chirp
    b = np.zeros(m, dtype=complex)
    b[:n] = np.conj(chirp)
    b[m - n + 1:] = np.conj(chirp[1:])[::-1]
    conv = fft_radix2(fft_radix2(a) * fft_radix2(b), inverse=True)
    out = chirp * conv[:n]
    return out / n if inverse else out


def fft(x) -> np.ndarray:
    """The forward transform of any length, in ``O(n log n)``.

    Dispatches to `fft_radix2` for a power of two and to `bluestein` otherwise. Checked against
    `nalib.dft.dft` at every size the tests use, which is the point of keeping the slow one.
    """
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    if v.size < 1:
        raise ValueError("cannot transform an empty signal")
    return fft_radix2(v) if is_power_of_two(v.size) else bluestein(v)


def ifft(X) -> np.ndarray:
    """The inverse transform of any length."""
    v = np.atleast_1d(np.asarray(X)).astype(complex).ravel()
    if v.size < 1:
        raise ValueError("cannot transform an empty spectrum")
    return fft_radix2(v, inverse=True) if is_power_of_two(v.size) else bluestein(v, inverse=True)


def real_fft(x) -> dict:
    """For real input, return only the ``n/2 + 1`` independent coefficients.

    The rest are the conjugates, by the Hermitian symmetry of lesson 58. Storing them is a waste
    of half the memory and computing them is a waste of half the work, which is why every
    library has a separate real transform.
    """
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n = v.size
    full = fft(v)
    return {"coefficients": full[:n // 2 + 1], "n": n,
            "stored": n // 2 + 1, "full_would_be": n,
            "saving": 1.0 - (n // 2 + 1) / n}


# --------------------------------------------------------------------------- cost


def operation_count(n: int) -> dict:
    """Count butterflies by instrumenting the transform, and compare with ``(n/2) log2 n``.

    The formula is quoted everywhere. Counting is better, because an implementation that quietly
    does more work than the formula says still passes a correctness test.
    """
    m = int(n)
    if not is_power_of_two(m):
        raise ValueError(f"the count is for radix-2, so a power of two, got {m}")
    rng = np.random.default_rng(0)
    counter = [0]
    fft_radix2(rng.standard_normal(m), _counter=counter)
    recursive = [0]
    fft_radix2_recursive(rng.standard_normal(m), _counter=recursive)
    predicted = (m // 2) * int(math.log2(m)) if m > 1 else 0
    return {"n": m, "butterflies_iterative": counter[0],
            "butterflies_recursive": recursive[0],
            "predicted": predicted,
            "matches": bool(counter[0] == predicted == recursive[0]),
            "dft_multiplications": m * m,
            "speedup": (m * m) / max(counter[0], 1)}


def cost_scaling(sizes=None) -> dict:
    """Butterfly count against ``n^2``, over a range of sizes, with the fitted exponent.

    The fit is on ``log(count)`` against ``log(n)``, and it should come out slightly above 1
    because ``n log n`` is not a power law. Reporting it as "measured 1.13, and ``n log n`` over
    this range has slope 1.13" is honest; reporting it as "measured order 1" would not be.
    """
    ns = ([2 ** k for k in range(2, 15)] if sizes is None
          else [int(v) for v in np.atleast_1d(sizes)])
    counts = np.asarray([(n // 2) * int(math.log2(n)) for n in ns], dtype=float)
    direct = np.asarray([float(n) ** 2 for n in ns])
    lg = np.log(np.asarray(ns, dtype=float))
    slope = float(np.polyfit(lg, np.log(np.maximum(counts, 1.0)), 1)[0])
    return {"sizes": np.asarray(ns), "butterflies": counts,
            "dft_multiplications": direct, "speedup": direct / np.maximum(counts, 1.0),
            "fitted_exponent": slope,
            "exponent_of_n_log_n": float(np.polyfit(
                lg, np.log(np.asarray(ns, dtype=float) * lg), 1)[0])}


# --------------------------------------------------------------------------- fitting


def trig_least_squares(t, y, n_harmonics: int, period: float = 1.0) -> dict:
    """Fit ``a_0 + sum_k [a_k cos(2 pi k t / T) + b_k sin(...)]`` by least squares.

    When ``t`` is an equally spaced full period this is the DFT, exactly, and the fit adds
    nothing. It earns its place when the samples are **unequally spaced**, or when there are more
    samples than harmonics and the data is noisy, because then there is no transform to use and
    the normal equations are the only route.

    The design matrix is nearly orthogonal, so its condition number stays near 1 while a
    polynomial design matrix of the same size is already unusable, which is Part 5's lesson
    arriving in a new setting.
    """
    x = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    v = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if x.size != v.size:
        raise ValueError(f"{x.size} times against {v.size} values")
    h = int(n_harmonics)
    if h < 0:
        raise ValueError(f"harmonics must be non-negative, got {h}")
    width = 2 * h + 1
    if x.size < width:
        raise ValueError(f"{width} unknowns need at least that many samples, got {x.size}")
    omega = 2.0 * math.pi / float(period)
    cols = [np.ones_like(x)]
    for k in range(1, h + 1):
        cols.append(np.cos(omega * k * x))
        cols.append(np.sin(omega * k * x))
    A = np.stack(cols, axis=1)
    c, *_ = np.linalg.lstsq(A, v, rcond=None)

    def evaluate(s):
        z = np.atleast_1d(np.asarray(s, dtype=float))
        out = np.full(z.shape, c[0])
        for k in range(1, h + 1):
            out = out + c[2 * k - 1] * np.cos(omega * k * z) + c[2 * k] * np.sin(omega * k * z)
        return out

    return {"coefficients": c, "harmonics": h, "period": float(period),
            "evaluate": evaluate,
            "condition": float(np.linalg.cond(A)),
            "residual": float(np.sqrt(np.mean((A @ c - v) ** 2)))}


def fit_matches_the_transform(f, n: int, n_harmonics: int) -> dict:
    """On an equally spaced full period, the least squares fit **is** the DFT.

    Worth checking, because it says the two are the same object and the fit is the general case.
    """
    m = int(n)
    grid = np.arange(m) / m
    v = np.asarray(f(grid), dtype=float)
    out = trig_least_squares(grid, v, n_harmonics)
    X = fft(v)
    h = int(n_harmonics)
    a = np.empty(h + 1)
    b = np.zeros(h + 1)
    a[0] = float(np.real(X[0])) / m
    for k in range(1, h + 1):
        a[k] = 2.0 * float(np.real(X[k])) / m
        b[k] = -2.0 * float(np.imag(X[k])) / m
    fitted = out["coefficients"]
    packed = np.empty(2 * h + 1)
    packed[0] = a[0]
    for k in range(1, h + 1):
        packed[2 * k - 1] = a[k]
        packed[2 * k] = b[k]
    scale = max(float(np.max(np.abs(packed))), 1e-300)
    return {"fit": fitted, "from_the_transform": packed,
            "relative_gap": float(np.max(np.abs(fitted - packed))) / scale,
            "agree": bool(float(np.max(np.abs(fitted - packed))) < 1e-8 * scale)}


# --------------------------------------------------------------------------- filtering


def _mask(n, keep):
    m = np.zeros(n, dtype=float)
    m[keep] = 1.0
    return m


def frequencies(n: int, rate: float = 1.0) -> np.ndarray:
    """The signed frequency of each DFT bin, in cycles per unit time."""
    m = int(n)
    k = np.arange(m)
    k = np.where(k > m // 2, k - m, k)
    return k * float(rate) / m


def lowpass(x, cutoff: float, rate: float = 1.0) -> np.ndarray:
    """Zero every coefficient above the cutoff frequency, and transform back.

    The bluntest possible filter, and it is worth doing once to see why nobody uses it: a sharp
    cut in frequency is a ``sinc`` in time, which rings. `filter_report` measures the ringing.
    """
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    freq = frequencies(v.size, rate)
    return ifft(fft(v) * _mask(v.size, np.abs(freq) <= float(cutoff)))


def highpass(x, cutoff: float, rate: float = 1.0) -> np.ndarray:
    """Zero every coefficient below the cutoff frequency."""
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    freq = frequencies(v.size, rate)
    return ifft(fft(v) * _mask(v.size, np.abs(freq) >= float(cutoff)))


def bandpass(x, low: float, high: float, rate: float = 1.0) -> np.ndarray:
    """Keep only the coefficients between two frequencies."""
    if not float(high) > float(low):
        raise ValueError(f"need low < high, got [{low}, {high}]")
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    freq = frequencies(v.size, rate)
    keep = (np.abs(freq) >= float(low)) & (np.abs(freq) <= float(high))
    return ifft(fft(v) * _mask(v.size, keep))


def filter_report(x, cutoff: float, rate: float = 1.0) -> dict:
    """What a brick wall filter does to a step, which is the Gibbs phenomenon.

    Cutting the spectrum sharply multiplies the signal by a ``sinc`` in time, whose tails decay
    only like ``1/t``. So a discontinuity picks up an overshoot that does **not** shrink as the
    cutoff rises: it stays at about 9 percent of the jump and merely gets narrower. That
    constant is Gibbs's, and it is why real filters taper.
    """
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    out = np.real(lowpass(v, cutoff, rate))
    jump = float(np.max(v) - np.min(v))
    overshoot = float(max(np.max(out) - np.max(v), np.min(v) - np.min(out)))
    return {"cutoff": float(cutoff), "filtered": out,
            "overshoot": overshoot,
            "overshoot_fraction": overshoot / max(jump, 1e-300),
            "energy_kept": float(np.sum(out ** 2) / max(np.sum(v ** 2), 1e-300))}


# --------------------------------------------------------------------------- Wiener


def wiener(noisy, signal_power=None, noise_power=None, rate: float = 1.0) -> dict:
    """The filter that minimises the mean square error, given the two power spectra.

        H_k = S_k / (S_k + N_k)

    Each coefficient is scaled by the fraction of its power that is signal. Where the signal
    dominates the coefficient passes untouched, where the noise dominates it is suppressed, and
    in between it is shrunk rather than cut. **That gradual shrinkage is the whole difference
    from a brick wall filter**, and it is why the Wiener filter does not ring.

    ``signal_power`` and ``noise_power`` are per bin. When ``signal_power`` is omitted it is
    estimated as ``max(|X_k|^2 - N_k, 0)``, which is what a practical implementation does and
    is the reason the filter needs a noise estimate but not a signal one.
    """
    v = np.atleast_1d(np.asarray(noisy, dtype=float)).ravel()
    n = v.size
    X = fft(v)
    if noise_power is None:
        raise ValueError("the noise power spectrum is required")
    N = np.broadcast_to(np.asarray(noise_power, dtype=float), (n,)).astype(float)
    if np.any(N < 0.0):
        raise ValueError("the noise power must be non-negative")
    if signal_power is None:
        S = np.maximum(np.abs(X) ** 2 - N, 0.0)
    else:
        S = np.broadcast_to(np.asarray(signal_power, dtype=float), (n,)).astype(float)
        if np.any(S < 0.0):
            raise ValueError("the signal power must be non-negative")
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where(S + N > 0.0, S / (S + N), 0.0)
    return {"gain": H, "filtered": np.real(ifft(X * H)),
            "spectrum": X, "estimated_signal_power": S, "noise_power": N}


def wiener_report(clean, noise_level: float, cutoff: float = None, rate: float = 1.0,
                  rng=None) -> dict:
    """Wiener against a brick wall low pass on the same noisy signal, four ways.

    The comparison is against the **clean** signal, not against the noisy one, because fitting
    the noise is exactly the failure being avoided. That is the same distinction lesson 51's
    smoothing spline needed.

    Three filtered results are reported, because the obvious two-way comparison gives a
    misleading answer.

    - ``wiener_true`` uses the exact signal power spectrum. This is the optimal filter and it
      wins every case measured: on two pure tones at noise 0.1 it reaches ``0.0085`` against the
      brick wall's ``0.0237``, and on a square wave ``0.056`` against ``0.202``.
    - ``wiener_estimated`` uses ``max(|X|^2 - N, 0)``, which is what you can actually compute.
      It costs a factor of 2 to 5 against the true version, and on a band limited signal that is
      enough to **lose** to a brick wall filter given the right cutoff: ``0.042`` against
      ``0.024``.
    - ``brickwall`` is given a cutoff, and in the tests that cutoff is chosen to contain the
      signal exactly, which makes it an oracle. Give it the wrong cutoff, or a signal that is
      not band limited, and it rings: on a square wave it is ``0.202`` against the noisy signal's
      own ``0.095``, so it is worse than doing nothing.

    The honest summary is that the Wiener filter is optimal **given the spectra**, the practical
    version pays for estimating one of them, and a brick wall filter is only competitive when
    you already know the answer.
    """
    gen = np.random.default_rng() if rng is None else rng
    c = np.atleast_1d(np.asarray(clean, dtype=float)).ravel()
    n = c.size
    noisy = c + float(noise_level) * gen.standard_normal(n)
    power = float(noise_level) ** 2 * n
    estimated = wiener(noisy, noise_power=power)["filtered"]
    exact = wiener(noisy, signal_power=np.abs(fft(c)) ** 2, noise_power=power)["filtered"]
    cut = float(cutoff) if cutoff is not None else 0.1 * float(rate)
    brick = np.real(lowpass(noisy, cut, rate))
    err = lambda a: float(np.sqrt(np.mean((a - c) ** 2)))
    return {"noise_level": float(noise_level), "cutoff": cut,
            "noisy_error": err(noisy),
            "wiener_true_error": err(exact),
            "wiener_estimated_error": err(estimated),
            "brickwall_error": err(brick),
            "true_wiener_is_best": bool(err(exact) <= min(err(estimated), err(brick),
                                                          err(noisy))),
            "estimated_beats_noisy": bool(err(estimated) < err(noisy)),
            "estimated_beats_brickwall": bool(err(estimated) < err(brick)),
            "improvement": err(noisy) / max(err(estimated), 1e-300)}
