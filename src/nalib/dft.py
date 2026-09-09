"""The discrete Fourier transform: interpolation by a basis that is orthogonal for free.

Where this comes from
---------------------
Part 7 interpolated with polynomials and had to fight for conditioning. Lesson 55 fixed that by
choosing an orthogonal basis, at the cost of computing inner products. **On equally spaced points
the trigonometric basis is orthogonal exactly, with no integral to compute**, and the inner
products are finite sums the grid hands you. That is the whole reason the DFT is everywhere.

Roots of unity
--------------
Let ``w = exp(-2 pi i / n)``. The ``n`` powers ``w^0 ... w^{n-1}`` are the ``n``-th roots of
unity, and three facts about them carry everything:

- **They sum to zero** for ``n > 1``, because they are the roots of ``z^n - 1`` and that
  polynomial has no ``z^{n-1}`` term.
- **``w^j`` and ``w^{j+n}`` are the same point.** Exponents live mod ``n``, which is aliasing.
- **``w^{n/2} = -1`` for even ``n``**, which is the fact the FFT of lesson 59 is built on.

`roots_of_unity` returns them and `root_properties` checks all three rather than asserting them.

The transform
-------------
    X_k = sum_{j=0}^{n-1} x_j w^{jk},        x_j = (1/n) sum_{k=0}^{n-1} X_k w^{-jk}

`dft` and `inverse_dft` are the direct ``O(n^2)`` sums, written to be read. Lesson 59 makes them
``O(n log n)`` without changing what they compute, and `nalib.fft` is checked against these.

The interpolation theorem
-------------------------
This is the result that makes the DFT an interpolation method rather than a change of basis. For
``n`` equally spaced points ``t_j = j/n`` on ``[0, 1)``, the function

    P(t) = (1/n) sum_k X_k exp(2 pi i k t)

satisfies ``P(t_j) = x_j`` **exactly**, for every ``j``, for any data. So the DFT coefficients are
the coefficients of the interpolating trigonometric polynomial, and no linear system was solved.
`interpolation_theorem` verifies it to ``1e-15`` at every size tried.

What orthogonality buys
-----------------------
``F^H F = n I`` exactly, so the inverse is the conjugate transpose over ``n``: the condition
number of the transform is **1**, at every size. `conditioning` measures it against the
Vandermonde matrix of lesson 44, which reaches ``1e16`` by 20 nodes. Nothing else in this course
has a condition number of exactly 1.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- roots of unity


def roots_of_unity(n: int, sign: int = -1) -> np.ndarray:
    """The ``n`` powers of ``exp(sign * 2 pi i / n)``, in order.

    ``sign = -1`` is the forward transform's convention, which is what numpy, MATLAB and every
    signal processing text use. ``sign = +1`` is the inverse.
    """
    m = int(n)
    if m < 1:
        raise ValueError(f"need at least one root, got {m}")
    s = int(sign)
    if s not in (-1, 1):
        raise ValueError(f"sign must be -1 or +1, got {sign}")
    return np.exp(s * 2j * math.pi * np.arange(m) / m)


def root_properties(n: int) -> dict:
    """The three facts the whole subject rests on, checked rather than stated."""
    m = int(n)
    w = roots_of_unity(m)
    powers = w ** m
    return {"n": m,
            "sum": complex(np.sum(w)),
            "sum_is_zero": bool(m == 1 or abs(np.sum(w)) < 1e-12 * m),
            "product": complex(np.prod(w)),
            "each_is_an_nth_root": float(np.max(np.abs(powers - 1.0))),
            "on_the_unit_circle": float(np.max(np.abs(np.abs(w) - 1.0))),
            "half_turn_is_minus_one": (complex(w[m // 2]) if m % 2 == 0 else None),
            "wraps_at_n": float(np.max(np.abs(np.exp(-2j * math.pi * np.arange(m, 2 * m) / m)
                                              - w)))}


# --------------------------------------------------------------------------- the transform


def dft_matrix(n: int, sign: int = -1) -> np.ndarray:
    """The ``n`` by ``n`` matrix with entries ``w^{jk}``, built explicitly.

    Never used for real work, because it costs ``O(n^2)`` storage and ``O(n^2)`` per transform
    against the FFT's ``O(n log n)``. It is here because every structural statement about the
    DFT is a statement about this matrix, and those are easier to check on the matrix than on
    the algorithm.
    """
    m = int(n)
    if m < 1:
        raise ValueError(f"need a positive size, got {m}")
    j = np.arange(m)
    return np.exp(int(sign) * 2j * math.pi * np.outer(j, j) / m)


def dft(x) -> np.ndarray:
    """The forward transform, as the defining sum.

    ``O(n^2)`` on purpose. `nalib.fft.fft` computes the same thing in ``O(n log n)`` and is
    tested against this one, which is the point of keeping a slow reference.
    """
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    n = v.size
    if n < 1:
        raise ValueError("cannot transform an empty signal")
    k = np.arange(n)
    return np.exp(-2j * math.pi * np.outer(k, k) / n) @ v


def inverse_dft(X) -> np.ndarray:
    """The inverse transform. The only differences are the sign and the ``1/n``."""
    v = np.atleast_1d(np.asarray(X)).astype(complex).ravel()
    n = v.size
    if n < 1:
        raise ValueError("cannot transform an empty spectrum")
    k = np.arange(n)
    return (np.exp(2j * math.pi * np.outer(k, k) / n) @ v) / n


def round_trip(x) -> float:
    """``|idft(dft(x)) - x|``, relative. The identity every implementation must satisfy first."""
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    back = inverse_dft(dft(v))
    return float(np.max(np.abs(back - v)) / max(float(np.max(np.abs(v))), 1e-300))


# --------------------------------------------------------------------------- orthogonality


def orthogonality(n: int) -> dict:
    """``F^H F = n I``, which is the statement that the basis is orthogonal.

    The proof is one line: the ``(j, l)`` entry of ``F^H F`` is
    ``sum_k w^{-jk} w^{lk} = sum_k w^{(l-j)k}``, a geometric series with ratio ``w^{l-j}``. That
    ratio is 1 exactly when ``l = j``, giving ``n``, and otherwise the series sums to
    ``(w^{(l-j)n} - 1)/(w^{l-j} - 1) = 0`` because ``w^n = 1``.

    So the columns are orthogonal with squared norm ``n``, the matrix ``F/sqrt(n)`` is unitary,
    and the condition number is exactly 1.
    """
    m = int(n)
    F = dft_matrix(m)
    G = F.conj().T @ F
    off = G - np.diag(np.diag(G))
    return {"n": m, "gram": G,
            "diagonal": np.real(np.diag(G)),
            "diagonal_is_n": float(np.max(np.abs(np.diag(G) - m))),
            "worst_off_diagonal": float(np.max(np.abs(off))),
            "relative_off_diagonal": float(np.max(np.abs(off)) / m),
            "is_orthogonal": bool(float(np.max(np.abs(off))) < 1e-10 * m)}


def conditioning(sizes=None) -> dict:
    """The DFT matrix against the polynomial Vandermonde matrix of lesson 44.

    Both are interpolation matrices on ``n`` nodes. One has condition number 1 at every size and
    the other reaches ``1e16`` by 20 nodes. That is the whole argument for working in this basis
    whenever the nodes are equally spaced and the function is periodic.
    """
    ns = ([4, 8, 16, 32, 64] if sizes is None else [int(v) for v in np.atleast_1d(sizes)])
    rows = []
    for n in ns:
        F = dft_matrix(n) / math.sqrt(n)
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        V = np.vander(t, n, increasing=True)
        rows.append((n, float(np.linalg.cond(F)), float(np.linalg.cond(V))))
    return {"sizes": np.asarray([r[0] for r in rows]),
            "dft_condition": np.asarray([r[1] for r in rows]),
            "vandermonde_condition": np.asarray([r[2] for r in rows])}


def parseval(x) -> dict:
    """``sum |x_j|^2 = (1/n) sum |X_k|^2``: the transform preserves energy.

    A direct consequence of ``F/sqrt(n)`` being unitary, and the identity every signal processing
    argument about noise power is really using.
    """
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    X = dft(v)
    lhs = float(np.sum(np.abs(v) ** 2))
    rhs = float(np.sum(np.abs(X) ** 2) / v.size)
    return {"time_energy": lhs, "frequency_energy": rhs,
            "relative_gap": abs(lhs - rhs) / max(lhs, 1e-300)}


# --------------------------------------------------------------------------- interpolation


def trig_interpolant(X, t) -> np.ndarray:
    """Evaluate ``P(t) = (1/n) sum_k X_k exp(2 pi i k t)`` at arbitrary ``t``.

    Note the frequencies are taken as ``0 ... n-1``, not centred. For evaluating **at the grid
    points** that makes no difference, because ``exp(2 pi i k j / n)`` is unchanged by shifting
    ``k`` by ``n``. **Between** the grid points it makes a large difference, and for real data
    the centred convention is the one that gives a real, smooth, sensible interpolant.
    `centred_interpolant` does that, and `naive_versus_centred` measures the gap.
    """
    c = np.atleast_1d(np.asarray(X)).astype(complex).ravel()
    n = c.size
    z = np.atleast_1d(np.asarray(t, dtype=float))
    k = np.arange(n)
    return (np.exp(2j * math.pi * np.outer(z, k)) @ c) / n


def centred_interpolant(X, t) -> np.ndarray:
    """The same sum with frequencies shifted to ``-n/2 ... n/2 - 1``, which is what you want.

    For a real signal of even length the Nyquist coefficient ``X_{n/2}`` is shared between
    ``+n/2`` and ``-n/2``, and splitting it in half is what makes the interpolant real. Without
    that split the imaginary part reaches ``0.5 |X_{n/2}|``, which is not a rounding artefact.
    """
    c = np.atleast_1d(np.asarray(X)).astype(complex).ravel()
    n = c.size
    z = np.atleast_1d(np.asarray(t, dtype=float))
    freq = np.arange(n)
    freq = np.where(freq > n // 2, freq - n, freq)
    weights = np.ones(n, dtype=complex)
    if n % 2 == 0:
        half = n // 2
        weights[half] = 0.5
        c = np.concatenate([c, [c[half]]])
        freq = np.concatenate([freq, [-half]])
        weights = np.concatenate([weights, [0.5]])
    return (np.exp(2j * math.pi * np.outer(z, freq)) @ (c * weights)) / n


def interpolation_theorem(x) -> dict:
    """The theorem: the DFT coefficients interpolate the data at the grid, exactly.

    Both conventions are checked, because both must reproduce the data and only one gives a
    sensible curve between the points.
    """
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    n = v.size
    X = dft(v)
    grid = np.arange(n) / n
    naive = trig_interpolant(X, grid)
    centred = centred_interpolant(X, grid)
    scale = max(float(np.max(np.abs(v))), 1e-300)
    return {"n": n,
            "naive_gap": float(np.max(np.abs(naive - v))) / scale,
            "centred_gap": float(np.max(np.abs(centred - v))) / scale,
            "both_interpolate": bool(float(np.max(np.abs(naive - v))) < 1e-10 * scale
                                     and float(np.max(np.abs(centred - v))) < 1e-10 * scale)}


def naive_versus_centred(f, n: int, n_probe: int = 2001) -> dict:
    """How different the two conventions are between the grid points, on a real signal.

    They agree at every grid point by construction. Away from the grid the uncentred version
    treats a frequency of ``n-1`` as a very fast oscillation instead of a slow one going the
    other way, so it wiggles violently and is complex. The centred one is the real trigonometric
    interpolant.
    """
    m = int(n)
    grid = np.arange(m) / m
    v = np.asarray(f(grid), dtype=float)
    X = dft(v)
    t = np.linspace(0.0, 1.0, int(n_probe), endpoint=False)
    a = trig_interpolant(X, t)
    b = centred_interpolant(X, t)
    truth = np.asarray(f(t), dtype=float)
    return {"n": m,
            "naive_max_imaginary": float(np.max(np.abs(np.imag(a)))),
            "centred_max_imaginary": float(np.max(np.abs(np.imag(b)))),
            "naive_error": float(np.max(np.abs(np.real(a) - truth))),
            "centred_error": float(np.max(np.abs(np.real(b) - truth))),
            "disagreement": float(np.max(np.abs(a - b)))}


def real_form(X) -> dict:
    """Rewrite a real signal's spectrum as cosine and sine amplitudes.

        x_j = a_0/2 + sum_k [ a_k cos(2 pi k t_j) + b_k sin(2 pi k t_j) ] + ...

    with ``a_k = 2 Re(X_k)/n`` and ``b_k = -2 Im(X_k)/n``. This is the form a textbook writes
    and the form a plot of "the frequency content" is really showing.
    """
    c = np.atleast_1d(np.asarray(X)).astype(complex).ravel()
    n = c.size
    half = n // 2
    keep = np.arange(half + 1)
    a = 2.0 * np.real(c[keep]) / n
    b = -2.0 * np.imag(c[keep]) / n
    a[0] = np.real(c[0]) / n * 2.0
    if n % 2 == 0:
        a[half] = np.real(c[half]) / n
        b[half] = 0.0
    return {"cosine": a, "sine": b, "frequencies": keep,
            "amplitude": np.sqrt(a ** 2 + b ** 2),
            "phase": np.arctan2(b, a)}


def hermitian_symmetry(x) -> dict:
    """For real input, ``X_{n-k} = conj(X_k)``, so half the spectrum is redundant.

    This is why a real FFT of length ``n`` returns ``n/2 + 1`` numbers and why storing the whole
    spectrum for real data wastes half the memory.
    """
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n = v.size
    X = dft(v)
    k = np.arange(1, (n + 1) // 2)
    gap = np.abs(X[n - k] - np.conj(X[k])) if k.size else np.zeros(0)
    scale = max(float(np.max(np.abs(X))), 1e-300)
    return {"worst_gap": float(np.max(gap)) if gap.size else 0.0,
            "relative": (float(np.max(gap)) / scale) if gap.size else 0.0,
            "independent_values": n // 2 + 1,
            "holds": bool(gap.size == 0 or float(np.max(gap)) < 1e-10 * scale)}


# --------------------------------------------------------------------------- aliasing


def aliasing(n: int, frequencies=None, n_probe: int = 2001) -> dict:
    """Which continuous frequencies a grid of ``n`` points cannot tell apart.

    ``exp(2 pi i k t)`` and ``exp(2 pi i (k + n) t)`` agree at every ``t_j = j/n``, because they
    differ by ``exp(2 pi i j)``, which is 1. So the grid sees frequency ``k`` and frequency
    ``k + n`` as the same signal, and there is no way to recover which was sampled.

    That is the DFT's version of lesson 46's interpolation error: information between the nodes
    is not merely approximated, it is absent.
    """
    m = int(n)
    ks = ([1, 1 + m, 1 + 2 * m, m - 1] if frequencies is None
          else [int(v) for v in np.atleast_1d(frequencies)])
    grid = np.arange(m) / m
    fine = np.linspace(0.0, 1.0, int(n_probe), endpoint=False)
    rows = []
    for k in ks:
        on_grid = np.exp(2j * math.pi * k * grid)
        base = np.exp(2j * math.pi * (k % m) * grid)
        off_grid = np.exp(2j * math.pi * k * fine)
        base_fine = np.exp(2j * math.pi * (k % m) * fine)
        rows.append((k, k % m,
                     float(np.max(np.abs(on_grid - base))),
                     float(np.max(np.abs(off_grid - base_fine)))))
    return {"n": m,
            "frequencies": np.asarray([r[0] for r in rows]),
            "folded_to": np.asarray([r[1] for r in rows]),
            "gap_on_the_grid": np.asarray([r[2] for r in rows]),
            "gap_off_the_grid": np.asarray([r[3] for r in rows])}


def nyquist(f, frequency: float, sizes=None, n_probe: int = 4001) -> dict:
    """Sample a pure tone at several rates and see where the reconstruction stops working.

    The sampling theorem says ``n`` samples per unit time resolve frequencies strictly below
    ``n/2``. At exactly ``n/2`` the phase is lost, and above it the tone is reported as a lower
    frequency. `folded_frequency` is what the grid thinks it saw.
    """
    ns = ([4, 8, 16, 32, 64] if sizes is None else [int(v) for v in np.atleast_1d(sizes)])
    freq = float(frequency)
    t = np.linspace(0.0, 1.0, int(n_probe), endpoint=False)
    truth = np.asarray(f(t, freq), dtype=float)
    rows = []
    for n in ns:
        grid = np.arange(n) / n
        v = np.asarray(f(grid, freq), dtype=float)
        X = dft(v)
        rec = np.real(centred_interpolant(X, t))
        k = int(np.argmax(np.abs(X[:n // 2 + 1])))
        rows.append((n, n / 2.0, float(np.max(np.abs(rec - truth))), k, bool(freq < n / 2.0)))
    return {"frequency": freq,
            "sizes": np.asarray([r[0] for r in rows]),
            "nyquist_limit": np.asarray([r[1] for r in rows]),
            "reconstruction_error": np.asarray([r[2] for r in rows]),
            "folded_frequency": np.asarray([r[3] for r in rows]),
            "above_nyquist": np.asarray([not r[4] for r in rows])}


# --------------------------------------------------------------------------- theorems


def shift_theorem(x, shift: int) -> dict:
    """Shifting the data multiplies the spectrum by a phase, and changes nothing else.

        DFT(x shifted by s)_k  =  w^{sk} X_k

    So the magnitude spectrum is unchanged by a shift, which is why it is used for matching and
    why phase carries the position information.
    """
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    n = v.size
    s = int(shift)
    X = dft(v)
    Y = dft(np.roll(v, s))
    k = np.arange(n)
    predicted = np.exp(-2j * math.pi * s * k / n) * X
    scale = max(float(np.max(np.abs(X))), 1e-300)
    return {"shift": s,
            "relative_gap": float(np.max(np.abs(Y - predicted))) / scale,
            "magnitudes_unchanged": float(np.max(np.abs(np.abs(Y) - np.abs(X)))) / scale}


def convolution_theorem(x, y) -> dict:
    """Circular convolution becomes multiplication, which is the reason the FFT matters.

        DFT(x * y)_k = X_k Y_k

    Direct convolution costs ``O(n^2)``. Through the transform it costs three transforms and
    ``n`` multiplications, so ``O(n log n)`` once lesson 59 supplies the fast transform. That
    single fact is behind fast polynomial multiplication, fast big-integer arithmetic, and every
    convolutional filter that is not tiny.
    """
    a = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    b = np.atleast_1d(np.asarray(y)).astype(complex).ravel()
    if a.size != b.size:
        raise ValueError(f"circular convolution needs equal lengths, got {a.size} and {b.size}")
    n = a.size
    direct = np.asarray([sum(a[j] * b[(i - j) % n] for j in range(n)) for i in range(n)])
    through = inverse_dft(dft(a) * dft(b))
    scale = max(float(np.max(np.abs(direct))), 1e-300)
    return {"direct": direct, "through_the_transform": through,
            "relative_gap": float(np.max(np.abs(direct - through))) / scale,
            "direct_multiplications": n * n,
            "transform_multiplications": 3 * n * n + n}


# --------------------------------------------------------------------------- evaluation


def twiddle_by_recurrence(n: int, method: str = "stable") -> dict:
    """Generate ``cos(2 pi k / n)`` and ``sin(2 pi k / n)`` for ``k = 0 ... n-1`` by recurrence.

    Calling the library ``cos`` and ``sin`` ``n`` times is accurate and slow, and an FFT needs
    these values constantly, so implementations generate them by recurrence. There are two
    recurrences and one of them is wrong.

    ``method="naive"`` uses ``c_{k+1} = c c_k - s s_k`` directly. The error grows like ``k``
    times the rounding of the first step, because nothing damps it.

    ``method="stable"`` uses the increment form of Singleton and Oliver,

        alpha = 2 sin^2(h/2),   beta = sin(h)
        c_{k+1} = c_k - (alpha c_k + beta s_k)
        s_{k+1} = s_k - (alpha s_k - beta c_k)

    which stores the **difference** rather than the value, so the leading term of the rounding
    cancels instead of accumulating.

    Returns both the values and the error against direct evaluation, so the difference is a
    measurement rather than a claim.
    """
    m = int(n)
    if m < 1:
        raise ValueError(f"need a positive size, got {m}")
    key = str(method).lower()
    if key not in ("naive", "stable", "direct"):
        raise ValueError(f"method must be 'naive', 'stable' or 'direct', got {method!r}")
    h = 2.0 * math.pi / m
    k = np.arange(m)
    exact_c = np.cos(h * k)
    exact_s = np.sin(h * k)
    if key == "direct":
        c, s = exact_c.copy(), exact_s.copy()
    else:
        c = np.empty(m)
        s = np.empty(m)
        c[0], s[0] = 1.0, 0.0
        if key == "naive":
            ch, sh = math.cos(h), math.sin(h)
            for i in range(1, m):
                c[i] = ch * c[i - 1] - sh * s[i - 1]
                s[i] = sh * c[i - 1] + ch * s[i - 1]
        else:
            alpha = 2.0 * math.sin(0.5 * h) ** 2
            beta = math.sin(h)
            for i in range(1, m):
                c[i] = c[i - 1] - (alpha * c[i - 1] + beta * s[i - 1])
                s[i] = s[i - 1] - (alpha * s[i - 1] - beta * c[i - 1])
    return {"n": m, "method": key, "cos": c, "sin": s,
            "max_error": float(np.max(np.hypot(c - exact_c, s - exact_s))),
            "worst_modulus_drift": float(np.max(np.abs(np.hypot(c, s) - 1.0)))}


def recurrence_comparison(sizes=None) -> dict:
    """The two recurrences and direct evaluation, side by side, as the length grows."""
    ns = ([64, 256, 1024, 4096, 16384] if sizes is None
          else [int(v) for v in np.atleast_1d(sizes)])
    rows = []
    for n in ns:
        naive = twiddle_by_recurrence(n, "naive")
        stable = twiddle_by_recurrence(n, "stable")
        rows.append((n, naive["max_error"], stable["max_error"],
                     naive["worst_modulus_drift"], stable["worst_modulus_drift"]))
    return {"sizes": np.asarray([r[0] for r in rows]),
            "naive_error": np.asarray([r[1] for r in rows]),
            "stable_error": np.asarray([r[2] for r in rows]),
            "naive_drift": np.asarray([r[3] for r in rows]),
            "stable_drift": np.asarray([r[4] for r in rows])}
