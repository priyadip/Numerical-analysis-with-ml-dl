"""Low rank approximation: Eckart-Young, truncation, randomised methods, and what they are for.

What this module is for
-----------------------
Lesson 33 used the Eckart-Young theorem on credit to derive total least squares. Lesson 32 used
truncation to regularize. This module states and verifies the theorem, and then does what it is
actually for.

**The theorem is one sentence.** Of all matrices of rank ``k``, the one closest to ``A`` is the
truncated SVD, and the distance is exactly ``sigma_{k+1}`` in the 2-norm. So the singular values
are not merely a description of ``A``: they are a **schedule of what it costs to simplify it**,
readable before any approximation is made.

**And that is what makes the SVD the most used decomposition there is.** Compression, noise
removal, latent factors, model reduction and rank determination are all the same operation,
differing only in what the discarded part is called.

`randomised_svd` is the reason it scales. Sampling the range with a random matrix finds the
dominant subspace in a handful of matrix products, and Halko, Martinsson and Tropp's analysis
says the error is within a modest factor of optimal with probability overwhelmingly close to 1.
`oversampling_matters` and `power_iteration_helps` measure both knobs.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Approximation:
    """A rank ``k`` approximation and the two error norms that matter."""

    matrix: np.ndarray
    rank: int
    error_2: float
    error_frobenius: float
    optimal_2: float
    optimal_frobenius: float

    @property
    def is_optimal(self) -> bool:
        return (self.error_2 <= self.optimal_2 * (1.0 + 1e-9) + 1e-12
                and self.error_frobenius <= self.optimal_frobenius * (1.0 + 1e-9) + 1e-12)


def truncate(A, k: int) -> Approximation:
    """The best rank ``k`` approximation, with its errors and the optimal ones alongside.

    **Eckart-Young-Mirsky.** Over all matrices ``B`` of rank at most ``k``,

        min ||A - B||_2 = sigma_{k+1},
        min ||A - B||_F = sqrt(sigma_{k+1}^2 + ... + sigma_r^2),

    and the truncated SVD attains both **at the same time**, which is the surprising part: one
    matrix is simultaneously optimal in two different norms. Mirsky extended it to every
    unitarily invariant norm.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    k = int(k)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if not 0 <= k <= s.size:
        raise ValueError(f"k must be between 0 and {s.size}, got {k}")
    Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]
    tail = s[k:]
    return Approximation(matrix=Ak, rank=k,
                         error_2=float(np.linalg.norm(A - Ak, 2)),
                         error_frobenius=float(np.linalg.norm(A - Ak)),
                         optimal_2=float(tail[0]) if tail.size else 0.0,
                         optimal_frobenius=float(np.sqrt(np.sum(tail ** 2))))


def beat_the_truncation(A, k: int, n_trials: int = 4000, rng=None) -> dict:
    """Try to find a rank ``k`` matrix closer to ``A`` than the truncated SVD. None exists.

    The theorem is an optimality claim, and an optimality claim is worth attacking rather than
    quoting. This searches over random rank ``k`` matrices fitted to ``A`` in the least squares
    sense, which is the best a random subspace can do, and reports the closest it found.

    **Nothing beats the truncation**, and the margin by which the best random attempt falls short
    is the honest measure of how special the singular subspace is.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    k = int(k)
    if not 1 <= k <= min(m, n):
        raise ValueError(f"k must be between 1 and {min(m, n)}, got {k}")
    gen = np.random.default_rng() if rng is None else rng
    best = np.inf
    for _ in range(int(n_trials)):
        Q, _ = np.linalg.qr(gen.standard_normal((m, k)))
        # The best rank k matrix with this column space is the projection of A onto it.
        best = min(best, float(np.linalg.norm(A - Q @ (Q.T @ A), 2)))
    optimal = truncate(A, k)
    return {"best_random": best, "truncated": optimal.error_2,
            "shortfall": best / optimal.error_2 if optimal.error_2 > 0 else float("inf"),
            "trials": int(n_trials)}


def energy_captured(A, k: int) -> float:
    """The fraction of ``||A||_F^2`` retained by a rank ``k`` truncation.

    ``sum_{i<=k} sigma_i^2 / sum_i sigma_i^2``. This is the quantity people actually choose ``k``
    by, and it is worth naming because it is a **squared** fraction: keeping 90 percent of the
    energy is not keeping 90 percent of the matrix.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    k = int(k)
    s = np.linalg.svd(A, compute_uv=False)
    if not 0 <= k <= s.size:
        raise ValueError(f"k must be between 0 and {s.size}, got {k}")
    total = float(np.sum(s ** 2))
    if total == 0.0:
        return 1.0
    return float(np.sum(s[:k] ** 2) / total)


def rank_for_energy(A, fraction: float) -> int:
    """The smallest ``k`` capturing at least ``fraction`` of the energy."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    s = np.linalg.svd(A, compute_uv=False)
    total = float(np.sum(s ** 2))
    if total == 0.0:
        return 0
    running = np.cumsum(s ** 2) / total
    return int(np.searchsorted(running, fraction - 1e-15) + 1)


def compression_ratio(m: int, n: int, k: int) -> float:
    """How much a rank ``k`` factorization saves: ``mn`` numbers against ``k(m + n + 1)``.

    **It only saves anything when ``k`` is small relative to ``mn/(m+n)``**, and for a square
    matrix that means ``k < n/2``. A rank ``n-1`` approximation of an ``n x n`` matrix costs
    nearly twice as much to store as the matrix, which is worth knowing before compressing
    anything.
    """
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1 or k < 0:
        raise ValueError(f"bad shape or rank: ({m}, {n}), k = {k}")
    stored = k * (m + n + 1)
    return float(m * n) / stored if stored else float("inf")


# ----------------------------------------------------------------------------- randomised


def randomised_svd(A, k: int, oversample: int = 10, n_power: int = 0,
                   rng=None) -> dict:
    """Halko, Martinsson and Tropp: find the dominant subspace by sampling the range.

    Multiply ``A`` by a random ``n x (k + p)`` matrix, orthonormalise the result, and the columns
    span most of the dominant left singular subspace. Project ``A`` onto it and take the SVD of
    the small projected matrix.

    **Cost is ``k + p`` matrix products instead of a full decomposition.** For a matrix that is
    sparse, structured, or only available as an operator, that is the difference between possible
    and not.

    ``oversample`` is the ``p``, and it is not optional: with ``p = 0`` the sampled subspace
    misses part of the target and the error is noticeably above optimal. ``n_power`` applies
    ``(A A^T)^q`` to sharpen the sampling, which helps when the singular values decay slowly and
    costs two more products per step.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    k = int(k)
    p = int(oversample)
    if not 1 <= k <= min(m, n):
        raise ValueError(f"k must be between 1 and {min(m, n)}, got {k}")
    if p < 0:
        raise ValueError(f"oversample must be non-negative, got {p}")
    ell = min(k + p, n)
    gen = np.random.default_rng() if rng is None else rng

    Omega = gen.standard_normal((n, ell))
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    for _ in range(int(n_power)):
        # Re-orthonormalise between products, or the extra powers destroy what they sharpen.
        Q, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Q)

    B = Q.T @ A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]
    products = 1 + 2 * int(n_power)
    return {"matrix": Ak, "U": U[:, :k], "s": s[:k], "Vt": Vt[:k, :],
            "error_2": float(np.linalg.norm(A - Ak, 2)),
            "error_frobenius": float(np.linalg.norm(A - Ak)),
            "matrix_products": products, "sketch_size": ell}


def oversampling_matters(A, k: int, oversamples=(0, 2, 5, 10, 20), n_trials: int = 20,
                         rng=None) -> dict:
    """Measure the randomised error against the oversampling, as a ratio to optimal."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    optimal = truncate(A, int(k)).error_2
    gen = np.random.default_rng() if rng is None else rng
    rows = []
    for p in oversamples:
        errs = [randomised_svd(A, k, oversample=int(p), rng=gen)["error_2"]
                for _ in range(int(n_trials))]
        rows.append({"oversample": int(p),
                     "median_ratio": float(np.median(errs) / optimal) if optimal > 0 else 1.0,
                     "worst_ratio": float(np.max(errs) / optimal) if optimal > 0 else 1.0})
    return {"optimal": optimal, "rows": rows}


def power_iteration_helps(A, k: int, powers=(0, 1, 2, 4), oversample: int = 5,
                          n_trials: int = 20, rng=None) -> dict:
    """Measure the randomised error against the number of power steps."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    optimal = truncate(A, int(k)).error_2
    gen = np.random.default_rng() if rng is None else rng
    rows = []
    for q in powers:
        errs = [randomised_svd(A, k, oversample=oversample, n_power=int(q),
                               rng=gen)["error_2"] for _ in range(int(n_trials))]
        rows.append({"power": int(q), "products": 1 + 2 * int(q),
                     "median_ratio": float(np.median(errs) / optimal) if optimal > 0 else 1.0})
    return {"optimal": optimal, "rows": rows}


# ----------------------------------------------------------------------------- applications


def synthetic_image(height: int, width: int, n_stripes: int = 6, noise: float = 0.0,
                    rng=None) -> np.ndarray:
    """A test image built from known ingredients, so its rank structure is understood.

    Smooth gradients are low rank; sharp edges are not. This mixes both, so the singular value
    plot has the shape a photograph's does: a few large values carrying the structure and a long
    tail carrying the detail. Built rather than loaded so the lesson needs no data file.
    """
    height, width = int(height), int(width)
    if height < 2 or width < 2:
        raise ValueError(f"the image must be at least 2 by 2, got ({height}, {width})")
    y = np.linspace(0.0, 1.0, height)[:, None]
    x = np.linspace(0.0, 1.0, width)[None, :]
    smooth = np.exp(-((x - 0.4) ** 2 + (y - 0.6) ** 2) / 0.15)
    ramp = 0.4 * (x + y)
    stripes = 0.3 * (np.floor(x * int(n_stripes)) % 2)
    disc = 0.5 * (((x - 0.7) ** 2 + (y - 0.3) ** 2) < 0.02)
    img = smooth + ramp + stripes + disc
    if noise:
        gen = np.random.default_rng() if rng is None else rng
        img = img + noise * gen.standard_normal((height, width))
    return img


def denoise(clean, noise_level: float, rank: int, rng=None) -> dict:
    """Add noise, truncate, and see whether the truncation removed more noise than signal.

    **Truncation is a filter, not a compression, when used this way.** The signal lives in a few
    singular directions and the noise is spread over all of them, so discarding the tail removes
    most of the noise and little of the signal. Whether it wins depends on the gap, and this
    reports the answer rather than assuming it.
    """
    clean = np.atleast_2d(np.asarray(clean, dtype=float))
    gen = np.random.default_rng() if rng is None else rng
    noisy = clean + float(noise_level) * gen.standard_normal(clean.shape)
    recovered = truncate(noisy, int(rank)).matrix
    scale = max(float(np.linalg.norm(clean)), 1e-300)
    return {"noisy": noisy, "recovered": recovered,
            "error_before": float(np.linalg.norm(noisy - clean) / scale),
            "error_after": float(np.linalg.norm(recovered - clean) / scale),
            "improvement": float(np.linalg.norm(noisy - clean)
                                 / max(np.linalg.norm(recovered - clean), 1e-300))}


def best_denoising_rank(clean, noise_level: float, rng=None) -> dict:
    """Sweep the rank and find where denoising stops helping.

    There is a minimum: too small a rank throws away signal, too large keeps noise. Its location
    is a property of the gap between the signal's singular values and the noise floor, which is
    lesson 32's discrete Picard condition in another costume.
    """
    clean = np.atleast_2d(np.asarray(clean, dtype=float))
    gen = np.random.default_rng() if rng is None else rng
    noisy = clean + float(noise_level) * gen.standard_normal(clean.shape)
    limit = min(clean.shape)
    scale = max(float(np.linalg.norm(clean)), 1e-300)
    errors = [float(np.linalg.norm(truncate(noisy, k).matrix - clean) / scale)
              for k in range(1, limit + 1)]
    best = int(np.argmin(errors)) + 1
    return {"errors": np.array(errors), "best_rank": best,
            "best_error": errors[best - 1],
            "no_denoising": float(np.linalg.norm(noisy - clean) / scale),
            "noise_floor": float(noise_level * np.sqrt(min(clean.shape)))}


def pagerank(links, damping: float = 0.85, tol: float = 1e-12,
             max_iter: int = 1000) -> dict:
    """The stationary distribution of a damped random walk: power iteration, at web scale.

    ``links[i, j] = 1`` means page ``j`` links to page ``i``. The Google matrix is

        G = d * P + (1 - d) / n * ones,

    with ``P`` the column-stochastic link matrix and dangling columns replaced by uniform ones.
    Its dominant eigenvector is the ranking.

    **It is lesson 36's power iteration and nothing more**, and the whole reason it works at web
    scale is that ``G`` is never formed: the rank-one damping term is applied as a scalar
    correction, so each step costs one sparse product. The damping ``d = 0.85`` is what makes it
    converge quickly, since the second eigenvalue of ``G`` is at most ``d``.
    """
    L = np.atleast_2d(np.asarray(links, dtype=float))
    n = L.shape[0]
    if L.shape[0] != L.shape[1]:
        raise ValueError(f"the link matrix is {L.shape}, it must be square")
    d = float(damping)
    if not 0.0 < d < 1.0:
        raise ValueError(f"damping must be strictly between 0 and 1, got {d}")
    out_degree = L.sum(axis=0)
    dangling = out_degree == 0
    P = np.where(dangling[None, :], 1.0 / n, L / np.where(out_degree > 0, out_degree, 1.0))

    x = np.full(n, 1.0 / n)
    history = []
    for step in range(int(max_iter)):
        # G x = d P x + (1-d)/n, with the rank one term applied as a scalar.
        y = d * (P @ x) + (1.0 - d) / n
        y /= y.sum()
        shift = float(np.linalg.norm(y - x, 1))
        history.append(shift)
        x = y
        if shift <= tol:
            return {"rank": x, "iterations": step + 1, "converged": True,
                    "history": history, "order": np.argsort(x)[::-1]}
    return {"rank": x, "iterations": int(max_iter), "converged": False,
            "history": history, "order": np.argsort(x)[::-1]}


def google_matrix(links, damping: float = 0.85) -> np.ndarray:
    """The dense Google matrix, for checking `pagerank` against a direct eigensolve.

    **Never built at scale**: it is ``n x n`` and completely dense, so for the real web it would
    be ``10^{22}`` entries. It exists here only so the iteration can be verified.
    """
    L = np.atleast_2d(np.asarray(links, dtype=float))
    n = L.shape[0]
    d = float(damping)
    out_degree = L.sum(axis=0)
    dangling = out_degree == 0
    P = np.where(dangling[None, :], 1.0 / n, L / np.where(out_degree > 0, out_degree, 1.0))
    return d * P + (1.0 - d) / n


def random_web(n: int, density: float = 0.05, n_dangling: int = 0, rng=None) -> np.ndarray:
    """A random link matrix with a controllable number of dangling pages."""
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    n_dangling = int(n_dangling)
    if not 0 <= n_dangling < n:
        raise ValueError(f"n_dangling must be between 0 and {n - 1}, got {n_dangling}")
    gen = np.random.default_rng() if rng is None else rng
    L = (gen.random((n, n)) < float(density)).astype(float)
    np.fill_diagonal(L, 0.0)
    for j in range(n):
        if j < n_dangling:
            L[:, j] = 0.0
        elif L[:, j].sum() == 0:
            target = (j + 1) % n
            L[target, j] = 1.0
    return L
