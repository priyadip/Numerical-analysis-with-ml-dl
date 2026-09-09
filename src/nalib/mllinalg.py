"""Where Part 5 and Part 6 turn up again inside a machine learning system.

Where the work is
-----------------
Nothing in this module is new mathematics. Principal component analysis is the singular value
decomposition of lesson 40 applied to a centred data matrix. Ridge regression is the regularized
least squares problem of lesson 33. A low-rank adapter is the truncated factorization of lesson 41.
The reason they get a lesson of their own is that the machine learning literature states each of
them in its own vocabulary, and the vocabulary hides three facts that this course already proved:

**The covariance matrix is the wrong object to form.** ``X.T @ X`` squares the condition number, and
lesson 33 measured the cost of that in digits. The eigenvalue route to PCA is exactly the normal
equations mistake wearing different clothes.

**Regularization is a filter on the spectrum.** Ridge multiplies the ``i``-th singular direction by
``sigma**2 / (sigma**2 + lam)``. Reading that one formula tells you everything ridge does: it leaves
the large directions alone and kills the small ones, which is why it fixes conditioning and why it
introduces bias.

**Low rank works because of how a matrix was made, not because matrices are low rank.** A random
matrix of the same shape has no usable low-rank approximation at all. The measurement below
separates the two cases and gives the numbers.

What the measurements here show
-------------------------------
* **The three routes to PCA are the same mathematics and not the same computation.** The relative
  error of the smallest principal value grows like the condition number for the SVD route, fitted at
  0.945, and like its square for the covariance route, fitted at 2.019. At a data condition number
  of 1e8 that is 2.8e-10 against 0.499.
* **The covariance route stops being defined**, not just inaccurate: from a condition number of 1e8
  the smallest eigenvalue of a positive definite matrix comes out negative, and its square root is
  not a number.
* **The Gram route is not a safe substitute.** It is worse than the covariance route here, at 0.93
  against 0.499, because it asks a 200 by 200 eigensolver to separate 12 tiny eigenvalues from 188
  zeros.
* **PCA is not scale invariant and whitening is.** Multiplying one feature by 1000 turns the leading
  principal direction through 45.4 degrees and hands it 100 per cent of the weight, while the
  whitened geometry does not move at all: 1.7e-13.
* **Whitening is not unique.** PCA and ZCA whitening both produce the identity covariance to 1.6e-15
  and differ by an exact rotation, orthogonal to 2.2e-15. ZCA is the one closest to doing nothing,
  at a distance of 5.86 from the identity against PCA whitening's 9.00.
* **An unregularized whitener fails on data it has not seen.** On its own sample it is exact, to
  2.4e-14. On fresh data from the same distribution its covariance is off by 1.34, and a penalty of
  1e-4 cuts that to 0.70. The training number carries no information at all.
* **Ridge is a spectral filter.** The measured filter factors match ``sigma**2/(sigma**2+lam)`` to
  2.3e-15, and the condition number is ``(sigma_max**2+lam)/(sigma_min**2+lam)`` exactly.
* **The augmented form is the stable one.** At ``lam = 1e-12`` the normal equations route is 76988
  times less accurate than the stacked least squares route, which is 4.9 digits, for the reason
  lesson 33 gave.
* **Ridge does not always help.** With no noise the best penalty is exactly zero, and at a noise
  level of 1e-6 it is still zero. Ridge answers noise, not conditioning.
* **A truncated SVD cannot be beaten**, and the alternatives lose by a measurable factor: a random
  projection by 1.65, the same projection with oversampling by 1.25, and a column subset by 2.41.
* **Full rank and effectively low rank are not in conflict.** A 256 by 256 matrix with a
  ``1/i`` spectrum has numerical rank 256, effective rank 4.99 and stable rank 1.64.
* **A trained update is low rank and a random one is not.** After ``k`` minibatch steps the update
  has numerical rank exactly ``min(k * batch, d)``, so a rank 8 adapter captures 100 per cent of an
  8-step update and 98.1 per cent of a 16-step one, while the same adapter captures 21.1 per cent of
  a random matrix of the same shape.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- test material


def random_orthogonal(size: int, seed: int = 42) -> np.ndarray:
    """A Haar-distributed orthogonal matrix, from the QR factorization of a Gaussian."""
    n = int(size)
    if n < 1:
        raise ValueError(f"need at least one dimension, got {n}")
    rng = np.random.default_rng(int(seed))
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def spectrum(count: int, kind: str = "power", decay: float = 1.0) -> np.ndarray:
    """A decreasing list of singular values, normalized so the largest one is 1.

    ``power`` gives ``(i+1)**-decay``, the shape an embedding table or a kernel matrix tends to
    have. ``exponential`` gives ``exp(-decay*i)``, much steeper. ``gap`` is flat then flat again,
    with one jump, which is the only case where a rank is unambiguous.
    """
    k = int(count)
    if k < 1:
        raise ValueError(f"need at least one singular value, got {k}")
    index = np.arange(k, dtype=float)
    name = str(kind).lower()
    if name == "power":
        values = (index + 1.0) ** (-float(decay))
    elif name == "exponential":
        values = np.exp(-float(decay) * index)
    elif name == "gap":
        cut = max(k // 4, 1)
        values = np.where(index < cut, 1.0, float(decay))
    else:
        raise ValueError(f"unknown spectrum kind {kind!r}")
    return values / values[0]


def matrix_with_spectrum(values, rows: int | None = None, seed: int = 42,
                         centred: bool = False) -> np.ndarray:
    """A matrix whose singular values are exactly ``values``, to rounding.

    With ``centred=True`` every column is built orthogonal to the all-ones vector, so subtracting
    the column means changes nothing and the spectrum survives centring exactly. That matters when
    the matrix is about to be used as a known-answer test for PCA.
    """
    s = np.asarray(values, dtype=float).ravel()
    k = s.size
    m = k if rows is None else int(rows)
    if m < k + (1 if centred else 0):
        raise ValueError(f"need at least {k} rows to carry {k} singular values, got {m}")
    if centred:
        rng = np.random.default_rng(int(seed))
        block = np.column_stack([np.ones(m), rng.standard_normal((m, k))])
        left = np.linalg.qr(block)[0][:, 1:k + 1]
    else:
        left = random_orthogonal(m, seed)[:, :k]
    right = random_orthogonal(k, seed + 1)
    return (left * s) @ right.T


def dataset(samples: int, features: int, kind: str = "power", decay: float = 1.0,
            noise: float = 0.0, seed: int = 42) -> dict:
    """A centred data matrix with a known spectrum, plus the noise that was added to it."""
    n, d = int(samples), int(features)
    if n < 2 or d < 1:
        raise ValueError(f"need at least two samples and one feature, got {n} and {d}")
    k = min(n - 1, d)
    values = spectrum(k, kind, decay) * math.sqrt(n)
    clean = matrix_with_spectrum(values, rows=n, seed=seed)
    if d > k:
        clean = np.hstack([clean, np.zeros((n, d - k))]) @ random_orthogonal(d, seed + 2)
    rng = np.random.default_rng(int(seed) + 3)
    added = float(noise) * rng.standard_normal((n, d))
    data = clean + added
    return {
        "data": data - data.mean(axis=0), "clean": clean - clean.mean(axis=0),
        "noise": added, "singular_values": values,
        "samples": n, "features": d, "rank": k,
    }


# --------------------------------------------------------------------------- principal components


def pca(data, components: int | None = None) -> dict:
    """Principal component analysis, as a singular value decomposition of the centred data.

    The data matrix is ``(samples, features)``. The directions come back as rows, largest variance
    first, which is the convention ``sklearn.decomposition.PCA`` uses for ``components_``.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"need a 2-D array with at least two rows, got shape {x.shape}")
    mean = x.mean(axis=0)
    centred = x - mean
    left, s, right = np.linalg.svd(centred, full_matrices=False)
    total = s.size if components is None else min(int(components), s.size)
    variance = s ** 2 / (x.shape[0] - 1)
    return {
        "mean": mean, "directions": right[:total], "scores": left[:, :total] * s[:total],
        "singular_values": s, "variance": variance,
        "explained": variance[:total] / variance.sum() if variance.sum() > 0 else variance[:total],
        "components": total, "route": "svd",
    }


def pca_by_covariance(data, components: int | None = None) -> dict:
    """The same thing through the eigenvalues of the covariance matrix. This is the fragile route."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"need a 2-D array with at least two rows, got shape {x.shape}")
    mean = x.mean(axis=0)
    centred = x - mean
    covariance = centred.T @ centred / (x.shape[0] - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    total = values.size if components is None else min(int(components), values.size)
    return {
        "mean": mean, "directions": vectors.T[:total], "scores": centred @ vectors[:, :total],
        "singular_values": np.sqrt(np.maximum(values, 0.0) * (x.shape[0] - 1)),
        "variance": values, "eigenvalues": values,
        "explained": values[:total] / values.sum() if values.sum() > 0 else values[:total],
        "components": total, "route": "covariance",
        "negative_eigenvalues": int(np.sum(values < 0.0)),
    }


def pca_by_gram(data, components: int | None = None) -> dict:
    """Through the Gram matrix instead, which is the cheap route when features outnumber samples."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"need a 2-D array with at least two rows, got shape {x.shape}")
    mean = x.mean(axis=0)
    centred = x - mean
    gram = centred @ centred.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    width = min(x.shape)
    values, vectors = np.maximum(values[order][:width], 0.0), vectors[:, order][:, :width]
    keep = values > 0.0
    s = np.sqrt(values)
    directions = np.zeros((values.size, x.shape[1]))
    directions[keep] = (vectors[:, keep] / s[keep]).T @ centred
    total = values.size if components is None else min(int(components), values.size)
    return {
        "mean": mean, "directions": directions[:total], "scores": vectors[:, :total] * s[:total],
        "singular_values": s, "variance": values / (x.shape[0] - 1),
        "explained": values[:total] / values.sum() if values.sum() > 0 else values[:total],
        "components": total, "route": "gram",
    }


def reconstruct(model: dict, scores=None) -> np.ndarray:
    """Map scores back to the original coordinates, which is the inverse of the projection."""
    z = model["scores"] if scores is None else np.asarray(scores, dtype=float)
    return z @ model["directions"] + model["mean"]


def subspace_angle(first, second) -> float:
    """The largest principal angle between two subspaces, in degrees, from lesson 39.

    ``arccos`` near 1 halves the available digits, exactly as lesson 05 described, so a zero
    angle comes back as about 1e-6 degrees rather than as 0.
    """
    a = np.linalg.qr(np.asarray(first, dtype=float).T)[0]
    b = np.linalg.qr(np.asarray(second, dtype=float).T)[0]
    cosines = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(cosines.min(), -1.0, 1.0))))


# --------------------------------------------------------------------------- whitening


def whiten(data, method: str = "pca", ridge: float = 0.0) -> dict:
    """Transform the data so its covariance is the identity.

    ``pca`` rotates to the principal axes and rescales. ``zca`` rescales and then rotates back, so
    the result stays in the original coordinate frame. Both give the identity covariance; they
    differ by an orthogonal factor, and that factor is the whole content of the choice.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"need a 2-D array with at least two rows, got shape {x.shape}")
    mean = x.mean(axis=0)
    centred = x - mean
    _, s, right = np.linalg.svd(centred, full_matrices=False)
    variance = s ** 2 / (x.shape[0] - 1)
    scale = 1.0 / np.sqrt(variance + float(ridge))
    name = str(method).lower()
    if name == "pca":
        matrix = right.T * scale
    elif name == "zca":
        matrix = (right.T * scale) @ right
    else:
        raise ValueError(f"unknown whitening method {method!r}")
    whitened = centred @ matrix
    covariance = whitened.T @ whitened / (x.shape[0] - 1)
    identity = np.eye(covariance.shape[0])
    return {
        "whitened": whitened, "matrix": matrix, "mean": mean, "covariance": covariance,
        "covariance_error": float(np.max(np.abs(covariance - identity))),
        "distance_from_identity": float(np.linalg.norm(matrix - identity, "fro")),
        "largest_amplification": float(scale.max()), "smallest_amplification": float(scale.min()),
        "method": name,
    }


# --------------------------------------------------------------------------- ridge regression


def ridge_filters(values, penalty: float) -> np.ndarray:
    """The factor ``sigma**2/(sigma**2+lam)`` that ridge applies to each singular direction."""
    s = np.asarray(values, dtype=float)
    return s ** 2 / (s ** 2 + float(penalty))


def ridge_svd(matrix, target, penalty: float) -> dict:
    """Ridge regression through the SVD, which is the definition the filter factors come from."""
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float)
    left, s, right = np.linalg.svd(a, full_matrices=False)
    lam = float(penalty)
    coefficients = right.T @ ((s / (s ** 2 + lam)) * (left.T @ b))
    return {
        "solution": coefficients, "filters": ridge_filters(s, lam), "singular_values": s,
        "residual": float(np.linalg.norm(a @ coefficients - b)),
        "condition": float((s[0] ** 2 + lam) / (s[-1] ** 2 + lam)),
        "route": "svd",
    }


def ridge_normal(matrix, target, penalty: float) -> dict:
    """Ridge through the normal equations, which is what almost every tutorial writes."""
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float)
    gram = a.T @ a + float(penalty) * np.eye(a.shape[1])
    coefficients = np.linalg.solve(gram, a.T @ b)
    return {
        "solution": coefficients, "gram": gram,
        "residual": float(np.linalg.norm(a @ coefficients - b)),
        "condition": float(np.linalg.cond(gram)),
        "route": "normal equations",
    }


def ridge_augmented(matrix, target, penalty: float) -> dict:
    """Ridge as one plain least squares problem, by stacking ``sqrt(lam) I`` under the matrix.

    This is the stable route, and it is stable for the reason lesson 33 gave: the condition number
    of the stacked matrix is the square root of the condition number of the Gram matrix.
    """
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float)
    stacked = np.vstack([a, math.sqrt(float(penalty)) * np.eye(a.shape[1])])
    padded = np.concatenate([b, np.zeros(a.shape[1])])
    coefficients, *_ = np.linalg.lstsq(stacked, padded, rcond=None)
    return {
        "solution": coefficients, "stacked": stacked,
        "residual": float(np.linalg.norm(a @ coefficients - b)),
        "condition": float(np.linalg.cond(stacked)),
        "route": "augmented",
    }


def ridge_path(matrix, target, penalties, truth=None) -> dict:
    """The solution and its error over a range of penalties, which is how ``lam`` gets chosen."""
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float)
    rows = []
    for lam in penalties:
        out = ridge_svd(a, b, lam)
        row = {"penalty": float(lam), "norm": float(np.linalg.norm(out["solution"])),
               "residual": out["residual"], "condition": out["condition"]}
        if truth is not None:
            row["error"] = float(np.linalg.norm(out["solution"] - np.asarray(truth, dtype=float)))
        rows.append(row)
    best = min(rows, key=lambda r: r.get("error", r["residual"]))
    return {"rows": rows, "best_penalty": best["penalty"], "best": best}


# --------------------------------------------------------------------------- low rank structure


def spectral_summary(values) -> dict:
    """Three different numbers that all get called "the rank" of a matrix with a decaying spectrum."""
    s = np.asarray(values, dtype=float)
    energy = s ** 2
    total = energy.sum()
    share = energy / total if total > 0 else energy
    nonzero = share[share > 0.0]
    entropy = float(-np.sum(nonzero * np.log(nonzero))) if nonzero.size else 0.0
    return {
        "effective_rank": float(math.exp(entropy)),
        "stable_rank": float(total / energy[0]) if energy[0] > 0 else 0.0,
        "numerical_rank": int(np.sum(s > s[0] * s.size * np.finfo(float).eps)),
        "condition": float(s[0] / s[-1]) if s[-1] > 0 else math.inf,
        "entropy": entropy,
    }


def energy_captured(values, rank: int) -> float:
    """The fraction of the squared Frobenius norm that the leading ``rank`` directions hold."""
    s = np.asarray(values, dtype=float)
    r = max(0, min(int(rank), s.size))
    total = float(np.sum(s ** 2))
    return float(np.sum(s[:r] ** 2) / total) if total > 0 else 0.0


def truncate(matrix, rank: int) -> dict:
    """The best rank-``r`` approximation, which lesson 41 proved is the truncated SVD."""
    a = np.asarray(matrix, dtype=float)
    left, s, right = np.linalg.svd(a, full_matrices=False)
    r = max(0, min(int(rank), s.size))
    approximation = (left[:, :r] * s[:r]) @ right[:r]
    tail = s[r:]
    return {
        "approximation": approximation, "rank": r,
        "spectral_error": float(tail[0]) if tail.size else 0.0,
        "frobenius_error": float(np.linalg.norm(tail)),
        "relative_error": float(np.linalg.norm(a - approximation) / np.linalg.norm(a)),
    }


def random_projection_approximation(matrix, rank: int, oversample: int = 0,
                                    seed: int = 42) -> dict:
    """A rank-``r`` approximation from a randomized range finder, as in lesson 41."""
    a = np.asarray(matrix, dtype=float)
    r = max(1, min(int(rank), min(a.shape)))
    width = min(r + max(0, int(oversample)), a.shape[1])
    rng = np.random.default_rng(int(seed))
    q, _ = np.linalg.qr(a @ rng.standard_normal((a.shape[1], width)))
    inner = truncate(q.T @ a, r)
    approximation = q @ inner["approximation"]
    return {
        "approximation": approximation, "rank": r,
        "relative_error": float(np.linalg.norm(a - approximation) / np.linalg.norm(a)),
    }


def column_subset_approximation(matrix, rank: int) -> dict:
    """A rank-``r`` approximation built from actual columns, chosen by pivoted QR (lesson 35)."""
    a = np.asarray(matrix, dtype=float)
    r = max(1, min(int(rank), min(a.shape)))
    chosen: list[int] = []
    residual = a.copy()
    for _ in range(r):
        norms = np.linalg.norm(residual, axis=0)
        index = int(np.argmax(norms))
        chosen.append(index)
        column = a[:, index]
        unit = column / np.linalg.norm(column) if np.linalg.norm(column) > 0 else column
        residual = residual - np.outer(unit, unit @ residual)
    basis = np.linalg.qr(a[:, chosen])[0]
    approximation = basis @ (basis.T @ a)
    return {
        "approximation": approximation, "columns": chosen, "rank": r,
        "relative_error": float(np.linalg.norm(a - approximation) / np.linalg.norm(a)),
    }


def adapter(rows: int, cols: int, rank: int, scale: float = 1.0, seed: int = 42) -> dict:
    """A low-rank adapter ``B @ A``, the object a LoRA layer adds to a frozen weight matrix."""
    m, n, r = int(rows), int(cols), int(rank)
    if min(m, n) < 1 or r < 1:
        raise ValueError(f"need positive shapes and rank, got {m}, {n}, {r}")
    rng = np.random.default_rng(int(seed))
    down = rng.standard_normal((r, n)) / math.sqrt(n)
    up = np.zeros((m, r))
    return {"up": up, "down": down, "rank": r, "scale": float(scale),
            "update": float(scale) * (up @ down),
            "parameters": r * (m + n), "dense_parameters": m * n}


def fit_adapter(update, rank: int) -> dict:
    """The best rank-``r`` adapter for a given update, which is again the truncated SVD."""
    target = np.asarray(update, dtype=float)
    cut = truncate(target, rank)
    left, s, right = np.linalg.svd(target, full_matrices=False)
    r = cut["rank"]
    return {
        "up": left[:, :r] * s[:r], "down": right[:r], "rank": r,
        "captured": energy_captured(s, r), "relative_error": cut["relative_error"],
        "parameters": r * (target.shape[0] + target.shape[1]),
        "dense_parameters": target.size,
    }


def train_linear_layer(features: int, outputs: int, samples: int, steps: int, batch: int,
                       step_size: float = 0.05, seed: int = 42) -> dict:
    """Minibatch gradient descent on a linear layer, returning the total change in the weights.

    The point of the return value is its rank. One minibatch gradient is
    ``inputs.T @ residual / batch``, a sum of ``batch`` outer products, so it has rank at most
    ``batch``. After ``steps`` of them the total update has rank at most ``steps * batch``, whatever
    the size of the weight matrix. That bound, and nothing about the data, is why a low-rank adapter
    can work.
    """
    d, k, n = int(features), int(outputs), int(samples)
    total_steps, size = int(steps), int(batch)
    if min(d, k, n, total_steps, size) < 1:
        raise ValueError("every dimension, step count and batch size has to be positive")
    rng = np.random.default_rng(int(seed))
    inputs = rng.standard_normal((n, d))
    truth = rng.standard_normal((d, k)) / math.sqrt(d)
    targets = inputs @ truth
    start = rng.standard_normal((d, k)) / math.sqrt(d)
    weights = start.copy()
    losses = []
    for step in range(total_steps):
        rows = rng.choice(n, size=min(size, n), replace=False)
        block, wanted = inputs[rows], targets[rows]
        residual = block @ weights - wanted
        losses.append(float(np.mean(residual ** 2)))
        weights = weights - float(step_size) * (block.T @ residual) / rows.size
    update = weights - start
    return {
        "update": update, "weights": weights, "start": start, "losses": losses,
        "rank_bound": min(total_steps * min(size, n), d, k),
        "numerical_rank": int(np.linalg.matrix_rank(update)),
        "singular_values": np.linalg.svd(update, compute_uv=False),
        "steps": total_steps, "batch": min(size, n),
    }


# --------------------------------------------------------------------------- measurements


def the_three_routes_to_pca_agree_until_they_do_not(
        conditions=(1e2, 1e4, 1e6, 1e8, 1e10, 1e12), samples: int = 200,
        features: int = 12, seed: int = 42) -> dict:
    """PCA by SVD, by the covariance matrix and by the Gram matrix, over a range of conditioning.

    All three are the same mathematics. Only one of them survives an ill-conditioned data matrix,
    and the reason is lesson 33's: forming ``X.T @ X`` squares the condition number, so it spends
    twice as many digits as the problem itself needs.
    """
    rows = []
    eps = float(np.finfo(float).eps)
    repeats = 5
    for target in conditions:
        exact = np.geomspace(1.0, 1.0 / float(target), min(int(features), int(samples) - 1))
        gathered = {"svd": [], "covariance": [], "gram": []}
        negatives, angle = 0, 0.0
        for offset in range(repeats):
            data = matrix_with_spectrum(exact, rows=int(samples), seed=seed + offset, centred=True)
            by_svd = pca(data)
            by_covariance = pca_by_covariance(data)
            by_gram = pca_by_gram(data)
            gathered["svd"].append(np.max(np.abs(by_svd["singular_values"] - exact) / exact))
            gathered["covariance"].append(
                np.max(np.abs(by_covariance["singular_values"] - exact) / exact))
            gathered["gram"].append(np.max(np.abs(by_gram["singular_values"] - exact) / exact))
            negatives += by_covariance["negative_eigenvalues"]
            angle = max(angle, subspace_angle(by_svd["directions"][:1],
                                              by_covariance["directions"][:1]))
        rows.append({
            "condition": float(target),
            "svd_error": float(np.sqrt(np.mean(np.square(gathered["svd"])))),
            "covariance_error": float(np.sqrt(np.mean(np.square(gathered["covariance"])))),
            "gram_error": float(np.sqrt(np.mean(np.square(gathered["gram"])))),
            "predicted_svd": eps * float(target),
            "predicted_covariance": eps * float(target) ** 2,
            "negative_eigenvalues": negatives,
            "leading_angle": angle,
        })
    logs = np.log10([r["condition"] for r in rows])
    svd_slope = float(np.polyfit(logs, np.log10([r["svd_error"] for r in rows]), 1)[0])
    usable = [r for r in rows if r["covariance_error"] < 1.0]
    covariance_slope = float(np.polyfit(np.log10([r["condition"] for r in usable]),
                                        np.log10([r["covariance_error"] for r in usable]), 1)[0])
    return {
        "rows": rows, "svd_slope": svd_slope, "covariance_slope": covariance_slope,
        "agree_when_well_conditioned": rows[0]["covariance_error"] < 1e-10,
        "covariance_route_fails": rows[-1]["covariance_error"] > 1e-2,
        "some_eigenvalue_goes_negative": any(r["negative_eigenvalues"] > 0 for r in rows),
        "svd_grows_like_the_condition_number": abs(svd_slope - 1.0) < 0.25,
        "covariance_grows_like_its_square": abs(covariance_slope - 2.0) < 0.25,
        "note": "the covariance route squares the condition number, so it runs out of digits at "
                "the square root of where the SVD would",
    }


def pca_is_not_scale_invariant(factors=(1.0, 10.0, 100.0, 1000.0), samples: int = 300,
                               features: int = 6, seed: int = 7) -> dict:
    """Multiply one feature by a constant and watch the principal directions move.

    Whitening is invariant under the same change, because it divides every direction by its own
    standard deviation. That is the practical argument for standardizing features before PCA, and it
    is a statement about units rather than about statistics.
    """
    base = dataset(int(samples), int(features), kind="power", decay=0.5, seed=seed)["data"]
    reference = pca(base, components=1)
    reference_gram = whiten(base, "zca")["whitened"] @ whiten(base, "zca")["whitened"].T
    scale = float(np.max(np.abs(reference_gram)))
    rows = []
    for factor in factors:
        scaled = base.copy()
        scaled[:, 0] = scaled[:, 0] * float(factor)
        model = pca(scaled, components=1)
        whitened = whiten(scaled, "zca")["whitened"]
        rows.append({
            "factor": float(factor),
            "angle": subspace_angle(reference["directions"], model["directions"]),
            "explained": float(model["explained"][0]),
            "leading_weight_on_the_scaled_feature": float(abs(model["directions"][0, 0])),
            "whitened_gram_change": float(np.max(np.abs(whitened @ whitened.T
                                                        - reference_gram)) / scale),
        })
    return {
        "rows": rows,
        "directions_move": rows[-1]["angle"] > 40.0,
        "the_scaled_feature_takes_over": rows[-1]["leading_weight_on_the_scaled_feature"] > 0.99,
        "whitening_is_invariant": max(r["whitened_gram_change"] for r in rows) < 1e-8,
        "note": "PCA reads the covariance in whatever units the features arrived in, while the "
                "whitened geometry does not depend on them at all",
    }


def whitening_is_not_unique(samples: int = 400, features: int = 8, seed: int = 3) -> dict:
    """PCA and ZCA whitening both give the identity covariance, and they are not the same map.

    They differ by exactly the orthogonal factor of the data, and ZCA is the choice that stays
    closest to the identity, which is the sense in which it "does the least".
    """
    data = dataset(int(samples), int(features), kind="power", decay=0.7, seed=seed)["data"]
    by_pca = whiten(data, "pca")
    by_zca = whiten(data, "zca")
    rotation = np.linalg.lstsq(by_pca["whitened"], by_zca["whitened"], rcond=None)[0]
    orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(rotation.shape[0]))))
    return {
        "pca_covariance_error": by_pca["covariance_error"],
        "zca_covariance_error": by_zca["covariance_error"],
        "pca_distance": by_pca["distance_from_identity"],
        "zca_distance": by_zca["distance_from_identity"],
        "rotation_orthogonality": orthogonality,
        "amplification_ratio": by_pca["largest_amplification"] / by_pca["smallest_amplification"],
        "both_whiten": max(by_pca["covariance_error"], by_zca["covariance_error"]) < 1e-10,
        "they_differ_by_a_rotation": orthogonality < 1e-8,
        "zca_is_closer_to_the_identity": by_zca["distance_from_identity"]
                                         < by_pca["distance_from_identity"],
        "note": "whitening is defined up to an orthogonal factor, so there are infinitely many "
                "whiteners and the choice is not statistical",
    }


def whitening_needs_a_penalty_to_generalize(penalties=(0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1),
                                            features: int = 20, decay: float = 2.0,
                                            held_out: int = 4000, seed: int = 11) -> dict:
    """Fit a whitener on a small sample, then apply it to fresh data from the same distribution.

    On its own training data every whitener produces the identity covariance exactly, by
    construction, so the training number says nothing. The held-out number says everything. The
    smallest sample direction is the one estimated worst and divided by the smallest number, so an
    unregularized whitener amplifies exactly the direction it knows least about.
    """
    d = int(features)
    deviations = spectrum(d, "power", float(decay))
    factor = random_orthogonal(d, seed) * deviations
    rng = np.random.default_rng(int(seed) + 1)
    training = rng.standard_normal((3 * d, d)) @ factor.T
    fresh = rng.standard_normal((int(held_out), d)) @ factor.T
    identity = np.eye(d)
    rows = []
    for lam in penalties:
        fitted = whiten(training, "zca", ridge=float(lam))
        moved = (fresh - fitted["mean"]) @ fitted["matrix"]
        covariance = moved.T @ moved / (moved.shape[0] - 1)
        rows.append({
            "penalty": float(lam), "amplification": fitted["largest_amplification"],
            "training_error": fitted["covariance_error"],
            "held_out_error": float(np.max(np.abs(covariance - identity))),
        })
    best = min(rows, key=lambda r: r["held_out_error"])
    return {
        "rows": rows, "best_penalty": best["penalty"], "best_error": best["held_out_error"],
        "unregularized_error": rows[0]["held_out_error"],
        "gain": rows[0]["held_out_error"] / best["held_out_error"],
        "training_error_says_nothing": max(r["training_error"] for r in rows[:1]) < 1e-10,
        "the_penalty_helps": best["penalty"] > 0.0,
        "too_much_penalty_hurts": rows[-1]["held_out_error"] > best["held_out_error"],
        "note": "the smallest sample direction is the worst estimated one, and whitening divides "
                "by it, so an unregularized whitener trusts its weakest measurement most",
    }


def ridge_is_a_spectral_filter(penalties=(0.0, 1e-8, 1e-4, 1e-2, 1.0), rows: int = 60,
                               cols: int = 12, seed: int = 5) -> dict:
    """Read ridge off the singular values: it multiplies direction ``i`` by ``s**2/(s**2+lam)``.

    Everything ridge does follows from that one line. Large directions are untouched, small ones are
    switched off, and the crossover is at ``s**2 = lam``.
    """
    values = np.geomspace(1.0, 1e-6, min(int(cols), int(rows)))
    matrix = matrix_with_spectrum(values, rows=int(rows), seed=seed)
    rng = np.random.default_rng(int(seed) + 1)
    truth = rng.standard_normal(matrix.shape[1])
    target = matrix @ truth
    left, s, right = np.linalg.svd(matrix, full_matrices=False)
    table = []
    for lam in penalties:
        out = ridge_svd(matrix, target, lam)
        plain = right.T @ ((left.T @ target) / s)
        measured = np.where(np.abs(plain) > 0, right @ out["solution"] / (right @ plain), np.nan)
        predicted = ridge_filters(s, lam)
        table.append({
            "penalty": float(lam),
            "filter_error": float(np.nanmax(np.abs(measured - predicted))),
            "smallest_filter": float(predicted[-1]), "largest_filter": float(predicted[0]),
            "condition": out["condition"],
            "predicted_condition": float((s[0] ** 2 + lam) / (s[-1] ** 2 + lam)),
            "kept": int(np.sum(predicted > 0.5)),
        })
    return {
        "rows": table, "singular_values": s,
        "filters_are_exact": max(r["filter_error"] for r in table) < 1e-12,
        "condition_formula_holds": max(abs(r["condition"] - r["predicted_condition"])
                                       / r["predicted_condition"] for r in table) < 1e-12,
        "conditioning_improves": table[-1]["condition"] < table[0]["condition"],
        "note": "ridge does not shrink the solution evenly; it shrinks the directions the data "
                "did not measure",
    }


def the_augmented_form_is_the_stable_one(penalties=(1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12),
                                         rows: int = 80, cols: int = 20, seed: int = 9) -> dict:
    """The same ridge problem three ways, at penalties small enough for conditioning to matter.

    The normal equations route works with ``A.T @ A + lam I``, whose condition number is the square
    of the stacked matrix's. That is lesson 33's argument with a shift added, and the shift only
    rescues it while ``lam`` is large enough to dominate the small singular values.
    """
    values = np.geomspace(1.0, 1e-7, min(int(cols), int(rows)))
    matrix = matrix_with_spectrum(values, rows=int(rows), seed=seed)
    rng = np.random.default_rng(int(seed) + 1)
    target = matrix @ rng.standard_normal(matrix.shape[1]) + 1e-3 * rng.standard_normal(int(rows))
    table = []
    for lam in penalties:
        exact = ridge_svd(matrix, target, lam)["solution"]
        scale = np.linalg.norm(exact)
        normal = ridge_normal(matrix, target, lam)
        stacked = ridge_augmented(matrix, target, lam)
        table.append({
            "penalty": float(lam),
            "normal_error": float(np.linalg.norm(normal["solution"] - exact) / scale),
            "augmented_error": float(np.linalg.norm(stacked["solution"] - exact) / scale),
            "normal_condition": normal["condition"],
            "augmented_condition": stacked["condition"],
        })
    worst = max(table, key=lambda r: r["normal_error"] / max(r["augmented_error"], 1e-300))
    return {
        "rows": table,
        "worst_penalty": worst["penalty"],
        "worst_gap": worst["normal_error"] / max(worst["augmented_error"], 1e-300),
        "digits_lost": math.log10(worst["normal_error"] / max(worst["augmented_error"], 1e-300)),
        "condition_is_squared": max(abs(r["normal_condition"] - r["augmented_condition"] ** 2)
                                    / r["normal_condition"] for r in table) < 1e-5,
        "worst_condition_mismatch": max(abs(r["normal_condition"] - r["augmented_condition"] ** 2)
                                        / r["normal_condition"] for r in table),
        "augmented_is_better": all(r["augmented_error"] <= r["normal_error"] + 1e-14
                                   for r in table),
        "note": "the stacked matrix has the square root of the Gram matrix's condition number, "
                "so it keeps twice as many digits",
    }


def ridge_does_not_always_help(noises=(0.0, 1e-6, 1e-3, 1e-1), penalties=None, rows: int = 80,
                               cols: int = 20, seed: int = 13) -> dict:
    """Sweep the penalty at several noise levels and find where the best penalty actually is.

    The textbook picture, a U-shaped error curve with a minimum in the middle, is what noise
    produces. Without noise the curve is monotone and the best penalty is zero, so ridge is a
    response to noise and not to conditioning on its own.
    """
    grid = (np.concatenate([[0.0], np.geomspace(1e-14, 1e2, 33)]) if penalties is None
            else np.asarray(penalties, dtype=float))
    values = np.geomspace(1.0, 1e-5, min(int(cols), int(rows)))
    matrix = matrix_with_spectrum(values, rows=int(rows), seed=seed)
    rng = np.random.default_rng(int(seed) + 1)
    truth = rng.standard_normal(matrix.shape[1])
    clean = matrix @ truth
    table = []
    for level in noises:
        perturbed = clean + float(level) * rng.standard_normal(int(rows))
        path = ridge_path(matrix, perturbed, grid, truth=truth)
        zero = ridge_svd(matrix, perturbed, 0.0)
        table.append({
            "noise": float(level), "best_penalty": path["best_penalty"],
            "best_error": path["best"]["error"],
            "error_at_zero": float(np.linalg.norm(zero["solution"] - truth)),
            "gain": float(np.linalg.norm(zero["solution"] - truth)) / path["best"]["error"],
        })
    return {
        "rows": table, "penalties": grid,
        "no_noise_wants_no_penalty": table[0]["best_penalty"] == 0.0,
        "noise_wants_a_penalty": table[-1]["best_penalty"] > 0.0,
        "best_penalty_grows_with_noise": all(
            table[i + 1]["best_penalty"] >= table[i]["best_penalty"]
            for i in range(len(table) - 1)),
        "largest_gain": max(r["gain"] for r in table),
        "note": "ridge trades bias for variance, and with no variance there is nothing to trade",
    }


def a_decaying_spectrum_has_no_single_rank(decays=(0.5, 1.0, 1.5, 2.0, 3.0), size: int = 256,
                                           seed: int = 17) -> dict:
    """Effective rank, stable rank and numerical rank on the same matrix, over a range of decays.

    An embedding table or an attention matrix has a power-law spectrum, so it is full rank and
    behaves as though it were not. Which of these three numbers to quote depends on what the rank
    was going to be used for, and they can disagree by two orders of magnitude.
    """
    table = []
    for decay in decays:
        values = spectrum(int(size), "power", float(decay))
        summary = spectral_summary(values)
        half = int(np.searchsorted(np.cumsum(values ** 2) / np.sum(values ** 2), 0.9) + 1)
        table.append({
            "decay": float(decay), "effective_rank": summary["effective_rank"],
            "stable_rank": summary["stable_rank"], "numerical_rank": summary["numerical_rank"],
            "rank_for_ninety_percent": half,
            "energy_in_the_top_sixteenth": energy_captured(values, max(int(size) // 16, 1)),
        })
    spread = max(r["numerical_rank"] / r["stable_rank"] for r in table)
    return {
        "rows": table, "size": int(size), "largest_spread": spread,
        "all_are_full_rank": all(r["numerical_rank"] == int(size) for r in table),
        "the_three_ranks_disagree": spread > 10.0,
        "steeper_decay_means_lower_effective_rank": all(
            table[i + 1]["effective_rank"] <= table[i]["effective_rank"]
            for i in range(len(table) - 1)),
        "note": "full rank and effectively low rank are not in conflict; they answer different "
                "questions about the same singular values",
    }


def nothing_beats_the_truncated_svd(ranks=(2, 4, 8, 16, 32), rows: int = 200, cols: int = 120,
                                    decay: float = 1.0, seed: int = 19) -> dict:
    """Compare the optimal rank-``r`` approximation against two cheaper ways of getting one.

    Eckart-Young says the truncated SVD is exactly optimal, so the only question about the
    alternatives is how much they lose. Oversampling by a few columns is what closes most of the
    gap for the randomized method, and that is the whole practical content of lesson 41.
    """
    values = spectrum(min(int(rows), int(cols)), "power", float(decay))
    matrix = matrix_with_spectrum(values, rows=int(rows), seed=seed)
    if int(cols) > values.size:
        matrix = np.hstack([matrix, np.zeros((int(rows), int(cols) - values.size))])
    table = []
    for r in ranks:
        best = truncate(matrix, r)
        sketch = random_projection_approximation(matrix, r, oversample=0, seed=seed + 1)
        padded = random_projection_approximation(matrix, r, oversample=r, seed=seed + 1)
        columns = column_subset_approximation(matrix, r)
        table.append({
            "rank": int(r), "optimal": best["relative_error"],
            "random": sketch["relative_error"], "oversampled": padded["relative_error"],
            "columns": columns["relative_error"],
            "random_ratio": sketch["relative_error"] / best["relative_error"],
            "oversampled_ratio": padded["relative_error"] / best["relative_error"],
            "column_ratio": columns["relative_error"] / best["relative_error"],
        })
    return {
        "rows": table,
        "optimal_is_optimal": all(r["optimal"] <= min(r["random"], r["columns"]) + 1e-12
                                  for r in table),
        "worst_random_ratio": max(r["random_ratio"] for r in table),
        "worst_oversampled_ratio": max(r["oversampled_ratio"] for r in table),
        "worst_column_ratio": max(r["column_ratio"] for r in table),
        "oversampling_helps": all(r["oversampled_ratio"] <= r["random_ratio"] + 1e-12
                                  for r in table),
        "note": "the truncated SVD is the answer, and the alternatives are answers to the "
                "question of what to do when you cannot afford it",
    }


def a_trained_update_is_low_rank_and_a_random_one_is_not(
        steps=(1, 2, 4, 8, 16), features: int = 128, outputs: int = 128, samples: int = 512,
        batch: int = 1, adapter_rank: int = 8, seed: int = 23) -> dict:
    """The rank of the change in a weight matrix, after a few minibatch steps and at random.

    One minibatch gradient is a sum of ``batch`` outer products, so ``k`` steps can move the weights
    in at most ``k * batch`` directions. That is an exact bound with no assumption about the data in
    it, and it is the reason a rank 8 adapter is not a wild guess. The comparison against a random
    matrix of the same shape is what stops the statement being circular.
    """
    table = []
    for count in steps:
        run = train_linear_layer(int(features), int(outputs), int(samples), int(count),
                                 int(batch), seed=seed)
        fitted = fit_adapter(run["update"], int(adapter_rank))
        table.append({
            "steps": int(count), "bound": run["rank_bound"],
            "numerical_rank": run["numerical_rank"],
            "captured_by_the_adapter": fitted["captured"],
        })
    rng = np.random.default_rng(int(seed) + 1)
    noise = rng.standard_normal((int(features), int(outputs)))
    random_fit = fit_adapter(noise, int(adapter_rank))
    dense = int(features) * int(outputs)
    inside = [r for r in table if r["bound"] <= int(adapter_rank)]
    outside = [r for r in table if r["bound"] > int(adapter_rank)]
    flat = int(adapter_rank) / min(int(features), int(outputs))
    return {
        "rows": table, "adapter_rank": int(adapter_rank),
        "random_captured": random_fit["captured"],
        "random_numerical_rank": int(np.linalg.matrix_rank(noise)),
        "flat_spectrum_share": flat,
        "captured_inside_the_budget": min(r["captured_by_the_adapter"] for r in inside),
        "captured_outside_the_budget": (min(r["captured_by_the_adapter"] for r in outside)
                                        if outside else float("nan")),
        "advantage_over_noise": min(r["captured_by_the_adapter"] for r in inside)
                                / random_fit["captured"],
        "parameter_ratio": dense / (int(adapter_rank) * (int(features) + int(outputs))),
        "the_bound_is_exact": all(r["numerical_rank"] == r["bound"] for r in table),
        "the_adapter_captures_what_fits": min(r["captured_by_the_adapter"]
                                              for r in inside) > 1.0 - 1e-12,
        "it_does_not_capture_noise": random_fit["captured"] < 0.25,
        "noise_is_not_flat_either": random_fit["captured"] > flat,
        "note": "low rank adaptation is a statement about the optimizer's path, not about the "
                "weight matrix",
    }
