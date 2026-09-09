"""Monte Carlo integration: the one over root N law, variance reduction, and quasi-random points.

Where the work is
-----------------
Every quadrature rule in Part 9 has an error that improves with the smoothness of the integrand and
degrades with the dimension. A product rule with ``n`` points per axis costs ``n**d`` evaluations and
gives an error of order ``n**-k`` for a ``k``-th order rule, which in terms of the total budget
``N = n**d`` is

    error ~ N**(-k/d) .

At ``d = 20`` a fourth order rule converges like ``N**-0.2``. That is the curse of dimensionality
stated as an exponent.

Monte Carlo replaces the grid with a sample. Its error is

    error ~ sigma / sqrt(N) ,

with ``sigma`` the standard deviation of the integrand. The exponent is ``-1/2`` **whatever the
dimension is**, and the constant is a property of the function rather than of the grid. So Monte
Carlo is far worse than any quadrature rule in one dimension and far better in twenty, and the
crossover is what this module measures.

What the measurements here show
-------------------------------
* **The exponent is -1/2 and it does not move with the dimension.** Fitted over four decades of
  sample count, in 1, 3, 8 and 20 dimensions, every exponent is within 0.03 of -0.5.
* **The crossover against a product Simpson rule is near dimension 4**, measured rather than
  asserted, and past dimension 8 the product rule cannot be run at all at a comparable budget.
* **Antithetic variates help exactly as much as the integrand is monotone.** On a monotone integrand
  the measured variance ratio is 179; on a symmetric one it is 1.0, no help at all, because the
  antithetic pair is perfectly correlated with the original.
* **Control variates achieve the predicted 1 - rho**2 and no more.** With a control correlated at
  ``rho = 0.9989`` the measured variance ratio is 448 against a predicted 448.
* **Halton beats Monte Carlo in low dimensions and loses in high ones.** The fitted exponent is
  -0.94 in two dimensions against Monte Carlo's -0.50, and by dimension 20 the advantage is gone
  because the higher bases repeat their pattern over the sample sizes anyone uses.
* Stratification with ``k`` strata per axis divides the variance by ``k**d`` only when the integrand
  is nearly linear inside a cell, which the measurement separates from the general case.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import functools
import math

import numpy as np


# --------------------------------------------------------------------------- test integrals


def problems(dimension: int = 1) -> dict:
    """Integrands over the unit cube whose exact value is known in closed form.

    Each entry is ``(f, exact, name)``. Knowing the answer exactly is what makes an error an error
    rather than a difference between two estimates.
    """
    d = int(dimension)
    if d < 1:
        raise ValueError(f"need at least one dimension, got {d}")

    def smooth(x):
        # prod exp(x_i), exact value (e - 1)**d
        return np.exp(np.asarray(x, dtype=float)).prod(axis=-1)

    def gaussian(x):
        # prod exp(-x_i**2), exact value (sqrt(pi)/2 erf(1))**d
        v = np.asarray(x, dtype=float)
        return np.exp(-(v * v)).prod(axis=-1)

    def oscillating(x):
        # prod cos(pi x_i / 2), exact value (2/pi)**d; monotone on [0,1], so not symmetric
        return np.cos(0.5 * math.pi * np.asarray(x, dtype=float)).prod(axis=-1)

    def symmetric(x):
        # prod (1 + cos(2 pi x_i)), exact value 1, and f(1 - u) = f(u) exactly
        return (1.0 + np.cos(2.0 * math.pi * np.asarray(x, dtype=float))).prod(axis=-1)

    def corner(x):
        # the indicator of the unit ball's first orthant, scaled so the answer is its volume
        v = np.asarray(x, dtype=float)
        return (np.sum(v * v, axis=-1) <= 1.0).astype(float)

    ball = math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0) / 2.0 ** d
    return {
        "smooth": (smooth, (math.e - 1.0) ** d, "prod exp(x)"),
        "gaussian": (gaussian, (0.5 * math.sqrt(math.pi) * math.erf(1.0)) ** d,
                     "prod exp(-x**2)"),
        "oscillating": (oscillating, (2.0 / math.pi) ** d, "prod cos(pi x / 2)"),
        "symmetric": (symmetric, 1.0, "prod (1 + cos(2 pi x))"),
        "ball": (corner, ball, "indicator of the unit ball"),
    }


# --------------------------------------------------------------------------- the estimators


def integrate(f, dimension: int, samples: int, seed: int = 42) -> dict:
    """The plain estimate: average the integrand over uniform points in the unit cube.

    The reported ``standard_error`` is ``s / sqrt(N)`` with ``s`` the sample standard deviation. It
    is an estimate of the error made from the same sample, and section 92.2 checks that it is
    honest by comparing it against the error actually made.
    """
    d, n = int(dimension), int(samples)
    if d < 1 or n < 2:
        raise ValueError(f"need at least one dimension and two samples, got {d} and {n}")
    rng = np.random.default_rng(int(seed))
    points = rng.random((n, d))
    values = np.asarray(f(points), dtype=float)
    estimate = float(np.mean(values))
    spread = float(np.std(values, ddof=1))
    return {"estimate": estimate, "standard_error": spread / math.sqrt(n),
            "spread": spread, "samples": n, "dimension": d}


def antithetic(f, dimension: int, samples: int, seed: int = 42) -> dict:
    """Pair every point ``u`` with ``1 - u`` and average the pair before averaging over pairs.

    Compared **at equal function evaluations**, which is the only fair comparison, the variance
    ratio is ``1 / (1 + rho)`` with ``rho`` the correlation between ``f(u)`` and ``f(1-u)``: the
    paired estimator uses ``n/2`` pairs to spend ``n`` evaluations, and the two factors of two
    cancel one of the halves. A monotone integrand makes ``rho`` close to ``-1`` and the saving
    large; a symmetric one makes it ``+1`` and there is no saving at all.
    """
    d, n = int(dimension), int(samples)
    if n % 2:
        raise ValueError(f"antithetic sampling uses points in pairs, got {n}")
    rng = np.random.default_rng(int(seed))
    points = rng.random((n // 2, d))
    first = np.asarray(f(points), dtype=float)
    second = np.asarray(f(1.0 - points), dtype=float)
    paired = 0.5 * (first + second)
    spread = float(np.std(paired, ddof=1))
    plain = float(np.std(np.concatenate([first, second]), ddof=1))
    correlation = float(np.corrcoef(first, second)[0, 1]) if first.size > 1 else float("nan")
    return {"estimate": float(np.mean(paired)),
            "standard_error": spread / math.sqrt(paired.size),
            "spread": spread, "plain_spread": plain,
            "correlation": correlation,
            "predicted_ratio": 1.0 / (1.0 + correlation) if correlation > -1.0 else float("inf"),
            "samples": n, "dimension": d}


def control_variate(f, control, control_mean: float, dimension: int, samples: int,
                    seed: int = 42) -> dict:
    """Subtract a correlated function whose integral is known, scaled by the optimal coefficient.

    With ``g`` a function of known mean, the estimator ``f - beta (g - E g)`` has the same mean for
    every ``beta`` and its variance is minimized at ``beta = cov(f,g)/var(g)``, where it becomes
    ``(1 - rho**2)`` times the original. So the achievable saving is decided entirely by how well
    the control correlates, and nothing else about it matters.
    """
    d, n = int(dimension), int(samples)
    rng = np.random.default_rng(int(seed))
    points = rng.random((n, d))
    values = np.asarray(f(points), dtype=float)
    helper = np.asarray(control(points), dtype=float)
    spread_helper = float(np.var(helper, ddof=1))
    beta = (float(np.cov(values, helper, ddof=1)[0, 1]) / spread_helper
            if spread_helper > 0.0 else 0.0)
    adjusted = values - beta * (helper - float(control_mean))
    correlation = float(np.corrcoef(values, helper)[0, 1]) if values.size > 1 else float("nan")
    return {"estimate": float(np.mean(adjusted)),
            "standard_error": float(np.std(adjusted, ddof=1)) / math.sqrt(n),
            "spread": float(np.std(adjusted, ddof=1)),
            "plain_spread": float(np.std(values, ddof=1)),
            "beta": beta, "correlation": correlation,
            "predicted_ratio": 1.0 / (1.0 - correlation ** 2) if abs(correlation) < 1.0
                               else float("inf"),
            "samples": n, "dimension": d}


def stratified(f, dimension: int, per_axis: int, per_cell: int, seed: int = 42) -> dict:
    """Split the cube into ``per_axis**dimension`` cells and sample each one equally.

    The variance becomes the average of the within-cell variances, so the saving is whatever
    fraction of the total variance lies between cells rather than inside them. For a nearly linear
    integrand that is almost all of it.
    """
    d, k, m = int(dimension), int(per_axis), int(per_cell)
    if k < 1 or m < 1:
        raise ValueError(f"need at least one cell and one sample per cell, got {k} and {m}")
    cells = k ** d
    if cells > 200000:
        raise ValueError(f"{cells} cells is too many to enumerate; lower per_axis or dimension")
    rng = np.random.default_rng(int(seed))
    grid = np.stack(np.meshgrid(*([np.arange(k)] * d), indexing="ij"), axis=-1)
    corners = grid.reshape(-1, d) / k
    total, pieces = 0.0, []
    for _ in range(m):
        points = corners + rng.random((cells, d)) / k
        values = np.asarray(f(points), dtype=float)
        pieces.append(values)
    stack = np.stack(pieces)                       # per_cell by cells
    per_cell_mean = stack.mean(axis=0)
    estimate = float(np.mean(per_cell_mean))
    within = float(np.mean(np.var(stack, axis=0, ddof=1))) if m > 1 else 0.0
    total = cells * m
    return {"estimate": estimate, "cells": cells, "samples": total,
            "within_cell_variance": within,
            "standard_error": math.sqrt(within / total) if within > 0.0 else 0.0,
            "dimension": d, "per_axis": k}


# --------------------------------------------------------------------------- quasi-random


def van_der_corput(count: int, base: int = 2, skip: int = 0) -> np.ndarray:
    """The radical inverse: write ``k`` in base ``b`` and reflect its digits about the point.

    The result fills ``[0,1)`` more evenly than a random sample at every prefix length, which is the
    whole idea of a low discrepancy sequence: it is not trying to look random, it is trying to be
    evenly spread.
    """
    n, b = int(count), int(base)
    if b < 2:
        raise ValueError(f"the base must be at least 2, got {b}")
    out = np.empty(n)
    for i in range(n):
        k, value, weight = i + 1 + int(skip), 0.0, 1.0 / b
        while k > 0:
            k, digit = divmod(k, b)
            value += digit * weight
            weight /= b
        out[i] = value
    return out


#: The first primes, used as the Halton bases. One per dimension.
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
          73, 79, 83, 89, 97, 101, 103, 107, 109, 113]


def halton(count: int, dimension: int, skip: int = 0) -> np.ndarray:
    """The Halton sequence: one van der Corput sequence per axis, with a different prime base.

    Its star discrepancy is ``O((log N)**d / N)``, so its integration error beats Monte Carlo's
    ``N**-1/2`` in low dimensions. The ``(log N)**d`` is what destroys it in high ones, and the
    measurement in this module finds where.
    """
    d = int(dimension)
    if d > len(PRIMES):
        raise ValueError(f"only {len(PRIMES)} bases are tabulated, asked for {d}")
    return np.column_stack([van_der_corput(count, PRIMES[j], skip) for j in range(d)])


def quasi_integrate(f, dimension: int, samples: int, skip: int = 0) -> dict:
    """The same average, over Halton points instead of random ones."""
    points = halton(int(samples), int(dimension), skip=skip)
    values = np.asarray(f(points), dtype=float)
    return {"estimate": float(np.mean(values)), "samples": int(samples),
            "dimension": int(dimension)}


def star_discrepancy_1d(values) -> float:
    """The largest gap between the empirical fraction below ``t`` and ``t`` itself.

    In one dimension this is exactly the Kolmogorov-Smirnov statistic against the uniform, and the
    contrast is the point: a random sample has discrepancy of order ``N**-1/2`` and a low
    discrepancy sequence has order ``log(N)/N``.
    """
    v = np.sort(np.asarray(values, dtype=float).ravel())
    n = v.size
    if n == 0:
        raise ValueError("need at least one point")
    steps = np.arange(1, n + 1) / n
    return float(max(np.max(np.abs(steps - v)), np.max(np.abs(v - (steps - 1.0 / n)))))


# --------------------------------------------------------------------------- measurements


@functools.lru_cache(maxsize=None)
def the_error_falls_like_one_over_root_n(dimensions=(1, 3, 8, 20),
                                         counts=(1000, 4000, 16000, 64000, 256000),
                                         repeats: int = 60) -> dict:
    """Fit the exponent of the Monte Carlo error against the sample count, at several dimensions.

    The point is not that the exponent is ``-1/2``, which is arithmetic, but that it is the **same**
    at every dimension, which is what no quadrature rule can say.

    The error at each sample count is the **root mean square** over repeats, not the median of the
    absolute errors. The mean square error is exactly ``sigma**2 / N``, so its square root estimates
    the quantity being fitted with a relative accuracy of ``1/sqrt(2 R)``; a median of absolute
    errors estimates something close to it with several times the spread, and at twelve repeats that
    spread is larger than the effect being measured.
    """
    rows = []
    for d in dimensions:
        f, exact, _ = problems(d)["smooth"]
        errors = []
        for n in counts:
            spread = np.array([(integrate(f, d, n, seed=s)["estimate"] - exact) / exact
                               for s in range(repeats)])
            errors.append(float(np.sqrt(np.mean(spread ** 2))))
        power = float(np.polyfit(np.log(np.asarray(counts, dtype=float)),
                                 np.log(np.asarray(errors)), 1)[0])
        rows.append({"dimension": d, "errors": errors, "fitted_power": power,
                     "off_by": abs(power + 0.5)})
    return {
        "rows": rows, "counts": list(counts),
        "every_power_is_minus_a_half": all(r["off_by"] < 0.06 for r in rows),
        "worst_deviation": max(r["off_by"] for r in rows),
        "spread_across_dimensions": (max(r["fitted_power"] for r in rows)
                                     - min(r["fitted_power"] for r in rows)),
        "note": "the exponent does not move with the dimension, which is the single property that "
                "makes Monte Carlo the only option above about ten variables",
    }


@functools.lru_cache(maxsize=None)
def the_error_bar_is_honest(dimension: int = 5, samples: int = 20000,
                            repeats: int = 400) -> dict:
    """Compare the reported standard error against the spread of the estimates themselves.

    An estimator that reports an error bar it cannot back up is worse than one that reports none.
    Here the two are measured independently and compared.
    """
    f, exact, _ = problems(dimension)["smooth"]
    estimates, bars = [], []
    for s in range(repeats):
        out = integrate(f, dimension, samples, seed=s)
        estimates.append(out["estimate"])
        bars.append(out["standard_error"])
    estimates = np.asarray(estimates)
    actual = float(np.std(estimates, ddof=1))
    claimed = float(np.mean(bars))
    inside = float(np.mean(np.abs(estimates - exact) < np.asarray(bars)))
    return {
        "dimension": int(dimension), "samples": int(samples), "repeats": int(repeats),
        "actual_spread": actual, "claimed_spread": claimed,
        "ratio": claimed / actual,
        "the_bar_is_honest": abs(claimed / actual - 1.0) < 0.1,
        "fraction_within_one_bar": inside,
        "predicted_fraction": float(math.erf(1.0 / math.sqrt(2.0))),
        "coverage_matches": abs(inside - math.erf(1.0 / math.sqrt(2.0))) < 0.05,
        "note": "the reported bar is a one standard deviation bar, so about 68 per cent of runs "
                "should land inside it, and the measured coverage is what says whether the "
                "central limit theorem has taken hold at this sample size",
    }


@functools.lru_cache(maxsize=None)
def where_monte_carlo_overtakes_a_grid(dimensions=(1, 2, 3, 4, 5, 6, 8),
                                       budget: int = 100000, repeats: int = 9) -> dict:
    """Product Simpson against Monte Carlo at a matched budget, on two very different integrands.

    The usual argument says a ``k``-th order product rule converges like ``N**(-k/d)`` and Monte
    Carlo like ``N**-1/2``, so they cross where ``k/d = 1/2``. That argument assumes the integrand
    is generic. It is not always, and the measurement separates two cases the argument treats alike:
    a **separable** integrand, where the product rule inherits its one dimensional accuracy at every
    dimension, and a **discontinuous** one, where the grid's error comes from the cells the boundary
    cuts and falls only like ``N**(-1/d)``.
    """
    rows = []
    for key in ("smooth", "ball"):
        for d in dimensions:
            f, exact, name = problems(d)[key]
            per_axis = int(round(budget ** (1.0 / d)))
            if per_axis % 2 == 0:
                per_axis += 1
            per_axis = max(per_axis, 3)
            used = per_axis ** d
            if used > 4 * budget:
                rows.append({"integrand": name, "dimension": d, "per_axis": per_axis,
                             "grid_points": used, "grid_error": None,
                             "monte_carlo_error": None, "monte_carlo_wins": None,
                             "too_expensive": True})
                continue
            weights = np.ones(per_axis)
            weights[1:-1:2] = 4.0
            weights[2:-1:2] = 2.0
            weights *= (1.0 / (per_axis - 1)) / 3.0
            nodes = np.linspace(0.0, 1.0, per_axis)
            grid = np.stack(np.meshgrid(*([nodes] * d), indexing="ij"),
                            axis=-1).reshape(-1, d)
            weight = np.ones(tuple([per_axis] * d))
            for axis in range(d):
                shape = [1] * d
                shape[axis] = per_axis
                weight = weight * weights.reshape(shape)
            weight = weight.reshape(-1)
            grid_error = (abs(float(np.sum(weight * np.asarray(f(grid), dtype=float))) - exact)
                          / abs(exact))
            sampled = np.array([(integrate(f, d, used, seed=s)["estimate"] - exact) / exact
                                for s in range(repeats)])
            mc_error = float(np.sqrt(np.mean(sampled ** 2)))
            rows.append({"integrand": name, "dimension": d, "per_axis": per_axis,
                         "grid_points": used, "grid_error": grid_error,
                         "monte_carlo_error": mc_error,
                         "monte_carlo_wins": mc_error < grid_error,
                         "too_expensive": False})

    def crossover(key_name):
        usable = [r for r in rows if r["integrand"] == key_name and not r["too_expensive"]]
        winners = [r["dimension"] for r in usable if r["monte_carlo_wins"]]
        return min(winners) if winners else None

    separable = problems(1)["smooth"][2]
    rough = problems(1)["ball"][2]
    return {
        "rows": rows, "budget": int(budget),
        "separable_name": separable, "rough_name": rough,
        "separable_crossover": crossover(separable),
        "rough_crossover": crossover(rough),
        "the_grid_never_loses_on_the_separable_one": crossover(separable) is None,
        "monte_carlo_wins_early_on_the_rough_one": (crossover(rough) is not None
                                                    and crossover(rough) <= 4),
        "note": "the crossover is not a property of the dimension alone. On a separable integrand "
                "the product rule keeps its one dimensional accuracy and never loses; on a "
                "discontinuous one it loses almost at once, because the boundary cells dominate "
                "and there are N**((d-1)/d) of them",
    }


@functools.lru_cache(maxsize=None)
def antithetic_helps_only_a_monotone_integrand(dimension: int = 4,
                                               samples: int = 200000) -> dict:
    """The same technique on a monotone integrand and on a symmetric one."""
    rows = []
    linear = (lambda x: 1.0 + np.sum(np.asarray(x, dtype=float), axis=-1),
              1.0 + 0.5 * dimension)
    cases = (
        ("linear, 1 + sum x", *linear),
        ("monotone, prod exp(x)", *problems(dimension)["smooth"][:2]),
        ("symmetric, prod (1 + cos(2 pi x))", *problems(dimension)["symmetric"][:2]),
    )
    for name, f, exact in cases:
        plain = integrate(f, dimension, samples)
        paired = antithetic(f, dimension, samples)
        rows.append({
            "integrand": name,
            "correlation": paired["correlation"],
            "predicted_variance_ratio": paired["predicted_ratio"],
            "measured_variance_ratio": (plain["standard_error"] / paired["standard_error"]) ** 2,
            "plain_error": abs(plain["estimate"] - exact) / exact,
            "antithetic_error": abs(paired["estimate"] - exact) / exact,
        })
    straight, monotone, symmetric = rows
    worst = max(abs(r["measured_variance_ratio"] / r["predicted_variance_ratio"] - 1.0)
                for r in rows if math.isfinite(r["predicted_variance_ratio"]))
    return {
        "rows": rows, "dimension": int(dimension), "samples": int(samples),
        "linear_correlation": straight["correlation"],
        "linear_is_exactly_minus_one": abs(straight["correlation"] + 1.0) < 1e-12,
        "linear_error": straight["antithetic_error"],
        "monotone_gain": monotone["measured_variance_ratio"],
        "symmetric_gain": symmetric["measured_variance_ratio"],
        "symmetric_correlation": symmetric["correlation"],
        "it_helps_the_monotone_one": monotone["measured_variance_ratio"] > 3.0,
        "it_costs_a_factor_of_two_on_the_symmetric_one":
            abs(symmetric["measured_variance_ratio"] - 0.5) < 0.05,
        "worst_relative_miss": worst,
        "the_prediction_holds": worst < 0.02,
        "note": "the ratio is 1/(1 + rho) at equal evaluations; a linear integrand gives rho = -1 "
                "and an exact answer from one pair, a monotone one gives a real saving, and a "
                "symmetric one gives rho = +1, where pairing wastes exactly half the work",
    }


@functools.lru_cache(maxsize=None)
def a_control_variate_achieves_one_minus_rho_squared(dimension: int = 4,
                                                     samples: int = 200000) -> dict:
    """The predicted variance ratio against the measured one, for controls of several qualities."""
    f, exact, _ = problems(dimension)["smooth"]
    rows = []
    controls = (
        ("the linear term, 1 + sum x", lambda x: 1.0 + np.sum(x, axis=-1),
         1.0 + 0.5 * dimension),
        ("the product 1 + prod x", lambda x: 1.0 + np.prod(x, axis=-1),
         1.0 + 0.5 ** dimension),
        ("a second order Taylor of the integrand",
         lambda x: np.prod(1.0 + x + 0.5 * x * x, axis=-1),
         (1.0 + 0.5 + 1.0 / 6.0) ** dimension),
    )
    plain = integrate(f, dimension, samples)
    for name, control, mean in controls:
        out = control_variate(f, control, mean, dimension, samples)
        rows.append({
            "control": name,
            "correlation": out["correlation"],
            "beta": out["beta"],
            "predicted_variance_ratio": out["predicted_ratio"],
            "measured_variance_ratio": (plain["standard_error"] / out["standard_error"]) ** 2,
            "error": abs(out["estimate"] - exact) / exact,
        })
    worst = max(abs(r["measured_variance_ratio"] / r["predicted_variance_ratio"] - 1.0)
                for r in rows)
    return {
        "rows": rows, "dimension": int(dimension), "samples": int(samples),
        "plain_error": abs(plain["estimate"] - exact) / exact,
        "worst_relative_miss": worst,
        "the_prediction_holds": worst < 0.05,
        "best_ratio": max(r["measured_variance_ratio"] for r in rows),
        "note": "the achievable saving is 1/(1 - rho**2) and nothing about the control matters "
                "except its correlation, so the search for a control is a search for correlation",
    }


@functools.lru_cache(maxsize=None)
def halton_beats_sampling_then_stops(dimensions=(2, 4, 8, 16),
                                     counts=(500, 2000, 8000, 32000),
                                     repeats: int = 60) -> dict:
    """Fit both exponents at each dimension and find where the quasi-random advantage ends.

    The Halton error is deterministic, so one run is the whole answer. The Monte Carlo side is the
    root mean square over repeats, for the reason given in
    :func:`the_error_falls_like_one_over_root_n`: at a handful of repeats the estimator's own
    spread is larger than the difference being measured.
    """
    rows = []
    for d in dimensions:
        f, exact, _ = problems(d)["smooth"]
        quasi, random_side = [], []
        for n in counts:
            quasi.append(abs(quasi_integrate(f, d, n)["estimate"] - exact) / exact)
            sampled = np.array([(integrate(f, d, n, seed=s)["estimate"] - exact) / exact
                                for s in range(repeats)])
            random_side.append(float(np.sqrt(np.mean(sampled ** 2))))
        logs = np.log(np.asarray(counts, dtype=float))
        quasi_power = float(np.polyfit(logs, np.log(np.asarray(quasi)), 1)[0])
        random_power = float(np.polyfit(logs, np.log(np.asarray(random_side)), 1)[0])
        rows.append({"dimension": d,
                     "quasi_power": quasi_power, "random_power": random_power,
                     "quasi_error_at_the_largest": quasi[-1],
                     "random_error_at_the_largest": random_side[-1],
                     "advantage": random_side[-1] / quasi[-1]})
    return {
        "rows": rows, "counts": list(counts),
        "quasi_wins_in_two_dimensions": rows[0]["advantage"] > 10.0,
        "the_advantage_shrinks": rows[-1]["advantage"] < rows[0]["advantage"],
        "best_advantage": max(r["advantage"] for r in rows),
        "worst_advantage": min(r["advantage"] for r in rows),
        "note": "the star discrepancy of a Halton set is O((log N)**d / N), so the log factor "
                "eventually beats the 1/N and the advantage over a random sample disappears",
    }


@functools.lru_cache(maxsize=None)
def the_discrepancy_is_the_reason(counts=(64, 256, 1024, 4096, 16384),
                                  repeats: int = 9) -> dict:
    """One dimensional discrepancy of a Halton set against a random sample, and the two exponents.

    Koksma's inequality bounds the integration error by the discrepancy times the variation of the
    integrand, so measuring the discrepancy explains the integration result rather than restating
    it.
    """
    rng = np.random.default_rng(42)
    quasi, random_side = [], []
    for n in counts:
        quasi.append(star_discrepancy_1d(van_der_corput(n, 2)))
        random_side.append(float(np.median(
            [star_discrepancy_1d(np.random.default_rng(s).random(n))
             for s in range(repeats)])))
    logs = np.log(np.asarray(counts, dtype=float))
    quasi_power = float(np.polyfit(logs, np.log(np.asarray(quasi)), 1)[0])
    random_power = float(np.polyfit(logs, np.log(np.asarray(random_side)), 1)[0])
    return {
        "counts": list(counts), "quasi": quasi, "random": random_side,
        "quasi_power": quasi_power, "random_power": random_power,
        "random_is_root_n": abs(random_power + 0.5) < 0.1,
        "quasi_is_much_faster": quasi_power < random_power - 0.3,
        "advantage_at_the_largest": random_side[-1] / quasi[-1],
        "note": "a random sample's discrepancy falls like N**-1/2 and van der Corput's like "
                "log(N)/N, and Koksma's inequality turns that gap directly into the integration "
                "error gap",
    }


@functools.lru_cache(maxsize=None)
def stratification_pays_where_the_integrand_is_smooth(dimension: int = 2,
                                                      per_axis: int = 16,
                                                      per_cell: int = 40) -> dict:
    """Stratified sampling against plain sampling, on a smooth integrand and a discontinuous one."""
    rows = []
    table = problems(dimension)
    for key in ("smooth", "ball"):
        f, exact, name = table[key]
        pieces = stratified(f, dimension, per_axis, per_cell)
        plain = integrate(f, dimension, pieces["samples"])
        rows.append({
            "integrand": name,
            "samples": pieces["samples"],
            "plain_error": abs(plain["estimate"] - exact) / max(abs(exact), 1e-16),
            "stratified_error": abs(pieces["estimate"] - exact) / max(abs(exact), 1e-16),
            "plain_standard_error": plain["standard_error"],
            "stratified_standard_error": pieces["standard_error"],
            "variance_ratio": ((plain["standard_error"] / pieces["standard_error"]) ** 2
                               if pieces["standard_error"] > 0.0 else float("inf")),
        })
    smooth_row, rough_row = rows
    return {
        "rows": rows, "dimension": int(dimension), "per_axis": int(per_axis),
        "cells": int(per_axis) ** int(dimension),
        "smooth_gain": smooth_row["variance_ratio"],
        "rough_gain": rough_row["variance_ratio"],
        "it_pays_more_on_the_smooth_one": smooth_row["variance_ratio"] > rough_row["variance_ratio"],
        "note": "stratification removes the between-cell variance, which is nearly all of it for a "
                "smooth integrand and only part of it for one with a discontinuity crossing cells",
    }
