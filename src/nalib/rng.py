"""Where random numbers come from, how the old generators fail, and how to catch one that does.

Where the work is
-----------------
Every method in Part 13 is an average over samples, so the samples have to be trustworthy. They are
not random. They come from a deterministic recurrence chosen so that its output is hard to tell
apart from random, and the whole subject is about what "hard to tell apart" is worth.

A linear congruential generator is the simplest such recurrence,

    x[k+1] = (a x[k] + c) mod m ,

and it is the one that shipped with almost every system for thirty years. Its period is at most
``m``, and it reaches ``m`` only under the Hull-Dobell conditions. Its low order bits have much
shorter periods than that. And every LCG's output, taken ``d`` at a time, lies on a lattice: the
points fall on a family of parallel hyperplanes, and if there are few enough of them the generator
is visibly not random.

What the measurements here show
-------------------------------
* **RANDU's triples lie on 15 planes.** IBM shipped it for a decade. The measurement finds the
  plane family from the recurrence identity ``x[k+2] = 6 x[k+1] - 9 x[k] (mod 2**31)``, which is
  exact in integer arithmetic, and counts the distinct plane indices directly.
* **The Hull-Dobell conditions are exactly right, not nearly right.** Over every multiplier and
  increment for a small modulus, the measured period equals the modulus in every case the
  conditions hold and in no case they do not.
* **The low bits of an LCG are much worse than the high bits.** With a power of two modulus, bit
  ``j`` counted from the bottom has period at most ``2**(j+1)``, so the last bit alternates. The
  measurement finds periods of 2, 4, 8 and 16 for the bottom four bits, exactly as predicted.
* **A generator can pass one test and fail another badly.** RANDU passes a chi-square test of its
  one dimensional uniformity outright, and fails a three dimensional one, which is the whole reason
  the spectral test exists.
* **Inverse transform is exact and Box-Muller is exact**, both to the accuracy of a
  Kolmogorov-Smirnov test at a million samples, and the difference between them is speed.
* The polar method rejects a known fraction of its draws, ``1 - pi/4 = 0.2146``, and the measured
  rejection rate matches to three digits.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import functools
import math

import numpy as np


# --------------------------------------------------------------------------- generators

#: Named linear congruential generators, as (multiplier, increment, modulus).
#: RANDU is the famous failure. The others are the ones the standard libraries actually used.
FAMOUS = {
    "randu": (65539, 0, 2 ** 31),
    "minstd": (16807, 0, 2 ** 31 - 1),
    "ansi c": (1103515245, 12345, 2 ** 31),
    "numerical recipes": (1664525, 1013904223, 2 ** 32),
    "borland": (22695477, 1, 2 ** 32),
}


def lcg(count: int, multiplier: int, increment: int, modulus: int, seed: int = 1):
    """``x[k+1] = (a x[k] + c) mod m``, returned as integers in ``[0, m)``.

    Exact integer arithmetic, so the recurrence is the recurrence and not a floating point
    approximation of it. That matters for lesson 91: every structural failure of an LCG is an
    integer identity, and seeing it needs the integers.
    """
    n = int(count)
    if n < 0:
        raise ValueError(f"need a non-negative count, got {n}")
    if modulus < 2:
        raise ValueError(f"the modulus must be at least 2, got {modulus}")
    out = np.empty(n, dtype=object)
    state = int(seed) % int(modulus)
    for k in range(n):
        state = (int(multiplier) * state + int(increment)) % int(modulus)
        out[k] = state
    return np.array([int(v) for v in out], dtype=np.int64)


def lcg_uniform(count: int, name: str = "randu", seed: int = 1) -> np.ndarray:
    """One of the named generators, scaled to ``[0, 1)``."""
    if name not in FAMOUS:
        raise ValueError(f"unknown generator {name!r}, expected one of {sorted(FAMOUS)}")
    a, c, m = FAMOUS[name]
    return lcg(count, a, c, m, seed=seed) / float(m)


def period(multiplier: int, increment: int, modulus: int, seed: int = 1) -> int:
    """The exact period, by following the sequence until a state repeats.

    Cheap only for a small modulus, which is the point: the structural facts about LCGs can be
    demonstrated at a modulus small enough to enumerate, and they carry over unchanged.
    """
    seen = {}
    state = int(seed) % int(modulus)
    step = 0
    while state not in seen:
        seen[state] = step
        state = (int(multiplier) * state + int(increment)) % int(modulus)
        step += 1
    return step - seen[state]


def hull_dobell(multiplier: int, increment: int, modulus: int) -> bool:
    """The three conditions under which an LCG has full period ``m``.

    ``c`` and ``m`` coprime; ``a - 1`` divisible by every prime factor of ``m``; and ``a - 1``
    divisible by 4 if ``m`` is. Exercise 91.2.1 proves them.
    """
    a, c, m = int(multiplier), int(increment), int(modulus)
    if math.gcd(c, m) != 1:
        return False
    for prime in _prime_factors(m):
        if (a - 1) % prime != 0:
            return False
    if m % 4 == 0 and (a - 1) % 4 != 0:
        return False
    return True


def _prime_factors(value: int):
    """The distinct primes dividing ``value``."""
    found, n, factor = [], int(value), 2
    while factor * factor <= n:
        if n % factor == 0:
            found.append(factor)
            while n % factor == 0:
                n //= factor
        factor += 1
    if n > 1:
        found.append(n)
    return found


# --------------------------------------------------------------------------- transforms


def inverse_transform_exponential(uniforms, rate: float = 1.0) -> np.ndarray:
    """``-log(1 - u) / rate``, the inverse of the exponential distribution function.

    The general recipe: if ``F`` is a distribution function and ``u`` is uniform on ``(0,1)`` then
    ``F^{-1}(u)`` has that distribution. It needs ``F^{-1}`` in closed form, which the exponential
    has and the normal does not, which is why the normal needs Box-Muller.
    """
    u = np.asarray(uniforms, dtype=float)
    if rate <= 0.0:
        raise ValueError(f"the rate must be positive, got {rate}")
    return -np.log1p(-u) / float(rate)


def box_muller(uniforms) -> np.ndarray:
    """Turn pairs of uniforms into pairs of standard normals, by the polar identity.

    Two independent uniforms give a radius and an angle whose Cartesian coordinates are two
    independent standard normals. It is exact, not an approximation, and it needs a logarithm, a
    square root and two trigonometric calls per pair.
    """
    u = np.asarray(uniforms, dtype=float).ravel()
    if u.size % 2:
        raise ValueError(f"box-muller consumes uniforms in pairs, got {u.size}")
    first, second = u[0::2], u[1::2]
    radius = np.sqrt(-2.0 * np.log1p(-first))
    angle = 2.0 * math.pi * second
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]).ravel()


def polar_normal(draws: int, seed: int = 42) -> dict:
    """Marsaglia's polar method: sample the square, reject outside the disc, no trigonometry.

    It replaces the sine and cosine with a rejection loop, and the acceptance probability is the
    ratio of the disc's area to the square's, ``pi/4``. So it wastes ``1 - pi/4`` of its uniforms
    and saves two trigonometric calls, and which is faster depends entirely on the machine.
    """
    wanted = int(draws)
    if wanted < 0:
        raise ValueError(f"need a non-negative count, got {wanted}")
    rng = np.random.default_rng(int(seed))
    out = np.empty(wanted)
    filled, tried, accepted = 0, 0, 0
    while filled < wanted:
        # one candidate pair per row, and an accepted pair yields two normals
        block = max((wanted - filled + 1) // 2, 1)
        first = rng.uniform(-1.0, 1.0, size=block)
        second = rng.uniform(-1.0, 1.0, size=block)
        square = first * first + second * second
        tried += block
        keep = (square > 0.0) & (square < 1.0)
        accepted += int(np.count_nonzero(keep))
        if not np.any(keep):
            continue
        radius = np.sqrt(-2.0 * np.log(square[keep]) / square[keep])
        pair = np.column_stack([first[keep] * radius, second[keep] * radius]).ravel()
        take = min(pair.size, wanted - filled)
        out[filled:filled + take] = pair[:take]
        filled += take
    return {"values": out, "pairs_tried": tried, "pairs_accepted": accepted,
            "acceptance": accepted / tried if tried else float("nan"),
            "predicted_acceptance": math.pi / 4.0}


# --------------------------------------------------------------------------- tests


def chi_square_uniformity(values, bins: int = 100) -> dict:
    """Count how many samples land in each of ``bins`` equal cells and compare with the expectation.

    The statistic is ``sum (observed - expected)**2 / expected``, which for a uniform sample has a
    chi-square distribution with ``bins - 1`` degrees of freedom. The mean of that distribution is
    ``bins - 1``, so the ratio of the statistic to its degrees of freedom is near one for a good
    generator and large for a bad one.
    """
    v = np.asarray(values, dtype=float).ravel()
    cells = int(bins)
    if cells < 2:
        raise ValueError(f"need at least two bins, got {cells}")
    if v.size == 0:
        raise ValueError("need at least one sample")
    counts = np.bincount(np.minimum((v * cells).astype(int), cells - 1), minlength=cells)
    expected = v.size / cells
    statistic = float(np.sum((counts - expected) ** 2) / expected)
    freedom = cells - 1
    return {"statistic": statistic, "degrees_of_freedom": freedom,
            "ratio": statistic / freedom,
            "looks_uniform": abs(statistic / freedom - 1.0) < 6.0 / math.sqrt(freedom)}


def kolmogorov_smirnov(values, reference=None) -> dict:
    """The largest gap between the empirical distribution function and the true one.

    For a sample of size ``n`` the statistic ``sqrt(n) D`` converges to a fixed distribution, so
    ``sqrt(n) D`` around 1 is ordinary and above about 1.63 is a one per cent event. Scaling by
    ``sqrt(n)`` is what makes the number comparable across sample sizes.
    """
    v = np.sort(np.asarray(values, dtype=float).ravel())
    n = v.size
    if n == 0:
        raise ValueError("need at least one sample")
    theory = v if reference is None else np.asarray([reference(t) for t in v], dtype=float)
    steps = np.arange(1, n + 1) / n
    gap = float(max(np.max(np.abs(steps - theory)),
                    np.max(np.abs(theory - (steps - 1.0 / n)))))
    return {"gap": gap, "scaled": gap * math.sqrt(n), "samples": n,
            "passes_at_one_percent": gap * math.sqrt(n) < 1.63}


def serial_correlation(values, lag: int = 1) -> float:
    """The correlation between the sequence and itself shifted by ``lag``.

    A generator whose successive outputs are correlated fails here even when its one dimensional
    distribution is perfect, which is the first hint that uniformity is not enough.
    """
    v = np.asarray(values, dtype=float).ravel()
    k = int(lag)
    if k < 1 or k >= v.size:
        raise ValueError(f"the lag must be between 1 and {v.size - 1}, got {k}")
    first, second = v[:-k], v[k:]
    first = first - first.mean()
    second = second - second.mean()
    bottom = math.sqrt(float(first @ first) * float(second @ second))
    return float(first @ second) / bottom if bottom > 0.0 else 0.0


def shortest_relation(multiplier: int, modulus: int, bound: int = 12) -> dict:
    """The shortest small integer ``(p, q, r)`` with ``p + q a + r a**2 = 0 (mod m)``.

    That vector is the normal of the plane family the triples lie on, and its length is what the
    spectral test measures. A short relation means few planes and a visibly structured generator; no
    short relation inside the search box means the test finds nothing at this size, which is the
    honest statement a bounded search can make.
    """
    a, m, box = int(multiplier), int(modulus), int(bound)
    if box < 1:
        raise ValueError(f"the search box must be at least 1, got {box}")
    square = (a * a) % m
    best = None
    for r in range(1, box + 1):
        for q in range(-box, box + 1):
            # p is then determined modulo m, so only a small p can satisfy the relation
            need = (-(q * a + r * square)) % m
            for p in (need, need - m):
                if abs(p) <= box:
                    length = p * p + q * q + r * r
                    if best is None or length < best[0]:
                        best = (length, (int(p), int(q), int(r)))
    if best is None:
        return {"found": False, "normal": None, "length": float("inf"),
                "bound": box, "multiplier": a, "modulus": m}
    return {"found": True, "normal": list(best[1]), "length": math.sqrt(best[0]),
            "bound": box, "multiplier": a, "modulus": m}


def planes_in_three_dimensions(name: str = "randu", draws: int = 30000,
                               seed: int = 1, bound: int = 12) -> dict:
    """How many parallel planes the consecutive triples of an LCG fall on.

    The plane normal is not asserted, it is found by searching for the shortest integer relation the
    recurrence satisfies. For RANDU that search returns ``(9, -6, 1)``, which is the identity

        x[k+2] - 6 x[k+1] + 9 x[k] = 0  (mod 2**31) ,

    and the number of planes is then the number of distinct values the combination takes divided by
    the modulus, counted from the sample.
    """
    if name not in FAMOUS:
        raise ValueError(f"unknown generator {name!r}, expected one of {sorted(FAMOUS)}")
    a, c, m = FAMOUS[name]
    relation = shortest_relation(a, m, bound=bound)
    raw = lcg(draws, a, c, m, seed=seed)
    if raw.size < 3:
        raise ValueError("need at least three draws to form a triple")
    if not relation["found"]:
        return {"generator": name, "multiplier": a, "modulus": m, "normal": None,
                "planes": None, "worst_residual": None, "exact": False,
                "draws": int(raw.size), "relation_length": float("inf")}
    triples = np.column_stack([raw[:-2], raw[1:-1], raw[2:]])
    normal = np.asarray(relation["normal"], dtype=np.int64)
    combination = triples @ normal
    indices = np.unique(combination // m)
    residual = int(np.max(np.abs(combination - (combination // m) * m
                                 - (combination % m))))
    exact = bool(np.all(combination % m == 0))
    return {"generator": name, "multiplier": a, "modulus": m,
            "normal": normal.tolist(), "planes": int(indices.size),
            "worst_residual": residual, "exact": exact,
            "relation_length": relation["length"], "draws": int(raw.size)}


# --------------------------------------------------------------------------- measurements


@functools.lru_cache(maxsize=None)
def randu_lies_on_a_few_planes(draws: int = 30000) -> dict:
    """The RANDU failure, counted rather than quoted.

    The usual sentence is "RANDU's triples lie on 15 planes". The relation that produces them is an
    exact integer identity, so the count is exact too, and the same construction applied to a
    modern generator finds no such relation at all.
    """
    out = planes_in_three_dimensions("randu", draws)
    rng = np.random.default_rng(42)
    modern = (rng.random(draws) * out["modulus"]).astype(np.int64)
    triples = np.column_stack([modern[:-2], modern[1:-1], modern[2:]])
    combination = triples @ np.asarray(out["normal"], dtype=np.int64)
    on_a_plane = int(np.sum(combination % out["modulus"] == 0))
    others = {name: planes_in_three_dimensions(name, draws) for name in sorted(FAMOUS)}
    return {
        "planes": out["planes"],
        "relation_is_exact": out["exact"],
        "worst_residual": out["worst_residual"],
        "normal": out["normal"],
        "relation_length": out["relation_length"],
        "pcg64_triples_on_the_same_planes": on_a_plane,
        "pcg64_triples_tested": int(triples.shape[0]),
        "the_relation_does_not_hold_for_pcg64": on_a_plane == 0,
        "rows": [{"generator": k, "normal": v["normal"], "planes": v["planes"],
                  "length": v["relation_length"]} for k, v in others.items()],
        "only_randu_has_a_short_relation": sum(
            1 for v in others.values() if v["normal"] is not None) == 1,
        "note": "the identity x[k+2] = 6 x[k+1] - 9 x[k] mod 2**31 is exact for RANDU, so the "
                "triples lie on exactly as many planes as the combination has values, and the "
                "same search finds no short relation for the other generators",
    }


@functools.lru_cache(maxsize=None)
def the_hull_dobell_conditions_are_exact(modulus: int = 64) -> dict:
    """Sweep every multiplier and increment below a small modulus and compare with the theorem."""
    m = int(modulus)
    rows, agree, full = [], 0, 0
    for a in range(m):
        for c in range(m):
            predicted = hull_dobell(a, c, m)
            measured = period(a, c, m) == m
            agree += int(predicted == measured)
            full += int(measured)
            if predicted != measured:
                rows.append({"multiplier": a, "increment": c,
                             "predicted": predicted, "measured": measured})
    return {"modulus": m, "combinations": m * m, "agreements": agree,
            "disagreements": rows, "full_period_count": full,
            "the_conditions_are_exact": not rows,
            "note": "the theorem is an equivalence, so a single disagreement anywhere in the "
                    "sweep would disprove it",
    }


@functools.lru_cache(maxsize=None)
def the_low_bits_are_worse(bits: int = 6, modulus: int = 2 ** 20) -> dict:
    """Bit ``j`` of a power of two LCG has period at most ``2**(j+1)``, and reaches it.

    This is the reason ``rand() % 6`` was the wrong way to roll a die for thirty years: taking a
    remainder uses the bottom bits, which are the ones with almost no period at all.
    """
    a, c, m = 1103515245, 12345, int(modulus)
    values = lcg(4 * m if m <= 4096 else 200000, a, c, m, seed=1)
    rows = []
    for j in range(int(bits)):
        stream = (values >> j) & 1
        found = None
        for candidate in (2 ** (j + 1), 2 ** (j + 2)):
            if candidate * 4 <= stream.size:
                head = stream[:candidate]
                if np.array_equal(stream[:4 * candidate].reshape(4, candidate), np.tile(head, (4, 1))):
                    found = candidate
                    break
        rows.append({"bit": j, "predicted_period": 2 ** (j + 1), "measured_period": found,
                     "matches": found == 2 ** (j + 1)})
    return {"rows": rows, "modulus": m,
            "every_bit_matches": all(r["matches"] for r in rows),
            "note": "the bottom bit alternates, the next has period 4, and so on, so a remainder "
                    "by a small number inherits a period of that size and not the modulus",
    }


@functools.lru_cache(maxsize=None)
def one_test_is_not_enough(draws: int = 200000) -> dict:
    """RANDU passes a one dimensional uniformity test and fails a three dimensional one."""
    values = lcg_uniform(draws, "randu")
    flat = chi_square_uniformity(values, bins=100)
    triples = np.column_stack([values[:-2], values[1:-1], values[2:]])
    side = 10
    cells = np.minimum((triples * side).astype(int), side - 1)
    index = (cells[:, 0] * side + cells[:, 1]) * side + cells[:, 2]
    counts = np.bincount(index, minlength=side ** 3)
    expected = index.size / side ** 3
    cube = float(np.sum((counts - expected) ** 2) / expected)
    freedom = side ** 3 - 1
    rng = np.random.default_rng(42)
    good = rng.random(draws)
    good_triples = np.column_stack([good[:-2], good[1:-1], good[2:]])
    good_cells = np.minimum((good_triples * side).astype(int), side - 1)
    good_index = (good_cells[:, 0] * side + good_cells[:, 1]) * side + good_cells[:, 2]
    good_counts = np.bincount(good_index, minlength=side ** 3)
    good_cube = float(np.sum((good_counts - expected) ** 2) / expected)
    return {
        "draws": int(draws),
        "flat_statistic": flat["statistic"], "flat_ratio": flat["ratio"],
        "flat_passes": flat["looks_uniform"],
        "cube_statistic": cube, "cube_ratio": cube / freedom,
        "cube_degrees_of_freedom": freedom,
        "pcg64_cube_ratio": good_cube / freedom,
        "randu_fails_in_three_dimensions": cube / freedom > 2.0,
        "pcg64_passes_in_three_dimensions": abs(good_cube / freedom - 1.0) < 0.2,
        "empty_cells": int(np.sum(counts == 0)),
        "note": "a generator can be perfectly uniform in one dimension and lie on planes in "
                "three, which is why the spectral test looks at tuples and not at values",
    }


@functools.lru_cache(maxsize=None)
def the_transforms_are_exact(draws: int = 1000000) -> dict:
    """Inverse transform and Box-Muller against the distributions they claim to produce."""
    rng = np.random.default_rng(42)
    uniforms = rng.random(draws)
    exponential = inverse_transform_exponential(uniforms, rate=1.0)
    exponential_test = kolmogorov_smirnov(exponential, lambda t: 1.0 - math.exp(-t))
    normals = box_muller(rng.random(2 * (draws // 2)))
    normal_test = kolmogorov_smirnov(
        normals, lambda t: 0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))
    return {
        "draws": int(draws),
        "exponential_gap": exponential_test["gap"],
        "exponential_scaled": exponential_test["scaled"],
        "exponential_passes": exponential_test["passes_at_one_percent"],
        "normal_gap": normal_test["gap"],
        "normal_scaled": normal_test["scaled"],
        "normal_passes": normal_test["passes_at_one_percent"],
        "normal_mean": float(np.mean(normals)),
        "normal_variance": float(np.var(normals)),
        "both_pass": (exponential_test["passes_at_one_percent"]
                      and normal_test["passes_at_one_percent"]),
        "note": "both transforms are exact identities, so the only thing a test can find is "
                "sampling error, and the scaled statistic stays near one at a million samples",
    }


@functools.lru_cache(maxsize=None)
def the_polar_method_wastes_a_known_fraction(draws: int = 400000) -> dict:
    """Marsaglia's rejection rate against ``1 - pi/4``, and its output against the normal."""
    out = polar_normal(draws)
    test = kolmogorov_smirnov(out["values"],
                              lambda t: 0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))
    return {
        "draws": int(draws),
        "acceptance": out["acceptance"],
        "predicted_acceptance": out["predicted_acceptance"],
        "ratio": out["acceptance"] / out["predicted_acceptance"],
        "matches_pi_over_four": abs(out["acceptance"] / out["predicted_acceptance"] - 1.0) < 0.01,
        "scaled_gap": test["scaled"],
        "passes": test["passes_at_one_percent"],
        "note": "the acceptance probability is the area of the unit disc over the area of the "
                "square it sits in, which is pi/4, and nothing about the generator changes it",
    }


@functools.lru_cache(maxsize=None)
def the_named_generators_compared(draws: int = 200000) -> dict:
    """Every named LCG and a modern generator, on the same three tests."""
    rows = []
    for name in sorted(FAMOUS):
        values = lcg_uniform(draws, name)
        flat = chi_square_uniformity(values, bins=100)
        rows.append({
            "generator": name,
            "modulus": FAMOUS[name][2],
            "chi_square_ratio": flat["ratio"],
            "passes_uniformity": flat["looks_uniform"],
            "lag_one_correlation": serial_correlation(values, 1),
            "lag_two_correlation": serial_correlation(values, 2),
        })
    rng = np.random.default_rng(42)
    values = rng.random(draws)
    flat = chi_square_uniformity(values, bins=100)
    rows.append({
        "generator": "pcg64",
        "modulus": 2 ** 128,
        "chi_square_ratio": flat["ratio"],
        "passes_uniformity": flat["looks_uniform"],
        "lag_one_correlation": serial_correlation(values, 1),
        "lag_two_correlation": serial_correlation(values, 2),
    })
    lcg_rows = [r for r in rows if r["generator"] != "pcg64"]
    return {
        "rows": rows,
        "every_generator_passes_uniformity": all(r["passes_uniformity"] for r in rows),
        "worst_lcg_correlation": max(abs(r["lag_one_correlation"]) for r in lcg_rows),
        "pcg64_correlation": abs(rows[-1]["lag_one_correlation"]),
        "note": "uniformity and low order correlation are the easy tests and every one of these "
                "passes them, which is why RANDU survived a decade of use",
    }
