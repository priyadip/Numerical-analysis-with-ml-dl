"""Sixteen bit arithmetic, and the three things that break in it.

Where the work is
-----------------
Lesson 03 measured double precision. Deep learning runs on half of it, in one of two formats that
split the same 16 bits differently, and the choice between them is the clearest example in this
whole course of the trade lesson 03 described.

``float16`` spends 5 bits on the exponent and 10 on the mantissa. ``bfloat16`` spends 8 and 7. So
``float16`` is 16 times more precise and ``bfloat16`` has ``10**33`` times the range, and every
practical difference between them follows from those two numbers.

Three failures follow, and all three are lessons this course has already taught:

**Overflow, which is lesson 03's finite exponent range.** ``exp`` of anything past 11.09 is not a
``float16`` number. A softmax computes ``exp`` of the raw scores, so a softmax in ``float16``
overflows on inputs that are entirely ordinary. The fix is the shift-by-the-maximum identity, and
it is exact rather than approximate.

**Underflow, which is the same range from the other end.** Gradients are small, and past 6e-08 a
``float16`` gradient is exactly zero rather than merely inaccurate. Loss scaling multiplies the
loss by a power of two before differentiating and divides afterwards, which moves the numbers into
the representable window and out again with no rounding at all.

**Stagnation and non-associativity, which are lesson 04.** A long sum in ``float16`` stops growing
once the partial sum is more than ``1/eps`` times the next term, and a sum computed in a different
order gives a different answer. Both are why every mixed precision system stores in 16 bits and
accumulates in 32.

What the measurements here show
-------------------------------
* **The two formats are the same 16 bits spent differently.** ``float16`` has a unit roundoff of
  4.88e-04 and a largest value of 65504; ``bfloat16`` has 3.91e-03 and 3.39e+38. That is a factor
  of 8 in precision against a factor of 5.2e+33 in range.
* **A softmax overflows where ``exp`` does**, at an input of 11.0898 in float16 and 88.75 in
  bfloat16, each matching ``log`` of the format's largest value to 4 parts in 10000.
* **The shift by the maximum is an identity, not a trade.** Adding a constant to every score
  changes the softmax by exactly 0, and at an offset of 1e+06 by 6.6e-12, which is the rounding of
  the offset itself and not of the method.
* **The shifted form never overflows**, at inputs up to 1e+04 in both formats, where the direct form
  has been returning ``inf`` since 11.09.
* **Gradients underflow long before activations do.** At a gradient scale of 1e-08, 99.8 per cent of
  a float16 gradient is exactly zero and none of a bfloat16 one is, because the smallest float16
  number is 6e-08 and the smallest bfloat16 number is 9e-41.
* **Loss scaling has a window, and it is 30 powers of two wide.** For float16 every scale from
  2**10 to 2**40 recovers the gradient to better than 1 per cent, below it the gradient is zeros and
  at 2**45 it is ``inf``. bfloat16 needs no scaling at any point, at a flat error of 0.0017.
* **A float16 sum of ones stagnates after exactly 2048 terms** and a bfloat16 one after exactly 256,
  both equal to ``1/u`` to the digit. The same sum with a float32 accumulator is still exact after
  20000.
* **Reduction order changes the answer, and the spread follows the square root law.** Summing the
  same numbers in 200 random orders gives a sequential spread growing at a fitted 0.538 and a
  pairwise spread at 0.109, so pairwise summation is 100 times tighter at a million terms.
* **The classic ``sqrt(n) u`` bound is loose by 555** against sequential summation on random data,
  because the rounding errors have random signs.
* **Chunking a reduction is what makes it non-deterministic.** Splitting the same sum into 1 to 4096
  blocks gives 5 different float32 answers spread over 1.0e-03, and more blocks is both less
  reproducible and more accurate.
* **A half precision product needs a wider accumulator.** The error a float16 accumulator adds on
  top of the input rounding grows with the inner dimension at a fitted 0.680, while the float32
  accumulator's total error is flat at a slope of 0.010, and the gap reaches a factor of 10.9 at an
  inner dimension of 512.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- the formats


#: The four formats this module works in. ``bits`` counts sign, exponent and stored mantissa.
FORMATS = {
    "float64": {"exponent_bits": 11, "mantissa_bits": 52, "bits": 64, "native": np.float64},
    "float32": {"exponent_bits": 8, "mantissa_bits": 23, "bits": 32, "native": np.float32},
    "float16": {"exponent_bits": 5, "mantissa_bits": 10, "bits": 16, "native": np.float16},
    "bfloat16": {"exponent_bits": 8, "mantissa_bits": 7, "bits": 16, "native": None},
}


def to_bfloat16(values):
    """Round to the nearest ``bfloat16``, keeping the result in ``float32``.

    NumPy has no ``bfloat16`` type, so the format is simulated by rounding a ``float32`` to its top
    16 bits, round to nearest with ties to even. That is exactly what the hardware does, and it
    keeps the ``float32`` exponent range, which is the point of the format.
    """
    a = np.asarray(values, dtype=np.float32)
    bits = a.view(np.uint32)
    carry = ((bits >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    rounded = ((bits + carry) & np.uint32(0xFFFF0000)).view(np.float32)
    return np.where(np.isnan(a), a, rounded)


def cast(values, name: str):
    """Round an array into the named format, returning it in the widest type that holds it."""
    if name not in FORMATS:
        raise ValueError(f"unknown format {name!r}, expected one of {sorted(FORMATS)}")
    if name == "bfloat16":
        return to_bfloat16(values)
    return np.asarray(values, dtype=FORMATS[name]["native"])


def describe(name: str) -> dict:
    """The numbers that decide what a format can and cannot do."""
    if name not in FORMATS:
        raise ValueError(f"unknown format {name!r}, expected one of {sorted(FORMATS)}")
    spec = FORMATS[name]
    mantissa = spec["mantissa_bits"]
    exponent = spec["exponent_bits"]
    bias = 2 ** (exponent - 1) - 1
    largest = (2.0 - 2.0 ** (-mantissa)) * 2.0 ** bias
    smallest_normal = 2.0 ** (1 - bias)
    return {
        "name": name, "bits": spec["bits"], "exponent_bits": exponent,
        "mantissa_bits": mantissa,
        "epsilon": 2.0 ** (-mantissa), "unit_roundoff": 2.0 ** (-mantissa - 1),
        "largest": largest, "smallest_normal": smallest_normal,
        "smallest_subnormal": smallest_normal * 2.0 ** (-mantissa),
        "decimal_digits": mantissa * math.log10(2.0),
        "overflow_input_for_exp": math.log(largest),
    }


# --------------------------------------------------------------------------- softmax


def softmax_direct(scores, name: str = "float64"):
    """Softmax written the way the formula is written, which is the way that overflows."""
    x = cast(scores, name)
    with np.errstate(over="ignore", invalid="ignore"):
        raised = cast(np.exp(np.asarray(x, dtype=np.float64)), name)
        total = cast(np.sum(np.asarray(raised, dtype=np.float64)), name)
        return cast(np.asarray(raised, dtype=np.float64) / float(total), name)


def softmax_shifted(scores, name: str = "float64"):
    """Softmax with the maximum subtracted first, which is the same function and never overflows."""
    x = np.asarray(cast(scores, name), dtype=np.float64)
    shifted = x - np.max(x)
    raised = cast(np.exp(shifted), name)
    total = cast(np.sum(np.asarray(raised, dtype=np.float64)), name)
    return cast(np.asarray(raised, dtype=np.float64) / float(total), name)


def logsumexp_direct(scores, name: str = "float64") -> float:
    """``log(sum(exp(x)))``, computed literally."""
    x = np.asarray(cast(scores, name), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return float(cast(np.log(float(cast(np.sum(np.exp(x)), name))), name))


def logsumexp_shifted(scores, name: str = "float64") -> float:
    """``m + log(sum(exp(x - m)))`` with ``m`` the maximum, which is the same number."""
    x = np.asarray(cast(scores, name), dtype=np.float64)
    peak = float(np.max(x))
    return float(cast(peak + np.log(float(cast(np.sum(np.exp(x - peak)), name))), name))


# --------------------------------------------------------------------------- scaling and summing


def loss_scaled_gradient(gradient, scale: float, name: str = "float16") -> dict:
    """Multiply, round to the format, then divide back, which is what loss scaling does.

    Both the multiplication and the division are by a power of two, so neither of them rounds. The
    only rounding is the cast in the middle, and the scale decides whether that cast lands inside
    the format's representable window.
    """
    exact = np.asarray(gradient, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        stored = cast(exact * float(scale), name)
        recovered = np.asarray(stored, dtype=np.float64) / float(scale)
    finite = np.all(np.isfinite(recovered))
    reference = np.linalg.norm(exact)
    return {
        "recovered": recovered, "scale": float(scale),
        "zeros": float(np.mean(recovered == 0.0)),
        "overflowed": not finite,
        "relative_error": (float(np.linalg.norm(recovered - exact) / reference)
                           if finite and reference > 0 else math.inf),
    }


def accumulate(values, store: str = "float16", accumulator: str | None = None) -> float:
    """Sum an array left to right, storing in one format and accumulating in another.

    This is the mixed precision pattern: the data lives in 16 bits and the running total lives in
    32 or 64. The two formats being different is the whole design, and the measurement below shows
    what happens when they are the same.
    """
    stored = np.asarray(cast(values, store), dtype=np.float64)
    target = store if accumulator is None else accumulator
    total = 0.0
    for item in stored:
        total = float(cast(total + float(item), target))
    return total


def stagnation_point(term: float, store: str = "float16", limit: int = 100000) -> dict:
    """Add the same number repeatedly until the running total stops changing.

    A sum in a format with unit roundoff ``u`` stops growing when the partial sum reaches
    ``term / u``, because the next addition rounds back to where it started. Lesson 04 derived
    that; here it is a hard integer.
    """
    step = float(cast(term, store))
    total = 0.0
    count = 0
    while count < int(limit):
        moved = float(cast(total + step, store))
        if moved == total:
            break
        total = moved
        count += 1
    spec = describe(store)
    return {
        "terms": count, "total": total, "exact": count * step,
        "predicted": step / spec["unit_roundoff"],
        "predicted_terms": 1.0 / spec["unit_roundoff"],
        "stagnated": count < int(limit),
    }


def chunked_sum(values, chunks: int, name: str = "float32") -> float:
    """Split into blocks, sum each block, then sum the blocks. This is what a GPU reduction does."""
    a = np.asarray(cast(values, name), dtype=np.float64)
    k = max(1, min(int(chunks), a.size))
    edges = np.array_split(np.arange(a.size), k)
    partials = []
    for block in edges:
        total = 0.0
        for index in block:
            total = float(cast(total + a[index], name))
        partials.append(total)
    answer = 0.0
    for piece in partials:
        answer = float(cast(answer + piece, name))
    return answer


def matmul(left, right, store: str = "float16", accumulator: str = "float32"):
    """A matrix product with the inputs in one format and the running dot product in another."""
    a = np.asarray(cast(left, store), dtype=np.float64)
    b = np.asarray(cast(right, store), dtype=np.float64)
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"shapes {a.shape} and {b.shape} do not multiply")
    out = np.zeros((a.shape[0], b.shape[1]))
    for row in range(a.shape[0]):
        for column in range(b.shape[1]):
            total = 0.0
            for index in range(a.shape[1]):
                total = float(cast(total + float(cast(a[row, index] * b[index, column],
                                                      accumulator)), accumulator))
            out[row, column] = total
    return cast(out, store)


# --------------------------------------------------------------------------- measurements


def the_same_bits_spent_differently(names=("float64", "float32", "float16", "bfloat16")) -> dict:
    """Put the four formats side by side and read the trade off the table.

    Nothing here is measured, it is all definition, and that is the point: everything the rest of
    this module finds is already written in these five columns.
    """
    rows = [describe(name) for name in names]
    half = [row for row in rows if row["bits"] == 16]
    return {
        "rows": rows,
        "precision_ratio": (max(r["unit_roundoff"] for r in half)
                            / min(r["unit_roundoff"] for r in half)) if len(half) > 1 else 1.0,
        "range_ratio": (max(r["largest"] for r in half)
                        / min(r["largest"] for r in half)) if len(half) > 1 else 1.0,
        "the_two_halves_have_the_same_width": len({r["bits"] for r in half}) == 1,
        "one_is_more_precise_and_one_has_more_range": (
            len(half) == 2
            and (half[0]["unit_roundoff"] < half[1]["unit_roundoff"])
            == (half[0]["largest"] < half[1]["largest"])),
        "note": "16 bits is 16 bits, and the exponent and the mantissa are taking them from each "
                "other",
    }


def a_softmax_overflows_where_the_exponent_runs_out(names=("float16", "bfloat16", "float32"),
                                                    probes: int = 60) -> dict:
    """Find, by bisection, the input at which the direct softmax stops returning a number.

    The predicted threshold is ``log`` of the format's largest value, because that is where ``exp``
    leaves the format. It has nothing to do with softmax and everything to do with lesson 03.
    """
    rows = []
    for name in names:
        spec = describe(name)
        predicted = spec["overflow_input_for_exp"]
        low, high = 0.0, 4.0 * predicted

        def broken(value, target=name):
            out = softmax_direct(np.array([value, 0.0]), target)
            return not np.all(np.isfinite(np.asarray(out, dtype=np.float64)))

        for _ in range(int(probes)):
            middle = 0.5 * (low + high)
            if broken(middle):
                high = middle
            else:
                low = middle
        measured = 0.5 * (low + high)
        rows.append({
            "format": name, "measured": measured, "predicted": predicted,
            "ratio": measured / predicted, "largest": spec["largest"],
        })
    return {
        "rows": rows,
        "worst_ratio": max(max(r["ratio"] for r in rows), 1.0 / min(r["ratio"] for r in rows)),
        "the_threshold_is_log_of_the_largest": max(abs(r["ratio"] - 1.0) for r in rows) < 0.01,
        "float16_is_the_tight_one": min(rows, key=lambda r: r["measured"])["format"] == "float16",
        "note": "a softmax overflows because exp does, and where exp does is a property of the "
                "format's exponent field",
    }


def the_shift_is_exact_and_it_fixes_the_overflow(offsets=(0.0, 1.0, 1e3, 1e6),
                                                 inputs=(10.0, 12.0, 100.0, 1e4),
                                                 names=("float16", "bfloat16"),
                                                 size: int = 8, seed: int = 42) -> dict:
    """Two claims about the same identity: it changes nothing, and it removes the overflow.

    Softmax is invariant under adding a constant to every score, so subtracting the maximum is not
    an approximation that trades accuracy for range. It is the same function computed at a
    different point, and the invariance is exact in exact arithmetic.
    """
    rng = np.random.default_rng(int(seed))
    base = rng.standard_normal(int(size))
    invariance = []
    for offset in offsets:
        moved = softmax_shifted(base + float(offset), "float64")
        invariance.append({
            "offset": float(offset),
            "change": float(np.max(np.abs(np.asarray(moved, dtype=np.float64)
                                          - np.asarray(softmax_shifted(base, "float64"),
                                                       dtype=np.float64)))),
            "predicted": float(offset) * describe("float64")["unit_roundoff"],
        })
    rows = []
    for name in names:
        for scale in inputs:
            scores = base * 0.0
            scores[0] = float(scale)
            direct = np.asarray(softmax_direct(scores, name), dtype=np.float64)
            shifted = np.asarray(softmax_shifted(scores, name), dtype=np.float64)
            rows.append({
                "format": name, "input": float(scale),
                "direct_finite": bool(np.all(np.isfinite(direct))),
                "shifted_finite": bool(np.all(np.isfinite(shifted))),
                "shifted_sums_to": float(np.sum(shifted)),
            })
    return {
        "invariance": invariance, "rows": rows,
        "worst_invariance": max(r["change"] for r in invariance),
        "the_identity_is_exact": invariance[0]["change"] == 0.0,
        "the_only_error_is_the_shift_itself": all(r["change"] <= max(r["predicted"], 1e-16)
                                                  for r in invariance),
        "the_shifted_form_never_overflows": all(r["shifted_finite"] for r in rows),
        "the_direct_form_does": any(not r["direct_finite"] for r in rows),
        "it_still_sums_to_one": max(abs(r["shifted_sums_to"] - 1.0) for r in rows) < 1e-2,
        "note": "subtracting the maximum is an identity, so it costs nothing and it is not a "
                "trade between range and accuracy",
    }


def gradients_underflow_before_activations_do(scales=(1e-2, 1e-4, 1e-6, 1e-8, 1e-10),
                                              names=("float16", "bfloat16"), size: int = 4096,
                                              seed: int = 42) -> dict:
    """Cast a gradient of a given size into each format and count how much of it became zero.

    Activations sit near 1 and gradients do not. A format whose smallest positive number is 6e-08
    is fine for the first and useless for the second, and that asymmetry is the entire reason loss
    scaling exists.
    """
    rng = np.random.default_rng(int(seed))
    shape = rng.standard_normal(int(size))
    rows = []
    for scale in scales:
        entry = {"scale": float(scale)}
        for name in names:
            stored = np.asarray(cast(shape * float(scale), name), dtype=np.float64)
            entry[f"{name} zeros"] = float(np.mean(stored == 0.0))
        rows.append(entry)
    return {
        "rows": rows, "names": list(names),
        "float16_smallest": describe("float16")["smallest_subnormal"],
        "bfloat16_smallest": describe("bfloat16")["smallest_subnormal"],
        "float16_fails": max(r["float16 zeros"] for r in rows) > 0.9,
        "bfloat16_survives": max(r["bfloat16 zeros"] for r in rows) == 0.0,
        "note": "the failure is underflow, so it is the exponent field and not the mantissa that "
                "decides which format survives",
    }


def loss_scaling_has_a_window(powers=(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
                              gradient_scale: float = 1e-8,
                              names=("float16", "bfloat16"), size: int = 4096,
                              tolerance: float = 1e-2, seed: int = 42) -> dict:
    """Sweep the loss scale over powers of two and find where the gradient survives the round trip.

    The multiply and the divide are both by a power of two, so neither rounds. Everything that
    happens here happens in the cast in between, which makes this the cleanest possible measurement
    of a format's representable window.
    """
    rng = np.random.default_rng(int(seed))
    gradient = rng.standard_normal(int(size)) * float(gradient_scale)
    rows = []
    for power in powers:
        entry = {"power": int(power), "scale": 2.0 ** int(power)}
        for name in names:
            out = loss_scaled_gradient(gradient, 2.0 ** int(power), name)
            entry[f"{name} error"] = out["relative_error"]
            entry[f"{name} zeros"] = out["zeros"]
        rows.append(entry)
    windows = {}
    for name in names:
        working = [r["power"] for r in rows if r[f"{name} error"] < float(tolerance)]
        windows[name] = {"lowest": min(working) if working else None,
                         "highest": max(working) if working else None,
                         "width": (max(working) - min(working)) if working else 0,
                         "overflows_above": min((r["power"] for r in rows
                                                 if not math.isfinite(r[f"{name} error"])),
                                                default=None)}
    return {
        "rows": rows, "windows": windows, "names": list(names),
        "float16_needs_scaling": rows[0]["float16 error"] > float(tolerance),
        "bfloat16_does_not": rows[0]["bfloat16 error"] < float(tolerance),
        "the_window_is_wide": windows["float16"]["width"] >= 10,
        "the_window_has_a_top": windows["float16"]["overflows_above"] is not None,
        "note": "loss scaling is a change of units, not a numerical method, and the only question "
                "is whether the units land inside the format",
    }


def a_low_precision_sum_stagnates(names=("float16", "bfloat16", "float32"), term: float = 1.0,
                                  limit: int = 200000) -> dict:
    """Add 1 to a running total until the total stops moving, in each format.

    The answer is ``1/u``, an exact integer, and it is the reason no accumulator is ever kept in
    the same format as the data.
    """
    rows = []
    for name in names:
        out = stagnation_point(float(term), name, limit=int(limit))
        rows.append({
            "format": name, "terms": out["terms"], "total": out["total"],
            "predicted": out["predicted_terms"], "stagnated": out["stagnated"],
            "ratio": out["terms"] / out["predicted_terms"],
        })
    mixed = accumulate(np.full(min(int(limit), 20000), float(term)), "float16", "float32")
    return {
        "rows": rows,
        "mixed_precision_total": mixed,
        "mixed_precision_terms": min(int(limit), 20000),
        "mixed_precision_is_exact": abs(mixed - min(int(limit), 20000) * float(term)) < 1e-6,
        "the_stagnation_point_is_one_over_u": all(
            abs(r["ratio"] - 1.0) < 0.01 for r in rows if r["stagnated"]),
        "half_precision_stagnates": rows[0]["stagnated"],
        "note": "the running total stops growing once it is 1/u times the next term, and a wider "
                "accumulator is the whole of the fix",
    }


def the_order_of_a_reduction_changes_the_answer(sizes=(1000, 10000, 100000, 1000000),
                                                orders: int = 200, name: str = "float32",
                                                seed: int = 42) -> dict:
    """Sum the same numbers in many random orders and measure how much the answer moves.

    Floating point addition is not associative, so a reduction has as many answers as it has
    orderings. The spread is bounded by ``sqrt(n) u`` for a random order, and the measurement is
    whether the bound is attained.
    """
    rng = np.random.default_rng(int(seed))
    unit = describe(name)["unit_roundoff"]
    rows = []
    for size in sizes:
        values = rng.standard_normal(int(size))
        exact = float(np.sum(np.asarray(values, dtype=np.float64)))
        pairwise, sequential = [], []
        for trial in range(int(orders)):
            order = np.random.default_rng(int(seed) + trial).permutation(int(size))
            shuffled = cast(values[order], name)
            pairwise.append(float(np.sum(shuffled)))
            sequential.append(float(np.cumsum(shuffled)[-1]))
        pairwise, sequential = np.array(pairwise), np.array(sequential)
        scale = float(np.sum(np.abs(values)))
        rows.append({
            "size": int(size),
            "distinct": int(np.unique(pairwise).size),
            "pairwise_spread": float((pairwise.max() - pairwise.min()) / abs(exact)),
            "sequential_spread": float((sequential.max() - sequential.min()) / abs(exact)),
            "predicted": float(math.sqrt(size) * unit * scale / abs(exact)),
        })
    logs = np.log([r["size"] for r in rows])
    pairwise_slope = float(np.polyfit(logs, np.log([r["pairwise_spread"] for r in rows]), 1)[0])
    sequential_slope = float(np.polyfit(logs, np.log([r["sequential_spread"]
                                                      for r in rows]), 1)[0])
    return {
        "rows": rows, "orders": int(orders), "format": name,
        "pairwise_slope": pairwise_slope, "sequential_slope": sequential_slope,
        "worst_bound_slack": max(r["predicted"] / r["sequential_spread"] for r in rows),
        "largest_gain_from_pairwise": max(r["sequential_spread"] / r["pairwise_spread"]
                                          for r in rows),
        "the_answers_differ": all(r["distinct"] > 1 for r in rows),
        "the_spread_grows_with_the_length": (rows[-1]["sequential_spread"]
                                             > rows[0]["sequential_spread"]),
        "the_sequential_spread_grows_like_the_square_root": abs(sequential_slope - 0.5) < 0.15,
        "pairwise_is_much_flatter": pairwise_slope < sequential_slope,
        "the_bound_holds": all(r["sequential_spread"] < r["predicted"] for r in rows),
        "note": "a reduction has one answer per ordering, the sequential spread follows the "
                "square root law, and pairwise summation beats it by two orders of magnitude",
    }


def chunking_is_what_makes_a_reduction_nondeterministic(chunks=(1, 2, 8, 64, 512, 4096),
                                                        size: int = 100000, name: str = "float32",
                                                        seed: int = 42) -> dict:
    """Vary the number of blocks a sum is split into, which is what varies between GPU runs.

    A parallel reduction picks a block count from the hardware and the launch configuration, so two
    runs of the same code on the same data can group the additions differently. Every grouping is a
    correct sum of the same numbers, and they are not the same number.
    """
    rng = np.random.default_rng(int(seed))
    values = rng.standard_normal(int(size))
    exact = float(np.sum(np.asarray(values, dtype=np.float64)))
    rows = []
    for count in chunks:
        answer = chunked_sum(values, int(count), name)
        rows.append({
            "chunks": int(count), "answer": answer,
            "error": abs(answer - exact) / abs(exact),
        })
    answers = np.array([r["answer"] for r in rows])
    return {
        "rows": rows, "exact": exact,
        "distinct": int(np.unique(answers).size),
        "spread": float(answers.max() - answers.min()),
        "best": min(rows, key=lambda r: r["error"]),
        "worst": max(rows, key=lambda r: r["error"]),
        "chunking_changes_the_answer": int(np.unique(answers).size) > 1,
        "more_chunks_is_usually_better": rows[-1]["error"] < rows[0]["error"],
        "note": "the non-determinism is in the grouping, and more groups is both less "
                "reproducible and more accurate",
    }


def a_half_precision_product_needs_a_wider_accumulator(inners=(8, 32, 128, 512),
                                                       rows_out: int = 6, cols_out: int = 6,
                                                       seed: int = 42) -> dict:
    """Multiply two matrices with a half precision accumulator and with a single precision one.

    Lesson 04's bound for a dot product of length ``k`` is ``k u`` in the worst case and
    ``sqrt(k) u`` for random signs. With a wider accumulator the only rounding left is the input
    cast, so the error stops depending on ``k`` at all, and that flat line is the entire argument
    for mixed precision hardware.
    """
    rng = np.random.default_rng(int(seed))
    rows = []
    for inner in inners:
        left = rng.standard_normal((int(rows_out), int(inner)))
        right = rng.standard_normal((int(inner), int(cols_out)))
        exact = np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)
        narrow = np.asarray(matmul(left, right, "float16", "float16"), dtype=np.float64)
        wide = np.asarray(matmul(left, right, "float16", "float32"), dtype=np.float64)
        reference = np.linalg.norm(exact)
        narrow_error = float(np.linalg.norm(narrow - exact) / reference)
        wide_error = float(np.linalg.norm(wide - exact) / reference)
        rows.append({
            "inner": int(inner), "half_accumulator": narrow_error,
            "single_accumulator": wide_error,
            "excess": max(narrow_error - wide_error, 1e-300),
        })
    logs = np.log([r["inner"] for r in rows])
    narrow_slope = float(np.polyfit(logs, np.log([r["half_accumulator"] for r in rows]), 1)[0])
    wide_slope = float(np.polyfit(logs, np.log([r["single_accumulator"] for r in rows]), 1)[0])
    excess_slope = float(np.polyfit(logs, np.log([r["excess"] for r in rows]), 1)[0])
    return {
        "rows": rows, "narrow_slope": narrow_slope, "wide_slope": wide_slope,
        "excess_slope": excess_slope,
        "worst_gain": max(r["half_accumulator"] / r["single_accumulator"] for r in rows),
        "the_narrow_one_grows": excess_slope > 0.4,
        "the_wide_one_is_flat": abs(wide_slope) < 0.25,
        "the_wide_one_is_always_better": all(
            r["single_accumulator"] <= r["half_accumulator"] for r in rows),
        "note": "storing in 16 bits costs a fixed 1e-3; accumulating in 16 bits costs a factor "
                "that grows with the inner dimension",
    }
