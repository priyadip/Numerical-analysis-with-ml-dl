"""IEEE 754 floating point: inspection, simulation and careful summation.

Three groups of tools:

1. **Inspection.** Pull a real double apart into its sign, exponent and mantissa bits, and
   report the spacing of the number line near a given value.
2. **Simulation.** Round a value to a chosen number of mantissa bits so you can watch what
   happens in a precision you can still count on your fingers.
3. **Summation.** Naive, pairwise, Kahan and Neumaier summation, so the cost of ignoring
   rounding error can be measured rather than asserted.

Used by lessons 03, 05 and 93.
"""

from __future__ import annotations

import math
import struct

import numpy as np

# ---------------------------------------------------------------- constants

#: Distance from 1.0 to the next representable double. IEEE double has a 52-bit stored
#: mantissa, so this is 2**-52.
EPS_DOUBLE = float(np.finfo(np.float64).eps)

#: Unit roundoff. With round-to-nearest, any real in the normal range is represented with
#: relative error at most this. It is half of machine epsilon.
UNIT_ROUNDOFF = EPS_DOUBLE / 2

#: Field widths of the three IEEE binary formats, as (sign, exponent, mantissa).
IEEE_FORMATS = {
    "half": (1, 5, 10),
    "single": (1, 8, 23),
    "double": (1, 11, 52),
}


# ---------------------------------------------------------------- inspection


def machine_epsilon(dtype=np.float64) -> float:
    """Find machine epsilon by search rather than by looking it up.

    Halve a number until adding it to one changes nothing, then undo the last halving.
    The answer is the gap between 1 and the next representable number.

    Compare against `numpy.finfo(dtype).eps`.
    """
    one = dtype(1.0)
    eps = dtype(1.0)
    while one + eps / dtype(2.0) != one:
        eps = eps / dtype(2.0)
    return float(eps)


def float_bits(x: float) -> str:
    """The 64 raw bits of a double, as a string of '0' and '1'."""
    (packed,) = struct.unpack(">Q", struct.pack(">d", float(x)))
    return format(packed, "064b")


def decompose(x: float) -> dict:
    """Pull a double apart into its IEEE 754 fields.

    Returns a dict with the raw bit fields, the unbiased exponent, the significand as a
    real number in [1, 2) for normal numbers, and a flag saying which class the value is
    in: 'normal', 'subnormal', 'zero', 'inf' or 'nan'.

    The reconstruction `sign * significand * 2**exponent` is exact for finite values.
    """
    bits = float_bits(x)
    sign_bit = int(bits[0])
    exp_bits = bits[1:12]
    man_bits = bits[12:]
    raw_exp = int(exp_bits, 2)
    raw_man = int(man_bits, 2)

    if raw_exp == 0 and raw_man == 0:
        kind, exponent, significand = "zero", 0, 0.0
    elif raw_exp == 0:
        kind = "subnormal"
        exponent = -1022
        significand = raw_man / 2.0**52
    elif raw_exp == 0x7FF:
        kind = "inf" if raw_man == 0 else "nan"
        exponent, significand = None, None
    else:
        kind = "normal"
        exponent = raw_exp - 1023
        significand = 1.0 + raw_man / 2.0**52

    return {
        "value": float(x),
        "bits": bits,
        "sign_bit": sign_bit,
        "sign": -1 if sign_bit else 1,
        "exponent_bits": exp_bits,
        "mantissa_bits": man_bits,
        "raw_exponent": raw_exp,
        "raw_mantissa": raw_man,
        "exponent": exponent,
        "significand": significand,
        "kind": kind,
    }


def format_bits(x: float) -> str:
    """The bits of a double grouped as `sign | exponent | mantissa` for reading."""
    d = decompose(x)
    return f"{d['sign_bit']} {d['exponent_bits']} {d['mantissa_bits']}"


def ulp(x: float) -> float:
    """Unit in the last place: the gap between x and the next double away from zero.

    This is the absolute spacing of the floating point grid at x. Relative spacing is
    roughly constant, absolute spacing is not, and that difference explains most of the
    surprises in lesson 03.
    """
    x = float(abs(x))
    if math.isnan(x) or math.isinf(x):
        return math.nan
    return math.nextafter(x, math.inf) - x


def _ordered_int(v: float) -> int:
    """Map a double to an integer so that float order becomes integer order.

    Reading the raw bits as an unsigned integer already orders the non-negative doubles
    correctly. Negative doubles come out reversed, so we strip the sign bit and negate.
    Positive and negative zero both map to 0, which is what we want: they are the same
    point on the number line.
    """
    (u,) = struct.unpack(">Q", struct.pack(">d", float(v)))
    if u >> 63:
        return -(u & ((1 << 63) - 1))
    return u


def count_floats_between(a: float, b: float) -> int:
    """How many representable doubles lie between a and b.

    Useful for saying "these two answers differ by 3 floating point numbers" instead of
    quoting a difference of 4.4e-16 that means nothing on its own.

    >>> count_floats_between(1.0, math.nextafter(1.0, math.inf))
    1
    >>> count_floats_between(1.0, 1.0)
    0
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("count_floats_between needs two finite values")
    return abs(_ordered_int(b) - _ordered_int(a))


# ---------------------------------------------------------------- simulation


def round_to_precision(x: float, mantissa_bits: int) -> float:
    """Round x to a floating point system with `mantissa_bits` bits after the binary point.

    This simulates a lower-precision machine while still computing in double, so you can
    watch rounding happen with numbers small enough to check by hand. The rule is round to
    nearest, ties to even, matching the IEEE default.

    `mantissa_bits` counts the bits stored after the leading 1, exactly like IEEE. So
    `mantissa_bits=52` reproduces double precision and leaves x unchanged.
    """
    if mantissa_bits < 0:
        raise ValueError("mantissa_bits must be non-negative")
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return x
    m, e = math.frexp(x)  # x = m * 2**e with 0.5 <= |m| < 1
    scale = 2.0 ** (mantissa_bits + 1)
    scaled = m * scale
    # Round half to even, which is what np.rint / Python round do for .5 cases.
    r = float(np.rint(scaled))
    return math.ldexp(r / scale, e)


def fl(x: float, mantissa_bits: int = 52) -> float:
    """The `fl` operator from the standard model of floating point arithmetic.

    The model says `fl(x) = x (1 + d)` with `|d| <= u`, where u is the unit roundoff of the
    simulated precision. `relative_rounding_error` below checks that claim numerically.
    """
    return round_to_precision(x, mantissa_bits)


def simulated_unit_roundoff(mantissa_bits: int) -> float:
    """Unit roundoff of a simulated system with the given mantissa width."""
    return 2.0 ** (-(mantissa_bits + 1))


def relative_rounding_error(x: float, mantissa_bits: int = 52) -> float:
    """The `d` in `fl(x) = x (1 + d)`, for a given x and simulated precision."""
    x = float(x)
    if x == 0.0:
        return 0.0
    return (fl(x, mantissa_bits) - x) / x


# ---------------------------------------------------------------- summation


def naive_sum(values) -> float:
    """Left to right summation. The obvious method, and the least accurate one."""
    total = 0.0
    for v in values:
        total += float(v)
    return total


def pairwise_sum(values, threshold: int = 8) -> float:
    """Divide and conquer summation.

    Splitting the sum in half recursively reduces the error growth from order n to order
    log n, at no extra arithmetic cost. This is what `numpy.sum` does internally, which is
    why NumPy is more accurate than a Python loop on the same data.
    """
    a = np.asarray(values, dtype=float).ravel()
    n = a.size
    if n == 0:
        return 0.0
    if n <= threshold:
        return naive_sum(a)
    mid = n // 2
    return pairwise_sum(a[:mid], threshold) + pairwise_sum(a[mid:], threshold)


def kahan_sum(values) -> float:
    """Kahan compensated summation.

    Each step estimates the part of the addend that was lost to rounding and carries it
    into the next step. The error bound becomes independent of n, at the price of four
    operations per element instead of one.
    """
    total = 0.0
    compensation = 0.0
    for v in values:
        y = float(v) - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total


def neumaier_sum(values) -> float:
    """Neumaier's improvement on Kahan.

    Kahan loses the correction when the running total is smaller in magnitude than the
    incoming value. Neumaier handles both orderings and adds the accumulated correction at
    the end. This is the algorithm behind `math.fsum`-quality results at lower cost.
    """
    total = 0.0
    compensation = 0.0
    for v in values:
        v = float(v)
        t = total + v
        if abs(total) >= abs(v):
            compensation += (total - t) + v
        else:
            compensation += (v - t) + total
        total = t
    return total + compensation


def summation_methods() -> dict:
    """The summation routines, keyed by display name, for side-by-side comparison."""
    return {
        "naive": naive_sum,
        "pairwise": pairwise_sum,
        "Kahan": kahan_sum,
        "Neumaier": neumaier_sum,
        "numpy.sum": lambda v: float(np.sum(np.asarray(v, dtype=float))),
        "math.fsum": lambda v: math.fsum(float(x) for x in v),
    }
