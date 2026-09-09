"""Conversion of numbers between positional number systems.

Bases 2 through 16 are supported, which covers binary, octal, decimal and hexadecimal.

Everything here uses `fractions.Fraction` internally so the conversions are exact. That
matters: the whole point of lesson 02 is that 0.1 has no finite binary expansion, and you
cannot demonstrate that honestly with floating point arithmetic.

Used by lesson 02 (number systems) and lesson 03 (floating point).
"""

from __future__ import annotations

from fractions import Fraction

DIGITS = "0123456789ABCDEF"
MAX_BASE = len(DIGITS)


def _check_base(base: int) -> None:
    if not isinstance(base, int) or not 2 <= base <= MAX_BASE:
        raise ValueError(f"base must be an integer in 2..{MAX_BASE}, got {base!r}")


def digit_value(ch: str, base: int) -> int:
    """Numeric value of a single digit character in the given base.

    >>> digit_value("F", 16)
    15
    """
    _check_base(base)
    v = DIGITS.find(ch.upper())
    if v < 0 or v >= base:
        raise ValueError(f"{ch!r} is not a valid digit in base {base}")
    return v


# ---------------------------------------------------------------- integers


def int_to_base(n: int, base: int) -> str:
    """Write a non-negative integer in the given base.

    This is repeated division by the base, collecting remainders from last to first.

    >>> int_to_base(53, 2)
    '110101'
    >>> int_to_base(255, 16)
    'FF'
    """
    _check_base(base)
    if n < 0:
        return "-" + int_to_base(-n, base)
    if n == 0:
        return "0"
    out: list[str] = []
    while n > 0:
        n, r = divmod(n, base)
        out.append(DIGITS[r])
    return "".join(reversed(out))


def int_to_base_steps(n: int, base: int) -> list[tuple[int, int, int]]:
    """The repeated-division table behind `int_to_base`.

    Returns a list of (dividend, quotient, remainder), in the order the division is
    actually carried out. The digits of the answer are the remainders read bottom to top.

    >>> int_to_base_steps(13, 2)
    [(13, 6, 1), (6, 3, 0), (3, 1, 1), (1, 0, 1)]
    """
    _check_base(base)
    if n < 0:
        raise ValueError("int_to_base_steps expects a non-negative integer")
    steps = []
    if n == 0:
        return [(0, 0, 0)]
    while n > 0:
        q, r = divmod(n, base)
        steps.append((n, q, r))
        n = q
    return steps


def int_from_base(s: str, base: int) -> int:
    """Read an integer written in the given base.

    This is Horner's rule applied to the digit string, which is exactly how lesson 01
    evaluates a polynomial: a number in base b *is* a polynomial in b.

    >>> int_from_base("110101", 2)
    53
    >>> int_from_base("FF", 16)
    255
    """
    _check_base(base)
    s = s.strip()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    if not s:
        raise ValueError("empty digit string")
    total = 0
    for ch in s:
        total = total * base + digit_value(ch, base)
    return -total if neg else total


# ---------------------------------------------------------------- fractions


def frac_to_base(x: Fraction | float, base: int, ndigits: int = 20) -> tuple[str, bool]:
    """Write a fractional part 0 <= x < 1 in the given base.

    This is repeated multiplication by the base, collecting the integer part each time.

    Returns (digits, exact) where `exact` says whether the expansion terminated within
    `ndigits`. If it did not, the digit string is truncated, not rounded.

    >>> frac_to_base(Fraction(1, 2), 2, 8)
    ('1', True)
    >>> frac_to_base(Fraction(1, 10), 2, 12)
    ('000110011001', False)
    """
    _check_base(base)
    x = Fraction(x)
    if not 0 <= x < 1:
        raise ValueError(f"frac_to_base expects 0 <= x < 1, got {x}")
    out: list[str] = []
    for _ in range(ndigits):
        if x == 0:
            return "".join(out) if out else "0", True
        x *= base
        d = int(x)
        out.append(DIGITS[d])
        x -= d
    return "".join(out), x == 0


def frac_to_base_steps(
    x: Fraction | float, base: int, ndigits: int = 10
) -> list[tuple[Fraction, Fraction, int]]:
    """The repeated-multiplication table behind `frac_to_base`.

    Returns a list of (value_before, value_after, digit_produced).
    """
    _check_base(base)
    x = Fraction(x)
    if not 0 <= x < 1:
        raise ValueError(f"frac_to_base_steps expects 0 <= x < 1, got {x}")
    steps = []
    for _ in range(ndigits):
        if x == 0:
            break
        before = x
        y = x * base
        d = int(y)
        x = y - d
        steps.append((before, x, d))
    return steps


def frac_from_base(s: str, base: int) -> Fraction:
    """Read a fractional digit string in the given base as an exact Fraction.

    >>> frac_from_base("101", 2) == Fraction(5, 8)
    True
    """
    _check_base(base)
    s = s.strip()
    total = Fraction(0)
    scale = Fraction(1)
    for ch in s:
        scale /= base
        total += digit_value(ch, base) * scale
    return total


# ---------------------------------------------------------------- full numbers


def to_base(x: Fraction | float | int, base: int, ndigits: int = 20) -> str:
    """Write a real number in the given base as `integer.fraction`.

    The fractional part is truncated after `ndigits` digits, and a trailing `...` is added
    when the expansion has not terminated.

    >>> to_base(Fraction(53, 8), 2)
    '110.101'
    >>> to_base(0.1, 2, 12)
    '0.000110011001...'
    """
    _check_base(base)
    x = Fraction(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    ipart = int(x)
    fpart = x - ipart
    head = int_to_base(ipart, base)
    if fpart == 0:
        return sign + head
    digits, exact = frac_to_base(fpart, base, ndigits)
    tail = "" if exact else "..."
    return f"{sign}{head}.{digits}{tail}"


def from_base(s: str, base: int) -> Fraction:
    """Read `integer.fraction` written in the given base, exactly.

    >>> from_base("110.101", 2) == Fraction(53, 8)
    True
    """
    _check_base(base)
    s = s.strip()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    if "." in s:
        head, tail = s.split(".", 1)
    else:
        head, tail = s, ""
    value = Fraction(int_from_base(head, base) if head else 0)
    if tail:
        value += frac_from_base(tail, base)
    return -value if neg else value


def convert(s: str, base_from: int, base_to: int, ndigits: int = 20) -> str:
    """Convert a number string directly from one base to another.

    The route is always through an exact Fraction, never through a float, so no accuracy
    is lost on the way.

    >>> convert("FF", 16, 2)
    '11111111'
    >>> convert("0.1", 10, 2, 12)
    '0.000110011001...'
    """
    return to_base(from_base(s, base_from), base_to, ndigits)


def is_exact_in_base(x: Fraction | float, base: int) -> bool:
    """Does x have a terminating expansion in this base?

    A fraction p/q in lowest terms terminates in base b exactly when every prime factor of
    q also divides b. That is the whole reason 1/10 terminates in base 10 but not in base 2.

    >>> is_exact_in_base(Fraction(1, 10), 10)
    True
    >>> is_exact_in_base(Fraction(1, 10), 2)
    False
    """
    _check_base(base)
    q = Fraction(x).denominator
    while q > 1:
        from math import gcd

        g = gcd(q, base)
        if g == 1:
            return False
        while q % g == 0:
            q //= g
    return True
