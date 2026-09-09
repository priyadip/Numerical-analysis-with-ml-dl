"""Measuring what an algorithm actually costs.

An operation count tells you how the work grows. It does not tell you how long the work
takes. On real hardware, an algorithm that touches memory in a friendly order can beat one
that does fewer arithmetic operations. Lesson 08 makes that concrete instead of asserting
it, and this module is the harness it uses.

Three tools:

- `count_ops` wraps arithmetic in a counter so a flop count can be measured, not derived.
- `time_scaling` times a function across a range of problem sizes.
- `fit_exponent` turns those timings into an observed exponent, so a claim of "order n
  cubed" can be checked against a measured 2.9.

Used by lessons 01, 08, and every lesson that states a complexity.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


# ---------------------------------------------------------------- operation counting


class OpCounter:
    """Counts arithmetic operations performed on values wrapped in `Counted`.

    Use it when you want the real operation count of an implementation rather than the
    count you believe it has. The two disagree more often than people expect.

        counter = OpCounter()
        x = counter.wrap(3.0)
        y = x * x + x          # counted
        counter.total          # 2

    Only the operations between wrapped values are counted, so wrapping the inputs of the
    routine under test is enough.
    """

    def __init__(self) -> None:
        self.adds = 0
        self.muls = 0
        self.divs = 0
        self.others = 0

    @property
    def total(self) -> int:
        return self.adds + self.muls + self.divs + self.others

    def reset(self) -> None:
        self.adds = self.muls = self.divs = self.others = 0

    def wrap(self, value: float) -> "Counted":
        return Counted(float(value), self)

    def summary(self) -> str:
        return (
            f"adds={self.adds}  muls={self.muls}  divs={self.divs}  "
            f"other={self.others}  total={self.total}"
        )


class Counted:
    """A float that reports every arithmetic operation to an `OpCounter`."""

    __slots__ = ("value", "counter")

    def __init__(self, value: float, counter: OpCounter) -> None:
        self.value = float(value)
        self.counter = counter

    def _other(self, o):
        return o.value if isinstance(o, Counted) else float(o)

    def __add__(self, o):
        self.counter.adds += 1
        return Counted(self.value + self._other(o), self.counter)

    __radd__ = __add__

    def __sub__(self, o):
        self.counter.adds += 1
        return Counted(self.value - self._other(o), self.counter)

    def __rsub__(self, o):
        self.counter.adds += 1
        return Counted(self._other(o) - self.value, self.counter)

    def __mul__(self, o):
        self.counter.muls += 1
        return Counted(self.value * self._other(o), self.counter)

    __rmul__ = __mul__

    def __truediv__(self, o):
        self.counter.divs += 1
        return Counted(self.value / self._other(o), self.counter)

    def __rtruediv__(self, o):
        self.counter.divs += 1
        return Counted(self._other(o) / self.value, self.counter)

    def __pow__(self, o):
        self.counter.others += 1
        return Counted(self.value ** self._other(o), self.counter)

    def __neg__(self):
        return Counted(-self.value, self.counter)

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Counted({self.value!r})"


def count_ops(func: Callable, *args) -> tuple[float, OpCounter]:
    """Run `func` with its float arguments wrapped, and report the operation count.

    Returns (result as a plain float, the counter).
    """
    counter = OpCounter()
    wrapped = [counter.wrap(a) if isinstance(a, (int, float)) else a for a in args]
    result = func(*wrapped)
    return float(result), counter


# ---------------------------------------------------------------- timing


def time_once(func: Callable, *args, **kwargs) -> tuple[object, float]:
    """Run a function once and report (result, elapsed seconds)."""
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    return out, time.perf_counter() - t0


def time_best_of(func: Callable, *args, repeats: int = 5, **kwargs) -> float:
    """Best wall time over several runs, in seconds.

    The minimum is used rather than the mean. Timing noise on a shared machine only ever
    makes a run slower, never faster, so the fastest run is the closest to the true cost.
    """
    best = float("inf")
    for _ in range(max(1, repeats)):
        _out, dt = time_once(func, *args, **kwargs)
        best = min(best, dt)
    return best


def time_scaling(
    make_and_run: Callable[[int], object], sizes, repeats: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Time `make_and_run(n)` for each n in sizes. Returns (sizes, best times).

    `make_and_run` should build its own input and do the work, so the setup cost is
    included the same way for every size.
    """
    sizes = np.asarray(list(sizes), dtype=int)
    times = np.array([time_best_of(make_and_run, int(n), repeats=repeats) for n in sizes])
    return sizes, times


def fit_exponent(sizes, times) -> float:
    """Observed exponent p in `time = C n**p`, from a least squares fit on the logs.

    A cubic algorithm should give something near 3. Values noticeably below the theory
    usually mean the sizes were too small for the asymptotic term to dominate, or that
    the routine is memory bound rather than arithmetic bound.
    """
    sizes = np.asarray(sizes, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    mask = np.isfinite(sizes) & np.isfinite(times) & (sizes > 0) & (times > 0)
    if mask.sum() < 2:
        return float("nan")
    p, _ = np.polyfit(np.log(sizes[mask]), np.log(times[mask]), 1)
    return float(p)


def scaling_table(sizes, times, expected_exponent: float | None = None) -> str:
    """A table of size, time, and the ratio to the previous row.

    For an algorithm of order n**p, doubling n should multiply the time by 2**p. The
    ratio column makes that immediately readable.
    """
    sizes = np.asarray(sizes, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    head = "        n      time (s)     ratio to previous"
    if expected_exponent is not None:
        head += f"   (expect {2 ** expected_exponent:.1f} on doubling)"
    lines = [head, "-" * len(head)]
    for i, (n, t) in enumerate(zip(sizes, times)):
        ratio = "" if i == 0 or times[i - 1] == 0 else f"{t / times[i - 1]:12.2f}"
        lines.append(f"{int(n):9d}  {t:12.6f}  {ratio:>12s}")
    return "\n".join(lines)
