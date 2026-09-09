"""Measuring how fast a sequence converges.

A method is said to converge with order p and rate C when

    |e_{k+1}|  ~  C |e_k|^p     as k grows,

where e_k is the error at step k. Order 1 is linear convergence, order 2 is quadratic.
Claiming an order is easy. This module measures it, so every convergence claim in the
course can be checked against a number rather than taken on faith.

Two situations come up and they need different tools:

- **Errors known.** You have the exact answer, so you have the true errors. Use
  `observed_order`.
- **Errors unknown.** You only have the iterates. Use `observed_order_from_iterates`,
  which uses successive differences as a proxy for the error.

Used by lessons 07, 09, 10, 11, 12 and every method that claims a convergence rate.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- order estimation


def _usable_prefix(errors) -> np.ndarray:
    """The leading run of errors that are finite and strictly positive.

    Once an error reaches exactly zero, or overflows, every later value is roundoff noise
    rather than information about the rate. Truncating there is more honest than filtering
    the bad entries out and computing a ratio that straddles the gap.
    """
    e = np.abs(np.asarray(errors, dtype=float).ravel())
    good = np.isfinite(e) & (e > 0)
    if not good.any():
        return np.array([])
    bad = np.nonzero(~good)[0]
    stop = int(bad[0]) if bad.size else e.size
    return e[:stop]


def observed_order(errors) -> np.ndarray:
    """Estimate the order of convergence from a sequence of known errors.

    From `|e_{k+1}| = C |e_k|^p`, taking logs of two consecutive steps and dividing
    removes C:

        p_k  =  log(|e_{k+1}| / |e_k|)  /  log(|e_k| / |e_{k-1}|)

    Returns one estimate per usable triple, so an input of n errors gives n-2 estimates.
    The last few are the ones to trust: early steps are not yet in the asymptotic regime,
    and once the errors reach roundoff level the estimates become noise.

    The sequence is truncated at the first zero or non-finite value, because everything
    after that point is roundoff noise rather than information about the rate.

    One case the truncation cannot catch: a sequence that never reaches exactly zero but
    stalls on a plateau, oscillating by one ulp. That happens when the "errors" are actually
    successive differences of iterates. On a flat plateau consecutive ratios equal 1, so the
    formula computes log(1)/log(1) and returns NaN. NaN is the correct answer, since a flat
    sequence says nothing about a rate, but it means you should read the estimates from the
    middle of the returned array rather than blindly taking the last one.
    """
    e = _usable_prefix(errors)
    if e.size < 3:
        return np.array([])
    logs = np.log(e)
    num = np.diff(logs[1:])
    den = np.diff(logs[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        p = num / den
    return p


def reliable_order(errors, rel_floor: float = 1e-14) -> float:
    """The last order estimate that is not contaminated by the roundoff floor.

    `observed_order` returns one estimate per triple of errors. The final triples usually
    straddle the point where the error stops being the true error and becomes rounding
    noise, so the last entry of that array is often meaningless. Lesson 07 calls this "late
    steps lie".

    This returns the last estimate computed **entirely** from errors above a floor, taken as
    `rel_floor` times the largest error in the sequence. For a method converging to a root of
    size 1 in double precision, the default cuts off about two orders of magnitude above
    unit roundoff, which is enough margin.

    Example of why it matters. A cubic method on a smooth problem might produce errors

        1.3e0, 3.1e-1, 8.0e-3, 1.7e-7, 1.1e-16

    where the last value is pure roundoff. `observed_order` reports 2.55, 2.94, 1.97 and the
    last of those is nonsense. `reliable_order` reports 2.94, which is the right answer.

    Returns NaN when there are not enough usable points.
    """
    e = _usable_prefix(errors)
    if e.size < 3:
        return float("nan")
    floor = e.max() * rel_floor
    keep = e[e > floor]
    if keep.size < 3:
        return float("nan")
    orders = observed_order(keep)
    if orders.size == 0:
        return float("nan")
    finite = orders[np.isfinite(orders)]
    return float(finite[-1]) if finite.size else float("nan")


def observed_order_from_iterates(iterates) -> np.ndarray:
    """Estimate the order of convergence when the exact answer is unknown.

    Uses successive differences `d_k = x_{k+1} - x_k` in place of the errors. For a
    convergent sequence these shrink at the same rate as the true errors, so the same
    formula applies.
    """
    x = np.asarray(iterates, dtype=float).ravel()
    return observed_order(np.diff(x))


def asymptotic_constant(errors, p: float) -> float:
    """Estimate the rate constant C in `|e_{k+1}| = C |e_k|^p`.

    Uses the last usable pair, which is the closest to the asymptotic regime.
    """
    e = _usable_prefix(errors)
    if e.size < 2:
        return float("nan")
    return float(e[-1] / e[-2] ** p)


def linear_rate(errors, floor_factor: float = 100.0) -> float:
    """Estimate the rate constant C for a linearly convergent sequence, `e_{k+1} ~ C e_k`.

    The estimate comes from fitting a straight line to `log(e_k)` against `k`, so the
    answer is `exp(slope)`. Fitting is used rather than the obvious ratio `e_{k+1}/e_k`
    because for some methods the error is **not monotone**, even though it decreases
    geometrically on average.

    Bisection is the standard example. Its bracket halves exactly every step, but the
    distance from the midpoint to the root jumps around: a midpoint can land almost on the
    root by luck and then the next one is much worse. Consecutive ratios for bisection
    range from 0.02 to 25 in a single run, so any single ratio is worthless while the fitted
    decay is 0.5 to two decimal places.

    The stalled tail is dropped before fitting. Once the errors stop falling they have hit
    the roundoff floor and say nothing about C. `floor_factor` sets how far above the
    smallest observed error the cut is made.

    For fixed point iteration the answer should approach `|g'(r)|`.
    """
    e = _usable_prefix(errors)
    if e.size < 3:
        return float("nan")
    above = np.nonzero(e > e.min() * floor_factor)[0]
    e_fit = e[: above[-1] + 1] if above.size >= 3 else e
    k = np.arange(e_fit.size, dtype=float)
    slope = np.polyfit(k, np.log(e_fit), 1)[0]
    return float(np.exp(slope))


# ---------------------------------------------------------------- reporting


def convergence_table(errors, exact_order: float | None = None) -> str:
    """A readable table of error, ratio and estimated order, one row per step.

    `exact_order`, when given, is printed in the header so the measured column can be
    compared against the theory at a glance.
    """
    e = np.abs(np.asarray(errors, dtype=float).ravel())
    p = observed_order(e)

    head = "  k        |e_k|      |e_k|/|e_(k-1)|   observed order"
    if exact_order is not None:
        head += f"   (theory: {exact_order:g})"
    lines = [head, "-" * len(head)]

    # Align the order estimates with the step they describe: p[j] uses e[j], e[j+1], e[j+2]
    # and so is reported against step j+2.
    finite_idx = [i for i, v in enumerate(e) if np.isfinite(v) and v > 0]
    order_by_step: dict[int, float] = {}
    for j, val in enumerate(p):
        if j + 2 < len(finite_idx):
            order_by_step[finite_idx[j + 2]] = float(val)

    for k, ek in enumerate(e):
        ratio = "" if k == 0 or e[k - 1] == 0 else f"{ek / e[k - 1]:14.6f}"
        order = order_by_step.get(k)
        ostr = "" if order is None or not np.isfinite(order) else f"{order:14.4f}"
        lines.append(f"{k:3d}  {ek:14.6e}  {ratio:>14s}  {ostr:>14s}")
    return "\n".join(lines)


def steps_to_tolerance(errors, tol: float) -> int:
    """The first step index whose error is at or below tol, or -1 if never.

    Used to compare methods fairly: not "which converges faster" in the abstract, but
    "which reaches 1e-10 in fewer function evaluations".
    """
    e = np.abs(np.asarray(errors, dtype=float).ravel())
    hits = np.nonzero(e <= tol)[0]
    return int(hits[0]) if hits.size else -1


# ---------------------------------------------------------------- power law fitting


def fit_power_law(x, y) -> tuple[float, float]:
    """Fit `y = C x**p` by least squares on the logs. Returns (p, C).

    This is the workhorse for every convergence-rate plot in the course. On a log-log plot
    the fitted p is the slope, so a second order method should give p near 2 and a
    quadrature rule of order 4 should give p near 4.

    Only points with positive x and y are used, since the fit lives in log space.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    lx, ly = np.log(x[mask]), np.log(y[mask])
    p, log_c = np.polyfit(lx, ly, 1)
    return float(p), float(np.exp(log_c))


def refinement_order(step_sizes, errors) -> float:
    """Order of accuracy from a step-size refinement study.

    For a method with `error = C h**p`, halving h should divide the error by 2**p. This
    fits p across all the step sizes at once, which is far more reliable than comparing a
    single pair.
    """
    p, _c = fit_power_law(step_sizes, errors)
    return p
