"""Adaptive quadrature: spend the evaluations where the integrand is difficult.

The idea
--------
A composite rule puts the same panel width everywhere. That is wasteful on an integrand that is
flat over most of its range and sharp over a little of it, and it is the usual case. Adaptive
quadrature instead estimates the error on each interval and subdivides only the intervals that
fail.

The error estimate
------------------
Compare a rule against itself on the two halves. Simpson's rule on a single interval of width
``h`` has error ``C h^5 f^(4)(xi)``: that is the **local** order, one higher than the composite
order 4, because the composite rule uses ``(b - a)/h`` of them. Applying it to the two halves
gives two errors of size ``C (h/2)^5``, so

    I - S(a,b)   ~  C h^5
    I - S2(a,b)  ~  2 C (h/2)^5  =  C h^5 / 16

and subtracting,

    S2 - S  ~  C h^5 (1 - 1/16)  =  (15/16) C h^5,   so   I - S2  ~  (S2 - S) / 15.

**The 15 comes from the local order 5, not the composite order 4.** Writing ``2^(p-1) - 1`` with
``p = 4`` gives 7, which is the single easiest mistake to make here and does not announce itself:
an estimate 15/7 too large just makes the routine conservative, so it passes every accuracy test
while spending twice the evaluations it needs. `local_error_estimate` returns the divisor so the
convention is visible rather than buried.

**The corrected value ``fine + estimate`` is one Romberg step**, and it is what most
implementations return while testing the uncorrected estimate. That makes them more accurate
than they claim. `correction_is_free_accuracy` measures the gap.

Where the estimate lies
-----------------------
The estimate assumes the error term exists, that is, that ``f`` has ``p`` continuous derivatives
on the interval. It is wrong exactly where adaptivity is most needed. Two failures matter:

- **A kink or a spike inside one interval.** The rule can sample around it and see nothing.
  `fooled_by_a_narrow_spike` builds an integrand that any fixed sample pattern misses.
- **A symmetric error that cancels.** ``S2 - S`` can be near zero while both are wrong.

Both are handled the same way in practice: never trust a single interval, and force a minimum
depth before accepting. `integrate` takes ``min_depth`` for that reason.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- the local rules


def simpson(f, lo: float, hi: float, _counter=None) -> float:
    """Simpson's rule on one interval, three evaluations."""
    a = float(lo)
    b = float(hi)
    mid = 0.5 * (a + b)
    x = np.asarray([a, mid, b], dtype=float)
    if _counter is not None:
        _counter[0] += 3
    v = np.asarray(f(x), dtype=float)
    return float((b - a) / 6.0 * (v[0] + 4.0 * v[1] + v[2]))


def local_error_estimate(f, lo: float, hi: float, order: int = 4) -> dict:
    """One interval against its two halves, with the estimate and the correction separated.

    ``coarse`` is the rule on the whole interval, ``fine`` is the same rule applied to each half
    and summed, and ``estimate`` is ``(fine - coarse) / (2^order - 1)``, which is 15 for
    Simpson. See the module docstring for why the exponent is the composite order and not one
    less than it.

    ``corrected = fine + estimate`` is one Richardson step. It uses no extra function values and
    is two orders more accurate, so a routine that accepts on ``|estimate| <= tol`` and returns
    ``corrected`` delivers far more than it was asked for. That is free accuracy, not a bound.
    """
    a = float(lo)
    b = float(hi)
    p = int(order)
    mid = 0.5 * (a + b)
    coarse = simpson(f, a, b)
    fine = simpson(f, a, mid) + simpson(f, mid, b)
    divisor = 2.0 ** p - 1.0
    estimate = (fine - coarse) / divisor
    return {"coarse": coarse, "fine": fine, "estimate": estimate,
            "corrected": fine + estimate,
            "divisor": divisor, "order": p}


# --------------------------------------------------------------------------- the recursion


def integrate(f, lo: float, hi: float, tol: float = 1e-10, max_depth: int = 50,
              min_depth: int = 2, order: int = 4, correct: bool = True,
              _counter=None) -> dict:
    """Adaptive Simpson, written as an explicit stack rather than a recursion.

    A recursive version is shorter and blows the interpreter's stack on a hard integrand at the
    depths this needs. The stack here holds ``(a, b, whole interval estimate, depth, tolerance)``
    and is drained until empty.

    The tolerance is **split** between the halves at each level, so the accepted intervals'
    errors sum to the requested tolerance rather than each being allowed the full amount. Codes
    that pass the same tolerance to both halves are asking for an answer up to ``2^depth`` times
    looser than the user requested.

    ``min_depth`` forces subdivision before any interval may be accepted, which is the cheapest
    protection against an estimate that is accidentally near zero.
    """
    a = float(lo)
    b = float(hi)
    if not b > a:
        raise ValueError(f"need lo < hi, got [{a}, {b}]")
    p = int(order)
    divisor = 2.0 ** p - 1.0
    total = 0.0
    intervals = []
    deepest = 0
    hit_the_limit = False
    stack = [(a, b, float(tol), 0)]
    while stack:
        left, right, budget, depth = stack.pop()
        deepest = max(deepest, depth)
        mid = 0.5 * (left + right)
        coarse = simpson(f, left, right, _counter)
        fine = simpson(f, left, mid, _counter) + simpson(f, mid, right, _counter)
        estimate = (fine - coarse) / divisor
        good = abs(estimate) <= budget and depth >= int(min_depth)
        if good or depth >= int(max_depth):
            if depth >= int(max_depth) and not good:
                hit_the_limit = True
            total += (fine + estimate) if correct else fine
            intervals.append((left, right, depth, abs(estimate)))
        else:
            half = 0.5 * budget
            stack.append((left, mid, half, depth + 1))
            stack.append((mid, right, half, depth + 1))
    intervals.sort()
    return {"value": total,
            "intervals": np.asarray([(r[0], r[1]) for r in intervals], dtype=float),
            "depths": np.asarray([r[2] for r in intervals]),
            "local_estimates": np.asarray([r[3] for r in intervals]),
            "panel_count": len(intervals),
            "deepest": deepest,
            "hit_the_depth_limit": hit_the_limit,
            "evaluations": (_counter[0] if _counter is not None else None)}


def where_the_work_went(f, lo: float, hi: float, tol: float = 1e-10) -> dict:
    """The panel widths the adaptive run chose, which is a picture of the integrand.

    A uniform rule would give one width everywhere. The ratio of the widest accepted panel to
    the narrowest is how much adaptivity was actually worth on this integrand, and on a smooth
    one it is 1.
    """
    out = integrate(f, lo, hi, tol)
    widths = out["intervals"][:, 1] - out["intervals"][:, 0]
    return {"edges": out["intervals"], "widths": widths,
            "widest": float(np.max(widths)), "narrowest": float(np.min(widths)),
            "width_ratio": float(np.max(widths) / np.min(widths)),
            "panel_count": out["panel_count"],
            "depths": out["depths"]}


def against_uniform(f, exact, lo: float, hi: float, tolerances=None) -> dict:
    """Adaptive against composite Simpson at the same number of evaluations.

    The comparison has to be at equal cost or it says nothing. For each tolerance the adaptive
    run is made first, its evaluations are counted, and composite Simpson is then given the same
    budget.
    """
    from . import newtoncotes as _nc

    tols = ([1e-4, 1e-6, 1e-8, 1e-10, 1e-12] if tolerances is None
            else [float(v) for v in np.atleast_1d(tolerances)])
    want = float(exact)
    rows = []
    for tol in tols:
        counter = [0]
        out = integrate(f, lo, hi, tol, _counter=counter)
        budget = counter[0]
        panels = max(1, (budget - 1) // 2)
        uniform_counter = [0]
        uniform = _nc.composite_shared(f, lo, hi, panels, 3, uniform_counter)
        rows.append((tol, budget, abs(out["value"] - want), uniform_counter[0],
                     abs(uniform - want), out["panel_count"]))
    return {"tolerance": np.asarray([r[0] for r in rows]),
            "adaptive_evaluations": np.asarray([r[1] for r in rows]),
            "adaptive_error": np.asarray([r[2] for r in rows]),
            "uniform_evaluations": np.asarray([r[3] for r in rows]),
            "uniform_error": np.asarray([r[4] for r in rows]),
            "adaptive_panels": np.asarray([r[5] for r in rows]),
            "adaptive_wins": np.asarray([r[2] < r[4] for r in rows])}


# --------------------------------------------------------------------------- honesty checks


def tolerance_is_met(f, exact, lo: float, hi: float, tolerances=None) -> dict:
    """Does the answer actually satisfy the tolerance it was given?

    The local estimate is an estimate. Summed over accepted intervals it is a guess at the
    global error, not a bound, and the routine can return an answer worse than requested. What
    it does on a smooth integrand and what it does on a hard one are different questions, so
    this reports the ratio of achieved error to requested tolerance rather than a yes or no.
    """
    tols = ([1e-4, 1e-6, 1e-8, 1e-10, 1e-12] if tolerances is None
            else [float(v) for v in np.atleast_1d(tolerances)])
    want = float(exact)
    rows = []
    for tol in tols:
        counter = [0]
        out = integrate(f, lo, hi, tol, _counter=counter)
        error = abs(out["value"] - want)
        rows.append((tol, error, error / tol, counter[0], out["panel_count"]))
    ratios = np.asarray([r[2] for r in rows])
    return {"tolerance": np.asarray([r[0] for r in rows]),
            "achieved_error": np.asarray([r[1] for r in rows]),
            "error_over_tolerance": ratios,
            "evaluations": np.asarray([r[3] for r in rows]),
            "panels": np.asarray([r[4] for r in rows]),
            "always_met": bool(np.all(ratios <= 1.0)),
            "worst_ratio": float(np.max(ratios))}


def correction_is_free_accuracy(f, exact, lo: float, hi: float, tolerances=None) -> dict:
    """The corrected answer against the uncorrected one, at identical cost.

    Both use exactly the same function values. The corrected version adds the Richardson term
    that the error estimate already computed, so the extra accuracy costs nothing but is also
    not covered by the tolerance test, which was applied to the uncorrected estimate.
    """
    tols = ([1e-4, 1e-6, 1e-8, 1e-10] if tolerances is None
            else [float(v) for v in np.atleast_1d(tolerances)])
    want = float(exact)
    rows = []
    for tol in tols:
        c1, c2 = [0], [0]
        with_correction = integrate(f, lo, hi, tol, correct=True, _counter=c1)
        without = integrate(f, lo, hi, tol, correct=False, _counter=c2)
        rows.append((tol, abs(with_correction["value"] - want),
                     abs(without["value"] - want), c1[0], c2[0]))
    return {"tolerance": np.asarray([r[0] for r in rows]),
            "corrected_error": np.asarray([r[1] for r in rows]),
            "uncorrected_error": np.asarray([r[2] for r in rows]),
            "corrected_evaluations": np.asarray([r[3] for r in rows]),
            "uncorrected_evaluations": np.asarray([r[4] for r in rows]),
            "same_cost": bool(all(r[3] == r[4] for r in rows))}


def _spike(centre: float, width: float, lo: float, hi: float):
    """A Gaussian bump and its exact integral over ``[lo, hi]``."""
    c = float(centre)
    w = float(width)
    if not w > 0.0:
        raise ValueError(f"need a positive width, got {w}")

    def f(x):
        t = np.asarray(x, dtype=float)
        return np.exp(-((t - c) / w) ** 2)

    exact = w * math.sqrt(math.pi) * 0.5 * (math.erf((float(hi) - c) / w)
                                            - math.erf((float(lo) - c) / w))
    return f, exact


def spike_centre_decides_everything(width: float = 1e-3, lo: float = 0.0, hi: float = 1.0,
                                    centres=None, tol: float = 1e-8) -> dict:
    """The same bump, moved, is either trivial or invisible.

    The very first step of adaptive Simpson on ``[a, b]`` evaluates at exactly five points: the
    two ends, the midpoint, and the two quarter points. **That is the whole grid the accept
    decision is made on.** A spike sitting on one of those five is found. A spike anywhere else
    is not there at all as far as the routine can tell: the coarse rule and both halves all read
    zero, the estimate is zero, the interval is accepted, and the answer comes back as 9
    evaluations of nothing.

    Measured at width 1e-3 on [0, 1], relative error of the returned value:

        centre                        relative error    evaluations
        0.5   (the midpoint)                 4.8e-08           1611
        0.25  (a quarter point)              4.8e-08           1593
        0.375 (never sampled)                1.0e+00              9
        0.4   (never sampled)                1.0e+00              9
        1/3   (never sampled)                1.0e+00              9

    Note that 0.375 is a dyadic rational and still fails: being dyadic is not enough, it has to
    be one of the five points this particular first step visits.

    A demonstration that puts the spike at the midpoint therefore proves the opposite of what it
    looks like it proves. That is why `fooled_by_a_narrow_spike` defaults its centre to two
    fifths of the way along, which is never a dyadic rational.
    """
    a = float(lo)
    b = float(hi)
    cs = ([0.5, 0.25, 0.375, 0.4, 1.0 / 3.0] if centres is None
          else [float(v) for v in np.atleast_1d(centres)])
    rows = []
    for c in cs:
        point = a + (b - a) * c
        f, exact = _spike(point, width, a, b)
        counter = [0]
        out = integrate(f, a, b, tol, min_depth=0, _counter=counter)
        rows.append((c, out["value"], abs(out["value"] - exact) / max(exact, 1e-300),
                     counter[0], out["panel_count"]))
    errors = np.asarray([r[2] for r in rows])
    return {"centre_fraction": np.asarray([r[0] for r in rows]),
            "value": np.asarray([r[1] for r in rows]),
            "relative_error": errors,
            "evaluations": np.asarray([r[3] for r in rows]),
            "panels": np.asarray([r[4] for r in rows]),
            "found": errors < 1e-3,
            "best": float(np.min(errors)), "worst": float(np.max(errors))}


def fooled_by_a_narrow_spike(width: float = 1e-3, lo: float = 0.0, hi: float = 1.0,
                             centre: float = None, tol: float = 1e-8) -> dict:
    """An integrand the estimate misses entirely, and how much forced depth it takes to find it.

    The bump is placed two fifths of the way along by default, which is not a dyadic rational,
    so no shallow sample lands on it. See `spike_centre_decides_everything` for why that choice
    is the honest one.

    Measured at width 1e-3 on [0, 1] with the centre at 0.4, the relative error of the returned
    value against ``min_depth``:

        min_depth   0    1    2    3    4      5
        rel error   1.0  1.0  1.0  1.0  1.0    8.4e-10

    So the answer is not slightly wrong, it is **the whole integral missing**, and it stays that
    way until the forced subdivision is fine enough that some node lands within a few widths of
    the peak. At depth 5 the panels are 1/32 wide and the nearest node falls 1.6 widths away,
    which is enough for the estimate to notice.
    """
    a = float(lo)
    b = float(hi)
    c = a + 0.4 * (b - a) if centre is None else float(centre)
    f, exact = _spike(c, width, a, b)
    rows = []
    for depth in (0, 1, 2, 3, 4, 5, 6, 8):
        counter = [0]
        out = integrate(f, a, b, tol, min_depth=depth, _counter=counter)
        rows.append((depth, out["value"], abs(out["value"] - exact) / max(exact, 1e-300),
                     counter[0], out["panel_count"]))
    found = [r[0] for r in rows if r[2] < 1e-3]
    return {"exact": exact, "width": float(width), "centre": c,
            "min_depth": np.asarray([r[0] for r in rows]),
            "value": np.asarray([r[1] for r in rows]),
            "relative_error": np.asarray([r[2] for r in rows]),
            "evaluations": np.asarray([r[3] for r in rows]),
            "panels": np.asarray([r[4] for r in rows]),
            "first_depth_that_finds_it": (found[0] if found else None)}


def spike_width_against_depth(widths=None, lo: float = 0.0, hi: float = 1.0,
                              tol: float = 1e-8) -> dict:
    """How deep the forced subdivision has to go as the spike narrows.

    Each halving of the width costs roughly one more level, because the subdivision is binary.
    That is the honest scaling: ``min_depth`` protects against spikes down to about
    ``(b - a) / 2^depth`` wide and no narrower, so **no finite choice protects against all of
    them**. A quadrature routine cannot promise to find structure it never samples, and no
    amount of adaptivity changes that. If the integrand has a spike, the caller has to say where.
    """
    ws = ([1e-1, 1e-2, 1e-3, 1e-4] if widths is None
          else [float(v) for v in np.atleast_1d(widths)])
    rows = []
    tried = None
    for w in ws:
        out = fooled_by_a_narrow_spike(w, lo, hi, tol=tol)
        tried = [int(v) for v in out["min_depth"]]
        rows.append((w, out["first_depth_that_finds_it"],
                     float(out["relative_error"][0])))
    return {"width": np.asarray([r[0] for r in rows]),
            "depth_needed": [r[1] for r in rows],
            "error_at_depth_zero": np.asarray([r[2] for r in rows]),
            "depths_tried": tried,
            "note": ("a depth of None means the spike was still missed at the deepest forced "
                     "level tried, which is %s, not that no depth would find it"
                     % (max(tried) if tried else None))}
