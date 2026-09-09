"""Adaptive step size control: embedded pairs, the control law, and what it actually buys.

The estimate
------------
Lesson 64 estimated a quadrature error by comparing a rule against itself on two halves. Here the
comparison is between **two methods sharing the same stages**.

An embedded pair carries two weight vectors ``b`` and ``b_hat`` over one tableau ``A``. The two
combinations have different orders, ``p`` and ``p + 1``, and their difference estimates the error
of the lower order one:

    error estimate = h * sum_i (b_i - b_hat_i) k_i

**The stages are computed once and used twice**, so the estimate costs nothing beyond the
evaluations the step already needed. That is the whole reason embedded pairs exist: an error
estimate by step halving costs three times as much.

The control law
---------------
Given an estimate ``e`` at step ``h`` from a method of order ``p``, the step that would have hit
the tolerance is

    h_new = h * (tol / e)^(1/(p+1))

with a safety factor of about 0.9 and limits on how far one step may change. The exponent is
``1/(p+1)`` because the **local** error is ``O(h^(p+1))``, which is the same off by one that set
lesson 64's Richardson divisor.

Which order to advance with
---------------------------
The estimate describes the lower order result and the higher order one is available for free.
Using it is **local extrapolation**, and every modern code does it, so the answer returned is
better than the tolerance it was tested against. That is the same free accuracy lesson 64
measured, and it is unearned in the same way.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .ivp import as_state
from .rungekutta import is_explicit

#: Embedded pairs, as ``name -> (A, b_high, b_low, c, order_high, order_low)``.
#: `bogacki shampine` is the 3(2) pair behind `ode23`; `fehlberg` is RKF45; `dormand prince`
#: is the 5(4) pair behind `ode45` and `dopri5`.
PAIRS = {
    "heun euler": (
        [[0, 0], [1, 0]],
        [0.5, 0.5], [1.0, 0.0], [0, 1], 2, 1),
    "bogacki shampine": (
        [[0, 0, 0, 0],
         [0.5, 0, 0, 0],
         [0, 0.75, 0, 0],
         [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0]],
        [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0],
        [7.0 / 24.0, 0.25, 1.0 / 3.0, 0.125],
        [0, 0.5, 0.75, 1], 3, 2),
    "fehlberg": (
        [[0, 0, 0, 0, 0, 0],
         [0.25, 0, 0, 0, 0, 0],
         [3.0 / 32.0, 9.0 / 32.0, 0, 0, 0, 0],
         [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0, 0, 0],
         [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0, 0],
         [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0]],
        [16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0],
        [25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -0.2, 0.0],
        [0, 0.25, 0.375, 12.0 / 13.0, 1.0, 0.5], 5, 4),
    "dormand prince": (
        [[0, 0, 0, 0, 0, 0, 0],
         [0.2, 0, 0, 0, 0, 0, 0],
         [3.0 / 40.0, 9.0 / 40.0, 0, 0, 0, 0, 0],
         [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0, 0, 0, 0],
         [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0, 0, 0],
         [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
          -5103.0 / 18656.0, 0, 0],
         [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
          11.0 / 84.0, 0]],
        [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
         11.0 / 84.0, 0.0],
        [5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0,
         187.0 / 2100.0, 1.0 / 40.0],
        [0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0], 5, 4),
}


def pair(name: str) -> tuple:
    """A named embedded pair as float arrays ``(A, b_high, b_low, c, p_high, p_low)``."""
    key = str(name).lower()
    if key not in PAIRS:
        raise ValueError(f"unknown pair {name!r}, expected one of {sorted(PAIRS)}")
    A, bh, bl, c, ph, pl = PAIRS[key]
    return (np.asarray(A, dtype=float), np.asarray(bh, dtype=float),
            np.asarray(bl, dtype=float), np.asarray(c, dtype=float), int(ph), int(pl))


def is_first_same_as_last(name: str) -> bool:
    """Does the last stage equal the next step's first, so one evaluation is reused?

    A tableau whose last row equals ``b_high`` and whose last ``c`` is 1 evaluates ``f`` at the
    new point as its final stage, so the next step's first stage is already known. Dormand-Prince
    has this property and Fehlberg does not, which is why ``ode45`` costs 6 evaluations per step
    rather than 7.
    """
    A, bh, bl, c, _ph, _pl = pair(name)
    return bool(abs(c[-1] - 1.0) < 1e-14
                and float(np.max(np.abs(A[-1] - bh))) < 1e-14)


def stages(f, t: float, y, h: float, A, c, _counter=None, first=None) -> np.ndarray:
    """The stage derivatives of an explicit tableau at one point.

    ``first`` supplies the first stage when it is already known, which is what the
    first-same-as-last property provides: for a tableau whose last row equals ``b`` and whose
    last node is 1, the final stage of an accepted step **is** ``f`` at the new point, so the
    next step's first stage costs nothing.
    """
    state = as_state(y)
    s = A.shape[0]
    k = np.empty((s, state.size))
    for i in range(s):
        if i == 0 and first is not None:
            k[0] = as_state(first)
            continue
        arg = state.copy()
        for j in range(i):
            if A[i, j] != 0.0:
                arg = arg + h * A[i, j] * k[j]
        if _counter is not None:
            _counter[0] += 1
        k[i] = as_state(f(t + c[i] * h, arg))
    return k


def embedded_step(f, t: float, y, h: float, name: str = "dormand prince",
                  _counter=None, first=None) -> dict:
    """One step of an embedded pair: both results and the error estimate, from one set of stages.

    The estimate is the difference of the two combinations, which is an estimate of the **lower**
    order result's error. The value normally advanced is the higher order one.
    """
    A, bh, bl, c, ph, pl = pair(name)
    if not is_explicit(A):
        raise ValueError(f"{name} is not explicit")
    k = stages(f, t, y, h, A, c, _counter, first)
    state = as_state(y)
    high = state + h * (bh @ k)
    low = state + h * (bl @ k)
    return {"high": high, "low": low, "estimate": high - low,
            "error_norm": float(np.max(np.abs(high - low))),
            "stages": k, "order_high": ph, "order_low": pl}


def the_estimate_is_free(name: str = "dormand prince") -> dict:
    """Count the evaluations an embedded estimate costs against step halving.

    Step halving computes one step at ``h`` and two at ``h/2`` to compare, so it costs three
    steps' worth of stages. The embedded pair computes one step's worth and reads the estimate
    off the same stages.
    """
    A, bh, bl, c, ph, pl = pair(name)
    s = int(bh.size)
    fsal = is_first_same_as_last(name)
    return {"name": name, "stages": s,
            "embedded_evaluations_per_step": s - (1 if fsal else 0),
            "step_halving_evaluations_per_step": 3 * s,
            "first_same_as_last": fsal,
            "ratio": (3.0 * s) / (s - (1 if fsal else 0))}


# --------------------------------------------------------------------------- the control law


def proposed_step(h: float, error: float, tol: float, order: int,
                  safety: float = 0.9, grow: float = 5.0, shrink: float = 0.2) -> float:
    """The step that would have hit the tolerance, with a safety factor and change limits.

    ``h_new = safety * h * (tol/error)^(1/(order+1))``, clipped to ``[shrink, grow]`` times ``h``.

    The exponent uses the **local** order, one above the method's global order, because that is
    what the one step error scales with. Using the global order instead makes the controller
    sluggish: it under corrects on every step and takes several to recover from a rejection.
    """
    if error <= 0.0:
        return float(h * grow)
    factor = safety * (tol / error) ** (1.0 / (int(order) + 1))
    return float(h * min(max(factor, shrink), grow))


def solve(f, t0: float, y0, t_end: float, tol: float = 1e-8, name: str = "dormand prince",
          h0: float = None, min_step: float = 1e-14, max_steps: int = 200000,
          local_extrapolation: bool = True, _counter=None) -> dict:
    """Adaptive integration with an embedded pair.

    Every accepted step reports its own size, so the returned ``h`` array is a picture of how
    hard the problem is where. `where_the_work_went` turns that into a summary.

    ``local_extrapolation`` advances with the higher order result. Turning it off advances with
    the one the estimate actually describes, which is more defensible and less accurate, and
    `local_extrapolation_is_free_accuracy` measures the gap.

    The first-same-as-last saving is taken whenever the pair has it **and** local extrapolation
    is on, because the reused stage is ``f`` at the point the high order result lands on. With
    extrapolation off the step advances somewhere else and the stage is not reusable, which is a
    real cost of the more defensible choice and is easy to overlook.
    """
    a = float(t0)
    b = float(t_end)
    if not b > a:
        raise ValueError(f"need t0 < t_end, got [{a}, {b}]")
    _A, _bh, _bl, _c, ph, pl = pair(name)
    order = pl
    y = as_state(y0)
    h = float((b - a) / 100.0 if h0 is None else h0)
    t = a
    ts, ys, hs = [a], [y.copy()], []
    rejected = 0
    reuse = is_first_same_as_last(name) and bool(local_extrapolation)
    carried = None
    for _ in range(int(max_steps)):
        if t >= b:
            break
        h = min(h, b - t)
        out = embedded_step(f, t, y, h, name, _counter, carried)
        error = out["error_norm"]
        scale = tol * max(1.0, float(np.max(np.abs(y))))
        if error <= scale or h <= min_step:
            t = t + h
            y = out["high"] if local_extrapolation else out["low"]
            ts.append(t)
            ys.append(y.copy())
            hs.append(h)
            carried = out["stages"][-1].copy() if reuse else None
        else:
            rejected += 1
            # a rejected step recomputes from the same point, so the carried stage still holds
        h = proposed_step(h, max(error, 1e-300), scale, order)
        if h <= min_step:
            h = min_step
    return {"t": np.asarray(ts), "y": np.stack(ys), "h": np.asarray(hs),
            "accepted": len(hs), "rejected": rejected,
            "reused_the_last_stage": reuse,
            "reached_the_end": bool(t >= b - 1e-12),
            "evaluations": (_counter[0] if _counter is not None else None),
            "name": name}


# --------------------------------------------------------------------------- measurements


def orders_are_what_they_claim(f, exact, t0: float, y0, t_end: float, names=None,
                               step_counts=None) -> dict:
    """Each half of each pair, run at fixed steps, against the order it is labelled with."""
    from .ivp import error_against_step
    from .rungekutta import step_from

    wanted = (sorted(PAIRS) if names is None else [str(v) for v in np.atleast_1d(names)])
    ns = ([10, 20, 40, 80, 160, 320] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    rows = []
    for name in wanted:
        A, bh, bl, c, ph, pl = pair(name)
        high = error_against_step(f, exact, t0, y0, t_end, ns, step_from(A, bh, c))
        low = error_against_step(f, exact, t0, y0, t_end, ns, step_from(A, bl, c))
        rows.append((name, ph, high["fitted_order"], pl, low["fitted_order"]))
    return {"names": [r[0] for r in rows],
            "claimed_high": np.asarray([r[1] for r in rows]),
            "fitted_high": np.asarray([r[2] for r in rows]),
            "claimed_low": np.asarray([r[3] for r in rows]),
            "fitted_low": np.asarray([r[4] for r in rows]),
            "all_match": bool(all(abs(r[2] - r[1]) < 0.35 and abs(r[4] - r[3]) < 0.35
                                  for r in rows))}


def the_estimate_predicts_the_error(f, exact, t0: float, y0, t_end: float,
                                    name: str = "dormand prince", steps=None) -> dict:
    """The embedded estimate against the true one step error of the lower order result.

    Starting each step from the exact solution isolates the local error, so what is measured is
    the estimate's quality rather than the accumulation. The ratio should go to 1 as the step
    shrinks, and where it does not the pair is being used outside its asymptotic regime.
    """
    hs = ([0.4, 0.2, 0.1, 0.05, 0.025, 0.0125] if steps is None
          else [float(v) for v in np.atleast_1d(steps)])
    a = float(t0)
    rows = []
    for h in hs:
        start = as_state(exact(a))
        out = embedded_step(f, a, start, h, name)
        truth = as_state(exact(a + h))
        true_low = float(np.max(np.abs(truth - out["low"])))
        true_high = float(np.max(np.abs(truth - out["high"])))
        rows.append((h, out["error_norm"], true_low, true_high))
    est = np.asarray([r[1] for r in rows])
    low = np.asarray([r[2] for r in rows])
    high = np.asarray([r[3] for r in rows])
    return {"h": np.asarray([r[0] for r in rows]), "estimate": est,
            "true_low_error": low, "true_high_error": high,
            "estimate_over_low": est / np.maximum(low, 1e-300),
            "estimate_over_high": est / np.maximum(high, 1e-300)}


def local_extrapolation_is_free_accuracy(f, exact, t0: float, y0, t_end: float,
                                         tolerances=None,
                                         name: str = "dormand prince") -> dict:
    """Advancing with the higher order result, against advancing with the tested one.

    Both use the same stages, so the cost is identical. The higher order answer is better by a
    margin that grows as the tolerance tightens, and nothing in the accept test knows about it,
    which is exactly lesson 64's free accuracy in a different setting.
    """
    tols = ([1e-4, 1e-6, 1e-8, 1e-10] if tolerances is None
            else [float(v) for v in np.atleast_1d(tolerances)])
    want = as_state(exact(float(t_end)))
    rows = []
    for tol in tols:
        c1, c2 = [0], [0]
        with_it = solve(f, t0, y0, t_end, tol, name, _counter=c1)
        without = solve(f, t0, y0, t_end, tol, name, local_extrapolation=False, _counter=c2)
        rows.append((tol,
                     float(np.max(np.abs(with_it["y"][-1] - want))),
                     float(np.max(np.abs(without["y"][-1] - want))),
                     c1[0], c2[0]))
    with_err = np.asarray([r[1] for r in rows])
    without_err = np.asarray([r[2] for r in rows])
    return {"tolerance": np.asarray([r[0] for r in rows]),
            "with_extrapolation": with_err, "without_extrapolation": without_err,
            "evaluations_with": np.asarray([r[3] for r in rows]),
            "evaluations_without": np.asarray([r[4] for r in rows]),
            "gain": without_err / np.maximum(with_err, 1e-300)}


def where_the_work_went(f, t0: float, y0, t_end: float, tol: float = 1e-8,
                        name: str = "dormand prince") -> dict:
    """The accepted step sizes, which draw a picture of the problem.

    On a problem with one time scale the steps are all the same and adaptivity bought nothing,
    exactly as in lesson 64. On one with several the ratio of largest to smallest says how many.
    """
    out = solve(f, t0, y0, t_end, tol, name)
    h = out["h"]
    return {"t": out["t"][:-1], "h": h, "accepted": out["accepted"],
            "rejected": out["rejected"],
            "largest": float(np.max(h)), "smallest": float(np.min(h)),
            "ratio": float(np.max(h) / np.min(h)),
            "evaluations": out["accepted"] + out["rejected"]}


def against_fixed_steps(f, exact, t0: float, y0, t_end: float, tolerances=None,
                        name: str = "dormand prince") -> dict:
    """Adaptive against a fixed step run of the same method at the same evaluation count.

    The only comparison that means anything is at equal cost, and the answer depends on the
    problem exactly as it did in lesson 64: adaptivity wins where the solution has structure on
    several time scales and loses where it does not.
    """
    from .ivp import integrate
    from .rungekutta import step_from

    tols = ([1e-4, 1e-6, 1e-8, 1e-10] if tolerances is None
            else [float(v) for v in np.atleast_1d(tolerances)])
    A, bh, bl, c, ph, pl = pair(name)
    fixed_step = step_from(A, bh, c)
    per_step = int(bh.size)
    want = as_state(exact(float(t_end)))
    rows = []
    for tol in tols:
        counter = [0]
        adaptive = solve(f, t0, y0, t_end, tol, name, _counter=counter)
        budget = counter[0]
        steps = max(1, budget // per_step)
        fixed_counter = [0]
        fixed = integrate(f, t0, y0, t_end, steps, fixed_step, fixed_counter)
        rows.append((tol, budget,
                     float(np.max(np.abs(adaptive["y"][-1] - want))),
                     fixed_counter[0],
                     float(np.max(np.abs(fixed["y"][-1] - want))),
                     adaptive["accepted"], adaptive["rejected"]))
    adaptive_err = np.asarray([r[2] for r in rows])
    fixed_err = np.asarray([r[4] for r in rows])
    return {"tolerance": np.asarray([r[0] for r in rows]),
            "adaptive_evaluations": np.asarray([r[1] for r in rows]),
            "adaptive_error": adaptive_err,
            "fixed_evaluations": np.asarray([r[3] for r in rows]),
            "fixed_error": fixed_err,
            "accepted": np.asarray([r[5] for r in rows]),
            "rejected": np.asarray([r[6] for r in rows]),
            "adaptive_wins": np.asarray([a < b for a, b in zip(adaptive_err, fixed_err)])}


def tolerance_is_met(f, exact, t0: float, y0, t_end: float, tolerances=None,
                     name: str = "dormand prince") -> dict:
    """Does the answer satisfy the tolerance it was given?

    The control law bounds the **local** error per step, and the global error is the accumulation
    of many of them, so there is no reason for the global error to respect the tolerance and
    routinely it does not. Reporting the ratio says by how much.
    """
    tols = ([1e-4, 1e-6, 1e-8, 1e-10, 1e-12] if tolerances is None
            else [float(v) for v in np.atleast_1d(tolerances)])
    want = as_state(exact(float(t_end)))
    rows = []
    for tol in tols:
        counter = [0]
        out = solve(f, t0, y0, t_end, tol, name, _counter=counter)
        error = float(np.max(np.abs(out["y"][-1] - want)))
        rows.append((tol, error, error / tol, counter[0], out["accepted"], out["rejected"]))
    ratios = np.asarray([r[2] for r in rows])
    return {"tolerance": np.asarray([r[0] for r in rows]),
            "achieved_error": np.asarray([r[1] for r in rows]),
            "error_over_tolerance": ratios,
            "evaluations": np.asarray([r[3] for r in rows]),
            "accepted": np.asarray([r[4] for r in rows]),
            "rejected": np.asarray([r[5] for r in rows]),
            "always_met": bool(np.all(ratios <= 1.0)),
            "worst_ratio": float(np.max(ratios))}
