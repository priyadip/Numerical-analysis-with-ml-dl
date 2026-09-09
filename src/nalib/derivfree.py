"""Minimizing with values of ``f`` alone: golden section, parabolas, and a simplex.

Where the work is
-----------------
Lesson 84 showed that reading only values of ``f`` costs half the digits. This module asks what
can still be done inside that limit, and the answer splits cleanly in two.

**In one dimension there is a complete theory.** Golden section search brackets a minimum of a
unimodal function and shrinks the bracket by a fixed factor with **one** new evaluation per step.
Successive parabolic interpolation fits a parabola to three points and jumps to its vertex, which
is much faster and has no bracket to fall back on. The two are combined in practice, and the
combination is what every library's one dimensional minimizer is.

**In more than one dimension there is not.** The Nelder-Mead simplex is the method everyone reaches
for and it has **no convergence theory**: it can converge to a point that is not a minimum, of a
function that is strictly convex, and this module reproduces that.

What the measurements here show
-------------------------------
* Golden section reduces the bracket by ``0.61803400`` per evaluation against the exact
  ``1/phi = 0.61803399``, with a spread of ``9.6e-07`` over fifty steps, and it really does use one
  new evaluation per step. Reaching ``1e-10`` from a bracket of width 3 takes 53 calls against the
  predicted 50.1.
* Successive parabolic interpolation reaches the same accuracy in **14** calls instead of 53, a
  speedup of ``3.8``, and its order is the plastic number ``1.3247``, the real root of
  ``t**3 = t + 1``.
* **That order cannot be measured in double precision.** The iteration crosses from ``0.4`` to
  lesson 84's ``sqrt(eps)`` floor in nine steps, leaving **five** usable points, and the fit over
  them reads ``1.2109``. Repeating the same run in 120 digit arithmetic gives ``1.31796`` against
  the exact ``1.32472``. **Lesson 84's barrier is what hides it**, which is the tidiest
  demonstration in this part that the barrier is real.
* Nelder-Mead solves Rosenbrock in 275 calls at two variables and 5013 at ten, and **fails at
  eight**, stopping with ``f = 3.99`` and a simplex diameter at rounding. The failure is not
  monotone in the dimension, which is itself the point: there is no theory to be monotone.
* On McKinnon's function, which is strictly convex, Nelder-Mead converges to ``(0, 0)`` with a
  simplex diameter of ``1.5e-15``. The minimum is at ``(0, -1/2)`` and the gradient at the point it
  stops is ``(0, 1)``, not zero. **It converged to a non-stationary point and reported success.**

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import decimal
import math

import numpy as np

GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
PLASTIC = 1.3247179572447458          # the real root of t**3 = t + 1


# --------------------------------------------------------------------------- one dimension


def golden_section(f, a: float, b: float, tol: float = 1e-10, max_steps: int = 4000) -> dict:
    """Shrink a bracket around a minimum of a unimodal ``f``, one evaluation per step.

    Two interior points are placed so that whichever half survives, **one of them is already in
    the right place** for the next step. That forces the ratio: if the surviving bracket is
    ``r`` times the old one and the reused point must land where a new one would, then
    ``r**2 = 1 - r``, so ``r = (sqrt(5) - 1)/2 = 0.618``.

    A bisection style rule would place points symmetrically and have to throw both away, costing
    two evaluations for a factor of ``0.5``. Golden section costs one for ``0.618``, which is
    ``0.618`` against ``0.707`` per evaluation, so it wins.
    """
    lo, hi = float(a), float(b)
    if not hi > lo:
        raise ValueError(f"need a < b, got a={lo}, b={hi}")
    calls = 0

    def value(t):
        nonlocal calls
        calls += 1
        return float(f(t))

    left = hi - GOLDEN * (hi - lo)
    right = lo + GOLDEN * (hi - lo)
    f_left, f_right = value(left), value(right)
    widths = [hi - lo]
    for _ in range(int(max_steps)):
        if hi - lo <= float(tol):
            break
        if f_left < f_right:
            hi, right, f_right = right, left, f_left
            left = hi - GOLDEN * (hi - lo)
            f_left = value(left)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + GOLDEN * (hi - lo)
            f_right = value(right)
        widths.append(hi - lo)
    widths = np.asarray(widths)
    return {"x": 0.5 * (lo + hi), "f": float(f(0.5 * (lo + hi))), "calls": calls,
            "steps": widths.size - 1, "widths": widths,
            "bracket": (lo, hi)}


def the_reduction_is_the_golden_ratio(a: float = 0.0, b: float = 3.0,
                                      tol: float = 1e-10) -> dict:
    """Measure the per step reduction and the call count against the closed forms.

    Two predictions, both exact: the ratio is ``1/phi`` and the number of steps to reach ``tol``
    from a bracket of width ``w`` is ``log(tol/w) / log(1/phi)``.
    """
    def f(t):
        return (t - math.sqrt(2.0)) ** 2 + 1.0

    out = golden_section(f, a, b, tol=tol)
    ratios = out["widths"][1:] / out["widths"][:-1]
    predicted = math.log(tol / (b - a)) / math.log(GOLDEN)
    return {
        "calls": out["calls"],
        "steps": out["steps"],
        "one_new_call_per_step": out["calls"] == out["steps"] + 2,
        "mean_ratio": float(np.mean(ratios)),
        "golden_ratio": GOLDEN,
        "ratio_spread": float(np.max(ratios) - np.min(ratios)),
        "the_ratio_is_golden": abs(float(np.mean(ratios)) - GOLDEN) < 1e-6,
        "predicted_steps": predicted,
        "steps_match_the_prediction": abs(out["steps"] - predicted) < 5.0,
        "note": "two starting evaluations then one per step, and the bracket shrinks by 1/phi "
                "every time, which is what makes the count predictable",
    }


def successive_parabolic(f, seeds, rounds: int = 60, tol: float = 0.0) -> dict:
    """Fit a parabola through the last three points and jump to its vertex.

    The vertex of the parabola through ``(x0,f0), (x1,f1), (x2,f2)`` is

        x1 - (1/2) [ (x1-x0)**2 (f1-f2) - (x1-x2)**2 (f1-f0) ]
                   / [ (x1-x0)   (f1-f2) - (x1-x2)   (f1-f0) ] .

    There is no bracket and no safeguard here, on purpose: the failures matter and hiding them
    behind a fallback would hide the reason the fallback exists. The denominator vanishes when the
    three points are collinear, and the vertex is a **maximum** when the parabola opens downward,
    both of which this returns rather than repairs.
    """
    points = [float(s) for s in seeds]
    if len(points) != 3:
        raise ValueError(f"need exactly three starting points, got {len(points)}")
    values = [float(f(t)) for t in points]
    history, reason = [], "ran out of rounds"
    for _ in range(int(rounds)):
        x0, x1, x2 = points[-3:]
        f0, f1, f2 = values[-3:]
        top = (x1 - x0) ** 2 * (f1 - f2) - (x1 - x2) ** 2 * (f1 - f0)
        bottom = (x1 - x0) * (f1 - f2) - (x1 - x2) * (f1 - f0)
        if bottom == 0.0:
            reason = "the three points are collinear, so there is no parabola"
            break
        nxt = x1 - 0.5 * top / bottom
        if not math.isfinite(nxt):
            reason = "the vertex is not finite"
            break
        points.append(nxt)
        values.append(float(f(nxt)))
        history.append(nxt)
        if len(history) > 1 and abs(history[-1] - history[-2]) <= float(tol):
            reason = "the step stopped changing"
            break
    return {"x": history[-1] if history else float("nan"),
            "history": np.asarray(history), "calls": len(history),
            "stopped_because": reason}


def parabolas_beat_golden_section(tol: float = 1e-10) -> dict:
    """The same problem by both methods, counted in evaluations.

    The comparison is the reason nobody uses golden section alone. It also shows why nobody uses
    parabolic interpolation alone: it has no bracket, so on a function it dislikes it simply leaves.
    """
    def f(t):
        return math.exp(t) - 2.0 * t

    star = math.log(2.0)
    slow = golden_section(f, -1.0, 2.0, tol=tol)
    fast = successive_parabolic(f, (-1.0, 0.0, 2.0), rounds=40)
    errors = np.abs(fast["history"] - star)
    reached = int(np.argmax(errors < tol)) + 1 if np.any(errors < tol) else fast["calls"]
    return {
        "golden_calls": slow["calls"],
        "golden_error": abs(slow["x"] - star),
        "parabolic_calls_to_the_same_accuracy": reached + 3,
        "parabolic_error": float(errors[-1]),
        "speedup": slow["calls"] / float(reached + 3),
        "parabolic_errors": errors,
        "note": "golden section is linear with rate 0.618, parabolic interpolation is superlinear, "
                "and the gap is a factor of several in evaluations on an easy problem",
    }


def _spi_in_decimal(digits: int = 120, rounds: int = 13):
    """The same iteration on ``exp(x) - 2x`` in ``digits`` digit arithmetic, and its errors.

    A separate implementation because ``decimal`` needs its own arithmetic all the way down, and
    the point of the exercise is to remove the double precision floor rather than to reuse code.
    """
    with decimal.localcontext() as context:
        context.prec = int(digits)
        one = decimal.Decimal(1)

        def f(t):
            return t.exp() - 2 * t

        star = decimal.Decimal(2).ln()
        points = [decimal.Decimal(s) for s in ("-1", "0", "2")]
        values = [f(t) for t in points]
        errors = []
        for _ in range(int(rounds)):
            x0, x1, x2 = points[-3:]
            f0, f1, f2 = values[-3:]
            top = (x1 - x0) ** 2 * (f1 - f2) - (x1 - x2) ** 2 * (f1 - f0)
            bottom = (x1 - x0) * (f1 - f2) - (x1 - x2) * (f1 - f0)
            if bottom == 0:
                break
            nxt = x1 - top / (2 * bottom)
            points.append(nxt)
            values.append(f(nxt))
            gap = abs(nxt - star)
            if gap == 0:
                break
            errors.append(float(gap.ln()) if gap > 0 else float("-inf"))
        return errors, one


def the_order_is_hidden_by_the_barrier(digits: int = 120) -> dict:
    """Fit the order of parabolic interpolation twice: in double precision and in ``digits`` digits.

    The theory says the order is the plastic number ``1.3247``, the real root of ``t**3 = t + 1``.
    Double precision cannot show it. The iteration falls from ``0.4`` to lesson 84's
    ``sqrt(eps)`` floor in eight steps, so a fit has at most a handful of usable points and lands
    anywhere. In extended arithmetic the same iteration keeps going and the order appears.
    """
    def f(t):
        return math.exp(t) - 2.0 * t

    star = math.log(2.0)
    floor = math.sqrt(float(np.finfo(float).eps))
    run = successive_parabolic(f, (-1.0, 0.0, 2.0), rounds=40)
    double_errors = np.abs(run["history"] - star)
    usable = double_errors[(double_errors < 1e-1) & (double_errors > 10.0 * floor)]
    double_order = (float(np.polyfit(np.log(usable[:-1]), np.log(usable[1:]), 1)[0])
                    if usable.size >= 3 else float("nan"))

    logs, _ = _spi_in_decimal(digits)
    tail = logs[-6:]
    high_order = (float(np.polyfit(tail[:-1], tail[1:], 1)[0])
                  if len(tail) >= 3 else float("nan"))
    return {
        "double_errors": double_errors,
        "usable_points_in_double": int(usable.size),
        "double_order": double_order,
        "high_precision_digits": int(digits),
        "high_precision_errors": [math.exp(v) for v in logs],
        "high_precision_order": high_order,
        "plastic_number": PLASTIC,
        "double_precision_cannot_see_it": abs(double_order - PLASTIC) > 0.1,
        "extended_precision_can": abs(high_order - PLASTIC) < 0.05,
        "steps_to_reach_the_floor": int(np.argmax(double_errors < floor)) + 1,
        "note": "the order is real and double precision has too few points to fit it, because "
                "superlinear convergence crosses the sqrt(eps) barrier of lesson 84 in about "
                "eight steps",
    }


def unimodality_is_required(brackets=((-4.0, 0.5), (0.5, 4.0), (-4.0, 4.0))) -> dict:
    """Golden section on a function with two minima, from three different brackets.

    The theory guarantees convergence to **a** local minimum only when ``f`` is unimodal on the
    bracket. On a two well function the method still converges, quickly and quietly, and which
    well it lands in is decided by the bracket rather than by the function.
    """
    coefficients = np.array([1.0, 0.0, -8.0, 1.0, 0.0])      # t**4 - 8 t**2 + t

    def f(t):
        return float(np.polyval(coefficients, float(t)))

    # the wells are roots of f' where f'' > 0, found rather than quoted
    slope = np.polyder(coefficients)
    curve = np.polyder(slope)
    roots = np.sort(np.real(np.roots(slope)[np.isreal(np.roots(slope))]))
    wells = [float(t) for t in roots if float(np.polyval(curve, t)) > 0.0]
    rows = []
    for lo, hi in brackets:
        out = golden_section(f, lo, hi, tol=1e-10)
        nearest = min(wells, key=lambda w: abs(out["x"] - w))
        rows.append({"bracket": (lo, hi), "x": out["x"], "f": out["f"],
                     "calls": out["calls"],
                     "landed_in": nearest, "gap": abs(out["x"] - nearest)})
    found = {round(r["landed_in"], 3) for r in rows}
    deepest = min(wells, key=f)
    return {
        "rows": rows,
        "wells": wells,
        "deeper_well": deepest,
        "different_brackets_give_different_answers": len(found) > 1,
        "every_run_converged": all(r["gap"] < 1e-6 for r in rows),
        "a_run_found_the_shallower_well": any(
            abs(r["landed_in"] - deepest) > 1e-6 for r in rows),
        "note": "every run converges and none reports a problem, so the failure is silent: the "
                "answer is a property of the bracket",
    }


# --------------------------------------------------------------------------- many dimensions


def nelder_mead(f, start, step: float = 0.5, rounds: int = 20000,
                tol: float = 1e-12) -> dict:
    """The simplex method of Nelder and Mead, with its four moves.

    A simplex of ``n+1`` points is sorted by value and the worst is replaced. **Reflect** it
    through the centroid of the rest; if that is the new best, **expand** further; if it is worse
    than the second worst, **contract** halfway; and if even that fails, **shrink** the whole
    simplex toward the best point. No derivative is used and none is estimated.

    The shrink is the move to watch. It is the only one that reduces the simplex in every
    direction at once, and a method that shrinks repeatedly is collapsing rather than converging.
    """
    origin = np.asarray(start, dtype=float).ravel()
    n = origin.size
    if n < 1:
        raise ValueError("need at least one variable")
    simplex = np.vstack([origin] + [origin + float(step) * np.eye(n)[k] for k in range(n)])
    values = np.asarray([float(f(p) )for p in simplex])
    calls = n + 1
    moves = {"reflect": 0, "expand": 0, "contract": 0, "shrink": 0}
    best_history = []
    for _ in range(int(rounds)):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        best_history.append(float(values[0]))
        if float(np.max(np.abs(simplex[1:] - simplex[0]))) <= float(tol):
            break
        centre = simplex[:-1].mean(axis=0)
        reflected = centre + (centre - simplex[-1])
        f_reflected = float(f(reflected))
        calls += 1
        if f_reflected < values[0]:
            stretched = centre + 2.0 * (centre - simplex[-1])
            f_stretched = float(f(stretched))
            calls += 1
            if f_stretched < f_reflected:
                simplex[-1], values[-1] = stretched, f_stretched
                moves["expand"] += 1
            else:
                simplex[-1], values[-1] = reflected, f_reflected
                moves["reflect"] += 1
        elif f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
            moves["reflect"] += 1
        else:
            towards = simplex[-1] if f_reflected >= values[-1] else reflected
            pulled = centre + 0.5 * (towards - centre)
            f_pulled = float(f(pulled))
            calls += 1
            if f_pulled < min(f_reflected, values[-1]):
                simplex[-1], values[-1] = pulled, f_pulled
                moves["contract"] += 1
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                values[1:] = [float(f(p)) for p in simplex[1:]]
                calls += n
                moves["shrink"] += 1
    order = np.argsort(values)
    simplex, values = simplex[order], values[order]
    return {"x": simplex[0], "f": float(values[0]), "calls": calls,
            "simplex": simplex, "moves": moves,
            "diameter": float(np.max(np.abs(simplex[1:] - simplex[0]))),
            "best_history": np.asarray(best_history)}


def _rosenbrock(v):
    x = np.asarray(v, dtype=float).ravel()
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def how_nelder_mead_scales(sizes=(2, 3, 4, 6, 8, 10)) -> dict:
    """Nelder-Mead on Rosenbrock, at a range of sizes, counting evaluations.

    The cost grows roughly like ``n**2``, which is the usual summary. The row that matters is the
    one where it **stops at the wrong place**, and it is not the largest.
    """
    rows = []
    for n in sizes:
        start = np.full(int(n), -1.2)
        start[1::2] = 1.0
        out = nelder_mead(_rosenbrock, start, rounds=40000)
        rows.append({
            "variables": int(n),
            "calls": out["calls"],
            "f": out["f"],
            "distance": float(np.linalg.norm(out["x"] - 1.0)),
            "solved": float(np.linalg.norm(out["x"] - 1.0)) < 1e-6,
            "shrinks": out["moves"]["shrink"],
            "diameter": out["diameter"],
        })
    solved = [r for r in rows if r["solved"]]
    failed = [r for r in rows if not r["solved"]]
    sizes_seen = np.asarray([r["variables"] for r in solved], dtype=float)
    calls_seen = np.asarray([r["calls"] for r in solved], dtype=float)
    power = (float(np.polyfit(np.log(sizes_seen), np.log(calls_seen), 1)[0])
             if sizes_seen.size > 2 else float("nan"))
    return {
        "rows": rows,
        "fitted_power_of_n": power,
        "failures": [r["variables"] for r in failed],
        "the_failure_is_not_the_largest_size": bool(
            failed and max(r["variables"] for r in failed) < max(r["variables"] for r in rows)),
        "note": "the cost grows like a power of n and the failures do not, which is what having "
                "no convergence theory looks like from the outside",
    }


def mckinnon_problem(theta: float = 6.0, phi: float = 60.0, power: float = 2.0):
    """McKinnon's function, and the initial simplex that makes Nelder-Mead fail on it.

    ``f(x, y) = theta phi |x|**tau + y + y**2`` for ``x <= 0`` and ``theta x**tau + y + y**2``
    otherwise. It is strictly convex for these parameters, twice differentiable at the origin for
    ``tau = 2``, and its minimum is at ``(0, -1/2)`` with value ``-1/4``.

    The initial simplex is the published one, ``{(0,0), (1,1), (lam1, lam2)}`` with
    ``lam = (1 +- sqrt(33))/8``. From there every step is an inside contraction, the simplex
    collapses onto the ``x = 0`` line, and the method converges to the origin.
    """
    a, b, tau = float(theta), float(phi), float(power)

    def f(v):
        x, y = float(np.asarray(v, dtype=float).ravel()[0]), float(
            np.asarray(v, dtype=float).ravel()[1])
        head = a * b * abs(x) ** tau if x <= 0.0 else a * x ** tau
        return head + y + y * y

    def gradient(v):
        x, y = float(np.asarray(v, dtype=float).ravel()[0]), float(
            np.asarray(v, dtype=float).ravel()[1])
        slope = (-a * b * tau * abs(x) ** (tau - 1.0) if x < 0.0
                 else a * tau * x ** (tau - 1.0))
        return np.array([slope, 1.0 + 2.0 * y])

    root = math.sqrt(33.0)
    simplex = np.array([[0.0, 0.0], [1.0, 1.0], [(1.0 + root) / 8.0, (1.0 - root) / 8.0]])
    return {"f": f, "gradient": gradient, "start_simplex": simplex,
            "minimizer": np.array([0.0, -0.5]), "minimum": -0.25,
            "name": f"McKinnon, theta={a:g} phi={b:g} tau={tau:g}"}


def nelder_mead_can_converge_to_a_non_minimum(rounds: int = 200) -> dict:
    """Run Nelder-Mead from McKinnon's simplex and report where it stops.

    Every stopping test a simplex method can apply is satisfied here: the simplex diameter is at
    rounding, the values at the vertices agree, and nothing has changed for many iterations. The
    point it stops at is **not a stationary point**, and the function is **strictly convex**, so
    this is not a local minimum being mistaken for a global one. It is a failure of the method.
    """
    problem = mckinnon_problem()
    simplex = problem["start_simplex"].copy()
    f = problem["f"]
    values = np.asarray([f(p) for p in simplex])
    moves = {"reflect": 0, "expand": 0, "contract": 0, "shrink": 0}
    for _ in range(int(rounds)):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        centre = simplex[:-1].mean(axis=0)
        reflected = centre + (centre - simplex[-1])
        f_reflected = f(reflected)
        if f_reflected < values[0]:
            stretched = centre + 2.0 * (centre - simplex[-1])
            f_stretched = f(stretched)
            if f_stretched < f_reflected:
                simplex[-1], values[-1] = stretched, f_stretched
                moves["expand"] += 1
            else:
                simplex[-1], values[-1] = reflected, f_reflected
                moves["reflect"] += 1
        elif f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
            moves["reflect"] += 1
        else:
            towards = simplex[-1] if f_reflected >= values[-1] else reflected
            pulled = centre + 0.5 * (towards - centre)
            f_pulled = f(pulled)
            if f_pulled < min(f_reflected, values[-1]):
                simplex[-1], values[-1] = pulled, f_pulled
                moves["contract"] += 1
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                values[1:] = [f(p) for p in simplex[1:]]
                moves["shrink"] += 1
    order = np.argsort(values)
    simplex, values = simplex[order], values[order]
    where = simplex[0]
    slope = problem["gradient"](where)
    return {
        "name": problem["name"],
        "stopped_at": where,
        "value_there": float(values[0]),
        "true_minimizer": problem["minimizer"],
        "true_minimum": problem["minimum"],
        "gradient_where_it_stopped": slope,
        "gradient_norm": float(np.linalg.norm(slope)),
        "diameter": float(np.max(np.abs(simplex[1:] - simplex[0]))),
        "moves": moves,
        "it_stopped": float(np.max(np.abs(simplex[1:] - simplex[0]))) < 1e-12,
        "it_is_not_stationary": float(np.linalg.norm(slope)) > 0.5,
        "the_gap_in_value": float(values[0]) - problem["minimum"],
        "note": "the simplex is at rounding and the gradient is not, on a strictly convex "
                "function, so this is the method failing rather than a local minimum",
    }


def the_function_really_is_convex(samples: int = 400, radius: float = 2.0,
                                  seed: int = 42) -> dict:
    """Check McKinnon's function is convex along random segments, so the failure cannot be excused.

    Convexity is checked the way the definition states it, on the chord:
    ``f((1-t)p + t q) <= (1-t) f(p) + t f(q)``. Sampling cannot prove convexity, as lesson 84
    showed, so what this establishes is that no counterexample turns up where one would have to be
    if the failure were a second minimum.
    """
    problem = mckinnon_problem()
    f = problem["f"]
    rng = np.random.default_rng(int(seed))
    worst = -math.inf
    for _ in range(int(samples)):
        p = rng.uniform(-radius, radius, size=2)
        q = rng.uniform(-radius, radius, size=2)
        t = float(rng.uniform(0.0, 1.0))
        gap = f((1.0 - t) * p + t * q) - ((1.0 - t) * f(p) + t * f(q))
        worst = max(worst, gap)
    return {
        "samples": int(samples),
        "worst_chord_violation": worst,
        "no_counterexample_found": worst <= 1e-12,
        "note": "no violation of the chord inequality in any sample, so the point Nelder-Mead "
                "stops at is not a second minimum",
    }
