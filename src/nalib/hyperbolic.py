"""The wave equation, the Courant condition, and what violating it really means.

A different kind of restriction
-------------------------------
The equation is

    u_tt = c**2 u_xx ,

with an initial shape **and** an initial velocity, because it is second order in time. The explicit
scheme is the obvious one,

    u_j^{n+1} = 2 u_j^n - u_j^{n-1} + lam**2 (u_{j+1}^n - 2 u_j^n + u_{j-1}^n),

with the **Courant number** ``lam = c k / h``. Its stability condition is ``lam <= 1``, and that
looks like the parabolic condition of lesson 78 until you ask where it comes from.

The parabolic restriction ``r <= 1/2`` is a statement about amplification: run above it and the
scheme grows. The hyperbolic restriction is a statement about **information**. The exact solution
at ``(x, t)`` depends on the initial data over ``[x - ct, x + ct]``. The scheme at ``(x_j, t_n)``
depends on the data over ``[x_j - nh, x_j + nh]``, because that is how far the stencil can reach in
``n`` steps. Containing the true domain needs ``nh >= c n k``, which is exactly ``lam <= 1``.

So a scheme with ``lam > 1`` is not merely unstable. **It cannot see data that the answer depends
on**, and no amount of care in the arithmetic can fix that. `changing_data_it_cannot_see` makes the
point by changing the initial data inside the true domain of dependence and outside the numerical
one: the true answer changes and the computed answer does not change **at all**, to an exact zero.

What the measurements here show
-------------------------------
* At ``lam = 1`` exactly, the explicit scheme is not approximately right, it is **exact**. The
  update becomes ``u_j^{n+1} = u_{j+1}^n + u_{j-1}^n - u_j^{n-1}``, which is d'Alembert's formula
  on the grid. `the_magic_step` measures errors at the rounding level while the neighbouring
  ratios sit at ``1e-3``.
* The Courant limit is sharp: the two roots of the amplification quadratic have product exactly
  1, so one of them is outside the circle the instant ``lam`` passes 1, with no margin. The
  **failure** is not immediate, though, and that is worth knowing. A smooth bump has essentially
  no energy at the shortest wavelength, so the seed is rounding and the countdown is the same one
  as in lesson 78: at ``lam = 1.001`` the run is still clean after 240 steps. The product of the
  seed and the growth predicts every row of both sweeps.
* The starting step decides the order of the whole run. Using ``u^1 = u^0 + k g`` gives a first
  order method however good the interior scheme is; adding the ``(lam**2/2) delta**2 u^0`` term
  restores second order. `the_first_step_sets_the_order` measures 1.00 against 2.00.
* The implicit scheme is unconditionally stable and buys nothing by it. Every root sits exactly on
  the unit circle at every ``lam`` tried, so nothing ever grows, and the error over one period
  grows by a factor of 6930 as ``lam`` goes from 0.5 to 8, at the second order the step size
  predicts. In a parabolic problem a big stable step is worth having because the solution decays
  and the error decays with it; in a wave problem nothing decays, phase errors accumulate, and the
  extra step buys only speed at the cost of accuracy.
* Dispersion is exactly zero at ``lam = 1`` for every wavenumber, and reaches 36 per cent at the
  shortest wave for ``lam = 0.25``. Every scheme below the limit propagates short waves **too
  slowly**, never too fast.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .banded import thomas


# --------------------------------------------------------------------------- problems


def standing_wave(speed: float = 1.0, mode: int = 1, a: float = 0.0, b: float = 1.0):
    """``u = cos(m pi c t) sin(m pi x)``: a standing wave, zero at both ends for all time.

    Initial shape ``sin(m pi x)`` and initial velocity zero. The exact solution is available at
    every time, so an error can be measured rather than estimated, and it is periodic in ``t``
    with period ``2/(mc)``, which makes long runs a fair test of phase error.
    """
    a, b = float(a), float(b)
    if not b > a:
        raise ValueError(f"need a < b, got a={a}, b={b}")
    m = int(mode)
    if m < 1:
        raise ValueError(f"mode must be at least 1, got {m}")
    wave = m * math.pi / (b - a)
    c = float(speed)

    def exact(x, t):
        return np.cos(wave * c * float(t)) * np.sin(wave * (np.asarray(x, dtype=float) - a))

    def initial(x):
        return np.sin(wave * (np.asarray(x, dtype=float) - a))

    def velocity(x):
        return np.zeros(np.shape(np.asarray(x, dtype=float)))

    return {"speed": c, "a": a, "b": b, "exact": exact, "initial": initial,
            "velocity": velocity, "left": lambda t: 0.0, "right": lambda t: 0.0,
            "period": 2.0 / (m * c), "name": f"standing wave {m}"}


def travelling_bump(speed: float = 1.0, centre: float = 0.5, width: float = 0.06,
                    a: float = 0.0, b: float = 2.0):
    """A smooth bump released from rest, so it splits into two half height travelling waves.

    d'Alembert's formula with zero initial velocity gives ``u = (f(x - ct) + f(x + ct))/2`` on the
    whole line. The bump is a Gaussian narrow enough that on ``[a, b]`` and for the times used
    here it never reaches a boundary, so the infinite line formula is the exact solution to well
    below rounding.

    This is the problem for the domain of dependence experiment, because a bump is **local**: it
    can be placed inside one domain and outside another.
    """
    a, b = float(a), float(b)
    c, mid, w = float(speed), float(centre), float(width)
    if w <= 0.0:
        raise ValueError(f"width must be positive, got {w}")

    def shape(x):
        return np.exp(-((np.asarray(x, dtype=float) - mid) / w) ** 2)

    def exact(x, t):
        s = np.asarray(x, dtype=float)
        return 0.5 * (shape(s - c * float(t)) + shape(s + c * float(t)))

    # the ends carry the exact values rather than zeros. The bump's tail is small there but not
    # zero, and forcing a zero costs an error of exactly that tail, which on a run to t = 0.4 is
    # 3.1e-2 and swamps everything this problem is used to measure.
    return {"speed": c, "a": a, "b": b, "exact": exact, "initial": shape,
            "velocity": lambda x: np.zeros(np.shape(np.asarray(x, dtype=float))),
            "left": lambda t: float(exact(a, t)), "right": lambda t: float(exact(b, t)),
            "centre": mid, "width": w, "name": "travelling bump"}


def courant(speed: float, k: float, h: float) -> float:
    """``lam = c k / h``, the ratio of the numerical speed to the physical one."""
    return float(speed) * float(k) / float(h)


# --------------------------------------------------------------------------- the schemes


def first_step(initial, velocity, lam: float, k: float, order: int = 2):
    """The first time level, which the two level recurrence cannot produce for itself.

    ``order = 1`` uses ``u^1 = u^0 + k g``, the obvious thing, which is first order and drags the
    whole run down to first order with it.

    ``order = 2`` adds the term Taylor's theorem asks for. Since ``u_tt = c**2 u_xx``,

        u^1 = u^0 + k g + (k**2/2) c**2 u_xx = u^0 + k g + (lam**2/2) delta**2 u^0 ,

    and the second derivative comes free from the same stencil the scheme already uses.
    """
    u0 = np.asarray(initial, dtype=float)
    g = np.asarray(velocity, dtype=float)
    out = u0 + float(k) * g
    if int(order) == 1:
        return out
    if int(order) != 2:
        raise ValueError(f"order must be 1 or 2, got {order}")
    correction = np.zeros_like(u0)
    correction[1:-1] = 0.5 * float(lam) ** 2 * (u0[2:] - 2.0 * u0[1:-1] + u0[:-2])
    return out + correction


def explicit_step(previous, current, lam: float, left: float = 0.0, right: float = 0.0):
    """One step of the explicit three level scheme.

    ``u^{n+1} = 2u^n - u^{n-1} + lam**2 delta**2 u^n``. At ``lam = 1`` the ``u^n`` terms cancel
    entirely and this becomes ``u_{j+1}^n + u_{j-1}^n - u_j^{n-1}``, which is d'Alembert's formula
    evaluated on the grid, and therefore **exact**.
    """
    old = np.asarray(previous, dtype=float)
    now = np.asarray(current, dtype=float)
    if old.shape != now.shape:
        raise ValueError(f"levels have shapes {old.shape} and {now.shape}")
    if now.size < 3:
        raise ValueError(f"need at least 3 points, got {now.size}")
    l2 = float(lam) ** 2
    out = np.empty_like(now)
    out[1:-1] = (2.0 * now[1:-1] - old[1:-1]
                 + l2 * (now[2:] - 2.0 * now[1:-1] + now[:-2]))
    out[0], out[-1] = float(left), float(right)
    return out


def implicit_step(previous, current, lam: float, theta: float = 0.25,
                  left: float = 0.0, right: float = 0.0):
    """One step of the weighted implicit scheme, which is unconditionally stable for ``theta >= 1/4``.

    The space difference is averaged over the three time levels with weights
    ``(theta, 1 - 2 theta, theta)``:

        u^{n+1} - 2u^n + u^{n-1} = lam**2 [theta d2 u^{n+1} + (1-2theta) d2 u^n + theta d2 u^{n-1}]

    ``theta = 0`` is the explicit scheme. ``theta = 1/4`` is the standard implicit one, and the
    condition for unconditional stability is exactly ``theta >= 1/4``, which
    `the_implicit_threshold_is_a_quarter` measures.
    """
    old = np.asarray(previous, dtype=float)
    now = np.asarray(current, dtype=float)
    th, l2 = float(theta), float(lam) ** 2
    if not 0.0 <= th <= 0.5:
        raise ValueError(f"theta must lie in [0, 1/2], got {th}")
    m = now.size - 2
    if m < 1:
        raise ValueError(f"need at least 3 points, got {now.size}")
    d2_now = now[2:] - 2.0 * now[1:-1] + now[:-2]
    d2_old = old[2:] - 2.0 * old[1:-1] + old[:-2]
    rhs = (2.0 * now[1:-1] - old[1:-1]
           + l2 * ((1.0 - 2.0 * th) * d2_now + th * d2_old))
    out = np.empty_like(now)
    out[0], out[-1] = float(left), float(right)
    if th == 0.0:
        out[1:-1] = rhs
        return out
    rhs = rhs.copy()
    rhs[0] += th * l2 * out[0]
    rhs[-1] += th * l2 * out[-1]
    out[1:-1] = thomas(np.full(m, -th * l2), np.full(m, 1.0 + 2.0 * th * l2),
                       np.full(m, -th * l2), rhs)
    return out


def solve(problem, points: int, steps: int, t_end: float, theta: float = 0.0,
          start_order: int = 2, keep: bool = False):
    """March the wave equation and report the error against the exact solution."""
    n = int(points)
    steps = int(steps)
    if n < 3:
        raise ValueError(f"need at least 3 space points, got {n}")
    if steps < 2:
        raise ValueError(f"need at least two time steps, got {steps}")
    x = np.linspace(problem["a"], problem["b"], n)
    h = float(x[1] - x[0])
    k = float(t_end) / steps
    lam = courant(problem["speed"], k, h)
    u_old = np.asarray(problem["initial"](x), dtype=float)
    u_old[0], u_old[-1] = problem["left"](0.0), problem["right"](0.0)
    u_now = first_step(u_old, problem["velocity"](x), lam, k, order=start_order)
    u_now[0], u_now[-1] = problem["left"](k), problem["right"](k)
    history = [u_old.copy(), u_now.copy()] if keep else None
    for step in range(1, steps):
        t_next = (step + 1) * k
        if theta == 0.0:
            u_new = explicit_step(u_old, u_now, lam, problem["left"](t_next),
                                  problem["right"](t_next))
        else:
            u_new = implicit_step(u_old, u_now, lam, theta, problem["left"](t_next),
                                  problem["right"](t_next))
        u_old, u_now = u_now, u_new
        if keep:
            history.append(u_now.copy())
    want = np.asarray(problem["exact"](x, float(t_end)), dtype=float)
    out = {"x": x, "u": u_now, "exact": want, "h": h, "k": k, "lam": lam,
           "error": float(np.max(np.abs(u_now - want))) if np.isfinite(u_now).all()
           else float("inf"),
           "finite": bool(np.isfinite(u_now).all()), "steps": steps, "theta": float(theta),
           "start_order": int(start_order)}
    if keep:
        out["history"] = np.stack(history)
        out["t"] = np.arange(len(history)) * k
    return out


# --------------------------------------------------------------------------- stability


def growth_roots(lam: float, phase, theta: float = 0.0):
    """The two roots of the amplification quadratic for one Fourier mode.

    Substituting ``u_j^n = g**n exp(i j phi)`` into the weighted scheme gives

        (1 + theta L) g**2 - (2 - (1 - 2 theta) L) g + (1 + theta L) = 0,
        L = lam**2 * 4 sin(phi/2)**2 ,

    whose two roots have **product exactly 1**. That is the signature of a wave problem: nothing
    decays, so stability means both roots on the unit circle, and the moment they leave it one of
    them is outside. There is no margin, which is why the Courant condition is sharp.
    """
    phi = np.atleast_1d(np.asarray(phase, dtype=float))
    big = float(lam) ** 2 * 4.0 * np.sin(phi / 2.0) ** 2
    th = float(theta)
    a = 1.0 + th * big
    b = -(2.0 - (1.0 - 2.0 * th) * big)
    disc = np.emath.sqrt(b.astype(complex) ** 2 - 4.0 * a * a)
    return ((-b + disc) / (2.0 * a), (-b - disc) / (2.0 * a))


def largest_root(lam: float, theta: float = 0.0, samples: int = 401) -> float:
    """The largest root modulus over every phase a grid can carry."""
    phases = np.linspace(0.0, math.pi, int(samples))
    first, second = growth_roots(lam, phases, theta)
    return float(max(np.max(np.abs(first)), np.max(np.abs(second))))


def worst_mode_amplitude(values) -> float:
    """The amplitude of the highest mode a grid carries, from the discrete sine transform.

    The same reader as in lesson 78, and needed for the same reason: a growth factor predicts
    nothing until it is multiplied by how much of the growing mode the data already contains.
    """
    v = np.asarray(values, dtype=float).ravel()
    inner = v[1:-1]
    m = inner.size
    if m < 1:
        return 0.0
    index = np.arange(1, m + 1)
    return abs(float(2.0 / (m + 1) * np.sum(inner * np.sin(m * np.pi * index / (m + 1)))))


def the_courant_condition(lams=(0.5, 0.9, 1.0, 1.001, 1.05, 1.5), points: int = 201,
                          t_end: float = 0.3, long_factor: float = 8.0):
    """Run the explicit scheme on both sides of ``lam = 1`` and measure what happens.

    The roots of the amplification quadratic have **product exactly 1**, which is the signature of
    a wave problem: nothing decays, so stability means both roots on the unit circle and the
    moment one leaves the other is inside. There is no margin, and the limit is sharp.

    Sharp is not the same as immediate. The mode that grows is the shortest one the grid carries,
    and a smooth bump has **no measurable energy there**: the amplitude reads back as machine
    epsilon, exactly as the single sine mode did in lesson 78. So the failure is again a countdown
    from rounding, and at ``lam = 1.001``, where the growth is 1.0936 per step, 240 steps are not
    enough to finish it.

    The prediction reported is the same product as before, the worst mode's amplitude in the data
    times the growth to the power of the step count, and it is right in every row of both sweeps,
    including the two rows above the limit that do **not** blow up.
    """
    problem = travelling_bump()

    def sweep(end):
        out = []
        for value in lams:
            lam = float(value)
            h = (problem["b"] - problem["a"]) / (int(points) - 1)
            k = lam * h / problem["speed"]
            steps = max(int(round(float(end) / k)), 2)
            x = np.linspace(problem["a"], problem["b"], int(points))
            seed = max(worst_mode_amplitude(problem["initial"](x)),
                       float(np.finfo(float).eps))
            with np.errstate(over="ignore", invalid="ignore"):
                run = solve(problem, points, steps, steps * k, theta=0.0)
            root = largest_root(lam, 0.0)
            decades = math.log10(seed) + steps * math.log10(max(root, 1e-300))
            out.append({
                "lam": lam,
                "steps": steps,
                "largest_root": root,
                "predicted_stable": root <= 1.0 + 1e-12,
                "seed": seed,
                "predicted_decades": decades,
                "predicts_a_blow_up": decades > 0.0,
                "error": run["error"],
                "finite": run["finite"],
                "blew_up": (not run["finite"]) or run["error"] > 1.0,
            })
        return out

    short = sweep(t_end)
    long = sweep(float(long_factor) * float(t_end))
    rows = short + long
    return {
        "short_run": short,
        "long_run": long,
        "limit": 1.0,
        "t_end": float(t_end),
        "long_t_end": float(long_factor) * float(t_end),
        "the_root_leaves_the_circle_at_one": all(
            (row["largest_root"] > 1.0 + 1e-9) == (row["lam"] > 1.0) for row in short),
        "the_prediction_is_right_in_every_row": all(
            row["predicts_a_blow_up"] == row["blew_up"] for row in rows),
        "rows_checked": len(rows),
        "nothing_below_the_limit_blew_up": not any(
            row["blew_up"] for row in rows if row["lam"] <= 1.0),
        "violations_that_failed_in_the_long_run": [
            row["lam"] for row in long if row["lam"] > 1.0 and row["blew_up"]],
        "violations_that_did_not": [
            row["lam"] for row in long if row["lam"] > 1.0 and not row["blew_up"]],
        "the_seed_is_rounding": all(
            row["seed"] <= 4.0 * float(np.finfo(float).eps) for row in short),
        "a_longer_run_finds_more_of_them": (
            len([row for row in long if row["lam"] > 1.0 and row["blew_up"]])
            > len([row for row in short if row["lam"] > 1.0 and row["blew_up"]])),
        "worst_stable_error": max(row["error"] for row in rows if row["lam"] <= 1.0),
    }


def the_implicit_threshold_is_a_quarter(thetas=(0.0, 0.1, 0.24, 0.25, 0.3, 0.5),
                                        lams=(0.5, 1.0, 2.0, 10.0, 100.0)):
    """Measure where the weighted scheme becomes unconditionally stable.

    The condition is ``theta >= 1/4``, and it is not obvious from the formula. Sweeping ``theta``
    against ``lam`` and reading off the largest root turns it into a table, and the boundary lands
    exactly between ``0.24`` and ``0.25``.
    """
    rows = []
    for theta in thetas:
        entry = {"theta": float(theta), "roots": {}}
        for lam in lams:
            entry["roots"][float(lam)] = largest_root(float(lam), float(theta))
        entry["stable_everywhere"] = all(
            value <= 1.0 + 1e-9 for value in entry["roots"].values())
        entry["largest_stable_lam"] = max(
            (lam for lam, value in entry["roots"].items() if value <= 1.0 + 1e-9),
            default=float("nan"))
        rows.append(entry)
    return {
        "rows": rows,
        "unconditional_above_a_quarter": all(
            row["stable_everywhere"] == (row["theta"] >= 0.25 - 1e-12) for row in rows),
        "threshold": 0.25,
        "conditional_ones": [row["theta"] for row in rows if not row["stable_everywhere"]],
    }


# --------------------------------------------------------------------------- the magic step


def the_magic_step(lams=(0.5, 0.8, 0.95, 1.0), points: int = 201, t_end: float = 0.4):
    """At ``lam = 1`` the explicit scheme is exact, not merely accurate.

    Put ``lam = 1`` into the update and the ``u_j^n`` terms cancel:

        u_j^{n+1} = u_{j+1}^n + u_{j-1}^n - u_j^{n-1} .

    That is d'Alembert's formula on the grid. The characteristics ``x +/- ct`` pass exactly through
    grid points, and the scheme follows them without approximating anything. The measured error is
    at the **rounding** level while the neighbouring Courant numbers sit near ``1e-3``.

    It is a real property and an unusable one in general: it needs a constant speed, a uniform
    grid and ``k = h/c`` exactly, and any of those failing brings the ordinary second order back.
    """
    problem = travelling_bump()
    rows = []
    for value in lams:
        lam = float(value)
        h = (problem["b"] - problem["a"]) / (int(points) - 1)
        k = lam * h / problem["speed"]
        steps = max(int(round(float(t_end) / k)), 2)
        run = solve(problem, points, steps, steps * k, theta=0.0)
        rows.append({"lam": lam, "steps": steps, "error": run["error"],
                     "h": run["h"], "k": run["k"]})
    exact_row = min(rows, key=lambda row: abs(row["lam"] - 1.0))
    others = [row for row in rows if row is not exact_row]
    return {
        "rows": rows,
        "error_at_one": exact_row["error"],
        "smallest_error_elsewhere": min(row["error"] for row in others),
        "gain": float(min(row["error"] for row in others) / exact_row["error"]),
        "exact_to_rounding": exact_row["error"] < 1e-12,
        "note": "at lam = 1 the update is d'Alembert's formula on the grid, so it is exact",
    }


def changing_data_it_cannot_see(points: int = 401, steps: int = 25, lam: float = 1.6,
                                bump_offset: float = 1.0):
    """The sharpest statement of what a Courant violation is: missing information.

    The exact solution at ``(x, t)`` depends on the initial data over ``[x - ct, x + ct]``. The
    scheme at ``(x_j, t_n)`` depends on the data over ``[x_j - nh, x_j + nh]``, because that is
    how far a three point stencil reaches in ``n`` steps. Containing the true interval needs
    ``nh >= c n k``, which is ``lam <= 1``.

    With ``lam > 1`` there is a region inside the true domain and outside the numerical one. Put a
    bump there and run twice, once with it and once without. The true answer at the target point
    **changes**; the computed answer changes by an **exact zero**, because no arithmetic in the
    scheme ever touched those values.

    That is a different failure from instability. An unstable scheme computes the wrong answer; a
    scheme that violates the Courant condition is not computing an answer to this problem at all.
    """
    speed = 1.0
    a, b = 0.0, 4.0
    n = int(points)
    x = np.linspace(a, b, n)
    h = float(x[1] - x[0])
    k = float(lam) * h / speed
    target = n // 2
    reach_numerical = steps * h
    reach_true = speed * steps * k
    # bump_offset = 1 puts the bump exactly on the characteristic through the target, where its
    # contribution to the true answer is largest. Anything below 1 moves it inwards, and anything
    # at or below 0 would put it inside the numerical domain, where the scheme would see it.
    offset = float(bump_offset) * (reach_true - reach_numerical) + reach_numerical
    if not reach_numerical < offset <= reach_true:
        raise ValueError("the bump must sit between the two domains of dependence")

    def march(with_bump):
        base = np.exp(-((x - x[target]) / 0.05) ** 2)
        if with_bump:
            base = base + np.exp(-((x - (x[target] + offset)) / 0.02) ** 2)
        old = base.copy()
        now = first_step(old, np.zeros_like(old), lam, k, order=2)
        old[0] = old[-1] = now[0] = now[-1] = 0.0
        for _ in range(int(steps) - 1):
            with np.errstate(over="ignore", invalid="ignore"):
                new = explicit_step(old, now, lam)
            old, now = now, new
        return float(now[target])

    without = march(False)
    with_it = march(True)
    # d'Alembert with zero initial velocity: u(x, t) = (f(x - ct) + f(x + ct)) / 2
    shift = speed * steps * k
    here = float(x[target])

    def profile(point, with_bump):
        value = math.exp(-((point - here) / 0.05) ** 2)
        if with_bump:
            value += math.exp(-((point - (here + offset)) / 0.02) ** 2)
        return value

    truth_without = 0.5 * (profile(here - shift, False) + profile(here + shift, False))
    truth_with = 0.5 * (profile(here - shift, True) + profile(here + shift, True))
    return {
        "lam": float(lam),
        "h": h, "k": k, "steps": int(steps),
        "numerical_reach": reach_numerical,
        "true_reach": reach_true,
        "bump_at": offset,
        "bump_is_inside_the_true_domain": offset <= reach_true,
        "bump_is_outside_the_numerical_domain": offset > reach_numerical,
        "the_run_is_also_unstable": abs(without) > 10.0,
        "computed_without": without,
        "computed_with": with_it,
        "computed_change": abs(with_it - without),
        "true_change": abs(truth_with - truth_without),
        "the_scheme_did_not_notice": abs(with_it - without) == 0.0,
        "the_truth_did": abs(truth_with - truth_without) > 1e-6,
        "note": "the computed change is an exact zero: no arithmetic in the scheme ever read "
                "those values. The run is unstable as well, so its answer is nonsense, and the "
                "nonsense is bit for bit identical with and without the bump",
    }


# --------------------------------------------------------------------------- accuracy


def the_first_step_sets_the_order(refinements=(21, 41, 81, 161), t_end: float = 0.4,
                                  lam: float = 0.8):
    """A first order starting step costs the whole run an order, however good the interior is.

    The three level recurrence needs two levels to begin. Taking ``u^1 = u^0 + k g`` is the obvious
    choice and is first order; the interior scheme is second order and cannot recover what the
    first step threw away. Adding ``(lam**2/2) delta**2 u^0``, which Taylor's theorem asks for and
    the equation supplies, restores it.

    This is the same finding as lesson 71's starting values for multistep methods and lesson 77's
    Neumann boundary: **one row, once, sets the order of everything.**

    The standing wave is used rather than the bump, because it has zero initial velocity **and** a
    nonzero second derivative, so the correction term is not zero and the two starts differ.
    """
    problem = standing_wave()
    rows = []
    for order in (1, 2):
        errors, hs = [], []
        for n in refinements:
            m = int(n)
            h = (problem["b"] - problem["a"]) / (m - 1)
            k = float(lam) * h / problem["speed"]
            steps = max(int(round(float(t_end) / k)), 2)
            run = solve(problem, m, steps, steps * k, theta=0.0, start_order=order)
            errors.append(run["error"])
            hs.append(h)
        errors, hs = np.asarray(errors), np.asarray(hs)
        rows.append({
            "start_order": order,
            "h": hs, "error": errors,
            "order": float(np.polyfit(np.log(hs), np.log(errors), 1)[0]),
        })
    first, second = rows[0], rows[1]
    return {
        "rows": rows,
        "first_order_start": first["order"],
        "second_order_start": second["order"],
        "one_row_sets_the_order": (abs(first["order"] - 1.0) < 0.15
                                   and abs(second["order"] - 2.0) < 0.15),
        "gain_at_the_finest": float(first["error"][-1] / second["error"][-1]),
    }


def numerical_wave_speed(lam: float, phase):
    """The speed at which the explicit scheme actually propagates a mode of phase ``phi``.

    The scheme's roots are ``exp(+/- i omega k)`` when stable, with

        sin(omega k / 2) = lam sin(phi / 2) ,

    so the numerical speed is ``omega / (phi / h) = (2 / (lam phi)) arcsin(lam sin(phi/2))``
    times ``c``. At ``lam = 1`` that is exactly 1 for every ``phi``, which is the magic step again.
    For ``lam < 1`` short waves travel **slower** than they should, which is numerical dispersion.
    """
    phi = np.atleast_1d(np.asarray(phase, dtype=float))
    l = float(lam)
    inner = np.clip(l * np.sin(phi / 2.0), -1.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 2.0 * np.arcsin(inner) / (l * phi)
    out = np.where(phi == 0.0, 1.0, out)
    return out


def dispersion_is_worst_for_short_waves(lams=(0.25, 0.5, 0.8, 1.0), samples: int = 200):
    """Measure the relative wave speed error across the whole spectrum, at several Courant numbers.

    Two things come out. The speed error grows with the wavenumber at every ``lam``, so a scheme
    that resolves the long waves well can still put the short ones in the wrong place; and at
    ``lam = 1`` it is **zero at every wavenumber**, which is the magic step stated in the
    frequency domain.
    """
    phases = np.linspace(1e-6, math.pi, int(samples))
    rows = []
    for value in lams:
        lam = float(value)
        speeds = numerical_wave_speed(lam, phases)
        rows.append({
            "lam": lam,
            "phase": phases,
            "speed": speeds,
            "worst_relative_error": float(np.max(np.abs(speeds - 1.0))),
            "error_at_the_shortest": float(abs(speeds[-1] - 1.0)),
            "error_at_ten_points_per_wave": float(
                abs(float(numerical_wave_speed(lam, 2.0 * math.pi / 10.0)[0]) - 1.0)),
            "always_too_slow": bool(np.all(speeds <= 1.0 + 1e-12)),
        })
    exact = min(rows, key=lambda row: abs(row["lam"] - 1.0))
    return {
        "rows": rows,
        "exact_at_lam_one": exact["worst_relative_error"] < 1e-12,
        "every_scheme_is_too_slow": all(row["always_too_slow"] for row in rows),
        "worst_grows_as_lam_falls": all(
            rows[i]["worst_relative_error"] >= rows[i + 1]["worst_relative_error"] - 1e-12
            for i in range(len(rows) - 1)),
        "note": "short waves travel too slowly at every Courant number below 1, and exactly "
                "right at 1",
    }


def the_implicit_scheme_buys_stability_and_pays_in_phase(
        lams=(0.5, 1.0, 2.0, 4.0, 8.0), points: int = 201, periods: float = 1.0):
    """The implicit scheme runs at any Courant number, and the answer stops being right.

    Unconditional stability removes the step size limit and does not remove the accuracy limit.
    Running one full period of a standing wave at increasing ``lam`` keeps every run bounded, and
    the error grows steadily, because the phase error per step grows with the step.

    This is the hyperbolic version of lesson 78's cost argument, and it comes out the other way:
    for a parabolic problem the extra step size is worth having because the solution is decaying
    and the error decays with it; for a wave problem nothing decays, phase errors accumulate, and
    a huge stable step buys nothing at all.
    """
    problem = standing_wave()
    t_end = float(periods) * problem["period"]
    rows = []
    for value in lams:
        lam = float(value)
        h = (problem["b"] - problem["a"]) / (int(points) - 1)
        k = lam * h / problem["speed"]
        steps = max(int(round(t_end / k)), 2)
        run = solve(problem, points, steps, steps * k, theta=0.25)
        rows.append({
            "lam": lam,
            "steps": steps,
            "error": run["error"],
            "finite": run["finite"],
            "largest_root": largest_root(lam, 0.25),
        })
    return {
        "rows": rows,
        "every_run_is_stable": all(row["finite"] for row in rows),
        "every_root_is_on_the_circle": all(
            abs(row["largest_root"] - 1.0) < 1e-9 for row in rows),
        "the_error_grows_with_the_step": all(
            rows[i]["error"] <= rows[i + 1]["error"] for i in range(len(rows) - 1)),
        "error_span": float(rows[-1]["error"] / rows[0]["error"]),
        "worst_error": max(row["error"] for row in rows),
        "note": "stability removed the step limit and not the accuracy limit; nothing decays in "
                "a wave problem, so phase error accumulates",
    }
