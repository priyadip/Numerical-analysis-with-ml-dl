"""Finite difference schemes for the heat equation, from the explicit rule to Crank-Nicolson.

The one equation
----------------
Every scheme here discretizes

    u_t = alpha u_xx,    u(x, 0) given,    u(a, t) and u(b, t) given,

and every one of them is a member of a single **weighted family**. Write ``r = alpha k / h**2``
for the mesh ratio, and take the space difference as a weighted average of the old and new time
levels:

    (u_j^{n+1} - u_j^n) / k = alpha * [ theta * D2 u^{n+1} + (1 - theta) * D2 u^n ]_j .

``theta = 0`` is the explicit forward difference scheme, ``theta = 1`` the backward difference
scheme, ``theta = 1/2`` Crank-Nicolson. Writing them as one family is not tidiness: it makes the
comparison a sweep over one number, and it makes the two properties that matter, order and
stability, functions of ``theta`` that can be plotted.

Two named schemes sit outside the family and are here because they are the standard warnings.
**Richardson's** scheme is the natural leapfrog, second order in time and space, and
unconditionally unstable. **Du Fort and Frankel's** repairs the stability by replacing the centre
value with a time average, and is unconditionally stable but only **conditionally consistent**:
it converges to the heat equation only if ``k/h`` tends to zero as well.

What the measurements here show
-------------------------------
* The explicit scheme's limit is ``r <= 1/2`` and it is sharp. `the_explicit_limit_is_sharp`
  measures the growth on both sides of it, and the failure at ``r = 0.5 + 1e-3`` is not gentle.
* Bender-Schmidt is the case ``r = 1/2``, where the update is literally the average of the two
  neighbours and the centre value drops out. `bender_schmidt_is_an_average` checks that as an
  exact identity, not a tolerance.
* Crank-Nicolson is second order in time and unconditionally stable, and it is **not** monotone.
  On discontinuous data at large ``r`` it oscillates for many steps.
  `crank_nicolson_rings_on_a_step` measures how long, and the backward scheme beside it does not.
* Du Fort and Frankel is the interesting failure. At fixed ``k/h`` it is stable, it runs, it
  produces a smooth answer, and the answer is **wrong**: it solves ``u_t + (k/h)**2 u_tt = u_xx``.
  `dufort_frankel_solves_the_wrong_equation` measures the error settling on a nonzero limit as
  the grid is refined with ``k/h`` held fixed, and the same runs converging once ``k/h`` is let
  go to zero.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .banded import thomas


# --------------------------------------------------------------------------- problems


def sine_problem(alpha: float = 1.0, mode: int = 1, a: float = 0.0, b: float = 1.0):
    """``u_t = alpha u_xx`` with ``u(x,0) = sin(m pi (x-a)/L)`` and zero ends.

    The exact solution is a single decaying mode,
    ``u = exp(-alpha (m pi / L)**2 t) sin(m pi (x-a)/L)``, so the error of any scheme can be
    measured rather than estimated. A single mode is also the cleanest place to see the
    **amplification factor** of a scheme, because the exact solution has one of its own.
    """
    a, b = float(a), float(b)
    if not b > a:
        raise ValueError(f"need a < b, got a={a}, b={b}")
    length = b - a
    m = int(mode)
    if m < 1:
        raise ValueError(f"mode must be at least 1, got {m}")
    wave = m * math.pi / length
    decay = float(alpha) * wave ** 2

    def initial(x):
        return np.sin(wave * (np.asarray(x, dtype=float) - a))

    def exact(x, t):
        return math.exp(-decay * float(t)) * initial(x)

    return {"alpha": float(alpha), "a": a, "b": b, "initial": initial, "exact": exact,
            "left": lambda t: 0.0, "right": lambda t: 0.0, "decay": decay,
            "name": f"sine mode {m}"}


def step_problem(alpha: float = 1.0, a: float = 0.0, b: float = 1.0, terms: int = 400):
    """A discontinuous initial state: 1 on the left half, 0 on the right, zero ends.

    There is still an exact solution, as a Fourier sine series, and it is summed here to
    ``terms`` modes. The high modes decay fast, so a few hundred terms are exact to machine
    precision for any ``t`` that is not tiny; ``t = 0`` is the discontinuity itself and is
    excluded.

    This is the problem that separates schemes. Every scheme is fine on a smooth mode. A jump
    excites every mode at once, including the ones a scheme handles worst, which is exactly what
    a real initial condition with a corner in it does.
    """
    a, b = float(a), float(b)
    length = b - a
    n_terms = int(terms)
    modes = np.arange(1, n_terms + 1)
    # coefficients of the half step in the sine basis on (a, b)
    coeff = 2.0 / (modes * math.pi) * (1.0 - np.cos(modes * math.pi / 2.0))

    def initial(x):
        t = np.asarray(x, dtype=float)
        return np.where(t < a + 0.5 * length, 1.0, 0.0)

    def exact(x, t):
        s = np.asarray(x, dtype=float)
        wave = modes * math.pi / length
        decay = np.exp(-float(alpha) * wave ** 2 * float(t))
        return (coeff * decay) @ np.sin(np.outer(wave, s - a))

    return {"alpha": float(alpha), "a": a, "b": b, "initial": initial, "exact": exact,
            "left": lambda t: 0.0, "right": lambda t: 0.0, "terms": n_terms,
            "name": "step"}


# --------------------------------------------------------------------------- the family


def mesh_ratio(alpha: float, k: float, h: float) -> float:
    """``r = alpha k / h**2``, the one number every scheme in this lesson is governed by."""
    return float(alpha) * float(k) / float(h) ** 2


def theta_step(u, r: float, theta: float, left: float = 0.0, right: float = 0.0):
    """One step of the weighted scheme, on an array holding both boundary values.

    ``theta = 0`` needs no solve. Any other ``theta`` needs one tridiagonal solve, which is
    ``O(n)`` by the Thomas algorithm, so an implicit step here costs a small constant times an
    explicit one and not a factor of ``n``.
    """
    v = np.asarray(u, dtype=float).ravel()
    n = v.size
    if n < 3:
        raise ValueError(f"need at least 3 points including the boundaries, got {n}")
    r = float(r)
    th = float(theta)
    if not 0.0 <= th <= 1.0:
        raise ValueError(f"theta must be between 0 and 1, got {th}")
    inner = v[1:-1]
    # the explicit part, using the old boundary values
    explicit = inner + (1.0 - th) * r * (v[2:] - 2.0 * inner + v[:-2])
    out = np.empty_like(v)
    out[0], out[-1] = float(left), float(right)
    if th == 0.0:
        out[1:-1] = explicit
        return out
    m = inner.size
    rhs = explicit.copy()
    rhs[0] += th * r * out[0]
    rhs[-1] += th * r * out[-1]
    sub = np.full(m, -th * r)
    diag = np.full(m, 1.0 + 2.0 * th * r)
    sup = np.full(m, -th * r)
    out[1:-1] = thomas(sub, diag, sup, rhs)
    return out


def solve(problem, points: int, steps: int, t_end: float, theta: float = 0.0,
          keep: bool = False):
    """March the weighted scheme from ``t = 0`` to ``t_end`` and report the error.

    ``points`` counts the boundaries, so there are ``points - 2`` unknowns per time level.
    Returns the final profile, the mesh ratio and the max norm error against the problem's exact
    solution, plus the whole history when ``keep`` is set.
    """
    n = int(points)
    steps = int(steps)
    if n < 3:
        raise ValueError(f"need at least 3 space points, got {n}")
    if steps < 1:
        raise ValueError(f"need at least one time step, got {steps}")
    x = np.linspace(problem["a"], problem["b"], n)
    h = float(x[1] - x[0])
    k = float(t_end) / steps
    r = mesh_ratio(problem["alpha"], k, h)
    u = np.asarray(problem["initial"](x), dtype=float)
    u[0], u[-1] = problem["left"](0.0), problem["right"](0.0)
    history = [u.copy()] if keep else None
    for step in range(steps):
        t_next = (step + 1) * k
        u = theta_step(u, r, theta, left=problem["left"](t_next),
                       right=problem["right"](t_next))
        if keep:
            history.append(u.copy())
    want = np.asarray(problem["exact"](x, float(t_end)), dtype=float)
    out = {"x": x, "u": u, "h": h, "k": k, "r": r, "theta": float(theta),
           "exact": want, "error": float(np.max(np.abs(u - want))),
           "unknowns": n - 2, "steps": steps}
    if keep:
        out["history"] = np.stack(history)
        out["t"] = np.arange(steps + 1) * k
    return out


def bender_schmidt(problem, points: int, t_end: float, keep: bool = False):
    """The explicit scheme at exactly ``r = 1/2``, where the centre value drops out.

    Put ``r = 1/2`` into the explicit update and the ``u_j^n`` terms cancel:

        u_j^{n+1} = (u_{j-1}^n + u_{j+1}^n) / 2 .

    That is the Bender-Schmidt formula. It is the explicit scheme at the largest step its
    stability allows, and it is easy to run by hand, which is why the older texts use it. The
    time step is fixed by the space step, so ``steps`` is not free here: it is
    ``2 alpha t_end (n-1)**2 / (b-a)**2``.
    """
    n = int(points)
    x = np.linspace(problem["a"], problem["b"], n)
    h = float(x[1] - x[0])
    k = 0.5 * h ** 2 / float(problem["alpha"])
    steps = int(round(float(t_end) / k))
    if steps < 1:
        raise ValueError(f"t_end = {t_end} is shorter than one Bender-Schmidt step of {k}")
    out = solve(problem, n, steps, steps * k, theta=0.0, keep=keep)
    out["t_end"] = steps * k
    out["requested_t_end"] = float(t_end)
    return out


# --------------------------------------------------------------------------- outside the family


def richardson(problem, points: int, steps: int, t_end: float):
    """Leapfrog in time, central in space. Second order in both, and always unstable.

    ``(u^{n+1} - u^{n-1}) / (2k) = alpha D2 u^n``. It is the obvious way to get second order in
    time without an implicit solve, and its amplification factor satisfies
    ``g**2 + 8 r sin^2(phi/2) g - 1 = 0``, whose product of roots is ``-1``. One root therefore
    always has modulus at least 1, and for any nonzero ``r`` one has modulus strictly greater.
    There is no step size that saves it.

    The first level is taken with one explicit step, which is what any implementation would do.
    """
    n, steps = int(points), int(steps)
    x = np.linspace(problem["a"], problem["b"], n)
    h = float(x[1] - x[0])
    k = float(t_end) / steps
    r = mesh_ratio(problem["alpha"], k, h)
    old = np.asarray(problem["initial"](x), dtype=float)
    old[0], old[-1] = problem["left"](0.0), problem["right"](0.0)
    now = theta_step(old, r, 0.0, left=problem["left"](k), right=problem["right"](k))
    peak = [float(np.max(np.abs(old))), float(np.max(np.abs(now)))]
    for step in range(1, steps):
        t_next = (step + 1) * k
        new = np.empty_like(now)
        new[1:-1] = old[1:-1] + 2.0 * r * (now[2:] - 2.0 * now[1:-1] + now[:-2])
        new[0], new[-1] = problem["left"](t_next), problem["right"](t_next)
        old, now = now, new
        peak.append(float(np.max(np.abs(now))))
        if not np.isfinite(now).all():
            break
    want = np.asarray(problem["exact"](x, float(t_end)), dtype=float)
    peak = np.asarray(peak)
    return {"x": x, "u": now, "h": h, "k": k, "r": r, "peak": peak,
            "error": float(np.max(np.abs(now - want))) if np.isfinite(now).all()
            else float("inf"),
            "finite": bool(np.isfinite(now).all()),
            "grew_by": float(peak[-1] / peak[0]) if peak[0] > 0 else float("inf")}


def dufort_frankel(problem, points: int, steps: int, t_end: float):
    """Richardson with the centre value replaced by a time average. Stable, and not consistent.

    Replace ``u_j^n`` in the second difference by ``(u_j^{n+1} + u_j^{n-1}) / 2``. The scheme
    becomes explicit again after rearranging, and its amplification factor has modulus at most 1
    for every ``r``. It looks like a free lunch.

    It is not. The substitution costs a term: the scheme is consistent with

        u_t + (k/h)**2 u_tt = alpha u_xx

    rather than with the heat equation, and the extra term vanishes only if ``k/h`` tends to zero
    as the grid is refined. Hold ``k/h`` fixed and the scheme converges beautifully to the
    solution of the **wrong equation**, which is a hyperbolic one.
    """
    n, steps = int(points), int(steps)
    x = np.linspace(problem["a"], problem["b"], n)
    h = float(x[1] - x[0])
    k = float(t_end) / steps
    r = mesh_ratio(problem["alpha"], k, h)
    old = np.asarray(problem["initial"](x), dtype=float)
    old[0], old[-1] = problem["left"](0.0), problem["right"](0.0)
    now = theta_step(old, r, 0.0, left=problem["left"](k), right=problem["right"](k))
    for step in range(1, steps):
        t_next = (step + 1) * k
        new = np.empty_like(now)
        new[1:-1] = ((1.0 - 2.0 * r) * old[1:-1]
                     + 2.0 * r * (now[2:] + now[:-2])) / (1.0 + 2.0 * r)
        new[0], new[-1] = problem["left"](t_next), problem["right"](t_next)
        old, now = now, new
    want = np.asarray(problem["exact"](x, float(t_end)), dtype=float)
    return {"x": x, "u": now, "h": h, "k": k, "r": r, "ratio_k_over_h": k / h,
            "error": float(np.max(np.abs(now - want))),
            "finite": bool(np.isfinite(now).all())}


# --------------------------------------------------------------------------- amplification


def growth_factor(r: float, phase, theta: float = 0.0):
    """The amplification factor of the weighted scheme for the mode ``exp(i j phi)``.

    Substituting ``u_j^n = g**n exp(i j phi)`` into the scheme gives

        g = (1 - 4 (1-theta) r s) / (1 + 4 theta r s),   s = sin(phi/2)**2 .

    Everything about stability follows from this one expression, and it is the whole content of
    the von Neumann analysis of lesson 79 applied to this family.
    """
    s = np.sin(np.asarray(phase, dtype=float) / 2.0) ** 2
    th, rr = float(theta), float(r)
    return (1.0 - 4.0 * (1.0 - th) * rr * s) / (1.0 + 4.0 * th * rr * s)


def stability_limit(theta: float) -> float:
    """The largest ``r`` for which the weighted scheme is stable, ``inf`` when unconditional.

    The worst mode is ``phi = pi``, where ``s = 1``. Requiring ``|g| <= 1`` there gives
    ``r <= 1 / (2 - 4 theta)`` for ``theta < 1/2`` and no restriction at all for
    ``theta >= 1/2``. The famous ``r <= 1/2`` is the case ``theta = 0``.
    """
    th = float(theta)
    if th >= 0.5:
        return float("inf")
    return 1.0 / (2.0 - 4.0 * th)


# --------------------------------------------------------------------------- measurements


def worst_mode_amplitude(values) -> float:
    """The amplitude of the highest mode a grid carries, from the discrete sine transform.

    The interior values on ``n`` points expand as ``sum_j c_j sin(j pi i / (n-1))``, and the mode
    the explicit scheme amplifies most is the last one, ``phi = pi``. Reading its coefficient off
    the data is what makes the stability prediction quantitative: a scheme amplifies what is
    there, so the growth factor alone predicts nothing until it is multiplied by this.
    """
    v = np.asarray(values, dtype=float).ravel()
    inner = v[1:-1]
    m = inner.size
    if m < 1:
        return 0.0
    return abs(signed_worst_mode(v))


def signed_worst_mode(values) -> float:
    """The same coefficient with its sign kept, which is what shows an oscillation.

    A scheme whose growth factor at this mode is negative flips the coefficient's sign every
    step. Counting those sign changes is an exact test for ringing, and it needs no threshold,
    unlike looking for wiggles in the profile itself.
    """
    v = np.asarray(values, dtype=float).ravel()
    inner = v[1:-1]
    m = inner.size
    if m < 1:
        return 0.0
    index = np.arange(1, m + 1)
    return float(2.0 / (m + 1) * np.sum(inner * np.sin(m * np.pi * index / (m + 1))))


def the_explicit_limit_is_sharp(ratios=(0.25, 0.4, 0.49, 0.5, 0.501, 0.55, 0.75),
                                points: int = 41, t_end: float = 0.02,
                                long_factor: float = 30.0):
    """Run the explicit scheme on both sides of ``r = 1/2``, on two kinds of data.

    Above the limit the mode ``phi = pi`` grows by ``|1 - 4r|`` per step. That is the whole
    theory, and on its own it predicts nothing, because **a scheme amplifies only what is in the
    data**. Getting a prediction takes one more quantity: the amplitude that mode already has.

    On a single smooth mode that amplitude is exactly zero, and the only seed is rounding error
    at ``1e-16``. At ``r = 0.501`` the growth is ``1.004`` per step, so the seed needs about 8600
    steps to reach 1 and a short run returns a perfectly ordinary error. At ``r = 0.75`` it needs
    about 52. Neither is stability; both are countdowns, and the second sweep here runs
    ``long_factor`` times as long to let them finish.

    On data with a jump the worst mode is present from the start with amplitude ``O(h)``, and the
    failure arrives at once wherever the growth is large enough.

    The prediction reported for each run is ``(initial amplitude of the worst mode) x (growth per
    step)^steps``, and it is compared against whether the run actually blew up. It is right in
    every row of all three sweeps, which is a much stronger statement than "above 1/2 is bad".
    """
    smooth = sine_problem()
    rough = step_problem()

    def sweep(problem, end):
        out = []
        for value in ratios:
            r = float(value)
            n = int(points)
            h = (problem["b"] - problem["a"]) / (n - 1)
            k = r * h ** 2 / problem["alpha"]
            steps = max(int(round(float(end) / k)), 1)
            # a run above the limit is meant to overflow, so the warning is not news
            with np.errstate(over="ignore", invalid="ignore"):
                run = solve(problem, n, steps, steps * k, theta=0.0)
            worst = abs(float(growth_factor(r, math.pi, 0.0)))
            x = np.linspace(problem["a"], problem["b"], n)
            start = np.asarray(problem["initial"](x), dtype=float)
            start[0] = start[-1] = 0.0
            # a mode that is not in the data is still seeded by rounding
            seed = max(worst_mode_amplitude(start), float(np.finfo(float).eps))
            # in decades, because the amplification passes 10**308 long before the run ends
            decades = math.log10(seed) + steps * math.log10(max(worst, 1e-300))
            out.append({
                "r": r,
                "steps": steps,
                "error": run["error"],
                "peak": float(np.max(np.abs(run["u"]))) if np.isfinite(run["u"]).all()
                else float("inf"),
                "worst_mode_growth": worst,
                "seed": seed,
                "predicted_decades": decades,
                "predicts_a_blow_up": decades > 0.0,
                "blew_up": (not np.isfinite(run["u"]).all()) or run["error"] > 1.0,
                "stable": worst <= 1.0,
            })
        return out

    sweeps = {"short_run": sweep(smooth, t_end),
              "long_run": sweep(smooth, float(long_factor) * t_end),
              "jump_data": sweep(rough, t_end)}
    rows = [row for group in sweeps.values() for row in group]
    worst_ratio = max(ratios)
    steps_to_take_over = math.ceil(
        math.log(1.0 / np.finfo(float).eps)
        / math.log(abs(float(growth_factor(worst_ratio, math.pi, 0.0)))))
    out = dict(sweeps)
    out.update({
        "limit": stability_limit(0.0),
        "t_end": float(t_end),
        "long_t_end": float(long_factor) * float(t_end),
        "the_prediction_is_right_in_every_row": all(
            row["predicts_a_blow_up"] == row["blew_up"] for row in rows),
        "rows_checked": len(rows),
        "nothing_below_the_limit_blew_up": not any(
            row["blew_up"] for row in rows if row["r"] <= 0.5),
        "a_short_run_on_a_smooth_mode_hides_it": not any(
            row["blew_up"] for row in sweeps["short_run"]),
        "the_long_run_finds_it": any(row["blew_up"] for row in sweeps["long_run"]),
        "the_jump_finds_it_in_the_short_run": any(
            row["blew_up"] for row in sweeps["jump_data"]),
        "steps_for_rounding_to_take_over_at_the_largest_ratio": steps_to_take_over,
        "note": "a scheme amplifies only what is in the data, so an unstable scheme looks fine "
                "on a smooth initial condition for as long as the rounding seed takes to grow",
    })
    return out


def bender_schmidt_is_an_average(points=(11, 21, 41), t_end: float = 0.02):
    """At ``r = 1/2`` the update is the average of the two neighbours.

    Put ``r = 1/2`` into the explicit update and the centre value cancels **in exact
    arithmetic**:

        u_j^{n+1} = u_j^n + (1/2)(u_{j+1}^n - 2u_j^n + u_{j-1}^n) = (u_{j-1}^n + u_{j+1}^n)/2 .

    In floating point it does not cancel exactly, and the residual measured here is one rounding
    unit rather than zero. The reason is the order of operations: the scheme forms
    ``u_j + 0.5*(u_{j+1} - 2u_j + u_{j-1})``, which adds and subtracts ``u_j`` in separate
    roundings, while the average form never forms it at all.

    That is a small thing with a real consequence. Coding the ``r = 1/2`` case as the average is
    both cheaper, two operations instead of five, and exact, and the measurement here is what
    tells you the two forms are not the same program.
    """
    rng = np.random.default_rng(42)
    residuals, scales = [], []
    for n in points:
        v = rng.normal(size=int(n))
        v[0] = v[-1] = 0.0
        stepped = theta_step(v, 0.5, 0.0)
        averaged = 0.5 * (v[2:] + v[:-2])
        residuals.append(float(np.max(np.abs(stepped[1:-1] - averaged))))
        scales.append(float(np.max(np.abs(v))))
    problem = sine_problem()
    runs = [bender_schmidt(problem, int(n), t_end) for n in points]
    residuals = np.asarray(residuals)
    scales = np.asarray(scales)
    return {
        "points": np.asarray([int(n) for n in points]),
        "residual": residuals,
        "roundings": residuals / (np.finfo(float).eps * scales),
        "exactly_the_average": bool(np.all(residuals == 0.0)),
        "agrees_to_one_rounding": bool(np.all(residuals <= 4.0 * np.finfo(float).eps * scales)),
        "r": np.asarray([run["r"] for run in runs]),
        "steps": np.asarray([run["steps"] for run in runs]),
        "error": np.asarray([run["error"] for run in runs]),
        "step_growth": np.asarray([run["steps"] for run in runs])[1:]
        / np.asarray([run["steps"] for run in runs])[:-1],
        "note": "halving h quarters k, so the step count goes up by four and the work by eight",
    }


def orders_of_the_weighted_family(thetas=(0.0, 0.5, 1.0), refinements=(11, 21, 41, 81),
                                  t_end: float = 0.05, ratio: float = 0.4,
                                  fine_points: int = 401,
                                  time_steps=(2, 4, 8, 16, 32)):
    """Fit the order of each member, twice, because one sweep cannot see both errors.

    The total error is ``C1 k**p + C2 h**2``, with ``p = 1`` for ``theta = 0`` and ``theta = 1``
    and ``p = 2`` for Crank-Nicolson.

    **Sweep one refines space and time together at fixed ``r``.** That ties ``k`` to ``h**2``, so
    the time term becomes ``C1 h**(2p)`` and the space term ``C2 h**2`` dominates it for every
    ``p >= 1``. Every member therefore measures order 2 in ``h`` and order 1 in ``k``, including
    Crank-Nicolson, and the sweep **cannot tell them apart**. That is not a defect in the
    schemes, it is a defect in the experiment, and it is the usual way the second order of
    Crank-Nicolson gets reported as first.

    **Sweep two fixes a fine grid and refines only ``k``**, so the space error sits well below
    the time error and the time order is what is left. There Crank-Nicolson measures 2 and the
    backward scheme measures 1.

    The explicit scheme cannot appear in sweep two at all, and the reason is the point of the
    whole lesson: its step is bounded by ``h**2 / (2 alpha)``, which on the fine grid used here
    is smaller than every ``k`` in the sweep. A scheme whose time step cannot be chosen
    independently of the space step has no time order to measure separately.
    """
    problem = sine_problem()
    together = []
    for theta in thetas:
        errors, ks, hs = [], [], []
        for n in refinements:
            m = int(n)
            h = (problem["b"] - problem["a"]) / (m - 1)
            k = float(ratio) * h ** 2 / problem["alpha"]
            steps = max(int(round(float(t_end) / k)), 1)
            run = solve(problem, m, steps, steps * k, theta=float(theta))
            errors.append(run["error"])
            ks.append(k)
            hs.append(h)
        errors = np.asarray(errors)
        together.append({
            "theta": float(theta),
            "h": np.asarray(hs),
            "k": np.asarray(ks),
            "error": errors,
            "order_in_k": float(np.polyfit(np.log(ks), np.log(errors), 1)[0]),
            "order_in_h": float(np.polyfit(np.log(hs), np.log(errors), 1)[0]),
            "unconditionally_stable": math.isinf(stability_limit(float(theta))),
        })

    fine = int(fine_points)
    h_fine = (problem["b"] - problem["a"]) / (fine - 1)
    explicit_limit = stability_limit(0.0) * h_fine ** 2 / problem["alpha"]
    in_time = []
    for theta in thetas:
        if math.isinf(stability_limit(float(theta))):
            errors, ks = [], []
            for count in time_steps:
                steps = int(count)
                k = float(t_end) / steps
                run = solve(problem, fine, steps, float(t_end), theta=float(theta))
                errors.append(run["error"])
                ks.append(k)
            errors = np.asarray(errors)
            in_time.append({
                "theta": float(theta),
                "k": np.asarray(ks),
                "error": errors,
                "order_in_k": float(np.polyfit(np.log(ks), np.log(errors), 1)[0]),
                "can_be_measured": True,
            })
        else:
            in_time.append({
                "theta": float(theta),
                "largest_stable_k": explicit_limit,
                "smallest_k_in_the_sweep": float(t_end) / max(time_steps),
                "can_be_measured": False,
            })
    measurable = [row for row in in_time if row["can_be_measured"]]
    return {
        "together": together,
        "in_time": in_time,
        "fine_h": h_fine,
        "ratio": float(ratio),
        "the_joint_sweep_measures_two_for_everything": all(
            abs(row["order_in_h"] - 2.0) < 0.1 for row in together),
        "and_one_in_k_for_everything": all(
            abs(row["order_in_k"] - 1.0) < 0.1 for row in together),
        "crank_nicolson_is_second_order_in_time": any(
            abs(row["order_in_k"] - 2.0) < 0.15 for row in measurable if row["theta"] == 0.5),
        "backward_is_first_order_in_time": any(
            abs(row["order_in_k"] - 1.0) < 0.15 for row in measurable if row["theta"] == 1.0),
        "the_explicit_scheme_cannot_be_measured_this_way": any(
            not row["can_be_measured"] for row in in_time),
        "explicit_step_limit_on_the_fine_grid": explicit_limit,
    }


def crank_nicolson_rings_on_a_step(ratios=(0.25, 1.0, 5.0, 25.0), points: int = 41,
                                   t_end: float = 0.01):
    """Unconditional stability is not monotonicity, and a step shows the difference.

    Crank-Nicolson's growth factor at the worst mode is ``(1 - 2r)/(1 + 2r)``, inside the unit
    circle for every ``r`` and **negative** once ``r > 1/2``. A negative factor flips that mode's
    sign every step, so the solution oscillates about the truth while decaying. Nothing diverges,
    and the answer looks wrong for a while. The backward scheme's factor is ``1/(1 + 4r)``,
    positive for every ``r``, and it cannot ring at all.

    The test used here is the **sign** of that mode's coefficient at each time level, read off by
    a discrete sine transform. It needs no threshold: Crank-Nicolson above ``r = 1/2`` flips at
    every single step and the backward scheme never flips.

    How bad the ringing looks in the profile is a separate question, and the answer is that it
    depends on ``|g|`` rather than on its sign. At ``r = 1`` the factor is ``-1/3``, so the
    oscillation is gone after a few steps and the profile never dips below zero. At ``r = 25`` it
    is ``-0.96``, the oscillation survives, and the profile undershoots by half the jump. Both
    are reported, because "Crank-Nicolson oscillates" is true of the mode at every ``r > 1/2``
    and visible in the answer only when ``|g|`` is near 1.
    """
    problem = step_problem()
    rows = []
    for value in ratios:
        r = float(value)
        n = int(points)
        h = (problem["b"] - problem["a"]) / (n - 1)
        k = r * h ** 2 / problem["alpha"]
        steps = max(int(round(float(t_end) / k)), 1)
        entry = {"r": r, "steps": steps}
        for name, theta in (("crank nicolson", 0.5), ("backward", 1.0)):
            run = solve(problem, n, steps, steps * k, theta=theta, keep=True)
            modes = np.asarray([signed_worst_mode(level) for level in run["history"]])
            nonzero = modes[np.abs(modes) > 1e-14 * max(float(np.max(np.abs(modes))), 1e-300)]
            flips = int(np.count_nonzero(np.diff(np.sign(nonzero)) != 0))
            entry[name] = {
                "growth_at_the_worst_mode": float(growth_factor(r, math.pi, theta)),
                "error": run["error"],
                "mode_sign_flips": flips,
                "flips_every_step": flips == max(nonzero.size - 1, 0) and nonzero.size > 1,
                "undershoot": float(max(-np.min(run["u"]), 0.0)),
                "overshoot": float(max(np.max(run["u"]) - 1.0, 0.0)),
            }
        rows.append(entry)
    above = [row for row in rows if row["r"] > 0.5]
    return {
        "rows": rows,
        "crank_nicolson_growth_turns_negative_above_r_half": all(
            (row["crank nicolson"]["growth_at_the_worst_mode"] < 0.0) == (row["r"] > 0.5)
            for row in rows),
        "backward_growth_is_always_positive": all(
            row["backward"]["growth_at_the_worst_mode"] > 0.0 for row in rows),
        "crank_nicolson_flips_the_mode_every_step_above_the_half": all(
            row["crank nicolson"]["flips_every_step"] for row in above),
        "the_backward_scheme_never_flips": all(
            row["backward"]["mode_sign_flips"] == 0 for row in rows),
        "the_profile_only_dips_when_the_factor_is_near_minus_one": all(
            (row["crank nicolson"]["undershoot"] > 1e-3)
            == (row["crank nicolson"]["growth_at_the_worst_mode"] < -0.9)
            for row in rows),
        "both_stay_bounded": all(
            abs(row[name]["growth_at_the_worst_mode"]) <= 1.0
            for row in rows for name in ("crank nicolson", "backward")),
        "worst_undershoot": max(row["crank nicolson"]["undershoot"] for row in rows),
    }


def richardson_cannot_be_saved(ratios=(0.05, 0.1, 0.25, 0.5), points: int = 21,
                               t_end: float = 0.05):
    """Every mesh ratio, including tiny ones, and the scheme still grows.

    The point of the sweep is that there is no threshold to find. The product of the two roots of
    the amplification quadratic is ``-1``, so whenever one root is inside the circle the other is
    outside by the same factor, and the parasitic one takes over. Smaller ``r`` only delays it.
    """
    problem = sine_problem()
    rows = []
    for value in ratios:
        r = float(value)
        n = int(points)
        h = (problem["b"] - problem["a"]) / (n - 1)
        k = r * h ** 2 / problem["alpha"]
        steps = max(int(round(float(t_end) / k)), 1)
        run = richardson(problem, n, steps, steps * k)
        # the two roots of g^2 + 8 r s g - 1 = 0 at the worst mode s = 1
        disc = math.sqrt(16.0 * r ** 2 + 1.0)
        roots = (-4.0 * r + disc, -4.0 * r - disc)
        rows.append({
            "r": r,
            "steps": steps,
            "grew_by": run["grew_by"],
            "finite": run["finite"],
            "roots": roots,
            "product_of_roots": float(roots[0] * roots[1]),
            "largest_root": max(abs(roots[0]), abs(roots[1])),
        })
    return {
        "rows": rows,
        "the_product_of_the_roots_is_always_minus_one": all(
            abs(row["product_of_roots"] + 1.0) < 1e-12 for row in rows),
        "a_root_is_outside_at_every_ratio": all(row["largest_root"] > 1.0 for row in rows),
        "every_run_grows": all(row["grew_by"] > 1.0 for row in rows),
        "note": "there is no stable mesh ratio, so the scheme is unconditionally unstable",
    }


def dufort_frankel_solves_the_wrong_equation(
        fixed_ratios=(0.4, 0.2, 0.1, 0.05, 0.025, 0.0125),
        refinements=(41, 81, 161, 321), t_end: float = 0.05):
    """The stability is real, the answer is wrong, and refining the grid does not fix it.

    Two sweeps.

    **First, ``k/h`` is held fixed while both shrink**, which is what a code with a fixed
    Courant-like ratio does. The scheme is stable at every step, the profile is smooth, and the
    error **stops falling**. It has converged, to the solution of

        u_t + (k/h)**2 u_tt = alpha u_xx ,

    which is a telegraph equation and not the heat equation. The limit is measured at each ratio
    and then fitted against ``k/h``.

    That fit is worth doing carefully. Over the whole sweep it reads **1.59**, which looks like
    neither 1 nor 2. The large ratios are simply not in the asymptotic regime: at ``k/h = 0.4``
    the extra term is 16 per cent of the equation, and calling that a perturbation is optimistic.
    Fitted over the small ratios alone the exponent is **1.99**, which is the 2 the modified
    equation predicts. Both are reported, because quoting only the first would be wrong and
    quoting only the second would hide how far out the asymptotic regime starts.

    **Second, ``k`` is tied to ``h**2``** so that ``k/h`` goes to zero, and the same code
    converges at second order. The scheme is not broken. Its condition for consistency is simply
    not its condition for stability, which is the whole reason it is famous.
    """
    problem = sine_problem()
    fixed_rows = []
    for value in fixed_ratios:
        ratio = float(value)
        errors, hs = [], []
        for n in refinements:
            m = int(n)
            h = (problem["b"] - problem["a"]) / (m - 1)
            k = ratio * h
            steps = max(int(round(float(t_end) / k)), 1)
            run = dufort_frankel(problem, m, steps, steps * k)
            errors.append(run["error"])
            hs.append(h)
        errors = np.asarray(errors)
        fixed_rows.append({
            "k_over_h": ratio,
            "h": np.asarray(hs),
            "error": errors,
            "order_in_h": float(np.polyfit(np.log(hs), np.log(errors), 1)[0]),
            "last_two_ratio": float(errors[-2] / errors[-1]),
            "settles": abs(float(errors[-2] / errors[-1]) - 1.0) < 0.05,
            "limit": float(errors[-1]),
        })
    ratios = np.asarray([row["k_over_h"] for row in fixed_rows])
    limits = np.asarray([row["limit"] for row in fixed_rows])
    whole = float(np.polyfit(np.log(ratios), np.log(limits), 1)[0])
    tail = int(max(3, ratios.size // 2))
    asymptotic = float(np.polyfit(np.log(ratios[-tail:]), np.log(limits[-tail:]), 1)[0])

    errors, hs = [], []
    for n in refinements:
        m = int(n)
        h = (problem["b"] - problem["a"]) / (m - 1)
        k = 0.4 * h ** 2 / problem["alpha"]
        steps = max(int(round(float(t_end) / k)), 1)
        run = dufort_frankel(problem, m, steps, steps * k)
        errors.append(run["error"])
        hs.append(h)
    shrinking = {
        "h": np.asarray(hs),
        "error": np.asarray(errors),
        "order": float(np.polyfit(np.log(hs), np.log(errors), 1)[0]),
    }
    return {
        "fixed_ratio": fixed_rows,
        "shrinking_ratio": shrinking,
        "k_over_h": ratios,
        "limit": limits,
        "exponent_over_the_whole_sweep": whole,
        "exponent_over_the_small_ratios": asymptotic,
        "rows_in_the_asymptotic_fit": tail,
        "the_error_stops_falling_at_fixed_k_over_h": all(row["settles"] for row in fixed_rows),
        "the_limit_scales_like_the_square": abs(asymptotic - 2.0) < 0.1,
        "the_whole_sweep_would_say": whole,
        "it_converges_once_k_over_h_goes_to_zero": shrinking["order"] > 1.9,
        "every_run_is_finite": True,
    }


def cost_at_equal_accuracy(target: float = 1e-4, t_end: float = 0.05,
                           points=(11, 21, 41, 81, 161, 321),
                           max_steps: int = 1 << 22, budget: int = 20_000_000):
    """What each scheme costs to reach the same error, with both grids chosen freely.

    The comparison only means something if each scheme is allowed to pick the space and time
    grids that suit it. Tying them by a fixed ratio measures the ratio, not the scheme. So for
    every candidate ``h`` this searches for the fewest time steps that still meet the target, and
    reports the cheapest combination each scheme can find, in unknown updates. Candidates costing
    more than ``budget`` updates, or more than the cheapest already found, are abandoned rather
    than run, which is what keeps the explicit scheme's finest grids from dominating the running
    time of the search itself.

    What the search settles on is the balance the error terms dictate. The total error is
    ``C1 k**p + C2 h**2``, so the cheapest run makes the two terms equal:

    * ``p = 1``, so ``k ~ h**2``, and the cost is ``(1/h)(1/k) ~ tol**(-3/2)``,
    * ``p = 2``, so ``k ~ h``, and the cost is ``tol**(-1)``.

    That is the real reason to prefer Crank-Nicolson, and it is a statement about the **order in
    time**, not about implicitness. The backward scheme is implicit, unconditionally stable, and
    lands in the **same asymptotic class** as the explicit one, because it too is first order in
    time.

    The measured ratio between those two is not 1, though, and the reason is worth knowing: at
    the mesh ratio the search lands on, the explicit scheme's time and space errors partly
    cancel, because ``u_tt = alpha**2 u_xxxx`` ties them together. At ``r = 1/6`` they cancel
    completely and the explicit scheme is fourth order in ``h``. `the_lucky_ratio` measures that
    separately, and it is why the explicit column here beats a first order asymptotic argument.
    """
    problem = sine_problem()
    rows = []
    for name, theta in (("explicit", 0.0), ("backward", 1.0), ("crank nicolson", 0.5)):
        limit = stability_limit(theta)
        best = None
        for n in points:
            m = int(n)
            h = (problem["b"] - problem["a"]) / (m - 1)
            # the fewest steps stability allows, then double until the target is met
            steps = 1
            if math.isfinite(limit):
                steps = max(int(math.ceil(float(t_end) * problem["alpha"]
                                          / (limit * h ** 2))), 1)
            previous = None
            while steps <= max_steps and (m - 2) * steps <= budget:
                if best is not None and (m - 2) * steps >= best["updates"]:
                    break            # already dearer than the cheapest found so far
                run = solve(problem, m, steps, float(t_end), theta=theta)
                if run["error"] <= target:
                    cost = (m - 2) * steps
                    if best is None or cost < best["updates"]:
                        best = {"scheme": name, "points": m, "steps": steps, "r": run["r"],
                                "k": run["k"], "h": run["h"], "error": run["error"],
                                "updates": cost,
                                "solves": 0 if theta == 0.0 else steps}
                    break
                # once the space error dominates, more time steps buy nothing and this h is
                # simply too coarse for the target. Detecting that is what keeps the search
                # from doubling its way to millions of useless steps.
                if previous is not None and run["error"] > 0.99 * previous:
                    break
                previous = run["error"]
                steps *= 2
        if best is None:
            raise SystemExit(f"{name} never reached {target} within the search")
        rows.append(best)
    cheapest = min(rows, key=lambda row: row["updates"])
    explicit = next(row for row in rows if row["scheme"] == "explicit")
    backward = next(row for row in rows if row["scheme"] == "backward")
    nicolson = next(row for row in rows if row["scheme"] == "crank nicolson")
    return {
        "rows": rows,
        "target": float(target),
        "cheapest": cheapest["scheme"],
        "spread": float(max(row["updates"] for row in rows)
                        / min(row["updates"] for row in rows)),
        "every_scheme_met_the_target": all(row["error"] <= target for row in rows),
        "crank_nicolson_is_cheapest": cheapest["scheme"] == "crank nicolson",
        "backward_over_explicit": float(backward["updates"] / explicit["updates"]),
        "explicit_over_crank_nicolson": float(explicit["updates"] / nicolson["updates"]),
        "backward_over_crank_nicolson": float(backward["updates"] / nicolson["updates"]),
        "k_over_h_squared": {row["scheme"]: float(row["k"] / row["h"] ** 2) for row in rows},
        "k_over_h": {row["scheme"]: float(row["k"] / row["h"]) for row in rows},
    }


def the_lucky_ratio(ratios=(0.1, 1.0 / 6.0, 0.2, 0.25, 0.4, 0.5),
                    refinements=(11, 21, 41, 81), t_end: float = 0.05):
    """The explicit scheme is fourth order in ``h`` at exactly one mesh ratio.

    The scheme's truncation error is

        (k/2) u_tt - (alpha h**2 / 12) u_xxxx + ... ,

    and on a solution of the heat equation ``u_tt = alpha**2 u_xxxx``, so the two terms are the
    same function of ``x`` and differ only by the factor ``alpha k / 2 - alpha h**2 / 12``. That
    vanishes when ``r = alpha k / h**2 = 1/6``, and there the leading error is gone.

    The measurement is unambiguous: order 2.00 at every other ratio, **4.00** at ``1/6``, with
    the error three orders of magnitude smaller on the same grids. It costs nothing to arrange,
    since ``1/6`` is comfortably inside the stability limit of ``1/2``.

    Two things stop this being the answer to everything. It is exact only for this equation with
    constant ``alpha``, since the cancellation used the equation itself; and it fixes the ratio,
    so refining ``h`` still forces ``k ~ h**2`` and the step count still grows like ``h**-2``.
    What it buys is accuracy per step, not freedom in the step.
    """
    problem = sine_problem()
    rows = []
    for value in ratios:
        r = float(value)
        errors, hs = [], []
        for n in refinements:
            m = int(n)
            h = (problem["b"] - problem["a"]) / (m - 1)
            k = r * h ** 2 / problem["alpha"]
            steps = max(int(round(float(t_end) / k)), 1)
            run = solve(problem, m, steps, steps * k, theta=0.0)
            errors.append(run["error"])
            hs.append(h)
        errors = np.asarray(errors)
        rows.append({
            "r": r,
            "h": np.asarray(hs),
            "error": errors,
            "order_in_h": float(np.polyfit(np.log(hs), np.log(errors), 1)[0]),
            "cancelling_factor": abs(0.5 * r - 1.0 / 12.0),
        })
    best = min(rows, key=lambda row: row["cancelling_factor"])
    others = [row for row in rows if row is not best]
    return {
        "rows": rows,
        "lucky_ratio": 1.0 / 6.0,
        "fourth_order_at_one_sixth": abs(best["order_in_h"] - 4.0) < 0.15,
        "second_order_everywhere_else": all(
            abs(row["order_in_h"] - 2.0) < 0.1 for row in others),
        "gain_at_the_finest_grid": float(
            min(row["error"][-1] for row in others) / best["error"][-1]),
        "the_cancelling_factor_is_zero_only_there": all(
            (row["cancelling_factor"] < 1e-12) == (row is best) for row in rows),
        "note": "the cancellation uses u_tt = alpha^2 u_xxxx, so it is a property of this "
                "equation and not of the scheme",
    }
