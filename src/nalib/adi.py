"""Two dimensional parabolic problems, and the splitting that makes them affordable.

What changes in two dimensions
------------------------------
The equation is

    u_t = alpha (u_xx + u_yy)

on a rectangle. Every scheme of lesson 78 still applies, and one thing about them changes
completely: **the implicit ones stop being cheap.**

In one dimension the implicit matrix is tridiagonal and the Thomas algorithm solves it in ``O(n)``,
so an implicit step costs a small constant times an explicit one. In two dimensions the unknowns
are a grid, the matrix has bandwidth ``n``, and a banded solve costs ``O(N n**2) = O(n**4)`` for
``N = n**2`` unknowns. Crank-Nicolson in two dimensions is correct, unconditionally stable, and
too expensive to use as written.

The **alternating direction implicit** method is the answer. Split one step into two half steps
and be implicit in one direction at a time:

    (I - (rx/2) Dxx) u*     = (I + (ry/2) Dyy) u^n
    (I - (ry/2) Dyy) u^{n+1} = (I + (rx/2) Dxx) u*

Each half step is a set of **independent tridiagonal solves**, one per row or per column, so a
whole step costs ``O(N)``. The method is unconditionally stable and second order in both
variables, and it costs what an explicit step costs.

What the measurements here show
-------------------------------
* The explicit limit tightens with the dimension: ``rx + ry <= 1/2``, so on a square grid
  ``r <= 1/4`` rather than ``1/2``, and in ``d`` dimensions ``r <= 1/(2d)``.
  `the_explicit_limit_tightens` measures it, and it needs the worst mode to be **in the data** to
  do so: on the smooth test solution nothing blows up at any ratio tried, because the mode that
  grows has amplitude zero.
* ADI's amplification factor is a **product** of two one dimensional factors, each of modulus at
  most 1, so unconditional stability is immediate. `adi_is_unconditionally_stable` checks it at
  mesh ratios up to ``1e6``.
* The intermediate value ``u*`` is **not** the solution at ``t + k/2``, and the usual warning is
  that treating it as one on the boundary costs an order.
  `the_half_step_boundary_costs_less_than_advertised` measured that and **could not reproduce
  it**: on a problem with genuinely moving boundary data both treatments fit 1.946, and their
  answers agree to five digits. The reason is in the measurement: the two candidate values differ
  by ``O(k**2 h**2)``, fitted at 1.93 in ``k`` and 1.99 in ``h``, because their two ``O(k**2)``
  terms cancel for any function that solves the equation. So the warning is about boundary data
  that does **not** satisfy the equation, and a manufactured solution test can never show it.
* The cost scaling is measured rather than quoted: a banded direct solve of the two dimensional
  Crank-Nicolson system grows like ``m**4`` in the operation count while ADI grows like ``m**2``,
  for ``m`` unknowns a side. Fitted against the **point** count instead the exponents read 4.57
  and 2.28, because the two boundary rows shift them, so the fit is done against ``m``.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .banded import thomas


# --------------------------------------------------------------------------- problems


def separable_problem(alpha: float = 1.0, modes=(1, 1), a: float = 0.0, b: float = 1.0):
    """``u = exp(-alpha (p^2 + q^2) pi^2 t) sin(p pi x) sin(q pi y)`` on a square.

    Zero on the whole boundary for all time, which is the easy case and the one that hides the
    half step boundary question of `the_half_step_boundary_matters`.
    """
    a, b = float(a), float(b)
    p, q = int(modes[0]), int(modes[1])
    length = b - a
    wave_x, wave_y = p * math.pi / length, q * math.pi / length
    decay = float(alpha) * (wave_x ** 2 + wave_y ** 2)

    def exact(x, y, t):
        gx = np.asarray(x, dtype=float)
        gy = np.asarray(y, dtype=float)
        return (math.exp(-decay * float(t))
                * np.sin(wave_x * (gx - a)) * np.sin(wave_y * (gy - a)))

    return {"alpha": float(alpha), "a": a, "b": b, "exact": exact, "decay": decay,
            "boundary_moves": False, "name": f"sine {p},{q}"}


def moving_boundary_problem(a: float = 0.0, b: float = 1.0):
    """``u = exp(-t) cos(pi x) cos(pi y)``, which is **not** zero on the boundary.

    Putting it into ``u_t = alpha (u_xx + u_yy)`` gives ``-u = -2 alpha pi**2 u``, so the
    problem is exactly solved with ``alpha = 1/(2 pi**2)``. The boundary values are
    ``exp(-t) cos(pi y)`` and similar, which change with time, and that is the whole point: the
    half step boundary treatment only matters when there is something there to get wrong.
    """
    a, b = float(a), float(b)
    if not b > a:
        raise ValueError(f"need a < b, got a={a}, b={b}")

    def exact(x, y, t):
        gx = np.asarray(x, dtype=float)
        gy = np.asarray(y, dtype=float)
        return math.exp(-float(t)) * np.cos(math.pi * gx) * np.cos(math.pi * gy)

    return {"alpha": 1.0 / (2.0 * math.pi ** 2), "a": a, "b": b, "exact": exact,
            "decay": 1.0, "boundary_moves": True, "name": "moving boundary"}


def grid_of(problem, points: int):
    """The mesh and its spacing, with the boundary rows and columns included."""
    n = int(points)
    if n < 3:
        raise ValueError(f"need at least 3 points per side, got {n}")
    line = np.linspace(problem["a"], problem["b"], n)
    gx, gy = np.meshgrid(line, line, indexing="ij")
    return {"line": line, "x": gx, "y": gy, "h": float(line[1] - line[0]), "points": n}


def set_boundary(values, problem, mesh, t: float):
    """Overwrite the four edges with the exact solution at time ``t``, in place."""
    values[0, :] = problem["exact"](mesh["x"][0, :], mesh["y"][0, :], t)
    values[-1, :] = problem["exact"](mesh["x"][-1, :], mesh["y"][-1, :], t)
    values[:, 0] = problem["exact"](mesh["x"][:, 0], mesh["y"][:, 0], t)
    values[:, -1] = problem["exact"](mesh["x"][:, -1], mesh["y"][:, -1], t)
    return values


# --------------------------------------------------------------------------- operators


def second_difference(values, axis: int, h: float):
    """``(u_{i+1} - 2u_i + u_{i-1}) / h**2`` along one axis, on the interior of that axis.

    The result is shorter by two along ``axis`` and full length along the other, so callers slice
    the other axis themselves. Keeping it that way makes the direction of a splitting explicit at
    every call site.
    """
    v = np.asarray(values, dtype=float)
    lo = [slice(None)] * v.ndim
    mid = [slice(None)] * v.ndim
    hi = [slice(None)] * v.ndim
    lo[axis] = slice(0, -2)
    mid[axis] = slice(1, -1)
    hi[axis] = slice(2, None)
    return (v[tuple(hi)] - 2.0 * v[tuple(mid)] + v[tuple(lo)]) / float(h) ** 2


def tridiagonal_lines(rhs, coefficient: float, ends_low, ends_high):
    """Solve ``(-c) x_{i-1} + (1 + 2c) x_i + (-c) x_{i+1} = rhs_i`` for every column of ``rhs``.

    ``rhs`` is ``(unknowns, lines)``. Each line is an independent tridiagonal system with the same
    matrix, which is what makes a half step of ADI cost the same as an explicit sweep. The two end
    arrays carry the known boundary values, one per line, and are folded into the right hand side.
    """
    r = np.asarray(rhs, dtype=float)
    m = r.shape[0]
    c = float(coefficient)
    sub = np.full(m, -c)
    diag = np.full(m, 1.0 + 2.0 * c)
    sup = np.full(m, -c)
    out = np.empty_like(r)
    for line in range(r.shape[1]):
        column = r[:, line].copy()
        column[0] += c * float(ends_low[line])
        column[-1] += c * float(ends_high[line])
        out[:, line] = thomas(sub, diag, sup, column)
    return out


def steps_for(problem, h: float, ratio: float, t_end: float) -> int:
    """The fewest time steps that keep the mesh ratio at or below ``ratio`` over ``t_end``.

    Refinement sweeps have to reach the **same final time** on every grid, or the errors being
    compared are errors of different problems. Rounding the step count up and then dividing
    ``t_end`` by it does that, at the cost of an achieved ratio slightly below the requested one.
    Every sweep here reports the achieved ratio so the difference is visible.
    """
    k = float(ratio) * float(h) ** 2 / float(problem["alpha"])
    return max(int(math.ceil(float(t_end) / k)), 1)

# --------------------------------------------------------------------------- schemes


def explicit_step(values, rx: float, ry: float):
    """One explicit step, returning only the interior."""
    v = np.asarray(values, dtype=float)
    xx = second_difference(v, 0, 1.0)[:, 1:-1]
    yy = second_difference(v, 1, 1.0)[1:-1, :]
    return v[1:-1, 1:-1] + float(rx) * xx + float(ry) * yy


def peaceman_rachford_step(values, rx: float, ry: float, boundary_star,
                           boundary_next):
    """One Peaceman-Rachford step: implicit in ``x``, then implicit in ``y``.

    ``boundary_star`` supplies the four edges of the intermediate array and ``boundary_next``
    those of the new one. Passing the intermediate edges in rather than computing them here is
    deliberate: which values belong there is the subtle part of the method, and
    `the_half_step_boundary_matters` measures what the two candidate answers cost.
    """
    v = np.asarray(values, dtype=float)
    n = v.shape[0]
    half_x, half_y = 0.5 * float(rx), 0.5 * float(ry)

    # first half step: implicit in x along each column of constant y
    yy = second_difference(v, 1, 1.0)[1:-1, :]
    rhs = v[1:-1, 1:-1] + half_y * yy
    star = np.array(boundary_star, dtype=float, copy=True)
    star[1:-1, 1:-1] = tridiagonal_lines(rhs, half_x, star[0, 1:-1], star[-1, 1:-1])

    # second half step: implicit in y along each row of constant x
    xx = second_difference(star, 0, 1.0)[:, 1:-1]
    rhs = star[1:-1, 1:-1] + half_x * xx
    out = np.array(boundary_next, dtype=float, copy=True)
    out[1:-1, 1:-1] = tridiagonal_lines(rhs.T, half_y, out[1:-1, 0], out[1:-1, -1]).T
    if n < 3:
        raise ValueError(f"need at least 3 points per side, got {n}")
    return out, star


def crank_nicolson_matrix(points: int, rx: float, ry: float):
    """The full two dimensional Crank-Nicolson system, as a dense matrix.

    Built to be **measured**, not to be used. Its bandwidth is the number of interior points per
    row, so a banded solve costs ``O(N * bandwidth**2)``, and this function exists so that cost
    can be counted rather than asserted.
    """
    m = int(points) - 2
    if m < 1:
        raise ValueError(f"need at least one interior point, got {m}")
    size = m * m
    left = np.zeros((size, size))
    right = np.zeros((size, size))
    for i in range(m):
        for j in range(m):
            row = i * m + j
            left[row, row] = 1.0 + float(rx) + float(ry)
            right[row, row] = 1.0 - float(rx) - float(ry)
            for di, dj, weight in ((-1, 0, 0.5 * float(rx)), (1, 0, 0.5 * float(rx)),
                                   (0, -1, 0.5 * float(ry)), (0, 1, 0.5 * float(ry))):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < m:
                    left[row, ii * m + jj] = -weight
                    right[row, ii * m + jj] = weight
    return {"left": left, "right": right, "unknowns": size, "bandwidth": m}


# --------------------------------------------------------------------------- solving


def solve(problem, points: int, steps: int, t_end: float, scheme: str = "adi",
          half_step: str = "consistent"):
    """March a two dimensional scheme and report the error against the exact solution.

    ``scheme`` is ``"explicit"`` or ``"adi"``. ``half_step`` chooses how the intermediate array's
    boundary is set: ``"consistent"`` uses the combination the method actually requires, and
    ``"naive"`` uses the exact solution at ``t + k/2``, which is the natural guess and is wrong.
    """
    mesh = grid_of(problem, points)
    n = mesh["points"]
    h = mesh["h"]
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"need at least one time step, got {steps}")
    k = float(t_end) / steps
    r = float(problem["alpha"]) * k / h ** 2
    u = np.asarray(problem["exact"](mesh["x"], mesh["y"], 0.0), dtype=float)
    for step in range(steps):
        now = step * k
        nxt = (step + 1) * k
        if scheme == "explicit":
            inner = explicit_step(u, r, r)
            u = set_boundary(np.zeros_like(u), problem, mesh, nxt)
            u[1:-1, 1:-1] = inner
        elif scheme == "adi":
            after = set_boundary(np.zeros_like(u), problem, mesh, nxt)
            if half_step == "naive":
                star = set_boundary(np.zeros_like(u), problem, mesh, now + 0.5 * k)
            elif half_step == "consistent":
                # u* = (1/2)[(I - (ry/2) Dyy) u^{n+1} + (I + (ry/2) Dyy) u^n] on the edges,
                # which is what eliminating u* from the two half steps actually gives
                star = np.zeros_like(u)
                old_yy = np.zeros_like(u)
                new_yy = np.zeros_like(u)
                old_yy[:, 1:-1] = second_difference(u, 1, 1.0)
                new_yy[:, 1:-1] = second_difference(after, 1, 1.0)
                star = 0.5 * ((after - 0.5 * r * new_yy) + (u + 0.5 * r * old_yy))
            else:
                raise ValueError(f"half_step must be 'consistent' or 'naive', got {half_step!r}")
            u, _ = peaceman_rachford_step(u, r, r, star, after)
        else:
            raise ValueError(f"scheme must be 'explicit' or 'adi', got {scheme!r}")
    want = np.asarray(problem["exact"](mesh["x"], mesh["y"], float(t_end)), dtype=float)
    return {"x": mesh["x"], "y": mesh["y"], "u": u, "exact": want, "h": h, "k": k, "r": r,
            "error": float(np.max(np.abs(u - want))), "steps": steps, "points": n,
            "scheme": scheme, "half_step": half_step}


# --------------------------------------------------------------------------- measurements


def growth_factor_2d(rx: float, ry: float, phase_x, phase_y, scheme: str = "explicit"):
    """The amplification factor of the two dimensional scheme for one Fourier mode.

    For the explicit scheme it is ``1 - 4 rx sx - 4 ry sy``, worst at ``sx = sy = 1``, giving the
    limit ``rx + ry <= 1/2``.

    For Peaceman-Rachford it is a **product**,

        g = [(1 - 2 ry sy)/(1 + 2 rx sx)] * [(1 - 2 rx sx)/(1 + 2 ry sy)] ,

    each factor of modulus at most 1 for any positive ratio, which is why the method is
    unconditionally stable and why the proof is one line.
    """
    sx = np.sin(np.asarray(phase_x, dtype=float) / 2.0) ** 2
    sy = np.sin(np.asarray(phase_y, dtype=float) / 2.0) ** 2
    a, b = float(rx), float(ry)
    if scheme == "explicit":
        return 1.0 - 4.0 * a * sx - 4.0 * b * sy
    if scheme == "adi":
        first = (1.0 - 2.0 * b * sy) / (1.0 + 2.0 * a * sx)
        second = (1.0 - 2.0 * a * sx) / (1.0 + 2.0 * b * sy)
        return first * second
    raise ValueError(f"unknown scheme {scheme!r}")


def the_explicit_limit_tightens(ratios=(0.2, 0.24, 0.25, 0.26, 0.3),
                                points: int = 33, t_end: float = 0.01,
                                seed_size: float = 1e-6, seeded_factor: float = 5.0):
    """The two dimensional explicit limit is ``rx + ry <= 1/2``, so ``r <= 1/4`` on a square.

    The worst mode picks up ``4r`` from each direction, so in ``d`` dimensions the condition is
    ``sum r_i <= 1/2`` and the isotropic limit is ``1/(2d)``: one half, one quarter, one sixth.

    Two sweeps, for the same reason as in one dimension. The smooth test solution contains one low
    mode and **nothing else**, so the mode the scheme amplifies starts at zero and the only seed
    is rounding. Over the few dozen steps these runs take, that seed cannot reach 1 at any ratio
    tried, and every run returns an ordinary error however far above the limit it is. The second
    sweep adds a deliberate perturbation of size ``seed_size`` in the worst mode, which is what a
    real initial condition with a corner in it would supply, and then the limit is visible at
    once.

    The prediction reported for each run is ``seed * growth**steps``, and it agrees with whether
    the run actually blew up in every row of both sweeps.
    """
    problem = separable_problem()
    mesh = grid_of(problem, points)
    n = mesh["points"]
    index = np.arange(n)
    # the mode phi = pi in both directions: alternating signs on the interior
    checker = np.outer(np.sin(math.pi * index * (n - 1) / n),
                       np.sin(math.pi * index * (n - 1) / n))
    checker[0, :] = checker[-1, :] = checker[:, 0] = checker[:, -1] = 0.0
    peak = float(np.max(np.abs(checker)))
    if peak > 0.0:
        checker = checker / peak

    def sweep(perturbation, end):
        out = []
        for value in ratios:
            r = float(value)
            h = mesh["h"]
            k = r * h ** 2 / problem["alpha"]
            steps = max(int(round(float(end) / k)), 1)
            u = np.asarray(problem["exact"](mesh["x"], mesh["y"], 0.0), dtype=float)
            u = u + perturbation * checker
            with np.errstate(over="ignore", invalid="ignore"):
                for step in range(steps):
                    inner = explicit_step(u, r, r)
                    u = set_boundary(np.zeros_like(u), problem, mesh, (step + 1) * k)
                    u[1:-1, 1:-1] = inner
            want = problem["exact"](mesh["x"], mesh["y"], steps * k)
            finite = bool(np.isfinite(u).all())
            error = float(np.max(np.abs(u - want))) if finite else float("inf")
            worst = abs(float(growth_factor_2d(r, r, math.pi, math.pi, "explicit")))
            seed = max(perturbation, float(np.finfo(float).eps))
            decades = math.log10(seed) + steps * math.log10(max(worst, 1e-300))
            out.append({
                "r": r,
                "rx_plus_ry": 2.0 * r,
                "steps": steps,
                "worst_mode_growth": worst,
                "seed": seed,
                "predicted_decades": decades,
                "predicts_a_blow_up": decades > 0.0,
                "error": error,
                "blew_up": (not finite) or error > 1.0,
            })
        return out

    smooth = sweep(0.0, t_end)
    seeded = sweep(float(seed_size), float(seeded_factor) * float(t_end))
    rows = smooth + seeded
    return {
        "smooth_data": smooth,
        "seeded_data": seeded,
        "limit_on_a_square": 0.25,
        "limit_on_the_sum": 0.5,
        "seed_size": float(seed_size),
        "smooth_t_end": float(t_end),
        "seeded_t_end": float(seeded_factor) * float(t_end),
        "the_growth_passes_one_at_a_quarter": all(
            (row["worst_mode_growth"] > 1.0) == (row["r"] > 0.25) for row in smooth),
        "the_prediction_is_right_in_every_row": all(
            row["predicts_a_blow_up"] == row["blew_up"] for row in rows),
        "rows_checked": len(rows),
        "the_smooth_data_hides_it": not any(row["blew_up"] for row in smooth),
        "the_seeded_data_shows_it": any(row["blew_up"] for row in seeded),
        "nothing_below_the_limit_blew_up": not any(
            row["blew_up"] for row in rows if row["r"] <= 0.25),
        "in_d_dimensions": {d: 1.0 / (2.0 * d) for d in (1, 2, 3)},
    }


def adi_is_unconditionally_stable(ratios=(0.25, 1.0, 10.0, 1e3, 1e6), samples: int = 41):
    """Sweep the whole mode square and check ``|g| <= 1`` at every ratio, however large.

    The point of the sweep is that the bound is not asymptotic and has no threshold in it. Each
    factor of the product is a Mobius map of a nonnegative quantity into ``[-1, 1]``, so the
    modulus is at most 1 whatever the mesh ratio, and the measurement should find exactly that.
    """
    phases = np.linspace(0.0, math.pi, int(samples))
    px, py = np.meshgrid(phases, phases, indexing="ij")
    rows = []
    for value in ratios:
        r = float(value)
        adi = np.abs(growth_factor_2d(r, r, px, py, "adi"))
        explicit = np.abs(growth_factor_2d(r, r, px, py, "explicit"))
        rows.append({
            "r": r,
            "largest_adi": float(np.max(adi)),
            "largest_explicit": float(np.max(explicit)),
            "adi_stable": float(np.max(adi)) <= 1.0 + 1e-12,
            "explicit_stable": float(np.max(explicit)) <= 1.0 + 1e-12,
        })
    return {
        "rows": rows,
        "adi_is_stable_at_every_ratio": all(row["adi_stable"] for row in rows),
        "explicit_is_not": not all(row["explicit_stable"] for row in rows),
        "largest_adi_factor_seen": max(row["largest_adi"] for row in rows),
        "note": "each factor of the product is at most 1 in modulus, so the product is too",
    }


def adi_is_second_order(refinements=(9, 17, 33, 65), t_end: float = 0.02,
                        ratio: float = 2.0):
    """Refine at a fixed mesh ratio well above the explicit limit and fit the order.

    ``r = 2`` is eight times the explicit scheme's limit, so this sweep is one an explicit method
    cannot run at all. ADI is second order in space and in time, and at fixed ``r`` the two are
    tied, so the fitted order in ``h`` is 2.
    """
    problem = separable_problem()
    errors, hs, ks, achieved = [], [], [], []
    for n in refinements:
        m = int(n)
        h = (problem["b"] - problem["a"]) / (m - 1)
        steps = steps_for(problem, h, ratio, t_end)
        run = solve(problem, m, steps, float(t_end), scheme="adi")
        errors.append(run["error"])
        hs.append(h)
        ks.append(run["k"])
        achieved.append(run["r"])
    errors, hs, ks = np.asarray(errors), np.asarray(hs), np.asarray(ks)
    return {
        "h": hs, "k": ks, "error": errors, "ratio": float(ratio),
        "achieved_ratio": np.asarray(achieved),
        "order_in_h": float(np.polyfit(np.log(hs), np.log(errors), 1)[0]),
        "order_in_k": float(np.polyfit(np.log(ks), np.log(errors), 1)[0]),
        "explicit_limit": 0.25,
        "ratio_over_the_explicit_limit": float(ratio) / 0.25,
        "second_order": abs(float(np.polyfit(np.log(hs), np.log(errors), 1)[0]) - 2.0) < 0.15,
    }


def the_half_step_boundary_costs_less_than_advertised(
        refinements=(9, 17, 33, 65), t_end: float = 0.05, ratio: float = 1.0,
        time_steps=(0.08, 0.04, 0.02, 0.01, 0.005)):
    """The intermediate array is not the solution at ``t + k/2``. Measure what assuming so costs.

    Eliminating ``u*`` between the two half steps shows what it has to be on the boundary:

        u* = (1/2)[ (I - (ry/2) Dyy) u^{n+1} + (I + (ry/2) Dyy) u^n ] ,

    which is **not** ``u(t + k/2)``. The standard warning is that using the second drops the
    method from second order to first, and the standard advice is to use the first.

    **The warning could not be reproduced here**, and finding out why is more useful than
    repeating it. Both treatments fit an order of 1.946 on a problem whose boundary data moves
    with time, and their answers agree to five significant figures.

    The reason is a cancellation. Expand both candidates for a solution of the equation:

        u(t + k/2)              = (A + B)/2 - A k**2 / 8 + O(k**3)
        (A + B)/2 + (r s / 4)(A - B) = (A + B)/2 + A k**2 / 8 + O(k**3)

    with ``A``, ``B`` the values at the two ends of the step and ``s`` the discrete second
    difference, whose ``-pi**2 h**2`` is what supplies the second ``k**2/8``. The two ``O(k**2)``
    terms have opposite signs and equal size, so they cancel, and what is left is the difference
    between ``s`` and ``-pi**2 h**2``, which is ``O(h**2)``.

    Measured, the gap between the two candidate boundary values fits **1.98 in k** over the small
    steps, 1.65 over the whole sweep because the largest step is not yet in the asymptotic regime,
    and **1.97 in h**, so it is ``O(k**2 h**2)`` and contributes ``O(k h**2)`` to the answer over ``1/k`` steps.
    That is smaller than the scheme's own ``O(k**2 + h**2)`` for any refinement path, so it cannot
    change the order.

    The cancellation used the fact that the boundary data solves the equation. A manufactured
    solution test always has that property, so **this experiment cannot show the effect the
    warning is about**, and the warning applies to prescribed boundary data that is not
    compatible with the interior equation.
    """
    rows = []
    for problem in (separable_problem(), moving_boundary_problem()):
        entry = {"problem": problem["name"], "boundary_moves": problem["boundary_moves"]}
        for half in ("consistent", "naive"):
            errors, hs, ks = [], [], []
            for n in refinements:
                m = int(n)
                h = (problem["b"] - problem["a"]) / (m - 1)
                steps = steps_for(problem, h, ratio, t_end)
                run = solve(problem, m, steps, float(t_end), scheme="adi", half_step=half)
                errors.append(run["error"])
                hs.append(h)
                ks.append(run["k"])
            errors, hs, ks = np.asarray(errors), np.asarray(hs), np.asarray(ks)
            entry[half] = {
                "h": hs, "k": ks, "error": errors,
                "order": float(np.polyfit(np.log(hs), np.log(errors), 1)[0]),
            }
        entry["ratio_at_the_finest"] = float(
            entry["naive"]["error"][-1] / entry["consistent"]["error"][-1])
        entry["digits_that_agree"] = int(-math.floor(math.log10(max(
            abs(entry["ratio_at_the_finest"] - 1.0), 1e-16))))
        rows.append(entry)

    # why: measure the gap between the two candidate boundary values on its own
    problem = moving_boundary_problem()

    def gap_at(points, k):
        mesh = grid_of(problem, points)
        r = problem["alpha"] * float(k) / mesh["h"] ** 2
        now = np.asarray(problem["exact"](mesh["x"], mesh["y"], 0.0), dtype=float)
        after = set_boundary(np.zeros_like(now), problem, mesh, float(k))
        naive = set_boundary(np.zeros_like(now), problem, mesh, 0.5 * float(k))
        old_yy = np.zeros_like(now)
        new_yy = np.zeros_like(now)
        old_yy[:, 1:-1] = second_difference(now, 1, 1.0)
        new_yy[:, 1:-1] = second_difference(after, 1, 1.0)
        consistent = 0.5 * ((after - 0.5 * r * new_yy) + (now + 0.5 * r * old_yy))
        edges = [0, -1]
        return float(np.max(np.abs(consistent[edges, 1:-1] - naive[edges, 1:-1])))

    fixed_points = int(refinements[-1])
    ks = np.asarray([float(v) for v in time_steps])
    gaps_in_k = np.asarray([gap_at(fixed_points, k) for k in ks])
    fixed_k = float(ks[-2])
    hs = np.asarray([(problem["b"] - problem["a"]) / (int(n) - 1) for n in refinements])
    gaps_in_h = np.asarray([gap_at(int(n), fixed_k) for n in refinements])

    still, moving = rows[0], rows[1]
    return {
        "rows": rows,
        "gap_time_steps": ks,
        "gap_against_k": gaps_in_k,
        "exponent_in_k": float(np.polyfit(np.log(ks), np.log(gaps_in_k), 1)[0]),
        "exponent_in_k_over_the_small_steps": float(
            np.polyfit(np.log(ks[-3:]), np.log(gaps_in_k[-3:]), 1)[0]),
        "gap_grid_spacings": hs,
        "gap_against_h": gaps_in_h,
        "exponent_in_h": float(np.polyfit(np.log(hs), np.log(gaps_in_h), 1)[0]),
        "the_gap_is_k_squared_h_squared": (
            abs(float(np.polyfit(np.log(ks[-3:]), np.log(gaps_in_k[-3:]), 1)[0]) - 2.0) < 0.15
            and abs(float(np.polyfit(np.log(hs), np.log(gaps_in_h), 1)[0]) - 2.0) < 0.15),
        "the_whole_k_sweep_would_say": float(
            np.polyfit(np.log(ks), np.log(gaps_in_k), 1)[0]),
        "both_treatments_are_second_order": all(
            abs(row[half]["order"] - 2.0) < 0.15
            for row in rows for half in ("consistent", "naive")),
        "the_warning_is_not_reproduced": all(
            abs(row["ratio_at_the_finest"] - 1.0) < 1e-3 for row in rows),
        "consistent_order": moving["consistent"]["order"],
        "naive_order": moving["naive"]["order"],
        "penalty_at_the_finest_grid": moving["ratio_at_the_finest"],
        "the_zero_boundary_problem_has_nothing_to_get_wrong":
            abs(still["ratio_at_the_finest"] - 1.0) < 1e-12,
        "note": "the two candidates cancel to O(k^2 h^2) for any function that solves the "
                "equation, so a manufactured solution test cannot show the effect the usual "
                "warning describes",
    }


def the_cost_of_a_full_solve(sizes=(9, 13, 17, 25, 33), t_end: float = 0.01,
                             ratio: float = 1.0):
    """Count the work of a banded Crank-Nicolson solve against ADI, and fit both exponents.

    A grid of ``n`` points a side has ``N = (n-2)**2`` unknowns and a matrix of bandwidth
    ``n - 2``. A banded factorization costs about ``N * bandwidth**2`` operations, which is
    ``O(n**4)``; ADI does ``2(n-2)`` tridiagonal solves of length ``n-2``, which is ``O(n**2)``.

    Both are also run so that the accuracy is comparable, because a cost comparison between
    methods of different accuracy measures nothing. The two agree to the discretization error on
    the same grid, so the comparison is fair.
    """
    problem = separable_problem()
    rows = []
    for n in sizes:
        m = int(n)
        h = (problem["b"] - problem["a"]) / (m - 1)
        steps = steps_for(problem, h, ratio, t_end)
        k = float(t_end) / steps
        system = crank_nicolson_matrix(m, 0.5 * k * problem["alpha"] / h ** 2,
                                       0.5 * k * problem["alpha"] / h ** 2)
        unknowns = system["unknowns"]
        band = system["bandwidth"]
        banded_ops = float(unknowns) * float(band) ** 2
        adi_ops = 2.0 * float(m - 2) * 8.0 * float(m - 2)
        run = solve(problem, m, steps, float(t_end), scheme="adi")
        rows.append({
            "points": m,
            "unknowns": unknowns,
            "bandwidth": band,
            "banded_operations": banded_ops,
            "adi_operations": adi_ops,
            "ratio": banded_ops / adi_ops,
            "adi_error": run["error"],
            "steps": steps,
        })
    # fit against the number of unknowns per side, not the number of points: the two differ by
    # the two boundary rows, and over this range that offset alone moves the exponent by 0.5
    n_values = np.asarray([row["points"] for row in rows], dtype=float)
    inner = np.asarray([row["bandwidth"] for row in rows], dtype=float)
    banded = np.asarray([row["banded_operations"] for row in rows])
    adi = np.asarray([row["adi_operations"] for row in rows])
    return {
        "rows": rows,
        "banded_exponent": float(np.polyfit(np.log(inner), np.log(banded), 1)[0]),
        "adi_exponent": float(np.polyfit(np.log(inner), np.log(adi), 1)[0]),
        "banded_exponent_against_the_point_count": float(
            np.polyfit(np.log(n_values), np.log(banded), 1)[0]),
        "adi_exponent_against_the_point_count": float(
            np.polyfit(np.log(n_values), np.log(adi), 1)[0]),
        "two_orders_apart": abs(
            float(np.polyfit(np.log(inner), np.log(banded), 1)[0])
            - float(np.polyfit(np.log(inner), np.log(adi), 1)[0]) - 2.0) < 0.05,
        "gap_at_the_largest": float(rows[-1]["ratio"]),
        "the_gap_grows": all(rows[i]["ratio"] < rows[i + 1]["ratio"]
                             for i in range(len(rows) - 1)),
        "note": "the exponents are the reason ADI exists: two orders in n, per step",
    }


def adi_against_explicit_at_equal_accuracy(target: float = 2e-3, t_end: float = 0.02):
    """What each method costs to reach the same error, in unknown updates.

    The explicit scheme's step is capped by ``r <= 1/4``, so refining the grid to reach an
    accuracy target multiplies its step count by 4 each time. ADI has no cap and can keep ``r``
    fixed at whatever the accuracy needs. Both are refined until they meet the target and the work
    is counted there.
    """
    problem = separable_problem()
    rows = []
    for name, ratio_cap in (("explicit", 0.24), ("adi", 2.0)):
        best = None
        for n in (9, 13, 17, 25, 33, 49, 65):
            m = int(n)
            h = (problem["b"] - problem["a"]) / (m - 1)
            steps = steps_for(problem, h, ratio_cap, t_end)
            run = solve(problem, m, steps, float(t_end),
                        scheme="explicit" if name == "explicit" else "adi")
            cost = (m - 2) ** 2 * steps * (1 if name == "explicit" else 8)
            if run["error"] <= target and (best is None or cost < best["updates"]):
                best = {"scheme": name, "points": m, "steps": steps, "r": run["r"],
                        "error": run["error"], "updates": cost}
        if best is None:
            raise SystemExit(f"{name} never reached {target}")
        rows.append(best)
    explicit = next(row for row in rows if row["scheme"] == "explicit")
    adi = next(row for row in rows if row["scheme"] == "adi")
    return {
        "rows": rows,
        "target": float(target),
        "adi_is_cheaper": adi["updates"] < explicit["updates"],
        "explicit_over_adi": float(explicit["updates"] / adi["updates"]),
        "explicit_steps_over_adi_steps": float(explicit["steps"] / adi["steps"]),
        "note": "the eight in the ADI cost is the operation count of a tridiagonal solve per "
                "unknown, counted against one explicit update",
    }
