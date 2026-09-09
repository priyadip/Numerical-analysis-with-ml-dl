"""Geometric integrators: methods that get the structure right rather than the trajectory.

The problem with accuracy
-------------------------
Lesson 73 measured RK4's energy on an undamped pendulum and found it drifting steadily downwards.
That drift is small per step and it is **one way**, so over a long integration it accumulates
without limit. RK4 is fourth order and the drift is still there; refining the step makes it
smaller and does not make it stop.

The reason is in lesson 72's stability picture: a Hamiltonian system has eigenvalues on the
**imaginary axis**, and RK4's stability region crosses that axis at $2\\sqrt{2}$ but is inside
the unit circle nowhere along it except at the origin. So the computed amplitude decays, always,
by an amount set by the step.

What a symplectic method does instead
-------------------------------------
A Hamiltonian flow preserves the area of any region of phase space, which is Liouville's theorem.
A **symplectic** integrator preserves that area exactly, in floating point, whatever the step.

The consequence is not that the energy is conserved. It is that the computed solution is the
**exact** solution of a nearby Hamiltonian, one differing from the true one by ``O(h^p)``. So
its energy is exactly conserved for that nearby problem, and the computed energy therefore
**oscillates within a bounded band** rather than drifting.

That is a qualitatively different guarantee, and it is why a solar system integration uses
Stormer-Verlet at second order rather than RK4 at fourth.

The price
---------
Symplectic methods are for problems in the separable form ``H(q, p) = T(p) + V(q)``, which covers
mechanics and not much else. They also want a **fixed step**: adapting the step destroys the
symplectic property, which is why the two ideas of Part 10 that each work beautifully do not
compose.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np


def as_pair(q, p) -> tuple:
    """Force the position and momentum into matching 1-D float arrays."""
    a = np.atleast_1d(np.asarray(q, dtype=float)).ravel()
    b = np.atleast_1d(np.asarray(p, dtype=float)).ravel()
    if a.size != b.size:
        raise ValueError(f"position has {a.size} components and momentum has {b.size}")
    return a, b


def symplectic_euler_step(force, q, p, h: float, _counter=None) -> tuple:
    """One symplectic Euler step: momentum first, then position with the **new** momentum.

    ``p_new = p + h force(q)`` then ``q_new = q + h p_new``.

    The only difference from ordinary Euler is which momentum the position update uses, and that
    single change turns a method whose energy grows without bound into one whose energy is
    bounded forever. It is first order either way.
    """
    a, b = as_pair(q, p)
    if _counter is not None:
        _counter[0] += 1
    p_new = b + h * np.atleast_1d(np.asarray(force(a), dtype=float)).ravel()
    return a + h * p_new, p_new


def explicit_euler_step(force, q, p, h: float, _counter=None) -> tuple:
    """Ordinary explicit Euler on the same system, for comparison.

    ``q_new = q + h p`` and ``p_new = p + h force(q)``, both using the **old** values.
    """
    a, b = as_pair(q, p)
    if _counter is not None:
        _counter[0] += 1
    return a + h * b, b + h * np.atleast_1d(np.asarray(force(a), dtype=float)).ravel()


def stormer_verlet_step(force, q, p, h: float, _counter=None) -> tuple:
    """One Stormer-Verlet step: half a momentum kick, a full position drift, half a kick.

    Second order, symplectic, time reversible, and one force evaluation per step once the last
    force is carried forward. It is the method behind essentially every molecular dynamics and
    long term orbital calculation there is.
    """
    a, b = as_pair(q, p)
    if _counter is not None:
        _counter[0] += 1
    half = b + 0.5 * h * np.atleast_1d(np.asarray(force(a), dtype=float)).ravel()
    q_new = a + h * half
    if _counter is not None:
        _counter[0] += 1
    p_new = half + 0.5 * h * np.atleast_1d(np.asarray(force(q_new), dtype=float)).ravel()
    return q_new, p_new


#: The available steps, as ``name -> (function, order, symplectic)``.
STEPS = {
    "explicit euler": (explicit_euler_step, 1, False),
    "symplectic euler": (symplectic_euler_step, 1, True),
    "stormer verlet": (stormer_verlet_step, 2, True),
}


def step_named(name: str):
    key = str(name).lower()
    if key not in STEPS:
        raise ValueError(f"unknown method {name!r}, expected one of {sorted(STEPS)}")
    return STEPS[key][0]


def integrate(force, q0, p0, t_end: float, steps: int, name: str = "stormer verlet",
              _counter=None) -> dict:
    """March a separable Hamiltonian system with one of the methods above."""
    n = int(steps)
    if n < 1:
        raise ValueError(f"need at least one step, got {n}")
    take = step_named(name)
    q, p = as_pair(q0, p0)
    h = float(t_end) / n
    qs = np.empty((n + 1, q.size))
    ps = np.empty((n + 1, p.size))
    qs[0], ps[0] = q, p
    for k in range(n):
        q, p = take(force, q, p, h, _counter)
        qs[k + 1], ps[k + 1] = q, p
    return {"t": np.linspace(0.0, float(t_end), n + 1), "q": qs, "p": ps,
            "step": h, "steps": n, "name": name,
            "evaluations": (_counter[0] if _counter is not None else None)}


# --------------------------------------------------------------------------- the problems


def harmonic_oscillator(omega: float = 1.0):
    """``H = p^2/2 + omega^2 q^2/2``, whose exact solution is a circle in phase space."""
    def force(q):
        return -omega ** 2 * np.atleast_1d(np.asarray(q, dtype=float))

    def energy(q, p):
        a = np.atleast_2d(np.asarray(q, dtype=float))
        b = np.atleast_2d(np.asarray(p, dtype=float))
        return 0.5 * np.sum(b ** 2, axis=1) + 0.5 * omega ** 2 * np.sum(a ** 2, axis=1)

    def exact(t, q0, p0):
        return (q0 * math.cos(omega * t) + (p0 / omega) * math.sin(omega * t),
                -q0 * omega * math.sin(omega * t) + p0 * math.cos(omega * t))

    return force, energy, exact


def pendulum(gravity: float = 9.81, length: float = 1.0):
    """``H = p^2/2 + g L (1 - cos q)``, the nonlinear pendulum in Hamiltonian form."""
    def force(q):
        return -(gravity / length) * np.sin(np.atleast_1d(np.asarray(q, dtype=float)))

    def energy(q, p):
        a = np.atleast_2d(np.asarray(q, dtype=float))
        b = np.atleast_2d(np.asarray(p, dtype=float))
        return 0.5 * np.sum(b ** 2, axis=1) + (gravity / length) * np.sum(1.0 - np.cos(a),
                                                                         axis=1)

    return force, energy


def kepler(mass: float = 1.0):
    """The planar Kepler problem as a separable Hamiltonian in ``(q, p)``."""
    def force(q):
        v = np.atleast_1d(np.asarray(q, dtype=float))
        r = float(np.linalg.norm(v))
        return -mass * v / max(r, 1e-300) ** 3

    def energy(q, p):
        a = np.atleast_2d(np.asarray(q, dtype=float))
        b = np.atleast_2d(np.asarray(p, dtype=float))
        r = np.linalg.norm(a, axis=1)
        return 0.5 * np.sum(b ** 2, axis=1) - mass / np.maximum(r, 1e-300)

    def angular_momentum(q, p):
        a = np.atleast_2d(np.asarray(q, dtype=float))
        b = np.atleast_2d(np.asarray(p, dtype=float))
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    return force, energy, angular_momentum


# --------------------------------------------------------------------------- measurements


def order_of(name: str, step_counts=None, t_end: float = 2.0) -> dict:
    """The order a method achieves on the harmonic oscillator, where the answer is exact."""
    force, energy, exact = harmonic_oscillator()
    ns = ([20, 40, 80, 160, 320, 640] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    want_q, want_p = exact(float(t_end), 1.0, 0.0)
    rows = []
    for n in ns:
        out = integrate(force, [1.0], [0.0], t_end, n, name)
        rows.append((n, max(abs(float(out["q"][-1, 0]) - want_q),
                            abs(float(out["p"][-1, 0]) - want_p))))
    errors = np.asarray([r[1] for r in rows])
    counts = np.asarray([r[0] for r in rows], dtype=float)
    keep = errors > 1e-13
    order = (float(-np.polyfit(np.log(counts[keep]), np.log(errors[keep]), 1)[0])
             if int(np.sum(keep)) >= 3 else float("nan"))
    return {"steps": counts, "errors": errors, "fitted_order": order,
            "claimed_order": STEPS[str(name).lower()][1],
            "matches": bool(abs(order - STEPS[str(name).lower()][1]) < 0.3)}


def energy_over_time(force, energy, q0, p0, t_end: float, steps: int,
                     name: str = "stormer verlet") -> dict:
    """The computed energy against time, and whether it drifts or oscillates.

    The distinction is the whole lesson. A drifting energy has a trend; a bounded one has a band.
    Both can be small over a short run, and only the trend says what happens over a long one.
    """
    out = integrate(force, q0, p0, t_end, int(steps), name)
    values = np.asarray(energy(out["q"], out["p"]), dtype=float)
    start = float(values[0])
    relative = (values - start) / max(abs(start), 1e-300)
    slope = float(np.polyfit(out["t"], relative, 1)[0])
    band = float(np.max(relative) - np.min(relative))
    return {"t": out["t"], "energy": values, "relative": relative,
            "band": band, "trend_per_unit_time": slope,
            "drift_over_the_run": abs(slope) * float(t_end),
            "final_relative": float(relative[-1]),
            "bounded": bool(abs(slope) * float(t_end) < 0.25 * max(band, 1e-300)),
            "name": name}


def drift_against_bounded(t_end: float = 200.0, steps: int = 20000) -> dict:
    """The three methods side by side on the same pendulum, over a long run.

    Explicit Euler gains energy without limit, symplectic Euler holds a band, and Stormer-Verlet
    holds a much narrower one. **The two symplectic methods are first and second order and the
    non symplectic one is also first order**, so the difference is not accuracy.
    """
    force, energy = pendulum()
    rows = []
    for name in ("explicit euler", "symplectic euler", "stormer verlet"):
        with np.errstate(over="ignore", invalid="ignore"):
            out = energy_over_time(force, energy, [1.0], [0.0], t_end, steps, name)
        rows.append((name, out["band"], out["trend_per_unit_time"],
                     out["drift_over_the_run"], out["bounded"],
                     out["final_relative"]))
    return {"names": [r[0] for r in rows],
            "band": np.asarray([r[1] for r in rows]),
            "trend_per_unit_time": np.asarray([r[2] for r in rows]),
            "drift_over_the_run": np.asarray([r[3] for r in rows]),
            "bounded": np.asarray([r[4] for r in rows]),
            "final_relative_error": np.asarray([r[5] for r in rows]),
            "t_end": float(t_end)}


def against_runge_kutta(t_end: float = 500.0, steps: int = 50000) -> dict:
    """Stormer-Verlet against RK4 on the same pendulum, at equal evaluations.

    RK4 is two orders higher and costs four evaluations per step against Verlet's two, so at
    equal cost Verlet takes twice as many steps and is still less accurate **on the trajectory**.

    It wins anyway on the quantity that matters over a long run, because RK4's energy error has
    a trend and Verlet's does not. Reporting both is the point: **neither method is better, they
    are better at different things.**
    """
    from .ivp import integrate as _integrate
    from .odesystems import pendulum as _system
    from .rungekutta import named_step

    system_f, system_energy = _system()
    force, energy = pendulum()
    n = int(steps)
    verlet_counter, rk_counter = [0], [0]
    verlet = integrate(force, [1.0], [0.0], t_end, n, "stormer verlet", verlet_counter)
    rk = _integrate(system_f, 0.0, [1.0, 0.0], t_end, n // 2, named_step("rk4"), rk_counter)
    verlet_energy = np.asarray(energy(verlet["q"], verlet["p"]), dtype=float)
    rk_energy = np.asarray(system_energy(rk["y"]), dtype=float)

    def summarise(t, values):
        start = float(values[0])
        relative = (values - start) / max(abs(start), 1e-300)
        slope = float(np.polyfit(t, relative, 1)[0])
        return (float(np.max(relative) - np.min(relative)), slope,
                abs(slope) * float(t_end))

    v_band, v_slope, v_drift = summarise(verlet["t"], verlet_energy)
    r_band, r_slope, r_drift = summarise(rk["t"], rk_energy)
    return {"verlet_evaluations": verlet_counter[0], "rk4_evaluations": rk_counter[0],
            "verlet_band": v_band, "rk4_band": r_band,
            "verlet_trend": v_slope, "rk4_trend": r_slope,
            "verlet_drift": v_drift, "rk4_drift": r_drift,
            "verlet_is_bounded": bool(v_drift < 0.25 * max(v_band, 1e-300)),
            "rk4_is_bounded": bool(r_drift < 0.25 * max(r_band, 1e-300)),
            "t_end": float(t_end)}


def area_is_preserved(name: str = "stormer verlet", h: float = 0.2,
                      corners: int = 200) -> dict:
    """The map's Jacobian determinant, which is exactly 1 for a symplectic method.

    Liouville's theorem says the exact flow preserves phase space area. A symplectic method
    preserves it in floating point, for any step, and that is the definition rather than a
    consequence.

    Measured by differencing the one step map, so what is reported includes the differencing
    error and is not a proof. A non symplectic method fails it by ``O(h^2)``, which is far above
    that noise.
    """
    force, energy, _exact = harmonic_oscillator()
    take = step_named(name)
    rng = np.random.default_rng(42)
    scale = math.sqrt(np.finfo(float).eps)
    determinants = []
    for _ in range(int(corners)):
        q = np.asarray([float(rng.uniform(-1.5, 1.5))])
        p = np.asarray([float(rng.uniform(-1.5, 1.5))])
        base_q, base_p = take(force, q, p, h)
        jac = np.empty((2, 2))
        for j, (dq, dp) in enumerate(((scale, 0.0), (0.0, scale))):
            nq, np_ = take(force, q + dq, p + dp, h)
            jac[0, j] = (float(nq[0]) - float(base_q[0])) / scale
            jac[1, j] = (float(np_[0]) - float(base_p[0])) / scale
        determinants.append(float(np.linalg.det(jac)))
    determinants = np.asarray(determinants)
    return {"determinants": determinants,
            "worst_departure_from_one": float(np.max(np.abs(determinants - 1.0))),
            "preserves_area": bool(np.max(np.abs(determinants - 1.0)) < 1e-6),
            "expected_for_explicit_euler": float(1.0 + h ** 2),
            "name": name, "h": float(h)}


def changing_the_step_destroys_the_bound(t_end: float = 400.0, base: float = 0.01,
                                         seed: int = 42) -> dict:
    """Why symplectic integration and adaptive stepping do not compose, measured.

    The backward error argument says the computed solution is the exact solution of a nearby
    Hamiltonian ``H + O(h^p)``. That nearby Hamiltonian **depends on ``h``**, so a run that keeps
    changing ``h`` keeps changing which problem it is solving exactly, and the energy error stops
    being a bounded oscillation and starts being a random walk.

    **It is irregularity that does the damage, not change.** A fixed step and a deterministic
    alternation between two steps both stay bounded, because a repeating pattern of symplectic
    maps is itself a symplectic map with its own modified Hamiltonian. A step drawn at random
    each time is not, and its energy drifts:

        step policy     band      trend per unit   drift over 400 units   verdict
        fixed          2.3e-04         2.0e-11              8.1e-09       bounded
        alternating    4.0e-04         7.2e-11              2.9e-08       bounded
        random         1.1e-03         1.4e-06              5.5e-04       drifts

    The drift is small in absolute terms over this run and it has a **trend**, which is the
    property the whole lesson is about: it grows without limit and the bounded ones do not.

    That is the practical obstruction. An adaptive controller chooses its steps from the local
    error estimate, which varies irregularly with the solution, so it produces exactly the third
    row. Codes that want both use a fixed step for the symplectic part, or a variable step in a
    transformed time variable that keeps the map symplectic.
    """
    force, energy = pendulum()
    rng = np.random.default_rng(int(seed))
    rows = []
    for policy in ("fixed", "alternating", "random"):
        q = np.asarray([1.0])
        p = np.asarray([0.0])
        t = 0.0
        ts, qs, ps = [0.0], [q.copy()], [p.copy()]
        while t < t_end:
            if policy == "fixed":
                h = base
            elif policy == "alternating":
                h = base * (0.5 if len(ts) % 2 else 1.5)
            else:
                h = base * float(rng.uniform(0.5, 1.5))
            h = min(h, t_end - t)
            if h <= 0.0:
                break
            q, p = stormer_verlet_step(force, q, p, h)
            t += h
            ts.append(t)
            qs.append(q.copy())
            ps.append(p.copy())
        times = np.asarray(ts)
        values = np.asarray(energy(np.stack(qs), np.stack(ps)), dtype=float)
        relative = (values - values[0]) / abs(values[0])
        slope = float(np.polyfit(times, relative, 1)[0])
        band = float(np.max(relative) - np.min(relative))
        drift = abs(slope) * float(t_end)
        rows.append((policy, len(ts) - 1, band, slope, drift,
                     float(relative[-1]), bool(drift < 0.25 * max(band, 1e-300))))
    return {"policy": [r[0] for r in rows],
            "steps": np.asarray([r[1] for r in rows]),
            "band": np.asarray([r[2] for r in rows]),
            "trend_per_unit_time": np.asarray([r[3] for r in rows]),
            "drift_over_the_run": np.asarray([r[4] for r in rows]),
            "final_relative": np.asarray([r[5] for r in rows]),
            "bounded": np.asarray([r[6] for r in rows]),
            "t_end": float(t_end),
            "only_the_irregular_one_drifts": bool(rows[0][6] and rows[1][6]
                                                  and not rows[2][6])}


def the_band_shrinks_with_the_step(h_values=None, t_end: float = 50.0) -> dict:
    """How the energy band depends on the step, for a fixed step symplectic run.

    The band is ``O(h^p)`` with ``p`` the method's order, so halving the step quarters it for
    Stormer-Verlet. The bands are **nested** rather than disjoint here, because this initial
    condition sits at the top of every one of them, so a run that changed step would not jump
    out of a band; what it would lose is the guarantee, which `changing_the_step_destroys_the_bound`
    measures directly.
    """
    force, energy = pendulum()
    hs = ([0.05, 0.025, 0.0125, 0.00625] if h_values is None
          else [float(v) for v in np.atleast_1d(h_values)])
    rows = []
    for h in hs:
        n = int(round(float(t_end) / h))
        out = integrate(force, [1.0], [0.0], t_end, n, "stormer verlet")
        values = np.asarray(energy(out["q"], out["p"]), dtype=float)
        relative = (values - values[0]) / abs(values[0])
        rows.append((h, float(np.min(relative)), float(np.max(relative))))
    lows = np.asarray([r[1] for r in rows])
    highs = np.asarray([r[2] for r in rows])
    widths = highs - lows
    ratios = widths[:-1] / np.maximum(widths[1:], 1e-300)
    return {"h": np.asarray([r[0] for r in rows]),
            "band_low": lows, "band_high": highs, "band_width": widths,
            "ratio_per_halving": ratios,
            "shrinks_like_h_squared": bool(np.all(np.abs(ratios - 4.0) < 0.5)),
            "bands_are_nested": bool(np.all(np.diff(lows) > 0.0))}


def kepler_orbit_stays_closed(eccentricity: float = 0.6, periods: float = 200.0,
                              steps: int = 200000) -> dict:
    """A long Kepler integration, where the difference is visible rather than statistical.

    A non symplectic method's orbit precesses: the ellipse rotates slowly because the energy
    error accumulates. A symplectic one holds the ellipse, and its only long term error is a
    drift **along** the orbit, which is a phase error rather than a shape error.
    """
    force, energy, angular_momentum = kepler()
    e = float(eccentricity)
    q0 = np.asarray([1.0 - e, 0.0])
    p0 = np.asarray([0.0, math.sqrt((1.0 + e) / (1.0 - e))])
    period = 2.0 * math.pi
    out = integrate(force, q0, p0, periods * period, int(steps), "stormer verlet")
    radius = np.linalg.norm(out["q"], axis=1)
    e_values = np.asarray(energy(out["q"], out["p"]), dtype=float)
    l_values = np.asarray(angular_momentum(out["q"], out["p"]), dtype=float)
    half = radius.size // 2
    return {"t": out["t"], "q": out["q"],
            "perihelion_early": float(np.min(radius[:half])),
            "perihelion_late": float(np.min(radius[half:])),
            "aphelion_early": float(np.max(radius[:half])),
            "aphelion_late": float(np.max(radius[half:])),
            "energy_band": float(np.max(e_values) - np.min(e_values)),
            "energy_trend": float(np.polyfit(out["t"], e_values, 1)[0]),
            "angular_momentum_drift": float(np.max(np.abs(l_values - l_values[0]))
                                            / abs(l_values[0])),
            "shape_is_held": bool(abs(float(np.max(radius[half:]))
                                      - float(np.max(radius[:half])))
                                  < 1e-3 * float(np.max(radius)))}
