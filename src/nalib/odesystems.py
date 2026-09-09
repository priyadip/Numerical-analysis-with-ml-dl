"""Systems and higher order equations: everything so far, applied to problems worth solving.

Reduction to first order
------------------------
An equation of order ``m`` becomes a system of ``m`` first order equations by naming the
derivatives:

    y'' = g(t, y, y')   becomes   u = (y, y'),   u' = (u_1, g(t, u_0, u_1)).

**Nothing else changes.** Every method in Part 10 was written against a vector state from the
start, so the same code runs. That is not a coincidence: it is why `nalib.ivp.as_state` exists
and why the drivers never index a scalar.

The problems
------------
Five systems, each chosen because it fails a different way:

- **the pendulum**, where the energy drifts and lesson 74 fixes it,
- **the two body problem**, where the orbit spirals for the same reason,
- **the Lorenz system**, where the solution is chaotic and no method tracks it for long,
- **Hodgkin-Huxley**, which is stiff and needs lesson 72,
- **the Tacoma Narrows model**, where a parameter crossing changes the answer qualitatively.

Together they make the point that a solver's error is not one number: it can be small and in the
wrong place, or large and harmless, and only knowing the problem says which.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .ivp import as_state, integrate
from .rungekutta import named_step


def to_first_order(g, order: int):
    """Turn ``y^(m) = g(t, y, y', ..., y^(m-1))`` into a first order system.

    The returned ``f`` takes a state ``(y, y', ..., y^(m-1))`` and returns its derivative, which
    is that state shifted up by one with ``g`` appended.
    """
    m = int(order)
    if m < 1:
        raise ValueError(f"order must be at least 1, got {m}")

    def f(t, u):
        state = as_state(u)
        if state.size != m:
            raise ValueError(f"state has {state.size} components, expected {m}")
        out = np.empty(m)
        out[:-1] = state[1:]
        out[-1] = float(np.ravel(g(t, *state))[0])
        return out

    return f


# --------------------------------------------------------------------------- the problems


def pendulum(length: float = 1.0, gravity: float = 9.81, damping: float = 0.0):
    """The nonlinear pendulum, ``theta'' = -(g/L) sin(theta) - c theta'``.

    Undamped it is Hamiltonian, so its energy is exactly conserved and any drift in the computed
    energy is entirely the method's. That makes it the cleanest diagnostic in Part 10.
    """
    def f(t, y):
        v = as_state(y)
        return np.asarray([v[1], -(gravity / length) * math.sin(v[0]) - damping * v[1]])

    def energy(y):
        v = np.atleast_2d(np.asarray(y, dtype=float))
        return (0.5 * (length * v[:, 1]) ** 2
                + gravity * length * (1.0 - np.cos(v[:, 0])))

    return f, energy


def small_angle_pendulum(length: float = 1.0, gravity: float = 9.81):
    """The linearised pendulum, whose solution is known exactly.

    Used to measure a method's order on an oscillatory system, where the exact answer is
    available and the qualitative failure of lesson 74 is not yet in the way.
    """
    omega = math.sqrt(gravity / length)

    def f(t, y):
        v = as_state(y)
        return np.asarray([v[1], -omega ** 2 * v[0]])

    def exact(t):
        return np.asarray([math.cos(omega * t), -omega * math.sin(omega * t)])

    return f, exact, omega


def two_body(mass: float = 1.0):
    """The planar two body problem in the heavy centre approximation.

    State is ``(x, y, vx, vy)``. Energy and angular momentum are both conserved exactly, and
    both are measured, because a method can conserve one and not the other.
    """
    def f(t, state):
        v = as_state(state)
        x, y, vx, vy = v[0], v[1], v[2], v[3]
        r = math.hypot(x, y)
        pull = -mass / max(r, 1e-300) ** 3
        return np.asarray([vx, vy, pull * x, pull * y])

    def energy(state):
        v = np.atleast_2d(np.asarray(state, dtype=float))
        r = np.hypot(v[:, 0], v[:, 1])
        return 0.5 * (v[:, 2] ** 2 + v[:, 3] ** 2) - mass / np.maximum(r, 1e-300)

    def angular_momentum(state):
        v = np.atleast_2d(np.asarray(state, dtype=float))
        return v[:, 0] * v[:, 3] - v[:, 1] * v[:, 2]

    return f, energy, angular_momentum


def circular_orbit_start(radius: float = 1.0, mass: float = 1.0) -> np.ndarray:
    """The initial state of a circular orbit, whose exact solution is a closed circle."""
    speed = math.sqrt(mass / radius)
    return np.asarray([radius, 0.0, 0.0, speed])


def elliptical_orbit_start(eccentricity: float = 0.6, mass: float = 1.0) -> np.ndarray:
    """A Kepler ellipse starting at perihelion, which is the hard case for a fixed step."""
    e = float(eccentricity)
    return np.asarray([1.0 - e, 0.0, 0.0, math.sqrt(mass * (1.0 + e) / (1.0 - e))])


def lorenz(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
    """The Lorenz system, whose solution is chaotic at the classical parameters.

    Nearby trajectories separate exponentially, at a rate given by the largest Lyapunov exponent
    of about 0.905 for these parameters. `divergence_is_the_problem` measures it and shows that
    the separation is a property of the equations, not of the method.
    """
    def f(t, y):
        v = as_state(y)
        return np.asarray([sigma * (v[1] - v[0]),
                           v[0] * (rho - v[2]) - v[1],
                           v[0] * v[1] - beta * v[2]])

    return f


def hodgkin_huxley(current: float = 10.0):
    """The Hodgkin-Huxley neuron, the classic stiff system from mathematical biology.

    State is ``(V, m, h, n)``: membrane voltage and three gating variables whose time constants
    differ by two orders of magnitude, which is what makes it stiff. `stiffness_along_the_path`
    measures the ratio along a trajectory rather than at one point.
    """
    g_na, g_k, g_l = 120.0, 36.0, 0.3
    e_na, e_k, e_l = 50.0, -77.0, -54.387
    capacitance = 1.0

    def rates(v):
        # the classical alpha and beta functions, guarded at their removable singularities
        def safe(numerator, denominator):
            return np.where(np.abs(denominator) < 1e-9, 1.0, numerator / denominator)

        am = safe(0.1 * (v + 40.0), 1.0 - np.exp(-(v + 40.0) / 10.0))
        bm = 4.0 * np.exp(-(v + 65.0) / 18.0)
        ah = 0.07 * np.exp(-(v + 65.0) / 20.0)
        bh = 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
        an = safe(0.01 * (v + 55.0), 1.0 - np.exp(-(v + 55.0) / 10.0))
        bn = 0.125 * np.exp(-(v + 65.0) / 80.0)
        return am, bm, ah, bh, an, bn

    def f(t, y):
        s = as_state(y)
        v, m, h, n = s[0], s[1], s[2], s[3]
        am, bm, ah, bh, an, bn = rates(v)
        ina = g_na * m ** 3 * h * (v - e_na)
        ik = g_k * n ** 4 * (v - e_k)
        il = g_l * (v - e_l)
        return np.asarray([(current - ina - ik - il) / capacitance,
                           am * (1.0 - m) - bm * m,
                           ah * (1.0 - h) - bh * h,
                           an * (1.0 - n) - bn * n])

    def resting_state():
        v = -65.0
        am, bm, ah, bh, an, bn = rates(v)
        return np.asarray([v, am / (am + bm), ah / (ah + bh), an / (an + bn)])

    return f, resting_state()


def tacoma_narrows(wind: float = 80.0, damping: float = 0.01, stiffness: float = 0.2,
                   length: float = 6.0, mass_ratio: float = 0.2):
    """McKenna's model of the Tacoma Narrows bridge deck: vertical motion coupled to twist.

    State is ``(y, y', theta, theta')``. The nonlinearity is the cable's inability to push, and
    the interest is that a small twist decays at some wind speeds and grows at others, so the
    answer changes qualitatively with a parameter rather than gradually.

    It is tempting to call the changeover a critical speed. It is not one:
    `a_parameter_crossing_changes_the_answer` measures the growth and finds it rising and then
    falling again, so what the sweep passes through is a resonance between the forcing and the
    twist mode, not a threshold.
    """
    def f(t, state):
        s = as_state(state)
        y, dy, theta, dtheta = s[0], s[1], s[2], s[3]
        a = math.exp(stiffness * (y - length * math.sin(theta)))
        b = math.exp(stiffness * (y + length * math.sin(theta)))
        return np.asarray([
            dy,
            -damping * dy - (stiffness / mass_ratio) * (a + b - 2.0) + 0.2 * wind * math.sin(4.0 * t),
            dtheta,
            -damping * dtheta + (6.0 * math.cos(theta) / length) * (stiffness / mass_ratio)
            * (a - b)])

    return f


# --------------------------------------------------------------------------- measurements


def order_on_a_system(exact=None, step_counts=None, name: str = "rk4") -> dict:
    """A method's order, measured on the linear pendulum where the answer is known.

    Reducing a second order equation to a system does not change the order, which is worth
    confirming rather than assuming: a mistake in the reduction usually shows up as a lost
    order rather than as a wrong answer.
    """
    from .ivp import error_against_step

    f, closed_form, omega = small_angle_pendulum()
    use = closed_form if exact is None else exact
    ns = ([20, 40, 80, 160, 320, 640] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    out = error_against_step(f, use, 0.0, [1.0, 0.0], 2.0, ns, named_step(name))
    from .rungekutta import tableau

    _A, _b, _c, claimed = tableau(name)
    return dict(out, claimed_order=claimed, omega=omega,
                matches=bool(abs(out["fitted_order"] - claimed) < 0.3))


def reduction_is_faithful(step_counts=None) -> dict:
    """The reduced system against the closed form of the original second order equation.

    ``y'' = -omega^2 y`` has the solution ``cos(omega t)``. Solving the system and reading off
    its first component must reproduce that, and the second component must be its derivative.

    The second claim is checked as the **exact identity** ``f(t, u)[0] = u[1]``, which the
    reduction guarantees at every state whatever the solver does. Checking it by differencing
    the computed solution instead gives an answer of ``8 x 10^-4`` that shrinks like ``h^2``,
    which is a measurement of `numpy.gradient` and not of the reduction.
    """
    f, exact, omega = small_angle_pendulum()
    ns = ([20, 40, 80, 160] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    rows = []
    for n in ns:
        out = integrate(f, 0.0, [1.0, 0.0], 2.0, n, named_step("rk4"))
        want = np.stack([exact(float(v)) for v in out["t"]])
        position = float(np.max(np.abs(out["y"][:, 0] - want[:, 0])))
        velocity = float(np.max(np.abs(out["y"][:, 1] - want[:, 1])))
        # The reduction's defining identity: f(t, u)[0] must BE u[1], exactly, at every state
        # visited. Checking it by differencing the computed solution instead measures the
        # differencing, which is second order in h and says nothing about the reduction.
        identity = 0.0
        for state in out["y"]:
            identity = max(identity, abs(float(f(0.0, state)[0]) - float(state[1])))
        rows.append((n, position, velocity, identity))
    return {"steps": np.asarray([r[0] for r in rows]),
            "position_error": np.asarray([r[1] for r in rows]),
            "velocity_error": np.asarray([r[2] for r in rows]),
            "reduction_identity_residual": np.asarray([r[3] for r in rows]),
            "identity_is_exact": bool(max(r[3] for r in rows) == 0.0)}


def energy_drift(f, energy, y0, t_end: float, steps: int, name: str = "rk4") -> dict:
    """How far a method's computed energy wanders on a conservative problem.

    The exact energy is constant, so every departure is the method's. Reporting the drift as a
    fraction of the initial energy, and its trend, separates a method that oscillates about the
    truth from one that walks away from it, which is lesson 74's whole subject.
    """
    out = integrate(f, 0.0, y0, t_end, int(steps), named_step(name))
    values = np.asarray(energy(out["y"]), dtype=float)
    start = float(values[0])
    relative = (values - start) / max(abs(start), 1e-300)
    slope = float(np.polyfit(out["t"], relative, 1)[0]) if out["t"].size > 2 else 0.0
    return {"t": out["t"], "energy": values, "relative_drift": relative,
            "initial": start, "final": float(values[-1]),
            "worst": float(np.max(np.abs(relative))),
            "trend_per_unit_time": slope,
            "drifts_one_way": bool(abs(slope) * float(t_end)
                                   > 0.5 * float(np.max(np.abs(relative))))}


def orbit_closes(eccentricity: float = 0.0, periods: float = 10.0, steps: int = 2000,
                 name: str = "rk4") -> dict:
    """Does the computed orbit return to where it started?

    A circular orbit is exactly periodic, so the gap between the final and initial states is the
    method's error with nothing else in it. For an eccentric orbit the period is still known by
    Kepler's third law, so the same check works.
    """
    f, energy, angular_momentum = two_body()
    if eccentricity == 0.0:
        y0 = circular_orbit_start()
        semi_major = 1.0
    else:
        y0 = elliptical_orbit_start(eccentricity)
        semi_major = 1.0
    period = 2.0 * math.pi * semi_major ** 1.5
    out = integrate(f, 0.0, y0, periods * period, int(steps), named_step(name))
    gap = float(np.max(np.abs(out["y"][-1] - np.asarray(y0))))
    e_values = np.asarray(energy(out["y"]), dtype=float)
    l_values = np.asarray(angular_momentum(out["y"]), dtype=float)
    radius = np.hypot(out["y"][:, 0], out["y"][:, 1])
    return {"t": out["t"], "y": out["y"], "period": period,
            "closure_gap": gap,
            "energy_drift": float(abs(e_values[-1] - e_values[0]) / abs(e_values[0])),
            "angular_momentum_drift": float(abs(l_values[-1] - l_values[0])
                                            / max(abs(l_values[0]), 1e-300)),
            "radius_range": float(np.max(radius) - np.min(radius)),
            "eccentricity": float(eccentricity)}


def divergence_is_the_problem(separation: float = 1e-8, t_end: float = 30.0,
                              steps: int = 30000) -> dict:
    """Two Lorenz trajectories from nearby starts, and the rate they separate.

    The separation grows like ``e^(lambda t)`` with ``lambda`` about 0.905, so an initial gap of
    ``10^-8`` reaches order 1 in about 21 time units whatever the method.

    **That is a property of the equations and not of the solver.** No amount of accuracy extends
    the horizon by more than the logarithm of the improvement, so halving the error buys 0.77
    extra time units. Reporting a Lorenz trajectory at ``t = 50`` as though it were the solution
    is meaningless, and reporting the statistics of the attractor is not.
    """
    f = lorenz()
    base = np.asarray([1.0, 1.0, 20.0])
    nudged = base + np.asarray([float(separation), 0.0, 0.0])
    a = integrate(f, 0.0, base, t_end, int(steps), named_step("rk4"))
    b = integrate(f, 0.0, nudged, t_end, int(steps), named_step("rk4"))
    gap = np.sqrt(np.sum((a["y"] - b["y"]) ** 2, axis=1))
    growing = np.flatnonzero((gap > 10.0 * separation) & (gap < 1.0))
    rate = float("nan")
    if growing.size >= 10:
        rate = float(np.polyfit(a["t"][growing], np.log(gap[growing]), 1)[0])
    reaches_one = np.flatnonzero(gap >= 1.0)
    return {"t": a["t"], "separation": gap, "initial_separation": float(separation),
            "fitted_lyapunov": rate,
            "time_to_order_one": (float(a["t"][reaches_one[0]]) if reaches_one.size
                                  else float("nan")),
            "extra_time_per_halving": (math.log(2.0) / rate if rate == rate and rate > 0
                                       else float("nan"))}


def stiffness_along_the_path(t_end: float = 20.0, steps: int = 20000,
                             samples: int = 40) -> dict:
    """The Hodgkin-Huxley stiffness ratio, measured along a trajectory rather than at a point.

    Stiffness is a local property and it changes: the neuron is mild at rest and violently stiff
    during the spike. A solver that chose its step from the resting state would be unstable
    within a millisecond of the spike starting.
    """
    from .stability import stiffness_ratio

    f, y0 = hodgkin_huxley()
    out = integrate(f, 0.0, y0, t_end, int(steps), named_step("rk4"))
    picks = np.linspace(0, out["t"].size - 1, int(samples)).astype(int)
    ratios, voltages = [], []
    scale = math.sqrt(np.finfo(float).eps)
    for i in picks:
        state = out["y"][i]
        n = state.size
        jac = np.empty((n, n))
        base = as_state(f(float(out["t"][i]), state))
        for j in range(n):
            bumped = state.copy()
            step = scale * max(abs(state[j]), 1.0)
            bumped[j] += step
            jac[:, j] = (as_state(f(float(out["t"][i]), bumped)) - base) / step
        ratios.append(stiffness_ratio(jac)["ratio"])
        voltages.append(float(state[0]))
    ratios = np.asarray(ratios)
    return {"t": out["t"][picks], "voltage": np.asarray(voltages),
            "stiffness_ratio": ratios,
            "smallest": float(np.min(ratios)), "largest": float(np.max(ratios)),
            "varies_by": float(np.max(ratios) / max(np.min(ratios), 1e-300)),
            "spike_voltage": float(np.max(out["y"][:, 0]))}


def a_parameter_crossing_changes_the_answer(winds=None, t_end: float = 200.0,
                                            steps: int = 40000) -> dict:
    """The Tacoma model's twist, grown or damped depending on the wind speed.

    A small initial twist either decays or grows, and which it does changes with the wind speed.
    **The transition is in the equations**, so a solver that is accurate on both sides is still
    reporting two qualitatively different answers, and no error estimate mentions the change.

    **The growth is not monotone in the wind speed**, which the phrase "critical speed"
    obscures. Measured at damping 0.01 over 200 time units, the ratio of late to early twist
    amplitude runs

        wind      40     60     80    100    140    200
        growth  0.62   0.59   0.73   1.31   17.6   4.39

    so it first exceeds 1 between 80 and 100, peaks near 140 and falls again by 200. The forcing
    is periodic and the response depends on how its frequency sits against the twist mode, so
    what is being crossed is a resonance rather than a threshold. ``growth_is_monotone`` says
    whether the sweep rose the whole way, and ``crossings_of_one`` counts how many times it
    passed the growing/decaying line. On the default sweep those read False and 1.
    """
    ws = ([40.0, 60.0, 80.0, 100.0, 140.0, 200.0] if winds is None
          else [float(v) for v in np.atleast_1d(winds)])
    rows = []
    for wind in ws:
        f = tacoma_narrows(wind=wind)
        out = integrate(f, 0.0, [0.0, 0.0, 1e-3, 0.0], t_end, int(steps), named_step("rk4"))
        theta = out["y"][:, 2]
        half = theta.size // 2
        early = float(np.max(np.abs(theta[:half])))
        late = float(np.max(np.abs(theta[half:])))
        rows.append((wind, early, late, late / max(early, 1e-300)))
    growth = np.asarray([r[3] for r in rows])
    grew = [r[0] for r in rows if r[3] > 1.0]
    crossings = int(np.sum(np.diff((growth > 1.0).astype(int)) != 0))
    return {"wind": np.asarray([r[0] for r in rows]),
            "early_amplitude": np.asarray([r[1] for r in rows]),
            "late_amplitude": np.asarray([r[2] for r in rows]),
            "growth": growth,
            "first_wind_that_grows": (grew[0] if grew else None),
            "crossings_of_one": crossings,
            "growth_is_monotone": bool(np.all(np.diff(growth) > 0.0)),
            "note": ("more than one crossing means there is no single critical speed, only a "
                     "resonance the sweep passes through")}
