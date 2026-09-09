"""Stability and stiffness: when the step size is decided by something other than accuracy.

The test equation
-----------------
Every stability statement in this part is about

    y' = lam y,     y(0) = 1,     exact solution  e^{lam t}.

Apply a one step method and the result is ``y_{n+1} = R(z) y_n`` with ``z = lam h``. The function
``R`` is the method's **stability function**, and the method reproduces the qualitative behaviour
of the solution exactly when ``|R(z)| <= 1`` wherever ``Re(z) < 0``.

That one scalar equation covers linear systems too: diagonalise ``y' = A y`` and every eigenvalue
of ``A`` must sit inside the region, so **the most negative eigenvalue sets the step for the whole
system**, however unimportant its component is.

Stiffness
---------
A problem is stiff when the eigenvalues span a wide range: some components decay in a time far
shorter than the interval of interest. Then

- **accuracy** needs a step resolving the slow component,
- **stability** needs a step resolving the fast one,

and the two differ by the eigenvalue ratio. `stiffness_ratio` measures it and
`the_step_is_set_by_stability` shows the gap directly: on a problem with eigenvalues -1 and
-1000, an explicit method needs a thousand times more steps than the answer requires.

The properties that fix it
--------------------------
- **A-stable**: the whole left half plane is inside the stability region, so no step is too
  large. Backward Euler and the trapezoid rule are; no explicit method is, ever.
- **L-stable**: A-stable and ``R(z) -> 0`` as ``Re(z) -> -infinity``, so very fast components
  are annihilated rather than merely bounded. Backward Euler is; **the trapezoid rule is not**,
  and `trapezoid_rings` shows the oscillation that causes.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .ivp import as_state


def growth_function(name: str):
    """The stability function ``R(z)`` of a named method, as a callable on complex ``z``.

    For an explicit Runge-Kutta method it is a polynomial; for an implicit one a rational
    function. Reading it off the method rather than tabulating it means a new tableau is covered
    automatically.
    """
    key = str(name).lower()
    if key == "backward euler":
        return lambda z: 1.0 / (1.0 - z)
    if key == "trapezoid":
        return lambda z: (1.0 + 0.5 * z) / (1.0 - 0.5 * z)
    if key == "implicit midpoint":
        return lambda z: (1.0 + 0.5 * z) / (1.0 - 0.5 * z)
    if key in ("bdf2",):
        # y_{n+1} = (4 y_n - y_{n-1})/3 + (2h/3) f_{n+1}; its growth factors are the roots of
        # (1 - 2z/3) r^2 - (4/3) r + 1/3, and the stability question is about the larger one
        def bdf2(z):
            a = 1.0 - 2.0 * z / 3.0
            roots = np.roots([a, -4.0 / 3.0, 1.0 / 3.0])
            return complex(roots[int(np.argmax(np.abs(roots)))])
        return bdf2

    from .rungekutta import tableau

    A, b, c, _order = tableau(key)
    s = b.size

    def explicit(z):
        stages = np.zeros(s, dtype=complex)
        for i in range(s):
            arg = 1.0 + sum(A[i, j] * stages[j] for j in range(i))
            stages[i] = z * arg
        return complex(1.0 + b @ stages)

    return explicit


#: Every method this module knows how to draw a region for.
METHODS = ("euler", "heun", "kutta third", "rk4",
           "backward euler", "trapezoid", "implicit midpoint", "bdf2")


def is_stable_at(name: str, z) -> np.ndarray:
    """Whether ``|R(z)| <= 1`` at each point of a complex array."""
    R = growth_function(name)
    grid = np.atleast_1d(np.asarray(z, dtype=complex))
    out = np.empty(grid.shape, dtype=bool)
    flat = grid.ravel()
    result = np.empty(flat.size)
    for i, value in enumerate(flat):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            try:
                result[i] = abs(R(value))
            except (ZeroDivisionError, FloatingPointError):
                result[i] = np.inf
    return (result <= 1.0 + 1e-12).reshape(grid.shape)


def stability_region(name: str, real=None, imaginary=None, points: int = 241) -> dict:
    """The stability region on a grid, plus the two numbers that summarise it."""
    lo_r, hi_r = (-6.0, 2.0) if real is None else (float(real[0]), float(real[1]))
    lo_i, hi_i = (-4.0, 4.0) if imaginary is None else (float(imaginary[0]),
                                                        float(imaginary[1]))
    n = int(points)
    x = np.linspace(lo_r, hi_r, n)
    y = np.linspace(lo_i, hi_i, n)
    X, Y = np.meshgrid(x, y)
    inside = is_stable_at(name, X + 1j * Y)
    return {"real": x, "imaginary": y, "inside": inside,
            "real_axis_limit": real_axis_limit(name),
            "imaginary_axis_limit": imaginary_axis_limit(name),
            "covers_the_left_half_plane": bool(np.all(inside[X < 0.0]))}


def real_axis_limit(name: str, tol: float = 1e-10, ceiling: float = 1e6) -> float:
    """The largest ``|z|`` on the negative real axis that is still stable.

    ``inf`` for an A-stable method. For an explicit one this is the number that decides the step
    on a stiff problem.
    """
    R = growth_function(name)

    def stable(x):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            try:
                return abs(R(complex(-x, 0.0))) <= 1.0 + 1e-12
            except (ZeroDivisionError, FloatingPointError):
                return False

    if stable(ceiling):
        return float("inf")
    lo, hi = 0.0, 1.0
    while stable(hi) and hi < ceiling:
        hi *= 2.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def imaginary_axis_limit(name: str, tol: float = 1e-10, ceiling: float = 1e6) -> float:
    """The largest ``|z|`` on the imaginary axis that is still stable.

    This is the number that matters for an oscillatory problem, where the eigenvalues are pure
    imaginary. Euler's is **zero**: it is unstable for every step on a pure oscillation, which is
    why lesson 74 exists.
    """
    R = growth_function(name)

    def stable(x):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            try:
                return abs(R(complex(0.0, x))) <= 1.0 + 1e-12
            except (ZeroDivisionError, FloatingPointError):
                return False

    if stable(ceiling):
        return float("inf")
    lo, hi = 0.0, 1.0
    while stable(hi) and hi < ceiling:
        hi *= 2.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def classify(names=None, tol: float = 1e-10) -> dict:
    """A-stability and L-stability for each method, decided by measurement.

    A-stable is tested by sampling the left half plane over many decades, which is a check
    rather than a proof and is enough to separate the families.

    L-stable additionally needs ``R(z) -> 0``, and testing that against a fixed threshold is
    wrong: BDF2's growth factor falls like ``|z|^(-1/2)``, so at ``z = -10^10`` it is still
    ``7 x 10^-6`` and any threshold at ``10^-6`` calls an L-stable method not L-stable. The test
    here compares two magnitudes decades apart and asks whether the factor is **still falling**,
    which is what the definition says.
    """
    wanted = (list(METHODS) if names is None else [str(v) for v in np.atleast_1d(names)])
    rows = []
    for name in wanted:
        R = growth_function(name)
        probes = []
        for magnitude in (1e-2, 1.0, 1e2, 1e4, 1e6):
            for angle in np.linspace(0.55 * math.pi, 1.45 * math.pi, 13):
                probes.append(magnitude * complex(math.cos(angle), math.sin(angle)))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = []
            for z in probes:
                try:
                    values.append(abs(R(z)))
                except (ZeroDivisionError, FloatingPointError):
                    values.append(np.inf)
        a_stable = bool(np.all(np.asarray(values) <= 1.0 + 1e-9))
        def magnitude_at(x):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                try:
                    return abs(R(complex(x, 0.0)))
                except (ZeroDivisionError, FloatingPointError):
                    return np.inf

        near = magnitude_at(-1e6)
        far = magnitude_at(-1e10)
        # L-stable means the factor tends to zero, so it must be small AND still shrinking
        decaying = bool(far < 0.1 and far < 0.5 * near)
        rows.append((name, a_stable, float(far), bool(a_stable and decaying), float(near)))
    return {"names": [r[0] for r in rows],
            "a_stable": np.asarray([r[1] for r in rows]),
            "magnitude_at_minus_1e6": np.asarray([r[4] for r in rows]),
            "magnitude_at_minus_1e10": np.asarray([r[2] for r in rows]),
            "l_stable": np.asarray([r[3] for r in rows]),
            "real_axis_limit": np.asarray([real_axis_limit(n) for n in wanted]),
            "imaginary_axis_limit": np.asarray([imaginary_axis_limit(n) for n in wanted])}


# --------------------------------------------------------------------------- stiffness


def stiffness_ratio(jacobian) -> dict:
    """The ratio of the fastest to the slowest decaying mode, which is what stiff means.

    A ratio near 1 is not stiff. A ratio of 1000 means an explicit method must take a thousand
    times more steps than the answer needs. The word has no threshold and the ratio is the
    quantity to report.
    """
    J = np.atleast_2d(np.asarray(jacobian, dtype=float))
    values = np.linalg.eigvals(J)
    negative = values[values.real < 0]
    if negative.size == 0:
        return {"eigenvalues": values, "ratio": 1.0, "stiff": False,
                "note": "no decaying modes, so stiffness does not apply"}
    magnitudes = np.abs(negative.real)
    ratio = float(np.max(magnitudes) / np.min(magnitudes))
    return {"eigenvalues": values, "fastest": float(np.max(magnitudes)),
            "slowest": float(np.min(magnitudes)), "ratio": ratio,
            "stiff": bool(ratio > 100.0),
            "note": "ratio of the fastest to the slowest decay rate"}


def the_step_is_set_by_stability(fast: float = -1000.0, slow: float = -1.0,
                                 t_end: float = 5.0, name: str = "rk4",
                                 tolerances=None) -> dict:
    """The step accuracy wants against the step stability allows, on a two mode problem.

    The slow component is the answer; the fast one has decayed to nothing within the first
    percent of the interval and is then irrelevant to it. **It still sets the step for the whole
    run**, because an explicit method has no way to ignore a mode it cannot resolve.

    The waste is reported across a range of tolerances, because it depends on both. Stability
    fixes one step whatever accuracy is wanted; accuracy allows a larger one the looser the
    request. So **the waste is worst when little accuracy is needed**, which is the situation a
    stiff problem is usually in: the fast transient is not wanted at all.
    """
    limit = real_axis_limit(name)
    stability_step = limit / abs(fast)
    from .rungekutta import tableau

    _A, b, _c, order = tableau(name)
    tols = ([1e-2, 1e-4, 1e-6, 1e-8, 1e-10] if tolerances is None
            else [float(v) for v in np.atleast_1d(tolerances)])
    rows = []
    for tol in tols:
        # the step the slow mode's own error term allows: |lam h|^(order) ~ tol per unit time
        accuracy_step = (tol / abs(slow) ** (order + 1)) ** (1.0 / order)
        rows.append((tol, accuracy_step,
                     int(math.ceil(t_end / accuracy_step)),
                     accuracy_step / stability_step))
    return {"fast": float(fast), "slow": float(slow),
            "stiffness_ratio": abs(fast / slow),
            "stability_step": float(stability_step),
            "steps_for_stability": int(math.ceil(t_end / stability_step)),
            "tolerance": np.asarray([r[0] for r in rows]),
            "accuracy_step": np.asarray([r[1] for r in rows]),
            "steps_for_accuracy": np.asarray([r[2] for r in rows]),
            "wasted_factor": np.asarray([r[3] for r in rows]),
            "worst_waste": float(np.max([r[3] for r in rows])),
            "method": name, "order": order}


def explicit_fails_on_a_stiff_problem(fast: float = -1000.0, slow: float = -1.0,
                                      t_end: float = 1.0, step_counts=None) -> dict:
    """Run an explicit and an implicit method on the same stiff system and compare.

    The system is diagonal with the two given eigenvalues, so the exact solution is known and
    the failure is unambiguous.
    """
    from .ivp import backward_euler_step, integrate
    from .rungekutta import named_step

    ns = ([10, 50, 100, 500, 1000, 2000, 5000] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])

    def f(t, y):
        v = as_state(y)
        return np.asarray([fast * v[0], slow * v[1]])

    want = np.asarray([math.exp(fast * t_end), math.exp(slow * t_end)])
    rows = []
    for n in ns:
        with np.errstate(over="ignore", invalid="ignore"):
            ex = integrate(f, 0.0, [1.0, 1.0], t_end, n, named_step("rk4"))
            im = integrate(f, 0.0, [1.0, 1.0], t_end, n, backward_euler_step)
        ex_value = ex["y"][-1]
        im_value = im["y"][-1]
        rows.append((n, abs(fast) * float(ex["step"]),
                     float(np.max(np.abs(ex_value - want)))
                     if np.all(np.isfinite(ex_value)) else float("inf"),
                     float(np.max(np.abs(im_value - want)))))
    explicit = np.asarray([r[2] for r in rows])
    implicit = np.asarray([r[3] for r in rows])
    usable = np.flatnonzero(np.isfinite(explicit) & (explicit < 1.0))
    return {"steps": np.asarray([r[0] for r in rows]),
            "lam_h": np.asarray([r[1] for r in rows]),
            "explicit_error": explicit, "implicit_error": implicit,
            "first_usable_explicit_steps": (int(rows[usable[0]][0]) if usable.size
                                            else None),
            "implicit_usable_everywhere": bool(np.all(implicit < 1.0))}


def trapezoid_rings(lam: float = -1000.0, steps: int = 10, t_end: float = 1.0) -> dict:
    """Why A-stability is not enough, and L-stability is what a stiff solver needs.

    The trapezoid rule is A-stable, so it never diverges. But its growth factor tends to **-1**
    as ``z -> -infinity``, not to 0, so a very fast mode is not damped: it is **inverted at every
    step** and rings forever at whatever amplitude it started with.

    Backward Euler's growth factor tends to 0, so the same mode is annihilated in one step. That
    is L-stability, and it is the property that separates a usable stiff solver from a merely
    stable one.
    """
    from .ivp import backward_euler_step, integrate

    f = lambda t, y: lam * as_state(y)
    n = int(steps)
    trap_R = growth_function("trapezoid")
    be_R = growth_function("backward euler")
    h = float(t_end) / n
    z = lam * h
    # the trapezoid rule run directly, since it is implicit and linear here
    values = [1.0]
    for _ in range(n):
        values.append(values[-1] * float(np.real(trap_R(z))))
    backward = integrate(f, 0.0, 1.0, t_end, n, backward_euler_step)
    return {"lam_h": float(z),
            "trapezoid_growth": float(np.real(trap_R(z))),
            "backward_euler_growth": float(np.real(be_R(z))),
            "trapezoid_values": np.asarray(values),
            "backward_euler_values": backward["y"][:, 0],
            "trapezoid_alternates": bool(np.all(np.asarray(values[1:]) * np.asarray(values[:-1]) < 0)),
            "trapezoid_amplitude_at_the_end": float(abs(values[-1])),
            "backward_euler_amplitude_at_the_end": float(abs(backward["y"][-1, 0])),
            "exact": math.exp(lam * float(t_end))}


def multistep_roots(alpha, beta, z) -> np.ndarray:
    """The roots of ``rho(w) - z sigma(w)``, which govern a multistep method's stability at ``z``.

    A one step method has a single growth factor ``R(z)``. A ``k`` step method has ``k`` of them,
    the roots of this polynomial, and the method is stable at ``z`` when all of them are inside
    the unit disc. Setting ``z = 0`` recovers lesson 71's root condition, so zero-stability is
    exactly stability at the origin.
    """
    a = np.asarray([complex(v) for v in alpha])
    b = np.asarray([complex(v) for v in beta])
    # a k step method may be written with a shorter alpha than beta or the other way round, as
    # AM3 is: alpha has two entries and beta three. Both describe the same polynomial degree, so
    # the shorter one is padded with the trailing zeros it left out.
    width = max(a.size, b.size)
    a = np.concatenate([a, np.zeros(width - a.size, dtype=complex)])
    b = np.concatenate([b, np.zeros(width - b.size, dtype=complex)])
    return np.roots(a - complex(z) * b)


def multistep_is_stable_at(alpha, beta, z, tol: float = 1e-9) -> bool:
    """Whether every root is inside the closed unit disc, with those on it simple."""
    roots = multistep_roots(alpha, beta, z)
    if roots.size == 0:
        return True
    moduli = np.abs(roots)
    if float(np.max(moduli)) > 1.0 + tol:
        return False
    on = np.flatnonzero(moduli > 1.0 - tol)
    for i in range(on.size):
        for j in range(i + 1, on.size):
            if abs(roots[on[i]] - roots[on[j]]) < 1e-6:
                return False
    return True


def stability_angle(alpha, beta, radii=None, tol: float = 1e-9,
                    resolution: float = 1e-3) -> float:
    """The largest half angle ``a`` for which the wedge ``|arg(-z)| < a`` is entirely stable.

    This is **A(alpha)-stability**, the weaker property a method settles for when it cannot have
    A-stability. An angle of 90 degrees is A-stability itself; an angle of 0 means the method is
    not even stable on the negative real axis.

    Found by bisection on the angle, testing a logarithmic spread of radii at each one, so the
    answer does not depend on a grid. The radii have to span many decades: a method can be stable
    close to the origin and unstable far out, which is the failure the whole idea is about.
    """
    rs = ([10.0 ** e for e in np.linspace(-4.0, 8.0, 121)] if radii is None
          else [float(v) for v in np.atleast_1d(radii)])

    def stable_along(degrees):
        theta = math.radians(float(degrees))
        for r in rs:
            z = -r * complex(math.cos(theta), math.sin(theta))
            if not multistep_is_stable_at(alpha, beta, z, tol):
                return False
        return True

    if not stable_along(0.0):
        return 0.0
    lo, hi = 0.0, 90.0
    if stable_along(90.0 - float(resolution)):
        return 90.0
    while hi - lo > float(resolution):
        mid = 0.5 * (lo + hi)
        if stable_along(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


#: The published A(alpha) angles of the BDF family, in degrees, for checking against.
BDF_PUBLISHED_ANGLES = {1: 90.0, 2: 90.0, 3: 86.03, 4: 73.35, 5: 51.84, 6: 17.84, 7: 0.0}


def bdf_angles(orders=None) -> dict:
    """The BDF family's stability angles, against the published table.

    BDF1 and BDF2 are A-stable, which the second Dahlquist barrier says is as far as order can go.
    Past that the family gives up A-stability and keeps a **wedge**, and the wedge closes:

        order       1      2      3      4      5      6      7
        angle    90.0   90.0  86.03  73.35  51.84  17.84    0.0

    BDF7 is not stable even on the negative real axis, which is why every production stiff solver
    stops at 6. The barrier is not a subtlety here; it is the whole shape of the table.

    Measured by bisection on the angle rather than read off a grid, and compared against the
    published values, which are the check that the whole construction is right.
    """
    from .multistep import bdf_coefficients

    ks = ([1, 2, 3, 4, 5, 6, 7] if orders is None else [int(v) for v in np.atleast_1d(orders)])
    rows = []
    for k in ks:
        alpha, beta = bdf_coefficients(k)
        angle = stability_angle(alpha, beta)
        published = BDF_PUBLISHED_ANGLES.get(k, float("nan"))
        rows.append((k, angle, published, abs(angle - published), angle > 89.9,
                     bool(multistep_is_stable_at(alpha, beta, -1.0))))
    angles = np.asarray([r[1] for r in rows])
    gaps = np.asarray([r[3] for r in rows])
    return {"order": np.asarray([r[0] for r in rows]),
            "angle": angles,
            "published_angle": np.asarray([r[2] for r in rows]),
            "gap": gaps,
            "a_stable": np.asarray([r[4] for r in rows]),
            "stable_on_the_negative_real_axis": np.asarray([r[5] for r in rows]),
            "worst_gap": float(np.max(gaps)) if gaps.size else float("nan"),
            "matches_the_published_table": bool(np.all(gaps < 1.5)),
            "a_stable_only_up_to": (int(max((r[0] for r in rows if r[4]), default=0))),
            "the_wedge_closes": bool(np.all(np.diff(angles) <= 0.0))}


def second_dahlquist_barrier() -> dict:
    """No A-stable linear multistep method has order above 2, and the trapezoid rule is best.

    The barrier is why the whole stiff solver literature is about **backward differentiation
    formulas**, which give up A-stability past order 2 in exchange for order, and about implicit
    Runge-Kutta methods, which are not linear multistep methods and so are not bound by it.

    Reported here by measuring the A-stability of every named multistep method **from its own
    coefficients**, using `multistep_is_stable_at` rather than the one step growth function, so a
    three step method can be tested the same way a one step one is. `bdf_angles` then measures
    what the family settles for instead.
    """
    from .multistep import NAMED, method, order_conditions

    rows = []
    for name in ("euler", "trapezoid", "backward euler", "ab2", "am3", "bdf2", "bdf3", "ab4"):
        if name not in NAMED:
            continue
        alpha, beta, claimed = method(name)
        oc = order_conditions(alpha, beta)
        rows.append((name, oc["order"], multistep_a_stable(alpha, beta)))
    orders = np.asarray([r[1] for r in rows])
    stable = np.asarray([r[2] for r in rows])
    highest = int(np.max(orders[stable])) if bool(np.any(stable)) else 0
    return {"names": [r[0] for r in rows],
            "order": orders,
            "a_stable": stable,
            "barrier": 2,
            "highest_a_stable_order_found": highest,
            "barrier_holds": bool(highest <= 2),
            "note": ("no A-stable linear multistep method exceeds order 2; the trapezoid rule "
                     "attains it with the smallest error constant")}


def multistep_a_stable(alpha, beta, tol: float = 1e-9) -> bool:
    """A-stability of a linear multistep method, measured from its stability angle.

    A-stable means the whole left half plane is stable, which is a 90 degree wedge. Measuring the
    angle rather than sampling a rectangle is what makes this work for methods whose region is a
    wedge closing slowly, where a rectangular grid would report whichever answer its corners
    happened to give.
    """
    return bool(stability_angle(alpha, beta, tol=tol) > 89.9)
