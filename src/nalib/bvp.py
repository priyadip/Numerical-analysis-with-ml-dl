"""Boundary value problems: when the conditions are at both ends instead of one.

What changes
------------
Every method in lessons 67 to 74 marched. It knew everything at ``t0`` and stepped forward, and
the only question was how much accuracy each step cost. A boundary value problem gives you part
of the information at ``a`` and part at ``b``, so there is nothing to march from and the whole
solution has to be found at once.

Two ways out, and they fail differently
---------------------------------------
**Shooting** turns the boundary value problem back into an initial value problem by guessing the
missing initial condition and adjusting the guess until the far boundary comes out right. That
makes it a root finding problem in one unknown, so lessons 8 to 12 apply directly, and the whole
of Part 10 applies to the inner solve. It is easy to write and it reuses everything.

Its weakness is that the map from guess to far boundary can be violently ill conditioned. For
``y'' = lam^2 y`` on ``[0, 1]`` the sensitivity is ``sinh(lam)/lam``, so at ``lam = 20`` a change
of ``10^-16`` in the guess moves the far end by ``10^-8``, and at ``lam = 40`` no guess in double
precision hits the target at all. `shooting_amplifies_the_guess` measures it.

**Finite differences** replace the derivatives by difference quotients at every interior node and
solve the resulting linear or nonlinear system in one go. There is no marching and nothing to
amplify, so the ill conditioning above simply does not arise. The price is a system to solve, but
it is tridiagonal, so lesson 22's Thomas algorithm does it in ``O(n)``.

The subtleties that are actually worth knowing
----------------------------------------------
* A BVP can have **no solution or infinitely many**, unlike an IVP with a Lipschitz right hand
  side. `existence_can_fail` measures the matrix becoming singular as the problem approaches a
  resonance, and `at_a_resonance_refining_makes_it_worse` measures shooting meeting the same
  situation and dividing by its own truncation error instead of announcing anything.
* The tridiagonal matrix is diagonally dominant only when ``h |p| < 2``. Above that the central
  difference solution **oscillates**, and refining the grid is the fix rather than a better
  solver. `oscillation_when_the_cell_number_is_too_big` measures it.
* On an **unequal** grid the truncation error is only first order, and the solution error is
  still second order anyway. That is supraconvergence, and
  `supraconvergence_on_an_unequal_grid` measures both so the gap is visible.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .banded import thomas
from .ivp import as_state, integrate
from .rungekutta import named_step


# --------------------------------------------------------------------------- test problems


def textbook_linear_problem() -> dict:
    """``y'' = -(2/x) y' + (2/x^2) y + sin(ln x)/x^2`` on ``[1, 2]``, ``y(1) = 1``, ``y(2) = 2``.

    The standard linear BVP test case. Its solution is
    ``c1 x + c2 x^-2 - (3/10) sin(ln x) - (1/10) cos(ln x)``, and the two constants are found
    here by solving the 2 by 2 boundary system rather than being typed in, so the exact solution
    satisfies the boundary conditions to machine precision by construction.
    """
    def particular(x):
        return -0.3 * np.sin(np.log(x)) - 0.1 * np.cos(np.log(x))

    a, b, alpha, beta = 1.0, 2.0, 1.0, 2.0
    M = np.asarray([[a, a ** -2.0], [b, b ** -2.0]])
    rhs = np.asarray([alpha - float(particular(a)), beta - float(particular(b))])
    c1, c2 = np.linalg.solve(M, rhs)

    def exact(x):
        v = np.asarray(x, dtype=float)
        return c1 * v + c2 * v ** -2.0 + particular(v)

    return {"p": lambda x: -2.0 / x, "q": lambda x: 2.0 / x ** 2,
            "r": lambda x: np.sin(np.log(x)) / x ** 2,
            "a": a, "b": b, "alpha": alpha, "beta": beta, "exact": exact,
            "constants": (float(c1), float(c2))}


def exponential_problem(rate: float = 10.0) -> dict:
    """``y'' = rate^2 y`` on ``[0, 1]`` with ``y(0) = 1`` and ``y(1) = 0``.

    The solution ``sinh(rate (1 - x)) / sinh(rate)`` is a boundary layer at the left end, and
    the homogeneous solutions grow and decay like ``e^(+-rate x)``. That growing one is exactly
    what shooting amplifies, so this is the problem that separates the two methods.
    """
    lam = float(rate)

    def exact(x):
        v = np.asarray(x, dtype=float)
        return np.sinh(lam * (1.0 - v)) / math.sinh(lam)

    return {"p": lambda x: np.zeros_like(np.asarray(x, dtype=float)),
            "q": lambda x: np.full_like(np.asarray(x, dtype=float), lam ** 2),
            "r": lambda x: np.zeros_like(np.asarray(x, dtype=float)),
            "a": 0.0, "b": 1.0, "alpha": 1.0, "beta": 0.0, "exact": exact, "rate": lam}


def convection_diffusion_problem(viscosity: float = 0.01) -> dict:
    """``eps y'' + y' = 0`` on ``[0, 1]``, ``y(0) = 0``, ``y(1) = 1``, written as ``y'' = p y'``.

    The exact solution ``(1 - e^(-x/eps)) / (1 - e^(-1/eps))`` is flat over most of the interval
    and turns through a layer of width ``eps`` at the right end. It is the standard model for
    what happens when convection beats diffusion, and its discrete version is where the cell
    number condition ``h < 2 eps`` comes from.
    """
    eps = float(viscosity)

    def exact(x):
        v = np.asarray(x, dtype=float)
        # written to avoid overflow for small eps, where both exponentials underflow
        return (1.0 - np.exp(-v / eps)) / (1.0 - math.exp(-1.0 / eps))

    return {"p": lambda x: np.full_like(np.asarray(x, dtype=float), -1.0 / eps),
            "q": lambda x: np.zeros_like(np.asarray(x, dtype=float)),
            "r": lambda x: np.zeros_like(np.asarray(x, dtype=float)),
            "a": 0.0, "b": 1.0, "alpha": 0.0, "beta": 1.0, "exact": exact,
            "viscosity": eps}


def textbook_nonlinear_problem() -> dict:
    """``y'' = (32 + 2x^3 - y y')/8`` on ``[1, 3]``, ``y(1) = 17``, ``y(3) = 43/3``.

    Its solution is ``x^2 + 16/x``, which can be checked by substitution rather than trusted.
    The partial derivatives ``f_y = -y'/8`` and ``f_y' = -y/8`` are supplied because the
    nonlinear finite difference method wants the Jacobian of the discrete system.
    """
    def f(x, y, dy):
        return (32.0 + 2.0 * np.asarray(x, dtype=float) ** 3 - y * dy) / 8.0

    def f_y(x, y, dy):
        return -np.asarray(dy, dtype=float) / 8.0

    def f_dy(x, y, dy):
        return -np.asarray(y, dtype=float) / 8.0

    def exact(x):
        v = np.asarray(x, dtype=float)
        return v ** 2 + 16.0 / v

    return {"f": f, "f_y": f_y, "f_dy": f_dy, "a": 1.0, "b": 3.0,
            "alpha": 17.0, "beta": 43.0 / 3.0, "exact": exact}


# --------------------------------------------------------------------------- shooting


def miss(f, a: float, b: float, alpha: float, slope: float, steps: int = 200,
         name: str = "rk4") -> float:
    """Solve the IVP with the given initial slope and report the value reached at ``b``.

    The boundary value problem is solved when this equals ``beta``, so the whole of shooting is
    finding a root of ``miss(s) - beta``.
    """
    out = integrate(f, float(a), [float(alpha), float(slope)], float(b), int(steps),
                    named_step(name))
    return float(out["y"][-1, 0])


def linear_shooting(p, q, r, a: float, b: float, alpha: float, beta: float,
                    steps: int = 200, name: str = "rk4",
                    resonance_tol: float = 1e-6) -> dict:
    """Shooting for a linear problem, where **two** IVPs give the answer with no iteration.

    For ``y'' = p y' + q y + r`` the map from initial slope to the value at ``b`` is affine, so
    solving once with slope 0 and once with slope 1 determines it exactly. The combination

        ``y = u + ((beta - u(b)) / v(b)) v``

    satisfies the equation because ``v`` solves the homogeneous problem, and satisfies both
    boundary conditions by construction. No root finder is involved and no iteration count
    appears, which is worth noticing: linearity is what buys that.

    **What happens at a resonance is not a division by zero.** When the boundary value problem
    has no unique solution, ``v(b)`` is exactly zero in exact arithmetic, and what the integrator
    returns instead is its own truncation error. For ``y'' + pi^2 y = 0`` on ``[0, 1]`` that is
    ``5 x 10^-10`` at 200 RK4 steps and ``5 x 10^-14`` at 2000. So the correction is divided by
    a number that **shrinks as the step is refined**, and the method returns a large confident
    answer that gets worse the harder you work.

    ``relative_end`` is ``|v(b)|`` measured against the size of ``v`` over the interval, and
    ``near_a_resonance`` says whether it fell below ``resonance_tol``. Checking it is the only
    protection there is, because nothing else about the run looks wrong.
    """
    def full(t, state):
        u = as_state(state)
        return np.asarray([u[1], float(p(t)) * u[1] + float(q(t)) * u[0] + float(r(t))])

    def homogeneous(t, state):
        u = as_state(state)
        return np.asarray([u[1], float(p(t)) * u[1] + float(q(t)) * u[0]])

    step = named_step(name)
    first = integrate(full, float(a), [float(alpha), 0.0], float(b), int(steps), step)
    second = integrate(homogeneous, float(a), [0.0, 1.0], float(b), int(steps), step)
    end = float(second["y"][-1, 0])
    scale = float(np.max(np.abs(second["y"][:, 0])))
    if end == 0.0:
        raise ZeroDivisionError(
            "the homogeneous solution is exactly zero at b, so this problem has no unique "
            "solution and shooting cannot correct the guess")
    relative = abs(end) / max(scale, 1e-300)
    weight = (float(beta) - float(first["y"][-1, 0])) / end
    y = first["y"][:, 0] + weight * second["y"][:, 0]
    dy = first["y"][:, 1] + weight * second["y"][:, 1]
    return {"x": first["t"], "y": y, "dy": dy, "slope": float(dy[0]),
            "sensitivity": end, "homogeneous_scale": scale, "relative_end": relative,
            "correction_weight": float(weight),
            "near_a_resonance": bool(relative < float(resonance_tol)),
            "steps": int(steps), "iterations": 0, "amplification": abs(end)}


def shooting(f, a: float, b: float, alpha: float, beta: float, guesses=None,
             steps: int = 200, name: str = "rk4", tol: float = 1e-10,
             max_iter: int = 50) -> dict:
    """Nonlinear shooting: the secant method applied to ``miss(s) - beta``.

    The secant method is the right root finder here because each evaluation of ``miss`` costs a
    whole IVP solve and a derivative would cost another one. Its order is 1.618 against Newton's
    2, and it needs half the work per step, so it wins on cost per digit exactly as lesson 10
    measured.
    """
    lo, hi = ((float(beta) - float(alpha)) / (float(b) - float(a)), 1.0) \
        if guesses is None else (float(guesses[0]), float(guesses[1]))
    if lo == hi:
        hi = lo + 1.0
    history = []
    s0, s1 = lo, hi
    f0 = miss(f, a, b, alpha, s0, steps, name) - float(beta)
    f1 = miss(f, a, b, alpha, s1, steps, name) - float(beta)
    history.append((s0, f0))
    history.append((s1, f1))
    converged = False
    for _ in range(int(max_iter)):
        if abs(f1) <= tol:
            converged = True
            break
        denominator = f1 - f0
        if denominator == 0.0:
            break
        s2 = s1 - f1 * (s1 - s0) / denominator
        s0, f0 = s1, f1
        s1 = s2
        f1 = miss(f, a, b, alpha, s1, steps, name) - float(beta)
        history.append((s1, f1))
    else:
        converged = abs(f1) <= tol
    out = integrate(f, float(a), [float(alpha), s1], float(b), int(steps), named_step(name))
    return {"x": out["t"], "y": out["y"][:, 0], "dy": out["y"][:, 1], "slope": float(s1),
            "residual": float(f1), "converged": bool(converged),
            "iterations": len(history) - 2,
            "solves": len(history),
            "guess_history": np.asarray([h[0] for h in history]),
            "miss_history": np.asarray([h[1] for h in history])}


def multiple_shooting(f, a: float, b: float, alpha: float, beta: float, pieces: int = 4,
                      steps: int = 200, name: str = "rk4", tol: float = 1e-10,
                      max_iter: int = 60) -> dict:
    """Shooting over several short intervals instead of one long one.

    The amplification of single shooting is exponential in the length of the interval, so
    cutting the interval into ``k`` pieces cuts the exponent by ``k``. The unknowns become the
    state at every interior break point, with continuity imposed as extra equations, and the
    resulting system is solved by Newton with a numerically built Jacobian.

    This is the standard fix, and it is where shooting and finite differences meet: with one
    piece it is shooting, and with a piece per node it is a finite difference method.
    """
    from .nlsystems import newton_system

    k = int(pieces)
    if k < 1:
        raise ValueError(f"need at least one piece, got {k}")
    breaks = np.linspace(float(a), float(b), k + 1)
    per = max(int(steps) // k, 2)
    step = named_step(name)

    # unknowns: the slope at a, then (y, y') at each interior break point
    def residual(z):
        v = np.asarray(z, dtype=float)
        state = np.asarray([float(alpha), float(v[0])])
        out = np.empty(v.size)
        for j in range(k):
            end = integrate(f, float(breaks[j]), state, float(breaks[j + 1]), per, step)
            reached = end["y"][-1]
            if j < k - 1:
                want = v[1 + 2 * j: 3 + 2 * j]
                out[2 * j] = float(reached[0] - want[0])
                out[2 * j + 1] = float(reached[1] - want[1])
                state = np.asarray(want, dtype=float)
            else:
                out[-1] = float(reached[0]) - float(beta)
        return out

    guess = np.empty(2 * k - 1)
    guess[0] = (float(beta) - float(alpha)) / (float(b) - float(a))
    for j in range(k - 1):
        t = (breaks[j + 1] - a) / (b - a)
        guess[1 + 2 * j] = float(alpha) + t * (float(beta) - float(alpha))
        guess[2 + 2 * j] = guess[0]
    result = newton_system(residual, None, guess, tol=tol, max_iter=int(max_iter))
    z = np.asarray(result.root, dtype=float)
    xs, ys, dys = [], [], []
    state = np.asarray([float(alpha), float(z[0])])
    for j in range(k):
        end = integrate(f, float(breaks[j]), state, float(breaks[j + 1]), per, step)
        take = slice(None) if j == 0 else slice(1, None)
        xs.append(end["t"][take])
        ys.append(end["y"][take, 0])
        dys.append(end["y"][take, 1])
        state = (np.asarray(z[1 + 2 * j: 3 + 2 * j], dtype=float) if j < k - 1
                 else end["y"][-1])
    return {"x": np.concatenate(xs), "y": np.concatenate(ys), "dy": np.concatenate(dys),
            "breaks": breaks, "pieces": k, "slope": float(z[0]),
            "converged": bool(result.converged), "iterations": int(result.n_iter),
            "residual": float(np.max(np.abs(residual(z))))}


# --------------------------------------------------------------------------- finite differences


def finite_difference_matrix(p, q, x, alpha: float, beta: float) -> dict:
    """Assemble the tridiagonal system for ``y'' = p y' + q y + r`` on the grid ``x``.

    Central differences at every interior node, so on a uniform grid row ``i`` reads

        ``(1 + h p_i / 2) y_(i-1) + (-2 - h^2 q_i) y_i + (1 - h p_i / 2) y_(i+1)``

    The three diagonals are returned separately, because the solve is by Thomas and building the
    dense matrix would throw away the whole saving. The grid may be unequal, in which case the
    unequal coefficients below are used.
    """
    v = np.asarray(x, dtype=float).ravel()
    if v.size < 3:
        raise ValueError(f"need at least one interior node, got a grid of {v.size} points")
    if np.any(np.diff(v) <= 0.0):
        raise ValueError("the grid must be strictly increasing")
    interior = v[1:-1]
    left = np.diff(v)[:-1]
    right = np.diff(v)[1:]
    total = left + right
    # the three point second and first difference coefficients on an unequal grid
    d2 = np.stack([2.0 / (left * total), -2.0 / (left * right), 2.0 / (right * total)])
    d1 = np.stack([-right / (left * total), (right - left) / (left * right),
                   left / (right * total)])
    pv = np.broadcast_to(np.asarray(p(interior), dtype=float), interior.shape)
    qv = np.broadcast_to(np.asarray(q(interior), dtype=float), interior.shape)
    sub = d2[0] - pv * d1[0]
    diag = d2[1] - pv * d1[1] - qv
    sup = d2[2] - pv * d1[2]
    return {"sub": sub, "diag": diag, "sup": sup, "interior": interior,
            "x": v, "alpha": float(alpha), "beta": float(beta),
            "second_difference": d2, "first_difference": d1}


def finite_difference_linear(p, q, r, a: float = None, b: float = None, alpha: float = 0.0,
                             beta: float = 0.0, n: int = 50, x=None) -> dict:
    """Solve a linear BVP by central differences and one tridiagonal solve.

    ``n`` is the number of subintervals, so the grid has ``n + 1`` points and ``n - 1`` unknowns.
    Pass ``x`` instead to use a grid of your own, equal or not.

    The whole cost is ``O(n)``, and there is no iteration and nothing to amplify. Compare that
    with `shooting`, whose cost is an IVP solve per secant step and whose accuracy is limited by
    a sensitivity that can be astronomically large.
    """
    grid = (np.linspace(float(a), float(b), int(n) + 1) if x is None
            else np.asarray(x, dtype=float).ravel())
    system = finite_difference_matrix(p, q, grid, alpha, beta)
    interior = system["interior"]
    rhs = np.broadcast_to(np.asarray(r(interior), dtype=float), interior.shape).astype(float)
    rhs = rhs.copy()
    rhs[0] -= system["sub"][0] * float(alpha)
    rhs[-1] -= system["sup"][-1] * float(beta)
    inner = thomas(system["sub"][1:], system["diag"], system["sup"][:-1], rhs)
    y = np.empty(grid.size)
    y[0], y[-1] = float(alpha), float(beta)
    y[1:-1] = inner
    return {"x": grid, "y": y, "interior": interior, "unknowns": inner.size,
            "sub": system["sub"], "diag": system["diag"], "sup": system["sup"],
            "step": (float(grid[1] - grid[0]) if x is None else float(np.max(np.diff(grid)))),
            "uniform": bool(x is None)}


def finite_difference_nonlinear(f, f_y, f_dy, a: float, b: float, alpha: float, beta: float,
                                n: int = 50, tol: float = 1e-12, max_iter: int = 50,
                                frozen: bool = False) -> dict:
    """Solve ``y'' = f(x, y, y')`` by central differences and Newton on the discrete system.

    The system is ``F_i = (y_(i-1) - 2 y_i + y_(i+1))/h^2 - f(x_i, y_i, (y_(i+1) - y_(i-1))/2h)``
    and its Jacobian is tridiagonal, so each Newton step is another ``O(n)`` Thomas solve. That
    is the reason nonlinearity costs so much less here than the ``O(n^3)`` a dense Newton would
    suggest.

    Newton is quadratic, so the residual should square each step until it hits the roundoff
    floor. `newton_squares_the_residual` measures that it does.

    The floor is not small. The discrete operator divides by ``h^2``, so the residual cannot be
    evaluated below about ``eps |y| / h^2``, which at ``n = 40`` on this problem is ``10^-12``.
    A tolerance set below that would never be met, so the iteration also stops when the residual
    **stops falling** near that floor, and reports it as ``stalled`` rather than as failure.

    ``frozen=True`` builds the Jacobian once from the initial guess and reuses it, which is the
    chord method. It reaches the same answer and converges linearly instead of quadratically,
    which is what `newton_squares_the_residual` compares against.
    """
    grid = np.linspace(float(a), float(b), int(n) + 1)
    h = float(grid[1] - grid[0])
    interior = grid[1:-1]
    # a straight line between the boundary values is the standard starting guess
    y = np.interp(interior, [float(a), float(b)], [float(alpha), float(beta)])
    residuals = []
    converged = False
    stalled = False
    held = None
    for _ in range(int(max_iter)):
        left = np.concatenate([[float(alpha)], y[:-1]])
        right = np.concatenate([y[1:], [float(beta)]])
        slope = (right - left) / (2.0 * h)
        F = (left - 2.0 * y + right) / h ** 2 - np.asarray(f(interior, y, slope), dtype=float)
        residuals.append(float(np.max(np.abs(F))))
        if residuals[-1] <= tol:
            converged = True
            break
        if (len(residuals) > 1 and residuals[-1] >= residuals[-2]
                and residuals[-1] < 1e-6 * residuals[0]):
            stalled = True
            converged = True
            break
        if held is None or not frozen:
            fy = np.broadcast_to(np.asarray(f_y(interior, y, slope), dtype=float),
                                 interior.shape)
            fdy = np.broadcast_to(np.asarray(f_dy(interior, y, slope), dtype=float),
                                  interior.shape)
            held = (1.0 / h ** 2 + fdy / (2.0 * h),
                    -2.0 / h ** 2 - fy,
                    1.0 / h ** 2 - fdy / (2.0 * h))
        sub, diag, sup = held
        y = y - thomas(sub[1:], diag, sup[:-1], F)
    full = np.empty(grid.size)
    full[0], full[-1] = float(alpha), float(beta)
    full[1:-1] = y
    return {"x": grid, "y": full, "step": h, "unknowns": y.size,
            "residuals": np.asarray(residuals), "iterations": len(residuals) - 1,
            "converged": bool(converged), "stalled": bool(stalled),
            "roundoff_floor": float(np.finfo(float).eps
                                    * max(abs(float(alpha)), abs(float(beta)), 1.0) / h ** 2),
            "final_residual": residuals[-1]}


# --------------------------------------------------------------------------- measurements


def shooting_amplifies_the_guess(rates=None, steps: int = 2000) -> dict:
    """How much a change in the initial slope moves the far boundary, against the growth rate.

    For ``y'' = lam^2 y`` on ``[0, 1]`` the homogeneous solution with slope 1 is
    ``sinh(lam x)/lam``, so the sensitivity at ``b`` is ``sinh(lam)/lam`` and it grows like
    ``e^lam / (2 lam)``. Working backwards, the best achievable accuracy in the slope is the
    target accuracy divided by that number.

    At ``lam = 40`` the sensitivity is ``3 x 10^15``, which is larger than ``1/eps``: no double
    precision slope exists that hits the target, and the failure is in the **problem's
    reformulation as an IVP**, not in the root finder or the integrator.
    """
    lams = ([1.0, 5.0, 10.0, 20.0, 30.0, 40.0] if rates is None
            else [float(v) for v in np.atleast_1d(rates)])
    rows = []
    for lam in lams:
        problem = exponential_problem(lam)
        with np.errstate(over="ignore", invalid="ignore"):
            out = linear_shooting(problem["p"], problem["q"], problem["r"], problem["a"],
                                  problem["b"], problem["alpha"], problem["beta"], steps)
            want = problem["exact"](out["x"])
            error = float(np.max(np.abs(out["y"] - want)))
        closed = math.sinh(lam) / lam
        rows.append((lam, out["sensitivity"], closed, error,
                     float(np.finfo(float).eps * abs(closed))))
    return {"rate": np.asarray([r[0] for r in rows]),
            "measured_sensitivity": np.asarray([r[1] for r in rows]),
            "closed_form_sensitivity": np.asarray([r[2] for r in rows]),
            "shooting_error": np.asarray([r[3] for r in rows]),
            "unavoidable_slope_error": np.asarray([r[4] for r in rows]),
            "grows_exponentially": bool(np.all(np.diff(np.asarray([r[1] for r in rows])) > 0.0))}


def shooting_against_finite_differences(rates=None, n: int = 400, steps: int = 400) -> dict:
    """The same problems solved both ways, so the failure is visible side by side.

    Finite differences are **not** immune to ``lam``: the solution has a layer of width
    ``1/lam``, so a uniform grid resolves it worse as ``lam`` grows and the error rises like
    ``lam^2``. What they avoid is the **exponential** growth, because they never form the
    growing solution: the matrix stays diagonally dominant for every ``lam``.

    So the comparison is polynomial against exponential, not immune against fragile. Shooting is
    the more accurate method at small ``lam`` because RK4 is fourth order and central differences
    are second, and the two cross over somewhere in between. ``crossover_rate`` reports where.
    Measured at ``n = 400`` grid points and 400 RK4 steps:

        rate               1        5       10       20       30        40
        shooting       3e-14    8e-11    1e-09    3e-08    5e-04   1.6e+01
        differences    3e-08    2e-06    1e-05    4e-05    9e-05     2e-04

    The finite difference row rises with a fitted exponent of 2.00 in the rate, which is the
    ``lam^2`` a layer of width ``1/lam`` on a uniform grid gives. The shooting row rises by
    fifteen orders of magnitude over the same sweep and ends with no correct digit at all.
    """
    lams = ([1.0, 5.0, 10.0, 20.0, 30.0, 40.0] if rates is None
            else [float(v) for v in np.atleast_1d(rates)])
    rows = []
    for lam in lams:
        problem = exponential_problem(lam)
        with np.errstate(over="ignore", invalid="ignore"):
            shot = linear_shooting(problem["p"], problem["q"], problem["r"], problem["a"],
                                   problem["b"], problem["alpha"], problem["beta"], steps)
            shot_error = float(np.max(np.abs(shot["y"] - problem["exact"](shot["x"]))))
        fd = finite_difference_linear(problem["p"], problem["q"], problem["r"], problem["a"],
                                      problem["b"], problem["alpha"], problem["beta"], n)
        fd_error = float(np.max(np.abs(fd["y"] - problem["exact"](fd["x"]))))
        rows.append((lam, shot_error, fd_error, shot["amplification"]))
    rate_values = np.asarray([r[0] for r in rows])
    shot_errors = np.asarray([r[1] for r in rows])
    fd_errors = np.asarray([r[2] for r in rows])
    worse = np.flatnonzero(shot_errors > fd_errors)
    tail = rate_values >= 10.0
    fd_exponent = (float(np.polyfit(np.log(rate_values[tail]), np.log(fd_errors[tail]), 1)[0])
                   if int(np.sum(tail)) >= 3 else float("nan"))
    return {"rate": rate_values,
            "shooting_error": shot_errors, "finite_difference_error": fd_errors,
            "amplification": np.asarray([r[3] for r in rows]),
            "finite_difference_exponent_in_the_rate": fd_exponent,
            "finite_differences_grow_polynomially": bool(fd_exponent < 3.0),
            "shooting_grows_faster_than_any_power": bool(
                shot_errors[-1] / max(shot_errors[0], 1e-300)
                > 1e6 * (rate_values[-1] / rate_values[0]) ** 4),
            "crossover_rate": (float(rate_values[worse[0]]) if worse.size else None),
            "shooting_wins_at_the_smallest_rate": bool(shot_errors[0] < fd_errors[0])}


def multiple_shooting_recovers_it(rate: float = 30.0, piece_counts=None,
                                  steps: int = 2000) -> dict:
    """Cutting the interval into pieces cuts the exponent, measured against the piece count.

    The amplification over a piece of length ``L`` is about ``e^(lam L)``, so with ``k`` equal
    pieces it falls to ``e^(lam/k)``. The improvement is therefore not gradual. At ``lam = 30``:

        pieces        1            2            4            8           16
        error    2.4e-04      1.6e-10      1.6e-10      1.6e-10      1.6e-10

    **One extra piece recovers six orders of magnitude and the rest change nothing**, because by
    then the amplification is no longer what limits the answer and RK4's own discretisation error
    is. Splitting further costs Newton iterations on a larger system and buys nothing.
    """
    ks = ([1, 2, 4, 8, 16] if piece_counts is None
          else [int(v) for v in np.atleast_1d(piece_counts)])
    problem = exponential_problem(rate)

    def f(t, state):
        u = as_state(state)
        return np.asarray([u[1], rate ** 2 * u[0]])

    rows = []
    for k in ks:
        with np.errstate(over="ignore", invalid="ignore"):
            try:
                out = multiple_shooting(f, problem["a"], problem["b"], problem["alpha"],
                                        problem["beta"], k, steps)
                error = float(np.max(np.abs(out["y"] - problem["exact"](out["x"]))))
                ok = bool(out["converged"])
            except (np.linalg.LinAlgError, ValueError, OverflowError):
                error, ok = float("inf"), False
        rows.append((k, error, ok, math.exp(rate / k)))
    errors = np.asarray([r[1] for r in rows])
    counts = np.asarray([r[0] for r in rows])
    # the smallest piece count within a factor of two of the best, because the tail is a tie at
    # the discretisation floor and picking the exact minimum out of it would be picking noise
    good = np.flatnonzero(errors <= 2.0 * float(np.min(errors)))
    return {"pieces": counts, "error": errors,
            "converged": np.asarray([r[2] for r in rows]),
            "amplification_per_piece": np.asarray([r[3] for r in rows]),
            "enough_pieces": (int(counts[good[0]]) if good.size else None),
            "gain_from_the_first_split": float(
                errors[0] / max(errors[min(1, errors.size - 1)], 1e-300)),
            "improves": bool(np.min(errors) < errors[0])}


def order_of_finite_differences(counts=None, problem=None) -> dict:
    """The central difference solution error against the grid, which must be second order.

    Both the second difference and the central first difference are second order accurate, so
    the truncation error is ``O(h^2)`` and the solution error inherits it. That inheritance is
    not automatic: it needs the discrete operator to be stable, which for this matrix means
    diagonally dominant.
    """
    use = textbook_linear_problem() if problem is None else problem
    ns = ([10, 20, 40, 80, 160, 320] if counts is None
          else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        out = finite_difference_linear(use["p"], use["q"], use["r"], use["a"], use["b"],
                                       use["alpha"], use["beta"], n)
        rows.append((n, float(out["step"]),
                     float(np.max(np.abs(out["y"] - use["exact"](out["x"]))))))
    hs = np.asarray([r[1] for r in rows])
    errors = np.asarray([r[2] for r in rows])
    keep = errors > 1e-13
    order = (float(np.polyfit(np.log(hs[keep]), np.log(errors[keep]), 1)[0])
             if int(np.sum(keep)) >= 3 else float("nan"))
    return {"n": np.asarray([r[0] for r in rows]), "h": hs, "error": errors,
            "fitted_order": order,
            "ratio_per_halving": errors[:-1] / np.maximum(errors[1:], 1e-300),
            "second_order": bool(abs(order - 2.0) < 0.15)}


def newton_squares_the_residual(n: int = 40) -> dict:
    """The residual of the nonlinear finite difference system, iteration by iteration.

    Newton is quadratic, so each residual should be about the square of the last one until the
    roundoff floor stops it. That is the sharpest available check that the Jacobian is right: a
    wrong Jacobian still converges, and looks fine unless the **rate** is measured.

    The comparison run here freezes the Jacobian at the initial guess, which is the chord
    method. It reaches the same answer, and it does so linearly, taking many times as many
    iterations. Both runs are reported.

    Only the steps above the roundoff floor are usable for measuring the rate, because a ratio
    computed across the floor says nothing. Which steps those are is reported too.
    """
    problem = textbook_nonlinear_problem()
    out = finite_difference_nonlinear(problem["f"], problem["f_y"], problem["f_dy"],
                                      problem["a"], problem["b"], problem["alpha"],
                                      problem["beta"], n)
    chord = finite_difference_nonlinear(problem["f"], problem["f_y"], problem["f_dy"],
                                        problem["a"], problem["b"], problem["alpha"],
                                        problem["beta"], n, max_iter=500, frozen=True)
    values = np.asarray(out["residuals"], dtype=float)
    floor = float(out["roundoff_floor"])
    usable, squared, linear = [], [], []
    for k in range(1, values.size):
        if values[k - 1] < 1.0 and values[k] > 10.0 * floor:
            usable.append(k)
            squared.append(float(values[k] / values[k - 1] ** 2))
            linear.append(float(values[k] / values[k - 1]))
    chord_values = np.asarray(chord["residuals"], dtype=float)
    chord_ratios = [float(chord_values[k] / chord_values[k - 1])
                    for k in range(1, chord_values.size)
                    if chord_values[k] > 10.0 * floor and chord_values[k - 1] < 1.0]
    return {"residuals": values, "roundoff_floor": floor,
            "usable_steps": np.asarray(usable, dtype=int),
            "squared_ratio": np.asarray(squared), "linear_ratio": np.asarray(linear),
            "iterations": int(out["iterations"]), "converged": bool(out["converged"]),
            "error": float(np.max(np.abs(out["y"] - problem["exact"](out["x"])))),
            "chord_residuals": chord_values, "chord_iterations": int(chord["iterations"]),
            "chord_ratio": (float(np.median(chord_ratios)) if chord_ratios else float("nan")),
            "chord_error": float(np.max(np.abs(chord["y"] - problem["exact"](chord["x"])))),
            "quadratic": bool(len(squared) > 0 and max(squared) < 100.0
                              and min(linear) < 1e-2),
            "chord_is_linear": bool(len(chord_ratios) >= 3
                                    and 0.01 < float(np.median(chord_ratios)) < 0.99),
            "both_reach_the_same_answer": bool(
                abs(float(np.max(np.abs(out["y"] - chord["y"])))) < 1e-8)}


def existence_can_fail(coefficients=None, n: int = 200) -> dict:
    """A boundary value problem with no solution, or with infinitely many.

    ``y'' + k y = 0`` with ``y(0) = y(1) = 0`` has only the zero solution unless ``k`` is
    ``(m pi)^2``, at which point it has a whole family. There is no analogue of the
    Picard-Lindelof theorem here: a perfectly smooth, perfectly Lipschitz right hand side is not
    enough, and **the two point problem is a genuinely different kind of question**.

    Numerically the discrete matrix becomes singular, and its condition number blows up like
    ``1 / (k - resonance)``, which is the honest warning the method gives.

    **The discrete problem resonates in a slightly different place.** The matrix is
    ``D + k I`` with ``D`` the second difference, whose eigenvalues are ``-(4/h^2) sin^2(j pi h/2)``
    rather than ``-(j pi)^2``. So it is singular at

        ``k = (4/h^2) sin^2(pi h / 2) = pi^2 (1 - (pi h)^2 / 12 + ...)``

    which sits below ``pi^2`` by ``O(h^2)``. At ``n = 200`` the shift is ``2.0293 x 10^-4``
    against a predicted ``2.0294 x 10^-4``, which is four matching digits.

    Measuring the distance from ``pi^2`` instead makes the ``1/gap`` law look broken. Over the
    four rows nearest the resonance the product ``cond x gap`` spreads by a factor of 3.03 using
    ``pi^2`` and by **1.000** using the discrete value: the law is exact, and the only thing
    wrong with the first version is which resonance it measured from. Both are reported.
    """
    from .banded import tridiagonal_matrix

    h = 1.0 / float(n)
    discrete = 4.0 / h ** 2 * math.sin(math.pi * h / 2.0) ** 2
    ks = ([5.0, 9.0, 9.8, discrete - 1e-2, discrete - 1e-4, 12.0] if coefficients is None
          else [float(v) for v in np.atleast_1d(coefficients)])
    rows = []
    for k in ks:
        out = finite_difference_linear(lambda x: 0.0 * x, lambda x: -k + 0.0 * x,
                                       lambda x: np.ones_like(np.asarray(x, dtype=float)),
                                       0.0, 1.0, 0.0, 0.0, n)
        A = tridiagonal_matrix(out["sub"][1:], out["diag"], out["sup"][:-1])
        rows.append((k, float(np.linalg.cond(A)), float(np.max(np.abs(out["y"]))),
                     abs(k - discrete), abs(k - math.pi ** 2)))
    conds = np.asarray([r[1] for r in rows])
    discrete_gaps = np.asarray([r[3] for r in rows])
    continuous_gaps = np.asarray([r[4] for r in rows])
    # away from the resonance the condition number is set by the h^-2 floor of the second
    # difference itself, so the 1/gap law is only measurable where the resonance dominates
    near = discrete_gaps < 1.0

    def spread(values):
        return (float(np.max(values)) / max(float(np.min(values)), 1e-300)
                if values.size else float("nan"))

    return {"coefficient": np.asarray([r[0] for r in rows]),
            "condition_number": conds,
            "largest_value": np.asarray([r[2] for r in rows]),
            "distance_from_the_discrete_resonance": discrete_gaps,
            "distance_from_pi_squared": continuous_gaps,
            "resonance": float(math.pi ** 2), "discrete_resonance": float(discrete),
            "shift": float(math.pi ** 2 - discrete),
            "predicted_shift": float(math.pi ** 2 * (math.pi * h) ** 2 / 12.0),
            "worst_is_nearest_the_resonance": bool(int(np.argmax(conds))
                                                   == int(np.argmin(discrete_gaps))),
            "spread_using_the_discrete_resonance": spread(conds[near] * discrete_gaps[near]),
            "spread_using_pi_squared": spread(conds[near] * continuous_gaps[near]),
            "condition_grows_like_one_over_the_gap": bool(
                spread(conds[near] * discrete_gaps[near]) < 2.0)}


def _violates_the_maximum_principle(y, alpha: float, beta: float) -> float:
    """How far outside the boundary values the discrete solution strays.

    For ``eps y'' + y' = 0`` with no source term the exact solution lies between its two boundary
    values everywhere, and so does the discrete solution when the matrix is an M-matrix. Any
    departure is spurious, and this measures it in the units of the solution itself.
    """
    v = np.asarray(y, dtype=float)
    low, high = min(float(alpha), float(beta)), max(float(alpha), float(beta))
    return float(max(np.max(v) - high, low - np.min(v), 0.0))


def _wiggle_count(y, tol: float = 1e-9) -> int:
    """Sign changes in the successive differences, ignoring changes at the roundoff level.

    Counting sign changes without a threshold reports 20 wiggles in a solution that is monotone
    to 15 digits, because the flat part of the solution has differences of size ``10^-16`` whose
    signs are arbitrary. The threshold is relative to the total variation.
    """
    v = np.asarray(y, dtype=float)
    d = np.diff(v)
    scale = float(np.max(v) - np.min(v)) if v.size else 0.0
    cut = float(tol) * max(scale, 1e-300)
    signs = np.sign(d[np.abs(d) > cut])
    return int(np.sum(np.diff(signs) != 0)) if signs.size > 1 else 0


def at_a_resonance_refining_makes_it_worse(coefficient=None, step_counts=None) -> dict:
    """Shooting at a resonance, where more work buys a worse answer.

    ``y'' + pi^2 y = 1`` with ``y(0) = y(1) = 0`` has no solution: the homogeneous solution
    ``sin(pi x)/pi`` vanishes at both ends, so the correction ``(beta - u(b)) / v(b)`` divides by
    zero in exact arithmetic. In floating point ``v(b)`` comes out as the integrator's own
    truncation error, which is ``O(h^4)`` for RK4, so the correction weight grows like ``h^-4``
    and the returned solution grows with it.

    **Refining the step makes the answer bigger, not better**, and nothing in the run reports a
    problem. Compare `existence_can_fail`, where the finite difference method meets the same
    situation and announces it as a condition number.
    """
    k = float(math.pi ** 2 if coefficient is None else coefficient)
    ns = ([100, 200, 400, 800, 1600] if step_counts is None
          else [int(v) for v in np.atleast_1d(step_counts)])
    rows = []
    for n in ns:
        out = linear_shooting(lambda x: 0.0 * np.asarray(x, dtype=float),
                              lambda x: -k + 0.0 * np.asarray(x, dtype=float),
                              lambda x: np.ones_like(np.asarray(x, dtype=float)),
                              0.0, 1.0, 0.0, 0.0, n)
        rows.append((n, out["relative_end"], abs(out["correction_weight"]),
                     float(np.max(np.abs(out["y"]))), bool(out["near_a_resonance"])))
    ns_array = np.asarray([r[0] for r in rows], dtype=float)
    ends = np.asarray([r[1] for r in rows])
    sizes = np.asarray([r[3] for r in rows])
    return {"steps": np.asarray([r[0] for r in rows]),
            "relative_end_value": ends,
            "correction_weight": np.asarray([r[2] for r in rows]),
            "largest_value": sizes,
            "flagged": np.asarray([r[4] for r in rows]),
            "end_value_order": float(-np.polyfit(np.log(ns_array), np.log(ends), 1)[0]),
            "answer_grows_with_refinement": bool(np.all(np.diff(sizes) > 0.0)),
            "every_run_is_flagged": bool(np.all([r[4] for r in rows]))}


def oscillation_when_the_cell_number_is_too_big(viscosity: float = 0.01, counts=None) -> dict:
    """The central difference solution of a convection dominated problem, wiggle and all.

    Row ``i`` has off-diagonal entries ``1/h^2 -+ p_i/(2h)``, and when ``h |p| > 2`` one of them
    changes sign. The matrix stops being an M-matrix, the discrete maximum principle goes with
    it, and the solution leaves the range of its own boundary values even though the exact one
    never does. At ``eps = 0.01``:

        n              10       20       40       80      160      320
        cell number  5.00     2.50     1.25     0.62     0.31     0.16
        overshoot    0.70     0.43     0.11     0.00     0.00     0.00

    At ``n = 10`` the computed values run ``0, 1.70, 0.57, 1.32, 0.82, 1.15, ...``: a clean
    alternation about the true answer, decaying inwards. **The overshoot vanishes exactly when
    the cell number drops below 1**, which is what the M-matrix condition predicts.

    The cure is a finer grid, a graded grid, or upwinding. `upwinding_removes_the_wiggle` and
    `graded_grid_for_a_layer` measure the last two.
    """
    problem = convection_diffusion_problem(viscosity)
    ns = ([10, 20, 40, 80, 160, 320] if counts is None
          else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        out = finite_difference_linear(problem["p"], problem["q"], problem["r"], problem["a"],
                                       problem["b"], problem["alpha"], problem["beta"], n)
        h = float(out["step"])
        rows.append((n, h, h / (2.0 * float(viscosity)),
                     _violates_the_maximum_principle(out["y"], problem["alpha"],
                                                     problem["beta"]),
                     _wiggle_count(out["y"]),
                     float(np.max(np.abs(out["y"] - problem["exact"](out["x"])))),
                     bool(np.all(out["sub"] > 0.0) and np.all(out["sup"] > 0.0))))
    cells = np.asarray([r[2] for r in rows])
    overshoot = np.asarray([r[3] for r in rows])
    m_matrix = np.asarray([r[6] for r in rows])
    return {"n": np.asarray([r[0] for r in rows]), "h": np.asarray([r[1] for r in rows]),
            "cell_number": cells, "overshoot": overshoot,
            "wiggles": np.asarray([r[4] for r in rows]),
            "error": np.asarray([r[5] for r in rows]),
            "is_an_m_matrix": m_matrix, "threshold": 1.0,
            "overshoots_exactly_when_the_cell_number_exceeds_one":
                bool(np.all((overshoot > 1e-12) == (cells > 1.0))),
            "the_m_matrix_condition_is_the_cell_number":
                bool(np.all(m_matrix == (cells <= 1.0)))}


def upwinding_removes_the_wiggle(viscosity: float = 0.01, counts=None,
                                 resolved_below: float = 0.1) -> dict:
    """A one sided first difference instead of a central one, and what it costs.

    Upwinding biases the first difference into the direction the flow comes from, which restores
    the M-matrix property at every cell number and so removes the oscillation. It is **first
    order** rather than second, so the wiggle is paid for with an order of accuracy.

    Measured at ``eps = 0.01``, with the error ratio from one row to the next:

        n              10      20      40      80     160     320     640    1280    2560
        cell         5.00    2.50    1.25    0.62    0.31    0.16   0.078   0.039   0.020
        central     0.696   0.435   0.193  0.0557  0.0121 3.0e-03 7.5e-04 1.9e-04 4.7e-05
          ratio         .    1.60    2.25    3.47    4.60    4.02    4.03    4.00    4.00
        upwind     0.0909   0.160   0.204   0.158  0.0922  0.0507  0.0270  0.0139 7.1e-03
          ratio         .    0.57    0.78    1.29    1.71    1.82    1.88    1.94    1.97

    Two things are worth reading off that table.

    **The orders only appear once the layer is resolved.** Fitting the whole sweep gives 1.85 and
    0.62, which is neither method's order. Fitting the rows with a cell number below 0.1 gives
    2.000 and 0.974. The head of the sweep is not a slow start, it is a different regime: the
    grid is not resolving the layer at all there.

    **Upwinding wins on the two coarsest grids and loses on every finer one**, ending 300 times
    worse. It also gets *worse* from ``n = 10`` to ``n = 40`` before it starts improving, because
    its artificial diffusion ``h/2`` is what it is really solving with, and at ``n = 10`` that
    happens to smear the layer to a shape whose maximum error is small.

    **A monotone wrong answer is not automatically better than an oscillating one.** Which to
    prefer depends on whether the wiggle would be fed into something that cannot take it, such as
    a logarithm or a reaction rate.
    """
    problem = convection_diffusion_problem(viscosity)
    ns = ([10, 20, 40, 80, 160, 320, 640, 1280, 2560] if counts is None
          else [int(v) for v in np.atleast_1d(counts)])
    rows = []
    for n in ns:
        grid = np.linspace(problem["a"], problem["b"], int(n) + 1)
        h = float(grid[1] - grid[0])
        interior = grid[1:-1]
        pv = np.broadcast_to(np.asarray(problem["p"](interior), dtype=float), interior.shape)
        central = finite_difference_linear(problem["p"], problem["q"], problem["r"],
                                           problem["a"], problem["b"], problem["alpha"],
                                           problem["beta"], n)
        # backward difference where p > 0 and forward where p < 0, which is the direction that
        # keeps both off-diagonals positive
        back = np.where(pv > 0.0, 1.0, 0.0)
        sub = 1.0 / h ** 2 - pv * (-back / h)
        diag = -2.0 / h ** 2 - pv * ((2.0 * back - 1.0) / h)
        sup = 1.0 / h ** 2 - pv * ((1.0 - back) / h)
        rhs = np.zeros(interior.size)
        rhs[0] -= sub[0] * problem["alpha"]
        rhs[-1] -= sup[-1] * problem["beta"]
        upwind = np.empty(grid.size)
        upwind[0], upwind[-1] = problem["alpha"], problem["beta"]
        upwind[1:-1] = thomas(sub[1:], diag, sup[:-1], rhs)
        want = problem["exact"](grid)
        rows.append((n, h, h / (2.0 * float(viscosity)),
                     float(np.max(np.abs(central["y"] - want))),
                     float(np.max(np.abs(upwind - want))),
                     _violates_the_maximum_principle(central["y"], problem["alpha"],
                                                     problem["beta"]),
                     _violates_the_maximum_principle(upwind, problem["alpha"],
                                                     problem["beta"]),
                     grid, central["y"], upwind, want))
    hs = np.asarray([r[1] for r in rows])
    cells = np.asarray([r[2] for r in rows])
    central_errors = np.asarray([r[3] for r in rows])
    upwind_errors = np.asarray([r[4] for r in rows])
    # the order is only measurable where the grid resolves the layer; below that the two schemes
    # are solving visibly different problems and the slope means nothing
    resolved = (cells < float(resolved_below)) & (central_errors > 1e-13)

    def fit(values, mask):
        return (float(np.polyfit(np.log(hs[mask]), np.log(values[mask]), 1)[0])
                if int(np.sum(mask)) >= 3 else float("nan"))

    everything = central_errors > 1e-13
    upwind_over = np.asarray([r[6] for r in rows])
    wins = np.flatnonzero(upwind_errors < central_errors)
    central_order = fit(central_errors, resolved)
    upwind_order = fit(upwind_errors, resolved)
    return {"n": np.asarray([r[0] for r in rows]), "h": hs, "cell_number": cells,
            "central_error": central_errors, "upwind_error": upwind_errors,
            "central_overshoot": np.asarray([r[5] for r in rows]),
            "upwind_overshoot": upwind_over,
            "resolved": resolved,
            "central_order": central_order, "upwind_order": upwind_order,
            "central_order_over_the_whole_sweep": fit(central_errors, everything),
            "upwind_order_over_the_whole_sweep": fit(upwind_errors, everything),
            "upwind_never_overshoots": bool(np.all(upwind_over <= 1e-12)),
            "central_is_second_order": bool(abs(central_order - 2.0) < 0.15),
            "upwind_is_first_order": bool(abs(upwind_order - 1.0) < 0.15),
            "upwind_costs_an_order": bool(central_order - upwind_order > 0.7),
            "fitting_the_whole_sweep_would_be_wrong": bool(
                abs(fit(upwind_errors, everything) - 1.0) > 0.15),
            "upwind_only_wins_on_coarse_grids": bool(
                wins.size > 0 and int(np.max(wins)) < len(rows) - 2),
            "x": rows[-1][7], "central": rows[-1][8], "upwind": rows[-1][9],
            "exact": rows[-1][10]}


def graded_grid_for_a_layer(viscosity: float = 0.01, n: int = 40, power: float = 3.0,
                            towards: str = "a") -> dict:
    """A grid clustered into the boundary layer against a uniform grid with the same node count.

    The layer occupies a fraction ``eps`` of the interval, so a uniform grid spends almost all
    its nodes where nothing happens. Mapping a uniform parameter through ``s -> s^power`` puts
    them where the solution actually varies.

    **Clustering at the wrong end is worse than not clustering at all.** For
    ``eps y'' + y' = 0`` the layer is at the inflow boundary ``a``, and grading towards ``b``
    makes the error larger rather than smaller. The ``towards`` argument is explicit for that
    reason, and the wrong choice is measurable rather than silent.

    Same number of unknowns, same solver, same tridiagonal cost. The only thing that changed is
    where the nodes are, and that is the whole content of mesh design.
    """
    problem = convection_diffusion_problem(viscosity)
    a, b = problem["a"], problem["b"]
    side = str(towards).lower()
    if side not in ("a", "b"):
        raise ValueError(f"towards must be 'a' or 'b', got {towards!r}")
    uniform = finite_difference_linear(problem["p"], problem["q"], problem["r"], a, b,
                                       problem["alpha"], problem["beta"], n)
    s = np.linspace(0.0, 1.0, int(n) + 1)
    graded = (a + (b - a) * s ** float(power) if side == "a"
              else a + (b - a) * (1.0 - (1.0 - s) ** float(power)))
    packed = finite_difference_linear(problem["p"], problem["q"], problem["r"],
                                      alpha=problem["alpha"], beta=problem["beta"], x=graded)
    uniform_error = float(np.max(np.abs(uniform["y"] - problem["exact"](uniform["x"]))))
    graded_error = float(np.max(np.abs(packed["y"] - problem["exact"](packed["x"]))))
    steps = np.diff(graded)
    return {"uniform_x": uniform["x"], "uniform_y": uniform["y"],
            "graded_x": packed["x"], "graded_y": packed["y"],
            "unknowns": uniform["unknowns"], "towards": side,
            "uniform_error": uniform_error, "graded_error": graded_error,
            "improvement": uniform_error / max(graded_error, 1e-300),
            "smallest_graded_step": float(np.min(steps)),
            "largest_graded_step": float(np.max(steps)),
            "uniform_step": float(uniform["step"]),
            "smallest_cell_number": float(np.min(steps)) / (2.0 * float(viscosity)),
            "uniform_cell_number": float(uniform["step"]) / (2.0 * float(viscosity)),
            "uniform_overshoot": _violates_the_maximum_principle(
                uniform["y"], problem["alpha"], problem["beta"]),
            "graded_overshoot": _violates_the_maximum_principle(
                packed["y"], problem["alpha"], problem["beta"]),
            "graded_wins": bool(graded_error < uniform_error)}


def unequal_grid(a: float, b: float, n: int, pattern: str = "alternating",
                 strength: float = 0.6, seed: int = 42) -> np.ndarray:
    """Build a deliberately unequal grid on ``[a, b]`` with ``n`` subintervals.

    ``"alternating"`` cycles the spacing between ``1 - strength`` and ``1 + strength`` times the
    average, so ``h_right - h_left`` is a fixed multiple of ``h`` at every node and the truncation
    error is cleanly first order. ``"random"`` jitters each node instead, which is the harder and
    more realistic case. ``"uniform"`` is the control.
    """
    kind = str(pattern).lower()
    count = int(n)
    if count < 2:
        raise ValueError(f"need at least two subintervals, got {count}")
    if kind == "uniform":
        return np.linspace(float(a), float(b), count + 1)
    if kind == "alternating":
        widths = np.where(np.arange(count) % 2 == 0, 1.0 - float(strength),
                          1.0 + float(strength))
        edges = np.concatenate([[0.0], np.cumsum(widths) / float(np.sum(widths))])
        return float(a) + (float(b) - float(a)) * edges
    if kind == "random":
        rng = np.random.default_rng(int(seed))
        s = np.linspace(0.0, 1.0, count + 1)
        jitter = np.zeros(s.size)
        jitter[1:-1] = rng.uniform(-0.5 * float(strength), 0.5 * float(strength),
                                   s.size - 2) / count
        return float(a) + (float(b) - float(a)) * np.sort(np.clip(s + jitter, 0.0, 1.0))
    raise ValueError(f"unknown pattern {pattern!r}, expected alternating, random or uniform")


def supraconvergence_on_an_unequal_grid(counts=None, pattern: str = "alternating",
                                        strength: float = 0.6, seed: int = 42) -> dict:
    """Truncation error and solution error on an irregular grid, measured separately.

    The three point second difference on an unequal grid has truncation error
    ``(h_right - h_left) y'''(x) / 3 + O(h^2)``, which is **first order** when the spacings
    differ by ``O(h)``. So the consistency of the method is first order.

    The solution error is second order anyway. The first order part of the truncation error is a
    difference of a smooth quantity, so it telescopes when the inverse of the operator is applied
    and only its variation survives. That is supraconvergence, and it is why irregular meshes are
    usable at all.

    **Which norm the truncation error is measured in changes the answer.** Measured on the
    textbook problem with spacings alternating between 0.4h and 1.6h:

        n                   20       40       80      160      320      640
        truncation max   4.3e-3   2.7e-3   1.5e-3   8.1e-4   4.1e-4   2.1e-4
        truncation rms   2.4e-3   1.2e-3   6.1e-4   3.1e-4   1.5e-4   7.7e-5
        solution         2.4e-5   5.9e-6   1.5e-6   3.7e-7   9.3e-8   2.3e-8

    The fitted orders are 0.88 in the max norm, **0.99 in the rms norm**, and 2.00 for the
    solution. The max norm is contaminated: it picks whichever node has the largest third
    derivative and mixes the second order part of the error in with the first order part. On a
    randomly jittered grid the max norm reads 0.62 and the rms still reads 0.99, so the rms is
    what to trust and the max is what makes the effect look like something else.
    """
    ns = ([20, 40, 80, 160, 320, 640] if counts is None
          else [int(v) for v in np.atleast_1d(counts)])
    problem = textbook_linear_problem()
    rows = []
    for n in ns:
        grid = unequal_grid(problem["a"], problem["b"], n, pattern, strength, seed)
        out = finite_difference_linear(problem["p"], problem["q"], problem["r"],
                                       alpha=problem["alpha"], beta=problem["beta"], x=grid)
        # the truncation error: apply the discrete operator to the exact solution
        system = finite_difference_matrix(problem["p"], problem["q"], grid,
                                          problem["alpha"], problem["beta"])
        want = problem["exact"](grid)
        applied = (system["sub"] * want[:-2] + system["diag"] * want[1:-1]
                   + system["sup"] * want[2:])
        rhs = np.broadcast_to(np.asarray(problem["r"](system["interior"]), dtype=float),
                              system["interior"].shape)
        local = np.abs(applied - rhs)
        spacings = np.diff(grid)
        rows.append((n, float(np.max(spacings)), float(np.max(local)),
                     float(np.sqrt(np.mean(local ** 2))),
                     float(np.max(np.abs(out["y"] - want))),
                     float(np.max(spacings) / np.min(spacings))))
    hs = np.asarray([r[1] for r in rows])
    max_trunc = np.asarray([r[2] for r in rows])
    rms_trunc = np.asarray([r[3] for r in rows])
    solution = np.asarray([r[4] for r in rows])

    def fit(values):
        keep = values > 1e-13
        return (float(np.polyfit(np.log(hs[keep]), np.log(values[keep]), 1)[0])
                if int(np.sum(keep)) >= 3 else float("nan"))

    rms_order = fit(rms_trunc)
    solution_order = fit(solution)
    return {"n": np.asarray([r[0] for r in rows]), "h": hs, "pattern": str(pattern).lower(),
            "spacing_ratio": np.asarray([r[5] for r in rows]),
            "truncation_max": max_trunc, "truncation_rms": rms_trunc,
            "solution_error": solution,
            "truncation_order_in_the_max_norm": fit(max_trunc),
            "truncation_order": rms_order, "solution_order": solution_order,
            "truncation_is_first_order": bool(abs(rms_order - 1.0) < 0.15),
            "solution_is_second_order": bool(abs(solution_order - 2.0) < 0.15),
            "solution_beats_consistency": bool(solution_order > rms_order + 0.7),
            "the_max_norm_hides_it": bool(abs(fit(max_trunc) - 1.0) > 0.1)}


def cost_of_the_two_methods(counts=None, rate: float = 5.0) -> dict:
    """Right hand side evaluations for each method at matched accuracy, on the same problem.

    Shooting for a linear problem needs exactly two IVP solves whatever the accuracy wanted, and
    each solve costs four evaluations per step for RK4. Finite differences need one tridiagonal
    solve, whose cost is about ``8n`` flops and ``2(n-1)`` coefficient evaluations.

    The numbers are close on a mild problem, which is the honest answer: **the reason to prefer
    finite differences is the conditioning, not the cost.**
    """
    ns = ([20, 40, 80, 160, 320] if counts is None
          else [int(v) for v in np.atleast_1d(counts)])
    problem = exponential_problem(rate)
    rows = []
    for n in ns:
        fd = finite_difference_linear(problem["p"], problem["q"], problem["r"], problem["a"],
                                      problem["b"], problem["alpha"], problem["beta"], n)
        fd_error = float(np.max(np.abs(fd["y"] - problem["exact"](fd["x"]))))
        shot = linear_shooting(problem["p"], problem["q"], problem["r"], problem["a"],
                               problem["b"], problem["alpha"], problem["beta"], n)
        shot_error = float(np.max(np.abs(shot["y"] - problem["exact"](shot["x"]))))
        rows.append((n, 2 * (n - 1), fd_error, 8 * n, shot_error))
    return {"n": np.asarray([r[0] for r in rows]),
            "finite_difference_evaluations": np.asarray([r[1] for r in rows]),
            "finite_difference_error": np.asarray([r[2] for r in rows]),
            "shooting_evaluations": np.asarray([r[3] for r in rows]),
            "shooting_error": np.asarray([r[4] for r in rows]),
            "shooting_is_more_accurate_here": bool(
                np.asarray([r[4] for r in rows])[-1]
                < np.asarray([r[2] for r in rows])[-1]),
            "note": ("shooting wins on a mild problem because RK4 is fourth order and central "
                     "differences are second; it loses on a stiff one for reasons of "
                     "conditioning, which no amount of order fixes")}
