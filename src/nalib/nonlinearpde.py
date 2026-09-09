"""Nonlinear PDEs, and the method of lines that connects this part back to Part 10.

Two ways to be nonlinear
------------------------
A **nonlinear elliptic** problem discretizes to a nonlinear system, which is Part 2's Newton method
applied to a residual with thousands of components and a very sparse Jacobian. Nothing about
Newton changes; what changes is that forming and factoring the Jacobian is now the whole cost, and
its five nonzeros a row are the reason the method is affordable at all.

A **nonlinear time dependent** problem offers a choice. Discretize both variables at once and get
a nonlinear system per time step, or discretize **space only** and get

    u'(t) = F(u(t)) ,

a large system of ordinary differential equations. That is the **method of lines**, and it means
every method of Part 10 applies unchanged: the step control of lesson 70, the stiffness analysis
of lesson 72, the implicit methods that analysis argues for.

What the measurements here show
-------------------------------
* Newton converges quadratically **only in its second phase**. From a zero start on a strongly
  nonlinear problem the first step overshoots and the residual **rises** from 40 to 1144 before it
  falls; fitting the whole history gives 1.75 and fitting the last three steps gives 1.95. The
  Jacobian is 92 per cent zeros, which is what makes each step cheap.
* The Bratu problem ``-u'' = lam exp(u)`` has **two** solutions below a critical ``lam`` and
  **none** above it. The measured turning point is ``3.5135`` at 81 points against the closed form
  ``3.5138307``, and the gap is the discretization error: it falls at order 2 as the grid is
  refined. Newton failing above the fold is not a defect, it is the absence of anything to
  converge to.
* The semi-discrete heat operator's eigenvalues are exactly lesson 77's, so lesson 72's stability
  region gives the explicit step limit, and it comes out as ``r <= 1/2``: **lesson 78's condition
  derived entirely from the ODE side**, with the two agreeing to seven digits.
* An adaptive Runge-Kutta solver from lesson 70, given the semi-discrete system and no information
  about stability at all, settles on a step within **0.6 per cent** of that method's own stability
  limit, and six orders of magnitude of tolerance move it by 2 per cent. It is not being accurate,
  it is being held.
* Space discretization plus Euler **is** lesson 78's explicit scheme, bit for bit. Replacing Euler
  by RK4 removes the time error entirely, from ``4e-5`` to ``6e-16``, and makes the **total** error
  twice as large, because Euler's time error had the opposite sign to the space error and was
  partly cancelling it.
* Fisher's equation travels at ``2 sqrt(D r)`` only for a steep enough initial front, and reaches
  it only algebraically slowly. A shallow front travels at ``D a + r / a``, measured to five
  digits; a steep one creeps up through 0.149, 0.177, 0.190, 0.195 towards 0.2, matching Bramson's
  ``c* - 3/(2 a* t)`` to 0.2 per cent in the last window.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from . import adaptivestep, ivp
from .nlsystems import newton_system
from .stability import growth_function


# --------------------------------------------------------------------------- problems


def cubic_problem(a: float = 0.0, b: float = 1.0, weight: float = 1.0):
    """``-u'' + c u**3 = f`` with ``u = sin(pi x)``, so ``f = pi**2 sin + c sin**3``.

    The simplest nonlinear elliptic problem with an exact solution: the nonlinearity is a cubic,
    which is monotone increasing, so the discrete system has exactly one solution and Newton
    converges from any sensible start. That makes it the right place to measure the convergence
    rate before meeting a problem where the solution set itself is the subject.
    """
    a, b = float(a), float(b)
    if not b > a:
        raise ValueError(f"need a < b, got a={a}, b={b}")
    c = float(weight)
    wave = math.pi / (b - a)

    def exact(x):
        return np.sin(wave * (np.asarray(x, dtype=float) - a))

    def source(x):
        return wave ** 2 * exact(x) + c * exact(x) ** 3

    return {"a": a, "b": b, "exact": exact, "source": source, "weight": c,
            "reaction": lambda u: c * np.asarray(u, dtype=float) ** 3,
            "reaction_derivative": lambda u: 3.0 * c * np.asarray(u, dtype=float) ** 2,
            "alpha": 0.0, "beta": 0.0, "name": "cubic"}


def bratu_problem(lam: float, a: float = 0.0, b: float = 1.0):
    """``-u'' = lam exp(u)``, ``u(a) = u(b) = 0``: the standard bifurcation example.

    This one has no solution at all above a critical ``lam``, two below it, and exactly one at the
    turning point. In one dimension the critical value satisfies ``lam* = 2 c**2 / cosh(c/4)**2``
    maximised over ``c``, and works out to ``3.513830719...``.

    There is no manufactured solution here, and that is the point: the question is not how
    accurate the answer is but **how many answers there are**.
    """
    a, b = float(a), float(b)
    lam = float(lam)

    return {"a": a, "b": b, "lam": lam, "exact": None,
            "source": lambda x: np.zeros(np.shape(np.asarray(x, dtype=float))),
            "reaction": lambda u: -lam * np.exp(np.asarray(u, dtype=float)),
            "reaction_derivative": lambda u: -lam * np.exp(np.asarray(u, dtype=float)),
            "alpha": 0.0, "beta": 0.0, "name": f"bratu lam={lam:g}"}


def bratu_critical_value() -> float:
    """The exact turning point of the one dimensional Bratu problem, by its own equation.

    The closed form solution is ``u(x) = -2 log( cosh((x - 1/2) c / 2) / cosh(c / 4) )`` with ``c``
    solving ``c = sqrt(2 lam) cosh(c/4)``. Eliminating ``lam`` gives

        lam = c**2 / (2 cosh(c/4)**2) ,

    and the turning point is where that is largest, which is where ``c tanh(c/4) = 4``. Found here
    by bisecting on that condition rather than by quoting a number, so the value can be checked
    against the measurement instead of the other way round. It comes out at ``3.5138307``.
    """
    def value(c):
        return c ** 2 / (2.0 * math.cosh(c / 4.0) ** 2)

    def derivative(c):
        # d/dc of value has the sign of 4 cosh(c/4) - c sinh(c/4)
        return 4.0 * math.cosh(c / 4.0) - c * math.sinh(c / 4.0)

    lo, hi = 1e-6, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if derivative(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return value(0.5 * (lo + hi))


# --------------------------------------------------------------------------- Newton


def residual_and_jacobian(problem, points: int):
    """The discrete residual ``F(u)`` and its Jacobian, for a one dimensional nonlinear problem.

    The operator is ``-u'' + g(u) - f``, differenced with lesson 77's three point stencil. The
    Jacobian is the same tridiagonal matrix plus a diagonal ``g'(u)``, which is the whole reason
    Newton is affordable here: a dense Jacobian would cost ``m**2`` entries and ``m**3`` to
    factor, and this one costs ``3m`` and ``m``.
    """
    n = int(points)
    if n < 3:
        raise ValueError(f"need at least 3 points, got {n}")
    x = np.linspace(problem["a"], problem["b"], n)
    h = float(x[1] - x[0])
    inner = x[1:-1]
    m = inner.size
    source = np.asarray(problem["source"](inner), dtype=float)
    ends = np.zeros(m)
    ends[0] += problem["alpha"] / h ** 2
    ends[-1] += problem["beta"] / h ** 2

    def residual(u):
        v = np.asarray(u, dtype=float)
        padded = np.concatenate([[problem["alpha"]], v, [problem["beta"]]])
        second = (padded[2:] - 2.0 * padded[1:-1] + padded[:-2]) / h ** 2
        return -second + np.asarray(problem["reaction"](v), dtype=float) - source

    def jacobian(u):
        v = np.asarray(u, dtype=float)
        out = (np.diag(np.full(m, 2.0 / h ** 2))
               + np.diag(np.full(m - 1, -1.0 / h ** 2), 1)
               + np.diag(np.full(m - 1, -1.0 / h ** 2), -1))
        out += np.diag(np.asarray(problem["reaction_derivative"](v), dtype=float))
        return out

    return {"residual": residual, "jacobian": jacobian, "x": x, "interior": inner,
            "h": h, "unknowns": m, "ends": ends}


def solve_nonlinear(problem, points: int, guess=None, tol: float = 1e-12,
                    max_iter: int = 100):
    """Newton's method on the discrete nonlinear system, reporting the residual history."""
    pieces = residual_and_jacobian(problem, points)
    m = pieces["unknowns"]
    start = np.zeros(m) if guess is None else np.asarray(guess, dtype=float)
    if start.size != m:
        raise ValueError(f"the guess has {start.size} entries but the grid has {m}")
    history = []

    def residual(u):
        value = pieces["residual"](u)
        history.append(float(np.max(np.abs(value))))
        return value

    out = newton_system(residual, pieces["jacobian"], start, tol=tol, max_iter=max_iter)
    u = np.asarray(out.root, dtype=float)
    answer = {"u": u, "x": pieces["x"], "interior": pieces["interior"], "h": pieces["h"],
              "iterations": int(out.n_iter), "converged": bool(out.converged),
              "residual_history": np.asarray(history),
              "final_residual": float(np.max(np.abs(pieces["residual"](u))))}
    if problem["exact"] is not None:
        want = np.asarray(problem["exact"](pieces["interior"]), dtype=float)
        answer["exact"] = want
        answer["error"] = float(np.max(np.abs(u - want)))
    return answer


def newton_is_quadratic(points: int = 41, weight: float = 30.0):
    """Measure the convergence order of Newton on the discrete nonlinear system.

    Quadratic convergence means ``log(e_{k+1}) = 2 log(e_k) + const``, so fitting the successive
    residuals against each other on a log scale gives the exponent directly.

    Two things have to be excluded from that fit or it measures something else. The residuals at
    the rounding floor, near ``1e-16``, would drag the exponent towards 1; and Newton's **global**
    phase, before it gets close, is not quadratic at all. From a zero start on this problem the
    first step overshoots and the residual rises from 40 to 1144 before it begins to fall. Fitting
    the whole history gives 1.75, which is a measurement of that overshoot. Fitting the last few
    steps gives 1.99.

    A large ``weight`` is used on purpose: with a weak nonlinearity Newton converges in two steps
    and there is nothing to fit.
    """
    problem = cubic_problem(weight=weight)
    run = solve_nonlinear(problem, int(points))
    history = run["residual_history"]
    floor = 1e3 * np.finfo(float).eps * max(float(history[0]), 1.0)
    keep = history > floor
    usable = history[keep]
    if usable.size < 3:
        raise SystemExit("not enough steps above the rounding floor to fit an order")
    whole = float(np.polyfit(np.log(usable[:-1]), np.log(usable[1:]), 1)[0])
    # Newton has two phases and only the second one is quadratic. From a zero start the first
    # step here overshoots and the residual RISES from 40 to 1144 before it begins to fall, so
    # fitting the whole history measures the global phase and reads 1.75. The exponent is fitted
    # over the last few steps above the rounding floor, where the local phase is, and both are
    # reported.
    tail = usable[-4:] if usable.size >= 4 else usable
    exponent = float(np.polyfit(np.log(tail[:-1]), np.log(tail[1:]), 1)[0])
    pieces = residual_and_jacobian(problem, int(points))
    j = pieces["jacobian"](run["u"])
    return {
        "residual_history": history,
        "steps_used": int(usable.size - 1),
        "exponent": exponent,
        "exponent_over_the_whole_history": whole,
        "steps_in_the_tail_fit": int(min(usable.size, 4) - 1),
        "the_residual_rises_before_it_falls": bool(history[1] > history[0]),
        "quadratic": abs(exponent - 2.0) < 0.15,
        "iterations": run["iterations"],
        "converged": run["converged"],
        "error": run["error"],
        "jacobian_nonzeros": int(np.count_nonzero(j)),
        "jacobian_entries": int(j.size),
        "sparsity": 1.0 - float(np.count_nonzero(j)) / float(j.size),
        "note": "the Jacobian is tridiagonal, so Newton costs O(m) per step rather than O(m**3)",
    }


def the_bratu_problem_has_a_fold(points: int = 81, coarse: int = 60, refine: int = 40):
    """Two solutions, then none: find the turning point by bisection on solvability.

    Below the critical ``lam`` the problem has a **lower** solution, reachable from a zero start,
    and an **upper** one, reachable only from a large start or by continuation. Above it there is
    no solution at all and Newton diverges.

    Bisecting on "does Newton converge from a zero start" therefore finds the fold, and the
    measurement is compared against `bratu_critical_value`, which comes from the closed form.

    Newton failing here is not a defect in Newton. There is nothing to converge to, and no
    initial guess, damping or tolerance changes that. This is the first problem in the course
    where the right answer to "it does not converge" is "there is no solution".
    """
    lam_star = bratu_critical_value()

    def solvable(lam, guess=None):
        problem = bratu_problem(float(lam))
        try:
            # above the fold Newton runs away and exp overflows, which is the answer rather
            # than an error, so the warning is silenced instead of being printed hundreds of
            # times during the bisection
            with np.errstate(over="ignore", invalid="ignore"):
                run = solve_nonlinear(problem, int(points), guess=guess, tol=1e-10,
                                      max_iter=60)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return None
        if not run["converged"] or not np.isfinite(run["u"]).all():
            return None
        if float(np.max(np.abs(run["u"]))) > 1e3:
            return None
        return run

    # a coarse sweep first, so the bracket is found rather than assumed
    grid = np.linspace(0.1, 6.0, int(coarse))
    peaks, lams = [], []
    for lam in grid:
        run = solvable(lam)
        if run is None:
            break
        peaks.append(float(np.max(run["u"])))
        lams.append(float(lam))
    lo = lams[-1]
    hi = float(grid[len(lams)]) if len(lams) < grid.size else float(grid[-1])
    for _ in range(int(refine)):
        mid = 0.5 * (lo + hi)
        if solvable(mid) is not None:
            lo = mid
        else:
            hi = mid
    measured = 0.5 * (lo + hi)

    def fold_at(grid_points):
        low, high = 0.5 * measured, 1.5 * measured
        for _ in range(40):
            middle = 0.5 * (low + high)
            problem = bratu_problem(float(middle))
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    run = solve_nonlinear(problem, int(grid_points), tol=1e-10,
                                          max_iter=60)
                ok = (run["converged"] and np.isfinite(run["u"]).all()
                      and float(np.max(np.abs(run["u"]))) < 1e3)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                ok = False
            if ok:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    refinement = [(int(n), fold_at(int(n))) for n in (21, 41, 81, 161)]
    hs = np.asarray([1.0 / (n - 1) for n, _ in refinement])
    gaps = np.asarray([abs(v - lam_star) for _, v in refinement])

    # both branches at one value below the fold, the upper one found by continuation
    probe = 0.9 * measured
    lower = solvable(probe)
    upper_guess = None
    for lam in np.linspace(measured * 0.999, probe, 40):
        found = solvable(lam, guess=upper_guess)
        if found is None:
            continue
        upper_guess = found["u"] + 1.0
    upper = solvable(probe, guess=upper_guess)
    return {
        "lam": np.asarray(lams),
        "peak": np.asarray(peaks),
        "measured_fold": measured,
        "closed_form_fold": lam_star,
        "relative_gap": abs(measured - lam_star) / lam_star,
        "agrees_to_four_digits": abs(measured - lam_star) / lam_star < 1e-3,
        "refinement": refinement,
        "gap_against_h": gaps,
        "gap_order_in_h": float(np.polyfit(np.log(hs), np.log(gaps), 1)[0]),
        "the_gap_is_the_discretization_error": abs(
            float(np.polyfit(np.log(hs), np.log(gaps), 1)[0]) - 2.0) < 0.2,
        "solvable_below": solvable(0.99 * measured) is not None,
        "unsolvable_above": solvable(1.01 * measured) is None,
        "probe": probe,
        "lower_peak": float(np.max(lower["u"])) if lower else float("nan"),
        "upper_peak": float(np.max(upper["u"])) if upper else float("nan"),
        "two_branches_found": (lower is not None and upper is not None
                               and abs(float(np.max(upper["u"]))
                                       - float(np.max(lower["u"]))) > 0.5),
        "note": "above the fold there is no solution, so Newton not converging is the right "
                "answer rather than a failure",
    }


# --------------------------------------------------------------------------- method of lines


def semi_discrete_heat(points: int, alpha: float = 1.0, a: float = 0.0, b: float = 1.0):
    """Discretize space only: the heat equation becomes ``u' = (alpha/h**2) T u``.

    Returns the right hand side, the matrix, and the eigenvalues in closed form. Those eigenvalues
    are lesson 77's, and everything about the stability of a time stepper applied to this system
    follows from where they sit relative to its stability region.
    """
    n = int(points)
    if n < 3:
        raise ValueError(f"need at least 3 points, got {n}")
    x = np.linspace(float(a), float(b), n)
    h = float(x[1] - x[0])
    m = n - 2
    scale = float(alpha) / h ** 2
    matrix = scale * (np.diag(np.full(m, -2.0))
                      + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1))
    index = np.arange(1, m + 1)
    values = -4.0 * scale * np.sin(index * math.pi * h / 2.0) ** 2

    def rhs(t, u):
        v = np.asarray(u, dtype=float).ravel()
        padded = np.concatenate([[0.0], v, [0.0]])
        return scale * (padded[2:] - 2.0 * padded[1:-1] + padded[:-2])

    return {"rhs": rhs, "matrix": matrix, "eigenvalues": np.sort(values), "h": h,
            "x": x, "interior": x[1:-1], "unknowns": m, "alpha": float(alpha),
            "stiffness_ratio": float(np.max(np.abs(values)) / np.min(np.abs(values)))}


def the_step_limit_comes_from_the_ode_side(sizes=(11, 21, 41), name: str = "euler"):
    """Derive lesson 78's ``r <= 1/2`` entirely from lesson 72's stability region.

    Euler's stability region on the negative real axis is ``h |lambda| <= 2``. The most negative
    eigenvalue of the semi-discrete operator is ``-4 alpha / h**2 * sin^2((m pi h)/2)``, which
    approaches ``-4 alpha / h**2``. Putting the two together gives

        k <= 2 / |lambda_min| = h**2 / (2 alpha sin^2(...)) -> h**2 / (2 alpha) ,

    which is ``r <= 1/2``. The two derivations are completely independent, one from a Fourier
    substitution into the scheme and one from an ODE stability region, and the measurement here is
    how closely they agree at a finite grid.

    They do not agree exactly, and the discrepancy is meaningful: the discrete eigenvalue is
    slightly smaller in modulus than ``4 alpha / h**2``, so the ODE derivation allows a step
    slightly **larger** than ``1/2``, by a factor ``1/sin^2(m pi h / 2)``. That is the same
    ``O(h**2)`` gap between a discrete eigenvalue and its continuous limit that lesson 75 found in
    the resonance of a boundary value problem.
    """
    rows = []
    for n in sizes:
        system = semi_discrete_heat(int(n))
        worst = float(np.min(system["eigenvalues"]))
        limit = None
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if abs(complex(growth_function(name)(mid * worst))) <= 1.0:
                lo = mid
            else:
                hi = mid
        limit = 0.5 * (lo + hi)
        ratio = limit * system["alpha"] / system["h"] ** 2
        rows.append({
            "points": int(n),
            "h": system["h"],
            "worst_eigenvalue": worst,
            "continuous_limit": -4.0 / system["h"] ** 2,
            "largest_stable_step": limit,
            "mesh_ratio_it_allows": ratio,
            "over_one_half": ratio / 0.5,
            "stiffness_ratio": system["stiffness_ratio"],
        })
    return {
        "rows": rows,
        "method": name,
        "the_two_derivations_agree": all(
            abs(row["mesh_ratio_it_allows"] - 0.5) < 0.02 for row in rows),
        "worst_disagreement": max(abs(row["mesh_ratio_it_allows"] - 0.5) for row in rows),
        "gap_order_in_h": float(np.polyfit(
            np.log(np.asarray([row["h"] for row in rows])),
            np.log(np.asarray([abs(row["mesh_ratio_it_allows"] - 0.5) for row in rows])),
            1)[0]),
        "the_gap_is_second_order": abs(float(np.polyfit(
            np.log(np.asarray([row["h"] for row in rows])),
            np.log(np.asarray([abs(row["mesh_ratio_it_allows"] - 0.5) for row in rows])),
            1)[0]) - 2.0) < 0.1,
        "the_ode_side_allows_slightly_more": all(
            row["mesh_ratio_it_allows"] >= 0.5 - 1e-12 for row in rows),
        "the_gap_shrinks_with_the_grid": all(
            rows[i]["over_one_half"] > rows[i + 1]["over_one_half"]
            for i in range(len(rows) - 1)),
        "stiffness_grows": all(
            rows[i]["stiffness_ratio"] < rows[i + 1]["stiffness_ratio"]
            for i in range(len(rows) - 1)),
    }


def an_adaptive_solver_finds_the_limit_by_itself(points: int = 41,
                                                 tolerances=(1e-4, 1e-6, 1e-8, 1e-10),
                                                 t_end: float = 0.05):
    """Hand the semi-discrete system to lesson 70's adaptive solver and watch what it picks.

    The solver is told a tolerance and nothing else. It knows no stability theory, and it does not
    need to: a step above the limit produces an error estimate that fails the test, so the
    controller rejects it and shrinks. The step it settles on is therefore the **stability** limit
    rather than an accuracy limit, and the giveaway is that tightening the tolerance by six orders
    of magnitude barely moves it.

    That is the practical content of stiffness: an explicit solver on a stiff problem is not being
    careful, it is being held.
    """
    system = semi_discrete_heat(int(points))
    limit = 2.0 / abs(float(np.min(system["eigenvalues"])))
    start = np.sin(math.pi * system["interior"])
    rows = []
    for tol in tolerances:
        run = adaptivestep.solve(system["rhs"], 0.0, start, float(t_end), tol=float(tol),
                                 name="dormand prince")
        steps = np.diff(run["t"])
        rows.append({
            "tolerance": float(tol),
            "accepted": int(run["accepted"]),
            "median_step": float(np.median(steps)),
            "over_the_euler_limit": float(np.median(steps)) / limit,
            "largest_step": float(np.max(steps)),
        })
    medians = np.asarray([row["median_step"] for row in rows])
    return {
        "rows": rows,
        "euler_limit": limit,
        "dormand_prince_limit": _real_axis_limit("dormand prince")
        / abs(float(np.min(system["eigenvalues"]))),
        "step_spread": float(np.max(medians) / np.min(medians)),
        "the_tolerance_barely_matters": float(np.max(medians) / np.min(medians)) < 2.0,
        "tolerance_span": float(max(tolerances) / min(tolerances)),
        "it_lands_near_the_stability_limit": abs(
            float(np.median(medians)) / (_real_axis_limit("dormand prince")
                                         / abs(float(np.min(system["eigenvalues"]))))
            - 1.0) < 0.2,
        "note": "six orders of magnitude of tolerance move the step by less than a factor of 2, "
                "because the step is set by stability and not by accuracy",
    }


def _real_axis_limit(name: str, hi: float = 20.0) -> float:
    """How far along the negative real axis a method's stability region reaches.

    Built from the tableau directly rather than from `stability.growth_function`, because the
    embedded pairs of lesson 70 are not in that module's list of classical tableaux, and the
    growth factor of an explicit method is the same expression whatever list it came from:
    ``R(z) = 1 + z b^T (I - zA)^{-1} 1``.
    """
    a, b, _b_hat, _c, _p, _q = adaptivestep.pair(name)
    eye = np.eye(b.size)
    ones = np.ones(b.size)

    def growth(z):
        return 1.0 + z * (b @ np.linalg.solve(eye - z * a, ones))

    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if abs(growth(-mid)) <= 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lines_against_a_direct_scheme(points: int = 41, t_end: float = 0.02,
                                  steps: int = 400):
    """The method of lines and a direct scheme are the same scheme, and the errors separate.

    Discretize space and apply Euler in time: that **is** lesson 78's explicit scheme, written
    differently. The measurement confirms it at an exact zero, the two arrays agreeing bit for bit.

    Apply RK4 to the same semi-discrete system instead and you get a scheme nobody derived by
    hand, fourth order in time and with a larger stability region. On this run it is **less
    accurate overall**, which is worth understanding rather than explaining away.

    The reason is that there are two errors and they are not independent:

    * the **space** error, from replacing ``u_xx`` by a second difference. The semi-discrete
      solution decays at ``-4/h**2 sin^2(pi h/2)`` instead of ``-pi**2``, which is slower, so it
      ends up too **large**.
    * the **time** error. Euler's amplification ``(1 + k lam)**n`` is smaller than ``exp(k lam n)``
      for negative ``lam``, so Euler decays too fast and ends up too **small**.

    The two have opposite signs and partly cancel. RK4 removes the time error almost entirely and
    so removes the cancellation with it, leaving the full space error. Measured against the
    **semi-discrete** solution, which isolates the time error, RK4 is better by a factor of about
    ``1e10``; measured against the PDE solution it is worse by a factor of 2. Both are reported,
    because quoting either one alone would be misleading.
    """
    system = semi_discrete_heat(int(points))
    start = np.sin(math.pi * system["interior"])
    k = float(t_end) / int(steps)
    r = system["alpha"] * k / system["h"] ** 2

    direct = start.copy()
    for _ in range(int(steps)):
        padded = np.concatenate([[0.0], direct, [0.0]])
        direct = direct + r * (padded[2:] - 2.0 * padded[1:-1] + padded[:-2])

    euler = ivp.integrate(system["rhs"], 0.0, start, float(t_end), int(steps),
                          step_fn=ivp.euler_step)
    runge = ivp.integrate(system["rhs"], 0.0, start, float(t_end), int(steps),
                          step_fn=_rk4_step)

    # the semi-discrete solution, exactly: the initial state is an eigenvector of the matrix
    rate = float(system["eigenvalues"][-1])
    semi = math.exp(rate * float(t_end)) * start
    pde = math.exp(-math.pi ** 2 * float(t_end)) * start

    def gap(a, b):
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    return {
        "mesh_ratio": r,
        "semi_discrete_rate": rate,
        "continuous_rate": -math.pi ** 2,
        "they_are_the_same_scheme": gap(direct, euler["y"][-1]),
        "identical_to_rounding": gap(direct, euler["y"][-1]) < 1e-13,
        "euler_time_error": gap(euler["y"][-1], semi),
        "rk4_time_error": gap(runge["y"][-1], semi),
        "rk4_beats_euler_in_time_by": gap(euler["y"][-1], semi)
        / max(gap(runge["y"][-1], semi), 1e-300),
        "space_error": gap(semi, pde),
        "euler_total_error": gap(euler["y"][-1], pde),
        "rk4_total_error": gap(runge["y"][-1], pde),
        "rk4_total_over_euler_total": gap(runge["y"][-1], pde)
        / max(gap(euler["y"][-1], pde), 1e-300),
        "the_two_errors_have_opposite_signs": bool(
            (float(semi[len(semi) // 2]) - float(pde[len(pde) // 2]))
            * (float(euler["y"][-1][len(semi) // 2]) - float(semi[len(semi) // 2])) < 0.0),
        "rk4_is_the_space_error": abs(gap(runge["y"][-1], pde) - gap(semi, pde))         < 0.01 * gap(semi, pde),
        "note": "Euler's time error and the space error have opposite signs and partly cancel; "
                "RK4 removes the time error and the cancellation with it",
    }


def _rk4_step(f, t, y, h, _counter=None):
    """Classical RK4, used here so the comparison does not depend on which module owns it."""
    v = np.asarray(y, dtype=float)
    if _counter is not None:
        _counter[0] += 4
    k1 = np.asarray(f(t, v), dtype=float)
    k2 = np.asarray(f(t + 0.5 * h, v + 0.5 * h * k1), dtype=float)
    k3 = np.asarray(f(t + 0.5 * h, v + 0.5 * h * k2), dtype=float)
    k4 = np.asarray(f(t + h, v + h * k3), dtype=float)
    return v + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def fisher_front_speed(diffusion: float, growth: float, decay: float) -> float:
    """The speed a Fisher front settles at, given how fast the initial profile decays.

    The textbook answer, ``2 sqrt(D r)``, is the **minimum** speed and not the general one. For an
    initial profile falling off like ``exp(-a x)`` the front is *pulled* by its own leading edge,
    which grows and translates at ``c(a) = D a + r / a``. That expression is minimised at
    ``a* = sqrt(r/D)``, where it equals ``2 sqrt(D r)``, and the rule is:

    * a **shallow** initial front, ``a < a*``, keeps its own leading edge and travels at
      ``D a + r / a``, which is faster than the minimum,
    * a **steep** one, ``a >= a*``, cannot keep it, and the front relaxes to the minimum
      ``2 sqrt(D r)``.

    Quoting the minimum for a shallow front is out by a factor of 8 on the parameters used in
    `a_nonlinear_time_dependent_problem`, and quoting the minimum for a steep front is right only
    in the limit: the approach to it is **algebraically slow**, which the same function measures.
    """
    d, r, a = float(diffusion), float(growth), float(decay)
    if d <= 0.0 or r <= 0.0 or a <= 0.0:
        raise ValueError("diffusion, growth and decay must all be positive")
    critical = math.sqrt(r / d)
    if a < critical:
        return d * a + r / a
    return 2.0 * math.sqrt(d * r)


def a_nonlinear_time_dependent_problem(points: int = 801, t_end: float = 40.0,
                                       diffusion: float = 1e-2, growth: float = 1.0,
                                       decays=(2.0, 20.0), span=(-2.0, 18.0),
                                       windows=((2, 5), (5, 10), (10, 20), (20, 40))):
    """Fisher's equation ``u_t = D u_xx + r u (1 - u)``, by the method of lines.

    A logistic reaction with diffusion, whose solutions are travelling fronts. Nothing about the
    time stepping changes because the problem is nonlinear: the right hand side is a function, and
    Part 10's solvers only ever call it.

    The property checked is the **front speed**, not an error against a manufactured solution. An
    error measurement would test the discretization; the speed tests whether the behaviour the
    equation is famous for survived it. Two regimes, and neither one is the textbook sentence.

    **A shallow front** (``a = 2``, well below ``a* = 10``) travels at ``D a + r / a = 0.52``, and
    the measurement agrees to half a per cent. The often quoted ``2 sqrt(D r) = 0.2`` is out by a
    factor of 2.6 here, because it is a lower bound and not the answer.

    **A steep front** (``a = 20``, above ``a*``) does relax to ``2 sqrt(D r)``, and it does so
    **algebraically slowly**. Fitted over ``t`` in ``[2, 5]`` the speed is 0.149, and only by
    ``[20, 40]`` has it reached 0.195. Bramson's correction says the instantaneous speed is
    ``c* - 3 / (2 a* t)``, and the measurement matches that to a few per cent in every window and
    to 0.2 per cent in the last one. A run to ``t = 4`` would have measured 0.14 and concluded the
    theory was wrong.
    """
    n = int(points)
    x = np.linspace(float(span[0]), float(span[1]), n)
    h = float(x[1] - x[0])
    d, r = float(diffusion), float(growth)
    critical = math.sqrt(r / d)
    minimum = 2.0 * math.sqrt(d * r)

    def rhs(t, u):
        v = np.asarray(u, dtype=float).ravel()
        padded = np.concatenate([[1.0], v, [0.0]])
        second = (padded[2:] - 2.0 * padded[1:-1] + padded[:-2]) / h ** 2
        return d * second + r * v * (1.0 - v)

    rows = []
    for decay in decays:
        a = float(decay)
        start_profile = 1.0 / (1.0 + np.exp(a * x[1:-1]))
        run = adaptivestep.solve(rhs, 0.0, start_profile, float(t_end), tol=1e-8,
                                 name="dormand prince")
        times, fronts = [], []
        for level, profile in zip(run["t"], run["y"]):
            crossing = np.flatnonzero(profile < 0.5)
            if crossing.size == 0 or crossing[0] == 0:
                continue
            i = int(crossing[0])
            left, right = profile[i - 1], profile[i]
            times.append(float(level))
            fronts.append(float(x[i] + h * (left - 0.5) / max(left - right, 1e-300)))
        times = np.asarray(times)
        fronts = np.asarray(fronts)
        # drop every sample after the front nears the right boundary, where it stalls against the
        # imposed value. Leaving them in made the shallow front's late speed read 0.42 instead of
        # 0.52, which is a measurement of the domain rather than of the equation.
        room = fronts < float(span[1]) - 2.0
        stopped_early = bool(not np.all(room))
        times, fronts = times[room], fronts[room]
        pieces = []
        for lo, hi in windows:
            inside = (times >= float(lo)) & (times <= float(hi))
            if int(np.count_nonzero(inside)) < 3:
                continue
            speed = float(np.polyfit(times[inside], fronts[inside], 1)[0])
            mid = 0.5 * (float(lo) + float(hi))
            bramson = minimum - 3.0 / (2.0 * critical * mid)
            pieces.append({
                "window": (float(lo), float(hi)),
                "speed": speed,
                "bramson": bramson,
                "relative_gap": abs(speed - bramson) / abs(bramson),
            })
        late = times >= 0.5 * float(times[-1])
        settled = float(np.polyfit(times[late], fronts[late], 1)[0])
        predicted = fisher_front_speed(d, r, a)
        rows.append({
            "decay": a,
            "critical_decay": critical,
            "steep": a >= critical,
            "t": times, "front": fronts,
            "windows": pieces,
            "late_speed": settled,
            "predicted_speed": predicted,
            "textbook_minimum": minimum,
            "relative_error": abs(settled - predicted) / predicted,
            "agrees": abs(settled - predicted) / predicted < 0.05,
            "the_minimum_would_be_wrong_by": abs(settled - minimum) / minimum,
            "speeds_up": all(pieces[i]["speed"] < pieces[i + 1]["speed"]
                             for i in range(len(pieces) - 1)) if len(pieces) > 1 else False,
            "accepted": int(run["accepted"]),
            "rejected": int(run["rejected"]),
            "left_the_domain": stopped_early,
            "last_usable_time": float(times[-1]),
            "final": run["y"][-1],
        })
    shallow = next(row for row in rows if not row["steep"])
    steep = next(row for row in rows if row["steep"])
    return {
        "rows": rows,
        "x": x,
        "critical_decay": critical,
        "textbook_minimum": minimum,
        "every_late_speed_matches_its_prediction": all(row["agrees"] for row in rows),
        "the_minimum_is_wrong_for_a_shallow_front":
            shallow["the_minimum_would_be_wrong_by"] > 1.0,
        "how_wrong_the_minimum_is": shallow["the_minimum_would_be_wrong_by"],
        "the_steep_front_speeds_up_towards_the_minimum": steep["speeds_up"],
        "bramson_matches_every_window": all(
            piece["relative_gap"] < 0.06 for piece in steep["windows"]),
        "worst_bramson_gap": max(piece["relative_gap"] for piece in steep["windows"]),
        "the_last_window_matches_to": min(
            piece["relative_gap"] for piece in steep["windows"]),
        "note": "the solver never learns that the problem is nonlinear; it only calls the right "
                "hand side. The speed depends on the initial decay rate, and a steep front "
                "reaches the minimum only algebraically slowly",
    }
