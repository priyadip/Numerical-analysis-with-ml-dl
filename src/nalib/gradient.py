"""Descent with a gradient: steepest descent, Newton, and the line searches that rescue both.

Where the work is
-----------------
Lesson 84 said the condition number of the Hessian is the aspect ratio of the set floating point
cannot separate. That was a statement about **precision**. Here the same number becomes a statement
about **speed**, and the two are unrelated: a problem can be perfectly conditioned and unsolvable,
and badly conditioned and easy.

Steepest descent goes downhill. With an exact line search on a quadratic, Kantorovich's inequality
gives the sharp bound

    ||e_(k+1)||_A / ||e_k||_A  <=  (kappa - 1) / (kappa + 1) ,

so the number of steps grows **linearly in kappa**. Newton's method solves ``H p = -g`` instead,
which is scale invariant and converges quadratically, and pays for it with an ``n**2`` Hessian, an
``n**3`` solve, and no guarantee at all far from the minimum.

What the measurements here show
-------------------------------
* **The Kantorovich bound is sharp, and only in enough dimensions.** At ten variables the measured
  asymptotic rate matches ``(kappa-1)/(kappa+1)`` to five digits at ``kappa = 10, 100, 1000``. At
  five variables it is within 0.1 per cent, at three within 2 per cent, and at **two** it is far
  below the bound and depends on where the search started. **Two dimensions is not enough to see
  the worst case**, which matters because two dimensions is where optimizers are usually drawn.
* The step count grows linearly in ``kappa``: 123, 1211, 12009 steps at ``kappa = 10, 100, 1000``,
  ratios of 9.85 and 9.92 for a tenfold increase.
* **A start along one eigenvector converges in one step**, at every condition number. What is left
  after that step is rounding amplified by ``kappa``, not plain rounding: ``6.8e-15`` at
  ``kappa = 10`` and ``8.9e-10`` at ``kappa = 10**5``, both within a factor of 40 of ``eps*kappa``.
  It is the same degeneracy lesson 82 found for conjugate gradients, and it is why a test problem
  has to be chosen rather than reached for.
* Newton on Rosenbrock reaches the minimum in 7 steps at two variables, 34 at ten and 46 at twenty,
  and its gradient norm is **not monotone** on the way: at two variables it goes
  ``2.3e+02, 4.6e+00, 1.4e+03, 4.7e-01, 2.5e+01, 8.6e-06, 8.3e-09``, rising by a factor of 300 at
  step 2 and by 1850 at ten variables.
* The convergence is quadratic and the evidence is ``e_(k+1) / e_k**2``, which comes out at
  ``112, 145, 153`` at two, ten and twenty variables: constant to within a factor of ``1.4`` across
  a tenfold change in size. A **fitted order** cannot be quoted here for lesson 85's reason, since
  squaring the error each step leaves only two usable points before the floor.
* **The plain iteration stops at whatever stationary point it reaches.** At four variables it
  converges to an outright **saddle**, gradient ``1.5e-14``, and at five and seven to a second
  local minimum with ``f`` near 3.9 instead of 0. Every stopping test passes in all three.
* The Hessian is indefinite at ``21.5`` per cent of sampled points in the box at two variables,
  ``47.7`` at five and ``70.8`` at ten, so the plain Newton direction is not even a descent
  direction over most of the domain in higher dimensions.
* **The safeguards help and hurt.** Shifting the Hessian and backtracking rescues the saddle at
  four variables, changes nothing at five and seven, and takes the runs at six and ten variables
  from the global minimum to a worse one. Guaranteed descent is not guaranteed descent to the
  right place, and that is measured here rather than assumed.
* Steepest descent needs 19436 iterations on Rosenbrock at two variables where Newton needs 7.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .optimize import all_problems, quadratic, rosenbrock


# --------------------------------------------------------------------------- line searches


def backtracking(f, value, slope, where, direction, alpha: float = 1.0,
                 shrink: float = 0.5, wolfe_c1: float = 1e-4,
                 max_halvings: int = 80) -> dict:
    """Halve the step until Armijo's sufficient decrease condition holds.

    Armijo asks for ``f(x + a p) <= f(x) + c1 a g.p``, that is, for a fraction ``c1`` of the
    decrease the linear model predicts. Accepting any decrease at all is not enough: a sequence of
    ever smaller improvements can converge to a non-stationary point, which is exactly the failure
    Nelder-Mead exhibited in lesson 85 and this condition rules out.
    """
    p = np.asarray(direction, dtype=float).ravel()
    x = np.asarray(where, dtype=float).ravel()
    directional = float(np.asarray(slope, dtype=float).ravel() @ p)
    if directional >= 0.0:
        raise ValueError("the direction must go downhill, got a slope of "
                         f"{directional:.3e}")
    step = float(alpha)
    calls = 0
    for _ in range(int(max_halvings)):
        trial = float(f(x + step * p))
        calls += 1
        if trial <= value + wolfe_c1 * step * directional:
            return {"step": step, "f": trial, "calls": calls, "accepted": True,
                    "armijo": True}
        step *= float(shrink)
    return {"step": step, "f": float(f(x + step * p)), "calls": calls + 1,
            "accepted": False, "armijo": False}


def strong_wolfe(f, gradient, where, direction, alpha: float = 1.0,
                 wolfe_c1: float = 1e-4, wolfe_c2: float = 0.9,
                 max_steps: int = 40) -> dict:
    """A step satisfying both strong Wolfe conditions, by bracketing then bisecting.

    Armijo alone allows steps that are far too **short**, which is the other way to fail: the
    iterates stop moving while the gradient is still large. The curvature condition
    ``|g(x + a p).p| <= c2 |g(x).p|`` rules that out by insisting the slope has flattened.

    ``c2 = 0.9`` is the usual choice for a Newton-like direction and ``0.1`` for conjugate
    gradients, because the second needs a more accurate step to keep its conjugacy.
    """
    x = np.asarray(where, dtype=float).ravel()
    p = np.asarray(direction, dtype=float).ravel()
    start_value = float(f(x))
    start_slope = float(np.asarray(gradient(x), dtype=float).ravel() @ p)
    if start_slope >= 0.0:
        raise ValueError(f"the direction must go downhill, got {start_slope:.3e}")
    calls = 0

    def probe(step):
        nonlocal calls
        calls += 1
        point = x + step * p
        return float(f(point)), float(np.asarray(gradient(point), dtype=float).ravel() @ p)

    lo, hi = 0.0, float(alpha)
    value, slope = probe(hi)
    # phase one: grow until a bracket exists
    for _ in range(int(max_steps)):
        if value > start_value + wolfe_c1 * hi * start_slope:
            break
        if abs(slope) <= -wolfe_c2 * start_slope:
            return {"step": hi, "f": value, "slope": slope, "calls": calls,
                    "armijo": True, "curvature": True, "accepted": True}
        if slope >= 0.0:
            break
        lo, hi = hi, 2.0 * hi
        value, slope = probe(hi)
    # phase two: bisect inside the bracket
    for _ in range(int(max_steps)):
        mid = 0.5 * (lo + hi)
        value, slope = probe(mid)
        armijo = value <= start_value + wolfe_c1 * mid * start_slope
        curvature = abs(slope) <= -wolfe_c2 * start_slope
        if armijo and curvature:
            return {"step": mid, "f": value, "slope": slope, "calls": calls,
                    "armijo": True, "curvature": True, "accepted": True}
        if not armijo or slope >= 0.0:
            hi = mid
        else:
            lo = mid
    mid = 0.5 * (lo + hi)
    value, slope = probe(mid)
    return {"step": mid, "f": value, "slope": slope, "calls": calls,
            "armijo": value <= start_value + wolfe_c1 * mid * start_slope,
            "curvature": abs(slope) <= -wolfe_c2 * start_slope,
            "accepted": False}


def exact_step_on_a_quadratic(matrix, slope) -> float:
    """The exact minimizer along ``-g`` for ``(1/2) x^T A x``, which is ``g.g / g^T A g``.

    Available only because the problem is a quadratic. Every measurement of a theoretical rate in
    this module uses it, so the measurement is of the method and not of a line search.
    """
    g = np.asarray(slope, dtype=float).ravel()
    a = np.asarray(matrix, dtype=float)
    bottom = float(g @ a @ g)
    if bottom <= 0.0:
        raise ValueError("the matrix must be positive definite along the gradient")
    return float(g @ g) / bottom


# --------------------------------------------------------------------------- the methods


def steepest_descent(problem, start=None, tol: float = 1e-10, max_steps: int = 200000,
                     search: str = "backtracking") -> dict:
    """Go downhill, with an exact line search on a quadratic or a backtracking one otherwise."""
    x = np.asarray(problem["start"] if start is None else start, dtype=float).ravel()
    f, g = problem["f"], problem["gradient"]
    exact = search == "exact"
    matrix = problem.get("matrix")
    if exact and matrix is None:
        raise ValueError("an exact line search needs a quadratic with a stored matrix")
    history, calls = [], 0
    for step_number in range(int(max_steps)):
        slope = np.asarray(g(x), dtype=float).ravel()
        size = float(np.linalg.norm(slope))
        history.append(size)
        if size <= float(tol):
            break
        if exact:
            x = x - exact_step_on_a_quadratic(matrix, slope) * slope
        else:
            out = backtracking(f, float(f(x)), slope, x, -slope)
            calls += out["calls"]
            if not out["accepted"]:
                break
            x = x - out["step"] * slope
    return {"x": x, "f": float(f(x)), "steps": step_number,
            "gradient_history": np.asarray(history), "line_search_calls": calls,
            "converged": bool(history and history[-1] <= float(tol))}


def newton(problem, start=None, tol: float = 1e-12, max_steps: int = 200,
           safeguard: bool = False) -> dict:
    """Newton's method for a minimum: solve ``H p = -g`` and step.

    With ``safeguard=False`` this is the textbook iteration, taking the full step and trusting the
    Hessian. That is quadratically convergent near a minimum and has **no** global guarantee: the
    step is a descent direction only when ``H`` is positive definite, and even then a full step can
    overshoot.

    With ``safeguard=True`` two repairs are applied. The Hessian is shifted by ``tau I``, with
    ``tau`` doubled until Cholesky succeeds, which makes the direction a descent direction; and the
    step is then cut by backtracking until Armijo holds. Both are inactive close to a minimum,
    where the plain method is already right, and the measurement reports how often each fires.
    """
    x = np.asarray(problem["start"] if start is None else start, dtype=float).ravel()
    f, g, h = problem["f"], problem["gradient"], problem["hessian"]
    history, shifts, cuts = [], 0, 0
    indefinite = 0
    for step_number in range(int(max_steps)):
        slope = np.asarray(g(x), dtype=float).ravel()
        size = float(np.linalg.norm(slope))
        history.append(size)
        if size <= float(tol):
            break
        matrix = np.asarray(h(x), dtype=float)
        matrix = 0.5 * (matrix + matrix.T)
        if float(np.min(np.linalg.eigvalsh(matrix))) <= 0.0:
            indefinite += 1
        if safeguard:
            scale = max(float(np.max(np.abs(np.diag(matrix)))), 1.0)
            tau = 0.0
            while True:
                try:
                    np.linalg.cholesky(matrix + tau * np.eye(x.size))
                    break
                except np.linalg.LinAlgError:
                    tau = max(2.0 * tau, 1e-3 * scale)
                    shifts += 1
            direction = np.linalg.solve(matrix + tau * np.eye(x.size), -slope)
            out = backtracking(f, float(f(x)), slope, x, direction)
            if out["step"] < 1.0:
                cuts += 1
            if not out["accepted"]:
                break
            x = x + out["step"] * direction
        else:
            try:
                x = x + np.linalg.solve(matrix, -slope)
            except np.linalg.LinAlgError:
                break
    return {"x": x, "f": float(f(x)), "steps": step_number,
            "gradient_history": np.asarray(history),
            "indefinite_hessians": indefinite, "shifts": shifts, "cuts": cuts,
            "converged": bool(history and history[-1] <= float(tol))}


# --------------------------------------------------------------------------- measurements


def _quadratic_rate(dimension: int, condition: float, floor: float = 1e-11,
                    max_steps: int = 400000, seed: int = 42) -> dict:
    """Run steepest descent with an exact line search and fit the asymptotic A-norm rate."""
    problem = quadratic(condition=condition, dimension=dimension, seed=seed)
    matrix, centre = problem["matrix"], problem["minimizers"][0]
    values, vectors = np.linalg.eigh(matrix)
    # the worst case start: equal energy in every eigendirection, in the A-norm
    x = centre + vectors @ (1.0 / np.sqrt(values))
    errors = []
    for _ in range(int(max_steps)):
        gap = x - centre
        errors.append(math.sqrt(float(gap @ matrix @ gap)))
        if errors[-1] <= floor * errors[0]:
            break
        slope = matrix @ gap
        x = x - exact_step_on_a_quadratic(matrix, slope) * slope
    errors = np.asarray(errors)
    late = errors[errors.size // 2:]
    rate = float(np.mean(late[1:] / late[:-1])) if late.size > 3 else float("nan")
    return {"dimension": int(dimension), "condition": float(condition),
            "steps": int(errors.size), "measured_rate": rate,
            "bound": (condition - 1.0) / (condition + 1.0),
            "errors": errors}


def the_kantorovich_bound_is_sharp(dimensions=(2, 3, 5, 10),
                                   conditions=(10.0, 100.0, 1000.0)) -> dict:
    """Measure the asymptotic rate of steepest descent against ``(kappa-1)/(kappa+1)``.

    The bound is a **worst case** over starting points, so a measurement can only ever approach it
    from below. What is worth knowing is where it is approached, and the answer turns out to depend
    on the number of variables in a way the bound itself does not mention.
    """
    rows = []
    for n in dimensions:
        for kappa in conditions:
            out = _quadratic_rate(n, kappa)
            rows.append({"dimension": out["dimension"], "condition": out["condition"],
                         "steps": out["steps"], "measured_rate": out["measured_rate"],
                         "bound": out["bound"],
                         "fraction_of_the_bound": out["measured_rate"] / out["bound"]})
    biggest = max(dimensions)
    tight = [r for r in rows if r["dimension"] == biggest]
    smallest = min(dimensions)
    loose = [r for r in rows if r["dimension"] == smallest]
    return {
        "rows": rows,
        "sharp_at_the_largest_dimension": all(
            abs(r["fraction_of_the_bound"] - 1.0) < 1e-3 for r in tight),
        "not_sharp_at_the_smallest": all(
            r["fraction_of_the_bound"] < 0.9 for r in loose),
        "worst_fraction_at_the_smallest": max(r["fraction_of_the_bound"] for r in loose),
        "note": "the bound is attained in enough dimensions and not in two, so a two variable "
                "test problem understates how bad steepest descent is",
    }


def the_step_count_grows_with_the_condition_number(dimension: int = 10,
                                                   conditions=(10.0, 100.0, 1000.0)) -> dict:
    """Count steps to a fixed relative accuracy against ``kappa``, and fit the power."""
    rows = []
    for kappa in conditions:
        out = _quadratic_rate(dimension, kappa)
        rows.append({"condition": float(kappa), "steps": out["steps"],
                     "steps_over_condition": out["steps"] / float(kappa)})
    counts = np.asarray([r["steps"] for r in rows], dtype=float)
    kappas = np.asarray([r["condition"] for r in rows], dtype=float)
    power = float(np.polyfit(np.log(kappas), np.log(counts), 1)[0])
    return {
        "rows": rows,
        "dimension": int(dimension),
        "fitted_power_of_the_condition_number": power,
        "the_growth_is_linear": abs(power - 1.0) < 0.1,
        "note": "each factor of ten in kappa costs a factor of ten in steps, which is the "
                "practical content of the rate bound",
    }


def an_eigenvector_start_is_degenerate(dimension: int = 10,
                                       conditions=(10.0, 1000.0, 100000.0)) -> dict:
    """Start along a single eigenvector and watch steepest descent finish in one step.

    The gradient there is a multiple of the same eigenvector, so the exact line search lands on the
    minimum whatever the condition number is. Lesson 82 found the same degeneracy for conjugate
    gradients, from the other side: a right hand side that is an eigenvector gives a one
    dimensional Krylov space.

    The measurement is the **relative** A-norm error after exactly one step. Counting steps to an
    absolute threshold would measure rounding instead: one exact step leaves an error of order
    ``eps``, and whether that sits under a fixed bar depends on the scaling, which is why a first
    attempt here reported 1000 steps at ``kappa = 10**5``.
    """
    rows = []
    for kappa in conditions:
        problem = quadratic(condition=kappa, dimension=dimension)
        matrix, centre = problem["matrix"], problem["minimizers"][0]
        values, vectors = np.linalg.eigh(matrix)
        for which, label in ((0, "smallest eigenvalue"), (values.size - 1, "largest")):
            gap = vectors[:, which]
            before = math.sqrt(float(gap @ matrix @ gap))
            slope = matrix @ gap
            after_gap = gap - exact_step_on_a_quadratic(matrix, slope) * slope
            after = math.sqrt(float(after_gap @ matrix @ after_gap))
            rows.append({"condition": float(kappa), "eigenvector": label,
                         "relative_error_after_one_step": after / before,
                         "over_eps_times_condition":
                             (after / before)
                             / (float(np.finfo(float).eps) * float(kappa))})
    spread = _quadratic_rate(dimension, 1000.0)
    return {
        "rows": rows,
        "one_step_is_enough": all(
            r["relative_error_after_one_step"] < 1e-8 for r in rows),
        "worst_relative_error": max(r["relative_error_after_one_step"] for r in rows),
        "worst_over_eps_times_condition": max(
            r["over_eps_times_condition"] for r in rows),
        "the_residue_is_rounding_times_the_condition_number": max(
            r["over_eps_times_condition"] for r in rows) < 100.0,
        "a_spread_start_takes": spread["steps"],
        "note": "one exact step solves it, and what is left is rounding amplified by the "
                "condition number rather than plain rounding, so the condition number governs the "
                "residue and not the rate; the hard part of the problem is never excited, and a "
                "test problem has to be chosen rather than reached for",
    }


def newton_is_quadratic_and_not_monotone(dimensions=(2, 10, 20)) -> dict:
    """Newton on Rosenbrock: the gradient rises before it falls, and then falls very fast.

    Both halves are worth naming. The rise is not a bug and not a failure: Newton's method promises
    quadratic convergence **near** a minimum and nothing at all away from one.

    The fall is measured as ``e_(k+1) / e_k**2`` on the last transition rather than as a fitted
    order, and the reason is lesson 85's. Quadratic convergence squares the error each step, so
    between the last iterate above 1 and the rounding floor there are only two or three points, and
    a slope fitted through three points that straddle a rise means nothing. The ratio needs two
    points and says the same thing: roughly constant means quadratic.
    """
    rows = []
    for n in dimensions:
        problem = rosenbrock(n)
        out = newton(problem)
        history = out["gradient_history"]
        usable = history[history > 1e-11]
        last = usable[-2:]
        constant = float(last[1] / last[0] ** 2) if last.size == 2 else float("nan")
        rises = np.diff(history) > 0.0
        rows.append({
            "variables": int(n),
            "steps": int(out["steps"]),
            "distance": float(np.linalg.norm(out["x"] - 1.0)),
            "biggest_rise": float(np.max(history[1:] / history[:-1])),
            "rising_steps": int(np.count_nonzero(rises)),
            "last_error": float(last[0]),
            "next_error": float(last[1]),
            "quadratic_constant": constant,
            "indefinite_hessians": int(out["indefinite_hessians"]),
        })
    constants = np.asarray([r["quadratic_constant"] for r in rows])
    return {
        "rows": rows,
        "every_run_rises_first": all(r["rising_steps"] > 0 for r in rows),
        "biggest_rise_anywhere": max(r["biggest_rise"] for r in rows),
        "constants": constants,
        "spread_in_the_constant": float(np.max(constants) / np.min(constants)),
        "the_constant_is_size_independent": bool(
            float(np.max(constants) / np.min(constants)) < 3.0),
        "note": "e_(k+1) is about 10**2 times e_k**2 at every size, which is quadratic "
                "convergence; the gradient still rises by a large factor first, so any quoted "
                "order has to say which steps it used",
    }


def plain_newton_can_find_the_wrong_minimum(dimensions=(2, 4, 5, 6, 7, 10)) -> dict:
    """Rosenbrock has a second minimum for four to seven variables. Newton finds it at three sizes.

    For ``4 <= n <= 7`` the n-dimensional Rosenbrock function has a local minimum in addition to the
    global one at ``(1, ..., 1)``, roughly two away and near ``(-1, 1, ..., 1)`` but not at it.
    From the standard starting point the plain Newton iteration converges to the wrong one at some
    sizes and the right one at others.

    Nothing about those runs announces it. The gradient reaches ``10**-14``, the Hessian is
    positive definite, every stopping test passes. It **is** a local minimum, and it is not the one
    wanted. The classification is done here rather than assumed, because "converged somewhere
    wrong" and "converged to a saddle" are different failures and only one of them is happening.
    """
    from .optimize import classify

    rows = []
    for n in dimensions:
        problem = rosenbrock(n)
        out = newton(problem)
        far = np.full(n, 1.0)
        far[0] = -1.0
        distance = float(np.linalg.norm(out["x"] - 1.0))
        rows.append({
            "variables": int(n),
            "steps": int(out["steps"]),
            "final_gradient": float(out["gradient_history"][-1]),
            "f_there": float(out["f"]),
            "to_the_global_minimum": distance,
            "to_(-1,1,...,1)": float(np.linalg.norm(out["x"] - far)),
            "verdict": classify(problem["hessian"](out["x"])),
            "found_the_global_one": distance < 1e-6,
        })
    wrong = [r for r in rows if not r["found_the_global_one"]]
    return {
        "rows": rows,
        "sizes_that_stopped_elsewhere": [r["variables"] for r in wrong],
        "it_happens": bool(wrong),
        "wrong_stops_that_are_minima": [r["variables"] for r in wrong
                                        if r["verdict"] == "minimum"],
        "wrong_stops_that_are_saddles": [r["variables"] for r in wrong
                                         if r["verdict"] == "saddle"],
        "it_stops_at_saddles_too": any(r["verdict"] == "saddle" for r in wrong),
        "every_wrong_stop_converged_cleanly": all(
            r["final_gradient"] < 1e-9 for r in wrong),
        "the_wrong_stops_have_a_larger_value": all(
            r["f_there"] > 1e-6 for r in wrong),
        "note": "the plain iteration solves grad f = 0 and does not look at the sign of the "
                "Hessian, so it stops at whatever stationary point it reaches: a second local "
                "minimum at four of these sizes and an outright saddle at one of them, both with "
                "a gradient at 10**-14 and every stopping test satisfied",
    }


def the_hessian_is_not_always_positive_definite(dimension: int = 2, samples: int = 400,
                                                radius: float = 2.0,
                                                seed: int = 42) -> dict:
    """How often Rosenbrock's Hessian is indefinite, on the path and in the box.

    Where it is indefinite the Newton direction is not a descent direction at all, so the plain
    method can and does move uphill. The safeguarded version shifts the Hessian until Cholesky
    succeeds, which is the cheapest repair that provably works.
    """
    problem = rosenbrock(dimension)
    rng = np.random.default_rng(int(seed))
    bad = 0
    for _ in range(int(samples)):
        point = rng.uniform(-radius, radius, size=dimension)
        if float(np.min(np.linalg.eigvalsh(problem["hessian"](point)))) <= 0.0:
            bad += 1
    plain = newton(problem)
    safe = newton(problem, safeguard=True)
    return {
        "samples": int(samples),
        "indefinite_in_the_box": bad / float(samples),
        "indefinite_on_the_plain_path": plain["indefinite_hessians"],
        "plain_steps": plain["steps"],
        "safeguarded_steps": safe["steps"],
        "shifts_applied": safe["shifts"],
        "steps_cut": safe["cuts"],
        "both_converged": bool(plain["converged"] and safe["converged"]),
        "both_found_the_same_point": float(np.linalg.norm(plain["x"] - safe["x"])),
        "note": "the safeguards fire only away from the minimum and both runs end at the same "
                "point, so the repair costs steps and not accuracy",
    }


def the_safeguards_help_and_hurt(dimensions=(2, 4, 5, 6, 7, 10)) -> dict:
    """Plain Newton against safeguarded Newton on Rosenbrock, size by size.

    The safeguards are the standard pair: shift the Hessian by ``tau I`` until Cholesky succeeds,
    so the direction goes downhill, then backtrack until Armijo holds, so the step does not
    overshoot. Together they guarantee that ``f`` decreases every step and that the iterates
    converge to a **stationary point**.

    That guarantee is worth having and it is worth being exact about what it is not. Descending
    reliably is not the same as descending to the right place, and the measurement shows all three
    outcomes: the safeguards rescue a run that stopped at a saddle, leave two runs where they were,
    and steer two runs that had already found the global minimum into a worse one.
    """
    from .optimize import classify

    rows = []
    for n in dimensions:
        problem = rosenbrock(n)
        plain = newton(problem)
        safe = newton(problem, safeguard=True, max_steps=2000)
        plain_far = float(np.linalg.norm(plain["x"] - 1.0))
        safe_far = float(np.linalg.norm(safe["x"] - 1.0))
        rows.append({
            "variables": int(n),
            "plain_steps": int(plain["steps"]),
            "plain_distance": plain_far,
            "plain_verdict": classify(problem["hessian"](plain["x"])),
            "safe_steps": int(safe["steps"]),
            "safe_distance": safe_far,
            "safe_verdict": classify(problem["hessian"](safe["x"])),
            "shifts": int(safe["shifts"]),
            "cuts": int(safe["cuts"]),
            "rescued": bool(plain_far > 1e-6 and safe_far < 1e-6),
            "spoiled": bool(plain_far < 1e-6 and safe_far > 1e-6),
        })
    rescued = [r["variables"] for r in rows if r["rescued"]]
    spoiled = [r["variables"] for r in rows if r["spoiled"]]
    return {
        "rows": rows,
        "rescued": rescued,
        "spoiled": spoiled,
        "it_rescues_the_saddle": any(
            r["rescued"] and r["plain_verdict"] == "saddle" for r in rows),
        "it_also_makes_things_worse": bool(spoiled),
        "every_safeguarded_run_reached_a_stationary_point": all(
            r["safe_verdict"] in ("minimum", "maximum", "saddle") for r in rows),
        "most_cuts": max(r["cuts"] for r in rows),
        "note": "the safeguards guarantee descent to a stationary point and nothing about which "
                "one; here they rescue the saddle, change nothing at two sizes, and take two runs "
                "that had found the global minimum to a worse one",
    }


def wolfe_is_stricter_than_armijo(seed: int = 42, variables: int = 2) -> dict:
    """Where the curvature condition bites, and where it turns out to be free.

    Armijo asks the step not to be too **long**. The curvature condition asks it not to be too
    **short**, which is the failure Armijo alone permits: a sequence of ever tinier accepted steps
    that decreases ``f`` every time and stops moving while the gradient is still large.

    Two families are tested because one of them was not enough. On Rosenbrock, backtracking from
    ``alpha = 1`` halves several times before Armijo holds, and by then the step is small enough
    that the curvature condition holds too, so the extra condition buys nothing and the strong
    Wolfe search returns the identical step for several more gradient evaluations. On a badly
    scaled quadratic the full step passes Armijo immediately and is hundreds of times too short,
    and there the curvature condition is the only thing that notices.
    """
    rng = np.random.default_rng(int(seed))
    rows = []

    problem = rosenbrock(max(2, int(variables)))
    f, g = problem["f"], problem["gradient"]
    for _ in range(6):
        x = rng.uniform(-2.0, 2.0, size=problem["dimension"])
        slope = np.asarray(g(x), dtype=float)
        if float(np.linalg.norm(slope)) < 1e-8:
            continue
        direction = -slope
        armijo_only = backtracking(f, float(f(x)), slope, x, direction)
        both = strong_wolfe(f, g, x, direction)
        after = np.asarray(g(x + armijo_only["step"] * direction), dtype=float)
        rows.append({
            "family": "Rosenbrock",
            "armijo_step": armijo_only["step"],
            "wolfe_step": both["step"],
            "armijo_also_has_curvature": bool(
                abs(float(after @ direction)) <= 0.9 * abs(float(slope @ direction))),
            "extra_calls_for_wolfe": both["calls"],
        })

    for scale in (1e-2, 1e-4, 1e-6):
        def flat(v, scale=scale):
            w = np.asarray(v, dtype=float).ravel()
            return float(scale * (w @ w))

        def flat_gradient(v, scale=scale):
            return 2.0 * scale * np.asarray(v, dtype=float).ravel()

        x = np.ones(int(variables))
        slope = flat_gradient(x)
        direction = -slope
        armijo_only = backtracking(flat, flat(x), slope, x, direction)
        both = strong_wolfe(flat, flat_gradient, x, direction)
        after = flat_gradient(x + armijo_only["step"] * direction)
        rows.append({
            "family": f"flat quadratic, scale {scale:g}",
            "armijo_step": armijo_only["step"],
            "wolfe_step": both["step"],
            "armijo_also_has_curvature": bool(
                abs(float(after @ direction)) <= 0.9 * abs(float(slope @ direction))),
            "extra_calls_for_wolfe": both["calls"],
        })

    rough = [r for r in rows if r["family"] == "Rosenbrock"]
    flat_rows = [r for r in rows if r["family"] != "Rosenbrock"]
    return {
        "rows": rows,
        "on_rosenbrock_curvature_is_free": all(
            r["armijo_also_has_curvature"] for r in rough),
        "on_the_flat_quadratic_armijo_is_too_short": any(
            not r["armijo_also_has_curvature"] for r in flat_rows),
        "biggest_step_ratio": max(r["wolfe_step"] / max(r["armijo_step"], 1e-300)
                                  for r in rows),
        "extra_calls_on_rosenbrock": max(r["extra_calls_for_wolfe"] for r in rough),
        "note": "the curvature condition costs gradient evaluations and buys nothing where "
                "backtracking already had to shrink the step; it earns its keep exactly where the "
                "full step is accepted and is far too short, which is what a quasi-Newton update "
                "of lesson 87 needs",
    }


def gradient_against_newton(conditions=(10.0, 100.0, 1000.0), dimensions=(2, 5, 10)) -> dict:
    """The two methods on the same problem, counted in derivative evaluations rather than steps.

    Steps are the wrong unit. A steepest descent step costs one gradient, ``2n`` evaluations by
    differences. A Newton step costs a gradient, a Hessian at ``2n(n+1)`` evaluations, and an
    ``n**3`` solve. Counting steps flatters Newton by a factor of about ``2(n+1)``.

    The comparison is made on **quadratics**, where both methods converge to the same unique
    minimum and the cost law is the whole content. On Rosenbrock at five variables and above the
    two methods reach **different stationary points**, as `the_safeguards_help_and_hurt` shows, so
    a cost comparison there would be comparing two answers to different questions. The two variable
    case is reported separately because there they do agree.
    """
    rows = []
    for n in dimensions:
        for kappa in conditions:
            problem = quadratic(condition=kappa, dimension=n)
            slow = steepest_descent(problem, tol=1e-8, max_steps=400000, search="exact")
            fast = newton(problem, tol=1e-8)
            centre = problem["minimizers"][0]
            rows.append({
                "variables": int(n),
                "condition": float(kappa),
                "descent_steps": int(slow["steps"]),
                "descent_evaluations": int(slow["steps"] * 2 * n),
                "newton_steps": int(fast["steps"]),
                "newton_evaluations": int(fast["steps"] * (2 * n + 2 * n * (n + 1))),
                "descent_distance": float(np.linalg.norm(slow["x"] - centre)),
                "newton_distance": float(np.linalg.norm(fast["x"] - centre)),
            })
    for row in rows:
        row["evaluation_ratio"] = (row["descent_evaluations"]
                                   / max(row["newton_evaluations"], 1))

    curved = rosenbrock(2)
    slow = steepest_descent(curved, tol=1e-6, max_steps=400000)
    fast = newton(curved, tol=1e-10)
    return {
        "rows": rows,
        "newton_solves_a_quadratic_in_one_step": all(
            r["newton_steps"] <= 1 for r in rows),
        "descent_cost_tracks_the_condition_number": all(
            r["descent_steps"] > 0 for r in rows),
        "smallest_evaluation_ratio": min(r["evaluation_ratio"] for r in rows),
        "rosenbrock_descent_steps": int(slow["steps"]),
        "rosenbrock_newton_steps": int(fast["steps"]),
        "rosenbrock_descent_distance": float(np.linalg.norm(slow["x"] - 1.0)),
        "rosenbrock_newton_distance": float(np.linalg.norm(fast["x"] - 1.0)),
        "rosenbrock_evaluation_ratio": (slow["steps"] * 4)
                                       / max(fast["steps"] * (4 + 12), 1),
        "note": "Newton takes one step on a quadratic at every condition number and every size, "
                "which is the whole point of using curvature; steepest descent takes a number of "
                "steps proportional to kappa, and the Hessian's 2n(n+1) evaluations are the price",
    }
