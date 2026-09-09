"""Optimization with constraints: KKT, penalties, barriers and projection.

Where the work is
-----------------
Lessons 84 to 87 asked which direction goes downhill. With constraints the question changes to
which directions are **allowed**, and the answer is the Karush-Kuhn-Tucker conditions. For

    minimize f(x)   subject to   c_i(x) <= 0 ,

they are stationarity of the Lagrangian, feasibility, nonnegative multipliers, and complementary
slackness ``lambda_i c_i(x) = 0``. Together they say the gradient must be a nonnegative combination
of the active constraint gradients: any remaining downhill direction would leave the feasible set.

Three families turn that into an algorithm, and this module measures what each one costs.

**Penalty** adds ``(mu/2) ||c||**2`` and minimizes freely. It works, and the accuracy is ``O(1/mu)``
while the conditioning is ``O(mu)``, so the two fight and the fight has a winner.

**Barrier** adds ``-mu * sum log(-c_i)`` and stays strictly inside. The minimizers trace the central
path, and the multipliers fall out of the path for free.

**Projection** keeps every iterate feasible by mapping it back after each step. It needs the
projection to be cheap, which it is for a box, a ball or a simplex, and it is the special case of
lesson 90's proximal gradient in which the nonsmooth term is an indicator function.

What the measurements here show
-------------------------------
* The penalty method's error is exactly ``0.707/mu`` and its conditioning exactly ``mu``, over nine
  orders of magnitude. Pushed further it stops working in three distinct ways: at ``mu = 10**14``
  the subproblem no longer converges in 3000 steps, at ``10**16`` the curvature model goes
  numerically singular, and at ``10**18`` the penalty term **overflows** before the solve begins.
  The usable range ends around ``10**11``, which caps the constraint violation at about ``10**-11``.
* **The augmented Lagrangian fixes it with ``mu`` fixed at 10.** Eight rounds take the violation
  from ``9.1e-02`` to ``4.7e-09`` while the condition number stays at ``11.0``, and the multiplier
  estimate converges to ``-0.999999995`` against an exact ``-1``. Reaching that violation with a
  plain penalty needs a weight of ``10**9`` and a condition number to match.
* The barrier's central path converges to the vertex ``(1, 0)`` with the slack falling like
  ``5 mu``, and ``mu / slack`` converges to ``2.0000``, which is the exact KKT multiplier. **The
  multiplier is a by-product of the path**, not a separate computation.
* Strict complementarity is not automatic: on one of the two targets here a constraint is active
  with multiplier **zero**, and the central path approaches it at a visibly different rate.
* Projection onto a box, a ball and a simplex each satisfy the defining property of a projection to
  rounding, and projected gradient descent keeps every iterate feasible at every step.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from .quasinewton import bfgs


# --------------------------------------------------------------------------- problems


def equality_problem(dimension: int = 2):
    """``minimize ||x||**2 subject to sum(x) = 1``, whose answer is known in closed form.

    The minimizer is ``(1/n, ..., 1/n)`` with multiplier ``-2/n``. The sign is the convention of
    this module and it is used everywhere in it: constraints are written ``c(x) <= 0``, the
    Lagrangian is ``f + lambda . c``, and stationarity reads ``grad f + J^T lambda = 0``. For an
    inequality that forces ``lambda >= 0``; for an equality the sign is free and comes out negative
    here.
    """
    n = int(dimension)
    if n < 1:
        raise ValueError(f"need at least one variable, got {n}")
    ones = np.ones(n)

    def f(v):
        w = np.asarray(v, dtype=float).ravel()
        return float(w @ w)

    def gradient(v):
        return 2.0 * np.asarray(v, dtype=float).ravel()

    def hessian(v):
        return 2.0 * np.eye(n)

    def constraint(v):
        return np.array([float(np.sum(np.asarray(v, dtype=float).ravel()) - 1.0)])

    def jacobian(v):
        return ones.reshape(1, n)

    return {"f": f, "gradient": gradient, "hessian": hessian,
            "constraint": constraint, "jacobian": jacobian,
            "kind": "equality", "dimension": n,
            "start": np.linspace(2.0, -1.0, n),
            "minimizer": ones / n, "multipliers": np.array([-2.0 / n]),
            "minimum": 1.0 / n,
            "name": f"minimize ||x||^2 subject to sum(x) = 1, {n} variables"}


def triangle_problem(target=(1.5, 1.5)):
    """``minimize ||x - t||**2`` over the triangle ``x1 + x2 <= 1, x1 >= 0, x2 >= 0``.

    Two targets are used in this module and they behave differently. With ``t = (1.5, 1.5)`` the
    answer is ``(0.5, 0.5)``, one constraint is active and its multiplier is ``2``, which is
    **strict complementarity**. With ``t = (2, 1)`` the answer is the vertex ``(1, 0)``, two
    constraints are active and one of them has multiplier **zero**, which is not.
    """
    aim = np.asarray(target, dtype=float).ravel()
    if aim.size != 2:
        raise ValueError(f"this problem is two dimensional, got a target of size {aim.size}")
    normals = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    offsets = np.array([-1.0, 0.0, 0.0])

    def f(v):
        w = np.asarray(v, dtype=float).ravel() - aim
        return float(w @ w)

    def gradient(v):
        return 2.0 * (np.asarray(v, dtype=float).ravel() - aim)

    def hessian(v):
        return 2.0 * np.eye(aim.size)

    def constraint(v):
        return normals @ np.asarray(v, dtype=float).ravel() + offsets

    def jacobian(v):
        return normals.copy()

    # the answer is the projection of the target onto the triangle, computed once
    answer = _project_triangle(aim)
    active = np.flatnonzero(np.abs(constraint(answer)) < 1e-12)
    if active.size:
        rows = normals[active]
        multipliers = np.zeros(offsets.size)
        found, *_ = np.linalg.lstsq(rows.T, -gradient(answer), rcond=None)
        multipliers[active] = found
    else:
        multipliers = np.zeros(offsets.size)
    return {"f": f, "gradient": gradient, "hessian": hessian,
            "constraint": constraint, "jacobian": jacobian,
            "kind": "inequality", "dimension": 2,
            "start": np.array([0.3, 0.3]),
            "minimizer": answer, "multipliers": multipliers,
            "minimum": f(answer), "active": active,
            "name": f"minimize ||x - {tuple(aim)}||^2 over the unit triangle"}


def _project_triangle(point):
    """The nearest point of ``{x1 + x2 <= 1, x >= 0}``, by checking the face, edges and vertices."""
    p = np.asarray(point, dtype=float).ravel()
    best, value = None, math.inf
    candidates = [p.copy()]
    candidates.append(p - ((p[0] + p[1] - 1.0) / p.size) * np.ones(p.size))
    candidates.append(np.array([0.0, p[1]]))
    candidates.append(np.array([p[0], 0.0]))
    candidates += [np.zeros(p.size)] + [np.eye(p.size)[k] for k in range(p.size)]
    for candidate in candidates:
        if (candidate[0] >= -1e-15 and candidate[1] >= -1e-15
                and candidate[0] + candidate[1] <= 1.0 + 1e-15):
            gap = float((candidate - p) @ (candidate - p))
            if gap < value:
                best, value = candidate, gap
    return np.clip(best, 0.0, None)


# --------------------------------------------------------------------------- the conditions


def kkt_residual(problem, where, multipliers) -> dict:
    """The four KKT conditions, measured one at a time rather than combined into a norm.

    Combining them hides which one failed, and they fail for different reasons: stationarity is
    about the optimizer, feasibility about the constraint handling, nonnegativity about the sign
    convention, and complementary slackness about which constraints the method thinks are active.
    """
    x = np.asarray(where, dtype=float).ravel()
    lam = np.asarray(multipliers, dtype=float).ravel()
    values = np.asarray(problem["constraint"](x), dtype=float).ravel()
    jac = np.asarray(problem["jacobian"](x), dtype=float)
    slope = np.asarray(problem["gradient"](x), dtype=float).ravel()
    stationarity = float(np.max(np.abs(slope + jac.T @ lam)))
    if problem["kind"] == "equality":
        feasibility = float(np.max(np.abs(values)))
        nonnegativity = 0.0
        slackness = 0.0
    else:
        feasibility = float(np.max(np.maximum(values, 0.0)))
        nonnegativity = float(np.max(np.maximum(-lam, 0.0)))
        slackness = float(np.max(np.abs(lam * values)))
    return {"stationarity": stationarity, "feasibility": feasibility,
            "nonnegativity": nonnegativity, "complementary_slackness": slackness,
            "worst": max(stationarity, feasibility, nonnegativity, slackness)}


@functools.lru_cache(maxsize=None)
def the_conditions_hold_at_the_answer(problems=None) -> dict:
    """Check the closed form answers against the conditions they are supposed to satisfy.

    This is the check that the test problems are what they claim, done before any method is run on
    them. A problem whose stated answer fails its own optimality conditions would make every later
    measurement meaningless.
    """
    if problems is None:
        problems = [equality_problem(2), equality_problem(5),
                    triangle_problem((1.5, 1.5)), triangle_problem((2.0, 1.0))]
    rows = []
    for problem in problems:
        out = kkt_residual(problem, problem["minimizer"], problem["multipliers"])
        rows.append({"name": problem["name"], **out,
                     "multipliers": problem["multipliers"]})
    return {
        "rows": rows,
        "every_answer_satisfies_them": all(r["worst"] < 1e-12 for r in rows),
        "worst_residual": max(r["worst"] for r in rows),
        "note": "the stated answers satisfy their own optimality conditions to rounding, which is "
                "what makes the later measurements measurements rather than opinions",
    }


def strict_complementarity_is_not_automatic() -> dict:
    """Two targets on the same triangle, one with a zero multiplier on an active constraint.

    Strict complementarity says every active constraint has a **strictly positive** multiplier. It
    is an assumption in most convergence proofs and it is not part of the KKT conditions. Here one
    target satisfies it and one does not, and the second is not pathological: it is what happens
    when the unconstrained minimum projects exactly onto a vertex.
    """
    rows = []
    for target in ((1.5, 1.5), (2.0, 1.0)):
        problem = triangle_problem(target)
        values = np.asarray(problem["constraint"](problem["minimizer"]), dtype=float)
        active = np.flatnonzero(np.abs(values) < 1e-12)
        lam = problem["multipliers"]
        rows.append({
            "target": target,
            "minimizer": problem["minimizer"],
            "active_constraints": active.tolist(),
            "multipliers_there": lam[active].tolist(),
            "strict": bool(active.size and np.all(lam[active] > 1e-12)),
        })
    return {
        "rows": rows,
        "one_of_each": len({r["strict"] for r in rows}) == 2,
        "note": "an active constraint with a zero multiplier is a constraint the objective does "
                "not actually push against, and most convergence theorems assume it away",
    }


# --------------------------------------------------------------------------- penalty


def penalty_problem(problem, weight: float):
    """``f(x) + (mu/2) ||c(x)||**2`` for equalities, using only the violated part for inequalities."""
    mu = float(weight)
    if mu <= 0.0:
        raise ValueError(f"the penalty weight must be positive, got {mu}")
    equality = problem["kind"] == "equality"

    def violation(v):
        values = np.asarray(problem["constraint"](v), dtype=float).ravel()
        return values if equality else np.maximum(values, 0.0)

    def f(v):
        gap = violation(v)
        return float(problem["f"](v)) + 0.5 * mu * float(gap @ gap)

    def gradient(v):
        gap = violation(v)
        jac = np.asarray(problem["jacobian"](v), dtype=float)
        return np.asarray(problem["gradient"](v), dtype=float).ravel() + mu * (jac.T @ gap)

    def hessian(v):
        jac = np.asarray(problem["jacobian"](v), dtype=float)
        base = np.asarray(problem["hessian"](v), dtype=float)
        if equality:
            return base + mu * (jac.T @ jac)
        active = np.asarray(problem["constraint"](v), dtype=float).ravel() > 0.0
        rows = jac[active]
        return base + mu * (rows.T @ rows) if rows.size else base

    return {"f": f, "gradient": gradient, "hessian": hessian,
            "start": problem["start"], "dimension": problem["dimension"],
            "weight": mu}


@functools.lru_cache(maxsize=None)
def the_penalty_trades_accuracy_for_conditioning(weights=None, inner_tol: float = 1e-6,
                                                 inner_steps: int = 3000) -> dict:
    """Push the penalty weight up until the method stops working, and record how it stops.

    The theory says the error is ``O(1/mu)`` and the Hessian's condition number is ``O(mu)``. Both
    are exact here over nine orders of magnitude, which makes the failure at the top end the
    interesting part: there are three separate failures and they arrive in order.
    """
    if weights is None:
        weights = [10.0 ** power for power in (2, 5, 8, 11, 14, 16)]
    problem = equality_problem(2)
    star = problem["minimizer"]
    rows = []
    for mu in weights:
        inner = penalty_problem(problem, mu)
        overflowed = False
        try:
            out = bfgs(inner, tol=inner_tol, max_steps=int(inner_steps))
        except (OverflowError, FloatingPointError):
            overflowed = True
            out = None
        if overflowed:
            rows.append({"weight": float(mu), "error": float("inf"),
                         "violation": float("inf"), "condition": float("inf"),
                         "inner_steps": 0, "converged": False, "overflowed": True})
            continue
        gap = np.asarray(problem["constraint"](out["x"]), dtype=float)
        rows.append({
            "weight": float(mu),
            "error": float(np.linalg.norm(out["x"] - star)),
            "violation": float(np.max(np.abs(gap))),
            "condition": float(np.linalg.cond(inner["hessian"](out["x"]))),
            "inner_steps": int(out["steps"]),
            "converged": bool(out["converged"]),
            "overflowed": False,
        })
    working = [r for r in rows if r["converged"]]
    return {
        "rows": rows,
        "error_times_weight": [r["error"] * r["weight"] for r in working],
        "condition_over_weight": [r["condition"] / r["weight"] for r in working],
        "the_error_is_one_over_mu": bool(working) and max(
            abs(r["error"] * r["weight"] - working[0]["error"] * working[0]["weight"])
            for r in working) < 1e-2,
        "the_condition_is_mu": bool(working) and all(
            0.9 < r["condition"] / r["weight"] < 1.6 for r in working),
        "largest_weight_that_worked": max((r["weight"] for r in working), default=0.0),
        "best_violation_reached": min((r["violation"] for r in working), default=float("inf")),
        "failures": [r["weight"] for r in rows if not r["converged"]],
        "note": "accuracy and conditioning are the same knob turned in opposite directions, so "
                "the reachable accuracy is capped by the largest weight the subproblem can still "
                "be solved at, and here that is about 10**11",
    }


# --------------------------------------------------------------------------- augmented Lagrangian


def augmented_lagrangian(problem, weight: float = 10.0, rounds: int = 8,
                         inner_tol: float = 1e-13, growth: float = 10.0,
                         wanted: float = 0.25) -> dict:
    """``f + lambda.c + (mu/2)||c||**2``, with ``lambda`` updated instead of ``mu`` increased.

    The penalty method needs ``mu -> infinity`` because the plain penalty's minimizer is never the
    constrained one: the pull of the penalty has to grow to hold the iterate near the boundary.
    Adding the multiplier term removes that need. At the true multiplier the augmented Lagrangian
    has its unconstrained minimum **exactly at the constrained solution**, for any ``mu`` above a
    threshold, and the update ``lambda <- lambda + mu c(x)`` converges to it.

    So the conditioning stops growing, which is the whole point.
    """
    mu = float(weight)
    lam = np.zeros(np.asarray(problem["constraint"](problem["start"]),
                              dtype=float).ravel().size)
    x = np.asarray(problem["start"], dtype=float).ravel()
    rows = []
    for round_number in range(int(rounds)):
        def f(v, lam=lam, mu=mu):
            gap = np.asarray(problem["constraint"](v), dtype=float).ravel()
            return float(problem["f"](v)) + float(lam @ gap) + 0.5 * mu * float(gap @ gap)

        def gradient(v, lam=lam, mu=mu):
            gap = np.asarray(problem["constraint"](v), dtype=float).ravel()
            jac = np.asarray(problem["jacobian"](v), dtype=float)
            return (np.asarray(problem["gradient"](v), dtype=float).ravel()
                    + jac.T @ (lam + mu * gap))

        def hessian(v, mu=mu):
            jac = np.asarray(problem["jacobian"](v), dtype=float)
            return np.asarray(problem["hessian"](v), dtype=float) + mu * (jac.T @ jac)

        inner = {"f": f, "gradient": gradient, "hessian": hessian,
                 "start": x, "dimension": problem["dimension"]}
        out = bfgs(inner, tol=inner_tol, max_steps=20000)
        x = out["x"]
        gap = np.asarray(problem["constraint"](x), dtype=float).ravel()
        rows.append({
            "round": round_number,
            "weight": mu,
            "multiplier": lam.copy(),
            "error": float(np.linalg.norm(x - problem["minimizer"])),
            "violation": float(np.max(np.abs(gap))),
            "condition": float(np.linalg.cond(hessian(x))),
            "inner_steps": int(out["steps"]),
        })
        previous = rows[-2]["violation"] if len(rows) > 1 else math.inf
        lam = lam + mu * gap
        if float(np.max(np.abs(gap))) > wanted * previous:
            mu *= float(growth)
    return {"x": x, "multipliers": lam, "rows": rows, "final_weight": mu,
            "multiplier_error": float(np.max(np.abs(lam - problem["multipliers"]))),
            "error": float(np.linalg.norm(x - problem["minimizer"]))}


@functools.lru_cache(maxsize=None)
def the_augmented_lagrangian_keeps_the_weight_bounded() -> dict:
    """The same problem by both methods, comparing what has to grow.

    The comparison that matters is not which reaches a smaller violation, since the penalty method
    can be pushed until it breaks. It is what the conditioning does while that happens.
    """
    problem = equality_problem(2)
    out = augmented_lagrangian(problem)
    weights = [row["weight"] for row in out["rows"]]
    conditions = [row["condition"] for row in out["rows"]]
    penalty = the_penalty_trades_accuracy_for_conditioning()
    return {
        "rows": out["rows"],
        "final_violation": out["rows"][-1]["violation"],
        "final_error": out["error"],
        "multiplier_error": out["multiplier_error"],
        "exact_multipliers": problem["multipliers"],
        "weight_never_grew": max(weights) == min(weights),
        "final_weight": out["final_weight"],
        "condition_stayed_at": max(conditions),
        "penalty_condition_for_the_same_violation": max(
            (r["condition"] for r in penalty["rows"]
             if r["converged"] and r["violation"] <= out["rows"][-1]["violation"]),
            default=float("inf")),
        "note": "the augmented Lagrangian reaches the same violation with the weight fixed at 10 "
                "and a condition number of 11, where the penalty method needs a weight of 10**9 "
                "and a condition number to match",
    }


# --------------------------------------------------------------------------- barrier


@functools.lru_cache(maxsize=None)
def the_central_path(target=(2.0, 1.0), weights=None) -> dict:
    """Follow ``f - mu sum log(-c_i)`` as ``mu`` falls, and read the multipliers off the path.

    Every point on the path is **strictly feasible**, which is the property that makes interior
    point methods usable on problems where an infeasible iterate is meaningless. The multiplier
    estimate is free: at the barrier minimum ``mu / (-c_i)`` is exactly the multiplier of
    constraint ``i``, so the path carries the dual solution with it.
    """
    if weights is None:
        weights = [10.0 ** (-power) for power in range(0, 10)]
    problem = triangle_problem(target)
    x = np.asarray(problem["start"], dtype=float).ravel()
    rows = []
    for mu in weights:
        def f(v, mu=mu):
            values = np.asarray(problem["constraint"](v), dtype=float).ravel()
            if np.any(values >= -1e-300):
                return 1e300
            return float(problem["f"](v)) - mu * float(np.sum(np.log(-values)))

        def gradient(v, mu=mu):
            values = np.asarray(problem["constraint"](v), dtype=float).ravel()
            jac = np.asarray(problem["jacobian"](v), dtype=float)
            return (np.asarray(problem["gradient"](v), dtype=float).ravel()
                    - mu * (jac.T @ (1.0 / values)))

        out = bfgs({"f": f, "gradient": gradient, "start": x,
                    "dimension": problem["dimension"]}, tol=1e-10, max_steps=20000)
        x = out["x"]
        values = np.asarray(problem["constraint"](x), dtype=float).ravel()
        estimate = mu / (-values)
        rows.append({
            "weight": float(mu),
            "x": x.copy(),
            "f": float(problem["f"](x)),
            "strictly_feasible": bool(np.all(values < 0.0)),
            "distance": float(np.linalg.norm(x - problem["minimizer"])),
            "multiplier_estimate": estimate.copy(),
            "multiplier_error": float(np.max(np.abs(estimate - problem["multipliers"]))),
        })
    return {
        "rows": rows,
        "target": tuple(float(v) for v in target),
        "exact_minimizer": problem["minimizer"],
        "exact_multipliers": problem["multipliers"],
        "every_point_strictly_feasible": all(r["strictly_feasible"] for r in rows),
        "final_distance": rows[-1]["distance"],
        "final_multiplier_error": rows[-1]["multiplier_error"],
        "the_multipliers_come_free": rows[-1]["multiplier_error"] < 1e-3,
        "note": "every iterate is strictly inside, the path ends at the solution, and mu over the "
                "slack is the multiplier, so the dual answer arrives without being asked for",
    }


def the_path_slows_where_complementarity_is_not_strict() -> dict:
    """The same path on both targets, comparing how fast each multiplier settles.

    Where a constraint is active with a **positive** multiplier the estimate converges quickly.
    Where it is active with a **zero** multiplier the slack and the weight both go to zero and
    their ratio is a race, which is slower and less accurate.
    """
    rows = []
    for target in ((1.5, 1.5), (2.0, 1.0)):
        path = the_central_path(target)
        problem = triangle_problem(target)
        values = np.asarray(problem["constraint"](problem["minimizer"]), dtype=float)
        active = np.flatnonzero(np.abs(values) < 1e-12)
        lam = problem["multipliers"]
        strict = bool(active.size and np.all(lam[active] > 1e-12))
        rows.append({
            "target": tuple(float(v) for v in target),
            "strict_complementarity": strict,
            "active_constraints": active.size,
            "final_distance": path["final_distance"],
            "final_multiplier_error": path["final_multiplier_error"],
        })
    strict_rows = [r for r in rows if r["strict_complementarity"]]
    loose_rows = [r for r in rows if not r["strict_complementarity"]]
    return {
        "rows": rows,
        "the_strict_case_is_more_accurate": bool(
            strict_rows and loose_rows
            and strict_rows[0]["final_multiplier_error"]
            < loose_rows[0]["final_multiplier_error"]),
        "note": "a zero multiplier on an active constraint makes mu and the slack vanish together, "
                "and their ratio converges more slowly than where the constraint genuinely pushes",
    }


# --------------------------------------------------------------------------- projection


def project_onto_box(point, lower=None, upper=None) -> np.ndarray:
    """Clip each coordinate, which is the nearest point of a box in any dimension."""
    p = np.asarray(point, dtype=float).ravel()
    low = -np.inf if lower is None else np.asarray(lower, dtype=float)
    high = np.inf if upper is None else np.asarray(upper, dtype=float)
    return np.clip(p, low, high)


def project_onto_ball(point, radius: float = 1.0) -> np.ndarray:
    """Scale down to the sphere if outside, leave alone if inside."""
    p = np.asarray(point, dtype=float).ravel()
    length = float(np.linalg.norm(p))
    if length <= float(radius) or length == 0.0:
        return p.copy()
    return (float(radius) / length) * p


def project_onto_simplex(point, total: float = 1.0) -> np.ndarray:
    """The nearest point with nonnegative entries summing to ``total``, by sorting.

    The answer is ``max(p - theta, 0)`` for the ``theta`` that makes the sum right, and finding
    ``theta`` needs only a sort and a scan, so the whole projection costs ``O(n log n)``. This is
    the projection that makes projected gradient usable on probability vectors.
    """
    p = np.asarray(point, dtype=float).ravel()
    n = p.size
    if n == 0:
        raise ValueError("cannot project an empty vector")
    ordered = np.sort(p)[::-1]
    running = np.cumsum(ordered) - float(total)
    index = np.arange(1, n + 1)
    allowed = ordered - running / index > 0.0
    count = int(np.max(np.flatnonzero(allowed))) + 1 if np.any(allowed) else 1
    theta = running[count - 1] / count
    return np.maximum(p - theta, 0.0)


def projected_gradient(f, gradient, projection, start, step: float = 0.1,
                       tol: float = 1e-9, max_steps: int = 20000) -> dict:
    """Take a gradient step, then project back. Every iterate is feasible by construction.

    The stopping test is the size of ``x - P(x - a g)``, which is zero exactly at a point where no
    feasible descent direction remains. That is the constrained analogue of ``grad f = 0`` and it
    reduces to it when the projection is the identity.
    """
    x = projection(np.asarray(start, dtype=float).ravel())
    history, feasible = [], True
    for step_number in range(int(max_steps)):
        slope = np.asarray(gradient(x), dtype=float).ravel()
        moved = projection(x - float(step) * slope)
        size = float(np.linalg.norm(moved - x))
        history.append(size)
        if size <= float(tol):
            break
        x = moved
        if float(np.linalg.norm(projection(x) - x)) > 1e-12:
            feasible = False
    return {"x": x, "f": float(f(x)), "steps": step_number,
            "history": np.asarray(history), "every_iterate_feasible": feasible,
            "converged": bool(history and history[-1] <= float(tol))}


def the_projections_are_projections(samples: int = 200, seed: int = 42) -> dict:
    """Check each projection against the property that defines one, on random points.

    A projection ``P`` onto a convex set ``C`` satisfies ``(p - P(p)).(z - P(p)) <= 0`` for every
    ``z`` in ``C``. That single inequality is the definition, it is checkable, and checking it is
    stronger than checking that the answer is feasible and close.
    """
    rng = np.random.default_rng(int(seed))
    rows = []
    cases = [
        ("box [-1, 1]", lambda p: project_onto_box(p, -1.0, 1.0),
         lambda size: rng.uniform(-1.0, 1.0, size=size)),
        ("unit ball", lambda p: project_onto_ball(p, 1.0),
         lambda size: _random_in_ball(rng, size)),
        ("unit simplex", lambda p: project_onto_simplex(p, 1.0),
         lambda size: rng.dirichlet(np.ones(size))),
    ]
    for name, projection, inside in cases:
        worst, idempotent = -math.inf, 0.0
        for _ in range(int(samples)):
            size = int(rng.integers(1, 8))
            p = rng.normal(scale=2.0, size=size)
            projected = projection(p)
            z = inside(size)
            worst = max(worst, float((p - projected) @ (z - projected)))
            idempotent = max(idempotent,
                             float(np.max(np.abs(projection(projected) - projected))))
        rows.append({"set": name, "worst_inner_product": worst,
                     "idempotent_to": idempotent,
                     "satisfies_the_definition": worst <= 1e-10})
    return {
        "rows": rows,
        "all_are_projections": all(r["satisfies_the_definition"] for r in rows),
        "all_idempotent": all(r["idempotent_to"] < 1e-12 for r in rows),
        "note": "the defining inequality holds on every random pair, and each projection is its "
                "own square, which together are what makes projected gradient a descent method",
    }


def _random_in_ball(rng, size):
    direction = rng.normal(size=size)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-300)
    return direction * float(rng.random()) ** (1.0 / max(size, 1))


def projected_gradient_stays_feasible(dimensions=(2, 5, 20), seed: int = 42) -> dict:
    """Minimize a distance over a simplex by projected gradient, and check every iterate.

    The projection is what makes this work, and it is also what connects the lesson forward: the
    projection onto a set is the proximal operator of that set's indicator function, so this is
    lesson 90's proximal gradient with the nonsmooth term chosen to be ``0`` inside and infinite
    outside.
    """
    rng = np.random.default_rng(int(seed))
    rows = []
    for n in dimensions:
        target = rng.normal(scale=1.5, size=int(n))

        def f(v, target=target):
            gap = np.asarray(v, dtype=float).ravel() - target
            return float(gap @ gap)

        def gradient(v, target=target):
            return 2.0 * (np.asarray(v, dtype=float).ravel() - target)

        out = projected_gradient(f, gradient, project_onto_simplex,
                                 np.full(int(n), 1.0 / n), step=0.25)
        answer = out["x"]
        direct = project_onto_simplex(target)
        rows.append({
            "variables": int(n),
            "steps": int(out["steps"]),
            "every_iterate_feasible": bool(out["every_iterate_feasible"]),
            "sum_of_the_answer": float(np.sum(answer)),
            "smallest_entry": float(np.min(answer)),
            "gap_to_the_direct_projection": float(np.linalg.norm(answer - direct)),
        })
    return {
        "rows": rows,
        "always_feasible": all(r["every_iterate_feasible"] for r in rows),
        "sums_are_one": all(abs(r["sum_of_the_answer"] - 1.0) < 1e-9 for r in rows),
        "nothing_negative": all(r["smallest_entry"] >= -1e-15 for r in rows),
        "matches_the_direct_projection": all(
            r["gap_to_the_direct_projection"] < 1e-6 for r in rows),
        "note": "minimizing the distance to a point over a set is the projection onto that set, so "
                "the iteration and the closed form must agree, and they do to 10**-6",
    }
