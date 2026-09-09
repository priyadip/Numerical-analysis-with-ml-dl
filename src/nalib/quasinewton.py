"""Newton's rate without Newton's Hessian: BFGS, limited memory, and trust regions.

Where the work is
-----------------
Lesson 86 left two problems. Newton converges quadratically and costs ``2n(n+1)`` evaluations for a
Hessian and ``n**3`` to factor it. And its step is only a descent direction where the Hessian is
positive definite, which on Rosenbrock is under a third of the domain at ten variables.

Quasi-Newton methods fix the first. They build curvature from the gradients they were already
computing, through the **secant condition**

    B_(k+1) s_k = y_k ,   s_k = x_(k+1) - x_k ,   y_k = g_(k+1) - g_k ,

which in one dimension is the secant method of lesson 11. In ``n`` dimensions it is ``n`` equations
for ``n**2`` unknowns, so the update is chosen to be symmetric, positive definite and as close to
the previous one as possible. That is BFGS, and it costs no extra evaluations at all.

Trust regions fix the second. Instead of choosing a direction and then a length, they bound the
length first and minimize the quadratic model inside that ball. An indefinite model is then not a
problem to be shifted away: the constrained minimum exists regardless.

What the measurements here show
-------------------------------
* The BFGS update satisfies the secant condition to ``10**-10`` at every step of a ten variable
  run, which is the check that the algebra is the algebra.
* ``y.s > 0`` at every step, and that is exactly lesson 86's curvature condition. Without it the
  update is not positive definite, so **the Wolfe line search is not a refinement of BFGS, it is a
  precondition for it**.
* **BFGS is not quadratic, and that is the resolvable half.** At two variables ``e_(k+1)/e_k**2``
  rises through ``5.6e-01, 2.7e+02, 1.5e+03, 1.3e+05, 1.8e+07`` where Newton's stays at ``112``.
  The per step ratio sits near ``0.01`` over the whole visible window, and whether it actually
  tends to zero cannot be settled in double precision: the run crosses from ``10**-2`` to the floor
  in six steps. This is lesson 85's problem for the third time in this part.
* On a quadratic BFGS terminates in ``n`` steps only with an exact line search. With the usual
  ``c2 = 0.9`` it takes ``3, 10, 19, 30`` steps at ``n = 2, 5, 10, 20``; tightening to
  ``c2 = 0.01`` gives ``3, 7, 12, 23``, within three of ``n``.
* **The line search decides the rate, not only the update.** On ten variable Rosenbrock,
  ``c2 = 0.9`` takes 103 steps and ends with ``||H B - I|| = 1.58``; ``c2 = 0.01`` takes 60 and ends
  at ``0.42``, with per step ratios of ``0.08`` instead of ``0.5``. The same update, three times
  the accuracy in the curvature model, and the run at ``c2 = 0.1`` lands on a different minimum
  entirely.
* Limited memory BFGS reaches the same answer storing ``m`` pairs instead of an ``n by n`` matrix,
  and the iteration count is flat in ``m`` well below ``n``.
* **A trust region escapes the saddle that lesson 86's plain Newton fell into**, without any
  Hessian shift, because a bounded model has a minimum whether or not it is convex.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .gradient import backtracking, strong_wolfe
from .optimize import quadratic, rosenbrock


# --------------------------------------------------------------------------- BFGS


def bfgs(problem, start=None, tol: float = 1e-10, max_steps: int = 5000,
         wolfe_c2: float = 0.9) -> dict:
    """BFGS in the inverse form, so each step is a matrix-vector product and no solve.

    The update keeps the approximate inverse Hessian ``H``,

        H_(k+1) = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T ,   rho = 1 / (y.s) ,

    which is the unique symmetric update satisfying the secant condition and minimizing a weighted
    distance to ``H_k``. It costs ``O(n**2)`` per step and **no extra function or gradient
    evaluations**, because ``s`` and ``y`` are made of quantities the line search already computed.

    ``y.s > 0`` is required and is not automatic. It is exactly the curvature half of the Wolfe
    conditions from lesson 86, which is why the line search here is a strong Wolfe search and not
    plain backtracking. When it fails the update is skipped rather than applied, and the count of
    skips is reported instead of hidden.
    """
    x = np.asarray(problem["start"] if start is None else start, dtype=float).ravel()
    f, g = problem["f"], problem["gradient"]
    n = x.size
    inverse = np.eye(n)
    identity = np.eye(n)
    history, curvatures, secant, skipped, calls = [], [], [], 0, 0
    for step_number in range(int(max_steps)):
        slope = np.asarray(g(x), dtype=float).ravel()
        history.append(float(np.linalg.norm(slope)))
        if history[-1] <= float(tol):
            break
        direction = -inverse @ slope
        if float(slope @ direction) >= 0.0:
            inverse = np.eye(n)
            direction = -slope
        search = strong_wolfe(f, g, x, direction, wolfe_c2=wolfe_c2)
        calls += search["calls"]
        move = search["step"] * direction
        ahead = x + move
        change = np.asarray(g(ahead), dtype=float).ravel() - slope
        curvature = float(change @ move)
        curvatures.append(curvature)
        if curvature > 1e-14:
            rho = 1.0 / curvature
            left = identity - rho * np.outer(move, change)
            inverse = left @ inverse @ left.T + rho * np.outer(move, move)
            # the secant residual is a diagnostic, so it must not be able to stop the solve:
            # on a badly scaled problem the approximation can become numerically singular, and
            # that is a fact to report rather than an exception to raise
            try:
                secant.append(float(np.max(np.abs(
                    np.linalg.solve(inverse, move) - change))))
            except np.linalg.LinAlgError:
                secant.append(float("inf"))
        else:
            skipped += 1
        x = ahead
    return {"x": x, "f": float(f(x)), "steps": step_number,
            "gradient_history": np.asarray(history),
            "curvatures": np.asarray(curvatures),
            "secant_residuals": np.asarray(secant),
            "skipped_updates": skipped, "line_search_calls": calls,
            "inverse_hessian": inverse,
            "converged": bool(history and history[-1] <= float(tol))}


def lbfgs(problem, start=None, memory: int = 5, tol: float = 1e-10,
          max_steps: int = 5000, wolfe_c2: float = 0.9) -> dict:
    """Limited memory BFGS: keep the last ``memory`` pairs and never form a matrix.

    The two loop recursion applies the same operator ``H_k`` that BFGS would have built, using only
    the stored ``(s, y)`` pairs, in ``4 m n`` operations and ``2 m n`` storage. For ``n`` in the
    millions that is the difference between possible and impossible, and the price is that the
    curvature information older than ``m`` steps is thrown away.
    """
    x = np.asarray(problem["start"] if start is None else start, dtype=float).ravel()
    f, g = problem["f"], problem["gradient"]
    keep = int(memory)
    if keep < 1:
        raise ValueError(f"need at least one stored pair, got {keep}")
    pairs = []
    history, calls, skipped = [], 0, 0
    for step_number in range(int(max_steps)):
        slope = np.asarray(g(x), dtype=float).ravel()
        history.append(float(np.linalg.norm(slope)))
        if history[-1] <= float(tol):
            break
        # two loop recursion
        q = slope.copy()
        alphas = []
        for move, change, rho in reversed(pairs):
            alpha = rho * float(move @ q)
            alphas.append(alpha)
            q = q - alpha * change
        if pairs:
            move, change, _ = pairs[-1]
            q = q * (float(move @ change) / float(change @ change))
        for (move, change, rho), alpha in zip(pairs, reversed(alphas)):
            beta = rho * float(change @ q)
            q = q + (alpha - beta) * move
        direction = -q
        if float(slope @ direction) >= 0.0:
            direction = -slope
        search = strong_wolfe(f, g, x, direction, wolfe_c2=wolfe_c2)
        calls += search["calls"]
        move = search["step"] * direction
        ahead = x + move
        change = np.asarray(g(ahead), dtype=float).ravel() - slope
        curvature = float(change @ move)
        if curvature > 1e-14:
            pairs.append((move, change, 1.0 / curvature))
            if len(pairs) > keep:
                pairs.pop(0)
        else:
            skipped += 1
        x = ahead
    return {"x": x, "f": float(f(x)), "steps": step_number,
            "gradient_history": np.asarray(history), "memory": keep,
            "stored_pairs": len(pairs), "skipped_updates": skipped,
            "line_search_calls": calls,
            "converged": bool(history and history[-1] <= float(tol))}


def nonlinear_cg(problem, start=None, rule: str = "polak-ribiere-plus",
                 tol: float = 1e-10, max_steps: int = 20000,
                 wolfe_c2: float = 0.1, restart_every: int | None = None) -> dict:
    """Conjugate gradients for a general function, with a choice of ``beta``.

    On a quadratic every rule reduces to lesson 24's method. Away from one they differ:

    - ``fletcher-reeves``: ``beta = g_(k+1).g_(k+1) / g_k.g_k``. Provably convergent with a strong
      Wolfe search and ``c2 < 1/2``, and it stalls: once a bad direction appears, ``beta`` stays
      near 1 and the method repeats it.
    - ``polak-ribiere``: ``beta = g_(k+1).(g_(k+1) - g_k) / g_k.g_k``. Self restarting, because a
      short step makes ``beta`` near zero, which resets to steepest descent. Not convergent in
      general.
    - ``polak-ribiere-plus``: ``max(beta, 0)``, which is convergent and keeps the self restart.

    ``c2 = 0.1`` rather than ``0.9`` here, because the conjugacy needs a more accurate step.
    """
    x = np.asarray(problem["start"] if start is None else start, dtype=float).ravel()
    f, g = problem["f"], problem["gradient"]
    slope = np.asarray(g(x), dtype=float).ravel()
    direction = -slope
    history, restarts, calls = [], 0, 0
    for step_number in range(int(max_steps)):
        history.append(float(np.linalg.norm(slope)))
        if history[-1] <= float(tol):
            break
        if float(slope @ direction) >= 0.0:
            direction = -slope
            restarts += 1
        search = strong_wolfe(f, g, x, direction, wolfe_c2=wolfe_c2)
        calls += search["calls"]
        x = x + search["step"] * direction
        ahead = np.asarray(g(x), dtype=float).ravel()
        bottom = float(slope @ slope)
        if bottom <= 0.0:
            break
        if rule == "fletcher-reeves":
            beta = float(ahead @ ahead) / bottom
        elif rule == "polak-ribiere":
            beta = float(ahead @ (ahead - slope)) / bottom
        elif rule == "polak-ribiere-plus":
            beta = max(0.0, float(ahead @ (ahead - slope)) / bottom)
        else:
            raise ValueError(f"unknown rule {rule!r}")
        if restart_every and (step_number + 1) % int(restart_every) == 0:
            beta = 0.0
            restarts += 1
        direction = -ahead + beta * direction
        slope = ahead
    return {"x": x, "f": float(f(x)), "steps": step_number, "rule": rule,
            "gradient_history": np.asarray(history), "restarts": restarts,
            "line_search_calls": calls,
            "converged": bool(history and history[-1] <= float(tol))}


# --------------------------------------------------------------------------- trust region


def dogleg(slope, matrix, radius: float) -> dict:
    """The dogleg step: the Newton step if it fits, otherwise a path toward the Cauchy point.

    The Cauchy point is the minimizer of the model along ``-g`` inside the ball, and it always
    exists. The Newton point is the model's unconstrained minimum, which exists only when the model
    is convex. The dogleg follows the two segments from the origin to the Cauchy point and on to
    the Newton point, and takes the point where that path leaves the ball.

    When the model is **not** convex there is no Newton point, and this falls back to the Cauchy
    step, which is exactly the behaviour that lets a trust region handle an indefinite Hessian with
    no shift at all.
    """
    g = np.asarray(slope, dtype=float).ravel()
    b = np.asarray(matrix, dtype=float)
    curvature = float(g @ b @ g)
    if curvature <= 0.0:
        # the model falls forever along -g, so go to the edge of the ball
        return {"step": -radius * g / float(np.linalg.norm(g)), "kind": "edge"}
    cauchy = -(float(g @ g) / curvature) * g
    try:
        newton = np.linalg.solve(b, -g)
        convex = bool(np.all(np.linalg.eigvalsh(0.5 * (b + b.T)) > 0.0))
    except np.linalg.LinAlgError:
        newton, convex = None, False
    if convex and float(np.linalg.norm(newton)) <= radius:
        return {"step": newton, "kind": "newton"}
    if float(np.linalg.norm(cauchy)) >= radius or not convex:
        return {"step": radius * cauchy / float(np.linalg.norm(cauchy)), "kind": "cauchy"}
    # the leg from the Cauchy point to the Newton point, cut at the boundary
    gap = newton - cauchy
    a = float(gap @ gap)
    bb = 2.0 * float(cauchy @ gap)
    cc = float(cauchy @ cauchy) - radius ** 2
    t = (-bb + math.sqrt(max(bb * bb - 4.0 * a * cc, 0.0))) / (2.0 * a)
    return {"step": cauchy + t * gap, "kind": "dogleg"}


def trust_region(problem, start=None, radius: float = 1.0, max_radius: float = 100.0,
                 accept: float = 0.1, tol: float = 1e-10, max_steps: int = 5000) -> dict:
    """Minimize the quadratic model inside a ball, and adjust the ball by how well it predicted.

    The ratio ``rho = (actual decrease) / (predicted decrease)`` is the whole control law. Near 1
    the model is good and the radius grows; near 0 it is bad and the radius shrinks and the step is
    rejected. Nothing here needs the Hessian to be positive definite, which is the difference from
    lesson 86's line search approach and the reason this section exists.
    """
    x = np.asarray(problem["start"] if start is None else start, dtype=float).ravel()
    f, g, h = problem["f"], problem["gradient"], problem["hessian"]
    delta = float(radius)
    history, kinds, rejected, radii = [], [], 0, []
    for step_number in range(int(max_steps)):
        slope = np.asarray(g(x), dtype=float).ravel()
        history.append(float(np.linalg.norm(slope)))
        radii.append(delta)
        if history[-1] <= float(tol):
            break
        matrix = np.asarray(h(x), dtype=float)
        matrix = 0.5 * (matrix + matrix.T)
        out = dogleg(slope, matrix, delta)
        move = out["step"]
        kinds.append(out["kind"])
        predicted = -(float(slope @ move) + 0.5 * float(move @ matrix @ move))
        actual = float(f(x)) - float(f(x + move))
        ratio = actual / predicted if predicted > 0.0 else -1.0
        if ratio < 0.25:
            delta *= 0.25
        elif ratio > 0.75 and abs(float(np.linalg.norm(move)) - delta) < 1e-12 * delta:
            delta = min(2.0 * delta, float(max_radius))
        if ratio > float(accept):
            x = x + move
        else:
            rejected += 1
        if delta < 1e-16:
            break
    counts = {kind: kinds.count(kind) for kind in set(kinds)}
    return {"x": x, "f": float(f(x)), "steps": step_number,
            "gradient_history": np.asarray(history), "radii": np.asarray(radii),
            "step_kinds": counts, "rejected": rejected, "final_radius": delta,
            "converged": bool(history and history[-1] <= float(tol))}


# --------------------------------------------------------------------------- measurements


def the_secant_condition_holds_and_needs_positive_curvature(dimension: int = 10) -> dict:
    """Check the update does what it was derived to do, and that ``y.s > 0`` throughout.

    Two separate claims. The secant condition is an algebraic identity the update satisfies by
    construction, so a residual above rounding would mean the implementation is wrong. Positive
    curvature is **not** an identity: it is a property of the step the line search returned, and it
    is the reason a strong Wolfe search is used here rather than backtracking.
    """
    out = bfgs(rosenbrock(int(dimension)))
    residuals = out["secant_residuals"]
    curvatures = out["curvatures"]
    return {
        "variables": int(dimension),
        "steps": out["steps"],
        "distance": float(np.linalg.norm(out["x"] - 1.0)),
        "worst_secant_residual": float(np.max(residuals)) if residuals.size else 0.0,
        "smallest_curvature": float(np.min(curvatures)) if curvatures.size else 0.0,
        "every_curvature_positive": bool(np.all(curvatures > 0.0)),
        "curvatures_below_the_update_threshold": int(np.count_nonzero(curvatures <= 1e-14)),
        "skipped_updates": out["skipped_updates"],
        "the_secant_condition_holds": bool(residuals.size and np.max(residuals) < 1e-8),
        "note": "the secant condition is an identity and holds to rounding; positive curvature is "
                "a property of the line search and is what the Wolfe conditions of lesson 86 were "
                "for, so the two are not independent choices",
    }


def superlinear_but_not_quadratic(dimension: int = 2) -> dict:
    """Two ratios, because one of them alone cannot tell the two rates apart.

    Superlinear means ``e_(k+1)/e_k -> 0``. Quadratic means ``e_(k+1)/e_k**2`` stays bounded. A
    method can satisfy the first and fail the second, and BFGS is exactly that case, so reporting
    only the first would leave it indistinguishable from Newton.
    """
    out = bfgs(rosenbrock(int(dimension)), tol=1e-13)
    history = out["gradient_history"]
    usable = history[history > 1e-14]
    tail = usable[-6:]
    linear = tail[1:] / tail[:-1]
    square = tail[1:] / tail[:-1] ** 2
    newton_constants = []
    from .gradient import newton
    other = newton(rosenbrock(int(dimension)))["gradient_history"]
    other = other[other > 1e-11]
    if other.size >= 2:
        newton_constants = [float(other[-1] / other[-2] ** 2)]
    return {
        "variables": int(dimension),
        "steps": out["steps"],
        "tail": tail,
        "linear_ratios": linear,
        "square_ratios": square,
        "biggest_linear_ratio": float(np.max(linear)),
        "ratios_are_far_below_one": bool(np.max(linear) < 0.1),
        "square_ratios_grow": bool(square[-1] > 1e4 * square[0]),
        "square_ratio_growth": float(square[-1] / square[0]),
        "newton_square_constant": newton_constants[0] if newton_constants else float("nan"),
        "usable_points": int(tail.size),
        "whether_it_tends_to_zero_is_not_resolvable": True,
        "note": "the second ratio grows by seven orders of magnitude, which rules out quadratic "
                "convergence; the first stays near 0.01 over the whole visible window, so the "
                "convergence is at least fast linear, and whether that ratio actually tends to "
                "zero cannot be settled in double precision because the run reaches the floor in "
                "six steps. Newton's second ratio stays at about 10**2 on the same problem, which "
                "is the contrast that matters.",
    }


def the_n_step_property_needs_an_exact_line_search(dimensions=(2, 5, 10, 20),
                                                   condition: float = 1000.0,
                                                   tightness=(0.9, 0.1, 0.01)) -> dict:
    """BFGS terminates in ``n`` steps on a quadratic, in exact arithmetic with exact line searches.

    Neither hypothesis holds in practice, and the measurement separates them. Tightening the
    curvature parameter ``c2`` toward zero makes the line search more nearly exact, and the step
    count falls toward ``n``, so the gap is the line search rather than the update.
    """
    rows = []
    for n in dimensions:
        problem = quadratic(condition=condition, dimension=int(n))
        counts = [bfgs(problem, tol=1e-10, wolfe_c2=c)["steps"] for c in tightness]
        rows.append({"variables": int(n), "n": int(n),
                     **{f"c2={c:g}": k for c, k in zip(tightness, counts)},
                     "tightest_over_n": counts[-1] / float(n)})
    loosest = f"c2={tightness[0]:g}"
    tightest = f"c2={tightness[-1]:g}"
    return {
        "rows": rows,
        "tightening_never_costs_steps": all(
            r[tightest] <= r[loosest] for r in rows),
        "tightest_is_within_three_of_n": all(
            r[tightest] - r["n"] <= 3 for r in rows),
        "worst_ratio_to_n": max(r["tightest_over_n"] for r in rows),
        "note": "the n step theorem assumes exact line searches, and the measured gap closes as "
                "the search is tightened, which is the evidence that the assumption and not the "
                "update is what the practical count is paying for",
    }


def the_line_search_decides_the_rate(dimension: int = 10,
                                     tightness=(0.9, 0.1, 0.01)) -> dict:
    """The same update with three line searches, on Rosenbrock, compared three ways.

    Step count, final accuracy of the curvature model, and per step error ratio all move together,
    and one of the runs lands on a **different minimum**. The update is identical in all three, so
    everything that differs is the line search.
    """
    n = int(dimension)
    problem = rosenbrock(n)
    exact = np.asarray(problem["hessian"](np.ones(n)), dtype=float)
    rows = []
    for c2 in tightness:
        out = bfgs(problem, tol=1e-13, wolfe_c2=c2)
        history = out["gradient_history"]
        tail = history[history > 1e-14][-6:]
        model_error = float(np.linalg.norm(out["inverse_hessian"] @ exact - np.eye(n))
                            / np.linalg.norm(np.eye(n)))
        rows.append({
            "wolfe_c2": float(c2),
            "steps": out["steps"],
            "distance": float(np.linalg.norm(out["x"] - 1.0)),
            "found_the_global_minimum": float(np.linalg.norm(out["x"] - 1.0)) < 1e-6,
            "curvature_model_error": model_error,
            "median_tail_ratio": float(np.median(tail[1:] / tail[:-1])),
            "line_search_calls": out["line_search_calls"],
        })
    return {
        "rows": rows,
        "tightening_improves_the_model": (rows[-1]["curvature_model_error"]
                                          < rows[0]["curvature_model_error"]),
        "tightening_reduces_the_steps": rows[-1]["steps"] < rows[0]["steps"],
        "model_error_improvement": (rows[0]["curvature_model_error"]
                                    / rows[-1]["curvature_model_error"]),
        "someone_found_a_different_minimum": any(
            not r["found_the_global_minimum"] for r in rows),
        "note": "the same update with a tighter line search builds a better curvature model, takes "
                "fewer steps and converges faster per step, so the rate is a property of the pair "
                "and not of the update alone",
    }


def limited_memory_costs_little(dimension: int = 100, memories=(1, 3, 5, 10, 25, 50)) -> dict:
    """L-BFGS at several memory sizes against full BFGS, in steps and in storage.

    The question is how much curvature has to be remembered. Full BFGS stores ``n**2`` numbers and
    L-BFGS stores ``2mn``, so at ``m = 5`` and ``n = 1000`` that is ``10**4`` against ``10**6``.
    """
    n = int(dimension)
    problem = rosenbrock(n)
    full = bfgs(problem, tol=1e-8)
    rows = []
    for m in memories:
        out = lbfgs(problem, memory=int(m), tol=1e-8)
        rows.append({
            "memory": int(m),
            "steps": out["steps"],
            "distance": float(np.linalg.norm(out["x"] - 1.0)),
            "stored_numbers": 2 * int(m) * n,
            "steps_over_full": out["steps"] / max(full["steps"], 1),
            "converged": bool(out["converged"]),
        })
    useful = [r for r in rows if r["converged"] and r["memory"] >= 5]
    return {
        "rows": rows,
        "full_bfgs_steps": full["steps"],
        "full_bfgs_storage": n * n,
        "variables": n,
        "best_memory": min(rows, key=lambda r: r["steps"])["memory"],
        "spread_above_five_pairs": (
            max(r["steps"] for r in useful) / min(r["steps"] for r in useful)
            if useful else float("nan")),
        "storage_saving_at_five": (n * n) / float(2 * 5 * n),
        "cheapest_that_beats_full": min(
            (r for r in rows if r["converged"] and r["steps"] <= full["steps"] * 1.5),
            key=lambda r: r["stored_numbers"], default={"memory": None})["memory"],
        "note": "the storage falls from n**2 to 2mn, so the saving is real only when m is well "
                "below n/2; the step count is not monotone in m, because a longer memory can hold "
                "curvature from a part of the valley the iterate has already left",
    }


def a_trust_region_escapes_the_saddle(dimensions=(2, 4, 5, 6, 7, 10)) -> dict:
    """The problem lesson 86's plain Newton failed on, given to a trust region instead.

    At four variables the plain Newton iteration converged to a **saddle**, because it solves
    ``grad f = 0`` and does not look at the curvature. A trust region minimizes the model inside a
    ball, and where the model is not convex the constrained minimum sits on the boundary in a
    direction of negative curvature: the method walks **downhill off the saddle** rather than
    stopping on it, with no Hessian shift anywhere.
    """
    from .gradient import newton
    from .optimize import classify

    rows = []
    for n in dimensions:
        problem = rosenbrock(int(n))
        plain = newton(problem)
        region = trust_region(problem, max_steps=20000)
        plain_far = float(np.linalg.norm(plain["x"] - 1.0))
        region_far = float(np.linalg.norm(region["x"] - 1.0))
        rows.append({
            "variables": int(n),
            "newton_distance": plain_far,
            "newton_verdict": classify(problem["hessian"](plain["x"])),
            "trust_steps": region["steps"],
            "trust_distance": region_far,
            "trust_verdict": classify(problem["hessian"](region["x"])),
            "rejected": region["rejected"],
            "step_kinds": region["step_kinds"],
            "rescued": bool(plain_far > 1e-6 and region_far < 1e-6),
        })
    saddles = [r for r in rows if r["newton_verdict"] == "saddle"]
    return {
        "rows": rows,
        "newton_saddles": [r["variables"] for r in saddles],
        "rescued": [r["variables"] for r in rows if r["rescued"]],
        "it_escapes_every_saddle": bool(saddles) and all(
            r["trust_verdict"] != "saddle" for r in saddles),
        "no_hessian_was_shifted": True,
        "note": "a bounded quadratic model has a minimum whether or not it is convex, so negative "
                "curvature is a direction to exploit rather than a defect to repair",
    }


def which_beta_for_nonlinear_cg(dimensions=(2, 5, 10),
                                rules=("fletcher-reeves", "polak-ribiere",
                                       "polak-ribiere-plus")) -> dict:
    """The three standard formulas for ``beta``, on the same problems from the same start.

    On a quadratic all three are lesson 24's method and agree. On Rosenbrock they do not, and the
    difference is entirely in what happens after a bad direction: Fletcher-Reeves keeps it and
    Polak-Ribiere throws it away.
    """
    rows = []
    for n in dimensions:
        problem = rosenbrock(int(n))
        for rule in rules:
            out = nonlinear_cg(problem, rule=rule, tol=1e-8, max_steps=50000)
            rows.append({
                "variables": int(n),
                "rule": rule,
                "steps": out["steps"],
                "distance": float(np.linalg.norm(out["x"] - 1.0)),
                "restarts": out["restarts"],
                "converged": bool(out["converged"]),
            })
    by_rule = {}
    for rule in rules:
        mine = [r for r in rows if r["rule"] == rule]
        by_rule[rule] = {
            "converged": sum(1 for r in mine if r["converged"]),
            "total_steps": sum(r["steps"] for r in mine),
        }
    return {
        "rows": rows,
        "by_rule": by_rule,
        "best_rule": min(by_rule, key=lambda r: (-by_rule[r]["converged"],
                                                 by_rule[r]["total_steps"])),
        "they_differ": len({by_rule[r]["total_steps"] for r in rules}) > 1,
        "note": "the three agree on a quadratic and separate on Rosenbrock, and the separation is "
                "about recovering from a bad direction rather than about the good ones",
    }


def the_cost_of_curvature(dimensions=(2, 5, 10, 20)) -> dict:
    """Newton, BFGS and L-BFGS on the same problem, counted in derivative evaluations.

    A gradient costs ``2n`` evaluations by differences and a Hessian ``2n(n+1)``, so a Newton step
    costs ``2(n+2)`` times a quasi-Newton step. Newton needs far fewer steps. Which wins is a
    measurement, and it is the measurement that decides what production codes use.
    """
    from .gradient import newton

    rows = []
    for n in dimensions:
        problem = rosenbrock(int(n))
        fast = newton(problem, tol=1e-8)
        quasi = bfgs(problem, tol=1e-8)
        limited = lbfgs(problem, memory=5, tol=1e-8)
        gradient_cost = 2 * n
        hessian_cost = 2 * n * (n + 1)
        rows.append({
            "variables": int(n),
            "newton_steps": fast["steps"],
            "newton_evaluations": fast["steps"] * (gradient_cost + hessian_cost),
            "newton_distance": float(np.linalg.norm(fast["x"] - 1.0)),
            "bfgs_steps": quasi["steps"],
            "bfgs_evaluations": quasi["steps"] * gradient_cost,
            "bfgs_distance": float(np.linalg.norm(quasi["x"] - 1.0)),
            "lbfgs_steps": limited["steps"],
            "lbfgs_evaluations": limited["steps"] * gradient_cost,
        })
    for row in rows:
        row["bfgs_over_newton"] = (row["bfgs_evaluations"]
                                   / max(row["newton_evaluations"], 1))
    return {
        "rows": rows,
        "bfgs_is_cheaper_somewhere": any(r["bfgs_over_newton"] < 1.0 for r in rows),
        "sizes_where_bfgs_is_cheaper": [r["variables"] for r in rows
                                        if r["bfgs_over_newton"] < 1.0],
        "note": "the Hessian costs 2(n+2) gradients, so Newton has to be that many times faster in "
                "steps before it is faster at all, and the crossover is measured here rather than "
                "asserted",
    }


# --------------------------------------------------------------------------- an application


def lennard_jones(atoms: int = 5):
    """A molecular conformation problem: place ``atoms`` particles to minimize a pair potential.

    The Lennard-Jones potential between two atoms a distance ``r`` apart is
    ``4(r**-12 - r**-6)``, repulsive at short range and attractive at long. The energy of a cluster
    is the sum over all pairs, so the problem has ``3 * atoms`` variables and the minimum is the
    shape the cluster actually takes.

    This is Sauer's molecular conformation example, and it is here because it is what the machinery
    of this lesson is for: the gradient is available analytically, the Hessian is ``(3N)**2`` and
    nobody forms it, and the number of local minima grows so fast with ``N`` that the interesting
    question stops being the rate and becomes the basin.
    """
    n = int(atoms)
    if n < 2:
        raise ValueError(f"need at least two atoms, got {n}")

    def pairs_of(flat):
        return np.asarray(flat, dtype=float).reshape(n, 3)

    def f(flat):
        points = pairs_of(flat)
        total = 0.0
        for i in range(n):
            gap = points[i + 1:] - points[i]
            distance_squared = np.maximum(np.sum(gap * gap, axis=1), 1e-12)
            inverse_six = distance_squared ** -3
            total += float(np.sum(4.0 * (inverse_six * inverse_six - inverse_six)))
        return total

    def gradient(flat):
        points = pairs_of(flat)
        out = np.zeros_like(points)
        for i in range(n):
            gap = points[i + 1:] - points[i]
            distance_squared = np.maximum(np.sum(gap * gap, axis=1), 1e-12)
            inverse_six = distance_squared ** -3
            scale = 24.0 * (inverse_six - 2.0 * inverse_six * inverse_six) / distance_squared
            push = scale[:, None] * gap
            out[i + 1:] += push
            out[i] -= np.sum(push, axis=0)
        return out.ravel()

    rng = np.random.default_rng(42)
    return {"f": f, "gradient": gradient, "hessian": None, "atoms": n,
            "dimension": 3 * n, "start": rng.normal(scale=0.8, size=3 * n),
            "minimizers": [], "minimum": None, "convex": False,
            "name": f"Lennard-Jones cluster, {n} atoms"}


def a_cluster_has_many_minima(atoms=(3, 5, 7), tries: int = 25, seed: int = 42) -> dict:
    """Minimize the same cluster from many random starts and count the distinct energies found.

    Every run converges: the gradient reaches ``10**-8`` and L-BFGS reports success each time. The
    energies they converge to are not the same, and the number of distinct ones grows with the
    number of atoms, which is the practical face of everything lessons 85 to 87 have said about
    basins.
    """
    rng = np.random.default_rng(int(seed))
    rows = []
    for n in atoms:
        problem = lennard_jones(int(n))
        energies, failures = [], 0
        for _ in range(int(tries)):
            start = rng.normal(scale=0.9, size=3 * int(n))
            out = lbfgs(problem, start=start, memory=10, tol=1e-8, max_steps=4000)
            if out["converged"]:
                energies.append(round(out["f"], 4))
            else:
                failures += 1
        distinct = sorted(set(energies))
        rows.append({
            "atoms": int(n),
            "variables": 3 * int(n),
            "runs_that_converged": len(energies),
            "failed": failures,
            "distinct_energies": len(distinct),
            "lowest": min(energies) if energies else float("nan"),
            "highest": max(energies) if energies else float("nan"),
            "how_often_the_best_was_found": (energies.count(min(energies)) / len(energies)
                                             if energies else float("nan")),
        })
    return {
        "rows": rows,
        "tries": int(tries),
        "runs_that_hit_the_iteration_cap": sum(r["failed"] for r in rows),
        "almost_every_run_converged": (
            sum(r["failed"] for r in rows) <= 0.05 * int(tries) * len(rows)),
        "the_count_grows_with_the_atoms": all(
            rows[k]["distinct_energies"] >= rows[k - 1]["distinct_energies"]
            for k in range(1, len(rows))),
        "worst_success_rate": min(r["how_often_the_best_was_found"] for r in rows),
        "known_global_minima": {3: -3.0, 5: -9.1039, 7: -16.5054},
        "matches_the_published_values": all(
            abs(r["lowest"] - {3: -3.0, 5: -9.1039, 7: -16.5054}[r["atoms"]]) < 5e-3
            for r in rows if r["atoms"] in (3, 5, 7)),
        "note": "nearly every run converges and they do not agree, so a converged local method on "
                "a cluster answers a question about the starting point; the lowest energy found "
                "does match the published global minimum at each size, and how often it is found "
                "falls from 80 per cent to 12 as the atoms go from three to seven",
    }
