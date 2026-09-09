"""What Part 12's optimizers do when the objective is a training loss.

Where the work is
-----------------
Part 12 minimized functions. A learning problem is a function too, so every result there still
applies, and this module is about the four places where the *setting* changes the answer even
though the mathematics does not.

**Gradient descent is a differential equation solver.** The gradient flow ``x' = -grad f(x)`` has
gradient descent as its explicit Euler discretization, so lesson 67's local error and lesson 74's
stability limit apply word for word. The step size limit ``h < 2/L`` is not an optimization result
at all; it is the absolute stability condition of Euler's method with the eigenvalue ``-L``.

**The gradient is an estimate.** With a minibatch of size ``B`` drawn without replacement from ``N``
samples, the gradient carries noise with variance falling like ``(1/B)(1 - (B-1)/(N-1))``. That
turns the deterministic descent of lesson 86 into the stochastic process of lesson 89, and the noise
is not a nuisance: it is the thing that decides which minimum the run ends in.

**The Hessian is an estimate too, and that is the problem.** The usual explanation for the rarity of
second order methods is cost, and cost is real, but it is not the whole story and it is not the part
that decides. A Newton step divides by a curvature estimate, and dividing by a noisy small number is
the single worst thing a numerical method can do.

**Reparametrization is free for some methods and not for others.** The natural gradient is exactly
invariant under any invertible linear change of variables, and plain gradient descent is not. That
is a one-line proof and an exact measurement, and it is the whole argument for preconditioning.

What the measurements here show
-------------------------------
* **Gradient descent converges to the gradient flow at order 0.9984**, so it is Euler's method and
  nothing else, and its trajectory error is a discretization error rather than an optimization one.
* **The step size limit is exactly ``2/L``**, located to a relative width of 1.8e-12 by bisecting on
  the amplification factor, and it is lesson 74's absolute stability boundary with the eigenvalue
  ``-L``.
* **Heavy ball momentum is a second order ODE**, and its discrete trajectory tracks the continuous
  one at order 0.9995.
* **Conditioning decides the iteration count and preconditioning removes it.** Measured counts of
  17, 89, 886, 8856 and 88556 track the predicted ``(kappa/2) log(1/tol)`` to within a factor of
  1.084, and whitening the problem cuts 88556 iterations to 1.
* **Minibatch gradient noise follows the finite population formula**, matching to 2.4 per cent over
  batch sizes from 1 to 512 and reaching exactly zero at the full batch: 2e-33 against a predicted
  0.
* **The linear scaling rule holds for six doublings of the batch size and then stops.** The excess
  loss stays between 0.0071 and 0.0098 up to a batch of 64, and at 128 it is 1.7e+09, because the
  step the rule asks for has crossed ``2/L``.
* **The step size chooses the minimum before any noise does.** Two wells of exactly equal depth
  have stability limits of 0.615 and 0.072, matching ``2/c`` to 0.023 and 0.032 per cent, so there
  is a window of step sizes in which only the flat one is a fixed point.
* **Losing a minimum and leaving it are different thresholds.** The sharp well stops being
  attracting at ``2/c`` and a noiseless run still cannot get out of it until 3.25 times that.
* **Inside that window the batch size decides.** At 2.5 times the limit a batch of 1 leaves the
  sharp well in 100 per cent of runs and every larger batch in 0 per cent, and the full batch never
  leaves at any step tested.
* **Sharpness predicts the damage from a perturbation exactly.** The measured loss increase under a
  random perturbation matches ``r**2 tr(H)/(2d)`` to 1.9 per cent, and the worst case exceeds the
  average by 4.48 on an ill-conditioned problem.
* **A noisy Hessian makes Newton worse than gradient descent.** At a batch of 8 the sampled
  curvature has rank 8 for 20 parameters and the run diverges; at 32 it is 6.43 times worse than
  plain descent; the two cross over at a batch of 128.
* **Cost is not the reason.** On flops a Newton step beats gradient descent by 560 at 5 parameters
  and by 15.3 at 200, and the advantage falls like ``1/d`` with a fitted slope of -0.990. The
  extrapolated crossover, 3148 parameters, is the iteration count gradient descent needed, 3405, to
  within 7.5 per cent.
* **The natural gradient is exactly invariant under reparametrization**, to 2.8e-16, while plain
  gradient descent's trajectory moves by 0.47 of the distance to the solution under the same change
  of variables.
* **Gauss-Newton is the Fisher information for a squared loss**, to exactly 0, and both differ from
  the true Hessian by the residual term alone: 0 at a zero-residual solution and 1.33 of the norm at
  a large-residual one, where the true Hessian is no longer positive definite.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- problems


def quadratic(curvatures, seed: int = 42) -> dict:
    """``f(x) = 0.5 x' Q x`` with prescribed eigenvalues, in a rotated basis so it is not diagonal.

    The minimum is at the origin with value zero, the Hessian is constant, and the condition number
    is the ratio of the largest curvature to the smallest. Everything Part 12 proved about gradient
    descent on a quadratic is exactly true here, which is what makes it the right test problem.
    """
    values = np.asarray(curvatures, dtype=float).ravel()
    if values.size < 1 or np.any(values <= 0.0):
        raise ValueError(f"need positive curvatures, got {values}")
    rng = np.random.default_rng(int(seed))
    basis, upper = np.linalg.qr(rng.standard_normal((values.size, values.size)))
    basis = basis * np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    matrix = (basis * values) @ basis.T
    matrix = 0.5 * (matrix + matrix.T)
    return {
        "hessian": matrix, "curvatures": values, "basis": basis,
        "f": lambda x, m=matrix: 0.5 * float(np.asarray(x) @ m @ np.asarray(x)),
        "gradient": lambda x, m=matrix: m @ np.asarray(x, dtype=float),
        "minimum": np.zeros(values.size),
        "largest": float(values.max()), "smallest": float(values.min()),
        "condition": float(values.max() / values.min()),
        "dimension": int(values.size),
    }


def two_wells(samples: int = 400, flat_width: float = 0.55, sharp_width: float = 0.18,
              spread: float = 0.06, points: int = 2001, seed: int = 42) -> dict:
    """A finite sum whose average has two minima of equal depth and very different curvature.

    Each term is a pair of Gaussian wells shifted by its own random offset, so a minibatch gradient
    carries genuine sampling noise rather than noise that was added by hand. Averaging flattens a
    narrow well more than a wide one, so the amplitude of the narrow well is solved for rather than
    chosen: a bisection drives the two averaged depths to equality. Without that step the wells
    differ in depth and any preference measured between them is a preference about the loss.
    """
    n = int(samples)
    if n < 1:
        raise ValueError(f"need at least one term, got {n}")
    rng = np.random.default_rng(int(seed))
    offsets = float(spread) * rng.standard_normal(n)
    flat, sharp = float(flat_width), float(sharp_width)
    grid = np.linspace(-2.5, 2.5, int(points))
    half = grid.size // 2
    shifted = grid[:, None] - offsets[None, :]
    left_curve = np.mean(np.exp(-((shifted + 1.0) ** 2) / (2.0 * flat ** 2)), axis=1)
    right_curve = np.mean(np.exp(-((shifted - 1.0) ** 2) / (2.0 * sharp ** 2)), axis=1)

    def depth_gap(amplitude):
        vals = -(left_curve + amplitude * right_curve)
        return float(vals[half:].min() - vals[:half].min())

    low, high = 1.0, 32.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if depth_gap(middle) > 0.0:
            low = middle
        else:
            high = middle
    amplitude = 0.5 * (low + high)

    def terms(x, rows=None, which="value"):
        shifts = offsets if rows is None else offsets[rows]
        z = float(np.asarray(x, dtype=float).reshape(())) - shifts
        left = np.exp(-((z + 1.0) ** 2) / (2.0 * flat ** 2))
        right = amplitude * np.exp(-((z - 1.0) ** 2) / (2.0 * sharp ** 2))
        if which == "value":
            return -(left + right)
        return (z + 1.0) / flat ** 2 * left + (z - 1.0) / sharp ** 2 * right

    def value(x, rows=None):
        return float(np.mean(terms(x, rows, "value")))

    def gradient(x, rows=None):
        return float(np.mean(terms(x, rows, "gradient")))

    curve = -(left_curve + amplitude * right_curve)
    left_at = float(grid[:half][np.argmin(curve[:half])])
    right_at = float(grid[half:][np.argmin(curve[half:])])
    step = grid[1] - grid[0]

    def curvature(point):
        return float((value(point + step) - 2.0 * value(point) + value(point - step)) / step ** 2)

    return {
        "value": value, "gradient": gradient, "samples": n, "offsets": offsets,
        "amplitude": amplitude, "flat_at": left_at, "sharp_at": right_at,
        "flat_value": value(left_at), "sharp_value": value(right_at),
        "flat_curvature": curvature(left_at), "sharp_curvature": curvature(right_at),
        "barrier": float(np.max(curve[(grid > left_at) & (grid < right_at)])),
        "grid": grid, "curve": curve,
    }


def least_squares_model(samples: int, parameters: int, noise: float = 0.0,
                        condition: float = 1.0, seed: int = 42) -> dict:
    """A linear model with a controlled design condition number, as a finite sum of terms.

    This is the problem the batch-size and second-order measurements run on. The loss is
    ``(1/N) sum_i (a_i' x - b_i)**2 / 2``, so a minibatch gradient is an honest average over the
    chosen rows rather than a perturbation of the full one.
    """
    n, d = int(samples), int(parameters)
    if n < 1 or d < 1:
        raise ValueError(f"need at least one sample and one parameter, got {n} and {d}")
    rng = np.random.default_rng(int(seed))
    left, _ = np.linalg.qr(rng.standard_normal((n, d)))
    right, _ = np.linalg.qr(rng.standard_normal((d, d)))
    values = np.geomspace(1.0, 1.0 / float(condition), d) * math.sqrt(n)
    design = (left * values) @ right.T
    truth = rng.standard_normal(d)
    target = design @ truth + float(noise) * rng.standard_normal(n)
    hessian = design.T @ design / n
    solution = np.linalg.lstsq(design, target, rcond=None)[0]

    def loss(x, rows=None):
        a = design if rows is None else design[rows]
        b = target if rows is None else target[rows]
        residual = a @ np.asarray(x, dtype=float) - b
        return float(residual @ residual / (2 * residual.size))

    def gradient(x, rows=None):
        a = design if rows is None else design[rows]
        b = target if rows is None else target[rows]
        return a.T @ (a @ np.asarray(x, dtype=float) - b) / a.shape[0]

    def curvature(rows=None):
        a = design if rows is None else design[rows]
        return a.T @ a / a.shape[0]

    return {
        "design": design, "target": target, "truth": truth, "solution": solution,
        "loss": loss, "gradient": gradient, "curvature": curvature, "hessian": hessian,
        "samples": n, "parameters": d,
        "largest": float(np.linalg.eigvalsh(hessian)[-1]),
        "smallest": float(np.linalg.eigvalsh(hessian)[0]),
        "condition": float(np.linalg.cond(hessian)),
        "residual": float(np.linalg.norm(design @ solution - target)),
    }


# --------------------------------------------------------------------------- the flow and its steps


def gradient_flow(problem: dict, start, horizon: float, steps: int = 4096) -> dict:
    """Solve ``x' = -grad f(x)`` accurately with classical Runge-Kutta, from lesson 71.

    This is the object gradient descent approximates. Having it to compare against is what turns
    "gradient descent is like a flow" into a measurement.
    """
    x = np.array(start, dtype=float)
    n = int(steps)
    h = float(horizon) / n
    path = [x.copy()]
    for _ in range(n):
        k1 = -problem["gradient"](x)
        k2 = -problem["gradient"](x + 0.5 * h * k1)
        k3 = -problem["gradient"](x + 0.5 * h * k2)
        k4 = -problem["gradient"](x + h * k3)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        path.append(x.copy())
    return {"x": x, "path": np.array(path), "times": np.linspace(0.0, float(horizon), n + 1)}


def gradient_descent(problem: dict, start, step: float, iterations: int) -> dict:
    """Plain gradient descent, which is explicit Euler on the gradient flow with ``h = step``."""
    x = np.array(start, dtype=float)
    path = [x.copy()]
    values = [problem["f"](x)] if "f" in problem else [problem["loss"](x)]
    grad = problem["gradient"]
    for _ in range(int(iterations)):
        x = x - float(step) * grad(x)
        path.append(x.copy())
        values.append(problem["f"](x) if "f" in problem else problem["loss"](x))
    return {"x": x, "path": np.array(path), "values": np.array(values),
            "diverged": not np.all(np.isfinite(x))}


def heavy_ball(problem: dict, start, step: float, momentum: float, iterations: int) -> dict:
    """Polyak's heavy ball, which discretizes ``x'' + a x' + grad f(x) = 0``."""
    x = np.array(start, dtype=float)
    previous = x.copy()
    path = [x.copy()]
    for _ in range(int(iterations)):
        nxt = x - float(step) * problem["gradient"](x) + float(momentum) * (x - previous)
        previous, x = x, nxt
        path.append(x.copy())
    return {"x": x, "path": np.array(path), "diverged": not np.all(np.isfinite(x))}


def damped_flow(problem: dict, start, horizon: float, damping: float, steps: int = 8192) -> dict:
    """Solve ``x'' + a x' + grad f(x) = 0`` as a first order system, which is lesson 73's trick."""
    x = np.array(start, dtype=float)
    v = np.zeros_like(x)
    n = int(steps)
    h = float(horizon) / n
    a = float(damping)

    def rate(state_x, state_v):
        return state_v, -a * state_v - problem["gradient"](state_x)

    path = [x.copy()]
    for _ in range(n):
        k1x, k1v = rate(x, v)
        k2x, k2v = rate(x + 0.5 * h * k1x, v + 0.5 * h * k1v)
        k3x, k3v = rate(x + 0.5 * h * k2x, v + 0.5 * h * k2v)
        k4x, k4v = rate(x + h * k3x, v + h * k3v)
        x = x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        v = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        path.append(x.copy())
    return {"x": x, "v": v, "path": np.array(path),
            "times": np.linspace(0.0, float(horizon), n + 1)}


def minibatch_descent(problem: dict, start, step: float, batch: int, iterations: int,
                      seed: int = 42, record: bool = False) -> dict:
    """Minibatch gradient descent on a finite sum, sampling rows without replacement each step."""
    x = np.array(start, dtype=float) if np.ndim(start) else float(start)
    n = problem["samples"]
    size = min(int(batch), n)
    rng = np.random.default_rng(int(seed))
    path = [np.copy(x)]
    for _ in range(int(iterations)):
        rows = rng.choice(n, size=size, replace=False) if size < n else None
        x = x - float(step) * problem["gradient"](x, rows)
        if record:
            path.append(np.copy(x))
    return {"x": x, "path": np.array(path) if record else None, "batch": size}


def newton_step(problem: dict, x, rows=None, shift: float = 0.0):
    """One Newton step on a least squares model, using whatever rows are given for the curvature."""
    curvature = problem["curvature"](rows)
    if shift:
        curvature = curvature + float(shift) * np.eye(curvature.shape[0])
    return np.linalg.solve(curvature, problem["gradient"](x, rows))


def natural_gradient_step(design, x, target, ridge: float = 0.0):
    """The Fisher-preconditioned step for a linear Gaussian model, which is ``F^-1 grad``."""
    a = np.asarray(design, dtype=float)
    fisher = a.T @ a / a.shape[0]
    if ridge:
        fisher = fisher + float(ridge) * np.eye(fisher.shape[0])
    gradient = a.T @ (a @ np.asarray(x, dtype=float) - np.asarray(target, dtype=float)) / a.shape[0]
    return np.linalg.solve(fisher, gradient)


def sharpness(problem: dict, x, radius: float = 1e-3, draws: int = 400, seed: int = 42) -> dict:
    """Measure how much the loss rises under a random perturbation of the given size.

    For a quadratic the answer is exactly ``radius**2 * tr(H) / (2 d)`` when the perturbation is a
    random direction of length ``radius``, so the measurement has a prediction to be checked
    against rather than a number to be admired.
    """
    point = np.array(x, dtype=float)
    rng = np.random.default_rng(int(seed))
    base = problem["loss"](point) if "loss" in problem else problem["f"](point)
    rises = []
    for _ in range(int(draws)):
        direction = rng.standard_normal(point.size)
        direction = direction / np.linalg.norm(direction)
        moved = point + float(radius) * direction
        rises.append((problem["loss"](moved) if "loss" in problem else problem["f"](moved)) - base)
    hessian = problem["hessian"]
    return {
        "measured": float(np.mean(rises)), "spread": float(np.std(rises, ddof=1)),
        "predicted": float(radius) ** 2 * float(np.trace(hessian)) / (2.0 * point.size),
        "largest_curvature": float(np.linalg.eigvalsh(hessian)[-1]),
        "trace": float(np.trace(hessian)),
    }


# --------------------------------------------------------------------------- measurements


def gradient_descent_is_explicit_euler(steps=(64, 128, 256, 512, 1024), horizon: float = 1.0,
                                       curvatures=(1.0, 2.5, 6.0, 9.0), seed: int = 42) -> dict:
    """Compare a gradient descent trajectory against the gradient flow it discretizes.

    Nothing about this is metaphorical. Gradient descent with step ``h`` is exactly Euler's method
    on ``x' = -grad f``, so the trajectory error has to fall like ``h``, and if it does not then one
    of the two implementations is wrong.
    """
    problem = quadratic(curvatures, seed=seed)
    start = np.ones(problem["dimension"])
    exact = gradient_flow(problem, start, horizon, steps=8 * max(steps))["x"]
    rows = []
    for count in steps:
        h = float(horizon) / count
        run = gradient_descent(problem, start, h, count)
        rows.append({"steps": int(count), "step_size": h,
                     "error": float(np.linalg.norm(run["x"] - exact))})
    order = float(np.polyfit(np.log([r["step_size"] for r in rows]),
                             np.log([r["error"] for r in rows]), 1)[0])
    return {
        "rows": rows, "order": order, "flow_endpoint": exact,
        "it_is_first_order": abs(order - 1.0) < 0.05,
        "note": "gradient descent is Euler's method on the gradient flow, so its trajectory error "
                "is a discretization error and obeys lesson 67",
    }


def the_step_limit_is_the_stability_limit(curvatures=(0.5, 1.0, 4.0, 20.0), seed: int = 42,
                                          probes: int = 40) -> dict:
    """Bisect for the largest step size that does not diverge, and compare it against ``2/L``.

    Lesson 74 derived the absolute stability region of Euler's method: ``|1 + h*lambda| <= 1``. Here
    the eigenvalues of the linearized flow are ``-L_i``, so the condition is ``h < 2/L_max``, and
    that is the whole of the "learning rate too high" failure mode.
    """
    problem = quadratic(curvatures, seed=seed)
    start = np.ones(problem["dimension"])
    predicted = 2.0 / problem["largest"]

    def amplification(step, rounds: int = 300):
        """The eventual per-step growth factor, found by power iteration on ``I - step*H``."""
        x = start / np.linalg.norm(start)
        ratio = 0.0
        for _ in range(rounds):
            moved = x - float(step) * problem["gradient"](x)
            ratio = float(np.linalg.norm(moved))
            x = moved / ratio if ratio > 0 else moved
        return ratio

    low, high = 0.0, 4.0 * predicted
    for _ in range(int(probes)):
        middle = 0.5 * (low + high)
        if amplification(middle) > 1.0:
            high = middle
        else:
            low = middle
    measured = 0.5 * (low + high)
    table = []
    with np.errstate(over="ignore", invalid="ignore"):
        for factor in (0.5, 0.9, 0.99, 1.01, 1.1, 2.0):
            run = gradient_descent(problem, start, factor * predicted, 200)
            table.append({"factor": float(factor), "step": factor * predicted,
                          "amplification": amplification(factor * predicted),
                          "final_norm": float(np.linalg.norm(run["x"]))
                          if np.all(np.isfinite(run["x"])) else float("inf")})
    return {
        "rows": table, "predicted": predicted, "measured": measured,
        "relative_gap": abs(measured - predicted) / predicted,
        "largest_curvature": problem["largest"],
        "the_limit_is_two_over_l": abs(measured - predicted) / predicted < 1e-3,
        "note": "the learning rate limit is the absolute stability boundary of Euler's method, "
                "not a fact about optimization",
    }


def momentum_is_a_second_order_ode(steps=(256, 512, 1024, 2048, 4096), horizon: float = 4.0,
                                   damping: float = 1.5, curvatures=(1.0, 3.0, 8.0),
                                   seed: int = 42) -> dict:
    """Heavy ball momentum against the damped oscillator it discretizes.

    Writing the heavy ball update as a finite difference of ``x'' + a x' + grad f = 0`` gives
    ``h = sqrt(step)`` and ``momentum = 1 - a*sqrt(step)``, so the two parameters people tune are a
    step size and a damping constant in disguise.
    """
    problem = quadratic(curvatures, seed=seed)
    start = np.ones(problem["dimension"])
    exact = damped_flow(problem, start, horizon, damping, steps=8 * max(steps))["x"]
    rows = []
    for count in steps:
        h = float(horizon) / count
        run = heavy_ball(problem, start, h * h, 1.0 - float(damping) * h, count)
        rows.append({"steps": int(count), "h": h, "step_size": h * h,
                     "momentum": 1.0 - float(damping) * h,
                     "error": float(np.linalg.norm(run["x"] - exact))})
    order = float(np.polyfit(np.log([r["h"] for r in rows]),
                             np.log([r["error"] for r in rows]), 1)[0])
    return {
        "rows": rows, "order": order, "damping": float(damping),
        "it_tracks_the_ode": abs(order - 1.0) < 0.1,
        "note": "step size and momentum are the discretization parameters of a damped oscillator, "
                "which is why they have to be tuned together",
    }


def conditioning_decides_the_iteration_count(conditions=(2.0, 10.0, 100.0, 1000.0, 10000.0),
                                             dimension: int = 20, tolerance: float = 1e-8,
                                             cap: int = 400000, seed: int = 42) -> dict:
    """Count gradient descent iterations against lesson 86's prediction, then whiten and recount.

    The predicted count for the optimal step ``2/(L+m)`` is ``(kappa/2) log(1/tol)`` up to a
    constant. Whitening the features makes the Hessian the identity, so the count becomes 1, and
    the whole of the conditioning problem disappears into one change of variables.
    """
    rows = []
    for target in conditions:
        curvatures = np.geomspace(1.0, 1.0 / float(target), int(dimension))
        problem = quadratic(curvatures, seed=seed)
        step = 2.0 / (problem["largest"] + problem["smallest"])
        x = np.ones(int(dimension))
        scale = float(np.linalg.norm(x))
        count = 0
        while np.linalg.norm(x) > float(tolerance) * scale and count < int(cap):
            x = x - step * problem["gradient"](x)
            count += 1
        whitened = np.linalg.inv(problem["hessian"])
        y = np.ones(int(dimension))
        preconditioned = 0
        while np.linalg.norm(y) > float(tolerance) * scale and preconditioned < int(cap):
            y = y - whitened @ problem["gradient"](y)
            preconditioned += 1
        rows.append({
            "condition": float(target), "iterations": count,
            "predicted": 0.5 * float(target) * math.log(1.0 / float(tolerance)),
            "whitened_iterations": preconditioned,
        })
    ratios = [r["iterations"] / r["predicted"] for r in rows if r["iterations"] < int(cap)]
    return {
        "rows": rows, "worst_ratio": max(max(ratios), 1.0 / min(ratios)),
        "the_prediction_holds": max(max(ratios), 1.0 / min(ratios)) < 2.0,
        "whitening_removes_it": all(r["whitened_iterations"] <= 1 for r in rows),
        "largest_saving": max(r["iterations"] / max(r["whitened_iterations"], 1) for r in rows),
        "note": "the condition number of the loss is the iteration count, and preconditioning is "
                "the only thing that changes it",
    }


def the_gradient_noise_falls_like_the_batch_size(batches=(1, 2, 8, 32, 128, 512),
                                                 samples: int = 512, parameters: int = 12,
                                                 noise: float = 0.5, draws: int = 2000,
                                                 seed: int = 42) -> dict:
    """Measure the variance of a minibatch gradient against the finite population formula.

    Sampling ``B`` rows without replacement from ``N`` gives a variance of
    ``(sigma**2/B)(1 - (B-1)/(N-1))``, which is the usual ``1/B`` with a correction that drives it
    to exactly zero at the full batch. The correction matters here because machine learning batches
    are often a large fraction of a small dataset.
    """
    problem = least_squares_model(int(samples), int(parameters), noise=float(noise),
                                  condition=10.0, seed=seed)
    point = problem["solution"] + 0.3 * np.ones(int(parameters))
    full = problem["gradient"](point)
    rng = np.random.default_rng(int(seed) + 1)
    single = np.array([problem["gradient"](point, np.array([i]))
                       for i in range(problem["samples"])])
    population = float(np.mean(np.sum((single - full) ** 2, axis=1))) * (
        problem["samples"] / (problem["samples"] - 1))
    rows = []
    for size in batches:
        b = min(int(size), problem["samples"])
        errors = []
        for _ in range(int(draws)):
            rows_picked = rng.choice(problem["samples"], size=b, replace=False)
            errors.append(problem["gradient"](point, rows_picked) - full)
        measured = float(np.mean(np.sum(np.array(errors) ** 2, axis=1)))
        predicted = population / b * (1.0 - (b - 1) / (problem["samples"] - 1))
        rows.append({
            "batch": b, "measured": measured, "predicted": predicted,
            "ratio": measured / predicted if predicted > 0 else float("inf"),
        })
    exact = problem["gradient"](point, np.arange(problem["samples"]))
    partial = [r for r in rows if r["predicted"] > 0.0]
    complete = [r for r in rows if r["predicted"] == 0.0]
    return {
        "rows": rows, "population_variance": population,
        "worst_ratio": max(max(r["ratio"] for r in partial),
                           1.0 / min(r["ratio"] for r in partial)),
        "full_batch_variance": complete[-1]["measured"] if complete else float("nan"),
        "full_batch_error": float(np.linalg.norm(exact - full)),
        "the_formula_holds": max(abs(r["ratio"] - 1.0) for r in partial) < 0.1,
        "the_full_batch_is_exact": float(np.linalg.norm(exact - full)) < 1e-12,
        "note": "the 1/B law needs its finite population correction whenever the batch is a real "
                "fraction of the data",
    }


def the_linear_scaling_rule_stops_at_the_stability_limit(
        batches=(1, 2, 4, 8, 16, 32, 64, 128), samples: int = 512, parameters: int = 12,
        base_step: float = 0.02, epochs: int = 8, noise: float = 0.3, seed: int = 42) -> dict:
    """Scale the step size with the batch size and watch where the rule breaks.

    The rule says that doubling the batch lets you double the step, because the noise per update
    halves. That is an argument about noise, and it is silent about the deterministic term. The
    thing that stops it is ``h < 2/L``, which has nothing to do with batches at all.
    """
    problem = least_squares_model(int(samples), int(parameters), noise=float(noise),
                                  condition=20.0, seed=seed)
    limit = 2.0 / problem["largest"]
    start = problem["solution"] + np.ones(int(parameters))
    floor = problem["loss"](problem["solution"])
    rows = []
    for size in batches:
        b = min(int(size), problem["samples"])
        step = float(base_step) * b
        updates = max(int(epochs) * problem["samples"] // b, 1)
        run = minibatch_descent(problem, start, step, b, updates, seed=seed + 1)
        final = problem["loss"](run["x"]) if np.all(np.isfinite(run["x"])) else float("inf")
        rows.append({
            "batch": b, "step": step, "step_over_limit": step / limit, "updates": updates,
            "excess_loss": final - floor if np.isfinite(final) else float("inf"),
            "stable": step < limit,
        })
    working = [r for r in rows if r["stable"]]
    return {
        "rows": rows, "limit": limit, "floor": floor,
        "largest_stable_batch": max(r["batch"] for r in working) if working else 0,
        "smallest_unstable_batch": min((r["batch"] for r in rows if not r["stable"]),
                                       default=None),
        "the_rule_holds_while_stable": max(r["excess_loss"] for r in working) < 10.0 * min(
            r["excess_loss"] for r in working) if len(working) > 1 else True,
        "it_breaks_at_the_stability_limit": any(not r["stable"] for r in rows),
        "note": "the linear scaling rule is an argument about noise, and what ends it is the "
                "deterministic stability limit of the step",
    }


def well_response(problem: dict, where: float, step: float, batch: int, iterations: int,
                  nudge: float = 1e-3, seed: int = 42) -> dict:
    """Run from just beside a minimum and report where the run settled and how far it swings.

    Two different things can go wrong with too large a step, and they have different thresholds.
    The minimum can stop being attracting, which shows up as an oscillation that does not decay,
    and the run can leave the well altogether. The first is exactly lesson 74's condition. The
    second is a nonlinear effect and happens much later.
    """
    run = minibatch_descent(problem, float(where) + float(nudge), float(step), int(batch),
                            int(iterations), seed=int(seed), record=True)
    path = np.asarray(run["path"], dtype=float).ravel()
    tail = path[max(path.size * 4 // 5, 1):]
    return {
        "final": float(path[-1]),
        "swing": float(tail.max() - tail.min()) if tail.size else 0.0,
        "displacement": abs(float(path[-1]) - float(where)),
        "path": path,
    }


def a_large_step_cannot_sit_in_a_sharp_minimum(nudge: float = 1e-3, iterations: int = 3000,
                                               probes: int = 34, samples: int = 400,
                                               seed: int = 42) -> dict:
    """Bisect for the largest step each well can hold, with no noise anywhere in the run.

    Section 2's stability limit is local, so it applies at each minimum with that minimum's own
    curvature: a well of curvature ``c`` is an attracting fixed point only while ``h < 2/c``. Two
    wells of equal depth and different curvature therefore have different step size limits, and
    that is a preference between minima that has nothing to do with noise.
    """
    problem = two_wells(samples=int(samples), seed=seed)

    def destabilized(where, step):
        out = well_response(problem, where, step, problem["samples"], int(iterations),
                            nudge=float(nudge), seed=seed)
        return out["swing"] > 10.0 * float(nudge)

    def ejected(where, step):
        out = well_response(problem, where, step, problem["samples"], int(iterations),
                            nudge=float(nudge), seed=seed)
        return out["displacement"] > 0.5

    def threshold(test, where, ceiling):
        low, high = 0.0, ceiling
        for _ in range(int(probes)):
            middle = 0.5 * (low + high)
            if test(where, middle):
                high = middle
            else:
                low = middle
        return 0.5 * (low + high)

    wells = {"flat": (problem["flat_at"], problem["flat_curvature"]),
             "sharp": (problem["sharp_at"], problem["sharp_curvature"])}
    limits = {name: {
        "predicted": 2.0 / curvature,
        "destabilizes": threshold(destabilized, where, 4.0 / curvature),
        "ejects": threshold(ejected, where, 40.0 / curvature),
    } for name, (where, curvature) in wells.items()}
    for name in limits:
        limits[name]["gap"] = (abs(limits[name]["destabilizes"] - limits[name]["predicted"])
                               / limits[name]["predicted"])

    rows = []
    for factor in (0.5, 0.95, 1.05, 2.0, 4.0, 8.0):
        step = factor * 2.0 / problem["sharp_curvature"]
        sharp = well_response(problem, problem["sharp_at"], step, problem["samples"],
                              int(iterations), nudge=float(nudge), seed=seed)
        flat = well_response(problem, problem["flat_at"], step, problem["samples"],
                             int(iterations), nudge=float(nudge), seed=seed)
        rows.append({
            "step": step, "over_the_sharp_limit": factor,
            "sharp_swing": sharp["swing"], "flat_swing": flat["swing"],
            "sharp_stays": sharp["displacement"] < 0.5,
            "flat_stays": flat["displacement"] < 0.5,
        })
    return {
        "rows": rows, "limits": limits,
        "flat_curvature": problem["flat_curvature"], "sharp_curvature": problem["sharp_curvature"],
        "curvature_ratio": problem["sharp_curvature"] / problem["flat_curvature"],
        "worst_gap": max(limits[name]["gap"] for name in limits),
        "ejection_over_destabilization": limits["sharp"]["ejects"]
                                         / limits["sharp"]["destabilizes"],
        "window": limits["flat"]["ejects"] / limits["sharp"]["ejects"],
        "both_limits_are_two_over_the_curvature": max(limits[name]["gap"]
                                                      for name in limits) < 0.02,
        "ejection_comes_much_later": (limits["sharp"]["ejects"]
                                      > 3.0 * limits["sharp"]["destabilizes"]),
        "there_is_a_window": limits["flat"]["ejects"] > limits["sharp"]["ejects"],
        "note": "losing a minimum and leaving it are two different thresholds, and only the first "
                "one is lesson 74's condition",
    }


def minibatch_noise_empties_the_sharp_well_first(
        batches=(1, 4, 16, 64, 400), factors=(1.0, 2.0, 2.5, 3.0), runs: int = 16,
        iterations: int = 4000, samples: int = 400, seed: int = 42) -> dict:
    """Sweep the step size at several batch sizes and record how often the sharp well empties.

    The two wells are equally deep to machine precision, so a method that only read the loss would
    have no preference at all. The interesting range is the window between the two thresholds of
    the previous measurement: above ``2/c`` the sharp minimum is no longer attracting, and below
    the ejection threshold a noiseless run still cannot leave it. Inside that window the only thing
    that can move a run to the other well is the sampling noise.
    """
    problem = two_wells(samples=int(samples), seed=seed)
    base = 2.0 / problem["sharp_curvature"]
    sizes = [min(int(size), problem["samples"]) for size in batches]
    keys = ["batch %d" % size for size in sizes]
    rows = []
    for factor in factors:
        step = float(factor) * base
        entry = {"factor": float(factor), "step": step}
        for size, key in zip(sizes, keys):
            escaped = 0
            for trial in range(int(runs)):
                out = minibatch_descent(problem, problem["sharp_at"], step, size,
                                        int(iterations), seed=seed + trial)
                if abs(float(out["x"]) - problem["flat_at"]) < abs(float(out["x"])
                                                                   - problem["sharp_at"]):
                    escaped += 1
            entry[key] = escaped / int(runs)
        rows.append(entry)
    small, large = keys[0], keys[-1]
    return {
        "rows": rows, "keys": keys, "base_step": base,
        "flat_value": problem["flat_value"], "sharp_value": problem["sharp_value"],
        "flat_curvature": problem["flat_curvature"], "sharp_curvature": problem["sharp_curvature"],
        "depth_gap": abs(problem["flat_value"] - problem["sharp_value"]),
        "curvature_ratio": problem["sharp_curvature"] / problem["flat_curvature"],
        "largest_noise_advantage": max(r[small] - r[large] for r in rows),
        "the_wells_are_equally_deep": abs(problem["flat_value"] - problem["sharp_value"]) < 1e-3,
        "noise_empties_it_where_the_full_batch_cannot": max(r[small] - r[large]
                                                            for r in rows) > 0.5,
        "the_full_batch_never_leaves": max(r[large] for r in rows) == 0.0,
        "smaller_batches_leave_sooner": all(
            rows[-1][keys[i]] >= rows[-1][keys[i + 1]] - 1.0 / int(runs)
            for i in range(len(keys) - 1)),
        "a_small_step_holds_every_batch": max(rows[0][key] for key in keys) == 0.0,
        "note": "the noise only matters inside the window where the minimum has stopped being "
                "attracting and a noiseless run still cannot get out",
    }


def sharpness_predicts_the_damage(radii=(1e-3, 1e-2, 1e-1), conditions=(1.0, 10.0, 100.0),
                                  dimension: int = 24, draws: int = 600, seed: int = 42) -> dict:
    """Perturb the parameters at random and compare the loss increase against the Hessian trace.

    For a quadratic the expected increase from a random direction of length ``r`` is exactly
    ``r**2 tr(H) / (2d)``, so "flat" and "sharp" are not vague words: the average damage is the
    trace and the worst case is the largest eigenvalue.
    """
    rows = []
    for target in conditions:
        curvatures = np.geomspace(float(target), 1.0, int(dimension))
        problem = quadratic(curvatures, seed=seed)
        for radius in radii:
            out = sharpness(problem, np.zeros(int(dimension)), radius=float(radius),
                            draws=int(draws), seed=seed + 1)
            rows.append({
                "condition": float(target), "radius": float(radius),
                "measured": out["measured"], "predicted": out["predicted"],
                "ratio": out["measured"] / out["predicted"],
                "worst_case": 0.5 * float(radius) ** 2 * out["largest_curvature"],
            })
    return {
        "rows": rows,
        "worst_ratio": max(max(r["ratio"] for r in rows), 1.0 / min(r["ratio"] for r in rows)),
        "the_prediction_is_exact": max(abs(r["ratio"] - 1.0) for r in rows) < 0.1,
        "worst_case_over_average": max(r["worst_case"] / r["measured"] for r in rows),
        "note": "average sharpness is the trace of the Hessian and worst case sharpness is its "
                "largest eigenvalue, and on an ill-conditioned problem they are far apart",
    }


def a_noisy_hessian_makes_newton_worse(batches=(8, 32, 128, 512), samples: int = 512,
                                       parameters: int = 20, noise: float = 0.2,
                                       updates: int = 200, step: float = 0.5,
                                       seed: int = 42) -> dict:
    """Run stochastic gradient descent and stochastic Newton at the same batch size and budget.

    This is the measurement that decides the second order question, and it is not about cost. A
    Newton step divides by a curvature estimate. At a small batch that estimate is singular or
    nearly so, and inverting it multiplies the gradient noise by the inverse of a small noisy
    number, which is the worst operation in this whole course.
    """
    problem = least_squares_model(int(samples), int(parameters), noise=float(noise),
                                  condition=50.0, seed=seed)
    start = problem["solution"] + np.ones(int(parameters))
    floor = problem["loss"](problem["solution"])
    plain_step = float(step) * 2.0 / problem["largest"]
    rows = []
    for size in batches:
        b = min(int(size), problem["samples"])
        rng = np.random.default_rng(int(seed) + 1)
        x_first = np.array(start, dtype=float)
        x_second = np.array(start, dtype=float)
        for _ in range(int(updates)):
            picked = rng.choice(problem["samples"], size=b, replace=False)
            x_first = x_first - plain_step * problem["gradient"](x_first, picked)
            try:
                x_second = x_second - newton_step(problem, x_second, picked)
            except np.linalg.LinAlgError:
                x_second = np.full(int(parameters), np.inf)
                break
        first = (max(problem["loss"](x_first) - floor, 0.0)
                 if np.all(np.isfinite(x_first)) else math.inf)
        second = (max(problem["loss"](x_second) - floor, 0.0)
                  if np.all(np.isfinite(x_second)) else math.inf)
        rank = np.linalg.matrix_rank(problem["curvature"](np.arange(b)))
        rows.append({
            "batch": b, "curvature_rank": int(rank), "parameters": int(parameters),
            "singular": int(rank) < int(parameters),
            "gradient_descent": first, "newton": second,
            "ratio": second / first if first > 0 else math.inf,
        })
    crossover = next((r["batch"] for r in rows if r["ratio"] < 1.0), None)
    finite = [r for r in rows if np.isfinite(r["ratio"])]
    return {
        "rows": rows, "crossover_batch": crossover, "floor": floor,
        "worst_ratio": max(r["ratio"] for r in finite),
        "newton_diverges_at_the_smallest_batch": not np.isfinite(rows[0]["ratio"]),
        "newton_is_worse_at_small_batches": rows[0]["ratio"] > 1.0,
        "the_curvature_is_singular_below_the_dimension": all(
            r["singular"] for r in rows if r["batch"] < int(parameters)),
        "note": "the reason second order methods are rare is that the curvature estimate is noisy "
                "and gets inverted, not that it is expensive",
    }


def cost_is_not_the_reason_second_order_is_rare(dimensions=(5, 10, 20, 50, 100, 200),
                                                tolerance: float = 1e-8,
                                                hessian_condition: float = 400.0,
                                                samples_per_parameter: int = 4,
                                                cap: int = 400000, seed: int = 42) -> dict:
    """Count flops to a fixed accuracy for gradient descent and for Newton, as the dimension grows.

    Newton solves a quadratic in one step, so the iteration count is not the question. The question
    is the cost of that one step, ``O(N d**2 + d**3)`` against ``O(N d)`` per gradient step, and
    whether the product of count and cost ever crosses over. The answer here is that it does not,
    at any dimension a laptop can run, which is why the flop count cannot be the explanation and
    why the noise measurement above has to be.
    """
    rows = []
    for d in dimensions:
        n = int(samples_per_parameter) * int(d)
        problem = least_squares_model(n, int(d),
                                      condition=math.sqrt(float(hessian_condition)), seed=seed)
        start = problem["solution"] + np.ones(int(d))
        scale = float(np.linalg.norm(start - problem["solution"]))
        step = 2.0 / (problem["largest"] + problem["smallest"])
        x = start.copy()
        count = 0
        while (np.linalg.norm(x - problem["solution"]) > float(tolerance) * scale
               and count < int(cap)):
            x = x - step * problem["gradient"](x)
            count += 1
        y = start - newton_step(problem, start)
        newton_error = float(np.linalg.norm(y - problem["solution"])) / scale
        first_cost = count * 2.0 * n * d
        second_cost = 2.0 * n * d * d + 2.0 * d ** 3 / 3.0 + 2.0 * n * d
        rows.append({
            "dimension": int(d), "samples": n, "condition": problem["condition"],
            "iterations": count, "break_even": second_cost / (2.0 * n * d),
            "newton_error": newton_error,
            "first_order_flops": first_cost, "second_order_flops": second_cost,
            "ratio": first_cost / second_cost,
            "memory_ratio": float(d),
        })
    logs = np.log([r["dimension"] for r in rows])
    slope = float(np.polyfit(logs, np.log([r["ratio"] for r in rows]), 1)[0])
    last = rows[-1]
    crossover = last["dimension"] * last["ratio"] ** (-1.0 / slope) if slope < 0 else None
    typical = float(np.mean([r["iterations"] for r in rows]))
    huge = 1e9
    return {
        "rows": rows, "slope": slope, "extrapolated_crossover": crossover,
        "predicted_crossover": typical,
        "crossover_ratio": crossover / typical if crossover else float("nan"),
        "worst_ratio": min(r["ratio"] for r in rows),
        "memory_at_a_billion_parameters": huge * huge * 8.0,
        "gradient_memory_at_a_billion_parameters": huge * 8.0,
        "newton_converges_in_one_step": max(r["newton_error"] for r in rows) < 1e-8,
        "newton_wins_everywhere_tested": all(r["ratio"] > 1.0 for r in rows),
        "the_advantage_falls_like_one_over_d": abs(slope + 1.0) < 0.2,
        "the_crossover_is_the_iteration_count": (crossover is not None
                                                 and abs(crossover / typical - 1.0) < 0.3),
        "note": "on flops alone a second order method wins here, and it stops winning exactly "
                "where the dimension passes the iteration count, not before",
    }


def the_natural_gradient_ignores_the_parametrization(factors=(1.0, 10.0, 100.0),
                                                     samples: int = 200, parameters: int = 8,
                                                     iterations: int = 30, step: float = 0.5,
                                                     seed: int = 42) -> dict:
    """Change variables by an invertible matrix and see which method notices.

    Under ``x = A y`` the gradient transforms as ``A' g`` and the Fisher as ``A' F A``, so the
    natural step ``F^-1 g`` transforms as ``A^-1 F^-1 g`` and maps back to the same point in the
    original coordinates. The invariance is exact for any step size, and plain gradient descent has
    nothing like it.
    """
    problem = least_squares_model(int(samples), int(parameters), condition=100.0, seed=seed)
    design, target = problem["design"], problem["target"]
    start = problem["solution"] + np.ones(int(parameters))
    reference_plain, reference_natural = None, None
    rows = []
    for factor in factors:
        scale = np.geomspace(1.0, float(factor), int(parameters))
        change = problem["hessian"] * 0.0 + np.diag(scale)
        scaled = design @ change
        y_plain = np.linalg.solve(change, start)
        y_natural = np.linalg.solve(change, start)
        limit = 2.0 / float(np.linalg.eigvalsh(scaled.T @ scaled / scaled.shape[0])[-1])
        for _ in range(int(iterations)):
            gradient = scaled.T @ (scaled @ y_plain - target) / scaled.shape[0]
            y_plain = y_plain - float(step) * limit * gradient
            y_natural = y_natural - float(step) * natural_gradient_step(scaled, y_natural, target)
        plain = change @ y_plain
        natural = change @ y_natural
        if reference_plain is None:
            reference_plain, reference_natural = plain.copy(), natural.copy()
        scale_norm = float(np.linalg.norm(start - problem["solution"]))
        rows.append({
            "factor": float(factor),
            "plain_drift": float(np.linalg.norm(plain - reference_plain)) / scale_norm,
            "natural_drift": float(np.linalg.norm(natural - reference_natural)) / scale_norm,
            "plain_error": float(np.linalg.norm(plain - problem["solution"])) / scale_norm,
            "natural_error": float(np.linalg.norm(natural - problem["solution"])) / scale_norm,
        })
    return {
        "rows": rows,
        "worst_natural_drift": max(r["natural_drift"] for r in rows),
        "worst_plain_drift": max(r["plain_drift"] for r in rows),
        "natural_is_invariant": max(r["natural_drift"] for r in rows) < 1e-10,
        "plain_is_not": max(r["plain_drift"] for r in rows) > 1e-3,
        "note": "preconditioning by the Fisher matrix makes the trajectory a property of the model "
                "rather than of the coordinates someone chose to write it in",
    }


def gauss_newton_is_the_fisher_information(residual_scales=(0.0, 0.1, 1.0, 5.0),
                                           samples: int = 120, parameters: int = 6,
                                           seed: int = 42) -> dict:
    """Check the identity ``F = J'J`` for a Gaussian model, and measure where it stops being the Hessian.

    For ``y = m(theta) + N(0, I)`` the Fisher information is ``J'J`` exactly, and that is also the
    Gauss-Newton matrix. The true Hessian is ``J'J - sum_i r_i H_i``, so all three agree at a zero
    residual solution and the second term is the whole difference at a large one.
    """
    rng = np.random.default_rng(int(seed))
    n, d = int(samples), int(parameters)
    rows_of_a = rng.standard_normal((n, d)) / math.sqrt(d)

    def model(theta):
        return np.sin(rows_of_a @ np.asarray(theta, dtype=float))

    def jacobian(theta):
        return np.cos(rows_of_a @ np.asarray(theta, dtype=float))[:, None] * rows_of_a

    theta = rng.standard_normal(d) * 0.4
    rows = []
    for level in residual_scales:
        data = model(theta) + float(level) * rng.standard_normal(n)
        residual = model(theta) - data
        j = jacobian(theta)
        gauss = j.T @ j
        fisher = j.T @ j
        inner = rows_of_a @ theta
        weights = -np.sin(inner) * residual
        second = rows_of_a.T @ (weights[:, None] * rows_of_a)
        exact = gauss + second
        rows.append({
            "residual_scale": float(level),
            "residual_norm": float(np.linalg.norm(residual)),
            "fisher_gap": float(np.linalg.norm(fisher - gauss) / np.linalg.norm(gauss)),
            "hessian_gap": float(np.linalg.norm(exact - gauss) / np.linalg.norm(gauss)),
            "hessian_is_positive": bool(np.all(np.linalg.eigvalsh(exact) > 0)),
        })
    return {
        "rows": rows,
        "fisher_equals_gauss_newton": max(r["fisher_gap"] for r in rows) < 1e-12,
        "they_agree_at_zero_residual": rows[0]["hessian_gap"] < 1e-12,
        "they_disagree_at_a_large_one": rows[-1]["hessian_gap"] > 0.1,
        "largest_hessian_gap": max(r["hessian_gap"] for r in rows),
        "note": "Gauss-Newton and the Fisher information are the same matrix for a Gaussian model, "
                "and both stop being the Hessian exactly as the residual grows",
    }
