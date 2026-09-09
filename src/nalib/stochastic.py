"""Optimization when the gradient is a sample: SGD, momentum, and the adaptive methods.

Where the work is
-----------------
Lessons 84 to 88 all assumed ``grad f`` could be computed. In a great many modern problems the
objective is a sum over ``N`` terms with ``N`` in the millions, one gradient costs ``N`` evaluations,
and only a **sample** of it is affordable:

    f(x) = (1/N) sum_i f_i(x) ,   and a step uses a batch of size b << N .

That single change breaks the theory of the previous four lessons. A descent method needs a descent
direction, and a sampled gradient is not one: it is a random vector whose mean is the gradient. So
the line searches of lesson 86 are unusable, the curvature pairs of lesson 87 are noise, and the
first thing the new theory says is that **a constant step size does not converge**.

What replaces it is a step size that shrinks, momentum to recover some of the lost speed, and
diagonal rescaling to cope with badly scaled coordinates.

What the measurements here show
-------------------------------
* **A constant step does not converge, it settles into a noise ball**, and the radius of that ball
  grows like ``sqrt(step)``: the fitted power is ``0.48`` against a predicted ``0.5`` over three
  decades of step size. Halving the step buys a factor of ``1.4`` in accuracy and doubles the work.
* **Both Robbins-Monro conditions bite, and each has a failing case here.** A constant step has
  ``sum a**2 = infinity`` and stalls at ``3.5e-02``. A step of ``0.5/k**1.5`` has
  ``sum a = 1.3 < infinity`` and stops moving at ``3.0e-01``, a hundred times worse, because the
  steps shrink faster than the distance does. ``0.5/k`` satisfies both and reaches ``1.7e-03``.
* Gradient noise falls like ``1/sqrt(b)``: the fitted power is ``-0.484`` and the noise times
  ``sqrt(b)`` settles at ``4.86, 5.08, 5.11, 5.12, 5.10`` from ``b = 4`` to ``1024``, a spread of
  five per cent. **Sixteen times the work buys four times the accuracy**, which is the arithmetic
  behind every batch size decision.
* **Momentum turns ``kappa`` into ``sqrt(kappa)``.** On quadratics the measured gradient descent
  rate matches ``(kappa-1)/(kappa+1)`` to six digits and heavy ball matches
  ``(sqrt(kappa)-1)/(sqrt(kappa)+1)`` to within one per cent. At ``kappa = 1000`` that is ``0.9980``
  against ``0.9413``, which is 3450 steps against 95.
* On Nesterov's worst function the accelerated method's decay exponent is **exactly twice**
  gradient descent's, ``-1.02`` against ``-0.50``. Neither is the ``-1`` and ``-2`` usually quoted,
  because those are worst case bounds over a class and this is one instance: the ratio is the part
  that transfers.
* The adaptive methods are diagonal preconditioners, and the measurement says where that matters.
  Given a step chosen without looking at the problem, plain SGD and momentum **diverge** on the
  badly scaled one and every adaptive method converges. Given the best step a plain method can have,
  ``1/L``, they converge and are still beaten, and the margin is what grows with the scaling.
* Adam and AdamW are the same algorithm with no weight decay and agree **exactly**, bit for bit.
  With decay they differ, and the difference is spread unevenly across the coordinates, which is
  the whole content of the decoupling.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import functools
import math

import numpy as np


# --------------------------------------------------------------------------- the problem


def least_squares_problem(samples: int = 2000, variables: int = 10,
                          noise: float = 0.5, scaling: float = 1.0,
                          seed: int = 42):
    """``f(x) = (1/N) ||A x - b||**2``, a finite sum whose minimizer is known exactly.

    ``scaling`` stretches the columns of ``A`` geometrically, so the Hessian's condition number can
    be dialled without changing anything else. That is what separates the two regimes the adaptive
    methods live between: ``scaling = 1`` is well scaled, ``scaling = 1000`` is not.
    """
    n, m = int(samples), int(variables)
    if n < 1 or m < 1:
        raise ValueError(f"need at least one sample and one variable, got {n} and {m}")
    rng = np.random.default_rng(int(seed))
    design = rng.normal(size=(n, m))
    if float(scaling) != 1.0:
        design = design * np.logspace(0.0, math.log10(float(scaling)), m)
    truth = rng.normal(size=m)
    targets = design @ truth + float(noise) * rng.normal(size=n)
    answer, *_ = np.linalg.lstsq(design, targets, rcond=None)
    hessian = 2.0 * (design.T @ design) / n
    values = np.linalg.eigvalsh(hessian)

    def f(x):
        gap = design @ np.asarray(x, dtype=float).ravel() - targets
        return float(gap @ gap) / n

    def gradient(x):
        gap = design @ np.asarray(x, dtype=float).ravel() - targets
        return 2.0 * (design.T @ gap) / n

    def batch_gradient(x, rows):
        picked = np.asarray(rows, dtype=int)
        part = design[picked]
        gap = part @ np.asarray(x, dtype=float).ravel() - targets[picked]
        return 2.0 * (part.T @ gap) / picked.size

    return {"f": f, "gradient": gradient, "batch_gradient": batch_gradient,
            "design": design, "targets": targets, "samples": n, "dimension": m,
            "minimizer": answer, "minimum": f(answer), "hessian": hessian,
            "condition": float(values[-1] / values[0]),
            "smoothness": float(values[-1]), "strong_convexity": float(values[0]),
            "start": np.zeros(m),
            "name": f"least squares, {n} samples, {m} variables, scaling {scaling:g}"}


# --------------------------------------------------------------------------- the methods


def schedule(name: str, base: float):
    """Turn a name and a base step into a function of the iteration number.

    The Robbins-Monro conditions are ``sum a_k = infinity`` and ``sum a_k**2 < infinity``. The first
    says the steps must be able to travel any distance; the second that the noise they let in must
    be summable. ``1/k`` satisfies both, ``1/sqrt(k)`` fails the second, a constant fails the second
    badly, and ``1/k**1.5`` fails the **first**, which is the failure nobody expects.
    """
    a = float(base)
    rules = {
        "constant": lambda k: a,
        "one over k": lambda k: a / (k + 1.0),
        "one over root k": lambda k: a / math.sqrt(k + 1.0),
        "one over k to the 1.5": lambda k: a / (k + 1.0) ** 1.5,
    }
    if name not in rules:
        raise ValueError(f"unknown schedule {name!r}, expected one of {sorted(rules)}")
    return rules[name]


def cosine_schedule(base: float, rounds: int, warmup: int = 0):
    """Linear warmup then a cosine decay to zero, which is what large models actually use."""
    a, total, heat = float(base), int(rounds), int(warmup)

    def rule(k):
        if heat and k < heat:
            return a * (k + 1.0) / heat
        progress = (k - heat) / max(total - heat, 1)
        return 0.5 * a * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return rule


def stochastic_descent(problem, method: str = "sgd", step: float = 0.01,
                       batch: int = 16, rounds: int = 20000,
                       rule=None, momentum: float = 0.9, decay: float = 0.999,
                       weight_decay: float = 0.0, epsilon: float = 1e-8,
                       seed: int = 0, record_every: int = 100) -> dict:
    """One driver for SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam and AdamW.

    They differ in two places only: what is accumulated, and what the accumulated quantity is used
    for. Momentum accumulates the **gradient** and uses it as the direction. The adaptive methods
    accumulate the **squared** gradient and use it to rescale each coordinate, which makes them
    diagonal preconditioners. Adam does both.

    AdamW differs from Adam in one line. Adam with L2 regularization adds ``wd * x`` to the gradient
    and then divides by the accumulated scale, so the amount of decay a coordinate receives depends
    on its gradient history. AdamW subtracts ``step * wd * x`` from the iterate directly, outside
    the rescaling, which is the same thing only when the scaling is uniform.
    """
    x = np.asarray(problem["start"], dtype=float).ravel().copy()
    n = x.size
    rng = np.random.default_rng(int(seed))
    velocity = np.zeros(n)
    squares = np.zeros(n)
    steps_taken, distances, values = [], [], []
    star = problem["minimizer"]
    picker = rule if rule is not None else schedule("constant", step)
    # a run that blows up is one of the outcomes this module measures, so overflow is a result to
    # report rather than a warning to print: the caller sees it in the returned distance
    errors = np.errstate(over="ignore", invalid="ignore", divide="ignore")
    errors.__enter__()
    for k in range(int(rounds)):
        if k % int(record_every) == 0:
            steps_taken.append(k)
            distances.append(float(np.linalg.norm(x - star)))
            values.append(float(problem["f"](x)))
        alpha = float(picker(k))
        rows = rng.integers(0, problem["samples"], size=int(batch))
        if method == "nesterov":
            probe = x + momentum * velocity
            slope = np.asarray(problem["batch_gradient"](probe, rows), dtype=float)
        else:
            slope = np.asarray(problem["batch_gradient"](x, rows), dtype=float)
        if weight_decay and method != "adamw":
            slope = slope + weight_decay * x
        if method in ("sgd",):
            x = x - alpha * slope
        elif method in ("momentum", "nesterov"):
            velocity = momentum * velocity - alpha * slope
            x = x + velocity
        elif method == "adagrad":
            squares = squares + slope * slope
            x = x - alpha * slope / (np.sqrt(squares) + epsilon)
        elif method == "rmsprop":
            squares = decay * squares + (1.0 - decay) * slope * slope
            x = x - alpha * slope / (np.sqrt(squares) + epsilon)
        elif method in ("adam", "adamw"):
            velocity = momentum * velocity + (1.0 - momentum) * slope
            squares = decay * squares + (1.0 - decay) * slope * slope
            first = velocity / (1.0 - momentum ** (k + 1))
            second = squares / (1.0 - decay ** (k + 1))
            move = alpha * first / (np.sqrt(second) + epsilon)
            if method == "adamw" and weight_decay:
                move = move + alpha * weight_decay * x
            x = x - move
        else:
            errors.__exit__(None, None, None)
            raise ValueError(f"unknown method {method!r}")
    errors.__exit__(None, None, None)
    return {"x": x, "method": method, "steps": np.asarray(steps_taken),
            "distances": np.asarray(distances), "values": np.asarray(values),
            "final_distance": float(np.linalg.norm(x - star)),
            "final_value": float(problem["f"](x)),
            "gap": float(problem["f"](x)) - problem["minimum"]}


def _tail_distance(problem, method: str, fraction: float = 0.25, **options) -> float:
    """The mean distance over the last ``fraction`` of a run, which is what a noise ball needs."""
    out = stochastic_descent(problem, method=method, **options)
    keep = out["distances"][int((1.0 - fraction) * out["distances"].size):]
    return float(np.mean(keep))


# --------------------------------------------------------------------------- measurements


@functools.lru_cache(maxsize=None)
def the_noise_ball_grows_like_the_square_root_of_the_step(
        steps=(1e-3, 3e-3, 1e-2, 3e-2), batch: int = 16, rounds: int = 20000) -> dict:
    """A constant step does not converge. Measure what it does instead.

    Once the iterate is near the answer the gradient is small and the sampling noise is not, so the
    step is mostly noise and the iterate random walks. The walk balances against the pull of the
    gradient at a radius where the two are comparable, and the theory says the **squared** radius is
    proportional to the step, so the radius itself grows like ``sqrt(step)``.
    """
    problem = least_squares_problem()
    rows = []
    for alpha in steps:
        mean = _tail_distance(problem, "sgd", step=float(alpha), batch=int(batch),
                              rounds=int(rounds), rule=schedule("constant", float(alpha)))
        rows.append({"step": float(alpha), "tail_distance": mean,
                     "over_root_step": mean / math.sqrt(float(alpha))})
    sizes = np.asarray([r["step"] for r in rows])
    radii = np.asarray([r["tail_distance"] for r in rows])
    power = float(np.polyfit(np.log(sizes), np.log(radii), 1)[0])
    return {
        "rows": rows,
        "fitted_power_of_the_step": power,
        "the_power_is_a_half": abs(power - 0.5) < 0.08,
        "spread_in_the_constant": float(np.max([r["over_root_step"] for r in rows])
                                        / np.min([r["over_root_step"] for r in rows])),
        "note": "the radius grows like the square root of the step, so halving the step buys a "
                "factor of 1.4 in accuracy and doubles the number of steps to get anywhere",
    }


@functools.lru_cache(maxsize=None)
def the_robbins_monro_conditions_decide_convergence(rounds: int = 40000,
                                                    batch: int = 16) -> dict:
    """Four schedules, and each condition has a case that fails it.

    ``sum a_k = infinity`` says the steps can still travel: without it the iterate stops before
    arriving, however much time it is given. ``sum a_k**2 < infinity`` says the noise they admit is
    summable: without it the iterate keeps being kicked and never settles.

    The second failure is the famous one. The first is the one that surprises people, because a
    schedule that decays faster looks more careful and is in fact broken.
    """
    problem = least_squares_problem()
    rows = []
    for name, base in (("constant", 0.01), ("one over k", 0.5),
                       ("one over root k", 0.1), ("one over k to the 1.5", 0.5)):
        rule = schedule(name, base)
        first = sum(rule(k) for k in range(int(rounds)))
        second = sum(rule(k) ** 2 for k in range(int(rounds)))
        out = stochastic_descent(problem, method="sgd", batch=int(batch),
                                 rounds=int(rounds), rule=rule)
        rows.append({
            "schedule": name, "base": float(base),
            "sum_of_steps": first, "sum_of_squares": second,
            "final_distance": out["final_distance"],
        })
    best = min(rows, key=lambda r: r["final_distance"])
    worst = max(rows, key=lambda r: r["final_distance"])
    return {
        "rows": rows,
        "best_schedule": best["schedule"],
        "worst_schedule": worst["schedule"],
        "the_best_satisfies_both": best["schedule"] == "one over k",
        "the_worst_decays_too_fast": worst["schedule"] == "one over k to the 1.5",
        "spread": worst["final_distance"] / best["final_distance"],
        "note": "the schedule that decays fastest is the worst of the four, by a factor of a "
                "hundred, because its steps add up to a finite distance and the iterate simply "
                "stops before it arrives",
    }


@functools.lru_cache(maxsize=None)
def the_noise_falls_like_one_over_root_batch(batches=(1, 4, 16, 64, 256, 1024),
                                             trials: int = 400, seed: int = 1) -> dict:
    """Measure the spread of the sampled gradient around the true one, against the batch size.

    This is the only quantity in the lesson that has nothing to do with any algorithm: it is a
    property of sampling. Everything else in stochastic optimization is a response to it.
    """
    problem = least_squares_problem()
    rng = np.random.default_rng(int(seed))
    where = problem["minimizer"] + 0.3 * rng.normal(size=problem["dimension"])
    exact = np.asarray(problem["gradient"](where), dtype=float)
    rows = []
    for batch in batches:
        gaps = []
        for _ in range(int(trials)):
            rows_picked = rng.integers(0, problem["samples"], size=int(batch))
            gaps.append(float(np.linalg.norm(
                problem["batch_gradient"](where, rows_picked) - exact)))
        mean = float(np.mean(gaps))
        rows.append({"batch": int(batch), "noise": mean,
                     "times_root_batch": mean * math.sqrt(batch)})
    sizes = np.asarray([r["batch"] for r in rows], dtype=float)
    noises = np.asarray([r["noise"] for r in rows])
    power = float(np.polyfit(np.log(sizes), np.log(noises), 1)[0])
    settled = [r["times_root_batch"] for r in rows if r["batch"] >= 4]
    return {
        "rows": rows,
        "fitted_power_of_the_batch": power,
        "the_power_is_minus_a_half": abs(power + 0.5) < 0.05,
        "constant_above_four": float(np.max(settled) / np.min(settled)),
        "note": "sixteen times the work buys four times the accuracy, which is why batch sizes are "
                "chosen by hardware and not by this exponent",
    }


@functools.lru_cache(maxsize=None)
def momentum_turns_the_condition_number_into_its_square_root(
        conditions=(10.0, 100.0, 1000.0), dimension: int = 10) -> dict:
    """Heavy ball against plain descent on quadratics, with the optimal parameters for each.

    This measurement uses the **exact** gradient, not a sampled one, because the point being made
    is about the rate and not about the noise. Momentum was invented for the deterministic problem
    and carried over to the stochastic one, and separating the two is the only way to see what each
    contributes.

    The optimal heavy ball parameters are known in closed form for a quadratic,
    ``beta = ((sqrt(L) - sqrt(mu))/(sqrt(L) + sqrt(mu)))**2`` and
    ``alpha = 4/(sqrt(L) + sqrt(mu))**2``, and they give the rate
    ``(sqrt(kappa) - 1)/(sqrt(kappa) + 1)``, the same square root conjugate gradients gets in
    lesson 24.
    """
    from .optimize import quadratic

    rows = []
    for kappa in conditions:
        problem = quadratic(condition=float(kappa), dimension=int(dimension))
        matrix, centre = problem["matrix"], problem["minimizers"][0]
        values, vectors = np.linalg.eigh(matrix)
        top, bottom = float(values[-1]), float(values[0])
        start = centre + vectors @ (1.0 / np.sqrt(values))

        plain, previous = start.copy(), start.copy()
        errors = []
        pace = 2.0 / (top + bottom)
        for _ in range(20000):
            gap = plain - centre
            errors.append(float(np.linalg.norm(gap)))
            if errors[-1] <= 1e-12 * errors[0]:
                break
            plain = plain - pace * (matrix @ gap)
        errors = np.asarray(errors)
        late = errors[errors.size // 2:]
        plain_rate = float(np.mean(late[1:] / late[:-1]))

        beta = ((math.sqrt(top) - math.sqrt(bottom))
                / (math.sqrt(top) + math.sqrt(bottom))) ** 2
        alpha = 4.0 / (math.sqrt(top) + math.sqrt(bottom)) ** 2
        heavy, previous = start.copy(), start.copy()
        errors2 = []
        for _ in range(20000):
            gap = heavy - centre
            errors2.append(float(np.linalg.norm(gap)))
            if errors2[-1] <= 1e-12 * errors2[0]:
                break
            ahead = heavy - alpha * (matrix @ gap) + beta * (heavy - previous)
            previous, heavy = heavy, ahead
        errors2 = np.asarray(errors2)
        late2 = errors2[errors2.size // 2:]
        heavy_rate = float(np.mean(late2[1:] / late2[:-1]))

        root = math.sqrt(kappa)
        rows.append({
            "condition": float(kappa),
            "plain_steps": int(errors.size), "plain_rate": plain_rate,
            "plain_bound": (kappa - 1.0) / (kappa + 1.0),
            "heavy_steps": int(errors2.size), "heavy_rate": heavy_rate,
            "heavy_bound": (root - 1.0) / (root + 1.0),
            "speedup": int(errors.size) / max(int(errors2.size), 1),
        })
    return {
        "rows": rows,
        "plain_matches_its_bound": all(
            abs(r["plain_rate"] / r["plain_bound"] - 1.0) < 1e-3 for r in rows),
        "heavy_matches_its_bound": all(
            abs(r["heavy_rate"] / r["heavy_bound"] - 1.0) < 0.05 for r in rows),
        "biggest_speedup": max(r["speedup"] for r in rows),
        "note": "the same square root conjugate gradients gets in lesson 24, from a method with "
                "one extra vector of storage and no line search",
    }


@functools.lru_cache(maxsize=None)
def acceleration_doubles_the_exponent(variables: int = 4000, budget: int = 800,
                                      smoothness: float = 1.0) -> dict:
    """Nesterov's worst function, where the accelerated method is provably as good as possible.

    The function is built so that after ``k`` steps only the first ``k`` coordinates have moved, so
    with ``n`` far larger than the budget no first order method can exploit the finite dimension.

    The measured exponents are ``-0.50`` and ``-1.02``, not the ``-1`` and ``-2`` usually quoted.
    Those quoted values are worst case **upper bounds over a class of functions**, and this is a
    measurement on one instance, so they are not the same statement. What does transfer is the
    **ratio**, and it comes out at 2.
    """
    n, steps, L = int(variables), int(budget), float(smoothness)
    if n <= 2 * steps:
        raise ValueError("the dimension must exceed twice the budget for this construction")

    def gradient(x):
        out = np.zeros_like(x)
        out[0] = 2.0 * x[0] - x[1] - 1.0
        out[1:-1] = 2.0 * x[1:-1] - x[:-2] - x[2:]
        out[-1] = 2.0 * x[-1] - x[-2]
        return (L / 4.0) * out

    def f(x):
        gaps = np.diff(x)
        return (L / 8.0) * (x[0] ** 2 + float(gaps @ gaps) + x[-1] ** 2) - (L / 4.0) * x[0]

    index = np.arange(1, n + 1)
    answer = 1.0 - index / (n + 1.0)
    best = f(answer)
    rows = []
    for name in ("gradient descent", "Nesterov"):
        x = np.zeros(n)
        y = np.zeros(n)
        t = 1.0
        gaps = []
        for _ in range(steps + 1):
            gaps.append(f(x) - best)
            if name == "gradient descent":
                x = x - (1.0 / L) * gradient(x)
            else:
                ahead = y - (1.0 / L) * gradient(y)
                nt = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
                y = ahead + ((t - 1.0) / nt) * (ahead - x)
                x, t = ahead, nt
        gaps = np.asarray(gaps)
        counts = np.arange(gaps.size)
        keep = (counts >= 50) & (gaps > 0.0)
        rows.append({
            "method": name,
            "exponent": float(np.polyfit(np.log(counts[keep]), np.log(gaps[keep]), 1)[0]),
            "gap_at_100": float(gaps[100]),
            "gap_at_the_end": float(gaps[-1]),
        })
    ratio = rows[1]["exponent"] / rows[0]["exponent"]
    return {
        "rows": rows,
        "variables": n, "budget": steps,
        "exponent_ratio": ratio,
        "the_ratio_is_two": abs(ratio - 2.0) < 0.1,
        "neither_matches_the_quoted_bound": all(
            abs(r["exponent"] + k) > 0.2 for r, k in zip(rows, (1.0, 2.0))),
        "accuracy_gain_at_the_end": rows[0]["gap_at_the_end"] / rows[1]["gap_at_the_end"],
        "note": "the ratio of exponents is what a measurement on one instance can establish; the "
                "quoted -1 and -2 are worst case bounds over a class, and comparing the two "
                "directly would be comparing different statements",
    }


@functools.lru_cache(maxsize=None)
def the_adaptive_methods_rescale_the_coordinates(
        methods=("sgd", "momentum", "adagrad", "rmsprop", "adam"),
        scalings=(1.0, 1000.0), rounds: int = 20000, batch: int = 16) -> dict:
    """The same five methods on a well scaled problem and a badly scaled one, twice over.

    Dividing each coordinate by the square root of its accumulated squared gradient is a diagonal
    preconditioner built from the data, so the honest comparison has to be careful about the step
    size: handing a plain method a step that suits the other problem would be measuring the step
    and calling it the method.

    So each non-adaptive method is run twice: once with a **fixed** step of 0.01, which is what
    someone who has not looked at the problem would use, and once with ``1/L`` from the measured
    smoothness, which is the best a plain method can be given. The adaptive methods get their usual
    0.05 in both cases, because choosing their step is precisely what they are supposed to make
    unnecessary.
    """
    rows = []
    for scaling in scalings:
        problem = least_squares_problem(scaling=float(scaling))
        safe = 1.0 / problem["smoothness"]
        for method in methods:
            adaptive = method in ("adagrad", "rmsprop", "adam", "adamw")
            choices = ([("adaptive default", 0.05)] if adaptive
                       else [("fixed 0.01", 0.01), ("one over L", safe)])
            for label, step in choices:
                out = stochastic_descent(problem, method=method, step=float(step),
                                         batch=int(batch), rounds=int(rounds),
                                         rule=schedule("constant", float(step)))
                finite = math.isfinite(out["final_distance"])
                rows.append({
                    "scaling": float(scaling),
                    "condition": problem["condition"],
                    "method": method,
                    "step_rule": label,
                    "step": float(step),
                    "final_distance": out["final_distance"] if finite else float("inf"),
                    "diverged": not finite or out["final_distance"] > 1e3,
                })
    best = {}
    for scaling in scalings:
        mine = [r for r in rows if r["scaling"] == scaling and not r["diverged"]]
        best[float(scaling)] = min(mine, key=lambda r: r["final_distance"])
    hard = max(scalings)
    easy = min(scalings)
    blown = [r for r in rows if r["diverged"]]
    return {
        "rows": rows,
        "best_when_well_scaled": best[float(easy)]["method"],
        "best_when_badly_scaled": best[float(hard)]["method"],
        "methods_that_diverged": sorted({r["method"] for r in blown}),
        "everything_that_diverged_was_non_adaptive": all(
            r["method"] in ("sgd", "momentum") for r in blown),
        "adaptive_wins_when_badly_scaled": best[float(hard)]["method"] in (
            "adagrad", "rmsprop", "adam", "adamw"),
        "margin_when_badly_scaled": (
            min(r["final_distance"] for r in rows
                if r["scaling"] == hard and not r["diverged"]
                and r["method"] in ("sgd", "momentum"))
            / best[float(hard)]["final_distance"]),
        "margin_when_well_scaled": (
            min(r["final_distance"] for r in rows
                if r["scaling"] == easy and not r["diverged"]
                and r["method"] in ("sgd", "momentum"))
            / best[float(easy)]["final_distance"]),
        "note": "with a step chosen without looking at the problem the plain methods blow up on "
                "the badly scaled one and the adaptive methods do not; given the right step the "
                "plain methods work and are still beaten. The margin is what changes with the "
                "scaling, not the ranking",
    }


@functools.lru_cache(maxsize=None)
def adam_and_adamw_differ(weight_decay: float = 0.05, rounds: int = 20000,
                          scaling: float = 100.0, batch: int = 16) -> dict:
    """The one line difference, and when it matters.

    Adam with L2 regularization folds ``wd * x`` into the gradient, so it then passes through the
    per coordinate rescaling: a coordinate with a large gradient history gets **less** decay. AdamW
    subtracts ``step * wd * x`` from the iterate directly, so every coordinate decays at the same
    rate whatever its history.

    Two claims, and both are checked. With no decay the two are the **same algorithm**, so they must
    agree to the last bit and a difference would be a bug. With decay they produce different
    iterates, and the measurement reports the distance between them rather than a proxy, because
    which of the two is better depends on what the decay was for.
    """
    problem = least_squares_problem(scaling=float(scaling))
    answers, rows = {}, []
    for method in ("adam", "adamw"):
        for decay_value in (0.0, float(weight_decay)):
            out = stochastic_descent(problem, method=method, step=0.05,
                                     batch=int(batch), rounds=int(rounds),
                                     weight_decay=decay_value,
                                     rule=schedule("constant", 0.05))
            answers[(method, decay_value)] = out["x"]
            rows.append({
                "method": method, "weight_decay": decay_value,
                "gap": out["gap"],
                "norm_of_the_answer": float(np.linalg.norm(out["x"])),
                "final_distance": out["final_distance"],
            })
    same = float(np.max(np.abs(answers[("adam", 0.0)] - answers[("adamw", 0.0)])))
    apart = float(np.linalg.norm(answers[("adam", float(weight_decay))]
                                 - answers[("adamw", float(weight_decay))]))
    spread = np.abs(answers[("adam", float(weight_decay))]
                    - answers[("adamw", float(weight_decay))])
    return {
        "rows": rows,
        "condition": problem["condition"],
        "difference_without_decay": same,
        "identical_without_decay": same == 0.0,
        "distance_with_decay": apart,
        "different_with_decay": apart > 1e-6,
        "largest_coordinate_difference": float(np.max(spread)),
        "smallest_coordinate_difference": float(np.min(spread)),
        "the_difference_is_uneven_across_coordinates": float(
            np.max(spread) / max(float(np.min(spread)), 1e-300)) > 10.0,
        "note": "with no decay the two are the same algorithm and agree exactly, bit for bit; with "
                "decay they differ, and the difference is spread unevenly across the coordinates, "
                "which is the point: the coupling is to the per coordinate scaling",
    }


@functools.lru_cache(maxsize=None)
def which_schedule(rounds: int = 20000, batch: int = 16, base: float = 0.05) -> dict:
    """Five schedules on the same problem from the same start, compared on the same budget.

    The comparison is deliberately at a fixed number of steps rather than to a fixed accuracy,
    because that is the situation these schedules exist for: the budget is decided in advance and
    the question is how to spend it.
    """
    problem = least_squares_problem()
    rows = []
    named = [("constant", schedule("constant", base)),
             ("one over k", schedule("one over k", base * 10.0)),
             ("one over root k", schedule("one over root k", base)),
             ("cosine", cosine_schedule(base, int(rounds))),
             ("cosine with warmup", cosine_schedule(base, int(rounds),
                                                    warmup=int(rounds) // 20))]
    for name, rule in named:
        out = stochastic_descent(problem, method="sgd", batch=int(batch),
                                 rounds=int(rounds), rule=rule)
        rows.append({"schedule": name, "gap": out["gap"],
                     "final_distance": out["final_distance"]})
    best = min(rows, key=lambda r: r["final_distance"])
    constant = [r for r in rows if r["schedule"] == "constant"][0]
    return {
        "rows": rows,
        "best": best["schedule"],
        "constant_is_worst": constant["final_distance"] == max(
            r["final_distance"] for r in rows),
        "best_over_constant": constant["final_distance"] / best["final_distance"],
        "note": "any schedule that decays beats the constant one at a fixed budget, and the "
                "differences between the decaying ones are much smaller than the difference "
                "between decaying and not",
    }
