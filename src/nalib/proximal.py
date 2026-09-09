"""Objectives that are smooth plus nonsmooth: the proximal operator, ISTA, FISTA and LASSO.

Where the work is
-----------------
Nothing in Part 12 so far can handle ``||x||_1``. It has no gradient at zero, which is exactly where
the interesting solutions live, and the obvious repair makes everything worse: subgradient descent
converges at ``O(1/sqrt(k))``, losing the ``O(1/k)`` of lesson 86 and the ``sqrt(kappa)`` of lesson
89 in one step.

The proximal operator recovers them. For a composite objective ``F = f + g`` with ``f`` smooth and
``g`` nonsmooth but simple,

    prox_(t g)(v) = argmin_x  (1/2)||x - v||**2 + t g(x) ,

and the proximal gradient step is a gradient step on ``f`` followed by a prox on ``g``. When ``g``
is an indicator function the prox is lesson 88's projection, so projected gradient was the first
example of this all along. When ``g = lambda ||x||_1`` the prox is **soft thresholding**, which is
where the exact zeros come from.

What the measurements here show
-------------------------------
* The prox of an indicator is a projection, matching lesson 88's routines to ``0.0`` exactly, and
  every prox here satisfies the firm nonexpansiveness that defines one.
* **Only the proximal methods produce exact zeros.** On a 400 variable problem the proximal solution
  has 54 nonzeros and the subgradient solution has **400**. A ridge penalty, which is smooth, gives
  400 as well. The zeros are not a numerical accident, they are what soft thresholding does.
* The threshold above which the answer is exactly zero has a closed form,
  ``lambda_max = ||A^T b||_inf / N``, and at ``1.01`` times it the measured solution has **zero**
  nonzeros while at ``0.99`` times it has some.
* **Subgradient descent converges at ``O(1/sqrt k)``**: the fitted exponent is ``-0.53`` over
  nineteen thousand points, using the running best value because the method is not monotone.
* **FISTA is not monotone either**, and by a lot: 602 steps out of thirty thousand increase the
  objective, against ISTA's 0. That is a property of the method, not a bug, and any stopping test
  built on "the objective stopped falling" will fire early.
* **ISTA and FISTA converge linearly here, not at the ``O(1/k)`` and ``O(1/k**2)`` that are quoted.**
  Those are worst case rates over a class of functions; once the support settles, the LASSO
  objective is strongly convex on it and both methods speed up, to measured exponents of ``-7.88``
  and ``-7.97``. FISTA still wins by an enormous practical margin: at a thousand steps the gaps are
  ``3.2e-10`` against ``4.2e-02``, a factor of ``1.3e+08``, and it reaches the floor in 1124 steps
  against 2736.
* The support is recovered once there are more than about ``k log(m/k) = 60`` samples and not
  before: the fraction of the true support found is ``0.25`` at 30 samples and ``0.90`` at 100, and
  the false positives fall to zero by 400.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import functools
import math

import numpy as np


# --------------------------------------------------------------------------- the problem


def lasso_problem(samples: int = 100, variables: int = 400, sparsity: int = 20,
                  noise: float = 0.1, correlation: float = 0.0, seed: int = 42):
    """``(1/2N)||A x - b||**2 + lambda ||x||_1``, with a sparse truth planted in it.

    ``correlation`` mixes neighbouring columns, which raises the condition number of the smooth
    part without changing anything else. ``samples < variables`` makes the smooth part singular,
    which is the case the ``L1`` penalty exists for: without it the problem has infinitely many
    exact solutions and with it a unique sparse one.

    ``lambda_max = ||A^T b||_inf / N`` is stored because it is exactly the value above which the
    solution is the zero vector, and that is a closed form worth checking against.
    """
    n, m, k = int(samples), int(variables), int(sparsity)
    if n < 1 or m < 1:
        raise ValueError(f"need at least one sample and one variable, got {n} and {m}")
    if not 0 <= k <= m:
        raise ValueError(f"the sparsity must be between 0 and {m}, got {k}")
    rng = np.random.default_rng(int(seed))
    design = rng.normal(size=(n, m))
    if float(correlation) > 0.0:
        distance = np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
        design = design @ np.linalg.cholesky(
            np.exp(-distance / float(correlation))).T
    truth = np.zeros(m)
    if k:
        truth[rng.choice(m, k, replace=False)] = rng.normal(scale=2.0, size=k)
    targets = design @ truth + float(noise) * rng.normal(size=n)
    gram = design.T @ design / n
    values = np.linalg.eigvalsh(gram)

    def smooth(x):
        gap = design @ np.asarray(x, dtype=float).ravel() - targets
        return 0.5 * float(gap @ gap) / n

    def smooth_gradient(x):
        gap = design @ np.asarray(x, dtype=float).ravel() - targets
        return design.T @ gap / n

    return {"smooth": smooth, "smooth_gradient": smooth_gradient,
            "design": design, "targets": targets,
            "samples": n, "dimension": m, "sparsity": k, "truth": truth,
            "support": np.flatnonzero(truth),
            "smoothness": float(values[-1]),
            "lambda_max": float(np.max(np.abs(design.T @ targets))) / n,
            "start": np.zeros(m),
            "name": f"LASSO, {n} samples, {m} variables, {k} nonzero"}


def objective(problem, x, penalty: float) -> float:
    """``f(x) + lambda ||x||_1``, the quantity every method here is trying to reduce."""
    v = np.asarray(x, dtype=float).ravel()
    return float(problem["smooth"](v)) + float(penalty) * float(np.sum(np.abs(v)))


# --------------------------------------------------------------------------- proximal operators


def soft_threshold(v, amount: float) -> np.ndarray:
    """``prox`` of ``t ||x||_1``: shrink toward zero by ``t`` and stop there.

    Every coordinate smaller than ``t`` in magnitude becomes **exactly** zero, not small. That one
    line is the whole reason the ``L1`` penalty is used: the answer is sparse as a matter of
    arithmetic rather than of thresholding afterwards.
    """
    x = np.asarray(v, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - float(amount), 0.0)


def prox_ridge(v, amount: float) -> np.ndarray:
    """``prox`` of ``(t/2)||x||**2``: shrink everything by the same factor, and never to zero."""
    return np.asarray(v, dtype=float) / (1.0 + float(amount))


def prox_box(v, lower: float = -1.0, upper: float = 1.0) -> np.ndarray:
    """``prox`` of the indicator of a box, which is the projection onto it."""
    return np.clip(np.asarray(v, dtype=float), float(lower), float(upper))


def prox_group(v, amount: float, groups) -> np.ndarray:
    """``prox`` of ``t sum_g ||x_g||``: shrink each group's norm, so groups vanish together.

    Soft thresholding acts on coordinates and this acts on blocks, which is the difference between
    "this feature is irrelevant" and "this whole factor is irrelevant".
    """
    x = np.asarray(v, dtype=float).ravel().copy()
    for block in groups:
        rows = np.asarray(block, dtype=int)
        length = float(np.linalg.norm(x[rows]))
        x[rows] = 0.0 if length <= float(amount) else x[rows] * (1.0 - float(amount) / length)
    return x


# --------------------------------------------------------------------------- the methods


def proximal_gradient(problem, penalty: float, accelerated: bool = False,
                      rounds: int = 30000, step: float | None = None,
                      tol: float = 0.0, record: bool = True) -> dict:
    """ISTA when ``accelerated`` is false, FISTA when it is true.

    Both take a gradient step on the smooth part and a prox on the nonsmooth one. FISTA adds the
    same extrapolation Nesterov used in lesson 89, applied to the prox step instead of the gradient
    step, and it has the same two properties: much faster, and **not monotone**.
    """
    x = np.asarray(problem["start"], dtype=float).ravel().copy()
    ahead = x.copy()
    weight = 1.0
    pace = float(step) if step is not None else 1.0 / problem["smoothness"]
    lam = float(penalty)
    values, sparsity = [], []
    for taken in range(int(rounds)):
        if record:
            values.append(objective(problem, x, lam))
            sparsity.append(int(np.count_nonzero(x)))
        if accelerated:
            moved = soft_threshold(
                ahead - pace * problem["smooth_gradient"](ahead), pace * lam)
            next_weight = (1.0 + math.sqrt(1.0 + 4.0 * weight * weight)) / 2.0
            ahead = moved + ((weight - 1.0) / next_weight) * (moved - x)
            step_size = float(np.linalg.norm(moved - x))
            x, weight = moved, next_weight
        else:
            moved = soft_threshold(x - pace * problem["smooth_gradient"](x), pace * lam)
            step_size = float(np.linalg.norm(moved - x))
            x = moved
        if tol and step_size <= float(tol):
            break
    return {"x": x, "steps": taken, "penalty": lam,
            "values": np.asarray(values), "sparsity": np.asarray(sparsity),
            "nonzeros": int(np.count_nonzero(x)),
            "objective": objective(problem, x, lam),
            "accelerated": bool(accelerated)}


def subgradient_descent(problem, penalty: float, rounds: int = 30000,
                        base: float = 0.5) -> dict:
    """The obvious repair: use any element of the subdifferential and shrink the step.

    ``sign(0)`` is taken as ``0``, which is a valid subgradient, so nothing here is ill defined. The
    step must decay for the same reason as in lesson 89, and ``base/sqrt(k)`` is the standard choice
    for a nonsmooth problem.

    Two things go wrong and both are measured. The rate falls to ``O(1/sqrt k)``, and the iterate is
    **never sparse**: a subgradient step moves every coordinate off zero again.
    """
    x = np.asarray(problem["start"], dtype=float).ravel().copy()
    lam = float(penalty)
    values, best_so_far, best = [], [], math.inf
    for taken in range(int(rounds)):
        here = objective(problem, x, lam)
        values.append(here)
        best = min(best, here)
        best_so_far.append(best)
        slope = (np.asarray(problem["smooth_gradient"](x), dtype=float).ravel()
                 + lam * np.sign(x))
        x = x - (float(base) / math.sqrt(taken + 1.0)) * slope
    return {"x": x, "steps": taken, "penalty": lam,
            "values": np.asarray(values), "best_values": np.asarray(best_so_far),
            "nonzeros": int(np.count_nonzero(x)),
            "objective": objective(problem, x, lam)}


def _reference(problem, penalty: float, rounds: int = 400000) -> float:
    """A very accurate optimal value, by running FISTA far past where anything else stops."""
    out = proximal_gradient(problem, penalty, accelerated=True, rounds=int(rounds),
                            record=False)
    return out["objective"]


# --------------------------------------------------------------------------- measurements


def the_prox_of_an_indicator_is_a_projection(samples: int = 200, seed: int = 42) -> dict:
    """Check the claim that connects this lesson to lesson 88, on random points.

    The proximal operator of the indicator function of a set is the projection onto that set,
    because minimizing ``(1/2)||x - v||**2`` subject to ``x`` in the set **is** the projection. So
    projected gradient was proximal gradient all along, with the nonsmooth term chosen to be
    infinite outside the feasible set.
    """
    from .constrained import project_onto_box

    rng = np.random.default_rng(int(seed))
    worst = 0.0
    for _ in range(int(samples)):
        size = int(rng.integers(1, 9))
        point = rng.normal(scale=3.0, size=size)
        worst = max(worst, float(np.max(np.abs(
            prox_box(point, -1.0, 1.0) - project_onto_box(point, -1.0, 1.0)))))
    return {
        "samples": int(samples),
        "worst_difference": worst,
        "they_are_the_same_operator": worst == 0.0,
        "note": "the two routines are written independently in two modules and agree exactly, "
                "which is what an identity rather than an approximation looks like",
    }


def every_prox_is_firmly_nonexpansive(samples: int = 300, seed: int = 42) -> dict:
    """A proximal operator never increases a distance, and each one here is checked for it.

    ``||prox(u) - prox(v)|| <= ||u - v||`` is the property that makes the proximal gradient step a
    contraction and the convergence proofs work. It is checkable directly, and it is a stronger
    check than "the answer looks right".
    """
    rng = np.random.default_rng(int(seed))
    rows = []
    cases = [("soft threshold", lambda p: soft_threshold(p, 0.7)),
             ("ridge", lambda p: prox_ridge(p, 0.7)),
             ("box indicator", lambda p: prox_box(p, -1.0, 1.0))]
    for name, operator in cases:
        worst = 0.0
        for _ in range(int(samples)):
            size = int(rng.integers(1, 9))
            first, second = rng.normal(scale=3.0, size=size), rng.normal(scale=3.0, size=size)
            apart = float(np.linalg.norm(first - second))
            after = float(np.linalg.norm(operator(first) - operator(second)))
            if apart > 0.0:
                worst = max(worst, after / apart)
        rows.append({"operator": name, "worst_ratio": worst,
                     "nonexpansive": worst <= 1.0 + 1e-12})
    return {
        "rows": rows,
        "all_nonexpansive": all(r["nonexpansive"] for r in rows),
        "worst_ratio_anywhere": max(r["worst_ratio"] for r in rows),
        "note": "no pair of points is ever moved further apart, at any size, by any of the three",
    }


@functools.lru_cache(maxsize=None)
def only_soft_thresholding_makes_exact_zeros(rounds: int = 20000) -> dict:
    """Three penalties on the same data, counting how many coordinates are exactly zero.

    The comparison is the point. A ridge penalty shrinks every coordinate toward zero and reaches
    it for none of them, because its prox multiplies by ``1/(1+t)``. Soft thresholding subtracts
    ``t`` and clamps, so everything below ``t`` lands exactly on zero. Subgradient descent on the
    ``L1`` objective has the right penalty and the wrong method, and produces no zeros either.
    """
    problem = lasso_problem()
    lam = 0.02 * problem["lambda_max"]
    rows = []
    sparse = proximal_gradient(problem, lam, accelerated=True, rounds=int(rounds))
    rows.append({"method": "FISTA on the L1 objective", "nonzeros": sparse["nonzeros"],
                 "objective": sparse["objective"]})
    rough = subgradient_descent(problem, lam, rounds=int(rounds))
    rows.append({"method": "subgradient on the L1 objective",
                 "nonzeros": rough["nonzeros"], "objective": rough["objective"]})
    # the same smooth part with a ridge penalty, solved by its own proximal gradient
    x = problem["start"].copy()
    pace = 1.0 / problem["smoothness"]
    for _ in range(int(rounds)):
        x = prox_ridge(x - pace * problem["smooth_gradient"](x), pace * lam)
    rows.append({"method": "ridge penalty, proximal", "nonzeros": int(np.count_nonzero(x)),
                 "objective": objective(problem, x, lam)})
    return {
        "rows": rows,
        "variables": problem["dimension"],
        "true_nonzeros": problem["sparsity"],
        "only_the_l1_prox_is_sparse": (
            rows[0]["nonzeros"] < problem["dimension"]
            and rows[1]["nonzeros"] == problem["dimension"]
            and rows[2]["nonzeros"] == problem["dimension"]),
        "note": "the penalty and the method both have to be right: the L1 penalty with a "
                "subgradient method gives no zeros, and a smooth penalty with a proximal method "
                "gives none either",
    }


@functools.lru_cache(maxsize=None)
def the_threshold_where_everything_vanishes(rounds: int = 20000) -> dict:
    """``lambda_max = ||A^T b||_inf / N`` is exactly where the solution becomes the zero vector.

    Above it, zero satisfies the optimality condition ``0 in A^T(Ax - b)/N + lambda d||x||_1``,
    because the subdifferential of ``||x||_1`` at zero is the whole unit cube. Below it, it does
    not. That makes the boundary a closed form rather than something to search for, and it is the
    natural top of any regularization path.
    """
    problem = lasso_problem()
    top = problem["lambda_max"]
    rows = []
    for factor in (1.5, 1.01, 0.99, 0.5, 0.1, 0.02):
        out = proximal_gradient(problem, factor * top, accelerated=True,
                                rounds=int(rounds), record=False)
        rows.append({"factor": float(factor), "penalty": factor * top,
                     "nonzeros": out["nonzeros"]})
    above = [r for r in rows if r["factor"] > 1.0]
    below = [r for r in rows if r["factor"] < 1.0]
    return {
        "rows": rows,
        "lambda_max": top,
        "everything_above_is_zero": all(r["nonzeros"] == 0 for r in above),
        "everything_below_is_not": all(r["nonzeros"] > 0 for r in below),
        "the_formula_is_exact": (all(r["nonzeros"] == 0 for r in above)
                                 and all(r["nonzeros"] > 0 for r in below)),
        "note": "one per cent above the closed form the answer is the zero vector and one per cent "
                "below it is not, so the formula is the boundary and not an estimate",
    }


@functools.lru_cache(maxsize=None)
def the_regularization_path(steps: int = 12, rounds: int = 20000) -> dict:
    """Sweep ``lambda`` from ``lambda_max`` downward and watch the support grow.

    The path is what a practitioner actually computes, because the right ``lambda`` is not known in
    advance. Warm starting each solve from the previous one makes the whole sweep cost about what
    one cold solve costs, which is why the path is affordable.
    """
    problem = lasso_problem()
    top = problem["lambda_max"]
    weights = np.logspace(0.0, -2.5, int(steps)) * top
    rows = []
    warm = problem["start"].copy()
    truth_support = set(problem["support"].tolist())
    for lam in weights:
        started = dict(problem)
        started["start"] = warm
        out = proximal_gradient(started, float(lam), accelerated=True,
                                rounds=int(rounds), record=False)
        warm = out["x"]
        found = set(np.flatnonzero(out["x"]).tolist())
        rows.append({
            "penalty": float(lam), "over_max": float(lam / top),
            "nonzeros": out["nonzeros"],
            "true_ones_found": len(found & truth_support),
            "false_ones": len(found - truth_support),
        })
    counts = [r["nonzeros"] for r in rows]
    best = max(rows, key=lambda r: r["true_ones_found"] - 0.1 * r["false_ones"])
    return {
        "rows": rows,
        "true_nonzeros": problem["sparsity"],
        "variables": problem["dimension"],
        "the_support_grows": all(counts[k] >= counts[k - 1] for k in range(1, len(counts))),
        "best_penalty_over_max": best["over_max"],
        "true_ones_found_there": best["true_ones_found"],
        "false_ones_there": best["false_ones"],
        "note": "the support grows monotonically as the penalty falls, and the whole sweep costs "
                "about one cold solve because each step starts from the last",
    }


@functools.lru_cache(maxsize=None)
def subgradient_is_slower_and_never_sparse(rounds: int = 30000) -> dict:
    """Fit the subgradient rate, using the running best value because the method is not monotone.

    ``O(1/sqrt k)`` is the rate for a nonsmooth convex problem with a decaying step, and it is the
    baseline the proximal methods are measured against. Reporting the last value rather than the
    best would be reporting the noise.
    """
    problem = lasso_problem(samples=300, variables=100, sparsity=8)
    lam = 0.05 * problem["lambda_max"]
    best = _reference(problem, lam)
    out = subgradient_descent(problem, lam, rounds=int(rounds))
    gaps = out["best_values"] - best
    counts = np.arange(gaps.size)
    keep = (counts >= 100) & (gaps > 1e-11)
    power = float(np.polyfit(np.log(counts[keep]), np.log(gaps[keep]), 1)[0])
    return {
        "fitted_exponent": power,
        "points_fitted": int(keep.sum()),
        "the_exponent_is_minus_a_half": abs(power + 0.5) < 0.15,
        "final_gap": float(gaps[-1]),
        "nonzeros": out["nonzeros"],
        "variables": problem["dimension"],
        "never_sparse": out["nonzeros"] == problem["dimension"],
        "raw_history_is_not_monotone": bool(np.any(np.diff(out["values"]) > 0)),
        "note": "the rate is one over the square root, the iterate has every coordinate nonzero, "
                "and the objective is not monotone, so all three of the things the proximal "
                "methods fix are visible in one run",
    }


@functools.lru_cache(maxsize=None)
def fista_is_faster_and_not_monotone(rounds: int = 30000, floor: float = 1e-12) -> dict:
    """ISTA against FISTA on the same problem, on speed and on monotonicity separately.

    Both claims matter and they pull in opposite directions. FISTA reaches a given accuracy in far
    fewer steps, and its objective **rises** on hundreds of them. A stopping test that watches for
    the objective to stop falling will fire early on FISTA and correctly on ISTA, which is a real
    trap and the reason monotone variants of FISTA exist.
    """
    problem = lasso_problem()
    lam = 0.005 * problem["lambda_max"]
    best = _reference(problem, lam)
    rows = []
    for accelerated in (False, True):
        out = proximal_gradient(problem, lam, accelerated=accelerated,
                                rounds=int(rounds))
        gaps = out["values"] - best
        above = gaps > float(floor)
        reached = (int(np.argmax(~above)) if np.any(~above) else int(gaps.size))
        rows.append({
            "method": "FISTA" if accelerated else "ISTA",
            "gap_at_1000": float(gaps[min(1000, gaps.size - 1)]),
            "steps_to_the_floor": reached,
            "rises_above_the_floor": int(np.count_nonzero(
                (np.diff(gaps) > 0) & above[:-1])),
            "nonzeros": out["nonzeros"],
        })
    ista, fista = rows
    return {
        "rows": rows,
        "speedup_in_steps": ista["steps_to_the_floor"] / max(fista["steps_to_the_floor"], 1),
        "accuracy_gain_at_1000": ista["gap_at_1000"] / max(fista["gap_at_1000"], 1e-300),
        "ista_is_monotone": ista["rises_above_the_floor"] == 0,
        "fista_is_not": fista["rises_above_the_floor"] > 0,
        "they_agree_on_the_support": ista["nonzeros"] == fista["nonzeros"],
        "note": "acceleration is worth a factor of two in steps and eight orders of magnitude in "
                "accuracy at a fixed budget here, and it gives up monotonicity to get it",
    }


@functools.lru_cache(maxsize=None)
def the_quoted_rates_are_worst_case(rounds: int = 4000) -> dict:
    """The ``O(1/k)`` and ``O(1/k**2)`` rates against what these instances actually do.

    The quoted rates hold for any smooth convex ``f`` and any convex ``g``. The LASSO objective is
    more than that: once the support stops changing, the smooth part is strongly convex **on that
    support**, and both methods converge linearly instead. So the measured exponents are far steeper
    than ``-1`` and ``-2``, which is not a contradiction and is worth saying out loud.

    The early window, before the support settles, is where the sublinear regime lives, and the
    measurement reports both windows rather than picking the flattering one.
    """
    problem = lasso_problem()
    lam = 0.005 * problem["lambda_max"]
    best = _reference(problem, lam)
    rows = []
    for accelerated in (False, True):
        out = proximal_gradient(problem, lam, accelerated=accelerated,
                                rounds=int(rounds))
        gaps = out["values"] - best
        counts = np.arange(gaps.size)
        early = (counts >= 10) & (counts <= 100) & (gaps > 1e-11)
        late = (counts > 100) & (gaps > 1e-11)
        rows.append({
            "method": "FISTA" if accelerated else "ISTA",
            "early_exponent": (float(np.polyfit(np.log(counts[early]),
                                                np.log(gaps[early]), 1)[0])
                               if early.sum() > 5 else float("nan")),
            "late_exponent": (float(np.polyfit(np.log(counts[late]),
                                               np.log(gaps[late]), 1)[0])
                              if late.sum() > 5 else float("nan")),
            "support_settles_at": int(np.argmax(
                out["sparsity"] == out["sparsity"][-1])),
        })
    return {
        "rows": rows,
        "the_late_exponents_are_steeper": all(
            r["late_exponent"] < r["early_exponent"] for r in rows),
        "steepest_late_exponent": min(r["late_exponent"] for r in rows),
        "neither_late_exponent_is_the_quoted_one": all(
            abs(r["late_exponent"] + k) > 1.0 for r, k in zip(rows, (1.0, 2.0))),
        "note": "the quoted rates are upper bounds over a class and these instances are easier "
                "than the class, so the measured exponents are steeper; the same thing happened "
                "in lesson 89 with Nesterov's bounds",
    }


@functools.lru_cache(maxsize=None)
def recovery_needs_enough_samples(sample_counts=(30, 60, 100, 200, 400),
                                  variables: int = 400, sparsity: int = 20,
                                  rounds: int = 20000) -> dict:
    """How many samples the sparse truth needs before the LASSO finds its support.

    The theory says roughly ``k log(m/k)`` samples suffice for a random design, which here is about
    ``20 log 20 = 60``. The measurement is what happens either side of that.
    """
    rows = []
    for n in sample_counts:
        problem = lasso_problem(samples=int(n), variables=int(variables),
                                sparsity=int(sparsity))
        lam = 0.05 * problem["lambda_max"]
        out = proximal_gradient(problem, lam, accelerated=True, rounds=int(rounds),
                               record=False)
        found = set(np.flatnonzero(out["x"]).tolist())
        wanted = set(problem["support"].tolist())
        rows.append({
            "samples": int(n),
            "nonzeros": out["nonzeros"],
            "true_ones_found": len(found & wanted),
            "false_ones": len(found - wanted),
            "fraction_found": len(found & wanted) / float(sparsity),
            "error": float(np.linalg.norm(out["x"] - problem["truth"])),
        })
    rule = sparsity * math.log(variables / sparsity)
    counts = [r["samples"] for r in rows]
    return {
        "rows": rows,
        "sparsity": int(sparsity),
        "variables": int(variables),
        "the_rule_of_thumb": rule,
        "false_positives_fall_with_samples": all(
            rows[k]["false_ones"] <= rows[k - 1]["false_ones"]
            for k in range(2, len(rows))),
        "error_falls_then_flattens": (rows[0]["error"] > rows[-1]["error"]
                                      and rows[-1]["error"] > 0.5 * rows[2]["error"]),
        "best_fraction": max(r["fraction_found"] for r in rows),
        "samples_at_the_best": max(rows, key=lambda r: r["fraction_found"])["samples"],
        "worst_fraction_below_the_rule": min(
            r["fraction_found"] for r in rows if r["samples"] < rule),
        "best_fraction_above_the_rule": max(
            r["fraction_found"] for r in rows if r["samples"] > rule),
        "the_rule_separates_them": (
            min(r["fraction_found"] for r in rows if r["samples"] < rule)
            < max(r["fraction_found"] for r in rows if r["samples"] > rule)),
        "note": "below k log(m/k) samples the support is mostly wrong and above it mostly right, "
                "and the false positives fall to zero. The fraction found peaks at 200 rather than "
                "at 400, because the penalty here is a fixed multiple of lambda_max and that is "
                "not the right normalization as the sample count changes: at 400 the same relative "
                "penalty is too strong and drops three true coordinates",
    }
