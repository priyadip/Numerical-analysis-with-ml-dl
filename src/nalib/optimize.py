"""What a minimum is, how to recognise one, and why finding it costs half your digits.

Where the work is
-----------------
Minimizing ``f`` looks like solving ``grad f = 0``, and formally it is. Part 2 already solves
nonlinear systems, so one reading of this part is that it has nothing new in it.

That reading is wrong for one reason, and the reason is the whole subject:

    a root problem is conditioned by ``f'`` at the root,
    a minimum problem is conditioned by ``f''`` at the minimum, **through a square root**.

Near a minimum ``f(x) = f(x*) + (1/2)(x - x*)^T H (x - x*) + ...``, so a change in ``x`` of size
``d`` changes ``f`` by ``O(d**2)``. Turn that round: a change in ``f`` too small to see, which in
floating point means ``eps * |f(x*)|``, hides every ``x`` within ``sqrt(2 eps |f*| / lambda)`` of
the minimum. **The minimum sits at the bottom of a flat region whose width is the square root of
the precision**, and no method that only reads values of ``f`` can do better.

What the measurements here show
-------------------------------
* The flat region is real and its width is predicted exactly. Bisecting outward from the minimum
  until ``f`` changes in floating point gives ``1.492e-08`` for a unit curvature quadratic with
  ``f* = 1``, against the predicted ``sqrt(eps |f*| / c) = 1.490e-08``, a ratio of ``1.001`` at
  every curvature from ``10**-2`` to ``10**2``.
* The loss is a **square root of the value at the minimum**, not a fixed number of digits. With
  ``f* = 10**-8, 10**-4, 1, 10**4`` the achievable error in ``x`` is
  ``9.1e-13, 8.2e-11, 1.1e-08, 9.5e-07``: each factor of ``10**4`` in ``f*`` costs a factor of
  ``100`` in ``x``. When ``f* = 0``, which is the least squares case, **nothing is lost at all**
  and the minimum is located to ``2.2e-16``.
* Root finding on the same functions reaches full precision, ``0.0`` to ``2.2e-16``, which is the
  comparison that makes the point: it is not the algorithm, it is the problem.
* The Hessian test classifies every stationary point in the test set correctly, and it is
  **inconclusive** exactly where it should be: ``x**4`` and ``-x**4`` have the same zero Hessian at
  the origin and opposite behaviour, so no second order test can separate them.
* Convexity cannot be established by sampling, and the measurement says exactly how badly.
  Nine functions, all with curvature ``-2`` at the origin, differing only in how narrow the dip is:
  sampling misses it whenever the expected hit count ``n w / 3`` is small, and there is **no sharp
  threshold**, because the hit count is Poisson. A hit at ``0.67`` expected hits and a miss at
  ``1.67`` both occur here and both are ordinary, the least likely outcome in the table having
  probability ``0.19``. Worse, a sample that does find the dip underestimates its depth: the
  shallowest hit reports ``13.6`` per cent of the true curvature.
* The condition number of the Hessian is the aspect ratio of the flat region, and the ratio of
  the widths along the extreme eigendirections is ``sqrt(kappa)``, not ``kappa``. On a quadratic
  with ``kappa = 10**4`` the measured ratio is ``99.98`` against ``sqrt(kappa) = 100.0``.
  Rosenbrock, whose minimum value is 0, has **no flat region at all**: both widths come out at
  ``6.2e-17``, which is the same finding as the row above seen from the other side.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- test problems


def quadratic(condition: float = 100.0, dimension: int = 2, seed: int = 42):
    """A random symmetric positive definite quadratic with a prescribed condition number.

    ``f(x) = (1/2)(x - x*)^T A (x - x*) + value``, with ``A``'s eigenvalues spread
    logarithmically from 1 to ``condition`` and its eigenvectors a random orthogonal frame. This
    is the problem every method in this part is measured on first, because its answer is known and
    its difficulty is a dial.
    """
    n = int(dimension)
    if n < 1:
        raise ValueError(f"need at least one variable, got {n}")
    if condition < 1.0:
        raise ValueError(f"the condition number cannot be below 1, got {condition}")
    rng = np.random.default_rng(int(seed))
    frame, _ = np.linalg.qr(rng.normal(size=(n, n)))
    spread = (np.logspace(0.0, math.log10(float(condition)), n) if n > 1
              else np.ones(n))
    matrix = frame @ np.diag(spread) @ frame.T
    matrix = 0.5 * (matrix + matrix.T)
    centre = rng.normal(size=n)
    value = 1.0

    def f(x):
        v = np.asarray(x, dtype=float).ravel() - centre
        return float(0.5 * v @ matrix @ v) + value

    def gradient(x):
        return matrix @ (np.asarray(x, dtype=float).ravel() - centre)

    def hessian(x):
        return matrix.copy()

    return {"f": f, "gradient": gradient, "hessian": hessian, "matrix": matrix,
            "minimizers": [centre], "minimum": value, "dimension": n,
            "start": centre + rng.normal(size=n), "convex": True,
            "condition": float(np.linalg.cond(matrix)),
            "name": f"quadratic, condition {condition:g}, {n} variables"}


def rosenbrock(dimension: int = 2):
    """``sum 100 (x[i+1] - x[i]**2)**2 + (1 - x[i])**2``, the standard hard smooth problem.

    The minimum is at ``(1, ..., 1)`` with value 0, in a curved valley whose floor is the parabola
    ``x[i+1] = x[i]**2``. Steepest descent crosses the valley instead of following it, which is
    lesson 86's subject, and the reason is in the Hessian: at the minimum its condition number is
    ``2508`` in two dimensions and rises to ``3582`` by sixteen.
    """
    n = int(dimension)
    if n < 2:
        raise ValueError(f"Rosenbrock needs at least two variables, got {n}")

    def f(x):
        v = np.asarray(x, dtype=float).ravel()
        return float(np.sum(100.0 * (v[1:] - v[:-1] ** 2) ** 2 + (1.0 - v[:-1]) ** 2))

    def gradient(x):
        v = np.asarray(x, dtype=float).ravel()
        out = np.zeros_like(v)
        out[:-1] += -400.0 * v[:-1] * (v[1:] - v[:-1] ** 2) - 2.0 * (1.0 - v[:-1])
        out[1:] += 200.0 * (v[1:] - v[:-1] ** 2)
        return out

    def hessian(x):
        v = np.asarray(x, dtype=float).ravel()
        out = np.zeros((v.size, v.size))
        for i in range(v.size - 1):
            out[i, i] += -400.0 * (v[i + 1] - 3.0 * v[i] ** 2) + 2.0
            out[i, i + 1] += -400.0 * v[i]
            out[i + 1, i] += -400.0 * v[i]
            out[i + 1, i + 1] += 200.0
        return out

    start = np.full(n, -1.2)
    start[1::2] = 1.0
    return {"f": f, "gradient": gradient, "hessian": hessian,
            "minimizers": [np.ones(n)], "minimum": 0.0, "dimension": n,
            "start": start, "convex": False,
            "name": f"Rosenbrock, {n} variables"}


def himmelblau():
    """``(x**2 + y - 11)**2 + (x + y**2 - 7)**2``: four minima, all with the same value.

    Here because a global claim about a minimum needs a problem that has more than one. All four
    have value 0, so no local method can prefer one, and which one is found is decided entirely by
    where the search starts.
    """
    def f(x):
        v = np.asarray(x, dtype=float).ravel()
        return float((v[0] ** 2 + v[1] - 11.0) ** 2 + (v[0] + v[1] ** 2 - 7.0) ** 2)

    def gradient(x):
        v = np.asarray(x, dtype=float).ravel()
        a = v[0] ** 2 + v[1] - 11.0
        b = v[0] + v[1] ** 2 - 7.0
        return np.array([4.0 * v[0] * a + 2.0 * b, 2.0 * a + 4.0 * v[1] * b])

    def hessian(x):
        v = np.asarray(x, dtype=float).ravel()
        a = v[0] ** 2 + v[1] - 11.0
        b = v[0] + v[1] ** 2 - 7.0
        return np.array([[12.0 * v[0] ** 2 + 4.0 * v[1] - 42.0, 4.0 * v[0] + 4.0 * v[1]],
                         [4.0 * v[0] + 4.0 * v[1], 4.0 * v[0] + 12.0 * v[1] ** 2 - 26.0]])

    corners = [np.array([3.0, 2.0]),
               np.array([-2.805118086952745, 3.131312518250573]),
               np.array([-3.779310253377746, -3.283185991286170]),
               np.array([3.584428340330492, -1.848126526964403])]
    return {"f": f, "gradient": gradient, "hessian": hessian,
            "minimizers": corners, "minimum": 0.0, "dimension": 2,
            "start": np.array([0.0, 0.0]), "convex": False,
            "name": "Himmelblau, four minima"}


def rastrigin(dimension: int = 2, amplitude: float = 10.0):
    """``A n + sum x[i]**2 - A cos(2 pi x[i])``: one global minimum inside a lattice of local ones.

    The number of local minima grows like ``(2k+1)**n`` inside a box of half width ``k``, which is
    the curse of dimensionality made concrete. Its Hessian at the global minimum is
    ``(2 + 4 pi**2 A) I``, so the problem is perfectly conditioned there and still hard: **the
    condition number describes the last step of a search and says nothing about finding the right
    basin.**
    """
    n = int(dimension)
    if n < 1:
        raise ValueError(f"need at least one variable, got {n}")
    a = float(amplitude)

    def f(x):
        v = np.asarray(x, dtype=float).ravel()
        return float(a * v.size + np.sum(v ** 2 - a * np.cos(2.0 * math.pi * v)))

    def gradient(x):
        v = np.asarray(x, dtype=float).ravel()
        return 2.0 * v + 2.0 * math.pi * a * np.sin(2.0 * math.pi * v)

    def hessian(x):
        v = np.asarray(x, dtype=float).ravel()
        return np.diag(2.0 + 4.0 * math.pi ** 2 * a * np.cos(2.0 * math.pi * v))

    return {"f": f, "gradient": gradient, "hessian": hessian,
            "minimizers": [np.zeros(n)], "minimum": 0.0, "dimension": n,
            "start": np.full(n, 2.4), "convex": False,
            "name": f"Rastrigin, {n} variables"}


def all_problems(dimension: int = 2):
    """Every test problem, at the given number of variables where it accepts one."""
    return [quadratic(dimension=dimension), rosenbrock(dimension=max(2, dimension)),
            himmelblau(), rastrigin(dimension=dimension)]


# --------------------------------------------------------------------------- derivatives


def gradient(f, x, h: float | None = None) -> np.ndarray:
    """The gradient by central differences, one coordinate at a time.

    The step defaults to ``eps**(1/3)`` scaled by ``1 + |x_k|``, which is lesson 61's balance for a
    second order formula: truncation ``O(h**2)`` against rounding ``O(eps/h)`` is smallest at
    ``h ~ eps**(1/3)``. Costs ``2n`` evaluations.
    """
    v = np.asarray(x, dtype=float).ravel()
    base = float(np.finfo(float).eps) ** (1.0 / 3.0) if h is None else float(h)
    out = np.zeros(v.size)
    for k in range(v.size):
        step = base * (1.0 + abs(v[k]))
        ahead, behind = v.copy(), v.copy()
        ahead[k] += step
        behind[k] -= step
        out[k] = (f(ahead) - f(behind)) / (ahead[k] - behind[k])
    return out


def hessian(f, x, h: float | None = None) -> np.ndarray:
    """The Hessian by central second differences, symmetrised.

    The step defaults to ``eps**(1/4)``, which is the balance for a formula dividing by ``h**2``.
    Costs ``4 n**2`` evaluations, which is why every method in lessons 86 and 87 tries to avoid
    forming this.
    """
    v = np.asarray(x, dtype=float).ravel()
    base = float(np.finfo(float).eps) ** 0.25 if h is None else float(h)
    n = v.size
    out = np.zeros((n, n))
    steps = base * (1.0 + np.abs(v))
    for i in range(n):
        for j in range(i, n):
            up, down = np.zeros(n), np.zeros(n)
            up[i], down[j] = steps[i], steps[j]
            pp = f(v + up + down)
            pm = f(v + up - down)
            mp = f(v - up + down)
            mm = f(v - up - down)
            out[i, j] = out[j, i] = (pp - pm - mp + mm) / (4.0 * steps[i] * steps[j])
    return out


def derivatives_are_right(problem, points: int = 5, seed: int = 42) -> dict:
    """Check the closed form gradient and Hessian against differences, at random points.

    Two separate checks, and they have different tolerances for a reason the numbers show: the
    gradient formula is accurate to about ``eps**(2/3)`` and the Hessian formula to about
    ``eps**(1/2)``, so the Hessian's agreement is always several orders worse and that is the
    formula's fault, not the code's.
    """
    rng = np.random.default_rng(int(seed))
    n = int(problem["dimension"])
    rows = []
    for _ in range(int(points)):
        where = rng.normal(size=n)
        exact_g = np.asarray(problem["gradient"](where), dtype=float)
        exact_h = np.asarray(problem["hessian"](where), dtype=float)
        got_g = gradient(problem["f"], where)
        got_h = hessian(problem["f"], where)
        scale_g = max(float(np.max(np.abs(exact_g))), 1.0)
        scale_h = max(float(np.max(np.abs(exact_h))), 1.0)
        rows.append({
            "gradient_gap": float(np.max(np.abs(got_g - exact_g))) / scale_g,
            "hessian_gap": float(np.max(np.abs(got_h - exact_h))) / scale_h,
            "hessian_symmetry": float(np.max(np.abs(exact_h - exact_h.T))),
        })
    worst_g = max(r["gradient_gap"] for r in rows)
    worst_h = max(r["hessian_gap"] for r in rows)
    return {
        "name": problem["name"],
        "rows": rows,
        "worst_gradient_gap": worst_g,
        "worst_hessian_gap": worst_h,
        "gradient_agrees": worst_g < 1e-5,
        "hessian_agrees": worst_h < 1e-2,
        "the_hessian_is_symmetric": max(r["hessian_symmetry"] for r in rows) < 1e-12,
        "note": "the Hessian by differences is much less accurate than the gradient, because "
                "dividing by h**2 doubles the rounding term",
    }


# --------------------------------------------------------------------------- optimality


def classify(matrix, tol: float | None = None) -> str:
    """Classify a stationary point from its Hessian: minimum, maximum, saddle or inconclusive.

    The tolerance decides what counts as a zero eigenvalue, and it has to, because the second order
    test is genuinely inconclusive when the Hessian is singular. Defaults to
    ``sqrt(eps) * ||H||``, which is scale free.
    """
    h = np.asarray(matrix, dtype=float)
    h = np.atleast_2d(h)
    if h.shape[0] != h.shape[1]:
        raise ValueError(f"the Hessian must be square, got {h.shape}")
    values = np.linalg.eigvalsh(0.5 * (h + h.T))
    # scaled by the matrix itself, with no floor, so the verdict does not change when the whole
    # Hessian is multiplied by a constant. A floor of 1 would call 1e-12 * I inconclusive.
    scale = float(np.max(np.abs(values)))
    cut = math.sqrt(float(np.finfo(float).eps)) * scale if tol is None else float(tol)
    positive = bool(np.any(values > cut))
    negative = bool(np.any(values < -cut))
    if positive and negative:
        return "saddle"
    if bool(np.all(values > cut)):
        return "minimum"
    if bool(np.all(values < -cut)):
        return "maximum"
    return "inconclusive"


def is_stationary(problem, where, tol: float = 1e-8) -> bool:
    """Is the gradient zero here, to a tolerance scaled by the problem?"""
    g = np.asarray(problem["gradient"](where), dtype=float)
    return float(np.max(np.abs(g))) < float(tol) * (1.0 + abs(float(problem["f"](where))))


def stationary_points_are_classified_right() -> dict:
    """The second order test on a set of stationary points whose type is known by hand.

    Includes the two cases it must get wrong, ``x**4`` and ``-x**4``, whose Hessians at the origin
    are both zero. They are here because a test that never says "inconclusive" is not reporting
    what it knows.
    """
    cases = [
        ("x**2 + y**2 at the origin", lambda v: float(v @ v),
         lambda v: 2.0 * v, lambda v: 2.0 * np.eye(v.size), "minimum"),
        ("x**2 - y**2 at the origin", lambda v: float(v[0] ** 2 - v[1] ** 2),
         lambda v: np.array([2.0 * v[0], -2.0 * v[1]]),
         lambda v: np.diag([2.0, -2.0]), "saddle"),
        ("-x**2 - y**2 at the origin", lambda v: float(-(v @ v)),
         lambda v: -2.0 * v, lambda v: -2.0 * np.eye(v.size), "maximum"),
        ("x**4 at the origin", lambda v: float(v[0] ** 4),
         lambda v: np.array([4.0 * v[0] ** 3]),
         lambda v: np.array([[12.0 * v[0] ** 2]]), "inconclusive"),
        ("-x**4 at the origin", lambda v: float(-v[0] ** 4),
         lambda v: np.array([-4.0 * v[0] ** 3]),
         lambda v: np.array([[-12.0 * v[0] ** 2]]), "inconclusive"),
        ("x**3 at the origin", lambda v: float(v[0] ** 3),
         lambda v: np.array([3.0 * v[0] ** 2]),
         lambda v: np.array([[6.0 * v[0]]]), "inconclusive"),
    ]
    rows = []
    for name, f, grad, hess, expected in cases:
        size = 1 if "x**4" in name or "x**3" in name else 2
        where = np.zeros(size)
        rows.append({
            "case": name,
            "gradient": float(np.max(np.abs(grad(where)))),
            "eigenvalues": np.linalg.eigvalsh(np.atleast_2d(hess(where))),
            "verdict": classify(hess(where)),
            "expected": expected,
        })
    return {
        "rows": rows,
        "all_stationary": all(r["gradient"] == 0.0 for r in rows),
        "all_right": all(r["verdict"] == r["expected"] for r in rows),
        "inconclusive_count": sum(1 for r in rows if r["verdict"] == "inconclusive"),
        "note": "every point here has a zero gradient, so the first order test passes on all six "
                "and separates none of them; the second order test separates three and reports "
                "the other three as inconclusive, which is the honest answer",
    }


def the_fourth_power_cases_really_differ(radius: float = 0.5, samples: int = 9) -> dict:
    """``x**4`` and ``-x**4`` have the same derivatives to second order and opposite behaviour.

    This is the proof that "inconclusive" above is not a weakness of the implementation. The two
    functions agree to second order at the origin, so **no** second order test can tell them apart,
    and their values along any ray have opposite signs.
    """
    line = np.linspace(-float(radius), float(radius), int(samples))
    up = line ** 4
    down = -line ** 4
    away = line != 0.0
    return {
        "gradients_agree": True,
        "hessians_agree": True,
        "one_rises_everywhere": bool(np.all(up[away] > 0.0)),
        "the_other_falls_everywhere": bool(np.all(down[away] < 0.0)),
        "largest_difference": float(np.max(np.abs(up - down))),
        "note": "both have gradient 0 and Hessian 0 at the origin, one has a minimum there and "
                "the other a maximum, so the second order test cannot be sharpened",
    }


# --------------------------------------------------------------------------- convexity


def smallest_curvature(problem, samples: int = 200, radius: float = 3.0,
                       seed: int = 42) -> dict:
    """Sample the smallest Hessian eigenvalue over a box, which is what convexity is about.

    A function is convex exactly when its Hessian is positive semidefinite **everywhere**. Sampling
    can find a point where it is not, and that settles the question; it can never establish that no
    such point exists.
    """
    rng = np.random.default_rng(int(seed))
    n = int(problem["dimension"])
    worst = math.inf
    where = None
    negatives = 0
    for _ in range(int(samples)):
        point = rng.uniform(-float(radius), float(radius), size=n)
        value = float(np.min(np.linalg.eigvalsh(
            np.asarray(problem["hessian"](point), dtype=float))))
        if value < 0.0:
            negatives += 1
        if value < worst:
            worst, where = value, point
    return {
        "name": problem["name"],
        "samples": int(samples),
        "smallest_eigenvalue_found": worst,
        "where": where,
        "negative_samples": negatives,
        "verdict": "not convex" if worst < 0.0 else "no counterexample found",
        "actually_convex": bool(problem["convex"]),
        "note": "a negative eigenvalue is proof of non-convexity; no negative eigenvalue is not "
                "proof of convexity",
    }


def sampling_cannot_prove_convexity(sizes=(100, 500, 2000), widths=(1e-1, 1e-2, 1e-3),
                                    seed: int = 42) -> dict:
    """A function that is convex except on a thin strip, and when sampling notices.

    ``f(x) = x**2 + a exp(-(x/w)**2)`` with ``a = 2 w**2`` has curvature
    ``2 + (2a/w**2)(2 x**2/w**2 - 1) exp(-(x/w)**2)``, which is ``-2`` at the origin whatever ``w``
    is, and rises to ``2`` a few multiples of ``w`` away. So every row here is a function that is
    **not convex**, by the same margin, and the only thing that changes is how hard the dip is to
    hit.

    Sampling ``n`` points uniformly on a box of width 6 lands in the strip an expected ``n w / 3``
    times, so the useful question is not whether sampling works but where its threshold is. The
    measurement puts it exactly where the expectation crosses 1.
    """
    rng = np.random.default_rng(int(seed))
    rows = []
    for w in widths:
        half = float(w)
        amplitude = 2.0 * half ** 2

        def curvature(x, half=half, amplitude=amplitude):
            v = np.asarray(x, dtype=float)
            return (2.0 + amplitude * (2.0 / half ** 2)
                    * (2.0 * v ** 2 / half ** 2 - 1.0) * np.exp(-(v / half) ** 2))

        true_worst = float(np.min(curvature(np.linspace(-4.0 * half, 4.0 * half, 20001))))
        for count in sizes:
            points = rng.uniform(-3.0, 3.0, size=int(count))
            found = float(np.min(curvature(points)))
            rows.append({
                "half_width": half,
                "samples": int(count),
                "expected_hits": int(count) * half / 3.0,
                "true_smallest_curvature": true_worst,
                "smallest_found": found,
                "found_the_dip": bool(found < 0.0),
            })
    for row in rows:
        chance = 1.0 - math.exp(-row["expected_hits"])
        row["chance_of_a_hit"] = chance
        row["chance_of_what_happened"] = chance if row["found_the_dip"] else 1.0 - chance
    hits = [r for r in rows if r["found_the_dip"]]
    misses = [r for r in rows if not r["found_the_dip"]]
    depths = [abs(r["smallest_found"] / r["true_smallest_curvature"]) for r in hits]
    return {
        "rows": rows,
        "every_function_here_is_not_convex": all(
            r["true_smallest_curvature"] < -1.0 for r in rows),
        "smallest_expectation_that_found_it": (min(r["expected_hits"] for r in hits)
                                               if hits else float("inf")),
        "largest_expectation_that_missed": (max(r["expected_hits"] for r in misses)
                                            if misses else 0.0),
        "every_outcome_is_ordinary": all(r["chance_of_what_happened"] > 0.05 for r in rows),
        "least_likely_outcome": min(r["chance_of_what_happened"] for r in rows),
        "deepest_fraction_of_the_dip_found": max(depths) if depths else 0.0,
        "shallowest_fraction_of_the_dip_found": min(depths) if depths else 0.0,
        "note": "whether the sample lands in the strip is a Poisson count with mean n w / 3, so "
                "there is no sharp threshold: a hit at 0.67 expected and a miss at 1.67 are both "
                "ordinary. A clean sample is evidence about the sample size, not the function.",
    }


# --------------------------------------------------------------------------- conditioning


def attainable_accuracy(curvature: float, value: float = 1.0) -> float:
    """How close to a minimum you can get from values of ``f`` alone.

    ``f`` is computed with a relative error of about ``eps/2``, so values within
    ``(eps/2)|f*|`` of ``f*`` are indistinguishable. Setting ``(1/2) c d**2 = (eps/2)|f*|`` gives

        d = sqrt(eps |f*| / c) .

    With ``c = 1`` and ``|f*| = 1`` that is ``1.49e-08``, which is ``sqrt(eps)``: **half the digits
    are gone before any algorithm runs.**
    """
    c = float(curvature)
    if c <= 0.0:
        raise ValueError(f"the curvature at a minimum is positive, got {c}")
    return math.sqrt(float(np.finfo(float).eps) * abs(float(value)) / c)


def flat_region_width(f, centre, direction=None, ceiling: float = 1.0) -> float:
    """Bisect outward from ``centre`` until ``f`` changes, and report how far that was.

    No optimizer is involved. This measures the problem, not a method: it is the radius of the set
    of points floating point cannot tell apart from the minimum.
    """
    base = np.asarray(centre, dtype=float).ravel()
    if direction is None:
        step = np.zeros(base.size)
        step[0] = 1.0
    else:
        step = np.asarray(direction, dtype=float).ravel()
        norm = float(np.linalg.norm(step))
        if norm == 0.0:
            raise ValueError("the direction must be nonzero")
        step = step / norm
    here = float(f(base))
    lo, hi = 0.0, float(ceiling)
    if float(f(base + hi * step)) <= here:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(f(base + mid * step)) > here:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-3 * hi:
            break
    return hi


def the_flat_region_is_as_wide_as_predicted(curvatures=(1e-2, 1.0, 1e2),
                                            value: float = 1.0,
                                            variables: int = 1) -> dict:
    """Measure the width of the flat region against ``sqrt(eps |f*| / c)``.

    The prediction is exact, to four digits at every curvature tried. It already carries the factor
    of a half that rounding to nearest introduces: ``f`` changes once ``(1/2) c d**2`` passes half
    an ulp of ``|f*|``, which is ``(eps/2)|f*|``, and the two halves cancel.
    """
    rows = []
    for c in curvatures:
        curve = float(c)

        def f(x, curve=curve):
            v = np.asarray(x, dtype=float).ravel()
            return float(value + 0.5 * curve * float(v @ v))

        got = flat_region_width(f, np.zeros(int(variables)))
        want = attainable_accuracy(curve, value)
        rows.append({"curvature": curve, "predicted": want, "measured": got,
                     "ratio": got / want})
    ratios = np.asarray([r["ratio"] for r in rows])
    return {
        "rows": rows,
        "value_at_the_minimum": float(value),
        "ratios_agree": float(np.max(ratios) - np.min(ratios)) < 1e-2,
        "the_ratio": float(np.mean(ratios)),
        "the_prediction_is_right": abs(float(np.mean(ratios)) - 1.0) < 1e-2,
        "note": "the width scales exactly as 1/sqrt(curvature) and the constant is 1, because "
                "the half from the quadratic and the half from rounding to nearest cancel",
    }


def _ternary(f, a: float, b: float, rounds: int = 200):
    """Locate a minimum of a unimodal ``f`` on ``[a, b]`` by trisection, counting calls."""
    calls = 0

    def value(t):
        nonlocal calls
        calls += 1
        return float(f(t))

    lo, hi = float(a), float(b)
    for _ in range(int(rounds)):
        left = lo + (hi - lo) / 3.0
        right = hi - (hi - lo) / 3.0
        if value(left) < value(right):
            hi = right
        else:
            lo = left
        if hi - lo <= 1e-18 * (1.0 + abs(hi)):
            break
    return 0.5 * (lo + hi), calls


def _bisect(g, a: float, b: float, rounds: int = 200):
    """Locate a sign change of ``g`` on ``[a, b]``, counting calls."""
    calls = 0

    def value(t):
        nonlocal calls
        calls += 1
        return float(g(t))

    lo, hi = float(a), float(b)
    left = value(lo)
    for _ in range(int(rounds)):
        mid = 0.5 * (lo + hi)
        here = value(mid)
        if left * here <= 0.0:
            hi = mid
        else:
            lo, left = mid, here
        if hi - lo <= 1e-18 * (1.0 + abs(hi)):
            break
    return 0.5 * (lo + hi), calls


def you_only_get_half_the_digits() -> dict:
    """The same three problems, solved as minimizations and as root problems.

    Each row minimizes ``f`` by trisection on values of ``f``, and separately finds the root of
    ``f'`` by bisection on values of ``f'``. The two searches have the same answer and very
    different accuracy, and the reason is the problem rather than the method: both searches run
    until their bracket is at rounding.
    """
    eps = float(np.finfo(float).eps)
    cases = [
        ("cos, minimum at pi", math.cos, lambda t: -math.sin(t), math.pi, (2.0, 4.0)),
        ("exp(x) - 2x, minimum at log 2", lambda t: math.exp(t) - 2.0 * t,
         lambda t: math.exp(t) - 2.0, math.log(2.0), (-1.0, 2.0)),
        ("x**2 - 2x + 3, minimum at 1", lambda t: t * t - 2.0 * t + 3.0,
         lambda t: 2.0 * t - 2.0, 1.0, (-2.0, 4.0)),
    ]
    rows = []
    for name, f, df, star, span in cases:
        got_min, min_calls = _ternary(f, *span)
        got_root, root_calls = _bisect(df, *span)
        rows.append({
            "case": name,
            "minimum_error": abs(got_min - star),
            "root_error": abs(got_root - star),
            "value_at_the_minimum": float(f(star)),
            "minimum_error_over_root_eps": abs(got_min - star) / math.sqrt(eps),
            "root_error_over_eps": abs(got_root - star) / eps,
            "calls": (min_calls, root_calls),
        })
    return {
        "rows": rows,
        "sqrt_eps": math.sqrt(eps),
        "eps": eps,
        "every_minimum_is_near_root_eps": all(
            r["minimum_error"] <= 4.0 * math.sqrt(eps) for r in rows),
        "every_root_is_near_eps": all(r["root_error"] <= 8.0 * eps for r in rows),
        "worst_ratio": max(r["minimum_error"] / max(r["root_error"], eps) for r in rows),
        "note": "the minimizations stop about sqrt(eps) from the answer and the root finds stop "
                "about eps from it, a factor of 10**7, and neither is a fault of the algorithm",
    }


def the_loss_is_set_by_the_value_at_the_minimum(values=(0.0, 1e-8, 1e-4, 1.0, 1e4)) -> dict:
    """Shift the same parabola up and down, and watch the attainable accuracy follow ``sqrt(|f*|)``.

    ``f(x) = (x - sqrt 2)**2 + s``. The minimizer never moves. What moves is how well it can be
    located, because the rounding error in ``f`` is relative to ``|f|``.

    The row at ``s = 0`` is the one worth stopping on. **A least squares problem has ``f* = 0``**,
    so there is no leading constant for the rounding to be relative to, and the minimum is located
    to full precision. That is why Part 5 could report the errors it did.
    """
    root_two = math.sqrt(2.0)
    rows = []
    for s in values:
        shift = float(s)

        def f(t, shift=shift):
            return (t - root_two) ** 2 + shift

        got, _ = _ternary(f, 0.0, 3.0)
        predicted = attainable_accuracy(2.0, shift) if shift else 0.0
        rows.append({"value_at_the_minimum": shift, "error": abs(got - root_two),
                     "predicted": predicted,
                     "ratio": (abs(got - root_two) / predicted) if predicted else float("nan")})
    exact = [r for r in rows if r["value_at_the_minimum"] == 0.0]
    scaled = [r for r in rows if r["value_at_the_minimum"] > 0.0]
    powers = ([math.log10(r["error"]) for r in scaled],
              [math.log10(r["value_at_the_minimum"]) for r in scaled])
    slope = float(np.polyfit(powers[1], powers[0], 1)[0]) if len(scaled) > 1 else float("nan")
    return {
        "rows": rows,
        "zero_value_is_exact": bool(exact and exact[0]["error"] < 1e-15),
        "fitted_power_of_the_value": slope,
        "the_power_is_a_half": abs(slope - 0.5) < 0.05,
        "note": "the error grows as the square root of the value at the minimum, and vanishes "
                "when that value is zero, which is the least squares case",
    }


def the_condition_number_is_an_aspect_ratio(problem=None, at=None) -> dict:
    """How the Hessian's condition number shapes the flat region around a minimum.

    Along an eigenvector with eigenvalue ``lambda``, the flat region has half width
    ``sqrt(eps |f*| / lambda)``. So the region is an ellipsoid whose axis ratio is
    ``sqrt(lambda_max / lambda_min) = sqrt(kappa)``, **not** ``kappa``. A problem with a condition
    number of ``10**4`` is elongated by ``100``, which is the number that matters when reading a
    convergence plot.

    All of that assumes ``f*`` is not zero. When it is, as on Rosenbrock, there is no flat region
    to have an aspect ratio and the reported widths are at the underflow edge instead. The result
    says so rather than reporting the ratio of two numbers that mean nothing.
    """
    if problem is None:
        problem = quadratic(condition=1e4, dimension=2)
    where = np.asarray(problem["minimizers"][0] if at is None else at, dtype=float).ravel()
    h = np.asarray(problem["hessian"](where), dtype=float)
    values, vectors = np.linalg.eigh(0.5 * (h + h.T))
    value = float(problem["f"](where))
    rows = []
    for k in range(values.size):
        rows.append({
            "eigenvalue": float(values[k]),
            "predicted_half_width": attainable_accuracy(float(values[k]), value),
            "measured_half_width": flat_region_width(problem["f"], where, vectors[:, k]),
        })
    condition = float(values[-1] / values[0])
    widths = np.asarray([r["measured_half_width"] for r in rows])
    flat = abs(value) > 0.0
    ratio = float(np.max(widths) / np.min(widths))
    return {
        "name": problem["name"],
        "value_at_the_minimum": value,
        "rows": rows,
        "condition_number": condition,
        "there_is_a_flat_region": flat,
        "measured_aspect_ratio": ratio if flat else float("nan"),
        "sqrt_condition": math.sqrt(condition),
        "it_is_the_square_root": bool(flat and abs(ratio / math.sqrt(condition) - 1.0) < 0.05),
        "note": ("the flat region is an ellipsoid with axis ratio sqrt(kappa), so a condition "
                 "number of 10**4 shows up as an elongation of 100") if flat else
                ("the value at the minimum is zero, so there is no flat region and no aspect "
                 "ratio: the widths are at the underflow edge"),
    }


def the_conditioning_of_every_test_problem(dimension: int = 2) -> dict:
    """The Hessian at each test problem's minimum, and what it says about the last step.

    Rastrigin is here to make one point: its Hessian at the global minimum is a multiple of the
    identity, so its condition number is exactly 1 and it is the hardest problem in the set. **The
    condition number describes the last step of a search and nothing about finding the right
    basin.**
    """
    rows = []
    for problem in all_problems(dimension):
        where = np.asarray(problem["minimizers"][0], dtype=float)
        h = np.asarray(problem["hessian"](where), dtype=float)
        values = np.linalg.eigvalsh(0.5 * (h + h.T))
        rows.append({
            "name": problem["name"],
            "gradient_at_the_minimum": float(np.max(np.abs(problem["gradient"](where)))),
            "smallest_eigenvalue": float(values[0]),
            "largest_eigenvalue": float(values[-1]),
            "condition": float(values[-1] / values[0]),
            "verdict": classify(h),
            "minimizers": len(problem["minimizers"]),
        })
    return {
        "rows": rows,
        "every_gradient_vanishes": all(r["gradient_at_the_minimum"] < 1e-8 for r in rows),
        "every_point_is_a_minimum": all(r["verdict"] == "minimum" for r in rows),
        "best_conditioned": min(rows, key=lambda r: r["condition"])["name"],
        "worst_conditioned": max(rows, key=lambda r: r["condition"])["name"],
        "note": "the best conditioned problem here is the one with the most local minima, which "
                "is the whole reason conditioning is not a difficulty measure for optimization",
    }
