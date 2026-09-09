"""Consistency, stability and convergence: the three properties and the theorem that ties them.

The one theorem
---------------
For a well posed linear initial value problem and a consistent difference scheme,

    stability  <=>  convergence,

which is the Lax equivalence theorem. It is the reason stability is worth so much attention:
consistency is a Taylor expansion and easy to check, convergence is what you actually want and
hard to check, and stability is the bridge.

Each of the three is a separate property and this module measures each one separately.

**Consistency** is a statement about the exact solution: put it into the scheme and see how fast
the residual goes to zero. `truncation_order` does that by finite differences on the known exact
solution, so nothing is assumed about the scheme beyond being able to run it.

**Stability** is a statement about the scheme alone. There are two standard tests and they answer
slightly different questions:

* the **matrix method** builds the one step operator ``G`` for the actual grid and the actual
  boundary conditions and asks whether powers of it stay bounded,
* **von Neumann analysis** substitutes a single Fourier mode and asks whether the resulting scalar
  has modulus at most 1. It assumes constant coefficients and it ignores the boundaries.

**Convergence** is measured against the exact solution, which is the only one of the three that
needs the answer.

What the measurements here show
-------------------------------
* For the heat equation the two stability tests agree **exactly**, not approximately.
  ``G`` is symmetric, so its eigenvalues are its singular values, and they are precisely the von
  Neumann symbols evaluated at the grid's own phases. `the_two_tests_agree_on_the_heat_equation`
  measures the largest gap and it is at the rounding level.
* Adding convection breaks the symmetry, and then the two stop being the same statement, though
  not in the way the textbook warning suggests. `the_spectral_radius_is_only_the_limit` searched
  the whole stable region and found **no transient growth anywhere**: the 2-norm of the one step
  operator is always just below 1, so the powers never rise. What does happen is that the decay
  **rate** takes hundreds of steps to become the spectral radius. On a normal matrix the observed
  rate is the spectral radius from the first step; with convection at ``Pe = 1.9`` the observed
  rate is still 0.9995 after 20 steps for a matrix whose spectral radius is 0.594, and it takes
  244 steps to come within 10 per cent of it.
* Lax's theorem is checked in the only way it can be, by exhibiting all three cases.
  `lax_in_three_runs` runs one consistent stable scheme, one consistent unstable scheme, and one
  stable inconsistent scheme, and shows that only the first converges.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from . import parabolic as pb


# --------------------------------------------------------------------------- consistency


def truncation_residual(problem, points: int, steps: int, t_end: float, theta: float = 0.0):
    """Put the exact solution into the scheme and return the leftover, divided by ``k``.

    Consistency says this goes to zero as the grid is refined. Dividing by ``k`` is what makes it
    the residual of the **differential** equation rather than of the difference equation, which is
    the convention that makes the order come out as the order of the method.

    The exact solution is evaluated at the grid points, so nothing here depends on the scheme
    being able to solve anything: this is a property of the formula and the equation, measured
    without ever running the scheme forward.
    """
    n = int(points)
    steps = int(steps)
    x = np.linspace(problem["a"], problem["b"], n)
    h = float(x[1] - x[0])
    k = float(t_end) / steps
    r = pb.mesh_ratio(problem["alpha"], k, h)
    worst = 0.0
    for step in range(steps):
        now = np.asarray(problem["exact"](x, step * k), dtype=float)
        nxt = np.asarray(problem["exact"](x, (step + 1) * k), dtype=float)
        old_d2 = now[2:] - 2.0 * now[1:-1] + now[:-2]
        new_d2 = nxt[2:] - 2.0 * nxt[1:-1] + nxt[:-2]
        residual = ((nxt[1:-1] - now[1:-1])
                    - r * (theta * new_d2 + (1.0 - theta) * old_d2)) / k
        worst = max(worst, float(np.max(np.abs(residual))))
    return {"h": h, "k": k, "r": r, "residual": worst, "steps": steps, "theta": float(theta)}


def truncation_order(problem, refinements=(11, 21, 41, 81), t_end: float = 0.05,
                     theta: float = 0.0, ratio: float = 0.4):
    """Fit the order of the truncation error, in ``h`` and in ``k``.

    At fixed ``r`` the two are tied, ``k = r h**2 / alpha``, so an order of ``p`` in ``h``
    corresponds to ``p/2`` in ``k``. Both are reported because the literature quotes both and
    which one is meant is often left to the reader.
    """
    hs, ks, residuals = [], [], []
    for n in refinements:
        m = int(n)
        h = (problem["b"] - problem["a"]) / (m - 1)
        k = float(ratio) * h ** 2 / problem["alpha"]
        steps = max(int(round(float(t_end) / k)), 1)
        out = truncation_residual(problem, m, steps, steps * k, theta=theta)
        hs.append(out["h"])
        ks.append(out["k"])
        residuals.append(out["residual"])
    hs, ks, residuals = np.asarray(hs), np.asarray(ks), np.asarray(residuals)
    return {
        "h": hs, "k": ks, "residual": residuals,
        "order_in_h": float(np.polyfit(np.log(hs), np.log(residuals), 1)[0]),
        "order_in_k": float(np.polyfit(np.log(ks), np.log(residuals), 1)[0]),
        "theta": float(theta), "ratio": float(ratio),
        "consistent": bool(residuals[-1] < residuals[0]),
    }


def consistency_of_the_family(thetas=(0.0, 0.25, 0.5, 0.75, 1.0), ratio: float = 0.4):
    """Measure every member's truncation constant, and check it against the formula.

    Expanding the weighted scheme about ``(x_j, t_n + k/2)`` and using ``u_tt = alpha**2 u_xxxx``
    on a solution of the equation gives one expression for the whole family:

        T = alpha h**2 * [ (1/2 - theta) r - 1/12 ] * u_xxxx + O(h**4) .

    Two things follow, and both are measured here.

    First, **the constant is predicted exactly**, and the measurement agrees to three digits at
    every ``theta``. The bracket is the only thing that changes, and multiplying it by
    ``alpha * max|u_xxxx| = alpha (m pi)**4`` for the sine problem gives the number.

    Second, and less expected, **the smallest constant in the family is not Crank-Nicolson's**.
    The bracket vanishes at

        theta = 1/2 - 1/(12 r) ,

    which at ``r = 0.4`` is ``0.2917``, and Crank-Nicolson's constant there is five times larger
    than the best member's. Crank-Nicolson is the member whose ``k`` term vanishes for **every**
    ``r``, which is a different and more useful property, but at a fixed ``r`` it is not the most
    accurate one. Lesson 78's ``r = 1/6`` result is the same statement read the other way: at
    ``theta = 0`` the bracket vanishes when ``r = 1/6``.
    """
    problem = pb.sine_problem()
    rr = float(ratio)
    fourth = 0.5 - 1.0 / (12.0 * rr)
    sweep = sorted({float(t) for t in thetas} | {fourth})
    rows = []
    for theta in sweep:
        out = truncation_order(problem, theta=float(theta), ratio=rr)
        bracket = (0.5 - float(theta)) * rr - 1.0 / 12.0
        predicted = abs(bracket) * problem["alpha"] * math.pi ** 4
        measured = float(out["residual"][-1] / out["h"][-1] ** 2)
        rows.append({
            "theta": float(theta),
            "order_in_h": out["order_in_h"],
            "order_in_k": out["order_in_k"],
            "residual": out["residual"],
            "constant": measured,
            "predicted_constant": predicted,
            "agrees": abs(measured - predicted) <= 0.01 * max(predicted, 1e-12) + 1e-9,
            "is_the_cancelling_theta": abs(bracket) < 1e-12,
        })
    ordinary = [row for row in rows if not row["is_the_cancelling_theta"]]
    best = min(ordinary, key=lambda row: row["constant"])
    nicolson = next((row for row in rows if abs(row["theta"] - 0.5) < 1e-12), None)
    return {
        "rows": rows,
        "every_member_is_consistent": all(
            abs(row["order_in_h"] - 2.0) < 0.2 for row in ordinary),
        "the_formula_predicts_every_constant": all(row["agrees"] for row in ordinary),
        "cancelling_theta": fourth,
        "smallest_constant_at_theta": best["theta"],
        "crank_nicolson_is_not_the_most_accurate_at_this_ratio": (
            nicolson is not None and best["theta"] != nicolson["theta"]),
        "crank_nicolson_over_the_best": (
            float(nicolson["constant"] / best["constant"]) if nicolson else float("nan")),
        "order_at_the_cancelling_theta": next(
            row["order_in_h"] for row in rows if row["is_the_cancelling_theta"]),
        "ratio": rr,
    }


# --------------------------------------------------------------------------- the matrix method


def amplification_matrix(theta: float, r: float, unknowns: int):
    """The one step operator ``G`` with ``u^{n+1} = G u^n``, for Dirichlet ends.

    Built explicitly as ``(I + theta r T)^{-1} (I - (1-theta) r T)`` with ``T`` the second
    difference matrix. Explicit inversion is the wrong way to **run** the scheme and the right way
    to **study** it: the object of interest is the operator itself, and its powers.
    """
    m = int(unknowns)
    if m < 1:
        raise ValueError(f"need at least one unknown, got {m}")
    th, rr = float(theta), float(r)
    t = (np.diag(np.full(m, -2.0))
         + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)) if m > 1 \
        else np.asarray([[-2.0]])
    left = np.eye(m) - th * rr * t
    right = np.eye(m) + (1.0 - th) * rr * t
    return np.linalg.solve(left, right)


def matrix_method(theta: float, r: float, unknowns: int, powers=(1, 2, 5, 10, 20, 50)):
    """Spectral radius, 2-norm, and the norms of the powers.

    The spectral radius governs the **limit**: ``||G**n||`` behaves like ``rho**n`` eventually,
    for any matrix. The 2-norm governs the **first step**. When ``G`` is normal the two are equal
    and there is nothing more to say; when it is not, the powers can grow before they decay, and
    only the norms of the powers show it.
    """
    g = amplification_matrix(theta, r, unknowns)
    values = np.linalg.eigvals(g)
    rho = float(np.max(np.abs(values)))
    norm = float(np.linalg.norm(g, 2))
    growth = []
    current = np.eye(g.shape[0])
    seen = 0
    for p in sorted(int(v) for v in powers):
        while seen < p:
            current = current @ g
            seen += 1
        growth.append((p, float(np.linalg.norm(current, 2))))
    normality = float(np.linalg.norm(g @ g.T - g.T @ g, 2))
    return {
        "spectral_radius": rho,
        "two_norm": norm,
        "eigenvalues": values,
        "power_norms": growth,
        "peak_growth": max(v for _p, v in growth),
        "normal": normality <= 1e-10 * max(norm ** 2, 1.0),
        "departure_from_normality": normality,
        "stable_in_the_limit": rho <= 1.0 + 1e-12,
        "monotone": norm <= 1.0 + 1e-12,
    }


def von_neumann_symbols(theta: float, r: float, unknowns: int):
    """The growth factor evaluated at the phases a grid of this size actually carries.

    A Dirichlet grid with ``m`` unknowns carries the modes ``sin(j pi i /(m+1))`` for
    ``j = 1 ... m``, so the phases are ``j pi / (m+1)``. Those are the only ones present, which is
    why the continuous von Neumann sweep over all ``phi`` is conservative rather than exact.
    """
    m = int(unknowns)
    phases = np.arange(1, m + 1) * math.pi / (m + 1)
    return pb.growth_factor(float(r), phases, float(theta))


def the_two_tests_agree_on_the_heat_equation(thetas=(0.0, 0.5, 1.0),
                                             ratios=(0.2, 0.4, 0.6, 2.0),
                                             unknowns=(5, 9, 17, 33)):
    """For this equation the matrix method and von Neumann are the same statement, exactly.

    The second difference matrix is symmetric, so ``G`` is a rational function of a symmetric
    matrix and is symmetric too. Its eigenvalues are therefore its singular values, spectral
    radius equals 2-norm, and there is no transient growth to worry about at all.

    Better than that: the eigenvectors of the second difference matrix are exactly the discrete
    sine modes, so ``G``'s eigenvalues are exactly the von Neumann symbols at the grid's phases.
    The measurement is a comparison of two sorted lists, and the gap is at the rounding level.

    This is why so much of the literature treats the two tests as interchangeable. They are, for
    this problem, and `a_non_normal_matrix_grows_before_it_decays` is what happens when they are
    not.
    """
    rows = []
    for theta in thetas:
        for r in ratios:
            for m in unknowns:
                out = matrix_method(float(theta), float(r), int(m))
                symbols = von_neumann_symbols(float(theta), float(r), int(m))
                gap = float(np.max(np.abs(np.sort(np.real(out["eigenvalues"]))
                                          - np.sort(symbols))))
                rows.append({
                    "theta": float(theta), "r": float(r), "unknowns": int(m),
                    "spectral_radius": out["spectral_radius"],
                    "two_norm": out["two_norm"],
                    "eigenvalue_gap": gap,
                    "normal": out["normal"],
                    "radius_equals_norm": abs(out["spectral_radius"] - out["two_norm"])
                    < 1e-10 * max(out["two_norm"], 1.0),
                })
    return {
        "rows": rows,
        "cases": len(rows),
        "every_matrix_is_normal": all(row["normal"] for row in rows),
        "radius_equals_norm_everywhere": all(row["radius_equals_norm"] for row in rows),
        "worst_eigenvalue_gap": max(row["eigenvalue_gap"] for row in rows),
        "the_two_tests_are_the_same_statement": max(
            row["eigenvalue_gap"] for row in rows) < 1e-10,
        "and_they_both_say_the_same_about_stability": all(
            (row["spectral_radius"] <= 1.0 + 1e-12)
            == (row["r"] <= pb.stability_limit(row["theta"]) + 1e-12) for row in rows),
    }


# --------------------------------------------------------------------------- non-normality


def convection_diffusion_matrix(peclet: float, r: float, unknowns: int, upwind: bool = False):
    """The explicit scheme for ``u_t + a u_x = alpha u_xx``, as a one step matrix.

    ``peclet`` is the cell Peclet number ``a h / alpha``, which is lesson 75's cell number. The
    convection term is differenced centrally by default, which makes the matrix **not symmetric**,
    and that is the whole point of this function.
    """
    m = int(unknowns)
    if m < 2:
        raise ValueError(f"need at least two unknowns, got {m}")
    rr, pe = float(r), float(peclet)
    g = np.eye(m) + rr * (np.diag(np.full(m, -2.0))
                          + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1))
    if upwind:
        # a > 0: the value comes from the left, so difference backwards
        g += rr * pe * (np.diag(np.ones(m - 1), -1) - np.eye(m))
    else:
        g += rr * pe * 0.5 * (np.diag(np.ones(m - 1), -1) - np.diag(np.ones(m - 1), 1))
    return g


def convection_symbol_is_stable(peclet: float, r: float) -> bool:
    """Von Neumann stability of the explicit convection-diffusion scheme.

    Substituting one mode gives ``g = 1 - 4 r s - i r Pe sin(phi)`` with ``s = sin(phi/2)**2``,
    and ``|g| <= 1`` for every ``phi`` reduces to two conditions:

        r <= 1/2        (the diffusion limit, from s = 1)
        r * Pe**2 <= 2  (the convection limit, from s -> 0)

    The second is the one that bites in a convection dominated problem: at ``Pe = 8`` it caps
    ``r`` at ``0.031``, sixteen times below the diffusion limit.
    """
    return float(r) <= 0.5 + 1e-12 and float(r) * float(peclet) ** 2 <= 2.0 + 1e-12


def the_spectral_radius_is_only_the_limit(cases=((0.0, 0.4), (1.0, 0.4), (1.9, 0.4),
                                                 (4.0, 0.1125), (2.0, 0.45)),
                                          unknowns: int = 60, horizon: int = 400):
    """What non-normality costs here, measured rather than assumed.

    Central differencing of a convection term makes the one step matrix **not normal**, and the
    standard warning is that a non-normal operator with spectral radius below 1 can still have
    powers that grow first. That warning is worth checking rather than repeating, and on this
    scheme it does not happen: sweeping the whole stable region, the 2-norm of ``G`` comes out
    just **below** 1 every time, so the powers decrease monotonically from the first step. There
    is no transient growth to find. That is a negative result and it is reported as one.

    What non-normality does cost is the **rate**. The spectral radius is an asymptotic statement,
    ``||G**n||**(1/n) -> rho``, and how long that takes is exactly what normality decides:

    * pure diffusion, ``Pe = 0``: the matrix is symmetric, and the observed rate equals ``rho``
      at ``n = 1``,
    * ``Pe = 1.9``: ``rho = 0.594``, and the observed rate is still 0.9995 at ``n = 20``. It takes
      244 steps to come within 10 per cent of ``rho``,
    * ``Pe = 2`` at ``r = 0.45``: ``rho = 0.100``, and 400 steps are not enough to get there.

    So the spectral radius is right about where the solution ends up and can be badly wrong about
    how fast it gets there, which is the practical content of "stability is not the whole story".
    Only cases inside the von Neumann region are run, so every row is a stable scheme.
    """
    rows = []
    for peclet, r in cases:
        pe, rr = float(peclet), float(r)
        if not convection_symbol_is_stable(pe, rr):
            raise ValueError(f"Pe = {pe}, r = {rr} is outside the stability region")
        g = (convection_diffusion_matrix(pe, rr, unknowns) if pe != 0.0
             else amplification_matrix(0.0, rr, unknowns))
        rho = float(np.max(np.abs(np.linalg.eigvals(g))))
        norms, current = [], np.eye(g.shape[0])
        for _ in range(int(horizon)):
            current = current @ g
            norms.append(float(np.linalg.norm(current, 2)))
        norms = np.asarray(norms)
        steps = np.arange(1, norms.size + 1)
        rate = norms ** (1.0 / steps)
        close = np.flatnonzero(rate < 1.1 * rho)
        rows.append({
            "peclet": pe,
            "r": rr,
            "spectral_radius": rho,
            "two_norm": float(norms[0]),
            "peak_power_norm": float(np.max(norms)),
            "grows_at_any_point": bool(np.max(norms) > 1.0 + 1e-9),
            "monotone": bool(np.all(np.diff(norms) <= 1e-12)),
            "rate_at_1": float(rate[0]),
            "rate_at_20": float(rate[min(19, rate.size - 1)]),
            "rate_at_the_end": float(rate[-1]),
            "steps_to_the_asymptotic_rate": int(close[0] + 1) if close.size else -1,
            "departure_from_normality": float(np.linalg.norm(g @ g.T - g.T @ g, 2)),
            "normal": float(np.linalg.norm(g @ g.T - g.T @ g, 2)) < 1e-10,
            "power_norms": norms,
        })
    diffusion = rows[0]
    return {
        "rows": rows,
        "unknowns": int(unknowns),
        "horizon": int(horizon),
        "nothing_grows_anywhere": not any(row["grows_at_any_point"] for row in rows),
        "every_run_is_monotone": all(row["monotone"] for row in rows),
        "the_normal_one_hits_its_rate_at_once":
            diffusion["normal"] and diffusion["steps_to_the_asymptotic_rate"] == 1,
        "the_others_take_longer": all(
            row["steps_to_the_asymptotic_rate"] != 1 for row in rows[1:]),
        "worst_delay": max(row["steps_to_the_asymptotic_rate"] for row in rows),
        "some_never_get_there": any(
            row["steps_to_the_asymptotic_rate"] < 0 for row in rows),
        "note": "no transient growth was found anywhere in the stable region; what "
                "non-normality costs on this scheme is the time it takes for the spectral "
                "radius to describe the decay",
    }


# --------------------------------------------------------------------------- Lax


def lax_in_three_runs(refinements=(21, 41, 81, 161), t_end: float = 0.05):
    """The theorem needs all three cases to mean anything, so all three are run.

    Lax says: for a well posed linear problem, a **consistent** scheme converges **if and only
    if** it is stable. Checking that on one scheme proves nothing, because a scheme that has both
    properties would converge under any theorem. What the statement forbids is the other two
    combinations, and both are run here:

    * **consistent and stable**: the explicit scheme at ``r = 0.4``. Converges, at order 2.
    * **consistent and unstable**: the explicit scheme at ``r = 0.6``. Diverges, and the
      divergence gets worse as the grid is refined, which is the opposite of convergence.
    * **stable and not consistent**: Du Fort and Frankel at fixed ``k/h``. Every run is finite and
      smooth and the error settles on a nonzero limit, so it converges to something that is not
      the solution.

    The third is the one worth the attention. It is not that the scheme fails to converge: it
    converges, beautifully, to the wrong answer, and only comparing against the exact solution
    catches it.
    """
    problem = pb.sine_problem()
    rough = pb.step_problem()
    cases = []

    for label, r in (("consistent and stable", 0.4), ("consistent and unstable", 0.6)):
        errors, hs = [], []
        for n in refinements:
            m = int(n)
            h = (problem["b"] - problem["a"]) / (m - 1)
            k = r * h ** 2 / problem["alpha"]
            steps = max(int(round(float(t_end) / k)), 1)
            with np.errstate(over="ignore", invalid="ignore"):
                run = pb.solve(rough, m, steps, steps * k, theta=0.0)
            errors.append(run["error"])
            hs.append(h)
        errors = np.asarray(errors)
        finite = np.isfinite(errors)
        cases.append({
            "case": label,
            "r": r,
            "h": np.asarray(hs),
            "error": errors,
            "consistent": True,
            "stable": r <= pb.stability_limit(0.0),
            "converges": bool(np.all(finite) and errors[-1] < errors[0]),
            "gets_worse_when_refined": bool(np.all(finite) and errors[-1] > errors[0])
            or not bool(np.all(finite)),
        })

    errors, hs = [], []
    for n in refinements:
        m = int(n)
        h = (problem["b"] - problem["a"]) / (m - 1)
        k = 0.1 * h
        steps = max(int(round(float(t_end) / k)), 1)
        run = pb.dufort_frankel(problem, m, steps, steps * k)
        errors.append(run["error"])
        hs.append(h)
    errors = np.asarray(errors)
    cases.append({
        "case": "stable but not consistent",
        "r": float("nan"),
        "h": np.asarray(hs),
        "error": errors,
        "consistent": False,
        "stable": True,
        "converges": bool(errors[-1] < 0.5 * errors[0]),
        "settles_on": float(errors[-1]),
        "gets_worse_when_refined": False,
    })
    return {
        "cases": cases,
        "only_the_first_converges": (cases[0]["converges"] and not cases[1]["converges"]
                                     and not cases[2]["converges"]),
        "the_unstable_one_gets_worse_when_refined": cases[1]["gets_worse_when_refined"],
        "the_inconsistent_one_settles_on": cases[2]["settles_on"],
        "note": "consistency plus stability is convergence; drop either one and the run still "
                "produces numbers",
    }


# --------------------------------------------------------------------------- building a scheme


def scheme_order(weights, r: float):
    """The order in space and time of a two level scheme given as ``(old, new)`` weight rows.

    The scheme is ``sum_j new_j u_{i+j}^{n+1} = sum_j old_j u_{i+j}^n`` on offsets centred on
    zero. Substituting a Taylor expansion of a solution of ``u_t = alpha u_xx`` and collecting
    powers of ``h`` gives the order, and this does that numerically by applying the scheme to
    exact polynomial-in-``x``, exponential-in-``t`` solutions and measuring what survives.

    Returns the largest ``p`` for which the scheme is exact on every solution of the form
    ``exp(-alpha m**2 pi**2 t) sin(m pi x)`` to ``O(h**p)``, measured by refinement rather than
    by symbolic expansion, so an arbitrary set of weights can be checked.
    """
    old = np.asarray(weights[0], dtype=float).ravel()
    new = np.asarray(weights[1], dtype=float).ravel()
    if old.size != new.size or old.size % 2 == 0:
        raise ValueError("both weight rows must have the same odd length")
    half = old.size // 2
    offsets = np.arange(-half, half + 1)
    problem = pb.sine_problem()
    residuals, hs = [], []
    for m in (21, 41, 81, 161):
        h = (problem["b"] - problem["a"]) / (m - 1)
        k = float(r) * h ** 2 / problem["alpha"]
        x = np.linspace(problem["a"], problem["b"], m)
        now = problem["exact"](x, 0.0)
        nxt = problem["exact"](x, k)
        inner = slice(half, m - half)
        total = np.zeros(m - 2 * half)
        for w, s in zip(new, offsets):
            total = total + w * nxt[half + s: m - half + s]
        for w, s in zip(old, offsets):
            total = total - w * now[half + s: m - half + s]
        residuals.append(float(np.max(np.abs(total))) / k)
        hs.append(h)
        del inner
    hs, residuals = np.asarray(hs), np.asarray(residuals)
    return {
        "h": hs, "residual": residuals,
        "order": float(np.polyfit(np.log(hs), np.log(residuals), 1)[0]),
        "consistent": bool(residuals[-1] < 0.5 * residuals[0]),
    }


def symbol_of(weights, phases):
    """The amplification factor of a two level scheme, from its weight rows.

    For ``sum_j new_j u_{i+j}^{n+1} = sum_j old_j u_{i+j}^n`` the substitution
    ``u_i^n = g**n exp(i i phi)`` gives ``g = P_old(phi) / P_new(phi)`` with each ``P`` the
    trigonometric polynomial of that row. A vanishing denominator is not instability but a scheme
    that cannot be solved at that mode at all, so it is reported separately.
    """
    old = np.asarray(weights[0], dtype=float).ravel()
    new = np.asarray(weights[1], dtype=float).ravel()
    if old.size != new.size or old.size % 2 == 0:
        raise ValueError("both weight rows must have the same odd length")
    half = old.size // 2
    offsets = np.arange(-half, half + 1)
    phi = np.atleast_1d(np.asarray(phases, dtype=float))
    top = np.sum(old[None, :] * np.exp(1j * np.outer(phi, offsets)), axis=1)
    bottom = np.sum(new[None, :] * np.exp(1j * np.outer(phi, offsets)), axis=1)
    return top, bottom


def build_your_own_scheme(r: float = 0.25, samples: int = 801):
    """Design a scheme by consistency, then find out about its stability afterwards.

    The design space here is the three point, two level scheme

        d u_{i-1}^{n+1} + e u_i^{n+1} + f u_{i+1}^{n+1}
            = a u_{i-1}^n + b u_i^n + c u_{i+1}^n .

    Consistency with ``u_t = alpha u_xx`` is a set of **linear** conditions on the six weights:
    both rows must sum to the same thing (exact on constants), the first moments must match
    (exact on ``x``), and the second moments must differ by ``2 alpha k / h**2 = 2r`` (this is
    where the equation enters). Those conditions can always be met, and the solutions are exactly
    the weighted family of lesson 78 with ``theta`` free.

    **Stability is not a linear condition and cannot be arranged the same way.** It has to be
    checked afterwards, and it depends on ``r`` as well as on the weights. The table here builds
    five schemes and measures both properties independently:

    * three members of the family at a stable ``r``, all consistent and all stable,
    * the explicit scheme at ``r = 0.75``, consistent to exactly the same order and **unstable**,
    * a scheme whose second moment condition is wrong by half, which is stable, well behaved, and
      solves ``u_t = 1.5 alpha u_xx``.

    The last two are the point. Consistency and stability are independent, which is why the Lax
    theorem needs both, and neither one is visible in the other's test.
    """
    rr = float(r)
    unstable_r = 0.75
    fourth = 0.5 - 1.0 / (12.0 * rr)

    def family(theta, ratio):
        th = float(theta)
        return (np.asarray([(1.0 - th) * ratio, 1.0 - 2.0 * (1.0 - th) * ratio,
                            (1.0 - th) * ratio]),
                np.asarray([-th * ratio, 1.0 + 2.0 * th * ratio, -th * ratio]))

    designs = [
        ("explicit, r = %.2f" % rr, family(0.0, rr), rr),
        ("Crank-Nicolson, r = %.2f" % rr, family(0.5, rr), rr),
        ("theta = %.4f, r = %.2f" % (fourth, rr), family(fourth, rr), rr),
        ("explicit, r = %.2f" % unstable_r, family(0.0, unstable_r), unstable_r),
        ("second moment wrong by half",
         (np.asarray([1.5 * rr, 1.0 - 3.0 * rr, 1.5 * rr]), np.asarray([0.0, 1.0, 0.0])), rr),
    ]
    phases = np.linspace(0.0, math.pi, int(samples))
    rows = []
    for name, weights, ratio in designs:
        old, new = weights
        offsets = np.arange(-(old.size // 2), old.size // 2 + 1)
        # expanding both sides about (x_i, t_n) and matching u, u_x and u_xx in turn
        moment0 = float(np.sum(new) - np.sum(old))
        moment1 = float(np.sum(new * offsets) - np.sum(old * offsets))
        moment2 = float((np.sum(old * offsets ** 2) - np.sum(new * offsets ** 2))
                        / np.sum(new))
        order = scheme_order(weights, ratio)
        top, bottom = symbol_of(weights, phases)
        singular = bool(np.any(np.abs(bottom) < 1e-12))
        g = np.abs(top / bottom) if not singular else np.full(phases.shape, math.inf)
        rows.append({
            "scheme": name,
            "r": ratio,
            "constant_condition": moment0,
            "first_moment_condition": moment1,
            "second_moment_condition": moment2,
            "second_moment_should_be": 2.0 * ratio,
            "consistent": (abs(moment0) < 1e-12 and abs(moment1) < 1e-12
                           and abs(moment2 - 2.0 * ratio) < 1e-12),
            "measured_order": order["order"],
            "residual_falls": order["consistent"],
            "largest_growth": float(np.max(g)),
            "stable": bool(np.max(g) <= 1.0 + 1e-12),
            "solvable": not singular,
            "worst_phase": float(phases[int(np.argmax(g))]) if not singular else math.nan,
        })
    consistent = [row for row in rows if row["consistent"]]
    inconsistent = [row for row in rows if not row["consistent"]]
    return {
        "rows": rows,
        "r": rr,
        "cancelling_theta": fourth,
        "the_conditions_are_linear_and_are_met_by_four_of_five":
            len(consistent) == len(rows) - 1,
        "a_consistent_scheme_can_be_unstable": any(
            not row["stable"] for row in consistent),
        "an_inconsistent_scheme_can_be_stable": any(row["stable"] for row in inconsistent),
        "consistency_and_stability_are_independent": (
            any(not row["stable"] for row in consistent)
            and any(row["stable"] for row in inconsistent)),
        "the_wrong_second_moment_still_converges_to_something": all(
            not row["residual_falls"] for row in inconsistent),
        "every_scheme_is_solvable": all(row["solvable"] for row in rows),
    }
