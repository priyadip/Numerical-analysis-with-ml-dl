"""Random walks, Brownian motion, and solving differential equations that have noise in them.

Where the work is
-----------------
Part 10 solved ``y' = f(t, y)``. This module solves

    dX = a(t, X) dt + b(t, X) dW ,

with ``W`` a Brownian motion. Nothing about that is a small change to the previous problem, and
three things in particular have to be restated.

**A Brownian path is not differentiable.** Its increment over a step of size ``h`` is of order
``sqrt(h)``, not ``h``, so every Taylor expansion in Part 10 has to be redone with ``dW**2 = dt``.
That single rule is Ito calculus, and it is where the extra term in the chain rule comes from.

**There are two orders of accuracy, not one.** **Strong** order measures the error of the path
against the same path of the exact solution, driven by the same noise. **Weak** order measures the
error of an expectation, where the individual paths need not match at all. Euler-Maruyama has strong
order ``1/2`` and weak order ``1``, and a method chosen for the wrong one of those is the wrong
method.

**The step size controls something new.** In Part 10 a smaller step meant a better approximation of
one curve. Here it also means more noise increments, so the estimate of an expectation has a Monte
Carlo error of lesson 92's ``1/sqrt(M)`` on top of the discretization error, and the two have to be
balanced.

What the measurements here show
-------------------------------
* **The scaled random walk converges to Brownian motion**, with the measured variance at time 1
  inside 2.6 per cent of 1 at 64 steps and 0.3 per cent at 16384, and the increments uncorrelated to
  0.002.
* **The quadratic variation of a Brownian path is the elapsed time**, measured at 1.006 against 1,
  while the same sum on a differentiable path falls like ``n**-1.0000``. That one contrast is
  ``dW**2 = dt``, and therefore the whole of Ito calculus, stated as a measurement.
* **Euler-Maruyama has strong order 1/2 and weak order 1**, fitted at 0.4974 and 0.9950 on the same
  problem, which is the cleanest way to see that the two are different questions.
* **Milstein has strong order 1** and its extra term is one line, fitted at 0.9912 against
  Euler-Maruyama's 0.4974 on the same problem and the same noise, a factor of 31 in the error.
* **Milstein is not better in the weak sense.** Both are weak order 1, at 0.9950 and 0.9923, and the
  measured weak gain is **0.970**, so the extra term buys nothing at all when only an expectation is
  wanted. That decides which method to use in practice, and it is the opposite of the usual instinct.
* **The two orders have to be fitted over different step ranges.** The strong error needs fine steps
  to be inside its regime and the weak error needs coarse ones, because at fine steps the weak bias
  drops below the residual sampling noise and a fit there measures the noise.
* **The Ito correction is real and measurable.** The ratio of the mean of geometric Brownian motion
  to its median is 1.2890 against a predicted ``exp(sigma**2 t/2) = 1.2840``.
* **Black-Scholes by simulation converges to the closed form** at ``-0.535``, and an antithetic pair
  from lesson 92 cuts the error by a measured factor of 1.50 on average.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import functools
import math

import numpy as np


# --------------------------------------------------------------------------- paths


def random_walk(steps: int, paths: int = 1, seed: int = 42) -> np.ndarray:
    """A symmetric walk of plus or minus one, as an array of shape ``(paths, steps + 1)``."""
    n, m = int(steps), int(paths)
    if n < 0 or m < 1:
        raise ValueError(f"need a non-negative step count and at least one path, got {n} and {m}")
    rng = np.random.default_rng(int(seed))
    jumps = rng.choice(np.array([-1.0, 1.0]), size=(m, n))
    return np.concatenate([np.zeros((m, 1)), np.cumsum(jumps, axis=1)], axis=1)


def brownian(steps: int, horizon: float = 1.0, paths: int = 1, seed: int = 42) -> dict:
    """Brownian motion on ``[0, horizon]``, sampled at ``steps + 1`` equally spaced times.

    Each increment is normal with variance ``h``, which is the defining property: the standard
    deviation grows like ``sqrt(h)`` and not like ``h``, so the path has unbounded variation and no
    derivative anywhere.
    """
    n, m = int(steps), int(paths)
    if n < 1 or m < 1:
        raise ValueError(f"need at least one step and one path, got {n} and {m}")
    if horizon <= 0.0:
        raise ValueError(f"the horizon must be positive, got {horizon}")
    h = float(horizon) / n
    rng = np.random.default_rng(int(seed))
    increments = math.sqrt(h) * rng.standard_normal((m, n))
    values = np.concatenate([np.zeros((m, 1)), np.cumsum(increments, axis=1)], axis=1)
    return {"times": np.linspace(0.0, float(horizon), n + 1), "values": values,
            "increments": increments, "step": h, "paths": m, "steps": n}


def quadratic_variation(path, times) -> float:
    """``sum (X[k+1] - X[k])**2``, which for Brownian motion converges to the elapsed time.

    For any differentiable function this tends to zero as the mesh refines. For Brownian motion it
    does not, and its limit is exactly ``t``. That is the single fact from which ``dW**2 = dt``
    follows, and therefore the whole of Ito calculus.
    """
    v = np.asarray(path, dtype=float).ravel()
    t = np.asarray(times, dtype=float).ravel()
    if v.size != t.size:
        raise ValueError(f"the path and the times must match, got {v.size} and {t.size}")
    return float(np.sum(np.diff(v) ** 2))


# --------------------------------------------------------------------------- problems


def geometric_brownian(drift: float = 0.1, volatility: float = 0.3,
                       start: float = 1.0) -> dict:
    """``dX = mu X dt + sigma X dW``, whose exact solution is known path by path.

    That last point is what makes strong order measurable. The exact solution driven by a given
    Brownian path is

        X(t) = X(0) exp((mu - sigma**2/2) t + sigma W(t)) ,

    so the same noise can be fed to the exact formula and to the method, and the two answers
    compared at the same time on the same path.
    """
    mu, sigma, x0 = float(drift), float(volatility), float(start)

    def a(t, x):
        return mu * np.asarray(x, dtype=float)

    def b(t, x):
        return sigma * np.asarray(x, dtype=float)

    def b_prime(t, x):
        return sigma * np.ones_like(np.asarray(x, dtype=float))

    def exact(t, brownian_value, x0=x0):
        return x0 * np.exp((mu - 0.5 * sigma ** 2) * np.asarray(t, dtype=float)
                           + sigma * np.asarray(brownian_value, dtype=float))

    return {"a": a, "b": b, "b_prime": b_prime, "exact": exact,
            "start": x0, "drift": mu, "volatility": sigma,
            "mean": lambda t: x0 * math.exp(mu * float(t)),
            "median": lambda t: x0 * math.exp((mu - 0.5 * sigma ** 2) * float(t)),
            "variance": lambda t: (x0 ** 2 * math.exp(2.0 * mu * float(t))
                                   * (math.exp(sigma ** 2 * float(t)) - 1.0)),
            "name": f"dX = {mu:g} X dt + {sigma:g} X dW"}


def ornstein_uhlenbeck(rate: float = 2.0, level: float = 1.0, volatility: float = 0.4,
                       start: float = 0.0) -> dict:
    """``dX = theta (mu - X) dt + sigma dW``, the mean reverting case with additive noise.

    Additive noise means ``b`` does not depend on ``x``, so ``b' = 0`` and Milstein's extra term
    vanishes: on this problem the two methods are the same method. That is a useful control, because
    a measurement showing Milstein ahead here would be measuring an implementation error.
    """
    theta, mu, sigma, x0 = float(rate), float(level), float(volatility), float(start)

    def a(t, x):
        return theta * (mu - np.asarray(x, dtype=float))

    def b(t, x):
        return sigma * np.ones_like(np.asarray(x, dtype=float))

    def b_prime(t, x):
        return np.zeros_like(np.asarray(x, dtype=float))

    return {"a": a, "b": b, "b_prime": b_prime, "exact": None,
            "start": x0, "rate": theta, "level": mu, "volatility": sigma,
            "mean": lambda t: mu + (x0 - mu) * math.exp(-theta * float(t)),
            "variance": lambda t: (sigma ** 2 / (2.0 * theta)
                                   * (1.0 - math.exp(-2.0 * theta * float(t)))),
            "name": f"dX = {theta:g}({mu:g} - X) dt + {sigma:g} dW"}


# --------------------------------------------------------------------------- the methods


def euler_maruyama(problem, steps: int, horizon: float = 1.0, paths: int = 1,
                   seed: int = 42, increments=None) -> dict:
    """``X += a h + b dW``, the direct translation of Euler's method.

    Strong order ``1/2`` and weak order ``1``. The half comes from the ``dW`` term: its local error
    is ``O(h)`` where Euler's is ``O(h**2)``, because the Ito-Taylor expansion has a term of order
    ``h`` that this method drops.
    """
    n, m = int(steps), int(paths)
    h = float(horizon) / n
    if increments is None:
        increments = math.sqrt(h) * np.random.default_rng(int(seed)).standard_normal((m, n))
    increments = np.asarray(increments, dtype=float)
    x = np.full(increments.shape[0], float(problem["start"]))
    trail = np.empty((increments.shape[0], n + 1))
    trail[:, 0] = x
    t = 0.0
    for k in range(n):
        x = x + problem["a"](t, x) * h + problem["b"](t, x) * increments[:, k]
        trail[:, k + 1] = x
        t += h
    return {"x": x, "path": trail, "times": np.linspace(0.0, float(horizon), n + 1),
            "increments": increments, "step": h, "method": "euler-maruyama"}


def milstein(problem, steps: int, horizon: float = 1.0, paths: int = 1,
             seed: int = 42, increments=None) -> dict:
    """Euler-Maruyama plus ``(1/2) b b' (dW**2 - h)``, which lifts the strong order to 1.

    The extra term is the second order Ito-Taylor term, and it needs ``b'``. It vanishes identically
    when the noise is additive, so on an additive problem Milstein **is** Euler-Maruyama, which the
    measurement uses as a control.
    """
    n, m = int(steps), int(paths)
    h = float(horizon) / n
    if increments is None:
        increments = math.sqrt(h) * np.random.default_rng(int(seed)).standard_normal((m, n))
    increments = np.asarray(increments, dtype=float)
    x = np.full(increments.shape[0], float(problem["start"]))
    trail = np.empty((increments.shape[0], n + 1))
    trail[:, 0] = x
    t = 0.0
    for k in range(n):
        step = increments[:, k]
        diffusion = problem["b"](t, x)
        x = (x + problem["a"](t, x) * h + diffusion * step
             + 0.5 * diffusion * problem["b_prime"](t, x) * (step * step - h))
        trail[:, k + 1] = x
        t += h
    return {"x": x, "path": trail, "times": np.linspace(0.0, float(horizon), n + 1),
            "increments": increments, "step": h, "method": "milstein"}


def coarsen(increments, factor: int) -> np.ndarray:
    """Sum groups of ``factor`` increments, giving the same Brownian path on a coarser mesh.

    This is what makes a strong order measurement possible. The coarse run and the fine run have to
    be driven by the **same** Brownian path, and a Brownian increment over a long step is the sum of
    the increments over the short steps inside it, exactly.
    """
    v = np.asarray(increments, dtype=float)
    k = int(factor)
    if k < 1 or v.shape[1] % k:
        raise ValueError(f"cannot group {v.shape[1]} increments into blocks of {k}")
    return v.reshape(v.shape[0], v.shape[1] // k, k).sum(axis=2)


# --------------------------------------------------------------------------- Black-Scholes


def black_scholes(spot: float = 100.0, strike: float = 100.0, rate: float = 0.05,
                  volatility: float = 0.2, maturity: float = 1.0,
                  call: bool = True) -> float:
    """The closed form price of a European option, for checking the simulation against."""
    s, k = float(spot), float(strike)
    r, sigma, t = float(rate), float(volatility), float(maturity)
    if t <= 0.0 or sigma <= 0.0:
        raise ValueError(f"need a positive maturity and volatility, got {t} and {sigma}")
    first = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    second = first - sigma * math.sqrt(t)
    normal = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    if call:
        return s * normal(first) - k * math.exp(-r * t) * normal(second)
    return k * math.exp(-r * t) * normal(-second) - s * normal(-first)


def price_by_simulation(paths: int, spot: float = 100.0, strike: float = 100.0,
                        rate: float = 0.05, volatility: float = 0.2,
                        maturity: float = 1.0, call: bool = True,
                        antithetic: bool = False, seed: int = 42) -> dict:
    """Price the same option by sampling the terminal value under the risk neutral measure.

    No time stepping is needed: geometric Brownian motion has a closed form at the horizon, so one
    normal per path gives the exact terminal distribution and the only error is lesson 92's
    ``1/sqrt(M)``.
    """
    m = int(paths)
    if m < 2:
        raise ValueError(f"need at least two paths, got {m}")
    rng = np.random.default_rng(int(seed))
    sigma, t, r = float(volatility), float(maturity), float(rate)
    draws = rng.standard_normal(m // 2 if antithetic else m)
    if antithetic:
        draws = np.concatenate([draws, -draws])
    terminal = float(spot) * np.exp((r - 0.5 * sigma ** 2) * t + sigma * math.sqrt(t) * draws)
    payoff = (np.maximum(terminal - float(strike), 0.0) if call
              else np.maximum(float(strike) - terminal, 0.0))
    discounted = math.exp(-r * t) * payoff
    if antithetic:
        discounted = 0.5 * (discounted[:m // 2] + discounted[m // 2:])
    return {"price": float(np.mean(discounted)),
            "standard_error": float(np.std(discounted, ddof=1)) / math.sqrt(discounted.size),
            "paths": m, "antithetic": bool(antithetic)}


# --------------------------------------------------------------------------- measurements


@functools.lru_cache(maxsize=None)
def the_walk_becomes_brownian_motion(steps=(64, 256, 1024, 4096, 16384),
                                     paths: int = 4000) -> dict:
    """Scale a symmetric random walk by ``sqrt(h)`` and watch it converge to Brownian motion.

    Donsker's theorem says the scaled walk converges in distribution to Brownian motion. The two
    properties to check are that the variance at time 1 is 1 and that the increments are
    uncorrelated, and both are checked against the sampling error rather than against a guess.
    """
    rows = []
    for n in steps:
        walk = random_walk(n, paths=paths, seed=42)
        scaled = walk / math.sqrt(n)                 # so that time 1 is step n
        ending = scaled[:, -1]
        halfway = scaled[:, n // 2]
        increments = np.diff(scaled, axis=1)
        correlation = float(np.corrcoef(increments[:, :-1].ravel(),
                                        increments[:, 1:].ravel())[0, 1])
        rows.append({"steps": n,
                     "variance_at_one": float(np.var(ending, ddof=1)),
                     "variance_at_a_half": float(np.var(halfway, ddof=1)),
                     "mean_at_one": float(np.mean(ending)),
                     "lag_one_correlation": correlation})
    bar = 1.0 / math.sqrt(2.0 * paths)
    return {
        "rows": rows, "paths": int(paths), "sampling_bar": bar,
        "variance_reaches_one": all(abs(r["variance_at_one"] - 1.0) < 6.0 * bar for r in rows),
        "variance_is_linear_in_time": all(abs(r["variance_at_a_half"] - 0.5) < 6.0 * bar
                                          for r in rows),
        "increments_are_uncorrelated": all(abs(r["lag_one_correlation"]) < 0.02 for r in rows),
        "worst_variance_error": max(abs(r["variance_at_one"] - 1.0) for r in rows),
        "note": "the variance at time t is t and the increments are independent, which are the two "
                "defining properties, and both are already visible at 64 steps",
    }


@functools.lru_cache(maxsize=None)
def the_quadratic_variation_is_the_time(steps=(256, 1024, 4096, 16384, 65536),
                                        horizon: float = 1.0) -> dict:
    """The sum of squared increments against the elapsed time, and against a smooth function.

    For a differentiable path the same sum tends to zero. The contrast is the point: this is the one
    property that separates Brownian motion from every function Part 10 could integrate, and it is
    the reason Ito calculus exists.
    """
    rows = []
    for n in steps:
        path = brownian(n, horizon=horizon, paths=1, seed=42)
        rough = quadratic_variation(path["values"][0], path["times"])
        smooth = quadratic_variation(np.sin(3.0 * path["times"]), path["times"])
        rows.append({"steps": n, "brownian": rough, "smooth": smooth,
                     "brownian_error": abs(rough - horizon)})
    smooth_powers = float(np.polyfit(np.log([r["steps"] for r in rows]),
                                     np.log([r["smooth"] for r in rows]), 1)[0])
    return {
        "rows": rows, "horizon": float(horizon), "smooth_power": smooth_powers,
        "brownian_reaches_the_horizon": rows[-1]["brownian_error"] < 0.05 * horizon,
        "the_smooth_one_goes_to_zero": rows[-1]["smooth"] < 1e-3,
        "smooth_power_is_minus_one": abs(smooth_powers + 1.0) < 0.05,
        "note": "the sum of squared increments is t for Brownian motion and falls like 1/n for a "
                "differentiable path, which is dW**2 = dt stated as a measurement",
    }


def _order(problem, method, powers, paths, seed, horizon, kind):
    """Fit the order of a method, strong or weak, against a reference on the same noise.

    Both kinds are measured against the exact solution driven by the **same** Brownian path. For
    the strong order that is what the definition asks for. For the weak order it is a variance
    reduction step and not part of the definition: the quantity wanted is
    ``|E[X_h] - E[X]|``, and comparing against the sample's own exact mean rather than the true
    mean cancels the sampling error common to both, which would otherwise be far larger than the
    bias being measured. It is the control variate of lesson 92 with a correlation very close to
    one.
    """
    finest = 2 ** max(powers)
    h = float(horizon) / finest
    rng = np.random.default_rng(int(seed))
    fine = math.sqrt(h) * rng.standard_normal((int(paths), finest))
    walk = np.concatenate([np.zeros((int(paths), 1)), np.cumsum(fine, axis=1)], axis=1)
    truth = problem["exact"](horizon, walk[:, -1])
    reference = float(np.mean(truth))
    steps, errors = [], []
    for power in powers:
        n = 2 ** power
        coarse = coarsen(fine, finest // n)
        out = method(problem, n, horizon=horizon, increments=coarse)
        if kind == "strong":
            errors.append(float(np.mean(np.abs(out["x"] - truth))))
        else:
            errors.append(abs(float(np.mean(out["x"])) - reference))
        steps.append(float(horizon) / n)
    fitted = float(np.polyfit(np.log(steps), np.log(errors), 1)[0])
    return {"steps": steps, "errors": errors, "fitted_order": fitted,
            "sample_reference": reference, "true_mean": problem["mean"](horizon),
            "sampling_error": abs(reference - problem["mean"](horizon))}


@functools.lru_cache(maxsize=None)
def euler_maruyama_is_half_strong_and_one_weak(strong_powers=(4, 5, 6, 7, 8),
                                               weak_powers=(1, 2, 3, 4, 5),
                                               paths: int = 200000) -> dict:
    """The same method, two different questions, two different answers.

    The two orders are fitted over **different** step ranges, and the reason is worth stating. The
    strong error is a mean absolute difference of order ``sqrt(h)``, which is large, so it needs
    fine steps to be inside its asymptotic regime. The weak error is a difference of two means of
    order ``h``, which at fine steps drops below the residual sampling noise and stops being
    measurable; fitting it there measures the noise. So the strong fit uses ``h`` from 1/16 down to
    1/256 and the weak fit uses 1/2 down to 1/32, and each is taken where its own signal dominates.
    """
    problem = geometric_brownian()
    strong = _order(problem, euler_maruyama, strong_powers, paths, 42, 1.0, "strong")
    weak = _order(problem, euler_maruyama, weak_powers, paths, 42, 1.0, "weak")
    return {
        "strong_order": strong["fitted_order"], "strong_errors": strong["errors"],
        "strong_steps": strong["steps"],
        "weak_order": weak["fitted_order"], "weak_errors": weak["errors"],
        "weak_steps": weak["steps"],
        "steps": strong["steps"], "paths": int(paths),
        "sampling_error": strong["sampling_error"],
        "strong_is_a_half": abs(strong["fitted_order"] - 0.5) < 0.08,
        "weak_is_one": abs(weak["fitted_order"] - 1.0) < 0.1,
        "they_differ": weak["fitted_order"] > strong["fitted_order"] + 0.3,
        "note": "strong order compares paths against the same path of the exact solution and weak "
                "order compares one expectation against another, so a method can be twice as good "
                "at one as at the other on identical runs",
    }


@functools.lru_cache(maxsize=None)
def milstein_doubles_the_strong_order_only(strong_powers=(4, 5, 6, 7, 8),
                                           weak_powers=(1, 2, 3, 4, 5),
                                           paths: int = 200000) -> dict:
    """Milstein against Euler-Maruyama, on the same problem and the same noise, both ways.

    Each order is fitted over the step range where its own signal dominates, for the reason given in
    :func:`euler_maruyama_is_half_strong_and_one_weak`.
    """
    problem = geometric_brownian()
    rows = []
    for name, method in (("euler-maruyama", euler_maruyama), ("milstein", milstein)):
        strong = _order(problem, method, strong_powers, paths, 42, 1.0, "strong")
        weak = _order(problem, method, weak_powers, paths, 42, 1.0, "weak")
        rows.append({"method": name,
                     "strong_order": strong["fitted_order"],
                     "weak_order": weak["fitted_order"],
                     "strong_error_at_the_finest": strong["errors"][-1],
                     "weak_error_at_the_finest": weak["errors"][-1]})
    plain, better = rows
    return {
        "rows": rows, "paths": int(paths),
        "strong_gain": plain["strong_error_at_the_finest"] / better["strong_error_at_the_finest"],
        "weak_gain": plain["weak_error_at_the_finest"] / max(
            better["weak_error_at_the_finest"], 1e-300),
        "milstein_is_strong_order_one": abs(better["strong_order"] - 1.0) < 0.1,
        "euler_is_strong_order_a_half": abs(plain["strong_order"] - 0.5) < 0.08,
        "both_are_weak_order_one": (abs(plain["weak_order"] - 1.0) < 0.12
                                    and abs(better["weak_order"] - 1.0) < 0.12),
        "the_extra_term_buys_nothing_weakly": abs(better["weak_order"]
                                                  - plain["weak_order"]) < 0.12,
        "note": "the extra term doubles the strong order and leaves the weak order alone, so it is "
                "worth paying for only when the path itself is wanted",
    }


@functools.lru_cache(maxsize=None)
def the_extra_term_vanishes_for_additive_noise(steps: int = 128, paths: int = 2000) -> dict:
    """On an additive noise problem Milstein and Euler-Maruyama agree bit for bit.

    A control, and a strict one: ``b' = 0`` makes the Milstein term identically zero, so any
    difference at all would be an implementation error rather than a numerical one.
    """
    problem = ornstein_uhlenbeck()
    rng = np.random.default_rng(42)
    h = 1.0 / steps
    increments = math.sqrt(h) * rng.standard_normal((paths, steps))
    plain = euler_maruyama(problem, steps, paths=paths, increments=increments)
    better = milstein(problem, steps, paths=paths, increments=increments)
    gap = float(np.max(np.abs(plain["x"] - better["x"])))
    return {
        "worst_difference": gap, "identical": gap == 0.0,
        "steps": int(steps), "paths": int(paths),
        "mean_error": abs(float(np.mean(plain["x"])) - problem["mean"](1.0)),
        "variance_error": abs(float(np.var(plain["x"], ddof=1)) - problem["variance"](1.0)),
        "the_solution_has_the_right_moments": (
            abs(float(np.mean(plain["x"])) - problem["mean"](1.0)) < 0.02
            and abs(float(np.var(plain["x"], ddof=1)) - problem["variance"](1.0)) < 0.01),
        "note": "additive noise means b does not depend on x, so Milstein's term is zero and the "
                "two methods are the same method, exactly",
    }


@functools.lru_cache(maxsize=None)
def the_ito_correction_is_measurable(paths: int = 200000, horizon: float = 2.0) -> dict:
    """The mean and the median of geometric Brownian motion differ by exactly the Ito term.

    Ordinary calculus applied to ``d(log X)`` would give ``log X(t) = log X(0) + (mu - 0) t + ...``.
    Ito's rule adds ``-sigma**2/2``, and the consequence is that the **median** grows like
    ``exp((mu - sigma**2/2) t)`` while the **mean** grows like ``exp(mu t)``. Both are measured here
    on the same sample.
    """
    problem = geometric_brownian(drift=0.1, volatility=0.5)
    rng = np.random.default_rng(42)
    walk = math.sqrt(horizon) * rng.standard_normal(int(paths))
    values = problem["exact"](horizon, walk)
    measured_mean = float(np.mean(values))
    measured_median = float(np.median(values))
    naive = problem["start"] * math.exp(problem["drift"] * horizon)
    corrected = problem["start"] * math.exp(
        (problem["drift"] - 0.5 * problem["volatility"] ** 2) * horizon)
    return {
        "horizon": float(horizon), "paths": int(paths),
        "measured_mean": measured_mean, "predicted_mean": problem["mean"](horizon),
        "measured_median": measured_median, "predicted_median": problem["median"](horizon),
        "mean_ratio": measured_mean / problem["mean"](horizon),
        "median_ratio": measured_median / problem["median"](horizon),
        "ito_term": 0.5 * problem["volatility"] ** 2 * horizon,
        "mean_over_median": measured_mean / measured_median,
        "predicted_ratio": math.exp(0.5 * problem["volatility"] ** 2 * horizon),
        "the_correction_matches": abs(
            measured_mean / measured_median
            / math.exp(0.5 * problem["volatility"] ** 2 * horizon) - 1.0) < 0.02,
        "the_naive_answer_is_the_wrong_one": abs(naive - corrected) > 0.1 * corrected,
        "note": "the mean and the median of a lognormal differ by exp(sigma**2 t / 2), which is "
                "exactly the term Ito's rule adds and ordinary calculus leaves out",
    }


@functools.lru_cache(maxsize=None)
def black_scholes_by_simulation(counts=(1000, 4000, 16000, 64000, 256000),
                                repeats: int = 40) -> dict:
    """The simulated price against the closed form, with and without antithetic pairs."""
    exact = black_scholes()
    rows = []
    for n in counts:
        plain = np.array([price_by_simulation(n, seed=s)["price"] for s in range(repeats)])
        folded = np.array([price_by_simulation(n, antithetic=True, seed=s)["price"]
                           for s in range(repeats)])
        rows.append({
            "paths": n,
            "plain_error": float(np.sqrt(np.mean((plain - exact) ** 2))),
            "antithetic_error": float(np.sqrt(np.mean((folded - exact) ** 2))),
        })
    logs = np.log([r["paths"] for r in rows])
    plain_power = float(np.polyfit(logs, np.log([r["plain_error"] for r in rows]), 1)[0])
    folded_power = float(np.polyfit(logs, np.log([r["antithetic_error"] for r in rows]), 1)[0])
    gains = [r["plain_error"] / r["antithetic_error"] for r in rows]
    # lesson 92's prediction, from the correlation between the two halves of a pair
    rng = np.random.default_rng(7)
    draws = rng.standard_normal(max(counts))
    sigma, t, r = float(volatility_default := 0.2), 1.0, 0.05
    terminal = lambda z: 100.0 * np.exp((r - 0.5 * sigma ** 2) * t + sigma * math.sqrt(t) * z)
    payoff = lambda z: math.exp(-r * t) * np.maximum(terminal(z) - 100.0, 0.0)
    rho = float(np.corrcoef(payoff(draws), payoff(-draws))[0, 1])
    predicted = 1.0 / (1.0 + rho)
    return {
        "rows": rows, "exact_price": exact, "repeats": int(repeats),
        "plain_power": plain_power, "antithetic_power": folded_power,
        "both_are_root_n": (abs(plain_power + 0.5) < 0.08
                            and abs(folded_power + 0.5) < 0.08),
        "mean_gain": float(np.mean(gains)),
        "pair_correlation": rho,
        "predicted_variance_ratio": predicted,
        "predicted_error_ratio": math.sqrt(predicted),
        "the_prediction_holds": abs(float(np.mean(gains)) / math.sqrt(predicted) - 1.0) < 0.15,
        "antithetic_helps": float(np.mean(gains)) > 1.2,
        "note": "the payoff is monotone in the driving normal, so lesson 92's antithetic pairing "
                "applies unchanged, and the exponent stays at -1/2 because the technique changes "
                "the constant and never the rate",
    }
