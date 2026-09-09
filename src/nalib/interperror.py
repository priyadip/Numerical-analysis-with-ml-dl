"""Interpolation error: the formula, what each factor does, and the Runge phenomenon.

The error formula
-----------------
For ``f`` with ``n + 1`` continuous derivatives and nodes ``x_0, ..., x_n``,

    f(t) - p(t)  =  f^(n+1)(xi) / (n+1)!  *  prod_i (t - x_i)

for some ``xi`` in the interval. Three factors, and they behave completely differently:

**The derivative** ``f^(n+1)(xi)``. A property of ``f`` alone. Nothing about the method changes
it, and for some functions it grows faster than ``(n+1)!`` shrinks.

**The factorial** ``(n+1)!``. Grows enormously, which is the reason interpolation works at all
when the derivative is tame.

**The node polynomial** ``prod_i (t - x_i)``. **The only factor you control.** Its size is
determined entirely by where the nodes are, and minimising it is exactly what lesson 47 does.

`error_formula_report` measures all three separately, `node_polynomial` computes the third, and
`node_polynomial_extremes` finds where it is worst, which is always near the ends for equally
spaced nodes.

Runge
-----
``f(t) = 1/(1 + 25 t^2)`` on ``[-1, 1]`` is smooth, bounded, and infinitely differentiable, and
equally spaced polynomial interpolation of it **diverges**: the error at the ends grows without
bound as nodes are added. `runge` is the function and `runge_divergence` measures the growth.

The mechanism is that ``f^(n+1)`` grows faster than ``(n+1)!``, because the poles at
``t = +- i/5`` sit close to the real interval, and the node polynomial is largest near the ends,
so the two bad factors multiply exactly where you can least afford it.

**It is not a rounding problem.** `runge_divergence` measures the same divergence in exact
rational arithmetic if asked, and lesson 47 shows the fix is to move the nodes, not to compute
more carefully.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import interp as _ip


def node_polynomial(nodes, t) -> np.ndarray:
    """``w(t) = prod_i (t - x_i)``, the only factor of the error you control."""
    x = np.atleast_1d(np.asarray(nodes, dtype=float)).ravel()
    z = np.asarray(t, dtype=float)
    out = np.ones_like(np.atleast_1d(z), dtype=float)
    for xi in x:
        out = out * (np.atleast_1d(z) - xi)
    return out.reshape(np.shape(t)) if np.ndim(t) else float(out[0])


def node_polynomial_extremes(nodes, n_probe: int = 4001) -> dict:
    """Where and how large ``|w|`` is, on the interval the nodes span.

    For equally spaced nodes the maximum sits in the outermost gap and is enormously larger than
    the typical value. That imbalance is the whole story of the Runge phenomenon, and Chebyshev
    nodes exist to remove it.
    """
    x = np.sort(np.atleast_1d(np.asarray(nodes, dtype=float)).ravel())
    lo, hi = float(x[0]), float(x[-1])
    probe = np.linspace(lo, hi, int(n_probe))
    w = np.abs(node_polynomial(x, probe))
    k = int(np.argmax(w))
    interior = probe[(probe > lo + 0.25 * (hi - lo)) & (probe < hi - 0.25 * (hi - lo))]
    middle = float(np.max(np.abs(node_polynomial(x, interior)))) if interior.size else 0.0
    return {"max": float(w[k]), "argmax": float(probe[k]),
            "max_in_middle_half": middle,
            "end_to_middle_ratio": float(w[k] / middle) if middle > 0 else float("inf"),
            "distance_from_nearest_end": float(min(probe[k] - lo, hi - probe[k]))}


@dataclass
class ErrorReport:
    """The three factors of the error formula, measured separately."""

    n_nodes: int
    derivative_bound: float
    factorial: float
    node_polynomial_max: float
    predicted_bound: float
    measured_error: float
    bound_overstates: float


def error_formula_report(f, dfn, n_nodes: int, lo: float = -1.0, hi: float = 1.0,
                         chebyshev: bool = False, n_probe: int = 2001) -> ErrorReport:
    """Compare the error bound with the error, factor by factor.

    ``dfn(k, t)`` must return the ``k``-th derivative of ``f`` at ``t``. The bound uses the
    maximum of ``|f^(n+1)|`` over the interval, which is the standard sharpening of "some xi".
    """
    n = int(n_nodes)
    if n < 1:
        raise ValueError(f"need at least one node, got {n}")
    if chebyshev:
        k = np.arange(n)
        x = np.sort(0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos((2 * k + 1) * np.pi / (2 * n)))
    else:
        x = np.linspace(lo, hi, n)
    y = np.asarray([f(v) for v in x], dtype=float)
    probe = np.linspace(lo, hi, int(n_probe))
    truth = np.asarray([f(v) for v in probe], dtype=float)
    got = _ip.evaluate_barycentric(x, y, probe)

    deriv = float(np.max(np.abs([dfn(n, v) for v in probe])))
    fact = float(math.factorial(n))
    wmax = float(np.max(np.abs(node_polynomial(x, probe))))
    bound = deriv / fact * wmax
    err = float(np.max(np.abs(got - truth)))
    return ErrorReport(n_nodes=n, derivative_bound=deriv, factorial=fact,
                       node_polynomial_max=wmax, predicted_bound=bound,
                       measured_error=err,
                       bound_overstates=bound / max(err, 1e-300))


# ----------------------------------------------------------------------------- Runge


def runge(t, a: float = 25.0):
    """``1 / (1 + a t^2)``. Smooth everywhere on the real line, and interpolation still fails.

    The poles sit at ``t = +- i / sqrt(a)``, so raising ``a`` moves them closer to the real
    interval and makes the divergence faster. That is the honest statement of what goes wrong:
    the trouble is in the complex plane, not on the interval where the function looks harmless.
    """
    z = np.asarray(t, dtype=float)
    return 1.0 / (1.0 + float(a) * z * z)


def runge_divergence(n_values=None, a: float = 25.0, lo: float = -1.0, hi: float = 1.0,
                     chebyshev: bool = False, n_probe: int = 2001) -> dict:
    """Maximum error against the number of nodes, and where the maximum sits.

    For equally spaced nodes the error grows without bound and its location stays near the ends.
    For Chebyshev nodes it falls geometrically.
    """
    ns = ([4, 6, 8, 10, 12, 16, 20, 24] if n_values is None
          else [int(v) for v in np.atleast_1d(n_values)])
    probe = np.linspace(lo, hi, int(n_probe))
    truth = runge(probe, a)
    errors, where = [], []
    for n in ns:
        if chebyshev:
            k = np.arange(n)
            x = np.sort(0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos((2 * k + 1) * np.pi / (2 * n)))
        else:
            x = np.linspace(lo, hi, n)
        got = _ip.evaluate_barycentric(x, runge(x, a), probe)
        d = np.abs(got - truth)
        errors.append(float(np.max(d)))
        where.append(float(probe[int(np.argmax(d))]))
    e = np.asarray(errors)
    live = e > 0
    growth = (float(np.polyfit(np.asarray(ns, dtype=float)[live], np.log(e[live]), 1)[0])
              if live.sum() >= 2 else float("nan"))
    return {"n_values": np.asarray(ns), "errors": e, "argmax": np.asarray(where),
            "log_growth_per_node": growth,
            "diverges": bool(e[-1] > e[0])}


def where_the_error_lives(n_nodes: int, lo: float = -1.0, hi: float = 1.0, a: float = 25.0,
                          n_probe: int = 2001) -> dict:
    """Split the error into the middle half of the interval and the outer quarters.

    The point of the split is that the Runge phenomenon is entirely an end effect: the middle is
    interpolated well at every degree, and reporting a single maximum hides that.
    """
    n = int(n_nodes)
    x = np.linspace(lo, hi, n)
    probe = np.linspace(lo, hi, int(n_probe))
    d = np.abs(_ip.evaluate_barycentric(x, runge(x, a), probe) - runge(probe, a))
    span = hi - lo
    middle = (probe > lo + 0.25 * span) & (probe < hi - 0.25 * span)
    return {"max_overall": float(np.max(d)),
            "max_middle_half": float(np.max(d[middle])),
            "max_outer_quarters": float(np.max(d[~middle])),
            "ratio": float(np.max(d[~middle]) / max(np.max(d[middle]), 1e-300))}
