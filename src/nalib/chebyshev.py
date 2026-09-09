"""Chebyshev polynomials and Chebyshev nodes: the fix for lesson 46's divergence.

The one theorem
---------------
Among all monic polynomials of degree ``n`` on ``[-1, 1]``, the one with the **smallest maximum
absolute value** is ``T_n / 2^(n-1)``, and its maximum is ``2^(1-n)``. Since the node polynomial
``prod_i (t - x_i)`` is monic of degree ``n``, choosing the nodes to be the roots of ``T_n``
makes the one controllable factor of lesson 46's error formula as small as it can possibly be.

`minimax_property` measures that against random and equally spaced alternatives, and it wins
every time, which is what a theorem means.

The polynomials
---------------
``T_0 = 1``, ``T_1 = x``, ``T_{k+1} = 2 x T_k - T_{k-1}``. Equivalently ``T_k(cos θ) = cos(kθ)``,
which makes every property obvious: the extrema are ``+-1`` at ``k + 1`` points, the roots are
``cos((2j+1)π/2k)``, and ``|T_k| <= 1`` on the interval.

`recurrence` and `by_cosine` compute them both ways and `recurrence_vs_cosine` compares them.
The recurrence is stable on ``[-1, 1]`` and diverges outside it, which is measured rather than
asserted.

The nodes
---------
`nodes` gives the roots of ``T_n``, `extrema_nodes` gives the Chebyshev-Lobatto points, which
include the endpoints and are what a practical code uses. `change_of_interval` maps either set
from ``[-1, 1]`` to ``[a, b]``, which is an affine map and so preserves the minimax property up
to the obvious scaling.

Lebesgue constants
------------------
The **Lebesgue constant** ``Λ_n = max_t sum_i |L_i(t)|`` bounds how much worse interpolation can
be than the best possible polynomial approximation:

    ||f - p_n||  <=  (1 + Λ_n) ||f - best_n||

For Chebyshev nodes ``Λ_n`` grows like ``(2/π) log n``, which is negligible. For equally spaced
nodes it grows like ``2^n / (e n log n)``, which is catastrophic. `lebesgue_constant` computes
it and `lebesgue_growth` fits both rates. That single comparison explains the whole of lesson 46.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import interp as _ip


# ----------------------------------------------------------------------------- polynomials


def recurrence(k: int, t) -> np.ndarray:
    """``T_k`` by the three term recurrence ``T_{k+1} = 2 t T_k - T_{k-1}``.

    Costs ``k`` steps and is stable on ``[-1, 1]``. Outside it, both solutions of the recurrence
    grow and the computed one is fine because the wanted solution is the dominant one, which is
    the opposite of the usual unstable-recurrence situation.
    """
    k = int(k)
    if k < 0:
        raise ValueError(f"degree must be non-negative, got {k}")
    z = np.asarray(t, dtype=float)
    if k == 0:
        return np.ones_like(z)
    prev, cur = np.ones_like(z), z.copy()
    for _ in range(k - 1):
        prev, cur = cur, 2.0 * z * cur - prev
    return cur


def by_cosine(k: int, t) -> np.ndarray:
    """``T_k(t) = cos(k arccos t)`` on ``[-1, 1]``, and ``cosh(k arccosh|t|)`` outside it.

    The identity is the definition that makes every property of these polynomials immediate, and
    it is also a second independent way to compute them, which is what makes the comparison in
    `recurrence_vs_cosine` meaningful.
    """
    k = int(k)
    z = np.asarray(t, dtype=float)
    inside = np.abs(z) <= 1.0
    out = np.empty_like(z, dtype=float)
    out[inside] = np.cos(k * np.arccos(z[inside]))
    outside = ~inside
    if outside.any():
        s = np.sign(z[outside])
        out[outside] = (s ** k) * np.cosh(k * np.arccosh(np.abs(z[outside])))
    return out


def recurrence_vs_cosine(k: int, lo: float = -1.0, hi: float = 1.0, n_probe: int = 801) -> dict:
    """Two independent computations of the same polynomial, compared."""
    probe = np.linspace(lo, hi, int(n_probe))
    a, b = recurrence(k, probe), by_cosine(k, probe)
    scale = max(float(np.max(np.abs(b))), 1e-300)
    return {"max_gap": float(np.max(np.abs(a - b))),
            "relative_gap": float(np.max(np.abs(a - b)) / scale),
            "max_value": scale}


def coefficients(k: int) -> np.ndarray:
    """The power form coefficients of ``T_k``, ascending, built by the recurrence.

    Included so the polynomial can be inspected, and as a reminder that these coefficients grow
    like ``2^k`` and are exactly what you should not evaluate with.
    """
    k = int(k)
    if k < 0:
        raise ValueError(f"degree must be non-negative, got {k}")
    prev = np.array([1.0])
    if k == 0:
        return prev
    cur = np.array([0.0, 1.0])
    for _ in range(k - 1):
        shifted = np.concatenate([[0.0], 2.0 * cur])
        padded = np.concatenate([prev, np.zeros(shifted.size - prev.size)])
        prev, cur = cur, shifted - padded
    return cur


# ----------------------------------------------------------------------------- nodes


def nodes(n: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """The ``n`` roots of ``T_n``, mapped to ``[lo, hi]`` and returned in increasing order.

    ``x_j = cos((2j + 1) π / (2n))`` on ``[-1, 1]``. These are the minimax nodes of the theorem.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"need at least one node, got {n}")
    j = np.arange(n)
    base = np.cos((2.0 * j + 1.0) * np.pi / (2.0 * n))
    return np.sort(change_of_interval(base, lo, hi))


def extrema_nodes(n: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """The Chebyshev-Lobatto points, ``cos(jπ/(n-1))``, which include both endpoints.

    A practical code uses these rather than the roots, because including the endpoints means the
    interpolant is defined on the closed interval without extrapolating, and because they have a
    fast transform. Their Lebesgue constant grows at the same logarithmic rate.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"need at least two Lobatto points, got {n}")
    j = np.arange(n)
    base = np.cos(j * np.pi / (n - 1))
    return np.sort(change_of_interval(base, lo, hi))


def change_of_interval(t, lo: float, hi: float) -> np.ndarray:
    """Map from ``[-1, 1]`` to ``[lo, hi]``: ``x = (lo + hi)/2 + (hi - lo)/2 * t``.

    Affine, so it preserves the minimax property with the node polynomial's maximum scaled by
    ``((hi - lo)/2)^n``. That scaling is why a long interval is harder, and it is the reason a
    practical code subdivides rather than raising the degree.
    """
    z = np.asarray(t, dtype=float)
    return 0.5 * (float(lo) + float(hi)) + 0.5 * (float(hi) - float(lo)) * z


def inverse_change_of_interval(x, lo: float, hi: float) -> np.ndarray:
    """The inverse map, back to ``[-1, 1]``."""
    z = np.asarray(x, dtype=float)
    half = 0.5 * (float(hi) - float(lo))
    if half == 0.0:
        raise ValueError("the interval has zero width")
    return (z - 0.5 * (float(lo) + float(hi))) / half


# ----------------------------------------------------------------------------- the theorem


def minimax_property(n: int, n_random: int = 200, n_probe: int = 4001, rng=None) -> dict:
    """The Chebyshev node polynomial has the smallest maximum among monic degree ``n``.

    Compared against equally spaced nodes and against random node sets, since a theorem that
    says "smallest" should beat everything offered, not merely the obvious alternative.
    """
    from . import interperror as _ie

    n = int(n)
    gen = np.random.default_rng() if rng is None else rng
    probe = np.linspace(-1.0, 1.0, int(n_probe))
    worst = lambda x: float(np.max(np.abs(_ie.node_polynomial(x, probe))))

    cheb = worst(nodes(n))
    equal = worst(np.linspace(-1.0, 1.0, n))
    random_best = min(worst(np.sort(gen.uniform(-1.0, 1.0, n)))
                      for _ in range(max(int(n_random), 1)))
    theory = 2.0 ** (1 - n)
    return {"chebyshev": cheb, "equally_spaced": equal, "best_of_random": random_best,
            "theoretical_minimum": theory,
            "chebyshev_matches_theory": abs(cheb - theory) / theory,
            "beats_equally_spaced_by": equal / cheb,
            "beats_random_by": random_best / cheb}


# ----------------------------------------------------------------------------- Lebesgue


def lebesgue_function(node_set, t) -> np.ndarray:
    """``Λ(t) = sum_i |L_i(t)|``, the amplification at the point ``t``."""
    x = np.atleast_1d(np.asarray(node_set, dtype=float)).ravel()
    z = np.atleast_1d(np.asarray(t, dtype=float))
    return np.abs(_ip.lagrange_basis(x, z)).sum(axis=0)


def lebesgue_constant(node_set, n_probe: int = 8001) -> float:
    """``Λ_n = max_t Λ(t)``, the constant in ``||f - p|| <= (1 + Λ_n) ||f - best||``."""
    x = np.sort(np.atleast_1d(np.asarray(node_set, dtype=float)).ravel())
    probe = np.linspace(float(x[0]), float(x[-1]), int(n_probe))
    return float(np.max(lebesgue_function(x, probe)))


@dataclass
class LebesgueGrowth:
    """The two growth rates, side by side. The whole of lesson 46 in one table."""

    n_values: np.ndarray
    equally_spaced: np.ndarray
    chebyshev: np.ndarray
    equal_fitted_base: float
    chebyshev_fitted_log_slope: float


def lebesgue_growth(n_values=None, n_probe: int = 4001) -> LebesgueGrowth:
    """Fit ``Λ_n`` against ``n`` for both node families.

    Equally spaced is fitted as ``C b^n`` and Chebyshev as ``C + s log n``, because those are the
    forms the theory predicts and fitting the wrong form to either would hide the difference.
    """
    ns = ([4, 6, 8, 10, 12, 14, 16, 20] if n_values is None
          else [int(v) for v in np.atleast_1d(n_values)])
    eq = np.asarray([lebesgue_constant(np.linspace(-1.0, 1.0, n), n_probe) for n in ns])
    ch = np.asarray([lebesgue_constant(nodes(n), n_probe) for n in ns])
    arr = np.asarray(ns, dtype=float)
    base = float(np.exp(np.polyfit(arr, np.log(eq), 1)[0]))
    slope = float(np.polyfit(np.log(arr), ch, 1)[0])
    return LebesgueGrowth(n_values=np.asarray(ns), equally_spaced=eq, chebyshev=ch,
                          equal_fitted_base=base, chebyshev_fitted_log_slope=slope)


def barycentric_weights(n: int, lobatto: bool = False) -> np.ndarray:
    """The barycentric weights for Chebyshev nodes, in **closed form**, in increasing node order.

    This is the second reason Chebyshev nodes are used. For a general node set the weights cost
    ``O(n^2)`` to compute and can overflow; for these they are known exactly:

    - roots of ``T_n``:   ``w_j = (-1)^j sin((2j + 1) π / 2n)``
    - Lobatto points:     ``w_j = (-1)^j``, halved at the two ends

    Both are ``O(n)`` to write down, both are bounded by 1, and neither involves a product.

    The formulas above are indexed in the order ``cos`` produces, which is **decreasing**. The
    nodes are returned in increasing order, so the weights are reversed to match.
    """
    n = int(n)
    if lobatto:
        if n < 2:
            raise ValueError(f"need at least two Lobatto points, got {n}")
        w = np.empty(n)
        w[::2] = 1.0
        w[1::2] = -1.0
        w[0] *= 0.5
        w[-1] *= 0.5
    else:
        if n < 1:
            raise ValueError(f"need at least one node, got {n}")
        j = np.arange(n)
        w = (-1.0) ** j * np.sin((2.0 * j + 1.0) * np.pi / (2.0 * n))
    return w[::-1]                      # cos gives decreasing nodes; `nodes` returns increasing


def interpolate(f, n: int, lo: float = -1.0, hi: float = 1.0, lobatto: bool = False):
    """Interpolate ``f`` at Chebyshev nodes and return the evaluator, nodes and values.

    The evaluator is barycentric with the closed form weights above, so the whole setup is
    ``O(n)`` beyond the ``n`` evaluations of ``f``, and every evaluation afterwards is ``O(n)``.
    """
    n = int(n)
    x = extrema_nodes(n, lo, hi) if lobatto else nodes(n, lo, hi)
    y = np.asarray([f(v) for v in x], dtype=float)
    w = barycentric_weights(n, lobatto=lobatto)
    return (lambda t: _ip.evaluate_barycentric(x, y, t, weights=w)), x, y
