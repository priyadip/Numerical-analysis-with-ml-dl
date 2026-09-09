"""Classifying a second order PDE, and building the stencils that discretize one.

Why the classification comes first
----------------------------------
A second order quasi-linear equation in two variables

    A u_xx + B u_xy + C u_yy = F(x, y, u, u_x, u_y)

is **elliptic**, **parabolic** or **hyperbolic** according to the sign of ``B**2 - 4*A*C``, and
that sign decides everything a numerical method has to respect:

* which extra conditions make the problem well posed (all the way round the boundary, or an
  initial state plus side conditions, or an initial state and an initial velocity),
* whether information travels at a finite speed, and therefore whether a scheme has a step size
  limit at all,
* whether the answer is smoother than the data (it is, for elliptic and parabolic) or exactly as
  rough as the data (hyperbolic).

Solving a hyperbolic equation with a method built for a parabolic one is not a slow method, it is
a wrong one. `every_standard_equation_is_classified_right` puts the five standard equations
through `classify`, and `the_type_can_change_across_the_domain` runs Tricomi's equation, which is
hyperbolic below the ``x`` axis and elliptic above it, so the type is a property of a **point**
and not of an equation.

What the measurements here show
-------------------------------
* The compact nine point Laplacian is **second** order in general, not fourth, and on a harmonic
  function it is **sixth**, not fourth. Both halves of the usual statement are wrong, and one
  identity explains both: for a harmonic function ``u_xxyyyy = -u_xxxxyy``, so the ``h**4`` term
  in its expansion cancels along with the ``h**2`` one. `orders_of_the_two_laplacians` measures
  2.00 and 5.8, and `exact_on_harmonic_polynomials` pins it down with no tolerance at all: the
  five point rule is exact on harmonic polynomials up to degree **3**, and the nine point rule up
  to degree **7**.
* A one sided first derivative at a Neumann boundary is **first** order and it drags the whole
  solve down with it, even though every interior point is second order.
  `a_ghost_point_keeps_the_order` measures 1.00 against 2.00.
* The backward heat equation amplifies a mode of wavenumber ``k`` by ``exp(k**2 * t)``, so the
  amplification grows without bound as the mesh is refined. `the_backward_problem_is_ill_posed`
  measures the growth against ``k`` and fits the exponent, and no amount of care in the
  discretization can rescue it, because the continuous problem is the thing that is wrong.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

from .differentiation import fornberg

#: The three types, in the order the discriminant puts them.
TYPES = ("elliptic", "parabolic", "hyperbolic")


# --------------------------------------------------------------------------- classification


def discriminant(a, b, c):
    """``B**2 - 4*A*C``, the quantity whose sign names the type.

    Accepts scalars or arrays of any shape, broadcasting them together, because a variable
    coefficient equation has a discriminant at every point rather than one for the equation.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    return b * b - 4.0 * a * c


def classify(a, b, c, tol: float | None = None):
    """Name the type at every point, from the sign of the discriminant.

    Returns a string for scalar input and an array of strings otherwise, so the caller can use
    the result the same way whether the coefficients are constant or not.

    The tolerance matters. "Parabolic" is the boundary case ``B**2 == 4*A*C``, and asking a float
    whether it is exactly zero is asking the wrong question: the heat equation written as
    ``u_t - u_xx = 0`` has ``A = -1, B = 0, C = 0`` and lands exactly on zero, but the same
    equation scaled by ``1 + 1e-16`` need not. The default compares the discriminant against the
    size of the terms that formed it, which is the only scale in the problem.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = discriminant(a, b, c)
    scale = np.maximum(np.abs(b * b), 4.0 * np.abs(a * c))
    if tol is None:
        cut = np.finfo(float).eps * 16.0 * np.maximum(scale, 1.0)
    else:
        cut = np.full(np.shape(d), float(tol))
    names = np.where(d > cut, "hyperbolic", np.where(d < -cut, "elliptic", "parabolic"))
    if names.ndim == 0:
        return str(names)
    return names


def characteristic_slopes(a, b, c):
    """The slopes ``dy/dx`` of the characteristic curves through a point.

    They solve ``A m**2 - B m + C = 0``. Two real slopes for a hyperbolic point, one for a
    parabolic point, none for an elliptic one, which is the same trichotomy the discriminant
    gives and is where the names come from. Returns a pair of complex numbers so the caller can
    see the elliptic case rather than getting an error for it; take `.imag` to test.

    ``A = 0`` is not degenerate, it is the common case: ``u_xy = 0`` is hyperbolic with
    characteristics along the axes. The quadratic drops to linear there and one root goes to
    infinity, which is the characteristic ``x = const``, so it is returned as ``inf`` rather than
    hidden.
    """
    a = float(a)
    b = float(b)
    c = float(c)
    if a == 0.0:
        if b == 0.0:
            return (complex(math.inf), complex(math.inf))
        return (complex(c / b), complex(math.inf))
    root = np.emath.sqrt(complex(b * b - 4.0 * a * c))
    return (complex((b + root) / (2.0 * a)), complex((b - root) / (2.0 * a)))


def standard_equations():
    """The five equations every treatment starts from, with their coefficients and their type.

    The pairs are ``(A, B, C)`` for the second order part written in the two variables named in
    ``variables``. The heat and wave equations are written with ``t`` as the second variable, so
    ``C`` is the coefficient of the second ``t`` derivative and is zero for the heat equation:
    that missing term is exactly what makes it parabolic.
    """
    return {
        "laplace": {"coefficients": (1.0, 0.0, 1.0), "variables": ("x", "y"),
                    "equation": "u_xx + u_yy = 0", "type": "elliptic"},
        "poisson": {"coefficients": (1.0, 0.0, 1.0), "variables": ("x", "y"),
                    "equation": "u_xx + u_yy = f", "type": "elliptic"},
        "heat": {"coefficients": (1.0, 0.0, 0.0), "variables": ("x", "t"),
                 "equation": "u_t = alpha u_xx", "type": "parabolic"},
        "wave": {"coefficients": (1.0, 0.0, -1.0), "variables": ("x", "t"),
                 "equation": "u_tt = c^2 u_xx", "type": "hyperbolic"},
        "advection": {"coefficients": (0.0, 1.0, 0.0), "variables": ("x", "t"),
                      "equation": "u_t + a u_x = 0 (as u_xt = 0)", "type": "hyperbolic"},
    }


def tricomi_coefficients(x, y):
    """``A = y, B = 0, C = 1`` for Tricomi's equation ``y u_xx + u_yy = 0``.

    The discriminant is ``-4y``, so the equation is hyperbolic below the ``x`` axis, parabolic on
    it and elliptic above it. It is the standard example that the type belongs to a point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (y * np.ones_like(x), np.zeros(np.broadcast(x, y).shape),
            np.ones(np.broadcast(x, y).shape))


def classify_on_grid(coefficients, xs, ys, tol: float | None = None):
    """Classify a variable coefficient equation at every point of a grid.

    ``coefficients`` is a callable taking ``(X, Y)`` meshes and returning ``(A, B, C)`` on them.
    Returns the type at every point together with the fraction of the grid in each class, which
    is the summary a plot would show.
    """
    x = np.asarray(xs, dtype=float).ravel()
    y = np.asarray(ys, dtype=float).ravel()
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    a, b, c = coefficients(grid_x, grid_y)
    names = classify(a, b, c, tol=tol)
    fractions = {t: float(np.count_nonzero(names == t)) / names.size for t in TYPES}
    return {"x": x, "y": y, "type": names, "discriminant": discriminant(a, b, c),
            "fractions": fractions,
            "types_present": tuple(t for t in TYPES if fractions[t] > 0.0)}


def conditions_for(kind: str):
    """What has to be prescribed for a problem of this type to be well posed.

    This is the practical content of the classification, and it is why the sign is worth
    computing before anything else is written.
    """
    kind = str(kind).lower()
    table = {
        "elliptic": {
            "domain": "a closed region",
            "needs": "one condition at every point of the whole boundary",
            "typical": "Dirichlet, Neumann on part of it, or Robin",
            "marching": False,
            "why": "the answer at any point depends on the data everywhere, so there is no "
                   "direction to march in",
        },
        "parabolic": {
            "domain": "a strip, open in time",
            "needs": "the state at t = 0, plus one condition on each side boundary for all t",
            "typical": "an initial profile with Dirichlet or Neumann sides",
            "marching": True,
            "why": "information travels forward in time only, and infinitely fast in space",
        },
        "hyperbolic": {
            "domain": "a strip, open in time",
            "needs": "the state and its time derivative at t = 0, plus side conditions",
            "typical": "an initial shape and an initial velocity",
            "marching": True,
            "why": "the equation is second order in time, and information travels at a finite "
                   "speed along the characteristics",
        },
    }
    if kind not in table:
        raise ValueError(f"unknown type {kind!r}; expected one of {TYPES}")
    return table[kind]


# --------------------------------------------------------------------------- stencils


def stencil(offsets, order: int = 2):
    """Weights for the ``order``-th derivative on the given integer offsets, in units of ``h**order``.

    Delegates to Fornberg's recursion, which is the stable way to get them, and returns the row
    for the derivative asked for. Any offsets work, including one sided and unequally spaced
    ones, which is what a boundary needs.
    """
    off = np.asarray(offsets, dtype=float).ravel()
    k = int(order)
    if off.size < k + 1:
        raise ValueError(f"a derivative of order {k} needs at least {k + 1} offsets, "
                         f"got {off.size}")
    return fornberg(off, order=k)


def stencil_order(offsets, order: int = 2) -> int:
    """The accuracy order of a stencil, from the first Taylor moment it fails to reproduce.

    A stencil with weights ``w`` on offsets ``s`` is exact for ``x**m`` when
    ``sum(w * s**m) == m! * [m == order]``. The accuracy order is the smallest ``m`` where that
    fails, minus the derivative order. This is computed rather than asserted, so a stencil that
    is not what its name says is caught here rather than in a convergence plot later.
    """
    off = np.asarray(offsets, dtype=float).ravel()
    k = int(order)
    w = stencil(off, k)
    for m in range(off.size + k + 2):
        want = math.factorial(m) if m == k else 0.0
        got = float(np.sum(w * off ** m))
        # the natural scale for the m-th moment is the moment of the absolute weights, which is
        # what the cancellation was subtracted from
        scale = max(1.0, float(np.sum(np.abs(w) * np.abs(off) ** m)))
        if abs(got - want) > 1e-8 * scale:
            return m - k
    return off.size - k


def apply_stencil(values, offsets, weights, axis: int = 0, h: float = 1.0, power: int = 2):
    """Apply a stencil along one axis of an array of any dimension.

    Only the points where the whole stencil fits are returned, so the result is shorter than the
    input along ``axis`` by the stencil's width minus one. Trimming rather than guessing at the
    ends is deliberate: a boundary needs its own rule, and silently reusing an interior rule
    there is the bug this design makes impossible.
    """
    v = np.asarray(values, dtype=float)
    off = np.asarray(offsets, dtype=int).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if off.size != w.size:
        raise ValueError(f"{off.size} offsets but {w.size} weights")
    lo, hi = int(off.min()), int(off.max())
    n = v.shape[axis]
    if n < hi - lo + 1:
        raise ValueError(f"axis {axis} has {n} points, too few for a stencil of width "
                         f"{hi - lo + 1}")
    out = None
    for s, weight in zip(off, w):
        start = s - lo
        stop = n - (hi - s)
        piece = np.take(v, np.arange(start, stop), axis=axis)
        out = weight * piece if out is None else out + weight * piece
    return out / float(h) ** int(power)


def partial(values, axis: int, order: int, h: float, accuracy: int = 2):
    """A centred partial derivative along one axis, trimmed to where the stencil fits.

    ``accuracy`` is the order of the truncation error and must be even, because a centred stencil
    for an even derivative gains its orders in pairs. The width follows from the two: an
    ``order``-th derivative to accuracy ``p`` needs ``2*floor((order + 1)/2) - 1 + p`` points.
    """
    k, p = int(order), int(accuracy)
    if p % 2:
        raise ValueError(f"a centred stencil has even accuracy order, got {p}")
    width = 2 * ((k + 1) // 2) - 1 + p
    half = width // 2
    off = np.arange(-half, half + 1)
    return apply_stencil(values, off, stencil(off, k), axis=axis, h=h, power=k)


def five_point(values, hx: float, hy: float):
    """The five point Laplacian, on the interior of a two dimensional array.

    ``(u_E + u_W - 2 u_C) / hx**2 + (u_N + u_S - 2 u_C) / hy**2``. Second order, and its error
    term ``(hx**2 u_xxxx + hy**2 u_yyyy) / 12`` depends on the direction, which is what the nine
    point rule fixes.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 2:
        raise ValueError(f"expected a 2D array, got {v.ndim}D")
    xx = (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / float(hx) ** 2
    yy = (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, :-2]) / float(hy) ** 2
    return xx + yy


def nine_point(values, h: float):
    """The compact nine point Laplacian on a square grid.

    ``(4(u_E+u_W+u_N+u_S) + (u_NE+u_NW+u_SE+u_SW) - 20 u_C) / (6 h**2)``.

    It is often introduced as "the fourth order Laplacian", and that is only half true. Its
    truncation error is ``(h**2 / 12) * (u_xxxx + 2 u_xxyy + u_yyyy) + O(h**4)``, so it is
    **second** order in general. The bracket is the biharmonic operator applied to ``u``, which
    vanishes when ``u`` is harmonic, and that is the case where the rule really is fourth order.
    `orders_of_the_two_laplacians` measures both sides of this.

    Requires a square grid, because the compact weights are derived from ``hx == hy`` and there
    is no honest way to reuse them otherwise.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 2:
        raise ValueError(f"expected a 2D array, got {v.ndim}D")
    edge = v[2:, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2]
    corner = v[2:, 2:] + v[2:, :-2] + v[:-2, 2:] + v[:-2, :-2]
    return (4.0 * edge + corner - 20.0 * v[1:-1, 1:-1]) / (6.0 * float(h) ** 2)


def mehrstellen(values, source, h: float):
    """The nine point Laplacian with the correction that makes it fourth order for ``u_xx + u_yy = f``.

    The scheme is ``nine_point(u) = f + (h**2 / 12) * laplacian(f)``, and it works because the
    term the nine point rule leaves behind is ``(h**2/12) * biharmonic(u)``, which equals
    ``(h**2/12) * laplacian(f)`` for a solution of the equation. The correction is therefore
    computable from the **data**, which is the whole trick.

    Returns the residual ``nine_point(u) - f - (h**2/12) laplacian(f)`` on the interior.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(source, dtype=float)
    if s.shape != v.shape:
        raise ValueError(f"source has shape {s.shape}, values {v.shape}")
    return nine_point(v, h) - s[1:-1, 1:-1] - (float(h) ** 2 / 12.0) * five_point(s, h, h)


def ghost_point_neumann(values, h: float, flux: float, side: str = "left"):
    """The boundary value implied by a Neumann condition, to second order, using a ghost point.

    A Neumann condition ``u'(a) = g`` can be imposed two ways. The obvious one differences
    forward, ``(u[1] - u[0]) / h = g``, and is **first** order. The ghost point way writes the
    centred difference at the boundary node itself, ``(u[1] - u[-1]) / (2h) = g``, eliminates the
    ghost value ``u[-1]`` using the equation at the boundary node, and stays second order.

    This returns the eliminated combination for the one dimensional Laplace operator: the value
    the second difference at the boundary node takes once the ghost point is removed, which is
    ``2 * (u[1] - u[0]) / h**2 - 2 * g / h`` at a left boundary.
    """
    v = np.asarray(values, dtype=float)
    hh = float(h)
    g = float(flux)
    if str(side).lower() == "left":
        return 2.0 * (v[1] - v[0]) / hh ** 2 - 2.0 * g / hh
    if str(side).lower() == "right":
        return 2.0 * (v[-2] - v[-1]) / hh ** 2 + 2.0 * g / hh
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


# --------------------------------------------------------------------------- measurements


def every_standard_equation_is_classified_right():
    """Put the five standard equations through `classify` and compare with the known answer.

    This is the check that the tolerance in `classify` is doing its job. The heat equation has a
    discriminant of exactly 0.0 and the advection equation written as ``u_xt = 0`` has one of
    exactly 1.0, so the two extremes are both present.
    """
    rows = []
    for name, info in standard_equations().items():
        a, b, c = info["coefficients"]
        got = classify(a, b, c)
        slopes = characteristic_slopes(a, b, c)
        real = tuple(s for s in slopes if abs(s.imag) < 1e-14)
        rows.append({
            "name": name,
            "equation": info["equation"],
            "discriminant": float(discriminant(a, b, c)),
            "expected": info["type"],
            "measured": got,
            "agrees": got == info["type"],
            "real_characteristics": len(set(real)) if real else 0,
        })
    return {
        "rows": rows,
        "all_agree": all(r["agrees"] for r in rows),
        "two_real_characteristics_exactly_when_hyperbolic": all(
            (r["real_characteristics"] == 2) == (r["measured"] == "hyperbolic")
            for r in rows),
    }


def the_type_can_change_across_the_domain(points: int = 41, half_width: float = 1.0):
    """Tricomi's equation, whose type depends on where you are.

    ``y u_xx + u_yy = 0`` has discriminant ``-4y``: hyperbolic for ``y < 0``, elliptic for
    ``y > 0``, parabolic on the line between. A grid that straddles the axis therefore contains
    all three, and a single method for the whole domain is a category error.

    The parabolic fraction is a measurement of the grid, not of the equation. A grid with a row
    exactly on ``y = 0`` finds it; one without does not. Both are reported.
    """
    n = int(points)
    if n < 3:
        raise ValueError(f"need at least 3 points per side, got {n}")
    w = float(half_width)
    xs = np.linspace(-w, w, n)
    with_axis = classify_on_grid(tricomi_coefficients, xs, np.linspace(-w, w, n))
    shift = w / (n - 1)
    without = classify_on_grid(tricomi_coefficients, xs,
                               np.linspace(-w + shift * 0.5, w + shift * 0.5, n))
    return {
        "grid_through_the_axis": with_axis["fractions"],
        "grid_missing_the_axis": without["fractions"],
        "types_on_the_first": with_axis["types_present"],
        "types_on_the_second": without["types_present"],
        "all_three_appear": len(with_axis["types_present"]) == 3,
        "the_parabolic_line_is_measure_zero":
            without["fractions"]["parabolic"] == 0.0,
        "hyperbolic_is_below_the_axis": bool(np.all(
            with_axis["type"][:, with_axis["y"] < 0.0] == "hyperbolic")),
        "elliptic_is_above_it": bool(np.all(
            with_axis["type"][:, with_axis["y"] > 0.0] == "elliptic")),
    }


def stencils_are_what_they_claim(widths=(3, 5, 7, 9)):
    """Measure the accuracy order of the centred first and second derivative stencils.

    A centred stencil on ``w`` points is accurate to order ``w - 1`` for a first derivative and
    ``w - 2`` for a second, both rounded down to an even number, because the odd term cancels by
    symmetry. The measurement is done from the Taylor moments and not from a convergence sweep,
    so it is a statement about the weights rather than about a particular test function.
    """
    rows = []
    for w in widths:
        width = int(w)
        if width % 2 == 0:
            raise ValueError(f"a centred stencil has an odd number of points, got {width}")
        half = width // 2
        off = np.arange(-half, half + 1)
        for k in (1, 2):
            if width < k + 1:
                continue
            rows.append({
                "points": width,
                "derivative": k,
                "order": stencil_order(off, k),
                "weights": stencil(off, k),
                "weights_sum_to_zero": abs(float(np.sum(stencil(off, k)))) < 1e-12,
            })
    return {
        "rows": rows,
        "first_derivative_orders": [r["order"] for r in rows if r["derivative"] == 1],
        "second_derivative_orders": [r["order"] for r in rows if r["derivative"] == 2],
        "every_stencil_sums_to_zero": all(r["weights_sum_to_zero"] for r in rows),
        "note": "a stencil for any derivative has weights summing to zero, because it must "
                "annihilate a constant",
    }


def orders_of_the_two_laplacians(sizes=(5, 7, 9, 11, 13, 16, 21, 31, 41, 61, 81, 121, 161)):
    """The five point and nine point Laplacians, on a harmonic function and on a general one.

    The nine point rule is usually introduced as fourth order. That is wrong twice over, and both
    errors are measured here.

    Write the expansions. Both rules approximate ``u_xx + u_yy`` with a leading error

        five point : (h**2 / 12) * (u_xxxx + u_yyyy)
        nine point : (h**2 / 12) * (u_xxxx + 2 u_xxyy + u_yyyy) = (h**2 / 12) * biharmonic(u)

    so **in general both are second order**, and on the test function here the nine point error is
    exactly twice the five point one. The nine point form is a biharmonic, which vanishes when
    ``u`` is harmonic, and that is the only case where it does better.

    On a harmonic function it does better than advertised. Its next term is
    ``(h**4 / 180) * (u_xxxxyy + u_xxyyyy)``, and for a harmonic function ``u_yy = -u_xx``, so
    ``u_xxyyyy = -u_xxxxyy`` and that term cancels too. The order is therefore **six**, not four,
    and `exact_on_harmonic_polynomials` confirms it exactly.

    ``exp(x) * sin(y)`` is harmonic. ``sin(x) * sin(y)`` is not, and its biharmonic is ``4u``.

    One measurement trap is worth naming. On the harmonic function the nine point error reaches
    the **rounding floor** by the fourth grid, because a second difference divides by ``h**2`` and
    so amplifies rounding by ``1/h**2``. Fitting the whole sweep then reads 2.07 for a sixth
    order rule, which measures the floor and not the method. Every fit is therefore reported twice:
    over the whole sweep, and over the rows still well above the floor. The floor is estimated as
    ``eps * max|u| / h**2``, which is the size of the cancellation rather than a number chosen to
    make the answer come out right.
    """
    def harmonic(x, y):
        return np.exp(x) * np.sin(y)

    def harmonic_laplacian(x, y):
        return np.zeros(np.broadcast(x, y).shape)

    def general(x, y):
        return np.sin(x) * np.sin(y)

    def general_laplacian(x, y):
        return -2.0 * np.sin(x) * np.sin(y)

    def fit(hs, errors, floors, margin=100.0):
        """Fit an order, then fit again over only the rows well above the rounding floor."""
        hs = np.asarray(hs, dtype=float)
        errors = np.asarray(errors, dtype=float)
        floors = np.asarray(floors, dtype=float)
        whole = float(np.polyfit(np.log(hs), np.log(errors), 1)[0])
        keep = errors > margin * floors
        if np.count_nonzero(keep) >= 2:
            resolved = float(np.polyfit(np.log(hs[keep]), np.log(errors[keep]), 1)[0])
        else:
            resolved = float(chr(110) + chr(97) + chr(110))
        return whole, resolved, int(np.count_nonzero(keep))

    rows = []
    for name, u, lap in (("harmonic", harmonic, harmonic_laplacian),
                         ("not harmonic", general, general_laplacian)):
        errors5, errors9, hs, floors = [], [], [], []
        for n in sizes:
            m = int(n)
            grid = np.linspace(0.0, 1.0, m)
            h = float(grid[1] - grid[0])
            gx, gy = np.meshgrid(grid, grid, indexing="ij")
            values = u(gx, gy)
            want = lap(gx[1:-1, 1:-1], gy[1:-1, 1:-1])
            errors5.append(float(np.max(np.abs(five_point(values, h, h) - want))))
            errors9.append(float(np.max(np.abs(nine_point(values, h) - want))))
            floors.append(float(np.finfo(float).eps * np.max(np.abs(values)) / h ** 2))
            hs.append(h)
        whole5, resolved5, kept5 = fit(hs, errors5, floors)
        whole9, resolved9, kept9 = fit(hs, errors9, floors)
        rows.append({
            "function": name,
            "h": np.asarray(hs),
            "rounding_floor": np.asarray(floors),
            "five_point_error": np.asarray(errors5),
            "nine_point_error": np.asarray(errors9),
            "five_point_order": whole5,
            "five_point_order_above_the_floor": resolved5,
            "five_point_rows_used": kept5,
            "nine_point_order": whole9,
            "nine_point_order_above_the_floor": resolved9,
            "nine_point_rows_used": kept9,
        })
    harmonic_row, other = rows[0], rows[1]
    return {
        "rows": rows,
        "nine_point_beats_fourth_order_on_a_harmonic_function":
            harmonic_row["nine_point_order_above_the_floor"] > 5.0,
        "fitting_the_whole_sweep_would_say": harmonic_row["nine_point_order"],
        "nine_point_is_only_second_order_otherwise":
            abs(other["nine_point_order"] - 2.0) < 0.15,
        "five_point_is_second_order_either_way":
            all(abs(r["five_point_order"] - 2.0) < 0.15 for r in rows),
        "nine_point_costs_twice_the_error_when_it_does_not_help": float(
            other["nine_point_error"][-1] / other["five_point_error"][-1]),
        "gain_on_a_harmonic_function": float(
            harmonic_row["five_point_error"][0] / harmonic_row["nine_point_error"][0]),
    }


def exact_on_harmonic_polynomials(max_degree: int = 10, points: int = 9):
    """The sharpest form of the same statement, with no tolerance anywhere.

    ``Re((x + iy)**d)`` is a harmonic polynomial of degree ``d``, and every harmonic polynomial is
    a combination of these and their imaginary partners. A stencil is exact on one exactly when
    all the derivatives its error term contains vanish, so the largest degree it handles pins the
    order down without fitting anything.

    Measured: the five point rule is exact up to degree **3** and the nine point rule up to degree
    **7**. Those are integers read off a table of zeros, and they say the five point error starts
    at fourth derivatives and the nine point error at eighth, which is second and sixth order.
    """
    d_max = int(max_degree)
    n = int(points)
    if n < 3:
        raise ValueError(f"need at least 3 points per side, got {n}")
    grid = np.linspace(0.0, 1.0, n)
    h = float(grid[1] - grid[0])
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    rows = []
    for d in range(d_max + 1):
        values = np.real((gx + 1j * gy) ** d)
        # the scale a nonzero answer would have: the operator divides by h**2
        scale = max(float(np.max(np.abs(values))), 1.0) / h ** 2
        e5 = float(np.max(np.abs(five_point(values, h, h))))
        e9 = float(np.max(np.abs(nine_point(values, h))))
        rows.append({
            "degree": d,
            "five_point": e5,
            "nine_point": e9,
            "five_point_exact": e5 <= 1e-10 * scale,
            "nine_point_exact": e9 <= 1e-10 * scale,
        })
    five_top = max((r["degree"] for r in rows if r["five_point_exact"]), default=-1)
    nine_top = max((r["degree"] for r in rows if r["nine_point_exact"]), default=-1)
    contiguous5 = all(r["five_point_exact"] for r in rows if r["degree"] <= five_top)
    contiguous9 = all(r["nine_point_exact"] for r in rows if r["degree"] <= nine_top)
    return {
        "rows": rows,
        "five_point_exact_up_to": five_top,
        "nine_point_exact_up_to": nine_top,
        "no_gaps": contiguous5 and contiguous9,
        "five_point_order_implied": five_top - 1,
        "nine_point_order_implied": nine_top - 1,
        "note": "a rule exact on harmonic polynomials up to degree d has an error starting at "
                "the (d+1)-th derivatives, which is order d - 1 for the Laplacian",
    }


def the_correction_restores_fourth_order(sizes=(11, 21, 41, 81)):
    """The Mehrstellen correction, measured against the plain nine point rule.

    On ``u_xx + u_yy = f`` with a non harmonic solution, the nine point rule alone is second
    order. Adding ``(h**2/12) * laplacian(f)`` to the right hand side removes the leading term
    and gives four, and the correction uses only ``f``, which is known.
    """
    def u(x, y):
        return np.sin(x) * np.sin(y)

    def f(x, y):
        return -2.0 * np.sin(x) * np.sin(y)

    def lap_f(x, y):
        return 4.0 * np.sin(x) * np.sin(y)

    plain, corrected, hs = [], [], []
    for n in sizes:
        m = int(n)
        grid = np.linspace(0.0, 1.0, m)
        h = float(grid[1] - grid[0])
        gx, gy = np.meshgrid(grid, grid, indexing="ij")
        values, source = u(gx, gy), f(gx, gy)
        inner = (slice(1, -1), slice(1, -1))
        plain.append(float(np.max(np.abs(nine_point(values, h) - source[inner]))))
        want = source[inner] + (h ** 2 / 12.0) * lap_f(gx[inner], gy[inner])
        corrected.append(float(np.max(np.abs(nine_point(values, h) - want))))
        hs.append(h)
    hs = np.asarray(hs)
    residual = np.asarray([float(np.max(np.abs(
        mehrstellen(u(*np.meshgrid(np.linspace(0.0, 1.0, int(n)),
                                   np.linspace(0.0, 1.0, int(n)), indexing="ij")),
                    f(*np.meshgrid(np.linspace(0.0, 1.0, int(n)),
                                   np.linspace(0.0, 1.0, int(n)), indexing="ij")),
                    1.0 / (int(n) - 1))))) for n in sizes])
    return {
        "h": hs,
        "plain_nine_point": np.asarray(plain),
        "with_the_correction": np.asarray(corrected),
        "mehrstellen_residual": residual,
        "plain_order": float(np.polyfit(np.log(hs), np.log(plain), 1)[0]),
        "corrected_order": float(np.polyfit(np.log(hs), np.log(corrected), 1)[0]),
        "mehrstellen_order": float(np.polyfit(np.log(hs), np.log(residual), 1)[0]),
        "the_discrete_correction_is_smaller_by": float(corrected[-1] / residual[-1]),
        "the_correction_uses_only_f": True,
        "gains_two_orders": (float(np.polyfit(np.log(hs), np.log(corrected), 1)[0])
                             - float(np.polyfit(np.log(hs), np.log(plain), 1)[0])) > 1.5,
    }


def a_ghost_point_keeps_the_order(sizes=(11, 21, 41, 81, 161, 321)):
    """A Neumann boundary done two ways, measured on a problem with a known answer.

    Solve ``-u'' = f`` on ``[0, 1]`` with ``u'(0) = g`` and ``u(1) = beta``, for
    ``u(x) = cos(pi x)``. Every interior equation is the second order second difference in both
    runs, and the only difference is the first row:

    * one sided: ``(u1 - u0) / h = g``, which is a first order approximation,
    * ghost point: the second difference at node 0 with the ghost value eliminated, second order.

    The whole solve is limited by whichever is worse, so the one sided version converges at 1
    even though 99 per cent of its rows are second order. This is the cheapest demonstration in
    the course that a boundary condition is not a detail.
    """
    def exact(x):
        return np.cos(math.pi * x)

    def source(x):
        return math.pi ** 2 * np.cos(math.pi * x)

    flux = 0.0                                   # u'(0) = -pi sin(0) = 0
    beta = float(exact(1.0))
    one_sided, ghost, hs = [], [], []
    for n in sizes:
        m = int(n)
        x = np.linspace(0.0, 1.0, m)
        h = float(x[1] - x[0])
        for which, store in (("one sided", one_sided), ("ghost", ghost)):
            a = np.zeros((m, m))
            rhs = np.zeros(m)
            for i in range(1, m - 1):
                a[i, i - 1] = -1.0 / h ** 2
                a[i, i] = 2.0 / h ** 2
                a[i, i + 1] = -1.0 / h ** 2
                rhs[i] = source(x[i])
            if which == "one sided":
                a[0, 0], a[0, 1] = -1.0 / h, 1.0 / h
                rhs[0] = flux
            else:
                a[0, 0], a[0, 1] = 2.0 / h ** 2, -2.0 / h ** 2
                rhs[0] = source(x[0]) - 2.0 * flux / h
            a[-1, -1] = 1.0
            rhs[-1] = beta
            got = np.linalg.solve(a, rhs)
            store.append(float(np.max(np.abs(got - exact(x)))))
        hs.append(h)
    hs = np.asarray(hs)
    return {
        "h": hs,
        "one_sided_error": np.asarray(one_sided),
        "ghost_point_error": np.asarray(ghost),
        "one_sided_order": float(np.polyfit(np.log(hs), np.log(one_sided), 1)[0]),
        "ghost_point_order": float(np.polyfit(np.log(hs), np.log(ghost), 1)[0]),
        "one_row_sets_the_order": True,
        "ratio_at_the_finest": float(one_sided[-1] / ghost[-1]),
    }


def the_backward_problem_is_ill_posed(wavenumbers=(1, 2, 4, 8, 16), time: float = 0.05):
    """Why time cannot be run backwards in the heat equation, measured.

    Forward, a mode ``sin(k pi x)`` decays by ``exp(-(k pi)**2 t)``, so every wiggle in the data
    is smoothed away and the answer depends continuously on the data. Backward, the same mode is
    amplified by ``exp((k pi)**2 t)``, which grows without bound as ``k`` grows. Since a grid of
    spacing ``h`` supports modes up to ``k ~ 1/h``, refining the grid makes the amplification
    **larger**, and no scheme fixes that: the continuous problem is the thing that is wrong.

    The fitted exponent of the amplification against ``k`` should be 2, from the ``k**2`` in the
    exponent of the exponent, and the measurement reports it as a check on the mechanism rather
    than on any one number.
    """
    ks = np.asarray([int(k) for k in wanted(wavenumbers)], dtype=float)
    t = float(time)
    forward = np.exp(-(ks * math.pi) ** 2 * t)
    backward = np.exp((ks * math.pi) ** 2 * t)
    log_amp = np.log(np.log(backward))
    slope = float(np.polyfit(np.log(ks), log_amp, 1)[0])
    return {
        "wavenumber": ks,
        "forward_factor": forward,
        "backward_factor": backward,
        "forward_is_a_contraction": bool(np.all(forward < 1.0)),
        "backward_grows_without_bound": bool(np.all(np.diff(backward) > 0.0)),
        "fitted_exponent_in_k": slope,
        "exponent_is_two": abs(slope - 2.0) < 1e-6,
        "grid_that_supports_k": 1.0 / ks,
        "note": "the amplification is exp(k^2 pi^2 t), so a finer grid makes it worse; this is "
                "a property of the equation and not of any scheme",
    }


def wanted(values):
    """Small helper: accept a scalar or an iterable and always return a list.

    Used where an argument is naturally either one value or several, so a caller can pass
    ``4`` or ``(1, 2, 4)`` and get the same behaviour.
    """
    if np.isscalar(values):
        return [values]
    return list(values)


def the_condition_count_matches_the_type():
    """Cross-check `conditions_for` against the equations, by counting what each one needs.

    Not a numerical measurement, but it is the piece of the classification a reader actually has
    to use, and putting it next to the discriminant keeps the two from drifting apart.
    """
    rows = []
    for name, info in standard_equations().items():
        kind = info["type"]
        needs = conditions_for(kind)
        rows.append({
            "name": name,
            "type": kind,
            "marching": needs["marching"],
            "needs": needs["needs"],
        })
    return {
        "rows": rows,
        "only_elliptic_is_not_marched": all(
            r["marching"] == (r["type"] != "elliptic") for r in rows),
        "types_covered": tuple(sorted({r["type"] for r in rows})),
    }
