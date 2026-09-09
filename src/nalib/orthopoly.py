"""Orthogonal polynomials: the bases that make least squares approximation stable.

Why they exist
--------------
Lesson 54 measured the problem. Fitting in the monomial basis means solving a normal equation
system whose matrix is the Hilbert matrix, at condition number ``5.2e14`` by degree 10. The
fix is not a better solver. It is a better basis: if the basis is **orthogonal** in the inner
product the fit uses, the normal equations are diagonal and there is no system to solve at all.

    <p_i, p_j>_w = integral_a^b p_i p_j w  =  0   for i != j

Then the best ``L2`` approximation is just ``sum_k (<f, p_k> / <p_k, p_k>) p_k``, one integral
per coefficient, independent of every other coefficient. Truncating the sum gives the best
approximation of the lower degree, which is false in any non-orthogonal basis.

Where they come from
--------------------
`gram_schmidt` runs Gram-Schmidt on the monomials in a chosen weight and interval, which
constructs them from nothing. `stieltjes` runs the far better conditioned three-term version.

**Every** family of orthogonal polynomials satisfies a three-term recurrence

    p_{k+1}(x) = (x - a_k) p_k(x) - b_k p_{k-1}(x)

and that is a theorem, not a coincidence: it follows from ``<x p_k, p_j> = <p_k, x p_j>``, which
kills every coefficient below ``k - 1``. The proof is short and it is what makes the whole
subject computable, because a recurrence costs ``O(n)`` and is numerically stable, while
Gram-Schmidt costs ``O(n^2)`` inner products. Whether Gram-Schmidt also **loses** orthogonality
depends on which form of it you write: the classical form reaches ``1.0e-6`` by degree 28 and the
modified form stays at ``2e-16``. `gram_schmidt` uses the modified form and says so.

The classical families
----------------------
=============  ==================  ==================  ===========================
family         interval            weight              where it turns up
=============  ==================  ==================  ===========================
Legendre       ``[-1, 1]``         ``1``               Gauss-Legendre quadrature
Chebyshev T    ``[-1, 1]``         ``1/sqrt(1-x^2)``   minimax, lesson 47
Chebyshev U    ``[-1, 1]``         ``sqrt(1-x^2)``     second kind, Gauss-Chebyshev
Laguerre       ``[0, inf)``        ``e^{-x}``          integrals to infinity
Hermite        ``(-inf, inf)``     ``e^{-x^2}``        Gaussian integrals, physics
=============  ==================  ==================  ===========================

`FAMILIES` holds their recurrence coefficients in closed form, `evaluate` runs the recurrence,
and `orthogonality_report` checks the defining property rather than assuming it.

What they are used for later
----------------------------
The roots of the degree ``n`` member are the **Gauss quadrature nodes** of Part 9, and
`golub_welsch` computes them as the eigenvalues of the symmetric tridiagonal Jacobi matrix built
from the recurrence coefficients. That is the standard method and it is Part 6's symmetric
eigenvalue problem doing a job in a different subject.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- the families


def _legendre_ab(n):
    k = np.arange(int(n), dtype=float)
    a = np.zeros(int(n))
    b = np.zeros(int(n))
    b[0] = 2.0                                   # <p_0, p_0> on [-1, 1] with weight 1
    if n > 1:
        b[1:] = k[1:] ** 2 / (4.0 * k[1:] ** 2 - 1.0)
    return a, b


def _chebyshev_t_ab(n):
    a = np.zeros(int(n))
    b = np.zeros(int(n))
    b[0] = math.pi
    if n > 1:
        b[1] = 0.5
    if n > 2:
        b[2:] = 0.25
    return a, b


def _chebyshev_u_ab(n):
    a = np.zeros(int(n))
    b = np.zeros(int(n))
    b[0] = 0.5 * math.pi
    if n > 1:
        b[1:] = 0.25
    return a, b


def _laguerre_ab(n):
    k = np.arange(int(n), dtype=float)
    a = 2.0 * k + 1.0
    b = np.zeros(int(n))
    b[0] = 1.0
    if n > 1:
        b[1:] = k[1:] ** 2
    return a, b


def _hermite_ab(n):
    k = np.arange(int(n), dtype=float)
    a = np.zeros(int(n))
    b = np.zeros(int(n))
    b[0] = math.sqrt(math.pi)
    if n > 1:
        b[1:] = 0.5 * k[1:]
    return a, b


#: name -> (interval, weight, recurrence coefficient builder, plain English note)
#:
#: The coefficients are for the **monic** recurrence
#: ``p_{k+1} = (x - a_k) p_k - b_k p_{k-1}``, with ``b_0`` set to the total mass
#: ``integral w``, which is the convention that makes `golub_welsch` come out right.
FAMILIES = {
    "legendre": ((-1.0, 1.0), lambda x: np.ones_like(np.asarray(x, dtype=float)),
                 _legendre_ab, "weight 1, the plain L2 inner product"),
    "chebyshev_t": ((-1.0, 1.0),
                    lambda x: 1.0 / np.sqrt(np.maximum(1.0 - np.asarray(x, dtype=float) ** 2,
                                                       1e-300)),
                    _chebyshev_t_ab, "weight 1/sqrt(1-x^2), the minimax weight"),
    "chebyshev_u": ((-1.0, 1.0),
                    lambda x: np.sqrt(np.maximum(1.0 - np.asarray(x, dtype=float) ** 2, 0.0)),
                    _chebyshev_u_ab, "weight sqrt(1-x^2), the second kind"),
    "laguerre": ((0.0, np.inf), lambda x: np.exp(-np.asarray(x, dtype=float)),
                 _laguerre_ab, "weight e^-x on the half line"),
    "hermite": ((-np.inf, np.inf), lambda x: np.exp(-np.asarray(x, dtype=float) ** 2),
                _hermite_ab, "weight e^{-x^2} on the whole line"),
}


def recurrence_coefficients(family: str, n: int) -> dict:
    """The monic three-term coefficients ``a_k`` and ``b_k`` for one of the classical families."""
    key = str(family).lower()
    if key not in FAMILIES:
        raise ValueError(f"unknown family {family!r}, expected one of {sorted(FAMILIES)}")
    m = int(n)
    if m < 1:
        raise ValueError(f"need at least one coefficient, got {m}")
    interval, weight, builder, note = FAMILIES[key]
    a, b = builder(m)
    return {"family": key, "alpha": a, "beta": b, "interval": interval,
            "weight": weight, "note": note}


def evaluate(family: str, degree: int, x, monic: bool = True) -> np.ndarray:
    """Evaluate the degree ``degree`` member by its three-term recurrence.

    ``monic=True`` gives the monic normalisation, whose recurrence is the one `FAMILIES`
    stores. ``monic=False`` rescales to the classical normalisation, in which
    ``T_k(cos t) = cos kt`` and ``P_k(1) = 1``.
    """
    n = int(degree)
    if n < 0:
        raise ValueError(f"degree must be non-negative, got {n}")
    z = np.atleast_1d(np.asarray(x, dtype=float))
    rec = recurrence_coefficients(family, max(n, 1))
    a, b = rec["alpha"], rec["beta"]
    prev = np.zeros_like(z)
    curr = np.ones_like(z)
    for k in range(n):
        curr, prev = (z - a[k]) * curr - (b[k] if k > 0 else 0.0) * prev, curr
    if monic:
        return curr
    return curr * float(leading_coefficient(family, n))


def evaluate_derivative(family: str, degree: int, x) -> np.ndarray:
    """``p_k'`` from the differentiated recurrence, in the monic normalisation.

        p'_{k+1} = p_k + (x - a_k) p'_k - b_k p'_{k-1}

    Differentiating the recurrence is exact and costs the same ``O(n)``, so there is never a
    reason to difference the values numerically.
    """
    n = int(degree)
    if n < 0:
        raise ValueError(f"degree must be non-negative, got {n}")
    z = np.atleast_1d(np.asarray(x, dtype=float))
    rec = recurrence_coefficients(family, max(n, 1))
    a, b = rec["alpha"], rec["beta"]
    p_prev = np.zeros_like(z)
    p_curr = np.ones_like(z)
    d_prev = np.zeros_like(z)
    d_curr = np.zeros_like(z)
    for k in range(n):
        beta = b[k] if k > 0 else 0.0
        p_next = (z - a[k]) * p_curr - beta * p_prev
        d_next = p_curr + (z - a[k]) * d_curr - beta * d_prev
        p_prev, p_curr = p_curr, p_next
        d_prev, d_curr = d_curr, d_next
    return d_curr


def leading_coefficient(family: str, degree: int) -> float:
    """The leading coefficient of the **classical** normalisation, so `evaluate` can rescale.

    Monic means leading coefficient 1. The classical families are normalised differently, and
    these are the standard values.
    """
    key = str(family).lower()
    n = int(degree)
    if key == "legendre":
        return float(math.comb(2 * n, n)) / float(2 ** n) if n >= 0 else 1.0
    if key == "chebyshev_t":
        return 1.0 if n == 0 else float(2 ** (n - 1))
    if key == "chebyshev_u":
        return float(2 ** n)
    if key == "laguerre":
        return float((-1.0) ** n / math.factorial(n))
    if key == "hermite":
        return float(2 ** n)
    raise ValueError(f"unknown family {family!r}")


# --------------------------------------------------------------------------- construction


def _trapezoid_weights(u):
    """Trapezoid quadrature weights on a grid, so an integral becomes a plain dot product."""
    u = np.asarray(u, dtype=float)
    w = np.empty_like(u)
    d = np.diff(u)
    w[0] = 0.5 * d[0]
    w[-1] = 0.5 * d[-1]
    w[1:-1] = 0.5 * (d[:-1] + d[1:])
    return w


def weighted_grid(family: str, n_probe: int = 20001, degree: int = 8):
    """Nodes and quadrature weights for ``integral g(x) w(x) dx`` in the family's own weight.

    Each family gets the substitution that removes its own difficulty, so no grid ever samples
    a singular or exponentially large integrand:

    - Legendre: a plain uniform grid, the weight being 1.
    - Chebyshev T: ``x = cos(theta)``, which turns ``dx / sqrt(1-x^2)`` into ``d(theta)``
      exactly. Without it a uniform grid in ``x`` misses the endpoint singularity and the
      measured orthogonality reads **0.5 instead of 1e-15**, which looks like the theorem
      failing.
    - Chebyshev U: the same substitution, leaving ``sin^2(theta) d(theta)``.
    - Laguerre: a uniform grid on ``[0, x_max]`` with the weight carried explicitly, where
      ``x_max`` is set from ``degree``. The obvious substitution ``x = -log(u)`` with ``u``
      uniform turns ``e^{-x} dx`` into ``du`` exactly but puts almost no points where ``x`` is
      large, and the integrand ``p_i p_j`` grows like ``x^{i+j}`` there. At degree 6 that
      measured the orthogonality as **5e-2 instead of 1e-16**, so the honest grid is the one
      that resolves the tail.
    - Hermite: a stretched grid on the whole line, the Gaussian weight carried explicitly. It
      decays fast enough that no cutoff is needed.

    ``degree`` sets how far the Laguerre tail is followed, at ``x_max = 40 + 4 * degree``. The
    integrand there is ``x^{2 degree} e^{-x}``, which at the default is below ``1e-15`` relative
    to its peak. Raise it for higher degrees.

    Returns ``(nodes, weights)`` with ``sum(weights) = integral w`` up to quadrature error.
    """
    key = str(family).lower()
    if key not in FAMILIES:
        raise ValueError(f"unknown family {family!r}, expected one of {sorted(FAMILIES)}")
    m = int(n_probe)
    if m < 3:
        raise ValueError(f"need at least three grid points, got {m}")
    if key == "legendre":
        x = np.linspace(-1.0, 1.0, m)
        return x, _trapezoid_weights(x)
    if key in ("chebyshev_t", "chebyshev_u"):
        theta = np.linspace(0.0, math.pi, m)
        x = np.cos(theta)
        w = _trapezoid_weights(theta)
        if key == "chebyshev_u":
            w = w * np.sin(theta) ** 2
        return x, w
    if key == "laguerre":
        x = np.linspace(0.0, 40.0 + 4.0 * float(max(int(degree), 0)), m)
        return x, _trapezoid_weights(x) * np.exp(-x)
    s = np.linspace(-1.0 + 1e-9, 1.0 - 1e-9, m)
    x = s / (1.0 - s * s)
    jac = (1.0 + s * s) / (1.0 - s * s) ** 2
    return x, _trapezoid_weights(s) * jac * np.exp(-x * x)


def inner_product(f, g, family: str, n_probe: int = 20001) -> float:
    """``<f, g>_w`` in the family's own weight and interval."""
    x, dmu = weighted_grid(family, n_probe)
    a = np.asarray(f(x), dtype=float)
    b = np.asarray(g(x), dtype=float)
    v = a * b * dmu
    return float(np.sum(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)))


def gram_schmidt(degree: int, family: str = "legendre", n_probe: int = 4001,
                 classical: bool = False) -> dict:
    """Build the family from the monomials by Gram-Schmidt, and measure what it costs.

    This is the definition made into an algorithm, and it is the wrong way to do it, though not
    for the reason usually given. Each new member needs an inner product against every previous
    one, so the cost is ``O(n^2)`` integrals against `stieltjes`'s ``O(n)``. That is the real
    objection.

    **The orthogonality does not decay, because the loop below is the *modified* form.** It
    subtracts each projection as it goes rather than computing them all against the original
    vector, which is Part 5's distinction on vectors and behaves the same way on functions.
    Measured, the worst off-diagonal entry stays at ``2e-16`` out to degree 28. The *classical*
    form, which computes every projection against the original, reaches ``1.0e-6`` at the same
    degree, a factor of ``5e9`` worse. Writing the loop the obvious way would have lost the
    orthogonality; writing it this way does not.

    Returns the coefficients of each member in the monomial basis, which is the form that lets
    the loss be seen.
    """
    n = int(degree)
    if n < 0:
        raise ValueError(f"degree must be non-negative, got {n}")
    key = str(family).lower()
    interval, weight, _, _ = FAMILIES[key]
    if not (np.isfinite(interval[0]) and np.isfinite(interval[1])):
        raise ValueError(f"Gram-Schmidt here uses a finite interval, {key} lives on {interval}")
    t = np.linspace(interval[0], interval[1], int(n_probe))
    w = np.asarray(weight(t), dtype=float)
    dot = lambda u, v: float(np.trapezoid(u * v * w, t))
    columns = []
    coeffs = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        v = t ** k
        original = v
        c = np.zeros(n + 1)
        c[k] = 1.0
        for j, q in enumerate(columns):
            # classical projects the ORIGINAL vector, modified projects what is left of it
            factor = dot(original if classical else v, q) / dot(q, q)
            v = v - factor * q
            c = c - factor * coeffs[j]
        columns.append(v)
        coeffs[k] = c
    V = np.stack(columns, axis=1)
    gram = np.stack([[dot(V[:, i], V[:, j]) for j in range(n + 1)] for i in range(n + 1)])
    off = gram - np.diag(np.diag(gram))
    return {"monomial_coefficients": coeffs, "values": V, "grid": t,
            "gram": gram, "classical": bool(classical),
            "worst_off_diagonal": float(np.max(np.abs(off)) / max(np.max(np.abs(np.diag(gram))),
                                                                  1e-300)),
            "inner_products_used": (n + 1) * (n + 2) // 2}


def stieltjes(degree: int, family: str = "legendre", n_probe: int = 20001) -> dict:
    """Build the three-term coefficients by the Stieltjes procedure, in ``O(n)`` inner products.

        a_k = <x p_k, p_k> / <p_k, p_k>,     b_k = <p_k, p_k> / <p_{k-1}, p_{k-1}>

    Each step needs two inner products and nothing from further back than one step, because the
    recurrence theorem says every other coefficient is zero. That is the whole saving.

    Returns the computed coefficients next to the closed form ones, so the agreement can be
    checked instead of assumed.
    """
    n = int(degree)
    if n < 1:
        raise ValueError(f"need at least one step, got {n}")
    key = str(family).lower()
    t, dmu = weighted_grid(key, n_probe, degree=n)
    finite = np.isfinite(dmu) & np.isfinite(t)
    dot = lambda u, v: float(np.sum(np.where(finite, u * v * dmu, 0.0)))
    prev = np.zeros_like(t)
    curr = np.ones_like(t)
    norm_prev = 1.0
    a = np.zeros(n)
    b = np.zeros(n)
    for k in range(n):
        norm = dot(curr, curr)
        a[k] = dot(t * curr, curr) / norm
        b[k] = norm / norm_prev if k > 0 else norm
        curr, prev = (t - a[k]) * curr - (b[k] if k > 0 else 0.0) * prev, curr
        norm_prev = norm
    exact = recurrence_coefficients(key, n)
    scale = max(float(np.max(np.abs(exact["beta"]))), 1e-300)
    return {"alpha": a, "beta": b,
            "alpha_exact": exact["alpha"], "beta_exact": exact["beta"],
            "alpha_gap": float(np.max(np.abs(a - exact["alpha"]))),
            "beta_gap": float(np.max(np.abs(b - exact["beta"])) / scale),
            "inner_products_used": 2 * n}


# --------------------------------------------------------------------------- properties


def orthogonality_report(family: str, degree: int, n_probe: int = 40001) -> dict:
    """Check ``<p_i, p_j>_w = 0`` for ``i != j``, which is the defining property.

    Reported relative to the largest diagonal entry, so the number means "how far from
    orthogonal" rather than "how big are these polynomials".
    """
    n = int(degree)
    key = str(family).lower()
    t, dmu = weighted_grid(key, n_probe, degree=n)
    P = np.stack([evaluate(key, k, t) for k in range(n + 1)], axis=1)
    good = np.isfinite(dmu) & np.all(np.isfinite(P), axis=1)
    gram = np.stack([[float(np.sum(np.where(good, P[:, i] * P[:, j] * dmu, 0.0)))
                      for j in range(n + 1)] for i in range(n + 1)])
    diag = np.abs(np.diag(gram))
    off = np.abs(gram - np.diag(np.diag(gram)))
    return {"gram": gram, "diagonal": np.diag(gram),
            "worst_off_diagonal": float(np.max(off)),
            "relative": float(np.max(off) / max(float(np.max(diag)), 1e-300)),
            "is_orthogonal": bool(np.max(off) <= 1e-6 * max(float(np.max(diag)), 1e-300))}


def three_term_holds(family: str, degree: int, n_probe: int = 401) -> float:
    """The recurrence, checked pointwise against directly evaluated members.

    Returns the largest relative violation of
    ``p_{k+1}(x) = (x - a_k) p_k(x) - b_k p_{k-1}(x)`` over ``k`` and over a sample of ``x``.
    """
    n = int(degree)
    key = str(family).lower()
    interval, _, _, _ = FAMILIES[key]
    lo = interval[0] if np.isfinite(interval[0]) else -3.0
    hi = interval[1] if np.isfinite(interval[1]) else 3.0
    t = np.linspace(float(lo), float(hi), int(n_probe))
    rec = recurrence_coefficients(key, max(n + 1, 1))
    a, b = rec["alpha"], rec["beta"]
    worst = 0.0
    for k in range(n):
        lhs = evaluate(key, k + 1, t)
        rhs = (t - a[k]) * evaluate(key, k, t) - (b[k] if k > 0 else 0.0) * evaluate(key, k - 1, t) \
            if k > 0 else (t - a[k]) * evaluate(key, k, t)
        scale = max(float(np.max(np.abs(lhs))), 1e-300)
        worst = max(worst, float(np.max(np.abs(lhs - rhs))) / scale)
    return worst


def christoffel_darboux(family: str, degree: int, x, y, near_tol: float = 1e-7) -> dict:
    """The Christoffel-Darboux identity, which sums the reproducing kernel in closed form.

        sum_{k=0}^{n} p_k(x) p_k(y) / h_k
            = ( p_{n+1}(x) p_n(y) - p_n(x) p_{n+1}(y) ) / ( h_n (x - y) )

    with ``h_k = <p_k, p_k>``. It turns an ``O(n)`` sum into two evaluations, and it is what
    makes the Lebesgue function of a Gauss rule computable.

    **The closed form is 0/0 on the diagonal**, and it is not merely undefined at ``x = y``, it
    is inaccurate near it. Sampling ``x`` on a grid that lands within ``4e-17`` of ``y``
    measured an error of **0.505 against a value of 2.16**, which is a 23 percent error from a
    formula that is exact elsewhere to ``1e-16``. That is ordinary cancellation: a numerator
    that vanishes divided by a denominator that vanishes.

    The fix is the confluent form, obtained by L'Hopital,

        sum_{k=0}^{n} p_k(x)^2 / h_k
            = ( p'_{n+1}(x) p_n(x) - p'_n(x) p_{n+1}(x) ) / h_n

    which this function switches to whenever ``|x - y|`` is below ``near_tol`` times the scale
    of the points. Both branches are reported so the switch can be seen.
    """
    n = int(degree)
    key = str(family).lower()
    rec = recurrence_coefficients(key, n + 2)
    b = rec["beta"]
    h = np.cumprod(b[:n + 1])                    # h_k = b_0 b_1 ... b_k for the monic family
    a = np.atleast_1d(np.asarray(x, dtype=float))
    c = np.atleast_1d(np.asarray(y, dtype=float))
    direct = sum(evaluate(key, k, a) * evaluate(key, k, c) / h[k] for k in range(n + 1))
    gap = a - c
    scale_x = max(float(np.max(np.abs(a))), float(np.max(np.abs(c))), 1.0)
    near = np.abs(gap) <= float(near_tol) * scale_x
    with np.errstate(divide="ignore", invalid="ignore"):
        away = (evaluate(key, n + 1, a) * evaluate(key, n, c)
                - evaluate(key, n, a) * evaluate(key, n + 1, c)) / (h[n] * gap)
    mid = 0.5 * (a + c)
    confluent = (evaluate_derivative(key, n + 1, mid) * evaluate(key, n, mid)
                 - evaluate_derivative(key, n, mid) * evaluate(key, n + 1, mid)) / h[n]
    closed = np.where(near, confluent, away)
    scale = max(float(np.max(np.abs(direct))), 1e-300)
    naive = np.nan_to_num(away, nan=0.0, posinf=0.0, neginf=0.0)
    return {"sum": direct, "closed_form": closed, "closed_form_naive": away,
            "relative_gap": float(np.max(np.abs(direct - closed)) / scale),
            "relative_gap_without_the_confluent_branch":
                float(np.max(np.abs(direct - naive)) / scale),
            "used_confluent_branch": near,
            "norms": h}


# --------------------------------------------------------------------------- uses


def golub_welsch(family: str, n_nodes: int) -> dict:
    """Gauss quadrature nodes and weights from the recurrence, by one symmetric eigensolve.

    Build the symmetric tridiagonal **Jacobi matrix** with ``a_k`` on the diagonal and
    ``sqrt(b_k)`` off it. Its eigenvalues are the roots of ``p_n``, which are the nodes, and the
    weights are ``b_0`` times the squared first components of the eigenvectors.

    This is Part 6's symmetric eigenvalue problem answering a Part 9 question, and it is the
    method every library uses. Part 9 uses the nodes; this function exists to show where they
    come from.
    """
    n = int(n_nodes)
    if n < 1:
        raise ValueError(f"need at least one node, got {n}")
    rec = recurrence_coefficients(str(family).lower(), n)
    a, b = rec["alpha"], rec["beta"]
    J = np.diag(a)
    if n > 1:
        offs = np.sqrt(b[1:n])
        J = J + np.diag(offs, 1) + np.diag(offs, -1)
    vals, vecs = np.linalg.eigh(J)
    order = np.argsort(vals)
    nodes = vals[order]
    weights = b[0] * vecs[0, order] ** 2
    return {"nodes": nodes, "weights": weights, "total_weight": float(np.sum(weights)),
            "expected_total_weight": float(b[0]), "jacobi_matrix": J}


def least_squares_by_orthogonality(f, degree: int, family: str = "legendre",
                                   n_probe: int = 20001) -> dict:
    """The best ``L2`` approximation, one independent integral per coefficient.

        c_k = <f, p_k> / <p_k, p_k>

    No linear system is solved, because orthogonality has already diagonalised it. The
    consequence that matters in practice is that **truncating** the series gives the best
    approximation of the lower degree, so raising the degree never changes a coefficient
    already computed. `truncation_is_optimal` measures that.
    """
    n = int(degree)
    key = str(family).lower()
    t, dmu = weighted_grid(key, n_probe, degree=n)
    try:
        fv = np.asarray(f(t), dtype=float)
        if fv.shape != t.shape:
            raise ValueError
    except (TypeError, ValueError):
        fv = np.asarray([float(f(s)) for s in t], dtype=float)
    P = np.stack([evaluate(key, k, t) for k in range(n + 1)], axis=1)
    good = np.isfinite(dmu) & np.isfinite(fv) & np.all(np.isfinite(P), axis=1)
    intg = lambda v: float(np.sum(np.where(good, v, 0.0)))
    h = np.asarray([intg(P[:, k] * P[:, k] * dmu) for k in range(n + 1)])
    c = np.asarray([intg(fv * P[:, k] * dmu) for k in range(n + 1)]) / h

    def evaluate_fit(s):
        z = np.atleast_1d(np.asarray(s, dtype=float))
        return sum(c[k] * evaluate(key, k, z) for k in range(n + 1))

    fit = P @ c
    return {"coefficients": c, "norms": h, "family": key, "evaluate": evaluate_fit,
            "weighted_l2_error": float(np.sqrt(abs(intg((fv - fit) ** 2 * dmu)))),
            "integrals_used": 2 * (n + 1)}


def truncation_is_optimal(f, degree: int, family: str = "legendre",
                          n_probe: int = 20001) -> dict:
    """Fit at the full degree, then compare each truncation against a fresh fit of that degree.

    In an orthogonal basis the two are the same coefficients. In the monomial basis they are
    not: every coefficient changes when the degree changes. That is the practical reason to use
    an orthogonal basis even when a solver could handle the conditioning.
    """
    n = int(degree)
    full = least_squares_by_orthogonality(f, n, family, n_probe)
    gaps = []
    for k in range(n + 1):
        part = least_squares_by_orthogonality(f, k, family, n_probe)
        scale = max(float(np.max(np.abs(full["coefficients"][:k + 1]))), 1e-300)
        gaps.append(float(np.max(np.abs(part["coefficients"] - full["coefficients"][:k + 1])))
                    / scale)
    return {"degrees": np.arange(n + 1), "relative_change": np.asarray(gaps),
            "worst": float(np.max(gaps)),
            "coefficients_are_stable": bool(np.max(gaps) < 1e-8)}


def conditioning_against_monomials(degree: int, family: str = "legendre",
                                   n_probe: int = 4001) -> dict:
    """The Gram matrix condition number, orthogonal basis against monomials, same weight.

    Three numbers, because two of them are easy to misread.

    - ``monomial_condition`` is the real problem. On ``[0, 1]`` this matrix is exactly the
      Hilbert matrix and grows like ``e^{3.5 n}``.
    - ``orthogonal_condition`` is the **monic** family's Gram matrix. It is diagonal, so the
      number is only the ratio of the largest to the smallest ``<p_k, p_k>``, and for monic
      Legendre those norms decay like ``4^-k``. Reporting that as "the conditioning of the
      orthogonal basis" would be wrong: it measures a normalisation choice, not a difficulty.
    - ``orthonormal_condition`` divides each member by its own norm and is the honest figure.
      It is 1 up to quadrature error, at every degree.
    """
    n = int(degree)
    key = str(family).lower()
    interval = FAMILIES[key][0]
    if not (np.isfinite(interval[0]) and np.isfinite(interval[1])):
        raise ValueError(f"this comparison uses a finite interval, {key} lives on {interval}")
    t, dmu = weighted_grid(key, n_probe, degree=n)
    good = np.isfinite(dmu)
    intg = lambda v: float(np.sum(np.where(good, v, 0.0)))
    mono = np.stack([t ** k for k in range(n + 1)], axis=1)
    orth = np.stack([evaluate(key, k, t) for k in range(n + 1)], axis=1)
    gram = lambda B: np.stack([[intg(B[:, i] * B[:, j] * dmu) for j in range(n + 1)]
                               for i in range(n + 1)])
    gm = gram(mono)
    go = gram(orth)
    norms = np.sqrt(np.abs(np.diag(go)))
    gn = gram(orth / norms[None, :])
    return {"degree": n, "family": key,
            "monomial_condition": float(np.linalg.cond(gm)),
            "orthogonal_condition": float(np.linalg.cond(go)),
            "orthonormal_condition": float(np.linalg.cond(gn)),
            "ratio": float(np.linalg.cond(gm) / max(np.linalg.cond(gn), 1e-300))}
