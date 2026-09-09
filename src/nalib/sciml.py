"""Inverse problems, collocation with a trained basis, neural ODEs, and Krylov methods at scale.

Where the work is
-----------------
This module points Part 14's tools back at Parts 10 and 11, and the result is that most of what
gets called scientific machine learning is a method this course already has, with a different basis
or a different optimizer.

**An inverse problem is a least squares problem whose operator smooths.** The forward map integrates
against a kernel, so its singular values decay geometrically and the pseudoinverse amplifies
whatever the data did not contain. Every difficulty is lesson 33's, at a condition number that
refining the grid makes **worse** rather than better, which is the one place in this course where
more resolution is not more accuracy.

**A physics-informed network is collocation.** Lesson 76 picked a basis, wrote the residual at a set
of points, and solved for the coefficients. A physics-informed network does exactly that with a
nonlinear basis and a nonconvex fit, so the comparison to make is against collocation with a linear
basis rather than against nothing.

**A neural ODE is lesson 67's initial value problem with a fitted right-hand side.** Training it
needs the gradient of a solution with respect to the parameters, and lesson 97's two routes apply
unchanged: unroll the solver, or differentiate the equation. The second one is called the adjoint
method and it is the implicit function theorem again.

**A sparse grid is the classical competitor, not a tensor product.** The comparison that decides
whether a trained basis is worth using has to be against the best classical method, and for a smooth
problem on a cube that is a Smolyak sparse grid, whose size is ``C(level+d, d)`` rather than
``m**d``. Comparing against a tensor product measures the wrong thing and flatters the network.

**Krylov methods reappear because the matrices stop fitting.** Conjugate gradient needs only a
matrix-vector product, so it solves a regularized normal equation without forming it, and stopping
it early is itself a regularization with its own parameter: the iteration count.

What the measurements here show
-------------------------------
* **Refining the grid makes an inverse problem worse.** The condition number of a discretized blur
  reaches 8.9e+17 at 64 points, past what double precision can represent, while a second difference
  operator on the same grid grows like ``n**2`` with a fitted slope of 1.975 and stays usable.
* **The extra grid points are wasted.** Going from 128 to 256 points adds 128 unknowns and 3 usable
  singular values, because the spectrum decays by a factor of 0.657 per index whatever the
  resolution.
* **The discrete Picard condition locates the usable components.** The number of them falls from 61
  at exact data to 31 at a noise level of 1e-04 and 22 at 1e-02, and the plain pseudoinverse's error
  goes from 0.083 to 5.8e+07 over the same range.
* **Tikhonov is lesson 94's ridge with a different name**, matching its filter factors to 8.8e-12.
* **The L-curve corner is 18.7 times away from the oracle penalty and 1.087 times away from the
  oracle error**, because the error curve is nearly flat near its minimum. The penalty is much
  harder to find than it is worth finding.
* **The two penalty operators nearly agree here**, at a largest gain of 1.037, and the reason is
  measurable: the blur's own right singular vectors are ordered by roughness, over a range of 204,
  so shrinking the small directions is already a smoothness prior.
* **Conjugate gradient regularizes by stopping early.** The error falls to 0.1239 at iteration 60
  and then rises to 14.6 by iteration 400, and the best iteration moves from 98 to 7 as the noise
  goes from 1e-05 to 1e-02. At its best it beats the best Tikhonov solution by 0.9 per cent.
* **A physics-informed network is collocation, and training it buys a factor of 10.7** over the same
  nonlinear basis left at its random values, taking the error from 0.0143 to 0.00133.
* **A linear basis of the same size matches it in one solve**, at 0.00125 against 0.00133, and a
  basis twice the size beats it by 82000, at 1.6e-11. The Chebyshev error falls by 1.2e+10 as the
  term count goes from 8 to 24, which is spectral convergence and the nonlinear fit has nothing
  like it.
* **The adjoint method gives the gradient at constant memory.** Its memory is 18 numbers at every
  step count while unrolled differentiation needs 6156 at 512 steps, a ratio of 342.
* **Discretize-then-optimize and optimize-then-discretize are different gradients** that agree only
  in the limit, differing by 2.2e-02 at 16 steps and 6.8e-04 at 512, falling like ``h**0.998``.
* **A trained basis starts to beat a sparse grid at four dimensions**, the same crossover at two
  different parameter budgets. Below it the grid wins by up to 7.7e+10 and above it the network wins
  by at most 5.06, so both statements are true and they are not the same size.
* **The crossover is the grid collapsing, not the network improving.** The errors grow like
  ``10**(3.0 d)`` and ``10**(0.5 d)``, and at a fixed budget the sparse level falls from 16 to 4.
* **Past the crossover neither method is usable**, at a worst error of 1.44 against a threshold of
  0.1, so the winner there is the less bad of two answers nobody would use.
* **Training the hidden layer is worth between 0.02 and 15.** It makes things worse in one and two
  dimensions, where solving directly for the output layer already reaches 1e-13, and earns up to
  14.6 from three dimensions on.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- inverse problems


def blur(size: int, width: float = 0.05, span: float = 1.0) -> dict:
    """Discretize a Fredholm equation of the first kind with a Gaussian kernel.

    The forward map is ``(A x)(s) = integral k(s - t) x(t) dt``. Smoothing kernels have geometrically
    decaying singular values, so the inverse map amplifies high frequencies without bound. This is
    the standard model problem for an ill-posed inverse problem, and it is deblurring, tomography
    and heat conduction backwards in time all at once.
    """
    n = int(size)
    if n < 2:
        raise ValueError(f"need at least two grid points, got {n}")
    h = float(span) / n
    grid = (np.arange(n) + 0.5) * h
    difference = grid[:, None] - grid[None, :]
    matrix = h * np.exp(-0.5 * (difference / float(width)) ** 2) / (
        float(width) * math.sqrt(2.0 * math.pi))
    return {"matrix": matrix, "grid": grid, "step": h, "size": n, "width": float(width)}


def second_difference(size: int, span: float = 1.0) -> np.ndarray:
    """The operator of lesson 77, for comparison: also ill-conditioned, and far less so."""
    n = int(size)
    h = float(span) / (n + 1)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, -2.0 / h ** 2)
    index = np.arange(n - 1)
    matrix[index, index + 1] = 1.0 / h ** 2
    matrix[index + 1, index] = 1.0 / h ** 2
    return matrix


def test_signal(grid, kind: str = "mixed") -> np.ndarray:
    """The thing being recovered. ``smooth`` has no jump, ``mixed`` has one, ``box`` is all jump.

    Which one is used decides which penalty operator wins, so it is a parameter rather than a
    constant: a smoothness penalty is a statement about the answer and can only be checked against
    answers that do and do not have that property.
    """
    t = np.asarray(grid, dtype=float)
    smooth = np.exp(-60.0 * (t - 0.3) ** 2) + 0.4 * np.exp(-25.0 * (t - 0.7) ** 2)
    step = 0.7 * np.where((t > 0.6) & (t < 0.75), 1.0, 0.0)
    name = str(kind).lower()
    if name == "smooth":
        return smooth
    if name == "box":
        return step
    if name == "mixed":
        return np.exp(-60.0 * (t - 0.3) ** 2) + step
    raise ValueError(f"unknown signal kind {kind!r}")


def noisy_data(problem: dict, signal, level: float = 1e-4, seed: int = 42) -> dict:
    """Apply the forward map and add noise, keeping the exact answer for comparison."""
    exact = problem["matrix"] @ np.asarray(signal, dtype=float)
    rng = np.random.default_rng(int(seed))
    noise = float(level) * rng.standard_normal(exact.size)
    return {"data": exact + noise, "clean": exact, "noise": noise,
            "level": float(level), "truth": np.asarray(signal, dtype=float)}


def picard(problem: dict, data) -> dict:
    """The discrete Picard condition: compare ``|u_i' b|`` against ``sigma_i``.

    A solution exists and is stable only while the data coefficients decay faster than the singular
    values. Where they level off, at the noise floor, is where the useful information stops, and
    every component past that point contributes noise divided by a tiny number.
    """
    left, s, _ = np.linalg.svd(problem["matrix"], full_matrices=False)
    coefficients = np.abs(left.T @ np.asarray(data, dtype=float))
    ratio = coefficients / s
    floor = float(np.median(coefficients[-max(coefficients.size // 8, 1):]))
    crossing = int(np.argmax(s < floor)) if np.any(s < floor) else s.size
    return {
        "singular_values": s, "coefficients": coefficients, "ratio": ratio,
        "noise_floor": floor, "usable": crossing, "total": int(s.size),
    }


def tikhonov(problem: dict, data, penalty: float, order: int = 0) -> dict:
    """Regularized least squares with the identity or a derivative as the penalty operator.

    ``order = 0`` is lesson 94's ridge exactly. ``order = 2`` penalizes the second derivative of the
    solution instead of its size, which is a different prior and a different answer.
    """
    a = problem["matrix"]
    b = np.asarray(data, dtype=float)
    if int(order) == 0:
        operator = np.eye(a.shape[1])
    elif int(order) == 2:
        operator = np.zeros((a.shape[1] - 2, a.shape[1]))
        index = np.arange(operator.shape[0])
        operator[index, index] = 1.0
        operator[index, index + 1] = -2.0
        operator[index, index + 2] = 1.0
    else:
        raise ValueError(f"order must be 0 or 2, got {order}")
    stacked = np.vstack([a, math.sqrt(float(penalty)) * operator])
    padded = np.concatenate([b, np.zeros(operator.shape[0])])
    solution, *_ = np.linalg.lstsq(stacked, padded, rcond=None)
    return {
        "solution": solution, "penalty": float(penalty), "order": int(order),
        "residual": float(np.linalg.norm(a @ solution - b)),
        "penalty_norm": float(np.linalg.norm(operator @ solution)),
    }


def truncated_solution(problem: dict, data, keep: int) -> dict:
    """The other classical filter: keep the leading components and discard the rest outright."""
    left, s, right = np.linalg.svd(problem["matrix"], full_matrices=False)
    k = max(1, min(int(keep), s.size))
    coefficients = left.T @ np.asarray(data, dtype=float)
    solution = right[:k].T @ (coefficients[:k] / s[:k])
    return {"solution": solution, "keep": k,
            "residual": float(np.linalg.norm(problem["matrix"] @ solution
                                             - np.asarray(data, dtype=float)))}


def l_curve(problem: dict, data, penalties, order: int = 0, truth=None) -> dict:
    """Trace the residual against the solution norm and find the corner.

    The corner is where the curve stops trading a lot of penalty norm for a little residual and
    starts trading the other way, so it is a maximum of curvature in log-log coordinates. It needs
    no knowledge of the noise level, which is the whole reason it is used.
    """
    rows = []
    for lam in penalties:
        out = tikhonov(problem, data, float(lam), order=order)
        row = {"penalty": float(lam), "residual": out["residual"],
               "norm": out["penalty_norm"]}
        if truth is not None:
            row["error"] = float(np.linalg.norm(out["solution"] - np.asarray(truth, dtype=float))
                                 / np.linalg.norm(truth))
        rows.append(row)
    logs_r = np.log([max(r["residual"], 1e-300) for r in rows])
    logs_n = np.log([max(r["norm"], 1e-300) for r in rows])
    first_r = np.gradient(logs_r)
    first_n = np.gradient(logs_n)
    second_r = np.gradient(first_r)
    second_n = np.gradient(first_n)
    curvature = ((first_r * second_n - first_n * second_r)
                 / np.power(first_r ** 2 + first_n ** 2, 1.5))
    inside = slice(1, len(rows) - 1)
    corner = int(np.argmax(np.abs(curvature[inside]))) + 1
    out = {"rows": rows, "curvature": curvature, "corner": corner,
           "corner_penalty": rows[corner]["penalty"]}
    if truth is not None:
        best = int(np.argmin([r["error"] for r in rows]))
        out["oracle"] = best
        out["oracle_penalty"] = rows[best]["penalty"]
        out["corner_error"] = rows[corner]["error"]
        out["oracle_error"] = rows[best]["error"]
    return out


def conjugate_gradient(matvec, right_hand_side, iterations: int, truth=None) -> dict:
    """Plain conjugate gradient from lesson 26, recording the error at every iteration.

    Only the matrix-vector product is needed, so this runs on an operator that is never assembled.
    The reason it is here rather than only in Part 4 is the error history: on a noisy ill-posed
    problem the error falls and then rises, which makes the iteration count a regularization
    parameter.
    """
    b = np.asarray(right_hand_side, dtype=float)
    x = np.zeros_like(b)
    residual = b - matvec(x)
    direction = residual.copy()
    squared = float(residual @ residual)
    history = []
    for _ in range(int(iterations)):
        moved = matvec(direction)
        denominator = float(direction @ moved)
        if denominator <= 0.0 or squared == 0.0:
            break
        step = squared / denominator
        x = x + step * direction
        residual = residual - step * moved
        updated = float(residual @ residual)
        entry = {"residual": math.sqrt(updated)}
        if truth is not None:
            entry["error"] = float(np.linalg.norm(x - np.asarray(truth, dtype=float))
                                   / np.linalg.norm(truth))
        history.append(entry)
        direction = residual + (updated / squared) * direction
        squared = updated
    return {"x": x, "history": history, "iterations": len(history)}


# --------------------------------------------------------------------------- collocation


def chebyshev_basis(points, terms: int) -> dict:
    """A Chebyshev basis on ``[0, 1]``, with its first two derivatives, from lesson 47."""
    x = np.asarray(points, dtype=float)
    k = int(terms)
    mapped = 2.0 * x - 1.0
    values = np.zeros((x.size, k))
    first = np.zeros((x.size, k))
    second = np.zeros((x.size, k))
    for j in range(k):
        angle = np.arccos(np.clip(mapped, -1.0, 1.0))
        values[:, j] = np.cos(j * angle)
        with np.errstate(divide="ignore", invalid="ignore"):
            sine = np.sin(angle)
            first[:, j] = np.where(sine > 1e-12, j * np.sin(j * angle) / np.maximum(sine, 1e-300),
                                   float(j) ** 2)
            second[:, j] = np.where(
                sine > 1e-12,
                (-j * j * np.cos(j * angle) + j * np.cos(angle) * np.sin(j * angle)
                 / np.maximum(sine, 1e-300)) / np.maximum(sine ** 2, 1e-300),
                (float(j) ** 4 - float(j) ** 2) / 3.0)
    return {"values": values, "first": 2.0 * first, "second": 4.0 * second}


def tanh_basis(points, weights, offsets) -> dict:
    """A one hidden layer network's basis functions, with their first two derivatives.

    ``phi_k(x) = tanh(w_k x + b_k)``, so ``phi'' = -2 t (1 - t**2) w**2``. The derivatives are
    written out rather than differentiated numerically, and the lesson checks them against nested
    dual numbers from lesson 97.
    """
    x = np.asarray(points, dtype=float)[:, None]
    w = np.asarray(weights, dtype=float)[None, :]
    b = np.asarray(offsets, dtype=float)[None, :]
    t = np.tanh(w * x + b)
    return {"values": t, "first": (1.0 - t ** 2) * w,
            "second": -2.0 * t * (1.0 - t ** 2) * w ** 2}


def boundary_value_problem() -> dict:
    """``u'' = f`` on ``[0, 1]`` with ``u(0) = u(1) = 0`` and a solution written down in advance."""
    def exact(x):
        return np.sin(3.0 * math.pi * np.asarray(x, dtype=float)) * np.asarray(x, dtype=float) \
            * (1.0 - np.asarray(x, dtype=float))

    def source(x):
        t = np.asarray(x, dtype=float)
        k = 3.0 * math.pi
        return (-k * k * np.sin(k * t) * t * (1.0 - t)
                + 2.0 * k * np.cos(k * t) * (1.0 - 2.0 * t)
                - 2.0 * np.sin(k * t))
    return {"exact": exact, "source": source, "left": 0.0, "right": 0.0}


def collocate(basis: dict, problem: dict, points, weight: float = 1.0) -> dict:
    """Solve for the coefficients that make the residual vanish at the collocation points.

    This is lesson 76's method exactly. The rows of the system are the differential equation at each
    point plus the two boundary conditions, and it is linear in the coefficients whatever the basis
    is, as long as the basis itself is not being fitted.
    """
    x = np.asarray(points, dtype=float)
    rows = basis["second"]
    right = problem["source"](x)
    ends = np.array([0.0, 1.0])
    edge = basis["boundary"]
    matrix = np.vstack([rows, float(weight) * edge])
    target = np.concatenate([right, float(weight) * np.array([problem["left"],
                                                              problem["right"]])])
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return {"coefficients": coefficients, "points": x, "ends": ends,
            "residual": float(np.linalg.norm(matrix @ coefficients - target))}


def collocation_error(basis_values, coefficients, problem: dict, grid) -> float:
    """Relative error of a collocation solution against the written-down answer."""
    approximation = np.asarray(basis_values, dtype=float) @ np.asarray(coefficients, dtype=float)
    wanted = problem["exact"](grid)
    return float(np.linalg.norm(approximation - wanted) / np.linalg.norm(wanted))


# --------------------------------------------------------------------------- neural ODEs


def flow_problem(dimension: int = 2, seed: int = 42) -> dict:
    """``dz/dt = tanh(W z + c)``, which is one layer applied continuously in time.

    A neural ODE replaces a stack of layers with a single layer integrated over an interval. The
    solver is lesson 71's Runge-Kutta and the parameters are ``W`` and ``c``, so training it is
    lesson 67 with a gradient taken with respect to the right-hand side.
    """
    d = int(dimension)
    rng = np.random.default_rng(int(seed))
    weights = rng.standard_normal((d, d)) / math.sqrt(d)
    offsets = 0.3 * rng.standard_normal(d)
    target = rng.standard_normal(d)
    start = rng.standard_normal(d)

    def rate(z, w, c):
        return np.tanh(w @ z + c)

    def jacobian_state(z, w, c):
        derivative = 1.0 - np.tanh(w @ z + c) ** 2
        return derivative[:, None] * w

    def jacobian_weights(z, w, c):
        derivative = 1.0 - np.tanh(w @ z + c) ** 2
        block = np.zeros((z.size, w.size + c.size))
        for i in range(z.size):
            for j in range(z.size):
                block[i, i * z.size + j] = derivative[i] * z[j]
            block[i, w.size + i] = derivative[i]
        return block

    return {"rate": rate, "jacobian_state": jacobian_state, "jacobian_weights": jacobian_weights,
            "weights": weights, "offsets": offsets, "start": start, "target": target,
            "dimension": d,
            "parameters": np.concatenate([weights.ravel(), offsets])}


def integrate(problem: dict, parameters, steps: int, horizon: float = 1.0) -> dict:
    """Classical Runge-Kutta on the flow, keeping every state so the backward pass can use them."""
    d = problem["dimension"]
    theta = np.asarray(parameters, dtype=float)
    w = theta[:d * d].reshape(d, d)
    c = theta[d * d:]
    h = float(horizon) / int(steps)
    z = np.array(problem["start"], dtype=float)
    path = [z.copy()]
    for _ in range(int(steps)):
        k1 = problem["rate"](z, w, c)
        k2 = problem["rate"](z + 0.5 * h * k1, w, c)
        k3 = problem["rate"](z + 0.5 * h * k2, w, c)
        k4 = problem["rate"](z + h * k3, w, c)
        z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        path.append(z.copy())
    return {"z": z, "path": np.array(path), "steps": int(steps), "step": h}


def loss(problem: dict, parameters, steps: int, horizon: float = 1.0) -> float:
    """Half the squared distance from the endpoint to the target."""
    z = integrate(problem, parameters, steps, horizon)["z"]
    difference = z - problem["target"]
    return 0.5 * float(difference @ difference)


def adjoint_gradient(problem: dict, parameters, steps: int, horizon: float = 1.0) -> dict:
    """Optimize then discretize: integrate the adjoint equation backwards.

    The adjoint ``a = dL/dz`` satisfies ``da/dt = -a' (df/dz)`` backwards in time, and the parameter
    gradient is ``-integral a' (df/dtheta) dt``. Memory is one state and one adjoint, whatever the
    step count, which is the property the method exists for and it is lesson 97's implicit route.
    """
    d = problem["dimension"]
    theta = np.asarray(parameters, dtype=float)
    w = theta[:d * d].reshape(d, d)
    c = theta[d * d:]
    forward = integrate(problem, theta, steps, horizon)
    h = forward["step"]
    a = forward["z"] - problem["target"]
    grad = np.zeros(theta.size)
    z = forward["z"].copy()
    for index in range(int(steps) - 1, -1, -1):
        z = forward["path"][index]
        grad = grad + h * (a @ problem["jacobian_weights"](z, w, c))
        a = a + h * (a @ problem["jacobian_state"](z, w, c))
    return {"gradient": grad, "memory": 2 * d + theta.size, "route": "adjoint"}


def unrolled_gradient(problem: dict, parameters, steps: int, horizon: float = 1.0,
                      step: float = 1e-6) -> dict:
    """Discretize then optimize: differentiate the solver's own arithmetic.

    Here that is done with central differences, which is enough to compare the two gradients without
    building a second tape. The point of the comparison is not the last digit; it is that the two
    routes are different functions of the step size.
    """
    theta = np.asarray(parameters, dtype=float)
    grad = np.zeros(theta.size)
    for i in range(theta.size):
        moved = theta.copy()
        moved[i] += float(step)
        ahead = loss(problem, moved, steps, horizon)
        moved[i] -= 2.0 * float(step)
        behind = loss(problem, moved, steps, horizon)
        grad[i] = (ahead - behind) / (2.0 * float(step))
    return {"gradient": grad, "memory": theta.size * (int(steps) + 1),
            "route": "through the solver"}


# --------------------------------------------------------------------------- many dimensions


def poisson(dimension: int) -> dict:
    """``Laplacian u = f`` on the unit cube with ``u = 0`` on the boundary, solution written down.

    The manufactured solution is ``prod_i sin(pi x_i)``, which vanishes on every face and has
    Laplacian ``-d pi**2`` times itself. Nothing about the solution is hidden from any method: the
    bases below all have to represent a product of sines without being told that is what it is.
    """
    d = int(dimension)
    if d < 1:
        raise ValueError(f"need at least one dimension, got {d}")

    def exact(points):
        x = np.atleast_2d(np.asarray(points, dtype=float))
        return np.prod(np.sin(np.pi * x), axis=1)

    def source(points):
        return -d * math.pi ** 2 * exact(points)

    return {"exact": exact, "source": source, "dimension": d}


def tensor_indices(dimension: int, degree: int) -> np.ndarray:
    """Every multi-index with all components below ``degree``: the full tensor product, ``m**d``."""
    d, m = int(dimension), int(degree)
    if d < 1 or m < 1:
        raise ValueError(f"need a positive dimension and degree, got {d} and {m}")
    grids = np.meshgrid(*([np.arange(m)] * d), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def sparse_indices(dimension: int, level: int) -> np.ndarray:
    """Every multi-index whose components sum to at most ``level``: a Smolyak set.

    Its size is ``C(level + d, d)``, which grows polynomially in the dimension where the tensor
    product grows exponentially. That single fact is what postpones the curse, and the measurement
    below finds out by how much.
    """
    d, k = int(dimension), int(level)
    if d < 1 or k < 0:
        raise ValueError(f"need a positive dimension and a non-negative level, got {d} and {k}")
    rows = [[]]
    for axis in range(d):
        grown = []
        for prefix in rows:
            remaining = k - sum(prefix)
            for value in range(remaining + 1):
                grown.append(prefix + [value])
        rows = grown
    return np.array(rows, dtype=int)


def chebyshev_field(points, indices, degree: int | None = None) -> dict:
    """Values and Laplacians of a tensor product Chebyshev basis at a set of points.

    The basis function for a multi-index is ``prod_i T_{a_i}(2 x_i - 1)``, so its Laplacian is the
    sum over axes of the same product with one factor replaced by that factor's second derivative.
    Both come from the one dimensional table in :func:`chebyshev_basis`, one axis at a time. The
    table is made exactly wide enough for the indices given, so a caller cannot silently ask for a
    degree the table does not hold.
    """
    x = np.atleast_2d(np.asarray(points, dtype=float))
    alpha = np.asarray(indices, dtype=int)
    n, d = x.shape
    if alpha.shape[1] != d:
        raise ValueError(f"indices have width {alpha.shape[1]} for {d} dimensions")
    width = int(alpha.max()) + 1 if degree is None else int(degree)
    if width <= int(alpha.max()):
        raise ValueError(f"degree {width} cannot hold an index of {int(alpha.max())}")
    values = np.empty((d, n, width))
    seconds = np.empty((d, n, width))
    for axis in range(d):
        table = chebyshev_basis(x[:, axis], width)
        values[axis] = table["values"]
        seconds[axis] = table["second"]
    value = np.ones((n, alpha.shape[0]))
    for axis in range(d):
        value = value * values[axis][:, alpha[:, axis]]
    laplacian = np.zeros((n, alpha.shape[0]))
    for wrt in range(d):
        term = np.ones((n, alpha.shape[0]))
        for axis in range(d):
            table = seconds if axis == wrt else values
            term = term * table[axis][:, alpha[:, axis]]
        laplacian = laplacian + term
    return {"value": value, "laplacian": laplacian, "size": int(alpha.shape[0])}


def tanh_field(points, weights, offsets) -> dict:
    """Values and Laplacians of ``tanh(w . x + b)`` at a set of points, one column per unit.

    With ``t = tanh(w . x + b)``, the second derivative along axis ``j`` is
    ``w_j**2 * (-2 t (1 - t**2))``, so the Laplacian is ``|w|**2`` times that one scalar. The
    dimension enters only through ``|w|**2``, which is the whole reason a network's cost does not
    grow with it.
    """
    x = np.atleast_2d(np.asarray(points, dtype=float))
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    b = np.asarray(offsets, dtype=float)
    t = np.tanh(x @ w.T + b[None, :])
    sech = 1.0 - t ** 2
    norms = np.sum(w ** 2, axis=1)
    shape = -2.0 * t * sech
    return {"value": t, "laplacian": norms[None, :] * shape, "t": t, "sech": sech,
            "norms": norms, "shape": shape,
            "shape_slope": -2.0 * sech * (sech - 2.0 * t ** 2)}


def cube_points(dimension: int, interior: int, boundary: int, seed: int = 42) -> dict:
    """Low discrepancy points inside the unit cube, and points on its faces.

    The interior points carry the equation and the face points carry the boundary condition. Halton
    is used rather than a grid because a grid has ``m**d`` points and that is the thing being
    measured.
    """
    d, n, m = int(dimension), int(interior), int(boundary)
    if min(d, n, m) < 1:
        raise ValueError(f"need positive counts, got {d}, {n} and {m}")
    inside = _halton(n, d, skip=1)
    faces = _halton(m, d, skip=n + 1)
    rng = np.random.default_rng(int(seed))
    axis = rng.integers(0, d, size=m)
    side = rng.integers(0, 2, size=m).astype(float)
    faces[np.arange(m), axis] = side
    return {"interior": inside, "boundary": faces}


def _halton(count: int, dimension: int, skip: int = 0) -> np.ndarray:
    """A Halton point set, written here so this module does not depend on lesson 92's."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    d = int(dimension)
    if d > len(primes):
        raise ValueError(f"only {len(primes)} bases are tabulated, asked for {d}")
    out = np.empty((int(count), d))
    for axis in range(d):
        base = primes[axis]
        for row, index in enumerate(range(int(skip) + 1, int(skip) + int(count) + 1)):
            value, denominator, k = 0.0, 1.0, index
            while k > 0:
                denominator *= base
                value += (k % base) / denominator
                k //= base
            out[row, axis] = value
    return out


def linear_collocation(problem: dict, points: dict, field_inside: dict, field_edge: dict,
                       weight: float = 10.0) -> dict:
    """Solve for the coefficients of a fixed linear basis, which is one least squares problem."""
    rows = np.vstack([field_inside["laplacian"], float(weight) * field_edge["value"]])
    target = np.concatenate([problem["source"](points["interior"]),
                             np.zeros(field_edge["value"].shape[0])])
    coefficients, *_ = np.linalg.lstsq(rows, target, rcond=None)
    return {"coefficients": coefficients, "size": rows.shape[1],
            "residual": float(np.linalg.norm(rows @ coefficients - target))}


def train_tanh_collocation(problem: dict, points: dict, units: int, steps: int,
                           rate: float = 1e-2, weight: float = 10.0, seed: int = 42) -> dict:
    """Fit a one hidden layer network to the PDE residual, with gradients written out.

    The derivatives are analytic rather than differenced, because a finite difference gradient over
    ``units * (d + 2)`` parameters would cost that many extra evaluations per step and lesson 97
    already measured what that trade costs.
    """
    d = problem["dimension"]
    k = int(units)
    inside, edge = points["interior"], points["boundary"]
    source = problem["source"](inside)
    total = inside.shape[0] + edge.shape[0]
    rng = np.random.default_rng(int(seed))
    w = rng.standard_normal((k, d)) * (2.0 / math.sqrt(d))
    b = rng.standard_normal(k) * 0.5
    c = rng.standard_normal(k) * 0.01
    first_w, second_w = np.zeros_like(w), np.zeros_like(w)
    first_b, second_b = np.zeros_like(b), np.zeros_like(b)
    first_c, second_c = np.zeros_like(c), np.zeros_like(c)
    losses = []
    for step in range(1, int(steps) + 1):
        home = tanh_field(inside, w, b)
        wall = tanh_field(edge, w, b)
        residual = home["laplacian"] @ c - source
        edge_residual = float(weight) * (wall["value"] @ c)
        losses.append(0.5 * float(residual @ residual + edge_residual @ edge_residual) / total)

        grad_c = (home["laplacian"].T @ residual
                  + float(weight) * wall["value"].T @ edge_residual) / total

        inner = residual[:, None] * c[None, :]
        grad_w = 2.0 * w * np.sum(inner * home["shape"], axis=0)[:, None]
        grad_w = grad_w + home["norms"][:, None] * ((inner * home["shape_slope"]).T @ inside)
        grad_b = home["norms"] * np.sum(inner * home["shape_slope"], axis=0)

        outer = edge_residual[:, None] * c[None, :] * float(weight) * wall["sech"]
        grad_w = (grad_w + outer.T @ edge) / total
        grad_b = (grad_b + np.sum(outer, axis=0)) / total

        for value, gradient, first, second in ((w, grad_w, first_w, second_w),
                                               (b, grad_b, first_b, second_b),
                                               (c, grad_c, first_c, second_c)):
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient * gradient
            value -= float(rate) * (first / (1.0 - 0.9 ** step)) / (
                np.sqrt(second / (1.0 - 0.999 ** step)) + 1e-8)
    return {"weights": w, "offsets": b, "coefficients": c, "losses": losses,
            "parameters": k * (d + 2), "steps": int(steps)}


def field_error(problem: dict, predict, count: int = 4000, seed: int = 7) -> float:
    """Relative error against the written down solution, on points no method has seen."""
    rng = np.random.default_rng(int(seed))
    test = rng.random((int(count), problem["dimension"]))
    wanted = problem["exact"](test)
    got = np.asarray(predict(test), dtype=float)
    return float(np.linalg.norm(got - wanted) / np.linalg.norm(wanted))


# --------------------------------------------------------------------------- measurements


def refining_the_grid_makes_an_inverse_problem_worse(sizes=(32, 64, 128, 256),
                                                     width: float = 0.05) -> dict:
    """Condition numbers of a blur and of a second difference operator, at the same resolutions.

    A forward problem gets better conditioned relative to what it is asked for as the grid refines.
    An inverse problem does not: more grid points means more singular values below the noise, so the
    condition number grows without bound and the extra resolution is unusable.
    """
    rows = []
    for n in sizes:
        forward = blur(int(n), width=float(width))
        s = np.linalg.svd(forward["matrix"], compute_uv=False)
        derivative = np.linalg.svd(second_difference(int(n)), compute_uv=False)
        rows.append({
            "size": int(n),
            "blur_condition": float(s[0] / s[-1]),
            "difference_condition": float(derivative[0] / derivative[-1]),
            "decay_per_index": float(math.exp(np.polyfit(np.arange(min(n, 40)),
                                                         np.log(s[:min(n, 40)]), 1)[0])),
            "usable_at_double": int(np.sum(s > s[0] * np.finfo(float).eps)),
        })
    difference_growth = float(np.polyfit(np.log([r["size"] for r in rows]),
                                         np.log([r["difference_condition"] for r in rows]), 1)[0])
    ceiling = 1.0 / float(np.finfo(float).eps)
    saturates_at = next((r["size"] for r in rows if r["blur_condition"] > ceiling), None)
    return {
        "rows": rows, "difference_growth": difference_growth,
        "double_precision_ceiling": ceiling, "saturates_at": saturates_at,
        "usable_gain_from_the_last_doubling": (rows[-1]["usable_at_double"]
                                               - rows[-2]["usable_at_double"]),
        "grid_gain_from_the_last_doubling": rows[-1]["size"] - rows[-2]["size"],
        "the_difference_operator_grows_like_n_squared": abs(difference_growth - 2.0) < 0.2,
        "the_blur_exhausts_double_precision": saturates_at is not None,
        "the_extra_grid_points_are_wasted": (rows[-1]["usable_at_double"]
                                             - rows[-2]["usable_at_double"]) < 0.1
                                            * (rows[-1]["size"] - rows[-2]["size"]),
        "the_blur_is_unusable_at_the_finest": rows[-1]["blur_condition"] > 1e15,
        "the_spectrum_decays_geometrically": rows[-1]["decay_per_index"] < 0.9,
        "note": "an ill-posed problem is one where refining the discretization adds singular "
                "values you cannot use, so resolution and accuracy stop being the same thing",
    }


def the_picard_condition_locates_the_usable_components(levels=(0.0, 1e-8, 1e-6, 1e-4, 1e-2),
                                                       size: int = 128, seed: int = 42) -> dict:
    """Plot ``|u_i' b|`` against ``sigma_i`` at several noise levels and find where they cross.

    With exact data the coefficients keep falling and the pseudoinverse is fine. With noise they
    flatten at the noise level, and every component past the crossing divides a constant by a tiny
    singular value. The crossing index is the number of pieces of information the data contains.
    """
    problem = blur(int(size))
    truth = test_signal(problem["grid"])
    rows = []
    for level in levels:
        made = noisy_data(problem, truth, level=float(level), seed=seed)
        out = picard(problem, made["data"])
        plain = np.linalg.lstsq(problem["matrix"], made["data"], rcond=None)[0]
        rows.append({
            "noise": float(level), "usable": out["usable"], "total": out["total"],
            "noise_floor": out["noise_floor"],
            "pseudoinverse_error": float(np.linalg.norm(plain - truth) / np.linalg.norm(truth)),
        })
    reference = picard(problem, noisy_data(problem, truth, level=1e-4, seed=seed)["data"])
    return {
        "rows": rows, "singular_values": reference["singular_values"],
        "coefficients": reference["coefficients"],
        "usable_falls_with_noise": all(rows[i]["usable"] >= rows[i + 1]["usable"]
                                       for i in range(len(rows) - 1)),
        "the_pseudoinverse_fails": rows[-1]["pseudoinverse_error"] > 1.0,
        "exact_data_is_far_better": (rows[0]["pseudoinverse_error"]
                                     < 1e-3 * rows[1]["pseudoinverse_error"]),
        "even_exact_data_has_a_floor": rows[0]["noise_floor"] > 0.0,
        "usable_at_the_middle_level": rows[3]["usable"],
        "note": "the number of usable components is set by the noise, not by the grid, so a finer "
                "grid buys nothing at all",
    }


def tikhonov_is_the_ridge_of_lesson_94(penalties=None, size: int = 96, level: float = 1e-4,
                                       seed: int = 42) -> dict:
    """Check the filter factors against ``sigma**2/(sigma**2+lam)`` and compare the two selectors.

    The L-curve corner and the oracle penalty are different quantities: one uses the data alone and
    the other uses the answer. How far apart they are is the practical question, and it is a number
    rather than an argument.
    """
    grid = np.geomspace(1e-12, 1e2, 45) if penalties is None else np.asarray(penalties, dtype=float)
    problem = blur(int(size))
    truth = test_signal(problem["grid"])
    made = noisy_data(problem, truth, level=float(level), seed=seed)
    left, s, right = np.linalg.svd(problem["matrix"], full_matrices=False)
    coefficients = left.T @ made["data"]
    worst = 0.0
    for lam in (1e-8, 1e-4, 1e-1):
        predicted = right.T @ ((s ** 2 / (s ** 2 + lam)) * coefficients / s)
        worst = max(worst, float(np.max(np.abs(predicted
                                               - tikhonov(problem, made["data"], lam)["solution"]))))
    curve = l_curve(problem, made["data"], grid, truth=truth)
    return {
        "rows": curve["rows"], "curvature": curve["curvature"],
        "filter_gap": worst,
        "corner_penalty": curve["corner_penalty"], "oracle_penalty": curve["oracle_penalty"],
        "corner_error": curve["corner_error"], "oracle_error": curve["oracle_error"],
        "penalty_ratio": max(curve["corner_penalty"] / curve["oracle_penalty"],
                             curve["oracle_penalty"] / curve["corner_penalty"]),
        "error_ratio": curve["corner_error"] / curve["oracle_error"],
        "it_is_the_same_filter": worst < 1e-8,
        "the_corner_is_close_to_the_oracle": (curve["corner_error"]
                                              < 3.0 * curve["oracle_error"]),
        "regularization_helps": (curve["oracle_error"]
                                 < 0.5 * max(r["error"] for r in curve["rows"])),
        "note": "Tikhonov regularization and ridge regression are the same computation, and the "
                "L-curve is a way of choosing the penalty without knowing the answer",
    }


def a_derivative_penalty_knows_something_the_identity_does_not(
        penalties=None, kinds=("smooth", "mixed", "box"), size: int = 96, level: float = 1e-4,
        seed: int = 42) -> dict:
    """Compare a penalty on the size of the solution against one on its second derivative.

    Both are Tikhonov and neither is more correct. They encode different beliefs, that the answer is
    small or that it is smooth, so the honest test runs them on answers that have and do not have
    the property. A smoothness penalty that won on every signal would not be a prior at all.
    """
    grid = np.geomspace(1e-12, 1e4, 45) if penalties is None else np.asarray(penalties, dtype=float)
    problem = blur(int(size))
    rows = []
    for kind in kinds:
        truth = test_signal(problem["grid"], kind)
        made = noisy_data(problem, truth, level=float(level), seed=seed)
        best = {}
        for order in (0, 2):
            errors = [float(np.linalg.norm(tikhonov(problem, made["data"], float(lam),
                                                    order=order)["solution"] - truth)
                            / np.linalg.norm(truth)) for lam in grid]
            index = int(np.argmin(errors))
            best[order] = {"penalty": float(grid[index]), "error": errors[index]}
        keeps = [max(int(size) // 16, 1), max(int(size) // 8, 1), max(int(size) // 4, 1)]
        truncated = min(float(np.linalg.norm(truncated_solution(problem, made["data"],
                                                                keep)["solution"] - truth)
                              / np.linalg.norm(truth)) for keep in keeps)
        rows.append({
            "signal": kind,
            "identity_penalty": best[0]["penalty"], "identity_error": best[0]["error"],
            "derivative_penalty": best[2]["penalty"], "derivative_error": best[2]["error"],
            "gain": best[0]["error"] / best[2]["error"],
            "truncated_error": truncated,
        })
    smooth = next(r for r in rows if r["signal"] == "smooth")
    box = next(r for r in rows if r["signal"] == "box")
    right = np.linalg.svd(problem["matrix"], full_matrices=False)[2]
    rough = second_difference(problem["matrix"].shape[1])
    roughness = np.array([float(np.linalg.norm(rough @ right[i])) for i in range(right.shape[0])])
    reliable = max(3 * roughness.size // 4, 2)
    shown = [int(i) for i in np.linspace(0, roughness.size - 1, 8).astype(int)]
    inside = [i for i in shown if i < reliable]
    ordered = all(roughness[inside[i]] <= roughness[inside[i + 1]] * 1.05
                  for i in range(len(inside) - 1))
    peak = int(np.argmax(roughness))
    return {
        "rows": rows,
        "smooth_gain": smooth["gain"], "box_gain": box["gain"],
        "largest_gain": max(r["gain"] for r in rows),
        "roughness": roughness, "roughness_shown": [(i, float(roughness[i])) for i in shown],
        "roughness_range": float(roughness.max() / roughness[0]),
        "roughness_peaks_at": peak, "reliable_up_to": reliable,
        "the_derivative_penalty_wins_a_little_on_a_smooth_answer": 1.0 < smooth["gain"] < 1.2,
        "it_wins_by_less_on_a_discontinuous_one": box["gain"] < smooth["gain"],
        "the_two_penalties_nearly_agree": max(r["gain"] for r in rows) < 1.1,
        "the_operator_sorts_its_own_directions_by_smoothness": ordered,
        "the_ordering_stops_where_the_singular_values_hit_rounding": peak < roughness.size - 1,
        "truncation_is_in_the_same_range": all(r["truncated_error"] < 2.0 * r["identity_error"]
                                               for r in rows),
        "note": "the two penalties nearly agree here because the blur's own singular vectors are "
                "already ordered by frequency, so shrinking the small ones is already a smoothness "
                "prior",
    }


def stopping_early_is_regularization(levels=(1e-5, 1e-4, 1e-3, 1e-2), iterations: int = 400,
                                     size: int = 96, seed: int = 42) -> dict:
    """Run conjugate gradient on the noisy normal equations and watch the error fall, then rise.

    Krylov methods pick up the large singular directions first, so the early iterates are smooth
    approximations and the later ones fit noise. The error curve therefore has a minimum, the
    iteration count is a regularization parameter exactly like ``lam``, and the minimum moves
    earlier as the noise grows. It is called semiconvergence, and it is why an ill-posed problem
    must never be solved to convergence.
    """
    problem = blur(int(size))
    truth = test_signal(problem["grid"])
    a = problem["matrix"]
    grid = np.geomspace(1e-12, 1e2, 40)
    rows = []
    histories = {}
    for level in levels:
        made = noisy_data(problem, truth, level=float(level), seed=seed)

        def matvec(v, matrix=a):
            return matrix.T @ (matrix @ v)

        run = conjugate_gradient(matvec, a.T @ made["data"], int(iterations), truth=truth)
        errors = [entry["error"] for entry in run["history"]]
        best = int(np.argmin(errors))
        penalty_errors = [float(np.linalg.norm(tikhonov(problem, made["data"],
                                                        float(lam))["solution"] - truth)
                                / np.linalg.norm(truth)) for lam in grid]
        histories[float(level)] = errors
        rows.append({
            "noise": float(level), "best_iteration": best + 1, "best_error": errors[best],
            "final_error": errors[-1], "rise": errors[-1] / errors[best],
            "best_penalty_error": min(penalty_errors),
            "krylov_over_tikhonov": errors[best] / min(penalty_errors),
        })
    return {
        "rows": rows, "histories": histories, "iterations": int(iterations),
        "worst_rise": max(r["rise"] for r in rows),
        "it_falls_then_rises": all(r["rise"] > 10.0 for r in rows),
        "the_minimum_moves_earlier_with_noise": all(
            rows[i]["best_iteration"] >= rows[i + 1]["best_iteration"]
            for i in range(len(rows) - 1)),
        "it_is_competitive_with_tikhonov": max(r["krylov_over_tikhonov"] for r in rows) < 1.5,
        "note": "the iteration count is a regularization parameter, so an ill-posed problem is one "
                "you have to stop solving before the solver has finished",
    }


def collocation_beats_the_trained_basis(terms=(8, 12, 16, 20, 24), units: int = 12,
                                        points: int = 40, steps: int = 3000, rate: float = 1e-2,
                                        seed: int = 42) -> dict:
    """Solve one boundary value problem three ways and compare accuracy against cost.

    The three are: collocation with a Chebyshev basis, which is one linear solve; the same
    collocation with a tanh basis whose nonlinear parameters are left at their random values, also
    one linear solve; and the tanh basis fitted end to end by Adam, which is what a physics-informed
    network does. The middle one is the control that separates what the nonlinear basis contributes
    from what training it contributes.
    """
    problem = boundary_value_problem()
    k = int(units)
    inside = np.linspace(0.0, 1.0, int(points) + 2)[1:-1]
    fine = np.linspace(0.0, 1.0, 401)
    ends = np.array([0.0, 1.0])
    source = problem["source"](inside)

    linear = []
    for count in terms:
        basis = chebyshev_basis(inside, int(count))
        basis["boundary"] = chebyshev_basis(ends, int(count))["values"]
        fitted = collocate(basis, problem, inside, weight=10.0)
        linear.append({
            "terms": int(count), "solves": 1,
            "error": collocation_error(chebyshev_basis(fine, int(count))["values"],
                                       fitted["coefficients"], problem, fine),
        })

    rng = np.random.default_rng(int(seed))
    weights = rng.uniform(-6.0, 6.0, k)
    offsets = rng.uniform(-3.0, 3.0, k)
    frozen = tanh_basis(inside, weights, offsets)
    frozen["boundary"] = tanh_basis(ends, weights, offsets)["values"]
    solved = collocate(frozen, problem, inside, weight=10.0)
    frozen_error = collocation_error(tanh_basis(fine, weights, offsets)["values"],
                                     solved["coefficients"], problem, fine)

    def objective(theta):
        c, w, b = theta[:k], theta[k:2 * k], theta[2 * k:]
        basis = tanh_basis(inside, w, b)
        edge = tanh_basis(ends, w, b)
        r = np.concatenate([basis["second"] @ c - source,
                            10.0 * (edge["values"] @ c
                                    - np.array([problem["left"], problem["right"]]))])
        return 0.5 * float(r @ r) / r.size

    def gradient(theta, step=1e-6):
        base = objective(theta)
        out = np.zeros(theta.size)
        for i in range(theta.size):
            moved = theta.copy()
            moved[i] += step
            out[i] = (objective(moved) - base) / step
        return out

    theta = np.concatenate([rng.standard_normal(k) * 0.1, weights, offsets])
    first = np.zeros(theta.size)
    second = np.zeros(theta.size)
    losses = [objective(theta)]
    for step in range(1, int(steps) + 1):
        g = gradient(theta)
        first = 0.9 * first + 0.1 * g
        second = 0.999 * second + 0.001 * g * g
        theta = theta - float(rate) * (first / (1.0 - 0.9 ** step)) / (
            np.sqrt(second / (1.0 - 0.999 ** step)) + 1e-8)
        losses.append(objective(theta))
    trained_error = collocation_error(tanh_basis(fine, theta[k:2 * k], theta[2 * k:])["values"],
                                      theta[:k], problem, fine)

    matched = next(r for r in linear if r["terms"] == k)
    return {
        "linear": linear, "units": k, "points": int(points), "steps": int(steps),
        "losses": losses, "loss_fell": losses[0] / losses[-1],
        "matched_chebyshev_error": matched["error"],
        "best_chebyshev_error": min(r["error"] for r in linear),
        "frozen_tanh_error": frozen_error, "trained_tanh_error": trained_error,
        "training_gain": frozen_error / trained_error,
        "chebyshev_over_trained": trained_error / matched["error"],
        "evaluations": int(steps) * (3 * k + 1),
        "spectral_convergence": linear[0]["error"] / min(r["error"] for r in linear),
        "training_helps": frozen_error > trained_error,
        "the_linear_basis_matches_it_in_one_solve": matched["error"] < 2.0 * trained_error,
        "a_bigger_linear_basis_beats_it_outright": (min(r["error"] for r in linear)
                                                    < 0.01 * trained_error),
        "the_chebyshev_error_falls_spectrally": (linear[0]["error"]
                                                 > 100.0 * min(r["error"] for r in linear)),
        "note": "a physics-informed network is collocation with a nonlinear basis, and on a problem "
                "a linear basis can represent, training buys a factor over a random basis and "
                "loses to a good one",
    }


def the_adjoint_gives_the_same_gradient_at_constant_memory(
        counts=(16, 32, 64, 128, 256, 512), dimension: int = 3, seed: int = 42) -> dict:
    """Compare the adjoint gradient against differentiating through the solver, at several steps.

    These are not two implementations of one thing. Differentiating the solver gives the exact
    gradient of the discrete problem; the adjoint gives a discretization of the exact gradient of
    the continuous problem. They agree only as the step goes to zero, and the gap is the price of
    constant memory.
    """
    problem = flow_problem(int(dimension), seed=seed)
    theta = problem["parameters"]
    rows = []
    for steps in counts:
        adjoint = adjoint_gradient(problem, theta, int(steps))
        unrolled = unrolled_gradient(problem, theta, int(steps))
        scale = float(np.linalg.norm(unrolled["gradient"]))
        rows.append({
            "steps": int(steps), "step": 1.0 / int(steps),
            "gap": float(np.linalg.norm(adjoint["gradient"] - unrolled["gradient"]) / scale),
            "adjoint_memory": adjoint["memory"], "unrolled_memory": unrolled["memory"],
            "memory_ratio": unrolled["memory"] / adjoint["memory"],
        })
    order = float(np.polyfit(np.log([r["step"] for r in rows]),
                             np.log([r["gap"] for r in rows]), 1)[0])
    return {
        "rows": rows, "order": order,
        "coarse_gap": rows[0]["gap"], "fine_gap": rows[-1]["gap"],
        "largest_memory_ratio": max(r["memory_ratio"] for r in rows),
        "they_converge": rows[-1]["gap"] < 0.1 * rows[0]["gap"],
        "the_gap_falls_with_the_step": order > 0.5,
        "adjoint_memory_is_constant": len({r["adjoint_memory"] for r in rows}) == 1,
        "unrolled_memory_grows": rows[-1]["unrolled_memory"] > 10 * rows[0]["unrolled_memory"],
        "note": "discretize then optimize and optimize then discretize are different gradients of "
                "different problems, and which one you want depends on what you are optimizing",
    }


def where_a_trained_basis_starts_to_win(dimensions=(1, 2, 3, 4, 5, 6), budgets=(200, 400),
                                        steps: int = 3000, max_degree: int = 16,
                                        usable: float = 0.1, seed: int = 9) -> dict:
    """Solve the same Poisson problem three ways at a matched parameter budget, as the dimension grows.

    Section 8 compared a linear basis against a trained one in a single dimension and found the
    linear basis winning by 82000. That is the case the method is not for, so this is the comparison
    that could reverse it.

    The three are a **sparse grid** Chebyshev basis, whose size is ``C(level+d, d)`` rather than the
    tensor product's ``m**d``; the same number of parameters spent on a **trained** tanh network; and
    the same network with its hidden layer **frozen** at its random values, which is the control that
    separates what the nonlinear basis contributes from what training it contributes.

    Every method gets the same parameter count, the same collocation points and the same problem.
    The per-axis degree is capped because a Chebyshev second derivative grows like ``n**4``, so past
    about degree 16 the collocation matrix is conditioned out of double precision and the failure
    measured would be lesson 33's rather than the dimension's.
    """
    rows = []
    for budget in budgets:
        for d in dimensions:
            problem = poisson(int(d))
            affordable = [k for k in range(1, 300)
                          if math.comb(k + int(d), int(d)) <= int(budget)]
            level = min(int(max_degree), max(affordable) if affordable else 1)
            indices = sparse_indices(int(d), level)
            units = max(1, int(budget) // (int(d) + 2))
            points = cube_points(int(d), 2 * int(budget), max(int(budget) // 2, 8), seed=seed - 4)

            inside = chebyshev_field(points["interior"], indices)
            edge = chebyshev_field(points["boundary"], indices)
            grid = linear_collocation(problem, points, inside, edge)
            grid_error = field_error(
                problem, lambda t, i=indices, c=grid["coefficients"]:
                chebyshev_field(t, i)["value"] @ c)

            trained = train_tanh_collocation(problem, points, units, int(steps), seed=seed)
            trained_error = field_error(
                problem, lambda t, n=trained:
                tanh_field(t, n["weights"], n["offsets"])["value"] @ n["coefficients"])

            rng = np.random.default_rng(int(seed))
            w = rng.standard_normal((units, int(d))) * (2.0 / math.sqrt(int(d)))
            b = rng.standard_normal(units) * 0.5
            frozen = linear_collocation(problem, points,
                                        tanh_field(points["interior"], w, b),
                                        tanh_field(points["boundary"], w, b))
            frozen_error = field_error(
                problem, lambda t, w=w, b=b, c=frozen["coefficients"]:
                tanh_field(t, w, b)["value"] @ c)

            rows.append({
                "budget": int(budget), "dimension": int(d), "level": level,
                "grid_size": int(indices.shape[0]), "grid_error": grid_error,
                "units": units, "parameters": units * (int(d) + 2),
                "trained_error": trained_error, "frozen_error": frozen_error,
                "ratio": grid_error / trained_error,
                "training_gain": frozen_error / trained_error,
            })

    crossovers = {}
    slopes = {}
    for budget in budgets:
        here = [r for r in rows if r["budget"] == budget]
        crossovers[budget] = next((r["dimension"] for r in here if r["ratio"] > 1.0), None)
        axis = np.array([r["dimension"] for r in here], dtype=float)
        slopes[budget] = {
            "grid": float(np.polyfit(axis, np.log10([r["grid_error"] for r in here]), 1)[0]),
            "trained": float(np.polyfit(axis, np.log10([r["trained_error"] for r in here]), 1)[0]),
        }
    found = [v for v in crossovers.values() if v is not None]
    last = {}
    for budget in budgets:
        here = [r for r in rows if r["budget"] == budget]
        last[budget] = {
            "grid": max((r["dimension"] for r in here if r["grid_error"] < usable), default=0),
            "trained": max((r["dimension"] for r in here if r["trained_error"] < usable),
                           default=0),
        }
    top = max(budgets)
    beyond = [r for r in rows if r["budget"] == top and crossovers[top] is not None
              and r["dimension"] >= crossovers[top]]
    return {
        "rows": rows, "crossovers": crossovers, "slopes": slopes, "usable_to": last,
        "usable_threshold": float(usable), "steps": int(steps), "max_degree": int(max_degree),
        "largest_classical_win": max(1.0 / r["ratio"] for r in rows if r["ratio"] < 1.0),
        "largest_trained_win": max((r["ratio"] for r in rows if r["ratio"] > 1.0), default=0.0),
        "worst_error_past_the_crossover": max((max(r["grid_error"], r["trained_error"])
                                               for r in beyond), default=0.0),
        "the_classical_basis_wins_in_low_dimensions": all(
            r["ratio"] < 1.0 for r in rows if r["dimension"] <= 3),
        "the_trained_basis_wins_somewhere": bool(found),
        "the_crossover_agrees_across_budgets": len(set(found)) == 1 and len(found) == len(budgets),
        "the_trained_error_is_flatter": all(slopes[b]["trained"] < slopes[b]["grid"]
                                            for b in budgets),
        "neither_is_usable_past_the_crossover": max(
            (max(r["grid_error"], r["trained_error"]) for r in beyond), default=0.0) > usable,
        "note": "the crossover is where the sparse grid's level collapses, not where the network "
                "improves, and it happens where neither method is accurate any more",
    }
