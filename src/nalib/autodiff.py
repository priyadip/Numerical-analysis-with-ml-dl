"""Automatic differentiation, both modes, written out.

Where the work is
-----------------
There are three ways to get a derivative from a program, and they are genuinely different things
rather than three approximations to one thing.

**Symbolic differentiation** transforms an expression into another expression. It is exact and it
can produce an answer exponentially larger than the input, because the product rule duplicates
subexpressions.

**Finite differences**, lesson 68, evaluate the function at nearby points. The step size has to
balance truncation against rounding, so the best achievable accuracy is ``sqrt(eps)`` for a forward
difference and ``eps**(2/3)`` for a central one, and no choice of step does better.

**Automatic differentiation** is neither. It applies the chain rule to the elementary operations the
program actually performed. There is no step size, so there is no truncation error at all, and the
result is exact to rounding. It is not a better finite difference; it is a different object.

Two modes follow from one choice: whether to carry derivatives forward with the values, or to record
the operations and walk back through them.

**Forward mode** attaches a derivative to every value, using dual numbers ``a + b*e`` with
``e**2 = 0``. One sweep gives the derivative in one input direction, so a gradient of ``n`` inputs
costs ``n`` sweeps. Memory is constant.

**Reverse mode** records every operation on a tape, then propagates adjoints backwards from the
output. One backward sweep gives the whole gradient whatever ``n`` is, which is why every deep
learning framework uses it, and the price is that the tape has to be kept.

What the measurements here show
-------------------------------
* **Both modes are exact to rounding**, at 1.3e-16 relative against derivatives worked out by hand
  on five test functions, and they agree with each other to 1.9e-16. Three of the five come out
  exactly right, with an error of 0.0.
* **A finite difference has a floor and this does not.** The forward difference bottoms out at
  1.9e-10 and the central one at 4.9e-12, while automatic differentiation's error on the same
  problem is exactly 0.
* **The best step is where lesson 68 said it would be.** The measured optimum for the forward
  difference, 2.15e-08, matches the predicted ``2 sqrt(eps |f| / |f''|)`` to a factor of 1.017.
* **The plain ``sqrt(eps)`` rule is off by 80 here**, because the floor carries the function's own
  ``|f|`` and ``|f''|``, and quoting it without them is quoting a scale rather than a number.
* **A gradient costs one reverse sweep and n forward sweeps.** Reverse mode's operation count is
  flat at 2.67 times one function evaluation from 2 inputs to 256, and forward mode's is exactly n
  times, so at 256 inputs reverse mode is 96 times cheaper.
* **The tape is the price.** Its length is exactly 3 entries per elementary step, fitted at a slope
  of 3.000, while forward mode carries 2 numbers whatever the depth.
* **Checkpointing costs exactly one extra forward pass** at every gap tested, and takes the memory
  from 769 to 58, a saving of 13.3. The best gap, 8, sits next to the predicted
  ``sqrt(depth/3) = 9.2``.
* **Which mode is cheaper depends only on the shape of the Jacobian.** From 2 inputs to 64 outputs
  forward mode wins by 32.6; from 64 inputs to 2 outputs reverse mode wins by 18.3; at 16 by 16 the
  two counts are within 8 per cent of each other.
* **Automatic differentiation differentiates the program, not the function.** At the kink of
  ``abs``, the same mathematical function written three ways returns -1.0, +1.0 and 0.0, with no
  warning of any kind, and the answer jumps between an input of -1e-16 and +1e-16.
* **Unrolling an iterative solver converges to the implicit derivative**, from a relative error of
  0.294 after one Newton step to exactly 0 after five, while the tape grows from 8 entries to 44
  and the implicit route needs 4 at any iteration count.
* **Forward over reverse gives an exact Hessian**, matching the analytic one to 0.0 and coming out
  symmetric on its own to 0.0, with no new code in either mode. A Hessian-vector product costs 2
  sweeps against the 10 a full 5 by 5 Hessian needs, and a finite difference Hessian of the same
  function is wrong by 7.4e-06.

Everything here derives its sizes from its input.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- forward mode


class Dual:
    """A number carrying its own derivative: ``a + b e`` with ``e**2 = 0``.

    Every rule of calculus falls out of the algebra. ``(a + b e)(c + d e) = ac + (ad + bc) e`` is
    the product rule, and it is the product rule because the ``e**2`` term vanishes. Nothing is
    approximated anywhere, which is why the result is exact.
    """

    __slots__ = ("value", "derivative")

    #: Elementary operations performed since the counter was last reset. Class level on purpose:
    #: the cost measurements need a count that survives across expressions.
    operations = 0

    def __init__(self, value, derivative=0.0):
        self.value = value
        self.derivative = derivative

    @classmethod
    def reset(cls) -> None:
        cls.operations = 0

    @classmethod
    def _count(cls, n: int = 1):
        cls.operations += n

    def __repr__(self) -> str:
        return f"Dual({self.value!r}, {self.derivative!r})"

    @staticmethod
    def _lift(other):
        return other if isinstance(other, Dual) else Dual(other, 0.0)

    def __add__(self, other):
        other = Dual._lift(other)
        Dual._count()
        return Dual(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __neg__(self):
        Dual._count()
        return Dual(-self.value, -self.derivative)

    def __sub__(self, other):
        return self + (-Dual._lift(other))

    def __rsub__(self, other):
        return Dual._lift(other) + (-self)

    def __mul__(self, other):
        other = Dual._lift(other)
        Dual._count()
        return Dual(self.value * other.value,
                    self.derivative * other.value + self.value * other.derivative)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Dual._lift(other)
        Dual._count()
        return Dual(self.value / other.value,
                    (self.derivative * other.value - self.value * other.derivative)
                    / (other.value * other.value))

    def __rtruediv__(self, other):
        return Dual._lift(other) / self

    def __pow__(self, power):
        Dual._count()
        return Dual(self.value ** power, power * self.value ** (power - 1) * self.derivative)

    def __lt__(self, other):
        return self.value < (other.value if isinstance(other, Dual) else other)

    def __gt__(self, other):
        return self.value > (other.value if isinstance(other, Dual) else other)


def sin(x):
    """``sin`` for a float, a Dual or a taped variable."""
    if isinstance(x, Dual):
        Dual._count()
        return Dual(_sin(x.value), _cos(x.value) * x.derivative)
    if isinstance(x, Var):
        return x._unary(_sin(x.value), _cos(x.value))
    return math.sin(x)


def cos(x):
    """``cos``, dispatching the same three ways."""
    if isinstance(x, Dual):
        Dual._count()
        return Dual(_cos(x.value), -_sin(x.value) * x.derivative)
    if isinstance(x, Var):
        return x._unary(_cos(x.value), -_sin(x.value))
    return math.cos(x)


def exp(x):
    """``exp``, whose derivative is itself."""
    if isinstance(x, Dual):
        Dual._count()
        value = _exp(x.value)
        return Dual(value, value * x.derivative)
    if isinstance(x, Var):
        value = _exp(x.value)
        return x._unary(value, value)
    return math.exp(x)


def log(x):
    """``log``."""
    if isinstance(x, Dual):
        Dual._count()
        return Dual(_log(x.value), x.derivative / x.value)
    if isinstance(x, Var):
        return x._unary(_log(x.value), 1.0 / x.value)
    return math.log(x)


def sqrt(x):
    """``sqrt``, whose derivative is unbounded at zero, which is a fact about the function."""
    if isinstance(x, Dual):
        Dual._count()
        value = _sqrt(x.value)
        return Dual(value, x.derivative / (2.0 * value))
    if isinstance(x, Var):
        value = _sqrt(x.value)
        return x._unary(value, 0.5 / value)
    return math.sqrt(x)


def tanh(x):
    """``tanh``, the one activation function with no kink."""
    if isinstance(x, Dual):
        Dual._count()
        value = _tanh(x.value)
        return Dual(value, (1.0 - value * value) * x.derivative)
    if isinstance(x, Var):
        value = _tanh(x.value)
        return x._unary(value, 1.0 - value * value)
    return math.tanh(x)


def _sin(v):
    return sin(v) if isinstance(v, Dual) else math.sin(v)


def _cos(v):
    return cos(v) if isinstance(v, Dual) else math.cos(v)


def _exp(v):
    return exp(v) if isinstance(v, Dual) else math.exp(v)


def _log(v):
    return log(v) if isinstance(v, Dual) else math.log(v)


def _sqrt(v):
    return sqrt(v) if isinstance(v, Dual) else math.sqrt(v)


def _tanh(v):
    return tanh(v) if isinstance(v, Dual) else math.tanh(v)


def directional_derivative(f, point, direction) -> dict:
    """One forward sweep: the value and the derivative along one direction."""
    x = np.asarray(point, dtype=float)
    v = np.asarray(direction, dtype=float)
    Dual.reset()
    seeded = [Dual(float(x[i]), float(v[i])) for i in range(x.size)]
    out = f(seeded)
    return {"value": out.value if isinstance(out, Dual) else out,
            "derivative": out.derivative if isinstance(out, Dual) else 0.0,
            "operations": Dual.operations}


def gradient_forward(f, point) -> dict:
    """A gradient by forward mode, which needs one sweep per input."""
    x = np.asarray(point, dtype=float)
    grad = np.zeros(x.size)
    total = 0
    value = 0.0
    for i in range(x.size):
        seed = np.zeros(x.size)
        seed[i] = 1.0
        out = directional_derivative(f, x, seed)
        grad[i] = out["derivative"]
        value = out["value"]
        total += out["operations"]
    return {"value": value, "gradient": grad, "operations": total, "sweeps": x.size}


# --------------------------------------------------------------------------- reverse mode


class Tape:
    """The record of every elementary operation, with its local partial derivatives.

    Reverse mode is two passes. The forward pass evaluates the function and writes this list. The
    backward pass walks the list from the end, accumulating adjoints, and finishes with the
    derivative of the output with respect to every input at once. The list is the whole idea, and
    its length is the memory cost.
    """

    def __init__(self):
        self.entries: list[list[tuple[int, float]]] = []

    def __len__(self) -> int:
        return len(self.entries)

    def push(self, dependencies) -> int:
        self.entries.append(list(dependencies))
        return len(self.entries) - 1

    def variable(self, value) -> "Var":
        return Var(value, self, self.push(()))

    def backward(self, index):
        """Adjoints of every recorded value, seeded with 1 at ``index``."""
        adjoint = [0.0] * len(self.entries)
        adjoint[index] = 1.0
        for position in range(len(self.entries) - 1, -1, -1):
            weight = adjoint[position]
            if weight == 0.0:
                continue
            for parent, partial in self.entries[position]:
                adjoint[parent] = adjoint[parent] + weight * partial
        return adjoint


class Var:
    """A value on a tape. Arithmetic on it records what was done and to what."""

    __slots__ = ("value", "tape", "index")

    def __init__(self, value, tape: Tape, index: int):
        self.value = value
        self.tape = tape
        self.index = index

    def __repr__(self) -> str:
        return f"Var({self.value!r}, #{self.index})"

    def _unary(self, value, partial) -> "Var":
        return Var(value, self.tape, self.tape.push([(self.index, partial)]))

    def _binary(self, other, value, own, theirs) -> "Var":
        if isinstance(other, Var):
            return Var(value, self.tape,
                       self.tape.push([(self.index, own), (other.index, theirs)]))
        return Var(value, self.tape, self.tape.push([(self.index, own)]))

    @staticmethod
    def _value(item):
        return item.value if isinstance(item, Var) else item

    def __add__(self, other):
        return self._binary(other, self.value + Var._value(other), 1.0, 1.0)

    __radd__ = __add__

    def __neg__(self):
        return self._unary(-self.value, -1.0)

    def __sub__(self, other):
        return self._binary(other, self.value - Var._value(other), 1.0, -1.0)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        return self._binary(other, self.value * Var._value(other),
                            Var._value(other), self.value)

    __rmul__ = __mul__

    def __truediv__(self, other):
        value = Var._value(other)
        return self._binary(other, self.value / value, 1.0 / value,
                            -self.value / (value * value))

    def __rtruediv__(self, other):
        value = Var._value(other)
        return self._unary(value / self.value, -value / (self.value * self.value))

    def __pow__(self, power):
        return self._unary(self.value ** power, power * self.value ** (power - 1))

    def __lt__(self, other):
        return self.value < Var._value(other)

    def __gt__(self, other):
        return self.value > Var._value(other)


def gradient_reverse(f, point) -> dict:
    """A gradient by reverse mode: one forward sweep to build the tape, one backward sweep."""
    x = np.asarray(point, dtype=float)
    tape = Tape()
    inputs = [tape.variable(float(item)) for item in x]
    out = f(inputs)
    if not isinstance(out, Var):
        return {"value": float(out), "gradient": np.zeros(x.size), "tape": len(tape),
                "operations": len(tape), "sweeps": 1}
    adjoint = tape.backward(out.index)
    grad = np.array([adjoint[item.index] for item in inputs])
    return {"value": out.value, "gradient": grad, "tape": len(tape),
            "operations": 2 * len(tape), "sweeps": 1}


def jacobian_forward(f, point, outputs: int) -> dict:
    """A Jacobian by forward mode: one sweep per input."""
    x = np.asarray(point, dtype=float)
    rows = np.zeros((int(outputs), x.size))
    total = 0
    for i in range(x.size):
        Dual.reset()
        seeded = [Dual(float(x[j]), 1.0 if j == i else 0.0) for j in range(x.size)]
        values = f(seeded)
        for r, item in enumerate(values):
            rows[r, i] = item.derivative if isinstance(item, Dual) else 0.0
        total += Dual.operations
    return {"jacobian": rows, "operations": total, "sweeps": x.size}


def jacobian_reverse(f, point, outputs: int) -> dict:
    """A Jacobian by reverse mode: one backward sweep per output."""
    x = np.asarray(point, dtype=float)
    rows = np.zeros((int(outputs), x.size))
    tape = Tape()
    inputs = [tape.variable(float(item)) for item in x]
    values = f(inputs)
    for r, item in enumerate(values):
        adjoint = tape.backward(item.index)
        rows[r] = [adjoint[node.index] for node in inputs]
    return {"jacobian": rows, "tape": len(tape),
            "operations": len(tape) * (1 + int(outputs)), "sweeps": int(outputs)}


def hessian(f, point) -> dict:
    """Forward over reverse: run the tape with Dual values so the adjoints carry derivatives.

    Nothing in the tape knows it is being handed dual numbers. The chain rule is the chain rule
    whatever the arithmetic underneath, so composing the two modes needs no new code at all, which
    is the cleanest argument that these are algebra rather than approximation.
    """
    x = np.asarray(point, dtype=float)
    rows = np.zeros((x.size, x.size))
    sweeps = 0
    for i in range(x.size):
        tape = Tape()
        inputs = [tape.variable(Dual(float(x[j]), 1.0 if j == i else 0.0))
                  for j in range(x.size)]
        out = f(inputs)
        adjoint = tape.backward(out.index)
        for j, node in enumerate(inputs):
            entry = adjoint[node.index]
            rows[i, j] = entry.derivative if isinstance(entry, Dual) else 0.0
        sweeps += 2
    scale = float(np.max(np.abs(rows))) or 1.0
    return {"hessian": 0.5 * (rows + rows.T), "raw": rows, "sweeps": sweeps,
            "asymmetry": float(np.max(np.abs(rows - rows.T))) / scale}


def hessian_vector(f, point, direction) -> dict:
    """The product ``H v`` in two sweeps, without ever forming ``H``."""
    x = np.asarray(point, dtype=float)
    v = np.asarray(direction, dtype=float)
    tape = Tape()
    inputs = [tape.variable(Dual(float(x[j]), float(v[j]))) for j in range(x.size)]
    out = f(inputs)
    adjoint = tape.backward(out.index)
    product = np.array([adjoint[node.index].derivative
                        if isinstance(adjoint[node.index], Dual) else 0.0
                        for node in inputs])
    return {"product": product, "tape": len(tape), "sweeps": 2}


# --------------------------------------------------------------------------- finite differences


def forward_difference(f, point, direction, step: float) -> float:
    """Lesson 68's one-sided difference, for comparison."""
    x = np.asarray(point, dtype=float)
    v = np.asarray(direction, dtype=float)
    return (f(list(x + step * v)) - f(list(x))) / step


def central_difference(f, point, direction, step: float) -> float:
    """Lesson 68's two-sided difference, which is second order and has a better floor."""
    x = np.asarray(point, dtype=float)
    v = np.asarray(direction, dtype=float)
    return (f(list(x + step * v)) - f(list(x - step * v))) / (2.0 * step)


# --------------------------------------------------------------------------- test functions


def problems() -> dict:
    """Five scalar functions with derivatives worked out by hand, as the known answers."""
    return {
        "polynomial": {
            "f": lambda v: v[0] ** 3 - 2.0 * v[0] ** 2 + 5.0 * v[0] - 1.0,
            "derivative": lambda x: 3.0 * x[0] ** 2 - 4.0 * x[0] + 5.0,
            "at": np.array([1.3]),
        },
        "product": {
            "f": lambda v: v[0] * v[1] * v[2],
            "derivative": lambda x: np.array([x[1] * x[2], x[0] * x[2], x[0] * x[1]]),
            "at": np.array([1.1, -0.7, 2.3]),
        },
        "trig": {
            "f": lambda v: sin(v[0]) * cos(v[1]),
            "derivative": lambda x: np.array([math.cos(x[0]) * math.cos(x[1]),
                                              -math.sin(x[0]) * math.sin(x[1])]),
            "at": np.array([0.6, 1.9]),
        },
        "exponential": {
            "f": lambda v: exp(v[0] * v[1]) / (1.0 + v[0] ** 2),
            "derivative": lambda x: np.array([
                math.exp(x[0] * x[1]) * (x[1] * (1.0 + x[0] ** 2) - 2.0 * x[0])
                / (1.0 + x[0] ** 2) ** 2,
                x[0] * math.exp(x[0] * x[1]) / (1.0 + x[0] ** 2)]),
            "at": np.array([0.8, 0.4]),
        },
        "layer": {
            "f": lambda v: tanh(v[0] + 2.0 * v[1]) * log(1.0 + v[2] ** 2),
            "derivative": lambda x: np.array([
                (1.0 - math.tanh(x[0] + 2.0 * x[1]) ** 2) * math.log(1.0 + x[2] ** 2),
                2.0 * (1.0 - math.tanh(x[0] + 2.0 * x[1]) ** 2) * math.log(1.0 + x[2] ** 2),
                math.tanh(x[0] + 2.0 * x[1]) * 2.0 * x[2] / (1.0 + x[2] ** 2)]),
            "at": np.array([0.5, -0.3, 1.4]),
        },
    }


def chain(depth: int):
    """A deep composition, whose tape length is proportional to its depth."""
    def f(v):
        state = v[0]
        for _ in range(int(depth)):
            state = tanh(state * 1.1 + 0.3)
        return state
    return f


def rosenbrock(v):
    """The lesson 84 test function, in any dimension, for the Hessian measurement."""
    total = 0.0
    for i in range(len(v) - 1):
        total = total + 100.0 * (v[i + 1] - v[i] ** 2) ** 2 + (1.0 - v[i]) ** 2
    return total


def sum_of_squares(v):
    """A quadratic whose gradient and Hessian are known exactly, in any dimension."""
    total = 0.0
    for i in range(len(v)):
        total = total + float(i + 1) * v[i] ** 2
    return total


# --------------------------------------------------------------------------- measurements


def forward_mode_is_exact_with_no_step_size() -> dict:
    """Compare dual numbers against derivatives worked out by hand, on five functions.

    There is no tolerance to choose here and no step size to tune. Either the algebra is right and
    the answer is exact to rounding, or it is wrong.
    """
    rows = []
    for name, case in problems().items():
        point = case["at"]
        wanted = np.atleast_1d(np.asarray(case["derivative"](point), dtype=float))
        got = gradient_forward(case["f"], point)["gradient"]
        taped = gradient_reverse(case["f"], point)["gradient"]
        scale = float(np.linalg.norm(wanted))
        rows.append({
            "name": name, "inputs": point.size,
            "forward_error": float(np.linalg.norm(got - wanted) / scale),
            "reverse_error": float(np.linalg.norm(taped - wanted) / scale),
            "modes_agree": float(np.linalg.norm(got - taped) / scale),
        })
    return {
        "rows": rows,
        "worst_forward": max(r["forward_error"] for r in rows),
        "worst_reverse": max(r["reverse_error"] for r in rows),
        "both_are_exact": max(max(r["forward_error"], r["reverse_error"]) for r in rows) < 1e-14,
        "the_modes_agree": max(r["modes_agree"] for r in rows) < 1e-14,
        "note": "there is no step size in either mode, so there is no truncation error to trade "
                "against rounding",
    }


def a_finite_difference_has_a_floor_and_this_does_not(powers=None, name: str = "layer") -> dict:
    """Sweep the finite difference step over many decades and find the best each one can do.

    Lesson 68 predicts the two floors: ``sqrt(eps)`` for the forward difference, at a step of
    ``sqrt(eps)``, and ``eps**(2/3)`` for the central one, at a step of ``eps**(1/3)``. Automatic
    differentiation has no step, so it has no such curve.
    """
    case = problems()[name]
    point = case["at"]
    exact = np.atleast_1d(np.asarray(case["derivative"](point), dtype=float))
    direction = np.zeros(point.size)
    direction[0] = 1.0
    wanted = float(exact[0])
    eps = float(np.finfo(float).eps)
    steps = np.geomspace(1e-1, 1e-14, 40) if powers is None else np.asarray(powers, dtype=float)
    rows = []
    for step in steps:
        one_sided = forward_difference(case["f"], point, direction, float(step))
        two_sided = central_difference(case["f"], point, direction, float(step))
        rows.append({
            "step": float(step),
            "forward": abs(one_sided - wanted) / abs(wanted),
            "central": abs(two_sided - wanted) / abs(wanted),
        })
    automatic = abs(directional_derivative(case["f"], point, direction)["derivative"]
                    - wanted) / abs(wanted)
    best_forward = min(rows, key=lambda r: r["forward"])
    best_central = min(rows, key=lambda r: r["central"])
    size = abs(float(case["f"](list(point))))
    curvature = abs(float(hessian(case["f"], point)["hessian"][0, 0]))
    detailed_floor = 2.0 * math.sqrt(eps * size * curvature) / abs(wanted)
    detailed_step = 2.0 * math.sqrt(eps * size / curvature)
    return {
        "rows": rows, "automatic": automatic,
        "best_forward": best_forward, "best_central": best_central,
        "predicted_forward_floor": math.sqrt(eps),
        "predicted_central_floor": eps ** (2.0 / 3.0),
        "predicted_forward_step": math.sqrt(eps),
        "predicted_central_step": eps ** (1.0 / 3.0),
        "detailed_forward_floor": detailed_floor,
        "detailed_forward_step": detailed_step,
        "function_value": size, "second_derivative": curvature,
        "forward_floor_over_root_eps": best_forward["forward"] / math.sqrt(eps),
        "forward_floor_over_the_detailed_one": best_forward["forward"] / detailed_floor,
        "forward_step_over_the_detailed_one": best_forward["step"] / detailed_step,
        "central_floor_over_eps_two_thirds": best_central["central"] / eps ** (2.0 / 3.0),
        "central_step_over_eps_one_third": best_central["step"] / eps ** (1.0 / 3.0),
        "automatic_is_exact": automatic < 1e-15,
        "automatic_beats_both": automatic < min(best_forward["forward"],
                                                best_central["central"]),
        "the_best_step_is_where_it_was_predicted": (
            0.2 < best_forward["step"] / detailed_step < 5.0),
        "the_forward_floor_scales_like_root_eps": (
            0.01 < best_forward["forward"] / detailed_floor < 100.0),
        "the_central_floor_is_eps_to_two_thirds": (
            0.02 < best_central["central"] / eps ** (2.0 / 3.0) < 50.0),
        "the_constants_matter": abs(math.log10(best_forward["forward"] / math.sqrt(eps))) > 1.0,
        "note": "the finite difference floor is a trade between truncation and rounding, with the "
                "function's own constants in it, and automatic differentiation is not on that "
                "trade at all",
    }


def a_gradient_costs_one_reverse_sweep(dimensions=(2, 4, 8, 16, 32, 64, 128, 256)) -> dict:
    """Count elementary operations for a gradient, both modes, as the input count grows.

    The theory says forward mode costs ``n`` times a function evaluation and reverse mode costs a
    small constant times one, whatever ``n`` is. Both statements are about counted operations, so
    both can be checked exactly rather than by timing.
    """
    rows = []
    for n in dimensions:
        point = np.linspace(0.4, 1.2, int(n))
        Dual.reset()
        base = [Dual(float(item), 0.0) for item in point]
        sum_of_squares(base)
        single = Dual.operations
        ahead = gradient_forward(sum_of_squares, point)
        behind = gradient_reverse(sum_of_squares, point)
        rows.append({
            "inputs": int(n), "one_evaluation": single,
            "forward_operations": ahead["operations"],
            "reverse_operations": behind["operations"],
            "forward_ratio": ahead["operations"] / single,
            "reverse_ratio": behind["operations"] / single,
            "tape": behind["tape"],
            "agree": float(np.max(np.abs(ahead["gradient"] - behind["gradient"]))),
        })
    reverse_ratios = [r["reverse_ratio"] for r in rows]
    return {
        "rows": rows,
        "reverse_is_flat": max(reverse_ratios) / min(reverse_ratios) < 1.3,
        "worst_reverse_ratio": max(reverse_ratios),
        "forward_grows_with_the_dimension": all(
            abs(r["forward_ratio"] / r["inputs"] - 1.0) < 0.05 for r in rows),
        "largest_saving": rows[-1]["forward_operations"] / rows[-1]["reverse_operations"],
        "the_modes_agree": max(r["agree"] for r in rows) < 1e-12,
        "note": "reverse mode is why a network with a billion parameters can be trained at all, "
                "and the constant is about 3",
    }


def the_tape_is_the_price(depths=(8, 16, 32, 64, 128, 256)) -> dict:
    """Measure the tape length against the depth of the computation being differentiated.

    Reverse mode has to keep every intermediate value until the backward pass reaches it, so its
    memory grows with the length of the computation while forward mode's does not. That is the whole
    trade between the two modes, and it is linear.
    """
    rows = []
    for depth in depths:
        f = chain(int(depth))
        point = np.array([0.7])
        behind = gradient_reverse(f, point)
        Dual.reset()
        ahead = directional_derivative(f, point, np.array([1.0]))
        rows.append({
            "depth": int(depth), "tape": behind["tape"],
            "forward_memory": 2, "operations": ahead["operations"],
            "agree": abs(behind["gradient"][0] - ahead["derivative"]),
        })
    slope = float(np.polyfit([r["depth"] for r in rows], [r["tape"] for r in rows], 1)[0])
    return {
        "rows": rows, "slope": slope,
        "the_tape_grows_linearly": abs(slope - rows[-1]["tape"] / rows[-1]["depth"]) < 0.5,
        "forward_memory_is_constant": len({r["forward_memory"] for r in rows}) == 1,
        "the_modes_agree": max(r["agree"] for r in rows) < 1e-12,
        "largest_tape": rows[-1]["tape"],
        "note": "forward mode carries two numbers and reverse mode carries the whole history, "
                "which is the only reason anyone still uses forward mode",
    }


def checkpointing_trades_memory_for_time(gaps=(1, 2, 4, 8, 16, 32, 64), depth: int = 256) -> dict:
    """Store every k-th state, recompute the rest, and count what has to be held at once.

    Two things occupy memory: the stored checkpoints, of which there are ``depth/k``, and the tape
    for the one segment being replayed, which holds about ``3k`` entries. Their sum is smallest near
    ``k = sqrt(depth)``, and the extra work is a constant factor of 2 whatever ``k`` is, because the
    forward pass is done exactly twice.
    """
    point = np.array([0.7])
    reference = gradient_reverse(chain(int(depth)), point)
    rows = []
    for gap in gaps:
        segment = min(int(gap), int(depth))
        pieces = int(math.ceil(depth / segment))
        peak = 0
        recomputed = 0
        derivative = 1.0
        state = float(point[0])
        stored = [state]
        for piece in range(pieces):
            for _ in range(min(segment, int(depth) - piece * segment)):
                state = math.tanh(state * 1.1 + 0.3)
            stored.append(state)
        for piece in range(pieces - 1, -1, -1):
            length = min(segment, int(depth) - piece * segment)
            local = Tape()
            node = local.variable(stored[piece])
            walker = node
            for _ in range(length):
                walker = tanh(walker * 1.1 + 0.3)
            adjoint = local.backward(walker.index)
            derivative = derivative * adjoint[node.index]
            peak = max(peak, len(local))
            recomputed += length
        total = peak + len(stored)
        rows.append({
            "gap": segment, "segments": pieces, "checkpoints": len(stored),
            "segment_tape": peak, "held_at_once": total,
            "forward_passes": 1 + recomputed / int(depth),
            "memory_saving": reference["tape"] / total,
            "error": abs(derivative - reference["gradient"][0]) / abs(reference["gradient"][0]),
        })
    best = min(rows, key=lambda r: r["held_at_once"])
    predicted = math.sqrt(int(depth) / 3.0)
    return {
        "rows": rows, "depth": int(depth), "full_tape": reference["tape"],
        "best_gap": best["gap"], "best_memory": best["held_at_once"],
        "predicted_gap": predicted,
        "largest_saving": max(r["memory_saving"] for r in rows),
        "extra_work": max(r["forward_passes"] for r in rows),
        "answers_are_unchanged": max(r["error"] for r in rows) < 1e-10,
        "the_work_is_two_passes_whatever_the_gap": (
            max(r["forward_passes"] for r in rows) - min(r["forward_passes"] for r in rows) < 1e-9),
        "the_memory_has_an_interior_minimum": (rows[0]["held_at_once"] > best["held_at_once"]
                                               and rows[-1]["held_at_once"]
                                               > best["held_at_once"]),
        "the_optimum_is_near_the_square_root": 0.3 < best["gap"] / predicted < 3.0,
        "note": "checkpointing changes nothing about the answer, costs one extra forward pass, and "
                "takes the memory from linear in the depth to its square root",
    }


def which_mode_wins_is_only_the_shape(shapes=((2, 64), (8, 32), (16, 16), (32, 8), (64, 2))) -> dict:
    """Build functions with different input and output counts and count the work for a Jacobian.

    Forward mode needs one sweep per input and reverse mode one per output, so the choice is
    decided by the shape of the Jacobian and by nothing else. A loss function has one output, which
    is the entire reason deep learning uses reverse mode.
    """
    rows = []
    for inputs, outputs in shapes:
        n, m = int(inputs), int(outputs)
        weights = np.linspace(0.2, 1.1, n)

        def f(v, count=m, scale=weights):
            values = []
            for r in range(count):
                total = 0.0
                for i in range(len(v)):
                    total = total + float(scale[i]) * v[i] * float(r + 1)
                values.append(tanh(total))
            return values

        point = np.linspace(-0.4, 0.6, n)
        ahead = jacobian_forward(f, point, m)
        behind = jacobian_reverse(f, point, m)
        rows.append({
            "inputs": n, "outputs": m,
            "forward_operations": ahead["operations"],
            "reverse_operations": behind["operations"],
            "ratio": ahead["operations"] / behind["operations"],
            "winner": "forward" if ahead["operations"] < behind["operations"] else "reverse",
            "agree": float(np.max(np.abs(ahead["jacobian"] - behind["jacobian"]))),
        })
    return {
        "rows": rows,
        "forward_wins_tall": rows[0]["winner"] == "forward",
        "reverse_wins_wide": rows[-1]["winner"] == "reverse",
        "largest_forward_advantage": max(1.0 / r["ratio"] for r in rows),
        "largest_reverse_advantage": max(r["ratio"] for r in rows),
        "the_modes_agree": max(r["agree"] for r in rows) < 1e-12,
        "note": "the Jacobian is the same matrix either way, and only the cost of getting it "
                "depends on which mode is used",
    }


def it_differentiates_the_program_not_the_function(point: float = 0.0) -> dict:
    """Ask for the derivative of ``abs``, ``relu`` and ``max`` at their kink, three ways.

    None of these has a derivative at zero. Automatic differentiation does not say so: it returns
    whatever the branch that was taken implies, with no warning, and different but mathematically
    identical ways of writing the same function give different answers.
    """
    def by_branch(v):
        return v[0] if v[0] > 0.0 else -v[0]

    def by_other_branch(v):
        return -v[0] if v[0] < 0.0 else v[0]

    def relu(v):
        return v[0] if v[0] > 0.0 else 0.0 * v[0]

    def smooth(v):
        return sqrt(v[0] * v[0] + 1e-12)

    rows = []
    for name, f in (("abs, positive branch first", by_branch),
                    ("abs, negative branch first", by_other_branch),
                    ("relu", relu),
                    ("a smoothed abs", smooth)):
        out = directional_derivative(f, np.array([float(point)]), np.array([1.0]))
        rows.append({"written_as": name, "value": out["value"],
                     "derivative": out["derivative"]})
    answers = {round(r["derivative"], 12) for r in rows}
    nearby = []
    for offset in (-1e-8, -1e-16, 0.0, 1e-16, 1e-8):
        out = directional_derivative(by_branch, np.array([float(point) + offset]),
                                     np.array([1.0]))
        nearby.append({"at": float(point) + offset, "derivative": out["derivative"]})
    return {
        "rows": rows, "nearby": nearby,
        "distinct_answers": len(answers),
        "it_answers_without_complaining": all(math.isfinite(r["derivative"]) for r in rows),
        "the_answer_depends_on_how_it_was_written": len(answers) > 1,
        "it_jumps_across_the_kink": nearby[0]["derivative"] != nearby[-1]["derivative"],
        "note": "the derivative returned at a kink is a property of the source code, and every "
                "value it returns there is defensible and none is the derivative",
    }


def unrolling_a_solver_converges_to_the_implicit_answer(counts=(1, 2, 3, 4, 5, 6),
                                                        parameter: float = 0.7) -> dict:
    """Differentiate a Newton solve two ways: through the iterations, and through the equation.

    The solution of ``x**3 + t x - 1 = 0`` depends on ``t``, and the implicit function theorem gives
    ``dx/dt`` from one linear solve at the answer. Unrolling the iteration instead differentiates
    the *approximation*, so its derivative is only as good as the approximation, and its tape grows
    with the iteration count.
    """
    t = float(parameter)

    def residual(x, param):
        return x ** 3 + param * x - 1.0

    exact = float(np.roots([1.0, 0.0, t, -1.0]).real[
        np.argmin(np.abs(np.roots([1.0, 0.0, t, -1.0]).imag))])
    implicit = -exact / (3.0 * exact ** 2 + t)
    rows = []
    for steps in counts:
        def unrolled(v, k=int(steps)):
            x = Dual(1.0, 0.0) if isinstance(v[0], Dual) else 1.0
            for _ in range(k):
                x = x - residual(x, v[0]) / (3.0 * x * x + v[0])
            return x

        out = directional_derivative(unrolled, np.array([t]), np.array([1.0]))
        tape = gradient_reverse(unrolled, np.array([t]))
        rows.append({
            "steps": int(steps), "x": out["value"],
            "solution_error": abs(out["value"] - exact) / abs(exact),
            "derivative": out["derivative"],
            "derivative_error": abs(out["derivative"] - implicit) / abs(implicit),
            "tape": tape["tape"],
        })
    return {
        "rows": rows, "root": exact, "implicit_derivative": implicit,
        "implicit_tape": 4,
        "it_converges": rows[-1]["derivative_error"] < 1e-12,
        "one_step_is_wrong": rows[0]["derivative_error"] > 0.1,
        "the_tape_grows_with_the_iterations": rows[-1]["tape"] > 3 * rows[0]["tape"],
        "final_error": rows[-1]["derivative_error"],
        "note": "the implicit function theorem gives the derivative from the equation rather than "
                "from the iteration, at constant memory and one linear solve",
    }


def forward_over_reverse_gives_an_exact_hessian(dimension: int = 5) -> dict:
    """Run the reverse tape over dual numbers, and compare against a Hessian worked out by hand.

    Nothing in the tape code changes. The adjoints become dual numbers because the partials do, and
    their derivative parts are the second derivatives. Two sweeps give one Hessian column, or one
    Hessian-vector product with no Hessian anywhere.
    """
    n = int(dimension)
    point = np.linspace(0.3, 1.1, n)
    exact = np.diag(2.0 * np.arange(1, n + 1, dtype=float))
    out = hessian(sum_of_squares, point)
    rng = np.random.default_rng(42)
    direction = rng.standard_normal(n)
    product = hessian_vector(sum_of_squares, point, direction)
    rough = np.linspace(-1.2, 1.1, n)
    rosenbrock_hessian = hessian(rosenbrock, rough)["hessian"]
    step = 1e-5
    numeric = np.zeros((n, n))
    for i in range(n):
        forward, backward = rough.copy(), rough.copy()
        forward[i] += step
        backward[i] -= step
        for j in range(n):
            plus, minus = forward.copy(), backward.copy()
            plus[j] += step
            minus[j] += step
            numeric[i, j] = ((rosenbrock(list(plus)) - rosenbrock(list(forward))
                              - rosenbrock(list(minus)) + rosenbrock(list(backward)))
                             / (2.0 * step * step))
    return {
        "hessian": out["hessian"], "exact": exact,
        "error": float(np.max(np.abs(out["hessian"] - exact))),
        "sweeps_for_the_full_hessian": out["sweeps"],
        "sweeps_for_one_product": product["sweeps"],
        "product_error": float(np.max(np.abs(product["product"] - exact @ direction))),
        "asymmetry": out["asymmetry"],
        "rosenbrock_gap": float(np.max(np.abs(rosenbrock_hessian - numeric))
                                / np.max(np.abs(rosenbrock_hessian))),
        "finite_difference_step": step,
        "it_is_exact": float(np.max(np.abs(out["hessian"] - exact))) < 1e-12,
        "the_product_is_exact": float(np.max(np.abs(product["product"]
                                                    - exact @ direction))) < 1e-12,
        "the_product_is_cheaper": product["sweeps"] < out["sweeps"],
        "it_comes_out_symmetric_on_its_own": out["asymmetry"] < 1e-12,
        "finite_differences_are_not": float(np.max(np.abs(rosenbrock_hessian - numeric))
                                            / np.max(np.abs(rosenbrock_hessian))) > 1e-9,
        "note": "composing the two modes needs no new code, and a Hessian-vector product costs two "
                "sweeps against the n a full Hessian needs",
    }
