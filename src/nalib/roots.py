"""Root finding for a single nonlinear equation f(x) = 0.

Every method here returns a `RootResult`, which carries the whole iteration history rather
than just the answer. That is deliberate: most of what Part 2 teaches is about *how* a method
approaches the root, not where it ends up, and you cannot measure a convergence rate from a
single number.

The methods split into two families, and the split matters more than any individual method:

**Bracketing methods** keep an interval known to contain a root, so they cannot fail. They are
slow and they are unconditionally safe.

    bisection, regula_falsi, illinois

**Open methods** iterate from a guess with no interval to fall back on. They are fast and they
can diverge.

    fixed_point, newton, secant, muller, chebyshev, halley

**Hybrid** methods try to get both.

    brent

Used by lessons 09 to 14.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- result type


@dataclass
class RootResult:
    """The outcome of a root finding run, including the full history.

    Attributes
    ----------
    root
        The final approximation.
    iterates
        Every approximation produced, in order, starting from the initial guess.
    residuals
        `|f(x)|` at each iterate. This is the **backward error** of the computed root
        (lesson 06), so it is the right thing to use in a stopping test.
    converged
        Whether the stopping test was met before the iteration limit.
    n_iter
        Iterations performed.
    n_feval
        Evaluations of `f`. This is the honest cost measure: the secant method does one
        evaluation per step and Newton does two, so comparing by iteration count alone
        flatters Newton.
    message
        Why the iteration stopped.
    brackets
        For bracketing methods, `(a, b)` at each step. `bracket_widths` derives from it.
    """

    root: float
    iterates: np.ndarray
    residuals: np.ndarray
    converged: bool
    n_iter: int
    n_feval: int
    message: str
    brackets: list[tuple[float, float]] = field(default_factory=list)

    def errors(self, exact: float) -> np.ndarray:
        """|x_k - exact| at every step. Feed this to `nalib.convergence`."""
        return np.abs(self.iterates - float(exact))

    @property
    def bracket_widths(self) -> np.ndarray:
        """Width of the bracket at each step, for bracketing methods.

        Unlike the actual error, this decreases monotonically for bisection. Lesson 07
        explains why that distinction matters when measuring a convergence rate.
        """
        return np.array([abs(b - a) for a, b in self.brackets])

    @property
    def error_bounds(self) -> np.ndarray:
        """Guaranteed bound on |x_k - r| at each step, for bracketing methods.

        Half the bracket width, because the reported iterate is the midpoint (bisection) or
        at worst somewhere inside (the others). This is the number to quote when you want to
        say "the root is known to within", and it is what `bisection_steps_needed` predicts.
        """
        return self.bracket_widths / 2.0

    def __repr__(self) -> str:
        return (f"RootResult(root={self.root!r}, converged={self.converged}, "
                f"n_iter={self.n_iter}, n_feval={self.n_feval}, "
                f"residual={self.residuals[-1]:.3e})")


class _Counter:
    """Wraps f and counts how many times it is actually called.

    The return value is passed through unchanged rather than cast to `float`, because
    `muller` works in complex arithmetic. Every other method here calls it with real
    arguments and gets real values back.
    """

    def __init__(self, f: Callable[[float], float]) -> None:
        self.f = f
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.f(x)


def _opposite_signs(p: float, q: float) -> bool:
    """True when p and q straddle zero, tested WITHOUT forming their product.

    The obvious test is ``p * q < 0``. It is wrong, and lesson 09 explains why, so none of the
    bracketing methods here may use it. Three ways the product fails:

    - **overflow**: two values around 1e200 multiply to inf, and the sign can be lost,
    - **underflow**: two values around 1e-200 multiply to exactly 0, so ``p*q < 0`` is False
      and the bracket is updated the wrong way. Measured: bisection on
      ``f(x) = 1e-200 (x - 1.3)`` over [0, 3] returned 3.0 instead of 1.3, an error of 1.7,
    - **nan**: ``inf * 0`` is nan, and every comparison against nan is False.

    Comparing signs directly cannot overflow, cannot underflow, and needs no arithmetic at
    all. Zero is treated as non-negative here; callers test ``f == 0`` separately before
    reaching this, since an exact zero means the root has been found.
    """
    return (p < 0.0) != (q < 0.0)


def _same_sign(p: float, q: float) -> bool:
    """The negation of `_opposite_signs`, for readability at the guard clauses."""
    return (p < 0.0) == (q < 0.0)


def _finish(xs, rs, converged, counter, message, brackets=None) -> RootResult:
    return RootResult(
        root=float(xs[-1]),
        iterates=np.asarray(xs, dtype=float),
        residuals=np.asarray(rs, dtype=float),
        converged=converged,
        n_iter=len(xs) - 1,
        n_feval=counter.count,
        message=message,
        brackets=brackets or [],
    )


# ---------------------------------------------------------------- bracketing methods


def bisection(f, a, b, tol=1e-14, max_iter=200) -> RootResult:
    """Bisection. Halve the bracket, keep the half where the sign changes.

    Guaranteed to converge to a root whenever `f(a)` and `f(b)` have opposite signs and `f`
    is continuous, by the intermediate value theorem (lesson 07). After k steps the root is
    known to within `(b-a)/2**(k+1)`, and that bound holds no matter what `f` does inside.

    Linear convergence with rate exactly 1/2. The **bracket** halves exactly every step; the
    distance from the midpoint to the root does not, and can even grow. Use
    `result.bracket_widths` when measuring the rate.

    The midpoint is computed as `a + (b-a)/2` rather than `(a+b)/2`, which cannot overflow
    and is guaranteed to lie inside the bracket even in floating point.
    """
    counter = _Counter(f)
    a, b = float(a), float(b)
    fa, fb = counter(a), counter(b)
    if fa == 0.0:
        return _finish([a], [0.0], True, counter, "left endpoint is a root", [(a, b)])
    if fb == 0.0:
        return _finish([b], [0.0], True, counter, "right endpoint is a root", [(a, b)])
    if _same_sign(fa, fb):
        raise ValueError(
            f"bisection needs a sign change: f({a}) = {fa:g} and f({b}) = {fb:g} "
            "have the same sign"
        )

    xs, rs, brackets = [], [], []
    for _ in range(max_iter):
        m = a + (b - a) / 2
        fm = counter(m)
        xs.append(m)
        rs.append(abs(fm))
        brackets.append((a, b))
        if fm == 0.0:
            return _finish(xs, rs, True, counter, "exact root found", brackets)
        if abs(b - a) / 2 <= tol:
            return _finish(xs, rs, True, counter, "bracket within tolerance", brackets)
        if _opposite_signs(fa, fm):
            b, fb = m, fm
        else:
            a, fa = m, fm
    return _finish(xs, rs, False, counter, "iteration limit reached", brackets)


def regula_falsi(f, a, b, tol=1e-14, max_iter=200) -> RootResult:
    """False position. Like bisection, but interpolate a line instead of taking the midpoint.

    Also guaranteed to converge, since it keeps a bracket. Usually faster than bisection, but
    it has a notorious failure mode: on a function with fixed convexity, **one endpoint never
    moves**, the bracket never shrinks, and convergence degrades to linear with a rate that
    can be close to 1. Lesson 09 measures this.

    See `illinois` for the standard cure.
    """
    return _false_position(f, a, b, tol, max_iter, illinois=False)


def illinois(f, a, b, tol=1e-14, max_iter=200) -> RootResult:
    """The Illinois variant of false position.

    When the same endpoint is retained twice in a row, halve its function value before the
    next interpolation. That artificially pulls the secant line toward the stuck endpoint and
    forces it to move. The bracket is still maintained, so the guarantee survives, and the
    observed order rises to about 1.44.

    One of the best cost-to-safety ratios of any method here: as safe as bisection, nearly as
    fast as the secant method.
    """
    return _false_position(f, a, b, tol, max_iter, illinois=True)


def _false_position(f, a, b, tol, max_iter, illinois: bool) -> RootResult:
    counter = _Counter(f)
    a, b = float(a), float(b)
    fa, fb = counter(a), counter(b)
    if _same_sign(fa, fb):
        raise ValueError("false position needs a sign change on the bracket")

    xs, rs, brackets = [], [], []
    side = 0  # +1 if the right endpoint was kept last, -1 if the left one was
    for _ in range(max_iter):
        if fb == fa:
            return _finish(xs or [a], rs or [abs(fa)], False, counter,
                           "flat secant, cannot interpolate", brackets)
        c = (a * fb - b * fa) / (fb - fa)
        fc = counter(c)
        xs.append(c)
        rs.append(abs(fc))
        brackets.append((a, b))

        if fc == 0.0 or abs(b - a) <= tol or abs(fc) <= tol:
            return _finish(xs, rs, True, counter, "converged", brackets)

        if _opposite_signs(fa, fc):
            b, fb = c, fc
            if illinois and side == -1:
                fa *= 0.5
            side = -1
        else:
            a, fa = c, fc
            if illinois and side == +1:
                fb *= 0.5
            side = +1
    return _finish(xs, rs, False, counter, "iteration limit reached", brackets)


# ---------------------------------------------------------------- open methods


def fixed_point(g, x0, tol=1e-14, max_iter=500) -> RootResult:
    """Fixed point iteration x <- g(x).

    Converges to a fixed point r when |g'(r)| < 1, linearly with rate |g'(r)|. Diverges when
    |g'(r)| > 1, no matter how good the starting guess is. Lesson 10 makes both visible.

    The residual reported is |g(x) - x|, which is zero exactly at a fixed point.
    """
    counter = _Counter(g)
    x = float(x0)
    xs, rs = [x], [abs(counter(x) - x)]
    counter.count -= 1  # that call was only to report the initial residual
    for _ in range(max_iter):
        gx = counter(x)
        xs.append(gx)
        rs.append(abs(gx - x))
        if not math.isfinite(gx):
            return _finish(xs, rs, False, counter, "iteration diverged to a non-finite value")
        if abs(gx - x) <= tol:
            return _finish(xs, rs, True, counter, "converged")
        if abs(gx) > 1e300:
            return _finish(xs, rs, False, counter, "iteration diverged")
        x = gx
    return _finish(xs, rs, False, counter, "iteration limit reached")


def newton(f, df, x0, tol=1e-14, max_iter=100, multiplicity=1) -> RootResult:
    """Newton's method.

    Quadratically convergent to a **simple** root, given a good enough starting guess. At a
    root of multiplicity m the convergence drops to linear with rate 1 - 1/m, and passing
    `multiplicity=m` restores the quadratic rate by taking the step `m f/f'` instead.

    Two evaluations per step, one of f and one of f'. Compare against the secant method,
    which needs one.

    Fails when f'(x) is zero or tiny, and can cycle or run away from a poor start. Lesson 11
    shows both.
    """
    counter = _Counter(f)
    dcounter = _Counter(df)
    x = float(x0)
    xs, rs = [x], [abs(counter(x))]
    for _ in range(max_iter):
        fx = counter(x)
        dfx = dcounter(x)
        if dfx == 0.0:
            return _finish(xs, rs, False, counter,
                           "derivative vanished, Newton step undefined")
        step = multiplicity * fx / dfx
        x_new = x - step
        if not math.isfinite(x_new):
            return _finish(xs, rs, False, counter, "iteration diverged")
        xs.append(x_new)
        rs.append(abs(counter(x_new)))
        if abs(step) <= tol or rs[-1] == 0.0:
            counter.count += dcounter.count
            return _finish(xs, rs, True, counter, "converged")
        x = x_new
    counter.count += dcounter.count
    return _finish(xs, rs, False, counter, "iteration limit reached")


def secant(f, x0, x1, tol=1e-14, max_iter=100) -> RootResult:
    """Secant method. Newton with the derivative replaced by a finite difference.

    Superlinear with order the golden ratio, about 1.618. One new evaluation of f per step,
    against Newton's two, so per *evaluation* it is often faster than Newton even though it
    needs more iterations.

    No bracket, so it can diverge. The step is written as
    `x1 - f1 (x1 - x0) / (f1 - f0)` rather than the algebraically identical
    `(x0 f1 - x1 f0) / (f1 - f0)`, because the first form is an update to a good value and
    the second subtracts two nearly equal products near convergence.
    """
    counter = _Counter(f)
    x0, x1 = float(x0), float(x1)
    f0, f1 = counter(x0), counter(x1)
    xs, rs = [x0, x1], [abs(f0), abs(f1)]
    for _ in range(max_iter):
        if f1 == f0:
            return _finish(xs, rs, False, counter, "flat secant, cannot continue")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if not math.isfinite(x2):
            return _finish(xs, rs, False, counter, "iteration diverged")
        f2 = counter(x2)
        xs.append(x2)
        rs.append(abs(f2))
        if abs(x2 - x1) <= tol or f2 == 0.0:
            return _finish(xs, rs, True, counter, "converged")
        x0, f0, x1, f1 = x1, f1, x2, f2
    return _finish(xs, rs, False, counter, "iteration limit reached")


def muller(f, x0, x1, x2, tol=1e-14, max_iter=100) -> RootResult:
    """Muller's method. Fit a parabola through three points and take one of its roots.

    The secant method fits a line through two points; Muller fits a quadratic through three.
    Order about 1.84, between the secant method and Newton.

    Its real advantage is that a quadratic can have complex roots, so Muller finds **complex
    roots from real starting points**. Neither bisection nor the secant method can. This
    implementation works in complex arithmetic and returns a real value when the imaginary
    part is negligible.
    """
    counter = _Counter(f)
    x0, x1, x2 = complex(x0), complex(x1), complex(x2)
    f0, f1, f2 = counter(x0), counter(x1), counter(x2)
    xs = [x0, x1, x2]
    rs = [abs(f0), abs(f1), abs(f2)]

    for _ in range(max_iter):
        h0, h1 = x1 - x0, x2 - x1
        if h0 == 0 or h1 == 0 or (h1 + h0) == 0:
            break
        d0, d1 = (f1 - f0) / h0, (f2 - f1) / h1
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        disc = np.sqrt(complex(b * b - 4 * a * f2))
        # Choose the denominator of larger magnitude: same cancellation fix as the stable
        # quadratic formula in lesson 05.
        denom = b + disc if abs(b + disc) >= abs(b - disc) else b - disc
        if denom == 0:
            break
        x3 = x2 - 2 * f2 / denom
        f3 = counter(x3)
        xs.append(x3)
        rs.append(abs(f3))
        if abs(x3 - x2) <= tol or f3 == 0:
            break
        x0, f0, x1, f1, x2, f2 = x1, f1, x2, f2, x3, f3

    arr = np.asarray(xs)
    if np.max(np.abs(arr.imag)) < 1e-12 * max(1.0, np.max(np.abs(arr.real))):
        arr = arr.real
    root = arr[-1]
    return RootResult(
        root=root if np.iscomplexobj(arr) else float(root),
        iterates=arr,
        residuals=np.asarray(rs, dtype=float),
        converged=rs[-1] <= max(tol, 1e-12),
        n_iter=len(xs) - 3,
        n_feval=counter.count,
        message="converged" if rs[-1] <= max(tol, 1e-12) else "stopped",
    )


def chebyshev(f, df, d2f, x0, tol=1e-14, max_iter=100) -> RootResult:
    """Chebyshev's third order method.

    Newton keeps one more term of the Taylor expansion than the secant method; Chebyshev
    keeps one more than Newton:

        x <- x - (f/f') (1 + (f f'') / (2 f'^2))

    Cubic convergence at a simple root, so the number of correct digits roughly **triples**
    each step. The price is a second derivative, three evaluations per step. Whether that is
    worth it depends entirely on how expensive f is.
    """
    counter = _Counter(f)
    c1, c2 = _Counter(df), _Counter(d2f)
    x = float(x0)
    xs, rs = [x], [abs(counter(x))]
    for _ in range(max_iter):
        fx, dfx, d2fx = counter(x), c1(x), c2(x)
        if dfx == 0.0:
            return _finish(xs, rs, False, counter, "derivative vanished")
        t = fx / dfx
        x_new = x - t * (1.0 + (fx * d2fx) / (2.0 * dfx * dfx))
        if not math.isfinite(x_new):
            return _finish(xs, rs, False, counter, "iteration diverged")
        xs.append(x_new)
        rs.append(abs(counter(x_new)))
        if abs(x_new - x) <= tol or rs[-1] == 0.0:
            counter.count += c1.count + c2.count
            return _finish(xs, rs, True, counter, "converged")
        x = x_new
    counter.count += c1.count + c2.count
    return _finish(xs, rs, False, counter, "iteration limit reached")


def halley(f, df, d2f, x0, tol=1e-14, max_iter=100) -> RootResult:
    """Halley's method, the other standard third order method.

        x <- x - 2 f f' / (2 f'^2 - f f'')

    Same cubic order as `chebyshev` and the same cost. Halley is usually the more robust of
    the two because its denominator is less likely to be destroyed by a large second
    derivative.
    """
    counter = _Counter(f)
    c1, c2 = _Counter(df), _Counter(d2f)
    x = float(x0)
    xs, rs = [x], [abs(counter(x))]
    for _ in range(max_iter):
        fx, dfx, d2fx = counter(x), c1(x), c2(x)
        denom = 2.0 * dfx * dfx - fx * d2fx
        if denom == 0.0:
            return _finish(xs, rs, False, counter, "Halley denominator vanished")
        x_new = x - 2.0 * fx * dfx / denom
        if not math.isfinite(x_new):
            return _finish(xs, rs, False, counter, "iteration diverged")
        xs.append(x_new)
        rs.append(abs(counter(x_new)))
        if abs(x_new - x) <= tol or rs[-1] == 0.0:
            counter.count += c1.count + c2.count
            return _finish(xs, rs, True, counter, "converged")
        x = x_new
    counter.count += c1.count + c2.count
    return _finish(xs, rs, False, counter, "iteration limit reached")


# ---------------------------------------------------------------- acceleration


def aitken(sequence) -> np.ndarray:
    """Aitken's delta-squared acceleration of a linearly convergent sequence.

        x'_k = x_k - (x_{k+1} - x_k)^2 / (x_{k+2} - 2 x_{k+1} + x_k)

    Assumes the error behaves like `e_{k+1} = C e_k` and extrapolates to where the sequence
    is heading. Turns linear convergence into something much faster, at no extra function
    evaluations, using only the numbers you already have.

    Returns a sequence two elements shorter than the input. The denominator is a difference
    of nearly equal numbers once the sequence has converged, so this is lesson 05 territory:
    stop accelerating once the terms stop changing.
    """
    x = np.asarray(sequence, dtype=float).ravel()
    if x.size < 3:
        return np.array([])
    d1 = x[1:-1] - x[:-2]
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    out = np.full(x.size - 2, np.nan)
    ok = d2 != 0
    out[ok] = x[:-2][ok] - d1[ok] ** 2 / d2[ok]
    return out


def steffensen(g, x0, tol=1e-14, max_iter=100) -> RootResult:
    """Steffensen's method: Aitken acceleration folded into a fixed point iteration.

        x <- x - (g(x) - x)^2 / (g(g(x)) - 2 g(x) + x)

    Achieves **quadratic** convergence using only values of g, with no derivative at all.
    Two evaluations of g per step. This is the answer to "can I get Newton's speed without
    Newton's derivative", and the answer is yes, at the cost of a second evaluation and less
    robustness.
    """
    counter = _Counter(g)
    x = float(x0)
    xs, rs = [x], [abs(counter(x) - x)]
    counter.count -= 1
    for _ in range(max_iter):
        g1 = counter(x)
        g2 = counter(g1)
        denom = g2 - 2.0 * g1 + x
        if denom == 0.0:
            # Once the iteration has converged this difference is genuinely zero, which is
            # success rather than breakdown. Distinguish the two by the residual.
            done = abs(g1 - x) <= max(tol, 1e-14)
            return _finish(xs, rs, done, counter,
                           "converged, denominator now exactly zero" if done
                           else "Steffensen denominator vanished before convergence")
        x_new = x - (g1 - x) ** 2 / denom
        if not math.isfinite(x_new):
            return _finish(xs, rs, False, counter, "iteration diverged")
        xs.append(x_new)
        rs.append(abs(x_new - x))
        if abs(x_new - x) <= tol:
            return _finish(xs, rs, True, counter, "converged")
        x = x_new
    return _finish(xs, rs, False, counter, "iteration limit reached")


# ---------------------------------------------------------------- hybrid


def brent(f, a, b, tol=1e-14, max_iter=200) -> RootResult:
    """Brent's method. Inverse quadratic interpolation, guarded by bisection.

    The design goal is to have both properties at once: the guaranteed convergence of
    bisection and the superlinear speed of interpolation. Each step tries inverse quadratic
    interpolation, falls back to the secant step if only two points are usable, and falls
    back to bisection whenever the interpolated point is unacceptable.

    "Unacceptable" means outside the bracket, or not shrinking the interval fast enough. That
    guard is what makes the worst case no worse than bisection while the typical case is
    nearly as fast as the secant method.

    This is what `scipy.optimize.brentq` implements, and it is the sensible default when you
    have a bracket and no derivative.
    """
    counter = _Counter(f)
    eps = np.finfo(float).eps

    a, b = float(a), float(b)
    fa, fb = counter(a), counter(b)
    if _same_sign(fa, fb):
        raise ValueError("Brent needs a sign change on the bracket")

    # b is always the best current estimate, a is the previous one, c is the point on the
    # opposite side of the root from b, so [b, c] is the live bracket.
    c, fc = a, fa
    d = e = b - a
    xs, rs, brackets = [b], [abs(fb)], [(min(b, c), max(b, c))]

    for _ in range(max_iter):
        if _same_sign(fb, fc):        # c must stay on the far side of the root
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):         # keep b as the better of the two
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa

        tol1 = 2.0 * eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or fb == 0.0:
            return _finish(xs, rs, True, counter, "converged", brackets)

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:                                   # only two points: secant step
                p, q = 2.0 * xm * s, 1.0 - s
            else:                                        # three points: inverse quadratic
                q_, r_ = fa / fc, fb / fc
                p = s * (2.0 * xm * q_ * (q_ - r_) - (b - a) * (r_ - 1.0))
                q = (q_ - 1.0) * (r_ - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            # Accept the interpolated step only if it stays safely inside the bracket and
            # is shrinking fast enough. This guard is what stops Brent ever being worse
            # than bisection.
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm                               # reject, bisect
        else:
            d = e = xm                                   # bisect

        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fb = counter(b)
        xs.append(b)
        rs.append(abs(fb))
        brackets.append((min(b, c), max(b, c)))
    return _finish(xs, rs, False, counter, "iteration limit reached", brackets)


# ---------------------------------------------------------------- helpers


def bisection_steps_needed(a: float, b: float, tol: float) -> int:
    """How many bisection steps guarantee an error below tol.

    From `(b-a)/2**(k+1) <= tol`. This is the only method in the file where the iteration
    count can be predicted exactly in advance, without knowing anything about f.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    return max(0, int(math.ceil(math.log2(abs(b - a) / tol)) - 1))


def METHODS_NEEDING_DERIVATIVE() -> tuple[str, ...]:
    return ("newton", "chebyshev", "halley")
