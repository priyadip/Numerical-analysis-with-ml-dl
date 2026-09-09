# 85. Derivative Free Optimization

**Part 12: Numerical Optimization**

## Learning objectives

By the end of this lesson you will be able to:

1. Derive the golden ratio from the requirement that one evaluation per step suffices, and predict
   the number of evaluations a golden section search will take.
2. Say what unimodality buys and what happens, silently, when it does not hold.
3. Implement successive parabolic interpolation, name its two failure modes, and say why every
   practical routine safeguards it with a bracket.
4. Explain why its convergence order cannot be measured in double precision, and measure it in
   extended precision instead.
5. Use the Nelder-Mead simplex, and state precisely what it does not guarantee.

## Prerequisites

Lesson 84 (the $\sqrt\varepsilon$ barrier, which is the reason this lesson exists and the reason
one of its measurements needs extended precision). Lesson 09 (bisection and the idea of a bracket).
Lesson 11 (the secant method and superlinear order, which parabolic interpolation is the
minimization analogue of). Lesson 44 (interpolating three points with a quadratic).

---

## 1. What is left when only values are available

Lesson 84 established the limit: reading values of $f$ locates a minimum to about
$\sqrt{\varepsilon\lvert f^{*}\rvert / c}$ and no better. That is a floor on accuracy. It says
nothing about **cost**, and cost is what this lesson is about.

Three methods, and the shape of the answer is different in one dimension and in more.

In one dimension the theory is complete. **Golden section search** keeps a bracket and shrinks it
by a guaranteed factor, so it cannot fail on a unimodal function and its cost is known in advance.
**Successive parabolic interpolation** throws the bracket away and is much faster when it works.
The practical answer is to run both and let the safe one catch the fast one, which is what Brent's
method is.

In more than one dimension there is no such theory, and section 5 shows what that costs.

---

## 2. Golden section search

Keep a bracket $[a, b]$ containing a minimum and two interior points $c < d$. Compare $f(c)$ and
$f(d)$: if $f(c) < f(d)$ the minimum is in $[a, d]$, otherwise in $[c, b]$. Either way the bracket
shrinks.

The question is where to put $c$ and $d$. Place them symmetrically at a fraction $r$ from each end,
so the surviving bracket is $r$ times the old one. **One of the two old points is inside the new
bracket**, and it should land exactly where a new point would go, so no evaluation is wasted. That
requires the old point to be at fraction $r$ of the new bracket measured from the other side:

$$
r^{2} = 1 - r \quad\Longrightarrow\quad r = \frac{\sqrt 5 - 1}{2} = 0.6180339887\ldots
$$

which is $1/\phi$. Every step then costs **one** evaluation and multiplies the bracket by $0.618$.

Compare that with the symmetric alternative, which places the two points close together near the
middle: it gets a factor of $0.5$ but must discard both points, so it costs two evaluations for
$0.5$, that is $\sqrt{0.5} = 0.707$ per evaluation. Golden section's $0.618$ per evaluation wins.

### 2.1 The measurement

```python
from nalib import derivfree as df

out = df.the_reduction_is_the_golden_ratio()
print(f"calls {out['calls']}, steps {out['steps']}, "
      f"one new call per step: {out['one_new_call_per_step']}")
print(f"mean reduction per step {out['mean_ratio']:.8f}")
print(f"1/phi                   {out['golden_ratio']:.8f}")
print(f"spread over all the steps {out['ratio_spread']:.2e}")
print(f"steps taken {out['steps']}, predicted {out['predicted_steps']:.1f}")
print(out["note"])

assert out["the_ratio_is_golden"]
assert out["one_new_call_per_step"]
assert out["steps_match_the_prediction"]
```

The measured factor matches $1/\phi$ to seven digits with a spread of $10^{-6}$ over fifty steps,
and the count matches $\log(\text{tol}/w)/\log(1/\phi)$. **The cost of a golden section search is
known before it runs**, which is a property bisection has and almost nothing else in this part
does.

### 2.2 Unimodality is required, and its absence is silent

The guarantee needs $f$ to be unimodal on the bracket: one minimum, decreasing then increasing. The
comparison $f(c) < f(d)$ is what discards half the interval, and with two wells that comparison can
discard the half containing the better one.

Nothing detects this. Here is the same function from three brackets.

```python
out = df.unimodality_is_required()
print(f"the two wells are at {out['wells'][0]:.6f} and {out['wells'][1]:.6f}, "
      f"the deeper one at {out['deeper_well']:.6f}")
print(f"{'bracket':>16}{'found x':>12}{'f there':>12}{'calls':>8}{'gap to a well':>16}")
for row in out["rows"]:
    print(f"{str(row['bracket']):>16}{row['x']:>12.6f}{row['f']:>12.4f}"
          f"{row['calls']:>8}{row['gap']:>16.2e}")
print(f"\ndifferent brackets give different answers: "
      f"{out['different_brackets_give_different_answers']}")
print(out["note"])

assert out["every_run_converged"]
assert out["a_run_found_the_shallower_well"]
```

Every run converges to a genuine local minimum, to $10^{-8}$, in the same 53 calls, and reports
nothing unusual. One of them returns the shallower well. **The answer is a property of the
bracket**, and the only defence is to know your function is unimodal or to bracket it yourself.

---

## 3. Successive parabolic interpolation

Fit a parabola through the last three points and jump to its vertex. With
$(x_0, f_0), (x_1, f_1), (x_2, f_2)$,

$$
x_{\text{new}} = x_1 - \frac{1}{2}\,
\frac{(x_1-x_0)^{2}(f_1-f_2) - (x_1-x_2)^{2}(f_1-f_0)}
     {(x_1-x_0)\,(f_1-f_2) - (x_1-x_2)\,(f_1-f_0)} .
$$

This is the minimization analogue of lesson 11's secant method: no derivative, no bracket, and
superlinear convergence when it works.

```python
out = df.parabolas_beat_golden_section()
print(f"golden section:  {out['golden_calls']} calls, "
      f"error {out['golden_error']:.3e}")
print(f"parabolic:       {out['parabolic_calls_to_the_same_accuracy']} calls to the "
      f"same accuracy, error {out['parabolic_error']:.3e}")
print(f"speedup {out['speedup']:.2f}x in evaluations")
print(f"\nthe parabolic errors, step by step:")
print("  " + " ".join(f"{v:.1e}" for v in out["parabolic_errors"]))
```

Fourteen evaluations against fifty three, on an easy problem. On a harder one the gap is larger,
because golden section's rate is fixed and this one's is not.

### 3.1 The two ways it fails

Neither is exotic and both appear in ordinary use.

```python
flat = df.successive_parabolic(lambda t: 0.0 * t + 1.0, (0.0, 1.0, 2.0))
print(f"three equal values: {flat['calls']} steps, stopped because "
      f"{flat['stopped_because']}")

import math
up = df.successive_parabolic(math.cos, (-1.0, 0.0, 1.0), rounds=5)
print(f"cos from (-1, 0, 1): first step goes to {up['history'][0]:.6f}, "
      f"where cos has a maximum")

assert flat["calls"] == 0
assert abs(up["history"][0]) < 1e-9
```

**A vanishing denominator.** Three collinear values give no parabola at all. Three nearly collinear
values give one whose vertex is arbitrarily far away, which is worse, because the method takes the
step.

**A vertex that is a maximum.** The parabola through three points can open downward, and its vertex
is then the worst point nearby rather than the best. Starting from $(-1, 0, 1)$ on $\cos$ the method
walks straight to the maximum at the origin and stays.

Both are why every practical one dimensional minimizer keeps a golden section bracket underneath.
Brent's method takes the parabolic step when it lands inside the bracket and is smaller than half
the previous step, and a golden section step otherwise. It gets the superlinear rate when the
function cooperates and the guaranteed rate when it does not.

### 3.2 The order, and why double precision cannot see it

The theoretical order is the real root of $t^{3} = t + 1$, the **plastic number**
$1.3247179\ldots$ Slower than the secant method's $1.618$, because a minimum carries less
information than a root, which is lesson 84's point again.

Trying to measure it runs straight into lesson 84's barrier.

```python
out = df.the_order_is_hidden_by_the_barrier()
print("errors in double precision:")
print("  " + " ".join(f"{v:.1e}" for v in out["double_errors"]))
print(f"it reaches the sqrt(eps) floor after {out['steps_to_reach_the_floor']} steps, "
      f"leaving {out['usable_points_in_double']} usable points")
print(f"fitted order in double precision: {out['double_order']:.4f}")
print()
print(f"errors in {out['high_precision_digits']} digit arithmetic:")
print("  " + " ".join(f"{v:.1e}" for v in out["high_precision_errors"]))
print(f"fitted order there:               {out['high_precision_order']:.5f}")
print(f"the plastic number:               {out['plastic_number']:.5f}")

assert out["double_precision_cannot_see_it"]
assert out["extended_precision_can"]
```

In double precision the iteration falls from $0.4$ to the $\sqrt\varepsilon$ floor in nine steps.
That leaves five points to fit a straight line through, and the fit reads $1.21$. Nothing is wrong
with the method or the fit: **there is not enough room between the starting error and the floor to
measure a superlinear rate.**

Repeat the identical iteration in 120 digit arithmetic and the errors keep falling,
$5.8\times10^{-10}, 2.2\times10^{-13}, 1.8\times10^{-17}, 2.1\times10^{-23}, 6.6\times10^{-31}$,
and the fit reads $1.31796$ against the exact $1.32472$.

This is the tidiest demonstration in this part that lesson 84's barrier is real and not a
bookkeeping detail. **It is not that the answer was inaccurate. It is that the experiment could not
be run.**

---

## 4. From scratch: golden section in ten lines

Neither method is complicated, and writing the search out makes the one evaluation per step visible
rather than asserted.

```python
import math

import numpy as np


def my_golden(f, a, b, tol=1e-10):
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    calls = 0

    def value(t):
        nonlocal calls
        calls += 1
        return float(f(t))

    left, right = b - ratio * (b - a), a + ratio * (b - a)
    f_left, f_right = value(left), value(right)
    while b - a > tol:
        if f_left < f_right:
            b, right, f_right = right, left, f_left
            left = b - ratio * (b - a)
            f_left = value(left)
        else:
            a, left, f_left = left, right, f_right
            right = a + ratio * (b - a)
            f_right = value(right)
    return 0.5 * (a + b), calls


print(f"{'function':>18}{'my answer':>14}{'library':>14}{'my calls':>10}"
      f"{'library calls':>15}")
for name, f, span in (("cos", math.cos, (2.0, 4.0)),
                      ("exp(x) - 2x", lambda t: math.exp(t) - 2.0 * t, (-1.0, 2.0)),
                      ("(x-2)**2 + 1", lambda t: (t - 2.0) ** 2 + 1.0, (0.0, 5.0))):
    mine, calls = my_golden(f, *span)
    theirs = df.golden_section(f, *span, tol=1e-10)
    print(f"{name:>18}{mine:>14.9f}{theirs['x']:>14.9f}{calls:>10}"
          f"{theirs['calls']:>15}")
    assert abs(mine - theirs["x"]) < 1e-9
```

Two evaluations to start and one for every step after, at every problem size, which is the property
the golden ratio was chosen for.

---

## 5. More than one dimension: the Nelder-Mead simplex

Keep $n+1$ points, a simplex. Sort them by value and try to replace the worst.

- **Reflect** it through the centroid of the others.
- If the reflection is the new best, **expand** further in the same direction.
- If it is still worse than the second worst, **contract** halfway.
- If even the contraction fails, **shrink** the whole simplex toward the best point.

No derivative is used and none is estimated. It handles noisy and non-smooth functions, it needs no
tuning, and it is the most used optimization method in science. It also has **no convergence
theory** in more than one dimension, and the rest of this section is what that means.

### 5.1 How it scales

```python
out = df.how_nelder_mead_scales()
print(f"{'variables':>11}{'calls':>9}{'f at the end':>15}{'distance to (1..1)':>21}"
      f"{'shrinks':>9}{'solved':>9}")
for row in out["rows"]:
    print(f"{row['variables']:>11}{row['calls']:>9}{row['f']:>15.4e}"
          f"{row['distance']:>21.4e}{row['shrinks']:>9}{str(row['solved']):>9}")
print(f"\nfitted power of n on the runs that worked: {out['fitted_power_of_n']:.4f}")
print(f"failed at: {out['failures']}")
print(out["note"])

assert out["the_failure_is_not_the_largest_size"]
```

The cost grows like $n^{1.8}$ on the runs that succeed. The interesting row is the one that does
not: at **eight** variables it stops with $f = 3.99$ and the simplex diameter at $8\times10^{-13}$,
having shrunk 16 times, while ten variables solves the same problem cleanly.

The failure is not monotone in the dimension, and that is the point rather than a caveat. A method
with a convergence theory fails predictably. This one fails where the particular sequence of moves
happens to collapse the simplex, and no property of the problem announces where.

### 5.2 It can converge to a point that is not a minimum

This is not a local minimum being mistaken for a global one. McKinnon constructed a function that
is **strictly convex**, so it has exactly one minimum and no other stationary point, on which
Nelder-Mead from a specific starting simplex converges to a point that is not it.

```python
import numpy as np

out = df.nelder_mead_can_converge_to_a_non_minimum()
print(f"{out['name']}")
print(f"it stopped at        {np.array2string(out['stopped_at'], precision=17)}")
print(f"with f =             {out['value_there']:.6f}")
print(f"the true minimum is at {np.array2string(out['true_minimizer'])} "
      f"with f = {out['true_minimum']}")
print(f"the gradient where it stopped is "
      f"{np.array2string(out['gradient_where_it_stopped'])}, norm "
      f"{out['gradient_norm']:.6f}")
print(f"the simplex diameter is {out['diameter']:.3e}")
print(f"moves used: {out['moves']}")
print(f"\nthe gap in value: {out['the_gap_in_value']:.4f}")

assert out["it_stopped"]
assert out["it_is_not_stationary"]
```

Every stopping test a simplex method has is satisfied: the diameter is $1.5\times10^{-15}$, the
vertex values agree, nothing has changed for many iterations. The gradient there is $(0, 1)$. The
minimum is at $(0, -1/2)$ with value $-1/4$, and the method returns $0$.

The move counts say how it happened: **200 contractions, no reflections and no expansions.** Every
step pulled the worst vertex in toward the centroid, the simplex flattened onto the line $x = 0$,
and once it is flat it can only move along a direction in which $f$ happens to be flat too.

The obvious objection is that the function must have a second minimum. It does not.

```python
out = df.the_function_really_is_convex()
print(f"{out['samples']} random chords tested")
print(f"worst violation of f((1-t)p + tq) <= (1-t)f(p) + t f(q): "
      f"{out['worst_chord_violation']:.3e}")
print(out["note"])

assert out["no_counterexample_found"]
```

Every chord lies above the function, by a margin, at every sample. Lesson 84 was clear that this
does not prove convexity, and here it does not need to: McKinnon's proof does that, and the
sampling only confirms there is no second well hiding where one would have to be.

### 5.3 What to do about it

Three things, in increasing order of effort.

**Restart it.** When the simplex collapses, rebuild it around the current best point with fresh
axis directions and run again. If the answer moves, the previous one was wrong. This costs a factor
of two and catches most of these failures.

**Use a method with a theory.** If the gradient is available, lesson 86's methods converge to a
stationary point under stated conditions. If it is not, the modern derivative free methods based on
trust regions and model interpolation have convergence proofs that Nelder-Mead does not.

**Check the answer.** A finite difference gradient at the reported point costs $2n$ evaluations and
would have caught this immediately: the norm is $1$, not $0$.

```python
import matplotlib.pyplot as plt
import numpy as np

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.0))

golden_run = df.golden_section(lambda t: math.exp(t) - 2.0 * t, -1.0, 2.0, tol=1e-12)
star = math.log(2.0)
parabolic = df.successive_parabolic(lambda t: math.exp(t) - 2.0 * t,
                                    (-1.0, 0.0, 2.0), rounds=20)
left.semilogy(np.arange(golden_run["widths"].size) + 2, golden_run["widths"],
              lw=1.6, label="golden section, bracket width")
errors = np.maximum(np.abs(parabolic["history"] - star), 1e-17)
left.semilogy(np.arange(errors.size) + 3, errors, "o--", ms=3.5, lw=1.4,
              label="parabolic, error")
left.axhline(float(np.sqrt(np.finfo(float).eps)), color="0.5", lw=1.0)
left.text(30, 2.4e-8, "sqrt(eps)", fontsize=8, color="0.35")
left.set_xlabel("function evaluations")
left.set_ylabel("bracket width or error")
left.set_title("linear against superlinear")
left.legend(fontsize=8)
left.grid(alpha=0.3, which="both")

problem = df.mckinnon_problem()
grid = np.linspace(-0.12, 1.1, 220)
up = np.linspace(-0.8, 1.1, 220)
gx, gy = np.meshgrid(grid, up, indexing="ij")
surface = np.array([[problem["f"]((x, y)) for y in up] for x in grid])
right.contour(gx, gy, surface, levels=np.linspace(-0.25, 6.0, 16), linewidths=0.6,
              colors="0.6")
corners = np.vstack([problem["start_simplex"], problem["start_simplex"][:1]])
right.plot(corners[:, 0], corners[:, 1], "-", lw=1.5, label="the starting simplex")
right.plot(0.0, 0.0, "o", ms=7, label="where it stops")
right.plot(0.0, -0.5, "*", ms=12, label="the actual minimum")
right.set_xlabel("x")
right.set_ylabel("y")
right.set_title("strictly convex, and it stops short")
right.legend(fontsize=8, loc="upper right")
right.grid(alpha=0.3)

fig.tight_layout(); fig.savefig("../figures/85_derivfree.png", dpi=110); plt.close(fig)
print("saved ../figures/85_derivfree.png")
```

![Golden section against parabolic interpolation, and the McKinnon failure](../figures/85_derivfree.png)

---

## 6. Exercises

**Level 1, understanding**

1.1 Explain why the golden ratio and not some other fraction.

1.2 Say what unimodality guarantees and what happens without it.

1.3 State the two failure modes of parabolic interpolation.

1.4 Explain why the plastic number cannot be measured in double precision.

1.5 Say what Nelder-Mead guarantees and what it does not.

**Level 2, derivation**

2.1 Derive $r^{2} = 1 - r$ from the requirement that one point is reused, and solve it.

2.2 Show that golden section beats a symmetric two point rule per evaluation.

2.3 Derive the parabolic interpolation formula from the three point quadratic of lesson 44.

2.4 Derive the error recurrence for parabolic interpolation and show its order is the root of
$t^{3} = t + 1$.

2.5 Show that the Nelder-Mead reflection preserves the simplex volume and the shrink divides it by
$2^{n}$.

**Level 3, computational**

3.1 Implement Brent's method, combining golden section with parabolic interpolation, and confirm it
is never slower than golden section alone.

3.2 Implement a coordinate descent search using a one dimensional minimizer, and measure it against
Nelder-Mead on Rosenbrock.

3.3 Implement Fibonacci search, which is optimal for a fixed budget, and compare its final bracket
against golden section's at the same number of evaluations.

3.4 Implement the restart rule of section 5.3 and measure whether it repairs the eight variable
Rosenbrock failure.

3.5 Implement a pattern search on a fixed mesh, which does have a convergence theory, and compare
its cost against Nelder-Mead.

**Level 4, experimental**

4.1 Measure the golden section call count against the tolerance over eight decades and fit the
slope.

4.2 Measure how often Nelder-Mead fails on Rosenbrock across many random starting simplices, at
each dimension from 2 to 12.

4.3 Measure how the parabolic interpolation order estimate changes with the working precision, from
double up to 200 digits.

**Level 5, advanced**

5.1 **Optimality of Fibonacci search.** Show that for a fixed number of evaluations decided in
advance, Fibonacci search gives the smallest possible final bracket, and say why golden section is
used anyway.

5.2 **Why the simplex collapses.** Analyse McKinnon's example and identify the property of the
function that makes every step an inside contraction.

5.3 **Noise.** Add noise of size $\sigma$ to $f$ and measure how the attainable accuracy of each
method changes. Predict the answer from lesson 84 first.

## 7. Key takeaways

- **The golden ratio comes from reusing a point.** $r^2 = 1 - r$, measured at $0.61803400$ against
  $0.61803399$, with one new evaluation per step.

- **A golden section search's cost is known in advance**, $\log(\text{tol}/w)/\log(1/\phi)$, which
  matched the measurement to within three steps.

- **Unimodality fails silently.** Three brackets on the same two well function all converge
  cleanly, in the same 53 calls, and one returns the shallower well.

- **Parabolic interpolation is worth a factor of 3.8** in evaluations here, and has no bracket, so
  it walks to maxima and divides by zero.

- **Its order is the plastic number $1.3247$ and double precision cannot show it.** The iteration
  reaches the $\sqrt\varepsilon$ floor in nine steps, the fit over the five usable points reads
  $1.21$, and the same run in 120 digits reads $1.31796$.

- **Nelder-Mead costs about $n^{1.8}$ evaluations and has no convergence theory.** It failed at
  eight variables and succeeded at ten, on the same problem.

- **It can stop at a non-stationary point of a strictly convex function.** 200 contractions, a
  simplex diameter of $1.5\times10^{-15}$, a gradient of norm $1$, and an answer wrong by $0.25$.

- **A $2n$ evaluation gradient check would have caught it.** That is the cheapest insurance in this
  part.

## Where this goes next

Lesson 86 assumes the gradient is available and asks what it buys. The answer is a convergence
theory, and a new problem: steepest descent's rate is set by the condition number of lesson 84,
which is where the ellipsoid picture stops being about precision and starts being about speed.
