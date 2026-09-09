# 70. Adaptive Step Size Control

**Part 10: Ordinary Differential Equations**

## Learning objectives

By the end of this lesson you will be able to:

1. Build an embedded pair and see why its error estimate costs nothing extra.
2. Derive the step control law $h_{\text{new}} = h(\text{tol}/e)^{1/(p+1)}$ and say where each
   piece comes from.
3. Recognise the first-same-as-last property and implement the stage reuse it allows.
4. Use local extrapolation, and say exactly what you give up by turning it on.
5. Compare an adaptive solver against fixed steps at the same accuracy, honestly.

## Prerequisites

Lesson 69 (Butcher tableaux, order conditions, stage counts). Lesson 64 (adaptive quadrature, the
same idea in one dimension less, including the local extrapolation argument). Lesson 63
(Richardson extrapolation).

---

## 1. Two answers from one set of stages

Take a Runge-Kutta method and find a **second** set of weights $\hat b$ that uses the same stages
and achieves a different order. The two answers are

$$
y_{n+1} = y_n + h\sum b_i k_i \quad (\text{order } p), \qquad
\hat y_{n+1} = y_n + h\sum \hat b_i k_i \quad (\text{order } p-1),
$$

and their difference estimates the lower order method's local error:

$$
e = \lVert y_{n+1} - \hat y_{n+1}\rVert.
$$

The stages $k_i$ are shared. **The estimate costs nothing.** That is the whole idea, and it is
written as one tableau with two bottom rows:

$$
\begin{array}{c|c} c & A \\ \hline & b^{\mathsf T} \\ \hline & \hat b^{\mathsf T}\end{array}
$$

```python
from nalib import adaptivestep as ad

for name in sorted(ad.PAIRS):
    A, bh, bl, c, ph, pl = ad.pair(name)
    print(f"{name:>18}: {len(bh)} stages, orders {ph} and {pl}, "
          f"first-same-as-last: {ad.is_first_same_as_last(name)}")
```

The alternative is **step halving**: take one step of size $h$, then two of size $h/2$, and
compare. That is Richardson extrapolation from lesson 63, it works, and it costs 3.5 times as
many stage evaluations per step.

```python
out = ad.the_estimate_is_free()
print(f"method: {out['name']}, {out['stages']} stages, FSAL: {out['first_same_as_last']}")
print(f"embedded pair:  {out['embedded_evaluations_per_step']} evaluations per step")
print(f"step halving:   {out['step_halving_evaluations_per_step']} evaluations per step")
print(f"ratio: {out['ratio']:.2f}x")
```

## 2. The step control law

The local error of a method of order $q$ behaves like $e \approx C h^{q+1}$. If the current step
$h$ gave error $e$ and you want error $\text{tol}$, then

$$
\frac{\text{tol}}{e} \approx \left(\frac{h_{\text{new}}}{h}\right)^{q+1}
\quad\Longrightarrow\quad
h_{\text{new}} = h\left(\frac{\text{tol}}{e}\right)^{1/(q+1)}.
$$

Three practical modifications, all of them there because the estimate is an estimate:

- A **safety factor** of about $0.9$, so a step aimed at the tolerance usually lands under it.
- A **growth cap**, typically 5, so one lucky step cannot launch the solver into a region it has
  not looked at.
- A **shrink floor**, typically $0.2$, so one unlucky step cannot collapse the step to nothing.

```python
print(f"{'error/tol':>12}{'raw factor':>14}{'with safety':>14}{'after caps':>13}")
for ratio in (1e-4, 1e-2, 0.5, 1.0, 2.0, 100.0, 1e6):
    error = ratio * 1e-6
    raw = (1e-6 / error) ** (1.0 / 5.0)
    capped = ad.proposed_step(0.1, error, 1e-6, order=4) / 0.1
    print(f"{ratio:>12.0e}{raw:>14.4f}{0.9 * raw:>14.4f}{capped:>13.4f}")
```

The caps bite at both ends. At $e = \text{tol}/10^4$ the law asks for a factor of $6.31$,
safety takes it to $5.68$, and the growth cap holds it at $5$. At $e = 10^6\,\text{tol}$ the law
asks for $0.063$, safety takes it to $0.057$, and the shrink floor lifts it back to $0.2$.

In the middle nothing is capped, which is where the controller spends nearly all its time. The
caps are there for the two step transitions the law handles badly.

## 3. Do the pairs have the orders they claim?

```python
from nalib import ivp

def f(t, y):
    return ivp.as_state(y) - t ** 2 + 1.0

def exact(t):
    return (t + 1.0) ** 2 - 0.5 * np.exp(t)

out = ad.orders_are_what_they_claim(f, exact, 0.0, 0.5, 2.0)
print(f"{'name':>18}{'claimed high':>14}{'fitted':>9}{'claimed low':>14}{'fitted':>9}")
for n, ch, fh, cl, fl in zip(out["names"], out["claimed_high"], out["fitted_high"],
                             out["claimed_low"], out["fitted_low"]):
    print(f"{n:>18}{ch:>14}{fh:>9.3f}{cl:>14}{fl:>9.3f}")
print(f"\nall match: {out['all_match']}")
assert out["all_match"], "both halves of every pair must have the order they claim"
```

## 4. Does the estimate predict the error?

The difference $y - \hat y$ estimates the **low** order method's error, not the high order one's.
That distinction is the whole content of section 5.

```python
out = ad.the_estimate_predicts_the_error(f, exact, 0.0, 0.5, 2.0)
print(f"{'h':>12}{'estimate':>14}{'low error':>14}{'high error':>14}"
      f"{'est/low':>10}{'est/high':>12}")
for h, e, lo, hi, rl, rh in zip(out["h"], out["estimate"], out["true_low_error"],
                                out["true_high_error"], out["estimate_over_low"],
                                out["estimate_over_high"]):
    print(f"{h:>12.5f}{e:>14.3e}{lo:>14.3e}{hi:>14.3e}{rl:>10.3f}{rh:>12.3e}")
```

Against the low order error the ratio is close to 1 and stays there. Against the high order error
it is enormous and grows, because the high order answer is much better than the estimate says.

## 5. Local extrapolation

You have two answers. Which one do you keep?

**Keep the low order one** and the error estimate applies to what you kept. The solver is honest:
it controls the error of the answer it returns.

**Keep the high order one** and you get accuracy you did not pay for, and the estimate no longer
describes what you kept. This is called **local extrapolation** and every production solver does
it, `ode45` and `solve_ivp` included.

```python
out = ad.local_extrapolation_is_free_accuracy(f, exact, 0.0, 0.5, 2.0)
print(f"{'tolerance':>12}{'with':>14}{'without':>14}{'gain':>12}"
      f"{'evals with':>13}{'evals without':>15}")
for tol, w, wo, g, ew, ewo in zip(out["tolerance"], out["with_extrapolation"],
                                  out["without_extrapolation"], out["gain"],
                                  out["evaluations_with"], out["evaluations_without"]):
    print(f"{tol:>12.0e}{w:>14.3e}{wo:>14.3e}{g:>12.1f}{ew:>13}{ewo:>15}")
```

The gain is 4 times at a loose tolerance and 25 times at a tight one, and the extrapolating
version uses **fewer** evaluations as well, so there is no trade here at all on those two axes.

What it costs is the meaning of the tolerance. With extrapolation on, `tol` is a **step size
control parameter** rather than an error bound: it sizes the steps, and section 7 measures what
the error then does.

There is a second cost, and it is easy to miss. **Turning off local extrapolation gives up
first-same-as-last.** Dormand-Prince's 7th stage is $f(t_{n+1}, y_{n+1})$ computed with the
**high order** $y_{n+1}$; if you return the low order answer instead, the stage you carried
forward is evaluated at the wrong point and has to be recomputed.

```python
out = ad.the_estimate_is_free("dormand prince")
counter = [0]
run = ad.solve(f, 0.0, 0.5, 2.0, tol=1e-8, name="dormand prince", _counter=counter)
print(f"with extrapolation:   {counter[0] / run['accepted']:.2f} evaluations per accepted step")
counter = [0]
run = ad.solve(f, 0.0, 0.5, 2.0, tol=1e-8, name="dormand prince",
               local_extrapolation=False, _counter=counter)
print(f"without:              {counter[0] / run['accepted']:.2f} evaluations per accepted step")
print(f"stages in the tableau: {out['stages']}")
```

## 6. Where the work goes

An adaptive solver puts its steps where the solution is hard. Watching where they land is the
best diagnostic there is for a problem you do not understand yet.

```python
import math

AMPLITUDE, WIDTH, CENTRE = 200.0, 400.0, 1.5

def pulse(t, y):
    """y' = -y + a narrow Gaussian pulse, so the solution is flat, then abrupt, then flat."""
    return -ivp.as_state(y) + AMPLITUDE * np.exp(-WIDTH * (t - CENTRE) ** 2)

def pulse_exact(t):
    """Available in closed form, by completing the square inside the convolution integral."""
    shift = CENTRE + 1.0 / (2.0 * WIDTH)
    root = math.sqrt(WIDTH)
    scale = (AMPLITUDE * math.exp(WIDTH * shift ** 2 - WIDTH * CENTRE ** 2)
             * math.sqrt(math.pi) / (2.0 * root))
    v = np.atleast_1d(np.asarray(t, dtype=float))
    return np.asarray([scale * math.exp(-float(x))
                       * (math.erf(root * (float(x) - shift)) + math.erf(root * shift))
                       for x in v])

out = ad.where_the_work_went(pulse, 0.0, 0.0, 3.0, tol=1e-8)
print(f"accepted {out['accepted']} steps, rejected {out['rejected']}, "
      f"{out['evaluations']} evaluations")
print(f"smallest step {out['smallest']:.3e}, largest {out['largest']:.3e}, "
      f"ratio {out['ratio']:.1f}x")
print(f"\n{'t':>10}{'step':>14}")
picks = np.linspace(0, len(out["h"]) - 1, 14).astype(int)
for i in picks:
    print(f"{out['t'][i]:>10.4f}{out['h'][i]:>14.3e}")
```

The solution is flat until $t \approx 1.4$, jumps by 16 over about a tenth of a time unit, and
decays smoothly afterwards. The step follows: $0.75$ where nothing is happening, $0.0047$ at the
steepest part, and back to $0.09$ once it is over. **That factor of 159 is what adaptivity buys.**
A fixed step solver would have to use the smallest step for the whole interval.

Notice the 18 rejected steps out of 86. The controller chooses the next step from the last one's
error, so it cannot see the pulse coming and has to be told about it by failing. **That is the
price of a purely local controller**, and it is why exercise 3.5 asks for a PI controller, which
uses two past errors instead of one and rejects less.

```python
fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.5))
run = ad.solve(pulse, 0.0, 0.0, 3.0, tol=1e-8)
top.plot(run["t"], run["y"][:, 0], "-", lw=1.5)
top.set_ylabel("y"); top.set_title("the solution, and the step the controller chose")
steps = ad.where_the_work_went(pulse, 0.0, 0.0, 3.0, tol=1e-8)
bottom.semilogy(steps["t"], steps["h"], ".", markersize=4)
bottom.set_xlabel("t"); bottom.set_ylabel("step size")
fig.tight_layout(); fig.savefig("../figures/70_adaptive_steps.png", dpi=110); plt.close(fig)
print("saved ../figures/70_adaptive_steps.png")
```

![Where the adaptive solver puts its steps](../figures/70_adaptive_steps.png)

The step collapses by two orders of magnitude exactly where the solution turns and recovers
immediately afterwards. **Nobody told the solver where the pulse was**; it found it by failing a
few steps and shrinking.

## 7. Against fixed steps, and is the tolerance met?

```python
out = ad.against_fixed_steps(pulse, pulse_exact, 0.0, 0.0, 3.0)
print(f"{'tolerance':>12}{'adaptive error':>16}{'evals':>9}{'fixed error':>15}"
      f"{'evals':>9}{'wins':>7}")
for tol, ae, av, fe, fv, w in zip(out["tolerance"], out["adaptive_error"],
                                  out["adaptive_evaluations"], out["fixed_error"],
                                  out["fixed_evaluations"], out["adaptive_wins"]):
    print(f"{tol:>12.0e}{ae:>16.3e}{av:>9}{fe:>15.3e}{fv:>9}{str(w):>7}")
```

Both solvers get the same number of evaluations; the question is which gets further with them.

**The adaptive solver wins by 1700 times at $10^{-4}$, by 3800 at $10^{-6}$, by 7 at $10^{-8}$,
and then loses by 160 at $10^{-10}$.** The advantage does not just shrink, it reverses.

The reason is worth working out, because it is the general shape. The fixed grid has to be fine
enough for the pulse, and once it is, it is **far** finer than needed for the flat parts, and a
fifth order method converts that surplus into accuracy for nothing. The adaptive solver refuses
to do that: it spends exactly the tolerance everywhere, including where a bigger step would have
been wasteful and a smaller one would have been free.

So adaptivity pays when the tolerance is loose enough that the easy regions can genuinely be
skipped, and stops paying once the hard region forces a grid that resolves everything anyway.
**On this problem the crossover is around $10^{-9}$.** Where it sits depends on how localised the
difficulty is, and on a problem with a real singularity it never arrives.

```python
out = ad.tolerance_is_met(f, exact, 0.0, 0.5, 2.0)
print(f"{'tolerance':>12}{'achieved':>14}{'error/tol':>12}{'accepted':>10}{'rejected':>10}")
for tol, err, ratio, a, r in zip(out["tolerance"], out["achieved_error"],
                                 out["error_over_tolerance"], out["accepted"],
                                 out["rejected"]):
    print(f"{tol:>12.0e}{err:>14.3e}{ratio:>12.4f}{a:>10}{r:>10}")
print(f"\nalways met: {out['always_met']}, worst ratio {out['worst_ratio']:.4f}")
assert not out["always_met"], "the tolerance is local and the error reported is global"
assert out["worst_ratio"] > 1.0, "and here the global error exceeds it"
```

**The tolerance is not met.** The ratio runs from $0.59$ at $10^{-4}$ up to $2.41$ at
$10^{-12}$, so the achieved error is up to 2.4 times larger than what was asked for, and the
overshoot **grows** as the tolerance tightens.

Nothing is broken. The tolerance controls the **local** error of one step, and what is reported
here is the **global** error at the end, and no theorem connects them: local errors accumulate.
The controller is doing exactly what it was told and what it was told is not what was wanted.

The direction is worth noticing. Local extrapolation returns the more accurate of the two answers,
which lets the controller take **larger** steps for the same estimated local error, and the extra
accuracy per step is then spent on taking fewer of them rather than kept. Over a run those larger
steps accumulate more global error than the tolerance suggests.

A factor of 2.4 is not a disaster and is easily absorbed by asking for a tolerance one decade
tighter than you need. What would be a disaster is believing the tolerance is a bound.

## 8. Exercises

**Level 1, understanding**

1.1 Explain why an embedded pair's error estimate is free and step halving's is not, in
evaluations per step.

1.2 Derive the step control law from $e \approx Ch^{q+1}$ and say which $q$ belongs in it.

1.3 State what the first-same-as-last property is and what it saves.

1.4 Explain what local extrapolation is and what it costs.

1.5 Say why the tolerance in an adaptive solver is not a bound on the final error.

**Level 2, derivation**

2.1 Derive the number of function evaluations per step for step halving with a $p$ stage method,
including the shared first stage, and compare against an embedded pair.

2.2 Show that a first-same-as-last pair must have $\hat b_s = 0$ and $b_i = a_{si}$, and check
those conditions on Dormand-Prince.

2.3 Derive the growth and shrink caps' effect on the controller's stability, treating it as a
discrete feedback loop.

2.4 Show that the difference of two embedded solutions estimates the low order method's error to
leading order, and find the next term.

2.5 Design a PI controller for the step size and say what it fixes about the plain law.

**Level 3, computational**

3.1 Implement an embedded pair solver from scratch for any tableau with two bottom rows, and
verify it against `ad.solve`.

3.2 Implement step doubling error control for RK4 and compare its cost against Dormand-Prince at
matched accuracy.

3.3 Implement a dense output formula for Dormand-Prince and use it to produce output at times the
solver did not step to.

3.4 Implement event detection: stop the integration when a scalar function of the state crosses
zero, using the dense output of 3.3 to find the crossing.

3.5 Implement the PI controller of exercise 2.5 and measure its rejection rate against the plain
controller on a problem with a sudden change.

**Level 4, experimental**

4.1 Measure the rejection rate against the safety factor over the range 0.5 to 1.0 and find the
value that minimises total evaluations.

4.2 Measure the ratio of achieved error to tolerance over eight decades of tolerance, with and
without local extrapolation, and describe the two shapes.

4.3 Measure the step size an adaptive solver chooses against the local Lipschitz constant along
the trajectory, and see how well they track.

**Level 5, advanced**

5.1 **Why the estimate is for the wrong method.** The estimate describes the low order answer and
the solver returns the high order one. Construct a problem on which that mismatch makes the
controller behave badly, and say what property of the problem causes it.

5.2 **Order reduction.** An embedded pair applied to a stiff problem can achieve a lower order
than it claims. Explain the mechanism and construct an example.

5.3 **Stiffness detection.** Some solvers estimate the dominant eigenvalue from the stages they
already computed and switch method when the step is stability limited. Derive an estimate from
Dormand-Prince's stages.

## 9. Key takeaways

- **An embedded pair gets an error estimate for nothing**, because the two answers share their
  stages. Step halving gets the same estimate for about three times the evaluations.

- **The control law is $h(\text{tol}/e)^{1/(q+1)}$**, with safety 0.9, growth capped at 5 and
  shrink floored at 0.2. All three modifications exist because $e$ is an estimate.

- **The estimate describes the low order answer.** Against the low order error the ratio is near
  1; against the high order error it is orders of magnitude out and getting worse.

- **Local extrapolation is free accuracy and it costs the meaning of the tolerance.** It is
  4 to 25 times more accurate **and** uses fewer evaluations, because turning it off also gives up
  first-same-as-last: 6.39 evaluations per accepted step against 7.39. The honest solver is worse
  on both axes, which is why nobody ships it.

- **Adaptivity buys the ratio between the largest and smallest step** a problem needs: 159 on
  the pulse problem here. It also costs 18 rejected steps out of 86, because a controller that
  looks only at the last step cannot see a pulse coming.

- **Adaptivity is not always the winner.** Against a uniform grid at matched evaluations it wins
  by 1700 times at a tolerance of $10^{-4}$ and **loses by 160 times** at $10^{-10}$, because a
  grid fine enough for the hard part is more than fine enough for the rest, and high order turns
  that surplus into free accuracy.

- **The tolerance is local and the error is global, and here the error is larger.** The ratio
  runs from 0.59 to 2.41 and grows as the tolerance tightens. The tolerance is a step size
  control parameter, not a bound, and the overshoot goes the wrong way.

## Where this goes next

Lesson 71 gets high order from **past** steps rather than extra stages, at one evaluation each,
and pays for it with a new kind of instability that has nothing to do with accuracy. Lesson 72
explains why the solver in section 6 had to take tiny steps long after the transient was over,
which is the question adaptivity cannot answer.
