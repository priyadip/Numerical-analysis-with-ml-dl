# 89. Stochastic Optimization

**Part 12: Numerical Optimization**

## Learning objectives

By the end of this lesson you will be able to:

1. Say why a constant step size does not converge, and predict the size of what it converges to
   instead.
2. State the Robbins-Monro conditions and give a failing case for each.
3. Predict how gradient noise depends on the batch size, and price a batch size decision.
4. Explain what momentum buys, in the same language lesson 86 used for the condition number.
5. Say what the adaptive methods actually do, where they help, and what AdamW decouples.

## Prerequisites

Lesson 86 (the condition number as a rate, and the line searches that stop being usable here).
Lesson 87 (BFGS, whose curvature pairs become noise when the gradient is sampled). Lesson 24
(conjugate gradients and its $\sqrt\kappa$, which momentum reaches by a different route). Lesson 84
(the $\sqrt\varepsilon$ barrier, which the noise floor here dwarfs).

---

## 1. What changes when the gradient is a sample

Everything in lessons 84 to 88 assumed $\nabla f$ could be computed. Now the objective is a sum,

$$
f(x) = \frac{1}{N}\sum_{i=1}^{N} f_i(x) ,
$$

with $N$ large enough that one full gradient is unaffordable, and a step uses a **batch** of size
$b \ll N$. The sampled gradient is unbiased, $\mathbb{E}[g_b] = \nabla f$, and that is all that
survives.

Three things break at once.

**The line searches of lesson 86 are unusable.** Armijo compares $f(x + \alpha p)$ against
$f(x)$, and neither is available: evaluating $f$ costs $N$ terms too. Comparing sampled values
compares two random numbers.

**The curvature pairs of lesson 87 are noise.** BFGS needs $y = g_{k+1} - g_k$ to be a difference of
gradients at two points. A difference of two *sampled* gradients at two points is dominated by the
sampling, not by the curvature, unless the batches are enormous.

**A descent direction is no longer guaranteed.** $-g_b$ points downhill only on average.

So the theory restarts, and its first statement is the one this lesson is built around.

---

## 2. A constant step does not converge

Near the answer the true gradient is small and the sampling noise is not, so a step is mostly noise
and the iterate random walks. The walk balances against the pull of the gradient at a radius where
the two are comparable, and the balance gives a **noise ball** whose squared radius is proportional
to the step.

```python
from nalib import stochastic as st

out = st.the_noise_ball_grows_like_the_square_root_of_the_step()
print(f"{'step':>10}{'tail distance':>16}{'over sqrt(step)':>18}")
for row in out["rows"]:
    print(f"{row['step']:>10.0e}{row['tail_distance']:>16.4e}"
          f"{row['over_root_step']:>18.5f}")
print(f"\nfitted power of the step: {out['fitted_power_of_the_step']:.4f}  "
      f"(the prediction is 0.5)")
print(f"spread in the constant: {out['spread_in_the_constant']:.4f}")
print(out["note"])

assert out["the_power_is_a_half"]
```

The fitted power is $0.484$ against a predicted $0.5$, and the constant $0.41$ moves by six per cent
across three decades of step size.

That exponent is the whole economics of the method. **Halving the step buys a factor of $1.41$ in
accuracy and doubles the number of steps needed to get anywhere.** There is no step small enough to
converge and no step large enough to be fast, which is why the step has to change as the run
proceeds.

---

## 3. The Robbins-Monro conditions

A step sequence $a_k$ gives convergence when

$$
\sum_k a_k = \infty \qquad\text{and}\qquad \sum_k a_k^{2} < \infty .
$$

The first says the steps can still travel any distance: without it the iterate runs out of movement
before it arrives. The second says the noise they admit is summable: without it the iterate keeps
being kicked and never settles.

Four schedules, and each condition has a failing case.

```python
out = st.the_robbins_monro_conditions_decide_convergence()
print(f"{'schedule':>24}{'base':>7}{'sum a':>10}{'sum a^2':>11}{'final distance':>17}")
for row in out["rows"]:
    print(f"{row['schedule']:>24}{row['base']:>7g}{row['sum_of_steps']:>10.2f}"
          f"{row['sum_of_squares']:>11.4f}{row['final_distance']:>17.4e}")
print(f"\nbest: {out['best_schedule']}, worst: {out['worst_schedule']}, "
      f"a factor of {out['spread']:.0f} between them")
print(out["note"])

assert out["the_best_satisfies_both"]
assert out["the_worst_decays_too_fast"]
```

- **Constant $0.01$**: $\sum a = 400$ and $\sum a^{2} = 4$, both growing without bound. Fails the
  second condition and stalls at $3.5\times10^{-2}$, which is section 2's noise ball.
- **$0.5/k$**: $\sum a = 5.6$ and still growing like $\log k$, $\sum a^{2} = 0.41$ and converging.
  Satisfies both, reaches $1.7\times10^{-3}$, best of the four.
- **$0.1/\sqrt k$**: $\sum a^{2}$ grows like $\log k$, so it fails the second condition, slowly.
  Reaches $8.5\times10^{-3}$, five times worse than $1/k$ and still usable.
- **$0.5/k^{1.5}$**: $\sum a = 1.30$ and **converging**. It fails the *first* condition, and it is
  the worst of the four by a factor of $176$: it stops at $3.0\times10^{-1}$ and would stop there
  however long it ran.

That last row is the one worth remembering. A faster decaying schedule looks more careful and is
in fact **broken**: the steps add up to a finite total distance, and if the answer is further away
than that total, the iterate never reaches it. Both conditions are load bearing and they fail in
opposite directions.

---

## 4. The batch size, priced

The only quantity in this lesson that has nothing to do with any algorithm is the noise itself.

```python
out = st.the_noise_falls_like_one_over_root_batch()
print(f"{'batch':>8}{'noise':>12}{'noise times sqrt(batch)':>26}")
for row in out["rows"]:
    print(f"{row['batch']:>8}{row['noise']:>12.5f}{row['times_root_batch']:>26.5f}")
print(f"\nfitted power of the batch: {out['fitted_power_of_the_batch']:.4f}  "
      f"(the prediction is -0.5)")
print(f"the constant above batch four moves by "
      f"{100 * (out['constant_above_four'] - 1):.1f} per cent")
print(out["note"])

assert out["the_power_is_minus_a_half"]
```

The fitted power is $-0.484$, and the noise times $\sqrt b$ settles at $5.08, 5.11, 5.12, 5.10$
from $b = 16$ upward.

So the price is fixed: **sixteen times the work buys four times the accuracy.** A batch of $1024$
costs $64$ times a batch of $16$ and reduces the noise by $8$. That exponent is why batch sizes in
practice are chosen by what the hardware does efficiently rather than by this trade, since the
trade itself never improves.

---

## 5. Momentum

Momentum was invented for the deterministic problem, so this section measures it there. Separating
the two is the only way to see what momentum contributes and what the noise contributes.

$$
v_{k+1} = \beta v_k - \alpha\, g_k, \qquad x_{k+1} = x_k + v_{k+1} .
$$

On a quadratic the optimal parameters are known in closed form and give a rate of
$(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$, against plain descent's $(\kappa - 1)/(\kappa + 1)$. **The
same square root conjugate gradients gets in lesson 24**, from a method with one extra vector of
storage and no line search.

```python
out = st.momentum_turns_the_condition_number_into_its_square_root()
print(f"{'kappa':>9}{'plain steps':>13}{'plain rate':>13}{'(k-1)/(k+1)':>14}"
      f"{'heavy steps':>13}{'heavy rate':>13}{'sqrt form':>12}{'speedup':>10}")
for row in out["rows"]:
    print(f"{row['condition']:>9g}{row['plain_steps']:>13}{row['plain_rate']:>13.6f}"
          f"{row['plain_bound']:>14.6f}{row['heavy_steps']:>13}"
          f"{row['heavy_rate']:>13.6f}{row['heavy_bound']:>12.6f}"
          f"{row['speedup']:>10.2f}")
print(f"\nbiggest speedup: {out['biggest_speedup']:.1f} times")
print(out["note"])

assert out["plain_matches_its_bound"]
assert out["heavy_matches_its_bound"]
```

Plain descent matches its closed form to six digits and heavy ball matches its own to within one per
cent. At $\kappa = 1000$ that is $13662$ steps against $493$, a speedup of $27.7$, and the speedup
grows with $\kappa$ exactly as $\sqrt\kappa/\kappa$ says it should.

### 5.1 Acceleration, measured honestly

For functions that are convex but not strongly convex, Nesterov's method is the accelerated one and
the textbook statement is that gradient descent gives $O(1/k)$ and Nesterov $O(1/k^{2})$.

Measuring that needs a function where the finite dimension cannot help, which is Nesterov's worst
function with $n$ far larger than the step budget.

```python
out = st.acceleration_doubles_the_exponent()
print(f"{out['variables']} variables, {out['budget']} steps")
print(f"{'method':>20}{'exponent':>12}{'gap at 100':>14}{'gap at the end':>17}")
for row in out["rows"]:
    print(f"{row['method']:>20}{row['exponent']:>12.4f}{row['gap_at_100']:>14.4e}"
          f"{row['gap_at_the_end']:>17.4e}")
print(f"\nratio of exponents: {out['exponent_ratio']:.4f}")
print(f"accuracy gain at the end: {out['accuracy_gain_at_the_end']:.2f} times")
print(out["note"])

assert out["the_ratio_is_two"]
assert out["neither_matches_the_quoted_bound"]
```

The measured exponents are $-0.501$ and $-1.020$. **Neither is the $-1$ and $-2$ that are quoted**,
and the ratio is $2.04$.

That gap is not an error in the measurement or in the theory. The quoted values are worst case
**upper bounds over a class of functions**; this is a measurement on one instance. A single instance
can be easier than the worst case, and here both methods are. Comparing the two directly would be
comparing different statements, so what is reported is the part that does transfer: **acceleration
doubles the exponent**, whatever the exponent happens to be.

---

## 6. The adaptive methods

AdaGrad, RMSProp and Adam all do the same thing: accumulate the **squared** gradient per coordinate
and divide by its square root. That is a diagonal preconditioner built from the data.

$$
s_{k+1} = \rho\,s_k + (1-\rho)\,g_k^{2}, \qquad
x_{k+1} = x_k - \frac{\alpha}{\sqrt{s_{k+1}} + \epsilon}\,g_k .
$$

AdaGrad uses $\rho = 1$, so the accumulation never forgets and the step decays to zero on its own.
RMSProp uses $\rho < 1$, so it forgets. Adam adds momentum on top and corrects both accumulators for
their zero initialization.

The honest way to measure them is to be careful about the step size, because handing a plain method
a step chosen for a different problem would be measuring the step and calling it the method.

```python
out = st.the_adaptive_methods_rescale_the_coordinates()
print(f"{'scaling':>9}{'condition':>13}{'method':>10}{'step rule':>20}{'step':>11}"
      f"{'final distance':>17}{'diverged':>11}")
for row in out["rows"]:
    shown = "inf" if row["diverged"] else f"{row['final_distance']:.4e}"
    print(f"{row['scaling']:>9g}{row['condition']:>13.4g}{row['method']:>10}"
          f"{row['step_rule']:>20}{row['step']:>11.3g}{shown:>17}"
          f"{str(row['diverged']):>11}")
print(f"\nbest when well scaled: {out['best_when_well_scaled']}, "
      f"badly scaled: {out['best_when_badly_scaled']}")
print(f"methods that diverged: {out['methods_that_diverged']}")
print(f"margin over the best plain method: "
      f"{out['margin_when_well_scaled']:.2f} well scaled, "
      f"{out['margin_when_badly_scaled']:.1f} badly scaled")
print(out["note"])

assert out["everything_that_diverged_was_non_adaptive"]
assert out["adaptive_wins_when_badly_scaled"]
```

Three readings.

**With a step chosen blind, the plain methods blow up and the adaptive ones do not.** At a condition
number of $10^{6}$, SGD and momentum with the same $0.01$ that worked on the easy problem both
diverge to infinity. Every adaptive method converges with its usual $0.05$, unchanged. **That is the
practical reason these methods are used**: not that they are faster, but that they remove a choice
that is easy to get catastrophically wrong.

**Given the best step a plain method can have, it works and is still beaten.** With $\alpha = 1/L$
from the measured smoothness, SGD reaches $1.83$ and momentum $1.44$, against Adam's $0.027$: a
margin of $54$.

**The margin is what changes, not the ranking.** On the well scaled problem the best adaptive method
beats the best plain one by $2.5$, on the badly scaled one by $54$. An adaptive method is a diagonal
preconditioner, so it helps exactly where a diagonal preconditioner would, and the amount it helps
is how badly the coordinates were scaled.

### 6.1 Adam and AdamW

The difference is one line, and it is worth being exact about.

**Adam with L2 regularization** adds $\lambda x$ to the gradient. That sum then passes through the
per coordinate rescaling, so a coordinate with a large gradient history is divided by a large number
and receives **less** decay.

**AdamW** subtracts $\alpha\lambda x$ from the iterate directly, outside the rescaling, so every
coordinate decays at the same rate whatever its history.

They coincide exactly when the rescaling is uniform, which is exactly when an adaptive method was
not needed.

```python
out = st.adam_and_adamw_differ()
print(f"condition number {out['condition']:.4g}")
print(f"{'method':>9}{'weight decay':>15}{'gap':>12}{'norm of x':>13}"
      f"{'distance':>12}")
for row in out["rows"]:
    print(f"{row['method']:>9}{row['weight_decay']:>15g}{row['gap']:>12.5f}"
          f"{row['norm_of_the_answer']:>13.6f}{row['final_distance']:>12.5f}")
print(f"\nwith no decay the two answers differ by "
      f"{out['difference_without_decay']:.1e}")
print(f"with decay they are {out['distance_with_decay']:.4e} apart")
print(f"largest coordinate difference {out['largest_coordinate_difference']:.3e}, "
      f"smallest {out['smallest_coordinate_difference']:.3e}")
print(out["note"])

assert out["identical_without_decay"]
assert out["the_difference_is_uneven_across_coordinates"]
```

**With no decay the difference is exactly zero**, bit for bit, which is the check that they are the
same algorithm when the extra term is switched off. A nonzero difference there would be a bug.

**With decay they are $3.2\times10^{-2}$ apart**, and the difference is spread unevenly: the largest
coordinate moves by $2.7\times10^{-2}$ and the smallest by $4.5\times10^{-4}$, a factor of $61$.
That unevenness is the entire content of the decoupling. It is not that one is better in general, it
is that under Adam the amount of regularization a parameter receives depends on its gradient history,
which is almost never what was intended.

---

### 6.2 All seven updates, from scratch

The seven methods in this lesson differ in about ten lines between them, and writing them side by
side makes that concrete. Each one accumulates something and uses it; the table below is the whole
difference.

```python
import numpy as np


def one_step(method, x, slope, state, alpha, k, beta=0.9, rho=0.999,
             decay=0.0, tiny=1e-8):
    """One update of any of the seven, sharing the same state dictionary."""
    velocity = state.setdefault("velocity", np.zeros_like(x))
    squares = state.setdefault("squares", np.zeros_like(x))
    if decay and method != "adamw":
        slope = slope + decay * x
    if method == "sgd":
        return x - alpha * slope, state
    if method in ("momentum", "nesterov"):
        state["velocity"] = beta * velocity - alpha * slope
        return x + state["velocity"], state
    if method == "adagrad":
        state["squares"] = squares + slope * slope
        return x - alpha * slope / (np.sqrt(state["squares"]) + tiny), state
    if method == "rmsprop":
        state["squares"] = rho * squares + (1.0 - rho) * slope * slope
        return x - alpha * slope / (np.sqrt(state["squares"]) + tiny), state
    state["velocity"] = beta * velocity + (1.0 - beta) * slope
    state["squares"] = rho * squares + (1.0 - rho) * slope * slope
    first = state["velocity"] / (1.0 - beta ** (k + 1))
    second = state["squares"] / (1.0 - rho ** (k + 1))
    move = alpha * first / (np.sqrt(second) + tiny)
    if method == "adamw" and decay:
        move = move + alpha * decay * x
    return x - move, state


problem = st.least_squares_problem(samples=800, variables=6)
budget = 2000
rng = np.random.default_rng(0)
print(f"{'method':>10}{'mine':>14}{'the library':>14}{'gap':>12}")
for method in ("sgd", "momentum", "adagrad", "rmsprop", "adam", "adamw"):
    x = problem["start"].copy()
    state = {}
    picker = np.random.default_rng(0)
    for k in range(budget):
        rows = picker.integers(0, problem["samples"], size=16)
        slope = problem["batch_gradient"](x, rows)
        x, state = one_step(method, x, slope, state, 0.02, k)
    theirs = st.stochastic_descent(problem, method=method, step=0.02, batch=16,
                                   rounds=budget, seed=0)
    mine = float(np.linalg.norm(x - problem["minimizer"]))
    print(f"{method:>10}{mine:>14.6f}{theirs['final_distance']:>14.6f}"
          f"{abs(mine - theirs['final_distance']):>12.2e}")
```

Every method reproduces the library's answer to the last digit, from the same seed, which is the
check that the ten lines above really are the whole difference between them.

## 7. Choosing a schedule

```python
out = st.which_schedule()
print(f"{'schedule':>22}{'gap':>14}{'final distance':>17}")
for row in out["rows"]:
    print(f"{row['schedule']:>22}{row['gap']:>14.4e}"
          f"{row['final_distance']:>17.4e}")
print(f"\nbest: {out['best']}, and it beats the constant step by "
      f"{out['best_over_constant']:.1f} times")
print(out["note"])

assert out["constant_is_worst"]
```

At a fixed budget of twenty thousand steps, every decaying schedule beats the constant one, by
between $6$ and $28$ times. The differences **between** the decaying schedules are much smaller than
the difference between decaying and not, which is the practically useful shape of the result: the
first decision matters and the second one much less.

Cosine decay and cosine with warmup land within three per cent of each other here. Warmup exists to
protect the first few steps of a run whose curvature is not yet known, and this problem's curvature
is fixed and mild, so there is nothing for it to protect against. **A technique that does nothing on
a well behaved problem is not thereby useless**, it is being tested on the wrong problem.

```python
import matplotlib.pyplot as plt
import numpy as np

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.0))

problem = st.least_squares_problem()
for name, rule, style in (
        ("constant 0.05", st.schedule("constant", 0.05), "-"),
        ("0.5 / k", st.schedule("one over k", 0.5), "--"),
        ("0.05 / sqrt(k)", st.schedule("one over root k", 0.05), ":"),
        ("cosine", st.cosine_schedule(0.05, 20000), "-.")):
    run = st.stochastic_descent(problem, method="sgd", batch=16, rounds=20000,
                                rule=rule)
    left.loglog(np.maximum(run["steps"], 1), run["distances"], style, lw=1.5,
                label=name)
left.set_xlabel("steps")
left.set_ylabel("distance to the answer")
left.set_title("a constant step stops improving")
left.legend(fontsize=8)
left.grid(alpha=0.3, which="both")

table = st.the_noise_ball_grows_like_the_square_root_of_the_step()
sizes = np.array([r["step"] for r in table["rows"]])
radii = np.array([r["tail_distance"] for r in table["rows"]])
right.loglog(sizes, radii, "o", ms=7, label="measured noise ball")
fine = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 50)
right.loglog(fine, np.mean(radii / np.sqrt(sizes)) * np.sqrt(fine), "-", lw=1.4,
             label="a constant times sqrt(step)")
right.set_xlabel("step size")
right.set_ylabel("radius of the noise ball")
right.set_title(f"fitted power {table['fitted_power_of_the_step']:.3f}")
right.legend(fontsize=8)
right.grid(alpha=0.3, which="both")

fig.tight_layout(); fig.savefig("../figures/89_stochastic.png", dpi=110); plt.close(fig)
print("saved ../figures/89_stochastic.png")
```

![Schedules against a constant step, and the noise ball](../figures/89_stochastic.png)

---

## 8. Exercises

**Level 1, understanding**

1.1 Explain why a constant step size does not converge and say what it converges to.

1.2 State the Robbins-Monro conditions and give a schedule failing each.

1.3 Say how gradient noise depends on the batch size and price a batch size decision.

1.4 Explain what the adaptive methods do to the coordinates.

1.5 Say what AdamW decouples and when it makes no difference.

**Level 2, derivation**

2.1 Derive the noise ball radius $\sqrt{\alpha\sigma^{2}/\mu}$ for a strongly convex quadratic.

2.2 Show that the batch gradient is unbiased and that its variance is $\sigma^{2}/b$.

2.3 Derive the optimal heavy ball parameters for a quadratic and the rate they give.

2.4 Show that AdaGrad's effective step decays like $1/\sqrt k$ without any schedule.

2.5 Show that Adam with L2 regularization and AdamW coincide when the second moment is constant
across coordinates.

**Level 3, computational**

3.1 Implement SVRG or SAGA, which use stored gradients to remove the noise, and measure whether the
convergence becomes linear.

3.2 Implement gradient clipping and measure how it changes the divergence threshold of section 6.

3.3 Implement Polyak averaging and measure whether it recovers the $1/k$ rate from a $1/\sqrt k$
schedule.

3.4 Implement a stochastic line search using a fixed batch across the trial steps, and say why the
batch has to be fixed.

3.5 Implement AdaBelief or Lion and compare against Adam on both scalings of section 6.

**Level 4, experimental**

4.1 Measure the noise ball radius against the batch size as well as the step, and fit both
exponents at once.

4.2 Measure the largest stable step for SGD, momentum and Adam against the condition number, over
four decades.

4.3 Measure how the best schedule changes with the budget, from a hundred steps to a million.

**Level 5, advanced**

5.1 **Why the noise floor beats the rounding floor.** Compare the $\sqrt\varepsilon$ barrier of
lesson 84 against the noise ball of section 2 and say which binds, at what batch size.

5.2 **Momentum under noise.** Momentum's $\sqrt\kappa$ rate is a deterministic result. Measure what
it becomes with a sampled gradient and explain the difference.

5.3 **Why adaptive methods are hard to analyse.** Identify what breaks in the convergence proof when
the step size depends on the gradient history, and describe how the published fixes get around it.

## 9. Key takeaways

- **A constant step does not converge.** It settles into a noise ball whose radius grows like
  $\sqrt{\text{step}}$, fitted at $0.484$ against $0.5$.

- **Both Robbins-Monro conditions bite.** A constant step fails $\sum a^{2} < \infty$ and stalls;
  $0.5/k^{1.5}$ fails $\sum a = \infty$ and is $176$ times worse, because its steps add up to a
  finite distance.

- **Sixteen times the work buys four times the accuracy.** The noise falls like $b^{-0.484}$ and
  the constant is fixed to five per cent above $b = 16$.

- **Momentum turns $\kappa$ into $\sqrt\kappa$**, matching the closed form to one per cent, a
  speedup of $27.7$ at $\kappa = 1000$.

- **Acceleration doubles the decay exponent**, measured at $-0.501$ and $-1.020$. Neither is the
  quoted $-1$ and $-2$, because a measurement on one instance is not a worst case over a class.

- **The adaptive methods remove a choice rather than add speed.** With a step chosen blind the
  plain methods diverge on a badly scaled problem and every adaptive method converges.

- **The margin is what the scaling changes**, from $2.5$ to $54$ times, because an adaptive method
  is a diagonal preconditioner.

- **Adam and AdamW are bit identical without decay** and $3.2\times10^{-2}$ apart with it, unevenly
  spread across coordinates by a factor of $61$.

- **Any decaying schedule beats a constant one** by $6$ to $28$ times at a fixed budget, and the
  decaying ones differ from each other far less than that.

## Where this goes next

Lesson 90 changes the objective rather than the gradient. A great many problems are
`smooth + nonsmooth`, and nothing in Part 12 so far can handle one: subgradients converge at
$O(1/\sqrt k)$ and lose everything the last six lessons built. The proximal operator recovers it,
and lesson 88's projection turns out to be its first example.
