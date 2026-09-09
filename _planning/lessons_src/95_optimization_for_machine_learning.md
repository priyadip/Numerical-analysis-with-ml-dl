# 95. Optimization for Machine Learning

**Part 14: Numerical Analysis in Machine Learning and AI**

## Learning objectives

By the end of this lesson you will be able to:

1. Derive the learning rate limit from Part 10's absolute stability condition rather than from a
   rule of thumb.
2. Say what a minibatch gradient's noise is, write down its variance, and predict how a training run
   changes when the batch size does.
3. Separate the two reasons a run leaves a sharp minimum, and say which one is doing the work.
4. Give the real reason second order methods are rare, and give the measurement that rules out the
   usual one.
5. State what the natural gradient is invariant to, and why that is the argument for preconditioning.

## Prerequisites

Lesson 86 (steepest descent and the condition number). Lesson 89 (stochastic gradient descent,
momentum, the adaptive methods). Lesson 67 and lesson 74 (Euler's method and its absolute stability
region, which turn out to be this lesson's subject). Lesson 87 (quasi-Newton). Lesson 92 (sampling
error).

---

## 1. The same optimizers, a different problem

Part 12 minimized functions. A training loss is a function, so nothing there stops being true. What
changes is the setting, in exactly four ways, and each one turns a Part 12 result into a machine
learning result you have probably heard stated without its derivation.

| What is said there | What it is here | Lesson |
|---|---|---|
| "The learning rate was too high" | $h > 2/L$, the absolute stability limit of Euler's method | 74 |
| "Gradient flow" | The ODE that gradient descent is the explicit Euler discretization of | 67 |
| "Momentum smooths the trajectory" | A damped second order ODE, with two parameters that are $h$ and the damping | 73 |
| "Batch normalization helps optimization" | Preconditioning, which changes $\kappa$ and therefore the iteration count | 86 |
| "Gradient noise" | The sampling error of lesson 92, with a finite population correction | 92 |
| "Second order methods are too expensive" | Not the reason, as section 11 measures | 87 |
| "Natural gradient" | Preconditioning by the Fisher matrix, exactly invariant under reparametrization | 30 |

## 2. Gradient descent is Euler's method

The **gradient flow** is the differential equation

$$
\frac{dx}{dt} = -\nabla f(x) .
$$

Apply Euler's method from lesson 67 with step $h$ and you get $x_{k+1} = x_k - h\nabla f(x_k)$,
which is gradient descent with learning rate $h$. This is an identity, not an analogy, and it means
the trajectory error obeys lesson 67's first order bound.

```python
from nalib import mlopt

out = mlopt.gradient_descent_is_explicit_euler()
print("distance from the gradient flow at the same final time")
print(f"{'steps':>8}{'step size':>13}{'error':>14}{'ratio':>10}")
previous = None
for row in out["rows"]:
    ratio = "" if previous is None else f"{previous / row['error']:.4f}"
    print(f"{row['steps']:>8}{row['step_size']:>13.6f}{row['error']:>14.6e}{ratio:>10}")
    previous = row["error"]
print(f"\nfitted order: {out['order']:.4f}")

assert out["it_is_first_order"]
```

**The fitted order is $0.9984$**, and every halving of the step halves the error. Gradient descent
is a first order ODE solver, and the "path" it takes to the minimum is a numerical approximation of
a continuous path that exists whether or not anyone discretizes it.

That reframing is worth having, because it says where to look for everything else. Anything true of
Euler's method is true of gradient descent, and the first thing that is true of Euler's method is a
stability limit.

## 3. The learning rate limit is the stability limit

Near a minimum, $\nabla f(x) \approx H(x - x^\star)$, so the flow is linear with eigenvalues $-L_i$,
the negative curvatures. Lesson 74 says Euler's method is absolutely stable when
$|1 + h\lambda| \le 1$, which here is

$$
|1 - h L_i| \le 1 \quad\text{for every } i,
\qquad\text{that is}\qquad
h \le \frac{2}{L_{\max}} .
$$

```python
from nalib import mlopt

out = mlopt.the_step_limit_is_the_stability_limit()
print(f"largest curvature {out['largest_curvature']:g}, so the predicted limit is "
      f"{out['predicted']:.6f}")
print(f"bisected limit {out['measured']:.12f}, a relative gap of {out['relative_gap']:.2e}")
print(f"\n{'h / limit':>12}{'step':>12}{'amplification':>17}{'norm after 200 steps':>24}")
for row in out["rows"]:
    print(f"{row['factor']:>12.2f}{row['step']:>12.6f}{row['amplification']:>17.6f}"
          f"{row['final_norm']:>24.4e}")

assert out["the_limit_is_two_over_l"]
```

**The bisected limit agrees with $2/L$ to $1.8\times10^{-12}$.** The amplification column crosses
$1$ exactly at the limit, and one per cent past it the run has already grown by a factor of $16$
in 200 steps.

Two things follow that are usually stated separately:

- **The largest usable learning rate is set by the largest curvature, not by the average one.** One
  stiff direction in a million caps the step for all of them. That is lesson 74's stiffness, and
  it is the reason a learning rate has to be found by search rather than by reasoning about the
  typical scale of the loss.

- **"Diverged" and "too big a step" are the same event.** There is nothing to diagnose beyond
  $hL_{\max} > 2$.

## 4. Momentum is a second order equation

Heavy ball momentum is

$$
x_{k+1} = x_k - \alpha\nabla f(x_k) + \beta(x_k - x_{k-1}) ,
$$

which is the central difference discretization of

$$
\ddot x + a\dot x + \nabla f(x) = 0 ,
\qquad
\alpha = h^2, \quad \beta = 1 - a h .
$$

So $\alpha$ and $\beta$ are not two independent knobs. They are a step size and a damping constant,
and they have to move together: halving $h$ divides $\alpha$ by $4$ and moves $\beta$ towards $1$.

```python
from nalib import mlopt

out = mlopt.momentum_is_a_second_order_ode()
print(f"damping a = {out['damping']:g}, and alpha = h**2, beta = 1 - a h")
print(f"{'steps':>8}{'h':>12}{'alpha':>14}{'beta':>12}{'error':>14}")
for row in out["rows"]:
    print(f"{row['steps']:>8}{row['h']:>12.6f}{row['step_size']:>14.3e}"
          f"{row['momentum']:>12.6f}{row['error']:>14.6e}")
print(f"\nfitted order: {out['order']:.4f}")

assert out["it_tracks_the_ode"]
```

The fitted order is $0.9995$. **The reason $\beta = 0.9$ is a common default is that it is
$1 - ah$ for a small $h$**, and the reason tuning $\alpha$ and $\beta$ separately is awkward is that
they are two views of one pair of numbers.

## 5. Conditioning is the iteration count

Lesson 86 gave the rate of steepest descent on a quadratic with the optimal step: the error
contracts by $(\kappa-1)/(\kappa+1)$ each iteration, so reaching a tolerance takes about
$\tfrac{\kappa}{2}\log(1/\text{tol})$ iterations.

```python
from nalib import mlopt

out = mlopt.conditioning_decides_the_iteration_count()
print(f"{'condition':>12}{'iterations':>13}{'predicted':>13}{'ratio':>9}"
      f"{'preconditioned':>17}")
for row in out["rows"]:
    print(f"{row['condition']:>12.0f}{row['iterations']:>13}{row['predicted']:>13.0f}"
          f"{row['iterations'] / row['predicted']:>9.4f}{row['whitened_iterations']:>17}")
print(f"\nworst ratio to the prediction: {out['worst_ratio']:.4f}")
print(f"largest saving from preconditioning: {out['largest_saving']:.0f} to 1")

assert out["the_prediction_holds"]
assert out["whitening_removes_it"]
```

**The prediction holds to a factor of $1.084$ over four decades**, and preconditioning by
$H^{-1}$ takes every one of them to a single iteration.

That last column is the whole argument for feature scaling, batch normalization, layer
normalization and every other trick that reshapes the loss surface. **None of them make the gradient
better. They change $\kappa$, and $\kappa$ is the iteration count.** The saving here is a factor of
$88556$.

## 6. The gradient is an estimate

A minibatch gradient averages $B$ of the $N$ per-sample gradients. Sampling **without replacement**,
which is what shuffling an epoch does, gives

$$
\operatorname{Var}\left[\hat g_B\right]
= \frac{\sigma^2}{B}\left(1 - \frac{B-1}{N-1}\right) ,
$$

with $\sigma^2$ the variance across single samples. The bracket is the finite population correction,
and it matters here in a way it does not in lesson 92: it is exactly $0$ when $B = N$, so the full
batch gradient is not merely accurate, it is exact.

```python
from nalib import mlopt

out = mlopt.the_gradient_noise_falls_like_the_batch_size()
print(f"per sample gradient variance {out['population_variance']:.6f}")
print(f"{'batch':>8}{'measured':>15}{'predicted':>15}{'ratio':>10}")
for row in out["rows"]:
    ratio = "-" if row["predicted"] == 0.0 else f"{row['ratio']:.4f}"
    print(f"{row['batch']:>8}{row['measured']:>15.6e}{row['predicted']:>15.6e}{ratio:>10}")
print(f"\nat the full batch the prediction is exactly 0 and the measurement is "
      f"{out['full_batch_variance']:.1e}")

assert out["the_formula_holds"]
assert out["the_full_batch_is_exact"]
```

The formula holds to $2.4$ per cent, and the full batch row is exact to $2\times10^{-33}$, which is
rounding and nothing else.

## 7. The linear scaling rule, and what ends it

The rule says: double the batch, double the learning rate. The argument is about noise. Doubling the
batch halves the gradient variance, so doubling the step restores the same amount of noise per unit
of progress.

That argument is silent about the deterministic part of the step, and the deterministic part has a
limit.

```python
from nalib import mlopt

out = mlopt.the_linear_scaling_rule_stops_at_the_stability_limit()
print(f"the stability limit is h = {out['limit']:.6f}")
print(f"{'batch':>8}{'step':>10}{'h / limit':>12}{'updates':>10}{'excess loss':>16}{'stable':>9}")
for row in out["rows"]:
    shown = "overflow" if not np.isfinite(row["excess_loss"]) else f"{row['excess_loss']:.4e}"
    print(f"{row['batch']:>8}{row['step']:>10.4f}{row['step_over_limit']:>12.4f}"
          f"{row['updates']:>10}{shown:>16}{str(row['stable']):>9}")

assert out["the_rule_holds_while_stable"]
assert out["it_breaks_at_the_stability_limit"]
```

**The rule holds for six doublings**, with the excess loss staying between $0.0071$ and $0.0098$
while the batch goes from $1$ to $64$. At $128$ the step it asks for is $1.28$ times the stability
limit and the loss is $1.7\times10^{9}$.

The failure is worth naming precisely. It is not that the noise argument became wrong. It is that
the noise argument was never the binding constraint, and at large batch sizes a different constraint,
one that does not mention batches at all, takes over. **Every published "the linear scaling rule
breaks down past batch size $B^\star$" is a measurement of $2/L$ divided by the base step.**

## 8. Sharp and flat minima: the step decides first

Take two minima of **exactly** equal depth and different curvature. A method that only reads the
loss has no reason to prefer either. Here is the first reason one gets preferred, and it is not
noise.

Section 3's condition is local, so it applies separately at each minimum with that minimum's own
curvature. A well of curvature $c$ is an attracting fixed point of gradient descent only while
$h < 2/c$.

```python
from nalib import mlopt

out = mlopt.a_large_step_cannot_sit_in_a_sharp_minimum()
print(f"curvatures: flat {out['flat_curvature']:.4f}, sharp {out['sharp_curvature']:.4f}, "
      f"a ratio of {out['curvature_ratio']:.4f}")
print(f"\n{'well':>8}{'2/c':>12}{'destabilizes':>15}{'gap':>12}{'ejects':>12}")
for name, entry in out["limits"].items():
    print(f"{name:>8}{entry['predicted']:>12.6f}{entry['destabilizes']:>15.6f}"
          f"{entry['gap']:>12.2e}{entry['ejects']:>12.6f}")
print(f"\n{'h / (2/c sharp)':>18}{'sharp swing':>15}{'flat swing':>14}"
      f"{'sharp stays':>14}{'flat stays':>13}")
for row in out["rows"]:
    print(f"{row['over_the_sharp_limit']:>18.2f}{row['sharp_swing']:>15.6f}"
          f"{row['flat_swing']:>14.6f}{str(row['sharp_stays']):>14}"
          f"{str(row['flat_stays']):>13}")

assert out["both_limits_are_two_over_the_curvature"]
assert out["there_is_a_window"]
assert out["ejection_comes_much_later"]
```

**Both limits are $2/c$**, to $0.023$ and $0.032$ per cent. So there is a window of step sizes,
here a factor of $8.5$ wide, in which the flat minimum is a fixed point and the sharp one is not.

The table also separates two things that are usually run together. **Losing a minimum and leaving it
are different thresholds.** Past $2/c$ the run stops converging and starts oscillating, with a swing
that grows from $0$ to $0.117$ and then to $0.575$; it does not leave the well until $3.25$ times
that, where the oscillation has finally grown past the barrier. The swing column reads $0$ again at
$4$ and $8$ times the limit only because the run has by then settled in the *other* well, which is
still stable at those steps. A run sitting at the edge of stability is a real state, and
it is neither converged nor diverged.

## 9. And then the batch size decides

Inside the window between those two thresholds, a noiseless run stays in the sharp well and an
oscillation is all that happens. That is exactly where the noise can act.

```python
from nalib import mlopt

out = mlopt.minibatch_noise_empties_the_sharp_well_first()
print(f"both wells have depth {out['flat_value']:.10f}, and their curvatures differ by "
      f"{out['curvature_ratio']:.2f}")
print(f"\nfraction of runs that left the sharp well")
header = f"{'h / (2/c)':>11}" + "".join(f"{key:>12}" for key in out["keys"])
print(header)
for row in out["rows"]:
    print(f"{row['factor']:>11.2f}" + "".join(f"{row[key]:>12.3f}" for key in out["keys"]))

assert out["the_wells_are_equally_deep"]
assert out["the_full_batch_never_leaves"]
assert out["noise_empties_it_where_the_full_batch_cannot"]
```

The two wells are equally deep to machine precision, so nothing in this table is a preference about
the loss.

**At $2.5$ times the limit a batch of $1$ leaves in $100$ per cent of runs and every larger batch in
$0$ per cent.** At $3.0$ times, batches up to $16$ all leave and $64$ and above do not. The full
batch never leaves at any step tested.

So the popular claim, that stochastic gradient descent finds flat minima, is true here, and it is
true in a narrower way than it is usually stated. It needs the step size to be inside a specific
window first. Below that window the noise does nothing at all: the first row of the table is zero
everywhere. **The step size opens the door and the batch size walks through it.**

## 10. What sharpness actually predicts

"Flat" and "sharp" are not vague. For a quadratic, a random perturbation of length $r$ raises the
loss by

$$
\mathbb{E}\left[f(x^\star + r u) - f(x^\star)\right] = \frac{r^2}{2}\,\frac{\operatorname{tr}H}{d} ,
\qquad u \text{ uniform on the unit sphere},
$$

while the **worst** direction raises it by $\tfrac12 r^2 L_{\max}$.

```python
from nalib import mlopt

out = mlopt.sharpness_predicts_the_damage()
print(f"{'condition':>11}{'radius':>10}{'measured rise':>16}{'predicted':>14}"
      f"{'ratio':>9}{'worst case':>13}")
for row in out["rows"]:
    print(f"{row['condition']:>11.0f}{row['radius']:>10.0e}{row['measured']:>16.6e}"
          f"{row['predicted']:>14.6e}{row['ratio']:>9.4f}{row['worst_case']:>13.4e}")
print(f"\nworst ratio to the prediction: {out['worst_ratio']:.4f}")
print(f"worst case over average, at the worst point: {out['worst_case_over_average']:.4f}")

assert out["the_prediction_is_exact"]
```

The prediction holds to $1.9$ per cent everywhere. Two consequences:

- **Average sharpness is the trace and worst-case sharpness is the largest eigenvalue.** On a
  well-conditioned problem they are the same number. At a condition number of $100$ they differ by
  $4.48$, so a sharpness measure that samples random directions and one that maximizes over them
  will rank the same two minima differently.

- The rise scales exactly as $r^2$, checked over three decades of radius. So any claim of the form
  "this minimum generalizes better because it is flatter" is a claim about $\operatorname{tr}H$,
  and it can be stated in those terms and measured.

## 11. Why second order methods stay rare

The usual explanation is cost. Let us check it.

```python
from nalib import mlopt

out = mlopt.cost_is_not_the_reason_second_order_is_rare()
print(f"{'d':>6}{'samples':>9}{'kappa':>9}{'gd iterations':>16}{'break even':>13}"
      f"{'gd flops':>13}{'newton flops':>15}{'ratio':>11}")
for row in out["rows"]:
    print(f"{row['dimension']:>6}{row['samples']:>9}{row['condition']:>9.0f}"
          f"{row['iterations']:>16}{row['break_even']:>13.1f}"
          f"{row['first_order_flops']:>13.3e}{row['second_order_flops']:>15.3e}"
          f"{row['ratio']:>11.1f}")
print(f"\nthe advantage falls like d to the {out['slope']:.4f}")
print(f"extrapolated crossover {out['extrapolated_crossover']:.0f} parameters, against a "
      f"gradient descent iteration count of {out['predicted_crossover']:.0f}")

assert out["newton_wins_everywhere_tested"]
assert out["the_crossover_is_the_iteration_count"]
```

**On flops Newton wins by $560$ at $5$ parameters and by $15.3$ at $200$**, and the advantage falls
like $1/d$. So on a deterministic problem the flop count is not the objection. The break-even
column says why: one Newton step costs about $d$ gradient steps, and gradient descent needs
$\tfrac{\kappa}{2}\log(1/\text{tol}) = 3405$ of them regardless of $d$. **The crossover is where the
dimension passes the iteration count**, extrapolated at $3148$ against a predicted $3405$.

Two real objections survive. One is memory: a Hessian for $10^9$ parameters is $8\times10^{18}$
bytes against $8\times10^{9}$ for a gradient, and no factorization fixes that. The other is the one
this lesson can measure.

```python
from nalib import mlopt

out = mlopt.a_noisy_hessian_makes_newton_worse()
print(f"20 parameters, and the excess loss over the optimum after 200 updates")
print(f"{'batch':>8}{'curvature rank':>17}{'singular':>11}{'sgd':>14}{'newton':>14}{'ratio':>12}")
for row in out["rows"]:
    newton = "diverged" if not np.isfinite(row["newton"]) else f"{row['newton']:.6e}"
    ratio = "-" if not np.isfinite(row["ratio"]) else f"{row['ratio']:.4f}"
    print(f"{row['batch']:>8}{row['curvature_rank']:>17}{str(row['singular']):>11}"
          f"{row['gradient_descent']:>14.6e}{newton:>14}{ratio:>12}")
print(f"\nthe two cross over at a batch of {out['crossover_batch']}")

assert out["the_curvature_is_singular_below_the_dimension"]
assert out["newton_is_worse_at_small_batches"]
```

**At a batch of $8$ the sampled curvature has rank $8$ for $20$ parameters, and the run diverges.**
That is not bad luck. A curvature estimate from $B$ samples has rank at most $B$, exactly as the
gradient in lesson 94 had rank at most $B$, so any batch smaller than the parameter count gives a
singular matrix. At $32$ it is invertible and Newton is still $6.43$ times worse than plain descent,
because the smallest eigenvalues of the estimate are noise and the step divides by them.

So the answer to "why not use curvature" is: **a Newton step inverts its estimate of the curvature,
and inverting a noisy near-singular matrix is the worst operation in this course.** Cost is a real
constraint at scale and it is not the one that decides. Everything that does get used, from Adam's
diagonal to L-BFGS's limited memory to trust regions, is a way of not inverting a noisy matrix.

## 12. What survives: preconditioning and Gauss-Newton

Two exact results survive the move to a stochastic setting, and both are worth having.

**The natural gradient is invariant under reparametrization.** Under $x = Ay$ the gradient becomes
$A^{\mathsf T}g$ and the Fisher matrix becomes $A^{\mathsf T}FA$, so the natural step is

$$
(A^{\mathsf T}FA)^{-1}A^{\mathsf T}g = A^{-1}F^{-1}g ,
$$

which maps back to the same point in the original coordinates. The invariance is exact, at every
step size.

```python
from nalib import mlopt

out = mlopt.the_natural_gradient_ignores_the_parametrization()
print("rescale the parameters, run both methods, map back, and compare against the unscaled run")
print(f"{'scale spread':>14}{'plain drift':>15}{'natural drift':>17}"
      f"{'plain error':>14}{'natural error':>16}")
for row in out["rows"]:
    print(f"{row['factor']:>14.0f}{row['plain_drift']:>15.6f}{row['natural_drift']:>17.2e}"
          f"{row['plain_error']:>14.6f}{row['natural_error']:>16.2e}")

assert out["natural_is_invariant"]
assert out["plain_is_not"]
```

**The natural gradient's trajectory does not move, to $2.8\times10^{-16}$**, while plain gradient
descent's moves by $0.47$ of the distance to the solution. That is the argument for preconditioning
stated as a property rather than as a speedup: an unpreconditioned method is answering a question
about the coordinates someone happened to choose.

**Gauss-Newton is the Fisher information.** For $y = m(\theta) + \mathcal N(0, I)$ the Fisher matrix
is $J^{\mathsf T}J$, which is also the Gauss-Newton matrix, while the true Hessian is
$J^{\mathsf T}J - \sum_i r_i \nabla^2 m_i$.

```python
from nalib import mlopt

out = mlopt.gauss_newton_is_the_fisher_information()
print(f"{'noise':>8}{'residual':>12}{'fisher vs gauss-newton':>26}"
      f"{'hessian vs gauss-newton':>26}{'H is positive':>16}")
for row in out["rows"]:
    print(f"{row['residual_scale']:>8.1f}{row['residual_norm']:>12.4f}"
          f"{row['fisher_gap']:>26.3e}{row['hessian_gap']:>26.6f}"
          f"{str(row['hessian_is_positive']):>16}")

assert out["fisher_equals_gauss_newton"]
assert out["they_agree_at_zero_residual"]
assert out["they_disagree_at_a_large_one"]
```

They are the same matrix, to exactly $0$. They are the Hessian only at a zero-residual solution, and
by a large residual they differ by $1.33$ of the norm and the true Hessian is no longer positive
definite. **Gauss-Newton's positive definiteness is not a happy accident, it is what you get by
dropping the term that can be negative**, and that is why it is safe to invert where the Hessian is
not.

## 13. The picture

```python
from nalib import mlopt

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

problem = mlopt.quadratic(np.array([1.0, 12.0]), seed=42)
start = np.array([1.0, 1.0])
flow = mlopt.gradient_flow(problem, start, 1.2, steps=2000)
axes[0].plot(flow["path"][:, 0], flow["path"][:, 1], "k-", lw=2.0, label="gradient flow")
for step, style in ((0.02, "o-"), (0.14, "s-")):
    run = mlopt.gradient_descent(problem, start, step, int(1.2 / step))
    axes[0].plot(run["path"][:, 0], run["path"][:, 1], style, ms=3, lw=1.0,
                 label=f"h = {step}")
axes[0].plot([0.0], [0.0], "k*", ms=10)
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].set_title("gradient descent is Euler")
axes[0].legend(fontsize=7)

wells = mlopt.two_wells(samples=400, seed=42)
axes[1].plot(wells["grid"], wells["curve"], "k-", lw=1.5)
for where, label in ((wells["flat_at"], "flat"), (wells["sharp_at"], "sharp")):
    axes[1].plot([where], [wells["value"](where)], "o", ms=7, label=label)
axes[1].set_xlabel("x")
axes[1].set_ylabel("loss")
axes[1].set_title("equal depth, curvature 3.25 and 27.8")
axes[1].legend(fontsize=7)

noise = mlopt.the_gradient_noise_falls_like_the_batch_size()
partial = [row for row in noise["rows"] if row["predicted"] > 0.0]
axes[2].loglog([r["batch"] for r in partial], [r["measured"] for r in partial], "o",
               ms=6, label="measured")
axes[2].loglog([r["batch"] for r in partial], [r["predicted"] for r in partial], "-",
               label="(1/B)(1 - (B-1)/(N-1))")
axes[2].loglog([r["batch"] for r in partial],
               [noise["population_variance"] / r["batch"] for r in partial], ":",
               color="0.5", label="1/B alone")
axes[2].set_xlabel("batch size")
axes[2].set_ylabel("gradient variance")
axes[2].set_title("the finite population correction")
axes[2].legend(fontsize=7)

fig.tight_layout(); fig.savefig("../figures/95_mlopt.png", dpi=110); plt.close(fig)
print("saved ../figures/95_mlopt.png")
```

![Gradient descent against the gradient flow, two equally deep wells of different curvature, and the minibatch gradient variance](../figures/95_mlopt.png)

The left panel is section 2: the small step traces the flow and the large one cuts across it, which
is a discretization error and looks like one. The middle is sections 8 and 9: two minima at the same
height with visibly different widths. The right is section 6, with the correction curve bending away
from the plain $1/B$ line as the batch approaches the dataset.

## 14. From scratch

Every method in this lesson is three lines, and writing them next to each other makes the family
relationships visible.

```python
def my_descent(gradient, x, step, iterations):
    """Explicit Euler on the gradient flow."""
    for _ in range(iterations):
        x = x - step * gradient(x)
    return x


def my_heavy_ball(gradient, x, step, momentum, iterations):
    """The same, plus a second time derivative."""
    previous = np.array(x, dtype=float)
    for _ in range(iterations):
        x, previous = x - step * gradient(x) + momentum * (x - previous), x
    return x


def my_preconditioned(gradient, matrix, x, step, iterations):
    """The same, in a metric chosen by the problem instead of by the coordinates."""
    factor = np.linalg.cholesky(matrix)
    for _ in range(iterations):
        x = x - step * np.linalg.solve(factor.T, np.linalg.solve(factor, gradient(x)))
    return x


from nalib import mlopt

problem = mlopt.quadratic(np.geomspace(1.0, 100.0, 8), seed=5)
start = np.ones(problem["dimension"])
budget = 200
plain = my_descent(problem["gradient"], start, 2.0 / (problem["largest"]
                                                      + problem["smallest"]), budget)
heavy = my_heavy_ball(problem["gradient"], start, 1.0 / problem["largest"], 0.9, budget)
scaled = my_preconditioned(problem["gradient"], problem["hessian"], start, 1.0, budget)

print(f"condition number {problem['condition']:.1f}, {budget} iterations each")
print(f"plain descent    : {float(np.linalg.norm(plain)):.6e}")
print(f"heavy ball       : {float(np.linalg.norm(heavy)):.6e}")
print(f"preconditioned   : {float(np.linalg.norm(scaled)):.6e}")

library = mlopt.gradient_descent(problem, start, 2.0 / (problem["largest"]
                                                        + problem["smallest"]), budget)
print(f"\nmine agrees with the library to "
      f"{float(np.max(np.abs(library['x'] - plain))):.3e}")

assert np.allclose(library["x"], plain, atol=1e-12)
assert float(np.linalg.norm(scaled)) < float(np.linalg.norm(plain))
```

**Preconditioning wins by orders of magnitude and momentum wins by a factor**, on the same problem
with the same budget, and the only difference between the three functions is what multiplies the
gradient.

## 15. Exercises

**Level 1, understanding**

1.1 State the gradient flow and say which numerical method gradient descent is for it.

1.2 Derive $h < 2/L$ from the absolute stability condition of Euler's method.

1.3 Write the heavy ball update as a discretization of a second order ODE and identify $h$ and the
damping.

1.4 Give the variance of a minibatch gradient sampled without replacement and say what happens at
$B = N$.

1.5 Give two reasons a training run leaves a sharp minimum, and say which one needs noise.

**Level 2, derivation**

2.1 Derive the optimal step size $2/(L+m)$ for a quadratic and the resulting contraction factor.

2.2 Derive the expected loss increase under a random perturbation of fixed length, and show it is
the trace of the Hessian.

2.3 Show that the natural gradient step is invariant under an invertible linear reparametrization,
and say where the argument needs $F$ to transform as it does.

2.4 Derive the Fisher information of a Gaussian model and show it equals $J^{\mathsf T}J$.

2.5 Show that a curvature estimate from $B$ samples has rank at most $B$, and say what that implies
for a Newton step.

**Level 3, computational**

3.1 Implement Nesterov momentum, identify the ODE it discretizes, and measure its order against
that ODE the way section 4 does for heavy ball.

3.2 Implement a diagonal preconditioner from running gradient second moments, which is Adam without
the momentum, and measure how much of the condition number it removes.

3.3 Implement stochastic Newton with a Levenberg-Marquardt shift and find the shift at which it
stops being worse than gradient descent at a batch of 8.

3.4 Implement sharpness-aware minimization and measure whether it actually lands in a flatter
minimum, using section 10's trace measure.

3.5 Implement a Hessian-vector product by finite differences of the gradient and use it to run
Newton-CG without ever forming the Hessian, then measure the memory saved.

**Level 4, experimental**

4.1 Measure the largest stable learning rate over a training run and see whether it moves, which is
the "edge of stability" question.

4.2 Measure how the linear scaling rule's breaking point moves as the loss surface is preconditioned,
and confirm it tracks $2/L$.

4.3 Measure the noise-driven escape rate from the sharp well as a function of the step size, and fit
it against an exponential in the barrier height over the noise scale.

**Level 5, advanced**

5.1 **The edge of stability.** Explain what section 8's window means for a real training run in
which the curvature grows as the loss falls, and predict what the largest eigenvalue does.

5.2 **Why flat minima might generalize.** Given section 10's exact statement about the trace, build
the argument connecting flatness to generalization, and say precisely which step in it is not
numerical.

5.3 **What Adam actually approximates.** Adam divides by the square root of a running average of
squared gradients. Say what matrix that is estimating, why the square root is there, and how it
avoids the failure of section 11.

## 16. Key takeaways

- **Gradient descent is explicit Euler on the gradient flow**, measured at order $0.9984$, and every
  Part 10 result applies unchanged.

- **The learning rate limit is $2/L$**, located to $1.8\times10^{-12}$, and it is lesson 74's
  absolute stability condition rather than an optimization fact.

- **Heavy ball momentum is a damped second order ODE**, tracked at order $0.9995$, so its two
  parameters are a step size and a damping constant and cannot be tuned independently.

- **Conditioning is the iteration count**, matching $(\kappa/2)\log(1/\text{tol})$ to a factor of
  $1.084$ over four decades, and preconditioning takes $88556$ iterations to $1$.

- **Minibatch noise follows the finite population formula** to $2.4$ per cent, and the full batch
  gradient is exact to $2\times10^{-33}$ rather than merely accurate.

- **The linear scaling rule survives six doublings and then meets $2/L$.** The excess loss is flat
  from batch $1$ to $64$ and is $1.7\times10^{9}$ at $128$.

- **The step size chooses between two equally deep minima before any noise does.** Their stability
  limits are $0.615$ and $0.072$, both $2/c$ to within $0.03$ per cent.

- **Losing a minimum and leaving it are different thresholds**, at $2/c$ and at $3.25$ times $2/c$.

- **Inside that window the batch size decides.** At $2.5$ times the limit a batch of $1$ leaves the
  sharp well every time and every larger batch never does.

- **Sharpness is the trace of the Hessian**, predicting the measured loss rise to $1.9$ per cent,
  and the worst direction exceeds the average by $4.48$ at a condition number of $100$.

- **Cost is not why second order methods are rare.** Newton wins the flop count by $560$ to $15.3$
  over the dimensions tested, and the crossover is at the iteration count, $3148$ against $3405$.

- **A noisy Hessian is why.** A batch of $8$ gives a rank $8$ curvature estimate for $20$ parameters
  and the run diverges; at $32$ Newton is still $6.43$ times worse than plain descent.

- **The natural gradient is exactly invariant under reparametrization**, to $2.8\times10^{-16}$,
  while plain gradient descent moves by $0.47$.

- **Gauss-Newton is the Fisher information exactly**, and it is the Hessian only at a zero residual.

## Where this goes next

Lesson 96 drops the assumption every lesson so far has made, that arithmetic is done in double
precision. Training runs in 16 bits, and the three things that break there are overflow, underflow
and non-associativity, which are lessons 03 and 04 with the numbers changed.
