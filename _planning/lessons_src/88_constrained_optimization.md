# 88. Constrained Optimization

**Part 12: Numerical Optimization**

## Learning objectives

By the end of this lesson you will be able to:

1. State the KKT conditions, check them one at a time, and say what each one catches.
2. Explain why strict complementarity is an assumption and not a conclusion.
3. Predict the penalty method's accuracy and its conditioning from the same number, and say where
   the trade runs out.
4. Use an augmented Lagrangian and say what it fixes.
5. Read the multipliers off a barrier method's central path, and use projection when the feasible
   set is simple.

## Prerequisites

Lesson 84 (optimality conditions without constraints, which these generalize). Lesson 86 (the
condition number as a statement about speed, which the penalty method makes into the central
problem). Lesson 87 (BFGS, used here to solve every subproblem). Lesson 18 (the condition number of
a linear system, which is what a penalty weight does to one).

---

## 1. The conditions

For

$$
\min f(x) \quad\text{subject to}\quad c_i(x) \le 0, \; i = 1, \dots, m ,
$$

form the Lagrangian $L = f + \sum_i \lambda_i c_i$. The **KKT conditions** at a solution are

$$
\underbrace{\nabla f + J^{T}\lambda = 0}_{\text{stationarity}}, \qquad
\underbrace{c_i \le 0}_{\text{feasibility}}, \qquad
\underbrace{\lambda_i \ge 0}_{\text{sign}}, \qquad
\underbrace{\lambda_i c_i = 0}_{\text{complementary slackness}} .
$$

The reading that makes them memorable: $\nabla f$ must be a **nonnegative combination of the active
constraint gradients**. If it were not, some direction would still go downhill without leaving the
feasible set, and the point would not be a minimum. Complementary slackness says an inactive
constraint gets a zero multiplier, since a constraint you are not touching cannot be pushing back.

This module writes every constraint as $c \le 0$ and uses $L = f + \lambda^{T}c$ throughout. That
forces $\lambda \ge 0$ for inequalities, and for an equality the sign is free.

The four conditions are checked **separately**, because they fail for different reasons and a
combined norm hides which.

```python
from nalib import constrained as cs

out = cs.the_conditions_hold_at_the_answer()
print(f"{'problem':>62}{'worst residual':>16}")
for row in out["rows"]:
    print(f"{row['name']:>62}{row['worst']:>16.3e}")
print(f"\nevery stated answer satisfies its own conditions: "
      f"{out['every_answer_satisfies_them']}")
print(out["note"])

assert out["every_answer_satisfies_them"]
```

Every test problem in this lesson is checked against its own conditions **before** any method runs
on it. A stated answer that failed its own KKT residual would make every later measurement
worthless, and this is the cheapest place to find that out.

### 1.1 Strict complementarity is an assumption

Most convergence theorems in this area assume **strict** complementarity: every active constraint
has a strictly positive multiplier. That is not part of the KKT conditions and it is not automatic.

```python
out = cs.strict_complementarity_is_not_automatic()
for row in out["rows"]:
    print(f"target {row['target']}: minimizer {row['minimizer']}, "
          f"active constraints {row['active_constraints']}")
    print(f"    multipliers there: "
          f"{[f'{v:.3e}' for v in row['multipliers_there']]}, strict: {row['strict']}")
print(f"\none of each: {out['one_of_each']}")
print(out["note"])

assert out["one_of_each"]
```

Both problems minimize the distance to a target over the same triangle. With the target at
$(1.5, 1.5)$ the answer is $(0.5, 0.5)$, one constraint is active, its multiplier is $2$, and
everything is as the theorems want it.

With the target at $(2, 1)$ the answer is the **vertex** $(1, 0)$. Two constraints are active and
one of them has multiplier $2.7\times10^{-16}$, which is zero. That constraint is touching but not
pushing: the objective does not want to cross it anyway. Nothing is pathological about this, it is
what happens when the unconstrained minimum projects exactly onto a corner, and section 4 shows what
it costs.

---

## 2. The penalty method, and the trade it cannot escape

Add a term that grows when the constraint is violated, and minimize freely:

$$
P_\mu(x) = f(x) + \frac{\mu}{2}\lVert c(x)\rVert^{2} .
$$

The minimizer of $P_\mu$ is not the constrained solution for any finite $\mu$, and that is the
point: the constraint is only enforced by the pull of the penalty, so the pull has to grow.

Two consequences follow from the same $\mu$, in opposite directions. The error is $O(1/\mu)$, and
the Hessian is $\nabla^{2}f + \mu J^{T}J$, whose condition number is $O(\mu)$.

```python
out = cs.the_penalty_trades_accuracy_for_conditioning()
print(f"{'weight':>10}{'error':>13}{'violation':>13}{'condition':>13}"
      f"{'inner steps':>13}{'converged':>11}")
for row in out["rows"]:
    print(f"{row['weight']:>10.0e}{row['error']:>13.3e}{row['violation']:>13.3e}"
          f"{row['condition']:>13.3e}{row['inner_steps']:>13}"
          f"{str(row['converged']):>11}")
print(f"\nerror times weight: "
      f"{[f'{v:.6f}' for v in out['error_times_weight']]}")
print(f"condition over weight: "
      f"{[f'{v:.6f}' for v in out['condition_over_weight']]}")
print(f"largest weight that worked: {out['largest_weight_that_worked']:.0e}")
print(f"best violation reached:     {out['best_violation_reached']:.3e}")
print(out["note"])

assert out["the_error_is_one_over_mu"]
assert out["the_condition_is_mu"]
```

Both predictions are exact. The error times the weight is $0.7071$ at every size, which is
$1/\sqrt2$ for this problem, and the condition number over the weight is $1.0000$.

**Then it stops working, in three distinct ways and in order.**

- At $\mu = 10^{14}$ the subproblem no longer converges in 3000 BFGS steps. The condition number is
  $10^{14}$ and lesson 86 said what that costs.
- At $10^{16}$ the curvature approximation goes numerically singular. The BFGS diagnostic that
  reports the secant residual had to be guarded against exactly this while writing the lesson,
  since a diagnostic must not be able to stop a solve.
- At $10^{18}$ the penalty term **overflows** at the starting point, before the solve begins.

So the reachable accuracy is capped by the largest weight the subproblem survives, which here is
about $10^{11}$, giving a constraint violation of about $10^{-11}$. **The penalty method's accuracy
limit is a property of the arithmetic, not of the problem.**

---

## 3. The augmented Lagrangian

The penalty method needs $\mu \to \infty$ because it is trying to hold the iterate near the
boundary by force alone. Add the multiplier back and that stops being necessary:

$$
L_\mu(x, \lambda) = f(x) + \lambda^{T}c(x) + \frac{\mu}{2}\lVert c(x)\rVert^{2} .
$$

At the **true** multiplier, the unconstrained minimum of $L_\mu$ is exactly the constrained
solution, for any $\mu$ above a threshold. The multiplier is not known, so it is estimated, and the
estimate improves with the same update every time:

$$
\lambda \leftarrow \lambda + \mu\, c(x) .
$$

```python
out = cs.the_augmented_lagrangian_keeps_the_weight_bounded()
print(f"{'round':>7}{'weight':>9}{'multiplier':>16}{'error':>13}{'violation':>13}"
      f"{'condition':>12}{'inner steps':>13}")
for row in out["rows"]:
    print(f"{row['round']:>7}{row['weight']:>9.0f}{row['multiplier'][0]:>16.9f}"
          f"{row['error']:>13.3e}{row['violation']:>13.3e}"
          f"{row['condition']:>12.4f}{row['inner_steps']:>13}")
print(f"\nexact multiplier {out['exact_multipliers'][0]}, "
      f"estimate error {out['multiplier_error']:.3e}")
print(f"the weight never grew: {out['weight_never_grew']}, final value "
      f"{out['final_weight']:.0f}")
print(f"condition number throughout: {out['condition_stayed_at']:.4f}")
print(f"the penalty method needed a condition number of "
      f"{out['penalty_condition_for_the_same_violation']:.3e} for the same violation")
print(out["note"])

assert out["weight_never_grew"]
assert out["multiplier_error"] < 1e-7
```

Eight rounds take the violation from $9.1\times10^{-2}$ to $4.7\times10^{-9}$, a factor of ten each
round, while:

- the weight stays at **10**,
- the condition number stays at **11.0**,
- each subproblem takes **two** BFGS steps,
- and the multiplier converges to $-0.999999995$ against an exact $-1$.

The penalty method needed a weight of $10^{9}$ and a condition number of $10^{11}$ to reach the same
violation. **The augmented Lagrangian buys the accuracy with iterations instead of conditioning**,
and iterations are the cheap currency.

That is also why the multiplier is worth having for its own sake: it is the price of the constraint,
the rate at which the optimal value would change if the constraint moved, and here it arrives free.

---

## 4. Barrier methods and the central path

A barrier goes the other way. Instead of punishing violation, make the boundary infinitely
expensive and start inside:

$$
B_\mu(x) = f(x) - \mu \sum_i \log(-c_i(x)) .
$$

Every iterate is **strictly feasible**, which matters whenever an infeasible point is meaningless:
a negative concentration, a schedule that violates a hard deadline, a design that does not exist.
The minimizers as $\mu$ falls trace the **central path**, which ends at the solution.

```python
out = cs.the_central_path()
print(f"target {out['target']}, exact minimizer {out['exact_minimizer']}, "
      f"exact multipliers {[f'{v:.4f}' for v in out['exact_multipliers']]}")
print(f"{'weight':>10}{'x1':>14}{'x2':>14}{'distance':>12}"
      f"{'multiplier 1':>15}{'inside':>9}")
for row in out["rows"]:
    print(f"{row['weight']:>10.0e}{row['x'][0]:>14.9f}{row['x'][1]:>14.9f}"
          f"{row['distance']:>12.3e}{row['multiplier_estimate'][0]:>15.9f}"
          f"{str(row['strictly_feasible']):>9}")
print(f"\nevery point strictly feasible: {out['every_point_strictly_feasible']}")
print(out["note"])

assert out["every_point_strictly_feasible"]
assert out["the_multipliers_come_free"]
```

Two things worth separating.

**The path stays inside and ends at the answer.** Every one of the ten points has all three
constraints strictly negative, and the distance to the solution falls like $\sqrt\mu$.

**The multipliers are a by-product.** At the barrier minimum the stationarity condition reads
$\nabla f - \mu \sum (1/c_i)\nabla c_i = 0$, which is exactly the KKT stationarity condition with

$$
\lambda_i = \frac{\mu}{-c_i(x)} .
$$

The estimate for the first constraint runs $4.93, 2.47, 2.12, 2.03, 2.010, 2.0032, 2.0010, 2.00032,
2.00010, 2.000032$, converging on the exact $2$. **The dual solution arrives without being asked
for**, which is the property interior point methods are built on: they solve the primal and dual
problems together.

### 4.1 What the zero multiplier costs

Section 1.1 found a constraint that is active with multiplier zero. The central path notices.

```python
out = cs.the_path_slows_where_complementarity_is_not_strict()
print(f"{'target':>14}{'strict':>9}{'active':>9}{'final distance':>17}"
      f"{'final multiplier error':>25}")
for row in out["rows"]:
    print(f"{str(row['target']):>14}{str(row['strict_complementarity']):>9}"
          f"{row['active_constraints']:>9}{row['final_distance']:>17.3e}"
          f"{row['final_multiplier_error']:>25.3e}")
print(out["note"])

assert out["the_strict_case_is_more_accurate"]
```

With strict complementarity the multiplier error at the end of the path is
$1.65\times10^{-7}$. Without it, $6.32\times10^{-5}$: **380 times worse** on the same problem class
with the same weights.

The mechanism is a race. For a constraint with a positive multiplier, $-c_i$ settles at
$\mu/\lambda_i$, so the ratio converges quickly. For one with a zero multiplier both $\mu$ and
$-c_i$ go to zero and their ratio is decided by which goes faster, which is slower and less
accurate. This is the practical reason interior point codes watch for near-degenerate constraints.

---

## 5. Projection, and where this part is going

When the feasible set is simple, none of the above is needed. Take a gradient step and map the
result back:

$$
x_{k+1} = P_C\!\left(x_k - \alpha \nabla f(x_k)\right) .
$$

Every iterate is feasible by construction. This needs $P_C$ to be cheap, and for three sets it is.

- **A box**: clip each coordinate, $O(n)$.
- **A ball**: scale down if outside, $O(n)$.
- **A simplex**: $\max(p - \theta, 0)$ for the $\theta$ that makes the sum right, found by a sort,
  $O(n\log n)$.

A projection is not just "some feasible point nearby". It satisfies

$$
\big(p - P_C(p)\big)^{T}\big(z - P_C(p)\big) \le 0 \quad \text{for every } z \in C ,
$$

and that inequality is what makes projected gradient a descent method. It is checkable, and it is a
stronger check than feasibility.

```python
out = cs.the_projections_are_projections()
print(f"{'set':>16}{'worst inner product':>22}{'idempotent to':>16}{'a projection':>15}")
for row in out["rows"]:
    print(f"{row['set']:>16}{row['worst_inner_product']:>22.3e}"
          f"{row['idempotent_to']:>16.3e}{str(row['satisfies_the_definition']):>15}")
print(out["note"])

assert out["all_are_projections"]
assert out["all_idempotent"]
```

The defining inequality holds on every random pair of points, at random sizes from 1 to 7, and each
projection is its own square to $4\times10^{-16}$.

### 5.1 The simplex projection, from scratch

The simplex projection is the only one of the three that is not obvious, and the derivation is
short. The answer has the form $\max(p - \theta, 0)$ for a single scalar $\theta$, because the
Lagrangian of "minimize $\lVert x - p\rVert^2$ subject to $\sum x = 1$, $x \ge 0$" gives
$x_i = p_i - \theta + \mu_i$ with $\mu_i \ge 0$ active only where $x_i = 0$. So the whole problem is
finding the $\theta$ that makes the sum come out to one, and sorting turns that into a scan.

```python
import numpy as np


def my_simplex_projection(point, total=1.0):
    p = np.asarray(point, dtype=float).ravel()
    ordered = np.sort(p)[::-1]
    running = np.cumsum(ordered) - float(total)
    index = np.arange(1, p.size + 1)
    allowed = ordered - running / index > 0.0
    count = int(np.max(np.flatnonzero(allowed))) + 1 if np.any(allowed) else 1
    return np.maximum(p - running[count - 1] / count, 0.0)


def my_projected_gradient(gradient, projection, start, step=0.25, tol=1e-10,
                          budget=20000):
    x = projection(np.asarray(start, dtype=float).ravel())
    for taken in range(budget):
        moved = projection(x - step * np.asarray(gradient(x), dtype=float).ravel())
        if float(np.linalg.norm(moved - x)) <= tol:
            break
        x = moved
    return x, taken


rng = np.random.default_rng(42)
print(f"{'variables':>11}{'gap to the library':>21}{'sum':>10}"
      f"{'steps':>8}{'gap to the closed form':>25}")
for size in (1, 4, 12, 40):
    point = rng.normal(scale=2.0, size=size)
    mine = my_simplex_projection(point)
    theirs = cs.project_onto_simplex(point)
    target = rng.normal(scale=1.5, size=size)
    answer, steps = my_projected_gradient(
        lambda v: 2.0 * (np.asarray(v, dtype=float) - target),
        my_simplex_projection, np.full(size, 1.0 / size))
    print(f"{size:>11}{float(np.max(np.abs(mine - theirs))):>21.3e}"
          f"{float(np.sum(mine)):>10.6f}{steps:>8}"
          f"{float(np.linalg.norm(answer - cs.project_onto_simplex(target))):>25.3e}")
```

The two projections agree exactly, every result sums to one, and the iteration lands on the same
point the closed form gives. The last column is the check worth having: **minimizing the distance to
a point over a set is the projection onto that set**, so an iteration that disagreed with the closed
form would be wrong in one of the two.

```python
out = cs.projected_gradient_stays_feasible()
print(f"{'variables':>11}{'steps':>8}{'feasible throughout':>22}{'sum':>10}"
      f"{'smallest entry':>17}{'gap to the closed form':>25}")
for row in out["rows"]:
    print(f"{row['variables']:>11}{row['steps']:>8}"
          f"{str(row['every_iterate_feasible']):>22}"
          f"{row['sum_of_the_answer']:>10.6f}{row['smallest_entry']:>17.3e}"
          f"{row['gap_to_the_direct_projection']:>25.3e}")
print(out["note"])

assert out["always_feasible"]
assert out["matches_the_direct_projection"]
```

Every iterate is feasible, the answers sum to exactly $1$ with no negative entry, and they agree
with the closed form projection to $1.7\times10^{-9}$, which is the right check because minimizing
the distance to a point over a set **is** the projection onto that set.

### 5.1 The connection forward

Projection onto $C$ is the **proximal operator of the indicator function** of $C$:

$$
P_C(v) = \arg\min_x \; \tfrac12\lVert x - v\rVert^{2} + \iota_C(x),
\qquad \iota_C(x) = \begin{cases}0 & x \in C\\ \infty & x \notin C .\end{cases}$$

So projected gradient is the special case of **proximal gradient** in which the nonsmooth part of
the objective is an indicator. Lesson 90 keeps the same iteration and replaces the indicator by
something finite, such as $\lVert x\rVert_1$, whose proximal operator is soft thresholding. The
machinery is already here; only the operator changes.

```python
import matplotlib.pyplot as plt
import numpy as np

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.0))

table = cs.the_penalty_trades_accuracy_for_conditioning()
good = [r for r in table["rows"] if r["converged"]]
left.loglog([r["weight"] for r in good], [r["violation"] for r in good], "o-",
            ms=5, lw=1.5, label="constraint violation")
left.loglog([r["weight"] for r in good], [r["condition"] for r in good], "s--",
            ms=5, lw=1.5, label="condition number")
for row in table["rows"]:
    if not row["converged"]:
        left.axvline(row["weight"], color="0.7", lw=1.0)
left.text(2e13, 1e-4, "the subproblem\nstops converging", fontsize=8, color="0.35")
left.set_xlabel("penalty weight")
left.set_title("the same knob, turned both ways")
left.legend(fontsize=8)
left.grid(alpha=0.3, which="both")

path = cs.the_central_path()
points = np.array([row["x"] for row in path["rows"]])
corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
right.plot(corners[:, 0], corners[:, 1], "-", color="0.5", lw=1.2,
           label="the feasible triangle")
right.plot(points[:, 0], points[:, 1], "o-", ms=4, lw=1.4, label="the central path")
right.plot(*path["exact_minimizer"], "*", ms=13, label="the solution")
right.plot(*path["target"], "x", ms=9, label="the unconstrained minimum")
right.set_aspect("equal")
right.set_xlabel("x1")
right.set_ylabel("x2")
right.set_title("every point strictly inside")
right.legend(fontsize=8, loc="upper right")
right.grid(alpha=0.3)

fig.tight_layout(); fig.savefig("../figures/88_constrained.png", dpi=110); plt.close(fig)
print("saved ../figures/88_constrained.png")
```

![The penalty trade-off, and the central path inside the feasible set](../figures/88_constrained.png)

---

## 6. Exercises

**Level 1, understanding**

1.1 State the KKT conditions and say what each one rules out.

1.2 Explain why an inactive constraint has a zero multiplier.

1.3 Say why the penalty method needs $\mu \to \infty$ and the augmented Lagrangian does not.

1.4 Explain how a barrier method produces the multipliers.

1.5 Say what a projection is, beyond returning a feasible point.

**Level 2, derivation**

2.1 Derive the KKT conditions from the geometry of feasible descent directions.

2.2 Derive the penalty method's $O(1/\mu)$ error and $O(\mu)$ conditioning for a linear constraint.

2.3 Show that at the true multiplier the augmented Lagrangian's unconstrained minimum is the
constrained solution.

2.4 Derive $\lambda_i = \mu / (-c_i)$ on the central path.

2.5 Derive the simplex projection $\max(p - \theta, 0)$ and the equation $\theta$ solves.

**Level 3, computational**

3.1 Implement sequential quadratic programming, which solves a quadratic model with linearized
constraints at each step, and compare it against the augmented Lagrangian.

3.2 Implement an active set method for a quadratic program and compare it against the barrier
method on the same problem.

3.3 Implement a primal-dual interior point method and compare its iteration count against the
barrier method's.

3.4 Implement the projection onto the intersection of two convex sets by alternating projections,
and measure the rate.

3.5 Implement projected Newton, which projects the Newton step rather than the gradient step, and
measure what the curvature buys under constraints.

**Level 4, experimental**

4.1 Measure the penalty method's error and conditioning over twelve decades of weight and find both
breakdown points.

4.2 Measure the augmented Lagrangian's convergence rate against the fixed weight $\mu$, and find
the threshold below which it fails.

4.3 Measure the barrier method's iteration count against the number of constraints, at several
problem sizes.

**Level 5, advanced**

5.1 **Why the multiplier is a price.** Show that $\lambda_i$ is the derivative of the optimal value
with respect to relaxing constraint $i$, and verify it numerically.

5.2 **Degeneracy.** Construct a problem where the active constraint gradients are linearly
dependent, show the multipliers are not unique, and say what that does to each method here.

5.3 **The proximal view.** Show that projection is the proximal operator of an indicator function,
and identify which results of this lesson carry over to a general nonsmooth term and which do not.

## 7. Key takeaways

- **The KKT conditions are four separate statements** and are checked here one at a time, because
  a combined norm hides which one failed.

- **Strict complementarity is an assumption.** One of the two targets here has an active constraint
  with multiplier $2.7\times10^{-16}$, and it is not pathological.

- **The penalty method's accuracy and its conditioning are the same number.** Error times weight is
  $0.7071$ and condition over weight is $1.0000$, at every size.

- **It breaks in three ways, in order**: the subproblem stops converging at $10^{14}$, the
  curvature model goes singular at $10^{16}$, and the objective overflows at $10^{18}$. The
  reachable violation is capped near $10^{-11}$.

- **The augmented Lagrangian reaches the same violation with the weight fixed at 10**, a condition
  number of $11$, and two BFGS steps per round.

- **The multiplier arrives free either way**, from the update or from the central path, and it
  converges to $-0.999999995$ and $2.000032$ against exact values of $-1$ and $2$.

- **Every point of the central path is strictly feasible**, which is the property that makes
  interior point methods usable where an infeasible iterate has no meaning.

- **A zero multiplier costs 380 times the accuracy** in the multiplier estimate at the end of the
  path.

- **A projection satisfies an inequality, not just feasibility**, and all three here satisfy it to
  $1.6\times10^{-15}$ and are idempotent to $4\times10^{-16}$.

- **Projected gradient is proximal gradient with an indicator function**, which is where lesson 90
  picks the thread up.

## Where this goes next

Lesson 89 changes what a gradient is. Everything in lessons 84 to 88 assumed $\nabla f$ could be
computed exactly, and in a great many modern problems it is a sum over millions of terms and only a
sample of it is affordable. The methods that result, SGD and its momentum and adaptive variants,
have their own convergence theory, and the first thing that theory says is that a constant step size
does not converge.
