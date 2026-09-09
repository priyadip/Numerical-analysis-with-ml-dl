# 69. Runge-Kutta Methods

**Part 10: Ordinary Differential Equations**

## Learning objectives

By the end of this lesson you will be able to:

1. Derive the second order Runge-Kutta family from the order conditions and see that it is a one
   parameter family, not a single method.
2. Read and write a Butcher tableau, and check a tableau against the order conditions in exact
   arithmetic.
3. Verify a method's order by refinement, and know what a mismatch means.
4. Say why order 5 needs 6 stages, and what the stage barriers cost.
5. Compare methods **at equal evaluation count**, and find the one place where the highest order
   loses.

## Prerequisites

Lesson 68 (Taylor methods and the elementary differentials, which are exactly the order
conditions here). Lesson 67 (local and global order). Lesson 62 (the quadrature rules the tableaux
reduce to when $f$ does not depend on $y$).

---

## 1. The idea

Euler uses one slope. A Runge-Kutta method uses several, sampled inside the step, and combines
them:

$$
k_i = f\left(t_n + c_i h,\; y_n + h\sum_{j} a_{ij} k_j\right), \qquad
y_{n+1} = y_n + h \sum_i b_i k_i.
$$

The coefficients $a_{ij}$, $b_i$, $c_i$ are the method. They are written as a **Butcher tableau**:

$$
\begin{array}{c|c} c & A \\ \hline & b^{\mathsf T}\end{array}
$$

The method is **explicit** when $A$ is strictly lower triangular, so each stage uses only stages
already computed. Everything in this lesson is explicit; lesson 72 needs implicit ones.

```python
# Standard setup, the same in every lesson of this course.
import sys, pathlib

_root = pathlib.Path.cwd()
while not (_root / "src" / "nalib").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import matplotlib.pyplot as plt

SEED = 42                                  # fixed so your numbers match the text
rng = np.random.default_rng(SEED)

np.set_printoptions(precision=6, linewidth=100, suppress=False)
plt.rcParams.update({
    "figure.figsize": (7.5, 4.5), "figure.dpi": 110,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})
```

```python
from nalib import rungekutta as rk

A, b, c, order = rk.tableau_exact("rk4")       # exact Fractions, not floats
print("classical RK4")
print(f"{'c':>6} | A")
print("-" * 34)
for i in range(len(b)):
    print(f"{str(c[i]):>6} | " + "".join(f"{str(v):>7}" for v in A[i]))
print("-" * 34)
print(f"{'':>6} | " + "".join(f"{str(v):>7}" for v in b))
print(f"\nclaimed order {order}, explicit: {rk.is_explicit(A)}")
```

*Output:*

```text
classical RK4
     c | A
----------------------------------
     0 |       0      0      0      0
   1/2 |     1/2      0      0      0
   1/2 |       0    1/2      0      0
     1 |       0      0      1      0
----------------------------------
       |     1/6    1/3    1/3    1/6

claimed order 4, explicit: True
```

`tableau_exact` keeps the coefficients as `Fraction` objects. `tableau` returns the same
numbers as floats, which is what a stepper wants. Section 3 needs the exact ones.

## 2. Deriving the second order family

Take two stages:

$$
k_1 = f(t_n, y_n), \qquad
k_2 = f(t_n + \alpha h, y_n + \alpha h k_1), \qquad
y_{n+1} = y_n + h(b_1 k_1 + b_2 k_2).
$$

Expand $k_2$ in a Taylor series about $(t_n, y_n)$:

$$
k_2 = f + \alpha h(f_t + f_y f) + O(h^2),
$$

so

$$
y_{n+1} = y_n + h(b_1 + b_2)f + \alpha b_2 h^2 (f_t + f_y f) + O(h^3).
$$

Compare with the true expansion $y_n + hf + \tfrac{h^2}{2}(f_t + f_y f) + O(h^3)$. Matching gives

$$
b_1 + b_2 = 1, \qquad \alpha b_2 = \tfrac12.
$$

**Two equations, three unknowns.** There is a one parameter family of second order two stage
methods, and the named ones are just choices of $\alpha$:

| $\alpha$ | $b_1$ | $b_2$ | name |
|---|---|---|---|
| $1/2$ | $0$ | $1$ | midpoint |
| $1$ | $1/2$ | $1/2$ | Heun |
| $2/3$ | $1/4$ | $3/4$ | Ralston |

```python
for name in ("midpoint", "heun", "ralston"):
    A, b, c, p = rk.tableau(name)
    print(f"{name:>10}: alpha = {c[1]}, b = {b}, claimed order {p}")
```

*Output:*

```text
  midpoint: alpha = 0.5, b = [0. 1.], claimed order 2
      heun: alpha = 1.0, b = [0.5 0.5], claimed order 2
   ralston: alpha = 0.6666666666666666, b = [0.25 0.75], claimed order 2
```

Ralston's choice minimises the leading error coefficient. It is not more accurate by an order,
only by a constant, which is worth remembering when a table of methods presents it as better.

Notice what happens when $f$ does not depend on $y$: the midpoint tableau becomes the midpoint
quadrature rule and Heun becomes the trapezoid rule. **Every explicit Runge-Kutta method reduces
to a quadrature rule on $\int f(t)\,dt$**, and its $b_i$ and $c_i$ are that rule's weights and
nodes.

## 3. Order conditions, checked exactly

Matching the Taylor expansions to order $p$ gives one condition per elementary differential of
lesson 68, so the counts 1, 2, 4, 8 are the number of conditions at orders 1, 2, 3, 4. In terms
of the tableau, with $c_i = \sum_j a_{ij}$:

$$
\sum b_i = 1, \quad
\sum b_i c_i = \tfrac12, \quad
\sum b_i c_i^2 = \tfrac13, \quad
\sum b_i a_{ij} c_j = \tfrac16, \quad \dots
$$

```python
out = rk.order_conditions(*rk.tableau_exact("rk4")[:3], up_to=5)
print(f"exact arithmetic: {out['exact']}\n")
print(f"{'order':>6}  {'condition':>20}{'required':>10}{'computed':>10}{'miss':>10}  holds")
for o, name, req, got, miss, ok in zip(out["order"], out["condition"], out["required"],
                                       out["computed"], out["residual"], out["holds"]):
    print(f"{o:>6}  {name:>20}{str(req):>10}{str(got):>10}{str(miss):>10}  {ok}")
print(f"\nachieved order: {out['achieved_order']}")
assert out["exact"], "the tableau was given as Fractions, so the check must be exact"
assert out["achieved_order"] == 4, "RK4 is fourth order and not fifth"
assert all(r == 0 for o, r in zip(out["order"], out["residual"]) if o <= 4)
assert all(r != 0 for o, r in zip(out["order"], out["residual"]) if o == 5)
```

*Output:*

```text
exact arithmetic: True

 order             condition  required  computed      miss  holds
     1                 sum b         1         1         0  True
     2               sum b c       1/2       1/2         0  True
     3             sum b c^2       1/3       1/3         0  True
     3             sum b A c       1/6       1/6         0  True
     4             sum b c^3       1/4       1/4         0  True
     4         sum b c (A c)       1/8       1/8         0  True
     4           sum b A c^2      1/12      1/12         0  True
     4           sum b A A c      1/24      1/24         0  True
     5             sum b c^4       1/5      5/24    -1/120  False
     5       sum b c^2 (A c)      1/10      5/48    -1/240  False
     5         sum b (A c)^2      1/20      1/16     -1/80  False
     5       sum b c (A c^2)      1/15      1/16     1/240  False
     5           sum b A c^3      1/20      1/24     1/120  False
     5       sum b c (A A c)      1/30      1/24    -1/120  False
     5     sum b A (c (A c))      1/40      1/48     1/240  False
     5         sum b A A c^2      1/60      1/48    -1/240  False
     5         sum b A A A c     1/120         0     1/120  False

achieved order: 4
```

Every condition through order 4 has a **miss of exactly zero**, and every one of the nine
order 5 conditions misses by an exact nonzero rational, the worst being $-1/80$. That is the
difference exact arithmetic buys: not a smaller number, a **different kind of statement**. In
floating point the first condition sums to $0.9999999999999999$ and the only available conclusion
is "within the tolerance I chose".

The condition counts are $1, 1, 2, 4, 9$ at orders 1 to 5, which are the rooted tree counts of
lesson 68. The order conditions **are** the elementary differentials, one equation each.

Checking all eight tableaux at once, both ways:

```python
out = rk.all_tableaux_are_what_they_claim()
print(f"{'name':>16}{'stages':>8}{'claimed':>9}{'exact says':>12}{'float says':>12}"
      f"{'worst float miss':>19}")
for n, st, p, ex, fl, res in zip(out["names"], out["stages"], out["claimed_order"],
                                 out["conditions_satisfied_to"],
                                 out["float_conditions_satisfied_to"],
                                 out["worst_float_residual"]):
    print(f"{n:>16}{st:>8}{p:>9}{ex:>12}{fl:>12}{res:>19.3e}")
print(f"\nall agree with their claims: {out['all_agree']}")
print(f"exact and float reach the same verdict: {out['exact_and_float_agree']}")
assert out["all_agree"] and out["exact_and_float_agree"]
```

*Output:*

```text
            name  stages  claimed  exact says  float says   worst float miss
           euler       1        1           1           1          5.000e-01
        midpoint       2        2           2           2          1.667e-01
            heun       2        2           2           2          1.667e-01
         ralston       2        2           2           2          1.667e-01
     kutta third       3        3           3           3          4.167e-02
      heun third       3        3           3           3          4.167e-02
             rk4       4        4           4           4          1.110e-16
   three eighths       4        4           4           4          5.551e-17

all agree with their claims: True
exact and float reach the same verdict: True
```

The two verdicts agree here, and that is the point of running both: **the float column cannot
say so on its own.** Look at the last column. For RK4 the worst float residual is
$1.1\times10^{-16}$ and for Euler it is $0.5$; those are different in kind, and a reader given
only that column has to decide where between them to draw a line. The exact column has no line to
draw.

A tableau either satisfies its order conditions or it does not.

## 4. Order by refinement

The order conditions say what the method should do. Refinement says what it does.

```python
from nalib import ivp

def f(t, y):
    return ivp.as_state(y) - t ** 2 + 1.0

def exact(t):
    return (t + 1.0) ** 2 - 0.5 * np.exp(t)

print(f"{'name':>16}{'stages':>8}{'claimed':>9}{'fitted':>10}{'error at 320':>16}")
for name in ("euler", "heun", "kutta third", "rk4"):
    out = rk.verified_order(f, exact, 0.0, 0.5, 2.0, name)
    A, b, c, p = rk.tableau(name)
    print(f"{name:>16}{len(b):>8}{p:>9}{out['fitted_order']:>10.4f}"
          f"{out['errors'][-1]:>16.3e}")
```

*Output:*

```text
            name  stages  claimed    fitted    error at 320
           euler       1        1    0.9641       4.203e-03
            heun       2        2    1.9863       4.790e-06
     kutta third       3        3    2.9954       8.565e-10
             rk4       4        4    3.9897       4.547e-13
```

Each fitted order is within a few hundredths of the claim. A mismatch would mean one of three
things, in decreasing order of likelihood: a coding mistake in the stepper, a problem too smooth
or too degenerate to show the order (lesson 67 section 4.1), or a wrong tableau.

## 5. Stage barriers

Up to order 4, a method of order $p$ needs $p$ stages. **Then it stops.** Order 5 needs 6 stages,
order 6 needs 7, order 7 needs 9, order 8 needs 11.

```python
out = rk.order_against_stages()
print(f"{'name':>16}{'stages':>8}{'order':>7}{'order per stage':>18}")
for n, st, o, r in zip(out["names"], out["stages"], out["order"], out["order_per_stage"]):
    print(f"{n:>16}{st:>8}{o:>7}{r:>18.4f}")
print(f"\nthe Butcher barriers, minimum stages for each order:")
print("   order  " + "".join(f"{v:>5}" for v in out["barrier_order"]))
print("  stages  " + "".join(f"{v:>5}" for v in out["barrier_stages"]))
print(f"\nstages equal order up to {out['efficient_up_to']}, and not past it")
```

*Output:*

```text
            name  stages  order   order per stage
           euler       1      1            1.0000
        midpoint       2      2            1.0000
            heun       2      2            1.0000
         ralston       2      2            1.0000
     kutta third       3      3            1.0000
      heun third       3      3            1.0000
             rk4       4      4            1.0000
   three eighths       4      4            1.0000

the Butcher barriers, minimum stages for each order:
   order      1    2    3    4    5    6    7    8
  stages      1    2    3    4    6    7    9   11

stages equal order up to 4, and not past it
```

That barrier is why RK4 is the default in every textbook and every quick script. It is the last
order that is free, in the sense that each extra stage buys a whole order.

Past it, the arithmetic changes. Dormand-Prince, the method behind `ode45` and `solve_ivp`'s
default, is order 5 in 7 stages (6 of them effective, see lesson 70). It is worth the extra stages
only because the order is higher, and the crossover depends on the accuracy wanted.

## 6. Comparing at equal cost

Comparing methods per **step** is meaningless: RK4 does four evaluations to Euler's one. The
honest comparison fixes the number of evaluations of $f$.

```python
out = rk.compare_at_equal_cost(f, exact, 0.0, 0.5, 2.0)
print(f"{'evaluations':>13}" + "".join(f"{n:>14}" for n in out["names"]))
for j, e in enumerate(out["evaluations"]):
    row = "".join(f"{out['errors'][n][j]:>14.3e}" for n in out["names"])
    print(f"{e:>13}{row}")
print("\nbest at each budget: " + ", ".join(out["best_at_each_budget"]))
print(f"highest order always wins: {out['highest_order_always_wins']}")
```

*Output:*

```text
  evaluations         euler          heun   kutta third           rk4
           48     1.072e-01     1.322e-02     4.320e-04     5.302e-05
           96     5.483e-02     3.358e-03     5.446e-05     3.385e-06
          192     2.774e-02     8.458e-04     6.832e-06     2.136e-07
          384     1.395e-02     2.122e-04     8.553e-07     1.341e-08
          768     6.996e-03     5.315e-05     1.070e-07     8.403e-10

best at each budget: rk4, rk4, rk4, rk4, rk4
highest order always wins: True
```

The highest order wins at every budget here, and the margin widens as the budget grows, because
the orders differ. At the smallest budget the gap is smallest, which is the general shape: **a
high order method needs enough steps to be in its asymptotic regime before its order helps.**

```python
fig, ax = plt.subplots()
work = rk.compare_at_equal_cost(f, exact, 0.0, 0.5, 2.0)
for name in work["names"]:
    ax.loglog(work["evaluations"], work["errors"][name], "o-", label=name)
ax.set_xlabel("evaluations of f"); ax.set_ylabel("error at t = 2")
ax.set_title("cost against accuracy, at equal evaluation count")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/69_equal_cost.png", dpi=110); plt.close(fig)
print("saved ../figures/69_equal_cost.png")
```

*Output:*

```text
saved ../figures/69_equal_cost.png
```

![Runge-Kutta methods at equal cost](../figures/69_equal_cost.png)

The lines are straight on log axes and their slopes are the orders. **The gaps widen to the
right**, which is what a higher order buys: not a constant advantage but a growing one.

## 7. Where high order loses

Accuracy is not the only thing a step size has to satisfy. On a rapidly decaying problem the step
is limited by **stability**, and there the ranking changes.

```python
print(f"{'name':>16}{'stability limit':>18}{'largest stable h':>19}"
      f"{'steps needed':>15}{'evaluations':>14}")
out = rk.explicit_stability_is_not_the_answer(lam=-1000.0, t_end=1.0)
for n, lim, h, s, e in zip(out["names"], out["stability_limit"],
                           out["largest_stable_step"], out["steps_needed"],
                           out["evaluations_needed"]):
    print(f"{n:>16}{lim:>18.6f}{h:>19.3e}{s:>15}{e:>14}")
print(f"\nspread in the stability limits: {out['spread']:.2f}x")
print(f"spread in the evaluations they cost: {out['evaluation_spread']:.2f}x")
print(f"cheapest: {out['cheapest']}, and it is the lowest order one: "
      f"{out['cheapest_is_the_lowest_order']}")
```

*Output:*

```text
            name   stability limit   largest stable h   steps needed   evaluations
           euler          2.000000          2.000e-03            500           500
            heun          2.000000          2.000e-03            500          1000
     kutta third          2.512745          2.513e-03            398          1194
             rk4          2.785294          2.785e-03            360          1440

spread in the stability limits: 1.39x
spread in the evaluations they cost: 2.88x
cheapest: euler, and it is the lowest order one: True
```

For $y' = -1000y$, Euler needs $h < 0.002$ and RK4 needs $h < 0.00279$. **The stability limits
span a factor of 1.39 and the stage counts span a factor of 4**, so the evaluations span 2.88 the
other way: 500 for Euler against 1440 for RK4. At the stability limit the lowest order method is
the cheapest one.

This is not an accuracy comparison. At those step sizes both methods are far more accurate than
anyone needs; the step is set by stability alone, and paying for order is paying for nothing. The
whole answer is in lesson 72, which stops using explicit methods for these problems.

```python
out = rk.stability_limit_is_where_the_solver_fails(name="rk4", lam=-50.0)
print(f"predicted limit for rk4: lam*h < {out['predicted_limit']:.6f}")
print(f"{'steps':>8}{'lam*h':>12}{'final value':>16}{'stable':>9}")
for s, lh, v, ok in zip(out["steps"], out["lam_h"], out["final_value"], out["stable"]):
    print(f"{s:>8}{lh:>12.4f}{v:>16.4e}{str(ok):>9}")
print(f"\nlargest stable lam*h observed: {out['largest_observed_stable_lam_h']:.4f}, "
      f"agrees with the prediction: {out['agrees']}")
```

*Output:*

```text
predicted limit for rk4: lam*h < 2.785294
   steps       lam*h     final value   stable
      13      3.8462      1.2101e+08    False
      14      3.5714      4.6217e+06    False
      15      3.3333      1.3086e+05    False
      16      3.1250      2.8805e+03    False
      17      2.9412      5.1968e+01    False
      18      2.7778      8.1542e-01     True
      19      2.6316      1.1885e-02     True
      20      2.5000      1.7273e-04     True
      21      2.3810      2.6935e-06     True
      22      2.2727      4.8402e-08     True

largest stable lam*h observed: 2.7778, agrees with the prediction: True
```

The boundary is exactly where the polynomial says it is.

## 8. Exercises

**Level 1, understanding**

1.1 Write the Butcher tableau of Heun's method and of the classical RK4, and identify which
entries make each explicit.

1.2 Explain why an explicit method's $A$ must be strictly lower triangular and what changes if a
diagonal entry is nonzero.

1.3 Say what a Runge-Kutta method reduces to when $f$ depends only on $t$, and name the rule RK4
reduces to.

1.4 State the four order conditions through order 3 in terms of $A$, $b$, $c$.

1.5 Explain why comparing two methods per step rather than per evaluation is misleading, with a
worked example.

**Level 2, derivation**

2.1 Derive the two order conditions for a two stage method in full, including the terms you
discarded, and identify the leading error term of the family.

2.2 Show that Ralston's $\alpha = 2/3$ minimises the leading error coefficient over the family,
and compute the ratio of its constant to Heun's.

2.3 Derive the third order conditions for a three stage method and find the general solution.

2.4 Show that $c_i = \sum_j a_{ij}$ is necessary for order 2 and explain the invariance it
expresses.

2.5 Show that an explicit $s$ stage method has a stability polynomial of degree $s$, and that for
$s \le 4$ a method of order $s$ has $R(z) = \sum_{j=0}^{s} z^j/j!$ exactly.

**Level 3, computational**

3.1 Implement `step_from(A, b, c)` for a general explicit tableau and verify it reproduces Euler,
Heun and RK4.

3.2 Search the two parameter family of three stage third order methods numerically for the one
that minimises the fourth order error terms.

3.3 Implement an implicit Runge-Kutta step for a general $A$ using Newton's method for the stage
equations, and verify the two stage Gauss method has order 4.

3.4 Implement the order conditions in exact rational arithmetic up to order 6 and use them to
check a published order 5 tableau.

3.5 Measure the stability limit of every tableau in `TABLEAUX` by bisection and compare against
the published values.

**Level 4, experimental**

4.1 Measure the equal cost comparison of section 6 on a problem whose solution has a
discontinuous fourth derivative, and see whether RK4 still wins.

4.2 Measure the crossover budget at which RK4 overtakes Heun, as a function of the accuracy
wanted, over four decades of tolerance.

4.3 Measure how the fitted order of RK4 degrades as the problem's smoothness is reduced, using
$y' = |t - 1|^{a} $ for a range of $a$.

**Level 5, advanced**

5.1 **The order 5 barrier.** Show that no explicit 5 stage method has order 5, by counting order
conditions against free parameters and then explaining why the count alone is not a proof.

5.2 **Butcher's group.** The composition of two Runge-Kutta methods is another one. Work out the
composition rule on the tableaux and say what structure it gives the set of methods.

5.3 **The equivalence with Taylor methods.** Every Runge-Kutta method of order $p$ reproduces the
same Taylor expansion as the Taylor method of order $p$. Say what is therefore being traded, and
in what units.

## 9. Key takeaways

- **The second order methods are a one parameter family.** Midpoint, Heun and Ralston are choices
  of $\alpha$, not different ideas, and they differ by a constant rather than an order.

- **A Butcher tableau is the method.** Checking one against the order conditions in exact rational
  arithmetic gives an exact yes or no; floating point arithmetic cannot.

- **Every explicit Runge-Kutta method is a quadrature rule in disguise**, and reduces to one when
  $f$ does not depend on $y$.

- **Stages equal order up to 4 and then stop.** Order 5 needs 6 stages, which is why RK4 is the
  default and why anything higher has to justify itself.

- **Compare at equal evaluation count.** Per step, RK4 looks four times more expensive than Euler
  and it is; per evaluation it wins by orders of magnitude.

- **At the stability limit the ranking reverses.** For $y' = -1000y$ the stability limits span
  1.39 and the stage counts span 4, so Euler costs 500 evaluations and RK4 costs 1440. The lowest
  order method is the cheapest. That is not an accuracy statement, and lesson 72 is the answer.

## Where this goes next

Every method so far uses a fixed step. Lesson 70 gets an error estimate for free by running two
tableaux that share their stages, and uses it to choose the step. Lesson 71 goes the other way and
reuses **past** steps instead of extra stages, which costs one evaluation per step and brings its
own kind of instability. Lesson 72 comes back to section 7 and explains it properly.
