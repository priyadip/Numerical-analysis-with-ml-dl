# 80. Multidimensional Parabolic Problems and ADI

**Part 11: Partial Differential Equations**

## Learning objectives

By the end of this lesson you will be able to:

1. Say what changes about implicit schemes when a second space dimension is added, and put a number
   on it.
2. Derive the explicit stability limit in any number of dimensions.
3. Build the alternating direction implicit method and show its stability in one line.
4. Say what the intermediate array actually is, and measure what assuming otherwise costs.
5. Compare methods at equal accuracy in two dimensions, where the step limit bites much harder.

## Prerequisites

Lesson 78 (the weighted family, mesh ratios and growth factors). Lesson 79 (von Neumann analysis
and the matrix method). Lesson 22 (the Thomas algorithm, used once per line here). Lesson 21
(banded factorization and the cost of a bandwidth).

---

## 1. What breaks in two dimensions

The equation is

$$
u_t = \alpha\left(u_{xx} + u_{yy}\right)
$$

on a rectangle. Every scheme of lesson 78 still applies, and one thing about them changes
completely: **the implicit ones stop being cheap.**

In one dimension the implicit matrix is tridiagonal, so the Thomas algorithm solves it in $O(n)$
and an implicit step costs a small constant times an explicit one. In two dimensions the unknowns
are a grid. Order them row by row and the matrix has bandwidth $m$, the number of unknowns per
row, so a banded factorization costs about $N m^2 = m^4$ for $N = m^2$ unknowns.

```python
from nalib import adi
import numpy as np

print(f"{'points':>8}{'unknowns':>10}{'bandwidth':>11}{'banded ops':>15}{'ADI ops':>13}"
      f"{'ratio':>9}")
out = adi.the_cost_of_a_full_solve()
for row in out["rows"]:
    print(f"{row['points']:>8}{row['unknowns']:>10}{row['bandwidth']:>11}"
          f"{row['banded_operations']:>15.3e}{row['adi_operations']:>13.3e}"
          f"{row['ratio']:>9.1f}")
print(f"\nfitted exponents in the unknowns per side: banded {out['banded_exponent']:.3f}, "
      f"ADI {out['adi_exponent']:.3f}")
print(f"fitted against the point count instead: "
      f"{out['banded_exponent_against_the_point_count']:.3f} and "
      f"{out['adi_exponent_against_the_point_count']:.3f}")

assert out["two_orders_apart"]
```

The exponents come out at **4.000** and **2.000**, exactly. Two orders in $m$ per step, and the
gap has already reached 60 by a 33 by 33 grid.

Notice the second fit. Against the number of grid **points** the same data reads 4.57 and 2.28,
because the point count and the unknown count differ by the two boundary rows and that offset is
not negligible over this range. Fitting an exponent means choosing the variable it is an exponent
in, and here the right one is the number of unknowns.

## 2. The explicit limit tightens with the dimension

Substituting one Fourier mode gives

$$
g = 1 - 4r_x\sin^2(\phi_x/2) - 4r_y\sin^2(\phi_y/2),
$$

worst at $\phi_x = \phi_y = \pi$, so $\lvert g\rvert \le 1$ needs

$$
r_x + r_y \le \tfrac12 .
$$

On a square grid that is $r \le 1/4$, half of the one dimensional limit. In $d$ dimensions each
direction contributes $4r$ to the same sum, so the isotropic limit is $1/(2d)$: one half, one
quarter, one sixth.

```python
out = adi.the_explicit_limit_tightens()
print(f"limit in d dimensions: {out['in_d_dimensions']}")
for label, key in (("smooth data", "smooth_data"), ("with a seeded mode", "seeded_data")):
    print(f"\n{label}, to t = "
          f"{out['smooth_t_end'] if key == 'smooth_data' else out['seeded_t_end']}")
    print(f"{'r':>8}{'rx+ry':>9}{'steps':>8}{'growth':>10}{'predicted':>13}{'error':>12}"
          f"{'blew up':>10}")
    for row in out[key]:
        print(f"{row['r']:>8.3f}{row['rx_plus_ry']:>9.3f}{row['steps']:>8}"
              f"{row['worst_mode_growth']:>10.4f}"
              f"{'10^' + format(row['predicted_decades'], '.1f'):>13}"
              f"{row['error']:>12.3e}{str(row['blew_up']):>10}")
print(f"\nthe prediction is right in all {out['rows_checked']} rows: "
      f"{out['the_prediction_is_right_in_every_row']}")

assert out["the_prediction_is_right_in_every_row"]
assert out["the_smooth_data_hides_it"]
assert out["the_seeded_data_shows_it"]
```

The two sweeps repeat lesson 78's lesson in two dimensions. On the smooth test solution the mode
that grows has amplitude **zero**, so nothing blows up at any ratio tried, however far above the
limit. Seed that mode at $10^{-6}$ and run five times as long, and the crossing at $r = 1/4$ is
exact: $r = 0.25$ is fine, $r = 0.26$ gives an error of 1.58, $r = 0.3$ gives $4.9\times10^{18}$.

The predictions are quantitative, not just qualitative: at $r = 0.3$ the predicted amplitude is
$10^{19.0}$ and the measured error is $4.9\times10^{18}$.

## 3. Alternating direction implicit

Split one step into two half steps and be implicit in **one direction at a time**:

$$
\left(I - \tfrac{r_x}{2}\delta_x^2\right)u^{*} = \left(I + \tfrac{r_y}{2}\delta_y^2\right)u^{n},
$$

$$
\left(I - \tfrac{r_y}{2}\delta_y^2\right)u^{n+1} = \left(I + \tfrac{r_x}{2}\delta_x^2\right)u^{*}.
$$

The first half step is implicit in $x$ only, so it is a **separate tridiagonal system for each
line of constant $y$**, all with the same matrix. The second is the same with the roles swapped.
A whole step is $2m$ tridiagonal solves of length $m$, which is $O(m^2) = O(N)$.

Stability is one line. Substituting a mode gives a **product**,

$$
g = \frac{1 - 2r_y s_y}{1 + 2r_x s_x}\cdot\frac{1 - 2r_x s_x}{1 + 2r_y s_y},
$$

and each factor is a Mobius map of a nonnegative quantity into $[-1, 1]$, so each has modulus at
most 1 whatever the mesh ratio. The method is unconditionally stable, and nothing about that proof
depends on the size of $r$.

```python
out = adi.adi_is_unconditionally_stable()
print(f"{'r':>12}{'largest |g| ADI':>18}{'largest |g| explicit':>22}")
for row in out["rows"]:
    print(f"{row['r']:>12.0e}{row['largest_adi']:>18.6f}{row['largest_explicit']:>22.1f}")
print(f"\nADI stable at every ratio: {out['adi_is_stable_at_every_ratio']}")
print(out["note"])

assert out["adi_is_stable_at_every_ratio"]
assert out["explicit_is_not"]
```

At $r = 10^6$ the explicit factor is $8\times10^{6}$ and ADI's is exactly $1.000000$.

### 3.1 One step, from scratch

The whole method is two loops of tridiagonal solves. Writing it out shows that the second loop is
the first one transposed, which is the only thing the name "alternating direction" means.

```python
def thomas_line(sub, diag, sup, rhs):
    """Tridiagonal solve, lesson 22, one line of the grid at a time."""
    n = diag.size
    c, d = np.empty(n), np.empty(n)
    c[0], d[0] = sup[0] / diag[0], rhs[0] / diag[0]
    for i in range(1, n):
        pivot = diag[i] - sub[i] * c[i - 1]
        c[i] = sup[i] / pivot if i + 1 < n else 0.0
        d[i] = (rhs[i] - sub[i] * d[i - 1]) / pivot
    out = np.empty(n)
    out[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        out[i] = d[i] - c[i] * out[i + 1]
    return out

def implicit_sweep(rhs, c, ends_low, ends_high):
    """Solve (I - c D2) x = rhs along the first axis, for every line of the second."""
    m = rhs.shape[0]
    sub, diag, sup = np.full(m, -c), np.full(m, 1.0 + 2.0 * c), np.full(m, -c)
    out = np.empty_like(rhs)
    for line in range(rhs.shape[1]):
        column = rhs[:, line].copy()
        column[0] += c * ends_low[line]
        column[-1] += c * ends_high[line]
        out[:, line] = thomas_line(sub, diag, sup, column)
    return out

def peaceman_rachford_from_scratch(u, r, star, after):
    """One full step: implicit in x, then implicit in y."""
    inner = u[1:-1, 1:-1]
    yy = u[1:-1, 2:] - 2.0 * inner + u[1:-1, :-2]
    half = np.array(star, copy=True)
    half[1:-1, 1:-1] = implicit_sweep(inner + 0.5 * r * yy, 0.5 * r,
                                      half[0, 1:-1], half[-1, 1:-1])
    middle = half[1:-1, 1:-1]
    xx = half[2:, 1:-1] - 2.0 * middle + half[:-2, 1:-1]
    out = np.array(after, copy=True)
    out[1:-1, 1:-1] = implicit_sweep((middle + 0.5 * r * xx).T, 0.5 * r,
                                     out[1:-1, 0], out[1:-1, -1]).T
    return out

problem = adi.moving_boundary_problem()
mesh = adi.grid_of(problem, 21)
k = 0.01
r = problem["alpha"] * k / mesh["h"] ** 2
u = problem["exact"](mesh["x"], mesh["y"], 0.0)
after = adi.set_boundary(np.zeros_like(u), problem, mesh, k)
star = adi.set_boundary(np.zeros_like(u), problem, mesh, 0.5 * k)
mine = peaceman_rachford_from_scratch(u, r, star, after)
theirs, _ = adi.peaceman_rachford_step(u, r, r, star, after)
print(f"largest disagreement with nalib: {float(np.max(np.abs(mine - theirs))):.3e}")
print(f"largest error against the exact solution: "
      f"{float(np.max(np.abs(mine - problem['exact'](mesh['x'], mesh['y'], k)))):.3e}")
assert float(np.max(np.abs(mine - theirs))) < 1e-12
```

```python
out = adi.adi_is_second_order()
print(f"running at r = {out['ratio']}, which is "
      f"{out['ratio_over_the_explicit_limit']:.0f} times the explicit limit")
print(f"{'h':>10}{'k':>12}{'achieved r':>13}{'error':>13}{'ratio':>9}")
prev = None
for h, k, r, e in zip(out["h"], out["k"], out["achieved_ratio"], out["error"]):
    shown = "" if prev is None else f"{prev / e:.2f}"
    print(f"{h:>10.5f}{k:>12.3e}{r:>13.3f}{e:>13.3e}{shown:>9}")
    prev = e
print(f"\norder in h {out['order_in_h']:.3f}")

assert out["second_order"]
```

Every sweep here refines to the **same final time** on every grid. That is not a detail: the first
version of this measurement chose the step count by rounding, so the coarsest grid integrated to
$t = 0.31$ and the finest to $t = 0.048$, and the two errors being compared were errors of
different problems. Rounding up and dividing keeps the horizon exact at the cost of an achieved
ratio slightly below the requested one, which is why the third column is reported.

```python
fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.0))
problem = adi.separable_problem(modes=(2, 1))
run = adi.solve(problem, 41, 40, 0.02, scheme="adi")
image = left.contourf(run["x"], run["y"], run["u"], levels=14, cmap="RdBu_r")
left.set_xlabel("x"); left.set_ylabel("y")
left.set_title(f"ADI at t = 0.02, r = {run['r']:.2f}")
fig.colorbar(image, ax=left, shrink=0.85)

sizes = np.asarray([row["bandwidth"] for row in adi.the_cost_of_a_full_solve()["rows"]],
                   dtype=float)
banded = np.asarray([row["banded_operations"]
                     for row in adi.the_cost_of_a_full_solve()["rows"]])
cheap = np.asarray([row["adi_operations"]
                    for row in adi.the_cost_of_a_full_solve()["rows"]])
right.loglog(sizes, banded, "o-", lw=1.3, label="banded Crank-Nicolson")
right.loglog(sizes, cheap, "s-", lw=1.3, label="ADI")
right.loglog(sizes, banded[0] * (sizes / sizes[0]) ** 4, ":", color="k", lw=1.0,
             label="m^4")
right.loglog(sizes, cheap[0] * (sizes / sizes[0]) ** 2, "--", color="k", lw=1.0,
             label="m^2")
right.set_xlabel("unknowns per side"); right.set_ylabel("operations per step")
right.set_title("cost of one step"); right.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/80_adi.png", dpi=110); plt.close(fig)
print("saved ../figures/80_adi.png")
```

![An ADI solution and the cost of a step](../figures/80_adi.png)

## 4. What the intermediate array is, and what assuming otherwise costs

$u^{*}$ is not the solution at $t + k/2$. Eliminating it between the two half steps shows what it
has to be on the boundary:

$$
u^{*} = \tfrac12\left[\left(I - \tfrac{r_y}{2}\delta_y^2\right)u^{n+1}
+ \left(I + \tfrac{r_y}{2}\delta_y^2\right)u^{n}\right].
$$

The standard warning is that using $u(t + k/2)$ instead drops the method to first order. That is
worth measuring rather than repeating.

```python
out = adi.the_half_step_boundary_costs_less_than_advertised()
print(f"{'problem':>18}{'boundary moves':>16}{'consistent':>12}{'naive':>9}"
      f"{'naive / consistent':>20}")
for row in out["rows"]:
    print(f"{row['problem']:>18}{str(row['boundary_moves']):>16}"
          f"{row['consistent']['order']:>12.3f}{row['naive']['order']:>9.3f}"
          f"{row['ratio_at_the_finest']:>20.6f}")
print(f"\nthe gap between the two candidate boundary values:")
print(f"  exponent in k: {out['exponent_in_k_over_the_small_steps']:.3f} over the small steps, "
      f"{out['the_whole_k_sweep_would_say']:.3f} over the whole sweep")
print(f"  exponent in h: {out['exponent_in_h']:.3f}")
print(f"\n{out['note']}")

assert out["both_treatments_are_second_order"]
assert out["the_gap_is_k_squared_h_squared"]
```

**The warning could not be reproduced.** Both treatments fit 1.946 on a problem with genuinely
moving boundary data, and their answers agree to six digits.

Finding out why is more useful than repeating the warning. Expand both candidates for a solution
of the equation, writing $A$ and $B$ for the values at the two ends of the step:

$$
u(t + \tfrac{k}{2}) = \tfrac{A+B}{2} - \frac{Ak^2}{8} + O(k^3),
\qquad
\tfrac{A+B}{2} + \tfrac{r\sigma}{4}(A - B) = \tfrac{A+B}{2} + \frac{Ak^2}{8} + O(k^3),
$$

where $\sigma$ is the discrete second difference, whose leading term $-\pi^2h^2$ supplies the
second $k^2/8$. **The two $O(k^2)$ terms have opposite signs and equal size, so they cancel**, and
what survives is the difference between $\sigma$ and $-\pi^2h^2$, which is $O(h^2)$.

Measured, the gap fits **1.97 in $k$** over the small steps and **1.97 in $h$**, so it is
$O(k^2h^2)$ and contributes $O(kh^2)$ to the answer over $1/k$ steps. That is smaller than the
scheme's own $O(k^2 + h^2)$ on any refinement path, so it cannot change the order.

The cancellation used the fact that the boundary data solves the equation. **A manufactured
solution test always has that property, so this experiment cannot show the effect the warning is
about.** The warning applies to prescribed boundary data that is not compatible with the interior
equation, and the measurement here is a warning about the test rather than a refutation of the
advice.

## 5. Cost at equal accuracy

```python
out = adi.adi_against_explicit_at_equal_accuracy()
print(f"target error {out['target']:.0e}")
print(f"{'scheme':>10}{'points':>8}{'steps':>8}{'r':>9}{'error':>12}{'updates':>10}")
for row in out["rows"]:
    print(f"{row['scheme']:>10}{row['points']:>8}{row['steps']:>8}{row['r']:>9.3f}"
          f"{row['error']:>12.3e}{row['updates']:>10}")
print(f"\nADI is cheaper by {out['explicit_over_adi']:.2f}x in updates and takes "
      f"{out['explicit_steps_over_adi_steps']:.0f}x fewer steps")
print(out["note"])

assert out["adi_is_cheaper"]
```

At this fairly loose tolerance ADI is 2.6 times cheaper in unknown updates and takes 11 times
fewer steps. The step ratio is the number that keeps growing: the explicit scheme's step is capped
at $r \le 1/4$, so halving $h$ quadruples its step count, while ADI can hold $r$ at whatever the
accuracy needs. The gap in updates is smaller than the gap in steps because a tridiagonal solve
costs several times an explicit update, and counting that honestly is what makes the comparison
worth anything.

## 6. Exercises

**Level 1, understanding**

1.1 Explain why a two dimensional implicit step is not $O(N)$ by direct solution, and give the
bandwidth.

1.2 State the explicit stability limit in $d$ dimensions and say where the $2d$ comes from.

1.3 Explain in one sentence why ADI is unconditionally stable.

1.4 Say what $u^{*}$ is and what it is not.

1.5 Explain why fitting a cost exponent against the point count gave 4.57 instead of 4.

**Level 2, derivation**

2.1 Derive the two dimensional growth factor and the condition $r_x + r_y \le 1/2$.

2.2 Derive the ADI growth factor and show it is a product of two factors each bounded by 1.

2.3 Eliminate $u^{*}$ between the two half steps and recover the full step operator, then show it
is second order in $k$.

2.4 Derive the boundary condition for $u^{*}$ and show it differs from $u(t + k/2)$ by
$O(k^2h^2)$ for a solution of the equation.

2.5 Count the operations of one ADI step and of one banded Crank-Nicolson step, and get the
exponents 2 and 4.

**Level 3, computational**

3.1 Implement the Douglas-Rachford splitting and compare its order against Peaceman-Rachford.

3.2 Implement ADI on a rectangle with $h_x \ne h_y$ and confirm the stability is still
unconditional.

3.3 Implement ADI with a Neumann boundary on one side using lesson 77's ghost point, and measure
the order.

3.4 Implement the three dimensional version, which needs three half steps, and measure whether it
is still second order.

3.5 Implement the two dimensional Crank-Nicolson system as a sparse solve with conjugate gradients
from lesson 24, and compare its cost against ADI at equal accuracy.

**Level 4, experimental**

4.1 Measure the ADI splitting error directly by comparing against a fine full solve, and fit its
order in $k$.

4.2 Measure how the equal accuracy comparison of section 5 changes across four decades of
tolerance, and fit the exponents.

4.3 Measure the effect of the ADI sweep direction by alternating $x$ first and $y$ first on
successive steps, and say whether it changes anything.

**Level 5, advanced**

5.1 **Why ADI is second order and Douglas-Rachford is not.** Expand both splittings as operator
products and identify the term that survives.

5.2 **ADI in three dimensions.** Show that the natural three way Peaceman-Rachford splitting is
only first order, and describe what has to change.

5.3 **Where the half step boundary warning is real.** Construct a problem with prescribed boundary
data that is not compatible with the interior equation, and measure whether the naive treatment
loses an order there.

## 7. Key takeaways

- **Implicit stops being cheap in two dimensions.** The bandwidth is the unknown count per side,
  so a banded solve is $O(m^4)$ against ADI's $O(m^2)$. Measured exponents: **4.000 and 2.000**.

- **Fit against the right variable.** The same data fitted against the point count reads 4.57 and
  2.28, because two boundary rows are not negligible over the range measured.

- **The explicit limit is $1/(2d)$.** One half, one quarter, one sixth, and on a square grid in two
  dimensions the crossing at $r = 1/4$ is exact in the runs.

- **A scheme amplifies only what is in the data**, in two dimensions as in one. Nothing blew up on
  the smooth solution at any ratio; seeding the worst mode at $10^{-6}$ made $r = 0.26$ fail and
  $r = 0.3$ reach $5\times10^{18}$.

- **ADI's stability proof is one line**, because its growth factor is a product of two factors each
  bounded by 1. At $r = 10^6$ its worst factor is $1.000000$.

- **Hold the final time fixed in a refinement sweep.** The first version of the order measurement
  ran the coarsest grid to $t = 0.31$ and the finest to $t = 0.048$, which compared errors of
  different problems.

- **The half step boundary warning could not be reproduced.** Both treatments fit 1.946 and agree
  to six digits, because the two candidate values cancel to $O(k^2h^2)$ for anything that solves
  the equation. The warning is about incompatible prescribed data, and a manufactured solution
  test cannot see it.

- **ADI wins on steps first and on updates second.** At the tolerance measured it is 11 times
  fewer steps and 2.6 times fewer updates, and the first number is the one that keeps growing.

## Where this goes next

Lesson 81 changes the type. Hyperbolic problems have a finite speed of propagation, so their step
restriction is a completely different animal: the CFL condition compares the numerical domain of
dependence against the true one, and violating it is not slow convergence but a scheme that cannot
possibly get the right answer.
