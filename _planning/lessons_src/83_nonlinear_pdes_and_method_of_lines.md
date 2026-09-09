# 83. Nonlinear PDEs and the Method of Lines

**Part 11: Partial Differential Equations**

## Learning objectives

By the end of this lesson you will be able to:

1. Turn a nonlinear elliptic problem into a nonlinear system and solve it with Newton's method.
2. Separate Newton's global phase from its quadratic one, and measure only the second.
3. Recognise a problem where the question is how many solutions there are, and find the turning
   point.
4. Use the method of lines to hand a PDE to Part 10, and see the two parts' stability conditions
   turn out to be the same one.
5. Say what an adaptive step controller is actually doing on a stiff semi-discrete system.

## Prerequisites

Lesson 13 (Newton for systems, the Jacobian and quadratic convergence). Lesson 82 (the discrete
elliptic operator and its sparsity). Lesson 72 (stability regions, A-stability and stiffness).
Lesson 70 (embedded pairs and the step control law). Lesson 78 (the explicit scheme, which turns
out to be the method of lines in disguise).

---

## 1. A nonlinear elliptic problem is a nonlinear system

Take

$$
-u'' + g(u) = f, \qquad u(a) = u(b) = 0 .
$$

Difference the second derivative and the discrete residual is

$$
F(u)_j = -\frac{u_{j+1} - 2u_j + u_{j-1}}{h^2} + g(u_j) - f_j ,
$$

whose Jacobian is the tridiagonal second difference matrix plus $\mathrm{diag}(g'(u))$. **Nothing
about Newton's method changes.** What changes is that the Jacobian is large and sparse, and that
sparsity is the whole reason the method is affordable: tridiagonal costs $O(m)$ to factor where a
dense matrix would cost $O(m^3)$.

```python
from nalib import nonlinearpde as nl
import numpy as np

out = nl.newton_is_quadratic()
print(f"{'step':>6}{'residual':>14}")
for k, value in enumerate(out["residual_history"]):
    print(f"{k:>6}{value:>14.3e}")
print(f"\nfitted exponent over the last {out['steps_in_the_tail_fit']} steps: "
      f"{out['exponent']:.3f}")
print(f"fitted over the whole history:  {out['exponent_over_the_whole_history']:.3f}")
print(f"the Jacobian is {100 * out['sparsity']:.1f}% zeros "
      f"({out['jacobian_nonzeros']} of {out['jacobian_entries']} entries)")

assert out["quadratic"]
assert out["the_residual_rises_before_it_falls"]
```

**Newton has two phases and only the second one is quadratic.** From a zero start on this problem
the first step overshoots badly and the residual **rises** from 40 to 1144 before it begins to
fall. Fitting the whole history gives 1.75, which measures the overshoot; fitting the last three
steps gives 1.95, which measures the method. Reporting the first as "Newton's convergence order"
is a common and avoidable mistake, and the fix is to say which steps were used.

### 1.1 The whole solve, from scratch

Newton on a PDE is four lines, and nothing in them mentions partial differential equations. The
only thing that makes it a PDE solver is what goes into the residual.

```python
def newton_on_a_pde(source, reaction, reaction_prime, points, a=0.0, b=1.0,
                    tol=1e-12, max_iter=40):
    """Solve -u'' + g(u) = f with Dirichlet zeros, by Newton on the discrete residual."""
    x = np.linspace(a, b, points)
    h = float(x[1] - x[0])
    inner = x[1:-1]
    m = inner.size
    rhs = source(inner)
    off = np.diag(np.full(m - 1, -1.0 / h ** 2), 1) + np.diag(np.full(m - 1, -1.0 / h ** 2), -1)
    u = np.zeros(m)
    history = []
    for _ in range(max_iter):
        padded = np.concatenate([[0.0], u, [0.0]])
        second = (padded[2:] - 2.0 * padded[1:-1] + padded[:-2]) / h ** 2
        residual = -second + reaction(u) - rhs
        history.append(float(np.max(np.abs(residual))))
        if history[-1] < tol:
            break
        jacobian = np.diag(2.0 / h ** 2 + reaction_prime(u)) + off
        u = u + np.linalg.solve(jacobian, -residual)
    return x, u, np.asarray(history)

problem = nl.cubic_problem(weight=30.0)
x, mine, history = newton_on_a_pde(problem["source"], problem["reaction"],
                                   problem["reaction_derivative"], 41)
theirs = nl.solve_nonlinear(problem, 41)
gap = float(np.max(np.abs(mine - theirs["u"])))
print(f"disagreement with nalib: {gap:.3e}")
print(f"steps: {history.size}, final residual {history[-1]:.3e}")
print(f"error against the exact solution: "
      f"{float(np.max(np.abs(mine - problem['exact'](x[1:-1])))):.3e}")
assert gap < 1e-10
```

The Jacobian here is built dense because it is easier to read that way, and at 39 unknowns it
costs nothing. At the sizes a real problem reaches, the ``np.diag`` calls are exactly what has to
go: the same four lines with a banded solve from lesson 22 in place of `np.linalg.solve` scale to
grids this version could not hold.

## 2. When the question is how many solutions there are

The Bratu problem,

$$
-u'' = \lambda e^{u}, \qquad u(0) = u(1) = 0,
$$

has **two** solutions below a critical $\lambda$, **one** at it, and **none** above. There is no
manufactured solution to measure an error against, and that is the point: the interesting question
is not how accurate the answer is but whether there is one.

Its closed form solution is
$u(x) = -2\log\left(\cosh\left(\tfrac{(x - 1/2)c}{2}\right)/\cosh\tfrac{c}{4}\right)$ with
$\lambda = c^2/(2\cosh^2(c/4))$, and the turning point is where that is largest.

```python
out = nl.the_bratu_problem_has_a_fold()
print(f"closed form fold: {out['closed_form_fold']:.9f}")
print(f"measured fold:    {out['measured_fold']:.9f}")
print(f"relative gap:     {out['relative_gap']:.3e}")
print(f"\n{'points':>8}{'measured fold':>18}{'gap':>14}")
for points, value in out["refinement"]:
    print(f"{points:>8}{value:>18.9f}{abs(value - out['closed_form_fold']):>14.3e}")
print(f"\nthe gap falls at order {out['gap_order_in_h']:.4f} in h")
print(f"solvable just below the fold: {out['solvable_below']}")
print(f"unsolvable just above it:     {out['unsolvable_above']}")

assert out["agrees_to_four_digits"]
assert out["the_gap_is_the_discretization_error"]
```

The measured turning point at 81 points is $3.5135$ against a closed form $3.5138307$, and the gap
is **not** an error in the method: it falls at order $2.0003$ as the grid refines, so it is the
discretization error of the second difference, arriving in an unfamiliar place.

```python
print(f"at lam = {out['probe']:.4f}, below the fold:")
print(f"  lower branch peak {out['lower_peak']:.4f}")
print(f"  upper branch peak {out['upper_peak']:.4f}")
print(f"\n{out['note']}")

assert out["two_branches_found"]
```

Both solutions exist at the same $\lambda$. The lower one is reachable from a zero start; the upper
one needs continuation, walking $\lambda$ down from near the fold and carrying each solution
forward as the next guess.

**Newton failing above the fold is not a defect in Newton.** There is nothing to converge to, and
no initial guess, damping or tolerance changes that. It is the first problem in this course where
the right answer to "it does not converge" is "there is no solution".

```python
import numpy as np

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.0))
left.plot(out["lam"], out["peak"], lw=1.4)
left.axvline(out["closed_form_fold"], color="k", ls=":", lw=1.0)
left.set_xlabel("lambda"); left.set_ylabel("max u")
left.set_title("the lower branch, up to the fold")

limit_out = nl.the_step_limit_comes_from_the_ode_side()
system = nl.semi_discrete_heat(41)
right.plot(np.real(system["eigenvalues"]) * limit_out["rows"][-1]["largest_stable_step"],
           np.zeros_like(system["eigenvalues"]), "o", ms=3, label="h * eigenvalues")
theta = np.linspace(0.0, 2.0 * np.pi, 400)
right.plot(np.cos(theta) - 1.0, np.sin(theta), lw=1.2, color="k",
           label="Euler stability region")
right.set_aspect("equal"); right.set_xlim(-2.6, 0.6); right.set_ylim(-1.3, 1.3)
right.set_xlabel("Re"); right.set_ylabel("Im")
right.set_title("the step that just fits"); right.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/83_bratu.png", dpi=110); plt.close(fig)
print("saved ../figures/83_bratu.png")
```

![The Bratu branch and the step that just fits](../figures/83_bratu.png)

## 3. The method of lines

Discretize **space only**, and

$$
u_t = \alpha u_{xx} \quad\longrightarrow\quad u'(t) = \frac{\alpha}{h^2}T\,u(t),
$$

a system of ordinary differential equations. Every method of Part 10 now applies unchanged, and
the connection is not decorative.

```python
out = nl.the_step_limit_comes_from_the_ode_side()
print(f"{'points':>8}{'h':>9}{'worst eigenvalue':>19}{'-4/h^2':>13}"
      f"{'largest stable k':>19}{'r it allows':>14}")
for row in out["rows"]:
    print(f"{row['points']:>8}{row['h']:>9.5f}{row['worst_eigenvalue']:>19.2f}"
          f"{row['continuous_limit']:>13.2f}{row['largest_stable_step']:>19.4e}"
          f"{row['mesh_ratio_it_allows']:>14.6f}")
print(f"\nthe two derivations agree: {out['the_two_derivations_agree']}")
print(f"the gap between them falls at order {out['gap_order_in_h']:.3f} in h")

assert out["the_two_derivations_agree"]
assert out["the_gap_is_second_order"]
```

Euler's stability region reaches $-2$ along the negative real axis, and the most negative
eigenvalue of the semi-discrete operator is $-\frac{4\alpha}{h^2}\sin^2\frac{m\pi h}{2}$. Putting
the two together gives $k \le h^2/(2\alpha)$, which is **$r \le 1/2$**: lesson 78's condition,
derived here entirely from the ODE side with no Fourier substitution into any scheme.

They do not agree exactly, and the discrepancy is meaningful. The discrete eigenvalue is slightly
smaller in modulus than $4\alpha/h^2$, so the ODE route allows a step slightly **larger** than
$r = 1/2$, by a factor $1/\sin^2(m\pi h/2)$. That gap falls at order 2.011 in $h$, and it is the
same $O(h^2)$ difference between a discrete eigenvalue and its continuous limit that lesson 75
found in the resonance of a boundary value problem.

### 3.1 It really is the same scheme

```python
out = nl.lines_against_a_direct_scheme()
print(f"mesh ratio {out['mesh_ratio']:.3f}")
print(f"the direct scheme and lines-plus-Euler differ by "
      f"{out['they_are_the_same_scheme']:.1e}")
print()
print(f"{'':>22}{'time error':>14}{'total error':>14}")
print(f"{'lines with Euler':>22}{out['euler_time_error']:>14.3e}"
      f"{out['euler_total_error']:>14.3e}")
print(f"{'lines with RK4':>22}{out['rk4_time_error']:>14.3e}"
      f"{out['rk4_total_error']:>14.3e}")
print(f"{'space error alone':>22}{'':>14}{out['space_error']:>14.3e}")
print(f"\nRK4 beats Euler in time by {out['rk4_beats_euler_in_time_by']:.1e}x "
      f"and loses overall by {out['rk4_total_over_euler_total']:.2f}x")
print(out["note"])

assert out["identical_to_rounding"]
assert out["rk4_is_the_space_error"]
```

Space discretization plus Euler **is** lesson 78's explicit scheme, bit for bit: the two arrays
differ by an exact zero.

Replacing Euler by RK4 gives a scheme nobody derived by hand, fourth order in time and with a
larger stability region. On this run it is **less accurate overall**, and understanding that is
worth more than explaining it away.

There are two errors and they are not independent. The **space** error makes the semi-discrete
solution decay at $-9.8645$ instead of $-9.8696$, which is too slowly, so the answer ends up too
large. The **time** error of Euler makes $(1 + k\lambda)^n$ smaller than $e^{k\lambda n}$, so it
decays too fast and ends up too small. They have opposite signs and partly cancel. RK4 removes the
time error entirely, from $4\times10^{-5}$ to $6\times10^{-16}$, and removes the cancellation with
it, leaving exactly the space error.

**Quoting either column alone would be misleading**, and this is the general shape of the trap: a
more accurate component can make a coupled answer worse.

## 4. What an adaptive solver is really doing

Hand the semi-discrete system to lesson 70's adaptive Runge-Kutta solver. It is told a tolerance
and nothing else; it knows no stability theory.

```python
out = nl.an_adaptive_solver_finds_the_limit_by_itself()
print(f"Euler's limit on this system:          {out['euler_limit']:.6e}")
print(f"Dormand-Prince's limit on this system: {out['dormand_prince_limit']:.6e}")
print()
print(f"{'tolerance':>12}{'accepted steps':>16}{'median step':>15}{'over the limit':>16}")
for row in out["rows"]:
    print(f"{row['tolerance']:>12.0e}{row['accepted']:>16}{row['median_step']:>15.6e}"
          f"{row['over_the_euler_limit']:>16.3f}")
print(f"\n{out['tolerance_span']:.0e} of tolerance moves the step by a factor of "
      f"{out['step_spread']:.3f}")
print(out["note"])

assert out["the_tolerance_barely_matters"]
assert out["it_lands_near_the_stability_limit"]
```

Six orders of magnitude of tolerance move the median step by **2 per cent**, and the step it
settles on is within 0.6 per cent of Dormand-Prince's own stability limit on this system.

The controller has no idea about stability. It does not need one: a step above the limit produces
an error estimate that fails the test, so the step is rejected and shrunk. **The step it converges
to is the stability limit, not an accuracy limit**, and the tolerance is nearly irrelevant to it.

That is the practical content of stiffness. An explicit solver on a stiff problem is not being
careful, it is being held, and the only escape is lesson 72's implicit methods.

## 5. A nonlinear time dependent problem

Fisher's equation,

$$
u_t = D u_{xx} + r\,u(1 - u),
$$

is diffusion plus logistic growth, and its solutions are travelling fronts. Nothing about the time
stepping changes because the problem is nonlinear: **the right hand side is a function, and Part
10's solvers only ever call it.**

The property worth checking is the front **speed**, not an error against a manufactured solution.
An error would test the discretization; the speed tests whether the behaviour the equation is
famous for survived it.

```python
out = nl.a_nonlinear_time_dependent_problem()
print(f"critical decay rate a* = sqrt(r/D) = {out['critical_decay']:.1f}, "
      f"minimum speed 2 sqrt(D r) = {out['textbook_minimum']:.4f}")
for row in out["rows"]:
    kind = "steep" if row["steep"] else "shallow"
    print(f"\ninitial decay a = {row['decay']:.0f} ({kind})")
    print(f"  predicted {row['predicted_speed']:.5f}, "
          f"measured {row['late_speed']:.5f}, "
          f"relative error {row['relative_error']:.2e}")
    print(f"  {'window':>14}{'speed':>11}{'c* - 3/(2 a* t)':>19}{'gap':>10}")
    for piece in row["windows"]:
        print(f"  {str(piece['window']):>14}{piece['speed']:>11.5f}"
              f"{piece['bramson']:>19.5f}{piece['relative_gap']:>10.4f}")

assert out["every_late_speed_matches_its_prediction"]
assert out["bramson_matches_every_window"]
```

Two findings, and the famous formula is in neither of them unqualified.

**The textbook speed $2\sqrt{Dr}$ is a minimum, not the answer.** A front whose initial profile
decays like $e^{-ax}$ is *pulled* by its own leading edge and travels at $Da + r/a$, as long as
that is above the minimum. For $a = 2$ here that is $0.52$ against a minimum of $0.2$: **wrong by a
factor of 2.6** if you quote the minimum.

**A steep front does reach the minimum, and only algebraically slowly.** Fitted over $t \in [2,5]$
the speed is $0.149$; only by $[20, 40]$ has it reached $0.195$. Bramson's correction says the
instantaneous speed is $c^{*} - 3/(2a^{*}t)$, and the measurement matches that to 4.9, 1.9, 0.25
and 0.22 per cent across the four windows. A run to $t = 4$ would have measured $0.14$ and
concluded the theory was wrong.

There is one more thing the sweep had to handle. The shallow front travels 20.8 units in the run,
and the domain is only 20 wide, so the last samples are of a front **stalled against the
boundary**. Leaving them in made the measured speed read $0.42$ instead of $0.52$: a measurement of
the domain rather than of the equation. Every sample after the front comes within 2 units of the
edge is dropped, and the run reports where it stopped.

## 6. Exercises

**Level 1, understanding**

1.1 Explain why a nonlinear elliptic problem gives a nonlinear system rather than a recurrence.

1.2 Say why fitting Newton's whole residual history gives 1.75 rather than 2.

1.3 Explain what happens to the Bratu problem above its critical $\lambda$.

1.4 State the method of lines in one sentence and say what it buys.

1.5 Explain why an adaptive solver's step barely changes when the tolerance changes by $10^6$.

**Level 2, derivation**

2.1 Derive the Jacobian of the discrete residual and show it is tridiagonal plus a diagonal.

2.2 Derive the Bratu closed form and the condition $c\tanh(c/4) = 4$ at the fold.

2.3 Derive the explicit step limit from Euler's stability region and the semi-discrete spectrum,
and get $r \le 1/2$.

2.4 Show that the space and time errors of section 3.1 have opposite signs.

2.5 Derive $c(a) = Da + r/a$ for a pulled Fisher front and find where it is minimised.

**Level 3, computational**

3.1 Implement pseudo-arclength continuation and follow the Bratu branch **around** the fold onto
the upper solution.

3.2 Implement the two dimensional Bratu problem and find its fold, which is a different number.

3.3 Implement the method of lines with an implicit BDF solver from lesson 71 and confirm the step
is no longer held by stability.

3.4 Implement a Newton-Krylov solver: Newton's method with the linear solve done by GMRES from
lesson 25 and the Jacobian applied matrix free.

3.5 Implement the Allen-Cahn equation $u_t = \varepsilon^2 u_{xx} + u - u^3$ and measure the
coarsening rate of its interfaces.

**Level 4, experimental**

4.1 Measure Newton's convergence order as a function of the strength of the nonlinearity, and find
where the global phase stops being visible.

4.2 Measure the step an adaptive solver settles on against the grid size, and confirm it follows
$h^2$ rather than the tolerance.

4.3 Measure the Fisher front speed for a range of initial decay rates spanning $a^{*}$, and plot
the measured speed against $\min(Da + r/a,\ 2\sqrt{Dr})$.

**Level 5, advanced**

5.1 **Why the fold is a fold.** Show that the Jacobian is singular at the turning point and that
this is what makes Newton fail there rather than merely nearby.

5.2 **Pulled fronts.** Derive Bramson's $-3/(2a^{*}t)$ correction from the linearised equation
ahead of the front, and say why the coefficient is universal.

5.3 **Splitting.** For a reaction-diffusion problem, compare solving the diffusion and the reaction
in alternate half steps against solving the coupled system, and identify the splitting error's
order.

## 7. Key takeaways

- **Newton has two phases.** The residual **rises** from 40 to 1144 before it falls, and the
  exponent reads 1.75 over the whole history and 1.95 over the last three steps. Say which steps
  you used.

- **Sparsity is what makes Newton affordable.** The Jacobian is tridiagonal and 92 per cent zeros,
  so a step costs $O(m)$ rather than $O(m^3)$.

- **Above the fold there is no solution.** The measured turning point is $3.5135$ against a closed
  form $3.5138307$, and the gap is the discretization error, falling at order $2.0003$.

- **The two stability conditions are the same condition.** Euler's stability region plus the
  semi-discrete spectrum gives $r \le 1/2$, matching lesson 78's Fourier derivation, with an
  $O(h^2)$ gap fitted at 2.011.

- **Lines plus Euler is the explicit scheme, exactly**, differing by a floating point zero.

- **A more accurate component can make the answer worse.** RK4 beats Euler in **time** error by
  $7\times10^{10}$ and loses in **total** error by a factor of 2, because Euler's time error was
  cancelling part of the space error.

- **An adaptive solver on a stiff problem is being held, not being careful.** Six decades of
  tolerance move its step by 2 per cent, and the step is within 0.6 per cent of the stability
  limit.

- **$2\sqrt{Dr}$ is a lower bound.** A shallow Fisher front travels at $Da + r/a = 0.52$, and
  quoting the minimum would be wrong by a factor of 2.6.

- **A steep front reaches the minimum algebraically slowly.** 0.149, 0.177, 0.190, 0.195 over four
  time windows, matching $c^{*} - 3/(2a^{*}t)$ to 0.2 per cent in the last one.

- **Watch the domain.** Samples taken after the front stalls against the boundary turned a speed of
  0.52 into 0.42, and dropping them is the difference between measuring the equation and measuring
  the box.

## Where this goes next

That completes Part 11. Part 12 changes the question from "solve this equation" to "find where this
function is smallest", and much of what the last two parts built comes straight across: Newton's
method with its two phases, the condition number deciding how fast an iteration converges, and the
same sparse linear algebra inside every step.
