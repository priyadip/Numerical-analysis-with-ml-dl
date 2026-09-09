# 81. Hyperbolic Equations and the CFL Condition

**Part 11: Partial Differential Equations**

## Learning objectives

By the end of this lesson you will be able to:

1. Discretize the wave equation, including the starting step the recurrence cannot supply itself.
2. Derive the Courant condition two ways: from the amplification roots and from domains of
   dependence.
3. Say why a Courant violation is a different failure from a parabolic instability, and demonstrate
   it with an exact zero.
4. Explain why the explicit scheme is exactly right at $\lambda = 1$ and only there.
5. Measure numerical dispersion and say what unconditional stability does and does not buy for a
   wave problem.

## Prerequisites

Lesson 77 (classification, and why a hyperbolic problem needs two initial conditions). Lesson 78
(amplification factors and mesh ratios). Lesson 71 (root conditions for three level recurrences,
which is what the stability analysis here is). Lesson 22 (the Thomas algorithm, for the implicit
scheme).

---

## 1. The equation and the scheme

$$
u_{tt} = c^2 u_{xx},
$$

with an initial shape **and** an initial velocity, because the equation is second order in time.
Difference both sides:

$$
u_j^{n+1} = 2u_j^n - u_j^{n-1} + \lambda^2\left(u_{j+1}^n - 2u_j^n + u_{j-1}^n\right),
\qquad \lambda = \frac{ck}{h}.
$$

$\lambda$ is the **Courant number**, the ratio of the distance the scheme's stencil reaches in one
step to the distance the wave actually travels. Everything in this lesson is about that ratio.

The recurrence needs **two** levels to start and the initial data supplies one. Getting the second
one right matters more than it looks.

```python
from nalib import hyperbolic as hy

out = hy.the_first_step_sets_the_order()
print(f"{'start':>7}{'h':>10}{'error':>13}{'ratio':>9}")
for row in out["rows"]:
    prev = None
    for h, e in zip(row["h"], row["error"]):
        shown = "" if prev is None else f"{prev / e:.2f}"
        print(f"{row['start_order']:>7}{h:>10.5f}{e:>13.3e}{shown:>9}")
        prev = e
print(f"\nfitted orders: {out['first_order_start']:.3f} and {out['second_order_start']:.3f}")
print(f"at the finest grid the second order start is "
      f"{out['gain_at_the_finest']:.0f} times more accurate")

assert out["one_row_sets_the_order"]
```

$u^1 = u^0 + kg$ is the obvious choice and it is first order, and **the second order interior
scheme cannot recover what the first step threw away**: the fitted order is 1.003. Taylor's theorem
asks for one more term, and the equation supplies it for free:

$$
u^1 = u^0 + kg + \tfrac{k^2}{2}u_{tt} = u^0 + kg + \tfrac{\lambda^2}{2}\delta^2u^0 .
$$

That gives 2.001 and a factor of 1082 at the finest grid. This is the third time in this part that
**one row, computed once, sets the order of everything**: lesson 77's Neumann boundary and lesson
71's multistep starting values are the same finding.

### 1.1 The whole solver, from scratch

Three lines of arithmetic and one starting step. Writing it out makes the domain of dependence
argument of section 2.2 visible: the new value at $j$ reads $j-1$, $j$ and $j+1$, so after $n$
steps it has reached exactly $n$ grid points either side and no further.

```python
import numpy as np

def wave_from_scratch(shape, velocity, lam, k, steps, left, right, start_order=2):
    """March u_tt = c^2 u_xx on a uniform grid, keeping only the current two levels."""
    old = np.asarray(shape, dtype=float).copy()
    now = old + k * np.asarray(velocity, dtype=float)
    if start_order == 2:
        now[1:-1] += 0.5 * lam ** 2 * (old[2:] - 2.0 * old[1:-1] + old[:-2])
    now[0], now[-1] = left(k), right(k)
    for step in range(1, steps):
        new = np.empty_like(now)
        new[1:-1] = (2.0 * now[1:-1] - old[1:-1]
                     + lam ** 2 * (now[2:] - 2.0 * now[1:-1] + now[:-2]))
        new[0], new[-1] = left((step + 1) * k), right((step + 1) * k)
        old, now = now, new
    return now

problem = hy.standing_wave(mode=2)
for n, lam in ((41, 0.5), (81, 0.8), (161, 0.95)):
    x = np.linspace(problem["a"], problem["b"], n)
    h = float(x[1] - x[0])
    k = lam * h / problem["speed"]
    steps = max(int(round(0.4 / k)), 2)
    mine = wave_from_scratch(problem["initial"](x), problem["velocity"](x), lam, k, steps,
                             problem["left"], problem["right"])
    theirs = hy.solve(problem, n, steps, steps * k, theta=0.0)
    gap = float(np.max(np.abs(mine - theirs["u"])))
    print(f"n = {n:>4}, lam = {lam}: disagreement {gap:.3e}, error {theirs['error']:.3e}")
    assert gap < 1e-12

# the reach after n steps is exactly n grid points either side, which is section 2.2 in code.
# The grid only has to be wide enough that the disturbance never reaches its ends, so its size
# is fixed by the number of steps rather than written down.
shown = 5
centre = 2 * shown
probe = np.zeros(2 * centre + 1)
probe[centre] = 1.0
old_level, now_level = np.zeros_like(probe), probe.copy()
print()
for step in range(1, shown + 1):
    new_level = np.zeros_like(now_level)
    new_level[1:-1] = (2.0 * now_level[1:-1] - old_level[1:-1]
                       + 1.0 * (now_level[2:] - 2.0 * now_level[1:-1] + now_level[:-2]))
    old_level, now_level = now_level, new_level
    touched = np.flatnonzero(np.abs(now_level) > 0.0)
    print(f"after {step} step(s) the support runs from {int(touched[0]) - centre:+d} to "
          f"{int(touched[-1]) - centre:+d} grid points from the source")
    assert int(touched[0]) == centre - step and int(touched[-1]) == centre + step
```

## 2. Two derivations of one condition

### 2.1 From the roots

Substituting $u_j^n = g^n e^{ij\phi}$ gives a quadratic in $g$ whose **product of roots is exactly
1**. That is the signature of a wave problem: nothing decays, so stability means both roots
**on** the unit circle, and the instant one leaves it the other is inside. There is no margin, and
the condition comes out as $\lambda \le 1$.

### 2.2 From domains of dependence

The exact solution at $(x, t)$ depends on the initial data over $[x - ct,\, x + ct]$: that is
d'Alembert's formula. The scheme at $(x_j, t_n)$ depends on the data over $[x_j - nh,\, x_j + nh]$,
because a three point stencil reaches one grid point per step. Containing the true interval needs

$$
nh \ge c\,nk \iff \lambda \le 1 .
$$

**These are the same condition, and the second derivation says something the first does not.** A
parabolic instability means the scheme computes the wrong answer. A Courant violation means the
scheme **cannot see data the answer depends on**, and no amount of care in the arithmetic can fix
that.

```python
out = hy.changing_data_it_cannot_see()
print(f"Courant number {out['lam']}, {out['steps']} steps of k = {out['k']}")
print(f"  the scheme reaches   {out['numerical_reach']:.3f}")
print(f"  the true wave reaches {out['true_reach']:.3f}")
print(f"  bump placed at        {out['bump_at']:.3f}")
print(f"\nthe true answer changes by      {out['true_change']:.6f}")
print(f"the computed answer changes by  {out['computed_change']:.6f}")
print(f"\n{out['note']}")

assert out["the_scheme_did_not_notice"]
assert out["computed_change"] == 0.0
assert out["the_truth_did"]
```

Two runs, identical except for a bump placed on the characteristic through the target point, which
is inside the true domain of dependence and outside the numerical one. The true answer changes by
**0.5**. The computed answer changes by an **exact zero**: no arithmetic in the scheme ever read
those values. The run is also unstable, so its answer is nonsense, and the nonsense is bit for bit
identical with and without the bump.

## 3. Sharp is not the same as immediate

```python
out = hy.the_courant_condition()
for label, key, end in (("short run", "short_run", out["t_end"]),
                        ("eight times longer", "long_run", out["long_t_end"])):
    print(f"\n{label}, to t = {end}")
    print(f"{'lam':>8}{'steps':>8}{'largest root':>14}{'predicted':>13}{'error':>13}"
          f"{'blew up':>10}")
    for row in out[key]:
        print(f"{row['lam']:>8.3f}{row['steps']:>8}{row['largest_root']:>14.4f}"
              f"{'10^' + format(row['predicted_decades'], '.1f'):>13}"
              f"{row['error']:>13.3e}{str(row['blew_up']):>10}")
print(f"\nthe root leaves the circle exactly at 1: "
      f"{out['the_root_leaves_the_circle_at_one']}")
print(f"the prediction is right in all {out['rows_checked']} rows: "
      f"{out['the_prediction_is_right_in_every_row']}")
print(f"violations that failed in the long run: {out['violations_that_failed_in_the_long_run']}")
print(f"and one that still had not: {out['violations_that_did_not']}")

assert out["the_prediction_is_right_in_every_row"]
assert out["nothing_below_the_limit_blew_up"]
```

The limit is sharp, and the failure is not immediate. The mode that grows is the shortest one the
grid carries, and a smooth bump has **no measurable energy there**: its amplitude reads back as
machine epsilon, exactly as the single sine mode did in lesson 78. So the failure is again a
countdown from rounding, and at $\lambda = 1.001$, where the growth is $1.0936$ per step, 240 steps
are not enough to finish it.

The same prediction as in lessons 78 and 80, the seed times the growth to the power of the step
count, is right in all twelve rows, **including the two rows above the limit that do not blow up**.

## 4. The one step size that is exact

Put $\lambda = 1$ into the update and the $u_j^n$ terms cancel completely:

$$
u_j^{n+1} = u_{j+1}^n + u_{j-1}^n - u_j^{n-1}.
$$

That is d'Alembert's formula evaluated on the grid. The characteristics $x \pm ct$ pass **exactly
through grid points**, and the scheme follows them without approximating anything.

```python
out = hy.the_magic_step()
print(f"{'lam':>7}{'steps':>8}{'error':>14}")
for row in out["rows"]:
    print(f"{row['lam']:>7.2f}{row['steps']:>8}{row['error']:>14.3e}")
print(f"\nerror at lam = 1: {out['error_at_one']:.3e}")
print(f"smallest error elsewhere: {out['smallest_error_elsewhere']:.3e}")
print(f"gain: {out['gain']:.3e}x")

assert out["exact_to_rounding"]
```

$6\times10^{-16}$ against $1.5\times10^{-3}$ at $\lambda = 0.95$: a factor of $2\times10^{12}$, and
the first number is rounding rather than truncation. It is a real property and an unusable one in
general, because it needs a constant speed, a uniform grid and $k = h/c$ exactly. Any of those
failing brings the ordinary second order straight back.

## 5. Dispersion

Below the limit the scheme is stable, and stable is not the same as right. Its roots are
$e^{\pm i\omega k}$ with

$$
\sin\!\left(\frac{\omega k}{2}\right) = \lambda\sin\!\left(\frac{\phi}{2}\right),
$$

so a mode of phase $\phi$ travels at speed

$$
\frac{c_{\text{numerical}}}{c} = \frac{2}{\lambda\phi}\arcsin\!\left(\lambda\sin\frac{\phi}{2}\right).
$$

```python
out = hy.dispersion_is_worst_for_short_waves()
print(f"{'lam':>7}{'worst error':>14}{'at the shortest wave':>22}"
      f"{'at 10 points per wave':>24}")
for row in out["rows"]:
    print(f"{row['lam']:>7.2f}{row['worst_relative_error']:>14.4e}"
          f"{row['error_at_the_shortest']:>22.4e}"
          f"{row['error_at_ten_points_per_wave']:>24.3e}")
print(f"\nexact at lam = 1: {out['exact_at_lam_one']}")
print(f"every scheme is too slow, never too fast: {out['every_scheme_is_too_slow']}")
print(out["note"])

assert out["exact_at_lam_one"]
assert out["every_scheme_is_too_slow"]
```

Three things worth keeping.

**The error grows with the wavenumber.** At $\lambda = 0.5$ the shortest wave the grid carries
travels at two thirds of the right speed while a wave resolved by ten points is off by one per cent.
A scheme that looks fine on the long waves can put the short ones in completely the wrong place.

**Every stable scheme is too slow, never too fast.** The sign is the same at every $\lambda$ and
every $\phi$, so numerical dispersion always produces a trailing wake rather than a leading one.

**At $\lambda = 1$ the error is exactly zero at every wavenumber.** That is section 4 restated in
the frequency domain, and it is the sharper statement of the two.

```python
import numpy as np

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.0))
problem = hy.travelling_bump()
for lam, style in ((0.5, "--"), (1.0, "-")):
    h = (problem["b"] - problem["a"]) / 200
    k = lam * h / problem["speed"]
    steps = max(int(round(0.6 / k)), 2)
    run = hy.solve(problem, 201, steps, steps * k, theta=0.0)
    left.plot(run["x"], run["u"], style, lw=1.3, label=f"lam = {lam}")
left.plot(run["x"], run["exact"], ":", color="k", lw=1.2, label="exact")
left.set_xlim(0.0, 1.6); left.set_xlabel("x")
left.set_title("after 0.6 time units"); left.legend(fontsize=8)

phases = np.linspace(1e-6, np.pi, 300)
for row in hy.dispersion_is_worst_for_short_waves()["rows"]:
    right.plot(phases / np.pi, hy.numerical_wave_speed(row["lam"], phases), lw=1.3,
               label=f"lam = {row['lam']}")
right.axhline(1.0, color="k", lw=0.8)
right.set_xlabel("phase / pi"); right.set_ylabel("numerical speed / c")
right.set_ylim(0.6, 1.05); right.set_title("numerical wave speed")
right.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/81_dispersion.png", dpi=110); plt.close(fig)
print("saved ../figures/81_dispersion.png")
```

![Dispersion and the magic step](../figures/81_dispersion.png)

The left panel shows the wake: at $\lambda = 0.5$ the short wavelength content lags behind the
travelling pulse, and at $\lambda = 1$ the computed curve sits exactly on the exact one.

## 6. The implicit scheme, and what unconditional stability buys

Average the space difference over the three time levels with weights
$(\theta, 1 - 2\theta, \theta)$. The scheme becomes implicit for $\theta > 0$, one tridiagonal
solve per step, and it is unconditionally stable exactly when $\theta \ge 1/4$.

```python
out = hy.the_implicit_threshold_is_a_quarter()
print(f"{'theta':>8}   largest root at lam =")
print(f"{'':>8}" + "".join(f"{lam:>12.1f}" for lam in out["rows"][0]["roots"]))
for row in out["rows"]:
    values = "".join(f"{v:>12.4f}" for v in row["roots"].values())
    mark = "  unconditional" if row["stable_everywhere"] else ""
    print(f"{row['theta']:>8.2f}{values}{mark}")
print(f"\nthe threshold is exactly {out['threshold']}: "
      f"{out['unconditional_above_a_quarter']}")

assert out["unconditional_above_a_quarter"]
```

The boundary lands **exactly between 0.24 and 0.25**, which is the kind of statement worth
measuring rather than quoting: at $\theta = 0.24$ the root at $\lambda = 100$ is $1.4992$, and at
$\theta = 0.25$ it is $1.0000$.

Now the question that matters: what is the unconditional stability worth?

```python
out = hy.the_implicit_scheme_buys_stability_and_pays_in_phase()
print(f"{'lam':>8}{'steps':>8}{'largest root':>14}{'error':>14}")
for row in out["rows"]:
    print(f"{row['lam']:>8.1f}{row['steps']:>8}{row['largest_root']:>14.6f}"
          f"{row['error']:>14.4e}")
print(f"\nevery run stable: {out['every_run_is_stable']}, "
      f"every root on the circle: {out['every_root_is_on_the_circle']}")
print(f"the error spans a factor of {out['error_span']:.0f} across the sweep")
print(out["note"])

assert out["every_run_is_stable"]
assert out["the_error_grows_with_the_step"]
```

Every root sits at exactly $1.000000$, so nothing ever grows, and the error over one period grows
by a factor of **6930** as $\lambda$ goes from 0.5 to 8.

**That is the opposite of the parabolic answer.** For a parabolic problem a large stable step is
worth having, because the solution is decaying and the error decays with it. For a wave problem
nothing decays, phase errors accumulate over every step, and the extra step buys speed at the
direct cost of accuracy. Unconditional stability is not a licence to take big steps; it is a
licence to choose the step for accuracy instead of for stability.

## 7. Exercises

**Level 1, understanding**

1.1 Say why the wave equation needs two initial conditions and the heat equation needs one.

1.2 State the Courant condition and give both derivations in one sentence each.

1.3 Explain why the product of the two amplification roots being 1 makes the limit sharp.

1.4 Explain why the scheme is exact at $\lambda = 1$.

1.5 Say what unconditional stability buys for a parabolic problem and what it buys for a
hyperbolic one.

**Level 2, derivation**

2.1 Derive the amplification quadratic and show its roots multiply to 1.

2.2 Derive the condition $\lambda \le 1$ from the requirement that both roots lie on the unit
circle.

2.3 Derive the second order starting step from Taylor's theorem and the equation.

2.4 Derive the numerical wave speed formula and show it equals $c$ for every $\phi$ at
$\lambda = 1$.

2.5 Derive the condition $\theta \ge 1/4$ for the weighted implicit scheme.

**Level 3, computational**

3.1 Implement the wave equation with a free end, $u_x(b, t) = 0$, using a ghost point, and confirm
second order.

3.2 Implement the first order advection equation with upwind, Lax-Friedrichs and Lax-Wendroff, and
compare their CFL conditions and their dispersion.

3.3 Implement a variable wave speed $c(x)$ and find the step restriction it imposes.

3.4 Implement the two dimensional wave equation and find its Courant condition.

3.5 Implement a scheme on a non uniform grid and measure whether the $\lambda = 1$ exactness
survives anywhere.

**Level 4, experimental**

4.1 Measure how many steps a violated run survives as a function of $\lambda$ and of the roughness
of the data, and compare against the seed and growth prediction.

4.2 Measure the wake in section 5 quantitatively: fit the position error of the pulse against time
and against $\lambda$.

4.3 Measure the energy $\tfrac12\int(u_t^2 + c^2u_x^2)$ over a long run for the explicit and the
implicit scheme, and say which conserves it.

**Level 5, advanced**

5.1 **The CFL condition is necessary, not sufficient.** Give a scheme that satisfies the domain of
dependence requirement and is still unstable, and explain what the missing ingredient is.

5.2 **Group velocity.** Derive the numerical group velocity $d\omega/d\kappa$ and show it can be
**negative** for short waves, so a wave packet travels backwards.

5.3 **Conservation.** Show that the explicit scheme conserves a discrete energy exactly when
$\lambda \le 1$, and identify what that energy becomes at $\lambda = 1$.

## 8. Key takeaways

- **The starting step sets the order.** $u^1 = u^0 + kg$ fits 1.003; adding
  $\tfrac{\lambda^2}{2}\delta^2u^0$ fits 2.001 and is 1082 times more accurate at the finest grid.

- **The Courant condition has two derivations and the second one is the important one.** From the
  roots it is $\lambda \le 1$; from domains of dependence it is the requirement that the scheme
  can **see** the data the answer depends on.

- **A Courant violation is not an accuracy failure.** Moving a bump the true answer depends on
  changes the true value by 0.5 and the computed value by an **exact zero**.

- **Sharp is not the same as immediate.** The root leaves the circle exactly at 1, and at
  $\lambda = 1.001$ a run of 240 steps is still clean, because a smooth bump seeds the growing
  mode only at machine epsilon.

- **At $\lambda = 1$ the scheme is exact**, to $6\times10^{-16}$ against $1.5\times10^{-3}$ next
  door, because the update is d'Alembert's formula on the grid.

- **Every stable scheme propagates short waves too slowly**, never too fast, so dispersion always
  makes a trailing wake. At $\lambda = 0.5$ the shortest wave travels at two thirds speed.

- **The implicit threshold is exactly $\theta = 1/4$**, with the root at $\lambda = 100$ going from
  1.4992 to 1.0000 between $0.24$ and $0.25$.

- **Unconditional stability buys nothing for a wave problem.** Every root sits on the unit circle
  and the error still grows by a factor of 6930 across the sweep, because nothing decays and phase
  errors accumulate.

## Where this goes next

Lesson 82 drops the time variable. An elliptic problem has no direction to march in, so every
unknown is coupled to every other one and the whole thing is one large sparse linear system. That
turns the subject back into Parts 3 and 4: the discretization is the easy half, and solving what it
produces is the rest of the work.
