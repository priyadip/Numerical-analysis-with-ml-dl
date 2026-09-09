# 73. Systems and Higher Order Equations

**Part 10: Ordinary Differential Equations**

## Learning objectives

By the end of this lesson you will be able to:

1. Reduce an equation of any order to a first order system, and check the reduction by its
   defining identity rather than by solving.
2. Apply every method of lessons 67 to 72 to a system without changing them.
3. Measure the conserved quantities of a mechanical system and see the solver lose them.
4. Measure stiffness **along a trajectory**, where it varies by orders of magnitude.
5. Recognise the three failures that are properties of the equations and not of the method:
   drift, chaos and a parameter crossing.

## Prerequisites

Lessons 67 to 72 (every method used here). Lesson 26 (eigenvalues, for the local stiffness).
Lesson 72 (the stiffness ratio and what it forces).

---

## 1. Reduction to first order

Any equation $y^{(m)} = g(t, y, y', \dots, y^{(m-1)})$ becomes a first order system by naming the
derivatives:

$$
u = (y, y', \dots, y^{(m-1)}), \qquad
u' = (u_2, u_3, \dots, u_m, g(t, u)).
$$

The first $m-1$ components of $u'$ are the last $m-1$ of $u$, shifted. That is the whole
transformation, and it is completely mechanical.

```python
from nalib import odesystems as od

def third_order(t, y, dy, d2y):
    """The equation y''' = -3y'' - 3y' - y, whose characteristic roots are all -1."""
    return -3.0 * d2y - 3.0 * dy - y

f = od.to_first_order(third_order, 3)
state = np.asarray([1.0, 2.0, -0.5])
print(f"state        {state}")
print(f"f(t, state)  {f(0.0, state)}")
print(f"the first two components are the last two of the state, shifted: "
      f"{np.array_equal(f(0.0, state)[:-1], state[1:])}")
```

### 1.1 How to check a reduction

The temptation is to solve the system, difference the first component, and see whether it matches
the second. Do not: that measures the differencing.

The reduction guarantees an **exact identity**, $f(t, u)_1 = u_2$, at every state whatever the
solver does. Check that instead.

```python
out = od.reduction_is_faithful()
print(f"{'steps':>8}{'position error':>17}{'velocity error':>17}"
      f"{'identity residual':>20}")
for n, p, v, r in zip(out["steps"], out["position_error"], out["velocity_error"],
                      out["reduction_identity_residual"]):
    print(f"{n:>8}{p:>17.4e}{v:>17.4e}{r:>20.1e}")
print(f"\nthe identity is exact: {out['identity_is_exact']}")
assert out["identity_is_exact"], "f(t, u)[0] is u[1] by construction, not approximately"
```

The residual is **exactly zero**, at every step count, because it is an identity. Checking it by
differencing instead returns $8\times10^{-4}$ that shrinks like $h^2$, which would look like a
second order property of the reduction and is a measurement of `np.gradient`.

## 2. Every method already works

Nothing in lessons 67 to 72 assumed $y$ was a scalar. The orders are unchanged.

```python
print(f"{'method':>16}{'claimed':>9}{'fitted':>9}{'matches':>9}")
for name in ("euler", "heun", "kutta third", "rk4"):
    out = od.order_on_a_system(name=name)
    print(f"{name:>16}{out['claimed_order']:>9}{out['fitted_order']:>9.4f}"
          f"{str(out['matches']):>9}")
```

That is worth confirming rather than assuming. A mistake in the reduction usually shows up as a
**lost order** rather than as a wrong answer, so the trajectory can look plausible while the
convergence is one order short.

## 3. The pendulum, and energy drift

$\theta'' = -(g/L)\sin\theta$ is Hamiltonian when undamped, so its energy

$$
E = \tfrac12 (L\theta')^2 + gL(1 - \cos\theta)
$$

is exactly constant. Every departure in the computed energy is the method's.

```python
f, energy = od.pendulum()
print(f"{'method':>10}{'worst drift':>16}{'trend per unit time':>22}"
      f"{'drifts one way':>17}")
for name in ("euler", "heun", "rk4"):
    out = od.energy_drift(f, energy, [1.0, 0.0], 200.0, 20000, name)
    print(f"{name:>10}{out['worst']:>16.4e}{out['trend_per_unit_time']:>22.3e}"
          f"{str(out['drifts_one_way']):>17}")
```

Euler **gains** energy: the pendulum swings higher and higher, and after 200 time units the error
is 21 times the true energy. Heun and RK4 lose it slowly instead.

The size is not the interesting part. Look at the last column: **all three drift one way.** RK4's
drift is $10^{-7}$ over this run and it has a trend, so over a run a thousand times longer it is
$10^{-4}$, and there is no step size at which it stops being a trend. Lesson 74 is about that.

## 4. Orbits

The two body problem conserves energy **and** angular momentum, and a method can lose one and keep
the other.

```python
print(f"{'eccentricity':>14}{'closure gap':>15}{'energy drift':>15}"
      f"{'angular momentum drift':>25}{'radius range':>15}")
for e in (0.0, 0.3, 0.6, 0.9):
    out = od.orbit_closes(eccentricity=e, periods=10.0, steps=4000)
    print(f"{e:>14.1f}{out['closure_gap']:>15.3e}{out['energy_drift']:>15.3e}"
          f"{out['angular_momentum_drift']:>25.3e}{out['radius_range']:>15.4f}")
```

Two things to read off.

**Eccentricity is what costs.** A circular orbit closes to $10^{-7}$ after ten periods; at
$e = 0.9$ the gap is $1.68$, which is the size of the orbit. The reason is that the speed at
perihelion is $\sqrt{(1+e)/(1-e)}$ times the mean, which is $4.36$ at $e = 0.9$, and a fixed step
cannot serve both ends of that.

The radius range column is the check that the orbits are the ones intended: it should be exactly
$2e$, and it reads $0.0000$, $0.6000$ and $1.2000$ for the first three. **At $e = 0.9$ it reads
$1.7465$ instead of $1.8$**, and that discrepancy is not a bug in the setup. It is the same
failure as the closure gap: the computed orbit no longer reaches its own perihelion.

**Angular momentum survives where energy does not.** RK4 is not designed to conserve either, and
it happens to conserve angular momentum far better, because the angular momentum is a **linear
invariant** of the state in a sense energy is not. Measuring both is the only way to find that
out.

```python
fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 4.0))
for e, style in ((0.0, "-"), (0.6, "-"), (0.9, "--")):
    orbit = od.orbit_closes(eccentricity=e, periods=3.0, steps=3000)
    left.plot(orbit["y"][:, 0], orbit["y"][:, 1], style, lw=1.2, label=f"e = {e}")
left.plot([0.0], [0.0], "k*", markersize=10)
left.set_aspect("equal"); left.set_xlabel("x"); left.set_ylabel("y")
left.set_title("three Kepler orbits, RK4"); left.legend(fontsize=8)
pend_f, pend_energy = od.pendulum()
for name in ("euler", "heun", "rk4"):
    drift = od.energy_drift(pend_f, pend_energy, [1.0, 0.0], 60.0, 6000, name)
    right.plot(drift["t"], np.abs(drift["relative_drift"]) + 1e-18, label=name)
right.set_yscale("log"); right.set_xlabel("t"); right.set_ylabel("|relative energy error|")
right.set_title("energy drift, all one way"); right.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/73_orbits_and_energy.png", dpi=110)
plt.close(fig)
print("saved ../figures/73_orbits_and_energy.png")
```

![Kepler orbits and the energy drift](../figures/73_orbits_and_energy.png)

The left panel shows why eccentricity costs: the $e = 0.9$ orbit spends most of its period far out
moving slowly and crosses perihelion in a rush, and a fixed step serves one end or the other. The
right panel is the same story as the table, on a log scale, so the straight lines are the trends.

## 5. Stiffness along a trajectory

Lesson 72 measured the stiffness ratio of a matrix. A nonlinear system has a different Jacobian at
every point, so the ratio is a function of time.

```python
out = od.stiffness_along_the_path()
print(f"smallest ratio {out['smallest']:.1f}, largest {out['largest']:.1f}, "
      f"varying by {out['varies_by']:.1f}x")
print(f"the spike reaches {out['spike_voltage']:.1f} mV\n")
print(f"{'t':>8}{'voltage':>12}{'stiffness ratio':>18}")
picks = np.linspace(0, out["t"].size - 1, 14).astype(int)
for i in picks:
    print(f"{out['t'][i]:>8.2f}{out['voltage'][i]:>12.2f}{out['stiffness_ratio'][i]:>18.1f}")
```

The ratio runs from $19.7$ to $882.8$, a factor of $45$ along one trajectory. **A solver that
chose its step from the resting state would be choosing from the wrong information within a
millisecond**, which is why stiff solvers re-estimate the Jacobian along the way rather than
forming it once.

Where the peak sits is worth looking at. It is **not** at the top of the spike: at $t = 16.92$ the
voltage is $+20.6$ mV and the ratio is $19.7$, the smallest in the whole run. The peak of $695.7$
comes at $t = 18.46$ on the way down, when the sodium gate has shut and the potassium gate has
not, so one time constant is very short while another is not.

That is the general shape. **Stiffness lives where the time constants disagree, not where the
solution is moving fastest**, and those are different places.

## 6. Three failures that are not the solver's fault

### 6.1 Chaos

```python
out = od.divergence_is_the_problem()
print(f"initial separation {out['initial_separation']:.0e}")
print(f"fitted Lyapunov exponent {out['fitted_lyapunov']:.4f}")
print(f"the gap reaches order 1 at t = {out['time_to_order_one']:.2f}")
print(f"halving the error buys {out['extra_time_per_halving']:.3f} extra time units\n")
print(f"{'t':>8}{'separation':>16}")
for t in (0, 5, 10, 15, 20, 25, 30):
    i = int(t / 30.0 * (out["t"].size - 1))
    print(f"{out['t'][i]:>8.1f}{out['separation'][i]:>16.3e}")
```

Two Lorenz trajectories starting $10^{-8}$ apart separate at rate $e^{0.88t}$ and are order 1
apart by $t \approx 18$. **That rate is in the equations.** Improving the solver by a factor of
$10^{6}$ buys $\log(10^6)/0.88 = 16$ extra time units and no more.

So a Lorenz trajectory reported at $t = 50$ is not a solution of anything. The statistics of the
attractor are computable and the trajectory is not.

### 6.2 A parameter that changes the answer

```python
out = od.a_parameter_crossing_changes_the_answer()
print(f"{'wind':>8}{'early twist':>15}{'late twist':>15}{'growth':>12}")
for w, e, l, g in zip(out["wind"], out["early_amplitude"], out["late_amplitude"],
                      out["growth"]):
    print(f"{w:>8.0f}{e:>15.4e}{l:>15.4e}{g:>12.3f}")
print(f"\nfirst wind that grows: {out['first_wind_that_grows']}")
print(f"crossings of 1: {out['crossings_of_one']}, "
      f"monotone: {out['growth_is_monotone']}")
print(out["note"])
assert out["first_wind_that_grows"] is not None, "some wind in the sweep grows the twist"
assert not out["growth_is_monotone"], "and the growth is not monotone in the wind"
```

A small initial twist decays at 40, 60 and 80 and grows at 100, 140 and 200. The transition is in
the equations, and **no error estimate mentions it**: both answers are computed to the same
accuracy and they are qualitatively different.

**It is not a critical speed.** The growth peaks at 17.6 near a wind of 140 and falls back to 4.4
by 200, so what the sweep passes through is a resonance between the periodic forcing and the twist
mode, not a threshold. Calling it a critical speed would be reading a monotone story into a
non-monotone measurement.

### 6.3 Energy drift, again

Section 3 already showed it. It belongs in this list because it is the same kind of thing: a
property of the method that no error estimate reports and no tolerance controls, visible only if
you know to look for the invariant.

## 7. Exercises

**Level 1, understanding**

1.1 Reduce $y''' + 2y'' - y' + 3y = \sin t$ to a first order system and write out $f$.

1.2 Explain why the reduction does not change a method's order.

1.3 Say why checking a reduction by differencing the computed solution is the wrong check, and
give the right one.

1.4 Explain what it means for a solver to conserve angular momentum but not energy.

1.5 Say why the Lyapunov exponent of the Lorenz system is a property of the equations rather than
of the solver.

**Level 2, derivation**

2.1 Show that the reduction of $y^{(m)} = g$ has a Jacobian in companion form, and find its
eigenvalues for the linear case.

2.2 Derive the energy of the pendulum from its equation and confirm $dE/dt = 0$ along solutions.

2.3 Derive the relation between eccentricity and the ratio of perihelion to aphelion speed, and
use it to predict how the step must vary around a Kepler orbit.

2.4 Show that angular momentum is conserved by any Runge-Kutta method applied to the two body
problem in polar form, and say why the Cartesian form does not conserve it exactly.

2.5 Derive the local Lyapunov exponent from the Jacobian and say how it relates to the fitted
global one.

**Level 3, computational**

3.1 Implement the reduction for an arbitrary order and verify the identity of section 1.1 for
$m$ from 1 to 6.

3.2 Solve the restricted three body problem and reproduce a periodic Arenstorf orbit.

3.3 Implement the Hodgkin-Huxley system with a stiff solver and compare the step counts against
RK4 at matched accuracy.

3.4 Compute the largest Lyapunov exponent of the Lorenz system by the standard renormalisation
algorithm and compare against the fitted value here.

3.5 Implement the Tacoma model with a slowly increasing wind speed and see whether the transition
happens where the fixed wind sweep says it does.

**Level 4, experimental**

4.1 Measure the closure gap of a Kepler orbit against eccentricity from 0 to 0.95 and fit the
relationship.

4.2 Measure the Hodgkin-Huxley stiffness ratio against the injected current and find where the
neuron starts spiking.

4.3 Measure the Lorenz separation rate at several parameter values and find where the attractor
stops being chaotic.

**Level 5, advanced**

5.1 **Shadowing.** A computed chaotic trajectory is not the solution from its initial condition,
but it may be a solution from a nearby one. State the shadowing lemma and say what it does and
does not rescue.

5.2 **Conserving invariants by projection.** After each step, project the state back onto the
level set of the invariant. Show this restores conservation and say what it does to the order,
and to the trajectory.

5.3 **Why angular momentum survives.** Identify the structural property of the two body problem
that makes angular momentum better conserved than energy under a Runge-Kutta method, and
construct a system where the reverse happens.

## 8. Key takeaways

- **The reduction to first order is mechanical and free.** Every method of lessons 67 to 72 works
  on a system unchanged, and the orders are the same.

- **Check a reduction by its identity, not by differencing.** $f(t, u)_1 = u_2$ holds exactly;
  differencing the computed solution returns $8\times10^{-4}$ and measures `np.gradient`.

- **Every method here drifts in energy, one way.** Euler gains 21 times the true energy over 200
  time units and RK4 loses $10^{-7}$, and both have a trend. Size is not the point; the trend is.

- **A method can conserve one invariant and lose another.** On a Kepler orbit RK4's angular
  momentum drift is far smaller than its energy drift, so measuring only one gives the wrong
  impression of the run.

- **Stiffness varies along a trajectory** by a factor of 45 on the Hodgkin-Huxley neuron, and
  its peak is on the falling edge rather than at the top of the spike: stiffness lives where the
  time constants disagree, not where the solution moves fastest.

- **Three failures belong to the equations, not the solver**: chaos, where accuracy buys only its
  own logarithm; a parameter crossing, where the answer changes kind and no error estimate says
  so; and energy drift, which no tolerance controls.

- **The Tacoma transition is a resonance, not a threshold.** The growth runs 0.62, 0.59, 0.73,
  1.31, 17.6, 4.39 across the wind sweep, which has a peak in it rather than a step.

## Where this goes next

Section 3 found that every method here drifts in energy with a trend. Lesson 74 explains why, in
terms of lesson 72's stability picture on the imaginary axis, and builds methods that get the
structure right instead of the digits. Lesson 75 changes the question entirely: conditions at both
ends instead of one, where there is nothing to march from.
