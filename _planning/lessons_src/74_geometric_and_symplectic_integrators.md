# 74. Geometric and Symplectic Integrators

**Part 10: Ordinary Differential Equations**

## Learning objectives

By the end of this lesson you will be able to:

1. Explain why every method so far loses energy on a Hamiltonian system, in terms of lesson 72's
   stability picture.
2. Write symplectic Euler and Stormer-Verlet, and see how small the change from ordinary Euler is.
3. Measure the defining property directly: the one step map preserves phase space area exactly.
4. Distinguish a **bounded band** from a **trend**, and know why the second is what matters.
5. Say why symplectic integration and adaptive stepping do not compose, with a measurement rather
   than an assertion.

## Prerequisites

Lesson 73 (energy drift measured on the pendulum). Lesson 72 (regions of absolute stability,
especially on the imaginary axis). Lesson 69 (RK4, the method being beaten here).

---

## 1. Why the drift happens

Lesson 73 measured RK4's energy on an undamped pendulum drifting steadily downwards. Refining the
step makes the drift smaller and does not make it stop.

The reason is in lesson 72. A Hamiltonian system has eigenvalues on the **imaginary axis**, and the
exact growth factor there has modulus exactly 1. Every method here has $|R(i\theta)| \ne 1$:

```python
from nalib import stability as st

def modulus_on_the_imaginary_axis(name, theta):
    """|R(i theta)|, which is exactly 1 for the true flow of a Hamiltonian system."""
    return abs(complex(st.growth_function(name)(1j * float(theta))))

print(f"{'method':>16}{'|R(0.1i)|':>16}{'|R(0.3i)|':>16}{'per step':>16}")
for name in ("euler", "heun", "rk4"):
    a = modulus_on_the_imaginary_axis(name, 0.1)
    b = modulus_on_the_imaginary_axis(name, 0.3)
    print(f"{name:>16}{a:>16.12f}{b:>16.12f}{a - 1.0:>16.3e}")
print("\nthe exact factor on the imaginary axis has modulus exactly 1")
assert modulus_on_the_imaginary_axis("euler", 0.1) > 1.0, "Euler grows on the axis"
assert modulus_on_the_imaginary_axis("rk4", 0.1) < 1.0, "and RK4 decays"
```

Euler's is above 1, so amplitude grows. RK4's is below 1, so it decays. Either way the departure
is **one way** and per step, so over $n$ steps it compounds. Halving the step makes the per step
departure smaller and doubles the number of steps, and the product still goes one direction.

## 2. Symplectic Euler: one character changed

For a separable Hamiltonian $H(q, p) = T(p) + V(q)$ the equations are $q' = p$, $p' = -V'(q)$.
Ordinary Euler:

$$
q_{n+1} = q_n + h p_n, \qquad p_{n+1} = p_n + h F(q_n).
$$

Symplectic Euler updates the momentum first and uses the **new** momentum for the position:

$$
p_{n+1} = p_n + h F(q_n), \qquad q_{n+1} = q_n + h p_{n+1}.
$$

That is the whole difference. Both are first order.

```python
from nalib import symplectic as sy

force, energy = sy.pendulum()
q, p, h = np.asarray([0.7]), np.asarray([-0.4]), 0.1
sq, sp = sy.symplectic_euler_step(force, q, p, h)
eq, ep = sy.explicit_euler_step(force, q, p, h)
print(f"explicit  euler: q = {eq[0]:.10f}, p = {ep[0]:.10f}")
print(f"symplectic euler: q = {sq[0]:.10f}, p = {sp[0]:.10f}")
print(f"the momenta are identical: {abs(sp[0] - ep[0]) < 1e-16}")
print(f"the positions differ by h times the momentum change: "
      f"{abs((sq[0] - eq[0]) - h * (sp[0] - p[0])) < 1e-15}")
```

**Stormer-Verlet** is the second order version: half a momentum kick, a full position drift, half
a kick. It is time reversible as well as symplectic, and it is the method behind essentially every
molecular dynamics and long term orbital calculation there is.

```python
for name in sy.STEPS:
    out = sy.order_of(name)
    print(f"{name:>18}: claimed order {out['claimed_order']}, "
          f"fitted {out['fitted_order']:.4f}, matches {out['matches']}")
```

Note the orders: **the two symplectic methods are first and second order and the non symplectic
one is also first order.** Whatever the difference turns out to be, it is not accuracy.

## 3. The defining property

Liouville's theorem says the exact flow of a Hamiltonian system preserves the area of any region
of phase space. A **symplectic** integrator preserves it exactly, in floating point, for any step.

That is a property of the one step map, so it can be measured directly: build the map's Jacobian
by differencing and take its determinant.

```python
print(f"{'method':>18}{'h':>7}{'worst |det - 1|':>18}{'preserves area':>17}")
for name in sy.STEPS:
    for h in (0.05, 0.2, 0.5):
        out = sy.area_is_preserved(name, h=h, corners=60)
        print(f"{name:>18}{h:>7.2f}{out['worst_departure_from_one']:>18.3e}"
              f"{str(out['preserves_area']):>17}")
```

The symplectic methods sit at $10^{-8}$, which is the differencing noise. Explicit Euler misses by
exactly $h^2$: $0.0025$, $0.04$ and $0.25$ at the three steps.

```python
for h in (0.05, 0.2, 0.5):
    out = sy.area_is_preserved("explicit euler", h=h, corners=60)
    print(f"h = {h}: departure {out['worst_departure_from_one']:.6f}, "
          f"h^2 = {h ** 2:.6f}, ratio {out['worst_departure_from_one'] / h ** 2:.6f}")
```

Every explicit Euler step stretches phase space by a factor of $1 + h^2$. That is exactly why its
energy grows: the stretching is uniform and compounding.

## 4. Bounded versus drifting

The consequence of preserving area is **not** that the energy is conserved. It is that the
computed solution is the exact solution of a **nearby** Hamiltonian, differing from the true one
by $O(h^p)$. Its energy is exactly conserved for that nearby problem, so the computed energy
oscillates in a bounded band rather than drifting.

```python
out = sy.drift_against_bounded()
print(f"over {out['t_end']:.0f} time units:\n")
print(f"{'method':>18}{'band':>14}{'trend per unit':>17}{'drift over run':>17}"
      f"{'bounded':>10}{'final error':>15}")
for n, b, t, d, bo, fr in zip(out["names"], out["band"], out["trend_per_unit_time"],
                              out["drift_over_the_run"], out["bounded"],
                              out["final_relative_error"]):
    print(f"{n:>18}{b:>14.4e}{t:>17.3e}{d:>17.3e}{str(bo):>10}{fr:>15.3e}")
```

Explicit Euler ends at 21.7 times the true energy and its trend accounts for all of it. The two
symplectic methods have bands of $0.03$ and $0.0002$ and trends of $10^{-7}$ and $10^{-11}$, which
over the run amount to nothing.

**The band shrinks like $h^p$ and the trend does not exist.** That is the guarantee, and it is
qualitatively different from a small error.

```python
out = sy.the_band_shrinks_with_the_step()
print(f"{'h':>10}{'band low':>16}{'band high':>16}{'width':>14}{'ratio':>9}")
for i, (h, lo, hi, w) in enumerate(zip(out["h"], out["band_low"], out["band_high"],
                                       out["band_width"])):
    ratio = "" if i == 0 else f"{out['ratio_per_halving'][i - 1]:.3f}"
    print(f"{h:>10.5f}{lo:>16.3e}{hi:>16.3e}{w:>14.3e}{ratio:>9}")
print(f"\nshrinks like h^2: {out['shrinks_like_h_squared']}, "
      f"bands are nested: {out['bands_are_nested']}")
```

Halving the step quarters the band, exactly, which is Stormer-Verlet's second order. Note that the
bands are **nested**, all with their high end at exactly zero, rather than disjoint: this initial
condition sits at the top of every band.

```python
fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 4.0))
force, energy, closed_form = sy.harmonic_oscillator()
for name, style in (("explicit euler", "-"), ("symplectic euler", "-"),
                    ("stormer verlet", "--")):
    run = sy.integrate(force, [1.0], [0.0], 30.0, 300, name)
    left.plot(run["q"][:, 0], run["p"][:, 0], style, lw=1.0, label=name)
left.set_aspect("equal"); left.set_xlabel("q"); left.set_ylabel("p")
left.set_title("phase portrait, h = 0.1"); left.legend(fontsize=8)
pend_force, pend_energy = sy.pendulum()
for name in sy.STEPS:
    with np.errstate(over="ignore", invalid="ignore"):
        run = sy.energy_over_time(pend_force, pend_energy, [1.0], [0.0], 60.0, 6000, name)
    right.plot(run["t"], run["relative"], label=name)
right.set_yscale("symlog", linthresh=1e-6)
right.set_xlabel("t"); right.set_ylabel("relative energy error")
right.set_title("a band, or a trend"); right.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/74_symplectic.png", dpi=110); plt.close(fig)
print("saved ../figures/74_symplectic.png")
```

![Phase portraits and energy](../figures/74_symplectic.png)

The left panel is the whole lesson in one picture. The exact orbit is a circle; explicit Euler
spirals **out**, symplectic Euler traces a closed ellipse slightly off the circle, and
Stormer-Verlet sits on it. The right panel shows what that costs in energy: two bounded bands and
one line walking away.

## 5. At equal cost, against RK4

Stormer-Verlet is second order and RK4 is fourth. Verlet costs two force evaluations per step and
RK4 costs four, so at equal cost Verlet takes twice as many steps.

```python
out = sy.against_runge_kutta()
print(f"both methods use {out['verlet_evaluations']} evaluations over "
      f"{out['t_end']:.0f} time units\n")
print(f"{'method':>18}{'energy band':>16}{'trend':>15}{'drift over run':>17}{'bounded':>10}")
print(f"{'stormer verlet':>18}{out['verlet_band']:>16.3e}{out['verlet_trend']:>15.3e}"
      f"{out['verlet_drift']:>17.3e}{str(out['verlet_is_bounded']):>10}")
print(f"{'rk4':>18}{out['rk4_band']:>16.3e}{out['rk4_trend']:>15.3e}"
      f"{out['rk4_drift']:>17.3e}{str(out['rk4_is_bounded']):>10}")
```

**RK4's band is fifteen times smaller than Verlet's, and RK4 is the worse method here.** Its band
moves: the drift over the run accounts for the whole band, so extending the run extends the error
in proportion. Verlet's drift is $5\times10^{-9}$ against a band of $2\times10^{-4}$, so extending
the run changes nothing.

A comparison that only looked at error sizes would rank these two the wrong way round. That is the
whole reason to measure the trend separately.

```python
out = sy.kepler_orbit_stays_closed()
print(f"Kepler, e = 0.6, over 200 periods:")
print(f"  perihelion {out['perihelion_early']:.6f} -> {out['perihelion_late']:.6f}")
print(f"  aphelion   {out['aphelion_early']:.6f} -> {out['aphelion_late']:.6f}")
print(f"  energy band {out['energy_band']:.3e}, trend {out['energy_trend']:.3e}")
print(f"  angular momentum drift {out['angular_momentum_drift']:.3e}")
print(f"  the shape is held: {out['shape_is_held']}")
```

Over two hundred orbits the ellipse keeps its shape to six digits. A non symplectic method's
ellipse **precesses**, slowly rotating as its energy error accumulates, which is a visible and
qualitatively wrong feature rather than a small error.

## 6. The catch: it does not compose with adaptivity

The backward error argument says the computed solution exactly solves a nearby Hamiltonian
$H + O(h^p)$. **That nearby Hamiltonian depends on $h$.** Keep changing $h$ and you keep changing
which problem you are solving exactly, so the energy error stops being a bounded oscillation.

It is worth being precise about what breaks, because the obvious statement is wrong.

```python
out = sy.changing_the_step_destroys_the_bound()
print(f"over {out['t_end']:.0f} time units:\n")
print(f"{'step policy':>14}{'steps':>9}{'band':>14}{'trend per unit':>17}"
      f"{'drift over run':>17}{'bounded':>10}")
for p, s, b, t, d, bo in zip(out["policy"], out["steps"], out["band"],
                             out["trend_per_unit_time"], out["drift_over_the_run"],
                             out["bounded"]):
    print(f"{p:>14}{s:>9}{b:>14.3e}{t:>17.3e}{d:>17.3e}{str(bo):>10}")
print(f"\nonly the irregular one drifts: {out['only_the_irregular_one_drifts']}")
assert out["only_the_irregular_one_drifts"], "a repeating pattern keeps its own band"
```

**A fixed step stays bounded and so does a deterministic alternation between two steps.** A
repeating pattern of symplectic maps is itself a symplectic map, with its own modified
Hamiltonian, so it keeps its own band. What breaks the bound is **irregularity**: a step drawn at
random each time is not a repeating map, and its energy performs a random walk.

An adaptive controller chooses its steps from the local error estimate, which varies irregularly
with the solution, so it produces the third row. That is the practical obstruction, and codes that
want both either fix the step for the symplectic part or vary the step in a transformed time
variable that keeps the map symplectic.

## 7. Exercises

**Level 1, understanding**

1.1 Explain why a Hamiltonian system's eigenvalues lie on the imaginary axis, for the harmonic
oscillator.

1.2 State the difference between explicit and symplectic Euler in one line and say which quantity
each update uses.

1.3 Explain what it means for the computed solution to exactly solve a nearby Hamiltonian.

1.4 Say why a bounded band is a better guarantee than a small error, and when it is not.

1.5 Explain why symplectic methods are for separable Hamiltonians and what that excludes.

**Level 2, derivation**

2.1 Compute the Jacobian of the explicit Euler map for the harmonic oscillator and show its
determinant is $1 + h^2\omega^2$.

2.2 Compute the same determinant for symplectic Euler and show it is exactly 1.

2.3 Derive Stormer-Verlet from the composition of two symplectic Euler half steps of opposite
kind, and use that to show it is symplectic.

2.4 Derive the modified Hamiltonian of symplectic Euler to first order in $h$ and confirm the
band's $O(h)$ size.

2.5 Show that Stormer-Verlet is time reversible and that explicit Euler is not.

**Level 3, computational**

3.1 Implement the fourth order Yoshida composition of Stormer-Verlet steps and verify its order
and its symplecticity.

3.2 Implement a symplectic integrator for the outer solar system and integrate for $10^6$ years,
reporting the energy band.

3.3 Implement the implicit midpoint rule and confirm it is symplectic for a **non separable**
Hamiltonian, where Verlet is not available.

3.4 Implement a projection method that restores exact energy conservation after each step, and
measure what it does to the phase error.

3.5 Implement a variable step symplectic method using a Sundman transformation of time and check
the band survives.

**Level 4, experimental**

4.1 Measure the energy band against the step for symplectic Euler and confirm it is $O(h)$ rather
than $O(h^2)$.

4.2 Measure the phase error of Stormer-Verlet on the harmonic oscillator over a long run and
confirm it grows linearly in time while the energy does not.

4.3 Measure how much irregularity is needed to break the bound, by interpolating between the
fixed and random step policies of section 6.

**Level 5, advanced**

5.1 **Backward error analysis.** State the theorem that the modified Hamiltonian is an asymptotic
series, not a convergent one, and say what that implies about very long integrations.

5.2 **Why RK4 cannot be symplectic.** Show that an explicit Runge-Kutta method cannot be
symplectic, and identify the structural obstruction.

5.3 **Symplecticity versus energy conservation.** Prove that no method can conserve energy exactly
and be symplectic for a general Hamiltonian, and say which of the two the field chose and why.

## 8. Key takeaways

- **The drift comes from the imaginary axis.** Every method's growth factor there has modulus
  other than 1, so the amplitude changes by a fixed factor per step and compounds.

- **Symplectic Euler differs from Euler in one character.** Use the new momentum for the position
  update. Both are first order and only one of them is bounded.

- **Area preservation is measurable and exact.** The symplectic methods sit at the differencing
  noise for every step size, and explicit Euler misses by exactly $h^2$.

- **The guarantee is a band, not conservation.** The band shrinks like $h^p$ and there is no
  trend, which is a different kind of statement from a small error.

- **At equal cost RK4 is more accurate and worse.** Its energy band is fifteen times narrower and
  the whole band is a trend, so it grows with the length of the run while Verlet's does not.

- **It is irregularity that breaks the bound, not change.** A fixed step and a two step
  alternation both stay bounded; a random step drifts. That is why adaptivity is the obstruction,
  because an error controller produces exactly the irregular pattern.

## Where this goes next

Part 10 has one more question. Everything so far marched forward from a known state. Lesson 75
puts conditions at **both** ends, where there is nothing to march from, and finds that the obvious
way to fix that is catastrophically ill conditioned. Lesson 76 then stops asking for the solution
at points and asks for it as a function, which is where the finite element method comes from.
