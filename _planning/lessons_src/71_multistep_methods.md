# 71. Multistep Methods

**Part 10: Ordinary Differential Equations**

## Learning objectives

By the end of this lesson you will be able to:

1. Derive the Adams-Bashforth and Adams-Moulton coefficients by integrating an interpolating
   polynomial, and check them against the published tables.
2. Run a predictor-corrector pair and say what the corrector buys and what it costs.
3. Test zero-stability with the root condition, and watch a consistent method diverge.
4. State both Dahlquist barriers and check the first one by search.
5. Get the starting values right, and measure what happens when you do not.

## Prerequisites

Lesson 69 (Runge-Kutta, used here to start the multistep methods and to compare against). Lesson
62 (Newton-Cotes, which is where the coefficients come from). Lesson 34 (Newton's divided
difference form of the interpolating polynomial). Lesson 12 (polynomial root finding, used for
the root condition).

---

## 1. Reuse the past instead of adding stages

A Runge-Kutta method throws away everything it computed once the step is done. A **multistep**
method keeps it. The general linear $k$ step method is

$$
\sum_{j=0}^{k} \alpha_j y_{n+1-j} = h \sum_{j=0}^{k} \beta_j f_{n+1-j},
$$

with $\alpha_0 = 1$ by normalisation. It is **explicit** when $\beta_0 = 0$, because then
$y_{n+1}$ appears only on the left.

The cost is the point. An explicit $k$ step method uses exactly **one** new evaluation of $f$ per
step whatever $k$ is, because the others are already known. RK4 uses four.

```python
from nalib import multistep as ms

print(f"{'name':>20}{'steps':>7}{'order':>7}{'explicit':>10}")
for name in ms.NAMED:
    alpha, beta, order = ms.method(name)
    print(f"{name:>20}{max(len(alpha), len(beta)) - 1:>7}{order:>7}"
          f"{str(ms.is_explicit(name)):>10}")
```

## 2. Where the coefficients come from

Integrate the equation over one step:

$$
y_{n+1} = y_n + \int_{t_n}^{t_{n+1}} f(t, y(t))\,dt.
$$

Now replace $f$ by the polynomial that interpolates it at points you already have.

**Adams-Bashforth** interpolates at $t_n, t_{n-1}, \dots, t_{n-k+1}$, all in the past, and
extrapolates across $[t_n, t_{n+1}]$. Explicit.

**Adams-Moulton** includes $t_{n+1}$ as well, so it interpolates rather than extrapolating.
Implicit, and better for it.

```python
print("Adams-Bashforth, coefficients of f at t_n, t_(n-1), ...")
for k in range(1, 6):
    print(f"  {k} step: {[str(v) for v in ms.adams_bashforth_coefficients(k)]}")
print("\nAdams-Moulton, coefficients of f at t_(n+1), t_n, ...")
for k in range(1, 6):
    print(f"  {k} step: {[str(v) for v in ms.adams_moulton_coefficients(k)]}")
```

These are exact rationals, derived by integrating the Lagrange basis, and they match the tables in
every textbook: $\tfrac{1}{24}(55, -59, 37, -9)$ for AB4 and $\tfrac{1}{24}(9, 19, -5, 1)$ for
AM4.

## 3. The order conditions

A linear multistep method has order $p$ when it is exact for $y = 1, t, t^2, \dots, t^p$. Written
out, that is

$$
\sum_j \alpha_j = 0, \qquad
\sum_j \alpha_j (1-j)^m = m \sum_j \beta_j (1-j)^{m-1} \quad (m = 1, \dots, p).
$$

The first is **consistency of order 0**, and the rest come one per order.

```python
alpha, beta, order = ms.method("ab4")
out = ms.order_conditions(alpha, beta, up_to=6)
print(f"AB4 claims order {order}")
print(f"{'power':>7}{'left':>18}{'right':>18}{'gap':>16}  holds")
for m, lo, hi, ok in zip(out["power"], out["left"], out["right"], out["holds"]):
    print(f"{m:>7}{lo:>18.10f}{hi:>18.10f}{hi - lo:>16.6e}  {ok}")
print(f"\nachieved order: {out['order']}")
```

The first five rows agree to fourteen digits and the sixth misses by 42. That is the shape to
expect: an order condition either holds to roundoff or fails by something of order 1, and there is
nothing in between to be uncertain about.

## 4. Zero-stability: consistency is not enough

Here is the difference from Runge-Kutta, and it is the whole reason this lesson is harder than
lesson 69.

A multistep method's recurrence has its own dynamics, independent of the differential equation.
Set $h = 0$ and the method becomes $\sum_j \alpha_j y_{n+1-j} = 0$, a linear recurrence whose
solutions are powers of the roots of the **characteristic polynomial**

$$
\rho(z) = \sum_{j=0}^{k} \alpha_j z^{k-j}.
$$

The **root condition** is: every root satisfies $|z| \le 1$, and any root with $|z| = 1$ is
simple. If it fails, the recurrence has a solution that grows without bound, and it will grow
whatever the equation is.

**Dahlquist's equivalence theorem**: consistent + zero-stable $\iff$ convergent. Both halves are
needed and neither implies the other.

```python
for name in ("ab4", "am4", "milne simpson", "unstable order two"):
    alpha, beta, order = ms.method(name)
    out = ms.root_condition(alpha)
    roots = ", ".join(f"{v:.4f}" for v in np.real_if_close(out["roots"]))
    print(f"{name:>20}: roots [{roots}]  largest |z| = {out['largest_modulus']:.4f}  "
          f"zero-stable: {out['zero_stable']}")
```

Milne-Simpson has roots $1$ and $-1$, both on the circle and both simple, so it passes. It is
**weakly stable**: the root at $-1$ does not grow, and it does not decay either, so an error
introduced into that mode oscillates forever.

### 4.1 A consistent method that diverges

```python
out = ms.unstable_example()
print(f"alpha = {[str(v) for v in out['alpha']]}, beta = {[str(v) for v in out['beta']]}")
print(f"claimed order {out['claimed_order']}, measured order {out['measured_order']}")
print(f"consistent: {out['consistent']}, zero-stable: {out['zero_stable']}")
print(f"roots: {np.round(np.real_if_close(out['roots']), 4)}, "
      f"largest |z| = {out['largest_modulus']:.4f}")
```

This method satisfies the order conditions through order 3. It is consistent, it is high order,
and it is useless.

```python
out = ms.unstable_method_diverges()
print(f"exact answer: {out['exact']:.10f}")
print(f"{'steps':>8}{'final value':>18}{'error':>14}")
for n, v, e in zip(out["steps"], out["final_value"], out["errors"]):
    print(f"{n:>8}{v:>18.4e}{e:>14.3e}")
print(f"\nrefining makes it worse: {out['gets_worse_with_refinement']}")
assert out["gets_worse_with_refinement"], "the parasitic root grows like |z| to the n"
```

**Refining the step makes it worse.** The parasitic root grows like $|z|^n$ and $n$ grows as the
step shrinks, so more steps means more amplification. There is no tolerance you can set and no
step you can take that fixes it.

That is what zero-stability protects against, and it has no analogue in lesson 69: an explicit
Runge-Kutta method has $\rho(z) = z - 1$ and passes the root condition automatically.

```python
fig, ax = plt.subplots()
circle = np.exp(1j * np.linspace(0.0, 2.0 * np.pi, 401))
ax.plot(circle.real, circle.imag, "k--", lw=1, label="unit circle")
for name, marker in (("ab4", "o"), ("milne simpson", "s"), ("unstable order two", "x")):
    alpha, beta, order = ms.method(name)
    roots = np.asarray(ms.root_condition(alpha)["roots"], dtype=complex)
    ax.plot(roots.real, roots.imag, marker, markersize=9, label=name)
ax.set_aspect("equal"); ax.set_xlabel("Re z"); ax.set_ylabel("Im z")
ax.set_title("roots of rho(z): inside the circle, or the method diverges")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/71_root_condition.png", dpi=110); plt.close(fig)
print("saved ../figures/71_root_condition.png")
```

![The root condition](../figures/71_root_condition.png)

Every method has a root at $z = 1$, which is what consistency requires. The unstable method's
second root sits at $-5$, far outside, and that one point is the whole difference between a usable
method and one whose answer reaches $10^{46}$.

## 5. Both named methods, checked

```python
out = ms.all_named_methods()
print(f"{'name':>20}{'steps':>7}{'claimed':>9}{'measured':>10}{'largest |z|':>13}"
      f"{'zero-stable':>13}")
for n, s, c, m, r, z in zip(out["names"], out["steps"], out["claimed_order"],
                            out["measured_order"], out["largest_root_modulus"],
                            out["zero_stable"]):
    print(f"{n:>20}{s:>7}{c:>9}{m:>10}{r:>13.4f}{str(z):>13}")
print(f"\nevery order matches its claim: {out['orders_all_match']}")
```

## 6. The Dahlquist barriers

**First barrier.** A zero-stable $k$ step method has order at most $k+2$ when $k$ is even and
$k+1$ when $k$ is odd. So going from 2 steps to 3 buys **nothing**: both stop at order 4.

```python
out = ms.first_dahlquist_barrier()
print(f"{'steps':>7}{'maximum order':>16}{'gain over the last':>21}")
last = None
for k, p in zip(out["steps"], out["maximum_order"]):
    gain = "" if last is None else str(p - last)
    print(f"{k:>7}{p:>16}{gain:>21}")
    last = p
print(f"\n{out['note']}")
```

The gain column alternates 2, 0, 2, 0. **Every second step is free and every other one is
wasted**, which is why the standard families use every step they take: AB$k$ has order $k$,
comfortably under the barrier, and buys its accuracy from the coefficients instead.

**Second barrier**, from lesson 72: an A-stable linear multistep method has order at most 2, and
the best one is the trapezoid rule. That is why the BDF family stops at order 6 and why nothing
in this lesson solves a stiff problem.

## 7. Predictor-corrector

The implicit methods are more accurate and need a solve. Rather than solving, **predict** with an
explicit method and **correct** with one sweep of the implicit one:

$$
y^{*}_{n+1} = \text{AB4}, \qquad
y_{n+1} = y_n + h\left(\beta_0 f(t_{n+1}, y^{*}_{n+1}) + \sum_{j\ge1}\beta_j f_{n+1-j}\right).
$$

Two evaluations per step: one for the predictor's new point and one for the corrector's. This is
called PECE, and it is what "the Adams method" means in practice.

The gain is the corrector's error constant.

```python
out = ms.implicit_error_constant()
print(f"{'order':>7}{'Adams-Bashforth':>18}{'Adams-Moulton':>18}{'ratio':>10}")
for p, ab, am, r in zip(out["order"], out["adams_bashforth"], out["adams_moulton"],
                        out["ratio"]):
    print(f"{p:>7}{float(ab):>18.6f}{float(am):>18.6f}{r:>10.2f}")
print(f"\nthe implicit constant is smaller at every order: {out['implicit_is_smaller']}")
```

The ratio grows with the order: 5, 9, 13.2, 17.6. At order 4 the implicit method's error
constant is 13 times smaller for the same order and the same number of past points, which is the
whole reason to bother correcting.

It is also why the ratio is worth quoting rather than the constants. Both fall towards $0.33$ and
$0.02$ as the order rises, and looking at either column alone would suggest the methods are
converging on each other.

## 8. Starting values

A $k$ step method needs $k$ starting values and the problem gives one. The rest have to come from
somewhere, and the choice matters more than it looks.

```python
from nalib import ivp

def f(t, y):
    return ivp.as_state(y) - t ** 2 + 1.0

def exact(t):
    return (t + 1.0) ** 2 - 0.5 * np.exp(t)

out = ms.starting_values_matter(f, exact, 0.0, 0.5, 2.0, name="ab4")
print(f"AB4 has order {out['method_order']}")
print(f"{'starter':>18}{'its order':>11}{'predicted':>11}{'fitted':>9}{'error':>14}")
for s, q, pred, fit, err in zip(out["starters"], out["starter_order"],
                                out["predicted_order"], out["fitted_order"],
                                out["finest_error"]):
    print(f"{s:>18}{q:>11}{pred:>11}{fit:>9.3f}{err:>14.3e}")
print(f"\nthe run achieves min(p, q+1): {out['matches_min_p_q_plus_one']}")
assert out["matches_min_p_q_plus_one"], "a starter of order q caps the run at q + 1"
```

**The run achieves $\min(p, q+1)$**, where $p$ is the method's order and $q$ the starter's. Not
$p$, and not $q$ either. A first order starter caps a fourth order method at order 2.

The reason is that a starting error of size $h^{q+1}$ enters the recurrence once and is then
propagated, and it looks to the global error like one extra order. So a starter one order below
the method is enough, and Euler is never enough for anything above order 1.

## 9. Against Runge-Kutta at equal cost

```python
out = ms.against_runge_kutta(f, exact, 0.0, 0.5, 2.0)
print(f"{'budget':>9}{'AB4 error':>14}{'evals':>8}{'PECE error':>14}{'evals':>8}"
      f"{'RK4 error':>14}{'evals':>8}")
for b, ae, av, pe, pv, re, rv in zip(out["budget"], out["ab4_error"],
                                     out["ab4_evaluations"],
                                     out["predictor_corrector_error"],
                                     out["predictor_corrector_evaluations"],
                                     out["rk4_error"], out["rk4_evaluations"]):
    print(f"{b:>9}{ae:>14.3e}{av:>8}{pe:>14.3e}{pv:>8}{re:>14.3e}{rv:>8}")
```

All three are fourth order, so the comparison is entirely about the error constant per
evaluation. AB4 takes four times as many steps as RK4 for the same budget, and PECE takes twice
as many.

**AB4 beats RK4 by a factor of 4 at every budget, and PECE beats it by 8 at the loosest one.**
That is what one evaluation per step buys.

Two things stop this being the end of the story. PECE's advantage over plain AB4 shrinks from
1.9 at 80 evaluations to 0.87 at 1280, so at tight tolerances the corrector is no longer paying
for its extra evaluation. And a multistep method's step is fixed: changing it invalidates the
coefficients, so the adaptivity of lesson 70 is not available for free. Together those are why
Runge-Kutta is the default and Adams methods are the specialist's choice.

## 10. Exercises

**Level 1, understanding**

1.1 Write the AB2 and AM2 formulas and identify which quadrature rules they come from.

1.2 Explain why an explicit $k$ step method costs one evaluation per step whatever $k$ is.

1.3 State the root condition and explain what a root outside the unit circle does to a solution.

1.4 State both Dahlquist barriers and say which one this lesson checks and which one lesson 72
does.

1.5 Explain why a $k$ step method needs $k$ starting values and where they come from.

**Level 2, derivation**

2.1 Derive the AB3 coefficients by integrating the Lagrange interpolating polynomial through
$f_n, f_{n-1}, f_{n-2}$.

2.2 Derive the order conditions of section 3 from the requirement that the method be exact on
$y = t^m$.

2.3 Show that the method $y_{n+1} = -4y_n + 5y_{n-1} + h(4f_n + 2f_{n-1})$ has order 3 and roots
$1$ and $-5$.

2.4 Show that the error constant of AM$k$ is smaller than that of AB$k$ and find the ratio for
$k = 4$.

2.5 Derive the BDF2 formula from differentiating the interpolating polynomial rather than
integrating it, and say why the BDF family is built that way.

**Level 3, computational**

3.1 Implement Adams-Bashforth of any order from the coefficient generator and verify the orders
by refinement.

3.2 Implement a PECE solver with a variable number of corrector sweeps and measure the error
against the sweep count.

3.3 Implement Milne's device: use the predictor-corrector difference as an error estimate and
build an adaptive multistep solver from it.

3.4 Implement variable step Adams methods by re-deriving the coefficients for unequal spacing,
and compare against fixed step.

3.5 Implement the BDF family up to order 6 and verify each one's zero-stability, then confirm
BDF7 fails.

**Level 4, experimental**

4.1 Measure the growth rate of the parasitic mode in section 4.1 against the number of steps and
confirm it is $|z|^n$.

4.2 Measure the effect of the starting method's order on the achieved order for AB2 through AB6,
and confirm $\min(p, q+1)$ each time.

4.3 Measure Milne-Simpson's weak instability on a problem with a decaying solution and find how
long it takes for the oscillating mode to dominate.

**Level 5, advanced**

5.1 **Proving the first barrier.** Sketch Dahlquist's argument that a zero-stable $k$ step method
cannot exceed order $k+2$, and say where the parity enters.

5.2 **Why Milne-Simpson is used anyway.** It is weakly stable and it is still used inside
predictor-corrector pairs. Explain how the pairing suppresses the parasitic mode.

5.3 **Variable step multistep methods.** Changing the step invalidates the coefficients. Describe
the two standard fixes, and say what each does to the stability analysis.

## 11. Key takeaways

- **A multistep method reuses the past instead of adding stages**, so an explicit $k$ step method
  costs one evaluation per step whatever its order.

- **The coefficients come from integrating an interpolating polynomial.** Adams-Bashforth
  extrapolates and Adams-Moulton interpolates, which is why the implicit one has the smaller error
  constant.

- **Consistency is not convergence.** The method with roots $1$ and $-5$ has order 3 and
  diverges, and refining takes its answer from $-6.9$ at 10 steps to $-1.7\times10^{46}$ at 80.
  More work, worse answer, no warning.

- **The root condition is the missing half.** Every root inside or on the unit circle, and those
  on it simple. Runge-Kutta methods pass it automatically, which is why lesson 69 never mentioned
  it.

- **The first Dahlquist barrier is $k+2$ for even $k$ and $k+1$ for odd $k$.** The maximum
  order goes 2, 4, 4, 6, 6, 8, so every second extra step buys nothing at all.

- **A starter of order $q$ caps the run at $\min(p, q+1)$.** A fourth order method started with
  Euler runs at order 2, and nothing about the run says so.

## Where this goes next

Everything in lessons 67 to 71 has been about accuracy. Lesson 72 is about the other constraint,
which has nothing to do with accuracy at all: on a stiff problem the step is set by stability, and
the second Dahlquist barrier says no multistep method of order above 2 can escape it. That is why
stiff solvers are a separate subject.
