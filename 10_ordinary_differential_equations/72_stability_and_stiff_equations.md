# 72. Stability and Stiff Equations

**Part 10: Ordinary Differential Equations**

## Learning objectives

By the end of this lesson you will be able to:

1. Derive a method's growth function $R(z)$ and plot its region of absolute stability.
2. Define A-stability and L-stability, and test for each without a magic threshold.
3. Recognise stiffness as a **ratio of eigenvalues**, and measure it from a Jacobian.
4. Show that on a stiff problem the step is set by stability rather than accuracy, and measure
   the waste.
5. Say why A-stability is not enough, and what L-stability adds.

## Prerequisites

Lesson 69 (explicit tableaux and the stability limit measured there). Lesson 71 (linear multistep
methods and the first Dahlquist barrier). Lesson 26 (eigenvalues, used to measure stiffness).
Lesson 67 (the test equation and Euler's $1 + \lambda h$ factor).

---

## 1. The test equation

Everything here is about one problem:

$$
y' = \lambda y, \qquad y(0) = 1, \qquad \operatorname{Re}\lambda < 0,
$$

whose solution $e^{\lambda t}$ decays. Apply a one step method and you get
$y_{n+1} = R(\lambda h) y_n$ for some function $R$, so $y_n = R(z)^n$ with $z = \lambda h$.

The computed solution decays if and only if $|R(z)| < 1$. That set of $z$ is the **region of
absolute stability**.

Why one linear scalar problem is enough: a linear system diagonalises, so it becomes one such
equation per eigenvalue, and a nonlinear problem behaves locally like its Jacobian. It is not a
theorem about the general case, and it predicts the general case remarkably well.

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
from nalib import stability as st

def growth_at(name, z):
    """The one step growth factor of a named method at a complex z = lam h."""
    return complex(st.growth_function(name)(z))

print(f"{'method':>20}{'R(-0.5)':>12}{'R(-2)':>12}{'R(-10)':>12}{'R(-1e6)':>14}")
for name in st.METHODS:
    values = [growth_at(name, z).real for z in (-0.5, -2.0, -10.0, -1e6)]
    print(f"{name:>20}" + "".join(f"{v:>12.6f}" if abs(v) < 1e5 else f"{v:>12.2e}"
                                  for v in values[:3]) + f"{values[3]:>14.3e}")
assert abs(growth_at("euler", -0.5) - 0.5) < 1e-15, "R(z) = 1 + z for Euler"
assert abs(growth_at("backward euler", -0.5) - 1.0 / 1.5) < 1e-15, "R(z) = 1/(1 - z)"
```

*Output:*

```text
              method     R(-0.5)       R(-2)      R(-10)       R(-1e6)
               euler    0.500000   -1.000000   -9.000000    -1.000e+06
                heun    0.625000    1.000000   41.000000     5.000e+11
         kutta third    0.604167   -0.333333 -125.666667    -1.667e+17
                 rk4    0.606771    0.333333  291.000000     4.167e+22
      backward euler    0.666667    0.333333    0.090909     1.000e-06
           trapezoid    0.600000    0.000000   -0.666667    -1.000e+00
   implicit midpoint    0.600000    0.000000   -0.666667    -1.000e+00
                bdf2    0.500000    0.285714    0.086957     1.000e-06
```

For an explicit method $R$ is a **polynomial**, so $|R(z)| \to \infty$ as $z \to -\infty$ and the
region is bounded. For an implicit method $R$ is a **rational function**, and it can stay small
forever. That single sentence is the whole of this lesson.

## 2. Regions of absolute stability

```python
print(f"{'method':>20}{'real axis limit':>18}{'imaginary axis limit':>22}"
      f"{'covers the left half plane':>28}")
for name in st.METHODS:
    out = st.stability_region(name, real=(-6.0, 2.0), imaginary=(-6.0, 6.0), points=161)
    limit = out["real_axis_limit"]
    text = "unbounded" if not np.isfinite(limit) else f"{limit:.6f}"
    print(f"{name:>20}{text:>18}{out['imaginary_axis_limit']:>22.6f}"
          f"{str(out['covers_the_left_half_plane']):>28}")
```

*Output:*

```text
              method   real axis limit  imaginary axis limit  covers the left half plane
               euler          2.000000              0.000001                       False
                heun          2.000000              0.001682                       False
         kutta third          2.512745              1.732051                       False
                 rk4          2.785294              2.828427                       False
      backward euler         unbounded                   inf                        True
           trapezoid         unbounded                   inf                        True
   implicit midpoint         unbounded                   inf                        True
                bdf2         unbounded                   inf                        True
```

Read the explicit rows first. Euler's region is the disc $|1 + z| < 1$, of radius 1, touching the
imaginary axis only at the origin. Heun's crosses the real axis at the same $-2$ but is a
different shape. RK4 reaches $-2.785$ on the real axis and $2\sqrt2 = 2.828$ up the imaginary
axis, which is why it can integrate an oscillation and Euler cannot.

```python
out = st.stability_region("rk4", real=(-3.5, 1.0), imaginary=(-3.5, 3.5), points=71)
rows = out["inside"]
for i in range(0, 71, 5):
    line = "".join("#" if rows[i, j] else "." for j in range(0, 71, 2))
    print(f"{out['imaginary'][i]:>7.2f} |{line}")
print(f"{'':>8}  " + "".join("^" if abs(out["real"][j]) < 0.06 else " "
                             for j in range(0, 71, 2)))
print(f"{'':>8}  (the caret marks the imaginary axis; real from "
      f"{out['real'][0]:.1f} to {out['real'][-1]:.1f})")
```

*Output:*

```text
  -3.50 |....................................
  -3.00 |....................................
  -2.50 |....................#########.......
  -2.00 |..............###############.......
  -1.50 |.........###################........
  -1.00 |.......#####################........
  -0.50 |......######################........
   0.00 |......######################........
   0.50 |......######################........
   1.00 |.......#####################........
   1.50 |.........###################........
   2.00 |..............###############.......
   2.50 |....................#########.......
   3.00 |....................................
   3.50 |....................................
                                     ^        
          (the caret marks the imaginary axis; real from -3.5 to 1.0)
```

```python
print(f"euler on the imaginary axis: |R(0.1i)| = "
      f"{abs(complex(st.growth_function('euler')(0.1j))):.10f}")
print(f"rk4   on the imaginary axis: |R(0.1i)| = "
      f"{abs(complex(st.growth_function('rk4')(0.1j))):.10f}")
```

*Output:*

```text
euler on the imaginary axis: |R(0.1i)| = 1.0049875621
rk4   on the imaginary axis: |R(0.1i)| = 0.9999999931
```

Euler's factor is greater than 1 for **every** purely imaginary $z$, so an undamped oscillation
grows under Euler at any step. That is not a small effect that a fine step fixes; it is a
qualitative failure, and lesson 74 is about it.

```python
fig, ax = plt.subplots()
for name in ("euler", "heun", "kutta third", "rk4"):
    region = st.stability_region(name, real=(-3.5, 1.0), imaginary=(-3.5, 3.5), points=301)
    ax.contour(region["real"], region["imaginary"], region["inside"].astype(float),
               levels=[0.5], linewidths=1.6)
    ax.plot([], [], label=name)
ax.axhline(0.0, color="k", lw=0.5); ax.axvline(0.0, color="k", lw=0.5)
ax.set_aspect("equal"); ax.set_xlabel("Re z"); ax.set_ylabel("Im z")
ax.set_title("regions of absolute stability, all bounded because all explicit")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/72_regions.png", dpi=110); plt.close(fig)
print("saved ../figures/72_regions.png")
```

*Output:*

```text
saved ../figures/72_regions.png
```

![Stability regions](../figures/72_regions.png)

The regions grow with the order and stay **bounded**, which is the whole limitation of an explicit
method. Note where each one crosses the imaginary axis: Euler and Heun touch it only at the origin
and the third and fourth order methods reach $\sqrt3$ and $2\sqrt2$ up it.

## 3. A-stability and L-stability

A method is **A-stable** when its region contains the whole left half plane: every decaying
problem is stable at every step size. No explicit method is A-stable, because its region is
bounded.

A method is **L-stable** when it is A-stable and $R(z) \to 0$ as $z \to -\infty$. That extra
condition is about the very fast modes: an A-stable method keeps them bounded, and an L-stable
method **kills** them, which is what a fast transient deserves.

```python
out = st.classify()
print(f"{'method':>20}{'A-stable':>10}{'L-stable':>10}{'|R| at -1e6':>14}"
      f"{'|R| at -1e10':>15}")
for n, a, l, m6, m10 in zip(out["names"], out["a_stable"], out["l_stable"],
                            out["magnitude_at_minus_1e6"],
                            out["magnitude_at_minus_1e10"]):
    print(f"{n:>20}{str(a):>10}{str(l):>10}{m6:>14.3e}{m10:>15.3e}")
```

*Output:*

```text
              method  A-stable  L-stable   |R| at -1e6   |R| at -1e10
               euler     False     False     1.000e+06      1.000e+10
                heun     False     False     5.000e+11      5.000e+19
         kutta third     False     False     1.667e+17      1.667e+29
                 rk4     False     False     4.167e+22      4.167e+38
      backward euler      True      True     1.000e-06      1.000e-10
           trapezoid      True     False     1.000e+00      1.000e+00
   implicit midpoint      True     False     1.000e+00      1.000e+00
                bdf2      True      True     7.071e-04      7.071e-06
```

### 3.1 Why L-stability is not tested with a threshold

BDF2's growth factor decays like $|z|^{-1/2}$, so at $z = -10^{10}$ it is still $7\times10^{-6}$.
Any fixed threshold at $10^{-6}$ calls an L-stable method not L-stable.

The test used here is **small and still falling**: compare $|R|$ two magnitudes apart and require
both that it be small and that it have dropped. That is what "tends to zero" means, and it does
not need a number chosen in advance.

```python
out = st.classify(["backward euler", "trapezoid", "bdf2"])
for n, m6, m10 in zip(out["names"], out["magnitude_at_minus_1e6"],
                      out["magnitude_at_minus_1e10"]):
    print(f"{n:>18}: {m6:.3e} -> {m10:.3e}   ratio {m10 / m6:.4f}")
```

*Output:*

```text
    backward euler: 1.000e-06 -> 1.000e-10   ratio 0.0001
         trapezoid: 1.000e+00 -> 1.000e+00   ratio 1.0000
              bdf2: 7.071e-04 -> 7.071e-06   ratio 0.0100
```

Backward Euler falls by $10^{-4}$ per four decades, BDF2 by $10^{-2}$, and the trapezoid rule not
at all: it sits at exactly 1 forever, because $R(z) \to -1$.

## 4. What stiffness is

A problem is **stiff** when it has decay rates differing by orders of magnitude, and you only care
about the slow one. The fast mode has died long before the interval is over; the step is still
being held down by it, because the method has to remain stable on a mode that no longer
contributes anything.

Stiffness is a property of the **Jacobian**, not of the entries of the equation.

```python
print(f"{'matrix':>34}{'ratio':>14}{'stiff':>8}")
examples = {
    "diag(-1, -1000)": [[-1.0, 0.0], [0.0, -1000.0]],
    "diag(-1, -2)": [[-1.0, 0.0], [0.0, -2.0]],
    "[[-500.5, 499.5], [499.5, -500.5]]": [[-500.5, 499.5], [499.5, -500.5]],
}
for label, A in examples.items():
    out = st.stiffness_ratio(A)
    print(f"{label:>34}{out['ratio']:>14.2f}{str(out['stiff']):>8}")
print(f"\n{st.stiffness_ratio([[-500.5, 499.5], [499.5, -500.5]])['note']}")
```

*Output:*

```text
                            matrix         ratio   stiff
                   diag(-1, -1000)       1000.00    True
                      diag(-1, -2)          2.00   False
[[-500.5, 499.5], [499.5, -500.5]]       1000.00    True

ratio of the fastest to the slowest decay rate
```

The third matrix has entries of similar size in every position and a stiffness ratio of 1000. Read
off the eigenvalues, not the entries.

## 5. The step is set by stability, not accuracy

Take $y' = Ay$ with eigenvalues $-1000$ and $-1$, over $[0, 5]$. The fast mode is at $10^{-435}$
by $t = 1$. Nothing about the answer needs it resolved. RK4's stability limit still forces
$h < 2.785/1000$.

```python
out = st.the_step_is_set_by_stability(name="rk4")
print(f"eigenvalues {out['fast']} and {out['slow']}, stiffness ratio "
      f"{out['stiffness_ratio']:.0f}")
print(f"stability forces h < {out['stability_step']:.3e}, "
      f"which is {out['steps_for_stability']} steps\n")
print(f"{'tolerance':>12}{'accuracy step':>16}{'steps for accuracy':>20}"
      f"{'wasted factor':>16}")
for tol, h, s, w in zip(out["tolerance"], out["accuracy_step"],
                        out["steps_for_accuracy"], out["wasted_factor"]):
    print(f"{tol:>12.0e}{h:>16.3e}{s:>20}{w:>16.2f}")
print(f"\nworst waste across the sweep: {out['worst_waste']:.1f}x")
```

*Output:*

```text
eigenvalues -1000.0 and -1.0, stiffness ratio 1000
stability forces h < 2.785e-03, which is 1796 steps

   tolerance   accuracy step  steps for accuracy   wasted factor
       1e-02       3.162e-01                  16          113.53
       1e-04       1.000e-01                  50           35.90
       1e-06       3.162e-02                 159           11.35
       1e-08       1.000e-02                 500            3.59
       1e-10       3.162e-03                1582            1.14

worst waste across the sweep: 113.5x
```

**The waste depends on how little accuracy you want.** At a tolerance of $10^{-2}$ the solver does
113 times more work than accuracy needs; at $10^{-10}$ the accuracy requirement has caught up and
the waste is gone.

That shape is the whole practical problem: a stiff problem is usually one where the fast transient
is a nuisance and low accuracy would do, which is exactly where the waste is worst.

```python
out = st.explicit_fails_on_a_stiff_problem()
print(f"{'steps':>8}{'lam*h':>12}{'explicit error':>18}{'implicit error':>18}")
for s, lh, ee, ie in zip(out["steps"], out["lam_h"], out["explicit_error"],
                         out["implicit_error"]):
    text = "overflow" if not np.isfinite(ee) else f"{ee:.3e}"
    print(f"{s:>8}{lh:>12.4f}{text:>18}{ie:>18.3e}")
print(f"\nRK4 becomes usable at {out['first_usable_explicit_steps']} steps; "
      f"backward Euler is usable everywhere: {out['implicit_usable_everywhere']}")
```

*Output:*

```text
   steps       lam*h    explicit error    implicit error
      10    100.0000         1.061e+66         1.766e-02
      50     20.0000        1.188e+187         3.648e-03
     100     10.0000        2.451e+246         1.832e-03
     500      2.0000         4.863e-14         3.676e-04
    1000      1.0000         4.108e-15         1.839e-04
    2000      0.5000         1.110e-16         9.195e-05
    5000      0.2000         9.437e-16         3.678e-05

RK4 becomes usable at 500 steps; backward Euler is usable everywhere: True
```

## 6. A-stable is not enough

The trapezoid rule is A-stable, and on a very stiff mode it is still wrong in a way that matters.
Its growth factor is $(1 + z/2)/(1 - z/2)$, which tends to $-1$. So a fast mode is not amplified,
and it is not damped either: it **alternates in sign forever**.

```python
out = st.trapezoid_rings(lam=-1000.0, steps=10, t_end=1.0)
print(f"lam*h = {out['lam_h']:.1f}, exact answer at t = 1 is {out['exact']:.3e}\n")
print(f"{'step':>6}{'trapezoid':>16}{'backward euler':>18}")
for k, (a, b) in enumerate(zip(out["trapezoid_values"], out["backward_euler_values"])):
    print(f"{k:>6}{a:>16.6f}{b:>18.3e}")
print(f"\ntrapezoid growth factor {out['trapezoid_growth']:.6f}, alternates: "
      f"{out['trapezoid_alternates']}")
print(f"backward Euler growth factor {out['backward_euler_growth']:.6f}")
```

*Output:*

```text
lam*h = -100.0, exact answer at t = 1 is 0.000e+00

  step       trapezoid    backward euler
     0        1.000000         1.000e+00
     1       -0.960784         9.901e-03
     2        0.923106         9.803e-05
     3       -0.886906         9.706e-07
     4        0.852126         9.610e-09
     5       -0.818709         9.515e-11
     6        0.786603         9.420e-13
     7       -0.755756         9.327e-15
     8        0.726118         9.327e-15
     9       -0.697643         9.327e-15
    10        0.670284         9.327e-15

trapezoid growth factor -0.960784, alternates: True
backward Euler growth factor 0.009901
```

Backward Euler kills the mode: $10^{-2}$ after one step and $10^{-14}$ after seven. The
trapezoid rule loses only 3.9 per cent per step, so after ten steps it is still at $0.67$, sign
flipping the whole way, while the true answer has been $10^{-435}$ since the first step.

Both are A-stable. Only one of them is usable here, and the difference is not in the stability
region at all: it is in what $R$ does at $-\infty$.

```python
print(f"{'lam':>10}{'trapezoid factor':>20}{'backward euler factor':>24}")
for lam in (-10.0, -100.0, -1000.0, -10000.0):
    out = st.trapezoid_rings(lam=lam, steps=10, t_end=1.0)
    print(f"{lam:>10.0f}{out['trapezoid_growth']:>20.6f}"
          f"{out['backward_euler_growth']:>24.6f}")
```

*Output:*

```text
       lam    trapezoid factor   backward euler factor
       -10            0.333333                0.500000
      -100           -0.666667                0.090909
     -1000           -0.960784                0.009901
    -10000           -0.996008                0.000999
```

**The stiffer the problem, the worse the ringing**, which is the opposite of what is wanted. That
is precisely what L-stability rules out.

## 7. The second Dahlquist barrier

**No A-stable linear multistep method has order greater than 2**, and among the order 2 ones the
trapezoid rule has the smallest error constant.

```python
out = st.second_dahlquist_barrier()
print(f"{'method':>20}{'order':>7}{'A-stable':>10}")
for n, p, a in zip(out["names"], out["order"], out["a_stable"]):
    print(f"{n:>20}{p:>7}{str(a):>10}")
print(f"\nhighest A-stable order found: {out['highest_a_stable_order_found']}, "
      f"barrier holds: {out['barrier_holds']}")
print(out["note"])
```

*Output:*

```text
              method  order  A-stable
               euler      1     False
           trapezoid      2      True
      backward euler      1      True
                 ab2      2     False
                 am3      3     False
                bdf2      2      True
                bdf3      3     False
                 ab4      4     False

highest A-stable order found: 2, barrier holds: True
no A-stable linear multistep method exceeds order 2; the trapezoid rule attains it with the smallest error constant
```

A-stability is tested here from each method's **own coefficients**, through the roots of
$\rho(w) - z\sigma(w)$, not through a one step growth function. A three step method has three
growth factors, and zero-stability from lesson 71 is exactly the case $z = 0$.

### 7.1 What the BDF family settles for instead

Past order 2 the BDF methods give up A-stability and keep a **wedge**: they are stable wherever
$|\arg(-z)| < \alpha$ for some angle $\alpha$ short of 90 degrees. That is A($\alpha$)-stability,
and it is enough whenever the eigenvalues are not too close to the imaginary axis.

```python
out = st.bdf_angles()
print(f"{'order':>7}{'measured angle':>17}{'published':>12}{'gap':>9}"
      f"{'A-stable':>10}{'stable on the real axis':>26}")
for k, a, p, g, astab, real in zip(out["order"], out["angle"], out["published_angle"],
                                   out["gap"], out["a_stable"],
                                   out["stable_on_the_negative_real_axis"]):
    print(f"{k:>7}{a:>17.3f}{p:>12.2f}{g:>9.3f}{str(astab):>10}{str(real):>26}")
print(f"\nworst gap against the published table: {out['worst_gap']:.3f} degrees")
print(f"A-stable only up to order {out['a_stable_only_up_to']}, "
      f"and the wedge closes monotonically: {out['the_wedge_closes']}")
assert out["matches_the_published_table"], "measured against the published BDF angles"
assert out["a_stable_only_up_to"] == 2, "which is the second Dahlquist barrier"
assert out["the_wedge_closes"]
```

*Output:*

```text
  order   measured angle   published      gap  A-stable   stable on the real axis
      1           90.000       90.00    0.000      True                      True
      2           90.000       90.00    0.000      True                      True
      3           86.112       86.03    0.082     False                      True
      4           73.392       73.35    0.042     False                      True
      5           51.880       51.84    0.040     False                      True
      6           18.006       17.84    0.166     False                      True
      7            0.000        0.00    0.000     False                     False

worst gap against the published table: 0.166 degrees
A-stable only up to order 2, and the wedge closes monotonically: True
```

The angles are found by **bisection** on the angle, testing radii from $10^{-4}$ to $10^{8}$ at
each one, and they land within $0.17$ degrees of the published values. Reading them off a
rectangular grid would give whichever answer the corners happened to fall on.

**BDF7 has an angle of zero**: it is not stable even on the negative real axis, which is the same
zero-stability failure as lesson 71's example. That is why every production stiff solver stops at
order 6.

This is why stiff solvers are their own subject. Wanting high order and A-stability together means
giving up something:

- **BDF methods** give up A-stability past order 2, keeping a wedge instead: 86.1, 73.4, 51.9
  and 18.0 degrees at orders 3 to 6, and nothing at 7. That is where `LSODA` and `CVODE` stop.
- **Implicit Runge-Kutta** methods are not linear multistep methods, so the barrier does not
  apply. Radau IIA is order 5 and L-stable, at the cost of solving a coupled nonlinear system of
  three stages at once.
- **Rosenbrock methods** linearise the implicit solve, giving up exactness of the stage solve for
  a single linear system per step.

## 8. Exercises

**Level 1, understanding**

1.1 Derive $R(z)$ for Euler, backward Euler and the trapezoid rule.

1.2 Explain why no explicit method can be A-stable.

1.3 Define A-stability and L-stability and give a method that is the first and not the second.

1.4 Explain why stiffness is a property of the eigenvalues rather than of the equation's
coefficients.

1.5 Say why the waste in section 5 is worst when the accuracy demanded is lowest.

**Level 2, derivation**

2.1 Derive Euler's stability region as the disc $|1 + z| < 1$ and find its real and imaginary
axis limits.

2.2 Show that an explicit $s$ stage method of order $s$ has
$R(z) = \sum_{j=0}^{s} z^j / j!$ and use it to find RK4's real axis limit numerically.

2.3 Show that the trapezoid rule's growth factor tends to $-1$ and derive the ringing frequency.

2.4 Show BDF2's growth factor and its behaviour as $z \to -\infty$, and confirm the $|z|^{-1/2}$
decay.

2.5 Prove that A-stability of a one step method is equivalent to $|R(z)| \le 1$ on the imaginary
axis plus $R$ being analytic in the left half plane.

**Level 3, computational**

3.1 Plot the stability regions of every method in `METHODS` on one figure and identify each.

3.2 Implement a stiff solver using backward Euler with a Newton solve, and apply it to the
Hodgkin-Huxley system of lesson 73.

3.3 Implement BDF2 through BDF6 and plot their stability regions, showing the wedge of instability
opening as the order rises.

3.4 Implement the two stage Radau IIA method and confirm it is L-stable and of order 3.

3.5 Implement a Rosenbrock method of order 2 and compare its cost per step against backward Euler
with Newton.

**Level 4, experimental**

4.1 Measure the wasted factor of section 5 against the stiffness ratio over four decades and find
the relationship.

4.2 Measure how many steps backward Euler needs to reach a given accuracy against how many RK4
needs, across the stiffness range where the crossover happens.

4.3 Measure the trapezoid rule's ringing amplitude against $\lambda h$ and find where it stops
being visible.

**Level 5, advanced**

5.1 **Proving the second barrier.** Sketch Dahlquist's proof that A-stability caps a linear
multistep method at order 2, and identify where the linearity of the method is used.

5.2 **Order reduction on stiff problems.** An implicit Runge-Kutta method can lose order on a
stiff problem. Explain the mechanism in terms of the stage order and construct an example.

5.3 **Stiffness without a Jacobian.** Design a test that detects stiffness from a sequence of
solver steps alone, without forming the Jacobian, and say what it would cost.

## 9. Key takeaways

- **The growth function decides everything.** Explicit means polynomial means bounded region;
  implicit means rational means it can cover the left half plane.

- **Euler is unstable on every purely imaginary $z$**, so it cannot integrate an oscillation at
  any step size. RK4 reaches $2\sqrt2$ up the imaginary axis and can.

- **A-stability is not tested with a threshold.** BDF2's growth factor is still $7\times10^{-6}$
  at $z = -10^{10}$, so a fixed cutoff misclassifies it. Test whether the factor is small **and
  still falling.**

- **Stiffness is a ratio of eigenvalues.** A matrix with all its entries around 500 can have a
  stiffness ratio of 1000.

- **The waste is worst when the accuracy wanted is lowest**: 113 times at a tolerance of
  $10^{-2}$ and none at $10^{-10}$. That is exactly the situation stiff problems present.

- **A-stable is not enough.** At $\lambda h = -100$ the trapezoid rule loses 3.9 per cent per
  step and alternates in sign, so a mode whose true value is $10^{-435}$ is still at $0.67$ after
  ten steps. The factor goes $-0.667$, $-0.961$, $-0.996$ as the problem stiffens, so **the
  ringing gets worse the stiffer the problem**, which is exactly backwards. L-stability rules it
  out.

- **The second Dahlquist barrier caps A-stable multistep methods at order 2**, confirmed here by
  testing eight named methods from their own coefficients: the highest A-stable order found is 2.

- **What the BDF family settles for is a closing wedge**: 90, 90, 86.1, 73.4, 51.9, 18.0, 0
  degrees. Every serious stiff solver is either a BDF method living inside that wedge, or an
  implicit Runge-Kutta method escaping the barrier by not being a multistep method at all.

## Where this goes next

Lesson 73 applies everything so far to systems, where the stiffness ratio becomes something you
have to measure along the trajectory rather than once. Lesson 74 comes back to section 2's
imaginary axis and finds that even RK4, which is stable there, gets the long term behaviour of a
Hamiltonian system wrong in a way no step size fixes.
