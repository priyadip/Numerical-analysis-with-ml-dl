# Solutions: Part 10, Ordinary Differential Equations

Solutions to every exercise in lessons 67 to 76, 210 in all: 21 for lesson 67, 21 for lesson 68, 21 for lesson 69, 21 for lesson 70, 21 for lesson 71, 21 for lesson 72, 21 for lesson 73, 21 for lesson 74, 21 for lesson 75, 21 for lesson 76.

Every number quoted here was measured by running the code, on this repository, with the seeds
shown. Where a measurement contradicted the claim the exercise expects, the measurement is
reported and the claim is corrected. Several exercises in this part end in a negative result for
that reason, and those are marked rather than replaced with an example that would have worked.

Run any block from the repository root. Each one is self contained apart from `nalib`, and the
blocks within one lesson share a namespace, so they are meant to be run in order.

```python
import sys
sys.path.insert(0, "src")
```

---

## Lesson 67, Initial Value Problems and Euler's Method

### 1.1 The Lipschitz condition, and the two gaps around it

$f$ is **Lipschitz in $y$** on a region $D$ if there is a constant $L$ with

$$
\lvert f(t, u) - f(t, v)\rvert \le L\lvert u - v\rvert \quad\text{for all } (t,u), (t,v) \in D .
$$

It sits strictly between continuity and differentiability, and both gaps are occupied.

**Continuous and not Lipschitz:** $f(y) = \sqrt{\lvert y\rvert}$ near $y = 0$. Take $v = 0$ and
$u = \delta$: the ratio is $\sqrt\delta/\delta = \delta^{-1/2}$, which is unbounded as
$\delta \to 0$. The function is perfectly continuous and no constant works.

**Lipschitz and not differentiable:** $f(y) = \lvert y\rvert$. The ratio is at most 1 everywhere by
the reverse triangle inequality, so $L = 1$, and there is no derivative at $y = 0$.

The second example is why the condition is stated this way rather than as "$f$ is $C^1$". Lipschitz
is exactly what the contraction argument of lesson 68 needs, and demanding differentiability would
exclude problems the theory handles fine.

### 1.2 Why $y' = y^{2/3}$ is ambiguous at 0 and not at 1

The right hand side has $\partial f/\partial y = \tfrac23 y^{-1/3}$, which is bounded on any region
away from $y = 0$ and unbounded on any region containing it.

**At $y(0) = 1$** take the region $D = \{\lvert y - 1\rvert \le \tfrac12\}$. There
$\lvert \partial f/\partial y\rvert \le \tfrac23 (1/2)^{-1/3} = 0.84$, so $f$ is Lipschitz on $D$
and Picard-Lindelof gives exactly one solution for as long as the solution stays in $D$.

**At $y(0) = 0$** every region containing the initial point contains points where the derivative is
arbitrarily large, so no $L$ exists. And the uniqueness genuinely fails: for any $c \ge 0$,

$$
y_c(t) = \begin{cases} 0 & t \le c \\ \left(\frac{t-c}{3}\right)^3 & t > c\end{cases}
$$

satisfies $y_c(0) = 0$ and $y_c' = y_c^{2/3}$ everywhere, including at $t = c$ where both sides are
zero. There are uncountably many solutions and no numerical method can prefer one.

```python
import numpy as np
from nalib import ivp

out = ivp.uniqueness_fails_on()
for c, curve, r in zip(out["delays"], out["solutions"], out["worst_residual"]):
    print(f"takes off at {c:.2f}: y(2) = {curve[-1]:.6f}, residual {r:.2e}")
print("all start at exactly zero:", out["all_start_at_zero"])
```

### 1.3 Euler from the Taylor series

Expand the exact solution about $t_n$:

$$
y(t_{n+1}) = y(t_n) + h y'(t_n) + \frac{h^2}{2}y''(\xi_n), \qquad \xi_n \in (t_n, t_{n+1}).
$$

Substituting $y' = f(t, y)$ and dropping the last term gives $y_{n+1} = y_n + h f(t_n, y_n)$.

The **local truncation error** is what was dropped, measured per step:

$$
\tau_{n+1} = \frac{y(t_{n+1}) - y(t_n)}{h} - f(t_n, y(t_n)) = \frac{h}{2}y''(\xi_n) = O(h).
$$

Written as an absolute per step error rather than a per unit time one it is $\tfrac{h^2}{2}y''$,
which is $O(h^2)$. Both conventions appear in the literature and they differ by one power of $h$,
which is a standing source of confusion; the lesson uses the absolute form.

### 1.4 Why one order is lost

Two effects, and they push the same way.

**Counting.** Over $[t_0, T]$ there are $N = (T - t_0)/h$ steps. If each contributes an error of
size $Ch^{p+1}$ and they simply added, the total would be $N \cdot Ch^{p+1} = C(T-t_0)h^p$. The
factor $1/h$ from the step count eats one power.

**Propagation.** The errors do not simply add, because an error made at step $n$ is then carried
through every later step and amplified by the method's own dynamics. Let $e_n = y(t_n) - y_n$.
Subtracting the method from the exact relation and using the Lipschitz condition,

$$
\lvert e_{n+1}\rvert \le (1 + hL)\lvert e_n\rvert + \frac{h^2}{2}M .
$$

Solving the recurrence gives $\lvert e_N \rvert \le \frac{hM}{2L}(e^{L(T-t_0)} - 1)$, which is the
bound of section 5. The $h^2$ per step has become $h$ overall, and the amplification appears as
the exponential.

So **local order $p+1$ gives global order $p$**, and the exponential factor is the price of
allowing the errors to grow rather than assuming they cancel.

### 1.5 What $e^{L(t-t_0)}$ represents

It is the **worst case amplification of a perturbation** by the differential equation itself, not
by the method.

Two solutions of the same equation starting $\delta$ apart satisfy
$\lvert y_1(t) - y_2(t)\rvert \le \delta e^{L(t-t_0)}$, by Gronwall's inequality, and that bound is
attained by $y' = Ly$. So the factor is the condition number of the problem over the interval, and
no method can do better than it.

It is a worst case in two ways at once. It uses the largest $L$ over the whole region when the
local sensitivity varies, and it assumes every error is amplified in the growing direction. On a
decaying problem such as $y' = -2y$ the true amplification is $e^{-2t}$, so the bound is wrong by
$e^{4t}$, which at $t = 1$ is a factor of 55.

### 2.1 Deriving the classical bound

Write $e_n = y(t_n) - y_n$. The exact solution satisfies

$$
y(t_{n+1}) = y(t_n) + hf(t_n, y(t_n)) + \frac{h^2}{2}y''(\xi_n),
$$

and the method satisfies $y_{n+1} = y_n + hf(t_n, y_n)$. Subtract:

$$
e_{n+1} = e_n + h\left[f(t_n, y(t_n)) - f(t_n, y_n)\right] + \frac{h^2}{2}y''(\xi_n).
$$

**Here the Lipschitz hypothesis is used**, to bound the bracket by $L\lvert e_n\rvert$, and
**here the bound $\lvert y''\rvert \le M$ is used**, to bound the last term:

$$
\lvert e_{n+1}\rvert \le (1 + hL)\lvert e_n\rvert + \frac{h^2 M}{2}.
$$

This is a linear recurrence $a_{n+1} \le \alpha a_n + \beta$ with $\alpha = 1 + hL > 1$, whose
solution with $a_0 = 0$ is

$$
a_n \le \frac{\beta}{\alpha - 1}(\alpha^n - 1) = \frac{h^2M/2}{hL}\left((1+hL)^n - 1\right).
$$

Finally $(1 + hL)^n \le e^{nhL} = e^{L(t_n - t_0)}$, using $1 + x \le e^x$, which gives

$$
\lvert e_n \rvert \le \frac{hM}{2L}\left(e^{L(t_n-t_0)} - 1\right).
$$

Three places where slack enters: $L$ is a supremum, $M$ is a supremum, and $1 + x \le e^x$ is an
inequality. The measured ratio of 23 in the lesson is the product of those three.

### 2.2 Euler on the test equation, exactly

For $y' = \lambda y$ the method reads $y_{n+1} = y_n + h\lambda y_n = (1 + \lambda h)y_n$, so by
induction $y_n = (1 + \lambda h)^n y_0$ **exactly**, with no truncation error argument at all.

The exact solution is $y(t_n) = e^{\lambda h n}y_0$, so the method replaces $e^{z}$ by $1 + z$ with
$z = \lambda h$, which is its Taylor polynomial of degree 1. That is the whole method.

For $\lambda < 0$ the exact solution decays. The computed one decays only when
$\lvert 1 + \lambda h\rvert < 1$, that is $-2 < \lambda h < 0$. **For $\lambda h < -2$ the computed
solution grows in magnitude and alternates in sign**, so the answer is qualitatively wrong however
small the truncation error per step is.

```python
for steps in (8, 10, 12, 20):
    h = 1.0 / steps
    lam = -25.0
    factor = 1.0 + lam * h
    print(f"h = {h:.4f}, lam*h = {lam * h:6.2f}, growth {factor:8.4f}, "
          f"y(1) = {factor ** steps:12.4e}, exact {np.exp(lam):.4e}")
```

That is the stability limit of lesson 72, derived here from three lines of algebra.

### 2.3 Euler's error constant, and the logistic degeneracy

From exercise 1.3 the local truncation error over one step is
$\tfrac{h^2}{2}y''(\xi_n)$, so the **error constant** is $\tfrac12 y''$.

The logistic equation $y' = y(1-y)$ has

$$
y'' = \frac{d}{dt}\left[y(1-y)\right] = y'(1 - 2y) = y(1-y)(1-2y),
$$

which vanishes at $y = 0$, $y = 1$ and $y = \tfrac12$. The last is the inflection point of the
logistic curve.

Starting at $y(0) = \tfrac12$ the solution stays at $\tfrac12$ for all time, because that is an
inflection point of the curve through it and $y'' = 0$ there identically along the solution. So the
leading local error term is exactly zero at every step and the next term shows through, giving a
local order of 3 rather than 2.

```python
def logistic(t, y):
    v = ivp.as_state(y)
    return v * (1.0 - v)

def logistic_exact(t, y0):
    return y0 / (y0 + (1.0 - y0) * np.exp(-t))

for start in (0.5, 0.499, 0.45, 0.2):
    out = ivp.local_against_global(logistic, lambda t: logistic_exact(t, start),
                                   0.0, start, 2.0)
    print(f"y(0) = {start:5.3f}: local order {out['local_order']:.4f}, "
          f"global order {out['global_order']:.4f}")
```

Note the third column of that sweep: moving the start by $10^{-3}$ already restores order 2, so
this is a measure zero degeneracy rather than a region. It is still worth knowing about, because
$\tfrac12$ is exactly the value someone picks when choosing a "nice" starting point.

### 2.4 Backward Euler from the right endpoint rule

The right endpoint quadrature rule on $[t_n, t_{n+1}]$ is
$\int_{t_n}^{t_{n+1}} g \approx h\,g(t_{n+1})$, so

$$
y_{n+1} = y_n + h f(t_{n+1}, y_{n+1}),
$$

which is implicit in $y_{n+1}$.

Its local truncation error comes from expanding about $t_{n+1}$ instead:

$$
y(t_n) = y(t_{n+1}) - hy'(t_{n+1}) + \frac{h^2}{2}y''(\eta),
$$

so $y(t_{n+1}) = y(t_n) + hf(t_{n+1}, y(t_{n+1})) - \frac{h^2}{2}y''(\eta)$. The dropped term is
again $O(h^2)$, so **the local order is 2 and the global order is 1**, the same as forward Euler.

The two methods have error constants $+\tfrac12 y''$ and $-\tfrac12 y''$, equal in size and
opposite in sign, which is why averaging them gives the trapezoid rule and one more order. Their
difference in practice is entirely stability, which is lesson 72.

### 2.5 The solution is Lipschitz in $t$

Suppose $\lvert f(t,u) - f(t,v)\rvert \le L\lvert u - v\rvert$ and
$\lvert f(t,y) - f(s,y)\rvert \le K\lvert t - s\rvert$ on the region, and let $B$ bound
$\lvert f \rvert$ there. Then for $s < t$,

$$
\lvert y(t) - y(s)\rvert = \left\lvert \int_s^t f(r, y(r))\,dr \right\rvert
\le \int_s^t \lvert f(r, y(r))\rvert\,dr \le B(t - s).
$$

So the solution is Lipschitz in $t$ with constant $B = \sup\lvert f\rvert$, and neither $L$ nor $K$
is needed for that.

Where they do enter is the **derivative's** regularity. Differentiating $y' = f(t,y)$,

$$
\lvert y'(t) - y'(s)\rvert = \lvert f(t, y(t)) - f(s, y(s))\rvert
\le K\lvert t - s\rvert + L\lvert y(t) - y(s)\rvert \le (K + LB)\lvert t - s\rvert,
$$

so $y'$ is Lipschitz with constant $K + LB$, and hence $\lvert y''\rvert \le K + LB$ wherever it
exists. **That is where the $M$ in the error bound comes from**: it is not an extra hypothesis,
it follows from the two Lipschitz conditions.

### 3.1 Euler for a system, verified in three dimensions

```python
def spiral(t, y):
    v = ivp.as_state(y)
    return np.asarray([-v[1] - 0.1 * v[0], v[0] - 0.1 * v[1], -0.5 * v[2]])

def spiral_exact(t):
    return np.asarray([np.exp(-0.1 * t) * np.cos(t), np.exp(-0.1 * t) * np.sin(t),
                       np.exp(-0.5 * t)])

out = ivp.error_against_step(spiral, spiral_exact, 0.0, [1.0, 0.0, 1.0], 3.0,
                             [20, 40, 80, 160, 320, 640], ivp.euler_step)
print(f"{'steps':>8}{'error':>14}{'ratio':>9}")
last = None
for n, e in zip(out["steps"], out["errors"]):
    print(f"{n:>8}{e:>14.4e}" + ("" if last is None else f"{last / e:>9.3f}"))
    last = e
print(f"fitted order {out['fitted_order']:.4f}")
assert abs(out["fitted_order"] - 1.0) < 0.05
```

The ratio settles at 2 and the fitted order at 1. Nothing in `ivp.integrate` or `ivp.euler_step`
knows the dimension: `as_state` makes a 1-D array of whatever length arrives and every operation is
elementwise.

### 3.2 Backward Euler on a stiff problem

```python
def stiff(t, y):
    return -100.0 * ivp.as_state(y)

def stiff_exact(t):
    return np.exp(-100.0 * t)

print(f"{'h':>8}{'lam*h':>9}{'forward':>16}{'backward':>16}{'exact':>14}")
for steps in (5, 10, 50, 200):
    h = 1.0 / steps
    fwd = ivp.integrate(stiff, 0.0, 1.0, 1.0, steps, ivp.euler_step)
    bwd = ivp.integrate(stiff, 0.0, 1.0, 1.0, steps, ivp.backward_euler_step)
    print(f"{h:>8.3f}{-100.0 * h:>9.2f}{fwd['y'][-1, 0]:>16.4e}"
          f"{bwd['y'][-1, 0]:>16.4e}{stiff_exact(1.0):>14.4e}")
```

At $h = 0.1$ forward Euler has $\lambda h = -10$ and returns $(1-10)^{10} = 3.5\times10^{9}$ for a
solution whose true value is $3.7\times10^{-44}$. Backward Euler returns $(1/11)^{10}$, which is
$3.9\times10^{-11}$: **wrong by thirty three orders of magnitude in the relative sense and
qualitatively right**, because it decays.

That is the distinction lesson 72 formalises. Backward Euler is not accurate here; it is stable,
and on a stiff problem stability is what is being bought.

### 3.3 The midpoint method

```python
def midpoint_step(f, t, y, h, _counter=None):
    """One explicit midpoint step: sample the slope at the middle of the interval."""
    state = ivp.as_state(y)
    if _counter is not None:
        _counter[0] += 2
    k1 = ivp.as_state(f(t, state))
    return state + h * ivp.as_state(f(t + 0.5 * h, state + 0.5 * h * k1))

def decay(t, y):
    return -2.0 * ivp.as_state(y)

out = ivp.error_against_step(decay, lambda t: np.exp(-2.0 * t), 0.0, 1.0, 1.0,
                             [10, 20, 40, 80, 160, 320], midpoint_step)
print(f"fitted order {out['fitted_order']:.4f}, error at 320 steps {out['errors'][-1]:.3e}")
local = ivp.local_against_global(decay, lambda t: np.exp(-2.0 * t), 0.0, 1.0, 1.0,
                                 step_fn=midpoint_step)
print(f"local order {local['local_order']:.4f}, global order {local['global_order']:.4f}")
```

Second order, at two evaluations per step. This is the $\alpha = \tfrac12$ member of lesson 69's
one parameter family, and deriving it from the midpoint quadrature rule is exactly how that family
is found.

### 3.4 The growth exponent of $\sqrt{\lvert y\rvert}$

```python
def root(t, y):
    v = ivp.as_state(y)
    return np.sqrt(np.abs(v))

out = ivp.lipschitz_near(root, 0.0, 0.0)
print(f"{'radius':>12}{'estimated L':>16}")
for r, e in zip(out["radius"], out["estimate"]):
    print(f"{r:>12.2e}{e:>16.4f}")
print(f"fitted exponent {out['growth_exponent']:.4f}, expected -0.5")
assert abs(out["growth_exponent"] + 0.5) < 0.05
```

Exactly $-1/2$, as $\tfrac12 y^{-1/2}$ predicts. Compare with $y^{2/3}$, whose exponent is
$-1/3$: **the exponent identifies which singularity it is**, which a ratio test cannot do.

### 3.5 The roundoff floor in single precision

```python
def euler_single(steps):
    """Euler on y' = -2y in float32, returning the error at t = 1."""
    h = np.float32(1.0) / np.float32(steps)
    y = np.float32(1.0)
    for _ in range(steps):
        y = y + h * (np.float32(-2.0) * y)
    return abs(float(y) - float(np.exp(-2.0)))

eps32 = float(np.finfo(np.float32).eps)
print(f"single precision eps = {eps32:.3e}, sqrt = {eps32 ** 0.5:.3e}, "
      f"eps^(2/3) = {eps32 ** (2.0 / 3.0):.3e}\n")
print(f"{'steps':>10}{'h':>12}{'error':>14}{'ratio':>9}")
best, best_h = None, None
last = None
for k in range(3, 22):
    n = 2 ** k
    e = euler_single(n)
    print(f"{n:>10}{1.0 / n:>12.3e}{e:>14.4e}" + ("" if last is None else f"{last / e:>9.3f}"))
    if best is None or e < best:
        best, best_h = e, 1.0 / n
    last = e
print(f"\nsmallest error {best:.3e} at h = {best_h:.3e}")
print(f"h/sqrt(eps) = {best_h / eps32 ** 0.5:.3f}, "
      f"h/eps^(2/3) = {best_h / eps32 ** (2.0 / 3.0):.3f}")
```

The ratio column is the thing to read. It sits at 2.00 for eleven refinements, wobbles at
$h = 1.5\times10^{-5}$, and then **inverts**: 0.362, 0.227, 0.228, 0.253. Past the minimum each
halving of the step multiplies the error by about 4, which is the $\varepsilon/h$ term taking
over.

The minimum is at $h = 7.6\times10^{-6}$, with an error of $1.3\times10^{-6}$. Compare the two
candidate predictions:

| prediction | value | measured $h_{\text{opt}}$ over it |
|---|---|---|
| $\sqrt{\varepsilon_{32}}$ | $3.45\times10^{-4}$ | 0.022 |
| $\varepsilon_{32}^{2/3}$ | $2.42\times10^{-5}$ | 0.315 |

**The measured minimum is a factor of 3 below $\varepsilon^{2/3}$ and a factor of 45 below
$\sqrt\varepsilon$**, so the $2/3$ law is right to within a constant and the $1/2$ law is not
right at all. That agrees with the double precision sweep from the other side: there the crossing
sits at $3.7\times10^{-11}$, which no run reaches, so the sweep can only show the error still
falling.

Single precision is the honest way to see this effect, because there the crossing is reachable in a
few million steps.

### 4.1 How the bound's overestimate depends on $\lambda$

For $y' = \lambda y$ on $[0, 1]$ the bound's ingredients are known in closed form:
$L = \lvert\lambda\rvert$ and $M = \sup\lvert y''\rvert = \lambda^2 \max(1, e^{\lambda})$. So the
bound is computable and so is the exact Euler error, and the ratio can be predicted rather than
only measured.

```python
print(f"{'lambda':>8}{'observed':>13}{'bound':>13}{'ratio':>13}"
      f"{'e^(2|lam|)':>14}{'e^lam / lam':>14}{'ratio / law':>13}")
for lam in (-10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0):
    def f(t, y, lam=lam):
        return lam * ivp.as_state(y)

    def exact(t, lam=lam):
        return np.exp(lam * t)

    L = abs(lam)
    M = lam ** 2 * max(1.0, np.exp(lam))          # sup |y''| over [0, 1]
    out = ivp.error_bound(f, exact, 0.0, 1.0, 1.0, L, M, step_counts=[80])
    ratio = float(out["bound_over_observed"][0])
    decaying = np.exp(2.0 * L)
    growing = np.exp(lam) / lam if lam > 0 else float("nan")
    law = decaying if lam < 0 else growing
    print(f"{lam:>8.1f}{out['observed_error'][0]:>13.3e}{out['bound'][0]:>13.3e}"
          f"{ratio:>13.3e}{decaying:>14.3e}{growing:>14.3e}{ratio / law:>13.3f}")
```

**The ratio is not monotone in $\lambda$.** It is smallest at $\lambda = +1$, where it is 1.74, and
grows in **both** directions: $6\times10^{7}$ at $\lambda = -10$ and $3\times10^{3}$ at
$\lambda = +10$. Two different mechanisms, one on each side, and the last column checks each.

**For $\lambda < 0$ the exponential has the wrong sign.** The bound multiplies by
$e^{L} = e^{\lvert\lambda\rvert}$ where the true amplification of an error is
$e^{\lambda} = e^{-\lvert\lambda\rvert}$, so the ratio carries a factor of
$e^{2\lvert\lambda\rvert}$. The last column reads 0.13 to 0.63 against that law across four decades
of ratio, so the law has the growth right and a slowly varying constant.

**For $\lambda > 0$ the exponential is counted twice.** Here $e^{L}$ is the correct amplification,
but $M = \lambda^2 e^{\lambda}$ **already contains** $e^{\lambda}$, because $\lvert y''\rvert$ is
largest at the far end of the interval. The bound then multiplies that supremum by the growth
factor again, giving $e^{2\lambda}$ where the truth has $e^{\lambda}$. Working through the algebra,

$$
\frac{\text{bound}}{\text{true}} \approx
\frac{(h/2)(M/L)(e^{L}-1)}{(h\lambda^2/2)e^{\lambda}} \approx \frac{e^{\lambda}}{\lambda},
$$

and the last column reads 0.64, 0.89, 1.11, 1.43 against it, which is as close as a leading order
estimate gets.

So **the bound is worst where the problem is easiest and worst again where it is hardest**, and it
is within a factor of two only in the narrow band where the solution neither grows nor decays much.
That is the general shape of a bound assembled from independent suprema: each one is attained
somewhere, they are attained in different places, and the bound pays for all of them at once.

### 4.2 Three more starting points where the local order exceeds 2

The rule from exercise 2.3 is: start where $y'' = 0$ along the solution.

```python
cases = []

# 1. y' = y(1-y) at the inflection point, from the lesson
cases.append(("logistic at y = 1/2", lambda t, y: ivp.as_state(y) * (1.0 - ivp.as_state(y)),
              0.5, lambda t: 0.5 / (0.5 + 0.5 * np.exp(-t))))

# 2. y' = cos(t) started anywhere: y'' = -sin(t) vanishes at t = 0
cases.append(("y' = cos t from t = 0",
              lambda t, y: np.full_like(ivp.as_state(y), np.cos(t)), 0.0,
              lambda t: np.sin(t)))

# 3. y' = y - t, whose y'' = y - t - 1 vanishes when y = t + 1
cases.append(("y' = y - t from y(0) = 1",
              lambda t, y: ivp.as_state(y) - t, 1.0, lambda t: t + 1.0))

# 4. y' = -y^3, whose y'' = 3y^5 vanishes only at y = 0, which is the fixed point
cases.append(("y' = -y^3 from y(0) = 1 (control, no degeneracy)",
              lambda t, y: -ivp.as_state(y) ** 3, 1.0,
              lambda t: 1.0 / np.sqrt(1.0 + 2.0 * t)))

print(f"{'case':>50}{'local':>9}{'global':>9}{'local error at 320':>21}")
for label, f, y0, exact in cases:
    out = ivp.local_against_global(f, exact, 0.0, y0, 1.0)
    local = "exact" if not np.isfinite(out["local_order"]) else f"{out['local_order']:.3f}"
    glob = "exact" if not np.isfinite(out["global_order"]) else f"{out['global_order']:.3f}"
    print(f"{label:>50}{local:>9}{glob:>9}{out['local_error'][-1]:>21.3e}")
```

All three degenerate cases report a local order of 3, and the control reports 1.96. Two of
them deserve a second look.

$y' = \cos t$ has $y'' = -\sin t$, which vanishes at $t = 0$ and nowhere else on the interval.
That is enough, because `local_against_global` measures the local error of the **first step
only**, starting from the exact solution at $t_0$. So the reported local order is a statement
about one point, and here that point happens to be a zero of the error constant. The global order
is unaffected at 0.992, which is the honest number for the method.

$y' = y - t$ from $y(0) = 1$ has the exact solution $y = t + 1$, along which $y'' \equiv 0$ for
**all** $t$. Euler is exact on it, the error column reads $0$, and both fitted orders come back as
`exact` rather than as a number, because there is no error to fit an order to.

That is the real lesson here. A degenerate case can inflate a measured order, or destroy the
measurement outright, and **only the second failure announces itself.** The first returns a
plausible number that happens to describe one point rather than the method.

### 4.3 Blow up time against $p$

```python
def blow_up(p, y0=1.0):
    def f(t, y):
        return ivp.as_state(y) ** p
    return f, y0 ** (1.0 - p) / (p - 1.0)

print(f"{'p':>5}{'exact blow up':>16}{'steps':>9}{'first crossing':>17}{'overshoot':>13}")
for p in (1.5, 2.0, 3.0, 5.0):
    f, exact_time = blow_up(p)
    out = ivp.blows_up_at(f, 0.0, 1.0, 3.0 * exact_time, threshold=1e6)
    for n, c in zip(out["steps"], out["first_exceeds_threshold"]):
        if n in (50, 800):
            print(f"{p:>5.1f}{exact_time:>16.6f}{n:>9}{c:>17.6f}"
                  f"{c - exact_time:>13.6f}")
```

The exact blow up time of $y' = y^p$ with $y(0) = y_0$ is $y_0^{1-p}/(p-1)$, found by
separating variables. Every measured crossing is **later** than the truth and refining moves it
earlier:

| $p$ | exact | 50 steps | 800 steps | overshoot ratio |
|---|---|---|---|---|
| 1.5 | 2.000000 | 2.880 | 2.070 | 12.6 |
| 2.0 | 1.000000 | 1.380 | 1.035 | 10.9 |
| 3.0 | 0.500000 | 0.660 | 0.514 | 11.6 |
| 5.0 | 0.250000 | 0.330 | 0.257 | 11.6 |

The step count rose by 16 and the overshoot fell by about 11.5, so the fitted order is
$\log 11.5/\log 16 = 0.88$: **first order, as Euler's global error is.** The blow up time is not a
special quantity here; it is being computed by a first order method and it inherits that.

The overshoot also shrinks in absolute terms with $p$, because a sharper blow up gives the solver
less room to underestimate the growth before the threshold is crossed. In relative terms it does
not: 44 per cent of the exact time at $p = 1.5$ and 32 per cent at $p = 5$ for 50 steps.

### 5.1 Peano without Lipschitz

**Peano's theorem** says: if $f$ is continuous on a neighbourhood of $(t_0, y_0)$, the initial value
problem has at least one solution on some interval around $t_0$.

It cannot give uniqueness, and $y' = y^{2/3}$, $y(0) = 0$ is the construction that proves it:
$f(y) = y^{2/3}$ is continuous everywhere, so Peano applies, and exercise 1.2 exhibits uncountably
many solutions. **So continuity is enough for existence and provably not enough for uniqueness.**

Why the constructive proof needs the stronger hypothesis is the interesting half. Picard's proof
builds the solution as the limit of the iteration

$$
y_{k+1}(t) = y_0 + \int_{t_0}^{t}f(s, y_k(s))\,ds
$$

and shows the map is a **contraction** on a suitable space of functions. The contraction constant
is $L(t - t_0)$, so it needs $L$ to exist; without it there is no reason for successive iterates to
approach each other, and indeed exercise 1.2's solutions are all fixed points of the same map.

Peano's proof is different in kind. It builds approximate solutions (Euler polygons, or mollified
versions of $f$), extracts a uniformly convergent subsequence by the Arzela-Ascoli theorem, and
shows the limit solves the equation. **A subsequence limit is not unique**, and different
subsequences can converge to different solutions, which is exactly the ambiguity showing up in the
proof technique.

Notice also what this says about numerical methods. Euler polygons are one of the standard
constructions in Peano's proof, so Euler's method run on such a problem does converge to *a*
solution. Which one depends on the arithmetic, as the lesson's measurement shows.

### 5.2 Deriving the $\varepsilon^{2/3}$ crossing

Model the computed step as

$$
\hat y_{n+1} = \hat y_n + h f(t_n, \hat y_n) + \delta_n, \qquad
\lvert \delta_n \rvert \lesssim \varepsilon \lvert y \rvert,
$$

where $\delta_n$ is the rounding of the addition and the multiplication. Two extreme assumptions
about the $\delta_n$ give two different answers.

**If they all point the same way** (the classical assumption), the accumulated rounding after
$N = T/h$ steps is $N\varepsilon = T\varepsilon/h$. Setting that against the truncation error
$Ch$ and minimising $Ch + T\varepsilon/h$ gives

$$
h_{\text{opt}} = \sqrt{T\varepsilon / C} = O(\sqrt\varepsilon).
$$

**If they are independent with mean zero**, which is what round to nearest gives for a smooth
computation, the accumulated error is a random walk: its standard deviation is
$\sqrt N \varepsilon = \varepsilon\sqrt{T/h}$. Minimising $Ch + \varepsilon\sqrt{T/h}$ instead,

$$
\frac{d}{dh}\left[Ch + \varepsilon\sqrt T h^{-1/2}\right] = C - \tfrac12\varepsilon\sqrt T h^{-3/2}
= 0 \quad\Longrightarrow\quad h_{\text{opt}} = \left(\frac{\varepsilon\sqrt T}{2C}\right)^{2/3},
$$

which is $O(\varepsilon^{2/3})$.

```python
eps = float(np.finfo(float).eps)
print(f"eps^(1/2) = {eps ** 0.5:.3e}")
print(f"eps^(2/3) = {eps ** (2.0 / 3.0):.3e}")
print(f"ratio: the random walk crossing is {eps ** 0.5 / eps ** (2.0 / 3.0):.0f} "
      f"times smaller")
print(f"steps needed to reach it over t = 1: {int(1.0 / eps ** (2.0 / 3.0)):,}")
```

**What would have to fail for the classical answer to be right:** the errors would have to be
correlated in sign. That happens when the rounding is biased, which is exactly what a
round-towards-zero or round-down mode gives, and it happened routinely on machines that truncated
rather than rounded. On IEEE round to nearest the bias is negligible and the random walk model is
the right one.

The practical conclusion is unchanged and is worth stating plainly. In double precision
$\varepsilon^{2/3} = 3.7\times10^{-11}$, which needs $2.7\times10^{10}$ steps for a run of unit
length, so **no realistic double precision Euler computation is limited by roundoff.** It is
limited by its order. Exercise 3.5 reaches the crossing only because single precision moves it up
to $2.4\times10^{-5}$.

### 5.3 An $f$ that is Lipschitz nowhere

Take the **Weierstrass function**

$$
W(y) = \sum_{k=0}^{\infty} a^k \cos(b^k \pi y), \qquad 0 < a < 1,\ b \text{ odd},\ ab > 1 + \tfrac32\pi,
$$

which is continuous everywhere and differentiable nowhere. Stronger than that, it is **Holder
continuous of exponent $\log a/\log b < 1$ and not Lipschitz at any point**: the difference
quotients are unbounded near every $y$.

Set $f(t, y) = W(y)$. Peano's theorem gives existence, because $W$ is continuous and bounded.
Picard-Lindelof gives nothing anywhere, and uniqueness is genuinely unknown for such an $f$ in
general.

```python
def weierstrass(y, a=0.5, b=7.0, terms=40):
    v = np.atleast_1d(np.asarray(y, dtype=float))
    return sum(a ** k * np.cos(b ** k * np.pi * v) for k in range(terms))

print(f"{'radius':>12}{'largest difference quotient':>30}")
rng = np.random.default_rng(42)
for r in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
    u = 0.3 + rng.uniform(-r, r, 4000)
    v = 0.3 + rng.uniform(-r, r, 4000)
    keep = np.abs(u - v) > 1e-300
    ratios = np.abs(weierstrass(u[keep]) - weierstrass(v[keep])) / np.abs(u[keep] - v[keep])
    print(f"{r:>12.0e}{float(np.max(ratios)):>30.2f}")
```

The estimate keeps rising as the radius shrinks, at every point, which is the numerical signature
of the analytic fact.

**What a solver does with it:** exactly what it does with any continuous $f$. It marches, it
returns numbers, and the numbers converge as $h \to 0$ to *some* solution, by the Euler polygon
argument of exercise 5.1. Nothing in the run reports difficulty, because a solver only ever
evaluates $f$ at points and $W$ is perfectly well behaved pointwise.

That is the honest and slightly uncomfortable answer, and it is the same one as the $y^{2/3}$ case:
the failure is invisible from inside the computation, and the only way to know is to have looked at
the equation.

---

## Lesson 68, Taylor and Picard Methods

### 1.1 Four Picard iterates for $y' = 2ty$, $y(0) = 1$

The integral equation is $y(t) = 1 + \int_0^t 2s\,y(s)\,ds$, and $y_0(t) \equiv 1$.

$$
y_1 = 1 + \int_0^t 2s\,ds = 1 + t^2,
$$
$$
y_2 = 1 + \int_0^t 2s(1 + s^2)\,ds = 1 + t^2 + \frac{t^4}{2},
$$
$$
y_3 = 1 + \int_0^t 2s\left(1 + s^2 + \frac{s^4}{2}\right)ds
    = 1 + t^2 + \frac{t^4}{2} + \frac{t^6}{6},
$$
$$
y_4 = 1 + t^2 + \frac{t^4}{2} + \frac{t^6}{6} + \frac{t^8}{24}.
$$

The pattern is $\sum_{k=0}^{n} t^{2k}/k!$, which is the partial sum of $e^{t^2}$. And indeed the
exact solution is $y = e^{t^2}$, since $y'/y = 2t$ integrates to $\ln y = t^2$.

**Iteration $n$ is the Taylor polynomial of degree $2n$**, not degree $n$: each iteration adds one
term of the series, and here each term carries two powers of $t$. So the general statement is that
iteration $n$ reproduces the first $n+1$ **terms**, and how many powers of $t$ that is depends on
the equation.

### 1.2 Why the iteration returns zero and stays there

For $y' = y^{2/3}$, $y(0) = 0$ the map is

$$
y_{k+1}(t) = 0 + \int_0^t \left(y_k(s)\right)^{2/3}ds .
$$

Start at $y_0 \equiv 0$. Then $y_1(t) = \int_0^t 0\,ds = 0$, and by induction every iterate is the
zero function. **Zero is a fixed point of the map**, exactly, and the iteration is already at it.

It converges, immediately, to a genuine solution, and reports nothing unusual. The other solutions
of exercise 67.1.2 are also fixed points of the same map, and nothing in the iteration prefers one
over another. What the Lipschitz condition would have bought is uniqueness **of the fixed point**,
which is what the contraction mapping theorem gives and what is missing here.

### 1.3 The order of a Taylor method, with no convergence theorem

The method of order $k$ is

$$
y_{n+1} = y_n + h y'_n + \frac{h^2}{2!}y''_n + \dots + \frac{h^k}{k!}y^{(k)}_n,
$$

where $y^{(j)}_n$ means the $j$th total derivative of the solution evaluated at $(t_n, y_n)$.

Compare with the exact Taylor expansion of the solution through $(t_n, y(t_n))$:

$$
y(t_{n+1}) = y(t_n) + h y' + \dots + \frac{h^k}{k!}y^{(k)} + \frac{h^{k+1}}{(k+1)!}y^{(k+1)}(\xi).
$$

The method is the expansion with the remainder deleted, so **the local truncation error is exactly
that remainder**, $O(h^{k+1})$, and the global order is $k$ by exercise 67.1.4's counting argument.

No convergence theorem is needed for the local statement because there is nothing to prove: the
method was **defined** as the truncated series, so its agreement with the series is an identity
rather than an estimate. The step from local to global still uses the Lipschitz condition, exactly
as for Euler.

### 1.4 The derivatives of $y' = t + y^2$

Differentiate along the solution, using $y' = t + y^2$ each time it appears:

$$
y'' = \frac{d}{dt}\left(t + y^2\right) = 1 + 2yy' = 1 + 2y(t + y^2) = 1 + 2ty + 2y^3,
$$

$$
y''' = \frac{d}{dt}\left(1 + 2ty + 2y^3\right)
     = 2y + 2ty' + 6y^2 y'
     = 2y + (2t + 6y^2)(t + y^2)
     = 2y + 2t^2 + 8ty^2 + 6y^4 .
$$

Two things to notice. The expressions **grow**, and each one needs the previous one
differentiated in both variables, so the work per order compounds.

```python
import numpy as np
import sympy as sp

t, y = sp.symbols("t y")
f = t + y ** 2
expr = f
for k in range(2, 7):
    expr = sp.simplify(sp.diff(expr, t) + sp.diff(expr, y) * f)
    print(f"y^({k}) has {sp.count_ops(expr)} operations, degree {sp.Poly(expr, t, y).total_degree()}")
```

The measured counts are 5, 9, 15, 21, 27 operations at orders 2 to 6, and the degree rises by one
each time. **That growth is linear, not exponential**, and the reason is that $t + y^2$ is a
polynomial: every derivative is another polynomial and `simplify` collects it back into a compact
form.

The general case is not like that. Exercise 3.3 does the same experiment on $\sin(ty)$, where
nothing collects, and the operation count goes 2, 7, 32, 65, 124. **Polynomial right hand sides are
the easy case for a Taylor method** and are exactly the case where automatic differentiation
(exercise 3.4) is easiest too.

### 1.5 Why counting derivative evaluations as equal understates the cost

A comparison that counts one evaluation of $y^{(4)}$ as one evaluation of $f$ is measuring the
wrong thing, in three ways.

**Arithmetic cost.** From exercise 1.4, $y^{(k)}$ has many more operations than $f$ does, and the
gap grows with $k$. Section 5's term counts, 1, 1, 2, 4, 9, 20, 48, are the number of distinct
elementary differentials, and each one is itself a product of partial derivatives.

**Derivation cost.** The formulas have to exist before they can be evaluated, and for a real
problem that is a page of algebra per order, done by hand or by a symbolic system, and it has to be
redone whenever the equation changes. A Runge-Kutta method needs nothing but $f$.

**Correctness cost.** A wrong $y^{(4)}$ does not crash. It quietly drops the method to order 3, and
the only way to notice is a convergence study, which is why the lesson runs one.

The honest comparison is against Runge-Kutta, which reaches the same order from evaluations of $f$
alone, and lesson 69 makes it.

### 2.1 The factorial bound

Let $D_k(t) = \lVert y_{k+1}(t) - y_k(t)\rVert$ and suppose $f$ is Lipschitz in $y$ with constant
$L$. From the iteration,

$$
y_{k+1}(t) - y_k(t) = \int_{t_0}^{t}\left[f(s, y_k(s)) - f(s, y_{k-1}(s))\right]ds,
$$

so

$$
D_k(t) \le \int_{t_0}^{t} L\, D_{k-1}(s)\,ds .
$$

Now induct. Suppose $D_{k-1}(s) \le D_0 \frac{L^{k-1}(s - t_0)^{k-1}}{(k-1)!}$, where
$D_0 = \sup\lVert y_1 - y_0\rVert$. Then

$$
D_k(t) \le L \int_{t_0}^{t} D_0\frac{L^{k-1}(s-t_0)^{k-1}}{(k-1)!}\,ds
= D_0 \frac{L^{k}(t - t_0)^{k}}{k!},
$$

which closes the induction, with the base case $D_0 \le D_0$ immediate.

**The factorial is what makes this work.** The geometric factor $L^k(t-t_0)^k$ alone would need
$L(t-t_0) < 1$ to converge; the $k!$ beats it for every $L$ and every interval length, so the
series $\sum D_k$ converges absolutely and uniformly and the iterates converge to a limit. The
Lipschitz constant enters only in the rate.

For $y' = y$ with $L = 1$ the bound can be checked against the change **exactly**, because the
iterates are known: $y_k$ is the Taylor polynomial of degree $k$, so
$y_{k} - y_{k-1} = t^{k}/k!$ and the largest value on $[0, T]$ is $T^{k}/k!$.

```python
import math
from nalib import taylorode as to

def exponential(t, y):
    return np.asarray(y, dtype=float)

for t_end in (0.5, 1.0, 2.0):
    out = to.picard(exponential, 0.0, 1.0, t_end, iterations=8)
    changes = np.asarray(out["change_per_iteration"], dtype=float)
    print(f"L(T - t0) = {t_end:.1f}")
    print(f"{'k':>4}{'measured change':>18}{'T^k / k!':>14}{'ratio':>9}{'rising?':>10}")
    previous = None
    for k, d in enumerate(changes, start=1):
        bound = t_end ** k / math.factorial(k)
        rising = "" if previous is None else str(d > previous)
        print(f"{k:>4}{d:>18.4e}{bound:>14.4e}{d / bound:>9.4f}{rising:>10}")
        previous = d
    print()
```

**The ratio is exactly 1.0000 in every row of every block.** The bound of exercise 2.1 is not
merely valid here, it is attained with equality, and that is not luck: $y' = \lambda y$ is the
extremal case for the Lipschitz estimate, since the inequality
$\lvert f(u) - f(v)\rvert \le L\lvert u - v\rvert$ is an equality for a linear $f$.

The third block is the one to look at. At $LT = 2$ the changes go
$2.0, 2.0, 1.33, 0.67, 0.27, \dots$: **the first two are equal and nothing improves until the
third.** The factor $T^k/k!$ rises while $k < T$, is flat at $k = T$, and falls afterwards, so the
turning point sits at $k \approx LT$ and an iteration over a long interval spends its first $LT$
steps getting nowhere.

That is the practical content of the factorial. It guarantees eventual convergence for any interval
length, and "eventual" means after about $LT$ iterations of getting nowhere.

### 2.2 Picard on a linear problem

For $y' = a(t)y + b(t)$ with $y(t_0) = y_0$ the iteration is

$$
y_{k+1}(t) = y_0 + \int_{t_0}^{t}\left[a(s)y_k(s) + b(s)\right]ds .
$$

Write $A(t) = \int_{t_0}^t a$, and take the simplest case $a$ constant, $b = 0$, $y_0 = 1$. Then
$y_1 = 1 + a(t-t_0)$, $y_2 = 1 + a(t-t_0) + \tfrac{a^2(t-t_0)^2}{2}$, and by induction

$$
y_k(t) = \sum_{j=0}^{k}\frac{a^j(t-t_0)^j}{j!},
$$

which is the partial sum of $e^{a(t-t_0)}$, the exact solution.

For general $a(t)$ the same computation gives the **Peano-Baker series**

$$
y_k(t) = y_0\left[1 + \int_{t_0}^{t}a + \int_{t_0}^{t}\!\!\int_{t_0}^{s_1}a(s_1)a(s_2) + \dots\right],
$$

whose limit is $y_0 e^{A(t)}$ in the scalar case. In the matrix case the nested integrals do not
collapse to an exponential, because $A(s_1)$ and $A(s_2)$ need not commute, and the series is the
right answer rather than a step towards one. That is where the **time ordered exponential** of
quantum mechanics comes from, and it is Picard's iteration written out.

### 2.3 Taylor 2 against Heun

The Taylor method of order 2 is

$$
y_{n+1} = y_n + hf + \frac{h^2}{2}\left(f_t + f_y f\right),
$$

two evaluations: one of $f$, one of the combination $f_t + f_y f$. Heun's method is

$$
y_{n+1} = y_n + \frac{h}{2}\left[f(t_n, y_n) + f(t_n + h, y_n + hf(t_n, y_n))\right],
$$

two evaluations, both of $f$.

Expand Heun's second stage:

$$
f(t_n + h, y_n + hf) = f + h(f_t + f_y f) + O(h^2),
$$

so

$$
y_{n+1} = y_n + \frac{h}{2}\left[2f + h(f_t + f_y f)\right] + O(h^3)
        = y_n + hf + \frac{h^2}{2}(f_t + f_y f) + O(h^3).
$$

**The two methods agree through $h^2$ and differ at $h^3$.** They are both second order and they
are not the same method: their error constants differ, and Heun's third order terms are whatever
the expansion of $f$ at the shifted point happens to produce.

The trade is exactly the one this lesson is about. Taylor 2 needs $f_t$ and $f_y$ symbolically;
Heun needs a second evaluation of $f$ at a point it chooses. **Same order, same cost in
evaluations, and only one of them needs you to differentiate anything.**

### 2.4 The rooted trees at orders 1 to 4

Write $f$ and its partials at a point, and expand $y^{(k)}$ by the chain rule.

Order 1, one term:
$$y' = f.$$

Order 2, one new term:
$$y'' = f_t + f_y f.$$

Grouped as elementary differentials with the $t$ dependence absorbed (the standard autonomous
form, $y' = f(y)$), that is $f'f$: **one tree**, a root with one child.

Order 3, two new terms:
$$y''' = f''(f, f) + f'f'f.$$
Two trees on three nodes: a root with two children, and a chain of three.

Order 4, four new terms:
$$y'''' = f'''(f,f,f) + 3f''(f'f, f) + f'f''(f,f) + f'f'f'f.$$
Four trees on four nodes: the star, the "cherry on a stalk", the "stalk on a cherry", and the
chain.

So the counts are 1, 1, 2, 4, matching the number of rooted trees on 1, 2, 3, 4 nodes. The
bijection is: each node is a differentiation, each edge records which factor was differentiated,
and the tree shape records the nesting.

```python
out = to.taylor_term_count([1, 2, 3, 4, 5, 6, 7, 8])
print(f"{'order':>7}{'new terms':>12}{'cumulative':>13}")
for k, new, total in zip(out["order"], out["new_terms"], out["cumulative_terms"]):
    print(f"{k:>7}{new:>12}{total:>13}")
print("\nthese are the rooted tree counts, OEIS A000081")
```

### 2.5 The Taylor method on the test equation

For $y' = \lambda y$ every derivative is available in closed form: $y^{(j)} = \lambda^j y$. So

$$
y_{n+1} = y_n + \sum_{j=1}^{k}\frac{h^j \lambda^j}{j!}y_n
        = \left(\sum_{j=0}^{k}\frac{(\lambda h)^j}{j!}\right)y_n .
$$

The bracket is the **degree $k$ Taylor polynomial of $e^{\lambda h}$**, which is exactly what a
method of order $k$ should reproduce, since the true growth factor over one step is $e^{\lambda h}$.

```python
from nalib import stability as st

print(f"{'z':>8}{'exp(z)':>14}" + "".join(f"{'Taylor ' + str(k):>14}" for k in (1, 2, 3, 4)))
for z in (-0.5, -1.0, -2.0, -2.5, -3.0):
    row = [sum(z ** j / math.factorial(j) for j in range(k + 1)) for k in (1, 2, 3, 4)]
    print(f"{z:>8.1f}{np.exp(z):>14.8f}" + "".join(f"{v:>14.8f}" for v in row))
print()
print("and the same polynomials are the explicit Runge-Kutta growth functions:")
for name, k in (("euler", 1), ("heun", 2), ("kutta third", 3), ("rk4", 4)):
    R = st.growth_function(name)
    z = -1.7
    taylor = sum(z ** j / math.factorial(j) for j in range(k + 1))
    print(f"  {name:>14}: R({z}) = {complex(R(z)).real:.12f}, "
          f"Taylor {k} = {taylor:.12f}")
```

That identity is the bridge to lesson 72. **An explicit method of order $k$ with exactly $k$ stages
has $R(z) = \sum_{j\le k} z^j/j!$**, whatever the tableau, because the order conditions force it
and the polynomial has nowhere else to go. So Euler, Heun, Kutta's third order rule and RK4 all
have the growth function of the corresponding Taylor method, and their stability regions are the
same as its.

### 3.1 Picard with a spline representation, on three equations

```python
cases = [
    ("y' = y", lambda t, y: np.asarray(y, dtype=float), 1.0,
     lambda t: np.exp(np.asarray(t, dtype=float))),
    ("y' = 2ty", lambda t, y: 2.0 * t * np.asarray(y, dtype=float), 1.0,
     lambda t: np.exp(np.asarray(t, dtype=float) ** 2)),
    ("y' = -y + t", lambda t, y: -np.asarray(y, dtype=float) + t, 1.0,
     lambda t: 2.0 * np.exp(-np.asarray(t, dtype=float))
     + np.asarray(t, dtype=float) - 1.0),
]
print(f"{'equation':>14}{'iterations':>12}{'final gap to exact':>22}{'converging':>12}")
for label, f, y0, exact in cases:
    out = to.picard(f, 0.0, y0, 0.8, iterations=8)
    gap = float(np.max(np.abs(out["final"][:, 0] - exact(out["t"]))))
    print(f"{label:>14}{8:>12}{gap:>22.4e}{str(out['converging']):>12}")
```

All three converge. The third is worth noting: its solution is not a power series with a nice
closed form for the partial sums, and the iteration converges to it just the same, because the
factorial bound of exercise 2.1 does not care what the solution looks like.

### 3.2 A Taylor method of order 4 for a system

Take the linear system $u' = v$, $v' = -u$, whose solution from $(1, 0)$ is
$(\cos t, -\sin t)$. The total derivatives cycle with period 4:

$$
\begin{pmatrix}u\\v\end{pmatrix}' = \begin{pmatrix}v\\-u\end{pmatrix}, \quad
{}'' = \begin{pmatrix}-u\\-v\end{pmatrix}, \quad
{}''' = \begin{pmatrix}-v\\u\end{pmatrix}, \quad
{}'''' = \begin{pmatrix}u\\v\end{pmatrix}.
$$

```python
from nalib import ivp

def rotation(t, y):
    s = ivp.as_state(y)
    return np.asarray([s[1], -s[0]])

def rotation_derivatives(t, y, k):
    """The kth total derivative of the solution, which cycles with period 4."""
    s = ivp.as_state(y)
    table = {1: np.asarray([s[1], -s[0]]), 2: np.asarray([-s[0], -s[1]]),
             3: np.asarray([-s[1], s[0]]), 0: np.asarray([s[0], s[1]])}
    return table[int(k) % 4]

def rotation_exact(t):
    return np.asarray([np.cos(t), -np.sin(t)])

out = to.taylor_orders_are_exact(rotation, rotation_derivatives, rotation_exact,
                                 0.0, [1.0, 0.0], 3.0)
print(f"{'order':>7}{'fitted':>10}{'error at 320 steps':>22}")
for k, fit, err in zip(out["order"], out["fitted_order"], out["finest_error"]):
    print(f"{k:>7}{fit:>10.4f}{err:>22.4e}")
assert out["all_match"]
```

The derivative table is written from the structure rather than by differentiating expressions,
which is what makes a linear system the easy case. For a nonlinear system there is no such
shortcut and exercise 1.4's swell applies in every component.

### 3.3 Symbolic generation at order 5

```python
t, y = sp.symbols("t y")
f = sp.sin(t * y)
expr = f
print(f"{'order':>7}{'operations':>13}{'characters':>13}")
print(f"{1:>7}{sp.count_ops(expr):>13}{len(str(expr)):>13}")
for k in range(2, 6):
    expr = sp.diff(expr, t) + sp.diff(expr, y) * f
    simplified = sp.simplify(expr)
    print(f"{k:>7}{sp.count_ops(simplified):>13}{len(str(simplified)):>13}")
```

The operation counts are 2, 7, 32, 65, 124 and the printed lengths 8, 25, 99, 185, 379. **That
is expression swell**, the same phenomenon lesson 61 met when differentiating symbolically, and it
is the practical reason Taylor methods above order 4 are rare outside the few fields that need
them.

Compare with exercise 1.4, where the counts grew linearly. The difference is entirely the right
hand side: $t + y^2$ collapses back to a polynomial and $\sin(ty)$ does not, because differentiating
it produces both $\sin$ and $\cos$ terms with polynomial coefficients that multiply out.

Note also that `simplify` is itself expensive and does not stop the growth; it only postpones it,
and exercise 4.3 measures what it costs.

### 3.4 Taylor by automatic differentiation

The derivatives can be generated numerically rather than symbolically, by propagating a truncated
power series through the arithmetic. That is **Taylor mode automatic differentiation**, and it
costs $O(k^2)$ per step rather than the exponential blow up of the symbolic route.

```python
def taylor_coefficients(f_series, y0, order):
    """Solve y' = f(y) by propagating Taylor coefficients, for f a polynomial in y.

    f_series takes the coefficient list of y and returns that of f(y), truncated.
    """
    c = [float(y0)] + [0.0] * order
    for k in range(order):
        fk = f_series(c, k)                  # the kth Taylor coefficient of f(y(t))
        c[k + 1] = fk / (k + 1)
    return c

def square_series(c, k):
    """The kth coefficient of y^2, which is the Cauchy product of c with itself."""
    return sum(c[j] * c[k - j] for j in range(k + 1))

print(f"{'order':>7}{'coefficients of the series for 1/(1-t)':>44}")
for order in (3, 5, 8):
    c = taylor_coefficients(square_series, 1.0, order)
    print(f"{order:>7}   {np.round(c, 8)}")
print("\nthe exact solution of y' = y^2, y(0) = 1 is 1/(1-t), whose coefficients are all 1")
```

Every coefficient comes out as 1, which is the series for $1/(1-t)$. The recurrence is
$c_{k+1} = \frac{1}{k+1}\sum_j c_j c_{k-j}$, three lines, and it generalises to any $f$ built from
arithmetic and the standard functions. **This is how Taylor methods are actually implemented**, and
it is why they survive at all: the derivation cost of exercise 1.5 disappears.

### 3.5 The convergence rate against $L(t - t_0)$

```python
print(f"{'L(t-t0)':>10}{'iterations to 1e-10':>22}{'predicted from k! > 1/tol':>28}")
for L, t_end in ((1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (3.0, 1.0), (5.0, 1.0)):
    def f(t, y, L=L):
        return L * np.asarray(y, dtype=float)

    out = to.picard(f, 0.0, 1.0, t_end, iterations=30)
    changes = np.asarray(out["change_per_iteration"], dtype=float)
    reached = np.flatnonzero(changes < 1e-10)
    got = int(reached[0]) + 1 if reached.size else None
    x = L * t_end
    k = 1
    while x ** k / math.factorial(k) > 1e-10:
        k += 1
    print(f"{x:>10.1f}{str(got):>22}{k:>28}")
```

The two columns track each other closely: the number of iterations needed is set by where
$(L T)^k / k!$ drops below the tolerance, and nothing else about the problem enters. That is the
factorial bound of exercise 2.1 being tight in the only sense that matters, its rate.

Note the shape of the dependence. Doubling $LT$ does **not** double the iteration count; it adds a
few, because the factorial eventually beats any geometric factor. That is why Picard is a proof
technique rather than an algorithm: it converges for every problem, and it needs a global
representation of a function at every step, which is what makes it impractical.

### 4.1 The radius of convergence for $y' = y^2$

```python
print(f"{'t_end':>8}{'iterations':>12}{'final change':>16}{'converging':>12}{'exact y(t)':>14}")
for t_end in (0.3, 0.6, 0.8, 0.9, 0.95, 1.05):
    def square(t, y):
        return np.asarray(y, dtype=float) ** 2

    with np.errstate(over="ignore", invalid="ignore"):
        out = to.picard(square, 0.0, 1.0, t_end, iterations=12)
        changes = np.asarray(out["change_per_iteration"], dtype=float)
    exact = 1.0 / (1.0 - t_end) if t_end < 1.0 else float("inf")
    print(f"{t_end:>8.2f}{12:>12}{changes[-1]:>16.3e}"
          f"{str(out['converging']):>12}{exact:>14.4f}")
```

The exact solution $1/(1-t)$ has a pole at $t = 1$, which is lesson 67's blow up time, and the
Picard iterates are the partial sums of $\sum t^k$, whose radius of convergence is exactly 1.

**The two agree, and that is not a coincidence.** Picard's iterates converge on the interval where
the solution exists and is analytic, and the solution stops existing exactly where its power series
stops converging. The Picard-Lindelof theorem's "some interval" is, for an analytic right hand
side, the interval of convergence of that series.

The rate is worth reading down the column. At $T = 0.3$ the change after 12 iterations is
$1.7\times10^{-11}$; at $T = 0.8$ it is $8.7\times10^{-4}$; at $T = 0.95$ it is $0.59$, still
falling but nowhere near converged. **The convergence slows to a halt as $T$ approaches the pole**,
which is what a radius of convergence looks like from inside.

Past $t = 1$ the iterates grow with the iteration count instead of settling, reaching $10^{4}$ by
the twelfth.

### 4.2 How the interpolation limits the measured identity

`to.picard` represents its iterates by their values on a grid and integrates by quadrature, so the
identity of section 2 can only be measured to the accuracy of that representation.

```python
from nalib.splines import not_a_knot
from nalib.gaussquad import rule_on

def picard_with(kind, nodes=81, iterations=6, t_end=0.8):
    """Picard for y' = y, y(0) = 1, with a chosen representation of the iterates."""
    grid = np.linspace(0.0, t_end, nodes)
    values = np.ones_like(grid)
    for _ in range(iterations):
        new = np.empty_like(grid)
        new[0] = 1.0
        for i in range(1, grid.size):
            points, weights = rule_on(12, grid[0], grid[i])
            if kind == "linear":
                sampled = np.interp(points, grid, values)
            elif kind == "nearest":
                sampled = values[np.clip(np.searchsorted(grid, points), 0, grid.size - 1)]
            else:
                spline = not_a_knot(grid, values)
                sampled = spline(points)
            new[i] = 1.0 + float(np.sum(weights * sampled))
        values = new
    partial = sum(grid ** j / math.factorial(j) for j in range(iterations + 1))
    return float(np.max(np.abs(values - partial)))

print(f"{'representation':>18}{'gap to the Taylor partial sum':>34}")
for kind in ("nearest", "linear", "spline"):
    print(f"{kind:>18}{picard_with(kind):>34.3e}")
```

The gap is the representation's error, not the identity's. Nearest neighbour caps it around
$10^{-2}$, linear interpolation around $10^{-5}$, and the not-a-knot spline reaches $10^{-11}$.

**The identity is exact and the measurement is not**, so a careless version of this experiment
would report that Picard iterates are "close to" the partial sums, which is true and much weaker
than what is actually going on.

### 4.3 Where symbolic generation overtakes stepping

```python
import time

t, y = sp.symbols("t y")
f = sp.sin(t * y)

print(f"{'order':>7}{'generation seconds':>21}{'evaluation seconds per 1000':>30}")
expr = f
for k in range(2, 7):
    start = time.perf_counter()
    expr = sp.simplify(sp.diff(expr, t) + sp.diff(expr, y) * f)
    generate = time.perf_counter() - start
    fn = sp.lambdify((t, y), expr, "numpy")
    values = np.linspace(0.1, 1.0, 1000)
    start = time.perf_counter()
    for _ in range(1):
        fn(values, values)
    evaluate = time.perf_counter() - start
    print(f"{k:>7}{generate:>21.4f}{evaluate:>30.6f}")
```

Generation time grows steeply with the order while evaluation time barely moves, because the
generated expression is evaluated on arrays. **Generation dominates from the very first order
here**, and would only stop dominating if the same derivative were reused across millions of steps.

That is the correct conclusion and it comes with a caveat: it is a statement about this equation
and this symbolic system. What is robust is the shape, generation growing superlinearly in the
order and evaluation growing linearly, so a crossover exists and depends on the step count.

### 5.1 Cauchy-Kovalevskaya, and what analyticity buys

A Taylor method is the power series solution of the ordinary differential equation, truncated. The
partial differential equation analogue is the **Cauchy-Kovalevskaya theorem**: for a system

$$
\partial_t^m u = F\left(t, x, \partial_t^j \partial_x^\alpha u\right), \qquad j < m,
$$

with $F$ **analytic** and analytic initial data on a non characteristic surface, there is a unique
analytic solution near that surface.

Two things change from the ordinary case.

**Analyticity replaces continuity.** Picard-Lindelof needs $f$ continuous in $t$ and Lipschitz in
$y$, which is far less. Cauchy-Kovalevskaya needs $F$ analytic, and the reason is that the proof is
a majorant argument on power series: it builds the coefficients by matching, and then must show the
resulting series converges. For an ordinary differential equation the factorial in exercise 2.1's
bound does that for free. For a partial differential equation the coefficients involve $x$
derivatives of the data as well, and there is nothing to control them unless the data is analytic.

**The conclusion is only local and only analytic.** The theorem says nothing about whether the
solution persists, and Hadamard's example shows why the hypothesis cannot be relaxed: the Laplace
equation $u_{tt} + u_{xx} = 0$ with data $u(0,x) = 0$, $u_t(0,x) = n^{-1}\sin(nx)$ has solution
$n^{-2}\sin(nx)\sinh(nt)$, whose data goes to zero and whose solution does not. **The problem is
ill posed**, and Cauchy-Kovalevskaya still applies to it, because well posedness is not what it
claims.

That is the sharpest way to state the difference. For ordinary differential equations, existence
and uniqueness essentially give a usable problem. For partial differential equations they do not,
and Part 11's classification into elliptic, parabolic and hyperbolic is precisely the question of
which problems are well posed.

### 5.2 The bijection with rooted trees

Work in autonomous form $y' = f(y)$, which loses nothing since $t$ can be made a component.
Differentiating repeatedly by the chain rule,

$$
y'' = f'f, \qquad
y''' = f''(f,f) + f'f'f, \qquad
y'''' = f'''(f,f,f) + 3f''(f'f,f) + f'f''(f,f) + f'f'f'f .
$$

Each term is an **elementary differential**: a nesting of derivatives of $f$ applied to other
elementary differentials.

**The bijection.** Map each term to a rooted tree as follows. The outermost $f^{(m)}$ becomes the
root, with $m$ children, one per argument. Each argument is itself an elementary differential and
becomes the subtree rooted at that child. A bare $f$ is a leaf.

Under this map:

- $f$ is the single node.
- $f'f$ is a root with one child.
- $f''(f,f)$ is a root with two children; $f'f'f$ is a chain of three.
- The four order 4 terms are the four rooted trees on four nodes.

The map is injective because the tree records the nesting exactly, and surjective because every
tree can be read as a nesting. So **the number of elementary differentials at order $k$ is the
number of rooted trees on $k$ nodes**, which is OEIS A000081: 1, 1, 2, 4, 9, 20, 48, 115, 286.

```python
def rooted_trees(n):
    """Rooted tree counts by the standard Euler transform recurrence."""
    a = [0, 1]
    for k in range(2, n + 1):
        total = 0
        for j in range(1, k):
            inner = sum(d * a[d] for d in range(1, j + 1) if j % d == 0)
            total += inner * a[k - j]
        a.append(total // (k - 1))
    return a[1:]

counts = rooted_trees(10)
measured = [int(v) for v in to.taylor_term_count(list(range(1, 9)))["new_terms"]]
print("rooted trees on 1..10 nodes:", counts)
print("nalib's term counts:        ", measured)
assert measured == counts[:len(measured)]
```

The multiplicity 3 on $f''(f'f, f)$ is not part of the count; it is the number of ways the tree
embeds, and it appears as a coefficient in the Taylor expansion and as $\gamma(\tau)$ in Butcher's
order conditions. The **count** of distinct differentials is the tree count, and that is the
quantity that grows.

### 5.3 A step controller from the next Taylor term

A Taylor method of order $k$ computes $y^{(1)}, \dots, y^{(k)}$. Computing $y^{(k+1)}$ as well
gives the **leading term of the local error** directly:

$$
\tau_{n+1} \approx \frac{h^{k+1}}{(k+1)!}\,y^{(k+1)}(t_n, y_n),
$$

with no second solution and no comparison needed. Feed that into lesson 70's control law:

$$
h_{\text{new}} = h\left(\frac{\text{tol}}{\lvert \tau \rvert}\right)^{1/(k+1)} .
$$

```python
def taylor_with_control(t_end=2.0, tol=1e-8, order=4):
    """Taylor of the given order on y' = y - t^2 + 1, with the next term as the estimate."""
    def derivative(t, y, k):
        v = float(y)
        if k == 1:
            return v - t ** 2 + 1.0
        if k == 2:
            return v - t ** 2 + 1.0 - 2.0 * t
        return v - t ** 2 - 1.0 - 2.0 * t

    t, y, h = 0.0, 0.5, 0.1
    steps, rejected = 0, 0
    while t < t_end:
        h = min(h, t_end - t)
        estimate = abs(h ** (order + 1) / math.factorial(order + 1)
                       * derivative(t, y, order + 1))
        if estimate > tol and h > 1e-12:
            rejected += 1
            h *= max(0.2, 0.9 * (tol / max(estimate, 1e-300)) ** (1.0 / (order + 1)))
            continue
        y = y + sum(h ** j / math.factorial(j) * derivative(t, y, j)
                    for j in range(1, order + 1))
        t += h
        steps += 1
        h *= min(5.0, 0.9 * (tol / max(estimate, 1e-300)) ** (1.0 / (order + 1)))
    return t, y, steps, rejected

exact_end = (2.0 + 1.0) ** 2 - 0.5 * math.exp(2.0)
for tol in (1e-4, 1e-6, 1e-8, 1e-10):
    t, y, steps, rejected = taylor_with_control(tol=tol)
    print(f"tol {tol:.0e}: {steps:4d} steps, {rejected:2d} rejected, "
          f"error {abs(y - exact_end):.3e}")
```

The controller works, at **no extra evaluations of $f$**: the $(k+1)$th derivative is one more
term of a sequence already being generated, and in the automatic differentiation implementation of
exercise 3.4 it is one more turn of the recurrence.

The errors are $9.8\times10^{-4}$, $2.4\times10^{-5}$, $6.5\times10^{-7}$ and
$1.6\times10^{-8}$ against tolerances of $10^{-4}$ through $10^{-10}$, so the achieved error is
**10 to 160 times the tolerance and the overshoot grows as the tolerance tightens.** That is
exactly the local against global gap lesson 70 measures for an embedded pair, arrived at by a
completely different route, which is the point: the gap belongs to the idea of controlling a local
error, not to the mechanism used to estimate it.

**Why a Runge-Kutta method cannot do this.** Its stages are evaluations of $f$ at chosen points,
and nothing in them is the next term of a series. The only way to estimate the local error is to
form a **second answer** and subtract, which is what an embedded pair does. So the free error
estimate is one of the two genuine advantages a Taylor method has, along with arbitrary order at
no stage barrier, and both are why they persist in high precision celestial mechanics where
positions are wanted to thirty digits over millions of years.

---

## Lesson 69, Runge-Kutta Methods

### 1.1 Two tableaux, and what makes them explicit

Heun's method:

$$
\begin{array}{c|cc}
0 & 0 & 0\\
1 & 1 & 0\\ \hline
  & \tfrac12 & \tfrac12
\end{array}
\qquad
\text{RK4:}\qquad
\begin{array}{c|cccc}
0 & 0 & 0 & 0 & 0\\
\tfrac12 & \tfrac12 & 0 & 0 & 0\\
\tfrac12 & 0 & \tfrac12 & 0 & 0\\
1 & 0 & 0 & 1 & 0\\ \hline
  & \tfrac16 & \tfrac13 & \tfrac13 & \tfrac16
\end{array}
$$

**The zeros on and above the diagonal of $A$ are what make each explicit.** Row $i$ of $A$ gives
the combination of stages that stage $i$ is evaluated at, so a nonzero entry $a_{ij}$ with
$j \ge i$ would make stage $i$ depend on itself or on a stage not yet computed.

```python
import numpy as np
from nalib import rungekutta as rk

for name in ("heun", "rk4"):
    A, b, c, order = rk.tableau_exact(name)
    print(f"{name}, order {order}:")
    for i in range(len(b)):
        print(f"  {str(c[i]):>4} | " + "".join(f"{str(v):>7}" for v in A[i]))
    print(f"  {'':>4} | " + "".join(f"{str(v):>7}" for v in b))
    print(f"  strictly lower triangular: {rk.is_explicit(rk.tableau(name)[0])}\n")
```

### 1.2 Why $A$ must be strictly lower triangular

Stage $i$ is

$$
k_i = f\left(t_n + c_i h,\; y_n + h\sum_j a_{ij}k_j\right).
$$

If $a_{ij} = 0$ for all $j \ge i$ then the sum runs only over $j < i$, so $k_i$ is computable once
$k_1, \dots, k_{i-1}$ are known, and the stages can be evaluated in order. That is an **explicit**
method: $s$ evaluations, no equation to solve.

A nonzero **diagonal** entry $a_{ii}$ makes $k_i$ appear on both sides, so stage $i$ is an implicit
equation in $k_i$. If that is the only kind of coupling, the method is **diagonally implicit**
(DIRK) and the stages can still be solved one at a time, each a scalar or small nonlinear solve.

A nonzero entry **above** the diagonal couples all the stages at once, and the whole $s\cdot m$
dimensional system has to be solved together. That is a fully implicit method, expensive per step
and the only route to A-stability at high order (lesson 72).

```python
A, b, c, order = rk.tableau("rk4")
bad = A.copy()
bad[1, 2] = 0.25                                  # one entry above the diagonal
print(f"original explicit: {rk.is_explicit(A)}, modified: {rk.is_explicit(bad)}")
try:
    rk.step_from(bad, b, c)
except ValueError as e:
    print(f"step_from refuses it: {e}")
```

### 1.3 What a method reduces to when $f$ depends only on $t$

If $f = f(t)$ the stages are $k_i = f(t_n + c_i h)$ with no $y$ coupling at all, so

$$
y_{n+1} = y_n + h\sum_i b_i f(t_n + c_i h),
$$

which is the quadrature rule with **nodes $c_i$ and weights $b_i$** applied to
$\int_{t_n}^{t_{n+1}}f$.

RK4 has $c = (0, \tfrac12, \tfrac12, 1)$ and $b = (\tfrac16, \tfrac13, \tfrac13, \tfrac16)$. The
two repeated middle nodes collapse into one of weight $\tfrac23$, giving weights
$(\tfrac16, \tfrac23, \tfrac16)$ at $(0, \tfrac12, 1)$: **Simpson's rule.**

```python
def only_t(t, y):
    return np.full_like(np.atleast_1d(np.asarray(y, dtype=float)), np.cos(t))

from nalib import ivp

for name in ("euler", "midpoint", "heun", "rk4"):
    out = ivp.integrate(only_t, 0.0, 0.0, 1.0, 1, rk.named_step(name))
    print(f"{name:>10}: one step gives {float(out['y'][-1, 0]):.10f}")
print(f"{'exact':>10}: sin(1) =        {np.sin(1.0):.10f}")
print(f"{'Simpson':>10}: (1/6)(cos0 + 4cos(1/2) + cos1) = "
      f"{(np.cos(0.0) + 4 * np.cos(0.5) + np.cos(1.0)) / 6.0:.10f}")
```

Euler gives the left endpoint rule, midpoint gives the midpoint rule, Heun gives the trapezoid
rule, and RK4 gives Simpson's rule to the last digit.

### 1.4 The order conditions through order 3

With $c_i = \sum_j a_{ij}$ assumed:

$$
\text{order 1:}\quad \sum_i b_i = 1
$$
$$
\text{order 2:}\quad \sum_i b_i c_i = \tfrac12
$$
$$
\text{order 3:}\quad \sum_i b_i c_i^2 = \tfrac13, \qquad \sum_{i,j} b_i a_{ij} c_j = \tfrac16
$$

The counts are 1, 1, 2 new conditions at orders 1, 2, 3, matching lesson 68's rooted tree counts.
The first three are quadrature conditions on the nodes and weights alone; **the fourth is the first
one that involves $A$**, and it is the first place a method can be a genuine Runge-Kutta method
rather than a quadrature rule in disguise.

### 1.5 Why per step comparison misleads

RK4 does four evaluations of $f$ per step and Euler does one. So at the same number of **steps**,
RK4 has done four times the work, and any accuracy comparison at fixed step count is comparing two
different budgets.

```python
def f(t, y):
    return ivp.as_state(y) - t ** 2 + 1.0

def exact(t):
    return (t + 1.0) ** 2 - 0.5 * np.exp(t)

print(f"{'steps':>7}{'euler error':>15}{'euler evals':>13}"
      f"{'rk4 error':>13}{'rk4 evals':>11}")
for n in (10, 20, 40, 80):
    e = ivp.integrate(f, 0.0, 0.5, 2.0, n, rk.named_step("euler"))
    r = ivp.integrate(f, 0.0, 0.5, 2.0, n, rk.named_step("rk4"))
    print(f"{n:>7}{abs(float(e['y'][-1, 0]) - exact(2.0)):>15.3e}{n:>13}"
          f"{abs(float(r['y'][-1, 0]) - exact(2.0)):>13.3e}{4 * n:>11}")
print()
print("the same comparison at matched evaluations:")
print(f"{'evals':>7}{'euler steps':>13}{'euler error':>15}"
      f"{'rk4 steps':>11}{'rk4 error':>13}")
for evals in (40, 80, 160, 320):
    e = ivp.integrate(f, 0.0, 0.5, 2.0, evals, rk.named_step("euler"))
    r = ivp.integrate(f, 0.0, 0.5, 2.0, evals // 4, rk.named_step("rk4"))
    print(f"{evals:>7}{evals:>13}{abs(float(e['y'][-1, 0]) - exact(2.0)):>15.3e}"
          f"{evals // 4:>11}{abs(float(r['y'][-1, 0]) - exact(2.0)):>13.3e}")
```

RK4 wins either way here, so the point is not that the ranking changes. It is that the **margin**
does: quoting the per step numbers overstates RK4's advantage by a factor of $4^4 = 256$ in the
asymptotic regime, because four times the step means $4^4$ times the error for a fourth order
method.

Lesson 72 section 7 has a case where the ranking genuinely does reverse.

### 2.1 Deriving the two stage family, with the discarded terms

Write $f$ and its partials at $(t_n, y_n)$. The second stage is

$$
k_2 = f(t_n + \alpha h, y_n + \alpha h f)
    = f + \alpha h(f_t + f_y f) + \frac{\alpha^2h^2}{2}\left(f_{tt} + 2f_{ty}f + f_{yy}f^2\right)
      + O(h^3).
$$

So

$$
y_{n+1} = y_n + h(b_1 + b_2)f + \alpha b_2 h^2(f_t + f_y f)
        + \frac{\alpha^2 b_2 h^3}{2}\left(f_{tt} + 2f_{ty}f + f_{yy}f^2\right) + O(h^4).
$$

The exact expansion, from lesson 68 with $F = f_t + f_y f$, is

$$
y(t_{n+1}) = y_n + hf + \frac{h^2}{2}F
           + \frac{h^3}{6}\left(f_{tt} + 2f_{ty}f + f_{yy}f^2 + f_y F\right) + O(h^4).
$$

Matching $h$ and $h^2$ gives $b_1 + b_2 = 1$ and $\alpha b_2 = \tfrac12$. Matching $h^3$ would need

$$
\frac{\alpha^2 b_2}{2}\left(f_{tt} + 2f_{ty}f + f_{yy}f^2\right)
= \frac{1}{6}\left(f_{tt} + 2f_{ty}f + f_{yy}f^2\right) + \frac{1}{6}f_y F,
$$

which is **impossible for any $\alpha, b_2$**: the left side has no $f_y F$ term at all, and no
choice of two numbers can produce one. That is the proof that two stages cannot reach order 3, and
it shows where: the tree $f'f'f$ has no representative in a two stage expansion.

The **leading error term** is what is left over. Using $\alpha b_2 = \tfrac12$, so
$\alpha^2 b_2 = \alpha/2$:

$$
\tau = \frac{h^3}{6}\left[\left(1 - \tfrac{3\alpha}{2}\right)
\left(f_{tt} + 2f_{ty}f + f_{yy}f^2\right) + f_yF\right] + O(h^4).
$$

### 2.2 Ralston's choice

Take the leading error term above and measure it by the sum of the magnitudes of its two
coefficients, treating the two elementary differentials as independent:

$$
E(\alpha) = \left\lvert 1 - \tfrac{3\alpha}{2}\right\rvert + 1 .
$$

That is minimised by $\alpha = \tfrac23$, where the first coefficient vanishes entirely, leaving
only $f_yF$. **Ralston's method kills one of the two order 3 terms outright**, which is the
strongest thing a two stage method can do.

```python
print(f"{'alpha':>8}{'b1':>10}{'b2':>10}{'|1 - 3a/2|':>14}{'E(alpha)':>12}{'name':>12}")
for alpha, name in ((0.5, "midpoint"), (2.0 / 3.0, "ralston"), (1.0, "heun"),
                    (0.75, ""), (0.4, "")):
    b2 = 0.5 / alpha
    b1 = 1.0 - b2
    first = abs(1.0 - 1.5 * alpha)
    print(f"{alpha:>8.4f}{b1:>10.4f}{b2:>10.4f}{first:>14.4f}{first + 1.0:>12.4f}{name:>12}")
print()
print("ratio of Ralston's constant to Heun's:", f"{1.0 / (abs(1.0 - 1.5) + 1.0):.4f}")
```

Heun has $\lvert 1 - \tfrac32\rvert = \tfrac12$, so $E = 1.5$; Ralston has $E = 1$. **The ratio is
$2/3$**, a 33 per cent improvement in the constant and no change in the order.

Measure it on a problem rather than trusting the model:

```python
print(f"{'method':>10}{'error at 320 steps':>22}{'relative to ralston':>22}")
values = {}
for name in ("midpoint", "heun", "ralston"):
    out = rk.verified_order(f, exact, 0.0, 0.5, 2.0, name)
    values[name] = float(out["errors"][-1])
for name in ("midpoint", "heun", "ralston"):
    print(f"{name:>10}{values[name]:>22.4e}{values[name] / values['ralston']:>22.4f}")
```

### 2.3 The three stage third order family

A three stage explicit tableau has free parameters $c_2, c_3, a_{32}$ and $b_1, b_2, b_3$, with
$a_{21} = c_2$ and $a_{31} = c_3 - a_{32}$ forced by $c_i = \sum_j a_{ij}$. That is six unknowns
and four conditions (1 at order 1, 1 at order 2, 2 at order 3), so a **two parameter family**.

Solving the four conditions:

$$
b_1 + b_2 + b_3 = 1, \quad
b_2c_2 + b_3c_3 = \tfrac12, \quad
b_2c_2^2 + b_3c_3^2 = \tfrac13, \quad
b_3 a_{32} c_2 = \tfrac16 .
$$

The last determines $a_{32} = \dfrac{1}{6b_3c_2}$. The middle two are a 2 by 2 linear system in
$b_2, b_3$ once $c_2, c_3$ are chosen:

$$
b_2 = \frac{3c_3 - 2}{6c_2(c_3 - c_2)}, \qquad
b_3 = \frac{2 - 3c_2}{6c_3(c_3 - c_2)}, \qquad
b_1 = 1 - b_2 - b_3 .
$$

So $(c_2, c_3)$ parameterises the family, with $c_2 \ne c_3$, $c_2 \ne 0$, $c_3 \ne 0$.

```python
from fractions import Fraction

def three_stage(c2, c3):
    """The unique third order three stage tableau with these two nodes."""
    c2, c3 = Fraction(c2), Fraction(c3)
    b2 = (3 * c3 - 2) / (6 * c2 * (c3 - c2))
    b3 = (2 - 3 * c2) / (6 * c3 * (c3 - c2))
    b1 = 1 - b2 - b3
    a32 = Fraction(1, 6) / (b3 * c2)
    A = [[Fraction(0)] * 3, [c2, Fraction(0), Fraction(0)],
         [c3 - a32, a32, Fraction(0)]]
    return A, [b1, b2, b3], [Fraction(0), c2, c3]

print(f"{'(c2, c3)':>16}{'b':>34}{'a32':>10}{'order':>8}")
for c2, c3 in ((Fraction(1, 2), Fraction(1)), (Fraction(1, 3), Fraction(2, 3)),
               (Fraction(1, 2), Fraction(3, 4)), (Fraction(1), Fraction(1, 2))):
    A, b, c = three_stage(c2, c3)
    out = rk.order_conditions(A, b, c, up_to=4)
    label = f"({c2}, {c3})"
    print(f"{label:>16}{str([str(v) for v in b]):>34}{str(A[2][1]):>10}"
          f"{out['achieved_order']:>8}")
```

The first row is Kutta's third order rule and the second is Heun's, both in the library. Every
member of the family reaches order 3 exactly and none reaches 4, which is the two parameter family
behaving as the count predicts.

### 2.4 Why $c_i = \sum_j a_{ij}$

The condition says stage $i$ is evaluated at a $y$ argument consistent with its $t$ argument.

Apply the method to the trivial problem $y' = 1$, $y(t_0) = t_0$, whose solution is $y = t$. Then
$f \equiv 1$ and $k_i = 1$ for every $i$, so $y_{n+1} = y_n + h\sum b_i = y_n + h$, which is exact
whatever $A$ and $c$ are. That test says nothing.

Now apply it to $y' = 2t$, $y(0) = 0$, solution $y = t^2$. Here $f = 2t$, so
$k_i = 2(t_n + c_ih)$ and

$$
y_{n+1} = y_n + 2h\sum_i b_i (t_n + c_ih) = y_n + 2ht_n + 2h^2\sum_i b_ic_i,
$$

which matches $y(t_{n+1}) = (t_n+h)^2 = y_n + 2ht_n + h^2$ only if $\sum b_ic_i = \tfrac12$. That
is the order 2 condition, and it constrains $c$ but not $A$.

The link to $A$ comes from **invariance under autonomisation**. Any non autonomous problem
$y' = f(t,y)$ can be written autonomously by adding $\tau' = 1$, $\tau(t_0) = t_0$. Applying the
method to the enlarged system, the $\tau$ component of stage $i$ is
$\tau_n + h\sum_j a_{ij}$, whereas the method as written evaluates $f$ at $t_n + c_ih$. **The two
agree for every problem if and only if $c_i = \sum_j a_{ij}$.**

So the condition is not an order condition. It is the requirement that the method give the same
answer whether or not you autonomise, and a tableau violating it is really two different methods
depending on how the problem is written.

```python
A, b, c, order = rk.tableau("rk4")
print("row sums of A:", A.sum(axis=1), " c:", c)
print("they agree:", np.allclose(A.sum(axis=1), c))
```

### 2.5 The stability polynomial

Apply the method to $y' = \lambda y$ and write $z = \lambda h$. The stages satisfy

$$
k_i = \lambda\left(y_n + h\sum_j a_{ij}k_j\right)
\quad\Longrightarrow\quad
\mathbf{k} = \lambda y_n \mathbf{1} + z A\mathbf{k},
$$

so $\mathbf{k} = \lambda y_n (I - zA)^{-1}\mathbf{1}$ and

$$
y_{n+1} = y_n + h b^{\mathsf T}\mathbf{k} = \left(1 + z\,b^{\mathsf T}(I - zA)^{-1}\mathbf 1\right)y_n .
$$

For an **explicit** method $A$ is strictly lower triangular and hence nilpotent with $A^s = 0$, so

$$
(I - zA)^{-1} = \sum_{j=0}^{s-1}z^jA^j,
$$

a finite sum, and

$$
R(z) = 1 + \sum_{j=0}^{s-1}z^{j+1}b^{\mathsf T}A^j\mathbf 1
     = 1 + \sum_{m=1}^{s} z^m\, b^{\mathsf T}A^{m-1}\mathbf 1,
$$

a **polynomial of degree at most $s$**.

Now suppose the method has order $s$. The order conditions include $b^{\mathsf T}A^{m-1}\mathbf 1 =
1/m!$ for each $m \le s$ (these are the "tall tree" conditions, the chains). So

$$
R(z) = \sum_{m=0}^{s}\frac{z^m}{m!},
$$

**exactly, for every $s\le 4$ method of order $s$**, independent of which tableau it is. That is
lesson 68 exercise 2.5's identity, arrived at from the tableau side.

```python
import math
from nalib import stability as st

print(f"{'method':>16}{'stages':>8}{'order':>7}{'R(-1.3)':>16}{'Taylor sum':>16}{'gap':>12}")
for name in ("euler", "heun", "midpoint", "ralston", "kutta third", "heun third",
             "rk4", "three eighths"):
    A, b, c, order = rk.tableau(name)
    R = st.growth_function(name) if name in st.METHODS else None
    z = -1.3
    built = 1.0 + sum(z ** m * float(b @ np.linalg.matrix_power(A, m - 1) @ np.ones(len(b)))
                      for m in range(1, len(b) + 1))
    taylor = sum(z ** m / math.factorial(m) for m in range(order + 1))
    print(f"{name:>16}{len(b):>8}{order:>7}{built:>16.12f}{taylor:>16.12f}"
          f"{abs(built - taylor):>12.2e}")
```

Every method agrees with the Taylor polynomial of its own order to machine precision. **Ralston,
Heun and midpoint have identical stability regions** despite being different methods, which is why
lesson 72 lists one region per order rather than one per tableau.

### 3.1 A general explicit stepper

```python
def general_step(A, b, c):
    """Build a one step function from any explicit tableau, from scratch."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    if not np.all(np.triu(A) == 0.0):
        raise ValueError("this builder is for explicit tableaux only")
    stages = b.size

    def step(f, t, y, h, _counter=None):
        state = ivp.as_state(y)
        k = np.empty((stages, state.size))
        for i in range(stages):
            shift = state.copy()
            for j in range(i):
                if A[i, j] != 0.0:
                    shift = shift + h * A[i, j] * k[j]
            if _counter is not None:
                _counter[0] += 1
            k[i] = ivp.as_state(f(t + c[i] * h, shift))
        return state + h * (b @ k)

    return step

print(f"{'method':>16}{'mine':>22}{'nalib':>22}{'gap':>12}")
for name in ("euler", "heun", "kutta third", "rk4", "three eighths"):
    A, b, c, order = rk.tableau(name)
    mine = ivp.integrate(f, 0.0, 0.5, 2.0, 50, general_step(A, b, c))
    theirs = ivp.integrate(f, 0.0, 0.5, 2.0, 50, rk.named_step(name))
    a, t2 = float(mine["y"][-1, 0]), float(theirs["y"][-1, 0])
    print(f"{name:>16}{a:>22.14f}{t2:>22.14f}{abs(a - t2):>12.2e}")
```

Identical to the last bit, which is the check that the tableau convention is the same one.

### 3.2 Searching the third order family

Take the family of exercise 2.3 and minimise the order 4 residual, which is what is left over when
a third order method meets the order 4 conditions.

```python
from scipy.optimize import minimize

def order_four_residual(params):
    c2, c3 = params
    if abs(c2) < 1e-3 or abs(c3) < 1e-3 or abs(c3 - c2) < 1e-3:
        return 1e6
    b2 = (3 * c3 - 2) / (6 * c2 * (c3 - c2))
    b3 = (2 - 3 * c2) / (6 * c3 * (c3 - c2))
    b1 = 1 - b2 - b3
    a32 = (1.0 / 6.0) / (b3 * c2)
    A = np.asarray([[0.0, 0.0, 0.0], [c2, 0.0, 0.0], [c3 - a32, a32, 0.0]])
    b = np.asarray([b1, b2, b3])
    c = np.asarray([0.0, c2, c3])
    out = rk.order_conditions(A, b, c, up_to=4)
    fourth = [float(r) for o, r in zip(out["order"], out["residual"]) if o == 4]
    return float(np.sqrt(np.sum(np.square(fourth))))

best = minimize(order_four_residual, [0.4, 0.8], method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000})
print(f"best (c2, c3) = ({best.x[0]:.6f}, {best.x[1]:.6f})")
print(f"order 4 residual there: {best.fun:.6e}")
for label, params in (("Kutta third", [0.5, 1.0]), ("Heun third", [1 / 3, 2 / 3]),
                      ("optimised", list(best.x))):
    print(f"  {label:>12}: residual {order_four_residual(params):.6f}")
```

**The minimum is not zero**, which is the point: no three stage explicit method has order 4, so the
residual has a positive floor and the search finds where that floor is lowest. The optimised
tableau is third order with a smaller leading error constant than either named method, and it is
not order 4 and cannot be.

### 3.3 An implicit Runge-Kutta step, and the two stage Gauss method

The two stage Gauss-Legendre method has nodes at the Legendre points of $[0,1]$:

$$
c = \left(\tfrac12 - \tfrac{\sqrt3}{6},\ \tfrac12 + \tfrac{\sqrt3}{6}\right), \quad
A = \begin{pmatrix}\tfrac14 & \tfrac14 - \tfrac{\sqrt3}{6}\\
\tfrac14 + \tfrac{\sqrt3}{6} & \tfrac14\end{pmatrix}, \quad
b = \left(\tfrac12, \tfrac12\right).
$$

$A$ is full, so the two stages are coupled and must be solved together.

```python
from nalib.nlsystems import newton_system

root3 = np.sqrt(3.0)
GAUSS_A = np.asarray([[0.25, 0.25 - root3 / 6.0], [0.25 + root3 / 6.0, 0.25]])
GAUSS_B = np.asarray([0.5, 0.5])
GAUSS_C = np.asarray([0.5 - root3 / 6.0, 0.5 + root3 / 6.0])

def implicit_step(A, b, c):
    """A fully implicit Runge-Kutta step, solving all stages at once by Newton."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    stages = b.size

    def step(f, t, y, h, _counter=None):
        state = ivp.as_state(y)
        n = state.size

        def residual(flat):
            k = np.asarray(flat, dtype=float).reshape(stages, n)
            out = np.empty_like(k)
            for i in range(stages):
                out[i] = k[i] - ivp.as_state(f(t + c[i] * h, state + h * (A[i] @ k)))
            return out.ravel()

        guess = np.tile(ivp.as_state(f(t, state)), stages)
        answer = newton_system(residual, None, guess, tol=1e-13, max_iter=30)
        k = np.asarray(answer.root, dtype=float).reshape(stages, n)
        return state + h * (b @ k)

    return step

out = rk.order_conditions(GAUSS_A, GAUSS_B, GAUSS_C, up_to=5)
print(f"order conditions satisfied to order {out['achieved_order']} "
      f"(exact arithmetic: {out['exact']})")
run = ivp.error_against_step(f, exact, 0.0, 0.5, 2.0, [5, 10, 20, 40, 80],
                             implicit_step(GAUSS_A, GAUSS_B, GAUSS_C))
print(f"{'steps':>8}{'error':>14}{'ratio':>9}")
last = None
for n, e in zip(run["steps"], run["errors"]):
    print(f"{n:>8}{e:>14.4e}" + ("" if last is None else f"{last / e:>9.2f}"))
    last = e
print(f"fitted order {run['fitted_order']:.4f}")
```

**Two stages, order 4**: an implicit method escapes the $p \le s$ bound entirely, and the $s$ stage
Gauss method has order $2s$. That is the reward for solving the coupled system, and lesson 72 is
where it pays.

### 3.4 Exact order conditions up to order 6, on a published order 5 tableau

`nalib.rungekutta.order_conditions` already carries the 17 conditions through order 5 in exact
rational arithmetic. Checking a published tableau needs its coefficients as exact rationals rather
than as decimals, and that is the whole difficulty.

```python
from nalib import adaptivestep as ad

for name in ("dormand prince", "fehlberg", "bogacki shampine", "heun euler"):
    A, bh, bl, c, ph, pl = ad.pair(name)
    high = rk.order_conditions(A, bh, c, up_to=5)
    low = rk.order_conditions(A, bl, c, up_to=5)
    worst = max(abs(float(v)) for v in high["residual"])
    print(f"{name:>18}: high order claims {ph}, conditions give "
          f"{high['achieved_order']}; low claims {pl}, gives {low['achieved_order']}; "
          f"exact: {high['exact']}, worst residual {worst:.2e}")
```

The pairs are stored as floats, so `exact` comes back False and every verdict is "to within
the tolerance". **That distinction is real**: converting Dormand-Prince's $\tfrac{35}{384}$ through
binary floating point and then asking whether an order condition holds **exactly** gives False for
all 17, because it is not a binary fraction.

The worst residual column then has to be read with care, because a large value means two different
things. For Dormand-Prince and Fehlberg it is $2\times10^{-16}$, which is roundoff and says the
conditions hold. For Bogacki-Shampine it is $4.2\times10^{-2}$ and for Heun-Euler $3.0\times10^{-1}$,
which are the genuine misses at **order 4 and order 3** respectively, one past what each claims.
The function checks up to order 5 for every tableau, so a low order method necessarily fails some
of them.

The lesson is that a float check needs both a residual and an order to be interpretable, and that
is why `order_conditions` returns the achieved order rather than only the numbers.

### 3.5 Stability limits by bisection

```python
published = {"euler": 2.0, "midpoint": 2.0, "heun": 2.0, "ralston": 2.0,
             "kutta third": 2.5127, "heun third": 2.5127,
             "rk4": 2.7853, "three eighths": 2.7853}
print(f"{'method':>16}{'measured':>14}{'published':>12}{'gap':>12}")
for name in rk.TABLEAUX:
    got = rk.stability_limit(name)
    want = published[name]
    print(f"{name:>16}{got:>14.6f}{want:>12.4f}{abs(got - want):>12.2e}")
```

Every measured limit matches the published one to four decimals, and the methods pair up by order
rather than by name, which is exercise 2.5's identity again: same order and same stage count means
the same growth polynomial means the same limit.

### 4.1 Equal cost on a problem with a discontinuous fourth derivative

The obvious test problem is $y' = \lvert t-1\rvert^3$, whose solution is $C^3$ and not $C^4$. It is
also a trap, and it takes three goes to get out of it.

```python
def kinked(t, y):
    # y' = |t - 1|^3: the solution has a discontinuous fourth derivative at t = 1
    return np.full_like(ivp.as_state(y), abs(t - 1.0) ** 3)

def kinked_exact(t):
    # integral of |s-1|^3 from 0 to t, which is 0 at t = 0
    return np.asarray([(np.sign(t - 1.0) * abs(t - 1.0) ** 4 + 1.0) / 4.0])

out = rk.compare_at_equal_cost(kinked, kinked_exact, 0.0, 0.0, 2.0,
                               budgets=[48, 96, 192, 384])
print(f"{'evaluations':>13}" + "".join(f"{n:>14}" for n in out["names"]))
for j, e in enumerate(out["evaluations"]):
    print(f"{e:>13}" + "".join(f"{out['errors'][n][j]:>14.3e}" for n in out["names"]))
```

**Kutta's third order rule returns exactly zero and RK4 returns roundoff.** By exercise 1.3 a right
hand side depending only on $t$ turns every method into a quadrature rule; Kutta's third order rule
becomes Simpson's rule; $\lvert t-1\rvert^3$ is a **cubic on each side of the kink**; and with these
budgets $t = 1$ lands on a grid point, so every panel is a cubic and Simpson is exact on it.

**Second go.** Keep the kink off the grid by using an odd number of steps.

```python
print(f"{'steps':>7}{'on the kink':>14}"
      + "".join(f"{n:>14}" for n in ("euler", "heun", "kutta third", "rk4")))
for n in (25, 50, 51, 101, 201, 401):
    row = []
    for name in ("euler", "heun", "kutta third", "rk4"):
        run = ivp.integrate(kinked, 0.0, 0.0, 2.0, n, rk.named_step(name))
        row.append(abs(float(run["y"][-1, 0]) - float(kinked_exact(2.0)[0])))
    on_grid = abs((1.0 / (2.0 / n)) % 1.0) < 1e-12
    print(f"{n:>7}{str(on_grid):>14}" + "".join(f"{v:>14.3e}" for v in row))
print()
for name in ("euler", "heun", "kutta third", "rk4"):
    run = ivp.error_against_step(kinked, kinked_exact, 0.0, 0.0, 2.0,
                                 [25, 51, 101, 201, 401, 801], rk.named_step(name))
    print(f"{name:>14}: fitted order {run['fitted_order']:.4f}")
```

Still wrong, and now in two new ways. **Euler and Heun give identical errors, and so do Kutta's
third order rule and RK4.** And Euler's fitted order is 2, not 1.

All three follow from exercise 1.3 again. As quadrature rules, Euler is the left rectangle rule and
Heun is the trapezoid rule, and they differ by $\tfrac{h}{2}\left[g(b) - g(a)\right]$, which is zero
here because $g(0) = g(2) = 1$. The same cancellation removes the left rectangle rule's first order
error term, promoting it to second order. Kutta's third order rule and RK4 both reduce to Simpson's
rule and are therefore the same method on this problem.

**Third go.** The only way out is a right hand side that genuinely depends on $y$, so manufacture
one around the same non smooth solution:

$$
u(t) = \frac{1 + \operatorname{sign}(t-1)\lvert t-1\rvert^4}{4}, \qquad
f(t, y) = y - u(t) + \lvert t-1\rvert^3 .
$$

Then $y = u$ solves $y' = f(t,y)$ exactly, $u$ is $C^3$ and not $C^4$, and $f_y = 1$.

```python
def manufactured_exact(t):
    v = np.asarray(t, dtype=float)
    return np.atleast_1d((1.0 + np.sign(v - 1.0) * np.abs(v - 1.0) ** 4) / 4.0)

def manufactured(t, y):
    return ivp.as_state(y) - manufactured_exact(t) + abs(t - 1.0) ** 3

for name in ("euler", "heun", "kutta third", "rk4"):
    run = ivp.error_against_step(manufactured, manufactured_exact, 0.0, 0.0, 2.0,
                                 [25, 51, 101, 201, 401, 801], rk.named_step(name))
    print(f"{name:>14}: fitted order {run['fitted_order']:.4f}, "
          f"error at 801 steps {run['errors'][-1]:.3e}")
print()
out = rk.compare_at_equal_cost(manufactured, manufactured_exact, 0.0, 0.0, 2.0,
                               budgets=[51, 102, 204, 408, 816])
print(f"{'evaluations':>13}" + "".join(f"{n:>14}" for n in out["names"]))
for j, e in enumerate(out["evaluations"]):
    print(f"{e:>13}" + "".join(f"{out['errors'][n][j]:>14.3e}" for n in out["names"]))
print("\nbest at each budget: " + ", ".join(out["best_at_each_budget"]))
```

Now the four methods separate and each reaches its own order. **A single discontinuity in the
fourth derivative does not cost RK4 an order**, because it affects one panel out of $n$ and that
panel contributes $O(h^4)$ anyway. RK4 wins at every budget.

What would cost an order is a kink in a **lower** derivative, which is exercise 4.3.

The wider lesson is about the experiment, not the answer. **Five separate degeneracies stood
between the obvious test and a meaningful one**, and every one of them produced a clean looking
table. The only signal that something was wrong was that the numbers were too good: an exact zero
from a fourth order method on a non smooth problem, and two pairs of distinct methods agreeing to
the last digit.

### 4.2 Where RK4 overtakes Heun, against the tolerance

```python
print(f"{'target error':>14}{'heun evals':>13}{'rk4 evals':>12}{'winner':>10}")
for target in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10):
    costs = {}
    for name, order in (("heun", 2), ("rk4", 4)):
        n = 2
        while n < 4_000_000:
            run = ivp.integrate(f, 0.0, 0.5, 2.0, n, rk.named_step(name))
            if abs(float(run["y"][-1, 0]) - exact(2.0)) <= target:
                break
            n *= 2
        costs[name] = n * (2 if name == "heun" else 4)
    winner = min(costs, key=costs.get)
    print(f"{target:>14.0e}{costs['heun']:>13}{costs['rk4']:>12}{winner:>10}")
```

RK4 is cheaper at every tolerance here, including the loosest. The cost ratio runs
$4, 8, 16, 32, 128, 512$ across the sweep, so **the margin widens by roughly a factor of 2 per
decade of tolerance**, which is what the two rates predict: to gain one decade Heun needs
$\sqrt{10}$ times the steps and RK4 needs $10^{1/4}$, a ratio of $10^{1/4} = 1.78$ per decade.

**The crossover is below the table**, at tolerances so loose that Heun needs only two or three
steps, where neither method is in its asymptotic regime and the comparison stops meaning anything.
That is the honest answer for this problem: **there is no useful crossover**, and the reason people
still use second order methods is stability and simplicity rather than cost.

### 4.3 The order against the problem's smoothness

```python
print(f"{'a':>6}{'smoothness':>28}{'fitted order':>15}")
for a in (0.5, 1.5, 2.5, 3.5, 4.5, 6.0):
    def rough(t, y, a=a):
        return np.full_like(ivp.as_state(y), abs(t - 1.0) ** a)

    def rough_exact(t, a=a):
        return np.asarray([(np.sign(t - 1.0) * abs(t - 1.0) ** (a + 1.0) + 1.0)
                           / (a + 1.0)])

    run = ivp.error_against_step(rough, rough_exact, 0.0, 0.0, 2.0,
                                 [21, 41, 81, 161, 321, 641], rk.named_step("rk4"))
    label = f"C^{int(np.floor(a + 1.0))} and not C^{int(np.floor(a + 1.0)) + 1}"
    print(f"{a:>6.1f}{label:>28}{run['fitted_order']:>15.4f}")
```

The solution of $y' = \lvert t-1\rvert^a$ is $\lvert t-1\rvert^{a+1}/(a+1)$ up to sign, which is
$C^{\lfloor a+1\rfloor}$ and no better. The step counts are **odd**, so no grid point lands on
$t = 1$ and every run straddles the kink.

The fitted order rises with $a$ and saturates at 4, which is RK4's own order: **the achieved order
is $\min(4, a+1)$.** A method cannot use smoothness the problem does not have, and it cannot use
more smoothness than its own order needs.

### 5.1 The order 5 barrier

**The counting argument.** An explicit $s$ stage tableau has $s(s-1)/2$ free entries in $A$ and $s$
in $b$, so $s(s+1)/2$ parameters, minus $s$ for the constraints $c_i = \sum_j a_{ij}$ if $c$ is
regarded as determined. Order $p$ imposes $\sum_{k\le p} T_k$ conditions where $T_k$ is the rooted
tree count: 1, 2, 4, 8, 17 cumulative through order 5.

At $s = 5$: 15 parameters, 17 conditions. **Fewer unknowns than equations**, so generically there
is no solution.

**Why the count is not a proof.** The conditions are polynomial and highly structured, not generic.
Several are consequences of others: the whole point of the $c_i = \sum a_{ij}$ convention is that it
makes some conditions automatic, and Butcher's simplifying assumptions $\sum_i b_ia_{ij} = b_j(1 -
c_j)$ make more so. A system of 17 polynomial equations in 15 unknowns can perfectly well have
solutions if 3 of them are implied by the rest, and that is exactly what happens at orders 1 to 4,
where the naive count would also predict trouble at $s = 4$ (10 parameters, 8 conditions is fine,
but the margin is thin).

Butcher's actual proof is a contradiction argument on the specific structure. Assume order 5 with
5 stages; the order conditions force $b_5 c_5 (\text{stuff}) = 0$ for several distinct trees, and
combining them forces $b_5 = 0$, which reduces the method to 4 stages, which cannot have order 5
by the same argument applied recursively.

```python
def counts(s, p):
    trees = [1, 1, 2, 4, 9, 20, 48]
    parameters = s * (s - 1) // 2 + s
    conditions = sum(trees[:p])
    return parameters, conditions

print(f"{'stages':>8}{'order':>7}{'parameters':>13}{'conditions':>13}{'slack':>8}")
for s, p in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 5), (5, 4)):
    par, con = counts(s, p)
    print(f"{s:>8}{p:>7}{par:>13}{con:>13}{par - con:>8}")
```

The table shows the slack turning negative exactly at $s = p = 5$, and positive again at $s = 6$,
which is where order 5 does exist.

### 5.2 Composition of two methods

Let $M_1$ have tableau $(A^1, b^1, c^1)$ with $s_1$ stages and $M_2$ have $(A^2, b^2, c^2)$ with
$s_2$. Take a step of $M_1$ over $[t_n, t_n + \theta h]$ followed by a step of $M_2$ over
$[t_n + \theta h, t_n + h]$.

The composite is a Runge-Kutta method with $s_1 + s_2$ stages and

$$
A = \begin{pmatrix}\theta A^1 & 0\\ \theta\, \mathbf 1 (b^1)^{\mathsf T} & (1-\theta)A^2\end{pmatrix},
\qquad
b = \begin{pmatrix}\theta b^1\\ (1-\theta)b^2\end{pmatrix},
\qquad
c = \begin{pmatrix}\theta c^1\\ \theta \mathbf 1 + (1-\theta)c^2\end{pmatrix}.
$$

The block structure says: the second method's stages see the state **after** the first method's
whole step, which is the row $\theta\mathbf 1(b^1)^{\mathsf T}$.

```python
def compose(first, second, theta=0.5):
    A1, b1, c1, _ = rk.tableau(first)
    A2, b2, c2, _ = rk.tableau(second)
    s1, s2 = b1.size, b2.size
    A = np.zeros((s1 + s2, s1 + s2))
    A[:s1, :s1] = theta * A1
    A[s1:, :s1] = theta * np.outer(np.ones(s2), b1)
    A[s1:, s1:] = (1.0 - theta) * A2
    b = np.concatenate([theta * b1, (1.0 - theta) * b2])
    c = np.concatenate([theta * c1, theta * np.ones(s2) + (1.0 - theta) * c2])
    return A, b, c

for first, second in (("heun", "heun"), ("rk4", "rk4"), ("euler", "heun")):
    A, b, c = compose(first, second)
    out = rk.order_conditions(A, b, c, up_to=5)
    direct = ivp.integrate(f, 0.0, 0.5, 2.0, 100, general_step(A, b, c))
    print(f"{first} then {second}: {b.size} stages, order "
          f"{out['achieved_order']}, explicit {rk.is_explicit(A)}, "
          f"row sums match c: {np.allclose(A.sum(axis=1), c)}")
```

The composite of two order $p$ methods is order $p$, not $2p$: **composition does not raise the
order**, it only rearranges the work. What it does buy is symmetry, and composing a method with its
adjoint in the right proportions is exactly how Yoshida's fourth order symplectic methods are built
in lesson 74.

The set of Runge-Kutta methods under composition, together with the tree structure of the order
conditions, forms the **Butcher group**: the group of characters on the Hopf algebra of rooted
trees. The composition rule above is its product, written in coordinates.

### 5.3 What is being traded against Taylor

Every explicit Runge-Kutta method of order $p$ reproduces the same Taylor expansion of the solution
through $h^p$ as the Taylor method of order $p$ does, by construction: that is what the order
conditions say.

So the two methods produce **the same leading behaviour**, and they differ in three currencies.

**Information.** The Taylor method needs $f$ and its partial derivatives up to order $p-1$,
symbolically. Runge-Kutta needs $f$ alone, as a black box that can be called. That is the whole
trade, and it is why every general purpose library uses Runge-Kutta: a solver cannot differentiate
a function it was handed as a compiled subroutine.

**Evaluations.** Taylor order $p$ costs $p$ derivative evaluations; Runge-Kutta order $p$ costs $p$
evaluations of $f$ for $p \le 4$ and more above that. So for $p \le 4$ they are matched in count and
not in kind, and above 4 Runge-Kutta pays the stage barrier of exercise 5.1 while Taylor does not.

**Error constants.** They differ, and neither is uniformly better. Ralston's choice in exercise 2.2
shows a Runge-Kutta method tuning its constant, which a Taylor method cannot do: its coefficients
are forced.

```python
print(f"{'p':>4}{'Taylor derivative calls':>26}{'RK evaluations':>18}"
      f"{'RK stages needed':>20}")
barrier = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 9, 8: 11}
for p in range(1, 9):
    print(f"{p:>4}{p:>26}{barrier[p]:>18}{barrier[p]:>20}")
print("\nequal up to 4, and Runge-Kutta pays a stage barrier above it")
```

The last column is the honest summary of when a Taylor method is the right choice: **very high
order, on a problem whose derivatives are available cheaply**, which in practice means an
analytically defined right hand side and automatic differentiation. That is celestial mechanics,
and essentially nothing else.

---

## Lesson 70, Adaptive Step Size Control

### 1.1 Why an embedded estimate is free

An embedded pair computes two answers from the **same stages**:

$$
y_{n+1} = y_n + h\sum_i b_ik_i, \qquad \hat y_{n+1} = y_n + h\sum_i \hat b_ik_i .
$$

The $k_i$ are computed once. Forming the second answer is $s$ multiply-adds, which is not an
evaluation of $f$ at all, so the estimate costs **zero** evaluations.

Step halving computes three solutions instead: one step of size $h$, then two of size $h/2$. For an
$s$ stage method that is $3s$ evaluations, minus the shared first stage where the method has
$c_1 = 0$, so $3s - 1$ against $s$ for the plain step.

```python
import numpy as np
from nalib import adaptivestep as ad

out = ad.the_estimate_is_free()
print(f"{out['name']}: {out['stages']} stages, FSAL {out['first_same_as_last']}")
print(f"embedded pair: {out['embedded_evaluations_per_step']} evaluations per step")
print(f"step halving:  {out['step_halving_evaluations_per_step']} evaluations per step")
print(f"ratio {out['ratio']:.2f}x")
```

Dormand-Prince has 7 stages and the first-same-as-last property, so a plain step costs 6
evaluations and step halving costs 21, a factor of 3.5.

### 1.2 The control law, and which order goes in it

The local error of a method of order $q$ satisfies $e \approx Ch^{q+1}$ for a constant $C$ that
varies slowly with $t$. Two steps with the same $C$:

$$
\frac{\text{tol}}{e} = \frac{C h_{\text{new}}^{q+1}}{C h^{q+1}}
\quad\Longrightarrow\quad
h_{\text{new}} = h\left(\frac{\text{tol}}{e}\right)^{1/(q+1)} .
$$

**Which $q$?** The estimate $e = \lVert y - \hat y\rVert$ is dominated by the error of the **lower**
order answer, because the higher order one is much closer to the truth. So $q$ is the low order,
$p - 1$ if the pair is $(p, p-1)$, and the exponent is $1/p$.

That is what the code does, and it matters: using the wrong order gives an exponent wrong by a
factor of $(p+1)/p$, which makes the controller over- or under-react on every step.

```python
print(f"{'error / tol':>12}{'exponent 1/5':>14}{'exponent 1/6':>14}{'ratio':>9}")
for r in (1e-3, 0.1, 1.0, 10.0, 1e3):
    print(f"{r:>12.0e}{(1.0 / r) ** 0.2:>14.4f}{(1.0 / r) ** (1.0 / 6.0):>14.4f}"
          f"{(1.0 / r) ** 0.2 / (1.0 / r) ** (1.0 / 6.0):>9.4f}")
```

At a factor of $10^3$ out, the two exponents differ by 2.15 in the proposed step, which is the
difference between recovering in one step and taking three.

### 1.3 First-same-as-last

A tableau is **FSAL** when its last stage equals the first stage of the next step:
$c_s = 1$, and the last row of $A$ equals $b$. Then

$$
k_s = f\left(t_n + h,\ y_n + h\sum_j a_{sj}k_j\right) = f(t_{n+1}, y_{n+1}),
$$

which is exactly what the next step's $k_1$ would be. **Carry it forward and one evaluation per step
disappears.**

```python
from nalib import rungekutta as rk

for name in sorted(ad.PAIRS):
    A, bh, bl, c, ph, pl = ad.pair(name)
    print(f"{name:>18}: c_s = {c[-1]:.1f}, last row of A equals b_high: "
          f"{np.allclose(A[-1], bh)}, FSAL: {ad.is_first_same_as_last(name)}")
```

Dormand-Prince's 7 stages cost 6 evaluations per step this way, and Bogacki-Shampine's 4 cost 3.
Fehlberg is not FSAL and pays for all 6 of its stages.

### 1.4 Local extrapolation

The pair produces $y_{n+1}$ of order $p$ and $\hat y_{n+1}$ of order $p-1$, and the estimate
describes $\hat y$. **Local extrapolation** means returning $y$, the higher order one, anyway.

What it buys is an order: the returned answer converges at $h^p$ rather than $h^{p-1}$.

What it costs is the meaning of the tolerance. The controller sized the step so that
$\lVert y - \hat y\rVert \approx \text{tol}$, which bounds $\hat y$'s local error. Nothing bounds
$y$'s, and the relationship between the two is problem dependent.

It also costs FSAL, for a reason that is easy to miss: the carried stage is
$f(t_{n+1}, y_{n+1})$ computed with the **returned** $y_{n+1}$. Return $\hat y$ instead and the
stage sits at the wrong point.

### 1.5 Why the tolerance is not a bound on the final error

The controller works step by step. On each step it arranges

$$
\lVert \text{local error of this step} \rVert \lesssim \text{tol},
$$

and the final error is the accumulation of every step's local error, propagated by the equation.

Three things break the link. There are $N$ steps, so if the local errors added the total would be
$N\cdot\text{tol}$. They do not simply add, because each is propagated by a factor that can be
larger or smaller than 1, so on a decaying problem they partly cancel and on a growing one they
compound. And with local extrapolation on, the quantity controlled is not even the local error of
the answer returned.

**None of that is a defect.** It is what "local error control" means, and the fix is to treat the
tolerance as a knob and measure the error, which section 7 does.

### 2.1 Evaluations per step, step halving against an embedded pair

Let the method have $s$ stages and assume $c_1 = 0$, so the first stage is $f(t_n, y_n)$.

**Step halving.** One step of size $h$ costs $s$. Two steps of size $h/2$ cost $2s$, except that
the first half step's first stage is the same $f(t_n, y_n)$ already computed, saving 1. Total
$3s - 1$.

If the method is also FSAL, the last stage of the second half step is $f$ at the endpoint, which
can be carried to the next step, saving 1 more: $3s - 2$.

**Embedded pair.** $s$ evaluations, or $s-1$ with FSAL. The estimate is free.

$$
\text{ratio} = \frac{3s-1}{s} \to 3 \quad\text{as } s \text{ grows},
\qquad
\text{with FSAL: } \frac{3s-2}{s-1}.
$$

```python
print(f"{'stages':>8}{'plain':>8}{'halving':>10}{'ratio':>9}"
      f"{'FSAL plain':>13}{'FSAL halving':>15}{'FSAL ratio':>13}")
for s in (2, 4, 6, 7, 13):
    print(f"{s:>8}{s:>8}{3 * s - 1:>10}{(3 * s - 1) / s:>9.3f}"
          f"{s - 1:>13}{3 * s - 2:>15}{(3 * s - 2) / (s - 1):>13.3f}")
print()
out = ad.the_estimate_is_free("dormand prince")
print(f"nalib reports {out['embedded_evaluations_per_step']} against "
      f"{out['step_halving_evaluations_per_step']}, ratio {out['ratio']:.3f}")
```

The FSAL ratio is worse, at 3.17 for Dormand-Prince against 2.86 without, because FSAL saves one
evaluation from a small number and one from a large one.

### 2.2 The FSAL conditions

FSAL requires that the last stage of one step **is** the first stage of the next:

$$
k_s^{(n)} = f\left(t_n + c_sh,\ y_n + h\textstyle\sum_j a_{sj}k_j\right)
\quad\text{and}\quad
k_1^{(n+1)} = f\left(t_{n+1},\ y_{n+1}\right).
$$

Equating the arguments gives two conditions:

$$
c_s = 1 \qquad\text{and}\qquad a_{sj} = b_j \ \text{ for all } j,
$$

the second because $y_{n+1} = y_n + h\sum_j b_jk_j$.

There is a third, implied: since $A$ is strictly lower triangular, $a_{ss} = 0$, so $b_s = 0$. The
high order weights must **ignore the last stage**. That is not a restriction in practice, because
the last stage is being computed for the next step's benefit rather than this one's.

The low order weights $\hat b$ are unconstrained, and Dormand-Prince uses $\hat b_s \ne 0$, which is
what makes the two answers differ at all.

```python
A, bh, bl, c, ph, pl = ad.pair("dormand prince")
print(f"c_s = {c[-1]}")
print(f"last row of A: {np.round(A[-1], 8)}")
print(f"b_high:        {np.round(bh, 8)}")
print(f"they match: {np.allclose(A[-1], bh)}")
print(f"b_high[s] = {bh[-1]:.1f} (must be 0), b_low[s] = {bl[-1]:.8f} (must not be)")
```

### 2.3 The caps as a feedback loop

Treat the controller as a discrete dynamical system in $\log h$. With $e_n = C h_n^{q+1}$,

$$
\log h_{n+1} = \log h_n + \frac{1}{q+1}\left(\log\text{tol} - \log e_n\right)
             = \log h_n + \frac{1}{q+1}\left(\log\text{tol} - \log C - (q+1)\log h_n\right),
$$

so

$$
\log h_{n+1} = \frac{\log\text{tol} - \log C}{q+1},
$$

which is **constant**: with a perfectly known $C$ the controller reaches the right step in one move
and stays. It is a deadbeat controller.

What breaks that is $C$ varying. Write $C_n = C(1 + \delta_n)$ with $\delta_n$ the relative change
between steps. Then

$$
\log h_{n+1} - \log h^{*} = -\frac{1}{q+1}\log(1 + \delta_n),
$$

so a sudden change in $C$ by a factor $\Delta$ moves the step by $\Delta^{-1/(q+1)}$. **The
exponent damps the response**, which is why the plain controller is stable but slow.

The caps bound that response. Growth capped at 5 means the step cannot rise faster than $5^n$ into
a region the solver has not seen; shrink floored at 0.2 means one bad estimate cannot collapse the
step. Both are there because $\delta_n$ can be enormous, as it is when a solution turns.

```python
print(f"{'C changes by':>14}{'raw response':>15}{'after caps':>13}")
for change in (1e-6, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6):
    raw = change ** (-1.0 / 5.0)
    capped = ad.proposed_step(0.1, change * 1e-6, 1e-6, order=4) / 0.1
    print(f"{change:>14.0e}{raw:>15.4f}{capped:>13.4f}")
```

### 2.4 What the difference estimates, and the next term

Let $y(t_{n+1})$ be the exact solution through $(t_n, y_n)$. Then

$$
y_{n+1} = y(t_{n+1}) + C_p h^{p+1} + O(h^{p+2}), \qquad
\hat y_{n+1} = y(t_{n+1}) + C_{p-1}h^{p} + O(h^{p+1}),
$$

so

$$
y_{n+1} - \hat y_{n+1} = -C_{p-1}h^{p} + \left(C_p - C_p'\right)h^{p+1} + O(h^{p+2}).
$$

**The leading term is exactly the low order method's local error**, with a sign, and the next term
is $O(h^{p+1})$, which is the same size as the **high** order method's error. So the estimate
describes $\hat y$ to leading order and says nothing useful about $y$.

```python
from nalib import ivp

def f(t, y):
    return ivp.as_state(y) - t ** 2 + 1.0

def exact(t):
    return (t + 1.0) ** 2 - 0.5 * np.exp(t)

out = ad.the_estimate_predicts_the_error(f, exact, 0.0, 0.5, 2.0)
print(f"{'h':>10}{'estimate':>13}{'low error':>13}{'high error':>13}"
      f"{'est/low':>10}{'est/high':>12}")
for h, e, lo, hi, rl, rh in zip(out["h"], out["estimate"], out["true_low_error"],
                                out["true_high_error"], out["estimate_over_low"],
                                out["estimate_over_high"]):
    print(f"{h:>10.5f}{e:>13.3e}{lo:>13.3e}{hi:>13.3e}{rl:>10.4f}{rh:>12.3e}")
```

The `est/low` column climbs to 1 as $h$ shrinks, at a rate of one order, which is the $O(h^{p+1})$
correction dying. The `est/high` column **doubles** each time the step halves, because the estimate
is $O(h^p)$ and the high order error is $O(h^{p+1})$.

### 2.5 A PI controller

The plain law reacts to one error. A **PI controller** uses two:

$$
h_{n+1} = h_n\left(\frac{\text{tol}}{e_n}\right)^{\alpha}
              \left(\frac{e_{n-1}}{e_n}\right)^{\beta},
$$

with $\alpha \approx 0.7/(q+1)$ and $\beta \approx 0.4/(q+1)$ the standard choice (Gustafsson).

What it fixes: the plain controller is deadbeat and therefore **maximally oscillatory** when the
estimate is noisy. It moves the step all the way to where one measurement says it should be, so a
single unlucky estimate produces a large move and then a large move back. The PI form uses a
smaller $\alpha$ and adds a term that damps the change, trading response speed for smoothness.

```python
def solve_with(controller, tol=1e-8, t_end=2.0):
    """PECE-free adaptive Dormand-Prince with a pluggable step controller."""
    t, y, h = 0.0, np.asarray([0.5]), 0.05
    accepted, rejected, previous = 0, 0, None
    while t < t_end:
        h = min(h, t_end - t)
        step = ad.embedded_step(f, t, y, h, "dormand prince")
        e = max(float(step["error_norm"]), 1e-300)
        if e <= tol:
            y, t = step["high"], t + h
            accepted += 1
            h = controller(h, e, previous, tol)
            previous = e
        else:
            rejected += 1
            h = min(h, controller(h, e, previous, tol))
    return float(y[0]), accepted, rejected

def plain(h, e, previous, tol):
    return h * min(5.0, max(0.2, 0.9 * (tol / e) ** 0.2))

def pi(h, e, previous, tol):
    factor = 0.9 * (tol / e) ** (0.7 / 5.0)
    if previous is not None:
        factor *= (previous / e) ** (0.4 / 5.0)
    return h * min(5.0, max(0.2, factor))

print(f"{'tolerance':>12}{'controller':>12}{'accepted':>10}{'rejected':>10}{'error':>13}")
for tol in (1e-6, 1e-8, 1e-10):
    for name, controller in (("plain", plain), ("PI", pi)):
        value, a, r = solve_with(controller, tol)
        print(f"{tol:>12.0e}{name:>12}{a:>10}{r:>10}{abs(value - exact(2.0)):>13.3e}")
```

On this smooth problem the two are close, because the error constant barely varies and there is
nothing to damp. Exercise 3.5 puts them on a problem where it does.

### 3.1 An embedded pair solver from scratch

```python
def my_solve(f, t0, y0, t_end, tol=1e-8, name="dormand prince", extrapolate=True):
    """Adaptive embedded pair solver, written out."""
    A, bh, bl, c, ph, pl = ad.pair(name)
    stages = bh.size
    order = ph if extrapolate else pl
    t = float(t0)
    y = ivp.as_state(y0)
    h = (float(t_end) - t) / 100.0
    accepted, rejected, evaluations = 0, 0, 0
    while t < t_end - 1e-14:
        h = min(h, t_end - t)
        k = np.empty((stages, y.size))
        for i in range(stages):
            shift = y.copy()
            for j in range(i):
                if A[i, j] != 0.0:
                    shift = shift + h * A[i, j] * k[j]
            k[i] = ivp.as_state(f(t + c[i] * h, shift))
            evaluations += 1
        high = y + h * (bh @ k)
        low = y + h * (bl @ k)
        e = float(np.max(np.abs(high - low)))
        if e <= tol or h < 1e-14:
            y = high if extrapolate else low
            t += h
            accepted += 1
        else:
            rejected += 1
        factor = 0.9 * (tol / max(e, 1e-300)) ** (1.0 / (min(ph, pl) + 1))
        h *= min(5.0, max(0.2, factor))
    return {"y": y, "accepted": accepted, "rejected": rejected,
            "evaluations": evaluations}

print(f"{'tolerance':>12}{'mine':>16}{'nalib':>16}{'gap':>12}"
      f"{'my steps':>11}{'nalib steps':>13}")
for tol in (1e-4, 1e-6, 1e-8, 1e-10):
    mine = my_solve(f, 0.0, 0.5, 2.0, tol)
    theirs = ad.solve(f, 0.0, 0.5, 2.0, tol=tol)
    a, b = float(mine["y"][0]), float(theirs["y"][-1, 0])
    print(f"{tol:>12.0e}{a:>16.10f}{b:>16.10f}{abs(a - b):>12.2e}"
          f"{mine['accepted']:>11}{theirs['accepted']:>13}")
```

The answers agree to the tolerance, which is all that can be asked: two adaptive solvers do not
take the same steps unless every detail of the controller matches, and the step counts differ
because `nalib`'s reuses the FSAL stage.

### 3.2 Step doubling for RK4

```python
def rk4_with_doubling(f, t0, y0, t_end, tol=1e-8):
    """RK4 with a step doubling error estimate: one step of h against two of h/2."""
    step = rk.named_step("rk4")
    t = float(t0)
    y = ivp.as_state(y0)
    h = (float(t_end) - t) / 100.0
    accepted, rejected, evaluations = 0, 0, 0
    while t < t_end - 1e-14:
        h = min(h, t_end - t)
        big = ivp.as_state(step(f, t, y, h))
        half = ivp.as_state(step(f, t, y, 0.5 * h))
        small = ivp.as_state(step(f, t + 0.5 * h, half, 0.5 * h))
        evaluations += 12
        e = float(np.max(np.abs(small - big))) / 15.0     # Richardson, order 4
        if e <= tol or h < 1e-14:
            y = small + (small - big) / 15.0              # local extrapolation
            t += h
            accepted += 1
        else:
            rejected += 1
        h *= min(5.0, max(0.2, 0.9 * (tol / max(e, 1e-300)) ** 0.2))
    return {"y": y, "accepted": accepted, "evaluations": evaluations,
            "rejected": rejected}

print(f"{'tolerance':>12}{'doubling error':>16}{'evals':>8}"
      f"{'DP error':>14}{'evals':>8}{'ratio':>8}")
for tol in (1e-6, 1e-8, 1e-10, 1e-12):
    d = rk4_with_doubling(f, 0.0, 0.5, 2.0, tol)
    counter = [0]
    p = ad.solve(f, 0.0, 0.5, 2.0, tol=tol, _counter=counter)
    de = abs(float(d["y"][0]) - exact(2.0))
    pe = abs(float(p["y"][-1, 0]) - exact(2.0))
    print(f"{tol:>12.0e}{de:>16.3e}{d['evaluations']:>8}"
          f"{pe:>14.3e}{counter[0]:>8}{d['evaluations'] / counter[0]:>8.2f}")
```

Step doubling costs 12 evaluations per attempted step against Dormand-Prince's 6, and it is a
fourth order method against a fifth. **Both disadvantages point the same way**, and the ratio in
the last column is what it costs.

Step doubling is still worth knowing: it works for **any** method, including ones with no embedded
partner, and it is how an error estimate is obtained for a symplectic method or a
problem specific scheme.

### 3.3 Dense output for Dormand-Prince

A **continuous extension** gives $y(t_n + \theta h)$ for any $\theta \in [0,1]$ from stages already
computed. The cheapest useful one is cubic Hermite, which needs four pieces of information and has
all four for free:

$$
y_n, \qquad y_{n+1}, \qquad f(t_n, y_n) = k_1, \qquad f(t_{n+1}, y_{n+1}) = k_7 ,
$$

the last by FSAL. In terms of $\theta$,

$$
u(\theta) = (1 - \theta)y_n + \theta y_{n+1}
  + \theta(\theta - 1)\Big[(1-2\theta)(y_{n+1} - y_n)
  + (\theta - 1)h k_1 + \theta h k_7\Big].
$$

```python
def hermite(theta, y_now, y_next, k_first, k_last, h):
    """Cubic Hermite interpolation across one step, from data the step already produced."""
    s = float(theta)
    return ((1.0 - s) * y_now + s * y_next
            + s * (s - 1.0) * ((1.0 - 2.0 * s) * (y_next - y_now)
                               + (s - 1.0) * h * k_first + s * h * k_last))

t0, y0, h = 0.0, np.asarray([0.5]), 0.25
step = ad.embedded_step(f, t0, y0, h, "dormand prince")
k = step["stages"]
print(f"{'theta':>8}{'dense output':>18}{'exact':>18}{'error':>12}")
for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
    value = float(hermite(theta, y0, step["high"], k[0], k[-1], h)[0])
    want = exact(t0 + theta * h)
    print(f"{theta:>8.2f}{value:>18.12f}{want:>18.12f}{abs(value - want):>12.2e}")
gap = abs(float(hermite(1.0, y0, step["high"], k[0], k[-1], h)[0])
          - float(step["high"][0]))
print(f"\nat theta = 1 it reproduces the step exactly: gap {gap:.2e}")
print(f"at theta = 0 it reproduces y_n exactly:      gap "
      f"{abs(float(hermite(0.0, y0, step['high'], k[0], k[-1], h)[0]) - 0.5):.2e}")
```

```python
print(f"{'h':>8}{'worst interior error':>24}{'ratio':>9}")
last = None
for h in (0.4, 0.2, 0.1, 0.05, 0.025):
    step = ad.embedded_step(f, 0.0, np.asarray([0.5]), h, "dormand prince")
    k = step["stages"]
    worst = max(abs(float(hermite(s, np.asarray([0.5]), step["high"], k[0], k[-1], h)[0])
                    - float(exact(s * h)))
                for s in np.linspace(0.02, 0.98, 49))
    print(f"{h:>8.3f}{worst:>24.4e}" + ("" if last is None else f"{last / worst:>9.2f}"))
    last = worst
```

The error ratio is 16 per halving, so **the interpolant is fourth order**, one below the step's
fifth. It matches the step exactly at both ends, it costs **no extra evaluations at all**, and it
is enough for the event detection of exercise 3.4.

Dormand-Prince also has a genuine fifth order continuous extension, using all seven stages with
$\theta$ dependent weights rather than only the two endpoint derivatives. It is what `solve_ivp`
uses, and the extra order costs only the coefficient table.

### 3.4 Event detection

With an interpolant in hand, an event $g(t, y) = 0$ is found by watching for a sign change across a
step and then calling a root finder on the interpolant.

```python
from nalib.roots import brent

def solve_until_event(g, t0, y0, t_end, tol=1e-10):
    """Integrate until g(t, y) crosses zero, locating the crossing on the interpolant."""
    t = float(t0)
    y = ivp.as_state(y0)
    h = 0.05
    while t < t_end - 1e-14:
        h = min(h, t_end - t)
        step = ad.embedded_step(f, t, y, h, "dormand prince")
        e = max(float(step["error_norm"]), 1e-300)
        if e > tol:
            h *= max(0.2, 0.9 * (tol / e) ** 0.2)
            continue
        before = float(g(t, y))
        after = float(g(t + h, step["high"]))
        if before * after < 0.0:
            k = step["stages"]

            def on_interpolant(s, k=k, t=t, y=y, h=h, step=step):
                value = hermite(s, y, step["high"], k[0], k[-1], h)
                return float(g(t + s * h, value))

            found = brent(on_interpolant, 0.0, 1.0, tol=1e-14)
            theta = float(found.root)
            return t + theta * h, hermite(theta, y, step["high"], k[0], k[-1], h)
        y, t = step["high"], t + h
        h *= min(5.0, 0.9 * (tol / e) ** 0.2)
    return None, None

target = 3.0
when, where = solve_until_event(lambda t, y: float(np.ravel(y)[0]) - target, 0.0, 0.5, 2.0)
print(f"y reaches {target} at t = {when:.12f}, where y = {float(np.ravel(where)[0]):.12f}")

low, high = 0.0, 2.0
for _ in range(200):
    mid = 0.5 * (low + high)
    if (exact(mid) - target) * (exact(low) - target) <= 0.0:
        high = mid
    else:
        low = mid
truth = 0.5 * (low + high)
print(f"the exact crossing is at   {truth:.12f}")
print(f"error in the located event: {abs(when - truth):.2e}")
```

The event is located to within the interpolant's own accuracy, and **no extra evaluations of $f$
were needed to find it**, which is the second payoff from the continuous extension. The root finder
runs entirely on stages that already existed.

### 3.5 A PI controller, and a failure mode neither controller escapes

Before comparing controllers, there is a larger problem to report.

```python
AMPLITUDE, WIDTH, CENTRE = 200.0, 400.0, 1.5

def pulse(t, y):
    return -ivp.as_state(y) + AMPLITUDE * np.exp(-WIDTH * (t - CENTRE) ** 2)

def pulse_exact(t):
    import math
    shift = CENTRE + 1.0 / (2.0 * WIDTH)
    root = math.sqrt(WIDTH)
    scale = (AMPLITUDE * math.exp(WIDTH * shift ** 2 - WIDTH * CENTRE ** 2)
             * math.sqrt(math.pi) / (2.0 * root))
    v = np.atleast_1d(np.asarray(t, dtype=float))
    return np.asarray([scale * math.exp(-float(x))
                       * (math.erf(root * (float(x) - shift)) + math.erf(root * shift))
                       for x in v])

def run_pulse(tol, h0, max_step, controller):
    t, y, h = 0.0, np.asarray([0.0]), h0
    accepted, rejected, previous, largest = 0, 0, None, 0.0
    while t < 3.0 - 1e-14:
        h = min(h, 3.0 - t, max_step)
        step = ad.embedded_step(pulse, t, y, h, "dormand prince")
        e = max(float(step["error_norm"]), 1e-300)
        if e <= tol:
            largest = max(largest, h)
            y, t = step["high"], t + h
            accepted += 1
            h = controller(h, e, previous, tol)
            previous = e
        else:
            rejected += 1
            h = min(h, controller(h, e, previous, tol))
    return float(y[0]), accepted, rejected, largest

want = float(pulse_exact(3.0)[0])
print(f"the exact answer is y(3) = {want:.10f}\n")
print(f"{'tolerance':>11}{'first step':>12}{'max step':>11}{'accepted':>10}"
      f"{'largest h':>12}{'error':>13}")
for tol in (1e-6, 1e-8, 1e-10):
    for h0, cap in ((0.05, 1e9), (0.005, 1e9), (0.05, 0.05)):
        value, a, r, big = run_pulse(tol, h0, cap, plain)
        print(f"{tol:>11.0e}{h0:>12.3f}{cap:>11.3g}{a:>10}{big:>12.4f}"
              f"{abs(value - want):>13.3e}")
```

**The first row of each block is a total failure and it reports success.** Starting from
$h_0 = 0.05$ in the flat region, the controller sees a tiny error, multiplies the step by 5, and
does it again: $0.05, 0.25, 1.25$. The third step spans $[0.3, 1.55]$, jumping the pulse in one
stride, and the estimate for that step is small because **both ends of it are in flat regions**.

The returned answer is 3.957 away from the truth, which is the whole size of the solution. It
happens at every tolerance, $10^{-10}$ included, because tightening the tolerance does not change
the first three steps: they were already far below it.

Two independent fixes work: start smaller, or cap the step. Both are standard, and a solver that
offers neither is not safe on a problem with a localised feature.

Now the controller comparison, under a step cap so that both controllers are solving the problem.

```python
def plain(h, e, previous, tol):
    return h * min(5.0, max(0.2, 0.9 * (tol / e) ** 0.2))

def pi(h, e, previous, tol):
    factor = 0.9 * (tol / e) ** (0.7 / 5.0)
    if previous is not None:
        factor *= (previous / e) ** (0.4 / 5.0)
    return h * min(5.0, max(0.2, factor))

print(f"{'tolerance':>11}{'controller':>12}{'accepted':>10}{'rejected':>10}"
      f"{'rejection rate':>16}{'error':>13}")
for tol in (1e-6, 1e-8, 1e-10):
    for name, controller in (("plain", plain), ("PI", pi)):
        value, a, r, big = run_pulse(tol, 0.005, 0.05, controller)
        print(f"{tol:>11.0e}{name:>12}{a:>10}{r:>10}{r / (a + r):>16.3f}"
              f"{abs(value - want):>13.3e}")
```

Both controllers must reject steps at the leading edge of the pulse, because **a controller looking
backwards cannot see forwards** and the only way to discover the pulse is to step into it and fail.
What the PI form buys is what happens afterwards: it does not overshoot on the way back out, so its
rejection rate is lower once the pulse has been found.

### 4.1 The rejection rate against the safety factor

```python
def solve_with_safety(safety, tol=1e-8):
    t, y, h = 0.0, np.asarray([0.0]), 0.005
    accepted, rejected, evaluations = 0, 0, 0
    while t < 3.0 - 1e-14:
        h = min(h, 3.0 - t, 0.05)
        step = ad.embedded_step(pulse, t, y, h, "dormand prince")
        evaluations += 7
        e = max(float(step["error_norm"]), 1e-300)
        if e <= tol:
            y, t = step["high"], t + h
            accepted += 1
        else:
            rejected += 1
        h *= min(5.0, max(0.2, safety * (tol / e) ** 0.2))
    return accepted, rejected, evaluations, abs(float(y[0]) - want)

print(f"{'safety':>8}{'accepted':>10}{'rejected':>10}{'rate':>9}"
      f"{'evaluations':>13}{'error':>12}")
best, best_evals = None, None
for safety in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0):
    a, r, e, err = solve_with_safety(safety)
    print(f"{safety:>8.2f}{a:>10}{r:>10}{r / (a + r):>9.3f}{e:>13}{err:>12.3e}")
    if best_evals is None or e < best_evals:
        best, best_evals = safety, e
print(f"\nfewest evaluations at safety = {best}, costing {best_evals}")
```

Two effects pull against each other. A **small** safety factor takes steps well under the tolerance,
so it rejects almost nothing and takes more steps than it needs. A safety factor of **1** aims
exactly at the tolerance, so about half the steps land above it and each rejection throws away a
whole step's evaluations.

The optimum sits where the marginal cost of a rejection equals the marginal saving from a larger
step, which is the range the standard value of 0.9 comes from.

### 4.2 Achieved error against tolerance, both ways

```python
print(f"{'tolerance':>11}{'with extrapolation':>22}{'ratio':>9}"
      f"{'without':>16}{'ratio':>9}{'per decade':>12}")
previous = None
for tol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
    on = ad.solve(pulse, 0.0, 0.0, 3.0, tol=tol, local_extrapolation=True)
    off = ad.solve(pulse, 0.0, 0.0, 3.0, tol=tol, local_extrapolation=False)
    a = abs(float(on["y"][-1, 0]) - want)
    b = abs(float(off["y"][-1, 0]) - want)
    step = "" if previous is None else f"{((b / tol) / previous) ** 0.5:.3f}"
    print(f"{tol:>11.0e}{a:>22.3e}{a / tol:>9.2f}{b:>16.3e}{b / tol:>9.2f}{step:>9}")
    previous = b / tol
```

**The two shapes are opposite, and not the way round you might guess.**

With extrapolation the ratio is roughly constant, wandering between 1.6 and 3.7 over eight
decades. Without it the ratio **grows**, from 4.3 to 252, by a factor of 1.62 to 1.74 per decade.

The mechanism is the returned answer's order. The controller sizes the step from the same estimate
in both runs, so the step sequence is nearly the same; what differs is which answer is kept. With
extrapolation the kept answer is fifth order, and its error falls at the same rate the tolerance
does. Without it the kept answer is fourth order, so its error falls one power of $h$ more slowly,
and since $h \sim \text{tol}^{1/5}$ the ratio grows like $\text{tol}^{-1/5}$, which is
$10^{0.2} = 1.58$ per decade. **The measured per decade factors are 1.74, 1.67, 1.63, 1.62**,
settling towards that law from above.

So the honest summary inverts the usual worry. Local extrapolation is the version whose tolerance
behaves like a tolerance, and the "honest" version without it has a tolerance whose meaning drifts.

### 4.3 The chosen step against the local Lipschitz constant

```python
from nalib.ivp import lipschitz_constant

out = ad.where_the_work_went(pulse, 0.0, 0.0, 3.0, tol=1e-8)
picks = np.linspace(0, out["t"].size - 1, 14).astype(int)
print(f"{'t':>8}{'step':>13}{'|f| there':>14}{'local L':>12}{'step x L':>12}")
for i in picks:
    t = float(out["t"][i])
    y = float(pulse_exact(t)[0])
    L = lipschitz_constant(pulse, t, y - 0.5, y + 0.5, samples=200)["estimate"]
    scale = abs(float(pulse(t, np.asarray([y]))[0]))
    print(f"{t:>8.4f}{out['h'][i]:>13.3e}{scale:>14.4f}{L:>12.4f}"
          f"{out['h'][i] * L:>12.3e}")
print(f"\nstep varies by {out['ratio']:.1f}x, Lipschitz constant does not vary at all")
```

**The Lipschitz constant is exactly 1 everywhere**, because $\partial f/\partial y = -1$ for this
problem, and the step varies by a factor of 159. So the step is not tracking the Lipschitz constant
at all.

What it tracks is the **local error constant**, which for a fifth order method involves the sixth
derivative of the solution. That is large where the pulse is and small elsewhere, and it has nothing
to do with $f_y$.

The distinction is worth stating plainly. The Lipschitz constant controls **stability**, which is
what sets the step on a stiff problem (lesson 72). The high derivatives of the solution control
**accuracy**, which is what sets it here. An adaptive controller measures accuracy and is blind to
stability, which is exactly why an explicit adaptive solver on a stiff problem takes tiny steps and
reports no difficulty at all.

### 5.1 When the mismatch makes the controller misbehave

The estimate describes $\hat y$ and the solver returns $y$. That is safe as long as the estimate is
an **over**estimate of the returned answer's error, which it normally is by a wide margin: the two
answers differ by an order, so $e \approx \lVert$ low order error $\rVert$ is much larger than
$\lVert$ high order error $\rVert$.

What breaks it is a point where the **two errors happen to coincide**. The estimate is their
difference, so it collapses there while neither error does, and the controller reads a step it is
nowhere near as accurate on as it believes.

```python
def scan(h, t_end=2.0, n=400):
    """Walk the exact solution and compare the estimate against both true errors."""
    rows = []
    for i in range(n):
        t = t_end * i / n
        y = np.asarray([float(exact(t))])
        step = ad.embedded_step(f, t, y, h, "dormand prince")
        want = float(exact(t + h))
        rows.append((t, float(step["error_norm"]),
                     abs(float(step["low"][0]) - want),
                     abs(float(step["high"][0]) - want)))
    return rows

print(f"{'h':>7}{'median est/low':>16}{'median est/high':>17}"
      f"{'smallest est/high':>19}{'under-predicts':>16}")
for h in (0.2, 0.1, 0.05):
    rows = scan(h)
    low_ratio = np.asarray([r[1] / max(r[2], 1e-300) for r in rows])
    high_ratio = np.asarray([r[1] / max(r[3], 1e-300) for r in rows])
    under = sum(1 for r in rows if r[1] < r[3])
    print(f"{h:>7.3f}{float(np.median(low_ratio)):>16.4f}"
          f"{float(np.median(high_ratio)):>17.2f}{float(np.min(high_ratio)):>19.4f}"
          f"{under:>10} of 400")
print()
rows = scan(0.2)
high_ratio = np.asarray([r[1] / max(r[3], 1e-300) for r in rows])
j = int(np.argmin(high_ratio))
print(f"the worst point at h = 0.2 is t = {rows[j][0]:.4f}:")
print(f"  estimate        {rows[j][1]:.3e}")
print(f"  low order error {rows[j][2]:.3e}   (the estimate tracks this, ratio "
      f"{rows[j][1] / rows[j][2]:.3f})")
print(f"  returned error  {rows[j][3]:.3e}   (the estimate is {rows[j][3] / rows[j][1]:.1f} "
      f"times too small)")
```

Read the last two columns. **The estimate is normally 12 to 47 times larger than the returned
answer's error**, which is why local extrapolation is safe most of the time. And at 5 points out of
400 at $h = 0.2$ it is **smaller**, by up to a factor of 7.5.

The worst point makes the mechanism explicit. At $t = 1.235$ the low order error is
$5.90\times10^{-9}$ and the returned error is $5.20\times10^{-9}$: **the two answers are almost
equally wrong**, in nearly the same direction, so their difference is $7\times10^{-10}$, an eighth
of either. The estimate is not tracking the low order error there either; it is tracking a
cancellation.

At such a point the controller believes it is well inside the tolerance while the answer it is
keeping is not. The step it then proposes is too large, and the next step inherits a state that is
already worse than advertised.

Two things keep this from being a disaster in practice. The bad points are isolated, so the
controller passes through them in one or two steps and recovers. And the growth cap of 5 bounds how
far the step can run away in a single move, which is the third reason that cap exists: the first
two are stability of the feedback loop and safety against entering an unexplored region, and the
third is that the estimate is occasionally simply wrong.

Notice also the trend in the last column: 5 points at $h = 0.2$, 2 at $h = 0.1$, 1 at $h = 0.05$.
**Refining the step makes the failure rarer and does not remove it**, because the zero of the error
constant is a property of the problem and the step only controls how often the solver lands near it.

### 5.2 Order reduction on a stiff problem

A Runge-Kutta method has a **stage order** as well as an order: the largest $q$ for which

$$
\sum_j a_{ij}c_j^{k-1} = \frac{c_i^k}{k}, \qquad k = 1, \dots, q,
$$

holds for every stage $i$. For an explicit method with $c_1 = 0$ the first stage forces $q \le 1$.

On a non stiff problem the stage order is invisible, because the classical analysis assumes
$h\lambda \to 0$ and the stage errors are absorbed into higher order terms. On a stiff problem run
with $h\lambda$ **large and fixed**, the stage errors are not damped, and the observed order falls
towards the stage order plus one.

```python
def order_on(lam, name="rk4", counts=(20, 40, 80, 160, 320)):
    def stiff(t, y):
        return lam * (ivp.as_state(y) - np.cos(t)) - np.sin(t)

    def stiff_exact(t):
        return np.asarray([np.cos(t)])

    run = ivp.error_against_step(stiff, stiff_exact, 0.0, 1.0, 1.0, list(counts),
                                 rk.named_step(name))
    return run["fitted_order"], float(run["errors"][-1]), float(run["errors"][0])

print(f"{'lambda':>10}{'|lam h| at 20 steps':>22}{'fitted order':>15}"
      f"{'error at 320':>15}")
for lam in (-1.0, -10.0, -100.0, -500.0):
    order, fine, coarse = order_on(lam)
    label = "diverged" if not np.isfinite(order) else f"{order:.4f}"
    print(f"{lam:>10.0f}{abs(lam) / 20.0:>22.2f}{label:>15}{fine:>15.3e}")
```

**The effect cannot be exhibited with RK4 at all**, and the reason is the answer to the exercise. To
see order reduction you need $\lambda h$ large across the whole refinement sweep. For an explicit
method that is outside the stability region, so the coarse runs do not merely lose order, they
diverge, and the fit returns nothing.

So order reduction is a phenomenon of **implicit** methods, because they are the only ones that can
be run with $\lambda h$ large and still produce numbers. It is why Radau IIA, whose stage order is
$s$ rather than 1, is the standard high order stiff solver: the classical order matters less than
the stage order once the problem is stiff enough.

### 5.3 Stiffness detection from the stages

A solver can estimate the dominant eigenvalue from stages it has already computed. If two stages
$k_i$ and $k_j$ are evaluated at nearby points $Y_i$ and $Y_j$, then

$$
\frac{\lVert k_i - k_j\rVert}{\lVert Y_i - Y_j\rVert}
\approx \left\lVert \frac{\partial f}{\partial y} \right\rVert,
$$

a difference quotient of $f$ that costs nothing because both stages exist already.

For Dormand-Prince the standard choice is stages 6 and 7, which are evaluated at $c = 1$ and $c = 1$
with nearly the same argument. The solver then compares $h\lambda_{\text{est}}$ against the method's
real axis stability limit, and if the ratio approaches 1 the step is stability limited rather than
accuracy limited, which is the signal to switch to a stiff method.

```python
def stiffness_estimate(f, t, y, h, name="dormand prince"):
    """|df/dy| from two stages that are already computed, at no extra cost."""
    A, bh, bl, c, ph, pl = ad.pair(name)
    step = ad.embedded_step(f, t, y, h, name)
    k = step["stages"]
    state = ivp.as_state(y)
    args = [state + h * (A[i] @ k) for i in range(len(bh))]
    gap_k = np.linalg.norm(k[-1] - k[-2])
    gap_y = np.linalg.norm(args[-1] - args[-2])
    return gap_k / max(gap_y, 1e-300)

limit = rk.stability_limit("rk4")
print(f"{'true lambda':>12}{'estimated':>13}{'h':>10}{'h * est':>10}"
      f"{'limit':>9}{'stability limited?':>21}")
for lam in (-1.0, -10.0, -100.0, -1000.0):
    def test(t, y, lam=lam):
        return lam * ivp.as_state(y)

    for h in (0.05, 0.002):
        est = stiffness_estimate(test, 0.0, np.asarray([1.0]), h)
        print(f"{lam:>12.0f}{est:>13.4f}{h:>10.4f}{h * est:>10.4f}{limit:>9.4f}"
              f"{str(h * est > 0.5 * limit):>21}")
```

The estimate recovers $\lvert\lambda\rvert$ to a few digits from stages the solver had anyway, and
$h\lambda_{\text{est}}$ against the stability limit is the diagnostic. **The cost is two norms per
step**, which is why `LSODA` and its relatives can afford to run this check continuously and switch
between an Adams method and a BDF method as the problem changes.

---

## Lesson 71, Multistep Methods

### 1.1 AB2 and AM2, and the rules behind them

**AB2.** Interpolate $f$ at $t_n$ and $t_{n-1}$ by the straight line through
$(t_{n-1}, f_{n-1})$ and $(t_n, f_n)$, then integrate that line over $[t_n, t_{n+1}]$, which is
**outside** the interpolation interval:

$$
y_{n+1} = y_n + \frac{h}{2}\left(3f_n - f_{n-1}\right).
$$

It is the two point **Newton-Cotes open** idea used as an extrapolation, and the weights $3/2$ and
$-1/2$ sum to 1 as they must.

**AM2.** Interpolate at $t_n$ and $t_{n+1}$ instead and integrate over the same interval, which is
now **inside**:

$$
y_{n+1} = y_n + \frac{h}{2}\left(f_{n+1} + f_n\right),
$$

the **trapezoid rule**, implicit because $f_{n+1}$ depends on $y_{n+1}$.

```python
import numpy as np
from nalib import multistep as ms

print("Adams-Bashforth:")
for k in range(1, 4):
    print(f"  AB{k}: {[str(v) for v in ms.adams_bashforth_coefficients(k)]}")
print("Adams-Moulton:")
for k in range(1, 4):
    print(f"  AM{k+1}: {[str(v) for v in ms.adams_moulton_coefficients(k)]}")
```

### 1.2 Why an explicit multistep method costs one evaluation per step

Take AB4:

$$
y_{n+1} = y_n + \frac{h}{24}\left(55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3}\right).
$$

At step $n$ the values $f_{n-1}, f_{n-2}, f_{n-3}$ were computed at steps $n-1$, $n-2$ and $n-3$ and
can be kept in a short queue. The **only** new quantity is $f_n = f(t_n, y_n)$, and that is one
evaluation whatever $k$ is.

Compare RK4, whose four stages are evaluated at points **inside** the step and are therefore new
every time. Nothing computed in step $n$ lies on the grid of step $n+1$, so nothing can be reused.

```python
from nalib import ivp

counter = [0]
run = ms.solve_explicit(lambda t, y: ivp.as_state(y) - t ** 2 + 1.0, 0.0, 0.5, 2.0, 100,
                        "ab4", _counter=counter)
print(f"AB4, 100 steps: {counter[0]} evaluations, "
      f"{counter[0] / 100:.2f} per step ({run['start_up_points']} start-up points)")
counter = [0]
from nalib.rungekutta import named_step
ivp.integrate(lambda t, y: ivp.as_state(y) - t ** 2 + 1.0, 0.0, 0.5, 2.0, 100,
              named_step("rk4"), counter)
print(f"RK4, 100 steps: {counter[0]} evaluations, {counter[0] / 100:.2f} per step")
```

### 1.3 The root condition, and what a root outside does

Set $h = 0$ in $\sum_j \alpha_j y_{n+1-j} = h\sum_j\beta_jf_{n+1-j}$ to get the homogeneous
recurrence $\sum_j\alpha_jy_{n+1-j} = 0$, whose solutions are spanned by $z^n$ for the roots $z$ of

$$
\rho(z) = \sum_{j=0}^{k}\alpha_jz^{k-j}.
$$

**The root condition:** every root has $\lvert z\rvert \le 1$, and every root with $\lvert z\rvert
= 1$ is simple.

A root with $\lvert z\rvert > 1$ contributes a solution component growing like $\lvert z\rvert^n$.
Since $n = (T - t_0)/h$, refining the step **increases** $n$ and hence the growth, so the
divergence gets worse the harder you work. A repeated root on the circle gives $n\lvert z\rvert^n$,
which also grows, just polynomially.

None of this depends on the differential equation. It is the method's own dynamics, present even
for $y' = 0$.

### 1.4 The two barriers

**First** (this lesson). A zero-stable $k$ step method has order at most $k+2$ for $k$ even and
$k+1$ for $k$ odd. Checked here by search over the coefficient space.

**Second** (lesson 72). An A-stable linear multistep method has order at most 2, and the trapezoid
rule attains it with the smallest error constant. Checked there, from each method's own
coefficients.

The first is about **convergence** and needs no stability region. The second is about **stiffness**
and is the reason the whole stiff solver literature exists.

### 1.5 Starting values

A $k$ step method's formula involves $y_{n+1}, y_n, \dots, y_{n+1-k}$, so it cannot be applied at
all until $k$ consecutive values exist. The problem supplies one, $y_0$.

The other $k-1$ have to come from a **one step** method, which is the only kind that can run from a
single value. Exercise 8's measurement is that the starter's order matters: the whole run achieves
$\min(p, q+1)$ where $q$ is the starter's order.

```python
def f(t, y):
    return ivp.as_state(y) - t ** 2 + 1.0

def exact(t):
    return (t + 1.0) ** 2 - 0.5 * np.exp(t)

for name in ("ab2", "ab3", "ab4"):
    alpha, beta, order = ms.method(name)
    run = ms.solve_explicit(f, 0.0, 0.5, 2.0, 100, name)
    print(f"{name}: {max(len(alpha), len(beta)) - 1} step method, needs "
          f"{run['start_up_points']} starting values")
```

### 2.1 Deriving AB3

Interpolate $f$ at $t_n, t_{n-1}, t_{n-2}$, which in the local variable $s = (t - t_n)/h$ are at
$s = 0, -1, -2$. The Lagrange basis is

$$
L_0(s) = \frac{(s+1)(s+2)}{2}, \quad
L_1(s) = -s(s+2), \quad
L_2(s) = \frac{s(s+1)}{2}.
$$

The weights are $h\int_0^1 L_j(s)\,ds$:

$$
\int_0^1 \frac{(s+1)(s+2)}{2}ds = \frac{1}{2}\left[\frac{s^3}{3} + \frac{3s^2}{2} + 2s\right]_0^1
= \frac{1}{2}\cdot\frac{23}{6} = \frac{23}{12},
$$
$$
\int_0^1 -s(s+2)\,ds = -\left[\frac{s^3}{3} + s^2\right]_0^1 = -\frac43,
$$
$$
\int_0^1 \frac{s(s+1)}{2}ds = \frac12\left[\frac{s^3}{3} + \frac{s^2}{2}\right]_0^1
= \frac{5}{12}.
$$

So

$$
y_{n+1} = y_n + \frac{h}{12}\left(23f_n - 16f_{n-1} + 5f_{n-2}\right),
$$

matching the library exactly.

```python
from fractions import Fraction
import sympy as sp

s = sp.symbols("s")
basis = [sp.Rational(1, 2) * (s + 1) * (s + 2), -s * (s + 2), sp.Rational(1, 2) * s * (s + 1)]
derived = [sp.integrate(b, (s, 0, 1)) for b in basis]
print("derived by hand:", [str(v) for v in derived])
print("nalib gives:    ", [str(v) for v in ms.adams_bashforth_coefficients(3)])
assert [Fraction(str(v)) for v in derived] == list(ms.adams_bashforth_coefficients(3))
```

### 2.2 The order conditions from exactness on $t^m$

A method has order $p$ if it is exact whenever the solution is a polynomial of degree $\le p$.
Substitute $y(t) = t^m$, so $f = y' = mt^{m-1}$, and evaluate at $t_{n+1-j} = t_n + (1-j)h$. Take
$t_n = 0$ without loss:

$$
\sum_j \alpha_j\left((1-j)h\right)^m = h\sum_j\beta_j\, m\left((1-j)h\right)^{m-1}.
$$

Divide by $h^m$:

$$
\sum_j \alpha_j(1-j)^m = m\sum_j\beta_j(1-j)^{m-1}, \qquad m = 1, \dots, p,
$$

and at $m = 0$ the right side is empty, giving $\sum_j\alpha_j = 0$.

**That $m = 0$ condition is consistency of order zero**, and it is what makes $z = 1$ a root of
$\rho$: $\rho(1) = \sum_j\alpha_j = 0$. So every consistent method has the root at 1, which is the
one that carries the actual solution.

```python
for name in ("ab4", "am4", "bdf3", "milne simpson"):
    alpha, beta, order = ms.method(name)
    roots = np.asarray(ms.root_condition(alpha)["roots"], dtype=complex)
    near_one = min(abs(roots - 1.0)) if roots.size else float("nan")
    print(f"{name:>16}: sum of alpha = {float(np.sum(alpha)):+.2e}, "
          f"closest root to 1 is {near_one:.2e} away")
```

### 2.3 The unstable method

The method $y_{n+1} = -4y_n + 5y_{n-1} + h(4f_n + 2f_{n-1})$ has

$$
\alpha = (1, 4, -5), \qquad \beta = (0, 4, 2).
$$

**Order.** Check $\sum_j\alpha_j(1-j)^m = m\sum_j\beta_j(1-j)^{m-1}$ with the offsets
$1-j = 1, 0, -1$:

| $m$ | left | right |
|---|---|---|
| 0 | $1 + 4 - 5 = 0$ | 0 |
| 1 | $1 + 0 + 5 = 6$ | $1\cdot(0 + 4 + 2) = 6$ |
| 2 | $1 + 0 - 5 = -4$ | $2(0 + 0 - 2) = -4$ |
| 3 | $1 + 0 + 5 = 6$ | $3(0 + 0 + 2) = 6$ |
| 4 | $1 + 0 - 5 = -4$ | $4(0 + 0 - 2) = -8$ |

It matches through $m = 3$ and fails at $m = 4$, so **the order is 3**.

**Roots.** $\rho(z) = z^2 + 4z - 5 = (z-1)(z+5)$, so the roots are $1$ and $-5$. The second is far
outside the unit circle and the root condition fails.

```python
alpha, beta, order = ms.method("unstable order two")
print(f"alpha = {[str(v) for v in alpha]}, beta = {[str(v) for v in beta]}")
out = ms.order_conditions(alpha, beta, up_to=5)
print(f"{'m':>4}{'left':>12}{'right':>12}  holds")
for m, lo, hi, ok in zip(out["power"], out["left"], out["right"], out["holds"]):
    print(f"{m:>4}{lo:>12.4f}{hi:>12.4f}  {ok}")
print(f"order {out['order']}")
r = ms.root_condition(alpha)
print(f"roots {np.round(np.real_if_close(r['roots']), 6)}, "
      f"zero stable {r['zero_stable']}")
```

The library's name for it, `unstable order two`, understates it: the method is order **three**,
which is what makes it a genuinely tempting mistake to make.

### 2.4 The implicit error constant, and the ratio at order 4

The error constant of a linear multistep method is

$$
C_{p+1} = \frac{1}{(p+1)!}\left[\sum_j\alpha_j(1-j)^{p+1}
- (p+1)\sum_j\beta_j(1-j)^{p}\right],
$$

and the local truncation error is $C_{p+1}h^{p+1}y^{(p+1)}$.

For the Adams families, AB$k$ has order $k$ and AM$k$ has order $k$ using one fewer past point, and
the standard values are

$$
C_{\text{AB}} : \tfrac12, \tfrac{5}{12}, \tfrac38, \tfrac{251}{720}, \dots \qquad
C_{\text{AM}} : -\tfrac{1}{12}, -\tfrac{1}{24}, -\tfrac{19}{720}, -\tfrac{3}{160}, \dots
$$

```python
out = ms.implicit_error_constant()
print(f"{'order':>7}{'Adams-Bashforth':>20}{'Adams-Moulton':>18}{'ratio':>10}")
for p, ab, am, r in zip(out["order"], out["adams_bashforth"], out["adams_moulton"],
                        out["ratio"]):
    print(f"{p:>7}{float(ab):>20.8f}{float(am):>18.8f}{r:>10.4f}")
print(f"\nat order 4 the ratio is "
      f"{float(out['ratio'][list(out['order']).index(4)]):.4f}")
```

**At order 4 the implicit constant is 13.2 times smaller.** The reason is geometric: AB
extrapolates the interpolating polynomial beyond its data and AM interpolates within it, and
extrapolation is always the worse of the two.

### 2.5 BDF2 by differentiation instead of integration

The Adams families integrate an interpolant of $f$. The **BDF** family differentiates an
interpolant of $y$.

Fit the quadratic through $(t_{n-1}, y_{n-1})$, $(t_n, y_n)$, $(t_{n+1}, y_{n+1})$ and demand that
its derivative at $t_{n+1}$ equal $f_{n+1}$. In the local variable $s = (t - t_{n+1})/h$, with
nodes at $s = 0, -1, -2$,

$$
P(s) = y_{n+1}\frac{(s+1)(s+2)}{2} - y_n\, s(s+2) + y_{n-1}\frac{s(s+1)}{2},
$$

$$
P'(0) = y_{n+1}\cdot\frac{3}{2} - y_n\cdot 2 + y_{n-1}\cdot\frac{1}{2} = h f_{n+1},
$$

which after dividing by $3/2$ is

$$
y_{n+1} - \frac43 y_n + \frac13 y_{n-1} = \frac23 h f_{n+1}.
$$

```python
alpha, beta = ms.bdf_coefficients(2)
print(f"derived: alpha = [1, -4/3, 1/3], beta = [2/3, 0, 0]")
print(f"nalib:   alpha = {[str(v) for v in alpha]}, beta = {[str(v) for v in beta]}")
named_alpha, named_beta, named_order = ms.method("bdf2")
print(f"table:   alpha = {[str(v) for v in named_alpha]}, "
      f"beta = {[str(v) for v in named_beta]}, order {named_order}")
```

**Why the family is built that way.** Every $\beta_j$ except $\beta_0$ is zero, so the method is
implicit in exactly one evaluation of $f$ and nothing else. That structure is what makes the
stability region enormous: the growth polynomial is $\rho(w) - z\sigma(w)$ with
$\sigma(w) = \beta_0 w^k$, so as $z \to -\infty$ the roots go to the roots of $\sigma$, which are
all at 0. **All the growth factors tend to zero**, which is exactly L-stability, and it is why BDF
methods and not Adams methods are what stiff solvers use.

### 3.1 Adams-Bashforth of any order

```python
def my_ab(f, t0, y0, t_end, steps, k, starter="rk4"):
    """AB-k from the coefficient generator, started with a Runge-Kutta method."""
    coefficients = [float(v) for v in ms.adams_bashforth_coefficients(k)]
    h = (float(t_end) - float(t0)) / int(steps)
    step = named_step(starter)
    t = float(t0)
    y = ivp.as_state(y0)
    history = []
    ts, ys = [t], [y]
    for _ in range(k - 1):
        history.insert(0, ivp.as_state(f(t, y)))
        y = ivp.as_state(step(f, t, y, h))
        t += h
        ts.append(t)
        ys.append(y)
    for _ in range(int(steps) - (k - 1)):
        history.insert(0, ivp.as_state(f(t, y)))
        del history[k:]
        y = y + h * sum(c * fv for c, fv in zip(coefficients, history))
        t += h
        ts.append(t)
        ys.append(y)
    return np.asarray(ts), np.stack(ys)

print(f"{'k':>4}{'fitted order':>15}{'error at 640 steps':>22}")
for k in range(1, 7):
    errors, counts = [], [80, 160, 320, 640]
    for n in counts:
        ts, ys = my_ab(f, 0.0, 0.5, 2.0, n, k)
        errors.append(abs(float(ys[-1, 0]) - exact(2.0)))
    order = float(-np.polyfit(np.log(counts), np.log(errors), 1)[0])
    print(f"{k:>4}{order:>15.4f}{errors[-1]:>22.4e}")
```

AB1 through AB5 reach their orders, using an RK4 starter, which is enough for every $k$ up to 5
by the $\min(p, q+1)$ rule of exercise 4.2.

**AB6 reads 4.78 and that is not the method's fault.** Its error at 640 steps is
$2.5\times10^{-13}$, which is the roundoff floor for this problem, so the last points of the fit
are measuring arithmetic rather than truncation. It is also capped at 5 by the RK4 starter, which
is the other reason not to expect 6 from that row.

### 3.2 PECE with a variable number of corrector sweeps

```python
print(f"{'sweeps':>8}{'evaluations':>14}{'error':>14}{'error per evaluation':>24}")
for sweeps in (0, 1, 2, 3, 5):
    counter = [0]
    if sweeps == 0:
        run = ms.solve_explicit(f, 0.0, 0.5, 2.0, 200, "ab4", _counter=counter)
    else:
        run = ms.solve_predictor_corrector(f, 0.0, 0.5, 2.0, 200, "ab4", "am4",
                                           sweeps=sweeps, _counter=counter)
    err = abs(float(run["y"][-1, 0]) - exact(2.0))
    print(f"{sweeps:>8}{counter[0]:>14}{err:>14.4e}{err * counter[0]:>24.4e}")
```

**The first sweep is worth a lot and the rest are worth nothing.** Going from 0 to 1 sweep improves
the error by a large factor for one extra evaluation per step; going from 1 to 5 changes almost
nothing while costing four more.

The reason is that the corrector is being iterated towards the solution of the implicit AM4
equation, and one sweep from a fourth order predictor is already within $O(h^5)$ of it. Further
sweeps converge to a quantity that is itself only $O(h^5)$ away from the answer, so they refine
something already below the discretisation error.

### 3.3 Milne's device

The predictor and corrector have known error constants, $C_P$ and $C_C$, both multiplying
$h^{p+1}y^{(p+1)}$. Subtracting the two answers eliminates the unknown derivative:

$$
y^C_{n+1} - y^P_{n+1} \approx (C_C - C_P)h^{p+1}y^{(p+1)}
\quad\Longrightarrow\quad
\text{local error of } y^C \approx \frac{C_C}{C_C - C_P}\left(y^C_{n+1} - y^P_{n+1}\right).
$$

For AB4 and AM4, $C_P = \tfrac{251}{720}$ and $C_C = -\tfrac{19}{720}$, so the factor is
$-19/270 = -0.0704$.

```python
def milne_estimate(f, t0, y0, t_end, steps):
    """PECE with Milne's error estimate, reported against the true local error."""
    predictor = [float(v) for v in ms.adams_bashforth_coefficients(4)]
    corrector = [float(v) for v in ms.adams_moulton_coefficients(3)]
    h = (float(t_end) - float(t0)) / int(steps)
    factor = -19.0 / 270.0
    start_t, start_y = ms.start_with(f, t0, y0, h, 4, named_step("rk4"))
    ts = list(start_t)
    ys = [ivp.as_state(v) for v in start_y]
    history = [ivp.as_state(f(ts[i], ys[i])) for i in range(3, -1, -1)]
    rows = []
    for _ in range(int(steps) - 3):
        t, y = ts[-1], ys[-1]
        predicted = y + h * sum(c * fv for c, fv in zip(predictor, history[:4]))
        f_new = ivp.as_state(f(t + h, predicted))
        corrected = y + h * (corrector[0] * f_new
                             + sum(c * fv for c, fv
                                   in zip(corrector[1:], history[:3])))
        estimate = abs(factor * float(corrected[0] - predicted[0]))
        # the local error: what one step costs starting from the exact solution here
        local_start = np.asarray([float(exact(t))])
        back = len(predictor)
        local_history = [np.asarray([float(exact(t - j * h))]) for j in range(back)]
        local_history = [ivp.as_state(f(t - j * h, local_history[j]))
                         for j in range(back)]
        local_pred = local_start + h * sum(c * fv for c, fv
                                           in zip(predictor, local_history))
        local_fnew = ivp.as_state(f(t + h, local_pred))
        local_corr = local_start + h * (corrector[0] * local_fnew
                                        + sum(c * fv for c, fv
                                              in zip(corrector[1:], local_history[:3])))
        truth = abs(float(local_corr[0]) - float(exact(t + h)))
        rows.append((t + h, estimate, truth))
        history.insert(0, ivp.as_state(f(t + h, corrected)))
        del history[4:]
        ts.append(t + h)
        ys.append(corrected)
    return np.asarray(ts), np.stack(ys), rows

print(f"{'steps':>8}{'median estimate':>18}{'median true':>16}{'ratio':>9}")
for n in (50, 100, 200, 400):
    ts, ys, rows = milne_estimate(f, 0.0, 0.5, 2.0, n)
    est = np.median([r[1] for r in rows])
    tru = np.median([r[2] for r in rows])
    print(f"{n:>8}{est:>18.4e}{tru:>16.4e}{est / max(tru, 1e-300):>9.3f}")
```

The estimate tracks the true local error to within a factor of order 1 and both fall like $h^5$,
which is what makes Milne's device usable as a step controller.

**What stops it being used more.** Changing the step invalidates every coefficient, since they were
derived for equally spaced past points. So an adaptive multistep code has to either interpolate the
history onto a new grid or re-derive the coefficients for the actual spacing, which is exercise
3.4, and both are substantially more machinery than an embedded Runge-Kutta pair needs.

### 3.4 Variable step Adams coefficients

Re-derive the AB weights for arbitrary past points $t_n > t_{n-1} > \dots$ by integrating the
Lagrange basis on the actual nodes:

$$
y_{n+1} = y_n + \sum_j \left(\int_{t_n}^{t_{n+1}}L_j(t)\,dt\right) f_{n-j}.
$$

```python
from nalib.gaussquad import rule_on

def variable_ab_weights(nodes, a, b, order=8):
    """Integrate the Lagrange basis on arbitrary nodes over [a, b]."""
    x = np.asarray(nodes, dtype=float)
    points, quad = rule_on(order, float(a), float(b))
    weights = np.empty(x.size)
    for j in range(x.size):
        basis = np.ones_like(points)
        for m in range(x.size):
            if m != j:
                basis = basis * (points - x[m]) / (x[j] - x[m])
        weights[j] = float(np.sum(quad * basis))
    return weights

h = 0.1
even = np.asarray([0.0, -h, -2 * h, -3 * h])
print("equal spacing gives back the AB4 weights:")
print(f"  {np.round(variable_ab_weights(even, 0.0, h) / h, 10)}")
print(f"  {[float(v) for v in ms.adams_bashforth_coefficients(4)]}")
print()
print(f"{'spacing pattern':>26}{'weights / h':>52}{'sum':>10}")
for label, nodes in (("equal", [0.0, -h, -2 * h, -3 * h]),
                     ("last step halved", [0.0, -h / 2, -3 * h / 2, -5 * h / 2]),
                     ("last step doubled", [0.0, -2 * h, -3 * h, -4 * h]),
                     ("geometric", [0.0, -h, -3 * h, -7 * h])):
    w = variable_ab_weights(np.asarray(nodes), 0.0, h) / h
    print(f"{label:>26}{str(np.round(w, 5)):>52}{float(np.sum(w)):>10.6f}")
```

The weights sum to 1 in every case, which is consistency, and they change substantially with the
spacing. **A code that changed the step and kept the equal spacing weights would be running a
method of order 1**, because the interpolation it is implicitly doing is wrong.

```python
def variable_step_ab4(f, t0, y0, t_end, base, pattern, steps=4):
    """Adams-Bashforth with a repeating step pattern, re-deriving the weights every step."""
    t = float(t0)
    y = ivp.as_state(y0)
    step_fn = named_step("rk4")
    ts, ys = [t], [y]
    for i in range(steps - 1):
        h = base * pattern[i % len(pattern)]
        y = ivp.as_state(step_fn(f, t, y, h))
        t += h
        ts.append(t)
        ys.append(y)
    i = steps - 1
    while t < t_end - 1e-12:
        h = min(base * pattern[i % len(pattern)], t_end - t)
        nodes = np.asarray([ts[-1 - j] - t for j in range(steps)])
        w = variable_ab_weights(nodes, 0.0, h)
        history = [ivp.as_state(f(ts[-1 - j], ys[-1 - j])) for j in range(steps)]
        y = y + sum(w[j] * history[j] for j in range(steps))
        t += h
        ts.append(t)
        ys.append(y)
        i += 1
    return np.asarray(ts), np.stack(ys)

print(f"{'pattern':>22}{'steps':>8}{'error':>14}")
for label, pattern in (("uniform", [1.0]), ("alternating 0.5, 1.5", [0.5, 1.5]),
                       ("cycling 1, 2, 0.5", [1.0, 2.0, 0.5])):
    ts, ys = variable_step_ab4(f, 0.0, 0.5, 2.0, 0.005, pattern)
    print(f"{label:>22}{len(ts) - 1:>8}"
          f"{abs(float(ys[-1, 0]) - exact(2.0)):>14.4e}")
```

All three reach comparable accuracy, which is what re-deriving the weights buys. The cost is the
weight computation every step, which for four nodes is small and for a twelfth order method is not.

### 3.5 The BDF family and where it stops

```python
from nalib import stability as st

print(f"{'k':>4}{'order':>7}{'zero stable':>14}{'largest |root|':>17}"
      f"{'A(alpha) angle':>17}")
for k in range(1, 8):
    alpha, beta = ms.bdf_coefficients(k)
    root = ms.root_condition(alpha)
    order = ms.order_conditions(alpha, beta, up_to=k + 2)["order"]
    angle = st.stability_angle(alpha, beta)
    print(f"{k:>4}{order:>7}{str(root['zero_stable']):>14}"
          f"{root['largest_modulus']:>17.6f}{angle:>17.3f}")
```

BDF1 through BDF6 are zero stable and BDF7 is not: its largest root has modulus greater than 1, so
it fails the root condition and diverges on **every** problem, including $y' = 0$.

The angle column shows the second failure happening first. A(alpha) stability shrinks from 90
degrees to 18 by order 6, so BDF6 is already only usable when the eigenvalues sit within 18 degrees
of the negative real axis. **Two separate walls, and the useful range ends at the first of them.**

### 4.1 The parasitic growth rate

```python
out = ms.unstable_method_diverges()
alpha, beta, order = ms.method("unstable order two")
at_zero = max(abs(np.asarray(ms.root_condition(alpha)["roots"], dtype=complex)))
print(f"at h = 0 the parasitic root has modulus {at_zero:.4f}")
print(f"the problem is y' = -y on [0, 1], so z_h = -1/n\n")
print(f"{'steps':>8}{'|root| at z_h':>16}{'final value':>16}"
      f"{'|root|^n':>16}{'ratio':>12}")
for n, v in zip(out["steps"], out["final_value"]):
    zh = -1.0 / n
    coefficients = [float(a) - zh * float(b) for a, b in zip(alpha, beta)]
    roots = np.roots(coefficients)
    parasitic = float(np.max(np.abs(roots)))
    print(f"{n:>8}{parasitic:>16.6f}{abs(v):>16.4e}{parasitic ** n:>16.4e}"
          f"{abs(v) / parasitic ** n:>12.4e}")
ratios = [abs(v) / (np.max(np.abs(np.roots([float(a) + float(b) / n
                                            for a, b in zip(alpha, beta)]))) ** n)
          for n, v in zip(out["steps"], out["final_value"])]
counts = np.asarray(out["steps"], dtype=float)
print(f"\nthe leftover constant falls with fitted exponent "
      f"{float(-np.polyfit(np.log(counts), np.log(ratios), 1)[0]):.3f} in the step count")
```

The parasitic root moves with $h$: 5.305 at ten steps down towards 5 as $h \to 0$. Using the
$h = 0$ value of 5 instead changes the predicted growth by a constant factor of 1.8 and not by
anything that grows, so **either root recovers the shape**; the moving one gets the constant right
as well.

The leftover column is the interesting part. It falls with a fitted exponent of **4.0 in the step
count**, which is $O(h^4)$, which is exactly RK4's global order. That identifies what the constant
is: **the size of the component the starter injected into the parasitic mode**, which is the
starter's own error.

So the divergence is the product of two things, a starting error that shrinks like $h^4$ and an
amplification that grows like $5^{1/h}$, and the second wins by an enormous margin. **No starter can
make the first zero**, because the starting values come from a different method and there is no
reason for them to lie in the span of the principal mode. Refining the step shrinks the seed and
increases the exponent, and the exponent wins.

### 4.2 The starter's order for AB2 through AB6

```python
print(f"{'method':>7}{'p':>4}" + "".join(f"{'q=' + str(q):>12}" for q in (1, 2, 3, 4)))
for name in ("ab2", "ab3", "ab4"):
    alpha, beta, p = ms.method(name)
    out = ms.starting_values_matter(f, exact, 0.0, 0.5, 2.0, name=name)
    fitted = {int(q): float(v) for q, v in zip(out["starter_order"],
                                               out["fitted_order"])}
    print(f"{name:>7}{p:>4}"
          + "".join(f"{fitted.get(q, float('nan')):>12.3f}" for q in (1, 2, 3, 4)))
print()
print("predicted min(p, q+1):")
print(f"{'method':>7}{'p':>4}" + "".join(f"{'q=' + str(q):>12}" for q in (1, 2, 3, 4)))
for name, p in (("ab2", 2), ("ab3", 3), ("ab4", 4)):
    print(f"{name:>7}{p:>4}" + "".join(f"{min(p, q + 1):>12}" for q in (1, 2, 3, 4)))
```

Every entry matches $\min(p, q+1)$ to within the fitting noise. **An Euler starter caps AB4 at order
2**, which is a 50 per cent loss of order for a saving of three RK4 steps out of a hundred, and the
run reports nothing unusual.

### 4.3 Milne-Simpson's weak instability

Milne-Simpson is $y_{n+1} = y_{n-1} + \tfrac{h}{3}(f_{n+1} + 4f_n + f_{n-1})$, which is Simpson's
rule over two steps. Its $\rho(z) = z^2 - 1$ has roots $1$ and $-1$, both simple and both **on**
the circle, so it passes the root condition and is zero stable.

The root at $-1$ is the problem. It does not grow when $h = 0$, and for $h > 0$ it moves, and the
direction it moves depends on the sign of $\lambda$.

```python
def milne_simpson(lam, t_end, steps):
    """Milne-Simpson on y' = lam y, solved exactly for the implicit stage."""
    h = float(t_end) / int(steps)
    y = [1.0, float(np.exp(lam * h))]
    for _ in range(int(steps) - 1):
        rhs = y[-2] + h / 3.0 * (4.0 * lam * y[-1] + lam * y[-2])
        y.append(rhs / (1.0 - h * lam / 3.0))
    return np.asarray(y)

print(f"{'lambda':>9}{'steps':>8}{'computed y(5)':>18}{'exact':>14}{'ratio':>12}")
for lam in (-1.0, -2.0, -5.0):
    for steps in (200, 400):
        y = milne_simpson(lam, 5.0, steps)
        want = np.exp(lam * 5.0)
        print(f"{lam:>9.1f}{steps:>8}{y[-1]:>18.6e}{want:>14.6e}"
              f"{y[-1] / want:>12.4e}")
print()
print("the parasitic root of rho(z) - z_h sigma(z) for z_h = lam h:")
for lam in (-1.0, -2.0, -5.0):
    h = 5.0 / 200
    zh = lam * h
    roots = np.roots([1.0 - zh / 3.0, -4.0 * zh / 3.0, -1.0 - zh / 3.0])
    print(f"  lam = {lam:>5.1f}: roots {np.round(roots, 6)}, "
          f"largest modulus {float(np.max(np.abs(roots))):.6f}")
```

For $\lambda < 0$ the parasitic root moves **outside** the unit circle, so it grows while the true
solution decays, and it eventually dominates. That is what "weakly stable" means: stable at $h = 0$
and unstable for every $h > 0$ on a decaying problem.

**How long it takes** depends on how far outside the root sits and how small the parasitic component
starts. Both are $O(h)$ effects, so refining the step delays the takeover without preventing it.

### 5.1 Sketching the first barrier

Dahlquist's argument works with the two generating polynomials

$$
\rho(z) = \sum_j\alpha_jz^{k-j}, \qquad \sigma(z) = \sum_j\beta_jz^{k-j},
$$

and the observation that order $p$ is equivalent to

$$
\frac{\rho(z)}{\log z} - \sigma(z) = C\,(z-1)^{p+1} + O\!\left((z-1)^{p+2}\right)
\quad\text{as } z \to 1 .
$$

Substitute $z = \dfrac{1+s}{1-s}$, which maps the unit disc to the left half plane and $z = 1$ to
$s = 0$. Write

$$
\hat\rho(s) = (1-s)^k\rho\!\left(\tfrac{1+s}{1-s}\right), \qquad
\hat\sigma(s) = (1-s)^k\sigma\!\left(\tfrac{1+s}{1-s}\right),
$$

both polynomials of degree $\le k$ in $s$.

**Zero stability** says every root of $\rho$ is in the closed disc with those on the boundary
simple, which in $s$ says every root of $\hat\rho$ is in the closed left half plane with those on
the imaginary axis simple. That forces $\hat\rho$ to have **non negative coefficients** up to a
common sign.

**Order** translates into $\hat\rho(s)/\hat\sigma(s)$ agreeing with
$\log\frac{1+s}{1-s} = 2(s + \tfrac{s^3}{3} + \tfrac{s^5}{5} + \dots)$ to order $p+1$ at $s = 0$.
That expansion has **only odd powers**, and the parity is where the barrier comes from: matching an
odd function with a ratio of two degree $k$ polynomials, subject to the sign constraint from
stability, runs out of room at $p = k+1$ for odd $k$ and $p = k+2$ for even $k$.

```python
out = ms.first_dahlquist_barrier()
print(f"{'k':>4}{'parity':>9}{'max order':>12}{'k + 1':>8}{'k + 2':>8}{'matches':>10}")
for k, p in zip(out["steps"], out["maximum_order"]):
    parity = "even" if k % 2 == 0 else "odd"
    predicted = k + 2 if k % 2 == 0 else k + 1
    print(f"{k:>4}{parity:>9}{p:>12}{k + 1:>8}{k + 2:>8}"
          f"{str(p == predicted):>10}")
```

The parity is not an accident of the proof; it is the parity of the logarithm's series.

### 5.2 Why Milne-Simpson is used anyway

Milne-Simpson has the smallest error constant of any two step fourth order method, and exercise 4.3
shows its parasitic root outside the circle for $\lambda < 0$. Both are true, and it is still used,
because in a **predictor-corrector pair** the parasitic mode is suppressed.

The mechanism: a PECE pair does not solve the corrector's implicit equation. It applies the
corrector **once**, from a predictor's value, so the composite method's characteristic polynomial is
not the corrector's. It is a polynomial built from both, and the parasitic root of the pair sits
strictly inside the circle where the corrector's alone does not.

Work the algebra on $y' = \lambda y$ with $z = \lambda h$. The AB2 predictor gives

$$
y^{*}_{n+1} = y_n(1 + \tfrac32 z) - \tfrac12 z\, y_{n-1},
$$

and one Milne-Simpson correction from it gives

$$
y_{n+1} = y_{n-1} + \tfrac{z}{3}\left(y^{*}_{n+1} + 4y_n + y_{n-1}\right)
= y_n\,\tfrac{z}{3}\left(5 + \tfrac32 z\right)
+ y_{n-1}\left[1 + \tfrac{z}{3}\left(1 - \tfrac12 z\right)\right],
$$

so the pair's characteristic polynomial is

$$
w^2 - \tfrac{z}{3}\left(5 + \tfrac32 z\right)w
- \left[1 + \tfrac{z}{3}\left(1 - \tfrac12 z\right)\right] = 0 .
$$

```python
print(f"{'lam h':>9}{'corrector solved exactly':>28}{'PECE once':>24}"
      f"{'|root| exact':>14}{'|root| PECE':>14}")
for zh in (-0.01, -0.05, -0.1, -0.2, -0.5):
    exact_roots = np.roots([1.0 - zh / 3.0, -4.0 * zh / 3.0, -1.0 - zh / 3.0])
    pece = np.roots([1.0, -zh / 3.0 * (5.0 + 1.5 * zh),
                     -(1.0 + zh / 3.0 * (1.0 - 0.5 * zh))])
    print(f"{zh:>9.3f}{str(np.round(np.real_if_close(exact_roots), 5)):>28}"
          f"{str(np.round(np.real_if_close(pece), 5)):>24}"
          f"{float(np.max(np.abs(exact_roots))):>14.6f}"
          f"{float(np.max(np.abs(pece))):>14.6f}")
```

Compare the last two columns, and the exercise's premise does not survive the measurement.

**The PECE pair's parasitic root is further outside the circle than the corrector's, not inside
it**: 1.0067 against 1.0033 at $\lambda h = -0.01$, and 1.312 against 1.178 at $-0.5$. Pairing an
AB2 predictor with the Milne-Simpson corrector makes the weak instability **worse**, roughly
doubling the excess over 1 at every step size tried.

The reason is visible in the algebra above. The pair is a genuinely different method with its own
characteristic polynomial, and there is no principle saying that polynomial's roots are better
behaved; here the predictor's own $-\tfrac12 z$ term feeds the alternating mode rather than damping
it.

So the honest answer to "how does the pairing suppress the parasitic mode" is that **this pairing
does not**. Suppression is available, and it comes from the specific classical pairing (a fourth
order Adams-Bashforth predictor, giving a four step composite) together with the mode being
re-seeded from the predictor each step rather than compounding freely. Which pairings work is a
calculation to be done for each one, not a property of predictor-corrector schemes in general.

Either way it is why Milne-Simpson is a historical curiosity rather than a workhorse.
Adams-Moulton has a slightly worse error constant and no parasitic root to manage at all, and that
trade has gone one way for fifty years.

### 5.3 Variable step multistep methods

Changing the step invalidates the coefficients, because they were derived by integrating a
polynomial through equally spaced nodes. Two standard fixes.

**Fixed leading coefficient / interpolation.** Keep the equal spacing coefficients and manufacture
the history the method expects, by interpolating the stored solution onto a uniform grid of the new
step. Cheap, and it introduces an interpolation error that has to be kept below the method's own.

**Variable coefficient.** Re-derive the weights for the actual spacing at every step, which is
exercise 3.4. Exact, and it costs a small linear solve per step, plus the coefficients change every
step so nothing can be precomputed.

**What each does to the stability analysis** is the real cost, and it is the same for both. Every
result in this lesson, the root condition included, assumes a **fixed** step: $\rho$ and $\sigma$
are constants and the recurrence is linear with constant coefficients. With a varying step the
recurrence has varying coefficients and there is no characteristic polynomial at all.

The available theory is much weaker. Zero stability survives if the step ratios are bounded, with
the bound depending on $k$: for variable step BDF the classical result is that stability holds for
step ratios below about 1.29 at order 5 and below 1.13 at order 6. **The higher the order, the less
the step is allowed to change**, which is a constraint Runge-Kutta methods do not have at all.

```python
print("classical stability bounds on the step ratio for variable step BDF:")
for k, bound in ((1, float("inf")), (2, float("inf")), (3, 1.199),
                 (4, 1.155), (5, 1.078), (6, 1.036)):
    text = "unrestricted" if not np.isfinite(bound) else f"{bound:.3f}"
    print(f"  BDF{k}: {text}")
print()
print("a Runge-Kutta method has no such restriction: every step is independent")
```

That is the deepest reason the field settled on Runge-Kutta for general purpose work and kept
multistep methods for the cases where one evaluation per step is decisive: large stiff systems where
each evaluation is a Jacobian factorisation, which is where `CVODE` and `LSODA` live.

---

## Lesson 72, Stability and Stiff Equations

### 1.1 Three growth functions

Apply each method to $y' = \lambda y$ and write $z = \lambda h$.

**Euler.** $y_{n+1} = y_n + h\lambda y_n$, so

$$
R(z) = 1 + z .
$$

**Backward Euler.** $y_{n+1} = y_n + h\lambda y_{n+1}$, so $(1 - z)y_{n+1} = y_n$ and

$$
R(z) = \frac{1}{1 - z}.
$$

**Trapezoid.** $y_{n+1} = y_n + \tfrac{h}{2}\lambda(y_n + y_{n+1})$, so

$$
R(z) = \frac{1 + z/2}{1 - z/2}.
$$

```python
import numpy as np
from nalib import stability as st

print(f"{'z':>8}{'1 + z':>14}{'euler':>14}{'1/(1-z)':>14}{'backward':>14}")
for z in (-0.5, -2.0, -10.0):
    print(f"{z:>8.1f}{1.0 + z:>14.8f}{complex(st.growth_function('euler')(z)).real:>14.8f}"
          f"{1.0 / (1.0 - z):>14.8f}"
          f"{complex(st.growth_function('backward euler')(z)).real:>14.8f}")
print()
print(f"{'z':>8}{'(1+z/2)/(1-z/2)':>20}{'trapezoid':>16}")
for z in (-0.5, -2.0, -10.0, -1e6):
    got = complex(st.growth_function("trapezoid")(z)).real
    print(f"{z:>8.1e}{(1.0 + 0.5 * z) / (1.0 - 0.5 * z):>20.10f}{got:>16.10f}")
```

The shapes are the point. Euler's is a **polynomial** and the other two are **rational**, and every
difference in this lesson follows from that.

### 1.2 Why no explicit method is A-stable

An explicit $s$ stage Runge-Kutta method has $R(z) = 1 + zb^{\mathsf T}(I - zA)^{-1}\mathbf 1$ with
$A$ strictly lower triangular, hence nilpotent, so the inverse is a finite sum and **$R$ is a
polynomial of degree at most $s$** (lesson 69 exercise 2.5).

A non constant polynomial satisfies $\lvert R(z)\rvert \to \infty$ as $\lvert z\rvert \to \infty$
in every direction, so $\{z : \lvert R(z)\rvert \le 1\}$ is a **bounded** set. The left half plane is
unbounded. Therefore no explicit method's region contains it.

The same argument covers explicit multistep methods: their growth factors are the roots of
$\rho(w) - z\sigma(w)$ with $\sigma$ of degree $k-1$ (since $\beta_0 = 0$), and the leading
behaviour as $z \to -\infty$ makes at least one root grow without bound.

```python
print(f"{'method':>16}{'|R(-10)|':>14}{'|R(-100)|':>14}{'|R(-1e6)|':>14}")
for name in ("euler", "heun", "rk4"):
    R = st.growth_function(name)
    print(f"{name:>16}" + "".join(f"{abs(complex(R(z))):>14.3e}"
                                  for z in (-10.0, -100.0, -1e6)))
for name in ("backward euler", "trapezoid", "bdf2"):
    R = st.growth_function(name)
    print(f"{name:>16}" + "".join(f"{abs(complex(R(z))):>14.3e}"
                                  for z in (-10.0, -100.0, -1e6)))
```

### 1.3 A-stable and not L-stable

**A-stable:** the region of absolute stability contains the whole left half plane, so
$\lvert R(z)\rvert \le 1$ whenever $\operatorname{Re} z \le 0$.

**L-stable:** A-stable **and** $R(z) \to 0$ as $\operatorname{Re}z \to -\infty$.

The **trapezoid rule** is the standard example of the first without the second:
$R(z) = (1+z/2)/(1-z/2) \to -1$, so its modulus is 1 in the limit rather than 0. A fast mode is
neither amplified nor damped; it alternates in sign forever, which section 6 measures.

```python
out = st.classify()
print(f"{'method':>18}{'A-stable':>10}{'L-stable':>10}{'|R| at -1e6':>14}"
      f"{'|R| at -1e10':>15}")
for n, a, l, m6, m10 in zip(out["names"], out["a_stable"], out["l_stable"],
                            out["magnitude_at_minus_1e6"],
                            out["magnitude_at_minus_1e10"]):
    print(f"{n:>18}{str(a):>10}{str(l):>10}{m6:>14.3e}{m10:>15.3e}")
```

### 1.4 Stiffness is about the eigenvalues

For a linear system $y' = Ay$ the solution is a combination of $e^{\lambda_i t}v_i$ over the
eigenpairs, so the decay rates in the problem are the eigenvalues and nothing else. A change of
basis leaves the eigenvalues alone and can make the entries look like anything.

```python
print(f"{'matrix':>36}{'largest entry':>16}{'ratio':>12}{'stiff':>8}")
examples = {
    "diag(-1, -1000)": [[-1.0, 0.0], [0.0, -1000.0]],
    "[[-500.5, 499.5], [499.5, -500.5]]": [[-500.5, 499.5], [499.5, -500.5]],
    "[[-1, 0], [999, -1000]]": [[-1.0, 0.0], [999.0, -1000.0]],
    "[[-2, 1], [1, -2]]": [[-2.0, 1.0], [1.0, -2.0]],
}
for label, A in examples.items():
    out = st.stiffness_ratio(A)
    print(f"{label:>36}{float(np.max(np.abs(A))):>16.1f}"
          f"{out['ratio']:>12.2f}{str(out['stiff']):>8}")
```

The second and third have entries of similar size and stiffness ratios of 1000. The fourth has the
same entry sizes as the third's small entries and a ratio of 3. **Reading the entries tells you
nothing.**

### 1.5 Why the waste is worst at low accuracy

Two step sizes compete: the accuracy step $h_a$, set by the tolerance, and the stability step
$h_s = z^{*}/\lvert\lambda_{\max}\rvert$, set by the fastest mode and the method's region. The
solver must use $\min(h_a, h_s)$, and the waste is $h_a/h_s$ when that is larger than 1.

$h_s$ does not depend on the tolerance at all. $h_a$ **grows** as the tolerance loosens, like
$\text{tol}^{1/p}$. So the ratio grows as the tolerance loosens, and the waste is worst exactly
when the least accuracy is wanted.

```python
out = st.the_step_is_set_by_stability(name="rk4")
print(f"stability step {out['stability_step']:.3e} (independent of the tolerance)\n")
print(f"{'tolerance':>12}{'accuracy step':>16}{'wasted factor':>16}")
for tol, h, w in zip(out["tolerance"], out["accuracy_step"], out["wasted_factor"]):
    print(f"{tol:>12.0e}{h:>16.3e}{w:>16.2f}")
```

**That is exactly the situation a stiff problem presents.** The fast mode is a transient nobody
wants resolved, so the accuracy demanded of the slow part is modest, and the waste is at its worst.

### 2.1 Euler's region as a disc

$\lvert R(z)\rvert = \lvert 1 + z\rvert < 1$ says the distance from $z$ to $-1$ is less than 1, so
the region is the **open disc of radius 1 centred at $-1$**.

Its intersection with the negative real axis is $(-2, 0)$, so the real axis limit is 2. Its
intersection with the imaginary axis is the single point $z = 0$, since $\lvert 1 + iy\rvert =
\sqrt{1+y^2} > 1$ for every $y \ne 0$, so the imaginary axis limit is 0.

```python
print(f"euler real axis limit:      {st.real_axis_limit('euler'):.6f} (exactly 2)")
print(f"euler imaginary axis limit: {st.imaginary_axis_limit('euler'):.2e} (exactly 0)")
print()
R = st.growth_function("euler")
print(f"{'y':>8}{'|R(iy)|':>16}{'sqrt(1+y^2)':>16}")
for y in (0.01, 0.1, 1.0):
    print(f"{y:>8.2f}{abs(complex(R(1j * y))):>16.10f}{np.sqrt(1 + y ** 2):>16.10f}")
```

### 2.2 The order $s$ stage $s$ polynomial, and RK4's limit

From lesson 69 exercise 2.5, an explicit $s$ stage method of order $s$ has

$$
R(z) = \sum_{m=0}^{s}\frac{z^m}{m!},
$$

because the order conditions fix $b^{\mathsf T}A^{m-1}\mathbf 1 = 1/m!$ for $m \le s$ and there are
no further terms.

The real axis limit is the negative $z$ where $\lvert R(z)\rvert = 1$, found by bisection.

```python
import math

def taylor_growth(z, s):
    return sum(z ** m / math.factorial(m) for m in range(s + 1))

def limit_of(s, lo=0.0, hi=10.0, tol=1e-14):
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if abs(taylor_growth(-mid, s)) <= 1.0:
            lo = mid
        else:
            hi = mid
    return lo

print(f"{'s':>4}{'bisected limit':>18}{'nalib':>16}{'gap':>12}")
for s, name in ((1, "euler"), (2, "heun"), (3, "kutta third"), (4, "rk4")):
    got = limit_of(s)
    print(f"{s:>4}{got:>18.8f}{st.real_axis_limit(name):>16.8f}"
          f"{abs(got - st.real_axis_limit(name)):>12.2e}")
```

RK4's limit is 2.78529356, and it is a property of the **polynomial** rather than of the tableau, so
the three eighths rule shares it exactly.

### 2.3 The trapezoid rule's limit and its ringing frequency

$$
R(z) = \frac{1 + z/2}{1 - z/2} = \frac{2 + z}{2 - z} \longrightarrow -1 \quad (z \to -\infty),
$$

and the approach is from above: for $z = -x$ with $x > 0$,

$$
R(-x) = \frac{2 - x}{2 + x} = -1 + \frac{4}{2 + x},
$$

so $R$ is negative for $x > 2$ and $\lvert R\rvert = 1 - \frac{4}{2+x} < 1$.

Since $R < 0$, the computed solution **alternates in sign every step**, so its period is $2h$ and
its frequency is $\pi/h$ radians per unit time. That is the Nyquist frequency of the grid: the
fastest oscillation a grid of spacing $h$ can represent, which is exactly what an unresolved mode
aliases onto.

```python
print(f"{'lam h':>10}{'R':>14}{'|R|':>12}{'-1 + 4/(2+x)':>16}")
for x in (2.0, 10.0, 100.0, 1000.0, 10000.0):
    R = (2.0 - x) / (2.0 + x)
    print(f"{-x:>10.0f}{R:>14.8f}{abs(R):>12.8f}{-1.0 + 4.0 / (2.0 + x):>16.8f}")
print()
out = st.trapezoid_rings(lam=-1000.0, steps=10, t_end=1.0)
signs = np.sign(out["trapezoid_values"])
print(f"signs of the computed values: {signs.astype(int)}")
print(f"they alternate: {out['trapezoid_alternates']}, "
      f"period 2h = {2.0 / 10:.3f}")
```

### 2.4 BDF2's growth factor

BDF2 is $y_{n+1} - \tfrac43 y_n + \tfrac13 y_{n-1} = \tfrac23 hf_{n+1}$, so on $y' = \lambda y$ the
characteristic equation is

$$
\left(1 - \tfrac23 z\right)w^2 - \tfrac43 w + \tfrac13 = 0,
\qquad
w = \frac{2 \pm \sqrt{4 - 3(1 - \tfrac23 z)}}{3(1 - \tfrac23 z)}
= \frac{2 \pm \sqrt{1 + 2z}}{3 - 2z}.
$$

As $z \to -\infty$ the numerator behaves like $\sqrt{2\lvert z\rvert}$ and the denominator like
$2\lvert z\rvert$, so

$$
\lvert w\rvert \sim \frac{\sqrt{2\lvert z\rvert}}{2\lvert z\rvert}
= \frac{1}{\sqrt{2\lvert z\rvert}} = O\!\left(\lvert z\rvert^{-1/2}\right),
$$

which tends to zero, so BDF2 is L-stable, and it does so **slowly**.

```python
print(f"{'z':>12}{'dominant root':>18}{'|z|^(-1/2) law':>18}{'ratio':>10}")
for z in (-1e2, -1e4, -1e6, -1e8, -1e10):
    roots = np.roots([1.0 - 2.0 * z / 3.0, -4.0 / 3.0, 1.0 / 3.0])
    dominant = float(np.max(np.abs(roots)))
    law = 1.0 / np.sqrt(2.0 * abs(z))
    print(f"{z:>12.0e}{dominant:>18.6e}{law:>18.6e}{dominant / law:>10.4f}")
print()
out = st.classify(["bdf2"])
print(f"nalib: |R| at -1e6 is {float(out['magnitude_at_minus_1e6'][0]):.3e}, "
      f"at -1e10 is {float(out['magnitude_at_minus_1e10'][0]):.3e}")
print(f"classified L-stable: {bool(out['l_stable'][0])}")
```

**At $z = -10^{10}$ the factor is still $7\times10^{-6}$**, which is why a fixed threshold at
$10^{-6}$ would call an L-stable method not L-stable, and why the test in `nalib` is "small and
still falling" instead.

### 2.5 A-stability from the imaginary axis

Suppose $R$ is a rational function with no poles in the closed left half plane. Then $R$ is
analytic there, and $\lvert R\rvert$ is subharmonic, so by the **maximum modulus principle** applied
on the left half plane (with the point at infinity handled by $\lvert R(\infty)\rvert$ being finite,
which holds when $\deg$ numerator $\le \deg$ denominator), the maximum of $\lvert R\rvert$ over the
closed left half plane is attained on its boundary, the imaginary axis.

So

$$
\lvert R(z)\rvert \le 1 \text{ on } \operatorname{Re}z \le 0
\iff
\lvert R(iy)\rvert \le 1 \text{ for all real } y, \text{ and } R \text{ analytic there}.
$$

Both halves are needed. Analyticity rules out a pole in the left half plane, which would make
$\lvert R\rvert$ unbounded near it whatever it does on the axis.

```python
print(f"{'method':>18}{'poles':>34}{'in the left half plane':>24}"
      f"{'max |R(iy)|':>14}")
cases = {
    "backward euler": ([1.0], [-1.0, 1.0]),          # 1 / (1 - z)
    "trapezoid": ([0.5, 1.0], [-0.5, 1.0]),          # (1 + z/2) / (1 - z/2)
    "a bad one": ([1.0], [1.0, 1.0]),                # 1 / (1 + z), pole at z = -1
}
for label, (num, den) in cases.items():
    poles = np.roots(den) if len(den) > 1 else np.asarray([])
    left = [p for p in poles if p.real <= 0.0]
    ys = np.linspace(-1e4, 1e4, 200001)
    values = np.polyval(num, 1j * ys) / np.polyval(den, 1j * ys)
    print(f"{label:>18}{str(np.round(poles, 4)):>34}"
          f"{str(np.round(left, 4)) if left else 'none':>24}"
          f"{float(np.max(np.abs(values))):>14.6f}")
```

The third row is the point. Its $\lvert R(iy)\rvert$ is at most 1 on the whole imaginary axis, and
it has a pole at $z = -1$, so it is not A-stable despite passing the axis test. **The axis condition
alone is not enough.**

### 3.1 Every region on one figure

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2))
for name in ("euler", "heun", "kutta third", "rk4"):
    region = st.stability_region(name, real=(-3.5, 1.0), imaginary=(-3.5, 3.5),
                                 points=241)
    left.contour(region["real"], region["imaginary"], region["inside"].astype(float),
                 levels=[0.5], linewidths=1.6)
    left.plot([], [], label=name)
left.axhline(0.0, color="k", lw=0.4); left.axvline(0.0, color="k", lw=0.4)
left.set_aspect("equal"); left.set_title("explicit: bounded"); left.legend(fontsize=8)
for name in ("backward euler", "trapezoid", "implicit midpoint", "bdf2"):
    region = st.stability_region(name, real=(-6.0, 6.0), imaginary=(-6.0, 6.0),
                                 points=241)
    right.contour(region["real"], region["imaginary"], region["inside"].astype(float),
                  levels=[0.5], linewidths=1.6)
    right.plot([], [], label=name)
right.axhline(0.0, color="k", lw=0.4); right.axvline(0.0, color="k", lw=0.4)
right.set_aspect("equal"); right.set_title("implicit: unbounded")
right.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/solutions_72_regions.png", dpi=110)
plt.close(fig)
print("saved figures/solutions_72_regions.png")
print(f"{'method':>18}{'real axis limit':>18}{'imaginary limit':>18}")
for name in st.METHODS:
    r = st.real_axis_limit(name)
    i = st.imaginary_axis_limit(name)
    print(f"{name:>18}{('unbounded' if not np.isfinite(r) else f'{r:.6f}'):>18}"
          f"{('unbounded' if not np.isfinite(i) else f'{i:.6f}'):>18}")
```

The right panel's regions are the **complements** of bounded sets: backward Euler's is everything
outside the disc of radius 1 centred at $+1$, and the trapezoid rule's is the entire left half plane
exactly.

### 3.2 Backward Euler with a Newton solve, on Hodgkin-Huxley

```python
from nalib import ivp, odesystems as od
from nalib.rungekutta import named_step

f, y0 = od.hodgkin_huxley()
print(f"{'steps':>8}{'h':>10}{'backward euler V(20)':>24}{'RK4 V(20)':>16}")
reference = ivp.integrate(f, 0.0, y0, 20.0, 200000, named_step("rk4"))
print(f"{'reference':>8}{20.0 / 200000:>10.2e}{'':>24}"
      f"{float(reference['y'][-1, 0]):>16.6f}")
for steps in (200, 500, 2000, 10000):
    with np.errstate(over="ignore", invalid="ignore"):
        be = ivp.integrate(f, 0.0, y0, 20.0, steps, ivp.backward_euler_step)
        rk = ivp.integrate(f, 0.0, y0, 20.0, steps, named_step("rk4"))
    be_v = float(be["y"][-1, 0])
    rk_v = float(rk["y"][-1, 0])
    label = "overflow" if not np.isfinite(rk_v) else f"{rk_v:.6f}"
    print(f"{steps:>8}{20.0 / steps:>10.2e}{be_v:>24.6f}{label:>16}")
```

**Backward Euler produces a finite answer at every step size and RK4 does not.** The neuron's
stiffness ratio reaches 880 (lesson 73), so RK4's limit $\lvert\lambda\rvert h < 2.785$ needs
several thousand steps and anything coarser overflows.

Backward Euler is not accurate at 200 steps; it is **stable**, which on a stiff problem is the
prerequisite for being anything at all.

### 3.3 BDF2 through BDF6, and the closing wedge

```python
from nalib import multistep as ms

print(f"{'k':>4}{'order':>7}{'zero stable':>14}{'A(alpha) angle':>17}"
      f"{'A-stable':>11}")
for k in range(1, 8):
    alpha, beta = ms.bdf_coefficients(k)
    order = ms.order_conditions(alpha, beta, up_to=k + 2)["order"]
    stable = ms.root_condition(alpha)["zero_stable"]
    angle = st.stability_angle(alpha, beta)
    print(f"{k:>4}{order:>7}{str(stable):>14}{angle:>17.3f}{str(angle > 89.9):>11}")
```

```python
fig, ax = plt.subplots()
angles = np.linspace(0.0, 2.0 * np.pi, 721)
for k in (1, 2, 3, 4, 5, 6):
    alpha, beta = ms.bdf_coefficients(k)
    boundary = []
    for theta in angles:
        w = np.exp(1j * theta)
        rho = sum(complex(a) * w ** (len(alpha) - 1 - j) for j, a in enumerate(alpha))
        sigma = sum(complex(b) * w ** (len(beta) - 1 - j) for j, b in enumerate(beta))
        boundary.append(rho / sigma)
    boundary = np.asarray(boundary)
    ax.plot(boundary.real, boundary.imag, lw=1.2, label=f"BDF{k}")
ax.axhline(0.0, color="k", lw=0.4); ax.axvline(0.0, color="k", lw=0.4)
ax.set_xlim(-6.0, 14.0); ax.set_ylim(-9.0, 9.0); ax.set_aspect("equal")
ax.set_title("BDF boundary locus: unstable region grows with the order")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/solutions_72_bdf.png", dpi=110); plt.close(fig)
print("saved figures/solutions_72_bdf.png")
```

The **boundary locus** $z = \rho(e^{i\theta})/\sigma(e^{i\theta})$ traces the edge of the stability
region. For BDF1 and BDF2 it stays in the right half plane, so the whole left half plane is stable.
From BDF3 onwards it reaches into the left half plane, carving out the wedge of instability that
grows with the order until BDF7 leaves nothing.

### 3.4 Two stage Radau IIA

Radau IIA with two stages has

$$
c = \left(\tfrac13, 1\right), \quad
A = \begin{pmatrix}\tfrac{5}{12} & -\tfrac{1}{12}\\[2pt] \tfrac34 & \tfrac14\end{pmatrix}, \quad
b = \left(\tfrac34, \tfrac14\right).
$$

Note $c_2 = 1$ and the last row of $A$ equals $b$, which is what makes it **stiffly accurate**: the
step's answer is the last stage, so $R(\infty) = 0$ automatically.

```python
from fractions import Fraction
from nalib import rungekutta as rk

RADAU_A = [[Fraction(5, 12), Fraction(-1, 12)], [Fraction(3, 4), Fraction(1, 4)]]
RADAU_B = [Fraction(3, 4), Fraction(1, 4)]
RADAU_C = [Fraction(1, 3), Fraction(1)]

out = rk.order_conditions(RADAU_A, RADAU_B, RADAU_C, up_to=4)
print(f"exact arithmetic: {out['exact']}, achieved order {out['achieved_order']}")
print(f"last row of A equals b: {RADAU_A[-1] == RADAU_B}")

A = np.asarray([[float(v) for v in row] for row in RADAU_A])
b = np.asarray([float(v) for v in RADAU_B])
eye, ones = np.eye(b.size), np.ones(b.size)
print(f"\n{'z':>12}{'R(z)':>18}{'|R(z)|':>14}")
for z in (-1.0, -10.0, -1e3, -1e6, -1e10):
    R = 1.0 + z * (b @ np.linalg.solve(eye - z * A, ones))
    print(f"{z:>12.0e}{R:>18.6e}{abs(R):>14.6e}")
ys = np.linspace(-1e4, 1e4, 40001)
worst = max(abs(1.0 + 1j * y * (b @ np.linalg.solve(eye - 1j * y * A, ones)))
            for y in ys[::40])
print(f"\nlargest |R| on the imaginary axis: {worst:.10f} (A-stable needs <= 1)")
```

**Order 3 from two stages, and L-stable.** The order conditions give 3, which is $2s - 1$ for
Radau IIA, and $R(z) \to 0$ like $\lvert z\rvert^{-1}$, far faster than BDF2's
$\lvert z\rvert^{-1/2}$.

Escaping the second Dahlquist barrier costs a coupled two stage solve per step, which is the trade
lesson 72 section 7 names.

### 3.5 A Rosenbrock method

A Rosenbrock method replaces the nonlinear stage solve by **one linear solve with the Jacobian**.
The two stage second order member is

$$
(I - \gamma h J)k_1 = h f(t_n, y_n), \qquad
(I - \gamma h J)k_2 = h f(t_n + h,\ y_n + k_1) - 2k_1,
$$
$$
y_{n+1} = y_n + \tfrac32 k_1 + \tfrac12 k_2 .
$$

Both solves use the **same matrix**, so one factorisation serves the whole step.

```python
from nalib.nlsystems import numerical_jacobian

def make_ros2(gamma):
    """ROS2 with a chosen gamma: two linear solves, one matrix, no Newton iteration."""
    def step(f, t, y, h, _counter=None):
        state = ivp.as_state(y)
        n = state.size
        J = numerical_jacobian(lambda v: ivp.as_state(f(t, v)), state)
        M = np.linalg.inv(np.eye(n) - gamma * h * J)
        k1 = M @ (h * ivp.as_state(f(t, state)))
        k2 = M @ (h * ivp.as_state(f(t + h, state + k1)) - 2.0 * k1)
        return state + 1.5 * k1 + 0.5 * k2
    return step

def mild(t, y):
    return ivp.as_state(y) - t ** 2 + 1.0

def mild_exact(t):
    return np.asarray([(t + 1.0) ** 2 - 0.5 * np.exp(t)])

for gamma, label in ((1.0 + 1.0 / np.sqrt(2.0), "1 + 1/sqrt2"),
                     (1.0 - 1.0 / np.sqrt(2.0), "1 - 1/sqrt2")):
    out = ivp.error_against_step(mild, mild_exact, 0.0, 0.5, 2.0,
                                 [10, 20, 40, 80, 160], make_ros2(gamma))
    print(f"gamma = {label:>12} ({gamma:.6f}): non stiff fitted order "
          f"{out['fitted_order']:.4f}")
```

Second order on a non stiff problem, as designed. Now the stiff one.

```python
def stiff(t, y):
    v = ivp.as_state(y)
    return np.asarray([-1000.0 * (v[0] - np.cos(t)) - np.sin(t), -v[1]])

def stiff_exact(t):
    return np.asarray([np.cos(t), np.exp(-t)])

print(f"{'steps':>8}{'lam h':>10}{'ROS2 (1.707)':>16}{'ratio':>8}"
      f"{'ROS2 (0.293)':>16}{'ratio':>8}{'backward Euler':>17}{'RK4':>14}")
previous = {}
for steps in (10, 25, 100, 400, 1600):
    h = 1.0 / steps
    row = {}
    for gamma in (1.0 + 1.0 / np.sqrt(2.0), 1.0 - 1.0 / np.sqrt(2.0)):
        run = ivp.integrate(stiff, 0.0, [1.0, 1.0], 1.0, steps, make_ros2(gamma))
        row[gamma] = float(np.max(np.abs(run["y"][-1] - stiff_exact(1.0))))
    be = ivp.integrate(stiff, 0.0, [1.0, 1.0], 1.0, steps, ivp.backward_euler_step)
    with np.errstate(over="ignore", invalid="ignore"):
        r4 = ivp.integrate(stiff, 0.0, [1.0, 1.0], 1.0, steps, named_step("rk4"))
    be_err = float(np.max(np.abs(be["y"][-1] - stiff_exact(1.0))))
    r4_err = float(np.max(np.abs(r4["y"][-1] - stiff_exact(1.0))))
    cells = []
    for gamma in (1.0 + 1.0 / np.sqrt(2.0), 1.0 - 1.0 / np.sqrt(2.0)):
        ratio = "" if gamma not in previous else f"{previous[gamma] / row[gamma]:.2f}"
        cells.append(f"{row[gamma]:>16.3e}{ratio:>8}")
        previous[gamma] = row[gamma]
    r4_text = "overflow" if not np.isfinite(r4_err) else f"{r4_err:.2e}"
    print(f"{steps:>8}{1000.0 * h:>10.1f}" + "".join(cells)
          + f"{be_err:>17.3e}{r4_text:>14}")
```

Three findings, and the third is the interesting one.

**RK4 overflows until the step falls inside its stability region**, which needs about 360 steps
here. Both Rosenbrock variants and backward Euler produce finite answers at every step size.

**Rosenbrock is not better than backward Euler on this problem**, despite being second order and
backward Euler being first. At 10 steps it is three times worse.

**And its observed order on the stiff problem is not 2.** The ratio column for
$\gamma = 1 + 1/\sqrt2$ settles around 5, which is order 1.2, and for $\gamma = 1 - 1/\sqrt2$ it
reaches 13.9, which is order 1.9. **That is order reduction**, exactly the phenomenon of exercise
5.2, showing up here rather than in the explicit method that exercise tries it on.

The reason it can be shown here at all is that a Rosenbrock method is implicit enough to run with
$\lambda h = 100$ and still return numbers, which is precisely the argument exercise 5.2 reaches.
What Rosenbrock gives up against a true implicit method is exactness of the stage equations, which
is what the reduction is: the stages are solved to $O(h^2)$ rather than exactly, and on a stiff
problem that error is not damped.

### 4.1 The wasted factor against the stiffness ratio

```python
print(f"{'stiffness ratio':>18}{'stability steps':>18}{'accuracy steps':>18}"
      f"{'wasted factor':>16}")
for fast in (-10.0, -100.0, -1000.0, -10000.0, -100000.0):
    out = st.the_step_is_set_by_stability(fast=fast, slow=-1.0, t_end=5.0,
                                          tolerances=[1e-4])
    print(f"{out['stiffness_ratio']:>18.0f}{out['steps_for_stability']:>18}"
          f"{int(out['steps_for_accuracy'][0]):>18}{out['wasted_factor'][0]:>16.2f}")
```

The wasted factor is **exactly proportional to the stiffness ratio**, and the reason is immediate:
$h_s \propto 1/\lvert\lambda_{\text{fast}}\rvert$ while $h_a$ depends only on the slow mode and the
tolerance, so their ratio scales like $\lvert\lambda_{\text{fast}}\rvert$.

That linear law is what makes stiffness a **qualitative** problem rather than a quantitative one.
A ratio of $10^6$, which is ordinary in chemical kinetics, means a million times the work.

### 4.2 Where backward Euler overtakes RK4

```python
print(f"{'stiffness':>12}{'target':>10}{'RK4 steps':>12}{'RK4 evals':>12}"
      f"{'BE steps':>11}{'BE evals':>11}{'winner':>16}")
for fast in (-5.0, -50.0, -500.0, -3000.0, -10000.0, -30000.0):
    def system(t, y, fast=fast):
        v = ivp.as_state(y)
        return np.asarray([fast * (v[0] - np.cos(t)) - np.sin(t), -v[1]])

    def truth(t):
        return np.asarray([np.cos(t), np.exp(-t)])

    target = 1e-4
    costs = {}
    for label, step, per_step in (("rk4", named_step("rk4"), 4),
                                  ("be", ivp.backward_euler_step, 6)):
        n = 4
        while n < 4_000_000:
            with np.errstate(over="ignore", invalid="ignore"):
                run = ivp.integrate(system, 0.0, [1.0, 1.0], 1.0, n, step)
            value = float(np.max(np.abs(run["y"][-1] - truth(1.0))))
            if np.isfinite(value) and value <= target:
                break
            n *= 2
        costs[label] = (n, n * per_step)
    winner = "rk4" if costs["rk4"][1] <= costs["be"][1] else "backward euler"
    print(f"{abs(fast):>12.0f}{target:>10.0e}{costs['rk4'][0]:>12}"
          f"{costs['rk4'][1]:>12}{costs['be'][0]:>11}{costs['be'][1]:>11}"
          f"{winner:>16}")
```

**Backward Euler's cost does not change with the stiffness at all**, and RK4's doubles with every
doubling of $\lvert\lambda\rvert$ once the stability limit takes over. So the crossover exists and
this sweep finds it.

The mechanism is exactly section 5's. Backward Euler is A-stable, so its step is set by accuracy
alone and the fast mode is irrelevant to it. RK4's step is $\min(h_a, h_s)$ with
$h_s = 2.785/\lvert\lambda\rvert$, so past the crossover its cost is proportional to the stiffness.

The right conclusion is not that backward Euler wins. **First order is a heavy price**: it needs
thousands of steps for four digits on a problem RK4 handles in eight when stability allows. That is
why nobody uses backward Euler for stiff problems either, and why BDF and Radau exist to have
stability and order at once.

### 4.3 The ringing amplitude against $\lambda h$

```python
print(f"{'lam h':>10}{'growth factor':>16}{'amplitude after 10':>21}"
      f"{'steps to halve':>17}")
for lam in (-4.0, -10.0, -40.0, -100.0, -1000.0, -10000.0):
    out = st.trapezoid_rings(lam=lam, steps=10, t_end=1.0)
    g = abs(out["trapezoid_growth"])
    halving = float("inf") if g >= 1.0 else np.log(0.5) / np.log(g)
    print(f"{lam / 10:>10.1f}{out['trapezoid_growth']:>16.6f}"
          f"{out['trapezoid_amplitude_at_the_end']:>21.6f}{halving:>17.1f}")
```

The number of steps to halve the spurious oscillation is $\log(1/2)/\log\lvert R\rvert$, and it
grows without bound as $\lambda h \to -\infty$: 1.7 steps at $\lambda h = -10$, 17 at $-100$, and
**173 at $-1000$.** On a genuinely stiff problem the ringing outlives the run.

Note the first two rows, which are the control. At $\lambda h = -0.4$ and $-1.0$ the growth factor
is **positive**, so there is no ringing at all, and the amplitude columns match the $-10$ and $-4$
rows exactly because $\lvert R\rvert$ is the same. The sign change happens at $\lambda h = -2$,
where $R = 0$: that is the one step size at which the trapezoid rule annihilates a mode completely.

The ringing stops being a practical problem when $\lvert R\rvert$ is far enough below 1 that a few
steps kill it, which is $\lambda h$ of order $-10$: there $\lvert R\rvert = 2/3$ and two steps cut
the amplitude by more than half.

### 5.1 Sketching the second barrier

The argument runs in the same $s = \dfrac{z-1}{z+1}$ variable as the first barrier (lesson 71
exercise 5.1). Write

$$
\hat\rho(s) = (1-s)^k\rho\!\left(\tfrac{1+s}{1-s}\right), \qquad
\hat\sigma(s) = (1-s)^k\sigma\!\left(\tfrac{1+s}{1-s}\right).
$$

**A-stability** says the stability region contains the left half plane, which after the map says
the **unit disc** in $s$: $\lvert w\rvert \le 1$ for every root $w$ of $\hat\rho(s) - z\hat\sigma(s)$
whenever $\operatorname{Re}z \le 0$. Equivalently, the rational function
$\hat\rho/\hat\sigma$ maps the unit disc into the right half plane, so it is a **positive real**
function.

**Order $p$** says $\dfrac{\hat\rho(s)}{\hat\sigma(s)}$ agrees with
$\log\dfrac{1+s}{1-s} = 2\left(s + \tfrac{s^3}{3} + \tfrac{s^5}{5} + \dots\right)$ to order $p+1$ at
$s = 0$.

A positive real function has a strong structural restriction: its Taylor coefficients at 0 satisfy
inequalities that a truncation of an odd series violates from the $s^3$ term onwards. Working
through, the highest order compatible with both is **2**, and the unique attaining method up to
scaling is $\hat\rho/\hat\sigma = 2s$, which unwinds to the trapezoid rule.

**Where the linearity is used:** the whole argument is about the two polynomials $\rho$ and
$\sigma$, and a method has such a pair only if it is a **linear** multistep method. A Runge-Kutta
method has a rational $R(z)$ of any degree, unconstrained by the positive real condition, which is
why Radau IIA reaches order 3 and Gauss reaches $2s$ while remaining A-stable.

```python
out = st.second_dahlquist_barrier()
print(f"{'method':>18}{'order':>7}{'A-stable':>11}")
for n, p, a in zip(out["names"], out["order"], out["a_stable"]):
    print(f"{n:>18}{p:>7}{str(a):>11}")
print(f"\nhighest A-stable order found among linear multistep methods: "
      f"{out['highest_a_stable_order_found']}")
print("Runge-Kutta methods, which have no rho and sigma, are not bound by it:")
print(f"  two stage Radau IIA: order 3, A-stable and L-stable")
print(f"  two stage Gauss:     order 4, A-stable (lesson 69 exercise 3.3)")
```

### 5.2 Order reduction and the stage order

The **stage order** $q$ is the largest integer with

$$
\sum_j a_{ij}c_j^{m-1} = \frac{c_i^m}{m}, \qquad m = 1, \dots, q,
$$

for every stage $i$. It measures how accurately each **internal stage** approximates
$y(t_n + c_ih)$, as opposed to how accurately the step's output approximates $y(t_{n+1})$.

**The mechanism.** On a stiff problem the stage equation is
$(I - h a_{ii}J)k_i = \dots$, so the stage error is multiplied by $(I - ha_{ii}J)^{-1}$, which for
$\lvert h\lambda\rvert$ large is $O(1/(h\lambda))$ rather than $O(1)$. The classical order analysis
assumes $h\lambda \to 0$, where the stage errors are absorbed into higher order terms; with
$h\lambda$ large and fixed they are not, and what survives is governed by $q$ rather than $p$.

The observed order on a stiff problem is typically $\min(p, q+1)$.

```python
def stage_order(A, c, up_to=6):
    """The largest m for which every stage integrates t^(m-1) exactly."""
    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=float)
    for m in range(1, up_to + 1):
        if not np.allclose(A @ (c ** (m - 1)), c ** m / m, atol=1e-12):
            return m - 1
    return up_to

print(f"{'method':>18}{'stages':>8}{'order p':>9}{'stage order q':>15}"
      f"{'min(p, q+1)':>13}")
for name in ("euler", "heun", "kutta third", "rk4"):
    A, b, c, p = rk.tableau(name)
    q = stage_order(A, c)
    print(f"{name:>18}{b.size:>8}{p:>9}{q:>15}{min(p, q + 1):>13}")
A = np.asarray([[float(v) for v in row] for row in RADAU_A])
c = np.asarray([float(v) for v in RADAU_C])
print(f"{'radau IIA (2)':>18}{2:>8}{3:>9}{stage_order(A, c):>15}"
      f"{min(3, stage_order(A, c) + 1):>13}")
```

**Every explicit method with two or more stages has stage order 1**, because its first stage is
$f(t_n, y_n)$ with $c_1 = 0$ and no correction at all, so the condition fails at $m = 2$.

Euler's row reads 6 and should be read as vacuous rather than as a large stage order. Its single
stage has $c_1 = 0$ and $A = [0]$, so the condition $\sum_j a_{1j}c_j^{m-1} = c_1^m/m$ is
$0 = 0$ for every $m$ and the definition never bites. The meaningful statement begins at two stages.

Radau IIA's stage order is 2, so its stiff order is 3, matching its classical order. **That is what
it was designed for**, and it is why it is the standard high order stiff solver rather than a
higher classical order method with stage order 1.

### 5.3 Stiffness without a Jacobian

The estimate of lesson 70 exercise 5.3 needs no Jacobian, only two states and the values of $f$
there. Generalise it: over any two successive solver states,

$$
\lambda_{\text{est}} = \frac{\lVert f(t_{n+1}, y_{n+1}) - f(t_n, y_n)\rVert}
{\lVert y_{n+1} - y_n\rVert},
$$

which is a directional difference quotient of $f$ along the trajectory. It costs **nothing**: both
evaluations exist already, one as the current step's first stage and one as the previous step's.

```python
def detect(f, t0, y0, t_end, steps, step_fn, limit):
    """Estimate |df/dy| from consecutive states and compare against a stability limit."""
    h = (t_end - t0) / steps
    t, y = float(t0), ivp.as_state(y0)
    f_here = ivp.as_state(f(t, y))
    biggest = 0.0
    for _ in range(steps):
        y_next = ivp.as_state(step_fn(f, t, y, h))
        f_next = ivp.as_state(f(t + h, y_next))
        gap_y = float(np.linalg.norm(y_next - y))
        if gap_y > 1e-12:
            biggest = max(biggest, float(np.linalg.norm(f_next - f_here)) / gap_y)
        y, f_here, t = y_next, f_next, t + h
    return biggest, biggest * h, biggest * h > 0.5 * limit

limit = rk.stability_limit("rk4")
print(f"{'true fastest':>14}{'steps':>8}{'estimated':>13}{'h * est':>11}"
      f"{'limit':>9}{'stability limited?':>21}")
for fast in (-1.0, -50.0, -500.0):
    def system(t, y, fast=fast):
        v = ivp.as_state(y)
        return np.asarray([fast * (v[0] - np.cos(t)) - np.sin(t), -v[1]])

    for steps in (100, 2000):
        with np.errstate(over="ignore", invalid="ignore"):
            est, scaled, flag = detect(system, 0.0, [1.0, 1.0], 1.0, steps,
                                       named_step("rk4"), limit)
        print(f"{abs(fast):>14.0f}{steps:>8}{est:>13.2f}{scaled:>11.3f}"
              f"{limit:>9.3f}{str(flag):>21}")
```

The estimate recovers the fastest rate from quantities the solver already has, and the last two
rows show both what it can and cannot do.

At $\lvert\lambda\rvert = 500$ with 100 steps it reports 500.00 and correctly flags the run as
stability limited. At the same $\lambda$ with 2000 steps it reports **1.52**, which is the slow
mode's rate, not the fast one.

That is the limitation, and it is not a defect in the estimate. With 2000 steps the fast transient
has died within the first few, so consecutive states after that differ only in the slow direction
and the difference quotient measures the slow rate. **The estimate sees the modes the trajectory is
currently exciting**, which is the right thing for deciding whether the current step is stability
limited and the wrong thing for characterising the problem.

The cost is two vector norms per step, which is why `LSODA` can afford to run this check
continuously and switch between an Adams method and a BDF method as the problem changes. The switch
is usually triggered by **repeated step rejections** as well, which are the symptom the estimate
can miss.

---

## Lesson 73, Systems and Higher Order Equations

### 1.1 Reducing a third order equation

$y''' + 2y'' - y' + 3y = \sin t$ becomes $y''' = -2y'' + y' - 3y + \sin t$. Name the derivatives
$u = (y, y', y'')$ and read off

$$
u' = \begin{pmatrix}u_2 \\ u_3 \\ -2u_3 + u_2 - 3u_1 + \sin t\end{pmatrix}.
$$

```python
import numpy as np
from nalib import odesystems as od

def third(t, y, dy, d2y):
    return -2.0 * d2y + dy - 3.0 * y + np.sin(t)

f = od.to_first_order(third, 3)
state = np.asarray([0.4, -1.1, 2.0])
print(f"state       {state}")
print(f"f(0.7, u)   {f(0.7, state)}")
print(f"by hand     {np.asarray([state[1], state[2], third(0.7, *state)])}")
```

### 1.2 Why the reduction does not change the order

A one step method's order comes from matching the Taylor expansion of the solution, and the
expansion of a vector valued solution is the componentwise expansion of each part. Nothing in the
order conditions of lesson 69 refers to the dimension: they are conditions on $A$, $b$, $c$ alone.

The same holds for the local to global argument. The Lipschitz constant becomes a matrix norm
bound, the error recurrence $\lvert e_{n+1}\rvert \le (1+hL)\lvert e_n\rvert + Ch^{p+1}$ becomes the
same inequality in a norm, and its solution is unchanged.

```python
print(f"{'method':>16}{'claimed':>9}{'fitted on a system':>20}{'matches':>9}")
for name in ("euler", "heun", "kutta third", "rk4"):
    out = od.order_on_a_system(name=name)
    print(f"{name:>16}{out['claimed_order']:>9}{out['fitted_order']:>20.4f}"
          f"{str(out['matches']):>9}")
```

### 1.3 The right check, and the wrong one

**Wrong:** solve the system, difference the first component numerically, and compare with the
second. That measures the differencing scheme, which is second order in $h$, and would report
$8\times10^{-4}$ shrinking like $h^2$ for a relationship that is exact.

**Right:** the reduction guarantees $f(t, u)_1 = u_2$ at **every** state, whatever the solver does,
by construction. Check that.

```python
out = od.reduction_is_faithful()
print(f"{'steps':>8}{'position error':>17}{'identity residual':>20}")
for n, p, r in zip(out["steps"], out["position_error"],
                   out["reduction_identity_residual"]):
    print(f"{n:>8}{p:>17.4e}{r:>20.1e}")
print(f"the identity is exact: {out['identity_is_exact']}")
print()
from nalib import ivp
from nalib.rungekutta import named_step

g, exact, omega = od.small_angle_pendulum()
print(f"{'steps':>8}{'by differencing':>20}{'by the identity':>19}")
for n in (20, 40, 80, 160):
    run = ivp.integrate(g, 0.0, [1.0, 0.0], 2.0, n, named_step("rk4"))
    differenced = np.gradient(run["y"][:, 0], float(run["step"]))
    by_gradient = float(np.max(np.abs(differenced[2:-2] - run["y"][2:-2, 1])))
    by_identity = max(abs(float(g(0.0, s)[0]) - float(s[1])) for s in run["y"])
    print(f"{n:>8}{by_gradient:>20.4e}{by_identity:>19.1e}")
```

The differencing column falls like $h^2$ and the identity column is exactly zero at every step
count. **Only one of them is measuring the reduction.**

### 1.4 Conserving one invariant and not another

A method that conserves a quantity $I(y)$ returns $I(y_n) = I(y_0)$ for all $n$, exactly. Nothing
in a Runge-Kutta method is designed to do that, so in general it conserves nothing.

It can happen anyway when the invariant has the right structure. **A linear invariant $c^{\mathsf T}
y$ is conserved by every Runge-Kutta method**, because
$c^{\mathsf T}y_{n+1} = c^{\mathsf T}y_n + h\sum b_i c^{\mathsf T}k_i$ and
$c^{\mathsf T}f \equiv 0$ makes every term vanish. A **quadratic** invariant $y^{\mathsf T}Sy$ is
conserved exactly when the tableau satisfies $b_ia_{ij} + b_ja_{ji} = b_ib_j$, which Gauss methods
do and explicit methods cannot.

Angular momentum $xv_y - yv_x$ is quadratic; energy is not, because of the $-1/r$ term.

```python
print(f"{'eccentricity':>14}{'energy drift':>15}{'angular momentum drift':>25}"
      f"{'ratio':>10}")
for e in (0.0, 0.3, 0.6, 0.9):
    out = od.orbit_closes(eccentricity=e, periods=10.0, steps=4000)
    print(f"{e:>14.1f}{out['energy_drift']:>15.3e}"
          f"{out['angular_momentum_drift']:>25.3e}"
          f"{out['energy_drift'] / out['angular_momentum_drift']:>10.2f}")
```

RK4 is not a Gauss method, so it does not conserve angular momentum exactly either. It conserves it
**better**, by a factor of 4 to 45, and measuring only energy would have missed that.

### 1.5 Why the Lyapunov exponent belongs to the equations

Linearise about a trajectory: a perturbation $\delta$ satisfies $\delta' = J(t)\delta$ where
$J$ is the Jacobian along the solution. Its growth rate averaged over the trajectory is the
**largest Lyapunov exponent**, and it is defined without reference to any numerical method.

The consequence is that a solver's accuracy enters only through the **size** of the initial
perturbation it introduces, and the time to lose a digit is

$$
t = \frac{\ln 10}{\lambda},
$$

whatever the solver. Improving the solver by a factor $F$ buys $\ln F/\lambda$ extra time units and
no more.

```python
out = od.divergence_is_the_problem()
print(f"fitted exponent {out['fitted_lyapunov']:.4f}")
print(f"time to reach order 1 from 1e-8: {out['time_to_order_one']:.2f}")
print(f"each halving of the error buys {out['extra_time_per_halving']:.3f} time units")
print(f"a factor of 10^6 buys {np.log(1e6) / out['fitted_lyapunov']:.1f} time units")
```

### 2.1 The companion form

The reduction of $y^{(m)} = g(t, y, \dots, y^{(m-1)})$ has

$$
\frac{\partial f}{\partial u} =
\begin{pmatrix}
0 & 1 & 0 & \cdots & 0\\
0 & 0 & 1 & \cdots & 0\\
\vdots & & & \ddots & \vdots\\
0 & 0 & 0 & \cdots & 1\\
g_{u_1} & g_{u_2} & g_{u_3} & \cdots & g_{u_m}
\end{pmatrix},
$$

which is a **companion matrix**: ones on the superdiagonal and the derivatives of $g$ in the last
row.

For the linear equation $y^{(m)} + a_{m-1}y^{(m-1)} + \dots + a_0y = 0$ the last row is
$(-a_0, -a_1, \dots, -a_{m-1})$, and the characteristic polynomial of the companion matrix is
exactly

$$
\lambda^m + a_{m-1}\lambda^{m-1} + \dots + a_0,
$$

the characteristic polynomial of the differential equation. **The eigenvalues of the reduced
system are the roots of the original equation's characteristic polynomial**, which is why the
reduction changes nothing about stiffness or stability.

```python
coefficients = [3.0, -1.0, 2.0]        # y''' + 2y'' - y' + 3y = 0
companion = np.zeros((3, 3))
companion[0, 1] = companion[1, 2] = 1.0
companion[2] = [-c for c in coefficients]
print(f"companion matrix:\n{companion}")
print(f"its eigenvalues:      {np.sort_complex(np.linalg.eigvals(companion))}")
print(f"roots of the polynomial: "
      f"{np.sort_complex(np.roots([1.0, 2.0, -1.0, 3.0]))}")
from nalib.stability import stiffness_ratio
print(f"stiffness ratio of the reduced system: "
      f"{stiffness_ratio(companion)['ratio']:.4f}")
```

### 2.2 The pendulum's energy

From $\theta'' = -(g/L)\sin\theta$, multiply by $L^2\theta'$:

$$
L^2\theta'\theta'' = -gL\sin\theta\cdot L\theta'
\quad\Longrightarrow\quad
\frac{d}{dt}\left[\tfrac12(L\theta')^2\right] = \frac{d}{dt}\left[gL\cos\theta\right],
$$

so

$$
E = \tfrac12(L\theta')^2 + gL(1 - \cos\theta)
$$

is constant along solutions, the constant $gL$ chosen so $E = 0$ at rest.

```python
f, energy = od.pendulum(length=1.0, gravity=9.81)
run = ivp.integrate(f, 0.0, [1.0, 0.0], 20.0, 200000, named_step("rk4"))
values = np.asarray(energy(run["y"]), dtype=float)
print(f"E at t = 0 is {values[0]:.12f}")
print(f"largest departure over 20 units at 200000 steps: "
      f"{float(np.max(np.abs(values - values[0]))):.3e}")
print()
print("dE/dt along the solution, by differencing E and by the chain rule:")
print(f"{'t':>7}{'theta':>12}{'E':>16}{'dE/dt differenced':>21}")
h = float(run["step"])
sampled = np.linspace(2, run["t"].size - 3, 6).astype(int)
for i in sampled:
    slope = (values[i + 1] - values[i - 1]) / (2.0 * h)
    print(f"{float(run['t'][i]):>7.2f}{float(run['y'][i, 0]):>12.6f}"
          f"{values[i]:>16.10f}{slope:>21.3e}")
```

The energy is constant to $10^{-13}$ over 20 time units at this step, and its numerical derivative
is at the same level. **Both columns are measuring roundoff**, which is what "exactly conserved"
looks like once a good enough solver is used.

### 2.3 Perihelion and aphelion speeds

For an ellipse of semi major axis $a = 1$ and eccentricity $e$, the perihelion distance is $1-e$
and the aphelion distance is $1+e$. Angular momentum $L = rv$ is conserved at both, where the
velocity is perpendicular to the radius, so

$$
(1-e)v_p = (1+e)v_a
\quad\Longrightarrow\quad
\frac{v_p}{v_a} = \frac{1+e}{1-e}.
$$

With $L = \sqrt{1-e^2}$ this gives $v_p = \sqrt{\dfrac{1+e}{1-e}}$ and
$v_a = \sqrt{\dfrac{1-e}{1+e}}$.

```python
print(f"{'e':>6}{'v perihelion':>15}{'v aphelion':>13}{'ratio':>10}"
      f"{'steps needed if uniform':>26}")
for e in (0.0, 0.3, 0.6, 0.9, 0.99):
    vp = np.sqrt((1 + e) / (1 - e))
    va = np.sqrt((1 - e) / (1 + e))
    print(f"{e:>6.2f}{vp:>15.6f}{va:>13.6f}{vp / va:>10.2f}"
          f"{vp / va:>26.1f}x more than needed")
```

**A fixed step must be small enough for perihelion and is then $\frac{1+e}{1-e}$ times smaller than
aphelion needs.** At $e = 0.9$ that is a factor of 19 wasted over most of the orbit, and at
$e = 0.99$ it is 199. That is precisely what lesson 70's adaptive stepping is for, and it is the
standard example of a problem where adaptivity is not optional.

### 2.4 Angular momentum in polar form

In polar coordinates the two body problem is

$$
r'' = r\theta'^2 - \frac{\mu}{r^2}, \qquad
\frac{d}{dt}\left(r^2\theta'\right) = 0,
$$

and the second equation says $L = r^2\theta'$ is constant. Reduced to first order with state
$(r, r', L)$, the third component has $L' = 0$ identically.

**Every Runge-Kutta method conserves it exactly**, because every stage returns $k_{i,3} = 0$, so
$L_{n+1} = L_n + h\sum_i b_i\cdot 0 = L_n$ to the last bit.

In Cartesian form $L = xv_y - yv_x$ is a **quadratic** function of the state, not a component of
it, and the argument fails: a Runge-Kutta method conserves quadratic invariants only when
$b_ia_{ij} + b_ja_{ji} = b_ib_j$, which no explicit method satisfies.

```python
from nalib.rungekutta import tableau

print(f"{'method':>16}{'worst |b_i a_ij + b_j a_ji - b_i b_j|':>40}")
for name in ("heun", "kutta third", "rk4"):
    A, b, c, order = tableau(name)
    M = np.outer(b, np.ones(b.size)) * A
    check = M + M.T - np.outer(b, b)
    print(f"{name:>16}{float(np.max(np.abs(check))):>40.6f}")
print()
print("the two stage Gauss method, from lesson 69 exercise 3.3:")
root3 = np.sqrt(3.0)
A = np.asarray([[0.25, 0.25 - root3 / 6.0], [0.25 + root3 / 6.0, 0.25]])
b = np.asarray([0.5, 0.5])
M = np.outer(b, np.ones(b.size)) * A
print(f"  worst residual {float(np.max(np.abs(M + M.T - np.outer(b, b)))):.2e}")
```

Gauss satisfies it to machine precision, which is why implicit Gauss methods conserve quadratic
invariants exactly and are used where that matters.

### 2.5 The local exponent against the fitted global one

The **local** exponent at a point is the largest real part of an eigenvalue of the Jacobian there.
The **global** Lyapunov exponent is the long time average growth rate of a perturbation, and the
two are different quantities: the local one can be positive everywhere while the global one is
negative, and the reverse.

```python
from nalib.ivp import as_state

lorenz = od.lorenz()
run = ivp.integrate(lorenz, 0.0, [1.0, 1.0, 20.0], 30.0, 30000, named_step("rk4"))
picks = np.linspace(0, run["t"].size - 1, 2000).astype(int)
locals_ = []
scale = np.sqrt(np.finfo(float).eps)
for i in picks:
    state = run["y"][i]
    J = np.empty((state.size, state.size))
    base = as_state(lorenz(float(run["t"][i]), state))
    for j in range(state.size):
        bumped = state.copy()
        step = scale * max(abs(state[j]), 1.0)
        bumped[j] += step
        J[:, j] = (as_state(lorenz(float(run["t"][i]), bumped)) - base) / step
    locals_.append(float(np.max(np.linalg.eigvals(J).real)))
locals_ = np.asarray(locals_)
out = od.divergence_is_the_problem()
print(f"local exponent: smallest {float(np.min(locals_)):.4f}, "
      f"largest {float(np.max(locals_)):.4f}, mean {float(np.mean(locals_)):.4f}")
print(f"fitted global Lyapunov exponent: {out['fitted_lyapunov']:.4f}")
```

**The mean of the local exponents is not the global exponent** and here it is much larger. The
reason is that a perturbation does not simply grow at the local rate: it is also **rotated** by the
flow, and time spent in a contracting direction cancels time spent in an expanding one. The
average of the eigenvalues ignores the rotation entirely.

The correct computation follows the perturbation itself, which is exercise 3.4.

### 3.1 The reduction for orders 1 to 6

```python
rng = np.random.default_rng(42)
print(f"{'m':>4}{'shift is exact':>17}{'g appended':>13}{'identity residual':>20}")
for m in range(1, 7):
    def g(t, *state, m=m):
        return sum((k + 1) * v for k, v in enumerate(state)) - t

    f = od.to_first_order(g, m)
    shift_ok, g_ok, residual = True, True, 0.0
    for _ in range(50):
        state = rng.normal(size=m)
        out = f(0.3, state)
        shift_ok = shift_ok and np.array_equal(out[:-1], state[1:])
        g_ok = g_ok and abs(float(out[-1]) - float(g(0.3, *state))) < 1e-14
        if m > 1:
            residual = max(residual, float(np.max(np.abs(out[:-1] - state[1:]))))
    print(f"{m:>4}{str(shift_ok):>17}{str(g_ok):>13}{residual:>20.1e}")
```

The residual is exactly zero at every order, which it must be: the reduction copies array entries.

At $m = 1$ there is nothing to shift, so the comparison is between two empty arrays. That is
`True` for the equality and undefined for the maximum, which is why the residual column is only
computed from $m = 2$: **taking a maximum over an empty set is the failure, not the reduction.**

### 3.2 An Arenstorf orbit

The restricted three body problem with the Earth-Moon mass ratio $\mu = 0.012277471$ has periodic
solutions found by Arenstorf. One of them starts at

$$
(x, y) = (0.994, 0), \qquad (x', y') = (0, -2.00158510637908),
$$

with period $T = 17.0652165601579625588$.

```python
MU = 0.012277471
NU = 1.0 - MU

def arenstorf(t, state):
    s = as_state(state)
    x, y, dx, dy = s
    d1 = ((x + MU) ** 2 + y ** 2) ** 1.5
    d2 = ((x - NU) ** 2 + y ** 2) ** 1.5
    return np.asarray([
        dx, dy,
        x + 2.0 * dy - NU * (x + MU) / d1 - MU * (x - NU) / d2,
        y - 2.0 * dx - NU * y / d1 - MU * y / d2])

PERIOD = 17.0652165601579625588
start = np.asarray([0.994, 0.0, 0.0, -2.00158510637908])
print(f"{'steps':>10}{'closure gap':>16}{'ratio':>9}")
last = None
for steps in (10000, 40000, 160000, 640000):
    run = ivp.integrate(arenstorf, 0.0, start, PERIOD, steps, named_step("rk4"))
    gap = float(np.max(np.abs(run["y"][-1] - start)))
    print(f"{steps:>10}{gap:>16.4e}" + ("" if last is None else f"{last / gap:>9.1f}"))
    last = gap
```

The ratios are 80, 288 and then 264 for fourfold refinements, where fourth order predicts 256.
**The first two rows are not in the asymptotic regime at all**: at 10000 steps the closure gap is
1.83, which is larger than the orbit, so the computed trajectory is not the orbit and the number
means nothing.

That is what makes this the standard test case for adaptive solvers. The trajectory passes very
close to both bodies, where $d_1$ or $d_2$ becomes small and the solution turns sharply, and a
uniform grid must be fine enough for those passages everywhere. **640000 fixed steps for one
period** is the price, and an adaptive solver reaches the same accuracy in a few thousand.

### 3.3 Hodgkin-Huxley with a stiff solver

```python
f, y0 = od.hodgkin_huxley()
reference = ivp.integrate(f, 0.0, y0, 20.0, 400000, named_step("rk4"))
want = reference["y"][-1]
print(f"reference V(20) = {float(want[0]):.8f}\n")
print(f"{'target':>10}{'RK4 steps':>12}{'RK4 evals':>12}"
      f"{'backward Euler steps':>23}{'BE evals':>11}{'winner':>16}")
for target in (1e-1, 1e-2, 1e-3):
    costs = {}
    for label, step, per_step in (("rk4", named_step("rk4"), 4),
                                  ("be", ivp.backward_euler_step, 6)):
        n = 500
        while n < 400000:
            with np.errstate(over="ignore", invalid="ignore"):
                run = ivp.integrate(f, 0.0, y0, 20.0, n, step)
            value = float(np.max(np.abs(run["y"][-1] - want)))
            if np.isfinite(value) and value <= target:
                break
            n = int(n * 1.5)
        costs[label] = (n, n * per_step)
    winner = "rk4" if costs["rk4"][1] <= costs["be"][1] else "backward euler"
    print(f"{target:>10.0e}{costs['rk4'][0]:>12}{costs['rk4'][1]:>12}"
          f"{costs['be'][0]:>23}{costs['be'][1]:>11}{winner:>16}")
```

RK4 wins on evaluation count here, and the reason is worth stating: the Hodgkin-Huxley stiffness
ratio peaks at 880, which forces RK4 to about 4000 steps for stability, and its fourth order then
gives accuracy far beyond what backward Euler's first order reaches in the same budget.

**Stiffness of 880 is not stiff enough** for a first order A-stable method to win. Lesson 72
exercise 4.2 found the crossover at a ratio near $10^4$, and this problem sits an order of magnitude
below it.

### 3.4 The Lyapunov exponent by renormalisation

The standard algorithm evolves the trajectory and a perturbation together, renormalising the
perturbation to unit length at intervals and accumulating the logarithms of the stretching factors:

$$
\lambda = \lim_{N\to\infty}\frac{1}{N\tau}\sum_{k=1}^{N}\ln\lVert\delta_k\rVert .
$$

```python
def lyapunov(f, y0, t_end=200.0, tau=0.5, steps_per=200, seed=42):
    """The largest Lyapunov exponent by periodic renormalisation."""
    rng = np.random.default_rng(seed)
    y = as_state(y0)
    d = rng.normal(size=y.size)
    d = d / np.linalg.norm(d) * 1e-8
    total, count = 0.0, 0
    t = 0.0
    while t < t_end:
        a = ivp.integrate(f, t, y, t + tau, steps_per, named_step("rk4"))
        b = ivp.integrate(f, t, y + d, t + tau, steps_per, named_step("rk4"))
        gap = b["y"][-1] - a["y"][-1]
        size = float(np.linalg.norm(gap))
        total += np.log(size / np.linalg.norm(d))
        count += 1
        y = a["y"][-1]
        d = gap / size * 1e-8
        t += tau
    return total / (count * tau)

value = lyapunov(lorenz, [1.0, 1.0, 20.0])
out = od.divergence_is_the_problem()
print(f"by renormalisation:      {value:.4f}")
print(f"by fitting the gap:      {out['fitted_lyapunov']:.4f}")
print(f"published value:         0.9056")
print(f"gap between the two measurements: {abs(value - out['fitted_lyapunov']):.4f}")
```

The two agree to a couple of per cent and both sit near the published 0.9056. **The fitted version
is the cheaper and the cruder**: it works only while the gap is small enough to grow linearly and
large enough to be above roundoff, which is a window of about ten time units.

### 3.5 A slowly increasing wind

```python
def ramped_tacoma(t, state, rate=1.0, base=40.0):
    """The Tacoma model with the wind rising linearly in time."""
    wind = base + rate * t
    return od.tacoma_narrows(wind=wind)(t, state)

print(f"{'ramp rate':>11}{'final |twist|':>16}{'wind at the end':>18}"
      f"{'reaches 0.01':>14}")
for rate in (0.2, 0.5, 1.0, 2.0):
    def f(t, state, rate=rate):
        return ramped_tacoma(t, state, rate=rate)

    run = ivp.integrate(f, 0.0, [0.0, 0.0, 1e-3, 0.0], 300.0, 60000,
                        named_step("rk4"))
    theta = np.abs(run["y"][:, 2])
    big = np.flatnonzero(theta > 1e-2)
    when = f"{float(run['t'][big[0]]):.1f}" if big.size else "never"
    print(f"{rate:>11.1f}{float(theta[-1]):>16.4e}{40.0 + rate * 300.0:>18.1f}"
          f"{when:>14}")
print()
out = od.a_parameter_crossing_changes_the_answer()
print(f"the fixed wind sweep grows first at wind {out['first_wind_that_grows']}, "
      f"peaking near 140")
```

**The ramped run does not take off where the fixed sweep says it should**, and the direction is the
opposite of the obvious guess: it takes off **later**, or not at all within 300 time units.

At a ramp rate of 1.0 the wind passes through 140, the peak of the fixed sweep, at $t = 100$, and
nothing happens. At a rate of 2.0 it passes through 140 at $t = 50$ and takes off much later, at a
wind above 300.

The reason is that growth takes **time**. The fixed sweep measures growth over 200 time units at a
constant wind; a ramp at rate 1.0 spends only a few units within any given band of wind speeds, so
it accumulates a fraction of the amplification and moves on before the resonance can do its work.

**A slowly varying parameter is not a sequence of fixed parameter problems.** Which way the error
goes depends on the ratio of the ramp rate to the growth rate, and here the ramp is fast compared
with the resonance, so the resonance is effectively skipped. That is the subject of adiabatic
theory, and it is why bridge engineers care about how fast the wind rises as well as how high.

### 4.1 Closure gap against eccentricity

```python
print(f"{'e':>6}{'closure gap':>15}{'v_p / v_a':>12}")
gaps, ratios = [], []
for e in (0.2, 0.4, 0.6, 0.8, 0.9, 0.95):
    out = od.orbit_closes(eccentricity=e, periods=5.0, steps=5000)
    gaps.append(out["closure_gap"])
    ratios.append((1 + e) / (1 - e))
    print(f"{e:>6.2f}{out['closure_gap']:>15.3e}{ratios[-1]:>12.2f}")
fit = float(np.polyfit(np.log(ratios), np.log(gaps), 1)[0])
print(f"\nfitted exponent of the gap in (1+e)/(1-e): {fit:.3f}")
```

**The exponent is 7.0, not 4.** The naive guess is that the closure gap is the method's fourth order
error evaluated with an effective step scaled by the perihelion speed, giving
$\left(\frac{1+e}{1-e}\right)^4$. That is wrong, and by a wide margin.

Two effects compound. The local error is worst at perihelion, contributing the fourth power. And
the **error made there is then amplified** by the rest of the orbit, because a small error in the
energy changes the semi major axis, which changes the period, which turns a position error into a
phase error growing linearly with the number of orbits. The phase error at fixed time is what the
closure gap measures.

So the eccentricity enters twice, and refining a **uniform** grid is a very poor way to buy the
loss back: reaching $e = 0.95$'s accuracy at $e = 0.2$'s level would need $10^{9/4}$ times as many
steps.

### 4.2 Stiffness against the injected current

```python
print(f"{'current':>9}{'spikes':>8}{'peak V':>10}{'smallest ratio':>17}"
      f"{'largest ratio':>16}")
for current in (0.0, 2.0, 5.0, 10.0, 20.0, 50.0):
    f, y0 = od.hodgkin_huxley(current=current)
    run = ivp.integrate(f, 0.0, y0, 40.0, 40000, named_step("rk4"))
    voltage = run["y"][:, 0]
    crossings = int(np.sum((voltage[:-1] < 0.0) & (voltage[1:] >= 0.0)))
    picks = np.linspace(0, run["t"].size - 1, 60).astype(int)
    ratios = []
    scale = np.sqrt(np.finfo(float).eps)
    for i in picks:
        state = run["y"][i]
        J = np.empty((state.size, state.size))
        base = as_state(f(float(run["t"][i]), state))
        for j in range(state.size):
            bumped = state.copy()
            step = scale * max(abs(state[j]), 1.0)
            bumped[j] += step
            J[:, j] = (as_state(f(float(run["t"][i]), bumped)) - base) / step
        ratios.append(stiffness_ratio(J)["ratio"])
    print(f"{current:>9.1f}{crossings:>8}{float(np.max(voltage)):>10.2f}"
          f"{float(np.min(ratios)):>17.1f}{float(np.max(ratios)):>16.1f}")
```

**The neuron starts spiking between currents of 2 and 5**, and the peak voltage jumps from $-60$ mV
to $+39$ mV as it does.

The stiffness does **not** rise monotonically with the current. The largest ratio along the run goes
38.7, 70.2, 1284, 103.4, 175.0, 503.7 as the current rises through 0, 2, 5, 10, 20, 50, with its
maximum at the **onset** of spiking rather than at the largest current.

That shape has a reason. At current 5 the neuron fires exactly once and then relaxes very slowly
back towards rest, and it is that long slow tail alongside the spike's fast sodium dynamics that
gives the widest separation of time constants. At higher currents it fires repeatedly and never
relaxes fully, so the slow end of the spectrum is never as slow.

**Stiffness is a property of the equations and the trajectory together**, and a parameter change
that alters the trajectory qualitatively alters the stiffness in a way no inspection of the
equations predicts.

### 4.3 Where the Lorenz system stops being chaotic

```python
print(f"{'rho':>7}{'exponent':>12}{'positive?':>11}")
for rho in (0.5, 5.0, 15.0, 20.0, 24.0, 28.0, 100.0, 160.0, 200.0):
    system = od.lorenz(rho=rho)
    value = lyapunov(system, [1.0, 1.0, min(rho, 30.0)], t_end=120.0)
    print(f"{rho:>7.1f}{value:>12.4f}{str(value > 0.05):>11}")
```

The exponent is **negative for $\rho$ up to about 15 and positive from 24 onwards**, and at
$\rho = 28$ it reads 0.92 against the published 0.9056.

The values need reading against what is known about the system rather than as a clean threshold.
Below $\rho = 1$ the origin is the only fixed point and everything decays to it. Between 1 and
13.926 the two nonzero fixed points are stable and attract everything. Between 13.926 and 24.06
there is a **chaotic set coexisting with the stable fixed points**, so which answer a run gives
depends on where it starts, and the positive reading at $\rho = 24$ is a trajectory that has landed
on the chaotic set rather than in a basin. Above 24.06 the fixed points lose stability and the
chaotic attractor is the only thing left.

The readings at 100 and 160 are small and positive, which is the signature of the **periodic
windows**: the system passes through intervals of $\rho$ where a stable limit cycle exists, and a
finite time estimate near such a window returns a small number rather than a clean zero.

**None of this is visible from a single trajectory.** Nothing about a run at $\rho = 20$ hints that
$\rho = 25$ behaves differently, which is lesson 73 section 6.2's point in another problem.

### 5.1 Shadowing

**The shadowing lemma** (Anosov, Bowen). Let $\Lambda$ be a hyperbolic invariant set of a
diffeomorphism $\varphi$. For every $\varepsilon > 0$ there is a $\delta > 0$ such that every
$\delta$-pseudo-orbit in $\Lambda$, meaning a sequence with
$\lVert x_{n+1} - \varphi(x_n)\rVert < \delta$, is $\varepsilon$-shadowed by a **true** orbit: there
exists $y_0$ with $\lVert x_n - \varphi^n(y_0)\rVert < \varepsilon$ for all $n$.

A numerical trajectory with local errors below $\delta$ is exactly a $\delta$-pseudo-orbit. So the
computed trajectory is close to *some* true trajectory, for as long as the lemma applies.

**What it rescues.** Statements about the attractor: its shape, its dimension, the invariant
measure, long time averages of observables. Those are properties shared by all orbits on the
attractor, and a shadowing orbit is one of them, so the computation reports them correctly.

**What it does not rescue.** The initial condition. The shadowing orbit starts at some $y_0$ near
$x_0$ and generally not at $x_0$, so "the solution from $x_0$ at $t = 50$" is not what was computed.
The lemma converts a statement about **the** trajectory into a statement about **a** trajectory, and
the two questions have different answers.

Two further caveats worth knowing. The Lorenz attractor is not uniformly hyperbolic, so the lemma
does not apply to it directly; what is available is numerical shadowing over finite times, which has
been verified computationally for tens of time units and not for hundreds. And the shadowing time is
itself limited: for a $\delta$ of $10^{-16}$ and a Lyapunov exponent of 0.9, the orbit can be
shadowed for roughly $\ln(\varepsilon/\delta)/\lambda$ time units and no longer.

```python
out = od.divergence_is_the_problem()
lam = out["fitted_lyapunov"]
print(f"{'local error':>14}{'shadowing radius':>19}{'shadowing time':>17}")
for delta in (1e-16, 1e-12, 1e-8):
    for eps in (1e-3,):
        print(f"{delta:>14.0e}{eps:>19.0e}{np.log(eps / delta) / lam:>17.2f}")
```

### 5.2 Conserving invariants by projection

After each step, project the state back onto $\{y : I(y) = I(y_0)\}$. For a single scalar invariant
the standard projection solves

$$
y_{n+1}^{\text{proj}} = \hat y_{n+1} + \mu\,\nabla I(\hat y_{n+1}), \qquad
I\!\left(y_{n+1}^{\text{proj}}\right) = I(y_0),
$$

a scalar equation in $\mu$, solved by a few Newton steps.

```python
def project_energy(f, energy, gradient, target, step_fn):
    """Wrap a step function so the energy is restored exactly after every step."""
    def step(g, t, y, h, _counter=None):
        raw = as_state(step_fn(g, t, y, h, _counter))
        mu = 0.0
        for _ in range(20):
            point = raw + mu * gradient(raw)
            value = float(energy(point[None, :])[0]) - target
            if abs(value) < 1e-14:
                break
            grad_here = gradient(point)
            derivative = float(grad_here @ gradient(raw))
            if abs(derivative) < 1e-300:
                break
            mu -= value / derivative
        return raw + mu * gradient(raw)
    return step

f, energy = od.pendulum()

def energy_gradient(state):
    return np.asarray([9.81 * np.sin(float(state[0])), float(state[1])])

start = np.asarray([1.0, 0.0])
target = float(energy(start[None, :])[0])
plain = named_step("rk4")
projected = project_energy(f, energy, energy_gradient, target, plain)

print(f"{'method':>12}{'energy drift':>16}{'trend':>14}{'phase error at t=200':>24}")
reference = ivp.integrate(f, 0.0, start, 200.0, 400000, named_step("rk4"))
for label, step in (("rk4", plain), ("projected", projected)):
    run = ivp.integrate(f, 0.0, start, 200.0, 20000, step)
    values = np.asarray(energy(run["y"]), dtype=float)
    drift = float(np.max(np.abs(values - values[0]))) / abs(values[0])
    trend = float(np.polyfit(run["t"], (values - values[0]) / values[0], 1)[0])
    phase = float(np.max(np.abs(run["y"][-1] - reference["y"][-1])))
    print(f"{label:>12}{drift:>16.3e}{trend:>14.3e}{phase:>24.3e}")
```

**Projection restores the energy exactly and does not fix the trajectory.** The phase error is
comparable, and often slightly worse, because the projection moves the state along the energy
gradient rather than along the true flow: it corrects the invariant and adds a small error in the
direction it corrected.

The order is preserved, since the correction is the size of the error being corrected, which is
$O(h^{p+1})$ per step.

Compare with lesson 74's symplectic methods, which do not conserve energy at all and get the long
term behaviour right anyway. **Conserving the invariant is not the same as preserving the
structure**, and it is the second that buys the qualitative correctness.

### 5.3 Why angular momentum survives

The structural property is that **angular momentum is a quadratic invariant and energy is not.**

For the Cartesian two body problem $L(y) = xv_y - yv_x = y^{\mathsf T}Sy$ with

$$
S = \tfrac12\begin{pmatrix}0&0&0&1\\0&0&-1&0\\0&-1&0&0\\1&0&0&0\end{pmatrix},
$$

a symmetric matrix. Energy $\tfrac12\lVert v\rVert^2 - \mu/r$ has a quadratic kinetic part and a
$-\mu/r$ potential part, which is not polynomial in the state at all.

The theorem (exercise 2.4) is that a Runge-Kutta method conserves **every** quadratic invariant
exactly when $b_ia_{ij} + b_ja_{ji} = b_ib_j$. The cleanest demonstration is a method that
satisfies it against one that does not, on the same problem.

```python
from nalib.nlsystems import newton_system

root3 = np.sqrt(3.0)
GAUSS_A = np.asarray([[0.25, 0.25 - root3 / 6.0], [0.25 + root3 / 6.0, 0.25]])
GAUSS_B = np.asarray([0.5, 0.5])
GAUSS_C = np.asarray([0.5 - root3 / 6.0, 0.5 + root3 / 6.0])

def gauss_step(f, t, y, h, _counter=None):
    """The two stage Gauss method: order 4, and it conserves quadratic invariants."""
    state = as_state(y)
    n = state.size

    stages = GAUSS_B.size

    def residual(flat):
        k = np.asarray(flat, dtype=float).reshape(stages, n)
        out = np.empty_like(k)
        for i in range(stages):
            out[i] = k[i] - as_state(f(t + GAUSS_C[i] * h, state + h * (GAUSS_A[i] @ k)))
        return out.ravel()

    answer = newton_system(residual, None, np.tile(as_state(f(t, state)), stages),
                           tol=1e-14, max_iter=40)
    k = np.asarray(answer.root, dtype=float).reshape(stages, n)
    return state + h * (GAUSS_B @ k)

f, energy, angular = od.two_body()
y0 = od.elliptical_orbit_start(0.3)
print(f"{'method':>10}{'satisfies the condition':>26}{'energy drift':>16}"
      f"{'angular momentum drift':>25}{'ratio':>10}")
for label, step, A, b in (("rk4", named_step("rk4"),) + tuple(tableau("rk4")[:2]),
                          ("gauss 2", gauss_step, GAUSS_A, GAUSS_B)):
    M = np.outer(b, np.ones(b.size)) * A
    satisfied = float(np.max(np.abs(M + M.T - np.outer(b, b)))) < 1e-14
    run = ivp.integrate(f, 0.0, y0, 2 * np.pi * 20, 20000, step)
    e = np.asarray(energy(run["y"]), dtype=float)
    L = np.asarray(angular(run["y"]), dtype=float)
    de = float(np.max(np.abs(e - e[0]))) / abs(e[0])
    dL = float(np.max(np.abs(L - L[0]))) / abs(L[0])
    print(f"{label:>10}{str(satisfied):>26}{de:>16.4e}{dL:>25.4e}{de / dL:>10.1f}")
```

**Gauss conserves angular momentum to $10^{-14}$ and its energy still drifts at $10^{-11}$**, a
separation of a factor of 4600. RK4 separates them by only 5. That is the theorem showing itself:
the quadratic invariant is exact for the method that satisfies the condition and merely small for
the one that does not.

**Constructing the reverse** is harder than it looks, and the attempt is worth recording. The
obvious try is a system with a quadratic invariant and a non quadratic one and hoping the second is
better conserved. The two dimensional harmonic oscillator has both: energy
$\tfrac12\lVert y\rVert^2$ is quadratic, and the product of the two independent amplitudes
$\sqrt{q_1^2+p_1^2}\cdot\sqrt{q_2^2+p_2^2}$ is conserved and not quadratic.

```python
def oscillator(t, state):
    q1, q2, p1, p2 = as_state(state)
    return np.asarray([p1, p2, -q1, -q2])

def quadratic_energy(y):
    v = np.atleast_2d(np.asarray(y, dtype=float))
    return 0.5 * np.sum(v ** 2, axis=1)

def amplitude_product(y):
    v = np.atleast_2d(np.asarray(y, dtype=float))
    return (np.sqrt(v[:, 0] ** 2 + v[:, 2] ** 2)
            * np.sqrt(v[:, 1] ** 2 + v[:, 3] ** 2))

start = np.asarray([1.0, 0.5, 0.0, 0.8])
run = ivp.integrate(oscillator, 0.0, start, 200.0, 20000, named_step("rk4"))
for label, invariant in (("quadratic energy", quadratic_energy),
                         ("amplitude product", amplitude_product)):
    values = np.asarray(invariant(run["y"]), dtype=float)
    print(f"{label:>20}: relative drift "
          f"{float(np.max(np.abs(values - values[0]))) / abs(values[0]):.4e}")
```

**The two drift by exactly the same relative amount**, so the construction fails to separate them.
The reason is that on this system RK4 damps both amplitudes by the same factor per step, and every
smooth function of the amplitudes then inherits the same relative drift. The invariant's **shape**
is irrelevant here because the error has only one degree of freedom.

That is the honest answer to the exercise. The forward direction is a theorem and it is
demonstrable; the reverse is not obviously constructible, and the attempt above shows why a naive
construction collapses.

---

## Lesson 74, Geometric and Symplectic Integrators

### 1.1 Why the eigenvalues are imaginary

The harmonic oscillator $q' = p$, $p' = -\omega^2 q$ has

$$
\frac{d}{dt}\begin{pmatrix}q\\p\end{pmatrix}
= \begin{pmatrix}0 & 1\\ -\omega^2 & 0\end{pmatrix}
\begin{pmatrix}q\\p\end{pmatrix},
$$

whose characteristic polynomial is $\lambda^2 + \omega^2 = 0$, so $\lambda = \pm i\omega$: **purely
imaginary, with zero real part.**

That is not special to this system. A Hamiltonian flow's Jacobian is $J\nabla^2H$ with
$J = \begin{pmatrix}0&I\\-I&0\end{pmatrix}$, and such a matrix has eigenvalues in quadruples
$\{\lambda, -\lambda, \bar\lambda, -\bar\lambda\}$: **the spectrum is symmetric about both axes**.
So a stable equilibrium, where no eigenvalue has positive real part, must have all of them on the
imaginary axis, since any with negative real part would be paired with one having positive real
part.

```python
import numpy as np

for omega in (0.5, 1.0, 3.0):
    A = np.asarray([[0.0, 1.0], [-omega ** 2, 0.0]])
    print(f"omega = {omega}: eigenvalues {np.linalg.eigvals(A)}")
print()
J = np.asarray([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]])
rng = np.random.default_rng(42)
H = rng.normal(size=(4, 4))
H = H + H.T                                     # a symmetric Hessian
values = np.linalg.eigvals(J @ H)
print(f"a random Hamiltonian Jacobian J H:")
print(f"  eigenvalues {np.round(np.sort_complex(values), 6)}")
print(f"  the set is closed under negation: "
      f"{np.allclose(np.sort_complex(values), np.sort_complex(-values))}")
```

### 1.2 The one line difference

**Explicit Euler:** $q_{n+1} = q_n + hp_n$, $p_{n+1} = p_n + hF(q_n)$. Both updates use the
**old** values.

**Symplectic Euler:** $p_{n+1} = p_n + hF(q_n)$ first, then $q_{n+1} = q_n + hp_{n+1}$ using the
**new** momentum.

The momenta are identical; only the position update differs, by $h(p_{n+1} - p_n) = h^2F(q_n)$.

```python
from nalib import symplectic as sy

force, energy = sy.pendulum()
q, p, h = np.asarray([0.7]), np.asarray([-0.4]), 0.1
sq, sp = sy.symplectic_euler_step(force, q, p, h)
eq, ep = sy.explicit_euler_step(force, q, p, h)
print(f"explicit:   q = {float(eq[0]):.12f}, p = {float(ep[0]):.12f}")
print(f"symplectic: q = {float(sq[0]):.12f}, p = {float(sp[0]):.12f}")
print(f"momenta identical: {abs(float(sp[0]) - float(ep[0])) == 0.0}")
print(f"position gap {float(sq[0] - eq[0]):.3e}, "
      f"h^2 F(q) = {h ** 2 * float(force(q)[0]):.3e}")
```

### 1.3 The exact solution of a nearby Hamiltonian

**Backward error analysis.** For a symplectic method of order $p$ applied to $H$, there is a
modified Hamiltonian

$$
\tilde H = H + h^pH_p + h^{p+1}H_{p+1} + \dots
$$

such that the numerical trajectory is (to any chosen order in $h$) the **exact** flow of
$\tilde H$, sampled at the grid points.

Two consequences follow immediately. $\tilde H$ is exactly conserved along the numerical
trajectory, because it is a Hamiltonian and its own flow conserves it. And $\tilde H$ differs from
$H$ by $O(h^p)$, so the computed value of $H$ stays within $O(h^p)$ of its initial value **for all
time**, oscillating rather than drifting.

**The caveat**, which exercise 5.1 develops: the series is asymptotic and not convergent, so the
statement is "up to exponentially small terms over exponentially long times" rather than "forever".

### 1.4 A band against a small error

A **band** says: the error stays inside a fixed interval, whose width shrinks with $h$, no matter
how long the run.

A **small error** says: the error is small now. If it has a trend, extending the run extends the
error in proportion.

The band is the better guarantee whenever the run is long enough for a trend to matter, which for a
solar system integration means always.

**It is not better when the run is short**, and that is worth being explicit about. Over a few
periods a fourth order method with a trend is simply more accurate than a second order method with
a band, and the trend has not had time to matter. The crossover is at the run length where the
trend's accumulated contribution equals the band's width.

```python
out = sy.against_runge_kutta()
verlet_band = out["verlet_band"]
rk_trend = abs(out["rk4_trend"])
print(f"Verlet's band is {verlet_band:.3e}")
print(f"RK4's trend is {rk_trend:.3e} per unit time")
print(f"they are equal after {verlet_band / rk_trend:.0f} time units")
print(f"the run measured here was {out['t_end']:.0f} time units")
```

### 1.5 What "separable" excludes

Stormer-Verlet needs $H(q,p) = T(p) + V(q)$, so that $\partial H/\partial q$ depends only on $q$
and $\partial H/\partial p$ only on $p$, which is what lets each half step be explicit.

That covers mechanics with velocity independent forces: gravity, springs, molecular potentials.

It **excludes** anything with a $q$-$p$ cross term:

- A **magnetic field**, where $H = \frac{1}{2m}\lVert p - eA(q)\rVert^2 + e\phi(q)$ and the
  cross term $-\frac{e}{m}p\cdot A(q)$ couples them.
- **Relativistic** kinetic energy $\sqrt{m^2c^4 + c^2\lVert p\rVert^2}$ is still separable, but a
  relativistic particle in a field is not.
- **Rigid body** rotation, where the kinetic energy depends on the orientation through the inertia
  tensor.
- Any problem in **non canonical** coordinates, where the symplectic form itself is not the
  standard one.

For those the available symplectic methods are implicit: the **implicit midpoint rule** is
symplectic for any Hamiltonian, and the Gauss methods of lesson 69 exercise 3.3 are symplectic at
every order. Exercise 3.3 measures the first of them.

### 2.1 Explicit Euler's determinant

For $H = \tfrac12 p^2 + \tfrac12\omega^2q^2$ the map is

$$
\begin{pmatrix}q_{n+1}\\p_{n+1}\end{pmatrix}
= \begin{pmatrix}1 & h\\ -h\omega^2 & 1\end{pmatrix}
\begin{pmatrix}q_n\\p_n\end{pmatrix},
$$

whose determinant is

$$
1\cdot 1 - h\cdot(-h\omega^2) = 1 + h^2\omega^2 .
$$

**Every step multiplies phase space area by $1 + h^2\omega^2 > 1$.** After $n$ steps the area is
$(1+h^2\omega^2)^n \approx e^{n h^2\omega^2} = e^{h\omega^2 t}$, which grows exponentially in time
and is exactly why the energy grows.

```python
print(f"{'h':>8}{'det':>16}{'1 + h^2':>16}{'area after t = 100':>22}")
for h in (0.05, 0.2, 0.5):
    out = sy.area_is_preserved("explicit euler", h=h, corners=40)
    det = 1.0 + float(out["worst_departure_from_one"])
    n = int(100.0 / h)
    print(f"{h:>8.2f}{det:>16.10f}{1.0 + h ** 2:>16.10f}{det ** n:>22.4e}")
```

### 2.2 Symplectic Euler's determinant

$$
p_{n+1} = p_n - h\omega^2q_n, \qquad
q_{n+1} = q_n + hp_{n+1} = q_n + h(p_n - h\omega^2q_n) = (1 - h^2\omega^2)q_n + hp_n,
$$

so the map is

$$
\begin{pmatrix}1 - h^2\omega^2 & h\\ -h\omega^2 & 1\end{pmatrix},
$$

with determinant

$$
(1 - h^2\omega^2)\cdot 1 - h\cdot(-h\omega^2) = 1 - h^2\omega^2 + h^2\omega^2 = 1
$$

**exactly, for every $h$ and every $\omega$.** No approximation and no limit.

```python
print(f"{'h':>8}{'omega':>8}{'determinant':>20}{'exactly 1?':>13}")
for h in (0.05, 0.2, 0.5, 2.0):
    for omega in (1.0, 3.0):
        M = np.asarray([[1.0 - h ** 2 * omega ** 2, h], [-h * omega ** 2, 1.0]])
        det = float(np.linalg.det(M))
        print(f"{h:>8.2f}{omega:>8.1f}{det:>20.16f}{str(det == 1.0):>13}")
```

The determinant is 1 even at $h = 2$, where the method is wildly inaccurate. **Symplecticity is not
accuracy**, and it survives step sizes at which nothing else does.

### 2.3 Verlet as a composition

Write $S_h$ for symplectic Euler (kick then drift) and $S_h^{*}$ for its **adjoint** (drift then
kick). Then

$$
\text{Stormer-Verlet} = S^{*}_{h/2}\circ S_{h/2}
$$

is a half kick, then two drifts of $h/2$ which combine into one of $h$, then a half kick:

$$
p_{n+1/2} = p_n + \tfrac{h}{2}F(q_n), \quad
q_{n+1} = q_n + hp_{n+1/2}, \quad
p_{n+1} = p_{n+1/2} + \tfrac{h}{2}F(q_{n+1}).
$$

**Symplecticity follows immediately:** a composition of symplectic maps is symplectic, because the
Jacobian of a composition is the product of the Jacobians and each has determinant 1. No separate
calculation is needed.

```python
force, energy, closed_form = sy.harmonic_oscillator()

def composed(force, q, p, h):
    """Symplectic Euler at h/2, then its adjoint at h/2."""
    p_mid = p + 0.5 * h * force(q)
    q_mid = q + 0.5 * h * p_mid
    q_new = q_mid + 0.5 * h * p_mid
    p_new = p_mid + 0.5 * h * force(q_new)
    return q_new, p_new

q, p, h = np.asarray([0.8]), np.asarray([-0.3]), 0.17
a = sy.stormer_verlet_step(force, q, p, h)
b = composed(force, q, p, h)
print(f"Verlet:     q = {float(a[0][0]):.14f}, p = {float(a[1][0]):.14f}")
print(f"composition: q = {float(b[0][0]):.14f}, p = {float(b[1][0]):.14f}")
print(f"identical: {abs(float(a[0][0]) - float(b[0][0])) < 1e-15 and abs(float(a[1][0]) - float(b[1][0])) < 1e-15}")
```

### 2.4 The modified Hamiltonian of symplectic Euler

For symplectic Euler applied to $H(q,p) = T(p) + V(q)$, the leading correction is

$$
\tilde H = H + \frac{h}{2}\,T'(p)\,V'(q) + O(h^2)
= H - \frac{h}{2}\,p\,F(q) + O(h^2)
$$

with $F = -V'$, obtained by matching the Baker-Campbell-Hausdorff expansion of the composition
against the flow of $\tilde H$.

Since $\tilde H$ is exactly conserved and differs from $H$ by $\tfrac{h}{2}pF(q)$, the computed $H$
oscillates with amplitude

$$
\frac{h}{2}\max_{\text{orbit}}\lvert p\,F(q)\rvert = O(h),
$$

which is **first order**, matching symplectic Euler's order.

```python
force, energy = sy.pendulum()
print(f"{'h':>10}{'measured band':>18}{'h/2 max |p F(q)|':>22}{'ratio':>9}")
for h in (0.2, 0.1, 0.05, 0.025):
    steps = int(100.0 / h)
    run = sy.integrate(force, [1.0], [0.0], 100.0, steps, "symplectic euler")
    values = np.asarray(energy(run["q"], run["p"]), dtype=float)
    band = float(np.max(values) - np.min(values))
    predicted = 0.5 * h * float(np.max(np.abs(run["p"][:, 0]
                                              * np.asarray([force(np.asarray([v]))[0]
                                                            for v in run["q"][:, 0]]))))
    print(f"{h:>10.4f}{band:>18.6e}{predicted:>22.6e}{band / predicted:>9.4f}")
```

The band is $O(h)$ and the predicted amplitude tracks it to within a constant factor of order 1.
Compare Stormer-Verlet, whose band is $O(h^2)$ because its leading modified term is one order
higher.

### 2.5 Time reversibility

A method is **time reversible** if stepping forward by $h$ and then backward by $h$ returns the
starting state exactly.

**Stormer-Verlet.** The forward step is half kick, drift, half kick. Reversing $h$ gives half
negative kick, negative drift, half negative kick, which undoes each piece in reverse order. So the
composition is the identity, exactly, up to roundoff.

**Explicit Euler.** Forward: $q_{n+1} = q_n + hp_n$, $p_{n+1} = p_n + hF(q_n)$. Backward from
there: $q' = q_{n+1} - hp_{n+1} = q_n + h p_n - h(p_n + hF(q_n)) = q_n - h^2F(q_n) \ne q_n$. The
mismatch is $O(h^2)$ per attempt.

```python
force, energy = sy.pendulum()
print(f"{'h':>8}{'Verlet round trip':>22}{'Euler round trip':>21}{'h^2 |F|':>14}")
for h in (0.4, 0.2, 0.1, 0.05):
    q, p = np.asarray([0.9]), np.asarray([0.3])
    fq, fp = sy.stormer_verlet_step(force, q, p, h)
    bq, bp = sy.stormer_verlet_step(force, fq, fp, -h)
    verlet = max(abs(float(bq[0] - q[0])), abs(float(bp[0] - p[0])))
    fq, fp = sy.explicit_euler_step(force, q, p, h)
    bq, bp = sy.explicit_euler_step(force, fq, fp, -h)
    euler = max(abs(float(bq[0] - q[0])), abs(float(bp[0] - p[0])))
    print(f"{h:>8.2f}{verlet:>22.3e}{euler:>21.3e}"
          f"{h ** 2 * abs(float(force(q)[0])):>14.3e}")
```

Verlet's round trip error is at roundoff and does not shrink with $h$, because there is nothing to
shrink: the identity is exact. Euler's is $h^2\lvert F\rvert$ to three digits.

### 3.1 Yoshida's fourth order composition

Yoshida's construction composes three Verlet steps with carefully chosen sub step sizes:

$$
\Phi_h = V_{w_1h}\circ V_{w_0h}\circ V_{w_1h}, \qquad
w_1 = \frac{1}{2 - 2^{1/3}}, \qquad w_0 = 1 - 2w_1 = \frac{-2^{1/3}}{2 - 2^{1/3}}.
$$

The weights satisfy $2w_1 + w_0 = 1$ (consistency) and $2w_1^3 + w_0^3 = 0$, which cancels the
third order error term of the symmetric composition.

```python
W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
W0 = 1.0 - 2.0 * W1
print(f"w1 = {W1:.12f}, w0 = {W0:.12f}")
print(f"2 w1 + w0 = {2.0 * W1 + W0:.15f} (must be 1)")
print(f"2 w1^3 + w0^3 = {2.0 * W1 ** 3 + W0 ** 3:.3e} (must be 0)")
print(f"note w0 < 0: one of the three sub steps goes backwards in time")

def yoshida_step(force, q, p, h, _counter=None):
    """Fourth order symplectic, three Verlet sub steps."""
    for w in (W1, W0, W1):
        q, p = sy.stormer_verlet_step(force, q, p, w * h, _counter)
    return q, p

sy.STEPS["yoshida"] = (yoshida_step, 4, True)
out = sy.order_of("yoshida")
print(f"\nfitted order {out['fitted_order']:.4f}, claimed {out['claimed_order']}, "
      f"matches {out['matches']}")
area = sy.area_is_preserved("yoshida", h=0.2, corners=60)
print(f"worst |det - 1| at h = 0.2: {area['worst_departure_from_one']:.3e}, "
      f"preserves area: {area['preserves_area']}")
```

**Order 4 and symplectic**, from three second order steps and nothing else. The negative weight
$w_0$ is unavoidable: a composition of positive steps cannot exceed order 2, which is a theorem of
Suzuki, and it is why high order symplectic methods take steps backwards in time.

### 3.2 A long solar system integration

```python
force, energy, angular = sy.kepler()

def long_run(steps_per_period, periods, name="stormer verlet", e=0.2):
    q0 = np.asarray([1.0 - e, 0.0])
    p0 = np.asarray([0.0, np.sqrt((1.0 + e) / (1.0 - e))])
    n = int(steps_per_period * periods)
    run = sy.integrate(force, q0, p0, 2.0 * np.pi * periods, n, name)
    values = np.asarray(energy(run["q"], run["p"]), dtype=float)
    relative = (values - values[0]) / abs(values[0])
    trend = float(np.polyfit(run["t"], relative, 1)[0])
    return (float(np.max(relative) - np.min(relative)), trend,
            abs(trend) * 2.0 * np.pi * periods, n)

print(f"{'periods':>10}{'steps':>10}{'band':>14}{'trend/unit':>15}"
      f"{'drift over run':>17}")
for periods in (30, 300, 3000, 30000):
    band, trend, drift, n = long_run(200, periods)
    print(f"{periods:>10}{n:>10}{band:>14.4e}{trend:>15.3e}{drift:>17.4e}")
```

**The band does not grow with the length of the run**, over three orders of magnitude in run
length, and the trend stays at the roundoff level. Thirty thousand orbits at 200 steps each is six
million steps, and the energy is as accurate at the end as at the beginning.

That is the property a solar system integration needs, and it is why Stormer-Verlet at second order
is used for it rather than a higher order non symplectic method.

### 3.3 The implicit midpoint rule on a non separable Hamiltonian

The implicit midpoint rule

$$
y_{n+1} = y_n + hf\!\left(t + \tfrac{h}{2},\ \tfrac{y_n + y_{n+1}}{2}\right)
$$

is symplectic for **any** Hamiltonian, separable or not, because it is the one stage Gauss method
and Gauss methods are symplectic.

Take a non separable Hamiltonian: a charged particle in a magnetic field,
$H = \tfrac12\lVert p - A(q)\rVert^2$ with $A(q) = (-q_2/2,\ q_1/2)$, which has the cross term
$-p\cdot A(q)$.

```python
from nalib.nlsystems import newton_system
from nalib.ivp import as_state

def magnetic(t, state):
    """q' = p - A(q), p' = (dA/dq)^T (p - A(q)), with A = (-q2/2, q1/2)."""
    s = as_state(state)
    q = s[:2]
    p = s[2:]
    A = np.asarray([-0.5 * q[1], 0.5 * q[0]])
    v = p - A
    dA = np.asarray([[0.0, -0.5], [0.5, 0.0]])
    return np.concatenate([v, dA.T @ v])

def magnetic_energy(state):
    v = np.atleast_2d(np.asarray(state, dtype=float))
    A = np.stack([-0.5 * v[:, 1], 0.5 * v[:, 0]], axis=1)
    return 0.5 * np.sum((v[:, 2:] - A) ** 2, axis=1)

def midpoint_step(f, t, y, h, _counter=None):
    state = as_state(y)

    def residual(z):
        z = np.asarray(z, dtype=float)
        return z - state - h * as_state(f(t + 0.5 * h, 0.5 * (state + z)))

    answer = newton_system(residual, None, state.copy(), tol=1e-14, max_iter=40)
    return np.asarray(answer.root, dtype=float)

from nalib import ivp
from nalib.rungekutta import named_step

start = np.asarray([1.0, 0.0, 0.0, 0.6])
print(f"{'method':>18}{'t_end':>8}{'energy band':>15}{'trend per unit':>17}"
      f"{'drift / band':>14}")
bands = {}
for t_end, steps in ((400.0, 40000), (2000.0, 200000)):
    for label, step in (("implicit midpoint", midpoint_step),
                        ("rk4", named_step("rk4"))):
        run = ivp.integrate(magnetic, 0.0, start, t_end, steps, step)
        values = np.asarray(magnetic_energy(run["y"]), dtype=float)
        relative = (values - values[0]) / abs(values[0])
        band = float(np.max(relative) - np.min(relative))
        trend = float(np.polyfit(run["t"], relative, 1)[0])
        bands.setdefault(label, []).append(band)
        print(f"{label:>18}{t_end:>8.0f}{band:>15.4e}{trend:>17.3e}"
              f"{abs(trend) * t_end / max(band, 1e-300):>14.4f}")
print()
for label, values in bands.items():
    print(f"{label:>18}: band grew by {values[1] / values[0]:.2f}x when the run grew "
          f"by 5x")
```

Read the growth factors rather than the individual bands.

**RK4's band grows by exactly 5.0 when the run grows by 5**, which is what a pure trend does: the
whole band is drift, and its drift-to-band ratio is 1.0000 at both lengths.

**The implicit midpoint rule's band grows by 3.2**, sublinearly, and its drift-to-band ratio is 0.38
at 400 units. It is not perfectly flat at these lengths, because the energy of a non separable
Hamiltonian under a symplectic method oscillates on several timescales and 400 units is not long
enough to have sampled the slowest of them.

The clean statement is the one the sizes give. **The midpoint rule's band is 54 times narrower at
400 units and 84 times narrower at 2000**, so the gap widens with the run, which is the signature of
one method drifting and the other not.

All of that is on a Hamiltonian where Stormer-Verlet is **not available at all**, because
$H = \tfrac12\lVert p - A(q)\rVert^2$ is not separable. That is what the implicit methods are for,
and the cost is a nonlinear solve per step.

### 3.4 Projection onto the energy level set

```python
def energy_projected(step_fn, energy_of, gradient, target):
    """Restore the exact energy after every step, by projecting along its gradient."""
    def step(f, t, y, h, _counter=None):
        raw = as_state(step_fn(f, t, y, h, _counter))
        mu = 0.0
        for _ in range(30):
            point = raw + mu * gradient(raw)
            value = float(energy_of(point[None, :])[0]) - target
            if abs(value) < 1e-14:
                break
            slope = float(gradient(point) @ gradient(raw))
            if abs(slope) < 1e-300:
                break
            mu -= value / slope
        return raw + mu * gradient(raw)
    return step

from nalib import odesystems as od

f_sys, energy_sys = od.pendulum()

def pendulum_gradient(state):
    return np.asarray([9.81 * np.sin(float(state[0])), float(state[1])])

start = np.asarray([1.0, 0.0])
target = float(energy_sys(start[None, :])[0])
reference = ivp.integrate(f_sys, 0.0, start, 200.0, 400000, named_step("rk4"))

print(f"{'method':>22}{'energy band':>16}{'trend':>13}{'phase error':>15}")
for label, step in (("rk4", named_step("rk4")),
                    ("rk4 + projection",
                     energy_projected(named_step("rk4"), energy_sys,
                                      pendulum_gradient, target))):
    run = ivp.integrate(f_sys, 0.0, start, 200.0, 20000, step)
    values = np.asarray(energy_sys(run["y"]), dtype=float)
    relative = (values - values[0]) / abs(values[0])
    print(f"{label:>22}{float(np.max(relative) - np.min(relative)):>16.3e}"
          f"{float(np.polyfit(run['t'], relative, 1)[0]):>13.3e}"
          f"{float(np.max(np.abs(run['y'][-1] - reference['y'][-1]))):>15.3e}")
force_p, energy_p = sy.pendulum()
verlet = sy.integrate(force_p, [1.0], [0.0], 200.0, 20000, "stormer verlet")
verlet_values = np.asarray(energy_p(verlet["q"], verlet["p"]), dtype=float)
verlet_relative = (verlet_values - verlet_values[0]) / abs(verlet_values[0])
verlet_state = np.asarray([float(verlet["q"][-1, 0]), float(verlet["p"][-1, 0])])
print(f"{'stormer verlet':>22}"
      f"{float(np.max(verlet_relative) - np.min(verlet_relative)):>16.3e}"
      f"{float(np.polyfit(verlet['t'], verlet_relative, 1)[0]):>13.3e}"
      f"{float(np.max(np.abs(verlet_state - reference['y'][-1]))):>15.3e}")
```

**Projection restores the energy to machine precision and makes the phase error slightly worse.**
It corrects the invariant by moving along its gradient, which is not the direction of the true flow,
so the correction is itself an error in a different direction.

Verlet's row needs reading carefully, and it is not a point in Verlet's favour. Its energy band is
the **largest** of the three and its phase error is the **largest** too, by four orders of
magnitude. That is simply the order: Verlet is second order and RK4 is fourth, at the same step.

What Verlet has that neither of the others does is the absence of a **trend**, and at this run
length that has not yet paid for the order. Section 5 of the lesson measures the comparison at
equal cost over 500 time units, where it does.

So the three rows say three different things. **Projection buys the invariant and costs the
trajectory. Verlet buys the trend and costs the order. RK4 buys the order and costs both.**

### 3.5 A variable step method in transformed time

The obstruction of section 6 is that changing $h$ changes which modified Hamiltonian is being
solved exactly. The **Sundman transformation** avoids it by making the step uniform in a new time
variable:

$$
\frac{dt}{d\tau} = g(q, p),
$$

so that a fixed step in $\tau$ is a variable step in $t$. Applying a symplectic method to the
extended system in $\tau$ keeps the map symplectic, because the step in $\tau$ never changes.

For Kepler the classical choice is $g = r$, which slows the clock near perihelion.

```python
def sundman_kepler(tau, state):
    """The Kepler problem in Sundman time, with dt/dtau = r."""
    s = as_state(state)
    dim = (s.size - 1) // 2
    q, p, t = s[:dim], s[dim:2 * dim], s[-1]
    r = float(np.linalg.norm(q))
    return np.concatenate([r * p, -r * q / max(r, 1e-300) ** 3, [r]])

e = 0.7
q0 = np.asarray([1.0 - e, 0.0])
p0 = np.asarray([0.0, np.sqrt((1.0 + e) / (1.0 - e))])
force_k, energy_k, angular_k = sy.kepler()

print(f"{'method':>26}{'steps':>8}{'energy band':>15}{'trend':>13}{'bounded':>10}")
for label, uniform in (("Verlet, uniform t", True), ("midpoint, Sundman time", False)):
    if uniform:
        run = sy.integrate(force_k, q0, p0, 2.0 * np.pi * 30, 30000, "stormer verlet")
        values = np.asarray(energy_k(run["q"], run["p"]), dtype=float)
        times = run["t"]
    else:
        start = np.concatenate([q0, p0, [0.0]])
        run = ivp.integrate(sundman_kepler, 0.0, start, 60.0, 30000, midpoint_step)
        values = np.asarray(energy_k(run["y"][:, :2], run["y"][:, 2:4]), dtype=float)
        times = run["y"][:, 4]
    relative = (values - values[0]) / abs(values[0])
    band = float(np.max(relative) - np.min(relative))
    trend = float(np.polyfit(times, relative, 1)[0])
    span = float(times[-1] - times[0])
    print(f"{label:>26}{30000:>8}{band:>15.4e}{trend:>13.3e}"
          f"{str(abs(trend) * span < 0.25 * max(band, 1e-300)):>10}")
```

The Sundman version keeps its energy bounded because the step in $\tau$ never changed, and its band
is 67 times narrower than uniform Verlet's at the same step count. What it is doing to the step in
$t$ is worth seeing directly.

```python
start = np.concatenate([q0, p0, [0.0]])
run = ivp.integrate(sundman_kepler, 0.0, start, 60.0, 30000, midpoint_step)
times = run["y"][:, 4]
steps_in_t = np.diff(times)
radii = np.linalg.norm(run["y"][:-1, :2], axis=1)
print(f"smallest step in t {float(np.min(steps_in_t)):.6f} at r = "
      f"{float(radii[int(np.argmin(steps_in_t))]):.4f}")
print(f"largest step in t  {float(np.max(steps_in_t)):.6f} at r = "
      f"{float(radii[int(np.argmax(steps_in_t))]):.4f}")
print(f"ratio {float(np.max(steps_in_t) / np.min(steps_in_t)):.2f}, "
      f"and (1+e)/(1-e) = {(1 + e) / (1 - e):.2f}")
```

**The step in $t$ is shortest at perihelion and longest at aphelion**, in the ratio
$(1+e)/(1-e)$ exactly, because $dt/d\tau = r$ and $r$ runs from $1-e$ to $1+e$. That is the
adaptivity, and it came from the transformation rather than from an error estimate.

Its limitation is exactly that. $g$ must be chosen in advance from knowledge of the problem, and
$g = r$ is the right choice for Kepler because the difficulty is known to scale with $r$. There is
no general recipe, and an error controller cannot supply one without reintroducing the
irregularity that section 6 measures.

### 4.1 The band against the step for symplectic Euler

```python
force, energy = sy.pendulum()
print(f"{'h':>10}{'band':>16}{'ratio per halving':>20}")
last = None
for h in (0.16, 0.08, 0.04, 0.02, 0.01):
    steps = int(200.0 / h)
    run = sy.integrate(force, [1.0], [0.0], 200.0, steps, "symplectic euler")
    values = np.asarray(energy(run["q"], run["p"]), dtype=float)
    band = float(np.max(values) - np.min(values))
    print(f"{h:>10.4f}{band:>16.6e}" + ("" if last is None else f"{last / band:>20.4f}"))
    last = band
```

The ratio is 2 per halving, so the band is $O(h)$ and **not** $O(h^2)$. That is symplectic Euler's
order, and it confirms that the band's exponent is the method's order rather than a universal 2.

Compare Stormer-Verlet, whose ratio is 4 (lesson 74 section 4), and Yoshida's method from exercise
3.1, whose should be 16.

```python
print(f"{'method':>18}{'order':>7}{'band ratio per halving':>26}")
for name, order in (("symplectic euler", 1), ("stormer verlet", 2), ("yoshida", 4)):
    bands = []
    for h in (0.08, 0.04, 0.02):
        run = sy.integrate(force, [1.0], [0.0], 100.0, int(100.0 / h), name)
        values = np.asarray(energy(run["q"], run["p"]), dtype=float)
        bands.append(float(np.max(values) - np.min(values)))
    ratios = [bands[i] / bands[i + 1] for i in range(len(bands) - 1)]
    print(f"{name:>18}{order:>7}{str([round(r, 2) for r in ratios]):>26}")
```

### 4.2 The phase error grows and the energy does not

```python
force, energy, closed_form = sy.harmonic_oscillator()
print(f"{'t':>8}{'energy band so far':>22}{'phase error':>16}{'error / t':>13}")
h = 0.05
steps = int(400.0 / h)
run = sy.integrate(force, [1.0], [0.0], 400.0, steps, "stormer verlet")
values = np.asarray(energy(run["q"], run["p"]), dtype=float)
for t in (25.0, 50.0, 100.0, 200.0, 400.0):
    i = int(t / 400.0 * steps)
    want_q, want_p = closed_form(float(run["t"][i]), 1.0, 0.0)
    phase = abs(float(run["q"][i, 0]) - want_q)
    band = float(np.max(values[:i + 1]) - np.min(values[:i + 1]))
    print(f"{t:>8.0f}{band:>22.6e}{phase:>16.6e}{phase / t:>13.3e}")
```

**The band settles within the first few periods and stops growing; the phase error grows linearly
in $t$**, as the fourth column's near constancy shows.

That is the honest description of what a symplectic method buys. The **shape** of the orbit and the
energy on it are held forever; **where on the orbit** the particle is drifts steadily. For a solar
system integration that means the orbits are right and the planets' positions along them are not,
which is exactly the trade astronomers make.

### 4.3 How much irregularity breaks the bound

```python
def band_and_trend(policy, t_end=400.0, base=0.01, jitter=1.0, seed=42):
    """Verlet on the pendulum with a step drawn from a chosen amount of randomness."""
    force, energy = sy.pendulum()
    rng = np.random.default_rng(seed)
    q, p, t = np.asarray([1.0]), np.asarray([0.0]), 0.0
    ts, qs, ps = [0.0], [q.copy()], [p.copy()]
    k = 0
    while t < t_end:
        if policy == "fixed":
            h = base
        elif policy == "alternating":
            h = base * (1.0 + 0.5 * jitter * (1 if k % 2 else -1))
        else:
            h = base * (1.0 + 0.5 * jitter * float(rng.uniform(-1.0, 1.0)))
        h = min(h, t_end - t)
        if h <= 0.0:
            break
        q, p = sy.stormer_verlet_step(force, q, p, h)
        t += h
        k += 1
        ts.append(t)
        qs.append(q.copy())
        ps.append(p.copy())
    times = np.asarray(ts)
    values = np.asarray(energy(np.stack(qs), np.stack(ps)), dtype=float)
    relative = (values - values[0]) / abs(values[0])
    band = float(np.max(relative) - np.min(relative))
    trend = float(np.polyfit(times, relative, 1)[0])
    return band, trend, abs(trend) * t_end

print(f"{'jitter':>8}{'policy':>14}{'band':>14}{'drift over run':>17}"
      f"{'drift / band':>15}{'bounded':>10}")
for jitter in (0.0, 0.1, 0.5, 1.0, 1.8):
    for policy in ("alternating", "random"):
        band, trend, drift = band_and_trend(policy, jitter=jitter)
        ratio = drift / max(band, 1e-300)
        print(f"{jitter:>8.2f}{policy:>14}{band:>14.4e}{drift:>17.4e}"
              f"{ratio:>15.4f}{str(ratio < 0.25):>10}")
```

**The alternating policy stays bounded at every level of jitter**, including the largest, because a
repeating two step pattern is itself a symplectic map with its own modified Hamiltonian, however
different the two steps are.

**The random policy drifts at the smallest jitter tried.** At 10 per cent the drift is already 26
per cent of the band, and by 50 per cent it is 54 per cent. There is no small amount of randomness
that is safe; there is only an amount small enough that the drift has not become visible over the
run length measured.

The last row reads bounded and should not be trusted. At 180 per cent jitter the band has widened
to $1.5\times10^{-3}$ while the drift stayed near $3\times10^{-4}$, so the **ratio** fell even
though the drift did not. A ratio test against a widening band is exactly the kind of measurement
that needs a second look, and the drift column is the one to read.

### 5.1 The asymptotic series

The backward error theorem says: for a symplectic method of order $p$, there exist functions
$H_j$ such that the truncated modified Hamiltonian

$$
\tilde H_N = H + h^pH_p + \dots + h^{N}H_{N}
$$

satisfies: the numerical map differs from the exact $h$-flow of $\tilde H_N$ by $O(h^{N+1})$.

**The series does not converge.** The $H_j$ generally grow like $j!$, so for fixed $h$ the terms
eventually increase and no limit exists. What is available instead is an **optimal truncation**: cut
the series at $N \approx c/h$ where the terms are smallest, and the remainder is exponentially
small, $O(e^{-c/h})$.

**The implication for very long integrations** is precise. The energy stays within $O(h^p)$ of its
initial value for times of order $e^{c/h}$, and beyond that nothing is guaranteed. For a typical
$h$ that time is astronomically long, longer than any computation, which is why the practical
statement "the energy is bounded forever" is safe.

```python
import math

print(f"{'h':>8}{'e^(c/h) with c = 1':>22}{'in steps':>16}")
for h in (0.5, 0.2, 0.1, 0.05):
    horizon = math.exp(1.0 / h)
    print(f"{h:>8.3f}{horizon:>22.4e}{horizon / h:>16.4e}")
print()
print("even at h = 0.05 the guaranteed horizon is 5e8 time units, or 1e10 steps;")
print("the exercise 3.2 run of 100000 orbits was 6e5 time units, well inside it")
```

The second consequence is subtler and worth naming. **The theorem does not say the energy error is
$O(h^p)$ for all time; it says so up to exponentially small corrections.** A measurement that ran
long enough to see the exponentially small term would find a slow drift, and no run in this lesson
is long enough.

### 5.2 Why RK4 cannot be symplectic

A Runge-Kutta method is symplectic **exactly when**

$$
b_ia_{ij} + b_ja_{ji} - b_ib_j = 0 \qquad\text{for all } i, j,
$$

the same condition as conserving quadratic invariants (lesson 73 exercise 2.4), because the
symplectic form $\omega(u,v) = u^{\mathsf T}Jv$ is a quadratic invariant of the variational
equation.

**The obstruction for an explicit method.** Take $i = j = s$, the last stage. Since $A$ is strictly
lower triangular, $a_{ss} = 0$, so the condition reads

$$
2b_sa_{ss} - b_s^2 = -b_s^2 = 0 \quad\Longrightarrow\quad b_s = 0 .
$$

The last stage must have zero weight. Now take $i = j = s-1$: the same argument gives $b_{s-1} = 0$,
and by downward induction **every** $b_i = 0$, contradicting $\sum b_i = 1$.

So no explicit Runge-Kutta method of any order is symplectic. Not RK4, and not anything else with a
strictly lower triangular $A$.

```python
from nalib import rungekutta as rk

print(f"{'method':>16}{'worst |b_i a_ij + b_j a_ji - b_i b_j|':>40}"
      f"{'diagonal terms -b_i^2':>24}")
for name in ("euler", "heun", "kutta third", "rk4", "three eighths"):
    A, b, c, order = rk.tableau(name)
    M = np.outer(b, np.ones(b.size)) * A
    check = M + M.T - np.outer(b, b)
    diagonal = float(np.max(np.abs(np.diag(check) + b ** 2)))
    print(f"{name:>16}{float(np.max(np.abs(check))):>40.6f}{diagonal:>24.2e}")
print()
print("the diagonal of the condition is exactly -b_i^2 for every explicit method,")
print("so satisfying it would force every weight to zero")
```

The last column is zero for every method, confirming the algebra: the diagonal entries of the
symplecticity residual are exactly $-b_i^2$, and they cannot vanish for a consistent method.

### 5.3 Symplecticity against exact energy conservation

**The theorem (Ge and Marsden, 1988).** A method that is symplectic, conserves the Hamiltonian
exactly, and depends smoothly on $h$ must be the exact flow of the Hamiltonian, up to a
reparameterisation of time. In particular no such method exists for a general $H$ that is
computable in finitely many operations.

**Sketch.** A symplectic map that conserves $H$ preserves the level sets of $H$ and the symplectic
form on them. For a generic Hamiltonian the only such maps are the flows of $H$ itself, because the
level set carries the flow direction as its only symplectic vector field commuting with the
constraint. So the method would have to *be* the flow, which it is not.

**Which one the field chose, and why.** Symplecticity.

- The bound on the energy that symplecticity gives is $O(h^p)$ and **flat in time**, which is what
  a long integration needs. Exact energy conservation gives nothing about the other invariants or
  about the phase.
- Exercise 3.4 measures the alternative directly: projecting onto the energy level set restores the
  energy to machine precision and makes the phase error **worse**, and gives up the symplectic
  structure in the process.
- Symplecticity is preserved under composition, so higher order methods can be built from lower
  order ones (exercise 3.1). Energy conservation is not composable in the same way.
- Symplecticity implies conservation of **all** the Poincare invariants and the phase space volume,
  not only one scalar.

```python
force, energy = sy.pendulum()
print("what each choice actually delivers over 200 time units at h = 0.01:\n")
print(f"{'method':>26}{'energy band':>15}{'energy trend':>15}"
      f"{'phase error':>14}{'area error':>13}")
run = sy.integrate(force, [1.0], [0.0], 200.0, 20000, "stormer verlet")
values = np.asarray(energy(run["q"], run["p"]), dtype=float)
relative = (values - values[0]) / abs(values[0])
reference = ivp.integrate(f_sys, 0.0, np.asarray([1.0, 0.0]), 200.0, 400000,
                          named_step("rk4"))
verlet_state = np.asarray([float(run["q"][-1, 0]), float(run["p"][-1, 0])])
area = sy.area_is_preserved("stormer verlet", h=0.01, corners=40)
print(f"{'symplectic (Verlet)':>26}"
      f"{float(np.max(relative) - np.min(relative)):>15.3e}"
      f"{float(np.polyfit(run['t'], relative, 1)[0]):>15.3e}"
      f"{float(np.max(np.abs(verlet_state - reference['y'][-1]))):>14.3e}"
      f"{area['worst_departure_from_one']:>13.3e}")

projected = energy_projected(named_step("rk4"), energy_sys, pendulum_gradient, target)
run2 = ivp.integrate(f_sys, 0.0, np.asarray([1.0, 0.0]), 200.0, 20000, projected)
values2 = np.asarray(energy_sys(run2["y"]), dtype=float)
relative2 = (values2 - values2[0]) / abs(values2[0])
rng = np.random.default_rng(42)
dets = []
scale = 1e-4
for _ in range(40):
    state = np.asarray([float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0))])
    base = as_state(projected(f_sys, 0.0, state, 0.01))
    J = np.empty((state.size, state.size))
    for j in range(state.size):
        bumped = state.copy()
        bumped[j] += scale
        J[:, j] = (as_state(projected(f_sys, 0.0, bumped, 0.01)) - base) / scale
    dets.append(float(np.linalg.det(J)))
print(f"{'energy conserving':>26}"
      f"{float(np.max(relative2) - np.min(relative2)):>15.3e}"
      f"{float(np.polyfit(run2['t'], relative2, 1)[0]):>15.3e}"
      f"{float(np.max(np.abs(run2['y'][-1] - reference['y'][-1]))):>14.3e}"
      f"{float(np.median(dets)):>13.3e}")
print()
print(f"the projected map's determinant, over 40 random states:")
print(f"  median {float(np.median(dets)):.3e}, and it should be 1 for a symplectic map")
```

The table is the argument, and the last column is the sharpest part of it.

**The projected map's Jacobian determinant is essentially zero, not 1.** It has to be: the
projection sends a two dimensional neighbourhood onto the one dimensional level set
$\{E = E_0\}$, so the map is rank deficient and its determinant collapses. A symplectic map has
determinant exactly 1; this one destroys phase space area completely rather than merely perturbing
it.

So the energy conserving method wins the energy column by eleven orders of magnitude and **loses
every other column**, including the one that measures the structure itself.

For a run whose purpose is the energy that is the right trade. For a run whose purpose is the
orbit, which is nearly every long integration ever done, it is not.

---

## Lesson 75, Boundary Value Problems

### 1.1 Why a boundary value problem can have no solution

Picard-Lindelof works by **marching**: it builds the solution forward from one point, and the
contraction argument of lesson 68 gives existence and uniqueness together. Every ingredient of that
argument uses the fact that the data sits at one end.

A two point problem has no such construction. The natural statement is a **Fredholm alternative**:
for a linear problem $Ly = f$ with homogeneous boundary conditions, either the homogeneous problem
$Ly = 0$ has only the zero solution, in which case $Ly = f$ has exactly one solution for every $f$,
or the homogeneous problem has a nontrivial solution space, in which case $Ly = f$ has either none
or infinitely many depending on whether $f$ is orthogonal to it.

That is exactly the alternative for a **matrix** $Ax = b$, which is not a coincidence: after
discretisation the problem is a matrix, and the discrete alternative is the continuous one.

```python
import numpy as np
from nalib import bvp

for k in (5.0, 9.8696044011, 39.478417604):
    out = bvp.existence_can_fail(coefficients=[k], n=200)
    print(f"k = {k:>12.6f}: condition number {float(out['condition_number'][0]):.4e}, "
          f"largest |y| {float(out['largest_value'][0]):.4e}")
print()
print(f"the resonances are (m pi)^2: {np.pi ** 2:.6f}, {4 * np.pi ** 2:.6f}, "
      f"{9 * np.pi ** 2:.6f}")
```

### 1.2 Why linear shooting needs two solves

Let $u$ solve the full equation with $u(a) = \alpha$, $u'(a) = 0$, and $v$ solve the **homogeneous**
equation with $v(a) = 0$, $v'(a) = 1$. For a linear equation any solution with $y(a) = \alpha$ is

$$
y = u + s\,v
$$

for some $s$, because the difference of two solutions of the inhomogeneous equation solves the
homogeneous one, and $v$ spans the solutions vanishing at $a$.

Impose the far condition:

$$
y(b) = u(b) + s\,v(b) = \beta \quad\Longrightarrow\quad s = \frac{\beta - u(b)}{v(b)} .
$$

**One equation, one unknown, solved in closed form.** No iteration is needed because the map
$s \mapsto y(b; s)$ is affine, and two solves determine an affine function of one variable exactly.

```python
problem = bvp.textbook_linear_problem()
out = bvp.linear_shooting(problem["p"], problem["q"], problem["r"], problem["a"],
                          problem["b"], problem["alpha"], problem["beta"], 200)
print(f"iterations {out['iterations']}, slope {out['slope']:.10f}")
print(f"correction weight s = {out['correction_weight']:.10f}")
print(f"error {float(np.max(np.abs(out['y'] - problem['exact'](out['x'])))):.3e}")
print()
print("checking the affine claim directly:")
from nalib import ivp

def system(t, state):
    s = ivp.as_state(state)
    return np.asarray([s[1],
                       float(problem["p"](t)) * s[1] + float(problem["q"](t)) * s[0]
                       + float(problem["r"](t))])

values = [bvp.miss(system, problem["a"], problem["b"], problem["alpha"], slope, 400)
          for slope in (-2.0, -1.0, 0.0, 1.0, 2.0)]
print(f"  y(b) for slopes -2..2: {np.round(values, 10)}")
print(f"  second differences: {np.round(np.diff(np.diff(values)), 14)}")
```

### 1.3 The sensitivity is the homogeneous solution

From exercise 1.2, $y(b; s) = u(b) + s\,v(b)$, so

$$
\frac{\partial y(b)}{\partial s} = v(b),
$$

the homogeneous solution with slope 1 at $a$, evaluated at $b$. That is exact for a linear problem;
for a nonlinear one the same quantity is the solution of the **variational** equation
$w'' = f_y w + f_{y'}w'$ with $w(a) = 0$, $w'(a) = 1$.

**Why it is the right quantity.** An error $\delta$ in the slope produces an error $v(b)\delta$ at
the far end, so hitting the target to within $\varepsilon$ needs the slope to $\varepsilon/v(b)$.
When $v(b)$ is large, no representable slope is accurate enough.

```python
out = bvp.shooting_amplifies_the_guess()
print(f"{'rate':>6}{'v(b) measured':>18}{'sinh(lam)/lam':>18}{'agree':>8}")
import math
for r, m, c in zip(out["rate"], out["measured_sensitivity"],
                   out["closed_form_sensitivity"]):
    print(f"{r:>6.0f}{m:>18.6e}{c:>18.6e}"
          f"{str(abs(m - c) < 1e-6 * abs(c)):>8}")
```

### 1.4 The M-matrix condition

The finite difference row at node $i$ is

$$
\underbrace{\left(\frac{1}{h^2} + \frac{p_i}{2h}\right)}_{\text{sub}}y_{i-1}
+ \underbrace{\left(-\frac{2}{h^2} - q_i\right)}_{\text{diag}}y_i
+ \underbrace{\left(\frac{1}{h^2} - \frac{p_i}{2h}\right)}_{\text{sup}}y_{i+1} = r_i .
$$

Both off diagonal entries are positive exactly when $\lvert p_i\rvert/(2h) < 1/h^2$, that is

$$
\frac{h\lvert p_i\rvert}{2} < 1 .
$$

With the diagonal negative and both off diagonals positive and $q \le 0$, the matrix is an
**M-matrix**: its inverse has all non negative entries.

**What that guarantees** is the discrete maximum principle. If $r \equiv 0$, then $y = -A^{-1}(
\text{boundary terms})$ has every entry a non negative combination of the boundary values, so the
solution lies between them. Any excursion outside the boundary range is impossible.

```python
out = bvp.oscillation_when_the_cell_number_is_too_big()
print(f"{'n':>6}{'cell number':>14}{'M-matrix':>11}{'overshoot':>13}")
for n, c, m, o in zip(out["n"], out["cell_number"], out["is_an_m_matrix"],
                      out["overshoot"]):
    print(f"{n:>6}{c:>14.3f}{str(m):>11}{o:>13.4e}")
print(f"\nthe two columns agree exactly: "
      f"{out['the_m_matrix_condition_is_the_cell_number']}")
```

### 1.5 Supraconvergence

**Consistency** is how well the discrete operator approximates the differential operator, measured
by applying it to the exact solution. On an unequal grid that is $O(h)$.

**Convergence** is how well the discrete solution approximates the exact one. Here it is $O(h^2)$.

Convergence being **better than** consistency is called supraconvergence, and it is what makes
irregular meshes usable at all: a naive reading of consistency would say an irregular mesh costs an
order, and it does not.

```python
for pattern in ("alternating", "random", "uniform"):
    out = bvp.supraconvergence_on_an_unequal_grid(pattern=pattern)
    print(f"{pattern:>13}: truncation {out['truncation_order']:.3f}, "
          f"solution {out['solution_order']:.3f}, "
          f"gap {out['solution_order'] - out['truncation_order']:.3f}")
```

### 2.1 The linear shooting combination

With $u$ and $v$ as in exercise 1.2, define

$$
y = u + \frac{\beta - u(b)}{v(b)}\,v .
$$

**It solves the equation.** $u$ solves $y'' = py' + qy + r$ and $v$ solves $y'' = py' + qy$, so any
combination $u + cv$ solves the first, since the operator is linear and $v$ contributes nothing to
the $r$ term.

**It satisfies the left condition.** $y(a) = u(a) + c\,v(a) = \alpha + c\cdot 0 = \alpha$.

**It satisfies the right condition.**

$$
y(b) = u(b) + \frac{\beta - u(b)}{v(b)}v(b) = u(b) + \beta - u(b) = \beta .
$$

```python
run = bvp.linear_shooting(problem["p"], problem["q"], problem["r"], problem["a"],
                          problem["b"], problem["alpha"], problem["beta"], 400)
print(f"y(a) = {float(run['y'][0]):.16f}, wanted {problem['alpha']}")
print(f"y(b) = {float(run['y'][-1]):.16f}, wanted {problem['beta']}")
print(f"left condition exact: {float(run['y'][0]) == problem['alpha']}")
print(f"right condition to: {abs(float(run['y'][-1]) - problem['beta']):.2e}")
```

The left condition is satisfied **exactly**, because $v(a) = 0$ is exact. The right one is
satisfied to the integrator's accuracy, because $u(b)$ and $v(b)$ are computed.

### 2.2 The sensitivity of $y'' = \lambda^2 y$

The homogeneous equation is $v'' = \lambda^2 v$ with $v(0) = 0$, $v'(0) = 1$, whose solution is

$$
v(t) = \frac{\sinh(\lambda t)}{\lambda},
\qquad
v(1) = \frac{\sinh\lambda}{\lambda} \sim \frac{e^{\lambda}}{2\lambda}.
$$

**The largest usable $\lambda$.** Shooting can find the slope only to a relative accuracy of
$\varepsilon$, so the far boundary is hit to within $\varepsilon\lvert s\rvert v(1)$. Demanding one
correct digit in $y(b)$, whose scale is 1, needs

$$
\varepsilon\,\frac{e^\lambda}{2\lambda} < 0.1
\quad\Longrightarrow\quad
e^\lambda < \frac{0.2\lambda}{\varepsilon}.
$$

```python
eps = float(np.finfo(float).eps)
print(f"{'precision':>12}{'eps':>12}{'largest usable lambda':>24}")
for label, e in (("single", float(np.finfo(np.float32).eps)),
                 ("double", eps),
                 ("quad (est)", 1.9e-34)):
    lam = 1.0
    while math.sinh(lam) / lam * e < 0.1:
        lam += 0.01
    print(f"{label:>12}{e:>12.2e}{lam - 0.01:>24.2f}")
print()
out = bvp.shooting_amplifies_the_guess()
print(f"{'lambda':>8}{'sensitivity':>16}{'floor on the slope':>21}{'measured error':>17}")
for r, c, f, e in zip(out["rate"], out["closed_form_sensitivity"],
                      out["unavoidable_slope_error"], out["shooting_error"]):
    print(f"{r:>8.0f}{c:>16.4e}{f:>21.4e}{e:>17.4e}")
```

The largest usable $\lambda$ in double precision is about 38, and the sweep's failure at
$\lambda = 40$ is exactly there.

### 2.3 The three point difference on an unequal grid

Let $h_- = x_i - x_{i-1}$ and $h_+ = x_{i+1} - x_i$. Expand about $x_i$:

$$
y_{i-1} = y_i - h_-y'_i + \tfrac{h_-^2}{2}y''_i - \tfrac{h_-^3}{6}y'''_i + O(h^4),
$$
$$
y_{i+1} = y_i + h_+y'_i + \tfrac{h_+^2}{2}y''_i + \tfrac{h_+^3}{6}y'''_i + O(h^4).
$$

The second difference

$$
D_2y_i = \frac{2}{h_-+h_+}\left[\frac{y_{i-1}}{h_-}
- \left(\frac{1}{h_-}+\frac{1}{h_+}\right)y_i + \frac{y_{i+1}}{h_+}\right]
$$

is built so the $y_i$ and $y'_i$ terms cancel and the $y''_i$ term has coefficient 1. What survives
at the next order is

$$
D_2y_i - y''_i = \frac{h_+ - h_-}{3}\,y'''_i + O(h^2).
$$

**On a uniform grid $h_+ = h_-$ and the term vanishes**, leaving the familiar $O(h^2)$. On an
unequal grid it does not, and the truncation error is first order.

```python
def second_difference_at(x, values, i):
    """The three point second difference on an unequal grid, at node i."""
    hm = x[i] - x[i - 1]
    hp = x[i + 1] - x[i]
    return 2.0 / (hm + hp) * (values[i - 1] / hm
                              - (1.0 / hm + 1.0 / hp) * values[i]
                              + values[i + 1] / hp)

print(f"{'h':>10}{'h+ - h-':>12}{'measured error':>18}{'predicted':>16}{'ratio':>9}")
for h in (0.1, 0.05, 0.025, 0.0125):
    x = np.asarray([1.0 - h, 1.0, 1.0 + 1.6 * h])
    values = np.sin(x)
    got = second_difference_at(x, values, 1) + np.sin(1.0)
    predicted = (0.6 * h) / 3.0 * (-np.cos(1.0))
    print(f"{h:>10.4f}{0.6 * h:>12.4f}{got:>18.6e}{predicted:>16.6e}"
          f"{got / predicted:>9.4f}")
```

The measured error halves as the grid is refined, where a uniform grid would quarter it, and the
ratio to the predicted $(h_+ - h_-)y'''/3$ converges to 1 from below: 0.871, 0.936, 0.968, 0.984.
The gap is the $O(h^2)$ term the prediction omits, and it is halving too.

### 2.4 The cell number and the maximum principle

The condition $h\lvert p\rvert < 2$ comes from requiring both off diagonals positive (exercise 1.4).
Its consequence is worth deriving.

Write the interior system as $Ay = r$ with the boundary values moved to the right. If $A$ is an
M-matrix, $A^{-1} \ge 0$ entrywise. Take $r = 0$ and boundary values $\alpha$, $\beta$; then the
right hand side is $-\text{sub}_1\alpha\,e_1 - \text{sup}_{n-1}\beta\,e_{n-1}$, whose entries have
the **opposite** sign to the off diagonals.

With $A$'s diagonal negative, write $A = -B$ with $B$ having positive diagonal and negative off
diagonals, so $B$ is an M-matrix in the standard orientation and $B^{-1}\ge 0$. Then
$y = B^{-1}(\text{sub}_1\alpha e_1 + \text{sup}_{n-1}\beta e_{n-1})$ has every entry a non negative
combination of $\alpha$ and $\beta$, and since the row sums of $B^{-1}$ times the boundary
coefficients equal 1 (because a constant solves the problem), **$y$ is a convex combination of the
two boundary values at every node.**

That is the discrete maximum principle, and it is exactly what fails when a cell number exceeds 1.

```python
from nalib.banded import tridiagonal_matrix

layer = bvp.convection_diffusion_problem(0.01)
print(f"{'n':>6}{'cell':>9}{'min entry of the inverse':>28}{'overshoot':>13}")
for n in (10, 20, 40, 80, 160):
    run = bvp.finite_difference_linear(layer["p"], layer["q"], layer["r"],
                                       0.0, 1.0, 0.0, 1.0, n)
    A = tridiagonal_matrix(run["sub"][1:], run["diag"], run["sup"][:-1])
    inverse = np.linalg.inv(A)
    print(f"{n:>6}{1.0 / n / 0.02:>9.3f}{float(np.max(inverse)):>28.4e}"
          f"{max(0.0, float(np.max(run['y'])) - 1.0):>13.4e}")
```

The inverse has entries of the wrong sign exactly while the cell number exceeds 1, and the
overshoot appears with them.

### 2.5 The discrete resonance

The matrix for $y'' + ky = r$ with Dirichlet conditions is $D + kI$ where $D$ is the second
difference. $D$'s eigenvalues are known in closed form:

$$
\mu_j = -\frac{4}{h^2}\sin^2\!\left(\frac{j\pi h}{2}\right), \qquad j = 1, \dots, n-1,
$$

with eigenvectors $\sin(j\pi x_i)$. So $D + kI$ is singular exactly when

$$
k = \frac{4}{h^2}\sin^2\!\left(\frac{\pi h}{2}\right)
$$

for the first mode. Expanding,

$$
\frac{4}{h^2}\left(\frac{\pi h}{2} - \frac{(\pi h)^3}{48} + \dots\right)^2
= \pi^2\left(1 - \frac{(\pi h)^2}{12} + O(h^4)\right),
$$

**below $\pi^2$ by $\pi^4h^2/12$.**

```python
out = bvp.existence_can_fail()
print(f"pi^2                 = {out['resonance']:.12f}")
print(f"discrete resonance   = {out['discrete_resonance']:.12f}")
print(f"shift                = {out['shift']:.6e}")
print(f"predicted pi^4 h^2 / 12 = {out['predicted_shift']:.6e}")
print()
print(f"{'n':>7}{'shift':>16}{'predicted':>16}{'ratio':>9}")
for n in (50, 100, 200, 400, 800):
    run = bvp.existence_can_fail(coefficients=[5.0], n=n)
    print(f"{n:>7}{run['shift']:>16.6e}{run['predicted_shift']:>16.6e}"
          f"{run['shift'] / run['predicted_shift']:>9.6f}")
```

The ratio is 1 to five decimals at every $n$, and the shift falls like $h^2$.

### 3.1 Newton shooting with the variational equation

For $y'' = f(x, y, y')$ the derivative of the miss function is $w(b)$ where $w$ solves the
**variational** equation

$$
w'' = f_y(x, y, y')\,w + f_{y'}(x, y, y')\,w', \qquad w(a) = 0, \quad w'(a) = 1,
$$

integrated alongside the solution. Newton's step is then $s \leftarrow s - F(s)/w(b)$.

```python
from nalib.rungekutta import named_step

nonlinear = bvp.textbook_nonlinear_problem()

def augmented(t, state):
    """(y, y', w, w') together: the solution and its sensitivity to the initial slope."""
    y, dy, w, dw = ivp.as_state(state)
    return np.asarray([dy, nonlinear["f"](t, y, dy),
                       dw, float(nonlinear["f_y"](t, y, dy)) * w
                       + float(nonlinear["f_dy"](t, y, dy)) * dw])

def newton_shoot(guess, steps=200, tol=1e-12, max_iter=30):
    history = []
    s = float(guess)
    for _ in range(max_iter):
        run = ivp.integrate(augmented, nonlinear["a"],
                            [nonlinear["alpha"], s, 0.0, 1.0], nonlinear["b"],
                            steps, named_step("rk4"))
        miss = float(run["y"][-1, 0]) - nonlinear["beta"]
        slope = float(run["y"][-1, 2])
        history.append((s, miss, slope))
        if abs(miss) <= tol:
            break
        s -= miss / slope
    return s, history

s, history = newton_shoot(-1.0)
print(f"{'guess':>16}{'miss':>14}{'dy(b)/ds':>14}")
for g, m, d in history:
    print(f"{g:>16.10f}{m:>14.3e}{d:>14.4f}")
print(f"\nNewton: {len(history) - 1} steps, {2 * len(history)} solves"
      f" (the augmented system is twice the size)")
secant = bvp.shooting(lambda t, st: np.asarray([st[1],
                                                nonlinear["f"](t, st[0], st[1])]),
                      nonlinear["a"], nonlinear["b"], nonlinear["alpha"],
                      nonlinear["beta"], guesses=(-1.0, 0.0))
print(f"secant: {secant['iterations']} steps, {secant['solves']} solves")
```

**Newton converges in fewer iterations and each one costs twice as much**, because the augmented
system has four components instead of two. Counting evaluations of the right hand side, the two are
close, and the secant method needs no derivatives at all.

That is lesson 10's comparison arriving in a new setting: superlinear at half the cost per step
beats quadratic at full cost, when the cost per step is what dominates.

### 3.2 A mixed boundary condition

Replace $y(a) = \alpha$ by $c_1y(a) + c_2y'(a) = \gamma$. The finite difference treatment adds a
**ghost node** $x_{-1}$ and one extra equation.

Write the interior equation at $i = 0$ as well, introducing $y_{-1}$, and discretise the boundary
condition with a **centred** difference so it keeps second order:

$$
c_1y_0 + c_2\frac{y_1 - y_{-1}}{2h} = \gamma
\quad\Longrightarrow\quad
y_{-1} = y_1 + \frac{2h}{c_2}\left(c_1y_0 - \gamma\right).
$$

Substituting into the $i = 0$ row eliminates the ghost.

```python
def robin_left(p, q, r, a, b, c1, c2, gamma, beta, n):
    """Finite differences with c1 y(a) + c2 y'(a) = gamma and y(b) = beta."""
    x = np.linspace(a, b, n + 1)
    h = float(x[1] - x[0])
    unknowns = n                                   # y_0 .. y_(n-1)
    A = np.zeros((unknowns, unknowns))
    rhs = np.zeros(unknowns)
    for i in range(unknowns):
        pi, qi, ri = float(p(x[i])), float(q(x[i])), float(r(x[i]))
        sub = 1.0 / h ** 2 + pi / (2.0 * h)
        diag = -2.0 / h ** 2 - qi
        sup = 1.0 / h ** 2 - pi / (2.0 * h)
        rhs[i] = ri
        if i == 0:
            # y_(-1) = y_1 + (2h/c2)(c1 y_0 - gamma)
            A[0, 0] = diag + sub * (2.0 * h * c1 / c2)
            A[0, 1] = sup + sub
            rhs[0] += sub * (2.0 * h * gamma / c2)
        else:
            A[i, i - 1] = sub
            A[i, i] = diag
            if i + 1 < unknowns:
                A[i, i + 1] = sup
            else:
                rhs[i] -= sup * beta
    inner = np.linalg.solve(A, rhs)
    return x, np.concatenate([inner, [beta]])

# manufactured: y = sin(pi x / 2), y'' = -(pi/2)^2 y on [0, 1]
w = np.pi / 2.0
print(f"{'n':>7}{'error':>14}{'ratio':>9}")
last = None
for n in (20, 40, 80, 160, 320):
    x, y = robin_left(lambda t: 0.0 * t, lambda t: -w ** 2 + 0.0 * t,
                      lambda t: 0.0 * t, 0.0, 1.0,
                      c1=1.0, c2=1.0, gamma=w, beta=1.0, n=n)
    e = float(np.max(np.abs(y - np.sin(w * x))))
    print(f"{n:>7}{e:>14.4e}" + ("" if last is None else f"{last / e:>9.3f}"))
    last = e
```

The ratio is 4 per halving, so the centred treatment of the Robin condition keeps second order. **A
one sided first order treatment of the boundary would drop the whole solution to first order**,
which is the standard trap: the boundary is one equation out of $n$ and it still sets the order.

### 3.3 A Sturm-Liouville eigenvalue problem

$-y'' = \lambda y$ with $y(0) = y(1) = 0$ has eigenvalues $(m\pi)^2$ and eigenfunctions
$\sin(m\pi x)$. Discretising gives the matrix eigenvalue problem $-Dy = \lambda y$.

```python
from nalib.banded import second_difference

print(f"{'m':>4}{'exact (m pi)^2':>18}{'n = 50':>14}{'n = 200':>14}"
      f"{'n = 800':>14}{'error ratio':>14}")
for m in (1, 2, 3, 5, 10):
    exact_value = (m * np.pi) ** 2
    row, errors = [], []
    for n in (50, 200, 800):
        h = 1.0 / n
        # second_difference builds the positive definite form: 2/h^2 on the diagonal
        D = second_difference(n - 1, h)
        values = np.sort(np.linalg.eigvalsh(D))
        row.append(values[m - 1])
        errors.append(abs(values[m - 1] - exact_value))
    print(f"{m:>4}{exact_value:>18.6f}" + "".join(f"{v:>14.6f}" for v in row)
          + f"{errors[0] / errors[-1]:>14.1f}")
print()
print(f"{'m':>4}{'computed at n = 200':>22}{'(4/h^2) sin^2(m pi h/2)':>26}{'gap':>12}")
h = 1.0 / 200
D = second_difference(199, h)
values = np.sort(np.linalg.eigvalsh(D))
for m in (1, 2, 5, 10, 50):
    closed = 4.0 / h ** 2 * np.sin(m * np.pi * h / 2.0) ** 2
    print(f"{m:>4}{values[m - 1]:>22.8f}{closed:>26.8f}"
          f"{abs(values[m - 1] - closed):>12.2e}")
```

The computed eigenvalues match $(4/h^2)\sin^2(m\pi h/2)$ to twelve digits, which is the closed form
for the discrete operator and the check that the matrix is the one intended.

Against the **continuous** eigenvalues $(m\pi)^2$ they converge at second order, so a sixteenfold
refinement cuts the error by 256. **The higher modes are worse.** Expanding the closed form,

$$
\frac{4}{h^2}\sin^2\!\left(\frac{m\pi h}{2}\right)
= (m\pi)^2\left(1 - \frac{(m\pi h)^2}{12} + \dots\right),
$$

so the relative error is $(m\pi h)^2/12$, growing like $m^2$. At $m = 10$ it is a hundred times the
$m = 1$ error on the same grid.

That is the general rule for a discretised operator. **The low modes are accurate and the high ones
are not**, and the number of usable modes is a fixed fraction of the grid size rather than all of
them.

### 3.4 Richardson extrapolation on the finite difference solution

The error expansion for central differences on a uniform grid is
$y_h = y + c_2h^2 + c_4h^4 + \dots$, with only even powers. So

$$
\frac{4y_{h/2} - y_h}{3} = y + O(h^4).
$$

```python
def extrapolated(problem, n):
    """Solve at h and h/2 on nested grids, then combine."""
    coarse = bvp.finite_difference_linear(problem["p"], problem["q"], problem["r"],
                                          problem["a"], problem["b"],
                                          problem["alpha"], problem["beta"], n)
    fine = bvp.finite_difference_linear(problem["p"], problem["q"], problem["r"],
                                        problem["a"], problem["b"],
                                        problem["alpha"], problem["beta"], 2 * n)
    return coarse["x"], (4.0 * fine["y"][::2] - coarse["y"]) / 3.0

print(f"{'n':>7}{'plain':>14}{'ratio':>8}{'extrapolated':>16}{'ratio':>8}")
last_plain, last_rich = None, None
for n in (10, 20, 40, 80, 160):
    run = bvp.finite_difference_linear(problem["p"], problem["q"], problem["r"],
                                       problem["a"], problem["b"],
                                       problem["alpha"], problem["beta"], n)
    plain = float(np.max(np.abs(run["y"] - problem["exact"](run["x"]))))
    x, rich = extrapolated(problem, n)
    better = float(np.max(np.abs(rich - problem["exact"](x))))
    a = "" if last_plain is None else f"{last_plain / plain:.2f}"
    b = "" if last_rich is None else f"{last_rich / better:.2f}"
    print(f"{n:>7}{plain:>14.4e}{a:>8}{better:>16.4e}{b:>8}")
    last_plain, last_rich = plain, better
```

The plain column's ratio is 4 and the extrapolated column's is 16, so **the extrapolation reaches
fourth order** at the cost of one extra solve, which is $3n$ operations against $n$.

That is the cheapest fourth order method available here, and it works only because the error
expansion has no odd powers, which is a property of the **uniform** central difference and not of
the problem.

### 3.5 Deferred correction

Deferred correction estimates the leading truncation error from the computed solution and solves a
second time with it as a correction to the right hand side:

$$
A y^{(1)} = r, \qquad
A y^{(2)} = r + \tau(y^{(1)}),
$$

where $\tau$ approximates $-\frac{h^2}{12}y''''$ from differences of $y^{(1)}$.

The truncation error of the whole operator, not only of the second difference, is what has to be
corrected. Expanding both differences,

$$
\tau_i = \frac{h^2}{12}y''''_i - \frac{h^2}{6}p_i\,y'''_i + O(h^4),
$$

the second term coming from the central **first** difference in the $py'$ term. Missing it, or
getting either sign wrong, leaves the method at second order.

```python
from nalib.banded import thomas

def deferred(problem, n, use_p_term=True):
    """One deferred correction sweep on the central difference solution."""
    run = bvp.finite_difference_linear(problem["p"], problem["q"], problem["r"],
                                       problem["a"], problem["b"],
                                       problem["alpha"], problem["beta"], n)
    x, y, h = run["x"], run["y"], run["step"]
    fourth = np.zeros(x.size)
    third = np.zeros(x.size)
    for i in range(2, x.size - 2):
        fourth[i] = (y[i - 2] - 4 * y[i - 1] + 6 * y[i] - 4 * y[i + 1]
                     + y[i + 2]) / h ** 4
        third[i] = (-y[i - 2] + 2 * y[i - 1] - 2 * y[i + 1] + y[i + 2]) / (2 * h ** 3)
    for arr in (fourth, third):
        arr[:2] = arr[2]
        arr[-2:] = arr[-3]
    interior = run["interior"]
    p_here = np.broadcast_to(np.asarray(problem["p"](interior), dtype=float),
                             interior.shape)
    tau = h ** 2 / 12.0 * fourth[1:-1]
    if use_p_term:
        tau = tau - h ** 2 / 6.0 * p_here * third[1:-1]
    rhs = (np.asarray(problem["r"](interior), dtype=float) + tau).copy()
    rhs[0] -= run["sub"][0] * problem["alpha"]
    rhs[-1] -= run["sup"][-1] * problem["beta"]
    inner = thomas(run["sub"][1:], run["diag"], run["sup"][:-1], rhs)
    out = np.empty(x.size)
    out[0], out[-1] = problem["alpha"], problem["beta"]
    out[1:-1] = inner
    return x, out

print(f"{'n':>7}{'plain':>13}{'deferred, no p term':>22}{'ratio':>8}"
      f"{'deferred, full':>17}{'ratio':>8}{'Richardson':>14}{'ratio':>8}")
last = {}
for n in (20, 40, 80, 160):
    run = bvp.finite_difference_linear(problem["p"], problem["q"], problem["r"],
                                       problem["a"], problem["b"],
                                       problem["alpha"], problem["beta"], n)
    plain = float(np.max(np.abs(run["y"] - problem["exact"](run["x"]))))
    values = {}
    x, partial = deferred(problem, n, use_p_term=False)
    values["partial"] = float(np.max(np.abs(partial - problem["exact"](x))))
    x, full = deferred(problem, n, use_p_term=True)
    values["full"] = float(np.max(np.abs(full - problem["exact"](x))))
    xr, rich = extrapolated(problem, n)
    values["rich"] = float(np.max(np.abs(rich - problem["exact"](xr))))
    cells = []
    for key in ("partial", "full", "rich"):
        ratio = "" if key not in last else f"{last[key] / values[key]:.2f}"
        cells.append((values[key], ratio))
        last[key] = values[key]
    print(f"{n:>7}{plain:>13.3e}{cells[0][0]:>22.3e}{cells[0][1]:>8}"
          f"{cells[1][0]:>17.3e}{cells[1][1]:>8}{cells[2][0]:>14.3e}"
          f"{cells[2][1]:>8}")
```

Read the three ratio columns.

**Without the $p\,y'''$ term the correction buys a constant and not an order.** The ratio stays
at 4.00, so the method is still second order; the error is 40 per cent smaller than doing no
correction at all, and that is the whole gain.

That is the shape to expect. A correction that removes part of the leading truncation term leaves
the rest of it, which is still $O(h^2)$, so the order is unchanged and only the constant moves.

**With it the ratio is 24 to 30 per halving**, comfortably past fourth order's 16 and short of
sixth order's 64. The correction removes the $h^2$ term exactly and the estimate of it is itself
$O(h^2)$ accurate, so what is left is a mixture that has not settled by $n = 160$.

**Richardson gives a clean 16**, and it costs $3n$ Thomas solves against deferred correction's
$2n$. So deferred correction is cheaper and its error is smaller here, and it needs the truncation
error written out by hand, which for a general operator is real work and easy to get wrong in
exactly the way the middle column shows.

### 4.1 The largest workable $\lambda$ in three precisions

```python
def shoot_in(precision, lam, steps=2000):
    """Linear shooting on y'' = lam^2 y in a chosen precision."""
    dtype = precision
    h = dtype(1.0) / dtype(steps)
    lam = dtype(lam)

    def run(y0, dy0):
        y, dy = dtype(y0), dtype(dy0)
        for _ in range(steps):
            k1y, k1d = dy, lam * lam * y
            k2y, k2d = dy + h / 2 * k1d, lam * lam * (y + h / 2 * k1y)
            k3y, k3d = dy + h / 2 * k2d, lam * lam * (y + h / 2 * k2y)
            k4y, k4d = dy + h * k3d, lam * lam * (y + h * k3y)
            y = y + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
            dy = dy + h / 6 * (k1d + 2 * k2d + 2 * k3d + k4d)
        return y

    u_end = run(dtype(1.0), dtype(0.0))
    v_end = run(dtype(0.0), dtype(1.0))
    slope = (dtype(0.0) - u_end) / v_end
    # the error at the midpoint, where the exact solution is known
    exact_mid = math.sinh(float(lam) * 0.5) / math.sinh(float(lam))
    y, dy = dtype(1.0), slope
    for _ in range(steps // 2):
        k1y, k1d = dy, lam * lam * y
        k2y, k2d = dy + h / 2 * k1d, lam * lam * (y + h / 2 * k1y)
        k3y, k3d = dy + h / 2 * k2d, lam * lam * (y + h / 2 * k2y)
        k4y, k4d = dy + h * k3d, lam * lam * (y + h * k3y)
        y = y + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        dy = dy + h / 6 * (k1d + 2 * k2d + 2 * k3d + k4d)
    return abs(float(y) - exact_mid)

print(f"{'lambda':>8}{'single':>16}{'double':>16}")
for lam in (5, 10, 15, 20, 25, 30, 35, 40):
    with np.errstate(over="ignore", invalid="ignore"):
        s = shoot_in(np.float32, lam)
        d = shoot_in(np.float64, lam)
    print(f"{lam:>8}{s:>16.3e}{d:>16.3e}")
print()
print(f"{'precision':>12}{'eps':>12}{'largest lambda with one digit':>32}")
for label, e in (("single", float(np.finfo(np.float32).eps)),
                 ("double", float(np.finfo(float).eps))):
    lam = 1.0
    while math.sinh(lam) / lam * e < 0.1:
        lam += 0.1
    print(f"{label:>12}{e:>12.2e}{lam - 0.1:>32.1f}")
```

**Single precision fails between $\lambda = 15$ and 20 and double between 35 and 40.** The formula
$\varepsilon\sinh\lambda/\lambda < 0.1$ predicts 17.1 and 38.0, and both agree with the sweep.

**Extra precision buys $\lambda$ logarithmically.** The threshold satisfies
$e^{\lambda} \approx 0.2\lambda/\varepsilon$, so $\lambda \approx \ln(1/\varepsilon)$ up to a
logarithmic correction: $\ln(1/\varepsilon_{32}) = 16.1$ and $\ln(1/\varepsilon_{64}) = 36.0$,
matching the two thresholds directly. Quadruple precision would reach about $\lambda = 78$, and
that is the whole gain from four times the storage and twenty times the arithmetic.

Multiple shooting, by contrast, divides $\lambda$ by the number of pieces, so **ten pieces buy what
five extra decimal digits would**, at a fraction of the cost. That is why it is the fix and extra
precision is not.

### 4.2 The optimal grading power

```python
print(f"{'eps':>8}" + "".join(f"{'p=' + str(p):>12}" for p in (1, 2, 3, 4, 6, 8)))
for eps in (0.1, 0.03, 0.01, 0.003, 0.001):
    row = []
    for power in (1, 2, 3, 4, 6, 8):
        out = bvp.graded_grid_for_a_layer(viscosity=eps, n=40, power=float(power),
                                          towards="a")
        row.append(out["graded_error"])
    best = int(np.argmin(row))
    print(f"{eps:>8.3f}" + "".join(f"{v:>12.3e}" for v in row)
          + f"   best p = {(1, 2, 3, 4, 6, 8)[best]}")
```

**The optimal power grows as the layer thins**: 1, 3, 4, 4, 6 as $\varepsilon$ falls from $0.1$ to
$0.001$. The layer occupies a fraction $\varepsilon$ of the interval and a power $p$ grading puts
the first node at $n^{-p}$, so matching them wants $p \approx \ln(1/\varepsilon)/\ln n$, which at
$n = 40$ and $\varepsilon = 10^{-3}$ is 1.9 and at $\varepsilon = 0.1$ is 0.6. The measured optima
are larger, because the criterion is the error over the whole interval rather than the position of
one node.

The **Shishkin mesh** takes a different route: instead of a smooth grading it uses two uniform
pieces, a fine one of width $\sigma\varepsilon\ln n$ covering the layer and a coarse one covering
the rest, with half the nodes in each.

```python
def shishkin(eps, n, sigma=2.0):
    """A piecewise uniform Shishkin mesh for a layer at the left end."""
    tau = min(0.5, sigma * eps * np.log(n))
    left = np.linspace(0.0, tau, n // 2 + 1)
    right = np.linspace(tau, 1.0, n // 2 + 1)[1:]
    return np.concatenate([left, right])

layer = bvp.convection_diffusion_problem(0.01)
print(f"{'n':>7}{'uniform':>14}{'graded p=3':>14}{'Shishkin':>14}{'best':>12}")
for n in (16, 32, 64, 128, 256):
    uniform = bvp.finite_difference_linear(layer["p"], layer["q"], layer["r"],
                                           0.0, 1.0, 0.0, 1.0, n)
    u = float(np.max(np.abs(uniform["y"] - layer["exact"](uniform["x"]))))
    graded = bvp.graded_grid_for_a_layer(viscosity=0.01, n=n, power=3.0, towards="a")
    g = graded["graded_error"]
    mesh = shishkin(0.01, n)
    run = bvp.finite_difference_linear(layer["p"], layer["q"], layer["r"],
                                       alpha=0.0, beta=1.0, x=mesh)
    s = float(np.max(np.abs(run["y"] - layer["exact"](run["x"]))))
    best = ["uniform", "graded", "Shishkin"][int(np.argmin([u, g, s]))]
    print(f"{n:>7}{u:>14.4e}{g:>14.4e}{s:>14.4e}{best:>12}")
```

Shishkin wins at every size but the smallest, where its coarse half is still too coarse. Exercise
5.2 measures the property that makes it worth the trouble, and finds it needs the upwind scheme to
deliver.

### 4.3 The crossover against the grid size

```python
print(f"{'n':>8}{'crossover rate':>18}{'shooting error there':>24}"
      f"{'finite difference there':>26}")
for n in (100, 200, 400, 800, 1600, 3200):
    out = bvp.shooting_against_finite_differences(n=n, steps=400)
    rate = out["crossover_rate"]
    if rate is None:
        print(f"{n:>8}{'none in the sweep':>18}")
        continue
    i = list(out["rate"]).index(rate)
    print(f"{n:>8}{rate:>18.0f}{float(out['shooting_error'][i]):>24.3e}"
          f"{float(out['finite_difference_error'][i]):>26.3e}")
```

**The crossover moves to a lower rate as the grid is refined**, from 40 at $n = 100$ to 30 from
$n = 200$ onwards. It then stops moving, and that is an artefact of the sweep rather than of the
problem: the rates tried are $1, 5, 10, 20, 30, 40$, so the crossover can only be reported at one of
them and the next one down is a long way away.

The shape to expect is clear from the two error columns. **Shooting's accuracy has a floor** at
$\varepsilon\sinh\lambda/\lambda$ that no amount of work removes, and finite differences have no
floor above roundoff. So for any $\lambda$ large enough a fine enough grid wins, and the crossover
rate falls like the grid's error, which is $n^{-2}$: halving the crossover rate needs the grid's
error to fall by the factor by which shooting's floor grows, and that is exponential in $\lambda$.

Which is to say the crossover moves **very slowly**, and refining the grid is not the way to make
shooting competitive at large $\lambda$. Multiple shooting is.

### 5.1 Why supraconvergence happens

Write the discrete operator as $L_h$ and the truncation error as $\tau_i = (L_h y)_i - (Ly)_i$. The
solution error satisfies $L_h e = -\tau$, so $e = -L_h^{-1}\tau$ and the naive bound is
$\lVert e\rVert \le \lVert L_h^{-1}\rVert\,\lVert\tau\rVert$, which loses the extra order.

The sharper statement uses the **structure** of $\tau$. From exercise 2.3,

$$
\tau_i = \frac{h_{i+1} - h_i}{3}y'''_i + O(h^2)
$$

on an unequal grid. Write $g_i = \tfrac13 h_ih_{i+1}y'''_i$ and observe

$$
\frac{h_{i+1} - h_i}{3}y'''_i
= \frac{2}{h_i + h_{i+1}}\left(\frac{g_{i}}{h_{i}} - \frac{g_{i}}{h_{i+1}}\right)
\cdot\frac{h_ih_{i+1}}{2}\cdot\frac{2}{h_ih_{i+1}},
$$

so to leading order $\tau$ is a **discrete divergence** of a smooth quantity of size $O(h^2)$: it
has the form $\tau = D_h g$ with $\lVert g\rVert = O(h^2)$ and $D_h$ a first difference.

Then $e = -L_h^{-1}D_h g$, and the operator $L_h^{-1}D_h$ is **bounded**, because $L_h$ is a second
difference and $D_h$ a first, so their combination is a first order antiderivative rather than a
second. That gives $\lVert e\rVert = O(\lVert g\rVert) = O(h^2)$.

**The key point is that the first order part of $\tau$ telescopes.** Summing a difference gives a
boundary term rather than $n$ times a typical term, so the $O(h)$ part contributes $O(h)\cdot h =
O(h^2)$ overall rather than $O(h)\cdot 1$.

```python
out = bvp.supraconvergence_on_an_unequal_grid(pattern="alternating")
print(f"{'n':>7}{'truncation rms':>18}{'solution':>14}"
      f"{'solution / truncation':>24}")
for n, t, s in zip(out["n"], out["truncation_rms"], out["solution_error"]):
    print(f"{n:>7}{t:>18.4e}{s:>14.4e}{s / t:>24.6f}")
print()
print("the ratio falls like h, which is the extra order the telescoping supplies")
```

### 5.2 The Shishkin mesh

The construction, for $\varepsilon y'' + y' = 0$ with a layer of width $\varepsilon$ at $x = 0$:

**The transition point.** Put it at

$$
\tau = \min\left(\tfrac12,\ \sigma\varepsilon\ln n\right),
$$

with $\sigma$ chosen from the method's order, $\sigma = 2$ for a second order scheme.

**The mesh.** $n/2$ uniform intervals on $[0, \tau]$, of width $2\tau/n$, and $n/2$ uniform
intervals on $[\tau, 1]$, of width $2(1-\tau)/n$.

**Why $\ln n$.** The exact solution's layer term is $e^{-x/\varepsilon}$. At $x = \tau$ it equals
$n^{-\sigma}$, which is below the target accuracy $O(n^{-2})$ when $\sigma \ge 2$. So beyond $\tau$
the layer contributes nothing and the coarse mesh is enough; inside, the fine mesh resolves it.

**The bound.** $\lVert y - y_h\rVert_\infty \le Cn^{-2}\ln^2 n$ with $C$ **independent of
$\varepsilon$**, which is what a graded mesh cannot promise.

```python
def upwind_on(mesh, layer):
    """The upwind scheme on an arbitrary mesh."""
    x = np.asarray(mesh, dtype=float)
    interior = x[1:-1]
    left = np.diff(x)[:-1]
    right = np.diff(x)[1:]
    total = left + right
    d2 = np.stack([2.0 / (left * total), -2.0 / (left * right), 2.0 / (right * total)])
    pv = np.broadcast_to(np.asarray(layer["p"](interior), dtype=float), interior.shape)
    back = np.where(pv > 0.0, 1.0, 0.0)
    d1 = np.stack([-back / left,
                   (2.0 * back - 1.0) / np.where(back > 0, left, right),
                   (1.0 - back) / right])
    sub = d2[0] - pv * d1[0]
    diag = d2[1] - pv * d1[1]
    sup = d2[2] - pv * d1[2]
    rhs = np.zeros(interior.size)
    rhs[0] -= sub[0] * layer["alpha"]
    rhs[-1] -= sup[-1] * layer["beta"]
    y = np.empty(x.size)
    y[0], y[-1] = layer["alpha"], layer["beta"]
    y[1:-1] = thomas(sub[1:], diag, sup[:-1], rhs)
    return x, y

print(f"{'eps':>8}{'n':>6}{'central error':>16}{'upwind error':>15}"
      f"{'central C':>14}{'upwind C':>12}")
for eps in (0.01, 0.001, 0.0001):
    for n in (32, 128, 512):
        layer = bvp.convection_diffusion_problem(eps)
        mesh = shishkin(eps, n)
        run = bvp.finite_difference_linear(layer["p"], layer["q"], layer["r"],
                                           alpha=0.0, beta=1.0, x=mesh)
        central = float(np.max(np.abs(run["y"] - layer["exact"](run["x"]))))
        ux, uy = upwind_on(mesh, layer)
        upwind = float(np.max(np.abs(uy - layer["exact"](ux))))
        print(f"{eps:>8.4f}{n:>6}{central:>16.4e}{upwind:>15.4e}"
              f"{central * n ** 2 / np.log(n) ** 2:>14.3f}"
              f"{upwind * n / np.log(n):>12.3f}")
```

**The central difference scheme on a Shishkin mesh is not $\varepsilon$ uniform**, and the constant
column says so loudly: 0.97 at $\varepsilon = 10^{-2}$ and 1875 at $10^{-4}$, on the same grid size.

The reason is section 7's condition, which the mesh does not repeal. The **coarse** half has width
$2(1-\tau)/n \approx 0.06$ at $n = 32$, so its cell number is $0.06/(2\varepsilon) = 312$ at
$\varepsilon = 10^{-4}$, far above 1. The fine half resolves the layer and the coarse half
oscillates.

**Upwinding on the same mesh is $\varepsilon$ uniform.** Its constant is 0.63, 0.69, 0.72 across
every combination of $\varepsilon$ and $n$ tried, three orders of magnitude in the first and a
factor of 16 in the second. The bound is $Cn^{-1}\ln n$ with $C$ independent of $\varepsilon$, which
is the classical Shishkin result, and it is a **first order** bound.

So the honest summary of the construction is two claims rather than one. The mesh delivers an
$\varepsilon$ independent bound, and it delivers it **for the upwind scheme**, whose order is 1.
Getting an $\varepsilon$ independent second order bound needs a mesh and a scheme designed
together, which is the subject of the whole layer adapted literature.

**What it costs in implementation.** The mesh depends on $\varepsilon$, on $n$, and on the scheme's
order through $\sigma$, so it has to be built from knowledge of where the layer is and how wide. For
several layers, or a layer whose position is not known in advance, the construction has to be
replaced by adaptivity driven by an error estimator, which brings back all of lesson 70's
machinery.

### 5.3 A condition number for the two point problem

Define the condition number of the boundary value problem itself, independent of any method, as the
norm of the solution operator:

$$
\kappa = \sup\left\{\lVert y\rVert_\infty : \lVert r\rVert_\infty \le 1,\
\lvert\alpha\rvert, \lvert\beta\rvert \le 1\right\},
$$

which for $Ly = r$ with the given boundary data is the norm of the Green's operator plus the
boundary contributions.

```python
def problem_condition(rate, n=400):
    """The norm of the discrete solution operator, as a proxy for the problem's own."""
    layer = bvp.exponential_problem(rate)
    run = bvp.finite_difference_linear(layer["p"], layer["q"], layer["r"],
                                       layer["a"], layer["b"], 0.0, 0.0, n)
    A = tridiagonal_matrix(run["sub"][1:], run["diag"], run["sup"][:-1])
    inverse = np.linalg.inv(A)
    return float(np.max(np.sum(np.abs(inverse), axis=1)))

print(f"{'lambda':>8}{'problem condition':>21}{'shooting sensitivity':>24}"
      f"{'ratio':>16}")
for rate in (1.0, 5.0, 10.0, 20.0, 30.0, 40.0):
    kappa = problem_condition(rate)
    sensitivity = math.sinh(rate) / rate
    print(f"{rate:>8.0f}{kappa:>21.4e}{sensitivity:>24.4e}"
          f"{sensitivity / kappa:>16.4e}")
```

**The problem's own condition number is bounded and shooting's sensitivity is not.** As $\lambda$
grows, $\kappa$ settles to about $1/\lambda^2$, decreasing, while the shooting sensitivity grows
like $e^\lambda/(2\lambda)$. The ratio grows without bound.

That is the precise statement the lesson gestures at. **The boundary value problem is well
conditioned, and the initial value problem it is turned into is not.** The ill conditioning is
manufactured by the reformulation, entirely, and it is why multiple shooting works: cutting the
interval reduces the manufactured amplification without touching the problem.

The general lesson is one that recurs throughout numerical analysis. A well conditioned problem can
be turned into an ill conditioned one by a change of formulation, and the resulting difficulty is
neither the problem's fault nor the algorithm's: it is the formulation's, and the fix is to change
it rather than to work harder inside it.

---

## Lesson 76, Collocation and Finite Elements

### 1.1 The weak form of $-u'' + u = f$

Multiply by a test function $v$ with $v(0) = v(1) = 0$ and integrate over $[0,1]$:

$$
-\int_0^1 u''v\,dx + \int_0^1 uv\,dx = \int_0^1 fv\,dx .
$$

Integrate the first term by parts:

$$
-\int_0^1 u''v = \left[-u'v\right]_0^1 + \int_0^1 u'v' = \int_0^1 u'v',
$$

the boundary term vanishing because $v$ does. So the weak form is: find $u$ with $u(0) = u(1) = 0$
such that

$$
a(u, v) \equiv \int_0^1 \left(u'v' + uv\right)dx = \int_0^1 fv\,dx \equiv \ell(v)
$$

for every admissible $v$.

**The bilinear form is $a(u,v) = \int (u'v' + uv)$.** It is symmetric, $a(u,v) = a(v,u)$, and
positive definite, $a(v,v) = \int (v'^2 + v^2) > 0$ for $v \ne 0$, which is why the assembled
matrix is symmetric positive definite.

```python
import numpy as np
from nalib import femode as fe

nodes = np.linspace(0.0, 1.0, 9)
system = fe.assemble(nodes, k=lambda x: np.ones_like(x), c=lambda x: np.ones_like(x),
                     f=lambda x: np.ones_like(x))
A = fe.dense(system)[1:-1, 1:-1]
print(f"symmetric: {np.allclose(A, A.T)}")
print(f"smallest eigenvalue: {float(np.min(np.linalg.eigvalsh(A))):.6f}")
print(f"positive definite: {float(np.min(np.linalg.eigvalsh(A))) > 0.0}")
```

### 1.2 The partition of unity

The hat functions satisfy $\sum_i \phi_i(x) = 1$ for every $x$ in the interval. On an element
$[x_i, x_{i+1}]$ only $\phi_i$ and $\phi_{i+1}$ are nonzero, and they are

$$
\phi_i(x) = \frac{x_{i+1}-x}{h_i}, \qquad
\phi_{i+1}(x) = \frac{x-x_i}{h_i},
$$

whose sum is $\frac{x_{i+1}-x + x-x_i}{h_i} = 1$.

**What it buys** is that constants are represented exactly: $\sum_i c\,\phi_i(x) = c$. Combined
with the hats reproducing linear functions exactly (exercise 3.1's check), the space contains all
polynomials of degree 1, which is what makes the approximation second order in $L^2$.

Without it the space could not represent a constant, and no amount of refinement would fix that:
the error would not go to zero at all.

```python
def partition_of_unity(nodes, x):
    grid = np.asarray(nodes, dtype=float)
    return sum(fe.hat(grid, i, x) for i in range(grid.size))

rng = np.random.default_rng(42)
for n in (4, 9, 21):
    grid = np.linspace(-1.0, 2.0, n)
    grid[1:-1] += rng.uniform(-0.3, 0.3, n - 2) * (3.0 / (n - 1))
    grid = np.sort(grid)
    x = np.linspace(-1.0, 2.0, 601)
    total = partition_of_unity(grid, x)
    linear = sum(grid[i] * fe.hat(grid, i, x) for i in range(grid.size))
    print(f"n = {n:>3}: sum of hats deviates by "
          f"{float(np.max(np.abs(total - 1.0))):.2e}, "
          f"reproduces x to {float(np.max(np.abs(linear - x))):.2e}")
```

### 1.3 Why the matrix is tridiagonal and symmetric

**Tridiagonal.** $A_{ij} = a(\phi_j, \phi_i)$ is an integral of products of $\phi_i, \phi_j$ and
their derivatives. $\phi_i$ is supported on $[x_{i-1}, x_{i+1}]$ and $\phi_j$ on
$[x_{j-1}, x_{j+1}]$, and those overlap in a set of positive length only when
$\lvert i - j\rvert \le 1$. Every other entry is an integral of a function that is zero everywhere.

**Symmetric.** $a(u,v) = \int(ku'v' + cuv)$ is unchanged by swapping $u$ and $v$, so
$A_{ij} = A_{ji}$ identically, before any discretisation choice.

```python
nodes = np.linspace(0.0, 1.0, 11)
system = fe.assemble(nodes, f=lambda x: np.ones_like(x))
A = fe.dense(system)
print(f"entries outside the tridiagonal band: "
      f"{int(np.count_nonzero(A - np.triu(np.tril(A, 1), -1)))}")
print(f"asymmetry: {float(np.max(np.abs(A - A.T))):.2e}")
print(f"nonzeros: {int(np.count_nonzero(A))} out of {A.size}, "
      f"density {np.count_nonzero(A) / A.size:.3f}")
```

### 1.4 Why the two orders differ by one

The finite element solution $u_h$ is close to the true $u$ in a sense measured by two different
norms:

$$
\lVert u - u_h\rVert_{L^2} = \left(\int (u-u_h)^2\right)^{1/2}, \qquad
\lVert u - u_h\rVert_E = \left(\int k(u'-u_h')^2\right)^{1/2}.
$$

The energy norm measures the **derivative**. For a piecewise linear interpolant of a smooth
function, standard interpolation theory gives

$$
\lVert u - I_hu\rVert_{L^2} = O(h^2), \qquad
\lVert u' - (I_hu)'\rVert_{L^2} = O(h),
$$

because differentiating a piecewise linear approximation of a curve loses one power of $h$: the
slope of the chord differs from the tangent by $O(h)$ where the values differ by $O(h^2)$.

Cea's lemma (exercise 2.3) says the Galerkin solution is within a constant of the best
approximation in the **energy** norm, giving $O(h)$ there directly. The extra order in $L^2$ then
comes from a duality argument (exercise 5.2).

```python
out = fe.orders_in_two_norms()
print(f"{'n':>7}{'h':>10}{'L2':>14}{'energy':>14}{'energy / L2':>15}")
for n, h, l2, en in zip(out["n"], out["h"], out["l2_error"], out["energy_error"]):
    print(f"{n:>7}{h:>10.5f}{l2:>14.4e}{en:>14.4e}{en / l2:>15.2f}")
print(f"\nL2 order {out['l2_order']:.4f}, energy order {out['energy_order']:.4f}")
print(f"the ratio doubles each time h halves, which is the missing power of h")
```

### 1.5 Collocation with a jumping coefficient

Collocation imposes the **strong** form at points. Expanded,

$$
-(ku')' + cu = -ku'' - k'u' + cu = f .
$$

When $k$ jumps, $k'$ is a delta function at the interface, so the term $k'u'$ is not a function and
the strong form has no meaning there.

A collocation code does not notice: it evaluates $k$ at the collocation points, none of which is
the interface, and imposes $-ku'' + cu = f$. That equation has a solution and it is the wrong one.

```python
problem = fe.jumping_coefficient_problem(1.0, 100.0, 0.5)
out = fe.collocation_cannot_do_a_kink()
print(f"{'unknowns':>10}{'collocation error':>20}")
for u, e in zip(out["unknowns"], out["collocation_error"]):
    print(f"{u:>10}{e:>20.8f}")
probe = np.linspace(0.0, 1.0, 1001)
straight = float(np.max(np.abs(probe - problem["exact"](probe))))
print(f"\nthe gap between u = x and the true solution: {straight:.8f}")
print(f"collocation returns exactly that, at every degree")
```

### 2.1 The element matrices by hand

On $[x_i, x_{i+1}]$ of width $h$, with $\phi_1 = (x_{i+1}-x)/h$ and $\phi_2 = (x-x_i)/h$, the
derivatives are $-1/h$ and $1/h$, both constant.

**Stiffness**, with $k$ constant:

$$
K_{ab} = k\int_{x_i}^{x_{i+1}}\phi_a'\phi_b'\,dx
= k\,h\cdot\left(\pm\tfrac1h\right)\left(\pm\tfrac1h\right)
= \frac{k}{h}\begin{pmatrix}1&-1\\-1&1\end{pmatrix}.
$$

**Mass**, with $c$ constant. Substituting $s = (x - x_i)/h$,

$$
M_{ab} = c\,h\int_0^1\psi_a(s)\psi_b(s)\,ds, \qquad \psi_1 = 1-s,\ \psi_2 = s,
$$

and $\int_0^1(1-s)^2 = \tfrac13$, $\int_0^1 s(1-s) = \tfrac16$, $\int_0^1 s^2 = \tfrac13$, so

$$
M = ch\begin{pmatrix}\tfrac13 & \tfrac16\\ \tfrac16 & \tfrac13\end{pmatrix}
= \frac{ch}{6}\begin{pmatrix}2&1\\1&2\end{pmatrix}.
$$

```python
print(f"{'h':>8}{'k':>6}{'c':>6}{'stiffness matches':>20}{'mass matches':>16}")
for h in (0.1, 0.5, 3.0):
    for k, c in ((1.0, 1.0), (4.0, 0.25)):
        local = fe.element_matrices(2.0, 2.0 + h, k=lambda x, k=k: k + 0.0 * x,
                                    c=lambda x, c=c: c + 0.0 * x)
        want_k = (k / h) * np.asarray([[1.0, -1.0], [-1.0, 1.0]])
        want_m = (c * h / 6.0) * np.asarray([[2.0, 1.0], [1.0, 2.0]])
        print(f"{h:>8.2f}{k:>6.2f}{c:>6.2f}"
              f"{str(np.allclose(local['stiffness'], want_k, rtol=1e-13)):>20}"
              f"{str(np.allclose(local['mass'], want_m, rtol=1e-12)):>16}")
```

### 2.2 The assembled matrix on a uniform grid

Assembling the element stiffness matrices means adding element $i$'s $2\times2$ block into rows and
columns $i, i+1$. On a uniform grid with $k = 1$ every block is $\frac1h\begin{pmatrix}1&-1\\-1&1
\end{pmatrix}$, so an interior node $i$ receives $\tfrac1h$ from element $i-1$'s corner and $\tfrac1h$
from element $i$'s, giving $\tfrac2h$ on the diagonal and $-\tfrac1h$ on each off diagonal:

$$
K = \frac{1}{h}\operatorname{tridiag}(-1, 2, -1) = -h\,D_2,
$$

where $D_2$ is the second difference of lesson 75.

**On an unequal grid** the blocks are $\frac{1}{h_i}\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$ with
different $h_i$, so node $i$ gets $\frac{1}{h_{i-1}} + \frac{1}{h_i}$ on the diagonal and
$-\frac{1}{h_{i-1}}$, $-\frac{1}{h_i}$ off it. Comparing with lesson 75's unequal second
difference, which has $\frac{2}{h_{i-1}(h_{i-1}+h_i)}$ and so on, the two differ by a **row
scaling** of $\frac{h_{i-1}+h_i}{2}$: the finite element matrix is the finite difference one with
each row multiplied by the node's share of the interval.

```python
out = fe.the_stiffness_matrix_is_the_second_difference()
print(f"{'n':>7}{'|h diag - 2|':>16}{'|h offdiag + 1|':>19}")
for n, d, o in zip(out["n"], out["diagonal_gap"], out["off_diagonal_gap"]):
    print(f"{n:>7}{d:>16.1e}{o:>19.1e}")
print(f"the same matrix: {out['same_matrix']}\n")

rng = np.random.default_rng(42)
grid = np.linspace(0.0, 1.0, 9)
grid[1:-1] += rng.uniform(-0.3, 0.3, 7) / 8.0
grid = np.sort(grid)
system = fe.assemble(grid, f=lambda x: np.ones_like(x))
spacings = np.diff(grid)
print(f"{'node':>6}{'FEM diagonal':>16}{'1/h- + 1/h+':>16}"
      f"{'FD diagonal':>16}{'row scale':>14}")
from nalib import bvp
fd = bvp.finite_difference_matrix(lambda x: 0.0 * x, lambda x: 0.0 * x, grid, 0.0, 0.0)
for i in range(1, grid.size - 1):
    hm, hp = spacings[i - 1], spacings[i]
    print(f"{i:>6}{float(system['diag'][i]):>16.6f}"
          f"{1.0 / hm + 1.0 / hp:>16.6f}{-float(fd['diag'][i - 1]):>16.6f}"
          f"{float(system['diag'][i]) / -float(fd['diag'][i - 1]):>14.6f}")
print(f"\nthe row scale is (h- + h+)/2: "
      f"{np.round((spacings[:-1] + spacings[1:]) / 2.0, 6)}")
```

### 2.3 Cea's lemma

**Statement.** Let $a$ be bounded, $\lvert a(u,v)\rvert \le M\lVert u\rVert\lVert v\rVert$, and
coercive, $a(v,v) \ge \alpha\lVert v\rVert^2$. Let $u$ solve the weak problem and $u_h$ the
Galerkin problem on a subspace $V_h$. Then

$$
\lVert u - u_h\rVert \le \frac{M}{\alpha}\inf_{v_h\in V_h}\lVert u - v_h\rVert .
$$

**Proof.** Galerkin orthogonality first: $a(u, v_h) = \ell(v_h)$ and $a(u_h, v_h) = \ell(v_h)$ for
every $v_h \in V_h$, so

$$
a(u - u_h,\ v_h) = 0 \quad\text{for all } v_h\in V_h .
$$

Now for any $v_h \in V_h$,

$$
\alpha\lVert u-u_h\rVert^2 \le a(u-u_h,\ u-u_h)
= a(u-u_h,\ u-v_h) + \underbrace{a(u-u_h,\ v_h-u_h)}_{=\,0}
\le M\lVert u-u_h\rVert\,\lVert u-v_h\rVert,
$$

using $v_h - u_h \in V_h$ for the vanishing term. Dividing by $\lVert u-u_h\rVert$ and taking the
infimum over $v_h$ gives the result.

**In the energy norm** $a(v,v) = \lVert v\rVert_E^2$ exactly, so $M = \alpha = 1$ and the constant
is 1: **the Galerkin solution is the best approximation**, not merely within a constant of it.

```python
from nalib.gaussquad import rule_on

problem = fe.sine_problem(wavenumber=2)
print(f"{'n':>7}{'Galerkin energy error':>24}{'interpolant energy error':>27}"
      f"{'ratio':>9}")
for n in (9, 17, 33, 65):
    grid = np.linspace(0.0, 1.0, n)
    run = fe.galerkin(grid, problem["f"], alpha=0.0, beta=0.0)
    galerkin = fe.energy_error(grid, run["u"], problem["exact_derivative"])
    interpolant = fe.energy_error(grid, problem["exact"](grid),
                                  problem["exact_derivative"])
    print(f"{n:>7}{galerkin:>24.6e}{interpolant:>27.6e}"
          f"{galerkin / interpolant:>9.6f}")
```

The ratio is below 1 at every size: **the Galerkin solution is a better energy approximation than
the interpolant is**, which is what "best in the space" means, since the interpolant is one member
of the space and the Galerkin solution is the optimum over all of them.

### 2.4 Nodal exactness by the Green's function

For $-u'' = f$ on $(0,1)$ with $u(0) = u(1) = 0$, the solution is

$$
u(x) = \int_0^1 G(x, s)f(s)\,ds, \qquad
G(x,s) = \begin{cases}x(1-s) & x \le s\\ s(1-x) & x > s.\end{cases}
$$

**$G(\cdot, s)$ is piecewise linear with a kink at $s$.** So if $s$ is a node of the mesh,
$G(\cdot,s) \in V_h$ exactly.

Now the argument. The nodal error at node $x_i$ is

$$
u(x_i) - u_h(x_i) = a\!\left(u - u_h,\ G(\cdot, x_i)\right),
$$

because $a(w, G(\cdot,x_i)) = w(x_i)$ for any admissible $w$, which is the defining property of the
Green's function for this operator. And $G(\cdot, x_i) \in V_h$, so Galerkin orthogonality makes
the right side **zero**.

Hence $u_h(x_i) = u(x_i)$ exactly, at every node.

**What breaks it.** Adding $cu$ changes the operator, and the Green's function of $-u'' + cu$ is
built from $\sinh(\sqrt c\,x)$ rather than from straight lines. It is not piecewise linear, so it is
not in $V_h$, and the orthogonality argument has nothing to act on.

```python
out = fe.nodal_exactness()
print(f"{'n':>7}{'uniform':>15}{'irregular':>15}{'with c u':>15}")
for n, u, i, r in zip(out["n"], out["uniform_nodal_error"],
                      out["irregular_nodal_error"],
                      out["with_reaction_nodal_error"]):
    print(f"{n:>7}{u:>15.2e}{i:>15.2e}{r:>15.2e}")
print(f"\nexact on any grid: {out['exact_on_a_uniform_grid']} and "
      f"{out['exact_on_an_irregular_grid']}")
print(f"the reaction term leaves order {out['reaction_order']:.3f}")
```

The middle column is the check that the argument is right rather than a coincidence of uniform
grids: **the Green's function is piecewise linear on any mesh whose nodes include the source
point**, so the exactness survives an irregular grid unchanged.

### 2.5 Strang's first lemma

**Statement.** Let $a_h$ be the bilinear form actually computed, using a quadrature rule, and
$\ell_h$ the computed load. If $a_h$ is uniformly coercive on $V_h$, then

$$
\lVert u - u_h\rVert \le C\left[
\inf_{v_h}\left(\lVert u - v_h\rVert
+ \sup_{w_h}\frac{\lvert a(v_h,w_h) - a_h(v_h,w_h)\rvert}{\lVert w_h\rVert}\right)
+ \sup_{w_h}\frac{\lvert \ell(w_h) - \ell_h(w_h)\rvert}{\lVert w_h\rVert}\right].
$$

The first term is Cea's lemma. The second and third are the **quadrature consistency errors**, and
they enter additively rather than multiplying anything.

**The consequence.** The order is preserved as long as the quadrature errors are no larger than the
approximation error. For degree $m$ elements the approximation error in the energy norm is
$O(h^m)$, and a quadrature rule exact for polynomials of degree $2m-2$ makes the consistency terms
$O(h^m)$ too. For $m = 1$ that is degree 0: **the one point rule already suffices.**

```python
out = fe.quadrature_that_is_too_crude()
print(f"{'n':>7}{'1 point':>14}{'2 point':>14}{'exact':>14}"
      f"{'1pt / exact':>14}")
for n, a, b, c in zip(out["n"], out["one_point_error"], out["two_point_error"],
                      out["exact_error"]):
    print(f"{n:>7}{a:>14.4e}{b:>14.4e}{c:>14.4e}{a / c:>14.4f}")
print(f"\norders: 1 point {out['one_point_order']:.3f}, "
      f"2 point {out['two_point_order']:.3f}, exact {out['exact_order']:.3f}")
print(f"the ratio to exact integration is constant, so only the constant moved")
```

### 3.1 Quadratic elements

A quadratic element carries three basis functions: two at the endpoints and one at the midpoint.
On the reference interval $s \in [0,1]$ they are

$$
\psi_1 = (1-s)(1-2s), \qquad \psi_2 = 4s(1-s), \qquad \psi_3 = s(2s-1) .
$$

```python
def quadratic_galerkin(a, b, elements, f, exact=None, order=8):
    """P2 Galerkin for -u'' = f with zero boundary values."""
    edges = np.linspace(float(a), float(b), int(elements) + 1)
    nodes = np.empty(2 * int(elements) + 1)
    nodes[0::2] = edges
    nodes[1::2] = 0.5 * (edges[:-1] + edges[1:])
    total = nodes.size
    A = np.zeros((total, total))
    load = np.zeros(total)
    for e in range(int(elements)):
        lo, hi = float(edges[e]), float(edges[e + 1])
        h = hi - lo
        points, weights = rule_on(order, lo, hi)
        s = (points - lo) / h
        psi = np.stack([(1 - s) * (1 - 2 * s), 4 * s * (1 - s), s * (2 * s - 1)])
        dpsi = np.stack([(4 * s - 3) / h, (4 - 8 * s) / h, (4 * s - 1) / h])
        local = psi.shape[0]
        index = [(local - 1) * e + i for i in range(local)]
        values = np.asarray(f(points), dtype=float)
        for i in range(local):
            load[index[i]] += float(np.sum(weights * values * psi[i]))
            for j in range(local):
                A[index[i], index[j]] += float(np.sum(weights * dpsi[i] * dpsi[j]))
    inner = np.linalg.solve(A[1:-1, 1:-1], load[1:-1])
    u = np.zeros(total)
    u[1:-1] = inner
    return nodes, u, edges

def p2_errors(elements, problem):
    nodes, u, edges = quadratic_galerkin(0.0, 1.0, elements, problem["f"])
    l2, energy = 0.0, 0.0
    for e in range(elements):
        lo, hi = float(edges[e]), float(edges[e + 1])
        h = hi - lo
        points, weights = rule_on(10, lo, hi)
        s = (points - lo) / h
        psi = np.stack([(1 - s) * (1 - 2 * s), 4 * s * (1 - s), s * (2 * s - 1)])
        dpsi = np.stack([(4 * s - 3) / h, (4 - 8 * s) / h, (4 * s - 1) / h])
        local = u[[2 * e, 2 * e + 1, 2 * e + 2]]
        value = local @ psi
        slope = local @ dpsi
        l2 += float(np.sum(weights * (value - problem["exact"](points)) ** 2))
        energy += float(np.sum(weights
                               * (slope - problem["exact_derivative"](points)) ** 2))
    return np.sqrt(l2), np.sqrt(energy)

problem = fe.sine_problem(wavenumber=1)
print(f"{'elements':>10}{'L2':>14}{'ratio':>8}{'energy':>14}{'ratio':>8}")
last_l2, last_e = None, None
for elements in (4, 8, 16, 32, 64):
    l2, energy = p2_errors(elements, problem)
    a = "" if last_l2 is None else f"{last_l2 / l2:.2f}"
    b = "" if last_e is None else f"{last_e / energy:.2f}"
    print(f"{elements:>10}{l2:>14.4e}{a:>8}{energy:>14.4e}{b:>8}")
    last_l2, last_e = l2, energy
```

**Third order in $L^2$ and second in energy**, as the ratios 8 and 4 show. The pattern is the
general one: degree $m$ elements give $O(h^{m+1})$ in $L^2$ and $O(h^m)$ in energy, and the two
always differ by one.

### 3.2 Neumann and Robin conditions

A **Neumann** condition $u'(b) = g$ is called *natural* because it appears in the weak form on its
own. Redo the integration by parts without assuming $v(b) = 0$:

$$
-\int_a^b (ku')'v = \left[-ku'v\right]_a^b + \int_a^b ku'v'
= \int_a^b ku'v' - k(b)g\,v(b) + k(a)u'(a)v(a).
$$

So with $v(a) = 0$ and $v(b)$ free, the weak form gains the term $k(b)g\,v(b)$ on the right, and the
node $b$ becomes an unknown. **Nothing is imposed on the trial space at $b$.**

A **Robin** condition $ku'(b) + \sigma u(b) = g$ substitutes $ku'(b) = g - \sigma u(b)$, giving a
term $\sigma u(b)v(b)$ on the **left** and $g\,v(b)$ on the right.

```python
def galerkin_with_right_condition(nodes, f, kind, value, sigma=0.0, reaction=None,
                                  order=8):
    """P1 Galerkin with u(a) = 0 and a Dirichlet, Neumann or Robin condition at b."""
    system = fe.assemble(nodes, c=reaction, f=f, order=order)
    n = np.asarray(nodes).size
    A = fe.dense(system)
    load = np.asarray(system["load"], dtype=float).copy()
    if kind == "dirichlet":
        keep = slice(1, n - 1)
        rhs = load[keep] - A[keep, n - 1] * value
        u = np.zeros(n)
        u[-1] = value
        u[1:-1] = np.linalg.solve(A[keep, keep], rhs)
        return u
    keep = slice(1, n)
    B = A[keep, keep].copy()
    rhs = load[keep].copy()
    if kind == "neumann":
        rhs[-1] += value                      # + k(b) g v(b)
    else:
        B[-1, -1] += sigma                    # + sigma u(b) v(b)
        rhs[-1] += value
    u = np.zeros(n)
    u[1:] = np.linalg.solve(B, rhs)
    return u

# A reaction term is essential here. With -u'' = f the P1 answer is nodally exact
# (exercise 2.4), so every nodal error would be roundoff and no order could be measured.
# Manufactured: u = sin(pi x / 2) solving -u'' + u = ((pi/2)^2 + 1) u on [0, 1].
w = np.pi / 2.0
one = lambda x: np.ones_like(np.asarray(x, dtype=float))
source = lambda x: (w ** 2 + 1.0) * np.sin(w * np.asarray(x, dtype=float))
print(f"{'n':>7}{'Dirichlet':>14}{'ratio':>8}{'Neumann':>14}{'ratio':>8}"
      f"{'Robin':>14}{'ratio':>8}")
last = {}
for n in (9, 17, 33, 65, 129):
    grid = np.linspace(0.0, 1.0, n)
    want = np.sin(w * grid)
    cells = []
    for kind, value, sigma in (("dirichlet", 1.0, 0.0),
                               ("neumann", 0.0, 0.0),
                               ("robin", 3.0, 3.0)):
        u = galerkin_with_right_condition(grid, source, kind, value, sigma,
                                          reaction=one)
        e = float(np.max(np.abs(u - want)))
        ratio = "" if kind not in last else f"{last[kind] / e:.2f}"
        last[kind] = e
        cells.append((e, ratio))
    print(f"{n:>7}" + "".join(f"{v:>14.4e}{r:>8}" for v, r in cells))
```

All three converge at second order. **The Neumann condition needed no equation at all**, only the
extra boundary term and one more unknown, which is what "natural" means. The Robin condition needed
one entry added to the matrix, which is why it is barely more work.

The reaction term in the comment is not decoration. Without it the answer is nodally exact by
exercise 2.4 and every column reads $10^{-15}$, which measures the arithmetic rather than the
boundary treatment.

### 3.3 An a posteriori estimator and an adaptive mesh

The standard residual estimator for $-u'' = f$ with P1 elements is

$$
\eta_e^2 = h_e^2\int_e f^2\,dx + \tfrac12 h_e\left[\!\left[u_h'\right]\!\right]^2
$$

summed over the element's two endpoints, where $[\![u_h']\!]$ is the jump in the slope across a
node. Refine the elements with the largest $\eta_e$.

```python
def estimate(nodes, u, f, order=8):
    """Residual estimator per element: interior residual plus slope jumps."""
    x = np.asarray(nodes, dtype=float)
    slopes = np.diff(u) / np.diff(x)
    jumps = np.zeros(x.size)
    jumps[1:-1] = np.abs(np.diff(slopes))
    eta = np.empty(x.size - 1)
    for e in range(x.size - 1):
        h = float(x[e + 1] - x[e])
        points, weights = rule_on(order, float(x[e]), float(x[e + 1]))
        interior = h ** 2 * float(np.sum(weights * np.asarray(f(points),
                                                              dtype=float) ** 2))
        eta[e] = np.sqrt(interior + 0.5 * h * (jumps[e] ** 2 + jumps[e + 1] ** 2))
    return eta

def peaked(x):
    """A source concentrated near x = 0.3, so the solution has a local feature."""
    v = np.asarray(x, dtype=float)
    return 400.0 * np.exp(-2000.0 * (v - 0.3) ** 2)

reference_grid = np.linspace(0.0, 1.0, 20001)
reference = fe.galerkin(reference_grid, peaked)

def error_against_reference(nodes, u):
    probe = np.linspace(0.0, 1.0, 4001)
    got = fe.evaluate(nodes, u, probe)
    want = fe.evaluate(reference_grid, reference["u"], probe)
    return float(np.max(np.abs(got - want)))

print(f"{'unknowns':>10}{'uniform error':>16}{'adaptive error':>17}{'gain':>8}")
grid = np.linspace(0.0, 1.0, 9)
for _ in range(7):
    run = fe.galerkin(grid, peaked)
    eta = estimate(grid, run["u"], peaked)
    uniform = np.linspace(0.0, 1.0, grid.size)
    uniform_run = fe.galerkin(uniform, peaked)
    adaptive_error = error_against_reference(grid, run["u"])
    uniform_error = error_against_reference(uniform, uniform_run["u"])
    print(f"{grid.size - 2:>10}{uniform_error:>16.4e}{adaptive_error:>17.4e}"
          f"{uniform_error / max(adaptive_error, 1e-300):>8.2f}")
    cut = np.quantile(eta, 0.7)
    new = [grid[0]]
    for e in range(grid.size - 1):
        if eta[e] >= cut:
            new.append(0.5 * (grid[e] + grid[e + 1]))
        new.append(grid[e + 1])
    grid = np.asarray(new)
```

**The adaptive mesh beats the uniform one by a growing factor**, because it puts its nodes where
the peak is and the uniform mesh spreads them evenly. The estimator never sees the exact solution;
it works from the computed one and the data alone, which is what "a posteriori" means.

### 3.4 Collocation with cubic B-splines

Cubic B-splines on a uniform knot vector give a $C^2$ basis with **local support**, so collocation
with them has a banded matrix rather than a dense one.

```python
def bspline(knots, i, x, degree=3):
    """The i-th B-spline of the given degree, by the Cox-de Boor recursion."""
    t = np.asarray(knots, dtype=float)
    v = np.atleast_1d(np.asarray(x, dtype=float))
    if degree == 0:
        return np.where((t[i] <= v) & (v < t[i + 1]), 1.0, 0.0)
    left = np.zeros_like(v)
    if t[i + degree] > t[i]:
        left = (v - t[i]) / (t[i + degree] - t[i]) * bspline(t, i, v, degree - 1)
    right = np.zeros_like(v)
    if t[i + degree + 1] > t[i + 1]:
        right = ((t[i + degree + 1] - v) / (t[i + degree + 1] - t[i + 1])
                 * bspline(t, i + 1, v, degree - 1))
    return left + right

def bspline_derivative(knots, i, x, degree=3, times=1):
    """Exact derivatives, from the standard recursion on the degree."""
    t = np.asarray(knots, dtype=float)
    if times == 0:
        return bspline(t, i, x, degree)
    a = 0.0 if t[i + degree] == t[i] else degree / (t[i + degree] - t[i])
    b = 0.0 if t[i + degree + 1] == t[i + 1] else degree / (t[i + degree + 1] - t[i + 1])
    out = np.zeros_like(np.atleast_1d(np.asarray(x, dtype=float)))
    if a:
        out = out + a * bspline_derivative(t, i, x, degree - 1, times - 1)
    if b:
        out = out - b * bspline_derivative(t, i + 1, x, degree - 1, times - 1)
    return out

def bspline_collocation(f, a, b, alpha, beta, elements):
    """Collocate -u'' = f on a cubic B-spline basis, at evenly spaced interior sites."""
    h = (b - a) / elements
    knots = np.concatenate([a - 3 * h + np.arange(3) * h,
                            np.linspace(a, b, elements + 1),
                            b + h * (1 + np.arange(3))])
    count = elements + 3
    sites = np.concatenate([[a], np.linspace(a + h / 2, b - h / 2, count - 2), [b]])
    A = np.zeros((count, count))
    rhs = np.zeros(count)
    for r, s in enumerate(sites):
        point = np.asarray([s])
        for j in range(count):
            if r in (0, count - 1):
                A[r, j] = float(bspline(knots, j, point)[0])
            else:
                A[r, j] = -float(bspline_derivative(knots, j, point, times=2)[0])
        rhs[r] = alpha if r == 0 else (beta if r == count - 1 else float(f(s)))
    coefficients = np.linalg.solve(A, rhs)

    def value(query):
        v = np.atleast_1d(np.asarray(query, dtype=float))
        return sum(coefficients[j] * bspline(knots, j, v) for j in range(count))

    return value, A

problem = fe.sine_problem(wavenumber=1)
probe = np.linspace(0.02, 0.98, 201)
want = problem["exact"](probe)
print(f"{'elements':>10}{'unknowns':>10}{'error':>16}{'ratio':>8}{'bandwidth':>12}")
last = None
for elements in (8, 16, 32, 64, 128):
    value, A = bspline_collocation(problem["f"], 0.0, 1.0, 0.0, 0.0, elements)
    e = float(np.max(np.abs(value(probe) - want)))
    band = int(np.max([abs(i - j) for i in range(A.shape[0])
                       for j in range(A.shape[1]) if abs(A[i, j]) > 1e-10]))
    ratio = "" if last is None else f"{last / e:.2f}"
    print(f"{elements:>10}{A.shape[0]:>10}{e:>16.4e}{ratio:>8}{band:>12}")
    last = e
print()
out = fe.collocation_against_galerkin()
print(f"Chebyshev collocation reaches {out['collocation_reaches']:.3e}, dense matrix")
print(f"P1 Galerkin reaches          {out['galerkin_reaches']:.3e}, tridiagonal")
```

**Second order, and the bandwidth is 2.** The basis is $C^2$ and the derivatives above are exact,
so the second order is not an artefact of differentiating numerically. It is the **collocation
points**: evenly spaced sites give second order, and the fourth order version needs the two Gauss
points of each element, which changes the unknown count as well as the sites.

That is the lesson worth taking. **The basis and the collocation points are two separate choices**,
and a $C^2$ basis collocated in the wrong places is no better than hat functions.

What the B-spline version does buy is the bandwidth. Chebyshev collocation is spectral and its
matrix is **dense**; this one is banded with bandwidth 2, which is what matters in two and three
dimensions where a dense matrix is unaffordable. In one dimension the trade goes the other way and
Chebyshev wins outright.

On the jumping coefficient problem both fail for the same reason as section 8: they impose the
strong form.

### 3.5 Discontinuous Galerkin

A discontinuous Galerkin method drops the continuity requirement between elements, using a basis
supported on **one** element each, and reimposes continuity weakly through numerical fluxes.

```python
def dg_poisson(a, b, elements, f, alpha, beta, penalty=10.0, order=8):
    """Symmetric interior penalty DG for -u'' = f, with discontinuous linear elements."""
    edges = np.linspace(a, b, elements + 1)
    per = 2
    total = elements * per
    A = np.zeros((total, total))
    load = np.zeros(total)
    for e in range(elements):
        lo, hi = float(edges[e]), float(edges[e + 1])
        h = hi - lo
        points, weights = rule_on(order, lo, hi)
        s = (points - lo) / h
        psi = np.stack([1.0 - s, s])
        dpsi = np.stack([-np.ones_like(s) / h, np.ones_like(s) / h])
        for i in range(per):
            load[e * per + i] += float(np.sum(weights
                                              * np.asarray(f(points), dtype=float)
                                              * psi[i]))
            for j in range(per):
                A[e * per + i, e * per + j] += float(np.sum(weights * dpsi[i] * dpsi[j]))

    def face(jump, avg, h):
        """Add -{u'}[v] - {v'}[u] + (sigma/h)[u][v] for one face."""
        return (-np.outer(jump, avg) - np.outer(avg, jump)
                + (penalty / h) * np.outer(jump, jump))

    for f_index in range(1, elements):
        hL = float(edges[f_index] - edges[f_index - 1])
        hR = float(edges[f_index + 1] - edges[f_index])
        L, R = (f_index - 1) * per, f_index * per
        jump, avg = np.zeros(total), np.zeros(total)
        jump[L + 1], jump[R + 0] = 1.0, -1.0
        avg[L + 0], avg[L + 1] = -0.5 / hL, 0.5 / hL
        avg[R + 0] += -0.5 / hR
        avg[R + 1] += 0.5 / hR
        A += face(jump, avg, 0.5 * (hL + hR))
    h0 = float(edges[1] - edges[0])
    jump, avg = np.zeros(total), np.zeros(total)
    jump[0] = -1.0
    avg[0], avg[1] = -1.0 / h0, 1.0 / h0
    A += face(jump, avg, h0)
    load += alpha * (-avg + (penalty / h0) * jump)
    hn = float(edges[-1] - edges[-2])
    jump, avg = np.zeros(total), np.zeros(total)
    jump[total - 1] = 1.0
    avg[total - 2], avg[total - 1] = -1.0 / hn, 1.0 / hn
    A += face(jump, avg, hn)
    load += beta * (-avg + (penalty / hn) * jump)
    return edges, np.linalg.solve(A, load).reshape(elements, per), A

problem = fe.sine_problem(wavenumber=1)
print(f"{'elements':>10}{'DG unknowns':>13}{'CG unknowns':>13}{'DG error':>14}"
      f"{'ratio':>8}{'DG nonzeros':>14}{'CG nonzeros':>14}")
last = None
for elements in (8, 16, 32, 64):
    edges, values, A = dg_poisson(0.0, 1.0, elements, problem["f"], 0.0, 0.0)
    mids = 0.5 * (edges[:-1] + edges[1:])
    got = 0.5 * (values[:, 0] + values[:, 1])
    e = float(np.max(np.abs(got - problem["exact"](mids))))
    grid = np.linspace(0.0, 1.0, elements + 1)
    cg = fe.assemble(grid, f=problem["f"])
    ratio = "" if last is None else f"{last / e:.2f}"
    print(f"{elements:>10}{A.shape[0]:>13}{elements - 1:>13}{e:>14.4e}{ratio:>8}"
          f"{int(np.count_nonzero(A)):>14}"
          f"{int(np.count_nonzero(fe.dense(cg))):>14}")
    last = e
```

**Second order, as the ratio 3.99 shows**, and that is the check that the flux terms are right: an
error in the numerical flux drops the method to first order without breaking anything visible.

**DG has twice the unknowns and roughly three times the nonzeros for the same accuracy** in one
dimension, which is why nobody uses it here. Its advantages appear elsewhere: it handles
discontinuous solutions naturally, it parallelises because each element's block is independent
apart from the face terms, and it extends to hyperbolic problems where continuous Galerkin is
unstable.

The comparison in one dimension is the honest one to show. **The method that wins in Part 11 loses
here**, and knowing which property is buying what is the point.

### 4.1 The constant in Cea's lemma

```python
def full_energy_error(nodes, u, exact, exact_derivative, k=None, c=None, order=10):
    """The error in the norm a(v,v) = integral (k v'^2 + c v^2), which is the right one."""
    grid = np.asarray(nodes, dtype=float)
    values = np.asarray(u, dtype=float)
    total = 0.0
    for e in range(grid.size - 1):
        points, weights = rule_on(order, float(grid[e]), float(grid[e + 1]))
        slope = (values[e + 1] - values[e]) / (grid[e + 1] - grid[e])
        gap_d = slope - np.asarray(exact_derivative(points), dtype=float)
        gap_v = (fe.evaluate(grid, values, points)
                 - np.asarray(exact(points), dtype=float))
        kv = np.ones(points.shape) if k is None else np.asarray(k(points), dtype=float)
        cv = np.zeros(points.shape) if c is None else np.asarray(c(points), dtype=float)
        total += float(np.sum(weights * (kv * gap_d ** 2 + cv * gap_v ** 2)))
    return np.sqrt(total)

print(f"{'problem':>26}{'n':>6}{'Galerkin':>14}{'best in space':>16}{'ratio':>9}"
      f"{'is the interpolant?':>22}")
cases = [("-u'' = f, smooth", fe.sine_problem(1), None),
         ("-u'' = f, wavenumber 4", fe.sine_problem(4), None),
         ("-u'' + u = f", fe.reaction_problem(1.0, 2),
          fe.reaction_problem(1.0, 2)["c"])]
for label, problem, c in cases:
    for n in (17, 65):
        grid = np.linspace(0.0, 1.0, n)
        run = fe.galerkin(grid, problem["f"], c=c, alpha=0.0, beta=0.0)
        interpolant = problem["exact"](grid)
        galerkin = full_energy_error(grid, run["u"], problem["exact"],
                                     problem["exact_derivative"], c=c)
        best = full_energy_error(grid, interpolant, problem["exact"],
                                 problem["exact_derivative"], c=c)
        same = float(np.max(np.abs(run["u"] - interpolant))) < 1e-12
        print(f"{label:>26}{n:>6}{galerkin:>14.6e}{best:>16.6e}"
              f"{galerkin / best:>9.6f}{str(same):>22}")
```

Read the last column first. **For $-u'' = f$ the Galerkin solution and the interpolant are the same
function**, by the nodal exactness of exercise 2.4, so the ratio is exactly 1 for a reason that has
nothing to do with Cea's lemma.

The third problem separates them. There the Galerkin solution is **not** the interpolant, and its
energy norm error is smaller: the ratio dips below 1, though only by four parts in a million at
$n = 17$. That is the projection property of exercise 5.1 doing its work, and the margin is small
because the interpolant is already a very good approximation in this norm.

The $L^2$ column of exercise 5.1 shows the same comparison with a larger margin, 2 per cent rather
than $10^{-6}$, and the projection carries **no guarantee** there at all.

**The constant in Cea's lemma is 1 in the energy norm and cannot be measured any other way here.**
The lemma's $M/\alpha$ exceeds 1 only when the norm used for the interpolation estimate differs
from the energy norm, which is the situation in a general problem and not in this one.

### 4.2 The interface offset

```python
problem_base = fe.jumping_coefficient_problem(1.0, 100.0, 0.5)
n = 41
grid = np.linspace(0.0, 1.0, n)
h = float(grid[1] - grid[0])
print(f"{'offset (fraction of h)':>24}{'error':>14}")
worst = (0.0, 0.0)
for fraction in np.linspace(0.0, 1.0, 11):
    interface = 0.5 + fraction * h
    problem = fe.jumping_coefficient_problem(1.0, 100.0, interface)
    run = fe.galerkin(grid, problem["f"], k=problem["k"], alpha=0.0, beta=1.0,
                      order=10)
    e = float(np.max(np.abs(run["u"] - problem["exact"](grid))))
    if e > worst[1]:
        worst = (fraction, e)
    print(f"{fraction:>24.2f}{e:>14.4e}")
print(f"\nworst at offset {worst[0]:.2f} of an element")
```

**The error is zero at both ends and largest in the middle**, because at offsets 0 and 1 the
interface lands on a node and the kinked solution is in the space. The curve between is the
distance from the true solution to the nearest member of the space, which is largest when the kink
is furthest from every node.

That is a sharper statement than "put a node on the interface". **Being close is worth nothing**:
the error at a tenth of an element off is already within a factor of a few of the worst case.

### 4.3 The collocation matrix's condition number

```python
problem = fe.sine_problem(1)
print(f"{'degree':>8}{'unknowns':>10}{'condition number':>20}{'ratio':>9}")
last = None
for m in (4, 8, 16, 24, 32, 40):
    out = fe.collocation(problem["f"], 0.0, 1.0, 0.0, 0.0, m)
    ratio = "" if last is None else f"{out['condition_number'] / last:.2f}"
    print(f"{m:>8}{out['unknowns']:>10}{out['condition_number']:>20.4e}{ratio:>9}")
    last = out["condition_number"]
degrees = np.asarray([4, 8, 16, 24, 32, 40], dtype=float)
conds = np.asarray([fe.collocation(problem["f"], 0.0, 1.0, 0.0, 0.0, int(m))
                    ["condition_number"] for m in degrees])
print(f"\nfitted exponent in the degree: "
      f"{float(np.polyfit(np.log(degrees), np.log(conds), 1)[0]):.3f}")
print(f"for comparison, the P1 Galerkin matrix grows like h^-2, so n^2")
```

**The condition number grows like $m^4$**, which is the square of the second derivative operator's
$m^2$, exactly as the finite difference matrix grows like $h^{-2} = n^2$ for a second order
operator.

That is the price of spectral accuracy: at $m = 40$ the condition number is $10^7$, so seven digits
are lost to conditioning while the approximation error is at $10^{-15}$. **The two floors meet
somewhere**, and the roundoff floor of section 2 is where.

### 5.1 Galerkin as an orthogonal projection

Galerkin orthogonality says $a(u - u_h, v_h) = 0$ for every $v_h \in V_h$. In the **energy inner
product** $\langle w, v\rangle_E \equiv a(w, v)$, which is a genuine inner product when $a$ is
symmetric and coercive, that statement reads

$$
\langle u - u_h,\ v_h\rangle_E = 0 \quad \text{for all } v_h \in V_h,
$$

which is exactly the definition of $u_h$ being the **orthogonal projection** of $u$ onto $V_h$ in
that inner product.

**What follows immediately.** The projection minimises the distance, so

$$
\lVert u - u_h\rVert_E = \min_{v_h\in V_h}\lVert u - v_h\rVert_E,
$$

which is Cea's lemma with constant exactly 1. And the Pythagorean identity holds:
$\lVert u\rVert_E^2 = \lVert u_h\rVert_E^2 + \lVert u - u_h\rVert_E^2$.

**What does not follow.** Nothing about the $L^2$ error. A projection is optimal in **its own**
norm and can be arbitrarily bad in another one, and there is no general reason for an energy
optimal approximation to be $L^2$ optimal.

```python
problem = fe.reaction_problem(1.0, 2)
c = problem["c"]
print(f"{'n':>7}{'energy: Galerkin':>18}{'energy: interpolant':>21}"
      f"{'ratio':>9}{'L2: Galerkin':>15}{'L2: interpolant':>18}")
for n in (9, 17, 33, 65):
    grid = np.linspace(0.0, 1.0, n)
    run = fe.galerkin(grid, problem["f"], c=c, alpha=0.0, beta=0.0)
    interpolant = problem["exact"](grid)
    g_e = full_energy_error(grid, run["u"], problem["exact"],
                            problem["exact_derivative"], c=c)
    i_e = full_energy_error(grid, interpolant, problem["exact"],
                            problem["exact_derivative"], c=c)
    print(f"{n:>7}{g_e:>18.6e}{i_e:>21.6e}{g_e / i_e:>9.6f}"
          f"{fe.l2_error(grid, run['u'], problem['exact']):>15.4e}"
          f"{fe.l2_error(grid, interpolant, problem['exact']):>18.4e}")
```

**The Galerkin solution beats the interpolant in the energy norm at every size**, which is what the
projection property requires: the interpolant is one member of $V_h$ and the projection is the best
of all of them.

In $L^2$ it does not have to, and here the two are close, because the projection is optimal in its
**own** norm and carries no guarantee in any other.

```python
grid = np.linspace(0.0, 1.0, 33)
run = fe.galerkin(grid, problem["f"], c=c, alpha=0.0, beta=0.0)

def energy_norm(nodes, values, order=10):
    """The a-norm of a finite element function, computed directly."""
    g = np.asarray(nodes, dtype=float)
    v = np.asarray(values, dtype=float)
    total = 0.0
    for e in range(g.size - 1):
        points, weights = rule_on(order, float(g[e]), float(g[e + 1]))
        slope = (v[e + 1] - v[e]) / (g[e + 1] - g[e])
        value = fe.evaluate(g, v, points)
        total += float(np.sum(weights * (slope ** 2
                                         + np.asarray(c(points), dtype=float)
                                         * value ** 2)))
    return np.sqrt(total)

points, weights = rule_on(12, 0.0, 1.0)
whole = np.sqrt(float(np.sum(weights
                             * (problem["exact_derivative"](points) ** 2
                                + np.asarray(c(points), dtype=float)
                                * problem["exact"](points) ** 2))))
computed = energy_norm(grid, run["u"])
gap = full_energy_error(grid, run["u"], problem["exact"],
                        problem["exact_derivative"], c=c)
print(f"\nthe Pythagorean identity at n = 33, each quantity computed independently:")
print(f"  |u|_E^2       = {whole ** 2:.12f}")
print(f"  |u_h|_E^2     = {computed ** 2:.12f}")
print(f"  |u - u_h|_E^2 = {gap ** 2:.12f}")
print(f"  sum of the last two = {computed ** 2 + gap ** 2:.12f}")
print(f"  residual: {abs(whole ** 2 - computed ** 2 - gap ** 2):.3e}")
```

Every quantity is computed independently and the identity holds to $10^{-9}$, which is the
quadrature's accuracy on a piecewise function. **That is the orthogonality, measured**: the error
is perpendicular to the space in the energy inner product, so the squared norms add.

### 5.2 The Aubin-Nitsche duality argument

**The setup.** Let $e = u - u_h$. Solve the **dual** problem: find $w$ with

$$
a(v, w) = \langle e, v\rangle_{L^2} \quad\text{for all admissible } v .
$$

For a symmetric $a$ the dual problem is the same as the primal one with right hand side $e$.

**The argument.** Take $v = e$:

$$
\lVert e\rVert_{L^2}^2 = a(e, w) = a(e,\ w - w_h)
$$

for any $w_h \in V_h$, by Galerkin orthogonality. Then by boundedness,

$$
\lVert e\rVert_{L^2}^2 \le M\lVert e\rVert_E\,\lVert w - w_h\rVert_E
\le M\lVert e\rVert_E\cdot Ch\lVert w\rVert_{H^2},
$$

using the interpolation estimate for $w$. Finally **elliptic regularity** gives
$\lVert w\rVert_{H^2} \le C'\lVert e\rVert_{L^2}$, and dividing through,

$$
\lVert e\rVert_{L^2} \le CMC'h\,\lVert e\rVert_E = O(h)\cdot O(h) = O(h^2).
$$

**The regularity it assumes** is the step $\lVert w\rVert_{H^2} \le C'\lVert e\rVert_{L^2}$: the
dual solution must have two derivatives in $L^2$, bounded by the data. That holds on a smooth
domain with smooth coefficients, and **fails** on a domain with a re-entrant corner, where the
solution has a singularity. There the extra order is not available and the $L^2$ order drops.

```python
print(f"{'problem':>26}{'L2 order':>11}{'energy order':>15}{'difference':>13}")
for label, wavenumber in (("smooth, wavenumber 1", 1), ("smooth, wavenumber 4", 4)):
    out = fe.orders_in_two_norms(wavenumber=wavenumber)
    print(f"{label:>26}{out['l2_order']:>11.4f}{out['energy_order']:>15.4f}"
          f"{out['l2_order'] - out['energy_order']:>13.4f}")

print()
print(f"{'a':>6}{'u in H^s for s <':>18}{'L2 order':>11}{'theory':>9}"
      f"{'energy order':>15}{'theory':>9}")
for a_exp in (1.2, 1.3, 1.5, 2.5):
    def rough_exact(x, a=a_exp):
        v = np.maximum(np.asarray(x, dtype=float), 0.0)
        return v ** a * (1.0 - v)

    def rough_derivative(x, a=a_exp):
        v = np.maximum(np.asarray(x, dtype=float), 1e-300)
        return a * v ** (a - 1.0) * (1.0 - v) - v ** a

    def rough_source(x, a=a_exp):
        v = np.maximum(np.asarray(x, dtype=float), 1e-300)
        return -(a * (a - 1.0) * v ** (a - 2.0) * (1.0 - v)
                 - 2.0 * a * v ** (a - 1.0))

    hs, l2s, ens = [], [], []
    for n in (33, 65, 129, 257, 513):
        grid = np.linspace(0.0, 1.0, n)
        run = fe.galerkin(grid, rough_source, alpha=0.0, beta=0.0, order=12)
        hs.append(1.0 / (n - 1))
        l2s.append(fe.l2_error(grid, run["u"], rough_exact, order=12))
        ens.append(fe.energy_error(grid, run["u"], rough_derivative, order=12))
    print(f"{a_exp:>6.1f}{a_exp + 0.5:>18.1f}"
          f"{float(np.polyfit(np.log(hs), np.log(l2s), 1)[0]):>11.3f}"
          f"{min(2.0, a_exp + 0.5):>9.2f}"
          f"{float(np.polyfit(np.log(hs), np.log(ens), 1)[0]):>15.3f}"
          f"{min(1.0, a_exp - 0.5):>9.2f}")
```

The smooth problems give the clean 2 and 1.

The rough family is the measurement that matters. $u = x^{a}(1-x)$ lies in $H^s$ for
$s < a + \tfrac12$, so at $a = 1.2$ it is not even in $H^2$, and the orders come out at **1.77 and
0.76** against the predicted 1.70 and 0.70. At $a = 1.3$ they are 1.83 and 0.83 against 1.80 and
0.80, and by $a = 2.5$ the solution is smooth enough and the full 2 and 1 are back.

**The regularity assumption is not a technicality.** The duality argument needs
$\lVert w\rVert_{H^2} \le C\lVert e\rVert_{L^2}$ for the dual solution, and where that fails the
extra order fails with it, by exactly the amount the regularity is short.

That is the practical face of it: **a re-entrant corner in a domain costs you an order**, and the
amount is set by the corner angle through the same $H^s$ bookkeeping.

### 5.3 Distributions

Make the claim precise. Let $k$ have a jump at $s$: $k = k_1$ on $(0,s)$ and $k_2$ on $(s,1)$.

**As a distribution**, $k' = (k_2 - k_1)\delta_s$, since for any test function $\varphi$,

$$
\langle k', \varphi\rangle = -\int_0^1 k\varphi'
= -k_1\int_0^s\varphi' - k_2\int_s^1\varphi'
= -k_1(\varphi(s) - \varphi(0)) - k_2(\varphi(1) - \varphi(s))
= (k_2-k_1)\varphi(s)
$$

for $\varphi$ vanishing at the endpoints.

**The strong form** contains $k'u'$. That is a product of a delta with a function that is
**discontinuous at the same point**, since $u' = \text{flux}/k$ jumps where $k$ does. The product of
a distribution and a discontinuous function is **not defined**: there is no way to assign a value
to $\delta_s\cdot u'$ that respects both the limit from the left and from the right.

So the strong form is not a distributional identity that happens to be hard to discretise. **It is
not an identity at all.**

**Which formulation survives.** The weak form

$$
\int_0^1 k u'v'\,dx = \int_0^1 fv\,dx
$$

never differentiates $k$. It asks for $u \in H^1_0$, and the integrand $ku'v'$ is a product of an
$L^\infty$ function with two $L^2$ functions, which is integrable. **The weak form is well defined
for any bounded measurable $k$ bounded away from zero**, and that is the class of coefficients every
real material problem lives in.

The correct strong statement, recovered from the weak one, is a **transmission condition**: $u$ is
continuous and the **flux** $ku'$ is continuous, with $-(ku')' = f$ holding on each side separately.
That is what the exact solution of section 8 satisfies, and it is what the finite element method
reproduces exactly when a node sits on the interface.

```python
problem = fe.jumping_coefficient_problem(1.0, 100.0, 0.5)
grid = np.linspace(0.0, 1.0, 9)
run = fe.galerkin(grid, problem["f"], k=problem["k"], alpha=0.0, beta=1.0, order=10)
slopes = np.diff(run["u"]) / np.diff(grid)
mids = 0.5 * (grid[:-1] + grid[1:])
conductivity = np.asarray([float(problem["k"](np.asarray([m]))[0]) for m in mids])
flux = conductivity * slopes
print(f"{'element midpoint':>18}{'k there':>10}{'slope':>14}{'flux':>16}")
for m, kk, s, q in zip(mids, conductivity, slopes, flux):
    print(f"{m:>18.4f}{kk:>10.1f}{s:>14.6f}{q:>16.10f}")
print(f"\nthe exact flux is {problem['flux']:.10f}")
print(f"the computed flux is constant to "
      f"{float(np.max(np.abs(flux - problem['flux']))):.2e}")
print(f"the slope ratio across the interface is "
      f"{float(slopes[0] / slopes[-1]):.4f}, and k2/k1 = "
      f"{problem['right'] / problem['left']:.4f}")
```

The computed flux is constant to machine precision across the jump, and the slope drops by exactly
the conductivity ratio. **The transmission conditions hold exactly**, which is the discrete version
of the statement that the weak form is the right one.

Notice what was **not** imposed. Nothing in the assembly says "make the flux continuous"; the
element integrals were computed and the linear system solved. The transmission condition is what
the weak form means, so a method built on the weak form satisfies it without being told.

---
