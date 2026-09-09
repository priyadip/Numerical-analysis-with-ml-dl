# Solutions: Part 11, Partial Differential Equations

Solutions to every exercise in lessons 77 to 83, 147 in all: 21 for lesson 77, 21 for lesson 78, 21 for lesson 79, 21 for lesson 80, 21 for lesson 81, 21 for lesson 82, 21 for lesson 83.

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

## Lesson 77, PDE Classification and Stencils

### 1.1 Classifying a degenerate case

$u_{xx} + 4u_{xy} + 4u_{yy} = 0$ has $A = 1$, $B = 4$, $C = 4$, so

$$
B^2 - 4AC = 16 - 16 = 0 .
$$

It is **parabolic**, and it is parabolic everywhere, since the coefficients are constant. So it
needs an initial state on one line plus one condition on each side boundary, and it can be marched
in the direction the missing second derivative points.

The degeneracy is visible in the principal part: $u_{xx} + 4u_{xy} + 4u_{yy} = (\partial_x +
2\partial_y)^2 u$, a perfect square. The change of variables $\xi = 2x - y$, $\eta = y$ turns it
into $u_{\eta\eta} = 0$, and the one remaining second derivative is what makes it parabolic.

```python
import numpy as np
from nalib import pdeclass as pc

print(f"discriminant {float(pc.discriminant(1.0, 4.0, 4.0)):.1f}, "
      f"type {pc.classify(1.0, 4.0, 4.0)}")
slopes = pc.characteristic_slopes(1.0, 4.0, 4.0)
print(f"characteristic slopes {slopes[0]:.4f}, {slopes[1]:.4f} (a repeated root)")
needs = pc.conditions_for(pc.classify(1.0, 4.0, 4.0))
print(f"needs: {needs['needs']}")
print(f"marched: {needs['marching']}")
```

### 1.2 Why one fraction depends on the grid and the other does not

The parabolic set of Tricomi's equation is the line $y = 0$, and a line has **zero area**. Whether
a grid contains points on it is an accident of where the grid points fall: a grid with a row at
$y = 0$ reports a parabolic fraction of $1/n$, and a grid shifted by half a spacing reports zero.
Neither number is converging to anything, because the quantity being estimated is zero.

The elliptic set is the open half plane $y > 0$, which has **positive area**. A grid samples it in
proportion to that area, so the reported fraction converges to $1/2$ as the grid is refined, and it
does not care where the grid points fall.

The rule is general: a fraction reported from a grid estimates a **measure**, and it is meaningful
only when the set has positive measure in the dimension the grid samples.

```python
for n in (21, 41, 81, 161):
    xs = np.linspace(-1.0, 1.0, n)
    on = pc.classify_on_grid(pc.tricomi_coefficients, xs, xs)["fractions"]
    shift = 1.0 / (n - 1) * 0.5
    off = pc.classify_on_grid(pc.tricomi_coefficients, xs,
                              np.linspace(-1.0 + shift, 1.0 + shift, n))["fractions"]
    print(f"n = {n:>4}: on the axis  elliptic {on['elliptic']:.4f} "
          f"parabolic {on['parabolic']:.4f}   |   off it  "
          f"elliptic {off['elliptic']:.4f} parabolic {off['parabolic']:.4f}")
print("\nthe parabolic fraction on the axis is 1/n and off it is 0; neither converges")
print("the elliptic fraction converges to 1/2 either way")
```

### 1.3 The exactness condition, and why every stencil sums to zero

A stencil with weights $w_j$ on offsets $s_j$ approximating $\mathrm{d}^k/\mathrm{d}x^k$ is exact
for $x^m$ exactly when

$$
\sum_j w_j s_j^m = m!\,[m = k] .
$$

Put $m = 0$. Then $s_j^0 = 1$ for every $j$, and the right hand side is $0!\,[0 = k] = 0$ for any
$k \ge 1$. So

$$
\sum_j w_j = 0
$$

for every stencil of every derivative of order at least one. The statement in words is that a
derivative annihilates constants, and any approximation to it that does not is wrong before you
look at anything else.

### 1.4 Sixth order on a harmonic function, second in general

Expanding the compact nine point rule gives

$$
\Delta_9 u - \Delta u = \frac{h^2}{12}\left(u_{xxxx} + 2u_{xxyy} + u_{yyyy}\right)
+ \frac{h^4}{180}\left(u_{xxxxyy} + u_{xxyyyy}\right) + O(h^6).
$$

The $h^2$ bracket is $\Delta^2 u$, the biharmonic, which vanishes for a harmonic $u$ because
$\Delta^2 u = \Delta(\Delta u) = \Delta 0 = 0$. That alone would give fourth order.

The $h^4$ bracket vanishes too, and for a separate reason. For a harmonic function $u_{yy} =
-u_{xx}$, so

$$
u_{xxyyyy} = \partial_x^2\partial_y^2 u_{yy} = \partial_x^2\partial_y^2(-u_{xx}) = -u_{xxxxyy},
$$

and the two terms cancel. With both brackets gone the leading error is $O(h^6)$.

In general neither vanishes, and the first one already makes the rule second order, with the same
$h^2$ as the five point rule and a constant that is twice as large on the test function used in
the lesson.

### 1.5 Why one boundary row costs the whole solve

The discrete problem is $Au = b$. A first order boundary condition perturbs one row of $b$ by
$O(h)$ rather than $O(h^2)$, so the perturbation to the solution is $A^{-1}$ applied to a vector
with one entry of size $O(h)$.

That single entry does not stay local. $A^{-1}$ for the second difference operator is dense, and
its entries are $O(1)$: it is the discrete Green's function, and the Green's function of $-u''$ on
$[0,1]$ is $O(1)$ everywhere. So an $O(h)$ error in one row of $b$ produces an $O(h)$ error at
**every** node, and the whole solution is first order.

The measurement in the lesson agrees: the interior is second order at every one of hundreds of
rows, and the fitted order of the solve is $0.99$.

### 2.1 The discriminant from the change of variables

Look for new variables $\xi = \xi(x,y)$, $\eta = \eta(x,y)$. The chain rule gives

$$
u_{xx} = u_{\xi\xi}\xi_x^2 + 2u_{\xi\eta}\xi_x\eta_x + u_{\eta\eta}\eta_x^2 + \cdots,
$$

and similarly for the other two, where the omitted terms involve only first derivatives. Collecting
the second order part, the new coefficients are

$$
\tilde A = A\xi_x^2 + B\xi_x\xi_y + C\xi_y^2, \qquad
\tilde C = A\eta_x^2 + B\eta_x\eta_y + C\eta_y^2,
$$

$$
\tilde B = 2A\xi_x\eta_x + B(\xi_x\eta_y + \xi_y\eta_x) + 2C\xi_y\eta_y .
$$

A direct computation gives

$$
\tilde B^2 - 4\tilde A\tilde C = \left(B^2 - 4AC\right)\left(\xi_x\eta_y - \xi_y\eta_x\right)^2 .
$$

The factor on the right is the **square** of the Jacobian of the change of variables, which is
positive for any invertible change. So the discriminant is multiplied by a positive number, and
its **sign is invariant**. That is what makes the classification a property of the equation rather
than of the coordinates it happens to be written in.

To remove the $u_{xy}$ term, choose $\xi$ and $\eta$ constant along the characteristics, whose
slopes solve $Am^2 - Bm + C = 0$. Two real slopes give two real families and the hyperbolic normal
form $u_{\xi\eta} = \cdots$; one gives the parabolic form; none gives the elliptic one.

```python
rng = np.random.default_rng(42)
worst = 0.0
for _ in range(2000):
    a, b, c = rng.normal(size=3)
    xi_x, xi_y, eta_x, eta_y = rng.normal(size=4)
    jac = xi_x * eta_y - xi_y * eta_x
    if abs(jac) < 1e-3:
        continue
    a2 = a * xi_x ** 2 + b * xi_x * xi_y + c * xi_y ** 2
    c2 = a * eta_x ** 2 + b * eta_x * eta_y + c * eta_y ** 2
    b2 = 2 * a * xi_x * eta_x + b * (xi_x * eta_y + xi_y * eta_x) + 2 * c * xi_y * eta_y
    left = b2 ** 2 - 4 * a2 * c2
    right = (b ** 2 - 4 * a * c) * jac ** 2
    worst = max(worst, abs(left - right) / max(abs(right), 1e-12))
    assert (left > 0) == (b ** 2 - 4 * a * c > 0)
print(f"largest relative gap in the identity over 2000 random changes: {worst:.3e}")
print("and the sign agreed in every one")
```

### 2.2 The truncation errors of the two Laplacians

Expand about $(x, y)$ with $h_x = h_y = h$. Along one axis,

$$
u_E + u_W = 2u + h^2u_{xx} + \frac{h^4}{12}u_{xxxx} + \frac{h^6}{360}u_{xxxxxx} + \cdots
$$

so the edge sum over both axes is

$$
4u + h^2\Delta u + \frac{h^4}{12}(u_{4x} + u_{4y}) + \frac{h^6}{360}(u_{6x} + u_{6y}).
$$

Dividing the axis part by $h^2$ gives the **five point** rule with error

$$
\Delta_5 u - \Delta u = \frac{h^2}{12}\left(u_{4x} + u_{4y}\right) + O(h^4).
$$

For the corners, the odd terms cancel by symmetry and

$$
\sum_{s,t=\pm1} u(sh, th) = 4u + 2h^2\Delta u
+ h^4\left(\frac{u_{4x}}{6} + u_{2x2y} + \frac{u_{4y}}{6}\right)
+ h^6\left(\frac{u_{6x} + u_{6y}}{180} + \frac{u_{4x2y} + u_{2x4y}}{12}\right).
$$

Now $4\times$(edge) $+$ (corner) $- 20u$ is

$$
6h^2\Delta u + h^4\left[\tfrac12(u_{4x} + u_{4y}) + u_{2x2y}\right]
+ h^6\left[\tfrac{1}{60}(u_{6x} + u_{6y}) + \tfrac{1}{12}(u_{4x2y} + u_{2x4y})\right],
$$

and dividing by $6h^2$ gives

$$
\Delta_9 u - \Delta u = \frac{h^2}{12}\Delta^2 u
+ h^4\left[\frac{u_{6x} + u_{6y}}{360} + \frac{u_{4x2y} + u_{2x4y}}{72}\right] + O(h^6).
$$

For a harmonic function $\Delta^2u = 0$ and $\nabla^6 u = 0$, so $u_{6x} + u_{6y} =
-3(u_{4x2y} + u_{2x4y})$, and the $h^4$ bracket collapses to
$\frac{1}{180}(u_{4x2y} + u_{2x4y})$, which is exercise 2.3's zero.

### 2.3 The identity that gives the sixth order

For a harmonic $u$, $u_{xx} + u_{yy} = 0$ everywhere, so $u_{yy} = -u_{xx}$ as functions, and any
derivative of that identity is another identity. Apply $\partial_x^2\partial_y^2$:

$$
\partial_x^2\partial_y^2 u_{yy} = -\partial_x^2\partial_y^2 u_{xx}
\quad\Longrightarrow\quad
u_{xxyyyy} = -u_{xxxxyy}.
$$

So the bracket $u_{xxxxyy} + u_{xxyyyy}$ of exercise 2.2 is zero, the $h^4$ term vanishes, and the
error starts at $h^6$.

The check is direct: differentiate a harmonic function six times two ways and compare.

```python
import sympy as sp

x, y = sp.symbols("x y")
for expr in (sp.exp(x) * sp.sin(y), sp.re((x + sp.I * y) ** 7), sp.log(x ** 2 + y ** 2) / 2):
    lap = sp.simplify(sp.diff(expr, x, 2) + sp.diff(expr, y, 2))
    left = sp.simplify(sp.diff(expr, x, 2, y, 4))
    right = sp.simplify(-sp.diff(expr, x, 4, y, 2))
    print(f"laplacian {lap}, u_xxyyyy + u_xxxxyy = {sp.simplify(left - right)}")
```

### 2.4 The ghost point elimination

Take $-u'' = f$ on $[a, b]$ with $u'(a) = g$. Put a ghost node at $x_{-1} = a - h$ and write the
centred second difference **at the boundary node itself**:

$$
-\frac{u_1 - 2u_0 + u_{-1}}{h^2} = f_0 .
$$

The centred first derivative at the same node is second order:

$$
\frac{u_1 - u_{-1}}{2h} = g + O(h^2) \quad\Longrightarrow\quad u_{-1} = u_1 - 2hg .
$$

Substituting,

$$
-\frac{2u_1 - 2u_0 - 2hg}{h^2} = f_0
\quad\Longrightarrow\quad
\frac{2u_0 - 2u_1}{h^2} = f_0 - \frac{2g}{h}.
$$

Every term is a grid value, and the ghost node has gone. The row is second order because both
ingredients were: the second difference at node 0 and the centred first derivative at node 0.

Note what the elimination used: **the differential equation at the boundary node**. That is why
the source $f_0$ appears in a row that looks like a boundary condition, and why the one sided
alternative, which uses no equation, cannot reach second order with two points.

```python
for n in (17, 33, 65, 129, 257):
    x = np.linspace(0.0, 1.0, n)
    h = float(x[1] - x[0])
    exact = np.cos(np.pi * x)
    residual = (2.0 * exact[0] - 2.0 * exact[1]) / h ** 2 - (
        np.pi ** 2 * np.cos(np.pi * x[0]) - 2.0 * 0.0 / h)
    print(f"h = {h:.5f}: residual of the ghost point row {abs(residual):.3e}, "
          f"h^2 = {h ** 2:.3e}")
```

### 2.5 The eigenvalues of the second difference matrix

Let $T$ be the $n \times n$ matrix with $-2$ on the diagonal and $1$ off it, and try the vector
$v_j = \sin(j\theta)$ for $j = 1, \dots, n$. Then

$$
(Tv)_j = \sin((j+1)\theta) - 2\sin(j\theta) + \sin((j-1)\theta)
= 2\sin(j\theta)\cos\theta - 2\sin(j\theta) = -4\sin^2\!\tfrac{\theta}{2}\,v_j .
$$

The boundary conditions $v_0 = v_{n+1} = 0$ force $\theta = j\pi/(n+1) = j\pi h$, so

$$
\lambda_j = -\frac{4\alpha}{h^2}\sin^2\!\left(\frac{j\pi h}{2}\right), \qquad j = 1, \dots, n .
$$

The extremes are

$$
\lvert\lambda_1\rvert = \frac{4\alpha}{h^2}\sin^2\frac{\pi h}{2} \to \alpha\pi^2,
\qquad
\lvert\lambda_n\rvert = \frac{4\alpha}{h^2}\sin^2\frac{n\pi h}{2} \to \frac{4\alpha}{h^2},
$$

so the stiffness ratio is

$$
\frac{\lvert\lambda_n\rvert}{\lvert\lambda_1\rvert} \to \frac{4}{\pi^2h^2} = O(h^{-2}).
$$

```python
print(f"{'n':>6}{'ratio':>12}{'4/(pi^2 h^2)':>16}{'gap':>10}")
for n in (7, 15, 31, 63, 127, 255):
    h = 1.0 / (n + 1)
    j = np.arange(1, n + 1)
    lam = 4.0 / h ** 2 * np.sin(j * np.pi * h / 2.0) ** 2
    ratio = float(np.max(lam) / np.min(lam))
    print(f"{n:>6}{ratio:>12.2f}{4.0 / (np.pi ** 2 * h ** 2):>16.2f}"
          f"{abs(ratio - 4.0 / (np.pi ** 2 * h ** 2)) / ratio:>10.4f}")
```

### 3.1 A stencil on unequally spaced offsets

Moving one node off the grid costs the second derivative **one** order and the first derivative
**none**. That asymmetry is the free order of section 4 being handed back: a centred stencil gets
an extra order because the odd Taylor term cancels by symmetry, and destroying the symmetry
destroys the cancellation.

```python
rng = np.random.default_rng(42)
cases = (("equal, 5 points", np.arange(-2.0, 3.0)),
         ("one node moved", np.asarray([-2.0, -1.0, 0.0, 1.0, 2.3])),
         ("random, 5 points", np.sort(rng.uniform(-2.0, 2.0, 5))),
         ("equal, 4 points", np.asarray([-2.0, -1.0, 0.0, 1.0])))
print(f"{'offsets':>18}{'first derivative':>19}{'second derivative':>20}")
for label, off in cases:
    print(f"{label:>18}{pc.stencil_order(off, 1):>19}{pc.stencil_order(off, 2):>20}")
print("\nthe symmetric 5 point second difference is order 4 = w - 1")
print("the unsymmetric one is order 3 = w - 2, which is what w - k would predict")
```

### 3.2 The compact nine point on a rectangle

**It does not work at all**, and the failure is total rather than a loss of order.

```python
def nine_point_rect(values, hx, hy):
    v = np.asarray(values, dtype=float)
    edge_x = v[2:, 1:-1] + v[:-2, 1:-1]
    edge_y = v[1:-1, 2:] + v[1:-1, :-2]
    corner = v[2:, 2:] + v[2:, :-2] + v[:-2, 2:] + v[:-2, :-2]
    return (4.0 * (edge_x + edge_y) + corner - 20.0 * v[1:-1, 1:-1]) / (6.0 * hx * hy)

def harmonic(x, y):
    return np.exp(x) * np.sin(y)

print(f"{'hx/hy':>8}{'order':>10}{'finest error':>16}")
for ratio in (1.0, 1.5, 2.0, 4.0):
    errors, hs = [], []
    for n in (17, 33, 65):
        gx = np.linspace(0.0, 1.0, n)
        gy = np.linspace(0.0, 1.0, int(round((n - 1) / ratio)) + 1)
        hx, hy = gx[1] - gx[0], gy[1] - gy[0]
        mx, my = np.meshgrid(gx, gy, indexing="ij")
        errors.append(float(np.max(np.abs(nine_point_rect(harmonic(mx, my), hx, hy)))))
        hs.append(hx)
    print(f"{ratio:>8.1f}{float(np.polyfit(np.log(hs), np.log(errors), 1)[0]):>10.3f}"
          f"{errors[-1]:>16.3e}")
```

At $h_x = h_y$ the error is $10^{-11}$ and the fitted order is meaningless because it is at the
rounding floor. At $h_x/h_y = 1.5$ the error is **1.8 and does not fall at all**: the fitted order
is $-0.115$, which is zero within the noise.

The compact weights encode $h_x = h_y$ in a way no rescaling can undo. Their derivation used the
corner values, whose expansion contains $h_x^2u_{xx} + h_y^2u_{yy}$ mixed together in a single
combination; separating them again needs a different set of weights for each ratio. There is a
nine point rule for a rectangle, and it is not this one with a division fixed up.

### 3.3 A Robin condition by the ghost point

$\alpha u + \beta u' = \gamma$ at $x = a$. The centred first derivative at node 0 gives
$u_{-1} = u_1 - \frac{2h}{\beta}\left(\gamma - \alpha u_0\right)$, and substituting into the
second difference at node 0 gives the row

$$
\left(\frac{2}{h^2} + \frac{2\alpha}{\beta h}\right)u_0 - \frac{2}{h^2}u_1
= f_0 + \frac{2\gamma}{\beta h}.
$$

```python
import math

def robin_solve(n, alpha, beta):
    x = np.linspace(0.0, 1.0, n)
    h = float(x[1] - x[0])
    exact = lambda t: np.cos(math.pi * t)
    source = lambda t: math.pi ** 2 * np.cos(math.pi * t)
    gamma = alpha * exact(0.0) + beta * 0.0          # u'(0) = -pi sin(0) = 0
    a = np.zeros((n, n)); b = np.zeros(n)
    for i in range(1, n - 1):
        a[i, i - 1] = -1.0 / h ** 2
        a[i, i] = 2.0 / h ** 2
        a[i, i + 1] = -1.0 / h ** 2
        b[i] = source(x[i])
    a[0, 0] = 2.0 / h ** 2 + 2.0 * alpha / (beta * h)
    a[0, 1] = -2.0 / h ** 2
    b[0] = source(x[0]) + 2.0 * gamma / (beta * h)
    a[-1, -1] = 1.0
    b[-1] = exact(1.0)
    return h, float(np.max(np.abs(np.linalg.solve(a, b) - exact(x))))

print(f"{'alpha':>7}{'beta':>7}{'alpha/beta':>13}{'order':>10}{'finest error':>15}")
for alpha, beta in ((0.0, 1.0), (1.0, 1.0), (5.0, 1.0), (1.0, 0.2)):
    hs, errs = [], []
    for n in (17, 33, 65, 129, 257):
        h, e = robin_solve(n, alpha, beta)
        hs.append(h); errs.append(e)
    order = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
    print(f"{alpha:>7.1f}{beta:>7.1f}{alpha / beta:>13.1f}{order:>10.4f}{errs[-1]:>15.3e}")
```

Second order for every pair, as the derivation says. The last two rows are worth a look: they have
different $\alpha$ and $\beta$ and **identical** errors, because the condition depends only on the
ratio $\alpha/\beta$, which is 5 in both. Scaling a boundary condition by a constant does not
change it, and the discretization inherits that.

### 3.4 The Mehrstellen scheme as a solve

The operator order of lesson 77 section 5.1 was measured by applying the stencil to a known
function. The solve is a different question: does the fourth order survive being inverted?

```python
def poisson_solve(n, scheme):
    line = np.linspace(0.0, 1.0, n)
    h = float(line[1] - line[0])
    m = n - 2
    exact = lambda x, y: np.exp(x) * np.sin(math.pi * x) * np.sin(math.pi * y)
    source = lambda x, y: (np.exp(x) * np.sin(math.pi * y)
                           * ((2.0 * math.pi ** 2 - 1.0) * np.sin(math.pi * x)
                              - 2.0 * math.pi * np.cos(math.pi * x)))
    gx, gy = np.meshgrid(line, line, indexing="ij")
    known = exact(gx, gy)
    if scheme == "five":
        weights = {(0, 0): 4.0, (1, 0): -1.0, (-1, 0): -1.0, (0, 1): -1.0, (0, -1): -1.0}
    else:
        weights = {(0, 0): 20.0 / 6.0, (1, 0): -4.0 / 6.0, (-1, 0): -4.0 / 6.0,
                   (0, 1): -4.0 / 6.0, (0, -1): -4.0 / 6.0,
                   (1, 1): -1.0 / 6.0, (1, -1): -1.0 / 6.0,
                   (-1, 1): -1.0 / 6.0, (-1, -1): -1.0 / 6.0}
    a = np.zeros((m * m, m * m)); b = np.zeros(m * m)
    for i in range(m):
        for j in range(m):
            row = i * m + j
            here = source(line[i + 1], line[j + 1])
            if scheme == "five":
                b[row] = here
            else:
                lap_f = (source(line[i + 2], line[j + 1]) + source(line[i], line[j + 1])
                         + source(line[i + 1], line[j + 2]) + source(line[i + 1], line[j])
                         - 4.0 * here) / h ** 2
                b[row] = here + h ** 2 / 12.0 * lap_f
            for (di, dj), w in weights.items():
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < m:
                    a[row, ii * m + jj] += w / h ** 2
                else:
                    b[row] -= w / h ** 2 * known[ii + 1, jj + 1]
    u = np.linalg.solve(a, b)
    return h, float(np.max(np.abs(u - exact(gx[1:-1, 1:-1], gy[1:-1, 1:-1]).ravel())))

for scheme in ("five", "mehrstellen"):
    hs, errs = [], []
    for n in (9, 17, 33, 65):
        h, e = poisson_solve(n, scheme)
        hs.append(h); errs.append(e)
    print(f"{scheme:>12}: order {float(np.polyfit(np.log(hs), np.log(errs), 1)[0]):.3f}, "
          f"errors {np.array2string(np.asarray(errs), precision=3)}")
```

Order **3.993** against **1.999**, and the finest error is 4700 times smaller. The fourth order
survives the solve, which it had to: the discrete operator is second order accurate as an operator
and the scheme is fourth order accurate as a **scheme**, because the correction was applied to the
right hand side rather than to the stencil.

Note the correction used $\Delta f$ approximated by the five point rule on $f$, which is second
order. That is enough: it multiplies an $h^2$ prefactor, so its own error enters at $h^4$.

### 3.5 Transonic flow

$(1 - M^2)\phi_{xx} + \phi_{yy} = 0$ has $A = 1 - M^2$, $B = 0$, $C = 1$, so the discriminant is
$-4(1 - M^2)$. It is **elliptic where the flow is subsonic** ($M < 1$) and **hyperbolic where it is
supersonic** ($M > 1$), with the sonic line $M = 1$ between them. That is Tricomi's equation with
$y$ replaced by $1 - M^2$.

```python
def transonic(mach):
    def coefficients(x, y):
        m = mach(x, y)
        return (1.0 - m ** 2, np.zeros(np.shape(x)), np.ones(np.shape(x)))
    return coefficients

xs = np.linspace(0.0, 1.0, 201)
for name, mach in (("M = 0.5 + x", lambda x, y: 0.5 + x),
                   ("M = 0.8 + 0.5 y", lambda x, y: 0.8 + 0.5 * y),
                   ("M = 0.6 everywhere", lambda x, y: 0.6 * np.ones(np.shape(x)))):
    fractions = pc.classify_on_grid(transonic(mach), xs, xs)["fractions"]
    print(f"{name:>20}: " + ", ".join(f"{k} {v:.4f}" for k, v in fractions.items()))
```

The first case is sonic at $x = 0.5$ and splits the square in half. The second is sonic at
$y = 0.4$ and splits it 40/60. The third is subsonic everywhere and is elliptic everywhere, which
is why subsonic aerodynamics is a potential theory problem and transonic aerodynamics is a
different subject.

### 4.1 The rounding floor of the second difference

The total error of $(u(x+h) - 2u(x) + u(x-h))/h^2$ is

$$
\underbrace{\frac{h^2}{12}\lvert u''''\rvert}_{\text{truncation}}
+ \underbrace{\frac{4\varepsilon\lvert u\rvert}{h^2}}_{\text{rounding}},
$$

which is minimised at $h^{*} = (48\varepsilon\lvert u\rvert/\lvert u''''\rvert)^{1/4}$, giving
$3.2\times10^{-4}$ for $u = \sin$ at $x = 1$.

```python
print(f"{'h':>12}{'total':>13}{'truncation':>13}{'rounding':>13}")
values = []
for power in range(2, 24):
    h = 10.0 ** (-power * 0.25)
    got = (math.sin(1.0 + h) - 2.0 * math.sin(1.0) + math.sin(1.0 - h)) / h ** 2
    total = abs(got + math.sin(1.0))
    values.append((h, total))
    if power % 2 == 0:
        print(f"{h:>12.2e}{total:>13.3e}{h ** 2 / 12.0 * abs(math.sin(1.0)):>13.3e}"
              f"{4.0 * np.finfo(float).eps * abs(math.sin(1.0)) / h ** 2:>13.3e}")
best = min(values, key=lambda pair: pair[1])
print(f"\nmeasured best h {best[0]:.3e} with error {best[1]:.3e}")
print(f"predicted optimum (48 eps)^(1/4) = {(48.0 * np.finfo(float).eps) ** 0.25:.3e}")
```

The two columns cross near $h = 3\times10^{-4}$, exactly where the formula puts the optimum. The
measured minimum is at $5.6\times10^{-5}$, a factor of 5.7 below it, and that gap is not an error
in the formula.

The bound the formula minimises is the **worst case**: it assumes the rounding error takes its
largest possible value $4\varepsilon\lvert u\rvert$ at every $h$. What the rounding error actually
does is jump about between zero and that bound, depending on where the three sampled values happen
to fall in the floating point grid, so the realised total error dips well below the bound at
scattered values of $h$ and the argmin lands on whichever of those the sweep included. Changing the
sample points moves the argmin and does not move the crossing.

**The number worth quoting is the crossing, not the argmin.**

### 4.2 The wide nine point rule, and a test that cannot tell them apart

The wide rule uses the five point fourth order first difference in each direction separately:

$$
u_{xx} \approx \frac{-u_{i+2} + 16u_{i+1} - 30u_i + 16u_{i-1} - u_{i-2}}{12h^2},
$$

and the same in $y$. Nine points, fourth order by construction, and **not** compact: it reaches two
nodes out, so it needs a special rule next to every boundary.

```python
def wide_nine(values, h):
    v = np.asarray(values, dtype=float)
    inner = v[2:-2, 2:-2]
    xx = (-v[4:, 2:-2] + 16.0 * v[3:-1, 2:-2] - 30.0 * inner
          + 16.0 * v[1:-3, 2:-2] - v[:-4, 2:-2]) / (12.0 * h ** 2)
    yy = (-v[2:-2, 4:] + 16.0 * v[2:-2, 3:-1] - 30.0 * inner
          + 16.0 * v[2:-2, 1:-3] - v[2:-2, :-4]) / (12.0 * h ** 2)
    return xx + yy

n = 13
line = np.linspace(0.0, 1.0, n)
h = float(line[1] - line[0])
gx, gy = np.meshgrid(line, line, indexing="ij")
top_degree = 2 * (n - 3) // 2
print(f"{'degree':>8}{'five':>12}{'compact nine':>15}{'wide nine':>13}")
for d in range(top_degree + 1):
    values = np.real((gx + 1j * gy) ** d)
    scale = max(float(np.max(np.abs(values))), 1.0) / h ** 2
    print(f"{d:>8}{float(np.max(np.abs(pc.five_point(values, h, h)))) / scale:>12.1e}"
          f"{float(np.max(np.abs(pc.nine_point(values, h)))) / scale:>15.1e}"
          f"{float(np.max(np.abs(wide_nine(values, h)))) / scale:>13.1e}")
```

**Both nine point rules are exact on harmonic polynomials up to degree 7.** The exactness test of
section 5, which pinned the five point rule down so cleanly, **cannot tell these two apart at
all**, and they are not the same order.

The reason is that the test only probes harmonic behaviour, and both rules happen to be sixth
order there. A general function separates them at once.

```python
print(f"\n{'function':>14}{'rule':>10}{'order':>9}{'finest error':>16}")
for label, fn in (("harmonic", lambda x, y: np.exp(x) * np.sin(y)),
                  ("not harmonic", lambda x, y: np.sin(x) * np.sin(y))):
    for name, rule, trim in (("compact", lambda v, hh: pc.nine_point(v, hh), 1),
                             ("wide", wide_nine, 2)):
        hs, errs = [], []
        for m in (13, 17, 25, 33):
            g = np.linspace(0.0, 1.0, m)
            hh = float(g[1] - g[0])
            mx, my = np.meshgrid(g, g, indexing="ij")
            cut = slice(trim, m - trim)
            want = (np.zeros((m - 2 * trim, m - 2 * trim)) if label == "harmonic"
                    else -2.0 * np.sin(mx[cut, cut]) * np.sin(my[cut, cut]))
            errs.append(float(np.max(np.abs(rule(fn(mx, my), hh) - want))))
            hs.append(hh)
        print(f"{label:>14}{name:>10}"
              f"{float(np.polyfit(np.log(hs), np.log(errs), 1)[0]):>9.3f}{errs[-1]:>16.3e}")
```

On a function that is not harmonic the compact rule fits **1.92** and the wide one **3.83**. The
wide rule is genuinely fourth order and the compact one is genuinely second, and the harmonic
polynomial test said 7 for both.

The lesson from the lesson holds with an extra clause: an exactness test is only as informative as
the space it tests over. Harmonic polynomials probe exactly the case where both rules are good.

### 4.3 Where the interior order stops mattering

Three boundary treatments for $u'(0) = g$, with the same second order interior:

1. one sided two point, first order in the condition,
2. ghost point, second order,
3. one sided three point, second order in the condition.

```python
def neumann_solve(n, treatment):
    x = np.linspace(0.0, 1.0, n)
    h = float(x[1] - x[0])
    exact = lambda t: np.cos(math.pi * t)
    source = lambda t: math.pi ** 2 * np.cos(math.pi * t)
    a = np.zeros((n, n)); b = np.zeros(n)
    for i in range(1, n - 1):
        a[i, i - 1] = -1.0 / h ** 2
        a[i, i] = 2.0 / h ** 2
        a[i, i + 1] = -1.0 / h ** 2
        b[i] = source(x[i])
    if treatment == 1:
        a[0, 0], a[0, 1] = -1.0 / h, 1.0 / h
    elif treatment == 2:
        a[0, 0], a[0, 1] = 2.0 / h ** 2, -2.0 / h ** 2
        b[0] = source(x[0])
    else:
        a[0, 0], a[0, 1], a[0, 2] = -1.5 / h, 2.0 / h, -0.5 / h
    a[-1, -1] = 1.0
    b[-1] = exact(1.0)
    return h, float(np.max(np.abs(np.linalg.solve(a, b) - exact(x))))

for treatment in (1, 2, 3):
    hs, errs = [], []
    for n in (17, 33, 65, 129, 257, 513):
        h, e = neumann_solve(n, treatment)
        hs.append(h); errs.append(e)
    errs = np.asarray(errs); hs = np.asarray(hs)
    print(f"treatment {treatment}: whole fit "
          f"{float(np.polyfit(np.log(hs), np.log(errs), 1)[0]):.4f}, tail fit "
          f"{float(np.polyfit(np.log(hs[-3:]), np.log(errs[-3:]), 1)[0]):.4f}")
    print(f"             ratios {np.array2string(errs[:-1] / errs[1:], precision=3)}")
```

Treatment 1 is first order, 0.9945. Treatment 2 is second order to four digits with ratios of
exactly 4.000. Treatment 3 is **also second order**, and its whole sweep fit reads **1.57** because
it approaches that order slowly: the ratios climb 1.20, 2.89, 3.50, 3.76, 3.88 towards 4.

So the answer to "where does the interior order stop mattering" is: **it stops mattering above the
interior order and not below it.** A boundary better than second order buys nothing here, because
the interior caps the whole solve at second, and treatment 3's finest error ($6.09\times10^{-6}$)
is within 3 per cent of treatment 2's ($6.28\times10^{-6}$) after all its extra work.

### 5.1 Isotropy

The five point error is $\frac{h^2}{12}(u_{4x} + u_{4y})$ and the nine point error is
$\frac{h^2}{12}\Delta^2u$. For a plane wave $u = \sin(\mathbf{k}\cdot\mathbf{x})$ with
$\mathbf{k} = k(\cos\theta, \sin\theta)$,

$$
u_{4x} + u_{4y} = k^4(\cos^4\theta + \sin^4\theta)\,u,
\qquad
\Delta^2 u = k^4 u .
$$

The nine point constant does not depend on $\theta$ at all: it is **isotropic**. The five point one
varies between $k^4$ at $\theta = 0$ and $k^4/2$ at $\theta = 45°$, a factor of 2.

```python
print(f"{'angle':>7}{'five point':>14}{'nine point':>14}{'predicted five':>17}")
k = 2.0 * math.pi
for degrees in (0, 15, 30, 45):
    angle = math.radians(degrees)
    kx, ky = math.cos(angle), math.sin(angle)
    errs = {}
    for name in ("five", "nine"):
        m = 49
        g = np.linspace(0.0, 1.0, m)
        hh = float(g[1] - g[0])
        mx, my = np.meshgrid(g, g, indexing="ij")
        v = np.sin(k * (kx * mx + ky * my))
        want = -k ** 2 * v[1:-1, 1:-1]
        got = pc.five_point(v, hh, hh) if name == "five" else pc.nine_point(v, hh)
        errs[name] = float(np.max(np.abs(got - want)))
    predicted = hh ** 2 / 12.0 * k ** 4 * (kx ** 4 + ky ** 4)
    print(f"{degrees:>7}{errs['five']:>14.4e}{errs['nine']:>14.4e}{predicted:>17.4e}")
```

Measured: the nine point error is $5.634\times10^{-2}$, $5.634\times10^{-2}$,
$5.633\times10^{-2}$, $5.632\times10^{-2}$ across the four angles, constant to four digits. The
five point error runs $5.634\times10^{-2}$ down to $2.818\times10^{-2}$, exactly the predicted
factor of $\cos^4\theta + \sin^4\theta$.

**The conclusion is the opposite of the usual one.** Isotropy is often given as the reason to
prefer the nine point rule, and what the measurement shows is that the nine point rule is
**uniformly as bad as the five point rule's worst direction**, while the five point rule is never
worse and is up to twice as good. Isotropy here is bought by giving up the good directions, not by
improving the bad ones. It is still worth having when the error's **direction dependence** would
show up as a visible artefact, and it is not an accuracy argument.

### 5.2 Classification in more than two variables

Write the principal part as $\sum_{i,j} a_{ij}u_{x_ix_j}$ with $a_{ij}$ symmetric. Its
classification is the **signature** of $A = (a_{ij})$, which by Sylvester's law of inertia is
invariant under any real change of variables, exactly as the sign of $B^2 - 4AC$ was in two.

With $n$ variables and $A$ nonsingular, let $p$ be the number of positive eigenvalues and $q$ the
number of negative ones, $p + q = n$:

- $p = n$ or $q = n$: **elliptic**. Laplace in any dimension.
- $p = n-1, q = 1$ or the reverse: **hyperbolic**. The wave equation, one time direction.
- $A$ singular: **parabolic**, in the sense that at least one second derivative is missing. The
  heat equation in $n$ space variables plus time has $A$ of rank $n$ in $n+1$ variables.
- $p \ge 2$ **and** $q \ge 2$: **ultrahyperbolic**, and this case has no two variable analogue.

The last one is the interesting answer. In two variables the discriminant's sign is the whole
story, because $p + q \le 2$ leaves no room for the ultrahyperbolic case. From four variables on,
$u_{x_1x_1} + u_{x_2x_2} - u_{x_3x_3} - u_{x_4x_4} = 0$ is a real possibility, and it is **not**
well posed as an initial value problem in any direction: Asgeirsson's mean value theorem forces
solutions to satisfy extra global constraints, so prescribing Cauchy data on a hypersurface
generally admits no solution at all. There is no useful numerical method for it because there is
no well posed problem to solve.

```python
def classify_matrix(a):
    values = np.linalg.eigvalsh(np.asarray(a, dtype=float))
    tol = 1e-12 * max(float(np.max(np.abs(values))), 1.0)
    p = int(np.count_nonzero(values > tol))
    q = int(np.count_nonzero(values < -tol))
    zeros = values.size - p - q
    if zeros:
        return f"parabolic (rank {p + q} of {values.size})"
    if p == 0 or q == 0:
        return "elliptic"
    if p == 1 or q == 1:
        return "hyperbolic"
    return "ultrahyperbolic"

examples = {
    "Laplace in 3D": np.diag([1.0, 1.0, 1.0]),
    "wave in 2+1": np.diag([1.0, 1.0, -1.0]),
    "heat in 2+1": np.diag([1.0, 1.0, 0.0]),
    "ultrahyperbolic": np.diag([1.0, 1.0, -1.0, -1.0]),
    "wave in 3+1": np.diag([1.0, 1.0, 1.0, -1.0]),
}
for name, matrix in examples.items():
    print(f"{name:>18}: {classify_matrix(matrix)}")
```

### 5.3 Well posed and still divergent

Well posedness is a property of the **continuous** problem and stability is a property of the
**scheme**, and neither implies the other. The heat equation is as well posed as a problem gets,
and the explicit scheme above its mesh ratio limit diverges on it.

```python
from nalib import parabolic as pb

problem = pb.step_problem()
print(f"{'r':>7}{'steps':>8}{'error':>14}")
for r in (0.4, 0.5, 0.6, 0.75):
    h = 1.0 / 40
    k = r * h ** 2
    steps = max(int(round(0.02 / k)), 1)
    with np.errstate(over="ignore", invalid="ignore"):
        run = pb.solve(problem, 41, steps, steps * k, theta=0.0)
    print(f"{r:>7.2f}{steps:>8}{run['error']:>14.3e}")
print("\nthe problem is well posed at every one of these; only the scheme changes")
```

The classification protects you from asking an impossible question. It does not protect you from
answering a possible one badly, and the Lax equivalence theorem of lesson 79 is exactly the
statement that fills the gap: **well posed plus consistent plus stable** is what gives convergence,
and the classification supplies only the first of the three.

There is a sharper version of the same point. Consistency and well posedness together are not
enough even in principle: Richardson's scheme of lesson 78 is consistent to second order in both
variables on a perfectly well posed problem and is unstable at **every** mesh ratio, so no
refinement path makes it converge. Its failure is not a bad choice of step; it is the scheme.

---

## Lesson 78, Parabolic Equations

### 1.1 The family and its three named members

$$
\frac{u_j^{n+1} - u_j^n}{k}
= \alpha\left[\theta\,\delta_x^2u^{n+1} + (1-\theta)\,\delta_x^2u^n\right]_j,
\qquad
\delta_x^2v_j = \frac{v_{j+1} - 2v_j + v_{j-1}}{h^2}.
$$

| $\theta$ | name | solve per step | stability |
|---|---|---|---|
| $0$ | forward difference, explicit | none | $r \le 1/2$ |
| $1/2$ | Crank-Nicolson | tridiagonal | unconditional |
| $1$ | backward difference, implicit | tridiagonal | unconditional |

$\theta$ is the fraction of the space difference taken at the **new** time level. At $\theta = 0$
everything is known when the step starts; at $\theta > 0$ the unknowns appear on both sides.

```python
import numpy as np
from nalib import parabolic as pb

problem = pb.sine_problem()
for theta, name in ((0.0, "explicit"), (0.5, "Crank-Nicolson"), (1.0, "backward")):
    run = pb.solve(problem, 41, 400, 0.05, theta=theta)
    limit = pb.stability_limit(theta)
    shown = "unconditional" if np.isinf(limit) else f"r <= {limit}"
    print(f"{name:>16}: r = {run['r']:.4f}, error {run['error']:.3e}, {shown}")
```

### 1.2 Why an implicit step is $O(n)$ and not $O(n^3)$

The matrix is $(I - \theta r T)$ with $T$ the second difference operator: **tridiagonal**. Gaussian
elimination on a tridiagonal matrix never creates a nonzero outside the three diagonals, because
eliminating row $i$ touches only row $i+1$, and that row has a single subdiagonal entry to remove.
So the factorization does $O(1)$ work per row and $O(n)$ in total. That is the Thomas algorithm of
lesson 22.

A general dense matrix costs $O(n^3)$ because eliminating one column touches every remaining row
and column. The saving here is entirely a consequence of the stencil having three points, and it is
what makes an implicit method a small constant times an explicit one rather than a different cost
class.

Two further points matter in practice. The matrix does **not change** from step to step for a fixed
$r$, so it can be factored once and reused, which halves the work again. And in two dimensions the
argument fails, which is lesson 80's subject.

### 1.3 The stability limit as a function of $\theta$

Substituting $u_j^n = g^ne^{ij\phi}$ gives

$$
g(\phi) = \frac{1 - 4(1-\theta)r\,s}{1 + 4\theta r\,s}, \qquad s = \sin^2(\phi/2) \in [0,1].
$$

The numerator decreases in $s$ and the denominator increases, so $\lvert g\rvert$ is largest at
$s = 1$, that is at $\phi = \pi$: the shortest wavelength the grid carries. Requiring
$g(\pi) \ge -1$ gives

$$
1 - 4(1-\theta)r \ge -1 - 4\theta r
\quad\Longleftrightarrow\quad
r\left(2 - 4\theta\right) \le 1 .
$$

If $\theta < 1/2$ the bracket is positive and the condition is $r \le 1/(2 - 4\theta)$. If
$\theta \ge 1/2$ the bracket is zero or negative and the inequality holds for **every** $r$.

**$\theta = 1/2$ is the threshold because it is where the bracket $2 - 4\theta$ changes sign.**
Below it the numerator can outrun the denominator as $r$ grows; at and above it the denominator
grows at least as fast, and no $r$ is large enough.

```python
print(f"{'theta':>8}{'2 - 4 theta':>14}{'limit':>18}{'g(pi) at r = 100':>19}")
for theta in (0.0, 0.25, 0.4, 0.5, 0.75, 1.0):
    limit = pb.stability_limit(theta)
    shown = "unconditional" if np.isinf(limit) else f"{limit:.4f}"
    print(f"{theta:>8.2f}{2.0 - 4.0 * theta:>14.2f}{shown:>18}"
          f"{float(pb.growth_factor(100.0, np.pi, theta)):>19.6f}")
```

### 1.4 Why an unstable run can return a small error

A linear scheme acts independently on each Fourier mode: the mode of phase $\phi$ is multiplied by
$g(\phi)$ every step and never mixes with any other. So after $N$ steps the amplitude of that mode
is

$$
(\text{its amplitude in the initial data}) \times \lvert g(\phi)\rvert^{N},
$$

and if the first factor is zero the second one cannot make it anything else.

Instability means $\lvert g(\pi)\rvert > 1$. On a single smooth sine mode the amplitude of the mode
at $\phi = \pi$ is **exactly zero**, so the only thing that ever appears there is rounding error at
$10^{-16}$, and it needs $\log(10^{16})/\log\lvert g\rvert$ steps to reach 1. At $r = 0.55$ that is
198 steps, and a run of 58 returns a perfectly ordinary error.

The lesson's measurement is the quantitative form of this: the product of the seed and the growth
predicts whether each of 21 runs blew up, in every row.

### 1.5 Conditionally consistent

**Consistent** means the truncation error goes to zero as the grid is refined. **Conditionally
consistent** means it does so only along some refinement paths and not others.

Du Fort and Frankel's scheme is the example. Its truncation error contains a term proportional to
$(k/h)^2u_{tt}$, which goes to zero only if $k/h \to 0$. Refine with $k \propto h$ and the term
does not vanish: the scheme converges, to the solution of

$$
u_t + \left(\frac{k}{h}\right)^2u_{tt} = \alpha u_{xx},
$$

which is a telegraph equation. It is stable for every $r$, so the Lax theorem does not save you:
the theorem assumes consistency, and that is exactly the assumption being violated.

The lesson measures the error settling on a nonzero limit at fixed $k/h$, with that limit
proportional to $(k/h)^2$, and converging at order 2.00 once $k$ is tied to $h^2$ instead.

### 2.1 The amplification factor and the limit

Put $u_j^n = g^ne^{ij\phi}$ into the scheme. The second difference acting on $e^{ij\phi}$ gives

$$
e^{i(j+1)\phi} - 2e^{ij\phi} + e^{i(j-1)\phi} = \left(e^{i\phi} - 2 + e^{-i\phi}\right)e^{ij\phi}
= \left(2\cos\phi - 2\right)e^{ij\phi} = -4\sin^2\!\tfrac{\phi}{2}\,e^{ij\phi}.
$$

Write $s = \sin^2(\phi/2)$. The scheme becomes

$$
g - 1 = r\left[\theta(-4s)g + (1-\theta)(-4s)\right]
\quad\Longrightarrow\quad
g\left(1 + 4\theta rs\right) = 1 - 4(1-\theta)rs,
$$

which is the expression of exercise 1.3, and the limit follows as shown there.

Two things are worth noting. The factor is **real**, so the scheme cannot produce a phase error at
all, only a magnitude error and possibly a sign flip. And it is a Mobius map of $s$, so it is
monotone in $s$: checking the endpoints $s = 0$ and $s = 1$ is enough, and no interior search is
needed.

```python
n = 65
index = np.arange(n)
worst = 0.0
for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
    for r in (0.2, 0.5, 2.0, 20.0):
        for mode in (1, 9, 31, n - 2):
            phase = mode * np.pi / (n - 1)
            u = np.sin(phase * index)
            u[0] = u[-1] = 0.0
            stepped = pb.theta_step(u, r, theta)
            want = pb.growth_factor(r, phase, theta) * u
            worst = max(worst, float(np.max(np.abs(stepped - want))))
print(f"largest gap between one step and g(phi) times the mode: {worst:.3e}")
print("over 5 values of theta, 4 mesh ratios and 4 modes")
```

### 2.2 The Bender-Schmidt form

Put $\theta = 0$ and $r = 1/2$ into the explicit update:

$$
u_j^{n+1} = u_j^n + \tfrac12\left(u_{j+1}^n - 2u_j^n + u_{j-1}^n\right)
= u_j^n - u_j^n + \tfrac12 u_{j+1}^n + \tfrac12 u_{j-1}^n
= \frac{u_{j-1}^n + u_{j+1}^n}{2}.
$$

The centre value cancels because the coefficient $1 - 2r$ is zero exactly at $r = 1/2$. That is the
same $r$ at which $g(\pi) = 1 - 4r = -1$, the edge of stability, so **Bender-Schmidt is the
explicit scheme run at exactly its limit**, and it is neutrally stable at the shortest wavelength:
that mode neither grows nor decays, it flips sign every step forever.

In floating point the two forms are not the same program, which the lesson measures at one rounding
unit.

```python
rng = np.random.default_rng(42)
print(f"{'n':>6}{'residual':>14}{'in roundings':>15}")
for n in (11, 21, 41, 81):
    v = rng.normal(size=n)
    v[0] = v[-1] = 0.0
    stepped = pb.theta_step(v, 0.5, 0.0)
    averaged = 0.5 * (v[2:] + v[:-2])
    residual = float(np.max(np.abs(stepped[1:-1] - averaged)))
    print(f"{n:>6}{residual:>14.3e}"
          f"{residual / (np.finfo(float).eps * float(np.max(np.abs(v)))):>15.3f}")
print(f"\nthe growth factor at r = 1/2 and phi = pi is "
      f"{float(pb.growth_factor(0.5, np.pi, 0.0)):.1f}: neutrally stable")
```

### 2.3 The explicit scheme is fourth order at $r = 1/6$

Expand about $(x_j, t_n)$. The time difference is

$$
\frac{u^{n+1} - u^n}{k} = u_t + \frac{k}{2}u_{tt} + O(k^2),
$$

and the space difference is

$$
\delta_x^2u^n = u_{xx} + \frac{h^2}{12}u_{xxxx} + O(h^4).
$$

So the truncation error of $u_t - \alpha u_{xx}$ is

$$
T = \frac{k}{2}u_{tt} - \frac{\alpha h^2}{12}u_{xxxx} + O(k^2 + h^4).
$$

On a **solution** of the equation, $u_t = \alpha u_{xx}$ gives
$u_{tt} = \alpha u_{xxt} = \alpha^2u_{xxxx}$. Substituting,

$$
T = \left(\frac{k\alpha^2}{2} - \frac{\alpha h^2}{12}\right)u_{xxxx}
= \alpha h^2\left(\frac{r}{2} - \frac{1}{12}\right)u_{xxxx},
$$

using $r = \alpha k/h^2$. The bracket vanishes at

$$
r = \frac16 ,
$$

and there the error starts at the next term, which is $O(h^4)$.

Note what the derivation used: **the equation itself**, to convert $u_{tt}$ into $u_{xxxx}$. So the
cancellation holds for this equation with a constant $\alpha$ and for nothing else. It is a
property of the pair, not of the scheme.

```python
out = pb.the_lucky_ratio()
print(f"{'r':>10}{'|r/2 - 1/12|':>16}{'order in h':>13}{'finest error':>15}")
for row in out["rows"]:
    print(f"{row['r']:>10.4f}{abs(0.5 * row['r'] - 1.0 / 12.0):>16.5f}"
          f"{row['order_in_h']:>13.3f}{row['error'][-1]:>15.3e}")
print(f"\nthe bracket is zero only at r = 1/6: "
      f"{out['the_cancelling_factor_is_zero_only_there']}")
```

### 2.4 Richardson's amplification quadratic

The scheme is

$$
\frac{u_j^{n+1} - u_j^{n-1}}{2k} = \alpha\,\delta_x^2u_j^n .
$$

Substituting $u_j^n = g^ne^{ij\phi}$ and dividing by $g^{n-1}e^{ij\phi}$,

$$
\frac{g^2 - 1}{2k} = \frac{\alpha}{h^2}\left(-4\sin^2\tfrac{\phi}{2}\right)g
\quad\Longrightarrow\quad
g^2 + 8rs\,g - 1 = 0, \qquad s = \sin^2\tfrac{\phi}{2}.
$$

The product of the roots of $g^2 + bg + c$ is $c$, and here $c = -1$ regardless of $r$, $s$ or
anything else. So

$$
g_1g_2 = -1 \quad\Longrightarrow\quad \lvert g_1\rvert\lvert g_2\rvert = 1 .
$$

If both roots had modulus at most 1 their product would have modulus at most 1, with equality only
if both are exactly on the circle. The roots are real (the discriminant $64r^2s^2 + 4$ is positive),
so "on the circle" means $\pm1$, and $g^2 + 8rsg - 1 = 0$ has $g = \pm1$ only when $rs = 0$.

Therefore **for every $r > 0$ and every mode with $s > 0$, one root is strictly outside the unit
circle.** There is no stable mesh ratio.

```python
out = pb.richardson_cannot_be_saved()
print(f"{'r':>8}{'roots':>28}{'product':>10}{'largest |root|':>17}")
for row in out["rows"]:
    print(f"{row['r']:>8.2f}{f'{row.get(chr(114) + chr(111) + chr(111) + chr(116) + chr(115))[0]:+.4f}, {row['roots'][1]:+.4f}':>28}"
          f"{row['product_of_roots']:>10.4f}{row['largest_root']:>17.4f}")
print(f"\nthe product is always -1: {out['the_product_of_the_roots_is_always_minus_one']}")
```

### 2.5 Du Fort and Frankel's modified equation

Start from Richardson and replace $u_j^n$ inside the second difference by
$\frac12\left(u_j^{n+1} + u_j^{n-1}\right)$:

$$
\frac{u_j^{n+1} - u_j^{n-1}}{2k}
= \frac{\alpha}{h^2}\left(u_{j+1}^n - u_j^{n+1} - u_j^{n-1} + u_{j-1}^n\right).
$$

Now expand every term about $(x_j, t_n)$:

$$
\frac{u^{n+1} - u^{n-1}}{2k} = u_t + \frac{k^2}{6}u_{ttt} + \cdots,
$$

$$
u_{j+1} + u_{j-1} - 2u_j = h^2u_{xx} + \frac{h^4}{12}u_{xxxx} + \cdots,
$$

$$
u_j^{n+1} + u_j^{n-1} - 2u_j^n = k^2u_{tt} + \frac{k^4}{12}u_{tttt} + \cdots .
$$

The right hand side is $\frac{\alpha}{h^2}\left[(u_{j+1} + u_{j-1} - 2u_j) - (u_j^{n+1} +
u_j^{n-1} - 2u_j^n)\right]$, so

$$
u_t + O(k^2) = \alpha u_{xx} + O(h^2) - \alpha\frac{k^2}{h^2}u_{tt} + O\!\left(\frac{k^4}{h^2}\right),
$$

that is

$$
u_t + \alpha\left(\frac{k}{h}\right)^2u_{tt} = \alpha u_{xx} + O(h^2 + k^2).
$$

**The extra term is $\alpha(k/h)^2u_{tt}$**, and it is not small unless $k/h$ is. It turns a
parabolic equation into a hyperbolic one, with a finite propagation speed $h/k$, which is why the
scheme can be explicit and unconditionally stable at the same time: it is stable because it is
solving a different equation, one whose characteristics never outrun the stencil.

```python
out = pb.dufort_frankel_solves_the_wrong_equation()
print(f"{'k/h':>9}{'error at the finest h':>24}{'(k/h)^2':>12}")
for row in out["fixed_ratio"]:
    print(f"{row['k_over_h']:>9.4f}{row['limit']:>24.3e}{row['k_over_h'] ** 2:>12.5f}")
print(f"\nfitted exponent of the limit in k/h: "
      f"{out['exponent_over_the_small_ratios']:.3f} over the small ratios")
print(f"with k tied to h^2 instead, the order is {out['shrinking_ratio']['order']:.3f}")
```

### 3.1 A Neumann end by the ghost point

The ghost point row of lesson 77 becomes, for the weighted scheme, a first row with $2\theta r$ in
place of $\theta r$ on the superdiagonal and a doubled explicit term, which is the same elimination
done at both time levels.

```python
import math
from nalib.banded import thomas

def neumann_run(n, steps, t_end, theta, boundary):
    """u_t = u_xx with u_x(0,t) = 0 and u(1,t) given; exact u = exp(-pi^2 t) cos(pi x)."""
    x = np.linspace(0.0, 1.0, n)
    h = float(x[1] - x[0])
    k = t_end / steps
    r = k / h ** 2
    exact = lambda t: math.exp(-math.pi ** 2 * t) * np.cos(math.pi * x)
    u = exact(0.0)
    m = n - 1
    for step in range(steps):
        right = exact((step + 1) * k)[-1]
        if theta == 0.0:
            new = np.empty(n)
            new[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
            new[-1] = right
            new[0] = new[1] if boundary == 1 else u[0] + 2.0 * r * (u[1] - u[0])
            u = new
            continue
        d2 = np.empty(m)
        d2[0] = 2.0 * (u[1] - u[0]) if boundary == 2 else 0.0
        d2[1:] = u[2:] - 2.0 * u[1:-1] + u[:-2]
        rhs = u[:-1] + (1.0 - theta) * r * d2
        sub = np.full(m, -theta * r)
        diag = np.full(m, 1.0 + 2.0 * theta * r)
        sup = np.full(m, -theta * r)
        if boundary == 2:
            sup[0] = -2.0 * theta * r
        else:
            diag[0], sup[0], rhs[0] = 1.0, -1.0, 0.0
        rhs[-1] += theta * r * right
        u = np.concatenate([thomas(sub, diag, sup, rhs), [right]])
    return h, float(np.max(np.abs(u - exact(t_end))))

print(f"{'theta':>7}{'boundary':>10}{'order':>9}{'finest error':>15}")
for theta in (0.0, 0.5, 1.0):
    for boundary in (1, 2):
        hs, errs = [], []
        for n in (11, 21, 41, 81, 161):
            steps = max(int(round(0.05 / (0.4 * (1.0 / (n - 1)) ** 2))), 1)
            h, e = neumann_run(n, steps, 0.05, theta, boundary)
            hs.append(h); errs.append(e)
        print(f"{theta:>7.1f}{boundary:>10}"
              f"{float(np.polyfit(np.log(hs), np.log(errs), 1)[0]):>9.4f}{errs[-1]:>15.3e}")
```

Second order for every member with the ghost point, and first order for every member without it.
Lesson 77's finding is not a property of the steady problem: **the boundary caps the order of the
time dependent solve in exactly the same way**, and being implicit does not help.

### 3.2 A variable coefficient in conservation form

Write the operator as $(\alpha(x)u_x)_x$ and difference it as a difference of fluxes evaluated at
the **cell faces**, $x_{j\pm1/2}$:

$$
\left(\alpha u_x\right)_x \approx \frac{1}{h}\left[
\alpha_{j+1/2}\frac{u_{j+1} - u_j}{h} - \alpha_{j-1/2}\frac{u_j - u_{j-1}}{h}\right].
$$

That form is worth the extra care because it makes the discrete operator **symmetric negative
definite**, so the discrete energy $E = \sum u_j^2h$ decays exactly as the continuous one does:

$$
\frac{\mathrm{d}}{\mathrm{d}t}\frac{E}{2} = -\int \alpha u_x^2 .
$$

```python
def variable_run(n, steps, t_end, theta=0.5):
    x = np.linspace(0.0, 1.0, n)
    h = float(x[1] - x[0])
    k = t_end / steps
    faces = 1.0 + 0.5 * np.sin(2.0 * math.pi * (x[:-1] + 0.5 * h))
    u = np.sin(math.pi * x)
    energies = [float(np.sum(u[1:-1] ** 2) * h)]
    dissipation = []
    for _ in range(steps):
        flux = faces * np.diff(u) / h
        rhs = u[1:-1] + (1.0 - theta) * k * np.diff(flux) / h
        sub = -theta * k * faces[:-1] / h ** 2
        sup = -theta * k * faces[1:] / h ** 2
        diag = 1.0 + theta * k * (faces[:-1] + faces[1:]) / h ** 2
        inner = thomas(np.concatenate([[0.0], sub[1:]]), diag,
                       np.concatenate([sup[:-1], [0.0]]), rhs)
        u = np.concatenate([[0.0], inner, [0.0]])
        energies.append(float(np.sum(u[1:-1] ** 2) * h))
        dissipation.append(float(np.sum(faces * np.diff(u) ** 2 / h ** 2) * h))
    energies = np.asarray(energies)
    rate = -np.diff(energies) / (2.0 * k)
    return energies, float(np.max(np.abs(rate - np.asarray(dissipation))
                                  / np.asarray(dissipation)))

print(f"{'n':>6}{'E(0)':>10}{'E(T)':>10}{'monotone':>11}{'worst gap in the identity':>28}")
for n, steps in ((41, 200), (81, 800), (161, 3200)):
    energies, gap = variable_run(n, steps, 0.05)
    print(f"{n:>6}{energies[0]:>10.6f}{energies[-1]:>10.6f}"
          f"{str(bool(np.all(np.diff(energies) < 0))):>11}{gap:>28.5f}")
```

The energy decreases at every step on every grid, and the discrete identity holds to 0.38 per
cent at $n = 41$, 0.10 at $n = 81$ and 0.025 at $n = 161$: a factor of 4 per refinement, so the
identity is satisfied to second order and the discrete operator really is doing what the
continuous one does.

Differencing the non-conservative form $\alpha u_{xx} + \alpha'u_x$ instead gives a matrix that is
**not** symmetric, and the energy identity is then only approximate at every grid.

### 3.3 A moving boundary by remapping

Map $[0, s(t)]$ onto a fixed $[0, 1]$ with $\xi = x/s(t)$. Then

$$
u_t\big|_x = u_t\big|_\xi - \frac{\xi s'(t)}{s(t)}u_\xi,
\qquad
u_{xx} = \frac{1}{s(t)^2}u_{\xi\xi},
$$

so the equation becomes, on the fixed interval,

$$
u_t = \frac{\alpha}{s^2}u_{\xi\xi} + \frac{\xi s'}{s}u_\xi .
$$

The remapping has bought a fixed grid at the price of a **convection term** with a coefficient that
grows as the domain shrinks. Two things follow, and the second is the answer to the exercise.

```python
def moving_run(n, steps, t_end, theta=0.5):
    """u_t = u_xx on [0, s(t)] with s(t) = 1 + 0.4 t, exact u = exp(-pi^2 t / s^2) sin(pi x / s)
    is not a solution, so a manufactured source is used instead."""
    xi = np.linspace(0.0, 1.0, n)
    h = float(xi[1] - xi[0])
    k = t_end / steps
    s_of = lambda t: 1.0 + 0.4 * t
    ds_of = lambda t: 0.4
    exact = lambda t: math.exp(-t) * np.sin(math.pi * xi)
    # u_t - (1/s^2) u_xixi - (xi s'/s) u_xi = f, with u as above
    def source(t):
        s, ds = s_of(t), ds_of(t)
        return (-math.exp(-t) * np.sin(math.pi * xi)
                + math.pi ** 2 / s ** 2 * math.exp(-t) * np.sin(math.pi * xi)
                - xi * ds / s * math.pi * math.exp(-t) * np.cos(math.pi * xi))
    u = exact(0.0)
    m = n - 2
    for step in range(steps):
        t_now, t_next = step * k, (step + 1) * k
        s, ds = s_of(t_now), ds_of(t_now)
        d2 = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / h ** 2
        d1 = (u[2:] - u[:-2]) / (2.0 * h)
        rhs = (u[1:-1] + k * ((1.0 - theta) * (d2 / s ** 2 + xi[1:-1] * ds / s * d1)
                              + source(t_now)[1:-1] * (1.0 - theta)
                              + source(t_next)[1:-1] * theta))
        s2, ds2 = s_of(t_next), ds_of(t_next)
        drift = xi[1:-1] * ds2 / s2
        # thomas takes sub as the coefficient of x[i-1] and sup as that of x[i+1], and the
        # convection term contributes with opposite signs to the two
        sub = -theta * k * (1.0 / (s2 ** 2 * h ** 2) - drift / (2.0 * h))
        sup = -theta * k * (1.0 / (s2 ** 2 * h ** 2) + drift / (2.0 * h))
        diag = np.full(m, 1.0 + theta * k * 2.0 / (s2 ** 2 * h ** 2))
        inner = thomas(sub, diag, sup, rhs)
        u = np.concatenate([[0.0], inner, [0.0]])
    return h, float(np.max(np.abs(u - exact(t_end))))

print(f"{'h':>10}{'error':>13}{'ratio':>9}")
prev = None
hs, errs = [], []
for n in (21, 41, 81, 161):
    steps = max(int(round(0.2 / (0.4 * (1.0 / (n - 1)) ** 2))), 1)
    h, e = moving_run(n, steps, 0.2)
    shown = "" if prev is None else f"{prev / e:.2f}"
    print(f"{h:>10.5f}{e:>13.3e}{shown:>9}")
    prev = e
    hs.append(h); errs.append(e)
print(f"\nfitted order {float(np.polyfit(np.log(hs), np.log(errs), 1)[0]):.4f}")
```

The remapping costs **no order at all**: the fitted order is second, the same as the fixed domain
problem. What it costs is elsewhere. The convection term is differenced centrally, so it brings
lesson 75's cell number condition with it, $h\lvert\xi s'/s\rvert < 2/\alpha \cdot s^2$, and a
rapidly shrinking domain will violate that long before accuracy becomes the issue. The matrix also
stops being symmetric, which rules out the conjugate gradient solvers of Part 4 in two dimensions.

### 3.4 A source term

Add $f(x,t)$ and average it with the same weights, $(1-\theta)f^n + \theta f^{n+1}$, which is what
keeps Crank-Nicolson second order in time.

The manufactured solution is $u = e^{-t}\sin(\pi x) + t\,x(1-x)$, chosen so that neither the
initial condition nor the source is an eigenfunction.

```python
def source_run(n, steps, t_end, theta):
    x = np.linspace(0.0, 1.0, n)
    h = float(x[1] - x[0])
    k = t_end / steps
    r = k / h ** 2
    exact = lambda t: math.exp(-t) * np.sin(math.pi * x) + t * x * (1.0 - x)
    source = lambda t: ((math.pi ** 2 - 1.0) * math.exp(-t) * np.sin(math.pi * x)
                        + x * (1.0 - x) + 2.0 * t)
    u = exact(0.0)
    m = n - 2
    for step in range(steps):
        t_now, t_next = step * k, (step + 1) * k
        d2 = u[2:] - 2.0 * u[1:-1] + u[:-2]
        rhs = (u[1:-1] + (1.0 - theta) * r * d2
               + k * ((1.0 - theta) * source(t_now)[1:-1] + theta * source(t_next)[1:-1]))
        ends = exact(t_next)
        rhs[0] += theta * r * ends[0]
        rhs[-1] += theta * r * ends[-1]
        inner = (rhs if theta == 0.0 else
                 thomas(np.full(m, -theta * r), np.full(m, 1.0 + 2.0 * theta * r),
                        np.full(m, -theta * r), rhs))
        u = np.concatenate([[ends[0]], inner, [ends[-1]]])
    return u, float(np.max(np.abs(u - exact(t_end))))

print("refining space and time together at r = 0.4:")
for theta in (0.0, 0.5, 1.0):
    hs, errs = [], []
    for n in (11, 21, 41, 81):
        h = 1.0 / (n - 1)
        steps = max(int(round(0.05 / (0.4 * h ** 2))), 1)
        _, e = source_run(n, steps, 0.05, theta)
        hs.append(h); errs.append(e)
    print(f"  theta {theta:>4}: order in h "
          f"{float(np.polyfit(np.log(hs), np.log(errs), 1)[0]):.4f}, finest {errs[-1]:.3e}")
```

Second order in $h$ for every member, as it must be. The time order needs the same care as lesson
78 section 5, and here it needs more: at a fine grid the **space** error is $7.8\times10^{-6}$ and
Crank-Nicolson's time error on this solution is already below that at the coarsest step in the
sweep. Fitting the total error in $k$ then reads $-0.02$, which is the error sitting on the space
floor and not moving.

The fix is to measure the time error on its own, against a reference computed on the **same grid**
with a tiny step, so the space error cancels out of the comparison.

```python
n = 201
print(f"\n{'theta':>7}{'time order':>13}{'total order':>14}")
for theta in (0.5, 1.0):
    reference, _ = source_run(n, 8192, 0.05, theta)
    ks, time_errors, totals = [], [], []
    for steps in (4, 8, 16, 32, 64):
        u, total = source_run(n, steps, 0.05, theta)
        ks.append(0.05 / steps)
        time_errors.append(float(np.max(np.abs(u - reference))))
        totals.append(total)
    print(f"{theta:>7.1f}{float(np.polyfit(np.log(ks), np.log(time_errors), 1)[0]):>13.4f}"
          f"{float(np.polyfit(np.log(ks), np.log(totals), 1)[0]):>14.4f}")
    print(f"        time  {np.array2string(np.asarray(time_errors), precision=3)}")
    print(f"        total {np.array2string(np.asarray(totals), precision=3)}")
```

Crank-Nicolson: time order **2.0003**, total order $-0.02$. The backward scheme: time order
**0.9871**, total order $0.85$. Both time orders are exactly what the derivation says, and the
total column measures the grid.

### 3.5 The fourth order member

Lesson 79 derives the truncation constant of the family as $\alpha h^2\left[(\frac12 - \theta)r -
\frac{1}{12}\right]u_{xxxx}$, which vanishes at

$$
\theta^{*} = \frac12 - \frac{1}{12r}.
$$

Two questions follow: is it fourth order, and is it stable at the $r$ it was tuned for?

```python
print(f"{'r':>7}{'theta*':>10}{'order in h':>13}{'stability limit':>18}{'r inside it':>13}")
problem = pb.sine_problem()
for ratio in (0.2, 0.25, 0.4, 1.0, 2.0):
    theta = 0.5 - 1.0 / (12.0 * ratio)
    hs, errs = [], []
    for n in (11, 21, 41, 81):
        h = 1.0 / (n - 1)
        k = ratio * h ** 2
        steps = max(int(round(0.05 / k)), 1)
        run = pb.solve(problem, n, steps, steps * k, theta=max(theta, 0.0))
        hs.append(h); errs.append(run["error"])
    limit = pb.stability_limit(max(theta, 0.0))
    print(f"{ratio:>7.2f}{theta:>10.4f}"
          f"{float(np.polyfit(np.log(hs), np.log(errs), 1)[0]):>13.3f}"
          f"{limit:>18.4f}{str(ratio <= limit):>13}")
```

Fourth order at every ratio, 3.95 to 4.01, and **always stable**, with room to spare. That is not a
coincidence. Substituting $\theta^{*}$ into the limit $1/(2 - 4\theta)$ gives

$$
\frac{1}{2 - 4\left(\frac12 - \frac{1}{12r}\right)} = \frac{1}{\frac{1}{3r}} = 3r ,
$$

so the member designed for mesh ratio $r$ is stable up to $3r$: **a factor of 3 of margin, at every
$r$.** The design and the stability condition never conflict, which is a pleasant and not obvious
fact about this family.

The constraint that does bind is $\theta^{*} \ge 0$, which needs $r \ge 1/6$. Below that the
cancelling $\theta$ would be negative, which is outside the family, and $r = 1/6$ with $\theta = 0$
is lesson 78's lucky ratio: the same result read along the other axis.

### 4.1 How long an unstable run survives

The worst mode grows by $\lvert g(\pi)\rvert$ per step from whatever amplitude it starts with, so
the run stops looking reasonable after roughly

$$
N \approx \frac{\log(\text{threshold} / \text{seed})}{\log\lvert g(\pi)\rvert}
$$

steps. Taking the threshold to be ten times the data's own peak:

```python
print(f"{'data':>7}{'seed':>12}{'r':>7}{'|g|':>8}{'predicted':>12}{'measured':>11}{'ratio':>8}")
for name, prob in (("sine", pb.sine_problem()), ("step", pb.step_problem())):
    x = np.linspace(0.0, 1.0, 41)
    start = np.asarray(prob["initial"](x), dtype=float)
    start[0] = start[-1] = 0.0
    seed = max(pb.worst_mode_amplitude(start), float(np.finfo(float).eps))
    scale = float(np.max(np.abs(start)))
    for r in (0.55, 0.65, 0.75, 1.0):
        g = abs(float(pb.growth_factor(r, np.pi, 0.0)))
        predicted = math.log(10.0 * scale / seed) / math.log(g)
        u = start.copy()
        survived = None
        with np.errstate(over="ignore", invalid="ignore"):
            for step in range(1, 8001):
                u = pb.theta_step(u, r, 0.0)
                if not np.isfinite(u).all() or float(np.max(np.abs(u))) > 10.0 * scale:
                    survived = step
                    break
        print(f"{name:>7}{seed:>12.2e}{r:>7.2f}{g:>8.3f}{predicted:>12.1f}"
              f"{survived:>11}{survived / predicted:>8.3f}")
```

The prediction is right to within 15 per cent on the smooth data and within 20 per cent on the
jump, over a range of survival times from 5 steps to 240. The two data sets differ by 14 orders of
magnitude in the seed, and the formula absorbs all of it.

The systematic direction is worth noting: the prediction is a little **low** for the smooth data
(ratios 1.09 to 1.14) and a little **high** for the jump (0.82 to 0.95). Rounding has to
accumulate before it acts as a seed at all, which delays the smooth runs; and on the jump the other
modes are also large, so the threshold is crossed slightly before the worst mode alone would cross
it.

### 4.2 Where the ringing becomes visible

Two different questions, with two different answers.

**When does the worst mode flip sign?** Whenever $g(\pi) = (1-2r)/(1+2r) < 0$, that is for
**every** $r > 1/2$. There is no threshold to find beyond that.

**When does the profile dip below the data's own range?** That needs the flipped mode to be large
enough to overcome the smooth part, and it happens far later.

```python
print(f"{'r':>7}{'g(pi)':>10}{'undershoot':>13}")
for r in (0.6, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 8.0, 20.0, 50.0):
    run = pb.solve(pb.step_problem(), 41, 1, r * (1.0 / 40) ** 2, theta=0.5)
    print(f"{r:>7.2f}{float(pb.growth_factor(r, np.pi, 0.5)):>10.4f}"
          f"{max(-float(np.min(run['u'])), 0.0):>13.3e}")

lo, hi = 0.51, 60.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    run = pb.solve(pb.step_problem(), 41, 1, mid * (1.0 / 40) ** 2, theta=0.5)
    if max(-float(np.min(run["u"])), 0.0) > 1e-9:
        hi = mid
    else:
        lo = mid
r = 0.5 * (lo + hi)
print(f"\nthe profile first dips below zero at r = {r:.4f}, "
      f"where g(pi) = {float(pb.growth_factor(r, np.pi, 0.5)):.4f}")
```

The onset is at $r = 4.00$, where $g(\pi) = -0.778$, and it is **abrupt**: exactly zero at
$r = 3.5$, $1.9\times10^{-6}$ at $r = 4.0$, $0.073$ at $r = 5$ and $0.22$ at $r = 8$. The
abruptness is a discretization artefact rather than a feature of the scheme: the undershoot is
measured at grid nodes, and the profile's minimum crosses zero at a node only once $r$ is large
enough, however smoothly the underlying continuum quantity varies.

So "Crank-Nicolson rings above $r = 1/2$" and "Crank-Nicolson undershoots above $r = 4$" are both
true and are different statements. The first is about a mode and the second is about a plot.

### 4.3 The cost exponents

```python
targets = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
costs = {}
for target in targets:
    out = pb.cost_at_equal_accuracy(target=target)
    for row in out["rows"]:
        costs.setdefault(row["scheme"], []).append(row["updates"])
    print(f"tol {target:.0e}: " + ", ".join(
        f"{row['scheme']} {row['updates']}" for row in out["rows"]))
print()
for name, values in costs.items():
    exponent = float(np.polyfit(np.log(np.asarray(targets)),
                                np.log(np.asarray(values, dtype=float)), 1)[0])
    predicted = -1.0 if name == "crank nicolson" else -1.5
    print(f"{name:>16}: measured {exponent:>7.3f}, predicted {predicted}")
```

The backward scheme fits $-1.522$ against a predicted $-1.5$. The other two do not fit as well:
explicit $-1.102$ and Crank-Nicolson $-0.741$.

**That is a limitation of the search, not a contradiction of the theory.** The search picks a grid
from a list whose sizes roughly double and a step count from powers of two, so the cost it reports
jumps in factors of four or eight and cannot resolve an exponent to better than about 0.3 over
five points. The backward scheme fits best because its costs are the largest, so the quantization
is relatively smallest.

What survives the quantization is the ordering and the gap, and those are unambiguous: at
$10^{-4}$ Crank-Nicolson needs 312 updates, the explicit scheme 12480 and the backward scheme
161792. **A factor of 40 and 519**, and both grow as the tolerance tightens.

### 5.1 Rannacher startup

Replace the first few Crank-Nicolson steps by **pairs of half sized backward Euler steps**, so the
total time advanced is the same. Backward Euler is L-stable, so it annihilates the high modes
instead of flipping them, and Crank-Nicolson then starts from data those modes have already left.

```python
def rannacher(problem, n, steps, t_end, backward_steps):
    x = np.linspace(0.0, 1.0, n)
    h = float(x[1] - x[0])
    k = t_end / steps
    r = k / h ** 2
    u = np.asarray(problem["initial"](x), dtype=float)
    u[0] = u[-1] = 0.0
    for step in range(steps):
        if step < backward_steps:
            u = pb.theta_step(u, 0.5 * r, 1.0)
            u = pb.theta_step(u, 0.5 * r, 1.0)
        else:
            u = pb.theta_step(u, r, 0.5)
    return float(np.max(np.abs(u - problem["exact"](x, steps * k))))

shown_steps = 8
print(f"the error at r = 20 over {shown_steps} steps, "
      "against the number of backward starts:")
for backward in (0, 1, 2, 3, 4, 6):
    err = rannacher(pb.step_problem(), 41, shown_steps,
                    shown_steps * 20.0 * (1.0 / 40) ** 2, backward)
    print(f"  {backward:>2} backward: {err:.4e}")
```

One Rannacher step cuts the error by a factor of 21, from $0.195$ to $0.0092$, and further steps
change almost nothing.

**Why two suffice** is a one line calculation. Backward Euler's factor at $\phi = \pi$ is
$1/(1 + 4r)$, and a Rannacher step is two of them at $r/2$, so it damps that mode by
$\left(1/(1+2r)\right)^2$. At $r = 20$ that is $5.9\times10^{-4}$ per step, so after two the worst
mode is at $3.5\times10^{-7}$ and there is nothing left for Crank-Nicolson to ring with.

```python
r = 20.0
half = float(pb.growth_factor(0.5 * r, np.pi, 1.0)) ** 2
print(f"one Rannacher step damps the worst mode by {half:.3e}")
for backward in (0, 1, 2, 3):
    factor = 1.0
    for step in range(shown_steps):
        factor *= half if step < backward else float(pb.growth_factor(r, np.pi, 0.5))
    print(f"  {backward} backward: the worst mode's total factor over "
          f"{shown_steps} steps {factor:+.3e}")
```

**Is second order retained?** The question needs splitting, because it presumes there was second
order to retain.

```python
print(f"\n{'data':>13}{'backward starts':>18}{'order':>9}")
for name, prob in (("smooth mode", pb.sine_problem()), ("a jump", pb.step_problem())):
    for backward in (0, 2):
        hs, errs = [], []
        for n in (21, 41, 81, 161):
            h = 1.0 / (n - 1)
            k = 2.0 * h ** 2
            steps = max(int(round(0.02 / k)), 1)
            errs.append(rannacher(prob, n, steps, steps * k, min(backward, steps)))
            hs.append(h)
        print(f"{name:>13}{backward:>18}"
              f"{float(np.polyfit(np.log(hs), np.log(errs), 1)[0]):>9.4f}")
```

On **smooth** data the answer is yes: 1.95 without and 2.62 with, both consistent with second order
given that the second run's coarsest grid is still in a transient.

On the **jump** the answer is that neither is second order: 1.16 without and 1.04 with. Crank-
Nicolson on discontinuous data is limited by the data's regularity, not by the scheme's order, and
Rannacher does not change that. **What Rannacher fixes is the ringing, not the order**, and the
common summary "it restores second order" is describing a problem whose data was smooth enough to
have it in the first place.

### 5.2 The discrete maximum principle

The scheme satisfies a discrete maximum principle when the new value is a **convex combination**
of old values: all the explicit weights nonnegative and summing to 1. The neighbour weights are
$(1-\theta)r \ge 0$ automatically, so the condition is on the centre:

$$
1 - 2(1-\theta)r \ge 0
\quad\Longleftrightarrow\quad
r \le \frac{1}{2(1-\theta)} .
$$

Compare that with the stability limit $r \le 1/(2 - 4\theta)$. For $\theta = 0$ both give $1/2$.
For $\theta = 1/2$ the maximum principle needs $r \le 1$ while stability is unconditional. So the
maximum principle is **strictly stronger** for every $\theta > 0$.

```python
print(f"{'theta':>7}{'r':>7}{'centre weight':>16}{'convex':>9}{'undershoot':>13}")
for theta in (0.0, 0.25, 0.5, 1.0):
    for r in (0.25, 0.5, 1.0, 2.0, 8.0):
        centre = 1.0 - 2.0 * (1.0 - theta) * r
        run = pb.solve(pb.step_problem(), 41, 4, 4 * r * (1.0 / 40) ** 2, theta=theta)
        print(f"{theta:>7.2f}{r:>7.2f}{centre:>16.4f}{str(centre >= -1e-14):>9}"
              f"{max(-float(np.min(run['u'])), 0.0):>13.3e}")
```

The table says two things, and only one of them is the theorem.

**Every run with nonnegative weights has zero undershoot.** That is the maximum principle, and it
holds without exception.

**The converse is false.** $\theta = 1/2$ at $r = 2$ and $r = 8$ has a negative centre weight and
still shows no undershoot at all, and so does $\theta = 1/4$ at $r = 1$. The condition is
**sufficient and not necessary**: violating it makes an undershoot possible, not certain, and
whether one appears depends on the data, as exercise 4.2 measured.

The $\theta = 0$ rows at $r \ge 1$ are a different phenomenon entirely: those undershoots are 9.0
and $1.2\times10^{5}$, which is not a violated maximum principle but instability.

### 5.3 Why Du Fort and Frankel is not used

It is explicit and unconditionally stable, which sounds decisive. Work out what step it would
actually need.

Its error has two parts: the space error $O(h^2)$ and the modified equation term $O((k/h)^2)$ from
exercise 2.5. To reach a tolerance, both must fit inside it. Given $h$, the largest usable $k/h$ is
set by whatever the space error leaves over, and the step count is $T/(k) = T/((k/h)h)$.

```python
print(f"{'target':>9}{'best k/h':>11}{'h':>11}{'k':>12}{'DFF steps':>12}"
      f"{'explicit steps':>16}{'ratio':>8}")
for target in (1e-3, 1e-4, 1e-5, 1e-6):
    best = None
    for power in np.linspace(0.5, 4.5, 400):
        h = 10.0 ** (-power)
        space = 0.9 * h ** 2
        if space >= target:
            continue
        ratio = math.sqrt((target - space) / 3.0)
        steps = 0.05 / (ratio * h)
        if best is None or steps < best[0]:
            best = (steps, ratio, h)
    steps, ratio, h = best
    explicit = 0.05 / (0.5 * h ** 2)
    print(f"{target:>9.0e}{ratio:>11.4f}{h:>11.2e}{ratio * h:>12.2e}"
          f"{steps:>12.0f}{explicit:>16.0f}{steps / explicit:>8.3f}")
```

**Du Fort and Frankel needs 92 per cent of the explicit scheme's step count**, at every tolerance
from $10^{-3}$ to $10^{-6}$. The ratio barely moves.

The reason is in the two constraints. Accuracy forces $k/h$ small; stability forces the explicit
scheme's $k/h^2 \le 1/2$, that is $k/h \le h/2$. Both make $k$ proportional to $h$ times something
small, and the two somethings turn out to be within 10 per cent of each other on this problem.

So the unconditional stability is real and **buys nothing**. It removes a constraint that was not
the binding one. That is the general shape of the answer whenever a method's advertised advantage
is a bound that accuracy was already enforcing, and it is why "unconditionally stable" is a claim
worth costing rather than accepting.

---

## Lesson 79, Consistency, Convergence and Stability

### 1.1 The three properties in one sentence each

**Consistency**: the exact solution, substituted into the difference formula, leaves a residual
that goes to zero as the grid is refined.

**Stability**: the operator that advances one step has powers that stay bounded as the grid is
refined, uniformly over the steps taken.

**Convergence**: the computed solution approaches the exact one as the grid is refined.

**Only convergence needs the exact solution** in its statement, and consistency needs it in its
test but not in its conclusion: the truncation error is computed by expanding a general smooth
function, so it is a property of the formula. Stability mentions no solution at all, which is why
it is the one that can be checked without knowing anything about the problem's answer.

That asymmetry is what makes the Lax theorem useful. The property you want is the one you cannot
check; the two you can check imply it.

### 1.2 The Lax equivalence theorem

**For a well posed linear initial value problem and a consistent finite difference scheme,
stability is necessary and sufficient for convergence.**

The assumptions are worth listing because each one is doing work.

**Linear.** The proof writes the error as the difference of two solutions of the same operator, and
that step needs superposition. For nonlinear problems there are local analogues and no theorem this
clean.

**Well posed.** The continuous problem must have a solution, that solution must be unique, and it
must depend continuously on the data. Without the third, "convergence" has nothing to converge to
in any stable sense, and lesson 77's backward heat equation is the example.

**Initial value problem.** The theorem is about marching. Elliptic problems are not covered and do
not need to be.

**Consistent.** This is an assumption and not a conclusion. It is exactly the assumption Du Fort
and Frankel's scheme violates at fixed $k/h$, and the theorem says nothing about it there.

### 1.3 Why the two stability tests coincide for the heat equation

Three facts, and each is needed.

**$T$ is symmetric.** The second difference matrix has $-2$ on the diagonal and $1$ symmetrically
off it, and the Dirichlet boundary does not break that. So $G = (I - \theta rT)^{-1}(I +
(1-\theta)rT)$ is a rational function of a symmetric matrix, and is symmetric too.

**A symmetric matrix has $\rho = \lVert\cdot\rVert_2$.** Its eigenvalues are its singular values up
to sign, so the spectral radius and the 2-norm are the same number, and there is no gap between
the limit and the first step for the powers to hide in.

**The eigenvectors of $T$ are the discrete sine modes.** Exercise 2.3 shows it. So the eigenvalues
of $G$ are exactly $g(\phi)$ evaluated at $\phi = j\pi h$, which are exactly the phases the grid
carries. The matrix method's eigenvalues and von Neumann's symbols are the **same list of
numbers**.

The lesson measures the largest disagreement across 48 combinations of $\theta$, $r$ and grid size
at $10^{-14}$.

```python
import math

import numpy as np
from nalib import parabolic as pb
from nalib import pdestability as ps

out = ps.the_two_tests_agree_on_the_heat_equation()
print(f"{out['cases']} combinations, worst eigenvalue gap "
      f"{out['worst_eigenvalue_gap']:.3e}")
print(f"every matrix normal: {out['every_matrix_is_normal']}")
print(f"spectral radius equals the 2-norm everywhere: "
      f"{out['radius_equals_norm_everywhere']}")
```

### 1.4 What the spectral radius does and does not say

**It says**: $\lVert G^n\rVert^{1/n} \to \rho(G)$ for every matrix, so the powers eventually decay
like $\rho^n$ and $\rho \le 1$ is necessary for them to stay bounded.

**It does not say** anything about small $n$. $\lVert G\rVert$ can be much larger than $\rho(G)$,
in which case $\lVert G^n\rVert$ can grow before it decays; and even when it cannot grow, the rate
at which it decays can be far slower than $\rho^n$ for a long time.

On this lesson's convection scheme the second happens and the first does not. Sweeping the whole
stable region found $\lVert G\rVert_2$ just below 1 every time, so nothing ever grows, and the
observed rate takes 244 steps to come within 10 per cent of $\rho = 0.594$.

```python
out = ps.the_spectral_radius_is_only_the_limit()
print(f"{'Pe':>6}{'rho':>10}{'||G||':>10}{'peak':>10}{'rate at 20':>13}"
      f"{'steps to rho':>15}")
for row in out["rows"]:
    steps = "never" if row["steps_to_the_asymptotic_rate"] < 0 \
        else str(row["steps_to_the_asymptotic_rate"])
    print(f"{row['peclet']:>6.2f}{row['spectral_radius']:>10.5f}{row['two_norm']:>10.5f}"
          f"{row['peak_power_norm']:>10.5f}{row['rate_at_20']:>13.5f}{steps:>15}")
print(f"\nnothing grew anywhere in the stable region: {out['nothing_grows_anywhere']}")
```

### 1.5 Why a stable inconsistent scheme is the dangerous one

An **unstable** scheme announces itself. The numbers grow, overflow, or oscillate wildly, and
anyone looking at the output sees it. It also gets **worse** when the grid is refined, so the
standard response to a suspicious answer, run it again on a finer grid, exposes it immediately.

A **stable inconsistent** scheme does none of that. It is bounded, it produces a smooth profile, it
converges, and refining the grid makes the answer **more** stable and no more correct. Every check
that does not involve the exact solution passes.

The lesson measures it: Du Fort and Frankel at fixed $k/h$ gives errors of $0.0276$ on four
successive refinements, changing in the fourth digit. A convergence study on that scheme reports
"converged" and is right; what it does not report is that the limit is the solution of a different
equation.

The practical defence is that consistency is checked **symbolically before the code is written**,
by expanding the scheme. It is the one property of the three that can be established entirely with
a pencil, and it is the one no runtime check will find.

### 2.1 The truncation error of the weighted family

Expand about the **midpoint in time**, $(x_j, t_{n+1/2})$, which is what makes the Crank-Nicolson
cancellation visible. Write $u^{\pm} = u(x_j, t_{n+1/2} \pm k/2)$. Then

$$
\frac{u^{n+1} - u^n}{k} = u_t + \frac{k^2}{24}u_{ttt} + O(k^4),
$$

with the $u_{tt}$ term absent by symmetry, and

$$
\theta\,\delta_x^2u^{n+1} + (1-\theta)\,\delta_x^2u^n
= \delta_x^2u^{n+1/2} + \left(\theta - \tfrac12\right)k\,\delta_x^2u_t + O(k^2),
$$

since $\theta u^{n+1} + (1-\theta)u^n = u^{n+1/2} + (\theta - \frac12)k\,u_t + O(k^2)$. Using
$\delta_x^2 = u_{xx} + \frac{h^2}{12}u_{xxxx} + O(h^4)$, the truncation error
$T = \frac{u^{n+1}-u^n}{k} - \alpha[\cdots]$ is

$$
T = -\alpha\left(\theta - \tfrac12\right)k\,u_{xxt} - \frac{\alpha h^2}{12}u_{xxxx}
+ O(k^2 + h^4).
$$

On a solution, $u_{xxt} = \alpha u_{xxxx}$, so

$$
T = \alpha h^2\left[\left(\tfrac12 - \theta\right)r - \tfrac{1}{12}\right]u_{xxxx} + O(k^2 + h^4),
$$

using $\alpha k = rh^2$. That is the bracket.

Three readings of the same expression, and the lesson uses all three:

- $\theta = \frac12$ kills the $k$ term for **every** $r$, which is Crank-Nicolson's second order in
  time;
- $\theta = 0$ kills the whole bracket when $r = \frac16$, which is lesson 78's lucky ratio;
- for a given $r$, the bracket vanishes at $\theta = \frac12 - \frac{1}{12r}$, which is the fourth
  order member.

```python
out = ps.consistency_of_the_family(ratio=0.4)
print(f"{'theta':>9}{'measured':>12}{'predicted':>12}{'agrees':>9}")
for row in out["rows"]:
    print(f"{row['theta']:>9.4f}{row['constant']:>12.4f}"
          f"{row['predicted_constant']:>12.4f}{str(row['agrees']):>9}")
print(f"\nthe cancelling theta at r = 0.4 is {out['cancelling_theta']:.4f} "
      f"and its order is {out['order_at_the_cancelling_theta']:.3f}")
```

### 2.2 The consistency conditions on six weights

For

$$
d\,u_{i-1}^{n+1} + e\,u_i^{n+1} + f\,u_{i+1}^{n+1}
= a\,u_{i-1}^{n} + b\,u_i^{n} + c\,u_{i+1}^{n},
$$

expand every term about $(x_i, t_n)$. Writing $N = d + e + f$ and $O = a + b + c$ for the row sums,
$N_1 = -d + f$ and $O_1 = -a + c$ for the first moments, $N_2 = d + f$ and $O_2 = a + c$ for the
second:

$$
\text{LHS} - \text{RHS}
= (N - O)u + Nk\,u_t + h(N_1 - O_1)u_x + \frac{h^2}{2}(N_2 - O_2)u_{xx} + \cdots
$$

Matching order by order against $0 = k(u_t - \alpha u_{xx})\cdot(\text{something})$:

- **constants**: $N = O$. The two rows sum to the same thing.
- **$u_x$**: $N_1 = O_1$. The two first moments agree.
- **$u_t$ and $u_{xx}$ together**: the leading balance is $Nk\,u_t = \frac{h^2}{2}(O_2 - N_2)u_{xx}$,
  and matching that against $ku_t = k\alpha u_{xx}$ gives

$$
\frac{O_2 - N_2}{N} = \frac{2\alpha k}{h^2} = 2r .
$$

Three linear conditions on six unknowns, so a three parameter family of solutions, one scaling
degree of freedom (multiplying both rows by a constant changes nothing) and one symmetry degree of
freedom (a $u_x$ term), leaving exactly the one parameter $\theta$ family.

```python
out = ps.build_your_own_scheme(r=0.25)
print(f"{'scheme':>30}{'sum gap':>10}{'moment 1':>11}{'moment 2':>11}{'wanted':>9}"
      f"{'consistent':>12}")
for row in out["rows"]:
    print(f"{row['scheme']:>30}{row['constant_condition']:>10.2e}"
          f"{row['first_moment_condition']:>11.2e}{row['second_moment_condition']:>11.4f}"
          f"{row['second_moment_should_be']:>9.4f}{str(row['consistent']):>12}")
```

### 2.3 $G$ is symmetric and its eigenvectors are the sine modes

Let $T$ be the $m \times m$ second difference matrix. Try $v^{(j)}_i = \sin(ij\pi h)$ with
$h = 1/(m+1)$, so that the implied values at $i = 0$ and $i = m+1$ are zero, matching the Dirichlet
boundary. Then for $1 \le i \le m$,

$$
(Tv^{(j)})_i = \sin((i+1)j\pi h) - 2\sin(ij\pi h) + \sin((i-1)j\pi h)
= \left(2\cos(j\pi h) - 2\right)\sin(ij\pi h),
$$

so $v^{(j)}$ is an eigenvector with eigenvalue $\mu_j = -4\sin^2(j\pi h/2)$. There are $m$ of them
and they are orthogonal, so they diagonalise $T$: $T = Q\Lambda Q^{\mathsf T}$ with $Q$ orthogonal.

Now

$$
G = (I - \theta rT)^{-1}(I + (1-\theta)rT)
= Q\,\frac{I + (1-\theta)r\Lambda}{I - \theta r\Lambda}\,Q^{\mathsf T},
$$

which is symmetric because it is $Q$ times a diagonal times $Q^{\mathsf T}$. Its eigenvalues are

$$
\frac{1 + (1-\theta)r\mu_j}{1 - \theta r\mu_j}
= \frac{1 - 4(1-\theta)r\sin^2(j\pi h/2)}{1 + 4\theta r\sin^2(j\pi h/2)}
= g(j\pi h),
$$

which is exactly von Neumann's symbol at the phase $\phi = j\pi h$.

```python
for m in (5, 9, 17, 33):
    h = 1.0 / (m + 1)
    for theta in (0.0, 0.5, 1.0):
        for r in (0.3, 1.7):
            g = ps.amplification_matrix(theta, r, m)
            symmetric = float(np.max(np.abs(g - g.T)))
            values = np.sort(np.linalg.eigvalsh(g))
            symbols = np.sort(ps.von_neumann_symbols(theta, r, m))
            assert symmetric < 1e-13
            assert float(np.max(np.abs(values - symbols))) < 1e-12
print("symmetric to 1e-13 and the eigenvalues are the symbols to 1e-12,")
print("at 4 grid sizes, 3 values of theta and 2 mesh ratios")
```

### 2.4 The two conditions for the convection-diffusion scheme

The explicit scheme for $u_t + au_x = \alpha u_{xx}$ with a centred convection term is

$$
u_j^{n+1} = u_j^n + r\left(u_{j+1} - 2u_j + u_{j-1}\right)
- \frac{r\,\mathrm{Pe}}{2}\left(u_{j+1} - u_{j-1}\right),
$$

with $r = \alpha k/h^2$ and $\mathrm{Pe} = ah/\alpha$ the cell Peclet number. Substituting one mode
and writing $s = \sin^2(\phi/2)$:

$$
g = 1 - 4rs - i\,r\,\mathrm{Pe}\sin\phi,
\qquad
\lvert g\rvert^2 = (1 - 4rs)^2 + r^2\mathrm{Pe}^2\sin^2\phi .
$$

Use $\sin^2\phi = 4s(1-s)$ and require $\lvert g\rvert^2 \le 1$:

$$
(1 - 4rs)^2 - 1 + 4r^2\mathrm{Pe}^2s(1-s) \le 0
\quad\Longleftrightarrow\quad
s\left[\left(16r^2 - 4r^2\mathrm{Pe}^2\right)s + 4r^2\mathrm{Pe}^2 - 8r\right] \le 0 .
$$

For $s \in (0,1]$ the bracket must be at most zero. It is linear in $s$, so checking the two ends
suffices:

- $s \to 0$: $4r^2\mathrm{Pe}^2 - 8r \le 0$, that is $r\,\mathrm{Pe}^2 \le 2$;
- $s = 1$: $16r^2 - 8r \le 0$, that is $r \le \tfrac12$.

Both are needed and together they are sufficient. The second is the diffusion limit; **the first is
the one that bites when convection dominates**, since it caps $r$ at $2/\mathrm{Pe}^2$, which at
$\mathrm{Pe} = 8$ is $0.031$, sixteen times below the diffusion limit.

```python
print(f"{'Pe':>6}{'r':>8}{'r Pe^2 <= 2':>14}{'r <= 1/2':>11}{'predicted':>12}"
      f"{'measured':>11}")
for peclet, r in ((0.5, 0.4), (2.0, 0.45), (2.0, 0.6), (8.0, 0.1), (8.0, 0.03)):
    phases = np.linspace(0.0, np.pi, 2001)
    s = np.sin(phases / 2.0) ** 2
    g = np.abs(1.0 - 4.0 * r * s - 1j * r * peclet * np.sin(phases))
    print(f"{peclet:>6.1f}{r:>8.3f}{str(r * peclet ** 2 <= 2.0):>14}"
          f"{str(r <= 0.5):>11}"
          f"{str(ps.convection_symbol_is_stable(peclet, r)):>12}"
          f"{str(bool(np.max(g) <= 1.0 + 1e-12)):>11}")
```

### 2.5 Why $\lVert G^n\rVert^{1/n} \to \rho(G)$

**Lower bound.** If $\lambda$ is an eigenvalue with unit eigenvector $v$ then
$\lVert G^nv\rVert = \lvert\lambda\rvert^n$, so $\lVert G^n\rVert \ge \rho^n$ and
$\lVert G^n\rVert^{1/n} \ge \rho$ for every $n$.

**Upper bound.** Fix $\varepsilon > 0$. There is a norm in which $\lVert G\rVert_{\varepsilon} \le
\rho + \varepsilon$: put $G$ in Jordan form $G = SJS^{-1}$, scale the basis by
$D = \mathrm{diag}(1, \delta, \delta^2, \dots)$ so that the superdiagonal entries of
$D^{-1}JD$ are $\delta$ rather than 1, and choose $\delta \le \varepsilon$. Then
$\lVert G^n\rVert_{\varepsilon} \le (\rho + \varepsilon)^n$, and since all norms on a finite
dimensional space are equivalent, $\lVert G^n\rVert \le C_{\varepsilon}(\rho + \varepsilon)^n$ with
a constant that does not depend on $n$. Taking $n$-th roots, $\limsup \lVert G^n\rVert^{1/n} \le
\rho + \varepsilon$, and $\varepsilon$ was arbitrary.

**Where normality comes in.** The constant $C_{\varepsilon}$ is $\kappa(SD)$, the conditioning of
the basis that diagonalises $G$. For a **normal** matrix $S$ can be taken orthogonal and no scaling
is needed, so $C = 1$ and the bound $\lVert G^n\rVert \le \rho^n$ holds at $n = 1$ already, with
equality. For a non-normal matrix $C$ can be enormous, and the convergence of the $n$-th roots is
governed by how long it takes $\rho^n$ to overcome it, which is $\log C/\log(1/\rho)$ steps: the
delay the lesson measures at 244.

```python
out = ps.the_spectral_radius_is_only_the_limit()
print(f"{'Pe':>6}{'normal':>9}{'rho':>10}{'departure':>13}{'delay':>9}"
      f"{'log(kappa)/log(1/rho)':>24}")
for row in out["rows"]:
    g = (ps.convection_diffusion_matrix(row["peclet"], row["r"], out["unknowns"])
         if row["peclet"] else ps.amplification_matrix(0.0, row["r"], out["unknowns"]))
    values, vectors = np.linalg.eig(g)
    kappa = float(np.linalg.cond(vectors))
    predicted = (math.log(kappa) / math.log(1.0 / row["spectral_radius"])
                 if row["spectral_radius"] < 1.0 else float("inf"))
    delay = "never" if row["steps_to_the_asymptotic_rate"] < 0 \
        else str(row["steps_to_the_asymptotic_rate"])
    print(f"{row['peclet']:>6.2f}{str(row['normal']):>9}{row['spectral_radius']:>10.5f}"
          f"{row['departure_from_normality']:>13.3e}{delay:>9}{predicted:>24.1f}")
```

### 3.1 The matrix method with a Neumann boundary

Build the one step operator with the ghost point row of lesson 77 in place of the first Dirichlet
row, and bisect on the spectral radius.

```python
import math

def matrix_for(m, r, theta, boundary):
    t = (np.diag(np.full(m, -2.0)) + np.diag(np.ones(m - 1), 1)
         + np.diag(np.ones(m - 1), -1))
    if boundary == "neumann":
        t[0, 1] = 2.0                       # the ghost point elimination
    elif boundary == "periodic":
        t[0, -1] = 1.0
        t[-1, 0] = 1.0
    return np.linalg.solve(np.eye(m) - theta * r * t,
                           np.eye(m) + (1.0 - theta) * r * t)

def limit_for(m, theta, boundary, hi=10.0):
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g = matrix_for(m, mid, theta, boundary)
        if float(np.max(np.abs(np.linalg.eigvals(g)))) <= 1.0 + 1e-12:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

print(f"{'boundary':>11}{'theta':>7}" + "".join(f"{m:>12}" for m in (10, 20, 40, 80)))
for boundary in ("dirichlet", "neumann", "periodic"):
    for theta in (0.0, 0.5):
        limits = "".join(f"{limit_for(m, theta, boundary):>12.6f}"
                         for m in (10, 20, 40, 80))
        print(f"{boundary:>11}{theta:>7.1f}{limits}")
```

**The limit does not change, and the way it approaches $1/2$ does.**

The periodic case gives exactly $0.500000$ at every size, because a periodic grid carries the phase
$\phi = \pi$ exactly and the worst case is attained. Dirichlet and Neumann carry a largest phase
slightly **below** $\pi$, so they allow a mesh ratio slightly above $1/2$, by $O(h^2)$: the
Dirichlet limits are $0.5103, 0.5028, 0.5007, 0.5002$, falling by four each refinement.

The Neumann case converges from the same side and about twice as fast, because the ghost point row
adds a mode nearer the smooth end of the spectrum rather than the rough end.

At $\theta = 1/2$ the bisection runs to the top of its bracket for all three, which is
unconditional stability being reported as "no limit in this range".

### 3.2 A variable coefficient against a frozen coefficient test

The frozen coefficient test replaces $\alpha(x)$ by a constant at each point, applies von Neumann
there, and takes the worst case. It is a heuristic, not a theorem, so it is worth measuring how
close it gets.

```python
def variable_limit(m, alpha_of, theta=0.0, hi=2.0):
    h = 1.0 / (m + 1)
    faces = alpha_of(np.linspace(0.0, 1.0, m + 2)[:-1] + 0.5 * h)
    t = np.zeros((m, m))
    for i in range(m):
        t[i, i] = -(faces[i] + faces[i + 1])
        if i > 0:
            t[i, i - 1] = faces[i]
        if i + 1 < m:
            t[i, i + 1] = faces[i + 1]
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g = np.linalg.solve(np.eye(m) - theta * mid * t,
                            np.eye(m) + (1.0 - theta) * mid * t)
        if float(np.max(np.abs(np.linalg.eigvals(g)))) <= 1.0 + 1e-12:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), float(np.max(faces))

print(f"{'alpha range':>16}{'matrix method':>16}{'frozen worst case':>20}{'ratio':>9}")
for top in (1.0, 3.0, 10.0):
    alpha_of = lambda x, s=top: 1.0 + (s - 1.0) * x
    measured, worst = variable_limit(60, alpha_of)
    frozen = 0.5 / worst
    print(f"{'1 to ' + str(top):>16}{measured:>16.6f}{frozen:>20.6f}{measured / frozen:>9.4f}")
```

The frozen test is **conservative and close**: it under-predicts the true limit by 0.07 per cent
when $\alpha$ is constant, 7.7 per cent when it varies by a factor of 3, and 9.5 per cent when it
varies by a factor of 10. It never over-predicts on these runs, which is the property that makes
it usable.

Conservative is not the same as safe in general, and exercise 5.1 constructs the case where it is
not.

### 3.3 The pseudospectrum and the Kreiss bound

The $\varepsilon$-pseudospectrum is the set of $z$ with
$\sigma_{\min}(zI - G) \le \varepsilon$, equivalently the set of eigenvalues of $G + E$ over all
perturbations with $\lVert E\rVert \le \varepsilon$. The Kreiss matrix theorem bounds the powers by
the **Kreiss constant**

$$
K = \sup_{\varepsilon > 0}\frac{\rho_{\varepsilon}(G) - 1}{\varepsilon},
\qquad
K \le \sup_n\lVert G^n\rVert \le enK ,
$$

where $\rho_{\varepsilon}$ is the largest modulus in the $\varepsilon$-pseudospectrum.

```python
def pseudospectral_radius(g, epsilon, angles=121):
    """The largest |z| with sigma_min(zI - G) <= epsilon.

    Bisecting **on the radius** rather than along each ray. The set of radii that meet the
    pseudospectrum is not an interval containing zero: for a normal matrix the pseudospectrum is a
    union of discs around the eigenvalues, so along a ray it is a union of intervals, and a
    bisection started at the origin finds the edge of whichever component the origin is in. The
    question with a single answer is the outermost one: is any point of the circle of radius R
    inside?
    """
    n = g.shape[0]
    spectral = float(np.max(np.abs(np.linalg.eigvals(g))))
    directions = np.exp(1j * np.linspace(0.0, np.pi, angles))

    def touches(radius):
        for direction in directions:
            z = radius * direction
            if float(np.min(np.linalg.svd(z * np.eye(n) - g, compute_uv=False))) <= epsilon:
                return True
        return False

    lo, hi = spectral, spectral + 1.0 + 4.0 * epsilon
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if touches(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

from nalib import pdestability as ps

print(f"{'Pe':>6}{'rho':>10}{'peak ||G^n||':>15}" + "".join(
    f"{'radius at ' + f'{e:.0e}':>20}" for e in (1e-2, 1e-3, 1e-4)))
for peclet, r in ((0.0, 0.4), (1.9, 0.4), (2.0, 0.45)):
    g = (ps.convection_diffusion_matrix(peclet, r, 20) if peclet
         else ps.amplification_matrix(0.0, r, 20))
    rho = float(np.max(np.abs(np.linalg.eigvals(g))))
    norms, current = [], np.eye(g.shape[0])
    for _ in range(300):
        current = current @ g
        norms.append(float(np.linalg.norm(current, 2)))
    radii = "".join(f"{pseudospectral_radius(g, e):>20.5f}" for e in (1e-2, 1e-3, 1e-4))
    print(f"{peclet:>6.2f}{rho:>10.5f}{float(np.max(norms)):>15.5f}{radii}")

print()
print(f"{'Pe':>6}" + "".join(f"{'K at ' + f'{e:.0e}':>16}" for e in (1e-2, 1e-3, 1e-4)))
for peclet, r in ((0.0, 0.4), (1.9, 0.4), (2.0, 0.45)):
    g = (ps.convection_diffusion_matrix(peclet, r, 20) if peclet
         else ps.amplification_matrix(0.0, r, 20))
    kreiss = "".join(f"{(pseudospectral_radius(g, e) - 1.0) / e:>16.3f}"
                     for e in (1e-2, 1e-3, 1e-4))
    print(f"{peclet:>6.2f}{kreiss}")
```

Three things come out, and they agree with the lesson from a completely different direction.

**The normal matrix behaves exactly as the theory says it must.** For a symmetric $G$,
$\sigma_{\min}(zI - G)$ **is** the distance from $z$ to the spectrum, so the
$\varepsilon$-pseudospectrum is precisely the $\varepsilon$-neighbourhood of the eigenvalues and
its radius is $\rho + \varepsilon$. Measured: $1.00106$, $0.99206$ and $0.99116$ against
$\rho = 0.99106$, which is $\rho + \varepsilon$ to five digits at each of the three.

**The non-normal ones are much larger than that.** At $\mathrm{Pe} = 1.9$ the spectral radius is
$0.447$ and the $10^{-2}$-pseudospectral radius is $0.895$: a perturbation of one part in a hundred
moves an eigenvalue further than the whole spectrum's own radius. That gap **is** the
non-normality, measured directly rather than through a commutator.

**And nothing here grows.** Every radius measured for the non-normal matrices is below 1, so
$\rho_\varepsilon - 1 < 0$ and the Kreiss constant is 1, its smallest possible value. The theorem
then bounds $\sup_n\lVert G^n\rVert$ from below by 1 and says nothing useful above, which is exactly
consistent with the measured peak of $0.99890$: **nothing grows, and the pseudospectrum knew.**

The one row where the Kreiss constant is positive is the symmetric matrix at $\varepsilon = 10^{-2}$,
where $\rho + \varepsilon$ just exceeds 1. That is not a prediction of growth either: it is the
statement that a perturbation of $10^{-2}$ would make this matrix unstable, which is true and is a
question about a different matrix.

### 3.4 Searching the weights for the highest order

The three consistency conditions leave the $\theta$ family, so the search is a search over
$\theta$, and lesson 79 says the truncation constant is proportional to
$\left(\frac12 - \theta\right)r - \frac{1}{12}$.

```python
print(f"{'r':>9}{'ideal theta':>13}{'best found':>13}{'gap':>10}{'|bracket|':>12}"
      f"{'max |g|':>10}{'reachable':>11}")
for r in (0.1, 1.0 / 6.0, 0.25, 0.5, 1.0, 4.0):
    ideal = 0.5 - 1.0 / (12.0 * r)
    best = None
    for theta in np.linspace(0.0, 1.0, 4001):
        old = np.asarray([(1.0 - theta) * r, 1.0 - 2.0 * (1.0 - theta) * r,
                          (1.0 - theta) * r])
        new = np.asarray([-theta * r, 1.0 + 2.0 * theta * r, -theta * r])
        top, bottom = ps.symbol_of((old, new), np.linspace(0.0, np.pi, 401))
        growth = float(np.max(np.abs(top / bottom)))
        if growth > 1.0 + 1e-12:
            continue
        bracket = abs((0.5 - theta) * r - 1.0 / 12.0)
        if best is None or bracket < best[1]:
            best = (theta, bracket, growth)
    theta, bracket, growth = best
    print(f"{r:>9.4f}{ideal:>13.4f}{theta:>13.4f}{abs(theta - ideal):>10.2e}"
          f"{bracket:>12.2e}{growth:>10.6f}{str(0.0 <= ideal <= 1.0):>11}")
```

The search finds the ideal $\theta$ to the resolution of its own grid at every $r \ge 1/6$, and the
stability constraint is **never** the thing that stops it: the largest $\lvert g\rvert$ at the
optimum is $1.000000$ every time, meaning the optimum sits comfortably inside the stable set with
the maximum attained only in the limit $\phi \to 0$ where $g \to 1$ for any scheme.

The one case where the search is genuinely constrained is $r = 0.1$. There the ideal $\theta$ is
$-0.333$, outside the family, and the best reachable member is $\theta = 0$ with a residual bracket
of $0.0333$. **Below $r = 1/6$ the fourth order member does not exist**, and $r = 1/6$ itself is
lesson 78's lucky ratio arriving as the boundary of a feasible set.

### 3.5 Consistent, stable, convergent and unusable

Take the same Crank-Nicolson scheme and a solution with a high spatial frequency. Nothing about the
scheme changes: it is consistent, unconditionally stable and second order convergent. The
**constant** is $\left\lvert\left(\frac12-\theta\right)r - \frac{1}{12}\right\rvert\max\lvert
u_{xxxx}\rvert$, and for $\sin(m\pi x)$ that is proportional to $m^4$.

```python
bracket = abs(0.5 * 0.4 - 1.0 / 12.0)
print(f"{'mode':>6}{'measured constant':>20}{'predicted':>14}{'ratio':>9}"
      f"{'h for 1e-3':>13}{'points':>9}")
for mode, sizes in ((1, (21, 41, 81)), (2, (41, 81, 161)),
                    (4, (81, 161, 321)), (8, (161, 321, 641))):
    problem = pb.sine_problem(mode=mode)
    hs, residuals = [], []
    for n in sizes:
        h = 1.0 / (n - 1)
        k = 0.4 * h ** 2
        steps = max(int(round(0.002 / k)), 1)
        out = ps.truncation_residual(problem, n, steps, steps * k, theta=0.0)
        hs.append(h); residuals.append(out["residual"])
    constant = residuals[-1] / hs[-1] ** 2
    predicted = bracket * (mode * np.pi) ** 4
    needed = math.sqrt(1e-3 / constant)
    print(f"{mode:>6}{constant:>20.3f}{predicted:>14.3f}{constant / predicted:>9.4f}"
          f"{needed:>13.2e}{1.0 / needed:>9.0f}")
```

The constant matches $\left\lvert\text{bracket}\right\rvert(m\pi)^4$ to **0.03 per cent** at every
mode, and the grid needed for a fixed truncation error grows exactly like $m^2$: 107, 426, 1705 and
6822 points for modes 1, 2, 4 and 8, factors of four each time.

**How to detect it in practice.** Measure the truncation residual, not the error. It needs no exact
solution, only the ability to apply the scheme to a guess at the answer, and it is cheap: one
application of the stencil on a coarse grid. A residual of $46534h^2$ tells you the grid you need
before you have spent anything on the grid you do not.

The alternative that does not work is refining until the answer stops changing. On mode 8 at 200
points the scheme is stable and convergent and the answer is wrong by 12, and it changes by a
factor of 4 at each refinement all the way down, which looks exactly like healthy convergence
because it **is** healthy convergence, from a long way away.

### 4.1 The delay against the Peclet number

```python
print(f"{'Pe':>6}{'r':>8}{'rho':>10}{'||G||':>10}{'||G||/rho':>12}{'delay':>8}")
for peclet, r in ((0.25, 0.45), (0.5, 0.45), (1.0, 0.45), (1.5, 0.45), (1.9, 0.45),
                  (2.0, 0.45), (3.0, 0.21), (4.0, 0.118), (6.0, 0.052)):
    g = ps.convection_diffusion_matrix(peclet, r, 60)
    rho = float(np.max(np.abs(np.linalg.eigvals(g))))
    norm = float(np.linalg.norm(g, 2))
    norms, current = [], np.eye(g.shape[0])
    for _ in range(4000):
        current = current @ g
        norms.append(float(np.linalg.norm(current, 2)))
    norms = np.asarray(norms)
    steps = np.arange(1, norms.size + 1)
    close = np.flatnonzero(norms ** (1.0 / steps) < 1.1 * rho)
    delay = int(close[0]) + 1 if close.size else -1
    print(f"{peclet:>6.2f}{r:>8.3f}{rho:>10.5f}{norm:>10.5f}{norm / rho:>12.3f}{delay:>8}")
```

**There is no exponent to fit, and reporting one would be wrong.** The delay is $1, 1, 210, 353,
420, 455, 358, 241, 1$ as the Peclet number runs from $0.25$ to $6$. It rises, peaks at
$\mathrm{Pe} = 2$, and falls again.

Two separate things cause the shape.

**Along the first half** the mesh ratio is fixed at $0.45$ and raising $\mathrm{Pe}$ drives $\rho$
down, from $0.992$ to $0.100$, while $\lVert G\rVert$ stays pinned just below 1. The gap between
the two is what the powers have to cross, and it widens from $1.007$ to $10.0$.

**Along the second half** the stability condition $r\mathrm{Pe}^2 \le 2$ forces $r$ down, which
brings $\rho$ back up towards 1 and closes the gap again.

The delays of exactly 1 are an artefact of the criterion and should be read as such: the test asks
when the observed rate falls below $1.1\rho$, so any matrix whose $\lVert G\rVert$ is already
within 10 per cent of $\rho$ passes at $n = 1$ by construction. Excluding those, the delay grows
with the gap $\lVert G\rVert/\rho$ but only weakly, at an exponent near $0.35$, and with too much
scatter to call it a law.

**The honest summary is a shape, not a formula**: the delay is largest where the asymptotic rate is
furthest from the first step rate, which for this scheme is at the corner of the stability region
where both constraints are tight.

### 4.2 The stability limit against the boundary condition

Exercise 3.1 has the measurement. The conclusion is worth stating separately because it is easy to
expect the opposite.

**The limit is $1/2$ for all three boundary conditions**, and the differences are $O(h^2)$ effects
of which phases the grid happens to carry:

- **periodic**: exactly $0.500000$ at every size, because $\phi = \pi$ is on the grid;
- **Dirichlet**: $0.5103$ at $m = 10$ down to $0.5002$ at $m = 80$, from above;
- **Neumann**: $0.5031$ down to $0.5000$, from above and faster.

Von Neumann analysis ignores the boundary entirely and gets the right answer for all three, which
is not luck: the boundary changes only which discrete phases exist, and the continuous sweep over
$\phi \in [0, \pi]$ contains all of them. **That is why von Neumann is conservative rather than
wrong**: it takes a supremum over a larger set than the grid actually has.

### 4.3 The constant for a solution that is not one mode

The truncation formula is $T = \alpha h^2\left[\left(\frac12-\theta\right)r -
\frac1{12}\right]u_{xxxx}$. For a single mode $\max\lvert u_{xxxx}\rvert = (m\pi)^4$, which is
where the $(m\pi)^4$ in the lesson came from. For anything else, **$(m\pi)^4$ is replaced by
$\max\lvert u_{xxxx}\rvert$ over the grid**, and nothing else changes.

```python
def two_mode_problem(alpha=1.0):
    def exact(x, t):
        s = np.asarray(x, dtype=float)
        return (math.exp(-alpha * np.pi ** 2 * float(t)) * np.sin(np.pi * s)
                + 0.5 * math.exp(-alpha * (4.0 * np.pi) ** 2 * float(t))
                * np.sin(4.0 * np.pi * s))
    return {"alpha": alpha, "a": 0.0, "b": 1.0, "exact": exact,
            "initial": lambda x: exact(x, 0.0),
            "left": lambda t: 0.0, "right": lambda t: 0.0, "name": "two modes"}

bracket = abs(0.5 * 0.4 - 1.0 / 12.0)
def constant_of(problem, refinements=(21, 41, 81, 161)):
    hs, residuals = [], []
    for n in refinements:
        h = 1.0 / (n - 1)
        k = 0.4 * h ** 2
        steps = max(int(round(0.05 / k)), 1)
        out = ps.truncation_residual(problem, n, steps, steps * k, theta=0.0)
        hs.append(h); residuals.append(out["residual"])
    hs, residuals = np.asarray(hs), np.asarray(residuals)
    return (float(residuals[-1] / hs[-1] ** 2),
            float(np.polyfit(np.log(hs), np.log(residuals), 1)[0]))

x = np.linspace(0.0, 1.0, 2001)
fourth = (np.pi ** 4 * np.sin(np.pi * x)
          + 0.5 * (4.0 * np.pi) ** 4 * np.sin(4.0 * np.pi * x))
constant, order = constant_of(two_mode_problem())
print(f"two modes: measured constant {constant:.3f}, order {order:.3f}")
print(f"max |u_xxxx| = {float(np.max(np.abs(fourth))):.3f}, "
      f"times the bracket = {bracket * float(np.max(np.abs(fourth))):.3f}")
print(f"ratio {constant / (bracket * float(np.max(np.abs(fourth)))):.4f}")
```

Measured $1463.3$ against a predicted $1465.1$: **0.12 per cent**, on a solution that is a sum of
two eigenfunctions with different decay rates, so the shape of $u_{xxxx}$ changes with time and the
maximum is not attained at the same place throughout.

The replacement is worth stating carefully. It is $\max\lvert u_{xxxx}\rvert$ over the region the
error is measured on, **including over time**, and for a decaying solution that maximum is
essentially its value at $t = 0$. For a problem with a source that keeps injecting high frequency
content, the maximum can occur later and the constant has to be taken there.

### 5.1 Where a frozen coefficient test fails

The frozen coefficient heuristic asks: at each point, freeze $\alpha$ and apply von Neumann. If the
worst point is stable, call the scheme stable. Exercise 3.2 found it conservative and close for a
positive $\alpha$.

**It fails completely when $\alpha$ changes sign.**

```python
print(f"{'alpha range':>16}{'min alpha':>12}{'rho':>12}{'frozen test says':>19}")
for bottom in (0.0, -0.5, -2.0):
    alpha_of = lambda x, s=bottom: 1.0 + (s - 1.0) * x
    m, r = 40, 0.1
    h = 1.0 / (m + 1)
    faces = alpha_of(np.linspace(0.0, 1.0, m + 2)[:-1] + 0.5 * h)
    t = np.zeros((m, m))
    for i in range(m):
        t[i, i] = -(faces[i] + faces[i + 1])
        if i > 0:
            t[i, i - 1] = faces[i]
        if i + 1 < m:
            t[i, i + 1] = faces[i + 1]
    g = np.eye(m) + r * t
    rho = float(np.max(np.abs(np.linalg.eigvals(g))))
    frozen = all(r * a <= 0.5 for a in faces if a > 0)
    print(f"{'1 to ' + str(bottom):>16}{float(np.min(faces)):>12.4f}{rho:>12.4f}"
          f"{('stable' if frozen else 'unstable'):>19}")
```

With $\alpha$ running from 1 down to $-2$, the frozen test applied where $\alpha > 0$ reports
stable and the true spectral radius is $1.674$.

**The mechanism is not a subtlety of the analysis, it is that the problem changed type.** Where
$\alpha < 0$ the equation is a backward heat equation, which lesson 77 showed is ill posed: the
continuous problem amplifies every mode without bound. No scheme can be stable there, and Lax's
theorem does not apply because its well posedness hypothesis has failed.

So the frozen test does not fail because freezing is a bad idea. It fails because the person
applying it looked only at the points where $\alpha$ was positive, and the classification of
lesson 77 is what would have caught it first: compute the discriminant before computing anything
else.

### 5.2 The Kreiss matrix theorem

**Statement.** For a family of matrices, the following are equivalent:

1. the powers are uniformly bounded, $\lVert G^n\rVert \le C$ for all $n$;
2. the **resolvent condition**: $\lVert(zI - G)^{-1}\rVert \le \frac{K}{\lvert z\rvert - 1}$ for
   every $\lvert z\rvert > 1$;
3. there is a norm in which $\lVert G\rVert \le 1$.

The constants are related by $K \le C$ and $C \le enK$ in dimension $n$, so the equivalence is
uniform in the family only up to that factor of $n$.

**Why the spectral radius is not enough** is condition 2 read carefully. The spectral radius
controls where the resolvent is **singular**; the resolvent condition controls how **large** it is
everywhere outside the disc. A matrix can have every eigenvalue well inside the unit circle and a
resolvent that is enormous just outside it, and that gap is exactly non-normality:
$\lVert(zI-G)^{-1}\rVert = 1/\sigma_{\min}(zI - G)$, which equals $1/\mathrm{dist}(z,\Lambda(G))$
only when $G$ is normal.

The $\varepsilon$-pseudospectrum of exercise 3.3 is the sublevel set of $\sigma_{\min}$, so
measuring it is measuring the resolvent, and the Kreiss constant is exactly how far outside the
unit disc the pseudospectra reach relative to their own $\varepsilon$.

For this lesson's matrices the pseudospectra stay **inside** the disc at every $\varepsilon$
measured, so $K = 1$, and the theorem's bound $\sup_n\lVert G^n\rVert \le enK$ is $60e$, which is
true and useless, while its lower bound $K \le \sup_n\lVert G^n\rVert$ gives $1$, which is sharp:
the measured supremum is $0.99890$ and the powers never exceed 1.

### 5.3 The easy direction of Lax's theorem

**Claim.** For a well posed linear problem and a consistent scheme, if the scheme is unstable then
it does not converge.

**Proof.** Let $S_h$ be the one step operator on the grid of spacing $h$, and let $u_h^n$ be the
computed solution and $U_h^n$ the exact solution restricted to the grid. Consistency says

$$
U_h^{n+1} = S_hU_h^n + k\tau_h^n, \qquad \lVert\tau_h^n\rVert \to 0 .
$$

Subtracting the scheme $u_h^{n+1} = S_hu_h^n$ and writing $e^n = U_h^n - u_h^n$,

$$
e^{n+1} = S_he^n + k\tau_h^n
\quad\Longrightarrow\quad
e^N = S_h^Ne^0 + k\sum_{n=0}^{N-1}S_h^{N-1-n}\tau_h^n .
$$

Now suppose the scheme converges for **every** initial condition, that is $e^N \to 0$ whenever
$e^0 \to 0$. Take exact initial data, $e^0 = 0$, and the first term vanishes. Take instead a
perturbed start $e^0 = \delta v$ with $\lVert v\rVert = 1$ and the same data otherwise: the
difference of the two runs is exactly $S_h^N\delta v$, and convergence of both forces
$\lVert S_h^N v\rVert$ to stay bounded as the grid is refined with $Nk$ fixed. That is precisely
stability. So convergence for all data implies stability, and instability implies non convergence.

**Where well posedness is used.** In the step "consistency says $\lVert\tau_h\rVert \to 0$". The
truncation error is a Taylor expansion of the **exact solution**, and the expansion needs that
solution to exist and to have the derivatives the expansion uses. For an ill posed problem there
may be no solution to expand, and where there is one it may not depend continuously on the data, in
which case the comparison between the perturbed and unperturbed runs above compares two things that
were never close.

The lesson's third case is the constructive counterpart: Du Fort and Frankel is stable and
inconsistent, and it converges to something. The theorem says nothing about what, and only the
exact solution reveals that it is the wrong thing.

---

## Lesson 80, Multidimensional Parabolic Problems and ADI

### 1.1 Why a two dimensional implicit step is not $O(N)$ by direct solution

Order the unknowns row by row, $\text{row} = i\,m + j$ for $m$ interior points a side. The stencil
couples $(i,j)$ to $(i\pm1, j)$, which are $m$ apart in that ordering, so the matrix has
**bandwidth $m$**.

A banded factorization does $O(b^2)$ work per row for bandwidth $b$, so the whole factorization is
$O(N b^2) = O(m^2 \cdot m^2) = O(m^4)$, where $N = m^2$. In one dimension $b = 1$ and the same
count gives $O(n)$, which is why the Thomas algorithm exists and why nothing like it exists here.

Reordering does not fix it. Nested dissection reaches $O(N^{3/2}) = O(m^3)$ in two dimensions,
which is better and still not $O(N)$, and it is a substantially harder algorithm.

```python
import numpy as np
from nalib import adi

print(f"{'points':>8}{'unknowns':>10}{'bandwidth':>11}{'N b^2':>13}{'ADI':>11}{'ratio':>9}")
out = adi.the_cost_of_a_full_solve()
for row in out["rows"]:
    print(f"{row['points']:>8}{row['unknowns']:>10}{row['bandwidth']:>11}"
          f"{row['banded_operations']:>13.3e}{row['adi_operations']:>11.3e}"
          f"{row['ratio']:>9.1f}")
print(f"\nexponents in the unknowns per side: banded {out['banded_exponent']:.3f}, "
      f"ADI {out['adi_exponent']:.3f}")
```

### 1.2 The explicit limit in $d$ dimensions

The worst mode is $\phi_i = \pi$ in every direction, where each direction contributes $-4r_i$ to
the growth factor:

$$
g = 1 - 4\sum_{i=1}^{d}r_i\sin^2\frac{\phi_i}{2}
\quad\longrightarrow\quad
g(\pi,\dots,\pi) = 1 - 4\sum_i r_i .
$$

Requiring $g \ge -1$ gives $\sum_i r_i \le \tfrac12$, and on an isotropic grid with $r_i = r$ that
is

$$
r \le \frac{1}{2d}.
$$

**The $2d$ is the $4$ from the second difference times the $d$ directions, divided by the $2$ that
$g \ge -1$ allows.** One half, one quarter, one sixth.

```python
out = adi.the_explicit_limit_tightens()
print(f"limits: {out['in_d_dimensions']}")
print(f"{'r':>8}{'rx+ry':>9}{'growth at (pi, pi)':>22}{'stable':>9}")
for row in out["smooth_data"]:
    print(f"{row['r']:>8.3f}{row['rx_plus_ry']:>9.3f}"
          f"{row['worst_mode_growth']:>22.4f}"
          f"{str(row['worst_mode_growth'] <= 1.0 + 1e-12):>9}")
```

### 1.3 Why ADI is unconditionally stable, in one sentence

Its growth factor is a **product of two Mobius maps**, each of the form
$(1 - 2\beta)/(1 + 2\gamma)$ with $\beta, \gamma \ge 0$, and each such factor has modulus at most 1
for every mesh ratio, so their product does too.

### 1.4 What $u^{*}$ is

$u^{*}$ **is** the exact intermediate quantity of the two half step algorithm: the unique array for
which the first half step's equation holds, defined by that equation and nothing else.

$u^{*}$ **is not** an approximation to $u(t + k/2)$, and it is not an approximation to the solution
at any time. Eliminating it between the two half steps shows what it must be on the boundary:

$$
u^{*} = \tfrac12\left[\left(I - \tfrac{r_y}{2}\delta_y^2\right)u^{n+1}
+ \left(I + \tfrac{r_y}{2}\delta_y^2\right)u^{n}\right],
$$

which is an average of the two time levels with a second difference correction. It happens to
agree with $u(t + k/2)$ to $O(k^2h^2)$ for anything that solves the equation, which is exercise
2.4 and is why the confusion survives.

### 1.5 Why the point count gave 4.57

Because $4$ is the exponent in the **unknown count per side**, $m$, and the fit was done against
the **point count**, $n = m + 2$.

Over the range measured, $m$ runs from 7 to 31 while $n$ runs from 9 to 33. The two differ by a
constant, so $\log n$ and $\log m$ are not proportional: $\log(33/9) = 1.299$ against
$\log(31/7) = 1.488$. The measured cost ratio is $(31/7)^4$, and dividing its logarithm by
$\log(33/9)$ instead of $\log(31/7)$ inflates the exponent by exactly $1.488/1.299 = 1.145$, and
$4 \times 1.145 = 4.58$.

**Fitting an exponent means choosing the variable it is an exponent in.** The offset becomes
negligible as the grids grow, so the two fits agree asymptotically and disagree at every size
anyone actually runs.

```python
cost = adi.the_cost_of_a_full_solve()
n_values = np.asarray([row["points"] for row in cost["rows"]], dtype=float)
inner = np.asarray([row["bandwidth"] for row in cost["rows"]], dtype=float)
print(f"log range in the point count {np.log(n_values[-1] / n_values[0]):.4f}, "
      f"in the unknown count {np.log(inner[-1] / inner[0]):.4f}")
stretch = np.log(inner[-1] / inner[0]) / np.log(n_values[-1] / n_values[0])
print(f"their ratio {stretch:.4f}, times 4 = {4.0 * stretch:.3f}")
print(f"measured against the point count: "
      f"{cost['banded_exponent_against_the_point_count']:.3f}")
```

### 2.1 The two dimensional growth factor

Substitute $u_{j\ell}^n = g^ne^{i(j\phi_x + \ell\phi_y)}$ into

$$
u^{n+1} = u^n + r_x\delta_x^2u^n + r_y\delta_y^2u^n .
$$

Each second difference acts on the exponential as a multiplier, as in one dimension:
$\delta_x^2 \to -4\sin^2(\phi_x/2)$ and likewise for $y$. So

$$
g = 1 - 4r_x\sin^2\frac{\phi_x}{2} - 4r_y\sin^2\frac{\phi_y}{2},
$$

which is real and decreasing in both $\sin^2$ terms. The extremes are $g = 1$ at
$\phi_x = \phi_y = 0$ and $g = 1 - 4(r_x + r_y)$ at $\phi_x = \phi_y = \pi$, so
$\lvert g\rvert \le 1$ reduces to $r_x + r_y \le \tfrac12$.

The separability is the whole content: the two directions contribute **additively** to $g$, so
their restrictions add rather than each being imposed alone. That is why the limit halves rather
than staying at $1/2$ when a second dimension is added.

```python
import math

print(f"{'rx':>8}{'ry':>8}{'rx + ry':>10}{'g at (pi, pi)':>16}{'stable':>9}")
for rx, ry in ((0.25, 0.25), (0.4, 0.09), (0.45, 0.05), (0.3, 0.3), (0.1, 0.45)):
    g = float(adi.growth_factor_2d(rx, ry, math.pi, math.pi, "explicit"))
    print(f"{rx:>8.2f}{ry:>8.2f}{rx + ry:>10.2f}{g:>16.4f}"
          f"{str(abs(g) <= 1.0 + 1e-12):>9}")
```

### 2.2 The ADI growth factor

The first half step is

$$
\left(I - \tfrac{r_x}{2}\delta_x^2\right)u^{*} = \left(I + \tfrac{r_y}{2}\delta_y^2\right)u^n,
$$

so with $s_x = \sin^2(\phi_x/2)$ and $s_y = \sin^2(\phi_y/2)$,

$$
\left(1 + 2r_xs_x\right)\hat u^{*} = \left(1 - 2r_ys_y\right)\hat u^n .
$$

The second half step gives the mirror image, and multiplying,

$$
g = \frac{1 - 2r_ys_y}{1 + 2r_xs_x}\cdot\frac{1 - 2r_xs_x}{1 + 2r_ys_y}.
$$

Each factor has the form $(1 - 2\beta)/(1 + 2\gamma)$ with $\beta, \gamma \ge 0$. Its numerator
lies in $[-\infty, 1]$ and its denominator is at least 1, and

$$
\left\lvert\frac{1 - 2\beta}{1 + 2\gamma}\right\rvert \le 1
\iff
\lvert 1 - 2\beta\rvert \le 1 + 2\gamma
\iff
-1 - 2\gamma \le 1 - 2\beta \le 1 + 2\gamma,
$$

and the right inequality holds because $\beta \ge 0$, the left because $\beta \le 1 + \gamma + \beta$
always. Both hold for **every** nonnegative $\beta, \gamma$, so each factor is bounded by 1 and so
is the product.

**Nothing in that argument mentions the size of $r_x$ or $r_y$**, which is what unconditional
means.

```python
print(f"{'rx':>12}{'ry':>12}{'largest |g|':>14}")
for rx, ry in ((0.4, 0.4), (2.0, 0.5), (10.0, 0.1), (1e4, 1.0), (1.0, 1e4), (1e6, 1e6)):
    phases = np.linspace(0.0, np.pi, 61)
    px, py = np.meshgrid(phases, phases, indexing="ij")
    print(f"{rx:>12.1f}{ry:>12.1f}"
          f"{float(np.max(np.abs(adi.growth_factor_2d(rx, ry, px, py, 'adi')))):>14.6f}")
```

### 2.3 Eliminating $u^{*}$

Write $A = \tfrac{r_x}{2}\delta_x^2$ and $B = \tfrac{r_y}{2}\delta_y^2$. The two half steps are

$$
(I - A)u^{*} = (I + B)u^n, \qquad (I - B)u^{n+1} = (I + A)u^{*}.
$$

Apply $(I + A)$ to the first and $(I - A)$ to the second, and use that $A$ and $B$ **commute**,
which they do on a tensor product grid because each acts on a different index:

$$
(I - B)(I - A)u^{n+1} = (I + A)(I + B)u^n .
$$

So the full step operator is

$$
u^{n+1} = (I - A)^{-1}(I - B)^{-1}(I + A)(I + B)u^n .
$$

Compare with Crank-Nicolson, which is $(I - A - B)^{-1}(I + A + B)$. The two differ because
$(I-A)(I-B) = I - A - B + AB$: the products carry an extra $AB$ on each side, and

$$
(I - A - B + AB)^{-1}(I + A + B + AB)
= (I - A - B)^{-1}(I + A + B) + O(AB\cdot(A+B)) .
$$

Now $A$ and $B$ are each $O(k)$ as operators applied to a smooth function, so the discrepancy is
$O(k^3)$ per step and $O(k^2)$ over a fixed time: **second order**, and the same order as
Crank-Nicolson itself.

Exercise 4.1 measures that discrepancy directly and fits $2.002$.

### 2.4 The boundary value of $u^{*}$

From the second half step, $u^{*} = (I + A)^{-1}(I - B)u^{n+1}$; from the first,
$u^{*} = (I - A)^{-1}(I + B)u^n$. Averaging the two forms after multiplying through, and using
that on the boundary $x = a$ the operator $A$ acts along the boundary line so its inverse is not
available, the usable identity is the one obtained by adding the two half steps:

$$
u^{*} - Au^{*} + u^{*} + Au^{*} = (I + B)u^n + (I - B)u^{n+1}
\quad\Longrightarrow\quad
u^{*} = \tfrac12\left[(I - B)u^{n+1} + (I + B)u^n\right].
$$

Expanding both candidates for a solution, with $A_0 = u(t_n)$ and $B_0 = u(t_{n+1})$ at the
boundary point and $\sigma$ the discrete second difference in $y$:

$$
u(t + \tfrac{k}{2}) = \tfrac{A_0 + B_0}{2} - \frac{A_0k^2}{8} + O(k^3),
$$

$$
\tfrac{A_0 + B_0}{2} + \frac{r\sigma}{4}(A_0 - B_0)
= \tfrac{A_0 + B_0}{2} + \frac{A_0k^2}{8} + O(k^3),
$$

where the second $k^2/8$ comes from $r\sigma/4 \approx -\alpha\pi^2k/4$ and $A_0 - B_0 \approx A_0k$.
**The two $O(k^2)$ terms cancel**, and what survives is the difference between $\sigma$ and its
continuum limit $-\pi^2h^2$, which is $O(h^2)$. So the gap is $O(k^2h^2)$.

```python
out = adi.the_half_step_boundary_costs_less_than_advertised()
print(f"exponent in k: {out['exponent_in_k_over_the_small_steps']:.3f} over the small steps, "
      f"{out['the_whole_k_sweep_would_say']:.3f} over the whole sweep")
print(f"exponent in h: {out['exponent_in_h']:.3f}")
print(f"the gap is O(k^2 h^2): {out['the_gap_is_k_squared_h_squared']}")
print(f"both treatments second order: {out['both_treatments_are_second_order']}")
```

### 2.5 The two operation counts

**ADI.** One half step solves $m$ tridiagonal systems, one per line, each of length $m$. The Thomas
algorithm costs about 8 operations per unknown, so a half step is $8m^2$ and a full step
$16m^2 = O(m^2) = O(N)$.

**Banded Crank-Nicolson.** One matrix, $N = m^2$ unknowns, bandwidth $m$. A banded LU costs about
$Nb^2 = m^2\cdot m^2 = m^4$ for the factorization, and $O(Nb) = O(m^3)$ per solve afterwards. If the
matrix is factored once and reused, the per step cost is $O(m^3)$ and the setup is $O(m^4)$; if $r$
changes, every step pays $O(m^4)$.

So the exponents in $m$ are **2 for ADI and 4 for the banded factorization**, which is what the
lesson measures to three decimal places. The gap of two orders per step is the whole reason ADI
exists.

```python
print(f"{'points':>8}{'unknowns per side':>19}{'16 m^2':>12}{'m^4':>12}{'measured ratio':>17}")
for row in adi.the_cost_of_a_full_solve()["rows"]:
    m = row["bandwidth"]
    print(f"{row['points']:>8}{m:>19}{16 * m ** 2:>12}{m ** 4:>12}{row['ratio']:>17.1f}")
```

### 3.1 Douglas-Rachford against Peaceman-Rachford

Douglas-Rachford splits differently:

$$
(I - 2A)u^{*} = (I + 2B)u^n, \qquad (I - 2B)u^{n+1} = u^{*} - 2Bu^n .
$$

It is the standard alternative, and it is **first order in time** where Peaceman-Rachford is second.

```python
from nalib.banded import thomas

def douglas_rachford_step(u, r, boundary_next):
    v = np.asarray(u, dtype=float)
    yy = adi.second_difference(v, 1, 1.0)
    rhs = v[1:-1, 1:-1] + r * yy[1:-1, :]
    star = np.array(boundary_next, dtype=float, copy=True)
    star[1:-1, 1:-1] = adi.tridiagonal_lines(rhs, r, star[0, 1:-1], star[-1, 1:-1])
    rhs2 = star[1:-1, 1:-1] - r * yy[1:-1, :]
    out = np.array(boundary_next, dtype=float, copy=True)
    out[1:-1, 1:-1] = adi.tridiagonal_lines(rhs2.T, r, out[1:-1, 0], out[1:-1, -1]).T
    return out

def run_scheme(problem, points, steps, t_end, scheme):
    mesh = adi.grid_of(problem, points)
    h = mesh["h"]
    k = t_end / steps
    r = problem["alpha"] * k / h ** 2
    u = np.asarray(problem["exact"](mesh["x"], mesh["y"], 0.0), dtype=float)
    for step in range(steps):
        after = adi.set_boundary(np.zeros_like(u), problem, mesh, (step + 1) * k)
        if scheme == "peaceman":
            old_yy = np.zeros_like(u)
            new_yy = np.zeros_like(u)
            old_yy[:, 1:-1] = adi.second_difference(u, 1, 1.0)
            new_yy[:, 1:-1] = adi.second_difference(after, 1, 1.0)
            star = 0.5 * ((after - 0.5 * r * new_yy) + (u + 0.5 * r * old_yy))
            u, _ = adi.peaceman_rachford_step(u, r, r, star, after)
        else:
            u = douglas_rachford_step(u, r, after)
    return u

problem = adi.separable_problem()
print(f"{'scheme':>10}{'order in k':>13}   errors against a fine step reference")
for scheme in ("peaceman", "douglas"):
    reference = run_scheme(problem, 65, 2048, 0.02, scheme)
    ks, errs = [], []
    for steps in (4, 8, 16, 32, 64):
        u = run_scheme(problem, 65, steps, 0.02, scheme)
        ks.append(0.02 / steps)
        errs.append(float(np.max(np.abs(u - reference))))
    print(f"{scheme:>10}{float(np.polyfit(np.log(ks), np.log(errs), 1)[0]):>13.4f}   "
          f"{np.array2string(np.asarray(errs), precision=3)}")
```

Peaceman-Rachford fits **2.0004** and Douglas-Rachford **1.0080**.

The measurement had to be set up carefully. Comparing against the **exact solution** instead gives
non-monotone errors for Peaceman-Rachford, because the time error crosses the space error floor and
changes sign, and the fitted order comes out meaningless. Comparing against a fine step reference
**on the same grid** removes the space error from both sides and leaves only what is being asked
about.

What Douglas-Rachford buys for its lost order is that it extends to three dimensions cleanly, which
is exercise 3.4.

### 3.2 ADI on a rectangle

The stability argument of exercise 2.2 never used $r_x = r_y$: each factor is bounded by 1
separately. So a rectangle changes nothing, and the measurement confirms it at ratios differing by
six orders of magnitude.

```python
print(f"{'rx':>12}{'ry':>12}{'rx/ry':>12}{'largest |g|':>14}")
phases = np.linspace(0.0, np.pi, 81)
px, py = np.meshgrid(phases, phases, indexing="ij")
for rx, ry in ((0.4, 0.4), (2.0, 0.5), (10.0, 0.1), (1e4, 1.0), (1.0, 1e4), (1e6, 1e6)):
    largest = float(np.max(np.abs(adi.growth_factor_2d(rx, ry, px, py, "adi"))))
    print(f"{rx:>12.1f}{ry:>12.1f}{rx / ry:>12.1e}{largest:>14.6f}")
```

$1.000000$ every time. Compare with lesson 77's compact nine point Laplacian, which fails
completely on a rectangle: **ADI's derivation treats the directions separately and inherits nothing
from their ratio, while the compact stencil's derivation mixed them and cannot be rescaled.**

### 3.3 A Neumann side

Put the ghost point row of lesson 77 into the **second** half step, which is the one implicit in
$y$, so the Neumann row joins the tridiagonal system rather than being applied afterwards. The test
problem is $u = e^{-t}\cos(\pi x)\cos(\pi y)$, whose $u_y$ vanishes on $y = 0$.

```python
def neumann_adi(problem, points, steps, t_end, ghost=True):
    mesh = adi.grid_of(problem, points)
    h = mesh["h"]
    k = t_end / steps
    r = problem["alpha"] * k / h ** 2
    u = np.asarray(problem["exact"](mesh["x"], mesh["y"], 0.0), dtype=float)
    for step in range(steps):
        after = adi.set_boundary(np.zeros_like(u), problem, mesh, (step + 1) * k)
        old_yy = np.zeros_like(u)
        new_yy = np.zeros_like(u)
        old_yy[:, 1:-1] = adi.second_difference(u, 1, 1.0)
        new_yy[:, 1:-1] = adi.second_difference(after, 1, 1.0)
        star = 0.5 * ((after - 0.5 * r * new_yy) + (u + 0.5 * r * old_yy))
        yy = adi.second_difference(u, 1, 1.0)[1:-1, :]
        half = np.array(star, dtype=float, copy=True)
        half[1:-1, 1:-1] = adi.tridiagonal_lines(u[1:-1, 1:-1] + 0.5 * r * yy, 0.5 * r,
                                                 half[0, 1:-1], half[-1, 1:-1])
        xx_all = adi.second_difference(half, 0, 1.0)
        rhs2 = half[1:-1, 1:-1] + 0.5 * r * xx_all[:, 1:-1]
        m = rhs2.shape[1] + 1
        out = np.array(after, dtype=float, copy=True)
        for line in range(rhs2.shape[0]):
            column = np.empty(m)
            column[0] = half[line + 1, 0] + 0.5 * r * xx_all[line, 0]
            column[1:] = rhs2[line]
            sub = np.full(m, -0.5 * r)
            diag = np.full(m, 1.0 + r)
            sup = np.full(m, -0.5 * r)
            if ghost:
                sup[0] = -r
            else:
                diag[0], sup[0], column[0] = 1.0, -1.0, 0.0
            column[-1] += 0.5 * r * out[line + 1, -1]
            out[line + 1, :-1] = thomas(sub, diag, sup, column)
        u = out
    want = np.asarray(problem["exact"](mesh["x"], mesh["y"], steps * k), dtype=float)
    return h, float(np.max(np.abs(u - want)))

print(f"{'treatment':>13}{'order':>9}{'ratios':>28}")
for ghost in (True, False):
    hs, errs = [], []
    for n in (9, 17, 33, 65):
        prob = adi.moving_boundary_problem()
        h = 1.0 / (n - 1)
        steps = adi.steps_for(prob, h, 1.0, 0.05)
        h, e = neumann_adi(prob, n, steps, 0.05, ghost=ghost)
        hs.append(h); errs.append(e)
    hs, errs = np.asarray(hs), np.asarray(errs)
    label = "ghost point" if ghost else "one sided"
    print(f"{label:>13}{float(np.polyfit(np.log(hs), np.log(errs), 1)[0]):>9.4f}"
          f"{np.array2string(errs[:-1] / errs[1:], precision=3):>28}")
```

The one sided treatment gives ratios of $3.23, 2.44, 2.25$, converging on 2: **first order**, as
in one dimension and for the same reason.

The ghost point gives $1.95, 5.37, 6.18$ and a fitted order of $2.05$, and the sequence is
erratic rather than settling on 4. Two errors of similar size are competing here, the boundary
treatment and the ADI splitting, and they have opposite signs on this problem, so the total passes
through a partial cancellation at the second grid. **The honest statement is that the ghost point
is second order or better and the one sided treatment is first**, which is the question asked; the
sequence would need a sweep at fixed splitting error to be read more finely than that.

### 3.4 Three dimensions

The Douglas form generalises the splitting to three factors with a correction that keeps it second
order:

$$
(I - \tfrac{r}{2}\delta_x^2)v_1 = u^n + \tfrac{r}{2}\delta_x^2u^n + r\delta_y^2u^n + r\delta_z^2u^n,
$$
$$
(I - \tfrac{r}{2}\delta_y^2)v_2 = v_1 - \tfrac{r}{2}\delta_y^2u^n,
\qquad
(I - \tfrac{r}{2}\delta_z^2)u^{n+1} = v_2 - \tfrac{r}{2}\delta_z^2u^n .
$$

The alternative is the plain symmetric product,
$\prod_i(I - \tfrac{r}{2}\delta_i^2)u^{n+1} = \prod_i(I + \tfrac{r}{2}\delta_i^2)u^n$.

```python
def three_dimensional(points, steps, t_end, scheme, modes=(1, 2, 3), alpha=1.0):
    line = np.linspace(0.0, 1.0, points)
    h = float(line[1] - line[0])
    k = t_end / steps
    r = alpha * k / h ** 2
    gx, gy, gz = np.meshgrid(line, line, line, indexing="ij")
    p, q, s = modes
    start = (np.sin(p * np.pi * gx) * np.sin(q * np.pi * gy) * np.sin(s * np.pi * gz))
    decay = alpha * np.pi ** 2 * (p * p + q * q + s * s)
    m = points - 2
    u = start.copy()

    def d2(v, axis):
        out = np.zeros_like(v)
        lo, mid, hi = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        lo[axis] = slice(0, -2)
        mid[axis] = slice(1, -1)
        hi[axis] = slice(2, None)
        out[tuple(mid)] = v[tuple(hi)] - 2.0 * v[tuple(mid)] + v[tuple(lo)]
        return out

    def implicit(v, axis, c):
        moved = np.moveaxis(v, axis, 0)
        inner = moved[1:-1]
        flat = inner.reshape(m, -1)
        sub, diag, sup = np.full(m, -c), np.full(m, 1.0 + 2.0 * c), np.full(m, -c)
        solved = np.empty_like(flat)
        for column in range(flat.shape[1]):
            solved[:, column] = thomas(sub, diag, sup, flat[:, column])
        result = np.zeros_like(moved)
        result[1:-1] = solved.reshape(inner.shape)
        return np.moveaxis(result, 0, axis)

    for _ in range(steps):
        if scheme == "douglas":
            rhs = u + 0.5 * r * d2(u, 0) + r * d2(u, 1) + r * d2(u, 2)
            v1 = implicit(rhs, 0, 0.5 * r)
            v2 = implicit(v1 - 0.5 * r * d2(u, 1), 1, 0.5 * r)
            u = implicit(v2 - 0.5 * r * d2(u, 2), 2, 0.5 * r)
        else:
            rhs = u + 0.5 * r * d2(u, 0)
            rhs = rhs + 0.5 * r * d2(rhs, 1)
            rhs = rhs + 0.5 * r * d2(rhs, 2)
            v1 = implicit(rhs, 0, 0.5 * r)
            v2 = implicit(v1, 1, 0.5 * r)
            u = implicit(v2, 2, 0.5 * r)
    return u, math.exp(-decay * steps * k) * start

print(f"{'scheme':>9}{'order in h':>13}{'order in k':>13}")
for scheme in ("douglas", "product"):
    hs, errs = [], []
    for n in (9, 13, 17, 25):
        steps = max(int(math.ceil(0.01 / (1.0 / (n - 1)) ** 2)), 1)
        u, want = three_dimensional(n, steps, 0.01, scheme)
        hs.append(1.0 / (n - 1)); errs.append(float(np.max(np.abs(u - want))))
    reference, _ = three_dimensional(17, 512, 0.01, scheme)
    ks, time_errs = [], []
    for steps in (4, 8, 16, 32):
        u, _ = three_dimensional(17, steps, 0.01, scheme)
        ks.append(0.01 / steps); time_errs.append(float(np.max(np.abs(u - reference))))
    print(f"{scheme:>9}{float(np.polyfit(np.log(hs), np.log(errs), 1)[0]):>13.4f}"
          f"{float(np.polyfit(np.log(ks), np.log(time_errs), 1)[0]):>13.4f}")
```

**Both are second order**, in $h$ and in $k$: Douglas fits $1.79$ and $1.99$, the plain product
$1.63$ and $2.00$.

That contradicts the usual warning that a three factor splitting drops to first order, and the
reason is exercise 4.3's: on a tensor product grid with constant coefficients the three direction
operators **commute exactly**, so the product of the three factors is not an approximation to
anything, it is an exact factorization of a scheme that happens to be second order. The order loss
the literature describes needs operators that do not commute, which means variable coefficients or
a mixed derivative.

On the fully symmetric mode $(1,1,1)$ the Douglas form measures order $3.01$ in $k$, which is a
further cancellation particular to $\mu_x = \mu_y = \mu_z$ and disappears at $(1,2,3)$. It is worth
knowing only as a reminder to test on an asymmetric solution.

### 3.5 Conjugate gradients on the full system

```python
from nalib.krylov import conjugate_gradient

problem = adi.separable_problem()
print(f"{'points':>8}{'unknowns':>10}{'mesh ratio':>12}{'kappa':>9}{'CG iterations':>16}"
      f"{'ADI tridiagonal solves':>24}")
for n in (9, 17, 33, 65):
    m = n - 2
    h = 1.0 / (n - 1)
    steps = adi.steps_for(problem, h, 2.0, 0.02)
    ratio = problem["alpha"] * (0.02 / steps) / h ** 2
    system = adi.crank_nicolson_matrix(n, ratio, ratio)
    rng = np.random.default_rng(42)
    b = rng.normal(size=system["unknowns"])
    run = conjugate_gradient(system["left"], b, tol=1e-10, keep_history=False)
    print(f"{n:>8}{system['unknowns']:>10}{ratio:>12.4f}"
          f"{float(np.linalg.cond(system['left'])):>9.4f}{run.n_iter:>16}{2 * m:>24}")
```

**The two dimensional Crank-Nicolson matrix is well conditioned**, and that is the finding.

Its condition number is $4.96, 7.28, 8.28, 8.94$ as the unknown count goes from 49 to 3969: it is
$1 + 4r$ approximately, and **bounded independently of $N$**. The reason is that the identity
dominates: the matrix is $I - \frac{k}{2}L$, and $\frac{k}{2}L$ has norm $2r$, which is fixed by the
mesh ratio and not by the grid. Lesson 82's **steady** Poisson matrix has no identity to dominate
it and a condition number growing like $h^{-2}$.

So conjugate gradients needs 20, 29, 32 and 34 iterations, essentially constant. At $O(N)$ work per
iteration that is $O(N)$ per time step, which is **the same order as ADI**. ADI's own cost is $2m$
tridiagonal solves of length $m$, also $O(N)$.

The comparison is therefore a constant factor, not an order. At $n = 65$: CG does $34$ matrix
vector products of $5N$ operations each, about $6.7\times10^5$; ADI does $126$ solves of $63$
unknowns at 8 operations each, about $6.3\times10^4$. **ADI is about ten times cheaper and not
asymptotically better**, which is a much weaker claim than the one against a direct banded solve
and is the honest one to make against a Krylov method.

### 4.1 The splitting error, measured directly

Run the full two dimensional Crank-Nicolson system and ADI on the same grid with the same step, and
subtract. What is left is exactly the $AB$ terms of exercise 2.3.

```python
def full_crank_nicolson(problem, points, steps, t_end):
    mesh = adi.grid_of(problem, points)
    h = mesh["h"]
    k = t_end / steps
    ratio = problem["alpha"] * k / h ** 2
    system = adi.crank_nicolson_matrix(points, ratio, ratio)
    m = points - 2
    u = np.asarray(problem["exact"](mesh["x"], mesh["y"], 0.0), dtype=float)
    for step in range(steps):
        now = adi.set_boundary(np.array(u, copy=True), problem, mesh, step * k)
        after = adi.set_boundary(np.zeros_like(u), problem, mesh, (step + 1) * k)
        rhs = system["right"] @ u[1:-1, 1:-1].ravel()
        for i in range(m):
            for j in range(m):
                row = i * m + j
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ii, jj = i + di, j + dj
                    if not (0 <= ii < m and 0 <= jj < m):
                        rhs[row] += 0.5 * ratio * (after[ii + 1, jj + 1]
                                                   + now[ii + 1, jj + 1])
        u = after
        u[1:-1, 1:-1] = np.linalg.solve(system["left"], rhs).reshape(m, m)
    return u

print(f"{'k':>10}{'ADI error':>14}{'splitting gap':>16}{'ratio':>9}")
ks, gaps = [], []
prev = None
for steps in (2, 4, 8, 16, 32):
    full = full_crank_nicolson(problem, 33, steps, 0.02)
    run = adi.solve(problem, 33, steps, 0.02, scheme="adi")
    gap = float(np.max(np.abs(run["u"] - full)))
    shown = "" if prev is None else f"{prev / gap:.2f}"
    print(f"{0.02 / steps:>10.5f}{run['error']:>14.3e}{gap:>16.3e}{shown:>9}")
    prev = gap
    ks.append(0.02 / steps); gaps.append(gap)
print(f"\nfitted order of the splitting gap in k: "
      f"{float(np.polyfit(np.log(ks), np.log(gaps), 1)[0]):.4f}")
```

The gap falls by exactly 4 at each halving and fits **2.002**, which is exercise 2.3's derivation
confirmed to three digits.

Note the second column. The ADI error against the exact solution is **not** monotone: it is
$2.1\times10^{-6}$ at the coarsest step and $2.1\times10^{-4}$ at the finest, because the space
error is $2.1\times10^{-4}$ and at the coarsest step the splitting error happens to cancel most of
it. Measuring the splitting error directly avoids that entirely, and it is the only column here
that answers the question asked.

### 4.2 Equal accuracy across four decades

```python
print(f"{'tol':>8}{'scheme':>10}{'points':>8}{'steps':>8}{'r':>9}{'error':>12}{'updates':>12}")
costs = {}
for target in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
    for name, cap, weight in (("explicit", 0.24, 1), ("adi", 2.0, 8)):
        best = None
        for n in (9, 13, 17, 25, 33, 49, 65, 97, 129):
            h = 1.0 / (n - 1)
            steps = adi.steps_for(problem, h, cap, 0.02)
            if (n - 2) ** 2 * steps * weight > 4e8:
                continue
            run = adi.solve(problem, n, steps, 0.02,
                            scheme="explicit" if name == "explicit" else "adi")
            cost = (n - 2) ** 2 * steps * weight
            if run["error"] <= target and (best is None or cost < best[-1]):
                best = (n, steps, run["r"], run["error"], cost)
        n, steps, r, err, cost = best
        print(f"{target:>8.0e}{name:>10}{n:>8}{steps:>8}{r:>9.3f}{err:>12.3e}{cost:>12}")
        costs.setdefault(name, []).append((target, cost))
print()
for name, pairs in costs.items():
    t = np.asarray([p[0] for p in pairs], dtype=float)
    c = np.asarray([p[1] for p in pairs], dtype=float)
    print(f"{name:>10}: exponent {float(np.polyfit(np.log(t), np.log(c), 1)[0]):>7.3f}")
```

The exponents come out at $-2.233$ for the explicit scheme and $-1.678$ for ADI, against
predictions of $-2$ and $-3/2$.

The predictions come from balancing the error terms. In two dimensions the cost is
$N \times \text{steps} = h^{-2}k^{-1}$. For ADI the error is $C_1k^2 + C_2h^2$, so
$k \sim h \sim \text{tol}^{1/2}$ and the cost is $\text{tol}^{-3/2}$. For the explicit scheme
$k$ is capped at $h^2/4$, so $k \sim h^2 \sim \text{tol}$ and the cost is $\text{tol}^{-2}$.

**The crossover is visible in the table and it is not at the origin.** At $10^{-2}$ the explicit
scheme is **cheaper**, 294 updates against 392, because ADI pays 8 operations per unknown for its
tridiagonal solves and at a loose tolerance there is nothing to win back. By $10^{-4}$ the ratio is
16 the other way, and it keeps growing.

### 4.3 The sweep direction

**On this problem it makes no difference at all, to the last bit**, and the reason is worth
isolating from the PDE entirely.

```python
rng = np.random.default_rng(42)
print(f"{'case':>30}{'||AB - BA||':>15}{'||P - Q||':>13}")
for label, kind in (("Dxx and Dyy on a tensor grid", "tensor"),
                    ("two random symmetric matrices", "random")):
    m = 8
    if kind == "tensor":
        t = (np.diag(np.full(m, -2.0)) + np.diag(np.ones(m - 1), 1)
             + np.diag(np.ones(m - 1), -1))
        a, b = np.kron(t, np.eye(m)), np.kron(np.eye(m), t)
    else:
        a = rng.normal(size=(m * m, m * m)); a = 0.5 * (a + a.T)
        b = rng.normal(size=(m * m, m * m)); b = 0.5 * (b + b.T)
    r = 0.1
    eye = np.eye(a.shape[0])
    p = (np.linalg.solve(eye - 0.5 * r * b, eye + 0.5 * r * a)
         @ np.linalg.solve(eye - 0.5 * r * a, eye + 0.5 * r * b))
    q = (np.linalg.solve(eye - 0.5 * r * a, eye + 0.5 * r * b)
         @ np.linalg.solve(eye - 0.5 * r * b, eye + 0.5 * r * a))
    print(f"{label:>30}{float(np.linalg.norm(a @ b - b @ a, 2)):>15.3e}"
          f"{float(np.linalg.norm(p - q, 2)):>13.3e}")
```

The two direction operators on a tensor product grid commute **exactly**, $\lVert AB - BA\rVert =
0.000\times10^{0}$, because $A = T\otimes I$ and $B = I\otimes T$ and those commute for any $T$. So
the two orderings of the ADI step are the same operator, and the two computed answers differ by
$4.3\times10^{-17}$, which is rounding.

Replace them by two matrices that do not commute and the orderings differ by $0.678$, which is
enormous. **Alternating the sweep direction is a real technique and it is a technique for problems
this one is not**: variable coefficients that couple the directions, mixed derivatives, or
nonlinear terms. On a constant coefficient problem on a rectangle it is provably a no-op.

### 5.1 Why Peaceman-Rachford is second order and Douglas-Rachford is not

Write both as operator products, with $A$ and $B$ the two half operators.

**Peaceman-Rachford**, from exercise 2.3:

$$
P = (I - A)^{-1}(I - B)^{-1}(I + A)(I + B).
$$

Expand in powers, using $A, B = O(k)$:

$$
P = I + 2(A + B) + 2(A + B)^2 + \cdots + 2AB + \cdots
$$

and compare with $\exp(2(A+B))$, the exact solution operator for the split problem. The
**symmetry** of the construction, one factor of each on each side, makes the $O(k^2)$ terms match
and leaves a discrepancy at $O(k^3)$ per step.

**Douglas-Rachford**:

$$
D = (I - 2B)^{-1}\left[(I - 2A)^{-1}(I + 2B) - 2B\right],
$$

which expands to $I + 2(A+B) + 4AB + \cdots$ where the exponential has $2(A+B)^2 = 2A^2 + 4AB +
2B^2$. The $AB$ terms match and the $A^2$ and $B^2$ terms do not: the scheme is **not** symmetric
in $A$ and $B$, and the asymmetry survives at $O(k^2)$ per step, which is $O(k)$ over a fixed time.

The measurement in exercise 3.1 puts the two at $2.0004$ and $1.0080$.

What Douglas-Rachford is for is the case exercise 3.4 raises: its structure generalises to any
number of directions with one factor each, while Peaceman-Rachford's symmetry has no clean
generalisation past two.

### 5.2 Three dimensional splitting

The claim to be examined is that the natural three way splitting is only first order.

**On the problem this lesson uses, it is not**, and exercise 3.4 measures both forms at second
order in both variables. The three direction operators commute exactly on a tensor product grid,
so a product of three factors is an exact factorization rather than an approximation, and there is
no splitting error at all to lose an order to.

**What has to change** for the claim to bite is the commutation. Three sources of non-commuting
operators appear in practice:

- **variable coefficients** that do not factor, $\alpha(x,y,z)$ rather than
  $\alpha_1(x)\alpha_2(y)\alpha_3(z)$;
- **mixed derivatives**, from a non-rectangular domain mapped to a box or from an anisotropic
  material;
- **non-rectangular boundaries**, where the set of unknowns on a line changes with the line.

In each of those the product of the three one direction factors differs from the full operator at
$O(k^2)$ per step, and the Douglas correction terms are exactly what restores the lost order. That
is why the Douglas form is the one that generalises: **its correction is designed for the case the
constant coefficient measurement cannot see.**

Exercise 4.3 is the same statement measured on 8 by 8 matrices, and there the difference between
the orderings goes from $4\times10^{-17}$ to $0.678$ the moment the operators stop commuting.

### 5.3 Where the half step boundary warning is real

The lesson could not reproduce the standard warning, and exercise 2.4 explains why: for anything
that **solves the equation**, the two candidate boundary values agree to $O(k^2h^2)$.

The construction that breaks it prescribes boundary data that does **not** satisfy the interior
equation. The simplest is a corner in time: hold the boundary at zero, then switch it to 1 at
$t = t_1$. At the switch, $u_{tt}$ on the boundary is not $\alpha u_{xx}$ because $u_{tt}$ does not
exist, and the cancellation of the two $O(k^2)$ terms had used exactly that identity.

```python
def switching_boundary(points, steps, t_end, half_step, switch=0.5):
    """u_t = u_xx with u(0,t) = 0 for t < switch and 1 after, u(1,t) = 0."""
    line = np.linspace(0.0, 1.0, points)
    h = float(line[1] - line[0])
    k = t_end / steps
    r = k / h ** 2
    m = points - 2
    left = lambda t: 0.0 if t < switch else 1.0
    u = np.zeros(points)
    for step in range(steps):
        now, nxt = step * k, (step + 1) * k
        d2 = u[2:] - 2.0 * u[1:-1] + u[:-2]
        rhs = u[1:-1] + 0.5 * r * d2
        if half_step == "consistent":
            edge = 0.5 * (left(nxt) + left(now))
        else:
            edge = left(now + 0.5 * k)
        rhs[0] += 0.5 * r * edge
        inner = thomas(np.full(m, -0.5 * r), np.full(m, 1.0 + r),
                       np.full(m, -0.5 * r), rhs)
        u = np.concatenate([[left(nxt)], inner, [0.0]])
    return u

reference = switching_boundary(129, 8192, 1.0, "consistent")
print(f"{'steps':>8}{'consistent':>15}{'naive':>15}{'ratio':>9}")
for steps in (16, 32, 64, 128, 256):
    good = switching_boundary(129, steps, 1.0, "consistent")
    bad = switching_boundary(129, steps, 1.0, "naive")
    a = float(np.max(np.abs(good - reference)))
    b = float(np.max(np.abs(bad - reference)))
    print(f"{steps:>8}{a:>15.3e}{b:>15.3e}{b / max(a, 1e-300):>9.2f}")
```

The two treatments now differ, and by a **factor of 2 at every step count**: the ratio is
$1.99, 1.99, 1.99, 1.98, 1.98$ across four halvings of the step. Compare that with the
manufactured solution test of the lesson, where they agreed to six significant figures.

So the mechanism is real and the effect is a **constant**, not an order. Even with boundary data
that is not differentiable in time, both treatments converge at the same rate and the naive one is
twice as wrong. An order loss was not reproduced here either.

Two things follow.

**The warning is about a constant.** Doubling an error is worth avoiding and it is not what
"drops to first order" describes, and the difference matters when deciding how much effort the
correct treatment is worth.

**A manufactured solution test cannot find even the factor of 2.** Manufacturing a solution
guarantees the data satisfies the equation, which is exactly the hypothesis the cancellation of
exercise 2.4 uses. To see the effect at all, the data has to be prescribed independently and the
answer compared against a refined run rather than against a formula, which is what the code above
does.

That is the general shape of the trap this part keeps returning to: **the test that is easiest to
write is the one that guarantees the defect cannot appear.**

---

## Lesson 81, Hyperbolic Equations and the CFL Condition

### 1.1 Two initial conditions, and why the heat equation needs one

The wave equation is **second order in time**. Its general solution has two free functions, which
d'Alembert's formula shows explicitly: $u = F(x - ct) + G(x + ct)$, one arbitrary function
travelling each way. Two conditions are needed to pin them both down, and the natural pair is the
shape $u(x,0)$ and the velocity $u_t(x,0)$.

The heat equation is **first order in time**, so one condition fixes the solution. It is also
irreversible: the solution smooths as $t$ increases, so information is lost forward in time and
cannot be recovered backward, which is lesson 77's ill posedness.

The discrete statement is the same. The wave scheme is a three level recurrence and needs two
starting levels; the heat scheme is two level and needs one.

### 1.2 The Courant condition, twice

**From the roots.** Substituting one Fourier mode gives a quadratic in the amplification factor
whose roots multiply to $-1$, so both must lie on the unit circle for the powers to stay bounded.
That happens exactly when $\lambda \le 1$.

**From domains of dependence.** The true solution at $(x, t)$ depends on the initial data over
$[x - ct, x + ct]$; the scheme at $(x_j, t_n)$ depends on it over $[x_j - nh, x_j + nh]$. The
second must contain the first, which needs $nh \ge cnk$, that is $\lambda \le 1$.

### 1.3 Why the product of the roots makes the limit sharp

For a parabolic problem the roots have modulus **less** than 1 and there is room: a scheme can be
comfortably stable, with $\lvert g\rvert$ well inside the circle over most of the spectrum.

Here the product of the two roots is exactly $-1$, so $\lvert g_1\rvert\lvert g_2\rvert = 1$ for
every mode and every mesh ratio. Stability means both are **on** the circle, and the instant one
moves inside the other moves outside by the reciprocal factor. There is no interior of the stable
set to sit in: the stable configuration is a boundary case, which is what "sharp" means.

```python
import math

import numpy as np
from nalib import hyperbolic as hy

phases = np.linspace(1e-6, np.pi, 5)
print(f"{'lambda':>9}{'phi':>9}{'|g1|':>10}{'|g2|':>10}{'product':>11}")
for lam in (0.5, 1.0, 1.2):
    for phi in phases[1::2]:
        first, second = hy.growth_roots(lam, phi)
        print(f"{lam:>9.2f}{float(phi):>9.4f}{abs(first[0]):>10.5f}"
              f"{abs(second[0]):>10.5f}{abs(first[0] * second[0]):>11.6f}")
```

### 1.4 Why the scheme is exact at $\lambda = 1$

Put $\lambda = 1$ into the update and the centre terms cancel:

$$
u_j^{n+1} = 2u_j^n - u_j^{n-1} + \left(u_{j+1}^n - 2u_j^n + u_{j-1}^n\right)
= u_{j+1}^n + u_{j-1}^n - u_j^{n-1}.
$$

Now check that d'Alembert's solution satisfies it exactly. With $u = F(x - ct) + G(x + ct)$ and
$ck = h$, the grid values are $F_{j-n} + G_{j+n}$ writing $F_m = F(x_0 + mh)$. Then

$$
u_{j+1}^n + u_{j-1}^n - u_j^{n-1}
= (F_{j+1-n} + G_{j+1+n}) + (F_{j-1-n} + G_{j-1+n}) - (F_{j-n+1} + G_{j+n-1})
= F_{j-1-n} + G_{j+1+n},
$$

which is $u_j^{n+1}$. **Every term matches by index arithmetic alone**, with no Taylor expansion,
so the identity is exact for any $F$ and $G$ whatsoever.

The geometry behind the algebra: at $\lambda = 1$ the characteristics $x \pm ct$ pass through grid
points, so the scheme is transporting values along them rather than approximating the transport.

### 1.5 What unconditional stability buys, in each case

**Parabolic.** The solution decays, so an error made early decays with it. A large stable step is
worth taking: it costs accuracy per step and the accumulated error is bounded by the decay. Lesson
78 measures Crank-Nicolson at $\text{tol}^{-1}$ against the explicit scheme's $\text{tol}^{-3/2}$.

**Hyperbolic.** Nothing decays. A phase error made at step 1 is still there at step 1000, and the
errors of successive steps add rather than dying away. So a large stable step buys speed at a
proportional cost in accuracy, and the lesson measures the error over one period growing by a
factor of 6930 as $\lambda$ goes from $0.5$ to $8$.

**Unconditional stability is a licence to choose the step for accuracy instead of for stability.**
For a parabolic problem those two are far apart; for a hyperbolic one they are close together, and
the licence is worth much less.

### 2.1 The amplification quadratic

Substitute $u_j^n = g^ne^{ij\phi}$ into
$u_j^{n+1} - 2u_j^n + u_j^{n-1} = \lambda^2\delta^2u_j^n$ and divide by $g^{n-1}e^{ij\phi}$:

$$
g^2 - 2g + 1 = \lambda^2\left(-4\sin^2\tfrac{\phi}{2}\right)g
\quad\Longrightarrow\quad
g^2 - \left(2 - L\right)g + 1 = 0, \qquad L = 4\lambda^2\sin^2\tfrac{\phi}{2}.
$$

The **product of the roots is the constant term, which is 1**, for every $\lambda$ and every
$\phi$. So $\lvert g_1\rvert\lvert g_2\rvert = 1$ always, and the only way both can have modulus at
most 1 is for both to have modulus exactly 1.

### 2.2 Where $\lambda \le 1$ comes from

Both roots lie on the unit circle exactly when they are complex conjugates of unit modulus, which
for a real quadratic $g^2 - bg + 1$ means the discriminant is not positive:

$$
b^2 - 4 \le 0
\quad\text{with}\quad
b = 2 - L
\quad\Longrightarrow\quad
(2 - L)^2 \le 4
\quad\Longrightarrow\quad
0 \le L \le 4 .
$$

$L = 4\lambda^2\sin^2(\phi/2)$ is largest at $\phi = \pi$, where it is $4\lambda^2$. So the
condition over all modes is $4\lambda^2 \le 4$, that is

$$
\lambda \le 1 .
$$

Above it, $L > 4$ at the shortest wavelength, the discriminant is positive, the roots are real with
product 1, and one of them is strictly outside the circle.

```python
print(f"{'lambda':>9}{'L at phi = pi':>16}{'discriminant':>15}{'largest root':>15}")
for lam in (0.5, 0.9, 1.0, 1.05, 1.5):
    big = 4.0 * lam ** 2
    print(f"{lam:>9.3f}{big:>16.4f}{(2.0 - big) ** 2 - 4.0:>15.4f}"
          f"{hy.largest_root(lam, 0.0):>15.5f}")
```

### 2.3 The second order starting step

The recurrence needs $u^1$. Taylor about $t = 0$:

$$
u^1 = u(x, k) = u^0 + k\,u_t(x,0) + \frac{k^2}{2}u_{tt}(x,0) + O(k^3).
$$

$u_t(x,0) = g(x)$ is given. $u_{tt}(x,0)$ is **not** given, and the equation supplies it:
$u_{tt} = c^2u_{xx}$, so $u_{tt}(x,0) = c^2u''_0(x)$ where $u_0$ is the initial shape. Differencing
that with the same three point stencil,

$$
u^1_j = u^0_j + k\,g_j + \frac{c^2k^2}{2h^2}\left(u^0_{j+1} - 2u^0_j + u^0_{j-1}\right)
= u^0_j + k\,g_j + \frac{\lambda^2}{2}\delta^2u^0_j .
$$

Dropping the last term leaves an $O(k^2)$ error in $u^1$, which is an $O(k)$ error in the slope the
recurrence starts from, and the run inherits it: first order.

```python
out = hy.the_first_step_sets_the_order()
print(f"{'start':>7}{'order':>9}{'errors':>50}")
for row in out["rows"]:
    print(f"{row['start_order']:>7}{row['order']:>9.4f}"
          f"{np.array2string(row['error'], precision=3):>50}")
print(f"\ngain at the finest grid: {out['gain_at_the_finest']:.0f}x")
```

### 2.4 The numerical wave speed at $\lambda = 1$

When stable, the roots are $e^{\pm i\omega k}$ with $\lvert g\rvert = 1$, and matching real parts
in $g^2 - (2 - L)g + 1 = 0$ gives $2\cos(\omega k) = 2 - L$, that is

$$
1 - \cos(\omega k) = 2\lambda^2\sin^2\tfrac{\phi}{2}
\quad\Longrightarrow\quad
\sin^2\frac{\omega k}{2} = \lambda^2\sin^2\frac{\phi}{2},
$$

so $\sin(\omega k/2) = \lambda\sin(\phi/2)$. The numerical speed is
$\omega/\kappa = \omega h/\phi$, and with $k = \lambda h/c$,

$$
\frac{c_{\text{num}}}{c} = \frac{2}{\lambda\phi}\arcsin\!\left(\lambda\sin\frac{\phi}{2}\right).
$$

At $\lambda = 1$ the arcsine and the sine cancel: $\arcsin(\sin(\phi/2)) = \phi/2$ for
$\phi \in [0, \pi]$, so the ratio is $\frac{2}{\phi}\cdot\frac{\phi}{2} = 1$ **for every $\phi$**.

```python
out = hy.dispersion_is_worst_for_short_waves()
print(f"{'lambda':>9}{'worst relative error':>24}")
for row in out["rows"]:
    print(f"{row['lam']:>9.2f}{row['worst_relative_error']:>24.4e}")
print(f"\nexactly right at lambda = 1 for every wavenumber: {out['exact_at_lam_one']}")
```

### 2.5 The condition $\theta \ge 1/4$

The weighted scheme's quadratic is

$$
(1 + \theta L)g^2 - \left(2 - (1 - 2\theta)L\right)g + (1 + \theta L) = 0,
\qquad L = 4\lambda^2\sin^2\tfrac{\phi}{2}.
$$

The product of the roots is now $(1 + \theta L)/(1 + \theta L) = 1$ again, so the same argument
applies: both roots are on the circle exactly when the discriminant is not positive,

$$
\left(2 - (1 - 2\theta)L\right)^2 \le 4(1 + \theta L)^2 .
$$

Taking square roots of both nonnegative sides and keeping the binding branch,

$$
2 - (1 - 2\theta)L \ge -2(1 + \theta L)
\quad\Longrightarrow\quad
4 \ge L\left(1 - 2\theta - 2\theta\right) = L(1 - 4\theta).
$$

If $\theta \ge 1/4$ the right side is at most zero and the inequality holds for **every** $L$,
which is unconditional stability. If $\theta < 1/4$ it gives $L \le 4/(1 - 4\theta)$ and hence
$\lambda^2 \le 1/(1 - 4\theta)$.

```python
out = hy.the_implicit_threshold_is_a_quarter()
print(f"{'theta':>8}{'1 - 4 theta':>14}{'largest lambda':>17}{'unconditional':>16}")
for row in out["rows"]:
    limit = ("unconditional" if row["stable_everywhere"]
             else f"{1.0 / math.sqrt(max(1.0 - 4.0 * row['theta'], 1e-30)):.4f}")
    print(f"{row['theta']:>8.2f}{1.0 - 4.0 * row['theta']:>14.2f}{limit:>17}"
          f"{str(row['stable_everywhere']):>16}")
```

### 3.1 A free end by the ghost point

$u_x(b, t) = 0$. Put a ghost node at $x_{n}$ beyond the last real one, use the centred difference
there, $u_{n} = u_{n-2}$, and substitute into the scheme at the last real node:

$$
u^{n+1}_{N} = 2u^n_N - u^{n-1}_N + 2\lambda^2\left(u^n_{N-1} - u^n_N\right).
$$

The alternative, $u_N = u_{N-1}$, is the one sided first order condition.

```python
def free_end(n, steps, t_end, ghost=True):
    """u_tt = u_xx, u(0,t) = 0, u_x(1,t) = 0; exact u = cos(pi t / 2) sin(pi x / 2)."""
    x = np.linspace(0.0, 1.0, n)
    h = float(x[1] - x[0])
    k = t_end / steps
    lam = k / h
    exact = lambda t: math.cos(math.pi * t / 2.0) * np.sin(math.pi * x / 2.0)
    old = exact(0.0)
    now = old.copy()
    now[1:-1] += 0.5 * lam ** 2 * (old[2:] - 2.0 * old[1:-1] + old[:-2])
    now[-1] = (old[-1] + lam ** 2 * (old[-2] - old[-1])) if ghost else now[-2]
    now[0] = 0.0
    for _ in range(1, steps):
        new = np.empty(n)
        new[1:-1] = (2.0 * now[1:-1] - old[1:-1]
                     + lam ** 2 * (now[2:] - 2.0 * now[1:-1] + now[:-2]))
        new[-1] = ((2.0 * now[-1] - old[-1] + 2.0 * lam ** 2 * (now[-2] - now[-1]))
                   if ghost else new[-2])
        new[0] = 0.0
        old, now = now, new
    return h, float(np.max(np.abs(now - exact(steps * k))))

print(f"{'treatment':>12}{'order':>9}{'errors':>52}")
for ghost in (True, False):
    hs, errs = [], []
    for n in (21, 41, 81, 161, 321):
        h = 1.0 / (n - 1)
        steps = max(int(round(0.8 / (0.8 * h))), 2)
        hh, e = free_end(n, steps, steps * 0.8 * h, ghost)
        hs.append(hh); errs.append(e)
    label = "ghost point" if ghost else "one sided"
    print(f"{label:>12}{float(np.polyfit(np.log(hs), np.log(errs), 1)[0]):>9.4f}"
          f"{np.array2string(np.asarray(errs), precision=3):>52}")
```

Ghost point **2.0001**, one sided **0.9921**. The same finding as lesson 77's Neumann boundary and
lesson 71's starting values: one row sets the order.

### 3.2 Three schemes for advection

For $u_t + au_x = 0$ with $\nu = ak/h$:

$$
\textbf{upwind:}\quad u_j^{n+1} = u_j^n - \nu(u_j^n - u_{j-1}^n),
$$
$$
\textbf{Lax-Friedrichs:}\quad u_j^{n+1} = \tfrac12(u_{j+1}^n + u_{j-1}^n)
- \tfrac{\nu}{2}(u_{j+1}^n - u_{j-1}^n),
$$
$$
\textbf{Lax-Wendroff:}\quad u_j^{n+1} = u_j^n - \tfrac{\nu}{2}(u_{j+1}^n - u_{j-1}^n)
+ \tfrac{\nu^2}{2}(u_{j+1}^n - 2u_j^n + u_{j-1}^n).
$$

All three have the same CFL condition, $\nu \le 1$, and they differ entirely in what they do
**inside** it.

```python
def advect(n, steps, t_end, scheme, speed=1.0):
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    h = float(x[1] - x[0])
    k = t_end / steps
    nu = speed * k / h
    shape = lambda s: np.exp(-((np.mod(s + 0.5, 1.0) - 0.5) / 0.08) ** 2)
    u = shape(x)
    for _ in range(steps):
        left, right = np.roll(u, 1), np.roll(u, -1)
        if scheme == "upwind":
            u = u - nu * (u - left)
        elif scheme == "lax friedrichs":
            u = 0.5 * (left + right) - 0.5 * nu * (right - left)
        else:
            u = u - 0.5 * nu * (right - left) + 0.5 * nu ** 2 * (right - 2.0 * u + left)
    return nu, u, shape(x - speed * steps * k)

print(f"{'scheme':>16}{'nu':>7}{'error':>13}{'peak':>10}")
for scheme in ("upwind", "lax friedrichs", "lax wendroff"):
    for target in (0.5, 0.9, 1.0, 1.05):
        n = 201
        k = target / n
        steps = max(int(round(0.5 / k)), 1)
        with np.errstate(over="ignore", invalid="ignore"):
            nu, u, want = advect(n, steps, steps * k, scheme)
        finite = bool(np.isfinite(u).all())
        err = float(np.max(np.abs(u - want))) if finite else float("inf")
        print(f"{scheme:>16}{nu:>7.2f}{err:>13.3e}"
              f"{(float(np.max(u)) if finite else float('inf')):>10.4f}")
```

Three things, and the third is the interesting one.

**All three are exact at $\nu = 1$**, to $1.8\times10^{-15}$. Same reason as the wave equation: at
$\nu = 1$ each of them reduces to $u_j^{n+1} = u_{j-1}^n$, which transports along the
characteristic exactly.

**Their diffusion differs by an order of magnitude.** At $\nu = 0.5$, after one full traverse the
Gaussian's peak falls to $0.750$ under upwind, $0.548$ under Lax-Friedrichs and $0.997$ under
Lax-Wendroff. Lax-Friedrichs is by far the most diffusive; its modified equation has an artificial
viscosity of $\frac{h^2}{2k}(1 - \nu^2)$, which **grows as the step shrinks**.

**At $\nu = 1.05$ none of them blows up over this run.** The growth factors at $\phi = \pi$ are
$1.10$, $1.00$ and $1.205$, and over 95 steps the smooth data has not yet seeded the growing mode
enough. Lax-Friedrichs is subtler still: its factor at $\phi = \pi$ is exactly 1 for any $\nu$, and
its instability lives at $\phi = \pi/2$ where $\lvert g\rvert = \nu$.

```python
print(f"{'nu':>7}{'upwind':>10}{'Lax-Friedrichs':>18}{'Lax-Wendroff':>16}"
      f"{'LF worst phase':>17}")
for nu in (0.5, 0.9, 1.0, 1.05):
    phases = np.linspace(0.0, np.pi, 2001)
    up = np.abs(1.0 - nu * (1.0 - np.exp(-1j * phases)))
    lf = np.abs(np.cos(phases) - 1j * nu * np.sin(phases))
    lw = np.abs(1.0 - 1j * nu * np.sin(phases) - nu ** 2 * (1.0 - np.cos(phases)))
    print(f"{nu:>7.2f}{float(np.max(up)):>10.5f}{float(np.max(lf)):>18.5f}"
          f"{float(np.max(lw)):>16.5f}{float(phases[int(np.argmax(lf))]):>17.4f}")
```

### 3.3 A variable wave speed

Replace $\lambda$ by $\lambda_j = c(x_j)k/h$ in the scheme. The condition becomes

$$
\max_j \lambda_j \le 1 ,
$$

a **pointwise** condition, so the step is set by the fastest point in the domain.

```python
def variable_speed(n, steps, t_end, speed_of):
    x = np.linspace(0.0, 1.0, n)
    h = float(x[1] - x[0])
    k = t_end / steps
    lam = speed_of(x) * k / h
    old = np.exp(-((x - 0.5) / 0.06) ** 2)
    now = old.copy()
    now[1:-1] += 0.5 * lam[1:-1] ** 2 * (old[2:] - 2.0 * old[1:-1] + old[:-2])
    now[0] = now[-1] = 0.0
    peak = [float(np.max(np.abs(old)))]
    for _ in range(1, steps):
        new = np.empty(n)
        new[1:-1] = (2.0 * now[1:-1] - old[1:-1]
                     + lam[1:-1] ** 2 * (now[2:] - 2.0 * now[1:-1] + now[:-2]))
        new[0] = new[-1] = 0.0
        old, now = now, new
        peak.append(float(np.max(np.abs(now))))
    return float(np.max(lam)), np.asarray(peak)

print(f"{'c range':>14}{'max lambda':>13}{'grew by':>14}")
for top in (1.0, 2.0, 4.0):
    speed_of = lambda s, t=top: 1.0 + (t - 1.0) * s
    for target in (0.9, 1.0, 1.05):
        h = 1.0 / 200
        k = target * h / top
        steps = max(int(round(0.3 / k)), 2)
        with np.errstate(over="ignore", invalid="ignore"):
            worst, peak = variable_speed(201, steps, steps * k, speed_of)
        print(f"{'1 to ' + str(top):>14}{worst:>13.4f}{peak[-1] / peak[0]:>14.3e}")
```

**A variable speed supplies the seed that a constant speed run has to wait for.** With $c$ constant
and $\lambda = 1.05$ the run over 300 steps grows by $0.5$: nothing has happened, because the
smooth data contains none of the mode that grows. With $c$ running from 1 to 4 and the same
$\max\lambda = 1.05$, the run grows by $6.5\times10^{9}$.

The reason is that varying coefficients **couple the modes**. A constant coefficient scheme acts on
each Fourier mode independently, so a mode absent from the data stays absent until rounding
supplies it; a varying coefficient scatters energy from the modes that are present into the ones
that are not, and the growing mode is populated on the first step.

That is a useful diagnostic in both directions: a constant coefficient test can hide an instability
for a long time, and a variable coefficient test cannot.

### 3.4 The two dimensional wave equation

The growth factor picks up a contribution from each direction, so the quadratic's $L$ becomes
$4\lambda^2(\sin^2\frac{\phi_x}{2} + \sin^2\frac{\phi_y}{2})$, whose maximum is $8\lambda^2$. The
condition $L \le 4$ gives

$$
\lambda \le \frac{1}{\sqrt2},
$$

and in $d$ dimensions $\lambda \le 1/\sqrt d$.

```python
def wave_2d(n, steps, lam, modes=(1, 1)):
    line = np.linspace(0.0, 1.0, n)
    h = float(line[1] - line[0])
    k = lam * h
    gx, gy = np.meshgrid(line, line, indexing="ij")
    p, q = modes
    freq = np.pi * math.sqrt(p * p + q * q)
    exact = lambda t: (math.cos(freq * t) * np.sin(p * np.pi * gx)
                       * np.sin(q * np.pi * gy))
    old = exact(0.0)
    now = old.copy()
    now[1:-1, 1:-1] += 0.5 * lam ** 2 * (
        old[2:, 1:-1] + old[:-2, 1:-1] + old[1:-1, 2:] + old[1:-1, :-2]
        - 4.0 * old[1:-1, 1:-1])
    for _ in range(1, steps):
        new = np.zeros_like(now)
        new[1:-1, 1:-1] = (2.0 * now[1:-1, 1:-1] - old[1:-1, 1:-1]
                           + lam ** 2 * (now[2:, 1:-1] + now[:-2, 1:-1]
                                         + now[1:-1, 2:] + now[1:-1, :-2]
                                         - 4.0 * now[1:-1, 1:-1]))
        old, now = now, new
    return float(np.max(np.abs(now - exact(steps * k))))

print(f"{'lambda':>10}{'(1,1)':>13}{'(2,1)':>13}{'(3,2)':>13}")
for lam in (0.5, 0.7, 1.0 / math.sqrt(2.0), 0.8):
    row = ""
    for modes in ((1, 1), (2, 1), (3, 2)):
        with np.errstate(over="ignore", invalid="ignore"):
            row += f"{wave_2d(81, 100, lam, modes):>13.3e}"
    print(f"{lam:>10.5f}{row}")
```

Two findings.

**The limit is $1/\sqrt2 = 0.7071$**, and $\lambda = 0.8$ gives errors of $10^{26}$ and larger.

**The apparent exactness at $\lambda = 1/\sqrt2$ is a coincidence of the symmetric mode.** For
$(1,1)$ the error is $3.1\times10^{-15}$, at the rounding level, and it is tempting to call it a
two dimensional magic step. It is not: at $(2,1)$ the same $\lambda$ gives $2.7\times10^{-5}$ and
at $(3,2)$ it gives $3.4\times10^{-4}$. **There is no two dimensional analogue of the $\lambda = 1$
exactness**, because the characteristics of a two dimensional wave equation form a cone rather than
two lines, and no rectangular grid can lie along a cone.

### 3.5 A non uniform grid

The three point second difference on unequal spacings $h_-$ and $h_+$ is

$$
u'' \approx \frac{2}{h_-h_+(h_-+h_+)}\left[h_+u_{j-1} - (h_-+h_+)u_j + h_-u_{j+1}\right],
$$

which is lesson 75's formula. Use it in the scheme and ask what step is allowed.

```python
def non_uniform(n, steps, t_end, stretch):
    s = np.linspace(0.0, 1.0, n)
    x = s + stretch * s * (1.0 - s)
    x = (x - x[0]) / (x[-1] - x[0])
    hs = np.diff(x)
    k = t_end / steps
    exact = lambda t: math.cos(np.pi * t) * np.sin(np.pi * x)

    def second(v):
        left, right = hs[:-1], hs[1:]
        return 2.0 * (right * v[:-2] - (left + right) * v[1:-1] + left * v[2:]) / (
            left * right * (left + right))

    old = exact(0.0)
    now = old.copy()
    now[1:-1] += 0.5 * k ** 2 * second(old)
    now[0] = now[-1] = 0.0
    for _ in range(1, steps):
        new = np.empty(n)
        new[1:-1] = 2.0 * now[1:-1] - old[1:-1] + k ** 2 * second(now)
        new[0] = new[-1] = 0.0
        old, now = now, new
    return float(np.max(np.abs(now - exact(steps * k))))

print(f"{'stretch':>9}{'h min':>10}{'h max':>10}{'k = h min':>13}{'k = h mean':>13}"
      f"{'k = h max':>13}")
for stretch in (0.0, 0.3, 1.0):
    s = np.linspace(0.0, 1.0, 201)
    x = s + stretch * s * (1.0 - s)
    x = (x - x[0]) / (x[-1] - x[0])
    hs = np.diff(x)
    row = ""
    for choice in (float(np.min(hs)), float(np.mean(hs)), float(np.max(hs))):
        steps = max(int(round(0.4 / choice)), 2)
        with np.errstate(over="ignore", invalid="ignore"):
            row += f"{non_uniform(201, steps, steps * choice, stretch):>13.3e}"
    print(f"{stretch:>9.2f}{float(np.min(hs)):>10.5f}{float(np.max(hs)):>10.5f}{row}")
```

**The exactness does not survive any stretching at all.** On a uniform grid with $k = h$ the error
is $1.3\times10^{-14}$, rounding. Stretch the grid by 30 per cent and the same rule, $k = h_{\min}$,
gives $1.2\times10^{-5}$: nine orders of magnitude worse, and that is the **best** of the three
choices.

**And the step must be tied to the smallest spacing.** Taking $k = h_{\text{mean}}$ or
$k = h_{\max}$ blows the run up, because the Courant condition is local and the closest pair of
points sets it. A grid graded by a factor of 300, which the stretch of 1.0 produces here, costs a
factor of 300 in the step count for the whole domain.

That is the central difficulty with grading a hyperbolic problem, and it is why the remedies are
local time stepping or a mapped uniform grid rather than simply using an unequal mesh.

### 4.1 How long a violated run survives

The prediction is the same product as in lessons 78 and 80: the worst mode's amplitude in the data
times the growth per step, to the power of the step count.

```python
out = hy.the_courant_condition()
print(f"{'sweep':>12}{'lambda':>9}{'root':>9}{'seed':>11}{'predicted':>13}"
      f"{'error':>13}{'blew up':>10}")
for label, key in (("short", "short_run"), ("long", "long_run")):
    for row in out[key]:
        print(f"{label:>12}{row['lam']:>9.3f}{row['largest_root']:>9.4f}"
              f"{row['seed']:>11.2e}"
              f"{'10^' + format(row['predicted_decades'], '.1f'):>13}"
              f"{row['error']:>13.3e}{str(row['blew_up']):>10}")
print(f"\nright in all {out['rows_checked']} rows: "
      f"{out['the_prediction_is_right_in_every_row']}")
print(f"violations that failed in the long run {out['violations_that_failed_in_the_long_run']}, "
      f"and one that had not {out['violations_that_did_not']}")
```

The roughness of the data enters entirely through the seed. On the smooth bump it is machine
epsilon, so $\lambda = 1.001$ survives 240 steps; exercise 3.3 shows that a **variable coefficient**
supplies a seed of order 1 immediately and the same violation fails at once.

### 4.2 The wake, quantified

Track the peak of the right going pulse by fitting a parabola through the three grid values around
the maximum, and fit its position against time.

```python
def pulse_position(lam, points=401, t_end=0.6):
    problem = hy.travelling_bump(a=0.0, b=3.0, centre=0.5, width=0.06)
    h = (problem["b"] - problem["a"]) / (points - 1)
    k = lam * h / problem["speed"]
    steps = max(int(round(t_end / k)), 2)
    run = hy.solve(problem, points, steps, steps * k, theta=0.0, keep=True)
    x = run["x"]
    times, positions = [], []
    for level, profile in zip(run["t"], run["history"]):
        if level < 0.2 * t_end:
            continue
        right, xs = profile[x > 0.5], x[x > 0.5]
        peak = int(np.argmax(right))
        if peak == 0 or peak == right.size - 1:
            continue
        a, b, c = right[peak - 1], right[peak], right[peak + 1]
        shift = 0.5 * (a - c) / (a - 2.0 * b + c)
        positions.append(float(xs[peak] + shift * (xs[1] - xs[0])))
        times.append(float(level))
    times, positions = np.asarray(times), np.asarray(positions)
    return (float(np.polyfit(times, positions, 1)[0]),
            float(np.max(np.abs(positions - (0.5 + times)))))

print(f"{'lambda':>9}{'peak speed':>14}{'speed error':>15}{'largest position error':>25}")
for lam in (0.25, 0.5, 0.8, 0.95, 1.0):
    speed, drift = pulse_position(lam)
    print(f"{lam:>9.2f}{speed:>14.5f}{1.0 - speed:>15.2e}{drift:>25.3e}")
```

The measured peak speed is $0.99639$, $0.99709$, $0.99859$, $0.99962$ and exactly $1.00000$ at
$\lambda = 1$, and the accumulated position error falls from $2.2\times10^{-3}$ to
$1.1\times10^{-5}$ over the same range.

**The pulse is slow, never fast**, at every $\lambda < 1$, which is exercise 5.1's sign result
showing up as a physical lag. And the lag **accumulates linearly with time**, since it is a speed
error rather than a one off: a run twice as long has twice the position error, which is why phase
errors dominate long hyperbolic runs and amplitude errors dominate parabolic ones.

### 4.3 The discrete energy

The continuous energy $E = \tfrac12\int(u_t^2 + c^2u_x^2)$ is conserved exactly. Its natural
discrete analogue uses the centred time difference for $u_t$ and the average of the two levels for
$u_x$.

```python
def energy_run(theta, lam, points=201, periods=20.0):
    problem = hy.standing_wave()
    h = (problem["b"] - problem["a"]) / (points - 1)
    k = lam * h / problem["speed"]
    steps = max(int(round(periods * problem["period"] / k)), 2)
    x = np.linspace(problem["a"], problem["b"], points)
    old = problem["initial"](x)
    old[0] = old[-1] = 0.0
    now = hy.first_step(old, problem["velocity"](x), lam, k, order=2)
    now[0] = now[-1] = 0.0
    energies = []
    for _ in range(1, steps):
        new = (hy.explicit_step(old, now, lam) if theta == 0.0
               else hy.implicit_step(old, now, lam, theta))
        rate = (new - old) / (2.0 * k)
        slope = np.diff(0.5 * (new + old)) / h
        energies.append(0.5 * (float(np.sum(rate[1:-1] ** 2)) * h
                               + float(np.sum(slope ** 2)) * h))
        old, now = now, new
    return np.asarray(energies)

print(f"{'scheme':>10}{'lambda':>9}{'E first':>12}{'E last':>12}{'spread':>12}"
      f"{'relative drift':>17}")
for theta, lam in ((0.0, 0.8), (0.0, 1.0), (0.25, 0.8), (0.25, 4.0)):
    energies = energy_run(theta, lam)
    label = "explicit" if theta == 0.0 else "implicit"
    spread = float(np.max(energies) - np.min(energies)) / float(np.mean(energies))
    drift = (float(np.polyfit(np.arange(energies.size), energies, 1)[0])
             * energies.size / float(np.mean(energies)))
    print(f"{label:>10}{lam:>9.2f}{energies[0]:>12.6f}{energies[-1]:>12.6f}"
          f"{spread:>12.3e}{drift:>17.3e}")
```

**Both schemes conserve it**, and neither drifts. Over 20 periods the explicit scheme's energy
oscillates within $1.2\times10^{-4}$ of its mean with a trend of $1.3\times10^{-9}$, and the
implicit scheme at $\lambda = 4$ oscillates within $2.0\times10^{-3}$ with a trend of
$2.0\times10^{-6}$.

At $\lambda = 1$ the explicit scheme's trend is $1.8\times10^{-15}$: **exactly conserved**, as the
exactness of section 4 requires.

The pattern is the same one lesson 74 found for symplectic integrators: the energy of a
structure preserving method **oscillates and does not drift**, and the amplitude of the oscillation
grows with the step while the trend stays at zero. Both of these schemes are the discrete
Euler-Lagrange equations of a discrete action, which is why.

### 5.1 The CFL condition is necessary and not sufficient

The centred scheme for advection,

$$
u_j^{n+1} = u_j^n - \frac{\nu}{2}\left(u_{j+1}^n - u_{j-1}^n\right),
$$

reaches one grid point either way per step, so its numerical domain of dependence is
$[x_j - nh, x_j + nh]$, which contains the true one $[x_j - ant k, x_j]$ whenever $\nu \le 1$. It
**satisfies the CFL requirement** and is unconditionally unstable.

```python
def centred(n, steps, nu):
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    u = np.exp(-((np.mod(x, 1.0) - 0.5) / 0.08) ** 2)
    peak = [float(np.max(np.abs(u)))]
    for _ in range(steps):
        u = u - 0.5 * nu * (np.roll(u, -1) - np.roll(u, 1))
        peak.append(float(np.max(np.abs(u))))
    return np.asarray(peak)

print(f"{'nu':>7}{'largest |g|':>14}{'grew by over 400 steps':>26}")
for nu in (0.1, 0.5, 0.9, 1.0):
    phases = np.linspace(0.0, np.pi, 401)
    with np.errstate(over="ignore", invalid="ignore"):
        peak = centred(201, 400, nu)
    print(f"{nu:>7.2f}{float(np.max(np.abs(1.0 - 1j * nu * np.sin(phases)))):>14.5f}"
          f"{peak[-1] / peak[0]:>26.4e}")
```

At $\nu = 0.5$ the scheme grows by $1.5\times10^{3}$ over 400 steps, and at $\nu = 0.9$ by
$10^{35}$.

**The missing ingredient is the amplification factor.** Substituting one mode gives
$g = 1 - i\nu\sin\phi$, so

$$
\lvert g\rvert = \sqrt{1 + \nu^2\sin^2\phi} > 1
$$

for every $\nu > 0$ and every mode except the two with $\sin\phi = 0$. The CFL condition asks
whether the scheme **can see** the right data; von Neumann asks whether it does the right
arithmetic with it. Seeing is necessary and it is not enough.

The fix in this case is one term: adding $\frac{\nu^2}{2}\delta^2u^n$ gives Lax-Wendroff, whose
factor is $\lvert g\rvert^2 = 1 - \nu^2(1-\nu^2)(1 - \cos\phi)^2 \le 1$ for $\nu \le 1$. The extra
term is exactly the $\frac{k^2}{2}u_{tt}$ of the Taylor expansion, so the cure and the second order
accuracy arrive together.

### 5.2 The group velocity

The **phase** velocity is $\omega/\kappa$ and the **group** velocity, at which a wave packet's
envelope travels, is $\mathrm{d}\omega/\mathrm{d}\kappa$. For the wave scheme, differentiating
$\sin(\omega k/2) = \lambda\sin(\phi/2)$ with $\kappa = \phi/h$,

$$
\frac{\mathrm{d}\omega}{\mathrm{d}\kappa}
= \frac{\cos(\phi/2)}{\sqrt{1 - \lambda^2\sin^2(\phi/2)}} \cdot c .
$$

```python
print(f"{'lambda':>9}{'at phi = 0':>13}{'at phi = pi':>14}{'minimum':>11}")
for lam in (0.25, 0.5, 0.8, 1.0):
    phases = np.linspace(1e-6, np.pi - 1e-6, 4001)
    group = np.cos(phases / 2.0) / np.sqrt(1.0 - lam ** 2 * np.sin(phases / 2.0) ** 2)
    print(f"{lam:>9.2f}{group[0]:>13.5f}{group[-1]:>14.5f}{float(np.min(group)):>11.5f}")
```

**For this scheme the group velocity is never negative.** For $\lambda < 1$ it falls from $c$ at
the longest waves to **zero** at $\phi = \pi$, because $\cos(\phi/2)$ vanishes there. The shortest
wave the grid carries does not travel at all: energy in it stays where it was put.

At $\lambda = 1$ the numerator and the denominator vanish together, $\sqrt{1 - \sin^2(\phi/2)} =
\cos(\phi/2)$, and the ratio is **1 at every wavenumber**. That is the magic step again, now stated
for the group velocity: not only does every wave travel at the right speed, so does every packet.

That is a real defect and it is not the one the exercise expects. The negative group velocity does
occur, and it needs a different scheme.

```python
print(f"\nleapfrog for advection, sin(omega k) = nu sin(phi):")
print(f"{'nu':>7}{'at phi = 0':>13}{'at phi = pi':>14}{'negative on':>14}")
for nu in (0.2, 0.5, 0.8):
    phases = np.linspace(1e-6, np.pi - 1e-6, 4001)
    group = np.cos(phases) / np.sqrt(1.0 - nu ** 2 * np.sin(phases) ** 2)
    print(f"{nu:>7.2f}{group[0]:>13.5f}{group[-1]:>14.5f}"
          f"{100.0 * float(np.mean(group < 0.0)):>13.1f}%")
```

For leapfrog, $\sin(\omega k) = \nu\sin\phi$ gives a group velocity of
$\cos\phi/\sqrt{1 - \nu^2\sin^2\phi}$, which is **negative for every $\phi > \pi/2$**, that is on
**half the spectrum**, and reaches $-c$ at $\phi = \pi$.

The consequence is concrete: a wave packet made of short waves travels **backwards** at up to the
full wave speed, in the wrong direction entirely, while the scheme remains perfectly stable and
formally second order. It is the sharpest illustration in this part of the difference between an
error bound and a wrong answer: nothing about the amplitude is wrong, and the physics is reversed.

### 5.3 A discrete energy conserved exactly at $\lambda \le 1$

Define

$$
E^{n+1/2} = \frac12\sum_j\left[\left(\frac{u_j^{n+1} - u_j^n}{k}\right)^2
+ c^2\,\frac{(u_{j+1}^{n+1} - u_j^{n+1})(u_{j+1}^{n} - u_j^{n})}{h^2}\right]h .
$$

Note the **cross term** in the second part: the product of the two levels' gradients rather than
the square of either. That is what makes the quantity exactly conserved.

Multiplying the scheme by $(u^{n+1} - u^{n-1})/(2k)$ and summing by parts gives
$E^{n+1/2} = E^{n-1/2}$ identically, with no remainder and no smallness assumption.

**Positivity is where $\lambda$ enters.** The cross term is not a square, so $E$ is not
automatically nonnegative, and

$$
E^{n+1/2} \ge \frac{1 - \lambda^2}{2}\sum_j\left(\frac{u^{n+1}_j - u^n_j}{k}\right)^2h
$$

after a discrete Cauchy-Schwarz. For $\lambda < 1$ the coefficient is positive, so $E$ is a genuine
norm and its conservation bounds the solution: that is a **proof of stability** that never
mentions Fourier modes and extends to variable coefficients, where von Neumann's argument does not.

At $\lambda = 1$ exactly the coefficient is zero: $E$ is conserved and is no longer a norm, which
is the boundary case the roots of exercise 2.1 also sit on. And the measurement of exercise 4.3
finds the trend at $\lambda = 1$ to be $1.8\times10^{-15}$, exactly conserved, which is this
identity being satisfied to rounding.

```python
def cross_energy(lam, points=201, steps=400):
    problem = hy.standing_wave()
    h = (problem["b"] - problem["a"]) / (points - 1)
    k = lam * h / problem["speed"]
    x = np.linspace(problem["a"], problem["b"], points)
    old = problem["initial"](x)
    old[0] = old[-1] = 0.0
    now = hy.first_step(old, problem["velocity"](x), lam, k, order=2)
    now[0] = now[-1] = 0.0
    values = []
    for _ in range(steps):
        new = hy.explicit_step(old, now, lam)
        rate = (new - now) / k
        cross = np.diff(new) * np.diff(now) / h ** 2
        values.append(0.5 * (float(np.sum(rate ** 2)) + float(np.sum(cross))) * h)
        old, now = now, new
    return np.asarray(values)

print(f"{'lambda':>9}{'first':>13}{'last':>13}{'relative spread':>18}{'positive':>10}")
for lam in (0.5, 0.9, 1.0):
    values = cross_energy(lam)
    spread = float(np.max(values) - np.min(values)) / abs(float(np.mean(values)))
    print(f"{lam:>9.2f}{values[0]:>13.6f}{values[-1]:>13.6f}{spread:>18.3e}"
          f"{str(bool(np.all(values > 0))):>10}")
```

The cross energy is constant to $10^{-14}$ at every $\lambda \le 1$, which is the identity holding
to rounding, and it stays positive, which is the bound holding.

---

## Lesson 82, Elliptic Equations

### 1.1 Why an elliptic problem gives a system and not a recurrence

In a parabolic or hyperbolic problem one variable is time, and time has a direction. The scheme can
be arranged so that the values on the new level are written in terms of values already known, and
the whole solve is a march: compute level $n+1$, throw level $n-1$ away, repeat. Even an implicit
scheme only has to solve a one dimensional system per step.

An elliptic problem has no such variable. Laplace's equation is a statement about a point and its
neighbours in **every** direction at once, and the boundary conditions are given on the whole
boundary rather than on one end of it. Fix a value anywhere and the condition at its neighbours
changes, which changes their neighbours, all the way around to the other side of the domain. There
is no ordering of the grid points in which each one depends only on points already computed.

So the discrete problem is the whole set of interior equations, coupled, solved together: one
system of $(n-1)^2$ equations. The size is the price of having no direction to march in.

The same fact appears in the analysis. A parabolic problem is well posed forward in time and ill
posed backward; an elliptic problem has no time, so there is no direction in which it is ill posed
and no direction in which information propagates. Changing the boundary data anywhere changes the
solution everywhere, immediately, which is exactly the coupling the matrix records.

### 1.2 The eigenvalues and the condition number

On a square grid of $m$ interior points a side with spacing $h = 1/(m+1)$, the five point matrix
for $-\Delta$ has eigenvalues

$$\lambda_{p,q} = \frac{4}{h^{2}}\left[\sin^{2}\frac{p\pi h}{2} + \sin^{2}\frac{q\pi h}{2}\right],
\qquad p, q = 1, \dots, m ,$$

with eigenvectors $v_{p,q}(i,j) = \sin(p\pi i h)\sin(q\pi j h)$, the discrete sine modes in both
directions. There are exactly $m^2$ of them, which is the size of the matrix, so this is the whole
spectrum and not a family of examples.

The smallest is at $p = q = 1$ and the largest at $p = q = m$, so

$$\kappa = \frac{\sin^{2}\!\big(m\pi h/2\big)}{\sin^{2}\!\big(\pi h/2\big)}
        = \frac{\cos^{2}(\pi h/2)}{\sin^{2}(\pi h/2)} = \cot^{2}\frac{\pi h}{2}
        \approx \frac{4}{\pi^{2}h^{2}} ,$$

using $mh = 1 - h$. The condition number grows like $h^{-2}$, so halving the mesh multiplies it by
four. Everything about solving the system, direct or iterative, follows from that one fact.

### 1.3 Why conjugate gradients finishes in one step on the single mode problem

Conjugate gradients builds its approximation in the Krylov space
$\mathcal{K}_k = \operatorname{span}\{b, Ab, \dots, A^{k-1}b\}$ and, in exact arithmetic, finds the
best approximation in it at every step. If $b$ is an eigenvector of $A$ with eigenvalue $\lambda$,
then $Ab = \lambda b$, so every Krylov space is the same one dimensional space $\operatorname{span}\{b\}$.
The exact solution $A^{-1}b = b/\lambda$ lies in it. So the best approximation in
$\mathcal{K}_1$ is already exact, and the method stops after one step.

The manufactured problem $u = \sin(\pi x)\sin(\pi y)$ is precisely this case. Its discrete right
hand side is $2\pi^2$ times the $(1,1)$ sine mode plus a quadrature error of size $O(h^2)$ that is
itself smooth, and CG finishes in one or two iterations at every grid size. Any scaling law
measured on it, "CG takes $O(1)$ iterations, independent of $h$", is a measurement of that
degeneracy and not of the method.

This is why the lesson carries `mixed_modes_problem` as well: multiplying by $e^{x}$ spreads the
right hand side across the spectrum without changing the boundary values, and only then does the
iteration count start to grow the way the theory says it must.

### 1.4 The optimal relaxation factor, and what happens at $h = 1/2$

For a consistently ordered matrix with Jacobi spectral radius $\rho$,

$$\omega^{*} = \frac{2}{1 + \sqrt{1 - \rho^{2}}}, \qquad
  \rho_{\text{SOR}}(\omega^{*}) = \omega^{*} - 1 .$$

Here $\rho = \cos(\pi h)$, so $\sqrt{1 - \rho^2} = \sin(\pi h)$ and

$$\omega^{*} = \frac{2}{1 + \sin(\pi h)}, \qquad \rho^{*} = \frac{1 - \sin(\pi h)}{1 + \sin(\pi h)} .$$

At $h = 1/2$ there is exactly one interior point. Then $\rho = \cos(\pi/2) = 0$, so
$\omega^{*} = 1$: SOR reduces to Gauss-Seidel, and $\rho^{*} = 0$. Both are right and both are
trivial. With a single unknown the "iteration" is $u \leftarrow (h^2 f + \text{boundary})/4$, which
is the exact solution, so every method converges in one step and there is nothing to accelerate.
The formula reports that honestly rather than breaking down.

The useful reading is the other end. As $h \to 0$, $\omega^{*} \to 2$ and
$\rho^{*} \approx 1 - 2\pi h$, against $\rho_{\text{GS}} = \cos^2(\pi h) \approx 1 - \pi^2 h^2$.
The number of iterations goes from $O(h^{-2})$ to $O(h^{-1})$, a square root, and that is the whole
point of the parameter.

### 1.5 The diagonal neighbours, which do not cancel because they were never there

The exercise asks why the diagonal entries cancel. The measurement says something sharper: in the
right triangle mesh they are **individually zero in every element**, so there is nothing to cancel.

Take the unit square split along the diagonal from $(0,0)$ to $(1,1)$. In the lower triangle
$(0,0), (1,0), (1,1)$, the hat function at $(0,0)$ has gradient perpendicular to the opposite edge,
which is the vertical segment from $(1,0)$ to $(1,1)$, so $\nabla\phi_{(0,0)} = (-1, 0)$. The hat
at $(1,1)$ has gradient perpendicular to the opposite edge $(0,0)$ to $(1,0)$, which is horizontal,
so $\nabla\phi_{(1,1)} = (0, 1)$. Their dot product is zero, and since the element stiffness entry
is $\text{area}\;\nabla\phi_i \cdot \nabla\phi_j$, the entry is zero.

The reason is geometric and worth stating plainly: **the two vertices at the ends of the hypotenuse
of a right triangle have perpendicular basis gradients**, because each gradient is perpendicular to
the opposite side and the two opposite sides are the two legs, which are perpendicular. Both
triangles sharing the mesh diagonal are right triangles, so both contribute zero.

The other diagonal, from $(1,0)$ to $(0,1)$, is not an edge of any triangle in this mesh, so those
two nodes never appear in a common element and their entry is zero for the ordinary reason.

```python
import numpy as np
from nalib import elliptic as el

lower = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
upper = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
kl = el.triangle_stiffness(lower)
ku = el.triangle_stiffness(upper)
print("lower element matrix")
print(np.array2string(kl, precision=4))
print("upper element matrix")
print(np.array2string(ku, precision=4))
print(f"entry between the hypotenuse ends: lower {kl[0, 2]:+.4f}, upper {ku[0, 1]:+.4f}")

mesh = el.right_triangle_mesh(5)
assembled = el.assemble_fem(mesh)
side = int(round(np.sqrt(assembled.shape[0])))
middle = (side // 2) * side + side // 2
patch = assembled[middle].reshape(side, side)[side // 2 - 1:side // 2 + 2,
                                              side // 2 - 1:side // 2 + 2]
print("assembled row at an interior node")
print(np.array2string(patch, precision=4))
```

Every element matrix has the value $-1/2$ on the two legs and $0$ on the hypotenuse, and the
assembled interior row is exactly $[[0,-1,0],[-1,4,-1],[0,-1,0]]$, the five point stencil. The
$h^2$ cancels because the element matrix of a linear triangle is scale invariant in two dimensions,
which is the next exercise.

### 2.1 The two dimensional eigenvalues from the one dimensional ones

The five point matrix on a square grid is a **Kronecker sum**. Write $T$ for the one dimensional
second difference matrix on $m$ points,

$$T = \frac{1}{h^{2}}\operatorname{tridiag}(-1, 2, -1), \qquad
  T w_p = \mu_p w_p, \quad \mu_p = \frac{4}{h^{2}}\sin^{2}\frac{p\pi h}{2},
  \quad w_p(i) = \sin(p \pi i h) .$$

Ordering the unknowns row by row, the five point matrix is

$$A = T \otimes I + I \otimes T ,$$

because the $x$ differences act on the row index only and the $y$ differences on the column index
only, and the two act on separate factors. For a Kronecker sum the eigenvectors are the tensor
products and the eigenvalues are the sums:

$$A\,(w_p \otimes w_q) = (T w_p) \otimes w_q + w_p \otimes (T w_q) = (\mu_p + \mu_q)(w_p \otimes w_q) .$$

So

$$\lambda_{p,q} = \mu_p + \mu_q = \frac{4}{h^{2}}
  \left[\sin^{2}\frac{p\pi h}{2} + \sin^{2}\frac{q\pi h}{2}\right] ,$$

with $v_{p,q}(i,j) = \sin(p\pi i h)\sin(q \pi j h)$, which is what 1.2 quoted. The $m^2$ pairs
$(p,q)$ give $m^2$ independent eigenvectors, so this is the complete spectrum.

Two consequences are worth naming. The matrix is symmetric with a full set of orthogonal
eigenvectors, so it is **normal**, and the spectral radius really does control everything, unlike
the convection-diffusion matrix of lesson 79. And the construction works in any number of
dimensions: in three dimensions the eigenvalues are $\mu_p + \mu_q + \mu_r$, and the condition
number is still $\cot^2(\pi h/2)$ because the extremes scale together.

```python
import numpy as np
from nalib import elliptic as el

for unknowns in (4, 7, 12):
    step = 1.0 / (unknowns + 1)
    matrix = el.five_point_matrix(unknowns, step)
    computed = np.sort(np.linalg.eigvalsh(matrix))
    grid = np.arange(1, unknowns + 1)
    px, qy = np.meshgrid(grid, grid, indexing="ij")
    predicted = np.sort((4.0 / step ** 2 * (np.sin(px * np.pi * step / 2.0) ** 2
                                            + np.sin(qy * np.pi * step / 2.0) ** 2)).ravel())
    gap = float(np.max(np.abs(computed - predicted)) / np.max(np.abs(computed)))
    print(f"unknowns per side {unknowns:>3}: {computed.size} eigenvalues, "
          f"largest relative gap {gap:.3e}")
```

The formula reproduces the whole spectrum to rounding at every size, which is the check that the
Kronecker argument used the right ordering.

### 2.2 The condition number, its limit, and why the limit is too big

From 2.1 the extreme eigenvalues are at $p = q = 1$ and $p = q = m$, and $mh = 1 - h$, so

$$\kappa = \frac{2\sin^{2}(m\pi h/2)}{2\sin^{2}(\pi h/2)}
        = \frac{\sin^{2}\big(\tfrac{\pi}{2}(1-h)\big)}{\sin^{2}(\pi h/2)}
        = \frac{\cos^{2}(\pi h/2)}{\sin^{2}(\pi h /2)} = \cot^{2}\frac{\pi h}{2} .$$

The factor 2 cancels, which is why the answer is the same in one, two and three dimensions.

For the limit, expand the cotangent. Using $\cot x = 1/x - x/3 - x^{3}/45 - \cdots$,

$$\cot^{2}x = \frac{1}{x^{2}} - \frac{2}{3} - \frac{x^{2}}{15} - \cdots ,$$

so with $x = \pi h/2$,

$$\kappa = \frac{4}{\pi^{2}h^{2}} - \frac{2}{3} - O(h^{2}) .$$

The leading term is the familiar $4/(\pi^2 h^2) \approx 0.405\,h^{-2}$. The next term is
$-2/3$, a **negative** constant, so the leading term alone is an **overestimate** of the true
condition number, by about $0.667$ at every mesh size. That is a small absolute amount, and it
matters most on coarse grids where $\kappa$ itself is small: at $h = 1/8$ the exact value is
$25.274$ and the limit says $25.938$, an error of $2.6\%$; by $h = 1/32$ the error is $0.16\%$.

```python
import math

import numpy as np
from nalib import elliptic as el

print(f"{'unknowns':>10}{'h':>9}{'exact kappa':>14}{'cot^2':>12}{'4/(pi h)^2':>13}"
      f"{'limit - exact':>15}")
for unknowns in (3, 7, 15, 31):
    step = 1.0 / (unknowns + 1)
    matrix = el.five_point_matrix(unknowns, step)
    exact = float(np.linalg.cond(matrix))
    closed = 1.0 / math.tan(math.pi * step / 2.0) ** 2
    limit = 4.0 / (math.pi * step) ** 2
    print(f"{unknowns:>10}{step:>9.5f}{exact:>14.4f}{closed:>12.4f}{limit:>13.4f}"
          f"{limit - exact:>15.4f}")
```

The closed form matches the computed condition number to rounding, and the last column climbs
$0.6561, 0.6641, 0.6660, 0.6665$ toward $2/3 = 0.6667$, which is the $-2/3$ term appearing exactly
as the expansion predicts. Two things confirm the reading. The offset does **not** shrink, which is
the signature of a term that is $O(1)$ rather than $O(h^2)$: an $h^2$ term would have to shrink by
four each row. And the small remaining gap to $2/3$ does shrink by about four each row
($0.0106, 0.0026, 0.0007, 0.0002$), which is the next term, $-x^2/15$, behaving as it should.

### 2.3 The Jacobi radius and the optimal factor

The Jacobi iteration matrix is $J = I - D^{-1}A$. Here the diagonal of $A$ is the constant
$4/h^2$, so $D^{-1} = (h^2/4) I$ and $J = I - (h^2/4)A$ is a function of $A$ alone. It therefore
has the same eigenvectors, with eigenvalues

$$1 - \frac{h^{2}}{4}\lambda_{p,q}
  = 1 - \sin^{2}\frac{p\pi h}{2} - \sin^{2}\frac{q\pi h}{2}
  = \frac{\cos(p \pi h) + \cos(q \pi h)}{2} ,$$

using $\cos 2\theta = 1 - 2\sin^2\theta$ twice. The extremes are at $p = q = 1$ and $p = q = m$,
giving $\cos(\pi h)$ and $-\cos(\pi h)$, so

$$\rho_{\text{Jacobi}} = \cos(\pi h) .$$

Part 4's formula for a consistently ordered matrix then gives

$$\omega^{*} = \frac{2}{1 + \sqrt{1 - \cos^{2}(\pi h)}} = \frac{2}{1 + \sin(\pi h)}, \qquad
  \rho_{\text{SOR}}(\omega^{*}) = \omega^{*} - 1 = \frac{1 - \sin(\pi h)}{1 + \sin(\pi h)} .$$

The five point matrix on this ordering is consistently ordered, which is what lets the formula
apply: the red-black permutation puts it in the block form the theory requires, and exercise 4.3
shows the permutation does not change the Gauss-Seidel rate.

```python
import math

import numpy as np
from nalib import elliptic as el
from nalib.iterative import iteration_matrix, spectral_radius

print(f"{'unknowns':>10}{'h':>9}{'rho Jacobi':>13}{'cos(pi h)':>12}"
      f"{'rho GS':>10}{'rho J^2':>10}{'omega*':>9}")
for unknowns in (3, 7, 15):
    step = 1.0 / (unknowns + 1)
    matrix = el.five_point_matrix(unknowns, step)
    jacobi = spectral_radius(iteration_matrix(matrix, "jacobi"))
    gauss = spectral_radius(iteration_matrix(matrix, "gauss-seidel"))
    best = 2.0 / (1.0 + math.sin(math.pi * step))
    print(f"{unknowns:>10}{step:>9.5f}{jacobi:>13.6f}{math.cos(math.pi * step):>12.6f}"
          f"{gauss:>10.6f}{jacobi ** 2:>10.6f}{best:>9.6f}")
```

Three predictions come out at once and all three hold to rounding: $\rho_J = \cos(\pi h)$, the
Gauss-Seidel radius is exactly $\rho_J^2$ as consistent ordering requires, and $\omega^*$ follows
from the formula. Exercise 4.3 checks the last one against a direct search.

### 2.4 The element stiffness matrix of a linear triangle

On a triangle with vertices $(x_1,y_1), (x_2,y_2), (x_3,y_3)$ the P1 basis function $\phi_i$ is the
unique affine function equal to 1 at vertex $i$ and 0 at the other two. Affine means its gradient
is **constant on the element**, so the integral

$$K_{ij} = \int_{\tau} \nabla\phi_i \cdot \nabla\phi_j \, dA
        = |\tau| \; \nabla\phi_i \cdot \nabla\phi_j$$

needs no quadrature at all, only the area and the gradients.

For the gradients, write $\phi_i = (a_i + b_i x + c_i y)/(2|\tau|)$. Requiring $\phi_i$ to vanish
at the other two vertices makes it proportional to the signed distance from the opposite edge, so

$$b_i = y_j - y_k, \qquad c_i = x_k - x_j ,$$

with $(i,j,k)$ a cyclic permutation, and

$$2|\tau| = (x_2 - x_1)(y_3 - y_1) - (x_3 - x_1)(y_2 - y_1) .$$

Then $\nabla\phi_i = (b_i, c_i)/(2|\tau|)$ and

$$\boxed{\;K_{ij} = \frac{b_i b_j + c_i c_j}{4\,|\tau|}\;}$$

Three properties fall out and all three are worth checking. Each row sums to zero, because
$\sum_i \phi_i \equiv 1$ so $\sum_i \nabla \phi_i = 0$: a constant function has no energy, which is
the discrete statement that the pure Neumann problem is singular. The matrix is symmetric and
positive semidefinite, with the constants as its only null space. And it is **invariant under
scaling the triangle**, because $b_i, c_i$ scale like a length and $|\tau|$ like a length squared,
so the quotient is dimensionless. That last one is why the $h^2$ cancelled in 1.5 and why the
assembled matrix is the five point stencil without any $h$ in it.

```python
import numpy as np
from nalib import elliptic as el

rng = np.random.default_rng(42)


def by_hand(vertices):
    xs, ys = vertices[:, 0], vertices[:, 1]
    count = vertices.shape[0]
    order = np.arange(count)
    nxt, prv = (order + 1) % count, (order - 1) % count
    b = ys[nxt] - ys[prv]
    c = xs[prv] - xs[nxt]
    twice = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
    return (np.outer(b, b) + np.outer(c, c)) / (2.0 * abs(twice))


print(f"{'trial':>7}{'gap to library':>17}{'row sums':>12}{'smallest eigenvalue':>22}"
      f"{'scale invariant':>18}")
for trial in range(5):
    vertices = rng.normal(size=(3, 2))
    mine = by_hand(vertices)
    theirs = el.triangle_stiffness(vertices)
    scaled = el.triangle_stiffness(vertices * (1.0 + 10.0 * rng.random()))
    print(f"{trial:>7}{float(np.max(np.abs(mine - theirs))):>17.3e}"
          f"{float(np.max(np.abs(mine.sum(axis=1)))):>12.3e}"
          f"{float(np.min(np.linalg.eigvalsh(theirs))):>22.3e}"
          f"{float(np.max(np.abs(scaled - theirs))):>18.3e}")
```

On random triangles the closed form agrees with the library to rounding, the rows sum to zero, the
smallest eigenvalue is zero to within rounding, with sign noise at the $10^{-16}$ level because a
semidefinite matrix sits exactly on the boundary of the positive definite ones, and scaling the
triangle by a random factor changes nothing. Four independent checks of one formula.

### 2.5 The vertex rule reproduces the finite difference right hand side

The vertex rule, also called the lumped rule, approximates
$\int_{\tau} g \, dA$ by $\frac{|\tau|}{3}\sum_{\text{vertices}} g$. Applied to the load entry for
node $i$, and using $\phi_i = 1$ at its own vertex and 0 at the others,

$$F_i = \int_{\tau} f \phi_i \, dA \;\approx\; \frac{|\tau|}{3} f(x_i, y_i) .$$

Now sum over the triangles meeting at an interior node. In the right triangle mesh every interior
node belongs to **six** triangles, each of area $h^2/2$, so

$$F_i = 6 \cdot \frac{h^{2}/2}{3} f(x_i, y_i) = h^{2} f(x_i, y_i) .$$

That is exactly the finite difference right hand side. Together with 1.5, which showed the
assembled stiffness matrix is the five point stencil, the two systems are **identical, not merely
similar**: same matrix, same right hand side, same solution, to the last bit.

The count of six is where the mesh enters. A different triangulation gives a different count and
the identity breaks. On the criss-cross mesh, for instance, interior nodes have four or eight
neighbours depending on which they are, and the load is no longer uniform.

```python
import numpy as np
from nalib import elliptic as el

problem = el.mixed_modes_problem()
print(f"{'points':>8}{'h':>9}{'max |load - h^2 f|':>22}{'relative':>12}"
      f"{'triangles per node':>20}")
for points in (5, 9, 17):
    mesh = el.right_triangle_mesh(points)
    load = el.load_vector(mesh, problem["source"], rule="vertex")
    step = float(mesh["h"])
    nodes = mesh["nodes"]
    difference = problem["source"](nodes[:, 0], nodes[:, 1]) * step ** 2
    interior = el.solve_fem(problem, points)["interior"]
    gap = float(np.max(np.abs(load[interior] - difference[interior])))
    touching = np.bincount(mesh["cells"].ravel(), minlength=nodes.shape[0])
    print(f"{points:>8}{step:>9.5f}{gap:>22.3e}"
          f"{gap / float(np.max(np.abs(difference[interior]))):>12.3e}"
          f"{int(np.min(touching[interior])):>20}")
```

The relative gap is $2\times10^{-16}$ at every size, which is rounding, and every interior node
does belong to exactly six triangles. The identity is exact.

### 3.1 The Mehrstellen solve, and fourth order for the price of second

Lesson 77 measured the stencils in isolation. Here they are used to solve, which is the test that
matters, because a stencil that is fourth order pointwise can still give a second order solution if
its matrix is badly conditioned or its boundary treatment is worse than its interior.

For $-\Delta u = f$ the Mehrstellen system is

$$\frac{1}{6h^{2}}\begin{bmatrix}-1 & -4 & -1\\ -4 & 20 & -4\\ -1 & -4 & -1\end{bmatrix} u
  \;=\; f + \frac{h^{2}}{12}\,\Delta_h f ,$$

where $\Delta_h$ on the right is the ordinary five point Laplacian applied to the **known** source.
It only needs to be second order, because it multiplies $h^2$.

```python
import numpy as np
from nalib import elliptic as el

problem = el.mixed_modes_problem()
five = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
nine = np.array([[-1.0, -4.0, -1.0], [-4.0, 20.0, -4.0], [-1.0, -4.0, -1.0]]) / 6.0


def solve_with(stencil, points, corrected):
    line = np.linspace(problem["a"], problem["b"], points)
    step = float(line[1] - line[0])
    gx, gy = np.meshgrid(line, line, indexing="ij")
    exact = problem["exact"](gx, gy)
    source = problem["source"](gx, gy)
    right = source.copy()
    if corrected:
        laplacian = np.zeros_like(source)
        laplacian[1:-1, 1:-1] = (source[2:, 1:-1] + source[:-2, 1:-1] + source[1:-1, 2:]
                                 + source[1:-1, :-2] - 4.0 * source[1:-1, 1:-1]) / step ** 2
        right = right + step ** 2 / 12.0 * laplacian
    inner = points - 2
    index = -np.ones((points, points), dtype=int)
    index[1:-1, 1:-1] = np.arange(inner * inner).reshape(inner, inner)
    reach = stencil.shape[0] // 2
    matrix = np.zeros((inner * inner, inner * inner))
    vector = np.zeros(inner * inner)
    for i in range(1, points - 1):
        for j in range(1, points - 1):
            row = index[i, j]
            vector[row] = right[i, j]
            for p in range(stencil.shape[0]):
                for q in range(stencil.shape[1]):
                    weight = stencil[p, q] / step ** 2
                    if weight == 0.0:
                        continue
                    ii, jj = i + p - reach, j + q - reach
                    if index[ii, jj] >= 0:
                        matrix[row, index[ii, jj]] += weight
                    else:
                        vector[row] -= weight * exact[ii, jj]
    got = np.linalg.solve(matrix, vector)
    return step, float(np.max(np.abs(got - exact[1:-1, 1:-1].ravel())))


steps, errors = [], {"five point": [], "nine point": [], "Mehrstellen": []}
print(f"{'points':>8}{'h':>10}{'five point':>13}{'nine point':>13}{'Mehrstellen':>14}")
for points in (9, 17, 33, 65):
    row = []
    for name, stencil, corrected in (("five point", five, False),
                                     ("nine point", nine, False),
                                     ("Mehrstellen", nine, True)):
        step, err = solve_with(stencil, points, corrected)
        errors[name].append(err)
        row.append(err)
    steps.append(step)
    print(f"{points:>8}{step:>10.5f}{row[0]:>13.3e}{row[1]:>13.3e}{row[2]:>14.3e}")
steps = np.asarray(steps)
for name, values in errors.items():
    fit = float(np.polyfit(np.log(steps), np.log(np.asarray(values)), 1)[0])
    print(f"{name:>12} order {fit:.4f}")
```

The measured orders are $1.9992$, $2.0021$ and $3.9927$.

Three readings, and the middle one is the important one.

**The correction is what does the work, not the extra points.** The uncorrected nine point rule is
second order here, at $2.0021$, and in fact **worse than the five point rule** in absolute terms:
$6.6\times10^{-4}$ against $3.0\times10^{-4}$ at the finest grid, a factor of $2.2$ the wrong way.
Adding the corner points on their own buys nothing. This matches lesson 77, where the nine point
rule was second order in general and only fourth order on harmonic functions, and here the solution
is not harmonic.

**The correction buys three digits.** At $h = 1/64$ Mehrstellen gives $4.2\times10^{-8}$ against
the five point rule's $3.0\times10^{-4}$, a factor of $7000$, for a matrix with the same bandwidth
and a right hand side that costs one extra five point sweep to build. The five point rule would
need $h$ around $1/5000$ to match it, which is $2.5\times10^{7}$ unknowns against $4000$.

**The order is $3.99$, not $4.5$ or $3.5$.** That the fit lands within $0.008$ of 4 over four
grids, each error a clean factor of 16 below the last, is the check that the boundary treatment did
not spoil the interior order. It could easily have: the Dirichlet values here are exact, but a
second order boundary approximation would have pulled the whole thing back toward two.
### 3.2 The cooling fin, and a heat balance that is exact rather than convergent

This is the first problem in the part whose boundary conditions are not all Dirichlet. The equation
is

$$k\,\Delta u = \frac{2H}{t}\,(u - u_{\infty}) ,$$

so moving everything to the left gives $-\Delta u + c\,(u - u_{\infty}) = 0$ with
$c = 2H/(kt) = 0.059524$. Three edges are insulated and part of the fourth carries a flux
$q = P/(kt\,L_c) = 14.880952$.

Both conditions are Neumann and both are handled with the ghost point of lesson 77. For an
insulated edge $\partial u/\partial n = 0$ makes the ghost value equal to its mirror image, so the
outward neighbour is replaced by the inward one and that entry is **doubled**. For the flux edge
$-\partial u/\partial n = q$ makes the ghost value differ from its mirror by $2hq$, which lands in
the right hand side as $2q/h$ after dividing by $h^2$.

```python
import numpy as np
from nalib import elliptic as el

fin = el.cooling_fin()
print(f"reaction {fin['reaction']:.6f} per unit length squared, "
      f"flux {fin['flux']:.6f}, contact length {fin['contact']}")


def solve_fin(points):
    across = np.linspace(fin["a"], fin["b"], points)
    up = np.linspace(0.0, fin["height"], points)
    step = float(across[1] - across[0])
    if abs(step - float(up[1] - up[0])) > 1e-12:
        raise ValueError("this assembly assumes a square cell")
    size = points * points
    matrix = np.zeros((size, size))
    vector = np.zeros(size)
    half = fin["contact"] / 2.0
    for i in range(points):
        for j in range(points):
            row = i * points + j
            matrix[row, row] = 4.0 / step ** 2 + fin["reaction"]
            vector[row] = fin["reaction"] * fin["ambient"]
            for di, dj, side in ((-1, 0, "left"), (1, 0, "right"),
                                 (0, -1, "bottom"), (0, 1, "top")):
                ii, jj = i + di, j + dj
                if 0 <= ii < points and 0 <= jj < points:
                    matrix[row, ii * points + jj] += -1.0 / step ** 2
                else:
                    matrix[row, (i - di) * points + (j - dj)] += -1.0 / step ** 2
                    if side == "left" and abs(up[j] - fin["height"] / 2.0) <= half:
                        vector[row] += 2.0 * fin["flux"] / step
    return across, up, np.linalg.solve(matrix, vector).reshape(points, points), step


print(f"{'points':>8}{'h':>9}{'hottest':>10}{'tip mean':>11}{'crude gap':>12}"
      f"{'trapezoid gap':>16}")
steps, crude, trapezoid = [], [], []
for points in (11, 21, 41, 81):
    across, up, field, step = solve_fin(points)
    excess = field - fin["ambient"]
    weight = np.ones_like(excess)
    weight[0, :] *= 0.5
    weight[-1, :] *= 0.5
    weight[:, 0] *= 0.5
    weight[:, -1] *= 0.5
    lost_crude = 2.0 * fin["coefficient"] * float(np.sum(excess)) * step ** 2
    lost_trapezoid = 2.0 * fin["coefficient"] * float(np.sum(weight * excess)) * step ** 2
    gap_crude = abs(fin["power"] - lost_crude) / fin["power"]
    gap_trapezoid = abs(fin["power"] - lost_trapezoid) / fin["power"]
    steps.append(step)
    crude.append(gap_crude)
    trapezoid.append(gap_trapezoid)
    print(f"{points:>8}{step:>9.4f}{float(np.max(field)):>10.4f}"
          f"{float(np.mean(field[-1, :])):>11.4f}{gap_crude:>12.3e}{gap_trapezoid:>16.3e}")
steps = np.asarray(steps)
print(f"crude sum order {float(np.polyfit(np.log(steps), np.log(np.asarray(crude)), 1)[0]):.4f}")
```

The hottest point settles at $154.766$ and the mean tip temperature at $140.174$, both converging
from below and both changing by less than $0.001$ over the last halving. The interesting column is
the last two.

**The crude sum is first order and it is the quadrature, not the solution.** Adding
$\sum (u - u_\infty) h^2$ over every node overcounts the boundary, because the cell around a
boundary node is half a cell and the cell around a corner node is a quarter. That surplus is a
strip of width $h/2$ around the perimeter, so the error is $O(h)$, and the fit says $1.02$. It
would be easy to read that column as "the scheme is only first order accurate" and it is nothing of
the sort.

**With the trapezoid weights the balance is exact.** The gap drops to $10^{-14}$ at the coarsest
grid, on **eleven points a side**, and stays at rounding all the way down. It is not converging: it
is an identity. Sum the discrete equations with weights $1$ inside, $1/2$ on the edges and $1/4$ at
the corners, multiply by $h^2$, and every interior difference appears twice with opposite signs and
telescopes away. What survives is exactly the flux entering through the contact strip on one side
and $2H \int (u - u_\infty)\,dA$ on the other. The ghost point treatment is what makes the
telescoping work at the edges.

That is the real lesson of a conservation check. **A well built scheme satisfies the balance
exactly, at every mesh size, and a check that merely converges is usually testing the quadrature
you wrapped around it.** Seeing $2.1 \times 10^{-1}$ and concluding the scheme is broken, or seeing
it halve and concluding the scheme is first order, would both be wrong.

### 3.3 Sparse against dense

The matrix has five nonzeros per row out of $(n-2)^2$ columns. At $n = 65$ that is $5$ out of
$3969$, so $99.87\%$ of the dense array is zeros being stored, multiplied and factored.

```python
import time

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from nalib import elliptic as el

problem = el.mixed_modes_problem()
print(f"{'points':>8}{'unknowns':>10}{'dense MB':>11}{'sparse MB':>11}{'dense s':>10}"
      f"{'sparse s':>10}{'memory x':>11}{'time x':>9}")
for points in (17, 33, 65):
    system = el.assemble(problem, points)
    matrix, right = system["A"], system["b"]
    dense_mb = matrix.nbytes / 1e6
    start = time.perf_counter()
    np.linalg.solve(matrix, right)
    dense_seconds = time.perf_counter() - start
    compressed = sparse.csr_matrix(matrix)
    sparse_mb = (compressed.data.nbytes + compressed.indices.nbytes
                 + compressed.indptr.nbytes) / 1e6
    start = time.perf_counter()
    sparse_linalg.spsolve(compressed.tocsc(), right)
    sparse_seconds = time.perf_counter() - start
    print(f"{points:>8}{matrix.shape[0]:>10}{dense_mb:>11.3f}{sparse_mb:>11.3f}"
          f"{dense_seconds:>10.4f}{sparse_seconds:>10.4f}"
          f"{dense_mb / sparse_mb:>11.1f}{dense_seconds / sparse_seconds:>9.1f}")
```

At $n = 17$ the sparse solve is slightly **slower**, about $0.8$ times the dense time, because $225$
unknowns is small enough that the bookkeeping costs more than the arithmetic it saves. By $n = 65$
it is roughly $130$ times faster and uses $502$ times less memory: $0.25$ MB against $126$ MB. The
memory column is deterministic; the timings move by ten percent or so from run to run and between
machines, so read them as ratios and orders, not as figures.

The gap grows because the two have different exponents. The dense solve is $O(N^3)$ in the number
of unknowns $N = (n-2)^2$, so $O(n^6)$; a good sparse ordering on a two dimensional grid is about
$O(N^{1.5})$, so $O(n^3)$. Each doubling of $n$ therefore multiplies the ratio by roughly $2^3 = 8$,
and the measured ratios, near $0.8$, $10$ and $130$, grow by about $12$ each doubling, in that
range.

The memory is the harder limit in practice. Dense storage is $8N^2$ bytes, so $n = 200$ would need
$12$ GB and $n = 500$ about $4.9$ TB, while sparse storage is linear in $N$ and $n = 500$ needs
about $15$ MB. The crossover where dense stops being possible arrives long before the crossover
where it stops being sensible.

### 3.4 Multigrid, and an iteration count that does not grow

Every method so far has an iteration count that grows with the grid: Jacobi and Gauss-Seidel like
$h^{-2}$, SOR and CG like $h^{-1}$. Multigrid is the method whose count does not grow at all,
because it attacks each frequency on the grid where that frequency is easy.

The V cycle smooths a little on the fine grid, which kills the high frequencies quickly, moves the
residual to a grid twice as coarse, where the frequencies that are left look high again, recurses,
and interpolates the correction back. The smoother here is red-black Gauss-Seidel, which exercise
4.3 shows has the same rate as the row by row version and is a great deal easier to write without
a Python loop over points.

```python
import numpy as np
from nalib import elliptic as el


def smooth(field, right, step, sweeps):
    size = field.shape[0]
    rows, columns = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for _ in range(sweeps):
        for colour in (0, 1):
            padded = np.zeros((size + 2, size + 2))
            padded[1:-1, 1:-1] = field
            neighbours = (padded[2:, 1:-1] + padded[:-2, 1:-1]
                          + padded[1:-1, 2:] + padded[1:-1, :-2])
            field = np.where((rows + columns) % 2 == colour,
                             (step ** 2 * right + neighbours) / 4.0, field)
    return field


def residual(field, right, step):
    size = field.shape[0]
    padded = np.zeros((size + 2, size + 2))
    padded[1:-1, 1:-1] = field
    return right + (padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:]
                    + padded[1:-1, :-2] - 4.0 * field) / step ** 2


def restrict(fine):
    size = fine.shape[0]
    coarse_size = (size - 1) // 2
    padded = np.zeros((size + 2, size + 2))
    padded[1:-1, 1:-1] = fine
    centres = 2 * np.arange(coarse_size) + 2
    rows, columns = np.meshgrid(centres, centres, indexing="ij")
    return (4.0 * padded[rows, columns]
            + 2.0 * (padded[rows - 1, columns] + padded[rows + 1, columns]
                     + padded[rows, columns - 1] + padded[rows, columns + 1])
            + padded[rows - 1, columns - 1] + padded[rows - 1, columns + 1]
            + padded[rows + 1, columns - 1] + padded[rows + 1, columns + 1]) / 16.0


def prolong(coarse, size):
    coarse_size = coarse.shape[0]
    padded = np.zeros((coarse_size + 2, coarse_size + 2))
    padded[1:-1, 1:-1] = coarse
    place = (np.arange(size) - 1) / 2.0
    base = np.floor(place).astype(int) + 1
    weight = place - np.floor(place)
    br, bc = np.meshgrid(base, base, indexing="ij")
    wr, wc = np.meshgrid(weight, weight, indexing="ij")
    return ((1 - wr) * (1 - wc) * padded[br, bc] + wr * (1 - wc) * padded[br + 1, bc]
            + (1 - wr) * wc * padded[br, bc + 1] + wr * wc * padded[br + 1, bc + 1])


def v_cycle(field, right, step, down=2, up=2):
    size = field.shape[0]
    if size <= 3:
        return smooth(field, right, step, 30)
    field = smooth(field, right, step, down)
    coarse_shape = ((size - 1) // 2,) * 2
    correction = v_cycle(np.zeros(coarse_shape), restrict(residual(field, right, step)),
                         2.0 * step, down, up)
    return smooth(field + prolong(correction, size), right, step, up)


problem = el.mixed_modes_problem()
print(f"{'unknowns a side':>17}{'N':>9}{'V cycles':>10}{'residual':>12}{'error':>12}")
for unknowns in (7, 15, 31, 63, 127):
    step = 1.0 / (unknowns + 1)
    line = np.linspace(problem["a"], problem["b"], unknowns + 2)[1:-1]
    gx, gy = np.meshgrid(line, line, indexing="ij")
    source = problem["source"](gx, gy)
    truth = problem["exact"](gx, gy)
    field = np.zeros((unknowns, unknowns))
    scale = float(np.max(np.abs(source)))
    for cycles in range(1, 61):
        field = v_cycle(field, source, step)
        relative = float(np.max(np.abs(residual(field, source, step)))) / scale
        if relative < 1e-10:
            break
    print(f"{unknowns:>17}{unknowns * unknowns:>9}{cycles:>10}{relative:>12.2e}"
          f"{float(np.max(np.abs(field - truth))):>12.3e}")
```

The counts are $8, 8, 9, 9, 9$ while $N$ goes from $49$ to $16129$, a factor of $329$. That is the
claim, measured: **the iteration count is independent of $N$.** Compare CG on the same problem in
4.1, which goes $7, 17, 42, 92$, and Gauss-Seidel, which would go roughly $10^2, 10^3, 10^4$.

Two checks that the answer is real and not an artefact of a loose stopping test.

**The errors match the direct solve exactly.** They read
$1.892\times10^{-2},\ 4.715\times10^{-3},\ 1.183\times10^{-3},\ 2.958\times10^{-4}$, which are the
same digits as the five point column of exercise 3.1, where the system was solved by Gaussian
elimination. Multigrid is finding the same discrete solution, not a different one, and what is left
is the discretization error, not the iteration error.

**The work per cycle is linear.** A V cycle costs about $\frac{4}{3}$ times a fine grid sweep,
because the grids shrink by four each level and $1 + 1/4 + 1/16 + \cdots = 4/3$. Constant cycles at
linear cost per cycle is the $O(N)$ solver, which is the best any method can be, since reading the
right hand side already costs $O(N)$.

### 3.5 Finite elements on an L shaped domain

The five point difference method needs a grid that fits the domain. On the L shape it can be made
to fit, since the corners are on grid lines, but nothing about the difference stencil knows the
reentrant corner is there. Finite elements need only a triangulation, which is why this is the
exercise where they earn their keep.

The domain is $[-1,1]^2$ with the quadrant $x > 0,\, y < 0$ removed, so the interior angle at the
origin is $3\pi/2$. The function

$$u = r^{2/3}\sin\frac{2\theta}{3}$$

is harmonic, so the source is zero and all the data is on the boundary. It vanishes on both faces
of the reentrant corner, $\theta = 0$ and $\theta = 3\pi/2$, which is what makes it the natural
singular mode there.

```python
import math

import numpy as np
from nalib import elliptic as el


def l_shaped_mesh(refine, grading=1.0):
    step = 1.0 / refine
    nodes, index = [], {}
    for i in range(2 * refine + 1):
        for j in range(2 * refine + 1):
            x, y = -1.0 + i * step, -1.0 + j * step
            if x > 1e-12 and y < -1e-12:
                continue
            index[(i, j)] = len(nodes)
            nodes.append((math.copysign(abs(x) ** grading, x),
                          math.copysign(abs(y) ** grading, y)))
    cells = []
    for i in range(2 * refine):
        for j in range(2 * refine):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if not all(corner in index for corner in corners):
                continue
            a, b, c, d = (index[corner] for corner in corners)
            cells.append((a, b, c))
            cells.append((a, c, d))
    return {"nodes": np.asarray(nodes), "cells": np.asarray(cells, dtype=int), "h": step}


def corner_mode(points):
    radius = np.hypot(points[:, 0], points[:, 1])
    angle = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    return np.power(np.maximum(radius, np.finfo(float).tiny), 2.0 / 3.0) * np.sin(2.0 * angle / 3.0)


def solve_l_shaped(refine, grading=1.0):
    mesh = l_shaped_mesh(refine, grading)
    nodes = mesh["nodes"]
    stiffness = el.assemble_fem(mesh)
    on_boundary = []
    for k, (x, y) in enumerate(nodes):
        outer = abs(abs(x) - 1.0) < 1e-12 or abs(abs(y) - 1.0) < 1e-12
        cut = (abs(x) < 1e-12 and y < 1e-12) or (abs(y) < 1e-12 and x > -1e-12)
        if outer or cut:
            on_boundary.append(k)
    on_boundary = np.asarray(sorted(set(on_boundary)))
    inside = np.setdiff1d(np.arange(nodes.shape[0]), on_boundary)
    known = corner_mode(nodes[on_boundary])
    field = np.linalg.solve(stiffness[np.ix_(inside, inside)],
                            -stiffness[np.ix_(inside, on_boundary)] @ known)
    truth = corner_mode(nodes[inside])
    areas = np.zeros(nodes.shape[0])
    for cell in mesh["cells"]:
        corners = nodes[cell]
        twice = ((corners[1, 0] - corners[0, 0]) * (corners[2, 1] - corners[0, 1])
                 - (corners[2, 0] - corners[0, 0]) * (corners[1, 1] - corners[0, 1]))
        areas[cell] += abs(twice) / 6.0
    return (mesh["h"], float(np.max(np.abs(field - truth))),
            float(np.sqrt(np.sum(areas[inside] * (field - truth) ** 2))))


print(f"{'h':>10}{'nodes':>9}{'max error':>13}{'L2 error':>13}")
steps, biggest, in_l2 = [], [], []
for refine in (4, 8, 16, 32):
    step, worst, energy = solve_l_shaped(refine)
    steps.append(step)
    biggest.append(worst)
    in_l2.append(energy)
    print(f"{step:>10.5f}{l_shaped_mesh(refine)['nodes'].shape[0]:>9}"
          f"{worst:>13.3e}{energy:>13.3e}")
steps = np.asarray(steps)
print(f"max order {float(np.polyfit(np.log(steps), np.log(np.asarray(biggest)), 1)[0]):.4f}, "
      f"L2 order {float(np.polyfit(np.log(steps), np.log(np.asarray(in_l2)), 1)[0]):.4f}")
```

The measured orders are $0.6124$ in the maximum norm and $1.2253$ in $L^2$, against the $2$ that
the same elements give on a square. The prediction is exercise 5.3: the corner exponent is
$\alpha = \pi/\omega = 2/3$ for an interior angle $\omega = 3\pi/2$, which caps the achievable
order at $\alpha$ in the energy norm and $2\alpha$ in $L^2$, that is $0.667$ and $1.333$. The
measurement sits just under both, which is what a fit over four grids gives when the constant is
still settling.

Three things are worth separating here.

**The loss is in the solution, not the method.** Nothing is wrong with the elements. The true
solution has an unbounded gradient at the origin, $|\nabla u| \sim r^{-1/3}$, so it is not in
$H^2$, and the interpolation estimate that gives second order assumes it is. No method that uses
piecewise polynomials on a quasi-uniform mesh can do better on this function.

**It is global, not local.** The error does not stay near the corner. Because the discrete problem
is coupled, a bad approximation at the origin pollutes the solution across the whole domain, which
is why the $L^2$ order drops too and not just the pointwise one at the corner.

**A difference method would have been silently worse.** The five point stencil would still produce
a number at every grid point, with no indication that the order has fallen from $2$ to $2/3$, and
the usual refinement check would have shown convergence, just slower. This is why the order should
be measured and not assumed.

### 4.1 The conjugate gradient count against $\sqrt{\kappa}$

The classical bound is

$$\frac{\lVert e_k \rVert_A}{\lVert e_0 \rVert_A}
   \le 2\left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^{k} ,$$

which reaches a tolerance $\varepsilon$ after about $\tfrac12 \sqrt\kappa \ln(2/\varepsilon)$
iterations. With $\kappa = \cot^2(\pi h/2) \approx 4/(\pi h)^2$ this says the count should grow
like $h^{-1}$.

The measurement has to use `mixed_modes_problem`, because on the single mode problem CG converges
in one step at every size, as 1.3 explained.

```python
import math

import numpy as np
from nalib import elliptic as el

problem = el.mixed_modes_problem()
print(f"{'unknowns':>10}{'N':>8}{'kappa':>12}{'sqrt kappa':>13}{'CG':>5}"
      f"{'CG / sqrt kappa':>18}{'bound':>9}")
sizes, counts, roots = [], [], []
tolerance = 1e-10
for unknowns in (7, 15, 31, 63):
    step = 1.0 / (unknowns + 1)
    matrix = el.five_point_matrix(unknowns, step)
    condition = float(np.linalg.cond(matrix))
    run = el.solve(problem, unknowns + 2, method="cg", tol=tolerance, keep_history=False)
    bound = 0.5 * math.sqrt(condition) * math.log(2.0 / tolerance)
    sizes.append(1.0 / step)
    counts.append(run["iterations"])
    roots.append(math.sqrt(condition))
    print(f"{unknowns:>10}{unknowns ** 2:>8}{condition:>12.2f}{math.sqrt(condition):>13.3f}"
          f"{run['iterations']:>5}{run['iterations'] / math.sqrt(condition):>18.4f}"
          f"{bound:>9.0f}")
counts = np.asarray(counts, dtype=float)
print(f"fit against 1/h        {float(np.polyfit(np.log(np.asarray(sizes)), np.log(counts), 1)[0]):.4f}")
print(f"fit against sqrt kappa {float(np.polyfit(np.log(np.asarray(roots)), np.log(counts), 1)[0]):.4f}")
```

The counts are $7, 17, 42, 92$ against $\sqrt\kappa$ of $5.0, 10.2, 20.4, 40.7$.

**The bound holds at every size, with room.** The predicted counts are $60, 120, 241, 483$ against
the measured $7, 17, 42, 92$: a factor of $8.5$ of slack at the coarsest grid, falling to $5.3$ at
the finest. The bound is an upper bound and it is doing its job, but it is not tight.

**The constant is not constant.** The ratio $\text{CG}/\sqrt\kappa$ drifts up, $1.39, 1.67, 2.06,
2.26$. Fitting the count against $1/h$ gives $1.24$, not $1.00$, so over this range CG is growing a
little faster than $\sqrt\kappa$. That is not a contradiction. The bound comes from a Chebyshev
polynomial that only sees the two extreme eigenvalues, and CG actually sees the whole spectrum, so
the real rate depends on how the right hand side is spread across it, which changes as the grid is
refined. Reporting "the constant in the bound" as a single number would have been a fabrication:
the honest answer is that the measured constant runs from $1.39$ to $2.26$ over four grids and has
not settled.

**And the qualitative claim is what survives.** CG is $O(h^{-1})$ where Gauss-Seidel is $O(h^{-2})$,
which is the square root the theory promises, and multigrid in 3.4 is $O(1)$, which is better than
both.

### 4.2 The preconditioners, and which of them is worth it

Three preconditioners from lesson 24, on the same matrix and the same right hand side.

```python
import time

import numpy as np
from nalib import elliptic as el
from nalib.krylov import (conjugate_gradient, incomplete_cholesky,
                          jacobi_preconditioner, ssor_preconditioner)

problem = el.mixed_modes_problem()
print(f"{'unknowns':>10}{'method':>9}{'iters':>7}{'setup s':>10}{'solve s':>10}"
      f"{'total s':>10}{'diagonal constant':>20}")
for unknowns in (15, 31, 63):
    system = el.assemble(problem, unknowns + 2)
    matrix, right = system["A"], system["b"]
    diagonal = np.diag(matrix)
    constant = bool(np.allclose(diagonal, diagonal[0]))
    builders = {"none": lambda: None,
                "jacobi": lambda: jacobi_preconditioner(matrix),
                "ssor": lambda: ssor_preconditioner(matrix, omega=1.0),
                "ichol": lambda: incomplete_cholesky(matrix)}
    for name, build in builders.items():
        start = time.perf_counter()
        preconditioner = build()
        built = time.perf_counter()
        run = conjugate_gradient(matrix, right, tol=1e-10, M=preconditioner,
                                 keep_history=False)
        done = time.perf_counter()
        print(f"{unknowns:>10}{name:>9}{run.n_iter:>7}{built - start:>10.4f}"
              f"{done - built:>10.4f}{done - start:>10.4f}{str(constant):>20}")
```

The iteration counts are

| unknowns a side | none | Jacobi | SSOR | incomplete Cholesky |
| --- | --- | --- | --- | --- |
| 15 | 17 | 17 | 24 | 21 |
| 31 | 42 | 42 | 44 | 39 |
| 63 | 92 | 92 | 81 | 71 |

**Jacobi does nothing at all, and it cannot.** Every count is identical, not close. The reason is in
the last column: the diagonal of this matrix is the constant $4/h^2$, so $D = (4/h^2)I$ and
$D^{-1}A$ is a scalar multiple of $A$. Scaling a matrix by a positive number leaves its condition
number and all its eigenvector directions unchanged, so CG generates the same iterates. Jacobi
preconditioning is a strict no-op here, and any measurement that showed it helping would be a bug.
This is the same fact as lesson 79's observation about the constant diagonal, seen from the solver
side.

**SSOR and incomplete Cholesky only start helping at the largest size.** At $15$ they make things
**worse**, $24$ and $21$ against $17$: the extra structure they impose costs more Krylov directions
than it saves when the problem is small. At $63$ they give $81$ and $71$ against $92$, gains of
$12\%$ and $23\%$.

**Neither is worth it here, and the timings say so bluntly.** At $63$ unknowns a side the totals in
this run are $0.44$ s unpreconditioned, $2.82$ s for SSOR and $4.28$ s for incomplete Cholesky:
about six and ten times slower, for a quarter fewer iterations. Timings vary between runs and
machines, but not by anything close to that factor. Part of it is this implementation, which builds the
preconditioner densely and applies it with dense triangular solves. But even counting only
arithmetic that a sparse code would do, applying an incomplete Cholesky factor is a forward and a
back substitution with about the same nonzero count as the matrix-vector product, so it roughly
**doubles** the cost of an iteration. A $23\%$ saving in iterations against a $2\times$ cost per
iteration is a loss.

The honest conclusion is that **none of the three is worth its cost on this matrix at these sizes**.
That is not a general statement about preconditioning. It is a statement about a matrix that is
already about as well behaved as a badly conditioned matrix can be: symmetric, positive definite,
constant diagonal, uniform structure, no anisotropy and no jumping coefficients. Preconditioners
earn their keep on the problems that lack those properties, and multigrid, which is the thing that
actually fixes the $h^{-2}$, is in 3.4.

### 4.3 Red-black against row by row

Reordering the unknowns changes the Gauss-Seidel iteration matrix, because which neighbours count
as "already updated" depends on the order. The question is whether it changes the rate.

```python
import numpy as np
from nalib import elliptic as el
from nalib.iterative import iteration_matrix, spectral_radius

print(f"{'unknowns':>10}{'row by row':>13}{'red-black':>12}{'jacobi^2':>12}{'gap':>11}")
for unknowns in (7, 15, 31):
    step = 1.0 / (unknowns + 1)
    matrix = el.five_point_matrix(unknowns, step)
    grid = [(i, j) for i in range(unknowns) for j in range(unknowns)]
    order = np.asarray([i * unknowns + j for i, j in grid if (i + j) % 2 == 0]
                       + [i * unknowns + j for i, j in grid if (i + j) % 2 == 1])
    permuted = matrix[np.ix_(order, order)]
    plain = spectral_radius(iteration_matrix(matrix, "gauss-seidel"))
    coloured = spectral_radius(iteration_matrix(permuted, "gauss-seidel"))
    jacobi = spectral_radius(iteration_matrix(matrix, "jacobi"))
    print(f"{unknowns:>10}{plain:>13.6f}{coloured:>12.6f}{jacobi ** 2:>12.6f}"
          f"{abs(plain - coloured):>11.2e}")
```

The two radii are **identical to rounding** at every size, and both equal $\rho_{\text{Jacobi}}^2 =
\cos^2(\pi h)$ exactly.

That is the answer, and it is a negative one. Red-black ordering does not converge faster. What it
gives is the reason it is used anyway:

**It is the consistent ordering the SOR theory needs.** The formula
$\rho_{\text{GS}} = \rho_{\text{Jacobi}}^2$ and the optimal $\omega^*$ of 2.3 hold for consistently
ordered matrices, and the red-black permutation is what puts the matrix in the two by two block
form that definition asks for. The row by row ordering has the same radius because it is
consistently ordered too, which is exactly what the measurement confirms.

**It is parallel.** In the row by row order, updating point $j$ requires the already updated value
at $j-1$, so the sweep is sequential. In red-black, every red point depends only on black
neighbours, so all the red points can be updated at once and then all the black points. Two
vectorized passes replace a loop over $N$ points, which is why the multigrid smoother of 3.4 is
written this way and runs at $127$ unknowns a side without a Python loop over grid points.

So the ordering buys a factor of nothing in the rate and a large factor in wall time, and the two
should not be confused. A benchmark that reported red-black as "faster" would be reporting the
implementation, not the method.

### 5.1 Why $\kappa \sim h^{-2}$ cannot be avoided

**The argument.** Let $A_h$ be any discretization of $-\Delta$ on $[0,1]^2$ with spacing $h$ that
is consistent and stable, meaning $A_h v \to -\Delta v$ for smooth $v$ and $A_h^{-1}$ is bounded
uniformly. Two facts pin the spectrum from both ends.

The smallest eigenvalue cannot grow. Take $v$ the smooth first eigenfunction $\sin \pi x \sin \pi y$
of the continuous operator, sampled on the grid. Consistency says
$A_h v = 2\pi^2 v + O(h^p)$, so $A_h$ has an eigenvalue within $O(h^p)$ of $2\pi^2$, and
$\lambda_{\min}(A_h) \le 2\pi^2 + O(h^p)$: bounded above by a constant, for every $h$.

The largest eigenvalue must grow like $h^{-2}$. The stencil is local, spanning a fixed number of
cells, so its entries scale like $h^{-2}$ to be consistent with a second derivative. Applying it to
the roughest grid function, the checkerboard $(-1)^{i+j}$, differences neighbouring values that
alternate in sign, so every term adds rather than cancelling and the result is $C h^{-2}$ with
$C > 0$ independent of $h$. So $\lambda_{\max}(A_h) \ge C h^{-2}$.

Dividing, $\kappa(A_h) \ge C' h^{-2}$. Nothing in the argument used the five point stencil: it used
only that the operator is second order in the derivatives, that the stencil is local, and that the
grid resolves frequencies up to $\pi/h$. **Every** consistent local discretization of a second order
elliptic operator has this, in any dimension, at any order of accuracy. The frequency picture is the
same statement in one line: the operator's symbol is $|\xi|^2$, the resolvable frequencies run from
$O(1)$ to $O(1/h)$, and the ratio of symbols is $O(h^{-2})$.

**What a preconditioner actually does about it.** Two very different things, and the difference is
the exponent.

```python
import numpy as np
from nalib import elliptic as el
from nalib.krylov import incomplete_cholesky

print(f"{'unknowns':>10}{'kappa(A)':>12}{'jacobi':>10}{'incomplete Cholesky':>22}"
      f"{'ratio kappa':>13}{'ratio ichol':>13}")
previous = None
for unknowns in (7, 15, 31):
    step = 1.0 / (unknowns + 1)
    matrix = el.five_point_matrix(unknowns, step)
    plain = float(np.linalg.cond(matrix))
    scale = np.diag(1.0 / np.sqrt(np.diag(matrix)))
    scaled = float(np.linalg.cond(scale @ matrix @ scale))
    factor = incomplete_cholesky(matrix)
    size = matrix.shape[0]
    inverse = np.column_stack([factor(np.eye(size)[:, k]) for k in range(size)])
    values = np.abs(np.linalg.eigvals(inverse @ matrix))
    approximate = float(np.max(values) / np.min(values))
    ratios = ("", "") if previous is None else (
        f"{plain / previous[0]:.3f}", f"{approximate / previous[1]:.3f}")
    print(f"{unknowns:>10}{plain:>12.3f}{scaled:>10.3f}{approximate:>22.3f}"
          f"{ratios[0]:>13}{ratios[1]:>13}")
    previous = (plain, approximate)
```

Jacobi returns $25.274, 103.087, 414.345$: **exactly** the unpreconditioned numbers, for the reason
in 4.2. Incomplete Cholesky returns $3.074, 9.961, 37.482$, a factor of $8$ to $11$ smaller. But the
ratios between consecutive rows are $4.079, 4.019$ for the plain matrix and $3.240, 3.763$ for the
preconditioned one, both heading for $4$. **The exponent is untouched.** Incomplete Cholesky divides
$\kappa$ by a constant; it does not change $h^{-2}$ into anything else, so the CG count still grows
like $h^{-1}$ and the improvement is a constant factor on the constant.

That is the honest summary of what a classical preconditioner does: it moves the constant, and the
theorem above says the exponent is not available to move by any local approximate factorization,
because such a factorization is itself a local operator with the same symbol scaling.

The methods that do change the exponent are the ones that are not local. Multigrid works on a
hierarchy of grids, so it sees the low frequencies on a grid where they are high, and its
"condition number" is $O(1)$: exercise 3.4 measures $8, 8, 9, 9, 9$ cycles across a $329$ fold
increase in $N$. Domain decomposition with a coarse space does the same thing with two levels
instead of many. Both escape the bound by refusing the hypothesis it rests on, which is locality,
not by beating it.

### 5.2 Superconvergence at the nodes, which does not survive the second dimension

In one dimension the P1 finite element solution of $-u'' = f$ is **exact at the nodes**. The reason
is specific and it does not travel. The Galerkin error $u - u_h$ is orthogonal to the space in the
energy inner product, and in one dimension the Green's function of the operator, $G(x, y)$, is
piecewise linear in $x$ with its kink at $y$. If $y$ is a node, $G(\cdot, y)$ lies **in the finite
element space itself**, so

$$u(y) - u_h(y) = \int (u - u_h)'\, G'(\cdot, y) = 0$$

by Galerkin orthogonality. The nodal error is not small, it is zero.

```python
import numpy as np


def one_dimensional(elements):
    grid = np.linspace(0.0, 1.0, elements + 1)
    step = 1.0 / elements
    exact = lambda t: np.sin(np.pi * t) * np.exp(t)
    source = lambda t: ((np.pi ** 2 - 1.0) * np.sin(np.pi * t) * np.exp(t)
                        - 2.0 * np.pi * np.cos(np.pi * t) * np.exp(t))
    inner = elements - 1
    stiffness = (np.diag(np.full(inner, 2.0 / step))
                 + np.diag(np.full(inner - 1, -1.0 / step), 1)
                 + np.diag(np.full(inner - 1, -1.0 / step), -1))
    load = np.zeros(inner)
    panels = 400
    for k in range(1, elements):
        sample = np.linspace(grid[k - 1], grid[k + 1], 2 * panels + 1)
        hat = np.maximum(0.0, 1.0 - np.abs(sample - grid[k]) / step)
        weight = np.ones(sample.size)
        weight[1:-1:2] = 4.0
        weight[2:-1:2] = 2.0
        load[k - 1] = float(np.sum(weight * source(sample) * hat)) * (sample[1] - sample[0]) / 3.0
    got = np.linalg.solve(stiffness, load)
    return step, float(np.max(np.abs(got - exact(grid[1:-1]))))


print("one dimension")
print(f"{'elements':>10}{'h':>9}{'nodal max error':>18}")
for elements in (4, 8, 16, 32):
    step, worst = one_dimensional(elements)
    print(f"{elements:>10}{step:>9.5f}{worst:>18.3e}")
```

The nodal errors are $1.3\times10^{-13}$ through $6.3\times10^{-13}$: rounding, at every mesh, with
no order at all. Four elements already give the exact nodal values.

Now the same question in two dimensions.

```python
import math

import numpy as np
from nalib import elliptic as el

problem = el.mixed_modes_problem()
rng = np.random.default_rng(42)


def element_l2(points, samples=7):
    run = el.solve_fem(problem, points)
    mesh = run["mesh"]
    nodes = mesh["nodes"]
    full = np.zeros(nodes.shape[0])
    full[run["interior"]] = run["u"]
    edge = np.setdiff1d(np.arange(nodes.shape[0]), run["interior"])
    full[edge] = problem["boundary"](nodes[edge, 0], nodes[edge, 1])
    total = 0.0
    for cell in mesh["cells"]:
        corners = nodes[cell]
        twice = ((corners[1, 0] - corners[0, 0]) * (corners[2, 1] - corners[0, 1])
                 - (corners[2, 0] - corners[0, 0]) * (corners[1, 1] - corners[0, 1]))
        weights = rng.dirichlet(np.ones(corners.shape[0]), size=samples)
        places = weights @ corners
        approximate = weights @ full[cell]
        truth = problem["exact"](places[:, 0], places[:, 1])
        total += abs(twice) / 2.0 * float(np.mean((approximate - truth) ** 2))
    return math.sqrt(total)


print("two dimensions")
print(f"{'points':>8}{'h':>9}{'nodal max':>12}{'L2 over elements':>19}{'L2 / nodal':>12}")
steps, nodal = [], []
for points in (5, 9, 17, 33):
    run = el.solve_fem(problem, points)
    mesh = run["mesh"]
    truth = problem["exact"](mesh["nodes"][run["interior"], 0],
                             mesh["nodes"][run["interior"], 1])
    worst = float(np.max(np.abs(run["u"] - truth)))
    spread = element_l2(points)
    steps.append(float(mesh["h"]))
    nodal.append(worst)
    print(f"{points:>8}{float(mesh['h']):>9.5f}{worst:>12.3e}{spread:>19.3e}"
          f"{spread / worst:>12.3f}")
print(f"nodal order {float(np.polyfit(np.log(np.asarray(steps)), np.log(np.asarray(nodal)), 1)[0]):.4f}")
```

The nodal errors are $7.20\times10^{-2}, 1.89\times10^{-2}, 4.72\times10^{-3}, 1.18\times10^{-3}$,
a second order fit of $1.9785$, and the ratio of the $L^2$ error over the elements to the nodal
maximum sits at $1.40, 1.16, 1.20, 1.19$.

**So there is no superconvergence at the nodes in two dimensions.** The ratio is the test. If the
nodes were special, it would grow without bound as the mesh refines, the way it does in one
dimension where the nodal error is at rounding and the ratio is astronomical. Instead it is
**flat**, near $1.2$ across four grids, which says the two errors have the same order and the same
constant to within twenty percent. The nodal error is not zero, and it is not even smaller than the
error in between the nodes. Whatever advantage the nodes had in one dimension is gone completely.

The nodal column is also worth a second look for a different reason: $1.892\times10^{-2},
4.715\times10^{-3}, 1.183\times10^{-3}$ are the same digits as the five point difference solve of
exercise 3.1 and the multigrid solve of 3.4. That is exercises 1.5 and 2.5 again, from a third
direction: on this mesh with this quadrature the two methods are the same method, so of course the
nodal errors agree exactly.

The reason is the Green's function again. In two dimensions $G(\cdot, y)$ has a logarithmic
singularity at $y$ and is not piecewise linear anywhere, so it does not lie in the finite element
space, for any mesh, and the argument that made the nodal error vanish has no second dimension
version. There is a residue of the idea: the **gradient** superconverges at particular points
inside each element, the Gauss points, and recovery techniques such as Zienkiewicz-Zhu patch
recovery exploit exactly that. But the clean statement, exact at the nodes, is a one dimensional
accident.

This is worth holding on to as a general warning. A result proved in one dimension, verified there
to $10^{-13}$, and stated in a form that mentions nothing dimension specific, can still be false in
two dimensions. The one dimensional proof used a property of the one dimensional Green's function
without saying so out loud.

### 5.3 The reentrant corner, its exponent, and the two remedies

**The prediction.** Near a corner of interior angle $\omega$, separating variables in polar
coordinates for $\Delta u = 0$ with zero data on both faces gives solutions
$u \sim r^{\alpha}\sin(\alpha\theta)$ with

$$\alpha = \frac{\pi}{\omega} .$$

For a convex corner $\omega < \pi$ gives $\alpha > 1$ and everything is smooth enough. For a
reentrant corner $\omega > \pi$ gives $\alpha < 1$, so $u$ is continuous but its gradient blows up
like $r^{\alpha - 1}$. The L shape has $\omega = 3\pi/2$, so

$$\alpha = \frac{\pi}{3\pi/2} = \frac{2}{3}, \qquad |\nabla u| \sim r^{-1/3} .$$

Then $u \in H^{1+\alpha-\varepsilon}$ but not $H^2$, and the finite element estimates on a
quasi-uniform mesh give order $\alpha$ in the energy norm and $2\alpha$ in $L^2$, that is $0.667$
and $1.333$. Exercise 3.5 measures $0.6124$ and $1.2253$: both just under, both matching the
prediction and both far from the $2$ a smooth solution would give.

**The two remedies.** Both attack the same thing, which is that a uniform mesh spends its
resolution where the solution is smooth.

*Mesh grading.* Refine toward the corner, so the elements shrink where the gradient grows. With
node positions warped as $r \mapsto r^{\beta}$, the theory says the full order returns once
$\beta \ge p/\alpha$, which here is $2/(2/3) = 3$.

```python
print(f"{'grading':>9}{'max order':>12}   errors")
for grading in (1.0, 2.0, 3.0, 4.0):
    measured = [solve_l_shaped(refine, grading) for refine in (4, 8, 16, 32)]
    coarse = np.asarray([row[0] for row in measured])
    worst = np.asarray([row[1] for row in measured])
    order = float(np.polyfit(np.log(coarse), np.log(worst), 1)[0])
    print(f"{grading:>9.1f}{order:>12.4f}   {np.array2string(worst, precision=4)}")
```

The measured orders are $0.6124, 1.2685, 1.7357, 1.8710$ for gradings $1, 2, 3, 4$. The theory says
$\beta = 3$ should restore full second order, and $3$ gives $1.74$ while $4$ gives $1.87$: climbing
toward $2$ and not yet there over four grids, which is what a fit gives when the constant is still
moving. The direction and the size of the effect are right, and the practical point is unmistakable:
at the same $h$, grading $3$ gives $2.4\times10^{-4}$ where the uniform mesh gives
$8.3\times10^{-3}$, a factor of $34$ for nothing but moving the nodes.

*Singular enrichment.* Add the function $r^{2/3}\sin(2\theta/3)$ itself to the trial space, as an
extra basis function with one unknown coefficient. Since it is exactly the mode causing the trouble,
the remainder is smooth and the polynomials handle it at full order. This is what the singular
function method and the eXtended finite element method do, and it is sharper than grading because it
removes the singularity rather than resolving it. The cost is that the exponent has to be known in
advance, which requires the corner angle, so it works for polygons and not for general non-smooth
boundaries. Grading needs no such knowledge and is what a general purpose adaptive code does, which
is why adaptive refinement driven by a local error estimator ends up producing a graded mesh at
every reentrant corner without being told the corners are there.

**One more thing the measurement says.** The uniform mesh result is not wrong, it is slow, and
nothing in the output announces the problem. The errors fall, the solve succeeds, the residual is
at rounding. Only fitting the order reveals that the method has silently lost two thirds of its
accuracy. On a domain with several corners of different angles, the worst one sets the global rate,
and it is the one you did not think about.

---

## Lesson 83, Nonlinear PDEs and the Method of Lines

### 1.1 Why a nonlinear elliptic problem gives a nonlinear system

Lesson 82's argument carries over unchanged and then one thing is added on top.

The elliptic part is the same. There is no time variable, so there is no direction to march in. The
condition at a point couples it to its neighbours in every direction, the data is given on the whole
boundary, and no ordering of the grid points lets each one be computed from points already known.
So the discrete problem is the whole set of interior equations at once.

What the nonlinearity adds is that those equations are no longer linear in the unknowns. For
$-u'' + g(u) = f$ the $i$th equation is

$$-\frac{u_{i+1} - 2u_i + u_{i-1}}{h^{2}} + g(u_i) - f_i = 0 ,$$

which is affine in the neighbours but not in $u_i$. Writing the whole set as $F(u) = 0$, there is no
matrix $A$ with $F(u) = Au - b$, so Gaussian elimination has nothing to eliminate.

The consequences are the ones that make this lesson different from 82.

**Solving becomes iterating on a sequence of linear systems.** Newton's method replaces $F(u) = 0$
by a linear solve at each step, $J(u^{(k)}) \delta = -F(u^{(k)})$, so the linear algebra of lesson
82 is still what does the work, just repeatedly.

**The number of solutions is no longer one.** A nonsingular linear system has exactly one solution.
A nonlinear one can have none, one, two, or a continuum, and which of them Newton finds depends on
where it starts. Section 2 of the lesson is entirely about that, and the Bratu problem is the
example where the count changes as a parameter moves.

**"Convergence" now means two different things.** The discretization converges to the PDE as
$h \to 0$, and the iteration converges to the discrete solution as $k$ grows. They are independent,
they have separate orders, and confusing them is the error exercise 3.3 turns on.

### 1.2 Why fitting the whole residual history gives 1.75

Newton's order is measured by fitting $\log r_{k+1}$ against $\log r_k$, and the answer is the slope.
Quadratic convergence means slope 2, which is a statement about the **asymptotic** phase, once the
iterate is close enough that the second order Taylor remainder dominates.

A Newton run on a strongly nonlinear problem is not in that phase at the start. From a poor guess,
the first step can send the residual **up**, sometimes by many orders of magnitude, and then bring
it down by a roughly constant factor per step. That middle stretch is linear convergence, slope
about 1. Fitting a straight line through both phases returns a weighted average of 1 and 2, and
where the weight falls depends on how many steps each phase took. On the lesson's run it lands at
$1.75$.

So the number is not a measurement of Newton's order. It is a measurement of how much of the run
was spent not yet converging. The fix is to fit the tail alone, and exercise 4.1 does that across
five orders of magnitude of nonlinearity: the final order is $2.00$ every time, while the number of
slow steps before it runs from 1 to 18.

Reporting $1.75$ alone would state that Newton is not quadratic here, which is false. Reporting
$2.00$ alone would hide that most of the work happened before the quadratic phase began, which is
the practically important half. Both belong in the answer.

### 1.3 The Bratu problem above its critical $\lambda$

There is **no solution at all**. Not a solution that is hard to find, not one that Newton converges
to slowly: the boundary value problem $-u'' = \lambda e^{u}$, $u(0) = u(1) = 0$ has none for
$\lambda > \lambda^{*} = 3.5138307\ldots$.

The picture is a curve in the $(\lambda, \lVert u \rVert)$ plane that comes up from the origin,
turns around at $\lambda^{*}$, and goes back to the left with $\lVert u\rVert$ still growing. A
vertical line at $\lambda < \lambda^{*}$ crosses it twice, so there are two solutions: a small one
on the lower branch and a large one on the upper. At $\lambda^{*}$ the line is tangent and there is
exactly one. To the right of it the line misses the curve entirely.

The physical reading is thermal runaway. The term $\lambda e^{u}$ is heat generated at a rate that
grows with the temperature, and diffusion carries it away at a rate that grows only linearly. Below
$\lambda^{*}$ the two can balance. Above it, generation beats removal at every temperature, and the
solution does not exist because the steady state does not exist. In the time dependent version the
temperature blows up in finite time.

What Newton does about it is worth stating precisely, because it is the practical symptom. It does
not report "no solution". It fails to converge, or converges to a large residual, or the Jacobian
becomes singular during the iteration and the linear solve reports trouble. None of those says why.
Only continuation, which is exercise 3.1, shows the fold and makes the answer legible.

### 1.4 The method of lines, in one sentence

**Discretize space only, which turns the PDE into a system of ordinary differential equations in
time, and hand that system to any ODE solver.**

What it buys is that every result about ODE solvers becomes available at once. Order, embedded error
estimates, step size control, stiff solvers, dense output, event location: none of it has to be
rederived for PDEs. The heat equation with the five point stencil in space and Euler in time is
lesson 78's explicit scheme bit for bit, and section 3.1 of the lesson checks that the two arrays
differ by exact zero, so nothing was given up by taking the detour.

The second thing it buys is that the two errors are **separable**. The semi-discrete system has an
exact solution, $e^{At}u_0$, so the space error and the time error can each be measured on its own
rather than only in combination. Exercises 3.3 and 5.3 both depend on being able to do that, and
section 3.1's warning depends on it too: the two errors have opposite signs, and only measuring them
separately reveals that a more accurate time integrator can make the total worse.

The cost is that a purely spatial discretization can be stiff, and section 4 is about what that does
to an explicit solver.

### 1.5 Why the adaptive step barely moves when the tolerance moves by $10^{6}$

Because the step is not being set by accuracy. It is being set by stability.

An adaptive solver proposes a step, estimates the local error, and accepts or rejects. On a stiff
problem the semi-discrete eigenvalues reach out to $-4\alpha/h^{2}$, so an explicit method has a
hard boundary: below it the fastest mode decays, above it that mode is amplified every step. Just
above the boundary the error estimate is not slightly too big, it is enormous, because the estimate
is a difference of two formulas that both blow up. So the controller rejects and shrinks until it is
back inside the region.

Just below the boundary, the local error is already far under any reasonable tolerance, because the
step is tiny compared with what accuracy alone would need. So tightening the tolerance changes
nothing: the constraint that bites is the same one either way.

The measurement in exercise 4.2 makes it concrete. Six orders of magnitude of tolerance,
$10^{-4}$ down to $10^{-10}$, move the median step from $5.209\times10^{-4}$ to
$5.107\times10^{-4}$, a change of two percent. The same solver on the same grid, refined by a factor
of two in $h$, changes the step by a factor of four, because the stability limit is $O(h^{2})$.

That is the practical definition of stiffness. **An explicit solver on a stiff problem is not being
careful, it is being held**, and the giveaway is exactly this insensitivity to the tolerance.

### 2.1 The Jacobian, tridiagonal plus a diagonal

The discrete residual on the interior points $x_1, \dots, x_m$ is

$$F_i(u) = -\frac{u_{i+1} - 2u_i + u_{i-1}}{h^{2}} + g(u_i) - f_i, \qquad i = 1, \dots, m ,$$

with $u_0$ and $u_{m+1}$ the known boundary values. Differentiate with respect to $u_j$. The
difference term contributes only for $j \in \{i-1, i, i+1\}$, and the reaction term only for
$j = i$, because $g$ is applied pointwise:

$$\frac{\partial F_i}{\partial u_j} =
\begin{cases}
-1/h^{2}, & j = i \pm 1,\\[2pt]
2/h^{2} + g'(u_i), & j = i,\\[2pt]
0, & \text{otherwise}.
\end{cases}$$

So

$$\boxed{\;J(u) = \frac{1}{h^{2}}\operatorname{tridiag}(-1, 2, -1) + \operatorname{diag}\big(g'(u_1), \dots, g'(u_m)\big)\;}$$

**Linear part plus a diagonal.** The tridiagonal half does not depend on $u$ at all and can be built
once; only the diagonal changes from step to step. That is the structure, and it is the whole reason
Newton is affordable on a discretized PDE: a dense Jacobian would cost $m^2$ entries and $m^3$ to
factor, and this one costs $3m$ entries and $O(m)$ to factor by the Thomas algorithm.

Two consequences follow. In two dimensions the same argument gives the five point matrix plus a
diagonal, so the Jacobian inherits whatever sparsity the linear operator had, and lesson 82's
solvers apply unchanged. And when $g' \ge 0$ the Jacobian is symmetric positive definite, because
the tridiagonal part is and a nonnegative diagonal keeps it so, which is why conjugate gradients can
be used for the inner solve. The Bratu problem has $g'(u) = -\lambda e^{u} < 0$, which is exactly
what lets the Jacobian go singular, and exercise 5.1 is about that.

```python
import numpy as np
from nalib import nonlinearpde as nl

rng = np.random.default_rng(42)
problem = nl.cubic_problem(weight=5.0)
print(f"{'points':>8}{'unknowns':>10}{'gap to differences':>21}{'bandwidth':>11}"
      f"{'nonzeros':>10}{'dense entries':>15}")
for points in (11, 21, 41):
    pieces = nl.residual_and_jacobian(problem, points)
    m = pieces["unknowns"]
    where = rng.normal(size=m)
    exact = pieces["jacobian"](where)
    step = 1e-6
    numeric = np.column_stack([
        (pieces["residual"](where + step * np.eye(m)[:, k])
         - pieces["residual"](where - step * np.eye(m)[:, k])) / (2.0 * step)
        for k in range(m)])
    rows, columns = np.nonzero(np.abs(exact) > 1e-14)
    print(f"{points:>8}{m:>10}{float(np.max(np.abs(exact - numeric))):>21.3e}"
          f"{int(np.max(np.abs(rows - columns))):>11}{rows.size:>10}{m * m:>15}")
```

Central differences reproduce the closed form Jacobian to $10^{-6}$, which is the accuracy of the
difference itself, the bandwidth is $1$ at every size, and the nonzero count is $3m - 2$ against
$m^2$ dense entries.

### 2.2 The Bratu closed form and the condition at the fold

**The closed form.** For $-u'' = \lambda e^{u}$ on $(0,1)$ with $u(0) = u(1) = 0$, multiply by $u'$
and integrate once. Writing $u' u'' = \tfrac12 \frac{d}{dx}(u')^2$ and $e^{u}u' = \frac{d}{dx}e^{u}$
gives

$$\tfrac12 (u')^{2} + \lambda e^{u} = \text{constant} .$$

By symmetry the maximum $u_{\max}$ is at $x = 1/2$ with $u' = 0$ there, so the constant is
$\lambda e^{u_{\max}}$. Separating and integrating, and setting
$c = \sqrt{2\lambda}\,\cosh(c/4)$ to fix the boundary condition, the solution is

$$u(x) = -2\log\!\left[\frac{\cosh\big((x - \tfrac12)\tfrac{c}{2}\big)}{\cosh(c/4)}\right] ,$$

which is symmetric about $x = 1/2$, vanishes at both ends because
$\cosh(\pm c/4) = \cosh(c/4)$, and has $u_{\max} = 2\log\cosh(c/4)$.

**The parameter, eliminated.** Squaring $c = \sqrt{2\lambda}\cosh(c/4)$ gives

$$\lambda(c) = \frac{c^{2}}{2\cosh^{2}(c/4)} .$$

This is the branch, with $c$ the parameter: $c$ runs from $0$ upward, $\lambda$ rises, reaches a
maximum, and falls back toward zero while $u_{\max} = 2\log\cosh(c/4)$ keeps growing. Two values of
$c$ therefore give the same $\lambda$, which is the two solutions of 1.3.

**The fold.** The turning point is where $\lambda(c)$ is largest. Take logarithms,

$$\log\lambda = 2\log c - \log 2 - 2\log\cosh(c/4) ,$$

and differentiate:

$$\frac{\lambda'}{\lambda} = \frac{2}{c} - \frac{2}{4}\tanh\frac{c}{4}
 = \frac{2}{c} - \frac{1}{2}\tanh\frac{c}{4} = 0
\quad\Longleftrightarrow\quad \boxed{\;c\,\tanh\frac{c}{4} = 4\;}$$

Solving that gives $c^{*} = 4.7987$ and $\lambda^{*} = 3.5138307191\ldots$

```python
import math

import numpy as np
from nalib import nonlinearpde as nl


def bratu_solution(c, x):
    return -2.0 * np.log(np.cosh((np.asarray(x, dtype=float) - 0.5) * c / 2.0)
                         / math.cosh(c / 4.0))


def bratu_lambda(c):
    return c ** 2 / (2.0 * math.cosh(c / 4.0) ** 2)


lo, hi = 1e-9, 40.0
for _ in range(200):
    mid = 0.5 * (lo + hi)
    if mid * math.tanh(mid / 4.0) < 4.0:
        lo = mid
    else:
        hi = mid
critical_c = 0.5 * (lo + hi)
print(f"c* solving c tanh(c/4) = 4 is {critical_c:.10f}")
print(f"lambda* = c*^2 / (2 cosh(c*/4)^2) = {bratu_lambda(critical_c):.10f}")
print(f"the library value is {nl.bratu_critical_value():.10f}")
print(f"peak of the solution at the fold, 2 log cosh(c*/4) = "
      f"{2.0 * math.log(math.cosh(critical_c / 4.0)):.6f}")
print()
print(f"{'c':>7}{'lambda':>11}{'peak u':>10}{'residual of the equation':>28}")
grid = np.linspace(0.0, 1.0, 20001)
for c in (1.0, 3.0, critical_c, 6.0, 9.0):
    lam = bratu_lambda(c)
    values = bratu_solution(c, grid)
    step = float(grid[1] - grid[0])
    second = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / step ** 2
    worst = float(np.max(np.abs(-second - lam * np.exp(values[1:-1]))))
    print(f"{c:>7.4f}{lam:>11.6f}{float(np.max(values)):>10.4f}{worst:>28.2e}")
```

The bisection returns $c^{*}$ and $\lambda^{*}$ to ten digits and they match the library, the two
values of $c$ on either side of $c^{*}$ give $\lambda$ below $\lambda^{*}$ as the fold requires, and
substituting the closed form back into the equation leaves a residual at the level of the difference
formula's own error, which is the check that the formula solves the problem and not a different one.

### 2.3 The explicit step limit, from the stability region

This derives lesson 78's $r \le 1/2$ without ever writing down a difference scheme in time.

**The semi-discrete spectrum.** Discretizing $u_t = \alpha u_{xx}$ in space only gives
$u' = A u$ with $A = (\alpha/h^{2})\operatorname{tridiag}(1, -2, 1)$, whose eigenvalues are known in
closed form from lesson 82:

$$\mu_p = -\frac{4\alpha}{h^{2}}\sin^{2}\frac{p\pi h}{2}, \qquad p = 1, \dots, m .$$

They are all real and negative. The smallest in magnitude is near $-\alpha\pi^{2}$, which is the
continuous eigenvalue, and the largest is

$$|\mu_{\max}| = \frac{4\alpha}{h^{2}}\sin^{2}\frac{m\pi h}{2}
 = \frac{4\alpha}{h^{2}}\cos^{2}\frac{\pi h}{2} \;\longrightarrow\; \frac{4\alpha}{h^{2}} .$$

**Euler's stability region.** Forward Euler applied to $y' = \mu y$ gives $y_{n+1} = (1 + k\mu)y_n$,
so the iteration is bounded exactly when $|1 + k\mu| \le 1$. That is a disc of radius 1 centred at
$-1$. For real negative $\mu$ the condition reduces to $-2 \le k\mu \le 0$.

**Put them together.** Every eigenvalue must satisfy it, so the binding one is the largest in
magnitude:

$$k\,\frac{4\alpha}{h^{2}} \le 2 \quad\Longleftrightarrow\quad
  r = \frac{\alpha k}{h^{2}} \le \frac{1}{2} .$$

That is lesson 78's condition, derived from the ODE side alone. Nothing about parabolic PDEs was
used: only the eigenvalues of the spatial operator and the shape of one region in the complex plane.

The same recipe gives the limit for any explicit method by replacing the disc with that method's
region. RK4 reaches to $-2.7853$ on the negative real axis instead of $-2$, so its limit is
$r \le 0.6963$, a gain of $39\%$ for four times the work per step. That ratio is why an explicit
method is a bad answer to stiffness no matter which one is chosen: the region can be stretched by a
constant, and the requirement grows like $h^{-2}$.

```python
import numpy as np
from nalib import nonlinearpde as nl

print(f"{'points':>8}{'h':>9}{'largest |mu|':>15}{'4 alpha / h^2':>16}{'euler k limit':>15}"
      f"{'r at that k':>13}")
for points in (11, 21, 41, 81):
    system = nl.semi_discrete_heat(points)
    step = float(system["h"])
    worst = abs(float(np.min(system["eigenvalues"])))
    limit = 2.0 / worst
    print(f"{points:>8}{step:>9.5f}{worst:>15.2f}{4.0 * system['alpha'] / step ** 2:>16.2f}"
          f"{limit:>15.3e}{system['alpha'] * limit / step ** 2:>13.6f}")
```

The measured $r$ at the stability limit is $0.512543, 0.503097, 0.500772, 0.500193$: **just above**
$1/2$, approaching it from the wrong side as the grid refines. That is not an error and it is worth
reading carefully. The exact limit is

$$r_{\max} = \frac{1}{2\cos^{2}(\pi h/2)} > \frac12 ,$$

because the largest discrete eigenvalue is $(4\alpha/h^{2})\cos^{2}(\pi h/2)$, slightly **less** than
the $4\alpha/h^{2}$ the textbook bound uses. So $r \le 1/2$ is a safe rule that leaves a little on
the table, by $2.5\%$ at eleven points and $0.04\%$ at eighty one. Anyone who took $r \le 1/2$ as
exact and found a scheme running stably at $r = 0.505$ would have concluded the theory was wrong.

### 2.4 Why the space and time errors have opposite signs

Both errors act on the same thing, the decay rate of the slowest mode, and they push it in opposite
directions.

**The space error makes the decay too slow.** The exact first mode of $u_t = \alpha u_{xx}$ decays
at $-\alpha\pi^{2} = -9.8696$. The discrete operator's first eigenvalue is

$$\mu_1 = -\frac{4\alpha}{h^{2}}\sin^{2}\frac{\pi h}{2}
 = -\alpha\pi^{2}\left(1 - \frac{(\pi h)^{2}}{12} + \cdots\right) ,$$

using $\sin\theta = \theta - \theta^3/6 + \cdots$. The bracket is **less than one**, so
$|\mu_1| < \alpha\pi^{2}$: the semi-discrete solution decays **more slowly** than the true one and
ends up **too large**. On the lesson's grid it is $-9.8645$ against $-9.8696$.

**The time error makes the decay too fast.** Euler replaces $e^{k\mu}$ by $1 + k\mu$. For real
$\mu < 0$ and $k|\mu| < 1$,

$$e^{k\mu} = 1 + k\mu + \frac{(k\mu)^{2}}{2} + \cdots > 1 + k\mu ,$$

since the omitted terms are dominated by the positive quadratic one. So the amplification factor is
**smaller** than the true one, the numerical solution decays **too fast**, and it ends up **too
small**.

**So they cancel, partly.** One error is positive and the other negative, and the total is their sum
rather than the larger of the two. On the lesson's run the time error is $4.0\times10^{-5}$, the
space error is $8.3\times10^{-5}$, and the total for lines-plus-Euler is $4.3\times10^{-5}$, which
is close to the difference of the two rather than the sum.

That is why replacing Euler by RK4 makes the answer **worse**. RK4 drives the time error from
$4\times10^{-5}$ to $6\times10^{-16}$, and with it the cancellation, leaving the full space error of
$8.3\times10^{-5}$: a factor of $1.92$ worse overall, for four times the work per step.

The general warning is worth stating plainly. **A more accurate component can make a coupled answer
worse, and it usually means the previous answer was right for the wrong reason.** Anyone who tuned
the grid on the Euler results and then switched integrators would see the error double and blame the
new integrator.

```python
import numpy as np
from nalib import nonlinearpde as nl

out = nl.lines_against_a_direct_scheme()
print(f"mesh ratio {out['mesh_ratio']:.4f}")
print(f"lines with Euler is the direct scheme to {out['they_are_the_same_scheme']:.1e}")
print(f"{'':>20}{'time error':>14}{'total error':>14}")
print(f"{'Euler':>20}{out['euler_time_error']:>14.3e}{out['euler_total_error']:>14.3e}")
print(f"{'RK4':>20}{out['rk4_time_error']:>14.3e}{out['rk4_total_error']:>14.3e}")
print(f"{'space alone':>20}{'':>14}{out['space_error']:>14.3e}")
print(f"sum of the two magnitudes  "
      f"{out['space_error'] + out['euler_time_error']:.3e}")
print(f"difference of the two      "
      f"{abs(out['space_error'] - out['euler_time_error']):.3e}")
print(f"Euler's measured total     {out['euler_total_error']:.3e}")
```

The measured total sits at the **difference**, not the sum, which is the cancellation shown as a
number rather than asserted.

### 2.5 The pulled front speed $c(a) = Da + r/a$

Ahead of a Fisher front the solution is small, so $u(1-u) \approx u$ and the equation linearises to

$$u_t = D u_{xx} + r u .$$

Look for a travelling exponential tail, $u = e^{-a(x - ct)}$ with $a > 0$. Then
$u_t = ac\,u$, $u_{xx} = a^{2}u$, so

$$ac = Da^{2} + r \quad\Longleftrightarrow\quad \boxed{\;c(a) = Da + \frac{r}{a}\;}$$

**Where it is minimised.** Differentiate: $c'(a) = D - r/a^{2}$, which vanishes at

$$a^{*} = \sqrt{\frac{r}{D}}, \qquad c(a^{*}) = D\sqrt{\frac{r}{D}} + r\sqrt{\frac{D}{r}}
 = 2\sqrt{Dr} .$$

Since $c''(a) = 2r/a^{3} > 0$, this is a minimum, and $c(a) \ge 2\sqrt{Dr}$ for every $a$, with
equality only at $a^{*}$.

**Which speed the front actually chooses** is the part the formula does not answer on its own, and
exercise 4.3 measures it. If the initial data decays more slowly than $e^{-a^{*}x}$, that is
$a < a^{*}$, the tail is what drives the front and the speed is $c(a) = Da + r/a$, which is faster
than the minimum. This is the **pushed** case, and it is the one the often quoted $2\sqrt{Dr}$ gets
wrong. If the data decays faster, $a > a^{*}$, the tail cannot keep up and the front settles at the
minimum $2\sqrt{Dr}$: the **pulled** case. So

$$c = \begin{cases} Da + r/a, & a \le a^{*},\\ 2\sqrt{Dr}, & a \ge a^{*},\end{cases}$$

which is continuous at $a^{*}$ because that is where the two expressions agree. Note it is not the
smaller of the two: $Da + r/a \ge 2\sqrt{Dr}$ always, so taking a minimum would give the constant
$2\sqrt{Dr}$ everywhere and would be wrong for every shallow front. Exercise 4.3 measures both
regimes and the exponent of the transition.

### 3.1 Pseudo-arclength continuation around the fold

Ordinary continuation steps $\lambda$ forward and solves for $u$. At the fold that is impossible:
there is no solution for larger $\lambda$, so the solve fails, and the branch appears to end. The
fix is to stop treating $\lambda$ as the independent variable.

Pseudo-arclength adds $\lambda$ to the unknowns and adds one equation, a plane at distance $\Delta s$
along the tangent to the previous point:

$$\begin{cases} F(u, \lambda) = 0,\\[2pt]
\big\langle \dot u,\, u - u_0\big\rangle + \dot\lambda\,(\lambda - \lambda_0) = \Delta s .\end{cases}$$

Newton on the combined system uses the bordered matrix

$$\begin{bmatrix} F_u & F_\lambda \\ \dot u^{T} & \dot\lambda \end{bmatrix} ,$$

which stays **nonsingular at the fold** even though $F_u$ alone is singular there. That is the whole
trick, and exercise 5.1 shows why it works.

Two details matter for it to behave the same on every grid. The inner product on the $u$ part is
scaled by $1/m$, so an arclength of $\Delta s$ means the same thing whether there are 20 unknowns or
2000. And the new tangent is obtained from the same bordered matrix with the right hand side
$e_{m+1}$, which continues in the same direction automatically rather than needing a sign rule.

```python
import math

import numpy as np
from nalib import nonlinearpde as nl


def bratu_operator(points, dimension=1):
    """The discrete negative Laplacian on a line or a square, and the unknown count."""
    if dimension == 1:
        size = points - 2
        step = 1.0 / (points - 1)
        matrix = (np.diag(np.full(size, 2.0 / step ** 2))
                  + np.diag(np.full(size - 1, -1.0 / step ** 2), 1)
                  + np.diag(np.full(size - 1, -1.0 / step ** 2), -1))
        return matrix, size
    side = points - 2
    size = side * side
    step = 1.0 / (points - 1)
    matrix = np.zeros((size, size))
    for i in range(side):
        for j in range(side):
            row = i * side + j
            matrix[row, row] = 4.0 / step ** 2
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < side and 0 <= jj < side:
                    matrix[row, ii * side + jj] = -1.0 / step ** 2
    return matrix, size


def follow_branch(points, dimension=1, arc=0.02, max_steps=4000, stop_peak=3.0):
    """Pseudo-arclength continuation of -laplacian u = lam exp(u), around the fold."""
    matrix, size = bratu_operator(points, dimension)
    weight = 1.0 / size
    residual = lambda u, lam: matrix @ u - lam * np.exp(u)
    slope_u = lambda u, lam: matrix - np.diag(lam * np.exp(u))
    slope_lam = lambda u, lam: -np.exp(u)

    lam, u = 0.5, np.zeros(size)
    for _ in range(60):
        move = np.linalg.solve(slope_u(u, lam), -residual(u, lam))
        u = u + move
        if np.max(np.abs(move)) < 1e-13:
            break
    tangent = np.concatenate([np.linalg.solve(slope_u(u, lam), -slope_lam(u, lam)), [1.0]])
    tangent /= math.sqrt(weight * float(tangent[:size] @ tangent[:size]) + tangent[size] ** 2)

    lengths, lambdas, peaks, smallest = [], [], [], []
    walked = 0.0
    for _ in range(max_steps):
        anchor_u, anchor_lam = u.copy(), lam
        u, lam = anchor_u + arc * tangent[:size], anchor_lam + arc * tangent[size]
        for _ in range(50):
            bordered = np.zeros((size + 1, size + 1))
            right = np.zeros(size + 1)
            bordered[:size, :size] = slope_u(u, lam)
            bordered[:size, size] = slope_lam(u, lam)
            bordered[size, :size] = weight * tangent[:size]
            bordered[size, size] = tangent[size]
            right[:size] = -residual(u, lam)
            right[size] = -(weight * float(tangent[:size] @ (u - anchor_u))
                            + tangent[size] * (lam - anchor_lam) - arc)
            move = np.linalg.solve(bordered, right)
            u, lam = u + move[:size], lam + move[size]
            if np.max(np.abs(move)) < 1e-12:
                break
        walked += arc
        lengths.append(walked)
        lambdas.append(lam)
        peaks.append(float(np.max(u)))
        smallest.append(float(np.min(np.linalg.eigvalsh(slope_u(u, lam)))))
        bordered[size, :size] = weight * tangent[:size]
        bordered[size, size] = tangent[size]
        right = np.zeros(size + 1)
        right[size] = 1.0
        tangent = np.linalg.solve(bordered, right)
        tangent /= math.sqrt(weight * float(tangent[:size] @ tangent[:size])
                             + tangent[size] ** 2)
        if peaks[-1] > stop_peak:
            break
    return tuple(np.asarray(item) for item in (lengths, lambdas, peaks, smallest))


def turning_point(lengths, lambdas):
    peak = int(np.argmax(lambdas))
    fit = np.polyfit(lengths[peak - 1:peak + 2], lambdas[peak - 1:peak + 2], 2)
    place = -fit[1] / (2.0 * fit[0])
    return float(np.polyval(fit, place)), float(place)


critical = nl.bratu_critical_value()
print(f"exact critical value of the continuous problem {critical:.9f}")
print(f"{'points':>8}{'discrete fold':>16}{'gap':>12}{'ratio':>8}{'steps':>7}"
      f"{'lam range':>22}")
previous = None
for points in (21, 41, 81, 161):
    lengths, lambdas, peaks, smallest = follow_branch(points)
    fold, _ = turning_point(lengths, lambdas)
    gap = abs(fold - critical)
    ratio = f"{previous / gap:>8.2f}" if previous else f"{'':>8}"
    span = f"{lambdas.min():.4f} up to {lambdas.max():.5f}"
    print(f"{points:>8}{fold:>16.7f}{gap:>12.3e}{ratio}{len(lengths):>7}{span:>22}")
    previous = gap
lengths, lambdas, peaks, smallest = follow_branch(81)
print(f"the peak of u rises the whole way: {bool(np.all(np.diff(peaks) > 0))}, "
      f"from {peaks[0]:.4f} to {peaks[-1]:.4f}")
print(f"lambda rises then falls: it passes {lambdas.max():.5f} and returns to "
      f"{lambdas[-1]:.4f}")
```

The continuation walks straight through the fold. Starting at $\lambda = 0.52$ it climbs to
$3.51351$, then turns and comes back down to $1.88$ while the peak of $u$ keeps rising monotonically
from $0.069$ to $3.012$, where the run is stopped. Those returning points are the **upper branch**,
the second solution of 1.3, which natural continuation in $\lambda$ can never reach: at the fold it
would have to step to a $\lambda$ where nothing exists.

The discrete fold is $3.5092558, 3.5126861, 3.5135421, 3.5137556$ at $21, 41, 81, 161$ points, with
gaps to the exact continuous value of $4.58\times10^{-3}, 1.15\times10^{-3}, 2.89\times10^{-4},
7.51\times10^{-5}$ and ratios $4.00, 3.97, 3.84$. **The fold converges at second order**, which is
the order of the space discretization, so the bifurcation structure is inherited from the operator
rather than being an artefact of it. That is not obvious in advance: a turning point is a global
property of a curve, not a pointwise value, and it could have converged at half the rate.
### 3.2 The two dimensional Bratu problem, and a different number

The same equation on the unit square, $-\Delta u = \lambda e^{u}$ with $u = 0$ on the boundary, has
the same shape of answer and a different constant. There is no closed form this time: the separation
of variables trick of 2.2 needs one dimension. So the fold has to be computed, and continuation is
the way to do it.

The code is the code of 3.1 with `dimension=2`, which is the point of writing `bratu_operator` to
take a dimension in the first place. Nothing else changes: the residual, the bordered Newton solve,
the tangent, the arclength scaling by $1/m$, all of it is the same.

```python
import numpy as np

print(f"{'points':>8}{'unknowns':>10}{'h':>9}{'discrete fold':>16}{'peak u at fold':>16}")
squares, folds = [], []
for points in (9, 13, 17, 21):
    lengths, lambdas, peaks, smallest = follow_branch(points, dimension=2, arc=0.03,
                                                      stop_peak=2.5)
    fold, _ = turning_point(lengths, lambdas)
    at = int(np.argmax(lambdas))
    step = 1.0 / (points - 1)
    squares.append(step)
    folds.append(fold)
    print(f"{points:>8}{(points - 2) ** 2:>10}{step:>9.5f}{fold:>16.7f}{peaks[at]:>16.4f}")
squares, folds = np.asarray(squares), np.asarray(folds)
richardson = folds[-1] + (folds[-1] - folds[-2]) / ((squares[-2] / squares[-1]) ** 2 - 1.0)
print(f"Richardson extrapolation assuming second order: {richardson:.6f}")
print(f"the value in the literature for the unit square: 6.808124423")
print(f"gap {abs(richardson - 6.808124423):.2e}")
print(f"ratio to the one dimensional fold: {richardson / 3.513830719:.4f}")
```

The discrete folds are $6.7832764, 6.7972124, 6.8019472, 6.8040921$, and Richardson extrapolation
gives $6.807905$ against the literature value $6.808124423$: agreement to four significant figures,
a gap of $2.2\times10^{-4}$.

**It is a different number and not a simple multiple.** The ratio to the one dimensional
$3.513831$ is $1.9375$, close enough to 2 to be tempting and far enough from it to be wrong. There
is no reason for a clean ratio: the one dimensional value comes from $c\tanh(c/4) = 4$ and the two
dimensional one has no closed form at all.

**The extrapolation is doing real work here.** The coarsest grid gives $6.783$, which is $0.4\%$ low,
and even the finest gives $6.804$. Second order Richardson on the last two turns a third digit into
a fifth. That is only legitimate because 3.1 measured the order to be 2 on the problem where the
answer is known; assuming it here without that check would be guessing.

**What does not change is the structure.** Two solutions below, none above, a fold in between, a
Jacobian that goes singular exactly at the turn. The number moves, the picture does not, and that is
what makes the one dimensional problem worth studying.

### 3.3 An implicit solver, and a step no longer held by stability

Section 4 of the lesson showed an explicit solver stuck at the stability limit. The remedy is a
method whose stability region contains the whole negative real axis, and BDF is the family that has
one. The point of this exercise is to check the remedy actually works and by how much.

```python
import math

import numpy as np
from nalib import nonlinearpde as nl
from nalib.multistep import bdf_coefficients


def semi_discrete_flow(points):
    """The exact solution of the semi-discrete system, so time error can be measured alone."""
    system = nl.semi_discrete_heat(points)
    matrix = np.asarray(system["matrix"], dtype=float)
    values, vectors = np.linalg.eigh(matrix)
    start = np.sin(math.pi * system["interior"])
    projected = vectors.T @ start

    def flow(t):
        return vectors @ (np.exp(values * t) * projected)

    return system, matrix, start, flow


def march_bdf(points, steps, order, t_end=0.05, exact_start=False):
    system, matrix, start, flow = semi_discrete_flow(points)
    size = start.size
    step = t_end / float(steps)
    past = [start]
    if exact_start:
        for j in range(1, order):
            past.insert(0, flow(j * step))
    for used in range(len(past), steps + 1):
        level = min(used, order)
        alpha, beta = bdf_coefficients(level)
        alpha = [float(v) for v in alpha]
        lead = float(beta[0])
        right = -sum(alpha[i] * past[i - 1] for i in range(1, level + 1))
        past.insert(0, np.linalg.solve(alpha[0] * np.eye(size) - step * lead * matrix, right))
    return step, past[0], flow(t_end)


def march_euler(points, steps, t_end=0.05):
    system, matrix, start, flow = semi_discrete_flow(points)
    step = t_end / float(steps)
    field = start.copy()
    for _ in range(steps):
        field = field + step * (matrix @ field)
    return step, field, flow(t_end)


points = 41
system = nl.semi_discrete_heat(points)
limit = 2.0 / abs(float(np.min(system["eigenvalues"])))
print(f"points {points}, stiffness ratio {system['stiffness_ratio']:.1f}, "
      f"explicit limit {limit:.3e}")
print(f"{'steps':>7}{'k':>11}{'k / limit':>11}{'BDF1':>12}{'BDF2':>12}{'euler':>14}")
for steps in (200, 100, 50, 25, 12, 6):
    step, got1, truth = march_bdf(points, steps, 1)
    _, got2, _ = march_bdf(points, steps, 2)
    _, gote, _ = march_euler(points, steps)
    blew = float(np.max(np.abs(gote - truth)))
    shown = f"{blew:>14.3e}" if blew < 1e8 else f"{'diverged':>14}"
    print(f"{steps:>7}{step:>11.3e}{step / limit:>11.2f}"
          f"{float(np.max(np.abs(got1 - truth))):>12.3e}"
          f"{float(np.max(np.abs(got2 - truth))):>12.3e}" + shown)

print()
print(f"{'startup':>22}{'method':>8}{'order':>9}   errors")
for exact_start in (False, True):
    label = "exact starting values" if exact_start else "ramped up from BDF1"
    for order in (2, 3, 4):
        sizes, errors = [], []
        for steps in (25, 50, 100, 200, 400):
            step, got, truth = march_bdf(points, steps, order, exact_start=exact_start)
            sizes.append(step)
            errors.append(float(np.max(np.abs(got - truth))))
        fit = float(np.polyfit(np.log(sizes), np.log(errors), 1)[0])
        print(f"{label:>22}{'BDF' + str(order):>8}{fit:>9.4f}   "
              + " ".join(f"{v:.1e}" for v in errors))
```

**The step is no longer held.** Explicit Euler diverges as soon as $k$ passes the limit, which
happens between $200$ and $100$ steps, and it stays diverged for three more doublings before coming
back at $6$ steps for a reason that has nothing to do with stability. BDF1 and BDF2 run all the way
to $26.6$ times the limit without a hiccup, and their errors grow **smoothly and proportionately**
with the step: BDF1's go $3.7\times10^{-4}, 7.4\times10^{-4}, 1.5\times10^{-3}, 2.9\times10^{-3}$,
doubling as $k$ doubles, and BDF2's go $2.2\times10^{-6}, 8.7\times10^{-6}, 3.5\times10^{-5},
1.4\times10^{-4}$, quadrupling. That smooth proportionality is the signature of an error set by
**accuracy**: there is no cliff anywhere, because the stability region no longer has a boundary in
the way. At $6$ steps, where Euler has long since blown up, BDF2's error is $2.6\times10^{-3}$.

**The order is what BDF promises, once the startup allows it.** With exact starting values the fits
are $1.98$, $2.97$ and $3.93$ for BDF2, BDF3 and BDF4. With the usual ramp, starting at BDF1 and
raising the order one step at a time, every one of them reads about $2.00$.

That second row is the trap and it is worth naming. A single BDF1 step has a local error of
$O(k^{2})$, and a local error of $O(k^{2})$ committed once propagates as a global error of
$O(k^{2})$ no matter how good the rest of the run is. So the ramped startup **caps the whole method
at second order**, and BDF4 costs four times the storage of BDF2 to deliver exactly BDF2's accuracy.
This is lesson 68's "starting values matter" appearing in a place where it is easy to miss, because
the method looks like it is working: the errors fall, the solve succeeds, and only fitting the order
shows that two of the three methods are not delivering what their name says.

The remedy in a production code is a self starting ramp with the **step** reduced during the startup
so the low order steps contribute a small enough error, or a one step method of matching order for
the first few steps. Neither is free, and both are why variable order BDF codes are as intricate as
they are.

### 3.4 Newton-Krylov, with the Jacobian never formed

The Jacobian is only ever used to compute $J\delta$. It does not have to exist as a matrix, and it
does not have to be exact, because Newton's step is corrected at the next iteration anyway. Both
observations together give the Newton-Krylov method: solve $J\delta = -F$ by GMRES, and supply
$Jv$ by a directional difference,

$$Jv \approx \frac{F(u + \sigma v) - F(u)}{\sigma}, \qquad
  \sigma = \sqrt{\epsilon_{\text{mach}}}\;\frac{1 + \lVert u\rVert}{\lVert v \rVert} .$$

The scaling of $\sigma$ is lesson 14's: too large and the difference is inaccurate, too small and
cancellation destroys it, and $\sqrt{\epsilon}$ balances the two.

The inner tolerance is loose on purpose. There is no point solving for a Newton step to twelve
digits when the step itself is only an approximation to where the root is. Tightening the inner
tolerance as the outer residual falls, which is what `1e-3 * min(1, residual)` does, keeps the
quadratic convergence without paying for it early.

```python
import math

import numpy as np
from nalib import nonlinearpde as nl
from nalib.krylov import LinearOperator
from nalib.nonsymmetric import gmres

problem = nl.bratu_problem(3.0)
root_eps = math.sqrt(np.finfo(float).eps)
print(f"{'points':>8}{'unknowns':>10}{'newton':>8}{'gmres total':>13}{'products':>10}"
      f"{'gap to the direct solve':>25}{'floor':>10}")
for points in (41, 81, 161, 321):
    pieces = nl.residual_and_jacobian(problem, points)
    size = pieces["unknowns"]
    residual, exact_jacobian = pieces["residual"], pieces["jacobian"]

    def apply(u, v, at):
        length = float(np.linalg.norm(v))
        if length == 0.0:
            return np.zeros_like(v)
        nudge = root_eps * (1.0 + float(np.linalg.norm(u))) / length
        return (residual(u + nudge * v) - at) / nudge

    field = np.zeros(size)
    scale = float(np.max(np.abs(residual(field))))
    inner, products, taken, history = 0, 0, 0, []
    while taken < 40:
        here = residual(field)
        size_now = float(np.max(np.abs(here)))
        history.append(size_now)
        if size_now < 1e-10 * scale:
            break
        operator = LinearOperator(lambda v, u=field, at=here: apply(u, v, at),
                                  shape=(size, size))
        run = gmres(operator, -here, tol=1e-3 * min(1.0, size_now), keep_history=False)
        inner += run.n_iter
        products += operator.n_matvec
        field = field + run.x
        taken += 1
    straight = nl.solve_nonlinear(problem, points)
    print(f"{points:>8}{size:>10}{taken:>8}{inner:>13}{products:>10}"
          f"{float(np.max(np.abs(field - straight['u']))):>25.2e}"
          f"{history[-1]:>10.1e}")
    print(f"         residuals " + " ".join(f"{v:.1e}" for v in history))

print()
print(f"{'points':>8}{'unknowns':>10}{'dense entries':>15}{'tridiagonal':>13}"
      f"{'matrix free':>13}")
for points in (41, 161, 321, 1281):
    size = points - 2
    print(f"{points:>8}{size:>10}{size * size:>15}{3 * size - 2:>13}{0:>13}")
```

**It finds the same answer.** The gap to the direct Newton solve, which forms and factors the exact
tridiagonal Jacobian, is $2\times10^{-16}$ to $6\times10^{-15}$ at the three larger sizes, and
$3\times10^{-11}$ at the smallest, where the relative stopping test happened to be met one step
earlier. In every case the gap is at the level of the tolerance, not above it: the approximate
Jacobian did not move the root, it only affected the path taken to it. Newton takes $4$ or $5$ steps
with the same clean quadratic tail, $3.0,\ 5.4\times10^{-1},\ 2.3\times10^{-2},\ 4.5\times10^{-5},\
2.8\times10^{-9}$.

**It costs more iterations, not fewer.** The GMRES totals are $99, 283, 577, 1166$ for
$39, 79, 159, 319$ unknowns, roughly doubling with the grid: unpreconditioned GMRES on this matrix
needs $O(\sqrt\kappa) = O(h^{-1})$ steps, exactly as lesson 82's conjugate gradients did.
Newton-Krylov does not make the linear algebra easier. What it buys is that the Jacobian is never
stored, and on a two or three dimensional problem, or one where the residual comes from a black box
that cannot be differentiated, that is the difference between possible and impossible.

**The finite difference sets a floor on the residual, and finding it cost a run.** The final
residuals are $1.2\times10^{-12}, 4.6\times10^{-12}, 1.6\times10^{-11}$ at the three larger sizes,
multiplying by about $3.7$ at each doubling of the grid. That is $O(h^{-2})$, and it should be: the
residual contains $1/h^{2}$ differences, so its rounding noise scales the same way, and the
difference quotient that stands in for $J$ inherits it. The first attempt here stopped on an
**absolute** residual of $10^{-11}$; it converged in five steps up to $161$ points and then spun for
all $40$ allowed iterations at $321$, bouncing around $2\times10^{-11}$ and never getting under the
bar. The tolerance has to be relative to the scale of $F$, which is what `1e-10 * scale` above does,
and then every size stops in four or five steps.

That is a general hazard of matrix free methods and not a quirk of this one. **A method whose
Jacobian is approximate cannot drive the residual below the error in that approximation**, and where
that floor sits depends on the discretization, so a tolerance tuned on one grid can be unreachable
on the next.

### 3.5 Allen-Cahn, and coarsening that is exponentially slow

The Allen-Cahn equation $u_t = \varepsilon^{2}u_{xx} + u - u^{3}$ has two stable states, $u = \pm 1$,
separated by interfaces of width about $\varepsilon$. From random data it forms many interfaces
quickly and then they annihilate in pairs, and the question is how fast.

The obvious experiment, evolve random data and count the interfaces, gives a misleading answer, so
it is worth doing first. Started from noise with $\varepsilon = 0.02$ on a unit interval, the count
falls $514 \to 188 \to 94 \to 32 \to 10 \to 4$ over $t = 10^{-3}$ to $10$, which fits a power law
$t^{-0.42}$ well enough to publish. Then it stops. From $t = 10$ to $t = 1000$ the count stays at
$4$, unchanged, and a power law cannot do that.

So the right measurement is not the count against time but the **lifetime of one configuration
against its spacing**. Put a single region of $u = +1$ of width $d$ into a sea of $u = -1$ and time
how long it takes to disappear.

```python
import math

import numpy as np

epsilon, span, points = 0.1, 4.0, 256
line = np.linspace(0.0, span, points, endpoint=False)
spacing = float(line[1] - line[0])
frequencies = 2.0 * np.pi * np.fft.fftfreq(points, d=spacing)
step = 0.02
half_diffusion = np.exp(-epsilon ** 2 * frequencies ** 2 * (0.5 * step))


def lifetime(width, ceiling=2e4):
    """Strang splitting: half a diffusion step, an exact reaction step, half a diffusion step."""
    field = np.tanh((width / 2.0 - np.abs(line - span / 2.0)) / (epsilon * math.sqrt(2.0)))
    clock = 0.0
    while clock < ceiling:
        moved = np.real(np.fft.ifft(half_diffusion * np.fft.fft(field)))
        grown = math.exp(step)
        moved = moved * grown / np.sqrt(1.0 + moved ** 2 * (grown ** 2 - 1.0))
        field = np.real(np.fft.ifft(half_diffusion * np.fft.fft(moved)))
        clock += step
        if np.max(field) < 0.0:
            return clock
    return clock


print(f"epsilon {epsilon}, points {points}, spacing / epsilon {spacing / epsilon:.3f}")
print(f"{'width d':>9}{'d / epsilon':>13}{'lifetime':>12}{'log lifetime':>14}"
      f"{'increment':>11}")
widths = np.asarray([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
times = np.asarray([lifetime(float(w)) for w in widths])
previous = None
for width, clock in zip(widths, times):
    gap = "" if previous is None else f"{math.log(clock) - previous:>11.4f}"
    print(f"{width:>9.2f}{width / epsilon:>13.2f}{clock:>12.3f}{math.log(clock):>14.4f}{gap}")
    previous = math.log(clock)
straight = np.polyfit(widths / epsilon, np.log(times), 1)
power = np.polyfit(np.log(widths), np.log(times), 1)
print(f"exponential fit: log T = {straight[0]:.4f} d/epsilon + {straight[1]:.4f}, "
      f"largest residual "
      f"{float(np.max(np.abs(np.log(times) - np.polyval(straight, widths / epsilon)))):.4f}")
print(f"power law fit:   log T = {power[0]:.4f} log d + {power[1]:.4f}, largest residual "
      f"{float(np.max(np.abs(np.log(times) - np.polyval(power, np.log(widths))))):.4f}")
print(f"sqrt(2) = {math.sqrt(2.0):.5f}")
```

The lifetimes are $2.66, 7.68, 26.42, 102.68, 416.46, 1707.84$ for $d/\varepsilon$ from $3$ to $8$.

**The growth is exponential in $d/\varepsilon$, not a power of $d$.** The exponential fit's largest
residual is $0.17$ against the power law's $0.57$, more than three times better on the same data,
and the power law it would report, $T \sim d^{6.57}$, has no meaning at all.

**And the exponent is $\sqrt 2$.** The increments of $\log T$ per unit of $d/\varepsilon$ are
$1.0603, 1.2355, 1.3575, 1.4002, 1.4112$, climbing toward $\sqrt 2 = 1.41421$ and within $0.2\%$ of
it at the largest separation. That is not a fitted constant, it comes from the tail of the interface
profile: the steady wall is $u = \tanh\big(x/(\varepsilon\sqrt 2)\big)$, so
$1 - u \sim 2e^{-\sqrt 2 x/\varepsilon}$, two walls a distance $d$ apart feel a force of order
$e^{-\sqrt 2 d/\varepsilon}$, and a velocity that small takes a time $e^{\sqrt 2 d/\varepsilon}$ to
close the gap.

That explains the stall in the first experiment exactly. Four interfaces on a unit interval with
$\varepsilon = 0.02$ means $d/\varepsilon = 12.5$, so the expected lifetime is of order
$e^{\sqrt 2 \times 12.5} \approx 4\times10^{7}$, and a run to $t = 1000$ sees nothing. **In one
dimension Allen-Cahn does not coarsen at a rate, it coarsens in a sequence of exponentially
separated events**, and the apparent power law over any fixed window is an artefact of the window.

This is the same lesson as the Fisher fronts and the same lesson as Newton's whole-history fit: a
quantity that has not reached its asymptotic regime will still fit a straight line, and the fit will
be wrong in a way nothing in the residuals reveals.

### 4.1 Newton's order against the strength of the nonlinearity

The cubic problem $-u'' + c\,u^{3} = f$ has a knob. At $c = 0$ it is linear and Newton converges in
one step. As $c$ grows the nonlinearity grows with it, and the question is what that does to the
convergence.

The measurement has to separate two things that a single fit mixes together: how long the global
phase lasts, and what the order is once it ends. So the table reports both, with the order taken
from the last three residuals above the rounding floor.

```python
import numpy as np
from nalib import nonlinearpde as nl


def phases(history, floor=1e-9):
    values = np.asarray(history, dtype=float)
    fallen = np.where(values < values[0])[0]
    before = int(fallen[0]) if fallen.size else values.size
    ratios = values[1:] / values[:-1]
    slow = int(np.sum((ratios > 0.05) & (ratios < 0.95)))
    usable = values[values > floor]
    tail = usable[-3:]
    order = (float(np.polyfit(np.log(tail[:-1]), np.log(tail[1:]), 1)[0])
             if tail.size == 3 else float("nan"))
    return before, float(np.max(values) / values[0]), slow, order


print(f"{'weight':>9}{'steps':>7}{'steps to fall':>15}{'peak / start':>15}"
      f"{'slow steps':>12}{'final order':>13}")
for weight in (0.1, 1.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1000.0, 10000.0):
    run = nl.solve_nonlinear(nl.cubic_problem(weight=weight), 41)
    before, spike, slow, order = phases(run["residual_history"])
    print(f"{weight:>9g}{run['iterations']:>7}{before:>15}{spike:>15.4g}"
          f"{slow:>12}{order:>13.4f}")

print()
for weight in (1.0, 30.0, 1000.0):
    run = nl.solve_nonlinear(nl.cubic_problem(weight=weight), 41)
    print(f"weight {weight:>7g}: "
          + " ".join(f"{v:.1e}" for v in run["residual_history"]))
```

The answer is clean and it is not the one the question expects.

**The order is $2$ at every strength.** The final column reads
$1.88, 2.02, 2.00, 1.98, 1.98, 1.99, 2.01, 1.99, 2.00$ across five orders of magnitude of $c$. The
strength of the nonlinearity does not degrade Newton's asymptotic rate at all, and there is no reason
it should: quadratic convergence follows from $F$ being twice differentiable with a nonsingular
Jacobian at the root, and both hold here for every $c$.

**What changes is everything before that.** The number of slow steps runs $1, 1, 2, 2, 2, 5, 7, 12,
18$, and the residual spike runs $1, 1, 1, 1, 2.9, 28.7, 639, 5\times10^{5}, 4.9\times10^{8}$. At
$c = 10^{4}$ Newton takes $23$ steps, of which $18$ are the global phase and $5$ are the quadratic
one.

**The global phase becomes visible between $c = 5$ and $c = 10$.** Below that the residual falls
from the first step and there is nothing to see; at $c = 10$ it rises by a factor of $2.9$ first.
That is the answer to "find where it stops being visible", and it is a soft threshold rather than a
sharp one: the spike grows continuously from $1$, and $c \approx 7$ is where it first exceeds the
starting residual.

So the honest one line summary is: **the strength of the nonlinearity buys you a longer wait, not a
worse rate.** A globalization such as a line search or a trust region attacks the first column and
leaves the last one alone, which is exactly what it is for.

### 4.2 The adaptive step against the grid, not against the tolerance

Two sweeps, and the contrast between them is the whole exercise.

```python
import math

import numpy as np
from nalib import adaptivestep
from nalib import nonlinearpde as nl

print("refining the grid at a fixed tolerance")
print(f"{'points':>8}{'h':>10}{'euler limit':>14}{'median step':>14}{'step / h^2':>13}"
      f"{'step / limit':>14}")
spacings, chosen = [], []
for points in (11, 21, 41, 81):
    system = nl.semi_discrete_heat(points)
    limit = 2.0 / abs(float(np.min(system["eigenvalues"])))
    run = adaptivestep.solve(system["rhs"], 0.0, np.sin(math.pi * system["interior"]),
                             0.05, tol=1e-8, name="dormand prince")
    middle = float(np.median(np.diff(run["t"])))
    spacing = float(system["h"])
    spacings.append(spacing)
    chosen.append(middle)
    print(f"{points:>8}{spacing:>10.5f}{limit:>14.3e}{middle:>14.3e}"
          f"{middle / spacing ** 2:>13.4f}{middle / limit:>14.3f}")
print(f"fitted power of h: "
      f"{float(np.polyfit(np.log(spacings), np.log(chosen), 1)[0]):.4f}")

print()
print("tightening the tolerance at a fixed grid")
print(f"{'tolerance':>12}{'median step':>14}{'accepted':>10}{'rejected':>10}")
picks = []
for tolerance in (1e-4, 1e-6, 1e-8, 1e-10):
    system = nl.semi_discrete_heat(41)
    run = adaptivestep.solve(system["rhs"], 0.0, np.sin(math.pi * system["interior"]),
                             0.05, tol=tolerance, name="dormand prince")
    middle = float(np.median(np.diff(run["t"])))
    picks.append(middle)
    print(f"{tolerance:>12.0e}{middle:>14.4e}{run['accepted']:>10}{run['rejected']:>10}")
print(f"tolerance spans a factor of {1e-4 / 1e-10:.0e}, the step spans "
      f"{max(picks) / min(picks):.4f}")
```

**Against the grid: the step follows $h^{2}$.** The ratio step over $h^{2}$ is
$0.958, 0.846, 0.817, 0.825$, essentially constant, and the fitted power of $h$ is $2.07$. Halving
$h$ quarters the step, so the total work per unit of simulated time goes up by eight: four times more
steps, twice the points each.

**Against the tolerance: nothing happens.** Six orders of magnitude take the median step from
$5.209\times10^{-4}$ to $5.201\times10^{-4}$, a spread of $1.02$. The accepted step count goes
$82, 89, 90, 92$ and the rejections stay around $5$ to $9$ throughout.

**And the step sits at the stability limit.** The last column of the first table is
$1.87, 1.68, 1.63, 1.65$ times the forward Euler limit, which is right: Dormand-Prince is a fifth
order explicit pair whose stability region reaches further along the negative real axis than Euler's
does, by roughly that factor. The solver has found the boundary of its own stability region without
being told any of this, purely by rejecting steps whose error estimate came back enormous.

That is the sharpest statement of what stiffness costs. **The controller is not choosing this step
for accuracy, and no amount of relaxing the accuracy requirement will let it choose a bigger one.**
The only fixes are to change the method, which is exercise 3.3, or to change the spatial operator.

### 4.3 Fisher front speeds across the critical decay rate, and a correction

The exercise asks for the measured speed against $\min(Da + r/a,\ 2\sqrt{Dr})$. That expression is
wrong, and the measurement is what shows it.

Since $Da + r/a \ge 2\sqrt{Dr}$ for every $a$, with equality only at $a^{*}$, the minimum of the two
is **always** $2\sqrt{Dr}$. As a prediction it says the speed is constant, independent of the initial
data, which is exactly the claim exercise 2.5 disproved. The correct rule is the piecewise one:
$Da + r/a$ below $a^{*}$ and $2\sqrt{Dr}$ above.

```python
import math

import numpy as np
from nalib import adaptivestep
from nalib.nonlinearpde import fisher_front_speed

diffusion, growth = 1e-2, 1.0
critical = math.sqrt(growth / diffusion)
minimum = 2.0 * math.sqrt(diffusion * growth)


def measured_speed(decay, points=801, span=(-2.0, 18.0), t_end=40.0, window=(20.0, 40.0)):
    line = np.linspace(span[0], span[1], points)
    spacing = float(line[1] - line[0])

    def rhs(t, u):
        v = np.asarray(u, dtype=float).ravel()
        padded = np.concatenate([[1.0], v, [0.0]])
        second = (padded[2:] - 2.0 * padded[1:-1] + padded[:-2]) / spacing ** 2
        return diffusion * second + growth * v * (1.0 - v)

    run = adaptivestep.solve(rhs, 0.0, 1.0 / (1.0 + np.exp(decay * line[1:-1])),
                             t_end, tol=1e-8, name="dormand prince")
    clocks, places = [], []
    for clock, profile in zip(run["t"], run["y"]):
        crossing = np.flatnonzero(profile < 0.5)
        if crossing.size == 0 or crossing[0] == 0:
            continue
        i = int(crossing[0])
        left, right = profile[i - 1], profile[i]
        clocks.append(float(clock))
        places.append(float(line[i] + spacing * (left - 0.5) / max(left - right, 1e-300)))
    clocks, places = np.asarray(clocks), np.asarray(places)
    room = places < span[1] - 2.0
    clocks, places = clocks[room], places[room]
    late = (clocks >= window[0]) & (clocks <= window[1])
    if int(np.count_nonzero(late)) < 3:
        late = clocks >= clocks[-1] - 0.4 * (clocks[-1] - clocks[0])
    return float(np.polyfit(clocks[late], places[late], 1)[0])


print(f"D = {diffusion}, r = {growth}, a* = {critical:.1f}, 2 sqrt(Dr) = {minimum:.4f}")
print(f"{'a':>7}{'D a + r/a':>12}{'min of the two':>16}{'correct rule':>14}"
      f"{'measured':>11}{'gap to rule':>13}")
for decay in (1.0, 2.0, 4.0, 6.0, 10.0, 14.0, 20.0, 30.0):
    shallow = diffusion * decay + growth / decay
    rule = fisher_front_speed(diffusion, growth, decay)
    got = measured_speed(decay)
    print(f"{decay:>7.1f}{shallow:>12.4f}{min(shallow, minimum):>16.4f}{rule:>14.4f}"
          f"{got:>11.4f}{abs(got - rule) / rule:>13.2%}")
```

**Below $a^{*}$ the piecewise rule is exact and the minimum is badly wrong.** At $a = 1$ the front
travels at $1.0100$, exactly $Da + r/a$, while $\min(\cdot)$ predicts $0.2000$: wrong by a factor of
$5$. At $a = 2, 4, 6$ the agreement is $0.00\%, 0.01\%, 0.05\%$.

**Above $a^{*}$ both agree on the limit and the measurement falls short of it.** At $a = 14, 20, 30$
the measured speeds are $0.1959, 0.1954, 0.1953$ against the predicted $0.2000$, low by about $2.3\%$.
The three are nearly equal, which is the pulled regime doing what it should: once the initial decay
is steeper than $a^{*}$, the front forgets it entirely.

**The shortfall is not an error and exercise 5.2 names it.** Bramson's correction says the
instantaneous speed at time $t$ is $c^{*} - 3/(2a^{*}t)$, which at the middle of the fitting window,
$t = 30$, gives $0.2 - 3/(2 \times 10 \times 30) = 0.1950$. The measured $0.1954$ is $0.2\%$ from
that. So the $2.3\%$ shortfall is fully accounted for, and a longer run would shrink it, slowly.

The general point is the one the lesson keeps making. $2\sqrt{Dr}$ is quoted everywhere as "the"
Fisher speed, and it is a **lower bound** that is attained only for steep enough initial data. Used
as a formula it is wrong by a factor of $5$ here, and used as a target it is approached so slowly
that a short run makes it look wrong when it is not.

### 5.1 Why the fold is a fold: the Jacobian is singular exactly there

**The claim.** At a simple turning point the Jacobian $F_u$ has a one dimensional null space, and
that is not a coincidence of being near the fold: it is what defines it.

**The derivation.** Suppose the branch is parameterised smoothly by arclength $s$ as
$(u(s), \lambda(s))$ with $F(u(s), \lambda(s)) = 0$ throughout. Differentiate:

$$F_u\,\dot u + F_\lambda\,\dot\lambda = 0 .$$

At a fold the branch is vertical in the $\lambda$ direction, so $\dot\lambda = 0$, leaving

$$F_u\,\dot u = 0 .$$

The tangent cannot be the zero vector, since $\lVert \dot u\rVert^{2} + \dot\lambda^{2} = 1$ and
$\dot\lambda = 0$ forces $\lVert\dot u\rVert = 1$. So $F_u$ has a nontrivial null vector, namely the
tangent itself: **$F_u$ is singular at the fold, and the null direction is the direction the branch
is moving in.**

The converse holds too, which is why "singular Jacobian" and "turning point" are the same event
here. If $F_u$ is nonsingular, the implicit function theorem gives a unique smooth $u(\lambda)$ near
that point, so $\lambda$ is a good parameter and there is no turn.

**Why Newton fails there and not merely nearby.** Newton's step solves
$F_u \delta = -F$. As the fold is approached $F_u$ becomes singular, so $\lVert F_u^{-1}\rVert$ blows
up and the step grows without bound in the null direction. That is a failure of the **method**, not
of the problem: the solution exists at the fold and is perfectly well behaved. Past the fold it is a
failure of the problem too, since there is no solution to find.

Both are cured by the same device. Pseudo-arclength replaces $F_u$ by the bordered matrix

$$B = \begin{bmatrix} F_u & F_\lambda \\ \dot u^{T} & \dot\lambda \end{bmatrix} ,$$

which is nonsingular at a simple fold: if $B\begin{bmatrix}v \\ \mu\end{bmatrix} = 0$ then
$F_u v + \mu F_\lambda = 0$, and since $F_\lambda$ is not in the range of $F_u$ at a simple fold this
forces $\mu = 0$ and then $v \in \ker F_u$, so $v$ is a multiple of $\dot u$; the last row then gives
$\dot u^{T}v = 0$, hence $v = 0$. **The border repairs exactly the one direction that went singular.**

```python
import numpy as np

lengths, lambdas, peaks, smallest = follow_branch(81)
crossing = int(np.where(np.diff(np.sign(smallest)) != 0)[0][0])
zero_at = (lengths[crossing]
           + (lengths[crossing + 1] - lengths[crossing])
           * (-smallest[crossing]) / (smallest[crossing + 1] - smallest[crossing]))
fold, fold_at = turning_point(lengths, lambdas)
print(f"lambda is largest at arclength         {fold_at:.6f}")
print(f"the smallest eigenvalue crosses zero at {zero_at:.6f}")
print(f"they differ by {abs(zero_at - fold_at):.2e}, against a step of "
      f"{lengths[1] - lengths[0]:.3f}")
print()
print(f"{'arclength':>11}{'lambda':>11}{'smallest eigenvalue of F_u':>29}")
for j in range(max(0, crossing - 2), min(smallest.size, crossing + 4)):
    print(f"{lengths[j]:>11.3f}{lambdas[j]:>11.6f}{smallest[j]:>29.4e}")
print()
print("the branch is smooth in arclength even where lambda is not a usable parameter:")
print(f"  peak of u is monotone: {bool(np.all(np.diff(peaks) > 0))}")
print(f"  lambda is not monotone: {bool(np.any(np.diff(lambdas) < 0))}")
```

The smallest eigenvalue of $F_u$ crosses zero at arclength $3.216355$ and $\lambda$ is largest at
$3.216508$. They differ by $1.5\times10^{-4}$, which is **one hundredth of one continuation step**,
and the whole difference is the parabolic interpolation used to locate each. The two are the same
point.

The eigenvalue passes through zero **transversally**, running $+0.686, +0.447, +0.203, -0.045,
-0.293, -0.539$ across six steps, rather than touching zero and returning. That is what makes it a simple fold rather than a degenerate one,
and it is why the bordered system stayed solvable: Newton's inner solve at every one of the $265$
continuation steps converged in fewer than ten iterations, including the two that straddle the
singularity.

Meanwhile the branch itself is entirely smooth. The peak of $u$ increases monotonically the whole
way and $\lambda$ does not. **Nothing is wrong with the solution at the fold; what is wrong is the
choice of $\lambda$ as the independent variable**, and arclength is the change of variable that
repairs it.

### 5.2 Bramson's correction, and why the coefficient is universal

**Where the correction comes from.** Ahead of a pulled front the equation is linear,
$u_t = D u_{xx} + ru$, and the front is dragged along by that linear tail rather than by the
nonlinearity. Move to the frame of the critical speed, $\xi = x - c^{*}t$ with
$c^{*} = 2\sqrt{Dr}$, and write $u = e^{-a^{*}\xi}\,\phi(\xi, t)$ with $a^{*} = \sqrt{r/D}$. The
exponential factors cancel the growth and the advection exactly, because $a^{*}$ and $c^{*}$ are
chosen so that $Da^{*2} - a^{*}c^{*} + r = 0$, and what is left is the pure heat equation

$$\phi_t = D\,\phi_{\xi\xi} .$$

The front position is where $\phi$ reaches a fixed level. The boundary condition at the front acts
as an absorbing wall, so $\phi$ cannot simply spread as a Gaussian: the relevant solution is the one
that vanishes at the wall, which behaves like $\xi\,t^{-3/2}e^{-\xi^{2}/4Dt}$ rather than
$t^{-1/2}e^{-\xi^{2}/4Dt}$. The extra factor of $\xi/t$ is the whole story. Tracking the level set
of $e^{-a^{*}\xi}\phi$ gives

$$\xi(t) \sim -\frac{3}{2a^{*}}\log t ,$$

so the front lags the critical frame by $\tfrac{3}{2a^{*}}\log t$ and its instantaneous speed is

$$\boxed{\;c(t) = c^{*} - \frac{3}{2a^{*}t}\;}$$

**Why $3/2$ is universal.** Every problem specific quantity was absorbed before the exponent
appeared. The $3/2$ came from the power of $t$ in the absorbed heat kernel, $t^{-3/2}$ against
$t^{-1/2}$, and that power counts one dimension of diffusion plus one absorbing boundary. It knows
nothing about $D$, about $r$, about the shape of the nonlinearity, or about the initial data, as long
as the data decays faster than $e^{-a^{*}x}$ so the front is pulled. The only place the problem
enters is $a^{*} = \sqrt{r/D}$ in the denominator, and $c^{*}$ in the leading term.

That is the sense in which the correction is universal: **the exponent is a property of the
diffusion equation with an absorbing boundary, not of Fisher's equation.** The same $3/2$ appears
for any pulled front in one dimension, which is why the result is quoted for a whole class of
equations rather than for one.

```python
from nalib import nonlinearpde as nl

out = nl.a_nonlinear_time_dependent_problem()
print(f"a* = {out['critical_decay']}, textbook minimum speed = {out['textbook_minimum']}")
for row in out["rows"]:
    label = "steep, pulled" if row["steep"] else "shallow, pushed"
    print(f"initial decay a = {row['decay']} ({label}), late speed "
          f"{row['late_speed']:.4f}, prediction {row['predicted_speed']:.4f}")
    print(f"  {'window':>16}{'measured speed':>17}{'Bramson':>10}{'relative gap':>15}")
    for piece in row["windows"]:
        print(f"  {str(piece['window']):>16}{piece['speed']:>17.4f}"
              f"{piece['bramson']:>10.4f}{piece['relative_gap']:>15.4f}")
print(f"Bramson matches every window for the steep front: {out['bramson_matches_every_window']}")
print(f"worst gap {out['worst_bramson_gap']:.4f}, last window {out['the_last_window_matches_to']:.4f}")
```

For the steep front the measured speeds are $0.1494, 0.1766, 0.1895, 0.1954$ over windows
$[2,5], [5,10], [10,20], [20,40]$, against Bramson's $0.1571, 0.1800, 0.1900, 0.1950$. The relative
gaps are $4.9\%, 1.9\%, 0.25\%, 0.22\%$, tightening as the asymptotics take hold.

Two readings are worth separating.

**The approach is algebraically slow, and that is the practical content.** At $t = 40$ the speed is
still $2.3\%$ below $c^{*}$, and the gap shrinks like $1/t$, so reaching $0.1\%$ needs
$t \approx 750$. A run to $t = 4$ would have measured $0.149$ and concluded that the theory
overpredicts by a quarter.

**Bramson's correction is a much better prediction than the limit.** Over the same four windows the
constant $c^{*} = 0.2$ is wrong by $25\%, 12\%, 5\%, 2\%$, while $c^{*} - 3/(2a^{*}t)$ is wrong by
$4.9\%$ falling to $0.22\%$. A correction that reduces the error by an order of magnitude at every
time is doing real work, not decorating a limit.

The shallow front is the control. Its speeds are $0.5183, 0.5199, 0.5200, 0.5200$, matching
$Da + r/a = 0.52$ from the first window and never approaching Bramson's prediction at all, because
it is pushed and Bramson's argument does not apply to it.

### 5.3 Splitting, and the order that depends on what you split with

Operator splitting solves $u_t = A(u) + B(u)$ by alternating between $u_t = A(u)$ and $u_t = B(u)$.
For reaction-diffusion, $A$ is the diffusion, which is linear and stiff, and $B$ is the reaction,
which is nonlinear and here has a closed form: $u' = ru(1-u)$ is logistic, so

$$u(k) = \frac{u\,e^{rk}}{1 + u(e^{rk} - 1)} .$$

Two arrangements:

**Lie, or Godunov, splitting**: one full step of each, $u^{n+1} = \Phi^{B}_{k}\,\Phi^{A}_{k}\,u^{n}$.

**Strang splitting**: half a step of one, a full step of the other, half again,
$u^{n+1} = \Phi^{B}_{k/2}\,\Phi^{A}_{k}\,\Phi^{B}_{k/2}\,u^{n}$.

The splitting error comes from the operators not commuting. By the
Baker-Campbell-Hausdorff formula, $e^{kA}e^{kB} = e^{k(A+B) + \frac{k^{2}}{2}[A,B] + \cdots}$, so
Lie splitting is off by $\tfrac{k^{2}}{2}[A,B]$ per step, which is first order globally. The
symmetric arrangement makes the second order term cancel and leaves $O(k^{3})$ per step, second
order globally.

**The measurement only works if each half is solved exactly in time.** The first attempt here used a
forward Euler step for the diffusion half, and both Lie and Strang measured order $1.0000$: the
first order sub-solver dominated and the splitting error was invisible underneath it. Solving the
linear half with the matrix exponential and the reaction half with its closed form isolates the
splitting error, and only then does the difference appear.

```python
import math

import numpy as np
from nalib import adaptivestep

diffusion, growth, points, span, t_end = 1e-2, 1.0, 201, (-2.0, 8.0), 1.0
line = np.linspace(span[0], span[1], points)
spacing = float(line[1] - line[0])
inside = line[1:-1]
size = inside.size
operator = (np.diag(np.full(size, -2.0)) + np.diag(np.ones(size - 1), 1)
            + np.diag(np.ones(size - 1), -1)) * diffusion / spacing ** 2
edge = np.zeros(size)
edge[0] = diffusion / spacing ** 2
values, vectors = np.linalg.eigh(operator)
offset = np.linalg.solve(operator, edge)


def diffuse(u, k):
    """Exact in time: u' = operator u + edge, solved through the eigendecomposition."""
    return vectors @ (np.exp(values * k) * (vectors.T @ (u + offset))) - offset


def react(u, k):
    """Exact in time: the logistic equation has a closed form."""
    grown = math.exp(growth * k)
    return u * grown / (1.0 + u * (grown - 1.0))


def coupled(t, u):
    v = np.asarray(u, dtype=float).ravel()
    return operator @ v + edge + growth * v * (1.0 - v)


start = 1.0 / (1.0 + np.exp(2.0 * inside))
reference = adaptivestep.solve(coupled, 0.0, start, t_end, tol=1e-13,
                               name="dormand prince")["y"][-1]


def march(kind, steps):
    k = t_end / steps
    field = start.copy()
    for _ in range(steps):
        if kind == "lie":
            field = react(diffuse(field, k), k)
        else:
            field = react(diffuse(react(field, 0.5 * k), k), 0.5 * k)
    return float(np.max(np.abs(field - reference)))


print(f"{'steps':>7}{'k':>11}{'Lie':>12}{'Strang':>12}{'Lie / Strang':>14}")
sizes, lie, strang = [], [], []
for steps in (8, 16, 32, 64, 128):
    k = t_end / steps
    a, b = march("lie", steps), march("strang", steps)
    sizes.append(k)
    lie.append(a)
    strang.append(b)
    print(f"{steps:>7}{k:>11.3e}{a:>12.3e}{b:>12.3e}{a / b:>14.1f}")
sizes = np.asarray(sizes)
for name, errors in (("Lie", lie), ("Strang", strang)):
    print(f"{name:>8} order "
          f"{float(np.polyfit(np.log(sizes), np.log(np.asarray(errors)), 1)[0]):.4f}")
```

**Lie is first order, Strang is second, exactly.** The fits are $0.9992$ and $2.0002$ over four
halvings, and the ratio between the two errors grows $306, 613, 1227, 2455, 4911$, doubling every
time, which is the extra factor of $k$ made visible.

**Strang costs almost nothing extra.** It looks like three sub-solves per step against two, but
consecutive half steps of $B$ can be merged across the step boundary, so a run of $N$ steps costs
$N$ diffusion solves and $N + 1$ reaction solves, against $N$ and $N$ for Lie. The second order is
free apart from one extra half step at each end.

**And the comparison against solving the coupled system is not one sided.** Splitting lets each half
be solved by the method that suits it, an implicit or exact solver for the stiff linear diffusion and
a cheap explicit or exact one for the reaction, which is why it is used at all. What it costs is the
splitting error, second order at best, so a splitting scheme cannot be made higher order than two
without composing several substeps with cleverly chosen coefficients, some of them negative. Solving
the coupled system has no splitting error and can be any order, but it must handle the stiff and the
nonlinear parts with one method.

The measurement makes the trade concrete: at $k = 1/8$ Strang is already at $9.3\times10^{-7}$,
which is far below the space error on this grid, so the splitting error is not what limits the
answer. **That is usually the right test.** A splitting error that sits below the discretization
error costs nothing real, and asking whether the scheme is first or second order in the splitting is
only worth the effort when it does not.

---
