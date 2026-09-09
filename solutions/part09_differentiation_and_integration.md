# Solutions: Part 9, Numerical Differentiation and Integration

Solutions to every exercise in lessons 61 to 66, 115 in all: 19 for lesson 61, 19 for lesson 62, 19 for lesson 63, 18 for lesson 64, 19 for lesson 65, 21 for lesson 66.

Every number quoted here was measured by running the code, on this repository, with the seeds
shown. Where a measurement contradicted the claim the exercise expects, the measurement is
reported and the claim is corrected. Several exercises in this part end in a negative result for
that reason, and those are marked rather than replaced with an example that would have worked.

Run any block from the repository root. Each one is self contained apart from `nalib`.

```python
import sys
sys.path.insert(0, "src")
```

---

## Lesson 61, Numerical Differentiation

### 1.1 Why differentiation is ill conditioned and integration is not

Both operations take a function and return a number, so ask what each does to a perturbation.

Replace $f$ by $f + \delta$ with $\|\delta\|_\infty \le \varepsilon$. Then

$$
\left|\int_a^b (f + \delta) - \int_a^b f\right| = \left|\int_a^b \delta\right| \le (b-a)\varepsilon .
$$

The error in the answer is at most the error in the input, times a fixed constant. **Integration
is an averaging operation and averaging never amplifies.**

Differentiation has no such bound at all. Take $\delta(x) = \varepsilon \sin(x/\varepsilon^2)$,
which is uniformly small, and its derivative is $\varepsilon^{-1}\cos(x/\varepsilon^2)$, which is
uniformly large. **The derivative of a small function can be arbitrarily large**, so the condition
number of differentiation as a map from $C^0$ to $C^0$ is infinite.

In floating point the same statement appears as cancellation. $f(x+h)$ and $f(x)$ agree in their
leading digits, so their difference loses exactly the digits they agree in, and dividing by $h$
scales the surviving noise up. Both descriptions are the same fact: the numerator tends to zero
and so does the denominator, and the numerator hits the rounding floor first.

**The practical consequence is different in each direction.** For integration the difficulty is
efficiency, how few evaluations will do. For differentiation it is accuracy, and lesson 61's
table of digits lost is the honest answer: eight digits for a first order formula and no
algorithm recovers them.

### 1.2 Why the weights sum to zero, and the second moment

The weights satisfy the moment conditions

$$
\sum_j c_j\, o_j^{\,k} = k!\,\delta_{k,m}, \qquad k = 0, 1, \dots
$$

At $k = 0$ that reads $\sum_j c_j = 0$ for every $m \ge 1$, which is the statement that
**differentiating a constant gives zero**. Any stencil that violates it returns a nonzero
derivative for $f \equiv 1$, so it is wrong before any accuracy question arises.

For a first derivative stencil ($m = 1$) the conditions at $k = 1$ and $k = 2$ read

$$
\sum_j c_j o_j = 1, \qquad \sum_j c_j o_j^2 = 0 .
$$

The first says the stencil reproduces $f(x) = x$, whose derivative is 1. The second says it
annihilates $x^2$ at the origin, whose derivative there is 0. The $k = 2$ condition is the one
that fails first when a stencil is mistyped, so it is the useful second check.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df

TOP_MOMENT = 2            # the constant, linear and quadratic conditions

print(f"{'stencil':>28}{'order':>7}{'sum c':>12}{'sum c*o':>12}{'sum c*o^2':>14}")
for name, (offsets, order) in df.STENCILS.items():
    c = df.fornberg(offsets, order)
    o = np.asarray(offsets, dtype=float)
    # k = 0, 1, 2: the constant, linear and quadratic moment conditions
    moments = [float(np.sum(c * o ** k)) for k in range(TOP_MOMENT + 1)]
    print(f"{name:>28}{order:>7}{moments[0]:>12.1e}{moments[1]:>12.4f}{moments[2]:>14.4f}")
    assert abs(moments[0]) < 1e-9
```

Every entry in the `sum c` column is zero, as it must be.

**The `sum c*o^2` column is more interesting than it looks.** For a second derivative stencil it
reads 2, which is $2!$, and for a third or fourth derivative stencil it reads 0. For most of the
first derivative stencils it reads 0 as expected. But `forward first` reads $+1$ and
`backward first` reads $-1$.

That is not an error. Those two have only **two** offsets, so they can impose the conditions for
$k = 0$ and $k = 1$ and no more; the $k = 2$ moment is whatever it happens to be, and it is the
leading error term. `central first` also has two offsets and still gets $0$, for free, by the
symmetry argument of 2.2.

So the correct general statement is: **the conditions hold for $k < n$, and the first uncontrolled
moment is the error term.** Checking $\sum_j c_j o_j^2 = 0$ is a useful test only for stencils
wide enough or symmetric enough to have earned it.

### 1.3 Same weights, different order

The stencil is $[1, -2, 1]$ in both cases. The difference is where the nodes are.

Centred on $(-1, 0, 1)$, Taylor expansion gives

$$
\frac{f(x-h) - 2f(x) + f(x+h)}{h^2}
= f''(x) + \frac{h^2}{12}f^{(4)}(x) + O(h^4).
$$

**The $h^1$ and $h^3$ terms cancel by symmetry**, so the leading error is $O(h^2)$.

Shifted to $(0, 1, 2)$,

$$
\frac{f(x) - 2f(x+h) + f(x+2h)}{h^2} = f''(x) + h f'''(x) + O(h^2),
$$

and the $h f'''$ term survives, so the order is 1.

**So the weights do not determine the order; the offsets do.** The weights are the same because
they solve the same Vandermonde system up to a shift that happens to leave them fixed for this
particular stencil, but the error term is computed from the offsets and knows the difference.

```python
import math
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df

for name in ("central second", "forward second"):
    offsets, order = df.STENCILS[name]
    c = df.fornberg(offsets, order)
    o = np.asarray(offsets, dtype=float)
    p = df.accuracy_order(offsets, order)
    leading = float(np.sum(c * o ** (order + p))) / math.factorial(order + p)
    print(f"{name:>16}: offsets {[int(v) for v in o]}, weights {[float(v) for v in c]}, "
          f"order {p}")
    print(f"{'':>16}  leading error coefficient (moment {order + p} over its factorial): "
          f"{leading:.6f}")
```

### 1.4 Why `abs` fails and `sqrt(z**2)` does not

The complex step derivation needs $f$ to be **holomorphic in a neighbourhood of $x$**, so that
$f(x + ih) = f(x) + ihf'(x) + O(h^2)$ with a single complex derivative $f'(x)$.

`np.abs` of a complex number is the modulus $\sqrt{u^2+v^2}$, a real valued function of two real
variables. It satisfies the Cauchy-Riemann equations nowhere, and $|x + ih| = \sqrt{x^2+h^2}$ is
real, so its imaginary part is exactly zero and the complex step returns zero.

`sqrt(z**2)` is a different matter. On the real axis with $x > 0$ it equals $|x| = x$. In a
complex neighbourhood of a positive real $x$, $z^2$ stays in the right half plane, the principal
square root is holomorphic there, and $\sqrt{z^2} = z$. **So on that neighbourhood the expression
is the identity function, which is entire**, and the complex step returns 1 exactly.

The requirement is local holomorphy at the point being differentiated, not global smoothness of
the expression as written. That is why the complex step works on so much real code, and why it
fails so confusingly when it does: one `abs` or `max` in a branch that is not taken still poisons
the answer if it is evaluated.

```python
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df

for name in ("abs", "real", "conjugate", "sqrt of a square", "maximum"):
    out = df.complex_step_fails_on(name)
    verdict = "SILENTLY WRONG" if out["silently_wrong"] else "correct"
    print(f"{name:>18}: complex step {out['complex_step']:.4f}, "
          f"exact {out['exact']:.4f}   {verdict}")
```

### 2.1 The optimal step

Model the total error of an $m$th derivative stencil of accuracy order $p$ as

$$
E(h) = C h^p + \frac{\kappa \varepsilon}{h^m},
$$

where $C$ collects the leading Taylor coefficient and $\kappa$ the sum of $|c_j|$ times the scale
of $f$. Differentiate:

$$
E'(h) = pC h^{p-1} - \frac{m \kappa\varepsilon}{h^{m+1}} = 0
\;\Longrightarrow\;
h^{p+m} = \frac{m\kappa\varepsilon}{pC}
\;\Longrightarrow\;
h^\ast = \left(\frac{m\kappa\varepsilon}{pC}\right)^{1/(p+m)} .
$$

Substituting back,

$$
E(h^\ast) = C (h^\ast)^p + \kappa\varepsilon (h^\ast)^{-m}
= \text{const} \cdot \varepsilon^{p/(p+m)} .
$$

**The balance condition is not "the two terms are equal".** At the optimum,

$$
pC(h^\ast)^p = m\kappa\varepsilon (h^\ast)^{-m},
\qquad\text{that is}\qquad
p \cdot \text{truncation} = m \cdot \text{roundoff}.
$$

They are equal only when $p = m$. Asserting equality passes for the first order first derivative
and for the second order second derivative, and fails everywhere else, which is exactly how the
condition was found while writing this course.

```python
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df

print(f"{'p':>4}{'m':>4}{'truncation':>14}{'roundoff':>14}{'t/r':>9}{'m/p':>9}")
for p in (1, 2, 4, 6):
    for m in (1, 2):
        out = df.optimal_step(p, m)
        t, r = out["truncation"], out["roundoff"]
        print(f"{p:>4}{m:>4}{t:>14.4e}{r:>14.4e}{t / r:>9.4f}{m / p:>9.4f}")
        assert abs(p * t - m * r) < 1e-9 * m * r
```

The last two columns agree exactly, which is the identity $t/r = m/p$.

### 2.2 Symmetric stencils gain an order

Let the offsets be symmetric, $\{-k, \dots, -1, 0, 1, \dots, k\}$ or
$\{\pm\tfrac12, \pm\tfrac32, \dots\}$, and let $m$ be odd.

By uniqueness of the finite difference weights, the weights inherit the symmetry of the problem.
Reflecting $o \mapsto -o$ maps the $m$th derivative to $(-1)^m$ times itself, so for odd $m$ the
weights are **antisymmetric**: $c_{-j} = -c_j$, and in particular $c_0 = 0$.

Now examine the error. The stencil applied to $f$ gives

$$
\frac{1}{h^m}\sum_j c_j f(x + o_j h)
= \sum_{k \ge 0} \frac{h^{k-m}}{k!} f^{(k)}(x) \sum_j c_j o_j^{\,k} .
$$

The moment $\sum_j c_j o_j^k$ pairs $j$ with $-j$: the terms are $c_j o_j^k$ and
$c_{-j}o_{-j}^k = (-c_j)(-1)^k o_j^k$. Their sum is $c_j o_j^k(1 - (-1)^k)$, which vanishes for
**even** $k$. So every even moment is automatically zero.

The stencil is built to kill $k = 0, \dots, n-1$ except $k = m$. The first uncontrolled moment is
$k = n$. If $n - m$ is odd then $n$ has the opposite parity to $m$; since $m$ is odd, $n$ is even,
and that moment vanishes for free, so the leading error is at $k = n+1$ and the order is
$n + 1 - m$ rather than $n - m$.

The same argument with $c_{-j} = +c_j$ handles even $m$.

**So the bonus is one order, and only for a symmetric stencil whose node count has the right
parity.** The measured table in lesson 61 shows it: `central first` on 2 nodes has order 2 rather
than 1, and `central second` on 3 nodes has order 2 rather than 1.

### 2.3 The complex step formula

Let $f$ be holomorphic in a disc about $x$ and real on the real axis, so $f(\bar z) =
\overline{f(z)}$ and all the Taylor coefficients at $x$ are real. Expand along the imaginary
direction:

$$
f(x + ih) = f(x) + ihf'(x) + \frac{(ih)^2}{2}f''(x) + \frac{(ih)^3}{6}f'''(x) + \cdots
= \left[f(x) - \frac{h^2}{2}f''(x) + \cdots\right]
+ i\left[hf'(x) - \frac{h^3}{6}f'''(x) + \cdots\right].
$$

Taking the imaginary part and dividing by $h$,

$$
\frac{\operatorname{Im} f(x+ih)}{h} = f'(x) - \frac{h^2}{6}f'''(x) + O(h^4).
$$

Three hypotheses were used, and each is doing work:

1. **Holomorphy** near $x$, so a single Taylor series in $z$ exists. This is the one that fails
   for `abs`, `real` and `conj`.
2. **Real coefficients**, so that the real and imaginary parts separate as shown. Equivalently,
   $f$ real on the real axis. Without it the imaginary part mixes in $f, f'', \dots$
3. **Nothing else.** In particular no assumption about $h$ being small enough to control
   cancellation, because there is no subtraction: $\operatorname{Im}$ is a projection, not a
   difference.

Point 3 is the whole content. The formula has a truncation error $O(h^2)$ and a roundoff error
of $O(\varepsilon)$ **relative** rather than $O(\varepsilon/h)$ absolute, so $h$ can be taken to
$10^{-200}$ where the truncation term is $10^{-400}$, which is zero.

### 2.4 Richardson on a centred stencil gains two orders

Let $D(h)$ be a centred stencil for $f^{(m)}$. By the argument of 2.2 its error expansion
contains only terms whose parity matches, so

$$
D(h) = f^{(m)}(x) + a_1 h^{2} + a_2 h^{4} + a_3 h^{6} + \cdots
$$

with **no odd powers**. One extrapolation step:

$$
\frac{4 D(h/2) - D(h)}{3}
= f^{(m)} + \frac{4(a_1 h^2/4 + a_2 h^4/16 + \cdots) - (a_1 h^2 + a_2 h^4 + \cdots)}{3}
= f^{(m)} - \frac{a_2}{4}h^4 + O(h^6).
$$

The $h^2$ term is gone and the next surviving term is $h^4$, **not $h^3$**, because there is no
$h^3$ term to survive. So one step gains two orders, and level $j$ has order $2j + 2$.

For a one sided stencil the expansion has every power, so the same step removes $h^p$ and lands
on $h^{p+1}$: one order per level.

```python
import math
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df

out = df.richardson_report(np.sin, math.cos(1.0), 1.0, 0.5, levels=5)
for kind in ("forward", "central"):
    orders = [f"{v:.2f}" for v in out[kind]["orders"]]
    print(f"{kind:>9}: observed orders {orders}")
```

The forward orders are $1, 2, 3, \dots$ and the central ones $2, 4, 6, \dots$

### 2.5 The eigenvalues of the second difference matrix

Let $A$ be $n \times n$ tridiagonal with $-2$ on the diagonal and $1$ off it, scaled by $h^{-2}$,
with $h = 1/(n+1)$. This is the discrete Laplacian with Dirichlet conditions.

Try $v_k$ with components $(v_k)_j = \sin(k\pi j h)$ for $j = 1, \dots, n$. Then

$$
(Av_k)_j = \frac{1}{h^2}\Big[\sin(k\pi(j-1)h) - 2\sin(k\pi j h) + \sin(k\pi(j+1)h)\Big].
$$

Using $\sin(A \pm B) = \sin A\cos B \pm \cos A \sin B$, the outer two terms combine to
$2\sin(k\pi jh)\cos(k\pi h)$, so

$$
(Av_k)_j = \frac{2\cos(k\pi h) - 2}{h^2}\,\sin(k\pi jh)
= -\frac{4}{h^2}\sin^2\!\left(\frac{k\pi h}{2}\right)(v_k)_j ,
$$

using $1 - \cos\theta = 2\sin^2(\theta/2)$. The boundary works out because
$(v_k)_0 = \sin 0 = 0$ and $(v_k)_{n+1} = \sin(k\pi) = 0$, which is exactly the Dirichlet
condition the matrix already encodes.

So the eigenvalues are $\lambda_k = -\frac{4}{h^2}\sin^2(k\pi h/2)$ for $k = 1, \dots, n$, and
the eigenvectors are discrete sine modes. All the eigenvalues are negative, so $A$ is negative
definite, and the extremes are

$$
\lambda_1 \approx -\pi^2, \qquad \lambda_n \approx -\frac{4}{h^2},
$$

giving a condition number of about $4/(\pi h)^2$, which is $O(n^2)$. **That is why every
iterative solver in Part 6 was tested on this matrix**: it is the standard hard case, and its
spectrum is known in closed form so the answer is never in doubt.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df

import math

for n in (5, 20, 80):
    h = 1.0 / (n + 1)
    A = df.second_derivative_matrix(n, h, "dirichlet")
    got = np.sort(np.linalg.eigvalsh(A))
    k = np.arange(1, n + 1)
    want = np.sort(-4.0 / h ** 2 * np.sin(k * math.pi * h / 2.0) ** 2)
    cond = float(np.max(np.abs(got)) / np.min(np.abs(got)))
    print(f"n = {n:>3}: worst eigenvalue gap {float(np.max(np.abs(got - want))):.2e}, "
          f"condition number {cond:.4e}, 4/(pi h)^2 = {4.0 / (math.pi * h) ** 2:.4e}")
    assert np.max(np.abs(got - want)) < 1e-9 * np.max(np.abs(want))
```

### 3.1 Fornberg's recursion, implemented from the paper

The algorithm builds the weights for derivative orders $0 \dots M$ on node sets
$\{x_0\}, \{x_0,x_1\}, \dots$ incrementally, using only differences of node positions.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df

def fornberg_from_the_paper(offsets, order):
    """Fornberg (1998), Table 1, transcribed. Returns the weights for the given order."""
    o = np.atleast_1d(np.asarray(offsets, dtype=float)).ravel()
    n, M = o.size, int(order)
    c = np.zeros((n, M + 1))
    c1, c4 = 1.0, o[0]
    c[0, 0] = 1.0
    for i in range(1, n):
        top = min(i, M)
        c2, c5, c4 = 1.0, c4, o[i]
        for j in range(i):
            c3 = o[i] - o[j]
            c2 *= c3
            if j == i - 1:
                for k in range(top, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(top, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, M]

rng = np.random.default_rng(42)
worst = 0.0
for trial in range(20):
    n = int(rng.integers(3, 12))
    offsets = np.sort(rng.uniform(-2.0, 2.0, n))
    for order in range(1, min(4, n)):
        a = fornberg_from_the_paper(offsets, order)
        b = df.fornberg(offsets, order)
        worst = max(worst, float(np.max(np.abs(a - b))) / float(np.max(np.abs(b))))
print(f"worst relative disagreement over 20 random stencils: {worst:.2e}")
assert worst < 1e-12
```

They agree to rounding on random node sets, which is the check that matters: agreeing on the
symmetric stencils would not distinguish a transcription error that only shows up off centre.

### 3.2 The Chebyshev differentiation matrix from its closed form

For the Chebyshev extreme points $x_j = \cos(j\pi/n)$, $j = 0 \dots n$, the entries are

$$
D_{ij} = \frac{c_i}{c_j}\frac{(-1)^{i+j}}{x_i - x_j}\ (i \ne j),
\qquad
D_{ii} = -\sum_{j \ne i} D_{ij},
$$

with $c_0 = c_n = 2$ and $c_i = 1$ otherwise.

**The diagonal must be computed as the negative row sum**, not from its own closed form. That is
the "negative sum trick" and it makes the row sums exactly zero rather than approximately zero,
which matters because the row sum is what annihilates constants.

```python
import math
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df

def chebyshev_matrix(n):
    """The closed form spectral differentiation matrix on n+1 Chebyshev extreme points."""
    j = np.arange(n + 1)
    x = np.cos(j * math.pi / n)
    c = np.ones(n + 1)
    c[0] = c[n] = 2.0
    c = c * (-1.0) ** j
    X = np.tile(x.reshape(-1, 1), (1, n + 1))
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D = D - np.diag(D.sum(axis=1))          # the negative sum trick
    return x, D

print(f"{'n+1':>6}{'vs stencil route':>20}{'row sum, closed form':>24}{'row sum, stencil':>20}")
for n in (4, 8, 16, 32):
    x, D = chebyshev_matrix(n)
    D2 = df.differentiation_matrix(np.sort(x), 1)
    order = np.argsort(x)
    gap = float(np.max(np.abs(D[np.ix_(order, order)] - D2)))
    print(f"{n + 1:>6}{gap:>20.2e}{float(np.max(np.abs(D @ np.ones(n + 1)))):>24.2e}"
          f"{float(np.max(np.abs(D2 @ np.ones(n + 1)))):>20.2e}")
```

The two routes agree, and both have essentially exact row sums. The closed form is $O(n^2)$ to
build against $O(n^3)$ for the stencil route, which is why it is the one people use.

### 3.3 The complex step for a second derivative

The plain complex step cannot give a second derivative. Taking the imaginary part of
$f(x+ih)$ isolates the odd terms $hf' - h^3f'''/6 + \cdots$; the real part gives
$f - h^2f''/2 + \cdots$, and recovering $f''$ from it needs the subtraction
$(\operatorname{Re}f(x+ih) - f(x))$, which cancels exactly as badly as a real difference.

The fix is a number system with two independent nilpotent directions. **Dual numbers of second
order** are the simplest: work in $\mathbb{R}[\epsilon]/(\epsilon^3)$, so $\epsilon^3 = 0$ and

$$
f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{\epsilon^2}{2}f''(x).
$$

Both derivatives fall out of one evaluation, exactly, with no step size at all.

```python
import math

class Jet:
    """Truncated Taylor arithmetic to second order: value, first and second derivative."""

    def __init__(self, value, d1=0.0, d2=0.0):
        self.v, self.d1, self.d2 = float(value), float(d1), float(d2)

    def __add__(self, other):
        o = other if isinstance(other, Jet) else Jet(other)
        return Jet(self.v + o.v, self.d1 + o.d1, self.d2 + o.d2)

    __radd__ = __add__

    def __mul__(self, other):
        o = other if isinstance(other, Jet) else Jet(other)
        return Jet(self.v * o.v,
                   self.d1 * o.v + self.v * o.d1,
                   self.d2 * o.v + 2.0 * self.d1 * o.d1 + self.v * o.d2)

    __rmul__ = __mul__

    def sin(self):
        s, c = math.sin(self.v), math.cos(self.v)
        return Jet(s, c * self.d1, c * self.d2 - s * self.d1 ** 2)

    def exp(self):
        e = math.exp(self.v)
        return Jet(e, e * self.d1, e * (self.d2 + self.d1 ** 2))


point = 1.3
cases = (("sin", lambda z: z.sin(), math.sin, lambda x: -math.sin(x)),
         ("exp", lambda z: z.exp(), math.exp, math.exp),
         ("x^2 sin(x)", lambda z: z * z * z.sin(), None,
          lambda x: (2 - x ** 2) * math.sin(x) + 4 * x * math.cos(x)))
print(f"{'f':>12}{'jet second derivative':>25}{'exact':>20}{'error':>12}"
      f"{'central difference error':>26}")
for name, jet_f, _, exact_second in cases:
    got = jet_f(Jet(point, 1.0, 0.0)).d2
    want = exact_second(point)
    h = 1e-5
    plain = lambda t: jet_f(Jet(t, 0.0, 0.0)).v
    central = (plain(point + h) - 2 * plain(point) + plain(point - h)) / h ** 2
    print(f"{name:>12}{got:>25.12f}{want:>20.12f}{abs(got - want):>12.1e}"
          f"{abs(central - want):>26.1e}")
```

The jet gives the second derivative to the last bit, error $0$ or $2\times10^{-16}$. The central
difference at $h = 10^{-5}$ is at $6\times10^{-8}$ to $4\times10^{-6}$, and no choice of $h$ will
take it below about $10^{-11}$, because a second derivative divides by $h^2$ and section 5 of the
lesson gives $\varepsilon^{2/3}$ as the best it can do.

**No step size appears anywhere in the jet computation**, which is the same advantage the complex
step has, obtained by the same means: never subtract.

### 3.4 First order automatic differentiation against the complex step

```python
import math
import numpy as np

class Dual:
    """First order forward mode automatic differentiation."""

    def __init__(self, value, d=0.0):
        self.v, self.d = float(value), float(d)

    def __add__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.v + o.v, self.d + o.d)

    __radd__ = __add__

    def __mul__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.v * o.v, self.d * o.v + self.v * o.d)

    __rmul__ = __mul__

    def sin(self):
        return Dual(math.sin(self.v), math.cos(self.v) * self.d)

    def exp(self):
        return Dual(math.exp(self.v), math.exp(self.v) * self.d)


import time

def timed(fn, repeats=20000):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats

point = 1.3
dual_value = lambda: (Dual(point, 1.0) * Dual(point, 1.0)).sin().d
cs_value = lambda: float(np.imag(np.sin((point + 1e-200j) ** 2)) / 1e-200)
exact = 2 * point * math.cos(point ** 2)
print(f"dual number   : {dual_value():.16f}  error {abs(dual_value() - exact):.1e}"
      f"  {timed(dual_value) * 1e6:.2f} us")
print(f"complex step  : {cs_value():.16f}  error {abs(cs_value() - exact):.1e}"
      f"  {timed(cs_value) * 1e6:.2f} us")
print(f"exact         : {exact:.16f}")
```

Both are exact. The complex step is slower here because it goes through NumPy's complex
machinery for a scalar, and the dual number does two float multiplies. **The real difference is
not speed but reach**: the complex step needs the function to accept complex input and to be
holomorphic, and the dual number needs the function to be written against an overloadable type.
Neither is available for a black box, and only automatic differentiation extends cheaply to
gradients of many variables, which is why it, not the complex step, underlies machine learning.

### 4.1 Achievable accuracy against the order

The first thing to check is what `digits_lost` actually returns, because comparing it against
$\varepsilon^{p/(p+1)}$ turns out to be comparing a formula against itself.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df

eps = np.finfo(float).eps
out = df.digits_lost(orders=[1, 2, 4, 8])
print("digits_lost achievable :", [f"{v:.4e}" for v in out["achievable_error"]])
print("eps ** (p / (p + 1))   :",
      [f"{eps ** (p / (p + 1.0)):.4e}" for p in (1, 2, 4, 8)])
print("\nidentical, so that table IS the model; it cannot test itself")
```

So the test has to be against **measured** stencils.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df

eps = np.finfo(float).eps
steps = np.logspace(-1, -12, 200)
print(f"{'stencil':>28}{'p':>4}{'measured best':>16}{'model':>14}{'measured / model':>18}")
rows = []
for name in ("forward first", "central first", "forward first, 2nd order",
             "central first, 4th order", "central second", "central second, 4th order"):
    offsets, order = df.STENCILS[name]
    p = df.accuracy_order(offsets, order)
    exact = math.cos(1.0) if order == 1 else -math.sin(1.0)
    out = df.error_against_step(np.sin, exact, 1.0, offsets, order, steps=steps)
    model = eps ** (p / (p + order))
    rows.append(out["best_error"] / model)
    print(f"{name:>28}{p:>4}{out['best_error']:>16.2e}{model:>14.2e}"
          f"{out['best_error'] / model:>18.4f}")

ratios = rows
print(f"\nratios span {min(ratios):.4f} to {max(ratios):.4f}: "
      f"the model is a ceiling, not a prediction")
```

**The exponent is right and the constant is not.** Every measured value is below the model, by
factors ranging from $0.32$ down to $0.003$, because the model takes the leading Taylor
coefficient and $\sum_j|c_j|$ to be $O(1)$ and for $\sin$ at $x = 1$ they are smaller than
that. So the model is a ceiling rather than a prediction.

Notice also that the two second order stencils differ by a factor of a hundred from each other.
**The order fixes how the error scales; the constant decides where the curve sits**, and the
constant depends on the stencil and on the function.

Where the prediction stops holding, then, is not at some particular $p$. It never held as an
absolute value, only as a scaling law, and reading the table's rightmost column as a promise of
achievable digits is reading it for something it does not say. For wide stencils it fails in the
other direction too, because $\sum_j |c_j|$ grows with the stencil width and the true floor rises.

### 4.2 The spectral minimum against smoothness

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df


def power_law(a):
    """|x|^a and its derivative; a controls how many derivatives exist at 0."""
    return (lambda x: np.abs(np.asarray(x, dtype=float)) ** a,
            lambda x: a * np.sign(np.asarray(x, dtype=float))
            * np.abs(np.asarray(x, dtype=float)) ** (a - 1.0))


counts = [5, 9, 17, 33, 65, 129, 257]
print(f"{'integrand':>14}{'best n':>9}{'best error':>14}{'still falling?':>17}")
for label, f, fp in ([("sin", np.sin, np.cos)]
                     + [(f"|x|^{a}", *power_law(a)) for a in (1.5, 2.5, 3.5, 5.5, 8.5)]):
    out = df.spectral_against_finite_difference(f, fp, counts=counts)
    e = np.asarray(out["spectral_error"], dtype=float)
    best = int(np.argmin(e))
    falling = "yes" if best == e.size - 1 else "no"
    print(f"{label:>14}{int(out['counts'][best]):>9}{e[best]:>14.2e}{falling:>17}")
```

**The minimum moves right as the function gets rougher, and for the roughest cases it has not
been reached at 257 nodes at all.** For $\sin$ it is at 17.

That is the opposite of the intuition that a hard problem needs fewer points, and the reason is
worth stating. The minimum sits where the truncation error crosses the roundoff floor. The floor
is $O(n^2\varepsilon)$ and rises with $n$ whatever the function is. The truncation error is what
differs: for an analytic function it collapses geometrically and meets the rising floor almost
immediately, while for $|x|^{1.5}$ it falls only algebraically and is still far above the floor
at 257 nodes.

**So a rough integrand is not a reason to use fewer nodes.** It is a reason to expect a worse
answer from more of them.

### 4.3 Differentiating noisy data

```python
import math
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import differentiation as df

def noisy(f, noise, seed):
    gen = np.random.default_rng(seed)

    def g(x):
        t = np.atleast_1d(np.asarray(x, dtype=float))
        return np.asarray(f(t), dtype=float) + noise * gen.standard_normal(t.shape)

    return g

REALISATIONS = 9          # noise draws averaged at each step, to see past the fluctuation

print(f"{'noise':>10}{'best step':>13}{'best error':>13}{'predicted h*':>15}"
      f"{'predicted error':>18}")
for noise in (0.0, 1e-14, 1e-10, 1e-6, 1e-3):
    effective = max(noise, np.finfo(float).eps)
    errors, steps = [], np.logspace(-1, -8, 40)
    for h in steps:
        trials = [abs(float(df.differentiate(noisy(np.sin, noise, 1000 + t), 1.0, h,
                                             (-1, 0, 1), 1)[0]) - math.cos(1.0))
                  for t in range(REALISATIONS)]
        errors.append(float(np.median(trials)))
    errors = np.asarray(errors)
    best = int(np.argmin(errors))
    h_star = (effective * 3.0) ** (1.0 / 3.0)
    print(f"{noise:>10.0e}{steps[best]:>13.2e}{errors[best]:>13.2e}{h_star:>15.2e}"
          f"{effective ** (2.0 / 3.0):>18.2e}")
```

The optimal step moves as $\varepsilon_{\text{eff}}^{1/3}$ and the achievable error as
$\varepsilon_{\text{eff}}^{2/3}$, with $\varepsilon_{\text{eff}}$ the noise level rather than the
machine epsilon. **Noise at $10^{-3}$ makes the best possible central difference accurate to
about $10^{-2}$**, at a step of about $0.1$: the right answer is a wide stencil on smoothed data,
not a narrow one on the raw values. That is exercise 5.1.

### 5.1 Regularised differentiation

Finite differences on noisy data fail because differentiation amplifies high frequencies and
noise is all high frequency. The fix is to stop treating it as an evaluation and start treating
it as an **inverse problem**: find $u$ such that integrating $u$ reproduces the data, subject to
$u$ being smooth.

Discretely, with $A$ the trapezoidal integration matrix and $y$ the noisy samples, solve

$$
\min_u\ \|Au - y\|_2^2 + \lambda \|Lu\|_2^2,
$$

where $L$ is a first difference operator penalising roughness in $u$. That is a Tikhonov problem
of exactly the form lesson 39 solved, with normal equations $(A^TA + \lambda L^TL)u = A^Ty$.

```python
import math
import numpy as np

def regularised_derivative(y, h, lam):
    n = y.size
    A = h * np.tril(np.ones((n, n)))
    A[:, 0] *= 0.5
    for i in range(n):
        A[i, i] *= 0.5
    L = np.eye(n) - np.eye(n, k=-1)
    lhs = A.T @ A + lam * (L.T @ L)
    return np.linalg.solve(lhs, A.T @ (y - y[0]))

gen = np.random.default_rng(42)
n = 201
x = np.linspace(0.0, 2.0 * math.pi, n)
h = float(x[1] - x[0])
truth = np.cos(x)
for noise in (1e-3, 1e-2):
    y = np.sin(x) + noise * gen.standard_normal(n)
    fd = np.gradient(y, h)
    best = min(((np.linalg.norm(regularised_derivative(y, h, lam) - truth) / math.sqrt(n), lam)
                for lam in np.logspace(-8, 2, 40)))
    print(f"noise {noise:.0e}: plain finite difference rms "
          f"{float(np.linalg.norm(fd - truth) / math.sqrt(n)):.4f}, "
          f"regularised rms {best[0]:.4f} at lambda {best[1]:.2e}")
```

Read both rows, because they point in opposite directions.

At $10^{-2}$ noise the plain difference has an rms error of $0.24$, which is a quarter of the
amplitude of the signal, and the regularised version brings it to $0.066$. **A factor of four,
and the plain answer was worthless.**

At $10^{-3}$ noise the plain difference is at $0.020$ and the regularised version is at $0.031$,
so **regularisation makes it worse**. There was nothing much to fix, and the roughness penalty
bought a bias in exchange for a variance reduction that was not needed.

That is the honest shape of every regularisation result: it trades variance for bias, and it
only wins where the variance was the problem. **The price is a parameter**, and choosing
$\lambda$ needs a noise estimate, cross validation, or the L curve. The numbers above used an
oracle, sweeping $\lambda$ and reporting the best, which no real user has.

Smoothing with a spline first, as lesson 51 did, is the same idea in a different basis: the
smoothing spline's roughness penalty is $\int (u'')^2$ and its solution is a natural cubic spline.
Both approaches make the same trade and both need the same parameter.

### 5.2 Symbolic differentiation and expression swell

```python
import sympy

x = sympy.Symbol("x")
expr = x
for _ in range(4):
    expr = sympy.sin(expr * sympy.exp(expr) + 1)

DERIVATIVES = 4           # each one multiplies the expression length by about seven

print(f"{'derivatives':>13}{'expression length':>20}{'growth factor':>16}")
d, previous = expr, None
for k in range(DERIVATIVES):
    length = len(str(d))
    factor = "" if previous is None else f"{length / previous:.1f}x"
    print(f"{k:>13}{length:>20}{factor:>16}")
    previous = length
    d = sympy.diff(d, x)
```

Each `diff` multiplies the expression length by about seven, so four derivatives take it from 241
characters to 72000. That is **expression swell**, and it is why symbolic differentiation is the
wrong tool for evaluating a derivative at a point.

Automatic differentiation computes the same number without ever forming the expression: it
propagates values through the same computational graph the function already has, so its cost is
a small constant multiple of one function evaluation, independent of how the function is written.

**What symbolic differentiation is for** is different: getting a formula you can read, analyse,
simplify, or prove something about. If the answer needed is a number, use automatic
differentiation. If it is an expression, use symbolic.

### 5.3 The eigenvalues of the Chebyshev first derivative matrix

```python
import math

import numpy as np


def chebyshev_matrix(n):
    """The closed form spectral differentiation matrix on n+1 Chebyshev extreme points."""
    j = np.arange(n + 1)
    x = np.cos(j * math.pi / n)
    c = np.ones(n + 1)
    c[0] = c[n] = 2.0
    c = c * (-1.0) ** j
    X = np.tile(x.reshape(-1, 1), (1, n + 1))
    D = np.outer(c, 1.0 / c) / ((X - X.T) + np.eye(n + 1))
    return x, D - np.diag(D.sum(axis=1))


print(f"{'n+1':>6}{'largest |Re|':>15}{'largest |Im|':>15}{'spectral radius':>18}"
      f"{'radius / n^2':>15}")
sizes, radii = [8, 16, 32, 64, 96, 128], []
for n in sizes:
    x, D = chebyshev_matrix(n)
    values = np.linalg.eigvals(D)
    radius = float(np.max(np.abs(values)))
    radii.append(radius)
    print(f"{n + 1:>6}{float(np.max(np.abs(values.real))):>15.4e}"
          f"{float(np.max(np.abs(values.imag))):>15.4e}{radius:>18.4e}"
          f"{radius / n ** 2:>15.5f}")
whole = float(np.polyfit(np.log(sizes), np.log(radii), 1)[0])
tail = float(np.polyfit(np.log(sizes[-3:]), np.log(radii[-3:]), 1)[0])
print(f"\nfitted growth exponent over all sizes: {whole:.3f}")
print(f"fitted growth exponent over the last three: {tail:.3f}")
```

**The eigenvalues are not on the imaginary axis.** The exact operator $d/dx$ on periodic
functions is skew adjoint and has purely imaginary spectrum, but the Chebyshev matrix is
neither skew symmetric nor normal, and its eigenvalues have real parts of the same order as
their imaginary parts.

The spectral radius grows like $n^2$, and the fit shows how carefully that has to be stated. Over
the whole range the fitted exponent is $2.7$, which is not 2. Over the last three sizes it is
$2.1$, and $\rho/n^2$ has flattened to about $0.015$. **The $n^2$ law is asymptotic, and at
$n = 8$ it is not yet in force.** Fitting the whole range and reporting 2.7 would be a wrong
answer produced by a correct procedure applied outside its range, which is the same mistake
lesson 62 makes with pre-asymptotic panels.

The exponent is the crucial number for time stepping. Applying an explicit method to
$u_t = u_x$ discretised this way requires $\Delta t \lesssim 1/\rho(D)$, so
$\Delta t = O(n^{-2})$: **the time step is limited by the square of the spatial resolution**,
where an equally spaced first order scheme would only need $O(n^{-1})$. That is the price of
Chebyshev clustering near the boundary, where the nodes are $O(n^{-2})$ apart.

Imposing a boundary condition changes the picture materially. Deleting the row and column for
the inflow boundary gives a matrix whose eigenvalues all have negative real part, so the
semidiscrete problem is stable, whereas the full matrix has eigenvalues in the right half plane
and is not. **The boundary condition is not a detail added afterwards; it is what makes the
discrete operator well posed.**

## Lesson 62, Newton-Cotes Quadrature

### 1.1 Why the weights do not depend on $f$

The rule is built by interpolating and then integrating the interpolant. Write the interpolant in
the Lagrange basis:

$$
p(x) = \sum_j f(x_j)\,\ell_j(x),
\qquad
\int_a^b p = \sum_j f(x_j) \int_a^b \ell_j .
$$

The integral $\int_a^b \ell_j$ involves the **nodes only**. Every appearance of $f$ has been
pulled out in front as a coefficient. So the weights $w_j = \frac{1}{b-a}\int_a^b \ell_j$ are
determined by the node positions and nothing else.

What that buys is threefold.

**It is computed once.** The classical fractions can be tabulated in a book, and a routine that
integrates a million different functions on the same grid does the interpolation work zero times.

**It is a linear functional.** $Q(\alpha f + \beta g) = \alpha Q(f) + \beta Q(g)$ exactly, so the
rule commutes with everything linear and its error analysis is a statement about a linear map.

**Its stability is a property of the rule, not the problem.** $\sum_j |w_j|$ bounds the
amplification for every $f$ at once, which is what makes section 3 of the lesson a complete
argument rather than an observation about one example.

### 1.2 The odd node bonus and Simpson 3/8

Put the interval symmetrically at $[-1, 1]$ with $n$ nodes, $n$ odd, so the nodes are symmetric
about 0 and one sits at 0.

The rule is exact on degree $n-1$ by construction. Consider $x^{n}$. Since $n$ is odd, $x^n$ is an
**odd** function, so $\int_{-1}^1 x^n = 0$. The rule gives $\sum_j w_j x_j^n$, and pairing $j$
with its mirror image $j'$ (with $x_{j'} = -x_j$ and $w_{j'} = w_j$ by symmetry) gives terms
$w_j x_j^n + w_j(-x_j)^n = 0$. The centre node contributes $w\cdot 0 = 0$.

So both sides are zero and the rule is exact on $x^n$ too, giving degree $n$.

For even $n$ the same pairing shows the odd monomials are handled, but $x^n$ is now **even** and
nothing cancels, so the degree stays at $n-1$.

**Consequence for Simpson's 3/8 rule.** It uses 4 nodes, so its degree is 3. Simpson's 1/3 rule
uses 3 nodes and its degree is also 3. **The 3/8 rule costs one more evaluation per panel and
delivers nothing.**

```python
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc

print(f"{'nodes':>7}{'degree':>9}{'degree per node':>18}")
for n in range(2, 12):
    d = nc.degree_of_precision(n, True)
    print(f"{n:>7}{d:>9}{d / n:>18.4f}")
print("\nevery even count ties the odd count below it:")
for n in (4, 6, 8, 10):
    print(f"   {n} nodes: degree {nc.degree_of_precision(n, True)}, "
          f"{n - 1} nodes: degree {nc.degree_of_precision(n - 1, True)}")
```

The rule to remember: **use the odd members**. Trapezoid, Simpson 1/3, Boole and Weddle are the
family; the even ones between them are historical curiosities.

### 1.3 What $\sum|w_j| = 544$ costs, and why a polynomial test misses it

The rule computes $\sum_j w_j f(x_j)$. Each $f(x_j)$ carries a relative rounding error of order
$\varepsilon$, so the computed sum differs from the exact one by up to

$$
\varepsilon \sum_j |w_j| \max_j |f(x_j)| .
$$

The answer itself is about $\max|f|$ times the interval length. So the **relative** error of the
result is about $\varepsilon \sum_j |w_j|$, which for $\sum_j|w_j| = 544$ is $1.2\times10^{-13}$:
about two and a half digits gone before the mathematics starts.

At larger $n$ it gets worse without limit, and past roughly $n = 40$ nothing is left.

**Why a polynomial test does not show it.** Two reasons, and both matter.

First, the rule is **exact** on polynomials of low degree, so the truncation error is zero and
only the roundoff remains. A test that checks "does the 21 point rule integrate $x^5$ correctly"
gets $10^{-13}$ and calls it a pass.

Second, and worse, on the very polynomial the rule is designed for the cancellation is benign.
The large positive and negative weights multiply values of a smooth function that are all about
the same size, and the sum is genuinely small.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc

w = nc.weights(21, True)
x = nc.rule_nodes(21, 0.0, 1.0, True)
eps = np.finfo(float).eps
print(f"sum |w| = {float(np.sum(np.abs(w))):.4f}, "
      f"predicted relative floor {eps * float(np.sum(np.abs(w))):.3e}")
print()
print(f"{'integrand':>22}{'exact':>14}{'21 point rule':>18}{'error':>12}")
rng = np.random.default_rng(42)
cases = (("x^5", lambda t: t ** 5, 1.0 / 6.0),
         ("exp(x)", np.exp, math.e - 1.0),
         ("exp(30 x)", lambda t: np.exp(30.0 * t), (math.exp(30.0) - 1.0) / 30.0))
for name, f, exact in cases:
    got = float(np.sum(w * np.asarray(f(x), dtype=float)))
    print(f"{name:>22}{exact:>14.6e}{got:>18.6e}{abs(got - exact) / abs(exact):>12.2e}")
```

On $x^5$ and $\exp(x)$ the rule looks fine. On $\exp(30x)$, where the node values span thirteen
orders of magnitude, the relative error is far larger, because now the large weights are
multiplying values that genuinely differ in size and the cancellation is real.

**The stability bound is about the worst case over all $f$**, and a test on one benign $f$ cannot
find it. That is exactly why $\sum_j|w_j|$ is the thing to report.

### 1.4 Why composing is safe

Take a budget of $N$ evaluations and spend it two ways.

**One wide rule** interpolates a polynomial of degree $N-1$ through $N$ equally spaced points.
Lesson 46 showed that this interpolant diverges as $N$ grows for functions as ordinary as
$1/(1+25x^2)$, and its Lebesgue constant grows like $2^N/(N\log N)$. The quadrature weights
inherit that growth, which is section 3 of the lesson.

**Many small panels** interpolate a polynomial of degree 2 or 4 on each panel, and never form
anything bigger. The Lebesgue constant of a 3 point interpolation is a small constant, whatever
$N$ is.

So the two spend the same evaluations and construct completely different objects. The composite
rule's error is a sum of $N/k$ small local errors, each controlled by a fixed constant times
$h^{d+2}$, and the total is $O(h^{d+1})$ with a constant that does not grow.

**The key point is that convergence and stability come from different places here.** The composite
rule converges because $h \to 0$, not because the polynomial degree rises, and it is stable
because its weights are a fixed small positive set repeated. The wide rule tries to get both from
raising the degree and gets neither.

### 2.1 Trapezoid and Simpson from the Lagrange basis

On $[0, 1]$ with nodes $0, 1$:

$$
\ell_0(x) = 1 - x, \quad \ell_1(x) = x,
\qquad
\int_0^1 \ell_0 = \tfrac12, \quad \int_0^1 \ell_1 = \tfrac12 .
$$

So $w = [\tfrac12, \tfrac12]$, the trapezoid rule.

With nodes $0, \tfrac12, 1$:

$$
\ell_0 = 2(x - \tfrac12)(x-1), \quad
\ell_1 = -4x(x-1), \quad
\ell_2 = 2x(x-\tfrac12),
$$

and integrating,

$$
\int_0^1 \ell_0 = 2\left(\tfrac13 - \tfrac34 \cdot \tfrac12 + \tfrac12\right)\Big|
= \tfrac16, \qquad
\int_0^1 \ell_1 = -4\left(\tfrac13 - \tfrac12\right) = \tfrac23, \qquad
\int_0^1 \ell_2 = \tfrac16 .
$$

So $w = [\tfrac16, \tfrac23, \tfrac16]$, which is $\tfrac{h}{3}[1, 4, 1]$ after scaling by
$b - a = 2h$. **That is where the name "Simpson's one third rule" comes from.**

```python
import sys
from fractions import Fraction
sys.path.insert(0, "src")
from nalib import newtoncotes as nc


def lagrange_weights(nodes):
    """Integrate each Lagrange basis polynomial over [0, 1], exactly."""
    out = []
    n = len(nodes)
    for j in range(n):
        poly = [Fraction(1)] + [Fraction(0)] * (n - 1)
        degree, denominator = 0, Fraction(1)
        for k in range(n):
            if k == j:
                continue
            new = [Fraction(0)] * n
            for i in range(degree + 1):
                new[i + 1] += poly[i]
                new[i] -= nodes[k] * poly[i]
            poly, degree = new, degree + 1
            denominator *= (nodes[j] - nodes[k])
        out.append(sum(poly[i] / Fraction(i + 1) for i in range(n)) / denominator)
    return out


for n, name in ((2, "trapezoid"), (3, "Simpson 1/3"), (4, "Simpson 3/8"), (5, "Boole")):
    nodes = [Fraction(j, n - 1) for j in range(n)]
    mine = lagrange_weights(nodes)
    theirs = nc.exact_weights(n, True)
    print(f"{name:>14}: {[str(v) for v in mine]}")
    assert mine == theirs
print("\nall four agree with nalib.newtoncotes.exact_weights, exactly")
```

### 2.2 The odd node bonus, proved

Let the rule have $n$ nodes, $n$ odd, symmetric about the midpoint of $[a, b]$. Translate so the
interval is $[-c, c]$.

**Step 1, the weights are symmetric.** The Lagrange basis satisfies
$\ell_j(-x) = \ell_{n-1-j}(x)$, because reflecting the node set maps node $j$ to node $n-1-j$ and
the basis is determined by which node it interpolates. Integrating over a symmetric interval,
$w_j = w_{n-1-j}$.

**Step 2, the rule kills odd monomials.** For odd $k$,

$$
\sum_j w_j x_j^{\,k}
= \sum_{j < (n-1)/2} \left(w_j x_j^k + w_{n-1-j}(-x_j)^k\right) + w_{(n-1)/2}\cdot 0
= \sum_{j} w_j x_j^k \left(1 + (-1)^k\right)/1 = 0 .
$$

Meanwhile $\int_{-c}^{c} x^k\,dx = 0$ for odd $k$. **So the rule is exact on every odd monomial,
whatever $n$ is**, purely by symmetry and with no reference to the interpolation degree.

**Step 3, combine.** The rule is exact on degree $n-1$ by construction. For odd $n$ the next
monomial $x^n$ is odd, so Step 2 makes it exact too, and the degree is at least $n$. It is not
$n+1$, because $x^{n+1}$ is even and there is no reason for it to be exact, and a direct
computation shows it is not.

For even $n$, $x^n$ is even and Step 2 says nothing, so the degree stays $n-1$.

### 2.3 The trapezoid error term

The interpolation error formula for two nodes $a, b$ with $h = b - a$ is

$$
f(x) - p_1(x) = \frac{f''(\xi_x)}{2}(x-a)(x-b),
$$

with $\xi_x$ somewhere in $(a,b)$ depending on $x$. Integrate both sides:

$$
\int_a^b f - T = \int_a^b \frac{f''(\xi_x)}{2}(x-a)(x-b)\,dx .
$$

Now the key step. **The factor $(x-a)(x-b)$ does not change sign on $[a,b]$**, being negative
throughout. So the integral mean value theorem applies: there is a single $\xi \in (a,b)$ with

$$
\int_a^b \frac{f''(\xi_x)}{2}(x-a)(x-b)\,dx
= \frac{f''(\xi)}{2}\int_a^b (x-a)(x-b)\,dx
= \frac{f''(\xi)}{2}\cdot\left(-\frac{h^3}{6}\right)
= -\frac{h^3}{12}f''(\xi).
$$

**That non-sign-changing property is doing all the work**, and it is exactly what fails for even
node counts on higher rules, which is why their error terms need a different derivation
(integration against the Peano kernel rather than the mean value theorem).

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc

print(f"{'h':>10}{'measured error':>18}{'-h^3 f\"(mid)/12':>20}{'ratio':>10}")
for h in (0.4, 0.2, 0.1, 0.05, 0.025):
    a, b = 1.0, 1.0 + h
    exact = math.cos(a) - math.cos(b)
    rule = nc.integrate(np.sin, a, b, 2, True)
    predicted = -h ** 3 / 12.0 * (-math.sin(0.5 * (a + b)))
    print(f"{h:>10.3f}{exact - rule:>18.6e}{predicted:>20.6e}"
          f"{(exact - rule) / predicted:>10.5f}")
```

The ratio goes to 1 as $h$ shrinks, which is the statement that $\xi \to$ the midpoint.

### 2.4 Composite order

Let the one panel error on a panel of width $h$ be $E_{\text{panel}} = C h^{d+2} f^{(d+1)}(\xi)$
for some $\xi$ in the panel. Chop $[a,b]$ into $m = (b-a)/h$ panels and sum:

$$
E_{\text{total}} = \sum_{i=1}^{m} C h^{d+2} f^{(d+1)}(\xi_i)
= C h^{d+2} \sum_{i=1}^{m} f^{(d+1)}(\xi_i).
$$

If $f^{(d+1)}$ is continuous, the average $\frac{1}{m}\sum_i f^{(d+1)}(\xi_i)$ lies between the
minimum and maximum of $f^{(d+1)}$ on $[a,b]$, so by the intermediate value theorem it equals
$f^{(d+1)}(\eta)$ for some single $\eta \in [a,b]$. Then

$$
E_{\text{total}} = C h^{d+2} \cdot m\, f^{(d+1)}(\eta)
= C h^{d+2}\cdot \frac{b-a}{h} f^{(d+1)}(\eta)
= C(b-a)\, h^{d+1} f^{(d+1)}(\eta).
$$

**One power of $h$ is lost to the panel count.** The local order is $d+2$ and the global order is
$d+1$, and that off by one is exactly the one that sets the Richardson divisor in lesson 64.

### 2.5 The weights sum to 1, and $\sum|w_j|$ is unbounded

**The sum.** The rule integrates the interpolating polynomial exactly, and the interpolating
polynomial through the constant function $f \equiv 1$ is the constant 1. So

$$
(b-a)\sum_j w_j\cdot 1 = \int_a^b 1 = b - a
\;\Longrightarrow\;
\sum_j w_j = 1 .
$$

This holds for every $n$ and for both the closed and open families, because it only used
exactness on degree 0.

**The absolute sum.** Suppose $\sum_j |w_j| \le M$ for all $n$. Then for every $f$,

$$
\left|Q_n(f)\right| \le (b-a) M \max_j |f(x_j)| \le (b-a)M\|f\|_\infty ,
$$

so the rules $Q_n$ are uniformly bounded linear functionals on $C[a,b]$. They converge to
$\int_a^b$ on the dense set of polynomials, since $Q_n$ is exact on degree $n-1$. By the
Banach-Steinhaus theorem, uniform boundedness plus convergence on a dense set gives convergence
for **every** continuous $f$.

But Polya's theorem says exactly the opposite: for equally spaced Newton-Cotes rules there exists
a continuous $f$ for which $Q_n(f)$ diverges. So no such $M$ exists, and $\sum_j|w_j| \to \infty$.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc

print(f"{'nodes':>7}{'sum w':>18}{'sum |w|':>16}{'ratio to previous':>20}")
previous = None
for n in (3, 7, 11, 15, 19, 23, 27):
    w = nc.weights(n, True)
    total = float(np.sum(np.abs(w)))
    ratio = "" if previous is None else f"{total / previous:.2f}"
    print(f"{n:>7}{float(np.sum(w)):>18.12f}{total:>16.4f}{ratio:>20}")
    previous = total
```

The first column is exactly 1 at every $n$, and the second grows without any sign of stopping.

The node counts here are all odd, because **the sequence has a strong parity effect**. Taking
them consecutively instead, \(\sum_j|w_j|\) is 2.1e5 at 31 nodes and 6.0e4 at 32, so a
mixed parity table looks non monotone and is not.

### 3.1 Weights from the moment equations

Requiring exactness on $1, x, \dots, x^{n-1}$ gives a Vandermonde system in the nodes:

$$
\sum_j w_j x_j^{\,k} = \frac{1}{k+1}, \qquad k = 0, \dots, n-1 .
$$

```python
from fractions import Fraction

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc


def moment_weights_exact(n):
    """Solve the moment equations in exact rational arithmetic, by Gaussian elimination."""
    nodes = [Fraction(j, n - 1) for j in range(n)]
    A = [[nodes[j] ** k for j in range(n)] for k in range(n)]
    b = [Fraction(1, k + 1) for k in range(n)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if A[r][col] != 0)
        A[col], A[pivot] = A[pivot], A[col]
        b[col], b[pivot] = b[pivot], b[col]
        for r in range(n):
            if r == col:
                continue
            factor = A[r][col] / A[col][col]
            A[r] = [A[r][c] - factor * A[col][c] for c in range(n)]
            b[r] -= factor * b[col]
    return [b[i] / A[i][i] for i in range(n)]


def moment_weights_float(n):
    nodes = np.linspace(0.0, 1.0, n)
    A = np.stack([nodes ** k for k in range(n)])
    return np.linalg.solve(A, 1.0 / (np.arange(n) + 1.0))


print(f"{'nodes':>7}{'exact moments vs Lagrange':>28}{'float moments, rel error':>27}")
for n in (3, 5, 9, 13, 17, 21):
    exact = moment_weights_exact(n)
    lagrange = nc.exact_weights(n, True)
    same = "identical" if exact == lagrange else "DIFFER"
    reference = np.asarray([float(v) for v in lagrange])
    got = moment_weights_float(n)
    rel = float(np.max(np.abs(got - reference))) / float(np.max(np.abs(reference)))
    print(f"{n:>7}{same:>28}{rel:>27.2e}")
```

**In exact arithmetic the two routes are identical**, as they must be: both determine the unique
interpolatory rule on those nodes.

In floating point the moment route degrades, because the matrix is a Vandermonde in nodes on
$[0,1]$ and its condition number grows exponentially. This is the same finding as lesson 61's
exercise on the Vandermonde solve, in a different guise, and it has the same conclusion: the
derivation and the computation are different questions.

### 3.2 Reusing endpoints across panels and refinements

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc


class Cache:
    """Evaluate f, remembering every point already visited."""

    def __init__(self, f):
        self.f, self.seen, self.calls = f, {}, 0

    def __call__(self, x):
        t = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty(t.shape)
        for i, v in enumerate(t.ravel()):
            key = float(v)
            if key not in self.seen:
                self.seen[key] = float(self.f(np.asarray([key]))[0])
                self.calls += 1
            out.ravel()[i] = self.seen[key]
        return out


REFINEMENTS = 6           # levels of the ladder, 1, 2, 4, ... panels

exact = 1.0 - math.cos(1.0)
for n in (2, 3, 5):
    naive = shared = 0
    cache = Cache(np.sin)
    for level in range(REFINEMENTS):
        panels = 2 ** level
        counter = [0]
        value = nc.composite(cache, 0.0, 1.0, panels, n, True, counter)
        naive += counter[0]
    print(f"{n} point rule over {REFINEMENTS} refinements: {naive} evaluations without a cache, "
          f"{cache.calls} with one, saving {100 * (1 - cache.calls / naive):.1f} percent")
```

The cache removes both kinds of duplication at once: shared panel endpoints within a level, and
every point of a coarse level reappearing in the fine one.

**The saving is a constant factor, not an order**, so it does not change which rule wins. It is
worth having because function evaluations are usually the expensive part, and it costs one
dictionary.

### 3.3 The corrected trapezoid rule

Euler-Maclaurin, which lesson 63 derives, says the composite trapezoid error is

$$
T(h) - I = \frac{h^2}{12}\left[f'(b) - f'(a)\right] + O(h^4).
$$

Subtracting the leading term gives the **corrected trapezoid rule**

$$
T_c(h) = T(h) - \frac{h^2}{12}\left[f'(b) - f'(a)\right],
$$

which should be $O(h^4)$: second order gained for the price of two derivative evaluations,
independent of the panel count.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc

exact = 1.0 - math.cos(1.0)
print(f"{'panels':>8}{'trapezoid':>14}{'order':>8}{'corrected':>14}{'order':>8}")
previous_t = previous_c = None
for m in (2, 4, 8, 16, 32, 64):
    h = 1.0 / m
    t = nc.composite_shared(np.sin, 0.0, 1.0, m, 2)
    c = t - h ** 2 / 12.0 * (math.cos(1.0) - math.cos(0.0))
    et, ec = abs(t - exact), abs(c - exact)
    ot = "" if previous_t is None else f"{math.log2(previous_t / et):.2f}"
    oc = "" if previous_c is None else f"{math.log2(previous_c / ec):.2f}"
    print(f"{m:>8}{et:>14.3e}{ot:>8}{ec:>14.3e}{oc:>8}")
    previous_t, previous_c = et, ec
```

The trapezoid column has order 2 and the corrected column has order 4. **Two extra numbers, both
evaluated once, buy two orders at every panel count**, which is the best cost to benefit ratio in
this lesson.

The catch is that it needs $f'$ at the endpoints. When those are available it is the obvious
thing to do, and when they are not, lesson 63's Romberg table achieves the same gain from
function values alone.

### 3.4 A rule that is open at one end and closed at the other

For an integrand singular at $a$ only, the sensible rule uses interior nodes near $a$ and the
endpoint at $b$. The weights come from the same construction: pick the nodes, integrate the
Lagrange basis.

```python
import math
from fractions import Fraction

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc


def half_open_weights(n):
    """Nodes at (j + 1)/n for j = 0 .. n-1: open at the left, closed at the right."""
    nodes = [Fraction(j + 1, n) for j in range(n)]
    out = []
    for j in range(n):
        poly = [Fraction(1)] + [Fraction(0)] * (n - 1)
        degree, denominator = 0, Fraction(1)
        for k in range(n):
            if k == j:
                continue
            new = [Fraction(0)] * n
            for i in range(degree + 1):
                new[i + 1] += poly[i]
                new[i] -= nodes[k] * poly[i]
            poly, degree = new, degree + 1
            denominator *= (nodes[j] - nodes[k])
        out.append(sum(poly[i] / Fraction(i + 1) for i in range(n)) / denominator)
    return nodes, out


for n in (2, 3, 4):
    nodes, w = half_open_weights(n)
    print(f"{n} nodes at {[str(v) for v in nodes]}: weights {[str(v) for v in w]}, "
          f"sum {sum(w)}")

def composite_half_open(f, lo, hi, panels, n):
    nodes, w = half_open_weights(n)
    x = np.asarray([float(v) for v in nodes])
    weights = np.asarray([float(v) for v in w])
    edges = np.linspace(lo, hi, panels + 1)
    total = 0.0
    for i in range(panels):
        left, right = float(edges[i]), float(edges[i + 1])
        pts = left + (right - left) * x
        total += (right - left) * float(np.sum(weights * np.asarray(f(pts), dtype=float)))
    return total

def inverse_sqrt(x):
    return 1.0 / np.sqrt(np.asarray(x, dtype=float))

print(f"\n{'panels':>8}{'half open, 3 node':>20}{'closed Simpson':>18}")
for m in (8, 32, 128, 512):
    ho = abs(composite_half_open(inverse_sqrt, 0.0, 1.0, m, 3) - 2.0)
    closed = nc.composite_shared(inverse_sqrt, 0.0, 1.0, m, 3)
    shown = "inf" if not np.isfinite(closed) else f"{abs(closed - 2.0):.4f}"
    print(f"{m:>8}{ho:>20.6f}{shown:>18}")
```

The closed rule returns infinity at every panel count, because its leftmost node is the
singularity. The half open rule produces a number and converges: the error halves for each
fourfold refinement, so its observed order is one half.

**Look at the weights before celebrating.** At 2 nodes the right endpoint gets weight zero, so
the rule ignores the very endpoint it was built to include. At 4 nodes there is already a
negative weight, where the closed family holds out to 9 nodes and the open family to 3.

So this construction works and it is not a free lunch. It avoids the singularity, its order is
set by the singularity anyway, and its weights degrade sooner than either parent family's.
Lesson 66 does better by removing the singularity instead of stepping around it.

### 4.1 Growth of $\sum_j|w_j|$

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc

eps = np.finfo(float).eps
# odd node counts only: the sequence has a strong parity effect and mixing the two
# makes it look non monotone
counts = list(range(9, 78, 4))
closed_sums, open_sums = [], []
print(f"{'nodes':>7}{'closed sum |w|':>18}{'open sum |w|':>16}"
      f"{'closed digits lost':>21}")
for n in counts:
    c = float(np.sum(np.abs(nc.weights(n, True))))
    o = float(np.sum(np.abs(nc.weights(n, False))))
    closed_sums.append(c)
    open_sums.append(o)
    if n % 12 == 9:
        print(f"{n:>7}{c:>18.4e}{o:>16.4e}{math.log10(max(c, 1.0)):>21.2f}")

for name, values in (("closed", closed_sums), ("open", open_sums)):
    rate = float(np.polyfit(np.asarray(counts, dtype=float), np.log(values), 1)[0])
    gone = [n for n, v in zip(counts, values) if v * eps > 1.0]
    print(f"\n{name}: sum |w| grows like {math.exp(rate):.4f}^n")
    print(f"{name}: eps * sum |w| passes 1, so no digit survives, at n = "
          f"{gone[0] if gone else 'beyond ' + str(counts[-1])}")
```

The growth is **geometric in $n$**, with a measured base of $1.86$ for the closed family and
$1.94$ for the open one. Close to 2 and not equal to it, and the difference matters for
exercise 5.1.

The open family is worse at every count, by a factor that is itself growing, which is the price
of pushing the nodes inward. It runs out of digits first: **no digit of a double precision
result survives past 61 open nodes, or 69 closed ones.**

Long before either, the rule is useless in practice. Twenty one closed nodes already costs two
and a half digits, and that is on an integrand smooth enough for the rule to be worth
considering at all.

### 4.2 Convergence order against smoothness

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc

print(f"{'a':>6}{'exact':>12}{'trapezoid':>12}{'Simpson':>10}{'Boole':>10}  limited by")
for a in (0.5, 1.5, 2.5, 3.5, 4.5, 6.5):
    def f(x, a=a):
        return np.abs(np.asarray(x, dtype=float)) ** a

    exact = 2.0 / (a + 1.0)
    row = []
    for n in (2, 3, 5):
        out = nc.convergence(f, exact, -1.0, 1.0, n, True, [4, 8, 16, 32, 64, 128, 256])
        row.append(out["fitted_order"])
    # each rule is smoothness limited once its own order exceeds a + 1
    labels = " ".join("s" if p >= a + 0.6 else "r" for p in (2, 4, 6))
    print(f"{a:>6.1f}{exact:>12.6f}" + "".join(f"{v:>12.3f}" if i == 0 else f"{v:>10.3f}"
                                               for i, v in enumerate(row))
          + f"  {labels}")
print("\nr = limited by the rule's own order, s = limited by the smoothness of |x|^a")
```

The pattern is clear, and it is per rule rather than global.

At $a = 0.5$ all three rules read about $1.5$: the singularity limits everything and paying for
Boole buys nothing. At $a = 1.5$ trapezoid reaches its own order of 2 while Simpson and Boole
are both stuck at $2.5$. By $a = 6.5$ all three have reached their own orders, 2, 4 and 6.

**The observed order is the smaller of the rule's order and $a + 1$.** Below the crossover the
rule is wasted; above it the smoothness is not the constraint. Choosing a rule without knowing
which side you are on is choosing at random.

### 4.3 The fitted order over a sliding window

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc


def runge(x):
    return 1.0 / (1.0 + 25.0 * np.asarray(x, dtype=float) ** 2)


exact = 2.0 * math.atan(5.0) / 5.0
panels = np.asarray([2 ** k for k in range(1, 13)], dtype=float)
errors = np.asarray([abs(nc.composite_shared(runge, -1.0, 1.0, int(m), 3) - exact)
                     for m in panels])
print(f"{'window':>22}{'fitted order':>15}")
for i in range(errors.size - 2):
    window = slice(i, i + 3)
    if float(np.min(errors[window])) < 1e-14:
        print(f"{f'{int(panels[i])} to {int(panels[i + 2])}':>22}{'at the floor':>15}")
        continue
    order = float(-np.polyfit(np.log(panels[window]), np.log(errors[window]), 1)[0])
    print(f"{f'{int(panels[i])} to {int(panels[i + 2])}':>22}{order:>15.3f}")
print("\ncomposite Simpson is order 4")
```

The fitted order is meaningless at the coarse end, settles onto 4 in the middle, and becomes
meaningless again at the fine end where the error reaches the floor.

**There is a window and it is narrow.** Reporting a convergence order without saying which panel
counts produced it is reporting a number that can be made almost anything. This is the same point
lesson 62 makes in section 4, and the sliding window is the honest way to display it.

### 5.1 Why the weights go negative

The quadrature weights are integrals of the Lagrange basis:
$w_j = \frac{1}{b-a}\int_a^b \ell_j$. The Lebesgue function of the node set is
$\Lambda(x) = \sum_j |\ell_j(x)|$, and

$$
\sum_j |w_j| = \frac{1}{b-a}\sum_j \left|\int_a^b \ell_j\right|
\le \frac{1}{b-a}\int_a^b \sum_j |\ell_j| = \frac{1}{b-a}\int_a^b \Lambda .
$$

So $\sum_j|w_j|$ is bounded by the average Lebesgue function, and the two grow together.

For equally spaced nodes lesson 46 gives $\Lambda_n \sim \frac{2^n}{e\, n \log n}$, exponential
in $n$. **The exponential growth of the Lebesgue constant and the exponential growth of
$\sum_j|w_j|$ are the same phenomenon**, and both come from the same source: the Lagrange basis
polynomials for equally spaced nodes oscillate with enormous amplitude near the ends.

The sign change at $n = 9$ is where that oscillation first becomes large enough that some
$\int \ell_j$ comes out negative. Before that the basis polynomials, while not positive, are
positive enough on average.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc

print(f"{'nodes':>7}{'sum |w|':>14}{'2^n / (e n log n)':>22}{'ratio':>12}")
for n in (9, 13, 17, 21, 25, 29):
    total = float(np.sum(np.abs(nc.weights(n, True))))
    lebesgue = 2.0 ** n / (np.e * n * np.log(n))
    print(f"{n:>7}{total:>14.4e}{lebesgue:>22.4e}{total / lebesgue:>12.6f}")
```

**The ratio is not constant: it falls steadily, from $0.15$ at 9 nodes to $0.03$ at 29.**

So the two do not grow at the same rate. The Lebesgue constant grows like $2^n$ and the absolute
weight sum like $1.85^n$, measured in exercise 4.1. The bound derived above is real and it is
**not tight**, and the inequality already said why: it replaced the absolute value of an
integral by the integral of an absolute value, discarding all the cancellation inside each
basis polynomial, and that cancellation is substantial.

The qualitative conclusion survives intact. Both grow geometrically, from the same cause, and
the sign change at 9 nodes is where the oscillation of the equally spaced Lagrange basis first
overwhelms the averaging. What does not survive is the quantitative claim that the two grow at
the same rate.

### 5.2 Clenshaw-Curtis quadrature

Integrating the interpolant through **Chebyshev** points instead of equally spaced ones gives
Clenshaw-Curtis. The construction is identical; only the nodes change, and everything that was
wrong becomes right.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq


def clenshaw_curtis(n):
    """Nodes and weights on [-1, 1] with n+1 Chebyshev extreme points, via the DCT formula."""
    if n < 1:
        raise ValueError("need at least one interval")
    theta = np.pi * np.arange(n + 1) / n
    x = np.cos(theta)
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    for k in range(2, n, 2):
        v -= 2.0 * np.cos(k * theta[1:n]) / (k ** 2 - 1)
    if n % 2 == 0:
        v -= np.cos(n * theta[1:n]) / (n ** 2 - 1)
        w[0] = w[n] = 1.0 / (n ** 2 - 1)
    else:
        w[0] = w[n] = 1.0 / (n ** 2)
    w[1:n] = 2.0 * v / n
    return x[::-1], w[::-1]


print(f"{'points':>8}{'sum w':>12}{'positive':>11}{'measured degree':>17}{'margin':>12}")
for n in (4, 8, 16, 32, 64):
    x, w = clenshaw_curtis(n)
    degree = -1
    for k in range(0, 3 * n + 4):
        got = float(np.sum(w * x ** k))
        want = 0.0 if k % 2 else 2.0 / (k + 1.0)
        if abs(got - want) > 1e-10 * max(abs(want), 1.0):
            break
        degree = k
    k = degree + 1
    got = float(np.sum(w * x ** k))
    want = 0.0 if k % 2 else 2.0 / (k + 1.0)
    scale = float(np.sum(np.abs(w * x ** k)))
    margin = abs(got - want) / max(abs(want), scale, 1e-300)
    print(f"{n + 1:>8}{float(np.sum(w)):>12.8f}{str(bool(np.all(w > 0))):>11}"
          f"{degree:>17}{margin:>12.1e}")
print("\na margin near 1e-16 means the walk stopped at the roundoff floor, "
      "not at a monomial the rule misses")
```

**The weights are positive at every size**, and the weight sum is exactly 2, so Clenshaw-Curtis
has the stability that Newton-Cotes loses at 9 nodes.

The degree column needs the same warning as lesson 65 section 4. Up to 17 points it reads $n$,
which is the theoretical answer and half of Gauss's $2n-1$. Past that it reads 45 and 179,
which are not degrees.

The margin column says why. It is the rule's relative error on the first monomial the walk
rejected, and it collapses from $6.7\times10^{-2}$ at 5 points to $3\times10^{-7}$ at 17 and
$10^{-8}$ beyond. Once that margin is near the tolerance, the walk steps over the true boundary
and keeps going through monomials the rule misses by less than the test can see.

**The fix is the one lesson 65 uses: report the margin alongside the degree**, so a reader can
see when the number has stopped meaning anything.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq


def clenshaw_curtis(n):
    theta = np.pi * np.arange(n + 1) / n
    x = np.cos(theta)
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    for k in range(2, n, 2):
        v -= 2.0 * np.cos(k * theta[1:n]) / (k ** 2 - 1)
    if n % 2 == 0:
        v -= np.cos(n * theta[1:n]) / (n ** 2 - 1)
        w[0] = w[n] = 1.0 / (n ** 2 - 1)
    else:
        w[0] = w[n] = 1.0 / (n ** 2)
    w[1:n] = 2.0 * v / n
    return x[::-1], w[::-1]


cases = (("sin(3x)", lambda t: np.sin(3.0 * t), 0.0),
         ("1/(1+25x^2)", lambda t: 1.0 / (1.0 + 25.0 * t ** 2),
          2.0 * math.atan(5.0) / 5.0),
         ("exp(x)", np.exp, math.e - 1.0 / math.e))
print(f"{'integrand':>16}{'points':>8}{'Clenshaw-Curtis':>19}{'Gauss':>14}{'ratio':>10}")
for name, f, exact in cases:
    for points in (9, 17, 33):
        xc, wc = clenshaw_curtis(points - 1)
        cc = abs(float(np.sum(wc * f(xc))) - exact)
        xg, wg = gq.nodes_and_weights(points)
        g = abs(float(np.sum(wg * f(xg))) - exact)
        ratio = cc / g if g > 0 else float("inf")
        print(f"{name:>16}{points:>8}{cc:>19.3e}{g:>14.3e}{ratio:>10.2f}")
```

**Gauss has twice the degree of precision and the results do not show twice the accuracy.** On
$\sin(3x)$ Clenshaw-Curtis is the more accurate of the two at every size. On Runge's function
Gauss wins by a steady factor of $1.85$. On $e^x$ at 9 points Gauss wins by a factor of 7680,
and at 17 points Clenshaw-Curtis is exact and Gauss is not.

So the honest summary is that they are comparable, with the winner depending on the integrand
and the size, and with one case here where Gauss is far ahead. Nothing about that pattern is
predicted by the ratio of their degrees.

The explanation is that for an analytic integrand both converge geometrically, with rates set
by the same Bernstein ellipse from lesson 56, and **the degree of precision is the wrong
statistic for predicting the outcome**. Clenshaw-Curtis also has an advantage Gauss does not:
its node sets are nested, so refining reuses every previous evaluation.

### 5.3 The error term is not a bound

The classical error term is

$$
I - Q = C h^{d+2} f^{(d+1)}(\xi), \qquad \xi \in (a,b) \text{ unknown}.
$$

**What can be concluded.** The scaling. If $f^{(d+1)}$ is continuous, the error is $O(h^{d+2})$
on one panel and $O(h^{d+1})$ composite, and halving $h$ multiplies it by a known factor. That is
what the fitted orders in this lesson measure and confirm.

**What cannot.** A numerical bound, unless you have a bound on $f^{(d+1)}$ over the whole
interval. Replacing $f^{(d+1)}(\xi)$ by $f^{(d+1)}$ at a convenient point, usually the midpoint,
is the standard shortcut and it is not justified by anything.

Here is an integrand where that shortcut fails badly.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc


def spiky(x):
    """cos(16 pi x): every node of a 4 or 8 panel grid lands on a peak."""
    t = np.asarray(x, dtype=float)
    return np.cos(2.0 * math.pi * 8.0 * t)


exact = 0.0                                   # integral of cos(16 pi x) over [0, 1]
print(f"{'panels':>8}{'true error':>14}{'midpoint estimate':>20}{'estimate / true':>18}")
for m in (4, 8, 16):
    h = 1.0 / m
    t = nc.composite_shared(spiky, 0.0, 1.0, m, 2)
    true_error = exact - t
    midpoints = (np.arange(m) + 0.5) * h
    second = -(2.0 * math.pi * 8.0) ** 2 * np.cos(2.0 * math.pi * 8.0 * midpoints)
    estimate = -h ** 2 / 12.0 * float(np.sum(second)) * h
    ratio = estimate / true_error if true_error != 0 else float("inf")
    print(f"{m:>8}{true_error:>14.6e}{estimate:>20.6e}{ratio:>18.4f}")
```

Read the true error column first. At 4 and 8 panels it is exactly $-1$, meaning **the trapezoid
rule returns 1 when the answer is 0.** Every node lands on a peak of the cosine, so the rule
sees a constant function of height 1 and integrates that perfectly. That is aliasing, in the
sense of lesson 58, and no error estimate built from the same samples can detect it.

At 16 panels the nodes alternate between $+1$ and $-1$ and the rule is **exact by accident**,
error 0, while the estimate reports $-10^{-15}$ and the ratio is meaningless.

The midpoint estimate is wrong by a factor of $-13$ at 4 panels and $+3.3$ at 8: wrong sign,
wrong magnitude, and no pattern in $h$ at all.

**The lesson is that the error formula describes an asymptotic regime and says nothing outside
it.** Once $h$ is small compared with every feature of $f$ the formula is reliable; before that
it can be wrong in either direction and by any amount. Deciding whether you are in the asymptotic
regime is the actual problem, and the only honest tools for it are refinement and comparison,
which is what lessons 63 and 64 are about.

## Lesson 63, Richardson Extrapolation, Romberg Integration and Euler-Maclaurin

### 1.1 What even powers buy

Richardson extrapolation removes the leading term of an error expansion. Given
$E(h) = a_1 h^{q} + a_2 h^{q'} + \cdots$ with $q' > q$, the combination

$$
\frac{2^{q} E(h/2) - E(h)}{2^{q} - 1}
$$

kills the $h^q$ term and leaves $h^{q'}$. **So what one step gains is $q' - q$, the gap to the
next surviving term.**

For a general expansion in all powers, $q' = q + 1$ and each step gains one order. For an
expansion in even powers only, $q' = q + 2$ and each step gains two.

Euler-Maclaurin gives the trapezoid rule an expansion in $h^2, h^4, h^6, \dots$, so the gap is
always 2, and the Romberg table's column $j$ has order $2j + 2$ rather than $j + 2$.

**The practical size of that is large.** Reaching order 8 takes three Richardson steps on an even
expansion and six on a general one, and each step costs a refinement, so the difference is a
factor of 8 in the finest panel count.

The second consequence is that the divisor is $4^j$ and not $2^j$. Using $2^j$ removes nothing:
it forms a combination that still contains the $h^2$ term with a nonzero coefficient, so the
column has the same order as the one before it and the table stops working while looking fine.

### 1.2 What endpoint-only coefficients buy

Every Euler-Maclaurin coefficient is $\frac{B_{2k}}{(2k)!}\left[f^{(2k-1)}(b) -
f^{(2k-1)}(a)\right]$. The interior of $f$ appears **nowhere**.

Two things follow.

**A correction is cheap.** Subtracting the leading term costs two derivative evaluations,
independent of the panel count, and gains two orders at every panel count. That is exercise 3.3
of lesson 62, where the corrected trapezoid rule measured order 4.

**A periodic integrand has no error series at all.** If $f$ and all its derivatives take the
same values at $a$ and $b$, every bracket is exactly zero, so every term vanishes and the
trapezoid error is smaller than any power of $h$. That is section 5 of the lesson, and it is why
the worst rule of lesson 62 is the right rule for anything on a circle.

The same statement covers the real line: a function decaying fast enough at $\pm\infty$ has all
its endpoint terms zero in the limit, which is exercise 5.1 and the basis of lesson 66's double
exponential rule.

### 1.3 Why Romberg hurts on a periodic integrand

Romberg assumes an error expansion in even powers of $h$ and forms a linear combination designed
to cancel its leading terms. On a periodic integrand **that expansion is identically zero**, so
there is nothing to cancel and the combination is being applied to values whose errors have a
completely different structure.

The damage is concrete. The diagonal entry $R_{k,k}$ is a linear combination of the whole
trapezoid ladder $T(h), T(h/2), \dots, T(h/2^k)$ including the coarsest levels, and on a periodic
integrand the coarse levels are still badly wrong: two points and four points do not resolve the
function. The combination mixes those large errors into an answer that the finest level alone had
already computed correctly.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import romberg as rb


def periodic(x):
    return np.exp(np.cos(np.asarray(x, dtype=float)))


i0, term, k = 0.0, 1.0, 0
while i0 + term != i0:
    i0 += term
    k += 1
    term = 1.0 / (4.0 ** k * math.factorial(k) ** 2)
exact = 2.0 * math.pi * i0

out = rb.periodic_against_romberg(periodic, exact, 0.0, 2.0 * math.pi, 6)
print(f"{'evaluations':>13}{'trapezoid alone':>19}{'Romberg diagonal':>20}{'ratio':>12}")
for ev, plain, diag in zip(out["evaluations"], out["periodic_trapezoid_error"],
                           out["romberg_diagonal_error"]):
    ratio = diag / plain if plain > 0 else float("inf")
    print(f"{ev:>13}{plain:>19.3e}{diag:>20.3e}{ratio:>12.2e}")
```

**The general principle is that an accelerator encodes an assumption about the error**, and
applying it where the assumption is false is not neutral. It is the same trap as using Aitken's
$\Delta^2$ on a sequence that does not converge linearly, or a secant step on a function with a
double root.

The defence is to know what your error looks like before accelerating it, and the diagnostic is
cheap: compare the accelerated answer against the unaccelerated one at equal cost, which is what
the table above does.

### 1.4 A good answer with `converged=False`

**What to conclude: the routine's central assumption does not hold for this integrand.**

The stopping test compares two successive diagonal entries. Running to the level limit means
those never agreed, which happens when the columns are not gaining the orders the even power
expansion promises, which happens when $f$ lacks the endpoint derivatives Euler-Maclaurin names.

The answer being good anyway is luck, not evidence. It came from the first column, which is the
plain composite trapezoid rule and converges for any continuous $f$, and it came at the cost of a
million evaluations to reach an accuracy the trapezoid rule alone would have reached with the
same million.

**What to do:**

1. **Do not trust the level count next time.** A different integrand with the same symptom may
   return a bad answer just as quietly.
2. **Look at the integrand's endpoints.** A singularity, a kink, or an infinite derivative there
   is the usual cause.
3. **Change the method rather than the tolerance.** Substitute the singularity away (lesson 66
   section 2), use a modified extrapolation power (section 8 of this lesson), or use a rule that
   does not evaluate at the endpoints at all (lesson 66 section 3).

Raising `max_levels` is the one thing that certainly does not help: each level doubles the cost
and gains nothing, because the columns are not converging.

### 2.1 Euler-Maclaurin by parts

Work on $[0,1]$ and rescale afterwards. The Bernoulli polynomials $B_n(x)$ satisfy
$B_n'(x) = nB_{n-1}(x)$, $B_0 = 1$, and $\int_0^1 B_n = 0$ for $n \ge 1$.

Start from $\int_0^1 f = \int_0^1 f(x) B_0(x)\,dx$ and integrate by parts against
$B_1(x) = x - \tfrac12$, whose derivative is $B_0$:

$$
\int_0^1 f = \left[f(x)B_1(x)\right]_0^1 - \int_0^1 f'(x)B_1(x)\,dx
= \frac{f(0)+f(1)}{2} - \int_0^1 f' B_1 .
$$

**The first term is the trapezoid rule on one panel.** Everything after it is the error.

Continue, using $B_2' = 2B_1$:

$$
\int_0^1 f' B_1 = \left[\frac{f'(x)B_2(x)}{2}\right]_0^1 - \frac12\int_0^1 f'' B_2
= \frac{B_2}{2}\left[f'(1) - f'(0)\right] - \frac12\int_0^1 f''B_2 ,
$$

since $B_2(0) = B_2(1) = B_2$. Repeating $2m$ times and using $B_{2k+1}(0) = B_{2k+1}(1) = 0$ for
$k \ge 1$, which kills every odd term, gives

$$
T - \int_0^1 f = \sum_{k=1}^{m}\frac{B_{2k}}{(2k)!}\left[f^{(2k-1)}(1) - f^{(2k-1)}(0)\right]
+ R_m .
$$

**Summing over panels is where the interior cancels.** On panel $i$ the bracket is
$f^{(2k-1)}(x_{i+1}) - f^{(2k-1)}(x_i)$, and summing over $i$ telescopes to
$f^{(2k-1)}(b) - f^{(2k-1)}(a)$. Every interior node appears twice with opposite signs. That
telescoping is why the coefficients depend only on the endpoints, and it is the whole reason for
sections 5 and 6 of the lesson.

Rescaling a panel of width $h$ contributes $h^{2k}$ to term $k$.

### 2.2 Odd Bernoulli numbers vanish

From the generating function

$$
\frac{t}{e^t - 1} = \sum_{n\ge 0}\frac{B_n}{n!}t^n ,
$$

add $t/2$ to both sides:

$$
\frac{t}{e^t-1} + \frac{t}{2}
= \frac{t}{2}\cdot\frac{2 + e^t - 1}{e^t - 1}
= \frac{t}{2}\cdot\frac{e^{t/2} + e^{-t/2}}{e^{t/2} - e^{-t/2}}
= \frac{t}{2}\coth\frac{t}{2}.
$$

The right hand side is **even** in $t$: replacing $t$ by $-t$ leaves $\coth$ odd and the prefactor
odd, so the product is even.

The left hand side is $\sum_n \frac{B_n}{n!}t^n + \frac{t}{2}$, whose odd part is
$\left(B_1 + \tfrac12\right)t + \sum_{k\ge1}\frac{B_{2k+1}}{(2k+1)!}t^{2k+1}$.

An even function has zero odd part, so $B_1 = -\tfrac12$ and $B_{2k+1} = 0$ for all $k \ge 1$.

```python
from fractions import Fraction
import sys
sys.path.insert(0, "src")
from nalib import romberg as rb

odd = [(n, rb.bernoulli(n)) for n in range(3, 22, 2)]
print("odd Bernoulli numbers past B_1:")
print("  " + ", ".join(f"B_{n} = {v}" for n, v in odd))
assert all(v == Fraction(0) for _, v in odd)
print(f"\nB_1 = {rb.bernoulli(1)}, exactly -1/2 as the generating function requires")
print("even ones, for contrast:")
print("  " + ", ".join(f"B_{n} = {rb.bernoulli(n)}" for n in range(2, 15, 2)))
```

### 2.3 $R_{k,1}$ is composite Simpson

Let $T_k$ be the composite trapezoid rule on $2^k$ panels of width $h_k = (b-a)/2^k$, and write
the nodes of the finer level as $x_0, x_1, \dots, x_{2^k}$.

$$
R_{k,1} = T_k + \frac{T_k - T_{k-1}}{3} = \frac{4T_k - T_{k-1}}{3}.
$$

Take one panel of the coarse level, spanning $[x_{2i}, x_{2i+2}]$ with width $2h_k$. Its
contribution to $T_{k-1}$ is $h_k\left(f_{2i} + f_{2i+2}\right)$, and its contribution to $T_k$ is
$\frac{h_k}{2}\left(f_{2i} + 2f_{2i+1} + f_{2i+2}\right)$. So the contribution to $R_{k,1}$ is

$$
\frac{1}{3}\left[4\cdot\frac{h_k}{2}(f_{2i} + 2f_{2i+1} + f_{2i+2})
- h_k(f_{2i} + f_{2i+2})\right]
= \frac{h_k}{3}\left(f_{2i} + 4f_{2i+1} + f_{2i+2}\right).
$$

**That is exactly Simpson's rule on that panel.** Summing over $i$ gives composite Simpson on
$2^{k-1}$ Simpson panels, which is what the lesson measures to $5.6\times10^{-17}$.

The same algebra one level up gives composite Boole for $R_{k,2}$, with the $[7, 32, 12, 32, 7]/45$
weights appearing from the combination $\frac{16 R_{k,1} - R_{k-1,1}}{15}$.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc
from nalib import romberg as rb

for name, f, a, b, exact in (("sin", np.sin, 0.0, 1.0, 1.0 - math.cos(1.0)),
                             ("exp", np.exp, -1.0, 2.0, math.exp(2.0) - math.exp(-1.0))):
    out = rb.column_is_a_newton_cotes_rule(f, a, b, 6)
    print(f"{name}: column 1 vs Simpson {float(np.max(out['simpson_relative_gap'])):.2e}, "
          f"column 2 vs Boole {float(np.max(out['boole_relative_gap'])):.2e}")
    assert out["column_1_is_simpson"] and out["column_2_is_boole"]
```

### 2.4 Column $j$ has order $2j+2$

By induction. Write $E_{k,j} = R_{k,j} - I$.

**Base case.** $E_{k,0}$ is the composite trapezoid error, which by Euler-Maclaurin is
$\sum_{i\ge1} c_i h_k^{2i}$ with $c_i$ independent of $k$. Its leading order is $h_k^2 = 2j+2$
with $j = 0$.

**Inductive step.** Suppose $E_{k,j-1} = \sum_{i \ge j} c^{(j-1)}_i h_k^{2i}$, so the leading
order is $h_k^{2j}$. Then, since $h_k = h_{k-1}/2$,

$$
E_{k,j} = E_{k,j-1} + \frac{E_{k,j-1} - E_{k-1,j-1}}{4^j - 1}.
$$

Substitute the expansions. For the term $i = j$, $E_{k-1,j-1}$ contributes
$c^{(j-1)}_j (2h_k)^{2j} = 4^j c^{(j-1)}_j h_k^{2j}$, so

$$
c^{(j-1)}_j h_k^{2j}\left[1 + \frac{1 - 4^j}{4^j - 1}\right] = 0 .
$$

**The $h^{2j}$ term is annihilated exactly**, which is what the divisor $4^j - 1$ was chosen for.
For $i > j$ the same substitution gives a nonzero multiple of $h_k^{2i}$, so the new leading order
is $h_k^{2(j+1)} = h_k^{2j+2}$.

That completes the induction, and it also shows why the divisor cannot be $2^j - 1$: with that
choice the bracket is $1 + (1 - 4^j)/(2^j-1)$, which is not zero, and the $h^{2j}$ term survives.

### 2.5 The trapezoid error for $\int_0^1\sqrt{x}$

Euler-Maclaurin does not apply: $f'(x) = \tfrac12 x^{-1/2}$ is infinite at 0.

Split the integral at the first node. On $[h, 1]$ the function is smooth and Euler-Maclaurin gives
an $O(h^2)$ error. The whole difficulty is the first panel, $[0, h]$, where

$$
\int_0^h \sqrt{x}\,dx = \frac{2}{3}h^{3/2},
\qquad
T_{\text{first panel}} = \frac{h}{2}\left(0 + \sqrt{h}\right) = \frac{1}{2}h^{3/2}.
$$

The local error is $\left(\tfrac23 - \tfrac12\right)h^{3/2} = \tfrac16 h^{3/2}$, and unlike a
smooth panel it does not shrink faster than the panel count grows, because there is only one such
panel. So

$$
T(h) - I = -\frac{1}{6}h^{3/2} + O(h^2),
$$

giving order $\tfrac32$.

The full expansion is known and it is the reason section 8 works:

$$
T(h) - I = \zeta(-\tfrac12)h^{3/2} + \sum_{k\ge1} c_k h^{2k},
$$

so the powers present are $\tfrac32, 2, 4, 6, \dots$ **A modified table with $p = \tfrac12$
clears the lattice $\tfrac12, 1, \tfrac32, 2, \tfrac52, \dots$, which contains every power that
is actually there**, which is why $p = \tfrac12$ wins in section 8 rather than $p = \tfrac32$.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc


def root(x):
    return np.sqrt(np.asarray(x, dtype=float))


exact = 2.0 / 3.0
print(f"{'panels':>8}{'error':>15}{'error / h^1.5':>16}{'observed order':>17}")
previous = None
for m in (16, 64, 256, 1024, 4096):
    h = 1.0 / m
    e = nc.composite_shared(root, 0.0, 1.0, m, 2) - exact
    order = "" if previous is None else f"{math.log(previous / abs(e)) / math.log(4.0):.4f}"
    print(f"{m:>8}{e:>15.6e}{e / h ** 1.5:>16.6f}{order:>17}")
    previous = abs(e)
print(f"\nzeta(-1/2) = -0.2078862250, the predicted coefficient")
```

The measured coefficient converges to $-0.20789$, which is $\zeta(-\tfrac12)$, and the observed
order to $1.5$.

### 3.1 A column based stopping test

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import romberg as rb


def to_tolerance_by_column(f, lo, hi, tol=1e-10, max_levels=20):
    """Stop when the last two entries of the CURRENT ROW agree, not the diagonal."""
    a, b = float(lo), float(hi)
    h = b - a
    ends = np.asarray([a, b], dtype=float)
    values = np.asarray(f(ends), dtype=float)
    rows = [np.asarray([0.5 * h * float(values[0] + values[1])])]
    for level in range(1, max_levels + 1):
        h *= 0.5
        mid = a + h * (2.0 * np.arange(2 ** (level - 1)) + 1.0)
        trap = 0.5 * float(rows[-1][0]) + h * float(np.sum(np.asarray(f(mid), dtype=float)))
        row = np.empty(level + 1)
        row[0] = trap
        for j in range(1, level + 1):
            row[j] = row[j - 1] + (row[j - 1] - rows[-1][j - 1]) / (4.0 ** j - 1.0)
        rows.append(row)
        if level >= 2:
            gap = abs(float(row[-1]) - float(row[-2]))
            if gap <= tol * max(abs(float(row[-1])), 1.0):
                return {"value": float(row[-1]), "levels_used": level,
                        "converged": True, "gap": gap}
    return {"value": float(rows[-1][-1]), "levels_used": max_levels,
            "converged": False, "gap": float("nan")}


def root(x):
    return np.sqrt(np.asarray(x, dtype=float))


cases = (("sin", np.sin, 1.0 - math.cos(1.0)), ("exp", np.exp, math.e - 1.0),
         ("sqrt(x)", root, 2.0 / 3.0),
         ("x^(1/3)", lambda t: np.asarray(t, dtype=float) ** (1.0 / 3.0), 0.75))
print(f"{'integrand':>10}{'diagonal test':>28}{'column test':>28}")
print(f"{'':>10}{'levels':>10}{'error':>11}{'ok':>7}{'levels':>10}{'error':>11}{'ok':>7}")
for name, f, exact in cases:
    d = rb.to_tolerance(f, 0.0, 1.0, 1e-10, 14)
    c = to_tolerance_by_column(f, 0.0, 1.0, 1e-10, 14)
    print(f"{name:>10}{d['levels_used']:>10}{abs(d['value'] - exact):>11.1e}"
          f"{str(d['converged']):>7}{c['levels_used']:>10}"
          f"{abs(c['value'] - exact):>11.1e}{str(c['converged']):>7}")
```

**The column test fires where the diagonal test does not, and returns a worse answer when it
does.** On the square root it stops at level 9 with an error of $5.9\times10^{-6}$; the diagonal
test runs to level 14 and reaches $3.3\times10^{-8}$, still refusing to claim convergence.

That is the point. Consecutive columns of a row agree precisely when the extrapolation is
doing nothing, because then each column returns nearly what the one before it did. **A test
that fires on a failed method is worse than one that never fires**: the first hides the
failure behind a confident flag, the second reports it.

The diagonal test's refusal to fire is a feature, and the level count is the signal to read.

### 3.2 Euler-Maclaurin as a summation formula

Read Euler-Maclaurin backwards. It relates a sum to an integral, so it evaluates sums that are
too long to add:

$$
\sum_{k=M}^{N}g(k) = \int_M^N g + \frac{g(M)+g(N)}{2}
+ \sum_{j\ge1}\frac{B_{2j}}{(2j)!}\left[g^{(2j-1)}(N) - g^{(2j-1)}(M)\right].
$$

**Anchoring at $M = 1$ does not work, and the reason is the whole subject of exercise 5.3.**

```python
import math

import sys
sys.path.insert(0, "src")
from nalib import romberg as rb


def harmonic_from_one(N, terms=6):
    """The naive version: apply Euler-Maclaurin all the way down to k = 1."""
    total = math.log(N) + 0.5 * (1.0 + 1.0 / N)
    for j in range(1, terms + 1):
        coefficient = float(rb.bernoulli(2 * j)) / math.factorial(2 * j)
        derivative = -math.factorial(2 * j - 1)          # (2j-1)th derivative of 1/x
        total += coefficient * derivative * (N ** (-2 * j) - 1.0)
    return total


print(f"{'N':>10}{'Euler-Maclaurin from 1':>26}{'direct sum':>22}{'gap':>12}")
for N in (10, 100, 10000, 1000000):
    em = harmonic_from_one(N)
    direct = math.fsum(1.0 / k for k in range(1, N + 1))
    print(f"{N:>10}{em:>26.15f}{direct:>22.15f}{abs(em - direct):>12.1e}")
print("\nthe gap is the same 1.6e-02 at every N, so it is a constant, not a truncation error")
```

The error is a **constant** $1.6\times10^{-2}$, identical at $N = 10$ and at $N = 10^6$. That is
not a failure of the $N$ dependence, which is exact; it is that the series
$\tfrac12 + \sum_j \frac{B_{2j}}{2j}$, which is what the formula produces in place of Euler's
constant $\gamma$, **diverges**. Six terms give $0.5613$ and $\gamma = 0.5772$.

The fix is to sum the first few terms directly and start Euler-Maclaurin above them, so no
divergent constant is ever needed.

```python
import math

import sys
sys.path.insert(0, "src")
from nalib import romberg as rb


def harmonic(N, anchor=10, terms=6):
    """Direct sum below `anchor`, Euler-Maclaurin above it. No constant required."""
    if N <= anchor:
        return math.fsum(1.0 / k for k in range(1, N + 1))
    total = math.fsum(1.0 / k for k in range(1, anchor))
    total += math.log(N / anchor) + 0.5 * (1.0 / anchor + 1.0 / N)
    for j in range(1, terms + 1):
        coefficient = float(rb.bernoulli(2 * j)) / math.factorial(2 * j)
        derivative = -math.factorial(2 * j - 1)
        total += coefficient * derivative * (N ** (-2 * j) - anchor ** (-2 * j))
    return total


print(f"{'N':>10}{'Euler-Maclaurin from 10':>27}{'direct sum':>22}{'gap':>12}")
for N in (100, 1000, 100000, 1000000):
    em = harmonic(N)
    direct = math.fsum(1.0 / k for k in range(1, N + 1))
    print(f"{N:>10}{em:>27.15f}{direct:>22.15f}{abs(em - direct):>12.1e}")

gamma = 0.5772156649015329
for N in (10 ** 9, 10 ** 12):
    em = harmonic(N)
    print(f"\nN = 10^{len(str(N)) - 1}: {em:.15f}")
    print(f"log(N) + gamma + 1/(2N) : {math.log(N) + gamma + 0.5 / N:.15f}")
    print(f"gap                     : {abs(em - math.log(N) - gamma - 0.5 / N):.2e}")
```

Now it agrees with the direct sum to $10^{-15}$, and at $N = 10^{12}$, where the direct sum is
not worth attempting, it agrees with $\log N + \gamma + \tfrac{1}{2N}$ to the same accuracy.

**Nine terms of arithmetic in place of a trillion additions.** This is Euler-Maclaurin used in
the direction it was invented for, and it is how $\gamma$ was first computed to sixteen digits.

### 3.3 The corrected trapezoid rule

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc

cases = (("sin over [0,1]", np.sin, np.cos, 0.0, 1.0, 1.0 - math.cos(1.0)),
         ("exp over [0,2]", np.exp, np.exp, 0.0, 2.0, math.exp(2.0) - 1.0))
for name, f, fp, a, b, exact in cases:
    print(f"\n{name}")
    print(f"{'panels':>8}{'trapezoid':>14}{'order':>8}{'corrected':>14}{'order':>8}")
    pt = pc = None
    for m in (2, 4, 8, 16, 32, 64):
        h = (b - a) / m
        t = nc.composite_shared(f, a, b, m, 2)
        c = t - h ** 2 / 12.0 * (float(fp(b)) - float(fp(a)))
        et, ec = abs(t - exact), abs(c - exact)
        ot = "" if pt is None else f"{math.log2(pt / et):.2f}"
        oc = "" if pc is None else f"{math.log2(pc / ec):.2f}"
        print(f"{m:>8}{et:>14.3e}{ot:>8}{ec:>14.3e}{oc:>8}")
        pt, pc = et, ec
```

Order 2 becomes order 4, at a cost of two derivative evaluations that does not grow with the
panel count.

**This is the cheapest acceleration in Part 9.** Its limitation is that it needs $f'$ at the
endpoints in closed form. Romberg gets the same gain from function values alone, at the cost of
building a whole table.

### 3.4 Romberg on the midpoint rule

The midpoint rule also has an even power expansion, because it is symmetric on each panel:

$$
M(h) - I = -\frac{h^2}{24}\left[f'(b) - f'(a)\right] + O(h^4).
$$

So the same $4^j$ extrapolation works. **What changes is the refinement ratio.** Halving the
midpoint panel width does not reuse any old points, since the new midpoints are different from
the old ones. Trisecting does: the midpoints of a third-width grid include the old midpoints.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")


def midpoint(f, lo, hi, panels):
    h = (hi - lo) / panels
    x = lo + h * (np.arange(panels) + 0.5)
    return h * float(np.sum(np.asarray(f(x), dtype=float)))


def midpoint_romberg(f, lo, hi, levels, ratio=3):
    """Romberg on the midpoint rule, refining by `ratio` so points are reused."""
    R = np.zeros((levels + 1, levels + 1))
    for k in range(levels + 1):
        R[k, 0] = midpoint(f, lo, hi, ratio ** k)
    for j in range(1, levels + 1):
        factor = float(ratio) ** (2 * j)
        for i in range(j, levels + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (factor - 1.0)
    return R


exact = 1.0 - math.cos(1.0)
R = midpoint_romberg(np.sin, 0.0, 1.0, 5, 3)
print("midpoint Romberg with a trisection ladder, error at each entry:")
for i in range(R.shape[0]):
    print("  " + " ".join(f"{abs(R[i, j] - exact):9.2e}" for j in range(i + 1)))
print(f"\nfinest level uses 3^5 = {3 ** 5} points; corner error "
      f"{abs(R[-1, -1] - exact):.2e}")
```

The table behaves exactly like the trapezoid one, with $9^j$ in place of $4^j$ because the
refinement ratio is 3.

**The reason to want it** is that the midpoint rule never evaluates at the endpoints, so this is
a Romberg table for an integrand that is singular at $a$ or $b$. The expansion still needs
$f^{(2k-1)}$ to exist at the endpoints, though, so it does not rescue $\sqrt{x}$: it only avoids
the infinity in the first evaluation.

### 4.1 Levels needed against smoothness

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import romberg as rb

print(f"{'integrand':>16}{'levels to 1e-12':>18}{'evaluations':>14}{'converged':>12}")
for a in (0.5, 1.5, 2.5, 4.5, 8.5):
    def f(x, a=a):
        return np.abs(np.asarray(x, dtype=float) - 0.5) ** a

    exact = 2.0 * 0.5 ** (a + 1.0) / (a + 1.0)
    used = None
    for levels in range(2, 21):
        R = rb.table(f, 0.0, 1.0, levels)
        if abs(float(R[levels, levels]) - exact) <= 1e-12 * max(abs(exact), 1.0):
            used = levels
            break
    label = f"|x-0.5|^{a}"
    print(f"{label:>16}{str(used) if used else 'never by 20':>18}"
          f"{2 ** used + 1 if used else 0:>14}"
          f"{str(used is not None):>12}")

for name, f, exact in (("sin", np.sin, 1.0 - math.cos(1.0)),
                       ("exp", np.exp, math.e - 1.0)):
    used = next(levels for levels in range(2, 21)
                if abs(float(rb.table(f, 0.0, 1.0, levels)[levels, levels]) - exact) <= 1e-12)
    print(f"{name:>16}{used:>18}{2 ** used + 1:>14}{'True':>12}")
```

For an analytic integrand four levels suffice, at 17 evaluations. For $|x - \tfrac12|^a$ the
requirement falls steadily as $a$ rises: 14 levels at $a = 1.5$, then 10, 7 and 6. At $a = 0.5$
it never reaches $10^{-12}$ within 20 levels at all.

**The singularity here is at an interior point, not an endpoint**, which is a different failure
from the one section 7 of the lesson describes. Euler-Maclaurin's endpoint derivatives all
exist; what breaks is the derivation's assumption that $f$ is smooth **across** each panel.

The practical consequence is the same and the remedy is different. Splitting the interval at
$x = \tfrac12$ turns one hard problem into two easy ones, and lesson 64's adaptive routine
finds that split without being told where it is.

### 4.2 The geometric rate against the strip of analyticity

For a periodic analytic $f$, the trapezoid error on $n$ points falls like $e^{-\alpha n}$ where
$\alpha$ is the half width of the strip $|\operatorname{Im} z| < \alpha$ in which $f$ is analytic
and bounded.

For $f(x) = 1/(1 - r\cos x)$ with $0 < r < 1$, the poles are where $\cos z = 1/r$, that is
$z = \pm i\,\operatorname{arccosh}(1/r)$, so $\alpha = \operatorname{arccosh}(1/r)$.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import romberg as rb

print(f"{'r':>7}{'predicted alpha':>18}{'fitted rate':>14}{'ratio':>9}")
for r in (0.2, 0.4, 0.6, 0.8, 0.9):
    def f(x, r=r):
        return 1.0 / (1.0 - r * np.cos(np.asarray(x, dtype=float)))

    exact = 2.0 * math.pi / math.sqrt(1.0 - r * r)
    out = rb.periodic_convergence_is_geometric(
        f, exact, 0.0, 2.0 * math.pi, [4, 8, 12, 16, 24, 32, 48, 64])
    alpha = math.acosh(1.0 / r)
    print(f"{r:>7.2f}{alpha:>18.6f}{out['fitted_rate']:>14.6f}"
          f"{out['fitted_rate'] / alpha:>9.4f}")
```

The fitted rate tracks $\operatorname{arccosh}(1/r)$ closely. As $r \to 1$ the poles approach the
real axis, $\alpha \to 0$, and the convergence degrades to nothing, which is the periodic
counterpart of the Bernstein ellipse story in lesson 56.

### 4.3 How many Euler-Maclaurin terms are useful

Term $k$ of the series has size

$$
\left|\frac{B_{2k}}{(2k)!}h^{2k}\Delta_k\right|
\sim 2\left(\frac{h}{2\pi}\right)^{2k}\left|\Delta_k\right| ,
\qquad
\Delta_k = f^{(2k-1)}(b) - f^{(2k-1)}(a),
$$

using $|B_{2k}| \sim 2(2k)!/(2\pi)^{2k}$. **Whether the series converges is therefore decided
entirely by how fast the endpoint derivatives grow.**

For $f(x) = \sin(wx)$ the derivatives grow geometrically, $|f^{(n)}| = w^n$, so the term size is
proportional to $(hw/2\pi)^{2k}$ and the series **converges** if $hw < 2\pi$ and **diverges** if
$hw > 2\pi$. That is a sharp, checkable prediction.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc
from nalib import romberg as rb


def sin_derivative(x, k, w):
    return w ** k * math.sin(w * x + k * math.pi / 2.0)


print(f"{'w':>6}{'h':>7}{'h w':>8}{'h w / 2pi':>12}{'predicted':>12}"
      f"{'best terms':>12}{'gap at 16':>12}{'observed':>12}")
for w, m in ((4.0, 1), (20.0, 4), (40.0, 8), (8.0, 1), (12.0, 1), (20.0, 1), (40.0, 4)):
    h = 1.0 / m
    exact = (1.0 - math.cos(w)) / w
    trap = nc.composite_shared(lambda t, w=w: np.sin(w * np.asarray(t, dtype=float)),
                               0.0, 1.0, m, 2)
    measured = trap - exact
    gaps = []
    for terms in range(1, 17):
        out = rb.euler_maclaurin_terms(
            lambda x, k, w=w: sin_derivative(x, k, w), 0.0, 1.0, m, terms)
        gaps.append(abs(float(out["running_total"][-1]) - measured))
    best = int(np.argmin(gaps))
    predicted = "converges" if h * w < 2.0 * math.pi else "diverges"
    observed = "diverges" if gaps[-1] > 100.0 * gaps[best] else "converges"
    print(f"{w:>6.1f}{h:>7.3f}{h * w:>8.2f}{h * w / (2 * math.pi):>12.3f}"
          f"{predicted:>12}{best + 1:>12}{gaps[-1]:>12.1e}{observed:>12}")
```

**The prediction holds at every point.** Below the threshold the best truncation is the last term
tried, because the series is still improving. Above it the best truncation is **one term** and
every further term makes the answer worse, by a factor that reaches $10^{14}$ by sixteen terms.

Note what this means for $\sin$ on $[0,1]$, which is where the lesson's own numbers came from:
there $w = 1$ and $h \le 1$, so $hw \le 1 \ll 2\pi$ and the series converges comfortably. **The
divergence that makes Euler-Maclaurin asymptotic is real and it is invisible on the easy
examples.** Seeing it requires an integrand oscillating fast compared with the panel width.

### 5.1 The trapezoid rule on the real line

Let $f$ be analytic in a strip $|\operatorname{Im} z| < \alpha$ and decay fast enough that
$\int_{\mathbb{R}}|f(x + iy)|dx$ is bounded for $|y| < \alpha$. The infinite trapezoid rule is

$$
T_h = h\sum_{k=-\infty}^{\infty} f(kh).
$$

**The Poisson summation formula** gives the error directly:

$$
T_h - \int_{\mathbb{R}} f = \sum_{m \ne 0}\hat f\!\left(\frac{m}{h}\right),
$$

where $\hat f$ is the Fourier transform. Analyticity in a strip of half width $\alpha$ makes
$\hat f(\xi)$ decay like $e^{-2\pi\alpha|\xi|}$, so the error is $O(e^{-2\pi\alpha/h})$:
**geometric in $1/h$, not algebraic.**

That is the same conclusion as the periodic case and for the same underlying reason. The periodic
case is the statement that all the endpoint terms cancel; this is the statement that there are no
endpoints.

```python
import math

import numpy as np

print("infinite trapezoid rule on exp(-x^2), whose integral is sqrt(pi)")
exact = math.sqrt(math.pi)
print(f"{'h':>8}{'points used':>14}{'error':>14}{'exp(-pi^2/h^2)':>18}")
for h in (1.0, 0.75, 0.5, 0.4, 0.3):
    limit = math.sqrt(-math.log(1e-320))
    k = np.arange(-int(limit / h) - 1, int(limit / h) + 2)
    got = h * float(np.sum(np.exp(-(k * h) ** 2)))
    print(f"{h:>8.2f}{k.size:>14}{abs(got - exact):>14.2e}"
          f"{math.exp(-math.pi ** 2 / h ** 2):>18.2e}")
```

The error tracks $e^{-\pi^2/h^2}$, which for the Gaussian is the exact Poisson term, and reaches
machine precision at $h = 0.4$ with a few dozen points.

**This is the theorem lesson 66's double exponential rule is built on.** That rule substitutes
so that the transformed integrand decays doubly exponentially on the whole line, and then applies
exactly this.

### 5.2 The Romberg diagonal as a linear functional

Each $R_{k,j}$ is a fixed linear combination of the trapezoid ladder, so

$$
R_{k,k} = \sum_{i=0}^{k} \gamma^{(k)}_i T_i
$$

for coefficients $\gamma^{(k)}_i$ that depend on nothing but $k$. Their absolute sum is the
amplification factor of the extrapolation, exactly as $\sum_j|w_j|$ was for a quadrature rule.

```python
import numpy as np

def diagonal_coefficients(k):
    """R[k][k] as a linear combination of T_0 .. T_k.

    Each entry of the table is carried as its coefficient VECTOR against the trapezoid ladder,
    so the same recurrence that builds the table builds the functional. Note that `rows[0]`
    must be a list holding one vector, not the vector itself, or `rows[0][j]` indexes into the
    vector and the coefficients come out summing to 2/3 instead of 1.
    """
    rows = [[np.eye(1, k + 1, 0).ravel()]]
    for level in range(1, k + 1):
        row = [np.eye(1, k + 1, level).ravel()]
        for j in range(1, level + 1):
            row.append(row[j - 1] + (row[j - 1] - rows[level - 1][j - 1]) / (4.0 ** j - 1.0))
        rows.append(row)
    return rows[k][k]

print(f"{'k':>4}{'sum of coefficients':>22}{'sum of |coefficients|':>24}")
for k in range(1, 9):
    g = diagonal_coefficients(k)
    print(f"{k:>4}{float(np.sum(g)):>22.10f}{float(np.sum(np.abs(g))):>24.6f}")
```

The coefficients sum to exactly 1 at every $k$, as they must for the combination to reproduce
a constant. Their absolute sum rises from $1.667$ and **converges to about $1.9692$**: bounded,
not growing.

So the damage in section 6 of the lesson is not amplification. An amplification factor of 2
would turn a $10^{-15}$ error into $2\times10^{-15}$, which is nothing. What the diagonal does is
**include the coarse levels at all**.

On a periodic integrand $T_0$ and $T_1$ carry errors of order 1, and the diagonal gives them
coefficients of order 1. The result is an error of order 1 times whatever those coefficients
are, when the finest level alone was already at $10^{-15}$. **The extrapolation is not
corrupting a good answer; it is averaging a good answer with several bad ones.**

On a smooth non periodic integrand the same combination is designed so that those coarse
errors cancel to high order, and then including them is exactly the right thing to do.

### 5.3 Bernoulli growth and optimal truncation

The Bernoulli numbers satisfy

$$
|B_{2k}| = \frac{2(2k)!}{(2\pi)^{2k}}\,\zeta(2k) \sim \frac{2(2k)!}{(2\pi)^{2k}} ,
$$

since $\zeta(2k) \to 1$. **They grow faster than any geometric sequence**, because of the
$(2k)!$. In the Euler-Maclaurin series that factorial cancels against the $(2k)!$ in the
denominator, leaving term $k$ of size

$$
2\left(\frac{h}{2\pi}\right)^{2k}\left|\Delta_k\right|,
\qquad \Delta_k = f^{(2k-1)}(b) - f^{(2k-1)}(a).
$$

**So the series is not intrinsically divergent.** Everything depends on $\Delta_k$.

**Case one, geometric derivative growth.** If $|f^{(n)}| \le C M^n$, as for $\sin(wx)$ with
$M = w$ or for $e^{\lambda x}$ with $M = \lambda$, the terms are proportional to
$(hM/2\pi)^{2k}$ and the series **converges** for $hM < 2\pi$. Exercise 4.3 confirms that
threshold to the panel count.

**Case two, factorial derivative growth.** If $f$ is analytic only in a disc of radius $R$ about
the endpoint, Cauchy's estimate gives $|f^{(n)}| \lesssim n!/R^n$, so
$|\Delta_k| \sim (2k-1)!/R^{2k}$ and the term size is

$$
2(2k-1)!\left(\frac{h}{2\pi R}\right)^{2k}.
$$

By Stirling this falls until $2k \approx 2\pi R/h$ and grows thereafter, so the optimal
truncation is $k^\ast \approx \pi R/h$ terms and the smallest achievable error is
$O(e^{-2\pi R/h})$. **That is the same geometric bound as exercise 5.1**, which is no
coincidence: both measure how far into the complex plane $f$ extends.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import newtoncotes as nc
from nalib import romberg as rb


def pole_derivative(x, k, c):
    """kth derivative of 1/(c - x), whose nearest singularity is at distance c - x."""
    return math.factorial(k) / (c - x) ** (k + 1)


print("f(x) = 1/(c - x) on [0, 1]: the pole at c sets R = c - 1 at the right endpoint")
print(f"{'c':>6}{'R':>7}{'h':>7}{'pi R / h':>11}{'best terms':>12}{'best gap':>12}"
      f"{'gap at 14':>12}")
for c, m in ((1.5, 1), (1.5, 2), (2.0, 1), (2.0, 2), (3.0, 1), (3.0, 2)):
    h = 1.0 / m
    R = c - 1.0
    exact = math.log(c) - math.log(c - 1.0)
    trap = nc.composite_shared(lambda t, c=c: 1.0 / (c - np.asarray(t, dtype=float)),
                               0.0, 1.0, m, 2)
    measured = trap - exact
    gaps = []
    for terms in range(1, 15):
        out = rb.euler_maclaurin_terms(
            lambda x, k, c=c: pole_derivative(x, k, c), 0.0, 1.0, m, terms)
        gaps.append(abs(float(out["running_total"][-1]) - measured))
    best = int(np.argmin(gaps))
    print(f"{c:>6.1f}{R:>7.2f}{h:>7.3f}{math.pi * R / h:>11.2f}{best + 1:>12}"
          f"{gaps[best]:>12.1e}{gaps[-1]:>12.1e}")
```

**The prediction and the measurement agree to within one term at every row**: 1.57 against 1,
3.14 against 3, 6.28 against 6, 12.57 against 12. For a pole close to the interval only one term
is useful, and the fourteenth term is wrong by $2.6\times10^{14}$.

That is what an asymptotic series does, and it is worth seeing once: the answer improves, reaches
its best, and then deteriorates without limit while the formula that produced it stays correct.

**The practical rule that falls out of all this** is the one lesson 63 states in section 2: watch
the gap fall as the panels are refined, and never judge an asymptotic series by its size at a
single $h$. A series that is still improving at term 14 for one integrand can be diverging from
term 2 for another, and only the endpoint derivatives decide which.

## Lesson 64, Adaptive Quadrature

### 1.1 Local order, composite order, and which one sets the divisor

Simpson's rule on **one** interval of width $h$ has error $Ch^5f^{(4)}(\xi)$. That is the local
order, 5.

A composite rule applies it to $(b-a)/h$ intervals and sums, which trades one power of $h$ for
the panel count, as lesson 62 exercise 2.4 showed. That is the composite order, 4.

**The Richardson divisor comes from the local order**, because the comparison being made is
between one rule on one interval and the same rule on that interval's two halves. Nothing is
being composed over a fixed domain with shrinking $h$; a single interval is being subdivided
once.

Concretely, with $I - S = Ch^5$ and $I - S_2 = 2C(h/2)^5 = Ch^5/16$,

$$
S_2 - S = Ch^5\left(1 - \tfrac{1}{16}\right),
\qquad
I - S_2 = \frac{S_2 - S}{15}.
$$

The 16 is $2^{5-1}$, from the local order 5, and the divisor is $16 - 1 = 15$.

Writing the divisor as $2^{p-1}-1$ with the composite $p = 4$ gives 7. **That is wrong by
$15/7 = 2.14$ and nothing fails.** The estimate is too large, so the routine subdivides longer
than it needs to, passes every accuracy test, and spends about twice the evaluations.

### 1.2 Free accuracy that cannot be relied on

`corrected = fine + estimate` is one Richardson step. It uses the same function values that were
already computed to form `coarse` and `fine`, so it is free, and it removes the leading $h^5$
term, so it is two orders more accurate.

**Why that is good** is obvious: the routine returns a much better answer than it was asked for,
at no cost.

**Why it cannot be relied on** is that the tolerance test was applied to `estimate`, which
measures the error of `fine`. Nothing in the algorithm knows the size of the error in
`corrected`. The relationship between them depends on the next term of the expansion, which is
never computed.

So the guarantee the routine offers is about a value it does not return, and the value it does
return is better by an amount that varies with the integrand.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad


def root(x):
    return np.sqrt(np.asarray(x, dtype=float))


out = ad.correction_is_free_accuracy(root, 2.0 / 3.0, 0.0, 1.0)
print(f"{'tolerance':>11}{'uncorrected error':>20}{'corrected error':>18}"
      f"{'factor gained':>15}{'same cost':>11}")
for tol, c, u, ce, ue in zip(out["tolerance"], out["corrected_error"],
                             out["uncorrected_error"], out["corrected_evaluations"],
                             out["uncorrected_evaluations"]):
    print(f"{tol:>11.0e}{u:>20.3e}{c:>18.3e}{u / c:>15.1f}{str(ce == ue):>11}")
```

The factor gained is not constant: it grows from about 1.6 to about 1300 as the tolerance
tightens. **A quantity that varies by three orders of magnitude across a routine's normal
operating range is not something to build on.**

The correct posture is to treat the tolerance as applying to `fine`, take the extra accuracy as a
gift, and verify separately when the answer matters.

### 1.3 When adaptivity pays

It pays when the integrand has structure on very different scales, and the panel width the
algorithm chooses is the measurement.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad


def lorentzian(w):
    def f(x):
        return 1.0 / (w ** 2 + (np.asarray(x, dtype=float) - 0.3) ** 2)
    return f, (math.atan(0.7 / w) - math.atan(-0.3 / w)) / w


print(f"{'integrand':>26}{'panels':>9}{'widest':>12}{'narrowest':>12}{'ratio':>14}")
for name, f in (("sin, smooth", np.sin),
                ("sqrt(x), endpoint kink", lambda t: np.sqrt(np.asarray(t, dtype=float))),
                ("Lorentzian w = 1e-2", lorentzian(1e-2)[0]),
                ("Lorentzian w = 1e-4", lorentzian(1e-4)[0])):
    out = ad.where_the_work_went(f, 0.0, 1.0, 1e-10)
    print(f"{name:>26}{out['panel_count']:>9}{out['widest']:>12.2e}"
          f"{out['narrowest']:>12.2e}{out['width_ratio']:>14.3e}")
```

On a smooth integrand the ratio is 2, meaning **every accepted panel is essentially the same
width and adaptivity bought nothing.** A uniform rule would have produced the same partition
without the bookkeeping.

On the endpoint singularity the ratio is $10^{13}$: the algorithm has subdivided to $2^{-43}$
near zero and left the rest of the interval alone. That is a factor no uniform rule could reach
without spending $2^{43}$ evaluations.

**So the honest criterion is the scale ratio of the integrand's features, not its difficulty.**
A uniformly hard integrand is best handled by a uniform rule.

### 1.4 Why the spike centre decides everything

The first accept decision on $[a,b]$ evaluates $f$ at exactly five points: $a$, $b$, the midpoint,
and the two quarter points. Those five values are the entire evidence on which the interval is
accepted or split.

A Gaussian bump of width $10^{-3}$ is below $10^{-100}$ at a distance of $0.02$, so unless one of
those five points is within a few widths of the centre, all five read zero. Then `coarse` is
zero, `fine` is zero, `estimate` is zero, the interval is accepted, and the routine reports that
the integral of a strictly positive function is zero.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad

out = ad.spike_centre_decides_everything(width=1e-3)
print(f"{'centre':>10}{'relative error':>17}{'evaluations':>13}{'found':>8}")
for c, e, ev, found in zip(out["centre_fraction"], out["relative_error"],
                           out["evaluations"], out["found"]):
    print(f"{c:>10.6f}{e:>17.3e}{ev:>13}{str(found):>8}")
print("\nthe five points the first decision sees on [0, 1]: 0, 0.25, 0.5, 0.75, 1")
```

**What that implies about demonstrating the failure honestly** is the practical point. Placing
the spike at $x = 0.5$, which is the obvious choice and the one most write ups make, puts it
exactly on the first sample. The routine finds it instantly, and the demonstration shows the
opposite of what it appears to show.

Note that $0.375$ fails even though it is a dyadic rational. Being dyadic is not the criterion;
being one of the five points **this first step visits** is. A spike at $0.375$ would be found at
depth 2, if the routine ever got there, and it does not, because it accepted at depth 0.

### 2.1 The divisor from the local order

Let the local error on an interval of width $h$ be $E(h) = Ch^{q}$, where $q = p + 1$ is the
local order and $p$ the composite order.

One rule on the whole interval: $I - S = Ch^q$.

The same rule on each half: two errors of $C(h/2)^q$, so $I - S_2 = 2Ch^q/2^q = Ch^q/2^{q-1}$.

Subtracting,

$$
S_2 - S = (I - S) - (I - S_2) = Ch^q\left(1 - \frac{1}{2^{q-1}}\right)
= (I - S_2)\left(2^{q-1} - 1\right),
$$

so

$$
I - S_2 = \frac{S_2 - S}{2^{q-1} - 1} = \frac{S_2 - S}{2^{p} - 1}.
$$

For Simpson $p = 4$, $q = 5$, and the divisor is $2^4 - 1 = 15$.

**What goes wrong with $2^{p-1}-1$** is that it uses $2^{q-2}$ where $2^{q-1}$ belongs, giving 7.
The estimate is then $15/7$ times the true error of the fine rule.

That error is in the safe direction, which is why it survives. Every accuracy test passes,
because an overestimate of the error means the routine keeps subdividing past the point where it
should have stopped. The only symptom is the cost, and a routine's cost is rarely compared
against a theoretical minimum.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad

print(f"{'order p':>9}{'divisor 2^p - 1':>18}{'wrong 2^(p-1) - 1':>20}{'ratio':>9}")
for p in (2, 4, 6, 8):
    right = 2.0 ** p - 1.0
    wrong = 2.0 ** (p - 1) - 1.0
    print(f"{p:>9}{right:>18.0f}{wrong:>20.0f}{right / wrong:>9.4f}")
    assert ad.local_error_estimate(np.sin, 0.0, 1.0, p)["divisor"] == right

exact = 1.0 - math.cos(1.0)
edges = np.linspace(0.0, 1.0, 5)
print(f"\n{'subinterval':>16}{'true error of fine':>21}{'estimate, 15':>15}"
      f"{'estimate, 7':>14}")
for lo, hi in zip(edges[:-1], edges[1:]):
    out = ad.local_error_estimate(np.sin, float(lo), float(hi))
    true_error = (math.cos(lo) - math.cos(hi)) - out["fine"]
    with_seven = (out["fine"] - out["coarse"]) / 7.0
    print(f"{f'[{lo:.2f}, {hi:.2f}]':>16}{true_error:>21.4e}{out['estimate']:>15.4e}"
          f"{with_seven:>14.4e}")
```

The 15 column tracks the true error; the 7 column is consistently about twice too large.

### 2.2 The correction gains two orders

The corrected value is

$$
S_c = S_2 + \frac{S_2 - S}{2^p - 1}.
$$

Substituting $S = I - Ch^q$ and $S_2 = I - Ch^q/2^{q-1}$ with $q = p+1$:

$$
S_c = I - \frac{Ch^q}{2^{q-1}}
+ \frac{1}{2^{q-1}-1}\left(-\frac{Ch^q}{2^{q-1}} + Ch^q\right)
= I - \frac{Ch^q}{2^{q-1}} + \frac{Ch^q}{2^{q-1}} = I .
$$

**The leading term is removed exactly.** What remains is the next term of the expansion. For a
symmetric rule that expansion advances in steps of 2, so the next surviving term is $h^{q+2}$ and
the local order rises from $q$ to $q+2$: two orders, exactly as Richardson always gives on an
even expansion, which is lesson 63's section 1.

That is why the measured gain in exercise 1.2 is so large at tight tolerances: two orders
compound over the many subdivisions a tight tolerance forces.

### 2.3 Why the tolerance must be split

Let the routine accept intervals $J_1, \dots, J_m$ with local errors $e_1, \dots, e_m$. The
global error is $\left|\sum_i e_i\right| \le \sum_i |e_i|$.

**With splitting**, an interval at depth $d$ carries budget $\tau/2^d$. The accepted intervals
form a partition, and by induction on the binary tree, the budgets of the leaves of any subtree
sum to the budget of its root. So $\sum_i |e_i| \le \sum_i \tau_i = \tau$: **the requested
tolerance bounds the sum, which is what was asked for.**

**Without splitting**, every interval carries the full $\tau$, so the bound is $m\tau$ with $m$
the number of accepted intervals. Since $m$ can be as large as $2^{\text{depth}}$, the answer can
be that many times looser than requested.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad


def integrate_without_splitting(f, lo, hi, tol, max_depth=40, min_depth=2, _counter=None):
    """The same routine, giving each half the FULL tolerance instead of half of it."""
    total, panels, stack = 0.0, 0, [(float(lo), float(hi), 0)]
    while stack:
        left, right, depth = stack.pop()
        mid = 0.5 * (left + right)
        coarse = ad.simpson(f, left, right, _counter)
        fine = ad.simpson(f, left, mid, _counter) + ad.simpson(f, mid, right, _counter)
        estimate = (fine - coarse) / 15.0
        if (abs(estimate) <= tol and depth >= min_depth) or depth >= max_depth:
            total += fine + estimate
            panels += 1
        else:
            stack.append((left, mid, depth + 1))
            stack.append((mid, right, depth + 1))
    return total, panels


def root(x):
    return np.sqrt(np.asarray(x, dtype=float))


print(f"{'tolerance':>11}{'split: error':>15}{'error / tol':>13}{'evals':>8}"
      f"{'unsplit: error':>17}{'error / tol':>13}{'evals':>8}")
for tol in (1e-4, 1e-6, 1e-8):
    a = [0]
    split = ad.integrate(root, 0.0, 1.0, tol, _counter=a)
    b = [0]
    value, panels = integrate_without_splitting(root, 0.0, 1.0, tol, _counter=b)
    es, eu = abs(split["value"] - 2.0 / 3.0), abs(value - 2.0 / 3.0)
    print(f"{tol:>11.0e}{es:>15.2e}{es / tol:>13.4f}{a[0]:>8}"
          f"{eu:>17.2e}{eu / tol:>13.4f}{b[0]:>8}")
```

The unsplit version is cheaper and looser, which is exactly the trade it is silently making. On
this integrand the looseness is modest; on one that subdivides deeply it grows with the depth.

**The version that splits is the one whose tolerance means what it says.**

### 2.4 An estimate that is near zero while both values are wrong

The estimate is $(S_2 - S)/15$, which is zero whenever $S_2 = S$. That happens when the error
term happens to take the same value on the whole interval as on its two halves combined, and
symmetry can arrange it.

Take $f(x) = \cos(2\pi k x)$ on $[0,1]$ with $k$ an integer. Simpson's rule on $[0,1]$ uses the
points $0, \tfrac12, 1$, and Simpson on the halves adds $\tfrac14$ and $\tfrac34$, so the five
points are the quarter grid.

For that grid to see a constant, $f$ must have period dividing $\tfrac14$, which needs
$k$ a multiple of 4. At $k = 4$ the five values are $1, 1, 1, 1, 1$: both rules return 1, the
estimate is exactly zero, and the true integral is 0.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad


def wave(k):
    def f(x):
        return np.cos(2.0 * math.pi * k * np.asarray(x, dtype=float))
    return f


print(f"{'k':>4}{'coarse':>12}{'fine':>12}{'estimate':>13}{'true integral':>15}"
      f"{'accepted at depth 0?':>22}")
for k in (1, 2, 3, 4, 6, 8):
    out = ad.local_error_estimate(wave(k), 0.0, 1.0)
    accepted = abs(out["estimate"]) <= 1e-8
    print(f"{k:>4}{out['coarse']:>12.6f}{out['fine']:>12.6f}{out['estimate']:>13.2e}"
          f"{0.0:>15.6f}{str(accepted):>22}")
```

At $k = 4$ and $k = 8$ the estimate is exactly zero and both values are 1 when the answer is 0.
**The error estimate is not small because the error is small; it is small because the two wrong
answers agree.**

At $k = 2$ and $k = 6$ the two rules disagree, so the interval is split and the aliasing is
caught. The failure needs the sample grid to be a sublattice of the integrand's period, which is
a coincidence rather than a general property, and that is precisely what makes it dangerous: it
cannot be ruled out and it cannot be predicted.

The symmetry causing it is that the sample points are a subset of the period lattice of $f$, so
every sample lands at the same phase. That is aliasing, in the sense of lesson 58, and the
defence is `min_depth`: forcing at least two subdivisions changes the sample spacing and breaks
the coincidence for any particular $k$, though never for all $k$ at once.

### 3.1 A priority queue on the local estimate

```python
import heapq
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad


def globally_adaptive(f, lo, hi, tol, max_panels=200000, _counter=None):
    """Always refine the interval with the largest local error estimate."""
    def measure(left, right):
        mid = 0.5 * (left + right)
        coarse = ad.simpson(f, left, right, _counter)
        fine = ad.simpson(f, left, mid, _counter) + ad.simpson(f, mid, right, _counter)
        estimate = (fine - coarse) / 15.0
        return fine + estimate, abs(estimate)

    value, error = measure(float(lo), float(hi))
    heap = [(-error, float(lo), float(hi), value, error)]
    total, total_error = value, error
    panels = 1
    while total_error > tol and panels < max_panels:
        _, left, right, value, error = heapq.heappop(heap)
        total -= value
        total_error -= error
        mid = 0.5 * (left + right)
        for a, b in ((left, mid), (mid, right)):
            v, e = measure(a, b)
            heapq.heappush(heap, (-e, a, b, v, e))
            total += v
            total_error += e
            panels += 1
    return {"value": total, "panels": panels, "estimated_error": total_error}


def root(x):
    return np.sqrt(np.asarray(x, dtype=float))


exact = 2.0 / 3.0
print(f"{'tolerance':>11}{'local: evals':>15}{'error':>12}{'global: evals':>16}{'error':>12}")
for tol in (1e-6, 1e-8, 1e-10):
    a = [0]
    local = ad.integrate(root, 0.0, 1.0, tol, _counter=a)
    b = [0]
    glob = globally_adaptive(root, 0.0, 1.0, tol, _counter=b)
    print(f"{tol:>11.0e}{a[0]:>15}{abs(local['value'] - exact):>12.2e}"
          f"{b[0]:>16}{abs(glob['value'] - exact):>12.2e}")
```

The global strategy uses fewer evaluations for the same accuracy, because it never refines an
interval that is not the current worst. Its cost is a heap and the need to keep every interval
alive in memory, where the local strategy can discard an interval the moment it is accepted.

**The local strategy also parallelises trivially and the global one does not**, which is why
production codes for large problems often use the local form despite its extra evaluations.

### 3.2 Adaptive Gauss-Kronrod

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad
from nalib import gaussquad as gq


def adaptive_kronrod(f, lo, hi, tol, n=7, max_depth=50, _counter=None):
    """Adaptive subdivision using the Gauss-Kronrod pair for the local estimate."""
    total, panels = 0.0, 0
    stack = [(float(lo), float(hi), float(tol), 0)]
    while stack:
        left, right, budget, depth = stack.pop()
        out = gq.gauss_kronrod_estimate(f, left, right, n, _counter)
        estimate = out["raw_difference"]
        if estimate <= budget or depth >= max_depth:
            total += out["value"]
            panels += 1
        else:
            stack.append((left, 0.5 * (left + right), 0.5 * budget, depth + 1))
            stack.append((0.5 * (left + right), right, 0.5 * budget, depth + 1))
    return {"value": total, "panels": panels}


def root(x):
    return np.sqrt(np.asarray(x, dtype=float))


exact = 2.0 / 3.0
print(f"{'tolerance':>11}{'Simpson: evals':>17}{'error':>12}"
      f"{'Kronrod: evals':>17}{'error':>12}")
for tol in (1e-6, 1e-8, 1e-10, 1e-12):
    a = [0]
    simpson = ad.integrate(root, 0.0, 1.0, tol, _counter=a)
    b = [0]
    kronrod = adaptive_kronrod(root, 0.0, 1.0, tol, 7, _counter=b)
    print(f"{tol:>11.0e}{a[0]:>17}{abs(simpson['value'] - exact):>12.2e}"
          f"{b[0]:>17}{abs(kronrod['value'] - exact):>12.2e}")
```

Gauss-Kronrod needs far fewer evaluations for the same accuracy, because each panel is integrated
by a degree 23 rule instead of a degree 3 one. **That is why QUADPACK is built on it and not on
Simpson.**

The price is 15 evaluations per interval instead of 5, so on an integrand that needs very many
tiny intervals the advantage narrows.

### 3.3 A genuine bound rather than an estimate

Given $M \ge \max|f^{(4)}|$ on $[a,b]$, the composite Simpson error on an interval of width $h$
is bounded by $\frac{h^5}{2880}M$, with no unknown $\xi$ left in it. Summing over the accepted
intervals gives a real bound.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad


def bounded_integrate(f, lo, hi, tol, fourth_derivative_bound):
    """Adaptive Simpson returning a genuine bound, given a bound on |f''''|."""
    out = ad.integrate(f, lo, hi, tol)
    widths = out["intervals"][:, 1] - out["intervals"][:, 0]
    # each accepted interval was measured by the FINE rule, two Simpson panels of width w/2
    bound = float(np.sum(2.0 * (0.5 * widths) ** 5 / 2880.0)) * fourth_derivative_bound
    return out["value"], bound, out["panel_count"]


exact = 1.0 - math.cos(1.0)
print(f"{'tolerance':>11}{'value':>20}{'true error':>13}{'proved bound':>15}"
      f"{'bound / error':>15}")
for tol in (1e-4, 1e-6, 1e-8, 1e-10):
    value, bound, panels = bounded_integrate(np.sin, 0.0, 1.0, tol, 1.0)
    err = abs(value - exact)
    print(f"{tol:>11.0e}{value:>20.15f}{err:>13.2e}{bound:>15.2e}"
          f"{bound / max(err, 1e-300):>15.2e}")
```

The bound holds at every tolerance, which is the point: **it is a proof, not a guess.**

It is also enormously pessimistic, by factors of $1.5\times10^3$ to $7\times10^4$, because it
discards all the cancellation between panels and uses the global maximum of $f^{(4)}$
everywhere. That is the
usual price of a rigorous bound, and it is why estimates are what people actually use.

### 3.4 User supplied breakpoints

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad


def integrate_with_breakpoints(f, lo, hi, tol, breakpoints=(), _counter=None):
    """Split at the given points first, then run adaptive Simpson on each piece."""
    edges = sorted({float(lo), float(hi)} | {float(v) for v in breakpoints
                                             if lo < v < hi})
    total, panels = 0.0, 0
    share = tol / max(len(edges) - 1, 1)
    for left, right in zip(edges[:-1], edges[1:]):
        out = ad.integrate(f, left, right, share, min_depth=0, _counter=_counter)
        total += out["value"]
        panels += out["panel_count"]
    return {"value": total, "panel_count": panels}


centre, width = 0.4, 1e-3
exact = width * math.sqrt(math.pi) * 0.5 * (math.erf((1.0 - centre) / width)
                                            - math.erf((0.0 - centre) / width))


def spike(x):
    t = np.asarray(x, dtype=float)
    return np.exp(-((t - centre) / width) ** 2)


print(f"exact value {exact:.10e}")
print(f"\n{'strategy':>34}{'value':>16}{'relative error':>17}{'evaluations':>13}")
for label, kwargs in (("no help, min_depth 0", dict(min_depth=0)),
                      ("min_depth 5", dict(min_depth=5)),
                      ("min_depth 8", dict(min_depth=8))):
    counter = [0]
    out = ad.integrate(spike, 0.0, 1.0, 1e-8, _counter=counter, **kwargs)
    print(f"{label:>34}{out['value']:>16.6e}"
          f"{abs(out['value'] - exact) / exact:>17.3e}{counter[0]:>13}")

counter = [0]
out = integrate_with_breakpoints(spike, 0.0, 1.0, 1e-8,
                                 [centre - 10 * width, centre, centre + 10 * width],
                                 counter)
print(f"{'breakpoints at the spike':>34}{out['value']:>16.6e}"
      f"{abs(out['value'] - exact) / exact:>17.3e}{counter[0]:>13}")
```

**Three breakpoints reach the answer for 684 evaluations. Forced depth needs 2043 to do
marginally better, and 6003 to do no better at all.**

Read both columns before concluding. Forced depth 5 is the more accurate of the two,
$8.4\times10^{-10}$ against $1.1\times10^{-8}$, and it costs three times as much to get there.
Forced depth 8 triples the cost again for exactly the same accuracy.

The reason is that forcing depth blankets the whole interval uniformly, in the hope of landing
near the spike, and almost all of that work integrates zero to high precision. Telling the routine
where the spike is turns the problem into three subproblems, two of which are genuinely
integrals of nothing.

The lesson generalises past spikes. **Whenever you know something about the integrand, an
interface that lets you say it is worth more than any amount of automatic effort**, because the
automatic effort is searching a space you could have pointed at directly.

### 4.1 Achieved error against requested tolerance

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad


def lorentzian(w):
    def f(x):
        return 1.0 / (w ** 2 + (np.asarray(x, dtype=float) - 0.3) ** 2)
    return f, (math.atan(0.7 / w) - math.atan(-0.3 / w)) / w


cases = (("sin", np.sin, 1.0 - math.cos(1.0)),
         ("exp", np.exp, math.e - 1.0),
         ("sqrt(x)", lambda t: np.sqrt(np.asarray(t, dtype=float)), 2.0 / 3.0),
         ("Lorentzian 1e-2",) + lorentzian(1e-2),
         ("Lorentzian 1e-4",) + lorentzian(1e-4))
tolerances = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
print(f"{'integrand':>18}" + "".join(f"{t:>12.0e}" for t in tolerances) + f"{'worst':>10}")
for name, f, exact in cases:
    out = ad.tolerance_is_met(f, exact, 0.0, 1.0, tolerances)
    ratios = np.asarray(out["error_over_tolerance"], dtype=float)
    print(f"{name:>18}" + "".join(f"{r:>12.3g}" for r in ratios)
          + f"{float(np.max(ratios)):>10.3g}")
print("\nentries are achieved error divided by requested tolerance; above 1 means the "
      "request was not met")
```

**Four of the five integrands are met at every tolerance**, and the reason is the free correction
of exercise 1.2 rather than any conservatism in the test. The margin varies widely: $\sin$ is met
with four orders to spare, and the Lorentzian of half width $10^{-2}$ reaches $0.74$ of its
budget at $10^{-12}$, which is met and only just.

The Lorentzian of half width $10^{-4}$ fails at the two tightest tolerances, by factors of 2.3
and 215. The cause is not the algorithm: at $10^{-12}$ it is summing millions of panels, and the
accumulated roundoff of that summation is itself larger than the tolerance.

**The tolerance was unachievable and nothing said so.** A routine cannot deliver an accuracy
below the roundoff floor of its own summation, and none of them check.

### 4.2 The crossover in one dimension

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad


def lorentzian(w):
    def f(x):
        return 1.0 / (w ** 2 + (np.asarray(x, dtype=float) - 0.3) ** 2)
    return f, (math.atan(0.7 / w) - math.atan(-0.3 / w)) / w


print(f"{'half width':>12}{'scale ratio':>14}{'evaluations':>13}"
      f"{'adaptive':>12}{'uniform':>12}{'adaptive wins':>15}")
for w in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
    f, exact = lorentzian(w)
    out = ad.against_uniform(f, exact, 0.0, 1.0, [1e-8])
    work = ad.where_the_work_went(f, 0.0, 1.0, 1e-8)
    print(f"{w:>12.0e}{work['width_ratio']:>14.1f}{int(out['adaptive_evaluations'][0]):>13}"
          f"{float(out['adaptive_error'][0]):>12.2e}{float(out['uniform_error'][0]):>12.2e}"
          f"{str(bool(out['adaptive_wins'][0])):>15}")
```

**The crossover on this family is at a half width of $10^{-3}$**, where the scale ratio the
adaptive run discovers reaches about 8000. Above that width uniform Simpson resolves the peak
everywhere at the budget in play and its overhead free arithmetic wins; below it, most of the
interval is wasted effort for a uniform rule.

The crossover is therefore not at a particular half width in general; it is where the **scale
ratio** becomes large compared with what the budget can afford uniformly. While the Lorentzian is wide enough for composite Simpson to resolve it
everywhere at the budget in play, uniform wins. Once the peak is narrow enough that most of the
interval is wasted effort for a uniform rule, adaptive wins and the margin grows quickly.

**The measurable predictor is the width ratio column**, which is a property of the integrand that
the algorithm reports for free.

### 4.3 The effect of the wrong divisor

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad


def integrate_with_divisor(f, lo, hi, tol, divisor, max_depth=50, min_depth=2,
                           _counter=None):
    """ad.integrate with the Richardson divisor supplied by hand."""
    total, panels = 0.0, 0
    stack = [(float(lo), float(hi), float(tol), 0)]
    while stack:
        left, right, budget, depth = stack.pop()
        mid = 0.5 * (left + right)
        coarse = ad.simpson(f, left, right, _counter)
        fine = ad.simpson(f, left, mid, _counter) + ad.simpson(f, mid, right, _counter)
        estimate = (fine - coarse) / divisor
        if (abs(estimate) <= budget and depth >= min_depth) or depth >= max_depth:
            total += fine + estimate
            panels += 1
        else:
            stack.append((left, mid, 0.5 * budget, depth + 1))
            stack.append((mid, right, 0.5 * budget, depth + 1))
    return total, panels


def root(x):
    return np.sqrt(np.asarray(x, dtype=float))


exact = 2.0 / 3.0
print(f"{'tolerance':>11}{'divisor 15: evals':>19}{'error':>12}"
      f"{'divisor 7: evals':>19}{'error':>12}{'extra cost':>12}")
for tol in (1e-6, 1e-8, 1e-10, 1e-12):
    a, b = [0], [0]
    v15, _ = integrate_with_divisor(root, 0.0, 1.0, tol, 15.0, _counter=a)
    v7, _ = integrate_with_divisor(root, 0.0, 1.0, tol, 7.0, _counter=b)
    print(f"{tol:>11.0e}{a[0]:>19}{abs(v15 - exact):>12.2e}"
          f"{b[0]:>19}{abs(v7 - exact):>12.2e}{b[0] / a[0]:>12.2f}")
```

**The wrong divisor costs about 20 percent more evaluations and delivers an answer one to two
orders of magnitude worse.** It loses on both counts, which is worth understanding, because the
naive expectation is that a conservative estimate should at least buy accuracy.

It does not, because **the divisor appears twice**. In the accept test a larger estimate means
more subdivision, which helps. In the correction `fine + (fine - coarse)/divisor` a wrong
coefficient means the Richardson step no longer cancels the leading error term, so the free two
orders of exercise 2.2 are thrown away. The second effect is much larger than the first.

So a routine with divisor 7 is worse in every measurable way, and it still passes every accuracy
test, because those tests ask whether the answer meets the tolerance and it comfortably does. The
only reliable check is the one in exercise 2.1: compare the estimate against the true error of
the fine rule on a problem with a known answer, and confirm the ratio is 1 rather than 2.14.

### 5.1 Locally against globally adaptive

The routine in the lesson is **locally adaptive**: each interval is judged on its own budget and
accepted or split without reference to the rest. Exercise 3.1 built the **globally adaptive**
alternative, which always refines whichever interval currently has the largest estimated error.

**Global is better on evaluation count** and the measurement in 3.1 shows it. The reason is that
local adaptivity commits its budget in advance: an interval at depth $d$ gets $\tau/2^d$ whether
or not it deserves it, so easy regions are held to a standard they did not need and hard regions
are given one they cannot meet without going deeper than necessary.

**Local is better on everything else.**

- **Memory.** Local discards an accepted interval immediately; global holds the entire partition
  in a heap until the end.
- **Parallelism.** Local subproblems are independent, so a stack of intervals distributes across
  workers with no communication. Global must know the current worst interval, which is a
  synchronisation point at every step.
- **Predictability.** Local's cost depends on the integrand alone; global's depends on the order
  the heap happens to produce, which is sensitive to ties.

So the right measure is not "which uses fewer evaluations" but "which uses fewer of the resource
that is scarce". **On one core with an expensive integrand, global. On many cores, or with a
memory budget, local.**

### 5.2 The estimate as a hypothesis test

The local test asks whether $|S_2 - S|/15 \le \tau$. Written as a hypothesis, what is being
assumed is:

> $f$ has a continuous fourth derivative on this interval, and $h$ is small enough that the
> leading term $Ch^5f^{(4)}$ dominates the remainder.

The test only detects the **conclusion** of that hypothesis being small. It cannot distinguish
"the hypothesis holds and the error is small" from "the hypothesis fails and the two rules happen
to agree", which is exercise 2.4.

A test that also checks the hypothesis needs a third estimate. Compare the rule at three
resolutions and check that the errors fall at the predicted rate.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad


def three_level_check(f, lo, hi):
    """Compare S, S2 and S4: the ratio of successive differences tests the assumption."""
    a, b = float(lo), float(hi)
    mid = 0.5 * (a + b)
    quarters = np.linspace(a, b, 5)
    s1 = ad.simpson(f, a, b)
    s2 = ad.simpson(f, a, mid) + ad.simpson(f, mid, b)
    s4 = sum(ad.simpson(f, float(l), float(r))
             for l, r in zip(quarters[:-1], quarters[1:]))
    d1, d2 = s2 - s1, s4 - s2
    ratio = d1 / d2 if d2 != 0 else float("inf")
    return {"estimate": d2 / 15.0, "ratio": ratio,
            "asymptotic": bool(8.0 < abs(ratio) < 32.0)}


def wave(k):
    def f(x):
        return np.cos(2.0 * math.pi * k * np.asarray(x, dtype=float))
    return f


print(f"{'integrand':>22}{'two level estimate':>21}{'three level ratio':>20}"
      f"{'asymptotic?':>14}")
for name, f in (("sin", np.sin), ("exp", np.exp),
                ("cos(4 pi x)", wave(2)), ("cos(8 pi x)", wave(4)),
                ("sqrt(x)", lambda t: np.sqrt(np.asarray(t, dtype=float)))):
    two = ad.local_error_estimate(f, 0.0, 1.0)
    three = three_level_check(f, 0.0, 1.0)
    print(f"{name:>22}{two['estimate']:>21.3e}{three['ratio']:>20.4f}"
          f"{str(three['asymptotic']):>14}")
print("\nthe ratio should be 16 for a fourth order rule in its asymptotic regime")
```

**The three level ratio catches what the two level estimate cannot.** On the aliased cosines the
two level estimate is zero and the ratio is nowhere near 16, so the assumption is visibly false.
On the square root the estimate is not zero but the ratio is wrong too, correctly flagging that
the fourth derivative does not exist.

The cost is one more level, so about twice the evaluations. **QUADPACK does something related**,
comparing several rules and using the disagreement pattern rather than a single difference.

### 5.3 What QUADPACK adds

Three things beyond what this lesson implements, and each addresses a failure the lesson has
already met.

**The epsilon algorithm.** `qags` collects the sequence of partial results as intervals are
accepted and applies Wynn's epsilon algorithm, a nonlinear sequence accelerator, to it. On an
integrand with an endpoint singularity the sequence of partial sums converges algebraically, and
the epsilon algorithm converts that into something much faster. This is the same idea as lesson
63's Richardson extrapolation, generalised to sequences whose error expansion is unknown.

**Singularity detection.** If the local errors on successive subdivisions of the same region are
not falling at the expected rate, `qags` infers a singularity and switches strategy rather than
subdividing forever. That is a check on the hypothesis rather than on its conclusion, in the
sense of exercise 5.2.

**Roundoff detection.** `qags` monitors whether the local estimates have stopped improving as
intervals shrink, and returns an error code rather than continuing. **That is exactly the failure
of exercise 4.1**, where the Lorentzian at $10^{-12}$ spends millions of evaluations to return an
answer 215 times worse than requested, silently.

Here is the third one, which is the cheapest of the three and the one this lesson most needs.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad

EPS = np.finfo(float).eps


def integrate_with_roundoff_detection(f, lo, hi, tol, max_depth=50, min_depth=2):
    """Adaptive Simpson that reports when subdivision has stopped being able to help.

    An interval whose error estimate is already at the rounding level of its own value cannot
    be improved by splitting it: both halves will carry the same rounding. Such an interval is
    accepted and counted. If many are counted, the requested tolerance was below what the
    arithmetic can deliver, and the caller is told rather than left to find out.

    The test has to be against the roundoff floor and not against "the estimate stopped
    shrinking". The second version misfires on an unresolved feature, where the estimate is
    large and stays large because the feature is inside both halves, and accepting there throws
    the feature away. Written that way this returns 3.1e+04 when the answer is 3.1e+04, an error
    of 100 percent.
    """
    total, panels, floor_limited, floor_error = 0.0, 0, 0, 0.0
    stack = [(float(lo), float(hi), float(tol), 0)]
    while stack:
        left, right, budget, depth = stack.pop()
        mid = 0.5 * (left + right)
        coarse = ad.simpson(f, left, right)
        fine = ad.simpson(f, left, mid) + ad.simpson(f, mid, right)
        estimate = abs(fine - coarse) / 15.0
        at_floor = estimate <= 50.0 * EPS * max(abs(fine), 1e-300)
        good = estimate <= budget and depth >= min_depth
        if good or at_floor or depth >= max_depth:
            total += fine + (fine - coarse) / 15.0
            panels += 1
            if at_floor and not good:
                floor_limited += 1
                floor_error += estimate
        else:
            stack.append((left, mid, 0.5 * budget, depth + 1))
            stack.append((mid, right, 0.5 * budget, depth + 1))
    return {"value": total, "panels": panels, "floor_limited": floor_limited,
            "floor_error": floor_error,
            "tolerance_unachievable": floor_limited > 0.1 * panels}


def lorentzian(w):
    def f(x):
        return 1.0 / (w ** 2 + (np.asarray(x, dtype=float) - 0.3) ** 2)
    return f, (math.atan(0.7 / w) - math.atan(-0.3 / w)) / w


for label, w in (("smooth: sin", None), ("Lorentzian w = 1e-2", 1e-2),
                 ("Lorentzian w = 1e-4", 1e-4)):
    if w is None:
        f, exact = np.sin, 1.0 - math.cos(1.0)
    else:
        f, exact = lorentzian(w)
    print(f"\n{label}")
    print(f"{'tolerance':>11}{'error':>12}{'error / tol':>14}{'panels':>9}"
          f"{'floor limited':>15}{'unachievable':>14}")
    for tol in (1e-6, 1e-8, 1e-10, 1e-12):
        out = integrate_with_roundoff_detection(f, 0.0, 1.0, tol)
        err = abs(out["value"] - exact)
        print(f"{tol:>11.0e}{err:>12.2e}{err / tol:>14.4g}{out['panels']:>9}"
              f"{out['floor_limited']:>15}{str(out['tolerance_unachievable']):>14}")
```

**The detector fires exactly where the tolerance is missed and nowhere else.** On $\sin$ it never
fires. On the wide Lorentzian it fires only at $10^{-12}$, where the achieved error has climbed to
$0.17$ of the budget. On the narrow one it fires from $10^{-8}$ onward, and at $10^{-12}$, where
the request is missed by a factor of 7, it reports that 12403 of 15989 panels were accepted
because subdividing them could not help.

There is a second benefit that was not the goal. Exercise 4.1 measured this same integrand
missing $10^{-12}$ by a factor of **215** after several million evaluations. Stopping at the
roundoff floor instead misses by a factor of **7** after 16 thousand. Refusing to subdivide into
noise is not only more honest, it is more accurate, because every one of those millions of extra
panels was contributing rounding and no information.

**Writing this was itself an illustration of the danger.** The first version tested whether the
estimate had stopped shrinking under subdivision, which sounds equivalent and is not: on an
unresolved peak the estimate is large and stays large, so the test fired at depth 1 and accepted
an interval containing the entire feature. It returned an answer with 100 percent error at every
tolerance, while confidently reporting a small stall count.

**Adaptive quadrature is not reliable and QUADPACK ships anyway** because these three additions
turn most of the unreliability into a returned error code. That is the realistic goal: not a
routine that never fails, but one that says so when it does.

## Lesson 65, Gaussian Quadrature

### 1.1 The counting argument, both directions

**Achievable.** An $n$ point rule has $2n$ free parameters, $n$ nodes and $n$ weights. Requiring
exactness on $1, x, \dots, x^{2n-1}$ is $2n$ equations. Equal counts, so a solution can exist, and
the orthogonal polynomial construction shows one does.

**Upper bound.** No $n$ point rule is exact on degree $2n$. Take

$$
p(x) = \prod_{j=1}^{n}(x - x_j)^2 ,
$$

which has degree $2n$ and vanishes at every node. The rule returns $\sum_j w_j p(x_j) = 0$. The
true integral is $\int_a^b w(x)p(x)dx > 0$, because $p \ge 0$ and $p$ is not identically zero on
an interval of positive measure. **So the rule is wrong on $p$, whatever the nodes and weights
are.**

That argument uses only that the weight function is positive, so it applies to every family in
the lesson, and it shows $2n-1$ is exactly the ceiling rather than merely a value someone
achieved.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq

print(f"{'n':>4}{'rule on prod (x - xj)^2':>26}{'true integral':>16}{'degree':>9}")
for n in (2, 3, 5, 8):
    x, w = gq.nodes_and_weights(n)
    values = np.prod((x.reshape(-1, 1) - x.reshape(1, -1)) ** 2, axis=1)
    # the true integral of that degree 2n polynomial, by a rule big enough to be exact
    big_x, big_w = gq.nodes_and_weights(2 * n)
    truth = float(np.sum(big_w * np.prod((big_x.reshape(-1, 1) - x.reshape(1, -1)) ** 2,
                                         axis=1)))
    print(f"{n:>4}{float(np.sum(w * values)):>26.2e}{truth:>16.6e}{2 * n:>9}")
```

The rule returns zero for a strictly positive integrand at every $n$, which is as wrong as a
quadrature rule can be.

### 1.2 Positive weights, and why they matter more than the degree

**Why they are positive.** Apply the rule to $\ell_j(x)^2$, the square of the $j$th Lagrange
basis polynomial on the nodes. It has degree $2n-2 \le 2n-1$, so the rule is exact on it. The
rule's value is $\sum_i w_i \ell_j(x_i)^2 = w_j$, since $\ell_j(x_i) = \delta_{ij}$. The true
value is $\int w\,\ell_j^2 > 0$. Hence $w_j > 0$.

**Why that matters more than the degree.** The degree controls the truncation error, which falls
as $n$ grows. Positivity controls the **stability**, which does not improve with $n$ and can be
lost entirely.

With all $w_j > 0$ and $\sum_j w_j = \mu_0$, the rule satisfies

$$
\left|\sum_j w_j f(x_j)\right| \le \mu_0 \max_j |f(x_j)| ,
$$

so it is a weighted average and cannot amplify. The computed answer carries a relative error of
about $\varepsilon$, independent of $n$.

Newton-Cotes loses this at 9 nodes, and by 21 nodes its $\sum_j|w_j|$ is 544 and it has thrown
away two and a half digits before starting. **That is why Newton-Cotes must be composed and Gauss
need not be**, and it is a statement about arithmetic rather than about approximation.

### 1.3 What the Kronrod difference estimates

The difference $|G - K|$ estimates the error of the **Gauss** value. The value returned is the
**Kronrod** value, whose degree is roughly three times higher and whose error is therefore much
smaller.

**The consequence is a systematic overestimate**, in the safe direction, of a size that varies
with the integrand and the node count.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq

cases = (("sin", np.sin, 0.0, 1.0, 1.0 - math.cos(1.0)),
         ("exp", np.exp, 0.0, 1.0, math.e - 1.0),
         ("1/(1+25x^2)", lambda t: 1.0 / (1.0 + 25.0 * np.asarray(t, dtype=float) ** 2),
          -1.0, 1.0, 2.0 * math.atan(5.0) / 5.0))
print(f"{'integrand':>14}{'n':>4}{'|G - K|':>12}{'Gauss error':>14}{'Kronrod error':>16}"
      f"{'over Gauss':>12}{'over Kronrod':>14}")
for name, f, a, b, exact in cases:
    for n in (3, 7):
        out = gq.kronrod_estimate_quality(f, exact, a, b, [n])
        print(f"{name:>14}{n:>4}{float(out['difference'][0]):>12.2e}"
              f"{float(out['gauss_error'][0]):>14.2e}"
              f"{float(out['kronrod_error'][0]):>16.2e}"
              f"{float(out['difference_over_gauss_error'][0]):>12.2f}"
              f"{float(out['difference_over_kronrod_error'][0]):>14.2e}")
```

The `over Gauss` column is near 1, confirming the estimate does what it claims. The
`over Kronrod` column is what a caller actually experiences, and it ranges over many orders of
magnitude.

**So a caller who reads the reported error as "the error in the number I was given" is wrong,
always in their favour, by an unpredictable factor.** That is why adaptive routines built on this
pair meet their tolerances so comfortably, and why they cannot be used to certify an answer.

### 1.4 A 20 point rule measured to have degree 41

The true degree is $2n - 1 = 39$. Reading 41 means the measurement walked past $x^{40}$ and
$x^{41}$ without noticing they were wrong.

**What actually happened** is that the rule's relative error on $x^{40}$, the first monomial it
cannot integrate exactly, is $5.8\times10^{-11}$, which is below the $10^{-10}$ tolerance the walk
uses.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq

out = gq.degree_boundary_report()
print(f"{'nodes':>7}{'true':>7}{'measured':>10}{'margin on x^(2n)':>19}{'measurable':>13}")
for n, t, m, margin, ok in zip(out["nodes"], out["true_degree"], out["measured_degree"],
                               out["margin"], out["measurable"]):
    print(f"{n:>7}{t:>7}{m:>10}{margin:>19.2e}{str(ok):>13}")
print(f"\nlast node count where the boundary is measurable: {out['last_measurable']}")
```

The margin falls by roughly two orders of magnitude for every four nodes added, so the boundary
becomes unmeasurable at about 20 nodes and is below machine precision by 28.

**This is not a defect in the rule.** It is the rule being so accurate on the first monomial it
misses that "exact" and "not exact" stop being distinguishable in double precision. The honest
response is to report the margin alongside the degree, so a reader can see when the number has
stopped carrying information, which is what `degree_boundary_report` does.

### 2.1 The exactness theorem

**Claim.** Let $p_n$ be orthogonal to every polynomial of degree below $n$ with respect to the
positive weight $w$ on $(a,b)$, let $x_1, \dots, x_n$ be its roots, and let $w_j$ be the
interpolatory weights on those nodes. Then the rule is exact on every polynomial of degree
$\le 2n-1$, and not on some polynomial of degree $2n$.

**Proof of exactness.** Let $\deg f \le 2n-1$. Divide by $p_n$:

$$
f = q\,p_n + r, \qquad \deg q \le n-1, \quad \deg r \le n-1 .
$$

Then

$$
\int_a^b w f = \int_a^b w\,q\,p_n + \int_a^b w\,r = 0 + \int_a^b w\,r ,
$$

the first integral vanishing by orthogonality since $\deg q < n$.

On the other side,

$$
\sum_j w_j f(x_j) = \sum_j w_j\left[q(x_j)p_n(x_j) + r(x_j)\right]
= \sum_j w_j r(x_j),
$$

since $p_n(x_j) = 0$. And the rule is exact on $r$ because $\deg r \le n-1$ and any $n$ point
interpolatory rule is exact on degree $n-1$. So both sides equal $\int_a^b w\,r$.

**Proof of the bound.** Exercise 1.1.

Note which hypothesis each step used. Orthogonality killed $\int w\,q\,p_n$; the nodes being
roots killed $\sum w_j q(x_j) p_n(x_j)$. **Both are needed and they are the same fact used
twice**, once on the integral and once on the sum, which is why the nodes have to be exactly the
roots and not merely close to them.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq

for family in ("legendre", "chebyshev_t", "hermite"):
    for n in (3, 6, 10):
        out = gq.why_the_roots(n, family)
        print(f"{family:>14} n={n:>2}: p_n at the nodes {out['relative_residual_at_the_nodes']:.1e}"
              f", worst orthogonality residual "
              f"{float(np.max(out['orthogonality_residuals'])):.1e}")
```

### 2.2 Positivity from $\ell_j^2$

Fix $j$ and let

$$
\ell_j(x) = \prod_{i \ne j}\frac{x - x_i}{x_j - x_i},
$$

so $\deg \ell_j = n-1$ and $\ell_j(x_i) = \delta_{ij}$. Then $\ell_j^2$ has degree $2n-2$, which
is at most $2n-1$, so the rule is exact on it:

$$
\int_a^b w\,\ell_j^2 = \sum_i w_i\,\ell_j(x_i)^2 = w_j .
$$

The left side is strictly positive: $w > 0$ on $(a,b)$ and $\ell_j^2 \ge 0$ with $\ell_j^2 = 1$
at $x_j$, so the integrand is positive on a neighbourhood. Hence $w_j > 0$.

**The argument needs exactness on degree $2n-2$**, so it applies to Gauss but not to a rule of
lower degree. Newton-Cotes on $n$ nodes is exact only on degree $n-1$, and $\ell_j^2$ has degree
$2n-2 > n-1$, so the argument does not apply and indeed the conclusion fails.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq
from nalib import newtoncotes as nc

print(f"{'n':>5}{'Gauss smallest w':>20}{'Newton-Cotes smallest w':>26}")
for n in (5, 9, 15, 21, 31):
    gw = gq.nodes_and_weights(n)[1]
    cw = nc.weights(n, True)
    print(f"{n:>5}{float(np.min(gw)):>20.4e}{float(np.min(cw)):>26.4e}")
print("\nGauss stays positive at every n; Newton-Cotes goes negative at 9 and stays there")
```

### 2.3 The roots are real, simple and interior

Let $x_1, \dots, x_m$ be the points of $(a,b)$ where $p_n$ changes sign, and suppose $m < n$.
Form

$$
q(x) = \prod_{i=1}^{m}(x - x_i),
$$

of degree $m < n$. Then $p_n q$ does not change sign on $(a,b)$, because every sign change of
$p_n$ is matched by one of $q$. So

$$
\int_a^b w\,p_n\,q \ne 0 ,
$$

being the integral of a positive weight times a function of one sign that is not identically
zero. But orthogonality says that integral is zero, because $\deg q < n$. **Contradiction, so
$m \ge n$.**

A polynomial of degree $n$ has at most $n$ sign changes, so $m = n$ exactly: there are $n$ sign
changes inside $(a,b)$, hence $n$ distinct real roots, all interior.

Simplicity follows: $n$ distinct roots of a degree $n$ polynomial account for all of them, so
none is repeated.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq

print(f"{'family':>14}{'n':>4}{'all real':>10}{'all distinct':>14}{'strictly inside':>18}"
      f"{'smallest gap':>15}")
for family in ("legendre", "chebyshev_t", "chebyshev_u"):
    lo, hi = gq.FAMILIES[family][0]
    for n in (4, 12, 30):
        x, _ = gq.nodes_and_weights(n, family)
        gaps = float(np.min(np.diff(x)))
        print(f"{family:>14}{n:>4}{str(bool(np.all(np.isreal(x)))):>10}"
              f"{str(gaps > 0):>14}"
              f"{str(bool(np.min(x) > lo and np.max(x) < hi)):>18}{gaps:>15.4e}")
```

### 2.4 Golub-Welsch

Write the three term recurrence in monic form,

$$
p_{k+1}(x) = (x - \alpha_k)p_k(x) - \beta_k p_{k-1}(x), \qquad p_{-1} = 0,\ p_0 = 1 .
$$

Collect $p_0, \dots, p_{n-1}$ into a vector $P(x)$. The recurrence for $k = 0, \dots, n-1$ reads

$$
x P(x) = J_{\text{monic}} P(x) + p_n(x)\,e_n ,
$$

where $J_{\text{monic}}$ is tridiagonal with $\alpha_k$ on the diagonal, $\beta_k$ on the
subdiagonal and 1 on the superdiagonal.

**At a root of $p_n$ the last term vanishes**, so $P(x_j)$ is an eigenvector of
$J_{\text{monic}}$ with eigenvalue $x_j$. That is the nodes.

$J_{\text{monic}}$ is not symmetric, but it is similar to a symmetric matrix. Rescale by
$D = \operatorname{diag}(d_0, \dots, d_{n-1})$ with $d_k = \sqrt{\beta_1\cdots\beta_k}$, and
$J = D J_{\text{monic}} D^{-1}$ is symmetric tridiagonal with $\alpha_k$ on the diagonal and
$\sqrt{\beta_k}$ off it. Similar matrices share eigenvalues, so the nodes are unchanged.

**For the weights**, let $v_j$ be the unit eigenvector of $J$ for $x_j$. It is proportional to
$D P(x_j)$, whose entries are $d_k p_k(x_j)$. Using the Christoffel-Darboux identity of lesson 55
and the normalisation $\sum_k \tilde p_k(x_j)^2 = 1/w_j$ for the orthonormal polynomials,

$$
w_j = \mu_0\, v_{1j}^2 ,
$$

where $v_{1j}$ is the **first** component and $\mu_0 = \int_a^b w$ is the total mass. The first
component is the one carrying $p_0 = 1$, which is why it is the one that appears.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq

for family in ("legendre", "chebyshev_t", "laguerre"):
    for n in (4, 9):
        alpha, beta, mass = gq.recurrence(n, family)
        off = np.sqrt(beta[1:n])
        J = np.diag(alpha) + np.diag(off, 1) + np.diag(off, -1)
        values, vectors = np.linalg.eigh(J)
        order = np.argsort(values)
        x_mine, w_mine = values[order], mass * vectors[0, order] ** 2
        x, w = gq.nodes_and_weights(n, family)
        print(f"{family:>14} n={n}: nodes agree to "
              f"{float(np.max(np.abs(x_mine - x))):.1e}, weights to "
              f"{float(np.max(np.abs(w_mine - w))):.1e}")
```

### 2.5 The Gauss error term

The standard result is

$$
\int_a^b w f - \sum_j w_j f(x_j)
= \frac{f^{(2n)}(\xi)}{(2n)!}\int_a^b w(x)\,p_n(x)^2\,dx ,
$$

for some $\xi \in (a,b)$, with $p_n$ **monic**.

**Sketch.** Let $H$ be the Hermite interpolant of $f$ at the nodes, matching value and derivative
at each, so $\deg H \le 2n-1$. The Hermite error formula gives

$$
f(x) - H(x) = \frac{f^{(2n)}(\eta_x)}{(2n)!}\,p_n(x)^2 .
$$

The rule is exact on $H$, and $H(x_j) = f(x_j)$, so the quadrature error is
$\int_a^b w(f - H)$. Since $w p_n^2 \ge 0$ does not change sign, the integral mean value theorem
pulls out a single $\xi$, giving the formula.

**Why that implies geometric convergence for an analytic integrand.** For $f$ analytic in a
region containing $[a,b]$, Cauchy's estimate bounds $|f^{(2n)}|$ by $C(2n)!/R^{2n}$ where $R$
relates to the size of the region. The factorials cancel and the error is bounded by
$C R^{-2n}\int w p_n^2$. For Legendre, $\int_{-1}^1 p_n^2 \sim 2^{-2n}$ up to algebraic factors,
so the error falls like $(2R)^{-2n}$: **geometric in $n$**, with a rate set by how far into the
complex plane $f$ extends.

Making $R$ precise gives the Bernstein ellipse of lesson 56, which is exercise 4.1.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq

print("the constant int w p_n^2, computed by a rule big enough to be exact")
print(f"{'n':>5}{'int p_n^2':>16}{'ratio to previous':>20}{'4^-n':>14}")
previous = None
for n in (2, 4, 6, 8, 10):
    x, w = gq.nodes_and_weights(3 * n)
    alpha, beta, _ = gq.recurrence(n + 1, "legendre")
    p = np.ones_like(x)
    prev = np.zeros_like(x)
    for k in range(n):
        p, prev = (x - alpha[k]) * p - (beta[k] if k > 0 else 0.0) * prev, p
    value = float(np.sum(w * p ** 2))
    ratio = "" if previous is None else f"{value / previous:.4f}"
    print(f"{n:>5}{value:>16.6e}{ratio:>20}{4.0 ** (-n):>14.2e}")
    previous = value
```

### 3.1 Gauss-Lobatto

Lobatto fixes both endpoints as nodes and optimises the remaining $n-2$. That is $2n-2$ free
parameters, so the degree is $2n-3$.

The nodes are $\pm 1$ together with the roots of $P_{n-1}'$, and the weights come from the same
Jacobi matrix idea with the last two recurrence coefficients modified.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq


def lobatto(n):
    """n point Gauss-Lobatto on [-1, 1], including both endpoints.

    The Jacobi matrix has size n, and its last diagonal and off diagonal entries are chosen so
    that both -1 and +1 are eigenvalues. That is a 2 by 2 linear system in the monic orthogonal
    polynomials at the two endpoints. Building an (n-1) by (n-1) matrix instead is the easy
    mistake: it returns n-1 nodes, still hits both endpoints, still has positive weights summing
    to 2, and has degree 2n-5 rather than 2n-3.
    """
    if n < 2:
        raise ValueError("need at least the two endpoints")
    if n == 2:
        return np.asarray([-1.0, 1.0]), np.asarray([1.0, 1.0])
    alpha, beta, mass = gq.recurrence(n, "legendre")
    a, b = alpha[:n].copy(), beta[:n].copy()

    def monic_pair(t, k):
        """p_k(t) and p_{k-1}(t) by the three term recurrence."""
        p, prev = 1.0, 0.0
        for i in range(k):
            p, prev = (t - a[i]) * p - (b[i] if i > 0 else 0.0) * prev, p
        return p, prev

    p_lo, q_lo = monic_pair(-1.0, n - 1)
    p_hi, q_hi = monic_pair(1.0, n - 1)
    sol = np.linalg.solve(np.array([[p_lo, q_lo], [p_hi, q_hi]]),
                          np.array([-1.0 * p_lo, 1.0 * p_hi]))
    a[n - 1], b[n - 1] = sol[0], sol[1]
    off = np.sqrt(np.abs(b[1:n]))
    values, vectors = np.linalg.eigh(np.diag(a) + np.diag(off, 1) + np.diag(off, -1))
    order = np.argsort(values)
    return values[order], mass * vectors[0, order] ** 2


print(f"{'n':>4}{'nodes':>7}{'both ends':>12}{'sum w':>10}{'w > 0':>8}"
      f"{'degree':>9}{'2n - 3':>9}")
for n in (2, 3, 4, 5, 6, 8, 10):
    x, w = lobatto(n)
    ends = abs(x[0] + 1.0) < 1e-12 and abs(x[-1] - 1.0) < 1e-12
    degree = -1
    for k in range(0, 2 * n + 2):
        got = float(np.sum(w * x ** k))
        want = 0.0 if k % 2 else 2.0 / (k + 1.0)
        if abs(got - want) > 1e-10 * max(abs(want), 1.0):
            break
        degree = k
    print(f"{n:>4}{x.size:>7}{str(ends):>12}{float(np.sum(w)):>10.6f}"
          f"{str(bool(np.all(w > 0))):>8}{degree:>9}{2 * n - 3:>9}")
    assert degree == 2 * n - 3
```

The measured degree is $2n-3$ at every $n$, and the weights are positive and sum to 2.

**When you want it.** Whenever the endpoint values are needed anyway: in a spectral element
method the elements share a face, and a rule with nodes there lets neighbouring elements reuse
the same value and impose continuity directly. The lost two degrees are a small price for not
having to interpolate to the boundary.

### 3.2 Gauss-Radau

Radau fixes one endpoint, leaving $2n-1$ free parameters and degree $2n-2$.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq


def radau(n, fixed=-1.0):
    """n point Gauss-Radau on [-1, 1], with `fixed` as a node."""
    alpha, beta, mass = gq.recurrence(n, "legendre")
    a = alpha[:n].copy()
    b = beta[:n].copy()
    # shift the last diagonal entry so that `fixed` becomes an eigenvalue
    p, prev = 1.0, 0.0
    for k in range(n - 1):
        p, prev = (fixed - a[k]) * p - (b[k] if k > 0 else 0.0) * prev, p
    a[n - 1] = fixed - b[n - 1] * prev / p
    off = np.sqrt(b[1:n])
    values, vectors = np.linalg.eigh(np.diag(a) + np.diag(off, 1) + np.diag(off, -1))
    order = np.argsort(values)
    return values[order], mass * vectors[0, order] ** 2


print(f"{'n':>4}{'has the fixed end':>20}{'sum w':>10}{'w > 0':>8}{'degree':>9}"
      f"{'2n - 2':>9}")
for n in (2, 3, 4, 5, 6, 8):
    x, w = radau(n)
    degree = -1
    for k in range(0, 2 * n + 2):
        got = float(np.sum(w * x ** k))
        want = 0.0 if k % 2 else 2.0 / (k + 1.0)
        if abs(got - want) > 1e-10 * max(abs(want), 1.0):
            break
        degree = k
    print(f"{n:>4}{str(abs(x[0] + 1.0) < 1e-12):>20}{float(np.sum(w)):>10.6f}"
          f"{str(bool(np.all(w > 0))):>8}{degree:>9}{2 * n - 2:>9}")
```

Radau sits between Gauss and Lobatto: one degree better than Lobatto, one worse than Gauss, and
it is the natural choice for a half infinite interval where one end is special, or for the
implicit Runge-Kutta methods of Part 10 where one endpoint of the step must be a stage.

### 3.3 A Gauss rule for a non classical weight

The Stieltjes procedure from lesson 55 computes the recurrence coefficients for any positive
weight numerically:

$$
\alpha_k = \frac{\langle x p_k, p_k\rangle}{\langle p_k, p_k\rangle},
\qquad
\beta_k = \frac{\langle p_k, p_k\rangle}{\langle p_{k-1}, p_{k-1}\rangle} .
$$

The inner products are integrals against the weight, evaluated by a rule fine enough to be
accurate. Once the coefficients exist, Golub-Welsch does the rest.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq
from nalib import multiquad as mq


def rule_for_weight(weight, nodes, node_weights, n):
    """Discretised Stieltjes, then Golub-Welsch, for an arbitrary positive weight.

    `nodes` and `node_weights` are any quadrature rule accurate enough to compute the inner
    products against `weight`. Everything else follows from lesson 55 and section 3 of this
    lesson: no closed form for the orthogonal polynomials is needed anywhere.
    """
    t = np.asarray(nodes, dtype=float)
    mu = np.asarray(node_weights, dtype=float) * np.asarray(weight(t), dtype=float)
    alpha, beta = np.zeros(n), np.zeros(n)
    beta[0] = float(np.sum(mu))
    p_prev, p = np.zeros_like(t), np.ones_like(t)
    norm_prev = beta[0]
    for k in range(n):
        norm = float(np.sum(mu * p * p))
        alpha[k] = float(np.sum(mu * t * p * p)) / norm
        if k > 0:
            beta[k] = norm / norm_prev
        p, p_prev = (t - alpha[k]) * p - (beta[k] if k > 0 else 0.0) * p_prev, p
        norm_prev = norm
    off = np.sqrt(beta[1:n])
    values, vectors = np.linalg.eigh(np.diag(alpha) + np.diag(off, 1) + np.diag(off, -1))
    order = np.argsort(values)
    return values[order], float(beta[0]) * vectors[0, order] ** 2


def log_weight(x):
    """w(x) = -log(x) on (0, 1], a classic non classical weight."""
    return -np.log(np.maximum(np.asarray(x, dtype=float), 1e-300))


n = 6
plain = mq.tanh_sinh(lambda t: np.ones_like(np.asarray(t, dtype=float)), 0.0, 1.0, 6)
choices = (("Gauss-Legendre, 400 points",) + gq.rule_on(400, 0.0, 1.0),
           ("Gauss-Legendre, 2000 points",) + gq.rule_on(2000, 0.0, 1.0),
           ("tanh-sinh, level 6", plain["nodes"], plain["weights"]))
print(f"{'discretisation':>28}{'points':>9}{'sum of weights':>18}{'worst moment gap':>19}")
for label, t, u in choices:
    x, w = rule_for_weight(log_weight, t, u, n)
    worst = max(abs(float(np.sum(w * x ** k)) - 1.0 / (k + 1.0) ** 2) * (k + 1.0) ** 2
                for k in range(2 * n))
    print(f"{label:>28}{np.asarray(t).size:>9}{float(np.sum(w)):>18.12f}{worst:>19.2e}")

x, w = rule_for_weight(log_weight, plain["nodes"], plain["weights"], n)
print(f"\n{n} point Gauss rule for w(x) = -log(x) on (0, 1]")
print(f"   nodes   {np.array2string(x, precision=10)}")
print(f"   weights {np.array2string(w, precision=10)}")
print(f"\n{'k':>4}{'rule on x^k':>18}{'exact 1/(k+1)^2':>19}{'relative gap':>15}")
for k in range(2 * n):
    got = float(np.sum(w * x ** k))
    want = 1.0 / (k + 1.0) ** 2
    print(f"{k:>4}{got:>18.12f}{want:>19.12f}{abs(got - want) / want:>15.2e}")
    assert abs(got - want) < 1e-12 * want
```

**The discretisation is the whole difficulty, and the obvious choice is the wrong one.**

Gauss-Legendre on 400 points computes the mass of $-\log x$ to only $4\times10^{-6}$, because the
weight is singular at 0 and section 5 of the lesson says exactly what that costs: an algebraic
rate, no matter how many points. Quintupling to 2000 points buys one and a half digits.

The tanh-sinh rule of lesson 66 handles the endpoint singularity without being told about it, and
769 of its points give $3\times10^{-15}$: **nine orders better than 2000 Gauss-Legendre points.**

With that discretisation, **every moment up to degree $2n-1$ is exact to $10^{-15}$**, checked
against the closed form $\int_0^1 -\log(x)x^k\,dx = 1/(k+1)^2$. The rule was built with no
formula for its orthogonal polynomials, only the ability to integrate against the weight.

That is the general recipe, and it is how rules are built for weights arising from a physical
problem rather than from a table.

### 3.4 Adaptive Gauss-Kronrod

Built in lesson 64 exercise 3.2. Here is the comparison against adaptive Simpson on a wider set
of integrands, which is what decides whether it is worth the extra machinery.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import adaptive as ad
from nalib import gaussquad as gq


def adaptive_kronrod(f, lo, hi, tol, n=7, max_depth=50, _counter=None):
    total, panels = 0.0, 0
    stack = [(float(lo), float(hi), float(tol), 0)]
    while stack:
        left, right, budget, depth = stack.pop()
        out = gq.gauss_kronrod_estimate(f, left, right, n, _counter)
        if out["raw_difference"] <= budget or depth >= max_depth:
            total += out["value"]
            panels += 1
        else:
            mid = 0.5 * (left + right)
            stack.append((left, mid, 0.5 * budget, depth + 1))
            stack.append((mid, right, 0.5 * budget, depth + 1))
    return {"value": total, "panels": panels}


cases = (("sin", np.sin, 1.0 - math.cos(1.0)),
         ("sqrt(x)", lambda t: np.sqrt(np.asarray(t, dtype=float)), 2.0 / 3.0),
         ("1/(1e-4 + (x-0.3)^2)",
          lambda t: 1.0 / (1e-4 + (np.asarray(t, dtype=float) - 0.3) ** 2),
          (math.atan(0.7 / 1e-2) - math.atan(-0.3 / 1e-2)) / 1e-2))
print(f"{'integrand':>24}{'Simpson evals':>16}{'error':>12}"
      f"{'Kronrod evals':>16}{'error':>12}{'saving':>9}")
for name, f, exact in cases:
    a, b = [0], [0]
    s = ad.integrate(f, 0.0, 1.0, 1e-10, _counter=a)
    k = adaptive_kronrod(f, 0.0, 1.0, 1e-10, 7, _counter=b)
    print(f"{name:>24}{a[0]:>16}{abs(s['value'] - exact):>12.2e}"
          f"{b[0]:>16}{abs(k['value'] - exact):>12.2e}{a[0] / b[0]:>9.2f}")
```

Gauss-Kronrod wins on every one, by factors of a few in evaluations and by orders of magnitude in
accuracy. **The degree 23 local rule is simply much better than the degree 3 one**, and the extra
10 evaluations per interval are recovered many times over by needing far fewer intervals.

### 4.1 The rate against the Bernstein ellipse

For $f$ analytic inside the Bernstein ellipse $E_\rho$, the ellipse with foci $\pm1$ and
semiaxis sum $\rho$, Gauss-Legendre converges like $\rho^{-2n}$. The $2$ in the exponent is the
degree doubling: the rule is exact on degree $2n-1$, so it behaves like a degree $2n$
approximation.

For a pole at $x = c$ off the interval, $\rho = |c + \sqrt{c^2 - 1}|$.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq

print(f"{'pole at c':>11}{'rho':>10}{'predicted rate':>17}{'fitted rate':>14}{'ratio':>9}")
for c in (1.05, 1.2, 1.5, 2.0, 3.0):
    def f(x, c=c):
        return 1.0 / (c - np.asarray(x, dtype=float))

    exact = math.log((c + 1.0) / (c - 1.0))
    out = gq.convergence(f, exact, -1.0, 1.0, [2, 4, 6, 8, 12, 16, 24, 32])
    rho = c + math.sqrt(c * c - 1.0)
    predicted = 2.0 * math.log(rho)
    print(f"{c:>11.2f}{rho:>10.4f}{predicted:>17.6f}{out['fitted_rate']:>14.6f}"
          f"{out['fitted_rate'] / predicted:>9.4f}")
```

The fitted rate matches $2\log\rho$ closely. **A pole close to the interval gives $\rho$ near 1,
so the rate is near zero and convergence is slow**; moving the pole away increases $\rho$ and the
rate together.

This is the same ellipse that governs Chebyshev interpolation in lesson 56, which is why
Clenshaw-Curtis converges at a comparable rate despite having half the degree, and is exercise
5.3.

### 4.2 The rate for $|x|^a$

An algebraic singularity destroys the geometry and leaves an algebraic rate. **Where the
singularity sits turns out to matter more than how strong it is.**

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq

counts = [2, 4, 8, 16, 32, 64, 128, 256]
print(f"{'case':>24}{'a':>6}{'fitted power':>15}{'a + 1':>9}{'2a + 2':>9}"
      f"{'geometric?':>13}")
for a in (0.5, 1.5, 2.5, 3.5):
    def interior(x, a=a):
        return np.abs(np.asarray(x, dtype=float)) ** a

    out = gq.convergence(interior, 2.0 / (a + 1.0), -1.0, 1.0, counts)
    print(f"{'|x|^a on [-1,1], interior':>24}{a:>6.1f}{out['fitted_power']:>15.4f}"
          f"{a + 1:>9.1f}{2 * a + 2:>9.1f}{str(out['geometric_wins']):>13}")
for a in (0.5, 1.5, 2.5, 3.5):
    def endpoint(x, a=a):
        return np.asarray(x, dtype=float) ** a

    out = gq.convergence(endpoint, 1.0 / (a + 1.0), 0.0, 1.0, counts)
    print(f"{'x^a on [0,1], endpoint':>24}{a:>6.1f}{out['fitted_power']:>15.4f}"
          f"{a + 1:>9.1f}{2 * a + 2:>9.1f}{str(out['geometric_wins']):>13}")
```

The geometric fit loses in every row, which is the correct diagnosis: **there is nothing
geometric here to find.**

The two power columns separate cleanly. An **endpoint** singularity gives $2a+2$, the classical
result. An **interior** singularity gives only $a + 1$, half the exponent.

**That ordering is the reverse of the naive expectation** and the reason is node placement. Gauss
nodes cluster near the endpoints with spacing $O(n^{-2})$ and are sparsest in the middle with
spacing $O(n^{-1})$. A singularity at an endpoint is therefore resolved by nodes that crowd
towards it, and one in the middle is resolved by the coarsest part of the grid.

**Smoothness sets the rate and no rule manufactures what is not there.** Doubling the degree of
precision changes the constant, not the exponent, which is why lesson 66 changes the problem
rather than the rule, and why lesson 64's adaptive routine, which subdivides towards an interior
singularity, is the right tool for the second row.

### 4.3 Gauss against Clenshaw-Curtis

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq


def clenshaw_curtis(n):
    theta = np.pi * np.arange(n + 1) / n
    x = np.cos(theta)
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    for k in range(2, n, 2):
        v -= 2.0 * np.cos(k * theta[1:n]) / (k ** 2 - 1)
    if n % 2 == 0:
        v -= np.cos(n * theta[1:n]) / (n ** 2 - 1)
        w[0] = w[n] = 1.0 / (n ** 2 - 1)
    else:
        w[0] = w[n] = 1.0 / (n ** 2)
    w[1:n] = 2.0 * v / n
    return x[::-1], w[::-1]


cases = (("exp(x)", np.exp, math.e - 1.0 / math.e),
         ("1/(1+25x^2)", lambda t: 1.0 / (1.0 + 25.0 * t ** 2),
          2.0 * math.atan(5.0) / 5.0),
         ("|x|^1.5", lambda t: np.abs(t) ** 1.5, 2.0 / 2.5),
         ("1/(1.05 - x)", lambda t: 1.0 / (1.05 - t), math.log(2.05 / 0.05)))
print(f"{'integrand':>16}" + "".join(f"{n:>13}" for n in (9, 17, 33, 65)))
for name, f, exact in cases:
    row = []
    for points in (9, 17, 33, 65):
        xc, wc = clenshaw_curtis(points - 1)
        cc = abs(float(np.sum(wc * f(xc))) - exact)
        xg, wg = gq.nodes_and_weights(points)
        g = abs(float(np.sum(wg * f(xg))) - exact)
        row.append(cc / g if g > 0 else float("inf"))
    print(f"{name:>16}" + "".join(f"{r:>13.3g}" for r in row))
print("\nentries are Clenshaw-Curtis error divided by Gauss error; below 1 means CC is better")
```

**The answer depends entirely on the integrand, and no single sentence covers the table.**

On $|x|^{1.5}$ the two are within 6 to 60 percent of each other at every size, because exercise
4.2 says the rate is set by the singularity and neither rule can beat it. On $e^x$ they are
both exact past 17 points, so the ratio is 0 or 0.5 and means nothing. On $1/(1+25x^2)$ Gauss
is steadily better by a factor near 2. On the near pole $1/(1.05 - x)$ Gauss pulls ahead
without limit, reaching a factor of 764 by 65 points.

So the folklore that Clenshaw-Curtis is nearly as good as Gauss is **true for some integrands
and badly false for others**, and the discriminator is how close the nearest singularity of $f$
is to the interval. Exercise 5.3 measures that directly.

### 5.1 When the Kronrod extension does not exist

Kronrod's construction solves for $n+1$ new nodes and $2n+1$ weights. Nothing guarantees the
solution is real, inside the interval, or has positive weights, and for some weight functions it
is not.

The classical failures are the Gauss-Laguerre and Gauss-Hermite families, where Kronrod
extensions with real nodes do not exist for most $n$, and the Gegenbauer family for large
parameter.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq

print(f"{'n':>4}{'points':>8}{'all real':>10}{'inside (-1,1)':>16}{'w > 0':>8}"
      f"{'contains Gauss':>16}")
for n in (1, 2, 3, 5, 7, 10, 15, 20, 30):
    out = gq.kronrod_report(n)
    x, w = out["kronrod_nodes"], out["kronrod_weights"]
    print(f"{n:>4}{out['point_count']:>8}{str(bool(np.all(np.isreal(x)))):>10}"
          f"{str(bool(np.min(x) > -1.0 and np.max(x) < 1.0)):>16}"
          f"{str(out['weights_positive']):>8}"
          f"{str(out['gauss_nodes_are_contained']):>16}")
print("\nfor the Legendre weight the extension always exists and behaves; that is a "
      "property of\nthis weight, not a general theorem")
```

For Legendre, which is the case that matters in practice, the extension exists with real interior
nodes and positive weights at every $n$ tested. **That is a fact about the Legendre weight, not a
theorem about Kronrod extensions.**

**What production integrators do about it** is threefold. They tabulate the rules they need
rather than computing them, so a nonexistent case is discovered once by the author. They fall
back to a pair of ordinary Gauss rules, $n$ and $2n+1$ points with nothing reused, when no
extension exists. And for the infinite domains they transform to a finite interval and use
Gauss-Kronrod there, which is lesson 66 section 1.

### 5.2 A Gauss rule from moments, and why Lanczos is better

Given only the moments $\mu_k = \int w\,x^k$, the recurrence coefficients can be recovered by
solving a Hankel system. **That route is catastrophically ill conditioned**, for the same reason
the power basis is: the moments of a positive weight are nearly linearly dependent.

The Lanczos route computes the same coefficients from a sequence of orthogonalisations against a
discretised measure, never forming the moments, and is stable.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq


def coefficients_from_moments(n):
    """Recurrence coefficients for the Legendre weight from its moments, via Hankel."""
    mu = np.asarray([0.0 if k % 2 else 2.0 / (k + 1.0) for k in range(2 * n + 2)])
    H = np.asarray([[mu[i + j] for j in range(n + 1)] for i in range(n + 1)])
    alpha, beta = np.zeros(n), np.zeros(n)
    beta[0] = mu[0]
    for k in range(n):
        Hk = H[:k + 1, :k + 1]
        rhs = np.asarray([mu[k + 1 + i] for i in range(k + 1)])
        c = np.linalg.solve(Hk, rhs)
        alpha[k] = c[k] - (0.0 if k == 0 else 0.0)
        if k > 0:
            # the determinant of the empty leading minor is 1 by convention
            smaller = H[:k - 1, :k - 1] if k > 1 else np.zeros((0, 0))
            beta[k] = float(np.linalg.det(H[:k + 1, :k + 1])
                            * np.linalg.det(smaller)
                            / np.linalg.det(H[:k, :k]) ** 2)
    return alpha, beta


print(f"{'n':>5}{'Hankel condition number':>26}{'digits left':>14}")
for n in (3, 5, 8, 12, 16, 20):
    mu = np.asarray([0.0 if k % 2 else 2.0 / (k + 1.0) for k in range(2 * n + 2)])
    H = np.asarray([[mu[i + j] for j in range(n + 1)] for i in range(n + 1)])
    cond = float(np.linalg.cond(H))
    print(f"{n:>5}{cond:>26.4e}{max(0.0, 16.0 - math.log10(cond)):>14.2f}")
print("\nthe Jacobi matrix route never forms this matrix and has no such condition number")
```

**The Hankel matrix of moments has a condition number growing geometrically**, so by 16 nodes
there is nothing left to solve with. The moments are a complete description of the measure and a
useless representation of it.

The Lanczos and Stieltjes routes work with the measure directly, orthogonalising as they go, and
their conditioning is that of a sequence of orthogonalisations, which is benign. Exercise 3.3
built a rule that way for a weight with no closed form at all.

**The general lesson is the one lesson 47 stated for polynomials and lesson 43 for matrices: the
representation decides the conditioning, and moments are a bad representation** for exactly the
reason the power basis is, since the moments are the inner products of the power basis with
itself.

### 5.3 Why Clenshaw-Curtis is nearly as good

Gauss has degree $2n-1$; Clenshaw-Curtis has degree $n$. **If the degree of precision were the
right statistic, Gauss would need half as many points for the same accuracy.** It does not, and
exercise 4.3 measured ratios near 1.

The explanation is that neither rule's accuracy is controlled by its degree of precision. Both
integrate the polynomial interpolant of $f$ at their nodes, and for an analytic $f$ that
interpolant converges geometrically at a rate set by the Bernstein ellipse of lesson 56. The
error of the quadrature is bounded by the error of that approximation, so both rules inherit the
same $\rho^{-n}$ behaviour.

Gauss gets $\rho^{-2n}$ and Clenshaw-Curtis $\rho^{-n}$, so **asymptotically Gauss is twice as
fast in the exponent**. Trefethen's observation is that the asymptotic regime often arrives late:
for many integrands the two agree closely until the error is already near machine precision, so
the factor of 2 in the exponent never has room to matter.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq


def clenshaw_curtis(n):
    theta = np.pi * np.arange(n + 1) / n
    x = np.cos(theta)
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    for k in range(2, n, 2):
        v -= 2.0 * np.cos(k * theta[1:n]) / (k ** 2 - 1)
    if n % 2 == 0:
        v -= np.cos(n * theta[1:n]) / (n ** 2 - 1)
        w[0] = w[n] = 1.0 / (n ** 2 - 1)
    else:
        w[0] = w[n] = 1.0 / (n ** 2)
    w[1:n] = 2.0 * v / n
    return x[::-1], w[::-1]


c = 1.1
exact = math.log((c + 1.0) / (c - 1.0))
rho = c + math.sqrt(c * c - 1.0)
print(f"f(x) = 1/({c} - x), Bernstein parameter rho = {rho:.6f}")
print(f"{'points':>8}{'Clenshaw-Curtis':>19}{'Gauss':>14}{'ratio':>10}"
      f"{'rho^-n':>12}{'rho^-2n':>12}")
for points in (5, 9, 17, 33, 49):
    xc, wc = clenshaw_curtis(points - 1)
    cc = abs(float(np.sum(wc / (c - xc))) - exact)
    xg, wg = gq.nodes_and_weights(points)
    g = abs(float(np.sum(wg / (c - xg))) - exact)
    print(f"{points:>8}{cc:>19.3e}{g:>14.3e}{cc / max(g, 1e-300):>10.2f}"
          f"{rho ** (-points):>12.2e}{rho ** (-2 * points):>12.2e}")
```

**Gauss tracks $\rho^{-2n}$ closely**, the ratio of its error to that prediction sitting at about
4 at every size. Clenshaw-Curtis falls faster than $\rho^{-n}$ and slower than $\rho^{-2n}$, and
the gap between the two rules widens without limit: 0.8 at 5 points, 21 at 17, and 2450 at 33.

**So on this integrand the asymptotic separation is real and it arrives early.** The factor of 2
in the exponent has room to act, because the pole at $1.1$ makes $\rho = 1.56$ and neither rule
gets near machine precision until well past 30 points.

That does not contradict exercise 4.3, it explains it. The folklore holds when the error reaches
machine precision before the exponents separate, which happens when $\rho$ is large or the
integrand is entire, and both of those were in the 4.3 table. It fails when $\rho$ is close to 1,
which is exactly when the integral is hard and the choice of rule matters.

**The practical conclusion is unchanged and worth stating plainly.** For an easy integrand use
whichever is convenient, and Clenshaw-Curtis is more convenient: nested nodes, and an $O(n\log n)$
construction by the FFT against an eigenproblem for Gauss. For a hard one, meaning a singularity
close to the interval, use Gauss and expect the difference to matter. That is why Clenshaw-Curtis
underlies the `quadgk` style routines, which lean on nesting for adaptivity, and Gauss-Kronrod
underlies QUADPACK, which leans on degree.

## Lesson 66, Improper and Multiple Integrals

### 1.1 Why watching the values settle is not evidence

Truncating at $T$ discards exactly $\int_T^\infty f$. What a caller can see is the **change**
between successive cutoffs, $|I(T_2) - I(T_1)| = \int_{T_1}^{T_2} f$, which is a piece of the
tail rather than the whole of it.

The two are related by how fast $f$ decays, and the relationship can go either way.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq

cutoffs = [10.0, 20.0, 40.0, 80.0, 160.0]
# each entry is (integrand, exact tail from T); the label gives the tail's decay exponent
tails = {
    "exp(-x), tail e^-T": (lambda x: np.exp(-np.asarray(x, dtype=float)),
                           lambda t: math.exp(-t)),
    "1/(1+x)^2, tail T^-1": (lambda x: 1.0 / (1.0 + np.asarray(x, dtype=float)) ** 2,
                             lambda t: 1.0 / (1.0 + t)),
    "0.2/(1+x)^1.2, tail T^-0.2":
        (lambda x: 0.2 / (1.0 + np.asarray(x, dtype=float)) ** 1.2,
         lambda t: (1.0 + t) ** (-0.2)),
}
for name, (f, tail) in tails.items():
    out = mq.truncation_error(f, tail, 0.0, cutoffs)
    print(f"\n{name}")
    print(f"{'T':>8}{'change from last':>19}{'discarded tail':>17}{'change / tail':>16}")
    for t, change, bound in zip(out["cutoff"], out["change_from_previous"],
                                out["tail_bound"]):
        shown = "" if math.isnan(change) else f"{change:.2e}"
        ratio = "" if math.isnan(change) else f"{change / bound:.4g}"
        print(f"{t:>8.0f}{shown:>19}{bound:>17.2e}{ratio:>16}")

print("\nfor a tail decaying like T^(1-p), doubling T gives change / tail = 2^(p-1) - 1:")
for p in (1.2, 2.0, 2.2):
    print(f"   p = {p}: {2.0 ** (p - 1.0) - 1.0:.4f}")
```

For $e^{-x}$ the ratio is astronomical: the change between cutoffs is thousands, then billions,
of times larger than what remains. **Stopping when the values settle is safe here and merely
wasteful.**

For a tail decaying like $T^{1-p}$ the ratio is exactly $2^{p-1} - 1$, and the measurements match
that to three digits. At $p = 2$ it is 1, so the change is the same size as the remaining error:
already no margin. At $p = 1.2$ it is $0.149$, so **the values settle about seven times faster
than the error shrinks.**

That last row is the badly misleading case. Watching the successive values at $T = 80$ and
$T = 160$ shows them moving by $0.053$ and concluding two digits are stable, while $0.362$ of the
answer is still sitting in the discarded tail: **wrong in the first digit, with every appearance
of convergence.** As $p \to 1$ the ratio goes to 0 and the deception becomes total.

### 1.2 Why transforming creates a singularity, and when it does not

Map $[0,\infty)$ to $[0,1)$ by $x = t/(1-t)$. Then $dx = dt/(1-t)^2$ and

$$
\int_0^\infty f(x)\,dx = \int_0^1 \frac{f\!\left(\frac{t}{1-t}\right)}{(1-t)^2}\,dt .
$$

**The Jacobian blows up like $(1-t)^{-2}$**, so the transformed integrand is singular at $t = 1$
unless $f(t/(1-t))$ decays fast enough to cancel it.

The condition is exactly that: the transformed integrand is bounded at $t = 1$ if and only if
$f(x) = O(x^{-2})$ as $x \to \infty$. Faster decay makes it vanish there, which is even better.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq

print(f"{'f(x)':>18}{'decay':>10}" + "".join(f"{t:>14}" for t in (0.9, 0.99, 0.999)))
for name, f, decay in (("exp(-x)", lambda x: np.exp(-x), "e^-x"),
                       ("1/(1+x)^3", lambda x: 1.0 / (1.0 + x) ** 3, "x^-3"),
                       ("1/(1+x)^2", lambda x: 1.0 / (1.0 + x) ** 2, "x^-2"),
                       ("1/(1+x)^1.5", lambda x: 1.0 / (1.0 + x) ** 1.5, "x^-1.5")):
    g, lo, hi = mq.transform_to_finite(f, 0.0, "rational")
    row = [float(g(np.asarray([t]))[0]) for t in (0.9, 0.99, 0.999)]
    print(f"{name:>18}{decay:>10}" + "".join(f"{v:>14.4e}" for v in row))
```

The first two vanish at the endpoint, the third tends to a constant, and the fourth diverges.

**So the transformation trades an infinite range for an endpoint singularity precisely when $f$
decays more slowly than $x^{-2}$**, which is the case where truncation was worst too, by exercise
1.1. The two difficulties are the same difficulty in different coordinates.

### 1.3 What the double exponential rule exploits

Two facts from lesson 63, used together.

**First**, on the whole real line with an integrand decaying fast enough, the trapezoid rule has
no error series at all. Every Euler-Maclaurin term is an endpoint derivative bracket and there
are no endpoints. Exercise 5.1 of lesson 63 makes that precise by Poisson summation: the error is
$O(e^{-2\pi\alpha/h})$ for an integrand analytic in a strip of half width $\alpha$, which is
geometric in $1/h$ rather than algebraic.

**Second**, the substitution $x = \tanh\!\left(\tfrac{\pi}{2}\sinh t\right)$ maps
$(-\infty,\infty)$ onto $(-1,1)$ and makes the transformed integrand decay like $e^{-e^{|t|}}$.
That double exponential decay means the tail can be cut at a modest $|t|$ and the strip of
analyticity is wide, so the first fact applies with a good constant.

**The endpoint singularity then stops mattering for a reason that is almost a trick.** The
endpoints $\pm1$ are the images of $t = \pm\infty$, so the rule never evaluates there. It
approaches them doubly exponentially fast, and any algebraic singularity is overwhelmed by that
approach: $x^{-1/2}$ grows like $e^{e^{|t|}/2}$ while the Jacobian shrinks like $e^{-e^{|t|}}$,
so the product still vanishes.

**No knowledge of the singularity's strength is used anywhere**, which is what separates this
from the substitution of section 2.

### 1.4 What $p/d$ means in ten dimensions

A tensor rule of order $p$ with $n$ points per axis costs $N = n^d$ evaluations and has error
$O(n^{-p}) = O(N^{-p/d})$.

For $p = 4$ and $d = 10$ the exponent is $0.4$. **Halving the error costs a factor of
$2^{1/0.4} = 5.7$ in work.** Getting three more digits costs $1000^{2.5} \approx 3\times10^{7}$
times as much work.

Put concretely: a fourth order rule in ten dimensions converges more slowly than the trapezoid
rule does in one, and more slowly than Monte Carlo, whose exponent is $\tfrac12$ regardless of
$d$.

```python
import numpy as np
import sys
sys.path.insert(0, "src")

print(f"{'dimension':>11}{'order per evaluation':>22}{'work to halve the error':>26}")
for d in (1, 2, 3, 4, 6, 10, 20):
    exponent = 4.0 / d
    print(f"{d:>11}{exponent:>22.3f}{2.0 ** (1.0 / exponent):>26.1f}")
print(f"\nMonte Carlo, any dimension: {0.5:>10.3f}{4.0:>26.1f}")
```

**Monte Carlo overtakes a fourth order tensor rule at $d = 8$** on this measure, and everything
above it. That is the arithmetic behind section 6 of the lesson, and it is why high dimensional
integration is a different subject rather than a harder version of the same one.

### 1.5 What the two integrands must differ in

**Smoothness.**

The tensor rule's rate per evaluation is $\min(p, s)/d$ where $p$ is the rule's order and $s$ is
what the integrand's smoothness allows. Monte Carlo's is $\tfrac12$ whatever happens, provided
the variance is finite.

So the crossing is where $\min(p, s)/d = \tfrac12$, that is $d = 2\min(p, s)$.

For a product of cosines, analytic and separable, $s$ is effectively unbounded and the tensor
rule converges geometrically, so the crossing is far away and was not reached by twelve
dimensions. For a product of $|x - \tfrac12|^{1/2}$ each axis has a kink, $s \approx 1.5$, and
the crossing is at about $d = 3$, which is what the lesson measures.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq

print(f"{'integrand':>10}{'dimension':>11}{'order per axis':>17}"
      f"{'order per evaluation':>22}")
for kind in ("smooth", "rough"):
    out = mq.curse_of_dimensionality([1, 2, 3, 4], kind=kind)
    for d, axis, ev in zip(out["dimension"], out["order_per_axis"],
                           out["order_per_evaluation"]):
        print(f"{kind:>10}{d:>11}{axis:>17.3f}{ev:>22.3f}")
```

The smooth integrand keeps an order per axis of about 4.8 and the rough one about 2.4, and each
is divided by $d$. **The rough one crosses $\tfrac12$ between three and four dimensions; the
smooth one has not crossed by four.**

### 2.1 The substitution that removes an algebraic singularity

Let $f(x) \sim C(x-a)^{-p}$ near $a$, with $0 \le p < 1$ so the integral converges. Substitute

$$
x = a + u^{m}, \qquad m = \frac{1}{1-p},
\qquad dx = m\,u^{m-1}\,du .
$$

Then near $u = 0$,

$$
f(x)\,\frac{dx}{du} \sim C\,u^{-mp}\cdot m\,u^{m-1} = Cm\,u^{m(1-p) - 1} = Cm\,u^{0} = Cm ,
$$

using $m(1-p) = 1$. **The transformed integrand tends to the finite constant $Cm$**, so it is
bounded, and the singularity is gone.

The limits become $u \in [0, (b-a)^{1/m}]$.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq

print(f"{'p':>6}{'m = 1/(1-p)':>14}{'transformed integrand near u = 0':>36}")
for p in (0.25, 0.5, 0.75, 0.9):
    def f(x, p=p):
        return np.asarray(x, dtype=float) ** (-p)

    g, lo, hi = mq.remove_endpoint_singularity(f, 0.0, 1.0, p)
    small = [float(g(np.asarray([u]))[0]) for u in (1e-3, 1e-6, 1e-9)]
    print(f"{p:>6.2f}{1.0 / (1.0 - p):>14.4f}"
          + "".join(f"{v:>12.6f}" for v in small))
print("\nconstant in u, which is what bounded means")
```

Each row is constant across nine orders of magnitude in $u$, confirming the exponent cancellation
exactly.

### 2.2 The double exponential decay and geometric convergence

With $x = \tanh u$ and $u = \tfrac{\pi}{2}\sinh t$, the Jacobian is

$$
\frac{dx}{dt} = \frac{\pi}{2}\,\frac{\cosh t}{\cosh^2\!\left(\tfrac{\pi}{2}\sinh t\right)} .
$$

For large $|t|$, $\sinh t \sim \tfrac12 e^{|t|}$, $\cosh t \sim \tfrac12 e^{|t|}$ and
$\operatorname{sech}^2 u \sim 4e^{-2u}$ with $u \sim \tfrac{\pi}{4}e^{|t|}$, so

$$
\frac{dx}{dt} \sim \frac{\pi}{2}\cdot\frac{e^{|t|}}{2}\cdot 4e^{-2u}
= \pi\, e^{|t|}\,e^{-\frac{\pi}{2}e^{|t|}} ,
$$

which decays like $e^{-e^{|t|}}$ up to a polynomial factor: **doubly exponential**.

For a bounded $f$ the transformed integrand inherits that decay. For $f$ with an algebraic
singularity at the endpoint, $f$ grows only exponentially in $t$ (like $e^{p\pi e^{|t|}/2}$ for
$(1-x)^{-p}$ with $p < 1$), and the Jacobian's $e^{-\pi e^{|t|}/2}$ still wins.

Now apply lesson 63 exercise 5.1: the trapezoid rule on the whole line for a function analytic in
a strip of half width $\alpha$ has error $O(e^{-2\pi\alpha/h})$. **Geometric in $1/h$**, so
halving $h$ squares the error, and since the point count is proportional to $1/h$, the error is
geometric in the point count.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq


def jacobian(t):
    u = 0.5 * math.pi * np.sinh(t)
    return 0.5 * math.pi * np.cosh(t) / np.cosh(u) ** 2


print(f"{'t':>6}{'Jacobian':>16}{'pi e^t exp(-pi e^t / 2)':>26}{'ratio':>10}")
for t in (1.0, 2.0, 3.0, 4.0, 5.0):
    j = float(jacobian(np.asarray([t]))[0])
    predicted = math.pi * math.exp(t) * math.exp(-0.5 * math.pi * math.exp(t))
    print(f"{t:>6.1f}{j:>16.3e}{predicted:>26.3e}{j / predicted:>10.4f}")

print(f"\n{'level':>7}{'points':>8}{'error on 1/sqrt(x)':>22}{'ratio to previous':>20}")
previous = None
for k in range(1, 6):
    out = mq.tanh_sinh(lambda x: 1.0 / np.sqrt(np.asarray(x, dtype=float)), 0.0, 1.0, k)
    e = abs(out["value"] - 2.0)
    ratio = "" if previous is None or e == 0 else f"{previous / e:.2e}"
    print(f"{k:>7}{out['points']:>8}{e:>22.3e}{ratio:>20}")
    previous = e
```

The ratio settles towards $\tfrac12$ rather than 1, because the sub leading terms in
$\cosh$ and $\sinh$ contribute a factor of that size; the exponent, which is what matters, is
exactly right.

The error table below it falls by eight orders of magnitude for one doubling of the point count,
which no algebraic rate does.

### 2.3 Why $2/(1+e^{-2u})$ and not $1 + \tanh u$

Algebraically,

$$
1 + \tanh u = 1 + \frac{e^u - e^{-u}}{e^u + e^{-u}}
= \frac{2e^u}{e^u + e^{-u}}
= \frac{2}{1 + e^{-2u}} .
$$

**In floating point they are completely different.**

For $u \ll 0$, $\tanh u$ is within rounding of $-1$. Computing $1 + \tanh u$ adds two numbers of
opposite sign and nearly equal magnitude, so every significant digit cancels: the result is a
multiple of $\varepsilon$ with no correct digits.

The second form has no subtraction. $e^{-2u}$ is a large positive number, $1 + e^{-2u}$ is
accurate to a rounding unit, and the quotient is accurate to a rounding unit. **Full relative
precision, down to the underflow limit.**

```python
import math

import numpy as np

print(f"{'u':>8}{'1 + tanh(u)':>22}{'2 / (1 + exp(-2u))':>24}{'relative gap':>16}")
for u in (-5.0, -10.0, -20.0, -30.0, -40.0):
    naive = 1.0 + math.tanh(u)
    stable = 2.0 / (1.0 + math.exp(-2.0 * u))
    gap = abs(naive - stable) / stable
    print(f"{u:>8.1f}{naive:>22.12e}{stable:>24.12e}{gap:>16.2e}")
```

By $u = -20$ the naive form has lost every digit, and by $u = -30$ it returns exactly zero while
the true value is $10^{-26}$.

**That is the difference between a tanh-sinh rule that stalls at $10^{-8}$ and one that reaches
machine precision**, because those far out nodes are exactly the ones approaching the singular
endpoint.

### 2.4 The Monte Carlo rate

Let $X_1, \dots, X_N$ be independent and uniform on a region $\Omega$ of volume $V$, and let
$\hat I = \frac{V}{N}\sum_i f(X_i)$.

**Unbiasedness.** $E[f(X_i)] = \frac{1}{V}\int_\Omega f$, so $E[\hat I] = \int_\Omega f$.

**Variance.** The samples are independent, so

$$
\operatorname{Var}(\hat I) = \frac{V^2}{N^2}\sum_i \operatorname{Var}(f(X_i))
= \frac{V^2\sigma^2}{N},
\qquad
\sigma^2 = \frac{1}{V}\int_\Omega f^2 - \left(\frac{1}{V}\int_\Omega f\right)^2 .
$$

**So the standard deviation is $V\sigma/\sqrt{N}$**, and by the central limit theorem
$\sqrt{N}(\hat I - I)/(V\sigma)$ converges in distribution to a standard normal.

The dimension appears **nowhere**. It enters only through $\sigma$, which is a property of $f$ on
$\Omega$, not of how many coordinates $\Omega$ has.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq

VARIANCE_SAMPLES = 200000     # one long run, only to estimate sigma
CHECK_SAMPLES = 10000         # the run whose spread is compared against sigma / sqrt(N)
CHECK_REPEATS = 20            # independent repeats of that run

print(f"{'dimension':>11}{'fitted rate':>14}{'sigma':>12}{'predicted sd at N=10^4':>25}"
      f"{'measured rms':>15}")
for d in (1, 2, 4, 8, 16):
    f, exact = mq.cube_integrand("smooth", d)
    out = mq.monte_carlo_rate(f, exact, [(0.0, 1.0)] * d,
                              [400, 1600, 6400, 25600, 102400], repeats=60)
    single = mq.monte_carlo(f, [(0.0, 1.0)] * d, VARIANCE_SAMPLES, seed=1)
    sigma = single["standard_error"] * math.sqrt(VARIANCE_SAMPLES)
    errors = [abs(mq.monte_carlo(f, [(0.0, 1.0)] * d, CHECK_SAMPLES,
                                 seed=500 + r)["value"] - exact)
              for r in range(CHECK_REPEATS)]
    rms = float(np.sqrt(np.mean(np.asarray(errors) ** 2)))
    print(f"{d:>11}{out['fitted_rate']:>14.4f}{sigma:>12.6f}"
          f"{sigma / math.sqrt(CHECK_SAMPLES):>25.6f}{rms:>15.6f}")
```

The fitted rate is close to $\tfrac12$ in every dimension, and the measured root mean square
error matches $\sigma/\sqrt{N}$ closely.

**The fit needs many repeats to say that much.** With 12 repeats it reads between 0.43 and 0.48
and looks like a systematic deviation; with 60 it reads between 0.48 and 0.51. A single Monte
Carlo run has a fluctuation of the same size as the quantity being fitted, which is the whole
character of the method and is worth meeting once in a measurement rather than only in a
theorem. **The constant shrinks with $d$ here** because the integrand is a
product of cosines whose values cluster more tightly as more factors multiply, which is a
property of this integrand and not a general effect.

### 2.5 Coordinate degree against total degree

The tensor product of one dimensional rules exact on degree $m$ per axis is exact on

$$
x_1^{k_1}x_2^{k_2}\cdots x_d^{k_d}
\qquad\text{whenever}\qquad
\max_i k_i \le m ,
$$

because the integral factorises and each factor is handled by its own axis rule.

That is **coordinate degree** $m$: the maximum over axes. It is much weaker than **total degree**
$m$, which asks $\sum_i k_i \le m$.

The two sets differ enormously. Coordinate degree $m$ in $d$ dimensions contains $(m+1)^d$
monomials; total degree $m$ contains $\binom{m+d}{d}$, which is far fewer.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq

print(f"{'d':>4}{'m':>4}{'coordinate degree set':>24}{'total degree set':>20}"
      f"{'points used':>14}")
for d in (1, 2, 3, 5):
    for m in (3, 5):
        coordinate = (m + 1) ** d
        total = math.comb(m + d, d)
        print(f"{d:>4}{m:>4}{coordinate:>24}{total:>20}{(m + 1) ** d:>14}")

print(f"\n{'monomial':>22}{'coordinate degree':>20}{'total degree':>15}"
      f"{'tensor exact?':>16}")
d, per_axis = 3, 3
points, weights = mq.tensor_rule(mq.gauss_rule_1d, [2] * d, [(0.0, 1.0)] * d)
for powers in ((3, 0, 0), (1, 1, 1), (3, 3, 3), (4, 0, 0)):
    got = float(np.sum(weights * np.prod(points ** np.asarray(powers), axis=1)))
    want = float(np.prod([1.0 / (k + 1.0) for k in powers]))
    exact = abs(got - want) < 1e-12
    print(f"{str(powers):>22}{max(powers):>20}{sum(powers):>15}{str(exact):>16}")
```

A 2 point Gauss rule per axis has coordinate degree 3, so it integrates $x^3y^3z^3$, of total
degree 9, exactly. And it fails on $x^4$, of total degree 4.

**So the tensor rule is exact on a set that is large in one sense and lopsided in another.** It
spends its accuracy on high powers of single variables, which is rarely what an integrand needs.
Sparse grids, exercise 3.4, are built to target total degree instead, and that is precisely where
their advantage comes from.

### 3.1 Gauss-Laguerre and Gauss-Hermite against transformation

The comparison has to be at equal evaluation counts, so the tanh-sinh levels are chosen to match
the Laguerre node counts: level 1 gives 25 points and level 2 gives 49.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import gaussquad as gq
from nalib import multiquad as mq

cases = (("exp(-x) / (1+x)",
          lambda x: 1.0 / (1.0 + np.asarray(x, dtype=float)),
          lambda x: np.exp(-np.asarray(x, dtype=float))
          / (1.0 + np.asarray(x, dtype=float)), 0.5963473623231941),
         ("exp(-x) cos(x)",
          lambda x: np.cos(np.asarray(x, dtype=float)),
          lambda x: np.exp(-np.asarray(x, dtype=float))
          * np.cos(np.asarray(x, dtype=float)), 0.5),
         ("1/(1+x)^3, no exp factor",
          lambda x: np.exp(np.asarray(x, dtype=float))
          / (1.0 + np.asarray(x, dtype=float)) ** 3,
          lambda x: 1.0 / (1.0 + np.asarray(x, dtype=float)) ** 3, 0.5))
print(f"{'integrand':>26}{'points':>8}{'Gauss-Laguerre':>18}"
      f"{'transform + tanh-sinh':>24}")
for name, g, whole, exact in cases:
    for n, level in ((25, 1), (49, 2)):
        x, w = gq.nodes_and_weights(n, "laguerre")
        laguerre = abs(float(np.sum(w * np.asarray(g(x), dtype=float))) - exact)
        mapped, lo, hi = mq.transform_to_finite(whole, 0.0, "exponential")
        out = mq.tanh_sinh(mapped, lo, hi, level)
        assert out["points"] == n
        print(f"{name:>26}{n:>8}{laguerre:>18.3e}{abs(out['value'] - exact):>24.3e}")
```

**When the integrand really is $e^{-x}g(x)$ with $g$ smooth, Gauss-Laguerre is unbeatable.** On
$e^{-x}\cos x$ it reaches $6\times10^{-17}$ with 25 points, thirteen orders of magnitude ahead of
the transformed route at the same cost.

**When it is not, Gauss-Laguerre does not merely lose, it explodes.** The third row has no
exponential factor, so using Laguerre means approximating $g(x) = e^{x}/(1+x)^3$, which grows
without bound. At 25 nodes that is already the worst entry in the table; at 49 nodes the answer
is $2.7\times10^{17}$ when the truth is $0.5$.

The transformed route is unspectacular and never fails, which is the usual shape of the
trade. **The weight function is part of the problem statement rather than a free choice**: match
it when you can, and transform when you cannot.

### 3.2 exp-sinh and sinh-sinh

The same idea, with substitutions matched to the domain.

- **tanh-sinh** on $(a,b)$: $x = \tanh\!\left(\tfrac{\pi}{2}\sinh t\right)$, both ends finite.
- **exp-sinh** on $(0,\infty)$: $x = \exp\!\left(\tfrac{\pi}{2}\sinh t\right)$.
- **sinh-sinh** on $(-\infty,\infty)$: $x = \sinh\!\left(\tfrac{\pi}{2}\sinh t\right)$.

Each makes the transformed integrand decay doubly exponentially at both ends of the $t$ line, so
the same plain trapezoid rule applies.

```python
import math

import numpy as np


def exp_sinh(f, level=6, limit=4.0):
    """Double exponential quadrature on (0, infinity)."""
    h = 2.0 ** (-level)
    t = h * np.arange(-int(math.ceil(limit / h)), int(math.ceil(limit / h)) + 1)
    u = 0.5 * math.pi * np.sinh(t)
    x = np.exp(u)
    w = h * x * 0.5 * math.pi * np.cosh(t)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    values = np.asarray(f(x[good]), dtype=float)
    ok = np.isfinite(values)
    return float(np.sum(w[good][ok] * values[ok]))


def sinh_sinh(f, level=6, limit=4.0):
    """Double exponential quadrature on the whole real line."""
    h = 2.0 ** (-level)
    t = h * np.arange(-int(math.ceil(limit / h)), int(math.ceil(limit / h)) + 1)
    u = 0.5 * math.pi * np.sinh(t)
    x = np.sinh(u)
    w = h * np.cosh(u) * 0.5 * math.pi * np.cosh(t)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    values = np.asarray(f(x[good]), dtype=float)
    ok = np.isfinite(values)
    return float(np.sum(w[good][ok] * values[ok]))


print(f"{'integral':>34}{'exact':>20}{'computed':>20}{'error':>12}")
for name, f, exact, rule in (
        ("exp(-x) over (0, inf)", lambda x: np.exp(-x), 1.0, exp_sinh),
        ("1/(1+x)^2 over (0, inf)", lambda x: 1.0 / (1.0 + x) ** 2, 1.0, exp_sinh),
        ("1/(x^0.5 (1+x)) over (0, inf)",
         lambda x: 1.0 / (np.sqrt(x) * (1.0 + x)), math.pi, exp_sinh),
        ("exp(-x^2) over R", lambda x: np.exp(-x * x), math.sqrt(math.pi), sinh_sinh),
        ("1/(1+x^2) over R", lambda x: 1.0 / (1.0 + x * x), math.pi, sinh_sinh)):
    got = rule(f)
    print(f"{name:>34}{exact:>20.15f}{got:>20.15f}{abs(got - exact):>12.2e}")
```

All five reach machine precision, including the third, which is singular at $0$ **and** has an
infinite range, and the fifth, whose tail decays only like $x^{-2}$.

**That last one is the case truncation handled worst in exercise 1.1**, and here it costs nothing,
because the substitution reaches infinity in a bounded number of points rather than approaching
it.

### 3.3 Importance sampling

Plain Monte Carlo has variance $\sigma^2/N$ with $\sigma$ the variance of $f$ under the uniform
distribution. If $f$ is concentrated, most samples contribute nothing and $\sigma$ is large.

Sampling instead from a density $q$ and averaging $f/q$ gives an unbiased estimate with variance
set by the variance of $f/q$. **Choosing $q$ proportional to $|f|$ makes that ratio constant and
the variance zero**, which is unattainable but says what to aim at.

```python
import math

import numpy as np


def concentrated(x):
    """A narrow Gaussian bump on [0, 1], integral known in closed form."""
    return np.exp(-((x - 0.3) / 0.02) ** 2)


exact = 0.02 * math.sqrt(math.pi) * 0.5 * (math.erf(0.7 / 0.02) + math.erf(0.3 / 0.02))
rng = np.random.default_rng(42)
N = 20000
print(f"exact {exact:.12e}\n")
print(f"{'sampler':>28}{'estimate':>18}{'error':>12}{'sd of the estimate':>21}")

plain = concentrated(rng.random(N))
print(f"{'uniform':>28}{float(np.mean(plain)):>18.10e}"
      f"{abs(float(np.mean(plain)) - exact):>12.2e}"
      f"{float(np.std(plain, ddof=1)) / math.sqrt(N):>21.2e}")

for width in (0.05, 0.02):
    # sample from a Gaussian centred on the bump, truncated to [0, 1] by rejection
    draws = []
    while len(draws) < N:
        batch = 0.3 + width * rng.standard_normal(2 * N)
        draws.extend(batch[(batch > 0.0) & (batch < 1.0)].tolist())
    z = np.asarray(draws[:N])
    mass = 0.5 * (math.erf((1.0 - 0.3) / (width * math.sqrt(2.0)))
                  + math.erf(0.3 / (width * math.sqrt(2.0))))
    q = np.exp(-0.5 * ((z - 0.3) / width) ** 2) / (width * math.sqrt(2.0 * math.pi) * mass)
    ratio = concentrated(z) / q
    print(f"{f'Gaussian, width {width}':>28}{float(np.mean(ratio)):>18.10e}"
          f"{abs(float(np.mean(ratio)) - exact):>12.2e}"
          f"{float(np.std(ratio, ddof=1)) / math.sqrt(N):>21.2e}")
```

The uniform sampler wastes almost every draw: the bump occupies about 4 percent of the interval,
so 96 percent of the samples return essentially zero and contribute only variance.

**Matching the proposal to the integrand cuts the standard deviation by orders of magnitude**,
and matching it exactly, width $0.02$ against the bump's own width, is close to the zero variance
ideal.

The catch is that choosing $q$ needs to know where $f$ is large, which is the same knowledge
lesson 64's breakpoints needed. **The methods differ; the requirement does not.**

### 3.4 A sparse grid

Smolyak's construction combines tensor rules of **low total order** rather than one tensor rule of
uniform order. In $d$ dimensions at level $\ell$ it uses

$$
A(\ell, d) = \sum_{\ell - d + 1 \le |\mathbf{k}| \le \ell}
(-1)^{\ell - |\mathbf{k}|}\binom{d-1}{\ell - |\mathbf{k}|}
\left(Q_{k_1}\otimes\cdots\otimes Q_{k_d}\right),
$$

where $Q_k$ is a one dimensional rule of level $k$. The point count grows like
$O(2^\ell \ell^{d-1})$ instead of $O(2^{\ell d})$.

```python
import itertools
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq


def smolyak(d, level, rule_1d=None):
    """Smolyak sparse grid points and weights on the unit cube."""
    rule = mq.gauss_rule_1d if rule_1d is None else rule_1d
    points, weights = [], []
    for total in range(max(level - d + 1, d), level + 1):
        coefficient = ((-1) ** (level - total)
                       * math.comb(d - 1, level - total))
        if coefficient == 0:
            continue
        for k in itertools.product(range(1, level + 1), repeat=d):
            if sum(k) != total:
                continue
            axes = [rule(2 ** ki - 1, 0.0, 1.0) for ki in k]
            grids = np.meshgrid(*[a[0] for a in axes], indexing="ij")
            wgrids = np.meshgrid(*[a[1] for a in axes], indexing="ij")
            p = np.stack([g.ravel() for g in grids], axis=1)
            w = np.ones(p.shape[0])
            for g in wgrids:
                w = w * g.ravel()
            points.append(p)
            weights.append(coefficient * w)
    return np.concatenate(points, axis=0), np.concatenate(weights)


print(f"{'d':>4}{'level':>7}{'sparse points':>15}{'full tensor':>14}"
      f"{'sparse error':>15}{'full error':>13}{'sparse wins':>13}")
for d in (2, 4, 6, 8):
    f, exact = mq.cube_integrand("smooth", d)
    for level in (d + 2,):
        p, w = smolyak(d, level)
        sparse = abs(float(np.sum(w * f(p))) - exact)
        per_axis = max(2, int(round(p.shape[0] ** (1.0 / d))))
        full = abs(mq.integrate_box(f, [(0.0, 1.0)] * d, [per_axis] * d) - exact)
        print(f"{d:>4}{level:>7}{p.shape[0]:>15}{per_axis ** d:>14}"
              f"{sparse:>15.2e}{full:>13.2e}{str(sparse < full):>13}")
```

**The sparse grid loses at every row measured, and saying so is the point.**

At $d = 2$ it uses 29 points to reach $3\times10^{-8}$ while a tensor grid of 25 points reaches
$6\times10^{-13}$. Extending the sweep to $d = 4, 6, 8$ at comparable point counts keeps the same
ordering.

That does not contradict Smolyak's theorem, which is asymptotic in the level and describes a rate
rather than a value at any particular size. Here the integrand is a product of cosines, analytic
and separable, and the tensor rule of Gauss rules converges **geometrically** on it, so there is
no algebraic rate for the sparse construction to improve on. The advantage appears where the
tensor rate is algebraic and the dimension is high, which needs levels beyond what a direct
construction of the grid can reach.

Two further costs are worth recording. The weights are a signed combination, so they are not all
positive and the rule does not inherit the stability of the pieces it is built from. And the
construction itself is expensive: the point count above is before removing duplicates, which a
production implementation would do.

**The honest summary is that sparse grids are a real answer to the curse and that this
measurement does not demonstrate it.** Demonstrating it needs a rough integrand and a dimension
around 10, and exercise 5.3 shows what else it needs.

### 3.5 Quasi Monte Carlo

A low discrepancy sequence fills the cube more evenly than random points, and the
Koksma-Hlawka inequality bounds the error by the sequence's discrepancy times the integrand's
variation, giving $O(N^{-1}(\log N)^d)$ instead of $O(N^{-1/2})$.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq


def halton(n, d):
    """The first n points of the Halton sequence in d dimensions."""
    primes = []
    candidate = 2
    while len(primes) < d:
        if all(candidate % p for p in primes):
            primes.append(candidate)
        candidate += 1
    out = np.empty((n, d))
    for j, base in enumerate(primes):
        for i in range(n):
            value, f, k = 0.0, 1.0 / base, i + 1
            while k:
                value += (k % base) * f
                k //= base
                f /= base
            out[i, j] = value
    return out


MC_REPEATS = 12           # independent Monte Carlo runs averaged at each N

print(f"{'d':>4}{'N':>8}{'Monte Carlo rms':>18}{'Halton':>14}{'ratio':>10}")
for d in (1, 2, 4, 8):
    f, exact = mq.cube_integrand("smooth", d)
    for n in (1000, 8000):
        mc = float(np.sqrt(np.mean([
            (mq.monte_carlo(f, [(0.0, 1.0)] * d, n, seed=700 + r)["value"] - exact) ** 2
            for r in range(MC_REPEATS)])))
        q = abs(float(np.mean(f(halton(n, d)))) - exact)
        print(f"{d:>4}{n:>8}{mc:>18.3e}{q:>14.3e}{mc / max(q, 1e-300):>10.1f}")
```

Halton is far more accurate at every size and dimension tested. **The rate is better, not just
the constant**: doubling $N$ roughly halves the quasi Monte Carlo error and only reduces the
Monte Carlo one by $\sqrt{2}$.

The $(\log N)^d$ factor in the bound means the advantage erodes at very high dimension, and it is
also why quasi Monte Carlo has no useful error estimate: there is no analogue of the sample
standard deviation, because the points are not random.

### 4.1 The level requirement against the singularity strength

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq

print(f"{'p in x^-p':>12}{'exact':>14}{'level for 1e-12':>18}{'points':>10}"
      f"{'error there':>14}")
for p in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
    def f(x, p=p):
        return np.asarray(x, dtype=float) ** (-p)

    exact = 1.0 / (1.0 - p)
    found = None
    for level in range(1, 11):
        out = mq.tanh_sinh(f, 0.0, 1.0, level)
        if abs(out["value"] - exact) <= 1e-12 * exact:
            found = (level, out["points"], abs(out["value"] - exact))
            break
    if found:
        print(f"{p:>12.2f}{exact:>14.6f}{found[0]:>18}{found[1]:>10}{found[2]:>14.2e}")
    else:
        out = mq.tanh_sinh(f, 0.0, 1.0, 10)
        print(f"{p:>12.2f}{exact:>14.6f}{'not by 10':>18}{out['points']:>10}"
              f"{abs(out['value'] - exact):>14.2e}")
```

**The level requirement is completely flat from $p = 0.25$ to $p = 0.95$: two levels, 49 points,
every time.** A singularity strong enough to make the integral twenty times larger costs nothing
extra, which is the practical content of double exponential decay overwhelming algebraic growth.

**And then at $p = 0.99$ it fails outright**, still wrong by $0.18$ after ten levels and 12289
points.

The reason is that the double exponential decay is beaten only asymptotically. The transformed
integrand behaves like $e^{(p - 1)\frac{\pi}{2}e^{|t|}}$ near the singular end, and at
$p = 0.99$ that exponent is $-0.01\cdot\frac{\pi}{2}e^{|t|}$: still decaying, but a hundred
times more slowly, so the tail cut at $|t| \approx 6$ discards far too much. Pushing the cut out
would fix it, at a cost that grows as $1/(1-p)$.

So the method's indifference to the singularity's strength is real over a wide range and not
unconditional, and the boundary is at $1 - p$ comparable to the reciprocal of the tail cut.

### 4.2 The crossing dimension against smoothness

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq

CUBE = dict(mq.CUBE_INTEGRANDS)
print(f"{'a in |x-1/2|^a':>16}{'one axis exact':>17}{'crossing d':>13}"
      f"{'2(a+1)':>10}")
for a in (0.5, 1.0, 1.5, 2.5):
    one_axis = 2.0 * 0.5 ** (a + 1.0) / (a + 1.0)
    mq.CUBE_INTEGRANDS["sweep"] = (
        lambda p, a=a: np.prod(np.abs(p - 0.5) ** a, axis=1), one_axis)
    out = mq.monte_carlo_crossover([1, 2, 3, 4, 5, 6, 8, 10], 4096, 8, "sweep")
    crossing = out["first_dimension_monte_carlo_wins"]
    print(f"{a:>16.1f}{one_axis:>17.8f}{str(crossing):>13}{2.0 * (a + 1.0):>10.1f}")
mq.CUBE_INTEGRANDS.clear()
mq.CUBE_INTEGRANDS.update(CUBE)
```

The crossing dimension rises with $a$, which is the prediction of exercise 1.5: the tensor rule's
rate per evaluation is $\min(p, a+1)/d$ and Monte Carlo's is $\tfrac12$, so the crossing is near
$d = 2\min(p, a+1)$.

**Smoothing the integrand by one derivative pushes the crossing up by about two dimensions**, and
once $a$ is large enough that the rule's own order $p$ binds instead, the crossing stops moving.

### 4.3 The Monte Carlo constant against dimension

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq

VARIANCE_SAMPLES = 200000     # one long run per case, only to estimate sigma

print(f"{'kind':>8}{'d':>4}{'fitted rate':>14}{'sigma':>14}{'exact value':>16}"
      f"{'sigma / value':>16}")
for kind in ("smooth", "rough"):
    for d in (1, 2, 4, 8, 16):
        f, exact = mq.cube_integrand(kind, d)
        out = mq.monte_carlo_rate(f, exact, [(0.0, 1.0)] * d,
                                  [400, 1600, 6400, 25600, 102400], repeats=60)
        single = mq.monte_carlo(f, [(0.0, 1.0)] * d, VARIANCE_SAMPLES, seed=3)
        sigma = single["standard_error"] * math.sqrt(VARIANCE_SAMPLES)
        print(f"{kind:>8}{d:>4}{out['fitted_rate']:>14.4f}{sigma:>14.6f}"
              f"{exact:>16.8f}{sigma / exact:>16.4f}")
```

**The fitted rate is $\tfrac12$ everywhere to within the noise of the fit**, in both integrands
and at every dimension, exactly as the central limit theorem requires.

The constant $\sigma$ falls with $d$ for both, but the **relative** constant $\sigma/I$ rises,
which is the quantity that matters when the answer itself is shrinking. That is the honest
version of "the rate does not change but the problem still gets harder": Monte Carlo needs
$N \sim (\sigma/(\epsilon I))^2$ samples for a relative accuracy $\epsilon$, and $\sigma/I$ grows
with $d$.

### 5.1 Why double exponential and not single

The substitution $x = \tanh t$ alone already maps $\mathbb{R}$ onto $(-1,1)$ and pushes the
endpoints to infinity. Its Jacobian is $\operatorname{sech}^2 t \sim 4e^{-2|t|}$, decaying
**singly** exponentially.

Two things go wrong.

**The tail is much longer.** To reach $10^{-16}$ the single exponential needs $|t| \approx 18$,
against about 3 for the double exponential. At a fixed $h$ that is six times as many points.

**The poles sit closer.** $\tanh t$ has poles at $t = \pm i\pi/2$, so the transformed integrand is
analytic only in a strip of half width $\pi/2$ at best, and less once $f$ contributes its own
singularities. By lesson 63 exercise 5.1 the trapezoid error is $O(e^{-2\pi\alpha/h})$ with
$\alpha$ that half width, so the rate is fixed and modest.

The extra $\sinh$ pushes the decay to $e^{-e^{|t|}}$, which shortens the tail dramatically. It
also moves the effective singularities: the composite $\tanh(\tfrac\pi2\sinh t)$ has its nearest
problem where $\tfrac\pi2\sinh t = \pm i\pi/2$, that is $\sinh t = \pm i$, at $t = \pm i\pi/2$
again, but the integrand's decay near that strip is now so violent that the effective $\alpha$
is larger.

```python
import math

import numpy as np


def single_exponential(f, level, limit):
    h = 2.0 ** (-level)
    t = h * np.arange(-int(limit / h), int(limit / h) + 1)
    x = np.tanh(t)
    w = h / np.cosh(t) ** 2
    keep = np.isfinite(w) & (w > 0)
    values = np.asarray(f(x[keep]), dtype=float)
    ok = np.isfinite(values)
    return float(np.sum(w[keep][ok] * values[ok])), int(np.sum(keep))


import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq

exact = math.pi
print(f"{'level':>7}{'single exp points':>20}{'single exp error':>19}"
      f"{'double exp points':>20}{'double exp error':>19}")
for level in (2, 3, 4, 5):
    s, sp = single_exponential(lambda x: 1.0 / np.sqrt(1.0 - x * x), level, 18.0)
    d = mq.tanh_sinh(lambda x, dl, dh: 1.0 / np.sqrt(dl * dh * 0.5 * 2.0),
                     -1.0, 1.0, level, None, True)
    print(f"{level:>7}{sp:>20}{abs(s - exact):>19.3e}"
          f"{d['points']:>20}{abs(d['value'] - exact):>19.3e}")
```

**The single exponential rule does not improve at all past level 2.** It sits at
$5.9\times10^{-8}$ while quadrupling its point count, and the double exponential rule reaches
machine precision with a quarter as many points.

The stall is the tail cut, and it can be predicted exactly. With $x = \tanh t$ the transformed
integrand for $1/\sqrt{1-x^2}$ is $\operatorname{sech} t \sim 2e^{-|t|}$, so cutting at
$|t| = 18$ discards about $2e^{-18} = 3\times10^{-8}$, which is what the column shows.

**Refining $h$ cannot help, because the error is not a discretisation error.** Fixing it means
moving the cut, and reaching machine precision needs $|t| \approx 37$, doubling the point count
again. The double exponential rule reaches the same place at $|t| \approx 3$.

### 5.2 Optimal truncation of the double exponential rule

Two errors compete, and balancing them fixes both $h$ and the cut point.

**Discretisation.** By lesson 63 exercise 5.1, the trapezoid rule on the whole $t$ line has error
$O(e^{-2\pi\alpha/h})$, which **falls** as $h$ shrinks.

**Truncation.** Cutting the tail at $|t| = T$ discards an integrand of size $e^{-e^{T}}$, so the
error is $O(e^{-e^{T}})$, which falls as $T$ grows. With $N$ points, $T = Nh/2$.

Setting the two equal,

$$
\frac{2\pi\alpha}{h} = e^{Nh/2}
\qquad\Longrightarrow\qquad
h \approx \frac{2\log(\pi\alpha N / \log N)}{N},
$$

and substituting back gives the achievable error

$$
E \sim \exp\!\left(\frac{-c N}{\log N}\right),
$$

**geometric in $N/\log N$ rather than in $N$.** That logarithmic penalty is the price of the
double exponential, and it is why the method is very good but not literally geometric.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq

print(f"{'level':>7}{'points N':>10}{'error':>14}{'-log(error)':>15}"
      f"{'N / log(N)':>14}{'ratio':>10}")
for level in range(1, 7):
    out = mq.tanh_sinh(lambda x: 1.0 / np.sqrt(np.asarray(x, dtype=float)),
                       0.0, 1.0, level)
    e = abs(out["value"] - 2.0)
    n = out["points"]
    if e <= 0:
        print(f"{level:>7}{n:>10}{e:>14.2e}{'exact':>15}{n / math.log(n):>14.2f}{'':>10}")
        continue
    print(f"{level:>7}{n:>10}{e:>14.2e}{-math.log(e):>15.3f}"
          f"{n / math.log(n):>14.2f}{-math.log(e) / (n / math.log(n)):>10.4f}")
```

**There are only two or three usable rows, and that is the finding.**

The rule reaches exactly zero error by level 3, at 97 points, so the sequence of errors that the
scaling law describes never gets a chance to develop. The measured ratios, $2.28$ and $2.75$ and
then $1.67$, are three numbers from a two point range, and fitting a law to them would be
inventing a result.

**The prediction is not wrong; it is unobservable in double precision for this integrand.**
Confirming $-\log E \propto N/\log N$ needs either extended precision, so the error has room to
fall through many decades, or an integrand hard enough that the rule takes ten levels rather than
three. Reporting the derivation and then reporting that the measurement cannot reach it is the
honest outcome, and it is a common one for methods this good.

### 5.3 Sparse grids and the curse

Smolyak's rule achieves $O(N^{-p}(\log N)^{(d-1)(p+1)})$ for integrands in a **mixed derivative**
class: those whose derivatives $\partial^{|\mathbf{k}|}f / \partial x_1^{k_1}\cdots\partial
x_d^{k_d}$ are bounded for all $k_i \le p$ **simultaneously**.

That is a much stronger requirement than ordinary smoothness. It asks for control of derivatives
of total order up to $pd$, not up to $p$.

**An integrand for which it should fail**: take $f(\mathbf{x}) = g(x_1 + x_2 + \cdots + x_d)$
with $g$ smooth, a **ridge function**. Every ordinary derivative of $f$ of total order $k$ is
bounded by $\|g^{(k)}\|$. But the mixed derivative
$\partial^d f/\partial x_1\cdots\partial x_d$ is $g^{(d)}$, so the mixed norm the theory needs
grows with $d$ for any $g$ that is not a polynomial, while the ordinary smoothness does not.

```python
import itertools
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import multiquad as mq


def smolyak(d, level, rule_1d=None):
    """Smolyak sparse grid points and weights on the unit cube."""
    rule = mq.gauss_rule_1d if rule_1d is None else rule_1d
    points, weights = [], []
    for total in range(max(level - d + 1, d), level + 1):
        coefficient = (-1) ** (level - total) * math.comb(d - 1, level - total)
        if coefficient == 0:
            continue
        for k in itertools.product(range(1, level + 1), repeat=d):
            if sum(k) != total:
                continue
            axes = [rule(2 ** ki - 1, 0.0, 1.0) for ki in k]
            grids = np.meshgrid(*[a[0] for a in axes], indexing="ij")
            wgrids = np.meshgrid(*[a[1] for a in axes], indexing="ij")
            p = np.stack([g.ravel() for g in grids], axis=1)
            w = np.ones(p.shape[0])
            for g in wgrids:
                w = w * g.ravel()
            points.append(p)
            weights.append(coefficient * w)
    return np.concatenate(points, axis=0), np.concatenate(weights)


print(f"{'d':>4}{'sparse points':>15}{'separable error':>18}{'ridge error':>15}"
      f"{'ridge / separable':>20}")
for d in (2, 4, 6):
    p, w = smolyak(d, d + 3)
    separable = abs(float(np.sum(w * np.prod(np.cos(p), axis=1))) - math.sin(1.0) ** d)
    ridge_exact = float(np.real(((np.exp(1j) - 1.0) / 1j) ** d))
    ridge = abs(float(np.sum(w * np.cos(np.sum(p, axis=1)))) - ridge_exact)
    print(f"{d:>4}{p.shape[0]:>15}{separable:>18.3e}{ridge:>15.3e}"
          f"{ridge / separable:>20.3f}")
```

**The predicted separation does not appear at these sizes.** The ridge error is $0.7$ of the
separable one at $d = 2$, $0.7$ at $d = 4$ and $2.2$ at $d = 6$: the same order of magnitude
throughout, with no sign of a widening gap.

That is a negative result and it is worth stating rather than quietly choosing a different
example. Two things explain it. The mixed derivative norm of $\cos(x_1+\cdots+x_d)$ grows only
like 1 in absolute value, since every derivative of cosine is bounded by 1, so this particular
ridge is far gentler than the theory's worst case. And the levels reachable by a direct
construction, $d + 3$, are nowhere near asymptotic.

**What survives is the analysis rather than the demonstration.** The mixed derivative condition
is a genuine restriction, it is much stronger than ordinary smoothness, and a function whose
difficulty is not aligned with the coordinate axes is exactly what it excludes. Showing it in a
measurement needs a $g$ with rapidly growing derivatives and a dimension high enough that the
sparse construction is in its asymptotic regime, which is beyond what fits in an exercise.

The practical lesson is the same one as everywhere else in this part. **A method encodes an
assumption, and the assumption here is that the integrand's difficulty factors across the
coordinates.** When it does, sparse grids defeat the curse; when it does not, they inherit it in
full, and the right response is to change coordinates rather than to change quadrature rules.
