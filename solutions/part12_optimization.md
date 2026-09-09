# Solutions: Part 12, Optimization

Solutions to every exercise in lessons 84 to 90, 147 in all: 21 for lesson 84, 21 for lesson 85, 21 for lesson 86, 21 for lesson 87, 21 for lesson 88, 21 for lesson 89, 21 for lesson 90.

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

## Lesson 84, Optimization Fundamentals

### 1.1 Why a minimum costs half the digits and a root does not

The difference is the first term of the Taylor expansion.

At a **root** of $g$, the expansion is $g(x) \approx g'(x^{*})(x - x^{*})$, which is linear. An
error $\delta$ in the computed value of $g$ therefore hides an interval of width
$\delta / \lvert g'\rvert$ around the root: the error in $x$ is proportional to the error in $g$.

At a **minimum** of $f$ the gradient vanishes, so the linear term is gone and
$f(x) \approx f(x^{*}) + \tfrac12 c\,(x - x^{*})^{2}$. Now an error $\delta$ in $f$ hides an
interval of width $\sqrt{2\delta/c}$: the error in $x$ is proportional to the **square root** of the
error in $f$.

In floating point $\delta$ is not free to be small. Values of $f$ carry a relative error of about
$\varepsilon/2$, so $\delta \approx (\varepsilon/2)\lvert f^{*}\rvert$ and

$$
\text{root: } \frac{\varepsilon\lvert g\rvert}{\lvert g'\rvert} \sim \varepsilon ,
\qquad
\text{minimum: } \sqrt{\frac{\varepsilon\lvert f^{*}\rvert}{c}} \sim \sqrt\varepsilon .
$$

That is $2\times10^{-16}$ against $1.5\times10^{-8}$, a factor of $10^{7}$, and it is a property of
the **problem**. No algorithm reading only values of $f$ escapes it, which is why the lesson
measures it with trisection and bisection rather than with anything clever: the crude methods make
the point without inviting the objection that a better one would do better.

### 1.2 What the second order test decides and where it is silent

At a stationary point, with $H$ the Hessian:

| $H$ | verdict |
|---|---|
| positive definite | strict local minimum |
| negative definite | strict local maximum |
| indefinite | saddle |
| **singular** | **nothing** |

The first three are decisions and the fourth is a refusal. If $H$ has a zero eigenvalue, the
quadratic model is flat in that direction and the behaviour of $f$ there is decided by the third,
fourth or higher derivatives, none of which the test looks at.

The refusal is not a weakness of any implementation. It is a statement about how much information
two derivatives contain, and 1.3 shows it cannot be improved.

There is a practical corollary. A numerical test has to decide what counts as a zero eigenvalue, so
it has a tolerance, and that tolerance must scale with the matrix. The lesson's `classify` uses
$\sqrt{\varepsilon}\lVert H\rVert$, so multiplying $H$ by $10^{12}$ or $10^{-12}$ does not change
the verdict. A fixed absolute cutoff would have called $10^{-12}I$ inconclusive, which is wrong: it
is a perfectly good minimum in a badly scaled variable.

### 1.3 Why $x^4$ and $-x^4$ cannot be separated

Both have every derivative up to second order equal at the origin:

$$
f(0) = 0, \qquad f'(0) = 0, \qquad f''(0) = 0 ,
$$

for $f = x^{4}$ and for $f = -x^{4}$. Their second order Taylor polynomials are **identical**, both
equal to the zero polynomial.

A second order test is by definition a function of $f(0)$, $\nabla f(0)$ and $H(0)$. Feeding it two
functions with identical values of all three must produce the same answer. One of them has a
minimum at the origin and the other a maximum. So any second order test is wrong on at least one of
them, and the only answer that is wrong on neither is "inconclusive".

That argument is complete and does not depend on how the test is written, which is why the correct
behaviour is to report the refusal rather than to guess. The lesson checks the premise numerically
as well: away from the origin one function rises and the other falls, at every sampled point.

### 1.4 What a clean convexity sample proves

It proves that **the non-convex region, if there is one, is small**. Nothing more.

Convexity is a statement about every point of the domain. A sample of $n$ points on a box of width
$W$ leaves gaps, and any behaviour that fits inside a gap is invisible. If the Hessian has a
negative eigenvalue only on a set of width $w$, a uniform sample lands in it an expected $nw/W$
times, so a clean sample says only that

$$
w \;\lesssim\; \frac{W}{n} \quad\text{with reasonable confidence,}
$$

which is a real statement and is not "the function is convex".

The lesson makes it sharper by pointing out that the count of hits is **Poisson**, so there is no
threshold sample size: a hit at $0.67$ expected hits and a miss at $1.67$ both occur in the same
table. And a sample that does find the dip understates it, reporting $14$ per cent of the true depth
in one row, because landing in the strip is not the same as landing at its bottom.

What does prove convexity is an argument: a sum of convex functions, a composition with an affine
map, a supremum of affine functions, or a Hessian shown to be positive semidefinite symbolically.

### 1.5 Why the condition number is not a difficulty measure

Because it describes the **shape of one bowl** and says nothing about how many bowls there are.

The condition number of $H$ at a minimum controls two things: how elongated the indistinguishable
set around that minimum is, and how fast a local method converges once it is inside that bowl. Both
are about the last stage of a search.

Everything before that stage is untouched by it. Rastrigin's Hessian at the global minimum is
$(2 + 4\pi^{2}A)I$, a multiple of the identity, so $\kappa = 1$ exactly: perfectly conditioned. It
is also the hardest problem in the lesson's set, because the box around that minimum is filled with
a lattice of others and a local method stops at whichever it reaches first.

The reverse also happens: Rosenbrock has $\kappa = 2508$ and one minimum, so any method that gets
into the valley will find the right answer, slowly.

So the two questions are independent. **Conditioning is about speed at the end; the number of
basins is about whether the end is the right one.** A paper that reports "our method handles ill
conditioning" and tests on a bowl has answered only the first.

### 2.1 Deriving the width of the flat region

Near a minimum in one dimension,

$$
f(x) = f^{*} + \tfrac12 c\,(x - x^{*})^{2} + O((x-x^{*})^{3}), \qquad c = f''(x^{*}) > 0 .
$$

Two points are indistinguishable in floating point when their computed values of $f$ round to the
same number. A double precision value near $f^{*}$ is stored with a spacing of one ulp, and rounding
is to **nearest**, so a true difference smaller than half an ulp is lost:

$$
\tfrac12 c\,d^{2} \;<\; \tfrac12 \varepsilon \lvert f^{*}\rvert .
$$

**Both halves are now on the table.** The left one comes from the quadratic, $\tfrac12 c d^2$; the
right one from rounding to nearest, half of $\varepsilon\lvert f^*\rvert$. They cancel, leaving

$$
\boxed{\;d = \sqrt{\frac{\varepsilon\,\lvert f^{*}\rvert}{c}}\;}
$$

with no stray factor of two, which is why the measurement matches the formula with a ratio of
$1.001$ rather than $\sqrt2$ or $1/\sqrt2$.

```python
import math

import numpy as np
from nalib import optimize as op

print(f"{'variables':>11}{'curvature':>11}{'value':>9}{'formula':>13}"
      f"{'measured':>13}{'ratio':>9}")
for variables in (1, 3, 8):
    for curvature in (0.25, 1.0, 40.0):
        for value in (1.0, 100.0):
            def f(x, curvature=curvature, value=value):
                v = np.asarray(x, dtype=float).ravel()
                return float(value + 0.5 * curvature * float(v @ v))

            predicted = op.attainable_accuracy(curvature, value)
            measured = op.flat_region_width(f, np.zeros(variables))
            print(f"{variables:>11}{curvature:>11g}{value:>9g}{predicted:>13.4e}"
                  f"{measured:>13.4e}{measured / predicted:>9.4f}")

print()
print("the ratio away from 1 is the binade, not an error in the formula:")
print(f"{'value':>9}{'eps times value':>18}{'the real quantum':>19}"
      f"{'their ratio':>13}{'its square root':>17}")
for value in (1.0, 100.0):
    quantum = float(np.nextafter(value, math.inf) - value)
    share = quantum / (np.finfo(float).eps * value)
    print(f"{value:>9g}{np.finfo(float).eps * value:>18.4e}{quantum:>19.4e}"
          f"{share:>13.4f}{math.sqrt(share):>17.4f}")
```

**At $f^{*} = 1$ the ratio is $1.001$ at every curvature and every number of variables**, which is
the check that neither factor of a half was dropped or double counted. The width does not depend on
the dimension because the function is isotropic; exercise 2.2 works out the anisotropic case, where
the set becomes an ellipsoid with aspect ratio $\sqrt\kappa$.

**At $f^{*} = 100$ the ratio is $0.8008$, and that is the formula being slightly too crude rather
than the measurement being wrong.** The derivation uses $\varepsilon\lvert f^{*}
vert$ as the
smallest detectable change in $f$, but the real quantum is the gap to the next representable number,
which is $2^{-52}\cdot 2^{e}$ where $2^{e}$ is the bottom of $f^{*}$'s binade. At $f^{*} = 1$ those
coincide, because $1$ is exactly the bottom of its binade. At $f^{*} = 100$, whose binade starts at
$64$, the quantum is $64/100 = 0.64$ of $\varepsilon\lvert f^{*}
vert$, and the width scales as
the square root:

$$
\sqrt{0.64} = 0.8 ,
$$

which the last table confirms exactly, and $0.8 \times 1.001 = 0.8008$ is the measured ratio.

So the formula is right up to a factor between $\sqrt{1/2}$ and $1$, depending on where $f^{*}$ sits
inside its binade, and that factor is not worth carrying: the point of the barrier is its order of
magnitude and its $\sqrt\varepsilon$ scaling, both of which are unaffected.

### 2.2 The aspect ratio of the indistinguishable set

In $n$ dimensions, with $H = \sum_k \lambda_k v_k v_k^{T}$ the eigendecomposition,

$$
f(x) - f^{*} = \tfrac12 (x - x^{*})^{T} H (x - x^{*})
             = \tfrac12 \sum_k \lambda_k\, \big(v_k^{T}(x - x^{*})\big)^{2} .
$$

Move along one eigenvector, $x = x^{*} + t\,v_k$. Then the sum collapses to a single term,
$f - f^{*} = \tfrac12\lambda_k t^{2}$, which is the one dimensional problem of 2.1 with curvature
$\lambda_k$. So the half width along $v_k$ is

$$
d_k = \sqrt{\frac{\varepsilon \lvert f^{*}\rvert}{\lambda_k}} .
$$

The set of indistinguishable points is therefore the **ellipsoid** with axes along the eigenvectors
and half lengths $d_k \propto \lambda_k^{-1/2}$, and its aspect ratio is

$$
\frac{d_{\max}}{d_{\min}} = \sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}} = \sqrt{\kappa(H)} ,
$$

**the square root of the condition number**. A problem with $\kappa = 10^{4}$ is elongated by $100$,
not by $10^{4}$, which is the number to compare a convergence plot against.

Two consequences follow. The measured half widths must reproduce the eigenvalues, and a **random**
direction must give something between the extremes, dominated by the narrow ones since a random
vector has a component along every axis.

```python
import numpy as np
from nalib import optimize as op

rng = np.random.default_rng(7)
for variables in (2, 3, 5):
    problem = op.quadratic(condition=1e4, dimension=variables)
    centre = problem["minimizers"][0]
    values, vectors = np.linalg.eigh(problem["hessian"](centre))
    value = problem["f"](centre)
    along = [op.flat_region_width(problem["f"], centre, vectors[:, k])
             for k in range(variables)]
    predicted = [op.attainable_accuracy(float(v), value) for v in values]
    scattered = [op.flat_region_width(problem["f"], centre, rng.normal(size=variables))
                 for _ in range(5)]
    print(f"{variables} variables")
    print("  along the eigenvectors " + " ".join(f"{w:.3e}" for w in along))
    print("  the formula predicts   " + " ".join(f"{w:.3e}" for w in predicted))
    print(f"  random directions run from {min(scattered):.3e} to {max(scattered):.3e}, "
          f"inside [{min(along):.3e}, {max(along):.3e}]")
```

The measured widths reproduce the predictions to four digits at every size, and every random
direction lands inside the range the eigenvectors set.

### 2.3 The optimal step for a second difference

For a first derivative, the central difference has truncation error $\tfrac16 h^{2}f'''$ and
rounding error about $\varepsilon\lvert f\rvert/h$, since the numerator is a difference of two
values each carrying $\varepsilon\lvert f\rvert$ and it is divided by $2h$. The total is smallest
where the two are equal:

$$
h^{2} \sim \frac{\varepsilon}{h} \;\Rightarrow\; h \sim \varepsilon^{1/3},
\qquad \text{error} \sim \varepsilon^{2/3} \approx 3.7\times10^{-11} .
$$

For a second derivative, $(f(x+h) - 2f(x) + f(x-h))/h^{2}$ has truncation error
$\tfrac1{12}h^{2}f^{(4)}$ and rounding error about $\varepsilon\lvert f\rvert/h^{2}$, because the
same numerator is now divided by $h^{2}$. Equating,

$$
h^{2} \sim \frac{\varepsilon}{h^{2}} \;\Rightarrow\; h \sim \varepsilon^{1/4},
\qquad \text{error} \sim \varepsilon^{1/2} \approx 1.5\times10^{-8} .
$$

So the Hessian by differences is about $\varepsilon^{1/2}/\varepsilon^{2/3} = \varepsilon^{-1/6}
\approx 400$ times less accurate than the gradient, before counting that it also costs $4n^{2}$
evaluations against $2n$.

```python
import math

import numpy as np

value = lambda t: math.exp(t) * math.sin(t)
slope = lambda t: math.exp(t) * (math.sin(t) + math.cos(t))
where = 1.5
eps = float(np.finfo(float).eps)
best, best_step = math.inf, None
print(f"{'step':>10}{'relative error':>18}")
for power in range(1, 17):
    step = 10.0 ** (-power)
    got = (value(where + step) - value(where - step)) / (2.0 * step)
    error = abs(got - slope(where)) / abs(slope(where))
    if error < best:
        best, best_step = error, step
    if power % 2 == 1:
        print(f"{step:>10.0e}{error:>18.3e}")
print(f"\nbest step {best_step:.0e}, error {best:.3e}")
print(f"predicted step eps**(1/3) = {eps ** (1 / 3):.3e}, "
      f"predicted error eps**(2/3) = {eps ** (2 / 3):.3e}")
```

The measured best step is $10^{-5}$ against a predicted $6.1\times10^{-6}$, and the error there is
$2.5\times10^{-11}$ against a predicted $3.7\times10^{-11}$. Both within a factor of two, which is
all an order of magnitude argument claims.

The shape of the curve is the useful part: too large a step and truncation dominates, falling like
$h^{2}$; too small and rounding dominates, rising like $1/h$; and by $h = 10^{-15}$ the relative
error is $0.11$, so the "obvious" choice of the smallest possible step is catastrophic.

### 2.4 A stationary point of a convex function is global

Let $f$ be convex and differentiable with $\nabla f(x^{*}) = 0$. Convexity gives the first order
condition: for every $y$,

$$
f(y) \;\ge\; f(x^{*}) + \nabla f(x^{*})^{T}(y - x^{*}) .
$$

That inequality is the statement that a convex function lies above its tangent plane, and it holds
at every pair of points. Substituting $\nabla f(x^{*}) = 0$ leaves

$$
f(y) \ge f(x^{*}) \quad \text{for every } y ,
$$

which is exactly the definition of a global minimum. Nothing was assumed about $y$ being nearby, and
that is the whole content: **for a convex function, local information at a stationary point is
global information.**

Two remarks. Convexity is what makes it work, not differentiability: the same argument runs with the
subgradient inequality, which is lesson 90's setting. And the converse fails, since a non-convex
function can have a stationary point that happens to be global, so the theorem gives a certificate
and not a test.

The practical value is that it converts an impossible question into a possible one. Checking that a
point is a global minimum is in general hopeless; checking $\nabla f = 0$ costs one gradient, and
for a convex problem the two are the same check.

### 2.5 The Hessian of Rosenbrock is tridiagonal

$$
f(x) = \sum_{i=1}^{n-1}\Big[100\,(x_{i+1} - x_i^{2})^{2} + (1 - x_i)^{2}\Big] .
$$

Each term of the sum involves **only** $x_i$ and $x_{i+1}$. Differentiating twice, a second
derivative $\partial^{2}f/\partial x_j \partial x_k$ can be nonzero only if some single term
contains both $x_j$ and $x_k$, which needs $\lvert j - k\rvert \le 1$. That is tridiagonality, and it
follows from the summation pattern alone, before any algebra.

The entries themselves come from differentiating term $i$:

$$
\frac{\partial^{2}}{\partial x_i^{2}}: \; -400(x_{i+1} - 3x_i^{2}) + 2 ,
\qquad
\frac{\partial^{2}}{\partial x_i \partial x_{i+1}}: \; -400 x_i ,
\qquad
\frac{\partial^{2}}{\partial x_{i+1}^{2}}: \; 200 ,
$$

and the diagonal entries accumulate over the two terms that touch each interior variable.

```python
import numpy as np
from nalib import optimize as op

rng = np.random.default_rng(42)
print(f"{'variables':>11}{'bandwidth':>12}{'nonzeros':>10}{'dense entries':>15}"
      f"{'gap to differences':>21}")
for variables in (2, 5, 12, 30):
    problem = op.rosenbrock(variables)
    where = rng.normal(size=variables)
    exact = np.asarray(problem["hessian"](where), dtype=float)
    numeric = op.hessian(problem["f"], where)
    rows, columns = np.nonzero(np.abs(exact) > 1e-14)
    scale = float(np.max(np.abs(exact)))
    print(f"{variables:>11}{int(np.max(np.abs(rows - columns))):>12}"
          f"{rows.size:>10}{variables * variables:>15}"
          f"{float(np.max(np.abs(exact - numeric))) / scale:>21.3e}")
```

The bandwidth is $1$ at every size, the nonzero count is $3n - 2$ against $n^{2}$ dense entries, and
the closed form agrees with central differences to the accuracy 2.3 predicted for them.

The sparsity is not decoration. It is why Newton's method on Rosenbrock costs $O(n)$ per step
rather than $O(n^{3})$, and it is the one dimensional case of the pattern lesson 82 found for the
five point Laplacian: a local coupling in the objective gives a banded Hessian.

### 3.1 Every stationary point of Himmelblau

Himmelblau's function has nine stationary points, and finding them all needs a search rather than a
solve: Newton's method from a single start finds one, and which one depends on the start. Running it
from many random starts and collecting the distinct answers finds them all.

```python
import numpy as np
from nalib import optimize as op

problem = op.himmelblau()
rng = np.random.default_rng(42)
found = []
for _ in range(600):
    where = rng.uniform(-6.0, 6.0, size=problem["dimension"])
    for _ in range(200):
        slope = np.asarray(problem["gradient"](where), dtype=float)
        if float(np.linalg.norm(slope)) < 1e-12:
            break
        try:
            where = where + np.linalg.solve(problem["hessian"](where), -slope)
        except np.linalg.LinAlgError:
            break
    if (float(np.linalg.norm(problem["gradient"](where))) < 1e-9
            and float(np.max(np.abs(where))) < 8.0
            and not any(float(np.linalg.norm(where - seen)) < 1e-5 for seen in found)):
        found.append(where.copy())

found.sort(key=lambda p: (round(float(p[0]), 4), round(float(p[1]), 4)))
tally = {}
print(f"{'x':>12}{'y':>12}{'f':>14}{'verdict':>14}")
for point in found:
    verdict = op.classify(problem["hessian"](point))
    tally[verdict] = tally.get(verdict, 0) + 1
    print(f"{point[0]:>12.6f}{point[1]:>12.6f}{problem['f'](point):>14.6f}"
          f"{verdict:>14}")
print(f"\n{len(found)} distinct stationary points: {tally}")
```

Nine points: **four minima, four saddles and one maximum.** All four minima have $f = 0$ exactly, and
they are the four the lesson's `himmelblau` reports. The single maximum sits at $(-0.271, -0.923)$
with $f = 181.6$, ringed by the four saddles.

Two things worth noticing.

**The count is not an accident.** For a smooth function on the plane whose level sets close up, the
Morse relation `minima - saddles + maxima = 1` holds, and $4 - 4 + 1 = 1$. Finding eight points, or
ten, would have meant the search missed one or double counted.

**Every one of the nine has a zero gradient.** The first order test passes on all nine and separates
none of them, which is 1.2 seen at full size: the second order test is what does the work here, and
it is decisive at all nine because none of the Hessians is singular.

### 3.2 The complex step gradient

The complex step derivative comes from expanding $f$ at $x + ih$ for real analytic $f$:

$$
f(x + ih) = f(x) + ih f'(x) - \tfrac{h^{2}}{2}f''(x) - i\tfrac{h^{3}}{6}f'''(x) + \cdots
$$

Taking the imaginary part and dividing by $h$,

$$
\frac{\operatorname{Im} f(x + ih)}{h} = f'(x) - \tfrac{h^{2}}{6}f'''(x) + \cdots
$$

which is second order accurate and, crucially, involves **no subtraction of nearly equal numbers**.
So $h$ can be made as small as the exponent range allows and the rounding term never appears.

```python
import math

import numpy as np

value = lambda t: np.exp(t) * np.sin(t)
where = 1.5
exact = math.exp(where) * (math.sin(where) + math.cos(where))
print(f"{'step':>10}{'central difference':>21}{'complex step':>16}")
for step in (1e-2, 1e-5, 1e-8, 1e-11, 1e-14, 1e-200):
    if step > 1e-100:
        central = (value(where + step) - value(where - step)) / (2.0 * step)
        left = abs(central - exact) / abs(exact)
    else:
        left = float("nan")
    complex_form = float(np.imag(value(where + 1j * step)) / step)
    right = abs(complex_form - exact) / abs(exact)
    shown = "no step this small" if left != left else f"{left:.3e}"
    print(f"{step:>10.0e}{shown:>21}{right:>16.3e}")
```

The complex step reaches **exactly zero** relative error at $h = 10^{-8}$ and stays there down to
$h = 10^{-200}$, where a central difference is not computable at all. The central difference is best
at $h = 10^{-5}$ with $2.5\times10^{-11}$ and degrades to $7.5\times10^{-3}$ by $h = 10^{-14}$.

**Now the question the exercise asks: how much of the barrier does this remove?**

The answer is all of it, and the reason is worth being precise about. Lesson 84's barrier is on
locating a minimum **from values of $f$**. An exact derivative changes the problem: finding
$f' = 0$ is a root problem, and 1.1 said root problems cost $\varepsilon$ rather than
$\sqrt\varepsilon$.

```python
import math

import numpy as np

shifted = lambda t: np.exp(t) * np.sin(t) + 3.0
derivative = lambda t: float(np.imag(shifted(t + 1j * 1e-200)) / 1e-200)
answer = 3.0 * math.pi / 4.0 + math.pi          # where tan(t) = -1, the minimum

lo, hi = 5.0, 6.0
for _ in range(200):
    mid = 0.5 * (lo + hi)
    if derivative(lo) * derivative(mid) <= 0.0:
        hi = mid
    else:
        lo = mid
by_derivative = 0.5 * (lo + hi)


def trisect(f, a, b, rounds=300):
    for _ in range(rounds):
        left, right = a + (b - a) / 3.0, b - (b - a) / 3.0
        if f(left) < f(right):
            b = right
        else:
            a = left
    return 0.5 * (a + b)


by_values = trisect(lambda t: float(np.real(shifted(t))), 5.0, 6.0)
eps = float(np.finfo(float).eps)
print(f"bisection on the complex step derivative: error "
      f"{abs(by_derivative - answer):.3e}  ({abs(by_derivative - answer) / eps:.2f} eps)")
print(f"trisection on values of f:                error "
      f"{abs(by_values - answer):.3e}  "
      f"({abs(by_values - answer) / math.sqrt(eps):.3f} sqrt(eps))")
```

The derivative route lands $4$ machine epsilons from the answer and the value route $0.89$ square
roots of one: a factor of $1.5\times10^{7}$.

**So the barrier is a property of the information, not of the function.** Given only values, half
the digits are gone. Given an exact derivative, they are not. That is the strongest argument for
automatic differentiation there is, and it is the subject of 3.3.

### 3.3 Automatic differentiation by dual numbers

A dual number is $a + b\epsilon$ with $\epsilon^{2} = 0$. Arithmetic on the pair carries the value
and its derivative together:

$$
(a + b\epsilon)(c + d\epsilon) = ac + (ad + bc)\epsilon ,
$$

which is the product rule. Evaluating $f$ on $x + \epsilon$ returns $f(x) + f'(x)\epsilon$, so the
derivative comes out of an ordinary function evaluation with no step size, no subtraction and no
truncation.

```python
import numpy as np
from nalib import optimize as op


class Dual:
    """A value and its derivative, carried together through ordinary arithmetic."""

    __slots__ = ("value", "slope")

    def __init__(self, value, slope=0.0):
        self.value, self.slope = float(value), float(slope)

    def _lift(self, other):
        return other if isinstance(other, Dual) else Dual(other)

    def __add__(self, other):
        other = self._lift(other)
        return Dual(self.value + other.value, self.slope + other.slope)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._lift(other)
        return Dual(self.value - other.value, self.slope - other.slope)

    def __rsub__(self, other):
        return self._lift(other) - self

    def __mul__(self, other):
        other = self._lift(other)
        return Dual(self.value * other.value,
                    self.slope * other.value + self.value * other.slope)

    __rmul__ = __mul__

    def __pow__(self, power):
        return Dual(self.value ** power,
                    power * self.value ** (power - 1) * self.slope)


def rosenbrock_dual(values):
    total = Dual(0.0)
    for i in range(len(values) - 1):
        total = total + 100.0 * (values[i + 1] - values[i] ** 2) ** 2 \
            + (1.0 - values[i]) ** 2
    return total


rng = np.random.default_rng(42)
print(f"{'variables':>11}{'dual gap':>13}{'difference gap':>18}"
      f"{'dual passes':>14}{'difference calls':>19}")
for variables in (2, 5, 10, 25):
    where = rng.normal(size=variables)
    problem = op.rosenbrock(variables)
    exact = np.asarray(problem["gradient"](where), dtype=float)
    by_dual = np.zeros(variables)
    for k in range(variables):
        seeded = [Dual(float(where[j]), 1.0 if j == k else 0.0)
                  for j in range(variables)]
        by_dual[k] = rosenbrock_dual(seeded).slope
    by_difference = op.gradient(problem["f"], where)
    scale = max(float(np.max(np.abs(exact))), 1.0)
    print(f"{variables:>11}{float(np.max(np.abs(by_dual - exact))) / scale:>13.3e}"
          f"{float(np.max(np.abs(by_difference - exact))) / scale:>18.3e}"
          f"{variables:>14}{2 * variables:>19}")
```

Forward mode dual numbers give the gradient to $2\times10^{-16}$, which is machine precision, where
central differences give $10^{-10}$. And they do it in $n$ passes against the differences' $2n$
evaluations, so they are **both more accurate and cheaper** in this comparison.

Two honest qualifications.

**Forward mode costs $n$ passes.** For $n$ in the thousands that is worse than it sounds, and the
method that fixes it is reverse mode, which computes the whole gradient in one backward pass at the
cost of storing the computation. That is lesson 97's subject and it is why deep learning is
possible.

**The function has to be written in terms the dual number understands.** Every operation used above
had to be given a rule. A function calling a compiled library, or reading a table, or branching on
its input, cannot be differentiated this way without more work, which is why finite differences
survive as the fallback that always applies.

### 3.4 Positive definiteness by Cholesky

The Cholesky factorization $A = LL^{T}$ exists **if and only if** $A$ is symmetric positive
definite, so attempting it is a test. It costs $n^{3}/3$ operations against the
$\approx 9n^{3}$ of a symmetric eigendecomposition, and it stops at the first negative pivot rather
than computing everything.

```python
import time

import numpy as np

rng = np.random.default_rng(42)
print(f"{'size':>6}{'cholesky s':>13}{'eigenvalues s':>16}{'speedup':>10}{'agree':>8}")
for size in (50, 200, 500):
    matrix = rng.normal(size=(size, size))
    matrix = matrix @ matrix.T + size * np.eye(size)
    start = time.perf_counter()
    for _ in range(20):
        try:
            np.linalg.cholesky(matrix)
            by_cholesky = True
        except np.linalg.LinAlgError:
            by_cholesky = False
    quick = (time.perf_counter() - start) / 20
    start = time.perf_counter()
    for _ in range(20):
        by_eigenvalues = bool(np.min(np.linalg.eigvalsh(matrix)) > 0.0)
    slow = (time.perf_counter() - start) / 20
    print(f"{size:>6}{quick:>13.6f}{slow:>16.6f}{slow / quick:>10.1f}"
          f"{str(by_cholesky == by_eigenvalues):>8}")

print()
print("and on matrices that are not positive definite:")
for shift in (0.0, -1e-9, -1.0):
    size = 40
    base = rng.normal(size=(size, size))
    base = base @ base.T
    values = np.linalg.eigvalsh(base)
    test = base + (shift - float(values[0])) * np.eye(size)
    try:
        np.linalg.cholesky(test)
        verdict = "positive definite"
    except np.linalg.LinAlgError:
        verdict = "not"
    print(f"  smallest eigenvalue {float(np.linalg.eigvalsh(test)[0]):>12.3e}: "
          f"cholesky says {verdict}")
```

Cholesky is $3$ to $12$ times faster and agrees with the eigenvalue test on every matrix. The
speedup grows with the size, as the $n^{3}/3$ against $9n^{3}$ count says it should.

The second table is the more useful half. Cholesky is a **decision** and gives no measure of how
close to indefinite a matrix is, so when the smallest eigenvalue is $-10^{-9}$ it says "not" with no
more information than when it is $-1$. When the answer needed is a shift, as in lesson 86's modified
Newton, that is exactly enough: shift and try again. When the answer needed is a diagnosis, the
eigenvalues are worth their cost.

### 3.5 The flat region's axes in $n$ dimensions

2.2 derived that the indistinguishable set is an ellipsoid with axes along the eigenvectors of $H$.
That is a claim with three testable parts: the half widths along the eigenvectors match
$\sqrt{\varepsilon\lvert f^{*}\rvert/\lambda_k}$, they are ordered opposite to the eigenvalues, and
no other direction is wider than the widest eigenvector.

```python
import numpy as np
from nalib import optimize as op

rng = np.random.default_rng(11)
for variables in (2, 4, 6):
    problem = op.quadratic(condition=1e6, dimension=variables)
    centre = problem["minimizers"][0]
    values, vectors = np.linalg.eigh(problem["hessian"](centre))
    value = problem["f"](centre)
    along = np.array([op.flat_region_width(problem["f"], centre, vectors[:, k])
                      for k in range(variables)])
    predicted = np.array([op.attainable_accuracy(float(v), value) for v in values])
    scattered = np.array([op.flat_region_width(problem["f"], centre,
                                               rng.normal(size=variables))
                          for _ in range(20)])
    print(f"{variables} variables, condition {float(values[-1] / values[0]):.3e}")
    print(f"  worst gap to the prediction   "
          f"{float(np.max(np.abs(along / predicted - 1.0))):.3e}")
    print(f"  widths decrease as eigenvalues rise: "
          f"{bool(np.all(np.diff(along) < 0))}")
    print(f"  no random direction beats the widest axis: "
          f"{bool(np.max(scattered) <= np.max(along) * (1.0 + 1e-9))}")
    print(f"  aspect ratio {float(np.max(along) / np.min(along)):.2f} against "
          f"sqrt(kappa) = {float(np.sqrt(values[-1] / values[0])):.2f}")
```

All three hold at every size: the widths match the formula to within $10^{-3}$, they decrease as the
eigenvalues rise, and no random direction is wider than the flattest eigenvector. The aspect ratio
comes out at $1000$ against $\sqrt{10^{6}} = 1000$.

The last check is the one that establishes the shape rather than the sizes. A set with the right
widths along $n$ chosen directions could still be any shape between them; a set no wider than its
widest axis in **any** direction is an ellipsoid inscribed in that bound.

### 4.1 The accuracy against the value at the minimum, over ten decades

The formula says $d \propto \sqrt{\lvert f^{*}\rvert}$ with everything else fixed. Shifting the same
parabola up and down changes $f^{*}$ and nothing else, so the exponent can be fitted cleanly.

```python
import math

import numpy as np
from nalib import optimize as op

root_two = math.sqrt(2.0)


def trisect(f, a, b, rounds=300):
    for _ in range(rounds):
        left, right = a + (b - a) / 3.0, b - (b - a) / 3.0
        if f(left) < f(right):
            b = right
        else:
            a = left
    return 0.5 * (a + b)


values, errors = [], []
print(f"{'value at the minimum':>22}{'error':>13}{'formula':>13}{'ratio':>9}")
for power in range(-8, 5):
    shift = 10.0 ** power

    def f(t, shift=shift):
        return (t - root_two) ** 2 + shift

    error = abs(trisect(f, 0.0, 3.0) - root_two)
    predicted = op.attainable_accuracy(2.0, shift)
    values.append(shift)
    errors.append(error)
    if power % 3 == 0:
        print(f"{shift:>22.0e}{error:>13.3e}{predicted:>13.3e}"
              f"{error / predicted:>9.4f}")
power = float(np.polyfit(np.log(values), np.log(errors), 1)[0])
print(f"\nfitted power over {len(values)} decades: {power:.4f}   "
      f"(the prediction is 0.5)")
```

The fitted power is $0.4987$ over thirteen decades of $f^{*}$, and the ratio to the formula stays
near $1$ throughout. Ten decades is enough to distinguish $0.5$ from $0.45$ or $0.55$ beyond any
doubt, which two or three decades would not be.

The row at $f^{*} = 10^{-8}$ is worth pausing on: the error there is $10^{-11}$, better than
$\sqrt\varepsilon$. **The barrier is not $\sqrt\varepsilon$, it is
$\sqrt{\varepsilon\lvert f^{*}\rvert/c}$**, and quoting the first is only right when $f^{*}$ and $c$
are both of order one.

### 4.2 The finite difference error against the step

The prediction is a V shape in log-log: truncation falling like $h^{2}$ on the right, rounding
rising like $1/h$ on the left, and a minimum at $h \sim \varepsilon^{1/3}$ of size
$\varepsilon^{2/3}$.

```python
import math

import numpy as np

value = lambda t: math.exp(t) * math.sin(t)
slope = lambda t: math.exp(t) * (math.sin(t) + math.cos(t))
where = 1.5
eps = float(np.finfo(float).eps)
steps, errors = [], []
for power in np.arange(0.5, 16.5, 0.5):
    step = 10.0 ** (-power)
    got = (value(where + step) - value(where - step)) / (2.0 * step)
    steps.append(step)
    errors.append(max(abs(got - slope(where)) / abs(slope(where)), 1e-18))
steps, errors = np.asarray(steps), np.asarray(errors)
best = int(np.argmin(errors))

big = steps > 1e-3
small = (steps < 1e-9) & (errors > 1e-12)
print(f"best step {steps[best]:.3e}, error there {errors[best]:.3e}")
print(f"predicted step {eps ** (1 / 3):.3e}, predicted error {eps ** (2 / 3):.3e}")
print(f"slope on the truncation side: "
      f"{float(np.polyfit(np.log(steps[big]), np.log(errors[big]), 1)[0]):.4f}"
      f"   (the prediction is 2)")
print(f"slope on the rounding side:   "
      f"{float(np.polyfit(np.log(steps[small]), np.log(errors[small]), 1)[0]):.4f}"
      f"   (the prediction is -1)")
```

Both slopes come out where the argument says, the minimum sits within a factor of two of
$\varepsilon^{1/3}$, and the error there within a factor of two of $\varepsilon^{2/3}$.

The V is the whole reason a step size cannot be chosen by "smaller is better". Going two decades
below the optimum costs four orders of magnitude in accuracy, and going four decades below costs
nine.

### 4.3 How often a sample finds a narrow strip

The lesson predicted the hit probability as $1 - e^{-\lambda}$ with $\lambda = nw/W$ the expected
number of samples landing in a strip of width $2w$ on a box of width $W = 6$. Repeating the
experiment many times tests that.

```python
import math

import numpy as np

trials = 400
print(f"{'half width':>12}{'samples':>9}{'measured':>11}{'naive Poisson':>16}"
      f"{'corrected':>12}")
for half_width, count in ((0.01, 500), (0.01, 2000), (0.003, 2000)):
    amplitude = 2.0 * half_width ** 2
    rng = np.random.default_rng(42)
    hits = 0
    for _ in range(trials):
        points = rng.uniform(-3.0, 3.0, size=count)
        scaled = points / half_width
        curvature = (2.0 + amplitude * (2.0 / half_width ** 2)
                     * (2.0 * scaled ** 2 - 1.0) * np.exp(-scaled ** 2))
        hits += int(np.min(curvature) < 0.0)
    # the region where the curvature is actually negative, found rather than assumed
    grid = np.linspace(-4.0, 4.0, 200001)
    negative = (2.0 + amplitude * (2.0 / half_width ** 2)
                * (2.0 * grid ** 2 - 1.0) * np.exp(-grid ** 2)) < 0.0
    true_width = 2.0 * float(np.max(np.abs(grid[negative]))) * half_width
    naive = 1.0 - math.exp(-count * 2.0 * half_width / 6.0)
    corrected = 1.0 - math.exp(-count * true_width / 6.0)
    print(f"{half_width:>12g}{count:>9}{hits / trials:>11.3f}{naive:>16.3f}"
          f"{corrected:>12.3f}")
```

The measured rates are $0.540$, $0.930$ and $0.625$ against the lesson's naive prediction of
$0.811$, $0.999$ and $0.865$. **The naive prediction is consistently too high**, and finding out why
is the exercise.

The mistake is in the width. The strip where the bump lives is $2w$ across, but the curvature is
negative only where $2x^{2}/w^{2} < 1 - $ something, which works out to $\lvert x\rvert < 0.44w$: a
region of width $0.88w$, not $2w$. Landing in the bump is not the same as landing where the function
is non-convex, and the ratio is $2.3$.

Computing the true width from the curvature itself, rather than assuming it is the bump width, gives
predictions of $0.52$, $0.95$ and $0.58$ against the measured $0.54$, $0.93$ and $0.63$: agreement
to a few per cent, which is what four hundred repetitions can resolve.

**The Poisson model was right and the parameter fed into it was wrong**, which is a more useful
correction than abandoning the model would have been. The conclusion of the lesson is unchanged and
slightly strengthened: the strip is even harder to hit than it looked.

### 5.1 When the square root does not apply

The barrier is $d = \sqrt{\varepsilon\lvert f^{*}\rvert / c}$, and it has three ways of not binding.

**When $f^{*} = 0$.** The formula gives $d = 0$, and the lesson measured $2.2\times10^{-16}$: full
precision. The reason is that the relative error in a computed value of $f$ is relative to $f$
itself, and near a minimum of value zero the computed values are themselves tiny, so the noise
shrinks with them. **Least squares is exactly this case**: $f = \lVert r\rVert^{2}$ with $r \to 0$
when the model fits.

That is not a curiosity. It is why Part 5 could report the accuracies it did, why Gauss-Newton
behaves better than a general purpose minimizer on the same data, and why a solver that is told its
objective is a sum of squares can do better than one that is not. The information is in the
**structure** and a general purpose interface throws it away.

**When derivatives are available.** 3.2 showed that bisecting on an exactly computed derivative
lands $4\varepsilon$ from the answer against $0.89\sqrt\varepsilon$ from values, a factor of
$1.5\times10^{7}$. The barrier is on the information, not on the function, and automatic
differentiation supplies the missing information at a cost comparable to evaluating $f$.

**When the answer wanted is $f^{*}$ and not $x^{*}$.** This one is often missed. Inside the flat
region $f$ is constant to machine precision **by definition**, so the value at any point of it is as
good as the value at the minimum. If the quantity of interest is the minimum value, there is no
barrier at all: it is reached to full precision, and it is the location that is uncertain. Many
applications want the value.

**What a solver has to know to exploit any of this.** Each case needs information the plain
interface `minimize(f, x0)` cannot carry:

| Case | What must be known | Interface that carries it |
|---|---|---|
| $f^{*} = 0$ | that $f$ is a sum of squares | `least_squares(residual, x0)` |
| exact derivatives | how to differentiate $f$ | `minimize(f, x0, jac=...)` or a differentiable language |
| only the value matters | what the caller will do with the answer | the stopping test, chosen by the caller |

That table is the practical content of the whole lesson. **The barrier is a consequence of an
interface, not a law of arithmetic**, and every one of the three escapes is a matter of telling the
solver something it would otherwise have to discover.

### 5.2 Degenerate minima

For $f = x^{4}$ the Hessian at the minimum is zero, so the derivation of the barrier has $c = 0$ and
$d = \sqrt{\varepsilon\lvert f^{*}\rvert / 0}$ is undefined. The expansion has to be redone.

Near the minimum, $f(x) = f^{*} + a\,(x - x^{*})^{4}$ with $a > 0$. Two points are indistinguishable
when

$$
a\,d^{4} < \tfrac12 \varepsilon \lvert f^{*}\rvert
\qquad\Longrightarrow\qquad
d = \left(\frac{\varepsilon \lvert f^{*}\rvert}{2a}\right)^{1/4} .
$$

**A quarter power instead of a half.** With $a = 1$ and $f^{*} = 1$ that is
$(1.1\times10^{-16})^{1/4} = 1.0\times10^{-4}$: four significant digits out of sixteen, where the
quadratic case gave eight.

The general statement follows the same way. If the first nonvanishing derivative at the minimum is
of order $p$, which must be even for a minimum, then

$$
d \sim \left(\varepsilon\lvert f^{*}\rvert\right)^{1/p} ,
$$

so a quadratic minimum keeps half the digits, a quartic minimum a quarter, and a minimum flat to
order $16$ keeps **one**.

```python
import math

import numpy as np
from nalib import optimize as op


def trisect(f, a, b, rounds=400):
    for _ in range(rounds):
        left, right = a + (b - a) / 3.0, b - (b - a) / 3.0
        if f(left) < f(right):
            b = right
        else:
            a = left
    return 0.5 * (a + b)


eps = float(np.finfo(float).eps)
answer = math.sqrt(2.0)
print(f"{'power p':>9}{'measured error':>17}{'(eps |f*|)**(1/p)':>21}{'ratio':>9}")
for power in (2, 4, 6, 8):
    def f(t, power=power):
        return (t - answer) ** power + 1.0

    error = abs(trisect(f, 0.0, 3.0) - answer)
    predicted = (eps / 2.0) ** (1.0 / power)
    print(f"{power:>9}{error:>17.3e}{predicted:>21.3e}{error / predicted:>9.4f}")

print()
print("and the Hessian test at a quartic minimum:")
print(f"  H there is {0.0}, so classify says "
      f"{op.classify(np.zeros((1, 1)))!r}, which 1.3 says is the only honest answer")
```

The measured errors follow the quarter, sixth and eighth roots, and the ratio to the prediction is
**1.0000 at every power**, not merely the right order of magnitude. The formula is exact because the
argument is exact: the only approximation in it is dropping the higher terms of the expansion, and
for these functions there are none.

Two consequences worth keeping.

**Degeneracy is expensive and invisible.** The run reports success, the objective is at its minimum
to machine precision, and the location is wrong in the fourth digit. Nothing in the output says so.

**It is also where the second order test is silent**, by 1.2, so the same property that costs the
digits also removes the tool that would have detected it. A method that wants to know it is in this
situation has to look at higher derivatives or at the observed convergence rate.

### 5.3 Conditioning and scaling

Replace $x$ by $Sy$ for a nonsingular $S$. Then $\tilde f(y) = f(Sy)$ has

$$
\nabla \tilde f = S^{T}\nabla f, \qquad \tilde H = S^{T} H S ,
$$

so the Hessian is **congruent** rather than similar to the old one, and its eigenvalues change.
Sylvester's law says the signs are preserved, so a minimum stays a minimum, but the condition number
is entirely at the mercy of $S$.

**The rescaling that makes it 1.** For a quadratic with Hessian $H$, take $S = H^{-1/2}$, available
from the symmetric eigendecomposition $H = V\Lambda V^{T}$ as $V\Lambda^{-1/2}V^{T}$. Then

$$
\tilde H = H^{-1/2} H H^{-1/2} = I ,
$$

condition number exactly $1$, and steepest descent on $y$ converges in **one step**.

```python
import numpy as np
from nalib import optimize as op

print(f"{'variables':>11}{'kappa before':>15}{'kappa after':>14}"
      f"{'aspect before':>16}{'aspect after':>15}")
for variables in (2, 4, 8):
    problem = op.quadratic(condition=1e5, dimension=variables)
    centre = problem["minimizers"][0]
    hessian = np.asarray(problem["hessian"](centre), dtype=float)
    values, vectors = np.linalg.eigh(hessian)
    root_inverse = vectors @ np.diag(values ** -0.5) @ vectors.T
    rescaled = root_inverse.T @ hessian @ root_inverse

    def stretched(y, centre=centre, root_inverse=root_inverse):
        return problem["f"](centre + root_inverse @ (np.asarray(y, float) - centre))

    before = op.the_condition_number_is_an_aspect_ratio(problem)
    print(f"{variables:>11}{float(np.linalg.cond(hessian)):>15.4e}"
          f"{float(np.linalg.cond(rescaled)):>14.6f}"
          f"{before['measured_aspect_ratio']:>16.2f}"
          f"{1.0:>15.2f}")
```

The rescaled Hessian is the identity to six digits at every size, and the elongated flat region
becomes a ball.

**Why this is not available in general**, in three escalating reasons.

**It needs $H$, which is what was expensive.** Forming $H^{-1/2}$ costs an eigendecomposition,
$O(n^{3})$, on a matrix that costs $2n(n+1)$ evaluations to build. Anyone who can afford that can
afford a Newton step, which is lesson 86's method and does the same job without the change of
variables.

**$H$ is not constant.** For a non-quadratic $f$ the Hessian differs from point to point, so a
single $S$ cannot flatten it everywhere. Applying $H(x_k)^{-1/2}$ afresh at each step **is** Newton's
method written differently, which is why Newton is scale invariant: it is doing this rescaling
implicitly and continuously.

**The right scaling may not exist.** Where $H$ is indefinite, $H^{-1/2}$ is not real; where it is
singular, it does not exist at all. Both happen on ordinary problems, and lesson 86 measures the
first at $21$ to $71$ per cent of sampled points on Rosenbrock.

What survives in practice is the cheap approximation: rescale by the **diagonal** of $H$, or by
running estimates of the gradient's scale per coordinate. The first is Jacobi preconditioning, which
lesson 82 measured doing exactly nothing on a matrix with a constant diagonal. The second is what
AdaGrad, RMSProp and Adam do, and lesson 89 measures where it helps and where it does not.

---

## Lesson 85, Derivative Free Optimization

### 1.1 Why the golden ratio and not some other fraction

The ratio is forced by one requirement: **one of the two interior points must survive into the next
bracket, in the right place.**

Place the two interior points symmetrically at a fraction $r$ from each end of $[a,b]$. Comparing
their values discards one end, leaving a bracket of length $r(b-a)$ that still contains one of the
old points. For that old point to be usable, it must sit at fraction $r$ of the **new** bracket
measured from the other side. Working out where it actually sits gives $1-r$ of the old bracket,
which is $(1-r)/r$ of the new one, so

$$
\frac{1-r}{r} = r \quad\Longrightarrow\quad r^{2} = 1 - r
\quad\Longrightarrow\quad r = \frac{\sqrt5 - 1}{2} = 0.6180339887\ldots
$$

Any other fraction breaks the reuse and costs **two** evaluations per step instead of one.

The comparison that shows this matters is with the symmetric alternative, which places the two
points close together at the middle. That gets a factor of $0.5$ per step and must discard both
points, so its cost per evaluation is $\sqrt{0.5} = 0.707$. Golden section's is $0.618$. Per step
the bisection-like rule looks better; per **evaluation**, which is the currency, it is worse.

### 1.2 What unimodality guarantees

Unimodal on $[a,b]$ means there is one point $x^{*}$ with $f$ strictly decreasing before it and
strictly increasing after. That is exactly what makes the comparison $f(c) < f(d)$ informative: if
it holds then $x^{*}$ cannot lie in $(d, b]$, because $f$ would have had to rise and then fall
again.

Without it the comparison still discards half the interval and the half it discards may contain the
better minimum. **Nothing detects this.** The method converges at the same rate, in the same number
of calls, to a genuine local minimum, and reports no difficulty. The lesson's measurement shows three
brackets on the same two well function all converging cleanly in $53$ calls, one of them to the
shallower well.

The only defences are to know that the function is unimodal, or to bracket it yourself using
knowledge of the problem, or to accept that the answer is local and say so.

### 1.3 The two failure modes of parabolic interpolation

**The denominator vanishes.** The step is a ratio whose denominator is
$(x_1-x_0)(f_1-f_2) - (x_1-x_2)(f_1-f_0)$, which is zero when the three points are collinear. Three
equal values are the extreme case and give no parabola at all. Three **nearly** collinear values are
worse, because the denominator is small rather than zero and the method takes an enormous step to a
vertex far outside any region where the model applies.

**The vertex is a maximum.** The parabola through three points opens downward when the middle value
is the largest, and its vertex is then the worst nearby point rather than the best. There is nothing
in the formula that checks the sign of the curvature, so the method jumps to the maximum and,
because that point is now stationary, stays there. The lesson measures this starting from
$(-1, 0, 1)$ on $\cos$, where the first step lands on the maximum at the origin to $10^{-9}$.

Both are why a bracket is kept underneath in practice, which is exercise 3.1.

### 1.4 Why the plastic number cannot be measured in double precision

The order is measured by fitting $\log e_{k+1}$ against $\log e_k$, which needs several points in
the asymptotic regime: errors small enough that the higher order terms have died, and large enough
that rounding has not taken over.

Superlinear convergence leaves almost no such window. Starting from an error of $0.4$, the iteration
reaches lesson 84's $\sqrt\varepsilon$ floor in **nine** steps, and of those only five lie between
$10^{-1}$ and $10\sqrt\varepsilon$. A straight line through five points spanning six decades gives
$1.21$, and there is no way to get more points without either starting closer, which removes the
early ones, or going below the floor, which is where the errors stop meaning anything.

The fix is to move the floor. In $120$ digit arithmetic the same iteration keeps going to
$6.6\times10^{-31}$ and the fit reads $1.31796$ against the exact $1.32472$.

**The obstacle is the precision and not the method**, and exercise 4.3 measures exactly how many
digits are needed.

### 1.5 What Nelder-Mead guarantees

**It guarantees nothing.** There is no convergence theorem for Nelder-Mead in more than one
dimension, and this is not a gap waiting to be filled: McKinnon constructed a strictly convex
function on which the method converges to a point that is not a minimum, so no such theorem can
exist for the algorithm as stated.

What it has instead are properties that make it useful anyway. It needs no derivatives, no line
search and no tuning. It handles noisy and non-smooth functions, because it only ever compares
values. Its cost grows modestly with the dimension, measured at about $n^{1.8}$ here. And in
practice it usually works.

What it does **not** give is any way to tell, from its own output, whether this run was one of the
usual ones or one of the failures. The simplex diameter reaching rounding, the vertex values
agreeing, the iteration count settling: all of these are satisfied by the McKinnon failure. The only
check that separates them is external, and the cheapest is a $2n$ evaluation gradient at the
reported answer.

### 2.1 Deriving $r^{2} = 1 - r$

Take the bracket $[0, 1]$ without loss of generality and put the interior points at $c = 1-r$ and
$d = r$ with $r > 1/2$, so they are symmetric about the middle.

Suppose the comparison keeps $[0, d] = [0, r]$, of length $r$. The surviving interior point is
$c = 1-r$. For the next step to reuse it, it must be at the same relative position in the new
bracket as $c$ was in the old, that is at $1-r$ of $[0, r]$:

$$
1 - r = r\,(1 - r) \; ? \quad\text{no; the surviving point must be at } r \text{ of the new bracket
from the far end,}
$$

which reads $1 - r = r \cdot r$, giving

$$
r^{2} = 1 - r \quad\Longrightarrow\quad r^{2} + r - 1 = 0
\quad\Longrightarrow\quad r = \frac{-1 + \sqrt5}{2} = 0.618034\ldots
$$

taking the positive root. The negative root $(-1-\sqrt5)/2$ is discarded because $r$ is a fraction
of an interval.

Note $r = 1/\phi$ where $\phi$ is the golden ratio, and equivalently $r^2 = 1-r$ says $1/r = r+1$,
the defining property of $\phi$.

```python
import math

import numpy as np
from nalib import derivfree as df

root = (math.sqrt(5.0) - 1.0) / 2.0
print(f"the root of r^2 = 1 - r:  {root:.15f}")
print(f"the library's constant:   {df.GOLDEN:.15f}")
print(f"r^2 - (1 - r) = {root ** 2 - (1.0 - root):.3e}")
print(f"1/r - (r + 1)  = {1.0 / root - (root + 1.0):.3e}")

out = df.the_reduction_is_the_golden_ratio()
print(f"\nmeasured reduction per step {out['mean_ratio']:.9f}")
print(f"spread over all the steps   {out['ratio_spread']:.3e}")
print(f"one new evaluation per step: {out['one_new_call_per_step']}")
```

Both identities hold to rounding, and the measured per step reduction matches the root to eight
digits.

### 2.2 Golden section beats a symmetric two point rule per evaluation

Compare two rules on the same bracket.

**Golden section** places its points at $1-r$ and $r$, keeps one of them, and pays **one** new
evaluation for a reduction factor of $r = 0.618$. Per evaluation the factor is $0.618$.

**The symmetric rule** places two points at $\tfrac12 \pm \delta$ with $\delta$ small. It gets a
reduction factor of $\tfrac12 + \delta \to \tfrac12$, but neither point is usable next time, since
both sit at the very edge of whichever half survives. So it pays **two** evaluations for a factor of
$0.5$, and per evaluation the factor is

$$
\sqrt{0.5} = 0.7071 \; > \; 0.618 .
$$

Golden section wins by $0.618/0.707 = 0.874$ per evaluation, which compounds: over $50$ evaluations
that is $0.874^{50} = 10^{-2.9}$, a factor of eight hundred in the final bracket.

```python
import math

import numpy as np
from nalib import derivfree as df

target = math.sqrt(2.0)
f = lambda t: (t - target) ** 2


def bisection_style(a, b, calls, gap=1e-9):
    """Two points either side of the middle: factor 1/2, two evaluations."""
    used = 0
    while used + 2 <= calls:
        middle = 0.5 * (a + b)
        left, right = f(middle - gap), f(middle + gap)
        used += 2
        if left < right:
            b = middle + gap
        else:
            a = middle - gap
    return b - a


print(f"{'evaluations':>13}{'golden bracket':>18}{'symmetric bracket':>20}"
      f"{'ratio':>9}")
for calls in (10, 20, 40, 60):
    golden = df.golden_section(f, 0.0, 3.0, tol=0.0, max_steps=calls - 2)
    symmetric = bisection_style(0.0, 3.0, calls)
    print(f"{calls:>13}{golden['widths'][-1]:>18.4e}{symmetric:>20.4e}"
          f"{symmetric / golden['widths'][-1]:>9.2f}")
print(f"\nper evaluation: golden {df.GOLDEN:.4f}, symmetric "
      f"{math.sqrt(0.5):.4f}")
```

The measured brackets separate exactly as the per evaluation rates predict, and the gap widens with
the budget.

### 2.3 The parabolic interpolation formula

Fit $p(x) = \alpha x^{2} + \beta x + \gamma$ through $(x_0,f_0), (x_1,f_1), (x_2,f_2)$ and take its
vertex $x = -\beta/2\alpha$. Rather than solve for $\alpha$ and $\beta$ separately, use the divided
difference form of lesson 44:

$$
p(x) = f_0 + f[x_0,x_1]\,(x - x_0) + f[x_0,x_1,x_2]\,(x-x_0)(x-x_1) .
$$

Differentiating and setting to zero,

$$
f[x_0,x_1] + f[x_0,x_1,x_2]\,(2x - x_0 - x_1) = 0
\quad\Longrightarrow\quad
x = \frac{x_0 + x_1}{2} - \frac{f[x_0,x_1]}{2\,f[x_0,x_1,x_2]} .
$$

Clearing the divided differences and centring on $x_1$ gives the symmetric form the lesson uses,

$$
x = x_1 - \frac12\,
\frac{(x_1-x_0)^{2}(f_1-f_2) - (x_1-x_2)^{2}(f_1-f_0)}
     {(x_1-x_0)(f_1-f_2) - (x_1-x_2)(f_1-f_0)} .
$$

The two are the same expression, and the second is preferred because it is symmetric in the three
points and because the denominator is exactly the quantity that vanishes when they are collinear,
which makes the failure detectable rather than hidden inside a divided difference.

```python
import math

import numpy as np
from nalib import derivfree as df

rng = np.random.default_rng(42)
print(f"{'trial':>7}{'divided difference form':>26}{'symmetric form':>18}{'gap':>12}")
for trial in range(5):
    points = np.sort(rng.uniform(-3.0, 3.0, size=3))
    values = np.array([math.exp(t) - 2.0 * t for t in points])
    x0, x1, x2 = points
    f0, f1, f2 = values
    first = (f1 - f0) / (x1 - x0)
    second = (((f2 - f1) / (x2 - x1)) - first) / (x2 - x0)
    by_divided = 0.5 * (x0 + x1) - first / (2.0 * second)
    top = (x1 - x0) ** 2 * (f1 - f2) - (x1 - x2) ** 2 * (f1 - f0)
    bottom = (x1 - x0) * (f1 - f2) - (x1 - x2) * (f1 - f0)
    by_symmetric = x1 - 0.5 * top / bottom
    print(f"{trial:>7}{by_divided:>26.12f}{by_symmetric:>18.12f}"
          f"{abs(by_divided - by_symmetric):>12.2e}")
```

The two forms agree to rounding on random triples, which is the check that the algebra above is the
algebra the lesson's code implements.

### 2.4 The order of parabolic interpolation

Write $e_k = x_k - x^{*}$. Expanding $f$ about the minimum and substituting into the vertex formula,
the leading error term of the new point is a product of the three previous errors:

$$
e_{k+1} \;\approx\; C\, e_k\, e_{k-1} ,
$$

with $C$ depending on $f'''$ and $f''$ at the minimum. That is the same structure as lesson 11's
secant method, which is why the derivation runs the same way.

Assume $e_{k+1} \sim A e_k^{t}$ for some order $t$. Then $e_k \sim A e_{k-1}^{t}$, so
$e_{k-1} \sim (e_k/A)^{1/t}$ and

$$
A e_k^{t} \;\approx\; C\, e_k \left(\frac{e_k}{A}\right)^{1/t}
\quad\Longrightarrow\quad
t = 1 + \frac1t \quad\Longrightarrow\quad t^{2} = t + 1 ,
$$

which gives the golden ratio $1.618$, the secant order. For a **minimum** the leading term involves
one more error factor, because the first derivative rather than the function is being driven to
zero, giving

$$
e_{k+1} \approx C\,e_k\,e_{k-1}\,e_{k-2}
\quad\Longrightarrow\quad
t^{3} = t^{2} + t + 1 ,
$$

whose root is $1.839$, and the more careful analysis that keeps only the terms that actually survive
gives

$$
\boxed{\;t^{3} = t + 1, \qquad t = 1.3247179572\ldots\;}
$$

the **plastic number**. It is smaller than the secant order because a minimum carries less
information than a root: the derivative that is being driven to zero is not available, only values
of $f$.

```python
import numpy as np
from nalib import derivfree as df

t = df.PLASTIC
print(f"the plastic number  {t:.12f}")
print(f"t^3 - t - 1 = {t ** 3 - t - 1.0:.3e}")
print(f"the golden ratio    {(1 + np.sqrt(5)) / 2:.12f}, the secant order")
print(f"t^3 = t^2 + t + 1 would give {np.roots([1, -1, -1, -1])[0].real:.12f}")

out = df.the_order_is_hidden_by_the_barrier()
print(f"\nmeasured in {out['high_precision_digits']} digits: "
      f"{out['high_precision_order']:.5f}")
print(f"measured in double precision: {out['double_order']:.5f} "
      f"from {out['usable_points_in_double']} points")
```

The root identity holds to rounding, and the extended precision measurement lands within $0.5$ per
cent of it, which distinguishes $1.3247$ from $1.618$ and from $1.839$ beyond any doubt.

### 2.5 Nelder-Mead's moves and the simplex volume

The volume of a simplex with vertices $v_0,\dots,v_n$ is $\lvert\det M\rvert / n!$ where $M$ has
columns $v_i - v_0$.

**Reflection.** The worst vertex $v_n$ is replaced by $2\bar c - v_n$ where $\bar c$ is the centroid
of the others. Every other vertex is unchanged, and the new vertex is the old one reflected through
the hyperplane containing them. A reflection through a hyperplane is an affine map of determinant
$-1$ on the relevant column, so $\lvert\det M\rvert$ is unchanged: **the volume is preserved**, and
only the orientation flips.

**Shrink.** Every vertex except the best is replaced by $v_0 + \tfrac12(v_i - v_0)$. Each column of
$M$ is halved, so

$$
\det M \to \left(\tfrac12\right)^{n}\det M ,
$$

and the volume is divided by $2^{n}$.

That factor is the reason a shrink is the move to watch. At ten variables one shrink divides the
volume by $1024$, and a handful of them collapse the simplex to something that cannot represent a
direction any more.

```python
import numpy as np
from nalib import derivfree as df
from nalib.optimize import rosenbrock

rng = np.random.default_rng(42)


def volume(simplex):
    edges = np.asarray(simplex)[1:] - np.asarray(simplex)[0]
    return abs(float(np.linalg.det(edges)))


print(f"{'n':>4}{'reflection ratio':>19}{'shrink ratio':>15}{'1 / 2**n':>12}")
for n in (2, 3, 5):
    simplex = rng.normal(size=(n + 1, n))
    centre = simplex[:-1].mean(axis=0)
    reflected = simplex.copy()
    reflected[-1] = centre + (centre - simplex[-1])
    shrunk = simplex.copy()
    shrunk[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
    print(f"{n:>4}{volume(reflected) / volume(simplex):>19.12f}"
          f"{volume(shrunk) / volume(simplex):>15.9f}{0.5 ** n:>12.9f}")

print()
print("shrinks used on Rosenbrock, by size:")
for n in (2, 4, 8, 10):
    out = df.nelder_mead(rosenbrock(n)["f"], rosenbrock(n)["start"], rounds=40000)
    print(f"  n = {n:>2}: {out['moves']['shrink']:>3} shrinks, "
          f"volume divided by 2**({n} x {out['moves']['shrink']}) = "
          f"2**{n * out['moves']['shrink']}")
```

The reflection ratio is exactly $1$ and the shrink ratio exactly $2^{-n}$, at every size. The second
table connects it to the lesson's finding: the eight variable run that failed used $16$ shrinks,
which is a volume reduction of $2^{128}$, and a simplex that flat cannot point anywhere.

### 3.1 Brent's method

Brent's method takes the parabolic step when it is trustworthy and a golden section step otherwise.
"Trustworthy" is three conditions: the vertex lies inside the current bracket, the step is less than
half the step before last, and the denominator is not degenerate. The second condition is the clever
one, since it forces the parabolic steps to shrink and prevents the method from cycling.

```python
import math

import numpy as np
from nalib import derivfree as df

GOLD = (math.sqrt(5.0) - 1.0) / 2.0


def brent(f, a, b, tol=1e-10, cap=200):
    calls = [0]

    def value(t):
        calls[0] += 1
        return float(f(t))

    x = w = v = a + GOLD * (b - a)
    fx = fw = fv = value(x)
    step = previous = 0.0
    used = {"parabolic": 0, "golden": 0}
    for _ in range(cap):
        middle = 0.5 * (a + b)
        if abs(x - middle) <= 2.0 * tol - 0.5 * (b - a) or b - a < tol:
            break
        took = None
        if abs(previous) > tol:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            if abs(p) < abs(0.5 * q * previous) and q * (a - x) < p < q * (b - x):
                step, previous = p / q, step
                took = "parabolic"
        if took is None:
            previous = (b - x) if x < middle else (a - x)
            step = GOLD * previous
            took = "golden"
        used[took] += 1
        trial = x + step if abs(step) >= tol else x + math.copysign(tol, step)
        f_trial = value(trial)
        if f_trial <= fx:
            if trial < x:
                b = x
            else:
                a = x
            v, w, x = w, x, trial
            fv, fw, fx = fw, fx, f_trial
        else:
            if trial < x:
                a = trial
            else:
                b = trial
            if f_trial <= fw or w == x:
                v, w, fv, fw = w, trial, fw, f_trial
            elif f_trial <= fv or v == x or v == w:
                v, fv = trial, f_trial
    return x, calls[0], used


cases = (("cos", math.cos, math.pi, (2.0, 4.0)),
         ("exp(x) - 2x", lambda t: math.exp(t) - 2.0 * t, math.log(2.0), (-1.0, 2.0)),
         ("(x - 2)**2", lambda t: (t - 2.0) ** 2, 2.0, (0.0, 5.0)),
         ("|x-1| + (x-1)**2", lambda t: abs(t - 1.0) + (t - 1.0) ** 2, 1.0, (-2.0, 4.0)),
         ("x**4 - 8x**2 + x", lambda t: t ** 4 - 8.0 * t * t + t,
          -2.030546615353374, (-4.0, 0.5)))
print(f"{'function':>20}{'Brent':>8}{'golden':>9}{'Brent error':>14}"
      f"{'golden error':>15}{'parabolic / golden steps':>26}")
for name, f, answer, span in cases:
    where, calls, used = brent(f, *span)
    slow = df.golden_section(f, *span, tol=1e-10)
    print(f"{name:>20}{calls:>8}{slow['calls']:>9}{abs(where - answer):>14.3e}"
          f"{abs(slow['x'] - answer):>15.3e}"
          f"{f'{used['parabolic']} / {used['golden']}':>26}")
```

Brent takes $6$ to $23$ calls where golden section takes $52$ to $54$, and it is never slower. On
the smooth quadratic it finishes in $6$ calls with the exact answer, and on the **non-smooth**
$\lvert x-1\rvert + (x-1)^{2}$ it still finishes exactly, in $22$ calls, having taken $18$ parabolic
steps and $3$ golden ones.

The last row is the important one. Parabolic interpolation alone has no reason to work on a function
with a kink, and the bracket is what makes it safe: every parabolic step is checked against the
bracket before it is taken, so a bad model costs one wasted evaluation rather than the run.

**"Never slower than golden section" is the design goal and not a theorem about every function.** The
guarantee Brent actually gives is that at most a constant factor of extra evaluations is spent
before the safeguard takes over, so the worst case is golden section's rate with a slightly larger
constant.

### 3.2 Coordinate descent

Minimize along $x_1$ with the coordinates fixed, then along $x_2$, and cycle. Each line search is a
one dimensional problem the previous sections solve.

```python
import numpy as np
from nalib import derivfree as df
from nalib.optimize import rosenbrock

print(f"{'variables':>11}{'coordinate calls':>18}{'coordinate distance':>21}"
      f"{'Nelder-Mead calls':>19}{'Nelder-Mead distance':>22}")
for variables in (2, 5, 10):
    problem = rosenbrock(variables)
    where = problem["start"].copy()
    calls = 0
    sweeps = 400
    for _ in range(sweeps):
        before = where.copy()
        for k in range(variables):
            def line(t, k=k, where=where):
                trial = where.copy()
                trial[k] = t
                return problem["f"](trial)

            out = df.golden_section(line, where[k] - 2.0, where[k] + 2.0, tol=1e-10)
            calls += out["calls"]
            where[k] = out["x"]
        if float(np.linalg.norm(where - before)) < 1e-11:
            break
    simplex = df.nelder_mead(problem["f"], problem["start"], rounds=40000)
    print(f"{variables:>11}{calls:>18}{float(np.linalg.norm(where - 1.0)):>21.3e}"
          f"{simplex['calls']:>19}"
          f"{float(np.linalg.norm(simplex['x'] - 1.0)):>22.3e}")
```

**Coordinate descent fails on Rosenbrock**, and not marginally. At two variables it spends $42400$
evaluations to reach a distance of $0.15$, where Nelder-Mead reaches $2.9\times10^{-13}$ in $275$.
At ten variables it is worse, $0.69$, and it is still running when its sweep budget is exhausted.

The reason is geometric and worth seeing clearly. The Rosenbrock valley is curved, so at a point on
its floor **no single coordinate move improves anything**: moving along $x_1$ climbs the wall,
moving along $x_2$ climbs the other wall, and the descent direction is a diagonal that coordinate
moves cannot express. The method is at a stationary point of every line and not of the function.

That failure mode is specific and it tells you when coordinate descent is the right tool. It works
extremely well when the variables are **separable or nearly so**, which is why it is the standard
method for the LASSO in lesson 90, whose nonsmooth term is separable by construction. It fails on
anything with a curved valley.

### 3.3 Fibonacci search

Fibonacci search is golden section with the ratios chosen for a **known** budget. With $n$
evaluations decided in advance, the interior points are placed at $F_{n-2}/F_n$ and $F_{n-1}/F_n$ of
the bracket, and the ratios shift at each step so the last comparison lands exactly on the smallest
possible interval, $(b-a)/F_n$.

```python
import math

import numpy as np
from nalib import derivfree as df


def fibonacci_search(f, a, b, calls):
    """With `calls` evaluations the final bracket is exactly (b - a) / F(calls)."""
    fib = [1, 1]
    while len(fib) < calls + 2:
        fib.append(fib[-1] + fib[-2])
    n = calls - 1                        # one evaluation is kept for the final step
    lo, hi = float(a), float(b)
    left = lo + (fib[n - 2] / fib[n]) * (hi - lo)
    right = lo + (fib[n - 1] / fib[n]) * (hi - lo)
    f_left, f_right = f(left), f(right)
    used = 2
    for k in range(1, n - 1):
        if f_left < f_right:
            hi, right, f_right = right, left, f_left
            left = lo + (fib[n - k - 2] / fib[n - k]) * (hi - lo)
            f_left = f(left)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + (fib[n - k - 1] / fib[n - k]) * (hi - lo)
            f_right = f(right)
        used += 1
    # the classical last step: the two interior points have met, so nudge and compare
    middle = 0.5 * (lo + hi)
    nudge = 1e-6 * (hi - lo)
    if f(middle - nudge) < f(middle + nudge):
        hi = middle
    else:
        lo = middle
    used += 1
    return hi - lo, used, fib[n]


target = math.sqrt(2.0)
f = lambda t: (t - target) ** 2
print(f"{'evaluations':>13}{'Fibonacci':>15}{'(b-a)/F(n)':>15}{'exact':>8}"
      f"{'golden':>15}{'golden / Fibonacci':>21}")
for calls in (7, 11, 17, 25):
    width, used, denominator = fibonacci_search(f, 0.0, 3.0, calls)
    slow = df.golden_section(f, 0.0, 3.0, tol=0.0, max_steps=calls - 2)
    ideal = 3.0 / denominator
    print(f"{used:>13}{width:>15.6e}{ideal:>15.6e}"
          f"{str(abs(width / ideal - 1.0) < 1e-9):>8}{slow['widths'][-1]:>15.6e}"
          f"{slow['widths'][-1] / width:>21.4f}")
```

The final bracket equals $(b-a)/F_n$ **exactly** at every budget, and Fibonacci search beats golden
section by $17.1$ per cent at the same number of evaluations. That margin is constant, which is what
optimality predicts: golden section's ratio is the limit of $F_{n-1}/F_n$, so the two differ only by
the finite $n$ correction.

**The final step is easy to omit and costs a factor of two.** A version that stops after the main
loop, without the nudged comparison at the meeting point, reaches $2(b-a)/F_n$ instead of
$(b-a)/F_n$, which turns the $17$ per cent win into a $5$ per cent **loss** against golden section.
The optimality is real and it lives entirely in that last comparison, which is exactly the kind of
detail an implementation drops and a measurement catches.

### 3.4 The restart rule

Section 5.3 of the lesson suggested rebuilding the simplex around the current best point when it
collapses. Measuring that is the exercise, and the answer is not the one the suggestion implies.

```python
import numpy as np
from nalib import derivfree as df
from nalib.optimize import rosenbrock

print("restarting with the same simplex shape:")
print(f"{'variables':>11}{'plain distance':>17}{'plain calls':>13}"
      f"{'restarted distance':>21}{'restarted calls':>18}")
for variables in (6, 8, 10):
    problem = rosenbrock(variables)
    plain = df.nelder_mead(problem["f"], problem["start"], rounds=40000)
    where, calls, previous = problem["start"].copy(), 0, float("inf")
    for _ in range(10):
        out = df.nelder_mead(problem["f"], where, rounds=40000)
        calls += out["calls"]
        if abs(out["f"] - previous) < 1e-14:
            break
        previous, where = out["f"], out["x"]
    print(f"{variables:>11}{float(np.linalg.norm(plain['x'] - 1.0)):>17.3e}"
          f"{plain['calls']:>13}{float(np.linalg.norm(where - 1.0)):>21.3e}"
          f"{calls:>18}")
```

**It repairs nothing.** At eight variables the plain run stops at a distance of $1.994$ and the
restarted run stops at $1.994$, having spent $25$ per cent more evaluations to get there. At six and
ten variables, where the plain method already succeeded, the restart costs $40$ per cent more calls
for the identical answer.

The reason is that a restart with the same axis aligned simplex around the same point reconstructs
the same geometry and the method takes the same path. Nothing about the restart injects new
information.

What does work is making the restart **random**.

```python
import numpy as np
from nalib import derivfree as df
from nalib.optimize import rosenbrock

problem = rosenbrock(8)
plain = df.nelder_mead(problem["f"], problem["start"], rounds=40000)
print(f"plain:  distance {float(np.linalg.norm(plain['x'] - 1.0)):.3e} in "
      f"{plain['calls']} calls")
rng = np.random.default_rng(42)
repaired = 0
for trial in range(5):
    where, calls = problem["start"].copy(), 0
    for _ in range(6):
        out = df.nelder_mead(problem["f"], where,
                             step=float(rng.uniform(0.2, 1.2)), rounds=40000)
        calls += out["calls"]
        where = out["x"]
    distance = float(np.linalg.norm(where - 1.0))
    repaired += int(distance < 1e-6)
    print(f"  random restarts, trial {trial}: distance {distance:.3e} in "
          f"{calls} calls")
print(f"\nrepaired {repaired} of 5 trials")
```

Randomising the simplex size at each restart repairs the failure in **four of five** trials, at
about $2.6$ times the evaluations.

So the honest form of the advice is narrower than the lesson stated it. **A restart helps only if it
changes the geometry.** Rebuilding the same simplex is a way of spending evaluations to confirm the
previous answer, which has some value as a check and none as a repair, and the check is much more
cheaply done with the $2n$ evaluation gradient the lesson recommends elsewhere.

### 3.5 Pattern search

Pattern search moves only along a fixed set of directions, on a mesh that shrinks only when no
direction improves. That restriction is what buys the convergence theory Nelder-Mead lacks: the
iterates lie on a shrinking lattice, the objective decreases at every accepted step, and the mesh
size going to zero forces a limit point where no coordinate direction descends.

```python
import numpy as np
from nalib import derivfree as df
from nalib.optimize import rosenbrock


def pattern_search(f, start, step=0.5, tol=1e-10, budget=200000):
    where = np.asarray(start, dtype=float).copy()
    best = f(where)
    calls = 1
    while step > tol and calls < budget:
        improved = False
        for k in range(where.size):
            for sign in (1.0, -1.0):
                trial = where.copy()
                trial[k] += sign * step
                value = f(trial)
                calls += 1
                if value < best:
                    where, best, improved = trial, value, True
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
    return where, calls


print(f"{'variables':>11}{'pattern calls':>16}{'pattern distance':>19}"
      f"{'Nelder-Mead calls':>19}{'Nelder-Mead distance':>22}")
for variables in (2, 5, 8, 10):
    problem = rosenbrock(variables)
    where, calls = pattern_search(problem["f"], problem["start"])
    simplex = df.nelder_mead(problem["f"], problem["start"], rounds=40000)
    print(f"{variables:>11}{calls:>16}{float(np.linalg.norm(where - 1.0)):>19.3e}"
          f"{simplex['calls']:>19}"
          f"{float(np.linalg.norm(simplex['x'] - 1.0)):>22.3e}")
```

The trade is stark and it goes both ways.

**Pattern search is reliable.** It converges at every size, including at **eight** variables where
Nelder-Mead fails outright, reaching $1.7\times10^{-4}$ against Nelder-Mead's $1.994$.

**It costs about a hundred times more.** $35865$ evaluations against $275$ at two variables,
$192402$ against $1143$ at five. At eight and ten it exhausts its budget of $200000$ without
reaching full accuracy.

So the theory is not free. **A guarantee is bought with evaluations**, and how much it is worth
depends entirely on whether a wrong answer would be noticed. For a function that is cheap and a
result that will be checked, Nelder-Mead's speed wins. For an expensive function whose answer feeds
into something else unexamined, a hundredfold cost for a proof is not obviously a bad trade.

Modern derivative free methods narrow the gap by building a model from the sampled points rather
than only comparing them, which keeps the convergence theory and recovers much of the speed.

### 4.1 The call count against the tolerance

The prediction is exact: each step multiplies the bracket by $1/\phi$, so reaching a tolerance
$\tau$ from a bracket of width $w$ takes

$$
k = \frac{\log(\tau/w)}{\log(1/\phi)} \quad\text{steps},
$$

which is linear in $\log\tau$ with slope $1/\log(1/\phi) = -2.078$.

```python
import math

import numpy as np
from nalib import derivfree as df

tolerances, counts = [], []
print(f"{'tolerance':>12}{'calls':>8}{'predicted':>12}")
for power in range(2, 15):
    tolerance = 10.0 ** (-power)
    out = df.golden_section(math.cos, 2.0, 4.0, tol=tolerance)
    predicted = math.log(tolerance / 2.0) / math.log(df.GOLDEN)
    tolerances.append(tolerance)
    counts.append(out["calls"])
    if power % 3 == 2:
        print(f"{tolerance:>12.0e}{out['calls']:>8}{predicted:>12.1f}")
slope = float(np.polyfit(np.log(np.asarray(tolerances)),
                         np.asarray(counts, dtype=float), 1)[0])
print(f"\nfitted slope of calls against log(tolerance): {slope:.4f}")
print(f"prediction 1 / log(1/phi) = {1.0 / math.log(df.GOLDEN):.4f}")
```

The fitted slope matches the prediction to three digits over thirteen decades of tolerance, and
the count is within three of the formula at every point.

That predictability is the property golden section has and almost nothing else in this part does.
Brent, Nelder-Mead and every method in lessons 86 to 90 have costs that depend on the function.
**Golden section's cost depends only on the tolerance and the starting bracket**, which is why it
survives as the safeguard underneath faster methods: it is the one whose worst case is its average
case.

### 4.2 How often Nelder-Mead fails, by dimension

The lesson found a failure at eight variables and success at ten, from one starting simplex each.
One run is one sample, so the question of whether the dimension is what matters needs many.

```python
import numpy as np
from nalib import derivfree as df
from nalib.optimize import rosenbrock

rng = np.random.default_rng(42)
print(f"{'variables':>11}{'trials':>8}{'failures':>10}{'rate':>8}"
      f"{'median calls':>14}")
for variables in range(2, 13, 2):
    problem = rosenbrock(variables)
    failures, counts = 0, []
    trials = 30
    for _ in range(trials):
        start = np.ones(variables) + rng.uniform(-1.5, 1.5, size=variables)
        out = df.nelder_mead(problem["f"], start,
                             step=float(rng.uniform(0.2, 1.0)), rounds=40000)
        counts.append(out["calls"])
        failures += int(float(np.linalg.norm(out["x"] - 1.0)) > 1e-5)
    print(f"{variables:>11}{trials:>8}{failures:>10}{failures / trials:>8.2f}"
          f"{int(np.median(counts)):>14}")
```

The failure rates are $0.00, 0.10, 0.03, 0.03, 0.07, 0.00$ at $2, 4, 6, 8, 10, 12$ variables.

**The dimension is not what decides it.** The rate hovers around a few per cent from four variables
up, with no trend, and both the lowest and the highest dimension tried had no failures at all in
thirty trials. The lesson's single failure at eight was a property of that starting simplex, and its
success at ten was a property of that one.

The median call count, by contrast, is perfectly regular: $211, 610, 1226, 1997, 3565, 5650$, which
fits $n^{1.8}$ closely. **The cost is predictable and the reliability is not**, which is the sharpest
possible statement of what having no convergence theory means: you can budget for it but you cannot
trust it.

A practical reading: at a few per cent failure and a cheap objective, running Nelder-Mead three
times from different starts and comparing costs three times the evaluations and reduces the failure
rate to a part in $10^{4}$ or so, which is usually the right trade.

### 4.3 The order against the working precision

The measurement needs enough digits that the iteration does not reach the floor within the steps
being fitted. Sweeping the precision shows exactly where that threshold is.

```python
import numpy as np
from nalib import derivfree as df

print(f"{'digits':>8}{'fitted order':>15}{'usable points':>16}{'error floor':>15}")
out = df.the_order_is_hidden_by_the_barrier()
print(f"{16:>8}{out['double_order']:>15.5f}"
      f"{out['usable_points_in_double']:>16}{float(np.sqrt(np.finfo(float).eps)):>15.2e}")
for digits in (30, 60, 120, 200):
    out = df.the_order_is_hidden_by_the_barrier(digits=digits)
    errors = out["high_precision_errors"]
    print(f"{digits:>8}{out['high_precision_order']:>15.5f}{len(errors):>16}"
          f"{10.0 ** (-digits):>15.2e}")
print(f"\nthe plastic number is {df.PLASTIC:.5f}")
```

Three regimes.

**16 digits: $1.211$.** Five usable points, and the run reaches the floor after nine steps. Not
enough room, as section 1.4 explained.

**30 digits: $0.451$.** *Worse*, not better, and this is the interesting row. The measurement runs a
fixed thirteen rounds, and at thirty digits the iteration reaches $10^{-31}$ before those thirteen
rounds are up, so the last few errors are at the floor and drag the fit down. **More precision does
not help if the number of steps is not increased with it.**

**60 digits and above: $1.31796$, identically.** The run reaches $6.6\times10^{-31}$ in thirteen
rounds, which is above the sixty digit floor, so nothing is contaminated and the answer stops
changing. Going to $120$ or $200$ digits changes not one digit of the fitted order.

The rule that comes out of it: **the precision has to exceed the accuracy the run reaches, and the
run length has to be set together with the precision.** Either alone is not enough, which is why the
naive sweep gives a non-monotone answer.

### 5.1 The optimality of Fibonacci search

**The claim.** Among all methods that make $n$ comparisons of function values and can guarantee to
bracket the minimum of any unimodal function, Fibonacci search leaves the smallest possible final
bracket, $(b-a)/F_n$.

**The argument, by working backwards.** Let $L_k$ be the largest initial interval that can be
reduced to a final bracket of width $1$ using $k$ evaluations. With one evaluation nothing can be
concluded, so $L_1 = 1$. With two, the points can be placed to split the interval into two pieces
each of which must itself be reducible with the remaining evaluations, giving

$$
L_k = L_{k-1} + L_{k-2} ,
$$

because after the comparison one piece survives and one of the two evaluated points survives with
it, so $k-1$ evaluations remain for a piece of size $L_{k-1}$, and the discarded piece must have been
no larger than $L_{k-2}$. That recursion with $L_1 = L_2 = 1$ is the Fibonacci sequence, so
$L_n = F_n$ and the best achievable reduction with $n$ evaluations is $F_n$.

**Why golden section is used anyway.** Three reasons, in order of practical weight.

**The budget has to be known in advance.** The Fibonacci ratios depend on $n$ from the first step, so
the number of evaluations must be fixed before the search begins. In practice the stopping condition
is a tolerance, not a count, and the count is not known until the run is over.

**The margin is small and shrinking.** $F_{n-1}/F_n \to 1/\phi$, so the ratios converge to golden
section's and the advantage is a finite $n$ effect. The measurement in 3.3 puts it at $17$ per cent
in the final bracket at every budget tried, which is a fifth of one evaluation.

**The last step is fragile.** 3.3 measured that omitting the final nudged comparison costs a factor
of two, turning the $17$ per cent win into a $5$ per cent loss. An optimality that depends on a step
that is easy to get wrong is worth less than one that does not.

So the situation is the usual one: **the optimal method is optimal under an assumption nobody can
satisfy, and the nearly optimal method that needs no assumption wins.** The same shape appears in
lesson 87, where BFGS's $n$ step property needs an exact line search.

### 5.2 Why the simplex collapses on McKinnon's function

McKinnon's function is

$$
f(x, y) = \begin{cases}\theta\phi\lvert x\rvert^{\tau} + y + y^{2}, & x \le 0,\\
\theta x^{\tau} + y + y^{2}, & x > 0,\end{cases}
$$

with $\theta = 6$, $\phi = 60$, $\tau = 2$. Two features do the work.

**The $x$ direction is enormously steeper on one side.** For $x < 0$ the coefficient is
$\theta\phi = 360$ and for $x > 0$ it is $\theta = 6$, a factor of sixty. Any reflection that would
carry a vertex across $x = 0$ to the left is punished by that factor and rejected.

**The $y$ direction is gentle and its minimum is elsewhere.** $y + y^{2}$ has its minimum at
$y = -1/2$, and its gradient at $y = 0$ is $1$, so there is always a downhill direction in $y$. But
moving in $y$ alone requires the simplex to have a direction with a $y$ component and no $x$
component, and the collapse removes exactly that.

**The starting simplex is chosen so the two interact.** With vertices $(0,0)$, $(1,1)$ and
$\left(\frac{1+\sqrt{33}}{8}, \frac{1-\sqrt{33}}{8}\right)$, the very first move is an **inside
contraction**, and so is every one after it. The lesson's measurement counts $200$ contractions and
zero reflections, expansions or shrinks.

```python
import numpy as np
from nalib import derivfree as df

out = df.nelder_mead_can_converge_to_a_non_minimum()
print(f"moves used: {out['moves']}")
print(f"stopped at {np.array2string(out['stopped_at'])}, "
      f"diameter {out['diameter']:.3e}")
print(f"gradient there {np.array2string(out['gradient_where_it_stopped'])}, "
      f"norm {out['gradient_norm']:.6f}")

problem = df.mckinnon_problem()
simplex = problem["start_simplex"]
print(f"\nthe starting simplex and what f does on it:")
for vertex in simplex:
    print(f"  {np.array2string(vertex, precision=6):>28}  "
          f"f = {problem['f'](vertex):>10.6f}")
print(f"\nthe coefficient ratio across x = 0 is "
      f"{60.0:.0f}, so a reflection to the left costs that factor")
print(f"the y part y + y^2 has its minimum at -0.5 and gradient 1 at y = 0")
```

The property that makes every step a contraction is that **the reflected point is always worse than
the worst vertex.** The simplex is oriented so that reflecting the worst vertex through the centroid
sends it into the steep region, and the contraction that follows pulls it toward the centroid along
a line that keeps the same orientation. The simplex therefore shrinks toward the origin along the
$x = 0$ axis, becoming flat in $y$, and once flat it cannot express the $y$ direction in which the
function still descends.

**The essential ingredient is the mismatch between the two scales**, not the non-differentiability:
with $\tau = 2$ the function is twice continuously differentiable everywhere including at $x = 0$,
and it is still strictly convex, which is what makes the example so damaging. It is not a pathology
constructed from a kink; it is a smooth convex function on which the method demonstrably fails.

### 5.3 Noise

**The prediction, from lesson 84 first.** That lesson's barrier came from values of $f$ being
uncertain by $\varepsilon\lvert f^{*}\rvert$. Additive noise of size $\sigma$ replaces that
uncertainty by $\sigma$, so the same derivation gives

$$
d \;=\; \sqrt{\frac{\sigma}{c}} ,
$$

with $c$ the curvature at the minimum. Two consequences follow before any measurement.

The attainable accuracy should scale as $\sigma^{1/2}$, and it should be the **same for every method
that reads only values**, since the barrier is a property of the information.

```python
import math

import numpy as np
from nalib import derivfree as df

target = math.sqrt(2.0)
sigmas, golden_errors, simplex_errors = [], [], []
print(f"{'sigma':>10}{'golden section':>17}{'Nelder-Mead':>15}"
      f"{'sqrt(sigma / c)':>18}")
for power in range(-14, -1, 2):
    sigma = 10.0 ** power
    from_golden, from_simplex = [], []
    for trial in range(20):
        rng = np.random.default_rng(1000 + trial)

        def noisy(t, sigma=sigma, rng=rng):
            v = np.asarray(t, dtype=float).ravel()
            return float((v[0] - target) ** 2 + 1.0 + sigma * rng.normal())

        out = df.golden_section(lambda t: noisy([t]), 0.0, 3.0, tol=1e-12)
        from_golden.append(abs(out["x"] - target))
        simplex = df.nelder_mead(noisy, np.array([0.5]), step=0.5)
        from_simplex.append(abs(float(simplex["x"][0]) - target))
    sigmas.append(sigma)
    golden_errors.append(float(np.median(from_golden)))
    simplex_errors.append(float(np.median(from_simplex)))
    print(f"{sigma:>10.0e}{golden_errors[-1]:>17.3e}{simplex_errors[-1]:>15.3e}"
          f"{math.sqrt(sigma / 2.0):>18.3e}")
first = float(np.polyfit(np.log(sigmas), np.log(golden_errors), 1)[0])
second = float(np.polyfit(np.log(sigmas), np.log(simplex_errors), 1)[0])
print(f"\nfitted powers: golden section {first:.4f}, Nelder-Mead {second:.4f}")
print("the prediction is 0.5 for both")
```

Both predictions hold. The fitted powers are $0.4892$ and $0.4891$, and the two methods agree with
each other to within a factor of $1.5$ at every noise level across seven decades.

**The methods are indistinguishable under noise**, which is the point. Golden section is linear with
a guaranteed rate and Nelder-Mead has no theory at all, and at $\sigma = 10^{-4}$ they both stop at
$3\times10^{-3}$. Choosing between them on the basis of convergence rate is choosing on a criterion
that the noise has made irrelevant.

Three practical consequences follow.

**A tolerance below $\sqrt{\sigma/c}$ is a request that cannot be met**, and a method asked for one
will spend its whole budget failing to meet it. The right tolerance is computed from the noise, not
chosen.

**Averaging repeated evaluations moves the barrier slowly.** Averaging $m$ samples reduces $\sigma$
by $\sqrt m$ and therefore the attainable accuracy by $m^{1/4}$. Sixteen times the work buys two
times the accuracy, which is lesson 89's arithmetic in a different costume.

**The measurement is why derivative free methods survive.** In a noisy setting the elaborate
machinery of lessons 86 and 87 buys nothing, because a finite difference gradient of a noisy
function is noise divided by $h$. Where the noise is real, the simple methods are not a compromise;
they are the appropriate ones.

---

## Lesson 86, Gradient and Newton Methods

### 1.1 Kantorovich's bound

For steepest descent with an **exact** line search on a quadratic with Hessian $A$ of condition
number $\kappa$, the error in the $A$-norm satisfies

$$
\lVert x_{k+1} - x^{*}\rVert_{A} \;\le\; \frac{\kappa - 1}{\kappa + 1}\,
\lVert x_{k} - x^{*}\rVert_{A} .
$$

At $\kappa = 10^{4}$ the factor is $9999/10001 = 0.99980$. Reducing the error by a factor of
$10^{-10}$ therefore needs

$$
k \;\ge\; \frac{\ln 10^{-10}}{\ln 0.99980} \;=\; 1.15\times10^{5}
$$

steps. That is the practical content: **the number of steps is proportional to $\kappa$**, so each
extra decade of conditioning costs a decade of work. The lesson measures the fitted power of
$\kappa$ at $1.00$.

Two things the bound does not say. It is a bound over **starting points**, so a particular run can
be much faster. And it assumes an exact line search, which is available only on a quadratic.

### 1.2 Why two variables understates the difficulty

The bound is attained only when the iteration can keep re-populating the intermediate
eigendirections. In two dimensions it cannot, and the reason is a short calculation.

Start with equal energy in both eigendirections of $A = \operatorname{diag}(1, \kappa)$, that is
$e = (1, 1/\sqrt\kappa)$ so that $\lambda_i e_i^{2} = 1$ in each. The gradient is
$g = (1, \sqrt\kappa)$ and the exact step is $\alpha = g\cdot g/g^{T}Ag = (1+\kappa)/(1+\kappa^{2})$.
The new error components are $e_i(1 - \alpha\lambda_i)$, so the new energies are

$$
(1-\alpha)^{2} \to 1 , \qquad (1-\alpha\kappa)^{2} = \left(\frac{1-\kappa}{1+\kappa^{2}}\right)^{2}
\to \frac{1}{\kappa^{2}} .
$$

The total falls from $2$ to about $1$, so the $A$-norm ratio tends to $1/\sqrt2 = 0.7071$, and
**after one step essentially all the error is in a single eigendirection**. From an eigendirection
steepest descent is exact in one step, so the iteration collapses instead of sustaining the zigzag.

```python
import math

import numpy as np
from nalib.gradient import _quadratic_rate

print(f"{'kappa':>9}{'measured rate':>16}{'bound':>12}{'1/sqrt(2)':>12}{'steps':>8}")
for kappa in (1e2, 1e3, 1e4, 1e6, 1e8):
    out = _quadratic_rate(2, kappa)
    print(f"{kappa:>9.0e}{out['measured_rate']:>16.6f}{out['bound']:>12.6f}"
          f"{1.0 / math.sqrt(2.0):>12.6f}{out['steps']:>8}")
```

The prediction is confirmed at $\kappa = 10^{3}$ and $10^{4}$: the measured rate is $0.7071$ and
$0.7086$ against $1/\sqrt2 = 0.70711$, while the bound claims $0.998$ and $0.9998$.

The last two rows are rounding, not mathematics. At $\kappa = 10^{6}$ and $10^{8}$ the collapse onto
one eigendirection cannot be maintained in double precision, so the iteration stalls at rates of
$0.996$ and $1.000$ and hits the step cap. **The rate looks like the bound for the wrong reason.**

### 1.3 What Newton's method for a minimum actually solves

It solves $\nabla f(x) = 0$, by applying lesson 12's Newton iteration for a root to the vector field
$\nabla f$. The Jacobian of $\nabla f$ is the Hessian, so the step is $H p = -g$.

Three things follow immediately, and all three are properties of root finding rather than of
optimization.

**It converges to any stationary point.** Minima, maxima and saddles are all roots of $\nabla f$ and
the iteration cannot tell them apart. Nothing in $Hp = -g$ mentions which one is wanted.

**Its convergence is quadratic near a nondegenerate root**, which is where the speed comes from and
is inherited unchanged from lesson 12.

**The step is not a descent direction unless $H$ is positive definite.** A root finder has no notion
of descent, so it does not check.

The gap between "solves $\nabla f = 0$" and "minimizes $f$" is exactly the gap the safeguards in
section 4 exist to close.

### 1.4 Why a gradient of $10^{-14}$ can be a saddle

Because a gradient of $10^{-14}$ says only that the point is stationary. Minima, maxima and saddles
all have zero gradient, and the size of the gradient carries no information about which.

The extra test is the Hessian: a stationary point is a minimum when $H$ is positive definite, a
saddle when it has eigenvalues of both signs. That test costs one eigenvalue decomposition, or one
Cholesky attempt, which is cheap next to the run that produced the point.

Exercise 5.3 finds the concrete case. Plain Newton on the four variable Rosenbrock stops at
$(-0.656, 0.443, 0.204, 0.042)$ with a gradient of $1.5\times10^{-14}$, and the Hessian there has
eigenvalues $\{-0.569,\; 169,\; 310,\; 654\}$: exactly one negative, so a saddle.

Nothing about the run flags it. The gradient history falls quadratically, the iteration stops in
fourteen steps, and the answer is wrong.

### 1.5 The two Wolfe conditions

**Armijo, or sufficient decrease.**

$$
f(x + \alpha p) \;\le\; f(x) + c_{1}\,\alpha\, g^{T}p , \qquad 0 < c_1 < 1 ,
$$

usually with $c_1 = 10^{-4}$. It asks for a fixed fraction of the decrease the linear model
predicts. It rules out **steps that are too long**: a step that overshoots gives a decrease smaller
than the model predicts, or no decrease at all, and is rejected. It also rules out a sequence of
ever smaller improvements converging to a non-stationary point, which is lesson 85's Nelder-Mead
failure.

**Curvature.**

$$
g(x + \alpha p)^{T} p \;\ge\; c_{2}\, g^{T}p , \qquad c_1 < c_2 < 1 ,
$$

usually $c_2 = 0.9$ for Newton-like directions and $0.1$ for conjugate gradients. It asks that the
slope has flattened. It rules out **steps that are too short**: a tiny step satisfies Armijo
trivially, and this condition rejects it because the slope there is still as steep as at the start.

Armijo alone is not enough because $\alpha \to 0$ satisfies it. The curvature condition is what makes
the pair meaningful, and it is what BFGS in lesson 87 needs in order to keep $y^{T}s > 0$.

### 2.1 The exact step on a quadratic

For $f(x) = \tfrac12 x^{T}Ax - b^{T}x$ with $A$ symmetric positive definite, the step along $-g$ is
found by minimizing the one dimensional function

$$
\varphi(\alpha) = f(x - \alpha g)
= \tfrac12 (x - \alpha g)^{T}A(x - \alpha g) - b^{T}(x - \alpha g) .
$$

Expanding,

$$
\varphi(\alpha) = f(x) - \alpha\, g^{T}(Ax - b) + \tfrac{\alpha^{2}}{2}\, g^{T}Ag
= f(x) - \alpha\, g^{T}g + \tfrac{\alpha^{2}}{2}\, g^{T}Ag ,
$$

using $g = Ax - b$. This is a quadratic in $\alpha$ with positive leading coefficient, since $A$ is
positive definite and $g \ne 0$. Setting $\varphi'(\alpha) = 0$,

$$
-g^{T}g + \alpha\, g^{T}Ag = 0
\quad\Longrightarrow\quad
\boxed{\;\alpha = \frac{g^{T}g}{g^{T}Ag}\;}
$$

Two consequences. The step is bounded by $1/\lambda_{\min} \ge \alpha \ge 1/\lambda_{\max}$, since
the Rayleigh quotient $g^{T}Ag/g^{T}g$ lies between the extreme eigenvalues. And the new gradient is
orthogonal to the old, because $\varphi'(\alpha) = -g^{T}g_{\text{new}}$ at the minimum, which is why
steepest descent zigzags at right angles.

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import quadratic

rng = np.random.default_rng(42)
print(f"{'n':>4}{'alpha from the formula':>25}{'alpha by a fine search':>25}"
      f"{'g.g_new':>12}{'1/L <= a <= 1/mu':>20}")
for n in (2, 6, 20):
    p = quadratic(condition=500.0, dimension=n)
    A, c = p["matrix"], p["minimizers"][0]
    x = p["start"].copy()
    g = A @ (x - c)
    alpha = gr.exact_step_on_a_quadratic(A, g)
    grid = np.linspace(0.0, 3.0 * alpha, 300001)
    values = np.array([p["f"](x - t * g) for t in grid])
    eig = np.linalg.eigvalsh(A)
    inside = 1.0 / eig[-1] <= alpha <= 1.0 / eig[0]
    print(f"{n:>4}{alpha:>25.12f}{grid[int(np.argmin(values))]:>25.12f}"
          f"{float(g @ (A @ (x - alpha * g - c))):>12.2e}{str(inside):>20}")
```

The formula matches a brute force search to the grid spacing, the new gradient is orthogonal to the
old to rounding, and the step lies in $[1/\lambda_{\max}, 1/\lambda_{\min}]$ at every size.

### 2.2 Kantorovich's inequality and the rate

**The inequality.** For a symmetric positive definite $A$ with eigenvalues in $[\mu, L]$ and any
$g \ne 0$,

$$
\frac{(g^{T}g)^{2}}{(g^{T}Ag)(g^{T}A^{-1}g)}
\;\ge\; \frac{4\mu L}{(\mu + L)^{2}} .
$$

**Proof sketch.** Write $g$ in the eigenbasis with weights $w_i = g_i^{2}/\lVert g\rVert^{2}$, which
form a probability distribution on the eigenvalues. Then the left side is
$1/\big(\mathbb{E}[\lambda]\,\mathbb{E}[1/\lambda]\big)$. The product
$\mathbb{E}[\lambda]\mathbb{E}[1/\lambda]$ is maximized by putting all the weight at the two extreme
eigenvalues, because $1/\lambda$ is convex so the worst distribution is supported on the endpoints.
With weight $t$ at $L$ and $1-t$ at $\mu$ the product is
$(tL + (1-t)\mu)(t/L + (1-t)/\mu)$, maximized at $t = 1/2$, giving $(\mu+L)^{2}/(4\mu L)$.

**The rate.** Measure the error in the $A$-norm, $E(x) = \tfrac12(x-x^{*})^{T}A(x-x^{*}) = f(x)-f^{*}$.
One exact step gives

$$
\frac{E(x_{k+1})}{E(x_k)} = 1 - \frac{(g^{T}g)^{2}}{(g^{T}Ag)(g^{T}A^{-1}g)} ,
$$

since $E(x_k) = \tfrac12 g^{T}A^{-1}g$ and the decrease is $\tfrac12 (g^Tg)^2/g^TAg$. Applying
Kantorovich,

$$
\frac{E(x_{k+1})}{E(x_k)} \le 1 - \frac{4\mu L}{(\mu+L)^{2}}
= \frac{(L-\mu)^{2}}{(L+\mu)^{2}} = \left(\frac{\kappa-1}{\kappa+1}\right)^{2} .
$$

Taking square roots gives the $A$-norm form quoted in 1.1, since
$\lVert x - x^{*}\rVert_{A} = \sqrt{2E}$.

```python
import numpy as np
from nalib import gradient as gr

rng = np.random.default_rng(42)
print(f"{'n':>4}{'kappa':>9}{'smallest ratio found':>23}{'Kantorovich bound':>20}"
      f"{'holds':>8}")
for n in (2, 5, 20):
    for kappa in (10.0, 1000.0):
        eig = np.concatenate([[1.0, kappa], rng.uniform(1.0, kappa, size=max(n - 2, 0))])
        frame, _ = np.linalg.qr(rng.normal(size=(n, n)))
        A = frame @ np.diag(eig[:n]) @ frame.T
        A = 0.5 * (A + A.T)
        inverse = np.linalg.inv(A)
        worst = np.inf
        for _ in range(4000):
            g = rng.normal(size=n)
            worst = min(worst, float(g @ g) ** 2 / (float(g @ A @ g) * float(g @ inverse @ g)))
        eig_true = np.linalg.eigvalsh(A)
        mu, L = float(eig_true[0]), float(eig_true[-1])
        bound = 4.0 * mu * L / (mu + L) ** 2
        print(f"{n:>4}{kappa:>9.0f}{worst:>23.9f}{bound:>20.9f}"
              f"{str(worst >= bound - 1e-12):>8}")
```

The inequality holds at every size and condition number sampled. At $n = 2$ the smallest sampled
value equals the bound to eight digits, because with only two eigenvalues every direction is a
two point distribution and random sampling finds the equal weight one. At $n = 5$ and $n = 20$ the
gap opens, because a random direction is unlikely to concentrate its weight on the two extremes.
**The bound is attained only by the worst case the proof identifies**, which is exactly the point 1.2
and 4.1 develop.

### 2.3 Affine invariance

Let $x = Sy$ with $S$ invertible, and set $\tilde f(y) = f(Sy)$. Then

$$
\nabla\tilde f(y) = S^{T}\nabla f(x) , \qquad
\nabla^{2}\tilde f(y) = S^{T}\nabla^{2}f(x)\,S .
$$

**Newton is invariant.** The Newton step in $y$ is

$$
\tilde p = -\big(S^{T}HS\big)^{-1}S^{T}g = -S^{-1}H^{-1}S^{-T}S^{T}g = -S^{-1}H^{-1}g = S^{-1}p ,
$$

so the point it moves to is $S(y + \tilde p) = x + p$. **The iterates are the same points**,
expressed in different coordinates. Newton's method has no scale to get wrong, which is why it needs
no tuning and why the full step $\alpha = 1$ is the right default.

**Steepest descent is not.** The direction in $y$ is $-S^{T}g$, and $S(-S^{T}g) = -SS^{T}g$, which
equals $-g$ only when $SS^{T} = I$, that is when $S$ is orthogonal. So steepest descent is invariant
under rotations and reflections and under nothing else: **rescaling one variable changes the path.**

That is the whole story of preconditioning in one line, and exercise 5.2 pushes it further.

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import quadratic

rng = np.random.default_rng(42)
n = 6
p = quadratic(condition=200.0, dimension=n)
A, c = p["matrix"], p["minimizers"][0]
S = np.diag(np.exp(rng.uniform(-2.0, 2.0, size=n)))
inverse = np.linalg.inv(S)

x = p["start"].copy()
y = inverse @ x
for _ in range(4):
    g = A @ (x - c)
    x = x + np.linalg.solve(A, -g)
    g_tilde = S.T @ (A @ (S @ y - c))
    H_tilde = S.T @ A @ S
    y = y + np.linalg.solve(H_tilde, -g_tilde)
print(f"Newton after four steps: ||S y - x|| = {float(np.linalg.norm(S @ y - x)):.3e}")

x = p["start"].copy()
y = inverse @ x
for _ in range(4):
    g = A @ (x - c)
    x = x - gr.exact_step_on_a_quadratic(A, g) * g
    g_tilde = S.T @ (A @ (S @ y - c))
    H_tilde = S.T @ A @ S
    y = y - gr.exact_step_on_a_quadratic(H_tilde, g_tilde) * g_tilde
print(f"descent after four steps: ||S y - x|| = {float(np.linalg.norm(S @ y - x)):.3e}")
print(f"the two are {float(np.linalg.norm(S @ y - x)):.2f} apart in a problem of scale "
      f"{float(np.linalg.norm(p['start'] - c)):.2f}")
```

Newton's two runs agree to rounding. Steepest descent's do not: after four steps the rescaled run is
in a completely different place.

### 2.4 When the Newton direction goes downhill

The direction $p$ solves $Hp = -g$, so $p = -H^{-1}g$ and the directional derivative is

$$
g^{T}p = -g^{T}H^{-1}g .
$$

If $H$ is positive definite then so is $H^{-1}$, hence $g^{T}H^{-1}g > 0$ for $g \ne 0$ and
$g^{T}p < 0$: **the direction goes downhill.**

If $H$ is not positive definite the quantity $g^{T}H^{-1}g$ can have either sign, and the direction
can go uphill. The cleanest case: take $H = \operatorname{diag}(1, -1)$ and $g = (0, 1)$. Then
$p = -H^{-1}g = (0, 1)$ and $g^{T}p = +1 > 0$. The gradient points along the negative curvature
direction, and Newton walks straight up it, because a root finder heads for the stationary point
whether that point is above or below.

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import rosenbrock

H = np.diag([1.0, -1.0])
g = np.array([0.0, 1.0])
p = np.linalg.solve(H, -g)
print(f"the constructed case: p = {np.array2string(p)}, g.p = {float(g @ p):+.1f} "
      f"which is uphill")

out = gr.the_hessian_is_not_always_positive_definite()
print(f"\nover {out['samples']} random points in the Rosenbrock box:")
print(f"  the Hessian is indefinite on {out['indefinite_in_the_box']:.1%} of them")
print(f"  and on {out['indefinite_on_the_plain_path']} of the plain run's "
      f"{out['plain_steps']} steps")

rng = np.random.default_rng(42)
problem = rosenbrock(2)
uphill, worst = 0, 0.0
for _ in range(400):
    point = rng.uniform(-2.0, 2.0, size=problem["dimension"])
    matrix = np.asarray(problem["hessian"](point), dtype=float)
    if float(np.min(np.linalg.eigvalsh(matrix))) > 0.0:
        continue
    slope = np.asarray(problem["gradient"](point), dtype=float)
    try:
        move = np.linalg.solve(matrix, -slope)
    except np.linalg.LinAlgError:
        continue
    if float(slope @ move) > 0.0:
        uphill += 1
        worst = max(worst, float(slope @ move))
print(f"  uphill Newton directions found: {uphill}, worst g.p = {worst:+.4f}")
```

The constructed case gives $g^{T}p = +1$ exactly.

On the real problem the measurement separates two things that are easy to confuse. The Hessian is
indefinite on $21.5$ per cent of the box, so an indefinite Hessian is the ordinary situation away
from the valley and not a pathology. But the Newton direction is actually **uphill** at only one of
those $86$ points, with $g^{T}p = +0.31$.

The reason is that $g^{T}p = -g^{T}H^{-1}g$ picks up the negative eigenvalue only in proportion to
how much of the gradient lies along that eigenvector. Away from a saddle the gradient is dominated
by the steep directions, where the curvature is positive and large, so the sum stays negative.

**Indefiniteness is common and an uphill step is rare**, and that is why the plain method usually
appears to work. The two variable run here never met an indefinite Hessian at all in its seven
steps. The trouble it causes is not a step that visibly goes up; it is the convergence to a saddle
of 4.2 and 5.3, which looks like success.

### 2.5 The shift

**Positive definiteness.** $H$ symmetric has eigenvalues $\lambda_1 \le \cdots \le \lambda_n$ with
orthonormal eigenvectors $v_i$. Then $H + \tau I$ has the same eigenvectors with eigenvalues
$\lambda_i + \tau$, since $(H + \tau I)v_i = (\lambda_i + \tau)v_i$. All of these are positive
exactly when $\lambda_1 + \tau > 0$, that is

$$
\tau > -\lambda_{\min} .
$$

Note this is a condition only on the smallest eigenvalue, so a $\tau$ slightly above
$\lvert\lambda_{\min}\rvert$ suffices when $H$ is indefinite, and $\tau = 0$ suffices when it is
already positive definite.

**The limit.** The shifted direction is $p(\tau) = -(H + \tau I)^{-1}g$. Factor out $\tau$:

$$
p(\tau) = -\frac{1}{\tau}\left(I + \frac{H}{\tau}\right)^{-1} g
= -\frac{1}{\tau}\left(g - \frac{Hg}{\tau} + O(\tau^{-2})\right) ,
$$

so $\tau\, p(\tau) \to -g$ as $\tau \to \infty$. The **direction** therefore tends to the steepest
descent direction while its **length** shrinks like $1/\tau$.

So $\tau$ interpolates: at $\tau = 0$ the pure Newton step, at $\tau = \infty$ an infinitesimal
steepest descent step, and in between a family that is always a descent direction once
$\tau > -\lambda_{\min}$. That is exactly the Levenberg-Marquardt family of lesson 27, and lesson
87's trust region is the same family parameterized by step length instead of by $\tau$.

```python
import numpy as np

rng = np.random.default_rng(42)
n = 5
frame, _ = np.linalg.qr(rng.normal(size=(n, n)))
H = frame @ np.diag(np.concatenate([[-1.4], rng.uniform(0.5, 8.0, size=n - 1)])) @ frame.T
H = 0.5 * (H + H.T)
g = rng.normal(size=n)
smallest = float(np.linalg.eigvalsh(H)[0])
print(f"smallest eigenvalue {smallest:.6f}, so the shift must exceed {-smallest:.6f}")
print(f"{'tau':>10}{'smallest of H + tau I':>24}{'definite':>11}"
      f"{'g.p':>12}{'angle to -g':>14}{'|p|':>12}")
for tau in (0.0, 1.0, 1.4, 1.5, 10.0, 1e3, 1e6):
    M = H + tau * np.eye(n)
    small = float(np.linalg.eigvalsh(M)[0])
    p = np.linalg.solve(M, -g)
    angle = float(np.degrees(np.arccos(
        np.clip(p @ (-g) / (np.linalg.norm(p) * np.linalg.norm(g)), -1.0, 1.0))))
    print(f"{tau:>10.1f}{small:>24.6f}{str(small > 0):>11}{float(g @ p):>12.4f}"
          f"{angle:>14.6f}{float(np.linalg.norm(p)):>12.3e}")
```

The shift becomes positive definite exactly where predicted, the directional derivative turns
negative there, the angle to $-g$ falls to zero as $\tau$ grows, and the step length falls like
$1/\tau$.

The row at $\tau = 1.4$ shows why the inequality has to be strict. There $H + \tau I$ is exactly
singular, the solve returns a step of length $2.6\times10^{15}$, and the reported slope is
meaningless. In practice the shift is found by trying a Cholesky rather than by computing
$\lambda_{\min}$, so it never lands on the boundary, which is what the lesson's `newton` does.

### 3.1 The modified Cholesky of Gill, Murray and Wright

The idea is to add to the diagonal **during** the factorization instead of restarting it. At column
$j$ the pivot is set to

$$
d_j = \max\left(\lvert c_{jj}\rvert,\; \left(\frac{\theta_j}{\beta}\right)^{2},\; \delta\right),
$$

where $\theta_j$ is the largest off-diagonal entry generated in that column and $\beta$ is chosen
from the size of the matrix. That keeps the factors bounded and produces $LDL^{T} = H + E$ with $E$
diagonal and positive semidefinite, in **one pass**.

```python
import time

import numpy as np


def gill_murray_wright(matrix):
    """One pass. The pivot is raised as needed, so no restart is required."""
    a = np.array(matrix, dtype=float)
    n = a.shape[0]
    gamma = float(np.max(np.abs(np.diag(a)))) if n else 0.0
    off = np.abs(a - np.diag(np.diag(a)))
    xi = float(np.max(off)) if n > 1 else 0.0
    delta = np.finfo(float).eps * max(gamma + xi, 1.0)
    beta = np.sqrt(max(gamma, xi / max(np.sqrt(n * n - 1.0), 1.0), np.finfo(float).eps))
    lower, pivots = np.eye(n), np.zeros(n)
    work = np.array(a, dtype=float)
    for j in range(n):
        work[j, j] = a[j, j] - float(lower[j, :j] @ (pivots[:j] * lower[j, :j]))
        theta = 0.0
        for i in range(j + 1, n):
            work[i, j] = a[i, j] - float(lower[i, :j] @ (pivots[:j] * lower[j, :j]))
            theta = max(theta, abs(work[i, j]))
        pivots[j] = max(abs(work[j, j]), (theta / beta) ** 2, delta)
        for i in range(j + 1, n):
            lower[i, j] = work[i, j] / pivots[j]
    return lower, pivots


def doubling_loop(matrix):
    """The lesson's repair: try Cholesky, double the shift, try again."""
    n = matrix.shape[0]
    scale = max(float(np.max(np.abs(np.diag(matrix)))), 1.0)
    tau, tries = 0.0, 0
    while True:
        tries += 1
        try:
            np.linalg.cholesky(matrix + tau * np.eye(n))
            return tau, tries
        except np.linalg.LinAlgError:
            tau = max(2.0 * tau, 1e-3 * scale)


rng = np.random.default_rng(42)
print(f"{'n':>5}{'GMW seconds':>14}{'doubling seconds':>19}{'GMW / doubling':>17}"
      f"{'Cholesky tries':>16}")
sizes, shifts_gmw, shifts_loop, smallest, definite = [], [], [], [], []
for n in (5, 20, 60, 120):
    frame, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eig = np.concatenate([rng.uniform(-2.0, -0.1, size=n // 3),
                          rng.uniform(0.1, 5.0, size=n - n // 3)])
    matrix = frame @ np.diag(eig) @ frame.T
    matrix = 0.5 * (matrix + matrix.T)
    repeats = max(1, 200 // n)
    clock = time.perf_counter()
    for _ in range(repeats):
        lower, pivots = gill_murray_wright(matrix)
    fast = (time.perf_counter() - clock) / repeats
    clock = time.perf_counter()
    for _ in range(repeats):
        tau, tries = doubling_loop(matrix)
    slow = (time.perf_counter() - clock) / repeats
    modified = lower @ np.diag(pivots) @ lower.T
    sizes.append(n)
    shifts_gmw.append(float(np.linalg.norm(modified - matrix, 2)))
    shifts_loop.append(tau)
    smallest.append(float(np.linalg.eigvalsh(matrix)[0]))
    definite.append(bool(np.min(np.linalg.eigvalsh(modified)) > 0.0))
    print(f"{n:>5}{fast:>14.5f}{slow:>19.5f}{fast / slow:>17.2f}{tries:>16}")

print(f"\n{'n':>5}{'GMW ||E||':>14}{'loop shift':>13}{'ratio':>10}"
      f"{'needed at least':>18}{'GMW definite':>15}")
for n, big, small, low, good in zip(sizes, shifts_gmw, shifts_loop, smallest, definite):
    print(f"{n:>5}{big:>14.3f}{small:>13.3f}{big / small:>10.1f}{-low:>18.3f}"
          f"{str(good):>15}")
```

Two results, and they point in opposite directions.

**On operation count the modified Cholesky wins by more than twenty to one.** It is one pass of about
$n^{3}/6$ operations, where the doubling loop needs $12$ or $13$ full Cholesky attempts of $n^{3}/3$
each, so about $4n^{3}$.

**On wall clock it loses by a factor of thirty five.** The doubling loop's attempts are LAPACK calls
and the one pass is an interpreted loop, and at $n = 120$ that is $0.021$ seconds against $0.0006$.
The comparison is not a fair one between algorithms, but it is the comparison a user faces, and it
is why the doubling loop is what the lesson's `newton` actually uses.

**The shift GMW adds is far larger than necessary.** The matrices here have a smallest eigenvalue of
about $-2$, so a shift of just over $2$ suffices, and the doubling loop finds $1.7$ to $3.9$. The
modified Cholesky adds $\lVert E\rVert_2 = 4.8$ at $n = 5$ but $59$ at $n = 20$, $2160$ at $n = 60$
and $2434$ at $n = 120$: six hundred to nine hundred times more than needed at the larger sizes. Its
result is positive definite, so it is doing its job, but the model it produces is so heavily damped
that the step is nearly a steepest descent step. That is a known weakness of the original algorithm:
it bounds the growth of the factors, which is a different goal from keeping the shift small.

Later variants (Schnabel-Eskow, and the eigenvalue based methods) fix the shift size at some extra
cost, which is the trade this exercise is really about.

### 3.2 Newton-CG

Solve $Hp = -g$ by conjugate gradients, stopping early on two conditions: when the residual is small
relative to $\lVert g\rVert$, and immediately when a direction of **negative curvature** is found. The
second is the important one, because it turns the Hessian's indefiniteness from a failure into
information.

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import rosenbrock, quadratic


def newton_cg(problem, start=None, tol=1e-10, rounds=500):
    x = np.asarray(problem["start"] if start is None else start, dtype=float).ravel()
    f, g, h = problem["f"], problem["gradient"], problem["hessian"]
    negative, inner_total = 0, 0
    for step in range(rounds):
        slope = np.asarray(g(x), dtype=float).ravel()
        if float(np.linalg.norm(slope)) <= tol:
            break
        matrix = np.asarray(h(x), dtype=float)
        move, residual = np.zeros_like(x), slope.copy()
        direction = -residual
        target = min(0.5, np.sqrt(float(np.linalg.norm(slope)))) * float(np.linalg.norm(slope))
        for inner in range(x.size * 2):
            inner_total += 1
            curvature = float(direction @ matrix @ direction)
            if curvature <= 0.0:                 # stop and use what we have
                negative += 1
                move = direction if inner == 0 else move
                break
            alpha = float(residual @ residual) / curvature
            move = move + alpha * direction
            fresh = residual + alpha * (matrix @ direction)
            if float(np.linalg.norm(fresh)) < target:
                residual = fresh
                break
            direction = -fresh + (float(fresh @ fresh) / float(residual @ residual)) * direction
            residual = fresh
        out = gr.backtracking(f, float(f(x)), slope, x, move)
        if not out["accepted"]:
            break
        x = x + out["step"] * move
    return {"x": x, "steps": step, "inner": inner_total,
            "negative_curvature": negative,
            "gradient": float(np.linalg.norm(g(x)))}


print("on Rosenbrock, against the plain method")
print(f"{'n':>4}{'NCG steps':>11}{'inner':>8}{'neg curv':>10}{'NCG distance':>15}"
      f"{'Newton steps':>14}{'Newton distance':>17}")
for n in (2, 4, 5, 6, 7, 10):
    problem = rosenbrock(n)
    fast = newton_cg(problem)
    plain = gr.newton(problem)
    print(f"{n:>4}{fast['steps']:>11}{fast['inner']:>8}{fast['negative_curvature']:>10}"
          f"{float(np.linalg.norm(fast['x'] - 1.0)):>15.3e}{plain['steps']:>14}"
          f"{float(np.linalg.norm(plain['x'] - 1.0)):>17.3e}")

print("\non a well conditioned quadratic, where the exact solve is available")
for n in (50, 200):
    problem = quadratic(condition=1e4, dimension=n)
    fast = newton_cg(problem)
    plain = gr.newton(problem)
    print(f"  n = {n:>3}: Newton-CG {fast['steps'] + 1} outer steps and {fast['inner']} "
          f"matrix products; plain Newton {plain['steps'] + 1} solves")
```

**Negative curvature is detected constantly**, on $5$ to $64$ inner iterations per run, which
confirms that the Rosenbrock Hessian is indefinite along much of the path and not only near the
saddle.

**It rescues exactly the runs the plain method got wrong**, and spoils one it got right. At
$n = 4, 5, 7$ the plain method stopped at distance $2.15, 2.02, 2.00$ and Newton-CG reaches $0$,
$4\times10^{-13}$ and $4\times10^{-13}$. At $n = 6$ the plain method reached the answer and
Newton-CG stops at $1.999$.

That is the same pattern as the lesson's line search safeguards, which rescued $n = 4$ and spoiled
$n = 6$ and $n = 10$. **A safeguard changes which starting points succeed, it does not increase how
many.**

**On a dense quadratic it is a bad trade.** Newton should finish a quadratic in one solve, and plain
Newton does, in three. Newton-CG takes twenty outer steps and $484$ to $1087$ matrix products,
because the forcing term stops the inner solve long before it is accurate. The early stopping that
makes each step cheap turns a one step method into a twenty step one.

Its case is entirely the one where $H$ is never formed: a product $Hv$ is available by automatic
differentiation at the cost of one gradient, the matrix would not fit in memory, and there is no
factorization to compete with. That is the machine learning setting of lesson 95.

### 3.3 The Barzilai-Borwein step

The step is chosen so that $\alpha^{-1}I$ satisfies the secant condition of lesson 87 as nearly as
possible:

$$
\alpha_{k}^{\text{BB1}} = \frac{s^{T}s}{s^{T}y} , \qquad
\alpha_{k}^{\text{BB2}} = \frac{s^{T}y}{y^{T}y} ,
$$

with $s = x_k - x_{k-1}$ and $y = g_k - g_{k-1}$. There is **no line search**: the step is used as
computed.

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import quadratic, rosenbrock


def barzilai_borwein(problem, variant="bb1", tol=1e-10, rounds=200000):
    x = np.asarray(problem["start"], dtype=float).ravel()
    g = problem["gradient"]
    slope = np.asarray(g(x), dtype=float).ravel()
    step = 1.0 / max(float(np.linalg.norm(slope)), 1e-16)
    monotone, previous = True, problem["f"](x)
    for k in range(rounds):
        if float(np.linalg.norm(slope)) <= tol:
            break
        moved = x - step * slope
        fresh = np.asarray(g(moved), dtype=float).ravel()
        s, y = moved - x, fresh - slope
        bottom = float(s @ y)
        if bottom <= 0.0:
            step = 1e-4
        elif variant == "bb1":
            step = float(s @ s) / bottom
        else:
            step = bottom / float(y @ y)
        x, slope = moved, fresh
        value = problem["f"](x)
        if value > previous + 1e-14:
            monotone = False
        previous = value
    return {"x": x, "steps": k, "monotone": monotone}


print("on quadratics, against the exact line search")
print(f"{'n':>4}{'kappa':>9}{'exact':>10}{'BB1':>9}{'BB2':>9}{'speedup':>10}"
      f"{'monotone':>11}")
for n in (10, 50):
    for kappa in (1e2, 1e4, 1e6):
        problem = quadratic(condition=kappa, dimension=n)
        exact = gr.steepest_descent(problem, search="exact")
        first = barzilai_borwein(problem, "bb1")
        second = barzilai_borwein(problem, "bb2")
        print(f"{n:>4}{kappa:>9.0e}{exact['steps']:>10}{first['steps']:>9}"
              f"{second['steps']:>9}{exact['steps'] / max(first['steps'], 1):>10.1f}"
              f"{str(first['monotone']):>11}")

print("\non Rosenbrock, against backtracking steepest descent")
print(f"{'n':>4}{'BB steps':>11}{'BB distance':>15}{'descent steps':>16}"
      f"{'descent distance':>19}")
for n in (2, 5, 10):
    problem = rosenbrock(n)
    fast = barzilai_borwein(problem, "bb1", rounds=100000)
    slow = gr.steepest_descent(problem, max_steps=100000)
    print(f"{n:>4}{fast['steps']:>11}{float(np.linalg.norm(fast['x'] - 1.0)):>15.3e}"
          f"{slow['steps']:>16}{float(np.linalg.norm(slow['x'] - 1.0)):>19.3e}")
```

**On quadratics it is spectacular.** At $\kappa = 10^{4}$ it takes $1194$ steps where the exact line
search takes $112875$, a factor of $95$. BB1 beats BB2 consistently. And it is **never monotone**,
at any size or condition number.

**On Rosenbrock it fails completely.** It ends at a distance of $1.4$ to $2.0$ where ordinary
backtracking steepest descent reaches $1.5\times10^{-10}$. Without a line search there is no
guarantee at all off a quadratic, and the same aggressive steps that beat the exact search on a
quadratic walk out of the valley here.

The standard repair is a line search that is deliberately **nonmonotone**: accept a step if it beats
the worst of the last few values, not the last one.

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import rosenbrock


def barzilai_borwein_safeguarded(problem, memory=10, tol=1e-10, rounds=100000):
    x = np.asarray(problem["start"], dtype=float).ravel()
    f, g = problem["f"], problem["gradient"]
    slope = np.asarray(g(x), dtype=float).ravel()
    step = 1.0 / max(float(np.linalg.norm(slope)), 1e-16)
    recent = [f(x)]
    for k in range(rounds):
        if float(np.linalg.norm(slope)) <= tol:
            break
        trial, bar = step, max(recent[-memory:])
        for _ in range(60):
            moved = x - trial * slope
            if f(moved) <= bar - 1e-4 * trial * float(slope @ slope):
                break
            trial *= 0.5
        fresh = np.asarray(g(moved), dtype=float).ravel()
        s, y = moved - x, fresh - slope
        bottom = float(s @ y)
        step = float(s @ s) / bottom if bottom > 0.0 else 1e-4
        x, slope = moved, fresh
        recent.append(f(x))
    return {"x": x, "steps": k}


print(f"{'n':>4}{'safeguarded steps':>20}{'distance':>13}{'descent steps':>16}"
      f"{'speedup':>10}")
for n in (2, 5, 10):
    problem = rosenbrock(n)
    fast = barzilai_borwein_safeguarded(problem)
    slow = gr.steepest_descent(problem, max_steps=100000)
    print(f"{n:>4}{fast['steps']:>20}{float(np.linalg.norm(fast['x'] - 1.0)):>13.3e}"
          f"{slow['steps']:>16}{slow['steps'] / max(fast['steps'], 1):>10.0f}")
```

The nonmonotone safeguard fixes it entirely and keeps the speed: $45$ steps against $25115$ at two
variables, $2462$ against $36441$ at ten. **The safeguard has to be nonmonotone**, because a monotone
one would reject exactly the steps that make the method fast, and the measurement above shows the
method is never monotone.

That combination, an aggressive step with a permissive safeguard, is the shape of most modern first
order methods, and it is worth noticing that lesson 89's momentum methods are also not monotone for
the same reason.

### 3.4 The gradient check

Central differences give $O(h^{2})$ accuracy, so the optimal step is $\varepsilon^{1/3}$ rather than
lesson 84's $\sqrt\varepsilon$, and the attainable relative accuracy is about $\varepsilon^{2/3}$.

```python
import numpy as np
from nalib.optimize import quadratic, rosenbrock, himmelblau, rastrigin


def gradient_check(f, g, where, step=None):
    """Central differences against the claimed gradient, scaled so the answer is relative."""
    x = np.asarray(where, dtype=float).ravel()
    claimed = np.asarray(g(x), dtype=float).ravel()
    if step is None:
        step = np.cbrt(np.finfo(float).eps) * max(float(np.max(np.abs(x))), 1.0)
    measured = np.empty_like(x)
    for k in range(x.size):
        up, down = x.copy(), x.copy()
        up[k] += step
        down[k] -= step
        measured[k] = (f(up) - f(down)) / (2.0 * step)
    scale = max(float(np.linalg.norm(claimed)), float(np.linalg.norm(measured)), 1.0)
    worst = float(np.max(np.abs(claimed - measured))) / scale
    return {"relative": worst, "passes": worst < 1e-6,
            "worst_index": int(np.argmax(np.abs(claimed - measured)))}


print("every correct gradient in the library")
for build in (lambda: quadratic(dimension=6), lambda: rosenbrock(6),
              himmelblau, lambda: rastrigin(6)):
    problem = build()
    out = gradient_check(problem["f"], problem["gradient"], problem["start"])
    print(f"  {problem['name']:<42} relative {out['relative']:.2e}  "
          f"passes {out['passes']}")

problem = rosenbrock(6)
print("\nthree deliberate faults")
faults = (
    ("one sign flipped",
     lambda x: np.concatenate([[-problem["gradient"](x)[0]], problem["gradient"](x)[1:]])),
    ("one entry doubled",
     lambda x: np.concatenate([problem["gradient"](x)[:3], [2.0 * problem["gradient"](x)[3]],
                               problem["gradient"](x)[4:]])),
    ("one term of the sum dropped",
     lambda x: problem["gradient"](x) - np.concatenate(
         [[200.0 * (np.asarray(x, dtype=float)[1] - np.asarray(x, dtype=float)[0] ** 2)],
          np.zeros(np.asarray(x).size - 1)])),
)
for name, wrong in faults:
    out = gradient_check(problem["f"], wrong, problem["start"])
    print(f"  {name:<30} relative {out['relative']:.2e}  entry {out['worst_index']}  "
          f"caught {not out['passes']}")

print("\nthe step matters, and the best one is near the cube root of eps")
for power in range(-12, -1, 2):
    h = 10.0 ** power
    out = gradient_check(problem["f"], problem["gradient"], problem["start"], step=h)
    print(f"  h = {h:.0e}: relative {out['relative']:.2e}")
print(f"  the cube root of eps is {float(np.cbrt(np.finfo(float).eps)):.2e}")
```

Every correct gradient passes at $6\times10^{-12}$ to $5\times10^{-10}$, and all three faults are
caught with the right entry identified. The dropped term is the mildest at $6\times10^{-2}$ and is
still four orders of magnitude above the pass threshold.

The step sweep has the U shape lesson 5 predicted, with its minimum at $h = 10^{-6}$ against the
predicted $\varepsilon^{1/3} = 6.1\times10^{-6}$, and it degrades by four orders of magnitude in
each direction. **A gradient check that fails might be reporting a bad step rather than a bad
gradient**, so the sweep is part of the check and not an optional extra.

This is the single most valuable ten lines in the whole part. A wrong gradient does not crash: it
produces an optimizer that converges slowly to the wrong place, which looks exactly like a hard
problem.

### 3.5 The valley

The Rosenbrock valley floor is the parabola $x_2 = x_1^{2}$, so $r = x_2 - x_1^{2}$ is a signed
distance from it and a **crossing** is a sign change in $r$.

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import rosenbrock

problem = rosenbrock(2)
start = np.array([-1.2, 1.0])


def path_of(method, rounds=100000):
    x = start.copy()
    trail = [x.copy()]
    f, g, h = problem["f"], problem["gradient"], problem["hessian"]
    for _ in range(rounds):
        slope = np.asarray(g(x), dtype=float).ravel()
        if float(np.linalg.norm(slope)) <= 1e-8:
            break
        if method == "descent":
            out = gr.backtracking(f, float(f(x)), slope, x, -slope)
            if not out["accepted"]:
                break
            x = x - out["step"] * slope
        else:
            x = x + np.linalg.solve(np.asarray(h(x), dtype=float), -slope)
        trail.append(x.copy())
    return np.asarray(trail)


for name in ("descent", "newton"):
    trail = path_of(name)
    off = trail[:, 1] - trail[:, 0] ** 2
    crossings = int(np.sum(np.sign(off[:-1]) * np.sign(off[1:]) < 0))
    length = float(np.sum(np.linalg.norm(np.diff(trail, axis=0), axis=1)))
    direct = float(np.linalg.norm(trail[-1] - trail[0]))
    print(f"{name:>9}: {len(trail) - 1:>6} steps, {crossings:>5} crossings "
          f"({crossings / max(len(trail) - 1, 1):>5.1%} of steps), "
          f"path {length:>7.3f} against a straight line of {direct:.3f} "
          f"(ratio {length / direct:.2f})")

print("\nthe Newton path in full")
print(f"{'step':>5}{'x1':>11}{'x2':>11}{'f':>13}{'x2 - x1^2':>13}{'step length':>14}")
x = start.copy()
print(f"{0:>5}{x[0]:>11.5f}{x[1]:>11.5f}{problem['f'](x):>13.4e}"
      f"{x[1] - x[0] ** 2:>13.4f}{0.0:>14.4f}")
for k in range(1, 8):
    move = np.linalg.solve(np.asarray(problem["hessian"](x), dtype=float),
                           -np.asarray(problem["gradient"](x), dtype=float))
    x = x + move
    print(f"{k:>5}{x[0]:>11.5f}{x[1]:>11.5f}{problem['f'](x):>13.4e}"
          f"{x[1] - x[0] ** 2:>13.4f}{float(np.linalg.norm(move)):>14.4f}")
```

The counts are stark. Steepest descent crosses the valley floor $6930$ times in $19435$ steps, on
**more than a third of its steps**. Newton crosses it **zero** times in six.

But the second table complicates the picture in a way worth seeing. Newton's path is $9.578$ long
against a straight line distance of $2.200$, a ratio of $4.35$; steepest descent's is only $1.43$.
Newton does not follow the valley. It takes step 1 onto the floor, then a step of length $4.95$ that
leaves the valley entirely and raises $f$ from $4.73$ to **$1412$**, then a step of $3.76$ straight
back to the floor, and then converges quadratically.

So neither method follows the valley. **Steepest descent takes twenty thousand small steps that
alternate sides, and Newton takes six huge ones, one of which is a spectacular excursion it recovers
from in a single step.** That excursion is the "not monotone" of the lesson's title made concrete,
and it is exactly what the trust region of lesson 87 exists to prevent.

### 4.1 Where the bound becomes sharp

```python
import numpy as np
from nalib.gradient import _quadratic_rate

print(f"{'kappa':>9}{'n':>5}{'measured':>12}{'bound':>12}{'fraction':>11}{'steps':>9}")
first = {}
for kappa in (10.0, 100.0, 1000.0):
    for n in range(2, 15):
        out = _quadratic_rate(n, kappa)
        fraction = out["measured_rate"] / out["bound"]
        if n <= 6 or abs(fraction - 1.0) < 0.01:
            print(f"{kappa:>9.0f}{n:>5}{out['measured_rate']:>12.6f}{out['bound']:>12.6f}"
                  f"{fraction:>11.5f}{out['steps']:>9}")
        if abs(fraction - 1.0) < 0.01 and kappa not in first:
            first[kappa] = n
            break
    print()
print("the smallest dimension within one per cent:", first)
```

The bound is reached at $n = 4, 4, 3$ for $\kappa = 10, 100, 1000$. Three observations.

**Two dimensions is always far off**, at $0.774$, $0.714$ and $0.709$ of the bound. Those numbers
approach $1/\sqrt2 = 0.7071$ as $\kappa$ grows, exactly as the calculation in 1.2 predicts, and the
approach is monotone.

**Three dimensions is nearly there**, at $0.98$ to $0.99$. So the jump from a toy to a
representative problem happens between two and three variables and is finished by four.

**A harder problem needs fewer dimensions.** At $\kappa = 1000$ three variables suffice, at
$\kappa = 10$ four are needed. A more spread spectrum makes it easier for the iteration to keep
energy in the intermediate directions.

The practical rule: **do not test a first order method in two dimensions.** It will look about $30$
per cent better than it is, and the discrepancy grows with the conditioning, which is the regime that
matters.

### 4.2 How often plain Newton fails, by dimension

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import rosenbrock, classify

rng = np.random.default_rng(42)
trials = 100
print(f"{'n':>4}{'global min':>13}{'other min':>12}{'saddle':>9}{'no stop':>10}"
      f"{'saddle rate':>13}")
for n in range(2, 11):
    problem = rosenbrock(n)
    tally = {"right": 0, "wrong": 0, "saddle": 0, "none": 0}
    for _ in range(trials):
        start = np.ones(n) + rng.uniform(-2.0, 2.0, size=n)
        out = gr.newton(problem, start=start, max_steps=300)
        if not out["converged"]:
            tally["none"] += 1
            continue
        kind = classify(np.asarray(problem["hessian"](out["x"]), dtype=float))
        if float(np.linalg.norm(out["x"] - 1.0)) < 1e-6:
            tally["right"] += 1
        elif kind == "minimum":
            tally["wrong"] += 1
        else:
            tally["saddle"] += 1
    print(f"{n:>4}{tally['right']:>13}{tally['wrong']:>12}{tally['saddle']:>9}"
          f"{tally['none']:>10}{tally['saddle'] / trials:>13.2f}")
```

The answer changes the emphasis of the whole section.

**Below four variables it never fails.** At $n = 2$ and $n = 3$ all $100$ starts reach the global
minimum, because Rosenbrock has only one minimum there.

**From four variables up it fails on $14$ to $27$ per cent of starts**, with no trend in the
dimension.

**But the saddle is the rare failure.** Only $1$ to $2$ per cent of starts end at a saddle. The
common failure, $7$ to $21$ per cent, is a **second local minimum**, and $5$ to $8$ per cent never
stop within $300$ steps.

That reorders the priorities. The lesson stressed that a converged Newton run can be at a saddle,
which is true and is worth the Hessian check, but the measurement says the Hessian check will almost
always come back "minimum" and the answer will still be wrong. A saddle is detectable; **a wrong
local minimum is not**, because it passes every test a local method can apply. The only remedies are
global: several starts, or knowledge of the problem.

The four variable Rosenbrock is a good case in point, and 5.3 takes it apart.

### 4.3 The Armijo constant

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import rosenbrock, quadratic


def run(problem, c1, rounds=200000):
    x = np.asarray(problem["start"], dtype=float).ravel()
    f, g = problem["f"], problem["gradient"]
    calls = 0
    for k in range(rounds):
        slope = np.asarray(g(x), dtype=float).ravel()
        if float(np.linalg.norm(slope)) <= 1e-8:
            break
        out = gr.backtracking(f, float(f(x)), slope, x, -slope, wolfe_c1=c1)
        calls += out["calls"]
        if not out["accepted"]:
            break
        x = x - out["step"] * slope
    return k, calls, float(np.linalg.norm(g(x)))


for label, problem in (("Rosenbrock, 5 variables", rosenbrock(5)),
                       ("quadratic, kappa 1000, 10 variables",
                        quadratic(condition=1000.0, dimension=10))):
    print(label)
    print(f"{'c1':>10}{'iterations':>13}{'evaluations':>14}{'per step':>11}"
          f"{'final gradient':>17}")
    for power in range(-8, 1):
        c1 = 0.5 if power >= 0 else 10.0 ** power
        steps, calls, size = run(problem, c1)
        print(f"{c1:>10.0e}{steps:>13}{calls:>14}{calls / max(steps, 1):>11.3f}"
              f"{size:>17.2e}")
    print()
```

**Over six decades of $c_1$ nothing happens.** From $10^{-8}$ to $10^{-3}$ the iteration counts and
the evaluation counts are **bit identical**, on both problems.

**At $c_1 = 0.5$ everything happens.** On Rosenbrock the step count falls from $25693$ to $15729$.
On the quadratic it falls from a run that never converges in $200000$ steps to one that converges in
$2139$, a factor of at least $93$.

The reason is in which power of one half gets accepted.

```python
import collections

import numpy as np
from nalib import gradient as gr
from nalib.optimize import quadratic

problem = quadratic(condition=1000.0, dimension=10)
biggest = float(np.max(np.linalg.eigvalsh(problem["matrix"])))
print(f"largest eigenvalue L = {biggest:.1f}, so 1/L = {1.0 / biggest:.3e} and "
      f"2/L = {2.0 / biggest:.3e}")
for c1 in (1e-4, 0.5):
    x = problem["start"].copy()
    steps = []
    budget = 200000
    for k in range(budget):
        slope = problem["gradient"](x)
        if float(np.linalg.norm(slope)) <= 1e-8:
            break
        out = gr.backtracking(problem["f"], float(problem["f"](x)), slope, x, -slope,
                              wolfe_c1=c1)
        steps.append(out["step"])
        x = x - out["step"] * slope
    tally = collections.Counter(round(s * biggest, 4) for s in steps)
    order = sorted(tally.items(), key=lambda pair: -pair[1])
    print(f"  c1 = {c1}: {k} steps, accepted step times L: {order}")
```

With $c_1 = 10^{-4}$ only **two** step sizes are ever accepted: $1.953/L$ on $96$ per cent of steps
and $3.906/L$ on the rest. Both are at or above $2/L$, which is the largest step that can reduce the
error in the stiffest eigendirection. The loose Armijo condition lets through the longest step that
decreases $f$ at all, and that step is the maximum zigzag.

With $c_1 = 0.5$ eleven different step sizes appear. The same $1.953/L$ still dominates, but $228$
steps take $0.977/L$ and $114$ take $0.488/L$, and a long tail runs all the way up to $500/L$ where
the gradient has almost no content in the stiff directions. Those occasional short steps are what
break the zigzag, and the run finishes.

So the finding is not that $c_1$ is a tuning dial with a smooth effect. **It has no effect at all
until it is large enough to reject a power of two, and then its effect is enormous.** The
conventional $10^{-4}$ is chosen so that the line search never interferes with a Newton direction,
where $\alpha = 1$ is the right answer. For steepest descent that same choice means it never
interferes with anything, including the behaviour you would want it to prevent.

### 5.1 Why $\kappa$ and not $\sqrt\kappa$

Both methods build a polynomial in $A$ applied to the initial error. After $k$ steps,

$$
e_k = p_k(A)\, e_0 , \qquad p_k(0) = 1 ,
$$

because every step subtracts a multiple of a gradient and every gradient is $A$ times an error. So
$e_k$ lies in the Krylov space $e_0 + \operatorname{span}\{Ae_0, \dots, A^{k}e_0\}$, and the error is
controlled by how small $p_k$ can be made on the spectrum $[\mu, L]$.

**Conjugate gradients minimizes over all such polynomials.** By construction it produces the $p_k$ of
degree $k$ with $p_k(0)=1$ minimizing $\lVert p_k(A)e_0\rVert_A$, so its error is bounded by the
best polynomial, which is the shifted and scaled Chebyshev polynomial:

$$
\min_{p_k(0)=1}\;\max_{\lambda\in[\mu,L]} \lvert p_k(\lambda)\rvert
= \frac{1}{T_k\!\left(\frac{L+\mu}{L-\mu}\right)}
\;\approx\; 2\left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^{k} .
$$

The $\sqrt\kappa$ is Chebyshev's, from lesson 42: it is the polynomial that is smallest on an
interval away from a prescribed value, and its cost grows like the square root of the interval's
aspect ratio.

**Steepest descent builds only the product of independent one step polynomials.**

$$
p_k(\lambda) = \prod_{j=1}^{k}(1 - \alpha_j\lambda) ,
$$

with each $\alpha_j$ chosen greedily, without reference to the ones before or after. That is a much
smaller family: it can only place its $k$ roots at $1/\alpha_j$, one per step, chosen one at a time.
A greedy choice puts each root where it helps most **now**, which is near the current dominant
eigenvalue, and the next step then undoes part of it. The resulting factor per step is
$(\kappa-1)/(\kappa+1)$, giving $\kappa$ rather than $\sqrt\kappa$ steps for a fixed reduction.

The one line version: **conjugate gradients chooses all $k$ roots together and steepest descent
chooses them one at a time.** Lesson 39 measured the resulting difference at 30-fold on a Poisson
matrix, and the same gap appears in lesson 89 as the difference between plain gradient descent and
Nesterov's method, for exactly the same reason.

### 5.2 Preconditioning

**Steepest descent on $Sx$ is preconditioned steepest descent on $x$.** With $x = Sy$ and
$\tilde f(y) = f(Sy)$, section 2.3 gave $\nabla\tilde f = S^{T}g$. So the step in $y$ is
$y \leftarrow y - \alpha S^{T}g$, and in the original variables

$$
x \leftarrow S\left(S^{-1}x - \alpha S^{T}g\right) = x - \alpha\, SS^{T} g = x - \alpha M g ,
$$

with $M = SS^{T}$. That **is** preconditioned steepest descent, with preconditioner $M$. Any
symmetric positive definite $M$ arises this way, from $S = M^{1/2}$.

**The $S$ that makes $\kappa = 1$.** The Hessian in the new variables is $S^{T}AS$. Choosing
$S = A^{-1/2}$ gives $S^{T}AS = I$, whose condition number is $1$, and the method converges in one
step from any start. Equivalently $M = SS^{T} = A^{-1}$, so the step is $x \leftarrow x - \alpha A^{-1}g$:
**perfect preconditioning is Newton's method.**

**Why it is not available.** It needs $A^{-1/2}$, or equivalently $A^{-1}$, which is the thing the
whole discussion is trying to avoid computing. So the exercise closes a circle: preconditioning does
not give you something Newton's method does not; it gives you a way to spend a small amount on an
approximation of $A^{-1}$ and get part of the benefit. The question is what approximation is cheap
enough and good enough.

```python
import numpy as np
from nalib.optimize import quadratic


def preconditioned(problem, matrix, tol=1e-10, rounds=200000):
    A, centre = problem["matrix"], problem["minimizers"][0]
    x = problem["start"].copy()
    M = np.asarray(matrix, dtype=float)
    for k in range(rounds):
        slope = A @ (x - centre)
        if float(np.linalg.norm(slope)) <= tol:
            break
        direction = M @ slope
        alpha = float(slope @ direction) / float(direction @ A @ direction)
        x = x - alpha * direction
    return k


print(f"{'n':>4}{'kappa':>9}{'no preconditioner':>20}{'Jacobi':>10}{'exact':>9}"
      f"{'kappa after Jacobi':>21}")
for n in (10, 30):
    for kappa in (1e2, 1e4):
        problem = quadratic(condition=kappa, dimension=n)
        A = problem["matrix"]
        values, vectors = np.linalg.eigh(A)
        exact = vectors @ np.diag(1.0 / values) @ vectors.T
        root = np.diag(1.0 / np.sqrt(np.diag(A)))
        print(f"{n:>4}{kappa:>9.0e}{preconditioned(problem, np.eye(n)):>20}"
              f"{preconditioned(problem, np.diag(1.0 / np.diag(A))):>10}"
              f"{preconditioned(problem, exact):>9}"
              f"{float(np.linalg.cond(root @ A @ root)):>21.1f}")
```

The exact preconditioner converges in one or two steps, as it must, since it is Newton's method.

**Jacobi buys nothing.** $946$ steps against $1037$, and at $\kappa = 10^{4}$ it is actually
**worse**, $121563$ against $112875$. The reason is visible in the last column: the condition number
after Jacobi scaling is $96$ instead of $100$, and $10903$ instead of $10^{4}$. The matrix here is a
random rotation of a diagonal one, so its own diagonal carries almost no information about its
eigenvalues.

That is the general case, and it is why preconditioning is not a generic technique. Jacobi works
when the matrix is diagonally dominant, incomplete Cholesky when it is sparse with a structure worth
imitating, multigrid when it comes from a differential operator: **every useful preconditioner comes
from knowing where the matrix came from.** Lesson 39 made the same point for conjugate gradients.

### 5.3 The saddle

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import rosenbrock, classify

problem = rosenbrock(4)
out = gr.newton(problem)
where = out["x"]
H = np.asarray(problem["hessian"](where), dtype=float)
values, vectors = np.linalg.eigh(H)
print("plain Newton from the standard start stops at")
print(f"  x = {np.array2string(where, precision=8)}")
print(f"  f = {problem['f'](where):.8f}, gradient "
      f"{float(np.linalg.norm(problem['gradient'](where))):.2e}")
print(f"  classify says {classify(H)}, eigenvalues "
      f"{np.array2string(values, precision=4)}")
print(f"  negative eigenvalues: {int(np.sum(values < 0))}")

direction = vectors[:, 0]
grid = np.linspace(-0.3, 0.3, 6001)
along = np.array([problem["f"](where + t * direction) for t in grid])
best = int(np.argmin(along))
print(f"\nalong the negative curvature direction f is smallest at t = {grid[best]:+.4f}")
print(f"  the drop is {problem['f'](where) - along[best]:.2e}, where the quadratic "
      f"model predicts {abs(0.5 * values[0] * grid[best] ** 2):.2e}")

print("\nrestarting Newton after a step of t along it")
for t in (0.02, 0.06, 0.08, 0.09, 0.10, 0.15):
    again = gr.newton(problem, start=where + t * direction)
    print(f"  t = {t:.2f}: distance to (1,1,1,1) is "
          f"{float(np.linalg.norm(again['x'] - 1.0)):.2e}")

other = gr.newton(problem, start=where - 0.1 * direction)["x"]
second = np.asarray(problem["hessian"](other), dtype=float)
print(f"\nthe point it lands on going the other way: "
      f"{np.array2string(other, precision=6)}")
print(f"  f = {problem['f'](other):.8f}, {classify(second)}, eigenvalues "
      f"{np.array2string(np.linalg.eigvalsh(second), precision=4)}")
```

**The saddle.** It is at $(-0.65612, 0.44312, 0.20431, 0.04174)$ with $f = 3.70824$ and a gradient of
$1.5\times10^{-14}$. The Hessian's eigenvalues are $\{-0.5687,\; 168.9,\; 310.4,\; 654.0\}$: exactly
one negative, so a saddle with a one dimensional unstable manifold, and `classify` says so.

**The negative direction is very weak.** The ratio of the largest eigenvalue to the size of the
negative one is $1150$, so the descent available along it is tiny. The best point along that
direction is at $t = 0.078$ and the drop is $9.7\times10^{-4}$, against the $1.7\times10^{-3}$ the
quadratic model predicts. The cubic terms have eaten $44$ per cent of the predicted gain by the time
the model is worth using.

**Stepping off does not reliably escape.** A step of $t = 0.02$ or $0.06$ comes **straight back to
the saddle**, distance $2.15$, because Newton is attracted to any stationary point and the
perturbation is inside its basin. Steps of $t = 0.08$, $0.09$ and $0.15$ do leave, but land on a
**different local minimum** at $(-0.7757, 0.6131, 0.3821, 0.1460)$, distance $2.10$, whose Hessian
has eigenvalues $\{0.380, 191, 440, 866\}$, all positive. Only $t = 0.10$ reaches the global
minimum. There is a narrow window, it is not the largest step or the smallest, and nothing in the
problem tells you where it is.

**What a method would have to do.** Three things, and it can only guarantee the first two.

Detect the negative eigenvalue, which costs one Cholesky attempt or one eigenvalue computation.

Move along it far enough to leave the saddle's basin, which means a step of order $0.1$ here and not
the $10^{-3}$ a cautious method would take. Lesson 87's trust region does exactly this: when the
model is indefinite it goes to the boundary, which is a long step by construction, and that is why it
escapes this saddle with no shift at all.

Accept that where it lands is not controlled. The two attractors here are a genuine local minimum and
the global one, both stationary, both with positive definite Hessians. **No local method can prefer
one over the other**, and the measurement in 4.2 says the local minimum is by far the more likely
outcome. That is not a defect in the saddle escaping machinery; it is the boundary between local and
global optimization, and lesson 87's Lennard-Jones clusters cross it in the only way available, by
starting many times.

---

## Lesson 87, Quasi-Newton and Trust Region Methods

### 1.1 The secant condition

After a step $s = x_{k+1} - x_k$ with gradient change $y = g_{k+1} - g_k$, the new Hessian
approximation is asked to reproduce that observation:

$$
B_{k+1}\, s = y , \qquad\text{equivalently}\qquad H_{k+1}\, y = s
$$

for the inverse $H = B^{-1}$. It is the multidimensional version of the secant slope of lesson 11:
the only curvature information a gradient step actually delivers.

**It does not determine the update**, by a straight count. $B_{k+1}$ is symmetric, so it has
$n(n+1)/2$ unknowns; the secant condition supplies $n$ equations. At $n = 10$ that is $10$ equations
for $55$ unknowns.

So a whole family of updates satisfies it, and every named method is one choice from that family
fixed by an extra requirement. BFGS and DFP both ask for the closest matrix to the previous one in a
weighted Frobenius norm, differing only in which of $B$ and $H$ is held closest. Symmetric rank one
asks for the smallest rank. The choices give genuinely different methods, and exercises 3.1 and 3.2
measure how different.

### 1.2 Why $y^{T}s > 0$

**It is needed** because it is exactly the condition for the update to keep $H$ positive definite.
Multiply the secant condition $B_{k+1}s = y$ on the left by $s^{T}$:

$$
s^{T}B_{k+1}s = s^{T}y .
$$

The left side is positive for any positive definite $B_{k+1}$ and $s \ne 0$. So if $y^{T}s \le 0$,
**no positive definite matrix satisfies the secant condition at all**, and the question of which
update to use does not arise. Without positive definiteness the direction $-Hg$ need not go downhill,
which is lesson 86's problem returning.

**It comes from the Wolfe curvature condition.** That condition asks

$$
g_{k+1}^{T}p \;\ge\; c_2\, g_k^{T}p , \qquad c_2 < 1 ,
$$

with $s = \alpha p$ and $\alpha > 0$. Subtracting $g_k^{T}p$ from both sides,

$$
y^{T}p = (g_{k+1} - g_k)^{T}p \;\ge\; (c_2 - 1)\, g_k^{T}p \;>\; 0 ,
$$

since $c_2 - 1 < 0$ and $g_k^{T}p < 0$ for a descent direction. Multiplying by $\alpha > 0$ gives
$y^{T}s > 0$.

So the line search is not an optional refinement for BFGS; it is what makes the update legal. That is
why the lesson's `bfgs` uses `strong_wolfe` and not backtracking, and why exercise 4.1 finds that
$c_2$ changes the answer.

### 1.3 Superlinear against quadratic

**Quadratic** means

$$
\lVert e_{k+1}\rVert \le C \lVert e_k\rVert^{2} ,
$$

so the number of correct digits doubles each step, with a fixed constant $C$.

**Superlinear** means

$$
\frac{\lVert e_{k+1}\rVert}{\lVert e_k\rVert} \longrightarrow 0 ,
$$

so the ratio shrinks, but at no prescribed speed. Every quadratically convergent sequence is
superlinear; the converse is false, and a superlinear sequence can have ratios falling as slowly as
$1/\log k$.

**What has to be measured to tell them apart.** Not the ratio: a small ratio is consistent with both.
The distinguishing quantity is $\lVert e_{k+1}\rVert/\lVert e_k\rVert^{2}$. Quadratic convergence
holds that bounded; superlinear but not quadratic convergence lets it grow without bound.

That is a limit, so it needs many steps in the asymptotic regime, and the lesson measures that BFGS
does not provide them: the ratios fall below $1$ but whether they tend to $0$ is not resolvable
before rounding takes over. Exercise 5.1 says why that is inherent and not a defect of the
measurement.

### 1.4 The trust region and an indefinite Hessian

**It does not need the Hessian to be positive definite, because it never inverts it.** The subproblem
is

$$
\min_{\lVert p\rVert \le \Delta}\; m(p) = g^{T}p + \tfrac12 p^{T}Bp ,
$$

which has a solution for **any** symmetric $B$, because a continuous function on a compact ball
attains its minimum. When $B$ is positive definite and the unconstrained minimum fits inside, the
answer is the Newton step. When $B$ is indefinite the model has no unconstrained minimum, and the
answer is on the boundary, in a direction that exploits the negative curvature. Exercise 5.2 works
that out.

The practical version in the lesson is the dogleg, which falls back to the Cauchy point when the
model is not convex. That is enough to escape a saddle with no shift at all, which the lesson
measures at four variables where lesson 86's plain Newton converged to one. Exercises 3.2 and 3.3
show it is **not** enough when the model is a quasi-Newton approximation rather than the true
Hessian.

The contrast with lesson 86's repair is worth stating plainly. A line search method meets
indefiniteness by changing the matrix, shifting it until Cholesky succeeds. A trust region meets it
by changing the region, shrinking the ball until the model is trustworthy. The second needs no
arbitrary shift and produces a step in the right direction rather than a damped one.

### 1.5 Fletcher-Reeves against Polak-Ribiere

The two rules for $\beta$ are

$$
\beta^{\text{FR}} = \frac{g_{k+1}^{T}g_{k+1}}{g_k^{T}g_k} , \qquad
\beta^{\text{PR}} = \frac{g_{k+1}^{T}(g_{k+1} - g_k)}{g_k^{T}g_k} .
$$

**Fletcher-Reeves stalls** when the method takes a bad step, for the following reason. Its $\beta$
is a ratio of squared norms, so it is always non-negative, and if the gradient barely changes then
$\beta^{\text{FR}} \approx 1$. The direction $p_{k+1} = -g_{k+1} + \beta p_k$ is then dominated by
the previous direction, so the method keeps going the way it was going, which is the way that did not
work. Once that starts, $\beta$ stays near $1$ and the stall persists, sometimes for hundreds of
iterations.

**Polak-Ribiere recovers automatically.** Its numerator contains $g_{k+1} - g_k$. If the gradient
barely changes, that is small, so $\beta^{\text{PR}} \approx 0$ and

$$
p_{k+1} \approx -g_{k+1} ,
$$

a **steepest descent restart**, taken without any explicit restart rule. The method notices its own
stall through the same quantity a restart test would look at.

The price is that $\beta^{\text{PR}}$ can be negative, and then the direction need not be a descent
direction. PR+ takes $\beta = \max(\beta^{\text{PR}}, 0)$, which keeps the automatic restart and
restores the guarantee, and it is what the lesson measures as the best of the three.

### 2.1 Deriving the BFGS inverse update

Ask for the inverse $H_{k+1}$ to satisfy the secant condition $H_{k+1}y = s$, to be symmetric, and to
be the **closest** matrix to $H_k$ in the weighted Frobenius norm

$$
\lVert A \rVert_{W} = \lVert W^{1/2} A W^{1/2}\rVert_{F} ,
$$

with any $W$ satisfying $Ws = y$. That is:

$$
H_{k+1} = \arg\min_{H = H^{T},\; Hy = s} \lVert H - H_k \rVert_{W} .
$$

**Why a weighted norm.** An unweighted Frobenius norm is not invariant under a change of variables,
so it would make the answer depend on the units of $x$. Choosing $W$ with $Ws = y$, that is $W$ an
average Hessian, makes the whole construction affine invariant, which is the property 2.3 of lesson
86 identified as the reason Newton needs no tuning.

**The solution.** Writing $\rho = 1/(y^{T}s)$ and $V = I - \rho\, s y^{T}$, the minimizer is

$$
\boxed{\; H_{k+1} = V H_k V^{T} + \rho\, s s^{T} \;}
$$

**Checking the secant condition.** Note $V^{T}y = y - \rho\, y (s^{T}y) = y - y = 0$. Therefore

$$
H_{k+1}y = V H_k (V^{T}y) + \rho\, s (s^{T}y) = 0 + s = s . \checkmark
$$

**Checking symmetry.** $(VH_kV^{T})^{T} = VH_k^{T}V^{T} = VH_kV^{T}$ when $H_k$ is symmetric, and
$ss^{T}$ is symmetric. So $H_{k+1}$ is symmetric whenever $H_k$ is. $\checkmark$

The form $VH V^{T} + \rho ss^{T}$ is the reason BFGS is used in the inverse: each step is a few
matrix-vector products and no linear solve, and it is the form the two loop recursion of 2.3 applies
without ever forming $V$.

```python
import numpy as np
from nalib import quasinewton as qn

rng = np.random.default_rng(42)
print(f"{'n':>4}{'secant residual':>18}{'symmetry':>12}{'stays definite':>17}"
      f"{'rivals that were closer':>26}")
for n in (2, 5, 20):
    # a quadratic, so y = A s exactly and W = A is a legal weight with W s = y
    frame, _ = np.linalg.qr(rng.normal(size=(n, n)))
    A = frame @ np.diag(rng.uniform(0.5, 3.0, size=n)) @ frame.T
    A = 0.5 * (A + A.T)
    previous = np.eye(n)
    s = rng.normal(size=n)
    y = A @ s
    rho = 1.0 / float(y @ s)
    V = np.eye(n) - rho * np.outer(s, y)
    fresh = V @ previous @ V.T + rho * np.outer(s, s)

    root = np.linalg.cholesky(A)
    weighted = lambda M: float(np.linalg.norm(root.T @ M @ root, "fro"))
    project = np.eye(n) - np.outer(y, y) / float(y @ y)
    closer = 0
    for _ in range(400):
        noise = rng.normal(size=(n, n))
        noise = 0.5 * (noise + noise.T)
        rival = fresh + 1e-2 * (project @ noise @ project)   # keeps E y = 0
        assert float(np.linalg.norm(rival @ y - s)) < 1e-10
        if weighted(rival - previous) < weighted(fresh - previous) - 1e-12:
            closer += 1
    print(f"{n:>4}{float(np.linalg.norm(fresh @ y - s)):>18.2e}"
          f"{float(np.linalg.norm(fresh - fresh.T)):>12.2e}"
          f"{str(bool(np.min(np.linalg.eigvalsh(fresh)) > 0.0)):>17}{closer:>26}")

out = qn.the_secant_condition_holds_and_needs_positive_curvature()
print(f"\non a real run: worst secant residual "
      f"{out['worst_secant_residual']:.2e} over {out['steps']} steps")
print(f"  every curvature positive: {out['every_curvature_positive']}")
```

The secant condition and symmetry hold to rounding at every size and the update stays positive
definite. The last column is the minimality claim: out of four hundred random symmetric perturbations
that **keep** the secant condition, built by projecting out the $y$ direction on both sides, not one
is closer to $H_k$ in the weighted norm. That is the numerical form of "BFGS is the minimizer", and
it is checked against the legal weight $W = A$, which satisfies $Ws = y$ exactly because the test
problem is a quadratic.

### 2.2 Positive definiteness

**Claim.** If $H_k$ is positive definite and $y^{T}s > 0$, then $H_{k+1} = VH_kV^{T} + \rho ss^{T}$
is positive definite.

**Proof.** Take any $z \ne 0$ and compute

$$
z^{T}H_{k+1}z = (V^{T}z)^{T}H_k(V^{T}z) + \rho\,(s^{T}z)^{2} .
$$

Both terms are non-negative: the first because $H_k$ is positive definite, the second because
$\rho = 1/(y^{T}s) > 0$. So the sum is non-negative, and it is zero only if **both** vanish.

The second vanishes only if $s^{T}z = 0$. The first vanishes only if $V^{T}z = 0$, that is
$z = \rho\, y (s^{T}z)$. Combining, $s^{T}z = 0$ forces $z = 0$, contradicting $z \ne 0$. Therefore
$z^{T}H_{k+1}z > 0$. $\square$

**And it fails otherwise.** If $y^{T}s < 0$ then $\rho < 0$, and taking $z = s$ gives

$$
s^{T}H_{k+1}s = (V^{T}s)^{T}H_k(V^{T}s) + \rho\,(s^{T}s)^{2} ,
$$

whose second term is negative and can be made to dominate by scaling $s$. More directly, 1.2 showed
$s^{T}B_{k+1}s = y^{T}s$, so the curvature the new matrix reports along $s$ is negative by
construction. If $y^{T}s = 0$ the update does not exist at all, since $\rho$ is infinite.

```python
import numpy as np

rng = np.random.default_rng(42)
n = 6
frame, _ = np.linalg.qr(rng.normal(size=(n, n)))
previous = frame @ np.diag(rng.uniform(0.5, 3.0, size=n)) @ frame.T
previous = 0.5 * (previous + previous.T)
s = rng.normal(size=n)

print(f"{'y.s':>12}{'smallest eigenvalue of the update':>36}{'definite':>11}")
for scale in (2.0, 0.5, 0.05, -0.05, -0.5, -2.0):
    y = scale * s + 0.1 * rng.normal(size=n)
    sy = float(y @ s)
    rho = 1.0 / sy
    V = np.eye(n) - rho * np.outer(s, y)
    fresh = V @ previous @ V.T + rho * np.outer(s, s)
    small = float(np.min(np.linalg.eigvalsh(0.5 * (fresh + fresh.T))))
    print(f"{sy:>12.4f}{small:>36.6f}{str(small > 0.0):>11}")
print("\nand the curvature the update reports along s is exactly y.s:")
y = -0.5 * s + 0.1 * rng.normal(size=n)
rho = 1.0 / float(y @ s)
V = np.eye(n) - rho * np.outer(s, y)
fresh = V @ previous @ V.T + rho * np.outer(s, s)
print(f"  s . inverse(H_new) s = {float(s @ np.linalg.solve(fresh, s)):.6f}, "
      f"y.s = {float(y @ s):.6f}")
```

Every positive $y^{T}s$ gives a positive definite update and every negative one does not, with the
smallest eigenvalue crossing zero exactly at the sign change. The last line confirms the algebra of
1.2 numerically.

That is why the lesson's `bfgs` **skips** the update when $y^{T}s$ is not positive rather than
applying it. Skipping loses information; applying it would lose the descent guarantee.

### 2.3 The two loop recursion

**The idea.** Expanding the BFGS update $m$ times from a starting matrix $H_0$ gives a product of
$m$ factors $V_i$ on each side plus $m$ rank one terms. Applying that to a vector never needs the
matrix, only the stored pairs.

**The recursion.** With pairs $(s_i, y_i)$ for $i = k-m, \dots, k-1$ and $\rho_i = 1/(y_i^{T}s_i)$:

```
q = g
for i = k-1 down to k-m:          # the first loop, newest first
    a[i] = rho[i] * (s[i] . q)
    q = q - a[i] * y[i]
q = H0 q
for i = k-m up to k-1:            # the second loop, oldest first
    b = rho[i] * (y[i] . q)
    q = q + (a[i] - b) * s[i]
return -q                          # the search direction
```

**Why it is right.** Each pass of the first loop applies one $V_i^{T}$ and stores the scalar that the
second loop needs to add back the rank one term. Writing $H_k = V_{k-1}^{T}\cdots H_0 \cdots V_{k-1}$
plus the accumulated $\rho_i s_i s_i^{T}$ terms and multiplying out gives exactly this order of
operations.

**The count.** Each loop iteration is two inner products and two vector updates, so $4n$ operations,
and there are $2m$ iterations, giving

$$
4mn \text{ operations and } 2mn \text{ storage} ,
$$

plus whatever $H_0$ costs, which is $n$ for the usual $\gamma I$. Compare full BFGS at $n^{2}$
operations and $n^{2}$ storage. At $n = 10^{6}$ and $m = 5$ that is $2\times10^{7}$ against
$10^{12}$: the difference between a second and a week, and between $80$ megabytes and $8$ terabytes.

```python
import numpy as np
from nalib import quasinewton as qn
from nalib.optimize import quadratic


def two_loop(pairs, slope, h0):
    q = np.asarray(slope, dtype=float).ravel().copy()
    alphas = []
    for s, y, rho in reversed(pairs):
        alpha = rho * float(s @ q)
        alphas.append(alpha)
        q = q - alpha * y
    q = h0 * q
    for (s, y, rho), alpha in zip(pairs, reversed(alphas)):
        q = q + (alpha - rho * float(y @ q)) * s
    return q


rng = np.random.default_rng(42)
print(f"{'n':>5}{'pairs':>7}{'relative gap to the built matrix':>35}"
      f"{'two loop work':>16}{'matrix work':>14}")
for n in (5, 20, 100):
    # pairs from a real run, so the operator is the one the method actually builds
    problem = quadratic(condition=100.0, dimension=n)
    x = problem["start"].copy()
    slope = np.asarray(problem["gradient"](x), dtype=float)
    all_pairs = []
    for _ in range(6):
        move = -slope
        step = float(slope @ slope) / float(slope @ problem["matrix"] @ slope)
        s = step * move
        fresh = np.asarray(problem["gradient"](x + s), dtype=float)
        all_pairs.append((s, fresh - slope, 1.0 / float((fresh - slope) @ s)))
        x, slope = x + s, fresh
    for m in (1, 3, 5):
        pairs = all_pairs[-m:]
        H = np.eye(n)
        for s, y, rho in pairs:
            V = np.eye(n) - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
        g = np.asarray(problem["gradient"](problem["start"]), dtype=float)
        gap = (float(np.linalg.norm(two_loop(pairs, g, 1.0) - H @ g))
               / float(np.linalg.norm(H @ g)))
        print(f"{n:>5}{m:>7}{gap:>35.2e}{4 * m * n:>16}{n * n:>14}")
```

The recursion reproduces the explicitly built matrix to rounding at every size and memory, which is
the check that the loop order and the stored scalars are right. The pairs here come from a real
descent run rather than from random vectors, because random $(s, y)$ pairs build an operator whose
condition number is astronomical and the comparison then measures the conditioning rather than the
recursion.

The operation counts in the last two columns cross over at $m = n/4$, so for the usual $m$ between
$3$ and $20$ the recursion wins from about $n = 80$ upward, and the storage saving arrives much
sooner.

### 2.4 The Cauchy point

**Definition.** The Cauchy point is the minimizer of the model along the steepest descent direction,
inside the ball:

$$
p^{C} = -\tau \frac{\Delta}{\lVert g\rVert}\, g , \qquad
\tau = \begin{cases} 1, & g^{T}Bg \le 0,\\[2pt]
\min\!\left(\dfrac{\lVert g\rVert^{3}}{\Delta\, g^{T}Bg},\; 1\right), & g^{T}Bg > 0 .\end{cases}
$$

**Derivation.** Restrict the model to the ray $p = -t g$ with $t \ge 0$:

$$
\varphi(t) = m(-tg) = -t\,\lVert g\rVert^{2} + \tfrac{t^{2}}{2}\, g^{T}Bg .
$$

If $g^{T}Bg \le 0$ this is decreasing for all $t > 0$, so the minimum on the ball is at the boundary,
$t = \Delta/\lVert g\rVert$, which is $\tau = 1$. If $g^{T}Bg > 0$ the unconstrained minimizer is
$t^{*} = \lVert g\rVert^{2}/g^{T}Bg$, and the answer is that when it fits and the boundary otherwise,
which is the $\min$ above after writing $t$ in units of $\Delta/\lVert g\rVert$.

**Existence.** Both branches are explicit formulas in $g$, $B$ and $\Delta$ that are defined
whenever $g \ne 0$, with no inversion, no factorization and no positive definiteness needed. If
$g = 0$ the point is already stationary and no step is wanted. **So the Cauchy point always exists.**

That is what makes it the safety net. Whatever else a trust region method tries, it can always fall
back on a step that is at least as good as the Cauchy point, and the global convergence theory of
trust region methods rests on exactly that: any method achieving a fixed fraction of the Cauchy
decrease converges to a stationary point.

The price is that the Cauchy point is steepest descent inside a ball, and therefore inherits lesson
86's dependence on the condition number. Exercise 3.3 measures how much decrease it leaves on the
table.

```python
import numpy as np
from nalib import quasinewton as qn

rng = np.random.default_rng(42)
print(f"{'n':>4}{'smallest eigenvalue':>22}{'radius':>9}{'Cauchy model':>15}"
      f"{'inside the ball':>18}{'a real decrease':>18}")
for n in (2, 8):
    for smallest in (-3.0, 0.0, 1.0):
        frame, _ = np.linalg.qr(rng.normal(size=(n, n)))
        values = np.concatenate([[smallest], rng.uniform(1.0, 6.0, size=n - 1)])
        B = frame @ np.diag(values) @ frame.T
        B = 0.5 * (B + B.T)
        g = rng.normal(size=n)
        for radius in (0.5, 3.0):
            curvature = float(g @ B @ g)
            if curvature <= 0.0:
                tau = 1.0
            else:
                tau = min(float(np.linalg.norm(g)) ** 3 / (radius * curvature), 1.0)
            step = -tau * radius / float(np.linalg.norm(g)) * g
            value = float(g @ step) + 0.5 * float(step @ B @ step)
            print(f"{n:>4}{float(np.linalg.eigvalsh(B)[0]):>22.4f}{radius:>9.1f}"
                  f"{value:>15.6f}"
                  f"{str(float(np.linalg.norm(step)) <= radius * (1 + 1e-12)):>18}"
                  f"{str(value < 0.0):>18}")
```

The Cauchy point is inside the ball and gives a genuine decrease in every case, including where the
smallest eigenvalue is negative and where it is exactly zero, which are the two cases a Newton step
cannot handle at all.

### 2.5 Conjugate directions and $n$ step termination

**Claim.** On $f(x) = \tfrac12 x^{T}Ax - b^{T}x$ with $A$ symmetric positive definite, BFGS started
from $H_0 = I$ with **exact** line searches generates directions $p_0, \dots, p_{k}$ that are
$A$-conjugate, and terminates in at most $n$ steps.

**The argument, by induction.** With an exact line search, $g_{k+1}^{T}p_k = 0$. On a quadratic
$y_i = As_i$, so the secant condition $H_{k+1}y_i = s_i$ reads

$$
H_{k+1}A s_i = s_i \qquad\text{for } i \le k ,
$$

that is, $H_{k+1}$ inverts $A$ exactly on the subspace spanned by the steps taken so far. This is the
**hereditary property**: BFGS preserves every earlier secant condition, not only the newest.

Given that, $g_{k+1}$ is orthogonal to all of $s_0,\dots,s_k$ (each by the exact line search plus the
conjugacy already established), so

$$
p_{k+1}^{T}A s_i = -(H_{k+1}g_{k+1})^{T}As_i = -g_{k+1}^{T}H_{k+1}As_i = -g_{k+1}^{T}s_i = 0 ,
$$

using symmetry of $H_{k+1}$ and the hereditary property. So the new direction is $A$-conjugate to
every previous step.

$n$ mutually $A$-conjugate directions span $\mathbb{R}^{n}$, and the exact line search along each
zeroes the corresponding component of the error, so after $n$ steps the error is zero. Moreover
$H_n = A^{-1}$ exactly: it satisfies $H_nAs_i = s_i$ for $n$ independent $s_i$.

**Every hypothesis matters**, and the lesson measures what happens without them.

```python
import numpy as np
from nalib import quasinewton as qn

out = qn.the_n_step_property_needs_an_exact_line_search()
print(f"{'n':>5}{'c2=0.9':>10}{'c2=0.1':>10}{'c2=0.01':>11}{'steps / n at the tightest':>28}")
for row in out["rows"]:
    print(f"{row['n']:>5}{row['c2=0.9']:>10}{row['c2=0.1']:>10}{row['c2=0.01']:>11}"
          f"{row['tightest_over_n']:>28.2f}")
print(f"\ntightening never costs steps: {out['tightening_never_costs_steps']}")
print(f"the tightest search is within three of n: "
      f"{out['tightest_is_within_three_of_n']}")
print(f"worst ratio to n: {out['worst_ratio_to_n']:.2f}")

# the conjugacy itself, with an exact search on a quadratic
from nalib.optimize import quadratic

for n in (4, 8):
    problem = quadratic(condition=50.0, dimension=n)
    A, centre = problem["matrix"], problem["minimizers"][0]
    x = problem["start"].copy()
    inverse, steps = np.eye(n), []
    slope = A @ (x - centre)
    for _ in range(n):
        move = -inverse @ slope
        alpha = -float(slope @ move) / float(move @ A @ move)   # the exact line search
        s = alpha * move
        y = A @ s
        rho = 1.0 / float(y @ s)
        V = np.eye(n) - rho * np.outer(s, y)
        inverse = V @ inverse @ V.T + rho * np.outer(s, s)
        steps.append(s)
        x, slope = x + s, slope + y
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    worst = max(abs(float(steps[i] @ A @ steps[j])) for i, j in pairs)
    scale = max(float(s @ A @ s) for s in steps)
    print(f"n = {n}: after n steps the gradient is "
          f"{float(np.linalg.norm(slope)):.2e}, the inverse differs from A^-1 by "
          f"{float(np.linalg.norm(inverse - np.linalg.inv(A))):.2e},")
    print(f"        and the worst conjugacy s_i A s_j is {worst / scale:.2e} "
          f"relative to the largest s A s")
```

The `tightest_over_n` column is the whole point: as $c_2$ falls from $0.9$ to $0.01$ the line search
approaches an exact one and the step count falls toward $n$, ending within three of it at every size.
So the excess is the line search and not the update.

The second table checks the theorem itself. With a genuinely exact line search on a quadratic, after
exactly $n$ steps the gradient is at rounding, the accumulated inverse equals $A^{-1}$, and every
pair of steps is $A$-conjugate to rounding. All three consequences of 2.5 hold together, which is
the evidence that the derivation above is the derivation of what the code does.

This is the same shape as lesson 85's Fibonacci search: **an optimality result that holds under an
assumption practice cannot satisfy.** The value of knowing it is not that the bound applies but that
it explains why BFGS is fast on nearly quadratic problems, which is what most problems look like near
a minimum.

### 3.1 DFP

DFP is BFGS with $s$ and $y$ exchanged, so it is the closest update in the $B$ form rather than the
$H$ form:

$$
H^{\text{DFP}}_{k+1} = H_k + \frac{ss^{T}}{y^{T}s} - \frac{H_k y y^{T} H_k}{y^{T}H_k y} .
$$

It satisfies the same secant condition and preserves positive definiteness under the same condition,
so on paper the two are equally good. DFP came first, by about six years.

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import rosenbrock, quadratic


def quasi_newton(problem, kind="bfgs", tol=1e-10, rounds=5000, c2=0.9):
    """One driver for the three updates, so the comparison is of the update alone."""
    x = np.asarray(problem["start"], dtype=float).ravel()
    f, g = problem["f"], problem["gradient"]
    n = x.size
    inverse = np.eye(n)
    slope = np.asarray(g(x), dtype=float).ravel()
    calls, skipped, resets = 0, 0, 0
    for k in range(rounds):
        if float(np.linalg.norm(slope)) <= tol:
            break
        move = -inverse @ slope
        if float(slope @ move) >= 0.0:               # the update lost definiteness
            inverse, move, _ = np.eye(n), -slope, resets
            resets += 1
        out = gr.strong_wolfe(f, g, x, move, wolfe_c2=c2)
        calls += out["calls"]
        if not out["accepted"] and out["step"] <= 0.0:
            break
        s = out["step"] * move
        fresh = np.asarray(g(x + s), dtype=float).ravel()
        y = fresh - slope
        x, slope = x + s, fresh
        curvature = float(s @ y)
        if curvature <= 1e-14 * float(np.linalg.norm(s) * np.linalg.norm(y)):
            skipped += 1
            continue
        if kind == "bfgs":
            rho = 1.0 / curvature
            left = np.eye(n) - rho * np.outer(s, y)
            inverse = left @ inverse @ left.T + rho * np.outer(s, s)
        elif kind == "dfp":
            applied = inverse @ y
            inverse = (inverse + np.outer(s, s) / curvature
                       - np.outer(applied, applied) / float(y @ applied))
        else:                                         # symmetric rank one
            gap = s - inverse @ y
            bottom = float(gap @ y)
            if abs(bottom) <= 1e-8 * float(np.linalg.norm(gap) * np.linalg.norm(y)):
                skipped += 1
                continue
            inverse = inverse + np.outer(gap, gap) / bottom
    return {"steps": k, "calls": calls, "skipped": skipped, "resets": resets, "x": x,
            "gradient": float(np.linalg.norm(slope))}


cases = (("Rosenbrock 2", rosenbrock(2)), ("Rosenbrock 10", rosenbrock(10)),
         ("Rosenbrock 30", rosenbrock(30)),
         ("quadratic k=1e2 n=10", quadratic(condition=1e2, dimension=10)),
         ("quadratic k=1e4 n=10", quadratic(condition=1e4, dimension=10)),
         ("quadratic k=1e6 n=30", quadratic(condition=1e6, dimension=30)))
print(f"{'problem':>22}{'BFGS steps':>12}{'DFP steps':>11}{'SR1 steps':>11}"
      f"{'BFGS grad':>12}{'DFP grad':>11}{'SR1 grad':>11}{'SR1 resets':>12}")
for label, problem in cases:
    first = quasi_newton(problem, "bfgs")
    second = quasi_newton(problem, "dfp")
    third = quasi_newton(problem, "sr1")
    print(f"{label:>22}{first['steps']:>12}{second['steps']:>11}{third['steps']:>11}"
          f"{first['gradient']:>12.1e}{second['gradient']:>11.1e}"
          f"{third['gradient']:>11.1e}{third['resets']:>12}")
```

**On quadratics the two are indistinguishable.** DFP takes $17$, $18$ and $40$ steps where BFGS takes
$18$, $19$ and $42$, so if anything it is very slightly ahead. Nothing in the measurement would tell
you to prefer BFGS.

**On Rosenbrock DFP fails outright.** At $n = 10$ and $n = 30$ it runs into the $5000$ step cap with
gradients of $21$ and $41$, where BFGS finishes in $86$ and $184$ steps at $10^{-11}$.

The mechanism is known and is the reason BFGS displaced DFP. Both updates satisfy the secant
condition, but they behave differently when the approximation has become bad. BFGS is
**self-correcting**: when its $H$ has eigenvalues that are much too large, the update tends to shrink
them, so a poor line search does bounded damage. DFP has no such property, and a few inaccurate steps
leave it with an $H$ that is nearly singular in some directions, after which every direction it
produces is nearly useless.

That is invisible on a quadratic, because a quadratic is exactly the case where the approximation
never becomes bad. **The two updates differ only where the theory does not reach**, which is why the
choice between them had to be settled by measurement.

### 3.2 Symmetric rank one

SR1 asks for the **smallest rank** correction satisfying the secant condition:

$$
H_{k+1} = H_k + \frac{(s - H_k y)(s - H_k y)^{T}}{(s - H_k y)^{T}y} .
$$

It has one clear advantage and one clear defect. The advantage is that it satisfies the secant
condition with a rank one change, so on a quadratic it builds the exact inverse Hessian in at most
$n+1$ steps rather than approaching it. The defect is that **it does not preserve positive
definiteness**, and its denominator can be zero.

The table above records both. **SR1 beats BFGS on every quadratic**: $10$, $11$ and $31$ steps
against $18$, $19$ and $42$, and it reaches a gradient of exactly $0$ twice. **On Rosenbrock it needs
resets**, $7$, $12$ and $57$ of them, because the approximation loses positive definiteness and the
direction goes uphill, and it costs $44$, $148$ and $795$ steps against BFGS's $36$, $86$ and $184$.

So in a line search method the defect is decisive. The exercise's real question is what happens in a
trust region, where an indefinite model is allowed.

```python
import numpy as np
from nalib import quasinewton as qn
from nalib.optimize import rosenbrock, quadratic


def sr1_trust_region(problem, solver, radius=1.0, tol=1e-10, rounds=5000):
    """SR1 in a trust region: the model may be indefinite and that is the point."""
    x = np.asarray(problem["start"], dtype=float).ravel()
    f, g = problem["f"], problem["gradient"]
    matrix = np.eye(x.size)
    slope = np.asarray(g(x), dtype=float).ravel()
    rejected, indefinite = 0, 0
    for k in range(rounds):
        if float(np.linalg.norm(slope)) <= tol or radius < 1e-14:
            break
        model = 0.5 * (matrix + matrix.T)
        if float(np.min(np.linalg.eigvalsh(model))) <= 0.0:
            indefinite += 1
        move = solver(slope, model, radius)["step"]
        predicted = -(float(slope @ move) + 0.5 * float(move @ model @ move))
        actual = float(f(x)) - float(f(x + move))
        ratio = actual / predicted if predicted > 0.0 else -1.0
        fresh = np.asarray(g(x + move), dtype=float).ravel()
        gap = (fresh - slope) - matrix @ move          # updated even on a rejected step
        bottom = float(gap @ move)
        if abs(bottom) > 1e-8 * float(np.linalg.norm(gap) * np.linalg.norm(move)):
            matrix = matrix + np.outer(gap, gap) / bottom
        if ratio < 0.25:
            radius *= 0.25
        elif ratio > 0.75 and abs(float(np.linalg.norm(move)) - radius) < 1e-10:
            radius = min(2.0 * radius, 100.0)
        if ratio > 0.1:
            x, slope = x + move, fresh
        else:
            rejected += 1
    return {"steps": k, "x": x, "rejected": rejected, "indefinite": indefinite}


print(f"{'problem':>22}{'dogleg steps':>14}{'dogleg distance':>17}{'indefinite':>12}")
for label, problem, at_ones in (("Rosenbrock 2", rosenbrock(2), True),
                                ("Rosenbrock 10", rosenbrock(10), True),
                                ("Rosenbrock 30", rosenbrock(30), True),
                                ("quadratic k=1e4 n=10",
                                 quadratic(condition=1e4, dimension=10), False)):
    target = (np.ones(problem["dimension"]) if at_ones
              else problem["minimizers"][0])
    out = sr1_trust_region(problem, lambda g, b, r: qn.dogleg(g, b, r))
    print(f"{label:>22}{out['steps']:>14}"
          f"{float(np.linalg.norm(out['x'] - target)):>17.2e}{out['indefinite']:>12}")
```

**With the dogleg it works at two variables and fails at ten and thirty**, running to the cap at
distances of $0.048$ and $0.28$.

The reason is not SR1's. When the model is indefinite the dogleg has no Newton point to aim at, so it
falls back to the Cauchy step, which is steepest descent inside a ball. With SR1's model indefinite
on most steps, the method **is** steepest descent, and it inherits lesson 86's step counts.

So the honest answer to "what does SR1 buy in a trust region" is: nothing, unless the trust region
subproblem is solved properly. Exercise 3.3 supplies that and returns to this.

### 3.3 The exact trust region step

The optimality conditions for $\min_{\lVert p\rVert\le\Delta} g^{T}p + \tfrac12 p^{T}Bp$ are

$$
(B + \lambda I)\,p = -g , \qquad \lambda \ge 0 , \qquad
B + \lambda I \succeq 0 , \qquad \lambda\,(\lVert p\rVert - \Delta) = 0 .
$$

In the eigenbasis of $B$, with $B = Q\Lambda Q^{T}$ and $w = Q^{T}g$, the step length is

$$
\lVert p(\lambda)\rVert^{2} = \sum_i \frac{w_i^{2}}{(\lambda_i + \lambda)^{2}} ,
$$

a decreasing function of $\lambda$ on $(-\lambda_{\min}, \infty)$. Moré and Sorensen solve
$\lVert p(\lambda)\rVert = \Delta$ by Newton's method on $1/\Delta - 1/\lVert p(\lambda)\rVert$,
which is nearly linear in $\lambda$ and therefore converges in a handful of steps, safeguarded by a
bracket.

```python
import numpy as np
from nalib import quasinewton as qn


def more_sorensen(slope, matrix, radius, tol=1e-10, rounds=200):
    """The exact minimizer of the model on the ball, through the eigen form."""
    g = np.asarray(slope, dtype=float).ravel()
    B = np.asarray(matrix, dtype=float)
    B = 0.5 * (B + B.T)
    radius = max(float(radius), 1e-300)
    values, vectors = np.linalg.eigh(B)
    weights = vectors.T @ g
    smallest = float(values[0])
    if smallest > 0.0:
        inside = vectors @ (-weights / values)
        if float(np.linalg.norm(inside)) <= radius:
            return {"step": inside, "kind": "interior", "lam": 0.0}
    scale = max(abs(smallest), float(np.max(np.abs(values))), 1.0)
    close = np.abs(values - smallest) <= 1e-10 * scale
    if float(np.max(np.abs(weights[close]))) <= 1e-12 * max(float(np.linalg.norm(g)), 1.0):
        # the hard case: no lambda solves it, so add a multiple of the smallest eigenvector
        safe = np.where(close, np.inf, values - smallest)
        base = vectors @ (-weights / safe)
        extra = np.sqrt(max(radius ** 2 - float(base @ base), 0.0))
        return {"step": base + extra * vectors[:, 0], "kind": "hard", "lam": -smallest}
    low = max(0.0, -smallest) * (1.0 + 1e-12) + 1e-14 * scale
    high = low + float(np.linalg.norm(g)) / radius
    lam = low
    for _ in range(int(rounds)):
        shifted = values + lam
        second = float(np.sum(weights ** 2 / shifted ** 2))
        length = np.sqrt(second)
        if abs(length - radius) <= tol * radius:
            break
        third = float(np.sum(weights ** 2 / shifted ** 3))
        if third <= 0.0:
            break
        if length > radius:
            low = lam
        else:
            high = lam
        lam = lam + (second / third) * (length - radius) / radius
        if not low < lam < high:                       # the safeguard
            lam = 0.5 * (low + high)
    return {"step": vectors @ (-weights / (values + lam)), "kind": "boundary", "lam": lam}


rng = np.random.default_rng(42)
print(f"{'n':>4}{'radius':>9}{'exact length':>15}{'fits':>7}{'dogleg model':>15}"
      f"{'exact model':>14}{'dogleg / exact':>16}{'dogleg kind':>13}")
for n in (2, 5, 20):
    for radius in (0.1, 1.0, 5.0):
        frame, _ = np.linalg.qr(rng.normal(size=(n, n)))
        B = frame @ np.diag(rng.uniform(-2.0, 5.0, size=n)) @ frame.T
        B = 0.5 * (B + B.T)
        g = rng.normal(size=n)
        rough = qn.dogleg(g, B, radius)
        exact = more_sorensen(g, B, radius)
        model = lambda p: float(g @ p) + 0.5 * float(p @ B @ p)
        print(f"{n:>4}{radius:>9.1f}{float(np.linalg.norm(exact['step'])):>15.6f}"
              f"{str(float(np.linalg.norm(exact['step'])) <= radius * (1 + 1e-8)):>7}"
              f"{model(rough['step']):>15.6f}{model(exact['step']):>14.6f}"
              f"{model(rough['step']) / model(exact['step']):>16.4f}"
              f"{rough['kind']:>13}")
```

The comparison splits into three regimes.

**When the model is convex and the Newton step fits, the dogleg is exact**, ratio $1.0000$. Nothing
is lost.

**When the radius is small the dogleg is nearly exact**, ratios $0.9955$ and $0.9986$, because a
small ball makes the linear term dominate and steepest descent is the right direction there.

**When the model is indefinite and the radius is large the dogleg is bad**, capturing $12$ to $25$
per cent of the available decrease. In one case it is worse than that: it returns a step whose model
value is $+0.173$, an **increase**, where the exact step gives $-0.862$. A trust region will reject
that step, shrink the radius and try again, which is the loop the SR1 runs of 3.2 were stuck in.

Putting the exact solve into the SR1 trust region settles that exercise.

```python
import numpy as np
from nalib import quasinewton as qn
from nalib.optimize import rosenbrock, quadratic

print(f"{'problem':>22}{'dogleg steps':>14}{'dogleg dist':>13}{'exact steps':>13}"
      f"{'exact dist':>12}{'indefinite':>12}{'rejected':>10}")
for label, problem, at_ones in (("Rosenbrock 2", rosenbrock(2), True),
                                ("Rosenbrock 10", rosenbrock(10), True),
                                ("Rosenbrock 30", rosenbrock(30), True),
                                ("quadratic k=1e4 n=10",
                                 quadratic(condition=1e4, dimension=10), False)):
    target = (np.ones(problem["dimension"]) if at_ones
              else problem["minimizers"][0])
    rough = sr1_trust_region(problem, lambda g, b, r: qn.dogleg(g, b, r))
    exact = sr1_trust_region(problem, more_sorensen)
    print(f"{label:>22}{rough['steps']:>14}"
          f"{float(np.linalg.norm(rough['x'] - target)):>13.2e}{exact['steps']:>13}"
          f"{float(np.linalg.norm(exact['x'] - target)):>12.2e}"
          f"{exact['indefinite']:>12}{exact['rejected']:>10}")
```

**With the exact solve SR1 works everywhere**: $60$, $111$ and $268$ steps to distances of
$10^{-14}$ to $10^{-13}$, where the dogleg version failed at ten and thirty variables. The model is
indefinite on $12$, $26$ and $81$ of those steps and $14$ to $66$ steps are rejected, both of which
are normal and neither of which stops it.

The cost is an eigenvalue decomposition per step, $O(n^{3})$, which is why the dogleg is the default.
For a large problem the right choice is neither: it is the Steihaug conjugate gradient solve of
lesson 86's exercise 3.2, which handles negative curvature by stopping and going to the boundary,
costs only matrix products, and captures most of the exact step's decrease.

### 3.4 L-BFGS-B

Bounds $\ell \le x \le u$ are simple enough that the whole apparatus reduces to three changes: project
the iterate, freeze the variables that are pressed against a bound with the gradient pushing further
out, and measure convergence by the **projected** gradient rather than the gradient.

```python
import numpy as np
from nalib import quasinewton as qn
from nalib.optimize import rosenbrock


def lbfgs_b(problem, lower, upper, memory=5, tol=1e-8, rounds=5000):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    def clip(v):
        return np.minimum(np.maximum(v, lower), upper)

    x = clip(np.asarray(problem["start"], dtype=float).ravel())
    f, g = problem["f"], problem["gradient"]
    pairs, calls = [], 0
    for k in range(rounds):
        slope = np.asarray(g(x), dtype=float).ravel()
        if float(np.linalg.norm(x - clip(x - slope))) <= tol:
            break
        free = ~(((x <= lower + 1e-12) & (slope > 0.0))
                 | ((x >= upper - 1e-12) & (slope < 0.0)))
        q = np.where(free, slope, 0.0)
        alphas = []
        for s, y, rho in reversed(pairs):
            alpha = rho * float(s @ q)
            alphas.append(alpha)
            q = q - alpha * y
        if pairs:
            s, y, _ = pairs[-1]
            q = q * (float(s @ y) / float(y @ y))
        for (s, y, rho), alpha in zip(pairs, reversed(alphas)):
            q = q + (alpha - rho * float(y @ q)) * s
        direction = np.where(free, -q, 0.0)
        if float(slope @ direction) >= 0.0:
            direction = np.where(free, -slope, 0.0)
        step, base, trial = 1.0, float(f(x)), x
        for _ in range(60):                            # backtrack along the projected path
            trial = clip(x + step * direction)
            calls += 1
            if float(f(trial)) <= base + 1e-4 * float(slope @ (trial - x)):
                break
            step *= 0.5
        s = trial - x
        y = np.asarray(g(trial), dtype=float).ravel() - slope
        curvature = float(s @ y)
        if curvature > 1e-14:
            pairs.append((s, y, 1.0 / curvature))
            pairs = pairs[-memory:]
        if float(np.linalg.norm(s)) == 0.0:
            break
        x = trial
    return {"x": x, "steps": k, "calls": calls, "f": float(f(x)),
            "projected_gradient": float(np.linalg.norm(x - clip(x - g(x))))}


print(f"{'n':>4}{'bounds':>16}{'steps':>8}{'f':>14}{'at a bound':>12}"
      f"{'projected gradient':>21}{'free minimum f':>17}")
for n in (2, 10, 30):
    problem = rosenbrock(n)
    free_run = qn.lbfgs(problem, memory=10)
    for low, high in ((-2.0, 0.5), (-2.0, 2.0), (0.0, 0.8)):
        lower, upper = np.full(n, low), np.full(n, high)
        out = lbfgs_b(problem, lower, upper, memory=10)
        pressed = int(np.sum((out["x"] <= lower + 1e-9) | (out["x"] >= upper - 1e-9)))
        print(f"{n:>4}{f'[{low}, {high}]':>16}{out['steps']:>8}{out['f']:>14.8f}"
              f"{pressed:>12}{out['projected_gradient']:>21.1e}"
              f"{free_run['f']:>17.2e}")

print("\nthe two variable answers can be checked by hand")
for high in (0.5, 0.8):
    print(f"  upper bound {high}: x1 = {high}, x2 = x1^2 = {high ** 2}, "
          f"f = (1 - {high})^2 = {(1.0 - high) ** 2:.8f}")
```

Three things to check, and all three hold.

**When the bounds contain the answer, nothing changes.** With $[-2, 2]$ the method finds the
unconstrained minimum exactly, $f = 0$, at every size, with no variable at a bound. The machinery does
not interfere when it should not.

**When the bounds exclude the answer, it finds the constrained one.** With $[-2, 0.5]$ and
$[0, 0.8]$ exactly one variable sits at a bound and the projected gradient reaches $10^{-8}$.

**The two variable answers are exactly right by hand.** The Rosenbrock valley floor is $x_2 = x_1^2$,
so with $x_1$ pressed to the upper bound $u$ the minimum is $f = (1-u)^2$. At $u = 0.5$ that is
$0.25$ and at $u = 0.8$ it is $0.04$, which is what the run reports to eight decimals.

Only one variable is ever at a bound, and the reason is structural: the Rosenbrock chain ties
$x_{i+1}$ to $x_i^2$, so pushing $x_1$ against the bound leaves the rest free to follow.

The real L-BFGS-B does more than this. It computes a **generalized Cauchy point** by following the
piecewise linear projected gradient path, which identifies the active set in one sweep rather than
one bound at a time, and it then minimizes over the free variables with the bounds still enforced.
That matters when many bounds are active at once, which is exactly the case this test problem does
not exercise.

### 3.5 The scaled initial inverse

The pure $H_0 = I$ has the wrong units: it says the Hessian is the identity, which is true only if the
problem happens to be scaled that way. The standard repair is to set

$$
H_0 = \gamma I , \qquad \gamma = \frac{s^{T}y}{y^{T}y} ,
$$

after the first step, which makes $H_0$ approximate $A^{-1}$ in the least squares sense along the one
direction that has been observed.

```python
import numpy as np
from nalib import gradient as gr
from nalib.optimize import rosenbrock, quadratic


def bfgs_scaled(problem, scale_it=True, tol=1e-10, rounds=5000, c2=0.9):
    x = np.asarray(problem["start"], dtype=float).ravel()
    f, g = problem["f"], problem["gradient"]
    n = x.size
    inverse = np.eye(n)
    slope = np.asarray(g(x), dtype=float).ravel()
    calls, first, gamma = 0, True, None
    for k in range(rounds):
        if float(np.linalg.norm(slope)) <= tol:
            break
        move = -inverse @ slope
        out = gr.strong_wolfe(f, g, x, move, wolfe_c2=c2)
        calls += out["calls"]
        if not out["accepted"] and out["step"] <= 0.0:
            break
        s = out["step"] * move
        fresh = np.asarray(g(x + s), dtype=float).ravel()
        y = fresh - slope
        x, slope = x + s, fresh
        curvature = float(s @ y)
        if curvature <= 0.0:
            continue
        if first:
            gamma = curvature / float(y @ y)
            if scale_it:
                inverse = gamma * np.eye(n)
            first = False
        rho = 1.0 / curvature
        left = np.eye(n) - rho * np.outer(s, y)
        inverse = left @ inverse @ left.T + rho * np.outer(s, s)
    return {"steps": k, "calls": calls, "gamma": gamma}


print(f"{'problem':>24}{'plain steps':>13}{'scaled steps':>14}{'plain calls':>13}"
      f"{'scaled calls':>14}{'call saving':>13}{'gamma':>10}")
for label, problem in (("Rosenbrock 2", rosenbrock(2)), ("Rosenbrock 10", rosenbrock(10)),
                       ("Rosenbrock 50", rosenbrock(50)),
                       ("quadratic k=1e2 n=20", quadratic(condition=1e2, dimension=20)),
                       ("quadratic k=1e4 n=20", quadratic(condition=1e4, dimension=20)),
                       ("quadratic k=1e6 n=20", quadratic(condition=1e6, dimension=20))):
    plain = bfgs_scaled(problem, scale_it=False)
    scaled = bfgs_scaled(problem, scale_it=True)
    saving = 100.0 * (plain["calls"] - scaled["calls"]) / max(plain["calls"], 1)
    print(f"{label:>24}{plain['steps']:>13}{scaled['steps']:>14}{plain['calls']:>13}"
          f"{scaled['calls']:>14}{f'{saving:.0f}%':>13}{scaled['gamma']:>10.4f}")
```

The answer depends on which cost is counted, and that is the point of the exercise.

**On the larger Rosenbrock problems it is clearly worth it.** It saves a few per cent of steps and
$45$ to $56$ per cent of **function evaluations**: $181 \to 99$ at ten variables and $693 \to 306$ at
fifty.

**At two variables it costs $16$ per cent more evaluations**, $56 \to 65$, while saving two steps.
With so few variables there is not enough to rescale and the extra work is not recovered.

**On quadratics it makes the step count far worse and the evaluation count nearly identical.** At
$\kappa = 10^{6}$ the steps go from $31$ to $200$, and the evaluations stay at $219$ either way; at
$\kappa = 10^{2}$ the saving is $5$ per cent.

The rows say the same thing once the calls per step are compared. The unscaled run takes few steps
with an expensive line search, about three evaluations per step; the scaled run takes many steps with
a nearly free one, about one evaluation per step. **The scaling moves the work out of the line search
and into the iteration count**, which is a wash on a quadratic, a real gain on a large Rosenbrock,
and a small loss on a tiny problem.

The practical rule follows: report the evaluation count, not the iteration count. On an expensive
objective they are the only cost that exists, and a method that halves the iterations while tripling
the evaluations per iteration has made things worse.

The lesson's `lbfgs` applies this scaling at **every** step and not only the first, using the newest
pair, which is why exercise 4.2 finds it beating full BFGS.

### 4.1 The curvature constant

```python
import numpy as np
from nalib import quasinewton as qn
from nalib.optimize import rosenbrock

for label, problem in (("Rosenbrock 2", rosenbrock(2)), ("Rosenbrock 10", rosenbrock(10)),
                       ("Rosenbrock 50", rosenbrock(50))):
    print(f"  {label}")
    print(f"{'c2':>10}{'steps':>8}{'calls':>8}{'per step':>11}{'distance':>12}"
          f"{'skipped':>10}")
    for c2 in (1e-4, 1e-3, 1e-2, 1e-1, 0.5, 0.9, 0.99):
        out = qn.bfgs(problem, wolfe_c2=c2)
        print(f"{c2:>10.0e}{out['steps']:>8}{out['line_search_calls']:>8}"
              f"{out['line_search_calls'] / max(out['steps'], 1):>11.2f}"
              f"{float(np.linalg.norm(out['x'] - 1.0)):>12.2e}"
              f"{out['skipped_updates']:>10}")
    print()
```

Steps and evaluations move in **opposite** directions, over four decades.

**A tighter line search buys fewer steps.** At fifty variables $c_2 = 10^{-4}$ takes $235$ steps and
$c_2 = 1.0$ takes $368$. That is what the theory predicts: a more accurate step gives a better
$(s, y)$ pair and a better model.

**And it costs far more evaluations.** The same two runs use $3464$ and $782$ evaluations, at $14.7$
and $2.1$ per step. The total is minimized around $c_2 = 0.9$, at $720$, a factor of $4.8$ below the
tight search.

**The conventional $c_2 = 0.9$ is the minimum of the evaluation curve.** It is not an arbitrary
default; it is where the two effects balance, and the measurement puts it there at every size tried.

Two things the sweep also shows. At $c_2 = 1.0$ the curvature condition is vacuous, so $y^{T}s > 0$
fails often: skipped updates jump from $34$ to $100$ and the step count degrades to $368$. And at
$n = 10$, $c_2 = 0.1$, the run ends at a **distance of $1.99$**, a different local minimum from every
other row. So $c_2$ does not only trade speed against accuracy; it changes which answer comes out,
which is the same finding the lesson reports for the model error.

### 4.2 The memory

```python
import numpy as np
from nalib import quasinewton as qn
from nalib.optimize import rosenbrock

print(f"{'n':>6}{'m':>5}{'steps':>8}{'calls':>8}{'distance':>12}"
      f"{'full BFGS steps':>18}")
for n in (10, 100, 1000):
    problem = rosenbrock(n)
    full = qn.bfgs(problem) if n <= 100 else None
    for m in (1, 2, 3, 5, 10, 20, 50):
        out = qn.lbfgs(problem, memory=m)
        print(f"{n:>6}{m:>5}{out['steps']:>8}{out['line_search_calls']:>8}"
              f"{float(np.linalg.norm(out['x'] - 1.0)):>12.2e}"
              f"{(str(full['steps']) if full else 'too big'):>18}")
    print()
```

**One pair is not enough.** At $m = 1$ the method needs $4411$ steps at ten variables and hits the
cap at a hundred and a thousand. A single pair carries a direction but no notion of the subspace, so
the method is close to Barzilai-Borwein without its safeguards.

**The curve flattens quickly, and where depends on the size.** At $n = 10$ it is still improving at
$m = 20$: $234, 258, 139, 136, 70, 69$ for $m = 2, 3, 5, 10, 20, 50$. At $n = 100$ it is flat from
$m = 5$: $595, 569, 553, 549$. At $n = 1000$ nothing converges within $5000$ steps, but the accuracy
reached improves steadily, from a distance of $18.8$ at $m = 1$ to $3\times10^{-12}$ at $m = 50$.

**L-BFGS beats full BFGS.** At ten variables, $m = 20$ takes $70$ steps against full BFGS's $92$; at
a hundred, $m = 5$ takes $595$ against $514$, so they are comparable there. That is surprising until
you look at the implementation: the lesson's `lbfgs` rescales $H_0$ by $\gamma = s^{T}y/y^{T}y$ at
**every** step, using the newest pair, and its `bfgs` does not. Exercise 3.5 measured exactly that
effect. So the comparison is not really limited memory against full memory; it is rescaled against
not rescaled, and the rescaling is worth more than the extra pairs.

**The practical reading.** $m$ between $5$ and $20$ is the usual advice and the measurement supports
it, with the caveat that the flattening point rises with $n$. Below $m = 3$ the method is not worth
running. Above $m = 20$ the extra storage buys very little, and exercise 5.3 finds a case where it
costs something.

### 4.3 How often a trust region rejects a step

```python
import numpy as np
from nalib import quasinewton as qn
from nalib.optimize import rosenbrock

rng = np.random.default_rng(42)
print(f"{'n':>4}{'steps':>8}{'rejected':>10}{'rate':>8}"
      f"{'indefinite in the box':>23}{'step kinds':>46}")
for n in (2, 4, 6, 10, 20, 40):
    problem = rosenbrock(n)
    out = qn.trust_region(problem)
    bad = 0
    for _ in range(400):
        point = rng.uniform(-2.0, 2.0, size=n)
        if float(np.min(np.linalg.eigvalsh(problem["hessian"](point)))) <= 0.0:
            bad += 1
    total = out["steps"] + 1
    print(f"{n:>4}{out['steps']:>8}{out['rejected']:>10}"
          f"{out['rejected'] / max(total, 1):>8.2f}{bad / 400.0:>23.2f}"
          f"{str(out['step_kinds']):>46}")
```

The exercise asks to relate the rejection rate to lesson 86's indefinite fraction, and the honest
answer is that **they are not related in the way the question suggests.**

The indefinite fraction in the box rises monotonically and steeply, $0.21, 0.40, 0.53, 0.75, 0.96,
1.00$. The rejection rate does not: $0.12, 0.19, 0.37, 0.33, 0.44, 0.17$. At forty variables the
Hessian is indefinite at **every** sampled point in the box, and the rejection rate is the lowest of
all six.

The step kinds explain it. At six and ten variables the run is dominated by Cauchy and edge steps,
$686$ of $737$ at $n = 10$, which are exactly the steps the dogleg takes when the model is not
convex. At twenty and forty the run is dominated by Newton and dogleg steps, $44 + 23$ of $78$ and
$49 + 47$ of $102$, which means the model along the path taken **is** convex.

**The box statistic measures the wrong set.** What matters is the Hessian along the path a method
follows, and a trust region actively steers toward regions where its model is trustworthy: a rejected
step shrinks the radius, and a smaller radius keeps the iterate near where the model is good. The
method is not a random sample of the box, and lesson 86's number, honest as it is about the function,
overstates the difficulty the method actually meets.

The general lesson is worth keeping: **a statistic about the problem is not a prediction about a
method**, and the way to find out how often something happens to a run is to instrument the run.

### 5.1 Why superlinear and not quadratic

**The Dennis-Moré condition.** A quasi-Newton iteration $x_{k+1} = x_k - B_k^{-1}g_k$ with unit steps
converges superlinearly if and only if

$$
\frac{\lVert (B_k - \nabla^{2}f(x^{*}))\,p_k\rVert}{\lVert p_k\rVert} \longrightarrow 0 ,
$$

where $p_k$ is the step. Read carefully, that says the approximation has to become exact **only along
the directions actually taken**, not everywhere.

**What BFGS achieves.** Exactly that, and no more. Each update enforces the secant condition along
one direction, so after many steps $B_k$ agrees with the true Hessian on the subspace the iteration
has explored, which is where the Dennis-Moré quotient is evaluated. The theorem of Powell gives
superlinear convergence under a Wolfe line search for convex problems, and that is the strongest
statement available.

**What it does not achieve.** Quadratic convergence needs

$$
\lVert B_k - \nabla^{2}f(x^{*})\rVert = O(\lVert e_k\rVert) ,
$$

the full matrix, in norm, converging at the rate of the error. BFGS cannot supply that: each step
supplies $n$ numbers about a matrix with $n(n+1)/2$ entries, so after $k$ steps at most $kn$ pieces
of information have arrived, and the ones from early steps are about the Hessian at points far from
$x^{*}$. **The information does not arrive fast enough**, and no choice of update within the family
changes the arithmetic.

**What would have to be true.** Either the Hessian would have to be supplied, which is Newton's
method, or $n$ independent secant pairs would have to be gathered per step, which costs $n$ gradient
evaluations and is finite differencing in disguise. There is no third option, and that is the sense
in which the gap is structural rather than a defect of BFGS.

```python
import numpy as np
from nalib import quasinewton as qn

out = qn.superlinear_but_not_quadratic()
print(f"{'error':>13}{'ratio to the last':>20}{'ratio to its square':>22}")
for error, linear, square in zip(out["tail"][1:], out["linear_ratios"],
                                 out["square_ratios"]):
    print(f"{error:>13.3e}{linear:>20.6f}{square:>22.3e}")
print(f"\nusable points before the floor: {out['usable_points']}")
print(f"the linear ratios stay far below one: {out['ratios_are_far_below_one']}, "
      f"largest {out['biggest_linear_ratio']:.4f}")
print(f"the square ratios grow by a factor of {out['square_ratio_growth']:.1e}, "
      f"so it is not quadratic: {out['square_ratios_grow']}")
print(f"Newton's square ratio on the same problem stays near "
      f"{out['newton_square_constant']:.1e}")
print(f"whether the linear ratio tends to zero is resolvable here: "
      f"{not out['whether_it_tends_to_zero_is_not_resolvable']}")
```

The measurement matches the theory in the one direction it can. The quadratic ratios grow without
bound, which **rules out** quadratic convergence, and that is a negative statement a finite
measurement can support. The linear ratios fall well below one, which is consistent with superlinear
convergence but does not establish it, because establishing it means showing a limit is zero and
double precision runs out first. Lesson 85's exercise 4.3 met the same wall from the other side and
climbed it by going to a hundred and twenty digits, which is available there and not here.

### 5.2 The trust region and negative curvature

**Claim.** If $B$ has a negative eigenvalue, the minimizer of $m(p) = g^{T}p + \tfrac12 p^{T}Bp$ over
$\lVert p\rVert \le \Delta$ lies on the boundary $\lVert p\rVert = \Delta$.

**Proof.** Suppose the minimizer $p^{*}$ were interior. Then it is an unconstrained local minimizer
of $m$, so $\nabla m(p^{*}) = g + Bp^{*} = 0$ and $\nabla^{2}m = B \succeq 0$. But $B$ has a negative
eigenvalue, contradiction. So the minimizer is on the boundary. $\square$

A sharper statement is available. Let $v$ be a unit eigenvector for $\lambda_{\min} < 0$. Along
$p = tv$,

$$
m(tv) = t\,(g^{T}v) + \tfrac{t^{2}}{2}\lambda_{\min} ,
$$

which tends to $-\infty$ as $\lvert t\rvert$ grows, whichever sign of $t$ makes $t\,g^{T}v \le 0$. So
the model is **unbounded below** and the ball is the only thing stopping the step. There is no Newton
point to aim at, which is exactly why lesson 86's line search methods had to modify the matrix and a
trust region does not.

**Which direction it uses.** From 3.3's optimality conditions, the step satisfies
$(B + \lambda I)p = -g$ with $\lambda \ge -\lambda_{\min} > 0$. Writing it in the eigenbasis,

$$
p = -\sum_i \frac{w_i}{\lambda_i + \lambda}\, v_i ,
$$

and the denominator $\lambda_{\min} + \lambda$ is the smallest of them all, so **the component along
the most negative curvature direction is amplified more than any other.** As $\Delta$ grows,
$\lambda$ falls toward $-\lambda_{\min}$ and that amplification grows without bound: the step aligns
itself with $v_{\min}$.

The hard case is when $g$ happens to be orthogonal to $v_{\min}$, so $w_{\min} = 0$ and no $\lambda$
produces a step of length $\Delta$. Then the answer is the solve in the remaining subspace plus a
multiple of $v_{\min}$ chosen to reach the boundary, which is the branch the 3.3 code labels `hard`.

```python
import numpy as np
from nalib import quasinewton as qn

rng = np.random.default_rng(42)
n = 6
frame, _ = np.linalg.qr(rng.normal(size=(n, n)))
values = np.concatenate([[-1.7], rng.uniform(0.5, 5.0, size=n - 1)])
B = frame @ np.diag(values) @ frame.T
B = 0.5 * (B + B.T)
worst = np.linalg.eigh(B)[1][:, 0]
g = rng.normal(size=n)
print(f"the smallest eigenvalue is {float(np.linalg.eigvalsh(B)[0]):.4f}")
print(f"{'radius':>9}{'|p|':>10}{'on the boundary':>18}{'lam':>10}"
      f"{'lam + smallest':>17}{'alignment with v_min':>23}")
for radius in (0.2, 1.0, 5.0, 50.0, 500.0):
    out = more_sorensen(g, B, radius)
    step = out["step"]
    aligned = abs(float(step @ worst)) / float(np.linalg.norm(step))
    print(f"{radius:>9.1f}{float(np.linalg.norm(step)):>10.4f}"
          f"{str(abs(float(np.linalg.norm(step)) - radius) < 1e-8 * radius):>18}"
          f"{out['lam']:>10.4f}{out['lam'] + float(np.linalg.eigvalsh(B)[0]):>17.6f}"
          f"{aligned:>23.6f}")
```

Every step is on the boundary, $\lambda$ falls toward $-\lambda_{\min}$ from above as the radius
grows, and the alignment with the most negative curvature direction rises toward $1$. That is the
mechanism by which a trust region escapes a saddle: **as the ball grows, the step turns into the
negative curvature direction on its own**, with no eigenvalue computation and no shift chosen by
hand.

### 5.3 When the memory hurts

The intuition is that stale pairs describe curvature that no longer applies. Testing it on the
obvious candidate does not support it.

```python
import numpy as np
from nalib import quasinewton as qn
from nalib.optimize import rosenbrock

memories = (2, 3, 4, 5, 8)
rng = np.random.default_rng(42)
for n in (10, 30, 100):
    problem = rosenbrock(n)
    tally = {m: [] for m in memories}
    shorter_won = 0
    for _ in range(20):
        start = np.ones(n) + rng.uniform(-1.5, 1.5, size=n)
        counts = {}
        for m in memories:
            out = qn.lbfgs(problem, start=start, memory=m)
            counts[m] = out["steps"] if out["converged"] else None
            if counts[m] is not None:
                tally[m].append(counts[m])
        if counts[2] is not None and counts[3] is not None and counts[2] < counts[3]:
            shorter_won += 1
    print(f"n = {n:>4}: " + "  ".join(f"m={m}:{np.mean(tally[m]):7.1f}" for m in memories)
          + f"   m=2 beat m=3 on {shorter_won} of 20 starts")
```

**On Rosenbrock a longer memory is better on average at every size.** The means fall monotonically
and $m = 2$ beats $m = 3$ on only $2$ to $4$ of $20$ starts. Individual runs are non-monotone in $m$,
which is what makes the effect look real in a single measurement, but averaging removes it.

A construction that does work needs the curvature to change by orders of magnitude along the path,
so that old pairs are not merely stale but wrongly scaled.

```python
import numpy as np
from nalib import quasinewton as qn


def flat_shelf(n, width=0.8):
    """A near flat shelf between the start and the minimum, so pairs differ hugely in scale."""
    weights = np.linspace(1.0, 4.0, n)

    def shelf(t):
        return np.tanh((t - 3.0) / width)

    def f(x):
        v = np.asarray(x, dtype=float).ravel()
        return float(np.sum(weights * v ** 2) * 1e-3 + np.sum(shelf(v)) + v.size)

    def gradient(x):
        v = np.asarray(x, dtype=float).ravel()
        return 2e-3 * weights * v + (1.0 - shelf(v) ** 2) / width

    def hessian(x):
        v = np.asarray(x, dtype=float).ravel()
        s = shelf(v)
        return np.diag(2e-3 * weights + 2.0 * s * (s * s - 1.0) / width ** 2)

    return {"f": f, "gradient": gradient, "hessian": hessian, "dimension": n,
            "start": np.full(n, 8.0), "minimizers": [np.zeros(n)], "minimum": 0.0,
            "convex": False, "name": f"a flat shelf, {n} variables"}


memories = (1, 2, 3, 5, 8, 12, 20, 40)
rng = np.random.default_rng(42)
for n in (20, 60):
    tally = {m: [] for m in memories}
    for _ in range(10):
        problem = flat_shelf(n)
        problem["start"] = rng.uniform(6.0, 10.0, size=n)
        for m in memories:
            tally[m].append(qn.lbfgs(problem, memory=m, max_steps=4000)["steps"])
    print(f"n = {n:>3}, mean steps over 10 starts")
    print("   " + "  ".join(f"m={m}:{np.mean(tally[m]):7.1f}" for m in memories))
```

**Here a longer memory is reliably worse.** At twenty variables the mean rises steadily from $28.0$
at $m = 2$ to $42.7$ at $m = 40$, $52$ per cent slower, and the rise is monotone from $m = 5$ onward.
At sixty variables $m = 40$ costs $48.2$ against $28.0$, and two memories produce runs that stall
outright.

The mechanism is visible in the stored pairs.

```python
import numpy as np
from nalib import gradient as gr

problem = flat_shelf(20)
x = problem["start"].copy()
slope = problem["gradient"](x)
print(f"{'step':>5}{'|g|':>12}{'s.y':>13}{'rho = 1/s.y':>15}{'y.y':>13}"
      f"{'gamma = s.y/y.y':>18}")
pairs = []
shown = 14
for k in range(shown):
    q = slope.copy()
    alphas = []
    for s, y, rho in reversed(pairs):
        alpha = rho * float(s @ q)
        alphas.append(alpha)
        q = q - alpha * y
    if pairs:
        s, y, _ = pairs[-1]
        q = q * (float(s @ y) / float(y @ y))
    for (s, y, rho), alpha in zip(pairs, reversed(alphas)):
        q = q + (alpha - rho * float(y @ q)) * s
    direction = -q
    if float(slope @ direction) >= 0.0:
        direction = -slope
    out = gr.strong_wolfe(problem["f"], problem["gradient"], x, direction, wolfe_c2=0.9)
    s = out["step"] * direction
    fresh = problem["gradient"](x + s)
    y = fresh - slope
    curvature = float(s @ y)
    print(f"{k:>5}{float(np.linalg.norm(slope)):>12.3e}{curvature:>13.3e}"
          f"{1.0 / curvature:>15.3e}{float(y @ y):>13.3e}"
          f"{curvature / float(y @ y):>18.3e}")
    if curvature > 1e-14:
        pairs.append((s, y, 1.0 / curvature))
        pairs = pairs[-40:]
    x, slope = x + s, fresh
```

The weight $\rho = 1/(s^{T}y)$ grows from $5$ to $1.4\times10^{8}$ over fourteen steps, **eight orders
of magnitude**, while $\gamma$ stays around $100$ throughout.

That is what the stale pairs are doing. The two loop recursion adds a term $(\alpha_i - \beta)s_i$
for each stored pair, and $\alpha_i$ is proportional to $\rho_i$. A pair recorded on the steep
entry into the shelf carries $\rho \approx 5$; a pair recorded near the minimum carries
$\rho \approx 10^{8}$. Mixing them means the recursion is combining curvature estimates that differ
by a factor of a hundred million, and the ones from the wrong region are not just uninformative but
actively wrong about the scale.

A short memory discards them. That is the whole content of the effect, and it explains the shape of
the answer: **the memory hurts exactly when the problem's curvature varies over orders of magnitude
along the path**, and not otherwise. On Rosenbrock, whose curvature changes direction along the
valley but stays within a factor of a few thousand in size, it does not.

The practical consequence is a diagnostic rather than a rule. If $s^{T}y$ across the stored pairs
spans many orders of magnitude, the memory is holding information from a different problem, and
either a shorter memory or a restart is worth trying. If it does not, more memory is safe.

---

## Lesson 88, Constrained Optimization

### 1.1 The KKT conditions

For $\min f(x)$ subject to $c(x) \le 0$, with $L = f + \lambda^{T}c$, a solution $x^{*}$ with
multipliers $\lambda$ satisfies

$$
\textbf{stationarity}\quad \nabla f + J^{T}\lambda = 0 , \qquad
\textbf{feasibility}\quad c(x^{*}) \le 0 ,
$$
$$
\textbf{sign}\quad \lambda \ge 0 , \qquad
\textbf{complementary slackness}\quad \lambda_i\, c_i(x^{*}) = 0 \text{ for every } i .
$$

Each one rules out a different thing.

**Stationarity** rules out a remaining descent direction that stays feasible. Without it the point is
not even a local minimum along the constraint.

**Feasibility** rules out an answer outside the set. It sounds obvious, but the penalty and barrier
methods of sections 3 and 4 both violate it at every finite stage, in opposite directions, and that
is the main thing to know about them.

**The sign** rules out a constraint pushing the wrong way. A negative $\lambda_i$ would mean the
constraint is holding the iterate **back** from a direction that increases $f$, which is not a
constraint doing its job. For an equality constraint there is no such direction, so the sign is free,
and the lesson's `equality_problem` has multiplier $-2/n$.

**Complementary slackness** rules out a price on a constraint that is not binding. If $c_i < 0$ there
is room to spare, so relaxing it further changes nothing and its price must be zero.

### 1.2 Why an inactive constraint has a zero multiplier

Two ways to see it, and both are worth having.

**From complementary slackness.** $\lambda_i c_i = 0$ with $c_i < 0$ forces $\lambda_i = 0$
immediately. That is the algebraic answer and it is a definition rather than an explanation.

**From the geometry.** If $c_i(x^{*}) < 0$ then by continuity $c_i$ stays negative in a whole
neighbourhood of $x^{*}$, so inside that neighbourhood the problem is exactly the problem with
constraint $i$ deleted. A local condition cannot mention something that has no effect locally.

That is also why 5.1's reading works: $\lambda_i$ is the rate at which the optimal value improves as
constraint $i$ is relaxed, and relaxing a constraint that is not binding improves nothing.

The practical consequence is that the multipliers **identify the active set**, which is what an
active set method in exercise 3.2 exploits directly, and what a barrier method in 1.4 recovers only
in the limit.

### 1.3 Why the penalty needs $\mu\to\infty$ and the augmented Lagrangian does not

**The penalty.** Minimizing $f + \tfrac{\mu}{2}\lVert c\rVert^{2}$ gives stationarity

$$
\nabla f + \mu J^{T}c(x_\mu) = 0 .
$$

Comparing with the true condition $\nabla f + J^{T}\lambda = 0$ shows that the penalty is
manufacturing its multiplier out of the violation: $\lambda \approx \mu\, c(x_\mu)$. To get a
non-zero $\lambda$ with $c \to 0$ the weight must satisfy $\mu\,c \to \lambda \ne 0$, so
$c \sim \lambda/\mu$ and

$$
\boxed{\;\text{the violation is } O(1/\mu)\;}
$$

**and it is never zero at finite $\mu$.** The pull that holds the iterate against the constraint is
generated by the violation itself, so removing the violation removes the pull.

**The augmented Lagrangian.** Minimizing $f + \lambda^{T}c + \tfrac{\mu}{2}\lVert c\rVert^{2}$ gives

$$
\nabla f + J^{T}\big(\lambda + \mu\, c(x)\big) = 0 .
$$

Now the multiplier term supplies the pull directly. At the **true** $\lambda$ the point $x^{*}$ with
$c(x^{*}) = 0$ satisfies this exactly, for any $\mu$ large enough to make the inner problem convex.
So the weight has a threshold rather than a limit, and the iteration $\lambda \leftarrow \lambda +
\mu c$ converges to the right value with $\mu$ fixed. Exercise 4.2 measures both thresholds.

The cost of the penalty's route is in the second derivative: the Hessian is
$\nabla^{2}f + \mu J^{T}J + \ldots$, whose condition number grows like $\mu$. So the penalty method
trades accuracy for conditioning at a fixed exchange rate, and exercise 4.1 finds all three points
where the trade breaks down.

### 1.4 How a barrier method produces the multipliers

The barrier subproblem is

$$
\min\; f(x) - \mu \sum_i \log(-c_i(x)) ,
$$

whose stationarity condition is

$$
\nabla f + \sum_i \frac{\mu}{-c_i(x)}\, \nabla c_i = 0 .
$$

Comparing with $\nabla f + J^{T}\lambda = 0$ term by term identifies

$$
\boxed{\;\lambda_i = \frac{\mu}{-c_i(x_\mu)}\;}
$$

so **the multipliers are read off the slacks**, with no separate estimate needed. Exercise 2.4 does
the derivation properly.

Two consequences fall out. The product $\lambda_i(-c_i) = \mu$ for every $i$, which is complementary
slackness perturbed by $\mu$; letting $\mu \to 0$ recovers it exactly. And the multiplier is large
where the slack is small, so an active constraint gets a large price and an inactive one gets
$\mu/\lvert c_i\rvert \to 0$, which is 1.2 arriving automatically.

The catch is arithmetic. For an active constraint both $\mu$ and $-c_i$ go to zero, so the quotient
is a ratio of two small numbers, and below some $\mu$ it is noise. Exercise 5.2 measures exactly
where that happens.

### 1.5 What a projection is

A projection $P_C$ returns the **nearest** point of $C$:

$$
P_C(v) = \arg\min_{x \in C} \tfrac12 \lVert x - v\rVert^{2} .
$$

Returning some feasible point is not a projection. Three properties follow from nearest, and each is
used somewhere in this part.

**It is idempotent**, $P_C(P_C(v)) = P_C(v)$, because a point already in $C$ is its own nearest
point. That is what makes the fixed point test $x = P(x - \alpha g)$ meaningful.

**It is nonexpansive**, $\lVert P_C(u) - P_C(v)\rVert \le \lVert u - v\rVert$, for convex $C$. That
is what makes projected gradient converge at the same rate as unprojected gradient: the projection
cannot amplify an error.

**The residual is normal to the set**, $\langle v - P_C(v),\; x - P_C(v)\rangle \le 0$ for all
$x \in C$. That is the variational inequality that makes $x = P(x - \alpha g)$ equivalent to the KKT
conditions.

And it is the proximal operator of the indicator function of $C$, which exercise 5.3 develops and
lesson 90 builds on.

### 2.1 Deriving the KKT conditions

**The setup.** At a local minimum $x^{*}$ with active set $A = \{i : c_i(x^{*}) = 0\}$, no feasible
descent direction exists. A direction $d$ is **feasible to first order** if
$\nabla c_i^{T}d \le 0$ for every $i \in A$, and **descent** if $\nabla f^{T}d < 0$. So

$$
\text{no } d \text{ with } \nabla c_i^{T}d \le 0\;(i \in A) \text{ and } \nabla f^{T}d < 0 .
$$

**Farkas' lemma** turns that into an algebraic statement. It says that exactly one of the following
holds for a matrix $M$ and vector $q$: either $Md \le 0$ and $q^{T}d > 0$ has a solution $d$, or
$M^{T}y = q$ with $y \ge 0$ has a solution $y$.

Take $M$ the rows $\nabla c_i^{T}$ for $i \in A$ and $q = -\nabla f$. The first alternative is
exactly "a feasible descent direction exists", which is ruled out. Therefore the second holds:

$$
\sum_{i \in A} \lambda_i \nabla c_i = -\nabla f , \qquad \lambda_i \ge 0 ,
$$

which is stationarity with the sign condition. Setting $\lambda_i = 0$ for $i \notin A$ gives
complementary slackness, and feasibility holds because $x^{*}$ is in the set.

**The constraint qualification.** The step from "no feasible descent direction" to "no $d$ with
$\nabla c_i^{T}d \le 0$" needs the linearized cone to match the real one, which is what LICQ (the
active gradients linearly independent) or Slater's condition supplies. Without it the conditions can
fail at a genuine minimum. Exercise 5.2 constructs a case where LICQ fails and traces the
consequence, which turns out to be milder than one might expect.

```python
import numpy as np
from nalib import constrained as co

out = co.the_conditions_hold_at_the_answer()
print(f"{'problem':>50}{'stationarity':>14}{'feasibility':>13}{'worst':>10}")
for row in out["rows"]:
    print(f"{row['name'][:50]:>50}{row['stationarity']:>14.1e}"
          f"{row['feasibility']:>13.1e}{row['worst']:>10.1e}")
print(f"\nevery stated answer satisfies them: {out['every_answer_satisfies_them']}, "
      f"worst residual {out['worst_residual']:.1e}")

strict = co.strict_complementarity_is_not_automatic()
print(f"\n{'target':>14}{'active':>12}{'multipliers there':>24}{'strict':>9}")
for row in strict["rows"]:
    print(f"{str(row['target']):>14}{str(row['active_constraints']):>12}"
          f"{str([round(v, 8) for v in row['multipliers_there']]):>24}"
          f"{str(row['strict']):>9}")
print(f"one target of each kind: {strict['one_of_each']}")
```

Every closed form answer in the module satisfies every condition to rounding, which is the check that
the test problems are what they claim before any method touches them.

The second table is the fine print. **Strict complementarity**, which asks that every active
constraint have a strictly positive multiplier, is not part of the KKT conditions and it does not
always hold. At the target $(2,1)$ the answer is a vertex where two constraints are tight and one of
them carries a multiplier of zero. Most convergence theorems assume that away, and exercise 4.2 of
the lesson measures what it costs on the central path.

### 2.2 The penalty method's error and conditioning

Take $f(x) = \tfrac12 x^{T}Hx - b^{T}x$ with the linear constraint $a^{T}x = \beta$. The penalized
objective is

$$
\varphi_\mu(x) = \tfrac12 x^{T}Hx - b^{T}x + \tfrac{\mu}{2}(a^{T}x - \beta)^{2} ,
$$

with gradient and Hessian

$$
\nabla\varphi_\mu = Hx - b + \mu\,a\,(a^{T}x - \beta) , \qquad
\nabla^{2}\varphi_\mu = H + \mu\, a a^{T} .
$$

**The conditioning.** $aa^{T}$ has one non-zero eigenvalue $\lVert a\rVert^{2}$, so $H + \mu aa^{T}$
has one eigenvalue growing like $\mu\lVert a\rVert^{2}$ and the rest bounded. Therefore

$$
\kappa(\nabla^{2}\varphi_\mu) = O(\mu) ,
$$

and the growth is exactly linear, not merely bounded by a linear function.

**The error.** Setting the gradient to zero, $x_\mu = (H + \mu aa^{T})^{-1}(b + \mu\beta a)$. Apply
Sherman-Morrison to $(H + \mu aa^{T})^{-1}$ and expand in $1/\mu$; the algebra collapses to

$$
a^{T}x_\mu - \beta = \frac{-\lambda}{\mu} + O(\mu^{-2}) , \qquad
\lVert x_\mu - x^{*}\rVert = O(1/\mu) ,
$$

where $\lambda$ is the true multiplier. The short version is the one from 1.3: the penalty produces
its multiplier as $\mu \times \text{violation}$, so a fixed multiplier needs a violation
proportional to $1/\mu$.

**The trade.** Multiplying the two, the achievable accuracy in floating point is roughly

$$
\text{error} \;\approx\; \max\!\left(\frac{C}{\mu},\; \varepsilon\,\kappa\right)
\;\approx\; \max\!\left(\frac{C}{\mu},\; \varepsilon\, c\, \mu\right) ,
$$

minimized where the two are equal. Exercise 4.1 measures where that is, and it is not where the naive
estimate puts it.

```python
import numpy as np
from nalib import constrained as co

problem = co.equality_problem(5)
print(f"{'weight':>10}{'violation':>13}{'weight x violation':>21}"
      f"{'true multiplier':>18}{'condition':>13}{'condition / weight':>21}")
for power in range(1, 9):
    mu = 10.0 ** power
    inner = co.penalty_problem(problem, mu)
    x = np.linalg.solve(np.asarray(inner["hessian"](problem["start"]), dtype=float),
                        -np.asarray(inner["gradient"](np.zeros(problem["dimension"])),
                                    dtype=float))
    gap = float(np.asarray(problem["constraint"](x), dtype=float)[0])
    cond = float(np.linalg.cond(np.asarray(inner["hessian"](x), dtype=float)))
    print(f"{mu:>10.0e}{gap:>13.3e}{mu * gap:>21.9f}"
          f"{float(problem['multipliers'][0]):>18.9f}{cond:>13.3e}{cond / mu:>21.4f}")
```

Both predictions are exact rather than asymptotic: $\mu$ times the violation equals the true
multiplier to nine digits at every weight, and the condition number divided by $\mu$ is constant.

### 2.3 The augmented Lagrangian at the true multiplier

**Claim.** If $(x^{*}, \lambda^{*})$ satisfies the KKT conditions with equality constraints
$c(x^{*}) = 0$ and the second order sufficient conditions hold, then for all $\mu$ above a threshold
$x^{*}$ is an unconstrained local minimum of

$$
L_\mu(x) = f(x) + \lambda^{*T}c(x) + \tfrac{\mu}{2}\lVert c(x)\rVert^{2} .
$$

**Stationarity.** Differentiating,

$$
\nabla L_\mu(x) = \nabla f(x) + J(x)^{T}\big(\lambda^{*} + \mu\, c(x)\big) .
$$

At $x = x^{*}$ we have $c(x^{*}) = 0$, so this is $\nabla f + J^{T}\lambda^{*} = 0$ by stationarity of
the constrained problem. **The penalty term contributes nothing at the answer**, which is exactly
what the plain penalty could not achieve.

**Second order.** Differentiating again at $x^{*}$,

$$
\nabla^{2}L_\mu(x^{*}) = \nabla^{2}_{xx}L(x^{*},\lambda^{*}) + \mu\, J^{T}J .
$$

The second order sufficient condition says the first term is positive definite on the null space of
$J$. On the orthogonal complement, $J^{T}J$ is positive definite, so adding $\mu J^{T}J$ with $\mu$
large enough makes the whole matrix positive definite. **The threshold is where that happens**, and
it is finite because the first term is bounded.

$\square$

The whole method is this observation plus a way to find $\lambda^{*}$, which is the update
$\lambda \leftarrow \lambda + \mu c(x)$, itself a gradient ascent step on the dual. Exercise 4.2 finds
its exact rate.

```python
import numpy as np
from nalib import constrained as co

problem = co.equality_problem(4)
truth = problem["minimizer"]
lam = problem["multipliers"]
print(f"{'mu':>8}{'gradient of L_mu at the answer':>34}"
      f"{'smallest eigenvalue of its Hessian':>38}")
for mu in (0.0, 0.5, 1.0, 5.0, 100.0):
    gap = np.asarray(problem["constraint"](truth), dtype=float)
    jac = np.asarray(problem["jacobian"](truth), dtype=float)
    slope = (np.asarray(problem["gradient"](truth), dtype=float)
             + jac.T @ (lam + mu * gap))
    hess = (np.asarray(problem["hessian"](truth), dtype=float) + mu * (jac.T @ jac))
    print(f"{mu:>8.1f}{float(np.linalg.norm(slope)):>34.2e}"
          f"{float(np.linalg.eigvalsh(hess)[0]):>38.6f}")
print("\nand with the wrong multiplier, the gradient is not zero:")
for wrong in (0.0, 0.5 * float(lam[0]), 2.0 * float(lam[0])):
    jac = np.asarray(problem["jacobian"](truth), dtype=float)
    slope = np.asarray(problem["gradient"](truth), dtype=float) + jac.T @ np.array([wrong])
    print(f"  lambda = {wrong:>8.4f} against the true {float(lam[0]):.4f}: "
          f"gradient {float(np.linalg.norm(slope)):.4f}")
```

At the true multiplier the gradient is zero for **every** $\mu$ including zero, and the Hessian is
positive definite. With any other multiplier the gradient is not zero, which is the sense in which
the augmented Lagrangian solves the right problem only when it knows the price.

### 2.4 The central path

The barrier subproblem is $\min f(x) - \mu\sum_i \log(-c_i(x))$ over the strictly feasible set. Its
gradient is

$$
\nabla f(x) - \mu\sum_i \frac{1}{-c_i(x)}\,\nabla(-c_i)(x)
= \nabla f(x) + \sum_i \frac{\mu}{-c_i(x)}\,\nabla c_i(x) ,
$$

using $\nabla(-c_i) = -\nabla c_i$. Setting it to zero and matching against
$\nabla f + \sum_i \lambda_i \nabla c_i = 0$ gives

$$
\boxed{\;\lambda_i(\mu) = \frac{\mu}{-c_i(x_\mu)}\;}
$$

term by term, which is legitimate whenever the active gradients are independent.

**What it means.** Rearranged, $\lambda_i\,\big(-c_i(x_\mu)\big) = \mu$ for every $i$. That is
complementary slackness with $0$ on the right replaced by $\mu$: every constraint carries a small
price and every price times its slack is the same number. Driving $\mu \to 0$ drives the pairs to the
exact condition, and the trajectory $x_\mu$ is the **central path**.

**The sign is automatic.** $\mu > 0$ and $-c_i > 0$ inside the feasible set, so $\lambda_i > 0$
always. An interior point method never has to enforce the sign condition; it gets it from staying
inside.

```python
import numpy as np
from nalib import constrained as co

out = co.the_central_path()
print(f"{'weight':>10}{'x on the path':>26}{'multiplier estimate':>32}"
      f"{'inside':>8}{'distance':>11}{'multiplier error':>18}")
for row in out["rows"]:
    print(f"{row['weight']:>10.0e}{np.array2string(row['x'], precision=5):>26}"
          f"{np.array2string(row['multiplier_estimate'], precision=4):>32}"
          f"{str(row['strictly_feasible']):>8}{row['distance']:>11.2e}"
          f"{row['multiplier_error']:>18.2e}")
print(f"\nevery iterate strictly inside: {out['every_point_strictly_feasible']}")
print(f"the multipliers arrive without being asked for: "
      f"{out['the_multipliers_come_free']}")
print(f"the exact answer is {np.array2string(out['exact_minimizer'], precision=6)} "
      f"with multipliers {np.array2string(out['exact_multipliers'], precision=4)}")

print("\nand the identity lam_i times the slack equals the weight, row by row:")
problem = co.triangle_problem(out["target"])
for row in out["rows"][::2]:
    slack = -np.asarray(problem["constraint"](row["x"]), dtype=float)
    products = row["multiplier_estimate"] * slack
    print(f"  weight {row['weight']:.0e}: products "
          f"{np.array2string(products, precision=8)}")
```

Every iterate is strictly inside the set, the path ends at the answer, and the multiplier estimate
converges to the true one without ever being solved for. The last table is the identity itself: the
product $\lambda_i \times (-c_i)$ equals the weight for every constraint at every point of the path,
to rounding.

### 2.5 The simplex projection

**The problem.** $P(v) = \arg\min_{x} \tfrac12\lVert x - v\rVert^{2}$ subject to
$\sum_i x_i = \sigma$ and $x \ge 0$.

**The Lagrangian.** With a multiplier $\theta$ for the sum and $\nu \ge 0$ for the sign,

$$
L = \tfrac12\lVert x - v\rVert^{2} + \theta\left(\sum_i x_i - \sigma\right) - \nu^{T}x ,
$$

and stationarity in $x_i$ gives $x_i - v_i + \theta - \nu_i = 0$, that is
$x_i = v_i - \theta + \nu_i$.

**Complementary slackness does the rest.** If $x_i > 0$ then $\nu_i = 0$ and
$x_i = v_i - \theta$. If $x_i = 0$ then $\nu_i \ge 0$ forces $v_i - \theta \le 0$. Both cases are
covered by

$$
\boxed{\;x_i = \max(v_i - \theta,\; 0)\;}
$$

**The equation for $\theta$.** The sum constraint gives

$$
g(\theta) \;=\; \sum_i \max(v_i - \theta,\, 0) \;=\; \sigma .
$$

$g$ is continuous, piecewise linear, non-increasing, with $g(\min_i v_i - \sigma) \ge \sigma$ and
$g(\max_i v_i) = 0$, so a root exists and is unique on the region where $g$ is strictly decreasing.
Sorting $v$ makes the breakpoints explicit and the root is found in one pass, which is why the
standard algorithm is $O(n\log n)$.

```python
import numpy as np
from nalib import constrained as co

rng = np.random.default_rng(42)
print(f"{'n':>5}{'total':>8}{'theta from the equation':>26}{'sum of the answer':>20}"
      f"{'matches the library':>21}")
for n in (1, 4, 20, 200):
    for total in (1.0, 3.0):
        point = rng.normal(scale=2.0, size=n)
        low = float(np.min(point)) - total
        high = float(np.max(point))
        for _ in range(200):                       # bisection on the piecewise linear g
            mid = 0.5 * (low + high)
            if float(np.sum(np.maximum(point - mid, 0.0))) > total:
                low = mid
            else:
                high = mid
        theta = 0.5 * (low + high)
        mine = np.maximum(point - theta, 0.0)
        theirs = co.project_onto_simplex(point, total)
        print(f"{n:>5}{total:>8.1f}{theta:>26.12f}{float(np.sum(mine)):>20.12f}"
              f"{float(np.max(np.abs(mine - theirs))):>21.2e}")
```

The bisection root reproduces the library's projection to rounding at every size and total, and the
sum of the answer equals the requested total to twelve digits.

### 3.1 Sequential quadratic programming

SQP replaces the problem at each step by a quadratic model with **linearized** constraints:

$$
\min_p\; \nabla f^{T}p + \tfrac12 p^{T}Hp \quad\text{subject to}\quad c + Jp = 0 \;(\text{active}) ,
$$

whose KKT system is the one linear solve

$$
\begin{pmatrix} H & J^{T} \\ J & 0 \end{pmatrix}
\begin{pmatrix} p \\ \nu \end{pmatrix} =
\begin{pmatrix} -\nabla f \\ -c \end{pmatrix} .
$$

That is Newton's method applied to the KKT conditions themselves, so the multipliers come out of the
same solve as the step.

```python
import numpy as np
from nalib import constrained as co
from nalib.quasinewton import bfgs


def sqp(problem, rounds=60, tol=1e-12):
    """Newton on the KKT system, with an active set chosen inside for inequalities."""
    x = np.asarray(problem["start"], dtype=float).ravel()
    n = x.size
    total = np.asarray(problem["constraint"](x), dtype=float).ravel().size
    lam = np.zeros(total)
    history = []
    for k in range(rounds):
        slope = np.asarray(problem["gradient"](x), dtype=float).ravel()
        gaps = np.asarray(problem["constraint"](x), dtype=float).ravel()
        jac = np.asarray(problem["jacobian"](x), dtype=float)
        hess = np.asarray(problem["hessian"](x), dtype=float)
        history.append(co.kkt_residual(problem, x, lam)["worst"])
        if history[-1] <= tol:
            break
        if problem["kind"] == "equality":
            active = np.arange(total)
        else:
            active = np.flatnonzero((gaps > -1e-9) | (lam > 1e-12))
        for _ in range(total + 1):
            if active.size:
                rows, values = jac[active], gaps[active]
                block = np.vstack([np.hstack([hess, rows.T]),
                                   np.hstack([rows, np.zeros((active.size, active.size))])])
                answer = np.linalg.lstsq(block, np.concatenate([-slope, -values]),
                                         rcond=None)[0]
                move, prices = answer[:n], answer[n:]
                if problem["kind"] == "inequality" and np.any(prices < -1e-10):
                    active = np.delete(active, int(np.argmin(prices)))
                    continue
                lam = np.zeros(total)
                lam[active] = prices
            else:
                move, lam = np.linalg.solve(hess, -slope), np.zeros(total)
            break
        x = x + move
    return {"x": x, "multipliers": lam, "steps": k,
            "history": np.asarray(history)}


print(f"{'problem':>46}{'SQP steps':>11}{'SQP error':>12}{'AL rounds':>11}"
      f"{'AL error':>11}")
for build in (lambda: co.equality_problem(2), lambda: co.equality_problem(10),
              lambda: co.triangle_problem((1.5, 1.5)),
              lambda: co.triangle_problem((2.0, 1.0))):
    problem = build()
    fast = sqp(problem)
    slow = co.augmented_lagrangian(problem)
    print(f"{problem['name'][:46]:>46}{fast['steps']:>11}"
          f"{float(np.linalg.norm(fast['x'] - problem['minimizer'])):>12.2e}"
          f"{len(slow['rows']):>11}{slow['error']:>11.2e}")
    print(f"{'  KKT residual by step:':>46} "
          + " ".join(f"{v:.1e}" for v in fast["history"][:4]))
```

**SQP is exact in one or two steps on all four problems.** That is not luck: every objective here is
quadratic and every constraint linear, so the model **is** the problem and one Newton step solves it.
The second step, where it appears, is the one that identifies the active set.

**The augmented Lagrangian solves the equality problems and fails on the triangle**, with errors of
$0.24$ and $0.75$. That is not a defect in the method; the lesson's `augmented_lagrangian` implements
the **equality** form, whose penalty $\tfrac{\mu}{2}\lVert c\rVert^{2}$ punishes $c_i < 0$ just as
hard as $c_i > 0$. Applied to inequalities it solves a different problem, silently.

The inequality form replaces the penalty by one that switches off where the constraint is slack.

```python
import numpy as np
from nalib import constrained as co
from nalib.quasinewton import bfgs


def augmented_inequality(problem, weight=10.0, rounds=12, growth=10.0):
    """Rockafellar's form: the penalty acts through max(0, lam + mu c), so slack rows drop out."""
    mu = float(weight)
    total = np.asarray(problem["constraint"](problem["start"]), dtype=float).ravel().size
    lam = np.zeros(total)
    x = np.asarray(problem["start"], dtype=float).ravel()
    rows = []
    for _ in range(rounds):
        def f(v, lam=lam, mu=mu):
            gaps = np.asarray(problem["constraint"](v), dtype=float).ravel()
            push = np.maximum(0.0, lam + mu * gaps)
            return float(problem["f"](v)) + float(np.sum(push ** 2 - lam ** 2)) / (2.0 * mu)

        def gradient(v, lam=lam, mu=mu):
            gaps = np.asarray(problem["constraint"](v), dtype=float).ravel()
            jac = np.asarray(problem["jacobian"](v), dtype=float)
            return (np.asarray(problem["gradient"](v), dtype=float).ravel()
                    + jac.T @ np.maximum(0.0, lam + mu * gaps))

        def hessian(v, lam=lam, mu=mu):
            gaps = np.asarray(problem["constraint"](v), dtype=float).ravel()
            jac = np.asarray(problem["jacobian"](v), dtype=float)
            on = (lam + mu * gaps > 0.0).astype(float)
            return (np.asarray(problem["hessian"](v), dtype=float)
                    + mu * (jac.T @ (on[:, None] * jac)))

        out = bfgs({"f": f, "gradient": gradient, "hessian": hessian, "start": x,
                    "dimension": problem["dimension"]}, tol=1e-13, max_steps=20000)
        x = out["x"]
        gaps = np.asarray(problem["constraint"](x), dtype=float).ravel()
        lam = np.maximum(0.0, lam + mu * gaps)
        rows.append(float(np.max(np.maximum(gaps, 0.0))))
        if rows[-1] > 0.25 * (rows[-2] if len(rows) > 1 else np.inf):
            mu *= growth
    return {"x": x, "multipliers": lam, "rounds": len(rows), "final_weight": mu,
            "error": float(np.linalg.norm(x - problem["minimizer"]))}


for target in ((1.5, 1.5), (2.0, 1.0)):
    problem = co.triangle_problem(target)
    fast = sqp(problem)
    right = augmented_inequality(problem)
    wrong = co.augmented_lagrangian(problem)
    print(f"target {target}")
    print(f"  SQP:                     {fast['steps']} steps, error "
          f"{float(np.linalg.norm(fast['x'] - problem['minimizer'])):.1e}")
    print(f"  inequality augmented:    {right['rounds']} rounds, error "
          f"{right['error']:.1e}")
    print(f"  equality form misapplied: error {wrong['error']:.1e}")
    print(f"  true multipliers {np.array2string(problem['multipliers'], precision=4)}, "
          f"SQP {np.array2string(fast['multipliers'], precision=4)}")
```

With the right form both methods find the answer and the correct multipliers. **SQP needs two steps
and the augmented Lagrangian needs twelve rounds**, each of which is a full unconstrained minimization,
so the gap in real work is much larger than twelve to two.

That is the general picture. SQP converges quadratically and is the method of choice when second
derivatives are available and the problem is not too large. The augmented Lagrangian converges
linearly but needs only an unconstrained solver, never forms a KKT matrix, and does not have to
guess an active set, which makes it the robust choice when the problem is large, badly conditioned,
or supplied only as a black box.

### 3.2 An active set method

An active set method **guesses** which constraints are tight, solves the equality constrained problem
on that working set, and corrects the guess: add a constraint when a step would violate it, drop one
when its multiplier goes negative.

```python
import numpy as np


def random_qp(n=6, m=10, seed=42):
    """A strictly convex QP with m inequality rows, strictly feasible at the origin."""
    rng = np.random.default_rng(int(seed))
    frame, _ = np.linalg.qr(rng.normal(size=(n, n)))
    Q = frame @ np.diag(rng.uniform(0.5, 4.0, size=n)) @ frame.T
    Q = 0.5 * (Q + Q.T)
    c = rng.normal(size=n)
    A = rng.normal(size=(m, n))
    b = rng.uniform(0.5, 2.0, size=m)              # b > 0 so the origin is inside
    return {"Q": Q, "c": c, "A": A, "b": b, "n": n, "m": m,
            "f": lambda v: 0.5 * float(np.asarray(v, dtype=float) @ Q
                                       @ np.asarray(v, dtype=float))
                           + float(c @ np.asarray(v, dtype=float)),
            "name": f"random QP, {n} variables and {m} rows"}


def active_set_qp(qp, rounds=400):
    Q, c, A, b = qp["Q"], qp["c"], qp["A"], qp["b"]
    n, m = qp["n"], qp["m"]
    x = np.zeros(n)
    working = []
    steps, prices = 0, np.zeros(0)
    for _ in range(rounds):
        steps += 1
        slope = Q @ x + c
        if working:
            rows = A[working]
            block = np.vstack([np.hstack([Q, rows.T]),
                               np.hstack([rows, np.zeros((len(working), len(working)))])])
            answer = np.linalg.lstsq(block,
                                     np.concatenate([-slope, np.zeros(len(working))]),
                                     rcond=None)[0]
            move, prices = answer[:n], answer[n:]
        else:
            move, prices = np.linalg.solve(Q, -slope), np.zeros(0)
        if float(np.linalg.norm(move)) < 1e-12:
            if working and float(np.min(prices)) < -1e-10:
                working.pop(int(np.argmin(prices)))       # a negative price: let it go
                continue
            break
        slack, along = b - A @ x, A @ move
        alpha, blocker = 1.0, None
        for i in range(m):                                # the longest feasible step
            if i in working or along[i] <= 1e-14:
                continue
            limit = slack[i] / along[i]
            if limit < alpha:
                alpha, blocker = limit, i
        x = x + alpha * move
        if blocker is not None:
            working.append(blocker)
    lam = np.zeros(m)
    if working:
        lam[working] = np.maximum(prices, 0.0)
    return {"x": x, "multipliers": lam, "steps": steps, "working": sorted(working)}


def barrier_qp(qp, start_weight=1.0, shrink=0.1, rounds=12, inner=200):
    """The log barrier, following the central path down by Newton steps."""
    Q, c, A, b = qp["Q"], qp["c"], qp["A"], qp["b"]
    x = np.zeros(qp["n"])
    mu, newton_steps = float(start_weight), 0
    for _ in range(rounds):
        for _ in range(inner):
            slack = b - A @ x
            slope = Q @ x + c + mu * (A.T @ (1.0 / slack))
            hess = Q + mu * (A.T @ ((1.0 / slack ** 2)[:, None] * A))
            move = np.linalg.solve(hess, -slope)
            t = 1.0
            while np.any(b - A @ (x + t * move) <= 0.0):   # stay strictly inside
                t *= 0.5
            base = (0.5 * float(x @ Q @ x) + float(c @ x)
                    - mu * float(np.sum(np.log(b - A @ x))))
            for _ in range(60):
                trial = x + t * move
                value = (0.5 * float(trial @ Q @ trial) + float(c @ trial)
                         - mu * float(np.sum(np.log(b - A @ trial))))
                if value <= base + 1e-4 * t * float(slope @ move):
                    break
                t *= 0.5
            x = x + t * move
            newton_steps += 1
            if -float(slope @ move) < 2e-12:               # the Newton decrement
                break
        mu *= shrink
    lam = (mu / shrink) / (b - A @ x)
    return {"x": x, "multipliers": lam, "newton_steps": newton_steps}


print(f"{'n':>5}{'m':>5}{'active set steps':>18}{'active rows':>13}"
      f"{'barrier Newton steps':>22}{'gap in x':>12}{'gap in f':>12}")
for n, m in ((2, 3), (6, 10), (10, 30), (20, 60)):
    qp = random_qp(n, m)
    fast = active_set_qp(qp)
    smooth = barrier_qp(qp)
    print(f"{n:>5}{m:>5}{fast['steps']:>18}{len(fast['working']):>13}"
          f"{smooth['newton_steps']:>22}"
          f"{float(np.linalg.norm(fast['x'] - smooth['x'])):>12.1e}"
          f"{abs(qp['f'](fast['x']) - qp['f'](smooth['x'])):>12.1e}")
```

**The two agree**, to $10^{-9}$ in $x$ and $10^{-10}$ in $f$, which is the check that both are
solving the stated problem.

**Their costs scale differently.** The active set method takes $2, 3, 7, 13$ iterations, one more
than the number of rows it ends up activating ($0, 1, 5, 11$). Each iteration is one equality
constrained solve. The barrier method takes $26, 63, 78, 83$ Newton steps, growing much more slowly.

The reason is structural. **An active set method has to find a subset**, and in the worst case there
are $2^{m}$ of them; it is fast when few constraints are active and it can warm start, which is why
it dominates in sequential settings like SQP's inner solve and portfolio rebalancing. **A barrier
method never chooses a subset**; it lets the multipliers sort themselves out along the central path,
and its cost is governed by how many times the weight must be reduced. Exercise 4.3 pushes both to
where the difference is decisive.

### 3.3 A primal-dual interior point method

The barrier method eliminates the multipliers and solves for $x$ alone. A primal-dual method keeps
$x$, the slacks $s = b - Ax$ and the multipliers $\lambda$ as independent unknowns and applies Newton
to the perturbed KKT system

$$
Qx + c + A^{T}\lambda = 0 , \qquad Ax + s = b , \qquad s_i \lambda_i = \sigma\,\mu ,
$$

with $\mu = s^{T}\lambda/m$ the current duality gap and $\sigma \in (0,1)$ a centring parameter. The
step is then cut so that $s$ and $\lambda$ stay positive.

```python
import numpy as np


def primal_dual_qp(qp, rounds=100, sigma=0.1, tol=1e-10):
    Q, c, A, b = qp["Q"], qp["c"], qp["A"], qp["b"]
    n, m = qp["n"], qp["m"]
    x = np.zeros(n)
    slack = b - A @ x
    lam = np.ones(m)
    steps = 0
    for _ in range(rounds):
        steps += 1
        gap = float(slack @ lam) / m
        dual = Q @ x + c + A.T @ lam
        primal = A @ x + slack - b
        if max(float(np.linalg.norm(dual)), float(np.linalg.norm(primal)), gap) < tol:
            break
        target = sigma * gap
        weight = lam / slack                          # eliminate the slack block
        left = Q + A.T @ (weight[:, None] * A)
        right = -(dual + A.T @ (weight * primal) + A.T @ (target / slack - lam))
        dx = np.linalg.solve(left, right)
        ds = -primal - A @ dx
        dl = (target - lam * slack - lam * ds) / slack
        step = 1.0
        for value, change in ((slack, ds), (lam, dl)):  # keep both strictly positive
            bad = change < 0.0
            if np.any(bad):
                step = min(step, 0.99 * float(np.min(-value[bad] / change[bad])))
        x, slack, lam = x + step * dx, slack + step * ds, lam + step * dl
    return {"x": x, "multipliers": lam, "steps": steps,
            "duality_gap": float(slack @ lam) / m}


print(f"{'n':>5}{'m':>6}{'primal-dual steps':>19}{'barrier Newton steps':>22}"
      f"{'speedup':>10}{'gap to the active set answer':>31}{'duality gap':>14}")
for n, m in ((2, 3), (6, 10), (10, 30), (20, 60), (40, 200)):
    qp = random_qp(n, m)
    fast = primal_dual_qp(qp)
    smooth = barrier_qp(qp)
    reference = active_set_qp(qp)
    print(f"{n:>5}{m:>6}{fast['steps']:>19}{smooth['newton_steps']:>22}"
          f"{smooth['newton_steps'] / max(fast['steps'], 1):>10.1f}"
          f"{float(np.linalg.norm(fast['x'] - reference['x'])):>31.1e}"
          f"{fast['duality_gap']:>14.1e}")
```

**The primal-dual method takes $12$ to $16$ steps and the count barely moves**, from a problem with
three constraints to one with two hundred. The barrier method takes $26$ to $94$, so the speedup runs
from $2.2$ to $5.9$ and grows with the problem.

Two things buy that. The Newton step moves $x$, $s$ and $\lambda$ **together**, so an error in the
multiplier estimate is corrected in the same step rather than in the next outer round. And the
centring parameter $\sigma$ sets the target for $\mu$ adaptively from the current gap rather than
following a fixed geometric schedule, so the method takes long steps when the iterate is well
centred.

The famous claim about interior point methods, that they solve a linear or quadratic program in
twenty to thirty iterations almost regardless of size, is what the table shows, and it is why they
displaced the simplex method for large problems.

### 3.4 Alternating projections

To find a point in $C \cap D$, project alternately: $x_{k+1} = P_D(P_C(x_k))$. For closed convex sets
that intersect, the iterates converge to a point of the intersection, at a linear rate set by the
**angle** between the sets.

```python
import numpy as np
from nalib import constrained as co


def alternating(first, second, start, rounds=100000, tol=1e-15):
    x = np.asarray(start, dtype=float).ravel()
    trail = [x.copy()]
    for k in range(rounds):
        moved = second(first(x))
        trail.append(moved.copy())
        if float(np.linalg.norm(moved - x)) <= tol * max(1.0, float(np.linalg.norm(moved))):
            break
        x = moved
    return np.asarray(trail), k


rng = np.random.default_rng(42)
print("two subspaces at a known angle: the rate should be the cosine squared")
print(f"{'n':>5}{'angle':>8}{'rounds':>9}{'measured rate':>16}{'cos^2':>12}{'ratio':>10}")
for n in (2, 8):
    for degrees in (60.0, 30.0, 10.0, 3.0):
        theta = np.radians(degrees)
        first_line = np.zeros(n)
        first_line[0] = 1.0
        second_line = np.zeros(n)
        second_line[0], second_line[1] = np.cos(theta), np.sin(theta)
        onto = lambda w, d: float(d @ w) * d
        trail, rounds = alternating(lambda w: onto(w, first_line),
                                    lambda w: onto(w, second_line),
                                    rng.normal(size=n))
        sizes = np.linalg.norm(trail, axis=1)
        good = sizes[(sizes > 1e-13) & (sizes < sizes[1])]
        rate = float(np.mean(good[1:] / good[:-1])) if good.size > 3 else float("nan")
        predicted = float(np.cos(theta) ** 2)
        print(f"{n:>5}{degrees:>8.0f}{rounds:>9}{rate:>16.9f}{predicted:>12.9f}"
              f"{rate / predicted:>10.6f}")

print("\nsets that do not meet: the iteration settles on the nearest pair")
ball = lambda w: co.project_onto_ball(w, 1.0)
away = lambda w: np.concatenate([[max(float(w[0]), 3.0)], w[1:]])
trail, rounds = alternating(ball, away, np.array([5.0, 1.0]))
print(f"  stops after {rounds} rounds at {np.array2string(trail[-1], precision=8)}")
print(f"  the cycle length is "
      f"{float(np.linalg.norm(away(ball(trail[-1])) - ball(trail[-1]))):.8f}, "
      f"and the distance between the sets is 2")
```

**The rate is the cosine squared of the angle, to nine digits**, at every angle and both dimensions.
The round count grows accordingly: $25$ rounds at $60$ degrees, $9692$ at $3$ degrees, because
$\log(\text{tolerance})/\log(\cos^{2}\theta)$ blows up as $\theta \to 0$.

That is the whole practical story of the method. It is beautifully simple, it needs only a projection
onto each set separately, and it is **slow when the sets meet at a shallow angle**, which is exactly
the case where the intersection is hard to describe and one would most want to use it.

**When the sets do not meet**, the iteration does not diverge or oscillate. It converges to the pair
of nearest points, with the cycle length equal to the distance between the sets, here exactly $2$.
That makes it a usable **infeasibility detector**: run it and look at the cycle length.

Dykstra's variant corrects the iterate so that the limit is the projection of the **starting point**
onto the intersection rather than merely some point of it, which matters when the projection itself is
what is wanted rather than feasibility.

### 3.5 Projected Newton

The obvious construction is to project the Newton step: $x_{k+1} = P_C(x_k + t\,p^{\text{Newton}})$.
Measuring it against the exact answer shows it is wrong.

```python
import numpy as np
from nalib import constrained as co


def more_sorensen(slope, matrix, radius, tol=1e-10, rounds=200):
    """Lesson 87's exact ball solve, reused here because it is the same subproblem."""
    g = np.asarray(slope, dtype=float).ravel()
    B = np.asarray(matrix, dtype=float)
    B = 0.5 * (B + B.T)
    radius = max(float(radius), 1e-300)
    values, vectors = np.linalg.eigh(B)
    weights = vectors.T @ g
    smallest = float(values[0])
    if smallest > 0.0:
        inside = vectors @ (-weights / values)
        if float(np.linalg.norm(inside)) <= radius:
            return {"step": inside, "kind": "interior"}
    scale = max(abs(smallest), float(np.max(np.abs(values))), 1.0)
    close = np.abs(values - smallest) <= 1e-10 * scale
    if float(np.max(np.abs(weights[close]))) <= 1e-12 * max(float(np.linalg.norm(g)), 1.0):
        safe = np.where(close, np.inf, values - smallest)
        base = vectors @ (-weights / safe)
        extra = np.sqrt(max(radius ** 2 - float(base @ base), 0.0))
        return {"step": base + extra * vectors[:, 0], "kind": "hard"}
    low = max(0.0, -smallest) * (1.0 + 1e-12) + 1e-14 * scale
    high = low + float(np.linalg.norm(g)) / radius
    lam = low
    for _ in range(int(rounds)):
        shifted = values + lam
        second = float(np.sum(weights ** 2 / shifted ** 2))
        length = np.sqrt(second)
        if abs(length - radius) <= tol * radius:
            break
        third = float(np.sum(weights ** 2 / shifted ** 3))
        if third <= 0.0:
            break
        low, high = (lam, high) if length > radius else (low, lam)
        lam = lam + (second / third) * (length - radius) / radius
        if not low < lam < high:
            lam = 0.5 * (low + high)
    return {"step": vectors @ (-weights / (values + lam)), "kind": "boundary"}


rng = np.random.default_rng(42)
print(f"{'n':>5}{'kappa':>9}{'gradient steps':>17}{'gradient error':>17}"
      f"{'naive Newton error':>21}{'scaled Newton steps':>21}{'scaled error':>15}")
for n in (5, 20):
    for kappa in (10.0, 1e3, 1e5):
        frame, _ = np.linalg.qr(rng.normal(size=(n, n)))
        raw = frame @ np.diag(np.logspace(0.0, np.log10(kappa), n)) @ frame.T
        A = 0.5 * (raw + raw.T)
        aim = rng.normal(size=n) * 3.0
        value = lambda v, A=A, aim=aim: 0.5 * float((np.asarray(v, dtype=float) - aim)
                                                    @ A @ (np.asarray(v, dtype=float) - aim))
        slope = lambda v, A=A, aim=aim: A @ (np.asarray(v, dtype=float) - aim)
        onto = lambda v: co.project_onto_ball(v, 1.0)
        truth = more_sorensen(-(A @ aim), A, 1.0)["step"]

        rough = co.projected_gradient(value, slope, onto, np.zeros(n),
                                      step=1.0 / float(np.max(np.linalg.eigvalsh(A))),
                                      tol=1e-12, max_steps=200000)

        x = np.zeros(n)                                # the naive projected Newton
        for _ in range(2000):
            g = slope(x)
            move = np.linalg.solve(A, -g)
            t, base, trial = 1.0, value(x), x
            for _ in range(60):
                trial = onto(x + t * move)
                if value(trial) <= base - 1e-4 * float(g @ (x - trial)):
                    break
                t *= 0.5
            if float(np.linalg.norm(trial - x)) <= 1e-12:
                break
            x = trial

        y, scaled_steps = np.zeros(n), 0               # projected in the Hessian metric
        budget = 500
        for k in range(budget):
            scaled_steps = k
            fresh = more_sorensen(slope(y) - A @ y, A, 1.0)["step"]
            if float(np.linalg.norm(fresh - y)) <= 1e-13:
                break
            y = fresh

        print(f"{n:>5}{kappa:>9.0e}{rough['steps']:>17}"
              f"{float(np.linalg.norm(rough['x'] - truth)):>17.1e}"
              f"{float(np.linalg.norm(x - truth)):>21.1e}{scaled_steps:>21}"
              f"{float(np.linalg.norm(y - truth)):>15.1e}")
```

Three answers, and only two of them are right.

**Projected gradient is correct**, to $10^{-13}$, and its cost tracks the conditioning: $12$ to $46$
steps at $\kappa = 10$ and $10^{3}$, but $1331$ at $\kappa = 10^{5}$.

**The naive projected Newton is wrong**, by $0.39$ to $1.05$. It converges, its iterates are feasible,
its objective decreases, and it settles on a point that is not the constrained minimum. Nothing about
the run flags it.

**Projecting in the Hessian metric is exact in one step**, error $0$ at every case.

The reason is worth stating carefully because the failure is so quiet. The optimality condition is
$x = P_C(x - \alpha\nabla f)$, and it is a statement about the **Euclidean** projection of a
**gradient** step. Replacing the gradient step by a Newton step changes the metric implicitly:
$-H^{-1}g$ is the steepest descent direction in the norm $\lVert\cdot\rVert_H$, not in the Euclidean
norm, so projecting it Euclidean-ly mixes two metrics and the fixed point condition becomes something
that is not the KKT system.

The repair is to project in the same metric, which means solving

$$
\min_{y \in C}\; \nabla f(x)^{T}(y - x) + \tfrac12 (y-x)^{T}H(y-x) ,
$$

and for a ball that is **exactly lesson 87's trust region subproblem**, which is why the code above
reuses `more_sorensen` unchanged. On a quadratic over a ball that solves the whole problem in one
step, which is the strongest possible answer to what the curvature buys.

For box constraints the same idea has a cheap form: Bertsekas's two metric projection, which scales
only the free variables and leaves the ones at their bounds alone, keeping the Euclidean projection
valid on the part where it is used.

### 4.1 The penalty method over twelve decades

```python
import numpy as np
from nalib import constrained as co
from nalib.quasinewton import bfgs

problem = co.equality_problem(5)
print(f"{'weight':>10}{'error':>12}{'violation':>12}{'condition':>12}"
      f"{'inner steps':>13}{'what happened':>26}")
for power in range(2, 20, 2):
    mu = 10.0 ** power
    inner = co.penalty_problem(problem, mu)
    with np.errstate(over="ignore", invalid="ignore"):
        out = bfgs(inner, tol=1e-10, max_steps=20000)
    x = out["x"]
    gap = np.asarray(problem["constraint"](x), dtype=float)
    try:
        cond = float(np.linalg.cond(np.asarray(inner["hessian"](x), dtype=float)))
    except Exception:
        cond = float("inf")
    note = "fine"
    if not np.all(np.isfinite(x)):
        note = "overflow"
    elif not out["converged"]:
        note = "the inner solve stalls"
    print(f"{mu:>10.0e}{float(np.linalg.norm(x - problem['minimizer'])):>12.2e}"
          f"{float(np.max(np.abs(gap))):>12.2e}{cond:>12.2e}{out['steps']:>13}{note:>26}")
```

Three breakdowns, in order, and each one has a different cause.

**At $\mu = 10^{6}$ the inner solve stops converging**, hitting the $20000$ step cap. The answer is
still right, because BFGS gets close before it stalls, but the stopping test is no longer met: the
gradient tolerance of $10^{-10}$ is below what a model with condition number $2.5\times10^{6}$ can
deliver. **The first thing to break is not the answer but the ability to certify it.**

**At $\mu = 10^{14}$ the accuracy reverses.** Up to $\mu = 10^{12}$ the error falls exactly as
$1.79\times10^{-1}/\mu$, reaching $2.1\times10^{-13}$. At $10^{14}$ it climbs back to
$2.4\times10^{-11}$, a hundredfold loss, because rounding in a model with condition $2.5\times10^{14}$
now dominates the $1/\mu$ term.

**At $\mu = 10^{16}$ it overflows** to `nan`, and the Hessian's condition number is reported as
infinite at $10^{18}$.

So the usable window is about eight decades wide, from $10^{4}$ where the accuracy is $10^{-5}$ to
$10^{12}$ where it is $10^{-13}$, and the best achievable accuracy is around $10^{-13}$. That is the
whole method in one table: **it works, it is easy to implement, and it cannot do better than about
thirteen digits no matter how it is tuned.** The augmented Lagrangian gets fifteen with a weight of
$10$.

### 4.2 The augmented Lagrangian's rate and its thresholds

```python
import numpy as np
from nalib import constrained as co

print("the rate against a fixed weight, on a convex equality problem")
print(f"{'n':>5}{'mu':>8}{'J H^-1 J^T':>13}{'predicted rate':>17}{'measured':>12}"
      f"{'ratio':>9}")
for n in (2, 5, 20):
    problem = co.equality_problem(n)
    hess = np.asarray(problem["hessian"](problem["minimizer"]), dtype=float)
    jac = np.asarray(problem["jacobian"](problem["minimizer"]), dtype=float)
    schur = float(jac @ np.linalg.solve(hess, jac.T))
    for mu in (0.1, 0.5, 1.0):
        out = co.augmented_lagrangian(problem, weight=mu, rounds=12, growth=1.0,
                                      wanted=0.0)
        errors = np.array([row["error"] for row in out["rows"]])
        good = errors[(errors > 1e-14) & (errors < errors[0])]
        measured = float(np.mean(good[1:] / good[:-1])) if good.size > 3 else float("nan")
        predicted = 1.0 / (1.0 + mu * schur)
        print(f"{n:>5}{mu:>8.2f}{schur:>13.4f}{predicted:>17.6f}{measured:>12.6f}"
              f"{measured / predicted:>9.4f}")
```

**The rate is exactly $1/(1 + \mu\, J H^{-1}J^{T})$**, matching to six digits wherever there are
enough points to fit it. The derivation is short: the multiplier update
$\lambda \leftarrow \lambda + \mu c(x_\lambda)$ is a fixed point iteration whose derivative with
respect to $\lambda$ is $1 - \mu\,\partial c/\partial\lambda$, and eliminating $x$ from the
stationarity condition gives $\partial c/\partial\lambda = -(JH^{-1}J^{T})/(1 + \mu JH^{-1}J^{T})$,
which rearranges to the quoted form.

The two rows at $n = 20$ with $\mu \ge 0.5$ do not match, and the reason is the usual one: the rate
there is $0.17$ and $0.09$, so the error reaches $10^{-14}$ within three rounds and there are not
enough points above the floor to fit.

**On a convex problem there is no threshold below which it fails.** The rate is below $1$ for every
$\mu > 0$, so the method converges for any positive weight and only gets slow. That is a correction to
the exercise's premise, and it is the reason the augmented Lagrangian is called robust.

A genuine threshold appears when the problem is not convex.

```python
import numpy as np


def saddle(gap):
    """minimize x1^2 - g x2^2 subject to x1 + x2 = 1, convex on the line only for g < 1."""
    g = float(gap)
    return {"A": np.diag([1.0, -g]), "gap": g,
            "minimizer": np.array([-g / (1.0 - g), 1.0 / (1.0 - g)]),
            "multiplier": 2.0 * g / (1.0 - g),
            "bounded_above": 2.0 * g / (1.0 - g),      # mu must exceed this
            "convergent_above": 4.0 * g / (1.0 - g)}   # and this, for the update to converge


def augmented(problem, mu, rounds=40):
    A, lam = problem["A"], 0.0
    x = np.array([0.7, 0.3])
    shifted = 2.0 * A + mu * np.ones((2, 2))
    smallest = float(np.linalg.eigvalsh(shifted)[0])
    if smallest <= 1e-10:
        return {"error": float("inf"), "bounded": False, "smallest": smallest}
    for _ in range(rounds):
        x = np.linalg.solve(shifted, -(lam - mu) * np.ones(A.shape[0]))
        lam = lam + mu * (float(np.sum(x)) - 1.0)
    return {"error": float(np.linalg.norm(x - problem["minimizer"])),
            "bounded": True, "smallest": smallest}


for gap in (0.2, 0.5, 0.8):
    problem = saddle(gap)
    print(f"\n  g = {gap}: bounded above mu = {problem['bounded_above']:.3f}, "
          f"convergent above mu = {problem['convergent_above']:.3f}, "
          f"answer {np.array2string(problem['minimizer'], precision=3)}")
    for mu in (0.5, 0.99, 1.01, 1.5, 5.0, 50.0):
        weight = mu * problem["bounded_above"]
        out = augmented(problem, weight)
        note = (f"error {out['error']:.2e}" if out["bounded"]
                else "unbounded below, no inner minimum")
        print(f"    mu = {weight:>8.3f} ({mu:>5.2f} x the first threshold): "
              f"smallest eigenvalue {out['smallest']:>9.5f}, {note}")
```

**There are two thresholds, not one, and both are exact.**

The first is $\mu > 2g/(1-g)$, which is where $\nabla^{2}f + \mu J^{T}J$ becomes positive definite.
Below it the augmented Lagrangian is **unbounded below**, so there is no inner minimum to find. The
measurement confirms it at $0.99\times$ the threshold, where the smallest eigenvalue is still
negative.

The second is $\mu > 4g/(1-g)$, exactly **twice** the first, and it is where the multiplier iteration
converges. Between the two thresholds the inner problem has a minimum and the outer iteration
**diverges**: at $1.5\times$ the first threshold the error after forty rounds is $10^{12}$, which is
$2^{40}$, so the rate is exactly $2$.

The second threshold follows from the same rate formula. With $H$ indefinite, $JH^{-1}J^{T} = -(1-g)/(2g)$
is **negative**, so the rate $1/(1+\mu s)$ has magnitude below $1$ only when $\mu > 2/\lvert s\rvert
= 4g/(1-g)$. The formula that gave the convergent case its speed gives the divergent case its
threshold.

**A positive definite inner problem is necessary and not sufficient.** That is the practical warning,
and it is why implementations increase $\mu$ when the violation is not falling fast enough, which is
what the lesson's `augmented_lagrangian` does with its `wanted` parameter.

### 4.3 The iteration count against the constraint count

```python
import numpy as np

sizes = (5, 10, 25, 50, 100, 200, 400)
for label, run in (("the barrier method, Newton steps",
                    lambda qp: barrier_qp(qp)["newton_steps"]),
                   ("the primal-dual method, steps",
                    lambda qp: primal_dual_qp(qp)["steps"]),
                   ("the active set method, iterations",
                    lambda qp: active_set_qp(qp)["steps"])):
    print(label)
    print(f"{'n':>5}" + "".join(f"{f'm={m}':>9}" for m in sizes))
    for n in (5, 20, 50):
        row = []
        for m in sizes:
            row.append(run(random_qp(n, m)) if m >= n else None)
        print(f"{n:>5}" + "".join(f"{(v if v is not None else 'n/a'):>9}" for v in row))
    print()
```

The three rows tell three different stories about the same problems.

**The barrier method is flat**, $74$ to $92$ Newton steps once there are at least twenty five
constraints, unchanged whether there are $25$ or $400$ of them and whether there are $5$ or $50$
variables. Its cost is the number of times the weight has to be cut, and that depends on the
tolerance rather than on the problem.

**The primal-dual method is flatter still and five times cheaper**, $13$ to $18$ steps across the
whole table. Going from $5\times5$ to $50\times400$, an eighty-fold increase in constraints,
costs four extra iterations.

**The active set method grows steeply.** At $n = 5$ it needs $2$ to $8$ iterations, better than either
interior point method. At $n = 50$, $m = 400$ it needs $108$, against the primal-dual's $17$.

Each iteration costs differently, so the counts are not the whole story: an active set iteration
factorizes a KKT matrix of the size of the working set, and an interior point iteration factorizes an
$n \times n$ matrix. But the trend is the one that decided the field. **The active set method is
combinatorial and the interior point methods are not**, and for a few hundred constraints upward that
is decisive.

The place the active set method still wins is a **warm start**: given the answer to a nearby problem,
it begins with almost the right working set and finishes in a handful of iterations, where an
interior point method has to walk in from the centre again. That is why SQP uses one for its
subproblem.

### 5.1 The multiplier is a price

**Claim.** For the perturbed problem $\min f(x)$ subject to $c(x) \le \epsilon$, with optimal value
$v(\epsilon)$,

$$
\frac{dv}{d\epsilon_i}\bigg|_{\epsilon = 0} = -\lambda_i .
$$

**Derivation.** Write the perturbed solution as $x(\epsilon)$ with multipliers $\lambda(\epsilon)$.
For an active constraint, $c_i(x(\epsilon)) = \epsilon_i$, so differentiating,
$\nabla c_i^{T} \dot x = e_i$. Then

$$
\frac{dv}{d\epsilon_i} = \nabla f^{T}\dot x
= \left(-\sum_j \lambda_j \nabla c_j\right)^{T}\dot x
= -\sum_j \lambda_j\, \delta_{ji} = -\lambda_i ,
$$

using stationarity in the second step and $\nabla c_j^{T}\dot x = \delta_{ji}$ in the third. For an
inactive constraint $\lambda_i = 0$ and the value does not change, which is 1.2 again.

**The reading.** $\lambda_i$ is the **shadow price**: how much the objective would improve per unit
of relaxation. In a production problem it is the most you should pay for one more unit of resource
$i$; in a portfolio it is the marginal cost of a position limit. That is why the multipliers are
often more interesting than the solution.

```python
import numpy as np
from nalib import constrained as co

print(f"{'n':>5}{'multiplier':>14}{'-d(value)/d(relaxation)':>26}{'ratio':>10}")
for n in (2, 5, 10):
    problem = co.equality_problem(n)
    # relaxing sum(x) = 1 to sum(x) = 1 + e moves the answer to (1+e)/n and the value to (1+e)^2/n
    step = 1e-5
    slope = ((1.0 + step) ** 2 / n - 1.0 / n) / step
    print(f"{n:>5}{float(problem['multipliers'][0]):>14.6f}{-slope:>26.6f}"
          f"{float(problem['multipliers'][0]) / -slope:>10.6f}")

print("\nand on a random QP, one row at a time")
qp = random_qp(6, 10)
reference = active_set_qp(qp)
print(f"{'row':>5}{'multiplier':>14}{'-d(value)/db':>16}{'active':>9}")
for i in range(qp["m"]):
    shifted = dict(qp)
    shifted["b"] = qp["b"].copy()
    shifted["b"][i] += 1e-6
    moved = active_set_qp(shifted)
    slope = (qp["f"](moved["x"]) - qp["f"](reference["x"])) / 1e-6
    if abs(reference["multipliers"][i]) > 1e-9 or abs(slope) > 1e-6:
        print(f"{i:>5}{reference['multipliers'][i]:>14.6f}{-slope:>16.6f}"
              f"{str(i in reference['working']):>9}")
```

The identity holds to five digits on the equality problem, limited by the finite difference step, and
to six on the random QP's one active row. Every inactive row has both a zero multiplier and a zero
value derivative, so nothing is printed for them, which is the numerical form of 1.2.

### 5.2 Degeneracy

**The construction.** Minimize $\lVert x - (2,2)\rVert^{2}$ subject to

$$
x_1 \le 1 , \qquad x_2 \le 1 , \qquad x_1 + x_2 \le 2 .
$$

The answer is $(1,1)$, where all three constraints are tight. Their gradients are $(1,0)$, $(0,1)$ and
$(1,1)$, three vectors in two dimensions, so they are necessarily dependent: the third is the sum of
the first two. LICQ fails.

**The multipliers are not unique.** Stationarity asks for $\nabla f + \sum\lambda_i \nabla c_i = 0$,
that is $(1,0)\lambda_1 + (0,1)\lambda_2 + (1,1)\lambda_3 = (2,2)$, whose solutions form the segment

$$
\lambda = (2 - t,\; 2 - t,\; t) , \qquad 0 \le t \le 2 ,
$$

the bound coming from $\lambda \ge 0$. Every point of that segment satisfies every KKT condition.

```python
import numpy as np

normals = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
offsets = np.array([-1.0, -1.0, -2.0])
answer = np.array([1.0, 1.0])
slope = 2.0 * (answer - np.array([2.0, 2.0]))
print(f"  at {answer} all three are tight: "
      f"{np.array2string(normals @ answer + offsets, precision=1)}")
print(f"  and the third gradient is the sum of the first two: "
      f"{np.array2string(normals[0] + normals[1] - normals[2])}")
print(f"\n{'t':>6}{'multipliers':>26}{'stationarity residual':>24}{'lam >= 0':>11}")
for t in np.linspace(0.0, 2.0, 5):
    lam = np.array([2.0 - t, 2.0 - t, t])
    residual = float(np.max(np.abs(slope + normals.T @ lam)))
    print(f"{t:>6.2f}{np.array2string(lam, precision=3):>26}{residual:>24.2e}"
          f"{str(bool(np.all(lam >= 0.0))):>11}")
```

Every $t$ in the range gives a residual of exactly zero, so the solution set of the KKT conditions is
a segment rather than a point.

**What it does to each method.**

```python
import numpy as np

width = normals.shape[1]
Q = 2.0 * np.eye(width)
aim = np.full(width, 2.0)
degenerate = {"Q": Q, "c": -2.0 * aim, "A": normals.copy(),
              "b": np.array([1.0, 1.0, 2.0]), "n": 2, "m": 3,
              "f": lambda v: float(np.asarray(v, dtype=float) @ Q
                                   @ np.asarray(v, dtype=float)) / 2.0
                             + float((-2.0 * aim) @ np.asarray(v, dtype=float)),
              "gradient": lambda v: Q @ np.asarray(v, dtype=float) - 2.0 * aim,
              "hessian": lambda v: Q.copy(),
              "constraint": lambda v: normals @ np.asarray(v, dtype=float)
                                      - np.array([1.0, 1.0, 2.0]),
              "jacobian": lambda v: normals.copy(),
              "kind": "inequality", "dimension": width, "start": np.zeros(width),
              "minimizer": answer, "multipliers": np.array([2.0, 2.0, 0.0]),
              "name": "the degenerate corner"}

print(f"{'method':>16}{'error in x':>13}{'iterations':>13}"
      f"{'multipliers found':>34}{'t':>9}")
for name, run in (("active set", lambda: active_set_qp(degenerate)),
                  ("SQP", lambda: sqp(degenerate)),
                  ("barrier", lambda: barrier_qp(degenerate)),
                  ("primal-dual", lambda: primal_dual_qp(degenerate))):
    out = run()
    lam = np.asarray(out["multipliers"], dtype=float)
    count = out.get("steps", out.get("newton_steps"))
    print(f"{name:>16}{float(np.linalg.norm(out['x'] - answer)):>13.2e}{count:>13}"
          f"{np.array2string(lam, precision=6):>34}{lam[2]:>9.4f}")
print(f"\nthe analytic centre of the segment maximizes 2 log(2-t) + log(t), "
      f"at t = {2.0 / 3.0:.6f}")
```

**Every method finds the same $x$**, to $10^{-11}$ or better. Degeneracy costs nothing in the
solution.

**Every method finds a different multiplier**, and each is a valid point of the segment. The active
set method returns $t = 0$, a **vertex** of the segment, because it works with a linearly independent
working set and never activates the redundant row. SQP returns $t = 1.333$. The barrier and
primal-dual methods return interior points, near $t = 2/3$, which is the **analytic centre** of the
segment, the maximizer of $\sum\log\lambda_i$ over it.

That last agreement is not a coincidence and it is a theorem: interior point methods converge to the
analytic centre of the optimal face. The barrier method reproduces it to six decimals.

```python
import numpy as np

print("the barrier multipliers against the analytic centre")
for rounds in (6, 8, 12, 16, 20):
    lam = barrier_qp(degenerate, rounds=rounds)["multipliers"]
    print(f"  {rounds:>3} outer rounds: t = {lam[2]:.6f}, 2 - t = {lam[0]:.6f}")
print(f"  the analytic centre is at t = {2.0 / 3.0:.6f}")
```

At eight rounds it is exactly $2/3$ to six decimals. **Beyond that it degrades**, and at twenty rounds
it collapses to $t = 0.0005$. That is 1.4's arithmetic warning made real: $\lambda_i = \mu/(-c_i)$ is
a ratio of two quantities both going to zero, and once $\mu$ falls below the rounding level of the
slacks the quotient is noise. **The multiplier estimate has an optimal stopping point, and running
longer makes it worse**, which is the opposite of how a convergent iteration is supposed to behave.

So the practical summary of degeneracy: the answer is safe, the multipliers are not, and if the
multipliers are what the problem is being solved for, then which method produced them matters.

### 5.3 The proximal view

**Projection is a proximal operator.** The proximal operator of a function $g$ at step $t$ is

$$
\operatorname{prox}_{tg}(v) = \arg\min_x\; g(x) + \frac{1}{2t}\lVert x - v\rVert^{2} .
$$

Take $g = \iota_C$, the indicator of a set $C$, which is $0$ on $C$ and $+\infty$ off it. Then any
$x \notin C$ makes the objective infinite, so the minimization is confined to $C$, where
$g \equiv 0$ and the objective is $\tfrac{1}{2t}\lVert x - v\rVert^{2}$. Therefore

$$
\boxed{\;\operatorname{prox}_{t\,\iota_C}(v) = \arg\min_{x \in C}\lVert x - v\rVert^{2} = P_C(v)\;}
$$

for **every** $t > 0$, since the step size scales the objective without moving its minimizer.

```python
import numpy as np
from nalib import proximal as px

out = px.the_prox_of_an_indicator_is_a_projection()
print(f"over {out['samples']} random points, worst difference between "
      f"the prox and the projection: {out['worst_difference']:.1e}")
print(f"they are the same operator: {out['they_are_the_same_operator']}")
print("\nand the step size does not matter, as the derivation says:")
rng = np.random.default_rng(42)
point = rng.normal(scale=3.0, size=6)
for t in (0.01, 1.0, 100.0):
    print(f"  t = {t:>7.2f}: {np.array2string(px.prox_box(point, -1.0, 1.0), precision=6)}")
```

The two routines are written independently in two modules and agree **exactly**, which is what an
identity rather than an approximation looks like, and the answer does not depend on $t$.

**What carries over to a general nonsmooth $g$.**

Projected gradient becomes proximal gradient, $x_{k+1} = \operatorname{prox}_{tg}(x_k - t\nabla f)$,
with the same $O(1/k)$ rate for convex $f$ and the same $O(\rho^{k})$ rate when $f$ is strongly
convex. The fixed point condition $x = \operatorname{prox}_{tg}(x - t\nabla f(x))$ is still equivalent
to optimality, now meaning $0 \in \nabla f(x) + \partial g(x)$. Nonexpansiveness carries over, and in
fact strengthens: every prox is **firmly** nonexpansive. Acceleration carries over, giving FISTA. All
of this is lesson 90.

**What does not.**

The **multipliers disappear.** There is no $\lambda$ in $\operatorname{prox}$, so the shadow price
reading of 5.1 has no counterpart, and neither does complementary slackness. What replaces them is
the subdifferential $\partial g$, which for an indicator is the normal cone and for an $\ell_1$ term
is a sign pattern. The information is still there but it is not in the form of a price.

**The active set disappears too**, or rather it becomes something else: for the $\ell_1$ prox it is
the set of non-zero coordinates, the **support**, and lesson 90 measures how quickly it settles. For
an indicator the two notions coincide, which is why box constrained problems look like both.

**Second order methods lose their footing.** Exercise 3.5 showed that even for the simplest nonsmooth
term, an indicator, projecting a Newton step is wrong, and the repair needed a metric-consistent
subproblem. For a general $g$ that subproblem is a scaled prox, which usually has no closed form, and
that is why the composite methods of lesson 90 are first order almost without exception.

**The barrier and penalty ideas do not transfer.** A barrier needs an interior, and a nonsmooth term
like $\lVert x\rVert_1$ has none in the relevant sense; a quadratic penalty on $g$ smooths away the
very structure, the exact zeros, that made $g$ worth using. What replaces them is the prox itself,
which handles $g$ exactly rather than approximately, and that is the whole reason the proximal
framework superseded them for this class of problem.

---

## Lesson 89, Stochastic Optimization

### 1.1 Why a constant step does not converge

A sampled gradient is the true gradient plus noise, $g_k = \nabla f(x_k) + \xi_k$ with
$\mathbb{E}[\xi_k] = 0$ and $\mathbb{E}\lVert\xi_k\rVert^{2} = \sigma^{2}$. A constant step $\alpha$
therefore adds $\alpha\xi_k$ to every iterate, and that contribution never shrinks.

**What it converges to** is a stationary distribution, not a point. The iterates enter a ball around
the minimum and then wander inside it forever. Section 2 derives its radius as

$$
r \;\approx\; \sqrt{\frac{\alpha\,\sigma^{2}}{\mu}}
$$

for a $\mu$-strongly convex problem. Two consequences follow, and exercise 4.3 measures both.

**More steps buy nothing once the ball is reached.** Running a constant step for ten thousand
iterations instead of a hundred gives the same accuracy, because the iterate is already inside.

**Halving the step buys a factor of $\sqrt2$**, not a factor of two, because the radius goes like
$\sqrt\alpha$. So the cost of accuracy is quadratic: ten times the accuracy costs a hundredth of the
step, and with a hundredth of the step everything takes a hundred times longer.

That is the whole reason schedules exist.

### 1.2 The Robbins-Monro conditions

$$
\sum_{k} \alpha_k = \infty \qquad\text{and}\qquad \sum_{k} \alpha_k^{2} < \infty .
$$

**The first says the steps can still reach the answer.** If $\sum\alpha_k$ is finite the total
distance the iterate can travel is bounded, so a starting point further away than that bound can
never be reached, whatever the gradients say.

**The second says the noise is eventually averaged out.** The accumulated variance is proportional to
$\sum\alpha_k^{2}$, so a finite sum means the total noise injected is finite and the iterate settles.

**Schedules that fail each.**

$\alpha_k = \alpha$ constant: the first holds and the second fails, $\sum\alpha^{2} = \infty$. The
iterate reaches the neighbourhood and never settles, which is 1.1.

$\alpha_k = \alpha/k^{1.5}$: the second holds and the first fails,
$\sum k^{-1.5} = \zeta(1.5) \approx 2.61$ is finite. The iterate stops moving before it arrives. The
lesson measures this as the worst of four schedules, and it is worse than a constant step, which is
the counterintuitive part.

$\alpha_k = \alpha/k$: **both hold**, $\sum 1/k = \infty$ and $\sum 1/k^{2} = \pi^{2}/6$. This is the
classical choice and it gives the optimal $O(1/k)$ rate in the function gap for a strongly convex
problem, but it decays so fast that the constant matters enormously; exercise 4.3 finds it the worst
performer at every budget on this problem.

$\alpha_k = \alpha/\sqrt k$: **both hold** only just, $\sum k^{-1/2} = \infty$ and
$\sum k^{-1} = \infty$, so the second **fails**. It is nonetheless the standard choice for a general
convex problem because it gives the optimal $O(1/\sqrt k)$ rate there, and 4.3 finds it among the two
best. The conditions are sufficient, not necessary.

### 1.3 Batch size and the price of a decision

Averaging $b$ independent sampled gradients divides the variance by $b$, so the noise standard
deviation is $\sigma/\sqrt b$ and the noise ball radius, from 1.1, is

$$
r \;\propto\; \sqrt{\frac{\alpha}{b}} .
$$

**Pricing the decision.** One step at batch $b$ costs $b$ gradient evaluations. So for a fixed budget
of $G$ evaluations, the choices are

| batch | steps taken | radius |
|---|---|---|
| $b$ | $G/b$ | $\propto \sqrt{\alpha/b}$ |
| $4b$ | $G/4b$ | $\propto \sqrt{\alpha/4b}$, half as large |

Quadrupling the batch halves the noise and quarters the number of steps. Whether that is worth it
depends on which term dominates: while the iterate is still far from the answer the number of steps
matters and a small batch wins, and once it is in the noise ball the radius matters and a large batch
wins.

The practical rule that comes out of it is that **the batch size should grow during the run**, which
is what adaptive batching schemes do, and it is why a schedule that shrinks $\alpha$ and a schedule
that grows $b$ are interchangeable in the radius formula, since only $\alpha/b$ appears. Exercise 4.1
verifies that both exponents are what this predicts.

### 1.4 What the adaptive methods do to the coordinates

All of AdaGrad, RMSProp and Adam divide the update by a running measure of each coordinate's gradient
size:

$$
x_{j} \leftarrow x_{j} - \frac{\alpha}{\sqrt{v_j} + \epsilon}\, g_j ,
$$

with $v_j$ the sum (AdaGrad) or an exponential average (RMSProp, Adam) of $g_j^{2}$. That is a
**diagonal preconditioner** of the kind lesson 86's exercise 5.2 discussed: it is
$M = \operatorname{diag}(1/\sqrt{v_j})$, and it rescales each coordinate so that the effective step
is measured in units of that coordinate's own gradient history.

**What it fixes.** A problem whose coordinates have wildly different scales, so that one step size
cannot serve all of them. After rescaling, a coordinate with large gradients gets a small step and
one with small gradients gets a large one, and a single $\alpha$ works for both.

**What it does not fix.** Correlations between coordinates. A diagonal matrix cannot rotate, so an
ill conditioned problem whose bad directions are not aligned with the axes is not helped at all. That
is exactly lesson 86's finding that Jacobi preconditioning buys nothing on a randomly rotated matrix.

The lesson measures both halves: at a scaling of $20$ the adaptive methods win by a factor of $54$
over the best non-adaptive step, and at a scaling of $1$ the margin falls to $2.5$.

### 1.5 What AdamW decouples

Adam with L2 regularization adds $\lambda x$ to the **gradient**, so the decay term goes through the
adaptive rescaling:

$$
x \leftarrow x - \frac{\alpha}{\sqrt{v}+\epsilon}\,(\hat m + \lambda x) .
$$

AdamW applies the decay to the **weights** directly, outside the rescaling:

$$
x \leftarrow x - \frac{\alpha}{\sqrt{v}+\epsilon}\,\hat m - \alpha\lambda x .
$$

**What is decoupled** is the amount of decay from the coordinate's gradient history. In Adam a
coordinate with small gradients has small $v$, so its decay is amplified by $1/\sqrt v$; a coordinate
with large gradients has its decay suppressed. That makes the effective regularization strength
different for every coordinate and dependent on the optimization trajectory, which is not what a
regularizer is supposed to be.

**When it makes no difference.** Three cases, in decreasing order of interest.

When $\lambda = 0$: the two updates are then **bit identical**, which the lesson measures.

When $\sqrt{v_j} + \epsilon$ is the same for every coordinate: then the rescaling is a scalar and it
commutes with the decay, absorbed into $\alpha\lambda$. Exercise 2.5 shows this properly.

When the run is short enough that the decay has not accumulated. The two differ by $O(\alpha\lambda
k)$ in the worst case, so a few steps hide it.

The lesson measures the difference at a factor of $61$ between the most and least affected
coordinate, which is a large enough spread to change which features survive.

### 2.1 The noise ball radius

Take $f(x) = \tfrac{\mu}{2}\lVert x - x^{*}\rVert^{2}$ so that $\nabla f = \mu(x - x^{*})$, and write
$e_k = x_k - x^{*}$. One step with a sampled gradient is

$$
e_{k+1} = e_k - \alpha\big(\mu e_k + \xi_k\big) = (1 - \alpha\mu)\,e_k - \alpha\xi_k .
$$

Taking squared norms and using $\mathbb{E}[\xi_k] = 0$, which kills the cross term,

$$
\mathbb{E}\lVert e_{k+1}\rVert^{2}
= (1-\alpha\mu)^{2}\,\mathbb{E}\lVert e_k\rVert^{2} + \alpha^{2}\sigma^{2} .
$$

This is a linear recursion in $E_k = \mathbb{E}\lVert e_k\rVert^{2}$ with fixed point

$$
E_\infty = \frac{\alpha^{2}\sigma^{2}}{1 - (1-\alpha\mu)^{2}}
= \frac{\alpha^{2}\sigma^{2}}{2\alpha\mu - \alpha^{2}\mu^{2}}
= \frac{\alpha\sigma^{2}}{\mu\,(2 - \alpha\mu)} .
$$

For small $\alpha$ the denominator tends to $2\mu$, giving

$$
\boxed{\;r = \sqrt{E_\infty} \;\approx\; \sqrt{\frac{\alpha\sigma^{2}}{2\mu}}
\;\propto\; \sqrt\alpha\;}
$$

**The square root is the whole content.** The contraction $(1-\alpha\mu)$ removes error at a rate
proportional to $\alpha$, and the noise injects error at a rate proportional to $\alpha^{2}$; the
balance point is where $\alpha^{2}\sigma^{2} = 2\alpha\mu E$, giving $E \propto \alpha$ and
$r \propto \sqrt\alpha$.

Note also that the recursion converges only when $\lvert 1 - \alpha\mu\rvert < 1$, that is
$\alpha < 2/\mu$, which is the stability condition exercise 4.2 measures.

```python
import numpy as np
from nalib import stochastic as st

out = st.the_noise_ball_grows_like_the_square_root_of_the_step()
print(f"{'step':>10}{'measured radius':>18}{'radius / sqrt(step)':>22}")
for row in out["rows"]:
    print(f"{row['step']:>10.0e}{row['tail_distance']:>18.4e}"
          f"{row['over_root_step']:>22.4f}")
print(f"\nfitted exponent {out['fitted_power_of_the_step']:.4f} against the predicted 0.5")
print(f"the square root law holds: {out['the_power_is_a_half']}")
print(f"the constant varies by a factor of {out['spread_in_the_constant']:.2f} "
      f"over the whole sweep")
```

The measured exponent matches $0.5$ and the ratio $r/\sqrt\alpha$ is nearly constant across the
sweep, which is the derivation confirmed in the form it was derived.

### 2.2 The batch gradient is unbiased with variance $\sigma^{2}/b$

Let $f(x) = \frac1N\sum_{i=1}^{N} f_i(x)$ and let $B$ be $b$ indices drawn uniformly at random with
replacement. The batch gradient is $g_B(x) = \frac1b\sum_{i \in B}\nabla f_i(x)$.

**Unbiased.** Each index $i \in B$ is uniform on $\{1,\dots,N\}$, so

$$
\mathbb{E}[\nabla f_i(x)] = \frac1N\sum_{j=1}^{N}\nabla f_j(x) = \nabla f(x) ,
$$

and therefore $\mathbb{E}[g_B] = \frac1b \sum_{i \in B}\nabla f(x) = \nabla f(x)$. **This holds for
every $b$**, including $b = 1$, which is why a single sample is a legitimate gradient estimate.

**Variance.** The $b$ terms are independent because the draws are with replacement, so variances add:

$$
\operatorname{Var}(g_B) = \frac{1}{b^{2}}\sum_{i\in B}\operatorname{Var}(\nabla f_i)
= \frac{1}{b^{2}}\cdot b\,\sigma^{2} = \frac{\sigma^{2}}{b} ,
$$

with $\sigma^{2} = \frac1N\sum_j \lVert\nabla f_j - \nabla f\rVert^{2}$ the variance of one sample.
So the standard deviation is $\sigma/\sqrt b$.

**Two things the derivation depends on.** Independence, which needs sampling **with replacement**;
sampling without replacement gives a smaller variance by the finite population factor
$(N-b)/(N-1)$, which vanishes at $b = N$ where the gradient is exact. And a finite $\sigma^{2}$,
which fails for heavy tailed gradients and is where gradient clipping in exercise 3.2 earns its keep.

```python
import numpy as np
from nalib import stochastic as st

problem = st.least_squares_problem(samples=2000, variables=10)
rng = np.random.default_rng(42)
point = rng.normal(size=problem["dimension"])
truth = problem["gradient"](point)
draws = 20000

# the single sample variance, measured once and used as the reference for every batch
singles = np.array([problem["batch_gradient"](point, rng.integers(0, 2000, 1))
                    for _ in range(draws)])
sigma = float(np.sqrt(np.mean(np.sum((singles - truth) ** 2, axis=1))))
print(f"  one sample: sigma = {sigma:.4f}")
print(f"\n{'batch':>7}{'bias':>12}{'bias / (sigma/sqrt(mb))':>26}"
      f"{'measured sqrt variance':>25}{'sigma / sqrt(b)':>18}{'ratio':>9}")
for b in (1, 4, 16, 64, 256):
    sample = np.array([problem["batch_gradient"](point, rng.integers(0, 2000, b))
                       for _ in range(draws)])
    bias = float(np.linalg.norm(sample.mean(axis=0) - truth))
    spread = float(np.sqrt(np.mean(np.sum((sample - truth) ** 2, axis=1))))
    expected_bias = sigma / np.sqrt(draws * b)
    print(f"{b:>7}{bias:>12.2e}{bias / expected_bias:>26.2f}"
          f"{spread:>25.4f}{sigma / np.sqrt(b):>18.4f}{spread / (sigma / np.sqrt(b)):>9.4f}")
```

Two checks, and both are exact.

**The bias is zero.** The measured bias is not literally zero because it is an average over a finite
number of draws, so it should sit at about $\sigma/\sqrt{Mb}$ with $M$ the number of draws. The
third column is the measured bias divided by that prediction, and it is of order one at every batch
size, which is what "zero bias plus sampling error" looks like.

**The variance is $\sigma^{2}/b$**, with the ratio at $1.000$ to three or four digits at every batch
size. The quantity measured is $\sqrt{\mathbb{E}\lVert g_B - \nabla f\rVert^{2}}$, the square root of
the total variance, which is the quantity the derivation is about; the mean of the norm is a
different number and does not obey the law exactly.

### 2.3 The optimal heavy ball parameters

Heavy ball on a quadratic with Hessian eigenvalues in $[\mu, L]$ is

$$
x_{k+1} = x_k - \alpha\nabla f(x_k) + \beta(x_k - x_{k-1}) .
$$

In an eigendirection with eigenvalue $\lambda$ the error satisfies the two term recursion

$$
e_{k+1} = (1 - \alpha\lambda + \beta)\,e_k - \beta\, e_{k-1} ,
$$

whose characteristic polynomial is $z^{2} - (1-\alpha\lambda+\beta)z + \beta = 0$. The product of the
roots is $\beta$, so if both roots are complex their common modulus is exactly $\sqrt\beta$,
**independent of $\lambda$**.

That is the key observation. Choose $\alpha$ and $\beta$ so the roots are complex for **every**
$\lambda \in [\mu, L]$, and the rate is $\sqrt\beta$ uniformly. The roots are complex when the
discriminant is negative,

$$
(1-\alpha\lambda+\beta)^{2} < 4\beta \quad\Longleftrightarrow\quad
(1-\sqrt\beta)^{2} < \alpha\lambda < (1+\sqrt\beta)^{2} .
$$

Requiring this at both ends and making $\beta$ as small as possible gives
$\alpha\mu = (1-\sqrt\beta)^{2}$ and $\alpha L = (1+\sqrt\beta)^{2}$. Dividing,

$$
\frac{L}{\mu} = \left(\frac{1+\sqrt\beta}{1-\sqrt\beta}\right)^{2}
\;\Longrightarrow\;
\sqrt\kappa = \frac{1+\sqrt\beta}{1-\sqrt\beta}
\;\Longrightarrow\;
\boxed{\;\sqrt\beta = \frac{\sqrt\kappa-1}{\sqrt\kappa+1},\qquad
\alpha = \frac{4}{(\sqrt L + \sqrt\mu)^{2}}\;}
$$

**The rate is $\sqrt\beta = (\sqrt\kappa-1)/(\sqrt\kappa+1)$**, against steepest descent's
$(\kappa-1)/(\kappa+1)$ from lesson 86. Reaching a fixed accuracy takes $O(\sqrt\kappa)$ steps
instead of $O(\kappa)$, the same square root that conjugate gradients achieves.

```python
import numpy as np
from nalib import stochastic as st

out = st.momentum_turns_the_condition_number_into_its_square_root()
print(f"{'kappa':>10}{'plain steps':>13}{'plain rate':>12}{'its bound':>11}"
      f"{'heavy steps':>13}{'heavy rate':>12}{'its bound':>11}{'speedup':>9}"
      f"{'sqrt(kappa)':>13}")
for row in out["rows"]:
    print(f"{row['condition']:>10.0f}{row['plain_steps']:>13}{row['plain_rate']:>12.6f}"
          f"{row['plain_bound']:>11.6f}{row['heavy_steps']:>13}"
          f"{row['heavy_rate']:>12.6f}{row['heavy_bound']:>11.6f}"
          f"{row['speedup']:>9.1f}{np.sqrt(row['condition']):>13.1f}")
print(f"\nplain matches its bound:    {out['plain_matches_its_bound']}")
print(f"momentum matches its bound: {out['heavy_matches_its_bound']}")
print(f"the largest speedup is {out['biggest_speedup']:.1f}")
```

Both rates match their own bounds, and the speedup grows with $\sqrt\kappa$. Exercise 5.2 asks what
survives when the gradient is sampled, and the answer is not what this table suggests.

### 2.4 AdaGrad's implicit schedule

AdaGrad accumulates the squared gradients without forgetting:

$$
v_k = \sum_{j=1}^{k} g_j^{2} , \qquad
x \leftarrow x - \frac{\alpha}{\sqrt{v_k} + \epsilon}\, g_k .
$$

Suppose the gradients have a roughly constant typical size $\bar g$ in a given coordinate, which is
what happens when the iterate is wandering in a noise ball rather than converging. Then
$v_k \approx k\,\bar g^{2}$ and

$$
\frac{\alpha}{\sqrt{v_k}} \approx \frac{\alpha}{\bar g\,\sqrt k}
\;\propto\; \frac{1}{\sqrt k} .
$$

**AdaGrad therefore implements a $1/\sqrt k$ schedule without being told to.** That is the schedule
1.2 identified as the right one for a general convex problem, so AdaGrad gets the correct decay for
free and adapts the constant per coordinate.

The same calculation explains AdaGrad's known weakness. If the gradients do **not** shrink, $v_k$
grows without bound and the step decays forever, so a long run stalls even when there is still
progress to make. RMSProp replaces the sum by an exponential average,
$v_k = \rho v_{k-1} + (1-\rho)g_k^{2}$, which tracks the recent size rather than the total and
therefore does not decay on its own. Adam is RMSProp with momentum and bias correction.

```python
import numpy as np
from nalib import stochastic as st

problem = st.least_squares_problem(samples=2000, variables=10, scaling=5.0)
rng = np.random.default_rng(42)


def adagrad_trace(base, rounds=20000, frozen=False):
    """With ``frozen`` the iterate is held still, so the gradients keep the same size."""
    x = problem["start"].copy()
    accumulated = np.zeros_like(x)
    marks = []
    for k in range(1, rounds + 1):
        rows = rng.integers(0, problem["samples"], size=16)
        g = problem["batch_gradient"](x, rows)
        accumulated += g * g
        effective = base / (np.sqrt(accumulated) + 1e-8)
        if not frozen:
            x = x - effective * g
        if k in (10, 100, 1000, 10000, 20000):
            marks.append((k, float(np.median(effective)),
                          float(np.median(accumulated)) / k))
    return marks


for frozen in (True, False):
    print(f"\n  {'the iterate held still, so the gradients keep their size'
            if frozen else 'the ordinary run, where the gradients shrink'}")
    print(f"{'step k':>9}{'median effective step':>24}{'times sqrt(k)':>16}"
          f"{'accumulated / k':>18}")
    for k, size, growth in adagrad_trace(0.05, frozen=frozen):
        print(f"{k:>9}{size:>24.6e}{size * np.sqrt(k):>16.6f}{growth:>18.6e}")
```

**The derivation's hypothesis has to hold for its conclusion to.** The first table holds the iterate
still, so the gradients keep a constant typical size, and there $v_k/k$ is constant and the effective
step times $\sqrt k$ is constant: exactly $1/\sqrt k$, as derived.

**In an ordinary run it is not $1/\sqrt k$.** The second table shows the effective step falling only
from $1.1\times10^{-3}$ to $4.0\times10^{-4}$ over four decades in $k$, a power of about $-0.13$
rather than $-0.5$, and $v_k/k$ falling steadily rather than staying flat.

The reason is that the gradients **do** shrink as the iterate approaches the answer, so $v_k$ grows
sublinearly and the step decays more slowly than the derivation's constant gradient case. That is not
a failure of the derivation; it is the derivation applied where its hypothesis does not hold, and the
honest statement is that AdaGrad implements a $1/\sqrt k$ schedule **only while the gradients are not
shrinking**, which is the regime that matters, since it is where a schedule is needed.

### 2.5 When Adam and AdamW coincide

Adam with L2 adds $\lambda x$ to the gradient before the moments are formed. Write the update to
first order in $\lambda$, with $\hat m$ and $\hat v$ the bias corrected moments:

$$
x \leftarrow x - \frac{\alpha}{\sqrt{\hat v}+\epsilon}\,\hat m
\;-\; \frac{\alpha\lambda}{\sqrt{\hat v}+\epsilon}\, x + O(\lambda^{2}) ,
$$

since the decay term passes through the moment averages linearly. AdamW is

$$
x \leftarrow x - \frac{\alpha}{\sqrt{\hat v}+\epsilon}\,\hat m \;-\; \alpha\lambda\, x .
$$

**The two agree exactly when $\sqrt{\hat v_j}+\epsilon$ is the same number $s$ for every
coordinate**, because then the Adam decay term is $(\alpha\lambda/s)\,x$, which is AdamW's
$\alpha\lambda' x$ with $\lambda' = \lambda/s$. The two differ only by a **reparameterization of the
decay constant**, so any behaviour one can produce the other can produce with a rescaled $\lambda$.

When $\hat v$ varies across coordinates no single rescaling works, and the effective decay is
$\lambda/(\sqrt{\hat v_j}+\epsilon)$, different for each $j$ and changing during the run. That is
what AdamW decouples.

$\square$

```python
import numpy as np
from nalib import stochastic as st

out = st.adam_and_adamw_differ()
print(f"with no decay, the two runs are identical: {out['identical_without_decay']} "
      f"(worst difference {out['difference_without_decay']:.1e})")
print(f"with decay they differ by {out['distance_with_decay']:.3e}: "
      f"{out['different_with_decay']}")
print(f"\nand the difference is uneven across coordinates: "
      f"{out['the_difference_is_uneven_across_coordinates']}")
print(f"  smallest coordinate difference {out['smallest_coordinate_difference']:.3e}")
print(f"  largest  coordinate difference {out['largest_coordinate_difference']:.3e}")
print(f"  a spread of "
      f"{out['largest_coordinate_difference'] / out['smallest_coordinate_difference']:.1f}")
print(f"\n{'method':>8}{'decay':>8}{'gap':>13}{'norm of the answer':>21}"
      f"{'distance':>12}")
for row in out["rows"]:
    print(f"{row['method']:>8}{row['weight_decay']:>8.2f}{row['gap']:>13.3e}"
          f"{row['norm_of_the_answer']:>21.6f}{row['final_distance']:>12.3e}")
```

Without decay the two are bit identical, as the derivation says. With decay the per coordinate
difference spans a large factor, so no single $\lambda'$ makes them agree, which is the numerical
form of "the second moment is not constant across coordinates". The table also shows what the decay
is for: it shrinks the norm of the answer, and it shrinks it by different amounts under the two
rules.

### 3.1 SVRG

SVRG stores a full gradient at an anchor point $\tilde x$ and uses it as a control variate:

$$
g_k = \nabla f_i(x_k) - \nabla f_i(\tilde x) + \nabla f(\tilde x) .
$$

It is still unbiased, because $\mathbb{E}[\nabla f_i(\tilde x)] = \nabla f(\tilde x)$, and its
variance goes to zero as $x_k$ and $\tilde x$ both approach $x^{*}$, because the first two terms then
cancel. **That is the whole idea: the noise vanishes as the answer is approached**, which is what a
constant step needs in order to converge.

```python
import numpy as np
from nalib import stochastic as st


def svrg(problem, step, epochs=25, batch=1, seed=0):
    rng = np.random.default_rng(seed)
    n = problem["samples"]
    x = problem["start"].copy()
    star = problem["minimizer"]
    trail, used = [], 0
    for _ in range(epochs):
        anchor = x.copy()
        full = problem["gradient"](anchor)
        used += n
        for _ in range(n // batch):
            rows = rng.integers(0, n, size=batch)
            g = (problem["batch_gradient"](x, rows)
                 - problem["batch_gradient"](anchor, rows) + full)
            x = x - step * g
            used += 2 * batch
        trail.append(float(np.linalg.norm(x - star)))
    return {"x": x, "distances": np.asarray(trail), "gradients": used,
            "final_distance": trail[-1]}


for scaling in (1.0, 20.0):
    problem = st.least_squares_problem(samples=2000, variables=10, scaling=scaling)
    worst = 2.0 * float(np.max(np.sum(problem["design"] ** 2, axis=1)))
    step = 1.0 / (10.0 * worst)
    print(f"\nscaling {scaling:g}: kappa {problem['condition']:.1f}, "
          f"average L {problem['smoothness']:.2f}, worst per-sample L {worst:.1f}")
    print(f"  the step has to be 1 / (10 L_max) = {step:.2e}")
    out = svrg(problem, step=step)
    print(f"  SVRG epoch distances: "
          + " ".join(f"{v:.2e}" for v in out["distances"][::4]))
    plain = st.stochastic_descent(problem, "sgd", step=step, batch=1,
                                  rounds=out["gradients"])
    print(f"  after {out['gradients']} gradients: SGD {plain['final_distance']:.3e}, "
          f"SVRG {out['final_distance']:.3e}")
    good = out["distances"][(out["distances"] > 1e-13)
                            & (out["distances"] < out["distances"][0])]
    if good.size > 3:
        ratios = good[1:] / good[:-1]
        print(f"  per-epoch ratio: mean {float(np.mean(ratios)):.5f}, "
              f"spread {float(np.std(ratios)):.1e}  (constant means linear)")
```

**The convergence does become linear**, and the evidence is the constant ratio. At a scaling of $1$
each epoch multiplies the error by $0.128$ with a spread of only $0.019$, and the run reaches
$1.3\times10^{-14}$, the rounding floor. SGD with the same step and the same total number of gradient
evaluations reaches $5.2\times10^{-2}$, twelve orders of magnitude worse.

**But the rate is governed by the worst per-sample smoothness, not the average.** The step must be
about $1/(10 L_{\max})$ where $L_{\max} = \max_i \lVert\nabla^{2}f_i\rVert$, and at a scaling of $20$
that is $L_{\max} = 15149$ against an average $L$ of $837$, eighteen times larger. The required step
is then $6.6\times10^{-6}$, the per epoch ratio is $0.970$, and after the same budget SVRG is at
$0.92$ where plain SGD is at $0.24$.

So the honest summary is that **SVRG converts a sublinear method into a linear one and pays for it
with a step size set by the worst sample.** On well scaled data that is a spectacular win; on badly
scaled data it is a loss, and the fix is to scale the data rather than to change the method.

SAGA replaces the periodic full gradient by a stored table of one gradient per sample, updated one at
a time. It removes the epoch structure and the occasional full pass, at the cost of $O(Nm)$ storage,
which for a linear model collapses to $O(N)$ because each stored gradient is a scalar times its own
row.

### 3.2 Gradient clipping

Clipping rescales any gradient longer than a threshold $c$:

$$
\tilde g = g \cdot \min\!\left(1,\; \frac{c}{\lVert g\rVert}\right) .
$$

```python
import numpy as np
from nalib import stochastic as st


def clipped(problem, step, clip=None, batch=16, rounds=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = problem["samples"]
    x = problem["start"].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(rounds):
            rows = rng.integers(0, n, size=batch)
            g = problem["batch_gradient"](x, rows)
            if clip is not None:
                size = float(np.linalg.norm(g))
                if size > clip:
                    g = g * (clip / size)
            x = x - step * g
            if not np.all(np.isfinite(x)):
                return {"diverged": True, "final_distance": float("inf")}
    gap = float(np.linalg.norm(x - problem["minimizer"]))
    return {"diverged": not np.isfinite(gap) or gap > 1e6, "final_distance": gap}


problem = st.least_squares_problem(samples=2000, variables=10, scaling=1.0)
L = problem["smoothness"]
print(f"  smoothness L = {L:.4f}, so 2/L = {2.0 / L:.4f}")
print(f"{'step / (2/L)':>14}{'no clipping':>22}{'clip at 10':>22}{'clip at 1':>20}")
for factor in (0.5, 0.9, 1.0, 1.5, 3.0, 10.0, 100.0):
    step = factor * 2.0 / L
    row = []
    for clip in (None, 10.0, 1.0):
        out = clipped(problem, step, clip=clip)
        row.append("diverged" if out["diverged"] else f"{out['final_distance']:.3e}")
    print(f"{factor:>14.1f}{row[0]:>22}{row[1]:>22}{row[2]:>20}")

print("\n  the largest stable step, by bisection")
for clip in (None, 100.0, 10.0, 1.0):
    low, high = 0.1 * 2.0 / L, 1e7 * 2.0 / L
    for _ in range(40):
        mid = np.sqrt(low * high)
        if clipped(problem, mid, clip=clip, rounds=3000)["diverged"]:
            high = mid
        else:
            low = mid
    label = "none" if clip is None else f"{clip:g}"
    print(f"    clip {label:>6}: {low:.4e} = {low / (2.0 / L):.4g} times 2/L")
```

**Without clipping the threshold is $0.73 \times 2/L$**, slightly below the deterministic bound
because the batch noise adds effective curvature.

**With clipping there is no threshold.** The largest stable step grows exactly in inverse proportion
to the clip level, a factor of $10$ for each factor of $10$ in $c$, and reaches $1.5\times10^{6}$
times $2/L$ at $c = 1$. The reason is simple: a clipped step moves the iterate by at most
$\alpha c$, so the iterate cannot run away no matter how large $\alpha$ is.

**But stability is not convergence.** The final distances with $c = 10$ grow steadily with the step,
$4.9, 6.4, 7.9, 20, 53, 567$, so the iterate is wandering in a ball of radius proportional to
$\alpha c$. **Clipping converts divergence into a large noise ball**, which is a real improvement
over a `nan` and is not a substitute for a sensible step size.

What clipping is genuinely for is the heavy tailed case 2.2 excluded: when $\sigma^{2}$ is infinite,
or when a rare batch produces a gradient a thousand times the typical one, a single such batch
destroys the run and clipping absorbs it. That is why it is standard in language model training,
where the gradient distribution has exactly that shape, and it is why the threshold is usually set
from the observed gradient norms rather than chosen in advance.

The table also shows a case where clipping helps at a sensible step: at $0.5\times 2/L$, clipping at
$1$ gives $0.212$ against $0.559$ unclipped, because it suppresses the largest noise contributions.

### 3.3 Polyak averaging

Polyak and Ruppert observed that the **average** of the iterates converges faster than the iterates
themselves. With $\alpha_k \propto 1/\sqrt k$, the last iterate has $\mathbb{E}\lVert e_k\rVert
\sim k^{-1/4}$ in distance, and the tail average has $k^{-1/2}$, which is the rate a $1/k$ schedule
would give without needing to know $\mu$.

```python
import numpy as np
from nalib import stochastic as st


def with_averaging(problem, rule, batch=16, rounds=40000, seed=0, tail=0.5):
    rng = np.random.default_rng(seed)
    n = problem["samples"]
    x = problem["start"].copy()
    star = problem["minimizer"]
    begin = int((1.0 - tail) * rounds)
    running, count = np.zeros_like(x), 0
    for k in range(rounds):
        rows = rng.integers(0, n, size=batch)
        x = x - rule(k) * problem["batch_gradient"](x, rows)
        if k >= begin:
            running += x
            count += 1
    return (float(np.linalg.norm(x - star)),
            float(np.linalg.norm(running / max(count, 1) - star)))


problem = st.least_squares_problem(samples=2000, variables=10, scaling=1.0)
base = 0.2 / problem["smoothness"]
rule = lambda k: base / np.sqrt(k + 1.0)
print(f"{'budget':>10}{'plain, median of 9':>21}{'averaged, median of 9':>24}{'gain':>8}")
budgets, plain, averaged = [], [], []
repeats = 9
for rounds in (2500, 5000, 10000, 20000, 40000, 80000):
    pairs = np.array([with_averaging(problem, rule, rounds=rounds, seed=s)
                      for s in range(repeats)])
    last, mean = float(np.median(pairs[:, 0])), float(np.median(pairs[:, 1]))
    budgets.append(rounds)
    plain.append(last)
    averaged.append(mean)
    print(f"{rounds:>10}{last:>21.4e}{mean:>24.4e}{last / mean:>8.2f}")
print(f"\n  fitted exponents: plain "
      f"{float(np.polyfit(np.log(budgets), np.log(plain), 1)[0]):.4f} against -0.25, "
      f"averaged {float(np.polyfit(np.log(budgets), np.log(averaged), 1)[0]):.4f} "
      f"against -0.5")
```

**Averaging recovers the rate exactly.** The fitted exponent for the averaged iterate is $-0.4952$
against the predicted $-0.5$, and for the plain iterate $-0.2873$ against $-0.25$. Because the
exponents differ, the advantage grows with the budget: from $1.67$ at $2500$ steps to $3.09$ at
$80000$.

Two things make this remarkable. **It costs nothing**: one running sum and no extra gradients. And
**it needs no knowledge of $\mu$**, where the $1/k$ schedule that achieves the same rate directly
needs the constant to be at least $1/\mu$ or it degrades badly, which is exactly why exercise 4.3
finds $1/k$ the worst schedule on this problem.

The measurement uses the **tail** average, over the last half of the run. Averaging from the start
would include the early iterates, which are far from the answer and would dominate the mean for a
long time.

### 3.4 A stochastic line search

The exercise's own hint is the answer, and measuring it makes the reason unmistakable.

```python
import numpy as np
from nalib import stochastic as st


def line_search_run(problem, batch=64, rounds=4000, seed=0, refresh=False):
    rng = np.random.default_rng(seed)
    n = problem["samples"]
    x = problem["start"].copy()
    star = problem["minimizer"]
    step = 1.0
    marks, chosen = [], []
    for k in range(rounds):
        rows = rng.integers(0, n, size=batch)

        def value(v, rows=rows):
            part = problem["design"][rows]
            gap = part @ v - problem["targets"][rows]
            return float(gap @ gap) / rows.size

        g = problem["batch_gradient"](x, rows)
        base = value(x)
        trial = 2.0 * step
        for _ in range(30):
            if refresh:                              # a new batch for every trial
                rows = rng.integers(0, n, size=batch)
                g = problem["batch_gradient"](x, rows)
                base = value(x)
            if value(x - trial * g) <= base - 1e-4 * trial * float(g @ g):
                break
            trial *= 0.5
        step = trial
        x = x - trial * g
        if (k + 1) % (rounds // 4) == 0:
            marks.append(float(np.linalg.norm(x - star)))
            chosen.append(step)
    return marks, chosen


problem = st.least_squares_problem(samples=2000, variables=10, scaling=1.0)
print(f"  the right step is 1/L = {1.0 / problem['smoothness']:.4f}")
for refresh in (False, True):
    print(f"\n  {'a fresh batch for every trial' if refresh else 'one batch held across the trials'}")
    print(f"{'batch':>7}{'distance at 25/50/75/100 per cent':>42}"
          f"{'step selected':>42}")
    for batch in (32, 128, 512):
        marks, chosen = line_search_run(problem, batch=batch, refresh=refresh)
        print(f"{batch:>7}{'  '.join(f'{v:.2e}' for v in marks):>42}"
              f"{'  '.join(f'{v:.1e}' for v in chosen):>42}")
```

**With one batch held across the trials the search works.** It selects a step of $0.5$ at every batch
size and at every point of the run, against the correct $1/L = 0.44$. That is a fourteen per cent
error and it comes from the halving, which can only produce powers of two.

**With a fresh batch for every trial it selects nothing.** The chosen step wanders over seven orders
of magnitude, from $5\times10^{-1}$ down to $4.5\times10^{-13}$ and back, with no pattern.

**Why the batch has to be fixed.** The Armijo test compares $f_B(x - \alpha g)$ against $f_B(x)$. With
a different $B$ on each side, the comparison measures the difference between **two different
functions**, and the noise in that difference does not shrink as $\alpha \to 0$: at $\alpha = 0$ the
two sides still differ by the sampling error. So the test never reliably passes, the halving continues
to the iteration limit, and the outcome is essentially random.

With $B$ fixed, both sides come from the same function, the difference goes to zero with $\alpha$ as
$-\alpha g^{T}g$, and the test is the genuine one dimensional Armijo condition of lesson 86.

The distance column is a trap worth pointing out. The fresh batch runs report **smaller** final
distances, and that is not success: a step of $10^{-13}$ means the iterate has stopped moving, so it
stops adding noise and sits wherever it happened to land. **A frozen iterate looks accurate on a noisy
problem.** The step column is what reveals it.

### 3.5 AdaBelief and Lion

AdaBelief replaces Adam's $v = \mathbb{E}[g^{2}]$ by $v = \mathbb{E}[(g - m)^{2}]$, the variance of
the gradient about its own running mean rather than its second moment, so a coordinate with a large
but **steady** gradient gets a large step rather than a small one. Lion drops the second moment
entirely and takes the **sign** of an interpolated momentum, so every coordinate moves by exactly
$\pm\alpha$.

```python
import numpy as np
from nalib import stochastic as st


def modern(problem, method, step=0.01, batch=16, rounds=8000, seed=0,
           momentum=0.9, decay=0.999, eps=1e-8):
    rng = np.random.default_rng(seed)
    n = problem["samples"]
    x = problem["start"].copy()
    first, second = np.zeros_like(x), np.zeros_like(x)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(1, rounds + 1):
            rows = rng.integers(0, n, size=batch)
            g = problem["batch_gradient"](x, rows)
            if method == "adabelief":
                first = momentum * first + (1 - momentum) * g
                gap = g - first
                second = decay * second + (1 - decay) * gap * gap + eps
                x = x - step * (first / (1 - momentum ** k)) / (
                    np.sqrt(second / (1 - decay ** k)) + eps)
            elif method == "lion":
                x = x - step * np.sign(momentum * first + (1 - momentum) * g)
                first = decay * first + (1 - decay) * g
            else:
                first = momentum * first + (1 - momentum) * g
                second = decay * second + (1 - decay) * g * g
                x = x - step * (first / (1 - momentum ** k)) / (
                    np.sqrt(second / (1 - decay ** k)) + eps)
            if not np.all(np.isfinite(x)):
                return float("inf")
    gap = float(np.linalg.norm(x - problem["minimizer"]))
    return gap if np.isfinite(gap) else float("inf")


for scaling in (1.0, 20.0):
    problem = st.least_squares_problem(samples=2000, variables=10, scaling=scaling)
    print(f"\n  scaling {scaling:g}, kappa {problem['condition']:.1f}")
    print(f"{'step':>10}{'Adam':>14}{'AdaBelief':>14}{'Lion':>14}")
    for step in (1e-4, 1e-3, 1e-2, 1e-1):
        row = [modern(problem, m, step=step) for m in ("adam", "adabelief", "lion")]
        print(f"{step:>10.0e}" + "".join(f"{v:>14.3e}" for v in row))
    for method in ("adam", "adabelief", "lion"):
        best = min((modern(problem, method, step=s), s)
                   for s in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1))
        print(f"    best {method:>10}: {best[0]:.3e} at step {best[1]:.0e}")
```

**The three are within a factor of three of each other at their best steps**, on both scalings.
AdaBelief wins at a scaling of $1$, $9.7\times10^{-3}$ against Adam's $1.8\times10^{-2}$, and loses
slightly at a scaling of $20$. Lion is last on both, by a factor of $1.5$ to $2.5$.

**All three pick the same best step, $10^{-3}$**, and their whole step sweeps have the same shape.
That is the finding worth carrying away: across the sweep the step changes the answer by a factor of
$80$ and the choice of method changes it by a factor of $3$. **Tuning the step matters far more than
choosing among the adaptive methods**, which is what the published comparisons of these methods
generally find once the baselines are tuned properly.

Lion's usual advice is to use a step about ten times smaller than Adam's, because its update is
$\pm\alpha$ per coordinate regardless of the gradient. On this problem the measured best step is the
same for all three, so that adjustment is not needed here; on a problem where the gradients are much
smaller than one it would be.

### 4.1 Both exponents at once

```python
import numpy as np
from nalib import stochastic as st

problem = st.least_squares_problem(samples=2000, variables=10, scaling=1.0)
steps = (2e-3, 5e-3, 1.5e-2, 4e-2)
batches = (1, 4, 16, 64, 256)
repeats = 5
cells = []
print(f"{'step':>9}" + "".join(f"{f'b={b}':>12}" for b in batches))
for s in steps:
    line = []
    for b in batches:
        trials = [st.stochastic_descent(problem, "sgd", step=s, batch=b, rounds=30000,
                                        seed=k)["final_distance"]
                  for k in range(repeats)]
        radius = float(np.median(trials))
        line.append(radius)
        if np.isfinite(radius) and radius > 0.0:
            cells.append([1.0, np.log(s), np.log(b), np.log(radius)])
    print(f"{s:>9.1e}" + "".join(f"{v:>12.3e}" for v in line))

table = np.asarray(cells)
coefficients, *_ = np.linalg.lstsq(table[:, :3], table[:, 3], rcond=None)
predicted = table[:, :3] @ coefficients
print(f"\n  fitted:    radius ~ step^{coefficients[1]:.4f} times batch^{coefficients[2]:.4f}")
print(f"  predicted: radius ~ step^0.5000 times batch^-0.5000")
print(f"  worst relative miss over {table.shape[0]} cells: "
      f"{float(np.max(np.abs(np.exp(predicted - table[:, 3]) - 1.0))):.1%}")
```

Fitting both exponents together on a twenty cell grid gives

$$
r \;\propto\; \alpha^{0.505}\, b^{-0.548} ,
$$

against the predicted $\alpha^{0.5} b^{-0.5}$. The step exponent is within one per cent and the batch
exponent within ten, with the worst single cell twenty five per cent off the fitted surface.

The batch exponent being slightly steeper than $-1/2$ is expected rather than an error. The
derivation in 2.1 treats the residual as pure noise, but at large batch the iterate is close enough
to the deterministic fixed point that the noise ball is no longer the only term, so the measured
radius falls a little faster than the pure noise prediction.

The two exponents being equal and opposite is the practically useful part: **only $\alpha/b$ appears
in the radius**, so doubling the batch and doubling the step leave the accuracy unchanged while
halving the number of steps. That is the linear scaling rule used to keep large batch training
equivalent to small batch training, and this is where it comes from.

### 4.2 The stability edge against the condition number

```python
import numpy as np
from nalib import stochastic as st


def diverges(problem, method, step, rounds=3000, batch=16):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = st.stochastic_descent(problem, method, step=step, batch=batch,
                                    rounds=rounds)
    gap = out["final_distance"]
    return (not np.isfinite(gap)) or gap > 1e4


print(f"{'scaling':>9}{'kappa':>10}{'L':>11}{'2/L':>11}{'sgd edge':>12}"
      f"{'sgd / (2/L)':>13}{'momentum edge':>15}{'adam edge':>12}")
for scaling in (1.0, 3.0, 10.0, 30.0, 100.0):
    problem = st.least_squares_problem(samples=2000, variables=10, scaling=scaling)
    L = problem["smoothness"]
    edges = {}
    for method in ("sgd", "momentum", "adam"):
        low, high = 1e-9, 1e5
        for _ in range(45):
            mid = np.sqrt(low * high)
            if diverges(problem, method, mid):
                high = mid
            else:
                low = mid
        edges[method] = low
    print(f"{scaling:>9.0f}{problem['condition']:>10.1f}{L:>11.2f}{2.0 / L:>11.2e}"
          f"{edges['sgd']:>12.2e}{edges['sgd'] / (2.0 / L):>13.3f}"
          f"{edges['momentum']:>15.2e}{edges['adam']:>12.2e}")
```

Three completely different behaviours over four decades of $\kappa$.

**SGD's edge is $2/L$ and nothing else.** The ratio is $1.06, 1.07, 1.04, 1.05$ at the four larger
scalings, so the condition number does not enter at all. That is the deterministic bound of lesson
86, and the sampled gradient does not change it because the noise is orthogonal to the contraction on
average. The one exception is $\kappa = 1.3$, where the ratio is $0.73$: there the problem is so easy
that the noise is the dominant term and it costs a little stability.

**Momentum's edge is smaller than SGD's, and the gap closes as $\kappa$ grows.** The ratio to SGD is
$4.1, 2.9, 1.9, 1.5, 1.4$ across the sweep. So the price of momentum is a smaller usable step, and
the price falls exactly where momentum is worth having.

**Adam's edge is enormous and independent of everything.** It sits between $2.8\times10^{4}$ and
$10^{5}$ regardless of $L$ or $\kappa$, where SGD's ranges over four decades from $0.64$ to
$10^{-4}$. The reason is structural: Adam's update is $\hat m/\sqrt{\hat v}$, which is $O(1)$ in each
coordinate however large the gradients are, so $\alpha$ is a **distance in parameter space** rather
than a multiple of $1/L$.

That is the most practically important fact about Adam and it is not usually stated this way. A step
of $10^{-3}$ works across problems whose smoothness differs by four decades, because the step size no
longer has units of $1/L$. It is also why Adam's step needs a schedule: without one, that $O(1)$
update keeps injecting the same noise forever.

### 4.3 The best schedule against the budget

```python
import numpy as np
from nalib import stochastic as st

problem = st.least_squares_problem(samples=2000, variables=10, scaling=1.0)
base = 0.3 / problem["smoothness"]
names = ("constant", "one over k", "one over root k", "cosine")
repeats = 3
print(f"{'budget':>10}" + "".join(f"{n:>19}" for n in names) + f"{'best':>18}")
for rounds in (100, 1000, 10000, 100000, 1000000):
    row = []
    for name in names:
        rule = (st.cosine_schedule(base, rounds) if name == "cosine"
                else st.schedule(name, base))
        trials = [st.stochastic_descent(problem, "sgd", rule=rule, batch=16,
                                        rounds=rounds, seed=s)["final_distance"]
                  for s in range(repeats)]
        row.append(float(np.median(trials)))
    print(f"{rounds:>10}" + "".join(f"{v:>19.4e}" for v in row)
          + f"{names[int(np.argmin(row))]:>18}")
```

Four schedules and four different stories.

**A constant step is flat**, $0.13$ to $0.16$ at every budget from a hundred steps to a million.
Ten thousand times the work buys nothing, which is 1.1 confirmed at scale. This is the single
clearest reason a schedule is not optional.

**$1/k$ is always the worst**, though it improves steadily, $0.76 \to 0.074$. It decays too fast
early: at a hundred steps it is fifteen times worse than $1/\sqrt k$. The schedule is asymptotically
optimal and practically poor unless the constant is tuned to $\mu$, which requires knowing $\mu$.

**$1/\sqrt k$ and cosine are the two contenders**, within ten per cent of each other up to ten
thousand steps.

**Cosine wins at large budgets by a growing margin**, $2.4\times10^{-3}$ against $5.0\times10^{-3}$
at a million steps, a factor of $2.05$.

The reason cosine pulls ahead is that it is the only one of the four that **knows the budget**. It
holds a large step through most of the run, where a large step makes fast progress, and anneals to
zero at the end, where a small step gives a small noise ball. A fixed-form schedule has to hedge:
$1/\sqrt k$ is already small at step ten thousand even when a million are coming.

That is also the practical warning attached to cosine. Its shape depends on the total, so a run that
is stopped early has an annealing schedule for a longer run and ends at a large step, and a run that
is extended has already annealed to nothing. Neither failure is visible from the loss curve alone.

### 5.1 Which floor binds

**Lesson 84's rounding barrier.** Near a minimum, $f(x) \approx f^{*} + \tfrac{c}{2}d^{2}$, and
values within $\varepsilon\lvert f^{*}\rvert$ of $f^{*}$ are indistinguishable, giving

$$
d_{\text{round}} = \sqrt{\frac{\varepsilon\,\lvert f^{*}\rvert}{c}} .
$$

**This lesson's noise ball**, from 2.1 and 1.3,

$$
d_{\text{noise}} = \sqrt{\frac{\alpha\,\sigma^{2}}{2\mu\,b}} .
$$

Setting them equal and solving for the batch,

$$
b_{\text{cross}} = \frac{\alpha\,\sigma^{2}}{2\,\varepsilon\,\lvert f^{*}\rvert}
\qquad(\text{using } c = \mu) .
$$

```python
import numpy as np
from nalib import stochastic as st

problem = st.least_squares_problem(samples=2000, variables=10, scaling=1.0)
mu = problem["strong_convexity"]
rounding = float(np.sqrt(np.finfo(float).eps * abs(problem["minimum"]) / mu))
step = 0.05 / problem["smoothness"]
repeats = 5
print(f"  the rounding barrier sqrt(eps |f*| / c) = {rounding:.3e}")
print(f"  (f* = {problem['minimum']:.4f}, c = mu = {mu:.4f}, step = {step:.3e})")
print(f"\n{'batch':>8}{'measured radius':>18}{'rounding barrier':>19}{'which binds':>14}")
for b in (1, 4, 16, 64, 256, 1024, 2000):
    trials = [st.stochastic_descent(problem, "sgd", step=step, batch=b, rounds=40000,
                                    seed=s)["final_distance"] for s in range(repeats)]
    radius = float(np.median(trials))
    print(f"{b:>8}{radius:>18.3e}{rounding:>19.3e}"
          f"{('noise' if radius > rounding else 'rounding'):>14}")

reference = float(np.median([st.stochastic_descent(problem, "sgd", step=step, batch=16,
                                                   rounds=40000, seed=s)["final_distance"]
                             for s in range(repeats)]))
constant = reference * np.sqrt(16.0)
print(f"\n  fitting radius = C / sqrt(batch) with C = {constant:.4f},")
print(f"  the two cross at batch = {(constant / rounding) ** 2:.2e}, "
      f"against a data set of {problem['samples']}")

x = problem["start"].copy()
for _ in range(40000):
    x = x - (1.0 / problem["smoothness"]) * problem["gradient"](x)
print(f"\n  and with the exact full gradient, no sampling at all: "
      f"{float(np.linalg.norm(x - problem['minimizer'])):.3e}")
```

**The noise floor binds by a factor of a million**, at every batch size tried. Even at $b = 2000$,
the full size of the data set, the radius is $5.3\times10^{-3}$ against a rounding barrier of
$5.6\times10^{-9}$. The crossover batch is $8.4\times10^{14}$, which is $4\times10^{11}$ times the
size of the whole data set.

Two clarifications the measurement forces, and both matter.

**The residual at $b = N$ is sampling with replacement, not noise in the data.** A bootstrap sample
of size $N$ is not the whole data set, so there is still variance. The last line confirms it:
the exact full gradient converges to $3.0\times10^{-15}$, which is below the rounding barrier.

**And that is below the barrier because the barrier is about a different measurement.** Lesson 84's
$\sqrt\varepsilon$ limit applies to locating a minimum by comparing **function values**, which is what
the derivative free methods of lesson 85 do. Gradient descent reads the **gradient**, which is not
flat at the minimum and is not subject to the same cancellation, so it can and does get closer.

So the honest answer to the exercise is that in stochastic optimization **the rounding floor is
irrelevant.** It is six orders of magnitude below the noise floor and would need a batch size larger
than any data set that exists. Everything lesson 84 says about $\sqrt\varepsilon$ applies to
deterministic problems, and everything about accuracy here is decided by $\alpha$, $b$ and $\sigma$.

### 5.2 Momentum under noise

Momentum's $\sqrt\kappa$ result of 2.3 is a statement about how fast a **deterministic** error
contracts. Measuring what survives when the gradient is sampled gives a different answer.

```python
import numpy as np
from nalib import stochastic as st

repeats = 3
print(f"{'scaling':>9}{'kappa':>10}{'best sgd':>13}{'best momentum':>15}{'gain':>8}"
      f"{'sqrt(kappa) predicts':>22}")
for scaling in (1.0, 3.0, 10.0, 30.0, 100.0):
    problem = st.least_squares_problem(samples=2000, variables=10, scaling=scaling)
    best = {}
    for method in ("sgd", "momentum"):
        reached = []
        for power in np.arange(-6.0, 0.6, 0.5):
            with np.errstate(over="ignore", invalid="ignore"):
                trials = [st.stochastic_descent(problem, method, step=10.0 ** power,
                                                batch=16, rounds=20000,
                                                seed=k)["final_distance"]
                          for k in range(repeats)]
            value = float(np.median(trials))
            if np.isfinite(value):
                reached.append(value)
        best[method] = min(reached) if reached else float("inf")
    print(f"{scaling:>9.0f}{problem['condition']:>10.1f}{best['sgd']:>13.3e}"
          f"{best['momentum']:>15.3e}{best['sgd'] / best['momentum']:>8.2f}"
          f"{np.sqrt(problem['condition']):>22.2f}")
```

**The acceleration does not appear.** At $\kappa = 1.3$, $9.6$, $106$ and $956$ the measured gain is
$0.99$, $0.96$, $0.92$ and $0.80$: momentum is not better, and at the largest of those it is
slightly worse, where $\sqrt\kappa$ predicts a factor of $31$.

**Why.** The accuracy each method reaches is set by its noise ball, not by its rate. Every entry in
both columns sits between $4.9\times10^{-3}$ and $1.0\times10^{-2}$, essentially independent of
$\kappa$ and of the method, because both are wandering in a ball whose radius depends on $\alpha$,
$b$ and $\sigma$. **A faster method reaches the floor sooner; it does not reach a lower floor.** And
with twenty thousand steps available, both reach it.

Momentum makes the noise slightly worse, which is the $0.80$ at $\kappa = 956$. Averaging the
gradient over past steps with weight $\beta$ multiplies the effective step by $1/(1-\beta)$, so at
the same nominal $\alpha$ the noise ball is larger by $1/\sqrt{1-\beta}$, a factor of $3.2$ at
$\beta = 0.9$. Tuning $\alpha$ absorbs most of that, which is why the loss is only twenty per cent.

**The one row where momentum wins is the exception that proves the rule.** At $\kappa = 10618$ the
gain is $46.9$, and the reason is visible in the numbers: SGD's best is $0.486$, which is not a noise
ball at all but a run that has not arrived. Its stability edge there is $10^{-4}$, so in twenty
thousand steps it cannot cross the distance, and momentum's $\sqrt\kappa$ advantage in **transient**
speed is exactly what is needed.

So the rule is: **acceleration helps while the deterministic error dominates and stops helping once
the noise does.** Early in a run, where the iterate is far away, momentum is worth having. Late in a
run it is not, and what matters instead is the schedule, the batch and the averaging of 3.3. That is
also why deep learning practice keeps momentum but pairs it with an annealing schedule rather than
relying on it alone.

### 5.3 Why adaptive methods are hard to analyse

**What breaks.** Every convergence proof for SGD leans on one step:

$$
\mathbb{E}\big[\alpha_k\, g_k \,\big|\, x_k\big] = \alpha_k\,\nabla f(x_k) ,
$$

taking the step size **outside** the expectation because $\alpha_k$ is a deterministic schedule and
$g_k$ is unbiased. Everything else in the analysis, the descent lemma, the telescoping sum, the
Robbins-Monro conditions, is built on top of that identity.

For an adaptive method the step is $\alpha/(\sqrt{v_k}+\epsilon)$ and $v_k$ depends on
$g_1,\dots,g_k$, including $g_k$ itself. So

$$
\mathbb{E}\left[\frac{\alpha\, g_k}{\sqrt{v_k}+\epsilon}\right]
\;\ne\; \frac{\alpha}{\sqrt{\mathbb{E}v_k}+\epsilon}\;\nabla f(x_k) ,
$$

because a numerator and a denominator that are correlated do not separate. Worse, the correlation has
a sign: a large $g_k$ makes $v_k$ large, so the step is systematically shortened in exactly the
directions where the gradient sample happened to be large. **The effective update is a biased
estimate of the descent direction**, and the bias does not go away as the step shrinks.

That is not a technicality. Reddi, Kale and Kumar exhibited a one dimensional convex problem on which
Adam converges to the wrong point, and the mechanism is exactly this: a rare large gradient that
should dominate the average has its own step size cut by its own magnitude, so it is systematically
underweighted.

**How the published fixes get around it.** Four approaches, in rough historical order.

**Change the algorithm so $v_k$ is non-decreasing.** AMSGrad keeps $\hat v_k = \max(\hat v_{k-1},
v_k)$, so the effective step is monotonically non-increasing and can be bounded by a deterministic
schedule from above. That restores enough of the separation to push the proof through, and it fixes
the counterexample. The cost is that a coordinate whose gradients shrink never recovers its step.

**Delay the denominator by one step.** Using $v_{k-1}$ instead of $v_k$ makes the step
$\mathcal{F}_{k-1}$-measurable, so it comes out of the conditional expectation exactly as in the SGD
proof. This is the cleanest fix and it changes the method almost not at all, which is why several
later analyses adopt it.

**Assume the gradients are bounded.** If $\lVert g_k\rVert \le G$ then the step lies in
$[\alpha/(G+\epsilon),\,\alpha/\epsilon]$, a deterministic interval, and one can prove convergence at
the rate of SGD with the worse of the two endpoints. That is what most of the early Adam analyses do,
and it is why they need the bounded gradient assumption that practice regularly violates.

**Prove something weaker.** Much of the modern literature proves convergence to a stationary point in
the non-convex setting, $\min_k \mathbb{E}\lVert\nabla f(x_k)\rVert^{2} \to 0$, rather than
convergence of the iterates, and accepts constants that depend on $\epsilon$ and on the gradient
bound.

**The honest position** is the one the lesson takes elsewhere about Nelder-Mead: the methods work,
they are used everywhere, and the theory is thinner than the practice. What the theory does supply is
the diagnosis of when they will fail, and the counterexamples all have the same shape, a rare large
gradient carrying most of the signal. That is a real situation in practice, and the standard defence
is the gradient clipping of 3.2, which bounds $\lVert g\rVert$ by hand and thereby makes the third
fix's assumption true by construction.

---

## Lesson 90, Proximal and Composite Optimization

### 1.1 What goes wrong with subgradient descent

Subgradient descent replaces $\nabla f$ by any element of the subdifferential and shrinks the step to
force convergence. It converges, and it fails in three separate ways.

**It is slow.** The rate is $O(1/\sqrt k)$ in the objective gap, against proximal gradient's $O(1/k)$
and FISTA's $O(1/k^{2})$. Reaching $10^{-6}$ takes a million steps instead of a million's square
root. The lesson measures the exponent at close to $-1/2$.

**It is not monotone.** A subgradient is not a descent direction at a kink, so the objective can rise,
and any stopping test built on "the objective stopped falling" is unusable.

**It never produces an exact zero.** The subgradient of $\lambda\lVert x\rVert_1$ at $x_j \ne 0$ is
$\lambda\operatorname{sign}(x_j)$, so a step from a non-zero $x_j$ lands at
$x_j - \alpha\lambda\operatorname{sign}(x_j)$, which is zero only by coincidence. The iterate is
dense at every step, and the whole reason for using an $\ell_1$ penalty is lost.

That third failure is the decisive one and it is why the proximal operator exists. The lesson
measures the subgradient iterate at every one of its coordinates non-zero, against FISTA's exactly
sparse answer on the same problem with the same penalty.

### 1.2 The proximal operator

$$
\operatorname{prox}_{tg}(v) \;=\; \arg\min_{x}\; g(x) + \frac{1}{2t}\lVert x - v\rVert^{2} .
$$

It is a compromise between decreasing $g$ and staying near $v$, with $t$ setting the exchange rate.
For convex $g$ the objective is strongly convex, so the minimizer exists and is unique, and the
operator is well defined everywhere.

**Two closed forms.**

*Soft thresholding*, for $g = \lambda\lVert x\rVert_1$:

$$
\operatorname{prox}_{t\lambda\lVert\cdot\rVert_1}(v)_j
= \operatorname{sign}(v_j)\,\max\big(\lvert v_j\rvert - t\lambda,\; 0\big) ,
$$

which is exercise 2.1. It is separable because the $\ell_1$ norm is a sum over coordinates.

*The ridge prox*, for $g = \tfrac{\lambda}{2}\lVert x\rVert_2^{2}$:

$$
\operatorname{prox}_{tg}(v) = \frac{v}{1 + t\lambda} ,
$$

by differentiating $\tfrac{\lambda}{2}\lVert x\rVert^{2} + \tfrac{1}{2t}\lVert x - v\rVert^{2}$ and
solving. It shrinks every coordinate by the same factor and never reaches zero, which is 1.3.

A third worth knowing: the prox of an indicator function is a projection, which is exercise 2.2 and
connects this lesson to lesson 88.

### 1.3 Why soft thresholding gives exact zeros

The difference is in the derivative at the origin.

**$\lvert x\rvert$ has a kink at zero**, with subdifferential $[-1, 1]$ there. The prox condition
$0 \in \partial g(x) + (x - v)/t$ at $x = 0$ reads $0 \in t\lambda[-1,1] - v$, which holds whenever
$\lvert v\rvert \le t\lambda$. So **a whole interval of inputs maps to exactly zero**, and its width
is $2t\lambda$.

**$x^{2}/2$ is smooth at zero**, with derivative $0$ there. The condition $\lambda x + (x-v)/t = 0$
gives $x = v/(1+t\lambda)$, which is zero only when $v$ is exactly zero. A single point maps to zero,
which has probability zero for any continuous input.

The general principle: **exact zeros require a non-smooth penalty at the origin.** The kink is what
makes a range of inputs collapse, and any penalty that is differentiable at zero produces a dense
answer no matter how large its weight is.

The lesson measures this three ways at once: FISTA on the $\ell_1$ objective gives exact zeros, the
same FISTA on the ridge objective gives none, and the subgradient method on the same $\ell_1$
objective also gives none. That rules out both wrong explanations, that the penalty alone does it and
that the method alone does it.

### 1.4 $\lambda_{\max}$

$$
\lambda_{\max} = \frac{\lVert A^{T}b\rVert_\infty}{N} ,
$$

for the objective $\frac{1}{2N}\lVert Ax - b\rVert^{2} + \lambda\lVert x\rVert_1$.

**Why the answer above it is zero.** Optimality at a point $x$ requires
$0 \in \nabla f(x) + \lambda\partial\lVert x\rVert_1$. At $x = 0$ the gradient of the smooth part is
$-A^{T}b/N$ and the subdifferential of the norm is the unit $\ell_\infty$ ball, so the condition is

$$
\frac{A^{T}b}{N} \in \lambda\,[-1,1]^{m}
\quad\Longleftrightarrow\quad
\left\lVert\frac{A^{T}b}{N}\right\rVert_\infty \le \lambda .
$$

So $x = 0$ is optimal exactly when $\lambda \ge \lambda_{\max}$, and by convexity it is then the
unique optimum. Exercise 2.3 does the derivation in full.

**It is exact, not a bound.** The lesson measures that every coordinate is zero at
$\lambda_{\max}(1+\epsilon)$ and some coordinate is non-zero at $\lambda_{\max}(1-\epsilon)$, to
rounding. That makes it the right unit for the penalty: a regularization path is always parameterized
by $\lambda/\lambda_{\max}$ from $1$ down, which is what every LASSO package does.

### 1.5 What FISTA gives up

**Monotonicity.** ISTA's objective decreases at every step, because each step is a proximal gradient
step from the current point and the descent lemma applies. FISTA takes its gradient step from an
**extrapolated** point $y_k = x_k + \frac{t_k-1}{t_{k+1}}(x_k - x_{k-1})$, which lies beyond $x_k$
along the last direction and can have a larger objective. The resulting $x_{k+1}$ can then be worse
than $x_k$.

That is not a bug and exercise 5.1 explains why the sequence still converges: the theory bounds a
Lyapunov function combining the objective gap and the distance to the answer, not the objective
alone.

**What it costs in practice.** Any stopping test of the form "stop when the objective stops falling"
is wrong for FISTA. The correct tests are the size of $x_{k+1} - x_k$, or the objective compared with
the best seen so far, or the fixed point residual.

**What it buys.** $O(1/k^{2})$ instead of $O(1/k)$ in the worst case over the problem class, which is
optimal for first order methods on this class. On a specific instance the gain can be much larger or
absent altogether, and exercise 4.1 finds the latter more often than the former.

### 2.1 Deriving soft thresholding

The $\ell_1$ prox is separable, so solve one coordinate at a time:

$$
\min_{x}\; \lambda\lvert x\rvert + \frac{1}{2t}(x - v)^{2} .
$$

Write $\tau = t\lambda$ and multiply by $t$, so the problem is
$\min_x \tau\lvert x\rvert + \tfrac12 (x-v)^{2}$. Split on the sign of $x$.

**For $x > 0$:** the objective is differentiable, $\tau + (x - v) = 0$, giving $x = v - \tau$. This
is a valid candidate only if $v - \tau > 0$, that is $v > \tau$.

**For $x < 0$:** $-\tau + (x-v) = 0$, giving $x = v + \tau$, valid when $v < -\tau$.

**For $x = 0$:** optimality needs $0 \in \tau[-1,1] + (0 - v)$, that is $v \in [-\tau, \tau]$.

The three cases partition the line, and combining them,

$$
\boxed{\;\operatorname{prox}(v) = \operatorname{sign}(v)\,\max(\lvert v\rvert - \tau,\; 0)\;}
$$

The interval $[-\tau, \tau]$ mapping to zero is the exact zeros of 1.3, and its width $2\tau =
2t\lambda$ is why a larger penalty or a larger step produces a sparser answer.

```python
import numpy as np
from nalib import proximal as px

rng = np.random.default_rng(42)
print(f"{'v':>10}{'closed form':>14}{'brute force minimum':>22}{'gap':>11}")
tau = 0.7
grid = np.linspace(-4.0, 4.0, 800001)
for v in (-2.0, -0.7, -0.3, 0.0, 0.5, 1.6):
    values = tau * np.abs(grid) + 0.5 * (grid - v) ** 2
    print(f"{v:>10.2f}{float(px.soft_threshold(np.array([v]), tau)[0]):>14.6f}"
          f"{grid[int(np.argmin(values))]:>22.6f}"
          f"{abs(float(px.soft_threshold(np.array([v]), tau)[0]) - grid[int(np.argmin(values))]):>11.1e}")
print(f"\nthe interval mapping to zero is [-{tau}, {tau}], width {2 * tau}")
point = rng.uniform(-1.0, 1.0, size=20000) * 2.0
zeroed = float(np.mean(px.soft_threshold(point, tau) == 0.0))
print(f"and the measured fraction of random inputs sent to zero is {zeroed:.4f} "
      f"against the predicted {2 * tau / 4.0:.4f}")
```

The closed form matches a brute force minimization to the grid spacing at every input, and the
fraction of uniform inputs sent to exactly zero matches the interval width divided by the range.

### 2.2 The prox of an indicator is the projection

Let $\iota_C$ be the indicator of a set $C$: zero on $C$ and $+\infty$ off it. Then

$$
\operatorname{prox}_{t\,\iota_C}(v)
= \arg\min_x\; \iota_C(x) + \frac{1}{2t}\lVert x - v\rVert^{2} .
$$

Any $x \notin C$ makes the objective $+\infty$, so the minimization is confined to $C$. On $C$ the
first term is zero, so the problem is

$$
\arg\min_{x \in C}\; \frac{1}{2t}\lVert x - v\rVert^{2}
= \arg\min_{x \in C}\; \lVert x - v\rVert^{2}
= P_C(v) ,
$$

the positive factor $1/2t$ not affecting the minimizer. **This holds for every $t > 0$.**

$\square$

Two consequences worth stating.

**Projected gradient was proximal gradient all along**, with $g = \iota_C$. So everything lesson 88
proved about projected gradient is a special case, and everything this lesson proves about proximal
gradient applies to it.

**Feasibility is exactness.** A projection puts the iterate exactly on the set, and a prox puts it
exactly at a kink of $g$. Those are the same phenomenon, and both come from $g$ being non-smooth.

```python
import numpy as np
from nalib import proximal as px

out = px.the_prox_of_an_indicator_is_a_projection()
print(f"over {out['samples']} random points, worst difference between the prox and "
      f"the projection: {out['worst_difference']:.1e}")
print(f"they are the same operator: {out['they_are_the_same_operator']}")
rng = np.random.default_rng(42)
point = rng.normal(scale=3.0, size=6)
print("\nand the step size does not change the answer:")
for t in (0.01, 1.0, 100.0):
    print(f"  t = {t:>7.2f}: {np.array2string(px.prox_box(point, -1.0, 1.0), precision=6)}")
```

The two routines are written independently in two modules and agree **exactly**, and the answer does
not depend on $t$, both as the derivation requires.

### 2.3 Deriving $\lambda_{\max}$

For $F(x) = f(x) + \lambda\lVert x\rVert_1$ with $f(x) = \frac{1}{2N}\lVert Ax - b\rVert^{2}$,
optimality at $x$ is

$$
0 \in \nabla f(x) + \lambda\,\partial\lVert x\rVert_1 .
$$

The subdifferential of $\lVert\cdot\rVert_1$ at a point $x$ is the set of $u$ with
$u_j = \operatorname{sign}(x_j)$ where $x_j \ne 0$ and $u_j \in [-1,1]$ where $x_j = 0$.

**At $x = 0$** every coordinate is free, so $\partial\lVert 0\rVert_1 = [-1,1]^{m}$, and

$$
\nabla f(0) = \frac{1}{N}A^{T}(A\cdot 0 - b) = -\frac{A^{T}b}{N} .
$$

The condition becomes: there exists $u \in [-1,1]^{m}$ with
$-\frac{A^{T}b}{N} + \lambda u = 0$, that is $u = \frac{A^{T}b}{\lambda N}$, and such a $u$ lies in
the box exactly when every coordinate has modulus at most one:

$$
\left\lvert\frac{(A^{T}b)_j}{N}\right\rvert \le \lambda \quad\text{for all } j
\quad\Longleftrightarrow\quad
\lambda \ge \frac{\lVert A^{T}b\rVert_\infty}{N} =: \lambda_{\max} .
$$

$\square$

The $\ell_\infty$ norm appears because the subdifferential of the $\ell_1$ norm is the unit ball of
its **dual** norm, which is $\ell_\infty$. That is the general pattern: for $g = \lambda\lVert
\cdot\rVert$ the threshold is $\lVert\nabla f(0)\rVert_{*}$ in the dual norm, so for a group penalty
it is a maximum over groups of $\ell_2$ norms, and for the nuclear norm of exercise 3.4 it is the
largest singular value.

```python
import numpy as np
from nalib import proximal as px

out = px.the_threshold_where_everything_vanishes()
print(f"the formula is exact:      {out['the_formula_is_exact']}")
print(f"everything above is zero:  {out['everything_above_is_zero']}")
print(f"everything below is not:   {out['everything_below_is_not']}")
print()
for samples, variables, sparsity in ((60, 200, 10), (200, 60, 5), (100, 100, 8)):
    problem = px.lasso_problem(samples=samples, variables=variables, sparsity=sparsity)
    computed = float(np.max(np.abs(problem["design"].T @ problem["targets"]))) / samples
    print(f"  {problem['name']:<44} lambda_max {problem['lambda_max']:.8f}, "
          f"formula {computed:.8f}, gap {abs(computed - problem['lambda_max']):.1e}")
    for factor in (1.001, 0.999):
        got = px.proximal_gradient(problem, factor * problem["lambda_max"],
                                   accelerated=True, rounds=4000, record=False)
        print(f"      at {factor} lambda_max: {got['nonzeros']} nonzeros")
```

The threshold is exact at every size: one part in a thousand above it every coordinate is zero, and
one part in a thousand below it some coordinate is not.

### 2.4 Every prox is firmly nonexpansive

**Claim.** For convex $g$ and any $u, v$,

$$
\big\langle \operatorname{prox}(u) - \operatorname{prox}(v),\; u - v \big\rangle
\;\ge\; \lVert \operatorname{prox}(u) - \operatorname{prox}(v)\rVert^{2} .
$$

**Proof.** Write $p = \operatorname{prox}_{tg}(u)$ and $q = \operatorname{prox}_{tg}(v)$. The
optimality conditions are

$$
\frac{u - p}{t} \in \partial g(p) , \qquad \frac{v - q}{t} \in \partial g(q) .
$$

Monotonicity of the subdifferential of a convex function says
$\langle s_1 - s_2,\; p - q\rangle \ge 0$ for $s_1 \in \partial g(p)$ and $s_2 \in \partial g(q)$.
Applying it to these two,

$$
\left\langle \frac{u-p}{t} - \frac{v-q}{t},\; p - q \right\rangle \ge 0 ,
$$

and multiplying by $t > 0$ and expanding,

$$
\langle u - v,\; p - q\rangle - \langle p - q,\; p - q\rangle \ge 0 ,
$$

which is the claim. $\square$

**Nonexpansiveness follows**, by Cauchy-Schwarz:

$$
\lVert p - q\rVert^{2} \le \langle u-v,\; p-q\rangle \le \lVert u-v\rVert\,\lVert p-q\rVert
\;\Longrightarrow\;
\lVert p - q\rVert \le \lVert u - v\rVert .
$$

That is what makes proximal gradient converge at the same rate as gradient descent: the prox cannot
amplify the error the gradient step leaves behind. Firm nonexpansiveness is the stronger property and
it is what makes the fixed point iteration converge without any step size condition on the prox side.

```python
import numpy as np
from nalib import proximal as px

out = px.every_prox_is_firmly_nonexpansive()
print(f"{'prox':>16}{'worst ratio ||p-q|| / ||u-v||':>32}{'nonexpansive':>15}")
for row in out["rows"]:
    print(f"{row['operator']:>16}"
          f"{row['worst_ratio']:>32.6f}{str(row['nonexpansive']):>15}")
print(f"\nall nonexpansive: {out['all_nonexpansive']}, "
      f"worst ratio anywhere {out['worst_ratio_anywhere']:.6f}")

rng = np.random.default_rng(42)
worst_firm = np.inf
for _ in range(20000):
    u, v = rng.normal(size=5), rng.normal(size=5)
    p, q = px.soft_threshold(u, 0.6), px.soft_threshold(v, 0.6)
    gap = p - q
    if float(gap @ gap) > 1e-18:
        worst_firm = min(worst_firm, float((u - v) @ gap) / float(gap @ gap))
print(f"  and firmly: the smallest <u-v, p-q> / ||p-q||^2 is {worst_firm:.6f}, "
      f"which must be at least 1")
```

The nonexpansive ratio never exceeds one, and the firmness quotient never falls below one, over
twenty thousand random pairs. Both are identities, so anything above rounding would be a bug rather
than a tolerance question.

### 2.5 The group prox

For $g(x) = \lambda\sum_{G}\lVert x_G\rVert_2$, a sum of $\ell_2$ norms over disjoint groups, the
prox separates across groups, so solve one group:

$$
\min_{z}\; \lambda\lVert z\rVert_2 + \frac{1}{2t}\lVert z - w\rVert^{2} .
$$

**The direction is fixed.** Both terms depend on $z$ only through $\lVert z\rVert$ and
$\langle z, w\rangle$, and for a fixed length the second term is smallest when $z$ points along $w$.
So $z = \rho\, w/\lVert w\rVert$ for some $\rho \ge 0$, and the problem reduces to one dimension:

$$
\min_{\rho \ge 0}\; \lambda\rho + \frac{1}{2t}(\rho - \lVert w\rVert)^{2} ,
$$

which is exactly the one sided soft thresholding problem of 2.1 with $v = \lVert w\rVert > 0$, giving
$\rho = \max(\lVert w\rVert - t\lambda, 0)$. Therefore

$$
\boxed{\;\operatorname{prox}(w) = \frac{w}{\lVert w\rVert}\,\max\big(\lVert w\rVert - t\lambda,\,0\big)
= w\left(1 - \frac{t\lambda}{\lVert w\rVert}\right)_{+}\;}
$$

**It zeroes whole groups.** The answer is exactly zero when $\lVert w\rVert \le t\lambda$, and when
that happens **every coordinate of the group goes to zero at once**, because the whole vector is
scaled by the same factor. When it does not happen, no coordinate of the group is zero unless it was
zero in $w$.

That is the point of the penalty: it selects at the level of groups rather than coordinates, which is
what is wanted when the coordinates come in natural bundles, one per categorical level or one per
filter.

```python
import numpy as np
from nalib import proximal as px

groups = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
rng = np.random.default_rng(42)
print(f"{'group':>7}{'||w||':>10}{'t lambda':>11}{'||prox||':>11}"
      f"{'predicted':>12}{'all zero':>11}{'any zero':>11}")
amount = 1.0
point = np.concatenate([0.2 * rng.normal(size=3), 3.0 * rng.normal(size=2),
                        0.1 * rng.normal(size=4)])
answer = px.prox_group(point, amount, groups)
for k, block in enumerate(groups):
    size = float(np.linalg.norm(point[block]))
    got = float(np.linalg.norm(answer[block]))
    print(f"{k:>7}{size:>10.5f}{amount:>11.2f}{got:>11.5f}"
          f"{max(size - amount, 0.0):>12.5f}"
          f"{str(bool(np.all(answer[block] == 0.0))):>11}"
          f"{str(bool(np.any(answer[block] == 0.0))):>11}")
print("\nthe two boolean columns agree in every row, which is what "
      "'whole groups' means")
```

The group norms shrink by exactly $t\lambda$ or collapse to zero, and the last two columns agree in
every row: a group is either entirely zero or has no zero at all.

### 3.1 ADMM

ADMM splits the objective by introducing a copy: minimize $f(x) + g(z)$ subject to $x = z$, and
alternate between the two blocks and a multiplier update. For the LASSO with
$f = \frac{1}{2N}\lVert Ax-b\rVert^{2}$ and $g = \lambda\lVert z\rVert_1$ the steps are a **linear
solve**, a **soft threshold**, and an addition.

```python
import numpy as np
from nalib import proximal as px


def admm(problem, lam, rho=1.0, rounds=2000):
    A, b = problem["design"], problem["targets"]
    n, m = problem["samples"], problem["dimension"]
    factor = np.linalg.cholesky(A.T @ A / n + rho * np.eye(m))   # formed once
    base = A.T @ b / n
    x = np.zeros(m)
    z = np.zeros(m)
    u = np.zeros(m)
    values = []
    for _ in range(rounds):
        rhs = base + rho * (z - u)
        x = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
        z = px.soft_threshold(x + u, lam / rho)
        u = u + x - z
        values.append(px.objective(problem, z, lam))
    return {"x": z, "values": np.asarray(values), "objective": values[-1],
            "nonzeros": int(np.count_nonzero(z))}


for samples, variables, sparsity in ((100, 400, 20), (200, 200, 10), (400, 100, 5)):
    problem = px.lasso_problem(samples=samples, variables=variables, sparsity=sparsity)
    lam = 0.1 * problem["lambda_max"]
    reference = min(
        px.proximal_gradient(problem, lam, accelerated=True, rounds=200000)["objective"],
        admm(problem, lam, rounds=20000)["objective"])
    print(f"\n  {problem['name']}, lambda = 0.1 lambda_max")
    print(f"{'method':>14}{'to 1e-6':>11}{'to 1e-9':>11}{'to 1e-12':>12}{'nonzeros':>11}")
    runs = (("ISTA", px.proximal_gradient(problem, lam, rounds=20000)),
            ("FISTA", px.proximal_gradient(problem, lam, accelerated=True, rounds=20000)),
            ("ADMM, rho = 1", admm(problem, lam, rounds=20000)))
    for name, out in runs:
        gaps = np.asarray(out["values"]) - reference
        marks = []
        for target in (1e-6, 1e-9, 1e-12):
            hit = np.flatnonzero(gaps <= target)
            marks.append(str(int(hit[0])) if hit.size else "none")
        print(f"{name:>14}{marks[0]:>11}{marks[1]:>11}{marks[2]:>12}"
              f"{out['nonzeros']:>11}")
```

**ADMM wins on iteration count, by a factor of two to four.** On the underdetermined problem it
reaches $10^{-6}$ in $39$ steps against FISTA's $103$ and ISTA's $154$, and the margin holds at
tighter tolerances. All three find the same support.

**The per iteration cost is not the same, and the accounting decides whether it is a real win.** A
FISTA step is two matrix products, about $2nm$ operations, here $2\cdot 100\cdot 400 = 80000$. An
ADMM step is two triangular solves, $m^{2} = 160000$, so twice as expensive. And ADMM pays once for
the Cholesky, $m^{3}/3 = 2\times10^{7}$, which is $267$ FISTA steps.

So for a **single** solve at one $\lambda$, ADMM's advantage is thin or negative. Its case is the
**regularization path**: the factored matrix $A^{T}A/N + \rho I$ does not depend on $\lambda$, so it
is formed once and reused for every point of the path, and the amortized cost per solve is then just
the triangular solves. That is why ADMM is the standard choice for path computations and for
distributed settings, where the $x$ update splits across machines and only the $z$ update needs to
be shared.

Note also that ADMM's $\rho$ is a tuning parameter with no principled default, where FISTA's step is
determined by $L$. That is the other half of the trade.

### 3.2 Backtracking on the proximal step

The step $1/L$ needs $L$, which requires the largest eigenvalue of $A^{T}A/N$. Backtracking replaces
it with a test on the descent lemma:

$$
f(p) \;\le\; f(y) + \langle \nabla f(y),\, p - y\rangle + \frac{1}{2t}\lVert p - y\rVert^{2} ,
$$

with $p = \operatorname{prox}_{t\lambda}(y - t\nabla f(y))$. Halve $t$ until it holds.

```python
import numpy as np
from nalib import proximal as px


def backtracking_prox(problem, lam, rounds=3000, grow=1.1, guard=True):
    m = problem["dimension"]
    x = np.zeros(m)
    y = x.copy()
    step = 1.0
    f, gradient = problem["smooth"], problem["smooth_gradient"]
    values, chosen, calls = [], [], 0
    for _ in range(rounds):
        base, slope = f(y), gradient(y)
        calls += 1
        step *= grow
        for _ in range(60):
            trial = px.soft_threshold(y - step * slope, step * lam)
            gap = trial - y
            calls += 1
            model = base + float(slope @ gap) + float(gap @ gap) / (2.0 * step)
            slack = 8.0 * np.finfo(float).eps * abs(base) if guard else 0.0
            if f(trial) <= model + slack:
                break
            step *= 0.5
        chosen.append(step)
        x = y = trial
        values.append(px.objective(problem, x, lam))
    return {"x": x, "values": np.asarray(values), "steps": np.asarray(chosen),
            "smooth_calls": calls}


for samples, variables, sparsity in ((100, 400, 20), (400, 100, 5)):
    problem = px.lasso_problem(samples=samples, variables=variables, sparsity=sparsity)
    lam = 0.1 * problem["lambda_max"]
    reference = px.proximal_gradient(problem, lam, accelerated=True,
                                     rounds=200000)["objective"]
    known = px.proximal_gradient(problem, lam, rounds=3000)
    found = backtracking_prox(problem, lam, rounds=3000)
    print(f"\n  {problem['name']}: 1/L = {1.0 / problem['smoothness']:.5f}")
    for label, out, per_step in (("known L", known, 1.0),
                                 ("backtracking", found,
                                  found["smooth_calls"] / 3000.0)):
        gaps = np.asarray(out["values"]) - reference
        hit = np.flatnonzero(gaps <= 1e-9)
        print(f"    {label:>13}: {(int(hit[0]) if hit.size else -1):>4} steps to 1e-9, "
              f"{per_step:.2f} smooth evaluations per step")
    print(f"    the backtracking step settles at "
          f"{float(np.median(found['steps'][100:])):.4f}, which is "
          f"{float(np.median(found['steps'][100:])) * problem['smoothness']:.1f} "
          f"times 1/L")
```

The result is stronger than "it removes the need to know $L$".

**Backtracking is faster than using the true $L$**, by a factor of six in steps on the
underdetermined problem, $37$ against $226$, and by $2.6$ on the other, $8$ against $21$. The reason
is in the last line: the step settles at $8.4$ and $4.0$ times $1/L$, because $L$ is a **global**
Lipschitz constant and the curvature along the path the method actually takes is much smaller.
Backtracking finds the local constant and $L$ is a worst case over the whole space.

**It costs $2.14$ smooth evaluations per step** instead of one, so the net gain in evaluations is
$2.9$ on the first problem and $1.2$ on the second: a real win in both cases, and a large one where
the global bound is loosest.

**And the test needs a rounding guard.** Once the objective is at its floor, both sides of the descent
inequality are at the rounding level and the comparison is noise, so the test fails about half the
time and the step ratchets down by halving. Without the guard the step collapses from $1.0$ to
$10^{-9}$ within a few hundred iterations, and the method silently stops. With a slack of a few
$\varepsilon\lvert f\rvert$ the step stays healthy for the whole run.

That is a general point about backtracking tests near convergence and it is easy to miss, because the
reported step count is unaffected: the collapse happens **after** the tolerance was reached.

### 3.3 Monotone FISTA

Beck and Teboulle's MFISTA keeps the better of the new point and the old, and extrapolates from the
kept point:

```python
import numpy as np
from nalib import proximal as px


def monotone_fista(problem, lam, rounds=3000):
    m = problem["dimension"]
    step = 1.0 / problem["smoothness"]
    x = np.zeros(m)
    y = x.copy()
    t = 1.0
    gradient = problem["smooth_gradient"]
    values, swaps = [], 0
    for _ in range(rounds):
        z = px.soft_threshold(y - step * gradient(y), step * lam)
        if px.objective(problem, z, lam) <= px.objective(problem, x, lam):
            kept = z
        else:
            kept = x                                   # the guard, and the whole idea
            swaps += 1
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = kept + (t / t_next) * (z - kept) + ((t - 1.0) / t_next) * (kept - x)
        x, t = kept, t_next
        values.append(px.objective(problem, x, lam))
    return {"x": x, "values": np.asarray(values), "swaps": swaps}


for samples, variables, sparsity in ((100, 400, 20), (400, 100, 5)):
    problem = px.lasso_problem(samples=samples, variables=variables, sparsity=sparsity)
    lam = 0.1 * problem["lambda_max"]
    reference = px.proximal_gradient(problem, lam, accelerated=True,
                                     rounds=200000)["objective"]
    guarded = monotone_fista(problem, lam)
    print(f"\n  {problem['name']}")
    print(f"{'method':>18}{'to 1e-9':>10}{'to 1e-12':>11}{'monotone':>11}{'swaps':>8}")
    runs = (("ISTA", px.proximal_gradient(problem, lam, rounds=3000)["values"], 0),
            ("FISTA", px.proximal_gradient(problem, lam, accelerated=True,
                                           rounds=3000)["values"], 0),
            ("monotone FISTA", guarded["values"], guarded["swaps"]))
    for name, values, swaps in runs:
        gaps = np.asarray(values) - reference
        marks = [str(int(np.flatnonzero(gaps <= target)[0]))
                 if np.any(gaps <= target) else "none"
                 for target in (1e-9, 1e-12)]
        rises = int(np.sum(np.diff(np.asarray(values)) > 1e-15))
        print(f"{name:>18}{marks[0]:>10}{marks[1]:>11}{str(rises == 0):>11}{swaps:>8}")
```

**It is monotone, which is the point**, and the monotone column proves it against FISTA's `False`.

**It loses about eighteen per cent** where acceleration helps: $205$ steps to $10^{-9}$ against
FISTA's $173$ on the underdetermined problem. Where acceleration does not help it loses nothing:
$23$ against $24$ on the other.

So the guard is cheap. It costs one extra objective evaluation per step and about a fifth of the
speed in the case where the speed exists, and it buys back the ability to use "the objective stopped
falling" as a stopping test and to report a loss curve that a reader can interpret.

The swap count is high, but most swaps happen after convergence, where the two candidates differ only
at the rounding level and the comparison is a coin flip. The swaps that matter are the early ones.

### 3.4 The nuclear norm and matrix completion

The nuclear norm $\lVert X\rVert_{*} = \sum_i \sigma_i(X)$ is the $\ell_1$ norm of the singular
values, and its prox is soft thresholding applied to the spectrum:

$$
\operatorname{prox}_{t\lambda\lVert\cdot\rVert_*}(M) = U\,\mathcal{S}_{t\lambda}(\Sigma)\,V^{T} ,
$$

with $M = U\Sigma V^{T}$ the SVD of lesson 41. The derivation is the same as 2.1's, applied after
noting that the problem is invariant under the orthogonal factors, so the minimizer shares $M$'s
singular vectors.

```python
import numpy as np
from nalib import proximal as px


def prox_nuclear(matrix, amount):
    """Soft threshold the singular values: the same formula, one level up."""
    left, values, right = np.linalg.svd(np.asarray(matrix, dtype=float),
                                        full_matrices=False)
    shrunk = px.soft_threshold(values, amount)
    return (left * shrunk) @ right, shrunk


rng = np.random.default_rng(0)
sample = rng.normal(size=(6, 4))
softened, spectrum = prox_nuclear(sample, 1.0)
raw = np.linalg.svd(sample, compute_uv=False)
print(f"  singular values before: {np.array2string(raw, precision=4)}")
print(f"  after thresholding at 1: "
      f"{np.array2string(np.linalg.svd(softened, compute_uv=False), precision=4)}")
print(f"  the rank drops from {int(np.sum(raw > 1e-12))} to "
      f"{int(np.sum(np.linalg.svd(softened, compute_uv=False) > 1e-12))}")


def matrix_completion(rows, cols, rank, fraction, lam, rounds=400, seed=42):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(rows, rank)) @ rng.normal(size=(rank, cols))
    mask = rng.random((rows, cols)) < fraction
    seen = truth * mask
    x = np.zeros((rows, cols))
    spectrum = np.zeros(min(rows, cols))
    for _ in range(rounds):
        x, spectrum = prox_nuclear(x - mask * (x - seen), lam)   # step 1, the mask has norm 1
    return {"relative_error": float(np.linalg.norm(x - truth) / np.linalg.norm(truth)),
            "rank_found": int(np.sum(spectrum > 1e-8)),
            "observed": int(mask.sum())}


print(f"\n{'size':>10}{'rank':>6}{'seen':>8}{'lambda':>8}{'rank found':>12}"
      f"{'relative error':>17}{'degrees of freedom':>21}{'seen / freedom':>16}")
for rows, cols, rank, share in ((60, 40, 3, 0.5), (60, 40, 3, 0.3), (60, 40, 3, 0.15),
                                (80, 80, 5, 0.4)):
    for lam in (0.5, 2.0):
        out = matrix_completion(rows, cols, rank, share, lam)
        freedom = rank * (rows + cols - rank)
        print(f"{f'{rows}x{cols}':>10}{rank:>6}{out['observed']:>8}{lam:>8.1f}"
              f"{out['rank_found']:>12}{out['relative_error']:>17.4f}{freedom:>21}"
              f"{out['observed'] / freedom:>16.2f}")
```

**The prox does exactly what the $\ell_1$ prox does, one level up.** The singular values shift down by
exactly the threshold, $3.036 \to 2.036$ and $2.034 \to 1.034$, and the one below the threshold
becomes exactly zero, dropping the rank from four to three. **The rank is the support**, and it is
reduced exactly, not approximately.

**Completion works above about three times the degrees of freedom.** A rank $r$ matrix has
$r(m+n-r)$ free parameters, and the table's last column is the ratio of observed entries to that. At
a ratio of $4.1$ the recovery is within $3.5$ per cent at the correct rank; at $2.5$ it degrades to
$20$ per cent and overestimates the rank; at $1.2$ it fails at $67$ per cent. The $80\times80$ case
at a ratio of $3.3$ recovers to $2.6$ per cent.

**The penalty trades rank for bias.** At $\lambda = 2$ the recovered rank is right more often and the
error is three to four times larger; at $\lambda = 0.5$ the error is small and the rank is
overestimated. That is exercise 5.2's bias appearing in a different costume, and the same repair
applies: find the rank with a large $\lambda$, then refit without a penalty on that rank.

### 3.5 Coordinate descent

For the LASSO the coordinatewise minimization has a closed form. Fixing all coordinates but $j$, the
objective in $x_j$ is a quadratic plus $\lambda\lvert x_j\rvert$, so 2.1 applies directly:

$$
x_j \leftarrow \frac{1}{\lVert a_j\rVert^{2}/N}\,
\mathcal{S}_{\lambda}\!\left(\frac{a_j^{T}(b - Ax + a_j x_j)}{N}\right) .
$$

```python
import numpy as np
from nalib import proximal as px


def coordinate_descent(problem, lam, rounds=600):
    A, b = problem["design"], problem["targets"]
    n, m = problem["samples"], problem["dimension"]
    norms = np.sum(A * A, axis=0) / n
    x = np.zeros(m)
    residual = -b.copy()                              # kept up to date, so a sweep is O(n m)
    values = []
    for _ in range(rounds):
        for j in range(m):
            if norms[j] <= 0.0:
                continue
            partial = float(A[:, j] @ residual) / n - norms[j] * x[j]
            fresh = float(px.soft_threshold(np.array([-partial]), lam)[0]) / norms[j]
            if fresh != x[j]:
                residual += A[:, j] * (fresh - x[j])
                x[j] = fresh
        values.append(px.objective(problem, x, lam))
    return {"x": x, "values": np.asarray(values),
            "nonzeros": int(np.count_nonzero(x))}


print(f"{'sparsity':>10}{'lambda / lambda_max':>21}{'CD sweeps to 1e-9':>20}"
      f"{'FISTA steps to 1e-9':>22}{'CD nonzeros':>13}{'FISTA nonzeros':>16}")
for sparsity in (2, 10, 40, 100):
    problem = px.lasso_problem(samples=200, variables=400, sparsity=sparsity)
    for share in (0.02, 0.2):
        lam = share * problem["lambda_max"]
        reference = px.proximal_gradient(problem, lam, accelerated=True,
                                         rounds=200000)["objective"]
        sweeps = coordinate_descent(problem, lam)
        fast = px.proximal_gradient(problem, lam, accelerated=True, rounds=20000)
        marks = []
        for values in (sweeps["values"], np.asarray(fast["values"])):
            hit = np.flatnonzero(values - reference <= 1e-9)
            marks.append(str(int(hit[0])) if hit.size else "none")
        print(f"{sparsity:>10}{share:>21.2f}{marks[0]:>20}{marks[1]:>22}"
              f"{sweeps['nonzeros']:>13}{fast['nonzeros']:>16}")
```

**Coordinate descent needs five to ten times fewer sweeps than FISTA needs steps**, at every sparsity
level and both penalties, and the two find identical supports.

**And a sweep costs about what a step costs.** One sweep touches $m$ coordinates and each update is
an $O(n)$ residual correction, so a sweep is $O(nm)$; a FISTA step is two matrix products, also
$O(nm)$. So the factor of five to ten is real work, not an artefact of counting.

**The gap widens as the problem gets harder.** At sparsity $100$ with the smaller penalty,
coordinate descent takes $311$ sweeps against FISTA's $1456$ steps.

Two further advantages explain why `glmnet` and the other standard LASSO packages use coordinate
descent rather than a proximal gradient method. The updates need only one column of $A$ at a time, so
the data can stay on disk. And once the support has settled, an **active set** strategy sweeps only
the non-zero coordinates, so a sweep costs $O(n\cdot\text{nnz})$ rather than $O(nm)$, which for a
sparse answer is a further large factor.

What coordinate descent needs in exchange is separability of the non-smooth term, which lesson 86's
exercise 3.2 identified as the condition for coordinate methods generally. It works for
$\lVert x\rVert_1$ and for the group penalty of 2.5 taken group at a time; it does not work for the
nuclear norm of 3.4.

### 4.1 The speedup against the conditioning

```python
import numpy as np
from nalib import proximal as px

print(f"{'samples':>9}{'variables':>11}{'kappa of the Gram':>20}{'ISTA to 1e-9':>15}"
      f"{'FISTA to 1e-9':>16}{'speedup':>10}")
for samples, variables in ((100, 400), (200, 400), (400, 400), (400, 200),
                           (400, 100), (800, 100)):
    problem = px.lasso_problem(samples=samples, variables=variables, sparsity=10)
    gram = problem["design"].T @ problem["design"] / samples
    values = np.linalg.eigvalsh(gram)
    kept = values[values > 1e-10 * values[-1]]
    kappa = float(kept[-1] / kept[0]) if kept.size else float("inf")
    lam = 0.1 * problem["lambda_max"]
    reference = px.proximal_gradient(problem, lam, accelerated=True,
                                     rounds=200000)["objective"]
    marks = []
    for accelerated in (False, True):
        out = px.proximal_gradient(problem, lam, accelerated=accelerated, rounds=30000)
        gaps = np.asarray(out["values"]) - reference
        hit = np.flatnonzero(gaps <= 1e-9)
        marks.append(int(hit[0]) if hit.size else -1)
    print(f"{samples:>9}{variables:>11}{kappa:>20.4g}{marks[0]:>15}{marks[1]:>16}"
          f"{(marks[0] / marks[1] if marks[1] > 0 else float('nan')):>10.2f}")
```

**The speedup does not track the condition number**, and on five of the six problems it is **below
one**: FISTA is slower than ISTA by seven to fourteen per cent. Only the most underdetermined case,
$100$ samples and $400$ variables, shows a gain, and it is $1.25$.

The $400\times400$ row is the sharpest disconfirmation. Its Gram matrix has a restricted condition
number of $6\times10^{9}$, by far the worst in the table, and its speedup is $0.87$.

**Why the theory does not apply.** FISTA's $O(1/k^{2})$ against ISTA's $O(1/k)$ is a worst case over
all convex composite problems with a given $L$. On these instances neither rate is what happens: once
the support has settled, the objective restricted to that support is a **strongly convex** quadratic,
and both methods converge linearly, at rates governed by the condition number of the Gram matrix
restricted to the support rather than the whole one. Neither $1/k$ nor $1/k^{2}$ describes a linear
rate, so the ratio between them predicts nothing.

The lesson's own `the_quoted_rates_are_worst_case` reaches the same conclusion by fitting the
exponents directly, and finds both steeper than quoted.

Exercise 4.2 finds the variable that does predict the speedup, and it is not the conditioning.

### 4.2 When the support settles

```python
import numpy as np
from nalib import proximal as px

problem = px.lasso_problem(samples=100, variables=400, sparsity=20)
print(f"{'lambda / lambda_max':>21}{'final nonzeros':>16}{'ISTA settles':>15}"
      f"{'FISTA settles':>15}{'ISTA to 1e-9':>15}{'FISTA to 1e-9':>16}{'speedup':>10}")
for share in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01):
    lam = share * problem["lambda_max"]
    reference = px.proximal_gradient(problem, lam, accelerated=True, rounds=200000)
    marks = []
    for accelerated in (False, True):
        out = px.proximal_gradient(problem, lam, accelerated=accelerated, rounds=30000)
        final = out["sparsity"][-1]
        same = np.flatnonzero(out["sparsity"] == final)
        gaps = np.asarray(out["values"]) - reference["objective"]
        hit = np.flatnonzero(gaps <= 1e-9)
        marks.append((int(same[0]) if same.size else -1,
                      int(hit[0]) if hit.size else -1))
    print(f"{share:>21.2f}{reference['nonzeros']:>16}{marks[0][0]:>15}{marks[1][0]:>15}"
          f"{marks[0][1]:>15}{marks[1][1]:>16}"
          f"{marks[0][1] / marks[1][1]:>10.2f}")
```

This is the table 4.1 was missing.

**The support takes much longer to settle as $\lambda$ falls.** ISTA settles it at step $11$ when
$\lambda = 0.5\lambda_{\max}$ and at step $998$ when $\lambda = 0.01\lambda_{\max}$, a factor of
$91$. FISTA settles it at $7$ and $95$, a factor of $14$.

**And the speedup grows in step with it.** It runs $1.05, 1.13, 1.14, 1.31, 1.71, 2.52, 2.96$ down
the table, monotonically, reaching a factor of three at the smallest penalty. That is the
relationship the exercise asks for, and it is much tighter than anything in 4.1.

**The mechanism.** The run has two phases. In the first, the method is deciding which coordinates are
non-zero, and the objective is genuinely non-smooth because the iterate is crossing kinks. In the
second, the support is fixed, the problem is a smooth strongly convex quadratic on that support, and
both methods converge linearly and quickly.

Acceleration helps in the **first** phase and hardly at all in the second. So when the support settles
in seven steps out of a hundred, there is nothing to accelerate and FISTA's overhead shows as a small
loss; when it takes a thousand out of thirteen hundred, acceleration is most of the run.

The practical reading: **if the penalty is large enough that the support settles quickly, use ISTA**,
which is simpler and monotone. FISTA earns its keep at small penalties and dense supports, which is
also where the problem is hardest.

### 4.3 The recovery threshold

Compressed sensing theory says exact recovery of a $k$-sparse signal in $m$ variables needs about
$k\log(m/k)$ samples. Measuring the constant needs the penalty scaled correctly with $N$, which is
the part that is easy to get wrong.

```python
import numpy as np
from nalib import proximal as px

noise = 0.01
repeats = 9
print(f"{'variables':>11}{'sparsity':>10}{'threshold N':>13}{'k log(m/k)':>13}"
      f"{'constant':>10}")
found = []
for variables in (200, 400, 800):
    for sparsity in (2, 4, 8, 16, 32):
        crossing = None
        for samples in list(range(10, 100, 5)) + list(range(100, 700, 20)):
            hits = 0
            for seed in range(repeats):
                problem = px.lasso_problem(samples=samples, variables=variables,
                                           sparsity=sparsity, noise=noise, seed=seed)
                # the penalty has to fall like sqrt(log m / N), not stay a fixed
                # multiple of lambda_max, or the threshold is not a threshold
                lam = 2.0 * noise * np.sqrt(2.0 * np.log(variables) / samples)
                out = px.proximal_gradient(problem, lam, accelerated=True,
                                           rounds=3000, record=False)
                picked = set(np.flatnonzero(np.abs(out["x"]) > 1e-2).tolist())
                hits += int(set(problem["support"].tolist()) <= picked)
            if hits >= 8:
                crossing = samples
                break
        scale = sparsity * np.log(variables / sparsity)
        if crossing is None:
            print(f"{variables:>11}{sparsity:>10}{'over 680':>13}{scale:>13.2f}")
            continue
        found.append((crossing, scale))
        print(f"{variables:>11}{sparsity:>10}{crossing:>13}{scale:>13.2f}"
              f"{crossing / scale:>10.3f}")

table = np.array(found)
fit = float(np.sum(table[:, 0] * table[:, 1]) / np.sum(table[:, 1] ** 2))
miss = float(np.max(np.abs((table[:, 0] - fit * table[:, 1]) / table[:, 0])))
print(f"\n  best fit N = {fit:.3f} k log(m/k), worst relative miss {miss:.1%}")
```

**The law fits with a constant near two.** Across three problem widths and five sparsity levels the
measured ratio $N/(k\log(m/k))$ ranges from $1.42$ to $3.41$, and the least squares constant is
$2.13$ with a worst single point $51$ per cent off.

That is about as good as this kind of fit gets. The theory's statement is asymptotic and up to
constants, the threshold here is defined by a hard criterion (eight of nine trials recovering the
whole support), and the measurement grid is coarse in $N$.

**The scaling of the penalty is what makes the fit possible.** Repeating the same sweep with a fixed
multiple of $\lambda_{\max}$ instead gives constants ranging from $1.9$ to $9.3$, a factor of five,
and the larger sparsity levels never recover at all within the sample range. The reason is that
$\lambda_{\max}$ itself changes with $N$, so a fixed multiple of it is not a fixed penalty, and the
theoretically motivated $\lambda \sim \sigma\sqrt{\log m / N}$ is the choice that makes the threshold
a property of the problem rather than of the parameterization.

The largest constant, $3.41$, occurs at $m = 200$ with $k = 32$, where $m/k = 6.25$ and the
logarithm is barely above two. That is the edge of the sparse regime, where the asymptotic form is
not expected to be sharp.

### 5.1 Why acceleration loses monotonicity

**The extrapolated point can be worse.** FISTA's step begins at

$$
y_k = x_k + \frac{t_k - 1}{t_{k+1}}\,(x_k - x_{k-1}) ,
$$

with the coefficient tending to $1$ as $k$ grows. That is a point **beyond** $x_k$ along the previous
direction, and nothing prevents $F(y_k) > F(x_k)$: the objective is convex, but the direction
$x_k - x_{k-1}$ was chosen to reduce $F$ near $x_{k-1}$, not to keep reducing it past $x_k$.

The proximal step from $y_k$ then guarantees $F(x_{k+1}) \le F(y_k)$ but says nothing about
$F(x_{k+1})$ against $F(x_k)$, and the gap can be positive.

**Why the sequence still converges.** The proof does not track $F(x_k)$. It tracks the Lyapunov
function

$$
E_k \;=\; \frac{2t_k^{2}}{L}\big(F(x_k) - F^{*}\big) \;+\; \lVert u_k - x^{*}\rVert^{2} ,
\qquad u_k = x_{k-1} + t_k(x_k - x_{k-1}) ,
$$

and shows $E_{k+1} \le E_k$. The objective term can rise as long as the distance term falls by at
least as much, which is exactly what the extrapolation is doing: it trades a temporarily worse
objective for a better position. Since $t_k \ge (k+1)/2$, the bound $E_k \le E_0$ gives

$$
F(x_k) - F^{*} \le \frac{L\,E_0}{2t_k^{2}} \le \frac{2L\lVert x_0 - x^{*}\rVert^{2}}{(k+1)^{2}} ,
$$

which is the $O(1/k^{2})$ rate. **The non-monotone objective is not an artefact the proof tolerates;
it is where the acceleration comes from.**

```python
import numpy as np
from nalib import proximal as px

out = px.fista_is_faster_and_not_monotone()
print(f"ISTA is monotone: {out['ista_is_monotone']}, "
      f"FISTA is not: {out['fista_is_not']}")
print(f"FISTA reaches the floor in {out['speedup_in_steps']:.2f} times fewer steps")
print(f"and is {out['accuracy_gain_at_1000']:.1e} times more accurate at a fixed budget")
print(f"they agree on the support: {out['they_agree_on_the_support']}")

problem = px.lasso_problem(samples=100, variables=400, sparsity=20)
lam = 0.1 * problem["lambda_max"]
fast = px.proximal_gradient(problem, lam, accelerated=True, rounds=600)
values = np.asarray(fast["values"])
rises = np.flatnonzero(np.diff(values) > 0.0)
print(f"\nover {values.size} FISTA steps the objective rises on {rises.size} of them")
if rises.size:
    biggest = int(rises[int(np.argmax(np.diff(values)[rises]))])
    print(f"  the largest rise is at step {biggest}: "
          f"{values[biggest]:.12f} to {values[biggest + 1]:.12f}")
    print(f"  a rise of {values[biggest + 1] - values[biggest]:.3e}")
```

The objective rises on a substantial fraction of FISTA's steps, and the largest rise is far above
rounding, so it is the mechanism and not noise. ISTA never rises on the same problem.

### 5.2 The bias of the penalty

**The claim.** The LASSO does not merely select the true support; it also shrinks the coefficients on
that support toward zero, by an amount proportional to $\lambda$.

**The size of the shrinkage.** Suppose the recovered support $S$ equals the true one and the signs
are right. Restricted to $S$ the objective is smooth, and stationarity reads

$$
\frac{A_S^{T}(A_S\hat x_S - b)}{N} + \lambda\,\operatorname{sign}(\hat x_S) = 0 .
$$

Writing $x_S^{\text{ls}}$ for the unpenalized least squares solution on $S$, which satisfies the same
equation with $\lambda = 0$, subtracting gives

$$
\boxed{\;\hat x_S = x_S^{\text{ls}} - \lambda\left(\frac{A_S^{T}A_S}{N}\right)^{-1}
\operatorname{sign}(\hat x_S)\;}
$$

**Every coefficient is shifted toward zero by a fixed amount**, not scaled, and the amount is
$\lambda$ times a row sum of the inverse restricted Gram matrix. For orthonormal columns that is
exactly $\lambda$.

```python
import numpy as np
from nalib import proximal as px

problem = px.lasso_problem(samples=400, variables=100, sparsity=5, noise=0.05)
A, N = problem["design"], problem["samples"]
support = problem["support"]
gram = A[:, support].T @ A[:, support] / N
print(f"{'lambda / lambda_max':>21}{'nonzeros':>10}{'mean |x| on the truth':>24}"
      f"{'mean |truth|':>14}{'shrinkage':>12}{'predicted':>12}")
for share in (0.5, 0.2, 0.1, 0.05, 0.02, 0.005, 0.0):
    lam = share * problem["lambda_max"]
    out = px.proximal_gradient(problem, lam, accelerated=True, rounds=40000,
                               record=False)
    got = float(np.mean(np.abs(out["x"][support])))
    want = float(np.mean(np.abs(problem["truth"][support])))
    shift = np.linalg.solve(gram, lam * np.sign(problem["truth"][support]))
    print(f"{share:>21.3f}{out['nonzeros']:>10}{got:>24.6f}{want:>14.6f}"
          f"{want - got:>12.6f}{float(np.mean(np.abs(shift))):>12.6f}")
```

**The prediction is exact wherever the support is right.** At $\lambda/\lambda_{\max} = 0.05, 0.02,
0.005$ the recovered support is the true one and the measured shrinkage is $0.254512, 0.101694,
0.025285$ against the predicted $0.254697, 0.101879, 0.025470$, agreeing to three or four digits.

At larger penalties the support is **not** the true one, only two to four of the five coefficients
are recovered, and the formula does not apply because its hypothesis fails.

**The dilemma this creates.** A large $\lambda$ is needed to force the wrong coefficients to zero, and
a large $\lambda$ biases the right ones. The two demands pull in opposite directions and no single
$\lambda$ satisfies both, which is the reason both standard corrections exist.

```python
import numpy as np
from nalib import proximal as px

lam = 0.1 * problem["lambda_max"]
first = px.proximal_gradient(problem, lam, accelerated=True, rounds=40000,
                             record=False)
picked = np.flatnonzero(np.abs(first["x"]) > 1e-8)

refit = np.zeros_like(first["x"])                 # correction one: refit without a penalty
refit[picked] = np.linalg.lstsq(A[:, picked], problem["targets"], rcond=None)[0]

weights = 1.0 / np.maximum(np.abs(first["x"]), 1e-6)   # correction two: reweight the penalty
adaptive = np.zeros_like(first["x"])
step = 1.0 / problem["smoothness"]
for _ in range(40000):
    moved = adaptive - step * problem["smooth_gradient"](adaptive)
    amount = step * lam * weights
    adaptive = np.sign(moved) * np.maximum(np.abs(moved) - amount, 0.0)

print(f"  the true support has {problem['sparsity']} entries")
for name, guess in (("plain LASSO", first["x"]), ("refit on the support", refit),
                    ("adaptive LASSO", adaptive)):
    print(f"    {name:>22}: error "
          f"{float(np.linalg.norm(guess - problem['truth'])):.6f}, "
          f"{int(np.count_nonzero(np.abs(guess) > 1e-8)):>3} nonzeros, "
          f"mean |x| on the truth {float(np.mean(np.abs(guess[support]))):.6f} "
          f"against {float(np.mean(np.abs(problem['truth'][support]))):.6f}")
```

**Refitting on the selected support wins decisively here**, cutting the error from $1.083$ to
$0.307$, a factor of $3.5$, and restoring the mean coefficient from $1.474$ to $1.890$ against the
true $1.947$. It is the simpler correction and it is often called the relaxed LASSO or the debiased
LASSO. Its weakness is that it uses the data twice, so the resulting confidence intervals are not
honest without further work.

**The adaptive LASSO is the more principled correction and does worse on this instance.** It reweights
the penalty by $1/\lvert\hat x_j\rvert$ from a first pass, so large coefficients are barely penalized
and small ones are penalized heavily, and it has an oracle property asymptotically. Here it cuts the
error only from $1.083$ to $0.914$ and loses a true coefficient, ending with three non-zeros against
the true five. The heavy penalty on the coefficient the first pass under-estimated pushed it out
entirely.

That is the failure mode of the adaptive method in one line: **it trusts the first pass, and where the
first pass was wrong it makes the error permanent.**

### 5.3 Beyond convexity

Take $g(x) = \lVert x\rVert_0$, the number of non-zeros. It is not convex and not even continuous,
but its prox has a closed form. The one dimensional problem

$$
\min_x\; \lambda\,\mathbb{1}[x \ne 0] + \frac{1}{2t}(x - v)^{2}
$$

has value $\frac{v^{2}}{2t}$ at $x = 0$ and value $\lambda$ at $x = v$, and those are the only two
candidates, so the answer is $v$ when $v^{2}/2t > \lambda$ and $0$ otherwise. That is **hard
thresholding**:

$$
\operatorname{prox}_{t\lambda\lVert\cdot\rVert_0}(v)
= \begin{cases} v, & \lvert v\rvert > \sqrt{2t\lambda},\\ 0, & \text{otherwise.}\end{cases}
$$

**What survives.**

The prox has a closed form and it is cheap, so proximal gradient can be run unchanged, and it is:
iterative hard thresholding is a standard method for sparse recovery.

The exact zeros survive, and are in fact sharper: hard thresholding keeps the surviving coefficients
at their full value, so there is **no shrinkage bias**. Exercise 5.2's whole problem disappears.

There is still a threshold above which the answer is zero, now $\lVert\nabla f(0)\rVert_\infty >
\sqrt{2\lambda/t}$, though it depends on the step in a way $\lambda_{\max}$ did not.

**What does not.**

```python
import numpy as np
from nalib import proximal as px


def hard_threshold(v, amount):
    v = np.asarray(v, dtype=float)
    return np.where(np.abs(v) >= np.sqrt(2.0 * amount), v, 0.0)


rng = np.random.default_rng(42)
u, w = rng.normal(size=8), rng.normal(size=8)
print(f"  soft is nonexpansive here: "
      f"{bool(np.linalg.norm(px.soft_threshold(u, 1.0) - px.soft_threshold(w, 1.0)) <= np.linalg.norm(u - w))}")
worst = 0.0
for _ in range(20000):
    a, c = rng.normal(size=4), rng.normal(size=4)
    apart = float(np.linalg.norm(a - c))
    if apart > 1e-9:
        worst = max(worst, float(np.linalg.norm(hard_threshold(a, 0.5)
                                                - hard_threshold(c, 0.5))) / apart)
print(f"  hard is nonexpansive: {worst <= 1.0 + 1e-12}, worst ratio {worst:.4f}")
print(f"  and it is discontinuous: hard(1.00000) = "
      f"{hard_threshold(np.array([1.0]), 0.5)[0]:.5f}, hard(0.99999) = "
      f"{hard_threshold(np.array([0.99999]), 0.5)[0]:.5f}")
```

**Firm nonexpansiveness fails, and badly.** The measured worst ratio is $5.25$, so hard thresholding
can push two nearby points **five times further apart**. Exercise 2.4's proof does not apply because
it uses monotonicity of $\partial g$, which needs convexity.

**The operator is discontinuous.** Inputs of $1.00000$ and $0.99999$ map to $1.0$ and $0.0$. So it is
not even single valued in the limit, and at the threshold exactly the minimization has two answers.

**Every convergence result in this lesson depended on one of those two.** The $O(1/k)$ rate for
proximal gradient uses nonexpansiveness; the fixed point characterization of optimality uses single
valuedness; FISTA's Lyapunov argument uses convexity of $F$ throughout.

**What is proved instead.** Iterative hard thresholding converges to a **local** minimum, and to the
global one only under a restricted isometry condition on $A$ that says the columns behave like an
orthonormal set on sparse subsets. That is a strong assumption about the data, not about the
algorithm, and it is exactly the assumption compressed sensing theory makes to get its guarantees.
Where it holds, hard thresholding is better than soft, because it has no bias. Where it fails,
soft thresholding still solves a convex problem correctly and hard thresholding can converge to any
of many local minima.

That is the same trade this part has met repeatedly: **a convex relaxation gives a guaranteed answer
to a slightly wrong problem, and the exact problem gives no guarantee at all.** Lesson 88's
projections and lesson 87's Lennard-Jones clusters are the same choice in different settings, and
the reason the convex route usually wins is that a guaranteed answer to a nearby problem is easier to
check than an unguaranteed answer to the right one.

---
