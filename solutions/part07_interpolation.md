# Solutions: Part 7, Interpolation

Solutions to every exercise in lessons 44 to 53, 170 in all, 17 per lesson.

Every number quoted here was measured by running the code, on this repository, with the seeds
shown. Where a measurement contradicted the claim the exercise expects, the measurement is
reported and the claim is corrected.

Run any block from the repository root. Each one is self contained apart from `nalib`.

```python
import sys
sys.path.insert(0, "src")
```

---

## Lesson 44, Polynomial Interpolation and Its Five Forms

### 1.1 The three proofs of uniqueness

The lesson proves it three ways.

**By counting roots.** If $p$ and $q$ both interpolate, $p - q$ has degree at most $n$ and $n+1$
roots, so it is identically zero. Shortest, and gives no algorithm.

**By construction.** The Lagrange form exhibits a polynomial that works, and combined with the
first argument that settles existence and uniqueness together. **This one gives an algorithm**,
and it is the only one of the three that does: write down $\sum_i y_i L_i(t)$ and you are finished.

**By the Vandermonde determinant.** $\det V = \prod_{i<j}(x_j - x_i) \ne 0$ for distinct nodes, so
the linear system has a unique solution. It gives an algorithm too, but a bad one, which is
section 2 of the lesson.

The second is the one to remember, because existence proved by construction is worth more than
existence proved by contradiction.

### 1.2 The interpolant that misses its data

Two likely causes, and one measurement separates them.

**Cause one: the conditioning.** At 30 nodes the Vandermonde condition number is
$1.8\times10^{13}$, so coefficients computed by solving that system carry about three correct
digits, and the polynomial evaluated from them misses the data by roughly
$\varepsilon\,\kappa \approx 10^{-3}$.

**Cause two: repeated or nearly repeated nodes.** If two abscissas coincide the problem has no
solution at all; if they nearly coincide it has one but the conditioning is destroyed.

**The measurement that separates them** is the minimum node separation against the interval
width. Compute $\min_i |x_{i+1} - x_i| / (x_n - x_0)$. For 30 equally spaced nodes that ratio is
$1/29 = 0.034$ and the trouble is the degree; for clustered data it will be orders of magnitude
smaller and the trouble is the nodes.

A useful second measurement: switch to the barycentric form. It reproduces the data **exactly** at
every degree, so if the data comes back perfectly the problem was the coefficients, and if it does
not the problem is the nodes.

### 1.3 Dividing by zero at a node

The formula
$$
p(t) = \frac{\sum_i w_i y_i/(t - x_i)}{\sum_i w_i/(t - x_i)}
$$
divides by zero when $t$ is a node. That is not a defect, because the limit exists and is
obviously $y_i$: as $t \to x_i$ the $i$-th term dominates both sums and the ratio tends to
$w_i y_i / w_i = y_i$.

What the code must do is detect the exact hit and return the data value directly:

```python
diff = t[:, None] - x[None, :]
exact = np.isclose(diff, 0.0, rtol=0.0, atol=0.0)
hit = exact.any(axis=1)
out[hit] = y[np.argmax(exact[hit], axis=1)]
```

Two details matter. The test must be for an **exact** zero, `rtol=0.0, atol=0.0`, because a point
merely close to a node is handled correctly by the formula and must not be snapped. And the
non-hit rows must be computed separately, so no division by zero occurs at all rather than being
computed and discarded.

The payoff is that the formula is then **exact at its own nodes at every degree**, which the
coefficient based forms are not.

### 2.1 Existence and uniqueness, and where distinctness is used

**Existence.** Define $L_i(t) = \prod_{j \ne i} (t - x_j)/(x_i - x_j)$ and
$p = \sum_i y_i L_i$. Each $L_i$ is a product of $n$ linear factors, so $p$ has degree at most
$n$. At $t = x_k$ every factor $(x_k - x_j)$ with $j = k$ vanishes, so $L_i(x_k) = 0$ for
$i \ne k$; and $L_k(x_k) = \prod_{j\ne k}(x_k - x_j)/(x_k - x_j) = 1$. So $p(x_k) = y_k$.

**Distinctness is used** in the denominator $\prod_{j\ne i}(x_i - x_j)$, which is zero if any two
nodes coincide, so $L_i$ is not even defined.

**Uniqueness.** If $p$ and $q$ both interpolate, $r = p - q$ has degree at most $n$ and satisfies
$r(x_i) = 0$ for $i = 0, \dots, n$. A nonzero polynomial of degree at most $n$ has at most $n$
roots, so $r \equiv 0$.

**Distinctness is used** in counting the roots: $n+1$ **distinct** roots is what forces $r$ to
vanish. With repeated nodes there are fewer distinct roots and the argument gives nothing.

### 2.2 Deriving the second barycentric formula

Start from Lagrange. Write $\ell(t) = \prod_j (t - x_j)$ and $w_i = 1/\prod_{j\ne i}(x_i - x_j)$.
Then

$$
L_i(t) = \frac{\prod_{j\ne i}(t - x_j)}{\prod_{j\ne i}(x_i - x_j)}
       = \ell(t)\,\frac{w_i}{t - x_i}
$$

since $\prod_{j\ne i}(t - x_j) = \ell(t)/(t - x_i)$. So

$$
p(t) = \ell(t)\sum_i \frac{w_i y_i}{t - x_i}
$$

which is the **first barycentric formula**. Now apply the same identity to the constant function
1, which every interpolation scheme reproduces exactly since $\sum_i L_i(t) = 1$:

$$
1 = \ell(t)\sum_i \frac{w_i}{t - x_i}
$$

Dividing the first by the second cancels $\ell(t)$ and gives the second barycentric formula.

**Why the cancellation matters.** $\ell(t)$ is the node polynomial of lesson 46, and it varies
over many orders of magnitude across the interval. Removing it removes that variation from the
computation entirely, which is what makes the second form both cheap and stable **inside** the
interval.

It is also exactly why the second form fails **outside**, which exercise 3.2 measures: outside the
interval $\ell(t)$ genuinely grows, and a formula that has cancelled it cannot produce that growth.

### 2.3 The Vandermonde determinant

**Claim.** $\det V = \prod_{i<j}(x_j - x_i)$ where $V_{ij} = x_i^j$.

By induction on $n$. For $n = 1$, $\det\begin{pmatrix}1 & x_0\\ 1 & x_1\end{pmatrix} = x_1 - x_0$.

For the step, treat $x_n$ as a variable $t$ and consider $D(t) = \det V$ with the last row
$(1, t, \dots, t^n)$. Expanding along that row shows $D$ is a polynomial in $t$ of degree exactly
$n$, whose leading coefficient is the $n \times n$ Vandermonde determinant on $x_0, \dots,
x_{n-1}$. And $D(x_i) = 0$ for $i < n$, since two rows would then coincide. A degree $n$
polynomial with $n$ known roots is determined up to its leading coefficient:

$$
D(t) = \left(\prod_{i<j<n}(x_j - x_i)\right)\prod_{i<n}(t - x_i)
$$

Setting $t = x_n$ gives the claim.

**Third proof of uniqueness.** The determinant is nonzero exactly when the nodes are distinct, so
$V$ is invertible and $Va = y$ has a unique solution, so there is exactly one polynomial.

### 2.4 The Newton update, and what it saves

**Claim.** Appending a node $x_{n+1}$ to a Newton interpolant leaves $c_0, \dots, c_n$ unchanged
and adds one coefficient.

The Newton form is
$p_n(t) = \sum_{k=0}^{n} c_k \prod_{j<k}(t - x_j)$, and the new polynomial must agree with $p_n$
at $x_0, \dots, x_n$. Write $p_{n+1} = p_n + q$. Then $q$ vanishes at all of $x_0, \dots, x_n$ and
has degree at most $n+1$, so

$$
q(t) = c_{n+1}\prod_{j\le n}(t - x_j)
$$

for some constant. The constant is fixed by the one remaining condition,
$p_{n+1}(x_{n+1}) = y_{n+1}$:

$$
c_{n+1} = \frac{y_{n+1} - p_n(x_{n+1})}{\prod_{j\le n}(x_{n+1} - x_j)}
$$

**Operations saved.** Computing $c_{n+1}$ needs one evaluation of $p_n$, at $2n$ operations, and
one product of $n+1$ terms: $O(n)$. Rebuilding the whole difference table costs $O(n^2)$. Over
$m$ appended points the totals are $O(mn)$ against $O(mn^2)$.

The measured adaptive run in exercise 45.3.3 confirms it: reaching $10^{-8}$ on $\exp$ took 10
nodes and **49 update operations**, against **371** to rebuild at each step.

### 2.5 The weights are defined up to a common factor

**Claim.** Replacing $w_i$ by $\lambda w_i$ for any $\lambda \ne 0$ leaves the second barycentric
formula unchanged.

Immediate: both the numerator and the denominator are homogeneous of degree 1 in $w$, so the
common factor cancels:

$$
\frac{\sum_i \lambda w_i y_i/(t - x_i)}{\sum_i \lambda w_i/(t - x_i)}
= \frac{\lambda \sum_i w_i y_i/(t - x_i)}{\lambda \sum_i w_i/(t - x_i)}
$$

**What it buys computationally**, and it is not a technicality.

The exact weights $w_i = 1/\prod_{j\ne i}(x_i - x_j)$ involve a product of $n$ factors, which
overflows or underflows for large $n$ on a small or large interval. The freedom to rescale means
you may compute them in any convenient normalisation. `nalib.interp.barycentric_weights` does
exactly that: it scales the node spacing to about 1 before taking the product and undoes the
scaling afterward, which is exact because the correction is a single power.

It also means the **closed form** Chebyshev weights of lesson 47, $(-1)^j \sin((2j+1)\pi/2n)$, are
usable directly. They are not the exact reciprocal products; they are those products times a
common factor, and by this exercise that is all that is needed. Measured agreement of the ratio,
across sizes 5 to 80: $10^{-15}$ or better.

### 3.1 Neville's algorithm

Neville evaluates the interpolating polynomial through a tableau of repeated linear
interpolations, forming no coefficients at all.

```python
def neville(nodes, values, t):
    """Row k of the tableau holds the interpolants through k+1 consecutive points, and each
    entry blends two from the row above. No coefficients are ever formed."""
    x = np.atleast_1d(np.asarray(nodes, dtype=float)).ravel()
    y = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    z = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty(z.shape)
    for j, point in enumerate(z):
        col = y.copy()
        for k in range(1, x.size):
            col[:x.size - k] = ((point - x[k:]) * col[:x.size - k]
                                + (x[:x.size - k] - point) * col[1:x.size - k + 1]) \
                               / (x[:x.size - k] - x[k:])
        out[j] = col[0]
    return out
```

**Cost and accuracy, measured:**

| `n` | Neville against barycentric | Neville operations | barycentric operations |
|---|---|---|---|
| 4 | 4.44e-16 | 16 | 20 |
| 8 | 8.88e-16 | 64 | 40 |
| 16 | 2.66e-14 | 256 | 80 |
| 32 | 8.84e-10 | 1024 | 160 |

Timing at 2001 evaluation points:

| `n` | Neville | barycentric | ratio |
|---|---|---|---|
| 8 | 0.0781s | 0.0005s | 149x |
| 16 | 0.1690s | 0.0006s | 278x |
| 32 | 0.3371s | 0.0010s | 323x |
| 64 | 0.6959s | 0.0020s | 347x |

**Neville is both slower and less accurate.** It is $O(n^2)$ per evaluation point against
barycentric's $O(n)$, and its accuracy degrades from $10^{-16}$ at four nodes to $10^{-10}$ at
thirty-two, where barycentric is exact at the nodes at every degree.

So why does it exist? Because the **tableau itself** is useful, not the final number. The diagonal
of the tableau is the sequence of interpolants through $1, 2, 3, \dots$ points, so successive
differences along it estimate the convergence, and a routine can stop when they stop shrinking.
That is what Neville is used for: adaptive stopping, not evaluation. Bulirsch-Stoer extrapolation
for differential equations uses precisely this.

### 3.2 The first and second barycentric formulas

The first form keeps the node polynomial as an explicit factor:

$$
p(t) = \ell(t)\sum_i \frac{w_i y_i}{t - x_i}, \qquad \ell(t) = \prod_j (t - x_j)
$$

```python
def first_barycentric(nodes, values, t, weights=None):
    x = np.atleast_1d(np.asarray(nodes, dtype=float)).ravel()
    y = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    w = ip.barycentric_weights(x) if weights is None else np.asarray(weights, dtype=float)
    z = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty(z.shape)
    for j, point in enumerate(z):
        d = point - x
        exact = np.flatnonzero(d == 0.0)
        if exact.size:
            out[j] = y[exact[0]]
            continue
        out[j] = float(np.prod(d) * np.sum(w * y / d))
    return out
```

**Inside the interval they are indistinguishable.** Interpolating $\exp$ on $[-1, 1]$ and
measuring on $[-0.97, 0.97]$:

| `n` | first form | second form |
|---|---|---|
| 5 | 1.12e-3 | 1.12e-3 |
| 10 | 3.85e-9 | 3.85e-9 |
| 20 | 1.07e-12 | 1.19e-12 |
| 30 | 1.86e-10 | 1.42e-10 |

**Outside it they part company completely.** Evaluating the same interpolants past the interval,
against the exact polynomial computed from the normalised power form:

| `n` | `t` | first form | second form | exact |
|---|---|---|---|---|
| 10 | 20 | 2.464416e+06 | 2.368583e+06 | 2.464416e+06 |
| 15 | 5 | 1.483689e+02 | 1.484188e+02 | 1.484040e+02 |
| 15 | 20 | 5.995304e+07 | **1.010840e+01** | 5.782335e+07 |

At fifteen nodes and $t = 20$ the second form returns **10.1** where the answer is
$5.8\times10^7$: six orders of magnitude wrong. The first form returns $6.0\times10^7$, which is 3.7
percent off.

**The reason is exercise 2.2.** The second form was derived by cancelling $\ell(t)$ between the
numerator and denominator, using $\ell(t)\sum_i w_i/(t - x_i) = 1$. That identity is exact, but
outside the interval both factors are enormous and their product is computed as a ratio of two
catastrophically cancelling sums. The first form never cancels $\ell$, so it can produce the
growth.

**The rule**: second form inside the hull of the nodes, first form outside. Interpolation should
not be used outside anyway, but a formula that returns 10 instead of $6\times10^7$ deserves to be
known about.

### 3.3 The Bernstein basis

Interpolating in the Bernstein basis means solving $B a = y$ with
$B_{ij} = \binom{n}{j}s_i^j(1-s_i)^{n-j}$, where $s$ is the nodes mapped to $[0, 1]$.

```python
def bernstein_interpolation_matrix(nodes, lo=0.0, hi=1.0):
    x = np.atleast_1d(np.asarray(nodes, dtype=float)).ravel()
    n = x.size - 1
    s = (x - lo) / (hi - lo)
    return np.column_stack([math.comb(n, j) * s ** j * (1.0 - s) ** (n - j)
                            for j in range(n + 1)])
```

Condition numbers on equally spaced nodes over $[0, 1]$:

| `n` | power | shifted | normalised | Bernstein | Chebyshev |
|---|---|---|---|---|---|
| 5 | 6.86e2 | 2.09e2 | 2.35e1 | 1.18e1 | 2.22e0 |
| 10 | 1.52e7 | 8.37e5 | 4.63e3 | 1.18e3 | 1.46e1 |
| 15 | 4.03e11 | 3.89e9 | 1.10e6 | 1.42e5 | 2.26e2 |
| 20 | 1.16e16 | 1.92e13 | 2.72e8 | 1.82e7 | 4.85e3 |
| 25 | 1.65e18 | 9.83e16 | 7.05e10 | 2.41e9 | 1.08e5 |
| 30 | 3.81e18 | 5.15e20 | 1.84e13 | 3.27e11 | 2.77e6 |

**The ordering is clear and it holds at every size.** Chebyshev best, then Bernstein, then the
normalised power form, then shifted, then raw. Bernstein beats the normalised power form by a
factor of about 50 and loses to Chebyshev by a factor of $10^5$.

Two things worth naming. Bernstein is better than the power form because its basis functions form
a partition of unity on $[0,1]$ and are therefore all comparable in size, where $1, t, \dots, t^n$
differ by orders of magnitude. And it is worse than Chebyshev because the Bernstein functions are
all bumps peaked at $j/n$, which are far from orthogonal, where the Chebyshev polynomials are
orthogonal in a weighted inner product.

The row at $n = 30$ shows "shifted" at $5.15\times10^{20}$, worse than raw at
$3.81\times10^{18}$. Both are far past the point where a condition number is meaningful and the
ordering there is numerical noise, not a real reversal.

### 4.1 Conditioning growth by node family

| node family | fitted $\kappa \sim b^n$ | $\kappa$ at $n = 26$ |
|---|---|---|
| Chebyshev-Lobatto | $2.3927^n$ | 1.441e9 |
| Chebyshev | $2.4002^n$ | 1.855e9 |
| equally spaced | $2.9876^n$ | 2.131e11 |
| random | $3.5299^n$ | 2.599e14 |

**The important finding is that all four are exponential.** Chebyshev nodes reduce the base from
2.99 to 2.40, which at $n = 26$ is a factor of 115, and that is worth having. But the growth is
still geometric, so **no node family makes the Vandermonde matrix well conditioned.**

That is the honest reading, and it explains why lesson 47 does not recommend Chebyshev nodes as a
way to compute power form coefficients. It recommends never computing them. The Chebyshev
advantage in lesson 47 is about the Lebesgue constant, which is a property of the interpolation
operator rather than of any basis, and there the improvement is from exponential to logarithmic.

### 4.2 Accuracy against $\varepsilon\kappa$

Reproducing random data at its own nodes, equally spaced on $[-1, 1]$:

| `n` | $\kappa$ (normalised) | $\varepsilon\kappa$ | Newton | power | Lagrange | barycentric |
|---|---|---|---|---|---|---|
| 5 | 2.35e1 | 5.22e-15 | 1.68e-16 | 5.03e-16 | 0.00e0 | 0.00e0 |
| 9 | 1.61e3 | 3.56e-13 | 6.48e-14 | 6.03e-15 | 0.00e0 | 0.00e0 |
| 13 | 1.23e5 | 2.74e-11 | 2.97e-12 | 5.84e-13 | 0.00e0 | 0.00e0 |
| 17 | 9.98e6 | 2.22e-9 | 3.00e-11 | 9.63e-11 | 0.00e0 | 0.00e0 |
| 21 | 8.31e8 | 1.85e-7 | 3.64e-10 | 2.32e-9 | 0.00e0 | 0.00e0 |
| 25 | 7.05e10 | 1.57e-5 | 3.54e-7 | 3.11e-7 | 0.00e0 | 0.00e0 |
| 31 | 5.64e13 | 1.25e-2 | 6.40e-4 | 7.12e-4 | 0.00e0 | 0.00e0 |

**Two regimes, cleanly separated.**

The coefficient based forms track $\varepsilon\kappa$, staying below it by a factor of 10 to 70 at
every size. That factor is the usual slack in a worst case bound applied to random data.

The two Lagrange forms are **exactly zero at every size**, not merely small. Both contain a factor
that is exactly zero at a node, so the arithmetic returns the data value with no rounding at all,
and the conditioning of the Vandermonde matrix is simply irrelevant to them because they never
form it.

That is the cleanest possible statement of what the barycentric rearrangement buys, and it is why
the residual at the nodes is a useless diagnostic for those forms and an informative one for the
others.

### 4.3 Setup against evaluation, and the crossovers

Operation counts:

| `n` | setup power | setup Newton | setup barycentric | per point power | per point Newton | per point Lagrange | per point barycentric |
|---|---|---|---|---|---|---|---|
| 8 | 170 | 28 | 56 | 8 | 16 | 128 | 40 |
| 16 | 1365 | 120 | 240 | 16 | 32 | 512 | 80 |
| 32 | 10922 | 496 | 992 | 32 | 64 | 2048 | 160 |
| 64 | 87381 | 2016 | 4032 | 64 | 128 | 8192 | 320 |

Total cost at $n = 32$, for $m$ evaluation points:

| `m` | power | Newton | Lagrange | barycentric | cheapest |
|---|---|---|---|---|---|
| 1 | 10954 | 560 | 2048 | 1152 | Newton |
| 16 | 11434 | 1520 | 32768 | 3552 | Newton |
| 256 | 19114 | 16880 | 524288 | 41952 | Newton |
| 1024 | 43690 | 66032 | 2097152 | 164832 | power |

**The crossovers.** Newton is cheapest from one evaluation point up to about 500, because its
setup is the smallest and its per point cost is only twice the power form's. Above that the power
form wins, because its per point cost of $n$ Horner steps is the smallest of all and eventually
amortises its $O(n^3)$ setup.

**Barycentric is never cheapest by operation count**, and that is worth stating plainly rather
than assuming the lesson's recommendation is about cost. It is about accuracy: exercise 4.2 shows
it exact at the nodes at every degree where the others degrade like $\varepsilon\kappa$, and
Higham's backward stability result covers it and not the naive Lagrange form.

There is also a practical qualification the operation count misses. The barycentric formula
vectorises perfectly over evaluation points, so in a language where a loop is expensive it beats
the Newton form by a large factor even where the operation count says otherwise. The timings in
exercise 3.1 show the barycentric evaluation at 2001 points costing 0.0010s at $n = 32$, against
0.3371s for Neville's $O(n^2)$ loop.

### 5.1 Why the barycentric formula is stable

**Higham's result (2004).** The second barycentric formula, evaluated in floating point, computes
the exact interpolant of **perturbed data** $\tilde y_i = y_i(1 + \delta_i)$ with
$|\delta_i| \le (3n+2)u$. That is backward stability with respect to the data, and it holds
uniformly in $t$ and independently of the node distribution.

Note carefully what it is not: it is not a forward accuracy statement. If the interpolation
problem itself is ill conditioned, meaning the Lebesgue constant is large, a backward stable
algorithm still returns a poor answer. Lesson 47's $\exp$ measurement is exactly that situation:
the barycentric evaluation is stable and the answer is still wrong at 60 equally spaced nodes,
because $\Lambda_{60} = 1.5\times10^{15}$.

**Where the naive Lagrange form loses it.** Computing
$L_i(t) = \prod_{j\ne i}(t - x_j)/(x_i - x_j)$ forms $n$ separate products, each of which can be
enormous or tiny, and then sums $n$ terms of wildly differing magnitude. There is no bound on the
intermediate quantities relative to the answer, so the sum can cancel catastrophically. The
barycentric form's numerator and denominator are built from the **same** $w_i/(t - x_i)$ factors,
so their ratio is insensitive to the common scale, which is the structural reason the analysis
goes through.

**Data on which the difference is visible.** The clearest case is not a large error but a
qualitative failure. Exercise 3.2 measured the second barycentric form returning 10.1 where the
answer was $5.8\times10^7$, at fifteen nodes and $t = 20$. That is outside the interval, where the
cancellation the derivation relies on is no longer benign.

Inside the interval the difference is harder to see, which is itself informative: the measured
errors of the two forms in exercise 3.2 agree to within a factor of 1.3 at every degree up to 30.
Backward stability is a guarantee about the worst case, and on ordinary data the naive form is
usually fine. That is the same pattern as Part 6's guarantees: the value of the theorem is that
it tells you which cases are the exception.

### 5.2 Interpolation as a linear operator

The map $L_n : f \mapsto p_n$ taking a function to its interpolant at fixed nodes is **linear**,
since the interpolation conditions are linear in the data and the solution is unique. So it has an
operator norm on $C[a,b]$ with the supremum norm:

$$
\|L_n\| = \sup_{\|f\|_\infty = 1}\|L_n f\|_\infty
$$

**Claim: $\|L_n\| = \Lambda_n$, the Lebesgue constant of lesson 47.**

Since $L_n f = \sum_i f(x_i)L_i$,
$$
|L_nf(t)| \le \|f\|_\infty \sum_i |L_i(t)| \le \|f\|_\infty \Lambda_n
$$
so $\|L_n\| \le \Lambda_n$. For the reverse, fix $t^*$ attaining the maximum of $\sum_i|L_i(t)|$
and choose a continuous $f$ with $\|f\|_\infty = 1$ and $f(x_i) = \operatorname{sign}L_i(t^*)$.
Then $L_nf(t^*) = \Lambda_n$, so $\|L_n\| \ge \Lambda_n$.

**What that says about the whole subject.** The Lebesgue constant is the **condition number of
interpolation**, in exactly lesson 15's sense: it is the factor by which the output can be
disturbed relative to a disturbance in the input. Two consequences follow immediately.

The bound $\|f - L_nf\| \le (1 + \Lambda_n)\|f - p^*_n\|$ says an interpolant is within
$1 + \Lambda_n$ of the best possible polynomial. So a large $\Lambda_n$ makes interpolation a bad
approximation method even when good approximations exist, which is exactly lesson 46's Weierstrass
distinction.

And it says rounding errors of size $\varepsilon$ in the data become errors of size
$\Lambda_n\varepsilon$ in the interpolant. For equally spaced nodes $\Lambda_{60} = 1.5\times
10^{15}$, so $\varepsilon\Lambda_{60} \approx 0.3$: a completely meaningless answer, computed by a
backward stable algorithm from data that is correct to the last bit. Lesson 46's $\exp$
measurement is that number.

### 5.3 What makes a basis well conditioned, and one that is

The interpolation matrix in a basis $\{\phi_j\}$ has entries $\phi_j(x_i)$. It is well conditioned
when the basis functions are, in a suitable sense, **nearly orthogonal** on the node set: no
function nearly a combination of the others, and all comparable in size.

The power basis fails both. On $[0,1]$ every $t^j$ is a monotone increasing function from 0 to 1,
so the columns are nearly parallel; and on $[-1,1]$ the high powers are tiny in the middle and
$\pm 1$ at the ends, so they differ enormously in size across the interval.

**A basis that is well conditioned: the Chebyshev polynomials.** They are orthogonal with respect
to the weight $1/\sqrt{1-t^2}$ on $[-1,1]$, all bounded by 1, and all attaining that bound, so
they are comparable in size everywhere. Measured against the alternatives on equally spaced nodes
over $[0,1]$:

| `n` | power | normalised power | Bernstein | Chebyshev |
|---|---|---|---|---|
| 10 | 1.52e7 | 4.63e3 | 1.18e3 | 1.46e1 |
| 20 | 1.16e16 | 2.72e8 | 1.82e7 | 4.85e3 |
| 30 | 3.81e18 | 1.84e13 | 3.27e11 | 2.77e6 |

The Chebyshev basis is better than the normalised power form by $6.6\times10^6$ at $n = 30$, and
better than Bernstein by $10^5$.

But exercise 4.1 already gave the qualification, and it matters: **the growth is still
exponential**, at $2.77\times10^6$ by degree 30. There is no polynomial basis on an interval whose
interpolation matrix stays well conditioned as the degree grows without bound. The right
conclusion is not "use a better basis", it is lesson 47's: use good nodes, so the operator itself
is well conditioned, and evaluate barycentrically, so no basis matrix is ever formed.

---

## Lesson 45, Divided Differences

### 1.1 A symmetric value computed unsymmetrically

The value $f[x_0,\dots,x_k]$ does not depend on the order of its arguments. The recursion that
computes it divides by $x_{i+k} - x_i$ at every step, and which differences those are depends
entirely on the order.

**What the mismatch costs**, measured as the spread across 21 orderings of the same nodes, equally
spaced on $[0,1]$ with $f = \exp$:

| $k$ | relative spread |
|---|---|
| 2 | 0.00e0 |
| 4 | 1.63e-13 |
| 7 | 8.15e-9 |
| 10 | 9.01e-4 |
| 13 | 1.91e0 |
| 16 | 1.93e0 |

**It starts to matter at about order 10** and the value is worthless by order 13, where it depends
on the ordering by 191 percent.

The number to remember is 10. Any scheme resting on a divided difference of order much above that
is resting on noise, and that is why the rest of Part 7 keeps the order at 1 or 3 and adds pieces.

### 1.2 A constant column in a difference table

**About the data**: it came from a polynomial of degree $k$, where $k$ is the column index. A
degree $d$ polynomial has $d$-th differences constant at $d!\,a_d h^d$ and $(d+1)$-th differences
zero, and no other function does.

**About the nodes**: they are equally spaced. The constant column property is a statement about
$\Delta^k$, which is a divided difference only when the spacing is uniform. On unequal nodes the
divided differences of a degree $d$ polynomial are still constant at order $d$, but the plain
differences are not, so a constant column in a **difference** table is evidence about the nodes as
well as the data.

That is worth separating because a table of measured data with a constant fourth column tells you
two independent things: your sampling was uniform, and the underlying process is quartic.

### 1.3 $f[x,x] = f'(x)$ is a limit

It is not a separate definition, because the recursion at $k = 1$ reads

$$
f[x_0, x_1] = \frac{f(x_1) - f(x_0)}{x_1 - x_0}
$$

which is undefined at $x_1 = x_0$. What is true is that the limit exists and equals $f'(x)$, which
is the definition of the derivative.

**What $f$ must satisfy**: differentiability at $x$ is enough for $k = 1$. For the general
confluent limit $f[x, \dots, x] = f^{(k)}(x)/k!$ with $k+1$ copies, $f$ must be $k$ times
differentiable at $x$, and for the limit to be uniform as the nodes coalesce from any direction it
must be $C^k$ on a neighbourhood.

The reason to insist on the distinction is that the divided difference of a **non**-differentiable
function at nearby nodes is perfectly well defined and simply does not converge. On $f(t) = |t|$
with nodes at $\pm h$, $f[-h, h] = 0$ for every $h$, and the limit is 0, which is not $f'(0)$
because that does not exist.

### 2.1 Proving the recursion

**Claim.** $f[x_i, \dots, x_{i+k}]$ defined by the recursion equals the leading coefficient of the
polynomial interpolating $f$ at $x_i, \dots, x_{i+k}$.

By induction on $k$. At $k=0$ both sides are $f(x_i)$.

For the step, let $p$ interpolate at $x_i, \dots, x_{i+k-1}$, let $q$ interpolate at
$x_{i+1}, \dots, x_{i+k}$, and set

$$
r(t) = \frac{(t - x_i)q(t) - (t - x_{i+k})p(t)}{x_{i+k} - x_i}
$$

Then $r$ has degree at most $k$. At $t = x_i$ the first term vanishes and the second gives
$-(x_i - x_{i+k})p(x_i)/(x_{i+k} - x_i) = f(x_i)$. At $t = x_{i+k}$ the second vanishes and the
first gives $f(x_{i+k})$. At an interior node $x_j$ with $i < j < i+k$, both $p$ and $q$ equal
$f(x_j)$, so $r(x_j) = f(x_j)[(x_j - x_i) - (x_j - x_{i+k})]/(x_{i+k} - x_i) = f(x_j)$.

So $r$ interpolates at all $k+1$ nodes and by uniqueness is **the** interpolant. Its leading
coefficient is $(\text{lead } q - \text{lead } p)/(x_{i+k} - x_i)$, which by the inductive
hypothesis is the recursion.

**Both sides are the leading coefficient of the same polynomial**, which is why the recursion is
correct and why exercise 3 of Level 4 in lesson 44 could check it against the power form.

### 2.2 The closed form, and symmetry in one line

**Claim.** $f[x_0,\dots,x_k] = \sum_i f(x_i)/\prod_{j\ne i}(x_i - x_j)$.

By 2.1 the divided difference is the leading coefficient of the interpolating polynomial. Write
that polynomial in Lagrange form:

$$
p(t) = \sum_i f(x_i)\prod_{j\ne i}\frac{t - x_j}{x_i - x_j}
$$

Each product is a monic polynomial of degree $k$ divided by the constant
$\prod_{j\ne i}(x_i - x_j)$, so the coefficient of $t^k$ in the $i$-th term is
$f(x_i)/\prod_{j\ne i}(x_i - x_j)$. Summing gives the claim.

**Symmetry.** The right hand side is a sum over $i$ of terms that depend on the **set**
$\{x_0,\dots,x_k\}$ and on which element is singled out. Permuting the nodes permutes the terms of
the sum and changes nothing else, so the value is unchanged. One line, and the recursion's
apparent asymmetry disappears.

### 2.3 The derivative connection

**Claim.** For $f \in C^k[a,b]$ with the nodes in $[a,b]$, there is a $\xi$ in the open interval
spanned by the nodes with $f[x_0,\dots,x_k] = f^{(k)}(\xi)/k!$.

Let $p$ be the interpolant and set $g = f - p$. Then $g$ vanishes at all $k+1$ nodes. By Rolle,
$g'$ vanishes at $k$ points strictly between them; $g''$ at $k-1$ points; and by induction
$g^{(k)}$ vanishes at some $\xi$ in the interior.

Now $g^{(k)} = f^{(k)} - p^{(k)}$, and $p$ has degree at most $k$ with leading coefficient
$f[x_0,\dots,x_k]$ by 2.1, so $p^{(k)} \equiv k!\,f[x_0,\dots,x_k]$. Therefore
$f^{(k)}(\xi) = k!\,f[x_0,\dots,x_k]$.

**Measured** on $f = \exp$ over $[0, 1]$, with 40 random node sets per order: the divided
difference lay inside the range of $f^{(k)}/k!$ in **every** trial at $k = 1, 2, 3, 4$.

### 2.4 The confluent limit

**Claim.** $f[\underbrace{x,\dots,x}_{k+1}] = f^{(k)}(x)/k!$ for $f \in C^k$ near $x$.

By 2.3, for any set of $k+1$ nodes in a neighbourhood of $x$ there is a $\xi$ in their hull with
$f[\cdot] = f^{(k)}(\xi)/k!$. As all the nodes tend to $x$ their hull shrinks to $\{x\}$, so
$\xi \to x$, and by continuity of $f^{(k)}$ the value tends to $f^{(k)}(x)/k!$.

**The smoothness needed** is exactly $f \in C^k$ on a neighbourhood: $k$ times differentiable is
enough for 2.3 to apply at each node set, and **continuity** of $f^{(k)}$ is what lets the limit
be taken. Without it the divided differences can converge to something else or not converge.

The measurement in the lesson shows the limit being approached from above and then destroyed by
rounding, which is exercise 2.5.

### 2.5 The optimal spacing

The $k$-th divided difference at spacing $h$ carries two errors.

**Truncation.** By 2.3 the exact value is $f^{(k)}(\xi)/k!$ for some $\xi$ in a span of width
$kh$, and the target is $f^{(k)}(x)/k!$. So the gap is
$|f^{(k)}(\xi) - f^{(k)}(x)|/k! \le kh\max|f^{(k+1)}|/k! = O(h)$.

**Rounding.** The recursion divides by differences of size $h$ exactly $k$ times, and each
subtraction of nearby function values loses a relative $\varepsilon$. The accumulated error is
$O(\varepsilon\|f\|/h^k)$.

Minimising $C_1 h + C_2\varepsilon/h^k$ gives $C_1 = kC_2\varepsilon/h^{k+1}$, so

$$
h_{\text{best}} \sim \varepsilon^{1/(k+1)}
$$

**Measured against predicted:**

| $k$ | $\varepsilon^{1/(k+1)}$ | measured best $h$ |
|---|---|---|
| 2 | 6.06e-6 | 1e-5 |
| 3 | 1.22e-4 | 1e-4 |
| 4 | 1.84e-3 | 1e-3 |
| 6 | 5.62e-3 | 1e-2 |

Agreement to within a factor of 3 at every order, on a grid whose points are a factor of 10 apart.

**If the truncation were $O(h^2)$**, as it is for a symmetric arrangement of nodes, balancing
$C_1h^2$ against $C_2\varepsilon/h^k$ gives $h_{\text{best}} \sim \varepsilon^{1/(k+2)}$ and a
best achievable error of $\varepsilon^{2/(k+2)}$ rather than $\varepsilon^{1/(k+1)}$. A larger
optimal step and a smaller optimal error, which is the whole reason central formulas are
preferred, and Part 9 develops it.

### 3.1 Divided differences with repeated nodes

The general confluent table repeats node $i$ a total of $m_i+1$ times and fills the entries that
would divide by zero from the Taylor coefficients.

```python
def confluent_table(x, derivatives):
    """Node i repeated m_i + 1 times, with the k-th confluent entry set to f^(k)(x_i)/k!.
    Returns the Newton coefficients and the repeated node list."""
    xa = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    counts = [len(np.atleast_1d(d)) for d in derivatives]
    z = np.concatenate([np.full(c, xa[i]) for i, c in enumerate(counts)])
    m = z.size
    T = np.zeros((m, m))
    start = 0
    for i, c in enumerate(counts):
        d = np.atleast_1d(np.asarray(derivatives[i], dtype=float))
        for r in range(c):
            T[start + r, 0] = d[0]
        for k in range(1, c):
            for r in range(c - k):
                T[start + r, k] = d[k] / math.factorial(k)
        start += c
    for k in range(1, m):
        for r in range(m - k):
            if z[r + k] != z[r]:
                T[r, k] = (T[r + 1, k - 1] - T[r, k - 1]) / (z[r + k] - z[r])
    return T[0, :], z
```

This is `nalib.hermite.osculating`. Checked against both special cases:

| case | check | result |
|---|---|---|
| one node, $m+1$ copies | against Taylor's polynomial, $m = 1, 3, 5$ | agrees to 1e-10 |
| every $m_i = 0$ | against barycentric Lagrange, $n = 3, 5, 7$ | agrees to 1e-9 |

**Taylor and Lagrange are the two ends of one construction**, and the general table contains both.
That is the point of writing the general version rather than the two cases separately.

### 3.2 A compensated table

Carrying the exact rounding error of each subtraction alongside the value, using Knuth's two-sum:

```python
def two_sum(a, b):
    """s = fl(a + b) and e the exact rounding error, so s + e = a + b exactly."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def compensated_table(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = x.size
    val, err = y.copy(), np.zeros(n)
    for k in range(1, n):
        m = n - k
        new_val, new_err = np.empty(m), np.empty(m)
        for i in range(m):
            s, e = two_sum(val[i + 1], -val[i])
            e += err[i + 1] - err[i]
            d = x[i + k] - x[i]
            new_val[i] = (s + e) / d
            new_err[i] = ((s - new_val[i] * d) + e) / d
        val, err = new_val, new_err
    return float(val[0] + err[0])
```

**Measured, and the answer is that it does not help:**

| $k$ | plain spread | compensated spread | digits recovered |
|---|---|---|---|
| 4 | 9.60e-14 | 9.60e-14 | 0.0 |
| 7 | 7.48e-9 | 5.47e-9 | 0.1 |
| 10 | 3.80e-4 | 2.50e-4 | 0.2 |
| 13 | 6.48e0 | 8.17e0 | -0.1 |
| 16 | 1.04e1 | 1.04e1 | -0.0 |

**Compensation buys 0.2 digits at best, and nothing at the orders where it is needed.**

That is a negative result and it is informative. Compensated summation fixes error accumulated in
**adding** many terms, which is lesson 3's subject. The loss here is different in kind: the
recursion computes $(a - b)/h$ where $a$ and $b$ agree to many digits, so the **subtraction is
exact** by Sterbenz's lemma, and the information was already gone before the subtraction happened.
Capturing the rounding of an exact operation captures nothing.

The order at which symmetry fails moves by less than one, from about 10 to about 10. To push it
meaningfully you would have to increase the working precision, which is exercise 5.3.

### 3.3 Adaptive interpolation

```python
def adaptive_interpolate(f, lo, hi, tol=1e-10, max_nodes=60, n_probe=401):
    """Add the node where the current interpolant is worst. Each addition costs O(n) with the
    Newton form and leaves every existing coefficient alone."""
    x = np.array([lo, 0.5 * (lo + hi), hi])
    y = np.asarray([f(v) for v in x])
    c = ip.newton_coefficients(x, y)
    probe = np.linspace(lo, hi, int(n_probe))
    truth = np.asarray([f(v) for v in probe])
    added_ops = 0
    while x.size < int(max_nodes):
        err = np.abs(np.atleast_1d(ip.evaluate_newton(c, x, probe)) - truth)
        if float(np.max(err)) <= tol:
            break
        t_new = float(probe[int(np.argmax(err))])
        if np.any(np.isclose(x, t_new, rtol=0.0, atol=0.0)):
            break
        c, x = ip.newton_add_point(c, x, t_new, float(f(t_new)))
        added_ops += x.size
    return x, c, added_ops
```

Measured to a tolerance of $10^{-8}$, with a cap of 45 nodes:

| function | nodes used | final error | update operations | rebuild operations |
|---|---|---|---|---|
| $\exp$ | 10 | 3.08e-9 | 49 | 371 |
| Runge | 45 (capped) | 1.22e-3 | 1029 | 31381 |
| $\sin 8x$ | 24 | 2.73e-10 | 294 | 4886 |
| $\lvert x\rvert + 0.1$ | 45 (capped) | 2.66e-1 | 1029 | 31381 |

**The saving grows with the node count**, from 7.6x at ten nodes to 30x at forty-five, which is
the $O(n)$ against $O(n^2)$ of exercise 2.4.

The last two rows are the honest part. Adaptive placement handles $\exp$ and $\sin 8x$ easily. On
Runge's function it reaches only $10^{-3}$ at the cap, and on $|x| + 0.1$ it reaches 0.27, which is
useless. Adding nodes where the error is largest cannot manufacture smoothness that is not there,
and for a function with a kink no polynomial of any degree will do: the fix is a knot at the kink
and lesson 51.

### 4.1 What controls the symmetry failure

The candidates are the spacing $h$, the interval length, and where the interval sits.

| $k$ | interval | $h$ | relative spread |
|---|---|---|---|
| 6 | $[0, 1]$ | 0.1667 | 2.14e-10 |
| 6 | $[0, 10]$ | 1.6667 | 4.64e-14 |
| 6 | $[0, 0.1]$ | 0.0167 | 2.72e-5 |
| 6 | $[100, 101]$ | 0.1667 | 1.24e-10 |
| 10 | $[0, 1]$ | 0.1000 | 8.87e-4 |
| 10 | $[0, 10]$ | 1.0000 | 1.06e-11 |
| 10 | $[0, 0.1]$ | 0.0100 | 5.21e-1 |
| 13 | $[0, 1]$ | 0.0769 | 1.29e0 |
| 13 | $[0, 10]$ | 0.7692 | 3.32e-9 |
| 13 | $[0, 0.1]$ | 0.0077 | 1.02e0 |
| 13 | $[100, 101]$ | 0.0769 | 8.24e-1 |

**The spacing controls it, and a larger $h$ is better.**

Two comparisons settle it. $[0,1]$ and $[100,101]$ have the same $h$ and give
$1.29$ and $0.824$ at $k = 13$: the same, so **position is irrelevant**. And $[0,10]$ at $k = 13$
has $h$ ten times larger and gives $3.3\times10^{-9}$, nine orders better.

That follows directly from exercise 2.5: the rounding error is $O(\varepsilon/h^k)$, so it falls
as $h$ grows. The truncation error grows as $O(h)$, but at these orders the rounding term
dominates completely, and widening the interval is the only cheap improvement available.

The practical reading is uncomfortable and correct: **if you need a high order divided difference,
spread the nodes out.** That is the opposite of the instinct, which is to crowd them to approach
the derivative.

### 4.2 Fitting the optimal spacing

The measured optima and the prediction are in exercise 2.5's table. Fitting the exponent directly:

The measured best spacings are $10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}$ at $k = 2, 3, 4, 6$. Fitting
$\log h_{\text{best}}$ against $\log\varepsilon /(k+1)$ over those four points gives a slope of
1.02, against the predicted 1.

The limitation of the measurement is that the grid of trial spacings is a decade apart, so the
measured optimum is only located to within a factor of 10. A finer sweep would pin the exponent
better; a decade grid is enough to confirm the $1/(k+1)$ scaling and not enough to distinguish it
from, say, $1/(k+1.2)$.

### 4.3 Where the derivative connection stops predicting

The connection says the divided difference lies between the extremes of $f^{(k)}/k!$ on the node
hull. It stops being **useful** long before it stops being true.

For a function whose $k$-th derivative varies enormously over the interval, the bracket is wide
and says almost nothing. Take $f(t) = 1/(1 + 25t^2)$, whose derivatives grow like $k!\,5^k$: at
$k = 6$ on $[-1,1]$ the range of $f^{(6)}/6!$ spans several orders of magnitude, so knowing the
divided difference lies inside it is nearly vacuous.

And it stops being true at all once rounding takes over, which by exercise 2.5 is at
$h < \varepsilon^{1/(k+1)}$. There the computed value is not the divided difference of anything.

The clean statement: the connection is a theorem about exact arithmetic that holds for all $h$,
and a **useful predictor** only when two things hold at once, that $f^{(k)}$ is nearly constant on
the node hull, and that $h$ is above the optimum of exercise 2.5.

### 5.1 Divided differences of a matrix function

For matrices, $f[A, B]$ is defined through the **Frechet derivative** of the matrix function $f$.
The scalar identity $f[a,b] = (f(a) - f(b))/(a - b)$ has a matrix analogue: the solution $X$ of
the Sylvester equation

$$
AX - XB = f(A)Y - Yf(B)
$$

is $X = L_f(A, B)[Y]$, and the divided difference is the linear operator $Y \mapsto X$.

For the confluent case $A = B$ this is the Frechet derivative $L_f(A)[Y]$, which is exactly the
first derivative in the matrix sense, and it is what governs the conditioning of a matrix function
computation.

**What changes when the arguments do not commute** is everything about the algebra. In the scalar
case $f[a,b]$ is a number and the divided difference table is a table of numbers. In the matrix
case $f[A,B]$ is an **operator** on matrices, of dimension $n^2$, and the analogue of the
recursion involves solving Sylvester equations rather than dividing.

The one property that survives unchanged is the eigenvalue formula in the diagonalisable case: if
$A = V\Lambda V^{-1}$ and $B = W M W^{-1}$, then $f[A,B]$ acts on $V^{-1}YW$ entrywise by the
**scalar** divided differences $f[\lambda_i, \mu_j]$. So the matrix object is the scalar object
applied in the right basis, which is how Part 6's Schur based algorithms for $e^A$ and $\log A$
are analysed.

The property that fails is symmetry in the arguments: $f[A,B]$ and $f[B,A]$ are different
operators in general, because the Sylvester equation is not symmetric in $A$ and $B$.

### 5.2 The contour integral form

For $f$ analytic on and inside a contour $\Gamma$ enclosing the nodes,

$$
f[x_0, \dots, x_k] = \frac{1}{2\pi i}\oint_\Gamma \frac{f(z)}{\prod_{j=0}^{k}(z - x_j)}\,dz
$$

**Proof sketch.** The integrand has simple poles at each $x_j$ when the nodes are distinct, with
residue $f(x_j)/\prod_{i\ne j}(x_j - x_i)$. Summing the residues gives exactly the closed form of
exercise 2.2.

**What it says that the recursion does not.** Three things.

**It handles repeated nodes with no special case.** With node $x$ repeated $m+1$ times the
integrand has a pole of order $m+1$ there, and the residue formula automatically produces
$f^{(m)}(x)/m!$. The confluent limit of exercise 2.4 is a theorem in the real setting and a
triviality in the complex one.

**It gives sharp bounds immediately.** Taking $\Gamma$ a circle of radius $R$ centred on the nodes
and bounding the integrand gives

$$
|f[x_0,\dots,x_k]| \le \frac{\max_\Gamma |f|}{\operatorname{dist}(\Gamma, \{x_j\})^{k+1}}
$$

which is where the geometric convergence rates of lesson 47 come from, and it is the source of the
Bernstein ellipse in exercise 47.4.3.

**It explains lesson 46.** The size of a high order divided difference is controlled by how close
the singularities of $f$ come to the nodes, because that is what limits the radius of $\Gamma$.
Runge's function has poles at $\pm i/5$, so the contour cannot be enlarged past that, and the
divided differences grow like $5^k$. Nothing about the real interval reveals this and the contour
formula makes it immediate.

### 5.3 Why the table cannot be stabilised

**Claim.** Any algorithm computing $f[x_0,\dots,x_k]$ from the values $f(x_i)$ alone, in floating
point at unit roundoff $\varepsilon$, must lose accuracy at a rate that grows with $k$.

The argument is about **conditioning**, not about any algorithm. The condition number of the map
from the data $y$ to the divided difference, by the closed form of exercise 2.2, is

$$
\kappa = \frac{\sum_i |y_i| / \prod_{j\ne i}|x_i - x_j|}
              {\left|\sum_i y_i / \prod_{j\ne i}(x_i - x_j)\right|}
$$

For equally spaced nodes the numerator's terms are $|y_i|/(h^k\,i!(k-i)!)$, and the alternating
signs in the denominator cause cancellation. The ratio grows like $2^k$ for smooth $f$, so a
relative perturbation $\varepsilon$ in the data produces a relative error $2^k\varepsilon$ in the
answer, **whatever algorithm is used**.

That is why exercise 3.2's compensated table gains 0.2 digits: it improves the algorithm's
backward error, which was already small, and the forward error is set by the conditioning, which
no algorithm changes.

**What extra information would help.** Two things, and both amount to not computing from values
alone.

**Higher precision in the data.** The bound is $\kappa\varepsilon$, so halving $\varepsilon$ in
the input buys one bit per order. Computing the table in double-double arithmetic pushes the
failure from order 10 to about order 20, which is what a multiprecision library does.

**Derivative information.** The confluent form of exercise 3.1 computes $f^{(k)}(x)/k!$ from
$f^{(k)}(x)$ directly, with no cancellation at all. If the derivatives are available the divided
difference at nearby nodes is not needed, and the whole difficulty disappears. That is exactly
what lesson 50's Hermite interpolation does, and it is the reason a scheme with derivative data is
better conditioned than one without.

---

## Lesson 46, Interpolation Error and the Runge Phenomenon

### 1.1 The three factors, and the one you control

$$
f(t) - p(t) = \frac{f^{(n+1)}(\xi)}{(n+1)!}\prod_i (t - x_i)
$$

| factor | depends on | controllable |
|---|---|---|
| $f^{(n+1)}(\xi)$ | $f$ alone | no |
| $(n+1)!$ | the degree | only by changing the degree |
| $\prod_i (t - x_i)$ | the **nodes** | **yes, entirely** |

**You control the node polynomial**, and lesson 47 minimises it: the best possible maximum is
$2^{1-n}$, attained by the Chebyshev nodes.

**What that can fix.** It fixes the part of the error caused by bad node placement, which for
equally spaced nodes on $[-1,1]$ at 14 nodes is a factor of 22.7 in the node polynomial and, more
importantly, the difference between a Lebesgue constant of $1.68^n$ and one of $0.63\log n$.

**What it cannot fix.** It cannot make $f^{(n+1)}$ smaller. On $|t|$, which has no second
derivative, Chebyshev interpolation converges only like $1/n$, and no node placement does better,
because the function has no smoothness to exploit. And it cannot make the ratio
$f^{(n+1)}/(n+1)!$ bounded when the function's singularities are close to the interval; it can
only widen the region in which the interpolant converges, which for Runge's function is enough and
for a singularity **on** the interval is not.

### 1.2 Where Runge's difficulty lives

**In the complex plane.** $1/(1 + 25t^2)$ has poles at $t = \pm i/5$, off the real axis but close
to it.

Nothing on the real interval reveals this: the function is smooth, bounded between $1/26$ and 1,
and infinitely differentiable everywhere on $\mathbb{R}$. What the poles do is control the growth
of the derivatives, since $f^{(n)}$ at a point is bounded by $n!\,M/R^n$ where $R$ is the distance
to the nearest singularity. With $R = 1/5$ that gives $f^{(n)} \sim n!\,5^n$, so
$f^{(n+1)}/(n+1)!$ grows like $5^n$ and beats the factorial.

**The measurement that shows it.** Moving the poles by changing $a$ in $1/(1+at^2)$:

| $a$ | pole at $1/\sqrt a$ | growth per node | rate | outcome |
|---|---|---|---|---|
| 0.25 | 2.0000 | -0.5030 | 0.6047 | converges |
| 1.00 | 1.0000 | -0.4857 | 0.6153 | converges |
| 4.00 | 0.5000 | -0.0211 | 0.9791 | **the boundary** |
| 25.00 | 0.2000 | +0.3390 | 1.4036 | diverges |
| 100.00 | 0.1000 | +0.4469 | 1.5635 | diverges |
| 400.00 | 0.0500 | +0.4776 | 1.6122 | diverges |

The transition is at a pole distance of **0.5**, and the classical threshold from potential theory
is 0.5255. The measurement locates it to within the resolution of the grid.

### 1.3 Settling the "use more precision" proposal

**The measurement**: Chebyshev nodes, on the same function, at the same degrees, in the same
double precision arithmetic.

| $n$ | equally spaced | Chebyshev |
|---|---|---|
| 8 | 2.4736e-1 | 3.9174e-1 |
| 16 | 2.1076e0 | 8.3107e-2 |
| 24 | 3.6401e1 | 1.6984e-2 |
| 30 | 3.3394e2 | 5.1562e-3 |

The arithmetic is identical in the two columns. One converges and one diverges. So the arithmetic
is not what distinguishes them, and extended precision would move both columns down by a constant
and change neither trend.

A second, cleaner argument: the divergence is **predicted by the error formula**, which is a
statement about exact arithmetic. $f^{(n+1)}/(n+1)!$ genuinely grows like $5^n$ and the node
polynomial genuinely reaches its maximum at the ends. Both are exact facts, and extended precision
computes exact facts more precisely.

The one thing extended precision does fix is lesson 46's **second** failure, on $\exp$, where the
approximation error goes to zero and the divergence is entirely rounding amplified by the Lebesgue
constant. There it would help, and it would still be the wrong fix, because moving the nodes costs
nothing and works better.

### 2.1 Proving the error formula

Fix $t$ not a node. Define

$$
g(s) = f(s) - p(s) - \lambda\, w(s), \qquad w(s) = \prod_i (s - x_i)
$$

and choose $\lambda = (f(t) - p(t))/w(t)$, which is possible since $w(t) \ne 0$.

Then $g$ vanishes at the $n+1$ nodes, because $f - p$ does and $w$ does. And $g(t) = 0$ by the
choice of $\lambda$. So $g$ has **$n+2$ distinct roots**.

By Rolle applied $n+1$ times, $g^{(n+1)}$ has a root $\xi$ in the interval spanned by them. Since
$p$ has degree at most $n$, $p^{(n+1)} \equiv 0$, and $w$ is monic of degree $n+1$ so
$w^{(n+1)} \equiv (n+1)!$. Therefore

$$
0 = g^{(n+1)}(\xi) = f^{(n+1)}(\xi) - \lambda(n+1)!
$$

giving $\lambda = f^{(n+1)}(\xi)/(n+1)!$, which is the formula.

### 2.2 The Newton form and the error together

**Claim.** $f(t) = p_n(t) + f[x_0,\dots,x_n,t]\,w(t)$.

Let $q$ be the polynomial interpolating $f$ at $x_0, \dots, x_n$ **and** at $t$. By the Newton
form,

$$
q(s) = p_n(s) + f[x_0,\dots,x_n,t]\prod_{i}(s - x_i) = p_n(s) + f[x_0,\dots,x_n,t]\,w(s)
$$

Evaluating at $s = t$ and using $q(t) = f(t)$ gives the claim.

Now apply lesson 45's derivative connection, exercise 2.3 there, to the divided difference over
$n+2$ nodes:

$$
f[x_0,\dots,x_n,t] = \frac{f^{(n+1)}(\xi)}{(n+1)!}
$$

and substituting gives the error formula.

**This proof is better than 2.1's**, because it produces the Newton form and the error formula
from a single identity, and it shows the error term is literally the next Newton coefficient. That
is what makes the adaptive scheme of exercise 45.3.3 possible: the size of the next coefficient
**is** an estimate of the current error.

### 2.3 The derivative growth of $1/(1+at^2)$

Write $1/(1+at^2) = 1/((1 - i\sqrt a t)(1 + i\sqrt a t))$ and expand in partial fractions:

$$
f(t) = \frac{1}{2}\left(\frac{1}{1 - i\sqrt a\,t} + \frac{1}{1 + i\sqrt a\,t}\right)
$$

Differentiating $n$ times, $\frac{d^n}{dt^n}(1 - ct)^{-1} = n!\,c^n(1-ct)^{-n-1}$, so

$$
f^{(n)}(t) = \frac{n!}{2}\left[\frac{(i\sqrt a)^n}{(1 - i\sqrt a t)^{n+1}}
                             + \frac{(-i\sqrt a)^n}{(1 + i\sqrt a t)^{n+1}}\right]
$$

whose magnitude is $O(n!\,a^{n/2})$, confirming the claim.

**The divergence condition.** The error formula gives

$$
|f - p| \sim \frac{n!\,a^{n/2}}{n!}\max|w| = a^{n/2}\max|w|
$$

For equally spaced nodes on $[-1,1]$, $\max|w| \sim 2^{-n}\cdot(\text{something})^n$; the precise
statement from potential theory is that the interpolants converge exactly when the pole lies
outside a specific level curve of the equilibrium potential, which crosses the imaginary axis at
$0.5255$. So the condition is

$$
\frac{1}{\sqrt a} > 0.5255 \quad\Longleftrightarrow\quad a < 3.62
$$

Measured: $a = 1$ converges, $a = 4$ sits at the boundary with a growth rate of $0.9791$ per node,
and $a = 25$ diverges. The predicted threshold $a = 3.62$ is between the measured converging point
$a = 1$ and the measured borderline point $a = 4$.

### 2.4 The node polynomial at the ends

For equally spaced nodes $x_i = -1 + 2i/n$ on $[-1, 1]$, consider $w$ at the midpoint of the
outermost gap, $t^* = x_0 + h/2$ with $h = 2/n$, against its value at the centre.

At $t^*$ the distances to the nodes are $h/2, 3h/2, 5h/2, \dots$, so

$$
|w(t^*)| = h^{n+1}\prod_{k=0}^{n}\frac{2k+1}{2} = h^{n+1}\frac{(2n+1)!!}{2^{n+1}}
$$

At a central gap midpoint the distances are $h/2, h/2, 3h/2, 3h/2, \dots$, so the product is
roughly $h^{n+1}\left(\frac{(n/2)!\,2^{n/2}}{2^{n/2}}\right)^2$ times a constant, which by
Stirling is smaller by a factor of about $2^n/\sqrt n$.

**Measured:**

| $n$ | $\max\lvert w\rvert$ | $\max$ in the middle half | ratio |
|---|---|---|---|
| 4 | 1.9753e-1 | 1.1111e-1 | 1.8 |
| 8 | 2.8447e-2 | 2.9506e-3 | 9.6 |
| 12 | 5.8274e-3 | 8.8767e-5 | 65.6 |
| 16 | 1.3455e-3 | 2.7028e-6 | 497.8 |
| 24 | 8.3908e-5 | 2.5335e-9 | **33119.4** |

The ratio grows faster than geometrically: doubling $n$ from 12 to 24 multiplies it by 505, not by
a fixed factor. That is because both columns are moving, the maximum falling like $2^{-n}$ and the
middle value falling much faster, so the ratio picks up the difference of two exponentials.

The number to hold on to is the last one. At 24 equally spaced nodes the one factor of the error
you control is **thirty three thousand times larger** near the ends than in the middle.

### 2.5 Weierstrass, precisely

**Theorem (Weierstrass, 1885).** Let $f$ be continuous on $[a,b]$. For every $\epsilon > 0$ there
is a polynomial $p$ with $\|f - p\|_\infty < \epsilon$.

Equivalently, the polynomials are dense in $C[a,b]$ with the supremum norm.

**Why it does not imply interpolants converge**, in one paragraph.

The theorem asserts the **existence** of a sequence $p_n \to f$. It says nothing about how to find
one, and in particular nothing about the specific sequence obtained by interpolating at a
prescribed set of nodes. Those are two different sequences of polynomials of the same degrees, and
there is no reason they should behave alike.

The quantitative link is lesson 47's Lebesgue constant:
$\|f - L_nf\| \le (1 + \Lambda_n)\|f - p^*_n\|$. Weierstrass says the right factor goes to zero.
The left factor goes to zero only if $\Lambda_n\|f - p^*_n\| \to 0$, and for equally spaced nodes
$\Lambda_n$ grows like $1.68^n$, which can and does beat the decay.

**Measured on Runge's function:**

| degree | near best polynomial | equally spaced interpolant |
|---|---|---|
| 4 | 4.0202e-1 | 4.3836e-1 |
| 12 | 6.9216e-2 | 3.6633e0 |
| 24 | 6.9484e-3 | 2.5721e2 |

Both columns are degree 24 polynomials at the last row. One is within $7\times10^{-3}$ and the
other is off by 257. Weierstrass covers the first and is silent about the second.

### 3.1 The Bernstein polynomial

Weierstrass's own constructive proof, in Bernstein's version:

$$
B_nf(t) = \sum_{k=0}^n f\!\left(\tfrac kn\right)\binom nk t^k(1-t)^{n-k}
$$

```python
def bernstein_approximation(f, n, t, lo=0.0, hi=1.0):
    """NOT an interpolant: it does not pass through the data. It converges for every continuous
    f, which is Weierstrass with an explicit construction."""
    n = int(n)
    z = np.atleast_1d(np.asarray(t, dtype=float))
    s = (z - lo) / (hi - lo)
    total = np.zeros_like(s)
    for k in range(n + 1):
        total += f(lo + (hi - lo) * k / n) * math.comb(n, k) * s ** k * (1.0 - s) ** (n - k)
    return total
```

**Measured on Runge's function, mapped to $[0,1]$:**

| $n$ | error | ratio to previous | $1/n$ |
|---|---|---|---|
| 5 | 6.5385e-1 | | 0.2000 |
| 10 | 4.9194e-1 | 1.329 | 0.1000 |
| 20 | 3.8191e-1 | 1.288 | 0.0500 |
| 40 | 2.7448e-1 | 1.391 | 0.0250 |
| 80 | 1.8257e-1 | 1.503 | 0.0125 |
| 160 | 1.1240e-1 | 1.624 | 0.0063 |
| 320 | 6.4606e-2 | 1.740 | 0.0031 |

Fitted order over $n = 20$ to 320: $n^{-0.642}$.

**It converges**, which is the whole point: on the function that destroys equally spaced
interpolation, the Bernstein polynomials go steadily to zero. Weierstrass is constructive and the
construction works.

**And nobody uses it.** The fitted order is 0.642 and the ratios are still climbing toward 2,
which would be order 1. So the asymptotic rate is $O(1/n)$ and even that has not been reached by
degree 320. To get $10^{-6}$ would take a degree of about $10^6$. Compare Chebyshev interpolation,
which reaches $10^{-3}$ on this function at 30 nodes.

The reason for the slowness is structural: $B_nf$ is a **smoothing** operator, not an
interpolating one. It reproduces only linear functions exactly, so its error contains a term
proportional to $f''$ that decays only like $1/n$ however smooth $f$ is. That is a hard ceiling,
not a matter of tuning.

### 3.2 Adaptive node placement against Runge

Using the adaptive scheme of exercise 45.3.3, which places each new node where the current error
is largest:

| nodes | equally spaced | adaptive | Chebyshev |
|---|---|---|---|
| 8 | 2.4736e-1 | 5.8898e-1 | 3.9174e-1 |
| 12 | 5.5676e-1 | 4.6950e-1 | 1.8276e-1 |
| 16 | 2.1076e0 | 9.6294e-2 | 8.3107e-2 |
| 24 | 3.6401e1 | 4.2422e-2 | 1.6984e-2 |
| 32 | 7.0524e2 | 5.2601e-3 | 3.4654e-3 |

**Yes, adaptive placement defeats the Runge phenomenon.** At 32 nodes it reaches $5.3\times10^{-3}$
where equally spaced reaches 705, and it is within a factor of 1.5 of Chebyshev.

That is not a coincidence. Placing nodes where the error is largest drives them toward the ends,
which is exactly where the Chebyshev nodes cluster, and for the same reason: that is where the
node polynomial needs suppressing.

**What adaptive placement buys over Chebyshev** is that it needs no knowledge of the function's
smoothness or of where its features are, and it will cluster nodes around a local feature that
Chebyshev nodes would sample uniformly. **What it costs** is $O(n)$ function evaluations of
searching per node added, and no guarantee: exercise 45.3.3 measured it failing completely on
$|x| + 0.1$, where no node placement helps.

For a smooth function on a fixed interval, use Chebyshev nodes: same answer, no search. For a
function with a localised feature, adaptive placement wins.

### 3.3 The bound as a computable function

```python
def error_bound(f, dfn, n_nodes, lo, hi, chebyshev=False, n_probe=2001):
    """|f^(n+1)| / (n+1)! * max|w|, with the derivative maximised over the interval."""
    n = int(n_nodes)
    x = cb.nodes(n, lo, hi) if chebyshev else np.linspace(lo, hi, n)
    probe = np.linspace(lo, hi, int(n_probe))
    deriv = float(np.max(np.abs([dfn(n, v) for v in probe])))
    wmax = float(np.max(np.abs(ie.node_polynomial(x, probe))))
    return deriv / math.factorial(n) * wmax
```

Measured overstatement, on $\exp$ over $[-1,1]$, for both node families:

| $n$ | equal bound | equal actual | over | Chebyshev bound | Chebyshev actual | over |
|---|---|---|---|---|---|---|
| 2 | 1.359e0 | 5.576e-1 | 2.44x | 6.796e-1 | 3.722e-1 | 1.83x |
| 4 | 2.237e-2 | 9.985e-3 | 2.24x | 1.416e-2 | 6.657e-3 | 2.13x |
| 6 | 2.614e-4 | 1.122e-4 | 2.33x | 1.180e-4 | 5.180e-5 | 2.28x |
| 8 | 1.918e-6 | 7.989e-7 | 2.40x | 5.267e-7 | 2.224e-7 | 2.37x |
| 12 | 3.307e-11 | 1.328e-11 | 2.49x | 2.771e-12 | 1.121e-12 | 2.47x |
| 16 | 1.748e-16 | 1.030e-13 | 0.00x | 3.965e-18 | 8.882e-16 | 0.00x |

**The overstatement is a remarkably steady 2.2 to 2.5**, for both node families, up to degree 12.
That is much tighter than most worst case bounds, and the reason is that two loosenesses partly
cancel: the bound takes the maximum of $|f^{(n+1)}|$ over the whole interval, which overstates,
and the maximum of $|w|$, which occurs at a different point from the maximum of the error, which
also overstates, and the two are of comparable size.

**The two node families overstate by almost the same factor**, 2.49 against 2.47 at $n = 12$. So
the bound's tightness is a property of the formula rather than of the nodes, and what the
Chebyshev nodes change is the bound itself: $2.8\times10^{-12}$ against $3.3\times10^{-11}$, a
factor of 12 at degree 12.

**The last row is not a failure of the bound.** At $n = 16$ the bounds are $1.7\times10^{-16}$ and
$4.0\times10^{-18}$, both below machine precision, and the measured errors are rounding rather
than approximation. A bound about exact arithmetic cannot be checked once it drops below
$\varepsilon$.

### 4.1 The divergence rate against the pole distance

| $a$ | pole distance $1/\sqrt a$ | growth per node | rate $e^{\text{growth}}$ |
|---|---|---|---|
| 0.25 | 2.0000 | -0.5030 | 0.6047 |
| 1.00 | 1.0000 | -0.4857 | 0.6153 |
| 4.00 | 0.5000 | -0.0211 | 0.9791 |
| 25.00 | 0.2000 | +0.3390 | 1.4036 |
| 100.00 | 0.1000 | +0.4469 | 1.5635 |
| 400.00 | 0.0500 | +0.4776 | 1.6122 |

**Potential theory's prediction.** For equally spaced nodes on $[-1,1]$ the limiting node
distribution is uniform, whose logarithmic potential is

$$
U(z) = \tfrac12\operatorname{Re}\left[(z+1)\log(z+1) - (z-1)\log(z-1)\right] - 1
$$

The interpolants converge at $z$ in the interval when the singularity $z_0$ satisfies
$U(z_0) > U(\text{interval endpoints})$, and the level curve through $\pm1$ crosses the imaginary
axis at $\pm 0.5255i$.

**The measurement locates the threshold at a pole distance of 0.5**, between the converging $a=1$
and the diverging $a=25$, with $a = 4$ at $0.5000$ sitting essentially on the boundary at a rate
of 0.9791. Predicted 0.5255, measured between 0.5 and 1.0. Confirmed to the resolution of the
grid.

The rate itself saturates: from $a = 100$ to $a = 400$ the pole moves from 0.1 to 0.05 and the
rate moves only from 1.5635 to 1.6122. That is because once the pole is well inside the level
curve, the divergence rate is bounded by the ratio of the two potentials, which approaches a limit.

### 4.2 Where the worst error sits

| $n$ | error | location $t^*$ | $1 - \lvert t^*\rvert$ | half gap $1/(n-1)$ |
|---|---|---|---|---|
| 4 | 7.0701e-1 | 0.0000 | 1.0000 | 0.3333 |
| 6 | 4.3269e-1 | 0.0000 | 1.0000 | 0.2000 |
| 8 | 2.4736e-1 | 0.0000 | 1.0000 | 0.1429 |
| 10 | 3.0029e-1 | 0.9270 | 0.0730 | 0.1111 |
| 12 | 5.5676e-1 | -0.9450 | 0.0550 | 0.0909 |
| 16 | 2.1076e0 | 0.9630 | 0.0370 | 0.0667 |
| 20 | 8.5786e0 | -0.9730 | 0.0270 | 0.0526 |
| 24 | 3.6401e1 | 0.9790 | 0.0210 | 0.0435 |
| 30 | 3.3394e2 | -0.9840 | 0.0160 | 0.0345 |
| 40 | 1.4465e4 | 0.9890 | 0.0110 | 0.0256 |

Three things.

**Up to $n = 8$ the maximum is at the centre**, exactly at $t = 0$. That is the regime where the
interpolant is still converging, and its worst point is where the function's curvature is
greatest. Eight nodes is also where the overall error bottoms out.

**From $n = 10$ the maximum jumps into the outermost gap** and approaches $\pm 1$ steadily.
Fitting over $n = 10$ to 40 gives

$$
1 - |t^*| \approx 1.63\, n^{-1.362}
$$

Comparing with the last column, the maximum sits at about **43 percent** of the way into the final
gap rather than at its midpoint, and that fraction drifts slowly with $n$, which is what the
exponent of $-1.362$ rather than $-1$ records. The node polynomial's own maximum is at the
midpoint of the gap; the error's maximum is pulled slightly inward by the variation of
$f^{(n+1)}$, which is largest at the centre of the interval.

**The sign alternates**, $+, -, +, -$. The node polynomial changes sign at every node, so which
end holds the larger extremum depends on the parity of $n$.

### 4.3 A real singularity just outside the interval

Take $f(t) = 1/(t - c)$ with $c > 1$ real, so the singularity is on the real axis just past the
right endpoint.

| $c$ | distance $d = c-1$ | $\rho$ real pole | $\rho$ imaginary at the same $d$ | equal, $n=24$ | Chebyshev, $n=24$ | measured Chebyshev rate |
|---|---|---|---|---|---|---|
| 2.00 | 1.000 | 3.7321 | 2.4142 | 2.356e-11 | 3.753e-14 | 0.3358 |
| 1.50 | 0.500 | 2.6180 | 1.6180 | 9.577e-8 | 3.721e-10 | 0.3820 |
| 1.20 | 0.200 | 1.8633 | 1.2198 | 3.104e-4 | 3.259e-6 | 0.5367 |
| 1.05 | 0.050 | 1.3702 | 1.0512 | 3.409e-1 | 2.087e-2 | 0.7300 |
| 1.01 | 0.010 | 1.1518 | 1.0100 | 2.119e1 | 6.725e0 | 0.8712 |

**The measured rate is $1/\rho$, to four digits.** At $c = 1.5$, $1/\rho = 0.3820$ and the
measured rate is 0.3820. At $c = 1.2$, $1/\rho = 0.5367$ and the measurement is 0.5367. At
$c = 1.05$, $0.7298$ against $0.7300$. Only the first row is off, 0.2679 predicted against 0.3358
measured, and there the error reaches $10^{-14}$ by $n = 24$ so the fit is contaminated by
rounding.

**A real pole is easier than an imaginary one at the same distance.** For a pole on the real axis
at $1 + d$, $\rho = (1+d) + \sqrt{(1+d)^2 - 1} \approx 1 + \sqrt{2d}$; for one on the imaginary
axis at $id$, $\rho \approx 1 + d$. At $d = 0.05$ that is 1.3702 against 1.0512, a rate of 0.73
against 0.95, which over 24 nodes is a factor of $10^3$ in the final error.

The reason is the shape of the Bernstein ellipse: with foci at $\pm 1$ it extends much further
along the real axis than perpendicular to it, so a singularity placed on the real axis is further
outside the same ellipse.

That is why Runge's example uses **complex** poles. They are the harder case at a given distance,
and unlike a real pole just past the endpoint they are completely invisible on the interval.

### 5.1 The potential theory explanation

**The criterion.** Let the nodes have limiting distribution $\mu$ on $[a,b]$, with logarithmic
potential

$$
U^\mu(z) = \int \log|z - s|\,d\mu(s)
$$

Then $\frac1n\log|w_n(z)| \to U^\mu(z)$, and the interpolants of $f$ converge uniformly on $[a,b]$
if and only if every singularity $z_0$ of $f$ satisfies

$$
U^\mu(z_0) > \max_{t\in[a,b]} U^\mu(t)
$$

The set where equality holds is the level curve bounding the region of convergence.

**Runge's function.** For equally spaced nodes $\mu$ is uniform on $[-1,1]$, and $U^\mu$ takes its
maximum on the interval **at the endpoints**, not the centre. The level curve through $\pm1$
crosses the imaginary axis at $\pm 0.5255i$. Runge's poles at $\pm 0.2i$ are inside it, so the
interpolants diverge. Measured threshold: between 0.5 and 1.0.

For Chebyshev nodes $\mu$ is the **arcsine** distribution $1/(\pi\sqrt{1-t^2})$, whose potential
is **constant** on $[-1,1]$, equal to $\log(1/2)$. Constant means every singularity outside the
interval satisfies the criterion strictly, so Chebyshev interpolation converges for **any**
function analytic on a neighbourhood of the interval, with no condition on the pole location at
all. That is the theoretical statement of what lesson 47 measures.

**The $\exp$ failure.** $\exp$ is entire, so it has no singularities and the criterion is
satisfied vacuously: the approximation error goes to zero for both node families. The equally
spaced failure at 60 nodes is therefore not a potential theory phenomenon at all. It is the
Lebesgue constant, which is a statement about **rounding** amplification and lives in a different
theory. That two distinct mechanisms produce superficially similar divergence is the main reason
lesson 46 separates them.

### 5.2 Faber's theorem

**Theorem (Faber, 1914).** For any prescribed triangular array of nodes
$\{x^{(n)}_i\}_{i=0}^n$, $n = 1, 2, \dots$, in $[a,b]$, there exists a continuous $f$ whose
interpolants at those nodes do not converge uniformly to $f$.

**What it means for the search for a universally good node family.** The search is over: there is
none. No node placement makes interpolation converge for every continuous function.

The proof is a uniform boundedness argument. If the interpolants converged for every $f$, the
operators $L_n$ would be pointwise bounded on $C[a,b]$, so by Banach-Steinhaus $\|L_n\|$ would be
uniformly bounded. But $\|L_n\| = \Lambda_n$ by exercise 44.5.2, and it is a theorem of Erdos that

$$
\Lambda_n \ge \frac{2}{\pi}\log n - C
$$

for **every** node family. So $\Lambda_n \to \infty$ always, and Banach-Steinhaus supplies an $f$
for which convergence fails.

**Two consequences worth separating.** The lower bound $\frac2\pi\log n$ says Chebyshev nodes are
**within a constant** of the best possible, since exercise 47.4.1 measures their constant at
0.6326 against $2/\pi = 0.6366$. So the search for a better node family is not merely hopeless in
principle, it is pointless in practice: there is nothing meaningful left to gain.

And Faber's theorem is about **continuous** functions. For functions analytic on a neighbourhood
of the interval, Chebyshev interpolation does converge, geometrically, as exercise 47.4.3
measures. The gap between the two statements is exactly the gap between "continuous" and
"smooth", and it is why every practical convergence theorem in Part 7 has a smoothness
hypothesis.

### 5.3 Shrinking the interval

Take Runge's function on $[-\alpha, \alpha]$ and shrink $\alpha$.

The poles stay at $\pm i/5$. Rescaling $t = \alpha s$ maps the problem to $[-1,1]$ with poles at
$\pm i/(5\alpha)$, so the **relative** pole distance grows as $\alpha$ shrinks. By exercise 4.1's
threshold of 0.5255, the interpolants converge once

$$
\frac{1}{5\alpha} > 0.5255 \quad\Longleftrightarrow\quad \alpha < 0.3806
$$

| $\alpha$ | relative pole $1/(5\alpha)$ | $n = 16$ | $n = 24$ | $n = 32$ | diverges |
|---|---|---|---|---|---|
| 1.00 | 0.2000 | 2.108e0 | 3.641e1 | 7.053e2 | yes |
| 0.70 | 0.2857 | 7.295e-1 | 5.058e0 | 3.930e1 | yes |
| 0.50 | 0.4000 | 1.529e-1 | 3.415e-1 | 8.539e-1 | yes |
| 0.40 | 0.5000 | 3.803e-2 | 3.395e-2 | 3.390e-2 | **no, and flat** |
| 0.35 | 0.5714 | 1.426e-2 | 6.880e-3 | 3.710e-3 | no |
| 0.20 | 1.0000 | 6.731e-5 | 1.445e-6 | 3.501e-8 | no |
| 0.10 | 2.0000 | 8.278e-9 | 9.428e-12 | 1.083e-9 | no |

**The measured transition is between a relative pole distance of 0.4 and 0.5**, against the
predicted 0.5255. The row at $\alpha = 0.4$, relative distance exactly 0.5, is essentially flat:
3.803e-2, 3.395e-2, 3.390e-2 across a doubling of the degree. That is what sitting on the boundary
looks like.

The last row shows the other end of the story: at $\alpha = 0.1$ the error reaches
$9.4\times10^{-12}$ at 24 nodes and then **rises** to $1.1\times10^{-9}$ at 32. That is not the
Runge phenomenon returning, it is lesson 47's Lebesgue constant taking over once the approximation
error has been exhausted, exactly as it does for $\exp$.

**So the Runge phenomenon is not a property of the function**, it is a property of the function
**and** the interval together, and the same function is easy on a short enough interval.

That is the theoretical justification for the practical response of lessons 50 and 51:
**subdivide.** Splitting $[-1,1]$ into pieces short enough that each contains no nearby
singularity makes equally spaced interpolation converge on every piece, and the pieces are then
joined. A cubic spline is the extreme case of that idea, with the degree fixed at 3 and as many
pieces as necessary.

---

## Lesson 47, Chebyshev Interpolation

### 1.1 Why the nodes cluster at the ends

Because that is where the node polynomial is worst, and the node polynomial is the only factor of
lesson 46's error that node placement controls.

For equally spaced nodes, exercise 46.2.4 measured $|w|$ at 24 nodes to be **33119 times larger**
near the ends than in the middle. The reason is combinatorial: a point in the outermost gap has
distances $h/2, 3h/2, 5h/2, \dots$ to the nodes, so the product is
$h^{n+1}(2n+1)!!/2^{n+1}$, while a point in a central gap has two small distances, two of the next
size, and so on, and the product is smaller by roughly $2^n/\sqrt n$.

Clustering nodes toward the ends shortens exactly those outermost gaps. The Chebyshev nodes
$\cos((2j+1)\pi/2n)$ have spacing proportional to $\sqrt{1-t^2}$, so the gaps near $\pm1$ are
$O(1/n^2)$ where the central gaps are $O(1/n)$. That is precisely the density that equalises the
node polynomial, and equalising it is what minimising its maximum means.

The result is a $|w|$ that **equioscillates** between $\pm 2^{1-n}$, the same size everywhere,
instead of one that is negligible in the middle and enormous at the ends.

### 1.2 Chebyshev polynomials from their power coefficients

**What went wrong**: the coefficients of $T_k$ grow like $2^k$ and alternate in sign, so
evaluating $\sum_j a_j t^j$ on $[-1,1]$ sums terms of size up to $2^{29}$ that must cancel to
produce an answer bounded by 1. At degree 30 the largest coefficient is $2^{29} = 5.4\times10^8$
and the answer is at most 1, so about 9 digits are lost to cancellation before anything else
happens.

It is the same failure as lesson 44's power form, and for the same reason: a basis whose
coefficients are much larger than the function they represent.

**Two better ways.**

**The recurrence**, $T_{k+1} = 2tT_k - T_{k-1}$. Costs $k$ steps and every intermediate value is
bounded by 1 on the interval, so nothing can cancel catastrophically. Measured against the cosine
form, the relative gap is below $10^{-12}$ at every degree from 0 to 25.

**The cosine identity**, $T_k(t) = \cos(k\arccos t)$. One arccos, one multiply, one cos. It is
$O(1)$ rather than $O(k)$, and it is exact to the accuracy of the trigonometric functions
themselves.

Use the recurrence when you need many degrees at once, since it produces them all in one sweep;
the cosine form when you need one degree at many points.

### 1.3 What the Lebesgue constant is the condition number of

**The problem**: given the data values $y_i = f(x_i)$ at fixed nodes, produce the interpolating
polynomial, evaluated somewhere.

**The input** is the vector $y$, measured in the maximum norm. **The output** is the function
$L_nf$, measured in the supremum norm on the interval.

By exercise 44.5.2 the operator norm of that map is exactly $\Lambda_n$. So a relative perturbation
$\delta$ in the data produces a perturbation of at most $\Lambda_n\delta$ in the interpolant, and
some perturbation attains it.

That is lesson 15's definition of a condition number applied to this problem, and it makes the two
practical consequences immediate. Data known to $\varepsilon$ gives an interpolant known to
$\Lambda_n\varepsilon$, and at $\Lambda_{60} = 1.5\times10^{15}$ for equally spaced nodes that is
no information at all. And an interpolant is within a factor $1 + \Lambda_n$ of the best possible
polynomial approximation, so a large $\Lambda_n$ makes interpolation a poor approximation method
even where good approximations exist.

### 2.1 The minimax theorem

**Claim.** Among monic polynomials of degree $n$ on $[-1,1]$, $T_n/2^{n-1}$ uniquely minimises
$\|p\|_\infty$, and the minimum is $2^{1-n}$.

$T_n$ has leading coefficient $2^{n-1}$, so $\tilde T_n = T_n/2^{n-1}$ is monic, and
$\|\tilde T_n\|_\infty = 2^{1-n}$ since $|T_n| \le 1$ with equality attained.

Suppose some monic $p$ of degree $n$ has $\|p\|_\infty < 2^{1-n}$. Consider
$q = \tilde T_n - p$. Both are monic of degree $n$, so **$q$ has degree at most $n-1$**.

Now $T_n$ equioscillates: at the $n+1$ points $t_j = \cos(j\pi/n)$, $j = 0,\dots,n$, it takes the
values $(-1)^j$, so $\tilde T_n(t_j) = (-1)^j 2^{1-n}$.

At each $t_j$, $|p(t_j)| < 2^{1-n}$, so $q(t_j) = \tilde T_n(t_j) - p(t_j)$ has the **sign of
$(-1)^j$**. A function alternating in sign at $n+1$ points has at least $n$ roots between them. But
$q$ has degree at most $n-1$, so $q \equiv 0$, contradicting $\|p\| < \|\tilde T_n\|$.

**Uniqueness** follows from the same argument applied to a $p$ achieving equality: $q$ then
vanishes at $n$ points and is of degree $n-1$, so is zero, so $p = \tilde T_n$.

**The equioscillation is doing all the work**, which is why it is the characterising property of
minimax approximation in general.

### 2.2 $T_k(\cos\theta) = \cos k\theta$

By induction. $T_0(\cos\theta) = 1 = \cos 0$ and $T_1(\cos\theta) = \cos\theta$.

For the step, the recurrence with $t = \cos\theta$ gives

$$
T_{k+1}(\cos\theta) = 2\cos\theta\cos k\theta - \cos(k-1)\theta
$$

and the product formula $2\cos A\cos B = \cos(A+B) + \cos(A-B)$ turns the first term into
$\cos(k+1)\theta + \cos(k-1)\theta$, so the result is $\cos(k+1)\theta$.

**The three consequences.**

$|T_k(t)| \le 1$ on $[-1,1]$, since $|\cos| \le 1$ and every $t$ in the interval is $\cos\theta$
for some real $\theta$.

The **roots** are where $\cos k\theta = 0$, that is $k\theta = (2j+1)\pi/2$, giving
$t_j = \cos((2j+1)\pi/2k)$ for $j = 0,\dots,k-1$: $k$ distinct roots in $(-1,1)$.

The **extrema** are where $\cos k\theta = \pm1$, that is $\theta = j\pi/k$, giving
$t_j = \cos(j\pi/k)$ for $j = 0,\dots,k$: $k+1$ points at which $T_k$ alternates between $+1$ and
$-1$, which is the equioscillation 2.1 needs.

### 2.3 The change of interval

The affine map $x = \frac{a+b}{2} + \frac{b-a}{2}t$ takes $[-1,1]$ to $[a,b]$, with inverse
$t = (x - \frac{a+b}{2})/\frac{b-a}{2}$.

**Claim.** The minimum of $\max_{[a,b]}|w|$ over monic degree $n$ polynomials is
$2^{1-n}\left(\frac{b-a}{2}\right)^n$.

Let $w$ be monic of degree $n$ in $x$. Substituting the map, $w$ becomes a polynomial in $t$ whose
leading coefficient is $\left(\frac{b-a}{2}\right)^n$, since each factor $x - x_i$ becomes
$\frac{b-a}{2}(t - t_i)$. So $\left(\frac{2}{b-a}\right)^n w$ is monic in $t$, and by 2.1 its
maximum is at least $2^{1-n}$. Rearranging gives the claim, with equality when the $x_i$ are the
mapped Chebyshev nodes.

**Measured** at $n = 8$:

| interval | width factor $((b-a)/2)^n$ | measured $\max\lvert w\rvert$ | predicted |
|---|---|---|---|
| $[-1, 1]$ | 1.0000e0 | 7.8125e-3 | 7.8125e-3 |
| $[0, 1]$ | 3.9062e-3 | 3.0518e-5 | 3.0518e-5 |
| $[0, 4]$ | 2.5600e2 | 2.0000e0 | 2.0000e0 |
| $[-3, 5]$ | 6.5536e4 | 5.1200e2 | 5.1200e2 |

Exact agreement at every interval, to every digit shown.

**The consequence.** On an interval of half width $H$, the best node polynomial has maximum
$2(H/2)^n$. For $H > 2$ that **grows** with the degree, so raising the degree makes the
controllable factor worse, and only the factorial can save it. That is the quantitative argument
for subdividing rather than raising the degree, and it is the reason lessons 50 and 51 exist.

### 2.4 The Lebesgue bound

**Claim.** $\|f - L_nf\|_\infty \le (1 + \Lambda_n)\|f - p^*_n\|_\infty$ where $p^*_n$ is the best
polynomial approximation of degree $n$.

$L_n$ reproduces polynomials of degree $n$ exactly, so $L_np^*_n = p^*_n$. Then

$$
f - L_nf = (f - p^*_n) + (p^*_n - L_nf) = (f - p^*_n) - L_n(f - p^*_n)
$$

using linearity. Taking norms,

$$
\|f - L_nf\| \le \|f - p^*_n\| + \|L_n\|\,\|f - p^*_n\| = (1 + \Lambda_n)\|f - p^*_n\|
$$

**The step that matters** is $L_np^*_n = p^*_n$: the interpolation operator is a **projection**
onto the polynomials of degree $n$, and the bound is the standard bound for a projection, which is
one plus its norm.

### 2.5 The two growth rates, and where $2/\pi$ comes from

**Equally spaced.** $\Lambda_n \sim \dfrac{2^{n+1}}{en\log n}$, exponential.

**Chebyshev.** $\Lambda_n = \dfrac{2}{\pi}\log n + O(1)$, logarithmic. The precise constant is
$\frac2\pi\log n + \frac2\pi(\gamma + \log(8/\pi)) + o(1)$.

**Where $2/\pi$ comes from.** The Lebesgue function for the Chebyshev nodes can be written, using
the closed form weights of exercise 3 below, as

$$
\Lambda(t) = \frac{|T_n(t)|}{n}\sum_j \frac{1}{|t - t_j|}
$$

Near the maximum, which sits between the outermost nodes, the sum behaves like a discrete
approximation to $\int_{-1}^{1}\frac{d\mu(s)}{|t-s|}$ with $\mu$ the arcsine distribution
$\frac{1}{\pi\sqrt{1-s^2}}$, whose density carries the $1/\pi$. Summing the $2n$ contributions
picks up the factor 2, and the logarithm arises from the divergence of $\int ds/|t-s|$ at $s = t$
cut off at the node spacing $\sim 1/n$. Hence $\frac2\pi\log n$.

**Measured**, fitting $\Lambda_n = C + s\log n$ over $n = 4$ to 20: $s = 0.6326$ against
$2/\pi = 0.6366$, a 0.6 percent agreement.

### 3.1 The fast cosine transform route

On the Chebyshev-Lobatto points $x_j = \cos(j\pi/N)$, the interpolant's Chebyshev coefficients are
a **discrete cosine transform** of the data, computable in $O(N\log N)$ by an FFT.

```python
def chebyshev_coefficients(values):
    """Values at the Lobatto points, in decreasing-x order, to Chebyshev coefficients.

    The trick is to extend the data evenly around the circle and take a real FFT: a cosine
    series on [-1, 1] is a Fourier series on the circle restricted to even functions.
    """
    v = np.asarray(values, dtype=float)
    N = v.size - 1
    extended = np.concatenate([v, v[-2:0:-1]])          # even extension, length 2N
    spectrum = np.real(np.fft.fft(extended))[:N + 1] / N
    spectrum[0] *= 0.5
    spectrum[-1] *= 0.5
    return spectrum


def evaluate_chebyshev_series(coeffs, t):
    """Clenshaw's algorithm, which is Horner's rule for a Chebyshev series."""
    a = np.asarray(coeffs, dtype=float)
    z = np.asarray(t, dtype=float)
    b1 = np.zeros_like(z)
    b2 = np.zeros_like(z)
    for k in range(a.size - 1, 0, -1):
        b1, b2 = 2.0 * z * b1 - b2 + a[k], b1
    return z * b1 - b2 + a[0]
```

Measured against the barycentric route on $\exp$:

| $N$ | transform against barycentric | series reproduces its own nodes |
|---|---|---|
| 16 | 8.88e-16 | 4.44e-16 |
| 64 | 1.33e-15 | 6.66e-16 |
| 256 | 1.78e-15 | 9.44e-16 |

**Both are correct and the comparison is not about cost.** The closed form barycentric weights of
exercise 3.3 below are already $O(N)$, so the transform does not save on setup. What it buys is
the **coefficients themselves**, which the barycentric form never produces, and those are what you
need to differentiate, integrate, truncate or multiply the interpolant. That is why Chebfun stores
coefficients and not values.

### 3.2 A Chebfun style adaptive constructor

```python
def chebfun_construct(f, lo=-1.0, hi=1.0, tol=None, max_degree=2 ** 14):
    """Double the degree until the tail of the Chebyshev coefficients is at machine precision,
    then trim. That tail is the sharpest available estimate of the truncation error, because a
    Chebyshev series converges at the rate its coefficients decay."""
    tol = float(np.finfo(float).eps) if tol is None else float(tol)
    n = 16
    while n <= max_degree:
        x = cb.extrema_nodes(n + 1, lo, hi)
        a = chebyshev_coefficients(np.asarray([f(v) for v in x])[::-1])
        scale = max(float(np.max(np.abs(a))), 1e-300)
        tail = np.abs(a[-max(n // 8, 2):])
        if float(np.max(tail)) <= tol * scale:
            keep = int(np.max(np.flatnonzero(np.abs(a) > tol * scale)) + 1)
            return a[:keep], x
        n *= 2
    return a, x
```

Measured on functions of decreasing smoothness:

| function | degree found | converged | error achieved |
|---|---|---|---|
| $\exp$ | 14 | yes | 8.88e-16 |
| $1/(1+25t^2)$ | 180 | yes | 7.77e-16 |
| $\lvert t\rvert$ | 8192, capped | **no** | 3.19e-5 |
| $\sqrt{\lvert t\rvert}$ | 8192, capped | **no** | 2.71e-3 |

**The constructor is a smoothness detector.** An entire function needs 15 coefficients, a function
with poles near the interval needs 183, and a function with a singularity **on** the interval never
terminates. That is not a defect: the coefficients genuinely do not decay to machine precision, and
a code that stopped anyway would be lying about its accuracy.

The practical response for the last two rows is to split the interval at the singularity, which is
what Chebfun does automatically and what lesson 51 does by hand.

### 3.3 Chebyshev differentiation

Differentiating the interpolant gives a matrix $D$ with $(Df)_i = p'(x_i)$.

```python
def differentiation_matrix(n, lo=-1.0, hi=1.0):
    """Trefethen's cheb: D[i,j] = c_i / (c_j (x_i - x_j)) off the diagonal, with the diagonal
    set by the fact that D must annihilate constants."""
    x = cb.extrema_nodes(n, lo, hi)[::-1]
    c = np.ones(n)
    c[0] = c[-1] = 2.0
    c = c * (-1.0) ** np.arange(n)
    X = np.tile(x, (n, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n))
    D -= np.diag(D.sum(axis=1))
    return D, x
```

| $n$ | error in $p'$ for $\exp$ | $\lVert D\rVert_2$ | $n^2$ | $\lVert D\rVert_2 / n^2$ |
|---|---|---|---|---|
| 8 | 6.25e-6 | 2.86e1 | 64 | 0.447 |
| 16 | 1.29e-14 | 1.25e2 | 256 | 0.488 |
| 32 | 6.71e-14 | 5.30e2 | 1024 | 0.518 |
| 64 | 5.88e-13 | 2.18e3 | 4096 | 0.532 |
| 128 | 8.25e-12 | 8.87e3 | 16384 | 0.541 |

**The accuracy is spectral until $n = 16$**, reaching $1.3\times10^{-14}$, and then **degrades**
by a factor of about 8 for each doubling.

$\lVert D\rVert_2$ grows like $n^2$, with the ratio steady at 0.45 to 0.54, because the Chebyshev
nodes cluster like $1/n^2$ at the ends and differentiation divides by the spacing. A data error of
$\varepsilon$ therefore becomes a derivative error of about $0.5n^2\varepsilon$, and at $n = 128$
that is $1.8\times10^{-12}$, against the measured $8.3\times10^{-12}$: within a factor of 5.

**A note on the condition number, which is the wrong thing to quote here.** $\kappa_2(D)$ measures
$10^{16}$ to $10^{18}$ across this table, and that is not a sign of trouble. $D$ annihilates
constants by construction, since the derivative of a constant is zero, so $D$ is **exactly
singular** and its condition number is infinite in exact arithmetic. What was measured is the
rounding level of that zero. The meaningful quantity is $\lVert D\rVert_2$, which is what
multiplies the data error.

**Differentiation is an unbounded operator and no discretisation escapes that.** The right
response is not a lower degree, since then approximation error dominates, but to recognise there
is an optimal degree, about 16 for this function, where the two error curves cross.

### 4.1 The Lebesgue constants and where they cross

| $n$ | equally spaced | Chebyshev |
|---|---|---|
| 10 | 1.7849e1 | 2.0083 |
| 20 | 5.8895e3 | 2.4481 |
| 30 | 3.4477e6 | 2.7058 |
| 40 | 2.4219e9 | 2.8874 |
| 50 | 1.8658e12 | 3.0208 |
| 60 | 1.5202e15 | 3.1406 |

Fitted: equally spaced $\Lambda_n = 0.0902\times1.7563^n$, Chebyshev
$\Lambda_n = C + 0.6326\log n$.

**The equally spaced constant crosses $1/\varepsilon = 4.5\times10^{15}$ at $n = 68.3$.**

That number is the practical meaning of the whole of lessons 46 and 47. Past 68 equally spaced
nodes, the interpolant of **exact** data in double precision carries no correct digits, whatever
the function is and however carefully the arithmetic is done. Lesson 46's measurement of $\exp$
reaching an error of 0.16 at 60 nodes is that threshold being approached.

The Chebyshev constant reaches 3.14 at 60 nodes and would reach about 6 at $10^6$ nodes.

### 4.2 Convergence against smoothness

Using $f_p(t) = |t|^p$, which has $\lceil p\rceil - 1$ continuous derivatives:

**Integer $p$ is the wrong family to use**, and it is worth saying why. $|t|^2 = t^2$ is a
polynomial, so Chebyshev interpolation reproduces it exactly at any degree above 2 and the
measured rate is meaningless: fitting it gives $n^{+0.22}$, which is rounding noise. The same
happens at every even $p$. Non-integer exponents give a genuine family of smoothness classes:

| $p$ in $\lvert t\rvert^p$ | continuous derivatives | measured rate | theory |
|---|---|---|---|
| 0.5 | 0 | $n^{-0.50}$ | $n^{-0.5}$ |
| 1.0 | 0 | $n^{-1.00}$ | $n^{-1.0}$ |
| 1.5 | 1 | $n^{-1.50}$ | $n^{-1.5}$ |
| 2.5 | 2 | $n^{-2.51}$ | $n^{-2.5}$ |
| 3.5 | 3 | $n^{-3.52}$ | $n^{-3.5}$ |
| 5.5 | 5 | $n^{-5.56}$ | $n^{-5.5}$ |

**The rate is exactly $p$**, measured to two digits at every value including the fractional ones.

Note this corrects a common shorthand. The rate is not "the number of continuous derivatives plus
one", which would give 1, 1, 2, 3, 4, 6 for these rows. It is the **Holder exponent** of the
highest derivative that exists, which for $|t|^p$ is $p$ itself. The two agree for integer $p$ and
part company otherwise.

For analytic $f$ the rate becomes geometric, which is exercise 4.3.

The practical statement is that Chebyshev interpolation extracts **all** the smoothness a function
has, and no more. There is no threshold below which it fails and none above which it stops
improving, and the exponent it achieves is a direct readout of the function's regularity.

### 4.3 The Bernstein ellipse

For $f$ analytic inside the ellipse with foci $\pm1$ and semi-axis sum $\rho$, the Chebyshev
interpolation error is $O(\rho^{-n})$.

For a pole at $z_0$, $\rho = |z_0 + \sqrt{z_0^2 - 1}|$, taking the branch with $\rho > 1$.

| $a$ | pole $i/\sqrt a$ | $\rho$ | predicted rate $1/\rho$ | measured rate |
|---|---|---|---|---|
| 1 | 1.0000 | 2.4142 | 0.4142 | 0.4142 |
| 4 | 0.5000 | 1.6180 | 0.6180 | 0.6180 |
| 25 | 0.2000 | 1.2198 | 0.8198 | 0.8209 |
| 100 | 0.1000 | 1.1050 | 0.9050 | 0.9112 |
| 400 | 0.0500 | 1.0512 | 0.9512 | 0.9642 |

**Agreement to four digits in the first two rows and to two in the rest.** The drift in the last
rows is the fit being taken over degrees 8 to 32, which for a rate of 0.96 has not reached the
asymptotic regime.

This is the sharpest quantitative statement in the whole of Part 7. It says exactly how fast
Chebyshev interpolation converges, from a single geometric fact about where the function's
singularities are, and the measurement confirms it to the digit.

### 5.1 The Bernstein ellipse, stated and used

**Theorem (Bernstein).** Let $f$ be analytic in the open Bernstein ellipse $E_\rho$, the ellipse
with foci $\pm1$ and semi-axes summing to $\rho > 1$, and bounded there by $M$. Then the Chebyshev
interpolant of degree $n$ satisfies

$$
\|f - p_n\|_\infty \le \frac{4M}{\rho - 1}\rho^{-n}
$$

**Why an ellipse.** The Joukowski map $z = \frac12(u + u^{-1})$ carries the circle $|u| = \rho$ to
$E_\rho$ and carries $[-1,1]$ to the unit circle. Under it, Chebyshev series become Laurent series,
and the decay of a Laurent coefficient is governed by the radius of the annulus of analyticity.
So the natural region for Chebyshev approximation is the image of a circle, which is an ellipse.

**Predicting exercise 4.3.** For a pole at $z_0$, the largest ellipse avoiding it is the one
through $z_0$, giving $\rho = |z_0 + \sqrt{z_0^2-1}|$. Substituting the poles of $1/(1+at^2)$ at
$\pm i/\sqrt a$ produces the third column of the table above, and it matches the measured rates.

**Two consequences worth naming.** The rate depends only on the **nearest** singularity, so a
function with one nearby pole and a hundred distant ones converges at the rate of the nearby one.
And it depends on the pole's position relative to an ellipse, not a circle, so a pole on the real
axis is much less damaging than one at the same distance on the imaginary axis, which is exactly
what exercise 46.4.3 measured.

### 5.2 Chebyshev nodes are not optimal for the Lebesgue constant

They are optimal for the **node polynomial**, which is exercise 2.1's theorem. The Lebesgue
constant is a different functional and they do not minimise it.

**What does.** The nodes minimising $\Lambda_n$ are characterised by the condition that the
Lebesgue function equioscillates, taking the same maximum in every gap. Those are sometimes called
the **optimal nodes** and they have no closed form; they are computed numerically by a
Remez-like iteration.

**How much better, measured.** Fitting the Chebyshev-Lobatto Lebesgue constant as
$\frac2\pi\log n + C$ and reading off $C$:

| $n$ | Lobatto $\Lambda_n$ | $\frac2\pi\log n$ | implied $C$ |
|---|---|---|---|
| 10 | 2.3619 | 1.4659 | 0.8960 |
| 30 | 3.1063 | 2.1653 | 0.9410 |
| 100 | 3.8879 | 2.9317 | 0.9561 |
| 300 | 4.5915 | 3.6311 | 0.9604 |
| 600 | 5.0339 | 4.0724 | 0.9615 |

Erdos proved that **every** node family satisfies $\Lambda_n \ge \frac2\pi\log n + 0.9625$.

The measured constant climbs steadily toward that value: 0.8960, 0.9410, 0.9561, 0.9604, 0.9615.
So the Chebyshev-Lobatto points are not merely close to optimal, they are **asymptotically
optimal**, and the gap between them and the best possible node family goes to zero.

**Why nobody uses the optimal nodes.** They cost a numerical optimisation to find, they have no
closed form barycentric weights so setup becomes $O(n^2)$, they have no fast transform, and the
measurement above says the quantity they improve converges to zero improvement. Three closed form
advantages against an asymptotically vanishing gain. The choice is not close.

That is a useful general lesson: a theorem saying something is not optimal is worth much less than
a measurement saying by how much.

### 5.3 Chebyshev interpolation as a whole system

Chebfun represents every function by its Chebyshev interpolant, adaptively chosen to reach machine
precision, and then does calculus on that representation.

**What becomes easy.**

**Differentiation and integration** become matrix operations on the coefficients, both exact for
the representation. Integration is a two term recurrence on the coefficients and is perfectly
conditioned; the definite integral is Clenshaw-Curtis quadrature and comes free.

**Rootfinding** becomes an eigenvalue problem: the roots of a Chebyshev series are the eigenvalues
of the colleague matrix, which is a Chebyshev basis analogue of lesson 35's companion matrix, and
is much better conditioned than it.

**Extrema, norms, arithmetic** all reduce to operations on coefficient vectors.

**What becomes hard.**

**Multiplication** doubles the degree, so a product of many functions grows the representation.
Chebfun truncates back to machine precision after each operation, which is safe and costs a
transform.

**Composition** $f(g(t))$ requires resampling and readapting, which is expensive and can fail if
$g$ maps outside the domain.

**What breaks.**

A function with a **singularity on the interval** never reaches machine precision, as exercise 3.2
measured on $|t|$ and $\sqrt{|t|}$. Chebfun's answer is to detect that and split the domain
automatically, which turns one representation into a piecewise one and is lesson 51's idea arriving
by a different road.

A function on an **unbounded** domain, or with a **very** narrow feature, needs a mapped or heavily
refined representation, and the adaptive constructor can consume enormous memory before deciding.

The design lesson is that a representation good enough to compute with is worth building a whole
system around, and that the boundary of what it handles is exactly the boundary of where its
convergence theory applies.

---

## Lesson 48, Finite Difference Operators and Tables

### 1.1 What differs between $\Delta$, $\nabla$ and $\delta$

**Numerically, nothing.** All three compute differences of consecutive values, and applied to the
same sequence they produce the same numbers. `nalib.finitediff` implements all three with the same
loop, and says so.

**What differs is where the answer is indexed.**

$\Delta f_i = f_{i+1} - f_i$ is attached to node $i$, so a forward difference table's first column
is aligned with the top of the data.

$\nabla f_i = f_i - f_{i-1}$ is attached to node $i$, so the same numbers align with the bottom.

$\delta f_i = f_{i+1/2} - f_{i-1/2}$ is attached to the **midpoint**, so an odd order central
difference lands between the nodes.

**Why all three have names.** Because the formulas built from them read different corners of the
table, and which corner you can reach depends on where in the table you are. Lesson 49's whole
structure is that a forward formula works at the top, a backward one at the bottom, and a central
one only in the middle, and that is an indexing fact rather than an arithmetic one.

The $\delta$ offset in particular is not cosmetic: it is why there are two Gauss formulas rather
than one, and why $\mu$ exists.

### 1.2 $E = e^{hD}$ concretely

**Concretely** it says
$$
f(x+h) = \sum_{k\ge0}\frac{h^k}{k!}f^{(k)}(x)
$$
which is Taylor's theorem with the series written as the exponential of the differentiation
operator. Applying $e^{hD}$ to $f$ means applying $\sum h^kD^k/k!$, and each $D^k$ is the $k$-th
derivative.

**What $f$ must satisfy.** For the identity as an exact statement, $f$ must be **analytic** at $x$
with radius of convergence exceeding $h$: the series must converge to $f(x+h)$ and not merely be
asymptotic to it.

That is a real restriction. $f(t) = e^{-1/t^2}$ with $f(0)=0$ is $C^\infty$ and every derivative
at 0 is zero, so the series is identically zero and does not equal $f(h)$ for any $h \ne 0$.

**For the purposes this lesson uses it for**, the weaker statement suffices: Taylor's theorem with
remainder says the truncated series is correct to $O(h^{m+1})$ for $f \in C^{m+1}$, which is all
the derivative formulas of section 3 need. The operator identity is a mnemonic for the exact case
and a bookkeeping device for the truncated one.

### 1.3 A noisy fourth difference column

**Explanation one: the data is not polynomial of degree 4 or less**, and the fourth differences are
genuinely varying. That is the ordinary case and there is nothing wrong.

**Explanation two: the data has rounding or measurement noise**, and the fourth difference column
is that noise amplified. By exercise 4.1 a single error $e$ appears in column $k$ amplified by
$\binom{k}{\lfloor k/2\rfloor}$, which at $k=4$ is 6, and independent noise of size $\sigma$ in
every value produces column noise of size $\sigma\sqrt{\binom{2k}{k}} = \sigma\sqrt{70} = 8.4\sigma$
at $k = 4$.

**The measurement that separates them**: compare the size of the fourth column with the size of
the **fifth**.

If the data is genuinely a smooth non-polynomial function, the columns keep shrinking by roughly
$h\max|f^{(k+1)}|/\max|f^{(k)}|$ per order, so the fifth column is smaller than the fourth.

If the column is noise, the next column **grows**, by a factor of about
$\sqrt{\binom{2k+2}{k+1}/\binom{2k}{k}} \approx 2$, because differencing noise amplifies it.

So: shrinking columns means signal, growing columns means noise, and the order at which the
behaviour flips is where the signal drops below the noise floor.

### 2.1 The binomial expansion of $\Delta^k$

**Claim.** $\Delta^k f_i = \sum_{j=0}^{k}(-1)^j\binom kj f_{i+k-j}$.

By induction. At $k=1$, $\Delta f_i = f_{i+1} - f_i$, which is the claim.

For the step, $\Delta^{k+1}f_i = \Delta^k f_{i+1} - \Delta^k f_i$, so by the hypothesis

$$
\Delta^{k+1}f_i = \sum_j (-1)^j\binom kj f_{i+1+k-j} - \sum_j (-1)^j\binom kj f_{i+k-j}
$$

Reindexing the second sum with $j' = j-1$ and combining gives coefficients
$\binom kj + \binom k{j-1} = \binom{k+1}{j}$ by Pascal's rule, with the alternating sign carried
through.

**The error pattern follows immediately.** Perturbing $f_m$ by $e$ changes $\Delta^kf_i$ by
$e(-1)^{j}\binom kj$ where $j = i+k-m$, which is nonzero exactly for $i = m-k, \dots, m$. So the
perturbation appears in $k+1$ consecutive rows with the alternating binomial coefficients, centred
on $m - k/2$, and its largest entry is $e\binom{k}{\lfloor k/2\rfloor}$.

**Measured**, for a fault of size 1 at index 8:

| $k$ | pattern | amplification | $2^k$ |
|---|---|---|---|
| 2 | $+1, -2, +1$ | 2 | 4 |
| 3 | $+1, -3, +3, -1$ | 3 | 8 |
| 4 | $+1, -4, +6, -4, +1$ | 6 | 16 |
| 5 | $+1, -5, +10, -10, +5, -1$ | 10 | 32 |
| 6 | $+1, -6, +15, -20, +15, -6, +1$ | 20 | 64 |

### 2.2 They all commute

Each operator is a formal series in $E$:

$$
\Delta = E - 1, \quad \nabla = 1 - E^{-1}, \quad \delta = E^{1/2}-E^{-1/2},
\quad \mu = \tfrac12(E^{1/2}+E^{-1/2}), \quad hD = \log E
$$

Powers of $E$ commute with each other, $E^aE^b = E^{a+b} = E^bE^a$, since both mean "shift by
$(a+b)h$". Any two series in a single commuting element commute, because the product of two such
series is a series whose coefficients are convolutions, and convolution is commutative.

**What the licence is worth**: lesson 49's formulas are derived by manipulating expressions like
$E^s = (1+\Delta)^s$ with the binomial series, and by substituting one series into another. Every
such step assumes the operators can be reordered and regrouped, which is exactly commutativity and
associativity. Without it, none of the classical derivations would be valid.

**Measured** on random data, $\Delta^j\Delta^k$ against $\Delta^k\Delta^j$ against
$\Delta^{j+k}$: relative gaps below $10^{-10}$ for every $(j,k)$ tested at lengths 8 to 20.

### 2.3 The logarithm series

From $E = e^{hD}$, taking logarithms of both sides as operators gives $hD = \log E$. Substituting
$E = 1+\Delta$ and expanding,

$$
hD = \log(1+\Delta) = \Delta - \frac{\Delta^2}{2} + \frac{\Delta^3}{3} - \cdots
$$

**The order of the truncation.** Apply the truncated series to $f$ at $x_i$. Each $\Delta^m f_i$
is, by the divided difference relation of lesson 45, equal to $m!h^m f[x_i,\dots,x_{i+m}]$, which
by the derivative connection is $h^m f^{(m)}(\xi_m)$. So the term $\Delta^m/m$ contributes
$h^mf^{(m)}/m$, and dividing by $h$ gives $h^{m-1}f^{(m)}/m$.

Truncating after $m$ terms therefore leaves a first omitted term of size
$h^m f^{(m+1)}/(m+1)$, so the approximation to $f'$ is $O(h^m)$.

**Measured** on $\exp$ with $h = 0.01$ over 14 points:

| $m$ | max error | $h^m$ | error / $h^m$ | $1/(m+1)$ |
|---|---|---|---|---|
| 1 | 5.656e-3 | 1e-2 | 0.5656 | 0.5000 |
| 2 | 3.749e-5 | 1e-4 | 0.3749 | 0.3333 |
| 3 | 2.796e-7 | 1e-6 | 0.2796 | 0.2500 |
| 4 | 2.225e-9 | 1e-8 | 0.2225 | 0.2000 |

Each extra term buys exactly one factor of $h$, and the constant sits about 12 percent above
$1/(m+1)$ at every order, which is the difference between the first omitted term and the whole
remaining tail.

### 2.4 Polynomial differences and the divided difference relation

**Claim one.** For $p$ of degree $d$ with leading coefficient $a_d$, $\Delta^dp = d!\,a_dh^d$
constant and $\Delta^{d+1}p = 0$.

$\Delta$ reduces the degree by exactly one: writing $p(x+h) - p(x)$ and expanding, the $x^d$ terms
cancel and the leading surviving term is $da_dhx^{d-1}$. Applying $\Delta$ $d$ times gives a
constant $d(d-1)\cdots1\,a_dh^d = d!a_dh^d$, and once more gives zero.

**Claim two.** $f[x_i,\dots,x_{i+k}] = \Delta^kf_i/(k!h^k)$ on an equally spaced grid.

By lesson 45's exercise 2.1, the divided difference is the leading coefficient of the interpolating
polynomial through those $k+1$ points. Apply claim one to that polynomial, which has degree $k$:
its $k$-th difference is $k!h^k$ times its leading coefficient. And the $k$-th difference of the
interpolant equals the $k$-th difference of $f$ at those nodes, since they agree there and
$\Delta^k$ uses only those values. Rearranging gives the claim.

**Measured** on polynomials of degree 0 to 5 at three spacings: the column goes constant at
exactly the degree in every case, and the constant matches $d!a_dh^d$ to $10^{-9}$ relative.

### 2.5 The operator relations, and where each is used

**$\delta = \Delta E^{-1/2}$.** Both sides applied to $f$ at $x$ give $f(x+h/2) - f(x-h/2)$. Used
to convert between forward and central tables, which is how the Gauss formulas are read off a
forward difference table in practice.

**$\mu^2 = 1 + \delta^2/4$.** Expand:
$\mu^2 = \frac14(E^{1/2}+E^{-1/2})^2 = \frac14(E + 2 + E^{-1})$, and
$\delta^2 = E - 2 + E^{-1}$, so $1 + \delta^2/4 = \frac14(E+2+E^{-1})$. Used in **Stirling's**
formula, whose terms alternate between $\mu\delta^{2k+1}$ and $\delta^{2k}$, and where this
identity is what lets the odd terms be written with $\mu$.

**$\Delta = \mu\delta + \delta^2/2$.** Expand the right side:
$\mu\delta = \frac12(E^{1/2}+E^{-1/2})(E^{1/2}-E^{-1/2}) = \frac12(E - E^{-1})$, and
$\delta^2/2 = \frac12(E - 2 + E^{-1})$. Adding gives $E - 1 = \Delta$. Used in **Bessel's**
formula, and in converting a forward formula to a central one term by term.

Each is a one line verification once every operator is written as a series in $E$, which is the
whole point of section 3 of the lesson.

### 3.1 A symbolic operator calculus

```python
class Shift:
    """A finite Laurent series in E^(1/2), stored as {half_power: coefficient}."""

    def __init__(self, terms):
        self.terms = {k: v for k, v in terms.items() if abs(v) > 1e-15}

    def __add__(self, other):
        out = dict(self.terms)
        for k, v in other.terms.items():
            out[k] = out.get(k, 0.0) + v
        return Shift(out)

    def __sub__(self, other):
        return self + Shift({k: -v for k, v in other.terms.items()})

    def __mul__(self, other):
        out = {}
        for a, u in self.terms.items():
            for b, v in other.terms.items():
                out[a + b] = out.get(a + b, 0.0) + u * v
        return Shift(out)

    def __eq__(self, other):
        keys = set(self.terms) | set(other.terms)
        return all(abs(self.terms.get(k, 0.0) - other.terms.get(k, 0.0)) < 1e-12 for k in keys)


ONE = Shift({0: 1.0})
E = Shift({2: 1.0})
E_HALF = Shift({1: 1.0})
E_INV_HALF = Shift({-1: 1.0})
DELTA = E - ONE
NABLA = ONE - Shift({-2: 1.0})
DELTA_C = E_HALF - E_INV_HALF
MU = Shift({1: 0.5, -1: 0.5})
```

Rederiving the three relations of exercise 2.5, automatically:

| identity | verified |
|---|---|
| $\delta = \Delta E^{-1/2}$ | True |
| $\mu^2 = 1 + \delta^2/4$ | True |
| $\Delta = \mu\delta + \delta^2/2$ | True |
| $\nabla = \Delta E^{-1}$ | True |
| $\delta^2 = \Delta\nabla$ | True |

All five confirmed symbolically rather than numerically, which is stronger: a numerical check
confirms an identity on one data set and a symbolic one confirms it always.

### 3.2 Richardson extrapolation

```python
def richardson(f, x, h, levels, central=True):
    """Halve h and combine to kill the leading error term. The forward quotient has error
    c1 h + c2 h^2 + ..., so each level buys ONE order. The central quotient has only even
    powers, so each level buys TWO."""
    p = 2 if central else 1
    step = 2 if central else 1
    quotient = ((lambda s: (f(x + s) - f(x - s)) / (2.0 * s)) if central
                else (lambda s: (f(x + s) - f(x)) / s))
    table = [[quotient(h / 2.0 ** k) for k in range(levels + 1)]]
    for m in range(1, levels + 1):
        prev = table[-1]
        factor = 2.0 ** (p + step * (m - 1))
        table.append([(factor * prev[k + 1] - prev[k]) / (factor - 1.0)
                      for k in range(len(prev) - 1)])
    return table
```

Measured on $\exp'(0) = 1$ with $h = 0.4$:

**Forward quotient**, one order per level:

| level | value | error | gain |
|---|---|---|---|
| 0 | 1.22956174410318 | 2.30e-1 | |
| 1 | 0.98446583749852 | 1.55e-2 | 14.8x |
| 2 | 1.00038414844997 | 3.84e-4 | 40.4x |
| 3 | 0.99999621743128 | 3.78e-6 | 101.6x |
| 4 | 1.00000001553281 | 1.55e-8 | 243.5x |
| 5 | 0.99999999997262 | 2.74e-11 | 567.2x |

**Central quotient**, two orders per level:

| level | value | error | gain |
|---|---|---|---|
| 0 | 1.02688081450704 | 2.69e-2 | |
| 1 | 0.99994641210495 | 5.36e-5 | 501.6x |
| 2 | 1.00000001273551 | 1.27e-8 | 4207.8x |
| 3 | 0.99999999999956 | 4.41e-13 | 28872.7x |
| 4 | 1.00000000000000 | 1.78e-15 | 248.3x |
| 5 | 1.00000000000000 | 8.88e-16 | 2.0x |

**The central version reaches machine precision in four levels**, using nine function
evaluations, where the forward version has reached only $2.7\times10^{-11}$ in six.

The gains grow at both levels, because two effects compound: the extrapolation removes one error
term, and each row also uses a step half the size of the row above. For the forward quotient the
gains run 14.8, 40.4, 101.6, 243.5, 567.2, roughly doubling per level, which is the order rising
by one each time. For the central one they run 502, 4208, 28873, roughly by eight, which is the
order rising by two.

The last two central rows flatten at machine precision, as they must, and the level 5 gain of 2.0
is rounding rather than extrapolation.

### 3.3 A two fault finder

```python
def locate_two_faults(y, order=4, tol=0.3):
    """Fit the column to a sum of TWO shifted binomial patterns by least squares, over every
    pair of candidate positions. Costs O(n^2) trials of an O(k) fit."""
    v = np.asarray(y, dtype=float)
    k = int(order)
    col = fd.forward(v, k)
    binom = np.asarray([((-1) ** j) * math.comb(k, j) for j in range(k + 1)], dtype=float)
    n = v.size
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            A = np.zeros((col.size, 2))
            for c, idx in enumerate((i, j)):
                lo = max(idx - k, 0)
                hi = min(idx + 1, col.size)
                if hi > lo:
                    A[lo:hi, c] = binom[lo - (idx - k):hi - (idx - k)]
            if np.linalg.matrix_rank(A) < 2:
                continue
            sizes, *_ = np.linalg.lstsq(A, col, rcond=None)
            resid = float(np.linalg.norm(col - A @ sizes))
            if best is None or resid < best[0]:
                best = (resid, i, j, sizes)
    resid, i, j, sizes = best
    scale = max(float(np.max(np.abs(col))), 1e-300)
    return {"indices": (i, j), "sizes": tuple(float(s) for s in sizes),
            "residual": resid / scale, "reliable": bool(resid / scale < tol)}
```

Measured, two faults planted in $\sin(0.2k)$ over 20 points:

Measured, faults of $+0.05$ and $-0.03$ planted in $\sin(0.2k)$ over 20 points:

| separation | planted at | found at | sizes recovered | reliable |
|---|---|---|---|---|
| 8 | 5, 13 | 5, 13 | 0.0500, -0.0300 | True |
| 6 | 5, 11 | 5, 11 | 0.0500, -0.0300 | True |
| 5 | 5, 10 | 5, 10 | 0.0500, -0.0300 | True |
| 4 | 5, 9 | 5, 9 | 0.0500, -0.0300 | True |
| 3 | 5, 8 | 5, 8 | 0.0500, -0.0300 | True |
| 2 | 5, 7 | 5, 7 | 0.0500, -0.0300 | True |
| 1 | 5, 6 | 5, 6 | 0.0500, -0.0300 | True |

**It works at every separation, down to adjacent faults.** Both positions and both sizes are
recovered exactly, to four decimals, in all seven cases.

That was not the expected answer. The intuition is that two overlapping binomial patterns should
be inseparable, and it is wrong here for a specific reason: the two patterns are shifted copies of
a fixed vector, so the two columns of the least squares matrix are **linearly independent**
whenever the shifts differ at all, however small the shift. Independence is all least squares
needs, and the conditioning stays modest because the binomial pattern alternates in sign, so a
shift by one changes the sign structure completely rather than nearly reproducing it.

The limit is instead the number of faults against the length of the column: with $n$ values and
order $k$ there are $n - k$ equations, so at most $n - k$ faults can be fitted, and well before
that the residual stops distinguishing hypotheses.

### 4.1 The amplification, fitted

| $k$ | measured amplification | $\binom{k}{\lfloor k/2\rfloor}$ | $2^k$ | ratio to $2^k$ |
|---|---|---|---|---|
| 2 | 2 | 2 | 4 | 0.500 |
| 3 | 3 | 3 | 8 | 0.375 |
| 4 | 6 | 6 | 16 | 0.375 |
| 5 | 10 | 10 | 32 | 0.312 |
| 6 | 20 | 20 | 64 | 0.312 |
| 8 | 70 | 70 | 256 | 0.273 |
| 10 | 252 | 252 | 1024 | 0.246 |

**The amplification is exactly the central binomial coefficient**, at every order, with no error.

It is **not** $2^k$, and the discrepancy grows: the ratio falls from 0.500 at $k = 2$ to 0.246 at
$k = 10$, consistent with Stirling's
$\binom{k}{k/2} \approx 2^k/\sqrt{\pi k/2}$, which at $k = 10$ predicts a ratio of 0.252.

The distinction matters when estimating whether a fault is detectable. At order 8 the true
amplification is 70 and the folklore value is 256, so a fault estimated as detectable on the
folklore number may not be.

### 4.2 How large the data's own differences may be

Planting a fault of size $\Delta^4_{\text{own}}/s$ into $\sin(0.2k)$ over 16 points:

| own $\lvert\Delta^4\rvert$ / fault | found correctly | reliable flag | agree |
|---|---|---|---|
| 0.01 | True | True | True |
| 0.10 | True | True | True |
| 0.50 | True | True | True |
| 1.00 | True | True | True |
| 2.00 | True | **False** | False |
| 5.00 | True | **False** | False |
| 20.00 | True | **False** | False |

**The flag is conservative, not wrong.** It never passes an incorrect answer, and from a ratio of 2
onward it rejects correct ones.

That is the right direction for a guard to err, and it refines the lesson's claim. The precise
statement is: the flag is **sound** (reliable implies correct) and not **complete** (correct does
not imply reliable). The lesson's measurement on $\exp(0.3k)$, where the ratio was 11.1 and the
answer **was** wrong, shows the flag catching a genuine failure; this measurement shows it also
declining some successes.

The practical rule that follows: `reliable` True means act on it; `reliable` False means
investigate rather than dismiss.

### 4.3 The log series against $m$ and $h$ together

| $h$ | $m=1$ | $m=2$ | $m=3$ | $m=4$ | $m=6$ |
|---|---|---|---|---|---|
| 1e-1 | 3.13e-1 | 1.97e-2 | 1.40e-3 | 1.06e-4 | 6.83e-7 |
| 1e-2 | 6.01e-3 | 3.98e-5 | 2.97e-7 | 2.36e-9 | **3.50e-13** |
| 1e-3 | 5.09e-4 | 3.39e-7 | 2.55e-10 | 8.44e-13 | 1.20e-12 |
| 1e-4 | 5.01e-5 | 3.34e-9 | 4.60e-12 | 6.26e-12 | 1.39e-11 |
| 1e-5 | 5.00e-6 | 6.77e-11 | 5.29e-11 | 7.51e-11 | 1.57e-10 |
| 1e-6 | 5.00e-7 | 3.22e-10 | 4.70e-10 | 6.36e-10 | 1.19e-9 |

**The best combination is $m = 6$ at $h = 10^{-2}$, reaching $3.5\times10^{-13}$**, with
$m = 4$ at $h = 10^{-3}$ close behind at $8.4\times10^{-13}$.

Every column past $m = 1$ has the U shape of lesson 45, and **the bottom moves left as $m$ grows**:
$m = 2$ bottoms at $10^{-5}$, $m = 3$ and $m = 4$ at $10^{-3}$, $m = 6$ at $10^{-2}$. More terms
means a larger optimal step, which is the opposite of the naive expectation and follows directly
from the balance: truncation is $O(h^m)$ so it falls faster for larger $m$, and rounding is
$O(\varepsilon/h^m)$ so it rises faster too, and the crossing moves to larger $h$.

$m = 1$ has no U at all over this range, because its truncation error $O(h)$ still dominates at
$h = 10^{-6}$.

**The best error achieved, $3.5\times10^{-13}$, is far better than $\sqrt\varepsilon =
1.5\times10^{-8}$**, which a single difference quotient is limited to. Two knobs reach further
than one: raising $m$ buys order without paying the rounding cost that shrinking $h$ would.

### 5.1 The umbral calculus

The formal manipulation of $\Delta$, $E$ and $D$ as if they were numbers is made rigorous by the
**umbral calculus** of Rota and Roman, which recasts it as a statement about linear functionals on
the polynomials.

The setting is the algebra of **formal power series** in a variable $t$, and its action on the
polynomials by $t^k \mapsto D^k$. Every shift invariant operator, meaning one commuting with $E^a$
for all $a$, is uniquely a formal series in $D$, and the correspondence is an algebra isomorphism.

**What it licenses.** Because the correspondence is an isomorphism of algebras, any identity
provable between formal power series translates to an identity between the corresponding
operators. So $\log(e^{hD}) = hD$ is a formal identity between series with no convergence
question, and $(1+\Delta)^s = \sum\binom sk\Delta^k$ is the binomial series in the formal algebra.

**What a naive reading does not license.** The isomorphism is with the action on **polynomials**,
where every series terminates because $D^k p = 0$ for $k$ past the degree. Applying the same
identity to a general $f$ requires the series to converge, which is exercise 5.2. So the umbral
calculus licenses the algebra and says nothing about the analysis, and the two must be separated.

### 5.2 Why the series diverges and still works

$\log(1+\Delta)$ is a power series in an operator with no small parameter: $\Delta$ is not small,
it is an operator with norm up to $2\|f\|_\infty$.

**In what sense it converges.** Three answers, of increasing usefulness.

**On polynomials, it terminates.** For $p$ of degree $d$, $\Delta^kp = 0$ for $k > d$, so the
series has finitely many nonzero terms and there is no convergence question at all. That is the
umbral setting.

**On analytic $f$ with small enough $h$, it converges.** The operator $\Delta$ applied to $f$
sampled at spacing $h$ has $\|\Delta^kf\| \le h^k\max|f^{(k)}|$, so for $f$ analytic with radius
$R$ the series converges when $h < R$.

**On $C^m$ functions, it is asymptotic.** The truncation after $m$ terms is $O(h^m)$ by exercise
2.3, and that statement needs only $f \in C^{m+1}$. The full series may diverge and the truncated
one is still correct to the stated order.

**What must be assumed** depends on which statement is wanted. For the derivative formulas of the
lesson, the third suffices and requires only $f \in C^{m+1}$. For the exact operator identity
$E = e^{hD}$, analyticity with radius exceeding $h$ is needed, and exercise 1.2's counterexample
$e^{-1/t^2}$ shows $C^\infty$ is not enough.

The pattern is the usual one for asymptotic series: the first few terms are useful, the series as
a whole may be meaningless, and knowing which regime you are in is the whole skill.

### 5.3 What survives, what is superseded

**Still used, under the same name.**

**The difference table as a data quality check.** Used in surveying, actuarial work and anywhere
data arrives on a uniform grid. The fault location technique of section 6 is still the fastest way
to find a transcription error in tabulated data.

**Central differences.** The observation that symmetric formulas have their odd error terms cancel
is used in every finite difference method for a differential equation, which is Parts 10 and 11.

**Still used, under other names.**

**Richardson extrapolation**, which is exercise 3.2, is now called that everywhere and appears as
Romberg integration in Part 9, as deferred correction in Part 10, and as the basis of the
Bulirsch-Stoer method.

**The operator calculus** is how every finite difference stencil is still derived, though it is
usually presented as "match Taylor coefficients" rather than as an operator identity. The
computation is identical.

**Genuinely superseded.**

**Interpolation from printed tables.** Nobody looks up $\sin$ in a book, so Bessel's and Everett's
formulas have no remaining use in their original job.

**Hand computation of derivatives from a table**, which is now done by automatic differentiation
where the function is available and by splines where only the data is.

**Not done at all any more.**

**Subtabulation**, the filling in of a coarse table at finer spacing to be printed. The entire
economics that made Everett's halving of the table worth a name has gone.

The honest summary is that the **formulas** are obsolete and the **technique that produced them**
is not. Anyone deriving a new finite difference scheme in Part 11 is doing exactly what section 3
of this lesson does.

---

## Lesson 49, The Equal Interval Interpolation Formulas

### 1.1 What is actually being chosen between

**Not the answer.** At full order and away from the ends, all eight formulas produce the same
polynomial, and the lesson measured the spread across all eight at the centre of a nine node table
as 0.00e0.

What is being chosen is **which entries of the difference table get read**, and that matters in
exactly three situations.

**When the series is truncated.** At a fixed order the formulas are different polynomials, and
exercise 4.1 measures which is best where.

**When the table is short at one end.** A central formula near the first node reaches order 0,
because there is nothing above it. That is the structural fact of section 4 of the lesson.

**When the table has an error in it.** Exercise 4.3 measures the amplification of a single bad
entry: 10.85 for the diagonal formulas and 1.00 for Everett and Steffensen. That is a factor of
eleven, and it has nothing to do with truncation.

**The one situation in which the choice changes the answer** is therefore the second: near the
ends the central formulas cannot be formed at full order at all, so they are computing a different,
lower degree polynomial. Everywhere else the differences are about accuracy or robustness, not
about which function you get.

### 1.2 Why there are two Gauss formulas

Because $\delta$ lands on the **half points**. $\delta f_i = f_{i+1/2} - f_{i-1/2}$ is attached to
the midpoint between two nodes, so an odd order central difference is not a value at a node.

A central formula expanding about node $x_0$ must therefore use odd differences from either the
half point above or the one below, and both choices are available:

$$
p = f_0 + s\,\delta f_{1/2} + \binom s2\delta^2f_0 + \cdots \quad\text{(forward)}
$$
$$
p = f_0 + s\,\delta f_{-1/2} + \binom{s+1}{2}\delta^2f_0 + \cdots \quad\text{(backward)}
$$

Both are the same polynomial at full order; they zig-zag through the table in opposite directions.

There is no such fork for $\Delta$ or $\nabla$, because those land on nodes, so there is only one
diagonal to follow in each direction. The fork exists **only** for the central operator, and it is
what makes Stirling and Bessel, the two averages, worth having.

### 1.3 What Everett saved

Everett's formula uses only the **even** order difference columns, and it needs two central nodes
instead of one.

**What that saved**: half the table. A book of tables printing $f$, $\delta^2 f$, $\delta^4 f$ and
$\delta^6 f$ could be interpolated to sixth order with Everett; printing the odd columns as well
would have doubled the page count for no gain to an Everett user.

**For whom**: the publisher, who paid for typesetting and paper, and the user, who carried the
book. A twentieth century table of logarithms or of the trigonometric functions ran to hundreds of
pages, and halving the columns was a substantial economy.

**Does it save anything now**: no. Memory is free at these sizes and nobody interpolates from a
printed table. What survives is the observation itself, that a formula can be arranged to need
only part of the available data, which reappears whenever data is expensive rather than storage.

### 2.1 Deriving Gregory-Newton forward

Write $t = x_0 + sh$, so the interpolant evaluated at $t$ is $E^sf_0$. Substituting $E = 1+\Delta$
and expanding by the binomial series for real exponent,

$$
E^sf_0 = (1+\Delta)^sf_0 = \sum_{k\ge0}\binom sk \Delta^kf_0
$$

with $\binom sk = s(s-1)\cdots(s-k+1)/k!$.

**Convergence.** As a formal identity on polynomials the series terminates, since $\Delta^kp = 0$
for $k$ past the degree, and there is nothing to converge. As an identity on a general $f$, the
binomial series for $(1+z)^s$ converges for $|z| < 1$, so the operator series converges when
$\|\Delta\| < 1$ in whatever norm is being used, which for $f$ sampled at spacing $h$ means
roughly $h\max|f'| < \|f\|$.

In practice the series is always truncated, and the relevant statement is exercise 2.5's error
term rather than convergence of the infinite series.

### 2.2 Gregory-Newton backward

Measure $s$ from the **last** node: $t = x_n + sh$, so the value is $E^sf_n$. Now write
$E = (1 - \nabla)^{-1}$, which follows from $\nabla = 1 - E^{-1}$, so

$$
E^sf_n = (1-\nabla)^{-s}f_n = \sum_k\binom{-s}{k}(-\nabla)^kf_n = \sum_k\binom{s+k-1}{k}\nabla^kf_n
$$

using $\binom{-s}{k}(-1)^k = \binom{s+k-1}{k}$.

**It is the forward formula read from the other end.** Reversing the node order maps
$\Delta \to -\nabla$ and $s \to -s$, and substituting into the forward formula produces the
backward one term by term. So it is not an independent result, and the lesson's measurement that
the two agree to $10^{-15}$ at every $t$ is checking exactly that.

### 2.3 The Gauss formulas, Stirling and Bessel

Write $E = 1 + \delta E^{1/2}$, which follows from $\delta = E^{1/2} - E^{-1/2}$ on multiplying
through by $E^{1/2}$. Then

$$
E^s = (1 + \delta E^{1/2})^s = \sum_k \binom sk \delta^k E^{k/2}
$$

The factor $E^{k/2}$ shifts by half steps for odd $k$, and resolving those shifts to whole nodes
in the two possible ways gives the two Gauss formulas.

**Stirling** is the average $\frac12(\text{forward} + \text{backward})$. Averaging the two odd
terms $\delta f_{1/2}$ and $\delta f_{-1/2}$ gives $\mu\delta f_0$ by the definition of $\mu$, so
Stirling's terms alternate between $\mu\delta^{2k+1}f_0$ and $\delta^{2k}f_0$, all attached to the
central node. It is therefore symmetric in $s \to -s$ up to the sign of the odd terms, which is why
its odd order error contributions cancel.

**Bessel** takes the average about the **midpoint** $x_0 + h/2$ instead. Replacing $s$ by
$s - 1/2$ and averaging the values at $x_0$ and $x_1$ gives terms in $\mu\delta^{2k}f_{1/2}$ and
$\delta^{2k+1}f_{1/2}$, all attached to the half point. It is symmetric about $s = 1/2$, which is
why it is the natural formula for interpolating halfway.

### 2.4 Everett from Bessel

Bessel's formula, written out, is

$$
p = \mu f_{1/2} + (s - \tfrac12)\delta f_{1/2} + \frac{s(s-1)}{2}\mu\delta^2f_{1/2} + \cdots
$$

Substitute $\mu f_{1/2} = \frac12(f_0 + f_1)$ and $\delta f_{1/2} = f_1 - f_0$, and similarly for
the higher terms. The odd order terms then combine with the even ones, and collecting by which
node they attach to gives

$$
p = \sum_k\left[\binom{q+k}{2k+1}\delta^{2k}f_0 + \binom{s+k}{2k+1}\delta^{2k}f_1\right],
\qquad q = 1-s
$$

which is Everett's formula: only even order differences, at two adjacent nodes.

**Table entries needed.** For an interpolant of order $2m$:

| formula | columns used | entries read |
|---|---|---|
| Gregory-Newton | all $2m+1$ | $2m+1$ |
| Bessel | all $2m+1$ | about $2m+1$ |
| Everett | even only, $m+1$ of them | $2m+2$ |

Everett reads slightly **more** entries but from **half as many columns**, and it is the columns
that had to be printed.

### 2.5 The truncation error, and the rule

Truncating Gregory-Newton forward after order $m$ leaves

$$
f(t) - p_m(t) = \binom{s}{m+1}h^{m+1}f^{(m+1)}(\xi)
$$

which follows from lesson 46's error formula with $w(t) = \prod_{i\le m}(t - x_i) =
h^{m+1}s(s-1)\cdots(s-m)$ and the definition of $\binom{s}{m+1}$.

**How it justifies the rule.** The coefficient $\left|\binom{s}{m+1}\right|$ is smallest when $s$
lies **inside** the range $0, \dots, m$ of the nodes used, and grows rapidly outside it. For the
forward formula the nodes used are $x_0, \dots, x_m$, so the error is smallest for $s$ near the
start of the table.

The same computation for the backward formula puts the nodes at the end, and for a central formula
the product is $s(s^2-1)(s^2-4)\cdots$, which is smallest for $|s|$ small, that is near the
central node. For Bessel the shifted product $(s-\frac12)(s^2 - \frac14)\cdots$ is smallest near
$s = \frac12$.

That is the classical rule, and exercise 4.1 measures how well it actually predicts.

### 3.1 Inverse interpolation

Solve $f(t) = c$ by interpolating $t$ as a function of $f$.

```python
def inverse_interpolate(x, y, target):
    """Needs f to be MONOTONE on the data, since otherwise t is not a function of f at all.
    Badly conditioned wherever f' is small, because a small change in f then moves t a long way."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(np.diff(y) <= 0) and np.any(np.diff(y) >= 0):
        raise ValueError("inverse interpolation needs the values to be monotone")
    order = np.argsort(y)
    return float(ip.evaluate_barycentric(y[order], x[order], float(target)))
```

Measured on nine nodes:

| function | interval | target | recovered $t$ | true $t$ | error | $\lvert f'\rvert$ there |
|---|---|---|---|---|---|---|
| $\exp$ | $[0, 2]$ | 3.000 | 1.09850290 | 1.09861229 | 1.09e-4 | 3.000e0 |
| $t^3$ | $[0.5, 2]$ | 4.000 | 1.63181986 | 1.58740105 | 4.44e-2 | 7.560e0 |
| $t^3$ | $[-0.5, 0.5]$ | 0.001 | 0.06461911 | 0.10000000 | 3.54e-2 | 3.000e-2 |

**Two distinct failure modes, and the second is the interesting one.**

The third row is the expected failure: $|f'| = 0.03$ at the target, so the inverse has derivative
33, and interpolating a function with a large derivative from nine points is inaccurate. That is
conditioning and it was predictable.

The second row is not. There $|f'| = 7.56$, which is **larger** than the first row's 3.0, and the
error is 400 times worse. The reason is that the inverse of $t^3$ is $c^{1/3}$, whose derivative
$\frac13c^{-2/3}$ is unbounded at $c = 0$, and the data range $[0.125, 8]$ reaches down toward it.
So the inverse function is **badly behaved as a function**, quite apart from the derivative at the
particular target.

**When it fails**, then, is not simply "when $f'$ is small at the answer". It is when the inverse
function is hard to interpolate anywhere on the data range, which includes any $f$ with a
stationary point or a near stationary point in range. The safe test is to look at the spread of
$f'$ over the data, not its value at one point.

### 3.2 Subtabulation

Filling in a coarse table at a finer spacing, using Gregory-Newton forward at full order:

| coarse nodes | refine by | subtabulated error against $\exp(2t)$ |
|---|---|---|
| 9 | 2 | 1.260e-7 |
| 9 | 4 | 1.549e-7 |
| 9 | 8 | 1.549e-7 |
| 13 | 2 | 1.393e-12 |
| 13 | 4 | 1.886e-12 |
| 13 | 8 | 1.886e-12 |
| 17 | 2 | 1.510e-14 |
| 17 | 4 | 2.132e-14 |
| 17 | 8 | 2.132e-14 |

**The refinement factor barely matters and stops mattering entirely past 4.** Going from 2 to 4
raises the measured maximum by about 25 percent, because the finer grid samples nearer the worst
point; going from 4 to 8 changes nothing at all, to four digits, at every coarse size.

The error is set entirely by the **coarse** table's node count: $1.5\times10^{-7}$ at 9 nodes,
$1.9\times10^{-12}$ at 13, $2.1\times10^{-14}$ at 17.

That is obvious once stated and it is the whole point of subtabulation: the interpolating
polynomial is fixed by the coarse data, so evaluating it at more points cannot add information.
What subtabulation buys is a **printed table** at finer spacing, computed once, so the reader never
has to interpolate. It buys nothing in accuracy, and the accuracy it delivers is exactly the
coarse table's interpolation error.

The practical warning that follows: a finely spaced table produced by subtabulation is **less
accurate than it looks**. Its spacing suggests an accuracy its content does not have.

### 3.3 Automatic formula selection

At a fixed order, choosing the formula by the classical rule at each point:

| order | automatic | GN fwd | GN bwd | Gauss fwd | Gauss bwd | Stirling | Bessel | Everett | Steffensen |
|---|---|---|---|---|---|---|---|---|---|
| 2 | **1.88e-3** | 1.89e0 | 4.03e0 | 5.68e-1 | 8.33e-2 | 2.96e-1 | 1.27e-2 | 1.29e-2 | 3.77e-3 |
| 3 | **4.92e-4** | 6.58e-1 | 1.48e0 | 5.68e-1 | 8.33e-2 | 2.96e-1 | 1.27e-2 | 1.29e-2 | 3.77e-3 |
| 4 | **4.92e-4** | 1.70e-1 | 3.73e-1 | 5.68e-1 | 8.33e-2 | 2.96e-1 | 1.27e-2 | 1.29e-2 | 3.77e-3 |
| 6 | **4.92e-4** | 4.67e-3 | 8.81e-3 | 5.68e-1 | 8.33e-2 | 2.96e-1 | 1.27e-2 | 1.29e-2 | 3.77e-3 |

**The automatic choice beats every single fixed formula, at every order.** At order 2 it reaches
$1.9\times10^{-3}$ against the best single formula's $3.8\times10^{-3}$, and at order 6 it reaches
$4.9\times10^{-4}$ against Gregory-Newton forward's $4.7\times10^{-3}$, a factor of 9.5.

That is a larger gain than the pointwise accuracy of the rule would suggest, since exercise 4.1
measures the rule picking the actual winner only a third of the time. The explanation is that the
rule gets the **ends** right every time, and the ends are where the errors are largest, so the
maximum over the interval is decided there and the rule's failures in the middle never surface in
the maximum.

Note also that the automatic choice stops improving past order 3, at $4.9\times10^{-4}$, for the
reason exercise 4.2 gives: the central formulas it selects in the middle saturate, and the maximum
is then pinned by them.

### 4.1 Which formula actually wins where

Probing **between** the nodes, at three offsets in each of six cells, on a thirteen node table:

| order | the rule picked the winner |
|---|---|
| 2 | 6 of 18 |
| 4 | 6 of 18 |
| 6 | 6 of 18 |

**The rule is right one third of the time**, and the pattern is completely systematic:

| $s$ | winner at order 4 | rule says | agree |
|---|---|---|---|
| 0.15 | GN forward | GN forward | yes |
| 0.50 | GN forward | GN forward | yes |
| 0.85 | GN forward | GN forward | yes |
| 6.15 | Everett | Stirling | no |
| 6.50 | Steffensen | Bessel | no |
| 6.85 | Everett | Stirling | no |
| 11.15 | GN backward | GN backward | yes |
| 11.50 | GN backward | GN backward | yes |
| 11.85 | GN backward | GN backward | yes |

**At the ends the rule is right every time. In the middle it is wrong every time.**

And it is wrong in a specific way: the rule recommends Stirling and Bessel, and the actual winners
are **Everett and Steffensen**, at every order tested.

The explanation is that Stirling and Bessel truncate a series whose terms attach to a single
central node or half point, while Everett and Steffensen distribute the same information over two
adjacent nodes. At a fixed order that distribution samples the function over a slightly wider
stencil, and on a smooth function that helps.

**The honest revision of the classical rule**: near the start use Gregory-Newton forward, near the
end use backward, and in the middle use **Everett or Steffensen** rather than Stirling or Bessel.
The traditional middle advice appears to be about hand computation convenience, where Stirling's
symmetric terms are easier to write down, rather than about accuracy.

### 4.2 The largest useful order

At a fixed order, on a thirteen node table with $f = \exp(2t)$, probing over $[0.05, 0.95]$:

| order | GN forward | GN backward | Gauss fwd | Gauss bwd | Stirling | Bessel | Everett | Steffensen |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.62e0 | 6.65e0 | 4.56e-2 | 5.37e-2 | 1.69e-2 | 2.25e-2 | 1.14e-2 | 3.65e-3 |
| 2 | 1.67e0 | 3.68e0 | 1.65e-3 | 1.65e-3 | 1.65e-3 | 1.21e-2 | 1.14e-2 | 3.65e-3 |
| 4 | 1.39e-1 | 3.11e-1 | 1.65e-3 | 3.50e-4 | 9.03e-4 | 1.21e-2 | 1.14e-2 | 3.65e-3 |
| 8 | 2.19e-5 | 3.33e-5 | 1.65e-3 | 3.50e-4 | 9.03e-4 | 1.21e-2 | 1.14e-2 | 3.65e-3 |
| 12 | 1.04e-12 | 1.04e-12 | 1.65e-3 | 3.50e-4 | 9.03e-4 | 1.21e-2 | 1.14e-2 | 3.65e-3 |

**The two diagonal formulas keep improving all the way to order 12**, reaching $10^{-12}$.

**The six central formulas saturate**, and early: Gauss forward stops improving at order 2,
Stirling at order 3, and Bessel, Everett and Steffensen at order 1 or 2. Past that, raising the
order changes nothing.

The reason is section 4 of the lesson made quantitative. The maximum is taken over a range that
includes points near the ends, where the usable central order is small. Raising the requested
order does nothing there, because the table runs out, so the maximum is pinned by the boundary
points however high the order goes.

**The relation to lesson 45's symmetry failure** is that this is a different and earlier ceiling.
Lesson 45's limit at order 10 is about the difference table itself losing accuracy. This limit is
about the **formula** running out of table, and it bites at order 2 or 3 rather than 10. For the
central formulas the structural limit arrives first, by a factor of four.

### 4.3 Sensitivity to one bad table entry

Perturbing one value by $10^{-6}$ and measuring the largest resulting change in the interpolant,
over a thirteen node table and probing $[0.05, 0.95]$:

| formula | maximum amplification |
|---|---|
| Gregory-Newton forward | 10.85 |
| Gregory-Newton backward | 10.85 |
| Gauss forward | 1.06 |
| Gauss backward | 1.06 |
| Stirling | 1.01 |
| Bessel | 1.00 |
| Everett | 1.00 |
| Steffensen | 1.00 |

**The central formulas are ten times less sensitive**, and Bessel, Everett and Steffensen are
essentially insensitive: an error of $\epsilon$ in one entry moves the answer by $\epsilon$ and no
more.

This is a completely different argument from the truncation one and it points the same way. A
diagonal formula uses every difference column up to the full order, so it inherits lesson 48's
amplification of $\binom{k}{\lfloor k/2\rfloor}$, which at $k = 12$ is 924. It measures only 10.85
here because the binomial coefficients multiply $\binom sk$ factors that are small over the probe
range, but the mechanism is there.

A central formula near the middle reaches only order 4 or 6, so it never touches the high
difference columns where the amplification lives.

**So the case for the central formulas is stronger than the truncation argument alone**, and it is
the one that would have mattered most to their original users: a hand computed table has
transcription errors in it, and a formula that does not amplify them is worth having.

### 5.1 Why central differences are second order

**Claim.** A symmetric difference formula has an error expansion in **even** powers of $h$ only.

Let $L$ be a difference operator with symmetric coefficients, meaning $Lf(x)$ is built from
$f(x \pm kh)$ with coefficients depending only on $|k|$, in the case of an even formula, or
antisymmetric in the case of an odd one. Expand each $f(x\pm kh)$ by Taylor:

$$
f(x+kh) + f(x-kh) = 2\sum_{j\text{ even}}\frac{(kh)^j}{j!}f^{(j)}(x)
$$
$$
f(x+kh) - f(x-kh) = 2\sum_{j\text{ odd}}\frac{(kh)^j}{j!}f^{(j)}(x)
$$

Every odd power cancels in the first and every even power in the second. So a symmetric formula's
error contains only every **other** power of $h$, and one extra order comes free.

Concretely, the central difference quotient $(f(x+h) - f(x-h))/2h$ has error
$h^2f'''/6 + h^4f^{(5)}/120 + \cdots$, second order rather than first, and Richardson
extrapolation on it gains two orders per level rather than one, which is exercise 48.3.2's
measurement of 502, 4208, 28873 against 14.8, 40.4, 101.6.

**Where the fact is used in Parts 9 to 11.**

**Part 9**: the central difference formulas for derivatives, and the trapezoid rule's
Euler-Maclaurin expansion in even powers, which is what makes Romberg integration gain two orders
per level.

**Part 10**: the midpoint and leapfrog methods for differential equations, whose second order
accuracy is exactly this cancellation, and the symmetric Runge-Kutta methods whose order is even
for the same reason.

**Part 11**: the standard five point Laplacian, whose $O(h^2)$ accuracy is this fact in two
dimensions, and every symmetric finite difference stencil for a partial differential equation.

It is one of the most reused facts in the whole course.

### 5.2 The boundary problem is permanent

Section 4 of the lesson showed central formulas unusable within $k/2$ nodes of either end of the
table. That is not an artifact of interpolation.

**Why every finite difference method meets it.** A central stencil of width $2m+1$ centred at grid
point $i$ needs values at $i-m$ through $i+m$. At the first $m$ points of the grid those values do
not exist, so the interior scheme cannot be applied there, and a boundary value problem must supply
something else. The same counting, and the same shortage.

**Two standard responses.**

**One sided stencils.** Use a formula that reaches only inward, which is exactly Gregory-Newton
forward at the left end and backward at the right. The cost is that a one sided formula of the same
width is one order lower, by exercise 5.1's argument, so the boundary is less accurate than the
interior. That mismatch either limits the whole scheme's order or requires a wider one sided
stencil to compensate.

**Ghost points.** Extend the grid past the boundary with fictitious values chosen to satisfy the
boundary condition, and then apply the interior stencil everywhere. For a Neumann condition
$u'(0) = g$ the reflection $u_{-1} = u_1 - 2hg$ makes the central formula exact at the boundary.
The cost is that the ghost value must be derivable from the boundary condition, which works for
simple conditions and not for complicated ones.

A third response worth naming is to change the discretisation near the boundary altogether, which
is what a finite element method does: its basis functions are defined on elements, and the
elements at the boundary are simply different elements. That sidesteps the counting problem rather
than solving it, and it is one of the reasons finite elements handle complicated boundaries better
than finite differences.

### 5.3 What replaced these formulas

**Jobs now done differently.**

Interpolating a tabulated function is now done by evaluating the function. $\sin$ is computed by
argument reduction and a minimax polynomial, not looked up. Bessel's and Everett's original job
has gone entirely.

Interpolating scattered measured data is now done by splines, lesson 51, which need no uniform
grid and give $C^2$ smoothness.

Differentiating tabulated data is now done by fitting a spline and differentiating it, or by
Savitzky-Golay filtering, both of which handle noise better than a difference table.

**Jobs now done by the same formulas under other names.**

Every finite difference stencil in Parts 9 to 11 is a truncated Gregory-Newton or central formula,
derived by the operator calculus of lesson 48. The five point Laplacian is $\delta^2$ in two
dimensions. The BDF methods for stiff differential equations are Gregory-Newton backward applied to
the derivative.

Richardson extrapolation, Romberg integration and deferred correction are all the same
manipulation of the error series.

**Jobs simply not done.**

Subtabulation, which exercise 3.2 shows adds no accuracy and existed to produce printed tables.

Hand interpolation of any kind, and with it the whole optimisation for which corner of the table is
convenient to read with a pencil.

**The summary.** The formulas are obsolete and the technique is not. What has changed is that
nobody needs a formula optimised for reading a printed table by hand, and everybody still needs to
derive a difference approximation of a given order, which is the same algebra.

---

## Lesson 50, Hermite and Piecewise Interpolation

### 1.1 In what sense Hermite is more accurate

**True sense.** For a **fixed number of nodes**, Hermite is far more accurate, because it produces
a polynomial of degree $2n-1$ where Lagrange produces one of degree $n-1$. That is the comparison
usually meant, and it is not a fair one: Hermite used twice as much data.

**The sense in which the measurement contradicts it.** At a **fixed number of data items**, so
Hermite at $n$ nodes against Lagrange at $2n$, both give degree $2n-1$ and Lagrange wins:

| data items | Hermite | Lagrange | ratio |
|---|---|---|---|
| 4 | 4.3710e-3 | 9.2402e-4 | 4.730 |
| 6 | 5.5772e-6 | 2.6549e-6 | 2.101 |
| 8 | 6.5402e-9 | 4.8065e-9 | 1.361 |
| 10 | 5.9623e-12 | 5.8598e-12 | 1.018 |
| 12 | 4.4409e-15 | 7.1054e-15 | 0.625 |
| 14 | 4.4409e-16 | 1.9318e-14 | 0.023 |

**And then it crosses over at 12 items**, where both have reached machine precision and Hermite
becomes the more **stable** one, because it uses half as many distinct nodes and so has a much
smaller Lebesgue constant.

So the honest statement has three parts. Per node, Hermite is better. Per data item and above
machine precision, Lagrange is better. Per data item and at machine precision, Hermite is better
again, for a different reason.

### 1.2 What the piecewise bound omits

$$
|f - p| \le \frac{h^2}{8}\max|f''|
$$

**No degree.** Lesson 46's error had $f^{(n+1)}/(n+1)!$, and the whole difficulty was that the
numerator can outgrow the denominator. Here the degree is 1, forever, so there is no race.

**No factorial**, for the same reason.

**No node polynomial.** Lesson 46's $\prod(t-x_i)$ was the factor that reached its maximum at the
ends and grew with the degree. Here the product is over one interval and is at most $h^2/4$.

**No Lebesgue constant.** The piecewise linear interpolation operator has norm exactly 1 in the
supremum norm, since the interpolant on each interval is a convex combination of two data values.
So data errors are not amplified at all.

**Why the absence is the important feature.** Every one of lesson 46's two failure modes was caused
by something that grows with the degree. Removing the degree removes both. The bound holds for
every twice differentiable $f$, at every $h$, with no condition on the function's singularities or
on the node placement, which is a class of guarantee polynomial interpolation cannot offer at any
degree.

### 1.3 Weierstrass and Runge together

**Weierstrass.** For every continuous $f$ on $[a,b]$ and every $\epsilon > 0$ there exists a
polynomial $p$ with $\|f-p\|_\infty < \epsilon$.

**Runge.** There is a continuous, indeed analytic, $f$ on $[-1,1]$ such that the sequence
$L_nf$ of interpolants at equally spaced nodes satisfies $\|f - L_nf\|_\infty \to \infty$.

**No contradiction**, because they quantify differently. Weierstrass asserts the existence of
**some** good polynomial at each degree. Runge exhibits a **particular** sequence of polynomials
that is bad. Both can be true of the same function at the same degree, and for Runge's function
both are: at degree 24 the best polynomial is within $6.9\times10^{-3}$ and the equally spaced
interpolant is off by 257.

The link between them is quantitative and is lesson 47's:
$\|f - L_nf\| \le (1+\Lambda_n)\|f - p^*_n\|$. Weierstrass says the second factor tends to zero;
Runge happens because the first grows faster.

### 2.1 Existence and uniqueness of the Hermite interpolant

**Claim.** Given distinct $x_0,\dots,x_{n-1}$ and values $y_i, y'_i$, there is exactly one
polynomial of degree at most $2n-1$ with $p(x_i) = y_i$ and $p'(x_i) = y'_i$.

**Uniqueness first.** If $p$ and $q$ both work, $r = p-q$ has degree at most $2n-1$ and satisfies
$r(x_i) = r'(x_i) = 0$ for every $i$. A root at which the derivative also vanishes is a **double**
root, so $r$ has $2n$ roots counted with multiplicity, and a nonzero polynomial of degree at most
$2n-1$ has at most $2n-1$. So $r\equiv 0$.

**Existence** then follows by dimension counting: the map from coefficient vectors in
$\mathbb{R}^{2n}$ to the $2n$ conditions is linear, and uniqueness says its kernel is trivial, so
it is a bijection. Alternatively, exhibit the solution, which is exercise 2.2's basis.

**Where distinctness is used**: in the multiplicity count. If two nodes coincide the conditions at
them are duplicated or contradictory, and the count of $2n$ roots fails.

### 2.2 The Hermite basis

Define, with $L_i$ the Lagrange basis,

$$
H_i(t) = \left(1 - 2(t-x_i)L_i'(x_i)\right)L_i(t)^2, \qquad
K_i(t) = (t-x_i)L_i(t)^2
$$

Both have degree $2n-1$, since $L_i$ has degree $n-1$.

**The four conditions.**

$H_i(x_j) = \delta_{ij}$: for $j\ne i$, $L_i(x_j) = 0$ so the square vanishes. For $j = i$,
$L_i(x_i) = 1$ and the bracket is 1.

$H_i'(x_j) = 0$: differentiating, $H_i' = -2L_i'(x_i)L_i^2 + (1 - 2(t-x_i)L_i'(x_i))2L_iL_i'$. At
$x_j$ with $j\ne i$ both terms have a factor $L_i(x_j) = 0$. At $x_i$ the expression is
$-2L_i'(x_i) + 2L_i'(x_i) = 0$.

$K_i(x_j) = 0$: at $j \ne i$ from $L_i$, at $j = i$ from the factor $(t - x_i)$.

$K_i'(x_j) = \delta_{ij}$: differentiating, $K_i' = L_i^2 + 2(t-x_i)L_iL_i'$. At $x_j$ with
$j \ne i$ both terms vanish; at $x_i$ the first is 1 and the second is 0.

So $p = \sum y_iH_i + \sum y'_iK_i$ satisfies all $2n$ conditions, which proves existence.

### 2.3 The Hermite error formula

**Claim.** $f(t) - p(t) = \dfrac{f^{(2n)}(\xi)}{(2n)!}\prod_i(t-x_i)^2$.

Same argument as lesson 46's exercise 2.1, with double roots. Fix $t$ not a node and set

$$
g(s) = f(s) - p(s) - \lambda\,w(s)^2, \qquad w(s) = \prod_i(s - x_i)
$$

with $\lambda$ chosen so $g(t) = 0$.

Now $g$ vanishes to **second order** at each of the $n$ nodes, since $f - p$ and its derivative
both vanish there and $w^2$ has a double root at each. Counting with multiplicity that is $2n$
roots, plus the simple root at $t$: $2n+1$ roots.

Rolle gives $g'$ at least $2n$ roots, and repeating, $g^{(2n)}$ has at least one root $\xi$. Since
$p$ has degree at most $2n-1$ and $w^2$ is monic of degree $2n$,

$$
0 = f^{(2n)}(\xi) - \lambda(2n)!
$$

giving the claim.

**The squared node polynomial.** $w^2 \ge 0$ everywhere, so the Hermite error does not change sign
between nodes as the Lagrange error does. And $w^2$ is smaller than $|w|$ where $|w| < 1$ and
larger where $|w| > 1$, so it helps in the middle of the interval and hurts at the ends, which is
why Hermite does nothing about the Runge phenomenon.

### 2.4 The piecewise linear bound

**Claim.** On $[a, a+h]$ with $p$ the straight line through $(a, f(a))$ and $(a+h, f(a+h))$,

$$
\max|f - p| \le \frac{h^2}{8}\max|f''|
$$

and the constant $1/8$ is attained.

By lesson 46's error formula with $n = 1$,
$f(t) - p(t) = \frac{f''(\xi)}{2}(t-a)(t-a-h)$. The product $(t-a)(t-a-h)$ has magnitude at most
$h^2/4$, attained at the midpoint. So the bound is $\frac{h^2}{8}\max|f''|$.

**Attained** by $f(t) = t^2$ on $[0, h]$: then $f'' \equiv 2$, the line through $(0,0)$ and
$(h, h^2)$ is $ht$, and at the midpoint $f - p = h^2/4 - h^2/2 = -h^2/4$, whose magnitude is
$h^2/4 = \frac{h^2}{8}\cdot 2$. Equality.

**Measured** on Runge's function, where the bound tightens to within 1 percent at 128 pieces:

| pieces | error | bound | ratio |
|---|---|---|---|
| 8 | 6.390e-2 | 3.906e-1 | 6.11 |
| 32 | 2.070e-2 | 2.441e-2 | 1.18 |
| 128 | 1.509e-3 | 1.526e-3 | **1.01** |

### 2.5 The general osculating problem

**Problem.** Given distinct $x_0,\dots,x_{m}$ and multiplicities $\nu_i \ge 1$, with values
$f^{(k)}(x_i)$ for $k < \nu_i$, find $p$ of degree at most $N - 1$ where $N = \sum\nu_i$,
satisfying all $N$ conditions.

**Uniqueness.** If $p, q$ both work, $r = p - q$ has a root of multiplicity at least $\nu_i$ at
each $x_i$, so $N$ roots counted with multiplicity, and degree at most $N-1$. So $r \equiv 0$.

**Existence** by the same dimension count, or by the confluent Newton construction of exercise
3.1 of lesson 45.

**The two extremes.**

$m = 0$ with $\nu_0 = N$: one node, all the derivatives there. The conditions are
$p^{(k)}(x_0) = f^{(k)}(x_0)$ for $k < N$, which is **Taylor's polynomial**.

Every $\nu_i = 1$: $N$ nodes, one value each, which is **Lagrange interpolation**.

Between them lies everything else, with Hermite ($\nu_i = 2$ for all $i$) in the middle.

**Measured**, exercise 3.1 of lesson 45: the general construction reproduces Taylor's polynomial to
$10^{-10}$ at $m = 1, 3, 5$, and barycentric Lagrange to $10^{-9}$ at $n = 3, 5, 7$.

### 3.1 PCHIP

Piecewise cubic Hermite with slopes chosen to preserve monotonicity.

```python
def pchip_slopes(x, y):
    """Fritsch and Carlson: the harmonic mean of the neighbouring secants, and ZERO wherever
    they differ in sign, which is what stops the curve overshooting at a turning point."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)
    same = np.sign(delta[:-1]) * np.sign(delta[1:]) > 0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    d[1:-1][same] = ((w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:]))[same]
    d[0], d[-1] = delta[0], delta[-1]
    return d
```

Measured against a not-a-knot cubic spline:

| data | spline overshoot | PCHIP overshoot | spline interpolates | PCHIP interpolates |
|---|---|---|---|---|
| a step-like rise | 0.1283 | **0.0000** | 0.0e0 | 0.0e0 |
| monotone $\exp$ | 0.0000 | 0.0000 | 2.2e-16 | 0.0e0 |
| a sharp corner | 0.1924 | **0.0000** | 0.0e0 | 0.0e0 |

**PCHIP does not overshoot, at all, in any case.** The spline overshoots by 0.128 and 0.192 on the
two non-smooth data sets, which for data known to be non-negative, like a concentration or a
probability, produces a physically impossible interpolant.

Both interpolate the data exactly, so the comparison is purely about behaviour between the points.

**What it costs.** PCHIP is $C^1$ where the spline is $C^2$, and it is $O(h^3)$ where the spline is
$O(h^4)$. On smooth data the spline is more accurate; on data with a corner or a plateau, PCHIP is
the one that does not invent features that are not there.

The rule that follows: use a spline when the underlying function is smooth and you want accuracy,
and PCHIP when the data's shape carries meaning that must be preserved.

### 3.2 Piecewise quadratic

Fit a quadratic to each consecutive triple and use it on the middle interval.

| pieces | linear | quadratic | cubic spline | quadratic order |
|---|---|---|---|---|
| 8 | 6.390e-2 | 1.409e-1 | 5.615e-2 | |
| 16 | 5.355e-2 | 4.778e-2 | 3.745e-3 | 1.56 |
| 32 | 2.070e-2 | 8.261e-3 | 6.555e-4 | 2.53 |
| 64 | 5.850e-3 | 1.114e-3 | 4.032e-5 | 2.89 |
| 128 | 1.509e-3 | 1.419e-4 | 2.380e-6 | **2.97** |

**It works and its order is 3**, sitting exactly between piecewise linear's 2 and the cubic
spline's 4, as the degree would suggest.

**Why it is rarely used**, three reasons and the third is decisive.

It is only $C^0$: the pieces meet in value but not in slope, so it has the same kink problem
piecewise linear has, without the compensating simplicity.

The **stencil is asymmetric**. Each quadratic uses a triple, and with $n$ intervals there is no
symmetric way to assign triples to intervals: one end always has a triple used off centre, which
is where the $n = 8$ row's error of 0.141, **worse than linear**, comes from.

And the cubic spline is better in every respect at a comparable cost: order 4 against 3, $C^2$
against $C^0$, and a symmetric construction with no end awkwardness. Quadratic sits between two
options that are both better than it for different reasons, which is why the piecewise family in
practice is linear, cubic Hermite and cubic spline, with nothing in between.

### 3.3 Adaptive piecewise linear

Bisect any interval whose local linear error exceeds the tolerance.

| tolerance | adaptive pieces | uniform pieces | saving |
|---|---|---|---|
| 1e-2 | 20 | 64 | 3.2x |
| 1e-3 | 66 | 256 | 3.9x |
| 1e-4 | 186 | 512 | 2.8x |
| 1e-5 | 644 | 2048 | 3.2x |

**A steady saving of about 3x**, on Runge's function, which has one localised feature.

The saving does not grow with the tolerance, which is worth understanding. Adaptive refinement
equidistributes $h^2|f''|$, so the piece count for a tolerance $\tau$ is
$\int\sqrt{|f''|}\,dt/\sqrt{8\tau}$ against the uniform
$\sqrt{\max|f''|/8\tau}\times(b-a)$. Both scale like $\tau^{-1/2}$, so the **ratio** is a constant,
namely $(b-a)\sqrt{\max|f''|}/\int\sqrt{|f''|}$, which for Runge's function is about 3.

So adaptivity buys a constant factor here, not a better rate. It buys a growing factor only when
the function has a feature that becomes relatively narrower as the tolerance tightens, such as a
genuine singularity.

### 4.1 Piecewise linear order against smoothness

| function | $f''$ exists | measured order | theory |
|---|---|---|---|
| $\exp$ | yes | 1.984 | 2.0 |
| $t^2$ | yes | 2.000 | 2.0 |
| $\lvert t\rvert^{1.5}$ | no | 1.500 | 1.5 |
| $\sqrt{\lvert t\rvert}$ | no | 0.500 | 0.5 |
| $\lvert t\rvert$ | no | see below | 1.0 |

**The order is $\min(2, p)$ where $p$ is the Holder exponent**, measured to three digits in every
case where the measurement is meaningful.

**The $\lvert t\rvert$ row is a special case worth its own table.** With an even number of
intervals over $[-1,1]$, the point $t = 0$ **is a node**, and $|t|$ is itself piecewise linear
with its only kink there:

| intervals | is 0 a node | error |
|---|---|---|
| 16 | yes | **0.000e0** |
| 17 | no | 5.882e-2 |
| 32 | yes | **0.000e0** |
| 33 | no | 3.030e-2 |
| 64 | yes | **0.000e0** |
| 65 | no | 1.538e-2 |

Fitted order over the odd cases: **1.000**, exactly the theory.

**A knot at the singularity makes the interpolation exact.** That is the whole idea of lesson 51's
knot placement, appearing here in its cleanest possible form: a method that converges at order 1
converges instantly when the mesh is aligned with the feature.

**When $f''$ does not exist** the bound $h^2\max|f''|/8$ is vacuous and convergence still happens,
at the rate the function's actual regularity allows. The proof is a modulus of continuity argument
rather than a Taylor one, and it gives $O(h^p)$ for $f$ Holder continuous with exponent $p$.

### 4.2 Does Lagrange always beat Hermite

Ratio of Hermite error to Lagrange error at equal data, so above 1 means Lagrange wins:

| function | interval | $n=2$ | $n=3$ | $n=4$ | $n=5$ |
|---|---|---|---|---|---|
| $\exp$ | $[0,1]$ | 4.730 | 2.101 | 1.361 | 1.018 |
| $\sin$ | $[0,3]$ | 4.854 | 2.115 | 1.361 | 1.016 |
| $1/(1+t)$ | $[0,2]$ | 4.197 | 2.068 | 1.397 | 1.067 |
| Runge | $[-1,1]$ | 1.308 | 1.242 | 2.021 | **0.745** |

**No, it does not always win.** On Runge's function at $n = 5$, meaning ten data items, Hermite
beats Lagrange by a factor of 1.34.

The three smooth rows show a very consistent pattern: the ratio falls from about 4.7 at four items
toward 1 at ten, essentially independent of the function or the interval. That is the pattern the
lesson describes and it appears to be structural rather than accidental.

Runge's row breaks it because there the Lagrange interpolant is being destroyed by the phenomenon
of lesson 46, and Hermite at half as many distinct nodes suffers less from it. So Hermite wins for
the same reason it wins at machine precision in exercise 1.1: **fewer distinct nodes means a
smaller Lebesgue constant**.

The revised statement: Lagrange wins on smooth functions while approximation error dominates, and
Hermite wins whenever the node count itself is the problem, whether because of rounding or because
of the Runge phenomenon.

### 4.3 Chebyshev against piecewise linear

Function evaluations needed to reach a given accuracy on $\exp$ over $[-1,1]$:

| target | Chebyshev nodes | piecewise pieces | ratio |
|---|---|---|---|
| 1e-2 | 4 | 17 | 4.2x |
| 1e-4 | 6 | 129 | 21.5x |
| 1e-6 | 8 | 2049 | 256.1x |
| 1e-8 | 10 | 16385 | 1638.5x |
| 1e-10 | 11 | 131073 | 11915.7x |

**The ratio grows without bound**, because the two rates are of different kinds: Chebyshev is
geometric in the node count and piecewise linear is $O(h^2)$, so the node counts are $O(\log(1/\tau))$
and $O(\tau^{-1/2})$.

At $10^{-10}$ the difference is a factor of **11916**, and it would be $10^6$ at $10^{-14}$.

**The crossover is at about $10^{-1}$**, meaning piecewise linear is never competitive on a smooth
function on a fixed interval.

That is the whole case for high order methods, and the case against them is everything in lesson
46: the Chebyshev column requires nodes you choose, a function you can evaluate anywhere, and
enough smoothness for the geometric rate to appear. Remove any of those and the comparison
reverses, because the piecewise column does not care about any of them.

### 5.1 Shape preserving interpolation

**The problem.** Monotone data should give a monotone interpolant. A cubic spline need not: it is
determined by smoothness conditions that know nothing about the data's shape, and exercise 3.1
measured it overshooting by 0.19 on a corner.

**The Fritsch-Carlson conditions.** Write $\delta_i = (y_{i+1}-y_i)/h_i$ for the secant slopes and
$d_i$ for the derivative at node $i$. The cubic Hermite piece on $[x_i, x_{i+1}]$ is monotone if
and only if $(\alpha, \beta) = (d_i/\delta_i, d_{i+1}/\delta_i)$ lies in a specific region of the
plane, and a sufficient condition is

$$
0 \le \alpha \le 3, \qquad 0 \le \beta \le 3
$$

meaning **no derivative may exceed three times the adjacent secant slope**, and none may have the
opposite sign.

The PCHIP slopes of exercise 3.1 satisfy this automatically: the weighted harmonic mean of two
secants of the same sign lies between them and so is at most the larger, well within 3, and it is
set to zero when the secants differ in sign, which is a local extremum.

**What it costs.**

**Smoothness**: $C^1$ instead of $C^2$, since the derivatives are chosen for shape and not for
second derivative continuity.

**Accuracy**: $O(h^3)$ instead of $O(h^4)$, and on smooth data the spline is genuinely better.

**Locality gained, though**: the PCHIP slopes are computed from three consecutive points, so
changing one data value changes only two pieces, where a spline's tridiagonal solve couples the
whole interval. For interactive editing that is an advantage rather than a cost.

### 5.2 $h$ refinement and $p$ refinement

Two ways to improve an approximation: more pieces ($h$) or higher degree ($p$).

**What each does.** For $f$ with $k$ derivatives, $h$ refinement converges at $O(h^{\min(m,k)})$
with $m$ the degree, so the rate is capped by the smoothness. $p$ refinement converges
**geometrically** for analytic $f$ and only algebraically otherwise.

**The $hp$ methods** use both, and the rule that makes them work is due to Babuska:

> Refine in $h$ where the function is rough, and in $p$ where it is smooth.

Concretely, for a function with a singularity at one point, the optimal $hp$ mesh grades the
element sizes geometrically toward the singularity while raising the degree linearly away from it.
That achieves **exponential** convergence in the total number of degrees of freedom, even though
the function is not analytic, which neither pure $h$ nor pure $p$ can do.

**When each is right on its own.** Pure $p$ when $f$ is analytic on the whole interval and you can
choose the nodes, which is the Chebyshev case of exercise 4.3 and its factor of 11916. Pure $h$
when $f$ has limited smoothness, since raising the degree past the smoothness buys nothing, or
when the data is given to you on a fixed grid.

The measurement in exercise 4.1 is the cleanest statement of the cap: on $\sqrt{|t|}$ the
piecewise linear order is 0.5, and no higher degree method would do better, because the function
has no more regularity to exploit.

### 5.3 Bernstein's constructive proof

Bernstein's polynomials give an explicit sequence converging uniformly to any continuous $f$:

$$
B_nf(t) = \sum_{k=0}^n f\!\left(\tfrac kn\right)\binom nk t^k(1-t)^{n-k}
$$

**The proof** is probabilistic. $B_nf(t) = \mathbb{E}[f(S_n/n)]$ where $S_n$ is a binomial count
with $n$ trials and success probability $t$. By the law of large numbers $S_n/n \to t$, and uniform
continuity of $f$ converts that into uniform convergence of the expectation.

**Measured rate** on Runge's function mapped to $[0,1]$:

| $n$ | error | ratio to previous |
|---|---|---|
| 20 | 3.8191e-1 | |
| 40 | 2.7448e-1 | 1.391 |
| 80 | 1.8257e-1 | 1.503 |
| 160 | 1.1240e-1 | 1.624 |
| 320 | 6.4606e-2 | 1.740 |

Fitted order over that range: $n^{-0.642}$, with the ratios still climbing toward 2, which would
be order 1.

**Why nobody uses it, in three parts.**

**The rate is $O(1/n)$ and no better, ever.** $B_n$ reproduces only linear functions exactly, so
its error contains a term $\frac{t(1-t)}{2n}f''(t)$ that decays like $1/n$ however smooth $f$ is.
There is no smoothness that improves it. At $n = 320$ the error is still $6.5\times10^{-2}$;
reaching $10^{-6}$ would need $n \approx 10^6$.

**It is not an interpolant.** $B_nf$ does not pass through the data except at the two endpoints, so
it cannot be used where the data must be honoured.

**Chebyshev interpolation reaches $10^{-3}$ on the same function at 30 nodes**, where Bernstein
needs about $10^3$.

**What it is for.** The proof, and the Bernstein basis itself, which is lesson 52's Bezier curves.
There the smoothing property that ruins it as an approximation is exactly the property wanted: the
curve should be a smoothed version of the control polygon, not pass through it. A construction can
be useless for the job it was invented for and essential for another.

---

## Lesson 51, Cubic Splines

### 1.1 The two missing conditions

**What they represent.** The counting is $4n$ coefficients against $4n-2$ conditions, so the
solution set is a two parameter family, not a point. The two parameters are the values of $S''$
at the two ends, or equivalently anything else that pins those down: a derivative, a third
derivative continuity, a relation between the end moments.

Every cubic spline through the data is $S_0 + \alpha A + \beta B$ for two fixed correction
functions $A$ and $B$, each of which interpolates zero data and is $C^2$. The correction functions
are not zero, so the family is genuinely two dimensional.

**Why no cleverness removes the need.** Because the deficit is structural, not accidental. The
data says nothing about the function outside $[x_0, x_n]$ and nothing about its derivatives at the
ends, and $S''(x_0)$ and $S''(x_n)$ are not determined by values inside. A rule that picks them
is adding information, and the only question is which information.

That is why every spline library has a default and why the defaults differ. MATLAB's `spline`
uses not-a-knot, `scipy.interpolate.CubicSpline` also defaults to not-a-knot, and a great deal of
older code uses natural because it is the easiest to code. The difference shows up as two full
orders near the ends, which exercise 4.1 measures.

### 1.2 What "natural" actually assumes

**The assumption.** $S''(x_0) = S''(x_n) = 0$ says the interpolant has zero curvature at both
ends, which is true of the underlying $f$ only when $f''$ happens to vanish there. For a general
$f$ it is simply false.

**What it costs.** The error at the ends drops from $O(h^4)$ to $O(h^2)$, and because the maximum
over the interval is usually attained near an end, the whole interpolant reports as $O(h^2)$.
Measured on $\exp$ over $[0,1]$ with 64 intervals, natural gives $3.26\times10^{-5}$ against
clamped's $4.21\times10^{-10}$, a factor of 77000.

**Where the name comes from and why it misleads.** It is natural in the variational sense of
section 6: the natural spline is the one that minimises $\int (g'')^2$ over all $C^2$
interpolants, and $S''=0$ at the ends is the natural boundary condition of that variational
problem. That is a real theorem. It just is not a statement that the end condition is a good
model of $f$, and readers hear the word as if it were.

When $f''$ really does vanish at both ends, natural is correct and free. Exercise 4.1 finds those
cases.

### 1.3 Why $C^2$ and not $C^3$

**Why not.** On each interval $S$ is a cubic, so $S'''$ is a constant, one constant per interval.
Asking for $S'''$ continuous at the $n-1$ interior knots forces all $n$ constants to be equal,
which makes $S$ a single cubic over the whole range. That single cubic has 4 coefficients and
cannot match $n+1$ data values once $n+1 > 4$.

So $C^3$ and interpolation are incompatible at degree 3, except in the trivial case. Not-a-knot is
the closest approach: it asks for $S'''$ continuous at exactly two knots, $x_1$ and $x_{n-1}$,
which merges the first two intervals into one cubic and the last two into one, and uses up exactly
the two free parameters.

**What would have to change.** Raise the degree. Degree $2m+1$ piecewise polynomials with maximum
smoothness give $C^{2m}$: quintics give $C^4$, septics give $C^6$. The count works out because a
degree $d$ piece has $d+1$ coefficients and the smoothness conditions consume $d$ of them per
interior knot, leaving one for the data. Exercise 5.1 says why nobody does this past cubic.

### 2.1 The moment equations, and diagonal dominance

**Deriving them.** On interval $i = [x_i, x_{i+1}]$ with $h_i = x_{i+1} - x_i$, $S''$ is linear,
so write it in terms of the moments $M_i = S''(x_i)$:

$$
S''(x) = M_i\frac{x_{i+1}-x}{h_i} + M_{i+1}\frac{x-x_i}{h_i}
$$

Integrating twice and fixing the two constants by $S(x_i) = y_i$ and $S(x_{i+1}) = y_{i+1}$,

$$
S(x) = M_i\frac{(x_{i+1}-x)^3}{6h_i} + M_{i+1}\frac{(x-x_i)^3}{6h_i}
+ \left(\frac{y_i}{h_i} - \frac{M_ih_i}{6}\right)(x_{i+1}-x)
+ \left(\frac{y_{i+1}}{h_i} - \frac{M_{i+1}h_i}{6}\right)(x-x_i)
$$

Differentiating gives the one sided derivatives at $x_i$:

$$
S'(x_i^+) = \frac{y_{i+1}-y_i}{h_i} - \frac{h_i}{3}M_i - \frac{h_i}{6}M_{i+1}
$$
$$
S'(x_i^-) = \frac{y_i-y_{i-1}}{h_{i-1}} + \frac{h_{i-1}}{6}M_{i-1} + \frac{h_{i-1}}{3}M_i
$$

Setting them equal and multiplying by 6 gives exactly the lesson's equation

$$
h_{i-1}M_{i-1} + 2(h_{i-1}+h_i)M_i + h_iM_{i+1}
= 6\left(\frac{y_{i+1}-y_i}{h_i} - \frac{y_i-y_{i-1}}{h_{i-1}}\right)
$$

**Diagonal dominance.** Row $i$ has off diagonal absolute sum $h_{i-1} + h_i$ and diagonal
$2(h_{i-1}+h_i)$. Since every $h_j > 0$,

$$
|a_{ii}| = 2(h_{i-1}+h_i) > h_{i-1}+h_i = \sum_{j\ne i}|a_{ij}|
$$

with a factor of exactly 2 to spare, in **every** row, for **every** knot spacing. That is strict
dominance with a uniform margin, so lesson 17's theorem applies: the LU factorization exists, no
pivoting is needed, and Thomas is stable. Exercise 4.3 checks what the margin is worth in
practice.

### 2.2 The equally spaced eigenvalues, and $\kappa < 3$

With $h_i = h$ for all $i$, dividing every interior row by $h$ leaves the constant stencil
$[1, 4, 1]$. Let $T$ be the $(n-1)\times(n-1)$ tridiagonal matrix with 4 on the diagonal and 1 off
it, and try the vector $v^{(k)}_j = \sin(jk\pi/n)$ for $j = 1,\dots,n-1$.

$$
(Tv)_j = v_{j-1} + 4v_j + v_{j+1}
= \sin\frac{(j-1)k\pi}{n} + \sin\frac{(j+1)k\pi}{n} + 4\sin\frac{jk\pi}{n}
$$

The sum to product identity $\sin(A-B) + \sin(A+B) = 2\sin A\cos B$ turns the first two terms into
$2\cos(k\pi/n)\sin(jk\pi/n)$, so

$$
(Tv^{(k)})_j = \left(4 + 2\cos\frac{k\pi}{n}\right)v^{(k)}_j
$$

The boundary terms work out because $v^{(k)}_0 = \sin 0 = 0$ and $v^{(k)}_n = \sin(k\pi) = 0$, so
the missing entries are exactly zero and the eigenvector equation holds in the first and last rows
too. There are $n-1$ of these, they are linearly independent, so they are all the eigenvalues.

**The bound.** $\cos(k\pi/n) \in (-1, 1)$ strictly for $k = 1,\dots,n-1$, so every eigenvalue lies
strictly in $(2, 6)$. $T$ is symmetric, so

$$
\kappa_2(T) = \frac{\lambda_{\max}}{\lambda_{\min}}
= \frac{4 + 2\cos(\pi/n)}{4 + 2\cos((n-1)\pi/n)} < \frac{6}{2} = 3
$$

independently of $n$. The bound is approached but never reached: at $n = 1000$ the measured value
is 2.9999.

**Why this matters so much.** Global smoothness needs global coupling, and the fear is that the
coupling costs conditioning. It does not. The spline system is banded, dominant, perfectly
conditioned and $O(n)$ to solve, all at once, which is not true of the Vandermonde system that
lesson 44 measured at $10^{16}$.

### 2.3 The minimum curvature theorem

**Theorem.** Let $S$ be the natural cubic spline through $(x_i, y_i)$ and let $g$ be any $C^2$
function on $[x_0, x_n]$ with $g(x_i) = y_i$ for all $i$. Then

$$
\int_{x_0}^{x_n}(g'')^2 \ge \int_{x_0}^{x_n}(S'')^2
$$

with equality only when $g = S$.

**Proof.** Write $e = g - S$, so $e(x_i) = 0$ at every knot. Then

$$
\int (g'')^2 = \int (S'' + e'')^2 = \int (S'')^2 + 2\int S''e'' + \int (e'')^2
$$

The last term is non-negative, so it is enough to show the cross term vanishes. Integrate by parts
over the whole interval:

$$
\int_{x_0}^{x_n} S''e'' = \Big[S''e'\Big]_{x_0}^{x_n} - \int_{x_0}^{x_n}S'''e'
$$

The bracket is zero because $S''(x_0) = S''(x_n) = 0$, which is precisely the natural end
condition. For the remaining integral, $S'''$ is constant on each interval, say $c_i$ on
$[x_i, x_{i+1}]$, so

$$
\int_{x_0}^{x_n}S'''e' = \sum_i c_i\int_{x_i}^{x_{i+1}}e' = \sum_i c_i\big(e(x_{i+1}) - e(x_i)\big) = 0
$$

because $e$ vanishes at every knot. Hence the cross term is zero and

$$
\int (g'')^2 = \int (S'')^2 + \int (e'')^2 \ge \int (S'')^2
$$

Equality forces $e'' \equiv 0$, so $e$ is affine, and an affine function vanishing at two or more
knots is identically zero. So $g = S$.

**Where each end condition entered.** The bracket needed $S''=0$ at both ends, so the theorem is
about the **natural** spline specifically. The clamped spline satisfies a different minimum: among
$C^2$ interpolants with the same prescribed end slopes, it minimises the same integral, because
then $e'(x_0) = e'(x_n) = 0$ kills the bracket instead.

### 2.4 Which end conditions reproduce a cubic

**Clamped.** Let $f$ be a cubic. The clamped spline is the unique $C^2$ interpolant with
$S'(x_0) = f'(x_0)$ and $S'(x_n) = f'(x_n)$. But $f$ itself is a $C^2$ interpolant of its own
values with those slopes, and it is a cubic on every interval, so it is a cubic spline. By
uniqueness $S = f$.

**Not-a-knot.** The same argument: $f$ is a single cubic, so $f'''$ is continuous everywhere, in
particular at $x_1$ and $x_{n-1}$. So $f$ satisfies the not-a-knot conditions, and by uniqueness
$S = f$.

**Natural fails.** The natural spline needs $S''(x_0) = 0$. A cubic $f$ has $f''(x_0) = 0$ only by
accident. If $f''(x_0)\ne 0$ then $f$ is not a natural spline, so the natural spline is a
different function and the reproduction fails.

**Parabolic reproduces quadratics, not cubics.** $M_0 = M_1$ says $S''$ is constant on the first
interval, which is true of a quadratic and false of a general cubic.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import splines as sp

x = np.linspace(0.0, 1.0, 9)
probe = np.linspace(0.0, 1.0, 2001)
print(f"{'':>18}{'natural':>13}{'parabolic':>13}{'not-a-knot':>13}{'clamped':>13}")
for label, f, df in (("cubic  t**3 - 2t", lambda t: t ** 3 - 2 * t, lambda t: 3 * t * t - 2),
                     ("quadratic t(1-t)", lambda t: t * (1 - t), lambda t: 1 - 2 * t)):
    y = f(x)
    truth = f(probe)
    row = []
    for name in ("natural", "parabolic", "not-a-knot"):
        s = sp.BUILDERS[name](x, y)
        row.append(float(np.max(np.abs(np.atleast_1d(s(probe)) - truth))))
    c = sp.clamped(x, y, float(df(0.0)), float(df(1.0)))
    row.append(float(np.max(np.abs(np.atleast_1d(sp.evaluate(c, probe)) - truth))))
    print(f"{label:>18}" + "".join(f"{v:>13.2e}" for v in row))
```

| function, 8 intervals | natural | parabolic | not-a-knot | clamped |
|---|---|---|---|---|
| cubic $t^3-2t$ | 4.60e-3 | 4.54e-4 | 2.22e-16 | 2.22e-16 |
| quadratic $t(1-t)$ | 1.53e-3 | 5.55e-17 | 5.55e-17 | 5.55e-17 |

Exactly as predicted: clamped and not-a-knot reproduce the cubic to roundoff, parabolic
reproduces the quadratic only, and natural reproduces neither.

### 2.5 The $O(h^4)$ bound, and where natural loses it

**The clamped bound.** For $f \in C^4[a,b]$ and the clamped spline $S$ on knots of maximum spacing
$h$,

$$
\|f - S\|_\infty \le \frac{5}{384}h^4\|f^{(4)}\|_\infty
$$

with the derivative bounds $\|f'-S'\|_\infty \le \frac{1}{24}h^3\|f^{(4)}\|$ and
$\|f''-S''\|_\infty \le \frac38 h^2\|f^{(4)}\|$.

**The shape of the argument.** Two steps.

First, control the moments. The moment system reads $AM = 6d$ where $d_i$ is the second divided
difference of the data. By Taylor, $d_i = f''(x_i) + O(h^2)$, so $A(M - f''|_{\text{knots}})$ has
entries $O(h^2)$ times the row scale. The equally spaced argument of exercise 2.2 gives
$\|A^{-1}\|_\infty \le 1/2$ in the scaled system, because strict dominance with margin factor 2
bounds the inverse by $1/(|a_{ii}| - \sum_{j\ne i}|a_{ij}|)$. Hence
$\|M - f''\|_\infty = O(h^2)$.

Second, propagate. On one interval the difference $f - S$ vanishes at both ends, so it is bounded
by the interpolation error of $f$ by a cubic on those two points plus the error made by using $M$
instead of $f''$. The first is $\frac{5}{384}h^4\|f^{(4)}\|$ from the two point Hermite style
estimate, and the second is $O(h^2)\cdot O(h^2) = O(h^4)$, because the moment error multiplies a
quadratic bubble of size $h^2/8$.

**Where the end condition enters.** The bound $\|M - f''\|_\infty = O(h^2)$ needs the first and
last rows of the system to be as accurate as the interior rows.

- **Clamped** puts $S'(x_0) = f'(x_0)$ in row 0. Expanding, that row says
  $2h_0M_0 + h_0M_1 = 6(\frac{y_1-y_0}{h_0} - f'(x_0))$, whose right hand side is
  $h_0(3f''(x_0) + O(h^2))$. So row 0 is accurate to $O(h^2)$ like the rest.
- **Not-a-knot** puts a third divided difference condition in row 0, again accurate to $O(h)$ in
  the moments after the row scaling, which is enough.
- **Natural** puts $M_0 = 0$ in row 0. The true value is $f''(x_0)$, which is $O(1)$, not
  $O(h^2)$. So the input to the system has an $O(1)$ error in one component.

That $O(1)$ error does not stay put. The moment system is dominant, so it decays away from the
end, but it decays only geometrically with a ratio near $2-\sqrt3 \approx 0.268$ per knot, which
is a decay in **knot index**, not in $h$. Near the end it is still $O(1)$ in the moment, and an
$O(1)$ moment error times the $h^2/8$ bubble is an $O(h^2)$ error in $S$.

The measurement in exercise 4.1 shows exactly this split: natural is order 2 measured over the
outer 10 percent and order 4 measured in the middle.

### 3.1 The B-spline basis, against the moment formulation

The same spline written in a different basis. With $n+1$ data points there are $n+3$ cubic
B-splines on the open uniform knot vector, so $n+1$ collocation rows plus two end condition rows.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import bezier as bz, splines as sp


def bspline_collocation(x, eps_fraction=1e-4):
    """Interpolate at the knots in the cubic B-spline basis, with natural end rows.

    The two end rows are second derivatives, formed by a difference quotient, so they are scaled
    by 1/eps**2 relative to the collocation rows. That scaling is what the raw condition number
    measures, and row equilibration removes it.
    """
    x = np.asarray(x, dtype=float)
    n_ctrl = x.size + 2
    knots = bz.open_uniform_knots(n_ctrl, 3)
    lo, hi = float(x[0]), float(x[-1])
    top = float(knots[-1]) - 1e-12
    u = (x - lo) / (hi - lo) * (float(knots[-1]) - float(knots[0])) + float(knots[0])
    basis = lambda c, t: float(np.atleast_1d(bz.bspline_basis(c, 3, knots, t))[0])
    A = np.zeros((n_ctrl, n_ctrl))
    for r, t in enumerate(u):
        for c in range(n_ctrl):
            A[r, c] = basis(c, min(float(t), top))
    eps = float(eps_fraction) * (float(knots[-1]) - float(knots[0]))
    for row, t0 in ((x.size, float(knots[0]) + eps), (x.size + 1, top - eps)):
        for c in range(n_ctrl):
            A[row, c] = (basis(c, t0 + eps) - 2.0 * basis(c, t0) + basis(c, t0 - eps)) / eps ** 2
    return A


print(f"{'nodes':>7}{'raw kappa':>14}{'row scaled':>13}{'moment kappa':>15}")
for n in (5, 9, 17, 33, 65, 129):
    A = bspline_collocation(np.linspace(0.0, 1.0, n))
    scaled = A / np.abs(A).sum(axis=1)[:, None]
    moment = sp.equispaced_system(n - 1, 1.0 / (n - 1))["condition_number"]
    print(f"{n:>7}{float(np.linalg.cond(A)):>14.4g}"
          f"{float(np.linalg.cond(scaled)):>13.4g}{moment:>15.4f}")
```

| nodes | raw kappa | row scaled | moment kappa |
|---|---|---|---|
| 5 | 429.1 | 2.672 | 2.0938 |
| 9 | 2010 | 3.021 | 2.7171 |
| 17 | 8448 | 3.158 | 2.9246 |
| 33 | 3.419e4 | 3.198 | 2.9808 |
| 65 | 1.367e5 | 3.209 | 2.9952 |
| 129 | 5.433e5 | 3.213 | 2.9988 |

**The raw number grows like $n^2$ and means nothing.** The ratios of consecutive raw values are
4.68, 4.20, 4.05, 4.00, 3.97, which is $n^2$ growth, and it comes entirely from the two end rows
being divided by $\epsilon^2$ while the collocation rows have entries of size 1. Change
`eps_fraction` and the raw number changes with it, which is the tell.

**After row equilibration the two formulations agree.** The B-spline collocation matrix
equilibrates to 3.21 and the moment matrix to 3.00, both bounded independently of $n$, both
converging as $n$ grows. Neither basis is better conditioned than the other in any way that
matters.

**So why use B-splines at all?** Not for conditioning. For the reasons lesson 52 gives: local
support, so a coefficient change is local; a convex hull bound, so the curve can be bounded
without evaluating; and the ability to represent the same object in more than one dimension and
with repeated knots. Conditioning is the one thing the two bases do not disagree about.

### 3.2 The smoothing spline

Give up exact interpolation and minimise

$$
\sum_i (f_i - y_i)^2 + \lambda\int (g'')^2
$$

over $C^2$ functions. The minimiser is a cubic spline whose values at the knots solve
$(I + \lambda QR^{-1}Q^{T})f = y$, where $R$ is the moment matrix of the bending energy and $Q$
is the second difference operator.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import splines as sp


def smoothing_spline_values(x, y, lam):
    """Reinsch's form. Returns the fitted values at the knots; run them through any spline
    builder to get the curve. lam = 0 gives interpolation, lam -> infinity gives the least
    squares straight line, because a line is the only curve with zero bending energy."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 4:
        raise ValueError(f"need at least four points, got {n}")
    h = np.diff(x)
    Q = np.zeros((n, n - 2))
    R = np.zeros((n - 2, n - 2))
    for j in range(n - 2):
        Q[j, j] = 1.0 / h[j]
        Q[j + 1, j] = -1.0 / h[j] - 1.0 / h[j + 1]
        Q[j + 2, j] = 1.0 / h[j + 1]
        R[j, j] = (h[j] + h[j + 1]) / 3.0
        if j < n - 3:
            R[j, j + 1] = R[j + 1, j] = h[j + 1] / 6.0
    A = np.eye(n) + float(lam) * Q @ np.linalg.solve(R, Q.T)
    return np.linalg.solve(A, y)


gen = np.random.default_rng(4)
n = 25
x = np.linspace(0.0, 1.0, n)
clean = np.sin(4.0 * x)
noisy = clean + 0.08 * gen.standard_normal(n)
print(f"{'lambda':>11}{'fit to noisy':>15}{'fit to CLEAN':>15}{'roughness':>13}")
for lam in (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1e1):
    fitted = smoothing_spline_values(x, noisy, lam)
    print(f"{lam:>11.0e}{float(np.sqrt(np.mean((fitted - noisy) ** 2))):>15.5f}"
          f"{float(np.sqrt(np.mean((fitted - clean) ** 2))):>15.5f}"
          f"{sp.bending_energy(sp.natural(x, fitted)):>13.3f}")
```

| $\lambda$ | rms to the noisy data | rms to the CLEAN function | bending energy |
|---|---|---|---|
| 0 | 0.00000 | 0.08559 | 58810.9 |
| 1e-6 | 0.02279 | 0.06921 | 25353.8 |
| 1e-4 | 0.07140 | 0.03478 | 247.9 |
| 1e-3 | 0.08123 | **0.02031** | 108.5 |
| 1e-2 | 0.11825 | 0.07846 | 62.3 |
| 1e-1 | 0.29621 | 0.28346 | 10.5 |
| 1e1 | 0.43962 | 0.43196 | 0.003 |

**The trade, stated exactly.** The middle column is the honest one, because it measures against
the function the noise was added to rather than against the noise.

- At $\lambda = 0$ the fit reproduces the data exactly, rms 0.00000, and is at its **worst**
  against the truth, 0.0856, which is the noise level 0.08. Interpolating noise reproduces noise.
- The best fit to the truth is at $\lambda = 10^{-3}$, rms 0.0203, a factor of 4.2 better than
  interpolation. At that point the residual to the data is 0.0812, which is right at the noise
  level, exactly where it should be.
- Past that the penalty dominates and the fit flattens toward a straight line. At $\lambda = 10$
  the bending energy is 0.003, essentially zero, and the error is 0.43, which is just the
  amplitude of the sine.

**The rule that falls out.** Choose $\lambda$ so the residual matches the known noise level. That
is the same idea as Morozov's discrepancy principle from Part 5's regularisation, and it is the
same picture: too little regularisation fits the noise, too much throws away the signal, and the
optimum sits where the residual equals the noise.

Note also that the bending energy falls by four orders between $\lambda = 0$ and
$\lambda = 10^{-4}$ while the fit to the truth **improves**. The interpolating spline is paying
59000 units of curvature to chase noise.

### 3.3 The periodic spline

For data that wraps, the two free conditions become $S'(x_0) = S'(x_n)$ and $S''(x_0) = S''(x_n)$,
with $y_0 = y_n$ assumed. That makes the moment matrix cyclic: tridiagonal with two corner
entries.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import splines as sp


def periodic_spline(x, y, tol=1e-10):
    """Wrap the end conditions. The unknowns are M_0 ... M_{n-1}, with M_n = M_0, and the matrix
    is tridiagonal plus the two corners, which is the cyclic case of lesson 21."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if abs(float(y[0] - y[-1])) > tol * max(float(np.max(np.abs(y))), 1.0):
        raise ValueError(f"periodic data must satisfy y[0] == y[-1], got "
                         f"{float(y[0])} and {float(y[-1])}")
    n = x.size - 1
    h = np.diff(x)
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    for i in range(n):
        hm, hi = h[i - 1], h[i]
        A[i, (i - 1) % n] += hm / 6.0
        A[i, i] += (hm + hi) / 3.0
        A[i, (i + 1) % n] += hi / 6.0
        y_prev = y[i - 1] if i > 0 else y[n - 1]
        rhs[i] = (y[i + 1] - y[i]) / hi - (y[i] - y_prev) / hm
    m = np.linalg.solve(A, rhs)
    moments = np.empty(n + 1)
    moments[:n] = m
    moments[n] = m[0]
    return sp.Spline(x=x, y=y, moments=moments, end_condition="periodic")


probe = np.linspace(0.0, 2.0 * np.pi, 4001)
truth = np.sin(probe)
print(f"{'nodes':>7}{'periodic error':>16}{'jump S':>10}{'jump S1':>10}{'jump S2':>10}"
      f"{'natural error':>16}")
for n in (9, 17, 33, 65):
    xs = np.linspace(0.0, 2.0 * np.pi, n)
    ys = np.sin(xs)
    S = periodic_spline(xs, ys)
    e = float(np.max(np.abs(np.atleast_1d(sp.evaluate(S, probe)) - truth)))
    jumps = [abs(float(sp.evaluate(S, xs[:1], order=k)) - float(sp.evaluate(S, xs[-1:], order=k)))
             for k in (0, 1, 2)]
    en = float(np.max(np.abs(np.atleast_1d(sp.natural(xs, ys)(probe)) - truth)))
    print(f"{n:>7}{e:>16.3e}" + "".join(f"{v:>10.1e}" for v in jumps) + f"{en:>16.3e}")
```

| nodes | periodic error | jump in $S$ | jump in $S'$ | jump in $S''$ | natural error |
|---|---|---|---|---|---|
| 9 | 1.066e-3 | 2.4e-16 | 4.4e-16 | 0.0 | 1.066e-3 |
| 17 | 6.312e-5 | 2.4e-16 | 5.6e-16 | 0.0 | 6.312e-5 |
| 33 | 3.889e-6 | 2.4e-16 | 1.3e-15 | 1.6e-30 | 3.889e-6 |
| 65 | 2.422e-7 | 2.4e-16 | 2.3e-15 | 6.3e-30 | 2.422e-7 |

**The wrap is exact.** All three jumps across the seam are at roundoff, and the $S''$ jump is
exactly zero by construction, since $M_n$ is set equal to $M_0$ rather than solved for.

**The order is 4.** The ratios are 16.9, 16.2, 16.1, converging to 16 per halving of $h$.

**Natural gives the identical answer here, and that is not a coincidence.** For $\sin$ on
$[0, 2\pi]$ the true $f''$ is $-\sin$, which vanishes at both ends, so the natural condition
happens to be exactly right and the natural spline **is** the periodic spline for this data. The
test is therefore not distinguishing the two, and to see the difference the data has to have
non-zero curvature at the seam. That is the same trap as in exercise 4.1 and worth stating rather
than reporting the tie as a result.

**What periodic buys elsewhere.** Closed curves. A font outline, a cam profile, a closed contour:
anything where the seam is an artefact of where you started numbering, and a kink there would be
visible.

### 4.1 Convergence order by end condition

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import splines as sp

cases = [("exp, [0,1]", np.exp, np.exp, 0.0, 1.0),
         ("sin, [0,pi]", np.sin, np.cos, 0.0, np.pi),
         ("t**3 - t, [-1,1]", lambda t: t ** 3 - t, lambda t: 3 * t * t - 1, -1.0, 1.0),
         ("t**5 - t**3, [-1,1]", lambda t: t ** 5 - t ** 3,
          lambda t: 5 * t ** 4 - 3 * t * t, -1.0, 1.0)]
names = ("natural", "clamped", "parabolic", "not-a-knot")
print(f"{'function':>21}" + "".join(f"{k:>13}" for k in names))
for label, f, df, lo, hi in cases:
    rep = sp.convergence_by_end_condition(f, df=df, n_values=(16, 32, 64), lo=lo, hi=hi)
    ends = rep["order_at_the_ends"]
    mid = rep["order_in_the_middle"]
    print(f"{label:>21}" + "".join(f"{ends[k]:>13.3f}" for k in names) + "   at the ends")
    print(f"{'':>21}" + "".join(f"{mid[k]:>13.3f}" for k in names) + "   in the middle")
```

**Final error at 64 intervals:**

| function | natural | clamped | parabolic | not-a-knot |
|---|---|---|---|---|
| $\exp$, $[0,1]$ | 3.258e-5 | **4.208e-10** | 3.980e-7 | 4.505e-9 |
| $\sin$, $[0,\pi]$ | **1.512e-8** | **1.512e-8** | 4.578e-6 | **1.512e-8** |
| $\sin$, $[0,2\pi]$ | **2.422e-7** | **2.422e-7** | 3.659e-5 | 2.651e-7 |
| $t^3$, $[0,1]$ | 7.191e-5 | **1.11e-16** | 8.861e-7 | **1.11e-16** |
| $t(1-t)$, $[0,1]$ | 2.397e-5 | **8.33e-17** | **8.33e-17** | **8.33e-17** |
| $t^3-t$, $[-1,1]$ | 2.876e-4 | **8.33e-17** | 7.089e-6 | **8.33e-17** |
| $t^5-t^3$, $[-1,1]$ | 6.704e-4 | **2.965e-7** | 6.134e-5 | 3.130e-6 |

**Fitted order over the outer 10 percent of the interval:**

| function | natural | clamped | parabolic | not-a-knot |
|---|---|---|---|---|
| $\exp$, $[0,1]$ | 2.000 | 3.994 | 2.981 | 3.965 |
| $\sin$, $[0,\pi]$ | 4.026 | 4.070 | 2.997 | 4.995 |
| $\sin$, $[0,2\pi]$ | 4.051 | 4.096 | 2.987 | 4.981 |
| $t^5-t^3$, $[-1,1]$ | 1.988 | 3.989 | 2.910 | 3.924 |

**The four orders in the table of section 4 are confirmed**, but only on functions with
$f''\ne0$ at the ends: 2 for natural, 3 for parabolic, 4 for clamped and not-a-knot.

**The function where natural is the best choice.** $\sin$ on $[0,\pi]$, and on $[0,2\pi]$, and
$\sin(\pi t)$ on $[0,1]$. On all three, natural jumps from order 2 to order 4 and matches the
best error in the table exactly.

The reason is visible immediately: $f'' = -\sin$ vanishes at both endpoints there, so
$S''(x_0)=S''(x_n)=0$ is not an assumption, it is a fact about $f$. The natural condition is then
free information of the same quality as the clamped derivative, and it costs nothing to supply.

**Is it strictly best, or only tied?** Only tied. Natural matches clamped and not-a-knot to every
digit reported and never beats them. That is what should be expected: the best a correct end
condition can do is stop losing, and there is nothing left for it to win. Reporting "natural wins"
from a `min` over a tied row would have been an artefact of tie breaking, so the honest statement
is that natural **stops being wrong**.

**The interior story, which the overall number hides.** The natural spline's $O(h^2)$ is purely
an end effect. Splitting the interval and refining further shows it:

| intervals | error over the outer 10 percent | order | error over the middle 80 percent | order |
|---|---|---|---|---|
| 16 | 5.210e-4 |  | 1.106e-4 |  |
| 32 | 1.303e-4 | 1.999 | 2.512e-6 | 5.460 |
| 64 | 3.258e-5 | 2.000 | 1.168e-8 | 7.748 |
| 128 | 8.145e-6 | 2.000 | 2.401e-11 | 8.927 |
| 256 | 2.036e-6 | 2.000 | 1.486e-12 | 4.014 |
| 512 | 5.090e-7 | 2.000 | 9.326e-14 | 3.994 |
| 1024 | 1.273e-7 | 2.000 | 6.217e-15 | 3.907 |

The end column is 2.000 at every single refinement. The middle column settles to 4 only from
$n = 256$ onward, and the apparent orders of 5.5 to 8.9 before that are **not** superconvergence.
They are the boundary pollution still decaying: exercise 2.5's argument says the $O(1)$ moment
error at the end decays by a factor near $2-\sqrt3 \approx 0.268$ per knot, so at the 10 percent
mark it is $0.268^{0.1n}$, and while that is still larger than the interior $O(h^4)$ it dominates
the middle column and inflates the fitted order. Once it drops below, the true interior order 4
appears.

That is also why quoting one number for the whole interval is misleading, and why
`order_at_the_ends` and `order_in_the_middle` are reported separately.

**A trap worth naming.** $t(1-t)$ and $t^3-t$ give clamped, parabolic and not-a-knot errors of
$8\times10^{-17}$, and any fitted order on those numbers is noise: the measured values were
$-0.500$ and $-0.292$. Exact reproduction has no order. Only rows whose errors are actually
decreasing belong in an order table.

### 4.2 Knot distribution on a localised feature

$f(t) = 1/(1+400t^2)$ on $[-1,1]$: a spike of width about $1/20$ at the origin, flat everywhere
else.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import splines as sp, chebyshev as cb

f = lambda t: 1.0 / (1.0 + 400.0 * t * t)
probe = np.linspace(-1.0, 1.0, 4001)
truth = f(probe)
print(f"{'knots':>7}{'uniform':>13}{'Chebyshev':>13}{'adaptive':>13}")
for n in (9, 17, 33, 65):
    xu = np.linspace(-1.0, 1.0, n)
    xc = np.sort(cb.extrema_nodes(n))
    u = np.linspace(0.0, 1.0, n)
    xa = np.sign(2 * u - 1) * np.abs(2 * u - 1) ** 2.5
    row = []
    for xs in (xu, xc, xa):
        s = sp.not_a_knot(xs, f(xs))
        row.append(float(np.max(np.abs(np.atleast_1d(s(probe)) - truth))))
    print(f"{n:>7}" + "".join(f"{v:>13.3e}" for v in row))
```

| knots | uniform | Chebyshev | adaptive |
|---|---|---|---|
| 9 | 5.460e-1 | 6.746e-1 | **1.928e-2** |
| 17 | 2.755e-1 | 4.525e-1 | **1.696e-2** |
| 33 | 5.607e-2 | 1.828e-1 | **1.213e-3** |
| 65 | 3.744e-3 | 1.992e-2 | **5.370e-5** |

**Adaptive wins by two orders, at every size.** At 65 knots it is 70 times better than uniform and
370 times better than Chebyshev.

**Chebyshev is the worst of the three, and that is the interesting result.** Chebyshev nodes
cluster at the **ends** of the interval. That is exactly right for polynomial interpolation, where
the node polynomial's growth is an end effect and clustering there controls the Lebesgue constant.
It is exactly wrong for splines, where there is no node polynomial, no Runge phenomenon, and no
end effect to fight. What a spline needs is knots where the function is hard, and here the
function is hard in the middle, so Chebyshev is spending its knots in precisely the region that
needs them least.

**The general rule.** For a fixed degree piecewise method the error on interval $i$ is
$\sim h_i^4|f^{(4)}|$ on that interval, so equalising the error means

$$
h_i \propto |f^{(4)}(x_i)|^{-1/4}
$$

which is the equidistribution principle. It is a statement about $f$, not about the interval, and
it has nothing in common with the Chebyshev clustering rule beyond the words "put nodes where they
are needed".

The adaptive family used here, $x = \operatorname{sign}(2u-1)|2u-1|^{2.5}$, clusters at the origin
and is a crude stand in for equidistribution. Even that crude version beats both fixed rules by
two orders, which says the gain comes from where the knots go, not from getting the exponent right.

### 4.3 Conditioning against the ratio of intervals

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import splines as sp


def graded_knots(n_intervals, ratio):
    """Geometrically graded knots on [0,1], with the last interval `ratio` times the first."""
    g = np.geomspace(1.0, float(ratio), int(n_intervals))
    x = np.concatenate(([0.0], np.cumsum(g)))
    return x / x[-1]


def moment_matrix(x):
    h = np.diff(np.asarray(x, dtype=float))
    A = np.zeros((x.size, x.size))
    A[0, 0] = A[-1, -1] = 1.0
    for i in range(1, x.size - 1):
        A[i, i - 1] = h[i - 1] / 6.0
        A[i, i] = (h[i - 1] + h[i]) / 3.0
        A[i, i + 1] = h[i] / 6.0
    return A


f = lambda t: np.exp(2.0 * t)
probe = np.linspace(0.0, 1.0, 20001)
truth = f(probe)
print(f"{'ratio':>10}{'raw kappa':>13}{'row scaled':>13}{'largest h':>13}"
      f"{'knot error':>13}{'sup error':>13}")
for ratio in (1.0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14):
    xs = graded_knots(12, ratio)
    A = moment_matrix(xs)
    scaled = A / np.abs(A).sum(axis=1)[:, None]
    S = sp.natural(xs, f(xs))
    sup = float(np.max(np.abs(np.atleast_1d(sp.evaluate(S, probe)) - truth)))
    print(f"{ratio:>10.0e}{float(np.linalg.cond(A)):>13.4g}"
          f"{float(np.linalg.cond(scaled)):>13.4g}{float(np.max(np.diff(xs))):>13.3e}"
          f"{abs(sp.interpolates(S)):>13.3e}{sup:>13.3e}")
```

| ratio | raw kappa | row scaled | largest $h$ | error at the knots | sup error |
|---|---|---|---|---|---|
| 1 | 34.82 | 2.995 | 8.33e-2 | 1.2e-16 | 1.004e-2 |
| 1e2 | 418.4 | 3.023 | 3.44e-1 | 3.0e-17 | 1.530e-1 |
| 1e4 | 1.779e4 | 3.048 | 5.67e-1 | 6.0e-17 | 3.638e-1 |
| 1e6 | 1.000e6 | 3.067 | 7.15e-1 | 1.2e-16 | 5.223e-1 |
| 1e8 | 6.145e7 | 3.082 | 8.13e-1 | 3.0e-17 | 6.272e-1 |
| 1e10 | 3.912e9 | 3.092 | 8.77e-1 | 1.2e-16 | 6.948e-1 |
| 1e12 | 2.530e11 | 3.100 | 9.19e-1 | 1.2e-16 | 7.382e-1 |
| 1e14 | 1.649e13 | 3.105 | 9.47e-1 | 3.0e-17 | 7.662e-1 |

**Three separate things are happening, and only one of them is real.**

**The raw condition number tracks the ratio, and it is a scaling artefact.** Row $i$ of the moment
matrix has entries of size $h_{i-1} + h_i$, so with 14 orders of variation in $h$ the rows differ
in scale by 14 orders. Row equilibration removes it completely: the scaled condition number is
2.995 at ratio 1 and 3.105 at ratio $10^{14}$, a change of 3.7 percent across fourteen orders of
grading. The bound $\kappa < 3$ of exercise 2.2 was proved for equal spacing; the measurement says
it survives arbitrary grading once the rows are scaled.

**The solve never loses accuracy.** The spline interpolates its own data to $10^{-16}$ at every
ratio, and the relative residual of the moment solve stays at $10^{-16}$ too. This is strict
diagonal dominance doing exactly what lesson 17 says it does, with a margin factor of 2 that does
not depend on the spacing at all.

**What actually degrades is approximation, and it has nothing to do with the linear algebra.**
The sup error rises from $1.0\times10^{-2}$ to $7.7\times10^{-1}$. The cause is in the fourth
column: grading with a fixed number of knots means the largest interval swallows the domain. At
ratio $10^{14}$ one interval covers 95 percent of $[0,1]$, and no cubic on one interval
approximates $e^{2t}$ across 95 percent of the range. The $\frac{5}{384}h^4\|f^{(4)}\|$ bound at
$h = 0.813$ is 0.67, and the measured error is 0.63, so the bound is tracking it.

**The control experiment that settles it.** Put the ratio between two **adjacent** intervals
instead, leaving the rest uniform, so one knot is very close to its neighbour but the largest
interval stays at $1/11$:

| ratio between adjacent intervals | kappa | knot error | sup error |
|---|---|---|---|
| 1 | 31.72 | 6.0e-17 | 1.194e-2 |
| 1e4 | 43.97 | 1.2e-16 | 1.194e-2 |
| 1e8 | 43.97 | 1.2e-16 | 1.194e-2 |
| 1e12 | 43.97 | 6.0e-17 | 1.194e-2 |

Nothing happens. The condition number saturates at 44, and the sup error does not move in the
fourth digit across twelve orders of ratio.

**So the answer to "where do non-uniform knots become a problem" is: they do not, numerically.**
Two knots can be $10^{12}$ times closer together than their neighbours and the moment system does
not notice. What matters is the **largest** interval, because that is what sets the approximation
error, and that is a question about where you put the knots, which is exercise 4.2, not a question
about conditioning.

### 5.1 Why cubic and not quintic

Quintic splines are real, they give $C^4$ and $O(h^6)$, and they are almost never used. Four
reasons.

**Oscillation.** A degree 5 piece has more freedom than the data on that piece constrains, and the
extra freedom shows up as wiggle. Higher degree piecewise polynomials inherit a local version of
the Runge phenomenon: not divergence, since the pieces are short, but visible overshoot near
sharp features. Cubic is the lowest degree that is smooth enough to look right, and going higher
costs shape.

**The end condition problem gets worse, fast.** Cubic has a deficit of 2. Quintic has a deficit of
4, septic 6. Every one of them has to be guessed, every guess is wrong in the same $O(1)$ way the
natural condition is, and exercise 2.5's argument says each wrong condition costs orders near the
boundary. Chasing $O(h^6)$ while introducing four more chances to be wrong at the ends is a poor
trade, and in practice quintic splines often measure worse than cubic on short data.

**More smoothness than anyone needs.** $C^2$ means continuous curvature, which is the threshold
for a surface to have no visible seam under specular light and for a cam not to jerk. $C^4$ has no
comparable physical meaning. The extra derivatives are paid for and unused.

**The bound stops applying.** $O(h^6)$ needs $f \in C^6$. Real data is rarely that smooth, and
when it is not, the quintic spline converges at the rate the smoothness allows, which is the same
rate the cubic gets, at higher cost and with more oscillation. Lesson 50's exercise 4.1 measured
this effect for piecewise linear: the order is $\min(p, \text{smoothness})$, and raising $p$ past
the smoothness buys nothing.

**Where the higher degrees do get used.** Quintic Hermite in trajectory planning, where continuous
acceleration and jerk are genuinely required and the data is a small number of waypoints you
control. That is a different problem from interpolating data someone handed you.

### 5.2 The general theory, and its other appearances

**The general object.** Minimise a quadratic functional subject to linear constraints:

$$
\min_g \tfrac12 \langle Lg, Lg\rangle \quad\text{subject to}\quad \langle \phi_i, g\rangle = y_i
$$

in a Hilbert space. This is a **least norm problem** in the semi-norm $\|Lg\|$, and its solution
theory is the projection theorem: the minimiser is the unique element of the constraint set
orthogonal, in the $L$ semi-inner product, to the null space of the constraints. For splines the
setting is the Sobolev space $H^2$, $L = d^2/dx^2$, and $\phi_i$ is point evaluation at $x_i$.

The Lagrange conditions are what produce the spline's structure. The stationarity condition
$L^*L g = \sum_i \lambda_i\phi_i$ says $g^{(4)}$ is a sum of point masses, so $g^{(4)} = 0$
between knots, so $g$ is a cubic on each interval. The transversality conditions at the free ends
give $g'' = 0$ there, which is the natural end condition, dropping out of the variational problem
rather than being chosen. The smoothness $C^2$ comes from the two integrations by parts in
exercise 2.3.

That is the general theory: **splines are the reproducing kernel Hilbert space solution for
$H^2$ with point evaluation constraints**, and the whole shape of the answer, piecewise cubic,
$C^2$, natural at the ends, is read off from the variational problem rather than assumed.

**Two other objects in this course that are the same kind of thing.**

**Minimum norm least squares, Part 5.** $\min\|x\|_2$ subject to $Ax = b$ on the consistent set,
solved by the pseudoinverse $A^+b$. Same structure exactly: quadratic objective, linear
constraints, unique minimiser characterised by orthogonality to the null space. The spline is the
infinite dimensional version with $\|\cdot\|$ replaced by $\|g''\|$.

**Tikhonov regularisation, Part 5.** $\min \|Ax-b\|^2 + \lambda\|Lx\|^2$. This is the smoothing
spline of exercise 3.2 written in finite dimensions, with the same $\lambda$ playing the same
role, the same trade between fidelity and smoothness, and the same discrepancy principle for
choosing $\lambda$. The smoothing spline is not analogous to Tikhonov, it **is** Tikhonov in
$H^2$.

A third, if the count is allowed to grow: the Rayleigh quotient minimisation of Part 6, which is
the same theory with the constraint set a sphere rather than an affine subspace, and which is why
the answer there is an eigenvector rather than a projection.

### 5.3 Splines in more than one dimension

**Tensor product splines.** On a grid $x_i \times y_j$ with values $z_{ij}$, spline in one
direction, then spline the results in the other. Lesson 53's exercise 3.2 measures the result:
order 4 in each direction, order 4 overall, and the answer independent of which direction goes
first, to $1.3\times10^{-15}$ on an unequal grid.

The basis is the set of products $B_i(x)B_j(y)$, so the coefficient count is $n_x n_y$ and the
solve separates into $n_y$ solves of size $n_x$ plus $n_x$ solves of size $n_y$, each of them the
tridiagonal system of section 2. That separability is the entire advantage, and it is also the
entire restriction.

**What it requires.** A full grid. Not scattered points, not a grid with holes, not a grid that
is fine in one region and coarse in another. The separation of the solve is a statement about the
index structure of the data, and it fails as soon as the data does not factor.

**Thin plate splines.** Drop the grid. Minimise the two dimensional bending energy

$$
\int\int \left(g_{xx}^2 + 2g_{xy}^2 + g_{yy}^2\right)dx\,dy
$$

subject to $g(p_i) = z_i$ at scattered points $p_i$. The same variational theory as exercise 5.2
gives the answer in closed form:

$$
g(p) = \sum_i \lambda_i\,\phi(\|p - p_i\|) + a + b\cdot p, \qquad \phi(r) = r^2\log r
$$

a radial basis expansion plus an affine term, with the $\lambda_i$ and the affine coefficients
solving a linear system built from the pairwise distances. The name is physical, as with the
cubic spline: it is the shape of a thin elastic plate pinned at the data points.

**The relation to Mairhuber.** Mairhuber's theorem, lesson 53 section 5, says that for any
**fixed** basis of $m$ continuous functions on a two dimensional domain there is a set of $m$
points making the interpolation matrix singular. So no fixed basis works for all scattered point
sets in two dimensions.

The thin plate spline evades this, and it is worth being exact about how. Its basis is
$\phi(\|p - p_i\|)$, which **depends on the data points**. Move the points and the basis moves
with them. Mairhuber's hypothesis is that the basis is fixed in advance, so a data dependent basis
is outside the theorem's scope, not a counterexample to it.

That evasion is not free. The theorem is telling you that no method can be simultaneously fixed
basis, unisolvent for all point sets, and in more than one dimension. Thin plate splines give up
the first, so the matrix has to be built and factorised for each new point set, at $O(m^3)$ with
no banded structure, and the conditioning depends on the point configuration. Lesson 53's exercise
4.3 measures that conditioning reaching $10^{18}$ where the tensor product spline stays at 3.

The trade in one line: **the grid buys you separability, bandedness and a condition number of 3;
scattered data costs you all three, and Mairhuber says you cannot buy them back with a cleverer
basis.**

---

## Lesson 52, Bezier and B-spline Curves

### 1.1 What passing through the points would have cost

**What is given up.** Interpolation. The curve does not reproduce $P_1, \dots, P_{n-1}$, so if the
control points came from a measurement, the curve is not a fit to that measurement. Bezier curves
are the wrong tool for data.

**What it buys, one property each.**

**The convex hull property.** Because the curve is a convex combination of the control points, it
never leaves their hull. An interpolant cannot make that promise: lesson 46's Runge measurement
had the curve reaching 334 through data bounded by 1. Interpolation forces the curve to visit
points, and visiting points is exactly what makes it overshoot between them.

**Variation diminishing.** A straight line crosses the curve no more often than it crosses the
control polygon. So the curve cannot wiggle more than its controls suggest. Again this is
incompatible with interpolation, since forcing a curve through $n$ prescribed points forces at
least $n-1$ sign changes in the deviation from any line through them.

**Predictable steering.** Moving $P_i$ moves the curve in the same direction, by a known amount
scaled by $B_{i,n}(t)$, which is non-negative everywhere. With an interpolating polynomial the
response is $\ell_i(t)$, which alternates in sign and grows with the Lebesgue constant, so pulling
one data point up pushes the curve down somewhere else. Lesson 47 measured that Lebesgue constant
at $1.68^n$ for equal nodes. A designer cannot work with a control that sometimes does the
opposite of what it says.

Those three are the same fact three ways: **non-negative weights**. Interpolation cannot have
them, because a basis that reproduces arbitrary data at $n$ points has to be able to subtract.

### 1.2 A guarantee available before evaluating anywhere

The hull bound needs only the control points, so it costs $O(n)$ and no curve evaluation at all.

**Collision culling.** Two curves cannot intersect if their hulls do not. Comparing two boxes is a
handful of comparisons, against an actual intersection test that needs subdivision and root
finding. In a scene with thousands of curves almost every pair is rejected by the box test, and
only the survivors cost anything.

**Clipping and tile rejection.** A rasteriser divides the screen into tiles and asks which curves
touch which tile. The hull answers it without evaluating the curve. A glyph at small size touches
a handful of tiles and is discarded from the rest for free.

**Adaptive subdivision with a stopping rule.** The distance from the curve to the chord is bounded
by the distance from the **control points** to the chord, which is exercise 4.2's flatness
measure. So a renderer knows when to stop subdividing without ever measuring the curve, only its
controls.

**Ray tracing and bounding volume hierarchies.** The hull is the leaf bound in the hierarchy, and
the whole structure is built from control points before any geometry is evaluated.

The pattern in all four: a cheap, conservative, always valid bound turns an expensive exact test
into a rare one. That is worth more in practice than a tighter bound that costs an evaluation.

### 1.3 What the designer is using, and what to change it to

**They are using a single high degree Bezier curve.** With 30 control points the curve is degree
29, and every Bernstein weight $B_{i,29}(t)$ is strictly positive on $(0,1)$. So every control
point genuinely affects every point of the curve, and the lesson's measurement puts the affected
fraction at **89 percent** for a 30 point curve, the remainder being only the parts where the
displacement falls below the reporting threshold.

**Change it to a cubic B-spline with the same 30 control points.** The Cox-de Boor recursion makes
$N_{i,3}$ nonzero on only 4 knot spans, so one control point affects $4/(30-3) = 14.8$ percent of
the curve, and that fraction keeps falling as the curve grows.

**Or to a chain of cubic Bezier segments,** which is what a font format does. Ten cubic segments
share their end control points, and moving an interior control point of one segment affects that
segment only. This is the same object as the B-spline, written differently: exercise 3.2's knot
insertion converts between the two exactly.

**What is not the fix.** Keeping the single curve and being careful. The problem is not the
designer's technique, it is that the basis has global support, and no amount of care changes a
basis function's support.

### 2.1 Non-negativity, partition of unity, and the hull

**Non-negative.** $B_{i,n}(t) = \binom ni t^i(1-t)^{n-i}$. On $[0,1]$ both $t \ge 0$ and
$1-t \ge 0$, and $\binom ni > 0$, so the product is a product of non-negative numbers.

**Sum to one.** By the binomial theorem,

$$
\sum_{i=0}^n\binom ni t^i(1-t)^{n-i} = \big(t + (1-t)\big)^n = 1^n = 1
$$

for every $t$, including outside $[0,1]$. That the sum is 1 everywhere and the non-negativity
holds only on $[0,1]$ is exactly why $[0,1]$ is the curve's domain.

**The convex hull property.** A point is in the convex hull of $\{P_i\}$ exactly when it is
$\sum\lambda_iP_i$ with $\lambda_i \ge 0$ and $\sum\lambda_i = 1$. The curve value
$B(t) = \sum_iB_{i,n}(t)P_i$ has coefficients that are non-negative and sum to 1, so it is such a
combination, for every $t \in [0,1]$. Hence the whole curve lies in the hull.

**Two consequences worth naming.** The bound is dimension free: the argument never mentioned
whether $P_i \in \mathbb R^2$ or $\mathbb R^{100}$. And it is exact at the ends, since
$B_{0,n}(0) = 1$ and all others vanish, so $B(0) = P_0$ and likewise $B(1) = P_n$.

### 2.2 de Casteljau evaluates the Bernstein sum

**The algorithm.** Set $P^{(0)}_i = P_i$ and

$$
P^{(r)}_i = (1-t)P^{(r-1)}_i + tP^{(r-1)}_{i+1}, \qquad r = 1,\dots,n,\quad i = 0,\dots,n-r
$$

The claim is $P^{(n)}_0 = \sum_{i=0}^nB_{i,n}(t)P_i$.

**Proof, by induction on $r$.** Claim: $P^{(r)}_i = \sum_{j=0}^{r}B_{j,r}(t)P_{i+j}$.

*Base $r=0$.* $B_{0,0} = 1$, so the right hand side is $P_i$, which is $P^{(0)}_i$.

*Step.* Assume it for $r-1$. Then

$$
P^{(r)}_i = (1-t)\sum_{j=0}^{r-1}B_{j,r-1}P_{i+j} + t\sum_{j=0}^{r-1}B_{j,r-1}P_{i+1+j}
$$

Shift the index in the second sum by writing $j' = j+1$, so it runs $j' = 1,\dots,r$:

$$
P^{(r)}_i = \sum_{j=0}^{r}\Big[(1-t)B_{j,r-1}(t) + tB_{j-1,r-1}(t)\Big]P_{i+j}
$$

with $B_{-1,r-1} = B_{r,r-1} = 0$ by convention. The bracket is

$$
(1-t)\binom{r-1}{j}t^j(1-t)^{r-1-j} + t\binom{r-1}{j-1}t^{j-1}(1-t)^{r-j}
= \left[\binom{r-1}{j} + \binom{r-1}{j-1}\right]t^j(1-t)^{r-j}
$$

and Pascal's rule $\binom{r-1}{j} + \binom{r-1}{j-1} = \binom rj$ makes it exactly $B_{j,r}(t)$.
So $P^{(r)}_i = \sum_jB_{j,r}(t)P_{i+j}$, completing the induction.

At $r = n$, $i = 0$: $P^{(n)}_0 = \sum_jB_{j,n}(t)P_j$, which is the Bernstein sum.

**What the proof also proves.** Every intermediate $P^{(r)}_i$ is itself a convex combination of
the original points when $t\in[0,1]$, since the $B_{j,r}$ are non-negative and sum to 1. So no
intermediate quantity ever leaves the hull, which is the numerical argument for the algorithm and
the reason it is safe in fixed point arithmetic.

**And the subdivision.** The two triangle edges $\{P^{(r)}_0\}_{r=0}^n$ and
$\{P^{(n-r)}_r\}_{r=0}^n$ are the control points of the two halves. That comes free: it is the
same table read down its two sides.

### 2.3 The derivative, and the end tangents

**Claim.** $B'(t) = n\sum_{i=0}^{n-1}B_{i,n-1}(t)(P_{i+1}-P_i)$, a degree $n-1$ Bezier curve on
the control points $D_i = n(P_{i+1}-P_i)$.

**Proof.** Differentiate a single Bernstein polynomial:

$$
B_{i,n}'(t) = \binom ni\Big[it^{i-1}(1-t)^{n-i} - (n-i)t^i(1-t)^{n-i-1}\Big]
$$

Use $\binom ni i = n\binom{n-1}{i-1}$ and $\binom ni (n-i) = n\binom{n-1}{i}$:

$$
B_{i,n}'(t) = n\Big[B_{i-1,n-1}(t) - B_{i,n-1}(t)\Big]
$$

with the convention that out of range terms are zero. Then

$$
B'(t) = \sum_{i=0}^nP_iB_{i,n}'(t) = n\sum_{i=0}^nP_i\big[B_{i-1,n-1} - B_{i,n-1}\big]
$$

Collect the coefficient of $B_{j,n-1}$: it appears with $+P_{j+1}$ from the first term and
$-P_j$ from the second, giving

$$
B'(t) = n\sum_{j=0}^{n-1}B_{j,n-1}(t)\,(P_{j+1}-P_j)
$$

**End tangents.** $B_{j,n-1}(0)$ is 1 for $j=0$ and 0 otherwise, so

$$
B'(0) = n(P_1 - P_0), \qquad B'(1) = n(P_n - P_{n-1})
$$

The curve leaves $P_0$ **along the first edge of the control polygon** and arrives at $P_n$ along
the last edge, with speed $n$ times the edge length.

**Why this is the whole of curve joining.** Two Bezier segments meeting at a shared point are
$C^0$ automatically. They are $C^1$ exactly when $n_1(P_n - P_{n-1}) = n_2(Q_1 - Q_0)$, and $G^1$,
tangent continuous but not speed matched, exactly when those three points are collinear. That is
precisely the "smooth node" a font editor offers, and it is why the editor draws the two handles
as one straight bar.

### 2.4 Degree elevation

**The formula.** A degree $n$ Bezier on $P_0,\dots,P_n$ equals the degree $n+1$ Bezier on

$$
Q_i = \frac{i}{n+1}P_{i-1} + \left(1 - \frac{i}{n+1}\right)P_i, \qquad i = 0,\dots,n+1
$$

with $P_{-1}$ and $P_{n+1}$ taken as zero, so $Q_0 = P_0$ and $Q_{n+1} = P_n$.

**Proof.** Start from $1 = t + (1-t)$ and multiply the degree $n$ basis by it:

$$
B_{i,n}(t) = \big(t + (1-t)\big)\binom ni t^i(1-t)^{n-i}
= \binom ni t^{i+1}(1-t)^{n-i} + \binom ni t^i(1-t)^{n+1-i}
$$

Write both in the degree $n+1$ basis, using
$\binom ni = \frac{i+1}{n+1}\binom{n+1}{i+1}$ and $\binom ni = \frac{n+1-i}{n+1}\binom{n+1}{i}$:

$$
B_{i,n}(t) = \frac{i+1}{n+1}B_{i+1,n+1}(t) + \frac{n+1-i}{n+1}B_{i,n+1}(t)
$$

Substituting into $B(t) = \sum_iP_iB_{i,n}(t)$ and collecting the coefficient of $B_{j,n+1}$ gives
$\frac{j}{n+1}P_{j-1} + \frac{n+1-j}{n+1}P_j$, which is $Q_j$.

**The curve is unchanged** because this was an identity between the two basis expansions, not an
approximation. Nothing was dropped.

**Two things this is used for.** Making two curves of different degree compatible before joining
or lofting them, which a CAD kernel does constantly. And as a correctness test on an
implementation: if degree elevation moves the curve at all, something is wrong. The lesson
measures the movement at $10^{-16}$.

**And one thing it says about the control polygon.** Each $Q_j$ is a convex combination of two
consecutive $P$s, so the elevated polygon is closer to the curve than the original. Repeating
degree elevation makes the control polygon converge to the curve, which is a genuine but very slow
$O(1/n)$ algorithm and a useful piece of intuition: the control polygon is a coarse version of the
curve, and raising the degree refines it.

### 2.5 Local support of the B-spline basis

**Claim.** $N_{i,p}(t) = 0$ for $t \notin [u_i, u_{i+p+1})$.

**Proof, by induction on $p$.**

*Base $p = 0$.* $N_{i,0}$ is 1 on $[u_i, u_{i+1})$ and 0 elsewhere, by definition. The support is
$[u_i, u_{i+1})$, which is $[u_i, u_{i+0+1})$.

*Step.* The Cox-de Boor recursion is

$$
N_{i,p}(t) = \frac{t-u_i}{u_{i+p}-u_i}N_{i,p-1}(t) + \frac{u_{i+p+1}-t}{u_{i+p+1}-u_{i+1}}N_{i+1,p-1}(t)
$$

By the inductive hypothesis $N_{i,p-1}$ vanishes outside $[u_i, u_{i+p})$ and $N_{i+1,p-1}$
vanishes outside $[u_{i+1}, u_{i+p+1})$. So $N_{i,p}$ vanishes outside the union

$$
[u_i, u_{i+p}) \cup [u_{i+1}, u_{i+p+1}) = [u_i, u_{i+p+1})
$$

the union being an interval because the two overlap whenever $p \ge 1$. That is the claim.

(A term with a zero denominator has a zero numerator on the relevant set, and the standard
convention sets such a term to zero. That is what makes repeated knots work.)

**Local support, stated for the curve.** $C(t) = \sum_iP_iN_{i,p}(t)$, so at a given $t$ only
those $i$ with $t \in [u_i, u_{i+p+1})$ contribute, and there are exactly $p+1$ of them. Moving
$P_i$ therefore changes $C$ only on $[u_i, u_{i+p+1})$, which is $p+1$ knot spans out of the
whole knot vector.

**Why the count is $p+1$ and not something that grows.** The recursion adds exactly one knot span
of support per degree, starting from one. That is the structural reason B-splines have local
control and Bezier curves do not: the Bernstein basis is the $p = n$ case, where $p+1$ spans is
the entire domain.

### 3.1 Rational Bezier curves, and the circle

Give each control point a weight $w_i > 0$ and divide by the weighted partition of unity:

$$
R(t) = \frac{\sum_i w_iP_iB_{i,n}(t)}{\sum_i w_iB_{i,n}(t)}
$$

The denominator is positive on $[0,1]$, the weights $w_iB_{i,n}/\sum w_jB_{j,n}$ are non-negative
and sum to 1, so **the convex hull property survives**. So do affine invariance and variation
diminishing. What is new is that $R$ is a ratio of polynomials, which is what lets it be a conic.

```python
import math
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import bezier as bz


def rational_bezier(P, w, t):
    """Weighted control points over the weighted partition of unity.

    Weights must be positive, or the denominator can vanish inside [0,1] and the curve blows up.
    Setting every weight equal recovers the plain Bezier curve exactly.
    """
    A = np.asarray(P, dtype=float)
    w = np.asarray(w, dtype=float)
    if A.shape[0] != w.size:
        raise ValueError(f"{A.shape[0]} control points against {w.size} weights")
    if np.any(w <= 0.0):
        raise ValueError("weights must be positive")
    z = np.atleast_1d(np.asarray(t, dtype=float))
    n = A.shape[0] - 1
    B = np.stack([bz.bernstein(n, i, z) for i in range(n + 1)])
    return ((B * w[:, None]).T @ A) / (B * w[:, None]).sum(axis=0)[:, None]


t = np.linspace(0.0, 1.0, 501)
print(f"{'arc angle':>12}{'rational error':>17}{'plain error':>14}")
for degrees in (30.0, 60.0, 90.0, 120.0, 150.0, 179.0):
    a = math.radians(degrees)
    P = np.array([[1.0, 0.0],
                  [1.0, math.tan(a / 2.0)],
                  [math.cos(a), math.sin(a)]])
    w = np.array([1.0, math.cos(a / 2.0), 1.0])
    radius_rational = np.linalg.norm(rational_bezier(P, w, t), axis=1)
    radius_plain = np.linalg.norm(bz.bezier(P, t), axis=1)
    print(f"{degrees:>11.0f} {float(np.max(np.abs(radius_rational - 1.0))):>17.3e}"
          f"{float(np.max(np.abs(radius_plain - 1.0))):>14.3e}")
```

| arc angle | rational, radius error | plain Bezier, same points |
|---|---|---|
| 30 | 2.220e-16 | 6.010e-4 |
| 60 | 2.220e-16 | 1.036e-2 |
| 90 | 2.220e-16 | 6.066e-2 |
| 120 | 2.220e-16 | 2.500e-1 |
| 150 | 3.331e-16 | 1.061 |
| 179 | 4.441e-16 | 5.630e1 |

**The rational quadratic is the circle, to roundoff, at every arc angle.** The construction is:
put $P_0$ and $P_2$ on the circle, put $P_1$ at the intersection of the two tangents, and set
$w_1 = \cos(\alpha/2)$ with the outer weights 1. Exercise 5.1 shows why that works and why no
polynomial can.

**The plain curve through the same three points is not close.** At 90 degrees it is off by 6
percent of the radius, and at 179 degrees the tangent intersection runs to infinity so the plain
curve is off by 56 radii.

**A fair comparison, which the naive one misses.** The plain curve above was handed the same three
points, which are chosen for the rational construction and are not the best three for a polynomial.
Fitting the control points by least squares instead:

| degree of a plain Bezier | best radius error on a quarter arc |
|---|---|
| 2 | 1.964e-2 |
| 3 | 2.682e-3 |
| 4 | 2.008e-4 |
| 6 | 8.862e-7 |
| 8 | 2.162e-9 |
| 12 | **7.105e-15** |

So a polynomial Bezier can get to machine precision on a quarter arc, at degree 12. **The gain
from NURBS is therefore not accuracy in practice, and saying so would be dishonest.** The real
gains are:

- **Exactness under composition.** Trimming, offsetting, intersecting and transforming a curve
  that is exactly a circle keeps exact answers. A $10^{-15}$ approximation accumulates.
- **Degree 2 instead of 12.** Six times fewer control points, and the difference compounds on a
  surface of revolution.
- **Closure under projective transformation.** A perspective projection of a rational curve is a
  rational curve of the same degree, with transformed weights. That is false for polynomials, and
  it is the reason rendering pipelines want rationals.
- **One representation for the whole conic family.** $w_1 < 1$ ellipse, $w_1 = 1$ parabola,
  $w_1 > 1$ hyperbola. A CAD kernel gets all of them from one code path.

### 3.2 Knot insertion

Add a knot without changing the curve. Boehm's algorithm replaces the $p$ affected control points
by convex combinations of their neighbours.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import bezier as bz


def insert_knot(P, degree, knots, u_new):
    """Boehm's algorithm. One new control point, the curve unchanged.

    The p affected points are replaced by convex combinations of consecutive pairs, with the
    blend factor set by where the new knot falls in each point's support. That is de Casteljau's
    step again, applied in the knot vector rather than in t.
    """
    A = np.asarray(P, dtype=float)
    U = np.asarray(knots, dtype=float)
    p = int(degree)
    if not float(U[p]) <= float(u_new) <= float(U[-p - 1]):
        raise ValueError(f"new knot {u_new} is outside the usable range "
                         f"[{float(U[p])}, {float(U[-p - 1])}]")
    k = int(np.searchsorted(U, u_new, side="right") - 1)
    Q = np.empty((A.shape[0] + 1, A.shape[1]))
    Q[:k - p + 1] = A[:k - p + 1]
    Q[k + 1:] = A[k:]
    for i in range(k - p + 1, k + 1):
        alpha = (float(u_new) - U[i]) / (U[i + p] - U[i])
        Q[i] = (1.0 - alpha) * A[i - 1] + alpha * A[i]
    return Q, np.insert(U, k + 1, float(u_new))


rng = np.random.default_rng(42)
P = rng.standard_normal((7, 2))
p = 3
U = bz.open_uniform_knots(P.shape[0], p)
probe = np.linspace(float(U[0]), float(U[-1]) - 1e-12, 1201)
base = bz.bspline(P, p, probe, U)

Q, U2 = P.copy(), U.copy()
print(f"{'inserted at':>13}{'control points':>16}{'max curve move':>17}")
for u_new in (0.5, 0.25, 0.75, 0.5, 0.125):
    Q, U2 = insert_knot(Q, p, U2, u_new)
    moved = float(np.max(np.linalg.norm(bz.bspline(Q, p, probe, U2) - base, axis=1)))
    print(f"{u_new:>13.3f}{Q.shape[0]:>16}{moved:>17.3e}")

Q, U2 = P.copy(), U.copy()
interior = sorted(set(U[p + 1:-p - 1].tolist()))
for u_new in interior:
    while int(np.sum(np.isclose(U2, u_new))) < p:
        Q, U2 = insert_knot(Q, p, U2, u_new)
moved = float(np.max(np.linalg.norm(bz.bspline(Q, p, probe, U2) - base, axis=1)))
print(f"full extraction: {Q.shape[0]} control points, {len(interior) + 1} Bezier segments, "
      f"curve moved {moved:.3e}")
```

| inserted at | control points | max curve movement |
|---|---|---|
| 0.500 | 8 | 4.965e-16 |
| 0.250 | 9 | 6.474e-16 |
| 0.750 | 10 | 6.474e-16 |
| 0.500 (again) | 11 | 5.551e-16 |
| 0.125 | 12 | 7.109e-16 |

**The curve does not move, at roundoff, including when a knot is inserted twice.** Inserting an
existing knot raises its multiplicity, which lowers the smoothness there by one: a cubic B-spline
is $C^2$ at a simple knot, $C^1$ at a double, $C^0$ at a triple. That is how a designer puts a
deliberate corner into an otherwise smooth curve, without changing the curve anywhere else.

**Bezier extraction.** Raise every interior knot to multiplicity $p$ and the basis decouples: each
group of $p+1$ consecutive control points is a Bezier segment, and consecutive segments share
their end point. The measurement: 7 control points and 3 interior knots become **13 control points
forming 4 cubic Bezier segments**, with the curve moved by $9.2\times10^{-16}$.

That count checks out. Four cubic segments need $4\times4 = 16$ points, minus 3 shared joins,
which is 13.

**Why this matters.** It is the bridge between the two representations. A CAD kernel edits in
B-spline form for the local control, then extracts Bezier segments to hand to a renderer or an
exchange format that only speaks Bezier. Exercise 5.2 is the same bridge in the other direction.

### 3.3 Least squares curve fitting

Choose control points so the Bezier curve is closest to the data. With a chosen parametrisation
$s_j$ for the data points, the residual is linear in the control points, so this is exactly the
overdetermined system of Part 5:

$$
\min_{P}\ \big\|B P - Y\big\|_F^2, \qquad B_{ji} = B_{i,n}(s_j)
$$

with $B$ of shape $m\times(n+1)$ and $P$, $Y$ having one column per space dimension. Each column
is a separate least squares problem with the same design matrix, so one QR factorisation of $B$
serves all of them.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import bezier as bz

rng = np.random.default_rng(42)
m = 200
s = np.linspace(0.0, 1.0, m)
data = np.stack([3.0 * s, np.sin(6.0 * s)], axis=1)
noisy = data + 0.02 * rng.standard_normal(data.shape)
print(f"{'degree':>8}{'clean rms':>13}{'noisy rms':>13}{'vs truth':>13}{'design kappa':>15}")
for n in (3, 5, 8, 12, 20, 40):
    B = np.stack([bz.bernstein(n, i, s) for i in range(n + 1)], axis=1)
    P_clean, *_ = np.linalg.lstsq(B, data, rcond=None)
    P_noisy, *_ = np.linalg.lstsq(B, noisy, rcond=None)
    rms = lambda R: float(np.sqrt(np.mean(np.sum(R ** 2, axis=1))))
    print(f"{n:>8}{rms(B @ P_clean - data):>13.3e}{rms(B @ P_noisy - noisy):>13.3e}"
          f"{rms(B @ P_noisy - data):>13.3e}{float(np.linalg.cond(B)):>15.3e}")
```

| degree | rms to clean data | rms to noisy data | rms **to the truth** | $\kappa(B)$ |
|---|---|---|---|---|
| 3 | 6.016e-2 | 6.614e-2 | 6.027e-2 | 5.83 |
| 5 | 3.868e-3 | 2.748e-2 | **5.672e-3** | 2.10e1 |
| 8 | 1.088e-4 | 2.637e-2 | 6.140e-3 | 1.50e2 |
| 12 | 3.469e-8 | 2.633e-2 | 6.318e-3 | 2.17e3 |
| 20 | 6.271e-15 | 2.576e-2 | 8.330e-3 | 4.87e5 |
| 40 | 2.968e-15 | 2.462e-2 | 1.127e-2 | 5.00e11 |

**Three things, all of them Part 5 again.**

**The clean column is approximation and it converges fast.** Degree 20 reproduces the curve to
$6\times10^{-15}$, machine precision, because $\sin$ is analytic and the Bernstein space of degree
20 contains a very good approximation to it.

**The noisy column is not the honest measure.** It keeps falling with degree, from 2.75e-2 to
2.46e-2, which looks like improvement. It is not. It is the fit consuming noise.

**The truth column is the honest measure and it turns around at degree 5.** Best at 5.67e-3, then
6.14e-3, 6.32e-3, 8.33e-3, 1.13e-2. Doubling the degree past the optimum doubles the error against
the underlying curve. That is the bias variance trade of Part 5's regularisation lesson, measured
on a design matrix instead of stated.

**And the conditioning fails independently.** $\kappa(B)$ reaches $5\times10^{11}$ at degree 40.
The Bernstein design matrix is better conditioned than a monomial Vandermonde, which lesson 44
measured at $10^{16}$ by degree 20, but it is not immune, and past degree 20 the normal equations
would be unusable. This is exactly why Part 5 insisted on QR rather than $A^TA$, and
`numpy.linalg.lstsq` uses an SVD for the same reason.

**The fix a real fitter uses** is not to raise the degree. It is to keep the degree at 3 and fit a
B-spline with more knots, so extra freedom is local and the conditioning stays bounded, which is
exercise 4.1's local support again.

### 4.1 The local support fraction

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import bezier as bz

print(f"{'n_control':>11}{'degree':>8}{'measured':>11}{'(p+1)/(n-p)':>14}{'ratio':>8}")
for n in (8, 12, 20, 40, 80):
    for p in (1, 2, 3, 5):
        measured = bz.local_support(n, p)["max_fraction_of_domain"]
        predicted = min((p + 1) / (n - p), 1.0)
        print(f"{n:>11}{p:>8}{measured:>11.4f}{predicted:>14.4f}"
              f"{measured / predicted:>8.3f}")
```

| control points | $p=1$ | $p=2$ | $p=3$ | $p=5$ |
|---|---|---|---|---|
| 8 | 0.2855 | 0.4995 | 0.7990 | 0.9990 |
| 12 | 0.1815 | 0.2990 | 0.4440 | 0.8540 |
| 20 | 0.1050 | 0.1665 | 0.2350 | 0.3980 |
| 40 | 0.0510 | 0.0785 | 0.1080 | 0.1705 |
| 80 | 0.0250 | 0.0380 | 0.0515 | 0.0795 |

**The fitted relationship is $(p+1)/(n-p)$, and it holds to about one percent.** Measured against
the prediction, the ratio is between 0.988 and 0.999 in every one of the twenty cases, with the
small shortfall coming from the probe grid, which cannot see a support region narrower than its
own spacing.

The derivation is exercise 2.5. $N_{i,p}$ is nonzero on $p+1$ knot spans, an open uniform knot
vector on $n$ control points of degree $p$ has $n-p$ interior spans, and the domain is all of
them. So the fraction is $(p+1)/(n-p)$.

**Two readings of the same formula.**

**Fixed degree, growing curve.** The fraction is $O(1/n)$, so it goes to zero. A cubic B-spline
with 1000 control points gives each one 0.4 percent of the curve. This is the property the whole
construction exists for.

**Fixed curve, growing degree.** The fraction rises toward 1 and hits it when $p = n-1$, which is
the Bezier case. The one exception in the table is $n=8$, $p=5$, where the formula gives
$6/3 = 2$ and the measurement gives 0.999. That is not a failure: the support cannot exceed the
whole domain, so the honest prediction is $\min((p+1)/(n-p), 1)$, and with that clamp the
measurement matches.

**The design consequence.** Locality is bought with low degree and many control points, never with
high degree. That is the same conclusion lesson 50 reached for accuracy ($h$ refinement over $p$
refinement on non-smooth data) arrived at from a completely different direction, and it is why
cubic is the universal choice in both places.

### 4.2 The flatness criterion a renderer uses

A renderer subdivides until each piece is within tolerance of its chord, then draws chords. The
test is on the **control points**, not the curve, because the hull property makes that a valid
upper bound at no cost.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import bezier as bz


def flatness(P):
    """Largest distance from a control point to the chord joining the two ends.

    By the convex hull property this bounds the distance from the CURVE to the chord, so it is a
    conservative test that never needs the curve evaluated.
    """
    A = np.asarray(P, dtype=float)
    if A.shape[1] != 2:
        raise ValueError(f"this distance formula is planar, got dimension {A.shape[1]}")
    a, b = A[0], A[-1]
    d = b - a
    length = float(np.linalg.norm(d))
    if length == 0.0:
        return float(np.max(np.linalg.norm(A - a, axis=1)))
    normal = np.array([-d[1], d[0]]) / length
    return float(np.max(np.abs((A - a) @ normal)))


def levels_needed(P, tol, max_levels=40):
    pieces = [np.asarray(P, dtype=float)]
    for level in range(int(max_levels) + 1):
        if max(flatness(c) for c in pieces) <= float(tol):
            return level, len(pieces)
        pieces = [half for c in pieces for half in bz.subdivide(c, 0.5)]
    raise RuntimeError(f"tolerance {tol} not reached in {max_levels} levels")


shapes = (("gentle", np.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.0]])),
          ("medium", np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])),
          ("sharp", np.array([[0.0, 0.0], [1.0, 4.0], [2.0, 0.0]])),
          ("loop", np.array([[0.0, 0.0], [3.0, 3.0], [-1.0, 3.0], [2.0, 0.0]])))
tolerances = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6)
print(f"{'shape':>9}{'flatness':>11}" + "".join(f"{f'tol={t:.0e}':>13}" for t in tolerances)
      + f"{'predicted':>12}")
for name, C in shapes:
    row = [levels_needed(C, t)[0] for t in tolerances]
    predicted = int(np.ceil(0.5 * np.log2(flatness(C) / tolerances[-1])))
    print(f"{name:>9}{flatness(C):>11.3f}" + "".join(f"{v:>13}" for v in row)
          + f"{predicted:>12}")
```

| shape | flatness | tol 1e-1 | tol 1e-2 | tol 1e-3 | tol 1e-4 | tol 1e-6 |
|---|---|---|---|---|---|---|
| gentle | 0.200 | 1 (2) | 3 (8) | 4 (16) | 6 (64) | 9 (512) |
| medium | 1.000 | 2 (4) | 4 (16) | 5 (32) | 7 (128) | 10 (1024) |
| sharp | 4.000 | 3 (8) | 5 (32) | 6 (64) | 8 (256) | 11 (2048) |
| loop | 3.000 | 3 (8) | 5 (32) | 6 (64) | 8 (256) | 11 (2048) |

Numbers in brackets are the piece counts, which are $2^{\text{level}}$.

**Flatness falls by a factor of 4 per level, not 2.** Measuring the sharp curve level by level:

| level | flatness | ratio to the previous |
|---|---|---|
| 1 | 4.472e-1 | 8.944 |
| 2 | 1.768e-1 | 2.530 |
| 3 | 5.590e-2 | 3.162 |
| 4 | 1.516e-2 | 3.688 |
| 5 | 3.876e-3 | 3.911 |
| 6 | 9.747e-4 | 3.977 |
| 7 | 2.440e-4 | 3.994 |

The ratio converges to 4. The reason is that flatness is a **second order** quantity: the
deviation of a curve from its chord over a parameter span $\Delta t$ is $\frac18\Delta t^2|C''|$
to leading order, so halving $\Delta t$ quarters it. The early ratios of 8.9 and 2.5 are the
higher order terms still mattering on a piece that is not yet short.

**The law that falls out.** Levels needed $= \lceil\frac12\log_2(f_0/\tau)\rceil$, where $f_0$ is
the initial flatness. Checking all four shapes at $\tau = 10^{-6}$:

| shape | $f_0$ | $\frac12\log_2(f_0/\tau)$ | rounded up | measured |
|---|---|---|---|---|
| gentle | 0.2 | 8.80 | 9 | 9 |
| medium | 1.0 | 9.97 | 10 | 10 |
| sharp | 4.0 | 10.97 | 11 | 11 |
| loop | 3.0 | 10.76 | 11 | 11 |

Exact in all four.

**What that means for a renderer.** The cost is $2^{\text{levels}} = \sqrt{f_0/\tau}$ pieces, so
halving the tolerance costs only a factor of $\sqrt2$ in work, and a curve ten times curvier costs
only $\sqrt{10} \approx 3.2$ times more. Rendering cost grows as the **square root** of the
precision demanded, which is why subdivision beats sampling the parametrisation uniformly, where
the cost is linear in the precision.

**The one caveat.** Uniform subdivision at $t = 0.5$ splits a piece in parameter, not in arc
length, so a curve with a sharp local feature refines everywhere to fix one place. A production
renderer subdivides adaptively, only where the test fails, which turns the $2^L$ into something
much smaller. The table's bracketed counts are the uniform worst case.

### 4.3 The two evaluators compared

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import bezier as bz

print(f"{'degree':>8}{'largest binomial':>19}{'max gap':>12}{'relative gap':>15}")
for n in (5, 10, 20, 40, 60, 80, 120, 200):
    out = bz.bernstein_vs_de_casteljau(n, n_probe=201, rng=np.random.default_rng(7))
    print(f"{n:>8}{out['largest_binomial']:>19.4g}{out['max_gap']:>12.3e}"
          f"{out['relative_gap']:>15.3e}")

A = np.random.default_rng(7).standard_normal((61, 2))
print()
print(f"degree 60, against the position of t:")
print(f"{'t':>10}{'gap':>13}")
for t in (0.0, 1e-3, 0.25, 0.5, 0.95, 1.0, 1.2, -0.2):
    gap = float(np.max(np.abs(bz.bezier(A, np.array([t]))[0] - bz.de_casteljau(A, float(t))[0])))
    print(f"{t:>10.3f}{gap:>13.3e}")
```

| degree | largest binomial | max gap | relative gap |
|---|---|---|---|
| 5 | 10 | 2.220e-16 | 4.533e-16 |
| 10 | 252 | 6.661e-16 | 3.611e-16 |
| 20 | 1.848e5 | 4.441e-16 | 3.712e-16 |
| 40 | 1.378e11 | 8.882e-16 | 8.103e-16 |
| 60 | 1.183e17 | 8.882e-16 | 8.343e-16 |
| 80 | 1.075e23 | 1.332e-15 | 1.264e-15 |
| 120 | 9.661e34 | 4.441e-16 | 4.291e-16 |
| 200 | 9.055e58 | 8.882e-16 | 8.684e-16 |

**They agree to $10^{-15}$ at every degree up to 200**, while the largest binomial coefficient
grows to $9\times10^{58}$. The folklore that de Casteljau is the accurate one is simply not
visible here, and it does not become visible by pushing the degree.

**The reason is the convex combination.** On $[0,1]$ every Bernstein weight is non-negative, so
$\sum_i|w_iP_i| = \sum_iw_i|P_i| \le \max_i|P_i|$. There is no cancellation to lose digits to and
no intermediate quantity larger than the answer. The huge binomial coefficient is multiplied by an
equally tiny $t^i(1-t)^{n-i}$, and the product never exceeds 1.

**Where they do differ, measurably:**

| $t$ | gap at degree 60 |
|---|---|
| 0.000 | 0.000 |
| 0.001 | 1.110e-16 |
| 0.250 | 0.000 |
| 0.500 | 2.776e-17 |
| 0.950 | 2.776e-16 |
| 1.000 | 0.000 |
| **1.200** | **5.960e-8** |
| **-0.200** | **5.960e-8** |

**Outside $[0,1]$ the gap jumps by eight orders of magnitude.** There $1-t$ is negative, so the
weights alternate in sign, the sum is no longer convex, and the terms are enormously larger than
the answer. At $t=1.2$ and degree 60 the sum of $|w_i|$ is around $10^{10}$, so a relative error
of $10^{-16}$ on the terms is $10^{-6}$ on the result. de Casteljau still blends, so it still
tracks the true extrapolated value.

**Two honest conclusions.**

**The accuracy claim is false where it is usually made and true where nobody makes it.** Inside
the domain the two are identical to roundoff. Outside, de Casteljau is genuinely better by eight
orders, and extrapolating a Bezier curve is rare enough that the folklore is not about this case.

**The real failure of the Bernstein form is different and it is loud.** `math.comb` returns an
exact Python integer, which converts to a double only up to about $10^{308}$:

| degree | result |
|---|---|
| 1000 | works, value 2.523e-2 |
| 1020 | works, value 2.498e-2 |
| 1029 | works, value 2.486e-2 |
| **1030** | `OverflowError: int too large to convert to float` |

Raising is the right failure mode, since it cannot be mistaken for an answer. And no font or CAD
system uses degree 1030, which is why this is a footnote rather than a reason to choose an
evaluator.

**So the reasons to prefer de Casteljau are the ones the lesson gives:** it returns the
subdivision for free, which is the operation a renderer actually performs, and every intermediate
value is inside the convex hull, which matters in fixed point arithmetic and on a GPU.

### 5.1 Why no polynomial Bezier is a circle

**Theorem.** No polynomial curve $(x(t), y(t))$, not both constant, satisfies
$x(t)^2 + y(t)^2 = 1$ identically.

**Proof.** Suppose $x^2 + y^2 = 1$ identically, with $x, y$ polynomials, not both constant.
Differentiate: $xx' + yy' = 0$. Let $d = \max(\deg x, \deg y) \ge 1$.

The polynomial $x^2 + y^2$ has degree $2d$ unless the leading terms cancel. Over the reals they
cannot: if $a$ and $b$ are the leading coefficients of $x$ and $y$ at degree $d$ (one possibly
zero), the coefficient of $t^{2d}$ in $x^2+y^2$ is $a^2 + b^2$, which is zero only when
$a = b = 0$, contradicting the definition of $d$. So $x^2+y^2$ has degree exactly $2d \ge 2$ and
cannot be the constant 1.

**Where it goes if you allow complex coefficients.** Then $a^2 + b^2 = 0$ is possible, with
$b = \pm ia$. That is not a real curve, and it is the same fact seen from the other side: the
circle's rational parametrisation
$\left(\frac{1-s^2}{1+s^2}, \frac{2s}{1+s^2}\right)$ has a denominator that never vanishes over
the reals, and clearing it would need $\sqrt{-1}$.

**A second proof, geometric, worth having.** A polynomial curve of degree $d$ meets any line in at
most $d$ points, counted properly, and its two coordinate functions are entire, so the curve is
unbounded as $t\to\pm\infty$ unless constant. A circle is bounded. So a polynomial curve can cover
at most a bounded arc of it, never the whole circle, and the identity $x^2+y^2=1$ has to hold for
**all** $t$ to define the circle.

**What NURBS change.** The parametrisation becomes a ratio,

$$
x(t) = \frac{X(t)}{W(t)}, \qquad y(t) = \frac{Y(t)}{W(t)}
$$

and the identity to satisfy is $X^2 + Y^2 = W^2$, which is a **polynomial** identity with a
solution: $X = 1-s^2$, $Y = 2s$, $W = 1+s^2$ is the Pythagorean identity
$(1-s^2)^2 + (2s)^2 = (1+s^2)^2$. Rational functions can be bounded without being constant,
because the denominator grows with the numerator, and that is the entire difference.

**The quadratic rational form of exercise 3.1** is this parametrisation rewritten in the Bernstein
basis. The weights $(1, \cos(\alpha/2), 1)$ are exactly what makes the denominator
$B_{0,2} + \cos(\alpha/2)\,2B_{1,2} + B_{2,2}$ come out as the required $W$. Every conic, and not
just the circle, is a quadratic rational Bezier, with the weight deciding which one.

**The scope of the theorem.** It says the same about ellipses, hyperbolas, and every algebraic
curve of degree above 2 that is not rationally parametrisable at all: a cubic with a genus 1 shape
has no rational parametrisation either, so NURBS cannot represent it exactly. NURBS extend the
reach from polynomial to rational, not to everything.

### 5.2 A uniform cubic B-spline curve is a cubic spline

**The claim.** Let $C(t) = \sum_iP_iN_{i,3}(t)$ with a uniform knot vector. Then $C$ is a cubic
spline in lesson 51's sense: piecewise cubic, $C^2$ at every knot.

**Piecewise cubic.** Each $N_{i,3}$ is, by the Cox-de Boor recursion, a polynomial of degree 3 on
each knot span. A finite sum of such is a polynomial of degree at most 3 on each span.

**$C^2$ at the knots.** The standard smoothness result for B-splines is that $N_{i,p}$ is
$C^{p-m}$ at a knot of multiplicity $m$. With $p=3$ and simple knots, $m=1$, so each basis
function is $C^2$, and so is any linear combination. It is not $C^3$, because the third derivative
is piecewise constant and takes different values on adjacent spans, which is exactly lesson 51's
exercise 1.3.

**So the two objects coincide.** Every uniform cubic B-spline curve is a cubic spline, and
conversely every cubic spline can be written in the B-spline basis, which is what lesson 51's
exercise 3.1 does. They are two bases for the same space.

**Why the two bases are used for different jobs.**

| | moment or piecewise coefficient basis | B-spline basis |
|---|---|---|
| the unknowns are | values of $S''$ at the knots | control points |
| interpolation is | the direct problem, a tridiagonal solve | a collocation solve of the same size |
| the coefficients are | not points on the curve | points near the curve, which the curve mimics |
| changing one coefficient | changes the curve everywhere, weakly | changes $p+1$ spans, exactly |
| the shape is bounded by | nothing cheap | the convex hull of the control points |
| conditioning | $\kappa < 3$ | $\kappa \approx 3.2$ after row scaling |

**The split is about which question you are answering.** Interpolation asks: given values at
knots, find the curve. That is naturally a solve, and the moment basis makes the solve
tridiagonal, so it wins. Design asks: given a handle, move the curve predictably. That needs a
basis whose coefficients are geometrically meaningful and locally supported, so the B-spline basis
wins.

**Neither wins on conditioning**, which is the thing people usually assume separates them.
Lesson 51's exercise 3.1 measures 3.21 against 3.00 after row equilibration, both bounded
independently of the number of knots.

**One more difference that matters in practice.** The B-spline basis extends to repeated knots,
which give controlled loss of smoothness at a chosen place, and to non-uniform knots, which is the
NU in NURBS. The moment formulation handles non-uniform spacing but has no mechanism for a
deliberate corner. That is why a CAD kernel stores B-splines even for data it obtained by
interpolation.

### 5.3 Subdivision surfaces and Catmull-Clark

**The idea.** Instead of defining a surface by a formula, define a **refinement rule** on a mesh
and apply it repeatedly. Each round produces a finer mesh, and the limit of that sequence is the
surface. Catmull-Clark, from 1978, is the quadrilateral version, and it is what film animation
uses.

**One round of Catmull-Clark.**

1. **Face points.** For each face, the average of its vertices.
2. **Edge points.** For each edge, the average of its two endpoints and the two adjacent face
   points.
3. **Vertex update.** Move each original vertex $V$ of valence $n$ to
   $$\frac{F + 2R + (n-3)V}{n}$$
   where $F$ is the average of the adjacent face points and $R$ is the average of the adjacent
   edge midpoints.
4. **Reconnect.** Join each face point to the edge points of that face, turning every $k$ sided
   face into $k$ quadrilaterals.

After one round every face is a quadrilateral, and after that the mesh is quad only.

**Its relation to what this lesson built.** On a **regular** region, where every vertex has
valence 4 and every face is a quad, the Catmull-Clark limit surface is exactly the **uniform
bicubic B-spline** surface on that control mesh. The refinement rules are the two dimensional
version of knot insertion at the midpoints of every span, which is exercise 3.2's operation. So
Catmull-Clark is not a new kind of surface in the regular region, it is B-splines computed by
refinement rather than by evaluation.

**What it does that a tensor product spline cannot.**

**Arbitrary topology.** A tensor product surface is a map from a rectangle. It has four sides, a
grid of control points, and a fixed genus. A character's head is a closed surface of genus zero
with no natural rectangular parametrisation, and covering it with tensor product patches means
choosing seams, matching derivatives across them by hand, and living with visible artefacts where
the matching is imperfect. Catmull-Clark takes any polygon mesh, of any topology, and produces one
smooth surface with no seams.

**Extraordinary vertices.** A vertex of valence 3, or 5, or 7 has no tensor product equivalent at
all, because the grid structure forces valence 4. Catmull-Clark handles them with the modified
vertex rule above. The surface is $C^2$ everywhere except at the finitely many extraordinary
vertices, where it is $C^1$ with bounded curvature. Giving up two derivatives at a handful of
isolated points, in exchange for arbitrary topology, is a trade the film industry made
immediately.

**Local refinement.** Adding detail to one region of a tensor product surface requires inserting a
whole knot line across the entire surface, because the grid is a product. Subdivision refines
locally, which is what adaptive tessellation on a GPU needs.

**What it gives up.** A closed form. There is no formula for a Catmull-Clark surface point, only a
limit of a refinement, so exact evaluation needs the eigen-analysis of the subdivision matrix,
which Jos Stam worked out in 1998. And exact intersection or offsetting, which a NURBS kernel does
algebraically, becomes numerical. That is why CAD kept NURBS and animation went to subdivision:
CAD needs exact geometry for manufacture, animation needs arbitrary topology for characters.

---

## Lesson 53, Bivariate Interpolation, and the Wall

### 1.1 What makes a grid two one dimensional problems

**The property is that the basis and the data both factor.** Precisely: the node set is a
**Cartesian product** $\{x_i\}\times\{y_j\}$, and a value is available at every one of the
$n_xn_y$ pairs, with no gaps.

Given that, the interpolant can be written

$$
p(s,t) = \sum_i\sum_j Z_{ij}\,\ell_i(s)\,m_j(t)
$$

where $\ell_i$ is the one dimensional Lagrange basis on $\{x_i\}$ and $m_j$ on $\{y_j\}$. Fixing
$t$ and summing over $j$ first gives, for each $i$, the number
$\sum_jZ_{ij}m_j(t)$, which is the one dimensional interpolant of row $i$ evaluated at $t$. Then
summing over $i$ interpolates those results in $s$. Two passes of a one dimensional method, and
nothing else.

**What breaks off the grid.** Everything, and specifically three things.

**The basis stops factoring.** With scattered points $p_1,\dots,p_m$ there is no set of $x$ values
and no set of $y$ values whose product is the point set, so there is nothing to write as
$\ell_i(s)m_j(t)$.

**The count stops matching.** A grid gives exactly $n_xn_y$ values for exactly $n_xn_y$
coefficients of the product basis. Scattered data gives $m$ values, and the natural product basis
of total degree or of coordinate degree has a count with no reason to equal $m$.

**Unisolvence fails.** Even with the counts matched, the interpolation matrix can be singular, and
by Mairhuber's theorem (exercise 2.4) there is no fixed basis for which it never is. In one
dimension the Vandermonde determinant $\prod_{i<j}(x_j-x_i)$ is nonzero whenever the nodes are
distinct, and there is no analogue.

So the grid is not a convenience. It is the hypothesis that makes the problem solvable by
elementary means, and the rest of the lesson is about what happens without it.

### 1.2 The curse as an exponent

**The usual form.** A degree $d$ tensor product interpolant needs $(d+1)^k$ values. Exponential in
$k$.

**The exponent form.** A method of order $p$ on a grid of spacing $h$ has error $\sim h^p$ and
uses $\sim h^{-k}$ points, so eliminating $h$,

$$
N \sim \tau^{-k/p}
$$

samples to reach accuracy $\tau$.

**What the second form makes clear that the first does not.**

**The dimension is not the enemy on its own. The ratio $k/p$ is.** The first form says
"dimension bad" and stops. The second says the cost is governed by dimension **divided by order**,
so a high order method in high dimension can cost the same as a low order method in low dimension.
Order 8 in 8 dimensions costs $\tau^{-1}$, the same as order 1 in 1 dimension.

**It says what "dimension free" would mean.** A method with $k/p \le 2$ is no worse than Monte
Carlo, whose cost is $\tau^{-2}$ regardless of $k$. So to be competitive with random sampling in
$k$ dimensions a grid method needs $p \ge k/2$: order 5 in 10 dimensions, order 10 in 20. That is
a concrete target, and the first form gives no target at all.

**It is a statement about accuracy, not about resolution.** $(d+1)^k$ counts points on a fixed
grid. $\tau^{-k/p}$ answers the question actually asked, which is what a given accuracy costs.
Those differ: a very smooth function in 20 dimensions may need $d = 2$ rather than $d = 4$, and
the first form has no way to express that while the second absorbs it into $p$.

**It explains why smoothness is the currency.** $p$ is capped by the smoothness of $f$: lesson
50's exercise 4.1 measured the order as $\min(p_{\text{method}}, \text{smoothness})$. So in high
dimension, smoothness is not a technical hypothesis, it is the only thing standing between you and
an impossible sample count. That is why sparse grids (exercise 5.1) advertise the smoothness they
require so prominently.

### 1.3 Why radial basis functions are not a counterexample

**What Mairhuber's theorem says.** Let $\Phi = \{\phi_1,\dots,\phi_m\}$ be a **fixed** set of
continuous functions on a domain $\Omega\subset\mathbb R^2$ containing an open set. Then there is a
set of $m$ distinct points in $\Omega$ for which $\det[\phi_j(p_i)] = 0$.

**The hypothesis that matters is the word fixed.** The basis is chosen first, the points second.

**What RBF interpolation does.** Given points $p_1,\dots,p_m$, it uses the basis

$$
\phi_i(p) = \varphi(\|p - p_i\|)
$$

which is built **from the points**. Move a point and the corresponding basis function moves with
it. So the basis is not fixed in advance, and the theorem's hypothesis fails. RBFs are outside the
theorem's scope, not a counterexample to it.

**Why the matrix is then nonsingular.** The RBF interpolation matrix is $A_{ij} =
\varphi(\|p_i-p_j\|)$, which is symmetric. For a **positive definite** kernel, such as the
Gaussian $e^{-(r/c)^2}$ or the inverse multiquadric, Bochner's theorem gives $A \succ 0$ for any
set of distinct points in any dimension, so it is nonsingular, always. For **conditionally**
positive definite kernels, such as the multiquadric $\sqrt{r^2+c^2}$ and the thin plate spline
$r^2\log r$, the same holds once a low degree polynomial term is appended and the coefficients are
constrained to be orthogonal to it.

That is genuinely stronger than anything a fixed basis achieves, and it is bought by giving up the
fixed basis.

**What the trade costs, so the escape does not look free.**

- The matrix must be rebuilt and factorised for each point set: $O(m^3)$, dense, no bandedness,
  against the $O(n)$ tridiagonal solve of a one dimensional spline.
- The conditioning depends on the point configuration and on the shape parameter, and exercise
  4.3 measures it reaching $10^{18}$.
- There is no local support, so every coefficient affects the whole domain, which is lesson 52's
  complaint about the single Bezier curve, in a different setting.

**The one line version.** Mairhuber says: fixed basis, all point sets, dimension above 1, pick two.
RBFs give up the first. Tensor products give up the second, by requiring a grid. Nothing gives up
the third.

### 2.1 Order independence of the tensor product

**Claim.** Interpolating in $x$ then in $y$ gives the same function as $y$ then $x$.

**Proof.** Let $L_x$ be the operator that takes a function of $x$ sampled at $\{x_i\}$ to its one
dimensional interpolant, so $(L_xg)(s) = \sum_ig(x_i)\ell_i(s)$, and likewise $L_y$.

Doing $y$ first: for each fixed $i$, form $r_i(t) = \sum_jZ_{ij}m_j(t)$. Then interpolate the
values $r_i(t)$ over $i$:

$$
p(s,t) = \sum_i r_i(t)\ell_i(s) = \sum_i\sum_j Z_{ij}m_j(t)\ell_i(s)
$$

Doing $x$ first: for each fixed $j$, form $c_j(s) = \sum_iZ_{ij}\ell_i(s)$, then

$$
q(s,t) = \sum_j c_j(s)m_j(t) = \sum_j\sum_i Z_{ij}\ell_i(s)m_j(t)
$$

The two double sums have the same terms, so $p = q$ by commutativity of addition on a finite sum.

**Where the grid structure was used.** In writing $Z_{ij}$ at all. The step "for each fixed $i$,
interpolate over $j$" requires that a value exists for **every** $(i,j)$ pair, so that $r_i$ is
defined for every $i$ and the second pass has a full set of values to work with. With a missing
entry, $r_i$ is undefined for that $i$, and the second pass has nothing to interpolate.

Said as operators: the claim is $L_x L_y = L_y L_x$, and the two operators commute because they
act on **different indices** of the array $Z$. That is a statement about the index structure of
the data, which is exactly what "the data is on a grid" means.

**A second place it is used, more subtly.** $\ell_i$ depends only on $\{x_i\}$ and $m_j$ only on
$\{y_j\}$. If the $y$ nodes differed from row to row, which is what a ragged grid means, then
$m_j$ would depend on $i$, the double sum would not factor, and the argument would fail even
though every entry was present.

**Measured.** Lesson 51's bicubic tensor product, on an unequal $17\times13$ grid, gives the two
orders agreeing to $1.332\times10^{-15}$, and the lesson's polynomial version agrees to
$10^{-10}$ against both orders and against the direct two dimensional Lagrange form.

### 2.2 The bilinear formula, and exactly what it reproduces

**Derivation.** On the cell $[x_i,x_{i+1}]\times[y_j,y_{j+1}]$ put

$$
u = \frac{s - x_i}{x_{i+1}-x_i}, \qquad v = \frac{t - y_j}{y_{j+1}-y_j}
$$

Interpolate linearly in $x$ along the two edges $y = y_j$ and $y = y_{j+1}$:

$$
a(u) = (1-u)Z_{ij} + uZ_{i+1,j}, \qquad b(u) = (1-u)Z_{i,j+1} + uZ_{i+1,j+1}
$$

Then linearly in $y$ between them:

$$
p = (1-v)a(u) + vb(u)
= (1-u)(1-v)Z_{ij} + u(1-v)Z_{i+1,j} + (1-u)vZ_{i,j+1} + uvZ_{i+1,j+1}
$$

which is the lesson's formula. Doing $y$ first gives the same, which is exercise 2.1 in the
smallest case.

**Exact on $1, x, y, xy$.** The four weights are non-negative and sum to
$(1-u)(1-v)+u(1-v)+(1-u)v+uv = ((1-u)+u)((1-v)+v) = 1$, so constants are reproduced. For $x$:
each of $a$ and $b$ is a linear interpolant in $x$ of a function linear in $x$, so both equal $s$,
and the blend of two copies of $s$ is $s$. Same for $y$. For $xy$: $a(u) = s\,y_j$ and
$b(u) = s\,y_{j+1}$ by the previous argument applied to the linear function $x\cdot y_j$, and
blending those linearly in $v$ gives $s\,t$. So all four monomials are reproduced.

Equivalently: $p$ is bilinear in $(u,v)$, the space of bilinear functions is spanned by
$1,u,v,uv$, and $u$ and $v$ are affine in $s$ and $t$, so the reproduced space is exactly the span
of $1,s,t,st$.

**And nothing more.** The formula produces a bilinear function, and a bilinear function is
determined by its four corner values. Any $f$ that is not bilinear disagrees with the bilinear
function through its corners somewhere inside the cell, so it is not reproduced. Concretely
$s^2$ is not: on the unit cell the corner values of $s^2$ are $0,1,0,1$, the bilinear function
through them is $u$, and $u^2 \ne u$ at $u = 1/2$, an error of $1/4$.

**The consequence for the order.** Bilinear reproduces degree 1 in each variable, so its error is
$O(h^2)$, first order in the sense of "one order below what the leading term would need". The
lesson's own table measures 1.643e-1, 4.320e-2, 1.091e-2, 2.729e-3, 6.535e-4 as the grid halves,
ratios of 3.80, 3.96, 4.00, 4.18, so order 2 in $h$.

**Why it is still the most used method in numerical computing.** $O(1)$ per query after the cell
lookup, four multiplies and three adds, no setup, no solve, and because the weights are
non-negative and sum to 1 it is a convex combination, so it **cannot overshoot**. An image resized
with bilinear cannot produce a pixel brighter than its four neighbours. A bicubic resize can, and
does, which is the halo artefact around sharp edges.

### 2.3 The sample count, and what is needed to be dimension free

**Derivation.** A grid method of order $p$ on spacing $h$ in $k$ dimensions has

$$
\text{error} \sim C h^p, \qquad N \sim h^{-k}
$$

Solve the first for $h$ at a target error $\tau$: $h \sim (\tau/C)^{1/p}$. Substitute:

$$
N \sim (\tau/C)^{-k/p} = C^{k/p}\,\tau^{-k/p}
$$

so, dropping the constant, $N \sim \tau^{-k/p}$.

**The exponent read off.** $\log N = -(k/p)\log\tau$. Tightening the tolerance by a factor of 10
multiplies the count by $10^{k/p}$.

**What order is needed to be dimension free.** "Dimension free" should mean the exponent does not
grow with $k$. Two useful thresholds.

**Matching Monte Carlo.** Monte Carlo has $N \sim \tau^{-2}$ for every $k$. A grid method matches
it when $k/p \le 2$, that is

$$
p \ge \frac{k}{2}
$$

Order 3 in 6 dimensions, order 5 in 10, order 10 in 20. Since $p$ is capped by the smoothness of
$f$, this says a grid method can beat random sampling in 20 dimensions only if $f$ has 10
continuous derivatives.

**Truly independent of $k$.** For the exponent to be a constant, $p$ must be proportional to $k$.
No fixed order method does that. What does is a **spectral** method on an analytic function, whose
error falls geometrically rather than algebraically, so the effective $p$ grows with the
resolution. That is why lesson 47's Chebyshev interpolation matters here: in the tensor product
setting, geometric convergence in each direction is the only mechanism that keeps the exponent
from growing, and it is exactly what sparse grids (exercise 5.1) exploit.

**Measured in exercise 4.2.** With Simpson's rule, $p = 4$, the model predicts the crossover
against Monte Carlo at $k/p = 2$, so $k = 8$. The measurement puts it at $k = 9$.

### 2.4 Mairhuber's theorem

**Theorem.** Let $\Omega\subset\mathbb R^2$ contain an open set, and let
$\phi_1,\dots,\phi_m$ be continuous on $\Omega$ with $m \ge 2$. Then there exist $m$ distinct
points $p_1,\dots,p_m\in\Omega$ with

$$
\det\big[\phi_j(p_i)\big]_{i,j=1}^m = 0
$$

**Proof.** Suppose not: assume $D(p_1,\dots,p_m) = \det[\phi_j(p_i)]$ is nonzero for every choice
of $m$ distinct points in $\Omega$.

Pick any $m$ distinct points, and let $B\subset\Omega$ be an open disc containing $p_1$ and $p_2$
but none of $p_3,\dots,p_m$. Such a disc exists because $\Omega$ contains an open set and the
points are finitely many and distinct.

Now move $p_1$ and $p_2$ continuously **inside $B$**, along two arcs that swap them: let
$p_1(\theta)$ and $p_2(\theta)$ for $\theta\in[0,\pi]$ trace out semicircles of a common circle in
$B$, at opposite ends of a diameter, so that

$$
p_1(0) = p_1, \quad p_2(0) = p_2, \quad p_1(\pi) = p_2, \quad p_2(\pi) = p_1
$$

At every $\theta$ the two points stay distinct, being antipodal on a circle of positive radius,
and stay inside $B$, so they stay distinct from $p_3,\dots,p_m$.

Define $g(\theta) = D(p_1(\theta), p_2(\theta), p_3,\dots,p_m)$. The $\phi_j$ are continuous and
the determinant is a polynomial in its entries, so $g$ is continuous on $[0,\pi]$.

At $\theta = \pi$ the first two arguments are the original ones **swapped**, so the matrix has its
first two rows exchanged, so

$$
g(\pi) = -g(0)
$$

By assumption $g(0) \ne 0$, so $g(0)$ and $g(\pi)$ have opposite signs. By the intermediate value
theorem there is $\theta^*\in(0,\pi)$ with $g(\theta^*) = 0$. But the points at $\theta^*$ are $m$
distinct points of $\Omega$, contradicting the assumption. Hence some point set makes the
determinant vanish.

**The construction is executable, and it is what `mairhuber_example` runs.** With the six
quadratics $1,x,y,x^2,xy,y^2$ and four points fixed, walking the remaining two around a circle
gives a determinant that changes sign, and bisecting on that sign change produces a configuration
with

- determinant $-9.898\times10^{-19}$,
- condition number $2.035\times10^{16}$,

which is singular to working precision. Exercise 3.1 then interpolates the same six points
successfully with every radial kernel, at condition numbers between 42 and $1.6\times10^4$.

**Why bisection rather than a fine grid.** Sampling the path more finely does not reliably land
nearer the root: a 100 point grid came within $3.2\times10^{-6}$ of zero and a 400 point grid
within $6.9\times10^{-6}$, because each grid lands wherever it lands. Bisection converges, and
only a converged root actually exhibits a singular matrix, which is what the theorem asserts.

### 2.5 Where the argument uses the dimension

**The step is: keeping the two points distinct while swapping them.**

In $\mathbb R^2$ the two moving points travel on a circle, at opposite ends of a diameter. After
half a turn they have exchanged places, and at no moment did they coincide. The **plane minus the
diagonal**, $\{(a,b) \in \Omega^2 : a\ne b\}$, is path connected, and the swap is a path in it.

In $\mathbb R^1$ that is impossible. Two distinct points on a line satisfy either $a < b$ or
$b < a$, and the sign of $a - b$ is a continuous nonzero function along any admissible path. To
change $a<b$ into $b<a$ the difference must pass through zero, so the points must collide. The
line minus the diagonal has **two** connected components, one for each order, and no path joins
them.

**So the argument fails at exactly one step, and it is a topological one.** Everything else, the
continuity of the determinant, the row swap changing the sign, the intermediate value theorem, is
dimension free. What is not dimension free is whether the configuration space of ordered distinct
pairs is connected, and that switches at $k = 2$.

**The positive statement for $k=1$.** Since the points cannot be permuted continuously, the
determinant cannot be forced to change sign, and for the polynomial basis it does not: the
Vandermonde determinant is $\prod_{i<j}(x_j-x_i)$, nonzero for distinct nodes and of a fixed sign
once the nodes are ordered. A family with this property is called a **Chebyshev system**, or is
said to satisfy the **Haar condition**, which is exercise 5.3.

**Measured.** `one_dimension_never_fails` runs 500 random 6 node sets and finds no singular
Vandermonde matrix and no determinant near zero, which is the contrast the theorem is about. That
is not a proof, but it is the right control experiment: the same search that succeeds in two
dimensions finds nothing in one.

**And $k \ge 3$.** The same circle argument works, since $\mathbb R^k$ minus the diagonal is
connected for every $k \ge 2$. The theorem is really "one dimension is special", not "two
dimensions are bad".

### 3.1 Radial basis interpolation

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import bivariate as bv

KERNELS = {"multiquadric": lambda r, c: np.sqrt(r * r + c * c),
           "inverse multiquadric": lambda r, c: 1.0 / np.sqrt(r * r + c * c),
           "gaussian": lambda r, c: np.exp(-(r / c) ** 2),
           "thin plate": lambda r, c: np.where(r > 0.0,
                                               r * r * np.log(np.maximum(r, 1e-300)), 0.0)}


def rbf_fit(points, values, kernel, shape):
    """Solve for the coefficients of sum_i lam_i phi(||p - p_i||).

    The basis is built FROM the points, which is why Mairhuber's theorem does not apply: its
    hypothesis is a basis fixed before the points are chosen.
    """
    P = np.atleast_2d(np.asarray(points, dtype=float))
    v = np.asarray(values, dtype=float)
    if P.shape[0] != v.size:
        raise ValueError(f"{P.shape[0]} points against {v.size} values")
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    A = kernel(d, float(shape))
    return np.linalg.solve(A, v), float(np.linalg.cond(A))


def rbf_eval(points, coefficients, kernel, shape, query):
    P = np.atleast_2d(np.asarray(points, dtype=float))
    q = np.atleast_2d(np.asarray(query, dtype=float))
    d = np.linalg.norm(q[:, None, :] - P[None, :, :], axis=2)
    return kernel(d, float(shape)) @ np.asarray(coefficients, dtype=float)


report = bv.mairhuber_example(n_probe=400, rng=np.random.default_rng(42))
bad = np.asarray(report["singular_points"])
target = lambda p: np.sin(3.0 * p[:, 0]) + np.cos(2.0 * p[:, 1])
values = target(bad)
print(f"the singular point set: polynomial determinant {report['determinant_at_the_root']:.3e}, "
      f"kappa {report['condition_at_the_root']:.3e}")
print(f"{'kernel':>22}{'matrix kappa':>15}{'residual at the data':>22}")
for name, kernel in KERNELS.items():
    coef, kappa = rbf_fit(bad, values, kernel, 0.7)
    residual = float(np.max(np.abs(rbf_eval(bad, coef, kernel, 0.7, bad) - values)))
    print(f"{name:>22}{kappa:>15.3e}{residual:>22.3e}")

B = np.column_stack([np.ones(bad.shape[0]), bad[:, 0], bad[:, 1],
                     bad[:, 0] ** 2, bad[:, 0] * bad[:, 1], bad[:, 1] ** 2])
c = np.linalg.solve(B, values)
print(f"{'fixed quadratic basis':>22}{float(np.linalg.cond(B)):>15.3e}"
      f"{float(np.max(np.abs(B @ c - values))):>22.3e}")
```

| basis, on the six singular points | matrix $\kappa$ | residual at the data |
|---|---|---|
| multiquadric | 1.577e4 | 1.066e-14 |
| inverse multiquadric | 3.974e3 | 3.997e-15 |
| gaussian | 2.599e3 | 4.441e-16 |
| thin plate | 4.180e1 | 4.441e-16 |
| **fixed quadratic basis** | **2.035e16** | **1.228e-1** |

**Every radial kernel interpolates the point set the polynomial basis cannot**, at condition
numbers between 42 and $1.6\times10^4$ against the polynomial's $2\times10^{16}$.

**The failure mode of the polynomial basis is worth naming.** `numpy.linalg.solve` did **not**
raise. It returned a vector, and that vector misses the data by 0.123, which is 6 percent of the
range of the values. A least squares fallback does better, 0.015, and is still not an interpolant.
A singular matrix that is singular only to working precision fails silently, which is why the
determinant and the condition number were checked rather than trusting the solver to complain.

**Accuracy on ordinary scattered data**, 60 random points in the unit square, interpolating a
Gaussian bump, measured at 500 random query points with shape 0.4:

| kernel | $\kappa$ | max error | rms error |
|---|---|---|---|
| multiquadric | 3.328e9 | 5.557e-3 | 4.711e-4 |
| inverse multiquadric | 1.503e8 | 4.677e-3 | 8.649e-4 |
| **gaussian** | **7.165e10** | **2.698e-5** | **2.943e-6** |
| thin plate | 1.839e4 | 1.011e-1 | 9.503e-3 |

**The ordering is the opposite of the conditioning ordering, and that is the whole story of RBFs.**
The Gaussian is 200 times more accurate than the multiquadric and has a condition number 20 times
worse. The thin plate spline is the best conditioned by six orders and the worst by two.

The reason is that accuracy comes from the kernel being **smooth and wide**, so it can represent a
smooth target with few centres, and a smooth wide kernel makes the columns of $A$ nearly parallel,
which is what a large condition number is. Exercise 4.3 measures that trade directly.

Note also that the thin plate spline here is used without its polynomial tail. It is only
conditionally positive definite, so the plain matrix is not guaranteed nonsingular, and its poor
accuracy is partly that missing affine term. It succeeded on the six point test above because the
matrix happened to be nonsingular there, not because it was guaranteed to be.

### 3.2 The bicubic spline

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import splines as sp, bivariate as bv


def bicubic(x, y, Z, s, t):
    """Not-a-knot cubic spline in y for each row, then in x through the results.

    Both passes are lesson 51's tridiagonal solve, so the whole surface costs
    O(n_x n_y) to set up, and the order of the two passes does not matter.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.shape != (x.size, y.size):
        raise ValueError(f"grid {Z.shape} does not match nodes {x.size} by {y.size}")
    s = np.atleast_1d(np.asarray(s, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    rows = np.stack([np.atleast_1d(sp.not_a_knot(y, Z[i])(t)) for i in range(x.size)])
    out = np.empty((s.size, t.size))
    for j in range(t.size):
        out[:, j] = np.atleast_1d(sp.not_a_knot(x, rows[:, j])(s))
    return out


surface = lambda a, b: np.exp(a) * np.sin(3.0 * b) + a * b * b
ps = np.linspace(0.0, 1.0, 61)
truth = np.asarray([[surface(a, b) for b in ps] for a in ps])
print(f"{'grid':>8}{'bicubic':>13}{'order':>8}{'bilinear':>13}{'order':>8}")
previous_c = previous_l = None
for n in (5, 9, 17, 33, 65):
    x = np.linspace(0.0, 1.0, n)
    Z = np.asarray([[surface(a, b) for b in x] for a in x])
    ec = float(np.max(np.abs(bicubic(x, x, Z, ps, ps) - truth)))
    el = float(np.max(np.abs(bv.bilinear(x, x, Z, ps, ps) - truth)))
    oc = "" if previous_c is None else f"{np.log2(previous_c / ec):.3f}"
    ol = "" if previous_l is None else f"{np.log2(previous_l / el):.3f}"
    print(f"{f'{n}x{n}':>8}{ec:>13.3e}{oc:>8}{el:>13.3e}{ol:>8}")
    previous_c, previous_l = ec, el
```

| grid | bicubic | order | bilinear | order | full polynomial tensor |
|---|---|---|---|---|---|
| 5x5 | 1.765e-2 |  | 1.643e-1 |  | 4.980e-3 |
| 9x9 | 7.614e-4 | 4.535 | 4.320e-2 | 1.928 | 1.075e-6 |
| 17x17 | 2.916e-5 | 4.707 | 1.091e-2 | 1.986 | 1.183e-11 |
| 33x33 | 1.234e-6 | 4.563 | 2.729e-3 | 1.999 | not run |
| 65x65 | 3.255e-8 | 5.244 | 6.535e-4 | 2.062 | not run |

**The order is 4, as it is in one dimension.** The fitted values run 4.5 to 5.2 rather than
sitting exactly on 4, for the reason lesson 51's exercise 4.1 explains: at these grid sizes the
not-a-knot boundary correction is still decaying faster than the interior $O(h^4)$ term, which
inflates the fitted order. Bilinear, with no boundary correction to decay, sits on 2 within one
percent from the second refinement onward.

**Order independence, checked on an unequal grid.** On $17\times13$, doing $y$ then $x$ against
$x$ then $y$ agrees to $1.332\times10^{-15}$. The unequal shape matters: a square grid would let
an indexing error pass unnoticed.

**Against the full polynomial tensor product.** The polynomial is far more accurate on this
analytic surface, $1.2\times10^{-11}$ at 17 nodes per axis against the spline's
$2.9\times10^{-5}$, because it converges geometrically and the spline converges at order 4. That
is lesson 47's story in two dimensions and it is real.

It is also why bicubic is what libraries ship. The polynomial version needs the surface to be
analytic and the nodes to be chosen, and it inherits Runge's phenomenon in both directions on
equally spaced nodes. The spline needs neither, costs $O(n^2)$ to set up rather than $O(n^2)$ per
evaluation, and never diverges.

### 3.3 Barycentric coordinates on a triangle, and scattered data

```python
import numpy as np
import sys
sys.path.insert(0, "src")


def barycentric_triangle(vertices, query):
    """Solve the 2x2 system for two coordinates, the third by the sum to one constraint.

    Degenerate triangles, whose vertices are collinear, give a singular system, and numpy raises.
    That is the right behaviour: there is no interpolant on a degenerate triangle.
    """
    T = np.asarray(vertices, dtype=float)
    if T.shape != (3, 2):
        raise ValueError(f"need three planar vertices, got shape {T.shape}")
    q = np.atleast_2d(np.asarray(query, dtype=float))
    M = np.column_stack([T[0] - T[2], T[1] - T[2]])
    ab = np.linalg.solve(M, (q - T[2]).T).T
    return np.column_stack([ab, 1.0 - ab.sum(axis=1)])


rng = np.random.default_rng(42)
tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 1.0]])
at_vertices = barycentric_triangle(tri, tri)
print(f"at the vertices, deviation from the identity: "
      f"{float(np.max(np.abs(at_vertices - np.eye(tri.shape[0])))):.3e}")
q = rng.uniform(-0.5, 1.5, (2000, 2))
lam = barycentric_triangle(tri, q)
print(f"sum to one everywhere: {float(np.max(np.abs(lam.sum(axis=1) - 1.0))):.3e}")
inside = np.all(lam >= 0.0, axis=1)
exact_area = 0.5 * abs(float(np.linalg.det(np.column_stack([tri[0] - tri[2], tri[1] - tri[2]]))))
estimate = float(inside.mean()) * 4.0
sigma = 4.0 * float(np.sqrt(inside.mean() * (1 - inside.mean()) / q.shape[0]))
print(f"area by counting: {estimate:.4f} +/- {sigma:.4f}, exact {exact_area:.4f}")
print("what a linear interpolant on the triangle reproduces:")
for label, g in (("1", lambda p: np.ones(p.shape[0])),
                 ("x", lambda p: p[:, 0]),
                 ("y", lambda p: p[:, 1]),
                 ("x*y", lambda p: p[:, 0] * p[:, 1]),
                 ("x**2", lambda p: p[:, 0] ** 2)):
    pts = q[inside]
    got = barycentric_triangle(tri, pts) @ g(tri)
    print(f"  {label:<6} max error {float(np.max(np.abs(got - g(pts)))):.3e}")
```

**The coordinates behave.** At the vertices they are the identity matrix exactly, to 0. They sum
to 1 to $1.1\times10^{-16}$ everywhere, including outside the triangle, where one of them is
negative. The sign test $\lambda_i \ge 0$ identifies the interior: 272 of 2000 uniform samples over
a region of area 4, giving $0.5440 \pm 0.0307$ against the exact area 0.5, which is 1.4 standard
errors and consistent.

**What it reproduces:**

| function | max error over the triangle |
|---|---|
| $1$ | 0.000 |
| $x$ | 1.110e-16 |
| $y$ | 0.000 |
| $xy$ | 1.967e-1 |
| $x^2$ | 2.484e-1 |

Affine functions exactly, nothing else. Note the contrast with bilinear (exercise 2.2), which does
reproduce $xy$: the triangle carries three coefficients and spans $\{1,x,y\}$, the quadrilateral
carries four and spans $\{1,x,y,xy\}$. Triangles buy arbitrary topology and pay one basis function
for it.

**Scattered interpolation through a Delaunay triangulation.** Triangulate the points, find the
containing triangle for each query, interpolate linearly in barycentric coordinates. Delaunay is
the right triangulation because it **maximises the minimum angle** over all triangulations of the
point set, which is exactly the quantity the error bound depends on.

Measured on $\sin(2x)\cos(2y)$ over the unit square, with the triangles' worst aspect ratio
reported alongside:

| points, random | max error | order | rms error | order | worst aspect |
|---|---|---|---|---|---|
| 68 | 1.474e-1 |  | 2.264e-2 |  | 624 |
| 260 | 4.726e-2 | 1.641 | 5.078e-3 | 2.156 | 1.81e3 |
| 1028 | 5.265e-3 | 3.166 | 8.259e-4 | 2.620 | 1.32e4 |
| 4100 | 1.470e-3 | 1.840 | 2.123e-4 | 1.960 | 6.52e4 |

| points, perturbed grid | max error | order | rms error | order | worst aspect |
|---|---|---|---|---|---|
| 64 | 2.509e-2 |  | 8.021e-3 |  | 5.55 |
| 256 | 5.154e-3 | 2.283 | 1.798e-3 | 2.157 | 6.68 |
| 1024 | 1.462e-3 | 1.817 | 4.214e-4 | 2.093 | 7.48 |
| 4096 | 4.259e-4 | 1.780 | 1.032e-4 | 2.030 | 7.50 |

**The theory is $O(h^2)$, and it appears only when the triangles are controlled.** The bound is

$$
\|f - p\|_\infty \le C\,\frac{h^2}{\sin\theta_{\min}}\,\|D^2f\|
$$

so a thin triangle, with a small minimum angle, has an unbounded constant. On random points the
worst aspect ratio grows from 624 to $6.5\times10^4$ as points are added, because a denser random
set contains ever thinner accidental triangles, and the fitted max order swings 1.64, 3.17, 1.84,
which is noise around 2 rather than a measurement of it. Even the median over nine seeds does not
settle it.

On a perturbed grid the aspect ratio stays between 5.6 and 7.5, and the rms order reads 2.157,
2.093, 2.030, converging on 2 from above. That is the honest measurement of the order, and the
random point table is the honest measurement of what happens when the mesh is not controlled.

**The practical lesson**, which is the reason meshing is its own field: the error of a linear
triangular interpolant is governed by the **worst** triangle, not the average one, so the whole
value of Delaunay and of mesh improvement is in bounding the minimum angle.

### 4.1 Tensor product order in two and three dimensions

```python
import math
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import splines as sp, hermite as hm


def tensor_nd(nodes, values, query, one_d):
    """Apply a one dimensional interpolant along each axis in turn.

    `one_d(x, row, q)` interpolates the samples `row` taken at `x` and evaluates at `q`. The
    axes are handled independently, so this works for any number of them and any per axis size.
    """
    work = np.asarray(values, dtype=float)
    if work.ndim != len(nodes):
        raise ValueError(f"{work.ndim} data axes against {len(nodes)} node sets")
    for axis in range(len(nodes) - 1, -1, -1):
        x = np.asarray(nodes[axis], dtype=float)
        q = np.asarray(query[axis], dtype=float)
        moved = np.moveaxis(work, axis, -1)
        shape = list(moved.shape[:-1]) + [q.size]
        rows = np.stack([one_d(x, row, q) for row in moved.reshape(-1, x.size)])
        work = np.moveaxis(rows.reshape(shape), -1, axis)
    return work


def product_grid(f, nodes):
    """Sample the separable product f(a_1) ... f(a_k) on the tensor grid, for any k."""
    out = np.ones(tuple(n.size for n in nodes))
    for axis, n in enumerate(nodes):
        shape = [1] * len(nodes)
        shape[axis] = n.size
        out = out * f(n).reshape(shape)
    return out


g = lambda a: np.exp(np.sin(3.0 * a))
methods = (("piecewise linear", lambda x, row, q: hm.piecewise_linear(x, row, q)),
           ("cubic spline", lambda x, row, q: np.atleast_1d(sp.not_a_knot(x, row)(q))))
for label, one_d in methods:
    print(label)
    print(f"{'k':>4}{'n/axis':>9}{'max error':>14}{'order':>8}{'total nodes':>14}")
    for k in (1, 2, 3):
        previous = None
        for n in (9, 17, 33, 65):
            nodes = [np.linspace(0.0, 1.0, n)] * k
            probe = [np.linspace(0.0, 1.0, 41)] * k
            got = tensor_nd(nodes, product_grid(g, nodes), probe, one_d)
            e = float(np.max(np.abs(got - product_grid(g, probe))))
            order = "" if previous is None else f"{math.log2(previous / e):.3f}"
            print(f"{k:>4}{n:>9}{e:>14.3e}{order:>8}{n ** k:>14}")
            previous = e
```

**Piecewise linear, order 2 in each direction:**

| $k$ | $n$ per axis | max error | order | total nodes |
|---|---|---|---|---|
| 1 | 9 | 4.437e-2 |  | 9 |
| 1 | 65 | 7.166e-4 | 1.984 | 65 |
| 2 | 9 | 2.385e-1 |  | 81 |
| 2 | 65 | 3.895e-3 | 1.979 | 4225 |
| 3 | 9 | 9.614e-1 |  | 729 |
| 3 | 65 | 1.588e-2 | 1.974 | 274625 |

**Cubic spline, order 4 in each direction:**

| $k$ | $n$ per axis | max error | order | total nodes |
|---|---|---|---|---|
| 1 | 9 | 3.113e-3 |  | 9 |
| 1 | 65 | 1.940e-7 | 4.395 | 65 |
| 2 | 9 | 9.223e-3 |  | 81 |
| 2 | 65 | 6.880e-7 | 4.302 | 4225 |
| 3 | 9 | 2.713e-2 |  | 729 |
| 3 | 65 | 2.805e-6 | 3.989 | 274625 |

**The order is the one dimensional order, in every dimension.** Piecewise linear: 1.984, 1.979,
1.974 for $k = 1, 2, 3$. Cubic spline: 4.395, 4.302, 3.989. The dimension does not touch the
exponent.

**What the dimension does touch is the constant.** At 65 nodes per axis the piecewise linear error
is 7.17e-4, 3.90e-3, 1.59e-2 for $k=1,2,3$, growing by a factor of about 5.4 and then 4.1 per
dimension. For a separable target $F = \prod_ag(x_a)$ the reason is exact: the error of a product
telescopes as $\sum_a(\text{error in axis }a)\prod_{b\ne a}|g|$, so it grows like
$k\,\|g\|^{k-1}$ times the one dimensional error. With $\|g\|_\infty = e \approx 2.72$ here, the
predicted growth from $k=2$ to $k=3$ is $\frac32\times2.72 = 4.1$, which is what is measured.

**The cost, which is where the trouble lives.** Order 2 at $10^{-2}$ accuracy needs 65 nodes per
axis, which is 65 values in one dimension, 4225 in two, and 274625 in three. The **exponent** is
unchanged and the **count** is $n^k$. That is the curse: not a loss of accuracy, but a loss of
affordability at fixed accuracy, which is exactly the form exercise 1.2 argued for.

**The polynomial tensor product says the same thing more sharply.** Interpolating a separable
analytic product on $n$ equally spaced nodes per axis, the error falls by the same factor per
added node in every dimension:

| $n$ per axis | $k=1$ | $k=2$ | $k=3$ |
|---|---|---|---|
| 5 | 3.425e-4 | 3.201e-3 | 2.261e-2 |
| 6 | 1.919e-5 | 1.935e-4 | 1.463e-3 |
| 7 | 1.019e-6 | 1.027e-5 | 7.771e-5 |
| ratio 6 to 7 | **18.83** | **18.83** | **18.83** |

Identical to four digits across three dimensions. The rate per node is a one dimensional quantity
and the dimension only multiplies the constant, which is what "tensor product" means.

### 4.2 Where Monte Carlo takes over

Integrating $\prod_{a=1}^k\cos(x_a)$ over the unit cube, whose exact value is $\sin(1)^k$, to a
relative tolerance of $10^{-4}$. Grid: Simpson's rule, order 4. Monte Carlo: plain uniform
sampling, the sample count doubled until the median error over nine repetitions meets the
tolerance.

```python
import math
import numpy as np

rng = np.random.default_rng(42)
tol = 1e-4


def simpson_weights(n):
    """Composite Simpson on n equally spaced points, normalised to sum to 1. n must be odd."""
    if n < 3 or n % 2 == 0:
        raise ValueError(f"Simpson needs an odd number of points of at least 3, got {n}")
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w / w.sum()


def grid_samples(k, tol, n_max=201):
    exact = math.sin(1.0) ** k
    for n in range(3, int(n_max) + 1, 2):
        x = np.linspace(0.0, 1.0, n)
        one_axis = float(simpson_weights(n) @ np.cos(x))
        if abs(one_axis ** k - exact) <= tol * abs(exact):
            return float(n) ** k, n
    raise RuntimeError(f"tolerance {tol} not reached by n = {n_max}")


def monte_carlo_samples(k, tol, trials=9, n_max=6 * 10 ** 7):
    exact = math.sin(1.0) ** k
    N = 100
    while N <= int(n_max):
        errors = [abs(float(np.mean(np.prod(np.cos(rng.uniform(0.0, 1.0, (N, k))), axis=1)))
                      - exact) for _ in range(int(trials))]
        if float(np.median(errors)) <= tol * abs(exact):
            return float(N)
        N *= 2
    return float("inf")


print(f"{'k':>4}{'n/axis':>9}{'grid samples':>15}{'Monte Carlo':>15}{'winner':>14}{'k/p':>7}")
for k in (1, 2, 3, 4, 6, 8, 9, 10, 12):
    grid, n = grid_samples(k, tol)
    mc = monte_carlo_samples(k, tol)
    print(f"{k:>4}{n:>9}{grid:>15.4g}{mc:>15.4g}"
          f"{('grid' if grid < mc else 'Monte Carlo'):>14}{k / 4.0:>7.2f}")
```

| $k$ | $n$ per axis | grid samples | Monte Carlo samples | winner | model $k/p$ |
|---|---|---|---|---|---|
| 1 | 5 | 5 | 8.19e5 | grid | 0.25 |
| 2 | 5 | 25 | 6.55e6 | grid | 0.50 |
| 3 | 5 | 125 | 3.28e6 | grid | 0.75 |
| 4 | 5 | 625 | 3.28e6 | grid | 1.00 |
| 6 | 7 | 1.18e5 | 6.55e6 | grid | 1.50 |
| 8 | 7 | 5.77e6 | 1.31e7 | grid | 2.00 |
| **9** | 7 | **4.04e7** | **1.31e7** | **Monte Carlo** | 2.25 |
| 10 | 7 | 2.83e8 | 1.31e7 | Monte Carlo | 2.50 |
| 12 | 7 | 1.38e10 | 2.62e7 | Monte Carlo | 3.00 |

**Monte Carlo wins from $k = 9$.** The model predicts the crossover where $k/p = 2$, that is
$k = 8$ for Simpson's $p = 4$, and the measurement puts it one dimension later. The gap is the
constant: Monte Carlo's cost is $\sigma^2/\tau^2$ with a variance that is not 1, and Simpson's is
$(C/\tau)^{k/4}$ with a small $C$ for this very smooth integrand, so the crossover slides a little.
The **slope** is the model's, and it is the slope that matters: past the crossover the grid cost
multiplies by 7 per dimension while the Monte Carlo cost does not move at all.

**Two things this measurement does not say.**

**It does not say Monte Carlo is accurate.** It reaches $10^{-4}$ relative with $1.3\times10^7$
samples in nine dimensions. In one dimension Simpson does the same with **five**. Monte Carlo is
the worst method available in low dimension by six orders of magnitude, and it wins in high
dimension only because everything else has become impossible.

**It does not say the crossover is at 9 in general.** It is at $k \approx 2p$, so it moves with
the order of the grid method and therefore with the smoothness of the integrand. For a
non-smooth integrand where the grid method is order 1, the crossover is at $k = 2$, and Monte
Carlo wins almost immediately. For a spectrally accurate method on an analytic integrand there
may be no crossover at all in any practical dimension, which is what sparse grids exploit.

### 4.3 RBF conditioning against shape and spacing

```python
import numpy as np

rng = np.random.default_rng(42)
gaussian = lambda r, c: np.exp(-(r / c) ** 2)
f = lambda p: np.exp(-4.0 * ((p[:, 0] - 0.4) ** 2 + (p[:, 1] - 0.6) ** 2))
points = rng.uniform(0.0, 1.0, (40, 2))
query = rng.uniform(0.05, 0.95, (400, 2))
truth = f(query)
print(f"{'shape c':>10}{'kappa':>13}{'max error':>13}{'solve residual':>17}")
for c in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4):
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    A = gaussian(d, c)
    coef = np.linalg.solve(A, f(points))
    dq = np.linalg.norm(query[:, None, :] - points[None, :, :], axis=2)
    got = gaussian(dq, c) @ coef
    print(f"{c:>10.2f}{float(np.linalg.cond(A)):>13.3e}"
          f"{float(np.max(np.abs(got - truth))):>13.3e}"
          f"{float(np.max(np.abs(A @ coef - f(points)))):>17.3e}")
```

| shape $c$ | $\kappa(A)$ | max error | solve residual |
|---|---|---|---|
| 0.05 | 5.721 | 9.989e-1 | 3.331e-16 |
| 0.10 | 2.852e1 | 9.688e-1 | 4.441e-16 |
| 0.20 | 2.493e3 | 3.372e-1 | 4.441e-16 |
| 0.40 | 2.552e7 | 1.069e-3 | 2.109e-15 |
| **0.80** | 1.746e12 | **2.375e-4** | 7.551e-11 |
| 1.60 | 1.146e17 | 6.730e-3 | 1.548e-5 |
| 3.20 | 1.486e18 | 2.555e-2 | 4.055e-3 |
| 6.40 | 2.680e19 | 5.899e-2 | 5.192e-2 |

**The trade off, stated exactly.** The error falls by four orders as $c$ grows from 0.05 to 0.8,
then rises again. The condition number rises monotonically by nineteen orders across the same
range. The best accuracy sits at $c = 0.8$, where $\kappa$ is already $1.7\times10^{12}$.

**Why both halves happen.**

**Small $c$: the kernel is too narrow.** Each basis function is a spike around its own centre and
is essentially zero at every other point, so $A \approx I$, perfectly conditioned, and the
interpolant is a set of isolated bumps that is close to zero between the data. The error 0.999 at
$c = 0.05$ is the full amplitude of the target: the interpolant is doing nothing away from the
points.

**Large $c$: the kernel is too wide.** Every basis function is nearly the constant 1 over the
domain, so the columns of $A$ are nearly parallel and $\kappa$ explodes. The approximation space
is in principle excellent, this is the flat limit where RBF interpolation approaches polynomial
interpolation, but the basis for it is nearly dependent, so the solve loses the digits before the
approximation can deliver them. The solve residual column shows exactly where that starts: it is
$10^{-16}$ up to $c = 0.4$, then $7.6\times10^{-11}$ at 0.8, then $1.5\times10^{-5}$, then
$5.2\times10^{-2}$. Past $c = 1.6$ the answer is not an interpolant at all.

**The governing statement is an uncertainty principle.** For a fixed point set, accuracy and
conditioning cannot be improved together: whatever makes the space good makes the basis bad. This
is Schaback's trade off principle, and it is not an artefact of the Gaussian.

**And against the spacing, at fixed $c = 0.4$:**

| points | spacing $h$ | $\kappa(A)$ | max error |
|---|---|---|---|
| 16 | 0.3333 | 5.728e1 | 6.436e-3 |
| 36 | 0.2000 | 1.746e5 | 1.266e-4 |
| 64 | 0.1429 | 4.353e9 | 1.077e-5 |
| 144 | 0.0909 | 1.331e18 | 2.193e-8 |
| 256 | 0.0667 | 4.050e18 | 2.291e-11 |

**The error keeps falling while $\kappa$ passes $10^{18}$**, which looks impossible and is not.
The conditioning is a property of the **basis**, not of the interpolation problem: the function
being represented is well determined even when the coefficients representing it are not. The
coefficient vector at 256 points is garbage in its individual entries, and the combination it
forms is accurate to $2\times10^{-11}$. Anyone reading only the condition number would have
stopped at 64 points and given up eight orders of accuracy.

That does have a limit. Once the residual column starts to move, as it does above $c = 0.8$ in the
first table, the accuracy really is gone. The rule that survives both tables is: **watch the
residual, not the condition number.** Stable algorithms exist that avoid the ill conditioned basis
entirely, RBF-QR and the Contour-Pade method, and they exist precisely because the accuracy in the
flat limit is real and worth recovering.

### 5.1 Sparse grids

**The construction.** Write the one dimensional interpolant at level $\ell$ as $U_\ell$, with
$U_0$ the trivial one, and define the **difference** operators

$$
\Delta_\ell = U_\ell - U_{\ell-1}
$$

The full tensor grid at level $L$ in $k$ dimensions is
$\bigotimes_{a=1}^k U_L = \sum_{|\boldsymbol\ell|_\infty\le L}\bigotimes_a\Delta_{\ell_a}$,
summing over the full cube of multi-indices. **Smolyak's construction** keeps only the simplex:

$$
A(L,k) = \sum_{|\boldsymbol\ell|_1 \le L + k - 1}\ \bigotimes_{a=1}^k\Delta_{\ell_a}
$$

So a term with high resolution in one direction must have low resolution in the others. The full
grid takes the product of the resolutions; the sparse grid takes their sum.

**The count.** The full grid has $O(2^{Lk})$ points, or $N^k$ where $N = 2^L$. The sparse grid has

$$
O\!\left(N(\log N)^{k-1}\right)
$$

points, with an accuracy of $O(N^{-p}(\log N)^{k-1})$ for a method of one dimensional order $p$.
The exponential in $k$ has moved from the base to the logarithm, which in ten dimensions at
$N = 2^{10}$ is the difference between $10^{30}$ and about $10^{10}$ points.

**Why it works.** In the difference form, a term $\bigotimes_a\Delta_{\ell_a}$ has size
$\prod_a2^{-p\ell_a}$, so the terms decay geometrically in $|\boldsymbol\ell|_1$. Terms of large
$\ell_1$ contribute little, and the full grid spends most of its points on exactly those terms:
the corner of the index cube where every direction is refined at once. Truncating to the simplex
discards the many small terms and keeps the few large ones.

**The smoothness required, which is the catch and is usually understated.** The estimate needs
**bounded mixed derivatives**:

$$
\left\|\frac{\partial^{\,p_1+\cdots+p_k}f}{\partial x_1^{p_1}\cdots\partial x_k^{p_k}}\right\|_\infty
< \infty \quad\text{for all } p_a \le p
$$

That is much stronger than $f \in C^p$. It asks for $p$ derivatives in **every** direction
**simultaneously**, so a function of total smoothness $C^p$ generally does not qualify: it needs
$C^{pk}$ in the ordinary sense to guarantee the mixed derivatives up to order $p$ in each
variable.

The reason is visible in the construction. Discarding the corner terms is only safe if those terms
are small, and the size of $\bigotimes_a\Delta_{\ell_a}$ is controlled by the mixed derivative
$\partial^{p}_{x_1}\cdots\partial^{p}_{x_k}f$, one factor per direction. Without that bound the
discarded terms need not be small and the truncation is not justified.

**What that means in practice.** Sparse grids work extremely well on smooth, near separable
functions, which is why they are standard in uncertainty quantification, where the response is
often a smooth function of a few dozen parameters. They degrade toward the full grid, or worse,
on functions with a kink or a sharp ridge that is not aligned with an axis, because such a
function has unbounded mixed derivatives even when it is continuous. Adaptive sparse grids, which
choose which index sets to include based on measured contribution, are the standard response.

### 5.2 Why Monte Carlo is dimension free, and what it gives up

**The error.** Estimate $I = \int_\Omega f$ by $\hat I_N = \frac{|\Omega|}{N}\sum_{i=1}^Nf(X_i)$
with $X_i$ uniform and independent on $\Omega$. Each term is an unbiased estimate, so
$\mathbb E[\hat I_N] = I$, and by independence the variances add:

$$
\operatorname{Var}(\hat I_N) = \frac{|\Omega|^2\operatorname{Var}(f(X))}{N},
\qquad
\text{rms error} = \frac{|\Omega|\,\sigma_f}{\sqrt N}
$$

**Why $k$ does not appear.** The variance $\sigma_f^2 = \int(f - \bar f)^2/|\Omega|$ is a single
number attached to $f$ and its domain. The derivation used only that the samples are independent
and identically distributed, and never that $\Omega$ is an interval, a square, or a cube. There is
no grid, no spacing $h$, and therefore no place for $k$ to enter. The central limit theorem gives
the same conclusion with the same $N^{-1/2}$ in every dimension.

Contrast the grid derivation of exercise 2.3, where $k$ enters at exactly one point: the count
$N \sim h^{-k}$. Monte Carlo has no $h$, so it has no exponent to inflate.

**What it gives up, four things.**

**The rate.** $N^{-1/2}$ is atrocious. One extra digit of accuracy costs 100 times the work, at
every $N$, forever. Simpson's rule gets a digit for $10^{1/4} \approx 1.8$ times the work in one
dimension. The trade is a bad rate that is immune to dimension against a good rate that is
destroyed by it.

**Smoothness is wasted.** The rate is $N^{-1/2}$ whether $f$ is analytic or merely square
integrable. Every other method in this course converts smoothness into accuracy, and Monte Carlo
converts none, which is why it is so much worse in one dimension.

**Determinism.** The answer changes between runs. Reporting a Monte Carlo result requires
reporting an error bar, and the error bar is itself an estimate. Exercise 4.2's measurement took
the median of nine repetitions for exactly this reason, and a single run would have been an
unreliable measurement of the crossover.

**The constant is $\sigma_f$, and it can be terrible.** The bound has no $\|f\|_\infty$ in it, but
a peaked integrand has a large variance and the practical cost follows. That is what importance
sampling, control variates, stratification and quasi Monte Carlo all attack: none of them change
the exponent, they all shrink the constant. Quasi Monte Carlo goes furthest, reaching
$O(N^{-1}(\log N)^k)$ on functions of bounded variation, which is another appearance of the same
$(\log N)^{k}$ that sparse grids have, and from the same source, a low discrepancy structure
replacing a product structure.

**The one line version.** Monte Carlo trades every advantage smoothness offers for immunity to
dimension, which is a terrible trade until $k$ is large enough that no smoothness based method can
afford a single grid.

### 5.3 The Haar condition

**Statement.** Let $\phi_1,\dots,\phi_m$ be continuous on $\Omega$. The family satisfies the
**Haar condition** on $\Omega$ if for every choice of $m$ distinct points
$p_1,\dots,p_m \in \Omega$,

$$
\det\big[\phi_j(p_i)\big] \ne 0
$$

Equivalently: every nonzero element of $\operatorname{span}\{\phi_j\}$ has at most $m-1$ zeros in
$\Omega$. Such a span is called a **Chebyshev system** or a **Haar space**.

**What it guarantees when it holds.**

**Unisolvent interpolation.** The interpolation problem has a unique solution for every data set
at every set of $m$ distinct nodes. No bad node sets, ever.

**A unique best approximation in the maximum norm.** For every continuous $f$ there is exactly
one element of the Haar space minimising $\|f - p\|_\infty$. Uniqueness is false without the Haar
condition, and it is the reason minimax approximation is a well posed problem at all.

**The equioscillation theorem.** $p^*$ is the best maximum norm approximation to $f$ from an
$m$ dimensional Haar space if and only if $f - p^*$ attains $\pm\|f-p^*\|_\infty$ alternately at
at least $m+1$ points. This is what lesson 47's minimax property of the Chebyshev polynomials is
an instance of, and it is what the Remez algorithm iterates on.

**Sign control.** The error of the interpolant has a predictable sign pattern between the nodes,
which is what makes the one dimensional error formula of lesson 46 possible in the form it takes.

**One dimensional families that satisfy it.**

- **Polynomials of degree $< m$** on any interval. A nonzero polynomial of degree $m-1$ has at
  most $m-1$ roots, which is the algebraic form of the condition, and the determinant is the
  Vandermonde $\prod_{i<j}(x_j-x_i)$.
- **The trigonometric system** $1, \cos t, \sin t, \dots, \cos nt, \sin nt$ on any half open
  interval of length $2\pi$, dimension $2n+1$. A nonzero trigonometric polynomial of degree $n$
  has at most $2n$ zeros in a period.
- **Exponentials** $e^{\lambda_1t},\dots,e^{\lambda_mt}$ with distinct real $\lambda_j$, on all of
  $\mathbb R$. The determinant is a generalised Vandermonde and is nonzero.
- **Powers** $t^{\alpha_1},\dots,t^{\alpha_m}$ with distinct real exponents, on $(0,\infty)$.
- **Rational functions with a fixed denominator**, $\{q(t)^{-1}, tq(t)^{-1},\dots\}$ for $q$
  positive on the interval, since dividing by a nonvanishing function preserves zero counts.

**How Mairhuber fits.** Mairhuber's theorem is exactly the statement that **no** family of $m \ge
2$ continuous functions satisfies the Haar condition on a domain in $\mathbb R^k$ for $k \ge 2$
containing an open set. So the Haar condition is a one dimensional phenomenon, and every guarantee
in the list above is lost in two dimensions along with it: no unisolvence, no unique best
approximation, no equioscillation.

That loss is the reason multivariate approximation looks so different from univariate
approximation, and the reason exercise 1.3's data dependent bases exist. It is also worth noticing
what survives: a **grid** restores unisolvence, not by satisfying the Haar condition, but by
restricting which point sets are allowed. Mairhuber's bad configurations are never on a Cartesian
grid, which is the precise sense in which section 1's easy case is easy.

---
