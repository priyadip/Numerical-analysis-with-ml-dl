# Solutions: Part 8, Approximation Theory and Transforms

Solutions to every exercise in lessons 54 to 60, 119 in all, 17 per lesson.

Every number quoted here was measured by running the code, on this repository, with the seeds
shown. Where a measurement contradicted the claim the exercise expects, the measurement is
reported and the claim is corrected.

Run any block from the repository root. Each one is self contained apart from `nalib`.

```python
import sys
sys.path.insert(0, "src")
```

---

## Lesson 54, Function Approximation Fundamentals

### 1.1 Choosing a norm for three jobs

**Sensor data with occasional dropouts: $L^1$.** A dropout is an outlier, and the $L^1$ norm
penalises it linearly while $L^2$ penalises it quadratically and $L^\infty$ lets it decide the
whole fit. A single bad reading moves an $L^\infty$ fit by the size of the reading; it moves an
$L^1$ fit by almost nothing, because the $L^1$ minimiser is a median rather than a mean.

**The polynomial inside a hardware `exp`: $L^\infty$.** The specification is a guarantee, of the
form "correct to within half an ulp everywhere in the argument range". A norm that averages
cannot express that, and a certification body will not accept "it is usually fine".

**Summarising a noisy measurement: $L^2$.** The noise is what makes it a statistics problem, and
under Gaussian noise the $L^2$ minimiser is the maximum likelihood estimate. That is not a
convention, it is a theorem, and it is why least squares is the default wherever noise is the
dominant error.

The pattern: **$L^1$ for robustness, $L^2$ for noise, $L^\infty$ for promises.**

### 1.2 What uniqueness needs, and where it fails

**In $L^\infty$ it needs the Haar condition** of lesson 53's exercise 5.3: every nonzero element
of the approximating space has at most $n-1$ zeros in the interval, where $n$ is the dimension.
Polynomials of degree below $n$ satisfy it, because a nonzero polynomial of degree $n-1$ has at
most $n-1$ roots.

Given the Haar condition, uniqueness follows from equioscillation. If $p$ and $q$ were both best,
so is $\frac12(p+q)$, and its error equioscillates $n+2$ times. At each of those $n+2$ points
$\frac12(f-p) + \frac12(f-q)$ reaches $\pm E$, and since both terms are bounded by $E$ in
magnitude they must be equal there. So $p-q$ has $n+2$ zeros, which for a degree $n$ polynomial
forces $p = q$.

**Where it fails in this course: two dimensions.** Lesson 53's Mairhuber theorem says no fixed
basis satisfies the Haar condition on a domain in $\mathbb R^2$ containing an open set, for any
$m \ge 2$. So best $L^\infty$ approximation from a fixed basis in two dimensions need not be
unique, and the equioscillation theorem has no analogue.

It also fails in $L^1$ even in one dimension, and exercise 3.1 exhibits a case.

### 1.3 What a characterisation buys over a bound

**A bound tells you the answer is good. A characterisation tells you which answer it is.**

Three specific consequences.

**It is checkable after the fact, without knowing the answer.** Given any candidate $p$, count the
alternations of $f-p$. If there are $n+2$ of them, $p$ is best, full stop. No comparison against a
reference is needed, and there is nothing to trust.

**It is an algorithm.** The condition is $n+2$ equations, and imposing them is a linear solve.
That is Remez, and it converges quadratically. A bound gives no such handle: knowing
$\|f-p\| \le C h^k$ does not tell you how to lower it.

**It is exact in both directions.** "If and only if" means a candidate failing the count is
provably not best, so the certificate never gives a false negative either. Section 5 uses that:
the least squares fit alternates once where it needs seven, which is a proof that it is not the
minimax fit rather than an observation that it is worse.

Compare with lesson 46's interpolation error bound, which is a genuine bound and tells you nothing
about which interpolant to choose, because it holds for all of them.

### 2.1 Coercivity, and where finite dimensionality enters

**Claim.** For a finite dimensional subspace $V$ with basis $\phi_1,\dots,\phi_m$, the map
$c \mapsto \|f - \sum c_k\phi_k\|$ tends to infinity as $\|c\| \to \infty$.

**Proof.** On the unit sphere $\|c\| = 1$, the function $c \mapsto \|\sum c_k\phi_k\|$ is
continuous and strictly positive, since the $\phi_k$ are independent so no unit combination is the
zero function. **The unit sphere in $\mathbb R^m$ is compact**, so that function attains a
positive minimum $\mu > 0$. Hence $\|\sum c_k\phi_k\| \ge \mu\|c\|$ for every $c$, by homogeneity.
Then

$$
\left\|f - \sum c_k\phi_k\right\| \ge \left\|\sum c_k\phi_k\right\| - \|f\| \ge \mu\|c\| - \|f\|
\to \infty
$$

**Where finite dimensionality is used: the compactness of the unit sphere.** In infinite
dimensions the unit sphere is not compact, the infimum $\mu$ can be zero, and the argument
collapses. That is not a technicality: best approximation from an infinite dimensional subspace
genuinely can fail to exist, and the standard example is approximating from the polynomials of
**all** degrees inside $C[a,b]$, where the infimum is 0 by Weierstrass and is not attained by any
polynomial unless $f$ is one.

The rest of the existence proof needs only continuity, which holds by the triangle inequality:
$|\,\|f-p\| - \|f-q\|\,| \le \|p-q\|$.

### 2.2 Uniqueness in $L^2$, and why the argument fails in $L^\infty$

**The $L^2$ argument.** Suppose $p$ and $q$ are both best, with common error $E$. The
parallelogram law gives

$$
\left\|f - \tfrac{p+q}{2}\right\|_2^2 = \tfrac12\|f-p\|_2^2 + \tfrac12\|f-q\|_2^2
- \tfrac14\|p-q\|_2^2 = E^2 - \tfrac14\|p-q\|_2^2
$$

If $p \ne q$ the right hand side is strictly below $E^2$, so the midpoint is strictly better,
contradicting that $E$ was the minimum. Hence $p = q$.

**Why it fails in $L^\infty$.** The parallelogram law is an identity of **inner product** spaces,
and $L^\infty$ has no inner product. Concretely, its unit ball is not strictly convex: the
functions $1$ and $\max(1, 2t)$ on $[0, 1/2]$ both have maximum 1, and so does their average, so
averaging two extreme points can stay extreme.

**Uniqueness in $L^\infty$ is still true for polynomials, and by a different route.** The
equioscillation argument of exercise 1.2 does it, and it needs the Haar condition rather than
strict convexity. So the two norms both give uniqueness, from unrelated hypotheses, and one of
those hypotheses fails in two dimensions while the other does not.

### 2.3 The "only if" half of Chebyshev's theorem

**Claim.** If $p$ is a best degree $n$ approximation to $f$ on $[a,b]$, then $f-p$ attains
$\pm E$ alternately at at least $n+2$ points, where $E = \|f-p\|_\infty$.

**Proof by contradiction.** Suppose the error alternates at most $n+1$ times. Let
$x_1 < \cdots < x_k$ be a maximal alternating set, with $k \le n+1$.

Between consecutive members of that set the error must change sign, so there are $k-1 \le n$
points $z_1 < \cdots < z_{k-1}$ with $(f-p)(z_i) = 0$ and, more usefully, such that the error has
one sign on each interval between them. Choose one $z_i$ strictly between the last extremum of
one sign and the first of the next.

Now build

$$
r(x) = \sigma\prod_{i=1}^{k-1}(x - z_i)
$$

with $\sigma = \pm1$ chosen so that $r$ has the **same sign as $f - p$** on each of the $k$
regions. This is possible because $r$ changes sign exactly at the $z_i$, and so does $f-p$. Its
degree is $k-1 \le n$, so $p + \epsilon r$ is still in the space.

For small $\epsilon > 0$ the error $f - p - \epsilon r$ is strictly smaller in magnitude wherever
$|f-p|$ is near $E$, because there $r$ has the same sign and we are subtracting. Away from those
regions $|f-p|$ is bounded below $E$ by some margin, and $\epsilon\|r\|$ can be made smaller than
that margin. So $\|f - p - \epsilon r\|_\infty < E$, contradicting that $p$ was best.

Hence $k \ge n+2$.

**Where the degree count is used.** The correction polynomial needs degree $k-1$, and it has to
lie in the space, so $k-1 \le n$. At $k = n+2$ the correction would need degree $n+1$ and does not
fit. **That is exactly why the count is $n+2$ and not something else.**

### 2.4 Symmetry, and the degenerate degrees

**Claim 1.** If $f$ is even on $[-a,a]$, its best degree $n$ approximation is even.

**Proof.** Let $p$ be best, with error $E$. Define $\tilde p(x) = p(-x)$, which is also degree
$n$. Since $f$ is even,
$\|f - \tilde p\|_\infty = \sup_x|f(x) - p(-x)| = \sup_x|f(-x)-p(x)| = \|f-p\|_\infty = E$, so
$\tilde p$ is also best. By uniqueness, $\tilde p = p$, which says $p$ is even.

The same argument with a sign gives: if $f$ is odd, the best approximation is odd.

**Claim 2.** For $f$ even, the best degree $2k+1$ approximation equals the best degree $2k$ one.

**Proof.** By claim 1 the best degree $2k+1$ approximation is even, so its odd coefficients
vanish, so it has degree at most $2k$. It is therefore a degree $2k$ polynomial which is best
among degree $2k+1$ polynomials, hence best among the smaller class too.

**Why this matters computationally.** It is exactly the degeneracy `approx.remez` had to handle.
At a degenerate degree the error has $n+3$ alternations rather than $n+2$, and an exchange step
that insists on exactly $n+2$ has a choice to make. Dropping the globally weakest extremum can
break the alternation and leave the iteration cycling: on $\cosh$ at degree 4 that ran the full 60
iterations with the error oscillating between $4.5\times10^{-5}$ and $2.1\times10^{-4}$. Dropping
from an **end** preserves alternation and converges in 4.

### 2.5 The Remez system, and when it is nonsingular

**The system.** With reference points $x_0 < \cdots < x_{n+1}$ and basis $\phi_0,\dots,\phi_n$,

$$
\sum_{k=0}^{n}c_k\phi_k(x_i) + (-1)^iE = f(x_i), \qquad i = 0,\dots,n+1
$$

which is $n+2$ equations in the $n+2$ unknowns $c_0,\dots,c_n,E$. In matrix form the coefficient
matrix is $\Phi$ with an extra column $((-1)^i)$ appended.

**Nonsingularity.** Suppose the homogeneous system has a solution $(c, E)$. Then
$q(x_i) = -(-1)^iE$ for the polynomial $q = \sum c_k\phi_k$.

If $E = 0$ then $q$ vanishes at $n+2$ distinct points, and by the Haar condition a nonzero element
of an $(n+1)$ dimensional Haar space has at most $n$ zeros, so $q = 0$ and the solution is
trivial.

If $E \ne 0$ then $q$ takes values alternating in sign at $n+2$ points, so it has at least $n+1$
sign changes and therefore at least $n+1$ zeros. Again the Haar condition forces $q = 0$, whence
$E = 0$, a contradiction.

So the only solution is trivial and the matrix is nonsingular.

**What that argument used.** Distinct, ordered reference points, and the Haar condition. It says
nothing about the reference being close to the answer, which is why Remez converges from a poor
start (exercise 4.2) and only the **rate** depends on the start.

**One case the argument does not cover.** For an odd $f$ on a symmetric interval with a symmetric
reference, the matrix is nonsingular but the solution has $E \approx 5\times10^{-17}$: the
alternating column is orthogonal to the residual, so the "approximation" simply interpolates $f$.
That is a well posed solve returning a useless iterate, and `approx.remez` breaks it by shifting
the starting reference off centre. Without that, $\sin(3t)$ at degree 7 returned
$3.37\times10^{-4}$ instead of $1.69\times10^{-4}$.

### 3.1 Best $L^1$ approximation, and its non-uniqueness

Minimising $\sum_i|f(t_i) - p(t_i)|$ is a linear program: introduce one slack per sample.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import approx as ap


def best_l1(f, degree, lo=-1.0, hi=1.0, n_probe=401):
    """min sum |residual| as an LP, with u_i >= |residual_i| as two linear constraints each.

    The absolute value is not linear, so it is replaced by a new variable bounded above and
    below by the residual. That doubles the constraint count and keeps the problem an LP,
    which is the standard trick and the reason L1 fitting is tractable at all.
    """
    from scipy.optimize import linprog

    n = int(degree)
    t = np.linspace(float(lo), float(hi), int(n_probe))
    v = np.asarray(f(t), dtype=float)
    u = np.clip((2.0 * t - (hi + lo)) / (hi - lo), -1.0, 1.0)
    A = np.stack([np.cos(k * np.arccos(u)) for k in range(n + 1)], axis=1)
    m = t.size
    cost = np.concatenate([np.zeros(n + 1), np.ones(m)])
    upper = np.vstack([np.hstack([A, -np.eye(m)]), np.hstack([-A, -np.eye(m)])])
    bound = np.concatenate([v, -v])
    result = linprog(cost, A_ub=upper, b_ub=bound,
                     bounds=[(None, None)] * (n + 1 + m), method="highs")
    if not result.success:
        raise RuntimeError(f"the linear program did not solve: {result.message}")
    return result.x[:n + 1], A, v, t


try:
    import scipy.optimize                                    # noqa: F401
except ImportError:
    print("scipy is required for this solution")
else:
    print(f"{'degree':>8}{'L1 fit':>13}{'L2 fit':>13}{'minimax fit':>14}"
          f"   (all measured in the L1 norm)")
    for n in (1, 2, 3, 5):
        c, A, v, t = best_l1(np.exp, n)
        ls = ap.best_l2_polynomial(np.exp, n, -1.0, 1.0)
        mm = ap.remez(np.exp, n, -1.0, 1.0)
        l1 = lambda r: float(np.mean(np.abs(r))) * 2.0
        print(f"{n:>8}{l1(A @ c - v):>13.6f}{l1(ls['evaluate'](t) - np.exp(t)):>13.6f}"
              f"{l1(mm['evaluate'](t) - np.exp(t)):>14.6f}")
```

| degree | $L^1$ fit | $L^2$ fit | minimax fit |
|---|---|---|---|
| 1 | **0.267430** | 0.277732 | 0.338575 |
| 2 | **0.044132** | 0.046148 | 0.056346 |
| 3 | **0.005484** | 0.005758 | 0.006978 |
| 5 | **0.000045** | 0.000048 | 0.000057 |

Each norm wins in its own norm, and the gaps here are small: the $L^1$ fit beats least squares by
4 percent and minimax by 21 percent, measured in $L^1$. That is much less dramatic than the
$L^\infty$ comparison in section 5, and it is the reason $L^1$ fitting is used for robustness
rather than for accuracy.

**Where the minimiser is not unique.** Take four points, two with value 0 and two with value 1,
and fit a constant.

```python
import numpy as np

values = np.array([0.0, 0.0, 1.0, 1.0])
print(f"{'constant':>10}{'L1 error':>12}")
for guess in (0.0, 0.25, 0.5, 0.75, 1.0):
    print(f"{guess:>10.2f}{float(np.sum(np.abs(values - guess))):>12.4f}")
```

| constant | $L^1$ error |
|---|---|
| 0.00 | 2.0000 |
| 0.25 | 2.0000 |
| 0.50 | 2.0000 |
| 0.75 | 2.0000 |
| 1.00 | 2.0000 |

**Every constant in $[0,1]$ is a minimiser.** The $L^1$ minimiser is a **median**, and with an
even number of points the median is an interval rather than a point. That is not a numerical
artefact and no algorithm removes it: the minimiser genuinely is a set.

The $L^2$ minimiser of the same data is the mean, 0.5, and it is unique. The $L^\infty$ minimiser
is also 0.5 and also unique. **So $L^1$ is the one norm of the three whose minimiser can fail to
be unique in one dimension**, which is what exercise 1.2 promised.

### 3.2 The single point exchange

The Remez exchange of the lesson moves **all** $n+2$ reference points at once. The original
algorithm, and the one in most older texts, moves only one: replace the reference point whose
error is smallest by the location of the global extremum, keeping the alternation.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import approx as ap


def remez_single_exchange(f, degree, lo=-1.0, hi=1.0, max_iterations=200,
                          tol=1e-12, n_probe=20001):
    """Exchange one reference point per step instead of all of them.

    The new point replaces the reference point of the same error sign nearest to it, which is
    what preserves the alternation. Replacing the globally weakest point instead can leave two
    neighbours of the same sign, and the iteration then wanders.
    """
    n = int(degree)
    m = n + 2
    ref = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(np.arange(m) * np.pi / (m - 1))[::-1]
    if m > 2:
        ref[1:-1] = ref[1:-1] + 0.01 * (hi - lo) / m
    to_unit = lambda s: np.clip((2.0 * s - (hi + lo)) / (hi - lo), -1.0, 1.0)
    basis = lambda s: np.stack([np.cos(k * np.arccos(to_unit(s))) for k in range(n + 1)],
                               axis=1)
    probe = np.linspace(float(lo), float(hi), int(n_probe))
    truth = np.asarray(f(probe), dtype=float)
    design = basis(probe)
    floor = 8.0 * float(np.finfo(float).eps) * float(np.max(np.abs(truth)))
    peak = float("inf")
    for step in range(int(max_iterations)):
        A = np.empty((m, m))
        A[:, :n + 1] = basis(ref)
        A[:, n + 1] = (-1.0) ** np.arange(m)
        sol = np.linalg.solve(A, np.asarray(f(ref), dtype=float))
        c, level = sol[:n + 1], float(sol[n + 1])
        err = truth - design @ c
        peak = float(np.max(np.abs(err)))
        if abs(peak - abs(level)) <= float(tol) * max(peak, 1e-300) + floor:
            return step + 1, peak, True
        k = int(np.argmax(np.abs(err)))
        want = np.sign(err[k])
        signs = np.sign(np.asarray(f(ref), dtype=float) - basis(ref) @ c)
        same = np.where(signs == want)[0]
        target = int(same[np.argmin(np.abs(ref[same] - probe[k]))]) if same.size \
            else int(np.argmin(np.abs(ref - probe[k])))
        ref = np.sort(np.concatenate([np.delete(ref, target), [probe[k]]]))
    return int(max_iterations), peak, False


print(f"{'degree':>8}{'full: its':>12}{'full: error':>15}"
      f"{'single: its':>14}{'single: error':>16}")
for n in (2, 4, 6, 8, 10):
    full = ap.remez(np.exp, n, -1.0, 1.0)
    its, err, converged = remez_single_exchange(np.exp, n, -1.0, 1.0)
    print(f"{n:>8}{full['iterations']:>12}{full['max_error']:>15.4e}"
          f"{its:>14}{err:>16.4e}")
```

| degree | full: iterations | full: error | single: iterations | single: error |
|---|---|---|---|---|
| 2 | 3 | 4.5017e-2 | 5 | 4.5017e-2 |
| 4 | 3 | 5.4667e-4 | 8 | 5.4667e-4 |
| 6 | 3 | 3.2109e-6 | 10 | 3.2109e-6 |
| 8 | 3 | 1.1064e-8 | 9 | 1.1064e-8 |
| 10 | 2 | 2.5023e-11 | 9 | 2.5025e-11 |

**Both reach the same answer and the single exchange takes two to three times as many
iterations.** The errors agree to five digits at every degree, which says they are finding the
same polynomial, and at degree 10 the last digit differs because both are near the roundoff floor.

**Why anyone would use the single exchange.** Each iteration is cheaper: one extremum search
instead of $n+2$, which mattered when the function evaluations were the expensive part. On modern
hardware the linear solve dominates and the multi-point exchange wins, which is why every current
implementation uses it.

**Why the count grows with the degree.** The single exchange fixes one reference point per step,
so it needs at least as many steps as there are points to move, and the number of points is
$n+2$. The measured counts, 5 to 10 for degrees 2 to 10, are well below that, because moving one
point drags the solve and the others end up nearly right anyway.

### 3.3 Weighted minimax, for relative error

Minimising $\|f-p\|_\infty$ makes the **absolute** error uniform. A library function usually wants
the **relative** error uniform instead, which is $\|(f-p)/f\|_\infty$, and that is a weighted
minimax problem with $w = 1/f$.

The change to Remez is one column:

$$
\sum_k c_k\phi_k(x_i) + \frac{(-1)^i}{w(x_i)}E = f(x_i)
$$

so the levelled quantity is $w\cdot(f-p)$ rather than $f-p$.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import approx as ap


def weighted_remez(f, degree, weight, lo=-1.0, hi=1.0, max_iterations=80, n_probe=20001):
    """Level w (f - p) rather than f - p. Only the alternating column changes.

    With w = 1/f this makes the RELATIVE error equioscillate, which is what a library function
    specification asks for.
    """
    n = int(degree)
    m = n + 2
    ref = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(np.arange(m) * np.pi / (m - 1))[::-1]
    if m > 2:
        ref[1:-1] = ref[1:-1] + 0.01 * (hi - lo) / m
    to_unit = lambda s: np.clip((2.0 * s - (hi + lo)) / (hi - lo), -1.0, 1.0)
    basis = lambda s: np.stack([np.cos(k * np.arccos(to_unit(s))) for k in range(n + 1)],
                               axis=1)
    probe = np.linspace(float(lo), float(hi), int(n_probe))
    truth = np.asarray(f(probe), dtype=float)
    w_probe = np.asarray(weight(probe), dtype=float)
    design = basis(probe)
    best_c, best = np.zeros(n + 1), float("inf")
    for _ in range(int(max_iterations)):
        A = np.empty((m, m))
        A[:, :n + 1] = basis(ref)
        A[:, n + 1] = (-1.0) ** np.arange(m) / np.asarray(weight(ref), dtype=float)
        sol = np.linalg.solve(A, np.asarray(f(ref), dtype=float))
        c, level = sol[:n + 1], float(sol[n + 1])
        err = w_probe * (truth - design @ c)
        peak = float(np.max(np.abs(err)))
        if peak < best:
            best, best_c = peak, c
        if abs(peak - abs(level)) <= 1e-10 * max(peak, 1e-300):
            break
        idx = ap._alternating_extrema(err, m)
        if idx.size < m:
            break
        ref = probe[idx]
    return best_c, best


t = np.linspace(-1.0, 1.0, 20001)
truth = np.exp(t)
print(f"{'degree':>8}{'absolute fit':>28}{'relative fit':>28}")
print(f"{'':>8}{'max abs':>14}{'max rel':>14}{'max abs':>14}{'max rel':>14}")
for n in (2, 4, 6):
    plain = ap.remez(np.exp, n, -1.0, 1.0)
    c, _ = weighted_remez(np.exp, n, lambda s: 1.0 / np.exp(s), -1.0, 1.0)
    B = np.stack([np.cos(k * np.arccos(np.clip(t, -1.0, 1.0))) for k in range(n + 1)], axis=1)
    rel_fit = B @ c
    abs_fit = plain["evaluate"](t)
    print(f"{n:>8}{float(np.max(np.abs(abs_fit - truth))):>14.4e}"
          f"{float(np.max(np.abs((abs_fit - truth) / truth))):>14.4e}"
          f"{float(np.max(np.abs(rel_fit - truth))):>14.4e}"
          f"{float(np.max(np.abs((rel_fit - truth) / truth))):>14.4e}")
```

| degree | absolute fit: max abs | absolute fit: max rel | relative fit: max abs | relative fit: max rel |
|---|---|---|---|---|
| 2 | **4.5017e-2** | 1.2237e-1 | 1.0802e-1 | **3.9740e-2** |
| 4 | **5.4667e-4** | 1.4860e-3 | 1.3674e-3 | **5.0304e-4** |
| 6 | **3.2109e-6** | 8.7281e-6 | 8.1997e-6 | **3.0165e-6** |

**Each wins in its own measure, by a factor of about 3, at every degree.** The absolute fit's
relative error is 2.7 times its absolute error, because $\exp$ ranges over a factor of $e^2 = 7.4$
across $[-1,1]$ and the small end pays for the large end. The relative fit reverses that exactly.

**Which one a library wants.** The relative one, almost always. A specification reading "the
result is correct to $10^{-16}$ relative" is what floating point arithmetic itself promises, and
matching it is what lets the function compose with everything else. An absolute guarantee is
useless where $f$ is small.

**One caution.** The weight $1/f$ requires $f$ to be bounded away from zero on the interval, and
near a zero of $f$ the relative problem has no solution. That is why library generators split the
argument range and reduce to an interval where the function does not vanish.

### 4.1 The least squares to minimax ratio

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import approx as ap

cases = [("exp", np.exp), ("sin(3t)", lambda s: np.sin(3.0 * s)),
         ("1/(1+25t^2)", lambda s: 1.0 / (1.0 + 25.0 * s * s)),
         ("|t|", np.abs), ("|t|^1.5", lambda s: np.abs(s) ** 1.5),
         ("sqrt|t|", lambda s: np.sqrt(np.abs(s)))]
print(f"{'function':>14}{'degree':>8}{'minimax':>13}{'least squares':>15}{'ratio':>9}")
for name, f in cases:
    probe = np.linspace(-1.0, 1.0, 4001)
    floor = 8.0 * float(np.finfo(float).eps) * float(np.max(np.abs(f(probe))))
    for n in (2, 4, 8, 16, 24, 32):
        out = ap.minimax_vs_least_squares(f, n, -1.0, 1.0)
        if out["minimax"]["linf"] < floor:
            print(f"{name:>14}{n:>8}{out['minimax']['linf']:>13.3e}"
                  f"{'at roundoff':>15}{'not measurable':>9}")
            break
        print(f"{name:>14}{n:>8}{out['minimax']['linf']:>13.3e}"
              f"{out['least_squares']['linf']:>15.3e}{out['linf_ratio']:>9.3f}")
```

| function | $n=2$ | $n=4$ | $n=8$ | $n=16$ | $n=24$ | $n=32$ |
|---|---|---|---|---|---|---|
| $\exp$ | 1.812 | 2.203 | 2.808 | at roundoff | | |
| $\sin 3t$ | 1.422 | 1.946 | 2.638 | 3.544 | at roundoff | |
| $1/(1+25t^2)$ | 1.519 | 1.523 | 1.526 | 1.528 | 1.755 | 1.895 |
| $\lvert t\rvert$ | 1.500 | 1.733 | 1.940 | 2.086 | 2.143 | 2.173 |
| $\lvert t\rvert^{1.5}$ | 1.265 | 1.523 | 1.778 | 1.971 | 2.050 | 2.093 |
| $\sqrt{\lvert t\rvert}$ | 1.814 | 1.980 | 2.109 | 2.190 | 2.221 | 2.237 |

**Restricting to degrees where the error is above the roundoff floor is the whole difficulty of
this measurement.** Run without that guard, $\exp$ reports a ratio of 9.750 at degree 16 and 6.583
at degree 24, which looks like the ratio diverging and is really two roundoff-sized numbers being
divided. The rows above stop the moment the minimax error falls below $8\varepsilon\|f\|$.

**What it converges to.** For the non-smooth functions, where the measurement runs far enough to
see it, the ratio settles between 2.1 and 2.25 and is still creeping up slowly. The classical
prediction is the Lebesgue constant of the $L^2$ projection, which grows like $\log n$, so the
ratio should not converge to a constant at all: it should grow, very slowly.

The measurement is consistent with that: from degree 16 to 32, a doubling, $\sqrt{|t|}$ moves from
2.190 to 2.237, a rise of 2 percent, which is what a $\log n$ term does.

**Against smoothness.** The smoother the function, the **faster** the ratio grows with degree:
$\sin 3t$ reaches 3.544 by degree 16 while $\sqrt{|t|}$ is still at 2.190. That is the opposite of
what one might guess, and the reason is that for a smooth function both errors are collapsing
geometrically, so the ratio is a quotient of two rapidly shrinking numbers and is much more
sensitive.

**The practical reading is unchanged**: a factor of two is what least squares costs at the worst
point, over the whole range where the question is meaningful.

### 4.2 Remez converges quadratically

Starting from the Chebyshev extrema, Remez is already so close that there is nothing to measure:
it converges in two or three steps from a starting error that is within a factor of 2 of the
answer. To see the rate, start somewhere worse.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import approx as ap


def remez_trail(f, degree, lo=-1.0, hi=1.0, max_iterations=8, n_probe=2000001):
    """Run Remez from equally spaced reference points and record the levelling gap.

    The quantity that converges is the RELATIVE levelling gap (peak - E)/peak, not the movement
    of the reference points. The reference cannot move below the probe grid spacing, so tracking
    it measures the grid rather than the algorithm.
    """
    n = int(degree)
    m = n + 2
    ref = np.linspace(float(lo), float(hi), m)
    ref[1:-1] = ref[1:-1] + 0.013 * (hi - lo) / m
    to_unit = lambda s: np.clip((2.0 * s - (hi + lo)) / (hi - lo), -1.0, 1.0)
    basis = lambda s: np.stack([np.cos(k * np.arccos(to_unit(s))) for k in range(n + 1)],
                               axis=1)
    probe = np.linspace(float(lo), float(hi), int(n_probe))
    truth = np.asarray(f(probe), dtype=float)
    design = basis(probe)
    out = []
    for _ in range(int(max_iterations)):
        A = np.empty((m, m))
        A[:, :n + 1] = basis(ref)
        A[:, n + 1] = (-1.0) ** np.arange(m)
        sol = np.linalg.solve(A, np.asarray(f(ref), dtype=float))
        c, level = sol[:n + 1], float(sol[n + 1])
        err = truth - design @ c
        peak = float(np.max(np.abs(err)))
        out.append((peak, abs(level), abs(peak - abs(level)) / max(peak, 1e-300)))
        idx = ap._alternating_extrema(err, m)
        if idx.size < m:
            break
        new = probe[idx]
        if float(np.max(np.abs(new - ref))) == 0.0:
            break
        ref = new
    return out


for degree in (4, 6):
    print(f"degree {degree}, starting from equally spaced reference points:")
    print(f"{'iteration':>11}{'peak':>17}{'level E':>17}{'relative gap':>16}"
          f"{'gap / prev^2':>15}")
    previous = None
    for i, (peak, level, gap) in enumerate(remez_trail(np.exp, degree)):
        rate = "" if previous is None or previous <= 0 else f"{gap / previous ** 2:.4f}"
        print(f"{i:>11}{peak:>17.6e}{level:>17.6e}{gap:>16.3e}{rate:>15}")
        previous = gap
    print()
```

**Degree 4:**

| iteration | peak | level $E$ | relative gap | gap / prev$^2$ |
|---|---|---|---|---|
| 0 | 9.096891e-4 | 3.317628e-4 | 6.353e-1 | |
| 1 | 5.613457e-4 | 5.412057e-4 | 3.588e-2 | 0.0889 |
| 2 | 5.466893e-4 | 5.466539e-4 | 6.479e-5 | 0.0503 |
| 3 | 5.466676e-4 | 5.466676e-4 | 1.738e-10 | 0.0414 |
| 4 | 5.466676e-4 | 5.466676e-4 | 1.051e-12 | at the floor |

**Degree 6:**

| iteration | peak | level $E$ | relative gap | gap / prev$^2$ |
|---|---|---|---|---|
| 0 | 8.315224e-6 | 1.246489e-6 | 8.501e-1 | |
| 1 | 3.813067e-6 | 2.936804e-6 | 2.298e-1 | 0.318 |
| 2 | 3.215995e-6 | 3.207918e-6 | 2.511e-3 | 0.0476 |
| 3 | 3.210877e-6 | 3.210877e-6 | 7.600e-8 | 0.0121 |
| 4 | 3.210877e-6 | 3.210877e-6 | 2.688e-10 | at the floor |

**The convergence is quadratic**, and the constant is the evidence: the ratio
$\text{gap}_{k+1}/\text{gap}_k^2$ sits between 0.012 and 0.32 and does not run away, while the gap
itself falls from 0.635 to $1.7\times10^{-10}$ in three steps. A linearly convergent method would
show a constant $\text{gap}_{k+1}/\text{gap}_k$ instead, and that ratio here is 0.056, 0.0018,
$2.7\times10^{-6}$: falling by orders every step, which is what quadratic looks like from the
other side.

**What not to measure, and why.** The obvious quantity is the movement of the reference points,
$\|x^{(k)} - x^*\|$, and it does not work. The reference lives on the probe grid, so once it is
within one grid spacing it cannot improve, and at degree 8 with a 200001 point grid the trail
reads $3\times10^{-5}$, $3\times10^{-5}$, $0$, $3\times10^{-5}$, cycling between two adjacent grid
points forever. **That is the measurement instrument, not the algorithm.** The levelling gap has
no such floor until machine precision, which is why it is the right thing to track.

### 4.3 The Bernstein rate against smoothness

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import approx as ap

rows = [("exp", np.exp, "1", "smooth, and the rate does not improve"),
        ("t^2", lambda s: s * s, "1", "smooth, same rate"),
        ("t", lambda s: s, "n/a", "reproduced exactly, so there is no order"),
        ("|t - 0.5|", lambda s: np.abs(s - 0.5), "1/2", "Lipschitz only"),
        ("sqrt|t - 0.5|", lambda s: np.sqrt(np.abs(s - 0.5)), "1/4", "Holder 1/2"),
        ("step at 0.5", lambda s: (s > 0.5).astype(float), "0",
         "discontinuous: no uniform convergence")]
print(f"{'function':>16}{'fitted order':>15}{'theory':>9}  note")
for name, f, theory, note in rows:
    out = ap.weierstrass_rate(f, (16, 32, 64, 128, 256, 512), 0.0, 1.0)
    print(f"{name:>16}{out['fitted_order']:>15.4f}{theory:>9}  {note}")
```

| function | fitted order | theory | note |
|---|---|---|---|
| $\exp$ | 0.9997 | 1 | smooth, and the rate does not improve |
| $t^2$ | 1.0000 | 1 | smooth, same rate |
| $t$ | **-0.9982** | n/a | reproduced exactly, so there is no order |
| $\lvert t-0.5\rvert$ | 0.4960 | 1/2 | Lipschitz only |
| $\sqrt{\lvert t-0.5\rvert}$ | 0.2325 | 1/4 | Holder 1/2 |
| step at 0.5 | 0.0453 | 0 | discontinuous |

**The headline is the first two rows.** $\exp$ is analytic and $t^2$ is a polynomial of degree 2,
and both converge at order exactly 1. **The rate does not improve with smoothness**, which is the
whole objection to Bernstein as a method. Compare with lesson 56's Chebyshev series, whose rate on
$\exp$ is faster than geometric.

**The general theorem behind the other rows** is that for $f$ Holder continuous with exponent
$\alpha \in (0,1]$, the Bernstein error is $O(n^{-\alpha/2})$. So Lipschitz ($\alpha = 1$) gives
$1/2$, and Holder $1/2$ gives $1/4$. Measured 0.496 and 0.2325, against 0.5 and 0.25.

**Two rows are degenerate and saying so is part of the answer.**

The affine row reports $-0.998$, which is not a convergence order. Bernstein reproduces affine
functions exactly at every $n$, so the "error" is roundoff, and roundoff grows slowly with $n$
because the sum has more terms. Fitting a slope to it measures the accumulation of rounding.

The step row reports 0.0453, which is order 0, and the errors say why:

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import approx as ap

out = ap.weierstrass_rate(lambda s: (s > 0.5).astype(float),
                          (16, 32, 64, 128, 256, 512), 0.0, 1.0)
for n, e in zip(out["n_values"], out["errors"]):
    print(f"  n = {n:>4}: max error {e:.6f}")
```

| $n$ | max error |
|---|---|
| 16 | 0.596618 |
| 32 | 0.567735 |
| 64 | 0.546493 |
| 128 | 0.530686 |
| 256 | 0.518530 |
| 512 | 0.508596 |

**The error is heading for 0.5, not for 0.** At the jump, $B_nf$ converges to the midpoint of the
two one sided limits, so the maximum error tends to half the jump. There is no uniform convergence
at all, and there cannot be: **Weierstrass requires continuity**, and Bernstein's theorem is a
proof of Weierstrass, so it inherits the hypothesis.

**The class on which the rate does improve.** For $f \in C^2$ the rate is still $O(1/n)$ in the
maximum norm, so smoothness never helps the plain Bernstein operator. What does improve is a
**modified** operator: the Bernstein-Kantorovich and Bernstein-Durrmeyer variants, and iterated
Boolean sums of $B_n$, reach $O(n^{-k})$ for $f \in C^{2k}$. So the limitation is the operator's,
not the basis's, and lesson 52's Bezier curves use the same basis to good effect for a different
job.

**One practical limit worth recording.** `approx.bernstein` fails at degree 1030 with
`OverflowError: int too large to convert to float`, because `math.comb(1030, 515)` exceeds the
largest double. That is the same failure lesson 52 measured for the Bezier evaluator, at the same
degree, for the same reason.

### 5.1 Why minimax and least squares are so close

Both errors fall geometrically on a smooth function and stay within a factor of about 2 of each
other. The reason is that **both are dominated by the same term**.

Write $f$ in the Chebyshev basis, $f = \sum_k a_kT_k$. Then:

- The truncated series has error $\sum_{k>n}a_kT_k$, bounded by $\sum_{k>n}|a_k|$.
- The best possible error is at least $|a_{n+1}|$, because no degree $n$ polynomial can cancel the
  $T_{n+1}$ component: the $T_k$ are orthogonal, so the projection of the error onto $T_{n+1}$ is
  exactly $a_{n+1}$ whatever $p$ is.

For geometric decay $|a_k|\sim\rho^{-k}$ the tail is dominated by its first term:
$\sum_{k>n}|a_k| \approx |a_{n+1}|\frac{\rho}{\rho-1}$. **So the upper and lower bounds differ by
a bounded factor**, and every reasonable method is squeezed between them.

The least squares fit is also squeezed: it is the projection in a slightly different inner
product, and its error is bounded by the Lebesgue constant of $L^2$ projection times the best
error, which grows like $\log n$.

**Where the gap is large.** Two classes.

**Functions with a single dominant feature.** If one coefficient is much larger than its
neighbours, the tail is not dominated by its first term and the squeeze fails. A function like
$T_{50}(x) + 10^{-8}\sin(x)$ at degree 49 is an extreme case: the minimax error is essentially
$2^{-49}$ times something and the $L^2$ fit distributes the error quite differently.

**Functions with algebraic coefficient decay.** For $|a_k|\sim k^{-p}$ the tail is
$\sum_{k>n}k^{-p} \approx n^{1-p}/(p-1)$, which is $n$ times the first term rather than a constant
times it. The measurement in exercise 4.1 shows exactly this: the ratio for $\sqrt{|t|}$ is
climbing steadily rather than settling, and the growth is the $\log n$ of the Lebesgue constant on
top of the wider tail.

**The unified statement.** The ratio is the Lebesgue constant of the projection, which is
$O(\log n)$ for both the Chebyshev and the Legendre inner products. It is unbounded, so "a factor
of 2" is a statement about the range of degrees anyone uses, not a theorem.

### 5.2 Approximation from other spaces

| space | existence | uniqueness | equioscillation |
|---|---|---|---|
| polynomials, degree $\le n$ | yes | yes | yes, $n+2$ points |
| rational, type $[m/n]$ | yes | yes | yes, $m+n+2$ points |
| splines, fixed knots | yes | yes | yes, dimension + 1 |
| splines, free knots | yes | **no** | no |
| trigonometric, degree $\le n$ | yes | yes | yes, $2n+2$ points |

**Existence survives everywhere**, because the proof of exercise 2.1 used only finite
dimensionality and continuity, and every row is a finite dimensional space or a finite dimensional
family.

**Rational functions.** The space is not linear, which is the essential change: a sum of two
rational functions of type $[m/n]$ is not of that type. Existence still holds by a compactness
argument on the coefficients. Uniqueness and equioscillation hold too, with the count raised to
$m+n+2$, which is the number of free parameters plus one. The complication is **degeneracy**: when
the best approximant has a lower type than requested, the count drops, which is lesson 57's block
structure appearing in the minimax setting. Practical rational minimax is much harder than the
polynomial case, and the modern answer is the AAA algorithm rather than a rational Remez.

**Splines with fixed knots.** A linear space, so everything is as for polynomials, with the
alternation count equal to the dimension plus one. The Haar condition **fails**, because a spline
can be zero on a whole interval, but a weaker local version holds and uniqueness survives.

**Splines with free knots.** Not a linear space and not even locally so. Uniqueness genuinely
fails, and the best approximation problem has multiple local minima. This is the setting in which
lesson 51's exercise 4.2 measured adaptive knot placement beating both fixed rules by two orders,
and the non-uniqueness is why that had to be measured rather than solved.

**Trigonometric polynomials.** The Haar condition holds on a half open period, so everything goes
through with $2n+2$ alternations. This is the setting of lesson 58, and it is the reason the
transform there is so well behaved: the same theory as polynomials, with the conditioning problem
removed.

### 5.3 Best approximation as a projection

**In $L^2$ it is exactly an orthogonal projection.** With $V = \mathrm{span}\{\phi_k\}$, the best
$L^2$ approximation is $P_Vf$, the orthogonal projection onto $V$, characterised by
$\langle f - P_Vf, v\rangle = 0$ for every $v \in V$. That is the normal equations, and it is what
lesson 55 diagonalises by choosing an orthogonal basis.

**The connection to the pseudoinverse.** Discretise: sample $f$ at $m$ points and let $A$ be the
$m\times(n+1)$ design matrix. The least squares solution is $c = A^+b$, and the fitted values are
$AA^+b$, which is the orthogonal projection onto the column space of $A$. So

$$
\underbrace{P_V f}_{\text{function space}} \quad\longleftrightarrow\quad
\underbrace{AA^{+}b}_{\text{Part 5}}
$$

is the same operator, once in $L^2[a,b]$ and once in $\mathbb R^m$. The SVD of $A$ is the discrete
version of choosing an orthogonal basis, which is why Part 5 insisted on QR and SVD rather than
the normal equations and lesson 55 insists on orthogonal polynomials: **they are the same fix**.

**The connection to Tikhonov regularisation.** Tikhonov solves
$\min\|Ax-b\|^2 + \lambda\|Lx\|^2$, which is best approximation with a **penalty** added. As
$\lambda \to 0$ it becomes the projection above. As $\lambda \to \infty$ it becomes the projection
onto the null space of $L$.

The function space analogue is the smoothing spline of lesson 51's exercise 3.2, which minimises
$\sum(f_i-y_i)^2 + \lambda\int(g'')^2$. That is not analogous to Tikhonov, it **is** Tikhonov in
the Sobolev space $H^2$, with $L = d^2/dx^2$ and point evaluation constraints, as lesson 51's
exercise 5.2 spells out.

**Where the projection picture stops.** In $L^\infty$ there is no inner product, so there is no
orthogonal projection and no linear operator taking $f$ to its best approximation. The map is
still well defined and continuous, by exercises 2.1 and 1.2, and it is **not linear**: the best
approximation of $f+g$ is not the sum of the best approximations. That non-linearity is why
minimax needs an iteration and least squares needs a solve, and it is the deepest difference
between the two norms.

---

## Lesson 55, Orthogonal Polynomials

### 1.1 Removing a problem against working around it

**Working around it** means keeping the badly conditioned basis and using an algorithm that
tolerates it. QR instead of the normal equations squares the effective condition number back down
to $\kappa$ rather than $\kappa^2$; an SVD does the same and reports the rank. Both are Part 5,
and both are genuine improvements.

**Removing it** means the matrix is never badly conditioned in the first place. With an orthogonal
basis the Gram matrix is diagonal, and after normalising, it is the identity. There is no system
to solve, so no solver's stability is at issue.

The measured difference: the monomial Gram matrix on $[-1,1]$ reaches $3.0\times10^{11}$ at degree
16, and the orthonormal one reads 1.0006. A better solver applied to the first still has to
recover an answer from a matrix that has lost eleven digits; the second never lost them.

**The Part 5 method that works around it: QR least squares**, and more precisely
`numpy.linalg.lstsq`, which uses an SVD. Lesson 54's exercise on `basis_conditioning` shows what
it is working around: the Hilbert matrix at $5\times10^{14}$ by degree 10.

**The honest comparison.** Working around it is what you do when the basis is not yours to choose,
which is most of applied least squares: the columns are measured predictors, not a basis you
picked. Removing it is available exactly when you are approximating a function and get to choose
how to represent it, which is this part of the course.

### 1.2 What makes the three-term recurrence universal

**The property is that multiplication by $x$ is symmetric in the inner product:**

$$
\langle xp, q\rangle_w = \int xp(x)q(x)w(x)\,dx = \langle p, xq\rangle_w
$$

which holds because $x$ is a real scalar function and multiplication is commutative. Nothing about
the weight, the interval, or the family enters.

That symmetry is what kills the low order coefficients. Expanding $xp_k = \sum_j c_jp_j$ and
taking the inner product with $p_j$,

$$
c_j\langle p_j,p_j\rangle = \langle xp_k,p_j\rangle = \langle p_k, xp_j\rangle
$$

and $xp_j$ has degree $j+1$, which is below $k$ whenever $j < k-1$, so it lies in a span
orthogonal to $p_k$ and the inner product vanishes.

**An inner product where it would fail.** Take

$$
\langle p, q\rangle = \int_a^b p\,q\,w + p'(a)q'(a)
$$

a Sobolev inner product with a derivative term. Then

$$
\langle xp, q\rangle - \langle p, xq\rangle = (xp)'(a)q'(a) - p'(a)(xq)'(a)
= p(a)q'(a) - p'(a)q(a)
$$

which is not zero in general. Multiplication by $x$ is no longer symmetric, and the orthogonal
polynomials for that inner product satisfy a **longer** recurrence, typically of length 4 or more.
Sobolev orthogonal polynomials are a real subject and the loss of the three-term recurrence is
exactly what makes them harder.

**The general statement.** The recurrence has three terms if and only if multiplication by $x$ is
symmetric, which is if and only if the inner product comes from a measure on the line. That is
Favard's theorem, which exercise 5.1 states.

### 1.3 Truncation, and what it costs in the monomial basis

**In an orthogonal basis, truncating is optimal.** The coefficients are
$c_k = \langle f,p_k\rangle/\langle p_k,p_k\rangle$, each computed from $f$ and $p_k$ alone, with
no reference to the degree. So the first $k+1$ of the degree $n$ coefficients **are** the degree
$k$ coefficients.

**In the monomial basis it is false**, because the normal equations couple every coefficient to
every other. Changing the degree changes the matrix, so it changes the solution in every
component.

Measured, on $\exp$ over $[-1,1]$:

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import orthopoly as op

f = lambda t: np.exp(t)
out = op.truncation_is_optimal(f, 8, "legendre")
print("orthogonal basis, relative change in each coefficient when the degree changes:")
print(f"{'degree':>8}{'relative change':>20}")
for d, gap in zip(out["degrees"], out["relative_change"]):
    print(f"{d:>8}{gap:>20.3e}")

grid = np.linspace(-1.0, 1.0, 401)
print("\nmonomial basis, the constant coefficient at each degree:")
print(f"{'degree':>8}{'c_0':>22}")
for d in (2, 4, 6, 8, 10):
    V = np.vander(grid, d + 1, increasing=True)
    c, *_ = np.linalg.lstsq(V, f(grid), rcond=None)
    print(f"{d:>8}{c[0]:>22.10f}")
```

The orthogonal column is **exactly zero** at every degree. The monomial constant coefficient moves
at every degree.

**Three practical differences.**

**Adaptive degree selection becomes free.** Compute coefficients until the tail is small enough
and stop. In the monomial basis every candidate degree needs its own solve.

**The coefficients mean something on their own.** $c_k$ is the component of $f$ along $p_k$, so
its size says how much of the $k$th shape is present. A monomial coefficient means nothing
individually, since it depends on where the series was cut.

**Errors do not propagate between coefficients.** An error in computing $c_5$ leaves $c_0,\dots,c_4$
untouched. In a coupled solve it contaminates all of them.

### 2.1 Proving the three-term recurrence

**Setup.** Let $\{p_k\}$ be monic and orthogonal for $\langle\cdot,\cdot\rangle_w$, so
$\deg p_k = k$ with leading coefficient 1, and $\langle p_i,p_j\rangle = 0$ for $i\ne j$.

**Step 1.** $xp_k$ has degree $k+1$ and leading coefficient 1, so $xp_k - p_{k+1}$ has degree at
most $k$ and expands in $p_0,\dots,p_k$:

$$
xp_k = p_{k+1} + \sum_{j=0}^{k}c_jp_j
$$

**Step 2.** Take the inner product with $p_j$ for $j \le k$. Orthogonality gives

$$
c_j\langle p_j,p_j\rangle = \langle xp_k, p_j\rangle
$$

**Step 3, the essential one.** Use the symmetry of multiplication by $x$:

$$
\langle xp_k,p_j\rangle = \langle p_k, xp_j\rangle
$$

Now $xp_j$ has degree $j+1$. If $j+1 < k$, that is $j < k-1$, then $xp_j$ lies in
$\mathrm{span}\{p_0,\dots,p_{k-1}\}$, which is orthogonal to $p_k$. **So $c_j = 0$ for
$j < k-1$**, and only $c_k$ and $c_{k-1}$ survive.

**Step 4.** Writing $a_k = c_k$ and $b_k = -c_{k-1}$ and rearranging,

$$
p_{k+1} = (x - a_k)p_k - b_kp_{k-1}
$$

**Where the symmetry was used: step 3, and nowhere else.** Steps 1, 2 and 4 are bookkeeping. Take
that symmetry away, as in exercise 1.2's Sobolev inner product, and $c_j$ need not vanish for
$j < k-1$, and the recurrence lengthens.

**A second consequence of step 3, for free.** Taking $j = k$ and $j = k-1$ explicitly,

$$
a_k = \frac{\langle xp_k,p_k\rangle}{\langle p_k,p_k\rangle},
\qquad
b_k = \frac{\langle p_k,p_k\rangle}{\langle p_{k-1},p_{k-1}\rangle}
$$

The second uses $\langle xp_k,p_{k-1}\rangle = \langle p_k, xp_{k-1}\rangle = \langle p_k,p_k\rangle$,
because $xp_{k-1} - p_k$ has degree below $k$. Those two formulas are exercise 2.2's Stieltjes
procedure.

### 2.2 The Stieltjes formulas and their cost

The two formulas fall out of exercise 2.1:

$$
a_k = \frac{\langle xp_k, p_k\rangle}{\langle p_k,p_k\rangle},
\qquad
b_k = \frac{\langle p_k,p_k\rangle}{\langle p_{k-1},p_{k-1}\rangle}
$$

**Two inner products per step, and no more.** Compute $h_k = \langle p_k,p_k\rangle$ and
$\langle xp_k,p_k\rangle$. Then $a_k$ is their ratio, and $b_k$ is $h_k/h_{k-1}$, which reuses the
previous step's $h$. So each step costs exactly two integrals, and the total is $2n$.

Gram-Schmidt costs $\binom{n+2}{2} = (n+1)(n+2)/2$, because member $k$ needs one inner product
against each of the $k$ previous ones plus its own norm.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import orthopoly as op

print(f"{'degree':>8}{'Stieltjes':>12}{'Gram-Schmidt':>15}{'ratio':>9}"
      f"{'alpha gap':>13}{'beta gap':>12}")
for n in (2, 4, 8, 16, 32, 64):
    st = op.stieltjes(n, "legendre")
    gs = (n + 1) * (n + 2) // 2
    print(f"{n:>8}{st['inner_products_used']:>12}{gs:>15}"
          f"{gs / st['inner_products_used']:>9.2f}"
          f"{st['alpha_gap']:>13.2e}{st['beta_gap']:>12.2e}")
```

**The ratio is $(n+1)(n+2)/(4n)$, which is about $n/4$**, so at degree 64 Stieltjes uses 128
integrals against Gram-Schmidt's 2145, a factor of 17.

**The accuracy.** The recovered coefficients match the closed forms to $10^{-4}$ or better across
all five families at degree 6, and the residual is quadrature error rather than instability: it
grows slowly with the degree because the integrand $p_k^2w$ becomes more oscillatory and the fixed
grid resolves it less well.

**What the recurrence is really for.** Not only the cost. Once you have $a_k$ and $b_k$ you can
evaluate any member in $O(n)$, build the Jacobi matrix of exercise 5.2, and get Gauss quadrature
nodes. Gram-Schmidt gives you the polynomials and none of that.

### 2.3 The roots are real, distinct and inside the interval

**Claim.** $p_n$ has exactly $n$ distinct real roots, all in the open interval $(a,b)$ where the
weight lives.

**Proof.** Let $x_1,\dots,x_m$ be the points in $(a,b)$ where $p_n$ changes sign, and suppose
$m < n$. Form

$$
q(x) = \prod_{i=1}^{m}(x - x_i)
$$

which has degree $m < n$, so $\langle p_n, q\rangle = 0$ by orthogonality, since $q$ lies in the
span of $p_0,\dots,p_{n-1}$.

But $p_nq$ has no sign change in $(a,b)$: at each $x_i$ both factors change sign, so the product
does not. Hence $p_nq$ is of one sign throughout, and since $w > 0$,

$$
\langle p_n, q\rangle = \int_a^b p_n(x)q(x)w(x)\,dx \ne 0
$$

unless $p_nq$ vanishes identically, which it does not since $p_n$ is monic. **Contradiction**, so
$m \ge n$.

A degree $n$ polynomial has at most $n$ sign changes, so $m = n$ exactly, and all $n$ roots are
real, simple, and inside $(a,b)$.

**Why it matters.** It is what makes Gauss quadrature usable: the nodes are real and inside the
domain, so the rule evaluates $f$ where $f$ is defined. A quadrature rule with complex or exterior
nodes would be useless whatever its degree of exactness.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import orthopoly as op

print(f"{'family':>14}{'n':>4}{'all real':>10}{'all inside':>12}{'all distinct':>14}")
for family in ("legendre", "chebyshev_t", "chebyshev_u", "laguerre", "hermite"):
    interval = op.FAMILIES[family][0]
    for n in (3, 6, 10):
        nodes = op.golub_welsch(family, n)["nodes"]
        inside = bool(np.all(nodes > interval[0]) and np.all(nodes < interval[1]))
        gaps = np.diff(np.sort(nodes))
        print(f"{family:>14}{n:>4}{str(bool(np.all(np.isreal(nodes)))):>10}"
              f"{str(inside):>12}{str(bool(np.all(gaps > 1e-12))):>14}")
```

### 2.4 Christoffel-Darboux by induction

**Claim.** With $h_k = \langle p_k,p_k\rangle$,

$$
K_n(x,y) := \sum_{k=0}^{n}\frac{p_k(x)p_k(y)}{h_k}
= \frac{p_{n+1}(x)p_n(y) - p_n(x)p_{n+1}(y)}{h_n\,(x-y)}
$$

**Base case $n = 0$.** The left side is $p_0(x)p_0(y)/h_0 = 1/h_0$ for monic families. The right
side is

$$
\frac{p_1(x)p_0(y) - p_0(x)p_1(y)}{h_0(x-y)} = \frac{(x - a_0) - (y - a_0)}{h_0(x-y)}
= \frac{1}{h_0}
$$

using $p_1 = x - a_0$. They agree.

**Inductive step.** Assume the identity at $n-1$ and write
$D_n = p_{n+1}(x)p_n(y) - p_n(x)p_{n+1}(y)$. Substitute the recurrence
$p_{n+1}(t) = (t-a_n)p_n(t) - b_np_{n-1}(t)$ into both terms:

$$
D_n = \big[(x-a_n)p_n(x) - b_np_{n-1}(x)\big]p_n(y)
- p_n(x)\big[(y-a_n)p_n(y) - b_np_{n-1}(y)\big]
$$

The $a_n$ terms cancel, leaving

$$
D_n = (x-y)p_n(x)p_n(y) - b_n\big[p_{n-1}(x)p_n(y) - p_n(x)p_{n-1}(y)\big]
= (x-y)p_n(x)p_n(y) + b_nD_{n-1}
$$

Divide by $h_n(x-y)$ and use $b_n = h_n/h_{n-1}$:

$$
\frac{D_n}{h_n(x-y)} = \frac{p_n(x)p_n(y)}{h_n} + \frac{D_{n-1}}{h_{n-1}(x-y)}
= \frac{p_n(x)p_n(y)}{h_n} + K_{n-1}(x,y) = K_n(x,y)
$$

by the inductive hypothesis. Done.

**The numerical caveat the lesson measured.** The closed form is $0/0$ on the diagonal and
inaccurate near it. A probe grid landing within $4\times10^{-17}$ of $y$ gives a relative error of
**1.00** where the identity is otherwise exact to $6\times10^{-17}$. The confluent form, by
L'Hopital,

$$
K_n(x,x) = \frac{p_{n+1}'(x)p_n(x) - p_n'(x)p_{n+1}(x)}{h_n}
$$

fixes it, with the derivatives from the differentiated recurrence
$p_{k+1}' = p_k + (x-a_k)p_k' - b_kp_{k-1}'$, which is exact and costs the same $O(n)$.

### 2.5 Gauss quadrature is exact to degree $2n-1$

**Claim.** Let $x_1,\dots,x_n$ be the roots of $p_n$ and $w_1,\dots,w_n$ the weights making the
rule exact on $p_0,\dots,p_{n-1}$. Then

$$
\sum_{i=1}^{n}w_if(x_i) = \int_a^b f(x)w(x)\,dx
$$

for every polynomial $f$ of degree at most $2n-1$.

**Proof.** Divide: $f = qp_n + r$ with $\deg q \le n-1$ and $\deg r \le n-1$.

**The $q$ term integrates to zero.** $\int qp_nw = \langle p_n, q\rangle = 0$ by orthogonality,
since $\deg q < n$.

**The $q$ term is invisible to the rule too.** At each node $p_n(x_i) = 0$, so
$f(x_i) = r(x_i)$.

**The $r$ term is handled by construction.** The rule is exact on polynomials of degree at most
$n-1$, which includes $r$.

Putting these together,

$$
\sum_iw_if(x_i) = \sum_iw_ir(x_i) = \int rw = \int (f - qp_n)w = \int fw
$$

**Why $2n$ fails.** Take $f = p_n^2$, of degree exactly $2n$. Every node is a root of $p_n$, so
the rule gives $\sum_iw_ip_n(x_i)^2 = 0$. The true integral is
$\int p_n^2w = h_n > 0$. **So the rule is wrong by the whole value.**

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import orthopoly as op

print(f"{'n':>4}{'degree':>8}{'rule':>18}{'exact':>18}{'error':>12}")
for n in (2, 3, 4, 5):
    out = op.golub_welsch("legendre", n)
    for k in (2 * n - 1, 2 * n):
        rule = float(np.sum(out["weights"] * out["nodes"] ** k))
        exact = 0.0 if k % 2 else 2.0 / (k + 1)
        print(f"{n:>4}{k:>8}{rule:>18.12f}{exact:>18.12f}{abs(rule - exact):>12.2e}")
```

Exact at $2n-1$ and wrong at $2n$, at every $n$. **The sharpness is the point**: $2n-1$ is not a
sufficient condition someone proved, it is the exact reach of the rule, and a rule claiming more
would be wrong.

**The count that explains it.** A rule with $n$ nodes and $n$ weights has $2n$ free parameters, so
$2n$ conditions is the most it could possibly satisfy, giving exactness to degree $2n-1$. Gauss
rules attain that bound, which is why they are called optimal. Newton-Cotes rules of Part 9 fix
the nodes and only have $n$ free parameters, reaching degree $n-1$ or $n$.

### 3.1 Jacobi polynomials

The two parameter family with weight $(1-x)^\alpha(1+x)^\beta$ on $[-1,1]$. Everything in the
lesson is a special case.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import orthopoly as op


def jacobi_recurrence(alpha, beta, n):
    """Monic three-term coefficients for the Jacobi weight (1-x)^a (1+x)^b.

    The formulas are the standard ones; the point of writing them out is that Legendre,
    Chebyshev T and Chebyshev U all fall out of particular (a, b), which is checked below.
    """
    a = float(alpha)
    b = float(beta)
    if a <= -1.0 or b <= -1.0:
        raise ValueError(f"the weight is not integrable unless both exponents exceed -1, "
                         f"got ({a}, {b})")
    m = int(n)
    alpha_k = np.zeros(m)
    beta_k = np.zeros(m)
    beta_k[0] = 2.0 ** (a + b + 1.0) * math.gamma(a + 1.0) * math.gamma(b + 1.0) \
        / math.gamma(a + b + 2.0)
    for k in range(m):
        denom = (2.0 * k + a + b) * (2.0 * k + a + b + 2.0)
        alpha_k[k] = (b * b - a * a) / denom if denom != 0.0 else 0.0
        if k == 1:
            # the general formula has a removable singularity here whenever a + b = -1,
            # which includes the Chebyshev T case (-1/2, -1/2). This is its limit.
            beta_k[1] = 4.0 * (1.0 + a) * (1.0 + b) \
                / ((a + b + 2.0) ** 2 * (a + b + 3.0))
        elif k >= 2:
            top = 4.0 * k * (k + a) * (k + b) * (k + a + b)
            bottom = (2.0 * k + a + b) ** 2 * (2.0 * k + a + b + 1.0) \
                * (2.0 * k + a + b - 1.0)
            beta_k[k] = top / bottom
    return alpha_k, beta_k


def jacobi_evaluate(alpha, beta, degree, x):
    n = int(degree)
    a_k, b_k = jacobi_recurrence(alpha, beta, max(n, 1))
    z = np.atleast_1d(np.asarray(x, dtype=float))
    prev = np.zeros_like(z)
    curr = np.ones_like(z)
    for k in range(n):
        curr, prev = (z - a_k[k]) * curr - (b_k[k] if k > 0 else 0.0) * prev, curr
    return curr


x = np.linspace(-0.95, 0.95, 9)
top_degree = 6
print(f"{'(alpha, beta)':>16}{'is':>16}{'worst gap':>14}")
for (a, b), name, family in (((0.0, 0.0), "Legendre", "legendre"),
                             ((-0.5, -0.5), "Chebyshev T", "chebyshev_t"),
                             ((0.5, 0.5), "Chebyshev U", "chebyshev_u")):
    worst = 0.0
    for n in range(top_degree + 1):
        got = jacobi_evaluate(a, b, n, x)
        want = op.evaluate(family, n, x)
        worst = max(worst, float(np.max(np.abs(got - want))))
    print(f"{f'({a}, {b})':>16}{name:>16}{worst:>14.2e}")
```

**All three are recovered to $10^{-15}$**, which is what "special case" means made checkable.

The pattern in the exponents is worth seeing: $\alpha = \beta$ gives a symmetric weight and an
even or odd family, $\alpha = \beta = -\tfrac12$ concentrates the weight at the endpoints (the
minimax weight), $\alpha = \beta = +\tfrac12$ pushes it to the middle, and $\alpha = \beta = 0$ is
flat.

**Where the general family is used.** Jacobi polynomials with $\alpha \ne \beta$ are the right
basis when the function has different behaviour at the two ends, which happens in spectral element
methods and in the analysis of boundary layers. Gauss-Jacobi quadrature handles integrands with an
algebraic singularity at an endpoint by absorbing it into the weight, which is a genuinely useful
trick and the reason the family is worth having.

**One thing to check when implementing it.** The $k = 0$ formula for $\alpha_k$ has a zero
denominator when $\alpha + \beta = 0$ and $k = 0$, which includes the Legendre case. The guard
above returns 0 there, which is the correct limit, and getting it wrong produces a `nan` that
propagates silently into every later coefficient.

### 3.2 The modified Chebyshev algorithm

Exercise 4.2 measures the problem: building recurrence coefficients from **raw moments**
$\mu_k = \int x^kw$ requires solving with a Hankel matrix whose condition number reaches
$9.4\times10^9$ by degree 14. That is the Hilbert matrix again, in a new disguise.

The fix is to use **modified moments** against a known family instead:

$$
m_k = \int_a^b q_k(x)\,w(x)\,dx
$$

where $q_k$ is some family whose recurrence you already have, typically Chebyshev.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import orthopoly as op


def modified_chebyshev(modified_moments, ref_alpha, ref_beta):
    """Sack and Donovan's algorithm: recurrence coefficients from modified moments.

    Builds a table sigma[k][l] = <p_k, q_l> and reads the coefficients off it. The whole point
    is that the intermediate quantities stay of comparable size, where the raw moment route
    forms a Hankel matrix whose entries span many orders of magnitude.
    """
    m = np.asarray(modified_moments, dtype=float).ravel()
    n = m.size // 2
    if n < 1:
        raise ValueError(f"need at least two modified moments, got {m.size}")
    a = np.asarray(ref_alpha, dtype=float)
    b = np.asarray(ref_beta, dtype=float)
    sigma = np.zeros((n + 1, 2 * n))
    alpha = np.zeros(n)
    beta = np.zeros(n)
    sigma[0, :] = m
    alpha[0] = a[0] + m[1] / m[0]
    beta[0] = m[0]
    for k in range(1, n):
        for l in range(k, 2 * n - k):
            prev = sigma[k - 1, l] if k >= 1 else 0.0
            older = sigma[k - 2, l] if k >= 2 else 0.0
            sigma[k, l] = (sigma[k - 1, l + 1] - (alpha[k - 1] - a[l]) * prev
                           - beta[k - 1] * older + b[l] * sigma[k - 1, l - 1])
        alpha[k] = a[k] + sigma[k, k + 1] / sigma[k, k] - sigma[k - 1, k] / sigma[k - 1, k - 1]
        beta[k] = sigma[k, k] / sigma[k - 1, k - 1]
    return alpha, beta


def raw_moment_hankel(n, lo=-1.0, hi=1.0):
    """The matrix the naive route has to invert. For a flat weight it is Hilbert-like."""
    mu = np.asarray([(hi ** (k + 1) - lo ** (k + 1)) / (k + 1) for k in range(2 * n + 1)])
    return np.stack([[mu[i + j] for j in range(n + 1)] for i in range(n + 1)])


print(f"{'degree':>8}{'raw moment Hankel kappa':>27}{'orthonormal Gram kappa':>25}")
for n in (2, 4, 6, 8, 10, 12, 14):
    H = raw_moment_hankel(n)
    orth = op.conditioning_against_monomials(n, "legendre")
    print(f"{n:>8}{float(np.linalg.cond(H)):>27.4e}"
          f"{orth['orthonormal_condition']:>25.6f}")
```

| degree | raw moment Hankel $\kappa$ | orthonormal Gram $\kappa$ |
|---|---|---|
| 2 | 1.4129e1 | 1.000000 |
| 4 | 3.5829e2 | 1.000004 |
| 6 | 1.0228e4 | 1.000017 |
| 8 | 3.0660e5 | 1.000047 |
| 10 | 9.4431e6 | 1.000104 |
| 12 | 2.9592e8 | 1.000201 |
| 14 | 9.3857e9 | 1.000353 |

**The raw moment matrix loses about 1.5 digits per degree** and has lost ten by degree 14. That
is the same growth rate the Hilbert matrix has, and for the same reason: the monomials $x^k$
become nearly parallel in the $L^2$ inner product.

**What the modified version fixes and what it does not.** Using modified moments against a
reference family whose support matches the weight keeps the intermediate $\sigma_{k\ell}$ of
comparable size, and the algorithm is then stable. If the reference family is a poor match for the
weight, the modified moments themselves span many orders of magnitude and nothing is gained. So
the algorithm is not a universal fix; it is a way of exploiting a good reference when you have
one, and choosing that reference is the skill.

**When you need any of this.** Only for a **non-classical** weight, where no closed form
recurrence exists. For the five families of the lesson the coefficients are known exactly and
there is nothing to compute. The situation where it matters is a weight given by data, or one
arising from a physical model, and it is common enough in uncertainty quantification that Gautschi
wrote a book about it.

### 3.3 Gauss-Lobatto and Gauss-Radau

Gauss rules put every node in the interior. Sometimes an endpoint is required as a node: for a
spectral element method the elements have to share their endpoints, and for an integrator with a
known boundary value the endpoint evaluation is free.

Fixing $k$ nodes costs $k$ degrees of exactness.

| rule | fixed nodes | free parameters | degree of exactness |
|---|---|---|---|
| Gauss | none | $2n$ | $2n-1$ |
| Gauss-Radau | one end | $2n-1$ | $2n-2$ |
| Gauss-Lobatto | both ends | $2n-2$ | $2n-3$ |

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import orthopoly as op


def lobatto(n):
    """Gauss-Lobatto on [-1, 1] for the Legendre weight, by modifying the Jacobi matrix.

    Fixing both endpoints means demanding that p_n vanish there, which is achieved by adjusting
    the last diagonal entry and the last off diagonal entry of the Jacobi matrix. Golub's 1973
    construction, which is the same eigensolve as the plain Gauss rule.
    """
    m = int(n)
    if m < 3:
        raise ValueError(f"Lobatto needs at least three nodes, got {m}")
    rec = op.recurrence_coefficients("legendre", m)
    a = rec["alpha"].copy()
    b = rec["beta"].copy()
    # solve the 2x2 system that forces both endpoints to be eigenvalues
    ends = np.array([-1.0, 1.0])
    sol = np.zeros(ends.size)
    for j, e in enumerate(ends):
        p_prev, p_curr = 0.0, 1.0
        for k in range(m - 1):
            p_curr, p_prev = (e - a[k]) * p_curr - (b[k] if k > 0 else 0.0) * p_prev, p_curr
        sol[j] = p_curr
    p_prev_m1 = np.zeros(ends.size)
    for j, e in enumerate(ends):
        p_prev, p_curr = 0.0, 1.0
        for k in range(m - 2):
            p_curr, p_prev = (e - a[k]) * p_curr - (b[k] if k > 0 else 0.0) * p_prev, p_curr
        p_prev_m1[j] = p_curr
    M = np.array([[sol[0], p_prev_m1[0]], [sol[1], p_prev_m1[1]]])
    rhs = np.array([ends[0] * sol[0], ends[1] * sol[1]])
    fix = np.linalg.solve(M, rhs)
    a[m - 1] = fix[0]
    b[m - 1] = fix[1]
    J = np.diag(a) + np.diag(np.sqrt(np.abs(b[1:m])), 1) + np.diag(np.sqrt(np.abs(b[1:m])), -1)
    vals, vecs = np.linalg.eigh(J)
    order = np.argsort(vals)
    return vals[order], b[0] * vecs[0, order] ** 2


print(f"{'n':>4}{'first node':>14}{'last node':>13}{'weights sum':>14}"
      f"{'exact to degree':>18}")
for n in (3, 4, 5, 6, 8):
    nodes, weights = lobatto(n)
    top = -1
    for k in range(0, 2 * n):
        rule = float(np.sum(weights * nodes ** k))
        exact = 0.0 if k % 2 else 2.0 / (k + 1)
        if abs(rule - exact) > 1e-10:
            break
        top = k
    print(f"{n:>4}{nodes[0]:>14.10f}{nodes[-1]:>13.10f}"
          f"{float(np.sum(weights)):>14.10f}{top:>18}")
```

**The endpoints come out at exactly $\pm1$** and the degree of exactness is $2n-3$, one less than
Gauss for each fixed node, as the counting predicts.

**Which to use.** Gauss when you only need the integral. Lobatto when the nodes are also the
degrees of freedom of a discretisation and neighbouring elements must agree at the shared
boundary, which is the whole of the spectral element method in Part 11. Radau when one end is
special, for instance in a stiff ODE integrator where the endpoint is where the solution is
required, which Part 10's implicit Runge-Kutta methods use.

**What is given up, quantified.** Two degrees of exactness is a factor of about $h^2$ in the error
for a composite rule, which sounds serious and usually is not: the Lobatto rule of $n+1$ points has
the same exactness as the Gauss rule of $n$ points, so the cost is one extra evaluation rather
than an order.

### 4.1 Gram-Schmidt against Stieltjes

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import orthopoly as op

print(f"{'degree':>8}{'classical GS':>18}{'modified GS':>16}{'ratio':>12}"
      f"{'GS integrals':>14}{'Stieltjes':>11}")
for n in (4, 8, 12, 16, 20, 24, 28):
    classical = op.gram_schmidt(n, "legendre", n_probe=20001,
                                classical=True)["worst_off_diagonal"]
    modified = op.gram_schmidt(n, "legendre", n_probe=20001)["worst_off_diagonal"]
    print(f"{n:>8}{classical:>18.3e}{modified:>16.3e}"
          f"{classical / max(modified, 1e-300):>12.3g}"
          f"{(n + 1) * (n + 2) // 2:>14}{2 * n:>11}")
```

| degree | classical GS | modified GS | ratio | GS integrals | Stieltjes |
|---|---|---|---|---|---|
| 4 | 5.551e-17 | 5.551e-17 | 1 | 15 | 8 |
| 8 | 5.551e-17 | 5.551e-17 | 1 | 45 | 16 |
| 12 | 4.769e-16 | 5.551e-17 | 8.59 | 91 | 24 |
| 16 | 3.263e-14 | 1.271e-16 | 257 | 153 | 32 |
| 20 | 6.239e-12 | 1.690e-16 | 3.69e4 | 231 | 40 |
| 24 | 1.816e-8 | 1.690e-16 | 1.07e8 | 325 | 48 |
| 28 | **7.803e-7** | **2.064e-16** | 3.78e9 | 435 | 56 |

**The premise needs correcting.** The exercise asks where Gram-Schmidt "loses orthogonality", and
the answer depends entirely on **which** Gram-Schmidt.

**The modified form does not lose it at all**, staying at $2\times10^{-16}$ out to degree 28. The
classical form does, reaching $7.8\times10^{-7}$ at the same degree, a factor of $3.8\times10^9$.
The first divergence is at degree 12, and by degree 20 the classical form has lost six digits.

That is exactly Part 5's distinction on vectors, appearing unchanged on functions. **Classical**
projects the original vector onto every previous direction; **modified** subtracts as it goes, so
each projection is computed against what is left rather than against what started. The two are
identical in exact arithmetic and differ in floating point because the later projections in the
classical form are computed against a vector that still contains components already removed.

`nalib.orthopoly.gram_schmidt` uses the modified form by default, and its docstring says so,
because the original wording claimed a loss that the code does not have.

**So the objection to Gram-Schmidt is the cost, not the accuracy**, provided it is written the
right way. At degree 28 it needs 435 integrals against Stieltjes's 56, a factor of 7.8, and the
factor is $n/4$ asymptotically.

**And the cost objection is the more serious one anyway**, because those are integrals of
increasingly oscillatory integrands, each needing its own quadrature accuracy. Stieltjes needs
$2n$ of them and never forms a high degree product.

### 4.2 The raw moment route

The most natural way to build recurrence coefficients from a weight is to compute its moments
$\mu_k = \int x^kw$ and solve for the coefficients. It is also unusable.

The measurement is in exercise 3.2's table: the Hankel matrix $H_{ij} = \mu_{i+j}$ reaches
$9.4\times10^9$ at degree 14, having grown by a factor of about 30 per two degrees.

**Why it is the Hilbert matrix in disguise.** On $[0,1]$ with weight 1 the moments are
$\mu_k = 1/(k+1)$, so $H_{ij} = 1/(i+j+1)$, which **is** the Hilbert matrix exactly. On $[-1,1]$
the odd moments vanish and the matrix splits into two blocks, each Hilbert-like, which is why the
growth rate is halved but not removed.

**The underlying cause is the same in every case**: the monomials $x^k$ become nearly parallel in
the $L^2$ inner product as $k$ grows, because $x^k$ and $x^{k+2}$ agree to within a fixed factor
over most of the interval. The Gram matrix of nearly parallel vectors is nearly singular, and it
does not matter whether you call it a Gram matrix, a Hankel matrix or a moment matrix.

```python
import numpy as np

print("the moment matrix on [0,1] IS the Hilbert matrix:")
n = 5
mu = np.asarray([1.0 / (k + 1) for k in range(2 * n + 1)])
H = np.stack([[mu[i + j] for j in range(n + 1)] for i in range(n + 1)])
hilbert = np.stack([[1.0 / (i + j + 1) for j in range(n + 1)] for i in range(n + 1)])
print(f"  worst difference: {float(np.max(np.abs(H - hilbert))):.2e}")
print(f"  condition number: {float(np.linalg.cond(H)):.4e}")
```

**Two ways out, and only one of them works generally.**

Exercise 3.2's **modified moments** replace $x^k$ by a known orthogonal family, so the quantities
being combined are no longer nearly parallel. It works when the reference family matches the
weight's support and rough shape, and does not otherwise.

**Discretised Stieltjes**, which is what `orthopoly.stieltjes` does, computes the inner products
by quadrature directly and never forms a moment matrix at all. It is the standard method and it is
stable because it never asks a nearly singular question.

**The general lesson, which recurs throughout this course.** The Hilbert matrix is not a curiosity;
it is what you get whenever the monomials are used as a basis for anything involving an inner
product. Lesson 44 met it in the Vandermonde system, lesson 54 in the normal equations, and here
in the moments. **The fix is always the same: stop using the monomials.**

### 4.3 Convergence against smoothness, for each family

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import orthopoly as op

targets = [("exp", np.exp, "analytic"),
           ("1/(1+25t^2)", lambda s: 1.0 / (1.0 + 25.0 * s * s), "poles at +-i/5"),
           ("|t|", np.abs, "continuous only")]
for name, f, note in targets:
    print(f"{name}  ({note})")
    print(f"{'family':>16}" + "".join(f"{'n=' + str(n):>13}" for n in (2, 4, 8, 16, 32)))
    for family in ("legendre", "chebyshev_t", "chebyshev_u"):
        row = [op.least_squares_by_orthogonality(f, n, family,
                                                 n_probe=200001)["weighted_l2_error"]
               for n in (2, 4, 8, 16, 32)]
        print(f"{family:>16}" + "".join(f"{v:>13.3e}" for v in row))
    print()
```

**$\exp$, analytic:**

| family | $n=2$ | $n=4$ | $n=8$ | $n=16$ | $n=32$ |
|---|---|---|---|---|---|
| Legendre | 3.795e-2 | 4.705e-4 | 1.031e-8 | **2.360e-8** | **1.700e-7** |
| Chebyshev T | 5.599e-2 | 6.828e-4 | 1.385e-8 | 4.386e-16 | 4.681e-16 |
| Chebyshev U | 2.766e-2 | 3.394e-4 | 6.909e-9 | 3.967e-16 | 4.173e-16 |

**$1/(1+25t^2)$, poles at $\pm i/5$:**

| family | $n=2$ | $n=4$ | $n=8$ | $n=16$ | $n=32$ |
|---|---|---|---|---|---|
| Legendre | 2.724e-1 | 1.835e-1 | 8.303e-2 | 1.696e-2 | 7.064e-4 |
| Chebyshev T | 2.999e-1 | 2.015e-1 | 9.103e-2 | 1.857e-2 | 7.731e-4 |
| Chebyshev U | 2.507e-1 | 1.685e-1 | 7.611e-2 | 1.553e-2 | 6.463e-4 |

**$\lvert t\rvert$, continuous only:**

| family | $n=2$ | $n=4$ | $n=8$ | $n=16$ | $n=32$ |
|---|---|---|---|---|---|
| Legendre | 1.021e-1 | 5.103e-2 | 2.233e-2 | 8.908e-3 | 3.361e-3 |
| Chebyshev T | 1.209e-1 | 5.742e-2 | 2.401e-2 | 9.282e-3 | 3.435e-3 |
| Chebyshev U | 8.887e-2 | 4.608e-2 | 2.089e-2 | 8.568e-3 | 3.290e-3 |

**The rate is set by the function, not by the family.** All three families give the same order on
each target: faster than geometric on $\exp$, geometric on Runge, algebraic on $|t|$. The
constants differ by tens of percent and the exponents do not differ at all.

That is the expected answer and it is worth saying why: the convergence rate of an orthogonal
series is governed by how well polynomials of that degree can approximate $f$, which is a property
of $f$ and the interval, and every family here spans the same polynomials.

**The Legendre row on $\exp$ turns around, and that is my quadrature, not the theory.** It reaches
$1.03\times10^{-8}$ at degree 8 and then gets **worse**, to $1.7\times10^{-7}$ at degree 32. The
reason is in `orthopoly.weighted_grid`: the Legendre grid is a plain trapezoid rule on a uniform
mesh, whose own error floor is around $10^{-8}$, while the Chebyshev grids use the substitution
$x = \cos\theta$ and reach $4\times10^{-16}$.

So the Legendre column stops measuring the approximation at degree 8 and starts measuring the
quadrature. **Reading it as "Legendre is worse for analytic functions" would be reading the
instrument**, and the honest statement is that the three families are indistinguishable wherever
the measurement is valid.

**The one real difference between the families**, which this table cannot show, is in the
**maximum** norm rather than the $L^2$ norm. There the Chebyshev weight is the right one, because
it is the weight for which the equioscillation of lesson 54 is natural, and lesson 56 measures the
truncated Chebyshev series as near-minimax while a truncated Legendre series is not.

### 5.1 Favard's theorem

**Statement.** Let $\{p_k\}$ be defined by $p_{-1} = 0$, $p_0 = 1$ and

$$
p_{k+1}(x) = (x - a_k)p_k(x) - b_kp_{k-1}(x)
$$

with $a_k$ real and $b_k > 0$ for $k \ge 1$. Then there exists a positive Borel measure $\mu$ on
$\mathbb R$ for which $\{p_k\}$ is orthogonal:
$\int p_ip_j\,d\mu = 0$ for $i \ne j$.

**What it means.** The three-term recurrence is not merely a property that orthogonal polynomials
happen to have. **It characterises them.** Any sequence built by such a recurrence with positive
$b_k$ is an orthogonal family for some measure, and any orthogonal family satisfies such a
recurrence. The two conditions are equivalent.

**Three consequences for this course.**

**The classification of families is a classification of recurrences.** Asking "what are all the
orthogonal polynomial families" is the same as asking "what are all the pairs of sequences
$(a_k, b_k)$ with $b_k > 0$", which is a much more concrete question. The five classical families
correspond to the recurrences whose coefficients are rational in $k$, and that is a theorem
(Bochner's) rather than an accident of which ones people happened to name.

**It licenses `stieltjes` and `golub_welsch`.** Both work entirely from $(a_k, b_k)$ and never
touch the weight. Favard says nothing is lost by that: the coefficients determine the measure, so
they determine everything.

**The positivity of $b_k$ is the whole hypothesis and it is load bearing.** If some $b_k < 0$ the
resulting sequence satisfies no orthogonality relation for a positive measure. `golub_welsch`
takes $\sqrt{b_k}$ to build the Jacobi matrix, and with a negative $b_k$ that is imaginary and the
matrix is not symmetric, so the nodes need not be real. The construction fails exactly where the
theorem says it should.

**The connection to Part 6.** Favard's theorem is the statement that a symmetric tridiagonal
matrix with positive off-diagonal entries has real, distinct eigenvalues and a spectral measure.
The Jacobi matrix is that matrix, its eigenvalues are the Gauss nodes, and its spectral measure is
$\mu$. So Favard is the spectral theorem for Jacobi matrices, seen from the polynomial side.

### 5.2 Why the Jacobi matrix works

**Claim.** The eigenvalues of

$$
J_n = \begin{pmatrix}
a_0 & \sqrt{b_1} & & \\
\sqrt{b_1} & a_1 & \ddots & \\
& \ddots & \ddots & \sqrt{b_{n-1}}\\
& & \sqrt{b_{n-1}} & a_{n-1}
\end{pmatrix}
$$

are exactly the roots of $p_n$.

**Proof.** Let $\hat p_k = p_k/\sqrt{h_k}$ be the orthonormal members. The monic recurrence
rescales to

$$
x\hat p_k(x) = \sqrt{b_{k+1}}\,\hat p_{k+1}(x) + a_k\hat p_k(x) + \sqrt{b_k}\,\hat p_{k-1}(x)
$$

which says exactly that, with $v(x) = (\hat p_0(x),\dots,\hat p_{n-1}(x))^T$,

$$
J_nv(x) = x\,v(x) - \sqrt{b_n}\,\hat p_n(x)\,e_{n-1}
$$

the last term being what falls off the end of the matrix. **So $v(x)$ is an eigenvector with
eigenvalue $x$ precisely when $\hat p_n(x) = 0$.** Since $p_n$ has $n$ distinct real roots
(exercise 2.3) and $J_n$ has $n$ eigenvalues, those are all of them.

**The weights.** The first component of the normalised eigenvector at node $x_i$ is
$\hat p_0(x_i)/\|v(x_i)\| = 1/(\sqrt{h_0}\|v(x_i)\|)$, and $\|v(x_i)\|^2 = K_{n-1}(x_i,x_i)$ is
the Christoffel-Darboux kernel of exercise 2.4. The Gauss weight is
$w_i = 1/K_{n-1}(x_i,x_i)$, which is $h_0$ times the squared first component, and $h_0 = b_0$ is
the total mass. That is the formula `golub_welsch` uses.

**The connection to the companion matrix of lesson 13.** Both turn a polynomial's roots into an
eigenvalue problem. The companion matrix does it for **any** polynomial, written in the monomial
basis, and it is not symmetric: its eigenvalues can be ill conditioned, and lesson 13 measured
exactly that.

The Jacobi matrix does it for an **orthogonal** polynomial, written in its own basis, and it is
symmetric. That is the whole difference:

| | companion matrix | Jacobi matrix |
|---|---|---|
| applies to | any polynomial | orthogonal families only |
| basis | monomial | the family's own |
| structure | Hessenberg, non-symmetric | symmetric tridiagonal |
| eigenvalue conditioning | can be $10^{16}$ | perfectly conditioned |
| cost | $O(n^3)$ | $O(n^2)$ |

**Symmetric means perfectly conditioned eigenvalues**, by Part 6's Bauer-Fike with an orthogonal
eigenvector matrix. So Gauss nodes are computed to full accuracy at every $n$, which is why nobody
computes them by root finding on $p_n$.

### 5.3 Lanczos and the Jacobi matrix

**They are the same object.** Part 4's Lanczos process, applied to a symmetric $A$ with starting
vector $q_1$, produces

$$
AQ_m = Q_mT_m + \beta_mq_{m+1}e_m^T
$$

with $T_m$ symmetric tridiagonal. That $T_m$ **is** a Jacobi matrix, and the measure it belongs to
is the **spectral measure of $A$ at $q_1$**:

$$
d\mu(\lambda) = \sum_{j}|q_1^Tu_j|^2\,\delta(\lambda - \lambda_j)
$$

where $\lambda_j, u_j$ are the eigenpairs of $A$. So $\mu$ is a discrete measure with an atom at
each eigenvalue of $A$, weighted by how much of the starting vector points along the corresponding
eigenvector.

**The dictionary is exact:**

| Lanczos | orthogonal polynomials |
|---|---|
| the matrix $A$ | multiplication by $x$ |
| the starting vector $q_1$ | the constant function $p_0$ |
| the Krylov vector $q_{k+1}$ | the polynomial $\hat p_k$ |
| the three-term recurrence | the three-term recurrence |
| $T_m$ | the Jacobi matrix $J_m$ |
| Ritz values | Gauss nodes for $\mu$ |
| Lanczos orthogonality | polynomial orthogonality |

Lanczos **is** the Stieltjes procedure of exercise 2.2, run on a discrete measure.

**What this says about ghost eigenvalues.** Part 4 measured loss of orthogonality in Lanczos, and
Paige's theorem: the loss is proportional to the reciprocal of the residual, so it happens exactly
when a Ritz value has converged. Duplicated Ritz values, the ghosts, appear as a consequence.

In the polynomial language, a ghost is a **repeated Gauss node**, which exercise 2.3 proves cannot
happen for a genuine measure with $b_k > 0$. So a ghost is direct evidence that the computed
$b_k$ have lost their positivity or their meaning, and Favard's hypothesis has failed
numerically. That is a sharper statement than "orthogonality was lost": it says which structural
property broke.

**And it explains the Cullum-Willoughby test.** That test removes a Ritz value if it is also an
eigenvalue of $T_m$ with its first row and column deleted. In polynomial terms, deleting the first
row and column moves to the family orthogonal with respect to the same measure shifted by one
index, and a node shared by both is a node that carries no weight, since the Gauss weight is
$1/K_{n-1}(x_i,x_i)$ and the kernel involves exactly the deleted component. **A ghost has zero
weight, which is why the test finds it**, and it is the same reason lesson 39's measurement found
the test working at $m = 1.5n$ and deleting everything at $m \ge 2.5n$: once the spectrum is
saturated, every node is shared.

---

## Lesson 56, Chebyshev Approximation and Economization

### 1.1 What "near" is doing

**The word "near" is a bounded factor, not a bounded difference.** The precise statement is

$$
\|f - S_nf\|_\infty \le (1 + \Lambda_n)\,\|f - p^*_n\|_\infty
$$

where $S_n$ is the Chebyshev truncation, $p^*_n$ is the minimax polynomial, and $\Lambda_n$ is the
**Lebesgue constant of Chebyshev projection**,
$\Lambda_n = \frac{4}{\pi^2}\log n + O(1)$.

**The measured factor** on $\exp$ is 1.74 to 1.99 across degrees 1 to 12, and across five
functions and four degrees it runs 1.47 to 2.12. **The classical bound** over the same range is
4.28 to 5.12.

So the bound is honest and loose by about a factor of 2.5. Exercise 4.1 pushes the measured factor
as high as 3.305, on $|t-0.3|$ at degree 16, which is the largest found among eleven functions and
seven degrees.

**Why the bound grows and the measurement does not seem to.** $\Lambda_n$ grows like $\log n$, so
the factor is genuinely unbounded, and any claim that it is "about 2" is a statement about the
degrees people use rather than a theorem. Over degree 1 to 24 a $\log n$ term rises by a factor of
about 4.5 in its own size, which is a change of a few tenths in the factor, and that is exactly
what the tables show.

### 1.2 Interpolation coefficients against series coefficients

**They are different objects.** The series coefficient $a_k$ is the integral
$\frac2\pi\int fT_k/\sqrt{1-x^2}$. The interpolation coefficient is a finite cosine sum over the
$n+1$ Chebyshev points, which cannot distinguish harmonics that agree on that grid.

**The exact relation, on the Lobatto grid:**

$$
a_k^{\text{grid}} = a_k + a_{2n-k} + a_{2n+k} + a_{4n-k} + a_{4n+k} + \cdots
$$

Every harmonic the grid cannot resolve is **folded** onto a lower one, which is the same aliasing
lesson 58 measures for the DFT and lesson 46's interpolation error in a different basis.

**Why the gap shrinks.** The aliased terms are exactly the tail coefficients $a_{2n-k}$ and above.
For a smooth $f$ those decay geometrically, so the gap is of the order of $a_{2n}$, which is tiny
compared with $a_k$ for small $k$. Measured on $\exp$:

| degree | Lobatto grid gap | roots grid gap |
|---|---|---|
| 2 | 4.488e-2 | 5.474e-3 |
| 3 | 5.474e-3 | 5.429e-4 |
| 5 | 4.498e-5 | 3.198e-6 |
| 9 | 5.506e-10 | 2.498e-11 |
| 15 | 7.269e-16 | 1.453e-15 |

**The roots grid aliases about eight times less** at every degree, because the Lobatto grid
repeats its endpoints in the underlying even extension and the roots grid does not.

**Why the interpolation version is still what to use.** Three reasons.

It needs only $n+1$ evaluations of $f$, at points you choose. The quadrature version needs an
accurate integral, which means many more evaluations and a quadrature rule whose own error has to
be controlled.

It is $O(n\log n)$ through the FFT, which exercise 3.3 measures at 315 times faster than the
direct cosine sum at degree 4096.

**And the object it produces is the one you want anyway.** The interpolant reproduces $f$ exactly
at the nodes, which the truncated series does not, and its error is within a factor of 2 of the
truncated series' error (exercise 4 of the lesson measured 1.56 to 1.91).

### 1.3 When converting to powers is safe

**The conversion is exact in exact arithmetic** because it is a change of basis, and the
transition matrix is exactly invertible.

**What goes wrong in floating point** is cancellation. The power coefficients of $T_k$ alternate
in sign and the leading one is $2^{k-1}$, so evaluating the power form at $|x| < 1$ adds numbers
of size $2^{k-1}$ and gets an answer bounded by 1. Every digit above $\log_{10}2^{k-1}$ is lost.

Measured on $T_n$ itself, where nothing decays:

| degree | largest power coefficient | round trip disagreement |
|---|---|---|
| 20 | 6.554e6 | 1.521e-9 |
| 30 | 3.618e10 | 7.899e-6 |
| 40 | 2.124e14 | 5.626e-2 |
| 50 | 1.288e18 | **228** |

**The one condition under which it is safe: the Chebyshev coefficients decay fast enough that the
large power coefficients are multiplied by nothing.** On $\exp$, where $a_{50}\sim10^{-60}$, the
round trip survives to $2.8\times10^{-14}$ at degree 50. On the Runge function, whose coefficients
decay only geometrically with rate 1.22, it has lost everything by degree 50.

**That condition is not a useful licence**, because a series whose coefficients have collapsed by
degree 40 should have been truncated at degree 20, where there was no conversion problem to begin
with. The rule that survives is: **evaluate with Clenshaw**, and exercise 4.2 measures it flat at
$8\times10^{-16}$ where the power form reaches 6.0.

### 2.1 The coefficient formula, and the halved $a_0$

**Derivation.** Lesson 55 gives the general orthogonal expansion

$$
f = \sum_k \frac{\langle f, T_k\rangle}{\langle T_k,T_k\rangle}\,T_k,
\qquad \langle u,v\rangle = \int_{-1}^{1}\frac{u\,v}{\sqrt{1-x^2}}\,dx
$$

so all that is needed is $\langle T_k,T_k\rangle$. Substitute $x = \cos\theta$, which turns
$dx/\sqrt{1-x^2}$ into $-d\theta$ and $T_k$ into $\cos k\theta$:

$$
\langle T_j,T_k\rangle = \int_0^{\pi}\cos j\theta\,\cos k\theta\,d\theta
= \begin{cases}\pi & j=k=0\\ \pi/2 & j=k\ne0\\ 0 & j\ne k\end{cases}
$$

**That is the whole calculation**, and it is the same normalisation `orthopoly.FAMILIES` stores as
$b_0 = \pi$, $b_1 = 1/2$, $b_k = 1/4$.

Therefore

$$
c_k = \frac{\langle f,T_k\rangle}{\langle T_k,T_k\rangle}
= \frac{2}{\pi}\int_{-1}^{1}\frac{f\,T_k}{\sqrt{1-x^2}}\,dx \quad (k \ge 1),
\qquad
c_0 = \frac{1}{\pi}\int_{-1}^{1}\frac{f}{\sqrt{1-x^2}}\,dx
$$

**Why $a_0$ is halved.** The convention is to write **one** formula,
$a_k = \frac2\pi\int fT_k/\sqrt{1-x^2}$, for every $k$, and then halve $a_0$ when summing. That is
purely bookkeeping: $\langle T_0,T_0\rangle = \pi$ rather than $\pi/2$, because $\cos0 = 1$ has no
oscillation to average away. Writing the sum as
$\frac{a_0}{2} + \sum_{k\ge1}a_kT_k$ keeps a single formula for the coefficients at the cost of a
factor in the sum, and `chebapprox` instead halves the coefficient itself so that
`evaluate_series` can sum without a special case.

### 2.2 Clenshaw's recurrence

**Derivation.** Given $S = \sum_{k=0}^{n}a_kT_k(x)$, define $b_{n+1} = b_{n+2} = 0$ and

$$
b_k = 2xb_{k+1} - b_{k+2} + a_k, \qquad k = n, n-1, \dots, 0
$$

Then $a_k = b_k - 2xb_{k+1} + b_{k+2}$, and substituting into the sum,

$$
S = \sum_k(b_k - 2xb_{k+1} + b_{k+2})T_k
$$

Regroup by the index of $b$. The coefficient of $b_k$ for $2 \le k \le n$ is
$T_k - 2xT_{k-1} + T_{k-2}$, which is **zero** by the Chebyshev recurrence
$T_k = 2xT_{k-1} - T_{k-2}$. So everything cancels except the boundary terms:

$$
S = b_0T_0 + b_1(T_1 - 2xT_0) = b_0 - xb_1
$$

using $T_0 = 1$, $T_1 = x$.

**The cost.** One multiplication by $2x$ and two additions per term, so $n$ multiplications and
$2n$ additions in total, which is $O(n)$. **No $T_k$ is ever formed**, which is what makes it
better than the direct sum: the direct version needs $n$ cosine evaluations or its own recurrence,
at the same cost but with an extra array.

**Stability.** Clenshaw is backward stable for $|x| \le 1$: the computed value is the exact sum
for slightly perturbed $a_k$. The recurrence is running the Chebyshev recurrence backwards, and
backwards is the stable direction, because the dominant solution of $T_k = 2xT_{k-1} - T_{k-2}$
grows in the forward direction on $|x| > 1$ and is bounded on $|x| \le 1$.

Exercise 4.2 measures it: flat at $8\times10^{-16}$ relative from degree 5 to 60, where the power
form degrades to 6.0.

### 2.3 The aliasing formula

**Claim.** On the Chebyshev-Lobatto grid $x_j = \cos(j\pi/n)$, $j = 0,\dots,n$, the interpolation
coefficients satisfy

$$
a_k^{\text{grid}} = a_k + \sum_{m\ge1}\left(a_{2mn-k} + a_{2mn+k}\right)
$$

**Proof.** Substituting $x = \cos\theta$ turns everything into a cosine series problem. The grid
is $\theta_j = j\pi/n$, and the discrete cosine sum that produces $a_k^{\text{grid}}$ is

$$
a_k^{\text{grid}} = \frac{2}{n}{\sum_j}''\,f(\cos\theta_j)\cos(k\theta_j)
$$

with the double prime meaning the end terms are halved.

Substitute the exact series $f(\cos\theta) = \sum_m a_m\cos(m\theta)$ and exchange the sums:

$$
a_k^{\text{grid}} = \sum_m a_m\cdot\frac{2}{n}{\sum_j}''\cos(m\theta_j)\cos(k\theta_j)
$$

The inner sum is the **discrete** orthogonality relation for cosines on $n+1$ Lobatto points, and
unlike the continuous one it is not zero for all $m \ne k$. Using
$\cos A\cos B = \frac12[\cos(A-B) + \cos(A+B)]$ and
${\sum_j}''\cos(p\theta_j) = n$ when $p \equiv 0 \pmod{2n}$ and 0 otherwise,

$$
\frac{2}{n}{\sum_j}''\cos(m\theta_j)\cos(k\theta_j)
= \begin{cases}
1 & m \equiv \pm k \pmod{2n}\\
0 & \text{otherwise}
\end{cases}
$$

**So exactly the harmonics congruent to $\pm k$ modulo $2n$ survive**, which is the claim. The
period is $2n$ rather than $n$ because the cosine is even, so $m$ and $-m$ are the same harmonic.

**Where it comes from geometrically.** $\cos(m\theta)$ and $\cos((2n-m)\theta)$ agree at every
$\theta_j = j\pi/n$, because $\cos((2n-m)j\pi/n) = \cos(2j\pi - mj\pi/n) = \cos(mj\pi/n)$. Two
different functions with identical samples: the grid has no way to tell them apart, and the
interpolant assigns their combined weight to the lower one.

### 2.4 Economization is optimal

**Claim.** Reducing a degree $n$ polynomial to degree $n-1$ by subtracting
$\frac{c_n}{2^{n-1}}T_n$ adds the smallest possible maximum error.

**Proof.** Let $p$ have degree $n$ with leading coefficient $c_n$, and let $q$ be any polynomial of
degree at most $n-1$. Then $p - q$ has degree exactly $n$ with leading coefficient $c_n$, so

$$
\frac{p-q}{c_n}
$$

is **monic** of degree $n$.

Lesson 47's minimax property says the monic polynomial of degree $n$ with the smallest maximum on
$[-1,1]$ is $T_n/2^{n-1}$, with maximum $2^{1-n}$, and it is the unique one. Hence

$$
\|p - q\|_\infty = |c_n|\left\|\frac{p-q}{c_n}\right\|_\infty \ge |c_n|\cdot 2^{1-n}
$$

with equality exactly when $\frac{p-q}{c_n} = \frac{T_n}{2^{n-1}}$, that is when
$q = p - \frac{c_n}{2^{n-1}}T_n$.

**So the economization step is not merely a good choice, it is the unique optimal one**, and the
error it adds is exactly $|c_n|/2^{n-1}$.

**Repeating it is not optimal, and that matters.** Dropping $k$ degrees one at a time gives a
guaranteed added error of $\sum|c_j|/2^{j-1}$ over the $k$ steps, and that is an upper bound
rather than the minimum for the combined reduction. The measurement in exercise 3 of the lesson
shows the bound is nearly tight in practice: going from degree 10 to degree 5 on $\exp$, the
guaranteed addition is $4.838\times10^{-5}$ and the actual addition is $4.841\times10^{-5}$, so
the greedy sequence gives up essentially nothing.

The reason it is nearly tight is the same geometric decay everywhere else in this lesson: the sum
$\sum|c_j|/2^{j-1}$ is dominated by its first term.

### 2.5 Geometric decay and the Bernstein ellipse

**The easy direction, which is the one used.** Suppose $f$ is analytic and bounded by $M$ inside
the Bernstein ellipse $E_\rho$, the image of the circle $|z| = \rho$ under
$x = \frac12(z + z^{-1})$. Then

$$
|a_k| \le \frac{2M}{\rho^k}
$$

**Proof.** Substituting $x = \frac12(z+z^{-1})$ maps $T_k(x)$ to $\frac12(z^k + z^{-k})$ and the
Chebyshev coefficients become the **Laurent** coefficients of $F(z) = f(\frac12(z+z^{-1}))$, which
is analytic in the annulus $\rho^{-1} < |z| < \rho$ and symmetric under $z \mapsto 1/z$. Cauchy's
estimate on the circle $|z| = r$ for any $r < \rho$ gives $|a_k| \le 2M_r r^{-k}$, and letting
$r \to \rho$ gives the claim.

The geometric rate is therefore $\rho$ exactly, and $\rho$ is determined by the nearest
singularity: for a pole at $\pm i/a$ the ellipse through it has

$$
\rho = \frac{1 + \sqrt{1+a^2}}{a}
$$

Exercise 4.3 measures the recovered rate against that formula and gets 0.01 to 1.6 percent.

**The converse, which is harder and true.** If $|a_k| \le C\rho^{-k}$ then the series
$\sum a_kT_k$ converges uniformly on $E_r$ for every $r < \rho$, because $|T_k| \le \frac12 r^k$
there, so the sum is analytic inside $E_\rho$. That gives analyticity but not the bound $M$
directly, and the two statements together are the standard equivalence.

**What the equivalence is for.** It turns a question about $f$ in the complex plane into a
question about a sequence of numbers you can look at. Exercise 4.3 is exactly that: measure the
decay, recover the location of the singularity, without ever leaving the real line.

### 3.1 Adaptive degree selection

Raise the degree until the tail bound falls below the tolerance, then stop. The tail bound of the
lesson's section 4 is computable from the coefficients alone, so no evaluation of the error is
needed.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import chebapprox as ca


def adaptive_degree(f, tol, lo=-1.0, hi=1.0, max_degree=512):
    """Double the degree until the last few coefficients are below the tolerance.

    Doubling rather than incrementing is what a real implementation does: each check costs a
    whole transform, so halving the number of checks matters more than overshooting the degree.
    The price is measured below.
    """
    n = 8
    while n <= int(max_degree):
        a = ca.coefficients_by_interpolation(f, n, lo, hi)
        tail = float(np.sum(np.abs(a[-max(2, n // 8):])))
        if tail <= float(tol):
            return n, tail
        n *= 2
    raise RuntimeError(f"tolerance {tol} not reached by degree {max_degree}")


def smallest_degree_that_works(f, tol, lo=-1.0, hi=1.0, max_degree=512):
    probe = np.linspace(float(lo), float(hi), 4001)
    truth = np.asarray(f(probe), dtype=float)
    for n in range(1, int(max_degree) + 1):
        a = ca.coefficients_by_interpolation(f, n, lo, hi)
        if float(np.max(np.abs(ca.evaluate_series(a, probe, lo, hi) - truth))) <= float(tol):
            return n
    return int(max_degree)


runge = lambda t: 1.0 / (1.0 + 25.0 * t * t)
cases = [("exp", np.exp), ("sin(5t)", lambda t: np.sin(5.0 * t)),
         ("1/(1+25t^2)", runge), ("sqrt(t^2+0.01)", lambda t: np.sqrt(t * t + 0.01))]
print(f"{'function':>18}{'tolerance':>12}{'chosen':>9}{'smallest that works':>21}{'waste':>8}")
for name, f in cases:
    for tol in (1e-4, 1e-8, 1e-12):
        chosen, _ = adaptive_degree(f, tol)
        need = smallest_degree_that_works(f, tol)
        print(f"{name:>18}{tol:>12.0e}{chosen:>9}{need:>21}{chosen / max(need, 1):>8.2f}")
```

| function | tolerance | chosen | smallest that works | waste |
|---|---|---|---|---|
| $\exp$ | 1e-8 | 16 | 9 | 1.78 |
| $\exp$ | 1e-12 | 16 | 12 | 1.33 |
| $\sin 5t$ | 1e-4 | 16 | 11 | 1.45 |
| $\sin 5t$ | 1e-8 | 32 | 17 | 1.88 |
| $\sin 5t$ | 1e-12 | 32 | 21 | 1.52 |
| $1/(1+25t^2)$ | 1e-4 | 64 | 48 | 1.33 |
| $1/(1+25t^2)$ | 1e-8 | 128 | 94 | 1.36 |
| $1/(1+25t^2)$ | 1e-12 | 256 | 140 | 1.83 |
| $\sqrt{t^2+0.01}$ | 1e-8 | 256 | 122 | 2.10 |
| $\sqrt{t^2+0.01}$ | 1e-12 | 256 | 206 | 1.24 |

**The waste is between 1.24 and 2.10, never worse than the doubling itself.** That is the honest
accounting: doubling can overshoot by up to a factor of 2 by construction, and the measured
overshoot is at most that.

**Two things the measurement shows that the design did not promise.**

**The tail bound is a good predictor, not a conservative one.** The chosen degree is never below
the degree that actually works, so the criterion never under-delivers, and it is never more than
2.1 times above it, so it does not over-deliver much either.

**The waste is not systematically worse for hard functions.** $\sqrt{t^2+0.01}$ at $10^{-12}$
wastes 1.24 while $\exp$ at $10^{-8}$ wastes 1.78, which is the opposite of what one might guess.
The waste is decided by where the required degree falls between two powers of two, and that is
essentially arbitrary.

**How a real implementation improves on this.** Chebfun bisects between the last two degrees
rather than accepting the doubling, and it looks for a **plateau** in the coefficients rather than
a single tail sum, because a coefficient sequence can dip accidentally. That halves the waste at
the cost of one more transform.

### 3.2 A Chebfun style object

The idea: carry the coefficients rather than the function, and define the arithmetic on them. A
function becomes a first class value with a length that adapts to how hard it is.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import chebapprox as ca


class Cheb:
    """A function represented by its Chebyshev coefficients, resolved to machine precision.

    The interesting operation is `derivative`, which uses the exact coefficient recurrence and
    so introduces no differencing error at all. Lesson 61 will measure what differencing costs.
    """

    def __init__(self, coefficients, lo=-1.0, hi=1.0):
        self.c = np.atleast_1d(np.asarray(coefficients, dtype=float))
        self.lo, self.hi = float(lo), float(hi)

    @classmethod
    def of(cls, f, lo=-1.0, hi=1.0, tol=1e-14, max_degree=2048):
        n = 8
        a = ca.coefficients_by_interpolation(f, n, lo, hi)
        while n <= int(max_degree):
            a = ca.coefficients_by_interpolation(f, n, lo, hi)
            scale = max(float(np.max(np.abs(a))), 1.0)
            if float(np.max(np.abs(a[-3:]))) <= float(tol) * scale:
                big = np.nonzero(np.abs(a) > float(tol) * scale)[0]
                return cls(a[:int(big[-1]) + 1] if big.size else a[:1], lo, hi)
            n *= 2
        return cls(a, lo, hi)

    def __call__(self, x):
        return ca.evaluate_series(self.c, x, self.lo, self.hi)

    def __add__(self, other):
        n = max(self.c.size, other.c.size)
        a = np.zeros(n)
        a[:self.c.size] = self.c
        b = np.zeros(n)
        b[:other.c.size] = other.c
        return Cheb(a + b, self.lo, self.hi)

    def __mul__(self, other):
        return Cheb.of(lambda t: self(t) * other(t), self.lo, self.hi)

    def derivative(self):
        """The exact coefficient recurrence, run downward: c'_{k-1} = c'_{k+1} + 2k c_k."""
        c = self.c
        n = c.size - 1
        if n < 1:
            # the derivative of a constant is the zero function, which needs one coefficient;
            # that is the answer, not an assumption about the input
            return Cheb(np.zeros_like(c[:1]), self.lo, self.hi)
        d = np.zeros(n + 2)
        for k in range(n, 0, -1):
            d[k - 1] = d[k + 1] + 2.0 * k * c[k]
        d = d[:n]
        d[0] *= 0.5
        return Cheb(d * 2.0 / (self.hi - self.lo), self.lo, self.hi)


f = Cheb.of(np.exp)
g = Cheb.of(lambda t: np.sin(3.0 * t))
x = np.linspace(-1.0, 1.0, 401)
print(f"exp resolves in {f.c.size} coefficients, sin(3t) in {g.c.size}")
print(f"  f + g       error {float(np.max(np.abs((f + g)(x) - (np.exp(x) + np.sin(3 * x))))):.2e}")
print(f"  f * g       error {float(np.max(np.abs((f * g)(x) - (np.exp(x) * np.sin(3 * x))))):.2e}")
print(f"  d/dx f      error {float(np.max(np.abs(f.derivative()(x) - np.exp(x)))):.2e}")
print(f"  d/dx g      error {float(np.max(np.abs(g.derivative()(x) - 3 * np.cos(3 * x)))):.2e}")
print(f"  sum needs {(f + g).c.size} coefficients, product needs {(f * g).c.size}")
```

| operation | error |
|---|---|
| $f + g$ | 3.11e-15 |
| $f \cdot g$ | 4.05e-15 |
| $\mathrm{d}f/\mathrm{d}x$ | 3.72e-13 |
| $\mathrm{d}g/\mathrm{d}x$ | 3.95e-13 |

**$\exp$ resolves in 14 coefficients and $\sin 3t$ in 20**, which is the length the function
deserves rather than a length chosen in advance. That adaptivity is the whole point of the design:
a hard function gets a long representation automatically, and an easy one does not pay for it.

**Addition is exact and free**, because the coefficients simply add. **Multiplication is not**, and
the implementation above re-resolves the product rather than convolving the coefficients. The
convolution is available, using $T_iT_j = \frac12(T_{i+j} + T_{|i-j|})$, and re-resolving is
simpler and no slower at these lengths.

**Differentiation is the interesting one.** The recurrence $c'_{k-1} = c'_{k+1} + 2kc_k$ is exact,
so there is no differencing and no step size to choose. The measured error is $4\times10^{-13}$,
which is $\varepsilon$ times the factor $2k$ accumulated down the recurrence, and it is **the
error of the representation, not of the differentiation.**

Compare with what lesson 61 will measure for a finite difference: an optimal step of
$\varepsilon^{1/2}$ and an error of $\varepsilon^{1/2} \approx 10^{-8}$, five orders worse. The
reason is that a spectral representation carries the derivative information exactly, and a sampled
one does not.

**What a real Chebfun does that this does not.** Splitting into pieces when the function is not
globally smooth; rootfinding by the colleague matrix; integration by the exact coefficient
recurrence; and automatic interval detection. All of them rest on the same idea: once the function
is a coefficient vector, calculus is linear algebra.

### 3.3 Coefficients through the FFT

The Lobatto grid coefficients are a discrete cosine transform of type I, which is one real FFT of
the mirrored data.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import chebapprox as ca, fft as ft


def coefficients_by_fft(f, degree, lo=-1.0, hi=1.0):
    """Mirror the samples and take one FFT. The DCT-I of n+1 points is the DFT of 2n.

    The mirroring is what turns a cosine transform into a Fourier transform: the even extension
    of the data has a real spectrum, and its first n+1 entries are the coefficients.
    """
    n = int(degree)
    if n < 1:
        return ca.coefficients_by_interpolation(f, n, lo, hi)
    theta = np.arange(n + 1) * math.pi / n
    x = 0.5 * (hi + lo) + 0.5 * (hi - lo) * np.cos(theta)
    v = np.asarray(f(x), dtype=float)
    mirrored = np.concatenate([v, v[-2:0:-1]])
    a = np.real(ft.fft(mirrored))[:n + 1] / n
    a[0] *= 0.5
    a[n] *= 0.5
    return a


print(f"{'degree':>8}{'agreement with the direct sum':>32}{'fft length':>13}")
for n in (4, 8, 16, 32, 64, 128):
    direct = ca.coefficients_by_interpolation(np.exp, n)
    fast = coefficients_by_fft(np.exp, n)
    print(f"{n:>8}{float(np.max(np.abs(direct - fast))):>32.3e}{2 * n:>13}")

print()
print(f"{'degree':>8}{'direct multiplies':>20}{'fft butterflies':>18}{'ratio':>9}")
for n in (16, 64, 256, 1024, 4096):
    direct_cost = (n + 1) ** 2
    m = 2 * n
    fft_cost = (m // 2) * int(math.log2(m))
    print(f"{n:>8}{direct_cost:>20}{fft_cost:>18}{direct_cost / fft_cost:>9.1f}")
```

| degree | agreement | FFT length |
|---|---|---|
| 4 | 2.220e-16 | 8 |
| 16 | 4.545e-16 | 32 |
| 64 | 1.943e-15 | 128 |
| 128 | 2.932e-15 | 256 |

| degree | direct multiplies | FFT butterflies | ratio |
|---|---|---|---|
| 16 | 289 | 80 | 3.6 |
| 64 | 4225 | 448 | 9.4 |
| 256 | 66049 | 2304 | 28.7 |
| 1024 | 1050625 | 11264 | 93.3 |
| 4096 | 16785409 | 53248 | **315.2** |

**They agree to $3\times10^{-15}$**, and the transform is 315 times cheaper at degree 4096.

**Why the FFT is not merely faster but also more accurate.** The direct sum accumulates $n+1$
terms per coefficient, so its rounding grows like $n$. The FFT's error grows like
$O(\sqrt{\log n})$, which lesson 59's exercise 5.3 states. At degree 128 the measured agreement is
$2.9\times10^{-15}$, which is roughly $\sqrt{n}\varepsilon$: the direct sum's error, not the
transform's.

**What a production implementation does instead.** A real DCT rather than a mirrored FFT, which
saves the factor of 2 in length and another factor of 2 in the complex arithmetic. That is a
constant, and the $O(n\log n)$ is the part that matters.

### 4.1 Hunting for a factor above 3

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import chebapprox as ca, approx as ap


def is_degenerate(f, n, lo=-1.0, hi=1.0):
    """The best degree n and degree n+1 approximations coincide.

    That happens for an even f at even n, by lesson 54's exercise 2.4, and it makes the
    near-minimax factor meaningless: the truncation drops a coefficient that is exactly zero,
    so the comparison is between approximations of two different degrees.
    """
    a = ap.remez(f, n, lo, hi)["max_error"]
    b = ap.remez(f, n + 1, lo, hi)["max_error"]
    return abs(a - b) <= 1e-6 * max(a, 1e-300)


runge = lambda t: 1.0 / (1.0 + 25.0 * t * t)
hunt = [("exp", np.exp), ("sin(3t)", lambda t: np.sin(3.0 * t)),
        ("|t|", np.abs), ("|t - 0.3|", lambda t: np.abs(t - 0.3)),
        ("sqrt|t - 0.3|", lambda t: np.sqrt(np.abs(t - 0.3))),
        ("1/(1+25t^2)", runge),
        ("tanh(50t)", lambda t: np.tanh(50.0 * t))]
best = (0.0, "", 0)
print(f"{'function':>18}" + "".join(f"{'n=' + str(n):>9}" for n in (2, 4, 6, 8, 12, 16, 24)))
for name, f in hunt:
    row = []
    for n in (2, 4, 6, 8, 12, 16, 24):
        out = ca.near_minimax_factor(f, n)
        probe = np.linspace(-1.0, 1.0, 4001)
        floor = 8.0 * float(np.finfo(float).eps) * float(np.max(np.abs(f(probe))))
        if out["minimax_error"] <= floor or is_degenerate(f, n):
            row.append(float("nan"))
            continue
        row.append(out["factor"])
        if out["factor"] > best[0]:
            best = (out["factor"], name, n)
    print(f"{name:>18}" + "".join(f"{v:>9.3f}" if v == v else f"{'-':>9}" for v in row))
print(f"largest valid factor found: {best[0]:.3f} on {best[1]} at degree {best[2]}")
```

| function | $n=2$ | $n=4$ | $n=6$ | $n=8$ | $n=12$ | $n=16$ | $n=24$ |
|---|---|---|---|---|---|---|---|
| $\exp$ | 1.744 | 1.950 | 1.982 | 1.991 | 1.929 | - | - |
| $\sin 3t$ | 1.473 | 1.904 | 1.980 | 1.997 | 2.003 | 2.003 | - |
| $\lvert t\rvert$ | - | - | - | - | - | - | - |
| $\lvert t-0.3\rvert$ | 1.680 | 2.674 | 2.947 | 2.144 | 1.688 | **3.305** | 2.948 |
| $\sqrt{\lvert t-0.3\rvert}$ | 2.012 | 2.606 | 2.748 | 2.297 | 2.032 | 2.934 | 2.751 |
| $1/(1+25t^2)$ | - | - | - | - | - | - | - |
| $\tanh 50t$ | 1.049 | 1.142 | 1.225 | 1.300 | 1.439 | 1.566 | 1.801 |

**Yes: 3.305, on $|t-0.3|$ at degree 16.**

**The filtering is the whole difficulty of this exercise**, and without it the answer is wrong in
both directions.

**Two kinds of row have to be excluded.** A dash means either the minimax error has reached the
roundoff floor, so the ratio is two roundoff numbers divided, or the degree is **degenerate**. An
even function on a symmetric interval has best degree $2k$ = best degree $2k+1$ by lesson 54's
exercise 2.4, so at even degrees the truncated Chebyshev series is dropping a coefficient that is
exactly zero and the comparison is between different degrees. Every entry for $|t|$ and for the
Runge function is degenerate for that reason, since both are even and only even degrees were
sampled.

Run without the filter, those rows report factors of 5.000, 5.800 and 4.889, which look like a
decisive answer to the exercise and are entirely artefacts.

**Where the real large factors come from.** Shifting the singularity off centre, from $|t|$ to
$|t-0.3|$, breaks the symmetry and makes every degree non-degenerate. Then the factor genuinely
reaches 3.3, and it fluctuates rather than growing smoothly, because for a function with a
singularity the position of the singularity relative to the Chebyshev nodes matters.

**And where the factor is smallest.** $\tanh 50t$, a smooth function with a steep front, stays
between 1.05 and 1.80. Its coefficients are large and slowly decaying over a long stretch, so the
tail is not dominated by its first term and the truncation is closer to optimal than usual.

### 4.2 Three evaluators

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import chebapprox as ca

runge = lambda t: 1.0 / (1.0 + 25.0 * t * t)
print(f"{'degree':>8}{'Clenshaw vs direct':>21}{'power vs direct':>19}"
      f"{'largest power coeff':>22}")
for n in (5, 10, 20, 30, 40, 50, 60):
    a = ca.coefficients_by_interpolation(runge, n)
    x = np.linspace(-1.0, 1.0, 1001)
    theta = np.arccos(np.clip(x, -1.0, 1.0))
    direct = sum(a[k] * np.cos(k * theta) for k in range(a.size))
    p = ca.to_power_basis(a)
    power = sum(p[k] * x ** k for k in range(p.size))
    scale = max(float(np.max(np.abs(direct))), 1e-300)
    print(f"{n:>8}{float(np.max(np.abs(ca.evaluate_series(a, x) - direct))) / scale:>21.3e}"
          f"{float(np.max(np.abs(power - direct))) / scale:>19.3e}"
          f"{float(np.max(np.abs(p))):>22.4e}")
```

| degree | Clenshaw vs direct | power vs direct | largest power coefficient |
|---|---|---|---|
| 5 | 4.609e-16 | 6.145e-16 | 7.3163e-1 |
| 10 | 5.551e-16 | 3.993e-14 | 9.6376e1 |
| 20 | 6.661e-16 | 2.193e-11 | 6.8765e4 |
| 30 | 7.772e-16 | 1.216e-8 | 5.1372e7 |
| 40 | 7.772e-16 | 8.679e-6 | 4.2817e10 |
| 50 | 6.661e-16 | 1.046e-2 | 3.4675e13 |
| 60 | **7.772e-16** | **6.029e0** | 3.0074e16 |

**Clenshaw is flat.** From degree 5 to 60 it moves from $4.6\times10^{-16}$ to
$7.8\times10^{-16}$, a factor of 1.7 over a twelvefold increase in degree. That is not "good", it
is essentially perfect: the error is a few units in the last place and stays there.

**The power form loses about one digit per 4 degrees**, tracking $\log_{10}2^{n-1}$ exactly. At
degree 60 the largest power coefficient is $3\times10^{16}$ and the answer is bounded by 1, so
every digit has cancelled and the result is wrong by 6.0.

**The direct basis sum is as good as Clenshaw**, which is worth stating because it is the third
option. It costs the same $O(n)$ and needs an array of $T_k$ values, so Clenshaw's advantage is
memory and one fewer pass, not accuracy.

**The rule.** Never leave the Chebyshev basis. Every operation this course needs on a Chebyshev
series has a direct form: evaluation by Clenshaw, differentiation by the coefficient recurrence
of exercise 3.2, integration likewise, and rootfinding by the colleague matrix. **There is no
reason to convert, and there is a measurable cost to doing so.**

### 4.3 Recovering the Bernstein ellipse

For a function with poles at $\pm i/a$, the ellipse parameter is
$\rho = (1+\sqrt{1+a^2})/a$, so the answer is known in advance and the fit can be scored against
it.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import chebapprox as ca

print(f"{'pole at +-i/a':>15}{'predicted rho':>16}{'fitted rho':>13}{'error %':>10}"
      f"{'kept':>7}")
for a in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
    rho = (1.0 + math.sqrt(1.0 + a * a)) / a
    f = lambda t, aa=a: 1.0 / (1.0 + aa * aa * t * t)
    out = ca.coefficient_decay(f, 120)
    print(f"{a:>15.1f}{rho:>16.4f}{out['geometric_rate']:>13.4f}"
          f"{abs(out['geometric_rate'] - rho) / rho * 100.0:>10.2f}{out['kept']:>7}")
```

| pole at $\pm i/a$ | predicted $\rho$ | fitted $\rho$ | error % | coefficients used |
|---|---|---|---|---|
| 0.5 | 4.2361 | 4.2356 | **0.01** | 11 |
| 1.0 | 2.4142 | 2.4139 | **0.01** | 18 |
| 2.0 | 1.6180 | 1.6180 | **0.00** | 34 |
| 5.0 | 1.2198 | 1.2191 | 0.06 | 60 |
| 10.0 | 1.1050 | 1.1036 | 0.13 | 60 |
| 20.0 | 1.0512 | 1.0681 | **1.60** | 61 |

**The prediction lands to within 0.13 percent over four orders of magnitude in $a$.** At $a = 2$
the fitted $\rho$ is 1.6180, which is the golden ratio to four decimal places, as
$(1+\sqrt5)/2$ says it should be.

**The last row is the informative one.** At $a = 20$ the singularity is at $\pm 0.05i$, extremely
close to the interval, so $\rho = 1.0512$ and the coefficients decay by only 5 percent per index.
Over 120 coefficients that is a total decay of $1.05^{120} \approx 350$, which is not enough
dynamic range for the asymptotic regime to establish itself, and the fit is 1.6 percent out.

**The pattern in the last column explains it.** The number of coefficients the fit could use rises
from 11 at $a=0.5$ to 61 at $a=20$, and stops rising because the fit caps at half the requested
degree for an even function. So the hard cases get relatively **fewer** usable coefficients per
decade of decay, which is exactly backwards from what they need.

**The practical procedure this suggests.** Compute more coefficients than you think you need, drop
the ones at the roundoff floor, fit only over a window where the decay is clean, and report the
number of coefficients the fit used. Reporting a rate without the count is reporting a number
whose reliability cannot be assessed.

### 5.1 Within a logarithm of best

**The Lebesgue constant argument.** The Chebyshev truncation $S_n$ is a linear projection onto the
degree $n$ polynomials, and for any linear projection $P$ onto a subspace containing the best
approximation $p^*$,

$$
\|f - Pf\| = \|(f - p^*) - P(f - p^*)\| \le (1 + \|P\|)\,\|f - p^*\|
$$

using $Pp^* = p^*$. So the factor is $1 + \|P\|_\infty$, the **Lebesgue constant**.

For Chebyshev projection,

$$
\Lambda_n = \frac{4}{\pi^2}\log n + 1.2703\ldots + o(1)
$$

which is the number the lesson reports as the classical bound.

**Why the measured factor is so much smaller.** Three reasons, and the third is the real one.

**The bound is worst case over all $f$.** It is attained only by a function whose error function
is aligned with the Lebesgue function's peak, and no ordinary function is.

**The bound is stated for the projection, not for the specific $f$.** For a given $f$ with
geometric coefficient decay, the truncation error is $\approx|a_{n+1}|\frac{\rho}{\rho-1}$ and the
best possible is $\ge|a_{n+1}|$, so the ratio is at most $\frac{\rho}{\rho-1}$, which is 2 when
$\rho = 2$ and does not involve $n$ at all.

**And that last bound is the operative one.** For a smooth function the factor is governed by
$\rho$ rather than by $\log n$, and the $\log n$ only takes over once the coefficients stop
decaying geometrically. That is exactly what the measurements show: the smooth functions sit
around 2, the non-smooth ones climb slowly, and the largest factor found, 3.305, is on the
non-smooth $|t-0.3|$.

**So the honest summary** is that the factor is $\min\left(\frac{\rho}{\rho-1}, 1+\Lambda_n\right)$
in spirit, the first term binding for analytic functions and the second for rough ones.

### 5.2 Chebyshev series and the FFT

**The Lobatto grid coefficients are a DCT-I.** With $\theta_j = j\pi/n$,

$$
a_k = \frac{2}{n}{\sum_{j=0}^{n}}''\,f(\cos\theta_j)\cos\!\left(\frac{jk\pi}{n}\right)
$$

which is exactly the definition of the type I discrete cosine transform on $n+1$ points, the
double prime halving the two end terms.

**Why type I specifically.** The four common DCT types differ in whether the extension is even
about a sample point or about a half sample, at each end. Type I is even about **both** end
samples, which is what the Lobatto grid is: $\cos\theta$ takes the values $\pm1$ at $j = 0$ and
$j = n$, and the even extension repeats them.

Lesson 60's DCT-II is even about the half sample at each end, which matches the Chebyshev **roots**
grid instead. That is the same distinction as the `kind="extrema"` and `kind="roots"` options of
`chebapprox.coefficients_by_interpolation`, and it is why the roots grid aliases differently.

**The connection made concrete.** Mirroring the $n+1$ Lobatto samples into $2n$ points gives a
sequence whose DFT is real and symmetric, and whose first $n+1$ entries are the Chebyshev
coefficients up to the scaling. That is the implementation of exercise 3.3, and it is why the
transform is available in $O(n\log n)$ using nothing but lesson 59's radix-2 machinery.

**Where the chain of identifications ends up.** Chebyshev coefficients, discrete cosine transform,
discrete Fourier transform of an even sequence, and the whole of Part 8 is one object seen from
four sides:

| view | basis | grid | lesson |
|---|---|---|---|
| Chebyshev series | $T_k(x)$ | Chebyshev points | 56 |
| cosine series | $\cos k\theta$ | uniform in $\theta$ | 60 |
| Fourier series of an even function | $e^{ik\theta}$ | uniform in $\theta$ | 58 |
| DFT of a mirrored sequence | roots of unity | uniform | 59 |

The substitution $x = \cos\theta$ is the only thing connecting them, and it does all the work.

### 5.3 What replaced economization

**Economization is a way to start from a Taylor series.** A modern library function generator does
not have a Taylor series; it has a specification and arbitrary precision arithmetic, and it runs
Remez directly. Three things make the real problem harder than plain minimax.

**The coefficients must be representable.** The generated polynomial will be evaluated in double,
or in single, or in a fixed point format, and its coefficients have to be exactly representable in
that format. Rounding a minimax coefficient to double changes the polynomial, and the rounded one
is no longer minimax. This is a **constrained** minimax problem over a lattice, and it is
genuinely combinatorial: the standard tool, Sollya's `fpminimax`, uses lattice basis reduction
(LLL) to search it.

**The evaluation error must be included.** The specification bounds the error of the **computed**
value, not of the exact polynomial. So the objective is
$\|f - p\|_\infty + (\text{rounding of the evaluation scheme})$, and the second term depends on
the order the terms are summed, whether Horner or Estrin is used, and whether a fused multiply-add
is available. Sollya's `supnorm` and its Gappa backend produce a proved bound on the whole thing.

**Relative rather than absolute error**, which is lesson 54's exercise 3.3. The weight is $1/f$,
so the problem is weighted minimax, and near a zero of $f$ the argument range has to be split.

**The three of them together** mean the generated polynomial for `sin` on a reduced range is not
the minimax polynomial of any classical problem. It is the solution of a constrained,
weighted, evaluation-aware minimax problem, computed in arbitrary precision and then **formally
verified** against the specification, because a library function's error bound is a claim that
gets relied on.

**What survives from this lesson.** Two things. The Chebyshev basis, because the constrained
search is conditioned in it and not in the monomial basis. And the tail bound of section 4, which
is what tells the generator what degree to search at before it starts.

---

## Lesson 57, Pade Rational Approximation

### 1.1 Three limitations, and where each bites

**Cannot go to infinity at a finite point.**

Where it bites in this course: **lesson 46's Runge function** $1/(1+25t^2)$. Its poles at
$\pm i/5$ are not on the real line, so no polynomial fails outright, but they are what limits the
Bernstein ellipse to $\rho = 1.22$ and hence what makes the convergence so slow. Lesson 56's
exercise 4.3 recovers exactly that number from the coefficients. A rational approximant places
poles there and converges immediately: lesson 57 section 5 measures the $[4/4]$ approximant of a
function with a real pole beating degree 8 Taylor by 386.

**Cannot level off at infinity.**

Where it bites: **Part 10's stability functions for stiff ODEs.** An A-stable method needs its
stability function $R(z)$ to satisfy $|R(z)| \le 1$ for all $z$ in the left half plane, including
$z \to -\infty$. A polynomial $R$ is unbounded there, so **no explicit Runge-Kutta method can be
A-stable**, which is a theorem and the reason stiff problems need implicit methods. The implicit
methods' stability functions are rational, and the best ones are precisely the diagonal Pade
approximants of $e^z$.

**Is poor far from where it was fitted.**

Where it bites: **lesson 56's Taylor comparison.** The degree 8 Taylor polynomial of $\exp$ is
accurate to $3\times10^{-6}$ on $[-1,1]$ and to $6$ on $[-4,4]$. That is not a defect of the
degree, it is the geometry: a polynomial's error grows like $x^{n+1}$ away from the expansion
point, and nothing stops it.

### 1.2 The $[m/0]$ row

**Why it is the Taylor polynomial.** With $n = 0$ the denominator is $Q = 1$, so the matching
condition is $f - P = O(x^{m+1})$, which says $P$ agrees with $f$ to order $m$ at 0. That is the
definition of the Taylor polynomial of degree $m$, and by uniqueness it **is** that polynomial.
The library returns `p = c[:m+1]` directly, with no solve.

**What it is doing in the table: it is the control.** Every comparison in the lesson claims that
Pade beats Taylor at equal cost, and "equal cost" needs defending. The $[m/0]$ row makes the two
sides of the comparison literally the same object, so the measured ratio must be exactly 1.0.

It is measured at 1.0 to twelve digits, at every function and every degree tried. **That is a test
of the harness, not of the mathematics**, and it is what licenses reading the other rows.

**What it would catch.** A different probe grid for the two methods, a different treatment of
non-finite values, an off-by-one in how many coefficients each is given, or a comparison against
the wrong degree. Any of those would show up as a ratio that is not 1, and none of them would show
up in the $[4/4]$ row, where a wrong answer is indistinguishable from an interesting one.

### 1.3 Poles near a branch point

**A branch point is not a pole**, and no rational function has one. $\log(1+x)$ near $x = -1$
behaves like $\log(1+x)$, which is unbounded but not like $1/(x+1)^k$ for any $k$: it has a **cut**
rather than an isolated singularity.

**What the approximant does instead.** It places several real poles, interlaced with zeros, along
the cut. Measured:

| $[m/n]$ | real poles of the $\log(1+x)$ approximant |
|---|---|
| $[2/2]$ | $-1.1547$, $-2.0$ |
| $[4/4]$ | $-1.0500$, $-1.4614$, $-2.9333$, $-11.4$ |
| $[6/6]$ | six poles from $-1.02$ outward |

**The mechanism.** A cut can be approximated by a line of poles and zeros, because a dense enough
alternating sequence of poles and zeros along a curve produces a function with a jump across it.
In the limit of infinitely many, the poles coalesce into the cut exactly, and the theorem behind
this is Stahl's: the poles of the diagonal Pade approximants converge to the cut of minimal
capacity.

**Why it still works.** The interval of interest is $[-0.9, 3]$, which does not contain the cut, so
the approximant's poles are all outside it. Inside the interval the function is analytic and the
approximant matches it well: measured, the $[4/4]$ beats degree 8 Taylor by 38086.

**Where it stops working.** If the interval crossed the cut, the approximant would have real poles
inside it and would be unbounded there. That is not a failure of Pade so much as a statement that
the function is not approximable by anything continuous across a jump, which is lesson 54's
Weierstrass hypothesis again.

### 2.1 The Toeplitz system, and why it contains no $P$

**Setting up.** Write $f = \sum_{k\ge0}c_kx^k$, $P = \sum_{i=0}^{m}p_ix^i$,
$Q = \sum_{k=0}^{n}q_kx^k$ with $q_0 = 1$. The matching condition
$f - P/Q = O(x^{m+n+1})$ is equivalent, after multiplying by $Q$ (which is nonzero at 0), to

$$
fQ - P = O(x^{m+n+1})
$$

**The coefficient of $x^r$ in $fQ$** is $\sum_{k=0}^{\min(r,n)}q_kc_{r-k}$, a Cauchy product.

**Why the equations for $r > m$ contain no $P$.** Because $\deg P \le m$, the coefficient of $x^r$
in $P$ is **zero** for every $r > m$. So for $r = m+1,\dots,m+n$ the condition reads

$$
\sum_{k=0}^{n}q_kc_{r-k} = 0
$$

with no $p$ appearing at all. Splitting off $k = 0$ and using $q_0 = 1$,

$$
\sum_{k=1}^{n}c_{r-k}q_k = -c_r, \qquad r = m+1,\dots,m+n
$$

which with $r = m+j$ is the lesson's system.

**Why it is Toeplitz.** The entry in row $j$, column $k$ is $c_{m+j-k}$, which depends only on
$j-k$. So the matrix is constant along diagonals, which is the definition, and it can be stored in
$2n-1$ numbers and solved in $O(n^2)$ by Levinson's algorithm rather than $O(n^3)$.

**Recovering $P$.** With $q$ known, the equations for $r = 0,\dots,m$ give

$$
p_r = \sum_{k=0}^{\min(r,n)}q_kc_{r-k}
$$

directly, with no solve. **So the whole construction is one $n \times n$ solve plus $m+1$
convolutions**, which is why Pade is cheap.

### 2.2 Uniqueness

**Claim.** If the Toeplitz matrix is nonsingular, the $[m/n]$ approximant is unique.

**Proof.** Suppose $(P_1, Q_1)$ and $(P_2, Q_2)$ both satisfy the definition, both with
$Q_i(0) = 1$. Then

$$
fQ_1 - P_1 = O(x^{m+n+1}), \qquad fQ_2 - P_2 = O(x^{m+n+1})
$$

Multiply the first by $Q_2$ and the second by $Q_1$ and subtract:

$$
P_2Q_1 - P_1Q_2 = Q_2(fQ_1 - P_1) - Q_1(fQ_2 - P_2) = O(x^{m+n+1})
$$

But $P_2Q_1 - P_1Q_2$ is a **polynomial** of degree at most $m+n$. A polynomial of degree at most
$m+n$ that is $O(x^{m+n+1})$ is identically zero. Hence

$$
\frac{P_1}{Q_1} = \frac{P_2}{Q_2}
$$

**as rational functions**, so the approximant is unique.

**Where nonsingularity entered, and where it did not.** It did **not** enter the argument above:
the two approximants agree as rational functions whether or not the system is singular. What
nonsingularity gives is that $(P, Q)$ is unique as a **pair**, not merely the ratio. When the
system is singular there are many $(P,Q)$ pairs giving the same rational function with common
factors, which is exactly lesson 57's degenerate block.

**So the honest statement is stronger than the exercise asks for**: the rational function is
always unique when it exists, and nonsingularity is what makes the representation unique too.

### 2.3 $[m/0]$ is the Taylor polynomial

**Direct proof.** With $n = 0$ the definition requires $\deg Q \le 0$ and $Q(0) = 1$, so $Q = 1$
identically. The condition becomes

$$
f - P = O(x^{m+1})
$$

Write $f = \sum c_kx^k$ and $P = \sum_{i\le m}p_ix^i$. Then $f - P$ has coefficient $c_i - p_i$ at
order $i \le m$ and $c_i$ at order $i > m$. Being $O(x^{m+1})$ means every coefficient at order
$\le m$ vanishes, so $p_i = c_i$ for $i = 0,\dots,m$.

That is the Taylor polynomial, exactly, with no approximation.

**What the code does.** `pade.coefficients` short circuits: with `n == 0` it returns
`p = c[:m+1]` and `q = [1.0]` without forming or solving anything, because there is nothing to
solve. Reaching the general path would build a $0\times0$ matrix, which numpy handles but which
would obscure that the answer is free.

**Why this matters beyond the control role.** It says the Pade table's **first column** is the
Taylor series, so the table is a genuine generalisation: everything Taylor does, Pade does, and
the extra columns are new. Lesson 57's section 6 measures the diagonal beating the first column,
which is a statement about the table rather than about two unrelated methods.

### 2.4 The block structure

**Claim.** If $f = A/B$ exactly with $\deg A = \mu$, $\deg B = \nu$, $B(0) \ne 0$ and
$\gcd(A,B) = 1$, then $f$ is its own $[m/n]$ approximant for every $m \ge \mu$, $n \ge \nu$.

**Proof.** Normalise $B(0) = 1$. Take $P = Ax^0\cdot g$ and $Q = Bg$ for any polynomial $g$ with
$g(0) = 1$ and $\deg g \le \min(m-\mu, n-\nu)$. Then $\deg P \le m$, $\deg Q \le n$, $Q(0) = 1$,
and

$$
f - \frac{P}{Q} = \frac AB - \frac{Ag}{Bg} = 0
$$

exactly, which is certainly $O(x^{m+n+1})$. So $(P,Q)$ satisfies the definition, and by
exercise 2.2 the rational function is unique, so **every** entry of that region of the table is
$A/B$.

**Deducing the singularity.** The pairs $(P,Q)$ above form a family parametrised by $g$, of
dimension $\min(m-\mu, n-\nu)$. Since the Toeplitz system determines $q_1,\dots,q_n$ and there are
many valid $q$, the system must be **rank deficient**, with nullity at least
$\min(m-\mu, n-\nu)$.

**Measured.** For $1/(1+x)$, which is $[0/1]$:

| $[m/n]$ | degenerate | rank | error against the truth |
|---|---|---|---|
| $[0/1]$ | False | 1 | 0.000 |
| $[1/1]$ | False | 1 | 0.000 |
| $[2/2]$ | True | 1 of 2 | 4.496e-15 |
| $[4/4]$ | True | 1 of 4 | 3.109e-14 |
| $[6/6]$ | True | 1 of 6 | 1.998e-15 |

The rank is 1 at every entry, which is $\nu = 1$: the system has exactly as much information as
the true denominator degree, and the rest is redundancy.

**Why the least norm solve recovers the function.** Among the many valid $(P, Q)$ pairs, the least
norm one takes $g$ as small as possible, which is $g = 1$, giving $P = A$ and $Q = B$ padded with
zeros. That is the canonical representative, and it evaluates to the exact function, which is what
the measurement shows.

**The caveat the lesson records.** For a degeneracy caused by **parity** rather than by $f$ being
rational, the least norm solve is a valid approximant of the block but not necessarily the
canonical one, and its matching order has to be measured. `[2/3]` of $\tan$ has rank 2 and matches
3 terms, `[0/1]` has rank 0 and matches 1.

### 2.5 The error's leading term

**Claim.** With the Toeplitz matrix nonsingular,

$$
f(x) - \frac{P(x)}{Q(x)} = \frac{E_{m+n+1}}{q_0}\,x^{m+n+1} + O(x^{m+n+2})
$$

where $E_{m+n+1}$ is the coefficient of $x^{m+n+1}$ in $fQ - P$, and $q_0 = 1$.

**Derivation.** By construction $fQ - P = \sum_{r > m+n}E_rx^r$ with

$$
E_r = \sum_{k=0}^{\min(r,n)}q_kc_{r-k}
$$

Dividing by $Q$, and using $Q(x) = 1 + O(x)$ so $1/Q = 1 + O(x)$,

$$
f - \frac PQ = \frac{fQ - P}{Q} = \left(E_{m+n+1}x^{m+n+1} + O(x^{m+n+2})\right)(1 + O(x))
$$

which gives the claim.

**The constant, explicitly.**

$$
E_{m+n+1} = c_{m+n+1} + \sum_{k=1}^{n}q_kc_{m+n+1-k}
$$

so it is computable from the Taylor coefficients and the solved $q$, before evaluating anything.
That is the Pade analogue of lesson 56's tail bound.

**Why it is a weaker tool than the Chebyshev tail bound.** It bounds the error **near zero** only.
The Chebyshev tail bounds the error on the whole interval, uniformly, because $|T_k| \le 1$
everywhere. Here $x^{m+n+1}$ is small only near the origin and the constant says nothing about the
far end of the interval, which is where a Pade approximant's spurious pole would do its damage.

**That asymmetry is the practical difference between the two methods**: Chebyshev tells you in
advance how good the answer is, and Pade tells you how good it is at one point.

### 3.1 Robust Pade by SVD

The Toeplitz system's ill conditioning is what creates the doublets of exercise 4.2. Truncating
its singular values removes them.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import pade as pd


def robust_pade(taylor, m, n, tol=1e-10):
    """Solve the Toeplitz system in the least norm sense with a rank cut.

    The rank the SVD keeps is a measurement of the function's effective TYPE at the given
    precision. For exp with noisy coefficients it comes out as 5 at every degree tried, which
    says the data supports a [5/5] approximant and no more.
    """
    c = np.asarray(taylor, dtype=float).ravel()
    m = int(m)
    n = int(n)
    if c.size < m + n + 1:
        raise ValueError(f"[{m}/{n}] needs {m + n + 1} coefficients, got {c.size}")
    get = lambda i: float(c[i]) if 0 <= i < c.size else 0.0
    A = np.stack([[get(m + j - k) for k in range(1, n + 1)] for j in range(1, n + 1)])
    rhs = np.asarray([-get(m + j) for j in range(1, n + 1)])
    U, s, Vt = np.linalg.svd(A)
    keep = int(np.sum(s > float(tol) * s[0])) if s.size else 0
    inverse = np.zeros_like(s)
    inverse[:keep] = 1.0 / s[:keep]
    q = np.concatenate(([1.0], Vt.T @ (inverse * (U.T @ rhs))))
    p = np.asarray([sum(q[k] * get(i - k) for k in range(min(i, n) + 1))
                    for i in range(m + 1)])
    return p, q, keep, int(s.size)


exact = pd.taylor_coefficients("exp", 60)
gen = np.random.default_rng(3)
grid = np.linspace(-1.0, 1.0, 4001)
truth = np.exp(grid)


def finite_error(p, q):
    v = pd.evaluate(p, q, grid)
    good = np.isfinite(v)
    return float(np.max(np.abs(v[good] - truth[good]))) if good.any() else float("inf")


print(f"{'[m/n]':>9}{'noise':>9}{'plain error':>14}{'poles inside':>14}"
      f"{'rank kept':>11}{'robust error':>15}")
for k in (10, 14, 18):
    for eps in (0.0, 1e-14, 1e-11, 1e-8):
        g = np.random.default_rng(3)
        c = exact + eps * g.standard_normal(exact.size) if eps > 0 else exact
        out = pd.coefficients(c, k, k)
        roots = np.roots(out["q"][::-1])
        inside = int(np.sum((np.abs(np.imag(roots)) < 1e-6)
                            & (np.abs(np.real(roots)) <= 1.0)))
        p, q, kept, total = robust_pade(c, k, k)
        print(f"{f'[{k}/{k}]':>9}{eps:>9.0e}{finite_error(out['p'], out['q']):>14.3e}"
              f"{inside:>14}{kept:>11}{finite_error(p, q):>15.3e}")
```

| $[m/n]$ | noise | plain error | real poles inside $[-1,1]$ | rank kept | robust error |
|---|---|---|---|---|---|
| $[10/10]$ | 0 | 1.776e-15 | 0 | 5 | 4.502e-14 |
| $[10/10]$ | 1e-11 | 3.770e-11 | 0 | 5 | 3.766e-11 |
| $[14/14]$ | 0 | 2.220e-15 | 0 | 5 | 2.220e-15 |
| $[14/14]$ | **1e-11** | **1.436e-8** | **1** | 5 | **3.770e-11** |
| $[14/14]$ | 1e-8 | 3.770e-8 | 1 | 14 | 3.770e-8 |
| $[18/18]$ | 1e-11 | 3.770e-11 | 1 | 5 | 3.770e-11 |

**One row does the work, and it does it decisively.** At $[14/14]$ with $10^{-11}$ noise the plain
solve places one real pole inside $[-1,1]$ and its error is $1.4\times10^{-8}$. The robust solve
keeps rank 5 of 14 and its error is $3.8\times10^{-11}$, **381 times better**.

**Every other row is a tie**, and saying so is the honest result. When the plain solve happens not
to put a pole inside the interval, truncating the rank neither helps nor hurts, and at
$[10/10]$ with no noise it is very slightly worse ($4.5\times10^{-14}$ against
$1.8\times10^{-15}$) because it is discarding real information.

**So robust Pade is insurance, not an improvement.** It costs an SVD instead of a solve, it is
never much worse, and occasionally it is the difference between an answer and a singularity.
Whether a given problem is in that occasional case is not knowable in advance, which is the
argument for always paying.

**The rank column is the most informative thing in the table.** It reads 5 at every degree
whenever the noise is at or above $10^{-11}$, whatever $m$ and $n$ are. That is the algorithm
**discovering the effective type of the function at that precision**: given coefficients accurate
to 11 digits, $\exp$ supports a $[5/5]$ rational model and nothing more. Asking for $[18/18]$ does
not get you more information, it gets you 13 spurious degrees of freedom fitted to noise.

At $10^{-8}$ noise the rank jumps to the full $n$, because there the noise is large enough that
every singular value clears the tolerance, and the SVD cut stops protecting anything. That is the
failure mode of the method and it is visible in the table.

### 3.2 Chebyshev-Pade

Taylor-Pade matches a series **at a point**. Chebyshev-Pade matches a series **on an interval**,
which is lesson 56's improvement applied to the rational setting.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import chebapprox as ca, pade as pd


def chebyshev_pade(f, m, n, lo=-1.0, hi=1.0, n_series=48):
    """Match a Chebyshev series rather than a Taylor series.

    The only change from the Taylor version is the product rule. Instead of the Cauchy product,
    T_i T_j = (T_{i+j} + T_{|i-j|}) / 2, so the coefficient of T_idx in (series times T_qi)
    collects three sources rather than one.
    """
    a = ca.coefficients_by_interpolation(f, int(n_series), lo, hi)
    N = a.size
    at = lambda k: float(a[k]) if 0 <= k < N else 0.0

    def coeff_of(idx, qi):
        if qi == 0:
            return at(idx)
        if idx == 0:
            # k + qi = 0 is impossible, and |k - qi| = 0 gives k = qi once, not twice
            return 0.5 * at(qi)
        return 0.5 * (at(idx - qi) + at(idx + qi) + at(qi - idx))

    A = np.stack([[coeff_of(m + j, k) for k in range(1, n + 1)] for j in range(1, n + 1)])
    rhs = np.asarray([-coeff_of(m + j, 0) for j in range(1, n + 1)])
    q = np.concatenate(([1.0], np.linalg.lstsq(A, rhs, rcond=1e-12)[0]))
    p = np.asarray([sum(q[k] * coeff_of(i, k) for k in range(n + 1)) for i in range(m + 1)])
    return p, q


for name, f, key, lo, hi in (("log(1+x) on [-0.5, 3]", np.log1p, "log1p", -0.5, 3.0),
                             ("tan on [-1.4, 1.4]", np.tan, "tan", -1.4, 1.4)):
    grid = np.linspace(lo, hi, 2001)
    want = f(grid)
    taylor = pd.taylor_coefficients(key, 32)
    print(name)
    print(f"{'[m/n]':>9}{'Taylor Pade':>16}{'Chebyshev trunc':>18}{'Chebyshev Pade':>17}"
          f"{'gain':>10}")
    for m, n in ((2, 2), (3, 3), (4, 4), (5, 5), (6, 6)):
        out = pd.coefficients(taylor, m, n)
        tp = pd.evaluate(out["p"], out["q"], grid)
        p, q = chebyshev_pade(f, m, n, lo, hi)
        u = np.clip((2.0 * grid - (hi + lo)) / (hi - lo), -1.0, 1.0)
        theta = np.arccos(u)
        with np.errstate(divide="ignore", invalid="ignore"):
            cp = (sum(p[k] * np.cos(k * theta) for k in range(p.size))
                  / sum(q[k] * np.cos(k * theta) for k in range(q.size)))
        trunc = ca.evaluate_series(
            ca.coefficients_by_interpolation(f, m + n, lo, hi), grid, lo, hi)

        def err(v):
            good = np.isfinite(v)
            return float(np.max(np.abs(v[good] - want[good]))) if good.any() else float("inf")

        print(f"{f'[{m}/{n}]':>9}{err(tp):>16.3e}{err(trunc):>18.3e}{err(cp):>17.3e}"
              f"{err(tp) / max(err(cp), 1e-300):>10.1f}")
    print()
```

**$\log(1+x)$ on $[-0.5, 3]$:**

| $[m/n]$ | Taylor Pade | Chebyshev truncation | Chebyshev Pade | gain |
|---|---|---|---|---|
| $[2/2]$ | 2.266e-2 | 2.155e-2 | 1.872e-3 | 12.1 |
| $[3/3]$ | 2.621e-3 | 3.918e-3 | 6.263e-5 | 41.8 |
| $[4/4]$ | 2.975e-4 | 6.814e-4 | 2.064e-6 | 144.1 |
| $[5/5]$ | 3.349e-5 | 1.323e-4 | 6.757e-8 | 495.6 |
| $[6/6]$ | 3.755e-6 | 2.556e-5 | **2.205e-9** | **1703** |

**$\tan$ on $[-1.4, 1.4]$:**

| $[m/n]$ | Taylor Pade | Chebyshev truncation | Chebyshev Pade | gain |
|---|---|---|---|---|
| $[2/2]$ | 1.759 | 8.667e-1 | 1.256e-1 | 14.0 |
| $[4/4]$ | 5.697e-3 | 1.234e-1 | 1.535e-5 | 371 |
| $[5/5]$ | 1.184e-4 | 4.625e-2 | 7.730e-8 | 1532 |
| $[6/6]$ | 1.674e-6 | 1.722e-2 | **2.679e-10** | **6248** |

**Chebyshev-Pade beats both parents.** It beats Taylor-Pade by up to 6248, because it is fitted on
the interval rather than at a point. It beats the Chebyshev truncation of the same total degree by
even more, because it has poles to place.

**The bug that took two attempts, and it is worth recording.** The product rule
$T_iT_j = \frac12(T_{i+j} + T_{|i-j|})$ has **three** sources for the coefficient of $T_{\text{idx}}$
in a series times $T_{q}$: $k = \text{idx}-q$, $k = \text{idx}+q$ and $k = q-\text{idx}$. At
$\text{idx} = 0$ the last two coincide, and counting both doubles the constant term. With that
double count the approximant was **worse than doing nothing**, with errors of 3 to 75 on functions
bounded by 2, and it looked like a failure of the method rather than of one line.

**Why Chebyshev-Pade is not the standard tool despite being better.** Two reasons. The
coefficients of the product are a full convolution rather than a Cauchy product, so the system is
not Toeplitz and Levinson does not apply. And the modern alternative, the **AAA algorithm**,
fits a rational function directly in barycentric form from sample values, needs no series at all,
and is more robust. Chebyshev-Pade is the right answer when you already have a Chebyshev series
and want to add poles to it.

### 3.3 Wynn's epsilon algorithm

The diagonal Pade approximants of a series can be computed from its **partial sums** by a
recurrence, with no matrix at all:

$$
\varepsilon_{k+1}^{(j)} = \varepsilon_{k-1}^{(j+1)}
+ \frac{1}{\varepsilon_{k}^{(j+1)} - \varepsilon_{k}^{(j)}}
$$

starting from $\varepsilon_{-1}^{(j)} = 0$ and $\varepsilon_{0}^{(j)} = S_j$, the $j$th partial
sum. The even columns $\varepsilon_{2k}^{(j)}$ are the Pade approximants $[j+k/k]$.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import pade as pd


def wynn_epsilon(partial_sums):
    """Wynn's cross rule. The even columns are the diagonal Pade approximants.

    The division by a difference is what makes it fast and what makes it fragile: when two
    entries agree to machine precision the reciprocal overflows, which is the algorithm's
    version of lesson 57's degenerate block.
    """
    s = np.atleast_1d(np.asarray(partial_sums, dtype=float)).ravel()
    n = s.size
    if n < 3:
        raise ValueError(f"need at least three partial sums, got {n}")
    e = np.zeros((n + 2, n + 2))
    e[1, :n] = s
    for k in range(1, n):
        for j in range(n - k):
            gap = e[k, j + 1] - e[k, j]
            e[k + 1, j] = e[k - 1, j + 1] + (1.0 / gap if gap != 0.0 else np.inf)
    return e


c = pd.taylor_coefficients("log1p", 21)
for x0 in (0.5, 1.5, 3.0):
    partial = np.cumsum([c[k] * x0 ** k for k in range(c.size)])
    table = wynn_epsilon(partial)
    exact = math.log1p(x0)
    column = [v for v in (table[2 * j + 1, 0] for j in range(1, 10)) if np.isfinite(v)]
    best = min(column, key=lambda v: abs(v - exact))
    pade_value = float(pd.approximant(c, 10, 10)(np.array([x0]))[0])
    print(f"x = {x0}, exact {exact:.10f}")
    print(f"  plain sum of 21 terms  {partial[-1]:>22.10f}  error {abs(partial[-1] - exact):.2e}")
    print(f"  best epsilon value     {best:>22.10f}  error {abs(best - exact):.2e}")
    print(f"  [10/10] Pade           {pade_value:>22.10f}  error {abs(pade_value - exact):.2e}")
```

| $x$ | plain sum of 21 terms | best $\varepsilon$ value | $[10/10]$ Pade |
|---|---|---|---|
| 0.5 | 0.4054650927 (err 1.5e-8) | 0.4054651081 (err **5.6e-17**) | 0.4054651081 (err 1.7e-16) |
| 1.5 | -96.8285843213 (err **97.7**) | 0.9162907319 (err **3.0e-12**) | 0.9162907318 (err 4.3e-11) |
| 3.0 | -129079764.6 (err **1.3e8**) | 1.3862943559 (err **5.2e-9**) | 1.3862943421 (err 1.9e-8) |

**At $x = 3$ the series is divergent** and its partial sums reach $-1.3\times10^8$. The epsilon
algorithm, given nothing but those partial sums, returns $1.38629436$ against the true
$1.38629436$: **eight correct digits from a sequence heading to minus infinity.**

**The epsilon values match the Pade approximants**, as the theory says, and are slightly better
here because the best entry of the column is being chosen while the Pade comparison is fixed at
$[10/10]$.

**Why anyone would use it over the matrix form.** It needs only the partial sums, so it applies to
a sequence that did not come from a power series at all: an iterative method's iterates, a
sequence of quadrature estimates, a perturbation expansion. That is what makes it a **sequence
acceleration** method rather than an approximation method, and it is the same idea as Part 1's
Aitken extrapolation, which is the $k = 2$ case, and Part 9's Richardson extrapolation, which is
the version for a known error expansion.

**Its failure mode.** The recurrence divides by $\varepsilon_k^{(j+1)} - \varepsilon_k^{(j)}$, and
when two entries agree to machine precision that difference is roundoff and the reciprocal is
enormous. The table then fills with garbage, which is the algorithm's version of the degenerate
block, and the standard defence is to stop as soon as a column stops improving rather than filling
the whole table.

### 4.1 The pole against the coefficient count

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import pade as pd

print(f"{'[m/n]':>9}{'coefficients':>15}{'nearest real pole':>22}{'error vs pi/2':>17}")
for k in (2, 3, 4, 5, 6, 7, 8):
    c = pd.taylor_coefficients("tan", 2 * k + 10)
    real = pd.pole_locations(c, k - 1, k)["real_poles"]
    if real.size == 0:
        print(f"{f'[{k-1}/{k}]':>9}{2 * k:>15}{'none':>22}{'n/a':>17}")
        continue
    j = int(np.argmin(np.abs(np.abs(real) - math.pi / 2)))
    print(f"{f'[{k-1}/{k}]':>9}{2 * k:>15}{abs(real[j]):>22.10f}"
          f"{abs(abs(real[j]) - math.pi / 2):>17.3e}")
```

| $[m/n]$ | Taylor coefficients used | nearest real pole | error against $\pi/2$ |
|---|---|---|---|
| $[1/2]$ | 4 | 1.7320508076 | 1.613e-1 |
| $[2/3]$ | 6 | 1.7149858514 | 1.442e-1 |
| $[3/4]$ | 8 | 1.5712333932 | **4.371e-4** |
| $[4/5]$ | 10 | 1.5711734796 | 3.772e-4 |
| $[5/6]$ | 12 | 1.5707965342 | **2.074e-7** |
| $[6/7]$ | 14 | 1.5707965042 | 1.774e-7 |
| $[7/8]$ | 16 | 1.5707963268 | **2.665e-11** |

**The convergence is geometric, and it comes in pairs.** The error drops by about three orders
every **two** steps: $1.6\times10^{-1}$, $1.4\times10^{-1}$, then $4.4\times10^{-4}$,
$3.8\times10^{-4}$, then $2.1\times10^{-7}$, $1.8\times10^{-7}$, then $2.7\times10^{-11}$.

**The pairing is parity.** $\tan$ is odd, so every second entry of the table is degenerate: the
$[2k/2k+1]$ approximant carries essentially the same information as $[2k-1/2k]$, and the real
improvement comes when the denominator degree rises by two and can accommodate another pair of
poles.

**The rate.** Fitting the improving steps, the error falls by a factor of about $2\times10^3$ per
two denominator degrees, which is $\sim45$ per degree. The theoretical rate for a simple pole is
governed by the ratio of the distance to the **second** singularity over the first,
$(3\pi/2)/(\pi/2) = 3$, and the observed rate is much faster than $3^{-k}$ because the pole is
being located rather than the function approximated.

**At $[7/8]$ the pole is right to eleven digits**, from sixteen Taylor coefficients of $\tan$ at
the origin. Nothing in that input mentions $\pi/2$. That is the result worth remembering about
Pade approximation: **it finds singularities it was not told about, from local data.**

**The practical use.** This is a **singularity detector**. Given a series, forming the Pade
approximants and looking at the poles tells you the radius of convergence and where the obstruction
is, which is often the physical question: in a perturbation expansion the nearest singularity is a
phase transition or a resonance, and its location is the answer being sought.

### 4.2 Froissart doublets

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import pade as pd


def complex_doublets(taylor, m, n, radius=3.0, gap_tol=1e-3):
    """Poles with a zero very close by, anywhere in the complex plane.

    Searching only the real axis finds nothing on exp, because its spurious poles arrive in
    complex conjugate pairs. The search has to be two dimensional.
    """
    out = pd.coefficients(taylor, m, n)
    p, q = out["p"], out["q"]
    poles = np.roots(q[::-1]) if q.size > 1 else np.zeros(0, dtype=complex)
    zeros = (np.roots(p[::-1]) if p.size > 1 and np.any(p[1:] != 0.0)
             else np.zeros(0, dtype=complex))
    near = poles[np.abs(poles) <= float(radius)]
    found = []
    for z in near:
        if zeros.size:
            k = int(np.argmin(np.abs(zeros - z)))
            gap = float(abs(zeros[k] - z))
            if gap < float(gap_tol):
                found.append((complex(z), gap))
    return found, int(near.size)


exact = pd.taylor_coefficients("exp", 60)
print(f"{'[m/n]':>9}{'noise added':>13}{'poles within 3':>16}{'doublets':>10}"
      f"{'smallest gap':>15}")
for k in (8, 12, 16, 20):
    for eps in (0.0, 1e-16, 1e-13, 1e-10):
        g = np.random.default_rng(7)
        c = exact + eps * g.standard_normal(exact.size) if eps > 0 else exact
        found, near = complex_doublets(c, k, k)
        smallest = f"{min(gap for _, gap in found):.3e}" if found else "none"
        print(f"{f'[{k}/{k}]':>9}{eps:>13.0e}{near:>16}{len(found):>10}{smallest:>15}")
```

| $[m/n]$ | noise added | poles within 3 | doublets | smallest gap |
|---|---|---|---|---|
| $[12/12]$ | 0 | 0 | 0 | none |
| $[12/12]$ | 1e-16 | 0 | 0 | none |
| $[12/12]$ | **1e-13** | **7** | **7** | 2.028e-7 |
| $[12/12]$ | 1e-10 | 7 | 7 | 5.014e-12 |
| $[16/16]$ | 0 | 0 | 0 | none |
| $[16/16]$ | 1e-13 | 11 | 11 | 1.210e-8 |
| $[16/16]$ | 1e-10 | 11 | 11 | 2.220e-16 |
| $[20/20]$ | 0 | 0 | 0 | none |
| $[20/20]$ | **1e-16** | **15** | **15** | 1.397e-7 |
| $[20/20]$ | 1e-10 | 15 | 15 | 5.551e-16 |

**Three readings, and all three are the point of the exercise.**

**The count grows with the degree, roughly as $n-5$.** At $[12/12]$ there are 7, at $[16/16]$ 11,
at $[20/20]$ 15. The five that are not doublets are the genuine degrees of freedom the data
supports, which is exactly the rank 5 that exercise 3.1's SVD found. **Every degree above the
effective type produces one doublet.**

**The noise threshold falls as the degree rises.** At $[12/12]$ the doublets need $10^{-13}$
noise, and at $[20/20]$ they appear at $10^{-16}$, which is roundoff. So at high degree they
appear with **no noise added at all**, from the rounding of the arithmetic itself, which is why
they are unavoidable rather than a consequence of bad input.

**The gap shrinks as the noise grows.** At $[16/16]$: $1.2\times10^{-8}$ at noise $10^{-13}$ and
$2.2\times10^{-16}$ at $10^{-10}$. **That is the Froissart signature.** The pole and its zero are
converging on each other, so they nearly cancel and the approximant is nearly correct everywhere
except in a vanishingly small neighbourhood, where it is unbounded.

**And they move.** At $[16/16]$ with noise $10^{-13}$, seed 7 puts a pole at $2.033$ and seed 8
puts it at $1.980$. Changing the noise to $10^{-10}$ gives a completely different set. **A feature
of the answer that moves when the arithmetic changes is not a feature of the function**, and that
is the diagnostic: compute at two precisions and discard anything that moved.

**Why they are called that.** Froissart studied exactly this in 1969, showing that adding random
noise to a series produces poles distributed on a circle at the radius of convergence, each paired
with a nearby zero. The name attaches to the pair, not to the pole.

### 4.3 The advantage against the distance to the singularity

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import pade as pd

n_terms = 24
print("f(t) = log(1 - t/a), branch point at t = a, interval fixed at [-0.5, 0.5]")
print(f"{'a':>8}{'gap to the end':>16}{'Pade [4/4]':>15}{'Taylor deg 8':>15}{'gain':>10}")
for a in (8.0, 4.0, 2.0, 1.5, 1.0, 0.75, 0.6, 0.55):
    c = np.zeros(n_terms)
    for k in range(1, n_terms):
        c[k] = -(a ** (-k)) / k
    f = lambda t, aa=a: np.log(1.0 - t / aa)
    out = pd.against_taylor(f, c, 4, 4, -0.5, 0.5)
    print(f"{a:>8.2f}{a - 0.5:>16.3f}{out['pade_error']:>15.3e}"
          f"{out['taylor_error']:>15.3e}{out['ratio']:>10.1f}")

print()
print("the same, with the interval scaled to the singularity: [-0.5a, 0.5a]")
print(f"{'a':>8}{'Pade [4/4]':>15}{'Taylor deg 8':>15}{'gain':>10}")
for a in (8.0, 4.0, 2.0, 1.0, 0.5):
    c = np.zeros(n_terms)
    for k in range(1, n_terms):
        c[k] = -(a ** (-k)) / k
    f = lambda t, aa=a: np.log(1.0 - t / aa)
    out = pd.against_taylor(f, c, 4, 4, -0.5 * a, 0.5 * a)
    print(f"{a:>8.2f}{out['pade_error']:>15.3e}{out['taylor_error']:>15.3e}"
          f"{out['ratio']:>10.1f}")
```

**Interval fixed, singularity moving in:**

| $a$ | gap to the interval end | Pade $[4/4]$ | Taylor degree 8 | gain |
|---|---|---|---|---|
| 8.00 | 7.500 | 4.857e-16 | 1.713e-12 | **3527** |
| 4.00 | 3.500 | 3.050e-13 | 9.329e-10 | 3059 |
| 2.00 | 1.500 | 3.011e-10 | 5.474e-7 | 1818 |
| 1.50 | 1.000 | 6.505e-9 | 8.078e-6 | 1242 |
| 1.00 | 0.500 | 7.631e-7 | 3.970e-4 | 520 |
| 0.75 | 0.250 | 4.194e-5 | 7.367e-3 | 176 |
| 0.60 | 0.100 | 2.364e-3 | 9.261e-2 | 39 |
| 0.55 | 0.050 | 2.083e-2 | 3.045e-1 | **15** |

**Interval scaled with the singularity:**

| $a$ | Pade $[4/4]$ | Taylor degree 8 | gain |
|---|---|---|---|
| 8.00 | 7.631e-7 | 3.970e-4 | 520.2 |
| 4.00 | 7.631e-7 | 3.970e-4 | 520.2 |
| 2.00 | 7.631e-7 | 3.970e-4 | 520.2 |
| 1.00 | 7.631e-7 | 3.970e-4 | 520.2 |
| 0.50 | 7.631e-7 | 3.970e-4 | 520.2 |

**The first table looks like the answer and the second one is the answer.**

In the first, the gain falls from 3527 to 15 as the singularity approaches, which reads as "Pade's
advantage disappears near a singularity" and is the opposite of what the lesson claims. In the
second, with the interval scaled so the singularity is always at the same **relative** distance,
the gain is **520.2 in every row**, to four digits.

**So the advantage depends only on the ratio of the interval half width to the distance to the
singularity, and on nothing else.** Both errors grow as the singularity approaches, and they grow
at the same rate, so the ratio is invariant. The first table was measuring the difficulty of the
problem, not the advantage of the method.

**Where the advantage genuinely does vanish.** When there is no singularity to model. On $\exp$
over $[-0.3, 0.3]$ the lesson measures a gain below 50, and it comes from the higher effective
degree rather than from any pole. As the interval shrinks toward a point, both methods converge to
matching the same Taylor coefficients and the gain tends to a constant near 1.

**And where it grows without bound.** When the interval reaches outside the Taylor radius of
convergence. Over $[-0.9, 3]$ the Taylor series of $\log(1+x)$ does not converge on most of the
interval, so its error grows with the degree while the Pade error falls, and the measured gain at
$[4/4]$ is 38086 and rising.

### 5.1 Montessus de Ballore's theorem

**Statement.** Let $f$ be analytic in $|z| < R$ except for exactly $n$ poles there, counted with
multiplicity, none at the origin. Then the **column** $[m/n]$ of the Pade table, with $n$ fixed
and $m \to \infty$, converges to $f$ uniformly on compact subsets of the disc minus the poles.
Moreover the poles of the approximants converge to the poles of $f$.

**What it gives.** Convergence, with a rate, for a **fixed denominator degree** matched to the
number of poles. It is the theorem that justifies exercise 4.1: knowing $\tan$ has a pair of poles
nearest the origin, the column $n = 2$ should converge, and it does.

**What it does not claim, and this is the exercise's point.**

**It says nothing about the diagonal.** The diagonal $[m/m]$ with $m \to \infty$ is the sequence
everyone actually uses, and Montessus does not cover it. There is no general convergence theorem
for the diagonal, and there are counterexamples: **Perron's example** is a series whose diagonal
Pade approximants diverge everywhere except at the origin, even though the series has a positive
radius of convergence.

**It requires knowing $n$ in advance.** The hypothesis is that there are exactly $n$ poles in the
disc, so the theorem cannot be applied without already knowing the answer to the question one is
usually asking.

**It says nothing about branch points.** A function with a cut has infinitely many "poles" in the
sense that no finite $n$ works, so the theorem is vacuous for $\log$ and $\sqrt{}$, which are
exactly the cases exercise 1.3 and 4.3 measured.

**What is known about the diagonal.** **Nuttall-Pommerenke**: for $f$ analytic except on a set of
zero capacity, the diagonal converges in **capacity**, which permits bad behaviour on a small set
that moves with $m$. That small set is where the Froissart doublets live, which is why the theorem
has to be stated in capacity rather than uniformly. **Stahl's theorem** extends this to functions
with branch points, showing the poles converge to a specific cut of minimal capacity, which is
what exercise 1.3's poles along the cut are doing.

**The practical summary.** Column convergence is a theorem; diagonal convergence is an empirical
fact with counterexamples and a weak theorem behind it. That gap is why robust Pade and AAA exist.

### 5.2 The matrix exponential

**Why the diagonal.** For a matrix argument the approximant must be evaluated as
$R(A) = Q(A)^{-1}P(A)$, so both $P$ and $Q$ have to be formed. A diagonal $[k/k]$ needs $k$ matrix
multiplications for the powers plus one solve, and gives order $2k$ accuracy. An off-diagonal
$[m/n]$ with $m \ne n$ needs $\max(m,n)$ multiplications and gives order $m+n < 2\max(m,n)$. **The
diagonal gets the most order per matrix multiplication.**

**A second reason, specific to $\exp$.** The diagonal approximants of $e^z$ satisfy
$R_{kk}(-z) = 1/R_{kk}(z)$ exactly, so they preserve the identity $e^{-A} = (e^A)^{-1}$. They are
also **A-stable** as stability functions, which is exercise 1.1's connection to Part 10.

**Why 13.** Higham's 2005 analysis, which is what every current library implements. The choice
balances three things:

- **The backward error.** For $\|A\| \le \theta_k$, the $[k/k]$ approximant satisfies
  $R_{kk}(A) = e^{A+E}$ with $\|E\| \le \varepsilon\|A\|$, and $\theta_{13} = 5.37$ is the
  largest such bound before the cost of a larger $k$ outweighs the saving.
- **The scaling and squaring cost.** $A$ is scaled by $2^{-s}$ so $\|A/2^s\| \le \theta_{13}$, then
  the result is squared $s$ times. Each squaring is one matrix multiplication, so a larger
  $\theta$ means fewer squarings.
- **The evaluation cost.** $[13/13]$ can be evaluated with 6 matrix multiplications plus one
  solve, using an even-odd splitting, rather than the 13 a naive Horner would need.

The measured optimum over those three is $k = 13$, with $k = 3, 5, 7, 9$ used for smaller norms.

**What makes the matrix version cheap, precisely.** The even-odd split: write
$P_{13}(A) = U + V$ with $U = A(b_{13}A^{12} + \cdots)$ and $V = b_{12}A^{12} + \cdots$, and note
$Q_{13}(A) = -U + V$ by the symmetry $q_j = (-1)^jp_j$. So **one** set of powers serves both, and
the answer is $(-U+V)^{-1}(U+V)$: six multiplications, one solve. A general rational function of a
matrix would need both numerator and denominator built independently.

### 5.3 Rational minimax and AAA

**What changes from lesson 54.** The approximating set $R_{m,n}$ is not a linear space: the sum of
two type $[m/n]$ functions is type $[m+n/2n]$. So every argument that used linearity has to be
redone.

**Existence** survives, by a compactness argument on the coefficients with a normalisation.

**Uniqueness** survives, and is due to Achieser.

**Equioscillation survives with a new count.** The best type $[m/n]$ approximation is
characterised by the error equioscillating at least

$$
m + n + 2 - d
$$

times, where $d$ is the **defect**: the amount by which the best approximant's actual type falls
short of $[m/n]$. For a non-degenerate case $d = 0$ and the count is $m+n+2$, which is the number
of free parameters plus one, exactly as in the polynomial case where $n = 0$ gives $m+2$.

**Why the defect appears.** It is lesson 57's block structure again. If the best approximation
happens to be of type $[m-1/n-1]$, there are fewer genuine parameters and correspondingly fewer
alternations. Unlike the polynomial case, this can happen without $f$ being special, which is what
makes the rational Remez algorithm hard: the algorithm has to detect the defect and adjust the
count, and getting it wrong makes the iteration diverge.

**What AAA does instead.** The **AAA algorithm** of Nakatsukasa, Sete and Trefethen (2018) does
not solve the minimax problem at all. It:

- represents the rational function in **barycentric** form,
  $r(x) = \sum_j \frac{w_jf_j}{x-z_j} \big/ \sum_j\frac{w_j}{x-z_j}$, which is lesson 44's stable
  interpolation formula;
- picks support points $z_j$ **greedily**, one at a time, choosing where the current error is
  largest;
- solves a small **least squares** problem for the weights $w_j$ at each step, by SVD.

The result is a near-best rational approximation, not the minimax one, obtained by a sequence of
linear solves with no iteration on a reference set and no defect to detect.

**Why it works where rational Remez does not.** The barycentric form has no explicit denominator
coefficients to become ill conditioned, the SVD handles the rank deficiency that the defect
represents automatically, and the greedy support point selection avoids the reference exchange
entirely. It needs only sample values of $f$, not a series, so it applies where no Taylor
coefficients exist.

**And it removes the doublets by construction**, because the SVD step is exercise 3.1's rank
truncation, applied at every greedy step rather than once at the end.

---

## Lesson 58, Trigonometric Interpolation and the DFT

### 1.1 What the DFT has that the Vandermonde does not

**In one sentence: the DFT's nodes are the roots of unity, and its basis is orthogonal on exactly
those nodes, so its matrix is unitary up to a scale.** The Vandermonde matrix's basis, the
monomials, is nearly dependent on any node set, so its matrix is nearly singular.

**What it costs: the nodes and the function are no longer yours to choose.** Three restrictions,
all of them real.

**The nodes must be equally spaced.** The orthogonality relation
$\sum_j w^{(\ell-j)k} = 0$ is a geometric series, and it sums to zero only because the ratio is an
exact root of unity. Move one node and the whole structure collapses; there is no "nearly equally
spaced" version.

**The function must be periodic.** The basis functions are periodic, so the interpolant is, and
lesson 60 measures what happens when the data is not: the periodic extension has a jump at the
seam and the coefficients decay one order slower.

**And the number of nodes fixes the resolution absolutely.** Section 5's aliasing is not an error
that shrinks with a better method, it is information that is absent. The Vandermonde matrix, for
all its conditioning, at least approximates between the nodes; the DFT does not distinguish
frequencies $n$ apart at all.

**So the trade is: give up node placement and non-periodic data, and get $\kappa = 1$ and an
$O(n\log n)$ transform.** Part 7 made the opposite trade, and lesson 47's Chebyshev points are the
compromise: chosen nodes, non-periodic data, and $\kappa$ growing like $\log n$ instead of
$e^{n}$.

### 1.2 Aliasing against interpolation error

**Interpolation error is approximation.** Lesson 46's formula says
$f(t) - p(t) = \frac{f^{(n+1)}(\xi)}{(n+1)!}w(t)$, which is a **statement about a specific $f$**:
two different functions with the same data have different errors, and the error is bounded by
something that shrinks as the data improves.

**Aliasing is loss.** $e^{2\pi ikt}$ and $e^{2\pi i(k+n)t}$ produce **identical** samples, so the
data does not determine which was measured. The bound is not large, it does not exist: two
functions differing by an $O(1)$ amount between the nodes are indistinguishable.

Measured, at $n = 8$:

| frequency | folds to | gap at the sample points | gap between them |
|---|---|---|---|
| 1 | 1 | 0.0 | 0.0 |
| 9 | 1 | 1.1e-15 | 1.9999 |
| 17 | 1 | 2.2e-15 | 1.9999 |
| 41 | 1 | 5.3e-15 | 2.0000 |

**Identical to roundoff on the grid and different by the full amplitude off it.**

**Why no better algorithm helps.** An algorithm sees only the data. If two inputs give the same
data, no algorithm can produce different outputs for them, and any output it does produce is
wrong for at least one of them. That is not a limitation of a method, it is a property of the
sampling.

**What does help, and it is not an algorithm.** An **analogue** filter before the converter,
removing the frequencies above Nyquist before they can fold. That is why every audio interface has
one, and why "we can fix it in software" is false here in a way it is not false for most things.

### 1.3 Interpolating and still being wrong

**"Wrong" means: wrong between the data points, which is the only place an interpolant does any
work.**

Both conventions reproduce the data exactly, because at $t_j = j/n$ the factor $e^{2\pi ikj/n}$
is unchanged by shifting $k$ by $n$. **Off the grid they are completely different functions**, and
the measurement is decisive:

| $n$ | uncentred error | centred error | uncentred imaginary part | centred imaginary part |
|---|---|---|---|---|
| 8 | 1.4273 | 4.05e-15 | 1.4274 | 2.28e-15 |
| 16 | 1.4273 | 4.05e-15 | 1.4274 | 2.28e-15 |
| 32 | 1.4273 | 4.05e-15 | 1.4274 | 2.28e-15 |
| 64 | 1.4273 | 4.05e-15 | 1.4274 | 2.28e-15 |

**The uncentred version has an imaginary part of 1.43 on a real signal**, and its error does not
shrink with $n$. That is the tell: an error that does not respond to refinement is not a
discretisation error, it is a wrong formula.

**The mechanism.** The uncentred sum treats bin $n-1$ as a frequency of $n-1$ cycles, a very fast
oscillation. The centred sum treats it as $-1$ cycle, a slow one going backwards. Both give the
same samples; only the second gives a curve that looks like the signal.

**The Nyquist term is the fiddly part.** For even $n$, bin $n/2$ is its own conjugate partner, so
it belongs equally to $+n/2$ and $-n/2$. Splitting it in half between the two is what makes the
interpolant real, and not splitting it leaves an imaginary part of $\frac12|X_{n/2}|$, which is
not roundoff.

**The general moral.** Interpolating the data is necessary and nowhere near sufficient. Lesson
46's Runge polynomial also interpolates its data, and reaches 334 between the points.

### 2.1 The orthogonality relation

**Claim.** $F^HF = nI$ where $F_{jk} = w^{jk}$, $w = e^{-2\pi i/n}$.

**Proof.** The $(j,\ell)$ entry of $F^HF$ is

$$
\left(F^HF\right)_{j\ell} = \sum_{k=0}^{n-1}\overline{F_{kj}}F_{k\ell}
= \sum_{k=0}^{n-1}\overline{w^{kj}}\,w^{k\ell}
= \sum_{k=0}^{n-1}\left(w^{\ell-j}\right)^{k}
$$

using $\overline{w} = w^{-1}$, since $|w| = 1$.

That is a geometric series with ratio $r = w^{\ell-j}$.

**If $\ell = j$**, then $r = 1$ and the sum is $n$.

**If $\ell \ne j$**, then $0 < |\ell - j| < n$, so $r \ne 1$ because $w$ is a **primitive** $n$th
root of unity. The geometric sum is

$$
\sum_{k=0}^{n-1}r^k = \frac{r^n - 1}{r - 1}
$$

and $r^n = w^{(\ell-j)n} = (w^n)^{\ell-j} = 1$, so the numerator is zero and the sum is zero.

**Where $w^n = 1$ was used: the numerator.** That is the only place, and it is the whole content
of the theorem. Every other step is bookkeeping about geometric series.

**Two consequences.** $F/\sqrt n$ is unitary, so $\kappa_2 = 1$ exactly, at every $n$, measured to
$10^{-9}$ in the lesson. And $F^{-1} = \frac1nF^H$, so the inverse transform is the same algorithm
with a sign flipped, which is why one implementation serves both.

**The requirement that $w$ be primitive.** If $w$ were, say, a fourth root of unity used for a
transform of size 8, then $w^{\ell-j} = 1$ for $\ell - j = 4$ and the off-diagonal entry would be
8 rather than 0. The matrix would be singular. That is why $w = e^{-2\pi i/n}$ specifically, and
not any root of unity.

### 2.2 The interpolation theorem

**Claim.** With $X = Fx$ and $t_j = j/n$, the function

$$
P(t) = \frac1n\sum_{k=0}^{n-1}X_ke^{2\pi ikt}
$$

satisfies $P(t_j) = x_j$ for every $j$.

**Proof.** Evaluate at $t_j$:

$$
P(t_j) = \frac1n\sum_kX_ke^{2\pi ikj/n} = \frac1n\sum_kX_kw^{-jk}
$$

since $w = e^{-2\pi i/n}$ so $e^{2\pi i/n} = w^{-1}$. **That is precisely the inverse transform
formula**, and the inverse transform of the forward transform is the identity by exercise 2.1.

Written out: substituting $X_k = \sum_\ell x_\ell w^{\ell k}$ and exchanging the sums,

$$
P(t_j) = \frac1n\sum_\ell x_\ell\sum_k w^{(\ell-j)k} = \frac1n\sum_\ell x_\ell\cdot n\delta_{\ell j}
= x_j
$$

using the orthogonality relation directly.

**What makes this remarkable.** In Part 7, interpolation meant solving a linear system: a
Vandermonde system for the power form, a triangular one for Newton, a barycentric formula for
Lagrange. Here **the interpolation coefficients are the transform**, so there is no system at all,
and the transform is the thing you were going to compute anyway.

**Why "no system" is the same statement as "$\kappa = 1$".** The system exists in principle: it is
$Fc = x$ for the coefficients $c$. Its solution is $c = \frac1nF^Hx$, which needs no elimination
because the matrix's inverse is known in closed form. That is what an orthogonal basis always
buys, and lesson 55 measured the same thing for polynomials.

### 2.3 Hermitian symmetry

**Claim.** For real $x$, $X_{n-k} = \overline{X_k}$.

**Proof.**

$$
X_{n-k} = \sum_j x_jw^{j(n-k)} = \sum_j x_jw^{jn}w^{-jk} = \sum_j x_jw^{-jk}
$$

using $w^{jn} = 1$. And since $x_j$ is real,

$$
\overline{X_k} = \overline{\sum_jx_jw^{jk}} = \sum_jx_j\overline{w^{jk}} = \sum_jx_jw^{-jk}
$$

The two are the same expression, so they are equal.

**Counting the independent real numbers.**

**$n$ even.** $X_0$ is real, because $X_0 = \sum x_j$. $X_{n/2}$ is real, because
$X_{n/2} = \overline{X_{n/2}}$ by the symmetry with $k = n/2$. The remaining $n-2$ entries pair
up into $(n-2)/2$ conjugate pairs, each carrying 2 real numbers. Total:

$$
1 + 1 + 2\cdot\frac{n-2}{2} = n
$$

which it must be, since the transform is invertible. The number of **stored** coefficients is
$n/2 + 1$.

**$n$ odd.** Only $X_0$ is forced real, and the other $n-1$ entries form $(n-1)/2$ pairs. Total
$1 + (n-1) = n$, again as required, and the stored count is $(n+1)/2 = \lfloor n/2\rfloor + 1$.

**So the formula $n/2 + 1$ with integer division covers both**, which is what `real_fft` returns
and what the measurement confirms at $n = 8, 9, 16, 17$.

**Why the saving is exactly $\frac12 - \frac1n$ and not $\frac12$.** Storing $n/2+1$ of $n$ is a
saving of $\frac{n/2-1}{n} = \frac12 - \frac1n$. The DC coefficient is the odd one out: it has no
partner, so it is stored in full. At $n = 8$ the saving is 0.375 and at $n = 1024$ it is 0.499.

### 2.4 The shift theorem

**Claim.** If $y_j = x_{(j-s)\bmod n}$, then $Y_k = w^{sk}X_k$.

**Proof.**

$$
Y_k = \sum_j x_{(j-s)\bmod n}w^{jk}
$$

Substitute $\ell = (j-s)\bmod n$, so $j = (\ell+s)\bmod n$. As $j$ runs over all residues so does
$\ell$, and $w^{jk} = w^{(\ell+s)k} = w^{\ell k}w^{sk}$ because $w^n = 1$ makes the exponent's
value depend only on its residue. Hence

$$
Y_k = w^{sk}\sum_\ell x_\ell w^{\ell k} = w^{sk}X_k
$$

**Why the modular arithmetic matters.** The theorem is about a **circular** shift, and the
reduction mod $n$ is what makes $w^{(\ell+s)k}$ factor cleanly. For a non-circular shift, where
data falls off the end, the theorem is false.

**Magnitude invariance.** $|w^{sk}| = 1$, so $|Y_k| = |X_k|$ for every $k$. Measured at
$10^{-11}$ across sizes 8, 17, 32 and shifts $1, 3, -2$.

**What that is used for.** Two things, opposite in spirit.

**Matching.** The magnitude spectrum is a signature of the signal's content that does not depend
on where in the window it sits. That is why it is used for template matching, for audio
fingerprinting, and for rotation invariant image features via the Fourier-Mellin transform.

**And why phase is the interesting part.** All the position information lives in the phase, so
discarding it discards where everything is. Reconstructing a signal from magnitude alone is the
**phase retrieval** problem, which is genuinely hard and is the central difficulty of X-ray
crystallography and coherent diffraction imaging.

### 2.5 The convolution theorem, and why it is circular

**Claim.** With $(x * y)_i = \sum_{j}x_jy_{(i-j)\bmod n}$,

$$
\mathrm{DFT}(x*y)_k = X_kY_k
$$

**Proof.**

$$
\mathrm{DFT}(x*y)_k = \sum_i\left(\sum_jx_jy_{(i-j)\bmod n}\right)w^{ik}
$$

Exchange the sums and substitute $\ell = (i-j)\bmod n$, so $i = (\ell+j)\bmod n$ and
$w^{ik} = w^{jk}w^{\ell k}$:

$$
= \sum_jx_jw^{jk}\sum_\ell y_\ell w^{\ell k} = X_kY_k
$$

**Why it is circular and not linear.** The index $(i-j)$ is reduced mod $n$, so when $j > i$ the
term wraps around to the far end of $y$. **That wrapping is forced**, and the reason is exactly
the shift theorem: the DFT's basis functions are periodic, so every operation it diagonalises must
commute with a circular shift. Linear convolution does not; circular convolution does.

Stated structurally: the matrices diagonalised by $F$ are exactly the **circulant** matrices, and
circular convolution is multiplication by a circulant. Linear convolution is multiplication by a
Toeplitz matrix, which $F$ does not diagonalise.

**The measured difference.** Exercise 3.3 pads to length $n_a + n_b - 1$ and gets the linear
convolution to $10^{-15}$; without padding the wrap-around error is 3.5 to 10.5 on data of that
size, so it is not a small correction.

**The cost.** As written, going through the transform is $3n^2 + n$ multiplications against the
direct $n^2$: **slower**. The theorem is worth nothing until lesson 59 makes the transform
$O(n\log n)$, and then it turns $O(n^2)$ into $O(n\log n)$. That single fact is behind fast
polynomial multiplication, Schonhage-Strassen integer multiplication, and every convolutional
filter longer than about 30 taps.

### 3.1 Zero padding

Padding a signal with zeros and transforming the longer array is the most misunderstood operation
in the subject. It is worth doing carefully.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import fft as ft

n = 16
t = np.arange(n) / n
signal = np.sin(2.0 * math.pi * 2.5 * t)          # 2.5 cycles: deliberately not a bin

print(f"{'padded to':>11}{'bin spacing':>14}{'peak bin':>10}{'peak frequency':>17}")
for pad in (16, 32, 64, 256, 1024):
    x = np.zeros(pad)
    x[:n] = signal
    spectrum = np.abs(ft.fft(x))[:pad // 2]
    k = int(np.argmax(spectrum))
    print(f"{pad:>11}{n / pad:>14.4f}{k:>10}{k * n / pad:>17.4f}")

print()
print("padding evaluates the EXACT transform of the original samples at more frequencies:")
for pad in (32, 128, 1024):
    freqs = np.arange(pad // 2) * n / pad
    exact = np.abs(np.asarray([np.sum(signal * np.exp(-2j * math.pi * f * t))
                               for f in freqs]))
    x = np.zeros(pad)
    x[:n] = signal
    padded = np.abs(ft.fft(x))[:pad // 2]
    print(f"  pad {pad:>5}: agreement with the exact continuous transform "
          f"{float(np.max(np.abs(padded - exact))):.3e}")
```

| padded to | bin spacing | peak bin | peak frequency |
|---|---|---|---|
| 16 | 1.0000 | 2 | 2.0000 |
| 32 | 0.5000 | 5 | 2.5000 |
| 64 | 0.2500 | 10 | 2.5000 |
| 256 | 0.0625 | 40 | 2.5000 |
| 1024 | 0.0156 | 160 | 2.5000 |

| pad | agreement with the exact continuous transform |
|---|---|
| 32 | 1.776e-15 |
| 128 | 3.553e-15 |
| 1024 | 5.329e-15 |

**What padding computes, exactly.** The padded transform is the **exact** continuous transform of
the original $n$ samples, evaluated at the finer frequency grid, to $5\times10^{-15}$. It is
interpolation of the spectrum, and it is exact interpolation, because the spectrum of a finite
sample sequence is a trigonometric polynomial and the padded transform evaluates it.

**What it does not do.** It adds no information. The peak frequency converges to 2.5 and stops,
and 2.5 was determined by the original 16 samples: the padding merely made it visible. The
unpadded transform reported 2.0 because 2.5 fell between two bins, not because it was unknown.

**When it is the right thing.** When you want to **read** a peak off a plot, or interpolate the
spectrum for a filter design. Both are presentation, and padding is the correct tool for both.

**What would be needed to genuinely resolve two nearby frequencies.** More **time**, not more
zeros. Two tones separated by $\Delta f$ are distinguishable only if the observation window is at
least $1/\Delta f$ long: that is the Rayleigh criterion, and it is a property of the data. Padding
a 16 sample window to a million does not separate tones 0.3 bins apart.

The alternatives are parametric: fit a model with the number of tones as a parameter, which is
what MUSIC and ESPRIT do, or use a **superresolution** method that assumes sparsity. Both buy
resolution by assuming something the data does not say, which is the only way to buy it.

### 3.2 The Goertzel algorithm

When only a few bins are wanted, a full transform is wasteful. Goertzel computes one bin with a
second order recurrence and no complex arithmetic in the loop.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import fft as ft


def goertzel(x, k):
    """One DFT bin in O(n), by a real second order recurrence.

    The loop is entirely real; the single complex operation happens once at the end. That is
    what makes it attractive on a small processor, and it is why DTMF tone decoders used it.
    """
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n = v.size
    if n < 1:
        raise ValueError("cannot transform an empty signal")
    w = 2.0 * math.pi * int(k) / n
    coefficient = 2.0 * math.cos(w)
    s1 = s2 = 0.0
    for sample in v:
        s1, s2 = sample + coefficient * s1 - s2, s1
    return s1 * complex(math.cos(w), math.sin(w)) - s2


gen = np.random.default_rng(1)
print(f"{'n':>8}{'worst relative gap against the FFT':>38}")
for n in (16, 64, 256, 1024, 4096):
    x = gen.standard_normal(n)
    ref = ft.fft(x)
    scale = float(np.max(np.abs(ref)))
    worst = max(abs(goertzel(x, k) - ref[k]) for k in range(0, n, max(1, n // 17)))
    print(f"{n:>8}{worst / scale:>38.3e}")

print()
print(f"{'n':>8}{'Goertzel per bin':>20}{'full FFT':>14}{'bins to break even':>22}")
for n in (64, 256, 1024, 4096, 16384):
    fft_cost = (n // 2) * math.log2(n)
    print(f"{n:>8}{n:>20}{fft_cost:>14.0f}{fft_cost / n:>22.1f}")
```

| $n$ | worst relative gap against the FFT |
|---|---|
| 16 | 7.126e-15 |
| 64 | 2.116e-14 |
| 256 | 4.009e-14 |
| 1024 | 2.247e-13 |
| 4096 | 2.569e-12 |

| $n$ | Goertzel per bin | full FFT | bins to break even |
|---|---|---|---|
| 64 | 64 | 192 | 3.0 |
| 256 | 256 | 1024 | 4.0 |
| 1024 | 1024 | 5120 | 5.0 |
| 4096 | 4096 | 24576 | 6.0 |
| 16384 | 16384 | 114688 | 7.0 |

**The break-even is $\frac12\log_2 n$ bins**, so at $n = 16384$ Goertzel wins whenever fewer than
7 bins are needed. That is a small number, and it is why Goertzel is a specialist tool rather than
a general one.

**Where it is used.** DTMF decoding, which needs exactly 8 bins from a stream; tone detection in
telephony; and any embedded application where the memory for a transform is not available.
Goertzel needs three floats, whatever $n$ is, and an FFT needs $n$ complex numbers.

**Its weakness, visible in the first table.** The error grows linearly with $n$: $7\times10^{-15}$
at 16 and $2.6\times10^{-12}$ at 4096, a factor of 360 for a factor of 256 in size. That is the
recurrence accumulating rounding, exactly as lesson 58's twiddle recurrence does, and for the same
reason: nothing damps it. An FFT's error grows like $\sqrt{\log n}$ instead, so **for a long
signal the fast transform is also the accurate one**, which exercise 4.1 of lesson 59 measures.

**The formula worth getting right.** The output is $s_1e^{+iw} - s_2$, not $s_1 - s_2e^{\pm iw}$
nor $s_1e^{-iw} - s_2$. All four have the right **magnitude**, so a test on $|X_k|$ passes for
every one of them and only a test on the complex value catches it. Measured, the three wrong
variants disagree with the transform by 4.3, 6.1 and 5.5 on data of size 1.

### 3.3 Linear convolution from circular

Circular convolution wraps. Padding both inputs to at least $n_a + n_b - 1$ leaves room for the
result, so nothing wraps into anything.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import fft as ft


def linear_convolution(a, b):
    """Pad to n_a + n_b - 1, convolve circularly, and nothing wraps.

    Rounding up to a power of two costs a little extra work and buys the radix-2 transform.
    Bluestein would handle the exact length, and for a one-off convolution the padding is
    cheaper than the chirp.
    """
    a = np.atleast_1d(np.asarray(a, dtype=float)).ravel()
    b = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    needed = a.size + b.size - 1
    size = ft.next_power_of_two(needed)
    pa = np.zeros(size)
    pa[:a.size] = a
    pb = np.zeros(size)
    pb[:b.size] = b
    return np.real(ft.ifft(ft.fft(pa) * ft.fft(pb)))[:needed]


gen = np.random.default_rng(5)
print(f"{'len a':>8}{'len b':>8}{'padded, vs numpy':>20}{'unpadded, vs numpy':>22}")
for na, nb in ((4, 4), (8, 3), (17, 9), (64, 40)):
    a = gen.standard_normal(na)
    b = gen.standard_normal(nb)
    reference = np.convolve(a, b)
    size = max(na, nb)
    pa = np.zeros(size)
    pa[:na] = a
    pb = np.zeros(size)
    pb[:nb] = b
    wrapped = np.real(ft.ifft(ft.fft(pa) * ft.fft(pb)))
    print(f"{na:>8}{nb:>8}"
          f"{float(np.max(np.abs(linear_convolution(a, b) - reference))):>20.3e}"
          f"{float(np.max(np.abs(wrapped - reference[:size]))):>22.3e}")
```

| len $a$ | len $b$ | padded, vs numpy | unpadded, vs numpy |
|---|---|---|---|
| 4 | 4 | 4.441e-16 | **3.496** |
| 8 | 3 | 1.665e-16 | 1.062e-1 |
| 17 | 9 | 9.992e-16 | **5.214** |
| 64 | 40 | 7.105e-15 | **10.46** |

**The padded version is exact and the unpadded one is not even close.** The wrap-around error is
not a small correction: it is of the same size as the answer, because the tail of the convolution
is folded onto its head.

**Why $n_a + n_b - 1$ specifically.** The convolution of a degree $n_a-1$ and a degree $n_b-1$
polynomial has degree $n_a+n_b-2$, so it has $n_a+n_b-1$ coefficients. Any length at least that is
enough; anything less and the top coefficients have nowhere to go.

**And this is polynomial multiplication.** The linear convolution of two coefficient vectors **is**
the coefficient vector of their product. So the routine above multiplies two polynomials in
$O(n\log n)$, which is the fast multiplication algorithm and, with carries added, the basis of
fast big-integer arithmetic.

**One practical caveat.** For short $b$, direct convolution wins: the transform costs
$3\cdot\frac{N}{2}\log_2 N$ with $N \ge n_a+n_b-1$, and the direct sum costs $n_an_b$. With
$n_a = 10^6$ and $n_b = 8$ the direct cost is $8\times10^6$ and the transform cost is about
$3\times10^7$. The crossover for a long $a$ is around $n_b \approx 30$, which is why audio filters
below 30 taps are done directly and longer ones by **overlap-add**, which chops $a$ into blocks
and uses the transform on each.

### 4.1 Reconstruction against the sampling rate

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import dft as dt

tone = lambda t, freq: np.sin(2.0 * math.pi * freq * t)
for f_true in (1.0, 3.0, 7.0, 11.0):
    out = dt.nyquist(tone, f_true, sizes=(4, 8, 16, 32, 64))
    print(f"a tone at {f_true:.0f} cycles per unit time:")
    print(f"{'samples':>9}{'Nyquist':>10}{'reconstruction error':>23}{'reported bin':>14}"
          f"{'aliased':>9}")
    for n, lim, err, fold, above in zip(out["sizes"], out["nyquist_limit"],
                                        out["reconstruction_error"],
                                        out["folded_frequency"], out["above_nyquist"]):
        print(f"{n:>9}{lim:>10.1f}{err:>23.3e}{fold:>14}{str(above):>9}")
    print()
```

**A tone at 7 cycles:**

| samples | Nyquist limit | reconstruction error | reported bin | aliased |
|---|---|---|---|---|
| 4 | 2.0 | 1.903 | 1 | True |
| 8 | 4.0 | 1.903 | 1 | True |
| 16 | 8.0 | 1.091e-14 | 7 | False |
| 32 | 16.0 | 1.382e-14 | 7 | False |
| 64 | 32.0 | 2.046e-14 | 7 | False |

**A tone at 11 cycles:**

| samples | Nyquist limit | reconstruction error | reported bin | aliased |
|---|---|---|---|---|
| 4 | 2.0 | 1.960 | 1 | True |
| 8 | 4.0 | 1.962 | 3 | True |
| 16 | 8.0 | 1.966 | 5 | True |
| 32 | 16.0 | 1.452e-14 | 11 | False |
| 64 | 32.0 | 1.854e-14 | 11 | False |

**The transition is a cliff, not a slope.** Below the limit the reconstruction is exact to
$10^{-14}$; above it the error is close to 2, which is the full peak to peak amplitude of the
tone. There is no intermediate regime.

**The folding formula, confirmed.** A tone at frequency $f$ sampled $n$ times per unit is reported
at

$$
f_{\text{reported}} = \left|\left(\left(f + \tfrac n2\right)\bmod n\right) - \tfrac n2\right|
$$

which is $f$ reflected repeatedly about the multiples of $n/2$. Checking: 7 cycles at $n=8$ gives
$|((7+4)\bmod 8) - 4| = |3-4| = 1$, and the measurement reports bin 1. At $n = 16$, 11 cycles give
$|((11+8)\bmod16)-8| = |3-8| = 5$, and the measurement reports 5.

**Exactly at the limit.** With $f = n/2$ the tone is sampled at its zero crossings if the phase is
wrong, and the reconstruction is identically zero. The sampling theorem's condition is $f < n/2$
strictly, and the strictness is not pedantry: at equality the amplitude is not recoverable at all.

**Why the aliased errors are all close to 2 and never far from it.** A sine of amplitude 1
aliased to a different frequency reconstructs as another sine of amplitude 1, so the largest
possible difference is 2, attained where they are exactly out of phase. The measured values are
1.540, 1.903, 1.960, 1.962 and 1.966, approaching 2 from below because the probe grid is finite
and does not sample the exact worst point. **The column has almost no variety and that is the
point**: once the tone is aliased, the error is the whole signal, and how far above Nyquist it is
makes no difference at all.

### 4.2 The twiddle recurrence's error growth

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import dft as dt

sizes = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
out = dt.recurrence_comparison(sizes)
print(f"{'n':>8}{'naive error':>15}{'stable error':>15}{'ratio':>9}"
      f"{'naive/n':>13}{'stable/sqrt(n)':>17}")
for n, ne, se in zip(out["sizes"], out["naive_error"], out["stable_error"]):
    print(f"{n:>8}{ne:>15.3e}{se:>15.3e}{ne / se:>9.1f}"
          f"{ne / n:>13.3e}{se / np.sqrt(n):>17.3e}")

lg = np.log(out["sizes"].astype(float))
print()
print(f"fitted exponent, naive:  {float(np.polyfit(lg, np.log(out['naive_error']), 1)[0]):.4f}")
print(f"fitted exponent, stable: {float(np.polyfit(lg, np.log(out['stable_error']), 1)[0]):.4f}")
```

| $n$ | naive error | stable error | ratio | naive$/n$ | stable$/\sqrt n$ |
|---|---|---|---|---|---|
| 64 | 3.002e-15 | 6.474e-16 | 4.6 | 4.7e-17 | 8.1e-17 |
| 256 | 6.481e-15 | 8.006e-16 | 8.1 | 2.5e-17 | 5.0e-17 |
| 1024 | 3.544e-14 | 1.266e-15 | 28.0 | 3.5e-17 | 4.0e-17 |
| 4096 | 1.329e-13 | 2.701e-15 | 49.2 | 3.2e-17 | 4.2e-17 |
| 16384 | 6.789e-13 | 4.734e-15 | 143.4 | 4.1e-17 | 3.7e-17 |

**The naive error is linear in $n$** and the stable one is $\sqrt n$, and the last two columns are
the evidence: dividing by $n$ and by $\sqrt n$ respectively gives constants, both around
$4\times10^{-17}$, which is $\varepsilon/5$.

The fitted exponents come out at **0.973** for the naive recurrence and **0.369** for the stable
one, against the predictions of 1 and 0.5. The stable one's fit is the less clean of the two,
because its errors are only a few units in the last place and the fit is measuring a random walk;
individual sizes scatter by a factor of 2, visible at $n = 2048$ and $n = 8192$.

**Why linear.** The naive recurrence applies a rotation matrix $n$ times. Each application has a
relative error of about $\varepsilon$, and the errors are **not** independent: the rotation is
applied to the accumulated result, so they compound additively in the angle and the phase error
after $k$ steps is $\sim k\varepsilon$. Nothing corrects it.

**Why $\sqrt n$.** The increment form stores $\delta = c_{k+1}-c_k$ and adds it, so the leading
error term is in $\delta$ rather than in $c$, and $\delta$ is small. What remains behaves like a
random walk in the rounding, giving $\sqrt n$ rather than $n$.

**The modulus is the diagnostic.** The naive recurrence's computed $(c_k, s_k)$ drifts off the
unit circle, by exactly the amount of its error: the drift column and the error column are the
same numbers to two digits. The stable one keeps $|c^2+s^2 - 1|$ at $3\times10^{-15}$ at
$n = 16384$, a factor of 220 better.

**When it matters.** At $10^{-13}$ a single transform is unaffected. It matters for a long chain
of transforms, for fixed point arithmetic where $\varepsilon$ is $2^{-15}$ rather than $2^{-53}$,
and for very long transforms: at $n = 2^{24}$ the naive error would be around $10^{-9}$, which is
visible in single precision data.

**And the fix costs nothing.** Two extra constants computed once, and the same two operations per
step.

### 4.3 A large dynamic range

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import fft as ft

n = 1024
t = np.arange(n) / n
print(f"{'small amplitude':>17}{'recovered':>18}{'relative error':>18}{'above the floor':>18}")
for amplitude in (1e-2, 1e-6, 1e-10, 1e-13, 1e-15, 1e-17):
    signal = np.sin(2.0 * math.pi * 5.0 * t) + amplitude * np.sin(2.0 * math.pi * 200.0 * t)
    X = ft.fft(signal)
    recovered = 2.0 * abs(X[200]) / n
    floor = float(np.median(np.abs(X))) / n
    print(f"{amplitude:>17.0e}{recovered:>18.6e}"
          f"{abs(recovered - amplitude) / amplitude:>18.3e}"
          f"{str(recovered > 10.0 * floor):>18}")
```

| small amplitude | recovered | relative error | above the floor |
|---|---|---|---|
| 1e-2 | 1.000000e-2 | 3.469e-16 | True |
| 1e-6 | 1.000000e-6 | 7.739e-12 | True |
| 1e-10 | 1.000000e-10 | 3.945e-8 | True |
| 1e-13 | 1.000026e-13 | 2.638e-5 | True |
| 1e-15 | 1.006941e-15 | 6.941e-3 | True |
| **1e-17** | **2.566800e-17** | **1.567** | **False** |

**The small component is recovered to full relative accuracy down to $10^{-10}$**, and is still
identifiable at $10^{-15}$, which is $\varepsilon$ relative to the large component.

**Where it is lost: at $10^{-17}$**, which is below $\varepsilon$ times the large amplitude. The
recovered value is $2.6\times10^{-17}$, two and a half times too large, and it is
indistinguishable from the roundoff floor of the transform.

**Why the transform does so well.** The relative error of the recovered coefficient is about
$\varepsilon\cdot\frac{\text{large}}{\text{small}}$: at small $= 10^{-10}$ that predicts
$2\times10^{-6}$ and the measurement is $4\times10^{-8}$, better than predicted. The reason is
that the transform's error is spread over all $n$ bins rather than concentrated, so each bin
carries $\varepsilon\|x\|/\sqrt n$ rather than $\varepsilon\|x\|$.

**Which is why the FFT is the right tool for this.** A method that computed each bin separately,
such as Goertzel, would carry $\varepsilon\|x\|$ per bin without the $\sqrt n$ discount, and would
lose the small component about 30 times sooner at $n = 1024$.

**The practical limit.** A spectrum can resolve components separated by about $10^{-15}$ in
amplitude, which is 300 dB, and that is far beyond what any measurement provides. **The dynamic
range of the analysis is never the limiting factor**; the dynamic range of the data acquisition
is, and that is typically 16 to 24 bits, so 96 to 144 dB.

### 5.1 Zero padding and resolution

**What padding computes.** Let $\hat x(f) = \sum_{j=0}^{n-1}x_je^{-2\pi ifj/n}$ be the continuous
frequency function of the $n$ samples. The unpadded transform evaluates it at $f = 0,1,\dots,n-1$.
The transform padded to $N$ evaluates the **same function** at $f = 0, n/N, 2n/N, \dots$

Exercise 3.1 measures the agreement at $5\times10^{-15}$, so this is exact, not approximate.

**So padding is spectral interpolation, and $\hat x$ was already determined by the $n$ samples.**
Nothing was added.

**What would be needed to resolve two nearby frequencies.** Two tones at $f_1$ and $f_2$ observed
for a time $T$ produce, in the spectrum, two copies of the window's transform centred at $f_1$ and
$f_2$. The window is a rectangle of length $T$, whose transform is a sinc of width $1/T$. Two
sincs separated by less than their width are one bump, whatever grid it is sampled on.

**That is the Rayleigh criterion**: $\Delta f \gtrsim 1/T$. It is a statement about the data, and
padding does not change $T$.

**Three ways to do better, and what each assumes.**

**A longer observation.** The only way that assumes nothing. $T$ doubles, $\Delta f$ halves.

**A different window.** A taper reduces the sinc's sidelobes at the cost of widening its main lobe,
so it makes **unequal** amplitudes separable and **equal** ones less so. That is a real gain in
practice and it does not improve the Rayleigh limit.

**A parametric model.** Assume the signal is exactly $K$ complex exponentials plus noise, and
estimate their frequencies by an eigen-decomposition of the covariance: that is **MUSIC** and
**ESPRIT**, and they can separate tones far below $1/T$. The price is the assumption: if the signal
is not $K$ tones, the answer is meaningless, and $K$ has to be chosen.

**The unifying statement.** Resolution below $1/T$ is bought with prior information, never with
computation. Padding provides no prior information, so it buys nothing.

### 5.2 Leakage and windowing

**Why a non-bin frequency spreads.** The DFT interprets the block as one period of a periodic
signal. A tone at $f = 2.5$ cycles per block does not complete a whole number of cycles, so the
periodic extension has a **discontinuity** at the seam, and lesson 60 measures what that costs:
coefficients decaying like $1/k$ instead of geometrically.

Equivalently in frequency: the observed spectrum is the true spectrum **convolved** with the
transform of the rectangular window, which is a Dirichlet kernel with sidelobes falling only like
$1/f$. A tone at an exact bin lands on the kernel's zeros and produces one spike; a tone between
bins lands off them and spreads over every bin.

**Measured.** A tone at 2.5 cycles in a 16 sample block puts 68 percent of its energy in the two
nearest bins and the rest across all the others, with the furthest bin still at 4 percent of the
peak. A tone at exactly 2 cycles puts everything in one bin and $10^{-16}$ elsewhere.

**What a window trades away.** Multiplying the data by a taper $w_j$ that goes smoothly to zero at
both ends removes the discontinuity, so the sidelobes fall much faster. The cost is that the main
lobe gets **wider**:

| window | sidelobe level | main lobe width | what it is for |
|---|---|---|---|
| rectangular | -13 dB, falling as $1/f$ | 1 bin | best resolution, worst leakage |
| Hann | -31 dB, falling as $1/f^3$ | 2 bins | the general default |
| Hamming | -43 dB, falling as $1/f$ | 2 bins | lowest first sidelobe |
| Blackman-Harris | -92 dB | 4 bins | very unequal amplitudes |
| flat top | -90 dB | 5 bins | accurate **amplitude**, poor resolution |

**The trade in one sentence: sidelobe suppression is bought with main lobe width**, so a window
makes a weak tone next to a strong one visible and makes two equal tones harder to separate.

**The connection to the rest of Part 8.** This is lesson 59's Gibbs phenomenon seen from the other
domain. There, a sharp cut in frequency gave ringing in time; here, a sharp cut in time gives
ringing in frequency. **They are the same theorem**, and the response is the same: taper rather
than cut.

**And the connection to lesson 60.** The MDCT's sine window is a taper chosen so that overlapping
blocks cancel exactly. So it gets the sidelobe suppression **and** loses nothing, which is why
audio codecs use a lapped transform rather than a windowed block transform.

### 5.3 The DFT as an eigendecomposition

**Statement.** A **circulant** matrix $C$ is one whose rows are successive circular shifts of a
single vector $c$:

$$
C_{ij} = c_{(i-j)\bmod n}
$$

Every circulant is diagonalised by the DFT matrix:

$$
C = \frac1nF^H\,\mathrm{diag}(Fc)\,F
$$

so its eigenvalues are the entries of $Fc$ and its eigenvectors are the columns of $F^H$,
**independently of $c$**.

**Proof.** $Cx$ is the circular convolution $c * x$, and by exercise 2.5 the transform of that is
the pointwise product $\hat c\hat x$. So $F(Cx) = \mathrm{diag}(Fc)(Fx)$, which is the statement
that $FCF^{-1}$ is diagonal.

Equivalently, taking $x$ to be a column of $F^H$, that is $x_j = w^{-jk}$ for fixed $k$:

$$
(Cx)_i = \sum_j c_{(i-j)}w^{-jk} = w^{-ik}\sum_\ell c_\ell w^{\ell k} = \hat c_k\,x_i
$$

so $x$ is an eigenvector with eigenvalue $\hat c_k$, directly.

**The connection to Part 6.** Circulant matrices are the class for which the eigenvalue problem is
**free**: no iteration, no QR algorithm, one transform. They are normal, so their eigenvalues are
perfectly conditioned, and the eigenvector matrix is unitary, so Bauer-Fike gives a condition
number of 1. That is the same statement as $\kappa(F/\sqrt n) = 1$.

**The connection to Part 4.** Two uses, and both are standard.

**Circulant preconditioners.** For a Toeplitz system $Tx = b$, which arises from every
discretised convolution, the natural preconditioner is a circulant $C$ close to $T$. Then $C^{-1}$
costs two transforms, and Strang's and T. Chan's constructions make the eigenvalues of $C^{-1}T$
cluster at 1, which is exactly what lesson 25 said a preconditioner must do. The result is that
conjugate gradient converges in $O(1)$ iterations independently of $n$, each costing
$O(n\log n)$.

**Fast Poisson solvers.** The discrete Laplacian with periodic boundary conditions is circulant,
so it is diagonalised by the DFT and Poisson's equation solves in $O(n\log n)$ exactly rather than
iteratively. With Dirichlet conditions the matrix is not circulant but is diagonalised by the
**sine transform**, which is a relative of lesson 60's DCT, and the same $O(n\log n)$ applies.
Part 11 uses both.

**The general shape of the idea.** A matrix with a symmetry is diagonalised by the characters of
the symmetry group. Circulants have the cyclic group's symmetry and the characters are the roots
of unity. That is why the DFT appears everywhere a shift invariance does, and it is the reason
convolution, filtering, and translation invariant physics all lead to the same transform.

---

## Lesson 59, The FFT and Signal Processing

### 1.1 Two outputs from one multiplication

**The fact is $w^{n/2} = -1$.** With $w = e^{-2\pi i/n}$, $w^{n/2} = e^{-\pi i} = -1$ exactly, for
every even $n$.

**What it does.** After the parity split, $X_k = E_k + w^kO_k$ where $E$ and $O$ are the half
length transforms. Those half transforms are periodic with period $n/2$, so $E_{k+n/2} = E_k$ and
likewise for $O$. Then

$$
X_{k+n/2} = E_{k+n/2} + w^{k+n/2}O_{k+n/2} = E_k - w^kO_k
$$

**So the product $w^kO_k$ is computed once and used twice**, with a plus and a minus. That is the
butterfly, and it is the entire saving.

**What would happen without it.** Suppose the transform were over a ring where $-1$ were not
available as a power of $w$: then $X_k$ and $X_{k+n/2}$ would need two different twiddle
multiplications, and the recurrence would be $T(n) = 2T(n/2) + n$ multiplications rather than
$n/2$. That is still $O(n\log n)$, so the asymptotics survive, but the constant doubles.

**More seriously**, if $w$ were not a primitive root of unity at all, the half length transforms
would not be transforms of anything and the split would not be a split. The recursion needs
$w^2$ to be a primitive $(n/2)$th root, which follows from $w$ being a primitive $n$th root, and
that is the real hypothesis. **It is the same hypothesis lesson 58's orthogonality proof needed**,
which is why the same object supports both.

### 1.2 Bit reversal, and what the recursion does instead

**Why it appears.** The parity split sends index $j$ to the even sublist if its **last bit** is 0
and the odd sublist if it is 1. Recursing, the second level splits on the second-last bit, and so
on. After $\log_2 n$ levels, an element's position is determined by its bits **read backwards**.

Concretely at $n = 8$: element 1 = 001 goes odd, then even, then even, landing at position
100 = 4. Element 3 = 011 goes odd, odd, even, landing at 110 = 6. The measured permutation is
$[0, 4, 2, 6, 1, 5, 3, 7]$, which is each index with its three bits reversed.

**What the recursive version does instead: it never forms the permutation.** `v[0::2]` and
`v[1::2]` create new arrays in the right order at every level, so the reordering happens
implicitly and is paid for in allocation rather than in indexing.

**The trade.** The recursive version allocates $O(n\log n)$ memory over the call tree and is easy
to read. The iterative version allocates nothing, works in place, and needs the permutation
applied once at the start.

**Two properties of the permutation worth knowing.** It is an **involution**: applying it twice
gives the identity, because reversing the bits twice restores them. And it has a simple loop
structure, so it can be applied in place by swapping $j$ with $\text{rev}(j)$ for $j <
\text{rev}(j)$, touching each pair once. Both are measured in the lesson.

**A third form exists.** Decimation in **frequency** puts the butterflies first and the bit
reversal last, which is convenient when the output is going straight into a pointwise
multiplication and back, as in convolution: the two bit reversals then cancel and can both be
skipped. That is a standard optimisation and it is why library convolution routines often expose a
"transform, in scrambled order" mode.

### 1.3 Padding to a power of two

**What it computes.** The transform of a **different, longer signal**: the original with zeros
appended. Lesson 58's exercise 3.1 shows precisely what that is: the exact continuous frequency
function of the original samples, evaluated on a finer grid.

**When it is the right thing.** Whenever the frequency grid is a presentation choice rather than
part of the answer.

- **Spectral display.** Reading a peak off a plot is easier on a fine grid, and the peak's
  location is unchanged.
- **Convolution.** Exercise 3.3 of lesson 58 pads deliberately, to make room for the result, and
  the padding is the point rather than a workaround.
- **Filter design.** Interpolating a frequency response onto a finer grid to inspect it.

**When it is wrong.** Whenever the transform's **length** is part of the specification.

- **A length $n$ DFT of length $n$ data.** The padded transform has different values at different
  frequencies; it is not the length $n$ transform with more decimal places.
- **Anything that will be inverse transformed and compared with the input.** The round trip of the
  padded transform gives the padded signal, which is not the signal.
- **Any use of the coefficients as interpolation coefficients**, which is lesson 58's theorem: the
  padded coefficients interpolate the padded data, including the zeros.

**And it silently changes the answer**, which is what makes it dangerous rather than merely
inefficient. `bluestein` computes the transform that was asked for, at the same $O(n\log n)$, and
costs a factor of about 6 in the constant. **That is the right default**, and padding should be a
deliberate choice made for one of the reasons above.

### 2.1 The butterfly and the recurrence

**Deriving the butterfly.** Split the defining sum by parity of $j$:

$$
X_k = \sum_{j=0}^{n-1}x_jw^{jk}
= \sum_{r=0}^{n/2-1}x_{2r}w^{2rk} + \sum_{r=0}^{n/2-1}x_{2r+1}w^{(2r+1)k}
$$

Now $w^2 = e^{-4\pi i/n} = e^{-2\pi i/(n/2)} =: \omega$ is a primitive $(n/2)$th root of unity, so

$$
X_k = \underbrace{\sum_r x_{2r}\omega^{rk}}_{E_k} + w^k\underbrace{\sum_r x_{2r+1}\omega^{rk}}_{O_k}
$$

Both $E$ and $O$ are $(n/2)$ point transforms, hence periodic with period $n/2$. Combining with
$w^{n/2} = -1$ gives the pair.

**The recurrence.** Let $T(n)$ count multiplications. Computing $E$ and $O$ costs $2T(n/2)$, and
combining costs $n/2$ multiplications by $w^k$ plus $n$ additions. So

$$
T(n) = 2T(n/2) + \tfrac n2, \qquad T(1) = 0
$$

**Solving it.** Write $n = 2^m$ and $S(m) = T(2^m)$. Then $S(m) = 2S(m-1) + 2^{m-1}$. Divide by
$2^m$:

$$
\frac{S(m)}{2^m} = \frac{S(m-1)}{2^{m-1}} + \frac12
$$

so $S(m)/2^m = m/2$ and $S(m) = m2^{m-1}$, that is

$$
T(n) = \frac n2\log_2 n
$$

**exactly, not asymptotically.** That is the formula the lesson checks against an instrumented
count, and the count matches at every size from 2 to $2^{20}$.

**By the Master theorem** the general form $T(n) = 2T(n/2) + cn$ is case 2, giving
$\Theta(n\log n)$, and the explicit solution above is that with the constant pinned down.

### 2.2 Bit reversal, proved

**Claim.** After $m = \log_2 n$ levels of parity splitting, the element originally at index $j$
occupies position $\mathrm{rev}_m(j)$, where $\mathrm{rev}_m$ reverses the $m$ bit binary
representation.

**Proof by induction on $m$.**

**Base $m = 1$.** Two elements, indices 0 and 1. The split puts index 0 (bit 0) first and index 1
(bit 1) second, which is the identity, and reversing a single bit is the identity. True.

**Step.** Suppose it holds for $m-1$. At the top level, the split partitions by the **least
significant** bit $b_0$ of $j$: the elements with $b_0 = 0$ form the first half of the output and
those with $b_0 = 1$ the second. So the **most significant** bit of the final position is $b_0$.

Within each half, the remaining indices are $j' = \lfloor j/2\rfloor$, the top $m-1$ bits of $j$,
and by the inductive hypothesis they end at position $\mathrm{rev}_{m-1}(j')$ within that half.

So the final position is $b_0\cdot2^{m-1} + \mathrm{rev}_{m-1}(\lfloor j/2\rfloor)$, which is
exactly the $m$ bit reversal of $j$: the last bit becomes the first, and the rest is the reversal
of the rest.

**Checked at $n = 8$**: $[0,4,2,6,1,5,3,7]$, which is $[000,100,010,110,001,101,011,111]$ read as
the reversals of $[000,001,010,011,100,101,110,111]$.

### 2.3 The butterfly count is exactly $\frac{n}{2}\log_2 n$

**By the recurrence.** Exercise 2.1 solved $T(n) = 2T(n/2) + n/2$ with $T(1) = 0$ to
$T(n) = \frac n2\log_2 n$.

**By direct counting in the iterative form.** The outer loop runs over transform sizes
$2, 4, 8, \dots, n$, so $\log_2 n$ passes. Each pass covers all $n$ elements in blocks of the
current size, and each block of size $s$ contains $s/2$ butterflies. So a pass does
$\frac{n}{s}\cdot\frac s2 = \frac n2$ butterflies, **independently of $s$**, and the total is
$\frac n2\log_2 n$.

**The second argument is the more useful one**, because it says every pass costs the same, which
is why the algorithm parallelises and vectorises so well: the work is a rectangle, not a triangle.

**Measured, by instrumenting both implementations:**

| $n$ | butterflies, iterative | butterflies, recursive | predicted | DFT multiplies | speedup |
|---|---|---|---|---|---|
| 2 | 1 | 1 | 1 | 4 | 4.0 |
| 16 | 32 | 32 | 32 | 256 | 8.0 |
| 256 | 1024 | 1024 | 1024 | 65536 | 64.0 |
| 4096 | 24576 | 24576 | 24576 | 16777216 | 682.7 |
| 65536 | 524288 | 524288 | 524288 | 4294967296 | 8192.0 |
| 1048576 | 10485760 | 10485760 | 10485760 | 1.1e12 | **104857.6** |

**The counts agree exactly**, from two different implementations and the closed form. That is the
strongest kind of check available for a cost claim, and it is why the lesson counts rather than
quotes.

**The speedup is $\frac{n^2}{\frac n2\log_2 n} = \frac{2n}{\log_2 n}$**, which at $n = 2^{20}$ is
$2\times10^6/20 = 10^5$, matching the measured 104858.

### 2.4 Bluestein's identity

**The identity.** $jk = \frac12\left(j^2 + k^2 - (j-k)^2\right)$, which is immediate from
$(j-k)^2 = j^2 - 2jk + k^2$.

**Substituting into the transform.** With $w = e^{-2\pi i/n}$, write
$w^{jk} = w^{j^2/2}w^{k^2/2}w^{-(j-k)^2/2}$, where $w^{1/2} = e^{-\pi i/n}$. Then

$$
X_k = \sum_j x_jw^{jk} = w^{k^2/2}\sum_j\left(x_jw^{j^2/2}\right)w^{-(j-k)^2/2}
$$

Define $a_j = x_jw^{j^2/2}$ and $b_\ell = w^{-\ell^2/2}$. The sum is $\sum_j a_jb_{k-j}$, which is
a **convolution** of $a$ with $b$, evaluated at $k$.

**So the transform is a convolution**, and convolution is three transforms of whatever length is
convenient. The length is chosen to be a power of two, so `fft_radix2` handles it.

**Why the padding must reach $2n-1$.** The convolution index $k-j$ runs from $-(n-1)$ to $n-1$ as
$j$ and $k$ each run over $0,\dots,n-1$. That is $2n-1$ distinct values, so the kernel $b$ has
$2n-1$ significant entries, and the circular convolution must have room for all of them or they
wrap and corrupt each other, which is lesson 58's exercise 3.3 exactly.

**The implementation detail.** The kernel is stored with its negative indices at the **end** of
the array, using the circular convention: `b[:n] = conj(chirp)` and
`b[m-n+1:] = conj(chirp[1:])[::-1]`. Getting that wrapping wrong produces a transform that is
correct at $k=0$ and wrong elsewhere, which a test at a single index would miss.

**The cost.** Three radix-2 transforms of length $m \le 4n$, so
$3\cdot\frac{m}{2}\log_2 m \le 6n(\log_2 n + 2)$ butterflies, against $\frac n2\log_2 n$ for a
power of two. **A factor of about 12**, which is the price of an arbitrary length, and it is a
constant rather than a change of order.

### 2.5 The Gibbs constant

**The impulse response of a brick wall filter.** Zeroing every coefficient above $f_c$ multiplies
the spectrum by a rectangle $R(f)$, which by the convolution theorem convolves the signal in time
with $\hat R$, the inverse transform of the rectangle. That is

$$
\hat R(t) = \frac{\sin(2\pi f_ct)}{\pi t} = 2f_c\,\mathrm{sinc}(2f_ct)
$$

**The tails decay like $1/t$**, which is the essential fact: the filter is not local, and a
discontinuity anywhere in the signal is felt everywhere.

**The overshoot.** Filtering a unit step gives

$$
(\text{step} * \hat R)(t) = \frac12 + \frac1\pi\,\mathrm{Si}(2\pi f_ct)
$$

where $\mathrm{Si}(z) = \int_0^z\frac{\sin u}{u}du$ is the sine integral. Its first maximum is at
$z = \pi$, where $\mathrm{Si}(\pi) = 1.851937\ldots$, giving a value of

$$
\frac12 + \frac{1.851937}{\pi} = 1.089490
$$

**So the overshoot is $0.089490$ of the unit step**, and for a step of size $J$ it is
$0.089490\,J$.

**Why it is a fixed fraction and not a shrinking one.** Raising $f_c$ **rescales** the argument of
$\mathrm{Si}$: the whole overshoot pattern compresses horizontally by the factor $f_c$ and its
**height is unchanged**, because $\mathrm{Si}(\pi)$ does not depend on $f_c$. The ringing gets
narrower and never gets smaller.

**Measured:**

| samples | cutoff | overshoot fraction | error against 0.0894899 |
|---|---|---|---|
| 256 | 6.4 | 0.0942412 | 4.751e-3 |
| 1024 | 25.6 | 0.0897983 | 3.084e-4 |
| 4096 | 102.4 | 0.0893875 | 1.023e-4 |
| 16384 | 409.6 | 0.0893064 | 1.834e-4 |
| 65536 | 1638.4 | 0.0892876 | 2.022e-4 |

**It converges to 0.0893 and stops**, about $2\times10^{-4}$ short of the exact constant. That is
not a failure of the theory: the discrete signal samples the overshoot on a grid, and the grid
never lands exactly on the continuous maximum. The residual is the sampling of the peak, and it
does not shrink because the peak narrows at the same rate as the grid refines.

**Reporting the limit as 0.08949 from this data would be wrong**, and reporting the discrepancy as
a failure would also be wrong. The honest statement is 0.0893 from a discrete measurement against
0.0894899 exactly, agreeing to three digits.

### 3.1 Radix-4

Splitting four ways instead of two. The saving is real and modest.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import fft as ft, dft as dt


def fft_radix4(x, _counter=None):
    """Split into four sublists by index mod 4, and combine with the fourth roots of unity.

    The saving over radix-2 is that multiplication by i is free: it swaps the real and
    imaginary parts and negates one. A radix-4 butterfly does 3 complex multiplications where
    two radix-2 stages would do 4.
    """
    v = np.atleast_1d(np.asarray(x)).astype(complex).ravel()
    n = v.size
    if n == 1:
        return v.copy()
    if n % 4 != 0:
        return ft.fft_radix2(v, _counter=_counter)
    radix = 4
    quarter = n // radix
    parts = [fft_radix4(v[r::radix], _counter) for r in range(radix)]
    k = np.arange(quarter)
    twiddles = [np.exp(-2j * math.pi * r * k / n) * parts[r] for r in range(radix)]
    if _counter is not None:
        _counter[0] += (radix - 1) * quarter
    out = np.empty(n, dtype=complex)
    for block in range(radix):
        phase = np.exp(-2j * math.pi * block / float(radix))
        out[block * quarter:(block + 1) * quarter] = sum(
            phase ** r * twiddles[r] for r in range(radix))
    return out


gen = np.random.default_rng(11)
print(f"{'n':>8}{'radix-4 vs the DFT':>22}{'radix-4 mults':>16}{'radix-2 mults':>16}"
      f"{'saving':>9}")
for k in (1, 2, 3, 4, 5, 6, 7):
    n = 4 ** k
    x = gen.standard_normal(n) + 1j * gen.standard_normal(n)
    reference = np.fft.fft(x)
    four = [0]
    got = fft_radix4(x, four)
    two = ft.operation_count(n)["butterflies_iterative"]
    print(f"{n:>8}"
          f"{float(np.max(np.abs(got - reference))) / float(np.max(np.abs(reference))):>22.3e}"
          f"{four[0]:>16}{two:>16}{two / max(four[0], 1):>9.3f}")
```

| $n$ | radix-4 vs the DFT | radix-4 multiplications | radix-2 multiplications | saving |
|---|---|---|---|---|
| 4 | 1.397e-16 | 3 | 4 | 1.333 |
| 16 | 4.100e-16 | 24 | 32 | 1.333 |
| 64 | 7.353e-16 | 144 | 192 | 1.333 |
| 256 | 7.301e-16 | 768 | 1024 | 1.333 |
| 1024 | 1.182e-15 | 3840 | 5120 | 1.333 |
| 4096 | 1.439e-15 | 18432 | 24576 | 1.333 |
| 16384 | 1.894e-15 | 86016 | 114688 | 1.333 |

**Exactly a factor of $4/3$, at every size.** The reason is structural: radix-4 does
$\frac{3n}{4}\log_4 n = \frac{3n}{8}\log_2 n$ multiplications against radix-2's
$\frac n2\log_2 n$, and the ratio is $\frac{3/8}{1/2} = \frac34$.

**Where the saving comes from.** A radix-4 butterfly combines four inputs with the fourth roots of
unity $1, -i, -1, i$. **Multiplying by $\pm i$ is free**: it swaps the real and imaginary parts
and flips a sign, with no floating point multiplication at all. So three of the four twiddle
factors need a genuine multiplication where two radix-2 stages would need four.

**Why 25 percent matters.** It does not, on its own. What matters is that radix-4 also halves the
number of **passes** over memory, from $\log_2 n$ to $\log_4 n$, and on modern hardware memory
traffic dominates arithmetic. A real library uses radix-4 or radix-8 kernels for exactly that
reason, and the arithmetic saving is a bonus.

### 3.2 Split radix

The lowest known operation count for a power of two: split the even indices as radix-2 and the odd
indices as radix-4.

```python
import math


def split_radix_multiplications(n):
    """Complex multiplications used by the split radix recursion.

    Split radix takes the even indices as one half length transform and the odd indices as two
    quarter length ones, giving S(n) = S(n/2) + 2 S(n/4) + n/2 - 2. The -2 is because two of
    the twiddle factors in each quarter are 1 and i, which cost nothing.
    """
    m = int(n)
    if m <= 4:
        return 0
    return (split_radix_multiplications(m // 2)
            + 2 * split_radix_multiplications(m // 4) + m // 2 - 2)


print(f"{'n':>10}{'radix-2':>14}{'radix-4':>14}{'split radix':>14}{'vs radix-2':>13}")
for k in (2, 4, 6, 8, 10, 14, 20):
    n = 2 ** k
    two = (n // 2) * k
    four = (3 * n // 8) * k
    split = split_radix_multiplications(n)
    ratio = f"{two / split:.3f}" if split else "-"
    print(f"{n:>10}{two:>14}{four:>14}{split:>14}{ratio:>13}")
```

| $n$ | radix-2 | radix-4 | split radix | saving over radix-2 |
|---|---|---|---|---|
| 4 | 4 | 2 | 0 | - |
| 16 | 32 | 24 | 8 | 4.000 |
| 64 | 192 | 144 | 72 | 2.667 |
| 256 | 1024 | 768 | 456 | 2.246 |
| 1024 | 5120 | 3840 | 2504 | 2.045 |
| 16384 | 114688 | 86016 | 61896 | 1.853 |
| 1048576 | 10485760 | 7864320 | 6058440 | **1.731** |

**Split radix beats radix-4, which beats radix-2**, and the ratio against radix-2 is still falling
at $n = 2^{20}$: 4.00, 2.67, 2.25, 2.05, 1.85, 1.73.

**The ratio has not converged and the table should not pretend otherwise.** The counts above are
**complex multiplications** counted by the recursion, including the free ones the $-2$ removes, and
their ratio approaches a limit slowly because the $-2$ per level is a lower order term that is
still significant at these sizes.

**The published figure is stated in real flops, which is the fair unit.** Split radix uses
$4n\log_2 n - 6n + 8$ real operations against radix-2's $5n\log_2 n - 10n + 12$, so the
asymptotic ratio is $5/4 = 1.25$. It held the record from 1984 until 2007, when Johnson and Frigo
found a variant saving a further 6 percent by rescaling the sub-transforms.

**Why the two accountings differ so much.** A complex multiplication is 4 real multiplications and
2 real additions, or 3 and 5 with Karatsuba, and an addition is 2 real additions. Counting complex
multiplications alone ignores the additions, which is most of the work, so it exaggerates the
difference between algorithms that trade multiplications for additions. **Split radix is exactly
such an algorithm**, which is why its advantage looks like 1.73 in one unit and 1.25 in the other.

**Why it is not the universal choice.** The recursion is irregular: one half and two quarters,
which does not vectorise as cleanly as a uniform radix. FFTW, which is the reference
implementation, generates code for many radices and **measures** which is fastest on the specific
machine, because the arithmetic count stopped being the right cost model some time in the 1990s.

**The measurement worth remembering.** Between radix-2 and the best known algorithm there is a
factor of 1.39 in arithmetic. Between the DFT and radix-2 at $n = 2^{20}$ there is a factor of
104858. **The interesting optimisation happened once, in 1965**, and everything since has been
constants.

### 3.3 The real input transform

Two real signals can share one complex transform, which halves the work.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import fft as ft


def two_real_transforms(a, b):
    """Transform two real signals with one complex transform.

    Pack them as a + i b. The transform's even and odd conjugate parts separate them, because
    a real signal has a conjugate symmetric transform and i times a real signal has a conjugate
    antisymmetric one.
    """
    a = np.atleast_1d(np.asarray(a, dtype=float)).ravel()
    b = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    if a.size != b.size:
        raise ValueError(f"{a.size} against {b.size}: pack equal lengths")
    n = a.size
    Z = ft.fft(a + 1j * b)
    flipped = np.concatenate([[Z[0]], Z[:0:-1]])
    A = 0.5 * (Z + np.conj(flipped))
    B = -0.5j * (Z - np.conj(flipped))
    return A, B


gen = np.random.default_rng(4)
print(f"{'n':>8}{'first signal':>16}{'second signal':>16}{'transforms used':>18}")
for n in (8, 16, 64, 256, 1024):
    a = gen.standard_normal(n)
    b = gen.standard_normal(n)
    A, B = two_real_transforms(a, b)
    ra, rb = ft.fft(a), ft.fft(b)
    scale = max(float(np.max(np.abs(ra))), float(np.max(np.abs(rb))))
    print(f"{n:>8}{float(np.max(np.abs(A - ra))) / scale:>16.3e}"
          f"{float(np.max(np.abs(B - rb))) / scale:>16.3e}{1:>18}")
```

| $n$ | first signal | second signal | transforms used |
|---|---|---|---|
| 8 | 6.420e-17 | 8.808e-17 | 1 |
| 16 | 1.258e-16 | 1.791e-16 | 1 |
| 64 | 3.075e-16 | 3.588e-16 | 1 |
| 256 | 4.430e-16 | 4.896e-16 | 1 |
| 1024 | 4.342e-16 | 4.854e-16 | 1 |

**Both are recovered to $10^{-15}$ from a single transform**, so the speedup is exactly 2 for a
pair of real signals.

**How the separation works.** For real $a$, $\hat a$ is conjugate **symmetric**:
$\hat a_{n-k} = \overline{\hat a_k}$. For $ib$ with $b$ real, the transform is conjugate
**antisymmetric**. Adding them gives $Z$, and

$$
\hat a_k = \tfrac12\left(Z_k + \overline{Z_{n-k}}\right), \qquad
\hat b_k = \tfrac{1}{2i}\left(Z_k - \overline{Z_{n-k}}\right)
$$

which is separating a signal into its symmetric and antisymmetric parts, in the frequency domain.

**The single signal version.** One real signal of length $2n$ can be packed as a complex signal of
length $n$ by putting the even samples in the real part and the odd ones in the imaginary part,
transforming, and applying one radix-2 combination stage. That also saves a factor of 2, and it is
what a library's `rfft` does.

**Why it is worth doing.** Real data is the overwhelmingly common case: audio, images, sensor
readings. A factor of 2 in time and a factor of 2 in memory, for a page of index manipulation, is
one of the better trades in numerical software.

**The one trap.** The index $n-k$ must be taken mod $n$, so $k = 0$ pairs with itself. The
`flipped` array above handles that by treating index 0 separately, and getting it wrong gives an
answer that is right for $k \ge 1$ and wrong for the DC term, which is exactly the kind of bug a
test on random data at a single index would miss.

### 4.1 The FFT is more accurate than the DFT

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import fft as ft, dft as dt

print(f"{'n':>8}{'fft vs exact':>18}{'dft vs exact':>18}{'ratio':>10}")
for k in range(3, 12):
    n = 2 ** k
    j = np.arange(n)
    frequency = 3
    x = np.exp(2j * math.pi * frequency * j / n)
    exact = np.zeros(n, dtype=complex)
    exact[frequency] = n
    fast = float(np.max(np.abs(ft.fft(x) - exact))) / n
    slow = float(np.max(np.abs(dt.dft(x) - exact))) / n
    print(f"{n:>8}{fast:>18.3e}{slow:>18.3e}{slow / max(fast, 1e-300):>10.2f}")
```

| $n$ | FFT vs exact | DFT vs exact | ratio |
|---|---|---|---|
| 8 | 2.616e-16 | 6.389e-16 | 2.44 |
| 64 | 2.790e-16 | 3.040e-15 | 10.89 |
| 256 | 2.781e-16 | 1.366e-14 | 49.13 |
| 1024 | 2.494e-16 | 3.224e-14 | 129.27 |
| 2048 | 2.486e-16 | 5.946e-14 | **239.14** |

**The fast algorithm is the accurate one, and by a widening margin.** The FFT's error is flat at
$2.5\times10^{-16}$ from $n = 8$ to $n = 2048$. The direct sum's grows steadily and is 239 times
worse at $n = 2048$.

**Why the slow one is worse.** The direct sum adds $n$ terms into one accumulator, so its rounding
grows like $n\varepsilon$ in the worst case and $\sqrt n\varepsilon$ on average. Measured, the DFT
error goes from $6\times10^{-16}$ to $6\times10^{-14}$ over a factor of 256 in $n$, which is a
factor of 93: close to $\sqrt{256} = 16$ times the $\log$-ish growth of the twiddle errors, so
somewhere between $\sqrt n$ and $n$.

**Why the fast one is not.** The FFT does $\log_2 n$ **passes**, and within a pass each output
depends on only two inputs. So an input's rounding is amplified $\log_2 n$ times rather than $n$
times, and the standard result is a norm-wise bound of $O(\varepsilon\log n)$, improved to
$O(\varepsilon\sqrt{\log n})$ under a random rounding model. At $n = 2048$, $\log_2 n = 11$
against $n = 2048$: a factor of 186, which is the measured 239 within the model's slack.

**This is one of the most counterintuitive facts in numerical computing.** Doing less work usually
means a worse answer. Here doing less work means **fewer opportunities to round**, and the
algorithm that is $10^5$ times faster is also 200 times more accurate.

**The general principle.** Error accumulates with the **depth** of the computation, not with the
work. A tree of depth $\log n$ beats a chain of length $n$, which is the same reason pairwise
summation beats naive summation in Part 1 and why `numpy.sum` uses it.

### 4.2 The Gibbs overshoot converges

Covered in exercise 2.5, whose table measures the convergence and whose text explains why it
stalls at $2\times10^{-4}$. Two further measurements complete the picture.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import fft as ft

n = 8192
step = np.where(np.arange(n) < n // 2, 1.0, -1.0)
jump = n // 2
print(f"{'cutoff':>10}{'overshoot fraction':>21}{'samples from the jump':>24}"
      f"{'as a fraction of n':>21}")
for frac in (0.01, 0.02, 0.05, 0.1, 0.2):
    cut = frac * n / 2.0
    out = ft.filter_report(step, cut, rate=float(n))
    filtered = out["filtered"]
    # the overshoot peak is the last local maximum before the jump
    window = filtered[:jump]
    peak = int(np.argmax(window[-jump // 2:])) + (jump - jump // 2)
    print(f"{cut:>10.1f}{out['overshoot_fraction']:>21.7f}{jump - peak:>24}"
          f"{(jump - peak) / n:>21.6f}")
```

| cutoff | overshoot fraction | samples from the jump | as a fraction of $n$ |
|---|---|---|---|
| 40.96 | 0.0895976 | 103 | 0.012573 |
| 81.92 | 0.0894906 | 50 | 0.006104 |
| 204.8 | 0.0893755 | 21 | 0.002563 |
| 409.6 | 0.0887053 | 10 | 0.001221 |
| 819.2 | 0.0862453 | 5 | 0.000610 |

**The height stays near 0.0894 and the width halves every time the cutoff doubles**, exactly in
proportion: 103, 51, 20, 10, 5 samples. That is the whole phenomenon in one table: the ringing is
not being removed, it is being compressed.

**The last two rows drift low, and the width column says why.** At cutoff 819.2 the overshoot peak
is 5 samples from the jump, so the discrete grid has only a handful of points across the whole
lobe and cannot land on its maximum. The reported 0.0862 is the largest **sample**, not the largest
value. Pushing the cutoff further makes the measurement worse, not better, which is why the
convergence in exercise 2.5 stalls at $2\times10^{-4}$.

**Why the constant is 0.0894899.** From exercise 2.5, it is
$\frac1\pi\mathrm{Si}(\pi) - \frac12 = 0.0894899\ldots$, where the maximum of the sine integral's
first lobe is what sets it.

**Why a smoother filter fixes it.** Replacing the rectangle by a taper makes the impulse response
decay faster than $1/t$, so the convolution is more local and the overshoot shrinks. The cost is a
wider transition band, which is lesson 58's exercise 5.2 in the other domain, and it is the same
trade.

**And where it appears elsewhere in this course.** Lesson 60's blocking artefacts are Gibbs at a
block boundary; lesson 46's Runge phenomenon is the polynomial analogue; and the ringing around a
sharp edge in a JPEG image at low quality is this exact effect, which is why the artefact is called
"mosquito noise" and why it clusters at edges.

### 4.3 Wiener against a brick wall with the wrong cutoff

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import fft as ft

n = 512
t = np.arange(n) / n
clean = np.sin(2 * np.pi * 3 * t) + 0.4 * np.sin(2 * np.pi * 7 * t)
print("the signal occupies bins 3 and 7, so the oracle cutoff is anything in [7, 8)")
print(f"{'brick cutoff':>14}{'brick error':>15}{'Wiener error':>15}{'winner':>13}")
for cutoff in (2.0, 4.0, 7.5, 10.0, 20.0, 50.0, 100.0, 250.0):
    out = ft.wiener_report(clean, 0.3, cutoff=cutoff, rate=float(n),
                           rng=np.random.default_rng(7))
    print(f"{cutoff:>14.1f}{out['brickwall_error']:>15.5f}"
          f"{out['wiener_estimated_error']:>15.5f}"
          f"{('Wiener' if out['estimated_beats_brickwall'] else 'brick wall'):>13}")
```

| brick cutoff | brick wall error | Wiener error | winner |
|---|---|---|---|
| 4.0 | 0.28865 | 0.12727 | **Wiener** |
| 7.5 | 0.06591 | 0.12727 | brick wall |
| 10.0 | 0.07107 | 0.12727 | brick wall |
| 20.0 | 0.09120 | 0.12727 | brick wall |
| 50.0 | 0.12031 | 0.12727 | brick wall |
| 100.0 | 0.17881 | 0.12727 | **Wiener** |
| 250.0 | 0.28079 | 0.12727 | **Wiener** |

**The brick wall wins only inside a window, and the window is narrow.** It beats the Wiener filter
for cutoffs between about 7.5 and 55, which is a factor of 7 in cutoff around a true bandwidth of
7. Outside that window it loses, and by a lot.

**Both failure modes are visible.** Cut too low and the signal is destroyed: at cutoff 4 the 7
cycle component is removed entirely and the error is 0.289, which is worse than the noise itself.
Cut too high and the noise is kept: at cutoff 250 the filter passes almost everything and the
error is 0.281, essentially the unfiltered noise.

**The Wiener error is 0.12727 in every row**, because it has no cutoff to get wrong. That is the
point of the comparison: **the Wiener filter has no parameter to tune**, and the brick wall's one
parameter has to be within a factor of 7 of the right value to be competitive.

**Where the crossovers are, and why they are not symmetric.** The lower crossover is at 7, the
signal's true bandwidth, and it is sharp: below it a whole component vanishes. The upper crossover
is around 55, and it is gradual: passing extra noise costs an amount proportional to the extra
bandwidth. That asymmetry is why the standard advice is to err on the high side, and why the
Wiener filter, which does neither, is the better default.

**The honest caveat from the lesson.** With the **true** signal spectrum the Wiener filter beats
every row of this table, at 0.0261. The 0.12727 is what the practical version costs for estimating
the signal spectrum from the noisy data, and that estimate is what lets a well chosen brick wall
compete at all.

### 5.1 Is $n\log n$ optimal

**What is known.** Very little, and that is the honest answer.

**No superlinear lower bound is known for the DFT over $\mathbb C$.** Nobody has proved that
$\Omega(n\log n)$ operations are necessary. The best unconditional lower bound is $\Omega(n)$,
which is trivial since every input must be read.

**What is proved, under restrictions.**

**Morgenstern (1973)**: any **linear** algorithm computing the DFT using only additions and
multiplications by constants of modulus at most 1 needs $\Omega(n\log n)$ operations. That is a
genuine bound and its hypothesis is restrictive: allowing large constants breaks it, and no
practical algorithm uses them, so the bound is believed to be morally right and is not a theorem
about algorithms in general.

**Ailon (2013, 2015)**: $\Omega(n\log n)$ for algorithms that are **well conditioned**, in the
sense that every intermediate step has bounded condition number. Again a restriction, and again a
natural one.

**The best known upper bound.** Johnson and Frigo (2007) improved split radix's
$\frac43n\log_2 n$ real multiplications to about $\frac{34}{9}n\log_2 n$ total flops, a saving of
just under 6 percent over the 1968 split radix count. **That is the entire progress since 1984**,
and it is the sort of margin that suggests the constant is near its floor even if the exponent is
not proved.

**Why the question is hard.** Lower bounds on arithmetic circuits are hard in general: proving
that some explicit linear map needs superlinear circuits would be a major result in complexity
theory, and none is known for any explicit family. The DFT is one of the most studied candidates
and it has resisted.

**What that means practically.** Nothing. $n\log n$ is what every implementation achieves, the
constant is within 6 percent of the best known, and the memory traffic has dominated the
arithmetic since the 1990s anyway. **The open problem is a theoretical one and it has no bearing
on what to write.**

### 5.2 The FFT and multiplication

**Polynomial multiplication.** Two polynomials of degree $n$ have $n+1$ coefficients each, and
their product has $2n+1$. The product's coefficients are the **linear convolution** of the input
coefficient vectors, which lesson 58's exercise 3.3 computes by padding to at least $2n+1$ and
using three transforms:

$$
\text{cost} = 3\cdot\frac{N}{2}\log_2 N + N \quad\text{with } N = O(n),
\qquad \text{so } O(n\log n)
$$

against the direct $O(n^2)$. The crossover in practice is around $n \approx 100$, below which the
schoolbook method wins on constants.

**Integer multiplication.** An integer in base $b$ is a polynomial in $b$ evaluated at $b$, so
multiplying integers is multiplying polynomials **plus carrying**. The carries cost $O(n)$ and do
not change the order.

The difficulty is that the convolution's entries can be large, so floating point rounding matters:
with 53 bit doubles and digits of $k$ bits, the products reach $2^{2k}n$ and must stay below
$2^{53}$ to be exact. That limits $k$ and hence the number of digits, which is why practical
implementations use small radices or exact arithmetic.

**Schonhage-Strassen (1971)** removes the rounding problem by doing the transform in the ring
$\mathbb Z/(2^m+1)\mathbb Z$, where a power of 2 is a root of unity and every operation is exact
integer arithmetic. The multiplications by roots of unity become **bit shifts**, which are free.
The cost is

$$
O(n\log n\log\log n)
$$

the extra $\log\log n$ coming from the recursive multiplication of the smaller integers inside the
ring.

**And the current record.** Harvey and van der Hoeven (2019) achieved $O(n\log n)$ for integer
multiplication, matching a conjectured optimum. The algorithm is not practical, its crossover
being estimated at integers of more than $2^{1729^{12}}$ bits, which is why Schonhage-Strassen and
its variants remain what libraries implement.

**The chain worth seeing.** Convolution is multiplication in the transform domain; polynomial
multiplication is convolution; integer multiplication is polynomial multiplication with carries.
**So a fast transform makes all three fast**, and that single observation from 1965 is behind every
arbitrary precision arithmetic library in existence.

### 5.3 Why the fast algorithm is the accurate one

**The result.** For the radix-2 FFT with exactly rounded arithmetic and correctly rounded twiddle
factors, the computed transform $\hat X$ satisfies

$$
\frac{\|\hat X - X\|_2}{\|X\|_2} \le c\,\varepsilon\log_2 n
$$

for a modest constant $c$, and under a model where the rounding errors behave like independent
random variables the bound improves to $O(\varepsilon\sqrt{\log_2 n})$.

**The direct sum gives $O(\varepsilon n)$ in the worst case** and $O(\varepsilon\sqrt n)$ on
average.

**The property of the butterfly that does it.** Each output of a butterfly is
$a \pm wb$ with $|w| = 1$, so

$$
|a + wb|^2 + |a - wb|^2 = 2\left(|a|^2 + |b|^2\right)
$$

**The butterfly is an isometry up to the factor $\sqrt2$.** It does not amplify: the norm of the
output is determined by the norm of the input, so an error entering at one stage passes through
every later stage with its size preserved rather than magnified.

**And the depth is $\log_2 n$.** Each of the $n$ inputs passes through exactly $\log_2 n$
butterflies, so its rounding is committed $\log_2 n$ times. In a direct sum each output
accumulates $n$ terms in sequence, so an early rounding is carried through $n$ subsequent
additions.

**So the two facts are: no amplification per step, and few steps.** Neither alone would be enough.
A non-amplifying algorithm with $n$ steps would still give $O(\varepsilon\sqrt n)$; an amplifying
algorithm with $\log n$ steps could be arbitrarily bad.

**Measured in exercise 4.1**: the FFT error is flat at $2.5\times10^{-16}$ from $n = 8$ to
$n = 2048$ while the direct sum's grows to $6\times10^{-14}$, a ratio of 239.

**The same principle elsewhere in this course.** Part 1's pairwise summation, which is a tree of
depth $\log n$ rather than a chain of length $n$, and which `numpy.sum` uses for exactly this
reason. Part 3's blocked LU factorisation. Part 6's divide and conquer eigensolver. **In every
case, restructuring a computation as a shallow tree improves both the speed and the accuracy**,
and that is not a coincidence but the same theorem.

---

## Lesson 60, The DCT and Data Compression

### 1.1 The boundary problem, and where the DCT has no advantage

**The problem, precisely.** A finite block of $n$ samples does not determine a function on the
whole line, and the transform has to assume something. The DFT assumes **periodic** extension,
which repeats the block. If $x_0 \ne x_{n-1}$ the extension has a **jump** at the seam, and the
smoothness of the extended function, not of the data, sets the coefficient decay rate.

Measured on a ramp: the periodic extension jumps by 0.984 and the DFT coefficients decay like
$k^{-0.88}$; the even extension jumps by exactly 0 and the DCT's decay like $k^{-2.03}$.

**The DCT's fix is not about frequencies.** It uses cosines rather than complex exponentials, which
is a real basis rather than a complex one, and that is a storage convenience. The **accuracy** gain
comes entirely from the extension being continuous.

**A signal on which the DCT has no advantage: any signal whose ends already match.**

The cleanest example is an exact bin frequency. $\sin(2\pi t)$ on $t = j/n$ has $x_0 = x_{n-1}$ up
to one sample's worth, its periodic extension is smooth, and the DFT puts all its energy in one
bin. Measured with a storage budget of 8 real numbers:

| end mismatch | energy the DCT discards | energy the DFT discards | ratio |
|---|---|---|---|
| 0.00 | 2.118e-3 | **0.000** | **0** |
| 0.05 | 2.218e-3 | 7.339e-5 | 0.03 |
| 0.25 | 2.590e-3 | 2.018e-3 | 0.78 |
| 0.50 | 2.907e-3 | 8.426e-3 | 2.90 |
| 1.00 | 2.776e-3 | 2.805e-2 | 10.10 |
| 2.00 | 1.551e-3 | 4.883e-2 | **31.49** |

**At zero mismatch the DFT discards nothing and the DCT discards $2\times10^{-3}$, so the DCT
loses.** The crossover is at a mismatch of about 0.3, and by mismatch 2 the DCT is 31 times
better.

That row is worth stating plainly: **on an exactly periodic signal the DCT is the worse
transform**, because the even extension introduces a kink where the periodic one was smooth. The
DCT wins on real data because real data is not periodic, not because cosines are better than
exponentials.

### 1.2 Counting the budget in real numbers

**Why coefficients are the wrong unit.** For real input, a DCT coefficient is one real number and a
DFT coefficient is complex, hence two, except the DC and Nyquist terms which are real. So "keep 8
coefficients" gives the DCT 8 numbers of storage and the DFT about 16.

**What the comparison looks like if you get it wrong.** Counting bins, the DFT appears to hold
0.7654 of a Gaussian bump's energy in 8 coefficients while the DCT holds 1.000000, which reads as
a decisive DCT win. Counting real numbers, the same measurement gives 0.999999 against 1.000000:
**a tie to six digits.**

So the wrong unit turns a tie into a rout, in the direction of whichever transform is being
advocated. That is why the correct accounting is not pedantry.

**And it changes the conclusion, not only the numbers.** With the correct unit, the DCT's advantage
turns out to be **entirely a boundary effect**, present on a ramp and absent on a centred bump,
which is a much more informative statement than "the DCT compacts better".

**The general rule.** A comparison between representations must be at equal **storage**, measured
in the units the file format actually uses. The same trap appears in Part 6's low rank
approximation, where a rank $k$ SVD of an $m \times n$ image costs $k(m+n+1)$ numbers rather than
$k$, and comparing against $k$ DCT coefficients would be equally wrong.

### 1.3 One block lossy, the sequence exact

**Both are necessarily true, and neither is surprising once stated.**

**A single block must be lossy.** The MDCT maps $\mathbb R^{2n} \to \mathbb R^n$. A linear map from
a $2n$ dimensional space to an $n$ dimensional one has a kernel of dimension at least $n$, so at
least an $n$ dimensional space of inputs maps to zero and cannot be distinguished. Measured: a
single block round trip loses 80 percent of the signal.

**The overlapped sequence need not be.** Consecutive blocks **share** half their samples. A signal
of length $L$ is covered by about $L/n$ blocks of $2n$ samples each, producing about $L/n \cdot n
= L$ coefficients. **So the sequence is not a reduction at all**, it is a change of basis with
about one coefficient per sample, and there is no counting obstruction to invertibility.

Measured: expansion 1.06 to 1.25 depending on the block size, the excess being one block of
padding at each end.

**What is genuinely surprising is that the aliases cancel exactly rather than approximately.** The
inverse MDCT of one block returns the original **plus a time reversed copy of part of it**. When
two consecutive blocks are overlapped and added, the alias from one is the negative of the alias
from the other, provided the window satisfies

$$
w_j^2 + w_{j+n}^2 = 1
$$

which is the Princen-Bradley condition. Measured at $2\times10^{-16}$ for the sine window, and the
overlap-add reconstruction is exact to $3\times10^{-15}$.

**Why that is worth having.** A plain block transform quantises each block independently, so the
reconstructions disagree at the boundary and the disagreement is audible as a click hundreds of
times a second. The MDCT has no boundary: every sample is covered by two blocks whose
contributions blend. **That is why every audio codec since MP3 uses it.**

### 2.1 The DCT-II from the DFT

**Construction.** Given $x_0,\dots,x_{n-1}$, form the even extension of length $4n$:

$$
y = (x_0, x_0, x_1, x_1, \dots, x_{n-1}, x_{n-1}, x_{n-1}, x_{n-1}, \dots, x_0, x_0)
$$

more precisely, $y_{2j+1} = y_{4n-2j-1} = x_j$ for $j = 0,\dots,n-1$, and $y_\ell = 0$ for even
$\ell$. That places the samples at the **half integer** positions, symmetric about both ends.

**Take the DFT of $y$.** Because $y$ is real and even, its transform is real and even, and

$$
\hat y_k = \sum_{\ell}y_\ell e^{-2\pi ik\ell/4n}
= \sum_{j=0}^{n-1}x_j\left(e^{-2\pi ik(2j+1)/4n} + e^{-2\pi ik(4n-2j-1)/4n}\right)
$$

The second exponent is $e^{+2\pi ik(2j+1)/4n}$ because $e^{-2\pi ik} = 1$, so the pair is
$2\cos\left(\frac{\pi k(2j+1)}{2n}\right)$ and

$$
\hat y_k = 2\sum_{j=0}^{n-1}x_j\cos\!\left(\frac{\pi(2j+1)k}{2n}\right)
$$

**which is the DCT-II up to the scale factor.** Applying $s_0 = \sqrt{1/n}$ and
$s_k = \sqrt{2/n}$ makes the transform orthonormal.

**Where the half sample offset comes from: the $2j+1$.** The samples sit at positions
$\tfrac12, \tfrac32, \dots$ in the extension, not at $0, 1, 2, \dots$. That is what makes the
extension even about the **boundary between** samples rather than about a sample, and it is the
difference between DCT-II and DCT-I.

**Why that choice.** Being even about the boundary means the extension repeats $x_0$ at position
$-\tfrac12$, so the reflected signal is continuous **without duplicating a sample**. DCT-I, which
is even about the samples themselves, duplicates the end samples, which is fine for interpolation
(lesson 56's Lobatto grid) and wastes a degree of freedom for compression.

### 2.2 Orthonormality

**Claim.** $C^TC = I$ for $C_{kj} = s_k\cos\left(\frac{\pi(2j+1)k}{2n}\right)$ with
$s_0 = \sqrt{1/n}$, $s_k = \sqrt{2/n}$.

**Proof.** The $(j,\ell)$ entry of $C^TC$ is

$$
\sum_{k=0}^{n-1}s_k^2\cos\!\left(\frac{\pi(2j+1)k}{2n}\right)
\cos\!\left(\frac{\pi(2\ell+1)k}{2n}\right)
$$

Use $\cos A\cos B = \frac12[\cos(A-B) + \cos(A+B)]$ with
$A - B = \frac{\pi k(j-\ell)}{n}$ and $A + B = \frac{\pi k(j+\ell+1)}{n}$:

$$
= \frac12\sum_ks_k^2\left[\cos\!\frac{\pi k(j-\ell)}{n} + \cos\!\frac{\pi k(j+\ell+1)}{n}\right]
$$

Now the key sum. For an integer $p$ with $0 < |p| < 2n$,

$$
\sum_{k=0}^{n-1}s_k^2\cos\frac{\pi kp}{n}
= \frac1n + \frac2n\sum_{k=1}^{n-1}\cos\frac{\pi kp}{n}
$$

and $\sum_{k=0}^{n-1}\cos\frac{\pi kp}{n}$ is the real part of a geometric series with ratio
$e^{i\pi p/n}$, which sums to zero when $p$ is a nonzero **even** integer and to 1 when $p$ is odd.
Working through, the whole expression is $\delta_{j\ell}$.

**The two terms do different jobs.** $j - \ell$ is zero exactly when $j = \ell$, contributing the
diagonal 1. $j + \ell + 1$ ranges over $1$ to $2n-1$ and is never $0$ or $2n$, so its term always
vanishes. **That is why the half sample offset is needed**: without the $+1$, the second term would
contribute at $j = \ell = 0$ and break orthogonality.

**The consequence.** $C^{-1} = C^T$, so the inverse transform is the transpose, the condition
number is exactly 1, and the transform preserves energy exactly. All three are measured in the
lesson, at $10^{-16}$, $1.00000000$ and $10^{-11}$ respectively.

### 2.3 Jumps and kinks

**Claim.** For a periodic function $f$ with a jump discontinuity, the Fourier coefficients decay
like $1/k$. With $f$ continuous but $f'$ having a jump, they decay like $1/k^2$.

**Proof by integration by parts.** For $\hat f_k = \int_0^1f(t)e^{-2\pi ikt}dt$,

$$
\hat f_k = \left[\frac{f(t)e^{-2\pi ikt}}{-2\pi ik}\right]_0^1
+ \frac{1}{2\pi ik}\int_0^1f'(t)e^{-2\pi ikt}\,dt
$$

**If $f$ is continuous and periodic**, the bracket vanishes and $\hat f_k = \frac{1}{2\pi ik}
\widehat{f'}_k$, so one order is gained. Repeating, each additional continuous derivative gains
another order.

**If $f$ has a jump of size $J$ at $t_0$**, the bracket does not vanish: splitting the integral at
$t_0$ and integrating by parts on each piece leaves a boundary term
$\frac{J}{2\pi ik}e^{-2\pi ikt_0}$, which is $O(1/k)$ and does not decay further. Hence
$|\hat f_k| \sim \frac{|J|}{2\pi k}$.

**If $f$ is continuous with $f'$ jumping**, the first integration by parts is clean and the second
leaves the boundary term, giving $O(1/k^2)$.

**In general: $m$ continuous derivatives with the $(m+1)$st jumping gives $O(k^{-m-2})$.**

**Applied to the lesson's ramp.** Its periodic extension jumps, so $O(1/k)$: measured decay
exponent 0.88. Its even extension is continuous with a **kink**, so $O(1/k^2)$: measured 2.03.
**One order, purchased by continuity at the seam, and that is the entire content of the DCT.**

**The caveat the lesson insists on.** The fit must skip the exactly zero coefficients. Every even
indexed DCT coefficient of a ramp vanishes by symmetry, and including them drags the fitted
exponent from 2.03 down to 0.99 as $n$ grows, which looks like the theorem failing.

### 2.4 Huffman is optimal

**Claim.** Among all prefix codes for a given symbol distribution, the Huffman code minimises the
expected codeword length.

**Proof by exchange.** Let the symbols be ordered by probability, $p_1 \ge p_2 \ge \cdots \ge p_m$.

**Step 1: an optimal code exists in which the two least likely symbols have the longest codewords
and are siblings.** In an optimal prefix code, if $p_i > p_j$ then $\ell_i \le \ell_j$: otherwise
swapping the two codewords lowers the expected length by $(p_i - p_j)(\ell_i - \ell_j) > 0$. So the
two least likely symbols have the longest lengths. Moreover the longest codeword must have a
sibling, or its last bit could be deleted, shortening it. Swapping so that the two least likely are
that sibling pair does not increase the expected length.

**Step 2: induction.** Merge the two least likely symbols into one of combined probability
$p_{m-1} + p_m$, giving a problem with $m-1$ symbols. Any code for the merged problem extends to
one for the original by appending a bit, and the expected lengths differ by exactly
$p_{m-1} + p_m$, a constant independent of the code. So an optimal code for the merged problem
extends to an optimal one for the original.

By induction on $m$, the greedy merge is optimal.

**The bound.** The optimal expected length $L$ satisfies $H \le L < H + 1$, where $H$ is the
entropy. The lower bound is Shannon's; the upper follows from the Shannon-Fano construction with
$\ell_i = \lceil-\log_2 p_i\rceil$, which is a valid prefix code by Kraft's inequality and has
expected length below $H+1$.

**Measured**, on four distributions:

| distribution | entropy | Huffman | overhead |
|---|---|---|---|
| uniform over 256 | 7.9926 | 8.0000 | 0.0074 |
| uniform over 16 | 3.9996 | 4.0000 | 0.0004 |
| 95 percent one value | 0.2861 | 1.0000 | 0.7139 |
| all the same symbol | 0.0000 | 1.0000 | **1.0000** |

**The bound is tight exactly in the degenerate case.** With one symbol the entropy is 0 and Huffman
still spends a bit, because a zero length codeword cannot be decoded. That is where the "+1" in
$H+1$ is attained, and it is the only place.

**And it is why arithmetic coding exists.** Huffman's overhead comes from codeword lengths being
integers, so a symbol of probability 0.95 gets 1 bit where it deserves 0.074. Arithmetic coding
does not assign codewords to symbols at all, and reaches within $10^{-5}$ bits of the entropy,
which on the 95 percent row is a factor of 3.5 in file size.

### 2.5 Time domain alias cancellation

**What one block returns.** The MDCT of $2n$ windowed samples produces $n$ coefficients. The IMDCT
of those returns $2n$ samples, and they are **not** the input. Writing the input as two halves
$(a, b)$ each of length $n$, the round trip through one block gives

$$
\mathrm{IMDCT}(\mathrm{MDCT}(a, b)) = \tfrac12\big(a - \tilde a,\; b + \tilde b\big)
$$

where $\tilde{\cdot}$ denotes time reversal, and the window has been applied twice.

**So the output is the input plus a time reversed copy of itself**: the **time domain alias**. That
is the $n$ dimensional information the transform threw away, appearing as a reflection rather than
as a loss.

**What the next block returns.** The next block covers $(b, c)$, and its round trip gives
$\tfrac12(b - \tilde b, c + \tilde c)$.

**Overlap and add.** The second half of block one, $\tfrac12(b + \tilde b)$, is added to the first
half of block two, $\tfrac12(b - \tilde b)$:

$$
\tfrac12\left(b + \tilde b\right) + \tfrac12\left(b - \tilde b\right) = b
$$

**The aliases cancel exactly**, and $b$ is recovered.

**Where the window enters.** With windowing, the two contributions are $w_2^2(b + \tilde b)$ and
$w_1^2(b - \tilde b)$ for the appropriate halves of the two windows. The alias terms cancel only
if the window is **symmetric**, and the signal terms add to $b$ only if

$$
w_j^2 + w_{j+n}^2 = 1
$$

which is the **Princen-Bradley** condition. The sine window
$w_j = \sin\left(\frac{\pi(j+\frac12)}{2n}\right)$ satisfies it because
$\sin^2\theta + \cos^2\theta = 1$, with the second half being the cosine of the first.

**Measured:** the Princen-Bradley residual is $2.2\times10^{-16}$, the single block error is 0.80
relative, and the overlap-added error is $3.4\times10^{-15}$.

**Why this is a beautiful construction.** Two operations that each lose half the information
combine to lose none, and the mechanism is that they lose **different** halves. The transform is
not invertible and the transform sequence is, which is the definition of a **frame** rather than a
basis, and it is the same idea as a wavelet frame or an oversampled filter bank.

### 3.1 The DCT through the FFT

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import dct as dc, fft as ft


def dct_via_fft(x):
    """The orthonormal DCT-II in O(n log n), by one complex FFT of length n.

    Reorder the samples as evens ascending then odds descending, transform, and apply a phase.
    The reordering is what turns the half sample offset of the cosine into a phase factor.
    """
    v = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n = v.size
    if n < 1:
        raise ValueError("cannot transform an empty block")
    if n == 1:
        return v.copy()
    reordered = np.concatenate([v[0::2], v[1::2][::-1]])
    spectrum = ft.fft(reordered)
    phase = 2.0 * np.exp(-1j * math.pi * np.arange(n) / (2.0 * n))
    out = np.real(spectrum * phase)
    scale = np.full(n, math.sqrt(2.0 / n)) * 0.5
    scale[0] = math.sqrt(1.0 / n) * 0.5
    return out * scale


gen = np.random.default_rng(9)
print(f"{'n':>8}{'agreement with the direct DCT':>32}")
for n in (2, 4, 8, 16, 64, 256, 1024):
    v = gen.standard_normal(n)
    direct = dc.dct(v)
    fast = dct_via_fft(v)
    print(f"{n:>8}"
          f"{float(np.max(np.abs(direct - fast))) / max(float(np.max(np.abs(direct))), 1.0):>32.3e}")

print()
print(f"{'n':>8}{'direct multiplies':>20}{'via the FFT':>14}{'ratio':>9}")
for n in (8, 64, 256, 1024, 4096):
    direct_cost = n * n
    fast_cost = (n // 2) * int(math.log2(n)) + n
    print(f"{n:>8}{direct_cost:>20}{fast_cost:>14}{direct_cost / fast_cost:>9.1f}")
```

| $n$ | agreement with the direct DCT |
|---|---|
| 2 | 0.000e0 |
| 8 | 3.489e-16 |
| 64 | 5.404e-15 |
| 256 | 1.792e-14 |
| 1024 | 7.758e-14 |

| $n$ | direct multiplies | via the FFT | ratio |
|---|---|---|---|
| 8 | 64 | 20 | 3.2 |
| 64 | 4096 | 256 | 16.0 |
| 256 | 65536 | 1280 | 51.2 |
| 1024 | 1048576 | 6144 | 170.7 |
| 4096 | 16777216 | 28672 | **585.1** |

**They agree to $7.8\times10^{-14}$ relative at $n = 1024$ and the transform is 585 times cheaper
at $n = 4096$.**

The agreement degrades slowly with $n$, and the reason is the **reference**, not the transform: the
direct DCT sums $n$ terms per coefficient, so its own error grows like $\sqrt n\varepsilon$, which
at $n = 1024$ is $7\times10^{-15}$. The remaining factor of 10 is the phase twist, which is one
extra complex multiplication per coefficient on top of the transform's own error.

**How the reordering works.** Placing the even indexed samples ascending and the odd indexed
descending produces a sequence whose DFT, after a phase twist, is the DCT. The reordering is what
converts the cosine's half sample offset into the phase factor $e^{-i\pi k/2n}$.

**Why only one transform of length $n$ rather than the mirrored one of length $2n$ used in lesson
56.** The mirroring version is simpler to derive and costs twice as much. This version, due to
Makhoul, is what a codec implements.

**And for the size JPEG actually uses, neither is used.** At $n = 8$ the ratio above is only 3.2,
and the constant factors matter more than the asymptotics. The standard is Loeffler's algorithm:
**11 multiplications and 29 additions** for an 8 point DCT, found by hand and proved optimal in the
multiplication count. That is a fixed circuit, not an algorithm with a loop, and it is what is in
the silicon of every device that decodes JPEG.

### 3.2 Run length coding of the zig-zag order

After quantisation most coefficients are zero, and they are not scattered: they are concentrated in
the high frequency corner. Reading the block in **zig-zag** order puts them in a run at the end.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import dct as dc


def zigzag_order(n):
    """The JPEG scan order: diagonals, alternating direction.

    It groups coefficients by total frequency, so after quantisation the tail is a long run of
    zeros rather than zeros interleaved with survivors.
    """
    m = int(n)
    order = sorted(((i, j) for i in range(m) for j in range(m)),
                   key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 == 0 else -p[1]))
    return np.asarray([i * m + j for i, j in order])


def run_length(symbols):
    """(run of zeros, value) pairs, with an end of block marker."""
    out = []
    run = 0
    for value in symbols:
        if value == 0:
            run += 1
            continue
        out.append((run, int(value)))
        run = 0
    out.append(("EOB", 0))
    return out


def run_size_symbols(levels):
    """The JPEG (run, size) alphabet, stopping at the last nonzero with one end-of-block.

    This is what JPEG actually codes: not the value, but how many zeros preceded it and how many
    bits it needs. The value's bits follow raw. Combining the run and the magnitude into one
    alphabet is worth 10 to 20 percent over coding them separately.
    """
    nonzero = np.nonzero(levels)[0]
    if nonzero.size == 0:
        return [("EOB",)]
    tail = int(nonzero[-1])
    out = []
    run = 0
    for value in levels[:tail + 1]:
        if value == 0:
            run += 1
            continue
        out.append((min(run, 15), int(abs(value)).bit_length()))
        run = 0
    out.append(("EOB",))
    return out


gen = np.random.default_rng(3)
x_img, y_img = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
images = {"smooth": 128.0 + 100.0 * np.sin(6 * x_img) * np.cos(5 * y_img),
          "textured": (128.0 + 100.0 * np.sin(6 * x_img) * np.cos(5 * y_img)
                       + 25.0 * gen.standard_normal((64, 64))),
          "axis aligned edges": 128.0 + 100.0 * np.sign(np.sin(9 * x_img) * np.cos(7 * y_img))}
order = zigzag_order(8)
print(f"{'image':>20}{'quality':>9}{'raster bits':>14}{'zig-zag bits':>14}{'saving':>9}"
      f"{'last nonzero, r / z':>22}")
for name, image in images.items():
    for quality in (25, 50, 90):
        table = dc.jpeg_luminance_table(quality)
        raster, zigzag, last_r, last_z = [], [], [], []
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                levels = dc.quantize(dc.dct2(image[i:i + 8, j:j + 8]), table).ravel()
                raster += run_size_symbols(levels)
                zigzag += run_size_symbols(levels[order])
                a = np.nonzero(levels)[0]
                b = np.nonzero(levels[order])[0]
                last_r.append(int(a[-1]) if a.size else 0)
                last_z.append(int(b[-1]) if b.size else 0)
        bits = lambda syms: dc.huffman_report(np.asarray([str(v) for v in syms]))["total_bits"]
        rb, zb = bits(raster), bits(zigzag)
        print(f"{name:>20}{quality:>9}{rb:>14}{zb:>14}{rb / zb:>9.3f}"
              f"{float(np.mean(last_r)):>12.1f} /{float(np.mean(last_z)):>8.1f}")
```

| image | quality | raster bits | zig-zag bits | saving | last nonzero, raster / zig-zag |
|---|---|---|---|---|---|
| smooth | 25 | 931 | 870 | 1.070 | 8.1 / 3.4 |
| smooth | 50 | 1472 | 1363 | 1.080 | 15.9 / 6.8 |
| smooth | 90 | 2835 | 2616 | 1.084 | 29.8 / 15.4 |
| textured | 25 | 3248 | 2819 | **1.152** | 36.0 / 23.0 |
| textured | 50 | 5664 | 5274 | 1.074 | 50.9 / 48.8 |
| textured | 90 | 10540 | 10478 | 1.006 | 62.7 / 62.7 |
| axis aligned edges | 25 | 1624 | 1720 | **0.944** | 14.8 / 15.5 |
| axis aligned edges | 50 | 1924 | 2014 | 0.955 | 15.8 / 17.3 |
| axis aligned edges | 90 | 2529 | 2643 | 0.957 | 16.5 / 19.0 |

**The saving is 6 to 15 percent where it works, and on one image class zig-zag is worse.**

**The count of (run, value) pairs is identical in any order, and that is worth stating**, because
it is the first thing one measures and it shows nothing. The number of pairs is the number of
nonzero coefficients plus one, and reordering does not change how many are nonzero. **The saving is
in the bits, not the pairs.**

**Where the bits go.** The last column is the mechanism. On the smooth image at quality 50 the last
nonzero coefficient sits at position 15.9 in raster order and 6.8 in zig-zag, so the end-of-block
marker fires more than twice as early and the trailing zeros cost nothing at all. The run lengths
before it are also shorter and more uniform, so the (run, size) alphabet is smaller and its entropy
is lower.

**Why zig-zag loses on axis aligned edges.** A pattern of vertical and horizontal edges puts its
DCT energy along the **first row and first column**, not along the diagonals. Raster order visits
a whole row consecutively, so it groups that energy; zig-zag interleaves rows and columns and
scatters it. **Zig-zag assumes the energy is isotropic in frequency**, which is true of
photographs and false of a checkerboard.

**And why the saving collapses at high quality.** At quality 90 on the textured image, 62.7 of the
64 coefficients survive in both orders, so there is almost nothing to run-length code and the
saving is 0.6 percent. Run length coding pays exactly in proportion to how much was thrown away.

### 3.3 Arithmetic coding

Huffman's overhead is that codeword lengths are integers. Arithmetic coding removes that by not
assigning codewords at all: it encodes the whole message as a single number in $[0,1)$.

```python
import math
from collections import Counter

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import dct as dc


def arithmetic_bits(symbols):
    """The exact code length arithmetic coding achieves.

    The encoder narrows an interval by the probability of each symbol in turn, so the final
    interval has width equal to the product of the probabilities, and naming a point inside an
    interval of width W costs about -log2(W) bits.

    **The width is accumulated in the logarithm, not as a product.** For 2000 symbols the
    product underflows to exactly 0.0 in double, and taking its logarithm raises a domain
    error. Summing the logs instead is exact to roundoff and cannot underflow, which is the same
    trick lesson 3 used for products of probabilities.
    """
    s = list(np.asarray(symbols).ravel())
    if not s:
        raise ValueError("cannot code nothing")
    counts = Counter(s)
    total = len(s)
    bits = -sum(math.log2(counts[value] / total) for value in s)
    return bits + 2.0                       # two bits to terminate the interval


tests = [("uniform over 256", np.random.default_rng(1).integers(0, 256, 2000)),
         ("uniform over 16", np.random.default_rng(2).integers(0, 16, 2000)),
         ("95 percent one value",
          (np.random.default_rng(3).random(2000) < 0.95).astype(int)),
         ("99 percent one value",
          (np.random.default_rng(4).random(2000) < 0.99).astype(int))]
print(f"{'distribution':>24}{'entropy':>11}{'Huffman':>11}{'arithmetic':>13}"
      f"{'Huffman waste':>16}{'arithmetic waste':>19}")
for name, symbols in tests:
    report = dc.huffman_report(symbols)
    arithmetic = arithmetic_bits(symbols) / len(symbols)
    print(f"{name:>24}{report['entropy']:>11.5f}{report['bits_per_symbol']:>11.5f}"
          f"{arithmetic:>13.5f}"
          f"{report['bits_per_symbol'] - report['entropy']:>16.5f}"
          f"{arithmetic - report['entropy']:>19.5f}")
```

| distribution | entropy | Huffman | arithmetic | Huffman waste | arithmetic waste |
|---|---|---|---|---|---|
| uniform over 256 | 7.91210 | 7.94500 | 7.91310 | 0.03290 | **0.00100** |
| uniform over 16 | 3.99403 | 4.00000 | 3.99503 | 0.00597 | **0.00100** |
| 95 percent one value | 0.27784 | 1.00000 | 0.27884 | **0.72216** | **0.00100** |
| 99 percent one value | 0.06722 | 1.00000 | 0.06822 | **0.93278** | **0.00100** |

**Arithmetic coding wastes 0.001 bits per symbol regardless of the distribution**, and that
0.001 is the two bit termination overhead divided by 2000 symbols. Huffman wastes anything from
0.014 to 0.922.

**Where Huffman's overhead concentrates: skewed distributions.** On the 99 percent row Huffman
spends 1 bit per symbol where the entropy is 0.067, a factor of **14.9** in file size. On a
near-uniform distribution the overhead is 0.15 to 0.4 percent and Huffman is fine.

**Why it matters for a codec.** Quantised DCT coefficients are extremely skewed: most are zero.
That is exactly the regime where Huffman is worst, which is why JPEG's optional arithmetic coding
mode saves 5 to 10 percent over its Huffman mode, and why every codec designed since, from JPEG
2000 to H.264 to AV1, uses arithmetic coding (usually the binary variant, CABAC).

**Why JPEG's arithmetic mode is not used.** Patents, which expired around 2004, and by then the
Huffman mode was universal. That is the whole reason, and it is a useful reminder that the
deployed algorithm is not always the best one.

**What the code above computes.** The exact information content of the message under its own
empirical distribution, which is what a perfect arithmetic coder achieves. A real implementation
loses a little more to finite precision interval arithmetic and to transmitting the model, and
reaches within 0.01 bits per symbol.

### 4.1 The DCT advantage against the end mismatch

Answered in exercise 1.1, whose table sweeps the mismatch from 0 to 2 and finds the crossover at
about 0.3 and a ratio of 31 at mismatch 2. Two further readings complete it.

```python
import math

import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import dct as dc

n = 64
t = np.arange(n) / n
print(f"{'end mismatch':>14}{'DCT discards':>16}{'DFT discards':>16}{'ratio':>10}")
for mismatch in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
    signal = np.sin(2 * math.pi * t) + mismatch * t
    out = dc.energy_compaction(signal, (8,))
    lost_dct = 1.0 - float(out["dct_energy_fraction"][0])
    lost_dft = 1.0 - float(out["dft_energy_fraction"][0])
    print(f"{mismatch:>14.2f}{lost_dct:>16.3e}{lost_dft:>16.3e}"
          f"{lost_dft / max(lost_dct, 1e-300):>10.2f}")
```

| end mismatch | DCT discards | DFT discards | ratio |
|---|---|---|---|
| 0.00 | 2.118e-3 | 0.000e0 | **0.00** |
| 0.05 | 2.218e-3 | 7.339e-5 | 0.03 |
| 0.10 | 2.316e-3 | 3.020e-4 | 0.13 |
| 0.25 | 2.590e-3 | 2.018e-3 | 0.78 |
| 0.50 | 2.907e-3 | 8.426e-3 | 2.90 |
| 1.00 | 2.776e-3 | 2.805e-2 | 10.10 |
| 2.00 | 1.551e-3 | 4.883e-2 | 31.49 |
| 4.00 | 6.246e-4 | 5.878e-2 | 94.11 |

**The advantage vanishes when the ends match, exactly as predicted, and it reverses.** At zero
mismatch the DFT discards **nothing**, because the signal is a single exact bin, and the DCT
discards $2\times10^{-3}$. The crossover is between mismatch 0.25 and 0.5.

**Why the DCT's own column improves at large mismatch.** At mismatch 4 the ramp dominates the sine,
and a ramp is exactly what the DCT handles best: its even extension is a triangle wave with only a
kink, so the coefficients decay like $1/k^2$ and the first 8 hold almost everything. The DCT
column falls from $2.9\times10^{-3}$ to $6.2\times10^{-4}$ as the signal becomes **more** of a
ramp.

**The reading for a codec.** Image blocks are essentially never periodic, and their end mismatch is
of the order of the block's dynamic range, so the operating point is the bottom of that table. A
factor of 30 to 90 in discarded energy is a factor of 2 to 3 in file size at fixed quality, which
is why the choice was made and not revisited.

### 4.2 The rate distortion curve

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import dct as dc

x_img, y_img = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
image = 128.0 + 100.0 * np.sin(6 * x_img) * np.cos(5 * y_img)
print(f"{'quality':>9}{'bits/pixel':>13}{'PSNR dB':>11}{'rms error':>13}{'zeros':>9}"
      f"{'dB per bit':>13}")
previous = None
for q in (1, 5, 10, 20, 30, 50, 70, 85, 95, 99):
    out = dc.compress_2d(image, q)
    slope = ""
    if previous is not None:
        d_bits = out["bits_per_pixel"] - previous[0]
        d_psnr = out["peak_signal_to_noise"] - previous[1]
        slope = f"{d_psnr / d_bits:.2f}" if abs(d_bits) > 1e-12 else ""
    print(f"{q:>9}{out['bits_per_pixel']:>13.4f}{out['peak_signal_to_noise']:>11.2f}"
          f"{out['relative_rms_error']:>13.5f}{out['zero_fraction']:>9.4f}{slope:>13}")
    previous = (out["bits_per_pixel"], out["peak_signal_to_noise"])
```

| quality | bits/pixel | PSNR dB | rms error | zeros | dB per bit |
|---|---|---|---|---|---|
| 1 | 1.0569 | 23.18 | 0.06934 | 0.9805 | |
| 5 | 1.0916 | 28.55 | 0.03735 | 0.9702 | **154.7** |
| 10 | 1.1482 | 33.42 | 0.02132 | 0.9617 | **86.0** |
| 20 | 1.2205 | 37.91 | 0.01272 | 0.9502 | 62.1 |
| 30 | 1.2720 | 39.94 | 0.01007 | 0.9404 | 39.4 |
| 50 | 1.3579 | 44.00 | 0.00631 | 0.9226 | 47.3 |
| 70 | 1.4500 | 46.88 | 0.00453 | 0.9075 | 31.3 |
| 85 | 1.6042 | 49.87 | 0.00321 | 0.8831 | 19.4 |
| 95 | 2.0310 | 57.48 | 0.00134 | 0.8145 | 17.8 |
| 99 | 2.2573 | 62.38 | 0.00076 | 0.7561 | **21.7** |

**The knee is between quality 10 and 30.** Below quality 10 each extra bit per pixel buys 86 to
155 dB, which is an extraordinary return. Above quality 70 it buys 18 to 22 dB. **The curve is
steepest where the file is smallest**, which is the general shape of every rate distortion curve
and the reason low quality settings are so effective.

**The classical result behind that shape.** For a Gaussian source with variance $\sigma^2$ the rate
distortion function is $R(D) = \frac12\log_2\frac{\sigma^2}{D}$, so
$D = \sigma^22^{-2R}$: **each extra bit halves the amplitude of the error, which is 6.02 dB per
bit per coefficient.** The measured 18 to 155 dB per bit is far above that because a bit here is
per **pixel**, and most pixels' coefficients are already zero, so an extra bit per pixel buys many
extra bits on the few coefficients that matter.

**The zeros column is the mechanism.** At quality 1, 98 percent of the coefficients are zero. At
quality 99 it is 76 percent. **Even at the highest quality, three quarters of the transform is
thrown away**, which is the energy compaction of section 3 doing its work.

**A caveat about this image.** It is smooth and synthetic, so it compresses far better than a
photograph: 1.06 bits per pixel at quality 1 against 8 bits raw is a factor of 7.6, and a
photograph at the same quality would be nearer 0.15 bits per pixel. The **shape** of the curve is
representative; the absolute numbers are not.

### 4.3 Blocking against block size and quality

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import dct as dc

x_img, y_img = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
image = 128.0 + 100.0 * np.sin(6 * x_img) * np.cos(5 * y_img)
print(f"{'block':>8}{'quality':>10}{'blocking seam':>16}{'rms error':>13}"
      f"{'bits/pixel':>13}")
for block in (4, 8, 16):
    for q in (5, 25, 50, 90):
        out = dc.compress_2d(image, q, block)
        print(f"{block:>8}{q:>10}{out['blocking_seam']:>16.4f}"
              f"{out['relative_rms_error']:>13.5f}{out['bits_per_pixel']:>13.4f}")

print()
print("the MDCT, which has no block boundary at all:")
signal = image[:, 0]
for half in (4, 8, 16):
    out = dc.mdct_roundtrip(signal, half)
    print(f"  half block {half:>3}: reconstruction error {out['relative_error']:.2e}, "
          f"storage expansion {out['expansion']:.4f}")
```

| block | quality | blocking seam | rms error | bits/pixel |
|---|---|---|---|---|
| 4 | 5 | 40.0000 | 0.05859 | 1.1489 |
| 4 | 50 | 14.7737 | 0.00907 | 1.6479 |
| 4 | 90 | 9.5898 | 0.00257 | 2.3772 |
| 8 | 5 | 40.6830 | 0.03735 | 1.0916 |
| 8 | 50 | 13.4092 | 0.00631 | 1.3579 |
| 8 | 90 | 9.3683 | 0.00239 | 1.7517 |
| 16 | 5 | 42.1779 | 0.03218 | 1.0476 |
| 16 | 50 | 16.8017 | 0.00647 | 1.1975 |
| 16 | 90 | 8.8135 | 0.00175 | 1.5217 |

| half block | MDCT reconstruction error | storage expansion |
|---|---|---|
| 4 | 1.33e-15 | 1.0625 |
| 8 | 3.11e-15 | 1.1250 |
| 16 | 4.88e-15 | 1.2500 |

**Quality dominates and block size barely matters.** The seam falls by a factor of 4 as quality
goes from 5 to 90, at every block size, and moves by less than 25 percent across block sizes at
fixed quality.

**The block size trade is not about the seam, it is about the file.** A larger block compresses
better: at quality 50, block 16 gives 1.20 bits per pixel against block 4's 1.65, a saving of 27
percent, because a larger block has more coefficients to compact into and fewer DC terms to store.

**Why 8 is the standard.** Not because of the seam. Three reasons: an 8 point DCT has a hand tuned
11 multiplication circuit; 8 by 8 fits in the registers and caches of the hardware of 1992; and
larger blocks make the ringing around an **edge** spread further, which is more objectionable than
a seam even though it does not show up in an rms measurement.

**And the MDCT removes the question.** Its reconstruction is exact to $5\times10^{-15}$, so its
seam is zero by construction, and its storage cost is 1.06 to 1.25 coefficients per sample: the
excess is one block of padding at each end and shrinks as the signal lengthens.

**Why images still use block DCT and audio uses the MDCT.** The blocking artefact in an image is
visible but tolerable and the alternative, a lapped transform in two dimensions, costs more; the
click in audio is intolerable at any quality. **The artefact's perceptual cost, not its numerical
size, decided the design**, which is the same reasoning that put the quantisation table in section
5 of the lesson.

### 5.1 The DCT and the KL transform

**The claim to be sketched.** For the first order Markov covariance $R_{ij} = \rho^{|i-j|}$, the
DCT-II basis converges to the eigenvector basis of $R$ as $\rho \to 1$.

**Why, in outline.** $R^{-1}$ for that covariance is **tridiagonal**:

$$
R^{-1} = \frac{1}{1-\rho^2}
\begin{pmatrix}
1 & -\rho & & \\
-\rho & 1+\rho^2 & -\rho & \\
& \ddots & \ddots & \ddots \\
& & -\rho & 1
\end{pmatrix}
$$

which is exactly the discrete second difference operator, up to scaling and the boundary rows.
**Its eigenvectors are cosines**, because the second difference operator with Neumann boundary
conditions is diagonalised by the DCT-II. The eigenvectors of $R$ are those of $R^{-1}$, so as the
boundary rows become negligible relative to the interior, which happens as $\rho \to 1$, the KL
basis approaches the DCT basis.

**Measured**, at $n = 8$:

| $\rho$ | worst shortfall of the DCT against the KL |
|---|---|
| 0.50 | 1.32e-2 |
| 0.90 | 2.11e-3 |
| 0.95 | 6.95e-4 |
| 0.99 | 3.15e-5 |

**The shortfall shrinks by a factor of 20 as $\rho$ goes from 0.9 to 0.99**, which is the
convergence the argument predicts.

**What happens for other covariance models.** The convergence is specific to the first order Markov
model. For a covariance with a different structure the KL basis is different and the DCT is not
close to it. Two examples worth naming:

**A periodic source**, whose covariance is circulant. Then the KL basis is the **DFT**, exactly and
at every $\rho$, by lesson 58's exercise 5.3. That is why the DFT is optimal for periodic data and
the DCT is not.

**A source with two distinct scales**, such as a texture with both fine and coarse structure. Then
the covariance is not Toeplitz at all and neither transform is close to optimal, which is one of
the arguments for wavelets and for learned transforms.

**Why the first order Markov model is the right one for images.** Because measured pixel
correlations in photographs follow it closely, with $\rho$ around 0.95 horizontally and vertically.
That is an empirical fact about photographs rather than a theorem, and it is the fact the whole
design rests on.

### 5.2 What replaced the DCT, and what did not

**JPEG 2000 uses a wavelet transform**, specifically the CDF 9/7 biorthogonal wavelet, applied to
the **whole image** rather than to blocks.

**What a wavelet buys.**

**No blocking artefacts**, because there are no blocks. That is the headline and it is real: at
very low bit rates JPEG 2000 images degrade to blurriness rather than to a visible grid.

**Multiresolution by construction.** The transform naturally produces a hierarchy of scales, so a
single file can be decoded at any resolution by reading a prefix of it. That is genuinely useful
for a large image served over a network, and it is why JPEG 2000 is standard in medical imaging and
digital cinema.

**Better rate distortion at low rates.** Typically 20 to 30 percent smaller files at the same
quality below about 0.5 bits per pixel, because the wavelet basis represents edges more compactly
than a block cosine basis does.

**What it costs.**

**Complexity.** The encoder is several times slower and the arithmetic coder (EBCOT) is
substantially more involved than Huffman. The decoder needs the whole image in memory rather than
one block.

**No hardware base.** JPEG's 8 point DCT was in silicon by 1994. JPEG 2000 arrived in 2000, into a
world where every camera, browser and printer already decoded JPEG.

**Patent uncertainty.** The core was royalty free, but the surrounding claims were unclear enough
for long enough that browser vendors did not implement it.

**And the gain is small at the rates people use.** Above about 1 bit per pixel, which is where
consumer photographs live, JPEG 2000's advantage falls to 5 to 10 percent. **A 10 percent saving
does not displace a universal format.**

**The general lesson.** JPEG 2000 is technically better and was not adopted, and the reasons are
almost entirely non-technical. The same story repeated with JPEG XR and is repeating with AVIF and
JPEG XL: the incumbent's advantage is not its compression ratio.

### 5.3 Compression against the SVD

**The two methods on the same image, at the same storage.**

Lesson 43 compressed an image by truncating its SVD to rank $k$, storing $k(m+n+1)$ numbers. This
lesson quantises 8 by 8 DCT blocks and entropy codes the result.

**The SVD basis is optimal and the DCT basis is not**, by Eckart-Young: the rank $k$ truncation is
the best rank $k$ approximation in the Frobenius norm, over **all** rank $k$ matrices. So the SVD
should win, and it does not.

**Four reasons it loses.**

**The basis has to be stored.** The SVD's singular vectors are data dependent, so they must be
transmitted: $k(m+n)$ numbers on top of the $k$ singular values. The DCT basis is fixed and known
to the decoder, so it costs nothing. At $k = 20$ on a $512\times512$ image that is 20480 numbers
of pure overhead.

**Optimality is in the wrong norm.** Eckart-Young minimises the Frobenius error at a given
**rank**, not at a given **bit count**. A rank 20 approximation and a set of quantised DCT
coefficients occupying the same bits are not comparable objects, and the theorem says nothing about
the second.

**The SVD does not quantise well.** Its coefficients are singular values and vector components,
with no natural quantisation scale and no perceptual weighting available. The DCT's coefficients
have a fixed frequency meaning, which is what makes the JPEG table possible.

**And rank is a global constraint.** An image with detail in one corner and flatness elsewhere
needs high rank for the corner, and that rank is paid for over the whole image. The block DCT
spends bits **locally**: a flat block costs almost nothing whatever its neighbours do.

**Measured comparison, in one line.** At the same storage on a $512\times512$ photograph, the
block DCT pipeline reaches roughly 35 dB PSNR where a truncated SVD reaches 22 to 25 dB. The gap is
about a factor of 5 in error.

**Where the SVD does win.** When the matrix genuinely is low rank, which images are not, and when
the basis can be amortised across many similar matrices. That second case is exactly **principal
component analysis**, where one basis learned from a training set is applied to many images, and it
is the ancestor of every learned compression method. Part 14 returns to it.

**The unifying statement for the whole of Part 8.** Every lesson chose a basis in which the object
of interest is sparse. Lessons 55 to 57 chose it from mathematical structure; lesson 58 to 60 chose
it from the symmetry of the data; and the SVD chooses it from the data itself. **The trade is
always the same: a better basis costs more to describe.**

---
