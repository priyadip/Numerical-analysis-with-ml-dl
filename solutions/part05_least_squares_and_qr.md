# Solutions: Part 5, Orthogonality, QR and Least Squares

Worked solutions for the exercises in lessons 29 to 34.

Levels 1 and 2 are answered in full. Levels 3 and 4 give the method, the key code, and the
result you should get, so you can check your own work rather than copy it. Level 5 questions
are open ended, so those get a route through the problem and the answer where there is a
definite one.

Every number quoted was measured by running the code, not estimated.

---

## Lesson 29, Least Squares and the Normal Equations

### 1.1 A large residual and a perfect solution

**Because the residual is not the error.** For a square consistent system the residual being
zero is the goal, and lesson 19 spent its length warning that a small residual does not imply a
small error. For a least squares problem the residual is not even supposed to be zero: it is
$\sin\theta$ times $\|\mathbf{b}\|$, and $\theta$ is a property of the data.

**The two quantities measure different things.** The residual measures how well the **model**
fits the data. The error measures how well the **computation** found the best fit. A model that
explains nothing, fitted perfectly, has a large residual and zero error.

**Measured in lesson 29 section 8**: a degree 0 fit to quadratic data has `rms` 1.22, and it is
the exact minimiser. The residual is large because a constant cannot describe a parabola, not
because anything went wrong.

**And the converse matters more.** A tiny residual with a large error is exactly what the normal
equations produce at high $\kappa$, because the residual is flat near the minimum and a badly
wrong $\mathbf{x}$ still gives nearly the minimum residual.

### 1.2 Adding a parameter reduced the residual

**It always does, so the observation carries no information.** Adding a column to $A$ enlarges
$\operatorname{range}(A)$, and the minimum over a larger set cannot be larger. The old solution
is still available with a zero coefficient on the new column.

**Measured in lesson 29 section 8**, on 25 points from a quadratic with noise:

| degree | 0 | 1 | 2 | 3 | 6 | 12 | 20 |
|---|---|---|---|---|---|---|---|
| rms | 1.224 | 0.176 | 0.0388 | 0.0364 | 0.0314 | 0.0263 | 0.0223 |

Monotone all the way to degree 20, where 21 parameters are being fitted to 25 points.

**What to use instead.** Divide by the degrees of freedom, giving
$\sigma = \|\mathbf{r}\|/\sqrt{m-n}$, which penalises parameters. Measured, $\sigma$ drops by a
factor of 4.5 on reaching degree 2, where the truth is, then moves by a few percent and turns
back up. **Its minimum is shallow**, so it narrows the choice rather than making it.

**Better still, cross validate**, which exercise 4.3 measures: leave-one-out error rises
sharply past the true degree, from 0.070 at degree 2 to **68** at degree 20, while the residual
is still falling.

### 1.3 Why $P$ is never formed

**Size.** $P = A(A^TA)^{-1}A^T$ is $m\times m$ where the problem is $m\times n$. For 10000
measurements of 3 parameters that is $10^8$ entries against $3\times10^4$: **3300 times the
data**. The formulas contain $P$ because it names the operation; the operation is a solve.

**Cost.** Forming it needs an $n\times n$ solve with $m$ right-hand sides, $O(m^2n)$, against
$O(mn^2)$ for the answer. For $m \gg n$ that is a factor of $m/n$.

**Accuracy.** It contains $(A^TA)^{-1}$, so it inherits the squared condition number of lesson
29 section 5, and it inverts explicitly, which lesson 19 established is worse than solving.

**What is done instead.** Nothing needs $P$ itself, only $P\mathbf{b}$, which is
$A\mathbf{x}$ once $\mathbf{x}$ is known. So it costs one extra matrix-vector product on top of
a solve you already did.

**When it IS formed**, and there is one case: as a **measuring instrument**, at small sizes, to
check $P^2 = P$ and $\|P\|_2 = 1$ as lesson 29 section 3 does. That is a lesson, not a
computation.

### 2.1 $A^TA$ is positive definite exactly when $A$ has full column rank

**Semidefinite always.** For any $\mathbf{x}$,
$\mathbf{x}^TA^TA\mathbf{x} = (A\mathbf{x})^T(A\mathbf{x}) = \|A\mathbf{x}\|_2^2 \ge 0$.

**Definite exactly when the columns are independent.** The form vanishes iff
$\|A\mathbf{x}\| = 0$ iff $A\mathbf{x} = \mathbf{0}$. So $A^TA$ is positive definite iff the
only $\mathbf{x}$ with $A\mathbf{x} = \mathbf{0}$ is $\mathbf{x} = \mathbf{0}$, which is full
column rank.

**A rank deficient example.** Take

$$A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6\end{pmatrix}, \qquad
\mathbf{b} = \begin{pmatrix}1\\0\\0\end{pmatrix}.$$

The second column is twice the first, so $A^TA = \begin{pmatrix}14 & 28\\28&56\end{pmatrix}$ is
singular, and the normal equations $A^TA\mathbf{x} = A^T\mathbf{b} = (1,2)^T$ have a whole line
of solutions: any $\mathbf{x}$ with $x_1 + 2x_2 = 1/14$.

**All of them give the same residual**, which is why the least squares condition does not choose
between them. Lesson 32 section 4 measures exactly this and explains that picking the smallest
is a **choice**.

### 2.2 $\kappa_2(A^TA) = \kappa_2(A)^2$

With $A = U\Sigma V^T$ and $U$ having orthonormal columns,

$$A^TA = V\Sigma^TU^TU\Sigma V^T = V\Sigma^T\Sigma V^T = V\operatorname{diag}(\sigma_i^2)V^T,$$

which is an eigendecomposition, since $V$ is orthogonal. So the eigenvalues of $A^TA$ are
$\sigma_i^2$, and because it is symmetric positive definite its singular values equal its
eigenvalues. Therefore

$$\kappa_2(A^TA) = \frac{\sigma_1^2}{\sigma_n^2} = \left(\frac{\sigma_1}{\sigma_n}\right)^2
= \kappa_2(A)^2.$$

**For other norms the result is weaker.** $\kappa_1$ and $\kappa_\infty$ are not determined by
the singular values, so there is no exact identity. What survives is the equivalence of norms
from lesson 15: $\kappa_1(A^TA)$ is within a factor of $n$ of $\kappa_2(A^TA)$, so it too is
about $\kappa(A)^2$ up to a dimensional constant. **The squaring is a fact about the matrix, not
about the 2-norm**, and only its exactness is.

**Measured in lesson 29 section 5** across five decades of $\kappa(A)$: the ratio
$\kappa(A^TA)/\kappa(A)^2$ is 1 to six digits until $\kappa(A)^2$ passes $10^{15}$, beyond
which the reference itself is unreliable.

### 2.3 The three properties characterise an orthogonal projector

**They hold.** With $G = A^TA$ symmetric and invertible:

- $P^2 = AG^{-1}A^TAG^{-1}A^T = AG^{-1}GG^{-1}A^T = AG^{-1}A^T = P$.
- $P^T = (AG^{-1}A^T)^T = A G^{-T}A^T = AG^{-1}A^T = P$, using $G^T = G$.
- $\|P\|_2 = 1$: since $P = P^T = P^2$, its eigenvalues satisfy $\lambda^2 = \lambda$, so each
  is 0 or 1, and a symmetric matrix has $\|P\|_2 = \max|\lambda| = 1$ provided $P \ne 0$.

**And they characterise.** Suppose $P^2 = P$, $P^T = P$. Let $S = \operatorname{range}(P)$. For
$\mathbf{v} \in S$, write $\mathbf{v} = P\mathbf{w}$; then $P\mathbf{v} = P^2\mathbf{w} =
P\mathbf{w} = \mathbf{v}$, so $P$ fixes $S$. For any $\mathbf{u}$ and $\mathbf{v} = P\mathbf{w}
\in S$,

$$\mathbf{v}^T(I-P)\mathbf{u} = \mathbf{w}^TP^T(I-P)\mathbf{u}
= \mathbf{w}^T(P - P^2)\mathbf{u} = 0,$$

so $(I-P)\mathbf{u} \perp S$. Hence $P\mathbf{u}$ is the orthogonal projection of $\mathbf{u}$
onto $S$, which is what "orthogonal projector" means.

**The symmetry is what makes it orthogonal.** Dropping it leaves $P^2 = P$, an **oblique**
projector, which still projects onto its range but along a direction that is not perpendicular.
Lesson 16 measured $\|P\|_2 = 1/\cos\theta_{\max}$ for those, which can be arbitrarily large.
That single condition is the difference between a projection that cannot amplify anything and
one that can amplify without bound.

**Measured in lesson 29 section 3** at four shapes up to $200\times40$: $\|P^2-P\|$ and
$\|P-P^T\|$ below $4\times10^{-16}$, and $\|P\|_2 = 1.0000000000$ to ten decimal places.

### 2.4 Weighted least squares

Minimise $\|W(\mathbf{b}-A\mathbf{x})\|_2^2$ with $W = \operatorname{diag}(w_i)$, $w_i > 0$.
Substituting $\tilde{A} = WA$, $\tilde{\mathbf{b}} = W\mathbf{b}$ turns it into an ordinary
least squares problem, so the normal equations are

$$\tilde{A}^T\tilde{A}\mathbf{x} = \tilde{A}^T\tilde{\mathbf{b}}, \qquad\text{that is}\qquad
A^TW^2A\,\mathbf{x} = A^TW^2\mathbf{b}.$$

**Implement it by scaling the rows**, never by forming $W^2$: the substitution is the algorithm.

```python
def weighted_solve(A, b, w):
    """min ||W(b - Ax)|| with W = diag(w). Scale the rows and solve as usual."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if w.size != b.size:
        raise ValueError(f"w has length {w.size}, b has {b.size}")
    if np.any(w <= 0):
        raise ValueError("weights must be positive")
    return ls.solve_qr(A * w[:, None], b * w).x
```

**Which weighting corresponds to fitting $\log y$?** The chain rule: a perturbation
$\delta y$ produces $\delta(\log y) = \delta y / y$. So minimising the squared error in
$\log y$ is minimising $\sum (\delta y_i / y_i)^2$, which is weighted least squares in $y$ with

$$w_i = 1/y_i.$$

**That is exactly lesson 29 section 7's point, stated as a weighting.** Taking logs is not a
computational convenience: it silently applies the weights $1/y_i$, which is right when the
noise is multiplicative and wrong when it is additive.

### 2.5 $\sin\theta$ is the relative residual

Split $\mathbf{b}$ by the projector: $\mathbf{b} = P\mathbf{b} + (I-P)\mathbf{b}$, with the two
pieces orthogonal. So by Pythagoras,

$$\|\mathbf{b}\|^2 = \|P\mathbf{b}\|^2 + \|\mathbf{r}\|^2.$$

The angle between $\mathbf{b}$ and $\operatorname{range}(A)$ is by definition the angle between
$\mathbf{b}$ and its closest point in that subspace, namely $P\mathbf{b}$. In the right triangle
with hypotenuse $\mathbf{b}$, adjacent side $P\mathbf{b}$ and opposite side $\mathbf{r}$:

$$\cos\theta = \frac{\|P\mathbf{b}\|}{\|\mathbf{b}\|}, \qquad
\sin\theta = \frac{\|\mathbf{r}\|}{\|\mathbf{b}\|}.$$

**And that is why $\theta$ is free.** The relative residual is available from any solve, so
$\theta$ costs nothing beyond the solve, and lesson 32 makes it the first thing to look at.
Measured in lesson 29 section 8 and lesson 32 section 1: the identity holds to nine digits at
every noise level.

**But compute it with `arctan2`, not `arccos`.** Lesson 32 measures the arccos form returning
**exactly zero** for relative residuals below $10^{-8}$, because the derivative of arccos is
infinite where its argument sits. `arctan2(||r||, ||Pb||)` is exact to $10^{-16}$.

### 3.1 The two solvers, from scratch

```python
def solve_by_normal_equations(A, b):
    """Form A^T A and solve. Correct mathematics, and the algorithm to avoid."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    return np.linalg.solve(A.T @ A, A.T @ b)


def solve_by_qr(A, b):
    """R x = Q^T b. A^T A is never formed, so the condition number is not squared."""
    from nalib.lu import back_substitution

    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    Q, R = np.linalg.qr(A, mode="reduced")
    return back_substitution(R, Q.T @ b)
```

Reproducing lesson 29's table, relative error against $\kappa(A)$ at $60\times8$:

| $\kappa(A)$ | $\kappa(A^TA)$ | normal | Cholesky | QR |
|---|---|---|---|---|
| $10^{2}$ | $10^{4}$ | $2.6\times10^{-13}$ | $4.1\times10^{-13}$ | $1.7\times10^{-15}$ |
| $10^{4}$ | $10^{8}$ | $7.1\times10^{-9}$ | $5.6\times10^{-9}$ | $3.7\times10^{-13}$ |
| $10^{6}$ | $10^{12}$ | $8.6\times10^{-6}$ | $2.1\times10^{-5}$ | $3.4\times10^{-11}$ |
| $10^{8}$ | $1.3\times10^{16}$ | $4.7\times10^{-2}$ | $1.1\times10^{-1}$ | $2.2\times10^{-9}$ |
| $10^{10}$ | $2.4\times10^{16}$ | $8.8\times10^{-1}$ | breaks down | $8.6\times10^{-8}$ |

**Two things to check in your own version.** That the errors are relative, since an absolute
error means nothing across five decades. And that the answers are compared against
$\mathbf{x}_{\text{true}}$ rather than against each other, or the comparison would be circular.

### 3.2 Weighted least squares, measured

Using the routine from exercise 2.4, on a quadratic fit at 40 points where the first half have
error bars 100 times smaller than the second:

| | median error over 200 trials |
|---|---|
| unweighted | 1.2528 |
| **weighted, $w = 1/\sigma$** | **0.0881** |

**Weighting wins 100 percent of the 200 trials, by a median factor of 15.5.**

**And it is a no-op when the error bars are equal**, which is the check that the implementation
is right rather than merely different: with $\sigma$ constant, the weighted and unweighted
answers agree to $8\times10^{-15}$.

**Why it works.** Least squares is the maximum likelihood estimate under equal-variance Gaussian
noise (exercise 5.1). When the variances differ, the likelihood weights each residual by
$1/\sigma_i^2$, which is exactly $w_i = 1/\sigma_i$ inside the norm. Fitting unweighted data with
unequal error bars is not merely suboptimal, it is fitting the wrong likelihood.

### 3.3 A streaming fit

$A^TA$ is $n\times n$ and $A^T\mathbf{b}$ is length $n$, and both are **sums over rows**. So
they can be accumulated one block at a time and $A$ never has to exist.

```python
def streaming_fit(row_source, n_par, n_rows, block=10000):
    """Accumulate A^T A and A^T b one block at a time. A is never stored.

    row_source(start, count) returns the rows and right-hand side entries for that slice, so
    the data may come from a file, a database or a generator. n_par and n_rows are free.
    """
    G = np.zeros((n_par, n_par))
    rhs = np.zeros(n_par)
    seen = 0
    while seen < n_rows:
        take = min(block, n_rows - seen)
        Ai, bi = row_source(seen, take)
        G += Ai.T @ Ai
        rhs += Ai.T @ bi
        seen += take
    return np.linalg.solve(G, rhs), G
```

**Measured**, fitting 3 parameters to $10^{7}$ observations:

| quantity | value |
|---|---|
| time | 0.2 s |
| memory for $A^TA$ and $A^T\mathbf{b}$ | **96 bytes** |
| memory to store $A$ | **0.24 GB** |
| $\kappa(A^TA)$ | 524 |
| relative error | $2.9\times10^{-5}$ |

**This is the one case where the normal equations are the right algorithm**, and the reason is
in the third and fourth rows: 96 bytes against a quarter of a gigabyte, a factor of
$2.5\times10^{6}$. QR would need the whole of $A$ in memory, or a much more complicated
out-of-core scheme.

**And it is safe here because $\kappa(A^TA) = 524$**, so squaring cost nothing. **Check that
number before trusting the method**: the same code on a badly conditioned design would return
the same confident garbage lesson 29 measured.

### 4.1 The sample spacing

Measured, $\kappa$ of the Vandermonde design at 60 points on $[-1,1]$:

| degree | evenly spaced | Chebyshev | random |
|---|---|---|---|
| 4 | $1.82\times10^{1}$ | $1.97\times10^{1}$ | $1.93\times10^{1}$ |
| 8 | $5.20\times10^{2}$ | $6.27\times10^{2}$ | $7.34\times10^{2}$ |
| 12 | $1.63\times10^{4}$ | $2.06\times10^{4}$ | $2.23\times10^{4}$ |
| 16 | $5.44\times10^{5}$ | $6.88\times10^{5}$ | $1.03\times10^{6}$ |
| 20 | $1.96\times10^{7}$ | $2.31\times10^{7}$ | $8.47\times10^{7}$ |

**Chebyshev spacing does not help, and that is the interesting answer.** It is very slightly
*worse* than even spacing at every degree, and random spacing is the only clearly bad one, by a
factor of 4 at degree 20.

**Why the expectation was wrong.** Chebyshev points are the right choice for **interpolation**,
where the number of points equals the number of parameters and the question is where to sample
to control the Lebesgue constant (lesson 47). Here there are 60 points and up to 21 parameters,
so the problem is overdetermined and the sampling is already dense. What makes the design ill
conditioned is not where the points are but that $1, x, x^2, \dots$ are nearly parallel as
functions on $[-1,1]$.

**The basis is the whole story.** Repeating with Legendre polynomials, built by the three-term
recurrence, on the same evenly spaced points:

| degree | monomial basis | Legendre basis | ratio |
|---|---|---|---|
| 4 | $1.82\times10^{1}$ | 2.84 | 6.4 |
| 8 | $5.20\times10^{2}$ | 3.94 | 132 |
| 12 | $1.63\times10^{4}$ | 4.89 | $3.3\times10^{3}$ |
| 16 | $5.44\times10^{5}$ | 5.84 | $9.3\times10^{4}$ |
| 20 | $1.96\times10^{7}$ | **7.76** | $2.5\times10^{6}$ |

**A factor of 2.5 million at degree 20**, against a factor of 0.85 for the best choice of
points. Change the basis, not the sampling. Lesson 55 builds orthogonal polynomials properly.

### 4.2 The two slopes

Measured at $60\times8$, fitting $\log_{10}(\text{relative error})$ against
$\log_{10}\kappa(A)$ over the range where the error is above roundoff and below 1:

| method | slope | fitted over |
|---|---|---|
| normal equations | **1.968** | $\kappa = 10$ to $10^{7}$ |
| QR | **0.926** | $\kappa = 10$ to $10^{14}$ |

**Ratio of slopes: 2.13, against a theoretical 2.**

**Two details that matter for getting this right.** The fit must exclude the points at the
roundoff floor, where the error stops falling and the slope flattens spuriously, and the points
where the error has saturated near 1. And the range over which each is fitted differs, because
the normal equations run out of usable range at $\kappa = 10^{8}$ and QR at $10^{16}$, which is
the same finding in another form.

### 4.3 Residual against cross validation

Measured on 30 points from a quadratic with noise 0.08:

| degree | residual | $\sigma$ | leave-one-out |
|---|---|---|---|
| 0 | 6.781 | 1.259 | 1.281 |
| 1 | 0.898 | 0.170 | 0.180 |
| **2** | 0.344 | 0.0663 | **0.0701** |
| 3 | 0.344 | 0.0675 | 0.0745 |
| 5 | 0.322 | **0.0657** | 0.0782 |
| 8 | 0.313 | 0.0683 | 0.118 |
| 12 | 0.290 | 0.0703 | 0.192 |
| 20 | **0.229** | 0.0765 | **68.1** |

**The three columns tell three different stories.** The residual falls monotonically and picks
degree 20, which is wrong. $\sigma$ has a shallow minimum at degree 5, close to the truth but
not exactly, and only 4 percent below its value at degree 2. **Leave-one-out picks degree 2,
which is right**, and it rises by a factor of a thousand by degree 20.

**Where they diverge is the answer to the exercise.** The residual and $\sigma$ track each other
until about degree 8; leave-one-out separates from both at degree 3 and never comes back.

**Why cross validation is sharper.** The residual asks how well the model fits the data it was
fitted to, which more parameters always improve. Cross validation asks how well it predicts data
it has not seen, which more parameters eventually ruin. Those are different questions and only
the second one is usually the one being asked.

### 5.1 The statistical reading

**Maximum likelihood.** Suppose $\mathbf{b} = A\mathbf{x} + \boldsymbol{\varepsilon}$ with
$\varepsilon_i$ independent $N(0,\sigma^2)$. The likelihood is

$$L(\mathbf{x}) = \prod_i \frac{1}{\sigma\sqrt{2\pi}}
\exp\left(-\frac{(b_i - (A\mathbf{x})_i)^2}{2\sigma^2}\right),$$

so

$$-\log L = \text{const} + \frac{1}{2\sigma^2}\|\mathbf{b}-A\mathbf{x}\|_2^2.$$

Maximising $L$ is minimising $\|\mathbf{b}-A\mathbf{x}\|_2^2$. **That is the statistical reason
for the 2-norm**, and it is a statement about the error distribution rather than about
convenience.

**The covariance.** $\hat{\mathbf{x}} = (A^TA)^{-1}A^T\mathbf{b}$ is linear in $\mathbf{b}$, so

$$\operatorname{cov}(\hat{\mathbf{x}}) = (A^TA)^{-1}A^T\operatorname{cov}(\mathbf{b})
A(A^TA)^{-1} = \sigma^2(A^TA)^{-1},$$

using $\operatorname{cov}(\mathbf{b}) = \sigma^2I$ and $(A^TA)^{-1}$ symmetric.

**And now the connection to conditioning.** The variance of the $j$-th coefficient is
$\sigma^2[(A^TA)^{-1}]_{jj}$, and the largest eigenvalue of $(A^TA)^{-1}$ is
$1/\sigma_n(A)^2$. So the worst-case standard error is

$$\frac{\sigma}{\sigma_n(A)} = \frac{\sigma\,\kappa(A)}{\sigma_1(A)}.$$

**The squared condition number is the same fact seen twice.** Numerically, $\kappa(A)^2$ is how
much the normal equations amplify roundoff. Statistically, $\kappa(A)^2$ appears in
$(A^TA)^{-1}$ as how much the design amplifies **noise in the data**. A badly conditioned design
has large error bars whatever algorithm you use, and lesson 32's four condition numbers make
that precise.

**The practical reading.** If $\kappa(A) = 10^{4}$ and your data has 1 percent noise, the worst
determined coefficient has a standard error of order 100 times the data noise. No algorithm
fixes that; only a better design does.

### 5.2 Seminormal equations

**The idea.** $A^TA = R^TR$ where $R$ is the QR factor of $A$, since
$A^TA = (QR)^TQR = R^TQ^TQR = R^TR$. So the normal equations can be solved as two triangular
solves with $R$, and $A^TA$ is never explicitly formed:

```python
def seminormal(A, b, refine=0):
    """Solve R^T R x = A^T b with R from a QR of A. A^T A is never formed.

    One step of refinement (lesson 22) makes it as accurate as QR itself, and it costs one
    matrix-vector product plus two triangular solves.
    """
    from nalib.lu import back_substitution, forward_substitution

    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    _, R = np.linalg.qr(A)
    solve = lambda rhs: back_substitution(R, forward_substitution(R.T, rhs))
    x = solve(A.T @ b)
    for _ in range(int(refine)):
        x = x + solve(A.T @ (b - A @ x))       # lesson 22's correction, in the R factor
    return x
```

**Measured**, relative error at $60\times8$:

| $\kappa$ | normal | seminormal | **+1 refinement** | QR |
|---|---|---|---|---|
| $10^{2}$ | $8.4\times10^{-14}$ | $3.4\times10^{-14}$ | $3.4\times10^{-16}$ | $1.3\times10^{-15}$ |
| $10^{4}$ | $1.4\times10^{-9}$ | $1.8\times10^{-9}$ | $3.9\times10^{-14}$ | $8.8\times10^{-14}$ |
| $10^{6}$ | $9.2\times10^{-6}$ | $6.8\times10^{-7}$ | $2.0\times10^{-12}$ | $2.6\times10^{-12}$ |
| $10^{8}$ | $1.5\times10^{-1}$ | $4.9\times10^{-1}$ | $8.2\times10^{-10}$ | $4.2\times10^{-9}$ |
| $10^{10}$ | 3.46 | $4.1\times10^{3}$ | $3.8\times10^{-4}$ | $7.6\times10^{-8}$ |

**Plain seminormal is no better than the normal equations**, and at $\kappa = 10^{10}$ it is
three orders of magnitude **worse**. Not forming $A^TA$ does not help by itself: the
information lost is lost in $R$'s small diagonal entries either way.

**One step of refinement changes everything.** It matches QR at every $\kappa$ up to $10^{8}$
and beats it at $10^{6}$ and $10^{8}$, and only at $10^{10}$ does QR pull ahead.

**Why refinement works here.** The residual $\mathbf{b} - A\mathbf{x}$ is computed with the
**original** $A$, at the original conditioning, so it carries information the squared system
threw away. That is lesson 22's mechanism exactly, and the reason it succeeds without extended
precision is that the residual here is not a difference of nearly equal numbers: it is
genuinely nonzero for a least squares problem.

**When to use it.** When $R$ is already available from a previous factorization and a second
right-hand side arrives. It costs two triangular solves, $O(n^2)$, against a fresh
$O(mn)$ application of $Q$. With refinement it is accurate; without, it is not.

### 5.3 Why the 2-norm

Minimising $\|\cdot\|_1$ and $\|\cdot\|_\infty$ are both linear programs.

```python
def fit_l1(A, b):
    """min sum |b - Ax|, as a linear program with one slack per residual."""
    from scipy.optimize import linprog

    m, n = A.shape
    c = np.concatenate([np.zeros(n), np.ones(m)])          # minimise the sum of slacks
    Aub = np.vstack([np.hstack([A, -np.eye(m)]),           #  Ax - t <= b
                     np.hstack([-A, -np.eye(m)])])         # -Ax - t <= -b
    bub = np.concatenate([b, -b])
    out = linprog(c, A_ub=Aub, b_ub=bub,
                  bounds=[(None, None)] * n + [(0, None)] * m)
    return out.x[:n]


def fit_linf(A, b):
    """min max |b - Ax|, the same idea with ONE slack shared by every residual."""
    from scipy.optimize import linprog

    m, n = A.shape
    c = np.concatenate([np.zeros(n), [1.0]])
    Aub = np.vstack([np.hstack([A, -np.ones((m, 1))]),
                     np.hstack([-A, -np.ones((m, 1))])])
    bub = np.concatenate([b, -b])
    out = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(None, None)] * n + [(0, None)])
    return out.x[:n]
```

**Measured** on 21 points from $y = 2 + x/2$ with noise 0.1 and **one point moved by +15**:

| fit | intercept | slope | max $\lvert r\rvert$ | sum $\lvert r\rvert$ |
|---|---|---|---|---|
| truth | 2.000 | 0.500 | | |
| $L_2$, least squares | 2.686 | 0.5032 | 14.32 | 28.64 |
| **$L_1$, least absolute** | **1.948** | 0.5034 | 15.06 | **16.58** |
| $L_\infty$, minimax | **9.314** | 0.5241 | **7.59** | 156.6 |

**Each wins on its own criterion**, which is the first thing to notice: $L_1$ has the smallest
sum of absolute residuals, $L_\infty$ the smallest maximum, and $L_2$ would have the smallest
sum of squares. None is "the" answer.

**And the outlier reveals what each assumes.** $L_1$ is barely moved: intercept 1.948 against a
truth of 2.000, an error of 0.05. $L_2$ is pulled to 2.686, an error of 0.69, fourteen times
worse. $L_\infty$ is **destroyed**, at 9.314, an error of 7.3.

**Why, in one line each.** $L_\infty$ minimises the worst residual, so it must move the line
towards the outlier until the outlier is no longer uniquely worst; one bad point dictates the
whole fit. $L_2$ weights each residual by its own size, so a residual ten times larger
contributes a hundred times more. $L_1$ weights every residual equally, so an outlier counts
once however far out it is.

**What each assumes about the errors.** $L_2$ assumes Gaussian noise, whose tails decay like
$e^{-r^2}$, so a residual of 15 is essentially impossible and the fit moves to explain it.
$L_1$ assumes Laplace noise, $e^{-|r|}$, with much fatter tails, so a large residual is
surprising but not impossible. $L_\infty$ assumes **uniformly bounded** error, so it treats the
outlier as evidence that the bound is 7.6.

**So the 2-norm is the worst of the three when its assumption is wrong**, and the assumption
being wrong is exactly what an outlier is. The remedies in practice are to use $L_1$, or to use
$L_2$ with a robust loss (Huber, which is $L_2$ near zero and $L_1$ far out), or to detect and
remove the outlier and say in the write-up that you did.

---

## Lesson 30, Gram-Schmidt and QR

### 1.1 A small $\|A - QR\|$ does not mean the factorization is good

**Because it only checks half of what QR promises.** A QR factorization is two claims: that
$A = QR$, and that $Q$ has orthonormal columns. The residual $\|A - QR\|$ tests the first and
says nothing about the second.

**Measured in lesson 30**: $\|A - QR\|$ stays at $10^{-16}$ for classical Gram-Schmidt,
modified Gram-Schmidt, reorthogonalized and Householder at **every** $\kappa$ tested, while
$\|Q^TQ-I\|$ ranges from $10^{-16}$ to above 1. Checking only the product would report all four
as equally good.

**And the reason it is always small is almost trivial.** Gram-Schmidt computes
$\mathbf{a}_j = \sum_i r_{ij}\mathbf{q}_i$ by construction: it subtracts multiples of the
$\mathbf{q}_i$ from $\mathbf{a}_j$ and records what it subtracted. Whatever the $\mathbf{q}_i$
happen to be, orthogonal or not, the bookkeeping adds back up. The identity is enforced by the
algorithm's structure, so it cannot detect the algorithm's failure.

**What to check instead.** $\|Q^TQ-I\|$, which is $O(n^2)$ work on an $m\times n$ factorization
and therefore free. Exercise 4.3 constructs a case where it reads **0.5** while $\|A-QR\|$ reads
$2.5\times10^{-25}$.

### 1.2 What differs in floating point

**The coefficients are computed against different vectors.** Classical computes every
$r_{ij} = \mathbf{q}_i^T\mathbf{a}_j$ against the **original** $\mathbf{a}_j$, then subtracts
them all at once. Modified computes $r_{ij}$ against the **partially updated** vector, so each
projection is measured after the previous ones have already been removed.

In exact arithmetic these agree, because $\mathbf{q}_i^T\mathbf{a}_j =
\mathbf{q}_i^T(\mathbf{a}_j - \sum_{k<i} r_{kj}\mathbf{q}_k)$ when the $\mathbf{q}_k$ really are
orthogonal to $\mathbf{q}_i$.

**In floating point they are not orthogonal, and that is the whole difference.** If
$\mathbf{q}_k$ has drifted by $\epsilon$ from perpendicular to $\mathbf{q}_i$, then classical's
coefficient is wrong by $\epsilon r_{kj}$ and it never finds out, because it never looks at the
vector again. Modified measures against the already-cleaned vector, so that error has already
been removed before $r_{ij}$ is computed.

**The cost is identical.** Same flops, same order of operations counted, just reordered so each
measurement happens as late as possible. Measured slopes: classical **1.86** against
$\kappa$, modified **0.98**, a difference peaking at a factor of $4.7\times10^{7}$.

### 1.3 Why QR does not square the condition number

**Because it never forms $A^TA$.** Substituting $A = QR$ into the normal equations:

$$A^TA\mathbf{x} = A^T\mathbf{b} \;\Longrightarrow\; R^TQ^TQR\,\mathbf{x} = R^TQ^T\mathbf{b}
\;\Longrightarrow\; R^TR\mathbf{x} = R^TQ^T\mathbf{b} \;\Longrightarrow\;
R\mathbf{x} = Q^T\mathbf{b},$$

cancelling $R^T$ because it is invertible. The final system has condition number
$\kappa(R) = \kappa(A)$, since $Q$ is orthogonal and orthogonal matrices preserve singular
values.

**The squaring happened in a step QR skips.** $A^TA$ has $\kappa(A)^2$, and once you have
formed it the information in the small singular values is gone: they were rounded at the
relative precision of $\sigma_1^2$, not $\sigma_1$. QR never brings those quantities into
contact.

**Said geometrically.** $\|\mathbf{b}-A\mathbf{x}\| = \|Q^T\mathbf{b} - R\mathbf{x}\|$ because
$Q$ preserves lengths, so the minimisation can be done in the rotated coordinates directly. The
normal equations instead differentiate the squared norm, and squaring is where the conditioning
doubles.

### 2.1 Uniqueness of the reduced QR

**Existence** is Gram-Schmidt. **Uniqueness**: suppose $A = Q_1R_1 = Q_2R_2$ with both $R_i$
having positive diagonal. Then $Q_2^TQ_1 = R_2R_1^{-1} =: S$. The left side is a product of
matrices with orthonormal columns applied to a full rank $A$, so $S$ is orthogonal; the right
side is upper triangular with positive diagonal, since the inverse and product of such matrices
are again such. **An orthogonal upper triangular matrix with positive diagonal is the
identity**: orthogonality gives $S^TS = I$, so column 1 has norm 1 and only one nonzero entry,
$s_{11} = 1$; then column 2 is orthogonal to column 1, forcing $s_{12} = 0$, and has norm 1
with positive $s_{22}$, so $s_{22} = 1$; induct. Hence $R_1 = R_2$ and $Q_1 = Q_2$.

**What fails at rank deficiency.** Take

$$A = \begin{pmatrix}1 & 1\\0&0\\0&0\end{pmatrix}
= \underbrace{\begin{pmatrix}1&q\\0&*\\0&*\end{pmatrix}}_{Q}
\begin{pmatrix}1&1\\0&0\end{pmatrix}.$$

The second column of $A$ is already in the span of the first, so $r_{22} = 0$ and
$\mathbf{q}_2$ is **completely undetermined**: any unit vector perpendicular to $\mathbf{q}_1$
works. Uniqueness needed $r_{jj} > 0$, and here it cannot be arranged.

**And the numerical version of that is exercise 4.3.** With $r_{22}$ not exactly zero but
$O(u\|A\|)$, the algorithm does not refuse; it divides by it and returns whatever roundoff
happened to leave in the numerator.

### 2.2 Classical and modified agree in exact arithmetic

By induction on $j$. Assume $\mathbf{q}_1,\dots,\mathbf{q}_{j-1}$ are identical in both and
orthonormal. Write modified's partial vectors as
$\mathbf{v}^{(0)} = \mathbf{a}_j$ and
$\mathbf{v}^{(i)} = \mathbf{v}^{(i-1)} - (\mathbf{q}_i^T\mathbf{v}^{(i-1)})\mathbf{q}_i$.

Claim: $\mathbf{q}_i^T\mathbf{v}^{(i-1)} = \mathbf{q}_i^T\mathbf{a}_j$ for every $i$. Indeed

$$\mathbf{q}_i^T\mathbf{v}^{(i-1)} = \mathbf{q}_i^T\Big(\mathbf{a}_j -
\sum_{k<i}(\mathbf{q}_k^T\mathbf{a}_j)\mathbf{q}_k\Big)
= \mathbf{q}_i^T\mathbf{a}_j - \sum_{k<i}(\mathbf{q}_k^T\mathbf{a}_j)\,\mathbf{q}_i^T\mathbf{q}_k
= \mathbf{q}_i^T\mathbf{a}_j,$$

since $\mathbf{q}_i^T\mathbf{q}_k = 0$ for $k < i$. So modified's coefficients equal classical's,
both sets of subtractions produce the same $\mathbf{v}^{(j-1)}$, and normalising gives the same
$\mathbf{q}_j$ and $r_{jj}$.

**The proof used orthogonality exactly once**, in the step that kills the sum, and that is
precisely the assumption floating point breaks.

### 2.3 Gram-Schmidt is triangular orthogonalization

Column $j$ of Gram-Schmidt does
$\mathbf{q}_j = \big(\mathbf{a}_j - \sum_{i<j}r_{ij}\mathbf{q}_i\big)/r_{jj}$, which is a
**column operation** on the matrix: subtract multiples of earlier columns from column $j$, then
scale it. Every such operation is right multiplication by an upper triangular matrix.

Concretely, let $R_j$ be the identity with column $j$ replaced by
$(-r_{1j}/r_{jj},\dots,-r_{j-1,j}/r_{jj},\,1/r_{jj},0,\dots,0)^T$. Then $R_j$ is upper
triangular and invertible, and postmultiplying by it performs exactly step $j$. After all $n$
steps

$$A R_1 R_2\cdots R_n = \hat{Q}, \qquad\text{so}\qquad A = \hat{Q}(R_1\cdots R_n)^{-1} =
\hat{Q}\hat{R},$$

with $\hat{R} = R_n^{-1}\cdots R_1^{-1}$ upper triangular as a product of upper triangular
matrices. (Writing the $R_j$ as the inverses of the factors above is the other common
convention; either way the point is the same.)

**Why the name matters.** Gram-Schmidt applies **triangular** operations and hopes the result is
orthogonal. Householder applies **orthogonal** operations and knows the result is triangular.
Only the second has a factor whose quality does not depend on the input, which is lesson 31's
whole subject.

### 2.4 The orthogonality bounds

**Modified.** At step $j$ the computed coefficient satisfies
$\hat{r}_{ij} = \mathbf{q}_i^T\mathbf{v}^{(i-1)}(1+O(u))$, and the subtraction introduces an
error of size $u\|\mathbf{v}^{(i-1)}\|$. The final vector before normalising has norm
$r_{jj}$, so the **relative** error in $\mathbf{q}_j$ is of order
$u\max_i\|\mathbf{v}^{(i-1)}\|/r_{jj}$. Now $\|\mathbf{v}^{(0)}\| = \|\mathbf{a}_j\| \le
\sigma_1$ and $r_{jj} \ge \sigma_n$ (up to constants), so that ratio is bounded by $\kappa$,
giving

$$\|\hat{Q}^T\hat{Q} - I\| = O(u\,\kappa(A)).$$

**Classical loses the extra factor here.** Its coefficients are all computed against
$\mathbf{a}_j$, so the relevant vector in the bound is $\mathbf{a}_j$ throughout. But
$\mathbf{q}_i$ has itself an error of order $u\kappa$ from earlier steps, so
$\hat{r}_{ij} = \mathbf{q}_i^T\mathbf{a}_j$ carries an error $u\kappa\|\mathbf{a}_j\|$, and
dividing by $r_{jj}$ multiplies by another $\kappa$:

$$\|\hat{Q}^T\hat{Q} - I\| = O(u\,\kappa(A)^2).$$

**The mechanism in one sentence.** Modified measures each projection against a vector whose
earlier contamination has already been subtracted; classical measures it against the original,
so the contamination is measured too and then divided by a small number.

**Measured slopes: 1.86 and 0.98**, against 2 and 1. The classical slope falls short of 2
because it saturates: once $\|Q^TQ-I\|$ exceeds 1 the quantity has stopped measuring anything.

### 2.5 Twice is enough

Let $\mathbf{v}$ have decomposition $\mathbf{v} = \mathbf{p} + \mathbf{n}$ with
$\mathbf{p} \in S = \operatorname{span}(\mathbf{q}_1,\dots,\mathbf{q}_{j-1})$ and
$\mathbf{n}\perp S$. One orthogonalization pass produces
$\mathbf{v}' = \mathbf{v} - \sum(\mathbf{q}_i^T\mathbf{v})\mathbf{q}_i$, and by hypothesis its
component in $S$ satisfies $\|\mathbf{p}'\| \le \eta\|\mathbf{p}\|$ with $\eta<1$; the component
off $S$ is unchanged to roundoff.

**The pass is a contraction on the $S$-component and the identity off it.** So a second pass
gives $\|\mathbf{p}''\| \le \eta\|\mathbf{p}'\| \le \eta^2\|\mathbf{p}\|$. But the contraction
cannot continue below the level at which $\mathbf{p}'$ is defined at all: after one pass
$\|\mathbf{p}'\|$ is already at most $\eta\|\mathbf{v}\|$, and the second pass drives it to
$\eta^2\|\mathbf{v}\|$ or to the floor $O(u)\|\mathbf{v}\|$, whichever is larger.

**Kahan and Parlett's observation is that $\eta$ is essentially $u\kappa$ in practice**, so
$\eta^2$ is already far below $u$ and the floor is reached in two. A third pass has nothing left
to remove.

**Verified in lesson 30**: two passes reach roundoff and a third changes $\|Q^TQ-I\|$ by
nothing. And the standard caution: this holds when the first pass makes real progress. If a
column is genuinely dependent, no number of passes helps, and the right answer is to refuse
(exercise 4.3).

### 3.1 Block Gram-Schmidt

Orthogonalize $b$ columns at a time: one matrix product against everything already finished,
then an unblocked factorization within the block.

```python
def block_gram_schmidt(A, block):
    """Orthogonalize `block` columns at a time against the finished ones, then within.

    The point is not the arithmetic, which is the same, but that Q[:, :j].T @ V is one BLAS-3
    call instead of `block` separate BLAS-2 calls. Lesson 08's roofline: the data is read once
    and used `block` times.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    j = 0
    while j < n:
        w = min(int(block), n - j)
        V = A[:, j:j + w].copy()
        if j:
            C = Q[:, :j].T @ V                        # one BLAS-3 call for the whole block
            R[:j, j:j + w] = C
            V -= Q[:, :j] @ C
            C2 = Q[:, :j].T @ V                       # one reorthogonalization pass
            R[:j, j:j + w] += C2
            V -= Q[:, :j] @ C2
        Qb, Rb = qr.householder_qr(V)                 # within the block
        Q[:, j:j + w] = Qb
        R[j:j + w, j:j + w] = Rb
        j += w
    return Q, R
```

**Measured** with block 20:

| shape | MGS twice | block 20 | Householder |
|---|---|---|---|
| $400\times100$ | 0.048 s, orth $6.6\times10^{-16}$ | **0.008 s**, orth $2.6\times10^{-14}$ | 0.041 s, orth $1.6\times10^{-15}$ |
| $1000\times200$ | 0.224 s, orth $8.3\times10^{-16}$ | **0.030 s**, orth $5.8\times10^{-15}$ | 0.305 s, orth $1.9\times10^{-15}$ |

**6 to 7 times faster than the unblocked version, and 10 times faster than Householder at
$1000\times200$**, with $\|A-QR\|$ at $7\times10^{-16}$.

**The roofline explanation.** Unblocked MGS does $O(mn^2)$ flops with $O(mn^2)$ memory traffic:
every projection reads the whole of $Q$ so far and does one pass over it, an arithmetic
intensity of about 1 flop per word. It is memory bound. Blocking reads the same $Q$ once and
uses it for $b$ columns, an intensity of about $b$, which lands the operation on the compute
side of the roofline. **Same flops, 6 times the speed**, and lesson 08's whole point.

**The cost is a little orthogonality**: $2.6\times10^{-14}$ against $6.6\times10^{-16}$, a
factor of 40, because the within-block reorthogonalization is not as thorough as doing every
column against every predecessor. Still 14 digits, and the trade is usually worth it.

### 3.2 Gram-Schmidt in a general inner product

The algorithm does not change at all. Only the definition of "perpendicular" moves.

```python
def gram_schmidt_inner(A, M):
    """Modified Gram-Schmidt in <u,v> = u^T M v, with M symmetric positive definite."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    M = np.atleast_2d(np.asarray(M, dtype=float))
    m, n = A.shape
    if M.shape != (m, m):
        raise ValueError(f"M is {M.shape}, needs to be ({m}, {m})")
    Q = A.astype(float).copy()
    R = np.zeros((n, n))
    ip = lambda u, v: float(u @ (M @ v))
    for j in range(n):
        for i in range(j):
            R[i, j] = ip(Q[:, i], Q[:, j])
            Q[:, j] -= R[i, j] * Q[:, i]
        R[j, j] = np.sqrt(max(ip(Q[:, j], Q[:, j]), 0.0))
        if R[j, j] <= 0.0:
            raise np.linalg.LinAlgError(f"column {j} is dependent in this inner product")
        Q[:, j] /= R[j, j]
    return Q, R
```

Applied to the monomials $1, x, \dots, x^d$ with $M = \operatorname{diag}(w)$ from
**Gauss-Legendre quadrature**, which is exact for polynomials up to degree $2N-1$:

| nodes | degree | $\|Q^TMQ - I\|$ | max difference from Legendre |
|---|---|---|---|
| 20 | 6 | $3.7\times10^{-15}$ | $3.9\times10^{-14}$ |
| 30 | 10 | $4.0\times10^{-14}$ | $2.1\times10^{-13}$ |
| 40 | 14 | $2.5\times10^{-12}$ | $3.5\times10^{-12}$ |

**It reproduces the Legendre polynomials, it does not approximate them**, which is the right
answer: they are *defined* as the Gram-Schmidt orthogonalization of the monomials in this inner
product, so agreement to roundoff is the only acceptable outcome.

**One trap worth stating.** With the trapezoid rule on 201 evenly spaced points instead, the
same code at degree 6 disagrees with Legendre by $8.9\times10^{-3}$. That is **not** an
orthogonalization error, it is the quadrature error: the discrete inner product is a different
inner product, so it has different orthogonal polynomials. Check the discretisation before
blaming the algorithm.

**And note what does not need fixing.** Nothing above forms $M^{1/2}$, and nothing inverts $M$.
Lesson 25's A-inner product in conjugate gradients is the same idea, and lesson 55 builds these
polynomials by the three-term recurrence instead, which is cheaper and better conditioned.

### 3.3 Björck's MGS least squares solver

Run the orthogonalization on the **augmented** matrix, so $Q^T\mathbf{b}$ is produced by the
factorization rather than applied afterwards to a $Q$ that is only orthogonal to $u\kappa$.

```python
def mgs_least_squares(A, b):
    """Modified Gram-Schmidt on [A | b]. The last column of R holds Q^T b."""
    from nalib.lu import back_substitution

    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[1]
    _, R_aug = qr.gram_schmidt_modified(np.column_stack([A, b]))
    return back_substitution(R_aug[:n, :n], R_aug[:n, n])
```

**One setup detail that is easy to get wrong**, and worth stating because it looks like a bug:
$\mathbf{b}$ must have a component **outside** $\operatorname{range}(A)$. If you build
$\mathbf{b} = A\mathbf{x}_{\text{true}}$ exactly, then $[A\mid\mathbf{b}]$ is rank deficient by
construction and any honest Gram-Schmidt refuses it. A consistent system is not a least squares
problem. Build $\mathbf{b} = A\mathbf{x}_{\text{true}} + \mathbf{n}$ with
$\mathbf{n} \perp \operatorname{range}(A)$, and $\mathbf{x}_{\text{true}}$ is still the exact
answer.

**Measured** at $60\times8$, $\sin\theta = 0.287$:

| $\kappa$ | CGS | MGS, forming $Q$ | **MGS on $[A\mid\mathbf{b}]$** | Householder |
|---|---|---|---|---|
| $10^{2}$ | $2.7\times10^{-14}$ | $1.5\times10^{-14}$ | $\mathbf{2.9\times10^{-15}}$ | $3.0\times10^{-15}$ |
| $10^{4}$ | $2.3\times10^{-9}$ | $2.6\times10^{-9}$ | $\mathbf{2.5\times10^{-11}}$ | $5.4\times10^{-11}$ |
| $10^{6}$ | $1.0\times10^{-5}$ | $1.4\times10^{-5}$ | $\mathbf{1.8\times10^{-7}}$ | $1.7\times10^{-7}$ |
| $10^{8}$ | $2.4\times10^{-1}$ | $1.0\times10^{-1}$ | $\mathbf{2.9\times10^{-3}}$ | $1.9\times10^{-3}$ |
| $10^{10}$ | 7.64 | 6.11 | 45.2 | 16.7 |
| $10^{12}$ | 206 | $2.1\times10^{7}$ | $3.0\times10^{5}$ | $1.0\times10^{5}$ |

**It matches Householder to within a factor of 3 at every usable $\kappa$**, while plain MGS
forming $Q$ is 2 to 4 orders of magnitude worse. Above $\kappa = 10^{10}$ nothing works, which
is the problem talking rather than the algorithm.

### 4.1 The two slopes, and the ordering

**The slopes** are measured in lesson 30 itself: classical **1.86**, modified **0.98**, against
theoretical 2 and 1. Fit only over the range where the error is above roundoff and below 1, or
the saturation flattens the classical slope spuriously.

**The ordering.** $\|Q^TQ-I\|$ for classical Gram-Schmidt with the columns permuted:

| $\kappa$ | natural | reversed | random | by norm |
|---|---|---|---|---|
| $10^{4}$ | $3.1\times10^{-11}$ | $6.8\times10^{-10}$ | $1.6\times10^{-10}$ | $7.8\times10^{-11}$ |
| $10^{8}$ | $2.2\times10^{-3}$ | $3.5\times10^{-2}$ | $1.9\times10^{-1}$ | $3.4\times10^{-2}$ |
| $10^{12}$ | 2.8 | 1.8 | 2.0 | 2.5 |

**It matters by a factor of about 20 and no more, and no ordering rescues the method.** At
$\kappa = 10^{4}$ the spread is 22 times, at $10^{8}$ it is 86 times, and at $10^{12}$ every
ordering has lost everything, so the spread collapses to a factor of 1.6 on numbers that all
mean "totally non-orthogonal".

**Two things worth noticing.** Sorting by column norm, the obvious heuristic, is **not** best
here: it is 1.5 times worse than the natural order at $\kappa = 10^{8}$. And the loss is
governed by $\kappa$, which no permutation changes, so reordering can only shift where within
the sweep the damage happens, not how much there is.

**Column ordering does matter when it is chosen by the algorithm rather than fixed in advance.**
That is column-pivoted QR, which reorders by the *remaining* column norms at each step and does
change the numerical rank determination. Lesson 33 builds it.

### 4.2 Timing against the flop counts

$m = 2n$, times in seconds:

| $n$ | CGS | MGS | MGS twice | Householder | Givens | Householder/CGS | flop ratio |
|---|---|---|---|---|---|---|---|
| 40 | 0.0030 | 0.0035 | 0.0053 | **0.0015** | 0.0403 | 0.49 | 0.83 |
| 80 | 0.0143 | 0.0167 | 0.0242 | **0.0046** | 0.1656 | 0.32 | 0.83 |
| 160 | 0.0644 | 0.0778 | 0.1112 | **0.0494** | 0.7668 | 0.77 | 0.83 |

**Householder is fastest at every size**, which surprises people who expect its extra stability
to cost something. The flop ratio explains most of it: $2mn^2 - 2n^3/3$ against $2mn^2$ is 0.83,
so it should be about 17 percent cheaper.

**The measured ratio is 0.32 to 0.77, better than the flop count predicts**, and the
disagreement is the interesting part. Householder's inner loop is a rank-one update on a
shrinking trailing block, which numpy dispatches to a single BLAS call over contiguous memory.
Gram-Schmidt's inner loop is a sequence of dot products and axpys on individual columns, so it
pays Python and dispatch overhead per column. At $n = 160$ the ratio drifts back towards the
flop count as the arrays get large enough for that overhead to stop dominating.

**Givens is 10 to 15 times slower than everything**, and its flop count only says 1.5 times.
That gap is the same effect in the other direction: a Givens QR of a dense matrix is
$O(n^2)$ separate two-row updates, each far too small to amortise a call. It is not a fair
comparison, and exercise 4.3 shows the case where Givens is meant to be used.

**Two passes of MGS cost 1.5 to 1.7 times one pass, not 2**, because the second pass reuses
data already in cache.

### 4.3 Two numerically identical columns of $Q$

**The Läuchli matrix, extended to three columns**, with $\varepsilon = \sqrt{u} = 1.49\times10^{-8}$:

$$A = \begin{pmatrix}1&1&1\\ \varepsilon&0&0\\ 0&\varepsilon&0\\ 0&0&\varepsilon\end{pmatrix}.$$

**Measured, classical Gram-Schmidt:**

| quantity | value |
|---|---|
| $\|\mathbf{q}_2 - \mathbf{q}_3\|$ | **1.000** |
| $\mathbf{q}_2^T\mathbf{q}_3$ | **0.500000** (should be 0) |
| $\|Q^TQ - I\|$ | **0.500** |
| $\|A - QR\|$ | $2.5\times10^{-25}$ |
| $\lvert\operatorname{diag}(R)\rvert$ | $1.000,\; 2.107\times10^{-8},\; 2.107\times10^{-8}$ |

**The mechanism.** $\|\mathbf{a}_j\| = \sqrt{1+\varepsilon^2}$, and $1 + \varepsilon^2 = 1 + u$
**rounds to exactly 1**. So after normalising, $\mathbf{q}_1 = (1,\varepsilon,0,0)^T$ and the
projection of $\mathbf{a}_2$ onto it is computed as exactly 1, giving
$\mathbf{v}_2 = (0,-\varepsilon,\varepsilon,0)^T$. So far so good. But at column 3, classical
computes both coefficients against the **original** $\mathbf{a}_3$, and
$\mathbf{q}_2^T\mathbf{a}_3$ is $0$ to working precision even though $\mathbf{a}_3$ has a real
component along $\mathbf{q}_2$. The subtraction that should have removed it never happens, and
$\mathbf{q}_3$ comes out with the same $-\varepsilon$ in position 2 that $\mathbf{q}_2$ has.
Their inner product is then exactly $1/2$: they share one of their two nonzero entries.

**The $R$ diagonal shows the same thing from the other side.** $r_{22} = r_{33} = 2.1\times10^{-8}$,
both of order $\varepsilon$ rather than of order 1, so the normalisation divides by a quantity
that is entirely roundoff-contaminated.

**And $\|A-QR\|$ reads $2.5\times10^{-25}$.** The factorization identity is perfect while the
factorization is worthless, which is exercise 1.1 made concrete.

**The comparison.** Modified Gram-Schmidt gives $\mathbf{q}_2^T\mathbf{q}_3 =
1.1\times10^{-16}$ and $\|Q^TQ-I\| = 1.2\times10^{-8}$, which is $u\kappa$ as predicted, since
$\kappa(A) \approx 1/\varepsilon$. Householder gives $6.1\times10^{-16}$, unaffected.

### 5.1 The Läuchli matrix by hand

$$A = \begin{pmatrix}1&1\\ \varepsilon&0\\ 0&\varepsilon\end{pmatrix}, \qquad
\varepsilon = \sqrt{u} \approx 1.49\times10^{-8}.$$

**Exactly.** $\|\mathbf{a}_1\| = \sqrt{1+\varepsilon^2}$, so
$\mathbf{q}_1 = (1,\varepsilon,0)^T/\sqrt{1+\varepsilon^2}$ and
$r_{12} = \mathbf{q}_1^T\mathbf{a}_2 = 1/\sqrt{1+\varepsilon^2}$. Then

$$\mathbf{v}_2 = \mathbf{a}_2 - r_{12}\mathbf{q}_1
= \frac{1}{1+\varepsilon^2}\begin{pmatrix}\varepsilon^2\\ -\varepsilon\\ \varepsilon+\varepsilon^3\end{pmatrix},
\qquad \|\mathbf{v}_2\| = \frac{\varepsilon\sqrt{2+\varepsilon^2}}{\sqrt{1+\varepsilon^2}},$$

so exactly, $\mathbf{q}_2 \to (0,-1,1)^T/\sqrt{2}$ as $\varepsilon\to0$, and
$\mathbf{q}_1^T\mathbf{q}_2 = 0$.

**In floating point.** $1+\varepsilon^2 = 1+u$ rounds to **1**. So the computed
$\mathbf{q}_1 = (1,\varepsilon,0)^T$, the computed $r_{12} = 1$, and

$$\hat{\mathbf{v}}_2 = (1,0,\varepsilon)^T - 1\cdot(1,\varepsilon,0)^T = (0,-\varepsilon,\varepsilon)^T,$$

which is right. Normalising gives $\hat{\mathbf{q}}_2 = (0,-1,1)^T/\sqrt{2}$, and

$$\hat{\mathbf{q}}_1^T\hat{\mathbf{q}}_2 = \frac{-\varepsilon}{\sqrt{2}}
\approx -1.05\times10^{-8}.$$

**So at $2\times2$ the loss is $u\kappa \approx \varepsilon$, not total.** The prediction to
check is $\|Q^TQ-I\| \approx 10^{-8}$, and the measurement confirms it: modified gives
$1.2\times10^{-8}$.

**Total loss needs the third column**, which is exercise 4.3, where classical reaches
$\|Q^TQ-I\| = 0.5$. The two-column matrix is the smallest example of *measurable* loss;
the three-column one is the smallest example of *complete* loss, and only classical suffers it.

**What makes $\varepsilon = \sqrt{u}$ the right choice.** It is the largest $\varepsilon$ for
which $1+\varepsilon^2$ still rounds to 1, so it is the mildest matrix that triggers the
failure. Anything smaller triggers it too; anything larger does not.

### 5.2 Why MGS is better for least squares than for orthogonalization

**The mechanism.** Björck and Paige showed that MGS applied to $[A\mid\mathbf{b}]$ is
**equivalent** to Householder QR applied to the augmented matrix
$\begin{pmatrix}0\\ A\end{pmatrix}$ with an $n\times n$ block of zeros on top. Householder QR is
backward stable unconditionally, so the MGS least squares answer inherits that, even though the
$Q$ that MGS produces along the way is only orthogonal to $u\kappa$.

**Why that is not a contradiction.** Backward stability says the computed $\mathbf{x}$ is exact
for a nearby problem. It says nothing about intermediate quantities. The $Q$ is an intermediate
quantity, and its errors are **correlated** with the errors in $R$ in exactly the way that
cancels when you form the solution. Taking $Q$ out and using it for something else, such as
$Q^T\mathbf{b}$ on a $\mathbf{b}$ the factorization never saw, breaks the correlation.

**The experiment that separates the two claims.** Run three solvers on the same $A$ and
$\mathbf{b}$ over a range of $\kappa$, and record both $\|Q^TQ-I\|$ and the solution error:

| $\kappa$ | $\|Q^TQ-I\|$, MGS | MGS forming $Q$, error | MGS on $[A\mid\mathbf{b}]$, error |
|---|---|---|---|
| $10^{4}$ | $\sim10^{-12}$ | $2.6\times10^{-9}$ | $2.5\times10^{-11}$ |
| $10^{6}$ | $\sim10^{-10}$ | $1.4\times10^{-5}$ | $1.8\times10^{-7}$ |
| $10^{8}$ | $\sim10^{-8}$ | $1.0\times10^{-1}$ | $2.9\times10^{-3}$ |

**The middle column tracks the orthogonality loss and the right column does not.** Same
factorization, same $Q$, same $R$; the only difference is whether $Q^T\mathbf{b}$ was computed
during the factorization or afterwards. That is the cleanest possible separation of "the $Q$ is
bad" from "the answer is bad": one experiment, one variable, and the bad $Q$ is present in both
arms.

**The practical rule.** If you need the answer, orthogonalize $[A\mid\mathbf{b}]$. If you need
$Q$ itself, for a projection, a basis, or a second right-hand side, use Householder.

### 5.3 When you cannot revisit a column

**Why Householder is ruled out in Arnoldi.** A Householder reflector $P_k$ is built from
column $k$ of the **partially reduced matrix**, which requires columns $1$ through $k$ to exist
and to have already been transformed. In Arnoldi (lesson 26) the vectors arrive one at a time:
$\mathbf{v}_{k+1} = A\mathbf{q}_k$ cannot be computed until $\mathbf{q}_k$ is finished, and
$\mathbf{q}_k$ is the output of step $k$. There is no matrix to reduce, and $A$ itself may exist
only as a function that multiplies vectors.

**Householder QR needs the matrix; Gram-Schmidt needs only the vectors so far.** That is the
whole reason Krylov methods use Gram-Schmidt despite everything in this lesson.

**What GMRES does instead.** Modified Gram-Schmidt against all previous $\mathbf{q}_i$, with the
coefficients stored as the column of the Hessenberg matrix $H$. When that is not accurate enough
it reorthogonalizes: either always (twice is enough, exercise 2.5) or selectively, when the norm
drops by more than a set factor during the first pass, which is the standard test.

**What it costs.** A Householder-based Arnoldi does exist, Walker's, and it is backward stable.
It stores $k$ reflectors of length $m$ at step $k$, the same as storing the $\mathbf{q}_i$, but
it costs roughly **twice** the flops, because each new vector must be passed through all $k$
previous reflectors and then the new reflector must be constructed. Reorthogonalized MGS costs
the same factor of two and is far simpler, so it is what implementations use.

**And in practice the question is usually moot.** Lesson 27 measured GMRES converging in 25 to
100 steps on problems where $m$ was in the thousands, so $k$ stays small, $\kappa$ of the Krylov
basis stays modest, and one pass of MGS is enough. The stability question only bites in the
restarted and long-recurrence cases.

---

## Lesson 31, Householder and Givens QR

### 1.1 Why $P^{-1} = P$

$P = I - 2\mathbf{v}\mathbf{v}^T/\mathbf{v}^T\mathbf{v}$ is a **reflection** across the
hyperplane perpendicular to $\mathbf{v}$, and reflecting twice returns you to where you started.
Algebraically, with $\mathbf{v}$ normalised,

$$P^2 = (I-2\mathbf{v}\mathbf{v}^T)^2 = I - 4\mathbf{v}\mathbf{v}^T +
4\mathbf{v}(\mathbf{v}^T\mathbf{v})\mathbf{v}^T = I - 4\mathbf{v}\mathbf{v}^T +
4\mathbf{v}\mathbf{v}^T = I.$$

**What it saves.** Three things.

- **No inverse is ever computed or stored.** $Q^{-1} = Q^T = P_n\cdots P_1$, and each $P_k^T =
  P_k$, so applying $Q^{-1}$ means applying the same reflectors in the reverse order. Nothing
  is factored, nothing is solved.
- **$Q$ need not be formed.** Store the $n$ vectors $\mathbf{v}_k$, which is the same memory as
  the data, and apply $Q$ or $Q^T$ on demand in $O(mn)$. Lesson 31 measured the alternative at
  $m = 10^{5}$, $n = 10$: a dense $Q$ is 80 GB for data occupying 8 MB.
- **The identity is checkable at any size.** $Q^TQ\mathbf{y} = \mathbf{y}$ needs no reference
  solution, so it can be verified at $m = 10^{5}$ where forming anything $m\times m$ is
  impossible. Lesson 31 measured $10^{-15}$.

### 1.2 Why one sign destroys the factorization

Both $\alpha = +\|\mathbf{x}\|$ and $\alpha = -\|\mathbf{x}\|$ give
$P\mathbf{x} = \alpha\mathbf{e}_1$ in exact arithmetic. The reflector is
$\mathbf{v} = \mathbf{x} - \alpha\mathbf{e}_1$, so

$$v_1 = x_1 - \alpha.$$

**When $x_1$ and $\alpha$ have the same sign, $v_1$ is a difference of nearly equal numbers**,
because $|x_1| \le \|\mathbf{x}\| = |\alpha|$ and they are equal exactly when $\mathbf{x}$ is
already a multiple of $\mathbf{e}_1$. That is lesson 05's catastrophic cancellation.

**Measured in lesson 31**: at $x_1 = 10^{8}$ with the wrong sign, $v_1$ cancels to **exactly
zero**. The reflector then has no component along $\mathbf{e}_1$ at all, so reflecting across it
**negates the tail instead of removing it**, leaving the entries below the diagonal exactly as
large as before. And nothing reports an error: the code runs, the numbers are finite, and the
factorization is silently wrong.

**The fix is one character.** Choose $\alpha = -\operatorname{sign}(x_1)\|\mathbf{x}\|$, so
$v_1 = x_1 + \operatorname{sign}(x_1)\|\mathbf{x}\|$, a sum of like-signed numbers with no
cancellation at any input. Lesson 31 measured the entries below the diagonal as **exactly zero**
at every input tested, because with the right sign they are assigned rather than computed.

**Both reflections are geometrically valid.** One sends $\mathbf{x}$ to $+\|\mathbf{x}\|\mathbf{e}_1$
and the other to $-\|\mathbf{x}\|\mathbf{e}_1$; the mirror for the second is far from
$\mathbf{x}$, and a mirror far from the thing being reflected is the one you can locate
accurately.

### 1.3 Stability does not have to cost anything

**The reasoning assumes a trade that does not exist here.** Counting flops at $m = n$:

| | flops |
|---|---|
| Gram-Schmidt, either variant | $2n^3$ |
| Householder QR, $R$ only | $4n^3/3$ |

**Householder is cheaper by a third**, and measured, it was fastest at every size in exercise
4.2, by a factor of 1.3 to 3.

**Why it is cheaper.** Gram-Schmidt computes each $\mathbf{q}_j$ by $j-1$ separate projections,
touching the whole of $Q$ so far. Householder does one rank-one update on the trailing block,
which shrinks as it goes: $2(m-k)(n-k)$ flops at step $k$, summing to less.

**Where a real cost appears.** Forming $Q$ explicitly costs another $4(m^2n - mn^2 + n^3/3)$,
whereas Gram-Schmidt produces $Q$ as it goes. So the honest statement is: **Householder is
cheaper if you want $R$, and more expensive if you want a dense $Q$**, and you almost never
want a dense $Q$.

**The general point.** Stability comes from the *structure* of the operations, not from extra
work. Householder is stable because each step applies an exactly orthogonal transformation, and
an orthogonal transformation does not amplify errors, whatever it is applied to. That is a
property of the algorithm's shape, and shape is free.

### 2.1 The reflector's properties

Take $\mathbf{v}$ normalised, $P = I - 2\mathbf{v}\mathbf{v}^T$ (the general case divides by
$\mathbf{v}^T\mathbf{v}$ and the algebra is identical).

- **Symmetric.** $P^T = I^T - 2(\mathbf{v}\mathbf{v}^T)^T = I - 2\mathbf{v}\mathbf{v}^T = P$.
- **Involutive.** $P^2 = I$, computed in exercise 1.1.
- **Orthogonal.** $P^TP = P\cdot P = I$, immediately from the first two. So orthogonality here
  is a *consequence* of symmetry and involution, not a separate check.

**Eigenvalues.** $P\mathbf{v} = \mathbf{v} - 2\mathbf{v}(\mathbf{v}^T\mathbf{v}) = -\mathbf{v}$,
so $-1$ is an eigenvalue with eigenvector $\mathbf{v}$. For any $\mathbf{w}\perp\mathbf{v}$,
$P\mathbf{w} = \mathbf{w} - 2\mathbf{v}(\mathbf{v}^T\mathbf{w}) = \mathbf{w}$, so $+1$ has the
whole orthogonal complement of $\mathbf{v}$ as its eigenspace, of dimension $m-1$. Those
account for all $m$ dimensions, so the spectrum is exactly $\{-1$ once, $+1$ with multiplicity
$m-1\}$.

**Two consequences.** $\det P = -1$, so a reflector is an orthogonal matrix that is not a
rotation, and $\|P\|_2 = 1$ with $\kappa_2(P) = 1$, so it cannot amplify anything. The second is
the stability of the whole method in one line.

### 2.2 The sign and the relative error in $v_1$

**Both signs work.** With $\mathbf{v} = \mathbf{x} - \alpha\mathbf{e}_1$ and
$|\alpha| = \|\mathbf{x}\|$,

$$\mathbf{v}^T\mathbf{v} = \|\mathbf{x}\|^2 - 2\alpha x_1 + \alpha^2 = 2(\|\mathbf{x}\|^2 - \alpha x_1),
\qquad \mathbf{v}^T\mathbf{x} = \|\mathbf{x}\|^2 - \alpha x_1,$$

so $\mathbf{v}^T\mathbf{v} = 2\,\mathbf{v}^T\mathbf{x}$ and

$$P\mathbf{x} = \mathbf{x} - 2\mathbf{v}\frac{\mathbf{v}^T\mathbf{x}}{\mathbf{v}^T\mathbf{v}}
= \mathbf{x} - \mathbf{v} = \alpha\mathbf{e}_1.$$

**Nothing in that used the sign of $\alpha$**, which is why both are valid.

**The relative error for the bad choice.** Write $c = x_1/\|\mathbf{x}\| \in [-1,1]$ and take
$\alpha = +\|\mathbf{x}\|$ with $x_1 > 0$. Then

$$v_1 = x_1 - \|\mathbf{x}\| = \|\mathbf{x}\|(c-1).$$

Both $x_1$ and $\|\mathbf{x}\|$ carry a relative error of order $u$, so the absolute error in
$v_1$ is about $u\|\mathbf{x}\|$, and the relative error is

$$\frac{u\|\mathbf{x}\|}{\|\mathbf{x}\|\,|c-1|} = \frac{u}{1-c}.$$

**This blows up as $c\to1$**, that is, as $\mathbf{x}$ approaches a multiple of $\mathbf{e}_1$,
which is exactly the case that arises whenever a column is already nearly reduced. At
$c = 1-10^{-16}$ the relative error is $O(1)$: every digit gone. With the good sign,
$v_1 = \|\mathbf{x}\|(c+1)$ and the relative error is $u/(1+c) \le u$, bounded for every input.

### 2.3 The flop count

**Step $k$** works on the trailing block of size $(m-k+1)\times(n-k+1)$.

- Building $\mathbf{v}$: $\|\mathbf{x}\|$ costs $2(m-k+1)$, and the subtraction 1. Call it
  $2(m-k)$.
- Applying $P$ as $B \leftarrow B - 2\mathbf{v}(\mathbf{v}^TB)/\mathbf{v}^T\mathbf{v}$: the
  inner product $\mathbf{v}^TB$ is $2(m-k+1)(n-k+1)$, and the rank-one subtraction another
  $2(m-k+1)(n-k+1)$. So $4(m-k)(n-k)$ to leading order.

**Summing** with $j = k$ running $1$ to $n$, and using $\sum j = n^2/2$,
$\sum j^2 = n^3/3$:

$$\sum_{k=1}^{n} 4(m-k)(n-k) = 4\Big(mn^2 - \frac{(m+n)n^2}{2} + \frac{n^3}{3}\Big)
= 2mn^2 - \frac{2n^3}{3},$$

after cancelling. The reflector construction contributes $\sum 2(m-k) = 2mn - n^2$, which is
$O(mn)$ and drops out at leading order.

**Forming $Q$ explicitly** means applying the $n$ reflectors to $I_m$ (or to its first $n$
columns for the reduced form). Applying reflector $k$ to an $m\times m$ matrix costs
$4(m-k)m$, summing to about $4m^2n - 2mn^2$; building only the reduced $\hat{Q}$, applying
backwards from $k = n$ down, costs $4(mn^2 - n^3/3)$.

**At $m = n$**: $R$ alone is $4n^3/3$, forming $Q$ adds another $8n^3/3$, for $4n^3$ total.
**Forming $Q$ costs twice what the factorization does**, and that is the entire reason
implementations return the reflectors.

### 2.4 The orthogonality half of Theorem 31.2

**The claim.** Let $\hat{P}_1,\dots,\hat{P}_n$ be the *computed* reflectors, and
$\hat{Q} = \hat{P}_1\cdots\hat{P}_n$ their product. There exist *exactly* orthogonal
$P_k$ with $\|\hat{P}_k - P_k\| = O(u)$, and hence

$$\|\hat{Q} - Q\| = O(nu), \qquad Q = P_1\cdots P_n \text{ exactly orthogonal.}$$

**Step 1, one reflector.** The computed $\hat{\mathbf{v}}$ differs from the exact
$\mathbf{v}$ by a relative $O(u)$, because with the good sign every operation building it is a
sum of like-signed quantities (exercise 2.2). Define $P_k$ as the **exact** reflector for the
computed $\hat{\mathbf{v}}$: it is exactly orthogonal by construction, whatever
$\hat{\mathbf{v}}$ is, because $I - 2\mathbf{v}\mathbf{v}^T/\mathbf{v}^T\mathbf{v}$ is
orthogonal for any nonzero $\mathbf{v}$. Then the difference between applying $\hat{P}_k$ and
$P_k$ is only the rounding in the two matrix-vector operations, so $\|\hat{P}_k - P_k\| = O(u)$.

**This is the crucial point and it deserves stating separately.** Errors in $\mathbf{v}$ do not
cost orthogonality at all; they only change *which* orthogonal transformation is applied. There
is no $\kappa$ anywhere in that sentence, and that is why the bound is unconditional.

**Step 2, the product.** For orthogonal $P_k$ and perturbations $E_k$ with $\|E_k\| = O(u)$,

$$\prod_k (P_k + E_k) = \prod_k P_k + \sum_k P_1\cdots E_k \cdots P_n + O(u^2),$$

and each term in the sum has norm $O(u)$ because the surrounding $P$'s are orthogonal and
therefore norm-preserving. There are $n$ terms, giving $O(nu)$.

**Contrast with Gram-Schmidt**, where the analogous step divides by $r_{jj}$, a quantity that
can be as small as $\sigma_n$. That division is where $\kappa$ enters, and Householder has no
such division: it only ever multiplies by orthogonal things.

### 2.5 Givens QR of a Hessenberg matrix

**An upper Hessenberg matrix has exactly one nonzero below the diagonal in each of columns $1$
through $n-1$**, namely $h_{k+1,k}$, and nothing below that. So column $k$ needs exactly one
rotation, acting on rows $k$ and $k+1$, to zero $h_{k+1,k}$. That is $n-1$ rotations.

**And no rotation creates new subdiagonal fill.** Rotation $k$ mixes rows $k$ and $k+1$. Row
$k+1$ has zeros in columns $1..k-1$ and row $k$ has zeros in columns $1..k-1$ too (they were
cleared earlier), so the mix leaves those columns at zero. Below row $k+1$ nothing is touched.
So after rotation $k$ the matrix is still Hessenberg in columns $k+1..n$, and the induction runs
to the end.

**The bandwidth claim.** Rotation $k$ replaces row $k$ by a combination of rows $k$ and $k+1$.
If the original had upper bandwidth $p$ (nonzeros in columns $k$ to $k+p$ of row $k$), then row
$k+1$ reaches to column $k+1+p$, so the combined row $k$ reaches to $k+1+p$: **one more**. Each
row is touched by at most two rotations, $k-1$ and $k$, so the total growth is **2**, and the
result is upper triangular with bandwidth $p+2$.

**Why this matters.** It is the reason GMRES is cheap per step. Lesson 27 keeps a Hessenberg
$H_m$ and triangularizes it with exactly $m-1$ Givens rotations, updating the previous
factorization by one rotation rather than refactoring. Measured in lesson 31: at $n = 400$ a
Hessenberg matrix needs $n-1$ rotations against a dense matrix's $n^2/2$, a ratio of $2/n$, so
it is **200 times cheaper**.

### 3.1 Blocked Householder, the WY representation

The product of $b$ reflectors can be written as a single $I - WY^T$, so the trailing update
becomes one matrix product.

```python
def householder_wy(A, block=32):
    """Accumulate `block` reflectors into I - W Y^T and apply them as ONE matrix product.

    Same arithmetic as the unblocked version, regrouped from BLAS-2 into BLAS-3. The recurrence
    is z = tau (v - W (Y^T v)) with the new columns [W, z] and [Y, v], from
    (I - W Y^T)(I - tau v v^T) = I - [W, tau(v - W Y^T v)] [Y, v]^T.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    R = A.astype(float).copy()
    for start in range(0, min(m - 1, n), int(block)):
        stop = min(start + int(block), n)
        W = np.zeros((m - start, stop - start))
        Y = np.zeros((m - start, stop - start))
        for k in range(start, stop):                    # the panel, unblocked
            v, _ = qr.householder_vector(R[k:, k])
            R[k:, k:stop] = qr.apply_householder(v, R[k:, k:stop])
            vv = float(v @ v)
            if vv == 0.0:
                continue
            tau = 2.0 / vv                              # v is UNNORMALIZED, so tau is not 2
            vfull = np.zeros(m - start)
            vfull[k - start:] = v
            j = k - start
            W[:, j] = tau * (vfull - (W[:, :j] @ (Y[:, :j].T @ vfull) if j else 0.0))
            Y[:, j] = vfull
        if stop < n:
            # The panel product is Q = P_start ... P_(stop-1) = I - W Y^T, so what is applied
            # to the trailing columns is Q^T = I - Y W^T. The other way round gives NaN.
            R[start:, stop:] -= Y @ (W.T @ R[start:, stop:])
    return np.triu(R[:n, :]) if m >= n else np.triu(R)
```

**Two details that will bite.** `householder_vector` returns $\mathbf{v}$ **unnormalized**, so
$\tau = 2/\mathbf{v}^T\mathbf{v}$, not 2. And the trailing update needs $Q^T = I - YW^T$, not
$Q = I - WY^T$; getting it backwards produces overflow and NaN, which is the good kind of wrong.

**Measured**, block 32:

| shape | unblocked | blocked | speedup | $\lvert R\rvert$ agrees to |
|---|---|---|---|---|
| $600\times200$ | 0.186 s | **0.015 s** | **12.3** | $8.3\times10^{-16}$ |
| $1200\times300$ | 0.760 s | **0.040 s** | **19.1** | $1.2\times10^{-15}$ |
| $2000\times400$ | 2.534 s | **0.103 s** | **24.5** | $1.7\times10^{-15}$ |

**Up to 24 times faster with identical arithmetic and no loss of accuracy**, and the speedup
grows with size.

**The roofline reading.** The unblocked trailing update is $n$ separate rank-one updates, each
reading and writing the whole trailing block for $O(1)$ flops per element: arithmetic intensity
about 1, firmly memory bound. The blocked version reads the trailing block once and does
$O(b)$ flops per element: intensity about $b = 32$, which moves it past the roofline knee onto
the compute-bound side. This is why LAPACK's `dgeqrf` is blocked and why the speedup keeps
growing: the bigger the matrix, the more of the work sits in the BLAS-3 update rather than in
the BLAS-2 panel.

### 3.2 Complex Householder: the sign becomes a phase

For complex $\mathbf{x}$ the reflector is $P = I - 2\mathbf{v}\mathbf{v}^H/\mathbf{v}^H\mathbf{v}$,
and $\operatorname{sign}(x_1)$ becomes the **phase** $x_1/|x_1|$.

```python
def householder_vector_complex(x):
    """The complex reflector. alpha = -(x_1/|x_1|) ||x||, so v_1 = x_1 + (x_1/|x_1|)||x||
    adds two vectors pointing the same way, exactly as in the real case."""
    x = np.asarray(x, dtype=complex).ravel()
    nx = float(np.linalg.norm(x))
    if nx == 0.0:
        return np.zeros_like(x), 0.0 + 0.0j
    phase = x[0] / abs(x[0]) if x[0] != 0 else 1.0 + 0.0j
    alpha = -phase * nx                        # points AWAY from x, so no cancellation
    v = x.copy()
    v[0] -= alpha
    nv = float(np.linalg.norm(v))
    return (v / nv if nv > 0 else v), alpha
```

**What replaces $\operatorname{sign}(x_1)$** is $x_1/|x_1|$, the unit complex number with the
same argument. For real $x_1$ it reduces to $\pm1$, so the real rule is the special case.

**And $\alpha$ is now complex**, which means $R$ has a complex diagonal. If you want the
positive-real-diagonal convention, multiply row $k$ of $R$ and column $k$ of $Q$ by the
compensating phase afterwards; the factorization is equally valid either way.

**Measured, unitarity and the factorization:**

| shape | $\|Q^HQ - I\|$ | $\|A - QR\|/\|A\|$ | max below the diagonal |
|---|---|---|---|
| $8\times4$ | $7.5\times10^{-16}$ | $3.7\times10^{-16}$ | **exactly 0** |
| $40\times12$ | $1.5\times10^{-15}$ | $4.3\times10^{-16}$ | **exactly 0** |
| $120\times30$ | $4.0\times10^{-15}$ | $9.4\times10^{-16}$ | **exactly 0** |

**And the cancellation, with $\mathbf{x} = (1, 10^{-9}, 10^{-9})^T$**, where $x_1$ is nearly
$\|\mathbf{x}\|$:

| choice | $\lvert v_1\rvert$ before scaling | tail after reflecting |
|---|---|---|
| phase, $\alpha = -(x_1/\lvert x_1\rvert)\|\mathbf{x}\|$ | 2.000 | $5.0\times10^{-28}$ |
| naive, $\alpha = +\|\mathbf{x}\|$ | **exactly 0** | $1.0\times10^{-9}$ |

**The naive choice leaves the tail exactly as large as it started.** $\|\mathbf{x}\|$ rounds to
1.0 and $x_1$ is 1.0, so $v_1 = 0$ to the last bit, the reflector has no $\mathbf{e}_1$
component, and reflecting does nothing useful. The phase choice gives $|v_1| = 2$ and the tail
lands at $10^{-28}$.

### 3.3 QR updating with Givens

Appending a row to $A$ appends a row to $R$; $n$ Givens rotations put it back in triangular
form, and each one touches only two rows.

```python
def append_row(R, new_row):
    """Add a row to a triangular R and re-triangularize with n Givens rotations, O(n^2).

    Refactoring from scratch is O(m n^2). This does not need A at all, only R, which is the
    real saving when m is large: the old data can be discarded.
    """
    R = np.atleast_2d(np.asarray(R, dtype=float)).copy()
    v = np.asarray(new_row, dtype=float).ravel().copy()
    n = R.shape[1]
    if v.size != n:
        raise ValueError(f"the new row has {v.size} entries, R has {n} columns")
    for j in range(n):
        if v[j] == 0.0:
            continue
        c, s = qr.givens_rotation(R[j, j], v[j])
        Rj, vj = R[j, j:].copy(), v[j:].copy()
        R[j, j:] = c * Rj + s * vj
        v[j:] = -s * Rj + c * vj
    return R
```

**Measured**, $m = 3n$:

| $n$ | refactor | update | speedup | $\lvert R\rvert$ agrees to |
|---|---|---|---|---|
| 50 | 0.00026 s | 0.00050 s | **0.5** | $3.8\times10^{-16}$ |
| 100 | 0.00832 s | 0.00096 s | **8.6** | $5.5\times10^{-16}$ |
| 200 | 0.01806 s | 0.00187 s | **9.6** | $6.8\times10^{-16}$ |
| 400 | 0.04609 s | 0.00409 s | **11.3** | $5.9\times10^{-16}$ |

**The update loses at $n = 50$** and wins from $n = 100$ up, reaching 11 times at $n = 400$.
The crossover is the same story as everywhere else in this part: the update is $n$ small
two-row operations in a Python loop, and it takes a while before $O(mn^2)$ of LAPACK-speed work
exceeds $O(n^2)$ of interpreted work. In a compiled implementation the crossover is at $n$ of a
few.

**The saving that does not show in the table** is memory. The update never looks at $A$, only at
$R$, so a streaming fit can keep an $n\times n$ triangle and discard every row after absorbing
it. That is exercise 29.3.3's problem solved without the squared condition number: $O(n^2)$
storage, one pass, and QR accuracy.

**$\lvert R\rvert$ is compared, not $R$.** Signs are a convention, and Givens and Householder do
not make the same choice, so comparing signed entries would report a difference of 2 where there
is none.

### 4.1 The factors drift, the product does not

Built from **known exact factors** $A = Q_0R_0$, so the comparison needs no higher-precision
reference and works at any $\kappa$:

| $\kappa$ | $\|Q^TQ-I\|$ | error in $Q$ | error in $R$ | $\|A - QR\|/\|A\|$ |
|---|---|---|---|---|
| $10^{2}$ | $8.2\times10^{-16}$ | $5.6\times10^{-12}$ | $3.1\times10^{-15}$ | $5.8\times10^{-16}$ |
| $10^{6}$ | $8.0\times10^{-16}$ | $2.3\times10^{-8}$ | $7.8\times10^{-16}$ | $6.8\times10^{-16}$ |
| $10^{10}$ | $8.2\times10^{-16}$ | $2.3\times10^{-4}$ | $4.3\times10^{-16}$ | $9.6\times10^{-16}$ |
| $10^{14}$ | $1.0\times10^{-15}$ | **0.398** | $1.1\times10^{-14}$ | $7.2\times10^{-16}$ |
| $10^{18}$ | $9.3\times10^{-16}$ | **0.613** | $5.3\times10^{-15}$ | $8.5\times10^{-16}$ |

**Three columns are flat and one is not.** Orthogonality is flat at $10^{-15}$ across sixteen
orders of magnitude, the factorization residual is flat at $10^{-16}$, and $R$ is accurate
throughout. **$Q$ drifts like $\kappa$**, from $10^{-12}$ to 0.6.

**Why $Q$ is the one that moves, and why it does not matter.** The columns of $Q$ spanning the
directions where $A$ is nearly rank deficient are barely determined by $A$: a perturbation of
size $u\|A\|$ can rotate them by $u\kappa$. That is not instability, it is the problem's own
sensitivity, and a backward stable algorithm is entitled to return the exact $Q$ of a nearby
matrix. Since $\|A - QR\|$ stays at $10^{-16}$, the errors in $Q$ and $R$ are **correlated**:
whatever $Q$ got wrong, $R$ compensates for exactly.

**And nothing downstream uses them separately.** The solve computes $R\mathbf{x} =
Q^T\mathbf{b}$, and the correlation is preserved through it, which is the content of exercise
5.1.

**Note the $\kappa = 10^{18}$ row is past $1/u$**, so the matrix is numerically singular. The
orthogonality is still $10^{-15}$ there, because it never depended on the matrix at all.

### 4.2 Timings against the flop counts

Answered together with exercise 30.4.2 above: Householder fastest at every size, the measured
Householder/CGS ratio of 0.32 to 0.77 beating the flop-count prediction of 0.83, and Givens 10
to 15 times slower on dense input against a flop count saying 1.5.

**Where BLAS level 3 changes the ranking** is exercise 3.1: the blocked WY Householder is
**12 to 24 times** faster than the unblocked one at identical arithmetic, and the blocked
Gram-Schmidt of exercise 30.3.1 is 6 to 7 times faster than its unblocked version. Once both are
blocked the flop counts govern again, and Householder's $4n^3/3$ against $2n^3$ is decisive.

**The general lesson.** A flop count predicts the ranking only among implementations at the same
level of the memory hierarchy. Comparing a BLAS-2 algorithm with a BLAS-3 one by flops alone
will be wrong by an order of magnitude, in whichever direction the blocking happens to favour.

### 4.3 Givens against Householder as the bandwidth grows

$n = 200$, bandwidth $p$ each side:

| bandwidth | rotations | fraction of dense | Givens | Householder | winner |
|---|---|---|---|---|---|
| 1 | 199 | 0.010 | **0.0059 s** | 0.0234 s | Givens |
| 2 | 397 | 0.020 | **0.0094 s** | 0.0252 s | Givens |
| 5 | 985 | 0.050 | **0.0199 s** | 0.0230 s | Givens |
| **10** | 1945 | 0.098 | 0.0375 s | **0.0236 s** | Householder |
| 25 | 4675 | 0.235 | 0.0846 s | **0.0243 s** | Householder |
| 50 | 8725 | 0.438 | 0.1552 s | **0.0239 s** | Householder |
| 199 | 19900 | 1.000 | 0.3585 s | **0.0233 s** | Householder |

**The crossover is between $p = 5$ and $p = 10$**, at about 5 to 10 percent of the dense
rotation count.

**Two things to read off.** The rotation count is exactly linear in $p$, as theory says: a
banded matrix needs $np$ rotations against $n^2/2$ for dense, so the ratio is $2p/n$, and the
measured fractions 0.010, 0.020, 0.050 for $p = 1,2,5$ match $2p/200$ to two digits.

**And Householder's time does not vary with $p$ at all**, staying at 0.023 to 0.025 s across the
whole range. It does not exploit the structure: it treats a tridiagonal matrix as a dense one
with a lot of zeros in it, and pays the full $2mn^2$.

**So the crossover is not where the flop counts cross.** By flops, Givens should win until
$p \approx n/3$. It wins only to $p \approx 7$, because each rotation is a tiny two-row
operation with per-call overhead, while Householder's work goes to BLAS. In a compiled
implementation the crossover moves much closer to the flop prediction, and the qualitative
answer is the one to remember: **Givens for very narrow structure, Householder otherwise.**

### 5.1 Why the product is accurate when the factors are not

**The result, precisely.** For Householder QR in floating point there exist an exactly
orthogonal $Q_\star$ and a perturbation $\delta A$ with

$$\hat{R} = Q_\star^T(A + \delta A), \qquad \|\delta A\| \le c(m,n)\,u\,\|A\|,$$

and separately $\|\hat{Q} - Q_\star\| \le c(m,n)u$. So the *computed* $\hat{R}$ is the exact
triangular factor of a matrix within $u\|A\|$ of $A$, and the *computed* $\hat{Q}$ is within
$O(u)$ of an exactly orthogonal matrix. **Neither statement says $\hat{Q}$ is close to the
$Q$ of $A$**, and exercise 4.1 measured that it is not: 0.6 at $\kappa = 10^{18}$.

**The three quantities and how to separate them.** Build $A = Q_0R_0$ from known factors, so
each is available exactly:

1. $\|\hat{Q} - Q_0\|$, the error in $Q$. **Grows like $u\kappa$**: $5.6\times10^{-12}$ to
   0.613.
2. $\|\hat{R} - R_0\|/\|R_0\|$, the error in $R$. **Flat at $10^{-15}$** here, though on a
   general matrix it too can move; what is guaranteed is only the backward statement.
3. $\|A - \hat{Q}\hat{R}\|/\|A\|$, the error in the product. **Flat at $10^{-16}$** at every
   $\kappa$.

**Why only the third is bounded independently of $\kappa$.** The map $A \mapsto (Q,R)$ is itself
ill conditioned: two matrices differing by $u\|A\|$ can have $Q$ factors differing by $u\kappa$,
because the direction of a column of $Q$ in a nearly-degenerate subspace is barely determined.
No algorithm can return an accurate $Q$ for such an $A$, since the exact $Q$ is not a
well-defined function of the data at that precision. **The conditioning of the factorization
problem, not the algorithm, is the obstacle.**

The product is different: $\hat{Q}\hat{R}$ reconstructs $A$, and the errors in the two factors
are not independent. They are the same errors seen from two sides, so they cancel. A backward
stable algorithm promises exactly this and nothing more, and it is enough because every use of a
QR factorization applies $Q$ and $R$ in sequence.

**The one case where it is not enough** is when $Q$ is wanted for its own sake, as an
orthonormal basis for $\operatorname{range}(A)$. Then a nearly rank deficient $A$ has no
well-determined answer, and the honest response is the SVD, which reports the numerical rank
rather than pretending there are $n$ well-defined directions. That is lesson 32.

### 5.2 Reflections against rotations, counted three ways

**In flops.** A Householder QR is $2mn^2 - 2n^3/3$; a Givens QR is $3mn^2 - n^3$, so Givens
costs about **1.5 times** as much on a dense matrix. On a banded matrix of half-bandwidth $p$,
Givens needs $np$ rotations at $O(p)$ each, so $O(np^2)$, while Householder ignores the
structure and pays the full dense count. **By flops the crossover is at $p \approx n/3$.**

**In memory traffic.** A Householder step reads and writes the whole trailing block once per
reflector: $O(mn)$ words per step, $O(mn^2)$ total, and blocking (exercise 3.1) reduces it to
$O(mn^2/b)$. A Givens rotation reads and writes two rows: $O(n)$ words for $O(n)$ flops, an
arithmetic intensity of about 1 that **blocking cannot fix**, because the rotations for a
single column are sequentially dependent. **By memory traffic Givens is much worse than 1.5
times**, and the measured factor of 10 to 15 in exercise 4.2 is that, not the flop count.

**In parallel depth.** Here the ranking reverses. Householder reflectors must be applied in
order: $P_k$ is built from a column that $P_{k-1}$ has already modified, so the depth is $n$
sequential steps. Givens rotations acting on **disjoint row pairs commute**, so they can be
scheduled in parallel: the standard wavefront ordering gives depth $O(m+n)$ for a dense matrix
using $O(m)$ processors, and for a Hessenberg or banded matrix the depth is $O(n)$ with each
step cheap. **By parallel depth Givens wins**, and that is why it appears in systolic array and
GPU implementations where latency, not flop count, is the constraint.

**The three answers differ because they measure different resources**, and which one is scarce
depends on the machine and the matrix. That is the actual lesson: "which is cheaper" is not a
well-posed question until you say cheaper in what.

### 5.3 QR of a concatenation and of a product

**Concatenation, $[A\ B]$: yes, cheaply.** Given $A = Q_AR_A$, write

$$[A\ B] = [Q_AR_A\ \ B] = Q_A[R_A\ \ Q_A^TB] + (I - Q_AQ_A^T)[0\ \ B].$$

So: project $B$ onto $\operatorname{range}(A)$ with $C = Q_A^TB$, form the remainder
$B_\perp = B - Q_AC$, factor $B_\perp = Q_2R_2$, and then

$$[A\ B] = [Q_A\ Q_2]\begin{pmatrix}R_A & C\\ 0 & R_2\end{pmatrix}.$$

This is **exactly block Gram-Schmidt** (exercise 30.3.1), and it costs one factorization of
$B$ plus $O(mn_An_B)$, rather than refactoring the whole thing. It is what QR updating,
Arnoldi, and block Krylov methods all do.

**And it has Gram-Schmidt's stability**, so reorthogonalize: $C_2 = Q_A^TB_\perp$ should be
$O(u)$ and is not.

**Product, $AB$: no.** Given $A = Q_AR_A$ and $B = Q_BR_B$,

$$AB = Q_AR_AQ_BR_B,$$

and the middle $R_AQ_B$ is a general matrix with no structure at all. Factoring it costs a full
QR of an $n\times n$ matrix, which is as expensive as factoring $AB$ directly. **There is no
shortcut**, and the reason is structural: triangular times orthogonal is neither.

**Why that means QR is not a general matrix representation.** Compare with LU, where
$A = L_AU_A$ and $B = L_BU_B$ gives $AB = L_A(U_AL_B)U_B$, and $U_AL_B$ is still nothing useful
either. So LU is no better at products. The real difference is elsewhere:

- **LU is closed under the operation it is for.** $A^{-1}$ and solves with many right-hand
  sides come free from one factorization, and that is what a "representation" has to offer.
- **QR costs twice LU** ($4n^3/3$ against $2n^3/3$) for a square solve, so nothing recommends
  it there.
- **QR is not closed under inversion either**: $A^{-1} = R^{-1}Q^T$ is a triangular times an
  orthogonal, the wrong order, so it is an RQ factorization, not a QR.

**What QR is for** is the tall thin case, where LU has no meaning and orthogonality is the whole
point: least squares, orthonormal bases, and the reduction steps inside eigenvalue algorithms
(lessons 35 onward). Within that role its cost is not compared against LU at all.

---

## Lesson 32, Solving Least Squares in Practice

### 1.1 Why the sensitivity to $A$ carries a $\kappa^2$ and the sensitivity to $\mathbf{b}$ does not

**Because perturbing $A$ moves the subspace, and perturbing $\mathbf{b}$ does not.**

A perturbation of $\mathbf{b}$ only changes what is being projected. The projector $P$ is fixed,
$\|P\|_2 = 1$, so the projection moves by at most $\|\delta\mathbf{b}\|$, and recovering
$\mathbf{x}$ from it costs one factor of $\kappa$. Total: **one** $\kappa$.

A perturbation of $A$ does two things. It changes the map from $\mathbf{x}$ to $A\mathbf{x}$,
worth one $\kappa$, and it **tilts $\operatorname{range}(A)$ itself**. Tilting the subspace moves
the projection $P\mathbf{b}$ by an amount proportional to the part of $\mathbf{b}$ that is
*outside* the subspace, which is $\|\mathbf{r}\|$. So the second effect scales like
$\kappa\|\mathbf{r}\|$, and converting that back into a change in $\mathbf{x}$ costs another
$\kappa$. Total: $\kappa + \kappa^2\tan\theta/\eta$, which is the formula in lesson 32 section 1.

**So the $\kappa^2$ term is switched on by $\tan\theta$.** When the residual is zero the subspace
tilt has nothing to act on, the term vanishes, and the problem behaves like a square system with
one $\kappa$. When the residual is large it dominates.

**And this is a property of the problem.** It is not the normal equations' fault, and no
algorithm removes it. The normal equations add a *separate* $\kappa^2$, from squaring the matrix,
which is present even when $\theta = 0$; that one is an artefact and QR does remove it. Two
different $\kappa^2$'s, and only one of them is anybody's fault.

**Measured in lesson 32**: on one matrix with $\kappa = 10^{6}$, the four condition numbers
range over **seven orders of magnitude** as $\theta$ moves.

### 1.2 The minimum norm solution is a choice

**Because the least squares condition does not distinguish between them.** If $A$ is rank
deficient and $\mathbf{z} \in \operatorname{null}(A)$, then for any solution $\mathbf{x}$,

$$\|\mathbf{b} - A(\mathbf{x}+\alpha\mathbf{z})\| = \|\mathbf{b} - A\mathbf{x} -
\alpha A\mathbf{z}\| = \|\mathbf{b} - A\mathbf{x}\|$$

for **every** $\alpha$. The objective is exactly constant along the null space. Measured in
lesson 32: adding any multiple of a null vector leaves the residual identical to six decimals.

**So "minimise $\|\mathbf{b}-A\mathbf{x}\|$" is an underdetermined instruction**, and picking one
answer requires a second criterion that the problem statement did not supply. "Smallest
$\|\mathbf{x}\|$" is one such criterion. It is a good default, because it is what you get in the
limit of vanishing regularization (exercise 2.3) and it is continuous in the data, but it is a
decision.

**Other criteria are equally defensible and give different answers.**

- **Sparsest $\mathbf{x}$**, which is what basis pursuit and lasso choose, and it is usually the
  right one when the columns are candidate explanations and you want few of them.
- **Closest to a prior estimate $\mathbf{x}_0$**, minimising $\|\mathbf{x}-\mathbf{x}_0\|$,
  which is what you want when refitting a model you already had.
- **Smoothest $\mathbf{x}$**, minimising $\|L\mathbf{x}\|$ for a difference operator $L$, which
  exercise 3.2 measures and which beats the minimum norm choice by a factor of 3 when the truth
  has a large mean.

**The practical rule.** If your problem is rank deficient, say in the write-up which solution you
returned and why. `numpy.linalg.lstsq` returns the minimum norm one silently, and a reader who
does not know that will read your coefficients as if they meant something individually.

### 1.3 The SVD was worse than QR at $\kappa = 10^{14}$

**What happened: the threshold discarded a direction the answer needed.** `solve_svd` treats
singular values below `rcond * sigma_max` as zero. At $\kappa = 10^{14}$ with the default
`rcond` of $\max(m,n)\cdot u \approx 1.3\times10^{-14}$, the smallest singular value fell just
below the line. The SVD declared the matrix rank 7 instead of 8, projected the answer onto the
7 surviving directions, and returned the minimum norm solution of a **different problem**.
Lesson 32 measured it three orders of magnitude worse than QR.

**Neither of them was wrong.** They answered different questions.

- **QR** answered "what is the least squares solution of this matrix, taken at face value". It
  kept all 8 directions and got a better answer here, because the true solution genuinely had a
  component along the 8th.
- **The SVD** answered "what is the least squares solution of the nearest matrix of numerical
  rank 7". That is a legitimate and often better question, because a direction with
  $\sigma/\sigma_1 \approx u$ is indistinguishable from a direction the matrix does not have.

**The threshold is a judgement, and it can err in either direction.** Exercise 4.2 sweeps it and
finds the best value here is $10^{-18}$, keeping everything, with error $2.0\times10^{-4}$
against the default's 0.534, a factor of **2600**. On a genuinely rank deficient problem the same
sweep would go the other way and the default would be too small.

**What to actually do.** Look at the singular values. If there is a gap, put the threshold in it
and the choice does not matter. If there is no gap, then there is no rank, the threshold is
choosing your answer for you, and you should be regularizing explicitly and saying so rather than
letting a default do it silently. Exercise 4.3 measures both cases.

### 2.1 The four condition numbers from the SVD

Write $A = U\Sigma V^T$, $\kappa = \sigma_1/\sigma_n$, and

$$\eta = \frac{\|A\|\,\|\mathbf{x}\|}{\|A\mathbf{x}\|}, \qquad
\cos\theta = \frac{\|A\mathbf{x}\|}{\|\mathbf{b}\|}, \qquad
\sin\theta = \frac{\|\mathbf{r}\|}{\|\mathbf{b}\|}.$$

**Sensitivity of $\mathbf{r}$ to $\mathbf{b}$.** $\mathbf{r} = (I-P)\mathbf{b}$ is linear with
$\|I-P\|_2 = 1$, so $\|\delta\mathbf{r}\| \le \|\delta\mathbf{b}\|$. Relatively,

$$\frac{\|\delta \mathbf{r}\|/\|\mathbf{r}\|}{\|\delta\mathbf{b}\|/\|\mathbf{b}\|}
\le \frac{\|\mathbf{b}\|}{\|\mathbf{r}\|} = \frac{1}{\sin\theta}
\quad\text{(the lesson's convention writes this as } 1/\cos\theta\text{ relative to }
\|\mathbf{b}\|).$$

**Sensitivity of $\mathbf{r}$ to $A$: exactly $\kappa$.** The residual depends on $A$ only
through the subspace it spans; a perturbation $\delta A$ rotates that subspace by
$\|\delta A\|/\sigma_n$, and the resulting change in $\mathbf{r}$ is that angle times
$\|A\mathbf{x}\|$. Relative to $\|A\|$ and $\|\mathbf{b}\|$ the factors assemble to $\kappa$
with no $\theta$ and no $\eta$: the residual's sensitivity to the model is the plainest of the
four.

**Sensitivity of $\mathbf{x}$ to $\mathbf{b}$.** $\mathbf{x} = V\Sigma^{-1}U^T\mathbf{b}$, so
$\|\delta\mathbf{x}\| \le \|\delta\mathbf{b}\|/\sigma_n$. Relatively,

$$\frac{\|\mathbf{b}\|}{\sigma_n\|\mathbf{x}\|}
= \frac{\|\mathbf{b}\|}{\|A\mathbf{x}\|}\cdot\frac{\|A\mathbf{x}\|}{\|A\|\|\mathbf{x}\|}
\cdot\frac{\|A\|}{\sigma_n} = \frac{1}{\cos\theta}\cdot\frac{1}{\eta}\cdot\kappa
= \frac{\kappa}{\eta\cos\theta}.$$

**$\theta$ enters through $\|\mathbf{b}\|/\|A\mathbf{x}\|$** and $\eta$ through how much of
$\|A\|\|\mathbf{x}\|$ the product $\|A\mathbf{x}\|$ actually realises.

**Sensitivity of $\mathbf{x}$ to $A$.** Two effects, as in exercise 1.1: the map changes, worth
$\kappa$, and the subspace tilts, worth $\kappa^2\tan\theta/\eta$. Adding,

$$\kappa + \frac{\kappa^2\tan\theta}{\eta}.$$

**$\theta$ enters as $\tan\theta$, which is what switches the $\kappa^2$ on and off**, and
$\eta$ divides it, so a design where $\|A\mathbf{x}\|$ is much smaller than
$\|A\|\|\mathbf{x}\|$ (large $\eta$) is *less* sensitive here, not more.

### 2.2 The pseudoinverse satisfies the four Penrose conditions, uniquely

With $A = U\Sigma V^T$ (thin, rank $r$) and $A^+ = V\Sigma^+U^T$ where $\Sigma^+$ inverts the
nonzero singular values and transposes the shape, note $\Sigma\Sigma^+ = \Sigma^+\Sigma =
\operatorname{diag}(1,\dots,1,0,\dots,0)$ of rank $r$, call it $J$, symmetric with $J^2 = J$.

1. $AA^+A = U\Sigma V^TV\Sigma^+U^TU\Sigma V^T = U\Sigma\Sigma^+\Sigma V^T = U\Sigma V^T = A$,
   using $\Sigma J = \Sigma$.
2. $A^+AA^+ = V\Sigma^+\Sigma\Sigma^+U^T = V\Sigma^+U^T = A^+$, same way.
3. $(AA^+)^T = (UJU^T)^T = UJ^TU^T = UJU^T = AA^+$.
4. $(A^+A)^T = (VJV^T)^T = VJV^T = A^+A$.

**Uniqueness.** Suppose $X$ and $Y$ both satisfy all four. Then

$$X = XAX = X(AX)^T = XX^TA^T = XX^T(AYA)^T = XX^TA^TY^TA^T = X(AX)^T(AY)^T = XAXAY = XAY,$$

using conditions 2, 3, 1, 3 in turn. By the mirror argument with conditions 2, 4, 1, 4,
$Y = XAY$ as well. Hence $X = Y$.

**Which conditions do the work.** Conditions 1 and 2 alone define a *generalised* inverse, and
there are infinitely many. **The two symmetry conditions 3 and 4 are what force uniqueness**,
and geometrically they say that $AA^+$ and $A^+A$ are *orthogonal* projectors rather than
oblique ones. That is the same distinction as exercise 29.2.3: dropping symmetry leaves a
projection that still projects but along a skewed direction, and there is a continuum of those.

**Measured in lesson 32**: all four residuals below $10^{-13}$ for tall, wide, full rank and
rank deficient matrices alike.

### 2.3 Tikhonov as a filter, and its limit

The regularized problem is
$\min \|\mathbf{b}-A\mathbf{x}\|^2 + \lambda^2\|\mathbf{x}\|^2$, whose normal equations are
$(A^TA + \lambda^2I)\mathbf{x} = A^T\mathbf{b}$. Substituting the SVD,

$$(V\Sigma^2V^T + \lambda^2VV^T)\mathbf{x} = V\Sigma U^T\mathbf{b}
\;\Longrightarrow\; V(\Sigma^2+\lambda^2I)V^T\mathbf{x} = V\Sigma U^T\mathbf{b},$$

so multiplying by $V^T$ and inverting the diagonal,

$$\mathbf{x}_\lambda = \sum_i \frac{\sigma_i}{\sigma_i^2+\lambda^2}
(\mathbf{u}_i^T\mathbf{b})\,\mathbf{v}_i
= \sum_i f_i\,\frac{\mathbf{u}_i^T\mathbf{b}}{\sigma_i}\mathbf{v}_i,
\qquad f_i = \frac{\sigma_i^2}{\sigma_i^2+\lambda^2}.$$

**So $1/\sigma_i$ is replaced by $\sigma_i/(\sigma_i^2+\lambda^2)$**, and the filter factor
$f_i$ is near 1 for $\sigma_i\gg\lambda$ and near $\sigma_i^2/\lambda^2$ for
$\sigma_i\ll\lambda$. Large directions pass through; small ones are suppressed quadratically.

**The limit.** For $\sigma_i > 0$, $\sigma_i/(\sigma_i^2+\lambda^2) \to 1/\sigma_i$ as
$\lambda\to0$; for $\sigma_i = 0$ the coefficient is $0$ for every $\lambda$ and stays 0. That
is precisely $\Sigma^+$, so $\mathbf{x}_\lambda \to A^+\mathbf{b}$.

**And in floating point that limit is not attained.** The singular values that should be zero
are actually about $u\|A\|$, so their filter factors start growing again once
$\lambda < \sigma_i$. Measured in `test_leastsquares.py`: the error falls to $2.7\times10^{-8}$
at $\lambda = 10^{-3}$ and is back up to $2\times10^{5}$ by $\lambda = 10^{-10}$. **"As small as
possible" is the wrong instruction**; the useful range is bounded below by about
$\sqrt{u\|A\|}$, and the test suite asserts it.

### 2.4 The augmented system's condition number

Stack $\tilde{A} = \begin{pmatrix}A\\ \lambda I\end{pmatrix}$, which is $(m+n)\times n$. Then

$$\tilde{A}^T\tilde{A} = A^TA + \lambda^2 I = V(\Sigma^2+\lambda^2I)V^T,$$

so the singular values of $\tilde{A}$ are $\sqrt{\sigma_i^2+\lambda^2}$ and

$$\kappa_2(\tilde{A}) = \sqrt{\frac{\sigma_1^2+\lambda^2}{\sigma_n^2+\lambda^2}}.$$

**It is decreasing in $\lambda$**, since the derivative of
$(\sigma_1^2+t)/(\sigma_n^2+t)$ with respect to $t$ is
$(\sigma_n^2-\sigma_1^2)/(\sigma_n^2+t)^2 \le 0$. At $\lambda = 0$ it is $\kappa(A)$; as
$\lambda\to\infty$ it tends to 1. So **any** positive penalty improves the conditioning, and
enough of it makes the problem perfectly conditioned.

**The useful regime is $\sigma_n \ll \lambda \ll \sigma_1$**, where
$\kappa(\tilde{A}) \approx \sigma_1/\lambda$: the penalty replaces the smallest singular value
by $\lambda$ and the condition number becomes $\sigma_1/\lambda$ instead of
$\sigma_1/\sigma_n$.

**And this is why the augmented form is the right way to compute it.** Solving
$(A^TA+\lambda^2I)\mathbf{x} = A^T\mathbf{b}$ directly forms $A^TA$ and pays $\kappa^2$; a QR of
$\tilde{A}$ pays only $\kappa(\tilde{A})$, which is *better* than $\kappa(A)$. It is the same
substitution as the weighted problem in exercise 29.2.4: stack the constraint into the matrix
rather than putting it in the normal equations.

**Note what improves and what does not.** Conditioning improves monotonically in $\lambda$; the
**error** does not, because a large $\lambda$ solves a problem further from the one asked. The
two curves crossing is what the L-curve draws.

### 2.5 The bounds on $\eta$

$\eta = \|A\|\|\mathbf{x}\|/\|A\mathbf{x}\|$ with $\|A\| = \sigma_1$.

**Lower bound.** $\|A\mathbf{x}\| \le \|A\|\|\mathbf{x}\|$ always, so $\eta \ge 1$. Equality
needs $\|A\mathbf{x}\| = \sigma_1\|\mathbf{x}\|$, which happens exactly when $\mathbf{x}$ lies
in the top singular subspace. So **$\eta = 1$ when the solution points along the direction the
matrix stretches most**, and in particular $\eta = 1$ for every $\mathbf{x}$ when $A$ is a
multiple of an orthogonal matrix, where all singular values are equal.

**Upper bound.** $\|A\mathbf{x}\| \ge \sigma_n\|\mathbf{x}\|$, so

$$\eta \le \frac{\sigma_1\|\mathbf{x}\|}{\sigma_n\|\mathbf{x}\|} = \kappa.$$

Equality needs $\mathbf{x}$ in the bottom singular subspace. So **$\eta = \kappa$ when the
solution points along the direction the matrix shrinks most**, which is the worst case, and it
is common in practice: an ill posed problem whose answer lives in the poorly determined
directions is exactly the hard case.

**Reading $\eta$ as a diagnostic.** $\eta$ near 1 means the answer is in the well-determined
part of the space and the problem is easier than $\kappa$ suggests. $\eta$ near $\kappa$ means it
is not. Since $\eta$ divides the $\kappa^2$ term in the sensitivity of $\mathbf{x}$ to $A$, a
large $\eta$ actually *reduces* that term, which is counterintuitive until you notice it also
appears in the denominator of the $\mathbf{b}$ sensitivity, where it helps as well. Measured in
exercise 4.1: $\eta = 8.25\times10^{5}$ on a matrix with $\kappa = 10^{6}$, so the test solution
sat almost entirely in the worst direction.

### 3.1 Generalised cross validation against the L-curve

GCV minimises $\;m\|\mathbf{b}-A\mathbf{x}_\lambda\|^2 / \operatorname{trace}(I - AA_\lambda^+)^2$,
which is leave-one-out cross validation in closed form, and needs no knowledge of the noise
level or the true answer.

```python
def gcv_score(A, b, lam):
    """GCV(lam) = m ||b - A x_lam||^2 / trace(I - A A_lam^+)^2, minimised over lam.

    The trace is m - sum(filter factors), so one SVD serves the whole sweep.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    m = A.shape[0]
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    filt = s ** 2 / (s ** 2 + float(lam) ** 2)
    x = Vt.T @ ((filt / s) * (U.T @ b))
    trace = m - float(np.sum(filt))
    return m * float(np.linalg.norm(b - A @ x)) ** 2 / max(trace ** 2, 1e-300)
```

**Measured** on a discretised first kind integral equation, $80\times40$,
$\kappa = 4.5\times10^{16}$, noise $10^{-6}$:

| chooser | $\lambda$ | relative error |
|---|---|---|
| the best possible, using the true answer | $3.00\times10^{-5}$ | 0.0178 |
| **GCV** | $2.27\times10^{-5}$ | **0.0183** |
| the L-curve corner | $4.30\times10^{-6}$ | 0.0371 |
| no regularization at all | | $9.3\times10^{6}$ |

**Over 20 independent problems: GCV landed closer 16 times, the L-curve 4.** Median error
relative to the best achievable: **GCV 1.25 times, the L-curve 3.90 times**.

**GCV wins, and by a factor of 3.** Both are usable, and both are enormously better than not
regularizing, which is wrong by a factor of $5\times10^{8}$ here. But GCV's median penalty for
not knowing the answer is 25 percent, and the L-curve's is 290 percent.

**Where the L-curve is still preferable.** It is a picture, so you can see whether there is a
corner at all. GCV returns a number whether or not its minimum is meaningful, and on problems
where the noise is correlated rather than white its minimum is known to be flat and unreliable.
Plot the L-curve, then use GCV's $\lambda$.

### 3.2 Tikhonov with a general regularizer

```python
def tikhonov_general(A, b, L, lam):
    """min ||b - Ax||^2 + lam^2 ||L x||^2, through the augmented system [A; lam L]."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    L = np.atleast_2d(np.asarray(L, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    if L.shape[1] != A.shape[1]:
        raise ValueError(f"L has {L.shape[1]} columns, A has {A.shape[1]}")
    stacked = np.vstack([A, float(lam) * L])
    padded = np.concatenate([b, np.zeros(L.shape[0])])
    return np.linalg.lstsq(stacked, padded, rcond=None)[0]


def difference_operator(n, order):
    """The order-th finite difference, as an (n - order) by n matrix. Any n, any order."""
    D = np.eye(int(n))
    for _ in range(int(order)):
        D = D[1:] - D[:-1]
    return D
```

**Measured on the same integral equation.** With a solution that is small as well as smooth, the
three penalties tie:

| regularizer | best $\lambda$ | error | roughness | norm |
|---|---|---|---|---|
| identity | $3.00\times10^{-5}$ | 0.01778 | 0.6767 | 2.5075 |
| first difference | $3.00\times10^{-5}$ | 0.01694 | 0.6766 | 2.5075 |
| second difference | $3.00\times10^{-5}$ | 0.01669 | 0.6765 | 2.5075 |

**A 6 percent spread is not a demonstration of anything**, and this is worth saying because it
is the version most write-ups stop at. The test problem was chosen badly: its truth had
$\|\mathbf{x}\| = 2.51$ and roughness 0.679, comparable numbers, so "small" and "smooth" pointed
the same way.

**Adding a constant offset of 50 separates them**, leaving roughness unchanged at 0.6791 and
raising $\|\mathbf{x}\|$ to 317.8:

| regularizer | best $\lambda$ | error | roughness | norm |
|---|---|---|---|---|
| identity | $6.52\times10^{-6}$ | 0.00043 | 0.7095 | 317.8302 |
| **first difference** | $3.00\times10^{-5}$ | **0.00013** | 0.6766 | 317.8303 |
| **second difference** | $3.00\times10^{-5}$ | **0.00013** | 0.6765 | 317.8303 |

**The difference penalties are 3.3 times better**, and the mechanism is exact: a difference
operator **annihilates a constant**, $D_1\mathbf{1} = \mathbf{0}$, so the offset costs it
nothing at all. The identity penalty must fight the whole of it, and its best $\lambda$ drops by
a factor of 4.6 to compensate, which then under-regularizes everything else. Look at the
roughness column: the identity solution comes out at 0.7095 against a true 0.6791, visibly
rougher, while the difference solutions land at 0.6766.

**Where this is clearly the right choice**, and it is the common case: any inverse problem whose
unknown is a physical field sampled on a grid. Deconvolution, tomography, gravimetry, heat
source recovery. The unknown has an arbitrary datum, and penalising its size penalises the choice
of datum, which is not information. Penalising its roughness penalises what you actually believe
is unlikely.

**A caution.** $L$ has a null space, so $\|L\mathbf{x}\|$ alone does not determine $\mathbf{x}$;
the problem is well posed only because $A$ constrains the rest. If $A$ is also blind to constants
you get an infinite family back, and you need either a second penalty or an explicit constraint.

### 3.3 Golub-Kahan bidiagonalization

Householder from the left clears below the diagonal; Householder from the right clears to the
right of the superdiagonal. Alternating, only two diagonals survive.

```python
def bidiagonalize(A):
    """Householder from both sides until only the diagonal and superdiagonal remain.

    The right-hand reflector starts one column later than the left-hand one, which is exactly
    what leaves the superdiagonal alone and makes the process terminate.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    B = A.astype(float).copy()
    for k in range(n):
        v, _ = qr.householder_vector(B[k:, k])
        B[k:, k:] = qr.apply_householder(v, B[k:, k:])
        B[k + 1:, k] = 0.0
        if k < n - 2:
            w, _ = qr.householder_vector(B[k, k + 1:])
            B[k:, k + 1:] = qr.apply_householder(w, B[k:, k + 1:].T, from_left=True).T
            B[k, k + 2:] = 0.0
    return B[:n, :]
```

**Measured:**

| shape | entries off the bidiagonal | singular values agree to |
|---|---|---|
| $20\times6$ | **exactly 0** | $1.8\times10^{-15}$ |
| $60\times12$ | **exactly 0** | $3.6\times10^{-15}$ |
| $200\times30$ | **exactly 0** | $1.2\times10^{-14}$ |

**The singular values are preserved exactly**, because $B = U^TAV$ with $U$ and $V$ orthogonal,
and orthogonal transformations on either side leave the singular values alone. That identity is
checkable without a reference and is the reason this is how the SVD is actually computed: reduce
to bidiagonal in a finite number of steps, then run an iterative method on the bidiagonal
matrix, where each step is $O(n)$ instead of $O(n^3)$.

**Cost and accuracy against the alternatives**, for solving least squares:

| route | flops | accuracy |
|---|---|---|
| QR | $2mn^2 - 2n^3/3$ | backward stable, no rank information |
| **bidiagonalization** | $4mn^2 - 4n^3/3$ | backward stable, and the bidiagonal is one step from the SVD |
| full SVD | $\approx 4mn^2 + 8n^3$ (Golub-Reinsch) | plus the rank and the filter factors |

**Bidiagonalization costs about twice QR and a fraction of the full SVD**, and it buys the
structure the SVD needs without paying for the SVD's iteration. It is also the basis of LSQR
(the least squares analogue of the conjugate gradient method), which applies the same reduction
one vector at a time and never forms $A$, exactly as in exercise 30.5.3.

**One implementation note.** The right-hand reflector must start at column $k+1$, not $k$. Start
it at $k$ and it destroys the zeros the left-hand reflector just created, and the process never
terminates.

### 4.1 The four condition numbers against $\theta$

$30\times6$, $\kappa = 10^{6}$, $\eta = 8.25\times10^{5}$ held fixed while $\mathbf{b}$ rotates
away from $\operatorname{range}(A)$:

| $\theta$ (degrees) | $\sin\theta$ | $\mathbf{x}$ wrt $\mathbf{b}$ | $\mathbf{x}$ wrt $A$ | $\mathbf{r}$ wrt $\mathbf{b}$ | $\mathbf{r}$ wrt $A$ |
|---|---|---|---|---|---|
| 0.001 | $1.7\times10^{-5}$ | 1.21 | $1.00\times10^{6}$ | 1.00 | $10^{6}$ |
| 0.1 | $1.7\times10^{-3}$ | 1.21 | $1.00\times10^{6}$ | 1.00 | $10^{6}$ |
| 1.0 | $1.7\times10^{-2}$ | 1.21 | $1.02\times10^{6}$ | 1.00 | $10^{6}$ |
| 10 | 0.174 | 1.23 | $1.21\times10^{6}$ | 1.02 | $10^{6}$ |
| 45 | 0.707 | 1.71 | $2.21\times10^{6}$ | 1.41 | $10^{6}$ |
| 80 | 0.985 | 6.98 | $7.88\times10^{6}$ | 5.76 | $10^{6}$ |
| 89.9 | 1.000 | **695** | **$6.96\times10^{8}$** | 573 | $10^{6}$ |

**Four different behaviours, exactly as the formulas say.**

- **$\mathbf{r}$ with respect to $A$ is exactly $\kappa = 10^{6}$ at every angle**, with no
  $\theta$ dependence at all. Constant to every digit shown.
- **$\mathbf{r}$ with respect to $\mathbf{b}$ is $1/\cos\theta$**: 1.00 at small angles, 1.41 at
  45 degrees ($\sqrt{2}$, exactly), 573 at 89.9 degrees.
- **$\mathbf{x}$ with respect to $\mathbf{b}$ is $\kappa/(\eta\cos\theta)$.** With
  $\kappa/\eta = 1.21$, it starts at 1.21 and grows like $1/\cos\theta$, tracking the previous
  row times 1.21 throughout.
- **$\mathbf{x}$ with respect to $A$ is $\kappa + \kappa^2\tan\theta/\eta$.** At small $\theta$
  the second term vanishes and it sits at $\kappa = 10^{6}$; by 89.9 degrees $\tan\theta = 573$
  and it reaches $7\times10^{8}$.

**The range is $10^{6}$ to $7\times10^{8}$ on one axis and 1 to 695 on another**, on the same
matrix. Quoting "the condition number of this least squares problem" without saying which of the
four is meaningless.

**Against the achieved sensitivity**, lesson 32 measured the bound overstating a random
perturbation by a factor of **8 to 24**, which is roughly $\sqrt{m}$: a random direction puts
only $1/\sqrt{m}$ of itself along the worst one. Plan for the bound; expect the measurement.

### 4.2 Sweeping the SVD threshold

$60\times8$, $\kappa = 10^{14}$, consistent right-hand side, default
`rcond` $= \max(m,n)u = 1.33\times10^{-14}$:

| `rcond` | rank kept | relative error |
|---|---|---|
| $10^{-18}$ | 8 | $\mathbf{2.04\times10^{-4}}$ |
| $10^{-16}$ | 8 | $2.04\times10^{-4}$ |
| $10^{-15}$ | 8 | $2.04\times10^{-4}$ |
| $10^{-14}$ | 7 | **0.534**  (the default) |
| $10^{-12}$ | 6 | 0.603 |
| $10^{-10}$ | 5 | 0.722 |
| $10^{-8}$ | 4 | 0.733 |
| $10^{-6}$ | 3 | 0.736 |

**The best value keeps everything, and beats the default by a factor of 2600.**

**The gap explained.** The default is built for the case where you do not know whether the small
singular values are real. Here they are: the problem was constructed with a known
$\mathbf{x}_{\text{true}}$ having a genuine component along the smallest direction, and
$\mathbf{b}$ was formed exactly, so there is no noise for the small directions to amplify. Under
those conditions discarding anything is pure loss.

**Change one thing and the answer reverses.** Add noise at $10^{-6}$ to $\mathbf{b}$ and the
$\sigma \approx 10^{-14}$ direction contributes $10^{-6}/10^{-14} = 10^{8}$ of garbage, so the
default becomes far too generous, not too strict. The threshold trades bias against variance and
the right point depends on the noise level, which `rcond` cannot see.

**What that means in practice.** Do not tune `rcond` against a known answer, because you will not
have one. Estimate the noise level in $\mathbf{b}$, and set the threshold so that
$\sigma_i > \|\delta\mathbf{b}\|/\|\mathbf{x}\|$ roughly, which is the discrete Picard condition
of exercise 5.2 stated as a cutoff. Or regularize with Tikhonov and choose $\lambda$ by GCV,
which does the same thing continuously and picks the level from the data.

### 4.3 A gap and no gap

Both problems $40\times8$, noise $10^{-8}$, and the truth built to **satisfy the Picard
condition** (coefficients decaying with the singular values) so both methods have a real answer
to find.

**With a clear gap**, spectrum $(1,1,1,1,10^{-10},10^{-10},10^{-10},10^{-10})$:

| method | best parameter | error |
|---|---|---|
| truncation | rank 4 | 0.0000 |
| Tikhonov | $\lambda = 1.27\times10^{-4}$ | 0.0000 |

**The two best solutions differ by $1.6\times10^{-8}$.** The filter factors show why:

| | $\sigma_1..\sigma_4$ | $\sigma_5..\sigma_8$ |
|---|---|---|
| Tikhonov | $1.000$ | $6.2\times10^{-13}$ |
| truncation | $1$ | $0$ |

**Tikhonov's filter is a step function whenever there is a gap to put the step in.** $6\times10^{-13}$
is zero for every practical purpose, so the two methods are the same method.

**With no gap**, spectrum geometrically spaced from 1 to $10^{-10}$:

| method | best parameter | error |
|---|---|---|
| truncation | rank 4 | 0.0005 |
| Tikhonov | $\lambda = 1.16\times10^{-5}$ | 0.0006 |

**The two best solutions now differ by $2.9\times10^{-4}$**, which is about half the total error
of either. The filter factors:

| $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Tikhonov | 1.000 | 1.000 | 0.9999 | 0.9526 | 0.0272 | $3.9\times10^{-5}$ | $5.4\times10^{-8}$ | $7.5\times10^{-11}$ |
| truncation | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |

**They disagree at $i = 4$ and $i = 5$, and that is the whole difference.** Tikhonov keeps 95
percent of direction 4 and 2.7 percent of direction 5; truncation keeps all of 4 and none of 5.
There is no rank here, so there is no right answer to the question "how many directions", and the
two methods make different compromises at the boundary.

**Which is better.** Truncation is very slightly better here (0.0005 against 0.0006) but the
difference is inside the noise. The real distinction is that **Tikhonov's answer is a continuous
function of $\lambda$ and truncation's jumps**: moving $\lambda$ by 10 percent moves the answer
by 10 percent, whereas moving the rank by 1 moves the answer discontinuously. When you cannot
determine the parameter precisely, and with no gap you cannot, a continuous dependence is worth
more than a slightly lower optimum you cannot locate.

### 5.1 Tikhonov as the maximum a posteriori estimate

**The derivation.** Take $\mathbf{b} = A\mathbf{x} + \boldsymbol{\varepsilon}$ with
$\boldsymbol{\varepsilon}\sim N(0,\sigma^2I)$ and a prior $\mathbf{x}\sim N(0,\tau^2I)$. By Bayes,

$$p(\mathbf{x}\mid\mathbf{b}) \propto p(\mathbf{b}\mid\mathbf{x})\,p(\mathbf{x})
\propto \exp\left(-\frac{\|\mathbf{b}-A\mathbf{x}\|^2}{2\sigma^2}\right)
\exp\left(-\frac{\|\mathbf{x}\|^2}{2\tau^2}\right).$$

Taking $-2\log$ and dropping constants, the maximum a posteriori estimate minimises

$$\frac{\|\mathbf{b}-A\mathbf{x}\|^2}{\sigma^2} + \frac{\|\mathbf{x}\|^2}{\tau^2}
\;\propto\; \|\mathbf{b}-A\mathbf{x}\|^2 + \left(\frac{\sigma}{\tau}\right)^2\|\mathbf{x}\|^2.$$

**So $\lambda = \sigma/\tau$**, the ratio of the noise level to the prior width.

**Measured**, $40\times12$, $\kappa(A) = 10^{5}$, prior width $\tau = 0.7$, averaged over 40
draws:

| noise $\sigma$ | predicted $\lambda = \sigma/\tau$ | best $\lambda$ | ratio | rel. error at MAP | rel. error at best |
|---|---|---|---|---|---|
| $10^{-1}$ | $1.43\times10^{-1}$ | $1.23\times10^{-1}$ | 0.86 | 0.8968 | 0.8962 |
| $10^{-2}$ | $1.43\times10^{-2}$ | $1.54\times10^{-2}$ | 1.08 | 0.7697 | 0.7689 |
| $10^{-3}$ | $1.43\times10^{-3}$ | $1.65\times10^{-3}$ | 1.15 | 0.6485 | 0.6473 |
| $10^{-4}$ | $1.43\times10^{-4}$ | $2.06\times10^{-4}$ | 1.44 | 0.5261 | 0.5086 |

**The prediction is within a factor of 0.86 to 1.44 of the optimum**, and more importantly the
error *at* the predicted $\lambda$ matches the best achievable to three decimal places. The
optimum is flat near its minimum, so being 44 percent off in $\lambda$ costs 3 percent in error.

**What this says about choosing $\lambda$.** Three things.

- **$\lambda$ is not a tuning knob, it is a statement of belief**, specifically the ratio of how
  noisy you think the data is to how large you think the answer is. If you can estimate those two
  numbers, you do not need the L-curve or GCV at all.
- **Regularizing is not a numerical trick.** It is adding information, and the amount added is
  exactly $\tau$. Saying "I regularized with $\lambda = 10^{-3}$" is saying "I assumed the answer
  is smaller than about $\sigma/10^{-3}$", and that assumption should be defensible.
- **A general $L$ corresponds to a correlated prior**, $\mathbf{x}\sim N(0,(L^TL)^{-1})$, so
  exercise 3.2's difference operator is the belief that the answer is smooth, with a flat prior
  along the constants that $L$ annihilates. That is why it handled the offset for free.

**And the honest footnote.** The relative errors here are 0.51 to 0.90 whatever $\lambda$ you
choose, because at $\kappa = 10^{5}$ with this much noise the data does not determine the answer.
The MAP framework tells you how to weigh the prior; it does not conjure information that is not
there.

### 5.2 The discrete Picard condition

**The statement.** For $\mathbf{x} = \sum_i (\mathbf{u}_i^T\mathbf{b}/\sigma_i)\mathbf{v}_i$ to
be meaningful, the coefficients $|\mathbf{u}_i^T\mathbf{b}|$ must decay to zero **faster** than
the $\sigma_i$ do. Otherwise the ratio grows, the sum is dominated by its last terms, and the
"solution" is whatever noise happens to sit in the smallest singular directions.

**Tested** on the discretised integral equation, $80\times40$, noise $10^{-6}$:

| $i$ | $\sigma_i$ | $\lvert\mathbf{u}_i^T\mathbf{b}\rvert$ | ratio | clean ratio |
|---|---|---|---|---|
| 0 | $3.41\times10^{-1}$ | $6.39\times10^{-1}$ | 1.874 | 1.874 |
| 2 | $2.49\times10^{-1}$ | $7.79\times10^{-2}$ | 0.313 | 0.313 |
| 5 | $8.70\times10^{-2}$ | $4.57\times10^{-2}$ | 0.525 | 0.525 |
| 7 | $3.00\times10^{-2}$ | $7.29\times10^{-3}$ | 0.243 | 0.243 |
| 13 | $2.57\times10^{-4}$ | $2.84\times10^{-5}$ | 0.110 | 0.117 |
| **14** | $9.47\times10^{-5}$ | $9.58\times10^{-7}$ | 0.010 | 0.002 |
| 15 | $3.30\times10^{-5}$ | $2.79\times10^{-6}$ | 0.084 | 0.039 |
| 24 | $2.48\times10^{-10}$ | $2.03\times10^{-6}$ | $8.2\times10^{3}$ | 0.002 |
| 32 | $1.80\times10^{-16}$ | $5.73\times10^{-7}$ | $3.2\times10^{9}$ | 0.173 |

**The condition holds down to $i = 14$ and fails after.** The coefficients decay until they hit
$6.19\times10^{-7}$, which is the noise level, and then **stop**: from there on
$|\mathbf{u}_i^T\mathbf{b}|$ is flat at the noise floor while $\sigma_i$ keeps falling, so the
ratio explodes to $3.2\times10^{9}$.

**The clean column is the control.** With the noise removed, the ratio never grows past 1.874 at
any index. So the failure is entirely the noise, not the operator, and the Picard condition is a
statement about the **data**, not about $A$.

**Finding the turn without knowing the answer**, which is the whole point:

```python
tail = max(1, n // 4)
floor = float(np.median(c[-tail:]))       # the coefficients flatten at the noise level
turn = int(np.argmax(c < 3.0 * floor))    # the first index that reaches it
```

This uses only $U^T\mathbf{b}$, available from the SVD you already computed.

**Where it lands, against the alternatives:**

| criterion | index | error |
|---|---|---|
| the Picard turn | **14** | 0.0240 |
| the best possible rank | 17 | 0.0219 |
| where the L-curve corner's $\lambda$ crosses the spectrum | **17** | |

**The Picard turn is 3 short of optimal and costs 10 percent in error.** And the L-curve corner
lands at **exactly** the best rank, 17.

**So the two are the same diagnostic seen from two sides.** The L-curve plots
$\|\mathbf{x}_\lambda\|$ against $\|\mathbf{r}_\lambda\|$ and looks for the knee; the knee is
where including more directions starts inflating $\|\mathbf{x}\|$ without reducing
$\|\mathbf{r}\|$, which is exactly where $|\mathbf{u}_i^T\mathbf{b}|$ stops decaying faster than
$\sigma_i$. One reads it off the curve, the other off the coefficients, and here they agree to
within 3 indices with the coefficient version being the conservative one.

**The practical value.** Plotting $\sigma_i$ and $|\mathbf{u}_i^T\mathbf{b}|$ on the same log
axis is the single most informative picture you can make of an ill posed problem. If the
coefficients never decay faster than the singular values, **no amount of regularization will
help**, and you need better data rather than a better algorithm.

### 5.3 Why not always use the SVD

**The costs**, for $m\times n$ with $m \ge n$:

| | factorization | each extra right-hand side |
|---|---|---|
| Cholesky on the normal equations | $mn^2 + n^3/3$ | $O(n^2)$ |
| QR | $2mn^2 - 2n^3/3$ | $O(mn)$ |
| SVD (Golub-Reinsch) | $\approx 4mn^2 + 8n^3$ | $O(mn)$ |

**So the SVD costs 2 to 10 times QR**, depending on the aspect ratio, and it is iterative rather
than finite, so its cost is not exactly predictable.

**Where the cost actually matters, case by case.**

- **Many right-hand sides, same $A$.** Here the SVD is **fine**, and often best. The
  factorization is paid once and each solve is $O(mn)$, the same as QR, while the SVD also gives
  you the filter factors so you can regularize each right-hand side differently without
  refactoring. If you are solving hundreds of systems, the factorization cost amortises to
  nothing and the extra information is free.
- **$A$ changes slightly between solves.** Here the SVD is **the wrong choice**, and this is the
  strongest case against it. There is no cheap SVD update: perturbing $A$ by a rank-one term
  changes every singular value and vector, and the standard rank-one SVD update is $O(mn)$ per
  update but numerically delicate and rarely used. QR **does** update cheaply and stably:
  exercise 3.3 appends a row in $O(n^2)$ against $O(mn^2)$, measured at 11 times faster at
  $n = 400$, and there are equally standard updates for deleting a row, appending a column, and
  rank-one modification.
- **$A$ is large and sparse.** Here the SVD is **impossible**, not merely expensive: the
  bidiagonalization fills in completely. Sparse QR with a fill-reducing ordering works, and so do
  the iterative methods (LSQR, LSMR) that never form anything.
- **One solve, moderate size, and you are unsure of the rank.** Here the SVD is **worth it**. A
  factor of 3 in time is nothing against getting the wrong answer, and it is the only one of the
  three that tells you the numerical rank.

**What QR offers that the SVD does not**, gathered:

1. **Cheap, stable updating and downdating.** The decisive one.
2. **A predictable, finite cost.** No iteration, so no convergence question, which matters in
   real-time and embedded settings.
3. **It preserves sparsity**, with the right column ordering.
4. **It handles the streaming case**, absorbing rows one at a time into an $n\times n$ triangle
   and discarding the data (exercise 3.3).
5. **With column pivoting it gives most of the rank information anyway**, at a fraction of the
   cost, which is lesson 33's subject.

**The rule.** Use QR by default. Reach for the SVD when the rank is in question, when you want to
regularize and need the filter factors, or when you are going to look at the singular values as
a diagnostic (exercise 5.2). Use the normal equations only in the streaming case of exercise
29.3.3, and only after checking $\kappa$.

---

## Lesson 33, Rank Revealing QR and Total Least Squares

### 1.1 Why an ordinary QR says nothing about the rank

**Because its diagonal is not sorted, and every rule for reading a rank off it assumes that it
is.** The rule people apply is "walk down the diagonal and stop at the first big drop", and that
rule is only valid on a non-increasing sequence.

Column $j$ is orthogonalized against columns $1$ through $j-1$ **in the order supplied**, so
$r_{jj}$ is small exactly when column $j$ happens to be nearly explained by whichever columns
came before it. Reorder the columns and the small entries move.

**Measured in lesson 33 section 1** on a $40\times6$ matrix of rank 4 with the duplicate at
position 2:

| | diagonal |
|---|---|
| plain QR | $5.14,\ \mathbf{1.16\times10^{-15}},\ 4.73,\ 5.91,\ 5.21,\ 1.48\times10^{-15}$ |
| pivoted QR | $7.82,\ 5.55,\ 4.91,\ 3.71,\ 1.11\times10^{-15},\ 1.33\times10^{-17}$ |

**The cliff in the plain version is at position 2**, so the rule stops there and reports rank 1
for a matrix of rank 4. The pivoted version puts the cliff at 4.

**The information is not lost, only scrambled.** Four entries are large in both. What the plain
diagonal cannot tell you is *which four*, because their positions encode the order you supplied,
and no left-to-right rule recovers from that. You could count large entries instead of walking,
but that needs a threshold, which is exactly what you did not have.

### 1.2 Pivoted QR failed on the Kahan matrix

**Not a bug: a known counterexample, and the reason rank revealing QR is called a heuristic.**

Every column of the Kahan matrix has the same 2-norm, so the pivot search finds a tie at every
step and, taking the first maximum, never swaps. **Measured: zero swaps at $n = 10, 20, 30, 40,
50$.** The algorithm returns exactly what it was given, and the diagonal decays smoothly with no
gap at all, while the smallest singular value is far below the smallest diagonal entry.

At $n = 50$: $|r_{nn}|/|r_{11}| = 8.5\times10^{-21}$ against
$\sigma_n/\sigma_1 = 2.7\times10^{-35}$, so the condition number is understated by a factor of
$3\times10^{14}$. From $n = 24$ the QR and SVD rank verdicts disagree outright.

**What to conclude from a pivoted QR with no gap: nothing.** The guarantee is one-sided and it
points the wrong way for reassurance:

- **A gap in the diagonal IS evidence of rank deficiency.** If $|r_{k+1,k+1}|$ is tiny then
  $\sigma_{k+1} \le |r_{k+1,k+1}|$ is tiny too, because a submatrix's norm bounds a singular
  value from above.
- **No gap is NOT evidence of full rank.** $\sigma_{\min}$ can be far below $|r_{nn}|$, and
  Kahan's matrix shows there is no bound in that direction better than $2^{-n}$.

**So: pivot, look, and reach for the singular values whenever the picture is ambiguous or the
answer matters.** Exercise 3.2 shows there is also a middle option that costs 1.2 times the
plain pivoting and does defeat Kahan.

### 1.3 Better answers and worse conditioning at once

**They are answers to different questions.**

- **"Better" is about the model.** Ordinary least squares assumes all the error is in
  $\mathbf{b}$. When $A$ is measured too, that assumption is false and the estimate is
  **biased**: it converges to $\beta\,s^2/(s^2+e^2)$, not to $\beta$. Total least squares
  assumes the right thing and removes the bias.
- **"Worse conditioned" is about the arithmetic.** The closed form is
  $(A^TA - \sigma_{\min}^2 I)^{-1}A^T\mathbf{b}$, which **subtracts** from the diagonal where
  Tikhonov adds. Measured in lesson 33 section 8: the smallest eigenvalue falls by a factor of
  10.5 and $\kappa$ rises by 5.7.

**Which decides whether to use it: compare the two errors.** The bias is a fixed fraction of the
answer, computable from the noise level; the conditioning penalty amplifies roundoff, which
starts at $10^{-16}$. Measured in lesson 33 section 7, at noise 0.5 in $A$: the bias costs a
factor of 1.25 in the slope, and the conditioning penalty costs about one digit out of sixteen.
**The bias wins by fourteen orders of magnitude**, so use total least squares.

**It reverses when $A$ is nearly rank deficient.** Then the gap between $\sigma_n(A)$ and
$\sigma_{n+1}(C)$ closes, the amplification runs to $10^{4}$ or more, and the removed bias is no
longer the dominant error. Compute the gap and look at it: it costs one SVD.

### 2.1 Pivoting sorts the diagonal, and that is not $|r_{kk}| \ge \sigma_k$

**The sort.** At step $k$ the algorithm chooses the column of the trailing block with the largest
remaining norm, and $|r_{kk}|$ is exactly that norm, since the Householder reflector maps that
column to $\pm\|\cdot\|\mathbf{e}_1$. After step $k$ the reflector is applied to the whole
trailing block, and applying an orthogonal transformation to a set of vectors and then deleting
one coordinate can only **shrink** their norms:

$$c_j^{(k+1)} = \sqrt{(c_j^{(k)})^2 - r_{kj}^2} \le c_j^{(k)}.$$

So the maximum available at step $k+1$ is at most the maximum available at step $k$, which was
$|r_{kk}|$. Hence $|r_{k+1,k+1}| \le |r_{kk}|$ for every $k$.

**It does not imply $|r_{kk}| \ge \sigma_k(A)$**, and Kahan's matrix is the counterexample. At
$n = 40$ pivoting leaves the matrix unchanged, and

$$|r_{40,40}|/|r_{11}| = 1.06\times10^{-16}, \qquad
\sigma_{40}/\sigma_1 = 2.59\times10^{-28},$$

so $|r_{nn}|$ is **twelve orders of magnitude above** $\sigma_n$. The inequality that does hold
is the other one, $\sigma_k \le |r_{kk}|\sqrt{\text{something}}$ in one direction only, which is
exercise 5.1's subject.

### 2.2 The norm downdate

After eliminating column $k$, the remaining part of column $j$ has lost exactly the component
along the new $\mathbf{q}_k$, whose length is $|r_{kj}|$. That component is perpendicular to what
remains, so Pythagoras gives

$$\big(c_j^{\text{new}}\big)^2 = \big(c_j^{\text{old}}\big)^2 - r_{kj}^2,
\qquad\text{that is}\qquad
c_j^{\text{new}} = c_j^{\text{old}}\sqrt{1 - \left(\frac{r_{kj}}{c_j^{\text{old}}}\right)^2}.$$

**It loses accuracy when $|r_{kj}| \to c_j$**, because then the bracket is a difference of nearly
equal numbers: lesson 05's cancellation. If $r_{kj}/c_j = 1-\delta$ then
$1 - (r_{kj}/c_j)^2 \approx 2\delta$, and the relative error in that is about $u/\delta$, so a
column whose norm collapses to a fraction $\sqrt{2\delta}$ of its previous value has its new
norm computed with relative error $u/\delta$.

**And that is the case the algorithm cares about most**, which is the whole problem. A column
whose norm collapses is a column that has just been revealed as dependent, and pivoting exists to
find exactly those. So the cheap update is least reliable precisely where the answer matters.

**The standard fix, which `qr_column_pivoted` uses**, is LINPACK's: keep the original norm
alongside the running one, and recompute from the trailing block whenever the running value has
fallen below a fixed fraction of it. Recomputation costs $O(m-k)$ and happens only on columns
that have collapsed, so the total stays well below the $O(mn^2)$ of recomputing everything.

### 2.3 Every column of the Kahan matrix has the same norm

With $K = D(I - cU)$, $D = \operatorname{diag}(1,s,\dots,s^{n-1})$, $U$ strictly upper
triangular of ones, column $j$ (one-indexed) is

$$\mathbf{k}_j = \big(-cs^{0},\ -cs^{1},\ \dots,\ -cs^{j-2},\ s^{j-1},\ 0,\ \dots,\ 0\big)^T.$$

So

$$\|\mathbf{k}_j\|^2 = c^2\sum_{i=0}^{j-2}s^{2i} + s^{2(j-1)}
= c^2\,\frac{1-s^{2(j-1)}}{1-s^2} + s^{2(j-1)}.$$

Since $c^2 = 1-s^2$, the first term is exactly $1 - s^{2(j-1)}$, and

$$\|\mathbf{k}_j\|^2 = 1 - s^{2(j-1)} + s^{2(j-1)} = 1.$$

**Every column has norm exactly 1**, for every $j$ and every $\theta$. So the pivot search at
step 1 compares $n$ equal numbers.

**The same argument runs at every later step**, because the trailing block of a Kahan matrix
after one elimination is again a Kahan matrix, scaled. So the tie recurs and no swap is ever
made, which is what the measurement shows.

### 2.4 The total least squares solution

**Setting up.** The constraint $(A+\delta A)\mathbf{x} = \mathbf{b}+\delta\mathbf{b}$ says

$$\big([A\mid\mathbf{b}] + [\delta A\mid\delta\mathbf{b}]\big)
\begin{pmatrix}\mathbf{x}\\ -1\end{pmatrix} = \mathbf{0},$$

so the corrected $C = [A\mid\mathbf{b}]$ must have a nontrivial null vector, that is, it must be
**singular**.

**Minimising.** By Eckart-Young (lesson 43), the nearest singular matrix to $C$ in the Frobenius
norm is $C - \sigma_{n+1}\mathbf{u}_{n+1}\mathbf{v}_{n+1}^T$, and the distance is exactly
$\sigma_{n+1}$. Its null space is spanned by $\mathbf{v}_{n+1}$, so the constraint forces

$$\begin{pmatrix}\mathbf{x}\\-1\end{pmatrix} = \alpha\,\mathbf{v}_{n+1}$$

for some scalar. Matching the last entry gives $\alpha = -1/v_{n+1,n+1}$ and hence

$$\mathbf{x}_{\text{TLS}} = -\frac{(\mathbf{v}_{n+1})_{1:n}}{(\mathbf{v}_{n+1})_{n+1}}.$$

**When it fails.** Two conditions, and both must be checked:

1. **$(\mathbf{v}_{n+1})_{n+1} = 0$.** Then the null direction has no component along
   $\mathbf{b}$ and cannot be scaled. This happens exactly when $\mathbf{b}$ is orthogonal to
   the relevant singular direction of $A$.
2. **$\sigma_{n+1}(C) = \sigma_n(A)$.** Then $\sigma_{n+1}$ is not a simple singular value of the
   enlarged problem, $\mathbf{v}_{n+1}$ is not unique, and there is a whole family of solutions
   (or none).

**Golub and Van Loan's clean criterion is $\sigma_n(A) > \sigma_{n+1}(C)$ strictly**, and Cauchy
interlacing guarantees $\ge$ always, so the question is only whether the inequality is strict.

**In floating point, testing condition 1 alone is not enough**, and this is a real bug rather
than a nicety. When $\mathbf{b} \perp \operatorname{range}(A)$ the last entry of
$\mathbf{v}_{n+1}$ is zero exactly and about $10^{-16}$ numerically, which passes a naive test;
dividing by it returned components of order $10^{14}$. `total_least_squares` checks the gap as
well.

### 2.5 The attenuation factor

Take a single predictor, $y = \beta x + \varepsilon$, with $x$ observed as $\tilde{x} = x + e$,
$e$ independent of $x$ and $\varepsilon$, $\operatorname{var}(x) = s^2$,
$\operatorname{var}(e) = e^2$, all means zero.

The ordinary estimate is $\hat\beta = \sum\tilde{x}_iy_i / \sum\tilde{x}_i^2$. By the law of
large numbers, as $m\to\infty$,

$$\frac{1}{m}\sum \tilde{x}_iy_i \to \operatorname{cov}(\tilde{x},y)
= \operatorname{cov}(x+e,\ \beta x+\varepsilon) = \beta s^2,$$

using independence to kill the cross terms, and

$$\frac{1}{m}\sum \tilde{x}_i^2 \to \operatorname{var}(\tilde{x}) = s^2 + e^2.$$

So

$$\hat\beta \to \beta\,\frac{s^2}{s^2+e^2} < \beta.$$

**It is a bias, not a variance**, and the limit is the proof: the estimator converges, and it
converges to the **wrong number**. Variance is what shrinks as $m$ grows; this does not shrink,
because it is in the limit itself.

**Measured in lesson 33 section 7** from $m = 50$ to $m = 20000$ at noise 0.5: the mean slope
sits at 1.594, 1.600, 1.599, 1.598, 1.600 while the standard deviation falls from 0.099 to
0.0054, a factor of 18 which is $\sqrt{400}$. **More data makes the answer more precisely
wrong**, and any confidence interval computed from such a fit will exclude the truth more
confidently at larger $m$.

**The formula is confirmed to three decimals** at every noise level: 0.9905 against 0.9901,
0.9419 against 0.9412, 0.8002 against 0.8000, 0.4986 against 0.5000.

### 3.1 Deleting a column from a pivoted QR

Dropping column $j$ slides every later column one place left, so each keeps its subdiagonal
entry: $R$ becomes **upper Hessenberg** from column $j$ onwards. One Givens rotation per
surviving column clears it, which is lesson 31 exercise 2.5 exactly.

```python
def delete_column(R, j):
    """Remove column j from a triangular R and restore triangular form with Givens rotations.

    O(n^2) against O(m n^2) for a refactorization, and it never touches A.
    """
    R = np.atleast_2d(np.asarray(R, dtype=float))
    n = R.shape[1]
    j = int(j)
    if not 0 <= j < n:
        raise ValueError(f"column {j} is outside 0..{n - 1}")
    H = np.delete(R, j, axis=1).copy()
    rows, cols = H.shape
    for k in range(j, min(cols, rows - 1)):
        if H[k + 1, k] == 0.0:
            continue
        c, s = qr.givens_rotation(H[k, k], H[k + 1, k])
        top, bot = H[k, k:].copy(), H[k + 1, k:].copy()
        H[k, k:] = c * top + s * bot
        H[k + 1, k:] = -s * top + c * bot
    return np.triu(H[:cols, :]) if rows >= cols else np.triu(H)
```

**Measured:**

| shape | column dropped | refactor | update | speedup | $\lvert R\rvert$ agrees to |
|---|---|---|---|---|---|
| $200\times60$ | 20 | 0.00185 s | 0.00044 s | **4.2** | $2.4\times10^{-16}$ |
| $400\times100$ | 33 | 0.00889 s | 0.00076 s | **11.7** | $4.9\times10^{-16}$ |
| $800\times200$ | 66 | 0.01902 s | 0.00145 s | **13.1** | $6.0\times10^{-16}$ |

**Compared with lesson 31 exercise 3.3's row append**, which reached 11.3 times at $n = 400$,
this is the same shape of result for the same reason: $O(n^2)$ of Givens work against
$O(mn^2)$ of refactorization, with a crossover once $m$ is a few times $n$.

**The two together are what makes pivoted QR the tool for subset selection**, because a search
over subsets is a sequence of column deletions and additions, and each one costs $O(n^2)$
instead of a fresh factorization.

### 3.2 Strong rank revealing QR

Keep swapping a chosen column with a rejected one whenever the swap raises $|\det R_{11}|$ by
more than a factor $f$. Gu and Eisenstat showed the gain from swapping column $i$ (chosen) with
column $k+j$ (rejected) is computable in closed form from the blocks of $R$:

```python
def strong_rrqr(A, k, f=2.0, max_swaps=2000):
    """Start from the pivoted order, then swap while it measurably helps.

    The gain matrix is sqrt(W^2 + (rows of R11^{-1})^2 (cols of R22)^2) with W = R11^{-1} R12.
    Every accepted swap raises |det R11| by more than f, and |det R11| is bounded above, so the
    loop terminates.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[1]
    k = int(k)
    if not 1 <= k < n:
        raise ValueError(f"k must be between 1 and {n - 1}, got {k}")
    piv = rrqr.qr_column_pivoted(A).piv.copy()
    for _ in range(int(max_swaps)):
        _, R = np.linalg.qr(A[:, piv])
        R11, R12 = R[:k, :k], R[:k, k:]
        R22 = R[k:, k:] if R.shape[0] > k else np.zeros((0, n - k))
        inv11 = np.linalg.inv(R11)
        W = inv11 @ R12
        col_norms = np.linalg.norm(R22, axis=0) if R22.size else np.zeros(n - k)
        row_norms = np.linalg.norm(inv11, axis=1)
        gain = np.sqrt(W ** 2 + np.outer(row_norms ** 2, col_norms ** 2))
        i, j = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if gain[i, j] <= f:
            break
        piv[[i, k + j]] = piv[[k + j, i]]
    _, R = np.linalg.qr(A[:, piv])
    return piv, R
```

**Measured on the Kahan matrix, with $k = n-1$:**

| $n$ | plain $\lvert r_{kk}\rvert/\lvert r_{11}\rvert$ | **strong** | $\sigma_k/\sigma_1$ | swaps |
|---|---|---|---|---|
| 20 | $1.65\times10^{-8}$ | $\mathbf{1.38\times10^{-13}}$ | $2.69\times10^{-14}$ | **2** |
| 30 | $1.33\times10^{-12}$ | $\mathbf{1.62\times10^{-20}}$ | $2.56\times10^{-21}$ | **2** |
| 40 | $1.06\times10^{-16}$ | $\mathbf{1.90\times10^{-27}}$ | $2.59\times10^{-28}$ | **2** |

**It defeats the Kahan matrix completely.** At $n = 40$ the strong version lands within a factor
of **7.3** of the true smallest singular value, while plain pivoting is off by
$4\times10^{11}$.

**And it costs 1.2 times the plain version** at $n = 60$, because **two swaps** are enough at
every size. That is the practical point: the worst case for strong RRQR is bad, but the typical
case is a handful of swaps, and the matrix everyone cites as pivoting's counterexample needs
exactly two.

**Why the loop terminates.** Every accepted swap multiplies $|\det R_{11}|$ by more than $f > 1$,
and $|\det R_{11}| \le \prod\sigma_i(A)$ is bounded above, so only finitely many swaps are
possible. Gu and Eisenstat's bound is $O(kn\log_f n)$ swaps in the worst case.

### 3.3 Regularized total least squares

The idea is to combine sections 6 and 8 of lesson 32 with section 6 of lesson 33: minimise the
total correction subject to a penalty on $\|\mathbf{x}\|$.

```python
def regularized_tls(A, b, lam, tol=1e-12, max_iter=100):
    """Solve min ||[dA | db]||_F subject to (A+dA)x = b+db and a Tikhonov penalty on x.

    The fixed point form is the cleanest: x solves
        (A^T A - sigma^2 I + lam^2 I) x = A^T b,
    where sigma is the TLS shift and lam is the penalty. Iterate on sigma, which depends on x
    only weakly, and the whole thing is two lines. Note the two shifts have OPPOSITE signs, so
    the penalty is undoing precisely what the bias correction did.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[1]
    shift = float(np.linalg.svd(np.column_stack([A, b]), compute_uv=False)[-1]) ** 2
    G = A.T @ A + (float(lam) ** 2 - shift) * np.eye(n)
    if np.min(np.linalg.eigvalsh(G)) <= 0.0:
        raise np.linalg.LinAlgError(
            f"the penalty lam = {lam:.3e} is too small to keep A^T A - sigma^2 I + lam^2 I "
            f"positive definite; the TLS shift is {np.sqrt(shift):.3e}")
    return np.linalg.solve(G, A.T @ b)
```

**The structure is the whole answer.** $\lambda^2 - \sigma_{\min}^2$ appears as a single shift,
so:

- $\lambda = 0$ gives plain total least squares.
- $\lambda = \sigma_{\min}$ gives **ordinary** least squares exactly, because the two shifts
  cancel.
- $\lambda > \sigma_{\min}$ gives something more damped than ordinary least squares.

**So regularized TLS interpolates between total and ordinary least squares and then past it**,
with $\lambda = \sigma_{\min}$ the exact crossover. That single observation is worth more than
any table: the two techniques are the same one-parameter family seen from two ends.

**Where it beats both**, and it is the case exercise 1.3 identified: $A$ noisy *and* nearly rank
deficient. Then plain TLS removes the bias and amplifies the noise, ordinary least squares does
the reverse, and a $\lambda$ between 0 and $\sigma_{\min}$ does some of each. Choose it by the
L-curve, exactly as lesson 32 section 6 did.

### 4.1 As the rank gap narrows

$60\times10$, true rank 4, with the trailing singular values set to the stated gap:

| gap | QR rank | SVD rank | QR gap index | QR gap ratio |
|---|---|---|---|---|
| $10^{-16}$ | **4** | **4** | 4 | $2.2\times10^{15}$ |
| $10^{-12}$ | 10 | 10 | 4 | $4.8\times10^{11}$ |
| $10^{-8}$ | 10 | 10 | 4 | $5.0\times10^{7}$ |
| $10^{-5}$ | 10 | 10 | 4 | $4.5\times10^{4}$ |
| $10^{-3}$ | 10 | 10 | 4 | $4.5\times10^{2}$ |
| $10^{-2}$ | 10 | 10 | 4 | $4.5\times10^{1}$ |
| $10^{-1}$ | 10 | 10 | 4 | 4.6 |

**Two completely different behaviours in the same table, and separating them is the answer.**

**The rank verdict is useless immediately.** Both methods report rank 4 only when the gap is
below the threshold $\max(m,n)u \approx 1.3\times10^{-14}$, and full rank at every larger gap.
That is not a failure: a matrix whose smallest singular values are $10^{-8}$ **is** full rank at
double precision, and calling it rank 4 would be the error.

**The gap index is right at every level**, from $10^{-16}$ down to $10^{-1}$, and the gap ratio
tracks the gap exactly.

**Where it stops being usable is where the ratio stops being an outlier.** A full rank random
matrix showed largest ratios of 1.8 to 6.7 in lesson 33 section 3. So a ratio of 45 at a gap of
$10^{-2}$ is clearly distinguishable, and a ratio of 4.6 at a gap of $10^{-1}$ is not. **The
crossover is a gap of about $10^{-2}$**, and it is set by the spread of ratios in a matrix with
no structure, not by machine precision.

**The lesson for practice.** `rank` needs a threshold and is honest about it. `rank_gap` needs
none and works four orders of magnitude further, but returns a *ratio* you have to judge. Report
both, and judge the ratio against what a structureless matrix of that size gives.

### 4.2 Greedy against exhaustive, across correlations

50 trials at each level, $40\times9$ choosing 4, with the columns built as rank 3 plus noise:

| noise added | greedy optimal | median $\sigma_{\min}$ ratio | $\sigma_{\min}$ and residual pick the same set |
|---|---|---|---|
| 1.0 | 2/50 | 0.9166 | 4/50 |
| 0.5 | 2/50 | 0.9060 | 3/50 |
| 0.2 | 2/50 | 0.9000 | 1/50 |
| 0.05 | 2/50 | 0.8966 | 1/50 |
| 0.01 | 2/50 | 0.9002 | 1/50 |

**The greedy penalty barely moves with the correlation.** It is 4 percent at every level in the
median, and greedy picks the optimal set about twice in fifty regardless. Whatever drives the
suboptimality, it is not the amount of correlation.

**The second finding is the larger one: the two criteria are different criteria.** Selecting the
subset with the largest $\sigma_{\min}$ and selecting the one with the smallest projection
residual agree in **1 to 4 trials out of 50**. They are not two approximations to one answer;
they are answers to two questions.

- **$\sigma_{\min}$ asks: are the chosen columns well conditioned?** Use it when you will solve
  with them, so that the coefficients mean something individually.
- **The residual asks: do the chosen columns span the rest?** Use it when you want the subset as
  a summary of the whole matrix, which is what feature selection and low rank approximation want.

**Pivoted QR is greedy for neither exactly.** It maximises the *next* $|r_{kk}|$, which is a
one-step approximation to the residual criterion, and it comes within 10 percent of the
$\sigma_{\min}$ optimum as a side effect.

### 4.3 What predicts the total least squares sensitivity

60 problems at $40\times5$ with $\kappa(A)$ spread over four decades, perturbing $A$ and
$\mathbf{b}$ at $10^{-8}$ and measuring the median relative movement of $\mathbf{x}$:

| predictor | fitted slope | correlation |
|---|---|---|
| $\kappa(A)$ | 0.773 | 0.863 |
| $\sigma_1/(\sigma_n(A)-\sigma_{n+1}(C))$ | **0.700** | **0.949** |

**The gap-based quantity wins**, with a correlation of 0.949 against 0.863. That is the theory
confirmed: the TLS solution is read off the singular vector belonging to $\sigma_{n+1}(C)$, and a
singular vector is only as well determined as the separation of its singular value from the rest.

**Both slopes are below 1, and that is the usual story.** A bound is a worst case over
perturbation directions, and a random direction puts only about $1/\sqrt{m}$ of itself along the
worst one, which is exactly lesson 32's finding that the bound overstated a random perturbation
by a factor of 8 to 24. Here the deficit shows up as a slope of 0.70 rather than 1.

**$\kappa(A)$ is not useless, it is just not the right quantity.** It correlates at 0.863
because the two are related: a badly conditioned $A$ often has a small gap. But they come apart,
and lesson 33 section 8 shows the case that matters: $\kappa(A) = 2.0$ with an amplification of
$7.8\times10^{4}$. Nothing about $\kappa(A)$ predicts that row.

### 5.1 Rank revealing is not one property

**The two bounds people want.** For a factorization $AP = QR$ with
$R = \begin{pmatrix}R_{11}&R_{12}\\0&R_{22}\end{pmatrix}$ and $R_{11}$ of size $k$:

$$\text{(i)}\quad \sigma_{\min}(R_{11}) \ge \frac{\sigma_k(A)}{p(k,n)},
\qquad\qquad
\text{(ii)}\quad \sigma_{\max}(R_{22}) \le \sigma_{k+1}(A)\,p(k,n),$$

for some modest function $p$.

**They are different statements.** (i) says the *kept* columns are as well conditioned as the
top $k$ singular values allow: the chosen block is good. (ii) says the *rejected* columns are as
small as the bottom singular values allow: nothing important was thrown away. A factorization
can satisfy one and not the other.

**What column pivoting gives.** It gives easy one-sided facts in the trivial direction:
$\sigma_{\min}(R_{11}) \le \sigma_k(A)$ and $\sigma_{\max}(R_{22}) \ge \sigma_{k+1}(A)$, by
interlacing, since $R_{11}$ is essentially a submatrix. The useful directions (i) and (ii) hold
only with $p(k,n) = O(2^k)$, which is vacuous for any interesting $k$.

**Which one Kahan violates: (ii), and hence the rank verdict.** At $n = 40$ with $k = 39$,
$\sigma_{\max}(R_{22}) = |r_{40,40}| = 1.06\times10^{-16}\cdot|r_{11}|$ while
$\sigma_{40}(A) = 2.59\times10^{-28}\cdot\sigma_1$: the rejected part is $10^{12}$ times larger
than it should be. So the algorithm keeps a direction it should have rejected and reports a rank
too high.

**And that is the direction that matters for rank detection.** Bound (i) protects you against
choosing a badly conditioned subset; bound (ii) protects you against missing a rank deficiency.
Pivoting is decent at the first and has no guarantee for the second.

**Strong RRQR (exercise 3.2) gives both**, with $p(k,n) = O(\sqrt{k(n-k)})$ rather than
$2^k$, which is why it fixes Kahan.

### 5.2 The GPS problem with errors in the satellite positions

**The setup.** Position from pseudoranges is
$\rho_i = \|\mathbf{p}-\mathbf{s}_i\| + b$, four unknowns and $k$ measurements. Lesson 34 solves
it by Gauss-Newton, treating the satellite positions $\mathbf{s}_i$ as exact.

**They are not exact.** Broadcast ephemeris has errors of a metre or two, comparable to the
ranging noise. So this is an errors-in-variables problem, and lesson 33's analysis applies:
ordinary least squares on the linearized system is biased.

**What changes, in the linearized problem.** Each Gauss-Newton step solves
$\min\|J\mathbf{q}+\mathbf{r}\|$, and the rows of $J$ are

$$\Big(\frac{(\mathbf{p}-\mathbf{s}_i)^T}{\|\mathbf{p}-\mathbf{s}_i\|},\ 1\Big),$$

which **depend on $\mathbf{s}_i$**. So a satellite position error perturbs $J$ as well as
$\mathbf{r}$, and that is exactly the errors-in-variables structure.

**The route.** Replace the inner solve with a total least squares solve on
$[J\mid-\mathbf{r}]$, taking care that the last column of $J$ is exactly known (it is all ones,
from the clock bias) so it should **not** be corrected. That is the *mixed* least squares problem:
some columns exact, some measured, and its solution is a constrained TLS.

**What to expect, and why it is small.** The satellite is $2\times10^{7}$ metres away and the
error is a few metres, so the perturbation to the unit vector is of order
$10^{-7}$ relative, while the pseudorange error is a few metres out of $2\times10^{7}$, also
about $10^{-7}$. **The two are comparable**, so the bias is comparable to the noise rather than
dominant, and total least squares is a refinement here rather than a rescue.

**The honest summary**: for GPS the ephemeris error matters and is handled, but by broadcasting
better ephemeris (and by differential corrections) rather than by changing the estimator. The
errors-in-variables framing is the right diagnosis; the practical fix is upstream.

### 5.3 Scaled total least squares

**The assumption plain TLS makes.** Minimising $\|[\delta A\mid\delta\mathbf{b}]\|_F$ treats
every entry of the correction alike, so it assumes the errors in $A$ and in $\mathbf{b}$ have
**the same variance** and are uncorrelated. If $\mathbf{b}$ is measured 100 times more precisely
than $A$, plain TLS will happily move $\mathbf{b}$ as much as $A$, which is throwing away good
information.

**The fix is a scaling.** Minimise $\|[\delta A\mid\delta\mathbf{b}/\gamma]\|_F$ with
$\gamma = \sigma_b/\sigma_A$, which is the same substitution as lesson 29 exercise 2.4's weighted
least squares: divide each part by its own error scale so all corrections are measured in units
of their own uncertainty.

**The substitution, done carefully, because the unscaling is easy to get backwards.** Put
$\mathbf{e} = \delta\mathbf{b}/\gamma$. The constraint
$(A+\delta A)\mathbf{x} = \mathbf{b}+\gamma\mathbf{e}$ divided by $\gamma$ becomes

$$(A+\delta A)\,\frac{\mathbf{x}}{\gamma} = \frac{\mathbf{b}}{\gamma} + \mathbf{e},$$

and the objective $\|[\delta A\mid\mathbf{e}]\|_F$ is now plain TLS on
$(A,\ \mathbf{b}/\gamma)$ with unknown $\mathbf{x}/\gamma$. **So the answer is
$\gamma$ times the scaled solution, not the scaled solution divided by $\gamma$.**

```python
def scaled_tls(A, b, gamma):
    """TLS when the errors in b have standard deviation gamma times those in A.

    Scale b, solve, and MULTIPLY by gamma: the substitution y = x/gamma turns the scaled
    problem into plain TLS on (A, b/gamma), so x = gamma * y. Dividing instead is a real bug
    and a silent one, since gamma = 1 hides it.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    if b.size != A.shape[0]:
        raise ValueError(f"b has {b.size} entries, A has {A.shape[0]} rows")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    out = rrqr.total_least_squares(A, b / float(gamma))
    if out.exists:
        out.x = out.x * float(gamma)
    return out
```

**There are three limits, not two, and naming them correctly is the whole exercise.**

| $\gamma = \sigma_b/\sigma_A$ | what it assumes | what it reduces to |
|---|---|---|
| $\gamma \to 0$ | $\mathbf{b}$ is exact, all the error is in $A$ | **data least squares**: $\min\|\delta A\|$ subject to $(A+\delta A)\mathbf{x}=\mathbf{b}$ |
| $\gamma = 1$ | equal variances, uncorrelated | plain total least squares |
| $\gamma \to \infty$ | $A$ is exact, all the error is in $\mathbf{b}$ | **ordinary least squares** |

**Measured** on a $40\times5$ problem, distance from the scaled solution to each end:

| $\gamma$ | distance to OLS | distance to TLS |
|---|---|---|
| $10^{-4}$ | 8.501 | 4.022 |
| $10^{-2}$ | 8.500 | 4.021 |
| $1$ | 4.493 | **0** |
| $10^{2}$ | $9.2\times10^{-5}$ | 4.493 |
| $10^{4}$ | $\mathbf{9.2\times10^{-9}}$ | 4.493 |

**Ordinary least squares is the large $\gamma$ end**, and the small $\gamma$ end is a third
answer entirely. It is easy to get this backwards by reasoning "small $\gamma$ means
$\mathbf{b}$ barely moves, so it is ordinary least squares", which confuses *which* correction
is being suppressed: ordinary least squares is the one that does not move $A$.

**How wrong plain TLS is when the variances differ by 100.** If $\mathbf{b}$ is measured 100
times more precisely than $A$, then $\gamma = 0.01$ and the right answer is near the **data least
squares** end. Plain TLS assumes $\gamma = 1$, so it spends a hundred times too much of its
correction budget on $\mathbf{b}$, which was already good, and too little on $A$, which was not.
Measured above, the $\gamma = 0.01$ answer is 4.02 away from the plain TLS answer and both are
far from ordinary least squares, so on that problem plain TLS is **neither of the two things you
might have wanted**.

**The general version replaces the scalar by a covariance.** If the errors have covariance
$\Sigma$ across the whole of $[A\mid\mathbf{b}]$, minimise
$\|[\delta A\mid\delta\mathbf{b}]\Sigma^{-1/2}\|_F$, which is **generalised** total least
squares and is again a change of variables followed by the plain algorithm.

**So the rule is the one lesson 29 exercise 2.4 already gave**: weight each measurement by its
own uncertainty, and the algorithm follows unchanged. TLS with $\gamma = 1$ is the special case
where everything happens to be measured equally well, and it is worth checking rather than
assuming.

---

## Lesson 34, Nonlinear Least Squares

### 1.1 Quadratic on one problem, linear on another

**The residual at the solution is what differs**, and it is the only thing that matters here.

The exact Hessian of $f = \tfrac12\|\mathbf{r}\|^2$ is $J^TJ + S$ with
$S = \sum_i r_i\nabla^2 r_i$, and Gauss-Newton drops $S$. So $S$ is **the residual times the
model's curvature**:

- $\mathbf{r}(\mathbf{x}^*) = \mathbf{0}$ makes $S = 0$ at the solution, so Gauss-Newton *is*
  Newton there, and Newton is quadratic.
- $\mathbf{r}(\mathbf{x}^*) \ne \mathbf{0}$ leaves $S$ behind, and the method converges linearly
  with a factor of roughly $\rho\big((J^TJ)^{-1}S\big)$.

**How to tell in advance, before running anything: ask whether the model can fit the data.** If
the model is correct and the data is noiseless, the residual is zero and you will get quadratic
convergence. If there is noise, or the model is wrong, you will not.

**And after one run you can measure it.** `dropped_term` returns $\|S\|/\|J^TJ\|$ from the
current point, and lesson 34 section 3 measured the correspondence directly:

| $\|S\|/\|J^TJ\|$ | 0.000 | 0.048 | 0.117 | 0.175 | 0.238 | 0.280 |
|---|---|---|---|---|---|---|
| iterations | 5 | 8 | 10 | 12 | 14 | 16 |

**At 0.99 it stops converging entirely**: 300 iterations achieving nothing, on the same problem
that takes 12 from a different start.

### 1.2 When damping is a disadvantage

**Damping keeps the iterate near where it started, so it converges to the nearest minimum.**
That is what makes it reliable, and it is the same property that makes it worse when the nearest
minimum is not the best one.

Gauss-Newton's step is unbounded. On a problem with several minima that sometimes throws the
iterate clean out of a poor basin into a better one. Levenberg-Marquardt shortens exactly that
step.

**Measured in lesson 34 section 5**, from the start $(0.1, 4.5)$ on a problem with several
minima:

| | $\|r\|$ | at |
|---|---|---|
| Gauss-Newton | **5.9920** | $(2.331,\ 1.339)$ |
| Levenberg-Marquardt | 12.1568 | $(0.285,\ 4.452)$ |

**Gauss-Newton found a minimum 2.03 times better.** And exercise 4.1 measures it over a whole
grid: Gauss-Newton reaches the best minimum from **45 percent** of 676 starting points and
Levenberg-Marquardt from **20 percent**.

**What to do about it, and it is not to prefer Gauss-Newton.** Neither method searches globally,
and neither can: this is a nonconvex problem. The fix is **multistart**: run from several
starting points and keep the best result. Levenberg-Marquardt is the right method to run from
each of them, because it converges reliably from each; what it does not do is find the basins
for you.

**A cheap and effective source of starting points** is exercise 34.3.2's variable projection,
which reduces the search to the nonlinear parameters alone. In lesson 34 section 8 that turned a
two dimensional search into a one dimensional scan that could simply be plotted.

### 1.3 Why the exact Hessian gives worse answers

**Because $J^TJ$ is positive semidefinite by construction and $J^TJ+S$ is not.**

$J^TJ$ is a Gram matrix, so $\mathbf{v}^TJ^TJ\mathbf{v} = \|J\mathbf{v}\|^2 \ge 0$ for every
$\mathbf{v}$, whatever the problem, at every point. Therefore the Gauss-Newton step
$-(J^TJ)^{-1}J^T\mathbf{r}$ always points downhill.

$S$ carries no such guarantee. It is a sum of Hessians weighted by residuals **of either sign**,
so $J^TJ+S$ can be indefinite, and then Newton's step can point towards a saddle or uphill.

**Measured in lesson 34 section 6** over 200 random points on the offset 6 problem:

| | times indefinite |
|---|---|
| $J^TJ$ | **0 of 200** (it is a Gram matrix, so it cannot be) |
| $J^TJ + S$ | **144 of 200 (72 percent)** |

**And the consequence, at every residual level:**

| offset | GN iters | GN $\|r\|$ | Newton iters | Newton $\|r\|$ |
|---|---|---|---|---|
| 1.0 | 8 | **5.9920** | 9 | 12.2135 |
| 3.0 | 12 | **18.0070** | 7 | 22.8694 |
| 6.0 | 16 | **36.0734** | 7 | 40.9523 |

**Newton takes fewer iterations and arrives somewhere worse, every time.** At offset 3 it lands
at $a \approx 2.8\times10^{-16}$, where the model has collapsed to the zero function and the
residual no longer depends on $b$ at all. The gradient really is zero there, so Newton reports
convergence correctly, to a point nobody wanted.

**So the real argument for Gauss-Newton is not cost.** Dropping $S$ turns an unsafeguarded
Newton method into one whose every step goes downhill. Cheapness is the second reason.

### 2.1 The gradient and the Hessian

Write $f(\mathbf{x}) = \tfrac12\mathbf{r}(\mathbf{x})^T\mathbf{r}(\mathbf{x})$.

**Gradient.** By the chain rule,

$$\frac{\partial f}{\partial x_j} = \sum_i r_i\frac{\partial r_i}{\partial x_j}
= (J^T\mathbf{r})_j, \qquad\text{so}\qquad \nabla f = J^T\mathbf{r}.$$

**Hessian.** Differentiating again,

$$\frac{\partial^2 f}{\partial x_j\partial x_l}
= \sum_i \frac{\partial r_i}{\partial x_l}\frac{\partial r_i}{\partial x_j}
+ \sum_i r_i\frac{\partial^2 r_i}{\partial x_j\partial x_l}
= (J^TJ)_{jl} + S_{jl},$$

with $S = \sum_i r_i\nabla^2 r_i$. **The first term needs only first derivatives; the second
needs $m$ separate Hessians**, which is the cost Gauss-Newton avoids.

**Why $S$ vanishes at a zero residual solution.** Each term carries a factor $r_i(\mathbf{x})$,
and if $\mathbf{r}(\mathbf{x}^*) = \mathbf{0}$ every one is zero regardless of how curved the
model is. **Curvature only hurts in proportion to how badly the model fits.**

**Verified in lesson 34 section 3**: at the solution of an exact fit, $\|J^TJ\| = 22.6$ and
$\|S\| = 0.0$ to machine precision. The gradient identity is checked against a finite difference
of $f$ itself in section 1, agreeing to $1.0\times10^{-10}$.

### 2.2 The Gauss-Newton step is a descent direction

With $J$ of full column rank, $J^TJ$ is positive definite, so it is invertible and
$\mathbf{p} = -(J^TJ)^{-1}J^T\mathbf{r}$ exists. Then

$$\nabla f^T\mathbf{p} = (J^T\mathbf{r})^T\big(-(J^TJ)^{-1}J^T\mathbf{r}\big)
= -\mathbf{z}^T(J^TJ)^{-1}\mathbf{z}, \qquad \mathbf{z} = J^T\mathbf{r}.$$

$(J^TJ)^{-1}$ is positive definite, so that is **strictly negative** unless $\mathbf{z} = 0$,
which is precisely the stationarity condition. So away from a stationary point the step strictly
decreases $f$ to first order.

**Note what this does and does not promise.** It promises a direction, not a length: the
directional derivative is negative, so *some* step along $\mathbf{p}$ decreases $f$, and nothing
says the full step does. That is the gap Levenberg-Marquardt and the line search fill.

**Rank deficient $J$.** Take $\mathbf{r}(x_1,x_2) = (x_1-1,\ x_1-2)^T$, which does not involve
$x_2$ at all. Then

$$J = \begin{pmatrix}1&0\\1&0\end{pmatrix}, \qquad J^TJ = \begin{pmatrix}2&0\\0&0\end{pmatrix},$$

which is singular, so $\mathbf{p}$ is not defined by that formula. **The least squares problem
$\min\|J\mathbf{p}+\mathbf{r}\|$ still has solutions**, a whole line of them, and lesson 32
section 4 already established that choosing among them is a choice. `numpy.linalg.lstsq` takes
the minimum norm one, which here sets the $x_2$ step to zero: a sensible default, and one that
should be stated rather than assumed.

### 2.3 Levenberg-Marquardt is the trust region subproblem

**The subproblem.** Minimise the linear model subject to a step length bound:

$$\min_\mathbf{p}\ \tfrac12\|J\mathbf{p}+\mathbf{r}\|^2
\quad\text{subject to}\quad \|D^{1/2}\mathbf{p}\|\le\Delta.$$

**The Lagrangian.** With multiplier $\lambda/2 \ge 0$,

$$L(\mathbf{p},\lambda) = \tfrac12\|J\mathbf{p}+\mathbf{r}\|^2
+ \tfrac{\lambda}{2}\big(\mathbf{p}^TD\mathbf{p}-\Delta^2\big),$$

and setting $\nabla_\mathbf{p}L = 0$ gives

$$J^T(J\mathbf{p}+\mathbf{r}) + \lambda D\mathbf{p} = 0
\qquad\Longleftrightarrow\qquad (J^TJ+\lambda D)\mathbf{p} = -J^T\mathbf{r},$$

which is exactly the Levenberg-Marquardt equation. The complementarity condition says either
$\lambda = 0$ and the unconstrained step is already inside the ball, or $\lambda > 0$ and the
step lies **on** the boundary.

**The correspondence between $\lambda$ and $\Delta$.** Write $\mathbf{p}(\lambda)$ for the
solution. Then $\|D^{1/2}\mathbf{p}(\lambda)\|$ is a **continuous, strictly decreasing** function
of $\lambda$ on $(0,\infty)$, running from the full Gauss-Newton step length at $\lambda = 0$
down to 0 as $\lambda\to\infty$. So there is exactly one $\lambda$ for each achievable $\Delta$,
and the two parametrisations are equivalent.

**Which explains the damping heuristic.** Raising $\lambda$ when a step fails is shrinking
$\Delta$ without ever naming it, and lowering it when a step works is growing $\Delta$.

**And it also explains what the heuristic is missing**, which exercise 3.1 measures. A real trust
region method compares the **actual** reduction with the one the model **predicted**, and uses
that ratio to size $\Delta$. The damping heuristic only knows whether a step helped, not by how
much relative to its promise.

### 2.4 The asymptotic convergence factor

Near a solution $\mathbf{x}^*$, write $\mathbf{e}_k = \mathbf{x}_k-\mathbf{x}^*$. The
Gauss-Newton iteration is

$$\mathbf{x}_{k+1} = \mathbf{x}_k - (J^TJ)^{-1}J^T\mathbf{r}(\mathbf{x}_k).$$

Expanding $J^T\mathbf{r}$ about $\mathbf{x}^*$, where $J^T\mathbf{r} = 0$, and using
$\nabla(J^T\mathbf{r}) = J^TJ + S$:

$$J^T\mathbf{r}(\mathbf{x}_k) = (J^TJ+S)\mathbf{e}_k + O(\|\mathbf{e}_k\|^2),$$

so

$$\mathbf{e}_{k+1} = \mathbf{e}_k - (J^TJ)^{-1}(J^TJ+S)\mathbf{e}_k + O(\|\mathbf{e}_k\|^2)
= -(J^TJ)^{-1}S\,\mathbf{e}_k + O(\|\mathbf{e}_k\|^2).$$

**So the asymptotic factor is $\rho\big((J^TJ)^{-1}S\big)$**, the spectral radius, all quantities
evaluated at $\mathbf{x}^*$. Three consequences fall straight out:

- $S = 0$ gives $\mathbf{e}_{k+1} = O(\|\mathbf{e}_k\|^2)$: **quadratic**.
- $\rho < 1$ gives linear convergence at that rate.
- $\rho \ge 1$ gives **no convergence**, and the fixed point is not attracting.

**Against the measurements.** $\|S\|/\|J^TJ\|$ is an upper bound on $\rho\big((J^TJ)^{-1}S\big)$
rather than equal to it, so the measured linear factors should sit at or below the ratio column:

| ratio $\|S\|/\|J^TJ\|$ | 0.0482 | 0.1169 | 0.1749 | 0.2377 | 0.2803 |
|---|---|---|---|---|---|
| measured linear factor | 0.0023 | 0.0137 | 0.0307 | 0.0569 | 0.0794 |

**Every measured factor is below its bound, and the two grow together**, roughly as the square
of the ratio over this range. And the prediction that matters is the qualitative one, confirmed
in section 3: at $\rho \approx 0.99$ the iteration stops converging.

### 2.5 The GPS Jacobian, and when it is singular

**The residual** is $r_i(\mathbf{p},b) = \|\mathbf{p}-\mathbf{s}_i\| + b - \rho_i$.

**The position columns.** Differentiating the norm,

$$\frac{\partial}{\partial p_j}\|\mathbf{p}-\mathbf{s}_i\|
= \frac{(\mathbf{p}-\mathbf{s}_i)_j}{\|\mathbf{p}-\mathbf{s}_i\|},$$

so row $i$ of the position block is the **unit vector from satellite $i$ to the receiver**. Its
norm is 1 for every row, whatever the distance: the geometry enters only through direction.

**The bias column.** $\partial r_i/\partial b = 1$ for every $i$, so the last column is all ones.
The bias enters every measurement identically, which is why it is separable from the position
only through the *differences* in direction.

**Singularity when the satellites are coplanar with the receiver.** Suppose all $\mathbf{s}_i$
and $\mathbf{p}$ lie in a plane with unit normal $\mathbf{n}$. Then every $\mathbf{p}-\mathbf{s}_i$
lies in that plane, so every unit vector $\mathbf{u}_i \perp \mathbf{n}$, and

$$J\begin{pmatrix}\mathbf{n}\\0\end{pmatrix}
= \begin{pmatrix}\mathbf{u}_1^T\mathbf{n}\\ \vdots\\ \mathbf{u}_k^T\mathbf{n}\end{pmatrix}
= \mathbf{0}.$$

**$J$ has a null vector, so the problem is singular**, and it is singular in a specific and
interpretable direction: **motion perpendicular to the plane is unobservable**. No amount of data
fixes it, because every measurement is blind to that coordinate.

**And that is exactly the degradation lesson 34 section 7 measured.** Squeezing the satellites
into a narrow cone is approaching the coplanar (indeed collinear) case, and $\kappa(J)$ rises by
a factor of 50 as the cone narrows from 85 degrees to 10. Exercise 4.3 shows the unobservable
direction is the **vertical** one, which is why GPS altitude is always worse than its horizontal
position.

### 3.1 A proper trust region

Choose the step inside a ball of radius $\Delta$, and size $\Delta$ from the ratio of the actual
reduction to the predicted one.

```python
def trust_region(residual, jacobian, x0, delta0=1.0, tol=1e-10, max_iter=200):
    """Trust region Gauss-Newton, with the step found by bisecting on log lambda.

    The quantity the damping heuristic never looks at is rho, the ratio of the ACTUAL reduction
    to the reduction the linear model promised. It says whether the model is trustworthy at this
    radius, which is more information than "did the step help".
    """
    x = np.asarray(x0, dtype=float).ravel()
    r = np.asarray(residual(x), dtype=float).ravel()
    delta = float(delta0)
    for k in range(int(max_iter)):
        J = np.atleast_2d(np.asarray(jacobian(x), dtype=float))
        if (np.linalg.norm(J.T @ r)
                <= tol * max(np.linalg.norm(J, 2) * np.linalg.norm(r), 1e-300)):
            return {"x": x, "iterations": k, "converged": True,
                    "residual_norm": float(np.linalg.norm(r))}
        p_full = np.linalg.lstsq(J, -r, rcond=None)[0]
        if np.linalg.norm(p_full) <= delta:
            p = p_full
        else:
            lo, hi = -14.0, 14.0                    # bisect on log10(lambda)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                stacked = np.vstack([J, np.sqrt(10.0 ** mid) * np.eye(x.size)])
                p = np.linalg.lstsq(stacked, np.concatenate([-r, np.zeros(x.size)]),
                                    rcond=None)[0]
                lo, hi = (mid, hi) if np.linalg.norm(p) > delta else (lo, mid)
        predicted = 0.5 * (float(r @ r) - float((r + J @ p) @ (r + J @ p)))
        trial = np.asarray(residual(x + p), dtype=float).ravel()
        actual = 0.5 * (float(r @ r) - float(trial @ trial))
        rho = actual / predicted if predicted > 0 else -1.0
        if rho > 0.75 and abs(np.linalg.norm(p) - delta) < 1e-8 * delta:
            delta *= 2.0                            # the model is good out to the boundary
        elif rho < 0.25:
            delta *= 0.25                           # the model overpromised
        if rho > 0.0:
            before = float(np.linalg.norm(r))
            x, r = x + p, trial
            after = float(np.linalg.norm(r))
            # ftol and xtol, for the reason nlls.gauss_newton documents: WITHOUT these two this
            # loop ran to 300 iterations on four of the five problems below with the right
            # answer already in hand. The gradient test alone is not a stopping rule.
            if (abs(before - after) <= 1e-14 * max(after, 1.0)
                    or np.linalg.norm(p) <= 1e-14 * max(float(np.linalg.norm(x)), 1.0)):
                return {"x": x, "iterations": k + 1, "converged": True,
                        "residual_norm": after}
        if delta < 1e-14:
            break
    return {"x": x, "iterations": max_iter, "converged": False,
            "residual_norm": float(np.linalg.norm(r))}
```

**Measured against the damping heuristic:**

| problem | LM iters | LM $\|r\|$ | TR iters | TR $\|r\|$ |
|---|---|---|---|---|
| exact fit | 6 | 0.0000 | **5** | 0.0000 |
| offset 3, easy start | 12 | 18.0070 | 12 | 18.0070 |
| offset 10, easy start | **20** | 60.1913 | 41 | 60.1913 |
| **offset 3, hard start** | 264 | 22.8054 | **26** | **22.3549** |
| offset 1, bad start | **15** | 12.1568 | 23 | 12.1568 |

**The hard start is the whole answer: 26 iterations against 264, and a better minimum.** That is
the case the damping heuristic handles worst, because $\|S\|/\|J^TJ\| = 0.99$ there and the
linear model is nearly useless, which is exactly what $\rho$ detects and the accept-or-reject
test does not.

**It is not uniformly better**, and that is worth stating: it loses at offset 10 (41 against 20)
and at the bad start (23 against 15). The extra machinery buys robustness on the hardest problem
and costs a little on the easy ones.

**The bisection is the crude part.** A real implementation uses More's root finder on
$\|\mathbf{p}(\lambda)\| = \Delta$, which converges in two or three steps rather than 60
bisections. That changes the cost per iteration, not the iteration counts above.

### 3.2 Variable projection with the Golub-Pereyra derivative

**The reduced function.** For a model linear in $\mathbf{c}$ and nonlinear in $\mathbf{x}$,
write $\Phi(\mathbf{x})$ for the design matrix. The optimal $\mathbf{c}$ is
$\Phi^+\mathbf{y}$, so the reduced residual is

$$\mathbf{r}_2(\mathbf{x}) = \big(I - \Phi(\mathbf{x})\Phi(\mathbf{x})^+\big)\mathbf{y}
= P^\perp_{\Phi}\mathbf{y}.$$

**The derivative.** Golub and Pereyra showed

$$\frac{\partial\mathbf{r}_2}{\partial x_j}
= -\Big[P^\perp_\Phi\frac{\partial\Phi}{\partial x_j}\Phi^+
+ \big(P^\perp_\Phi\frac{\partial\Phi}{\partial x_j}\Phi^+\big)^T\Big]\mathbf{y}.$$

Kaufman's simplification drops the second (transposed) term, which costs a little in the
convergence rate and saves most of the work, and is what most implementations use.

**Why it converges from further away.** Two reasons, and only the first is usually mentioned.

- **The search space is smaller.** With $n_c$ linear and $n_x$ nonlinear parameters, the search
  is over $n_x$ dimensions instead of $n_c+n_x$.
- **The linear parameters are always at their optimum.** In the full problem, a bad guess for
  $\mathbf{c}$ and a bad guess for $\mathbf{x}$ can conspire to look locally good, creating
  spurious flat regions. In the reduced problem $\mathbf{c}$ is never wrong, so those regions do
  not exist.

**Measured in lesson 34 section 8** on the two exponential fit: the reduced scan over 200 points
plus a one dimensional polish found $b$ to **ten digits** with no starting point at all, while
Gauss-Newton on the full problem from three far starts found the answer once and **overflowed
twice**.

**The natural extension is two exponentials**, $c_1e^{b_1t}+c_2e^{b_2t}$, which is notoriously
hard: the reduced problem is 2 dimensional instead of 4, and its basin is much larger. That is
worth implementing because the full problem is where people actually get stuck.

### 3.3 Robust nonlinear least squares

Replace the squared loss with the Huber loss, which is quadratic near zero and linear far out,
and solve by iteratively reweighted Gauss-Newton.

```python
def huber_weights(r, delta):
    """1 inside the elbow, delta/|r| outside, so a large residual contributes linearly."""
    a = np.abs(np.asarray(r, dtype=float))
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-300))


def huber_gauss_newton(residual, jacobian, x0, delta=1.0, max_iter=100, tol=1e-10):
    """Each step is a WEIGHTED linear least squares problem, solved by scaling the rows, which
    is lesson 29 exercise 2.4's routine unchanged."""
    x = np.asarray(x0, dtype=float).ravel()
    for _ in range(int(max_iter)):
        r = np.asarray(residual(x), dtype=float).ravel()
        J = np.atleast_2d(np.asarray(jacobian(x), dtype=float))
        w = np.sqrt(huber_weights(r, delta))
        p = np.linalg.lstsq(J * w[:, None], -r * w, rcond=None)[0]
        x = x + p
        if np.linalg.norm(p) <= tol * max(np.linalg.norm(x), 1.0):
            break
    return x
```

**Measured**, 60 points from $y = 2e^{-0.7t}$ with noise 0.02 and **four gross outliers**, over
40 trials:

| | median relative error |
|---|---|
| plain least squares | 0.1586 |
| **Huber, $\delta = 0.1$** | **0.0061** |

**Huber is closer in 100 percent of trials, by a median factor of 29.1.**

**And with no outliers it costs nothing**: the two agree to a median factor of **1.000**. That is
the check that matters, because a robust method that sacrifices efficiency on clean data is a
poor trade. Huber's elbow means it *is* least squares wherever the residuals are small, so on
clean data every weight is 1 and the two are the same algorithm.

**Choosing $\delta$.** It has to sit above the honest noise and below the outliers. The standard
rule is $\delta = 1.345\hat\sigma$ with $\hat\sigma$ a robust scale estimate such as the median
absolute deviation over 0.6745, which gives 95 percent of least squares' efficiency on clean
Gaussian data. Here the noise is 0.02 and the outliers are of order 5, so anything from 0.05 to
0.5 works and $\delta = 0.1$ was not tuned.

**Compare with lesson 29 exercise 5.3**, which measured the same phenomenon in the linear case:
$L_1$ barely moved by an outlier that pulled $L_2$'s intercept out by 14 times as much. Huber is
the compromise between those two, and it is the usual default.

### 4.1 Basins of attraction

A $26\times26$ grid of starting points, 676 in total, on the offset 1 problem, each run to
convergence and scored by the residual it reached:

| method | reached the best minimum | blew up |
|---|---|---|
| Gauss-Newton | **45 percent** | 0 percent |
| Levenberg-Marquardt | 20 percent | 0 percent |

**Gauss-Newton finds the best minimum from more than twice as many starts**, which is the
opposite of the usual summary and it is the measured result.

**Neither blew up on this problem**, which is worth saying because the usual argument for damping
is that Gauss-Newton diverges. On this problem it does not; on the exponential fit of section 8
it does, from two of three far starts. So the divergence risk is real but problem dependent, and
it is not what separates the two here.

**The mechanism, once more.** Damping shortens the step, so the iterate stays near where it
started and settles into the nearest basin. The undamped step is long and sometimes lands in a
better one. Neither is searching; one is just noisier, and noise helps when the starting point is
poor.

**What a picture of this shows** that the percentages do not: the Gauss-Newton basin is
**fragmented**, with islands of good starting points scattered through bad regions, because
whether a long step lands well is close to arbitrary. The Levenberg-Marquardt basin is a single
connected region around the best minimum. **So Gauss-Newton's larger success rate is less
reliable**, in the sense that a small change in the starting point can flip it, and that is an
argument for multistart with a damped method rather than for undamped steps.

### 4.2 Fitting the iteration count

Theory (exercise 2.4) says linear convergence at factor $\rho$ needs about
$\log\varepsilon/\log\rho$ iterations to reach tolerance $\varepsilon$. Taking
$\rho = \|S\|/\|J^TJ\|$ as the proxy and $\varepsilon = 10^{-12}$:

| | value |
|---|---|
| problems used | 18, with the ratio from 0.017 to 0.333 |
| fitted slope of iterations against $\log\varepsilon/\log\rho$ | **0.663** |
| correlation | **0.985** |

**A correlation of 0.985 across 18 problems**, so the functional form is right.

**The slope of 0.663 rather than 1 is expected**, because $\|S\|/\|J^TJ\|$ is an upper bound on
the true factor $\rho\big((J^TJ)^{-1}S\big)$, not equal to it, so the formula overestimates the
work and the fitted slope absorbs the difference.

**The held out test, which is the part that matters:**

| | |
|---|---|
| held out problem | offset 7.5, ratio 0.310 |
| predicted | **16** iterations |
| actual | **17** iterations |

**Predicted 16, got 17**, on a problem the fit never saw. That is what makes this a model rather
than a curve through some points.

### 4.3 GPS error against the geometry

30 configurations per row, 8 satellites confined to a cone about the vertical, ranging noise held
at 5 metres throughout, medians reported:

| cone half-angle | median $\kappa(J)$ | horizontal error | vertical error | ratio |
|---|---|---|---|---|
| 85 degrees | 10.8 | 5.13 m | 7.46 m | 1.45 |
| 60 degrees | 24.2 | 6.80 m | 14.61 m | 2.15 |
| 40 degrees | 59.2 | 9.60 m | 33.26 m | 3.47 |
| 25 degrees | 145.7 | 15.19 m | 78.00 m | 5.14 |
| 12 degrees | 676.9 | 33.46 m | **358.59 m** | **10.72** |

**Both errors track $\kappa(J)$**, rising by factors of 6.5 and 48 as $\kappa$ rises by 63.

**The vertical error is always the worse of the two, and the gap widens.** At 85 degrees it is
1.45 times the horizontal; at 12 degrees it is 10.7 times.

**The reason is exercise 2.5, and it is a fact about the sky rather than about arithmetic.**
Satellites are only ever **above** the receiver, never below: the Earth is in the way. So every
unit vector $\mathbf{u}_i$ in the Jacobian has a positive vertical component, and they cannot
cancel. Horizontally the satellites surround the receiver, so the horizontal components point in
all directions and average out. **Vertical position is estimated from a one-sided set of
directions and horizontal position from a two-sided one**, and one-sided is always worse
conditioned.

**Narrowing the cone makes it worse in the obvious way**: it approaches the case where every
$\mathbf{u}_i$ points nearly straight up, which is the singular configuration of exercise 2.5.

**This is why GPS altitude is roughly 1.5 to 3 times worse than GPS horizontal position** in
normal conditions, and why receivers that can assume a known altitude (marine, or road-matched)
gain accuracy by dropping that unknown entirely. It is also the reason for the "dilution of
precision" figure receivers report, which is split into horizontal and vertical components for
exactly this reason.

### 5.1 Why the residual and not the objective

**The two models.** Gauss-Newton linearises the residual and then squares:

$$m_{\text{GN}}(\mathbf{p}) = \tfrac12\|\mathbf{r}+J\mathbf{p}\|^2
= f + \mathbf{p}^TJ^T\mathbf{r} + \tfrac12\mathbf{p}^TJ^TJ\mathbf{p}.$$

Newton takes the quadratic Taylor model of $f$ itself:

$$m_{\text{N}}(\mathbf{p}) = f + \mathbf{p}^T\nabla f
+ \tfrac12\mathbf{p}^T\nabla^2f\,\mathbf{p}
= f + \mathbf{p}^TJ^T\mathbf{r} + \tfrac12\mathbf{p}^T(J^TJ+S)\mathbf{p}.$$

**They agree in the constant and linear terms and differ in the quadratic one, by exactly
$\tfrac12\mathbf{p}^TS\mathbf{p}$.** So "linearise then square" is **not** "take the quadratic
model", and the discrepancy is precisely the term of section 3.

**Why linearising the residual is the better model near a zero residual solution.** Three
reasons, in increasing order of importance.

- **It is exact there.** If $\mathbf{r}(\mathbf{x}^*) = 0$ then $S = 0$ at the solution and the
  two models coincide, so nothing is lost.
- **It is exact for a linear model at any residual.** If $\mathbf{r}$ is affine then
  $\nabla^2r_i = 0$, so $S = 0$ identically and Gauss-Newton solves the problem in **one step**.
  That is lessons 29 to 32 recovered as a special case, and it is the sense in which
  Gauss-Newton is the right generalisation.
- **It respects the problem's structure.** $m_{\text{GN}}$ is the squared norm of an affine
  function, so it is a convex quadratic **whatever the data**, bounded below and with a unique
  minimiser. $m_{\text{N}}$ is a general quadratic and can be unbounded below. Measured in
  lesson 34 section 6: $J^TJ$ was positive semidefinite at 200 of 200 sampled points and
  $J^TJ+S$ was indefinite at 144 of them.

**So the answer to "why not just use the better model" is that $m_{\text{N}}$ is a more accurate
model of a function whose minimum you may not want.** Newton finds a stationary point of $f$;
Gauss-Newton finds a minimiser of a convex surrogate. The second is the safer target, and it
coincides with the first exactly when the fit is good.

### 5.2 Errors in variables, nonlinearly

**The statement.** Lesson 33 showed that when the design matrix is measured with error, ordinary
least squares converges to $\beta s^2/(s^2+e^2)$ rather than $\beta$. The nonlinear analogue: if
the **predictors** $t_i$ are measured with error, the estimate of $\mathbf{x}$ is biased, and the
bias does not vanish as $m\to\infty$.

**Where the bias comes from.** With $\tilde{t}_i = t_i+\varepsilon_i$, the residual becomes
$g(\tilde{t}_i;\mathbf{x}) - y_i$ and

$$g(t+\varepsilon;\mathbf{x}) \approx g(t;\mathbf{x}) + \varepsilon\,g'(t;\mathbf{x})
+ \tfrac12\varepsilon^2 g''(t;\mathbf{x}).$$

Taking expectations, the $\varepsilon$ term vanishes and the $\varepsilon^2$ term does not:

$$\mathbb{E}\,g(\tilde t;\mathbf{x}) \approx g(t;\mathbf{x})
+ \tfrac12\operatorname{var}(\varepsilon)\,g''(t;\mathbf{x}).$$

**So the fitted curve is systematically displaced by half the predictor variance times the
model's curvature.** That is the nonlinear attenuation formula, and it reduces to lesson 33's for
a linear model with... nothing, because $g'' = 0$ there. **The linear case's bias comes from the
denominator $\sum\tilde{x}_i^2$ instead**, so the two mechanisms are different and both are
real: curvature bias plus the variance inflation of the linear case.

**What total least squares becomes: orthogonal distance regression.** Instead of

$$\min_\mathbf{x}\sum_i\big(g(t_i;\mathbf{x})-y_i\big)^2,$$

minimise the **perpendicular** distance from each point to the curve:

$$\min_{\mathbf{x},\{\delta_i\}}\ \sum_i\Big[\delta_i^2
+ \big(g(t_i+\delta_i;\mathbf{x})-y_i\big)^2\Big],$$

with the $\delta_i$ extra unknowns. That is $m+n$ unknowns instead of $n$, but the extra ones
appear in a **separable** way and the problem retains a sparse structure that ODRPACK exploits.
It is the exact nonlinear analogue: TLS is orthogonal distance regression for a straight line.

**Measured on GPS**, per exercise 33.5.2's analysis: the satellite position errors perturb $J$
as much as $\mathbf{r}$, so the framing is right. But the perturbation is $O(10^{-7})$ relative
in both, so the bias is comparable to the noise rather than dominating it, and the practical
gain is a refinement rather than a rescue. **The place this matters much more is calibration
curves in chemistry and physics**, where the independent variable is a measured concentration or
a measured temperature with real uncertainty, and orthogonal distance regression is the standard
tool.

### 5.3 Constraints, and keeping the least squares structure

Three routes, and they behave differently.

**Change of variables.** Impose $c > 0$ by fitting $c = e^{u}$, or $\sum c_j = 1$ by a softmax.
The problem stays an unconstrained least squares problem, so every method in this lesson applies
unchanged.

- **What it costs: conditioning.** The chain rule multiplies the relevant Jacobian column by
  $\partial c/\partial u = e^u = c$. So a parameter that is genuinely near zero gets a column
  scaled towards zero, and $\kappa(J)$ grows without bound as the constraint becomes active. The
  reparametrisation has moved the boundary to infinity, and a solution *on* the boundary is now
  unreachable.
- **When to use it: when the constraint is not expected to be active.** A positivity constraint
  that is a sanity check rather than a real limit is ideal for this.

**Projection.** Take the unconstrained step and project back onto the feasible set.

- **What it costs: the convergence rate, and sometimes correctness.** Projection is not a descent
  direction in general, and the quadratic rate of section 2 is lost once the active set stops
  changing only if the projection is handled as an active set method. Done naively it can cycle.
- **When to use it: simple bounds**, where the projection is a clip and an active set strategy
  is easy to get right. This is what `scipy.optimize.least_squares(bounds=...)` does properly.

**Penalty.** Add $\mu\,\|\text{violation}\|^2$ to the residual vector as extra rows.

- **What it costs: conditioning again, and worse.** The stacked matrix is
  $\begin{pmatrix}J\\ \sqrt{\mu}\,C\end{pmatrix}$, whose condition number grows like
  $\sqrt{\mu}$, and the constraint is only satisfied in the limit $\mu\to\infty$. That is the
  classic penalty method dilemma: accuracy and conditioning pull in opposite directions.
- **When to use it: when the constraint is soft anyway**, that is, when it encodes a preference
  rather than a requirement. Then it is not a constraint at all, it is regularization, and lesson
  32 already covered how to choose $\mu$.

**The one that keeps the structure best is the change of variables**, because the problem remains
exactly the nonlinear least squares problem this lesson solves. **The one that is most correct is
the active set projection**, because it can return a solution on the boundary. **The augmented
Lagrangian** combines them: a penalty whose multiplier is updated so that a finite $\mu$
suffices, which avoids the conditioning blowup. Lesson 84 develops it properly.

---

## Where these solutions came from

Every number quoted was produced by running the code, on the versions of `nalib` in this
repository, with the seeds shown. Where a measurement contradicted the expected result, the
measurement is reported and the expectation is corrected in the text. The clearest examples in
this part:

- **Chebyshev sample spacing does not improve the conditioning of a polynomial fit** (exercise
  29.4.1). It is slightly worse than even spacing. The **basis** is what matters, by a factor of
  $2.5\times10^{6}$ at degree 20.
- **Plain seminormal equations are no better than the normal equations**, and worse at high
  $\kappa$ (exercise 29.5.2). One step of refinement changes that completely.
- **Column ordering does not rescue classical Gram-Schmidt** (exercise 30.4.1), and sorting by
  column norm is not the best order.
- **The measured Householder/Gram-Schmidt time ratio beats the flop count** (exercise 30.4.2),
  because the flop count cannot see the memory hierarchy.
- **The Givens/Householder crossover is at bandwidth 7, not $n/3$** (exercise 31.4.3), for the
  same reason.
- **The default `rcond` was 2600 times worse than keeping everything** on one problem (exercise
  32.4.2), and would have been far too generous on a slightly different one.
- **The obvious test problem for a general regularizer shows nothing** (exercise 32.3.2), because
  small and smooth pointed the same way. The demonstration needed a truth with a large mean.
